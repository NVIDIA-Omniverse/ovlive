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

//! b.v6d revision 4 — an `-ASK` a command cannot follow must not end the pooled
//! client's fred router task, and a router that ends any other way is
//! restarted (`storage::router_supervisor`).
//!
//! # The defect
//!
//! On stock fred 10.1.0 a command with no write attempt left that is answered
//! `-ASK` fails, and `process_ask` then returns the error out of the router
//! loop, ending the router task. Nothing marks the death: the client still
//! reads `Connected`, its command channel stays open, and every later command
//! routed to that pool member waits out its timeout until the process
//! restarts. Revision 3 made the commit `EVALSHA` and the start-record `HGET`
//! one-attempt commands; the stream readers' `XREAD`s already were.
//!
//! The fix has two halves, each tested on its own:
//! - **K**, a vendored one-line fred patch (`vendor/fred-10.1.0.SOURCE.md`):
//!   `process_ask` fails the command and returns `Ok(())`, so the router keeps
//!   running.
//! - **L**, a router supervisor: one watcher per pool member restarts a router
//!   that ended other than by `quit()`, counted on
//!   `live_edit_valkey_router_restarts_total{pool}` and logged at ERROR.
//!
//! # The clustered `-ASK` facade
//!
//! The store under test is a standalone Valkey. A [`ScriptReplyProxy`] in
//! front of it answers fred's two cluster discovery commands — `CLUSTER SLOTS`
//! maps every slot to the proxy itself, `CLUSTER INFO` reads `cluster_state:ok`
//! — so a client connected with a `redis-cluster://` URL runs fred's clustered
//! router against the real store. The test's trap then answers the FIRST frame
//! it matches with `-ASK 1234 <proxy>`, exactly the reply a migrating slot
//! gives once the keys a command names are gone from the source node. The R4-1
//! tests also have it answer `-MOVED 1234 <proxy>` instead, count the
//! `CLUSTER SLOTS` frames that arrive after the trap fired, and optionally
//! hold the first one back (`AskFacade::start_redirecting`). A fred
//! `Transaction`'s `EXEC` is trapped through the reply-rewriting proxy
//! instead (`AskFacade::start_redirecting_exec`), since a substitute command
//! sent inside `MULTI` would only be queued.
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`ask_on_the_commit_evalsha_leaves_every_data_pool_member_serving`] | T33 (a) | the commit's one-attempt `EVALSHA` is ASKed; `update_layer` still commits, and every data-pool member answers `PING` |
//! | [`ask_on_the_start_record_read_leaves_every_data_pool_member_serving`] | T33 (b) | the start-record `HGET` is ASKed: the call answers `UNAVAILABLE`, every member answers `PING`, a re-send applies |
//! | [`ask_on_a_reader_xread_leaves_every_reader_serving`] | T33 (c) | a reader's `XREAD` is ASKed: a later commit is still delivered, and every connected reader answers `PING` |
//! | [`ask_on_a_txn_slot_command_leaves_the_slot_serving`] | T33 control | the txn pool (`max_redirections = 0`) was never exposed: its slot keeps serving |
//! | [`ask_neither_ends_the_router_nor_fails_a_command_in_flight_beside_it`] | T34 (K) | after the `-ASK` no router restart is counted, and a command in flight on the same member is answered rather than failed `channel closed` |
//! | `ended_<pool>_router_is_restarted_and_counted` (six) | T35 (L) | an ended router of each pool class is restarted: the member answers `PING`, the counter rises by one, one ERROR line names `pool` and `member`, and the member's reconnect hook still counts a later `CLIENT KILL`. Stream reader: the reader is connected by a subscription first (readers connect lazily). Pub/sub: a later `PUBLISH` is received. Txn: the next checkout is clean, and ending the router a second time restarts it again — the restart's repair handed its new router over |
//! | [`ended_txn_router_restart_waits_for_a_permit_while_the_only_slot_is_held`] | T35 (L), txn | the restart parks on the permit queue (`valkey_pool_waiters{txn}` reads 1) and repairs nothing until the held slot is returned, then repairs once |
//! | [`ended_txn_router_restart_waits_for_its_own_slot_while_another_is_free`] | T35 (L), txn | holding a permit, the restart waits on its own slot's guard: no repair while that slot is held, none on the free slot instead |
//! | [`quit_stops_supervision_without_a_restart`] | T36 | after `quit()` no router is restarted and no watcher is left |
//! | [`ended_router_after_a_probe_driven_txn_recycle_restarts_the_new_router`] | T38 | a dirty-checkout recycle hands its new router to the slot's watcher, so ending THAT router restarts it |
//! | [`old_router_abort::txn_recycle_after_a_timed_out_quit_rebuilds_the_slot_on_a_new_connection`] | defer-9 | a `probe_timeout` recycle whose `QUIT` times out (a proxy holds the probe's `DISCARD`, so the old router is still running) leaves the next recycle able to rebuild the slot: its `CLIENT ID` changes and it answers `PING` |
//! | [`old_router_abort::txn_router_a_repair_aborted_is_neither_counted_nor_restarted`] | defer-9 | after a recycle that aborted its still-running old router and one whose `QUIT` succeeded, no router restart is counted or logged; ending the router afterwards counts and logs one |
//! | [`old_router_abort::back_to_back_txn_recycles_each_abort_the_router_the_previous_one_started`] | defer-9 | two `QUIT`-timeout recycles in a row: the second aborts the router the first started, so a third recycle still moves the slot to a new connection |
//! | `overwrite_commit_answered_<ask, moved>_commits_on_a_synced_second_try` (two) | R4-1 | an overwrite's first `create_layer_commit.lua` `EVALSHA` answered a real `-ASK` / `-MOVED` (the `txn_pool` follows neither) is tried once more after a routing sync (the sync counter reads 1, and on `-ASK` a `CLUSTER SLOTS` reached the facade), and commits; the new object takes an update |
//! | [`overwrite_commit_redirect::overwrite_commit_second_try_is_sent_after_a_routing_sync_that_times_out`] | R4-1 | the facade holds the sync's `CLUSTER SLOTS` past the command timeout: one sync WARN, carrying the layer-mapping key as `mapping_key` and no `key` field, and the second try is sent anyway and commits |
//! | [`overwrite_commit_redirect::finisher_syncs_on_its_first_run_and_again_after_a_redirect_class_failure`] | b.ua9 finisher | an overwrite that gave up (both runs failed, injected `IO`) leaves its finisher; the finisher's first run fails `IO`, its second with the injected `-ASK` shape, its third commits: the routing-sync counter reads 3 (the call's repeated run, the finisher's first and third) |
//! | [`overwrite_commit_redirect::finisher_resyncs_after_a_write_attempts_exhausted_failure`] | b.ua9 finisher | the same with the second run failing "Too many failed write attempts." (the shape a dropped connection or a `MOVED` gives a `txn_pool` command): the counter reads 3 — the extra sync after a dropped connection the finisher accepts |
//! | [`overwrite_commit_redirect::finisher_does_not_resync_after_an_io_failure`] | b.ua9 finisher | the same with a plain `IO` second run, which the redirect-class classifier does not match: the counter reads 2 (the call's repeated run and the finisher's first) |
//! | [`redirected_transaction_resync::redirected_session_checkpoint_step_b_resyncs_its_checkout_and_warns_on_a_slow_sync`] | b.cyu | session recovery whose Step B `EXEC` the facade answers `-ASK` (through the reply-rewriting proxy, `AskFacade::start_redirecting_exec`): `Transient` with "Max redirections exceeded", `exec_txn`'s re-sync counted once, and, its `CLUSTER SLOTS` held past the command timeout, one timed-out re-sync WARN naming the Step B transaction with the session-meta key as `key` |
//!
//! T37 (the storm guard's delay function) is a unit test beside
//! `router_restart_delay` in `src/storage/router_supervisor.rs`; T39 (the six
//! zero-seeded counter cells) is in `tests/integration/observability_metrics_test.rs`;
//! T40 is the B2 datapoint in `cluster_subscriber_test.rs`. A router ended
//! here is aborted, so its repair always reaches the connection-awaited arm;
//! the repair's hand-over on its other arms, which no fixture here can stage,
//! is pinned by source inspection beside `repair_held_txn_slot` in
//! `src/storage/client.rs`. So is the defer-9 rule that every repair arm ends
//! the slot's old router between `quit()` and `connect()`: only the
//! `QUIT`-timeout arm leaves an old router reliably running here, so the same
//! check also pins that the `StillRunning` arm logs its WARN. The three
//! `abort_current` cases no real router can stage — the router handed over
//! last is the one aborted, before the watcher has read it; a router that had
//! already ended is reported `AlreadyEnded` and restarted as a death; a router
//! still mid-poll past the abort budget is reported `StillRunning` — are unit
//! tests beside `RouterSupervisor` in `src/storage/router_supervisor.rs`.
//!
//! # Feature gate
//!
//! T38 and the defer-9 tests drive the txn checkout's `ProbeFaultInjector`,
//! the R4-1 and finisher tests read `LayerStore`'s create-commit routing-sync
//! counter (the finisher tests also its commit-fault queue and finisher
//! timing), and
//! the b.cyu Step B test reads `ValkeyClient::txn_routing_syncs_for_test`:
//! all are `test-fault-injection` seams, and those tests are compiled out
//! without that feature. Everything else needs only Valkey and `test-support`
//! (for `ValkeyClient::end_router_for_test`).
//!
//! # Wait budget
//!
//! Every wait is bounded by one of two named constants: [`CONNECT_BUDGET`]
//! (30 s) on the one client a test connects, and [`SETTLE`] (10 s) on
//! everything else — plus [`SLOW_SLOTS_DELAY`] on the one overwrite that is
//! held back by it by construction, and, in the defer-9 tests, the wait for a
//! held `DISCARD`'s release (`old_router_abort::DISCARD_HOLD` +
//! `ORPHAN_END_MARGIN`, 5 s from the hold). A `QUIT`-timeout recycle's
//! checkout is at most 8.15 s (650 ms probe, 2 s `QUIT`, 0.5 s abort, 5 s
//! reconnect), inside its `SETTLE`. A pool's `PING`s are sent together under one `SETTLE`;
//! the waits that follow one router end — the restart counted, the slot
//! checked out again, the member answering `PING` — share one `SETTLE`
//! deadline (a "phase"). Each test's worst case, every named wait at its full
//! budget:
//!
//! | Test | Named waits | Worst case |
//! |---|---|---|
//! | T33 (a) | connect, the commit, the `PING`s | 50 s |
//! | T33 (b) | connect, the refused call, the `PING`s, the re-send | 60 s |
//! | T33 (c) | connect, the subscription start, two `recv`s, the trap, the commit, the blocked-read wait, the `PING`s | 100 s |
//! | T33 control | connect, the `PING` | 40 s |
//! | T34 | connect, the trap, the in-flight `GET` (the update's own `SETTLE` runs alongside) | 50 s |
//! | T35 multiplexed | connect, the precondition `PING`, the restart phase, the reconnect; the stream reader's variant first connects its reader (the subscription start, the connected wait, the blocked-read wait) | 60 s; 90 s for the stream reader |
//! | T35 pub/sub | connect, the restart phase, the `PUBLISH` loop, the reconnect | 60 s |
//! | T35 txn | connect, the precondition phase, two restart phases, the reconnect | 70 s |
//! | T35 txn permit / guard | connect, the opening checkout(s), the parked phase, the repaired phase | 60 s |
//! | T36 | connect, `quit()` | 40 s |
//! | T38 | connect, two checkouts, the restart phase | 60 s |
//! | defer-9 rebuild | connect, two checkouts, the release wait, the last recycle's phase | 65 s |
//! | defer-9 not counted | connect, three checkouts, the restart phase | 70 s |
//! | defer-9 back to back | connect, four checkouts, the release wait, the last recycle's phase | 85 s |
//! | R4-1 `ASK` / `MOVED` | connect, the overwrite, the update | 50 s |
//! | R4-1 slow sync | connect, the overwrite (`SLOW_SLOTS_DELAY` + `SETTLE`, 17.5 s), the update | 57.5 s |
//! | finisher syncs (each of three) | connect, the overwrite, the poll for the finisher's landing, `wait_for_cleanup` | 60 s |
//! | b.cyu Step B slow re-sync | connect, the recovery (`SLOW_SLOTS_DELAY` + `SETTLE`, 17.5 s) | 47.5 s |
//!
//! On top of that, some steps are bounded only by fred's 5 s command timeout:
//! the setup commands of `layer()` (T33, T34, R4-1, the finisher tests), the
//! R4-1 mapping `GET`, and the finisher tests' counter `GET` and the one
//! mapping `GET` of their poll that may overrun its deadline; `SUBSCRIBE`,
//! `PUBLISH` (one may overrun the `PUBLISH` loop's deadline, plus its 200 ms
//! receive window) and
//! `UNSUBSCRIBE` in T35 pub/sub; `CLIENT ID` and `CLIENT KILL` in T35 (four
//! `CLIENT ID`s in the guard test); the uncontended checkout and the `GET` in
//! T33 control; the two `CLIENT ID`s in T38; the `CLIENT ID`s of the defer-9
//! tests (three each, four back to back). T35 pub/sub reaches about 85 s, and
//! defer-9 back to back, 85 s named plus four `CLIENT ID`s, about 105 s; T33
//! (c), the longest named total, leaves 20 s of nextest's 120 s hang-killer
//! for `layer()`'s setup commands. The closing `ValkeyClient::quit()` and
//! `TestValkey`'s drop-time cleanup carry their own bounds. The healthy path
//! takes well under a second, except in the defer-9 tests, which wait out a
//! held `DISCARD` by construction (about 3 s, 5 s and 8 s), and in the two
//! slow-sync tests, which wait out a held `CLUSTER SLOTS` (about 7.5 s for
//! the R4-1 overwrite; the b.cyu Step B recovery waits out the 5 s command
//! timeout on it, about 7 s for the whole test).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientLike, EventInterface, KeysInterface, PubsubInterface};
use tokio::time::Instant;

use live_edit::config::Config;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{
    operation_request_digest, LayerStore, Operation, OperationOutcome,
};
use live_edit::storage::lua::UPDATE_LAYER_FAST_SCRIPT;
use live_edit::storage::notifications::{consume_layer_stream, LayerNotification, StreamCursor};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::grpc::log_capture::{assert_field, Captured, ThreadCapture};
use crate::storage::mux_test_helpers::{await_reader_blocked_in_xread, connect_reader_zero};
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ConnProxyRule, ProxyAction, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and contract constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on connecting one client (the fixture's own connect timeout).
const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// Upper bound on every wait — a counter to tick, a delivery to arrive, a call
/// to settle, a member to answer `PING`. A deadline, not a speed claim:
/// unreachable on the healthy path, which takes milliseconds. A member whose
/// router has ended never answers `PING` at all, and fred's own 5 s command
/// timeout fails that `PING` well inside this.
const SETTLE: Duration = Duration::from_secs(10);

/// How long the facade holds back the `-ASK` reply of the trapped command in
/// [`ask_neither_ends_the_router_nor_fails_a_command_in_flight_beside_it`], so
/// the command beside it is on the wire before the `-ASK` lands.
const ASK_REPLY_DELAY: Duration = Duration::from_millis(1_000);

/// How long the facade holds back the command in flight beside the trapped
/// one: it must still be waiting for its reply when the `-ASK` lands.
const IN_FLIGHT_DELAY: Duration = Duration::from_millis(1_500);

/// The node id the facade's `CLUSTER SLOTS` reply names.
const FACADE_NODE_ID: &str = "b0v6d0000000000000000000000000000000ask0";

/// `live_edit_valkey_router_restarts_total`.
const ROUTER_RESTARTS: &str = "live_edit_valkey_router_restarts_total";

/// The ERROR line a restart logs. Restated, not imported: it is the operator
/// contract — the published metrics reference tells operators to report it.
const RESTART_MSG: &str = "Valkey client router task ended; restarting it";

/// Every `pool` label, restated (`VALKEY_POOLS`).
const POOLS: [&str; 6] = [
    "subscriber",
    "data",
    "txn",
    "subscriber_pool",
    "maintenance",
    "snapshot",
];

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

fn restarts(metrics: &Metrics, pool: &str) -> u64 {
    sample_value(&render(metrics), ROUTER_RESTARTS, &[("pool", pool)])
        .unwrap_or_else(|| panic!("{ROUTER_RESTARTS}{{pool={pool:?}}} is seeded in Metrics::new"))
        as u64
}

fn restarts_row(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    POOLS.iter().map(|p| (*p, restarts(metrics, p))).collect()
}

fn reconnects(metrics: &Metrics, pool: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_reconnects_total",
        &[("pool", pool)],
    )
    .unwrap_or_else(|| panic!("valkey_reconnects_total{{pool={pool:?}}} is seeded")) as u64
}

/// `live_edit_valkey_pool_waiters{pool}`: callers parked for a checkout permit.
/// Read as 0 until the gauge's first increment.
fn pool_waiters(metrics: &Metrics, pool: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_waiters",
        &[("pool", pool)],
    )
    .unwrap_or(0.0) as u64
}

/// [`SETTLE`] from now: the deadline of one wait, or of one phase of waits
/// that share it.
fn settle_deadline() -> Instant {
    Instant::now() + SETTLE
}

/// Poll until `metric(pool)` reads at least `want`, by `deadline`.
async fn await_at_least(
    metrics: &Metrics,
    read: fn(&Metrics, &str) -> u64,
    pool: &str,
    want: u64,
    deadline: Instant,
    what: &str,
) {
    loop {
        let got = read(metrics, pool);
        if got >= want {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: pool={pool:?} stayed at {got}, want at least {want}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// `PING` `member` and require a reply by `deadline`.
async fn assert_answers_ping<C: ClientLike>(member: &C, deadline: Instant, what: &str) {
    match tokio::time::timeout_at(deadline, member.ping::<fred::types::Value>(None)).await {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => panic!(
            "{what}: PING failed: {e:?}. A timeout here is the member's router task not \
             running, so every command routed to it waits out its timeout"
        ),
        Err(_) => panic!(
            "{what}: PING got no reply by its deadline — the member's router task is not \
             running, so every command routed to it waits out its timeout"
        ),
    }
}

/// [`assert_answers_ping`] for every member of a pool: one `PING` per member,
/// all in flight together under ONE [`SETTLE`] deadline, so the wait does not
/// grow with the pool size.
async fn assert_every_member_answers_ping(members: &[Client], pool: &str) {
    assert!(
        !members.is_empty(),
        "precondition: the {pool} pool has members"
    );
    let deadline = settle_deadline();
    futures::future::join_all(members.iter().enumerate().map(|(i, member)| async move {
        assert_answers_ping(member, deadline, &format!("{pool} pool member {i}")).await;
    }))
    .await;
}

/// The `txn` slot repairs a router restart has run (`reason` `router_ended`),
/// counted by the one line each repair logs once its reconnect is issued:
/// `txn slot recycled: …` (INFO) or `txn slot recycle: reconnect …` (ERROR).
fn router_ended_repairs(captured: &Captured) -> usize {
    captured
        .events
        .iter()
        .filter(|event| {
            event["fields"]["reason"] == "router_ended"
                && event["fields"]["message"].as_str().is_some_and(|message| {
                    message.starts_with("txn slot recycled:")
                        || message.starts_with("txn slot recycle: reconnect")
                })
        })
        .count()
}

/// A trap: answers the FIRST frame `matches` accepts, and only that one, with
/// `action` built from the proxy's port; counts how often it fired.
struct Trap {
    fired: Arc<AtomicUsize>,
}

impl Trap {
    fn fired(&self) -> usize {
        self.fired.load(Ordering::SeqCst)
    }
}

/// A [`ScriptReplyProxy`] in front of `TEST_VALKEY_URL` that plays a one-node
/// cluster whose every slot it owns, and fires `trap` once.
struct AskFacade {
    proxy: ScriptReplyProxy,
    port: u16,
    trap: Trap,
    /// `CLUSTER SLOTS` frames the facade received after its trap fired: a
    /// cluster routing sync that reached the wire. Read only by the
    /// `test-fault-injection` R4-1 tests.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    slots_after_trap: Arc<AtomicUsize>,
}

/// What the facade answers `CLUSTER SLOTS` with once its trap has fired.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotsAfterTrap {
    /// The same one-node map as before.
    Answer,
    /// The same map, but the FIRST `CLUSTER SLOTS` after the trap is answered
    /// only after [`SLOW_SLOTS_DELAY`], past the client's 5 s command timeout:
    /// the routing sync before an overwrite's second try, or `exec_txn`'s
    /// re-sync after a redirected transaction, times out. Used only by the
    /// `test-fault-injection` R4-1 and b.cyu Step B tests.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    Slow,
}

/// How long [`SlotsAfterTrap::Slow`] holds the sync's `CLUSTER SLOTS` reply.
/// With T the client's command timeout (`test_config`'s
/// `valkey_command_timeout_ms`, 5 s), it must sit strictly between three
/// limits:
///
/// - **above T**, with room to spare: the dispatch bounds the sync by T, and
///   that bound must fire — and its WARN be logged — before the sync's `Ok`
///   can arrive. A margin of 2.5 s means a dispatch task that is not polled
///   for a while after T still sees its timer first; just past T would let a
///   late poll find the sync already done, log no WARN, and flake.
/// - **below 2T**: the second try's `EVALSHA` is queued behind the sync at
///   about T, under its own T deadline, so it must be served before 2T.
/// - **below fred's 10 s `internal_command_timeout`**, which bounds the
///   backchannel `CLUSTER SLOTS` the sync sends. Past it,
///   `cluster_slots_backchannel` (`vendor/fred-10.1.0/src/router/clustered.rs:425-430`,
///   `:469-477`) falls back to `connect_any` and sends a fresh `CLUSTER SLOTS`
///   in the same sync attempt; the facade holds only the first one after the
///   trap, so it answers that one at once and the sync completes at about
///   10 s — too late for the queued `EVALSHA`'s 2T deadline.
///
/// (The last two limits serve the overwrite's second try; the b.cyu Step B
/// recovery sends nothing behind its re-sync, so only the first binds it.)
///
/// The overwrite, and the Step B recovery, carrying it are given [`SETTLE`]
/// on top of this delay.
#[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
const SLOW_SLOTS_DELAY: Duration = Duration::from_millis(7_500);

impl AskFacade {
    /// `matches` picks the frame to answer `-ASK`; `ask_delay` holds that reply
    /// back (zero: answer at once). `extra` may rewrite other frames.
    async fn start(
        matches: impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
        ask_delay: Duration,
        extra: impl Fn(&[Vec<u8>]) -> Option<ProxyAction> + Send + Sync + 'static,
    ) -> Self {
        Self::start_redirecting("ASK", SlotsAfterTrap::Answer, matches, ask_delay, extra).await
    }

    /// [`Self::start`], answering the trapped frame with `-<redirect> 1234
    /// <proxy>` (`ASK` or `MOVED`), and every `CLUSTER SLOTS` after it as
    /// `slots_after_trap` says.
    async fn start_redirecting(
        redirect: &'static str,
        slots_after_trap: SlotsAfterTrap,
        matches: impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
        ask_delay: Duration,
        extra: impl Fn(&[Vec<u8>]) -> Option<ProxyAction> + Send + Sync + 'static,
    ) -> Self {
        Self::start_inner(
            redirect,
            slots_after_trap,
            move |_conn, args| matches(args),
            ask_delay,
            extra,
            false,
        )
        .await
    }

    /// [`Self::start_redirecting`] for a fred `Transaction`'s `EXEC`: `matches`
    /// also sees the frame's connection (so it can pick the `EXEC` of the
    /// transaction that queued a given command), and the trapped `EXEC` is
    /// answered `-<redirect> 1234 <proxy>` through the reply-rewriting proxy
    /// ([`ProxyAction::DiscardReplying`]): the real connection leaves `MULTI`
    /// with nothing run, and fred's transaction router — which follows no
    /// redirect on the `txn_pool` — fails the transaction "Max redirections
    /// exceeded". (An `Eval` substitute would only be queued inside `MULTI`.)
    #[cfg(feature = "test-fault-injection")]
    async fn start_redirecting_exec(
        redirect: &'static str,
        slots_after_trap: SlotsAfterTrap,
        matches: impl Fn(usize, &[Vec<u8>]) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self::start_inner(
            redirect,
            slots_after_trap,
            matches,
            Duration::ZERO,
            |_| None,
            true,
        )
        .await
    }

    async fn start_inner(
        redirect: &'static str,
        slots_after_trap: SlotsAfterTrap,
        matches: impl Fn(usize, &[Vec<u8>]) -> bool + Send + Sync + 'static,
        ask_delay: Duration,
        extra: impl Fn(&[Vec<u8>]) -> Option<ProxyAction> + Send + Sync + 'static,
        exec_trap: bool,
    ) -> Self {
        let port_slot: Arc<OnceLock<u16>> = Arc::new(OnceLock::new());
        let armed = Arc::new(AtomicBool::new(true));
        let fired = Arc::new(AtomicUsize::new(0));
        let slots_seen = Arc::new(AtomicUsize::new(0));
        let rule: ConnProxyRule = {
            let (port_slot, fired, slots_seen) = (
                Arc::clone(&port_slot),
                Arc::clone(&fired),
                Arc::clone(&slots_seen),
            );
            Arc::new(move |conn: usize, args: &[Vec<u8>]| {
                let port = *port_slot
                    .get()
                    .expect("the proxy port is recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                if name == b"CLUSTER" {
                    let sub = args.get(1)?.to_ascii_uppercase();
                    let slots = format!(
                        "return {{{{0, 16383, {{'127.0.0.1', {port}, '{FACADE_NODE_ID}'}}}}}}"
                    );
                    if sub == b"SLOTS"
                        && fired.load(Ordering::SeqCst) > 0
                        && slots_seen.fetch_add(1, Ordering::SeqCst) == 0
                        && slots_after_trap == SlotsAfterTrap::Slow
                    {
                        return Some(ProxyAction::DelayedEval(SLOW_SLOTS_DELAY, slots));
                    }
                    return match sub.as_slice() {
                        b"SLOTS" => Some(ProxyAction::Eval(slots)),
                        b"INFO" => Some(ProxyAction::Eval(
                            "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                             cluster_known_nodes:1\\r\\ncluster_size:1\\r\\n'"
                                .to_owned(),
                        )),
                        _ => None,
                    };
                }
                if matches(conn, args) && armed.swap(false, Ordering::SeqCst) {
                    fired.fetch_add(1, Ordering::SeqCst);
                    let reply = format!("{redirect} 1234 127.0.0.1:{port}");
                    if exec_trap {
                        return Some(ProxyAction::DiscardReplying(reply));
                    }
                    let ask = error_reply_script(&reply);
                    return Some(if ask_delay.is_zero() {
                        ProxyAction::Eval(ask)
                    } else {
                        ProxyAction::DelayedEval(ask_delay, ask)
                    });
                }
                extra(args)
            })
        };
        let proxy = if exec_trap {
            ScriptReplyProxy::start_rewriting_replies(test_valkey_addr(), rule).await
        } else {
            ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule).await
        };
        let port = proxy.port();
        port_slot
            .set(port)
            .expect("the port is recorded exactly once");
        Self {
            proxy,
            port,
            trap: Trap { fired },
            slots_after_trap: slots_seen,
        }
    }

    /// A `redis-cluster://` URL naming the facade. It carries no database: a
    /// clustered fred client never sends `SELECT`. The proxy selects
    /// `TEST_VALKEY_URL`'s database on every upstream connection, so a client
    /// on this URL, the facade's own [`TestValkey`] included, uses that
    /// database.
    fn cluster_url(&self) -> String {
        format!("redis-cluster://127.0.0.1:{}", self.port)
    }

    /// A [`TestValkey`] connected through the facade in cluster mode, with a
    /// `pool_size` data / txn pool.
    async fn test_valkey(&self, pool_size: u32) -> TestValkey {
        tokio::time::timeout(
            CONNECT_BUDGET,
            TestValkey::setup_with_url(self.cluster_url(), pool_size),
        )
        .await
        .expect("a client must connect through the cluster facade")
    }
}

/// Does `args` name `cmd` with an argument containing `needle`?
fn names(args: &[Vec<u8>], cmd: &[u8], needle: &str) -> bool {
    args.first().is_some_and(|n| n.eq_ignore_ascii_case(cmd))
        && args[1..]
            .iter()
            .any(|a| String::from_utf8_lossy(a).contains(needle))
}

/// A fresh session and layer on `tv`, at `seq = 1`.
async fn layer(tv: &TestValkey, name: &str) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "bv6d-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            name,
            "bv6d-user",
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    (session_id, object_id)
}

async fn commit(
    store: &LayerStore,
    session_id: &str,
    layer: &str,
    object_id: u64,
    value: u8,
    operation: Option<Operation>,
) -> Result<live_edit::storage::layer::UpdateCommit, ServiceError> {
    tokio::time::timeout(
        SETTLE,
        store.update_layer_with_operation_relayed(
            session_id,
            layer,
            object_id,
            Some(delta_with_set_field_inline(1, "color", vec![value], 0)),
            "bv6d-router-writer",
            1_000,
            operation,
        ),
    )
    .await
    .expect("the update did not settle within its budget")
}

// ═════════════════════════════════════════════════════════════════════════════
// T33 — an -ASK to a one-attempt command leaves its pool member serving
// ═════════════════════════════════════════════════════════════════════════════

/// T33 (a): the commit `EVALSHA` is sent once, so the first `-ASK` it meets
/// has no attempt left to follow. The attempt fails transiently and
/// `update_layer`'s retry loop commits on its next attempt — and the member
/// that took the `-ASK` must still serve: every data-pool member answers
/// `PING`.
///
/// On stock fred with no supervisor that member's router has ended, so its
/// `PING` is never answered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_on_the_commit_evalsha_leaves_every_data_pool_member_serving() {
    let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_owned();
    let facade = AskFacade::start(
        move |args| names(args, b"EVALSHA", &sha),
        Duration::ZERO,
        |_| None,
    )
    .await;
    let tv = facade.test_valkey(4).await;
    let (session_id, object_id) = layer(&tv, "bv6d-t33-commit").await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

    let committed = commit(&store, &session_id, "bv6d-t33-commit", object_id, 1, None)
        .await
        .expect("the update must commit through its retry loop after the -ASK");
    assert_eq!(committed.seq, 2);
    assert_eq!(
        facade.trap.fired(),
        1,
        "precondition: the commit EVALSHA was answered -ASK"
    );

    assert_every_member_answers_ping(&tv.client().pool.clients(), "data").await;
    drop(facade.proxy);
}

/// T33 (b): the start-record `HGET` of an operation-id call is sent once.
/// Answered `-ASK`, the call ends `UNAVAILABLE` before any attempt (nothing
/// applied, a re-send is safe), every data-pool member still answers `PING`,
/// and the client's re-send applies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_on_the_start_record_read_leaves_every_data_pool_member_serving() {
    let facade = AskFacade::start(
        |args| names(args, b"HGET", ":obj:wtok"),
        Duration::ZERO,
        |_| None,
    )
    .await;
    let tv = facade.test_valkey(4).await;
    let (session_id, object_id) = layer(&tv, "bv6d-t33-start").await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let operation = || {
        Operation::new(
            "bv6d-t33-start-op",
            operation_request_digest("bv6d-t33-start", b"t33"),
        )
    };

    let refused = commit(
        &store,
        &session_id,
        "bv6d-t33-start",
        object_id,
        2,
        Some(operation()),
    )
    .await;
    assert!(
        matches!(refused, Err(ServiceError::Unavailable(_))),
        "an -ASKed start-record read must end the call UNAVAILABLE before any attempt; got \
         {refused:?}"
    );
    assert_eq!(
        facade.trap.fired(),
        1,
        "precondition: the start-record HGET was answered -ASK"
    );

    assert_every_member_answers_ping(&tv.client().pool.clients(), "data").await;

    let resent = commit(
        &store,
        &session_id,
        "bv6d-t33-start",
        object_id,
        2,
        Some(operation()),
    )
    .await
    .expect("the re-send must apply");
    assert_eq!(
        (resent.seq, resent.outcome),
        (2, OperationOutcome::Applied),
        "the refused call applied nothing, so its re-send is the first commit"
    );
    drop(facade.proxy);
}

/// T33 (c): a stream reader's `XREAD` is sent once (`CONSUMER_XREAD_MAX_ATTEMPTS
/// = 1`). Answered `-ASK`, the subscription must keep delivering — a commit
/// made after it arrives within [`SETTLE`] — and every reader answers `PING`.
///
/// This is the exposure that predates b.v6d: on stock fred the reader's router
/// ended, its next `XREAD` timed out, and its own heal (`force_reconnection`)
/// never answered, so the subscription stopped delivering until the replica
/// restarted. Since b.azh a reader is a single-node client even on a cluster
/// (here, connected to the facade, its slot's one owner), which follows no
/// redirect at all: the `-ASK` reaches the reader as a `Cluster`-kind error
/// and is retried as transient. The delivery and `PING` checks still pin that
/// it keeps serving.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_on_a_reader_xread_leaves_every_reader_serving() {
    let facade_stream = Arc::new(OnceLock::<String>::new());
    let facade = {
        let stream = Arc::clone(&facade_stream);
        AskFacade::start(
            move |args| {
                stream
                    .get()
                    .is_some_and(|key| names(args, b"XREAD", key.as_str()))
            },
            Duration::ZERO,
            |_| None,
        )
        .await
    };
    let tv = facade.test_valkey(4).await;
    let (session_id, object_id) = layer(&tv, "bv6d-t33-reader").await;
    facade_stream
        .set(tv.key_schema().layer_stream(object_id))
        .expect("the stream key is recorded once");

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + SETTLE,
    )
    .await
    .expect("the subscription must start");
    // Drain CreateLayer's own entry, so the delivery asserted below is the
    // commit made after the -ASK.
    let first = tokio::time::timeout(SETTLE, consumer.recv())
        .await
        .expect("CreateLayer's entry must be delivered")
        .expect("the subscription must not end");
    let deadline = tokio::time::Instant::now() + SETTLE;
    while facade.trap.fired() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no reader XREAD naming the layer stream reached the facade (first delivery \
             {first:?})"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let committed = commit(&store, &session_id, "bv6d-t33-reader", object_id, 3, None)
        .await
        .expect("the commit after the -ASK must land");
    let delivered = tokio::time::timeout(SETTLE, consumer.recv())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "the commit at seq {} was not delivered within {SETTLE:?} after the reader's \
                 XREAD was answered -ASK: the reader stopped serving",
                committed.seq
            )
        })
        .expect("the subscription must not end");
    assert!(
        matches!(delivered.1, LayerNotification::Update { sequence_num, .. } if sequence_num == committed.seq),
        "the delivery must be the commit made after the -ASK; got {delivered:?}"
    );

    // Each reader is parked in its next XREAD BLOCK, and a `PING` on its
    // client interrupts that read (`Blocking::Interrupt`) — but only if no
    // read is in flight that fred has not yet marked blocked, and only if the
    // reader does not re-issue its read ahead of the `PING`. So: wait until
    // the server shows every reader blocked in its XREAD (fred marks a read
    // blocked just after writing it), then drop the subscription before the
    // `PING`s. A subscription leaving does not wake the read, and the reader
    // applies the removal before it plans another, so once a `PING`
    // interrupts the read the reader has nothing to re-issue and the `PING`
    // is answered.
    let readers = tv.client().stream_mux.reader_clients_for_test();
    for reader in &readers {
        await_reader_blocked_in_xread(tv.client().pool.next(), reader).await;
    }
    drop(consumer);
    assert_every_member_answers_ping(&readers, "subscriber_pool").await;
    drop(facade.proxy);
}

/// T33 control: the txn pool runs `max_redirections = 0`, so an `-ASK` fails
/// its command inside `process_ask`, at the redirection budget
/// (`decr_check_redirections`, `vendor/fred-10.1.0/src/router/commands.rs:261`),
/// before any `ASKING` is sent — and that early return was `Ok(())` even on
/// stock fred, so the slot's router was never exposed. Its slot keeps serving
/// — true before the fix as well, which is what makes it the control.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_on_a_txn_slot_command_leaves_the_slot_serving() {
    let facade = AskFacade::start(
        |args| names(args, b"GET", "bv6d-t33-txn-trap"),
        Duration::ZERO,
        |_| None,
    )
    .await;
    let tv = facade.test_valkey(1).await;
    let key = format!("{}bv6d-t33-txn-trap", tv.prefix());

    let slot = tv.client().acquire_txn().await;
    let asked = slot.get::<Option<String>, _>(&key).await;
    assert!(
        asked.is_err(),
        "precondition: the ASKed txn command fails; got {asked:?}"
    );
    assert_eq!(facade.trap.fired(), 1, "precondition: the -ASK fired");
    assert_answers_ping(&*slot, settle_deadline(), "the txn slot").await;
    drop(slot);
    drop(facade.proxy);
}

// ═════════════════════════════════════════════════════════════════════════════
// T34 — K: the -ASK no longer ends the router at all
// ═════════════════════════════════════════════════════════════════════════════

/// T34: the vendored fred patch, on its own. With a one-member data pool, the
/// commit `EVALSHA` is answered `-ASK` while another command on the same
/// member is in flight (the facade holds back the `-ASK` for
/// [`ASK_REPLY_DELAY`] and that command's reply for [`IN_FLIGHT_DELAY`]).
///
/// - No router restart is counted: the router never ended. The supervisor
///   alone would restart it and count one.
/// - The command in flight beside the `-ASK` is answered. A router that ends
///   fails every command it was waiting on with fred's non-transient
///   `channel closed`, which callers report as `INTERNAL` — the supervisor
///   cannot undo that.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_neither_ends_the_router_nor_fails_a_command_in_flight_beside_it() {
    const IN_FLIGHT_KEY: &str = "bv6d-t34-in-flight";
    let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_owned();
    let facade = AskFacade::start(
        move |args| names(args, b"EVALSHA", &sha),
        ASK_REPLY_DELAY,
        |args| names(args, b"GET", IN_FLIGHT_KEY).then_some(ProxyAction::Delay(IN_FLIGHT_DELAY)),
    )
    .await;
    let tv = facade.test_valkey(1).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let (session_id, object_id) = layer(&tv, "bv6d-t34").await;
    let store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));

    let update = {
        let (store, session_id) = (Arc::clone(&store), session_id.clone());
        tokio::spawn(
            async move { commit(&store, &session_id, "bv6d-t34", object_id, 4, None).await },
        )
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while facade.trap.fired() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the commit EVALSHA never reached the facade"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // The EVALSHA is on the wire and its -ASK held back: this command goes out
    // behind it on the same (only) member, and is still waiting for its reply
    // when the -ASK lands.
    let key = format!("{}{IN_FLIGHT_KEY}", tv.prefix());
    let in_flight = tokio::time::timeout(
        SETTLE,
        tv.client().pool.next().get::<Option<String>, _>(&key),
    )
    .await
    .expect("the in-flight command must settle");
    assert!(
        matches!(in_flight, Ok(None)),
        "the command in flight beside the -ASK must be answered (nil). `channel closed` here is \
         the router ending under it; got {in_flight:?}"
    );

    let committed = update
        .await
        .expect("the update task must not panic")
        .expect("the update must commit through its retry loop");
    assert_eq!(committed.seq, 2);
    assert_eq!(
        restarts_row(&metrics),
        POOLS.iter().map(|p| (*p, 0)).collect::<Vec<_>>(),
        "an -ASK must not end any router: no restart may be counted"
    );
    drop(facade.proxy);
}

// ═════════════════════════════════════════════════════════════════════════════
// T35 — L: an ended router of every pool class is restarted
// ═════════════════════════════════════════════════════════════════════════════

/// A client for the T35–T38 scenarios: no proxy, `pool_size` data / txn pool.
async fn plain_client(pool_size: u32) -> ValkeyClient {
    let mut config: Config = crate::common::test_config();
    config.valkey_pool_size = pool_size;
    tokio::time::timeout(CONNECT_BUDGET, connect_test_client(&config))
        .await
        .expect("connect must settle")
        .expect("connect must succeed")
}

/// Member `member` of `pool`'s fred client, for the pools whose members are
/// plain multiplexed clients.
fn member_of(client: &ValkeyClient, pool: &str, member: usize) -> Client {
    match pool {
        "data" => client.pool.clients()[member].clone(),
        "subscriber_pool" => client
            .stream_mux
            .reader_client_for_test(member)
            .expect("the stream reader holds a connection: connect it first"),
        "maintenance" => client.maintenance_pool.clients()[member].clone(),
        "snapshot" => client.snapshot_pool.clients()[member].clone(),
        other => panic!("{other} has no multiplexed member"),
    }
}

/// `CLIENT ID` of `conn`.
async fn connection_id(conn: &impl ClientLike) -> i64 {
    conn.custom::<i64, String>(
        fred::types::CustomCommand::new_static("CLIENT ID", None::<u16>, false),
        vec![],
    )
    .await
    .expect("CLIENT ID must succeed")
}

/// `CLIENT KILL ID <victim>` from `killer`, then wait for `pool`'s reconnect
/// counter to rise: the member's reconnect hook still fires after its router
/// was restarted.
async fn kill_and_await_reconnect(metrics: &Metrics, killer: &Client, victim: i64, pool: &str) {
    let before = reconnects(metrics, pool);
    let killed: i64 = killer
        .custom(
            fred::types::CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_owned(), victim.to_string()],
        )
        .await
        .expect("CLIENT KILL must succeed");
    assert_eq!(
        killed, 1,
        "CLIENT KILL must close exactly the member's connection"
    );
    await_at_least(
        metrics,
        reconnects,
        pool,
        before + 1,
        settle_deadline(),
        "a CLIENT KILL after the restart must still be counted: the restarted router kept the \
         member's reconnect hook",
    )
    .await;
}

/// The restart a T35 variant must see, by `deadline`, after its member 0's
/// router was ended: counted once, logged once at ERROR naming `pool` and
/// member 0.
async fn assert_one_restart(
    capture: &ThreadCapture,
    metrics: &Metrics,
    pool: &str,
    deadline: Instant,
) {
    await_at_least(
        metrics,
        restarts,
        pool,
        1,
        deadline,
        "the ended router's restart must be counted",
    )
    .await;
    let captured = capture.snapshot();
    let line = captured.only(RESTART_MSG);
    assert_eq!(line["level"], "ERROR", "the restart logs at ERROR: {line}");
    assert_field(line, "pool", pool.into());
    assert_field(line, "member", 0.into());
    assert_eq!(
        restarts_row(metrics),
        POOLS
            .iter()
            .map(|p| (*p, u64::from(*p == pool)))
            .collect::<Vec<_>>(),
        "exactly one restart, on {pool}"
    );
}

/// T35 for a multiplexed pool: end member 0's router, then — within one
/// [`SETTLE`] — the restart is counted and logged once and the member answers
/// `PING`; and a later `CLIENT KILL` of its connection is still counted as a
/// reconnect. A stream reader is connected first ([`connect_reader_zero`]): it
/// has no router until its first connect, and its router is supervised from
/// then on.
///
/// Runs on a current-thread runtime so the [`ThreadCapture`] sees the line
/// the supervisor's spawned watcher logs.
async fn multiplexed_router_restart(pool: &'static str) {
    let capture = ThreadCapture::start();
    let client = plain_client(4).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    if pool == "subscriber_pool" {
        assert!(
            !client.end_router_for_test(pool, 0),
            "precondition: an unconnected reader has no router to end"
        );
        connect_reader_zero(&client).await;
    }
    let member = member_of(&client, pool, 0);
    assert_answers_ping(
        &member,
        settle_deadline(),
        "precondition: the member serves",
    )
    .await;

    assert!(
        client.end_router_for_test(pool, 0),
        "precondition: {pool} member 0 has a supervised router"
    );
    let restarted = settle_deadline();
    assert_one_restart(&capture, &metrics, pool, restarted).await;
    assert_answers_ping(&member, restarted, &format!("{pool} member 0")).await;

    let killer = if pool == "maintenance" {
        client.pool.next().clone()
    } else {
        client.maintenance_pool.next().clone()
    };
    let victim = connection_id(&member).await;
    kill_and_await_reconnect(&metrics, &killer, victim, pool).await;
    client.quit().await;
}

#[tokio::test]
async fn ended_data_router_is_restarted_and_counted() {
    multiplexed_router_restart("data").await;
}

#[tokio::test]
async fn ended_subscriber_pool_router_is_restarted_and_counted() {
    multiplexed_router_restart("subscriber_pool").await;
}

#[tokio::test]
async fn ended_maintenance_router_is_restarted_and_counted() {
    multiplexed_router_restart("maintenance").await;
}

#[tokio::test]
async fn ended_snapshot_router_is_restarted_and_counted() {
    multiplexed_router_restart("snapshot").await;
}

/// T35, pub/sub: the `subscriber` client's ended router is restarted, and the
/// channel subscription the service depends on survives it — a
/// `le:session:deleted` message published after the restart is received
/// (`manage_subscriptions` re-subscribes on the restarted connection).
#[tokio::test]
async fn ended_subscriber_router_is_restarted_and_counted() {
    const POOL: &str = "subscriber";
    let capture = ThreadCapture::start();
    let client = plain_client(4).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let schema =
        live_edit::storage::keys::KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()));
    let channel = schema.session_deleted_channel();
    live_edit::grpc::session::subscribe_session_deleted_channel(&client.subscriber, &schema)
        .await
        .expect("SUBSCRIBE must succeed");
    let mut messages = client.subscriber.message_rx();

    assert!(
        client.end_router_for_test(POOL, 0),
        "precondition: the pub/sub subscriber has a supervised router"
    );
    let restarted = settle_deadline();
    assert_one_restart(&capture, &metrics, POOL, restarted).await;
    assert_answers_ping(&client.subscriber, restarted, "the subscriber").await;

    // Re-published until it lands: the re-subscription rides the restarted
    // connection's reconnect event, so a message published a moment before
    // it is not delivered — that is pub/sub, not the defect.
    let publisher = client.pool.next().clone();
    let deadline = tokio::time::Instant::now() + SETTLE;
    'received: loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no `{channel}` message published after the restart was received: the restarted \
             subscriber is not subscribed"
        );
        let _: i64 = publisher
            .publish(&channel, "bv6d-t35-session")
            .await
            .expect("PUBLISH must succeed");
        let window = tokio::time::Instant::now() + Duration::from_millis(200);
        while let Ok(Ok(message)) = tokio::time::timeout_at(window, messages.recv()).await {
            if &*message.channel == channel.as_str() {
                break 'received;
            }
        }
    }

    // A subscribed connection refuses `CLIENT ID`; leave subscribe mode first.
    client
        .subscriber
        .unsubscribe(&channel)
        .await
        .expect("UNSUBSCRIBE must succeed");
    let victim = connection_id(&client.subscriber).await;
    kill_and_await_reconnect(&metrics, client.maintenance_pool.next(), victim, POOL).await;
    client.quit().await;
}

/// Check out a `txn` slot by `deadline`.
async fn checkout_by(
    client: &ValkeyClient,
    deadline: Instant,
    what: &str,
) -> live_edit::storage::checkout::CheckedOutClient {
    tokio::time::timeout_at(deadline, client.acquire_txn())
        .await
        .unwrap_or_else(|_| panic!("{what}: no txn slot was checked out by the deadline"))
}

/// T35, txn: a slot's ended router is restarted through the checkout (a permit,
/// then that slot's guard) and the repair a dirty checkout gets. The slot then
/// serves, its next checkout is clean (no dirty verdict is counted), and its
/// re-armed reconnect hook still counts a `CLIENT KILL`.
///
/// Then the slot's router is ended a SECOND time, and must be restarted again
/// (two restarts in all). That pins the restart's own hand-over: the repair
/// it runs hands the router it started to the slot's watcher, and the watcher
/// watches it next. Without the hand-over the watcher would still hold the
/// first, already-ended router's handle — the second end would abort nothing
/// it watches, and the slot's real router would go unsupervised.
///
/// An abort-ended router reaches the repair's connection-awaited arm: the
/// repair's `quit()` finds no stored command channel, marks the client
/// `Disconnecting` and fails to send, so the repair waits for the new
/// connection. The other arms — a router that ended by returning an error
/// after it had connected (the client still reads `Connected`), a failed or
/// timed-out reconnect — are not staged here; a source check beside
/// `repair_held_txn_slot` in
/// `src/storage/client.rs` pins the hand-over on every arm.
#[tokio::test]
async fn ended_txn_router_is_restarted_and_counted() {
    const POOL: &str = "txn";
    let capture = ThreadCapture::start();
    let client = plain_client(1).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    {
        let ready = settle_deadline();
        let slot = checkout_by(&client, ready, "precondition").await;
        assert_answers_ping(&*slot, ready, "precondition: the slot serves").await;
    }

    assert!(
        client.end_router_for_test(POOL, 0),
        "precondition: txn slot 0 has a supervised router"
    );
    let restarted = settle_deadline();
    assert_one_restart(&capture, &metrics, POOL, restarted).await;
    // The restart's repair holds the only slot until it has handed the new
    // router over; this checkout waits for it.
    let slot = checkout_by(&client, restarted, "the restarted slot").await;
    assert_answers_ping(&*slot, restarted, "the restarted txn slot").await;
    let dirty: f64 = render(&metrics)
        .lines()
        .filter(|l| l.starts_with("live_edit_valkey_txn_dirty_checkouts_total{"))
        .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
        .sum();
    assert_eq!(
        dirty, 0.0,
        "the checkout after the restart must find the slot clean: no dirty verdict may be counted"
    );
    let victim = connection_id(&*slot).await;
    drop(slot);
    kill_and_await_reconnect(&metrics, client.pool.next(), victim, POOL).await;

    assert!(
        client.end_router_for_test(POOL, 0),
        "precondition: txn slot 0 still has a supervised router"
    );
    let restarted_again = settle_deadline();
    await_at_least(
        &metrics,
        restarts,
        POOL,
        2,
        restarted_again,
        "the router the first restart started must be the one the supervisor watches: ending it \
         must restart it",
    )
    .await;
    let slot = checkout_by(&client, restarted_again, "the twice-restarted slot").await;
    assert_answers_ping(&*slot, restarted_again, "the twice-restarted txn slot").await;
    drop(slot);
    assert_eq!(restarts(&metrics, POOL), 2, "exactly two restarts");
    client.quit().await;
}

/// T35, txn, the permit: the restart of a slot's ended router waits for a
/// checkout permit like any borrower. With a one-slot pool whose only slot is
/// held, the restart is counted and then parks on the permit queue
/// (`live_edit_valkey_pool_waiters{pool="txn"}` reads 1) without running its
/// repair; once the slot is returned the repair runs, once, and the slot
/// serves.
#[tokio::test]
async fn ended_txn_router_restart_waits_for_a_permit_while_the_only_slot_is_held() {
    const POOL: &str = "txn";
    let capture = ThreadCapture::start();
    let client = plain_client(1).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let held = checkout_by(&client, settle_deadline(), "the held slot").await;

    assert!(
        client.end_router_for_test(POOL, 0),
        "precondition: txn slot 0 has a supervised router"
    );
    let parked = settle_deadline();
    await_at_least(
        &metrics,
        restarts,
        POOL,
        1,
        parked,
        "precondition: the ended router's restart was started",
    )
    .await;
    await_at_least(
        &metrics,
        pool_waiters,
        POOL,
        1,
        parked,
        "the restart must wait for a permit while the only slot is held",
    )
    .await;
    assert_eq!(
        router_ended_repairs(&capture.snapshot()),
        0,
        "the restart must not repair the slot while a borrower holds it"
    );

    drop(held);
    let repaired = settle_deadline();
    // The restart is first in the permit queue, so this checkout gets the slot
    // only after the repair has returned it.
    let slot = checkout_by(&client, repaired, "the repaired slot").await;
    assert_eq!(
        router_ended_repairs(&capture.snapshot()),
        1,
        "once the slot is returned, the restart repairs it exactly once"
    );
    assert_answers_ping(&*slot, repaired, "the repaired txn slot").await;
    drop(slot);
    client.quit().await;
}

/// Check out both slots of a two-slot `txn` pool by `deadline`, in slot
/// order: with every slot free, a checkout takes the first free one.
async fn both_slots(
    client: &ValkeyClient,
    deadline: Instant,
) -> (
    live_edit::storage::checkout::CheckedOutClient,
    live_edit::storage::checkout::CheckedOutClient,
) {
    let slot0 = checkout_by(client, deadline, "slot 0").await;
    let slot1 = checkout_by(client, deadline, "slot 1").await;
    (slot0, slot1)
}

/// T35, txn, the guard: the restart of a slot's ended router waits for THAT
/// slot's guard. With a two-slot pool, slot 0 — member 0, the slot an idle
/// pool's first checkout takes — is held and slot 1 is free. The restart takes
/// the free permit at once (no permit waiter is counted, and no permit is
/// left) and then waits on slot 0's guard: it repairs nothing while slot 0 is
/// held, and it never touches slot 1 — slot 1 keeps its connection, while
/// slot 0's is rebuilt once slot 0 is returned. Checked by connection id
/// rather than by timing: a restart that took the free slot instead would
/// rebuild slot 1's connection whenever it ran.
#[tokio::test]
async fn ended_txn_router_restart_waits_for_its_own_slot_while_another_is_free() {
    const POOL: &str = "txn";
    let capture = ThreadCapture::start();
    let client = plain_client(2).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let (held, slot1) = both_slots(&client, settle_deadline()).await;
    let (slot0_before, slot1_before) = (connection_id(&*held).await, connection_id(&*slot1).await);
    drop(slot1);

    assert!(
        client.end_router_for_test(POOL, 0),
        "precondition: txn slot 0 has a supervised router"
    );
    let parked = settle_deadline();
    await_at_least(
        &metrics,
        restarts,
        POOL,
        1,
        parked,
        "precondition: the ended router's restart was started",
    )
    .await;
    const NOT_WHILE_HELD: &str = "the restart must wait on slot 0's own guard while slot 0 is \
                                  held. A repair here ran on the free slot 1, or on a slot 0 \
                                  nothing guarded";
    while client.txn_pool.available_permits() > 0 {
        assert_eq!(
            router_ended_repairs(&capture.snapshot()),
            0,
            "{NOT_WHILE_HELD}"
        );
        assert!(
            Instant::now() < parked,
            "the restart never took the free permit"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        pool_waiters(&metrics, POOL),
        0,
        "precondition: the restart holds a permit and waits on no permit"
    );
    assert_eq!(
        router_ended_repairs(&capture.snapshot()),
        0,
        "{NOT_WHILE_HELD}"
    );

    drop(held);
    let repaired = settle_deadline();
    while router_ended_repairs(&capture.snapshot()) == 0 {
        assert!(
            Instant::now() < repaired,
            "the restart never repaired slot 0 after it was returned"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (slot0, slot1) = both_slots(&client, repaired).await;
    assert_answers_ping(&*slot0, repaired, "the repaired txn slot").await;
    assert_eq!(
        router_ended_repairs(&capture.snapshot()),
        1,
        "the restart repairs once"
    );
    let (slot0_after, slot1_after) = (connection_id(&*slot0).await, connection_id(&*slot1).await);
    assert_eq!(
        (slot0_after != slot0_before, slot1_after == slot1_before),
        (true, true),
        "the restart must rebuild slot 0's connection and leave the free slot 1's alone (slot 0: \
         {slot0_before} -> {slot0_after}; slot 1: {slot1_before} -> {slot1_after})"
    );
    drop((slot0, slot1));
    client.quit().await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T36 — quit() stops supervision first
// ═════════════════════════════════════════════════════════════════════════════

/// T36: `quit()` shuts the supervisor down before it closes any pool, so the
/// routers its `QUIT`s end are not restarted: no restart is counted and no
/// watcher is left running. Without the shutdown, every txn slot's watcher
/// would sit waiting for a recycle's hand-over that never comes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_stops_supervision_without_a_restart() {
    let client = plain_client(2).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    assert!(
        client.router_watchers_for_test() > 0,
        "precondition: connect started supervision"
    );

    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");

    assert_eq!(
        client.router_watchers_for_test(),
        0,
        "quit must leave no router watcher running"
    );
    assert_eq!(
        restarts_row(&metrics),
        POOLS.iter().map(|p| (*p, 0)).collect::<Vec<_>>(),
        "no router ended by quit may be restarted"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T38 — a recycle hands its new router to the slot's watcher
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod recycle_handover {
    use live_edit::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};

    use super::*;

    /// T38: a dirty checkout (a leaked `MULTI`, from the probe seam) recycles
    /// the slot: `quit()` ends its router cleanly and `connect()` starts a new
    /// one, which the repair hands to the slot's watcher. Ending the router
    /// after that must restart the NEW router — counted once, the slot serving
    /// again. Had the hand-over been lost, the watcher would still hold the
    /// quit router's handle: the end would abort nothing it watches, no
    /// restart would be counted, and a later death of the slot's real router
    /// would go unnoticed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ended_router_after_a_probe_driven_txn_recycle_restarts_the_new_router() {
        let client = plain_client(1).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let injector = Arc::new(ProbeFaultInjector::new());
        client.install_probe_fault_injector_for_test(Arc::clone(&injector));

        let before = {
            let slot = checkout_by(&client, settle_deadline(), "the first checkout").await;
            connection_id(&*slot).await
        };
        injector.push_discard(ProbeFault::ok());
        let recycled = {
            let slot = checkout_by(&client, settle_deadline(), "the dirty checkout").await;
            connection_id(&*slot).await
        };
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the leaked-MULTI probe fault was consumed"
        );
        assert_ne!(
            recycled, before,
            "precondition: the dirty checkout recycled the slot onto a new connection"
        );

        assert!(
            client.end_router_for_test("txn", 0),
            "precondition: txn slot 0 has a supervised router"
        );
        let restarted = settle_deadline();
        await_at_least(
            &metrics,
            restarts,
            "txn",
            1,
            restarted,
            "the recycled slot's router must be the one the supervisor watches",
        )
        .await;
        let slot = checkout_by(&client, restarted, "the restarted slot").await;
        assert_answers_ping(&*slot, restarted, "the restarted txn slot").await;
        drop(slot);
        assert_eq!(restarts(&metrics, "txn"), 1, "exactly one restart");
        client.quit().await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Defer-9 — a repair ends the slot's old router before it starts the new one
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod old_router_abort {
    use std::sync::Mutex;

    use live_edit::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};

    use super::*;
    use crate::storage::snapshot_read_support::ProxyRule;

    /// How long the proxy holds a held `DISCARD` probe before it forwards it.
    /// It must exceed the 650 ms probe deadline plus the repair's 2 s `QUIT`
    /// budget (2.65 s): fred writes a `QUIT` only once the commands already in
    /// flight on the connection are answered (`write_command`'s `drain_all`,
    /// `vendor/fred-10.1.0/src/router/commands.rs:122-124`), so the old
    /// router's `QUIT` waits behind the held `DISCARD`, the `QUIT` budget fires
    /// first, and the old router is still running when the repair moves on.
    /// 4 s leaves 1.35 s for either timer to fire late.
    const DISCARD_HOLD: Duration = Duration::from_secs(4);

    /// After the proxy releases a held `DISCARD`: long enough for an old
    /// router still running to read its reply, write its `QUIT`, end, and store
    /// its command receiver back on the client (a few milliseconds at most:
    /// one local round trip and the router's own teardown).
    const ORPHAN_END_MARGIN: Duration = Duration::from_secs(1);

    /// The opening words of the WARN a repair logs when its `QUIT` times out.
    const QUIT_TIMEOUT_WARN: &str = "txn slot recycle: QUIT did not complete in time";

    /// A [`ScriptReplyProxy`] in front of `TEST_VALKEY_URL` that holds each
    /// `DISCARD` it has been armed for, by [`DISCARD_HOLD`], and forwards
    /// every other frame unchanged.
    struct DiscardHolder {
        proxy: ScriptReplyProxy,
        armed: Arc<AtomicUsize>,
        last_held_at: Arc<Mutex<Option<Instant>>>,
    }

    impl DiscardHolder {
        async fn start() -> Self {
            let armed = Arc::new(AtomicUsize::new(0));
            let last_held_at = Arc::new(Mutex::new(None));
            let rule: ProxyRule = {
                let (armed, last_held_at) = (Arc::clone(&armed), Arc::clone(&last_held_at));
                Arc::new(move |args: &[Vec<u8>]| {
                    if !args.first()?.eq_ignore_ascii_case(b"DISCARD") {
                        return None;
                    }
                    armed
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                        .ok()?;
                    *last_held_at
                        .lock()
                        .expect("the hold clock is never poisoned") = Some(Instant::now());
                    Some(ProxyAction::Delay(DISCARD_HOLD))
                })
            };
            Self {
                proxy: ScriptReplyProxy::start(test_valkey_addr(), rule).await,
                armed,
                last_held_at,
            }
        }

        /// Hold the next `DISCARD` that reaches the proxy.
        fn hold_next_discard(&self) {
            self.armed.fetch_add(1, Ordering::SeqCst);
        }

        /// Wait until the last held `DISCARD` was released, plus
        /// [`ORPHAN_END_MARGIN`]. No signal marks the moment an old router
        /// left running stores its receiver back, so this waits out the hold
        /// the proxy itself times.
        async fn await_release(&self) {
            let held_at = self
                .last_held_at
                .lock()
                .expect("the hold clock is never poisoned")
                .expect("precondition: a DISCARD was held");
            tokio::time::sleep_until(held_at + DISCARD_HOLD + ORPHAN_END_MARGIN).await;
        }
    }

    /// A client connected through `proxy`, with a one-slot `txn` pool, its
    /// observability hooks installed and a probe fault injector.
    async fn proxied_client(
        proxy: &ScriptReplyProxy,
    ) -> (ValkeyClient, Arc<Metrics>, Arc<ProbeFaultInjector>) {
        let mut config: Config = crate::common::test_config();
        config.valkey_url = proxy.url();
        config.valkey_pool_size = 1;
        let client = tokio::time::timeout(CONNECT_BUDGET, connect_test_client(&config))
            .await
            .expect("connect must settle")
            .expect("a client must connect through the proxy");
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let injector = Arc::new(ProbeFaultInjector::new());
        client.install_probe_fault_injector_for_test(Arc::clone(&injector));
        (client, metrics, injector)
    }

    fn dirty_checkouts(metrics: &Metrics, reason: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_valkey_txn_dirty_checkouts_total",
            &[("reason", reason)],
        )
        .unwrap_or_else(|| {
            panic!("valkey_txn_dirty_checkouts_total{{reason={reason:?}}} is seeded")
        }) as u64
    }

    /// How many repairs have logged their `QUIT` timing out.
    fn quit_timeouts(capture: &ThreadCapture) -> usize {
        capture
            .snapshot()
            .events
            .iter()
            .filter(|event| {
                event["fields"]["message"]
                    .as_str()
                    .is_some_and(|message| message.starts_with(QUIT_TIMEOUT_WARN))
            })
            .count()
    }

    /// Check out the only slot (a clean probe) and read its `CLIENT ID`.
    async fn slot_connection_id(client: &ValkeyClient, what: &str) -> i64 {
        let slot = checkout_by(client, settle_deadline(), what).await;
        connection_id(&*slot).await
    }

    /// Recycle the only slot through a `probe_timeout` whose repair's `QUIT`
    /// times out with the old router still running, and return the slot's
    /// `CLIENT ID` after it. `before` is the slot's `CLIENT ID` before it, and
    /// `held` the number of `DISCARD`s held so far, this one included.
    ///
    /// The proxy holds the probe's `DISCARD`: the probe times out at 650 ms,
    /// the checkout recycles the slot, and the repair's `QUIT` queues behind
    /// the held `DISCARD` in the old router, so its 2 s budget fires while that
    /// router is still running, draining. Preconditions: the `DISCARD` was
    /// held, the verdict was `probe_timeout`, the `QUIT` timed out, and the
    /// slot moved to a new connection.
    async fn recycle_behind_a_held_discard(
        client: &ValkeyClient,
        holder: &DiscardHolder,
        metrics: &Metrics,
        capture: &ThreadCapture,
        before: i64,
        held: usize,
    ) -> i64 {
        holder.hold_next_discard();
        let after = slot_connection_id(client, "the probe_timeout checkout").await;
        assert_eq!(
            holder.proxy.hits(),
            held,
            "precondition: the checkout's DISCARD probe was held"
        );
        assert_eq!(
            dirty_checkouts(metrics, "probe_timeout"),
            held as u64,
            "precondition: the held DISCARD read as a probe_timeout and recycled the slot"
        );
        assert_eq!(
            quit_timeouts(capture),
            held,
            "precondition: the repair's QUIT timed out behind the held DISCARD, so the old router \
             was still running when the repair moved on"
        );
        assert_ne!(
            after, before,
            "precondition: the recycle moved the slot to a new connection"
        );
        after
    }

    /// Defer-9 regression. A `probe_timeout` recycle whose `QUIT` times out
    /// moves on while the old router is still running, draining the held
    /// `DISCARD`. The test then waits until the proxy has released that
    /// `DISCARD`: an old router left running would by then have its reply,
    /// write its `QUIT`, end, and store its command receiver back on the
    /// client — a receiver the recycle's `connect()` had already replaced. The
    /// next recycle (a leaked `MULTI`, injected) must still rebuild the slot:
    /// its `CLIENT ID` changes, and the slot answers `PING`.
    ///
    /// Without the repair's `abort_current`, the stored-back receiver makes the
    /// next recycle's `quit()` return at once, and its `connect()` starts a
    /// router on that dead channel, while the slot's commands still reach the
    /// router the first recycle started: the `CLIENT ID` does not change.
    #[tokio::test]
    async fn txn_recycle_after_a_timed_out_quit_rebuilds_the_slot_on_a_new_connection() {
        let capture = ThreadCapture::start();
        let holder = DiscardHolder::start().await;
        let (client, metrics, injector) = proxied_client(&holder.proxy).await;
        let opening = slot_connection_id(&client, "the opening checkout").await;

        let first =
            recycle_behind_a_held_discard(&client, &holder, &metrics, &capture, opening, 1).await;
        holder.await_release().await;

        injector.push_discard(ProbeFault::ok());
        let phase = settle_deadline();
        let slot = checkout_by(&client, phase, "the leaked-MULTI checkout").await;
        let second = connection_id(&*slot).await;
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the leaked-MULTI probe fault was consumed"
        );
        assert_ne!(
            second, first,
            "the recycle after a timed-out QUIT must move the slot to a new connection; the same \
             CLIENT ID means its commands still reach the router the first recycle started, and \
             the second recycle's router was started on a dead command channel"
        );
        assert_answers_ping(&*slot, phase, "the twice-recycled txn slot").await;
        drop(slot);
        client.quit().await;
    }

    /// The router a repair aborts is neither counted nor restarted: no
    /// `router_restarts_total{pool="txn"}` and no restart ERROR, after a
    /// recycle that aborted its old router while it was still running (the
    /// `QUIT`-timeout arm) and after one whose `QUIT` succeeded. Then a router
    /// ended by anything else is: ending the slot's router counts and logs one
    /// restart, which is also what shows the zero above is not a counter that
    /// never moves.
    #[tokio::test]
    async fn txn_router_a_repair_aborted_is_neither_counted_nor_restarted() {
        let capture = ThreadCapture::start();
        let holder = DiscardHolder::start().await;
        let (client, metrics, injector) = proxied_client(&holder.proxy).await;
        let opening = slot_connection_id(&client, "the opening checkout").await;

        let first =
            recycle_behind_a_held_discard(&client, &holder, &metrics, &capture, opening, 1).await;
        injector.push_discard(ProbeFault::ok());
        let second = slot_connection_id(&client, "the leaked-MULTI checkout").await;
        assert_eq!(
            (
                injector.remaining(),
                second != first,
                quit_timeouts(&capture)
            ),
            (0, true, 1),
            "precondition: the leaked-MULTI probe fault was consumed and recycled the slot, and \
             that repair's QUIT did not time out"
        );

        assert_eq!(
            restarts_row(&metrics),
            POOLS.iter().map(|p| (*p, 0)).collect::<Vec<_>>(),
            "a router a repair ended is not a router death: no restart may be counted"
        );
        let captured = capture.snapshot();
        assert!(
            captured.with_message(RESTART_MSG).is_empty(),
            "a router a repair ended must not log the restart ERROR; full capture:\n{}",
            captured.text
        );

        assert!(
            client.end_router_for_test("txn", 0),
            "precondition: txn slot 0 has a supervised router"
        );
        let restarted = settle_deadline();
        assert_one_restart(&capture, &metrics, "txn", restarted).await;
        let slot = checkout_by(&client, restarted, "the restarted slot").await;
        assert_answers_ping(&*slot, restarted, "the restarted txn slot").await;
        drop(slot);
        client.quit().await;
    }

    /// Two `probe_timeout` recycles back to back, each with its `QUIT` timed
    /// out and its old router still running: the second must abort the router
    /// the FIRST started (the supervisor records it when the first hands it
    /// over), not the one the slot was connected with, which the first already
    /// ended. A clean checkout between them ends the unknown-state streak, so
    /// the second `probe_timeout` recycles too. After the second hold is
    /// released, a third recycle (a leaked `MULTI`) must still move the slot
    /// to a new connection.
    ///
    /// Had the second repair aborted the wrong router, the first recycle's
    /// router would end later and store its receiver back, and the third
    /// recycle would leave the slot on the second's connection.
    #[tokio::test]
    async fn back_to_back_txn_recycles_each_abort_the_router_the_previous_one_started() {
        let capture = ThreadCapture::start();
        let holder = DiscardHolder::start().await;
        let (client, metrics, injector) = proxied_client(&holder.proxy).await;
        let opening = slot_connection_id(&client, "the opening checkout").await;

        let first =
            recycle_behind_a_held_discard(&client, &holder, &metrics, &capture, opening, 1).await;
        drop(checkout_by(&client, settle_deadline(), "the clean checkout between").await);
        let second =
            recycle_behind_a_held_discard(&client, &holder, &metrics, &capture, first, 2).await;
        holder.await_release().await;

        injector.push_discard(ProbeFault::ok());
        let phase = settle_deadline();
        let slot = checkout_by(&client, phase, "the leaked-MULTI checkout").await;
        let third = connection_id(&*slot).await;
        assert_eq!(
            injector.remaining(),
            0,
            "precondition: the leaked-MULTI probe fault was consumed"
        );
        assert_ne!(
            third, second,
            "the third recycle must move the slot to a new connection; the same CLIENT ID means \
             the second repair aborted a router other than the one the first started, which then \
             stored its command receiver back"
        );
        assert_answers_ping(&*slot, phase, "the thrice-recycled txn slot").await;
        drop(slot);
        client.quit().await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Amendment R4-1 — a redirected overwrite commit, tried once more
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod overwrite_commit_redirect {
    use live_edit::storage::lua::CREATE_LAYER_COMMIT_SCRIPT;

    use super::*;
    use crate::grpc::log_capture::capture;

    /// The opening words of the WARN `dispatch_create_layer_commit` logs when
    /// the sync before a repeated commit run — here the overwrite's second
    /// try — fails or times out.
    const SYNC_WARN_PREFIX: &str = "Cluster routing sync before a repeated CreateLayer commit run";

    /// Amendment R4-1 on the wire: an overwrite's first commit `EVALSHA`
    /// (`create_layer_commit.lua`, on the `txn_pool`, which follows no
    /// redirect) is answered `-<redirect>` by the clustered facade. The call
    /// reads the mapping, finds the old object, and tries once more on a fresh
    /// checkout — after syncing that checkout's cluster routing (the sync
    /// counter reads 1). The facade answers only that first `EVALSHA`, so the
    /// second try commits: the call answers the new object, the mapping names
    /// it, and an update of it commits.
    ///
    /// What this does and does not show: the facade is one node owning every
    /// slot, so the sync re-reads the same routing. It shows the real redirect
    /// takes the second-try path, that the sync runs before the second try,
    /// and that a second try on a fresh checkout recovers once the slot is
    /// answerable; it does not reproduce a resharded slot's new owner. (The
    /// injected-fault shapes are in `b_v6d_tests.rs`'s `overwrite_second_try`.)
    ///
    /// The sync counter counts at the sync's gate, before the sync is sent, so
    /// the facade also counts the `CLUSTER SLOTS` frames it receives after the
    /// trap. On an `-ASK` nothing else syncs — the `txn_pool` command fails
    /// inside `process_ask`, at its redirection budget
    /// (`decr_check_redirections`, `vendor/fred-10.1.0/src/router/commands.rs:261`)
    /// before any `ASKING` is sent, and `process_ask` syncs nothing on any
    /// path — so at least one such frame is the second try's sync on the wire. (A
    /// `-MOVED` makes fred sync on its own, so the `MOVED` variant does not
    /// read the count.)
    ///
    /// With `slots` [`SlotsAfterTrap::Slow`] the facade holds that sync's
    /// `CLUSTER SLOTS` reply past the client's command timeout: the sync's
    /// bound fires, the dispatch logs its sync WARN and sends the second try
    /// anyway, and the overwrite still commits once the sync lands.
    async fn redirected_overwrite_commit_recovers_on_a_second_try(
        redirect: &'static str,
        slots: SlotsAfterTrap,
    ) {
        let name = format!(
            "bv6d-r41-{}-{}",
            redirect.to_ascii_lowercase(),
            if slots == SlotsAfterTrap::Slow {
                "slow-sync"
            } else {
                "synced"
            }
        );
        let sha = CREATE_LAYER_COMMIT_SCRIPT.sha1().to_owned();
        // Armed only once the layer exists: its creation runs the same script.
        let armed = Arc::new(AtomicBool::new(false));
        let facade = {
            let armed = Arc::clone(&armed);
            AskFacade::start_redirecting(
                redirect,
                slots,
                move |args| armed.load(Ordering::SeqCst) && names(args, b"EVALSHA", &sha),
                Duration::ZERO,
                |_| None,
            )
            .await
        };
        let tv = facade.test_valkey(4).await;
        let (session_id, old) = layer(&tv, &name).await;
        let syncs = Arc::new(AtomicUsize::new(0));
        let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_create_layer_commit_cluster_sync_counter_for_test(Arc::clone(&syncs));

        armed.store(true, Ordering::SeqCst);
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..Default::default()
        };
        // A slow sync holds the call for SLOW_SLOTS_DELAY by construction.
        let overwrite_budget = if slots == SlotsAfterTrap::Slow {
            SLOW_SLOTS_DELAY + SETTLE
        } else {
            SETTLE
        };
        let (created, captured) = capture(tokio::time::timeout(
            overwrite_budget,
            store.create_layer_with_operation(
                &session_id,
                &name,
                "bv6d-user",
                &state,
                Arc::new(MockExternalValueStore::new()),
                None,
            ),
        ))
        .await;
        let created = created
            .expect("the overwrite did not settle within its budget")
            .unwrap_or_else(|error| {
                panic!(
                    "the overwrite must commit on its second try after the -{redirect}; got {error:?}"
                )
            });

        assert_eq!(
            facade.trap.fired(),
            1,
            "precondition: the first commit EVALSHA was answered -{redirect}"
        );
        assert_eq!(
            syncs.load(Ordering::SeqCst),
            1,
            "the second try must sync its checkout's cluster routing first"
        );
        if redirect == "ASK" {
            assert!(
                facade.slots_after_trap.load(Ordering::SeqCst) >= 1,
                "the second try's routing sync must reach the wire: no CLUSTER SLOTS arrived \
                 after the -ASK"
            );
        }
        let sync_warnings: Vec<_> = captured
            .events
            .iter()
            .filter(|event| {
                event["level"] == "WARN"
                    && event["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.starts_with(SYNC_WARN_PREFIX))
            })
            .collect();
        assert_eq!(
            sync_warnings.len(),
            usize::from(slots == SlotsAfterTrap::Slow),
            "a sync that outlives its bound logs one WARN and the second try is sent anyway; an \
             answered one logs none. Full capture:\n{}",
            captured.text
        );
        // The WARN names the layer-mapping key the checkout was pinned to under
        // its own field, `mapping_key`; the re-sync after a redirected
        // transaction uses `key`, and the two must not be merged.
        let mapping_key = tv.key_schema().layer_mapping(&session_id, &name);
        for warning in &sync_warnings {
            assert_eq!(
                warning["fields"]["mapping_key"].as_str(),
                Some(mapping_key.as_str()),
                "the second-try sync WARN carries the mapping key as `mapping_key`: {warning}"
            );
            assert!(
                warning["fields"]["key"].is_null(),
                "the second-try sync WARN has no `key` field: {warning}"
            );
        }
        assert_ne!(created.object_id, old);
        let mapped: Option<String> = tv
            .client()
            .pool
            .get(tv.key_schema().layer_mapping(&session_id, &name))
            .await
            .expect("GET of the layer mapping must succeed");
        assert_eq!(mapped, Some(created.object_id.to_string()));
        let updated = commit(&store, &session_id, &name, created.object_id, 7, None)
            .await
            .expect("an update of the new object must commit");
        assert_eq!(updated.seq, 2);
        drop(facade.proxy);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_commit_answered_ask_commits_on_a_synced_second_try() {
        redirected_overwrite_commit_recovers_on_a_second_try("ASK", SlotsAfterTrap::Answer).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_commit_answered_moved_commits_on_a_synced_second_try() {
        redirected_overwrite_commit_recovers_on_a_second_try("MOVED", SlotsAfterTrap::Answer).await;
    }

    /// The routing sync before the second try outlives its bound (the facade
    /// holds its `CLUSTER SLOTS` reply for [`SLOW_SLOTS_DELAY`]): the dispatch
    /// logs the sync WARN and sends the second try anyway, which commits once
    /// the sync lands. Slow by construction — the call waits out
    /// [`SLOW_SLOTS_DELAY`] (about 7.5 s in all).
    ///
    /// An error reply to `CLUSTER SLOTS` cannot reach the sync-failed WARN
    /// (the dispatch's `Ok(Err(_))` arm). fred's `sync_cluster_with_policy`
    /// (`vendor/fred-10.1.0/src/router/utils.rs:557-586`) retries a sync error
    /// under the client's unlimited reconnect policy, and on the errors it does
    /// not retry (`should_not_reconnect`: `Config` / `Url`, `:569-571`) it
    /// breaks out and still answers `Ok(())` (`:585`). So a refused
    /// `CLUSTER SLOTS` only keeps the sync running, and the timed-out WARN is
    /// what a slow or refused sync reaches. What does reach `Ok(Err(_))` is
    /// the member's router itself failing the sync: the router ending or
    /// closing while the sync is in it (its reply channel is dropped, so the
    /// caller gets fred's `channel closed`), the sync command failing to reach
    /// the router at all, or a close notice during the retry backoff
    /// (`read_and_sleep`'s `Canceled` "Connection(s) closed."). That arm is not
    /// staged here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_commit_second_try_is_sent_after_a_routing_sync_that_times_out() {
        redirected_overwrite_commit_recovers_on_a_second_try("ASK", SlotsAfterTrap::Slow).await;
    }

    /// How the finisher's second run fails, in [`finisher_routing_syncs`].
    #[derive(Clone, Copy, Debug)]
    enum FinisherSecondRun {
        /// A plain fred `IO` error, which the redirect-class classifier
        /// (`is_router_retry_budget_exhausted_error`) does not match. (A real
        /// dropped connection does not reach a one-attempt `txn_pool` command
        /// this way: see [`Self::WriteAttemptsExhausted`].)
        Io,
        /// `Unknown` "Too many failed write attempts."
        /// ([`BUDGET_EXHAUSTED_SHAPES`]`[0]`): what a dropped connection, or
        /// a `MOVED`, gives a one-attempt `txn_pool` command. The classifier
        /// matches it, so the finisher syncs once more after it — the extra
        /// sync the finisher's docs accept for a dropped connection.
        WriteAttemptsExhausted,
        /// The `-ASK` shape a `txn_pool` command fails with ("Too many
        /// redirections.", [`BUDGET_EXHAUSTED_SHAPES`]`[1]`).
        Redirected,
    }

    /// An overwrite on a clustered client (the facade, its trap never armed)
    /// whose two commit runs fail with a plain `IO` error: the call gives up
    /// and spawns its finisher. The finisher's first run fails the same way,
    /// its second fails as `second_run` says, and its third commits. Returns
    /// the routing syncs counted across the call and the finisher, once the
    /// finisher's run has landed and its tail is done.
    ///
    /// The runs, and which sync before a repeated run each one counts: the
    /// call's first (none) and second (one, its repeated run); the
    /// finisher's first (one, always), second (none: the first failed with
    /// a plain `IO` error), third (one only after a second run the
    /// redirect-class classifier matches). The faults are injected and the
    /// facade is one node owning every slot, so this counts the syncs, not a
    /// stale member rerouted.
    async fn finisher_routing_syncs(second_run: FinisherSecondRun) -> usize {
        use std::collections::VecDeque;
        use std::sync::Mutex;

        use fred::error::{Error as FredError, ErrorKind as FredErrorKind};

        use live_edit::storage::client::BUDGET_EXHAUSTED_SHAPES;
        use live_edit::storage::layer::CreateCommitFault;

        let name = format!(
            "bv6d-finisher-sync-{}",
            format!("{second_run:?}").to_ascii_lowercase()
        );
        let io = || CreateCommitFault {
            error: FredError::new(FredErrorKind::IO, "bv6d: connection reset"),
            script_ran: false,
        };
        let shape = |index: usize| {
            let (kind, message) = BUDGET_EXHAUSTED_SHAPES[index].clone();
            CreateCommitFault {
                error: FredError::new(kind, message),
                script_ran: false,
            }
        };
        let finisher_second = match second_run {
            FinisherSecondRun::Io => io(),
            FinisherSecondRun::WriteAttemptsExhausted => shape(0),
            FinisherSecondRun::Redirected => shape(1),
        };
        // The call's two runs, the finisher's first and second; its third
        // runs for real.
        let faults = Arc::new(Mutex::new(VecDeque::from([
            io(),
            io(),
            io(),
            finisher_second,
        ])));
        let facade = AskFacade::start(|_| false, Duration::ZERO, |_| None).await;
        let tv = facade.test_valkey(4).await;
        let (session_id, old) = layer(&tv, &name).await;
        let syncs = Arc::new(AtomicUsize::new(0));
        let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_create_layer_commit_cluster_sync_counter_for_test(Arc::clone(&syncs))
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&faults))
            .with_create_commit_finisher_timing_for_test(
                Duration::from_millis(10),
                Duration::from_millis(10),
                SETTLE,
            );

        let refused = tokio::time::timeout(
            SETTLE,
            store.create_layer_with_operation(
                &session_id,
                &name,
                "bv6d-user",
                &ObjectState {
                    sequence_num: 1,
                    first_version: 1,
                    source_format: "usda".to_owned(),
                    ..Default::default()
                },
                Arc::new(MockExternalValueStore::new()),
                None,
            ),
        )
        .await
        .expect("the overwrite did not settle within its budget");
        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "precondition: both of the call's runs failed and it gave up; got {refused:?}"
        );
        // The call's object is the last one allocated.
        let new_object: String = tv
            .client()
            .pool
            .get(tv.key_schema().counter_object_id())
            .await
            .expect("GET of the object-id counter must succeed");
        assert_ne!(new_object, old.to_string(), "precondition: a new object");
        let mapping_key = tv.key_schema().layer_mapping(&session_id, &name);
        let deadline = settle_deadline();
        loop {
            let mapped: Option<String> = tv
                .client()
                .pool
                .get(&mapping_key)
                .await
                .expect("GET of the layer mapping must succeed");
            if mapped.as_deref() == Some(new_object.as_str()) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the finisher's third run did not map the new object within {SETTLE:?}; the \
                 mapping is {mapped:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        store.wait_for_cleanup(SETTLE).await;
        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: the call's two runs and the finisher's first two each took a fault"
        );
        drop(facade.proxy);
        syncs.load(Ordering::SeqCst)
    }

    /// The finisher syncs its checkout's cluster routing before its first
    /// run and again before the run after a redirect-class failure: three
    /// syncs in all — the call's repeated run, the finisher's first run and
    /// its third — and none before its second, which follows a plain `IO`
    /// error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finisher_syncs_on_its_first_run_and_again_after_a_redirect_class_failure() {
        assert_eq!(
            finisher_routing_syncs(FinisherSecondRun::Redirected).await,
            3,
            "the call's repeated run, the finisher's first run, and its run after the \
             redirect-class failure sync; its run after the plain IO error does not"
        );
    }

    /// The finisher also syncs again after "Too many failed write attempts."
    /// — the shape a dropped connection gives a one-attempt `txn_pool`
    /// command — since the redirect-class classifier cannot tell it from a
    /// `MOVED`: three syncs, the same as after an `-ASK`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finisher_resyncs_after_a_write_attempts_exhausted_failure() {
        assert_eq!(
            finisher_routing_syncs(FinisherSecondRun::WriteAttemptsExhausted).await,
            3,
            "the call's repeated run, the finisher's first run, and its run after the \
             write-attempts-exhausted failure sync"
        );
    }

    /// After only plain `IO` errors, which the redirect-class classifier does
    /// not match, the finisher syncs before its first run and never again:
    /// two syncs in all, the call's repeated run and the finisher's first
    /// run, over the finisher's three runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finisher_does_not_resync_after_an_io_failure() {
        assert_eq!(
            finisher_routing_syncs(FinisherSecondRun::Io).await,
            2,
            "only the call's repeated run and the finisher's first run sync"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// b.cyu — a redirected fred `Transaction` re-syncs its checkout (`exec_txn`)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod redirected_transaction_resync {
    use std::sync::Mutex;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{
        CheckpointError, CheckpointMeta, CheckpointStore, SessionCheckpoint, SessionRecoveryHook,
        SessionRecoveryOutcome,
    };

    use super::*;
    use crate::grpc::log_capture::capture;

    /// The WARN `ValkeyClient::exec_txn`'s re-sync logs when the routing sync
    /// after Step B's redirected transaction outlives its bound.
    const STEP_B_SYNC_TIMED_OUT: &str = "Cluster routing sync of a txn-pool connection after the \
                                         session checkpoint apply Step B transaction was \
                                         redirected timed out";

    /// What fred's transaction router fails a redirected transaction with.
    const REDIRECTED: &str = "Max redirections exceeded";

    /// Session recovery through the clustered facade, whose Step B `EXEC`
    /// (the transaction that queued the session-meta `HSET`) is answered
    /// `-ASK`. The `txn_pool` follows no redirect, so fred fails the
    /// transaction "Max redirections exceeded" and `exec_txn` re-syncs the
    /// checkout's cluster routing before the applier sees the error: the sync
    /// counter rises by exactly one. The facade holds that sync's
    /// `CLUSTER SLOTS` past the client's command timeout, so the re-sync logs
    /// its timed-out WARN — once, naming the Step B transaction and carrying
    /// the session-meta key the checkout was pinned to as `key` — and the
    /// applier still answers `Transient` with the transaction's error.
    ///
    /// Step B's first behavioural test of the re-sync: a standalone store
    /// never syncs (the sync is skipped on a non-clustered client), and the
    /// cluster-gated `b_cyu_slot_migration_test` reaches only `DeleteSession`
    /// and `CreateLayer`. Without the re-sync the counter stays put and no
    /// WARN is logged. Slow by construction: the call waits out the client's
    /// 5 s command timeout on the held sync.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn redirected_session_checkpoint_step_b_resyncs_its_checkout_and_warns_on_a_slow_sync() {
        // The session-meta key, once known: the trap fires on the `EXEC` of
        // the connection that queued an `HSET` of it.
        let meta_key: Arc<Mutex<Option<String>>> = Arc::default();
        let queued_on = Arc::new(AtomicUsize::new(usize::MAX));
        let facade = {
            let (meta_key, queued_on) = (Arc::clone(&meta_key), Arc::clone(&queued_on));
            AskFacade::start_redirecting_exec(
                "ASK",
                SlotsAfterTrap::Slow,
                move |conn, args: &[Vec<u8>]| {
                    let Some(key) = meta_key.lock().unwrap().clone() else {
                        return false;
                    };
                    if names(args, b"HSET", &key) {
                        queued_on.store(conn, Ordering::SeqCst);
                        return false;
                    }
                    args.first()
                        .is_some_and(|n| n.eq_ignore_ascii_case(b"EXEC"))
                        && queued_on.load(Ordering::SeqCst) == conn
                },
            )
            .await
        };
        let tv = facade.test_valkey(4).await;

        let session_id = format!("bcyu-step-b-{}", uuid::Uuid::new_v4());
        let base_path = format!("/bcyu-step-b/{session_id}");
        let mut checkpoint = SessionCheckpoint {
            schema_version:
                live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.clone(),
            name: "bcyu-step-b".to_owned(),
            base_path,
            created_by: "bv6d-user".to_owned(),
            created_at: 1_700_000_000_000_000,
            layer_mappings: Default::default(),
            layers_index: Default::default(),
            path_name_members: Default::default(),
            path_index_session_id_member: session_id.clone(),
        };
        checkpoint.add_path_name_member("bcyu-step-b", &session_id);
        let store = Arc::new(InMemoryCheckpointStore::new());
        store
            .put_session_checkpoint(
                &session_id,
                checkpoint.to_bytes().expect("the checkpoint encodes"),
                CheckpointMeta::default(),
            )
            .await
            .expect("the in-memory store takes the checkpoint");
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(config),
            Arc::new(crate::common::metrics_scrape_only("bcyu-step-b")),
        );
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        *meta_key.lock().unwrap() = Some(session_meta_key.clone());

        let syncs_before = tv.client().txn_routing_syncs_for_test();
        let (outcome, captured) = capture(tokio::time::timeout(
            SLOW_SLOTS_DELAY + SETTLE,
            hook.try_recover_session(&session_id),
        ))
        .await;
        let outcome = outcome.expect("the recovery did not settle within its budget");
        let syncs = tv.client().txn_routing_syncs_for_test() - syncs_before;

        assert_eq!(
            facade.trap.fired(),
            1,
            "precondition: Step B's EXEC was answered -ASK; recovery answered {outcome:?}"
        );
        match &outcome {
            SessionRecoveryOutcome::Error(CheckpointError::Transient { source }) => {
                let source = source.to_string();
                assert!(
                    source.contains("Step B") && source.contains(REDIRECTED),
                    "Step B's redirected transaction answers Transient with fred's \
                     {REDIRECTED:?}; got {source}"
                );
            }
            other => panic!("a redirected Step B answers Transient; got {other:?}"),
        }
        assert_eq!(
            syncs, 1,
            "`exec_txn` re-syncs the redirected checkout's routing exactly once"
        );
        let warnings: Vec<_> = captured
            .events
            .iter()
            .filter(|event| {
                event["level"] == "WARN"
                    && event["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.starts_with(STEP_B_SYNC_TIMED_OUT))
            })
            .collect();
        assert_eq!(
            warnings.len(),
            1,
            "the re-sync that outlives its bound logs one WARN naming the Step B transaction. \
             Full capture:\n{}",
            captured.text
        );
        assert_eq!(
            warnings[0]["fields"]["key"].as_str(),
            Some(session_meta_key.as_str()),
            "the re-sync WARN carries the key the checkout was pinned to as `key`: {}",
            warnings[0]
        );
        assert!(
            warnings[0]["fields"]["timeout_ms"].is_u64(),
            "the timed-out WARN carries its bound: {}",
            warnings[0]
        );
        drop(facade.proxy);
    }
}
