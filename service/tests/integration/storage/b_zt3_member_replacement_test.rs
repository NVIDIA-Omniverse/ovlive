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

//! b.zt3 — the unresponsive policy every client is built with, the metrics'
//! cells, every way a member replacement can end, and (postcomp-1) how
//! members due at once are scheduled and how many old clients retire at once.
//!
//! The replacement tests drive the trigger with
//! `ValkeyClient::fast_verdict_for_test`, which counts a fast verdict against a
//! member's current client exactly as its error hook does when fred closes a
//! connection with one: the trigger, cooldown and budget apply unchanged. The
//! fast verdict itself, through a silenced connection, is CORE-1's subject
//! (`b_zt3_dead_connection_test.rs`), as is the happy path's lifecycle and the
//! retired client refusing sends at once (CORE-2). A fresh client's connect is
//! made to hang where a case needs it by a [`ScriptReplyProxy`] stalling every
//! `PING` (fred's connection setup sends one; nothing else in these tests
//! does).
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`every_pool_client_shares_one_peer_registry_and_only_the_readers_lack_the_fast_rule`] | T9 | the registry and bounds on every builder: data, maintenance, snapshot, `txn`, the pub/sub subscriber and a stream reader |
//! | [`the_unresponsive_bounds_fit_the_command_timeout_and_clear_every_wait_they_judge`] | T9 | the constants' arithmetic; pure |
//! | [`every_unresponsive_and_member_replacement_cell_is_registered_at_zero`] | T10 | pre-registration; pure |
//! | [`a_replacement_whose_fresh_client_cannot_connect_keeps_the_member_and_leaves_nothing_behind`] | T13 | `connect_failed`: old client kept, no spare cell used, no watch or hook pair left |
//! | [`quit_while_a_replacement_is_connecting_its_fresh_client_leaves_nothing_behind`] | T13 | `quit()` before activation |
//! | [`quit_after_a_replacement_took_its_member_over_ends_the_retirement_and_leaves_nothing_behind`] | T13 | `quit()` after activation, in the retirement grace: the old client released at once |
//! | [`once_the_spare_cells_are_used_up_a_due_replacement_is_refused_counted_each_time_and_warned_once`] | T13, T10 | `budget_exhausted` ticks per refusal, one WARN per pool, members kept |
//! | [`a_fast_verdict_inside_the_cooldown_logs_that_the_replacement_did_not_cure_it`] | T13 | the "did not cure" WARN naming both clients, no second replacement |
//! | [`a_retired_client_whose_router_was_aborted_keeps_no_credential_task_backchannel_or_open_command_channel`] | T13 | `fred::live_edit::pool::retire` after an overrun `QUIT`'s abort |
//! | [`ending_a_replaced_members_router_restarts_the_fresh_clients_router`] | T15 | `end_router_for_test` after a replacement reaches the fresh client |
//! | [`a_member_due_while_another_replacement_connects_is_replaced_at_its_swap_without_another_verdict`] | T-R1 (postcomp-1) | k = 2: the second due member waits for the first's connect, not its retirement; replaced within 2 × C of the slot's release, no further verdict |
//! | [`four_members_due_at_once_are_replaced_in_index_order_and_retire_side_by_side`] | T-R2 | k = 4: index order, the fourth within 4 × C, four retirements at once, all cut short by `quit()` |
//! | [`a_retirement_that_ends_while_a_later_replacement_holds_the_slot_leaves_the_slot_held`] | T-R4 | only the slot's holder releases it |
//! | [`six_members_due_at_once_are_all_replaced_and_never_more_than_the_cap_retire_at_once`] | Rider A | k = 6 on one pool: all replaced, at most `MEMBER_RETIRING_CAP` retiring, the two oldest ended (WARN each, refusing sends), the rest still serving |
//! | [`a_write_in_flight_on_a_retirement_the_cap_ends_reaches_its_caller_as_outcome_unknown`] | Rider A condition | the evicted client's in-flight write fails with fred's `Unknown`; no store-refusal hint; `INTERNAL`, no status details |
//! | [`a_lone_maintenance_retirement_in_its_grace_is_ended_by_the_cap_like_any_other`], [`two_maintenance_retirements_in_their_grace_are_ended_by_the_cap_oldest_first`] | Rider A, no pool exempt | `maintenance` retirements in their grace, the oldest at the cap: each later data swap ends one, oldest first (one WARN each, refusing sends); every data old client still serves; never more than the cap |
//! | [`four_maintenance_retirements_in_their_grace_never_hold_the_count_above_the_cap`] | Rider A bound, no pool exempt | four `maintenance` retirements in their grace fill the cap: four data swaps end them in turn, and the count never reaches five |
//! | [`a_member_due_behind_a_replacement_whose_connect_failed_is_started_when_that_attempt_ends`] | test review 4(iii) | `replace()`'s closing re-check starts a member pending behind a failed connect |
//!
//! Where a pool member's counters are asserted per event (T10's "each ticks
//! once"): `fast` per verdict and `replaced` per replacement in CORE-2, `slow`
//! per verdict in `b_zt3_liveness_test.rs`, `connect_failed` and
//! `budget_exhausted` here, the `txn` pool's `replaced` in
//! `b_zt3_txn_replacement_test.rs`.
//!
//! The waits that mean "a replacement, retirement included, has finished" read
//! `member_replacement_tasks_for_test() == 0`: a replacement frees its pool's
//! slot (`member_replacement_in_flight_for_test`) at the swap, and its task
//! runs on through the old client's retirement.
//!
//! # Wait budget
//!
//! Every wait at its full bound: a client's connect (30 s), fred's 5 s command
//! timeout per command, each polled condition ([`SETTLE`]), each bounded
//! `quit()` ([`SETTLE`]), each scheduling bound (k × C + [`SCHEDULING_SLACK`],
//! C = [`CONNECT_TIMEOUT`] = 5 s), each held connect's release
//! (`ReplyGate::reply_held`, 20 s), the sends through old clients that still
//! serve (one [`SETTLE`] shared by all of a test's), each send through an
//! ended one ([`EVICTED_SEND_BOUND`]). Under the 180 s kill deadline
//! `.config/nextest.toml` sets for the `b_zt3_` modules:
//!
//! | Test | Worst case |
//! |---|---|
//! | the spare-cell test | 30 + four replacements (10 × 2 each) + 10 = 120 s |
//! | T-R1 | 30 + 20 + (2 × 5 + 15) + 10 = 85 s |
//! | T-R2 | 30 + 20 + (4 × 5 + 15) + 10 = 95 s |
//! | T-R4 | 30 + 20 + 10 + 20 + 5 (`RETIREMENT_END_BOUND`) + 10 + 10 = 105 s |
//! | Rider A, k = 6 | 30 + (6 × 5 + 15) + 10 + 4 × 2 + 10 + 10 = 113 s |
//! | Rider A condition | fixture 30 + 30 + 20 + (5 × 5 + 15) + 10 + 10 + purge 10 = 150 s (its own command timeout is raised to 90 s, so its one later read is bounded by [`SETTLE`]) |
//! | Rider A, M = 1, 2 or 4 `maintenance` retirements | 30 + (M × 5 + 15) + ((4 − M) × 5 + 15) + (M × 5 + 15) + 10 + M × 4 + 10 + 10 = 125 + 9M s: 134, 143, 161 s |
//! | 4(iii) | 30 + 10 + (2 × 5 + 15) + 10 = 75 s |

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientLike, KeysInterface, ListInterface};
use fred::prelude::Builder;
use fred::types::config::{Blocking, Config as FredConfig};
use tokio::time::Instant;

use live_edit::error::ServiceError;
use live_edit::observability::metrics::{
    test_util, Metrics, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED,
    VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED, VALKEY_MEMBER_REPLACEMENT_OUTCOMES,
    VALKEY_MEMBER_REPLACEMENT_REPLACED, VALKEY_POOLS, VALKEY_POOL_DATA, VALKEY_REPLACEABLE_POOLS,
    VALKEY_UNRESPONSIVE_VERDICTS,
};
use live_edit::storage::client::{
    is_transient_fred_error, map_fred_err_for_grpc, map_transient_fred_err, store_write_refusal,
    FredErrOrigin, LivenessOverrides, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE,
};
use live_edit::storage::member_replacement::{
    member_retire_grace, ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
    MEMBER_REPLACE_WINDOW, MEMBER_RETIRING_CAP,
};
use live_edit::storage::notifications::XREAD_BLOCK_MS;
use live_edit::storage::test_support::{
    connect_test_client, TestValkey, UNRESPONSIVE_CHECK_INTERVAL, UNRESPONSIVE_FAST_TIMEOUT,
    UNRESPONSIVE_MAX_TIMEOUT, UNRESPONSIVE_PEER_EVIDENCE_LAG,
};

use super::b_zt3_support::{
    await_until, messages_containing, proxied_client, proxied_client_with, MEMBER_REPLACEMENTS,
    ROUTER_RESTARTS, UNRESPONSIVE,
};
use super::mux_test_helpers::{connect_reader_zero, read_series, ScriptedProvider};
use super::snapshot_read_support::{
    test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};
use crate::grpc::log_capture::{Captured, ThreadCapture};

/// Bound on each polled condition and on each `quit()`.
const SETTLE: Duration = Duration::from_secs(10);

/// The production default command timeout (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`
/// = 5000; `common::test_config` uses the same value).
const COMMAND_TIMEOUT: Duration = Duration::from_millis(5000);

/// The bound on a send through a released client: well above an in-process
/// refusal, well below fred's 5 s command timeout.
const RELEASED_SEND_BOUND: Duration = Duration::from_secs(1);

/// The start of the spare-cells WARN.
const BUDGET_WARN: &str = "a pool member is due for replacement but the pool's spare cells";

/// The start of the cooldown WARN.
const NOT_CURED_WARN: &str = "replacement did not cure it";

fn replacements(metrics: &Metrics, outcome: &str) -> i64 {
    read_series(
        metrics,
        MEMBER_REPLACEMENTS,
        &[("pool", VALKEY_POOL_DATA), ("outcome", outcome)],
    )
}

/// Hooks on `client` with a fresh registry, and `knobs`.
async fn armed(client: &ValkeyClient, knobs: ReplaceKnobs) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("zt3"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    client.set_member_replacement_knobs_for_test(knobs);
    metrics
}

/// The production trigger and window, with `cooldown` and `grace`.
fn knobs(cooldown: Duration, grace: Duration) -> ReplaceKnobs {
    ReplaceKnobs {
        after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
        window: MEMBER_REPLACE_WINDOW,
        cooldown,
        grace,
    }
}

/// A client straight to the store with a `members`-member data pool.
async fn direct_client(members: u32) -> ValkeyClient {
    direct_client_with(|config| config.valkey_pool_size = members).await
}

/// A client straight to the store, with `configure` applied to the service
/// config last.
async fn direct_client_with(
    configure: impl FnOnce(&mut live_edit::config::Config),
) -> ValkeyClient {
    let mut config = crate::common::test_config();
    configure(&mut config);
    connect_test_client(&config)
        .await
        .expect("the client must connect")
}

/// Count enough fast verdicts on data member `member`'s current client to
/// trigger a replacement.
fn trigger(client: &ValkeyClient, member: usize) {
    for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
        client.fast_verdict_for_test(POOL_LABEL_DATA, member);
    }
}

/// Replace data member `member` and wait for the replacement to finish,
/// retirement included: the swap frees the pool's slot, and the task ends
/// once the old client's retirement has. Returns the fresh client's id.
async fn replace(client: &ValkeyClient, member: usize) -> String {
    let old = ClientLike::id(client.pool.member(member)).to_owned();
    trigger(client, member);
    await_until(
        "the member must be pointed at a fresh client",
        Instant::now() + SETTLE,
        || ClientLike::id(client.pool.member(member)) != old,
    )
    .await;
    await_until(
        "the replacement's retirement must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == 0,
    )
    .await;
    ClientLike::id(client.pool.member(member)).to_owned()
}

fn armed_ids(client: &ValkeyClient) -> HashSet<String> {
    client
        .armed_hook_client_ids_for_test()
        .into_iter()
        .collect()
}

/// A proxy that stalls every `PING` while `stall` is set: a fresh client's
/// connection setup sends one, so its connect hangs.
async fn ping_stalling_proxy(stall: &Arc<AtomicBool>) -> ScriptReplyProxy {
    let rule: ProxyRule = {
        let stall = Arc::clone(stall);
        Arc::new(move |args: &[Vec<u8>]| {
            (stall.load(Ordering::SeqCst) && args.first()?.eq_ignore_ascii_case(b"PING"))
                .then_some(ProxyAction::Stall)
        })
    };
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// Quit `client` within [`SETTLE`] and check it left no router watcher, data
/// member watch, hook pair, replacement task or retirement behind: the
/// replacer's drain covered every replacement and retirement in progress.
async fn quit_leaves_nothing(client: &ValkeyClient) {
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        0,
        "no replacement task left"
    );
    assert_eq!(
        client.member_retirements_for_test(),
        0,
        "no retirement left"
    );
    assert_eq!(
        client.router_watchers_for_test(),
        0,
        "no router watcher left"
    );
    assert!(
        client.member_watches_for_test(POOL_LABEL_DATA).is_empty(),
        "no data member watch left"
    );
    assert!(
        client.armed_hook_client_ids_for_test().is_empty(),
        "no hook pair left"
    );
    assert!(!client.member_replacement_in_flight_for_test(POOL_LABEL_DATA));
}

/// T9: every client the service builds carries the unresponsive policy, and
/// all of them one peer-reply registry, so a reply on any connection is
/// evidence for any other: the data, maintenance and snapshot members, the
/// `txn` slots and the pub/sub subscriber with the fast rule on, the stream
/// readers (built from their own template) with it off.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_pool_client_shares_one_peer_registry_and_only_the_readers_lack_the_fast_rule() {
    let client = direct_client(2).await;
    connect_reader_zero(&client).await;
    let reader = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("reader 0 holds a connection");
    let mut checkouts = Vec::new();
    for _ in 0..client.txn_pool.slots() {
        checkouts.push(client.acquire_txn().await);
    }
    let registry = client
        .pool
        .member(0)
        .connection_config()
        .live_edit
        .liveness
        .peer_replies
        .clone()
        .expect("the data pool carries a peer registry");

    let mut fast: Vec<(String, fred::types::config::ConnectionConfig)> = Vec::new();
    for (pool, clients) in [
        ("data", client.pool.clients()),
        ("maintenance", client.maintenance_pool.clients()),
        ("snapshot", client.snapshot_pool.clients()),
    ] {
        for (m, member) in clients.iter().enumerate() {
            fast.push((
                format!("{pool} member {m}"),
                member.connection_config().clone(),
            ));
        }
    }
    for (slot, checkout) in checkouts.iter().enumerate() {
        fast.push((
            format!("txn slot {slot}"),
            checkout.connection_config().clone(),
        ));
    }
    fast.push((
        "the pub/sub subscriber".to_owned(),
        client.subscriber.connection_config().clone(),
    ));

    for (who, cfg) in &fast {
        let liveness = &cfg.live_edit.liveness;
        assert_eq!(
            liveness.fast_timeout,
            Some(UNRESPONSIVE_FAST_TIMEOUT),
            "{who}"
        );
        assert_eq!(
            liveness.peer_evidence_lag, UNRESPONSIVE_PEER_EVIDENCE_LAG,
            "{who}"
        );
        assert_eq!(
            cfg.unresponsive.max_timeout,
            Some(UNRESPONSIVE_MAX_TIMEOUT),
            "{who}"
        );
        assert_eq!(
            cfg.unresponsive.interval, UNRESPONSIVE_CHECK_INTERVAL,
            "{who}"
        );
        assert!(
            liveness
                .peer_replies
                .as_ref()
                .is_some_and(|r| r.same_as(&registry)),
            "{who} must share the one peer registry"
        );
    }
    let reader_cfg = reader.connection_config();
    assert_eq!(
        reader_cfg.live_edit.liveness.fast_timeout, None,
        "a stream reader gets the slow rule only"
    );
    assert_eq!(
        reader_cfg.unresponsive.max_timeout,
        Some(UNRESPONSIVE_MAX_TIMEOUT)
    );
    assert!(
        reader_cfg
            .live_edit
            .liveness
            .peer_replies
            .as_ref()
            .is_some_and(|r| r.same_as(&registry)),
        "a stream reader's replies stamp the same registry"
    );
    drop(checkouts);
    client.quit().await;
}

/// T9: the bounds' arithmetic. A fast verdict, one scan and a reconnect fit in
/// the default command timeout, so the re-sent command can still be answered;
/// the slow bound clears the longest legitimate wait (a stream reader's `XREAD
/// BLOCK` budget), so it never closes a healthy connection; the scan comes at
/// least four times per fast bound; the evidence lag is inside the fast bound;
/// a retired client's grace outlasts the command timeout (and, with the
/// timeout disabled, the slow bound), so a command handed to it can finish.
#[test]
fn the_unresponsive_bounds_fit_the_command_timeout_and_clear_every_wait_they_judge() {
    assert!(
        UNRESPONSIVE_FAST_TIMEOUT + UNRESPONSIVE_CHECK_INTERVAL + Duration::from_secs(1)
            <= COMMAND_TIMEOUT
    );
    assert!(UNRESPONSIVE_MAX_TIMEOUT > Duration::from_millis(XREAD_BLOCK_MS + 5_000));
    assert!(UNRESPONSIVE_CHECK_INTERVAL * 4 <= UNRESPONSIVE_FAST_TIMEOUT);
    assert!(UNRESPONSIVE_PEER_EVIDENCE_LAG < UNRESPONSIVE_FAST_TIMEOUT);
    assert!(member_retire_grace(COMMAND_TIMEOUT) >= COMMAND_TIMEOUT);
    assert!(member_retire_grace(Duration::ZERO) > UNRESPONSIVE_MAX_TIMEOUT);
}

/// T10: every `unresponsive_total{pool, verdict}` and
/// `member_replacements_total{pool, outcome}` cell exists, at zero, before
/// anything happened, so a dashboard sees a zero rather than no series.
#[test]
fn every_unresponsive_and_member_replacement_cell_is_registered_at_zero() {
    let metrics = crate::common::metrics_scrape_only("zt3");
    let text = test_util::render(&metrics);
    for pool in VALKEY_POOLS {
        for verdict in VALKEY_UNRESPONSIVE_VERDICTS {
            assert_eq!(
                test_util::sample_value(
                    &text,
                    UNRESPONSIVE,
                    &[("pool", *pool), ("verdict", *verdict)]
                ),
                Some(0.0),
                "{UNRESPONSIVE}{{pool={pool}, verdict={verdict}}}"
            );
        }
    }
    for pool in VALKEY_REPLACEABLE_POOLS {
        for outcome in VALKEY_MEMBER_REPLACEMENT_OUTCOMES {
            assert_eq!(
                test_util::sample_value(
                    &text,
                    MEMBER_REPLACEMENTS,
                    &[("pool", *pool), ("outcome", *outcome)]
                ),
                Some(0.0),
                "{MEMBER_REPLACEMENTS}{{pool={pool}, outcome={outcome}}}"
            );
        }
    }
}

/// T13: a replacement whose fresh client does not connect (its setup `PING`
/// stalled, so the connect bound cuts it) counts `connect_failed` once and
/// leaves the member on its client: no spare cell used, and no router watch,
/// member watch or hook pair left for the fresh client.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replacement_whose_fresh_client_cannot_connect_keeps_the_member_and_leaves_nothing_behind(
) {
    let stall = Arc::new(AtomicBool::new(false));
    let proxy = ping_stalling_proxy(&stall).await;
    let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
    let metrics = armed(
        &client,
        knobs(MEMBER_REPLACE_COOLDOWN, Duration::from_secs(1)),
    )
    .await;
    let old = ClientLike::id(client.pool.member(0)).to_owned();
    let spares = fred::live_edit::pool::spare_cells_left(&client.pool);
    let watchers = client.router_watchers_for_test();
    let member_watches = client.member_watches_for_test(POOL_LABEL_DATA);
    let hooks = armed_ids(&client);

    stall.store(true, Ordering::SeqCst);
    trigger(&client, 0);
    await_until(
        "the failed replacement must end",
        Instant::now() + SETTLE,
        || replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED) == 1,
    )
    .await;
    await_until(
        "the failed replacement must release its pool and end",
        Instant::now() + SETTLE,
        || {
            !client.member_replacement_in_flight_for_test(POOL_LABEL_DATA)
                && client.member_replacement_tasks_for_test() == 0
        },
    )
    .await;
    stall.store(false, Ordering::SeqCst);
    assert_eq!(
        client.member_retirements_for_test(),
        0,
        "a replacement that never swapped retires nothing"
    );

    assert!(
        proxy.hits() >= 1,
        "the fresh client's setup PING must have been stalled"
    );
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED),
        0
    );
    assert_eq!(
        ClientLike::id(client.pool.member(0)),
        old,
        "the member keeps its client"
    );
    assert_eq!(
        fred::live_edit::pool::spare_cells_left(&client.pool),
        spares,
        "no spare cell used"
    );
    assert_eq!(
        client.router_watchers_for_test(),
        watchers,
        "no watcher left for the fresh client"
    );
    assert_eq!(
        client.member_watches_for_test(POOL_LABEL_DATA),
        member_watches
    );
    assert_eq!(
        armed_ids(&client),
        hooks,
        "no hook pair left for the fresh client"
    );
    quit_leaves_nothing(&client).await;
}

/// T13: `quit()` while a replacement's fresh client is still connecting (its
/// setup `PING` stalled) ends the replacement before it takes the member
/// over: it counts neither `replaced` nor `connect_failed`, and leaves no
/// watcher, member watch or hook pair behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quit_while_a_replacement_is_connecting_its_fresh_client_leaves_nothing_behind() {
    let stall = Arc::new(AtomicBool::new(false));
    let proxy = ping_stalling_proxy(&stall).await;
    let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
    let metrics = armed(
        &client,
        knobs(MEMBER_REPLACE_COOLDOWN, Duration::from_secs(1)),
    )
    .await;
    let old = ClientLike::id(client.pool.member(0)).to_owned();

    stall.store(true, Ordering::SeqCst);
    trigger(&client, 0);
    await_until(
        "the fresh client's connect must be under way",
        Instant::now() + SETTLE,
        || proxy.hits() >= 1,
    )
    .await;
    assert!(client.member_replacement_in_flight_for_test(POOL_LABEL_DATA));
    quit_leaves_nothing(&client).await;

    assert_eq!(
        ClientLike::id(client.pool.member(0)),
        old,
        "the member was never taken over"
    );
    for outcome in [
        VALKEY_MEMBER_REPLACEMENT_REPLACED,
        VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED,
    ] {
        assert_eq!(replacements(&metrics, outcome), 0, "{outcome}");
    }
}

/// T13: `quit()` after a replacement took its member over, while the old
/// client sits in its retirement grace, ends the retirement at once: the old
/// client is released (a send through a `&Client` borrowed before the
/// replacement fails at once), and nothing is left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quit_after_a_replacement_took_its_member_over_ends_the_retirement_and_leaves_nothing_behind(
) {
    let client = direct_client(2).await;
    let _metrics = armed(
        &client,
        knobs(MEMBER_REPLACE_COOLDOWN, Duration::from_secs(300)),
    )
    .await;
    let old: &Client = client.pool.member(0);
    let old_id = ClientLike::id(old).to_owned();

    trigger(&client, 0);
    await_until(
        "the member must be pointed at a fresh client",
        Instant::now() + SETTLE,
        || ClientLike::id(client.pool.member(0)) != old_id,
    )
    .await;
    // The swap freed the pool's slot; the replacement's task runs on through
    // the old client's retirement.
    await_until(
        "the swap must free the pool's slot",
        Instant::now() + SETTLE,
        || !client.member_replacement_in_flight_for_test(POOL_LABEL_DATA),
    )
    .await;
    assert!(
        client.member_replacement_tasks_for_test() >= 1
            && client.member_retirements_for_test() == 1,
        "precondition: the old client is still in its retirement grace"
    );
    quit_leaves_nothing(&client).await;

    let sent = tokio::time::timeout(
        RELEASED_SEND_BOUND,
        old.get::<Option<String>, _>("zt3:never-written"),
    )
    .await
    .expect("a send through the released old client must not hang");
    let error = sent.expect_err("a send through the released old client must fail");
    assert_ne!(*error.kind(), ErrorKind::Timeout, "{error:?}");
}

/// T13 and T10: a two-member pool has `max(2, 4)` = 4 spare cells. Four
/// replacements use them up; every replacement due after that is refused
/// (`budget_exhausted` counts each one), the members keep their clients, and
/// the WARN is logged once for the pool. Current-thread runtime: the trigger
/// runs on the test's thread, so the capture sees its WARN.
#[tokio::test]
async fn once_the_spare_cells_are_used_up_a_due_replacement_is_refused_counted_each_time_and_warned_once(
) {
    let capture = ThreadCapture::start();
    let client = direct_client(2).await;
    let metrics = armed(&client, knobs(Duration::ZERO, Duration::ZERO)).await;
    let spare_cells = fred::live_edit::pool::spare_cells(&client.pool);
    assert_eq!(spare_cells, 4);

    for n in 0..spare_cells {
        replace(&client, n % 2).await;
    }
    assert_eq!(fred::live_edit::pool::spare_cells_left(&client.pool), 0);
    let ids: Vec<String> = (0..2)
        .map(|m| ClientLike::id(client.pool.member(m)).to_owned())
        .collect();
    // The verdict that makes one due, then one more: each is refused.
    trigger(&client, 0);
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED),
        1
    );
    client.fast_verdict_for_test(POOL_LABEL_DATA, 0);
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED),
        2,
        "every refused replacement is counted"
    );

    assert!(!client.member_replacement_in_flight_for_test(POOL_LABEL_DATA));
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        0,
        "a refused replacement starts no task"
    );
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED),
        spare_cells as i64
    );
    for (m, id) in ids.iter().enumerate() {
        assert_eq!(
            ClientLike::id(client.pool.member(m)),
            id,
            "member {m} keeps its client"
        );
    }
    let captured = capture.snapshot();
    assert_eq!(
        messages_containing(&captured, BUDGET_WARN).len(),
        1,
        "one WARN per pool; full capture:\n{}",
        captured.text
    );
    quit_leaves_nothing(&client).await;
}

/// T13: a fast verdict on a member's fresh client inside the cooldown after
/// its replacement logs the "did not cure it" WARN, naming the replaced and
/// the fresh client, and starts no second replacement. Current-thread
/// runtime, as above.
#[tokio::test]
async fn a_fast_verdict_inside_the_cooldown_logs_that_the_replacement_did_not_cure_it() {
    let capture = ThreadCapture::start();
    let client = direct_client(2).await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, Duration::ZERO)).await;
    let old = ClientLike::id(client.pool.member(0)).to_owned();
    let fresh = replace(&client, 0).await;

    trigger(&client, 0);

    let captured = capture.snapshot();
    let warns: Vec<&serde_json::Value> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(NOT_CURED_WARN))
        })
        .collect();
    assert_eq!(
        warns.len(),
        MEMBER_REPLACE_AFTER_FAST_VERDICTS,
        "one WARN per fast verdict inside the cooldown; full capture:\n{}",
        captured.text
    );
    for warn in warns {
        assert_eq!(warn["level"], "WARN");
        assert_eq!(warn["fields"]["previous_client_id"], old.as_str());
        assert_eq!(warn["fields"]["client_id"], fresh.as_str());
        assert_eq!(warn["fields"]["pool"], POOL_LABEL_DATA);
    }
    assert!(!client.member_replacement_in_flight_for_test(POOL_LABEL_DATA));
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        0,
        "a verdict inside the cooldown starts no task"
    );
    assert_eq!(
        ClientLike::id(client.pool.member(0)),
        fresh,
        "no second replacement"
    );
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED),
        1
    );
    quit_leaves_nothing(&client).await;
}

/// T13: a retired client whose `QUIT` overran has its router aborted, and
/// `fred::live_edit::pool::retire` then releases what fred releases only on a
/// completed `QUIT` or a dropped client: its credential-refresh task, its
/// backchannel connection, and its command channel (a later send fails at
/// once). Staged on a fred client of its own with a ticking credential
/// provider (so the refresh task exists) and `Blocking::Interrupt` (a command
/// sent behind a blocked `BLPOP` interrupts it over the backchannel, so the
/// backchannel is connected).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retired_client_whose_router_was_aborted_keeps_no_credential_task_backchannel_or_open_command_channel(
) {
    let url = crate::common::test_config().valkey_url;
    let mut config = FredConfig::from_url(&url).expect("TEST_VALKEY_URL must parse");
    config.blocking = Blocking::Interrupt;
    let provider = ScriptedProvider::new("zt3-password");
    provider.start_ticking();
    config.credential_provider = Some(provider);
    let client = Builder::from_config(config)
        .build()
        .expect("the client must build");
    let router = client.connect();
    client
        .wait_for_connect()
        .await
        .expect("the client must connect");
    let list = format!("zt3:blpop:{}", uuid::Uuid::new_v4());
    let blocked = {
        let client = client.clone();
        let list = list.clone();
        tokio::spawn(async move { client.blpop::<Option<Vec<String>>, _>(list, 5.0).await })
    };
    await_until(
        "the BLPOP must be marked blocked",
        Instant::now() + SETTLE,
        || client.inner().backchannel.blocked.lock().is_some(),
    )
    .await;
    let _: Option<String> = client
        .get(&list)
        .await
        .expect("a command behind the blocked BLPOP interrupts it and is answered");
    // The BLPOP's own outcome (unblocked) is not this test's subject.
    let _ = blocked.await;
    assert!(
        client.inner().credentials_task.read().is_some(),
        "precondition: the credential-refresh task runs"
    );
    assert!(
        client.inner().backchannel.transport.read().await.is_some(),
        "precondition: the backchannel is connected"
    );

    router.abort();
    fred::live_edit::pool::retire(&client, Some(router.abort_handle()));
    let deadline = Instant::now() + SETTLE;
    loop {
        let released = client.inner().credentials_task.read().is_none()
            && client.inner().backchannel.transport.read().await.is_none();
        if released {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "retire must stop the credential task and close the backchannel"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let sent = tokio::time::timeout(RELEASED_SEND_BOUND, client.get::<Option<String>, _>(&list))
        .await
        .expect("a send through the retired client must not hang");
    let error = sent.expect_err("a send through the retired client must fail");
    assert_ne!(*error.kind(), ErrorKind::Timeout, "{error:?}");
}

/// T15: after member 0 was replaced, `end_router_for_test("data", 0)` ends
/// the fresh client's router, the one the member's watch now supervises:
/// it is restarted and counted, and the member keeps serving on the fresh
/// client. (Ending the retired client's router instead would count nothing.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ending_a_replaced_members_router_restarts_the_fresh_clients_router() {
    let client = direct_client(2).await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, Duration::ZERO)).await;
    let fresh = replace(&client, 0).await;
    let restarts = || read_series(&metrics, ROUTER_RESTARTS, &[("pool", VALKEY_POOL_DATA)]);
    assert_eq!(restarts(), 0);

    assert!(client.end_router_for_test(POOL_LABEL_DATA, 0));
    await_until(
        "the fresh client's ended router must be restarted",
        Instant::now() + SETTLE,
        || restarts() == 1,
    )
    .await;

    assert_eq!(ClientLike::id(client.pool.member(0)), fresh);
    assert_eq!(
        client.member_watches_for_test(POOL_LABEL_DATA).get(&0),
        Some(&fresh)
    );
    let _: Option<String> = client
        .pool
        .member(0)
        .get("zt3:never-written")
        .await
        .expect("the member must serve on its restarted fresh client");
    quit_leaves_nothing(&client).await;
}

// ══ b.zt3 postcomp-1 — scheduling and the retiring cap ═════════════════════
//
// Real store, real time. The design's paused-clock end-to-end tests cannot be
// written: tokio's paused clock fires fred's connect and command timeouts
// during real socket round trips. The scheduling decisions themselves are
// pinned on a paused clock in `storage::member_replacement`'s unit tests;
// these pin what they add up to, against bounds derived from the client's
// connect timeout `C` ([`CONNECT_TIMEOUT`]) with [`SCHEDULING_SLACK`] on top,
// so a loaded runner does not turn them red. A fresh client's connect is held
// where a case needs it by a proxy that holds the replies on the connection of
// the next `PING` (fred's connection setup sends one; nothing else in these
// tests does) until the test opens its gate.

/// `C`: the connect timeout (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`) the
/// scheduling tests set on their client, the per-replacement term of every
/// scheduling bound.
const CONNECT_TIMEOUT_SECS: u64 = 5;

/// [`CONNECT_TIMEOUT_SECS`] as a `Duration`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(CONNECT_TIMEOUT_SECS);

/// Slack on top of `k × C` in every scheduling bound: a healthy run replaces
/// each member in milliseconds, so the bound is a deadline a hang would blow,
/// never a latency measurement.
const SCHEDULING_SLACK: Duration = Duration::from_secs(15);

/// A grace no test outlives: every retirement started stays in progress, inside
/// its grace, until the cap or `quit()` ends it.
const LONG_GRACE: Duration = Duration::from_secs(300);

/// The bound on a send through a client whose retirement the cap ended: its
/// router is aborted at once and its command channel closed once that router
/// has finished, on a detached task. Below fred's 5 s command timeout, so a
/// send left to time out still fails the test.
const EVICTED_SEND_BOUND: Duration = Duration::from_secs(4);

/// The start of the WARN of a replacement's swap.
const REPLACED_WARN: &str = "replaced a pool member that kept failing with a fresh Valkey client";

/// The start of the WARN of a retirement the cap ends early.
const EVICTED_WARN: &str = "ending a retired Valkey client's retirement early";

/// Count enough fast verdicts on member `member` of `pool`'s current client to
/// trigger a replacement.
fn trigger_in(client: &ValkeyClient, pool: &str, member: usize) {
    for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
        client.fast_verdict_for_test(pool, member);
    }
}

/// Each data member's current client id.
fn data_ids(client: &ValkeyClient, members: usize) -> Vec<String> {
    (0..members)
        .map(|m| ClientLike::id(client.pool.member(m)).to_owned())
        .collect()
}

/// A proxy that, once [`hold_next_connect`] has armed it, holds every reply on
/// the connection that sends the next `PING`, until the armed gate opens: a
/// fresh client's connect is then in flight, its setup `PING` sent and run,
/// and nothing answered.
async fn ping_holding_proxy(hold: &Arc<Mutex<Option<ReplyGate>>>) -> ScriptReplyProxy {
    let rule: ProxyRule = {
        let hold = Arc::clone(hold);
        Arc::new(move |args: &[Vec<u8>]| {
            if !args.first()?.eq_ignore_ascii_case(b"PING") {
                return None;
            }
            let gate = hold.lock().expect("the hold lock is not poisoned").take()?;
            Some(ProxyAction::HoldReplies(gate))
        })
    };
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// Arm the proxy to hold the next fresh client's connect; returns the gate
/// that releases it.
fn hold_next_connect(hold: &Arc<Mutex<Option<ReplyGate>>>) -> ReplyGate {
    let gate = ReplyGate::new();
    *hold.lock().expect("the hold lock is not poisoned") = Some(gate.clone());
    gate
}

/// A client through `proxy` with `data_members` data members and a connect
/// timeout of `connect_timeout_secs`.
async fn scheduling_client(
    proxy: &ScriptReplyProxy,
    data_members: u32,
    connect_timeout_secs: u64,
) -> ValkeyClient {
    proxied_client_with(
        proxy,
        data_members,
        LivenessOverrides::default(),
        |config| config.valkey_connect_timeout_secs = connect_timeout_secs,
    )
    .await
}

/// The `member` field of every captured swap WARN, in the order logged.
fn replaced_members(capture: &ThreadCapture) -> Vec<u64> {
    capture
        .snapshot()
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(REPLACED_WARN))
        })
        .map(|e| {
            e["fields"]["member"]
                .as_u64()
                .expect("the swap WARN names its member")
        })
        .collect()
}

/// A command through `old` — a client whose retirement is still inside its
/// grace — is served, by `deadline` (one deadline shared by every such check
/// in a test, so their worst case is one [`SETTLE`], not one each).
async fn serves(old: &Client, what: &str, deadline: Instant) {
    tokio::time::timeout_at(deadline, old.get::<Option<String>, _>("zt3:never-written"))
        .await
        .unwrap_or_else(|_| panic!("{what}: a GET must not hang"))
        .unwrap_or_else(|e| panic!("{what}: a client still retiring serves: {e:?}"));
}

/// A command through `old` — a client whose retirement the cap ended — fails
/// at once rather than being served or left to the command timeout.
async fn refuses_at_once(old: &Client, what: &str) {
    let sent = tokio::time::timeout(
        EVICTED_SEND_BOUND,
        old.get::<Option<String>, _>("zt3:never-written"),
    )
    .await
    .unwrap_or_else(|_| panic!("{what}: a send must not hang"));
    let error = sent.expect_err(&format!("{what}: a send through it must fail"));
    assert_ne!(*error.kind(), ErrorKind::Timeout, "{what}: {error:?}");
}

/// T-R1 (k = 2): a member that comes due while another member's replacement
/// holds the pool's slot (its fresh client's connect held at the proxy) is
/// not started then, and is replaced as soon as that replacement's swap frees
/// the slot — within 2 × C of the release, with no further verdict — while
/// the first member's old client is still retiring: the second replacement
/// waits for a connect, never for a retirement.
///
/// The post-swap reads (two retirements, two `replaced` ticks) are part of the
/// wait condition: on a multi-thread runtime the swap is visible before its
/// task registers the retirement and counts `replaced`.
///
/// Fail-on-old, by in-place perturbation: deleting the re-check in
/// `MemberReplacer::release_slot` leaves member 1 unreplaced; deleting the
/// multiplexed swap's `guard.release_slot()`, so the slot is released only
/// when the retirement ends, makes member 1 wait out member 0's 300 s grace.
/// Either turns the 2 × C wait red.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_due_while_another_replacement_connects_is_replaced_at_its_swap_without_another_verdict(
) {
    let hold = Arc::new(Mutex::new(None));
    let proxy = ping_holding_proxy(&hold).await;
    let client = scheduling_client(&proxy, 2, CONNECT_TIMEOUT_SECS).await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;
    let ids = data_ids(&client, 2);

    let gate = hold_next_connect(&hold);
    trigger(&client, 0);
    gate.reply_held().await;
    trigger(&client, 1);
    assert!(
        client.member_replacement_in_flight_for_test(POOL_LABEL_DATA),
        "member 0's replacement holds the pool's slot while its fresh client connects"
    );
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        1,
        "member 1, due while the slot is held, is not started"
    );
    assert_eq!(data_ids(&client, 2), ids, "neither member is replaced yet");

    let released = Instant::now();
    gate.open();
    // Two retirements at once: member 0's old client is still retiring, so
    // member 1's replacement did not wait for it.
    await_until(
        "member 1 must be replaced within 2 × C of the slot's release, with no further verdict, \
         while member 0's old client still retires",
        released + CONNECT_TIMEOUT * 2 + SCHEDULING_SLACK,
        || {
            ClientLike::id(client.pool.member(1)) != ids[1]
                && client.member_retirements_for_test() == 2
                && replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED) == 2
        },
    )
    .await;

    assert_ne!(ClientLike::id(client.pool.member(0)), ids[0]);
    assert!(client.member_replacement_tasks_for_test() >= 1);
    quit_leaves_nothing(&client).await;
}

/// T-R2 (k = 4): members 1–3 come due while member 0's replacement holds the
/// pool's slot. Once its connect is released they are replaced one at a time,
/// in index order (read off the swap WARNs), the fourth within 4 × C; all four
/// old clients then retire side by side (four tasks, four retirements), and
/// `quit()` cuts all four short within its bound. Current-thread runtime, so
/// the capture sees the replacement tasks' WARNs.
#[tokio::test]
async fn four_members_due_at_once_are_replaced_in_index_order_and_retire_side_by_side() {
    let capture = ThreadCapture::start();
    let hold = Arc::new(Mutex::new(None));
    let proxy = ping_holding_proxy(&hold).await;
    let client = scheduling_client(&proxy, 4, CONNECT_TIMEOUT_SECS).await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;
    let ids = data_ids(&client, 4);

    let gate = hold_next_connect(&hold);
    trigger(&client, 0);
    gate.reply_held().await;
    for member in 1..4 {
        trigger(&client, member);
    }
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        1,
        "members 1-3, due while the slot is held, are not started"
    );

    let released = Instant::now();
    gate.open();
    await_until(
        "all four members must be replaced within 4 × C of the slot's release",
        released + CONNECT_TIMEOUT * 4 + SCHEDULING_SLACK,
        || {
            data_ids(&client, 4)
                .iter()
                .zip(&ids)
                .all(|(now, before)| now != before)
        },
    )
    .await;

    assert_eq!(
        replaced_members(&capture),
        [0, 1, 2, 3],
        "replaced one at a time, lowest index first"
    );
    assert_eq!(
        (
            client.member_replacement_tasks_for_test(),
            client.member_retirements_for_test()
        ),
        (4, 4),
        "all four old clients retire at once"
    );
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED),
        4
    );
    quit_leaves_nothing(&client).await;
}

/// T-R4 (I-A2): only the replacement holding the pool's slot releases it.
/// Member 0 is replaced while its old client has a reply held in flight (a
/// `GET` the proxy holds), so that client's retirement stays open (its grace
/// is zero here): its `QUIT` drains the replies still owed first. Member 1's
/// replacement then takes the slot, its connect held at the proxy. When the
/// held reply is let through, member 0's retirement ends — and the slot still
/// reads held: that retirement's end does not clear a flag a later replacement
/// now holds.
///
/// **What limits the hold of member 1's connect.** First the replacer's own
/// connect timeout, `C` (`MemberReplacer`'s `connect_timeout`, from
/// `valkey_connect_timeout_secs`); then fred's 10 s internal command timeout,
/// which bounds a connect's setup `PING` whatever `C` is — and is why a long
/// grace cannot stand in for the held reply. This test sets `C` to
/// `HELD_CONNECT_TIMEOUT_SECS` = 10 s, so both limits fall at 10 s after
/// the `PING` was held, and bounds the wait for member 0's retirement to end
/// at `RETIREMENT_END_BOUND` = 5 s from that same moment: with a zero grace
/// the retirement ends milliseconds after the gate opens, and a run slow
/// enough to miss 5 s fails on that wait, never as a slot-ownership finding.
/// Member 1's connect is also checked not to have failed before the slot is
/// read. The `txn` twin, with no connect to time out, is
/// `b_zt3_txn_replacement_test::a_txn_retirement_that_ends_while_another_slot_replacement_holds_the_pool_leaves_it_held`.
///
/// Fail-on-old, by in-place perturbation: making the `Replacement` guard's
/// `Drop` clear the pool's flag unconditionally (not only while it holds the
/// slot) turns the "the slot must still read held" assertion red.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retirement_that_ends_while_a_later_replacement_holds_the_slot_leaves_the_slot_held() {
    /// `C` for this test: equal to fred's 10 s internal command timeout, the
    /// other limit on a held connect, so neither cuts it before 10 s.
    const HELD_CONNECT_TIMEOUT_SECS: u64 = 10;
    /// How long after member 1's `PING` was held member 0's retirement may
    /// take to end: half of `HELD_CONNECT_TIMEOUT_SECS`.
    const RETIREMENT_END_BOUND: Duration = Duration::from_secs(5);
    /// The fast and slow unresponsive bounds and the command timeout of this
    /// test's client: past anything it waits, so the held reply is neither
    /// judged nor timed out while the test holds it.
    const HELD_REPLY_BOUND: Duration = Duration::from_secs(90);
    let marker = format!("zt3:held-reply:{}", uuid::Uuid::new_v4());
    let reply_gate = ReplyGate::new();
    let hold = Arc::new(Mutex::new(None));
    let rule: ProxyRule = {
        let marker = marker.clone().into_bytes();
        let reply_gate = reply_gate.clone();
        let hold = Arc::clone(&hold);
        Arc::new(move |args: &[Vec<u8>]| {
            let command = args.first()?;
            if command.eq_ignore_ascii_case(b"GET") && *args.get(1)? == marker {
                return Some(ProxyAction::HoldReplies(reply_gate.clone()));
            }
            if !command.eq_ignore_ascii_case(b"PING") {
                return None;
            }
            let gate = hold.lock().expect("the hold lock is not poisoned").take()?;
            Some(ProxyAction::HoldReplies(gate))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client_with(
        &proxy,
        2,
        LivenessOverrides {
            fast_timeout: Some(HELD_REPLY_BOUND),
            slow_timeout: Some(HELD_REPLY_BOUND),
            ..LivenessOverrides::default()
        },
        |config| {
            config.valkey_connect_timeout_secs = HELD_CONNECT_TIMEOUT_SECS;
            config.valkey_command_timeout_ms = HELD_REPLY_BOUND.as_millis() as u64;
        },
    )
    .await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, Duration::ZERO)).await;
    let ids = data_ids(&client, 2);
    {
        let old = client.pool.member(0).clone();
        let marker = marker.clone();
        // Its outcome is not the subject: it ends when the gate opens.
        tokio::spawn(async move {
            let _: Result<Option<String>, _> = old.get(&marker).await;
        });
    }
    reply_gate.reply_held().await;

    trigger(&client, 0);
    await_until(
        "member 0 must be replaced, its slot freed, its old client retiring",
        Instant::now() + SETTLE,
        || {
            ClientLike::id(client.pool.member(0)) != ids[0]
                && !client.member_replacement_in_flight_for_test(POOL_LABEL_DATA)
                && client.member_retirements_for_test() == 1
        },
    )
    .await;
    let connect_gate = hold_next_connect(&hold);
    trigger(&client, 1);
    connect_gate.reply_held().await;
    let connect_held_at = Instant::now();
    assert_eq!(
        (
            client.member_replacement_in_flight_for_test(POOL_LABEL_DATA),
            client.member_replacement_tasks_for_test()
        ),
        (true, 2),
        "precondition: member 1's replacement holds the slot while member 0's old client is \
         still retiring"
    );

    reply_gate.open();
    await_until(
        "member 0's retirement must end once its held reply is through, well before member 1's \
         held connect can time out",
        connect_held_at + RETIREMENT_END_BOUND,
        || client.member_replacement_tasks_for_test() == 1,
    )
    .await;
    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED),
        0,
        "member 1's held connect must still be pending, or the slot read below proves nothing"
    );
    assert!(
        client.member_replacement_in_flight_for_test(POOL_LABEL_DATA),
        "the slot must still read held: only the replacement holding it releases it"
    );

    connect_gate.open();
    await_until(
        "member 1 must be replaced once its connect is released",
        Instant::now() + SETTLE,
        || ClientLike::id(client.pool.member(1)) != ids[1],
    )
    .await;
    quit_leaves_nothing(&client).await;
}

/// Rider A: six members of one pool due at once are all replaced, and never
/// more than [`MEMBER_RETIRING_CAP`] old clients retire at once: the fifth and
/// sixth swaps each end the oldest retirement early (one WARN each, naming
/// members 0's and 1's old clients), whose clients then refuse sends at once,
/// while the four retiring old clients still serve inside their grace.
/// Current-thread runtime, so the capture sees the replacement tasks' WARNs,
/// and the count is sampled between the replacement tasks' steps, never
/// inside one.
///
/// Fail-on-old, by in-place perturbation: removing the eviction from
/// `MemberReplacer::begin_retirement` lets the count reach six, turning the
/// "at most 4 old clients may retire at once" assertion red.
#[tokio::test]
async fn six_members_due_at_once_are_all_replaced_and_never_more_than_the_cap_retire_at_once() {
    const MEMBERS: usize = 6;
    let capture = ThreadCapture::start();
    let client = direct_client_with(|config| {
        config.valkey_pool_size = MEMBERS as u32;
        config.valkey_connect_timeout_secs = CONNECT_TIMEOUT_SECS;
    })
    .await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;
    let olds: Vec<Client> = (0..MEMBERS)
        .map(|m| client.pool.member(m).clone())
        .collect();
    let old_ids = data_ids(&client, MEMBERS);

    for member in 0..MEMBERS {
        trigger(&client, member);
    }
    let deadline = Instant::now() + CONNECT_TIMEOUT * MEMBERS as u32 + SCHEDULING_SLACK;
    let mut most_retiring = 0;
    loop {
        most_retiring = most_retiring.max(client.member_retirements_for_test());
        let all_replaced = data_ids(&client, MEMBERS)
            .iter()
            .zip(&old_ids)
            .all(|(now, before)| now != before);
        if all_replaced && !client.member_replacement_in_flight_for_test(POOL_LABEL_DATA) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "all six members must be replaced within 6 × C"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    most_retiring = most_retiring.max(client.member_retirements_for_test());
    assert!(
        most_retiring <= MEMBER_RETIRING_CAP,
        "at most {MEMBER_RETIRING_CAP} old clients may retire at once; saw {most_retiring}"
    );
    assert_eq!(client.member_retirements_for_test(), MEMBER_RETIRING_CAP);
    await_until(
        "the two retirements the cap ended must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == MEMBER_RETIRING_CAP,
    )
    .await;

    assert_eq!(
        replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED),
        MEMBERS as i64
    );
    let captured = capture.snapshot();
    assert_eq!(
        evicted_client_ids(&captured),
        &old_ids[..2],
        "the oldest retirements are ended, one WARN each; full capture:\n{}",
        captured.text
    );
    let ended = MEMBERS - MEMBER_RETIRING_CAP;
    for (member, old) in olds.iter().enumerate().take(ended) {
        refuses_at_once(old, &format!("member {member}'s old client")).await;
    }
    let serving = Instant::now() + SETTLE;
    for (member, old) in olds.iter().enumerate().skip(ended) {
        serves(old, &format!("member {member}'s old client"), serving).await;
    }
    quit_leaves_nothing(&client).await;
}

/// Rider A's condition: a write in flight on an old client whose retirement
/// the cap ends reaches its caller as an **unknown outcome** — gRPC
/// `INTERNAL`, with no `RefusalDetail` and no `RetryInfo` — never as a refusal
/// that applied nothing. Here the write did apply: its `SET` ran on the store
/// and only its reply was held (at the proxy), so a "nothing was applied" hint
/// would be false.
///
/// What it pins, and how far that reaches: the real error the eviction's
/// router abort gives the caller — fred's `Unknown` for a response channel
/// dropped unanswered — and the shared classifiers the write paths route a
/// store error through. `store_write_refusal` is the only source of the
/// applied-nothing hint on a store error, and it does not recognise this
/// error, so its `refuse_store_write` wrapper (crate-private) answers `None`
/// for it at every stage; the error is not transient, so both public mappers,
/// the in-loop `map_transient_fred_err` and the direct
/// `map_fred_err_for_grpc`, answer `Internal`, which reaches the wire as
/// `INTERNAL` with no status details. `UpdateLayer`'s script-error mapping
/// (`map_update_script_error` in `storage::layer`, crate-private and not
/// called here) matches only its own reply tags and the store refusals before
/// falling to the in-loop mapper, so it takes the same path. A write path with
/// a classification of its own beyond these is not covered.
///
/// The connection holding the write must not be closed by an unresponsive
/// verdict before the cap ends its client (a re-sent `SET` would then be
/// answered), nor the write timed out, so the fast and slow bounds and the
/// command timeout are raised past the write's longest time in flight; every
/// later command of the test is bounded on its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_in_flight_on_a_retirement_the_cap_ends_reaches_its_caller_as_outcome_unknown() {
    /// One old client to be evicted, and the four swaps after it that fill
    /// the cap and pass it.
    const MEMBERS: usize = MEMBER_RETIRING_CAP + 1;
    /// The fast and slow unresponsive bounds and the command timeout of this
    /// test's client: strictly longer than the held write can be in flight —
    /// the proxy's report that it ran (`reply_held`, 20 s) plus the write wait
    /// (5 × C + 15 s = 40 s) — so none fires on it before the cap ends its
    /// client.
    const HELD_WRITE_BOUND: Duration = Duration::from_secs(90);
    let tv = TestValkey::setup().await;
    let marker = format!("{}zt3:evicted-write", tv.prefix());
    let gate = ReplyGate::new();
    let rule: ProxyRule = {
        let marker = marker.clone().into_bytes();
        let gate = gate.clone();
        Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"SET") && *args.get(1)? == marker)
                .then(|| ProxyAction::HoldReplies(gate.clone()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client_with(
        &proxy,
        MEMBERS as u32,
        LivenessOverrides {
            fast_timeout: Some(HELD_WRITE_BOUND),
            slow_timeout: Some(HELD_WRITE_BOUND),
            ..LivenessOverrides::default()
        },
        |config| {
            config.valkey_command_timeout_ms = HELD_WRITE_BOUND.as_millis() as u64;
            config.valkey_connect_timeout_secs = CONNECT_TIMEOUT_SECS;
        },
    )
    .await;
    let _metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;

    let write = {
        let old = client.pool.member(0).clone();
        let marker = marker.clone();
        tokio::spawn(async move {
            old.set::<(), _, _>(&marker, "applied", None, None, false)
                .await
        })
    };
    gate.reply_held().await;
    for member in 0..MEMBERS {
        trigger(&client, member);
    }
    let error = tokio::time::timeout(CONNECT_TIMEOUT * MEMBERS as u32 + SCHEDULING_SLACK, write)
        .await
        .expect("the write must end once the cap ends its client's retirement")
        .expect("the write task does not panic")
        .expect_err("the write in flight on the evicted client must fail");

    assert_eq!(
        client.member_retirements_for_test(),
        MEMBER_RETIRING_CAP,
        "the fifth swap ended the oldest retirement, the one holding the write"
    );
    assert_eq!(
        *error.kind(),
        ErrorKind::Unknown,
        "fred fails a command its aborted router dropped with `Unknown` (its response \
         channel closed): {error:?}"
    );
    assert_eq!(
        store_write_refusal(&error),
        None,
        "not a store write refusal, so no write path answers it with the applied-nothing hint"
    );
    assert!(!is_transient_fred_error(&error), "{error:?}");
    for (mapper, mapped) in [
        (
            "map_fred_err_for_grpc",
            map_fred_err_for_grpc(
                FredErrOrigin::metrics(
                    None,
                    POOL_LABEL_DATA,
                    live_edit::storage::store_oom::StoreOomSignal::unwired(),
                ),
                "SET",
                error.clone(),
            ),
        ),
        (
            "map_transient_fred_err",
            map_transient_fred_err(
                FredErrOrigin::metrics(
                    None,
                    POOL_LABEL_DATA,
                    live_edit::storage::store_oom::StoreOomSignal::unwired(),
                ),
                "SET",
                error.clone(),
            ),
        ),
    ] {
        assert!(
            matches!(mapped, ServiceError::Internal(_)),
            "{mapper}: an unknown outcome, never a refusal: {mapped:?}"
        );
        let status = tonic::Status::from(mapped);
        assert_eq!(status.code(), tonic::Code::Internal, "{mapper}");
        assert!(
            status.details().is_empty(),
            "{mapper}: no RefusalDetail or RetryInfo on the status"
        );
    }
    // Bounded here: this client's command timeout is `HELD_WRITE_BOUND`.
    let landed: Option<String> = tokio::time::timeout(SETTLE, client.pool.member(0).get(&marker))
        .await
        .expect("the read through the fresh member must not hang")
        .expect("the fresh member serves");
    assert_eq!(
        landed.as_deref(),
        Some("applied"),
        "the write did apply: \"nothing was applied\" would have been false"
    );
    gate.open();
    quit_leaves_nothing(&client).await;
}

/// Rider A, no pool exempt, end to end: `maintenance_members` `maintenance`
/// old clients retire first — the oldest retirements, each inside its grace
/// ([`LONG_GRACE`] outlives the test) — and data old clients fill the cap
/// behind them; then `maintenance_members` more data swaps pass the cap. Each
/// of those ends the oldest retirement left, a `maintenance` one inside its
/// grace, exactly as it would end any other: one eviction WARN names each
/// `maintenance` old client, oldest first; each of them then refuses sends at
/// once; every data old client still serves; and the count is never above the
/// cap. Current-thread runtime, so the capture sees the replacement tasks'
/// WARNs and the count is sampled between the replacement tasks' steps, never
/// inside one (a swap registers its retirement in the same step).
///
/// Fail-on-old, by in-place perturbation: making `Retiring::evict_over_cap`
/// pass over `maintenance` entries (the in-grace exemption the cap once had)
/// ends the oldest data retirements instead — or, with no data retirement
/// older than the swap's own, nothing, so the count passes the cap — turning
/// the WARN-order assertion red (the "at most 4" one first, where the count
/// passes the cap).
async fn maintenance_retirements_in_their_grace_are_ended_by_the_cap_like_any_other(
    maintenance_members: usize,
) {
    let data_members = MEMBER_RETIRING_CAP;
    let filling = MEMBER_RETIRING_CAP - maintenance_members;
    let capture = ThreadCapture::start();
    let client = direct_client_with(|config| {
        config.valkey_pool_size = data_members as u32;
        config.maintenance_pool_size = maintenance_members as u32;
        config.valkey_connect_timeout_secs = CONNECT_TIMEOUT_SECS;
    })
    .await;
    let _metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;
    let maintenance_olds: Vec<Client> = (0..maintenance_members)
        .map(|m| client.maintenance_pool.member(m).clone())
        .collect();
    let data_olds: Vec<Client> = (0..data_members)
        .map(|m| client.pool.member(m).clone())
        .collect();
    let replaced = |pool: &fred::clients::Pool, olds: &[Client], upto: usize| {
        (0..upto).all(|m| ClientLike::id(pool.member(m)) != ClientLike::id(&olds[m]))
    };

    for member in 0..maintenance_members {
        trigger_in(&client, POOL_LABEL_MAINTENANCE, member);
    }
    await_until(
        "every maintenance member must be replaced, its retirement registered",
        Instant::now() + CONNECT_TIMEOUT * maintenance_members as u32 + SCHEDULING_SLACK,
        || {
            replaced(
                &client.maintenance_pool,
                &maintenance_olds,
                maintenance_members,
            ) && client.member_retirements_for_test() == maintenance_members
        },
    )
    .await;
    for member in 0..filling {
        trigger(&client, member);
    }
    await_until(
        "the data members filling the cap must be replaced, the cap reached with nothing ended",
        Instant::now() + CONNECT_TIMEOUT * filling as u32 + SCHEDULING_SLACK,
        || {
            replaced(&client.pool, &data_olds, filling)
                && client.member_retirements_for_test() == MEMBER_RETIRING_CAP
        },
    )
    .await;
    assert!(
        evicted_client_ids(&capture.snapshot()).is_empty(),
        "at the cap nothing is ended"
    );

    for member in filling..data_members {
        trigger(&client, member);
    }
    let deadline = Instant::now() + CONNECT_TIMEOUT * maintenance_members as u32 + SCHEDULING_SLACK;
    let mut most_retiring = 0;
    loop {
        most_retiring = most_retiring.max(client.member_retirements_for_test());
        if replaced(&client.pool, &data_olds, data_members)
            && !client.member_replacement_in_flight_for_test(POOL_LABEL_DATA)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the swaps past the cap must be made within {maintenance_members} × C"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    most_retiring = most_retiring.max(client.member_retirements_for_test());
    assert!(
        most_retiring <= MEMBER_RETIRING_CAP,
        "at most {MEMBER_RETIRING_CAP} old clients may retire at once, maintenance ones \
         included; saw {most_retiring}"
    );
    assert_eq!(client.member_retirements_for_test(), MEMBER_RETIRING_CAP);
    await_until(
        "the retirements the cap ended must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == MEMBER_RETIRING_CAP,
    )
    .await;

    let captured = capture.snapshot();
    let maintenance_old_ids: Vec<String> = maintenance_olds
        .iter()
        .map(|old| ClientLike::id(old).to_owned())
        .collect();
    assert_eq!(
        evicted_client_ids(&captured),
        maintenance_old_ids,
        "each maintenance retirement, in its grace, is ended in turn, oldest first, one \
         WARN each; full capture:\n{}",
        captured.text
    );
    for (member, old) in maintenance_olds.iter().enumerate() {
        refuses_at_once(
            old,
            &format!("maintenance member {member}'s old client, ended inside its grace"),
        )
        .await;
    }
    let serving = Instant::now() + SETTLE;
    for (member, old) in data_olds.iter().enumerate() {
        serves(old, &format!("data member {member}'s old client"), serving).await;
    }
    quit_leaves_nothing(&client).await;
}

/// Rider A, no pool exempt: a lone `maintenance` retirement in its grace,
/// the oldest at the cap, is the one the next swap ends.
#[tokio::test]
async fn a_lone_maintenance_retirement_in_its_grace_is_ended_by_the_cap_like_any_other() {
    maintenance_retirements_in_their_grace_are_ended_by_the_cap_like_any_other(1).await;
}

/// Rider A, no pool exempt: a `maintenance` pool of two, both retiring in
/// their grace; the next two swaps end them, older first.
#[tokio::test]
async fn two_maintenance_retirements_in_their_grace_are_ended_by_the_cap_oldest_first() {
    maintenance_retirements_in_their_grace_are_ended_by_the_cap_like_any_other(2).await;
}

/// Rider A bound, no pool exempt: four `maintenance` retirements in their
/// grace (a `maintenance` pool of four) fill the cap with no older retirement
/// of another pool; each of the next four data swaps ends one at once, so
/// five old clients never retire at once.
#[tokio::test]
async fn four_maintenance_retirements_in_their_grace_never_hold_the_count_above_the_cap() {
    maintenance_retirements_in_their_grace_are_ended_by_the_cap_like_any_other(MEMBER_RETIRING_CAP)
        .await;
}

/// The `client_id` of every captured eviction WARN ([`EVICTED_WARN`]), in the
/// order logged.
fn evicted_client_ids(captured: &Captured) -> Vec<String> {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(EVICTED_WARN))
        })
        .map(|e| {
            e["fields"]["client_id"]
                .as_str()
                .expect("the eviction WARN names its client")
                .to_owned()
        })
        .collect()
}

/// A proxy that stalls the next `PING` once `stall_next` is set, and only that
/// one (the flag is cleared as it is taken): the connect that sent it gets no
/// answer, and every later connect is answered.
async fn ping_stalling_once_proxy(stall_next: &Arc<AtomicBool>) -> ScriptReplyProxy {
    let rule: ProxyRule = {
        let stall_next = Arc::clone(stall_next);
        Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"PING")
                && stall_next.swap(false, Ordering::SeqCst))
            .then_some(ProxyAction::Stall)
        })
    };
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// Review 4(iii): a member that comes due while another member's replacement
/// holds the pool's slot, and whose holder then ends **before** its swap (its
/// fresh client's connect fails at the connect timeout), is started when that
/// attempt ends — by `replace()`'s closing re-check — without another verdict.
/// Member 0 keeps its client (`connect_failed` once); member 1 is replaced
/// within 2 × C of the trigger. The post-swap reads are part of the wait
/// condition (multi-thread runtime).
///
/// Fail-on-old, by in-place perturbation: deleting the `guard.release_slot()`
/// that closes `MemberReplacer::replace` leaves the slot to the guard's
/// `Drop`, which frees it but starts nothing, so member 1 is never replaced
/// and the 2 × C wait goes red.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_due_behind_a_replacement_whose_connect_failed_is_started_when_that_attempt_ends()
{
    let stall_next = Arc::new(AtomicBool::new(false));
    let proxy = ping_stalling_once_proxy(&stall_next).await;
    let client = scheduling_client(&proxy, 2, CONNECT_TIMEOUT_SECS).await;
    let metrics = armed(&client, knobs(MEMBER_REPLACE_COOLDOWN, LONG_GRACE)).await;
    let ids = data_ids(&client, 2);

    stall_next.store(true, Ordering::SeqCst);
    let triggered = Instant::now();
    trigger(&client, 0);
    await_until(
        "member 0's fresh connect must be stalled",
        Instant::now() + SETTLE,
        || proxy.hits() == 1,
    )
    .await;
    trigger(&client, 1);
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        1,
        "member 1, due while member 0's replacement holds the slot, is not started"
    );

    await_until(
        "member 0's replacement must fail at its connect timeout and member 1 then be \
         replaced, with no further verdict",
        triggered + CONNECT_TIMEOUT * 2 + SCHEDULING_SLACK,
        || {
            replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED) == 1
                && ClientLike::id(client.pool.member(1)) != ids[1]
                && client.member_retirements_for_test() == 1
                && replacements(&metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED) == 1
        },
    )
    .await;
    assert_eq!(
        ClientLike::id(client.pool.member(0)),
        ids[0],
        "member 0 keeps its client"
    );
    assert_eq!(proxy.hits(), 1, "only member 0's connect was stalled");
    quit_leaves_nothing(&client).await;
}
