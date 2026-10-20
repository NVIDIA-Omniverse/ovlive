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

//! b.zt3 T20 — replacing a `txn` slot's client.
//!
//! A `txn` slot whose client gets its second fast verdict has a fresh client
//! swapped in under the slot's own guard, taken through the checkout like any
//! caller: never while a caller holds the slot. The swap re-keys the slot's id,
//! its router watch and its hook pair to the fresh client; the old client is
//! quit with no grace (no caller can reach it once it left its slot). The
//! checkout's permits and scan are untouched, and the `txn` pool's budget is a
//! count of `max(size, 4)` replacements.
//!
//! The trigger is `ValkeyClient::fast_verdict_for_test`, which counts a fast
//! verdict against a slot's current client as its error hook does. Every test
//! uses a client of its own with the default four slots.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`a_txn_slot_replacement_waits_for_the_slot_to_be_released_then_moves_the_slots_id_watch_and_hooks`] | no swap while the slot is checked out, checked once the replacement holds a permit and waits on the slot; then slot id, slot watch and hook pair under the fresh id and nothing under the old; permits unchanged; `replaced` 1; the old client's `QUIT` counts no router restart; the fresh slot serves |
//! | [`a_router_restart_racing_a_txn_slot_swap_is_skipped_and_the_slot_keeps_the_fresh_client`] | the old client's router ended while the swap waits for the slot: its restart, behind the swap, finds the slot no longer holds that client and is skipped |
//! | [`ending_a_txn_slots_router_right_after_a_swap_restarts_the_fresh_clients_router`] | `end_router_for_test("txn", k)` reaches the fresh client's router |
//! | [`once_the_txn_budget_is_used_up_a_due_slot_replacement_is_refused_counted_and_warned_once`] | four replacements, then `budget_exhausted` and one WARN |
//! | [`a_txn_slot_due_while_another_is_replaced_is_started_at_that_swap_not_after_its_retirement`] | postcomp-1 (test review 4(ii)): a slot's swap frees the pool, and a slot pending behind it starts then, while the swapped-out client's retirement is held open |
//! | [`a_txn_retirement_that_ends_while_another_slot_replacement_holds_the_pool_leaves_it_held`] | postcomp-1 T-R4 (I-A2), deterministic: a retirement's end leaves the pool held by a later replacement |
//!
//! # Wait budget
//!
//! The client's connect (30 s), fred's 5 s command timeout per command, each
//! polled condition ([`SETTLE`]) and the closing `quit()` ([`SETTLE`]). The
//! largest is the budget test: 30 + four replacements (10 × 2 each) + 10 =
//! 120 s, under the 180 s kill deadline `.config/nextest.toml` sets for the
//! `b_zt3_` modules.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientLike, KeysInterface};
use tokio::time::Instant;

use live_edit::observability::metrics::{
    Metrics, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED, VALKEY_MEMBER_REPLACEMENT_REPLACED,
    VALKEY_POOL_TXN,
};
use live_edit::storage::client::{LivenessOverrides, ValkeyClient};
use live_edit::storage::member_replacement::{
    ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
    MEMBER_REPLACE_WINDOW,
};
use live_edit::storage::test_support::connect_test_client;

use super::b_zt3_support::{
    await_until, messages_containing, proxied_client_with, MEMBER_REPLACEMENTS, ROUTER_RESTARTS,
};
use super::mux_test_helpers::read_series;
use super::snapshot_read_support::{
    test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};
use crate::grpc::log_capture::ThreadCapture;

/// Bound on each polled condition and on the closing `quit()`.
const SETTLE: Duration = Duration::from_secs(10);

/// The start of the WARN of a slot router restart that found no slot holding
/// its client.
const RESTART_SKIPPED: &str = "txn slot router restart skipped";

/// The start of the `txn` budget WARN.
const TXN_BUDGET_WARN: &str =
    "a txn slot is due for replacement but the txn pool's replacement budget";

struct Fixture {
    client: ValkeyClient,
    metrics: Arc<Metrics>,
}

impl Fixture {
    /// A client of its own (four `txn` slots) with hooks and the production
    /// trigger, the given cooldown.
    async fn start(cooldown: Duration) -> Self {
        let client = connect_test_client(&crate::common::test_config())
            .await
            .expect("the client must connect");
        Self::over(client, cooldown).await
    }

    /// [`Self::start`] over a client the caller connected.
    async fn over(client: ValkeyClient, cooldown: Duration) -> Self {
        let metrics = Arc::new(crate::common::metrics_scrape_only("zt3"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        client.set_member_replacement_knobs_for_test(ReplaceKnobs {
            after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
            window: MEMBER_REPLACE_WINDOW,
            cooldown,
            grace: Duration::from_secs(1),
        });
        Self { client, metrics }
    }

    fn slot_ids(&self) -> Vec<String> {
        self.client.txn_slot_client_ids_for_test()
    }

    fn txn(&self, name: &str, outcome: Option<&str>) -> i64 {
        match outcome {
            Some(outcome) => read_series(
                &self.metrics,
                name,
                &[("pool", VALKEY_POOL_TXN), ("outcome", outcome)],
            ),
            None => read_series(&self.metrics, name, &[("pool", VALKEY_POOL_TXN)]),
        }
    }

    fn trigger(&self, slot: usize) {
        for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
            self.client.fast_verdict_for_test(VALKEY_POOL_TXN, slot);
        }
    }

    /// Wait until slot `slot` holds a client other than `old` and the
    /// replacement has finished, the old client's retirement included (the
    /// slot is freed at the swap, the task ends after the retirement);
    /// returns the fresh client's id.
    async fn swapped(&self, slot: usize, old: &str) -> String {
        await_until(
            "the slot must hold a fresh client",
            Instant::now() + SETTLE,
            || self.slot_ids()[slot] != old,
        )
        .await;
        await_until(
            "the slot replacement must finish",
            Instant::now() + SETTLE,
            || self.client.member_replacement_tasks_for_test() == 0,
        )
        .await;
        assert!(
            !self
                .client
                .member_replacement_in_flight_for_test(VALKEY_POOL_TXN),
            "a finished replacement holds no slot"
        );
        self.slot_ids()[slot].clone()
    }

    /// Wait until a hook pair is armed on a client no slot held at the start
    /// (`known`): the replacement's fresh client has connected and is
    /// waiting for its slot. Returns that client's id.
    async fn fresh_armed(&self, known: &HashSet<String>) -> String {
        await_until(
            "the fresh client must connect and be hooked",
            Instant::now() + SETTLE,
            || {
                self.client
                    .armed_hook_client_ids_for_test()
                    .iter()
                    .any(|id| !known.contains(id))
            },
        )
        .await;
        self.client
            .armed_hook_client_ids_for_test()
            .into_iter()
            .find(|id| !known.contains(id))
            .expect("checked above")
    }

    /// A command through every slot, the fresh one included.
    async fn every_slot_serves(&self) {
        let mut held = Vec::new();
        for _ in 0..self.client.txn_pool.slots() {
            let checkout = self.client.acquire_txn().await;
            let _: () = checkout.ping(None).await.expect("every slot must serve");
            held.push(checkout);
        }
    }

    async fn quit(&self) {
        tokio::time::timeout(SETTLE, self.client.quit())
            .await
            .expect("quit must settle");
        assert_eq!(self.client.router_watchers_for_test(), 0);
        assert!(self.client.armed_hook_client_ids_for_test().is_empty());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_txn_slot_replacement_waits_for_the_slot_to_be_released_then_moves_the_slots_id_watch_and_hooks(
) {
    let fx = Fixture::start(MEMBER_REPLACE_COOLDOWN).await;
    let permits = fx.client.txn_pool.available_permits();
    let initial: HashSet<String> = fx
        .client
        .armed_hook_client_ids_for_test()
        .into_iter()
        .collect();
    let checkout = fx.client.acquire_txn().await;
    let old = ClientLike::id(&*checkout).to_owned();
    let slot = fx
        .slot_ids()
        .iter()
        .position(|id| *id == old)
        .expect("the checked-out client is a slot's");

    fx.trigger(slot);
    let fresh = fx.fresh_armed(&initial).await;
    assert!(fx
        .client
        .member_replacement_in_flight_for_test(VALKEY_POOL_TXN));
    // The hooks are armed before the replacement reaches the slot. Two
    // permits out — this checkout's and the replacement's, which takes its
    // permit before it waits on the slot's lock — means the replacement is
    // now waiting on the slot this caller holds.
    await_until(
        "the replacement must take a permit and wait on the held slot",
        Instant::now() + SETTLE,
        || fx.client.txn_pool.available_permits() == permits - 2,
    )
    .await;
    assert_eq!(
        fx.slot_ids()[slot],
        old,
        "the fresh client is connected, but the slot is not swapped while a caller holds it"
    );
    drop(checkout);
    let swapped = fx.swapped(slot, &old).await;

    assert_eq!(
        swapped, fresh,
        "the slot holds the fresh client the replacement hooked"
    );
    let watches: HashSet<String> = fx.client.txn_slot_watches_for_test().into_iter().collect();
    assert!(
        watches.contains(&fresh) && !watches.contains(&old),
        "{watches:?}"
    );
    let hooked: HashSet<String> = fx
        .client
        .armed_hook_client_ids_for_test()
        .into_iter()
        .collect();
    assert!(
        hooked.contains(&fresh) && !hooked.contains(&old),
        "{hooked:?}"
    );
    assert_eq!(
        fx.client.txn_pool.available_permits(),
        permits,
        "the permit count is unchanged"
    );
    assert_eq!(
        fx.txn(
            MEMBER_REPLACEMENTS,
            Some(VALKEY_MEMBER_REPLACEMENT_REPLACED)
        ),
        1
    );
    assert_eq!(
        fx.txn(ROUTER_RESTARTS, None),
        0,
        "the old client's QUIT ends its router without counting a restart"
    );
    fx.every_slot_serves().await;
    fx.quit().await;
}

/// The old client's router ends (`end_router_for_test`) while the swap waits
/// for the slot. Its restart takes the slot by the old client's id; by the
/// time it has the slot, the swap has put the fresh client there, so it is
/// skipped (one WARN naming the old id) rather than repairing the fresh
/// client. Current-thread runtime, so the capture sees the restart task's
/// WARN.
#[tokio::test]
async fn a_router_restart_racing_a_txn_slot_swap_is_skipped_and_the_slot_keeps_the_fresh_client() {
    let capture = ThreadCapture::start();
    let fx = Fixture::start(MEMBER_REPLACE_COOLDOWN).await;
    let initial: HashSet<String> = fx
        .client
        .armed_hook_client_ids_for_test()
        .into_iter()
        .collect();
    let checkout = fx.client.acquire_txn().await;
    let old = ClientLike::id(&*checkout).to_owned();
    let slot = fx
        .slot_ids()
        .iter()
        .position(|id| *id == old)
        .expect("a slot's client");

    fx.trigger(slot);
    let fresh = fx.fresh_armed(&initial).await;
    assert!(fx.client.end_router_for_test(VALKEY_POOL_TXN, slot));
    await_until(
        "the ended router's restart must be under way",
        Instant::now() + SETTLE,
        || fx.txn(ROUTER_RESTARTS, None) == 1,
    )
    .await;
    drop(checkout);
    assert_eq!(fx.swapped(slot, &old).await, fresh);
    await_until(
        "the restart must be skipped",
        Instant::now() + SETTLE,
        || !messages_containing(&capture.snapshot(), RESTART_SKIPPED).is_empty(),
    )
    .await;

    let captured = capture.snapshot();
    let skipped: Vec<&serde_json::Value> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(RESTART_SKIPPED))
        })
        .collect();
    assert_eq!(skipped.len(), 1, "full capture:\n{}", captured.text);
    assert_eq!(skipped[0]["fields"]["client_id"], old.as_str());
    assert_eq!(
        fx.slot_ids()[slot],
        fresh,
        "the slot keeps the fresh client"
    );
    let watches: HashSet<String> = fx.client.txn_slot_watches_for_test().into_iter().collect();
    assert!(
        watches.contains(&fresh) && !watches.contains(&old),
        "{watches:?}"
    );
    fx.every_slot_serves().await;
    fx.quit().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ending_a_txn_slots_router_right_after_a_swap_restarts_the_fresh_clients_router() {
    let fx = Fixture::start(MEMBER_REPLACE_COOLDOWN).await;
    let old = fx.slot_ids()[0].clone();
    fx.trigger(0);
    let fresh = fx.swapped(0, &old).await;
    assert_eq!(fx.txn(ROUTER_RESTARTS, None), 0);

    assert!(fx.client.end_router_for_test(VALKEY_POOL_TXN, 0));
    await_until(
        "the fresh client's ended router must be restarted",
        Instant::now() + SETTLE,
        || fx.txn(ROUTER_RESTARTS, None) == 1,
    )
    .await;

    assert_eq!(fx.slot_ids()[0], fresh, "a restart keeps the slot's client");
    assert!(fx.client.txn_slot_watches_for_test().contains(&fresh));
    fx.every_slot_serves().await;
    fx.quit().await;
}

/// Current-thread runtime: the refusal's WARN is logged on the test's thread.
#[tokio::test]
async fn once_the_txn_budget_is_used_up_a_due_slot_replacement_is_refused_counted_and_warned_once()
{
    let capture = ThreadCapture::start();
    let fx = Fixture::start(Duration::ZERO).await;
    let budget = fx
        .client
        .txn_pool
        .slots()
        .max(fred::live_edit::pool::MIN_SPARE_CELLS);
    assert_eq!(budget, 4);

    for n in 0..budget {
        let slot = n % fx.client.txn_pool.slots();
        let old = fx.slot_ids()[slot].clone();
        fx.trigger(slot);
        fx.swapped(slot, &old).await;
    }
    let ids = fx.slot_ids();
    fx.trigger(0);
    assert_eq!(
        fx.txn(
            MEMBER_REPLACEMENTS,
            Some(VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED)
        ),
        1
    );
    fx.client.fast_verdict_for_test(VALKEY_POOL_TXN, 0);

    assert_eq!(
        fx.txn(
            MEMBER_REPLACEMENTS,
            Some(VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED)
        ),
        2,
        "every refused replacement is counted"
    );
    assert_eq!(
        fx.txn(
            MEMBER_REPLACEMENTS,
            Some(VALKEY_MEMBER_REPLACEMENT_REPLACED)
        ),
        budget as i64
    );
    assert!(!fx
        .client
        .member_replacement_in_flight_for_test(VALKEY_POOL_TXN));
    assert_eq!(
        fx.client.member_replacement_tasks_for_test(),
        0,
        "a refused replacement starts no task"
    );
    assert_eq!(fx.slot_ids(), ids, "the slots keep their clients");
    let captured = capture.snapshot();
    assert_eq!(
        messages_containing(&captured, TXN_BUDGET_WARN).len(),
        1,
        "one WARN per pool; full capture:\n{}",
        captured.text
    );
    fx.quit().await;
}

/// The connect timeout (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`) of the
/// held-retirement tests' client: `C`, the per-replacement term of the
/// scheduling bound.
const CONNECT_TIMEOUT_SECS: u64 = 5;

/// Slack on top of `k × C` in the scheduling bound; the healthy path takes
/// milliseconds, so this is a deadline a hang would blow, not a measurement.
const SCHEDULING_SLACK: Duration = Duration::from_secs(15);

/// The unresponsive bounds and command timeout of the held-retirement tests'
/// client: past anything they wait, so the held reply below is neither judged
/// nor timed out while the test holds it.
const HELD_REPLY_BOUND: Duration = Duration::from_secs(90);

/// A client with four `txn` slots, connected through a proxy that holds every
/// reply on the connection that sends a `GET` of `marker` — from that `GET` on,
/// until `gate` opens — and its fixture (hooks, the production trigger). The
/// proxy is returned to be kept alive.
///
/// A slot's old client that has such a `GET` in flight cannot finish its
/// retirement: its `QUIT` drains the replies still owed first, so its router
/// does not end until the gate opens (or `MEMBER_RETIRE_QUIT_BOUND`, 50 s,
/// passes). That is how these tests hold a retirement open on demand, with no
/// timing margin.
async fn held_reply_fixture(marker: &str, gate: &ReplyGate) -> (ScriptReplyProxy, Fixture) {
    let rule: ProxyRule = {
        let marker = marker.as_bytes().to_vec();
        let gate = gate.clone();
        Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"GET") && *args.get(1)? == marker)
                .then(|| ProxyAction::HoldReplies(gate.clone()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client_with(
        &proxy,
        4,
        LivenessOverrides {
            fast_timeout: Some(HELD_REPLY_BOUND),
            slow_timeout: Some(HELD_REPLY_BOUND),
            ..LivenessOverrides::default()
        },
        |config| {
            config.valkey_connect_timeout_secs = CONNECT_TIMEOUT_SECS;
            config.valkey_command_timeout_ms = HELD_REPLY_BOUND.as_millis() as u64;
        },
    )
    .await;
    let fx = Fixture::over(client, MEMBER_REPLACE_COOLDOWN).await;
    (proxy, fx)
}

/// Send a `GET` of `marker` through `client` (a checked-out slot's client) and
/// wait until the proxy holds its reply: the command is in flight on that
/// client's connection until the gate opens.
async fn hold_a_reply_on(client: &Client, marker: &str, gate: &ReplyGate) {
    let client = client.clone();
    let marker = marker.to_owned();
    // Its outcome is not the subject; the task ends when the gate opens or
    // the client is released.
    tokio::spawn(async move {
        let _: Result<Option<String>, _> = client.get(&marker).await;
    });
    gate.reply_held().await;
}

/// The slot index `fx`'s slots hold `id` at.
fn slot_of(fx: &Fixture, id: &str) -> usize {
    fx.slot_ids()
        .iter()
        .position(|slot| slot == id)
        .expect("the checked-out client is a slot's")
}

/// Review 4(ii): a `txn` slot that comes due while another slot's replacement
/// holds the pool (waiting for a caller to release the slot it replaces) is
/// started at that replacement's swap, not after the swapped-out client's
/// retirement. The swapped-out client has a reply held in flight, so its
/// retirement stays open; the second slot is still swapped within 2 × C of the
/// caller's release. The post-swap reads are part of the wait condition
/// (multi-thread runtime).
///
/// Fail-on-old, by in-place perturbation: deleting the `guard.release_slot()`
/// after the `txn` swap in `MemberReplacer::swap_txn_slot` leaves the pool held
/// until the replacement's task ends with the retirement, which the held reply
/// keeps open, so the second slot's replacement does not start and the 2 × C
/// wait goes red.
///
/// Wait budget: the client's connect (30 s), the held reply's report (20 s),
/// the replacement reaching the held slot ([`SETTLE`]), the bound
/// (2 × 5 + 15 s) and `quit()` ([`SETTLE`]): 95 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_txn_slot_due_while_another_is_replaced_is_started_at_that_swap_not_after_its_retirement()
{
    let marker = format!("zt3:txn-held-reply:{}", uuid::Uuid::new_v4());
    let gate = ReplyGate::new();
    let (_proxy, fx) = held_reply_fixture(&marker, &gate).await;
    let permits = fx.client.txn_pool.available_permits();
    let checkout = fx.client.acquire_txn().await;
    let old_a = ClientLike::id(&*checkout).to_owned();
    let slot_a = slot_of(&fx, &old_a);
    let slot_b = (slot_a + 1) % fx.slot_ids().len();
    let old_b = fx.slot_ids()[slot_b].clone();
    hold_a_reply_on(&checkout, &marker, &gate).await;

    fx.trigger(slot_a);
    // Two permits out — this checkout's and the replacement's — means the
    // replacement is connected and waiting on the slot this caller holds.
    await_until(
        "slot A's replacement must take a permit and wait on the held slot",
        Instant::now() + SETTLE,
        || fx.client.txn_pool.available_permits() == permits - 2,
    )
    .await;
    fx.trigger(slot_b);
    assert!(fx
        .client
        .member_replacement_in_flight_for_test(VALKEY_POOL_TXN));
    assert_eq!(
        fx.client.member_replacement_tasks_for_test(),
        1,
        "slot B, due while slot A's replacement holds the pool, is not started"
    );

    let released = Instant::now();
    drop(checkout);
    await_until(
        "slot B must be swapped within 2 × C of the caller's release, while slot A's old \
         client still retires",
        released + Duration::from_secs(CONNECT_TIMEOUT_SECS) * 2 + SCHEDULING_SLACK,
        || {
            let now = fx.slot_ids();
            now[slot_a] != old_a
                && now[slot_b] != old_b
                && fx.client.member_retirements_for_test() >= 1
                && fx.txn(
                    MEMBER_REPLACEMENTS,
                    Some(VALKEY_MEMBER_REPLACEMENT_REPLACED),
                ) == 2
        },
    )
    .await;

    gate.open();
    fx.quit().await;
}

/// T-R4 (I-A2) on the `txn` pool, with no timing margin: only the replacement
/// holding the pool releases it. Slot A is replaced while its old client has a
/// reply held in flight, so its retirement stays open; slot B's replacement
/// then holds the pool, waiting for a caller to release slot B. When the held
/// reply is let through, slot A's retirement ends — and the pool still reads
/// held: that retirement's end does not clear a flag a later replacement now
/// holds. The data-pool T-R4 in `b_zt3_member_replacement_test.rs` pins the
/// same with a multiplexed member's grace.
///
/// Fail-on-old, by in-place perturbation: making the `Replacement` guard's
/// `Drop` clear the pool's flag unconditionally turns the "the pool must still
/// read held" assertion red.
///
/// Wait budget: the client's connect (30 s), the held reply's report (20 s),
/// five polled conditions ([`SETTLE`] each) and `quit()` ([`SETTLE`]): 110 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_txn_retirement_that_ends_while_another_slot_replacement_holds_the_pool_leaves_it_held() {
    let marker = format!("zt3:txn-held-reply:{}", uuid::Uuid::new_v4());
    let gate = ReplyGate::new();
    let (_proxy, fx) = held_reply_fixture(&marker, &gate).await;
    let permits = fx.client.txn_pool.available_permits();
    let a = fx.client.acquire_txn().await;
    let old_a = ClientLike::id(&*a).to_owned();
    let slot_a = slot_of(&fx, &old_a);
    hold_a_reply_on(&a, &marker, &gate).await;
    let b = fx.client.acquire_txn().await;
    let old_b = ClientLike::id(&*b).to_owned();
    let slot_b = slot_of(&fx, &old_b);

    fx.trigger(slot_a);
    await_until(
        "slot A's replacement must take a permit and wait on the held slot",
        Instant::now() + SETTLE,
        || fx.client.txn_pool.available_permits() == permits - 3,
    )
    .await;
    drop(a);
    await_until(
        "slot A must be swapped, the pool freed, its old client retiring",
        Instant::now() + SETTLE,
        || {
            fx.slot_ids()[slot_a] != old_a
                && !fx
                    .client
                    .member_replacement_in_flight_for_test(VALKEY_POOL_TXN)
                && fx.client.member_retirements_for_test() == 1
        },
    )
    .await;

    fx.trigger(slot_b);
    await_until(
        "slot B's replacement must take a permit and wait on the slot this caller holds",
        Instant::now() + SETTLE,
        || fx.client.txn_pool.available_permits() == permits - 2,
    )
    .await;
    assert_eq!(
        (
            fx.client
                .member_replacement_in_flight_for_test(VALKEY_POOL_TXN),
            fx.client.member_replacement_tasks_for_test()
        ),
        (true, 2),
        "precondition: slot B's replacement holds the pool while slot A's old client retires"
    );

    gate.open();
    await_until(
        "slot A's retirement must end once its held reply is through",
        Instant::now() + SETTLE,
        || fx.client.member_replacement_tasks_for_test() == 1,
    )
    .await;
    assert!(
        fx.client
            .member_replacement_in_flight_for_test(VALKEY_POOL_TXN),
        "the pool must still read held: only the replacement holding it releases it"
    );

    drop(b);
    await_until(
        "slot B must be swapped once its caller releases it",
        Instant::now() + SETTLE,
        || fx.slot_ids()[slot_b] != old_b && fx.client.member_replacement_tasks_for_test() == 0,
    )
    .await;
    fx.quit().await;
}
