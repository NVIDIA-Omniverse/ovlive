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

//! b.zt3 T17 — the vendored fred `Pool`'s arena: a member re-pointed at a
//! fresh client by `fred::live_edit::pool::replace_member`.
//!
//! Tested on a fred `Pool` of its own, not a `ValkeyClient`'s, so the member
//! replacer (which keeps its own record of each pool's members) never sees
//! these swaps. Each fresh client is `Client::clone_new` of the member it
//! replaces, connected first, as the replacer does it.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`every_pool_operation_reaches_the_replaced_members_fresh_client_and_a_client_borrowed_before_stays_valid`] | `member`, `clients`, `next`, `next_connected`, `last`, `ClientLike::inner` (every command on the pool itself), `wait_for_connect` and `quit` act on the current members; a `&Client` taken before is still the same, memory-valid client |
//! | [`replace_member_refuses_once_the_spare_cells_are_used_up_and_changes_nothing`] | `max(size, 4)` spare cells, then `BudgetExhausted` with the member unchanged |
//!
//! "Stays valid" means memory-valid: the borrowed client is the retired one,
//! which in the service is quit and released and then fails every command by
//! design (CORE-2).
//!
//! # Wait budget
//!
//! Every wait is bounded: each command at the pool's 5 s command timeout
//! ([`COMMAND_TIMEOUT`], set on the builder: fred's own default is none), and
//! each connect, `QUIT` and polled client state at [`BOUND`] (5 s). The longer
//! test, the budget one, waits at most 5 + (5 + 5) × 4 + 5 = 50 s, under the
//! 180 s kill deadline `.config/nextest.toml` sets for the `b_zt3_` modules.

use std::collections::HashSet;
use std::time::Duration;

use fred::clients::{Client, Pool};
use fred::interfaces::{ClientLike, KeysInterface};
use fred::live_edit::pool::{replace_member, spare_cells, spare_cells_left, BudgetExhausted};
use fred::prelude::Builder;
use fred::types::config::Config as FredConfig;
use fred::types::ClientState;
use tokio::time::Instant;

use super::b_zt3_support::await_until;

/// Bound on each connect, `wait_for_connect`, `QUIT`, and polled client state.
const BOUND: Duration = Duration::from_secs(5);

/// The pool's command timeout: fred's default is none, so a command that
/// hangs would otherwise run into the silent nextest kill.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(5);

async fn pool_of(size: usize) -> Pool {
    let url = crate::common::test_config().valkey_url;
    let config = FredConfig::from_url(&url).expect("TEST_VALKEY_URL must parse");
    let pool = Builder::from_config(config)
        .with_performance_config(|perf| perf.default_command_timeout = COMMAND_TIMEOUT)
        .build_pool(size)
        .expect("the pool must build");
    tokio::time::timeout(BOUND, pool.init())
        .await
        .expect("the pool's connect must settle")
        .expect("the pool must connect");
    pool
}

/// `client.quit()`, bounded by [`BOUND`].
async fn quit_bounded(client: &Client, what: &str) {
    tokio::time::timeout(BOUND, client.quit())
        .await
        .unwrap_or_else(|_| panic!("{what}'s QUIT must settle"))
        .unwrap_or_else(|e| panic!("{what} must quit: {e:?}"));
}

/// Polls until `client`'s router has ended (`Disconnected`), by [`BOUND`]:
/// a `quit()` returns once its `QUIT` is answered, a little before the router
/// task records the state.
async fn await_disconnected(client: &Client, what: &str) {
    await_until(
        &format!("{what} must end up Disconnected"),
        Instant::now() + BOUND,
        || client.state() == ClientState::Disconnected,
    )
    .await;
}

/// A connected `clone_new` of member `i`'s current client.
async fn fresh_for(pool: &Pool, i: usize) -> Client {
    let fresh = pool.member(i).clone_new();
    tokio::time::timeout(BOUND, fresh.init())
        .await
        .expect("the fresh client's connect must settle")
        .expect("the fresh client must connect");
    fresh
}

fn id(client: &Client) -> String {
    ClientLike::id(client).to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_pool_operation_reaches_the_replaced_members_fresh_client_and_a_client_borrowed_before_stays_valid(
) {
    let pool = pool_of(2).await;
    let (original_0, original_1) = (id(pool.member(0)), id(pool.member(1)));
    let borrowed: &Client = pool.member(0);
    let fresh = fresh_for(&pool, 0).await;
    let fresh_id = id(&fresh);

    let retired = replace_member(&pool, 0, fresh.clone()).expect("a spare cell is free");
    quit_bounded(&retired, "the retired client").await;

    assert_eq!(
        id(&retired),
        original_0,
        "the retired client is the one member 0 named"
    );
    assert_eq!(id(pool.member(0)), fresh_id);
    assert_eq!(id(pool.member(1)), original_1);
    assert_eq!(
        pool.clients().iter().map(id).collect::<Vec<_>>(),
        [fresh_id.clone(), original_1.clone()]
    );
    let current: HashSet<String> = [fresh_id.clone(), original_1.clone()].into();
    let picked: HashSet<String> = (0..4)
        .flat_map(|_| [id(pool.next()), id(pool.next_connected())])
        .collect();
    assert_eq!(
        picked, current,
        "next and next_connected pick only current members"
    );
    for _ in 0..2 {
        let ran = id(pool.next());
        assert_eq!(
            id(pool.last()),
            ran,
            "last names the current member next picked"
        );
    }
    assert!(
        current.contains(&ClientLike::inner(&pool).id.to_string()),
        "a command on the pool itself goes to a current member"
    );
    assert_eq!(
        id(borrowed),
        original_0,
        "the borrowed client is still the same client"
    );
    // It is the retired one: its router ends after the QUIT above.
    await_disconnected(borrowed, "the borrowed (retired) client").await;

    tokio::time::timeout(BOUND, pool.wait_for_connect())
        .await
        .expect("wait_for_connect must not wait on the retired client")
        .expect("the current members are connected");
    let _: Option<String> = pool
        .get(format!("zt3:arena:{}", uuid::Uuid::new_v4()))
        .await
        .expect("a command on the pool must be served by a current member");
    tokio::time::timeout(BOUND, pool.quit())
        .await
        .expect("quit must settle")
        .expect("quit must succeed");
    await_disconnected(&fresh, "the fresh client (quit reaches it)").await;
    await_disconnected(pool.member(1), "member 1").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replace_member_refuses_once_the_spare_cells_are_used_up_and_changes_nothing() {
    let pool = pool_of(2).await;
    assert_eq!(spare_cells(&pool), 4, "max(size, 4)");

    for n in 0..4 {
        let fresh = fresh_for(&pool, n % 2).await;
        let retired = replace_member(&pool, n % 2, fresh).expect("a spare cell is free");
        quit_bounded(&retired, "the retired client").await;
        assert_eq!(spare_cells_left(&pool), 3 - n);
    }
    let before: Vec<String> = pool.clients().iter().map(id).collect();
    let refused = pool.member(0).clone_new();

    assert_eq!(
        replace_member(&pool, 0, refused).map(|c| id(&c)),
        Err(BudgetExhausted)
    );
    assert_eq!(pool.clients().iter().map(id).collect::<Vec<_>>(), before);
    assert_eq!(spare_cells_left(&pool), 0);
    tokio::time::timeout(BOUND, pool.quit())
        .await
        .expect("quit must settle")
        .expect("quit must succeed");
}
