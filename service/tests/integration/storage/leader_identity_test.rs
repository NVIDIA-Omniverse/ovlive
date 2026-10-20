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

//! SDD SR-2.5 — two processes that share a `service_label` never both hold a
//! leader lock.
//!
//! `service_label` is operator-chosen and not unique, so a lock value of the
//! bare label would let the acquire script's "already ours, extend it" branch
//! hand the lock to every same-labelled replica at once. The lock value is a
//! [`LeaderIdentity::lock_value`] (`label:instance`) instead. Each test here
//! drives two identities with ONE label and two instances through the public
//! lock wrappers (`try_acquire_leader_lock` / `refresh_leader_lock_if_held`)
//! and through [`LeaderLease`], and fails if both ever report holding.
//!
//! The tests run on `le:presence:sweeper:leader`, a lock the plain wrappers
//! and [`LeaderLease`] guard in production. `le:gc:leader` carries the
//! protocol-marked `LeaderIdentity::gc_lock_value` and its own takeover
//! script since b.1jt; its same-label case (a same-version peer is `held`) is
//! in `b_1jt_rollout_tests.rs`.
//!
//! The lapse is real, not simulated: the lock carries a short TTL (SR-2.3's
//! explicit `ttl_ms`) and the test bounded-polls `EXISTS` until it is gone,
//! which makes the hand-over test take about [`SHORT_TTL_MS`] of wall time.
//! [`LeaderLease`]'s deadlines are `std::time::Instant`s, so a lease is built
//! with an `acquired_at` in the past to make its refresh due without waiting.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fred::interfaces::KeysInterface;
use tokio::sync::Barrier;

use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::{refresh_leader_lock_if_held, try_acquire_leader_lock};
use live_edit::storage::leader_lease::{
    LeaderIdentity, LeaderLease, LeaseOutcome, LeaseState, LostCause,
};
use live_edit::storage::test_support::TestValkey;

/// The `service_label` both processes share.
const SHARED_LABEL: &str = "shared-replica-label";

/// The lock TTL for the hand-over test. Short so the lapse costs ~2 s of wall
/// time; long enough that the handful of round-trips between A's acquire and
/// B's checks cannot outlive it on a loaded runner.
const SHORT_TTL_MS: u64 = 2_000;

/// Budget for A's lock to lapse — five TTLs, so only a lock that never
/// expires (a broken `PEXPIRE`) can exhaust it.
const LAPSE_DEADLINE: Duration = Duration::from_secs(10);

/// Gap between `EXISTS` probes while waiting out the lapse.
const LAPSE_POLL: Duration = Duration::from_millis(25);

/// The [`LeaderLease`] TTL. The lease's refresh is made due by backdating its
/// `acquired_at` by half of this, which leaves the local deadline half a TTL
/// ahead — far enough that the check cannot slide onto the overshoot arm.
const LEASE_TTL: Duration = Duration::from_secs(30);

/// Rounds of the simultaneous-acquire race.
const RACE_ROUNDS: usize = 20;

/// Bound on one race round's two acquires.
const RACE_ROUND_DEADLINE: Duration = Duration::from_secs(5);

fn identities() -> (LeaderIdentity, LeaderIdentity) {
    (
        LeaderIdentity::new(SHARED_LABEL, "instance-a"),
        LeaderIdentity::new(SHARED_LABEL, "instance-b"),
    )
}

async fn acquire(client: &ValkeyClient, key: &str, who: &LeaderIdentity) -> bool {
    try_acquire_leader_lock(client, key, who, SHORT_TTL_MS)
        .await
        .unwrap_or_else(|e| panic!("acquire as {who} must not error: {e}"))
}

async fn refresh(client: &ValkeyClient, key: &str, who: &LeaderIdentity) -> bool {
    refresh_leader_lock_if_held(client, key, who, SHORT_TTL_MS)
        .await
        .unwrap_or_else(|e| panic!("refresh as {who} must not error: {e}"))
}

async fn lock_value(client: &ValkeyClient, key: &str) -> Option<String> {
    client.pool.get(key).await.expect("GET must succeed")
}

/// One sampled point: ask each identity, through refresh-if-held, whether it
/// holds the lock, and fail if both say yes. Returns `(a_holds, b_holds)`.
///
/// Refresh-if-held is the probe because it is the check a lease-holding
/// caller makes; it never claims, so sampling cannot create a holder.
async fn sample_holders(
    client: &ValkeyClient,
    key: &str,
    a: &LeaderIdentity,
    b: &LeaderIdentity,
    point: &str,
) -> (bool, bool) {
    let a_holds = refresh(client, key, a).await;
    let b_holds = refresh(client, key, b).await;
    assert!(
        !(a_holds && b_holds),
        "{point}: both same-label processes report holding the lock"
    );
    (a_holds, b_holds)
}

/// Bounded-poll until `key` no longer exists.
async fn wait_for_lapse(client: &ValkeyClient, key: &str) {
    let deadline = Instant::now() + LAPSE_DEADLINE;
    loop {
        let exists: i64 = client.pool.exists(key).await.expect("EXISTS must succeed");
        if exists == 0 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the {SHORT_TTL_MS} ms lock did not lapse within {LAPSE_DEADLINE:?}"
        );
        tokio::time::sleep(LAPSE_POLL).await;
    }
}

/// A lease as `who` whose refresh is due now and whose local deadline is
/// still half a [`LEASE_TTL`] away, so `ensure_held` round-trips.
fn refresh_due_lease(key: &str, who: &LeaderIdentity) -> LeaderLease {
    let acquired_at = Instant::now()
        .checked_sub(LEASE_TTL / 2)
        .expect("the monotonic clock must be older than half a lease TTL");
    LeaderLease::new(key.to_owned(), who.clone(), acquired_at, LEASE_TTL)
}

/// A holds the lock and B — same label, other instance — is locked out of
/// both acquire and refresh; once A's lock lapses B takes it, A is locked
/// out, and A's [`LeaderLease`] reports `Lost`. No sampled point has both
/// holding.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_label_processes_never_both_hold_the_lock_and_the_loser_reports_lost() {
    let tv = TestValkey::setup().await;
    let client = tv.client();
    let key = tv.key_schema().presence_sweeper_leader_lock();
    let (a, b) = identities();

    // ── A leads ──────────────────────────────────────────────────────────
    assert!(acquire(client, &key, &a).await, "A claims the free lock");
    assert!(
        !acquire(client, &key, &b).await,
        "B shares A's label but is another process: its acquire must not take A's lock"
    );
    assert!(
        !refresh(client, &key, &b).await,
        "B's refresh-if-held must not treat A's lock as its own"
    );
    assert_eq!(lock_value(client, &key).await, Some(a.lock_value()));
    assert_eq!(
        sample_holders(client, &key, &a, &b, "while A leads").await,
        (true, false)
    );

    // ── A's lock lapses; B takes over ────────────────────────────────────
    wait_for_lapse(client, &key).await;
    assert_eq!(
        sample_holders(client, &key, &a, &b, "after the lapse").await,
        (false, false),
        "nobody holds a lapsed lock, and refresh-if-held must not reclaim it"
    );
    assert!(acquire(client, &key, &b).await, "B claims the lapsed lock");
    assert!(
        !refresh(client, &key, &a).await,
        "A's refresh-if-held must not treat B's lock as its own"
    );
    assert!(
        !acquire(client, &key, &a).await,
        "A's acquire must not take B's lock"
    );
    assert_eq!(lock_value(client, &key).await, Some(b.lock_value()));
    assert_eq!(
        sample_holders(client, &key, &a, &b, "while B leads").await,
        (false, true)
    );

    // ── Through LeaderLease: the loser is Lost, the winner renews ────────
    let mut lease_a = refresh_due_lease(&key, &a);
    let check_a = lease_a.ensure_held(client).await;
    assert_eq!(check_a.state, LeaseState::Lost, "A's lease: {check_a:?}");
    assert_eq!(check_a.outcome, Some(LeaseOutcome::Lost));
    assert!(
        matches!(check_a.cause, Some(LostCause::NotHeld)),
        "A's refresh round-trip found the lock another process's: {check_a:?}"
    );

    let mut lease_b = refresh_due_lease(&key, &b);
    let check_b = lease_b.ensure_held(client).await;
    assert_eq!(check_b.state, LeaseState::Held, "B's lease: {check_b:?}");
    assert_eq!(check_b.outcome, Some(LeaseOutcome::Renewed));
    assert_eq!(
        lock_value(client, &key).await,
        Some(b.lock_value()),
        "A's lost check must leave B's lock in place"
    );
}

/// Two same-label processes acquire the free lock at the same instant, many
/// times over: exactly one wins every round, and the lock holds the winner's
/// lock value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_label_processes_racing_to_acquire_exactly_one_wins_every_round() {
    let tv = TestValkey::setup_with_pool_size(4).await;
    let key = tv.key_schema().presence_sweeper_leader_lock();
    let (a, b) = identities();

    for round in 0..RACE_ROUNDS {
        let _: i64 = tv.client().pool.del(&key).await.expect("DEL must succeed");
        let barrier = Arc::new(Barrier::new(2));
        let racers = [a.clone(), b.clone()].map(|who| {
            let client = tv.client().clone();
            let key = key.clone();
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                barrier.wait().await;
                acquire(&client, &key, &who).await
            })
        });
        let [won_a, won_b] = racers;
        let (won_a, won_b) = tokio::time::timeout(RACE_ROUND_DEADLINE, async {
            (
                won_a.await.expect("racer A must not panic"),
                won_b.await.expect("racer B must not panic"),
            )
        })
        .await
        .unwrap_or_else(|_| panic!("round {round}: the acquires did not finish in time"));

        assert!(
            won_a ^ won_b,
            "round {round}: exactly one same-label process must win (A={won_a}, B={won_b})"
        );
        let winner = if won_a { &a } else { &b };
        assert_eq!(
            lock_value(tv.client(), &key).await,
            Some(winner.lock_value()),
            "round {round}: the lock must hold the winner's lock value"
        );
    }
}
