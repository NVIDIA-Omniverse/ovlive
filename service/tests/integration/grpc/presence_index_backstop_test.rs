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

// The orphan sweep's presence index backstop (PRD FR-8: every session with
// presence entries is visited), driven through the real orphan-sweep and
// presence-sweeper tasks against a real Valkey:
//
//   * gap 1 — a joining replica dies between its register and its index
//     re-add, after a concurrent teardown's index removal took the entry out;
//   * gap 2 — a teardown's index removal fails after its `SREM` (its `SCARD`
//     times out) while a concurrent joiner, whose re-add landed before that
//     `SREM`, dies before its keeper's first renewal;
//
// either leaves a dead member un-indexed, which the presence sweeper alone
// never visits. Each gap test runs a presence pass first (it removes nothing),
// then one orphan-sweep pass (the session is indexed again, its member set
// untouched), then a second presence pass (the member is removed, exactly one
// `left` reaches a peer, and the emptied session leaves the index);
//
//   * the race — a member set the backstop's walk saw is emptied by a teardown
//     before the backstop's `SADD`: the session is indexed with no members, and
//     the next presence pass removes the entry, announcing nothing;
//   * a cancel while the first chunk is checked — that chunk is re-added and
//     no later one;
//   * an indexed session with a live member — left as it was by an
//     orphan-sweep pass and the presence pass after it;
//   * a refused `SCAN` — the backstop's walk fails (an ACL user denied `SCAN`):
//     the error is skipped, the phase and the pass do not fail, and the next
//     pass re-indexes the set;
//   * a refused `SADD` — the re-add fails (an ACL user denied `SADD`): counted
//     once, the session left un-indexed, and the later phases still run.
//
// The phase's own unit tests — chunked re-indexing, writing only index
// entries, a failed `SMISMEMBER` leaving the later phases running, running
// before a failing object phase, and cancellation — are in
// `src/storage/orphan_sweep.rs`.
//
// ── How a pass is run and observed ───────────────────────────────────────────
//
// An orphan-sweep pass is one `spawn_orphan_sweep` task, which runs its first
// pass at once and the next only after 300 s. The pass's last act publishes
// the retire-retry backlog gauge under the leader's service label, and nothing
// else publishes it here, so the gauge's appearance is the pass's end; the
// task is then cancelled.
//
// A presence pass is one `spawn_presence_sweeper_with_seam` task whose first
// pass starts one `PASS_INTERVAL` after the spawn. It is stopped as soon as the
// seam reports that pass ended, before its next is due, so nothing a test does
// between passes races a pass. Every sweeper of a test shares one leader
// identity, so each takes the lock at once.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every wait is bounded by `SETTLE` (10 s), or by `PASS_INTERVAL + SETTLE` for
// a presence pass; the healthy path returns from each in milliseconds, except
// a presence pass (one `PASS_INTERVAL`, 3 s), a dead joiner's lease lapsing
// (`DEAD_LEASE_SECS`, 1 s) and, in gap 2, the held `SCARD`'s 1 s command
// timeout. Each test body runs under one `SCENARIO_BUDGET` (100 s) deadline,
// below nextest's hang-killer; the heaviest healthy paths, the two gap tests
// (two presence passes each), take about 8 s.
//
// The whole file is compiled only with `--features test-fault-injection` (it
// drives the presence sweeper's seam), the cluster variant included. That one
// self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
// is unset; the refused-`SCAN` and refused-`SADD` tests print one when the
// store cannot host an ACL user.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, ClusterInterface, HashesInterface, KeysInterface, SetsInterface,
};
use tokio_util::sync::CancellationToken;

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, PRESENCE_SWEEP_OUTCOME_COMPLETED};
use live_edit::presence::{
    spawn_presence_sweeper_with_seam, ConnectedUser, ObservedSweepPass, PresenceEvent,
    PresenceSweeperSeam, SweeperBudget, DEFAULT_PRESENCE_LEASE_TTL_SECS,
};
use live_edit::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::presence::{
    add_session_to_index, register_connection, remove_session_from_index, store_time_ms,
    unregister_connection, RegisterOutcome,
};
use live_edit::storage::test_support::{connect_test_client, release_all, ConnStall, TestValkey};

use super::helpers::*;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on any single wait.
const SETTLE: Duration = PRESENCE_WAIT;

/// A presence sweeper's interval: its first pass starts this long after the
/// spawn, and it is stopped once that pass ends, before its next is due.
const PASS_INTERVAL: Duration = Duration::from_secs(3);

/// The lease of a joiner whose replica dies: short, so it lapses inside the
/// test.
const DEAD_LEASE_SECS: u64 = 1;

/// The data-pool command timeout of the client whose index removal fails in
/// gap 2: its `SCARD` is held past it.
const FAILING_COMMAND_TIMEOUT_MS: u64 = 1_000;

/// The command timeout of the backstop client whose check is held: far longer
/// than its held `SMISMEMBER` waits.
const HELD_COMMAND_TIMEOUT_MS: u64 = 10_000;

/// Maintenance connections of the backstop client in the tests that hold its
/// check: the race and the mid-phase cancel.
///
/// An orphan-sweep pass takes its leader probe's connection and then its
/// backstop walk's with an explicit `pool.next()` each, so they are the
/// pool's next two. The backstop's `SMISMEMBER` is the pass's first command
/// sent through the pool itself, and fred routes such a command by advancing
/// the same round-robin counter a fixed handful of times, so it lands on
/// another connection. The test holds every connection but the first two, so
/// the `SMISMEMBER`'s is held whichever it is, provided the pool is larger
/// than that handful plus two: eight is well clear of it. A miscount fails the
/// test rather than passing it: the backstop re-indexes the set before the
/// teardown, and the precondition that its check is held fails.
const RACE_MAINTENANCE_POOL: u32 = 8;

/// The `user_application` of every member registered here.
const APP: &str = "backstop-app";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The URL `TestValkey::setup` connects to.
fn standalone_url() -> String {
    std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// A `ValkeyClient` of its own over the store at `url`, with a command timeout
/// of `command_timeout_ms` and `maintenance_pool` maintenance connections.
async fn own_client(url: &str, command_timeout_ms: u64, maintenance_pool: u32) -> ValkeyClient {
    let config = live_edit::config::Config {
        valkey_url: url.to_owned(),
        valkey_command_timeout_ms: command_timeout_ms,
        maintenance_pool_size: maintenance_pool,
        valkey_connect_timeout_secs: 30,
        ..crate::common::test_config()
    };
    connect_test_client(&config)
        .await
        .expect("the test's own ValkeyClient must connect")
}

/// The pooled connection `ahead` commands after `pool`'s last: the one its
/// `ahead`-th next command takes (fred's pool is round-robin).
fn conn_ahead(pool: &fred::clients::Pool, ahead: usize) -> &fred::clients::Client {
    let last = pool.last().id();
    let clients = pool.clients();
    let at = clients
        .iter()
        .position(|c| c.id() == last)
        .expect("pool.last() is one of the pool's clients");
    pool.member((at + ahead) % clients.len())
}

/// Every byte the stalled connections have read so far, summed.
async fn held_net_in(stalls: &[ConnStall]) -> u64 {
    let mut total = 0;
    for stall in stalls {
        total += stall
            .net_in()
            .await
            .expect("a stalled connection must still exist");
    }
    total
}

/// Hold every maintenance connection of `backstop` except the next two, which
/// an orphan-sweep pass spawned on it takes for its leader probe and its
/// backstop walk; its backstop `SMISMEMBER` is then held (see
/// [`RACE_MAINTENANCE_POOL`]). Release every returned stall.
async fn hold_all_but_probe_and_walk(tv: &TestValkey, backstop: &ValkeyClient) -> Vec<ConnStall> {
    let pool_size = backstop.maintenance_pool.clients().len();
    let mut stalls = Vec::new();
    for ahead in 3..=pool_size {
        stalls.push(
            ConnStall::park(
                conn_ahead(&backstop.maintenance_pool, ahead),
                store_reader(tv),
                format!("{}conn-stall:maintenance:{ahead}", tv.prefix()),
            )
            .await,
        );
    }
    stalls
}

/// Wait until a command reaches one of `stalls`, which had read `baseline`
/// bytes before the pass began: the backstop's `SMISMEMBER`, its walk done.
async fn wait_for_held_command(stalls: &[ConnStall], baseline: u64) {
    poll_until(
        "the backstop's SMISMEMBER reaching a held connection",
        SETTLE,
        move || async move { held_net_in(stalls).await > baseline },
    )
    .await;
}

/// The data-pool error origin of `tv`'s client.
fn origin(tv: &TestValkey) -> FredErrOrigin<'_> {
    FredErrOrigin::client(tv.client(), POOL_LABEL_DATA)
}

/// Register `cid` in `sid` through the register script with a lease of
/// `lease_ttl_secs`, as a join does; the session index is not touched.
async fn register(tv: &TestValkey, sid: &str, cid: &str, lease_ttl_secs: u64) {
    let ks = tv.key_schema();
    let now_ms = store_time_ms(tv.client(), origin(tv), ks, sid)
        .await
        .expect("store_time_ms must succeed");
    let user_name = format!("User {cid}");
    let joined = PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: cid.to_owned(),
        user_name: user_name.clone(),
        user_application: APP.to_owned(),
    })
    .encode();
    let outcome = register_connection(
        tv.client(),
        origin(tv),
        ks,
        sid,
        cid,
        &user_name,
        APP,
        lease_ttl_secs,
        now_ms + lease_ttl_secs * 1000,
        &joined,
    )
    .await
    .expect("register_connection must succeed");
    assert_eq!(
        outcome,
        RegisterOutcome::Registered { added: true },
        "precondition: {cid} joined {sid}"
    );
}

/// A whole join of `cid`: the index add, the register, the index re-add.
async fn join(tv: &TestValkey, sid: &str, cid: &str, lease_ttl_secs: u64) {
    add_session_to_index(tv.client(), origin(tv), tv.key_schema(), sid)
        .await
        .expect("the join's first index add must succeed");
    register(tv, sid, cid, lease_ttl_secs).await;
    add_session_to_index(tv.client(), origin(tv), tv.key_schema(), sid)
        .await
        .expect("the join's index re-add must succeed");
}

/// The unregister half of `cid`'s teardown; asserts it removed `cid` and left
/// the member set empty, so the teardown's index removal is due.
async fn unregister_last(tv: &TestValkey, sid: &str, cid: &str) {
    let left = PresenceEvent::PeerLeft {
        connection_id: cid.to_owned(),
    }
    .encode();
    let unregistered = unregister_connection(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        DEFAULT_PRESENCE_LEASE_TTL_SECS,
        &left,
    )
    .await
    .expect("unregister_connection must succeed");
    assert!(
        unregistered.removed && unregistered.remaining == 0,
        "precondition: {cid} was {sid}'s last member; got {unregistered:?}"
    );
}

/// `sid`'s member set.
async fn members(tv: &TestValkey, sid: &str) -> BTreeSet<String> {
    store_reader(tv)
        .smembers(tv.key_schema().presence_members(sid))
        .await
        .expect("SMEMBERS must succeed")
}

/// Whether `sid`'s member set exists.
async fn member_set_exists(tv: &TestValkey, sid: &str) -> bool {
    store_reader(tv)
        .exists(tv.key_schema().presence_members(sid))
        .await
        .expect("EXISTS must succeed")
}

/// Wait for `cid`'s lease to lapse, as it does once its replica is gone.
async fn wait_lease_lapsed(tv: &TestValkey, sid: &str, cid: &str) {
    let lease = tv.key_schema().presence_lease(sid, cid);
    let lease_ref = &lease;
    poll_until(
        &format!("{cid}'s lease lapsing"),
        SETTLE,
        move || async move {
            !store_reader(tv)
                .exists::<bool, _>(lease_ref)
                .await
                .expect("EXISTS lease must succeed")
        },
    )
    .await;
}

/// The leader identity of every presence sweeper in a test.
fn presence_leader() -> LeaderIdentity {
    crate::common::test_leader("backstop-presence-sweeper")
}

/// The leader identity of every orphan sweep in a test. Its service label is
/// the label the retire-retry backlog gauge is published under.
fn orphan_leader() -> LeaderIdentity {
    crate::common::test_leader("backstop-orphan-sweep")
}

/// Run one presence-sweeper pass over `tv`'s keyspace as `leader`, stop the
/// sweeper before its next, and return that pass.
async fn presence_pass(tv: &TestValkey, leader: &LeaderIdentity, what: &str) -> ObservedSweepPass {
    let seam = Arc::new(PresenceSweeperSeam::new());
    let cancel = CancellationToken::new();
    let handle = spawn_presence_sweeper_with_seam(
        tv.client().clone(),
        tv.key_schema().clone(),
        leader.clone(),
        SweeperBudget::production(),
        Arc::new(crate::common::metrics_scrape_only("backstop-presence")),
        PASS_INTERVAL,
        cancel.clone(),
        Arc::clone(&seam),
    );
    let passes = tokio::time::timeout(PASS_INTERVAL + SETTLE, seam.wait_for_passes(1))
        .await
        .unwrap_or_else(|_| panic!("{what}: no presence pass ended"));
    cancel.cancel();
    tokio::time::timeout(SETTLE, handle)
        .await
        .expect("the presence sweeper must stop once cancelled")
        .expect("the presence sweeper must not panic");
    assert_eq!(
        seam.passes().len(),
        1,
        "{what}: the sweeper must be stopped before its next pass; it ran {:?}",
        seam.passes()
    );
    passes[0]
}

/// Whether the orphan-sweep pass reporting into `metrics` as `leader` has
/// ended: its last act publishes the retire-retry backlog gauge.
fn orphan_pass_ended(metrics: &Metrics, leader: &LeaderIdentity) -> bool {
    sample_value(
        &render(metrics),
        "live_edit_ext_retire_retry_backlog",
        &[("service", &leader.service_label)],
    )
    .is_some()
}

/// Spawn an orphan sweep on `client` over `tv`'s keyspace as `leader`; it runs
/// its first pass at once. Returns its metrics, cancel token and task.
fn spawn_orphan(
    client: &ValkeyClient,
    tv: &TestValkey,
    leader: &LeaderIdentity,
) -> (Arc<Metrics>, CancellationToken, tokio::task::JoinHandle<()>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only("backstop-orphan"));
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        client.clone(),
        tv.key_schema().clone(),
        Arc::new(MockExternalValueStore::new()),
        Arc::new(live_edit::storage::layer::LayerStore::new(
            client.clone(),
            tv.key_schema().clone(),
            0,
        )),
        leader.clone(),
        cancel.clone(),
        Some(Arc::clone(&metrics)),
    );
    (metrics, cancel, handle)
}

/// Wait for the orphan sweep's first pass to end, then stop the task.
async fn finish_orphan_pass(
    metrics: &Metrics,
    leader: &LeaderIdentity,
    cancel: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
) {
    poll_until("the orphan-sweep pass ending", SETTLE, || async {
        orphan_pass_ended(metrics, leader)
    })
    .await;
    cancel.cancel();
    tokio::time::timeout(SETTLE, handle)
        .await
        .expect("the orphan sweep must stop once cancelled")
        .expect("the orphan sweep must not panic");
}

/// One whole orphan-sweep pass on `client` as `leader`; returns its metrics.
async fn orphan_pass(
    client: &ValkeyClient,
    tv: &TestValkey,
    leader: &LeaderIdentity,
) -> Arc<Metrics> {
    let (metrics, cancel, handle) = spawn_orphan(client, tv, leader);
    finish_orphan_pass(&metrics, leader, cancel, handle).await;
    metrics
}

/// Every non-zero `live_edit_gc_sweep_batch_errors_total` cell, as its label
/// set and count.
fn gc_sweep_batch_errors(metrics: &Metrics) -> Vec<(String, u64)> {
    render(metrics)
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("live_edit_gc_sweep_batch_errors_total{")?;
            let (labels, value) = rest.rsplit_once(' ')?;
            let count = value.trim().parse::<f64>().ok()? as u64;
            (count > 0).then(|| (labels.trim_end_matches('}').to_owned(), count))
        })
        .collect()
}

/// The cluster primary that owns `key`'s slot.
fn owner_of(tv: &TestValkey, key: &str) -> fred::types::config::Server {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(fred::util::redis_keyslot(key.as_bytes()))
        .expect("every slot has a primary")
        .clone()
}

/// A tap on `sid`: on a cluster, attached to a node that does not own `sid`'s
/// slot, as a peer whose events cross the cluster bus.
async fn open_tap(tv: &TestValkey, sid: &str) -> Tap {
    let client = tv.client().pool.next();
    if !client.is_clustered() {
        return Tap::open(tv, sid).await;
    }
    let owner = owner_of(tv, &tv.key_schema().presence_members(sid));
    let node = client
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
        .into_iter()
        .find(|n| *n != owner)
        .expect("a cluster test needs at least two primaries");
    Tap::open_on_node(tv, sid, &node).await
}

/// The `PeerLeft`s among `events`, per connection id.
fn lefts_per_cid(events: &[PresenceEvent]) -> BTreeMap<String, usize> {
    let mut lefts = BTreeMap::new();
    for event in events {
        if let PresenceEvent::PeerLeft { connection_id } = event {
            *lefts.entry(connection_id.clone()).or_default() += 1;
        }
    }
    lefts
}

// ═════════════════════════════════════════════════════════════════════════════
// The two compound-failure gaps
// ═════════════════════════════════════════════════════════════════════════════

/// Gap 1 in `sid`: the session's last member `prev` tears down; the joiner
/// `cid` makes its first index add while the entry is still there; the
/// teardown's index removal then runs (`SREM`, `SCARD` = 0, no `SADD` back);
/// `cid` registers; and `cid`'s replica dies before its index re-add. `cid`'s
/// lease is [`DEAD_LEASE_SECS`].
async fn leave_gap_one(tv: &TestValkey, sid: &str, cid: &str) {
    join(tv, sid, "prev", DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
    unregister_last(tv, sid, "prev").await;
    add_session_to_index(tv.client(), origin(tv), tv.key_schema(), sid)
        .await
        .expect("the joiner's first index add must succeed");
    let reindexed = remove_session_from_index(tv.client(), origin(tv), tv.key_schema(), sid)
        .await
        .expect("the teardown's index removal must succeed");
    assert!(
        !reindexed,
        "precondition: the removal ran before the joiner registered, so it left {sid} out"
    );
    register(tv, sid, cid, DEAD_LEASE_SECS).await;
    // The joiner's replica dies here: no index re-add, no lease renewal.
    assert!(
        !is_indexed(tv, sid).await && is_member(tv, sid, cid).await,
        "precondition: {cid} is a member of {sid}, which is not indexed"
    );
}

/// Gap 2 in `sid`: the session's last member `prev` tears down; the joiner
/// `cid` completes its whole join, its re-add landing before the teardown's
/// index removal; that removal's `SREM` takes the entry out and its `SCARD`
/// (which would see `cid` and add the entry back) times out; and `cid`'s
/// replica dies before its keeper's first renewal re-add. `cid`'s lease is
/// [`DEAD_LEASE_SECS`].
///
/// The removal runs on a client of its own, so the test knows which pooled
/// connection the `SCARD` takes (the one after the `SREM`'s) and holds it with
/// a `ConnStall` past the client's command timeout.
async fn leave_gap_two(tv: &TestValkey, sid: &str, cid: &str) {
    join(tv, sid, "prev", DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
    unregister_last(tv, sid, "prev").await;
    join(tv, sid, cid, DEAD_LEASE_SECS).await;

    let removal = own_client(&standalone_url(), FAILING_COMMAND_TIMEOUT_MS, 1).await;
    let stall = ConnStall::park(
        conn_ahead(&removal.pool, 2),
        store_reader(tv),
        format!("{}conn-stall:scard", tv.prefix()),
    )
    .await;
    let failed = remove_session_from_index(
        &removal,
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            live_edit::storage::store_oom::StoreOomSignal::unwired(),
        ),
        tv.key_schema(),
        sid,
    )
    .await
    .expect_err("the teardown's index removal must fail at its held SCARD");
    assert_eq!(
        *failed.kind(),
        fred::error::ErrorKind::Timeout,
        "precondition: the removal failed by its SCARD timing out, after its SREM; got {failed:?}"
    );
    // Read while the held connection is still held: only then does an
    // un-indexed session prove the SREM ran and the SCARD was the held command.
    // Had the SREM been the held one, it too would have timed out, and would
    // still run once released, leaving the same un-indexed session behind.
    assert!(
        !is_indexed(tv, sid).await,
        "precondition: the removal's SREM took {sid} out of the index before its held SCARD"
    );
    stall
        .release()
        .await
        .expect("the stalling BLPOP must end on its release");
    removal.quit().await;
    // The joiner's replica dies here, before its keeper's first renewal.
    assert!(
        !is_indexed(tv, sid).await && is_member(tv, sid, cid).await,
        "precondition: {cid} is a member of {sid}, which the failed removal left un-indexed"
    );
}

/// From a store where each `(sid, cid)` is an un-indexed member whose replica
/// is gone: once each lease has lapsed,
///
/// * a presence pass visits none of them and removes nothing (the gap);
/// * one orphan-sweep pass puts every session back in the index and leaves
///   its member set as it was;
/// * the next presence pass removes each member, announces exactly one `left`
///   for it, and takes the emptied session out of the index.
async fn backstop_recovers(tv: &TestValkey, dead: &[(String, String)]) {
    let presence = presence_leader();
    for (sid, cid) in dead {
        wait_lease_lapsed(tv, sid, cid).await;
    }
    let mut taps = Vec::new();
    for (sid, _) in dead {
        taps.push(open_tap(tv, sid).await);
    }

    let unaided = presence_pass(tv, &presence, "the presence pass before the backstop").await;
    assert_eq!(
        unaided,
        ObservedSweepPass {
            outcome: PRESENCE_SWEEP_OUTCOME_COMPLETED,
            script_calls: 0,
        },
        "the presence sweeper finds sessions only through the index, so it must not reach an \
         un-indexed member"
    );
    for ((sid, cid), tap) in dead.iter().zip(taps.iter_mut()) {
        assert!(
            is_member(tv, sid, cid).await,
            "{cid}: the dead member must survive a pass that never visits {sid}"
        );
        assert_eq!(
            lefts_per_cid(&tap.drain(tv).await),
            BTreeMap::new(),
            "{sid}: no left may be announced before the backstop"
        );
    }

    orphan_pass(tv.client(), tv, &orphan_leader()).await;
    for (sid, cid) in dead {
        assert!(
            is_indexed(tv, sid).await,
            "{sid}: one orphan-sweep pass must put the un-indexed session back in the index"
        );
        assert_eq!(
            members(tv, sid).await,
            BTreeSet::from([cid.clone()]),
            "{sid}: the backstop writes only the index, never the member set"
        );
    }

    let recovered = presence_pass(tv, &presence, "the presence pass after the backstop").await;
    assert_eq!(
        recovered.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED,
        "the presence pass after the backstop must complete"
    );
    for ((sid, cid), tap) in dead.iter().zip(taps.iter_mut()) {
        assert!(
            !is_member(tv, sid, cid).await,
            "{cid}: the re-indexed dead member must be removed by the next presence pass"
        );
        assert!(
            !is_indexed(tv, sid).await,
            "{sid}: the emptied session must leave the index again"
        );
        let mut events = tap
            .until("the swept member's left", |e| is_left(e, cid))
            .await;
        events.extend(tap.drain(tv).await);
        assert_eq!(
            lefts_per_cid(&events),
            BTreeMap::from([(cid.clone(), 1)]),
            "{sid}: exactly one left, for the swept member"
        );
    }
    for tap in taps {
        tap.close().await;
    }
}

/// **Gap 1**: a joiner whose replica died between its register and its index
/// re-add, after a concurrent teardown's index removal, is indexed again by
/// one orphan-sweep pass and then removed and announced by the presence
/// sweeper. Without the backstop the session stays un-indexed and the member
/// is never swept.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_gap_one_dead_joiner_after_a_concurrent_index_removal_is_reindexed_and_swept() {
    scenario(async {
        let tv = TestValkey::setup().await;
        leave_gap_one(&tv, "gap-one", "joiner").await;
        backstop_recovers(&tv, &[("gap-one".to_owned(), "joiner".to_owned())]).await;
    })
    .await;
}

/// **Gap 2**: a joiner whose re-add a failed index removal undid, and whose
/// replica died before its keeper's first renewal, is indexed again by one
/// orphan-sweep pass and then removed and announced by the presence sweeper.
/// Without the backstop the session stays un-indexed and the member is never
/// swept.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_gap_two_dead_joiner_after_a_failed_index_removal_is_reindexed_and_swept() {
    scenario(async {
        let tv = TestValkey::setup().await;
        leave_gap_two(&tv, "gap-two", "joiner").await;
        backstop_recovers(&tv, &[("gap-two".to_owned(), "joiner".to_owned())]).await;
    })
    .await;
}

/// **Gap 1 on the 3-shard cluster**, with one dead joiner's member set on
/// every primary other than the one that owns the session index: the
/// backstop's walk must cover every shard, and its re-add crosses slots.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_on_a_cluster_reindexes_member_sets_on_non_owner_shards_and_they_are_swept() {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    scenario(async {
        let tv = TestValkey::setup_with_url(url, 4).await;
        assert!(
            tv.client().pool.next().is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
        );
        let index_owner = owner_of(&tv, &tv.key_schema().presence_active_sessions());
        let primaries = tv
            .client()
            .pool
            .next()
            .cached_cluster_state()
            .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
            .unique_primary_nodes();
        let mut dead = Vec::new();
        for node in primaries.into_iter().filter(|n| *n != index_owner) {
            let sid = (0..)
                .map(|i| format!("gap-one-{i}"))
                .find(|sid| owner_of(&tv, &tv.key_schema().presence_members(sid)) == node)
                .expect("some session id hashes to every node");
            dead.push((sid, format!("joiner-on-{}", node.port)));
        }
        assert!(
            !dead.is_empty(),
            "a cluster test needs a primary that does not own the session index"
        );
        for (sid, cid) in &dead {
            leave_gap_one(&tv, sid, cid).await;
        }
        backstop_recovers(&tv, &dead).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Safety
// ═════════════════════════════════════════════════════════════════════════════

/// A member set the backstop's walk saw, emptied by its last member's teardown
/// before the backstop's `SADD`, is indexed with no members — the backstop
/// creates no membership — and the next presence pass takes the entry out
/// again, announcing nothing (the teardown already announced its leave).
///
/// The backstop runs on a client of its own whose maintenance connections are
/// all held with a `ConnStall` except the two its leader probe and its walk
/// take, so its `SMISMEMBER` is held (see [`RACE_MAINTENANCE_POOL`]); the
/// teardown runs while it is held, after the walk has ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_a_set_emptied_after_the_walk_is_reindexed_empty_and_unindexed_by_the_next_sweep()
{
    scenario(async {
        let tv = TestValkey::setup().await;
        let (sid, cid) = ("emptied", "leaver");
        register(&tv, sid, cid, DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
        assert!(
            !is_indexed(&tv, sid).await,
            "precondition: {sid} is not indexed"
        );

        let backstop = own_client(
            &standalone_url(),
            HELD_COMMAND_TIMEOUT_MS,
            RACE_MAINTENANCE_POOL,
        )
        .await;
        let stalls = hold_all_but_probe_and_walk(&tv, &backstop).await;
        let baseline = held_net_in(&stalls).await;
        let leader = orphan_leader();
        let (metrics, cancel, handle) = spawn_orphan(&backstop, &tv, &leader);
        wait_for_held_command(&stalls, baseline).await;
        assert!(
            !is_indexed(&tv, sid).await,
            "precondition: the backstop's check is held, so it has re-added nothing yet"
        );

        unregister_last(&tv, sid, cid).await;
        let reindexed = remove_session_from_index(tv.client(), origin(&tv), tv.key_schema(), sid)
            .await
            .expect("the teardown's index removal must succeed");
        assert!(
            !reindexed && !member_set_exists(&tv, sid).await,
            "precondition: the teardown emptied {sid} and left it out of the index"
        );
        let mut tap = open_tap(&tv, sid).await;

        release_all(stalls).await;
        finish_orphan_pass(&metrics, &leader, cancel, handle).await;
        backstop.quit().await;
        assert!(
            is_indexed(&tv, sid).await,
            "the backstop re-adds a session whose member set its walk saw, even if it has emptied \
             since"
        );
        assert!(
            !member_set_exists(&tv, sid).await,
            "the re-add must create no membership"
        );

        let pass = presence_pass(&tv, &presence_leader(), "the presence pass after the race").await;
        assert_eq!(
            pass,
            ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_COMPLETED,
                script_calls: 0,
            },
            "the pass finds no member to sweep"
        );
        assert!(
            !is_indexed(&tv, sid).await,
            "the next presence pass must take the empty session out of the index"
        );
        assert_eq!(
            tap.drain(&tv).await,
            Vec::new(),
            "nothing is announced: the teardown already announced its leave"
        );
        tap.close().await;
    })
    .await;
}

/// Un-indexed member sets in [`backstop_cancelled_while_a_chunk_is_checked_finishes_that_chunk_and_stops`]:
/// more than two chunks.
const CANCEL_SESSIONS: usize = 250;

/// The backstop's chunk: the ids one `SMISMEMBER` checks and one `SADD`
/// re-adds. Mirrors `storage::orphan_sweep::PRESENCE_REINDEX_CHUNK`, which is
/// private to that module. A change to that constant fails the mid-chunk
/// cancel test with a count mismatch that is not a cancellation bug: the two
/// must move together.
const REINDEX_CHUNK: usize = 100;

/// A shutdown that lands while the backstop checks its first chunk lets that
/// chunk finish — its `SADD` re-adds exactly one chunk — and stops the phase
/// before the next: the phase checks cancellation before each chunk, so a
/// cancelled pass does not run on through every remaining chunk.
///
/// The first chunk's `SMISMEMBER` is held as in the race test; the orphan
/// sweep is cancelled while it is held and then released.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_cancelled_while_a_chunk_is_checked_finishes_that_chunk_and_stops() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let pipeline = store_reader(&tv).pipeline();
        for i in 0..CANCEL_SESSIONS {
            let _: () = pipeline
                .sadd(tv.key_schema().presence_members(&format!("c-{i:03}")), "m")
                .await
                .expect("queuing the member SADD must succeed");
        }
        let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");

        let backstop = own_client(
            &standalone_url(),
            HELD_COMMAND_TIMEOUT_MS,
            RACE_MAINTENANCE_POOL,
        )
        .await;
        let stalls = hold_all_but_probe_and_walk(&tv, &backstop).await;
        let baseline = held_net_in(&stalls).await;
        let (_metrics, cancel, handle) = spawn_orphan(&backstop, &tv, &orphan_leader());
        wait_for_held_command(&stalls, baseline).await;

        cancel.cancel();
        release_all(stalls).await;
        tokio::time::timeout(SETTLE, handle)
            .await
            .expect("the orphan sweep must stop once cancelled")
            .expect("the orphan sweep must not panic");
        backstop.quit().await;

        let indexed: u64 = store_reader(&tv)
            .scard(tv.key_schema().presence_active_sessions())
            .await
            .expect("SCARD session index must succeed");
        assert_eq!(
            indexed, REINDEX_CHUNK as u64,
            "the chunk in flight when the cancel landed is re-added, and no later chunk is"
        );
    })
    .await;
}

/// An already-indexed session with a live member is left as it was by an
/// orphan-sweep pass and by the presence pass after it: still indexed, the
/// member still there with its lease, and nothing announced.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_leaves_an_indexed_live_session_untouched() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (sid, cid) = ("live", "keeper");
        join(&tv, sid, cid, DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
        let keys_before = presence_keys(&tv, sid).await;
        let mut tap = open_tap(&tv, sid).await;

        orphan_pass(tv.client(), &tv, &orphan_leader()).await;
        let pass = presence_pass(&tv, &presence_leader(), "the presence pass").await;

        assert_eq!(pass.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);
        assert!(is_indexed(&tv, sid).await, "{sid} must stay indexed");
        assert_eq!(
            presence_keys(&tv, sid).await,
            keys_before,
            "the member set, info hash and lease must all remain"
        );
        assert_eq!(
            members(&tv, sid).await,
            BTreeSet::from([cid.to_owned()]),
            "the live member must remain"
        );
        assert_eq!(
            tap.drain(&tv).await,
            Vec::new(),
            "nothing may be announced for a live session"
        );
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Refused commands
// ═════════════════════════════════════════════════════════════════════════════

/// Password of the restricted ACL users. Fixed: each user name is unique per
/// run and the user is deleted before the test returns.
const ACL_PASSWORD: &str = "backstop-acl-pw";

/// Deletes the test's ACL user on the way out, including on a panic: a leaked
/// user with a password would survive on the shared Valkey.
struct AclUserGuard {
    control: fred::clients::Client,
    user: String,
}

impl Drop for AclUserGuard {
    fn drop(&mut self) {
        let client = self.control.clone();
        let user = self.user.clone();
        // Same guards as `TestValkey::drop`: `block_in_place` needs a
        // multi-thread runtime, and leaking the user beats panicking in a drop.
        let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if rt_handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }
        tokio::task::block_in_place(|| {
            rt_handle.block_on(async move {
                let _ = acl(&client, vec!["DELUSER".to_owned(), user]).await;
            });
        });
    }
}

/// One `ACL` subcommand on `client`.
async fn acl(
    client: &fred::clients::Client,
    args: Vec<String>,
) -> Result<fred::types::Value, fred::error::Error> {
    client
        .custom::<fred::types::Value, String>(
            fred::types::CustomCommand::new_static("ACL", None::<u16>, false),
            args,
        )
        .await
}

/// A `ValkeyClient` authenticated as a fresh ACL user that may run everything
/// but the command `denied` (lower case, as ACL rules name it), with the guard
/// that deletes the user. The denial is confirmed by sending `probe` (the
/// command, then its arguments) as that user. `None`, after a `SKIP:` line,
/// when the store or its URL cannot host one.
async fn acl_restricted_client(
    tv: &TestValkey,
    denied: &str,
    probe: Vec<String>,
) -> Option<(AclUserGuard, ValkeyClient)> {
    let url = standalone_url();
    let Some(rest) = url.strip_prefix("redis://").filter(|r| !r.contains('@')) else {
        eprintln!(
            "SKIP: TEST_VALKEY_URL is not a bare redis://host:port, so this test cannot connect \
             as an ACL user"
        );
        return None;
    };
    let user = format!("backstop_no{denied}_{}", uuid::Uuid::new_v4().simple());
    let control = tv.client().maintenance_pool.next().clone();
    let created = acl(
        &control,
        vec![
            "SETUSER".to_owned(),
            user.clone(),
            "on".to_owned(),
            format!(">{ACL_PASSWORD}"),
            "~*".to_owned(),
            "&*".to_owned(),
            "+@all".to_owned(),
            format!("-{denied}"),
        ],
    )
    .await;
    let guard = AclUserGuard {
        control,
        user: user.clone(),
    };
    if let Err(e) = created {
        eprintln!("SKIP: `ACL SETUSER` was rejected by this Valkey ({e})");
        return None;
    }
    let client = own_client(&format!("redis://{user}:{ACL_PASSWORD}@{rest}"), 5_000, 1).await;
    let (command, args) = probe.split_first().expect("the probe names a command");
    let refused = client
        .maintenance_pool
        .next()
        .custom::<fred::types::Value, String>(
            fred::types::CustomCommand::new(command.clone(), None::<u16>, false),
            args.to_vec(),
        )
        .await;
    if refused.is_ok() {
        eprintln!(
            "SKIP: this store ran {command} for a user denied it, so the ACL is not enforced"
        );
        client.quit().await;
        return None;
    }
    Some((guard, client))
}

/// A backstop walk whose `SCAN` the store refuses is skipped: the pass ends
/// normally, no sweep error is counted (the walk's error ends neither the
/// phase nor the pass), and the un-indexed set it could not see stays
/// un-indexed until the next pass, which re-indexes it.
///
/// Every `SCAN` of the pass is refused, so the object, layer and retire-retry
/// walks are skipped too; the pass's end is still observed (the retire-retry
/// backlog gauge is a `ZCARD`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_a_refused_scan_is_skipped_and_the_next_pass_reindexes() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let Some((guard, refused)) =
            acl_restricted_client(&tv, "scan", vec!["SCAN".to_owned(), "0".to_owned()]).await
        else {
            return;
        };
        let (sid, cid) = ("unwalked", "member");
        register(&tv, sid, cid, DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
        let leader = orphan_leader();

        let metrics = orphan_pass(&refused, &tv, &leader).await;
        refused.quit().await;
        assert_eq!(
            gc_sweep_batch_errors(&metrics),
            Vec::new(),
            "a refused SCAN is skipped: it must not fail the backstop phase or the pass"
        );
        assert!(
            !is_indexed(&tv, sid).await,
            "the refused walk saw no member set, so it re-indexed nothing"
        );

        orphan_pass(tv.client(), &tv, &leader).await;
        assert!(
            is_indexed(&tv, sid).await,
            "the next pass, whose walk succeeds, must re-index {sid}"
        );
        drop(guard);
    })
    .await;
}

/// A backstop whose re-add `SADD` the store refuses ends the phase with an
/// error that is counted exactly once on `gc_sweep_batch_errors_total`; the
/// session stays un-indexed (for the next pass to retry), and the object,
/// layer and retire-retry phases still run in the same pass.
///
/// The refusal comes from an ACL user denied `SADD` alone, so the walk and the
/// `SMISMEMBER` before it succeed and the failure is the `SADD`'s own: a
/// backstop that swallowed it would count nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backstop_a_refused_reindex_is_counted_once_and_the_later_phases_still_run() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let Some((guard, refused)) = acl_restricted_client(
            &tv,
            "sadd",
            vec![
                "SADD".to_owned(),
                format!("{}acl-probe", tv.prefix()),
                "x".to_owned(),
            ],
        )
        .await
        else {
            return;
        };
        let (sid, cid) = ("unindexed", "member");
        register(&tv, sid, cid, DEFAULT_PRESENCE_LEASE_TTL_SECS).await;
        // Work for the object and layer phases: an orphaned object (a header
        // naming no layer) and an orphaned layer mapping (no such session).
        // The retire-retry phase's work is the backlog gauge the pass's end is
        // observed by.
        let ks = tv.key_schema();
        let orphan_header = ks.obj_header("940301");
        let _: () = store_reader(&tv)
            .hset(&orphan_header, [("seq", "1")])
            .await
            .expect("seeding the orphaned header must succeed");
        let orphan_mapping = ks.layer_mapping("no-such-session", "layer");
        let _: () = store_reader(&tv)
            .set(&orphan_mapping, "940302", None, None, false)
            .await
            .expect("seeding the orphaned layer mapping must succeed");

        let metrics = orphan_pass(&refused, &tv, &orphan_leader()).await;
        refused.quit().await;

        let errors = gc_sweep_batch_errors(&metrics);
        assert_eq!(
            errors.iter().map(|(_, count)| count).sum::<u64>(),
            1,
            "the refused SADD is counted exactly once; got {errors:?}"
        );
        assert!(
            !is_indexed(&tv, sid).await,
            "the refused re-add leaves {sid} un-indexed"
        );
        assert!(
            !store_reader(&tv)
                .exists::<bool, _>(&orphan_header)
                .await
                .expect("EXISTS must succeed"),
            "the object phase must still run after the backstop failed"
        );
        assert!(
            !store_reader(&tv)
                .exists::<bool, _>(&orphan_mapping)
                .await
                .expect("EXISTS must succeed"),
            "the layer phase must still run after the backstop failed"
        );
        drop(guard);
    })
    .await;
}
