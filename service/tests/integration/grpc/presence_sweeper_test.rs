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

// The presence sweeper's bounded, fenced pass (SDD SR-3), driven through the
// real sweeper task against a real Valkey:
//
//   * leadership handoff — a leader whose batch is held in flight past its
//     lock's expiry makes no further sweep call once released, and the new
//     leader removes and announces every stale member exactly once (PRD AC-1);
//   * bounds and coverage — one presence command in flight, at most
//     `member_batch` ids per script call, pages rather than whole sets, and
//     every stale member removed, every live one kept and every empty session
//     unindexed over repeated passes, through joins, leaves and a failed batch
//     (PRD AC-2);
//   * duplicate enumeration — a member page delivered twice in one pass (by
//     the seam's replay, standing in for `SSCAN`'s repeats) re-runs an
//     idempotent chunk (SR-3.2);
//   * per-session error isolation — a timed-out chunk or member page ends its
//     session, not the pass, and a timed-out index removal does not stop the
//     pass either (SR-3.4);
//   * event order — every `left` of a swept chunk reaches a peer before the
//     `joined` of a keeper that rejoins the moment the chunk returns (PRD
//     FR-6a);
//   * the index race — a sweep's index removal landing between a join's first
//     index add and its register (SR-3.8).
//
// The pass-level unit tests — renewal, a lost lease, cancellation, a failed
// index read and the sweep metrics — are in `src/presence/sweeper.rs`, as are
// those of the seam's member-page replay.
//
// ── How a pass is held and observed ──────────────────────────────────────────
//
// Every sweeper here is spawned with a `PresenceSweeperSeam`: its one-shot
// `before_chunk` gate parks a pass before a chunk's cancel check, lease check
// and script, its one-shot member-page replay delivers a page's ids a second
// time, and its observations record every page, script call and pass.
// A sweeper whose data-pool commands a test holds runs on a `ValkeyClient` of
// its own, so the test knows which pooled connection the pass's next command
// takes (fred's pool is round-robin, and nothing else uses that client) and
// can hold exactly that one with a `ConnStall`. On a cluster a pooled client
// holds one connection per node, so that connection is held on every node.
//
// Peers' views are read with `Tap`, an independent subscriber whose message
// channel is sized so it cannot lag, proven complete with a fence (see
// `super::helpers`).
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every wait is bounded: a seam latch, a pass count, a store condition polled
// every 20 ms (the event-order test polls its chunk's removal back to back, to
// release the rejoins the moment it lands), or an event read, each within
// `SETTLE` (10 s); the healthy path returns from each in milliseconds, except:
//
//   * a lease to expire — the handoff's ~1.5 s lock TTL;
//   * a held command's timeout — a chunk's script, a member page or an index
//     removal held past the sweeper client's 1 s command timeout;
//   * a sweep interval — `COVERAGE_INTERVAL` (3 s) between the passes of the
//     coverage, duplicate-enumeration and error-isolation tests, longer than
//     any of their passes: the coverage, duplicate-enumeration, member-page
//     and index-removal tests stop their sweeper once its pass ends, before
//     the next is due, so no later pass overlaps what the test does next; the
//     chunk-isolation test keeps its sweeper for a second pass, and between
//     its two passes asserts only what a later pass cannot undo.
//     `SWEEP_INTERVAL` (100 ms) elsewhere.
//
// Each test body runs under one `SCENARIO_BUDGET` (100 s) deadline, below
// nextest's hang-killer. The heaviest healthy path, the coverage test, takes
// about 10 s.
//
// The whole file is compiled only with `--features test-fault-injection` (it
// drives the sweeper and JoinSession seams), the cluster variants included.
// Those self-gate on `LIVE_EDIT_TEST_CLUSTER_URL` and print a `SKIP:` line
// when it is unset.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, ClusterInterface, HashesInterface, KeysInterface, SetsInterface,
};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use live_edit::grpc::presence::{JoinRegistrationSeam, LeaseKeeperSeam};
use live_edit::grpc::proto::join_session_response::ResponseType;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, PRESENCE_SWEEP_OUTCOME_COMPLETED, PRESENCE_SWEEP_OUTCOME_LEASE_LOST,
    PUBSUB_RELAY_PRESENCE,
};
use live_edit::presence::{
    spawn_presence_sweeper_with_seam, ObservedSweepPass, PresenceEvent, PresenceSweeperSeam,
    SweeperBudget,
};
use live_edit::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::presence::store_time_ms;
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, connect_test_client, release_all, ConnStall,
    TestValkey,
};

use super::helpers::*;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on any single wait.
const SETTLE: Duration = PRESENCE_WAIT;

/// The sweep interval of a test that lets passes follow one another.
const SWEEP_INTERVAL: Duration = Duration::from_millis(100);

/// The sweep interval of the coverage, duplicate-enumeration and
/// error-isolation tests: a sweeper's first pass starts one interval after the
/// spawn, and the interval is longer than any of their passes (the longest
/// waits out a 1 s command timeout).
///
/// The tests that stop a sweeper once its pass ends — coverage,
/// duplicate-enumeration, member-page and index-removal — therefore stop it
/// before its next tick is due. A pass
/// that outran the interval would be followed at once by the next
/// (`MissedTickBehavior::Delay`); `stop_after_one_pass` fails rather than let
/// that pass race the test.
///
/// The chunk-isolation test keeps its sweeper running and waits for its
/// second pass, one interval after the first began. Between the two it reads
/// the store only at a parked chunk or for what a later pass cannot undo (the
/// next session emptied and unindexed).
const COVERAGE_INTERVAL: Duration = Duration::from_secs(3);

/// The handoff's lock TTL: short, so the stalled leader's lock lapses inside
/// the test, and long enough that the first leader reaches its held batch
/// before its lease's own deadline.
const HANDOFF_LOCK_TTL: Duration = Duration::from_millis(1_500);

/// The data-pool command timeout of a sweeper whose command — a chunk's
/// script, a member page or an index removal — a test fails by holding it.
/// Under the lock TTL of every such test, so the failure is the timeout and
/// never a lost lease.
const FAILING_COMMAND_TIMEOUT_MS: u64 = 1_000;

/// The data-pool connections of the sweeper client in the coverage,
/// member-page and index-removal tests: more than the commands a pass sends
/// while one connection is held — after a failed chunk until the next chunk
/// parks (coverage), or for the rest of the pass (member page, index removal)
/// — so the round-robin never sends a second command to the held connection.
const COVERAGE_POOL: u32 = 16;

/// The data-pool command timeout of a sweeper whose command a test holds and
/// then releases: far longer than any hold, so the held command never times
/// out.
const HELD_COMMAND_TIMEOUT_MS: u64 = 10_000;

/// How long before the seed a stale member's recorded lease expired.
const STALE_FOR_MS: u64 = 5_000;

/// A seeded live member's lease, in seconds: far past any test.
const LIVE_LEASE_SECS: i64 = 600;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The URL `TestValkey::setup` connects to.
fn standalone_url() -> String {
    std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// A `ValkeyClient` of its own over the store at `url` — its own data,
/// maintenance and subscriber pools — with a command timeout of
/// `command_timeout_ms` and `pool_size` data-pool connections.
async fn own_client(url: &str, command_timeout_ms: u64, pool_size: u32) -> ValkeyClient {
    let config = live_edit::config::Config {
        valkey_url: url.to_owned(),
        valkey_command_timeout_ms: command_timeout_ms,
        valkey_pool_size: pool_size,
        valkey_connect_timeout_secs: 30,
        ..crate::common::test_config()
    };
    connect_test_client(&config)
        .await
        .expect("the sweeper's own ValkeyClient must connect")
}

/// How [`seed`] writes one member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seed {
    /// A lease ([`LIVE_LEASE_SECS`]) and an info hash recording an expiry that
    /// far ahead: a connection whose replica is alive.
    Live,
    /// No lease, and an info hash recording a lease that expired
    /// [`STALE_FOR_MS`] ago: a connection whose replica crashed.
    Stale,
    /// No lease and no info hash.
    StaleUnrecorded,
}

/// Write `members` into `sid`'s presence state straight through the store, as
/// joins on other replicas would have left them, and index the session. One
/// pipeline on the session's slot, then the index add. Nothing is published.
async fn seed(tv: &TestValkey, sid: &str, members: &[(String, Seed)]) {
    let ks = tv.key_schema();
    let now_ms = store_time_ms(
        tv.client(),
        FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
        ks,
        sid,
    )
    .await
    .expect("store_time_ms must succeed");
    if !members.is_empty() {
        let pipeline = store_reader(tv).pipeline();
        let cids: Vec<String> = members.iter().map(|(cid, _)| cid.clone()).collect();
        let _: () = pipeline
            .sadd(ks.presence_members(sid), cids)
            .await
            .expect("queuing SADD must succeed");
        for (cid, kind) in members {
            let expires_at_ms = match kind {
                Seed::Live => Some(now_ms + (LIVE_LEASE_SECS as u64) * 1000),
                Seed::Stale => Some(now_ms - STALE_FOR_MS),
                Seed::StaleUnrecorded => None,
            };
            if let Some(expires_at_ms) = expires_at_ms {
                let _: () = pipeline
                    .hset(
                        ks.presence_user(sid, cid),
                        vec![
                            ("user_name", format!("User {cid}")),
                            ("user_application", "seeded-app".to_owned()),
                            ("lease_expires_at_ms", expires_at_ms.to_string()),
                        ],
                    )
                    .await
                    .expect("queuing HSET must succeed");
            }
            if *kind == Seed::Live {
                let _: () = pipeline
                    .set(
                        ks.presence_lease(sid, cid),
                        "1",
                        Some(fred::types::Expiration::EX(LIVE_LEASE_SECS)),
                        None,
                        false,
                    )
                    .await
                    .expect("queuing SET lease must succeed");
            }
        }
        let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");
    }
    index(tv, sid).await;
}

/// Add `sid` to the session index.
async fn index(tv: &TestValkey, sid: &str) {
    let _: i64 = store_reader(tv)
        .sadd(tv.key_schema().presence_active_sessions(), sid)
        .await
        .expect("SADD active_sessions must succeed");
}

/// Give `cid` a lease, as its keeper's renewal would.
async fn renew(tv: &TestValkey, sid: &str, cid: &str) {
    let _: () = store_reader(tv)
        .set(
            tv.key_schema().presence_lease(sid, cid),
            "1",
            Some(fred::types::Expiration::EX(LIVE_LEASE_SECS)),
            None,
            false,
        )
        .await
        .expect("SET lease must succeed");
}

/// `sid`'s member set.
async fn members(tv: &TestValkey, sid: &str) -> BTreeSet<String> {
    store_reader(tv)
        .smembers(tv.key_schema().presence_members(sid))
        .await
        .expect("SMEMBERS must succeed")
}

/// One `SSCAN` page, read as a pass reads it.
async fn sscan_page(
    tv: &TestValkey,
    key: &str,
    cursor: &str,
    count: usize,
) -> (String, Vec<String>) {
    store_reader(tv)
        .custom(
            fred::types::CustomCommand::new_static(
                "SSCAN",
                fred::types::ClusterHash::FirstKey,
                false,
            ),
            vec![
                key.to_owned(),
                cursor.to_owned(),
                "COUNT".to_owned(),
                count.to_string(),
            ],
        )
        .await
        .expect("SSCAN must succeed")
}

/// `n` connection ids `{prefix}-NNN`.
fn cids(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}-{i:03}")).collect()
}

/// A budget with the given lock TTL, page sizes and chunk size.
fn budget(
    lock_ttl: Duration,
    session_page: usize,
    member_page: usize,
    member_batch: usize,
) -> SweeperBudget {
    SweeperBudget {
        lock_ttl,
        session_page,
        member_page,
        member_batch,
    }
}

/// A presence sweeper task with its seam and its own metrics (installed as the
/// observability hooks of the client it runs on).
struct Sweep {
    seam: Arc<PresenceSweeperSeam>,
    metrics: Arc<Metrics>,
    leader: LeaderIdentity,
    cancel: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl Sweep {
    /// Spawn a sweeper as `label` on `client` over `tv`'s keyspace. With
    /// `park_first_chunk`, its chunk gate is armed before the spawn, so its
    /// first pass parks at that pass's first chunk.
    async fn spawn(
        tv: &TestValkey,
        client: &ValkeyClient,
        label: &str,
        budget: SweeperBudget,
        interval: Duration,
        park_first_chunk: bool,
    ) -> Self {
        let seam = PresenceSweeperSeam::new();
        if park_first_chunk {
            seam.before_chunk.arm();
        }
        Self::spawn_with_seam(tv, client, label, budget, interval, seam).await
    }

    /// Spawn a sweeper as [`Self::spawn`] does, on `seam` as the test set it
    /// up before the spawn (its gate or its member-page replay armed).
    async fn spawn_with_seam(
        tv: &TestValkey,
        client: &ValkeyClient,
        label: &str,
        budget: SweeperBudget,
        interval: Duration,
        seam: PresenceSweeperSeam,
    ) -> Self {
        let metrics = Arc::new(crate::common::metrics_scrape_only(label));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let seam = Arc::new(seam);
        let leader = crate::common::test_leader(label);
        let cancel = CancellationToken::new();
        let handle = spawn_presence_sweeper_with_seam(
            client.clone(),
            tv.key_schema().clone(),
            leader.clone(),
            budget,
            Arc::clone(&metrics),
            interval,
            cancel.clone(),
            Arc::clone(&seam),
        );
        Sweep {
            seam,
            metrics,
            leader,
            cancel,
            handle,
        }
    }

    async fn parked(&self, what: &str) {
        wait_parked(&self.seam.before_chunk, what).await;
    }

    /// Wait for at least `n` leader passes to have ended; all of them.
    async fn passes(&self, n: usize, what: &str) -> Vec<ObservedSweepPass> {
        tokio::time::timeout(SETTLE, self.seam.wait_for_passes(n))
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{what}: {n} passes did not end within {SETTLE:?}; ended: {:?}",
                    self.seam.passes()
                )
            })
    }

    /// Wait until the sweeper has issued `n` script calls.
    async fn script_calls(&self, n: usize, what: &str) {
        let seam = &self.seam;
        poll_until(what, SETTLE, move || async move {
            seam.script_batch_sizes().len() >= n
        })
        .await;
    }

    /// Stop the task and wait for it to end: release any parked chunk (a
    /// parked arrival does not see the cancel), then cancel.
    async fn join(self) {
        self.seam.before_chunk.release();
        self.cancel.cancel();
        tokio::time::timeout(SETTLE, self.handle)
            .await
            .expect("the sweeper must stop once cancelled")
            .expect("the sweeper task must not panic");
    }
}

/// Stop `sweep` once its first pass has ended, check it ran no other — its
/// next tick was not due, so nothing it did overlaps what the test does next
/// — and hand back its seam.
async fn stop_after_one_pass(sweep: Sweep, what: &str) -> Arc<PresenceSweeperSeam> {
    let seam = Arc::clone(&sweep.seam);
    sweep.join().await;
    assert_eq!(
        seam.passes().len(),
        1,
        "{what}: the sweeper must be stopped before its next pass; it ran {:?} (a pass outran \
         the sweep interval)",
        seam.passes()
    );
    seam
}

/// `live_edit_presence_sweep_lease_total{outcome}` on a sweeper's metrics.
fn lease_total(metrics: &Metrics, outcome: &str) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_presence_sweep_lease_total",
        &[("outcome", outcome)],
    )
    .expect("every sweep lease cell is seeded")
}

/// `live_edit_presence_stale_removed_total` on a sweeper's metrics.
fn stale_removed(metrics: &Metrics) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_presence_stale_removed_total",
        &[],
    )
    .expect("the stale-removed cell is seeded")
}

/// The pooled data connection `ahead` commands after `client`'s last: the one
/// its `ahead`-th next `pool.next()` returns (fred's pool is round-robin).
fn data_conn_ahead(client: &ValkeyClient, ahead: usize) -> &fred::clients::Client {
    let pool = &client.pool;
    let last = pool.last().id();
    let clients = pool.clients();
    let at = clients
        .iter()
        .position(|c| c.id() == last)
        .expect("pool.last() is one of the pool's clients");
    pool.member((at + ahead) % clients.len())
}

/// A standalone client on one cluster node.
async fn node_client(node: &fred::types::config::Server) -> fred::clients::Client {
    let url = format!("redis://{}:{}", node.host, node.port);
    let config = fred::types::config::Config::from_url(&url)
        .unwrap_or_else(|e| panic!("{url} must parse as a fred config: {e:?}"));
    let client = fred::types::Builder::from_config(config)
        .build()
        .expect("building the node client must succeed");
    client.init().await.expect("the node client must connect");
    client
}

/// Hold `conn` with a [`ConnStall`], so the next command sent on it gets no
/// reply until the stall is released. On a cluster `conn` has a connection to
/// every primary, and the command goes to whichever node owns its slot, so
/// every one of them is held, each observed from a standalone connection to
/// its node (`CLIENT LIST` and the releasing `LPUSH` must reach that node).
async fn stall_conn(tv: &TestValkey, conn: &fred::clients::Client, tag: &str) -> Vec<ConnStall> {
    let key = |i: usize| format!("{}conn-stall:{tag}:{i}", tv.prefix());
    if !conn.is_clustered() {
        return vec![ConnStall::park(conn, store_reader(tv), key(0)).await];
    }
    let routing = conn
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    let mut stalls = Vec::new();
    for node in routing.unique_primary_nodes() {
        let on_node = (0..)
            .map(key)
            .find(|k| routing.get_server(fred::util::redis_keyslot(k.as_bytes())) == Some(&node))
            .expect("some key hashes to every node");
        let observer = node_client(&node).await;
        stalls.push(ConnStall::park(conn, &observer, on_node).await);
    }
    stalls
}

/// Every byte the stalled connections have read so far, summed.
async fn net_in(stalls: &[ConnStall]) -> u64 {
    let mut total = 0;
    for stall in stalls {
        total += stall
            .net_in()
            .await
            .expect("a stalled connection must still exist");
    }
    total
}

/// A cluster primary that does not own `sid`'s slot: where a peer whose events
/// cross the cluster bus is attached.
fn node_other_than_sid_owner(tv: &TestValkey, sid: &str) -> fred::types::config::Server {
    let routing = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    let owner = routing
        .get_server(fred::util::redis_keyslot(
            tv.key_schema().presence_members(sid).as_bytes(),
        ))
        .expect("every slot has a primary")
        .clone();
    routing
        .unique_primary_nodes()
        .into_iter()
        .find(|n| *n != owner)
        .expect("a cluster test needs at least two primaries")
}

/// A tap on `sid`: on a cluster, attached to a node that does not own `sid`'s
/// slot.
async fn open_tap(tv: &TestValkey, sid: &str) -> Tap {
    if tv.client().pool.next().is_clustered() {
        let node = node_other_than_sid_owner(tv, sid);
        Tap::open_on_node(tv, sid, &node).await
    } else {
        Tap::open(tv, sid).await
    }
}

/// `TestValkey` on the 3-shard cluster at `LIVE_EDIT_TEST_CLUSTER_URL`, with
/// its URL, or `None` (after printing the `SKIP:` line) when it is unset.
async fn cluster() -> Option<(TestValkey, String)> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return None;
    };
    let tv = TestValkey::setup_with_url(url.clone(), 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
    );
    Some((tv, url))
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

/// Every event on `tap` up to the `lefts`-th `PeerLeft`, and then everything
/// else published before a fence: the events a sweep announced, proven
/// complete on a standalone store and on a cluster alike.
async fn read_lefts(tv: &TestValkey, tap: &mut Tap, lefts: usize) -> Vec<PresenceEvent> {
    let mut events = tap
        .until_count("PeerLeft events", lefts, |e| {
            matches!(e, PresenceEvent::PeerLeft { .. })
        })
        .await;
    events.extend(tap.drain(tv).await);
    events
}

// ═════════════════════════════════════════════════════════════════════════════
// Leadership handoff (PRD AC-1)
// ═════════════════════════════════════════════════════════════════════════════

/// Sessions in the handoff test.
const HANDOFF_SESSIONS: usize = 2;

/// Members per handoff session: three chunks of `member_batch = 2`, so the
/// chunk the first leader is held in is never its session's last.
const HANDOFF_MEMBERS: usize = 5;

/// Sweeper A leads and parks at its first chunk. A member of that chunk gets
/// its lease back (its keeper renewed it after A read the page), and A's
/// script for the chunk is held in flight on its data connection. A's lock
/// lapses, B takes it and completes a pass. Released, A's held script removes
/// nothing (B removed the chunk's stale member; the renewed one has a lease),
/// and A calls no further script: its next chunk's lease check finds the lease
/// lost. Every seeded stale member is removed by B, exactly one `left` per
/// removed cid reaches a peer, and the renewed member stays.
async fn handoff(tv: &TestValkey, url: &str) {
    let ks = tv.key_schema();
    let sessions: Vec<String> = (0..HANDOFF_SESSIONS)
        .map(|i| format!("handoff-{i}"))
        .collect();
    for sid in &sessions {
        let seeded: Vec<(String, Seed)> = cids(sid, HANDOFF_MEMBERS)
            .into_iter()
            .map(|cid| (cid, Seed::Stale))
            .collect();
        seed(tv, sid, &seeded).await;
    }
    let mut taps = Vec::new();
    for sid in &sessions {
        taps.push(open_tap(tv, sid).await);
    }
    let handoff_budget = budget(HANDOFF_LOCK_TTL, 10, 10, 2);

    let a_client = own_client(url, HELD_COMMAND_TIMEOUT_MS, 4).await;
    let a = Sweep::spawn(
        tv,
        &a_client,
        "replica-A",
        handoff_budget,
        SWEEP_INTERVAL,
        true,
    )
    .await;
    a.parked("A's first chunk").await;

    // A's pages are compactly stored sets, which SSCAN returns whole and in
    // order, so the same reads give the session and chunk A holds.
    let (_, index_page) = sscan_page(tv, &ks.presence_active_sessions(), "0", 10).await;
    let parked_sid = index_page[0].clone();
    let (_, member_page) = sscan_page(tv, &ks.presence_members(&parked_sid), "0", 10).await;
    assert_eq!(
        (a.seam.session_page_sizes(), a.seam.member_page_sizes()),
        (vec![HANDOFF_SESSIONS], vec![HANDOFF_MEMBERS]),
        "precondition: A read whole compact pages, as the test did"
    );
    let renewed = member_page[0].clone();
    renew(tv, &parked_sid, &renewed).await;

    let stalls = stall_conn(tv, data_conn_ahead(&a_client, 1), "a-batch").await;
    let baseline = net_in(&stalls).await;
    a.seam.before_chunk.release();
    let stalls_ref = &stalls;
    poll_until(
        "A's batch reaching its held connection",
        SETTLE,
        move || async move { net_in(stalls_ref).await > baseline },
    )
    .await;
    assert_eq!(
        a.seam.script_batch_sizes(),
        vec![2],
        "precondition: A sent its parked chunk's script, and only it"
    );

    let b = Sweep::spawn(
        tv,
        tv.client(),
        "replica-B",
        handoff_budget,
        SWEEP_INTERVAL,
        false,
    )
    .await;
    let lock_key = ks.presence_sweeper_leader_lock();
    let (b_value, lock_ref) = (b.leader.lock_value(), &lock_key);
    poll_until("A's lock expiring and B holding it", SETTLE, || {
        let b_value = b_value.clone();
        async move {
            let holder: Option<String> = store_reader(tv)
                .get(lock_ref)
                .await
                .expect("GET lock must succeed");
            holder == Some(b_value)
        }
    })
    .await;
    let b_first = b.passes(1, "B's first pass").await;
    assert_eq!(b_first[0].outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);

    release_all(stalls).await;
    let a_passes = a.passes(1, "A's held pass").await;
    let (a_seam, a_metrics) = (Arc::clone(&a.seam), Arc::clone(&a.metrics));
    a.join().await;
    assert_eq!(
        a_passes,
        vec![ObservedSweepPass {
            outcome: PRESENCE_SWEEP_OUTCOME_LEASE_LOST,
            script_calls: 1,
        }],
        "released after losing its lease, A must end its pass before another script"
    );
    assert_eq!(
        a_seam.script_batch_sizes().len(),
        1,
        "A must make no sweep call after its held one, in that pass or a later tick"
    );
    assert_eq!(
        lease_total(&a_metrics, "lost"),
        1.0,
        "A's pass ended on one lost lease"
    );
    let removed_by_b = stale_removed(&b.metrics);
    b.join().await;

    let stale: BTreeSet<String> = sessions
        .iter()
        .flat_map(|sid| cids(sid, HANDOFF_MEMBERS))
        .filter(|cid| *cid != renewed)
        .collect();
    for sid in &sessions {
        let left = members(tv, sid).await;
        let want: BTreeSet<String> = if *sid == parked_sid {
            [renewed.clone()].into()
        } else {
            BTreeSet::new()
        };
        assert_eq!(
            left, want,
            "{sid}: every stale member removed, the renewed one kept"
        );
    }
    assert_eq!(
        removed_by_b,
        stale.len() as f64,
        "B removed every stale member"
    );

    let mut announced = BTreeMap::new();
    for (sid, tap) in sessions.iter().zip(taps.iter_mut()) {
        let expected = cids(sid, HANDOFF_MEMBERS)
            .into_iter()
            .filter(|cid| *cid != renewed)
            .count();
        announced.extend(lefts_per_cid(&read_lefts(tv, tap, expected).await));
    }
    let want: BTreeMap<String, usize> = stale.iter().map(|cid| (cid.clone(), 1)).collect();
    assert_eq!(
        announced, want,
        "each removed cid has exactly one left, and the renewed member none"
    );
    for tap in taps {
        tap.close().await;
    }
}

/// **PRD AC-1**, standalone store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_handoff_stalled_leader_stops_and_the_new_leader_removes_each_stale_member_once() {
    scenario(async {
        let tv = TestValkey::setup().await;
        handoff(&tv, &standalone_url()).await;
    })
    .await;
}

/// **PRD AC-1** on the 3-shard cluster: the peers are attached to a node that
/// owns no session's slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_handoff_on_a_cluster_stalled_leader_stops_and_each_stale_member_is_announced_once()
{
    let Some((tv, url)) = cluster().await else {
        return;
    };
    scenario(handoff(&tv, &url)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Bounds and coverage (PRD AC-2)
// ═════════════════════════════════════════════════════════════════════════════

/// Sessions of [`COMPACT_MEMBERS`] members.
const COMPACT_SESSIONS: usize = 20;

/// Members of each compact session: a set Valkey stores compactly, which
/// `SSCAN` returns whole whatever its `COUNT`.
const COMPACT_MEMBERS: usize = 7;

/// Members of the large session: above the 128 a set may hold compactly
/// (`set-max-listpack-entries`), so `SSCAN` pages it.
const LARGE_MEMBERS: usize = 150;

/// Indexed sessions with no member set. Enough that the session index itself
/// holds more than 128 entries, so `SSCAN` pages it too.
const EMPTY_SESSIONS: usize = 113;

/// The coverage tests' budget: `session_page = 3`, `member_batch = 2`.
fn coverage_budget() -> SweeperBudget {
    budget(Duration::from_secs(30), 3, 5, 2)
}

/// What one compact session's member `j` is: a mix of live and stale.
fn compact_kind(i: usize, j: usize) -> Seed {
    match (i + j) % 3 {
        0 => Seed::Live,
        1 => Seed::Stale,
        _ => Seed::StaleUnrecorded,
    }
}

/// The store a coverage run must end in.
#[derive(Default)]
struct Expected {
    /// Members that must remain, per session.
    live: BTreeMap<String, BTreeSet<String>>,
    /// Members that must be gone, per session.
    stale: BTreeMap<String, BTreeSet<String>>,
    /// Sessions that must have left the index.
    empty: BTreeSet<String>,
}

impl Expected {
    fn add(&mut self, sid: &str, seeded: &[(String, Seed)]) {
        let live = self.live.entry(sid.to_owned()).or_default();
        let stale = self.stale.entry(sid.to_owned()).or_default();
        for (cid, kind) in seeded {
            if *kind == Seed::Live {
                live.insert(cid.clone());
            } else {
                stale.insert(cid.clone());
            }
        }
    }

    /// Sessions that still hold a stale member.
    async fn sessions_with_stale(&self, tv: &TestValkey) -> Vec<String> {
        let mut found = Vec::new();
        for (sid, stale) in &self.stale {
            if !members(tv, sid).await.is_disjoint(stale) {
                found.push(sid.clone());
            }
        }
        found
    }

    /// Every live member remains, every stale one is gone, every session with
    /// a live member is indexed, and every empty one is not.
    async fn assert_reached(&self, tv: &TestValkey, what: &str) {
        for (sid, live) in &self.live {
            let now = members(tv, sid).await;
            assert_eq!(
                &now, live,
                "{what}: {sid} must hold exactly its live members"
            );
            assert_eq!(
                is_indexed(tv, sid).await,
                !live.is_empty(),
                "{what}: {sid} must be indexed exactly while it has members"
            );
        }
        for sid in &self.empty {
            assert!(
                !is_indexed(tv, sid).await,
                "{what}: the empty session {sid} must have left the index"
            );
        }
    }
}

/// With `session_page = 3` and `member_batch = 2`, over 20 sessions × 7
/// members (a mix of live and stale), a 150-member session and 113 indexed
/// sessions with no member set:
///
/// * pass 1 removes every stale member, keeps every live one and unindexes
///   every empty session;
/// * between passes 1 and 2, stale members appear in every session, a new
///   session appears with stale and live members, two connections join and
///   one joins and leaves;
/// * pass 2's first chunk is held on its data connection until it times out —
///   one data-pool timeout, so the failure surfaced — which ends that
///   session's iteration but not the pass: pass 2 completes and leaves stale
///   members in that one session only;
/// * between passes 2 and 3 an empty session is indexed, and pass 3 then
///   leaves every stale member removed, every live one in place and every
///   empty session out of the index;
///
/// and across all of it the seams show one presence command in flight at a
/// time, at most two ids per script call (the seven-member sets come back
/// whole and are still chunked), and pages, never whole sets, of the
/// 150-member set and of the index.
///
/// Each pass runs on a sweeper of its own (the same leader identity, so each
/// takes the lock at once), stopped as soon as its pass ends, so nothing the
/// test does between passes races a pass; the bounds are checked over all
/// three seams.
///
/// The failure is a command timeout rather than `ConnStall::kill`: fred
/// re-sends a command in flight on a dropped connection, so a kill would
/// usually re-run the batch and surface nothing. It is placed in pass 2, whose
/// index holds no empty session, so the pass reaches its next chunk within a
/// few commands — before its round-robin returns to the held connection,
/// which would time out a second command.
async fn coverage(tv: &TestValkey, url: &str) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let feed = spawn_router_feed(tv, &svc).await;
    let mut expected = Expected::default();

    let mut sessions = Vec::new();
    for i in 0..COMPACT_SESSIONS {
        let (sid, _token) = create_session(tv, &svc, &format!("coverage-{i:02}"), true).await;
        let seeded: Vec<(String, Seed)> = cids(&format!("c{i:02}"), COMPACT_MEMBERS)
            .into_iter()
            .enumerate()
            .map(|(j, cid)| (cid, compact_kind(i, j)))
            .collect();
        seed(tv, &sid, &seeded).await;
        expected.add(&sid, &seeded);
        sessions.push(sid);
    }
    let (large, _token) = create_session(tv, &svc, "coverage-large", true).await;
    let seeded: Vec<(String, Seed)> = cids("large", LARGE_MEMBERS)
        .into_iter()
        .enumerate()
        .map(|(j, cid)| (cid, if j % 2 == 0 { Seed::Live } else { Seed::Stale }))
        .collect();
    seed(tv, &large, &seeded).await;
    expected.add(&large, &seeded);
    sessions.push(large);
    for k in 0..EMPTY_SESSIONS {
        let sid = format!("coverage-empty-{k:03}");
        index(tv, &sid).await;
        expected.empty.insert(sid);
    }
    let index_size = COMPACT_SESSIONS + 1 + EMPTY_SESSIONS;

    // Each pass runs on a sweeper of its own, stopped as soon as that pass
    // ends, so no pass overlaps what the test does between passes.
    let spawn_pass = |park_first_chunk: bool| async move {
        let client = own_client(url, FAILING_COMMAND_TIMEOUT_MS, COVERAGE_POOL).await;
        let sweep = Sweep::spawn(
            tv,
            &client,
            "coverage",
            coverage_budget(),
            COVERAGE_INTERVAL,
            park_first_chunk,
        )
        .await;
        (client, sweep)
    };

    // ── Pass 1 ──────────────────────────────────────────────────────────────
    let (_client_1, sweep_1) = spawn_pass(false).await;
    let pass_1 = sweep_1.passes(1, "pass 1").await[0];
    let seam_1 = stop_after_one_pass(sweep_1, "pass 1").await;
    assert_eq!(pass_1.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);
    expected.assert_reached(tv, "after pass 1").await;

    // ── Between passes 1 and 2: stale members appear, joins and a leave ─────
    for (n, sid) in sessions.iter().enumerate() {
        let late_stale: Vec<(String, Seed)> = cids(&format!("late-stale-{n:02}"), 3)
            .into_iter()
            .map(|cid| (cid, Seed::Stale))
            .collect();
        seed(tv, sid, &late_stale).await;
        expected.add(sid, &late_stale);
    }
    let late = "coverage-late".to_owned();
    let late_seeded = vec![
        ("late-0".to_owned(), Seed::Stale),
        ("late-1".to_owned(), Seed::StaleUnrecorded),
        ("late-2".to_owned(), Seed::Stale),
        ("late-3".to_owned(), Seed::Live),
    ];
    seed(tv, &late, &late_seeded).await;
    expected.add(&late, &late_seeded);
    let mut kept_streams = Vec::new();
    for sid in &sessions[..2] {
        let (mut stream, tx) = open_join(&svc, sid, "coverage-joiner").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        expected.live.get_mut(sid).expect("seeded").insert(cid);
        kept_streams.push((stream, tx));
    }
    {
        let (mut stream, tx) = open_join(&svc, &sessions[2], "coverage-leaver").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        drop(tx);
        read_to_end(&mut stream, "the leaving join").await;
        let (sid, cid) = (&sessions[2], cid.as_str());
        poll_until(
            "the leaver's teardown removing it",
            SETTLE,
            move || async move { !is_member(tv, sid, cid).await },
        )
        .await;
    }

    // ── Pass 2: one batch fails mid-page ────────────────────────────────────
    let (client_2, sweep_2) = spawn_pass(true).await;
    sweep_2.parked("pass 2's first chunk").await;
    let stalls = stall_conn(tv, data_conn_ahead(&client_2, 1), "coverage-batch").await;
    sweep_2.seam.before_chunk.release();
    sweep_2
        .script_calls(1, "the held chunk's script being sent")
        .await;
    // The next chunk the pass reaches is past the failure: the held script
    // must first time out.
    sweep_2.seam.before_chunk.arm();
    sweep_2.parked("the chunk after the failed one").await;
    assert_eq!(
        command_timeout_row(&sweep_2.metrics),
        command_timeouts_only_on("data", 1),
        "precondition: the held batch failed with one data-pool timeout"
    );
    release_all(stalls).await;
    sweep_2.seam.before_chunk.release();
    let pass_2 = sweep_2.passes(1, "pass 2").await[0];
    let seam_2 = stop_after_one_pass(sweep_2, "pass 2").await;
    assert_eq!(
        pass_2.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED,
        "a failed batch ends its session's iteration, not the pass"
    );
    let unfinished = expected.sessions_with_stale(tv).await;
    assert_eq!(
        unfinished.len(),
        1,
        "pass 2 must sweep every session but the failed batch's, and leave stale members in that \
         one; stale members left in {unfinished:?}"
    );

    // ── Between passes 2 and 3: an empty session is indexed ─────────────────
    let late_empty = "coverage-late-empty".to_owned();
    index(tv, &late_empty).await;
    expected.empty.insert(late_empty);

    // ── Pass 3 completes the work ───────────────────────────────────────────
    let (_client_3, sweep_3) = spawn_pass(false).await;
    let pass_3 = sweep_3.passes(1, "pass 3").await[0];
    let seam_3 = stop_after_one_pass(sweep_3, "pass 3").await;
    assert_eq!(pass_3.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);
    expected.assert_reached(tv, "after pass 3").await;

    // ── The bounds, over every pass ─────────────────────────────────────────
    let seams = [&seam_1, &seam_2, &seam_3];
    let max_in_flight = seams
        .iter()
        .map(|s| s.max_commands_in_flight())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_in_flight, 1,
        "a pass keeps one presence command in flight at a time"
    );
    let batches: Vec<usize> = seams.iter().flat_map(|s| s.script_batch_sizes()).collect();
    assert!(
        batches.iter().all(|n| (1..=2).contains(n)),
        "every script call carries 1 to member_batch = 2 ids; got {batches:?}"
    );
    let member_pages: Vec<usize> = seams.iter().flat_map(|s| s.member_page_sizes()).collect();
    assert!(
        member_pages.contains(&COMPACT_MEMBERS),
        "precondition: a {COMPACT_MEMBERS}-member compact set came back as one page, larger than \
         member_page; pages {member_pages:?}"
    );
    let max_member_page = member_pages.iter().copied().max().unwrap_or(0);
    assert!(
        max_member_page < LARGE_MEMBERS,
        "the {LARGE_MEMBERS}-member set must be read in pages, not whole; largest page \
         {max_member_page}"
    );
    let session_pages: Vec<usize> = seams.iter().flat_map(|s| s.session_page_sizes()).collect();
    let max_session_page = session_pages.iter().copied().max().unwrap_or(0);
    assert!(
        max_session_page < index_size,
        "the {index_size}-entry session index must be read in pages, not whole; largest page \
         {max_session_page}"
    );
    let max_ids_held = seams.iter().map(|s| s.max_ids_held()).max().unwrap_or(0);
    assert!(
        max_ids_held <= max_session_page + max_member_page,
        "a pass holds at most one session page and one member page: held {max_ids_held}, pages \
         up to {max_session_page} and {max_member_page}"
    );
    assert_eq!(
        pubsub_relay_lagged(&svc.metrics, &svc.service_label, PUBSUB_RELAY_PRESENCE),
        0,
        "precondition: the replica's presence relay never lagged"
    );

    drop(kept_streams);
    drain_tracker(&svc).await;
    feed.cancel();
}

/// **PRD AC-2**, standalone store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_passes_stay_bounded_and_cover_every_session_through_joins_leaves_and_a_failed_batch(
) {
    scenario(async {
        let tv = TestValkey::setup().await;
        coverage(&tv, &standalone_url()).await;
    })
    .await;
}

/// **PRD AC-2** on the 3-shard cluster.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_passes_on_a_cluster_stay_bounded_and_cover_every_session() {
    let Some((tv, url)) = cluster().await else {
        return;
    };
    scenario(coverage(&tv, &url)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Duplicate enumeration (SR-3.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Members of the duplicate-enumeration session: past Valkey's default
/// `set-max-listpack-entries` (128), so the set is a hashtable that `SSCAN`
/// pages — about a dozen pages at `member_page = 10` — where a listpack set
/// comes back whole in one page. Removing half of it keeps the set above the
/// fill at which Valkey shrinks its table (13% of 32 buckets of 7), so the
/// pass's own `SSCAN` repeats nothing.
const DUPLICATE_SET: usize = 140;

/// One pass delivers a member page twice and announces each removal once.
/// The seam's one-shot replay (`repeat_next_member_page`, armed before the
/// sweeper starts) holds the pass's first member page and delivers its ids
/// again at the front of the second page, so the second page's chunk re-runs
/// every id of the first — its stale members included, which the first
/// page's chunk has already removed. The script re-checks membership, so the
/// re-run removes and announces nothing: the left events, the removed count,
/// the remaining set and the index are what one delivery leaves.
///
/// The test reads the pass's first two pages itself before the pass (the set
/// is unchanged until the first chunk, and that chunk removes only ids in
/// buckets the scan has passed), and makes every other member of each page
/// stale, so the replayed ids re-run stale and live members alike, and land
/// in a chunk that also removes the second page's own stale members.
///
/// Why the seam rather than the store: `SSCAN` does repeat members — a scan
/// continued across a shrink of the set's table returns again what the
/// buckets folded together had already returned — but a test cannot make it
/// do so. In Valkey 9.0.3 a 600-member set (128 buckets) shrunk to 32 buckets
/// folds old buckets {0, 32, 64, 96} into new bucket 0, which `SSCAN` visits
/// in the order 0, 64, 32, 96. A first page that ends inside that group
/// (cursor 32, 64 or 96) is returned again; one that reads all four (cursor
/// 16) never is, whether or not the shrink's incremental rehash has finished
/// (`hashtable.c`, `hashtableScanDefrag`). Which of the two happens depends on
/// how the members fall into buckets, so on the hash seed the server draws
/// at every start (`server.c`, `hashtableSetHashFunctionSeed`): about one
/// fresh server in ten never repeats the page. A fork child (RDB save, AOF
/// rewrite) blocks it too: it sets the resize policy to AVOID, under which a
/// started shrink moves no bucket on a delete or a read, so nothing reaches
/// new bucket 0 for the scan to return again. A test on real re-delivery passed
/// on a long-lived local store and failed on CI's fresh ones.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_member_page_delivered_twice_in_one_pass_announces_each_removal_once() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let ks = tv.key_schema();
        let sid = "duplicate-pages";
        let all = cids("dup", DUPLICATE_SET);
        let seeded: Vec<(String, Seed)> = all.iter().map(|c| (c.clone(), Seed::Live)).collect();
        seed(&tv, sid, &seeded).await;

        // The pass's first two member pages, read as it will read them.
        let members_key = ks.presence_members(sid);
        let (first_cursor, first_page) = sscan_page(&tv, &members_key, "0", 10).await;
        assert_ne!(
            first_cursor, "0",
            "precondition: {DUPLICATE_SET} members take more than one SSCAN page (is the \
             store's set-max-listpack-entries at or above {DUPLICATE_SET}?)"
        );
        let (_, second_page) = sscan_page(&tv, &members_key, &first_cursor, 10).await;
        assert!(
            first_page.len() >= 2 && second_page.len() >= 2,
            "precondition: each of the first two pages holds a stale and a live member; got \
             {first_page:?} then {second_page:?}"
        );

        // Every other member of each page, and of the rest of the set, goes
        // stale: its lease is deleted (its info hash records a lease still
        // running, so its removal delay reads 0).
        let rest: Vec<&String> = all
            .iter()
            .filter(|c| !first_page.contains(c) && !second_page.contains(c))
            .collect();
        let stale: BTreeSet<String> = first_page
            .iter()
            .step_by(2)
            .chain(second_page.iter().step_by(2))
            .chain(rest.into_iter().step_by(2))
            .cloned()
            .collect();
        let live: BTreeSet<String> = all
            .iter()
            .filter(|c| !stale.contains(*c))
            .cloned()
            .collect();
        let conn = store_reader(&tv);
        for cid in &stale {
            let _: i64 = conn
                .del(ks.presence_lease(sid, cid))
                .await
                .expect("DEL lease must succeed");
        }

        let mut tap = Tap::open(&tv, sid).await;
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        let sweep = Sweep::spawn_with_seam(
            &tv,
            tv.client(),
            "duplicate",
            budget(Duration::from_secs(30), 10, 10, 100),
            COVERAGE_INTERVAL,
            seam,
        )
        .await;
        let metrics = Arc::clone(&sweep.metrics);
        let pass = sweep.passes(1, "the pass").await[0];
        assert_eq!(pass.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);
        let seam = stop_after_one_pass(sweep, "the duplicate-enumeration pass").await;

        let pages = seam.member_page_sizes();
        assert!(
            pages.len() >= 2,
            "the pass must read at least two member pages; it read {pages:?}"
        );
        assert_eq!(
            pages[..2],
            [first_page.len(), first_page.len() + second_page.len()],
            "the pass's second page must carry the first page's ids in front of its own (did \
             the replay fire?)"
        );
        let delivered = DUPLICATE_SET + first_page.len();
        assert_eq!(
            pages.iter().sum::<usize>(),
            delivered,
            "the scan delivers every member once and the first page's twice"
        );
        assert_eq!(
            seam.script_batch_sizes().iter().sum::<usize>(),
            delivered,
            "every delivered id, the replayed ones included, goes through a script call"
        );

        let events = read_lefts(&tv, &mut tap, stale.len()).await;
        let want: BTreeMap<String, usize> = stale.iter().map(|c| (c.clone(), 1)).collect();
        assert_eq!(
            lefts_per_cid(&events),
            want,
            "each stale member has exactly one left, and no live member any"
        );
        assert_eq!(
            stale_removed(&metrics),
            stale.len() as f64,
            "each removal is counted once"
        );
        assert_eq!(members(&tv, sid).await, live, "every live member remains");
        assert!(
            is_indexed(&tv, sid).await,
            "the session, not empty, stays indexed"
        );
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Per-session error isolation (SR-3.4)
// ═════════════════════════════════════════════════════════════════════════════

/// A chunk that times out ends its own session's iteration, and the same pass
/// still sweeps the next session; the next pass finishes the first. Two
/// sessions of four stale members, chunks of two. The first chunk's script is
/// held on its data connection past the sweeper's 1 s command timeout (a
/// timed-out command is not re-sent; a lock TTL of 30 s keeps the lease out of
/// it), and the pass is parked again at its next chunk before the hold is
/// released, so the store is read at a fixed point.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_chunk_that_times_out_ends_its_session_and_the_pass_sweeps_the_next() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let ks = tv.key_schema();
        let sessions = ["isolation-a", "isolation-b"];
        for sid in sessions {
            let seeded: Vec<(String, Seed)> = cids(sid, 4)
                .into_iter()
                .map(|cid| (cid, Seed::Stale))
                .collect();
            seed(&tv, sid, &seeded).await;
        }
        let client = own_client(&standalone_url(), FAILING_COMMAND_TIMEOUT_MS, 4).await;
        let sweep = Sweep::spawn(
            &tv,
            &client,
            "isolation",
            budget(Duration::from_secs(30), 10, 10, 2),
            COVERAGE_INTERVAL,
            true,
        )
        .await;
        sweep.parked("the first session's first chunk").await;
        let (_, index_page) = sscan_page(&tv, &ks.presence_active_sessions(), "0", 10).await;
        let (failed, next) = (index_page[0].clone(), index_page[1].clone());

        let stalls = stall_conn(&tv, data_conn_ahead(&client, 1), "isolation").await;
        sweep.seam.before_chunk.release();
        sweep
            .script_calls(1, "the held chunk's script being sent")
            .await;
        sweep.seam.before_chunk.arm();
        sweep.parked("the chunk after the failed one").await;

        assert_eq!(
            command_timeout_row(&sweep.metrics),
            command_timeouts_only_on("data", 1),
            "precondition: the held chunk failed with exactly one data-pool timeout"
        );
        assert_eq!(
            members(&tv, &failed).await.len(),
            4,
            "the failed chunk's session: its held chunk has not run and its next was not sent"
        );
        assert_eq!(
            sweep.seam.script_batch_sizes(),
            vec![2],
            "the pass moved on to the next session without another chunk of the failed one"
        );
        release_all(stalls).await;
        sweep.seam.before_chunk.release();

        let pass_1 = sweep.passes(1, "pass 1").await[0];
        assert_eq!(
            pass_1,
            ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_COMPLETED,
                script_calls: 3,
            },
            "the pass completes: one chunk of the failed session, both of the next"
        );
        assert!(
            members(&tv, &next).await.is_empty(),
            "the same pass must sweep the session after the failed one"
        );
        assert!(
            !is_indexed(&tv, &next).await,
            "the emptied session leaves the index"
        );

        sweep.passes(2, "pass 2").await;
        assert!(
            members(&tv, &failed).await.is_empty(),
            "the next pass completes the failed session"
        );
        sweep.join().await;
    })
    .await;
}

/// Two sessions of `per_session` stale members, indexed in that order, and
/// their ids in the order the index's compact page returns them.
async fn two_stale_sessions(tv: &TestValkey, prefix: &str, per_session: usize) -> (String, String) {
    for sid in [format!("{prefix}-a"), format!("{prefix}-b")] {
        let seeded: Vec<(String, Seed)> = cids(&sid, per_session)
            .into_iter()
            .map(|cid| (cid, Seed::Stale))
            .collect();
        seed(tv, &sid, &seeded).await;
    }
    let (_, index_page) =
        sscan_page(tv, &tv.key_schema().presence_active_sessions(), "0", 10).await;
    (index_page[0].clone(), index_page[1].clone())
}

/// A member page that times out ends its own session's iteration, not the
/// pass: the next session is swept in the same pass, which completes. The
/// failed session is left as it was — its members, and its index entry: a
/// page that failed is not a session found empty. The sweeper's client is new
/// and has sent nothing, so its first data-pool command is the index read and
/// its second the first session's member page, which is held past the 1 s
/// command timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_member_page_that_times_out_ends_its_session_and_the_pass_sweeps_the_next() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (failed, next) = two_stale_sessions(&tv, "page-error", 4).await;
        let client = own_client(&standalone_url(), FAILING_COMMAND_TIMEOUT_MS, COVERAGE_POOL).await;
        let stalls = stall_conn(&tv, data_conn_ahead(&client, 2), "member-page").await;
        let sweep = Sweep::spawn(
            &tv,
            &client,
            "page-error",
            budget(Duration::from_secs(30), 10, 10, 2),
            COVERAGE_INTERVAL,
            false,
        )
        .await;

        let pass = sweep.passes(1, "the pass").await[0];
        let metrics = Arc::clone(&sweep.metrics);
        stop_after_one_pass(sweep, "the pass").await;

        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "precondition: the held member page failed with one data-pool timeout"
        );
        assert_eq!(
            pass,
            ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_COMPLETED,
                script_calls: 2,
            },
            "the pass completes, sweeping only the next session (two chunks)"
        );
        assert_eq!(
            members(&tv, &failed).await.len(),
            4,
            "the failed page's session is not swept"
        );
        assert!(
            is_indexed(&tv, &failed).await,
            "a session whose page failed must not be unindexed as if it were empty"
        );
        assert!(
            members(&tv, &next).await.is_empty() && !is_indexed(&tv, &next).await,
            "the same pass must sweep, and unindex, the next session"
        );
        release_all(stalls).await;
    })
    .await;
}

/// An index removal that times out does not stop the pass: the next session
/// is still reached and swept, and the pass completes. The first session's one
/// chunk empties it; the removal's `SREM`, the command after that chunk's
/// script, is held past the 1 s command timeout, so the session stays indexed
/// (a later pass, or a keeper's re-add, deals with it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_index_removal_that_times_out_does_not_stop_the_pass() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (failed, next) = two_stale_sessions(&tv, "removal-error", 2).await;
        let client = own_client(&standalone_url(), FAILING_COMMAND_TIMEOUT_MS, COVERAGE_POOL).await;
        let sweep = Sweep::spawn(
            &tv,
            &client,
            "removal-error",
            budget(Duration::from_secs(30), 10, 10, 2),
            COVERAGE_INTERVAL,
            true,
        )
        .await;
        sweep.parked("the first session's chunk").await;
        // After the chunk's script: the index removal's SREM.
        let stalls = stall_conn(&tv, data_conn_ahead(&client, 2), "index-removal").await;
        sweep.seam.before_chunk.release();

        let pass = sweep.passes(1, "the pass").await[0];
        let metrics = Arc::clone(&sweep.metrics);
        stop_after_one_pass(sweep, "the pass").await;

        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "precondition: the held index removal failed with one data-pool timeout"
        );
        assert!(
            members(&tv, &failed).await.is_empty() && is_indexed(&tv, &failed).await,
            "precondition: the first session was emptied and its index removal did not run"
        );
        assert_eq!(
            pass,
            ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_COMPLETED,
                script_calls: 2,
            },
            "a failed index removal must not stop the pass"
        );
        assert!(
            members(&tv, &next).await.is_empty() && !is_indexed(&tv, &next).await,
            "the pass must reach, sweep and unindex the next session"
        );
        release_all(stalls).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Event order under a sweep (PRD FR-6a)
// ═════════════════════════════════════════════════════════════════════════════

/// Connections evicted by the one swept chunk.
const EVICTED: usize = 100;

/// Of them, connections whose keeper rejoins the moment the chunk returns,
/// each on a service of its own (a keeper seam parks one keeper).
const REJOINERS: usize = 4;

/// The sentinel the observer's stream is read up to.
const OBSERVER_SENTINEL: &str = "fr6a-sentinel";

/// One chunk of 100 evicted connections is swept: 96 whose replica crashed,
/// and 4 live joins whose keepers are parked before a refresh and whose leases
/// are then deleted. The moment the store shows the chunk removed, the 4
/// keepers are released; each finds its connection evicted and rejoins. A peer
/// (on a cluster, attached to a node that does not own the session's slot)
/// sees, for every cid, its `left` strictly before its `joined`; and the view
/// it replays from what it saw — and the view a live observer join replays
/// from its own stream — ends with every live connection present.
async fn event_order(tv: &TestValkey) {
    let observer_svc = build_service(tv, Arc::new(AllowAllPermissions));
    let feed = spawn_router_feed(tv, &observer_svc).await;
    let (sid, _token) = create_session(tv, &observer_svc, "fr6a", true).await;
    let crashed = cids("crashed", EVICTED - REJOINERS);
    let seeded: Vec<(String, Seed)> = crashed.iter().map(|c| (c.clone(), Seed::Stale)).collect();
    seed(tv, &sid, &seeded).await;
    let mut tap = open_tap(tv, &sid).await;

    // Each rejoiner joins on a service of its own, its keeper parked at its
    // first refresh.
    let mut keepers = Vec::new();
    for k in 0..REJOINERS {
        let seam = Arc::new(LeaseKeeperSeam::new());
        let svc = LiveEditService {
            presence_refresh_interval_secs: 1,
            presence_keepalive_interval_ms:
                live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            lease_keeper_seam_for_test: Some(Arc::clone(&seam)),
            ..build_service(tv, Arc::new(AllowAllPermissions))
        };
        svc.session_tokens
            .insert(sid.clone(), CancellationToken::new().into());
        seam.before_refresh.arm();
        let (mut stream, tx) = open_join(&svc, &sid, &format!("rejoiner-{k}")).await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(&seam.before_refresh, "a rejoiner's keeper").await;
        keepers.push((svc, seam, cid, stream, tx));
    }
    let rejoiners: Vec<String> = keepers
        .iter()
        .map(|(_, _, cid, _, _)| cid.clone())
        .collect();

    // A live observer join, read from its replay on.
    let (mut observer, observer_tx) = open_join(&observer_svc, &sid, "fr6a-observer").await;
    let observer_cid = expect_session_connected(&mut observer, SETTLE).await;
    let observer_reader = tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Some(item) = observer.next().await {
            match item
                .expect("the observer's stream must not fail")
                .response_type
            {
                Some(ResponseType::PeerJoined(p)) if p.connection_id == OBSERVER_SENTINEL => {
                    return (seen, observer);
                }
                Some(rt) => seen.push(rt),
                None => {}
            }
        }
        panic!("the observer's stream ended before its sentinel");
    });

    for cid in &rejoiners {
        let deleted: i64 = store_reader(tv)
            .del(tv.key_schema().presence_lease(&sid, cid))
            .await
            .expect("DEL lease must succeed");
        assert_eq!(
            deleted, 1,
            "precondition: the parked keeper's lease existed"
        );
    }
    // Everything published so far: the five joins.
    let setup = tap
        .until_count("the setup joins", REJOINERS + 1, |e| {
            matches!(e, PresenceEvent::PeerJoined(_))
        })
        .await;
    assert!(
        setup
            .iter()
            .all(|e| matches!(e, PresenceEvent::PeerJoined(_))),
        "precondition: only the joins were published before the sweep; got {setup:?}"
    );
    let mut view: BTreeSet<String> = members(tv, &sid).await;
    assert_eq!(
        view.len(),
        EVICTED + 1,
        "precondition: 100 evicted and the observer"
    );

    let sweep = Sweep::spawn(
        tv,
        tv.client(),
        "fr6a",
        budget(Duration::from_secs(30), 10, 200, EVICTED),
        SWEEP_INTERVAL,
        true,
    )
    .await;
    sweep.parked("the sweep's first chunk").await;
    sweep.seam.before_chunk.release();
    let members_key = tv.key_schema().presence_members(&sid);
    tokio::time::timeout(SETTLE, async {
        loop {
            let n: i64 = store_reader(tv)
                .scard(&members_key)
                .await
                .expect("SCARD must succeed");
            if n == 1 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the swept chunk must leave only the observer");
    for (_, seam, _, _, _) in &keepers {
        seam.before_refresh.release();
    }

    let rejoiner_set: BTreeSet<&String> = rejoiners.iter().collect();
    let mut events = tap
        .until_count("the rejoins' joined events", REJOINERS, |e| {
            matches!(e, PresenceEvent::PeerJoined(u) if rejoiner_set.contains(&u.connection_id))
        })
        .await;
    events.extend(tap.drain(tv).await);
    assert_eq!(
        sweep.seam.script_batch_sizes().first(),
        Some(&EVICTED),
        "precondition: the 100 evicted connections were swept in one chunk"
    );

    let position = |pred: &dyn Fn(&PresenceEvent) -> bool| -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter(|(_, e)| pred(e))
            .map(|(i, _)| i)
            .collect()
    };
    for cid in crashed.iter().chain(rejoiners.iter()) {
        let lefts = position(&|e| is_left(e, cid));
        let joins = position(&|e| is_joined(e, cid));
        assert_eq!(lefts.len(), 1, "{cid}: exactly one left; got {events:?}");
        if rejoiner_set.contains(cid) {
            assert_eq!(joins.len(), 1, "{cid}: exactly one rejoin joined");
            assert!(
                lefts[0] < joins[0],
                "{cid}: its left (event {}) must reach the peer strictly before its rejoin's \
                 joined (event {})",
                lefts[0],
                joins[0]
            );
        } else {
            assert!(
                joins.is_empty(),
                "{cid}: a crashed connection does not rejoin"
            );
        }
    }

    // The peer's view, replayed.
    for event in &events {
        match event {
            PresenceEvent::PeerJoined(u) => {
                view.insert(u.connection_id.clone());
            }
            PresenceEvent::PeerLeft { connection_id } => {
                view.remove(connection_id);
            }
            PresenceEvent::PeerMessage { .. } => {}
        }
    }
    let live: BTreeSet<String> = rejoiners
        .iter()
        .cloned()
        .chain([observer_cid.clone()])
        .collect();
    assert_eq!(
        members(tv, &sid).await,
        live,
        "precondition: the store holds the rejoined connections and the observer"
    );
    assert_eq!(
        view, live,
        "the peer's replayed view must hold every live connection, and no crashed one"
    );

    // The observer join's own view.
    publish_probe(tv, &sid, OBSERVER_SENTINEL).await;
    let (observed, observer) = tokio::time::timeout(SETTLE, observer_reader)
        .await
        .expect("the observer must read its sentinel")
        .expect("the observer's reader must not panic");
    assert_eq!(
        pubsub_relay_lagged(
            &observer_svc.metrics,
            &observer_svc.service_label,
            PUBSUB_RELAY_PRESENCE
        ),
        0,
        "precondition: the observer's replica relay never lagged"
    );
    let mut observer_view = BTreeSet::new();
    for rt in observed {
        match rt {
            ResponseType::PeerJoined(p) => {
                observer_view.insert(p.connection_id);
            }
            ResponseType::PeerLeft(p) => {
                observer_view.remove(&p.connection_id);
            }
            _ => {}
        }
    }
    let others: BTreeSet<String> = rejoiners.iter().cloned().collect();
    assert_eq!(
        observer_view, others,
        "the observer's replayed view must hold every other live connection, and no crashed one"
    );

    sweep.join().await;
    drop((observer, observer_tx));
    for (svc, _, _, stream, tx) in keepers {
        drop((stream, tx));
        drain_tracker(&svc).await;
    }
    drain_tracker(&observer_svc).await;
    feed.cancel();
    tap.close().await;
}

/// **PRD FR-6a**, standalone store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_of_100_evicted_with_rejoining_keepers_announces_each_left_before_its_joined() {
    scenario(async {
        let tv = TestValkey::setup().await;
        event_order(&tv).await;
    })
    .await;
}

/// **PRD FR-6a** on the 3-shard cluster, the peer attached to a node that does
/// not own the session's slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_on_a_cluster_announces_each_left_before_its_rejoin_on_another_node() {
    let Some((tv, _url)) = cluster().await else {
        return;
    };
    scenario(event_order(&tv)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The index race with the real sweeper (SR-3.8)
// ═════════════════════════════════════════════════════════════════════════════

/// A service with the registration seam installed.
fn registration_seamed_service(tv: &TestValkey) -> (LiveEditService, Arc<JoinRegistrationSeam>) {
    let seam = Arc::new(JoinRegistrationSeam::new());
    let svc = LiveEditService {
        join_registration_seam_for_test: Some(Arc::clone(&seam)),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    };
    (svc, seam)
}

/// A join parked between its first index add and its register; the sweeper's
/// pass sweeps the session's one stale member and, the set now empty, removes
/// the session from the index — the whole remove-then-recheck lands before the
/// register. The join's re-add after its register restores the entry, so the
/// session is indexed when the join's registration ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_index_removal_before_a_joins_register_is_restored_by_the_joins_re_add() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seam) = registration_seamed_service(&tv);
        let (sid, _token) = create_session(&tv, &svc, "index-race-re-add", true).await;
        seed(&tv, &sid, &[("stale".to_owned(), Seed::Stale)]).await;

        seam.before_register.arm();
        let (mut stream, _tx) = open_join(&svc, &sid, "index-race-app").await;
        wait_parked(&seam.before_register, "the join, before its register").await;

        let sweep = Sweep::spawn(
            &tv,
            tv.client(),
            "index-race",
            budget(Duration::from_secs(30), 10, 10, 2),
            SWEEP_INTERVAL,
            false,
        )
        .await;
        let pass = sweep.passes(1, "the sweep pass").await[0];
        sweep.join().await;
        assert_eq!(pass.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);
        assert!(
            !is_indexed(&tv, &sid).await,
            "precondition: the sweep emptied the session and removed it from the index"
        );

        seam.before_register.release();
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        assert!(
            is_indexed(&tv, &sid).await,
            "the session must be indexed when the join's registration ends"
        );
        assert!(is_member(&tv, &sid, &cid).await);

        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}

/// The sweep's index removal straddles the join's register: its `SREM` lands
/// while the join is parked before its register, and its re-check `SCARD` is
/// held on its data connection until the join has registered. The join's own
/// re-add is held too, so the re-check alone must restore the entry: once it
/// completes the session is indexed, before the join's re-add has run. Then
/// the join's re-add completes, and the session is indexed when the join's
/// registration ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_index_recheck_after_a_joins_register_restores_the_entry() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seam) = registration_seamed_service(&tv);
        let (sid, _token) = create_session(&tv, &svc, "index-race-recheck", true).await;
        seed(&tv, &sid, &[("stale".to_owned(), Seed::Stale)]).await;

        seam.before_register.arm();
        let (mut stream, _tx) = open_join(&svc, &sid, "index-race-app").await;
        wait_parked(&seam.before_register, "the join, before its register").await;

        let client = own_client(&standalone_url(), HELD_COMMAND_TIMEOUT_MS, 4).await;
        let sweep = Sweep::spawn(
            &tv,
            &client,
            "index-race",
            budget(Duration::from_secs(30), 10, 10, 1),
            SWEEP_INTERVAL,
            true,
        )
        .await;
        sweep.parked("the sweep's chunk").await;
        // After the chunk's script: the index removal's SREM, then its SCARD.
        let recheck = stall_conn(&tv, data_conn_ahead(&client, 3), "recheck").await;
        // After the join's register: its index re-add.
        let re_add = stall_conn(&tv, data_conn_ahead(tv.client(), 2), "re-add").await;

        sweep.seam.before_chunk.release();
        let (tv_ref, sid_ref) = (&tv, sid.as_str());
        poll_until(
            "the sweep's SREM removing the emptied session",
            SETTLE,
            move || async move { !is_indexed(tv_ref, sid_ref).await },
        )
        .await;
        assert!(
            members(&tv, &sid).await.is_empty(),
            "precondition: the sweep removed the stale member"
        );

        seam.before_register.release();
        poll_until("the join's register", SETTLE, move || async move {
            members(tv_ref, sid_ref).await.len() == 1
        })
        .await;
        assert!(
            !is_indexed(&tv, &sid).await,
            "precondition: neither the sweep's re-check nor the join's re-add has run"
        );

        release_all(recheck).await;
        poll_until(
            "the sweep's re-check re-adding the session",
            SETTLE,
            move || async move { is_indexed(tv_ref, sid_ref).await },
        )
        .await;
        assert!(
            re_add[0].is_blocked().await,
            "precondition: the join's re-add is still held, so the re-check restored the entry"
        );

        release_all(re_add).await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        assert!(
            is_indexed(&tv, &sid).await,
            "the session must be indexed when the join's registration ends"
        );
        assert!(is_member(&tv, &sid, &cid).await);
        let pass = sweep.passes(1, "the sweep pass").await[0];
        assert_eq!(pass.outcome, PRESENCE_SWEEP_OUTCOME_COMPLETED);

        sweep.join().await;
        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}
