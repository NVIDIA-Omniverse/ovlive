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

// A crash-sized sweep burst reaching the replica-wide pub/sub relays of the
// surviving replicas, through real JoinSession streams.
//
// The presence sweeper publishes each removal's `left` inside its batch
// script, so one pass after a replica crash publishes every stale member's
// `left` back to back. Each surviving replica receives them on its one pub/sub
// subscriber client, whose message channel
// (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY`, 4096) both of its
// relays read: the presence relay (`presence::spawn_presence_subscriber`) and
// the session-deleted relay (`grpc::session::spawn_session_delete_subscriber`),
// which skips the presence messages but still receives every one. A relay more
// than the capacity behind skips messages — for presence, lost for every local
// stream of their sessions — and counts them on
// `live_edit_pubsub_relay_lagged_total{relay}`.
//
// ── The scenario ─────────────────────────────────────────────────────────────
//
// Two surviving replicas, A and B, each a `LiveEditService` over a
// `ValkeyClient` of its own, with both relays on its own subscriber client as
// `main.rs` wires them, share the store with a crashed replica that is never
// built: its members are written straight into the store, no lease and an info
// hash recording a lease that expired `STALE_FOR_MS` ago, as a crash leaves
// them. Every affected session holds one real JoinSession observer on A and
// one on B, so every `left` reaches both replicas — the most a replica can be
// sent for a given crash. Sessions come in three sizes:
//
//   * demo — 5 members: 3 of the crashed replica's and the 2 observers;
//   * large — 50 members: 48 crashed and the 2 observers;
//   * full-batch — 102 members: `SWEEP_MEMBER_BATCH` (100) crashed and the 2
//     observers.
//
// A pass sends one script call per chunk of up to `SWEEP_MEMBER_BATCH` ids of
// a session, so the more crashed members a session holds, up to that limit,
// the more `left`s each call publishes at once and the faster the pass
// publishes. A full-batch session's crashed members fill exactly one full
// call: that shape is the fastest burst a pass can send. This depends on the
// store keeping the 102-member set compact (a listpack), which it does only
// while the set is within `set-max-listpack-entries` (128 by default; the shape
// needs at least 102): a compact set comes back whole, in insertion order.
// Checked before the pass with the sweeper's own page read.
//
// Replica A also runs the sweeper (production budget, `SweeperBudget::
// production`), as the leading survivor would. One pass removes every stale
// member. Afterwards a sentinel published on each session's channel — behind
// every `left` of the session, which leaves the same node first — bounds what
// each observer's stream is read up to.
//
// Asserted: every observer receives a `PeerLeft` for every removed member of
// its session, exactly once, and its sentinel, its stream still open; and no
// relay on either replica lagged. Reported (an `eprintln!` block, kept for
// passing runs by the overrides in `.config/nextest.toml`): the pass's
// duration, removal rate and script calls; the rate and peak 100 ms count at
// which each replica's streams were delivered the `left`s, and how long after
// (or before) the pass ended the last one arrived; sessions × observers; each
// relay's lag; and the process's peak RSS.
//
// ── Sizes ────────────────────────────────────────────────────────────────────
//
// All three shapes are crash scale, about 10,000 stale members:
//
//   * full-batch (`CRASH_SCALE_FULL_BATCH`, 100 sessions, 200 streams) runs
//     in the default suite, in a few seconds;
//   * demo (`CRASH_SCALE_DEMO`, 6,668 streams) and mixed (`CRASH_SCALE_MIX`,
//     3,604 streams) are `#[ignore]`d measurements: thousands of JoinSession
//     streams, each with its lease keeper and its session's router. Run them
//     with
//
//       cargo nextest run -p live-edit --test integration \
//         --features test-support,test-fault-injection --run-ignored all \
//         -E 'test(/presence_sweep_relay_burst_test/)'
//
// Headroom, measured once per capacity on the full-batch shape with each
// replica's subscriber client rebuilt at a smaller message-channel capacity,
// while the other 101 presence tests (sweeper, replay-dedup, join-lifecycle)
// ran beside it in the same run at `--test-threads 4`: no relay lagged at
// 2048, 1024, 512, 256 or 128; at 64 — below one script call's 100 `left`s —
// both relays of both replicas lagged (presence 40 and 41, session-deleted 62
// and 117) and 81 `PeerLeft`s were lost. The smallest passing capacity is
// 128, so the production 4096 has 32× headroom, and the default-suite test
// fails on a lag rather than retrying it (`retries = 0` in
// `.config/nextest.toml`).
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Setup steps are bounded by `SETUP_WAIT` each (a join's `SessionConnected`,
// a session-deleted relay's attach, and the parked first chunk); the pass by
// `PASS_BUDGET`; the read of every observer up to its sentinel by one
// `DRAIN_BUDGET` deadline across all of them, so a lost sentinel ends the read
// with whatever arrived rather than stalling it; the teardown by
// `DRAIN_BUDGET`. The default-suite test sits under `SCENARIO_BUDGET` (100 s);
// each `#[ignore]`d one under `CRASH_SCALE_BUDGET` (280 s), with a 300 s
// hang-killer override in `.config/nextest.toml`.
//
// Compiled only with `--features test-fault-injection`: it drives the
// sweeper seam.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use live_edit::grpc::proto::join_session_response::ResponseType;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{
    spawn_session_delete_subscriber, subscribe_session_deleted_channel,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, PRESENCE_SWEEP_OUTCOME_COMPLETED, PUBSUB_RELAY_PRESENCE, PUBSUB_RELAY_SESSION_DELETED,
};
use live_edit::presence::{spawn_presence_sweeper_with_seam, PresenceSweeperSeam, SweeperBudget};
use live_edit::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::presence::{
    scan_member_page, store_time_ms, SWEEP_MEMBER_BATCH, SWEEP_MEMBER_PAGE,
};
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use super::helpers::*;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The crashed replica's members of a demo session; with one observer on
/// each surviving replica, 5 members.
const DEMO_STALE: usize = 3;

/// The crashed replica's members of a large session; with the two observers,
/// 50 members.
const LARGE_STALE: usize = 48;

/// How long before the seed a crashed member's recorded lease expired.
const STALE_FOR_MS: u64 = 5_000;

/// The data-pool connections of each surviving replica's client.
const REPLICA_POOL: u32 = 8;

/// JoinSessions (and session creations) set up at once.
const SETUP_CONCURRENCY: usize = 64;

/// Sessions seeded per pipeline.
const SEED_PIPELINE_SESSIONS: usize = 100;

/// Bound on each setup wait.
const SETUP_WAIT: Duration = Duration::from_secs(30);

/// How long a session-deleted relay's attach probe waits before it is
/// published again: a message published before the relay's receiver exists
/// never reaches it.
const ATTACH_RETRY: Duration = Duration::from_millis(20);

/// Bound on the sweep pass, from its first chunk's release.
const PASS_BUDGET: Duration = Duration::from_secs(60);

/// One deadline across reading every observer up to its sentinel, and the
/// bound on the teardown.
const DRAIN_BUDGET: Duration = Duration::from_secs(30);

/// Whole-test deadline of an `#[ignore]`d crash-scale run, under its 300 s
/// hang-killer override.
const CRASH_SCALE_BUDGET: Duration = Duration::from_secs(280);

/// The window of the reported peak delivery rate.
const PEAK_WINDOW: Duration = Duration::from_millis(100);

/// The sweep interval: the first pass starts one interval after the spawn and
/// parks at its first chunk; the sweeper is stopped once that pass ends.
const SWEEP_INTERVAL: Duration = Duration::from_millis(100);

/// The connection id of the sentinel published on every session's channel
/// after the pass.
const SENTINEL: &str = "burst-sentinel";

// ═════════════════════════════════════════════════════════════════════════════
// Shapes
// ═════════════════════════════════════════════════════════════════════════════

/// The affected sessions of one crash: demo sessions, then bigger ones.
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// 5-member sessions: [`DEMO_STALE`] crashed members and two observers.
    demo_sessions: usize,
    /// Sessions of `big_stale` crashed members and two observers.
    big_sessions: usize,
    /// The crashed members of each big session.
    big_stale: usize,
}

impl Shape {
    fn sessions(&self) -> usize {
        self.demo_sessions + self.big_sessions
    }

    fn stale(&self) -> usize {
        self.demo_sessions * DEMO_STALE + self.big_sessions * self.big_stale
    }

    /// The crashed members of session `i` (demo sessions first).
    fn stale_in(&self, i: usize) -> usize {
        if i < self.demo_sessions {
            DEMO_STALE
        } else {
            self.big_stale
        }
    }
}

/// Crash scale in full-batch sessions, the fastest a pass publishes (one full
/// `SWEEP_MEMBER_BATCH`-id script call per session): 100 sessions, 10,000
/// stale members, 200 observers.
const CRASH_SCALE_FULL_BATCH: Shape = Shape {
    demo_sessions: 0,
    big_sessions: 100,
    big_stale: SWEEP_MEMBER_BATCH,
};

/// Crash scale in demo sessions only: 3,334 sessions, 10,002 stale members,
/// 6,668 observers.
const CRASH_SCALE_DEMO: Shape = Shape {
    demo_sessions: 3_334,
    big_sessions: 0,
    big_stale: 0,
};

/// Crash scale in a mix: 1,700 demo and 102 large (50-member) sessions, 9,996
/// stale members (4,896 of them in large sessions), 3,604 observers.
const CRASH_SCALE_MIX: Shape = Shape {
    demo_sessions: 1_700,
    big_sessions: 102,
    big_stale: LARGE_STALE,
};

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A surviving replica: a service over a `ValkeyClient` of its own, with the
/// presence relay feeding its router and the session-deleted relay over its
/// session tokens, both on that client's subscriber.
struct Replica {
    label: &'static str,
    svc: LiveEditService,
    /// Cancels both relays.
    relays: CancellationToken,
}

impl Replica {
    async fn start(tv: &TestValkey, label: &'static str) -> Self {
        let config = live_edit::config::Config {
            valkey_url: std::env::var("TEST_VALKEY_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            valkey_pool_size: REPLICA_POOL,
            valkey_connect_timeout_secs: 30,
            ..crate::common::test_config()
        };
        let client: ValkeyClient = connect_test_client(&config)
            .await
            .expect("a replica's ValkeyClient must connect");
        let svc = LiveEditService {
            valkey: client,
            metrics: Arc::new(crate::common::metrics_scrape_only(label)),
            service_label: label.to_owned(),
            ..build_service(tv, Arc::new(AllowAllPermissions))
        };
        let relays = spawn_router_feed(tv, &svc).await;
        start_session_deleted_relay(tv, &svc, label, &relays).await;
        Replica { label, svc, relays }
    }

    /// `(presence, session_deleted)` relay lag.
    fn relays_lagged(&self) -> (u64, u64) {
        (
            pubsub_relay_lagged(&self.svc.metrics, self.label, PUBSUB_RELAY_PRESENCE),
            pubsub_relay_lagged(&self.svc.metrics, self.label, PUBSUB_RELAY_SESSION_DELETED),
        )
    }
}

/// Subscribe `svc`'s subscriber client to the session-deleted channel and
/// spawn its relay, as `main.rs` does, and return once the relay applies a
/// deletion: an attach probe's token, re-published every [`ATTACH_RETRY`]
/// until the relay cancels it.
async fn start_session_deleted_relay(
    tv: &TestValkey,
    svc: &LiveEditService,
    label: &str,
    cancel: &CancellationToken,
) {
    use fred::interfaces::PubsubInterface;

    subscribe_session_deleted_channel(&svc.valkey.subscriber, tv.key_schema())
        .await
        .expect("SUBSCRIBE on the session-deleted channel must succeed");
    spawn_session_delete_subscriber(
        svc.valkey.subscriber.clone(),
        Arc::clone(&svc.session_tokens),
        svc.session_base_path_cache.clone(),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel.clone(),
    );
    let probe = format!("relay-attach-{label}");
    let token = CancellationToken::new();
    svc.session_tokens
        .insert(probe.clone(), token.clone().into());
    let channel = tv.key_schema().session_deleted_channel();
    tokio::time::timeout(SETUP_WAIT, async {
        loop {
            let _: i64 = store_reader(tv)
                .publish(&channel, probe.as_str())
                .await
                .expect("PUBLISH the attach probe must succeed");
            if tokio::time::timeout(ATTACH_RETRY, token.cancelled())
                .await
                .is_ok()
            {
                return;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{label}'s session-deleted relay never applied its attach probe"));
}

/// `live_edit_presence_stale_removed_total` on `metrics`.
fn stale_removed(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_presence_stale_removed_total",
        &[],
    )
    .expect("the stale-removed cell is seeded") as u64
}

/// The crashed members of session `i`.
fn crashed_cids(i: usize, n: usize) -> Vec<String> {
    (0..n).map(|j| format!("crashed-{i:05}-{j:03}")).collect()
}

/// Create the shape's sessions in the store, none with a token on either
/// replica (each observer's join registers one).
async fn create_sessions(tv: &TestValkey, svc: &LiveEditService, shape: Shape) -> Vec<String> {
    futures::stream::iter(0..shape.sessions())
        .map(|i| async move {
            let (sid, _token) = create_session(tv, svc, &format!("burst-{i:05}"), false).await;
            sid
        })
        .buffered(SETUP_CONCURRENCY)
        .collect()
        .await
}

/// Write each session's crashed members straight into the store, as the
/// crashed replica's joins left them once its keepers stopped: members, no
/// lease, an info hash whose recorded lease expired [`STALE_FOR_MS`] ago; and
/// index each session. Nothing is published.
async fn seed_crashed(tv: &TestValkey, sessions: &[String], shape: Shape) {
    use fred::interfaces::{HashesInterface, SetsInterface};

    let ks = tv.key_schema();
    let now_ms = store_time_ms(
        tv.client(),
        FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
        ks,
        &sessions[0],
    )
    .await
    .expect("store_time_ms must succeed");
    let expired_at = (now_ms - STALE_FOR_MS).to_string();
    for (page, sids) in sessions.chunks(SEED_PIPELINE_SESSIONS).enumerate() {
        let pipeline = store_reader(tv).pipeline();
        for (k, sid) in sids.iter().enumerate() {
            let i = page * SEED_PIPELINE_SESSIONS + k;
            let cids = crashed_cids(i, shape.stale_in(i));
            let _: () = pipeline
                .sadd(ks.presence_members(sid), cids.clone())
                .await
                .expect("queuing SADD must succeed");
            for cid in &cids {
                let _: () = pipeline
                    .hset(
                        ks.presence_user(sid, cid),
                        vec![
                            ("user_name", format!("User {cid}")),
                            ("user_application", "crashed-app".to_owned()),
                            ("lease_expires_at_ms", expired_at.clone()),
                        ],
                    )
                    .await
                    .expect("queuing HSET must succeed");
            }
            let _: () = pipeline
                .sadd(ks.presence_active_sessions(), sid.as_str())
                .await
                .expect("queuing the index SADD must succeed");
        }
        let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");
    }
}

/// What one observer's stream delivered up to its sentinel.
#[derive(Default)]
struct ObserverLog {
    /// Every `PeerLeft`, with when the test read it.
    lefts: Vec<(String, Instant)>,
    /// The sentinel arrived.
    sentinel: bool,
    /// How the stream ended before its sentinel, if it did.
    ended: Option<String>,
}

/// One JoinSession observer, read by a task of its own from before the sweep.
struct Observer {
    replica: usize,
    session: usize,
    log: Arc<Mutex<ObserverLog>>,
    reader: tokio::task::JoinHandle<Option<(JoinResponseStream, ClientTx)>>,
}

type ClientTx = tokio::sync::mpsc::UnboundedSender<bytes::Bytes>;

/// Open an observer on `replica` in `sid` and start reading it: every
/// `PeerLeft` is logged until the sentinel. The reader hands back the stream
/// and its client sender once the sentinel arrives, so the join stays open
/// until the test drops them.
async fn open_observer(
    replica: &Replica,
    replica_index: usize,
    session: usize,
    sid: &str,
) -> Observer {
    let (mut stream, tx) = open_join(&replica.svc, sid, "burst-observer").await;
    expect_session_connected(&mut stream, SETUP_WAIT).await;
    let log = Arc::new(Mutex::new(ObserverLog::default()));
    let task_log = Arc::clone(&log);
    let reader = tokio::spawn(async move {
        while let Some(item) = stream.next().await {
            match item {
                Ok(msg) => match msg.response_type {
                    Some(ResponseType::PeerLeft(p)) => task_log
                        .lock()
                        .expect("observer log poisoned")
                        .lefts
                        .push((p.connection_id, Instant::now())),
                    Some(ResponseType::PeerJoined(p)) if p.connection_id == SENTINEL => {
                        task_log.lock().expect("observer log poisoned").sentinel = true;
                        return Some((stream, tx));
                    }
                    _ => {}
                },
                Err(status) => {
                    task_log.lock().expect("observer log poisoned").ended =
                        Some(format!("{:?}: {}", status.code(), status.message()));
                    return None;
                }
            }
        }
        task_log.lock().expect("observer log poisoned").ended = Some("ended OK".to_owned());
        None
    });
    Observer {
        replica: replica_index,
        session,
        log,
        reader,
    }
}

/// Precondition of a full-batch shape, read once the observers have joined
/// and before the pass: the sweeper's own member-page read
/// (`scan_member_page`, `SSCAN <members> 0 COUNT SWEEP_MEMBER_PAGE`) returns
/// each session's whole set as one final page whose first
/// `SWEEP_MEMBER_BATCH` ids — the pass's first chunk, which it takes in page
/// order — are exactly that session's crashed members, the two observers
/// after them. So every first-chunk script call removes and announces a full
/// batch.
async fn assert_first_chunks_are_the_crashed_members(tv: &TestValkey, sessions: &[String]) {
    for (i, sid) in sessions.iter().enumerate() {
        let page = scan_member_page(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            sid,
            None,
            SWEEP_MEMBER_PAGE,
        )
        .await
        .expect("the member-page SSCAN must succeed");
        assert!(
            page.next_cursor.is_none(),
            "precondition: session {i}'s member set comes back as one final page, which needs \
             the store's set-max-listpack-entries (128 by default) to be at least {} so the \
             set stays compact",
            SWEEP_MEMBER_BATCH + 2
        );
        assert_eq!(
            page.ids.len(),
            SWEEP_MEMBER_BATCH + 2,
            "precondition: session {i} holds its crashed members and two observers"
        );
        assert_eq!(
            page.ids[..SWEEP_MEMBER_BATCH],
            crashed_cids(i, SWEEP_MEMBER_BATCH)[..],
            "precondition: session {i}'s first chunk is exactly its crashed members, no \
             observer among them; the page is in insertion order only while the set is \
             compact, which needs the store's set-max-listpack-entries (128 by default) to be \
             at least {}",
            SWEEP_MEMBER_BATCH + 2
        );
    }
}

/// The highest count of `instants` (sorted) inside any [`PEAK_WINDOW`].
fn peak_in_window(instants: &[Instant]) -> usize {
    let mut best = 0;
    let mut start = 0;
    for end in 0..instants.len() {
        while instants[end].duration_since(instants[start]) > PEAK_WINDOW {
            start += 1;
        }
        best = best.max(end - start + 1);
    }
    best
}

/// The process's peak resident set (`VmHWM`), where `/proc` has it.
fn peak_rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(|l| l["VmHWM:".len()..].trim().to_owned())
        })
        .unwrap_or_else(|| "n/a".to_owned())
}

// ═════════════════════════════════════════════════════════════════════════════
// The scenario
// ═════════════════════════════════════════════════════════════════════════════

/// Run `shape`'s crash and sweep (see the module header), report the numbers,
/// and assert every observer got every `left` once and its sentinel, and no
/// relay lagged.
async fn burst(tv: &TestValkey, shape: Shape, what: &str) {
    let replicas = [
        Replica::start(tv, "replica-a").await,
        Replica::start(tv, "replica-b").await,
    ];

    // ── Arrange: sessions, the crashed members, the observers ───────────────
    let sessions = create_sessions(tv, &replicas[0].svc, shape).await;
    seed_crashed(tv, &sessions, shape).await;
    let setup_started = Instant::now();
    let pairs: Vec<(usize, usize)> = (0..sessions.len())
        .flat_map(|s| (0..replicas.len()).map(move |r| (r, s)))
        .collect();
    let (replicas_ref, sessions_ref) = (&replicas, &sessions);
    let mut observers: Vec<Observer> = futures::stream::iter(pairs)
        .map(|(r, s)| async move { open_observer(&replicas_ref[r], r, s, &sessions_ref[s]).await })
        .buffer_unordered(SETUP_CONCURRENCY)
        .collect()
        .await;
    let setup_took = setup_started.elapsed();
    let lag_after_setup: Vec<(u64, u64)> = replicas.iter().map(Replica::relays_lagged).collect();
    if shape.big_stale == SWEEP_MEMBER_BATCH {
        assert_first_chunks_are_the_crashed_members(tv, &sessions).await;
    }

    // ── Act: one sweeper pass on replica A ──────────────────────────────────
    let seam = Arc::new(PresenceSweeperSeam::new());
    seam.before_chunk.arm();
    let cancel = CancellationToken::new();
    let sweeper = spawn_presence_sweeper_with_seam(
        replicas[0].svc.valkey.clone(),
        tv.key_schema().clone(),
        crate::common::test_leader(replicas[0].label),
        SweeperBudget::production(),
        Arc::clone(&replicas[0].svc.metrics),
        SWEEP_INTERVAL,
        cancel.clone(),
        Arc::clone(&seam),
    );
    tokio::time::timeout(SETUP_WAIT, seam.before_chunk.wait_until_parked())
        .await
        .unwrap_or_else(|_| panic!("{what}: the pass's first chunk did not park"));
    let pass_started = Instant::now();
    seam.before_chunk.release();
    let passes = tokio::time::timeout(PASS_BUDGET, seam.wait_for_passes(1))
        .await
        .unwrap_or_else(|_| panic!("{what}: the pass did not end within {PASS_BUDGET:?}"));
    let pass_ended = Instant::now();
    let pass_took = pass_ended.duration_since(pass_started);
    cancel.cancel();
    tokio::time::timeout(SETUP_WAIT, sweeper)
        .await
        .expect("the sweeper must stop once cancelled")
        .expect("the sweeper task must not panic");
    let removed = stale_removed(&replicas[0].svc.metrics);
    let batch_sizes = seam.script_batch_sizes();
    let full_batches = batch_sizes
        .iter()
        .filter(|n| **n == SWEEP_MEMBER_BATCH)
        .count();

    // Behind every `left` of its session.
    futures::stream::iter(sessions.iter())
        .for_each_concurrent(SETUP_CONCURRENCY, |sid| publish_probe(tv, sid, SENTINEL))
        .await;

    // Every observer read up to its sentinel, under one deadline.
    let drain_deadline = tokio::time::Instant::now() + DRAIN_BUDGET;
    let mut kept = Vec::new();
    for observer in observers.iter_mut() {
        match tokio::time::timeout_at(drain_deadline, &mut observer.reader).await {
            Ok(Ok(Some(open))) => kept.push(open),
            Ok(Ok(None)) => {}
            Ok(Err(e)) => panic!("{what}: an observer's reader panicked: {e:?}"),
            Err(_) => observer.reader.abort(),
        }
    }

    // ── The numbers ──────────────────────────────────────────────────────────
    let lag: Vec<(u64, u64)> = replicas.iter().map(Replica::relays_lagged).collect();
    let mut missing = 0usize;
    let mut duplicated = 0usize;
    let mut unexpected = 0usize;
    let mut short_observers = 0usize;
    let mut no_sentinel = 0usize;
    let mut ended = Vec::new();
    let mut delivered: Vec<BTreeMap<String, Instant>> = vec![BTreeMap::new(); replicas.len()];
    for observer in &observers {
        let log = observer.log.lock().expect("observer log poisoned");
        let want = crashed_cids(observer.session, shape.stale_in(observer.session));
        let mut got: BTreeMap<&str, usize> = BTreeMap::new();
        for (cid, at) in &log.lefts {
            *got.entry(cid.as_str()).or_default() += 1;
            let first = delivered[observer.replica]
                .entry(cid.clone())
                .or_insert(*at);
            *first = (*first).min(*at);
        }
        let mut short = false;
        for cid in &want {
            match got.remove(cid.as_str()) {
                None => {
                    missing += 1;
                    short = true;
                }
                Some(1) => {}
                Some(n) => duplicated += n - 1,
            }
        }
        unexpected += got.values().sum::<usize>();
        short_observers += usize::from(short);
        no_sentinel += usize::from(!log.sentinel);
        if let Some(how) = &log.ended {
            ended.push(how.clone());
        }
    }
    let mut report = format!(
        "{what}: sessions={} ({} demo x5, {} big with {} crashed each), stale={}, \
         observers={} ({} per replica, {} replicas) opened in {:.1?}\n  pass: outcome={} took {:.3?}, \
         removed={removed}, script calls={} ({full_batches} of {SWEEP_MEMBER_BATCH} ids), \
         removal rate {:.0}/s\n",
        shape.sessions(),
        shape.demo_sessions,
        shape.big_sessions,
        shape.big_stale,
        shape.stale(),
        observers.len(),
        sessions.len(),
        replicas.len(),
        setup_took,
        passes[0].outcome,
        pass_took,
        batch_sizes.len(),
        removed as f64 / pass_took.as_secs_f64(),
    );
    for (r, replica) in replicas.iter().enumerate() {
        let mut at: Vec<Instant> = delivered[r].values().copied().collect();
        at.sort();
        let span = match (at.first(), at.last()) {
            (Some(first), Some(last)) => last.duration_since(*first),
            _ => Duration::ZERO,
        };
        // How far the replica's delivery trailed the pass: its last `left`
        // read relative to the pass's end, as the test saw both.
        let trail = match at.last() {
            Some(last) if *last >= pass_ended => format!("{:.3?} after", *last - pass_ended),
            Some(last) => format!("{:.3?} before", pass_ended - *last),
            None => "n/a".to_owned(),
        };
        report.push_str(&format!(
            "  {}: lefts delivered={} over {:.3?} ({:.0}/s), peak {} in {:?} ({:.0}/s), last \
             {trail} the pass ended; relays lagged presence={} session_deleted={} (after \
             setup {}/{})\n",
            replica.label,
            at.len(),
            span,
            at.len() as f64 / span.as_secs_f64().max(f64::EPSILON),
            peak_in_window(&at),
            PEAK_WINDOW,
            peak_in_window(&at) as f64 / PEAK_WINDOW.as_secs_f64(),
            lag[r].0,
            lag[r].1,
            lag_after_setup[r].0,
            lag_after_setup[r].1,
        ));
    }
    report.push_str(&format!(
        "  delivery: missing={missing} duplicated={duplicated} unexpected={unexpected} \
         observers short={short_observers} without sentinel={no_sentinel} ended early={} \
         {ended:?}\n  process peak RSS {}",
        ended.len(),
        peak_rss()
    ));
    eprintln!("── presence sweep relay burst ──\n{report}");

    // ── Teardown, before asserting, so a failure leaves no join running ─────
    drop(kept);
    drop(observers);
    for replica in &replicas {
        replica.svc.presence_cleanup_tracker.close();
        tokio::time::timeout(DRAIN_BUDGET, replica.svc.presence_cleanup_tracker.wait())
            .await
            .unwrap_or_else(|_| panic!("{what}: {}'s joins did not drain", replica.label));
        replica.relays.cancel();
    }

    // ── Assert ───────────────────────────────────────────────────────────────
    assert_eq!(
        (passes[0].outcome, removed),
        (PRESENCE_SWEEP_OUTCOME_COMPLETED, shape.stale() as u64),
        "precondition: one completed pass removed every crashed member\n{report}"
    );
    if shape.big_stale == SWEEP_MEMBER_BATCH {
        assert_eq!(
            full_batches, shape.big_sessions,
            "precondition: the pass made one {SWEEP_MEMBER_BATCH}-id script call per \
             full-batch session — the first chunk of the one member page the read before the \
             pass showed holding exactly that session's crashed members\n{report}"
        );
    }
    assert_eq!(
        (missing, duplicated, unexpected, no_sentinel, ended.len()),
        (0, 0, 0, 0, 0),
        "every observer must receive a PeerLeft for every removed member of its session, \
         exactly once, and its sentinel, its stream still open\n{report}"
    );
    assert_eq!(
        lag,
        vec![(0, 0); replicas.len()],
        "no relay, presence or session-deleted, on either surviving replica may lag\n{report}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

/// [`CRASH_SCALE_FULL_BATCH`]: about 10,000 stale members in 102-member
/// sessions, one full `SWEEP_MEMBER_BATCH`-id script call each. Every observer
/// on both surviving replicas gets every `left` of the one-pass sweep once,
/// and no relay lags.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_burst_at_crash_scale_in_full_batch_sessions_reaches_every_observer_without_relay_lag(
) {
    scenario(async {
        let tv = TestValkey::setup().await;
        burst(
            &tv,
            CRASH_SCALE_FULL_BATCH,
            "crash scale, full-batch sessions",
        )
        .await;
    })
    .await;
}

/// [`CRASH_SCALE_DEMO`]: about 10,000 stale members in 5-member sessions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: crash-scale measurement (10,002 stale members, 6,668 JoinSession streams), too \
            heavy for the gating suite; run with cargo nextest run -p live-edit --test integration \
            --features test-support,test-fault-injection --run-ignored all -E \
            'test(/presence_sweep_relay_burst_test/)'"]
async fn sweep_burst_at_crash_scale_in_demo_sessions_reaches_every_observer_without_relay_lag() {
    tokio::time::timeout(CRASH_SCALE_BUDGET, async {
        let tv = TestValkey::setup().await;
        burst(&tv, CRASH_SCALE_DEMO, "crash scale, demo sessions").await;
    })
    .await
    .unwrap_or_else(|_| panic!("the crash-scale run did not finish within {CRASH_SCALE_BUDGET:?}"));
}

/// [`CRASH_SCALE_MIX`]: about 10,000 stale members in demo and 50-member
/// sessions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: crash-scale measurement (9,996 stale members, 3,604 JoinSession streams), too \
            heavy for the gating suite; run with cargo nextest run -p live-edit --test integration \
            --features test-support,test-fault-injection --run-ignored all -E \
            'test(/presence_sweep_relay_burst_test/)'"]
async fn sweep_burst_at_crash_scale_in_mixed_sessions_reaches_every_observer_without_relay_lag() {
    tokio::time::timeout(CRASH_SCALE_BUDGET, async {
        let tv = TestValkey::setup().await;
        burst(&tv, CRASH_SCALE_MIX, "crash scale, mixed sessions").await;
    })
    .await
    .unwrap_or_else(|_| panic!("the crash-scale run did not finish within {CRASH_SCALE_BUDGET:?}"));
}
