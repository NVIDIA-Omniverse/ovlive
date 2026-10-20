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

// The JoinSession presence lifecycle (SDD SR-4): index-first registration, the
// per-connection lease keeper that renews the lease from registration to
// teardown and re-registers its connection after an eviction (self-rejoin), the
// paged, cancel-aware membership replay (SR-4.4), and the fenced teardown that
// runs on every exit after the join's first store write.
//
// Every test drives `handle_join_session` against a real Valkey and checks the
// store, the presence channel and the join's own stream, never a log line:
//
//   * the connection's presence keys — member entry, info hash, lease, and the
//     teardown marker `:gone` that teardown's unregister arms;
//   * the session index `le:presence:active_sessions`;
//   * the presence-events channel, read by an independent subscriber (`Tap`),
//     which is how a peer on another replica sees `joined` / `left`.
//
// The renewal index re-add gate's tests also read and plant the session's
// index-renewal record on the service's own router (SR-7.2b). The replay tests
// also read the replay seam's observations — pipelines issued, page sizes,
// completed sends and how the replay ended — and the join-latency count.
//
// ── How events are observed ──────────────────────────────────────────────────
//
// "Exactly these events, and nothing else" is proven with a fence rather than a
// sleep: once everything under test has finished (the join tasks have drained
// from `presence_cleanup_tracker`), the test publishes a fence message on the
// channel and collects everything that arrives before it. A single node
// delivers one channel's messages to one subscriber in the order it executed
// the PUBLISHes, and every script's PUBLISH ran before the drain finished, so
// every event under test is in the collected list. The two cluster ordering
// tests cannot rely on that and wait for the events they expect instead.
//
// ── Seams ────────────────────────────────────────────────────────────────────
//
// Races that close too fast to hit are held open with the test-only presence
// seams (`JoinRegistrationSeam`, `PresenceReplaySeam`, `LeaseKeeperSeam`,
// `PresenceChannelSeam`), each a one-shot gate armed immediately before the
// arrival it should park; `PresenceChannelSeam` can also fail the next
// `SUBSCRIBE` without sending it, `LeaseKeeperSeam` parks a keeper step
// before its refresh or a rejoin before its register, and `PresenceReplaySeam`
// parks the join before its membership snapshot, or the replay before its
// first page or between pages. Every test that uses one is compiled only with
// `--features test-fault-injection`, and is compiled out without it.
//
// Evictions go through `storage::presence::evict_connection_for_test`, never
// the live presence sweeper, which removes a connection only once its lease has
// expired and only on its own leader-gated cadence, so a test cannot choose the
// moment of the eviction. The one exception is the blocked-replay test, which runs
// the live sweeper on purpose: a lease that lapsed during the held replay would
// be swept and announced as a `PeerLeft`, the failure that test guards against.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every wait is a bounded poll on a store condition, a seam's parked latch, or
// an observed event: `SETTLE` (10 s) unless named otherwise. The healthy path
// returns from each in milliseconds. Some waits are deliberately long, because
// the behaviour under test takes that long:
//
//   * `REPLAY_HOLD` (2 × `SHORT_LEASE_TTL_SECS` + 1 s = 7 s) — the blocked
//     replay must outlive two lease TTLs;
//   * `HELD_WATCH` (1 s) — joins that must stay held behind an in-flight
//     `SUBSCRIBE` or `UNSUBSCRIBE` are watched for that long, a window far
//     longer than an unheld join takes to register;
//   * a register, index SADD, store-clock read, membership snapshot or keeper
//     refresh held behind a `ConnStall` fails only at the fixture's 5 s
//     command timeout (a keeper
//     refresh first waits for its first renewal, 0.5–1 s after the join, or
//     its next 1 s tick); waiting for one such timeout stays
//     within `SETTLE`. A join whose keeper is parked, or blocked on a held
//     rejoin register, is torn down only after teardown's bounded wait for that
//     keeper (the 5 s command timeout + 1 s), which `TEARDOWN_WAIT` (20 s)
//     bounds, and whose lower bound the refresh race tests assert (load can
//     only lengthen it);
//   * the keeper-renewal checks shorten the lease and await a renewal, one
//     1 s refresh interval each, up to three rounds;
//   * `LATE_STEP_HOLD` (500 ms) — one keeper step is held that long on
//     purpose, the late step whose gate check the late-step test delays;
//   * an eviction is followed by the keeper's rejoin within one 1 s refresh
//     interval; the expired-rejoin test waits one `SHORT_LEASE_TTL_SECS`
//     (3 s) for the store clock to pass the parked register's deadline, and
//     the expired-marker test as long for the teardown marker to expire, each
//     within `SETTLE`;
//   * a keeper's next step, after a rejoin, arrives at `before_refresh` at
//     once (the parked step overran its tick), awaited within `SETTLE`;
//   * the replay tests read up to 250 messages per group under one `SETTLE`
//     deadline per group, never one per message; a replay whose pipeline is
//     held behind a `ConnStall` is cancelled and its stall released at once,
//     so teardown's unregister, queued behind it, needs no command timeout.
//     The store-error test holds the pipeline for the 5 s command timeout,
//     within `SETTLE`, and releases the stall once the replay has failed.
//
// Summed at their full budgets, the waits of the stall tests and the cluster
// ordering tests (200 trials, and 20 rejoins) exceed nextest's 120 s
// hang-killer, so every test
// body runs under one `SCENARIO_BUDGET` (100 s) deadline: a stuck test fails
// with its own message before the killer would discard it. The heaviest healthy
// paths are about 10 s (a keeper refresh failure followed by three renewal
// rounds; a rejoin register held past teardown's unregister and then past the
// teardown marker's expiry), about 6–7 s (blocked replay, teardown racing a
// parked refresh or, on a client disconnect, a parked rejoin, and the other
// rejoin stall tests) and about 4–5 s (the other stall tests and the expired
// registers).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, PubsubInterface, SetsInterface,
};
use futures::StreamExt;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Status};

use live_edit::grpc::proto::join_session_response;
use live_edit::grpc::service::LiveEditService;
use live_edit::presence::{publish_presence_event, ConnectedUser, PresenceEvent};
use live_edit::storage::client::{FredErrOrigin, POOL_LABEL_DATA};
use live_edit::storage::presence::{evict_connection_for_test, unregister_connection};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, release_all, TestValkey,
};

#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::presence::{
    JoinRegistrationSeam, LeaseKeeperSeam, ObservedReplayResult, PresenceChannelSeam,
    PresenceReplaySeam, PresenceSeamGate,
};
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::proto::DeleteSessionRequest;
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::proto::JoinSessionResponse;
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::session::handle_delete_session;
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::BearerToken;
#[cfg(feature = "test-fault-injection")]
use live_edit::presence::{spawn_presence_sweeper, SweeperBudget};
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::presence::REPLAY_PAGE;
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::presence::{remove_session_from_index, store_time_ms};
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::test_support::ConnStall;
#[cfg(feature = "test-fault-injection")]
use tonic::Request;

use super::helpers::*;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on any single wait for a store condition, a seam, or an event.
const SETTLE: Duration = Duration::from_secs(10);

/// Bound on waiting for a teardown that first waits out a parked keeper (its
/// bound is the 5 s command timeout + 1 s). With headroom.
#[cfg(feature = "test-fault-injection")]
const TEARDOWN_WAIT: Duration = Duration::from_secs(20);

// The whole-test deadline, `SCENARIO_BUDGET` (100 s, below nextest's 120 s
// hang-killer; see the header), and `scenario`, which applies it, live in
// `super::helpers`.

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(20);

/// The production lease TTL, for tests where the lease must not lapse and the
/// teardown marker (armed for one lease TTL) must still be readable at the end.
const LEASE_TTL_SECS: u64 = live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS;

/// The production refresh interval. A keeper's first renewal comes half an
/// interval to one interval after its join (7.5–15 s), so a test that uses it
/// and finishes well inside 7.5 s sees no keeper step. A test that a keeper
/// step would break uses [`QUIET_REFRESH_SECS`] instead.
const REFRESH_SECS: u64 = live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS;

/// A refresh interval under which no keeper step runs inside a test: a
/// keeper's first renewal comes at least half an interval (30 min) after its
/// join, far past [`SCENARIO_BUDGET`]. Paired with [`LEASE_TTL_SECS`] it is
/// below the two intervals per TTL that configuration requires, which is
/// harmless here: no test that uses it holds a connection for a lease TTL.
const QUIET_REFRESH_SECS: u64 = 3600;

/// A short lease TTL, so a lease the keeper failed to renew lapses inside the
/// test. Three refresh intervals, so one late keeper step on a loaded runner
/// does not lapse it.
#[cfg(feature = "test-fault-injection")]
const SHORT_LEASE_TTL_SECS: u64 = 3;

/// A short keeper refresh interval.
const SHORT_REFRESH_SECS: u64 = 1;

/// How long the blocked-replay test holds the replay: past two lease TTLs.
#[cfg(feature = "test-fault-injection")]
const REPLAY_HOLD: Duration = Duration::from_secs(2 * SHORT_LEASE_TTL_SECS + 1);

/// The presence sweeper's interval in the blocked-replay test.
#[cfg(feature = "test-fault-injection")]
const SWEEP_INTERVAL: Duration = Duration::from_millis(200);

/// Joins `join_session_cluster_peer_message_never_precedes_its_join_or_follows_its_leave`
/// runs.
const ORDERING_TRIALS: usize = 200;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════
//
// `scenario`, `poll_until`, `wait_parked`, `spawn_router_feed`,
// `register_remote_peer`, `evict`, `publish_probe` and `expect_peer_joined`
// live in `super::helpers`, shared with the replay-duplicate tests.

/// A service over `tv` with the given lease TTL and keeper refresh interval.
fn service(tv: &TestValkey, lease_ttl_secs: u64, refresh_secs: u64) -> LiveEditService {
    LiveEditService {
        presence_lease_ttl_secs: lease_ttl_secs,
        presence_refresh_interval_secs: refresh_secs,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        ..build_service(tv, Arc::new(AllowAllPermissions))
    }
}

/// The four presence seams installed on a [`seamed_service`], every gate
/// disarmed and no failure queued.
#[cfg(feature = "test-fault-injection")]
struct Seams {
    registration: Arc<JoinRegistrationSeam>,
    replay: Arc<PresenceReplaySeam>,
    keeper: Arc<LeaseKeeperSeam>,
    channel: Arc<PresenceChannelSeam>,
}

/// [`service`] with every presence seam installed.
#[cfg(feature = "test-fault-injection")]
fn seamed_service(
    tv: &TestValkey,
    lease_ttl_secs: u64,
    refresh_secs: u64,
) -> (LiveEditService, Seams) {
    let seams = Seams {
        registration: Arc::new(JoinRegistrationSeam::new()),
        replay: Arc::new(PresenceReplaySeam::new()),
        keeper: Arc::new(LeaseKeeperSeam::new()),
        channel: Arc::new(PresenceChannelSeam::new()),
    };
    let svc = LiveEditService {
        join_registration_seam_for_test: Some(Arc::clone(&seams.registration)),
        presence_replay_seam_for_test: Some(Arc::clone(&seams.replay)),
        lease_keeper_seam_for_test: Some(Arc::clone(&seams.keeper)),
        presence_channel_seam_for_test: Some(Arc::clone(&seams.channel)),
        ..service(tv, lease_ttl_secs, refresh_secs)
    };
    (svc, seams)
}

#[cfg(feature = "test-fault-injection")]
fn delete_session_request(session_id: &str) -> Request<DeleteSessionRequest> {
    let mut request = Request::new(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ── Store reads ──────────────────────────────────────────────────────────────
//
// Through `super::helpers::store_reader` (the maintenance pool), so a read
// never queues behind a data-pool stall and never moves the data pool's
// round-robin cursor (the ambiguous-register test relies on knowing which
// data connection the join's next command takes). `presence_keys`,
// `is_indexed` and `is_member` live in `super::helpers`.

async fn members(tv: &TestValkey, session_id: &str) -> Vec<String> {
    store_reader(tv)
        .smembers(tv.key_schema().presence_members(session_id))
        .await
        .expect("SMEMBERS must succeed")
}

async fn exists(tv: &TestValkey, key: &str) -> bool {
    store_reader(tv)
        .exists(key)
        .await
        .expect("EXISTS must succeed")
}

/// The lease's remaining TTL in milliseconds (`-2` when absent).
async fn pttl(tv: &TestValkey, key: &str) -> i64 {
    store_reader(tv).pttl(key).await.expect("PTTL must succeed")
}

/// Teardown's end state for `cid` (SR-4.3): no member entry, info hash or
/// lease, and the teardown marker armed for at most one lease TTL.
async fn assert_torn_down(
    tv: &TestValkey,
    session_id: &str,
    cid: &str,
    lease_ttl_secs: u64,
    what: &str,
) {
    let ks = tv.key_schema();
    assert!(
        !is_member(tv, session_id, cid).await,
        "{what}: teardown must remove the member entry"
    );
    assert!(
        !exists(tv, &ks.presence_user(session_id, cid)).await,
        "{what}: teardown must delete the info hash"
    );
    assert!(
        !exists(tv, &ks.presence_lease(session_id, cid)).await,
        "{what}: teardown must delete the lease"
    );
    let gone_ttl_ms = pttl(tv, &ks.presence_gone(session_id, cid)).await;
    assert!(
        gone_ttl_ms > 0 && gone_ttl_ms <= (lease_ttl_secs * 1000) as i64,
        "{what}: teardown must arm the :gone marker with a TTL of at most one lease TTL \
         ({lease_ttl_secs} s); PTTL is {gone_ttl_ms} ms"
    );
}

// ── Presence-channel tap ─────────────────────────────────────────────────────
//
// `Tap` (an independent subscriber on one session's presence channel) and the
// `is_joined` / `is_left` / `is_message` event matchers live in
// `super::helpers`, shared with the presence sweeper tests.

fn count(events: &[PresenceEvent], pred: impl Fn(&PresenceEvent) -> bool) -> usize {
    events.iter().filter(|e| pred(e)).count()
}

// ═════════════════════════════════════════════════════════════════════════════
// Blocked replay (PRD AC-3)
// ═════════════════════════════════════════════════════════════════════════════

/// A join whose membership replay is held for longer than two lease TTLs, with
/// the presence sweeper running, keeps its lease throughout: the keeper renews
/// it from registration, not from the end of the replay. The sweeper never
/// finds it lapsed, so no peer sees a `PeerLeft`, and the connection is still a
/// member once the replay completes. When the client then leaves, peers see
/// exactly one `PeerLeft`, from teardown.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_replay_held_past_two_lease_ttls_keeps_its_lease_and_announces_no_leave() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, SHORT_LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "blocked-replay", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let sweeper_cancel = CancellationToken::new();
        let sweeper = spawn_presence_sweeper(
            tv.client().clone(),
            tv.key_schema().clone(),
            crate::common::test_leader("blocked-replay-sweeper"),
            SweeperBudget::production(),
            Arc::clone(&svc.metrics),
            SWEEP_INTERVAL,
            sweeper_cancel.clone(),
        );

        let gate = &seams.replay.before_first_send;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "blocked-replay-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(gate, "the membership replay").await;

        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        let held_since = Instant::now();
        while held_since.elapsed() < REPLAY_HOLD {
            assert!(
                exists(&tv, &lease_key).await,
                "the lease must exist throughout the held replay; it was gone {:?} into the \
                 hold (lease TTL {SHORT_LEASE_TTL_SECS} s)",
                held_since.elapsed()
            );
            tokio::time::sleep(POLL).await;
        }
        gate.release();

        // Only the bidirectional loop publishes a client's message, so the tap
        // seeing it proves the join got past the replay.
        tx.send(encode_join_frame(&join_message(b"after-replay".to_vec())))
            .expect("the client channel must accept a message");
        let before_leave = tap
            .until("the client's message after the replay", |e| {
                is_message(e, &cid)
            })
            .await;
        assert_eq!(
            count(&before_leave, |e| is_left(e, &cid)),
            0,
            "no PeerLeft may be published for a connection whose replay was merely slow; got \
             {before_leave:?}"
        );
        assert!(
            is_member(&tv, &session_id, &cid).await,
            "the connection must still be a member after the held replay"
        );

        drop(tx);
        let (_, status) = read_to_end(&mut stream, "client close after the replay").await;
        assert!(
            status.is_none(),
            "the join must end OK when its client closes; got {status:?}"
        );
        drain_tracker(&svc).await;
        let after_leave = tap.drain(&tv).await;
        assert_eq!(
            count(&after_leave, |e| is_left(e, &cid)),
            1,
            "leaving must announce exactly one PeerLeft, teardown's; got {after_leave:?}"
        );

        sweeper_cancel.cancel();
        let _ = tokio::time::timeout(SETTLE, sweeper).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Shutdown mid-join
// ═════════════════════════════════════════════════════════════════════════════

/// Shutdown begins while a join sits between its registration and
/// `acquire_session_cancel`, in a session with no token entry on this replica.
/// The join stops at its check after `acquire_session_cancel`: teardown
/// removes what the registration wrote and arms the teardown marker, the
/// cleanup tracker drains, and the stream ends `UNAVAILABLE "server shutting
/// down"` with no `SessionConnected` — the ending the client retries.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_shutdown_during_registration_stops_before_session_connected_and_tears_down() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let (session_id, _unregistered_token) =
            create_session(&tv, &svc, "shutdown-mid-join", false).await;

        let gate = &seams.registration.before_session_cancel;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "shutdown-app").await;
        wait_parked(gate, "the join, after its register").await;
        let registered = members(&tv, &session_id).await;
        let [cid] = registered.as_slice() else {
            panic!("precondition: the parked join is registered; members {registered:?}");
        };

        svc.shutdown.cancel();
        gate.release();

        let (delivered, status) = read_to_end(&mut stream, "shutdown mid-join").await;
        assert_ending(
            Some((Code::Unavailable, SHUTTING_DOWN)),
            status,
            "shutdown mid-join",
        );
        assert!(
            delivered.is_empty(),
            "the join must stop before SessionConnected; got {delivered:?}"
        );
        drain_tracker(&svc).await;
        assert_torn_down(&tv, &session_id, cid, LEASE_TTL_SECS, "shutdown mid-join").await;
        assert!(
            !is_indexed(&tv, &session_id).await,
            "the session's only member left, so teardown must remove it from the index"
        );
    })
    .await;
}

/// Shutdown and a session cancel **both** land while a join sits between its
/// registration and `acquire_session_cancel`. The session's token is already
/// registered on this replica, so the join shares it; both tokens are
/// cancelled before the join is released, so its check after
/// `acquire_session_cancel` finds both. Shutdown is checked first: the stream
/// ends `UNAVAILABLE "server shutting down"` with no `SessionConnected` — not
/// the `OK` a deletion alone gets — and teardown runs as for either alone.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_shutdown_and_session_cancel_during_registration_end_unavailable() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let (session_id, session_token) =
            create_session(&tv, &svc, "shutdown-and-delete-mid-join", true).await;

        let gate = &seams.registration.before_session_cancel;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "both-app").await;
        wait_parked(gate, "the join, after its register").await;
        let registered = members(&tv, &session_id).await;
        let [cid] = registered.as_slice() else {
            panic!("precondition: the parked join is registered; members {registered:?}");
        };

        svc.shutdown.cancel();
        session_token.cancel();
        gate.release();

        let what = "shutdown and session cancel mid-join";
        let (delivered, status) = read_to_end(&mut stream, what).await;
        assert_ending(Some((Code::Unavailable, SHUTTING_DOWN)), status, what);
        assert!(
            delivered.is_empty(),
            "the join must stop before SessionConnected; got {delivered:?}"
        );
        drain_tracker(&svc).await;
        assert_torn_down(&tv, &session_id, cid, LEASE_TTL_SECS, what).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Teardown racing a keeper refresh (PRD AC-5, refresh half)
// ═════════════════════════════════════════════════════════════════════════════

/// The status message of a shutdown ending.
#[cfg(feature = "test-fault-injection")]
const SHUTTING_DOWN: &str = "server shutting down";

/// Assert a join stream's terminal status: `None` for an `OK` end, or the code
/// and message it must carry. Two endings share `UNAVAILABLE` — a shutdown and
/// a session found missing from the live store — so the message is what says
/// which one it was.
#[cfg(feature = "test-fault-injection")]
fn assert_ending(want: Option<(Code, &str)>, got: Option<Status>, what: &str) {
    match (want, got) {
        (None, None) => {}
        (Some((code, message)), Some(status)) => {
            assert_eq!(status.code(), code, "{what}: got {status:?}");
            assert_eq!(status.message(), message, "{what}");
        }
        (want, got) => panic!("{what}: the join must end {want:?} (None is OK); got {got:?}"),
    }
}

/// What ends an established join.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum Trigger {
    /// The client drops its response stream. The join sees it gone at its next
    /// step that sends or races `tx.closed()`: the `SessionConnected` send, the
    /// replay's next pipeline or send, or the loop's `tx.closed()` arm.
    ClientDisconnect,
    /// `DeleteSession` on this replica.
    DeleteSession,
    /// The session's cancellation token alone.
    SessionTokenCancel,
    /// The service shutdown token alone.
    ServiceShutdown,
    /// The service shutdown token, then the session token, with nothing in
    /// between that yields — so the join, woken by the first, normally finds
    /// both cancelled, and must end as a shutdown: its session still exists
    /// and its client has to rejoin, wherever the ingress routes it. Shutdown
    /// first, so a join that
    /// does wake between the two still sees only the shutdown, and a correct
    /// join can never end any other way.
    ShutdownAndSessionToken,
    /// The session's cancellation token alone, cancelled through the map entry
    /// with the cause the `session_tokens` sweeper and `acquire_session_cancel`
    /// record for a session missing from the live store with no deletion
    /// marker (`cancel_absent_from_store`). The session itself stays in the
    /// store.
    AbsentFromStore,
}

#[cfg(feature = "test-fault-injection")]
impl Trigger {
    /// The status the join's stream must end with: `UNAVAILABLE "server
    /// shutting down"` for a shutdown, which the client retries (where the
    /// retry lands is the ingress's choice), even when a session cancel is
    /// observable with it; the retryable `session_absent_status()` for a
    /// session missing from the live store, so the rejoin can recover it;
    /// `OK` (`None`) for a deleted session, which it must not rejoin.
    fn ending(self) -> Option<(Code, &'static str)> {
        match self {
            Trigger::ServiceShutdown | Trigger::ShutdownAndSessionToken => {
                Some((Code::Unavailable, SHUTTING_DOWN))
            }
            Trigger::AbsentFromStore => Some((
                Code::Unavailable,
                "session is missing from the live store; retry so it can be recovered",
            )),
            Trigger::ClientDisconnect | Trigger::DeleteSession | Trigger::SessionTokenCancel => {
                None
            }
        }
    }
}

/// Fire `trigger`. A disconnect drops `stream` and returns `None`; every other
/// trigger hands the stream back for the caller to read to its end.
#[cfg(feature = "test-fault-injection")]
async fn fire(
    trigger: Trigger,
    svc: &LiveEditService,
    session_id: &str,
    session_token: &CancellationToken,
    stream: JoinResponseStream,
) -> Option<JoinResponseStream> {
    match trigger {
        Trigger::ClientDisconnect => {
            drop(stream);
            return None;
        }
        Trigger::DeleteSession => {
            handle_delete_session(svc, delete_session_request(session_id))
                .await
                .expect("DeleteSession must succeed");
        }
        Trigger::SessionTokenCancel => session_token.cancel(),
        Trigger::ServiceShutdown => svc.shutdown.cancel(),
        Trigger::ShutdownAndSessionToken => {
            svc.shutdown.cancel();
            session_token.cancel();
        }
        Trigger::AbsentFromStore => svc
            .session_tokens
            .get(session_id)
            .expect("the session's token is registered on this replica")
            .cancel_absent_from_store(),
    }
    Some(stream)
}

/// An established join's keeper is parked inside a step, before its refresh,
/// when `trigger` ends the join. Teardown waits out its bound for the keeper
/// — the teardown marker appears no sooner than one command timeout after the
/// trigger — then unregisters, and only then is the refresh let go, so it
/// reaches the store after the unregister. Afterwards the member entry, info
/// hash and lease are gone, peers saw exactly one `PeerLeft` and no
/// `PeerJoined` after it, neither the join task nor the keeper is left
/// running, and the stream ended as `trigger` must end it
/// ([`Trigger::ending`]).
#[cfg(feature = "test-fault-injection")]
async fn teardown_racing_a_parked_refresh_leaves_nothing_behind(trigger: Trigger) {
    let what = format!("teardown on {trigger:?} racing a parked refresh");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
    let (session_id, token) = create_session(&tv, &svc, "teardown-vs-refresh", true).await;
    let mut tap = Tap::open(&tv, &session_id).await;

    let (mut stream, _tx) = open_join(&svc, &session_id, "race-app").await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;
    let gate = &seams.keeper.before_refresh;
    gate.arm();
    wait_parked(gate, "the keeper's refresh").await;

    let command_timeout = tv
        .client()
        .pool
        .next()
        .perf_config()
        .default_command_timeout;
    let fired_at = Instant::now();
    let stream = fire(trigger, &svc, &session_id, &token, stream).await;
    let gone_key = tv.key_schema().presence_gone(&session_id, &cid);
    let (tv_ref, gone_ref) = (&tv, gone_key.as_str());
    poll_until(
        "teardown's unregister, after its bounded wait for the parked keeper",
        TEARDOWN_WAIT,
        move || exists(tv_ref, gone_ref),
    )
    .await;
    // Teardown waits for the keeper (bounded by the command timeout plus
    // slack) before it unregisters, so the marker cannot appear any sooner.
    // Load only lengthens this, so the lower bound cannot fail spuriously.
    let unregistered_after = fired_at.elapsed();
    assert!(
        unregistered_after >= command_timeout,
        "{what}: teardown must wait for the parked keeper before unregistering; the \
         teardown marker appeared {unregistered_after:?} after the trigger, under the \
         {command_timeout:?} command timeout"
    );
    gate.release();

    if let Some(mut stream) = stream {
        let (_, status) = read_to_end(&mut stream, &what).await;
        assert_ending(trigger.ending(), status, &what);
    }
    drain_tracker(&svc).await;

    let events = tap.drain(&tv).await;
    assert_eq!(
        count(&events, |e| is_left(e, &cid)),
        1,
        "{what}: peers must see exactly one PeerLeft; got {events:?}"
    );
    let left_at = events
        .iter()
        .position(|e| is_left(e, &cid))
        .expect("counted above");
    assert_eq!(
        count(&events[left_at..], |e| is_joined(e, &cid)),
        0,
        "{what}: no PeerJoined may follow the teardown's PeerLeft; got {events:?}"
    );
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
    tap.close().await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_client_disconnect_racing_a_parked_refresh_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_refresh_leaves_nothing_behind(
        Trigger::ClientDisconnect,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_delete_session_racing_a_parked_refresh_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_refresh_leaves_nothing_behind(
        Trigger::DeleteSession,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_session_token_cancel_racing_a_parked_refresh_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_refresh_leaves_nothing_behind(
        Trigger::SessionTokenCancel,
    ))
    .await;
}

/// The shutdown trigger of AC-5: only `svc.shutdown` is cancelled.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_service_shutdown_racing_a_parked_refresh_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_refresh_leaves_nothing_behind(
        Trigger::ServiceShutdown,
    ))
    .await;
}

/// Shutdown and a session cancel together: the bidirectional loop checks its
/// shutdown arm before its session arm, so the join ends `UNAVAILABLE`, not
/// `OK`, and tears down as for either alone.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_shutdown_and_session_cancel_racing_a_parked_refresh_ends_unavailable() {
    scenario(teardown_racing_a_parked_refresh_leaves_nothing_behind(
        Trigger::ShutdownAndSessionToken,
    ))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The keeper stops on its own tokens
// ═════════════════════════════════════════════════════════════════════════════

/// Which of the keeper's own tokens is cancelled.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum KeeperStop {
    ServiceShutdown,
    SessionCancel,
}

#[cfg(feature = "test-fault-injection")]
impl KeeperStop {
    /// The status the join then ends with: `UNAVAILABLE` for a shutdown, `OK`
    /// (`None`) for a deleted session.
    fn ending(self) -> Option<(Code, &'static str)> {
        match self {
            KeeperStop::ServiceShutdown => Some((Code::Unavailable, SHUTTING_DOWN)),
            KeeperStop::SessionCancel => None,
        }
    }
}

/// With the join task parked at the replay's `before_first_send` gate (a test
/// park point, which is not cancellation-aware), cancelling the service
/// shutdown token or the session token stops the lease keeper by itself: the
/// tracker drops from the join task and its keeper to the join task alone.
/// Once the gate is released, the join ends — `UNAVAILABLE` for a shutdown,
/// `OK` for a deletion — and tears down. The joiner is the session's only
/// member, so its replay has no step that races the tokens, and the join's
/// loop is what ends it.
#[cfg(feature = "test-fault-injection")]
async fn keeper_stops_on_its_own_token_while_the_join_is_parked(stop: KeeperStop) {
    let what = format!("keeper stop on {stop:?}");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
    let (session_id, token) = create_session(&tv, &svc, "keeper-stop", true).await;

    let gate = &seams.replay.before_first_send;
    gate.arm();
    let (mut stream, _tx) = open_join(&svc, &session_id, "keeper-stop-app").await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;
    wait_parked(gate, "the membership replay").await;
    assert_eq!(
        svc.presence_cleanup_tracker.len(),
        2,
        "precondition: the join task and its lease keeper are running"
    );

    match stop {
        KeeperStop::ServiceShutdown => svc.shutdown.cancel(),
        KeeperStop::SessionCancel => token.cancel(),
    }
    let tracker = &svc.presence_cleanup_tracker;
    poll_until(
        &format!("{what}: the keeper exiting while the join is parked"),
        SETTLE,
        move || async move { tracker.len() == 1 },
    )
    .await;
    gate.release();

    let (_, status) = read_to_end(&mut stream, &what).await;
    assert_ending(stop.ending(), status, &what);
    drain_tracker(&svc).await;
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_stops_on_service_shutdown_while_the_join_is_parked_in_replay() {
    scenario(keeper_stops_on_its_own_token_while_the_join_is_parked(
        KeeperStop::ServiceShutdown,
    ))
    .await;
}

/// `live_edit_presence_lease_refresh_failures_total` on `svc`'s metrics.
fn lease_refresh_failures(svc: &LiveEditService) -> f64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(&svc.metrics),
        "live_edit_presence_lease_refresh_failures_total",
        &[("service", svc.service_label.as_str())],
    )
    .expect("the lease refresh failure cell is seeded in Metrics::new")
}

/// A lease TTL [`LEASE_TTL_SECS`] would be renewed back to reads at least this
/// much; far above the [`SHORTENED_LEASE_SECS`] a test plants.
const RENEWED_PTTL_FLOOR_MS: i64 = (LEASE_TTL_SECS as i64 - 10) * 1000;

/// A lease TTL planted so a later read can tell a renewal from no renewal.
const SHORTENED_LEASE_SECS: i64 = 5;

/// Shorten the lease to [`SHORTENED_LEASE_SECS`], then wait for a renewal to
/// bring it back above [`RENEWED_PTTL_FLOOR_MS`]: only the lease keeper renews
/// it, so this proves a keeper step ran after the call.
async fn expect_a_later_renewal(tv: &TestValkey, lease_key: &str, what: &str) {
    let shortened: bool = store_reader(tv)
        .expire(lease_key, SHORTENED_LEASE_SECS, None)
        .await
        .expect("EXPIRE must succeed");
    assert!(shortened, "{what}: precondition: the lease exists");
    let (tv_ref, key) = (tv, lease_key);
    poll_until(what, SETTLE, move || async move {
        pttl(tv_ref, key).await > RENEWED_PTTL_FLOOR_MS
    })
    .await;
}

/// The keeper stops on a `Gone` refresh alone. With no token cancelled, the
/// live connection is unregistered straight through the store (arming its
/// teardown marker); the keeper's next step finds the marker and the keeper
/// exits, while the join task itself stays in its loop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_stops_by_itself_once_its_connection_is_torn_down_in_the_store() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "keeper-gone", true).await;
        let (mut stream, tx) = open_join(&svc, &session_id, "keeper-gone-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        assert_eq!(
            svc.presence_cleanup_tracker.len(),
            2,
            "precondition: the join task and its lease keeper are running"
        );

        let unregistered = unregister_connection(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            &session_id,
            &cid,
            LEASE_TTL_SECS,
            &PresenceEvent::PeerLeft {
                connection_id: cid.clone(),
            }
            .encode(),
        )
        .await
        .expect("unregistering the live connection must succeed");
        assert!(
            unregistered.removed,
            "precondition: the connection was a member"
        );
        let tracker = &svc.presence_cleanup_tracker;
        poll_until(
            "the keeper stopping on its Gone refresh",
            SETTLE,
            move || async move { tracker.len() == 1 },
        )
        .await;

        drop(tx);
        read_to_ok_end(&mut stream, "client close after the keeper stopped").await;
        drain_tracker(&svc).await;
    })
    .await;
}

/// A keeper step whose refresh fails — held behind a stalled data pool past the
/// command timeout — counts one lease refresh failure, and the keeper goes on:
/// it is still running, and a later step renews the lease.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_keeps_renewing_after_a_failed_refresh() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "keeper-error", true).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "keeper-error-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        assert_eq!(lease_refresh_failures(&svc), 0.0);

        let stalls = tv.stall_data_pool().await;
        let svc_ref = &svc;
        poll_until(
            "a keeper refresh timing out against the stalled data pool",
            SETTLE,
            move || async move { lease_refresh_failures(svc_ref) >= 1.0 },
        )
        .await;
        release_all(stalls).await;

        // At most two refreshes sat behind the stall — the failed step's, which
        // still runs once released, and the step that followed at once on the
        // missed tick — and either may land after the release. Three
        // shorten-and-renew rounds therefore include at least one renewal by a
        // keeper step issued after the release.
        for round in 1..=3 {
            expect_a_later_renewal(
                &tv,
                &lease_key,
                &format!("renewal round {round} after the failed step"),
            )
            .await;
        }
        assert_eq!(
            svc.presence_cleanup_tracker.len(),
            2,
            "the keeper must still be running after a failed step"
        );
        assert_eq!(
            lease_refresh_failures(&svc),
            1.0,
            "exactly the one stalled step failed"
        );

        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_stops_on_session_cancel_while_the_join_is_parked_in_replay() {
    scenario(keeper_stops_on_its_own_token_while_the_join_is_parked(
        KeeperStop::SessionCancel,
    ))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Registration failures that still tear down (SR-4.1, Policy 3)
// ═════════════════════════════════════════════════════════════════════════════

/// The join's register is held behind a stalled data-pool connection past the
/// command timeout, so the join sees an error, and the script is then let run.
/// Because the register was issued, teardown still unregisters: peers see the
/// register's `PeerJoined` and then exactly one `PeerLeft`, the teardown marker
/// is armed, and no member entry, info hash or lease remains.
///
/// The register is parked at `before_register`, after the store-clock read,
/// so the stall catches the register alone. Which stalled connection it queued
/// on is read from `CLIENT LIST`'s `tot-net-in`, and that one is released
/// first, so the register applies before teardown's unregister (queued on the
/// next data connection) does. Every store read here goes through the
/// maintenance pool, so none of them moves the data pool's round-robin cursor.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_register_that_times_out_then_applies_is_still_torn_down() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        tv.client()
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let (session_id, _token) = create_session(&tv, &svc, "ambiguous-register", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        // Load the register and unregister scripts, so the stalled EVALSHA
        // applies when released instead of failing NOSCRIPT.
        register_remote_peer(&tv, "script-warmup", "warmup-cid").await;
        unregister_connection(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            "script-warmup",
            "warmup-cid",
            LEASE_TTL_SECS,
            &PresenceEvent::PeerLeft {
                connection_id: "warmup-cid".to_owned(),
            }
            .encode(),
        )
        .await
        .expect("warming the unregister script must succeed");

        let gate = &seams.registration.before_register;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "ambiguous-app").await;
        wait_parked(gate, "the join, before its register").await;

        let mut stalls = tv.stall_data_pool().await;
        let mut baseline = Vec::with_capacity(stalls.len());
        for stall in &stalls {
            baseline.push(
                stall
                    .net_in()
                    .await
                    .expect("the stalled connection is open"),
            );
        }
        gate.release();

        let deadline = Instant::now() + SETTLE;
        let register_conn = loop {
            let mut grew = None;
            for (i, stall) in stalls.iter().enumerate() {
                if stall.net_in().await.unwrap_or(0) > baseline[i] {
                    grew = Some(i);
                    break;
                }
            }
            if let Some(i) = grew {
                break i;
            }
            assert!(
                Instant::now() < deadline,
                "the register did not reach a stalled connection within {SETTLE:?}"
            );
            tokio::time::sleep(POLL).await;
        };
        let metrics = &svc.metrics;
        poll_until(
            "the register's command timeout",
            SETTLE,
            move || async move {
                command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
            },
        )
        .await;

        let released = stalls.remove(register_conn).release().await;
        assert!(
            matches!(&released, Ok(v) if !v.is_null()),
            "the register's connection must have stayed stalled until now; got {released:?}"
        );
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        poll_until(
            "the timed-out register applying once released",
            SETTLE,
            move || async move { !members(tv_ref, sid_ref).await.is_empty() },
        )
        .await;
        let registered = members(&tv, &session_id).await;
        let [cid] = registered.as_slice() else {
            panic!("exactly the join must be registered; members {registered:?}");
        };
        release_all(stalls).await;

        let (delivered, status) = read_to_end(&mut stream, "ambiguous register").await;
        assert!(
            delivered.is_empty(),
            "the join failed before SessionConnected; got {delivered:?}"
        );
        assert_eq!(
            status.as_ref().map(Status::code),
            Some(Code::Unavailable),
            "a register that timed out ends the join UNAVAILABLE; got {status:?}"
        );
        drain_tracker(&svc).await;

        let events = tap.drain(&tv).await;
        assert_eq!(
            count(&events, |e| is_joined(e, cid)),
            1,
            "the late register announced the join once; got {events:?}"
        );
        assert_eq!(
            count(&events, |e| is_left(e, cid)),
            1,
            "teardown's unregister must announce exactly one PeerLeft; got {events:?}"
        );
        assert!(
            events.iter().position(|e| is_joined(e, cid))
                < events.iter().position(|e| is_left(e, cid)),
            "the PeerLeft must follow the PeerJoined; got {events:?}"
        );
        assert_torn_down(&tv, &session_id, cid, LEASE_TTL_SECS, "ambiguous register").await;
        assert!(
            !is_indexed(&tv, &session_id).await,
            "the unregister left the session empty, so teardown must unindex it"
        );
        tap.close().await;
    })
    .await;
}

/// The lease TTL of the expired-register test: how long the join is held
/// before its register, and how long its teardown marker then lasts.
#[cfg(feature = "test-fault-injection")]
const EXPIRY_LEASE_TTL_SECS: u64 = 5;

/// A register that reaches the store after its deadline (the store-clock read
/// plus one lease TTL) returns `Expired`, which the join treats as an internal
/// error: the stream ends `INTERNAL`, nothing was registered or announced, and
/// teardown still ran because the register was issued — the teardown marker is
/// armed and the session is unindexed.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_register_past_its_deadline_ends_internal_and_is_torn_down() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, EXPIRY_LEASE_TTL_SECS, REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "expired-register", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        let gate = &seams.registration.before_register;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "expired-app").await;
        wait_parked(gate, "the join, before its register").await;

        // The join read the store clock before parking, so a read now is no
        // earlier; once the store clock is a lease TTL past it, the join's
        // deadline has passed.
        let origin = FredErrOrigin::client(tv.client(), POOL_LABEL_DATA);
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        let store_now = move || async move {
            store_time_ms(tv_ref.client(), origin, tv_ref.key_schema(), sid_ref)
                .await
                .expect("store_time_ms must succeed")
        };
        let past_deadline = store_now().await + EXPIRY_LEASE_TTL_SECS * 1000;
        poll_until(
            "the store clock passing the join's register deadline",
            SETTLE,
            move || async move { store_now().await > past_deadline },
        )
        .await;
        gate.release();

        let (delivered, status) = read_to_end(&mut stream, "expired register").await;
        assert!(
            delivered.is_empty(),
            "the join failed before SessionConnected; got {delivered:?}"
        );
        assert_eq!(
            status.as_ref().map(Status::code),
            Some(Code::Internal),
            "an expired register is an internal error; got {status:?}"
        );
        drain_tracker(&svc).await;

        let events = tap.drain(&tv).await;
        assert!(
            events.is_empty(),
            "an expired register publishes nothing, and teardown's unregister removed no member \
             so announces nothing; got {events:?}"
        );
        let keys = presence_keys(&tv, &session_id).await;
        let members_key = tv.key_schema().presence_members(&session_id);
        let [gone_key] = keys.as_slice() else {
            panic!("only the join's teardown marker may remain; presence keys {keys:?}");
        };
        let cid = gone_key
            .strip_prefix(&format!("{members_key}:"))
            .and_then(|rest| rest.strip_suffix(":gone"))
            .unwrap_or_else(|| {
                panic!("the one remaining key must be a teardown marker: {gone_key}")
            });
        assert_torn_down(
            &tv,
            &session_id,
            cid,
            EXPIRY_LEASE_TTL_SECS,
            "expired register",
        )
        .await;
        assert!(
            !is_indexed(&tv, &session_id).await,
            "teardown must remove the session its first index add indexed"
        );
        tap.close().await;
    })
    .await;
}

/// A join whose store-clock read (the base of its register deadline) fails —
/// held behind a stalled data pool past the command timeout — ends
/// `UNAVAILABLE` without issuing its register: teardown arms no teardown
/// marker, nothing is registered, and the session its first index add indexed
/// is unindexed again. The join is parked after that first index add while the
/// stall is put in place, so the stall catches the clock read (the subscribe
/// goes over the separate subscriber connection).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_whose_store_clock_read_fails_ends_unavailable_without_issuing_its_register() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        tv.client()
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let (session_id, _token) = create_session(&tv, &svc, "clock-read-fails", true).await;

        let gate = &seams.registration.after_index_add;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "clock-app").await;
        wait_parked(gate, "the join, after its first index add").await;
        assert!(
            is_indexed(&tv, &session_id).await,
            "precondition: the join's first add indexed the session"
        );

        let stalls = tv.stall_data_pool().await;
        gate.release();
        let metrics = &svc.metrics;
        poll_until(
            "the store-clock read timing out",
            SETTLE,
            move || async move {
                command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
            },
        )
        .await;
        release_all(stalls).await;

        let (delivered, status) = read_to_end(&mut stream, "failed clock read").await;
        assert!(
            delivered.is_empty(),
            "the join failed before SessionConnected; got {delivered:?}"
        );
        let status = status.expect("a failed clock read must end the join with an error");
        assert_eq!(
            status.code(),
            Code::Unavailable,
            "a timed-out clock read ends the join UNAVAILABLE; got {status:?}"
        );
        assert!(
            status.message().contains("TIME presence register deadline"),
            "the failure must be the register deadline's clock read; got {status:?}"
        );
        drain_tracker(&svc).await;

        assert_eq!(
            presence_keys(&tv, &session_id).await,
            Vec::<String>::new(),
            "an unissued register leaves no member set, info hash, lease or teardown marker"
        );
        assert!(
            !is_indexed(&tv, &session_id).await,
            "teardown must unindex the session the join's first index add indexed"
        );
    })
    .await;
}

/// A join whose membership snapshot (the `SMEMBERS` its replay reads, taken in
/// SR-4 step 4 after the store-clock read and before the register) fails —
/// held behind a stalled data pool past the command timeout — ends during
/// setup: `UNAVAILABLE` naming the snapshot read, nothing delivered (no
/// `SessionConnected`), and no register issued. So no member set, info hash,
/// lease or teardown marker exists afterwards, the session its first index add
/// indexed is unindexed again, and peers see neither a `joined` nor a `left`.
/// The join is parked at `before_snapshot`, after the clock read, while the
/// stall is put in place, so the stall catches the snapshot and nothing
/// earlier.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_whose_membership_snapshot_fails_ends_unavailable_during_setup_without_registering() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        tv.client()
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let (session_id, _token) = create_session(&tv, &svc, "snapshot-fails", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        let gate = &seams.replay.before_snapshot;
        gate.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "snapshot-app").await;
        wait_parked(gate, "the join, before its membership snapshot").await;
        assert!(
            is_indexed(&tv, &session_id).await,
            "precondition: the join's first index add indexed the session"
        );
        assert_eq!(
            command_timeout_row(&svc.metrics),
            command_timeouts_only_on(POOL_LABEL_DATA, 0),
            "precondition: the clock read before the gate completed without a timeout"
        );

        let stalls = tv.stall_data_pool().await;
        gate.release();
        let metrics = &svc.metrics;
        poll_until(
            "the membership snapshot timing out",
            SETTLE,
            move || async move {
                command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
            },
        )
        .await;
        release_all(stalls).await;

        let (delivered, status) = read_to_end(&mut stream, "failed membership snapshot").await;
        assert!(
            delivered.is_empty(),
            "the join failed during setup, before SessionConnected; got {delivered:?}"
        );
        let status = status.expect("a failed snapshot read must end the join with an error");
        assert_eq!(
            status.code(),
            Code::Unavailable,
            "a timed-out snapshot read ends the join UNAVAILABLE; got {status:?}"
        );
        assert!(
            status.message().contains("SMEMBERS presence"),
            "the failure must be the membership snapshot read; got {status:?}"
        );
        drain_tracker(&svc).await;
        assert_eq!(
            seams.replay.result(),
            None,
            "the join ended before its replay began"
        );

        assert_eq!(
            presence_keys(&tv, &session_id).await,
            Vec::<String>::new(),
            "an unissued register leaves no member set, info hash, lease or teardown marker"
        );
        assert!(
            !is_indexed(&tv, &session_id).await,
            "teardown must unindex the session the join's first index add indexed"
        );
        let events = tap.drain(&tv).await;
        assert!(
            events.iter().all(|e| !matches!(
                e,
                PresenceEvent::PeerJoined(_) | PresenceEvent::PeerLeft { .. }
            )),
            "a join that never registered announces neither joined nor left; got {events:?}"
        );
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Teardown at every step (PRD AC-6, end state)
// ═════════════════════════════════════════════════════════════════════════════

/// Where the join is when it is ended.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// Registered (register issued and applied), parked before
    /// `acquire_session_cancel`; no `SessionConnected` yet.
    Registration,
    /// Past `SessionConnected`, parked in the membership replay before its
    /// first page (`before_first_send`); released after the trigger fires.
    Replay,
}

/// How the replay of a join whose `trigger` fired while it was parked before
/// its first page, with a page still to read, must end: at its first raced
/// step, the page's pipeline, by the trigger's own arm (SR-4.4).
#[cfg(feature = "test-fault-injection")]
fn replay_ended_by(trigger: Trigger) -> ObservedReplayResult {
    match trigger {
        Trigger::ServiceShutdown | Trigger::ShutdownAndSessionToken => {
            ObservedReplayResult::ShuttingDown
        }
        Trigger::DeleteSession | Trigger::SessionTokenCancel | Trigger::AbsentFromStore => {
            ObservedReplayResult::SessionCancelled
        }
        Trigger::ClientDisconnect => ObservedReplayResult::Failed(Code::Cancelled),
    }
}

/// The join is parked at `stage`, `trigger` fires, the park is released, and
/// the join runs to its end. Whatever the path, the end state is the same: no
/// member entry, info hash or lease for the connection, exactly one `PeerLeft`
/// for it, the teardown marker armed for at most one lease TTL, and the session
/// indexed exactly when another connection (`with_peer`, registered from
/// "another replica") is still a member.
///
/// At [`Stage::Replay`] the replay seam also says which step ended the join.
/// With a peer, the replay has a page to read, and its first raced step sees
/// the trigger before that page's pipeline is issued ([`replay_ended_by`]).
/// Without one, the replay has no raced step and completes, and the loop ends
/// the join. The tests are named for that: `teardown_during_replay_*` with a
/// peer, `teardown_after_replay_*` without one.
#[cfg(feature = "test-fault-injection")]
async fn teardown_at_step(stage: Stage, trigger: Trigger, with_peer: bool) {
    const PEER: &str = "remote-peer";
    let what = format!("{trigger:?} during {stage:?} (peer present: {with_peer})");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
    let (session_id, token) = create_session(&tv, &svc, "teardown-at-step", true).await;
    if with_peer {
        register_remote_peer(&tv, &session_id, PEER).await;
    }
    let mut tap = Tap::open(&tv, &session_id).await;

    let gate = match stage {
        Stage::Registration => &seams.registration.before_session_cancel,
        Stage::Replay => &seams.replay.before_first_send,
    };
    gate.arm();
    let (mut stream, _tx) = open_join(&svc, &session_id, "teardown-at-step-app").await;
    let cid = match stage {
        Stage::Registration => {
            wait_parked(gate, &what).await;
            let mut joined: Vec<String> = members(&tv, &session_id)
                .await
                .into_iter()
                .filter(|m| m != PEER)
                .collect();
            assert_eq!(
                joined.len(),
                1,
                "{what}: precondition: the join is registered"
            );
            joined.remove(0)
        }
        Stage::Replay => {
            let cid = expect_session_connected(&mut stream, SETTLE).await;
            wait_parked(gate, &what).await;
            cid
        }
    };

    let stream = fire(trigger, &svc, &session_id, &token, stream).await;
    gate.release();

    if let Some(mut stream) = stream {
        let (delivered, status) = read_to_end(&mut stream, &what).await;
        let expected_code = match (stage, trigger) {
            // The join's own session re-check in `acquire_session_cancel` finds
            // the session deleted.
            (Stage::Registration, Trigger::DeleteSession) => Some(Code::NotFound),
            // The session is still in the store, so the re-check passes, and the
            // token is already cancelled with the store-absence cause: the check
            // after registration (SR-4 step 7) ends the join by that cause, and
            // so does the replay (with a peer to read) or the loop after it.
            (_, Trigger::AbsentFromStore) => Some(Code::Unavailable),
            _ => None,
        };
        assert_eq!(
            status.as_ref().map(Status::code),
            expected_code,
            "{what}: terminal status; got {status:?}"
        );
        if let (Trigger::AbsentFromStore, Some(status)) = (trigger, &status) {
            assert_eq!(
                status.message(),
                live_edit::grpc::session::session_absent_status().message(),
                "{what}: the store-absence ending, not a shutdown's"
            );
        }
        if matches!(stage, Stage::Registration) {
            assert!(
                delivered.is_empty(),
                "{what}: the join must end before SessionConnected; got {delivered:?}"
            );
        }
    }
    drain_tracker(&svc).await;

    if matches!(stage, Stage::Replay) {
        let expected = if with_peer {
            replay_ended_by(trigger)
        } else {
            ObservedReplayResult::Complete
        };
        assert_eq!(
            seams.replay.result(),
            Some(expected),
            "{what}: how the replay ended"
        );
        assert_eq!(
            seams.replay.pipelines_issued(),
            0,
            "{what}: the replay must issue no member-info pipeline: with a peer the trigger \
             stops it before its first page, and without one there is no page"
        );
        assert_eq!(
            seams.replay.peer_joined_sent_count(),
            0,
            "{what}: the replay must send no PeerJoined"
        );
    }

    let events = tap.drain(&tv).await;
    assert_eq!(
        count(&events, |e| is_left(e, &cid)),
        1,
        "{what}: peers must see exactly one PeerLeft; got {events:?}"
    );
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
    assert_eq!(
        is_indexed(&tv, &session_id).await,
        with_peer,
        "{what}: the session must stay indexed exactly while another member remains"
    );
    if with_peer {
        assert!(
            is_member(&tv, &session_id, PEER).await,
            "{what}: the other replica's member must be untouched"
        );
    }
    tap.close().await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_client_disconnect_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::ClientDisconnect,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_client_disconnect_with_peer_keeps_the_session_indexed() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::ClientDisconnect,
        true,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_delete_session_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::DeleteSession,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_delete_session_with_peer_keeps_the_session_indexed() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::DeleteSession,
        true,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_session_token_cancel_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::SessionTokenCancel,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_session_token_cancel_with_peer_keeps_the_session_indexed()
{
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::SessionTokenCancel,
        true,
    ))
    .await;
}

/// A session found missing from the live store while the join is registering:
/// the check after registration (SR-4 step 7) ends the join `UNAVAILABLE` with
/// the store-absence status, never `OK`, which Kit does not retry.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_registration_on_session_absent_from_the_store_ends_unavailable() {
    scenario(teardown_at_step(
        Stage::Registration,
        Trigger::AbsentFromStore,
        false,
    ))
    .await;
}

/// The same with the join parked at its replay's first gate, in a session with
/// no other member: that replay has no step that races the session token, so
/// it completes, and the loop's session-token arm ends the join by the same
/// cause. With a peer to read, the replay's own arm ends it; see
/// [`teardown_during_replay_on_session_absent_from_the_store_with_a_peer_ends_unavailable`].
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_after_replay_on_session_absent_from_the_store_ends_unavailable() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::AbsentFromStore,
        false,
    ))
    .await;
}

/// A session found missing from the live store while the join's replay still
/// has a peer to read: the trigger fires with the replay parked before its
/// first page, so the replay's own session-token arm ends it, before that
/// page's pipeline (`SessionCancelled`, no pipeline issued), and the join ends
/// by the cause: the retryable store-absence `UNAVAILABLE`, after teardown.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_replay_on_session_absent_from_the_store_with_a_peer_ends_unavailable() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::AbsentFromStore,
        true,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_after_replay_on_client_disconnect_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::ClientDisconnect,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_replay_on_client_disconnect_with_peer_keeps_the_session_indexed() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::ClientDisconnect,
        true,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_after_replay_on_delete_session_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::DeleteSession,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_replay_on_delete_session_with_peer_keeps_the_session_indexed() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::DeleteSession,
        true,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_after_replay_on_session_token_cancel_alone_unindexes_the_session() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::SessionTokenCancel,
        false,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_during_replay_on_session_token_cancel_with_peer_keeps_the_session_indexed() {
    scenario(teardown_at_step(
        Stage::Replay,
        Trigger::SessionTokenCancel,
        true,
    ))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The paged, cancel-aware membership replay (SR-4.4; PRD FR-3, FR-4, AC-6, AC-7)
// ═════════════════════════════════════════════════════════════════════════════
//
// The replay works from the one `SMEMBERS` snapshot the join took in SR-4
// step 4, before its register: it reads the other members' info in pages of
// up to `REPLAY_PAGE` (one pipeline each), and sends one `PeerJoined` per
// valid entry; every pipeline await and every send races shutdown, the
// session token and a client disconnect. The other members are seeded
// straight into the store (`seed_presence_members`): nothing is announced, and
// no second join's replay adds to the replay seam's observations, which every
// replay on the service accumulates.
//
// A cancel is landed at each place the replay can wait:
//
//   * between pages: parked at `between_pages`, after page one's sends, with
//     the client reading;
//   * on a send: the client reads `SessionConnected` and stops, so the replay
//     fills the response channel's delivery slots and waits on the next send;
//   * on a pipeline: every data-pool connection is held by a `ConnStall`, so
//     page one's pipeline cannot return. The replay's result is read while
//     the stall still holds, which an unraced pipeline await could not
//     produce, and the stall is released at once so teardown's unregister,
//     queued behind it, runs well inside the command timeout.
//
// With no trigger, the same held pipeline times out: the replay's store-error
// exit.

// `RESPONSE_DELIVERY_SLOTS`, `ONE_PAGE_OVERFILLING_THE_CHANNEL`,
// `seeded_cids` and `seed_presence_members` live in `super::helpers`, shared
// with the mid-replay logging tests.

/// Other members for a replay of two pages, the second partial.
#[cfg(feature = "test-fault-injection")]
const TWO_PAGES_OF_OTHERS: usize = REPLAY_PAGE + REPLAY_PAGE / 2;

/// The connection ids of the `PeerJoined` messages in `delivered`, in order.
#[cfg(feature = "test-fault-injection")]
fn peer_joined_ids(delivered: &[JoinSessionResponse]) -> Vec<String> {
    delivered
        .iter()
        .filter_map(|msg| match &msg.response_type {
            Some(join_session_response::ResponseType::PeerJoined(p)) => {
                Some(p.connection_id.clone())
            }
            _ => None,
        })
        .collect()
}

/// Read the stream's next `n` messages, every one a `PeerJoined`, under one
/// [`SETTLE`] deadline, and return them in order.
#[cfg(feature = "test-fault-injection")]
async fn read_peer_joined(
    stream: &mut JoinResponseStream,
    n: usize,
    what: &str,
) -> Vec<live_edit::grpc::proto::PeerJoined> {
    tokio::time::timeout(SETTLE, async {
        let mut peers = Vec::with_capacity(n);
        while peers.len() < n {
            let msg = stream
                .next()
                .await
                .unwrap_or_else(|| {
                    panic!(
                        "{what}: the join stream ended after {} of {n} PeerJoined",
                        peers.len()
                    )
                })
                .unwrap_or_else(|s| panic!("{what}: the join stream failed: {s:?}"));
            match msg.response_type {
                Some(join_session_response::ResponseType::PeerJoined(p)) => peers.push(p),
                other => panic!(
                    "{what}: message {} of {n} must be a PeerJoined; got {other:?}",
                    peers.len() + 1
                ),
            }
        }
        peers
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: {n} PeerJoined did not arrive within {SETTLE:?}"))
}

/// Wait for the replay to end, and return how it ended.
#[cfg(feature = "test-fault-injection")]
async fn replay_result(seam: &PresenceReplaySeam, what: &str) -> ObservedReplayResult {
    tokio::time::timeout(SETTLE, seam.wait_for_result())
        .await
        .unwrap_or_else(|_| panic!("{what}: the replay did not end within {SETTLE:?}"))
}

/// `live_edit_join_latency_seconds_count{endpoint="JoinSession"}` on `svc`'s
/// metrics; absent, so 0, until the first observation.
#[cfg(feature = "test-fault-injection")]
fn join_latency_count(svc: &LiveEditService) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(&svc.metrics),
        "live_edit_join_latency_seconds_count",
        &[
            ("endpoint", "JoinSession"),
            ("service", svc.service_label.as_str()),
        ],
    )
    .unwrap_or(0.0) as u64
}

// ── Ordering and malformed info ──────────────────────────────────────────────

/// A session of 250 members, more than two pages, 249 of them seeded: the
/// joiner's stream carries `SessionConnected`, then exactly the 249 others as
/// `PeerJoined`, each once, then the `PeerJoined` of a member that joined
/// after the replay's snapshot, as a live event.
///
/// The live member registers while the replay is parked between its first and
/// second pages, and its event is known to be queued on the join's presence
/// receiver before the replay resumes: the test waits for a local observer on
/// the same router entry to receive it before releasing the replay. So a
/// replay that let a live event through, or re-read the member set for a later
/// page, would interleave or duplicate it, and every message up to the live
/// `PeerJoined` must be exactly the replay.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_replay_of_250_members_sends_each_other_member_once_before_a_live_join() {
    const MEMBERS: usize = 250;
    const LIVE: &str = "live-peer";
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "replay-order", true).await;
        let others = seeded_cids(MEMBERS - 1);
        seed_presence_members(&tv, &session_id, &others).await;

        let gate = &seams.replay.between_pages;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "replay-order-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let mut replayed =
            read_peer_joined(&mut stream, REPLAY_PAGE, "the replay's first page").await;
        wait_parked(gate, "the replay, between its first and second pages").await;

        // The join's own receiver hangs off this router entry, so once the
        // observer holds the live event, the join's receiver holds it too.
        let (mut observer, _) = svc.presence_router.subscribe(&session_id);
        register_remote_peer(&tv, &session_id, LIVE).await;
        tokio::time::timeout(SETTLE, async {
            loop {
                match observer.recv().await {
                    Ok(event) if is_joined(&event, LIVE) => return,
                    Ok(_) => {}
                    Err(e) => panic!("the observer's receiver failed: {e:?}"),
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("the live PeerJoined did not reach the session's router within {SETTLE:?}")
        });
        gate.release();

        // Everything up to the live join: it must be exactly the replay.
        tokio::time::timeout(SETTLE, async {
            loop {
                let mut next = read_peer_joined(&mut stream, 1, "the rest of the replay").await;
                let peer = next.remove(0);
                if peer.connection_id == LIVE {
                    return;
                }
                replayed.push(peer);
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the live PeerJoined did not arrive within {SETTLE:?}"));
        assert_eq!(
            replay_result(&seams.replay, "the 250-member replay").await,
            ObservedReplayResult::Complete
        );

        let mut replayed_ids: Vec<String> =
            replayed.iter().map(|p| p.connection_id.clone()).collect();
        replayed_ids.sort();
        assert_eq!(
            replayed_ids, others,
            "before the live PeerJoined, the stream must carry every other member exactly \
             once, and neither the joiner ({cid}) nor the member that joined after the \
             snapshot"
        );
        // Bounded reads (PRD Goal 6): ⌈249 / REPLAY_PAGE⌉ pipelines, the last
        // partial, one at a time. The live member joined after the snapshot,
        // so it adds no page.
        assert_eq!(
            seams.replay.page_sizes(),
            vec![REPLAY_PAGE, REPLAY_PAGE, MEMBERS - 1 - 2 * REPLAY_PAGE],
            "one pipeline per page of at most {REPLAY_PAGE} members' info"
        );
        assert_eq!(
            seams.replay.max_pipelines_in_flight(),
            1,
            "at most one pipeline in flight"
        );

        assert!(
            !svc.presence_router.unsubscribe(&session_id),
            "the join is still subscribed, so the observer is not the last local receiver"
        );
        drop(tx);
        let rest = read_to_ok_end(&mut stream, "client close after the replay").await;
        assert_eq!(
            peer_joined_ids(&rest),
            Vec::<String>::new(),
            "nothing may be replayed or relayed twice"
        );
        drain_tracker(&svc).await;
        feed.cancel();
    })
    .await;
}

/// Members whose info hash is absent (empty), or lacks `user_name`, are skipped
/// and the replay goes on (PRD AC-7). The two malformed members lead the first
/// page, so valid members follow them in that page and fill a second.
///
/// Placement rests on Valkey keeping a small set (here 103 members, under the
/// default `set-max-listpack-entries` of 128) in insertion order; the test
/// reads the member set while the replay is parked after its own snapshot and
/// checks the layout before releasing it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_replay_skips_malformed_member_info_and_replays_the_rest() {
    const NO_INFO: &str = "malformed-no-info";
    const NO_USER_NAME: &str = "malformed-no-user-name";
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "replay-malformed", true).await;
        let ks = tv.key_schema();
        let added: i64 = store_reader(&tv)
            .sadd(
                ks.presence_members(&session_id),
                vec![NO_INFO, NO_USER_NAME],
            )
            .await
            .expect("seeding the malformed members must succeed");
        assert_eq!(added, 2);
        let _: i64 = store_reader(&tv)
            .hset(
                ks.presence_user(&session_id, NO_USER_NAME),
                ("user_application", "no-user-name-app"),
            )
            .await
            .expect("seeding the info hash without user_name must succeed");
        let valid = seeded_cids(REPLAY_PAGE);
        seed_presence_members(&tv, &session_id, &valid).await;

        let gate = &seams.replay.before_first_send;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "replay-malformed-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(gate, "the replay, after its snapshot").await;

        let snapshot: Vec<String> = members(&tv, &session_id)
            .await
            .into_iter()
            .filter(|m| *m != cid)
            .collect();
        let pages: Vec<&[String]> = snapshot.chunks(REPLAY_PAGE).collect();
        let last_malformed = pages[0]
            .iter()
            .rposition(|m| m == NO_INFO || m == NO_USER_NAME);
        assert!(
            pages.len() == 2
                && pages[0].iter().any(|m| m == NO_INFO)
                && pages[0].iter().any(|m| m == NO_USER_NAME)
                && last_malformed.is_some_and(|i| i + 1 < pages[0].len()),
            "precondition: both malformed members in the first page with valid members after \
             them, and a second page; Valkey did not keep the set in insertion order: \
             {snapshot:?}"
        );
        gate.release();

        let replayed = read_peer_joined(&mut stream, valid.len(), "the valid members").await;
        assert_eq!(
            replay_result(&seams.replay, "the replay with malformed members").await,
            ObservedReplayResult::Complete,
            "a malformed entry must not end the replay"
        );
        assert_eq!(
            seams.replay.page_sizes(),
            vec![REPLAY_PAGE, 2],
            "both pages are read, malformed entries included"
        );
        assert_eq!(
            seams.replay.peer_joined_sent_count(),
            valid.len(),
            "only the valid members are sent"
        );
        let mut replayed_ids: Vec<String> =
            replayed.iter().map(|p| p.connection_id.clone()).collect();
        replayed_ids.sort();
        assert_eq!(
            replayed_ids, valid,
            "every valid member, and no malformed one"
        );
        for peer in &replayed {
            assert_eq!(peer.user_name, seeded_user_name(&peer.connection_id));
            assert_eq!(peer.user_application, SEEDED_APP);
        }

        drop(tx);
        let rest = read_to_ok_end(&mut stream, "client close after the replay").await;
        assert_eq!(
            peer_joined_ids(&rest),
            Vec::<String>::new(),
            "no malformed member may be sent after the valid ones"
        );
        drain_tracker(&svc).await;
    })
    .await;
}

// ── Bounded reads ────────────────────────────────────────────────────────────

/// A complete replay for a session of `members` members, the joiner included,
/// issues one pipeline per page — ⌈(members − 1) / `REPLAY_PAGE`⌉ — of exactly
/// `expected_page_sizes`, never more than one in flight, and sends every other
/// member once.
#[cfg(feature = "test-fault-injection")]
async fn replay_reads_are_paged(members: usize, expected_page_sizes: &[usize]) {
    let what = format!("a replay for {members} members");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
    let (session_id, _token) = create_session(&tv, &svc, "replay-bounds", true).await;
    let others = seeded_cids(members - 1);
    seed_presence_members(&tv, &session_id, &others).await;

    let (mut stream, tx) = open_join(&svc, &session_id, "replay-bounds-app").await;
    expect_session_connected(&mut stream, SETTLE).await;
    let replayed = read_peer_joined(&mut stream, others.len(), &what).await;
    assert_eq!(
        replay_result(&seams.replay, &what).await,
        ObservedReplayResult::Complete
    );

    let pipelines = (members - 1).div_ceil(REPLAY_PAGE);
    assert_eq!(
        expected_page_sizes.len(),
        pipelines,
        "{what}: the test's expectation"
    );
    assert_eq!(
        seams.replay.pipelines_issued(),
        pipelines,
        "{what}: one pipeline per page, ⌈(n − 1) / {REPLAY_PAGE}⌉"
    );
    assert_eq!(
        seams.replay.page_sizes(),
        expected_page_sizes,
        "{what}: each pipeline reads at most {REPLAY_PAGE} members' info"
    );
    assert_eq!(
        seams.replay.max_pipelines_in_flight(),
        pipelines.min(1),
        "{what}: at most one pipeline in flight"
    );
    let mut replayed_ids: Vec<String> = replayed.iter().map(|p| p.connection_id.clone()).collect();
    replayed_ids.sort();
    assert_eq!(replayed_ids, others, "{what}: every other member once");

    drop(tx);
    read_to_ok_end(&mut stream, &what).await;
    drain_tracker(&svc).await;
    // The positive case of every cancel test's "no join latency" check, so a
    // drifted series name, which reads as 0, cannot make those vacuous.
    assert_eq!(
        join_latency_count(&svc),
        1,
        "{what}: a completed replay observes one join latency"
    );
}

// A replay of more than two pages, the last partial, is measured by the
// ordering test (`join_session_replay_of_250_members_…`); these are the
// boundaries.

/// The boundary: exactly one page of other members is one pipeline.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_replay_of_exactly_one_page_of_others_reads_one_pipeline() {
    scenario(replay_reads_are_paged(REPLAY_PAGE + 1, &[REPLAY_PAGE])).await;
}

/// A joiner alone in its session reads no member info at all.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_replay_of_a_lone_joiner_reads_no_pipeline() {
    scenario(replay_reads_are_paged(1, &[])).await;
}

// ── A cancel at every point the replay waits (PRD FR-4, AC-6) ────────────────

/// Where the replay is waiting when its join is ended.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum ReplayWait {
    /// Parked at `between_pages`, after page one's `REPLAY_PAGE` sends, before
    /// page two's pipeline ([`TWO_PAGES_OF_OTHERS`] other members).
    BetweenPages,
    /// On a send: the response channel is full, the client has stopped
    /// reading, and page one still has members to send
    /// ([`ONE_PAGE_OVERFILLING_THE_CHANNEL`] other members).
    FullChannel,
    /// On page one's pipeline, held behind a stalled data pool.
    PipelineInFlight,
}

#[cfg(feature = "test-fault-injection")]
impl ReplayWait {
    fn other_members(self) -> usize {
        match self {
            ReplayWait::BetweenPages => TWO_PAGES_OF_OTHERS,
            ReplayWait::FullChannel | ReplayWait::PipelineInFlight => {
                ONE_PAGE_OVERFILLING_THE_CHANNEL
            }
        }
    }

    /// Replay `PeerJoined` sends completed by the time the replay waits.
    fn sent_before_the_wait(self) -> usize {
        match self {
            ReplayWait::BetweenPages => REPLAY_PAGE,
            ReplayWait::FullChannel => RESPONSE_DELIVERY_SLOTS,
            ReplayWait::PipelineInFlight => 0,
        }
    }

    /// Replay `PeerJoined` messages still unread on the stream at the cancel:
    /// the client reads page one before parking between pages, and reads
    /// nothing after `SessionConnected` in the full-channel case.
    fn unread_at_the_cancel(self) -> usize {
        match self {
            ReplayWait::BetweenPages | ReplayWait::PipelineInFlight => 0,
            ReplayWait::FullChannel => RESPONSE_DELIVERY_SLOTS,
        }
    }
}

/// The join's replay is waiting at `wait` when `trigger` fires. The replay
/// stops there, by the trigger's own arm ([`replay_ended_by`]), without
/// issuing another pipeline or sending another `PeerJoined`; the stream ends as
/// the trigger's cause sets ([`Trigger::ending`]); no join latency is
/// observed; and the connection's full teardown state holds (PRD FR-12): no
/// member entry, info hash or lease, the teardown marker armed for at most one
/// lease TTL, exactly one `PeerLeft` on an external subscriber, and no join
/// task or keeper left running.
#[cfg(feature = "test-fault-injection")]
async fn replay_cancelled_while_waiting(wait: ReplayWait, trigger: Trigger) {
    let what = format!("{trigger:?} while the replay waits {wait:?}");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
    let replay = &seams.replay;
    let (session_id, token) = create_session(&tv, &svc, "replay-cancel", true).await;
    seed_presence_members(&tv, &session_id, &seeded_cids(wait.other_members())).await;
    let mut tap = Tap::open(&tv, &session_id).await;

    if matches!(wait, ReplayWait::BetweenPages) {
        replay.between_pages.arm();
    }
    if matches!(wait, ReplayWait::PipelineInFlight) {
        replay.before_first_send.arm();
    }
    let (mut stream, _tx) = open_join(&svc, &session_id, "replay-cancel-app").await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;
    let mut stalls = None;
    match wait {
        ReplayWait::BetweenPages => {
            read_peer_joined(&mut stream, REPLAY_PAGE, &what).await;
            wait_parked(&replay.between_pages, &what).await;
        }
        // The client reads nothing more.
        ReplayWait::FullChannel => {
            tokio::time::timeout(
                SETTLE,
                replay.wait_for_peer_joined_sent(RESPONSE_DELIVERY_SLOTS),
            )
            .await
            .unwrap_or_else(|_| panic!("{what}: the replay did not fill the channel"));
        }
        // Parked after the snapshot, so the stall catches page one's pipeline
        // and nothing before it.
        ReplayWait::PipelineInFlight => {
            wait_parked(&replay.before_first_send, &what).await;
            stalls = Some(tv.stall_data_pool().await);
            replay.before_first_send.release();
            poll_until(
                &format!("{what}: page one's pipeline being issued"),
                SETTLE,
                move || async move { replay.pipelines_issued() == 1 },
            )
            .await;
        }
    }
    assert_eq!(
        replay.pipelines_issued(),
        1,
        "{what}: precondition: the replay is on page one"
    );
    let sent_before = wait.sent_before_the_wait();
    assert_eq!(
        replay.peer_joined_sent_count(),
        sent_before,
        "{what}: precondition: the replay's sends so far"
    );
    assert_eq!(
        replay.result(),
        None,
        "{what}: precondition: still replaying"
    );

    let stream = fire(trigger, &svc, &session_id, &token, stream).await;
    if matches!(wait, ReplayWait::BetweenPages) {
        replay.between_pages.release();
    }
    let result = replay_result(replay, &what).await;
    if let Some(stalls) = stalls {
        let mut still_blocked = true;
        for stall in &stalls {
            still_blocked &= stall.is_blocked().await;
        }
        release_all(stalls).await;
        assert!(
            still_blocked,
            "{what}: the replay must end while its pipeline is still held, not when it \
             returns; the stall had already let go when the result was read"
        );
    }
    assert_eq!(
        result,
        replay_ended_by(trigger),
        "{what}: how the replay ended"
    );
    assert_eq!(
        replay.pipelines_issued(),
        1,
        "{what}: no pipeline may be issued after the cancel"
    );
    assert_eq!(
        replay.peer_joined_sent_count(),
        sent_before,
        "{what}: no PeerJoined may be sent after the cancel"
    );

    if let Some(mut stream) = stream {
        let (delivered, status) = read_to_end(&mut stream, &what).await;
        assert_eq!(
            peer_joined_ids(&delivered).len(),
            wait.unread_at_the_cancel(),
            "{what}: only the PeerJoined enqueued before the cancel may reach the client"
        );
        assert_ending(trigger.ending(), status, &what);
    }
    drain_tracker(&svc).await;
    assert_eq!(
        join_latency_count(&svc),
        0,
        "{what}: a join whose replay was cancelled observes no join latency"
    );

    let events = tap.drain(&tv).await;
    assert_eq!(
        count(&events, |e| is_left(e, &cid)),
        1,
        "{what}: peers must see exactly one PeerLeft; got {events:?}"
    );
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
    tap.close().await;
}

/// `DeleteSession` while the replay is parked between pages ends the stream
/// `OK` after teardown, without reading page two.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_between_pages_on_delete_session_ends_ok_and_tears_down() {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::BetweenPages,
        Trigger::DeleteSession,
    ))
    .await;
}

/// A service shutdown while the replay is parked between pages ends the
/// stream `UNAVAILABLE "server shutting down"` after teardown, without reading
/// page two.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_between_pages_on_service_shutdown_ends_unavailable_and_tears_down() {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::BetweenPages,
        Trigger::ServiceShutdown,
    ))
    .await;
}

/// A client that leaves while the replay is parked between pages ends it
/// `CANCELLED`, which the gone client cannot see, so the replay seam's result
/// is what is asserted; teardown runs without page two being read.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_between_pages_on_client_disconnect_ends_cancelled_and_tears_down() {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::BetweenPages,
        Trigger::ClientDisconnect,
    ))
    .await;
}

/// Shutdown and a session cancel both land while the replay is parked between
/// pages. Shutdown is checked first, so the replay ends `ShuttingDown` and the
/// stream `UNAVAILABLE`, not the `OK` a deletion alone gets.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_between_pages_on_shutdown_and_session_cancel_ends_as_a_shutdown() {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::BetweenPages,
        Trigger::ShutdownAndSessionToken,
    ))
    .await;
}

/// `DeleteSession` while the replay waits on a send to a client that stopped
/// reading ends the stream `OK` after teardown, however long the client takes.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_blocked_on_a_full_channel_on_delete_session_ends_ok_and_tears_down() {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::FullChannel,
        Trigger::DeleteSession,
    ))
    .await;
}

/// A service shutdown while the replay waits on a send to a client that
/// stopped reading ends the stream `UNAVAILABLE "server shutting down"` after
/// teardown.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_blocked_on_a_full_channel_on_service_shutdown_ends_unavailable_and_tears_down()
{
    scenario(replay_cancelled_while_waiting(
        ReplayWait::FullChannel,
        Trigger::ServiceShutdown,
    ))
    .await;
}

/// A client that leaves while the replay waits on a send ends it `CANCELLED`,
/// with teardown.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_blocked_on_a_full_channel_on_client_disconnect_ends_cancelled_and_tears_down()
{
    scenario(replay_cancelled_while_waiting(
        ReplayWait::FullChannel,
        Trigger::ClientDisconnect,
    ))
    .await;
}

/// The session token cancelled while page one's pipeline is held ends the
/// replay `SessionCancelled` before the pipeline returns, and the stream `OK`
/// after teardown. `DeleteSession` itself cannot be the trigger here: its own
/// data-pool reads would queue behind the stall, so it would never reach the
/// token; it is covered between pages and on a full channel.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_with_a_pipeline_in_flight_on_session_token_cancel_ends_ok_without_awaiting_it()
{
    scenario(replay_cancelled_while_waiting(
        ReplayWait::PipelineInFlight,
        Trigger::SessionTokenCancel,
    ))
    .await;
}

/// A service shutdown while page one's pipeline is held ends the replay
/// `ShuttingDown` before the pipeline returns, and the stream `UNAVAILABLE`.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_with_a_pipeline_in_flight_on_service_shutdown_ends_unavailable_without_awaiting_it(
) {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::PipelineInFlight,
        Trigger::ServiceShutdown,
    ))
    .await;
}

/// A client that leaves while page one's pipeline is held ends the replay
/// `CANCELLED` before the pipeline returns.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_with_a_pipeline_in_flight_on_client_disconnect_ends_cancelled_without_awaiting_it(
) {
    scenario(replay_cancelled_while_waiting(
        ReplayWait::PipelineInFlight,
        Trigger::ClientDisconnect,
    ))
    .await;
}

// ── A store error ends the replay (SR-4.4 exit paths) ────────────────────────

/// With no trigger, page one's pipeline is held behind a stalled data pool
/// past the command timeout: the replay's store-error exit. The replay ends
/// with the mapped status, `UNAVAILABLE "HGETALL presence user info replay
/// page failed (Timeout Error): …"` (fred's name for the error kind), which the
/// stream ends with after teardown; the
/// timeout is counted exactly once on the data pool (the read itself counts
/// nothing, its caller's mapping counts it); no join latency is observed; and
/// the connection's full teardown state holds.
///
/// The stall is released as soon as the replay's result is recorded, so
/// teardown's unregister, queued behind it on the data pool, runs well inside
/// its own command timeout.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_replay_whose_pipeline_times_out_ends_unavailable_and_counts_one_timeout() {
    scenario(async {
        let what = "a replay whose page pipeline timed out";
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        tv.client()
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let replay = &seams.replay;
        let (session_id, _token) = create_session(&tv, &svc, "replay-store-error", true).await;
        seed_presence_members(
            &tv,
            &session_id,
            &seeded_cids(ONE_PAGE_OVERFILLING_THE_CHANNEL),
        )
        .await;
        let mut tap = Tap::open(&tv, &session_id).await;

        replay.before_first_send.arm();
        let (mut stream, _tx) = open_join(&svc, &session_id, "replay-store-error-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(&replay.before_first_send, what).await;
        let stalls = tv.stall_data_pool().await;
        replay.before_first_send.release();
        let result = replay_result(replay, what).await;
        release_all(stalls).await;

        assert_eq!(
            result,
            ObservedReplayResult::Failed(Code::Unavailable),
            "{what}: how the replay ended"
        );
        assert_eq!(replay.pipelines_issued(), 1, "{what}: page one's pipeline");
        assert_eq!(
            replay.peer_joined_sent_count(),
            0,
            "{what}: nothing is sent from a page that failed"
        );
        let (delivered, status) = read_to_end(&mut stream, what).await;
        assert_eq!(
            peer_joined_ids(&delivered),
            Vec::<String>::new(),
            "{what}: no PeerJoined"
        );
        let status = status.expect("a replay store error must end the join with an error");
        assert_eq!(status.code(), Code::Unavailable, "{what}: got {status:?}");
        assert!(
            status
                .message()
                .starts_with("HGETALL presence user info replay page failed (Timeout Error)"),
            "{what}: the status must name the replay page read that timed out; got {status:?}"
        );
        drain_tracker(&svc).await;
        assert_eq!(
            command_timeout_row(&svc.metrics),
            command_timeouts_only_on(POOL_LABEL_DATA, 1),
            "{what}: the timeout must be counted exactly once, on the data pool"
        );
        assert_eq!(
            join_latency_count(&svc),
            0,
            "{what}: a failed replay observes no join latency"
        );

        let events = tap.drain(&tv).await;
        assert_eq!(
            count(&events, |e| is_left(e, &cid)),
            1,
            "{what}: peers must see exactly one PeerLeft; got {events:?}"
        );
        assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, what).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Early exits leave other local connections subscribed
// ═════════════════════════════════════════════════════════════════════════════

/// A join whose first index `SADD` fails — held behind a stalled data pool past
/// the command timeout — exits before its router subscribe and without issuing
/// its register. Teardown must not touch the router, so another local
/// connection in the same session keeps receiving presence events; it runs the
/// index-removal helper, which leaves the session indexed because that
/// connection is still a member. The router's count stays right: when that
/// connection then leaves, the session's router entry goes.
///
/// A step of the other connection's keeper during the stall, which a slow run
/// allows, changes none of this: it is issued after the failing `SADD`, so its
/// refresh times out, if at all, only after that `SADD`'s timeout has been
/// counted, and a renewal neither removes a member nor unindexes the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_whose_first_index_add_fails_leaves_the_other_local_connection_subscribed() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        tv.client()
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "early-exit-index", true).await;
        let (mut other, _other_tx) = open_join(&svc, &session_id, "other-app").await;
        let other_cid = expect_session_connected(&mut other, SETTLE).await;

        let stalls = tv.stall_data_pool().await;
        let (mut failing, _failing_tx) = open_join(&svc, &session_id, "failing-app").await;
        let metrics = &svc.metrics;
        poll_until(
            "the failing join's index SADD timing out",
            SETTLE,
            move || async move {
                command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
            },
        )
        .await;
        release_all(stalls).await;

        let (delivered, status) = read_to_end(&mut failing, "index add failure").await;
        assert!(
            delivered.is_empty(),
            "the failing join never connected; got {delivered:?}"
        );
        assert_eq!(
            status.as_ref().map(Status::code),
            Some(Code::Unavailable),
            "a timed-out index add ends the join UNAVAILABLE; got {status:?}"
        );
        assert_eq!(
            members(&tv, &session_id).await,
            vec![other_cid.clone()],
            "the failing join never issued its register"
        );
        assert!(
            is_indexed(&tv, &session_id).await,
            "the other connection is still a member, so the session stays indexed"
        );

        publish_probe(&tv, &session_id, "probe-after-index-failure").await;
        expect_peer_joined(
            &mut other,
            "probe-after-index-failure",
            "the other local connection after the failed join",
        )
        .await;

        drop(other);
        drain_tracker(&svc).await;
        assert!(
            !svc.presence_router.has_session(&session_id),
            "the other connection's leave must be the session's last router unsubscribe"
        );
        feed.cancel();
    })
    .await;
}

// A join sends its own Valkey `SUBSCRIBE` only when `join_channel` hands it
// `ChannelJoin::Subscribe`, which happens only while no local connection is
// subscribed to the session's channel: it may be the session's first local
// join, the first join after a failed round (while the failed joins still hold
// their counts), or a join that arrived during the last leaver's `UNSUBSCRIBE`.
// A join arriving while the channel is subscribed sends no `SUBSCRIBE`, and one
// arriving while a `SUBSCRIBE` is in flight waits on it (next section). So
// "another local connection keeps receiving events after a join's SUBSCRIBE
// fails" has two readings, each tested: a connection that was already
// subscribed is unaffected by a later join's early exit (the index-SADD test
// above, whose failing join exits before it even joins the router), and joins
// that arrived during the failing `SUBSCRIBE` fail with it, `UNAVAILABLE`,
// leaving the channel unsubscribed so the next join subscribes again, and their
// teardowns then remove the router entry (the failed-SUBSCRIBE scenario below).

// ═════════════════════════════════════════════════════════════════════════════
// Concurrent joins and the channel subscription (operator directive, Task 4)
// ═════════════════════════════════════════════════════════════════════════════

/// How long a test watches joins that must stay held behind an in-flight
/// `SUBSCRIBE` or `UNSUBSCRIBE`. A join that was not held reaches registration
/// in milliseconds, so it would show well within this window.
#[cfg(feature = "test-fault-injection")]
const HELD_WATCH: Duration = Duration::from_secs(1);

/// Watch for [`HELD_WATCH`] that no join registers in the session and none of
/// `streams` delivers anything: every join is still held at its channel
/// subscription.
#[cfg(feature = "test-fault-injection")]
async fn assert_joins_held(
    tv: &TestValkey,
    session_id: &str,
    streams: &mut [&mut JoinResponseStream],
    what: &str,
) {
    use futures::FutureExt;

    let watch_until = Instant::now() + HELD_WATCH;
    while Instant::now() < watch_until {
        let registered = members(tv, session_id).await;
        assert!(
            registered.is_empty(),
            "{what}: no join may register while the channel subscription is unconfirmed; \
             members {registered:?}"
        );
        for (i, stream) in streams.iter_mut().enumerate() {
            if let Some(item) = stream.next().now_or_never() {
                panic!(
                    "{what}: join {i} must stay held at its channel subscription, but its \
                     stream delivered {item:?}"
                );
            }
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Start `n` joins into the session and wait until each has joined the local
/// router (`connection_count` reaches `joined_before + n`).
#[cfg(feature = "test-fault-injection")]
async fn start_joins(
    svc: &LiveEditService,
    session_id: &str,
    n: usize,
    joined_before: usize,
) -> Vec<(
    JoinResponseStream,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
)> {
    let mut joins = Vec::with_capacity(n);
    for i in 0..n {
        joins.push(open_join(svc, session_id, &format!("concurrent-app-{i}")).await);
    }
    let router = &svc.presence_router;
    poll_until(
        "the concurrent joins joining the local router",
        SETTLE,
        move || async move { router.connection_count() == joined_before + n },
    )
    .await;
    joins
}

/// **Operator directive, success case.** Joins B and C arrive while the first
/// joiner A's `SUBSCRIBE` is in flight (A is parked just before sending it).
/// None of them registers or is sent anything until that `SUBSCRIBE` is
/// confirmed. An announcement published right after it — while the first join
/// past it is parked before its register, so before that join has sent
/// `SessionConnected`; the other two may already have, since the register gate
/// parks only its first arrival — reaches all three.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_joins_during_an_in_flight_subscribe_all_receive_an_announcement_sent_right_after_it(
) {
    const PROBE: &str = "announced-right-after-subscribe";
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "concurrent-success", true).await;

        let subscribe_gate = &seams.channel.before_subscribe;
        subscribe_gate.arm();
        let (mut a, _a_tx) = open_join(&svc, &session_id, "first-app").await;
        wait_parked(subscribe_gate, "A, just before its SUBSCRIBE").await;
        let mut later = start_joins(&svc, &session_id, 2, 1).await;
        let (mut b, _b_tx) = later.remove(0);
        let (mut c, _c_tx) = later.remove(0);
        assert_joins_held(
            &tv,
            &session_id,
            &mut [&mut a, &mut b, &mut c],
            "while A's SUBSCRIBE is in flight",
        )
        .await;

        let register_gate = &seams.registration.before_register;
        register_gate.arm();
        subscribe_gate.release();
        wait_parked(register_gate, "the first join past the confirmed SUBSCRIBE").await;
        publish_probe(&tv, &session_id, PROBE).await;
        register_gate.release();

        for (name, stream) in [("A", &mut a), ("B", &mut b), ("C", &mut c)] {
            expect_session_connected(stream, SETTLE).await;
            expect_peer_joined(stream, PROBE, &format!("join {name}")).await;
        }

        drop((a, b, c));
        drain_tracker(&svc).await;
        feed.cancel();
    })
    .await;
}

/// **Operator directive, failure case** (and the concurrent half of the Epic's
/// "early exit leaves peers subscribed"). Joins B and C arrive while A's
/// `SUBSCRIBE` is in flight, and that `SUBSCRIBE` fails. All three end
/// `UNAVAILABLE` with nothing delivered and nothing registered; each releases
/// its router count, so the session's router entry goes. The next join D then
/// subscribes for real and receives presence events.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_joins_during_a_failing_subscribe_all_end_unavailable_and_the_next_join_subscribes(
) {
    const PROBE: &str = "announced-after-the-failed-round";
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "concurrent-failure", true).await;

        let subscribe_gate = &seams.channel.before_subscribe;
        subscribe_gate.arm();
        let (a, _a_tx) = open_join(&svc, &session_id, "first-app").await;
        wait_parked(subscribe_gate, "A, just before its SUBSCRIBE").await;
        let mut joins = vec![a];
        joins.extend(
            start_joins(&svc, &session_id, 2, 1)
                .await
                .into_iter()
                .map(|(stream, _tx)| stream),
        );
        {
            let mut held: Vec<&mut JoinResponseStream> = joins.iter_mut().collect();
            assert_joins_held(
                &tv,
                &session_id,
                &mut held,
                "while A's SUBSCRIBE is in flight",
            )
            .await;
        }

        seams.channel.fail_next_subscribe();
        subscribe_gate.release();
        for (name, stream) in ["A", "B", "C"].into_iter().zip(joins.iter_mut()) {
            let what = format!("join {name} of the failed round");
            let (delivered, status) = read_to_end(stream, &what).await;
            assert!(
                delivered.is_empty(),
                "{what}: nothing may be delivered; got {delivered:?}"
            );
            assert_eq!(
                status.as_ref().map(Status::code),
                Some(Code::Unavailable),
                "{what}: must end UNAVAILABLE; got {status:?}"
            );
        }
        let router = &svc.presence_router;
        let sid = session_id.as_str();
        poll_until(
            "the failed round's router entry going",
            SETTLE,
            move || async move { !router.has_session(sid) },
        )
        .await;
        assert_eq!(
            router.connection_count(),
            0,
            "every failed join released its count"
        );
        assert!(
            members(&tv, &session_id).await.is_empty(),
            "no join of the failed round may have registered"
        );
        assert!(
            !is_indexed(&tv, &session_id).await,
            "with no member, the failed joins' teardown must unindex the session"
        );

        let (mut d, _d_tx) = open_join(&svc, &session_id, "next-app").await;
        expect_session_connected(&mut d, SETTLE).await;
        publish_probe(&tv, &session_id, PROBE).await;
        expect_peer_joined(&mut d, PROBE, "the join after the failed round").await;

        drop((joins, d));
        drain_tracker(&svc).await;
        assert!(
            !router.has_session(&session_id),
            "the next join's leave must be the session's last"
        );
        feed.cancel();
    })
    .await;
}

/// **Operator directive, last leaver racing a new joiner.** The session's last
/// local connection A leaves and is parked just before its `UNSUBSCRIBE`; join
/// B arrives meanwhile. B does not send its `SUBSCRIBE` until A's
/// `UNSUBSCRIBE` has completed (it cannot reach the point just before
/// `SUBSCRIBE` while A is held), so the channel ends subscribed: B receives
/// `SessionConnected` and a presence event published afterwards.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn last_leaver_racing_a_new_joiner_leaves_the_new_joiner_subscribed() {
    const PROBE: &str = "announced-after-the-race";
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "leaver-vs-joiner", true).await;

        let (mut a, _a_tx) = open_join(&svc, &session_id, "leaver-app").await;
        expect_session_connected(&mut a, SETTLE).await;

        let unsubscribe_gate = &seams.channel.before_unsubscribe;
        let subscribe_gate = &seams.channel.before_subscribe;
        unsubscribe_gate.arm();
        subscribe_gate.arm();
        drop(a);
        wait_parked(
            unsubscribe_gate,
            "A's teardown, just before its UNSUBSCRIBE",
        )
        .await;
        assert!(
            svc.presence_router.has_session(&session_id),
            "the router keeps the session's entry while its UNSUBSCRIBE is in flight"
        );

        let (mut b, _b_tx) = open_join(&svc, &session_id, "joiner-app").await;
        assert!(
            tokio::time::timeout(HELD_WATCH, subscribe_gate.wait_until_parked())
                .await
                .is_err(),
            "B must not reach its SUBSCRIBE while A's UNSUBSCRIBE is held"
        );
        unsubscribe_gate.release();
        wait_parked(subscribe_gate, "B, just before its SUBSCRIBE").await;
        subscribe_gate.release();

        expect_session_connected(&mut b, SETTLE).await;
        publish_probe(&tv, &session_id, PROBE).await;
        expect_peer_joined(&mut b, PROBE, "the new joiner after the race").await;
        assert_eq!(
            svc.presence_router.connection_count(),
            1,
            "only B holds a router count"
        );

        drop(b);
        drain_tracker(&svc).await;
        feed.cancel();
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Cleanup idempotence (PRD AC-7, teardown-after-sweep half)
// ═════════════════════════════════════════════════════════════════════════════

/// A connection evicted the way a sweep evicts (atomically, announcing
/// `PeerLeft`) and then torn down gets exactly one `PeerLeft` in total: the
/// teardown's unregister finds no membership to remove and announces nothing.
/// It still succeeds — the lease the eviction left behind is deleted and the
/// teardown marker armed — and the join ends `OK`. [`QUIET_REFRESH_SECS`]
/// keeps the keeper from stepping between the eviction and the teardown
/// however slowly the test runs; a step there would re-register the
/// connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_torn_down_after_an_eviction_announces_peer_left_exactly_once() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "evicted-then-torn-down", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, tx) = open_join(&svc, &session_id, "evicted-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;

        let eviction = evict_connection_for_test(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            &session_id,
            &cid,
            &PresenceEvent::PeerLeft {
                connection_id: cid.clone(),
            }
            .encode(),
        )
        .await
        .expect("the eviction must succeed");
        assert!(
            eviction.removed,
            "precondition: the eviction removed the member"
        );
        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        assert!(
            exists(&tv, &lease_key).await,
            "precondition: the eviction leaves the lease for teardown to delete"
        );

        drop(tx);
        let (_, status) = read_to_end(&mut stream, "teardown after eviction").await;
        assert!(
            status.is_none(),
            "the join must end OK when its client closes; got {status:?}"
        );
        drain_tracker(&svc).await;

        let events = tap.drain(&tv).await;
        assert_eq!(
            count(&events, |e| is_left(e, &cid)),
            1,
            "the eviction's PeerLeft must be the only one; got {events:?}"
        );
        assert_torn_down(
            &tv,
            &session_id,
            &cid,
            LEASE_TTL_SECS,
            "teardown after eviction",
        )
        .await;
        assert!(
            !is_indexed(&tv, &session_id).await,
            "the session has no member left, so teardown must unindex it"
        );
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Teardown's data-pool commands
// ═════════════════════════════════════════════════════════════════════════════

/// The data pool's round-robin position, read through `pool.last()`, which does
/// not advance it.
#[cfg(feature = "test-fault-injection")]
fn data_pool_cursor(tv: &TestValkey) -> usize {
    let pool = &tv.client().pool;
    let last = pool.last().id();
    pool.clients()
        .iter()
        .position(|client| client.id() == last)
        .expect("pool.last() is one of the pool's clients")
}

/// Teardown of a join whose lease keeper is running advances the data pool's
/// round-robin cursor exactly once, for its unregister: its keeper-stop bound
/// reads the command timeout without advancing the cursor, and a remote peer
/// keeps the session non-empty, so no index removal follows. The rejoin stall
/// tests size their pool on teardown's count.
///
/// The cursor is read before and after the teardown, and nothing else issues a
/// data-pool command in between: the join's replay has delivered its last
/// message, [`QUIET_REFRESH_SECS`] keeps the keeper from stepping, and the
/// test's own reads go through the maintenance pool.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_with_a_running_keeper_advances_the_data_pool_cursor_once() {
    scenario(async {
        const PEER: &str = "remote-peer";
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "teardown-cursor", true).await;
        register_remote_peer(&tv, &session_id, PEER).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "teardown-cursor-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        expect_peer_joined(&mut stream, PEER, "the join's replay of the remote peer").await;
        assert_eq!(
            svc.presence_cleanup_tracker.len(),
            2,
            "precondition: the join task and its lease keeper are running"
        );

        let before = data_pool_cursor(&tv);
        drop(stream);
        drain_tracker(&svc).await;
        let after = data_pool_cursor(&tv);

        assert_eq!(
            members(&tv, &session_id).await,
            vec![PEER.to_owned()],
            "precondition: teardown unregistered {cid} and the remote peer kept the session \
             non-empty"
        );
        let pool_size = tv.client().pool.clients().len();
        assert_eq!(
            (after + pool_size - before) % pool_size,
            1,
            "teardown with a running keeper must advance the data pool's cursor exactly once, for \
             its unregister (cursor {before} -> {after} of {pool_size})"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The session index heals
// ═════════════════════════════════════════════════════════════════════════════

/// The index-removal helper, run as a sweeper or another connection's
/// teardown would run it, lands between a join's first index add and its
/// register: with no member yet, it unindexes the session. The join's re-add
/// after its register restores the entry before `SessionConnected`, so a
/// registered member is never left out of the index. `gate_of` picks the park
/// point: before the router subscribe, or just before the register.
#[cfg(feature = "test-fault-injection")]
async fn index_removal_before_the_register_is_healed_by_the_join(
    gate_of: fn(&Seams) -> &PresenceSeamGate,
    what: &str,
) {
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
    let (session_id, _token) = create_session(&tv, &svc, "index-race", true).await;

    let gate = gate_of(&seams);
    gate.arm();
    let (mut stream, _tx) = open_join(&svc, &session_id, "index-race-app").await;
    wait_parked(gate, what).await;
    assert!(
        is_indexed(&tv, &session_id).await,
        "{what}: precondition: the join's first add indexed the session"
    );
    let still_indexed = remove_session_from_index(
        tv.client(),
        FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
        tv.key_schema(),
        &session_id,
    )
    .await
    .expect("the index-removal helper must succeed");
    assert!(
        !still_indexed && !is_indexed(&tv, &session_id).await,
        "{what}: precondition: with no member yet, the helper unindexes the session"
    );
    gate.release();

    let cid = expect_session_connected(&mut stream, SETTLE).await;
    assert!(
        is_indexed(&tv, &session_id).await,
        "{what}: the join's re-add after its register must restore the index entry"
    );
    assert!(is_member(&tv, &session_id, &cid).await);

    drop(stream);
    drain_tracker(&svc).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn index_removal_before_the_router_subscribe_is_healed_by_the_join() {
    scenario(index_removal_before_the_register_is_healed_by_the_join(
        |s| &s.registration.after_index_add,
        "after the first index add",
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn index_removal_just_before_the_register_is_healed_by_the_join() {
    scenario(index_removal_before_the_register_is_healed_by_the_join(
        |s| &s.registration.before_register,
        "just before the register",
    ))
    .await;
}

/// An established connection's session index entry, lost by any route (here a
/// plain `SREM`), is restored by the lease keeper's next renewal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_renewal_restores_a_lost_session_index_entry() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "keeper-heals-index", true).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "keeper-heal-app").await;
        expect_session_connected(&mut stream, SETTLE).await;

        let removed: i64 = store_reader(&tv)
            .srem(tv.key_schema().presence_active_sessions(), &session_id)
            .await
            .expect("SREM active_sessions must succeed");
        assert_eq!(removed, 1, "precondition: the join indexed the session");
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        poll_until(
            "the keeper's renewal re-adding the session to the index",
            SETTLE,
            move || is_indexed(tv_ref, sid_ref),
        )
        .await;

        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The renewal index re-add gate (SR-7.2b)
// ═════════════════════════════════════════════════════════════════════════════
//
// A keeper renewal re-adds its session to the index at most once per session
// per replica per refresh interval, gated by the service's router. Which step
// re-adds, and the unconditional re-adds after a rejoin or an info repair, are
// pinned step by step in `src/grpc/presence.rs`'s tests; these check the
// wiring through real joins.

/// Several local connections of one session: an index entry lost by any
/// route is still restored by their keepers' renewals, although only one
/// renewal per interval re-adds it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_renewals_of_several_local_connections_restore_a_lost_session_index_entry() {
    scenario(async {
        const CONNECTIONS: usize = 3;
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "keepers-heal-index", true).await;
        let mut streams = Vec::with_capacity(CONNECTIONS);
        for _ in 0..CONNECTIONS {
            let (mut stream, tx) = open_join(&svc, &session_id, "keepers-heal-app").await;
            expect_session_connected(&mut stream, SETTLE).await;
            streams.push((stream, tx));
        }

        let removed: i64 = store_reader(&tv)
            .srem(tv.key_schema().presence_active_sessions(), &session_id)
            .await
            .expect("SREM active_sessions must succeed");
        assert_eq!(removed, 1, "precondition: the joins indexed the session");
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        poll_until(
            "a keeper renewal re-adding the session to the index",
            SETTLE,
            move || is_indexed(tv_ref, sid_ref),
        )
        .await;

        drop(streams);
        drain_tracker(&svc).await;
    })
    .await;
}

/// The session's index-renewal record on the service's router survives a
/// leave that leaves another local connection, and is removed by the last
/// local connection's teardown: a connection joining afterwards is due to
/// re-add inside the interval the old record covered. [`QUIET_REFRESH_SECS`]
/// keeps every keeper from stepping, so the test's own calls are the only
/// renewals the router sees.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn last_local_leave_removes_the_sessions_index_renewal_record() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, QUIET_REFRESH_SECS);
        let router = &svc.presence_router;
        let interval = Duration::from_secs(QUIET_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "renewal-record", true).await;
        let (mut first, _first_tx) = open_join(&svc, &session_id, "renewal-record-app").await;
        expect_session_connected(&mut first, SETTLE).await;
        let (mut second, _second_tx) = open_join(&svc, &session_id, "renewal-record-app").await;
        expect_session_connected(&mut second, SETTLE).await;

        let recorded = Instant::now();
        assert!(
            router.index_renewal_due(&session_id, recorded, interval),
            "precondition: the first renewal is due and records its tick"
        );

        drop(first);
        poll_until(
            "the first connection's teardown",
            SETTLE,
            move || async move { router.connection_count() == 1 },
        )
        .await;
        assert!(
            !router.index_renewal_due(&session_id, recorded, interval),
            "a leave that leaves another local connection must keep the record"
        );

        drop(second);
        let sid_ref = session_id.as_str();
        poll_until(
            "the last connection's teardown",
            SETTLE,
            move || async move { !router.has_session(sid_ref) },
        )
        .await;
        let (mut third, _third_tx) = open_join(&svc, &session_id, "renewal-record-app").await;
        expect_session_connected(&mut third, SETTLE).await;
        assert!(
            router.index_renewal_due(&session_id, recorded, interval),
            "the last leave must have removed the record, so a later connection's renewal is due"
        );

        drop(third);
        drain_tracker(&svc).await;
    })
    .await;
}

/// How far ahead of now [`lease_keeper_renewal_re_add_is_gated_by_the_services_router`]
/// plants its record: every keeper tick in the test falls before it, so every
/// renewal inside the test is gated by it.
#[cfg(feature = "test-fault-injection")]
const PLANTED_RECORD_AHEAD: Duration = Duration::from_secs(3600);

/// The keeper consults the service's router, not a router of its own: with a
/// renewal re-add recorded on the service's router for the session (planted
/// ahead of every tick the test reaches), a keeper renewal that follows the
/// loss of the session's index entry renews the lease but does not re-add the
/// session. Without the record, the renewal re-adds it
/// (`lease_keeper_renewal_restores_a_lost_session_index_entry`).
///
/// The keeper is parked before its refresh while the record is planted and
/// the entry removed, so no renewal in flight can land its `SADD` after the
/// removal; the renewal released is then observed renewing the lease, and the
/// keeper's next step parking proves that renewal's step has finished.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_renewal_re_add_is_gated_by_the_services_router() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "gated-renewal", true).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "gated-renewal-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);

        let gate = &seams.keeper.before_refresh;
        gate.arm();
        wait_parked(gate, "the keeper's step, before its refresh").await;
        assert!(
            svc.presence_router.index_renewal_due(
                &session_id,
                Instant::now() + PLANTED_RECORD_AHEAD,
                Duration::from_secs(SHORT_REFRESH_SECS),
            ),
            "precondition: planting the record on the service's router"
        );
        let removed: i64 = store_reader(&tv)
            .srem(tv.key_schema().presence_active_sessions(), &session_id)
            .await
            .expect("SREM active_sessions must succeed");
        assert_eq!(removed, 1, "precondition: the join indexed the session");
        let shortened: bool = store_reader(&tv)
            .expire(&lease_key, SHORTENED_LEASE_SECS, None)
            .await
            .expect("EXPIRE must succeed");
        assert!(shortened, "precondition: the lease exists");
        gate.release();

        let (tv_ref, key) = (&tv, lease_key.as_str());
        poll_until(
            "the released step renewing the lease",
            SETTLE,
            move || async move { pttl(tv_ref, key).await > RENEWED_PTTL_FLOOR_MS },
        )
        .await;
        gate.arm();
        wait_parked(gate, "the keeper's next step, before its refresh").await;
        assert!(
            !is_indexed(&tv, &session_id).await,
            "a renewal gated by the service router's record must not re-add the session"
        );
        gate.release();

        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}

/// How long [`lease_keeper_renewal_after_a_late_step_still_re_adds_a_lost_session_index_entry`]
/// holds one keeper step before its refresh: half a refresh interval, so that
/// step's gate check runs half an interval after its tick while the next step
/// still starts on its own tick.
#[cfg(feature = "test-fault-injection")]
const LATE_STEP_HOLD: Duration = Duration::from_millis(SHORT_REFRESH_SECS * 1000 / 2);

/// A renewal's gate check uses the instant its step was started for, not the
/// time of the check. A session's only connection has one step held for
/// [`LATE_STEP_HOLD`] before its refresh, so that step's renewal re-adds the
/// session half an interval after the step started. The next step, not held,
/// starts on its own tick, one interval after the late step's, and its renewal
/// must re-add a removed index entry. Had the gate compared the times of its
/// checks, the two would be only half an interval apart and the next renewal
/// would skip its re-add.
///
/// The hold is inside the step, after `run` has chosen the instant it hands
/// the step, so this does not pin that choice. `src/grpc/presence.rs`'s
/// `run_hands_a_missed_tick_its_scheduled_instant_so_its_renewal_is_gated_by_it`
/// does.
///
/// Only the late step is parked. The next step's renewal is observed on the
/// lease, and parking the step after it proves the next step has finished.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_renewal_after_a_late_step_still_re_adds_a_lost_session_index_entry() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "late-step", true).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "late-step-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        let unindex = move || async move {
            let removed: i64 = store_reader(tv_ref)
                .srem(tv_ref.key_schema().presence_active_sessions(), sid_ref)
                .await
                .expect("SREM active_sessions must succeed");
            assert_eq!(removed, 1, "precondition: the session was indexed");
        };

        let gate = &seams.keeper.before_refresh;
        gate.arm();
        wait_parked(gate, "the late step, before its refresh").await;
        unindex().await;
        // The delay under test, not a synchronisation: it makes the parked
        // step's gate check run late.
        tokio::time::sleep(LATE_STEP_HOLD).await;
        gate.release();
        poll_until(
            "the late step's renewal re-adding the session",
            SETTLE,
            move || is_indexed(tv_ref, sid_ref),
        )
        .await;

        unindex().await;
        let shortened: bool = store_reader(&tv)
            .expire(&lease_key, SHORTENED_LEASE_SECS, None)
            .await
            .expect("EXPIRE must succeed");
        assert!(shortened, "precondition: the lease exists");
        let key = lease_key.as_str();
        poll_until(
            "the next step renewing the lease",
            SETTLE,
            move || async move { pttl(tv_ref, key).await > RENEWED_PTTL_FLOOR_MS },
        )
        .await;
        gate.arm();
        wait_parked(gate, "the step after the next, before its refresh").await;
        assert!(
            is_indexed(&tv, &session_id).await,
            "the renewal after a late step must re-add the session: its tick is one interval \
             after the late step's"
        );
        gate.release();

        drop(stream);
        drain_tracker(&svc).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Message ordering on a cluster (SR-1.M)
// ═════════════════════════════════════════════════════════════════════════════

// `subscribe_on_node` lives in `super::helpers`.

/// A peer on a cluster node that owns neither a session's `{sid}` slot nor its
/// presence channel's slot, subscribed to that channel.
struct ClusterObserver {
    session_id: String,
    channel: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    task: tokio::task::JoinHandle<()>,
    subscriber: fred::clients::SubscriberClient,
}

/// On the multi-shard cluster behind `tv`, create sessions (named `{name}-{i}`)
/// until one puts its `{sid}` slot and its channel's slot on different nodes,
/// register its cancellation token on `svc` as `CreateSession` on this replica
/// would, and subscribe an observer on a third primary. Those two nodes
/// differing is what lets a publish sent from the wrong one overtake another.
async fn observe_a_split_slot_session(
    tv: &TestValkey,
    svc: &LiveEditService,
    name: &str,
) -> ClusterObserver {
    use fred::interfaces::ClusterInterface;

    /// Sessions to try before concluding the cluster cannot place the `{sid}`
    /// slot and the channel slot on different nodes.
    const SESSION_CANDIDATES: usize = 64;

    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
    );
    let ks = tv.key_schema();
    let routing = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    let owner = |key: &str| {
        routing
            .get_server(fred::util::redis_keyslot(key.as_bytes()))
            .expect("every slot has a primary")
            .clone()
    };

    let store = SessionStore::new(tv.client().clone(), ks.clone());
    let mut chosen = None;
    for i in 0..SESSION_CANDIDATES {
        let session_id = store
            .create(&format!("{name}-{i}"), "/stage", "test-user")
            .await
            .expect("session creation must succeed");
        let sid_node = owner(&ks.presence_members(&session_id));
        let channel_node = owner(&ks.presence_events_channel(&session_id));
        if sid_node != channel_node {
            chosen = Some((session_id, sid_node, channel_node));
            break;
        }
    }
    let (session_id, sid_node, channel_node) = chosen.unwrap_or_else(|| {
        panic!(
            "no session among {SESSION_CANDIDATES} put its {{sid}} slot and its channel slot on \
             different nodes"
        )
    });
    let observer = routing
        .unique_primary_nodes()
        .into_iter()
        .find(|n| *n != sid_node && *n != channel_node)
        .expect("the ordering tests need at least three primaries");
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());
    let channel = ks.presence_events_channel(&session_id);
    let (rx, task, subscriber) = subscribe_on_node(&observer, &channel).await;
    ClusterObserver {
        session_id,
        channel,
        rx,
        task,
        subscriber,
    }
}

impl ClusterObserver {
    /// Every event the observer receives up to and including the `lefts`-th
    /// `PeerLeft`, under one [`SETTLE`] deadline.
    async fn until_lefts(&mut self, lefts: usize) -> Vec<PresenceEvent> {
        let rx = &mut self.rx;
        let mut events = Vec::new();
        let mut seen = 0;
        tokio::time::timeout(SETTLE, async {
            while seen < lefts {
                let event = rx.recv().await.expect("the node subscriber's task ended");
                if matches!(event, PresenceEvent::PeerLeft { .. }) {
                    seen += 1;
                }
                events.push(event);
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("only {seen} of {lefts} PeerLeft events arrived within {SETTLE:?}")
        });
        events
    }

    async fn close(self) {
        let _ = self.subscriber.unsubscribe(&self.channel).await;
        let _ = self.subscriber.quit().await;
        let _ = tokio::time::timeout(SETTLE, self.task).await;
    }
}

/// On a multi-shard cluster, a peer subscribed on a node that owns neither the
/// session's `{sid}` slot nor the channel name's slot never sees a
/// connection's `PeerMessage` before its `PeerJoined`, nor after its
/// `PeerLeft`, even when the client sends the message the moment it receives
/// `SessionConnected`. The register and unregister scripts publish from the
/// `{sid}` node, and the Rust-side `PeerMessage` publish is pinned there too;
/// unpinned it would leave from the channel's slot node, and could overtake the
/// `joined`. The session is chosen so those two nodes differ, which is what
/// makes an overtaking possible at all.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_cluster_peer_message_never_precedes_its_join_or_follows_its_leave() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    scenario(async {
        let tv = TestValkey::setup_with_url(cluster_url, 4).await;
        let svc = service(&tv, LEASE_TTL_SECS, REFRESH_SECS);
        let mut observer = observe_a_split_slot_session(&tv, &svc, "ordering").await;
        let session_id = observer.session_id.clone();

        let mut cids = Vec::with_capacity(ORDERING_TRIALS);
        for trial in 0..ORDERING_TRIALS {
            let (mut stream, tx) = open_join(&svc, &session_id, "ordering-app").await;
            let cid = expect_session_connected(&mut stream, SETTLE).await;
            tx.send(encode_join_frame(&join_message(
                format!("trial-{trial}").into_bytes(),
            )))
            .expect("the client channel must accept a message");
            drop(tx);
            let (_, status) = read_to_end(&mut stream, "ordering trial").await;
            assert!(
                status.is_none(),
                "trial {trial} must end OK; got {status:?}"
            );
            cids.push(cid);
        }

        // Every trial's PeerLeft has been published by now (each join ended
        // after its teardown); wait for all of them to arrive.
        let events = observer.until_lefts(ORDERING_TRIALS).await;

        for cid in &cids {
            let at = |pred: &dyn Fn(&PresenceEvent) -> bool| -> Vec<usize> {
                events
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| pred(e))
                    .map(|(i, _)| i)
                    .collect()
            };
            let joined = at(&|e| is_joined(e, cid));
            let messages = at(&|e| is_message(e, cid));
            let left = at(&|e| is_left(e, cid));
            assert!(
                joined.len() == 1 && messages.len() == 1 && left.len() == 1,
                "{cid}: expected one PeerJoined, one PeerMessage and one PeerLeft; got joined at \
                 {joined:?}, messages at {messages:?}, left at {left:?}"
            );
            assert!(
                joined[0] < messages[0],
                "{cid}: the PeerMessage (event {}) arrived before its PeerJoined (event {})",
                messages[0],
                joined[0]
            );
            assert!(
                messages[0] < left[0],
                "{cid}: the PeerMessage (event {}) arrived after its PeerLeft (event {})",
                messages[0],
                left[0]
            );
        }

        drain_tracker(&svc).await;
        observer.close().await;
    })
    .await;
}

/// `PUBLISH`es the pin check sends.
const PIN_PUBLISHES: u64 = 500;

/// `cmdstat_publish` calls on `node` so far, from `INFO commandstats` (0 when
/// the node has served none).
async fn publish_calls_on(tv: &TestValkey, node: &fred::types::config::Server) -> u64 {
    use fred::types::config::Options;
    use fred::types::InfoKind;

    let pinned = Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    };
    let info: String = tv
        .client()
        .pool
        .next()
        .with_options(&pinned)
        .info(Some(InfoKind::CommandStats))
        .await
        .expect("INFO commandstats must succeed");
    info.lines()
        .find_map(|line| line.trim().strip_prefix("cmdstat_publish:"))
        .and_then(|stats| {
            stats
                .split(',')
                .find_map(|field| field.strip_prefix("calls="))
                .and_then(|calls| calls.parse().ok())
        })
        .unwrap_or(0)
}

/// On a multi-shard cluster, `publish_presence_event` sends its `PUBLISH` to
/// the node owning the session's `{sid}` slot, not to the node owning the
/// channel name's slot, read off each node's `cmdstat_publish` call count
/// around [`PIN_PUBLISHES`] publishes for a session whose two slots live on
/// different nodes.
///
/// The cluster is shared with every other test in the run, and other tests'
/// `PUBLISH`es land on these nodes too, so the counts are bounded rather than
/// exact: the `{sid}` node gains at least all of ours, and the channel's node
/// gains fewer than [`PIN_PUBLISHES`] — which an unpinned `PUBLISH` (every one
/// of ours landing there) cannot satisfy. Both deltas are printed.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn publish_presence_event_on_a_cluster_is_sent_from_the_sid_node_not_the_channel_node() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    scenario(async {
        use fred::interfaces::ClusterInterface;

        let tv = TestValkey::setup_with_url(cluster_url, 4).await;
        assert!(
            tv.client().pool.next().is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
        );
        let ks = tv.key_schema().clone();
        let routing = tv
            .client()
            .pool
            .next()
            .cached_cluster_state()
            .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
        let owner = |key: &str| {
            routing
                .get_server(fred::util::redis_keyslot(key.as_bytes()))
                .expect("every slot has a primary")
                .clone()
        };
        let (session_id, sid_node, channel_node) = (0..1024)
            .map(|i| format!("pin-session-{i}"))
            .find_map(|sid| {
                let sid_node = owner(&ks.presence_members(&sid));
                let channel_node = owner(&ks.presence_events_channel(&sid));
                (sid_node != channel_node).then_some((sid, sid_node, channel_node))
            })
            .expect("some session must put its {sid} slot and its channel slot on different nodes");

        let sid_before = publish_calls_on(&tv, &sid_node).await;
        let channel_before = publish_calls_on(&tv, &channel_node).await;
        for i in 0..PIN_PUBLISHES {
            publish_presence_event(
                &tv.client().pool,
                &ks,
                &session_id,
                &PresenceEvent::PeerMessage {
                    connection_id: "pin-check".to_owned(),
                    data: i.to_be_bytes().to_vec(),
                },
            )
            .await
            .expect("PUBLISH must succeed");
        }
        let sid_delta = publish_calls_on(&tv, &sid_node).await - sid_before;
        let channel_delta = publish_calls_on(&tv, &channel_node).await - channel_before;
        eprintln!(
            "pin check: {PIN_PUBLISHES} publishes; {{sid}} node {sid_node} +{sid_delta}, channel \
             node {channel_node} +{channel_delta}"
        );
        assert!(
            sid_delta >= PIN_PUBLISHES,
            "every PUBLISH must be sent to the {{sid}} node {sid_node}; it gained {sid_delta} of \
             {PIN_PUBLISHES}"
        );
        assert!(
            channel_delta < PIN_PUBLISHES,
            "no PUBLISH may be sent to the channel's node {channel_node}; it gained \
             {channel_delta}, at least as many as were sent"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Self-rejoin after an eviction (SR-4.2 path A, PRD AC-4)
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_presence_rejoins_total` on `svc`'s metrics.
fn rejoins(svc: &LiveEditService) -> f64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(&svc.metrics),
        "live_edit_presence_rejoins_total",
        &[("service", svc.service_label.as_str())],
    )
    .expect("the rejoin cell is seeded in Metrics::new")
}

/// The one `PeerJoined` for `cid` among `events`: the user name and
/// application every later announcement of `cid` must carry.
fn the_join_of(events: &[PresenceEvent], cid: &str) -> ConnectedUser {
    let joins: Vec<&ConnectedUser> = events
        .iter()
        .filter_map(|e| match e {
            PresenceEvent::PeerJoined(user) if user.connection_id == cid => Some(user),
            _ => None,
        })
        .collect();
    let [user] = joins.as_slice() else {
        panic!("expected exactly one PeerJoined for {cid}; got {events:?}");
    };
    (*user).clone()
}

/// The `PeerJoined` and `PeerLeft` events for `cid` among `events`, in order.
fn membership_events_of(events: &[PresenceEvent], cid: &str) -> Vec<PresenceEvent> {
    events
        .iter()
        .filter(|e| is_joined(e, cid) || is_left(e, cid))
        .cloned()
        .collect()
}

fn left(cid: &str) -> PresenceEvent {
    PresenceEvent::PeerLeft {
        connection_id: cid.to_owned(),
    }
}

async fn info_hash(tv: &TestValkey, session_id: &str, cid: &str) -> HashMap<String, String> {
    store_reader(tv)
        .hgetall(tv.key_schema().presence_user(session_id, cid))
        .await
        .expect("HGETALL must succeed")
}

/// A register wrote `info`: the user name and application `user` was joined
/// with, and a lease expiry.
fn assert_info_complete(info: &HashMap<String, String>, user: &ConnectedUser, what: &str) {
    assert_eq!(
        info.get("user_name"),
        Some(&user.user_name),
        "{what}: the info hash must carry the join's user name; got {info:?}"
    );
    assert_eq!(
        info.get("user_application"),
        Some(&user.user_application),
        "{what}: the info hash must carry the join's user application; got {info:?}"
    );
    assert!(
        info.contains_key("lease_expires_at_ms"),
        "{what}: the info hash must carry the lease expiry; got {info:?}"
    );
}

/// Read a new joiner's stream until its replay delivers `cid`; returns the
/// replayed user name and application.
async fn expect_replayed(stream: &mut JoinResponseStream, cid: &str) -> (String, String) {
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            let msg = item.unwrap_or_else(|s| panic!("the new joiner's stream failed: {s:?}"));
            if let Some(join_session_response::ResponseType::PeerJoined(p)) = msg.response_type {
                if p.connection_id == cid {
                    return (p.user_name, p.user_application);
                }
            }
        }
        panic!("the new joiner's stream ended before its replay listed {cid}");
    })
    .await
    .unwrap_or_else(|_| panic!("the new joiner's replay did not list {cid} within {SETTLE:?}"))
}

/// An established connection whose lease lapsed and was evicted is re-registered
/// by its own keeper at the next tick, under the same connection id. Peers see
/// the eviction's `PeerLeft` and then exactly one `PeerJoined` carrying the
/// join's user name and application; the member entry, a complete info hash and
/// the lease exist again; a new joiner's replay lists the connection; and one
/// rejoin is counted. The keeper then goes back to renewing: a later renewal
/// announces nothing and counts no second rejoin.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_rejoins_an_evicted_connection_announcing_left_then_joined_exactly_once() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "rejoin", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "rejoin-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let registered = the_join_of(&tap.drain(&tv).await, &cid);
        assert_eq!(rejoins(&svc), 0.0, "precondition: no rejoin yet");

        evict(&tv, &session_id, &cid).await;
        let announced = tap
            .until("the rejoin's PeerJoined", |e| is_joined(e, &cid))
            .await;
        assert_eq!(
            announced,
            vec![left(&cid), PresenceEvent::PeerJoined(registered.clone())],
            "peers must see the eviction's PeerLeft, then one PeerJoined with the join's own user \
             name and application"
        );
        let svc_ref = &svc;
        poll_until("the rejoin being counted", SETTLE, move || async move {
            rejoins(svc_ref) == 1.0
        })
        .await;

        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        assert!(
            is_member(&tv, &session_id, &cid).await,
            "the rejoin must restore the member entry"
        );
        assert!(
            exists(&tv, &lease_key).await,
            "the rejoin must restore the lease"
        );
        assert_info_complete(
            &info_hash(&tv, &session_id, &cid).await,
            &registered,
            "after the rejoin",
        );

        let (mut newcomer, _newcomer_tx) = open_join(&svc, &session_id, "newcomer-app").await;
        expect_session_connected(&mut newcomer, SETTLE).await;
        assert_eq!(
            expect_replayed(&mut newcomer, &cid).await,
            (
                registered.user_name.clone(),
                registered.user_application.clone()
            ),
            "a new joiner's replay must list the rejoined connection as it joined"
        );

        expect_a_later_renewal(&tv, &lease_key, "a renewal after the rejoin").await;
        let later = tap.drain(&tv).await;
        assert_eq!(
            membership_events_of(&later, &cid),
            Vec::<PresenceEvent>::new(),
            "once rejoined, the keeper renews without announcing anything; got {later:?}"
        );
        assert_eq!(rejoins(&svc), 1.0, "one eviction, one rejoin");

        drop((stream, newcomer));
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

/// A rejoin register that reaches the store after its deadline (the store-clock
/// read plus one lease TTL) is refused with `Expired` and writes nothing. With
/// no stall and no teardown, the keeper itself gets `Expired`: it counts one
/// failed step and goes on, and its next step, reading a fresh clock,
/// re-registers the connection. Peers see the eviction's `PeerLeft` and then
/// exactly one `PeerJoined`, and exactly one rejoin is counted.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_register_past_its_deadline_writes_nothing_and_the_next_step_rejoins() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, SHORT_LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "expired-rejoin", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "expired-rejoin-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let registered = the_join_of(&tap.drain(&tv).await, &cid);

        let gate = &seams.keeper.before_rejoin_register;
        gate.arm();
        evict(&tv, &session_id, &cid).await;
        wait_parked(gate, "the keeper's rejoin, before its register").await;

        // The keeper read the store clock before parking, so a read now is no
        // earlier; once the store clock is a lease TTL past it, the parked
        // register's deadline has passed.
        let origin = FredErrOrigin::client(tv.client(), POOL_LABEL_DATA);
        let (tv_ref, sid_ref) = (&tv, session_id.as_str());
        let store_now = move || async move {
            store_time_ms(tv_ref.client(), origin, tv_ref.key_schema(), sid_ref)
                .await
                .expect("store_time_ms must succeed")
        };
        let past_deadline = store_now().await + SHORT_LEASE_TTL_SECS * 1000;
        poll_until(
            "the store clock passing the rejoin register's deadline",
            SETTLE,
            move || async move { store_now().await > past_deadline },
        )
        .await;
        gate.release();

        let announced = tap
            .until("the next step's rejoin PeerJoined", |e| is_joined(e, &cid))
            .await;
        assert_eq!(
            announced,
            vec![left(&cid), PresenceEvent::PeerJoined(registered.clone())],
            "the expired register must announce nothing and the next step's register exactly one \
             PeerJoined"
        );
        let svc_ref = &svc;
        poll_until(
            "the next step's rejoin being counted",
            SETTLE,
            move || async move { rejoins(svc_ref) == 1.0 },
        )
        .await;
        assert_eq!(
            lease_refresh_failures(&svc),
            1.0,
            "the expired register is the one failed step"
        );
        assert_info_complete(
            &info_hash(&tv, &session_id, &cid).await,
            &registered,
            "after the next step's rejoin",
        );

        drop(stream);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Refresh never creates the info hash (SR-4.2)
// ═════════════════════════════════════════════════════════════════════════════

/// A live member's info hash is deleted. The keeper's next refresh reports
/// `InfoMissing` rather than writing a lease expiry into a new, nameless hash,
/// and the keeper re-registers: the hash comes back whole, with the join's user
/// name and application. At no probe does the hash exist without its user name,
/// through the repair and two later renewals. The connection never stopped
/// being a member, so the repair announces nothing and counts no rejoin.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_keeper_repairs_a_deleted_info_hash_without_announcing_a_join() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "info-repair", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "info-repair-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let registered = the_join_of(&tap.drain(&tv).await, &cid);

        let deleted: i64 = store_reader(&tv)
            .del(tv.key_schema().presence_user(&session_id, &cid))
            .await
            .expect("DEL info hash must succeed");
        assert_eq!(deleted, 1, "precondition: the join wrote the info hash");

        let deadline = Instant::now() + SETTLE;
        loop {
            let info = info_hash(&tv, &session_id, &cid).await;
            if !info.is_empty() {
                assert_info_complete(&info, &registered, "the info hash's first reappearance");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the keeper did not re-register the deleted info hash within {SETTLE:?}"
            );
            tokio::time::sleep(POLL).await;
        }
        let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
        for round in 1..=2 {
            let what = format!("renewal round {round} after the repair");
            expect_a_later_renewal(&tv, &lease_key, &what).await;
            assert_info_complete(&info_hash(&tv, &session_id, &cid).await, &registered, &what);
        }
        assert!(
            is_member(&tv, &session_id, &cid).await,
            "the repaired connection is still a member"
        );

        let events = tap.drain(&tv).await;
        assert_eq!(
            membership_events_of(&events, &cid),
            Vec::<PresenceEvent>::new(),
            "repairing a current member's info hash announces nothing; got {events:?}"
        );
        assert_eq!(rejoins(&svc), 0.0, "an info repair is not a rejoin");

        drop(stream);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Teardown racing a rejoin (PRD AC-5, rejoin half)
// ═════════════════════════════════════════════════════════════════════════════

/// An established connection is evicted and its keeper parked inside the
/// rejoin, after the store-clock read and before the register, when `trigger`
/// ends the join. The rejoin's second stop check then stops the step, so no
/// register is issued:
///
/// - `DeleteSession`, a session-token cancel and a service shutdown cancel one
///   of the keeper's own tokens before `fire` returns, and the park is
///   released at once, while teardown is still waiting for the keeper. A
///   register issued here would re-add the member and announce `PeerJoined`
///   before teardown's unregister, which would then announce a second
///   `PeerLeft`.
/// - A client disconnect reaches the keeper only through the join task's
///   teardown, which cancels the keeper's teardown token at a moment the test
///   cannot observe, so the park is held until the join task has finished its
///   whole teardown (waited out its bound for the keeper, unregistered,
///   unindexed the session, released the router) and the parked keeper is all
///   that is left on the tracker. With only the teardown token cancelled, the
///   release is made against a fully stalled data pool: no stalled
///   connection may receive a byte, so the second check must have stopped the
///   step before its register. (Without that check the register would be
///   issued, and refused only in the store by the teardown marker, which the
///   store's end state alone cannot tell apart.)
///
/// Afterwards the eviction's `PeerLeft` is the only membership event for the
/// connection (teardown's unregister found no member to remove, so it
/// announced nothing, and no `PeerJoined` came after it); the member entry,
/// info hash and lease are absent; the session is unindexed; no rejoin is
/// counted; and the join task and keeper have both finished.
#[cfg(feature = "test-fault-injection")]
async fn teardown_racing_a_parked_rejoin_leaves_nothing_behind(trigger: Trigger) {
    let what = format!("teardown on {trigger:?} racing a parked rejoin");
    let tv = TestValkey::setup().await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
    let (session_id, token) = create_session(&tv, &svc, "teardown-vs-rejoin", true).await;
    let mut tap = Tap::open(&tv, &session_id).await;
    let (mut stream, _tx) = open_join(&svc, &session_id, "rejoin-race-app").await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;
    tap.drain(&tv).await;

    let gate = &seams.keeper.before_rejoin_register;
    gate.arm();
    evict(&tv, &session_id, &cid).await;
    wait_parked(gate, "the keeper's rejoin, before its register").await;

    let stream = fire(trigger, &svc, &session_id, &token, stream).await;
    if matches!(trigger, Trigger::ClientDisconnect) {
        let tracker = &svc.presence_cleanup_tracker;
        poll_until(
            "the join task finishing its teardown, leaving only the parked keeper",
            TEARDOWN_WAIT,
            move || async move { tracker.len() == 1 },
        )
        .await;
        let stalls = tv.stall_data_pool().await;
        let baseline = net_in_of(&stalls).await;
        gate.release();
        drain_tracker(&svc).await;
        let after = net_in_of(&stalls).await;
        // Released before asserting, so a failure does not leave the data pool
        // stalled under the fixture's cleanup.
        release_all(stalls).await;
        assert_eq!(
            after, baseline,
            "{what}: the released rejoin must send nothing to the store once teardown has begun"
        );
    } else {
        gate.release();
    }

    if let Some(mut stream) = stream {
        let (_, status) = read_to_end(&mut stream, &what).await;
        assert_ending(trigger.ending(), status, &what);
    }
    drain_tracker(&svc).await;

    let events = tap.drain(&tv).await;
    assert_eq!(
        membership_events_of(&events, &cid),
        vec![left(&cid)],
        "{what}: the eviction's PeerLeft must be the connection's only membership event, with no \
         PeerJoined from the stopped rejoin; got {events:?}"
    );
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
    assert!(
        !is_indexed(&tv, &session_id).await,
        "{what}: the session has no member left, so teardown must unindex it"
    );
    assert_eq!(rejoins(&svc), 0.0, "{what}: no rejoin may be counted");
    tap.close().await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_client_disconnect_racing_a_parked_rejoin_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_rejoin_leaves_nothing_behind(
        Trigger::ClientDisconnect,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_delete_session_racing_a_parked_rejoin_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_rejoin_leaves_nothing_behind(
        Trigger::DeleteSession,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_session_token_cancel_racing_a_parked_rejoin_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_rejoin_leaves_nothing_behind(
        Trigger::SessionTokenCancel,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teardown_on_service_shutdown_racing_a_parked_rejoin_leaves_nothing_behind() {
    scenario(teardown_racing_a_parked_rejoin_leaves_nothing_behind(
        Trigger::ServiceShutdown,
    ))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// What a rejoin does after its register
// ═════════════════════════════════════════════════════════════════════════════

/// **A rejoin register refused by the teardown marker stops the keeper.** The
/// connection is evicted and its keeper parked before the rejoin register.
/// The connection is then torn down straight through the store
/// (`unregister_connection`, which arms the teardown marker), with no token
/// cancelled, so the rejoin's stop checks pass and its register meets the
/// marker: `Gone`. The keeper stops then and there — its next step, which
/// would park at `before_refresh`, armed as a tripwire, never comes — while
/// the join task stays in its loop. The refusal counts neither a failed step
/// nor a rejoin, and writes nothing: no member entry, info hash or lease, and
/// no `PeerJoined` after the eviction's `PeerLeft`.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_register_refused_by_the_teardown_marker_stops_the_keeper() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "rejoin-gone", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, tx) = open_join(&svc, &session_id, "rejoin-gone-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        tap.drain(&tv).await;
        assert_eq!(
            svc.presence_cleanup_tracker.len(),
            2,
            "precondition: the join task and its lease keeper are running"
        );

        let rejoin_gate = &seams.keeper.before_rejoin_register;
        rejoin_gate.arm();
        evict(&tv, &session_id, &cid).await;
        wait_parked(rejoin_gate, "the keeper's rejoin, before its register").await;
        let unregistered = unregister_connection(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            &session_id,
            &cid,
            LEASE_TTL_SECS,
            &left(&cid).encode(),
        )
        .await
        .expect("unregistering the evicted connection must succeed");
        assert!(
            !unregistered.removed,
            "precondition: the eviction already removed the member"
        );

        let refresh_gate = &seams.keeper.before_refresh;
        refresh_gate.arm();
        rejoin_gate.release();
        let tracker = &svc.presence_cleanup_tracker;
        poll_until(
            "the keeper stopping on its refused rejoin register",
            SETTLE,
            move || async move { tracker.len() == 1 },
        )
        .await;
        refresh_gate.release();

        assert_eq!(
            lease_refresh_failures(&svc),
            0.0,
            "a register refused by the teardown marker is not a failed step"
        );
        assert_eq!(rejoins(&svc), 0.0, "a refused register is not a rejoin");
        let ks = tv.key_schema();
        assert!(
            !is_member(&tv, &session_id, &cid).await,
            "the refused register must not re-add the member"
        );
        assert!(
            !exists(&tv, &ks.presence_user(&session_id, &cid)).await,
            "the refused register must not recreate the info hash"
        );
        assert!(
            !exists(&tv, &ks.presence_lease(&session_id, &cid)).await,
            "the refused register must not recreate the lease"
        );

        drop(tx);
        read_to_ok_end(&mut stream, "client close after the keeper stopped").await;
        drain_tracker(&svc).await;
        let events = tap.drain(&tv).await;
        assert_eq!(
            membership_events_of(&events, &cid),
            vec![left(&cid)],
            "the eviction's PeerLeft must be the connection's only membership event; got \
             {events:?}"
        );
        tap.close().await;
    })
    .await;
}

/// **A rejoin re-adds its session to the index.** The connection is its
/// session's only member. It is evicted and its keeper parked before the
/// rejoin register, and the index-removal helper, run as a sweep or another
/// connection's teardown would run it, unindexes the now memberless session.
/// Released, the rejoin re-registers the connection and re-adds the session to
/// the index within the same step: once the `PeerJoined` has arrived and the
/// keeper's next step is parked before its refresh, so that no renewal can
/// have re-added it, the session is indexed.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_re_adds_its_session_to_the_session_index() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "rejoin-reindex", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, _tx) = open_join(&svc, &session_id, "rejoin-reindex-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        tap.drain(&tv).await;

        let rejoin_gate = &seams.keeper.before_rejoin_register;
        rejoin_gate.arm();
        evict(&tv, &session_id, &cid).await;
        wait_parked(rejoin_gate, "the keeper's rejoin, before its register").await;
        let still_indexed = remove_session_from_index(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            tv.key_schema(),
            &session_id,
        )
        .await
        .expect("the index-removal helper must succeed");
        assert!(
            !still_indexed && !is_indexed(&tv, &session_id).await,
            "precondition: with no member, the helper unindexes the session"
        );

        let refresh_gate = &seams.keeper.before_refresh;
        refresh_gate.arm();
        rejoin_gate.release();
        tap.until("the rejoin's PeerJoined", |e| is_joined(e, &cid))
            .await;
        wait_parked(refresh_gate, "the keeper's next step, before its refresh").await;
        assert!(
            is_indexed(&tv, &session_id).await,
            "the rejoin must re-add its session to the index"
        );
        refresh_gate.release();

        drop(stream);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A rejoin register held behind a stalled connection
// ═════════════════════════════════════════════════════════════════════════════
//
// The keeper is parked at `before_rejoin_register` (its store-clock read has
// completed), every data-pool connection is stalled, and the park is released:
// the register alone queues on one stalled connection, found by its
// `tot-net-in` growing, and fails at the 5 s command timeout while it waits.
//
// The data pool is `STALL_POOL_SIZE` connections so that the commands that
// follow the register never land on its connection: each `pool.next()`
// advances one round-robin cursor, and the most that follow the register
// before the test releases it are four: a next step's refresh, clock read,
// register and index re-add. Teardown takes three (its unregister, and the
// index removal's `SREM` and `SCARD`); its keeper-stop bound reads the command
// timeout through `pool.last()`, which does not advance the cursor. Every store
// read goes through the maintenance pool, and the tap publishes its fence
// only once every stall is released, so neither moves the cursor meanwhile.
// The tests do not rely on that count alone: where one needs a command to
// overtake the held register (teardown's unregister, a next step's register),
// it checks that the command did so while the register's connection was still
// stalled.

/// Data-pool size of the stall tests: the held register's connection plus the
/// four that the commands following it can take, with headroom.
#[cfg(feature = "test-fault-injection")]
const STALL_POOL_SIZE: u32 = 8;

/// `tot-net-in` of every stalled connection, in order.
#[cfg(feature = "test-fault-injection")]
async fn net_in_of(stalls: &[ConnStall]) -> Vec<u64> {
    let mut net_in = Vec::with_capacity(stalls.len());
    for stall in stalls {
        net_in.push(
            stall
                .net_in()
                .await
                .expect("the stalled connection is open"),
        );
    }
    net_in
}

/// The index of the first stalled connection whose `tot-net-in` grows past
/// `baseline`: the one a command released after the baseline queued on.
#[cfg(feature = "test-fault-injection")]
async fn stalled_connection_taking(stalls: &[ConnStall], baseline: &[u64], what: &str) -> usize {
    let deadline = Instant::now() + SETTLE;
    loop {
        for (i, stall) in stalls.iter().enumerate() {
            if stall.net_in().await.unwrap_or(0) > baseline[i] {
                return i;
            }
        }
        assert!(
            Instant::now() < deadline,
            "{what} did not reach a stalled connection within {SETTLE:?}"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// Wait until every command already sent on data-pool connection `conn` has
/// run: a `PING` sent on it now is answered only after them.
#[cfg(feature = "test-fault-injection")]
async fn flush_connection(tv: &TestValkey, conn: usize) {
    let pong: String = tokio::time::timeout(SETTLE, tv.client().pool.clients()[conn].ping(None))
        .await
        .unwrap_or_else(|_| {
            panic!("PING on data connection {conn} did not return within {SETTLE:?}")
        })
        .expect("PING must succeed");
    assert_eq!(pong, "PONG");
}

/// A keeper's rejoin register queued on one stalled data-pool connection.
#[cfg(feature = "test-fault-injection")]
struct HeldRegister {
    /// The stall the register is queued behind.
    register: ConnStall,
    /// That connection's index in the data pool (`stall_data_pool` stalls the
    /// pool's connections in order).
    conn: usize,
    /// Every other data-pool connection, still stalled.
    others: Vec<ConnStall>,
}

/// Evict `cid`, park its keeper's rejoin before the register, stall the whole
/// data pool, and release the park: the register queues on one stalled
/// connection.
#[cfg(feature = "test-fault-injection")]
async fn hold_the_rejoin_register(
    tv: &TestValkey,
    seams: &Seams,
    session_id: &str,
    cid: &str,
) -> HeldRegister {
    let gate = &seams.keeper.before_rejoin_register;
    gate.arm();
    evict(tv, session_id, cid).await;
    wait_parked(gate, "the keeper's rejoin, before its register").await;

    let mut stalls = tv.stall_data_pool().await;
    let baseline = net_in_of(&stalls).await;
    gate.release();
    let conn = stalled_connection_taking(&stalls, &baseline, "the rejoin register").await;
    let register = stalls.remove(conn);
    HeldRegister {
        register,
        conn,
        others: stalls,
    }
}

/// An established join, as the stall tests need it: observability hooks
/// installed so command timeouts are counted, the tap open, and the join's own
/// `PeerJoined` drained from it.
#[cfg(feature = "test-fault-injection")]
async fn stall_test_join(
    tv: &TestValkey,
    svc: &LiveEditService,
    name: &str,
) -> (
    String,
    Tap,
    String,
    JoinResponseStream,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
) {
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    let (session_id, _token) = create_session(tv, svc, name, true).await;
    let mut tap = Tap::open(tv, &session_id).await;
    let (mut stream, tx) = open_join(svc, &session_id, &format!("{name}-app")).await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;
    tap.drain(tv).await;
    (session_id, tap, cid, stream, tx)
}

/// How long a late rejoin register is held.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum LateRejoin {
    /// Past teardown's unregister, while the teardown marker exists.
    PastTheUnregister,
    /// Past the teardown marker's expiry, one lease TTL after the unregister.
    PastTheMarkersExpiry,
}

/// **Late write fenced.** An evicted connection's rejoin register is held on a
/// stalled connection and the client disconnects. The register was issued
/// before teardown cancelled the keeper, so it times out (one command timeout,
/// one failed keeper step) before teardown's bounded wait for the keeper runs
/// out, and the keeper stops. Teardown then unregisters on another connection:
/// the teardown marker appears while the register is still held. The register
/// is released only once it has been held as long as `hold` says, and a `PING`
/// sent behind it on the same connection proves it ran.
///
/// It must write nothing:
///
/// - held past the unregister, the teardown marker refuses it (the register
///   script's first check; the deadline, one production lease TTL away, has
///   not passed);
/// - held past the marker's expiry, with a short lease TTL, its deadline
///   refuses it: the keeper read the store clock before teardown began, so
///   the deadline fell before the marker expired.
///
/// Afterwards the member entry, info hash and lease are absent, the session
/// is unindexed, the eviction's `PeerLeft` is the connection's only membership
/// event, and no rejoin is counted.
#[cfg(feature = "test-fault-injection")]
async fn late_rejoin_register_writes_nothing(hold: LateRejoin) {
    let what = format!("rejoin register held {hold:?}");
    let lease_ttl_secs = match hold {
        LateRejoin::PastTheUnregister => LEASE_TTL_SECS,
        LateRejoin::PastTheMarkersExpiry => SHORT_LEASE_TTL_SECS,
    };
    let tv = TestValkey::setup_with_pool_size(STALL_POOL_SIZE).await;
    let (svc, seams) = seamed_service(&tv, lease_ttl_secs, SHORT_REFRESH_SECS);
    let (session_id, mut tap, cid, stream, _tx) = stall_test_join(&tv, &svc, "late-rejoin").await;

    let held = hold_the_rejoin_register(&tv, &seams, &session_id, &cid).await;
    drop(stream);
    release_all(held.others).await;

    let gone_key = tv.key_schema().presence_gone(&session_id, &cid);
    let (tv_ref, gone_ref) = (&tv, gone_key.as_str());
    poll_until(
        &format!("{what}: teardown's unregister"),
        TEARDOWN_WAIT,
        move || exists(tv_ref, gone_ref),
    )
    .await;
    assert!(
        held.register.is_blocked().await,
        "{what}: teardown's unregister must land while the register is still held, on another \
         connection"
    );
    assert_eq!(
        command_timeout_row(&svc.metrics),
        command_timeouts_only_on(POOL_LABEL_DATA, 1),
        "{what}: the keeper's register must have timed out before teardown unregistered"
    );
    assert_eq!(
        lease_refresh_failures(&svc),
        1.0,
        "{what}: the timed-out register is the keeper's one failed step"
    );
    if matches!(hold, LateRejoin::PastTheMarkersExpiry) {
        poll_until(
            &format!("{what}: the teardown marker expiring"),
            SETTLE,
            move || async move { !exists(tv_ref, gone_ref).await },
        )
        .await;
    }

    let released = held.register.release().await;
    assert!(
        matches!(&released, Ok(v) if !v.is_null()),
        "{what}: the register's connection must have stayed stalled until now; got {released:?}"
    );
    flush_connection(&tv, held.conn).await;
    drain_tracker(&svc).await;

    let ks = tv.key_schema();
    assert!(
        !is_member(&tv, &session_id, &cid).await,
        "{what}: the late register must not re-add the member"
    );
    assert!(
        !exists(&tv, &ks.presence_user(&session_id, &cid)).await,
        "{what}: the late register must not recreate the info hash"
    );
    assert!(
        !exists(&tv, &ks.presence_lease(&session_id, &cid)).await,
        "{what}: the late register must not recreate the lease"
    );
    assert!(
        !is_indexed(&tv, &session_id).await,
        "{what}: the session has no member, so it must stay unindexed"
    );
    let events = tap.drain(&tv).await;
    assert_eq!(
        membership_events_of(&events, &cid),
        vec![left(&cid)],
        "{what}: the eviction's PeerLeft must be the connection's only membership event; got \
         {events:?}"
    );
    assert_eq!(rejoins(&svc), 0.0, "{what}: no rejoin may be counted");
    tap.close().await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_register_held_past_teardowns_unregister_is_refused_by_the_teardown_marker() {
    scenario(late_rejoin_register_writes_nothing(
        LateRejoin::PastTheUnregister,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_register_held_past_the_teardown_markers_expiry_is_refused_by_its_deadline() {
    scenario(late_rejoin_register_writes_nothing(
        LateRejoin::PastTheMarkersExpiry,
    ))
    .await;
}

/// When a timed-out rejoin register applies, relative to the keeper's next
/// refresh.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum LateApply {
    /// The register applies (re-adding and announcing) first; the next
    /// refresh finds the connection renewed.
    BeforeTheNextRefresh,
    /// The next refresh finds the connection still evicted, and its step's
    /// register re-adds and announces; the late register then lands on a
    /// member and publishes nothing.
    AfterTheNextRefresh,
}

/// **Rejoin announced despite failure.** An evicted connection's rejoin
/// register is held on a stalled connection past its command timeout (the
/// keeper counts one failed step), then applies. Whichever way it lands
/// relative to the keeper's next refresh, peers see exactly one `PeerJoined`
/// between the eviction's `PeerLeft` and teardown's, the connection is fully
/// registered and renewed afterwards, and teardown leaves nothing behind.
///
/// The rejoin counter undercounts and never overcounts, and each order fixes
/// its value: applied before the next refresh, the re-adding register's reply
/// was lost to the timeout and no rejoin is counted; applied after it, the
/// next step's register saw `added: true` and counted the one rejoin, while
/// the late register landed on a member and counted nothing.
#[cfg(feature = "test-fault-injection")]
async fn timed_out_rejoin_register_that_applies_is_announced_once(order: LateApply) {
    let what = format!("timed-out rejoin register applying {order:?}");
    let tv = TestValkey::setup_with_pool_size(STALL_POOL_SIZE).await;
    let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
    let (session_id, mut tap, cid, stream, _tx) =
        stall_test_join(&tv, &svc, "timed-out-rejoin").await;

    let held = hold_the_rejoin_register(&tv, &seams, &session_id, &cid).await;
    let metrics = &svc.metrics;
    poll_until(
        &format!("{what}: the register's command timeout"),
        SETTLE,
        move || async move {
            command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
        },
    )
    .await;

    let (tv_ref, sid_ref, cid_ref) = (&tv, session_id.as_str(), cid.as_str());
    let mut events = Vec::new();
    match order {
        LateApply::BeforeTheNextRefresh => {
            let released = held.register.release().await;
            assert!(
                matches!(&released, Ok(v) if !v.is_null()),
                "{what}: the register's connection must have stayed stalled until now; got \
                 {released:?}"
            );
            poll_until(
                &format!("{what}: the late register re-adding the member"),
                SETTLE,
                move || is_member(tv_ref, sid_ref, cid_ref),
            )
            .await;
            release_all(held.others).await;
        }
        LateApply::AfterTheNextRefresh => {
            release_all(held.others).await;
            events.extend(
                tap.until("the next step's rejoin PeerJoined", |e| is_joined(e, &cid))
                    .await,
            );
            assert!(
                held.register.is_blocked().await,
                "{what}: the next step's register must re-add the member while the late one is \
                 still held"
            );
            let released = held.register.release().await;
            assert!(
                matches!(&released, Ok(v) if !v.is_null()),
                "{what}: the register's connection must have stayed stalled until now; got \
                 {released:?}"
            );
            flush_connection(&tv, held.conn).await;
        }
    }

    let lease_key = tv.key_schema().presence_lease(&session_id, &cid);
    expect_a_later_renewal(&tv, &lease_key, &format!("{what}: a renewal afterwards")).await;
    assert!(
        is_member(&tv, &session_id, &cid).await,
        "{what}: the connection must be a member again"
    );
    assert_eq!(
        lease_refresh_failures(&svc),
        1.0,
        "{what}: the timed-out register is the keeper's one failed step"
    );

    drop(stream);
    drain_tracker(&svc).await;
    events.extend(tap.drain(&tv).await);
    let rejoined = the_join_of(&events, &cid);
    assert_eq!(
        membership_events_of(&events, &cid),
        vec![left(&cid), PresenceEvent::PeerJoined(rejoined), left(&cid)],
        "{what}: peers must see the eviction's PeerLeft, exactly one PeerJoined, then teardown's \
         PeerLeft; got {events:?}"
    );
    let counted = match order {
        LateApply::BeforeTheNextRefresh => 0.0,
        LateApply::AfterTheNextRefresh => 1.0,
    };
    assert_eq!(
        rejoins(&svc),
        counted,
        "{what}: only a register whose added: true the keeper saw is counted"
    );
    assert_torn_down(&tv, &session_id, &cid, LEASE_TTL_SECS, &what).await;
    tap.close().await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timed_out_rejoin_register_applying_before_the_next_refresh_is_announced_once() {
    scenario(timed_out_rejoin_register_that_applies_is_announced_once(
        LateApply::BeforeTheNextRefresh,
    ))
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timed_out_rejoin_register_applying_after_the_next_refresh_is_announced_once() {
    scenario(timed_out_rejoin_register_that_applies_is_announced_once(
        LateApply::AfterTheNextRefresh,
    ))
    .await;
}

/// A rejoin whose store-clock read (the base of its register deadline) times
/// out issues no register: the keeper counts one failed step, nothing is
/// written, and its next step, reading a fresh clock, re-registers the
/// connection with exactly one `PeerJoined`.
///
/// The keeper is parked before its refresh while the connection is evicted and
/// the data pool stalled. Its refresh queues on one stalled connection, which
/// is released alone; the refresh then reports `Evicted`, and the rejoin's
/// clock read queues on another stalled connection, where it times out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_whose_store_clock_read_times_out_counts_one_failure_and_the_next_step_rejoins() {
    scenario(async {
        let tv = TestValkey::setup_with_pool_size(STALL_POOL_SIZE).await;
        let (svc, seams) = seamed_service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, mut tap, cid, stream, _tx) =
            stall_test_join(&tv, &svc, "rejoin-clock-read").await;

        let gate = &seams.keeper.before_refresh;
        gate.arm();
        wait_parked(gate, "the keeper's refresh").await;
        evict(&tv, &session_id, &cid).await;

        let mut stalls = tv.stall_data_pool().await;
        let mut baseline = net_in_of(&stalls).await;
        gate.release();
        let refresh_conn = stalled_connection_taking(&stalls, &baseline, "the refresh").await;
        baseline.remove(refresh_conn);
        let released = stalls.remove(refresh_conn).release().await;
        assert!(
            matches!(&released, Ok(v) if !v.is_null()),
            "the refresh's connection must have stayed stalled until now; got {released:?}"
        );
        stalled_connection_taking(&stalls, &baseline, "the rejoin's store-clock read").await;

        let metrics = &svc.metrics;
        poll_until(
            "the rejoin's store-clock read timing out",
            SETTLE,
            move || async move {
                command_timeout_row(metrics) == command_timeouts_only_on(POOL_LABEL_DATA, 1)
            },
        )
        .await;
        let svc_ref = &svc;
        poll_until(
            "the failed step being counted",
            SETTLE,
            move || async move { lease_refresh_failures(svc_ref) == 1.0 },
        )
        .await;
        assert!(
            !is_member(&tv, &session_id, &cid).await,
            "a rejoin whose clock read failed must not register"
        );
        release_all(stalls).await;

        let announced = tap
            .until("the next step's rejoin PeerJoined", |e| is_joined(e, &cid))
            .await;
        let rejoined = the_join_of(&announced, &cid);
        assert_eq!(
            membership_events_of(&announced, &cid),
            vec![left(&cid), PresenceEvent::PeerJoined(rejoined)],
            "the eviction's PeerLeft, then the next step's one PeerJoined; got {announced:?}"
        );
        poll_until(
            "the next step's rejoin being counted",
            SETTLE,
            move || async move { rejoins(svc_ref) == 1.0 },
        )
        .await;
        assert_eq!(
            lease_refresh_failures(&svc),
            1.0,
            "the failed clock read is the one failed step"
        );

        drop(stream);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Rejoin and teardown announced in script order (PRD FR-6a, path A)
// ═════════════════════════════════════════════════════════════════════════════

/// Connections the ordering tests rejoin and then tear down.
const ORDERING_REJOINS: usize = 20;

/// Open [`ORDERING_REJOINS`] joins into the session and evict each, so their
/// keepers rejoin concurrently. Then disconnect each as soon as its rejoin
/// register has re-added it, so its teardown's unregister follows closely.
/// Returns the connection ids once every join and keeper has finished.
///
/// For each connection the store ran four membership scripts in a fixed order
/// — the join's register, the eviction, the rejoin register, teardown's
/// unregister — announcing `joined`, `left`, `joined`, `left`. The order is
/// fixed by the handler, not by where the scripts publish: the disconnect
/// follows the observed re-add, and teardown waits for the keeper's step
/// before it unregisters.
async fn rejoin_then_disconnect_each(
    tv: &TestValkey,
    svc: &LiveEditService,
    session_id: &str,
) -> Vec<String> {
    let mut pending = Vec::with_capacity(ORDERING_REJOINS);
    for _ in 0..ORDERING_REJOINS {
        let (mut stream, tx) = open_join(svc, session_id, "ordering-rejoin-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        pending.push((cid, stream, tx));
    }
    for (cid, _, _) in &pending {
        evict(tv, session_id, cid).await;
    }

    let mut cids = Vec::with_capacity(ORDERING_REJOINS);
    let deadline = Instant::now() + SETTLE;
    while !pending.is_empty() {
        let current = members(tv, session_id).await;
        let (rejoined, waiting): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|(cid, _, _)| current.contains(cid));
        for (cid, stream, tx) in rejoined {
            drop((stream, tx));
            cids.push(cid);
        }
        pending = waiting;
        assert!(
            pending.is_empty() || Instant::now() < deadline,
            "{} of {ORDERING_REJOINS} evicted connections were not rejoined within {SETTLE:?}",
            pending.len()
        );
        tokio::time::sleep(POLL).await;
    }
    drain_tracker(svc).await;
    cids
}

/// Each of `cids` was announced `joined`, `left`, `joined`, `left` in `events`:
/// the order the store ran its register, eviction, rejoin register and
/// teardown unregister.
fn assert_announced_in_script_order(events: &[PresenceEvent], cids: &[String], what: &str) {
    for cid in cids {
        let kinds: Vec<&str> = membership_events_of(events, cid)
            .iter()
            .map(|e| if is_joined(e, cid) { "joined" } else { "left" })
            .collect();
        assert_eq!(
            kinds,
            ["joined", "left", "joined", "left"],
            "{what}: {cid} must be announced in the order its membership scripts ran"
        );
    }
}

/// With [`ORDERING_REJOINS`] keepers rejoining concurrently on a standalone
/// store, each connection is announced exactly `joined`, `left`, `joined`,
/// `left` (its join, its eviction, its rejoin, its teardown), and every rejoin
/// is counted once. On one node a subscriber sees publishes in execution order,
/// so this pins the per-connection sequence and the count, not the cross-node
/// delivery order; the cluster variant below proves that.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_and_teardown_are_announced_in_script_order() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let (session_id, _token) = create_session(&tv, &svc, "rejoin-ordering", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        let cids = rejoin_then_disconnect_each(&tv, &svc, &session_id).await;
        let events = tap.drain(&tv).await;
        assert_announced_in_script_order(&events, &cids, "standalone");
        assert_eq!(
            rejoins(&svc),
            ORDERING_REJOINS as f64,
            "every eviction was followed by one counted rejoin"
        );
        tap.close().await;
    })
    .await;
}

/// On a multi-shard cluster, a peer subscribed on a node that owns neither the
/// session's `{sid}` slot nor the channel name's slot sees every rejoined
/// connection's rejoin `PeerJoined` before its teardown's `PeerLeft`, each
/// exactly once, and every rejoin is counted once. Both scripts publish from
/// the `{sid}` node; the session is chosen so the channel's node differs,
/// which is what would let an unpinned publish overtake.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejoin_and_teardown_on_a_cluster_are_announced_in_script_order_on_another_node() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    scenario(async {
        let tv = TestValkey::setup_with_url(cluster_url, 4).await;
        let svc = service(&tv, LEASE_TTL_SECS, SHORT_REFRESH_SECS);
        let mut observer = observe_a_split_slot_session(&tv, &svc, "rejoin-ordering").await;
        let session_id = observer.session_id.clone();

        let cids = rejoin_then_disconnect_each(&tv, &svc, &session_id).await;

        // Every eviction's and teardown's PeerLeft has been published by now
        // (each join ended after its teardown); wait for all of them.
        let events = observer.until_lefts(2 * ORDERING_REJOINS).await;
        assert_announced_in_script_order(&events, &cids, "cluster");
        assert_eq!(
            rejoins(&svc),
            ORDERING_REJOINS as f64,
            "every eviction was followed by one counted rejoin"
        );

        observer.close().await;
    })
    .await;
}
