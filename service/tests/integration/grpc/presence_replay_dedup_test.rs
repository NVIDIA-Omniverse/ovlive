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

// Exactly one `PeerJoined` per member across a JoinSession's membership replay
// and its live events (SDD SR-4.4; PRD FR-3).
//
// A join subscribes to its session's presence events (SR-4 step 3) before it
// takes the membership snapshot it replays (step 4, before its register). A
// peer that registers in between is in the snapshot and also arrives live.
// The join's bidi loop suppresses that live duplicate with the set D of ids
// its replay delivered: a live `PeerJoined` for an id in D is dropped and the
// id leaves D; a live `PeerLeft` takes its id out of D and is forwarded. D
// ends, never to be re-created, at the first of the join's own first live
// event, a receiver lag, `REPLAY_DUPLICATE_WINDOW_LIMIT` after the loop
// starts (only for a D that held ids then), or the loop's end.
//
// Every test drives `handle_join_session` against a real Valkey and checks the
// joiner's own stream and the replay seam's observation of its D (size, live
// duplicates suppressed, why it ended).
//
// ── How a stream is proven complete ──────────────────────────────────────────
//
// The test publishes a sentinel `PeerJoined` (`publish_probe`) once every event
// under test has been published, and reads the joiner's stream up to it. A
// session's events leave one node in the order it ran them and reach every
// subscriber in that order, so everything published before the sentinel —
// the join's own `joined` included — has reached the join's loop by then. The
// tests that run without the router feed inject events with
// `presence_router.broadcast` instead, sentinel included, in the same order.
//
// ── Seams ────────────────────────────────────────────────────────────────────
//
// The subscribe-to-snapshot window is held open at `PresenceReplaySeam::
// before_snapshot` (after the clock read, before the `SMEMBERS`), and the
// moment after the snapshot at `JoinRegistrationSeam::before_register`. The
// time-limit tests shorten the limit with `set_duplicate_window_limit`. The
// whole file needs `--features test-fault-injection`.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every wait is a seam park, a seam observation, or a stream read, each
// bounded by `SETTLE` (10 s); the healthy path returns from each in
// milliseconds. Two waits are deliberately long: the time-limit tests run for
// one shortened limit (`SHORT_WINDOW_LIMIT`, 1 s), and the empty-D test lets
// its loop run past several limits on purpose before checking that none ended
// it. Every test body runs under `SCENARIO_BUDGET` (100 s).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;

use live_edit::grpc::presence::{
    DeliveredSetEnd, DeliveredSetObservation, JoinRegistrationSeam, ObservedReplayResult,
    PresenceReplaySeam, REPLAY_DUPLICATE_WINDOW_LIMIT,
};
use live_edit::grpc::proto::{join_session_response, JoinSessionResponse};
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::PUBSUB_RELAY_PRESENCE;
use live_edit::presence::{ConnectedUser, LocalPresenceRouter, PresenceEvent, PresenceRecvError};
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on any single wait for a seam, an observation, or a stream read.
const SETTLE: Duration = Duration::from_secs(10);

/// A refresh interval under which no lease keeper step runs inside a test
/// (its first renewal comes at least half an interval after its join).
const QUIET_REFRESH_SECS: u64 = 3600;

/// The connection id of the sentinel `PeerJoined` a stream is read up to.
const SENTINEL: &str = "sentinel";

/// D's time limit in the time-limit tests: long enough that an event queued
/// before the loop starts is handled in its first iteration, well inside it.
const SHORT_WINDOW_LIMIT: Duration = Duration::from_secs(1);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The two seams installed on a [`dedup_service`], every gate disarmed.
struct Seams {
    replay: Arc<PresenceReplaySeam>,
    registration: Arc<JoinRegistrationSeam>,
}

/// A service over `tv` with `router` as its local presence router, the replay
/// and registration seams installed, and no keeper step inside a test.
fn dedup_service(tv: &TestValkey, router: LocalPresenceRouter) -> (LiveEditService, Seams) {
    let seams = Seams {
        replay: Arc::new(PresenceReplaySeam::new()),
        registration: Arc::new(JoinRegistrationSeam::new()),
    };
    let svc = LiveEditService {
        presence_refresh_interval_secs: QUIET_REFRESH_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_router: Arc::new(router),
        presence_replay_seam_for_test: Some(Arc::clone(&seams.replay)),
        join_registration_seam_for_test: Some(Arc::clone(&seams.registration)),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    };
    (svc, seams)
}

/// `live_edit_presence_broadcast_dropped_total` on `svc`'s metrics.
fn dropped(svc: &LiveEditService) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_presence_broadcast_dropped_total",
        &[("service", svc.service_label.as_str())],
    )
    .expect("the drop counter is seeded in Metrics::new") as u64
}

// ── Events ───────────────────────────────────────────────────────────────────

/// A presence message a joiner's stream carried after `SessionConnected`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Seen {
    Joined(String),
    Left(String),
    Message(String),
}

fn joined(cid: &str) -> Seen {
    Seen::Joined(cid.to_owned())
}

fn left(cid: &str) -> Seen {
    Seen::Left(cid.to_owned())
}

fn seen_of(msg: JoinSessionResponse, what: &str) -> Seen {
    match msg.response_type {
        Some(join_session_response::ResponseType::PeerJoined(p)) => Seen::Joined(p.connection_id),
        Some(join_session_response::ResponseType::PeerLeft(p)) => Seen::Left(p.connection_id),
        Some(join_session_response::ResponseType::PeerMessage(m)) => Seen::Message(m.connection_id),
        other => panic!("{what}: expected a presence message; got {other:?}"),
    }
}

/// The stream's next message, which must be a presence message.
async fn read_one(stream: &mut JoinResponseStream, what: &str) -> Seen {
    let msg = tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what}: nothing arrived within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("{what}: the join stream ended"))
        .unwrap_or_else(|s| panic!("{what}: the join stream failed: {s:?}"));
    seen_of(msg, what)
}

/// Every presence message before the [`SENTINEL`]'s `PeerJoined`, in order,
/// under one [`SETTLE`] deadline.
async fn read_until_sentinel(stream: &mut JoinResponseStream, what: &str) -> Vec<Seen> {
    tokio::time::timeout(SETTLE, async {
        let mut seen = Vec::new();
        loop {
            match read_one(stream, what).await {
                Seen::Joined(cid) if cid == SENTINEL => return seen,
                other => seen.push(other),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the sentinel did not arrive within {SETTLE:?}"))
}

/// The `PeerJoined` and `PeerLeft` for `cid` among `seen`, in order.
fn membership_of(seen: &[Seen], cid: &str) -> Vec<Seen> {
    seen.iter()
        .filter(|s| matches!(s, Seen::Joined(c) | Seen::Left(c) if c == cid))
        .cloned()
        .collect()
}

fn peer_joined_event(cid: &str) -> PresenceEvent {
    PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: cid.to_owned(),
        user_name: "Injected".to_owned(),
        user_application: "injected-app".to_owned(),
    })
}

/// Broadcast `event` straight into `svc`'s local router, as the pub/sub feed
/// would deliver it; at least the joiner's receiver must take it.
fn inject(svc: &LiveEditService, session_id: &str, event: PresenceEvent) {
    let receivers = svc.presence_router.broadcast(session_id, event);
    assert!(
        receivers >= 1,
        "the joiner's receiver must be on the session's router entry"
    );
}

// ── The replay seam's observation of D ───────────────────────────────────────

/// Wait for `cid`'s D to have ended, and return its observation.
async fn delivered_set_ended(
    seam: &PresenceReplaySeam,
    cid: &str,
    what: &str,
) -> DeliveredSetObservation {
    tokio::time::timeout(
        SETTLE,
        seam.wait_for_delivered_set(cid, |d| d.ended_by.is_some()),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "{what}: the join's replay-delivered set did not end within {SETTLE:?}; last seen \
             {:?}",
            seam.delivered_set(cid)
        )
    })
}

/// The observation of a D that ended, by `ended_by`, after delivering
/// `created_len` and suppressing `suppressed`.
fn ended(
    created_len: usize,
    suppressed: usize,
    ended_by: DeliveredSetEnd,
) -> DeliveredSetObservation {
    DeliveredSetObservation {
        created_len,
        len: None,
        suppressed_peer_joined: suppressed,
        ended_by: Some(ended_by),
    }
}

/// Close the client side, read the stream to its `OK` end, and wait for the
/// join task to finish.
async fn leave(
    svc: &LiveEditService,
    stream: &mut JoinResponseStream,
    tx: tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
    what: &str,
) {
    drop(tx);
    read_to_ok_end(stream, what).await;
    drain_tracker(svc).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The window and the boundary (T1–T3)
// ═════════════════════════════════════════════════════════════════════════════

/// The regression gate. A peer registers while the join is parked inside its
/// subscribe-to-snapshot window (`before_snapshot`), so the peer is both in
/// the snapshot the replay delivers and, live, in the join's receiver, ahead
/// of the join's own `joined`. The joiner's stream carries exactly one
/// `PeerJoined` for it; the replay seam shows the live copy suppressed, and D
/// ended by the join's own event. Without suppression the stream carries two.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peer_registering_inside_the_subscribe_to_snapshot_window_reaches_the_joiner_once() {
    const PEER: &str = "window-peer";
    scenario(async {
        let what = "a peer registered inside the window";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-window", true).await;

        let gate = &seams.replay.before_snapshot;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-window-app").await;
        wait_parked(gate, "the join, inside its subscribe-to-snapshot window").await;
        register_remote_peer(&tv, &session_id, PEER).await;
        gate.release();

        let cid = expect_session_connected(&mut stream, SETTLE).await;
        publish_probe(&tv, &session_id, SENTINEL).await;
        let seen = read_until_sentinel(&mut stream, what).await;

        assert_eq!(
            seen,
            vec![joined(PEER)],
            "{what}: the peer reaches the joiner exactly once, and nothing else arrives"
        );
        assert_eq!(seams.replay.result(), Some(ObservedReplayResult::Complete));
        assert_eq!(
            seams.replay.peer_joined_sent_count(),
            1,
            "{what}: the replay delivered the peer, which registered before the snapshot"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(1, 1, DeliveredSetEnd::OwnEvent)),
            "{what}: the peer's live PeerJoined was suppressed as a duplicate, and D ended at \
             the join's own event"
        );

        leave(&svc, &mut stream, tx, what).await;
        feed.cancel();
    })
    .await;
}

/// A peer that registers after the join's snapshot — parked at
/// `before_register`, which now follows the snapshot — is not in the replay
/// and reaches the joiner exactly once, live, after the replay: it is not in D,
/// so nothing is suppressed. A member present before the join is replayed. Were
/// the snapshot taken after the register, the late peer would be replayed and
/// its live copy suppressed instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peer_registering_after_the_snapshot_is_delivered_once_live_and_not_suppressed() {
    const EARLIER: &str = "earlier-peer";
    const LATE: &str = "late-peer";
    scenario(async {
        let what = "a peer registered after the snapshot";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-after-snapshot", true).await;
        register_remote_peer(&tv, &session_id, EARLIER).await;

        let gate = &seams.registration.before_register;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-after-snapshot-app").await;
        wait_parked(gate, "the join, after its snapshot and before its register").await;
        register_remote_peer(&tv, &session_id, LATE).await;
        gate.release();

        let cid = expect_session_connected(&mut stream, SETTLE).await;
        publish_probe(&tv, &session_id, SENTINEL).await;
        let seen = read_until_sentinel(&mut stream, what).await;

        assert_eq!(
            seams.replay.peer_joined_sent_count(),
            1,
            "{what}: the replay delivers only the member present at the snapshot, not {LATE}"
        );
        assert_eq!(
            seen,
            vec![joined(EARLIER), joined(LATE)],
            "{what}: the replayed member, then the late peer once, live"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(1, 0, DeliveredSetEnd::OwnEvent)),
            "{what}: the late peer is not in D, so nothing is suppressed"
        );

        leave(&svc, &mut stream, tx, what).await;
        feed.cancel();
    })
    .await;
}

/// D ends at the join's own first live event. After that, a second `joined`
/// for a replayed member with no `left` before it — published after the
/// snapshot, so not one the replay covers — is forwarded, not suppressed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replay_duplicate_suppression_ends_at_the_joins_own_event() {
    const PEER: &str = "replayed-peer";
    scenario(async {
        let what = "a joined after the join's own event";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-boundary", true).await;
        register_remote_peer(&tv, &session_id, PEER).await;

        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-boundary-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        assert_eq!(
            delivered_set_ended(&seams.replay, &cid, what).await,
            ended(1, 0, DeliveredSetEnd::OwnEvent),
            "{what}: D ends at the join's own event"
        );

        publish_probe(&tv, &session_id, PEER).await;
        publish_probe(&tv, &session_id, SENTINEL).await;
        let seen = read_until_sentinel(&mut stream, what).await;
        assert_eq!(
            seen,
            vec![joined(PEER), joined(PEER)],
            "{what}: the replay's PeerJoined, then the later live one, forwarded"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(1, 0, DeliveredSetEnd::OwnEvent)),
            "{what}: an ended D suppresses nothing"
        );

        leave(&svc, &mut stream, tx, what).await;
        feed.cancel();
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Leave and rejoin are never suppressed (T4)
// ═════════════════════════════════════════════════════════════════════════════

/// A replayed member that leaves (an in-script `left`, from the test-only
/// atomic eviction) and registers again after the join's boundary reaches the
/// joiner as the replay's `PeerJoined`, then `PeerLeft`, then `PeerJoined`, and
/// is present at the end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_member_leaving_and_rejoining_after_the_boundary_is_announced_left_then_joined() {
    const PEER: &str = "rejoining-peer";
    scenario(async {
        let what = "a leave and rejoin after the boundary";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-rejoin-after", true).await;
        register_remote_peer(&tv, &session_id, PEER).await;

        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-rejoin-after-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        delivered_set_ended(&seams.replay, &cid, what).await;

        evict(&tv, &session_id, PEER).await;
        register_remote_peer(&tv, &session_id, PEER).await;
        publish_probe(&tv, &session_id, SENTINEL).await;
        let seen = read_until_sentinel(&mut stream, what).await;

        assert_eq!(
            membership_of(&seen, PEER),
            vec![joined(PEER), left(PEER), joined(PEER)],
            "{what}: replayed, left, joined again; got {seen:?}"
        );
        assert!(
            is_member(&tv, &session_id, PEER).await,
            "{what}: the peer ends present"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(1, 0, DeliveredSetEnd::OwnEvent))
        );

        leave(&svc, &mut stream, tx, what).await;
        feed.cancel();
    })
    .await;
}

/// The same inside the window: the member `peer` leaves and registers again
/// while the join is parked at `before_snapshot`, so it is a member at the
/// snapshot and its live `left` and `joined` both arrive while D still holds
/// it. The `left` takes it out of D first, so the rejoin is forwarded: the
/// joiner sees the replay's `PeerJoined`, then `PeerLeft`, then `PeerJoined`,
/// ending with the member present, and nothing is suppressed.
///
/// D keys a canonical UUID id (the form every connection id this service mints
/// takes) by its 128-bit value and any other id by its string, so the test runs
/// once with each kind of id.
async fn leave_and_rejoin_inside_the_window_is_announced_joined_left_joined(peer: &str) {
    let what = format!("a leave and rejoin of {peer} inside the window");
    let what = what.as_str();
    let tv = TestValkey::setup().await;
    let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
    let feed = spawn_router_feed(&tv, &svc).await;
    let (session_id, _token) = create_session(&tv, &svc, "dedup-rejoin-inside", true).await;
    register_remote_peer(&tv, &session_id, peer).await;

    let gate = &seams.replay.before_snapshot;
    gate.arm();
    let (mut stream, tx) = open_join(&svc, &session_id, "dedup-rejoin-inside-app").await;
    wait_parked(gate, "the join, inside its subscribe-to-snapshot window").await;
    evict(&tv, &session_id, peer).await;
    register_remote_peer(&tv, &session_id, peer).await;
    gate.release();

    let cid = expect_session_connected(&mut stream, SETTLE).await;
    publish_probe(&tv, &session_id, SENTINEL).await;
    let seen = read_until_sentinel(&mut stream, what).await;

    assert_eq!(
        seen,
        vec![joined(peer), left(peer), joined(peer)],
        "{what}: the replay's PeerJoined, then the live leave and the live rejoin"
    );
    assert_eq!(
        seams.replay.peer_joined_sent_count(),
        1,
        "{what}: the member was present at the snapshot"
    );
    assert_eq!(
        seams.replay.delivered_set(&cid),
        Some(ended(1, 0, DeliveredSetEnd::OwnEvent)),
        "{what}: the live left took the member out of D, so its rejoin was not suppressed"
    );
    assert!(
        is_member(&tv, &session_id, peer).await,
        "{what}: the peer ends present"
    );

    leave(&svc, &mut stream, tx, what).await;
    feed.cancel();
}

/// [`leave_and_rejoin_inside_the_window_is_announced_joined_left_joined`] for a
/// canonical UUID id, the form of every connection id the service mints: D
/// holds it in its 128-bit set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_member_with_a_uuid_id_leaving_and_rejoining_inside_the_window_is_announced_joined_left_joined(
) {
    scenario(
        leave_and_rejoin_inside_the_window_is_announced_joined_left_joined(
            "4f1c2a9e-7b3d-4c8e-9a51-2d6f0b8e3c17",
        ),
    )
    .await;
}

/// [`leave_and_rejoin_inside_the_window_is_announced_joined_left_joined`] for an
/// id that is not a canonical UUID: D holds it in its string set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_member_with_a_string_id_leaving_and_rejoining_inside_the_window_is_announced_joined_left_joined(
) {
    scenario(leave_and_rejoin_inside_the_window_is_announced_joined_left_joined("rejoining-peer"))
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A lag ends suppression (T5)
// ═════════════════════════════════════════════════════════════════════════════

/// Every receiver lag ends D, by choice: D is not kept across a lag even
/// when the lost events were all messages. (A lag that lost another
/// connection's join or leave ends the stream for a resync instead, which
/// `presence_resync_test` covers.) The service's router holds `CAPACITY` (4)
/// events per session. While the join is parked inside its window, `PROBES`
/// (8) message probes are published, then a peer registers, and every one of
/// those events is on the join's receiver (a second receiver on the same
/// router entry has seen the peer's `joined`) before the join goes on. So its
/// loop's first receive reports `Lagged`. At most two events follow the
/// peer's `joined` before the loop drains the queue — the join's own `joined`
/// and the sentinel — so it is among the newest four and survives the lag,
/// and every event lost is a message probe, so the stream continues. The peer
/// is in the replay, but D has ended, so its live `PeerJoined` is forwarded:
/// the joiner sees the peer twice. The lag is counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn receiver_lag_ends_suppression_and_the_surviving_duplicate_is_forwarded() {
    const CAPACITY: usize = 4;
    const PROBES: usize = 2 * CAPACITY;
    const PEER: &str = "lag-peer";
    scenario(async {
        let what = "a lag inside the window";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::with_capacity(CAPACITY));
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-lag", true).await;

        let gate = &seams.replay.before_snapshot;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-lag-app").await;
        wait_parked(gate, "the join, inside its subscribe-to-snapshot window").await;

        // On the join's router entry, so once it holds the peer's `joined`,
        // the join's own receiver holds it too, behind the probes.
        let (mut observer, _) = svc.presence_router.subscribe(&session_id);
        for i in 0..PROBES {
            publish_message_probe(&tv, &session_id, &format!("lag-probe-{i}")).await;
        }
        register_remote_peer(&tv, &session_id, PEER).await;
        tokio::time::timeout(SETTLE, async {
            loop {
                match observer.recv().await.as_deref() {
                    Ok(PresenceEvent::PeerJoined(user)) if user.connection_id == PEER => return,
                    Ok(_) | Err(PresenceRecvError::Lagged { .. }) => {}
                    Err(PresenceRecvError::Closed) => {
                        panic!("{what}: the session's router entry closed")
                    }
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("{what}: the peer's joined did not reach the router within {SETTLE:?}")
        });
        assert_eq!(dropped(&svc), 0, "{what}: precondition: no lag counted yet");
        gate.release();

        let cid = expect_session_connected(&mut stream, SETTLE).await;
        publish_probe(&tv, &session_id, SENTINEL).await;
        let seen = read_until_sentinel(&mut stream, what).await;

        assert_eq!(
            membership_of(&seen, PEER),
            vec![joined(PEER), joined(PEER)],
            "{what}: the replay's PeerJoined, then the live one that survived the lag, \
             forwarded because the lag ended suppression; got {seen:?}"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(1, 0, DeliveredSetEnd::Lag)),
            "{what}: D ended at the lag, before any duplicate could be suppressed"
        );
        assert!(
            dropped(&svc) > 0,
            "{what}: the join's lag must be counted on the drop counter"
        );

        assert!(
            !svc.presence_router.unsubscribe(&session_id),
            "the join is still subscribed, so the observer is not the last local receiver"
        );
        drop(observer);
        leave(&svc, &mut stream, tx, what).await;
        feed.cancel();
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The time limit (T6)
// ═════════════════════════════════════════════════════════════════════════════
//
// These run without the router feed, so the join's own `joined` never reaches
// it: D can end only by the time limit, a lag, or the loop's end. Events reach
// the join only through `inject`.

/// The time limit is the 60 s the policy decision fixed, and a test override
/// shortens it. With the join's own event never arriving, a `joined` for a
/// replayed member injected before the loop starts is suppressed (the loop
/// handles it in its first iteration, well inside the limit); then D ends at
/// the limit, and a further `joined` for that member is forwarded.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn suppression_ends_at_the_time_limit_when_the_joins_own_event_never_arrives() {
    const PEER: &str = "replayed-peer";
    assert_eq!(
        REPLAY_DUPLICATE_WINDOW_LIMIT,
        Duration::from_secs(60),
        "the duplicate window's limit is 60 s"
    );
    scenario(async {
        let what = "a D the join's own event never ends";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        seams.replay.set_duplicate_window_limit(SHORT_WINDOW_LIMIT);
        let (session_id, _token) = create_session(&tv, &svc, "dedup-limit", true).await;
        register_remote_peer(&tv, &session_id, PEER).await;

        let gate = &seams.replay.before_first_send;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-limit-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(gate, "the replay, before its first send").await;
        inject(&svc, &session_id, peer_joined_event(PEER));
        gate.release();

        assert_eq!(
            read_one(&mut stream, what).await,
            joined(PEER),
            "{what}: the replay's PeerJoined"
        );
        assert_eq!(
            delivered_set_ended(&seams.replay, &cid, what).await,
            ended(1, 1, DeliveredSetEnd::TimeLimit),
            "{what}: the injected duplicate was suppressed, then D ended at the time limit"
        );

        inject(&svc, &session_id, peer_joined_event(PEER));
        inject(&svc, &session_id, peer_joined_event(SENTINEL));
        assert_eq!(
            read_until_sentinel(&mut stream, what).await,
            vec![joined(PEER)],
            "{what}: after the limit, a joined for the replayed member is forwarded"
        );

        leave(&svc, &mut stream, tx, what).await;
    })
    .await;
}

/// A D emptied in the loop — here by a live leave of its only id — keeps its
/// timer and still ends at the time limit, which frees its table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivered_set_emptied_by_a_leave_still_ends_at_the_time_limit() {
    const PEER: &str = "leaving-peer";
    scenario(async {
        let what = "a D emptied by a leave";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        seams.replay.set_duplicate_window_limit(SHORT_WINDOW_LIMIT);
        let (session_id, _token) = create_session(&tv, &svc, "dedup-emptied", true).await;
        register_remote_peer(&tv, &session_id, PEER).await;

        let gate = &seams.replay.before_first_send;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-emptied-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(gate, "the replay, before its first send").await;
        inject(
            &svc,
            &session_id,
            PresenceEvent::PeerLeft {
                connection_id: PEER.to_owned(),
            },
        );
        inject(&svc, &session_id, peer_joined_event(SENTINEL));
        gate.release();

        assert_eq!(
            read_until_sentinel(&mut stream, what).await,
            vec![joined(PEER), left(PEER)],
            "{what}: the replay's PeerJoined, then the live leave, forwarded"
        );
        assert_eq!(
            delivered_set_ended(&seams.replay, &cid, what).await,
            ended(1, 0, DeliveredSetEnd::TimeLimit),
            "{what}: the emptied D still ends at the time limit"
        );

        leave(&svc, &mut stream, tx, what).await;
    })
    .await;
}

/// A D that holds nothing when the loop starts — a joiner alone in its
/// session — has no time limit: it never ends by it, however long the loop
/// runs, and ends at the loop's end. The loop is let run for several
/// shortened limits (the lateness under test), then a sentinel is injected and
/// read back. The time-limit arm comes before the event arm in the loop's
/// biased `select!`, so a timer that existed would have ended D before the
/// loop forwarded the sentinel.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_delivered_set_never_ends_at_the_time_limit() {
    const LIMIT: Duration = Duration::from_millis(50);
    scenario(async {
        let what = "an empty D";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        seams.replay.set_duplicate_window_limit(LIMIT);
        let (session_id, _token) = create_session(&tv, &svc, "dedup-empty", true).await;

        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-empty-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        tokio::time::timeout(SETTLE, seams.replay.wait_for_delivered_set(&cid, |_| true))
            .await
            .unwrap_or_else(|_| panic!("{what}: the replay did not complete within {SETTLE:?}"));

        tokio::time::sleep(LIMIT * 10).await;
        inject(&svc, &session_id, peer_joined_event(SENTINEL));
        assert_eq!(
            read_until_sentinel(&mut stream, what).await,
            Vec::<Seen>::new()
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(DeliveredSetObservation {
                created_len: 0,
                len: Some(0),
                suppressed_peer_joined: 0,
                ended_by: None,
            }),
            "{what}: no timer exists for a D that held nothing, so it is still live"
        );

        leave(&svc, &mut stream, tx, what).await;
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(ended(0, 0, DeliveredSetEnd::LoopEnd)),
            "{what}: it ends at the loop's end"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Only delivered ids enter D
// ═════════════════════════════════════════════════════════════════════════════

/// An id enters D only when the replay actually delivers it. A snapshot entry
/// the replay skips — here a member with no info hash — was never shown to the
/// client, so a live `joined` for it is the client's first and must be
/// forwarded, not suppressed (suppressing it would hide a present peer).
///
/// Runs without the router feed, so the join's own `joined` never arrives and
/// D stays live for its 60 s limit, holding the one valid member the replay
/// delivered. The live `joined` for the skipped member is injected before the
/// loop starts (the replay is parked at `before_first_send`), so the loop
/// handles it while D is live; D is checked still live, and still holding the
/// valid member, after the sentinel that follows it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_joined_for_a_snapshot_member_the_replay_skipped_is_forwarded() {
    const VALID: &str = "valid-peer";
    const NO_INFO: &str = "no-info-peer";
    scenario(async {
        use fred::interfaces::SetsInterface;

        let what = "a live joined for a skipped snapshot member";
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let (session_id, _token) = create_session(&tv, &svc, "dedup-skipped", true).await;
        register_remote_peer(&tv, &session_id, VALID).await;
        let added: i64 = store_reader(&tv)
            .sadd(tv.key_schema().presence_members(&session_id), NO_INFO)
            .await
            .expect("adding the member with no info hash must succeed");
        assert_eq!(
            added, 1,
            "precondition: {NO_INFO} is a member with no info hash"
        );

        let gate = &seams.replay.before_first_send;
        gate.arm();
        let (mut stream, tx) = open_join(&svc, &session_id, "dedup-skipped-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        wait_parked(gate, "the replay, before its first send").await;
        inject(&svc, &session_id, peer_joined_event(NO_INFO));
        inject(&svc, &session_id, peer_joined_event(SENTINEL));
        gate.release();

        assert_eq!(
            read_until_sentinel(&mut stream, what).await,
            vec![joined(VALID), joined(NO_INFO)],
            "{what}: the replay delivers only the valid member, and the live joined for the \
             skipped one is forwarded"
        );
        assert_eq!(
            seams.replay.peer_joined_sent_count(),
            1,
            "{what}: the replay skipped {NO_INFO}"
        );
        assert_eq!(
            seams.replay.delivered_set(&cid),
            Some(DeliveredSetObservation {
                created_len: 1,
                len: Some(1),
                suppressed_peer_joined: 0,
                ended_by: None,
            }),
            "{what}: D holds only the delivered member, is still live after the live joined \
             was handled, and suppressed nothing"
        );

        leave(&svc, &mut stream, tx, what).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Concurrent joins: distinct per client (T8)
// ═════════════════════════════════════════════════════════════════════════════

/// [`CLIENTS`] clients join one session at once, the scale of the rc.8 check.
/// Once every join has sent `SessionConnected` (so every register, and its
/// `joined`, has run), a sentinel is published and every client reads its
/// stream up to it. Each client's `PeerJoined` are then exactly the other
/// clients, each once — distinct ids, and as many messages as ids — whether a
/// peer reached it in the replay or live, and every D ended at its join's own
/// event. Preconditions, checked before the per-client counts:
/// - the replica-wide presence relay did not lag
///   (`live_edit_pubsub_relay_lagged_total{relay="presence"}` is 0): a lag of
///   the relay in front of the router, on the subscriber client's pub/sub
///   message channel, loses events for every join and would show as a peer
///   missing from some client. No session-deleted relay runs here, so its cell
///   is not checked;
/// - no join's router receiver lagged
///   (`live_edit_presence_broadcast_dropped_total` is 0), so no lag ended
///   suppression early.
///
/// This is the end-to-end gate for that channel's capacity: at fred's default
/// of 32 messages the relay lags in a burst of this size in roughly one run in
/// six, which fails the relay precondition. The deterministic gate is
/// `pubsub_relay_lag_test`'s capacity test.
///
/// How many live duplicates were suppressed depends on how the joins
/// interleave, so it is reported, not asserted: a zero count only says the
/// window was not hit in this run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_joins_each_see_every_other_client_exactly_once() {
    const CLIENTS: usize = 50;
    scenario(async {
        let tv = TestValkey::setup().await;
        let (svc, seams) = dedup_service(&tv, LocalPresenceRouter::new());
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "dedup-storm", true).await;

        let (svc_ref, sid) = (&svc, session_id.as_str());
        let opened = futures::future::join_all((0..CLIENTS).map(|i| async move {
            let app = format!("storm-app-{i}");
            open_join(svc_ref, sid, &app).await
        }))
        .await;

        // Every stream is read at once, from SessionConnected on, so no join
        // parks on a full response channel.
        let (connected_tx, mut connected_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let mut senders = Vec::with_capacity(CLIENTS);
        let mut readers = Vec::with_capacity(CLIENTS);
        for (i, (mut stream, tx)) in opened.into_iter().enumerate() {
            senders.push(tx);
            let connected_tx = connected_tx.clone();
            readers.push(tokio::spawn(async move {
                let what = format!("storm client {i}");
                let cid = expect_session_connected(&mut stream, SETTLE).await;
                connected_tx
                    .send(cid.clone())
                    .expect("the test holds the receiver");
                let seen = read_until_sentinel(&mut stream, &what).await;
                (cid, seen, stream)
            }));
        }
        drop(connected_tx);
        let mut cids = BTreeSet::new();
        tokio::time::timeout(SETTLE, async {
            while cids.len() < CLIENTS {
                let cid = connected_rx
                    .recv()
                    .await
                    .expect("a storm client's reader ended before SessionConnected");
                assert!(cids.insert(cid), "connection ids are distinct");
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{} of {CLIENTS} storm clients connected within {SETTLE:?}",
                cids.len()
            )
        });

        publish_probe(&tv, &session_id, SENTINEL).await;
        let results = tokio::time::timeout(SETTLE, futures::future::join_all(readers))
            .await
            .unwrap_or_else(|_| {
                panic!("the storm clients did not all read their sentinel within {SETTLE:?}")
            });

        assert_eq!(
            pubsub_relay_lagged(&svc.metrics, &svc.service_label, PUBSUB_RELAY_PRESENCE),
            0,
            "precondition: the replica-wide presence relay never lagged, so no event was lost \
             before the router"
        );
        assert_eq!(
            dropped(&svc),
            0,
            "precondition: no join's receiver lagged, so no D ended early"
        );
        let mut streams = Vec::with_capacity(CLIENTS);
        let (mut replayed, mut suppressed) = (0, 0);
        let mut end_causes: BTreeMap<String, usize> = BTreeMap::new();
        for result in results {
            let (cid, seen, stream) = result.expect("a storm client's reader must not panic");
            streams.push(stream);
            let mut per_peer: BTreeMap<&str, usize> = BTreeMap::new();
            for s in &seen {
                match s {
                    Seen::Joined(peer) => *per_peer.entry(peer.as_str()).or_default() += 1,
                    other => panic!(
                        "client {cid}: only PeerJoined may arrive before the sentinel; got \
                         {other:?} in {seen:?}"
                    ),
                }
            }
            let expected: BTreeMap<&str, usize> = cids
                .iter()
                .filter(|other| **other != cid)
                .map(|other| (other.as_str(), 1))
                .collect();
            assert_eq!(
                per_peer, expected,
                "client {cid}: every other client exactly once, and never itself"
            );
            assert_eq!(
                seen.len(),
                CLIENTS - 1,
                "client {cid}: as many PeerJoined as distinct peers"
            );

            let d = seams
                .replay
                .delivered_set(&cid)
                .unwrap_or_else(|| panic!("client {cid}: its replay completed"));
            *end_causes.entry(format!("{:?}", d.ended_by)).or_default() += 1;
            assert_eq!(
                d.ended_by,
                Some(DeliveredSetEnd::OwnEvent),
                "client {cid}: D ends at the join's own event; got {d:?}"
            );
            replayed += d.created_len;
            suppressed += d.suppressed_peer_joined;
        }
        assert_eq!(
            replayed,
            seams.replay.peer_joined_sent_count(),
            "every replayed PeerJoined is in some join's D"
        );
        let delivered = CLIENTS * (CLIENTS - 1);
        eprintln!(
            "storm of {CLIENTS} joins: {delivered} PeerJoined delivered, {replayed} by replays and \
             {} live; {suppressed} live duplicates suppressed; D end causes {end_causes:?}",
            delivered - replayed
        );

        drop(senders);
        drop(streams);
        drain_tracker(&svc).await;
        feed.cancel();
    })
    .await;
}
