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

// A JoinSession stream whose presence receiver lags past a participant's join
// or leave is ended for a resync (SDD §13.2): teardown runs — the member is
// unregistered and peers see its `PeerLeft` — then the stream ends
// `UNAVAILABLE` with `presence_resync_status`, so the client rejoins with a
// fresh participant list. The lost events count on
// `live_edit_presence_broadcast_dropped_total` and the ending on
// `live_edit_presence_resyncs_total`. A lag that lost only peer messages
// continues instead (`presence_events_metric_test`,
// `presence_replay_dedup_test`), and so does one that lost only this
// connection's own join, which its stream discards at the self-filter anyway
// (last test below).
//
// ── How the lag is forced, deterministically ─────────────────────────────────
//
// Without the pub/sub feed, so events reach the join only through
// `presence_router.broadcast` and its own `joined` never arrives unless a
// test injects it. The client reads nothing after the point named in each
// test, so the join can take at most `LOOP_HOLDS_AT_MOST` events off its
// receiver — the response channel's 32 delivery slots, the one it holds
// while waiting for a slot, and the rest of that one's read-ahead batch —
// before the test starts reading. Every event broadcast past that, beyond
// the router's capacity, is lost to the receiver whatever the scheduling, so
// the lost events, the join or leave among them, are fixed by the order of
// the broadcasts alone.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every read is bounded by `PRESENCE_WAIT` (10 s), the healthy path returning
// in milliseconds; each test body runs under `SCENARIO_BUDGET` (100 s).

use std::sync::Arc;

use live_edit::grpc::presence::presence_resync_status;
use live_edit::grpc::proto::join_session_response::ResponseType;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::presence::{LocalPresenceRouter, PresenceEvent, RECV_BATCH};
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

/// The most events a join's loop takes off its receiver while the client
/// reads nothing: the response channel's [`RESPONSE_DELIVERY_SLOTS`], the
/// event the loop holds while it waits for a slot, and the rest of that
/// event's read-ahead batch, up to `RECV_BATCH - 1` more (40 in all). Derived
/// from the shared constants, so a change to the channel's depth or the batch
/// reaches the lost-leave test's `CAPACITY > LOOP_HOLDS_AT_MOST` check.
const LOOP_HOLDS_AT_MOST: usize = RESPONSE_DELIVERY_SLOTS + RECV_BATCH;

/// A service over `tv` whose local router holds `capacity` events per
/// session.
fn service_with_capacity(tv: &TestValkey, capacity: usize) -> LiveEditService {
    LiveEditService {
        presence_router: Arc::new(LocalPresenceRouter::with_capacity(capacity)),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    }
}

/// Broadcast `event` straight into `svc`'s router, as the pub/sub feed would
/// deliver it; the join's receiver must take it.
fn inject(svc: &LiveEditService, session_id: &str, event: PresenceEvent) {
    assert!(
        svc.presence_router.broadcast(session_id, event) >= 1,
        "the join's receiver must be on the session's router entry"
    );
}

fn message(seq: usize) -> PresenceEvent {
    PresenceEvent::PeerMessage {
        connection_id: "remote-peer".to_owned(),
        data: (seq as u64).to_be_bytes().to_vec(),
    }
}

fn metric(svc: &LiveEditService, name: &str) -> u64 {
    sample_value(
        &render(&svc.metrics),
        name,
        &[("service", svc.service_label.as_str())],
    )
    .unwrap_or_else(|| panic!("{name} is seeded in Metrics::new")) as u64
}

fn dropped(svc: &LiveEditService) -> u64 {
    metric(svc, "live_edit_presence_broadcast_dropped_total")
}

fn resyncs(svc: &LiveEditService) -> u64 {
    metric(svc, "live_edit_presence_resyncs_total")
}

/// The status must be the resync ending: `UNAVAILABLE`, with its message.
fn assert_resync_status(status: Option<tonic::Status>, what: &str) {
    let status = status.unwrap_or_else(|| {
        panic!("{what}: the stream must end with the resync status, but it ended OK")
    });
    let expected = presence_resync_status();
    assert_eq!(
        (status.code(), status.message()),
        (expected.code(), expected.message()),
        "{what}: the stream must end UNAVAILABLE for a resync"
    );
}

/// The regression gate. A lag that lost a `PeerLeft` ends the stream
/// `UNAVAILABLE` for a resync, after teardown: the member is gone from the
/// store when the status arrives, and a peer on another replica sees its
/// `PeerLeft`. The resync is counted once, and the drop counter moves by
/// exactly the events the receiver lost.
///
/// `CAPACITY` (64) messages are broadcast — the loop can take at most
/// [`LOOP_HOLDS_AT_MOST`] of them — then a `PeerLeft` for a remote peer, then
/// `CAPACITY` more messages, which push the `PeerLeft` out of the queue. The
/// loop's next receive after the test starts reading reports the lag. Before
/// the fix the loop counted the lag and went on, delivering the queued
/// messages, and the stream never ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lag_that_loses_a_leave_ends_the_stream_unavailable_after_teardown() {
    const CAPACITY: usize = 64;
    const _: () = assert!(CAPACITY > LOOP_HOLDS_AT_MOST);
    const REMOTE: &str = "remote-peer-that-left";
    scenario(async {
        let what = "a lag that lost a PeerLeft";
        let tv = TestValkey::setup().await;
        let svc = service_with_capacity(&tv, CAPACITY);
        let (session_id, _token) = create_session(&tv, &svc, "resync-lost-leave", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, _client) = open_join(&svc, &session_id, "resync-app").await;
        let cid = expect_session_connected(&mut stream, PRESENCE_WAIT).await;

        for seq in 0..CAPACITY {
            inject(&svc, &session_id, message(seq));
        }
        inject(
            &svc,
            &session_id,
            PresenceEvent::PeerLeft {
                connection_id: REMOTE.to_owned(),
            },
        );
        for seq in CAPACITY..2 * CAPACITY {
            inject(&svc, &session_id, message(seq));
        }
        let (delivered, status) = read_to_end(&mut stream, what).await;

        assert_resync_status(status, what);
        // What the loop took before the test read, the batch it was reading
        // included (taken events are never lost to a lag): the first
        // messages, in order, and never the leave.
        let received = delivered.len();
        assert!(
            received <= LOOP_HOLDS_AT_MOST,
            "{what}: at most {LOOP_HOLDS_AT_MOST} events delivered, got {received}"
        );
        for (seq, msg) in delivered.into_iter().enumerate() {
            match msg.response_type {
                Some(ResponseType::PeerMessage(m)) => assert_eq!(
                    m.data,
                    (seq as u64).to_be_bytes().to_vec(),
                    "{what}: message {seq} delivered in order"
                ),
                other => panic!("{what}: expected only messages before the lag, got {other:?}"),
            }
        }
        // Everything up to and including the leave that was not delivered
        // was lost, in one lag; the newest `CAPACITY` stay queued, unread.
        assert_eq!(
            dropped(&svc),
            (CAPACITY + 1 - received) as u64,
            "{what}: the drop counter moves by the events the receiver lost"
        );
        assert_eq!(resyncs(&svc), 1, "{what}: one stream ended for a resync");
        assert!(
            !is_member(&tv, &session_id, &cid).await,
            "{what}: teardown unregisters the member before the status is sent"
        );
        tap.until("the resynced connection's PeerLeft", |event| {
            is_left(event, &cid)
        })
        .await;

        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

/// A join that is lost while the membership replay still waits on a slow
/// client ends the stream for a resync at the loop's first receive, after a
/// join that otherwise succeeded: the receiver is attached at SR-4 step 3 but
/// first read only once the replay is done. The client gets
/// `SessionConnected` and the whole replay, then the resync status.
///
/// Forced with a replay that cannot finish while the client reads nothing:
/// `ONE_PAGE_OVERFILLING_THE_CHANNEL` (40) seeded members, more than the
/// response channel's 32 delivery slots. With the replay waiting, a
/// `PeerJoined` and then two messages are broadcast into a capacity-2 router,
/// so the join's receiver loses the `PeerJoined` and nothing else.
///
/// Seeding the members needs `--features test-fault-injection`, so the test
/// compiles only with it: a run without that feature reports green without
/// this test.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_lost_while_the_replay_waits_ends_the_stream_at_the_loops_first_receive() {
    const CAPACITY: usize = 2;
    const LATE_PEER: &str = "late-peer";
    scenario(async {
        let what = "a join lost during a slow replay";
        let tv = TestValkey::setup().await;
        let svc = service_with_capacity(&tv, CAPACITY);
        let (session_id, _token) = create_session(&tv, &svc, "resync-late-first-read", true).await;
        let members = seeded_cids(ONE_PAGE_OVERFILLING_THE_CHANNEL);
        seed_presence_members(&tv, &session_id, &members).await;
        let (mut stream, _client) = open_join(&svc, &session_id, "resync-app").await;
        // The receiver is attached before `SessionConnected` is sent.
        expect_session_connected(&mut stream, PRESENCE_WAIT).await;

        inject(
            &svc,
            &session_id,
            PresenceEvent::PeerJoined(live_edit::presence::ConnectedUser {
                connection_id: LATE_PEER.to_owned(),
                user_name: "Late Peer".to_owned(),
                user_application: "late-app".to_owned(),
            }),
        );
        inject(&svc, &session_id, message(0));
        inject(&svc, &session_id, message(1));
        let (delivered, status) = read_to_end(&mut stream, what).await;

        let replayed: Vec<String> = delivered
            .into_iter()
            .map(|msg| match msg.response_type {
                Some(ResponseType::PeerJoined(p)) => p.connection_id,
                other => panic!("{what}: expected only the replay's PeerJoined, got {other:?}"),
            })
            .collect();
        let mut sorted = replayed.clone();
        sorted.sort();
        assert_eq!(
            sorted, members,
            "{what}: the join completes its whole replay before the lag is seen"
        );
        assert_resync_status(status, what);
        assert_eq!(
            dropped(&svc),
            1,
            "{what}: only the PeerJoined was lost; the two messages stayed queued"
        );
        assert_eq!(resyncs(&svc), 1, "{what}: one stream ended for a resync");

        drain_tracker(&svc).await;
    })
    .await;
}

/// A join whose first read lags past only its own `joined` (and peer
/// messages) goes on: the stream discards its own events at the self-filter,
/// so losing one loses nothing the client would have seen. The lag is
/// counted on the drop counter, no resync is, and the messages that
/// survived are delivered.
///
/// Forced as in the test above: the replay of
/// `ONE_PAGE_OVERFILLING_THE_CHANNEL` (40) seeded members waits on the
/// client, so the join's receiver is not read. Meanwhile the join's own
/// `PeerJoined` and then two messages are broadcast into a capacity-2 router,
/// so the receiver loses its own `PeerJoined` and nothing else. This is the
/// shape a busy session gives every joiner: its own `joined` reaches its
/// receiver just after it subscribes, while its replay is still sending.
///
/// Needs `--features test-fault-injection` to seed the members; without it
/// the test is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_first_read_lag_over_only_its_own_joined_continues_the_stream() {
    use futures::StreamExt;

    const CAPACITY: usize = 2;
    scenario(async {
        let what = "a first-read lag over only the join's own joined";
        let tv = TestValkey::setup().await;
        let svc = service_with_capacity(&tv, CAPACITY);
        let (session_id, _token) = create_session(&tv, &svc, "resync-own-joined", true).await;
        let members = seeded_cids(ONE_PAGE_OVERFILLING_THE_CHANNEL);
        seed_presence_members(&tv, &session_id, &members).await;
        let (mut stream, client) = open_join(&svc, &session_id, "resync-app").await;
        let cid = expect_session_connected(&mut stream, PRESENCE_WAIT).await;

        inject(
            &svc,
            &session_id,
            PresenceEvent::PeerJoined(live_edit::presence::ConnectedUser {
                connection_id: cid.clone(),
                user_name: "Test User".to_owned(),
                user_application: "resync-app".to_owned(),
            }),
        );
        inject(&svc, &session_id, message(0));
        inject(&svc, &session_id, message(1));

        // The whole replay, then both messages, with the stream still open.
        let (replayed, messages) = tokio::time::timeout(PRESENCE_WAIT, async {
            let (mut replayed, mut messages) = (Vec::new(), Vec::new());
            while messages.len() < 2 {
                let msg = stream
                    .next()
                    .await
                    .unwrap_or_else(|| panic!("{what}: the stream ended"))
                    .unwrap_or_else(|s| panic!("{what}: the stream failed: {s:?}"));
                match msg.response_type {
                    Some(ResponseType::PeerJoined(p)) => replayed.push(p.connection_id),
                    Some(ResponseType::PeerMessage(m)) => messages.push(m.data),
                    other => panic!("{what}: expected the replay and messages, got {other:?}"),
                }
            }
            (replayed, messages)
        })
        .await
        .unwrap_or_else(|_| panic!("{what}: the replay and messages within {PRESENCE_WAIT:?}"));

        let mut sorted = replayed;
        sorted.sort();
        assert_eq!(sorted, members, "{what}: the whole replay is delivered");
        assert_eq!(
            messages,
            vec![0u64.to_be_bytes().to_vec(), 1u64.to_be_bytes().to_vec()],
            "{what}: the stream went on past the lag and delivered both messages"
        );
        assert_eq!(dropped(&svc), 1, "{what}: only the own PeerJoined was lost");
        assert_eq!(
            resyncs(&svc),
            0,
            "{what}: losing its own joined is no resync"
        );

        drop(client);
        read_to_ok_end(&mut stream, what).await;
        drain_tracker(&svc).await;
    })
    .await;
}
