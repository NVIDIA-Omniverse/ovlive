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

// The JoinSession client keepalive (SDD §13.2): a join whose header set
// `client_keepalive` is told the configured interval in `SessionConnected`
// and is ended once its request stream has carried no bytes for three
// intervals, counted from the bidi loop's start. The ending is the normal
// teardown — the member is unregistered and peers see one `PeerLeft` — then
// `UNAVAILABLE` with the retry hint `CLIENT_KEEPALIVE_TIMEOUT`, logged once at
// INFO and counted on `live_edit_presence_keepalive_timeouts_total`. Any bytes
// of the request stream count: a keepalive, a message, or part of a message
// still arriving. A keepalive is never published or delivered to peers. A
// join that did not opt in, or a service whose interval is 0, is never ended
// for silence and is told interval 0.
//
// Every join here is in-process through `handle_join_session` with real
// Valkey, built with the body-progress record (`open_attested_join_streaming`)
// that the production stack attaches, with three exceptions: one opted-in
// join is built bare, without the record, to pin that the handler then leaves
// it unarmed rather than refusing it; the peer in the no-broadcast test is a
// bare join that did not opt in (`open_join`), which needs no record; and the
// last test serves the join through `live_service_server` to pin that the
// production stack attaches the record. Paused time is not usable: the joins'
// Valkey I/O races tokio's auto-advance.
//
// ── Time ─────────────────────────────────────────────────────────────────────
//
// The interval is `INTERVAL_MS` (1 s), so the deadline `DEADLINE` is 3 s. A test
// that a join stays watches it for `OBSERVE` (2 × the deadline): a join wrongly
// held to the deadline would have ended inside that window. The sleeps that
// pace a client's sends and the windows a client stays silent for are the
// behaviour under test, not synchronization. Lower bounds on an ending are
// measured from the client's last send, which is load-robust: the server
// stamps bytes when it reads them, never before they were sent. Upper bounds
// are hang guards only (`END_WITHIN` (13 s) for an ending, `PRESENCE_WAIT`
// (10 s) for any other wait), never speed assertions. Worst cases: each
// multi-thread body runs under `SCENARIO_BUDGET` (100 s), below nextest's 120 s
// hang-killer, whatever its waits sum to; the current-thread log test, which
// is not, sums to 43 s with every wait at its full budget (`PRESENCE_WAIT` for
// `SessionConnected`, `END_WITHIN` for the ending, `PRESENCE_WAIT` each for the
// tap's drain and close). A passing run takes 3–15 s per test.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use tokio::time::Instant;
use tonic::{Code, Status};

use live_edit::error::{refusal_reason, CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY};
use live_edit::grpc::presence::{
    client_keepalive_timeout_status, handle_join_session, CLIENT_KEEPALIVE_DEADLINE_INTERVALS,
};
use live_edit::grpc::proto::join_session_response::ResponseType;
use live_edit::grpc::proto::{
    join_session_request, JoinSessionHeader, JoinSessionKeepalive, JoinSessionRequest,
    SessionConnected,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::presence::PresenceEvent;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
use super::log_capture::{
    assert_common_fields, assert_field, purge_keys, ThreadCapture, RPC_FAILURE_MSG,
};
use crate::common::refusal_detail::assert_refusal_detail;

/// The keepalive interval every armed service here is configured with.
const INTERVAL_MS: u64 = 1_000;

/// The deadline an armed join is held to: three intervals (the count is the
/// client contract, pinned in `grpc::presence`'s unit tests).
const DEADLINE_MS: u64 = INTERVAL_MS * CLIENT_KEEPALIVE_DEADLINE_INTERVALS as u64;
const DEADLINE: Duration = Duration::from_millis(DEADLINE_MS);

/// How long a test watches a join it expects to stay: two deadlines.
const OBSERVE: Duration = Duration::from_millis(DEADLINE_MS * 2);

/// A client's keepalive cadence here: half the interval, so a send delayed
/// by a loaded runner still lands well inside every deadline.
const SEND_EVERY: Duration = Duration::from_millis(INTERVAL_MS / 2);

/// Hang guard on an ending the test expects, from the client's last send.
/// Unreachable on a healthy run: the ending comes one deadline in.
const END_WITHIN: Duration = Duration::from_millis(DEADLINE_MS + 10_000);

const APP: &str = "keepalive-app";

// ── Fixture ──────────────────────────────────────────────────────────────────

/// A service whose JoinSession keepalive interval is `interval_ms`.
fn keepalive_service(tv: &TestValkey, interval_ms: u64) -> LiveEditService {
    LiveEditService {
        presence_keepalive_interval_ms: interval_ms,
        ..build_service(tv, Arc::new(AllowAllPermissions))
    }
}

/// A JoinSession header, opted in to the client keepalive or not.
fn header(session_id: &str, client_keepalive: bool) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: session_id.to_owned(),
                user_application: APP.to_owned(),
                client_keepalive,
            },
        )),
    }
}

/// A client keepalive.
fn keepalive() -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Keepalive(
            JoinSessionKeepalive {},
        )),
    }
}

type ClientSender = tokio::sync::mpsc::UnboundedSender<Bytes>;

/// Send `msg` as one gRPC frame in one DATA frame. The channel closes when
/// the join drops its request stream, so a refused send means the join ended.
#[track_caller]
fn send(client: &ClientSender, msg: &JoinSessionRequest) {
    client.send(encode_join_frame(msg)).unwrap_or_else(|_| {
        panic!("the client channel is closed: the join ended while its client was still sending")
    });
}

/// Open a join on `svc` over a request carrying the body-progress record, as
/// the production stack builds it, with the header opted in or not. Returns
/// the response stream, the client side, and when the header was sent.
async fn open_attested_join(
    svc: &LiveEditService,
    session_id: &str,
    client_keepalive: bool,
) -> (JoinResponseStream, ClientSender, Instant) {
    let (stream, client) = open_attested_join_streaming();
    let header_sent = Instant::now();
    send(&client, &header(session_id, client_keepalive));
    let response = handle_join_session(svc, make_request(stream))
        .await
        .unwrap_or_else(|s| panic!("the JoinSession handler must admit the join: {s:?}"));
    (response.into_inner(), client, header_sent)
}

/// The stream's first item, which must be `SessionConnected`.
async fn expect_connected(stream: &mut JoinResponseStream) -> SessionConnected {
    let first = tokio::time::timeout(PRESENCE_WAIT, stream.next())
        .await
        .unwrap_or_else(|_| panic!("SessionConnected did not arrive within {PRESENCE_WAIT:?}"))
        .expect("the join stream ended before SessionConnected")
        .unwrap_or_else(|s| panic!("the join stream failed before SessionConnected: {s:?}"));
    match first.response_type {
        Some(ResponseType::Connected(connected)) => connected,
        other => panic!("the join stream's first message must be SessionConnected; got {other:?}"),
    }
}

/// Watch `stream` for `window`, calling `tick` (a client send, or nothing)
/// every `every`: the join must stay open and deliver nothing.
async fn stays_open(
    stream: &mut JoinResponseStream,
    window: Duration,
    every: Duration,
    mut tick: impl FnMut(),
    what: &str,
) {
    let until = Instant::now() + window;
    while Instant::now() < until {
        tick();
        if let Ok(item) = tokio::time::timeout(every, stream.next()).await {
            panic!(
                "{what}: the join must stay open and deliver nothing, but its stream yielded \
                 {item:?}"
            );
        }
    }
}

/// `live_edit_presence_keepalive_timeouts_total` on `svc`.
fn keepalive_timeouts(svc: &LiveEditService) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_presence_keepalive_timeouts_total",
        &[("service", svc.service_label.as_str())],
    )
    .expect("live_edit_presence_keepalive_timeouts_total is seeded in Metrics::new") as u64
}

/// The status must be the keepalive ending: `UNAVAILABLE` with its message,
/// the refusal detail `CLIENT_KEEPALIVE_TIMEOUT` with no metadata, and a
/// retry delay in `[0, 100 ms)`, decoded as a client decodes it.
#[track_caller]
fn assert_keepalive_ending(status: Option<Status>, what: &str) -> Status {
    let status = status.unwrap_or_else(|| {
        panic!("{what}: the join must end with the keepalive timeout, but it ended OK")
    });
    assert_eq!(
        (status.code(), status.message()),
        (
            Code::Unavailable,
            client_keepalive_timeout_status().message()
        ),
        "{what}: the keepalive ending's code and message"
    );
    assert_refusal_detail(
        &status,
        refusal_reason::CLIENT_KEEPALIVE_TIMEOUT,
        &[],
        CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY,
    );
    status
}

fn count(events: &[PresenceEvent], wanted: impl Fn(&PresenceEvent) -> bool) -> usize {
    events.iter().filter(|e| wanted(e)).count()
}

// ── silence ends an armed join ───────────────────────────────────────────────

/// An opted-in join whose client sends nothing after its header ends at its
/// deadline, through the normal teardown: `SessionConnected` advertised the
/// interval; the stream ends `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT` no sooner
/// than three intervals after the client's last send; the member is gone and
/// a peer saw exactly one `PeerLeft`; the ending is counted once and logged as
/// exactly one INFO `RPC failed` line.
///
/// Current-thread, so `ThreadCapture` sees the line the spawned join task
/// logs; the fixture's keys are purged at the end (`purge_keys`).
#[tokio::test]
async fn keepalive_join_silent_for_three_intervals_ends_unavailable_after_one_peer_left() {
    let what = "a silent opted-in join";
    let tv = TestValkey::setup().await;
    let svc = keepalive_service(&tv, INTERVAL_MS);
    let (session_id, _token) = create_session(&tv, &svc, "keepalive-silent", true).await;
    let mut tap = Tap::open(&tv, &session_id).await;
    let capture = ThreadCapture::start();

    let (mut stream, client, last_sent) = open_attested_join(&svc, &session_id, true).await;
    let connected = expect_connected(&mut stream).await;
    assert_eq!(
        connected.keepalive_interval_ms, INTERVAL_MS,
        "{what}: an armed join is told the configured interval"
    );
    let cid = connected.connection_id;
    let (delivered, status) = read_to_end_within(&mut stream, what, END_WITHIN).await;
    let silent_for = last_sent.elapsed();

    assert!(
        delivered.is_empty(),
        "{what}: nothing else to deliver: {delivered:?}"
    );
    let status = assert_keepalive_ending(status, what);
    assert!(
        silent_for >= DEADLINE,
        "{what}: ended {silent_for:?} after the client's last send, before the {DEADLINE:?} \
         deadline"
    );
    assert!(
        !is_member(&tv, &session_id, &cid).await,
        "{what}: teardown unregisters the member before the status is sent"
    );
    let events = tap.drain(&tv).await;
    assert_eq!(
        count(&events, |e| is_left(e, &cid)),
        1,
        "{what}: a peer sees exactly one PeerLeft: {events:?}"
    );
    assert_eq!(keepalive_timeouts(&svc), 1, "{what}: counted once");

    let captured = capture.snapshot();
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "{what}: exactly one completion line; full capture:\n{}",
        captured.text
    );
    let line = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(line, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(line, "connection_id", cid.as_str().into());
    assert_field(line, "status_message", status.message().into());

    drop(client);
    drop(capture);
    tap.close().await;
    purge_keys(&tv).await;
}

// ── any data keeps an armed join ─────────────────────────────────────────────

/// An opted-in join kept alive first by keepalives alone, then by messages
/// alone, each for two deadlines, is never ended; it ends `OK` when its
/// client closes. Every byte the client sends is stamped on its body's
/// progress record, and the deadline counts from the latest stamp.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_sending_keepalives_then_messages_stays_joined() {
    scenario(async {
        let what = "an opted-in join that keeps sending";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-alive", true).await;
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, true).await;
        let connected = expect_connected(&mut stream).await;
        assert_eq!(connected.keepalive_interval_ms, INTERVAL_MS);

        let keepalives = format!("{what}, keepalives only");
        stays_open(
            &mut stream,
            OBSERVE,
            SEND_EVERY,
            || send(&client, &keepalive()),
            &keepalives,
        )
        .await;
        let mut n = 0u32;
        let messages = format!("{what}, messages only");
        stays_open(
            &mut stream,
            OBSERVE,
            SEND_EVERY,
            || {
                n += 1;
                send(&client, &join_message(format!("alive-{n}").into_bytes()));
            },
            &messages,
        )
        .await;

        assert!(
            is_member(&tv, &session_id, &connected.connection_id).await,
            "{what}: still a member"
        );
        assert_eq!(keepalive_timeouts(&svc), 0, "{what}: never timed out");
        drop(client);
        read_to_ok_end(&mut stream, what).await;
        drain_tracker(&svc).await;
    })
    .await;
}

// ── a message still arriving counts ──────────────────────────────────────────

/// One gRPC frame split into `pieces` DATA frames.
fn split(frame: Bytes, pieces: usize) -> Vec<Bytes> {
    let size = frame.len().div_ceil(pieces);
    (0..frame.len())
        .step_by(size)
        .map(|at| frame.slice(at..(at + size).min(frame.len())))
        .collect()
}

/// A large message sent as many DATA frames over more than a whole deadline
/// keeps an opted-in join alive — the bytes of a message still arriving
/// count, not only completed messages — and peers receive it intact. Once
/// the client then goes silent, the join ends as a silent one does, no sooner
/// than one deadline after its last bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_dribbling_one_large_message_stays_and_delivers_it_intact() {
    /// 16 pieces, one every 300 ms: the last 4.5 s after the first, longer
    /// than the 3 s deadline, with no gap anywhere near it.
    const PIECES: usize = 16;
    const PIECE_EVERY: Duration = Duration::from_millis(300);
    scenario(async {
        let what = "an opted-in join sending one message slowly";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-dribble", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, true).await;
        let cid = expect_connected(&mut stream).await.connection_id;

        let payload: Vec<u8> = (0..48 * 1024).map(|i| (i % 251) as u8).collect();
        let pieces = split(encode_join_frame(&join_message(payload.clone())), PIECES);
        assert_eq!(pieces.len(), PIECES);
        let started = Instant::now();
        let mut last_sent = started;
        for piece in pieces {
            client
                .send(piece)
                .expect("the client channel must accept a piece");
            last_sent = Instant::now();
            if let Ok(item) = tokio::time::timeout(PIECE_EVERY, stream.next()).await {
                panic!("{what}: the join must stay open while its message arrives; got {item:?}");
            }
        }
        assert!(
            last_sent - started > DEADLINE,
            "precondition: the message took longer than one deadline to send"
        );

        let delivered = tap
            .until("the slowly sent message", |e| is_message(e, &cid))
            .await;
        match delivered.last() {
            Some(PresenceEvent::PeerMessage { data, .. }) => {
                assert!(*data == payload, "{what}: the message reached peers intact")
            }
            other => panic!("{what}: expected the message; got {other:?}"),
        }
        assert!(
            is_member(&tv, &session_id, &cid).await,
            "{what}: still a member after the message"
        );

        let (_, status) = read_to_end_within(&mut stream, what, END_WITHIN).await;
        let silent_for = last_sent.elapsed();
        assert_keepalive_ending(status, what);
        assert!(
            silent_for >= DEADLINE,
            "{what}: ended {silent_for:?} after the message's last bytes, before the \
             {DEADLINE:?} deadline"
        );
        assert_eq!(keepalive_timeouts(&svc), 1, "{what}: counted once");
        drop(client);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ── a join that did not opt in ───────────────────────────────────────────────

/// A join whose header did not set `client_keepalive`, on a service with the
/// keepalive on and over a request carrying the progress record, is told
/// interval 0 and is never ended for silence: every current client behaves
/// as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_without_keepalive_opt_in_silent_past_the_deadline_stays_and_is_told_zero() {
    scenario(async {
        let what = "a silent join that did not opt in";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-not-opted", true).await;
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, false).await;
        let connected = expect_connected(&mut stream).await;
        assert_eq!(
            connected.keepalive_interval_ms, 0,
            "{what}: a join that did not opt in is told 0"
        );

        stays_open(&mut stream, OBSERVE, OBSERVE, || {}, what).await;
        assert!(
            is_member(&tv, &session_id, &connected.connection_id).await,
            "{what}: still a member"
        );
        assert_eq!(keepalive_timeouts(&svc), 0, "{what}: never timed out");
        drop(client);
        read_to_ok_end(&mut stream, what).await;
        drain_tracker(&svc).await;
    })
    .await;
}

// ── the replay never counts ──────────────────────────────────────────────────

/// The deadline starts at the bidi loop's start, not at the join's start: a
/// membership replay held for two deadlines (the client silent throughout,
/// its stream not read) does not end the join when it is released, and the
/// join then ends no sooner than one deadline after the release.
///
/// Needs `--features test-fault-injection` for the replay seam; without it
/// the test is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_deadline_counts_from_the_loop_start_not_through_a_held_replay() {
    use live_edit::grpc::presence::PresenceReplaySeam;

    scenario(async {
        let what = "an opted-in join behind a held replay";
        let tv = TestValkey::setup().await;
        let seam = Arc::new(PresenceReplaySeam::new());
        let svc = LiveEditService {
            presence_replay_seam_for_test: Some(Arc::clone(&seam)),
            ..keepalive_service(&tv, INTERVAL_MS)
        };
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-replay", true).await;

        let gate = &seam.before_first_send;
        gate.arm();
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, true).await;
        let connected = expect_connected(&mut stream).await;
        assert_eq!(connected.keepalive_interval_ms, INTERVAL_MS);
        wait_parked(gate, "the membership replay").await;
        tokio::time::sleep(OBSERVE).await;
        let released = Instant::now();
        gate.release();

        let (delivered, status) = read_to_end_within(&mut stream, what, END_WITHIN).await;
        let ended_after = released.elapsed();
        assert!(
            delivered.is_empty(),
            "{what}: nothing to replay: {delivered:?}"
        );
        assert_keepalive_ending(status, what);
        assert!(
            ended_after >= DEADLINE,
            "{what}: ended {ended_after:?} after the replay's release, before the {DEADLINE:?} \
             deadline: the held replay counted against it"
        );
        assert!(!is_member(&tv, &session_id, &connected.connection_id).await);
        drop(client);
        drain_tracker(&svc).await;
    })
    .await;
}

// ── keepalives never reach peers ─────────────────────────────────────────────

/// Keepalives are never published or delivered: a peer's next event from the
/// keepalive sender, after its `PeerJoined`, is the marker message it sent
/// after five keepalives, and the session's channel carried exactly that one
/// message from it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_is_never_delivered_or_published_to_peers() {
    const MARKER: &[u8] = b"after-five-keepalives";
    scenario(async {
        let what = "keepalives then a marker";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let feed = spawn_router_feed(&tv, &svc).await;
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-not-broadcast", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        let (mut peer, peer_client) = open_join(&svc, &session_id, "peer-app").await;
        expect_session_connected(&mut peer, PRESENCE_WAIT).await;
        let (mut sender, client, _) = open_attested_join(&svc, &session_id, true).await;
        let sender_id = expect_connected(&mut sender).await.connection_id;
        expect_peer_joined(&mut peer, &sender_id, what).await;

        for _ in 0..5 {
            send(&client, &keepalive());
        }
        send(&client, &join_message(MARKER.to_vec()));

        let next = tokio::time::timeout(PRESENCE_WAIT, peer.next())
            .await
            .unwrap_or_else(|_| panic!("{what}: the peer got nothing within {PRESENCE_WAIT:?}"))
            .expect("the peer's stream ended")
            .unwrap_or_else(|s| panic!("{what}: the peer's stream failed: {s:?}"));
        match next.response_type {
            Some(ResponseType::PeerMessage(m)) => assert_eq!(
                (m.connection_id.as_str(), m.data.as_slice()),
                (sender_id.as_str(), MARKER),
                "{what}: the peer's next event is the marker"
            ),
            other => panic!("{what}: the peer's next event must be the marker; got {other:?}"),
        }
        let mut published = tap
            .until("the sender's first message", |e| is_message(e, &sender_id))
            .await;
        published.extend(tap.drain(&tv).await);
        let from_sender: Vec<&[u8]> = published
            .iter()
            .filter_map(|e| match e {
                PresenceEvent::PeerMessage {
                    connection_id,
                    data,
                } if *connection_id == sender_id => Some(data.as_slice()),
                _ => None,
            })
            .collect();
        assert_eq!(
            from_sender,
            vec![MARKER],
            "{what}: the marker is the only message published from the sender"
        );

        drop(client);
        read_to_ok_end(&mut sender, what).await;
        drop(peer_client);
        read_to_ok_end(&mut peer, what).await;
        drain_tracker(&svc).await;
        tap.close().await;
        feed.cancel();
    })
    .await;
}

// ── a join whose event send is blocked ───────────────────────────────────────

/// Peer messages to fill a join's response channel and leave its loop
/// waiting on a send: more than the channel's delivery slots.
const OVERFILL: usize = RESPONSE_DELIVERY_SLOTS + 8;

/// A peer message from a remote connection, carrying `seq`.
fn remote_message(seq: usize) -> PresenceEvent {
    PresenceEvent::PeerMessage {
        connection_id: "remote-peer".to_owned(),
        data: (seq as u64).to_be_bytes().to_vec(),
    }
}

/// Broadcast [`OVERFILL`] messages straight into `svc`'s router, as the
/// pub/sub feed would deliver them, for a join whose client reads nothing:
/// its loop sends until the response channel is full and then waits on the
/// next send.
fn overfill(svc: &LiveEditService, session_id: &str) {
    for seq in 0..OVERFILL {
        assert!(
            svc.presence_router
                .broadcast(session_id, remote_message(seq))
                >= 1,
            "the join's receiver must be on the session's router entry"
        );
    }
}

/// The messages a client that read nothing gets once it reads: the first
/// [`RESPONSE_DELIVERY_SLOTS`], in order — what the channel held.
#[track_caller]
fn assert_channel_contents(delivered: &[live_edit::grpc::proto::JoinSessionResponse], what: &str) {
    assert_eq!(
        delivered.len(),
        RESPONSE_DELIVERY_SLOTS,
        "{what}: the full channel's contents: {delivered:?}"
    );
    for (seq, msg) in delivered.iter().enumerate() {
        match &msg.response_type {
            Some(ResponseType::PeerMessage(m)) => assert_eq!(
                m.data,
                (seq as u64).to_be_bytes().to_vec(),
                "{what}: message {seq} in order"
            ),
            other => panic!("{what}: expected only the messages; got {other:?}"),
        }
    }
}

/// An opted-in join whose client stopped reading in a busy session — its
/// loop waiting on an event send with the channel full — and sent nothing is
/// still ended at its deadline: the member is torn down, and the client,
/// reading at last, gets what the channel held and then the keepalive
/// ending.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_not_reading_a_busy_session_still_ends_at_the_deadline() {
    scenario(async {
        let what = "a silent opted-in join not reading a busy session";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-blocked", true).await;
        let (mut stream, client, last_sent) = open_attested_join(&svc, &session_id, true).await;
        let cid = expect_connected(&mut stream).await.connection_id;
        overfill(&svc, &session_id);

        let (tv_ref, sid, cid_ref) = (&tv, session_id.as_str(), cid.as_str());
        poll_until(
            &format!("{what}: its teardown"),
            END_WITHIN,
            move || async move { !is_member(tv_ref, sid, cid_ref).await },
        )
        .await;
        let silent_for = last_sent.elapsed();
        assert!(
            silent_for >= DEADLINE,
            "{what}: torn down {silent_for:?} after the client's last send, before the \
             {DEADLINE:?} deadline"
        );

        let (delivered, status) = read_to_end(&mut stream, what).await;
        assert_channel_contents(&delivered, what);
        assert_keepalive_ending(status, what);
        assert_eq!(keepalive_timeouts(&svc), 1, "{what}: counted once");
        drop(client);
        drain_tracker(&svc).await;
    })
    .await;
}

/// As above, but the client keeps sending keepalives, and a message, while
/// its responses are back-pressured: the loop, waiting on the event send,
/// still reads its stream, so the join stays for two deadlines and the
/// message is published. The client then closes, and reading at last gets
/// what the channel held and an `OK` end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_not_reading_but_sending_keepalives_stays_and_publishes_its_messages() {
    const MARKER: &[u8] = b"sent-while-back-pressured";
    scenario(async {
        let what = "an opted-in join not reading a busy session, sending keepalives";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-blocked-alive", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, true).await;
        let cid = expect_connected(&mut stream).await.connection_id;
        overfill(&svc, &session_id);

        let until = Instant::now() + OBSERVE;
        while Instant::now() < until {
            send(&client, &keepalive());
            tokio::time::sleep(SEND_EVERY).await;
        }
        send(&client, &join_message(MARKER.to_vec()));
        let published = tap
            .until("the message sent while back-pressured", |e| {
                is_message(e, &cid)
            })
            .await;
        match published.last() {
            Some(PresenceEvent::PeerMessage { data, .. }) => assert_eq!(
                data.as_slice(),
                MARKER,
                "{what}: the sender's first published message is the marker"
            ),
            other => panic!("{what}: expected the marker; got {other:?}"),
        }
        assert!(
            is_member(&tv, &session_id, &cid).await,
            "{what}: still a member after two deadlines"
        );
        assert_eq!(keepalive_timeouts(&svc), 0, "{what}: never timed out");

        drop(client);
        let delivered = read_to_ok_end(&mut stream, what).await;
        assert_channel_contents(&delivered, what);
        drain_tracker(&svc).await;
        tap.close().await;
    })
    .await;
}

// ── interval 0 is off ────────────────────────────────────────────────────────

/// With the keepalive configured off (interval 0), an opted-in join is told
/// interval 0 and is never ended for silence: the rollback looks pre-feature
/// on the wire.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_interval_zero_advertises_zero_and_never_ends_an_opted_in_join() {
    scenario(async {
        let what = "a silent opted-in join with the keepalive off";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, 0);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-off", true).await;
        let (mut stream, client, _) = open_attested_join(&svc, &session_id, true).await;
        let connected = expect_connected(&mut stream).await;
        assert_eq!(
            connected.keepalive_interval_ms, 0,
            "{what}: told 0, so it sends no keepalives"
        );

        stays_open(&mut stream, OBSERVE, OBSERVE, || {}, what).await;
        assert!(
            is_member(&tv, &session_id, &connected.connection_id).await,
            "{what}: still a member"
        );
        assert_eq!(keepalive_timeouts(&svc), 0, "{what}: never timed out");
        drop(client);
        read_to_ok_end(&mut stream, what).await;
        drain_tracker(&svc).await;
    })
    .await;
}

// ── a request without the progress record ────────────────────────────────────

/// An opted-in join over a request that carries no body-progress record —
/// one built by any stack other than `live_service_server` — is admitted,
/// told interval 0, and never ended for silence: the handler leaves it
/// unarmed rather than refusing it, so a wiring regression degrades to the
/// pre-keepalive behaviour instead of failing every opted-in join.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_without_a_progress_record_is_admitted_unarmed_and_told_zero() {
    scenario(async {
        let what = "an opted-in join without a progress record";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-no-record", true).await;

        let (stream, client) = open_join_streaming();
        send(&client, &header(&session_id, true));
        let mut stream = handle_join_session(&svc, make_join_request(stream))
            .await
            .unwrap_or_else(|s| panic!("{what}: the handler must admit the join; got {s:?}"))
            .into_inner();
        let connected = expect_connected(&mut stream).await;
        assert_eq!(
            connected.keepalive_interval_ms, 0,
            "{what}: an unarmed join is told 0"
        );

        stays_open(&mut stream, OBSERVE, OBSERVE, || {}, what).await;
        assert!(
            is_member(&tv, &session_id, &connected.connection_id).await,
            "{what}: still a member"
        );
        assert_eq!(keepalive_timeouts(&svc), 0, "{what}: never timed out");
        drop(client);
        read_to_ok_end(&mut stream, what).await;
        drain_tracker(&svc).await;
    })
    .await;
}

// ── Served through the production stack ──────────────────────────────────────

/// Through a real server built by `live_service_server` — the production
/// stack, which attaches the body-progress record to every request — an
/// opted-in join is armed (told the interval), the keepalives a real client
/// sends over HTTP/2 keep it for two deadlines, and once they stop it ends
/// `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT` on the wire, no sooner than one
/// deadline after the last keepalive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_join_served_through_the_production_stack_is_armed_by_its_wire_bytes() {
    use live_edit::grpc::proto::live_service_client::LiveServiceClient;
    use live_edit::grpc::transport::{live_service_server, production_server_builder};
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::metadata::MetadataValue;

    scenario(async {
        let what = "a served opted-in join";
        let tv = TestValkey::setup().await;
        let svc = keepalive_service(&tv, INTERVAL_MS);
        let (session_id, _token) = create_session(&tv, &svc, "keepalive-served", true).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a random port");
        let addr = listener.local_addr().expect("local addr");
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder()
                .add_service(live_service_server(svc))
                .serve_with_incoming(incoming)
                .await
                .expect("test server error");
        });
        tokio::time::timeout(PRESENCE_WAIT, ready_rx)
            .await
            .expect("the server did not signal readiness")
            .expect("the server task dropped its ready channel");

        let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
            .expect("valid endpoint URI")
            .connect()
            .await
            .expect("connect to the test server");
        let (requests, rx) = tokio::sync::mpsc::channel::<JoinSessionRequest>(16);
        requests
            .send(header(&session_id, true))
            .await
            .expect("queue the header");
        let mut request = tonic::Request::new(ReceiverStream::new(rx));
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::from_static("Bearer keepalive-served-test"),
        );
        let mut responses = tokio::time::timeout(
            PRESENCE_WAIT,
            LiveServiceClient::new(channel).join_session(request),
        )
        .await
        .expect("the join was not admitted in time")
        .expect("the join is admitted")
        .into_inner();

        let first = tokio::time::timeout(PRESENCE_WAIT, responses.message())
            .await
            .expect("SessionConnected in time")
            .expect("SessionConnected, not an error")
            .expect("SessionConnected, not the end");
        match first.response_type {
            Some(ResponseType::Connected(c)) => assert_eq!(
                c.keepalive_interval_ms, INTERVAL_MS,
                "{what}: the production stack attaches the progress record, so the join is armed"
            ),
            other => panic!("{what}: the first response must be SessionConnected; got {other:?}"),
        }

        let until = Instant::now() + OBSERVE;
        let mut last_sent = Instant::now();
        while Instant::now() < until {
            requests.send(keepalive()).await.expect("queue a keepalive");
            last_sent = Instant::now();
            if let Ok(item) = tokio::time::timeout(SEND_EVERY, responses.message()).await {
                panic!("{what}: kept by its keepalives, the join yields nothing; got {item:?}");
            }
        }

        let ended = tokio::time::timeout(END_WITHIN, responses.message())
            .await
            .unwrap_or_else(|_| panic!("{what}: did not end within {END_WITHIN:?}"));
        let silent_for = last_sent.elapsed();
        let status = match ended {
            Err(status) => status,
            Ok(other) => panic!("{what}: expected the keepalive ending; got {other:?}"),
        };
        assert_keepalive_ending(Some(status), what);
        assert!(
            silent_for >= DEADLINE,
            "{what}: ended {silent_for:?} after the last keepalive, before the {DEADLINE:?} \
             deadline"
        );
        drop(requests);
        server.abort();
    })
    .await;
}
