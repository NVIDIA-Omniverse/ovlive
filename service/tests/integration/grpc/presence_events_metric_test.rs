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

// End-to-end coverage for the two JoinSession instruments, driven through real
// `handle_join_session` streams against Valkey and read off the service's own
// Prometheus exporter:
//
//   * `live_presence_events_total{event_type, service}` — once per live
//     presence event per local recipient, after the event was enqueued to that
//     recipient's stream. Never for the recipient's own events, the membership
//     replay a joiner receives, a live `joined` suppressed as a duplicate of
//     that replay (never sent), or events a lagging recipient dropped (those
//     count on `live_edit_presence_broadcast_dropped_total`).
//   * `live_edit_join_latency_seconds{endpoint="JoinSession", service}` — once
//     per join that has enqueued SessionConnected and its complete membership
//     replay; never on failure, nor for a join a shutdown or session cancel
//     ends mid-replay (`join_lifecycle_test` covers that case).
//
// A stream's presence deliveries reach the counter in batches: its tally
// flushes on a 1 s timer while any are pending, at 64 pending, and when the
// stream's loop ends. So every read here polls until the expected cell reaches
// AT LEAST its value and only then asserts the exact whole-family state. That
// exact read is sound because a flush publishes everything its tally holds: a
// stray tick recorded on the same stream BEFORE the event the poll waits on is
// published with it or earlier, and one on a stream that has already ended was
// published when that stream's loop returned. An over-count therefore fails on
// the exact assertion rather than on a poll that never sees its value.
//
// ── Wait budget ──────────────────────────────────────────────────────────────
//
// `SETTLE` (10 s) per message read and per polled condition; `DRAIN` (20 s) is
// ONE deadline over the whole lagged-recipient drain, however many events it
// reads. The heaviest test (the two-client one) reads six messages and polls
// four conditions: ~100 s at every budget's full value, inside nextest's 120 s
// hang-killer. The healthy path takes a few seconds: each polled presence
// count waits for up to one 1 s tally flush.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::BufMut;
use futures::{Stream, StreamExt};
use prost::Message;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::codec::{BufferSettings, Streaming};
use tonic::{Request, Status};
use tonic_prost::ProstCodec;

use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::{
    join_session_request, join_session_response, JoinSessionHeader, JoinSessionRequest,
    JoinSessionResponse, SessionMessageRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::presence::{
    spawn_presence_subscriber, ConnectedUser, LocalPresenceRouter, PresenceEvent,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

type ResponseStream = Pin<Box<dyn Stream<Item = Result<JoinSessionResponse, Status>> + Send>>;

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// One deadline for the whole lagged-recipient drain.
const DRAIN: Duration = Duration::from_secs(20);

/// `build_service`'s `service` label.
const SERVICE: &str = "test";

// ── JoinSession stream helpers ──────────────────────────────────────────────

fn encode_frame(msg: &JoinSessionRequest) -> bytes::Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0);
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

fn join_header(session_id: &str, user_application: &str) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: session_id.to_owned(),
                user_application: user_application.to_owned(),
                client_keepalive: false,
            },
        )),
    }
}

fn peer_message(data: &[u8]) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Message(
            SessionMessageRequest {
                data: data.to_vec(),
            },
        )),
    }
}

/// A request stream backed by an open channel: frames are sent through the
/// returned sender, and dropping it ends the client stream (a clean leave).
fn open_request_stream() -> (
    Streaming<JoinSessionRequest>,
    mpsc::UnboundedSender<bytes::Bytes>,
) {
    use http_body_util::StreamBody;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, rx) = mpsc::unbounded_channel::<bytes::Bytes>();
    let body = StreamBody::new(
        UnboundedReceiverStream::new(rx)
            .map(|b| Ok::<_, std::io::Error>(http_body::Frame::data(b))),
    );
    let decoder = ProstCodec::<(), JoinSessionRequest>::raw_decoder(BufferSettings::default());
    (Streaming::new_request(decoder, body, None, None), tx)
}

/// Join `session_id` and read through SessionConnected, returning the response
/// stream, the frame sender, and this client's connection id.
async fn join(
    svc: &LiveEditService,
    session_id: &str,
    user_application: &str,
) -> (ResponseStream, mpsc::UnboundedSender<bytes::Bytes>, String) {
    let (stream, tx) = open_request_stream();
    tx.send(encode_frame(&join_header(session_id, user_application)))
        .expect("the request channel is open");
    let mut request = Request::new(stream);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    let mut responses: ResponseStream = handle_join_session(svc, request)
        .await
        .expect("JoinSession must succeed")
        .into_inner();
    let connection_id = match next_response(&mut responses, "SessionConnected").await {
        join_session_response::ResponseType::Connected(c) => c.connection_id,
        other => panic!("expected SessionConnected, got {other:?}"),
    };
    (responses, tx, connection_id)
}

async fn next_response(
    stream: &mut ResponseStream,
    what: &str,
) -> join_session_response::ResponseType {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what}: nothing within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("{what}: the stream ended"))
        .unwrap_or_else(|e| panic!("{what}: error status {e:?}"))
        .response_type
        .unwrap_or_else(|| panic!("{what}: a response carried no payload"))
}

async fn create_session(tv: &TestValkey, svc: &LiveEditService, name: &str) -> String {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    // What CreateSession does on the handling replica.
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());
    session_id
}

// ── Metric readers ──────────────────────────────────────────────────────────

/// `live_presence_events_total{event_type}`. Every `event_type` is seeded, so
/// an absent series is a label bug, not an event that has not happened.
fn presence_events(metrics: &Metrics, event_type: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_presence_events_total",
        &[("event_type", event_type), ("service", SERVICE)],
    )
    .unwrap_or_else(|| {
        panic!("live_presence_events_total{{event_type=\"{event_type}\"}} is seeded")
    }) as u64
}

fn presence_dropped(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_presence_broadcast_dropped_total",
        &[("service", SERVICE)],
    )
    .expect("live_edit_presence_broadcast_dropped_total is seeded") as u64
}

fn presence_resyncs(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_presence_resyncs_total",
        &[("service", SERVICE)],
    )
    .expect("live_edit_presence_resyncs_total is seeded") as u64
}

/// `_count` of the JoinSession join-latency series; absent (0) until the first
/// observation, as histograms are not seeded.
fn join_latency_count(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_join_latency_seconds_count",
        &[("endpoint", "JoinSession"), ("service", SERVICE)],
    )
    .unwrap_or(0.0) as u64
}

async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = Instant::now() + SETTLE;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn assert_presence_events(metrics: &Metrics, joined: u64, left: u64, message: u64, when: &str) {
    assert_eq!(
        (
            presence_events(metrics, "joined"),
            presence_events(metrics, "left"),
            presence_events(metrics, "message"),
        ),
        (joined, left, message),
        "{when}: (joined, left, message)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// live_presence_events_total
// ═══════════════════════════════════════════════════════════════════════════

/// Two clients, A then B, on one replica. Only A ever receives a live event
/// about the other — B's join, B's message, B's leave — so each `event_type`
/// ends at exactly one. Everything else that flows is excluded by contract and
/// would show up as an extra tick:
///
/// - each client's own PeerJoined / PeerMessage, which comes back to it through
///   the pub/sub bridge and is self-filtered;
/// - the PeerJoined(A) that B receives as membership replay, not as a live
///   event.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn presence_events_count_each_live_delivery_once_per_recipient_by_event_type() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let bridge_cancel = CancellationToken::new();
    let bridge = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        bridge_cancel.clone(),
    );
    let session_id = create_session(&tv, &svc, "b21r-presence-events").await;
    assert_presence_events(&svc.metrics, 0, 0, 0, "before any join");

    let (mut a_stream, _a_tx, a_id) = join(&svc, &session_id, "app-a").await;
    let (mut b_stream, b_tx, b_id) = join(&svc, &session_id, "app-b").await;

    // B's membership replay names A. It is not a live event.
    match next_response(&mut b_stream, "B's membership replay").await {
        join_session_response::ResponseType::PeerJoined(p) => assert_eq!(p.connection_id, a_id),
        other => panic!("expected B's replay of A, got {other:?}"),
    }
    // A receives B's join live.
    match next_response(&mut a_stream, "A's PeerJoined(B)").await {
        join_session_response::ResponseType::PeerJoined(p) => assert_eq!(p.connection_id, b_id),
        other => panic!("expected PeerJoined(B) on A, got {other:?}"),
    }
    await_condition(
        || presence_events(&svc.metrics, "joined") >= 1,
        "joined reaches 1",
    )
    .await;

    b_tx.send(encode_frame(&peer_message(b"hello from b")))
        .expect("B's request channel is open");
    match next_response(&mut a_stream, "A's PeerMessage(B)").await {
        join_session_response::ResponseType::PeerMessage(m) => {
            assert_eq!(
                (m.connection_id.as_str(), m.data.as_slice()),
                (b_id.as_str(), &b"hello from b"[..])
            )
        }
        other => panic!("expected PeerMessage(B) on A, got {other:?}"),
    }
    await_condition(
        || presence_events(&svc.metrics, "message") >= 1,
        "message reaches 1",
    )
    .await;

    drop(b_tx);
    drop(b_stream);
    match next_response(&mut a_stream, "A's PeerLeft(B)").await {
        join_session_response::ResponseType::PeerLeft(p) => assert_eq!(p.connection_id, b_id),
        other => panic!("expected PeerLeft(B) on A, got {other:?}"),
    }
    await_condition(
        || presence_events(&svc.metrics, "left") >= 1,
        "left reaches 1",
    )
    .await;

    assert_presence_events(
        &svc.metrics,
        1,
        1,
        1,
        "after B joined, messaged and left: one live delivery of each type, all to A. A higher \
         count means a self-filtered event or B's membership replay was counted",
    );

    bridge_cancel.cancel();
    let _ = tokio::time::timeout(SETTLE, bridge).await;
}

/// A recipient that falls behind its session's presence capacity loses events
/// to `PresenceRecvError::Lagged`. Those count on
/// `live_edit_presence_broadcast_dropped_total`, never on
/// `live_presence_events_total`, so the two partition the events published:
/// `message` equals exactly what the recipient received, and received plus
/// dropped equals everything broadcast. Every lost event is a `PeerMessage`,
/// so the stream continues and no resync is counted
/// (`live_edit_presence_resyncs_total` stays 0).
///
/// The lag is forced without the pub/sub bridge: the recipient does not read
/// its response stream, so its bidi loop parks on a full response channel while
/// events are broadcast straight into a capacity-2 local router.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn presence_events_do_not_count_events_a_lagging_recipient_dropped() {
    const BROADCAST: u64 = 60;

    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    svc.presence_router = Arc::new(LocalPresenceRouter::with_capacity(2));
    let session_id = create_session(&tv, &svc, "b21r-presence-lag").await;

    let (mut stream, _tx, _id) = join(&svc, &session_id, "app-lagging").await;
    for i in 0..BROADCAST {
        svc.presence_router.broadcast(
            &session_id,
            PresenceEvent::PeerMessage {
                connection_id: "another-replica-peer".to_owned(),
                data: i.to_be_bytes().to_vec(),
            },
        );
    }

    let received = tokio::time::timeout(DRAIN, async {
        let mut received = 0u64;
        while received + presence_dropped(&svc.metrics) < BROADCAST {
            match tokio::time::timeout(Duration::from_millis(100), stream.next()).await {
                Ok(Some(Ok(msg))) => match msg.response_type {
                    Some(join_session_response::ResponseType::PeerMessage(_)) => received += 1,
                    other => panic!("expected only PeerMessage, got {other:?}"),
                },
                Ok(Some(Err(e))) => panic!("the stream failed: {e:?}"),
                Ok(None) => panic!("the stream ended"),
                // Nothing buffered right now; re-check the drop counter.
                Err(_) => {}
            }
        }
        received
    })
    .await
    .expect("every broadcast event must be either received or counted as dropped");

    let dropped = presence_dropped(&svc.metrics);
    assert!(
        dropped > 0,
        "precondition: the recipient must actually have lagged (received {received} of {BROADCAST})"
    );
    await_condition(
        || presence_events(&svc.metrics, "message") >= received,
        "the message cell reaches the number of messages received",
    )
    .await;
    assert_presence_events(
        &svc.metrics,
        0,
        0,
        received,
        "the message cell counts exactly the deliveries the recipient received; the lagged \
         events are on the drop counter only",
    );
    assert_eq!(received + dropped, BROADCAST);
    assert_eq!(
        presence_resyncs(&svc.metrics),
        0,
        "a lag that lost only peer messages continues the stream; it is no resync"
    );
}

/// A live `joined` for a peer the recipient's membership replay already
/// delivered, arriving before the recipient's own presence event, is a
/// duplicate the recipient's loop suppresses: it is neither sent nor counted.
/// A leave and rejoin of that peer afterwards are sent and counted.
///
/// Runs without the pub/sub bridge, so events reach the local streams only
/// through the router broadcasts below. Neither join's own `joined` ever
/// arrives, so A's set of replay-delivered ids stays live (for its 60 s time
/// limit, far longer than this test) and the injected `joined B` is
/// suppressed, exactly as a pre-snapshot announcement of B arriving live would
/// be. A's stream then carries the sentinel message next, not `PeerJoined(B)`.
///
/// Expected cells, A and B together: `joined` 1 (A, B's rejoin), `left` 1 (A,
/// B's leave), `message` 2 (the sentinel, on each stream). B's events about
/// itself are self-filtered on B.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn presence_events_do_not_count_a_suppressed_replay_duplicate() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "presence-events-duplicate").await;

    let (_b_stream, _b_tx, b_id) = join(&svc, &session_id, "app-b").await;
    let (mut a_stream, _a_tx, _a_id) = join(&svc, &session_id, "app-a").await;
    match next_response(&mut a_stream, "A's membership replay").await {
        join_session_response::ResponseType::PeerJoined(p) => assert_eq!(p.connection_id, b_id),
        other => panic!("expected A's replay of B, got {other:?}"),
    }

    let b_user = ConnectedUser {
        connection_id: b_id.clone(),
        user_name: "Test User".to_owned(),
        user_application: "app-b".to_owned(),
    };
    svc.presence_router
        .broadcast(&session_id, PresenceEvent::PeerJoined(b_user.clone()));
    svc.presence_router.broadcast(
        &session_id,
        PresenceEvent::PeerMessage {
            connection_id: "sentinel".to_owned(),
            data: b"sentinel".to_vec(),
        },
    );
    match next_response(&mut a_stream, "A's next live event").await {
        join_session_response::ResponseType::PeerMessage(m) => assert_eq!(
            m.connection_id, "sentinel",
            "A's next event must be the sentinel"
        ),
        other => panic!(
            "the duplicate PeerJoined(B) must be suppressed, so A's next event is the sentinel \
             message; got {other:?}"
        ),
    }

    svc.presence_router.broadcast(
        &session_id,
        PresenceEvent::PeerLeft {
            connection_id: b_id.clone(),
        },
    );
    svc.presence_router
        .broadcast(&session_id, PresenceEvent::PeerJoined(b_user));
    match next_response(&mut a_stream, "A's PeerLeft(B)").await {
        join_session_response::ResponseType::PeerLeft(p) => assert_eq!(p.connection_id, b_id),
        other => panic!("expected PeerLeft(B) on A, got {other:?}"),
    }
    match next_response(&mut a_stream, "A's PeerJoined(B) after the leave").await {
        join_session_response::ResponseType::PeerJoined(p) => assert_eq!(p.connection_id, b_id),
        other => panic!("expected B's rejoin on A, got {other:?}"),
    }

    await_condition(
        || {
            presence_events(&svc.metrics, "joined") >= 1
                && presence_events(&svc.metrics, "left") >= 1
                && presence_events(&svc.metrics, "message") >= 2
        },
        "every forwarded delivery reaches the counter",
    )
    .await;
    assert_presence_events(
        &svc.metrics,
        1,
        1,
        2,
        "the suppressed duplicate PeerJoined(B) must not be counted; only B's rejoin after its \
         leave is",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// live_edit_join_latency_seconds{endpoint="JoinSession"}
// ═══════════════════════════════════════════════════════════════════════════

/// Each successful join observes one join latency — the first client's and
/// the second's, whose join includes a membership replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_latency_is_observed_once_per_successful_join() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "b21r-join-latency").await;
    assert_eq!(join_latency_count(&svc.metrics), 0);

    let (_a_stream, _a_tx, _) = join(&svc, &session_id, "app-a").await;
    await_condition(
        || join_latency_count(&svc.metrics) >= 1,
        "the first join observes",
    )
    .await;
    assert_eq!(
        join_latency_count(&svc.metrics),
        1,
        "one join, one observation"
    );
    let (mut b_stream, _b_tx, _) = join(&svc, &session_id, "app-b").await;
    match next_response(&mut b_stream, "B's membership replay").await {
        join_session_response::ResponseType::PeerJoined(_) => {}
        other => panic!("expected B's replay of A, got {other:?}"),
    }
    await_condition(
        || join_latency_count(&svc.metrics) >= 2,
        "the second join observes",
    )
    .await;
    assert_eq!(
        join_latency_count(&svc.metrics),
        2,
        "two joins, two observations"
    );
}

/// A join to a session that does not exist fails NOT_FOUND and observes
/// nothing: join latency is success-only.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_latency_is_not_observed_for_an_unknown_session() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (stream, tx) = open_request_stream();
    tx.send(encode_frame(&join_header("no-such-session", "app")))
        .expect("the request channel is open");
    let mut request = Request::new(stream);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    let status = match handle_join_session(&svc, request).await {
        Err(status) => status,
        Ok(response) => {
            let mut responses: ResponseStream = response.into_inner();
            tokio::time::timeout(SETTLE, responses.next())
                .await
                .expect("the stream must end promptly")
                .expect("the stream must yield its status")
                .expect_err("the first item must be the NOT_FOUND status")
        }
    };
    assert_eq!(status.code(), tonic::Code::NotFound);
    assert_eq!(
        join_latency_count(&svc.metrics),
        0,
        "a join that never reached SessionConnected must not observe a join latency"
    );
}
