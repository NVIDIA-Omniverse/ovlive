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

//! HTTP/2 connections survive presence-sized small frames left unread
//! (SDD §5.1).
//!
//! h2 closes a connection with GOAWAY `ENHANCE_YOUR_CALM` once too much
//! small-DATA-frame overhead sits unread on it: each non-final DATA frame under
//! 256 bytes costs `256 − payload` bytes of a per-connection budget until the
//! application reads it. h2 0.4.16–0.4.18 fixed that budget at 25,600 bytes.
//! At worst, with one message per DATA frame, that closes a connection at
//! about 100 of the smallest frames, or about 120–150 presence events of 45–85
//! bytes, so ordinary `JoinSession` fan-out was enough to kill a whole
//! connection and every stream on it. From 0.4.19 the
//! budget is half the connection's initial receive window: 2.5 MiB for a tonic
//! client on hyper's defaults and 512 KiB for the service, whose windows
//! `production_server_builder` leaves at hyper's defaults.
//!
//! The invariant pinned here is **"a connection survives N unread small
//! frames"**, from both ends:
//!
//! * [`tonic_client_with_unread_join_streams_survives_presence_fanout`] — the
//!   client side. One tonic `Channel` configured like the Rust client library
//!   holds several `JoinSession` streams it does not read while the service
//!   fans small `PeerMessage` events out to them.
//! * [`service_connection_with_unread_small_join_messages_survives`] — the
//!   service side. A raw `h2` client sends hundreds of small `JoinSession`
//!   messages, one per DATA frame, while the handler is parked and reads none
//!   of them.
//!
//! Both counts sit well past the 0.4.16–0.4.18 kill point and far below the
//! 0.4.19 budget, so on h2 0.4.17 each test fails with the connection closed
//! (client side: a locally initiated GOAWAY; service side: a GOAWAY
//! `ENHANCE_YOUR_CALM` received from the service) and on 0.4.19 each passes.
//! The failure messages name the `h2::Error` flags, so a red run says which
//! side closed the connection.
//!
//! Both servers are built from [`production_server_builder`] — the builder
//! both production servers use — with `MetricsLayer` and the auth interceptor
//! wired as in `main.rs`, so a change to the service's HTTP/2 settings is
//! tested here rather than drifting past a test-only builder.
//!
//! The service-side test needs the `test-fault-injection` feature (it parks the
//! join at `PresenceReplaySeam::before_first_send`); without it the test is
//! compiled out and the run reports green having covered only the client side.
//!
//! # Wait budgets
//!
//! Every wait is a deadline, not a sleep; the healthy path pays milliseconds.
//! Worst case per test, every wait at its full budget, against nextest's
//! 120 s hang-killer:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | client side | `SETTLE` (server ready) + `JOIN_BUDGET` + `FANOUT_BUDGET` + `DRAIN_BUDGET` + `SETTLE` (Ping) + `PRESENCE_WAIT` (teardown) | 5 + 15 + 30 + 15 + 5 + 10 = 80 s |
//! | service side | `SETTLE` (server ready) + `JOIN_BUDGET` + 4 × `SETTLE` (stream ready, response head, `SessionConnected`, park) + `FLOOD_BUDGET` + `DRAIN_BUDGET` + 3 × `SETTLE` (Ping ready, Ping, message after the flood) + `SETTLE` (connection outcome, failure path only) + `PRESENCE_WAIT` (teardown) | 5 + 15 + 20 + 15 + 15 + 15 + 5 + 10 = 100 s |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Streaming};

use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::join_session_response::ResponseType;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{JoinSessionRequest, JoinSessionResponse, PingRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::{live_service_server, production_server_builder};
use live_edit::presence::spawn_presence_subscriber;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    build_service, create_session, drain_tracker, join_header, join_message, AllowAllPermissions,
};

/// Bound on each single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(5);

/// One deadline across every join a test opens.
const JOIN_BUDGET: Duration = Duration::from_secs(15);

/// One deadline across the whole paced fan-out of the client-side test.
const FANOUT_BUDGET: Duration = Duration::from_secs(30);

/// One deadline across a whole drain: every held stream in the client-side
/// test, the observer's messages in the service-side test.
const DRAIN_BUDGET: Duration = Duration::from_secs(15);

/// One deadline across the whole paced flood of the service-side test.
#[cfg(feature = "test-fault-injection")]
const FLOOD_BUDGET: Duration = Duration::from_secs(15);

/// The metrics layer's service label.
const SERVICE_LABEL: &str = "test";

/// Bearer for every request; the interceptor requires one and the mock
/// permission client ignores it.
const BEARER: &str = "h2-small-frame-budget-test";

/// The budget h2 0.4.16–0.4.18 fixed for every connection, in bytes of
/// small-frame overhead.
const PRE_0419_BUDGET: usize = 25_600;

/// h2's small-frame threshold: a non-final DATA frame whose payload is under
/// this many bytes costs `threshold − payload` of the budget.
const SMALL_FRAME_THRESHOLD: usize = 256;

/// h2 0.4.19's budget for a tonic client on hyper's defaults: half of hyper's
/// 5 MiB client connection window.
const CLIENT_BUDGET_0419: usize = 5 * 1024 * 1024 / 2;

/// h2 0.4.19's budget for the service: half of hyper's 1 MiB server connection
/// window, which `production_server_builder` leaves unchanged.
const SERVICE_BUDGET_0419: usize = 1024 * 1024 / 2;

/// Size of every `PeerMessage` payload and `SessionMessageRequest` payload:
/// a big-endian `u32` index.
const PAYLOAD_LEN: usize = 4;

/// The DATA-frame payload of one `PeerMessage` fan-out frame: the 5-byte gRPC
/// prefix, then `JoinSessionResponse { peer_message: PeerMessage {
/// connection_id: <36-char UUID>, data: <4 bytes> } }` — 2 + 36 + 2 + 4 inner
/// bytes wrapped in a 2-byte field header.
const PEER_MESSAGE_FRAME_LEN: usize = 5 + 2 + (2 + 36) + (2 + PAYLOAD_LEN);

/// The DATA-frame payload of one flood frame in the service-side test: the
/// 5-byte gRPC prefix, then `JoinSessionRequest { message:
/// SessionMessageRequest { data: <4 bytes> } }`.
const FLOOD_FRAME_LEN: usize = 5 + 2 + (2 + PAYLOAD_LEN);

// ── Client side ──────────────────────────────────────────────────────────────

/// `JoinSession` streams held unread on the client's one connection.
const HELD_STREAMS: usize = 4;

/// `PeerMessage`s fanned out to every held stream.
const FANOUT_MESSAGES: u32 = 125;

/// Fanned-out messages left unread on the held connection. This is an upper
/// bound on its small DATA frames, not an exact count: pacing normally gives
/// each message its own frame, but a held stream's encoder can put two queued
/// messages into one frame.
const HELD_FRAMES: usize = HELD_STREAMS * FANOUT_MESSAGES as usize;

// At least 400 messages, more than 3× the frames that close a 0.4.16–0.4.18
// connection (4× today), and under 5 % of the 0.4.19 client budget. The counts
// are of messages, so the margin above 1× is what absorbs occasional batching
// into fewer frames: a recorded 0.4.17 run failed at message 26 of 125 on the
// first held stream, well inside it. Cutting the margin toward 1× risks a test
// that passes on 0.4.17 without detecting anything.
const _: () = {
    let kill_point = PRE_0419_BUDGET / (SMALL_FRAME_THRESHOLD - PEER_MESSAGE_FRAME_LEN) + 1;
    let overhead = HELD_FRAMES * (SMALL_FRAME_THRESHOLD - PEER_MESSAGE_FRAME_LEN);
    assert!(HELD_FRAMES >= 400);
    assert!(HELD_FRAMES > 3 * kill_point);
    assert!(overhead * 20 < CLIENT_BUDGET_0419);
};

// ── Service side ─────────────────────────────────────────────────────────────

/// Small `JoinSession` messages the service leaves unread, one per DATA frame.
const FLOOD_FRAMES: u32 = 400;

// At least 400 frames, more than 3× the frames that close a 0.4.16–0.4.18
// connection, and under a quarter of the 0.4.19 service budget.
const _: () = {
    let kill_point = PRE_0419_BUDGET / (SMALL_FRAME_THRESHOLD - FLOOD_FRAME_LEN) + 1;
    let overhead = FLOOD_FRAMES as usize * (SMALL_FRAME_THRESHOLD - FLOOD_FRAME_LEN);
    assert!(FLOOD_FRAMES >= 400);
    assert!(FLOOD_FRAMES as usize > 3 * kill_point);
    assert!(overhead * 4 < SERVICE_BUDGET_0419);
};

// =============================================================================
// Fixture
// =============================================================================

/// The live service behind the production HTTP/2 settings, over TCP, wired as
/// `main.rs` wires it: `MetricsLayer` as a server layer, the `LiveService`
/// built by `live_service_server`, and the Valkey-to-router presence bridge that
/// carries every published presence event to the local joins.
struct ProductionTransportServer {
    addr: SocketAddr,
    server_task: tokio::task::JoinHandle<()>,
    bridge_cancel: CancellationToken,
}

impl ProductionTransportServer {
    async fn start(tv: &TestValkey, svc: LiveEditService) -> Self {
        let bridge_cancel = CancellationToken::new();
        // Detached: cancelled on drop, and it holds no test state.
        drop(spawn_presence_subscriber(
            tv.client().subscriber.clone(),
            Arc::clone(&svc.presence_router),
            tv.key_schema().clone(),
            Arc::clone(&svc.metrics),
            bridge_cancel.clone(),
        ));

        let layer_metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
        // 30 s / 2 s: the production defaults.
        let metrics_layer = MetricsLayer::new(layer_metrics, SERVICE_LABEL, 30_000, 2_000);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind to a random port");
        let addr = listener.local_addr().expect("no local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();

        let server_task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder()
                .layer(metrics_layer)
                .add_service(live_service_server(svc))
                .serve_with_incoming(incoming)
                .await
                .expect("test server error");
        });
        tokio::time::timeout(SETTLE, ready_rx)
            .await
            .expect("the server did not signal readiness")
            .expect("the server task dropped its ready channel");

        Self {
            addr,
            server_task,
            bridge_cancel,
        }
    }
}

impl Drop for ProductionTransportServer {
    fn drop(&mut self) {
        self.server_task.abort();
        self.bridge_cancel.cancel();
    }
}

/// A tonic channel configured like the Rust client library: a connect
/// timeout, HTTP/2 keepalive every 30 s with a 5 s timeout, `TCP_NODELAY`, and
/// no flow-control window settings. One channel is one HTTP/2 connection.
async fn rust_client_like_channel(addr: SocketAddr) -> Channel {
    Endpoint::from_shared(format!("http://{addr}"))
        .expect("valid endpoint URI")
        .connect_timeout(Duration::from_secs(10))
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(5))
        .tcp_nodelay(true)
        .connect()
        .await
        .expect("failed to connect to the test server")
}

/// A plain tonic channel for the observer and sender connections.
async fn plain_channel(addr: SocketAddr) -> Channel {
    Endpoint::from_shared(format!("http://{addr}"))
        .expect("valid endpoint URI")
        .connect()
        .await
        .expect("failed to connect to the test server")
}

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {BEARER}")).expect("valid metadata"),
    );
    request
}

/// The payload of message `i`.
fn indexed_payload(i: u32) -> Vec<u8> {
    i.to_be_bytes().to_vec()
}

/// An open `JoinSession` over tonic whose `SessionConnected` has been read.
struct TonicJoin {
    tx: mpsc::Sender<JoinSessionRequest>,
    stream: Streaming<JoinSessionResponse>,
    connection_id: String,
}

/// Open a `JoinSession` on `channel` and read its `SessionConnected`. Once
/// that has arrived the join is subscribed to the session's presence events.
async fn tonic_join(channel: &Channel, session_id: &str, app: &str) -> TonicJoin {
    let (tx, rx) = mpsc::channel::<JoinSessionRequest>(16);
    tx.send(join_header(session_id, app))
        .await
        .expect("the request channel must accept the header");
    let mut client = LiveServiceClient::new(channel.clone());
    let mut stream = client
        .join_session(with_bearer(ReceiverStream::new(rx)))
        .await
        .unwrap_or_else(|s| panic!("{app}: the join must be admitted: {s:?}"))
        .into_inner();
    let first = stream
        .next()
        .await
        .unwrap_or_else(|| panic!("{app}: the join stream ended before SessionConnected"))
        .unwrap_or_else(|s| panic!("{app}: the join stream failed before SessionConnected: {s:?}"));
    let connection_id = match first.response_type {
        Some(ResponseType::Connected(c)) => c.connection_id,
        other => panic!("{app}: the first message must be SessionConnected; got {other:?}"),
    };
    TonicJoin {
        tx,
        stream,
        connection_id,
    }
}

/// The next `PeerMessage` from `from` on `stream`, skipping every other event.
/// `Err` carries the status the stream failed with.
async fn next_peer_message(
    stream: &mut Streaming<JoinSessionResponse>,
    from: &str,
) -> Result<Vec<u8>, tonic::Status> {
    loop {
        match stream.next().await {
            Some(Ok(JoinSessionResponse {
                response_type: Some(ResponseType::PeerMessage(m)),
            })) if m.connection_id == from => return Ok(m.data),
            Some(Ok(_)) => continue,
            Some(Err(status)) => return Err(status),
            None => panic!("the join stream ended OK while PeerMessages were still owed"),
        }
    }
}

/// The first `h2::Error` in `err`'s source chain.
fn h2_error_in<'a>(err: &'a (dyn std::error::Error + 'static)) -> Option<&'a h2::Error> {
    let mut current = Some(err);
    while let Some(e) = current {
        if let Some(h2_err) = e.downcast_ref::<h2::Error>() {
            return Some(h2_err);
        }
        current = e.source();
    }
    None
}

/// An `h2::Error` with the flags that say which side closed the connection.
fn describe_h2(err: &h2::Error) -> String {
    format!(
        "h2::Error `{err}`: reason={:?} is_go_away={} is_remote={} is_library={} is_reset={} \
         is_io={}",
        err.reason(),
        err.is_go_away(),
        err.is_remote(),
        err.is_library(),
        err.is_reset(),
        err.is_io(),
    )
}

/// A tonic status with the `h2::Error` behind it, if any.
fn describe_status(status: &tonic::Status) -> String {
    let h2 = std::error::Error::source(status)
        .and_then(h2_error_in)
        .map_or_else(
            || "no h2::Error in the source chain".to_owned(),
            describe_h2,
        );
    format!("status {:?} `{}`; {h2}", status.code(), status.message())
}

// =============================================================================
// Client side
// =============================================================================

/// A tonic client that multiplexes several `JoinSession` streams on one
/// connection and leaves them unread survives the service's presence fan-out:
/// the connection survives up to `HELD_FRAMES` unread small frames.
///
/// One channel configured like the Rust client library's holds `HELD_STREAMS`
/// joins, reads each one's `SessionConnected` (so each is subscribed) and then
/// reads nothing. From a
/// second connection a sender posts `FANOUT_MESSAGES` small messages, and the
/// service fans each one out to every held stream. The sends are paced by an
/// observer on the second connection that reads every message before the next
/// is sent. Once the observer has message `i`, the service has broadcast it,
/// and message `i + 1` follows a Valkey pub/sub round trip later. So one small
/// DATA frame per message is the normal case, not a guarantee. The observer's
/// receipt does not prove every held join has already forwarded `i`, and
/// tonic's encoder puts every message already queued on a stream into one
/// chunk, so under load two messages can share a frame. `HELD_FRAMES` therefore
/// bounds the frame count from above, and its margin over the 0.4.16–0.4.18
/// kill point covers that batching.
///
/// Every held stream must then yield exactly `FANOUT_MESSAGES` messages in
/// order with no error, stay open, and a `Ping` on the same channel must
/// succeed. On h2 0.4.16–0.4.18 the client's own h2 closes the connection
/// (a locally initiated GOAWAY `ENHANCE_YOUR_CALM`) after about 125 of the
/// frames, and every held stream fails with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tonic_client_with_unread_join_streams_survives_presence_fanout() {
    // ── Arrange ──────────────────────────────────────────────────────────────
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _token) = create_session(&tv, &svc, "h2-client-budget", true).await;
    let server = ProductionTransportServer::start(&tv, svc.clone()).await;

    let held_channel = rust_client_like_channel(server.addr).await;
    let peer_channel = plain_channel(server.addr).await;

    let (mut held, mut observer, sender) = tokio::time::timeout(JOIN_BUDGET, async {
        let mut held = Vec::with_capacity(HELD_STREAMS);
        for i in 0..HELD_STREAMS {
            held.push(tonic_join(&held_channel, &session_id, &format!("held-{i}")).await);
        }
        let observer = tonic_join(&peer_channel, &session_id, "observer").await;
        let sender = tonic_join(&peer_channel, &session_id, "sender").await;
        (held, observer, sender)
    })
    .await
    .unwrap_or_else(|_| panic!("the joins did not all connect within {JOIN_BUDGET:?}"));

    // The sender joined last, so its own stream carries only the membership
    // replay of the held streams and the observer (its own messages are
    // self-filtered); drain it so nothing piles up on the peer connection.
    let TonicJoin {
        tx: sender_tx,
        stream: mut sender_stream,
        connection_id: sender_cid,
    } = sender;
    let sender_drain = tokio::spawn(async move { while sender_stream.next().await.is_some() {} });

    // ── Act: paced fan-out to the unread held streams ────────────────────────
    tokio::time::timeout(FANOUT_BUDGET, async {
        for i in 0..FANOUT_MESSAGES {
            sender_tx
                .send(join_message(indexed_payload(i)))
                .await
                .expect("the sender's request channel must accept message");
            let seen = next_peer_message(&mut observer.stream, &sender_cid)
                .await
                .unwrap_or_else(|s| {
                    panic!(
                        "the observer's stream failed at message {i}: {}",
                        describe_status(&s)
                    )
                });
            assert_eq!(
                seen,
                indexed_payload(i),
                "the observer must see the sender's messages in order"
            );
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the paced fan-out did not finish within {FANOUT_BUDGET:?}"));

    // ── Assert ───────────────────────────────────────────────────────────────
    tokio::time::timeout(DRAIN_BUDGET, async {
        for (s, join) in held.iter_mut().enumerate() {
            for i in 0..FANOUT_MESSAGES {
                let data = next_peer_message(&mut join.stream, &sender_cid)
                    .await
                    .unwrap_or_else(|status| {
                        panic!(
                            "held stream {s} failed after {i} of {FANOUT_MESSAGES} PeerMessages \
                             ({HELD_FRAMES} small messages were left unread on its connection): \
                             {}",
                            describe_status(&status)
                        )
                    });
                assert_eq!(
                    data,
                    indexed_payload(i),
                    "held stream {s} must deliver the sender's messages in order"
                );
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the held streams did not drain within {DRAIN_BUDGET:?}"));

    let pong = tokio::time::timeout(
        SETTLE,
        LiveServiceClient::new(held_channel.clone()).ping(with_bearer(PingRequest {
            client_timestamp_microseconds_since_epoch: 7,
        })),
    )
    .await
    .unwrap_or_else(|_| panic!("Ping did not answer within {SETTLE:?}"))
    .unwrap_or_else(|s| panic!("Ping on the held channel failed: {}", describe_status(&s)));
    assert_eq!(
        pong.into_inner().client_timestamp_microseconds_since_epoch,
        7
    );

    for (s, join) in held.iter_mut().enumerate() {
        match join.stream.next().now_or_never() {
            None | Some(Some(Ok(_))) => {}
            Some(Some(Err(status))) => {
                panic!(
                    "held stream {s} failed after the fan-out: {}",
                    describe_status(&status)
                )
            }
            Some(None) => panic!("held stream {s} ended after the fan-out"),
        }
    }

    // ── Teardown ─────────────────────────────────────────────────────────────
    sender_drain.abort();
    drop(sender_tx);
    drop(observer);
    drop(held);
    svc.shutdown.cancel();
    drain_tracker(&svc).await;
    drop(server);
}

// =============================================================================
// Service side
// =============================================================================

/// Accumulates HTTP/2 DATA chunks and splits them into gRPC messages.
#[cfg(feature = "test-fault-injection")]
#[derive(Default)]
struct GrpcMessages {
    buf: bytes::BytesMut,
}

#[cfg(feature = "test-fault-injection")]
impl GrpcMessages {
    /// The next complete, uncompressed gRPC message's protobuf bytes.
    fn next_message(&mut self) -> Option<bytes::Bytes> {
        use bytes::Buf;
        if self.buf.len() < 5 {
            return None;
        }
        assert_eq!(self.buf[0], 0, "the service must not compress messages");
        let len = u32::from_be_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]) as usize;
        if self.buf.len() < 5 + len {
            return None;
        }
        self.buf.advance(5);
        Some(self.buf.split_to(len).freeze())
    }

    /// Read `recv` until one whole message is buffered, returning its capacity
    /// to the connection as it goes. `Ok(None)` at the end of the stream.
    async fn read(&mut self, recv: &mut h2::RecvStream) -> Result<Option<bytes::Bytes>, h2::Error> {
        loop {
            if let Some(message) = self.next_message() {
                return Ok(Some(message));
            }
            match recv.data().await {
                Some(Ok(chunk)) => {
                    let _ = recv.flow_control().release_capacity(chunk.len());
                    self.buf.extend_from_slice(&chunk);
                }
                Some(Err(e)) => return Err(e),
                None => return Ok(None),
            }
        }
    }
}

/// A gRPC request head for `method` on the live service, carrying the bearer.
#[cfg(feature = "test-fault-injection")]
fn grpc_request(addr: SocketAddr, method: &str) -> http::Request<()> {
    http::Request::builder()
        .method("POST")
        .uri(format!(
            "http://{addr}/nvidia.omniverse.live.v1alpha.LiveService/{method}"
        ))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("authorization", format!("Bearer {BEARER}"))
        .body(())
        .expect("valid request head")
}

/// Why the raw connection is gone, for a failure message: the connection
/// task's result once it has finished.
#[cfg(feature = "test-fault-injection")]
async fn connection_outcome(
    conn_task: &mut tokio::task::JoinHandle<Result<(), h2::Error>>,
) -> String {
    match tokio::time::timeout(SETTLE, conn_task).await {
        Ok(Ok(Ok(()))) => "the connection closed cleanly".to_owned(),
        Ok(Ok(Err(e))) => format!("the connection failed: {}", describe_h2(&e)),
        Ok(Err(join_err)) => format!("the connection task failed: {join_err}"),
        Err(_) => format!("the connection was still running after {SETTLE:?}"),
    }
}

/// The service's side of a connection survives small messages its handler has
/// not read yet: the connection survives `FLOOD_FRAMES` unread small frames.
///
/// A raw `h2` client, so the test decides the framing, opens a `JoinSession`
/// on the production-configured server. The join is parked at
/// `PresenceReplaySeam::before_first_send`, after `SessionConnected` and before
/// the loop that reads the client's messages. The client then sends
/// `FLOOD_FRAMES` `JoinSessionRequest` messages, each in its own non-final
/// DATA frame under 256 bytes and each followed by a PING round trip, and one
/// more round trip proves the service's h2 has received all of them. Only then
/// is the join released.
///
/// The connection must stay up: no GOAWAY, an observer on another connection
/// receives every message as a `PeerMessage` in order, a unary `Ping` on the
/// same connection succeeds, and the join stream still carries a message sent
/// after it. On h2 0.4.16–0.4.18 the service's h2 closes the connection after
/// about 106 of the frames with GOAWAY `ENHANCE_YOUR_CALM`, which the raw
/// client sees as a remote GOAWAY.
///
/// Needs `test-fault-injection` for the seam; without it this test is compiled
/// out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn service_connection_with_unread_small_join_messages_survives() {
    use live_edit::grpc::presence::PresenceReplaySeam;
    use live_edit::grpc::proto::{PingResponse, SessionConnected};
    use prost::Message;

    use super::helpers::encode_join_frame;

    // ── Arrange ──────────────────────────────────────────────────────────────
    let tv = TestValkey::setup().await;
    let seam = Arc::new(PresenceReplaySeam::new());
    let svc = LiveEditService {
        presence_replay_seam_for_test: Some(Arc::clone(&seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let (session_id, _token) = create_session(&tv, &svc, "h2-service-budget", true).await;
    let server = ProductionTransportServer::start(&tv, svc.clone()).await;

    // The observer joins first, with the gate disarmed, so it is never parked.
    let observer_channel = plain_channel(server.addr).await;
    let mut observer = tokio::time::timeout(
        JOIN_BUDGET,
        tonic_join(&observer_channel, &session_id, "observer"),
    )
    .await
    .unwrap_or_else(|_| panic!("the observer did not connect within {JOIN_BUDGET:?}"));

    let tcp = tokio::net::TcpStream::connect(server.addr)
        .await
        .expect("TCP connect to the test server");
    tcp.set_nodelay(true).expect("TCP_NODELAY");
    let (send_request, mut connection) = h2::client::handshake(tcp)
        .await
        .expect("HTTP/2 handshake with the test server");
    let mut ping_pong = connection
        .ping_pong()
        .expect("the connection's PingPong is taken once");
    let mut conn_task = tokio::spawn(connection);

    // ── Open the join and park it before its read loop ───────────────────────
    seam.before_first_send.arm();
    let mut send_request = tokio::time::timeout(SETTLE, send_request.ready())
        .await
        .expect("the connection did not become ready in time")
        .expect("the connection must accept a stream");
    let (response, mut join_body) = send_request
        .send_request(grpc_request(server.addr, "JoinSession"), false)
        .expect("JoinSession request head");
    join_body
        .send_data(
            encode_join_frame(&join_header(&session_id, "raw-h2")),
            false,
        )
        .expect("JoinSession header message");
    let response = tokio::time::timeout(SETTLE, response)
        .await
        .expect("the JoinSession response head did not arrive in time")
        .expect("JoinSession response head");
    assert_eq!(response.status(), http::StatusCode::OK);
    let mut join_recv = response.into_body();
    let mut join_messages = GrpcMessages::default();
    let connected = tokio::time::timeout(SETTLE, join_messages.read(&mut join_recv))
        .await
        .expect("SessionConnected did not arrive in time")
        .expect("the join stream failed before SessionConnected")
        .expect("the join stream ended before SessionConnected");
    let raw_cid = match JoinSessionResponse::decode(connected)
        .expect("a JoinSessionResponse")
        .response_type
    {
        Some(ResponseType::Connected(SessionConnected { connection_id, .. })) => connection_id,
        other => panic!("the first message must be SessionConnected; got {other:?}"),
    };
    tokio::time::timeout(SETTLE, seam.before_first_send.wait_until_parked())
        .await
        .expect("the join did not park before its read loop in time");

    // The join's remaining output (the observer's PeerJoined) is read and
    // discarded; the task ends only if the stream does.
    let join_drain = tokio::spawn(async move {
        loop {
            match join_messages.read(&mut join_recv).await {
                Ok(Some(_)) => continue,
                Ok(None) => return "ended OK".to_owned(),
                Err(e) => return describe_h2(&e),
            }
        }
    });

    // ── Act: small frames the parked join does not read ──────────────────────
    // One PING round trip after each frame, and one more after the last.
    //
    // The final round trip is the barrier that the service's h2 has received
    // every frame before the join is released. h2 writes a user PING ahead of
    // DATA queued in the same flush, so a PONG proves only that the frames
    // queued before its PING were written; the next PING goes out after that,
    // so its PONG proves the service read them.
    //
    // Pacing every frame is what makes a failure legible. A connection the
    // service closes mid-burst is closed with frames still unread, which resets
    // it, and the client then sees a bare I/O error in place of the GOAWAY. Paced,
    // the frame that trips the service's guard is the last thing on the wire,
    // so the client reads the GOAWAY and its reason.
    tokio::time::timeout(FLOOD_BUDGET, async {
        for i in 0..=FLOOD_FRAMES {
            if i < FLOOD_FRAMES {
                let frame = encode_join_frame(&join_message(indexed_payload(i)));
                assert_eq!(frame.len(), FLOOD_FRAME_LEN, "every flood frame is small");
                // A connection closed mid-flood fails the next send with a
                // local "inactive stream" error; the connection task holds the
                // cause.
                if let Err(e) = join_body.send_data(frame, false) {
                    panic!(
                        "queueing flood frame {i} failed ({}); {}",
                        describe_h2(&e),
                        connection_outcome(&mut conn_task).await
                    );
                }
            }
            if let Err(e) = ping_pong.ping(h2::Ping::opaque()).await {
                panic!(
                    "the PING after flood frame {i} of {FLOOD_FRAMES} failed ({}); {}",
                    describe_h2(&e),
                    connection_outcome(&mut conn_task).await
                );
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the paced flood did not finish within {FLOOD_BUDGET:?}"));
    seam.before_first_send.release();

    // ── Assert ───────────────────────────────────────────────────────────────
    tokio::time::timeout(DRAIN_BUDGET, async {
        for i in 0..FLOOD_FRAMES {
            let data = next_peer_message(&mut observer.stream, &raw_cid)
                .await
                .unwrap_or_else(|s| {
                    panic!(
                        "the observer's stream failed at message {i}: {}",
                        describe_status(&s)
                    )
                });
            assert_eq!(
                data,
                indexed_payload(i),
                "the service must read every unread message, in order"
            );
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "the observer did not receive all {FLOOD_FRAMES} messages within {DRAIN_BUDGET:?}; \
             connection task finished: {}, join stream finished: {}",
            conn_task.is_finished(),
            join_drain.is_finished()
        )
    });

    // A unary call on the same connection.
    let mut send_request = tokio::time::timeout(SETTLE, send_request.ready())
        .await
        .expect("the connection did not become ready for Ping in time")
        .expect("the connection must accept the Ping stream");
    let (ping_response, mut ping_body) = send_request
        .send_request(grpc_request(server.addr, "Ping"), false)
        .expect("Ping request head");
    let mut ping_frame = bytes::BytesMut::new();
    let ping = PingRequest {
        client_timestamp_microseconds_since_epoch: 11,
    }
    .encode_to_vec();
    ping_frame.extend_from_slice(&[0]);
    ping_frame.extend_from_slice(&(ping.len() as u32).to_be_bytes());
    ping_frame.extend_from_slice(&ping);
    ping_body
        .send_data(ping_frame.freeze(), true)
        .expect("Ping request body");
    let (pong, trailers) = tokio::time::timeout(SETTLE, async {
        let response = ping_response.await.expect("Ping response head");
        assert_eq!(response.status(), http::StatusCode::OK);
        let mut recv = response.into_body();
        let mut messages = GrpcMessages::default();
        let pong = messages
            .read(&mut recv)
            .await
            .expect("Ping response body")
            .expect("Ping must answer with a message");
        let trailers = recv
            .trailers()
            .await
            .expect("Ping trailers")
            .expect("Ping must end with trailers");
        (pong, trailers)
    })
    .await
    .unwrap_or_else(|_| panic!("Ping did not complete within {SETTLE:?}"));
    assert_eq!(
        trailers.get("grpc-status").map(|v| v.as_bytes()),
        Some(&b"0"[..]),
        "Ping must end OK; trailers {trailers:?}"
    );
    assert_eq!(
        PingResponse::decode(pong)
            .expect("a PingResponse")
            .client_timestamp_microseconds_since_epoch,
        11
    );

    // The join stream is still open and still read.
    join_body
        .send_data(
            encode_join_frame(&join_message(indexed_payload(FLOOD_FRAMES))),
            false,
        )
        .expect("a message after the flood");
    let after = tokio::time::timeout(SETTLE, next_peer_message(&mut observer.stream, &raw_cid))
        .await
        .unwrap_or_else(|_| panic!("the message after the flood did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|s| panic!("the observer's stream failed: {}", describe_status(&s)));
    assert_eq!(after, indexed_payload(FLOOD_FRAMES));
    if conn_task.is_finished() {
        panic!(
            "the raw connection must still be up; {}",
            connection_outcome(&mut conn_task).await
        );
    }
    if join_drain.is_finished() {
        panic!(
            "the join's response stream must still be open; it {}",
            join_drain.await.expect("the join drain task")
        );
    }

    // ── Teardown ─────────────────────────────────────────────────────────────
    join_drain.abort();
    drop(join_body);
    drop(send_request);
    conn_task.abort();
    drop(observer);
    svc.shutdown.cancel();
    drain_tracker(&svc).await;
    drop(server);
}
