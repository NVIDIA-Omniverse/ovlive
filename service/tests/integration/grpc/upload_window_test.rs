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

//! An upload parked at admission leaves its connection's other streams
//! flowing, and the service's small-frame budget follows the connection window
//! the waiter cap sizes (SDD §5.1, §5.4).
//!
//! An upload waiting for admission has not read its content, so up to one
//! stream window of it (1 MiB) sits unread in the service's h2. With a
//! connection window no larger than one stream window, one parked upload could
//! hold all of its connection's flow-control window, and every other stream on
//! that connection — a unary call, another upload, a live `JoinSession` —
//! could send no DATA while the upload waited.
//! `production_server_builder` now sizes the connection window at one stream
//! window per upload that can wait, plus one
//! (`transport::http2_connection_window_bytes`).
//!
//! * [`upload_parked_at_admission_leaves_its_connection_open_to_a_ping_and_another_upload`]
//!   — a raw `h2` client parks the gate's whole admission queue on one
//!   connection, each parked upload holding a full stream window of content
//!   the service has not read; a unary `Ping` and an upload of another session
//!   on the same connection must still complete, while the parked uploads
//!   still wait.
//! * [`service_connection_survives_unread_small_join_frames_past_a_one_stream_window_budget`]
//!   — h2 sizes its small-DATA-frame budget from the connection window (half of
//!   it, `transport::h2_small_frame_budget_bytes`), so the larger window moved
//!   the budget too. A raw `h2` client floods a parked `JoinSession` with more
//!   small frames than a one-stream-window budget allows, and the connection
//!   must survive.
//!
//! Recorded red runs, with `production_server_builder` setting the connection
//! window back to one stream window: the first test fails because its `Ping`
//! and the other upload do not complete within their bound, with send
//! capacity granted to the first parked upload only (1,048,411 bytes; 0 to
//! the other two); the second fails mid-flood when the service closes the
//! connection with GOAWAY `ENHANCE_YOUR_CALM` (`too_many_data_frames`) and the
//! client cannot queue flood frame 2,158. That shows each test catches that
//! window; it is not a claim about any other change to the windows.
//!
//! Both servers are built from `production_server_builder` at the gate's own
//! waiter cap, with `MetricsLayer` and the auth interceptor wired as in
//! `main.rs`, so a change to the service's HTTP/2 settings is tested here
//! rather than drifting past a test-only builder.
//!
//! The flood test needs the `test-fault-injection` feature (it parks the join
//! at `PresenceReplaySeam::before_first_send`); without it that test is
//! compiled out and the run covers only the parked-upload test.
//!
//! The raw-client helpers here (the gRPC request head, the message reader, the
//! `h2::Error` description) repeat the private ones in
//! `h2_small_frame_budget_test.rs`.
//!
//! # Wait budgets
//!
//! Every wait is a deadline, not a sleep; the healthy path pays milliseconds.
//! The parked uploads' 30 s admission wait is never waited out: each test ends
//! while they still wait. Worst case per test, every wait at its full budget,
//! against nextest's 120 s hang-killer:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | parked upload | `SETTLE` (server ready) + `SETTLE` (holder admitted) + 5 × `SETTLE` (a stream opened: 3 parked, the `Ping`, the other upload) + `SETTLE` (uploads parked) + `SETTLE` (send capacity) + `SETTLE` (PING barrier) + `SETTLE` (`Ping` and the other upload) | 11 × 5 = 55 s |
//! | flood | `SETTLE` (server ready) + `JOIN_BUDGET` + 4 × `SETTLE` (stream ready, response head, `SessionConnected`, park) + `FLOOD_BUDGET` + `DRAIN_BUDGET` + 3 × `SETTLE` (Ping ready, Ping, message after the flood) + `SETTLE` (connection outcome, failure path only) + `PRESENCE_WAIT` (teardown) | 5 + 10 + 20 + 25 + 20 + 15 + 5 + 10 = 110 s |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::{BufMut, Bytes};
use prost::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataValue;
use tonic::transport::Endpoint;
use tonic::Request;

use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{CreateLayerResponse, PingRequest, PingResponse};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::{
    live_service_server, production_server_builder, HTTP2_STREAM_WINDOW_BYTES,
};
use live_edit::grpc::upload_admission::{
    per_session_cap_for, UploadAdmissionConfig, UploadAdmissionGate, RECEIVE_ALLOWANCE_BYTES,
};
use live_edit::presence::spawn_presence_subscriber;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline, with_root};

use super::helpers::{
    build_service, create_session, make_create_data, make_create_header, AllowAllPermissions,
};

/// Bound on each single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(5);

/// The metrics layer's service label.
const SERVICE_LABEL: &str = "test";

/// Bearer for every request; the interceptor requires one and the mock
/// permission client ignores it.
const BEARER: &str = "upload-window-test";

/// The server's RPC deadline: above [`ADMISSION_WAIT`], so a parked upload is
/// still waiting for admission, not ended by its deadline, while a test runs.
const RPC_DEADLINE_MS: u64 = 60_000;

// =============================================================================
// Fixture
// =============================================================================

/// The live service behind the production HTTP/2 settings, over TCP, wired as
/// `main.rs` wires it: `MetricsLayer` as a server layer, the `LiveService`
/// built by `live_service_server`, and the Valkey-to-router presence bridge
/// that carries every published presence event to the local joins.
struct ProductionServer {
    addr: SocketAddr,
    server_task: tokio::task::JoinHandle<()>,
    bridge_cancel: CancellationToken,
}

impl ProductionServer {
    /// Serve `svc` from `production_server_builder` at its own gate's waiter
    /// cap, the cap `main.rs` passes.
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
        let metrics_layer = MetricsLayer::new(layer_metrics, SERVICE_LABEL, RPC_DEADLINE_MS, 2_000);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind to a random port");
        let addr = listener.local_addr().expect("no local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let max_waiters = svc.upload_admission.max_waiters();
        let server_task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder(max_waiters)
                .layer(metrics_layer)
                .add_service(live_service_server(
                    svc,
                    live_edit::grpc::hold::HoldGate::always_open(),
                ))
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

impl Drop for ProductionServer {
    fn drop(&mut self) {
        self.server_task.abort();
        self.bridge_cancel.cancel();
    }
}

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {BEARER}")).expect("valid metadata"),
    );
    request
}

/// One message as a gRPC frame: compression flag, big-endian length, bytes.
fn grpc_frame<T: Message>(msg: &T) -> Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0);
    buf.put_u32(u32::try_from(encoded.len()).expect("a test message fits a gRPC frame"));
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

/// A gRPC request head for `method` on the live service, carrying the bearer.
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

/// Accumulates HTTP/2 DATA chunks and splits them into gRPC messages.
#[derive(Default)]
struct GrpcMessages {
    buf: bytes::BytesMut,
}

impl GrpcMessages {
    /// The next complete, uncompressed gRPC message's protobuf bytes.
    fn next_message(&mut self) -> Option<Bytes> {
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
    async fn read(&mut self, recv: &mut h2::RecvStream) -> Result<Option<Bytes>, h2::Error> {
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

/// Why the raw connection is gone, for a failure message: the connection
/// task's result once it has finished.
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

/// A raw HTTP/2 client connection to the test server, so the test decides
/// the framing and the flow control.
struct RawConnection {
    send_request: h2::client::SendRequest<Bytes>,
    ping_pong: h2::PingPong,
    conn_task: tokio::task::JoinHandle<Result<(), h2::Error>>,
}

impl RawConnection {
    /// Handshake with `addr`. The send buffer holds two stream windows, so
    /// `SendStream::capacity` can report a whole stream window granted.
    async fn connect(addr: SocketAddr) -> Self {
        let tcp = tokio::net::TcpStream::connect(addr)
            .await
            .expect("TCP connect to the test server");
        tcp.set_nodelay(true).expect("TCP_NODELAY");
        let (send_request, mut connection) = h2::client::Builder::new()
            .max_send_buffer_size(2 * HTTP2_STREAM_WINDOW_BYTES as usize)
            .handshake::<_, Bytes>(tcp)
            .await
            .expect("HTTP/2 handshake with the test server");
        let ping_pong = connection
            .ping_pong()
            .expect("the connection's PingPong is taken once");
        let conn_task = tokio::spawn(connection);
        Self {
            send_request,
            ping_pong,
            conn_task,
        }
    }

    /// Open a stream for `method` on the live service: its head sent, its
    /// body open.
    async fn open(
        &mut self,
        addr: SocketAddr,
        method: &str,
    ) -> (h2::client::ResponseFuture, h2::SendStream<Bytes>) {
        let mut ready = tokio::time::timeout(SETTLE, self.send_request.clone().ready())
            .await
            .unwrap_or_else(|_| panic!("the connection did not accept a {method} stream in time"))
            .unwrap_or_else(|e| panic!("the connection must accept a {method} stream: {e}"));
        ready
            .send_request(grpc_request(addr, method), false)
            .unwrap_or_else(|e| panic!("{method} request head: {e}"))
    }
}

/// A unary or client-streaming call's answer read off a raw stream: the
/// response message's bytes and the `grpc-status` trailer.
async fn read_answer(response: h2::client::ResponseFuture) -> Result<(Bytes, String), String> {
    let response = response
        .await
        .map_err(|e| format!("response head: {}", describe_h2(&e)))?;
    if response.status() != http::StatusCode::OK {
        return Err(format!("HTTP status {}", response.status()));
    }
    let mut recv = response.into_body();
    let mut messages = GrpcMessages::default();
    let message = messages
        .read(&mut recv)
        .await
        .map_err(|e| format!("response body: {}", describe_h2(&e)))?;
    let trailers = recv
        .trailers()
        .await
        .map_err(|e| format!("trailers: {}", describe_h2(&e)))?
        .ok_or("the response ended without trailers")?;
    let status = trailers
        .get("grpc-status")
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .unwrap_or_default();
    match message {
        Some(message) => Ok((message, status)),
        None => Err(format!("no response message; trailers {trailers:?}")),
    }
}

// =============================================================================
// A parked upload's unread content leaves its connection's window room
// =============================================================================

/// Uploads the gate lets wait for admission: the queue the test parks whole
/// on one connection.
const PARKED: usize = 3;

/// The admission wait: far longer than the test, so every parked upload is
/// still waiting when the test ends.
const ADMISSION_WAIT: Duration = Duration::from_secs(30);

/// `4A`: a budget whose per-session receiving cap is one upload, so a second
/// upload of a session parks at admission while its first is still
/// receiving.
const BUDGET: u64 = 4 * RECEIVE_ALLOWANCE_BYTES;

/// A `CreateLayer` content chunk whose whole gRPC frame is `len` bytes.
fn chunk_frame_of_len(len: usize) -> Bytes {
    let overhead = grpc_frame(&make_create_data(vec![0; len - 64])).len() - (len - 64);
    let frame = grpc_frame(&make_create_data(vec![0; len - overhead]));
    assert_eq!(
        frame.len(),
        len,
        "fixture self-check: the chunk frame's size"
    );
    frame
}

/// A `CreateLayer` body: the layer's root section, holding one inline field.
fn small_create_ompf() -> Vec<u8> {
    delta_to_ompf_bytes(
        &with_root(delta_with_set_field_inline(1, "key", vec![7; 64], 0)),
        &[],
    )
}

/// Poll `cond` until it holds, failing after [`SETTLE`].
async fn settle_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Uploads parked at admission, each holding a full stream window of content
/// the service has not read, leave their connection's other streams flowing.
///
/// The gate's queue holds [`PARKED`] uploads and its budget gives each
/// session one receiving upload. Session S's one upload is admitted and
/// stalled mid-body on its own connection. Then, on ONE raw `h2` connection,
/// [`PARKED`] more uploads of S send their headers and park at admission,
/// filling the queue. Each is granted a whole stream window of send capacity
/// and sends that much content — a stream window less its header frame,
/// which the service read — which the service leaves unread while the upload
/// waits; a PING round trip after the content, and one more after that, as
/// in `h2_small_frame_budget_test`, is the barrier before the checks.
///
/// On the same connection a unary `Ping` and a `CreateLayer` of another
/// session T must both complete, OK, within [`SETTLE`], while all [`PARKED`]
/// uploads still wait and S's first upload still receives. With the
/// connection window one stream window, as hyper's default left it, the first
/// parked upload's content takes the connection's whole window, and no later
/// stream on it can send DATA while the uploads wait (recorded red run: the
/// first parked upload was granted 1,048,411 bytes of send capacity, the
/// other two none, and neither the `Ping` nor T's upload completed within
/// the bound).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upload_parked_at_admission_leaves_its_connection_open_to_a_ping_and_another_upload() {
    // ── Arrange ──────────────────────────────────────────────────────────────
    let tv = TestValkey::setup().await;
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let svc = LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes: BUDGET,
                wait: ADMISSION_WAIT,
                max_waiters: PARKED,
                // No size-cap floor, so the budget can stay small.
                max_upload_bytes: 0,
            },
            Some(Arc::clone(&base.metrics)),
        )),
        ..base
    };
    assert_eq!(
        per_session_cap_for(BUDGET),
        1,
        "fixture self-check: one receiving upload per session"
    );
    let gate = Arc::clone(&svc.upload_admission);
    let (session_s, _) = create_session(&tv, &svc, "upload-window-s", false).await;
    let (session_t, _) = create_session(&tv, &svc, "upload-window-t", false).await;
    let server = ProductionServer::start(&tv, svc.clone()).await;

    // S's one receiving upload, admitted and stalled mid-body on its own
    // connection.
    let holder_channel = Endpoint::from_shared(format!("http://{}", server.addr))
        .expect("valid endpoint URI")
        .connect()
        .await
        .expect("connect the holder's channel");
    let (holder_tx, holder_rx) = mpsc::channel(4);
    holder_tx
        .send(make_create_header(&session_s, "held"))
        .await
        .expect("the holder's header");
    holder_tx
        .send(make_create_data(vec![1; 16]))
        .await
        .expect("the holder's first chunk");
    let holder = tokio::spawn(async move {
        LiveServiceClient::new(holder_channel)
            .create_layer(with_bearer(ReceiverStream::new(holder_rx)))
            .await
    });
    settle_until("S's first upload admitted", || {
        gate.receiving_uploads(&session_s) == 1
    })
    .await;

    // ── Act: park the whole queue on one connection ──────────────────────────
    let mut conn = RawConnection::connect(server.addr).await;
    let mut parked = Vec::with_capacity(PARKED);
    for i in 0..PARKED {
        let (response, mut body) = conn.open(server.addr, "CreateLayer").await;
        let header = grpc_frame(&make_create_header(&session_s, &format!("parked-{i}")));
        let content = HTTP2_STREAM_WINDOW_BYTES as usize - header.len();
        body.send_data(header, false)
            .expect("a parked upload's header");
        parked.push((response, body, content));
    }
    settle_until("every upload parked at admission", || {
        gate.waiters() == PARKED
    })
    .await;

    // Each parked upload asks for the rest of its stream window and sends what
    // it is granted. One deadline across all of them; a shortfall is reported
    // with the checks below, not here.
    let capacity_deadline = tokio::time::Instant::now() + SETTLE;
    let mut granted = Vec::with_capacity(PARKED);
    for (i, (_, body, content)) in parked.iter_mut().enumerate() {
        body.reserve_capacity(*content);
        while body.capacity() < *content {
            match tokio::time::timeout_at(
                capacity_deadline,
                std::future::poll_fn(|cx| body.poll_capacity(cx)),
            )
            .await
            {
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(e))) => {
                    panic!("parked upload {i}'s stream failed: {}", describe_h2(&e))
                }
                Ok(None) => panic!("parked upload {i}'s stream closed"),
                Err(_) => break,
            }
        }
        granted.push(body.capacity());
        body.send_data(chunk_frame_of_len(*content), false)
            .expect("a parked upload's content");
    }
    tokio::time::timeout(SETTLE, async {
        for _ in 0..2 {
            conn.ping_pong
                .ping(h2::Ping::opaque())
                .await
                .unwrap_or_else(|e| panic!("the PING barrier failed: {}", describe_h2(&e)));
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the PING barrier did not complete within {SETTLE:?}"));

    // ── Assert: a Ping and another session's upload on the same connection ──
    let (ping_response, mut ping_body) = conn.open(server.addr, "Ping").await;
    ping_body
        .send_data(
            grpc_frame(&PingRequest {
                client_timestamp_microseconds_since_epoch: 11,
            }),
            true,
        )
        .expect("Ping request body");
    let (create_response, mut create_body) = conn.open(server.addr, "CreateLayer").await;
    create_body
        .send_data(grpc_frame(&make_create_header(&session_t, "other")), false)
        .expect("T's header");
    create_body
        .send_data(grpc_frame(&make_create_data(small_create_ompf())), true)
        .expect("T's content");
    let answers = tokio::time::timeout(SETTLE, async {
        tokio::join!(read_answer(ping_response), read_answer(create_response))
    })
    .await;
    let still_parked = gate.waiters();
    let (pong, created) = answers.unwrap_or_else(|_| {
        panic!(
            "a Ping and another session's upload on a connection whose uploads are parked at \
             admission did not complete within {SETTLE:?}: the parked uploads' unread content \
             stalls the connection's other streams. Send capacity granted to the parked \
             uploads: {granted:?} bytes, of a stream window less its header each"
        )
    });

    let (pong, status) = pong.unwrap_or_else(|e| panic!("Ping failed: {e}"));
    assert_eq!(status, "0", "Ping must end OK");
    assert_eq!(
        PingResponse::decode(pong)
            .expect("a PingResponse")
            .client_timestamp_microseconds_since_epoch,
        11
    );
    let (created, status) = created.unwrap_or_else(|e| panic!("T's CreateLayer failed: {e}"));
    assert_eq!(status, "0", "T's CreateLayer must end OK");
    assert_ne!(
        CreateLayerResponse::decode(created)
            .expect("a CreateLayerResponse")
            .object_id,
        0
    );
    assert_eq!(
        still_parked, PARKED,
        "every parked upload was still waiting for admission when the other calls completed"
    );
    assert_eq!(gate.receiving_uploads(&session_s), 1);
    for (i, ((_, _, content), granted)) in parked.iter().zip(&granted).enumerate() {
        assert_eq!(
            granted, content,
            "fixture self-check: parked upload {i} was granted its whole stream window, so its \
             content sat unread in the service's window"
        );
    }
    if conn.conn_task.is_finished() {
        panic!(
            "the raw connection must still be up; {}",
            connection_outcome(&mut conn.conn_task).await
        );
    }

    // ── Teardown ─────────────────────────────────────────────────────────────
    drop(parked);
    conn.conn_task.abort();
    drop(holder_tx);
    holder.abort();
    drop(server);
}

// =============================================================================
// The small-frame budget follows the connection window
// =============================================================================

/// The budget h2 0.4.19 gave the service's connections while their window was
/// one stream window: the window this test's flood must outlast.
#[cfg(feature = "test-fault-injection")]
const ONE_STREAM_WINDOW_BUDGET: usize =
    live_edit::grpc::transport::h2_small_frame_budget_bytes(HTTP2_STREAM_WINDOW_BYTES);

/// h2's small-frame threshold: a non-final DATA frame whose payload is under
/// this many bytes costs `threshold − payload` of the budget.
#[cfg(feature = "test-fault-injection")]
const SMALL_FRAME_THRESHOLD: usize = 256;

/// Size of every flood message's payload: a big-endian `u32` index.
#[cfg(feature = "test-fault-injection")]
const PAYLOAD_LEN: usize = 4;

/// The DATA-frame payload of one flood frame: the 5-byte gRPC prefix, then
/// `JoinSessionRequest { message: SessionMessageRequest { data: <4 bytes> } }`.
#[cfg(feature = "test-fault-injection")]
const FLOOD_FRAME_LEN: usize = 5 + 2 + (2 + PAYLOAD_LEN);

/// Small `JoinSession` messages the service leaves unread, one per DATA frame.
#[cfg(feature = "test-fault-injection")]
const FLOOD_FRAMES: u32 = 3_000;

/// One deadline across the one join the flood test opens on tonic.
#[cfg(feature = "test-fault-injection")]
const JOIN_BUDGET: Duration = Duration::from_secs(10);

/// One deadline across the whole paced flood.
#[cfg(feature = "test-fault-injection")]
const FLOOD_BUDGET: Duration = Duration::from_secs(25);

/// One deadline across the observer's drain of every flood message.
#[cfg(feature = "test-fault-injection")]
const DRAIN_BUDGET: Duration = Duration::from_secs(20);

// More frames than close a connection at the one-stream-window budget (2,158
// of these 13-byte frames, at 243 bytes of budget each), by a quarter, and
// under a quarter of the budget at the default waiter cap: the flood tells the
// two windows apart.
#[cfg(feature = "test-fault-injection")]
const _: () = {
    use live_edit::grpc::transport::{h2_small_frame_budget_bytes, http2_connection_window_bytes};
    use live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;
    let per_frame = SMALL_FRAME_THRESHOLD - FLOOD_FRAME_LEN;
    let kill_point = ONE_STREAM_WINDOW_BUDGET / per_frame + 1;
    let budget = h2_small_frame_budget_bytes(http2_connection_window_bytes(
        UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
    ));
    assert!(kill_point == 2_158);
    assert!(4 * FLOOD_FRAMES as usize > 5 * kill_point);
    assert!(FLOOD_FRAMES as usize * per_frame * 4 < budget);
};

/// The payload of message `i`.
#[cfg(feature = "test-fault-injection")]
fn indexed_payload(i: u32) -> Vec<u8> {
    i.to_be_bytes().to_vec()
}

/// The service's side of a connection survives more unread small frames than
/// a budget sized from a one-stream-window connection allows: the small-frame
/// budget follows the connection window the waiter cap sizes.
///
/// The service runs at the default waiter cap, so h2's budget is half of a
/// 65 MiB window. A raw `h2` client opens a `JoinSession`, parked at
/// `PresenceReplaySeam::before_first_send` — after `SessionConnected`, before
/// the loop that reads the client's messages — and sends [`FLOOD_FRAMES`]
/// `JoinSessionRequest` messages, each in its own non-final DATA frame under
/// 256 bytes and each followed by a PING round trip; one more round trip ends
/// the flood. Only then is the join released.
///
/// The connection must stay up: no GOAWAY, an observer on another connection
/// receives every message as a `PeerMessage` in order, a unary `Ping` on the
/// same connection succeeds, and the join stream still carries a message sent
/// after the flood. With the connection window one stream window the budget
/// is 512 KiB, and the service's h2 closes the connection with GOAWAY
/// `ENHANCE_YOUR_CALM` partway through the flood (recorded red run: the
/// client could not queue flood frame 2,158, the connection having failed
/// with that GOAWAY, `too_many_data_frames`).
///
/// Needs `test-fault-injection` for the seam; without it this test is
/// compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn service_connection_survives_unread_small_join_frames_past_a_one_stream_window_budget() {
    use futures::StreamExt;
    use live_edit::grpc::presence::PresenceReplaySeam;
    use live_edit::grpc::proto::join_session_response::ResponseType;
    use live_edit::grpc::proto::{JoinSessionRequest, JoinSessionResponse, SessionConnected};
    use live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;

    use super::helpers::{drain_tracker, encode_join_frame, join_header, join_message};

    /// The next `PeerMessage` from `from` on `stream`, skipping every other
    /// event. `Err` carries the status the stream failed with.
    async fn next_peer_message(
        stream: &mut tonic::Streaming<JoinSessionResponse>,
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

    // ── Arrange ──────────────────────────────────────────────────────────────
    let tv = TestValkey::setup().await;
    let seam = Arc::new(PresenceReplaySeam::new());
    let svc = LiveEditService {
        presence_replay_seam_for_test: Some(Arc::clone(&seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    assert_eq!(
        svc.upload_admission.max_waiters(),
        UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
        "fixture self-check: the server runs at the default waiter cap the flood is sized for"
    );
    let (session_id, _token) = create_session(&tv, &svc, "upload-window-flood", true).await;
    let server = ProductionServer::start(&tv, svc.clone()).await;

    // The observer joins first, with the gate disarmed, so it is never parked.
    let observer_channel = Endpoint::from_shared(format!("http://{}", server.addr))
        .expect("valid endpoint URI")
        .connect()
        .await
        .expect("connect the observer's channel");
    let mut observer = tokio::time::timeout(JOIN_BUDGET, async {
        let (tx, rx) = mpsc::channel::<JoinSessionRequest>(16);
        tx.send(join_header(&session_id, "observer"))
            .await
            .expect("the observer's header");
        let mut stream = LiveServiceClient::new(observer_channel)
            .join_session(with_bearer(ReceiverStream::new(rx)))
            .await
            .expect("the observer's join must be admitted")
            .into_inner();
        match stream.next().await {
            Some(Ok(JoinSessionResponse {
                response_type: Some(ResponseType::Connected(_)),
            })) => {}
            other => panic!("the observer's first message must be SessionConnected; got {other:?}"),
        }
        (tx, stream)
    })
    .await
    .unwrap_or_else(|_| panic!("the observer did not connect within {JOIN_BUDGET:?}"));

    let RawConnection {
        send_request,
        mut ping_pong,
        mut conn_task,
    } = RawConnection::connect(server.addr).await;

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
    // One PING round trip after each frame, and one more after the last, as in
    // `h2_small_frame_budget_test`: the final round trip is the barrier before
    // the join is released, and pacing every frame puts the frame that trips
    // the service's guard last on the wire, so the client reads the GOAWAY and
    // its reason instead of a reset.
    tokio::time::timeout(FLOOD_BUDGET, async {
        for i in 0..=FLOOD_FRAMES {
            if i < FLOOD_FRAMES {
                let frame = encode_join_frame(&join_message(indexed_payload(i)));
                assert_eq!(frame.len(), FLOOD_FRAME_LEN, "every flood frame is small");
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
    let (_observer_tx, observer_stream) = &mut observer;
    tokio::time::timeout(DRAIN_BUDGET, async {
        for i in 0..FLOOD_FRAMES {
            let data = next_peer_message(observer_stream, &raw_cid)
                .await
                .unwrap_or_else(|s| panic!("the observer's stream failed at message {i}: {s:?}"));
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
    ping_body
        .send_data(
            grpc_frame(&PingRequest {
                client_timestamp_microseconds_since_epoch: 13,
            }),
            true,
        )
        .expect("Ping request body");
    let (pong, status) = tokio::time::timeout(SETTLE, read_answer(ping_response))
        .await
        .unwrap_or_else(|_| panic!("Ping did not complete within {SETTLE:?}"))
        .unwrap_or_else(|e| panic!("Ping failed: {e}"));
    assert_eq!(status, "0", "Ping must end OK");
    assert_eq!(
        PingResponse::decode(pong)
            .expect("a PingResponse")
            .client_timestamp_microseconds_since_epoch,
        13
    );

    // The join stream is still open and still read.
    join_body
        .send_data(
            encode_join_frame(&join_message(indexed_payload(FLOOD_FRAMES))),
            false,
        )
        .expect("a message after the flood");
    let after = tokio::time::timeout(SETTLE, next_peer_message(observer_stream, &raw_cid))
        .await
        .unwrap_or_else(|_| panic!("the message after the flood did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|s| panic!("the observer's stream failed: {s:?}"));
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
