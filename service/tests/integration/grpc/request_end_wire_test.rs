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

//! The end-of-stream record over a real tonic / hyper / h2 server, for each
//! way an HTTP/2 request stream can end on the wire (SDD §5.4).
//!
//! `CreateLayer` and `UpdateLayer` commit only on a recorded END_STREAM
//! (`grpc::request_end`). That rests on facts about the libraries below the
//! handler, at the `Cargo.lock` versions (hyper 1.11.0, h2 0.4.19, tonic
//! 0.14.6): that h2's end-of-stream state survives hyper's and tonic's
//! wrappers, that hyper ends the body cleanly — not with an error — on a
//! `RST_STREAM(NO_ERROR)`, and that a reset arriving after END_STREAM leaves
//! the upload complete. This matrix pins them, so a dependency upgrade that
//! changes one fails here rather than silently reopening a truncated commit.
//!
//! The service under test is a probe behind [`AttestRequestEnd`], served by
//! `transport::production_server_builder` as the LiveService is. For each
//! request it hands the test a [`Call`] and spawns a task that owns the body —
//! so hyper dropping the service future on a reset does not drop the body —
//! waits for the test's go signal, drains the body and reports the record.
//! The test sends the row's frames from a raw h2 client, waits until the
//! server has processed them (a PING round trip: h2 processes a connection's
//! frames in order), then gives the go. In each reset row it first waits
//! until hyper has dropped the probe's call on that reset — the evidence the
//! reset reached the server, so the END_STREAM-then-reset rows cannot pass on
//! the END_STREAM alone. No Valkey.
//!
//! | Row | Record |
//! |---|---|
//! | HEADERS with END_STREAM | `Complete` |
//! | DATA, then END_STREAM | `Complete` |
//! | `RST_STREAM(CANCEL)` before END_STREAM | `ResetBeforeEnd` |
//! | `RST_STREAM(NO_ERROR)` before END_STREAM | `EndedWithoutEndStream` |
//! | `RST_STREAM(INTERNAL_ERROR)` before END_STREAM | `ResetBeforeEnd` |
//! | TCP close mid-body | `ResetBeforeEnd` |
//! | END_STREAM, then `RST_STREAM(CANCEL)` | `Complete` |
//! | END_STREAM, then `RST_STREAM(NO_ERROR)` | `Complete` |
//!
//! The server's keepalive `GOAWAY(NO_ERROR)` is the `EndedWithoutEndStream`
//! row's other source; it takes the 35 s production keepalive to reach, so it
//! is covered once, end to end, by `upload_admission_test::cancelled_upload`.
//!
//! The same wrapper's second record, `RequestBodyProgress` — when the body
//! last carried bytes, which the JoinSession client keepalive reads (SDD
//! §13.2) — is pinned here too, by a second probe that reads the body one
//! frame at a time: the request head stamps nothing, each real DATA frame
//! moves the stamp forward when it is read, and the stream's end leaves it
//! where the last bytes put it. That an opted-in join served by
//! `live_service_server` is armed by it is pinned end to end in
//! `presence_keepalive_test`.
//!
//! Every wait is bounded by one [`TEST_BUDGET`] deadline per test; the
//! healthy path takes milliseconds.

use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use tokio::sync::{mpsc, oneshot};
use tonic::server::NamedService;
use tower::Service;

use live_edit::grpc::request_end::{
    AttestRequestEnd, RequestBodyProgress, RequestStreamEnd, StreamEnding,
};
use live_edit::grpc::transport::production_server_builder;

/// One test's whole wait budget. Unreachable on a healthy run.
const TEST_BUDGET: Duration = Duration::from_secs(30);

/// The path the probe serves under.
const PROBE_PATH: &str = "/test.EndProbe/Upload";

async fn settled<T>(deadline: tokio::time::Instant, what: &str, fut: impl Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not happen within the test's {TEST_BUDGET:?}"))
}

// ── The probe ────────────────────────────────────────────────────────────────

/// One request the probe received: the go signal for its drain, the record
/// it reports once drained, and a signal that fires if hyper drops the call
/// before it answers — what hyper does when the client resets the stream, so
/// the evidence that a reset reached the server.
struct Call {
    go: oneshot::Sender<()>,
    ended: oneshot::Receiver<Option<StreamEnding>>,
    dropped: oneshot::Receiver<()>,
}

impl Call {
    /// Wait until hyper has dropped the call on the client's reset.
    async fn dropped_by_the_reset(&mut self, deadline: tokio::time::Instant) {
        settled(deadline, "hyper dropping the call", &mut self.dropped)
            .await
            .expect("hyper dropped the call on the reset; it was not answered first");
    }
}

/// Fires its signal if dropped while armed.
struct DropSignal(Option<oneshot::Sender<()>>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        if let Some(signal) = self.0.take() {
            let _ = signal.send(());
        }
    }
}

/// A service that hands each request to the test as a [`Call`].
#[derive(Clone)]
struct EndProbe {
    calls: mpsc::UnboundedSender<Call>,
}

impl NamedService for EndProbe {
    const NAME: &'static str = "test.EndProbe";
}

impl Service<http::Request<tonic::body::Body>> for EndProbe {
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Infallible>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
        let (go_tx, go_rx) = oneshot::channel();
        let (ended_tx, ended_rx) = oneshot::channel();
        let (done_tx, done_rx) = oneshot::channel::<()>();
        let (dropped_tx, dropped_rx) = oneshot::channel::<()>();
        let _ = self.calls.send(Call {
            go: go_tx,
            ended: ended_rx,
            dropped: dropped_rx,
        });
        // The body lives in its own task: hyper drops this service's future
        // when the client resets the stream, and the drain must outlive that.
        tokio::spawn(async move {
            let end = request.extensions().get::<RequestStreamEnd>().cloned();
            let mut body = request.into_body();
            let _ = go_rx.await;
            while let Some(Ok(_)) = body.frame().await {}
            let _ = ended_tx.send(end.and_then(|end| end.ending()));
            let _ = done_tx.send(());
        });
        Box::pin(async move {
            let mut armed = DropSignal(Some(dropped_tx));
            let _ = done_rx.await;
            armed.0 = None;
            Ok(http::Response::new(tonic::body::Body::empty()))
        })
    }
}

/// A server with the production HTTP/2 settings serving the probe behind
/// [`AttestRequestEnd`]; dropped with the test.
struct ProbeServer {
    addr: SocketAddr,
    calls: mpsc::UnboundedReceiver<Call>,
    task: tokio::task::JoinHandle<()>,
}

impl ProbeServer {
    async fn start(deadline: tokio::time::Instant) -> Self {
        let (calls_tx, calls) = mpsc::unbounded_channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a random port");
        let addr = listener.local_addr().expect("local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder()
                .add_service(AttestRequestEnd::new(EndProbe { calls: calls_tx }))
                .serve_with_incoming(incoming)
                .await
                .expect("probe server error");
        });
        settled(deadline, "server readiness", ready_rx)
            .await
            .expect("the server task dropped its ready channel");
        Self { addr, calls, task }
    }

    /// The next request the probe received.
    async fn call(&mut self, deadline: tokio::time::Instant) -> Call {
        settled(deadline, "the probe called", self.calls.recv())
            .await
            .expect("the probe is alive")
    }
}

impl Drop for ProbeServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ── The raw client ───────────────────────────────────────────────────────────

/// A raw h2 client connection to the probe server.
struct WireClient {
    send_request: h2::client::SendRequest<Bytes>,
    ping_pong: h2::PingPong,
    driver: tokio::task::JoinHandle<()>,
    addr: SocketAddr,
}

impl WireClient {
    async fn connect(addr: SocketAddr, deadline: tokio::time::Instant) -> Self {
        let tcp = tokio::net::TcpStream::connect(addr)
            .await
            .expect("TCP connect");
        let (send_request, mut connection) =
            settled(deadline, "the h2 handshake", h2::client::handshake(tcp))
                .await
                .expect("h2 handshake");
        let ping_pong = connection.ping_pong().expect("the first ping_pong handle");
        let driver = tokio::spawn(async move {
            let _ = connection.await;
        });
        Self {
            send_request,
            ping_pong,
            driver,
            addr,
        }
    }

    /// Send a request head for the probe, with END_STREAM if `end_of_stream`.
    async fn start(
        &mut self,
        end_of_stream: bool,
        deadline: tokio::time::Instant,
    ) -> h2::SendStream<Bytes> {
        let mut ready = settled(
            deadline,
            "the connection ready",
            self.send_request.clone().ready(),
        )
        .await
        .expect("the connection accepts a stream");
        let head = http::Request::builder()
            .method("POST")
            .uri(format!("http://{}{PROBE_PATH}", self.addr))
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .body(())
            .expect("valid head");
        let (_response, body) = ready
            .send_request(head, end_of_stream)
            .expect("the request head");
        body
    }

    /// A PING round trip: once its PONG is back, the server has processed
    /// every frame sent before it.
    async fn round_trip(&mut self, deadline: tokio::time::Instant) {
        settled(
            deadline,
            "the PING round trip",
            self.ping_pong.ping(h2::Ping::opaque()),
        )
        .await
        .expect("the server answers the PING");
    }

    /// Close the TCP connection, without a GOAWAY or a reset: stop the
    /// driver and wait until it has dropped the connection.
    async fn close(mut self, deadline: tokio::time::Instant) {
        self.driver.abort();
        let driver = &mut self.driver;
        let stopped = settled(deadline, "the connection driver stopped", driver).await;
        assert!(
            stopped.is_err_and(|e| e.is_cancelled()),
            "the connection must still have been open"
        );
    }
}

impl Drop for WireClient {
    fn drop(&mut self) {
        self.driver.abort();
    }
}

/// One DATA frame's worth of request body.
fn payload() -> Bytes {
    Bytes::from_static(&[0, 0, 0, 0, 3, 1, 2, 3])
}

/// Give `call` its go and return the record its drained body reports.
async fn drained(call: Call, deadline: tokio::time::Instant) -> StreamEnding {
    call.go.send(()).expect("the probe's drain task is waiting");
    settled(deadline, "the probe's drain", call.ended)
        .await
        .expect("the drain reports")
        .expect("the request carries a written record once its body is drained")
}

// ── The matrix ───────────────────────────────────────────────────────────────

/// A stream whose body is sent as DATA, then ended by `ending` once the
/// server has accepted it, drained after the server has processed it.
async fn after_data(ending: impl FnOnce(&mut h2::SendStream<Bytes>)) -> StreamEnding {
    after_data_then(ending, false).await
}

/// [`after_data`] for an `ending` that resets the stream: also waits until
/// hyper has dropped the call on that reset, before the drain.
async fn reset_after_data(reason: h2::Reason) -> StreamEnding {
    after_data_then(|body| body.send_reset(reason), true).await
}

async fn after_data_then(
    ending: impl FnOnce(&mut h2::SendStream<Bytes>),
    resets: bool,
) -> StreamEnding {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let mut server = ProbeServer::start(deadline).await;
    let mut client = WireClient::connect(server.addr, deadline).await;
    let mut body = client.start(false, deadline).await;
    body.send_data(payload(), false).expect("a DATA frame");
    let mut call = server.call(deadline).await;
    client.round_trip(deadline).await;

    ending(&mut body);
    client.round_trip(deadline).await;
    if resets {
        call.dropped_by_the_reset(deadline).await;
    }
    drained(call, deadline).await
}

/// A stream ended with END_STREAM on its last DATA frame, then reset with
/// `reason` once the server has processed the END_STREAM.
async fn reset_after_end_stream(reason: h2::Reason) -> StreamEnding {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let mut server = ProbeServer::start(deadline).await;
    let mut client = WireClient::connect(server.addr, deadline).await;
    let mut body = client.start(false, deadline).await;
    body.send_data(payload(), true)
        .expect("a DATA frame with END_STREAM");
    let mut call = server.call(deadline).await;
    // The END_STREAM is processed before the reset is sent: h2 drops frames
    // still queued for a stream it resets.
    client.round_trip(deadline).await;

    body.send_reset(reason);
    client.round_trip(deadline).await;
    // The reset reached the server, so the row is not passing on an
    // END_STREAM alone.
    call.dropped_by_the_reset(deadline).await;
    drained(call, deadline).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn headers_with_end_stream_records_complete() {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let mut server = ProbeServer::start(deadline).await;
    let mut client = WireClient::connect(server.addr, deadline).await;
    let _body = client.start(true, deadline).await;
    let call = server.call(deadline).await;
    client.round_trip(deadline).await;
    assert_eq!(drained(call, deadline).await, StreamEnding::Complete);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_then_end_stream_records_complete() {
    let ending = after_data(|body| {
        body.send_data(payload(), true)
            .expect("a DATA frame with END_STREAM")
    })
    .await;
    assert_eq!(ending, StreamEnding::Complete);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rst_stream_cancel_before_end_stream_records_reset_before_end() {
    let ending = reset_after_data(h2::Reason::CANCEL).await;
    assert!(
        matches!(ending, StreamEnding::ResetBeforeEnd(_)),
        "a client's CANCEL reaches the body as an error; got {ending:?}"
    );
}

/// hyper ends the body cleanly on a `RST_STREAM(NO_ERROR)` — Envoy v1.36's
/// upstream reset for a client's cancel — so only END_STREAM's absence
/// tells it from a finished upload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rst_stream_no_error_before_end_stream_records_ended_without_end_stream() {
    let ending = reset_after_data(h2::Reason::NO_ERROR).await;
    assert_eq!(ending, StreamEnding::EndedWithoutEndStream);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rst_stream_internal_error_before_end_stream_records_reset_before_end() {
    let ending = reset_after_data(h2::Reason::INTERNAL_ERROR).await;
    assert!(
        matches!(ending, StreamEnding::ResetBeforeEnd(_)),
        "got {ending:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_close_mid_body_records_reset_before_end() {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let mut server = ProbeServer::start(deadline).await;
    let mut client = WireClient::connect(server.addr, deadline).await;
    let mut body = client.start(false, deadline).await;
    body.send_data(payload(), false).expect("a DATA frame");
    let call = server.call(deadline).await;
    client.round_trip(deadline).await;

    // Close the connection while the stream's handle is still held: a
    // dropped handle would have h2 reset the stream with `CANCEL` first.
    client.close(deadline).await;
    drop(body);
    let ending = drained(call, deadline).await;
    assert!(
        matches!(ending, StreamEnding::ResetBeforeEnd(_)),
        "a lost connection reaches the body as an error; got {ending:?}"
    );
}

/// A reset after END_STREAM cancels the call, not the upload: the body the
/// client finished is still complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_stream_then_rst_stream_cancel_records_complete() {
    assert_eq!(
        reset_after_end_stream(h2::Reason::CANCEL).await,
        StreamEnding::Complete
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_stream_then_rst_stream_no_error_records_complete() {
    assert_eq!(
        reset_after_end_stream(h2::Reason::NO_ERROR).await,
        StreamEnding::Complete
    );
}

// ── The body-progress record ─────────────────────────────────────────────────

/// What the progress probe's body read returned, or the request head.
#[derive(Debug, PartialEq, Eq)]
enum Seen {
    /// The request arrived, carrying a progress record or not.
    Head {
        has_record: bool,
    },
    /// A DATA frame of this many bytes.
    Data(usize),
    Trailers,
    /// The body ended (`None`).
    End,
    /// The body failed.
    Error,
}

/// One report of the progress probe: what it saw, and the record's stamp
/// right after (`None` if nothing has stamped it, or there is no record).
#[derive(Debug)]
struct ProgressReport {
    seen: Seen,
    last_data_at: Option<tokio::time::Instant>,
}

/// A service that reads each request's body one frame at a time, reporting
/// the body-progress record after the head and after every frame.
#[derive(Clone)]
struct ProgressProbe {
    reports: mpsc::UnboundedSender<ProgressReport>,
}

/// Served under the same name as [`EndProbe`], so [`WireClient::start`]'s
/// [`PROBE_PATH`] reaches it; each test serves only one of the two.
impl NamedService for ProgressProbe {
    const NAME: &'static str = "test.EndProbe";
}

impl Service<http::Request<tonic::body::Body>> for ProgressProbe {
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Infallible>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
        let reports = self.reports.clone();
        Box::pin(async move {
            let progress = request.extensions().get::<RequestBodyProgress>().cloned();
            let stamp = || {
                progress
                    .as_ref()
                    .and_then(RequestBodyProgress::last_data_at)
            };
            let _ = reports.send(ProgressReport {
                seen: Seen::Head {
                    has_record: progress.is_some(),
                },
                last_data_at: stamp(),
            });
            let mut body = request.into_body();
            loop {
                let seen = match body.frame().await {
                    Some(Ok(frame)) => match frame.data_ref() {
                        Some(data) => Seen::Data(data.len()),
                        None => Seen::Trailers,
                    },
                    Some(Err(_)) => Seen::Error,
                    None => Seen::End,
                };
                let done = matches!(seen, Seen::End | Seen::Error);
                let _ = reports.send(ProgressReport {
                    seen,
                    last_data_at: stamp(),
                });
                if done {
                    break;
                }
            }
            Ok(http::Response::new(tonic::body::Body::empty()))
        })
    }
}

/// The next report of the progress probe.
async fn next_report(
    reports: &mut mpsc::UnboundedReceiver<ProgressReport>,
    deadline: tokio::time::Instant,
) -> ProgressReport {
    settled(deadline, "the progress probe's report", reports.recv())
        .await
        .expect("the progress probe is alive")
}

/// Over a real HTTP/2 stream, the record the production wrapper attaches is
/// there from the request head, unstamped; each DATA frame stamps it when
/// the server reads it — never before the client sent it — and a later frame
/// moves it forward; the stream's end (an empty DATA frame with END_STREAM)
/// leaves it at the last bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_data_frame_read_moves_the_body_progress_record_forward() {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let (reports_tx, mut reports) = mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a random port");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        production_server_builder()
            .add_service(AttestRequestEnd::new(ProgressProbe {
                reports: reports_tx,
            }))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("progress probe server error");
    });
    let mut client = WireClient::connect(addr, deadline).await;
    let mut body = client.start(false, deadline).await;

    let head = next_report(&mut reports, deadline).await;
    assert_eq!(
        head.seen,
        Seen::Head { has_record: true },
        "the wrapper attaches the progress record to every request"
    );
    assert_eq!(head.last_data_at, None, "the request head stamps nothing");

    let before_first = tokio::time::Instant::now();
    body.send_data(payload(), false).expect("a DATA frame");
    let first = next_report(&mut reports, deadline).await;
    assert_eq!(first.seen, Seen::Data(payload().len()));
    let first_at = first
        .last_data_at
        .expect("a DATA frame read stamps the record");
    assert!(
        first_at >= before_first,
        "stamped when read, never before the bytes were sent"
    );

    let before_second = tokio::time::Instant::now();
    body.send_data(payload(), false)
        .expect("a second DATA frame");
    let second = next_report(&mut reports, deadline).await;
    assert_eq!(second.seen, Seen::Data(payload().len()));
    let second_at = second
        .last_data_at
        .expect("a DATA frame read stamps the record");
    assert!(
        before_second >= first_at && second_at >= before_second,
        "a later DATA frame moves the stamp forward: first {first_at:?}, sent after \
         {before_second:?}, second {second_at:?}"
    );

    body.send_data(Bytes::new(), true)
        .expect("an empty DATA frame with END_STREAM");
    loop {
        let report = next_report(&mut reports, deadline).await;
        assert!(
            matches!(report.seen, Seen::Data(0) | Seen::End),
            "only the empty frame and the end follow; got {report:?}"
        );
        assert_eq!(
            report.last_data_at,
            Some(second_at),
            "no bytes, no stamp: {report:?}"
        );
        if report.seen == Seen::End {
            break;
        }
    }
    server.abort();
}
