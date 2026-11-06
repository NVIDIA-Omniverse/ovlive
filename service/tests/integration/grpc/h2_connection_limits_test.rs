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

//! The per-connection HTTP/2 limits `transport::production_server_builder`
//! sets, observed on the wire of a real tonic / hyper / h2 server (SDD §5.1).
//!
//! - **The local-error-reset limit.** h2 answers a stream error it detects
//!   itself with RST_STREAM and counts it, over the connection's whole life.
//!   tonic's builder defaults that limit to `None`, which hyper and h2 read as
//!   no limit at all (RUSTSEC-2024-0003), so the builder sets
//!   [`HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS`]. The test drives exactly that
//!   many malformed requests, each reset `PROTOCOL_ERROR`, and the next one
//!   closes the connection with GOAWAY `ENHANCE_YOUR_CALM`
//!   "too_many_internal_resets" and the one h2 WARN operators see. A limit one
//!   lower closes the connection a request early; a limit one higher, or none,
//!   resets the extra request like the others and leaves the connection open.
//! - **The advertised SETTINGS.** The server's first SETTINGS frame, read off
//!   a raw socket because `h2::client` exposes no accessor for the peer's
//!   frame size: `SETTINGS_MAX_FRAME_SIZE` is [`HTTP2_MAX_FRAME_BYTES`],
//!   `SETTINGS_INITIAL_WINDOW_SIZE` is [`HTTP2_STREAM_WINDOW_BYTES`], and no
//!   `SETTINGS_MAX_CONCURRENT_STREAMS` is advertised.
//!
//! The malformed request is a HEADERS frame with END_STREAM and
//! `content-length: 1` (RFC 9113 §8.1.1). h2 resets it while decoding the
//! frame, before hyper or tonic sees a request, so the health service the
//! path routes to never runs, and the test needs no Valkey.
//! The requests go one at a time: the server's receive buffer is then empty
//! when it closes the connection, so the close is a clean FIN after the
//! GOAWAY rather than a TCP reset that could lose it.
//!
//! The reset-limit test makes 10,241 localhost round trips, so it runs for
//! over a second; that many is what pinning the exact limit takes.
//! Every wait in each test is bounded by one [`TEST_BUDGET`] deadline, under
//! nextest's 120 s hang-killer.

use std::net::SocketAddr;
use std::time::Duration;

use h2::Reason;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::Instant;

use live_edit::grpc::transport::{
    production_server_builder, HTTP2_MAX_FRAME_BYTES, HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS,
    HTTP2_STREAM_WINDOW_BYTES,
};
use live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;

use super::log_capture::ThreadCapture;

/// One test's whole wait budget. Unreachable on a healthy run.
const TEST_BUDGET: Duration = Duration::from_secs(60);

/// The path the malformed requests target: a real route on the server, so a
/// request h2 let through would reach a service rather than a 404.
const HEALTH_CHECK_PATH: &str = "/grpc.health.v1.Health/Check";

/// The fixed text of h2's WARN when the limit is reached; the limit follows
/// in parentheses.
const RESET_LIMIT_WARN: &str = "locally-reset streams reached limit";

/// The debug data of the GOAWAY h2 sends when the limit is reached.
const RESET_LIMIT_GOAWAY_DEBUG_DATA: &str = "too_many_internal_resets";

/// The HTTP/2 connection preface a client sends first (RFC 9113 §3.4).
const CLIENT_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// An empty SETTINGS frame: length 0, type 0x4, no flags, stream 0.
const EMPTY_SETTINGS_FRAME: [u8; 9] = [0, 0, 0, 0x4, 0, 0, 0, 0, 0];

/// SETTINGS frame type and its ACK flag (RFC 9113 §6.5).
const FRAME_TYPE_SETTINGS: u8 = 0x4;
const SETTINGS_FLAG_ACK: u8 = 0x1;

/// SETTINGS parameter identifiers (RFC 9113 §6.5.2).
const SETTINGS_MAX_CONCURRENT_STREAMS: u16 = 0x3;
const SETTINGS_INITIAL_WINDOW_SIZE: u16 = 0x4;
const SETTINGS_MAX_FRAME_SIZE: u16 = 0x5;

/// Await `fut` against the test's single deadline.
async fn settled<T>(deadline: Instant, what: &str, fut: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not happen within the test's {TEST_BUDGET:?}"))
}

/// Serve the health service from `production_server_builder` on a random
/// local port. The listener is bound before the server task starts, so a
/// connection made at once waits in the accept backlog.
async fn spawn_production_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a random port");
    let addr = listener.local_addr().expect("local addr");
    let (_health_reporter, health_service) = tonic_health::server::health_reporter();
    let server = tokio::spawn(async move {
        production_server_builder(UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT)
            .add_service(health_service)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("test server error");
    });
    (addr, server)
}

/// A request h2 must reset `PROTOCOL_ERROR` on arrival: it is sent as one
/// HEADERS frame with END_STREAM, yet declares a one-byte body.
fn malformed_request(addr: SocketAddr) -> http::Request<()> {
    http::Request::builder()
        .method(http::Method::POST)
        .uri(format!("http://{addr}{HEALTH_CHECK_PATH}"))
        .header(http::header::CONTENT_TYPE, "application/grpc")
        .header(http::header::TE, "trailers")
        .header(http::header::CONTENT_LENGTH, "1")
        .body(())
        .expect("a valid request head")
}

/// Send one malformed request on the connection and return its response
/// future's outcome.
async fn send_malformed(
    send_request: &h2::client::SendRequest<bytes::Bytes>,
    addr: SocketAddr,
    deadline: Instant,
    n: usize,
) -> Result<http::Response<h2::RecvStream>, h2::Error> {
    let mut ready = settled(
        deadline,
        &format!("the connection accepting request {n}"),
        send_request.clone().ready(),
    )
    .await
    .unwrap_or_else(|e| panic!("the connection must accept request {n}: {e}"));
    let (response, _) = ready
        .send_request(malformed_request(addr), true)
        .unwrap_or_else(|e| panic!("request {n}'s head: {e}"));
    settled(deadline, &format!("request {n}'s answer"), response).await
}

/// The value pin: the limit is ten times h2's and hyper's own default of
/// 1,024, a decision recorded with its reasons in SDD §5.1. The wire test
/// below follows the constant, so this is what fails on a silent change.
#[test]
fn http2_max_local_error_reset_streams_is_ten_times_the_library_default() {
    assert_eq!(
        HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS, 10_240,
        "the local-error-reset limit was decided at 10 × the library default of 1,024 \
         (SDD §5.1); changing it is a design decision, not a tuning edit"
    );
}

/// Over one connection, each of the first
/// [`HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS`] malformed requests is reset
/// `PROTOCOL_ERROR` by the server; the next one closes the connection with
/// GOAWAY `ENHANCE_YOUR_CALM` "too_many_internal_resets", and h2 logs one
/// WARN naming the limit.
///
/// Current-thread runtime: h2's WARN is emitted in tonic's spawned connection
/// task, which only a thread-local capture on the one runtime thread sees.
#[tokio::test]
async fn production_server_past_its_local_error_reset_limit_closes_the_connection_with_enhance_your_calm(
) {
    let capture = ThreadCapture::start();
    let deadline = Instant::now() + TEST_BUDGET;
    let (addr, server) = spawn_production_server().await;
    let tcp = settled(
        deadline,
        "TCP connect",
        tokio::net::TcpStream::connect(addr),
    )
    .await
    .expect("TCP connect to the test server");
    tcp.set_nodelay(true).expect("TCP_NODELAY");
    let (send_request, connection) =
        settled(deadline, "the HTTP/2 handshake", h2::client::handshake(tcp))
            .await
            .expect("HTTP/2 handshake with the test server");
    let mut conn_task = tokio::spawn(connection);

    for n in 1..=HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS {
        match send_malformed(&send_request, addr, deadline, n).await {
            Err(e)
                if e.is_reset() && e.is_remote() && e.reason() == Some(Reason::PROTOCOL_ERROR) => {}
            other => panic!(
                "request {n} of {HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS} must be reset \
                 PROTOCOL_ERROR by the server, within the limit; got {other:?}"
            ),
        }
    }

    let past_the_limit = HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS + 1;
    match send_malformed(&send_request, addr, deadline, past_the_limit).await {
        Err(e) if e.is_reset() && e.reason() == Some(Reason::PROTOCOL_ERROR) => panic!(
            "the server applies no local-error-reset limit, or one above \
             {HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS}: request {past_the_limit} was reset like \
             the others instead of closing the connection ({e})"
        ),
        Err(_) => {}
        Ok(response) => panic!(
            "request {past_the_limit} is malformed and must never be answered; got {response:?}"
        ),
    }
    let closed = tokio::time::timeout_at(deadline, &mut conn_task)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "the server applies no local-error-reset limit: the connection was still open \
                 at the test's {TEST_BUDGET:?} deadline after request {past_the_limit}, with no \
                 GOAWAY"
            )
        })
        .expect("the client connection task does not panic");
    let e = closed.expect_err("the connection must end in the server's GOAWAY, not cleanly");
    assert!(
        e.is_go_away() && e.is_remote(),
        "the connection must end in a GOAWAY the server sent; got {e:?}"
    );
    assert_eq!(
        e.reason(),
        Some(Reason::ENHANCE_YOUR_CALM),
        "the GOAWAY's error code; got {e:?}"
    );
    assert!(
        e.to_string().contains(RESET_LIMIT_GOAWAY_DEBUG_DATA),
        "the GOAWAY's debug data must name the cause, `{RESET_LIMIT_GOAWAY_DEBUG_DATA}`; \
         got `{e}`"
    );

    let captured = capture.snapshot();
    let warnings: Vec<_> = captured
        .events
        .iter()
        .filter(|event| {
            event["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(RESET_LIMIT_WARN))
        })
        .collect();
    assert_eq!(
        warnings.len(),
        1,
        "exactly one `{RESET_LIMIT_WARN}` line; full capture:\n{}",
        captured.text
    );
    let warning = warnings[0];
    assert_eq!(warning["level"], "WARN", "{warning}");
    assert!(
        warning["target"]
            .as_str()
            .is_some_and(|t| t.starts_with("h2::")),
        "h2 itself logs the line; {warning}"
    );
    let with_limit = format!("{RESET_LIMIT_WARN} ({HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS})");
    assert!(
        warning["fields"]["message"]
            .as_str()
            .is_some_and(|m| m.contains(&with_limit)),
        "the line names the limit, `{with_limit}`; {warning}"
    );

    server.abort();
}

/// The server's first SETTINGS frame advertises the pinned maximum frame
/// size and the stream window, and no limit on concurrent streams.
///
/// What this cannot tell apart: hyper's own server defaults are the same
/// 16,384-byte frame size and 1 MiB stream window, so removing
/// `.max_frame_size(HTTP2_MAX_FRAME_BYTES)` (or the stream-window call) from
/// `production_server_builder` fails nothing here today. The test pins the
/// advertised values, and it catches a missing call only once hyper's default
/// moves.
#[tokio::test]
async fn production_server_advertises_its_max_frame_size_and_stream_window_and_no_stream_limit() {
    let deadline = Instant::now() + TEST_BUDGET;
    let (addr, server) = spawn_production_server().await;
    let mut tcp = settled(
        deadline,
        "TCP connect",
        tokio::net::TcpStream::connect(addr),
    )
    .await
    .expect("TCP connect to the test server");
    settled(deadline, "writing the preface", async {
        tcp.write_all(CLIENT_PREFACE).await?;
        tcp.write_all(&EMPTY_SETTINGS_FRAME).await
    })
    .await
    .expect("write the client preface and SETTINGS");

    let mut header = [0_u8; 9];
    settled(
        deadline,
        "the server's first frame header",
        tcp.read_exact(&mut header),
    )
    .await
    .expect("read the server's first frame header");
    let length = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
    let stream_id = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
    assert_eq!(
        header[3], FRAME_TYPE_SETTINGS,
        "the server's first frame is its SETTINGS; header {header:?}"
    );
    assert_eq!(
        header[4] & SETTINGS_FLAG_ACK,
        0,
        "the server's own SETTINGS, not an ACK of the client's; header {header:?}"
    );
    assert_eq!(
        stream_id, 0,
        "SETTINGS is a connection frame; header {header:?}"
    );
    assert_eq!(
        length % 6,
        0,
        "a SETTINGS payload is whole 6-byte entries; header {header:?}"
    );

    let mut payload = vec![0_u8; length];
    settled(
        deadline,
        "the server's SETTINGS payload",
        tcp.read_exact(&mut payload),
    )
    .await
    .expect("read the server's SETTINGS payload");
    let entries: Vec<(u16, u32)> = payload
        .chunks_exact(6)
        .map(|e| {
            (
                u16::from_be_bytes([e[0], e[1]]),
                u32::from_be_bytes([e[2], e[3], e[4], e[5]]),
            )
        })
        .collect();
    let values_of = |id: u16| -> Vec<u32> {
        entries
            .iter()
            .filter(|(entry_id, _)| *entry_id == id)
            .map(|(_, value)| *value)
            .collect()
    };

    assert_eq!(
        HTTP2_MAX_FRAME_BYTES, 16_384,
        "the frame size is pinned at hyper's default, 16,384 bytes"
    );
    assert_eq!(
        values_of(SETTINGS_MAX_FRAME_SIZE),
        vec![HTTP2_MAX_FRAME_BYTES],
        "SETTINGS_MAX_FRAME_SIZE is advertised once, as the pinned frame size; \
         entries {entries:?}"
    );
    assert_eq!(
        values_of(SETTINGS_INITIAL_WINDOW_SIZE),
        vec![HTTP2_STREAM_WINDOW_BYTES],
        "SETTINGS_INITIAL_WINDOW_SIZE is advertised once, as the stream window; \
         entries {entries:?}"
    );
    assert!(
        values_of(SETTINGS_MAX_CONCURRENT_STREAMS).is_empty(),
        "no SETTINGS_MAX_CONCURRENT_STREAMS is advertised; entries {entries:?}"
    );

    server.abort();
}
