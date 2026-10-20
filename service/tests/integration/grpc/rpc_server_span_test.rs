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

//! Trace export behind a real tonic server: one server span per RPC, and
//! nothing from the transport crates.
//!
//! The regression for the dev incident in which h2's per-frame TRACE spans
//! reached the exporter — about 800 spans per commit and a queue that dropped
//! spans continuously. The unit tests in `src/observability` and
//! `src/grpc/metrics_layer.rs` pin the filter and the span in isolation; this
//! file checks the composed result on a server wired the way `main.rs` wires
//! it (`MetricsLayer` as a server layer, the `LiveService` built by
//! `live_service_server`), where the h2 frames are real.
//!
//! # Capture
//!
//! The subscriber is the service's own composition
//! (`compose_registry_for_tests`) over a tracer whose exporter keeps every
//! span synchronously as it ends. The server, its per-connection tasks and the
//! client all run on one **current-thread** runtime under a thread-local
//! default, so every span either side creates reaches the capture. The client
//! shares the capture deliberately: its h2 spans are filtered by the same
//! trace filter, so the default-filter assertion covers both directions of
//! the connection. Waits poll the capture with a bounded deadline; `TestValkey`
//! cannot clean up on a current-thread runtime, so every test ends with
//! [`purge_keys`].

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use opentelemetry::trace::{SpanId, TracerProvider as _};
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData};
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Server};
use tonic::{Code, Request};

use live_edit::config::{LogFormat, DEFAULT_TRACE_FILTER};
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::live_service_server;
use live_edit::observability::compose_registry_for_tests;
use live_edit::observability::trace_export::test_support::{
    keep_a_second_dispatcher_registered, CollectingExporter,
};
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    build_service_with_size_cap, create_session_and_layer, make_update_data, make_update_header,
    AllowAllPermissions, SMALL_SIZE_CAP,
};
use super::log_capture::purge_keys;

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// How long a test keeps watching for a late, extra span after the expected
/// ones arrived. An absence window, not a synchronisation sleep: the positive
/// half has already been observed.
const ABSENCE_WINDOW: Duration = Duration::from_millis(300);

/// Bearer for requests that must pass the interceptor. The mock permission
/// client ignores its value.
const BEARER: &str = "rpc-server-span-bearer";

/// Name prefix of every `LiveService` server span.
const LIVE_SERVICE_SPAN_PREFIX: &str = "nvidia.omniverse.live.v1alpha.LiveService/";

/// Crates whose spans are transport internals, never meant for export.
const TRANSPORT_TARGETS: &[&str] = &["h2", "hyper", "tonic", "tower"];

const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const PARENT_SPAN_ID: &str = "00f067aa0ba902b7";

// =============================================================================
// Capture
// =============================================================================

/// The service's subscriber composition with `trace_filter`, installed as this
/// thread's default, exporting into memory.
struct TraceCapture {
    exporter: CollectingExporter,
    _provider: SdkTracerProvider,
    _guard: tracing::subscriber::DefaultGuard,
}

impl TraceCapture {
    fn start(trace_filter: &str) -> Self {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread,
            "the thread-local capture only sees work polled on this thread"
        );
        keep_a_second_dispatcher_registered();
        let exporter = CollectingExporter::default();
        let provider = temp_env::with_vars(
            [
                ("OTEL_TRACES_SAMPLER", None::<&str>),
                ("OTEL_TRACES_SAMPLER_ARG", None),
            ],
            || {
                SdkTracerProvider::builder()
                    .with_simple_exporter(exporter.clone())
                    .build()
            },
        );
        let subscriber = compose_registry_for_tests(
            "warn",
            LogFormat::Json,
            trace_filter,
            std::io::sink,
            Some(provider.tracer("live-edit")),
        );
        Self {
            exporter,
            _provider: provider,
            _guard: tracing::subscriber::set_default(subscriber),
        }
    }

    fn spans(&self) -> Vec<SpanData> {
        self.exporter.spans()
    }

    fn server_spans(&self) -> Vec<SpanData> {
        self.spans()
            .into_iter()
            .filter(|span| span.name.starts_with(LIVE_SERVICE_SPAN_PREFIX))
            .collect()
    }

    /// Exported spans whose `target` is one of [`TRANSPORT_TARGETS`].
    fn transport_spans(&self) -> Vec<SpanData> {
        self.spans()
            .into_iter()
            .filter(|span| {
                attr(span, "target").is_some_and(|target| {
                    TRANSPORT_TARGETS
                        .iter()
                        .any(|krate| target == *krate || target.starts_with(&format!("{krate}::")))
                })
            })
            .collect()
    }

    /// Wait until `want` server spans are exported, then hold the absence
    /// window and return them — so a late extra span is still seen.
    async fn settled_server_spans(&self, want: usize) -> Vec<SpanData> {
        let deadline = tokio::time::Instant::now() + SETTLE;
        while self.server_spans().len() < want {
            assert!(
                tokio::time::Instant::now() < deadline,
                "only {} of {want} server spans exported within {SETTLE:?}",
                self.server_spans().len()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(ABSENCE_WINDOW).await;
        self.server_spans()
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.as_str().into_owned())
}

fn names(spans: &[SpanData]) -> Vec<String> {
    spans.iter().map(|span| span.name.to_string()).collect()
}

// =============================================================================
// Server fixture
// =============================================================================

/// A running server with the production layer stack around the `LiveService`.
struct LiveServer {
    addr: SocketAddr,
    shutdown_tx: oneshot::Sender<()>,
    server_task: tokio::task::JoinHandle<()>,
}

impl LiveServer {
    async fn start(svc: LiveEditService) -> Self {
        let metrics_layer = MetricsLayer::new(
            Arc::new(crate::common::metrics_scrape_only("test")),
            "test",
            30_000,
            2_000,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind to a random port");
        let addr = listener.local_addr().expect("no local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let server_task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            Server::builder()
                .layer(metrics_layer)
                .add_service(live_service_server(svc))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("test server error");
        });
        tokio::time::timeout(SETTLE, ready_rx)
            .await
            .expect("the server did not signal readiness")
            .expect("the server task dropped its ready channel");

        Self {
            addr,
            shutdown_tx,
            server_task,
        }
    }

    async fn client(&self) -> LiveServiceClient<Channel> {
        let channel = Channel::from_shared(format!("http://{}", self.addr))
            .expect("valid endpoint URI")
            .connect()
            .await
            .expect("failed to connect to the test server");
        LiveServiceClient::new(channel)
    }

    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        tokio::time::timeout(SETTLE, self.server_task)
            .await
            .expect("the server did not stop in time")
            .expect("the server task panicked");
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

fn stat_request(session_id: &str, layer_name: &str) -> StatLayerRequest {
    StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    }
}

/// A server over a session with one layer, for the StatLayer calls.
async fn start_with_layer(tv: &TestValkey, layer: &str) -> (LiveServer, String) {
    let svc = build_service_with_size_cap(tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let (session_id, _object_id) = create_session_and_layer(tv, &svc, "rpc-span", layer).await;
    (LiveServer::start(svc).await, session_id)
}

// =============================================================================
// Tests
// =============================================================================

/// The incident regression: under the default trace filter, N StatLayer and
/// UpdateLayer calls export exactly N spans — one server span each — and none
/// from h2, hyper, tonic or tower, on either end of the connection.
#[tokio::test]
async fn default_trace_filter_exports_one_server_span_per_rpc_and_no_transport_spans() {
    const LAYER: &str = "rpc-span-layer";
    const STAT_CALLS: usize = 3;
    const UPDATE_CALLS: usize = 2;

    let tv = TestValkey::setup().await;
    let (server, session_id) = start_with_layer(&tv, LAYER).await;
    let capture = TraceCapture::start(DEFAULT_TRACE_FILTER);
    let mut client = server.client().await;

    for _ in 0..STAT_CALLS {
        client
            .stat_layer(with_bearer(stat_request(&session_id, LAYER)))
            .await
            .expect("StatLayer must succeed");
    }
    for _ in 0..UPDATE_CALLS {
        // Over the size cap: refused INVALID_ARGUMENT after the upload is
        // read, which is enough RPC for a span without a valid delta.
        let upload = futures::stream::iter(vec![
            make_update_header(&session_id, LAYER, 1),
            make_update_data(vec![0xAB; SMALL_SIZE_CAP + 1]),
        ]);
        let status = client
            .update_layer(with_bearer(upload))
            .await
            .expect_err("an upload over the size cap must be refused");
        assert_eq!(status.code(), Code::InvalidArgument, "precondition");
    }

    let server_spans = capture
        .settled_server_spans(STAT_CALLS + UPDATE_CALLS)
        .await;
    let server_names = names(&server_spans);
    let count = |method: &str| {
        server_names
            .iter()
            .filter(|name| **name == format!("{LIVE_SERVICE_SPAN_PREFIX}{method}"))
            .count()
    };
    assert_eq!(count("StatLayer"), STAT_CALLS, "{server_names:?}");
    assert_eq!(count("UpdateLayer"), UPDATE_CALLS, "{server_names:?}");

    let transport = capture.transport_spans();
    assert!(
        transport.is_empty(),
        "no transport span may be exported under the default filter; got {:?}",
        names(&transport)
    );
    assert_eq!(
        capture.spans().len(),
        STAT_CALLS + UPDATE_CALLS,
        "exactly one exported span per RPC and nothing else; got {:?}",
        names(&capture.spans())
    );

    drop(client);
    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// The previous test depends on the filter, not on h2 creating no spans: with
/// `trace` as the trace filter, the same server exports h2's spans.
#[tokio::test]
async fn trace_filter_trace_exports_transport_spans() {
    const LAYER: &str = "rpc-span-trace-layer";

    let tv = TestValkey::setup().await;
    let (server, session_id) = start_with_layer(&tv, LAYER).await;
    let capture = TraceCapture::start("trace");
    let mut client = server.client().await;

    client
        .stat_layer(with_bearer(stat_request(&session_id, LAYER)))
        .await
        .expect("StatLayer must succeed");
    let server_spans = capture.settled_server_spans(1).await;
    assert_eq!(server_spans.len(), 1, "precondition: the RPC's own span");

    let h2_spans: Vec<String> = capture
        .transport_spans()
        .iter()
        .filter(|span| attr(span, "target").is_some_and(|t| t.starts_with("h2")))
        .map(|span| span.name.to_string())
        .collect();
    assert!(
        !h2_spans.is_empty(),
        "under `trace`, h2's spans must be exported; got {:?}",
        names(&capture.spans())
    );

    drop(client);
    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// An RPC the auth interceptor refuses still produces exactly one server span
/// — not zero, and a `traceparent` does not add any — in the caller's trace.
#[tokio::test]
async fn auth_rejected_rpc_with_traceparent_exports_exactly_one_server_span() {
    const LAYER: &str = "rpc-span-auth-layer";

    let tv = TestValkey::setup().await;
    let (server, session_id) = start_with_layer(&tv, LAYER).await;
    let capture = TraceCapture::start(DEFAULT_TRACE_FILTER);
    let mut client = server.client().await;

    let mut request = Request::new(stat_request(&session_id, LAYER));
    request.metadata_mut().insert(
        "traceparent",
        MetadataValue::try_from(format!("00-{TRACE_ID}-{PARENT_SPAN_ID}-01"))
            .expect("valid metadata"),
    );
    let status = client
        .stat_layer(request)
        .await
        .expect_err("a request without credentials must be refused");
    assert_eq!(status.code(), Code::Unauthenticated, "precondition");

    let server_spans = capture.settled_server_spans(1).await;
    assert_eq!(
        server_spans.len(),
        1,
        "a refused RPC has exactly one span; got {:?}",
        names(&server_spans)
    );
    let span = &server_spans[0];
    assert_eq!(
        span.name,
        format!("{LIVE_SERVICE_SPAN_PREFIX}StatLayer"),
        "the refused call's span"
    );
    assert_eq!(
        span.parent_span_id,
        SpanId::from_hex(PARENT_SPAN_ID).expect("valid span id"),
        "the span joins the caller's trace"
    );
    assert_eq!(
        attr(span, "rpc.status_code").as_deref(),
        Some("UNAUTHENTICATED")
    );
    assert_eq!(
        capture.spans().len(),
        1,
        "nothing else is exported for the refused call; got {:?}",
        names(&capture.spans())
    );

    drop(client);
    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}
