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

//! A disaster-recovery hold on a serving replica, end to end (SDD SR-6.1–6.4,
//! SR-7): the hold marker written in the store, the real hold poll reading
//! it, the hold reactor acting on it, and the real server stack `main.rs`
//! serves — `MetricsLayer` as a server layer, the LiveService built by
//! `live_service_server` with the hold gate inside it, and health, reflection
//! and the CapabilitiesService added bare — answering over TCP.
//!
//! - **Marker written** ([`a_written_marker_holds_a_serving_replica_behind_its_real_server`]).
//!   The open subscription and join end `SERVICE_HELD` naming the incident;
//!   every LiveService RPC — unary, upload, streaming, with or without
//!   credentials — is refused `SERVICE_HELD` before auth runs; no per-session
//!   token is cancelled, `begun` and the shutdown parent stay uncancelled;
//!   `live_edit_dr_held` is 1 and the entry is counted once under the
//!   marker's reason; and after the RPC deadline plus its close grace the
//!   server still answers health (`""` SERVING), Capabilities and, with its
//!   refusal, the LiveService.
//! - **One completion line per refusal**
//!   ([`a_refused_rpc_logs_one_info_line_and_never_reaches_auth`] for a hold,
//!   [`a_stale_refused_rpc_logs_one_warn_line_and_never_reaches_auth`] for a
//!   stale verdict, through one helper). Each refused call — unary, upload,
//!   streaming, uncredentialed — logs exactly one completion line from the
//!   gate, at INFO for a hold and at WARN for a stale verdict; the metrics
//!   layer's backstop logs nothing more for it, and the permission client is
//!   never asked.
//! - **Marker removed** ([`a_removed_marker_makes_a_held_replica_exit_zero`]):
//!   the reactor calls the exit seam with 0 within one poll interval plus the
//!   5 s jitter bound (a deadline here, not a measurement; the virtual-time
//!   bound is `hold_reactor_test.rs`'s).
//! - **Marker before the serving point**
//!   ([`a_marker_before_the_serving_point_exits_zero_without_entering_the_hold`]):
//!   exit 0 through the seam with the serving point unreached (that the
//!   entry routine does not run is `hold_reactor_test.rs`'s).
//! - **A dead hold-poll connection**
//!   ([`a_dead_hold_poll_connection_fails_the_replica_closed_and_its_next_read_restarts_it`];
//!   SDD SR-6.5): about 30 s after the poll's last read that answered, the
//!   open subscription and every refused call end `UPSTREAM_UNAVAILABLE`,
//!   never `SERVICE_HELD`; the writer is aborted, the stale gauge is 1,
//!   readiness goes `NOT_SERVING` through the health supervisor, and no
//!   marker is written. The next read that answers exits 0, and a server
//!   rebuilt on the same store admits a subscription.
//!
//! # Waits
//!
//! Every wait is a deadline a healthy run never reaches: [`SETTLE`] (10 s)
//! per awaited event (a catch-up drain, a stream's end, a poll condition, a
//! connect, a readiness change), [`EXIT_DEADLINE`] (20 s) for an exit,
//! [`STALE_DEADLINE`] (45 s) for the stale entry, and [`COMMAND_TIMEOUT_MS`]
//! (5 s) for a one-shot hold read. Except for the stale test, the heaviest
//! test, the first, waits at most nine times (the startup hold read, connect,
//! catch-up, `SessionConnected`, two stream ends, the gate's closing, the
//! server's stop, the deadline window), about 85 s at every budget's full
//! value, inside nextest's 120 s hang-killer. The first test's deliberate
//! pause is its [`RPC_DEADLINE_MS`] + [`DEADLINE_GRACE_MS`] window, the span
//! after which the AC says the server still answers.
//!
//! The stale test pays the 30 s threshold in every run (the constant is not
//! injectable), and the exit's jitter of up to 5 s: 30 to 40 s in all. At
//! every budget's full value it waits
//! about 220 s: fourteen [`SETTLE`]s (readiness twice; each of two servers'
//! start, connect, catch-up and stop; the stream's end; the stall's park and
//! release; the supervisor's stop), [`STALE_DEADLINE`], [`EXIT_DEADLINE`]
//! and three one-shot hold reads (the startup read, the reread while stale,
//! the restarted process's startup read). `.config/nextest.toml` gives it its
//! own 240 s deadline.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Code, Request, Status};
use tonic_reflection::server::Builder as ReflectionBuilder;

use live_edit::dr::hold::{
    evaluate_hold_at_startup, spawn_hold_poll, HoldObservation, HoldPollSettings, HoldWriter,
    STALE_VERDICT_THRESHOLD, UNHOLD_EXIT_JITTER_MAX,
};
use live_edit::grpc::capabilities::CapabilitiesServiceImpl;
use live_edit::grpc::graceful_shutdown::GracefulShutdown;
use live_edit::grpc::hold::{
    spawn_hold_reactor, HoldEntryParts, HoldGate, HoldGateState, HoldReactorSettings,
    HoldServingPoint, HoldWriterAborts,
};
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, CreateSessionRequest, JoinSessionRequest, PingRequest,
    StatLayerRequest, SubscribeReadLayerRequest, SubscribeReadLayerResponse, FILE_DESCRIPTOR_SET,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::{live_service_server, production_server_builder};
use live_edit::observability::health::{
    create_health_service, spawn_health_supervisor, HealthSupervisorConfig,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, DR_HOLD_REASONS, DR_HOLD_REASON_KEYSPACE_LOST};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::test_support::{
    connect_test_client, release_all, stall_pool, HoldMarkerSeed, TestValkey,
};
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_client::CapabilitiesServiceClient;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::CapabilitiesServiceServer;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::ListServicesRequest;

use super::hold_support::answering;
use super::support::{hold_entries, until_readiness, HealthSource, RecordingExit, READINESS};
use crate::common::refusal_detail::{assert_service_held, assert_stale_verdict};
use crate::grpc::helpers::{
    build_service, create_session_and_layer, join_header, make_create_data, make_create_header,
    RecordingPermissions,
};
use crate::grpc::log_capture::{
    assert_common_fields, assert_field, purge_keys, Captured, ThreadCapture, RPC_FAILURE_MSG,
};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The hold poll's interval in these tests: short, so the marker is read
/// within a fraction of a second.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The fixture clients' Valkey command timeout, which the poll's per-read
/// bound is derived from, and the bound of each one-shot hold read.
const COMMAND_TIMEOUT_MS: u64 = 5_000;

/// The server's RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`) in the first test:
/// short, so the window after which the server must still answer is short.
const RPC_DEADLINE_MS: u64 = 1_000;

/// The deadline layer's close grace in the first test.
const DEADLINE_GRACE_MS: u64 = 500;

/// The un-hold exit's deadline: one poll interval, the 5 s jitter bound, and
/// slack for a loaded runner.
const EXIT_DEADLINE: Duration = Duration::from_secs(20);

/// The metrics layer's service label.
const SERVICE_LABEL: &str = "test";

/// Bearer on every credentialed request. The recording permission client
/// ignores it.
const BEARER: &str = "hold-entry-bearer";

/// The layer every test here uses.
const LAYER: &str = "hold-entry-layer";

/// The incident the hold marker names.
const INCIDENT: &str = "20261015t0347-e5f6";

/// Bound on the stale-verdict entry after the poll's connection stalls: the
/// 30 s threshold, counted from the last read that answered, and slack for a
/// loaded runner. The one long wait of the stale test, and the only one a
/// healthy run pays most of.
const STALE_DEADLINE: Duration = Duration::from_secs(45);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The production server stack around `svc` behind `gate`, on a random port.
struct HeldServer {
    addr: SocketAddr,
    /// The metrics layer's registry: `live_endpoint_calls_total`.
    layer_metrics: Arc<Metrics>,
    shutdown_tx: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl HeldServer {
    async fn start(svc: LiveEditService, gate: HoldGate, deadline_ms: u64, grace_ms: u64) -> Self {
        let layer_metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
        let metrics_layer = MetricsLayer::new(
            Arc::clone(&layer_metrics),
            SERVICE_LABEL,
            deadline_ms,
            grace_ms,
        );
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_service_status("", tonic_health::ServingStatus::Serving)
            .await;
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .expect("reflection must build");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a random port");
        let addr = listener.local_addr().expect("local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            // The reporter lives as long as the server: dropping it would end
            // health `Watch` streams, and nothing else may touch it — the
            // hold tells the health supervisor nothing.
            let _health_reporter = health_reporter;
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder(svc.upload_admission.max_waiters())
                .layer(metrics_layer)
                .add_service(live_service_server(svc, gate))
                .add_service(health_service)
                .add_service(reflection_service)
                .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("the test server must not fail");
        });
        tokio::time::timeout(SETTLE, ready_rx)
            .await
            .expect("the server did not signal readiness")
            .expect("the server task dropped its ready channel");
        Self {
            addr,
            layer_metrics,
            shutdown_tx,
            task,
        }
    }

    async fn channel(&self) -> Channel {
        tokio::time::timeout(
            SETTLE,
            Channel::from_shared(format!("http://{}", self.addr))
                .expect("valid endpoint")
                .connect(),
        )
        .await
        .expect("connect within the deadline")
        .expect("connect to the test server")
    }

    /// `live_endpoint_calls_total{endpoint, grpc_code}`.
    fn calls(&self, endpoint: &str, code: &str) -> u64 {
        sample_value(
            &render(&self.layer_metrics),
            "live_endpoint_calls_total",
            &[
                ("endpoint", endpoint),
                ("service", SERVICE_LABEL),
                ("grpc_code", code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Wait until `endpoint` has recorded `n` `code` ticks: every completion
    /// line is emitted before the call is recorded.
    async fn await_calls(&self, endpoint: &str, code: &str, n: u64) {
        let deadline = tokio::time::Instant::now() + SETTLE;
        while self.calls(endpoint, code) < n {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{endpoint}/{code} did not reach {n} ticks within {SETTLE:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        tokio::time::timeout(SETTLE, self.task)
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

fn subscribe_request(session_id: &str, client_id: &str) -> SubscribeReadLayerRequest {
    SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: client_id.to_owned(),
    }
}

fn stat_request(session_id: &str) -> StatLayerRequest {
    StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
    }
}

/// Read a subscription through its caught-up ack, under one deadline.
async fn read_catch_up<S>(stream: &mut S)
where
    S: futures::Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, async {
        loop {
            let msg = stream
                .next()
                .await
                .expect("the stream ended during catch-up")
                .unwrap_or_else(|s| panic!("catch-up failed: {s:?}"));
            if let Some(subscribe_read_layer_response::ResponseType::Header(h)) = msg.response_type
            {
                if h.total_size == 0 {
                    return;
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not finish within {SETTLE:?}"));
}

/// Read a stream to its terminal status under one deadline; `None` for a
/// clean end.
async fn terminal_status<S, T>(stream: &mut S, what: &str) -> Option<Status>
where
    S: futures::Stream<Item = Result<T, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the stream did not end within {SETTLE:?}"))
}

/// Poll `cond` until it holds, within [`SETTLE`].
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The hold poll over `tv`'s store at [`POLL_INTERVAL`], seeded by the
/// startup evaluation exactly as `main.rs` seeds it, and the hold reactor on
/// its channel with a recording exit seam. The serving point is reached with
/// `parts(verdicts)` when `serve` is given, left unreached otherwise.
struct HoldWiring {
    verdicts: watch::Receiver<HoldObservation>,
    point: HoldServingPoint,
    /// Each exit-seam call's code and instant.
    exits: mpsc::UnboundedReceiver<(i32, tokio::time::Instant)>,
    shutdown: CancellationToken,
    /// The reactor's readiness signal (`HoldReactorSettings::verdict_stale`),
    /// as `main.rs` hands it to the health supervisor.
    verdict_stale: watch::Receiver<bool>,
}

impl HoldWiring {
    async fn start(tv: &TestValkey) -> Self {
        Self::start_on(tv, tv.client().clone()).await
    }

    /// [`Self::start`] with the poll reading the store through `client`.
    async fn start_on(tv: &TestValkey, client: ValkeyClient) -> Self {
        let initial =
            evaluate_hold_at_startup(&client, tv.key_schema(), answering(COMMAND_TIMEOUT_MS))
                .await
                .expect("the startup hold evaluation must answer");
        assert!(
            !initial.verdict.is_held(),
            "fixture: no marker is written before the test writes one"
        );
        let shutdown = CancellationToken::new();
        let (verdicts, _poll) = spawn_hold_poll(
            client,
            tv.key_schema().clone(),
            HoldPollSettings::new(POLL_INTERVAL, COMMAND_TIMEOUT_MS),
            initial,
            shutdown.clone(),
        );
        let point = HoldServingPoint::new();
        let (calls, exits) = mpsc::unbounded_channel();
        let stale_tx = watch::Sender::new(false);
        let verdict_stale = stale_tx.subscribe();
        let _reactor = spawn_hold_reactor(
            verdicts.clone(),
            point.clone(),
            shutdown.clone(),
            HoldReactorSettings {
                heartbeat_interval: Duration::from_secs(300),
                exit: Arc::new(RecordingExit { calls }),
                verdict_stale: stale_tx,
                held: watch::Sender::new(false),
            },
        );
        Self {
            verdicts,
            point,
            exits,
            shutdown,
            verdict_stale,
        }
    }

    /// Reach the serving point with a gate on this poll's channel, `streams`
    /// and `metrics`, and an empty abort set. Returns the parts.
    fn serve(&self, streams: GracefulShutdown, metrics: Arc<Metrics>) -> HoldEntryParts {
        let parts = HoldEntryParts {
            gate: HoldGate::new(self.verdicts.clone()),
            streams,
            writers: Arc::new(HoldWriterAborts::new()),
            metrics,
        };
        assert!(self.point.reach(parts.clone()));
        parts
    }

    async fn next_exit(&mut self, within: Duration, what: &str) -> i32 {
        tokio::time::timeout(within, self.exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: the exit seam was not called within {within:?}"))
            .unwrap_or_else(|| panic!("{what}: the reactor dropped its exit seam uncalled"))
            .0
    }
}

impl Drop for HoldWiring {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

fn dr_held(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_dr_held", &[])
        .expect("live_edit_dr_held is recorded at construction") as u64
}

fn verdict_stale_gauge(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_dr_hold_verdict_stale", &[])
        .expect("live_edit_dr_hold_verdict_stale is recorded at construction") as u64
}

/// A client of the test's own on the fixture's store with one data
/// connection, so a stall of that connection stalls everything the client
/// sends, and nothing else.
async fn one_connection_client() -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.valkey_pool_size = 1;
    config.valkey_command_timeout_ms = COMMAND_TIMEOUT_MS;
    connect_test_client(&config)
        .await
        .expect("the test's own client must connect")
}

// ═════════════════════════════════════════════════════════════════════════════
// Marker written: the server refuses, the streams end, health answers
// ═════════════════════════════════════════════════════════════════════════════

/// A marker written in the store holds a serving replica behind its real
/// server: the streams end `SERVICE_HELD`, every LiveService RPC is refused
/// `SERVICE_HELD` before auth, nothing is shut down or cancelled, the two
/// signals are recorded, and the server still answers after the RPC deadline
/// plus its close grace.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_written_marker_holds_a_serving_replica_behind_its_real_server() {
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, Arc::clone(&permissions) as _);
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "hold-entry-real-server", LAYER).await;

    let wiring = HoldWiring::start(&tv).await;
    let parts = wiring.serve(svc.shutdown.clone(), Arc::clone(&svc.metrics));
    let server = HeldServer::start(
        svc.clone(),
        parts.gate.clone(),
        RPC_DEADLINE_MS,
        DEADLINE_GRACE_MS,
    )
    .await;
    let channel = server.channel().await;
    let mut client = LiveServiceClient::new(channel.clone());

    // An idle subscription and an established join.
    let mut subscription = client
        .subscribe_read_layer(with_bearer(subscribe_request(&session_id, "held-reader")))
        .await
        .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
        .into_inner();
    read_catch_up(&mut subscription).await;
    let (join_tx, join_rx) = mpsc::channel::<JoinSessionRequest>(4);
    join_tx
        .send(join_header(&session_id, "held-app"))
        .await
        .expect("the request channel accepts the header");
    let mut join = client
        .join_session(with_bearer(ReceiverStream::new(join_rx)))
        .await
        .unwrap_or_else(|s| panic!("the join must be admitted: {s:?}"))
        .into_inner();
    let first = tokio::time::timeout(SETTLE, join.next())
        .await
        .expect("SessionConnected within the deadline")
        .expect("the join stream must not be empty")
        .unwrap_or_else(|s| panic!("the join failed before SessionConnected: {s:?}"));
    assert!(
        first.response_type.is_some(),
        "precondition: the join is established"
    );
    let session_token = svc
        .session_tokens
        .get(&session_id)
        .expect("the streams registered the session's token")
        .clone();

    // The marker.
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .with_reason(DR_HOLD_REASON_KEYSPACE_LOST),
    )
    .await;

    // The streams end SERVICE_HELD.
    let sub_end = terminal_status(&mut subscription, "the subscription")
        .await
        .expect("the subscription ended OK, which the Kit client never retries");
    assert_service_held(&sub_end, Some(INCIDENT));
    let join_end = terminal_status(&mut join, "the join")
        .await
        .expect("the join ended OK, which the Kit client never retries");
    assert_service_held(&join_end, Some(INCIDENT));
    assert_eq!(parts.gate.state(), HoldGateState::Held);
    drop(join_tx);
    let asked_before_refusals = permissions.calls().len();

    // Every LiveService RPC is refused SERVICE_HELD, before auth.
    let stat = client
        .stat_layer(with_bearer(stat_request(&session_id)))
        .await
        .expect_err("StatLayer is refused while held");
    assert_service_held(&stat, Some(INCIDENT));
    let ping = client
        .ping(with_bearer(PingRequest {
            client_timestamp_microseconds_since_epoch: 1,
        }))
        .await
        .expect_err("Ping is refused while held");
    assert_service_held(&ping, Some(INCIDENT));
    let create_session = client
        .create_session(with_bearer(CreateSessionRequest {
            path: "/stage".to_owned(),
            name: "created-while-held".to_owned(),
        }))
        .await
        .expect_err("CreateSession is refused while held");
    assert_service_held(&create_session, Some(INCIDENT));
    let upload = client
        .create_layer(with_bearer(futures::stream::iter(vec![
            make_create_header(&session_id, "uploaded-while-held"),
            make_create_data(Vec::new()),
        ])))
        .await
        .expect_err("CreateLayer is refused while held");
    assert_service_held(&upload, Some(INCIDENT));
    let late_subscription = client
        .subscribe_read_layer(with_bearer(subscribe_request(&session_id, "late-reader")))
        .await
        .expect_err("a new SubscribeReadLayer is refused while held");
    assert_service_held(&late_subscription, Some(INCIDENT));
    let uncredentialed = client
        .stat_layer(stat_request(&session_id))
        .await
        .expect_err("an uncredentialed call is refused while held");
    assert_service_held(&uncredentialed, Some(INCIDENT));
    assert_ne!(
        uncredentialed.code(),
        Code::Unauthenticated,
        "the gate runs before the auth interceptor"
    );
    assert_eq!(
        permissions.calls().len(),
        asked_before_refusals,
        "no refused call reaches the permission check"
    );

    // Not a shutdown, not a session ending; the signals.
    assert!(
        !session_token.is_cancelled(),
        "a hold cancels no per-session token"
    );
    assert!(
        !svc.shutdown.begun().is_cancelled() && !svc.shutdown.is_cancelled(),
        "a hold starts no shutdown: the scaling snapshot's shutting_down stays false"
    );
    assert_eq!(dr_held(&svc.metrics), 1);
    assert_eq!(
        hold_entries(&render(&svc.metrics)),
        DR_HOLD_REASONS
            .iter()
            .map(|r| (*r, u64::from(*r == DR_HOLD_REASON_KEYSPACE_LOST)))
            .collect::<Vec<_>>()
    );

    // After the RPC deadline and its close grace, the server still answers.
    // Health answering is the evidence here; that it answers SERVING is true
    // by construction — the hold's parts (`HoldEntryParts`: the gate, the
    // stream handle, the abort set and the metrics) hold no health reporter,
    // so nothing on the hold path can change readiness.
    tokio::time::sleep(Duration::from_millis(RPC_DEADLINE_MS + DEADLINE_GRACE_MS)).await;
    let health = tonic_health::pb::health_client::HealthClient::new(channel.clone())
        .check(tonic_health::pb::HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("health still answers while held")
        .into_inner();
    assert_eq!(
        health.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32,
        "\"\" still answers SERVING while held"
    );
    CapabilitiesServiceClient::new(channel.clone())
        .list_services(ListServicesRequest::default())
        .await
        .expect("the CapabilitiesService is not behind the gate");
    let still = client
        .stat_layer(with_bearer(stat_request(&session_id)))
        .await
        .expect_err("still refused after the deadline");
    assert_service_held(&still, Some(INCIDENT));

    drop(subscription);
    drop(join);
    drop(client);
    drop(channel);
    server.shutdown().await;
}

// ═════════════════════════════════════════════════════════════════════════════
// One completion line per refusal: INFO for a hold, WARN for a stale verdict
// (SDD §17.2.1)
// ═════════════════════════════════════════════════════════════════════════════

/// Every completion line in the capture.
fn completion_lines(captured: &Captured) -> Vec<&serde_json::Value> {
    captured.completion_lines()
}

/// How a refusal test closes the gate.
#[derive(Clone, Copy, Debug)]
enum GateClosing {
    /// A fleet hold (`close_held`), its verdict naming [`INCIDENT`].
    Held,
    /// A stale hold verdict (`close_stale`), over a `Serving` verdict.
    Stale,
}

/// Behind the full stack with the gate closed by `closing`, each refused
/// call — a unary, an upload, a server stream, and an uncredentialed unary —
/// is refused with the closing's status (`SERVICE_HELD` naming the incident,
/// or the stale verdict's `UPSTREAM_UNAVAILABLE`) and logs exactly one
/// completion line, `stage=header`, naming its endpoint, at the closing's
/// level: INFO for a hold (the fleet's expected state, reported once on its
/// own), WARN for a stale verdict (a dependency outage). The metrics layer
/// records each once as `UNAVAILABLE`: its backstop adds nothing. The
/// permission client is never asked.
async fn assert_each_refusal_logs_one_line_and_never_reaches_auth(closing: GateClosing) {
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, Arc::clone(&permissions) as _);
    let (verdict, level) = match closing {
        GateClosing::Held => (
            live_edit::dr::hold::HoldVerdict::Held {
                incident: Some(INCIDENT.to_owned()),
                reason: Some(DR_HOLD_REASON_KEYSPACE_LOST.to_owned()),
                evidence: None,
                detected_at_ms: None,
            },
            "INFO",
        ),
        GateClosing::Stale => (live_edit::dr::hold::HoldVerdict::Serving, "WARN"),
    };
    let (_tx, verdicts) = watch::channel(HoldObservation {
        verdict,
        at: tokio::time::Instant::now(),
    });
    let gate = HoldGate::new(verdicts);
    match closing {
        GateClosing::Held => gate.close_held(),
        GateClosing::Stale => gate.close_stale(),
    }
    let server = HeldServer::start(svc, gate, 30_000, 2_000).await;
    let mut client = LiveServiceClient::new(server.channel().await);
    let capture = ThreadCapture::start();

    let calls: [(&str, Status); 4] = [
        (
            "StatLayer",
            client
                .stat_layer(with_bearer(stat_request("some-session")))
                .await
                .expect_err("refused"),
        ),
        (
            "CreateLayer",
            client
                .create_layer(with_bearer(futures::stream::iter(vec![
                    make_create_header("some-session", LAYER),
                    make_create_data(Vec::new()),
                ])))
                .await
                .expect_err("refused"),
        ),
        (
            "SubscribeReadLayer",
            client
                .subscribe_read_layer(with_bearer(subscribe_request("some-session", "r")))
                .await
                .expect_err("refused"),
        ),
        (
            "Ping",
            client
                .ping(PingRequest {
                    client_timestamp_microseconds_since_epoch: 1,
                })
                .await
                .expect_err("refused"),
        ),
    ];
    for (endpoint, status) in &calls {
        match closing {
            GateClosing::Held => assert_service_held(status, Some(INCIDENT)),
            GateClosing::Stale => assert_stale_verdict(status),
        }
        server.await_calls(endpoint, "UNAVAILABLE", 1).await;
    }

    let captured = capture.snapshot();
    let lines = completion_lines(&captured);
    assert_eq!(
        lines.len(),
        calls.len(),
        "{closing:?}: exactly one completion line per refused call; full capture:\n{}",
        captured.text
    );
    for (endpoint, status) in &calls {
        let line = lines
            .iter()
            .find(|e| e["fields"]["endpoint"] == *endpoint)
            .unwrap_or_else(|| panic!("no completion line for {endpoint}:\n{}", captured.text));
        assert_eq!(line["fields"]["message"], RPC_FAILURE_MSG);
        assert_common_fields(line, level, endpoint, "header", "UNAVAILABLE");
        assert_field(line, "status_message", status.message().into());
        assert_eq!(
            server.calls(endpoint, "UNAVAILABLE"),
            1,
            "{endpoint}: one call, one tick"
        );
    }
    assert!(
        permissions.calls().is_empty(),
        "a refused call never reaches the permission check: {:?}",
        permissions.calls()
    );

    drop(capture);
    drop(client);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A hold's refusals: one INFO completion line each, never reaching auth
/// ([`assert_each_refusal_logs_one_line_and_never_reaches_auth`]).
#[tokio::test]
async fn a_refused_rpc_logs_one_info_line_and_never_reaches_auth() {
    assert_each_refusal_logs_one_line_and_never_reaches_auth(GateClosing::Held).await;
}

/// A stale verdict's refusals: one WARN completion line each, never reaching
/// auth ([`assert_each_refusal_logs_one_line_and_never_reaches_auth`]).
#[tokio::test]
async fn a_stale_refused_rpc_logs_one_warn_line_and_never_reaches_auth() {
    assert_each_refusal_logs_one_line_and_never_reaches_auth(GateClosing::Stale).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The exits (SDD SR-6.4; PM B1)
// ═════════════════════════════════════════════════════════════════════════════

/// A held replica whose marker is removed calls the exit seam with 0, within
/// one poll interval plus the jitter bound, and its gate stays closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_removed_marker_makes_a_held_replica_exit_zero() {
    let tv = TestValkey::setup().await;
    let mut wiring = HoldWiring::start(&tv).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("hold-exit"));
    let parts = wiring.serve(GracefulShutdown::new(), Arc::clone(&metrics));

    tv.write_hold_marker(&HoldMarkerSeed::default().with_incident(INCIDENT))
        .await;
    wait_until("the hold entry", || {
        parts.gate.state() == HoldGateState::Held
    })
    .await;
    assert!(
        tokio::time::timeout(Duration::from_secs(1), wiring.exits.recv())
            .await
            .is_err(),
        "a held replica does not exit while the marker stands"
    );

    assert!(tv.remove_hold_marker().await, "fixture: the marker existed");
    let code = wiring.next_exit(EXIT_DEADLINE, "the marker removed").await;

    assert_eq!(
        code, 0,
        "an un-held replica exits 0 to restart into normal service"
    );
    assert!(
        EXIT_DEADLINE > POLL_INTERVAL + UNHOLD_EXIT_JITTER_MAX,
        "the deadline covers one poll interval and the jitter bound"
    );
    assert_eq!(
        parts.gate.state(),
        HoldGateState::Held,
        "the gate never reopens in place"
    );
}

/// A marker that appears after startup but before the serving point makes
/// the replica exit 0 through the seam, at once, with the point still
/// unreached. Before the serving point the reactor holds no entry parts, so
/// there is nothing of the entry routine to observe here; that it does not
/// run (its ERROR line, on a paused clock) is `hold_reactor_test.rs`'s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_marker_before_the_serving_point_exits_zero_without_entering_the_hold() {
    let tv = TestValkey::setup().await;
    let mut wiring = HoldWiring::start(&tv).await;

    tv.write_hold_marker(&HoldMarkerSeed::default().with_incident(INCIDENT))
        .await;
    let code = wiring
        .next_exit(SETTLE, "a marker before the serving point")
        .await;

    assert_eq!(code, 0);
    assert!(!wiring.point.is_reached());
}

// ═════════════════════════════════════════════════════════════════════════════
// A stale verdict (SDD SR-6.5)
// ═════════════════════════════════════════════════════════════════════════════

/// A serving replica whose hold poll loses its connection — the dead
/// connection to the counter's shard SR-6.5 is about, staged as a stalled
/// connection of the poll's own client while the service's clients keep
/// answering — fails closed on its own about 30 s after its last read that
/// answered: the open subscription ends `UPSTREAM_UNAVAILABLE` (never
/// `SERVICE_HELD`), every LiveService RPC is refused the same way before
/// auth, the registered writer is aborted, `live_edit_dr_hold_verdict_stale`
/// is 1, and the health supervisor, fed the reactor's signal as `main.rs`
/// feeds it, takes readiness to `NOT_SERVING` while the store's `PING`
/// still passes. No marker is written: a fresh startup evaluation still reads
/// `Serving`. It is no hold and no shutdown.
///
/// Once the connection answers again, the next read exits the replica 0,
/// and a server rebuilt on the same store, as the restarted process builds
/// it, admits a subscription.
///
/// The 30 s threshold is the production constant and is not injectable, so
/// this test runs 30 to 40 s in a healthy run: the end-to-end proof that a
/// real stalled read reaches the reactor's clock. Every reaction's timing is
/// pinned on virtual time in `hold_reactor_test.rs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dead_hold_poll_connection_fails_the_replica_closed_and_its_next_read_restarts_it() {
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, Arc::clone(&permissions) as _);
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "hold-stale-real-server", LAYER).await;

    // The poll reads through a client of its own, the connection this test
    // kills; the service keeps the fixture's.
    let poll_client = one_connection_client().await;
    let mut wiring = HoldWiring::start_on(&tv, poll_client.clone()).await;
    let parts = wiring.serve(svc.shutdown.clone(), Arc::clone(&svc.metrics));
    let writer = CancellationToken::new();
    parts
        .writers
        .register_token(HoldWriter::CheckpointLoop, writer.clone());

    // Readiness as `main.rs` wires it: the reactor's signal into the
    // supervisor, beside a Valkey probe that keeps passing.
    let (reporter, _) = create_health_service(READINESS).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        tv.client().clone(),
        Arc::clone(&svc.metrics),
        HealthSupervisorConfig {
            probe_interval: Duration::from_secs(1),
            probe_timeout: Duration::from_secs(1),
            readiness_service_name: READINESS.to_owned(),
            service_token_provider: None,
            counter_seed_ready: None,
            api_serving: None,
            verdict_stale: Some(wiring.verdict_stale.clone()),
        },
        supervisor_cancel.clone(),
    );
    until_readiness(
        HealthSource::Reporter(&reporter),
        tonic_health::pb::health_check_response::ServingStatus::Serving,
        SETTLE,
    )
    .await;

    let server = HeldServer::start(svc.clone(), parts.gate.clone(), 30_000, 2_000).await;
    let channel = server.channel().await;
    let mut client = LiveServiceClient::new(channel.clone());
    let mut subscription = client
        .subscribe_read_layer(with_bearer(subscribe_request(&session_id, "stale-reader")))
        .await
        .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
        .into_inner();
    read_catch_up(&mut subscription).await;
    let session_token = svc
        .session_tokens
        .get(&session_id)
        .expect("the subscription registered the session's token")
        .clone();

    // The poll's connection dies: every read it sends from now on is queued
    // behind a block and answers nothing.
    let stalls = stall_pool(
        &poll_client.pool,
        &tv.client().maintenance_pool.next().clone(),
        &format!("{}conn-stall:hold-poll:", tv.prefix()),
    )
    .await;
    let stalled = tokio::time::Instant::now();
    assert!(
        STALE_DEADLINE > STALE_VERDICT_THRESHOLD,
        "the deadline covers the threshold"
    );
    while parts.gate.state() != HoldGateState::Stale {
        assert!(
            tokio::time::Instant::now() < stalled + STALE_DEADLINE,
            "no stale-verdict entry within {STALE_DEADLINE:?} of the poll's connection stalling"
        );
        assert!(
            wiring.exits.try_recv().is_err(),
            "a serving replica whose verdict goes stale does not exit"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The stream ends, and every call is refused, UPSTREAM_UNAVAILABLE.
    let sub_end = terminal_status(&mut subscription, "the subscription")
        .await
        .expect("the subscription ended OK, which the Kit client never retries");
    assert_stale_verdict(&sub_end);
    let asked_before_refusals = permissions.calls().len();
    let stat = client
        .stat_layer(with_bearer(stat_request(&session_id)))
        .await
        .expect_err("StatLayer is refused while stale");
    assert_stale_verdict(&stat);
    let late = client
        .subscribe_read_layer(with_bearer(subscribe_request(&session_id, "late-reader")))
        .await
        .expect_err("a new SubscribeReadLayer is refused while stale");
    assert_stale_verdict(&late);
    assert_eq!(
        permissions.calls().len(),
        asked_before_refusals,
        "no refused call reaches the permission check"
    );

    // The writers, the signals, readiness.
    assert!(parts.writers.is_aborted());
    assert!(writer.is_cancelled(), "the registered writer is stopped");
    assert_eq!(verdict_stale_gauge(&svc.metrics), 1);
    assert_eq!(dr_held(&svc.metrics), 0, "a stale verdict is not a hold");
    until_readiness(
        HealthSource::Reporter(&reporter),
        tonic_health::pb::health_check_response::ServingStatus::NotServing,
        SETTLE,
    )
    .await;

    // No marker, no shutdown, no session ending.
    let reread =
        evaluate_hold_at_startup(tv.client(), tv.key_schema(), answering(COMMAND_TIMEOUT_MS))
            .await
            .expect("the store answers the service's client");
    assert!(
        !reread.verdict.is_held(),
        "a stale verdict writes no hold marker"
    );
    assert!(
        !svc.shutdown.begun().is_cancelled() && !svc.shutdown.is_cancelled(),
        "a stale verdict starts no shutdown"
    );
    assert!(
        !session_token.is_cancelled(),
        "a stale verdict cancels no per-session token"
    );

    // The connection answers again: the next read exits the replica 0.
    release_all(stalls).await;
    let code = wiring
        .next_exit(EXIT_DEADLINE, "the poll's reads answering again")
        .await;
    assert_eq!(code, 0, "a stale replica whose read succeeds exits 0");
    assert_eq!(
        parts.gate.state(),
        HoldGateState::Stale,
        "the gate never reopens in place"
    );

    drop(subscription);
    drop(client);
    drop(channel);
    server.shutdown().await;
    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(SETTLE, supervisor).await;
    drop(wiring);

    // The restarted process: a fresh startup evaluation, an open gate, a new
    // server on the same store. It serves.
    let startup =
        evaluate_hold_at_startup(tv.client(), tv.key_schema(), answering(COMMAND_TIMEOUT_MS))
            .await
            .expect("the restarted process's startup evaluation answers");
    assert!(!startup.verdict.is_held(), "it starts serving, not held");
    let (_startup_tx, startup_rx) = watch::channel(startup);
    let rebuilt_svc = build_service(&tv, Arc::clone(&permissions) as _);
    let rebuilt = HeldServer::start(rebuilt_svc, HoldGate::new(startup_rx), 30_000, 2_000).await;
    let mut rebuilt_client = LiveServiceClient::new(rebuilt.channel().await);
    let mut resubscription = rebuilt_client
        .subscribe_read_layer(with_bearer(subscribe_request(
            &session_id,
            "restarted-reader",
        )))
        .await
        .unwrap_or_else(|s| panic!("the restarted server must admit a subscription: {s:?}"))
        .into_inner();
    read_catch_up(&mut resubscription).await;

    drop(resubscription);
    drop(rebuilt_client);
    rebuilt.shutdown().await;
}
