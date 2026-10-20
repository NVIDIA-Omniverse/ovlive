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

//! `run::Scaler` end to end: one `wiremock` server plays the Kubernetes API
//! server and one per server pod plays that pod's metrics port. Each pod has
//! its own EndpointSlice, whose `metrics` port is its mock's port, all on
//! 127.0.0.1.
//!
//! Polls run in real time (`poll_once`, or `run` for the cancellation and
//! liveness-loop tests): the scaler's request timeouts are tokio timers,
//! which a paused clock would fire while a real socket is still answering.
//! The two liveness tests and the window-restart test pause the clock only
//! for an instant ([`jump_clock`]): pause, advance (past the heartbeat's
//! bound, or half a scale-down window), resume, while no request is in
//! flight, so no request timer can fire early.
//! Scale-down tests set the window and interval to 0, so the first poll may
//! scale down, except the window-restart test, whose 600 s window the clock
//! jumps cross. What the scaler sent the cluster is read from the API
//! server's request log: the exact sequence of verbs and paths it received.
//!
//! **Wait budget.** Every wait is a deadline, not a sleep: [`SETTLE`] (5 s)
//! for each polled condition (in the cancellation and liveness-loop tests)
//! and for each `/healthz` or `/metrics` GET (in the HTTP liveness test and
//! the λ̂ gauge tests), and [`RUN_EXIT`] (10 s) for `run` to return after
//! its cancel. Requests to the fake cluster are bounded by the scaler's own
//! timeouts (2 s per pod snapshot, 4 s per API request). Worst case per
//! test, every wait at its full budget:
//! - `healthz_follows_the_heartbeat_over_http` and
//!   `catchup_rate_gauge_follows_each_decided_poll_and_drops_what_is_gone`:
//!   four GETs each, 4 × 5 s = 20 s, the heaviest;
//! - the two cancellation tests and the liveness-loop test: one condition
//!   and one `run` exit, 5 s + 10 s = 15 s. The mid-poll test's held
//!   `/scale` answer ([`HELD_ANSWER`], 3 s) is paid only by a loop that
//!   fails to cancel;
//! - `an_api_read_failure_leaves_the_catchup_rate_and_pod_gauges_as_they_were`:
//!   two scrapes, 10 s;
//! - the rest: no wait of their own, only the scaler's request timeouts.
//!
//! All are far inside nextest's 120 s hang-killer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use live_edit_scaler::config::ScalerConfig;
use live_edit_scaler::decide::{Action, BlockReason, UpReason};
use live_edit_scaler::kube::KubeClient;
use live_edit_scaler::obs::{Heartbeat, ScalerMetrics};
use live_edit_scaler::run::Scaler;
use live_edit_scaler::snapshot::ScalingSnapshot;
use serde_json::json;
use tempfile::NamedTempFile;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::fmt::MakeWriter;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common::{idle_snapshot, render, sample};

const NAMESPACE: &str = "editing";
const DEPLOYMENT: &str = "live-edit-service";
const SERVICE: &str = "live-edit-service";
const SLICES_PATH: &str = "/apis/discovery.k8s.io/v1/namespaces/editing/endpointslices";
const SCALE_PATH: &str = "/apis/apps/v1/namespaces/editing/deployments/live-edit-service/scale";
const PODS_PATH: &str = "/api/v1/namespaces/editing/pods/";

/// Deadline for each polled condition in the cancellation tests and the
/// liveness-loop test, and for each `/healthz` or `/metrics` GET in
/// `healthz_follows_the_heartbeat_over_http` and the λ̂ gauge tests.
const SETTLE: Duration = Duration::from_secs(5);
/// Deadline for `Scaler::run` to return once cancelled.
const RUN_EXIT: Duration = Duration::from_secs(10);
/// How long the API server holds its `/scale` answer in the mid-poll
/// cancellation test. Under the scaler's 4 s API request timeout, so a poll
/// that is not cancelled gets the answer and goes on to scale up — which is
/// what the test must see not happen.
const HELD_ANSWER: Duration = Duration::from_secs(3);

/// The WARN `Scaler` logs when Ready pods exist but none answered
/// `/scaling/v1`.
const NO_SNAPSHOT_WARN: &str = "no Ready server pod answered /scaling/v1";

/// What a fake server pod's metrics port answers on `/scaling/v1`.
#[derive(Clone)]
enum Answer {
    Snapshot(ScalingSnapshot),
    /// A 200 with this exact JSON body.
    Json(serde_json::Value),
    Status(u16),
    /// A 200 whose body is not a snapshot.
    Garbled,
    /// The port accepts the connection and closes it without answering.
    Hangup,
}

/// One server pod: its EndpointSlice state and what its metrics port does.
struct PodSpec {
    name: &'static str,
    ready: bool,
    terminating: bool,
    answer: Answer,
}

fn ready(name: &'static str, answer: Answer) -> PodSpec {
    PodSpec {
        name,
        ready: true,
        terminating: false,
        answer,
    }
}

fn idle() -> Answer {
    Answer::Snapshot(idle_snapshot())
}

fn with(edit: impl FnOnce(&mut ScalingSnapshot)) -> Answer {
    let mut s = idle_snapshot();
    edit(&mut s);
    Answer::Snapshot(s)
}

/// A listener held for the fake pod's lifetime that accepts every connection
/// and drops it unanswered, counting them. Holding the port (rather than
/// binding and releasing one so the connection is refused) keeps another
/// test's mock server from taking it meanwhile.
struct HangupPort {
    accepted: Arc<std::sync::atomic::AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl HangupPort {
    async fn start() -> (Self, u16) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback port");
        let port = listener.local_addr().expect("its address").port();
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&accepted);
        let task = tokio::spawn(async move {
            while let Ok((connection, _)) = listener.accept().await {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(connection);
            }
        });
        (Self { accepted, task }, port)
    }
}

impl Drop for HangupPort {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A fake pod's metrics port: a mock server, or for [`Answer::Hangup`] a
/// [`HangupPort`].
struct FakePod {
    port: u16,
    server: Option<MockServer>,
    hangup: Option<HangupPort>,
}

impl FakePod {
    async fn start(answer: &Answer) -> Self {
        let response = match answer {
            Answer::Hangup => {
                let (hangup, port) = HangupPort::start().await;
                return Self {
                    port,
                    server: None,
                    hangup: Some(hangup),
                };
            }
            Answer::Snapshot(s) => ResponseTemplate::new(200).set_body_json(s),
            Answer::Json(body) => ResponseTemplate::new(200).set_body_json(body),
            Answer::Status(code) => ResponseTemplate::new(*code),
            Answer::Garbled => ResponseTemplate::new(200).set_body_string("{ not a snapshot"),
        };
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/scaling/v1"))
            .respond_with(response)
            .mount(&server)
            .await;
        Self {
            port: server.address().port(),
            server: Some(server),
            hangup: None,
        }
    }

    /// Requests the mock server received, or connections the hangup port
    /// accepted.
    async fn requests(&self) -> usize {
        match (&self.server, &self.hangup) {
            (Some(s), _) => s.received_requests().await.expect("recording is on").len(),
            (None, Some(h)) => h.accepted.load(std::sync::atomic::Ordering::SeqCst),
            (None, None) => 0,
        }
    }
}

/// The fake cluster: an API server with the service's EndpointSlices and
/// the Deployment's `/scale`, and one fake metrics port per pod.
struct Cluster {
    api: MockServer,
    pods: Vec<FakePod>,
    token: NamedTempFile,
}

impl Cluster {
    /// `replicas` is `spec.replicas`, `status_replicas` `status.replicas`.
    /// Every PATCH (on `/scale` or a pod) is accepted.
    async fn start(pods: &[PodSpec], replicas: u32, status_replicas: u32) -> Self {
        let mut fakes = Vec::new();
        for spec in pods {
            fakes.push(FakePod::start(&spec.answer).await);
        }
        let slices: Vec<_> = pods
            .iter()
            .zip(&fakes)
            .map(|(spec, fake)| {
                json!({
                    "addressType": "IPv4",
                    "ports": [{ "name": "metrics", "port": fake.port }],
                    "endpoints": [{
                        "addresses": ["127.0.0.1"],
                        "conditions": { "ready": spec.ready, "terminating": spec.terminating },
                        "targetRef": { "kind": "Pod", "name": spec.name, "uid": format!("uid-{}", spec.name) }
                    }]
                })
            })
            .collect();

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(SLICES_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "items": slices })))
            .mount(&api)
            .await;
        Mock::given(method("GET"))
            .and(path(SCALE_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "spec": { "replicas": replicas },
                "status": { "replicas": status_replicas }
            })))
            .mount(&api)
            .await;
        Mock::given(method("PATCH"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&api)
            .await;

        let mut token = NamedTempFile::new().expect("a token file");
        std::io::Write::write_all(&mut token, b"scaler-token\n").expect("write the token");
        Self {
            api,
            pods: fakes,
            token,
        }
    }

    /// Mount `mock` ahead of the healthy defaults [`Self::start`] mounted:
    /// wiremock serves the highest-priority match, and 1 is the highest.
    async fn override_api(&self, mock: Mock) {
        mock.with_priority(1).mount(&self.api).await;
    }

    fn scaler(&self, cfg: ScalerConfig) -> (Scaler, Arc<ScalerMetrics>) {
        self.scaler_with_heartbeat(cfg, Arc::new(Heartbeat::new(Duration::from_secs(60))))
    }

    /// [`Self::scaler`] beating `heartbeat`.
    fn scaler_with_heartbeat(
        &self,
        cfg: ScalerConfig,
        heartbeat: Arc<Heartbeat>,
    ) -> (Scaler, Arc<ScalerMetrics>) {
        let kube = KubeClient::new(&self.api.uri(), self.token.path(), None)
            .expect("an http client for the mock API server");
        let metrics = Arc::new(ScalerMetrics::new());
        let scaler = Scaler::new(cfg, kube, Arc::clone(&metrics), heartbeat)
            .expect("the scaler's pod client");
        (scaler, metrics)
    }

    /// Every request the API server received, as `(verb, path, body)`, in
    /// arrival order.
    async fn api_requests(&self) -> Vec<(String, String, serde_json::Value)> {
        self.api
            .received_requests()
            .await
            .expect("recording is on")
            .into_iter()
            .map(|r| {
                let body = if r.body.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::from_slice(&r.body).expect("a request body is JSON")
                };
                (r.method.to_string(), r.url.path().to_owned(), body)
            })
            .collect()
    }
}

/// The scaler settings for these tests: window and interval 0, so a
/// scale-down needs no history; the chart's defaults otherwise.
fn config(pod_deletion_cost: bool) -> ScalerConfig {
    ScalerConfig {
        namespace: NAMESPACE.to_owned(),
        deployment: DEPLOYMENT.to_owned(),
        service: SERVICE.to_owned(),
        min_replicas: 1,
        max_replicas: 10,
        cpu: None,
        poll_interval: Duration::from_secs(5),
        catchup_target: Duration::from_secs(10),
        occupancy_headroom_percent: 80,
        scale_down_window: Duration::ZERO,
        scale_down_interval: Duration::ZERO,
        pod_deletion_cost,
        http_port: 9090,
    }
}

/// `(verb, path)` of the two reads every poll starts with: one EndpointSlice
/// list and one `/scale` GET.
fn poll_reads() -> Vec<(String, String)> {
    vec![
        ("GET".to_owned(), SLICES_PATH.to_owned()),
        ("GET".to_owned(), SCALE_PATH.to_owned()),
    ]
}

fn verbs_and_paths(requests: &[(String, String, serde_json::Value)]) -> Vec<(String, String)> {
    requests
        .iter()
        .map(|(m, p, _)| (m.clone(), p.clone()))
        .collect()
}

/// A join-burst backlog on every pod scales up with exactly one PATCH of
/// `/scale`, after exactly the poll's two reads — one EndpointSlice list
/// and one `/scale` GET, and never a GET of the Deployment object.
/// 1 200 pending catch-ups at the prior rate need 18 replicas, clamped to
/// 10; one poll adds at most 4 to the 3 there are.
#[tokio::test]
async fn a_backlog_wave_scales_up_with_one_scale_patch_after_one_list_and_one_scale_get() {
    let backlog = || with(|s| s.catchups_pending = 400);
    let cluster = Cluster::start(
        &[
            ready("server-a", backlog()),
            ready("server-b", backlog()),
            ready("server-c", backlog()),
        ],
        3,
        3,
    )
    .await;
    let (mut scaler, _) = cluster.scaler(config(false));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(
        decision.action,
        Action::ScaleUp {
            to: 7,
            reason: UpReason::Catchup
        }
    );
    let requests = cluster.api_requests().await;
    let mut expected = poll_reads();
    expected.push(("PATCH".to_owned(), SCALE_PATH.to_owned()));
    assert_eq!(verbs_and_paths(&requests), expected);
    assert_eq!(requests[2].2, json!({ "spec": { "replicas": 7 } }));
    for pod in &cluster.pods {
        assert_eq!(
            pod.requests().await,
            1,
            "each pod's snapshot is read once per poll"
        );
    }
}

/// With deletion cost on, every pod's cost is written before — and only
/// before — the scale-down PATCH, the lowest on the emptiest pod (cost =
/// admitted subscriptions + pending catch-ups).
#[tokio::test]
async fn deletion_costs_are_written_just_before_the_scale_down_lowest_on_the_emptiest_pod() {
    let cluster = Cluster::start(
        &[
            ready("server-a", with(|s| s.subscriptions_admitted = 30)),
            ready("server-b", with(|s| s.subscriptions_admitted = 5)),
            ready("server-c", with(|s| s.subscriptions_admitted = 12)),
        ],
        3,
        3,
    )
    .await;
    let (mut scaler, _) = cluster.scaler(config(true));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert!(
        matches!(decision.action, Action::ScaleDown { to: 2, .. }),
        "{:?}",
        decision.action
    );

    let requests = cluster.api_requests().await;
    assert_eq!(verbs_and_paths(&requests[..2]), poll_reads());
    let costs: HashMap<String, serde_json::Value> = requests[2..5]
        .iter()
        .map(|(verb, path, body)| {
            assert_eq!(verb, "PATCH");
            let pod = path.strip_prefix(PODS_PATH).unwrap_or_else(|| {
                panic!("a pod annotation PATCH before the scale-down, got {path}")
            });
            (
                pod.to_owned(),
                body["metadata"]["annotations"]["controller.kubernetes.io/pod-deletion-cost"]
                    .clone(),
            )
        })
        .collect();
    assert_eq!(
        costs,
        HashMap::from([
            ("server-a".to_owned(), json!("30")),
            ("server-b".to_owned(), json!("5")),
            ("server-c".to_owned(), json!("12")),
        ])
    );
    assert_eq!(
        requests[5],
        (
            "PATCH".to_owned(),
            SCALE_PATH.to_owned(),
            json!({ "spec": { "replicas": 2 } })
        ),
        "the scale-down comes last"
    );
    assert_eq!(requests.len(), 6);
}

/// The same idle fleet without deletion cost scales down with no pod PATCH:
/// the baseline the held cases below are measured against.
#[tokio::test]
async fn an_idle_settled_fleet_scales_down_with_one_scale_patch_and_no_pod_patch() {
    let cluster = Cluster::start(
        &[
            ready("server-a", idle()),
            ready("server-b", idle()),
            ready("server-c", idle()),
        ],
        3,
        3,
    )
    .await;
    let (mut scaler, _) = cluster.scaler(config(false));
    scaler.poll_once().await.expect("the API reads succeed");
    let requests = cluster.api_requests().await;
    let mut expected = poll_reads();
    expected.push(("PATCH".to_owned(), SCALE_PATH.to_owned()));
    assert_eq!(verbs_and_paths(&requests), expected);
    assert_eq!(requests[2].2, json!({ "spec": { "replicas": 2 } }));
}

/// When no pod answers with a snapshot — one answers 404, one 503, one
/// closes the connection unanswered, one sends a body that is not a
/// snapshot — the scaler decides nothing and writes nothing, though at an
/// idle reading the same fleet would scale down. The two pods that gave no
/// usable answer count as snapshot errors; the 404 and 503 do not (they
/// answered).
#[tokio::test]
async fn no_patch_is_made_when_no_pod_reports() {
    let cluster = Cluster::start(
        &[
            ready("server-a", Answer::Status(404)),
            ready("server-b", Answer::Status(503)),
            ready("server-c", Answer::Hangup),
            ready("server-d", Answer::Garbled),
        ],
        4,
        4,
    )
    .await;
    let (mut scaler, metrics) = cluster.scaler(config(true));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(decision.action, Action::NoData);
    assert_eq!(verbs_and_paths(&cluster.api_requests().await), poll_reads());
    assert_eq!(
        cluster.pods[2].requests().await,
        1,
        "fixture self-check: the hangup pod was contacted and closed the connection"
    );
    let text = render(&metrics);
    assert_eq!(
        sample(
            &text,
            "live_edit_scaler_poll_errors_total",
            &[("stage", "snapshot")]
        ),
        Some(2.0)
    );
    assert_eq!(
        sample(
            &text,
            "live_edit_scaler_decisions_total",
            &[("direction", "none"), ("reason", "no_data")]
        ),
        Some(1.0)
    );
}

/// A pod whose snapshot was written by another build — without fields this
/// build added, and with one this build does not know — still reports: its
/// body parses, its queue counts, and the fleet scales up on it.
#[tokio::test]
async fn a_snapshot_without_newer_fields_and_with_unknown_ones_still_reports() {
    let other_build = json!({
        "version": 1,
        "ready": true,
        "catchups_pending": 1_000,
        "a_field_from_a_later_build": { "nested": [1, 2, 3] }
    });
    let cluster = Cluster::start(&[ready("server-a", Answer::Json(other_build))], 1, 1).await;
    let (mut scaler, _) = cluster.scaler(config(false));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(decision.pods.reporting, 1, "the pod reports");
    assert_eq!(decision.terms.catchup, 15, "ceil(1 000 ÷ 70)");
    assert_eq!(
        decision.action,
        Action::ScaleUp {
            to: 5,
            reason: UpReason::Catchup
        }
    );
}

/// A terminating endpoint holds the scale-down (`unsettled`), and its pod is
/// never asked for a snapshot: it is leaving, and not load.
#[tokio::test]
async fn a_fleet_with_a_terminating_endpoint_gets_no_scale_down() {
    let cluster = Cluster::start(
        &[
            ready("server-a", idle()),
            ready("server-b", idle()),
            ready("server-c", idle()),
            PodSpec {
                name: "server-old",
                ready: false,
                terminating: true,
                answer: idle(),
            },
        ],
        3,
        3,
    )
    .await;
    let (mut scaler, metrics) = cluster.scaler(config(false));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(
        (decision.action, decision.scale_down_blocked),
        (Action::Hold, Some(BlockReason::Unsettled))
    );
    assert_eq!(verbs_and_paths(&cluster.api_requests().await), poll_reads());
    assert_eq!(
        cluster.pods[3].requests().await,
        0,
        "a terminating pod is not polled"
    );
    assert_eq!(
        sample(
            &render(&metrics),
            "live_edit_scaler_scale_down_blocked_total",
            &[("reason", "unsettled")]
        ),
        Some(1.0)
    );
}

/// With deletion cost on, a Ready pod that does not report holds the
/// scale-down, so no pod is annotated and `/scale` is not patched.
#[tokio::test]
async fn a_not_reporting_ready_pod_with_deletion_cost_on_gets_no_pod_or_scale_patch() {
    let cluster = Cluster::start(
        &[
            ready("server-a", idle()),
            ready("server-b", idle()),
            ready("server-c", Answer::Status(404)),
        ],
        3,
        3,
    )
    .await;
    let (mut scaler, _) = cluster.scaler(config(true));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(decision.scale_down_blocked, Some(BlockReason::Unsettled));
    assert_eq!(verbs_and_paths(&cluster.api_requests().await), poll_reads());
}

/// A failed API read ends the poll with no action: an EndpointSlice list
/// refused reads nothing further and polls no pod; a `/scale` GET that fails
/// polls no pod either. Each counts on its own stage.
#[tokio::test]
async fn a_failed_api_read_ends_the_poll_with_no_action() {
    for (stage, failing_path, reads) in
        [("endpoints", SLICES_PATH, 1usize), ("scale", SCALE_PATH, 2)]
    {
        // A backlog the scaler would act on, had it read the cluster state.
        let cluster = Cluster::start(
            &[ready("server-a", with(|s| s.catchups_pending = 10_000))],
            1,
            1,
        )
        .await;
        cluster
            .override_api(
                Mock::given(method("GET"))
                    .and(path(failing_path))
                    .respond_with(ResponseTemplate::new(500)),
            )
            .await;
        let (mut scaler, metrics) = cluster.scaler(config(false));

        assert!(scaler.poll_once().await.is_none(), "{stage}: no decision");
        let requests = cluster.api_requests().await;
        assert_eq!(requests.len(), reads, "{stage}: {requests:?}");
        assert!(
            requests.iter().all(|(verb, _, _)| verb == "GET"),
            "{stage}: no write"
        );
        assert_eq!(
            cluster.pods[0].requests().await,
            0,
            "{stage}: no pod is polled"
        );
        assert_eq!(
            sample(
                &render(&metrics),
                "live_edit_scaler_poll_errors_total",
                &[("stage", stage)]
            ),
            Some(1.0),
            "{stage}"
        );
    }
}

/// A deletion-cost write that fails cancels the scale-down: without the
/// marks the ReplicaSet would not necessarily remove the emptiest pod, so the
/// scale-down could end more streams than it needs to.
#[tokio::test]
async fn a_failed_deletion_cost_write_cancels_the_scale_down() {
    let cluster = Cluster::start(
        &[
            ready("server-a", idle()),
            ready("server-b", idle()),
            ready("server-c", idle()),
        ],
        3,
        3,
    )
    .await;
    cluster
        .override_api(
            Mock::given(method("PATCH"))
                .and(path(format!("{PODS_PATH}server-b")))
                .respond_with(ResponseTemplate::new(403)),
        )
        .await;
    let (mut scaler, metrics) = cluster.scaler(config(true));

    let decision = scaler.poll_once().await.expect("the API reads succeed");
    assert!(
        matches!(decision.action, Action::ScaleDown { .. }),
        "{:?}",
        decision.action
    );
    let requests = cluster.api_requests().await;
    assert_eq!(
        requests
            .iter()
            .filter(|(verb, _, _)| verb == "PATCH")
            .count(),
        3,
        "the three deletion-cost writes were attempted: {requests:?}"
    );
    assert!(
        requests
            .iter()
            .all(|(verb, path, _)| !(verb == "PATCH" && path == SCALE_PATH)),
        "no scale-down after a failed deletion-cost write: {requests:?}"
    );
    assert_eq!(
        sample(
            &render(&metrics),
            "live_edit_scaler_poll_errors_total",
            &[("stage", "deletion_cost")]
        ),
        Some(1.0)
    );
}

// ── The no-snapshot WARN ───────────────────────────────────────────────────

/// `MakeWriter` capturing every byte into a shared buffer.
#[derive(Clone, Default)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl CapturingWriter {
    /// The captured WARN records carrying [`NO_SNAPSHOT_WARN`], as JSON.
    fn no_snapshot_warns(&self) -> Vec<serde_json::Value> {
        let text = String::from_utf8(self.0.lock().unwrap().clone()).expect("UTF-8 log");
        text.lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a JSON record"))
            .filter(|r| {
                r["level"] == "WARN"
                    && r["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.starts_with(NO_SNAPSHOT_WARN))
            })
            .collect()
    }
}

/// Point the single fake pod's `/scaling/v1` at `answer`.
async fn answer_with(pod: &FakePod, answer: ResponseTemplate) {
    let server = pod.server.as_ref().expect("a listening fake pod");
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/scaling/v1"))
        .respond_with(answer)
        .mount(server)
        .await;
}

/// The scaler WARNs when Ready server pods exist but none of them answered
/// `/scaling/v1` — once, not every poll, and again only after a poll that
/// did reach a Ready pod's snapshot. The line is the operator's only signal
/// that the scaler is deciding nothing, so its level, message and counts
/// are pinned. No Ready pod at all is not this condition and logs nothing.
///
/// Runs on the current-thread runtime with a thread-local subscriber: the
/// poll's pod requests run inside the test task (`join_all`), not on spawned
/// tasks, so every event is captured.
#[tokio::test]
async fn the_no_snapshot_warn_fires_once_and_rearms_only_after_a_successful_poll() {
    let writer = CapturingWriter::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(writer.clone())
        .with_max_level(tracing::Level::WARN)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    // No Ready pod: nothing to warn about.
    let unready = Cluster::start(
        &[PodSpec {
            name: "server-a",
            ready: false,
            terminating: false,
            answer: Answer::Status(404),
        }],
        1,
        1,
    )
    .await;
    let (mut scaler, _) = unready.scaler(config(false));
    scaler.poll_once().await.expect("the API reads succeed");
    assert!(
        writer.no_snapshot_warns().is_empty(),
        "no Ready pod, no WARN"
    );

    let cluster = Cluster::start(&[ready("server-a", Answer::Status(404))], 1, 1).await;
    let (mut scaler, _) = cluster.scaler(config(false));
    let pod = &cluster.pods[0];

    scaler.poll_once().await.expect("the API reads succeed");
    let warns = writer.no_snapshot_warns();
    assert_eq!(warns.len(), 1, "the first such poll WARNs");
    assert_eq!(warns[0]["fields"]["ready_pods"], 1);
    assert_eq!(warns[0]["fields"]["not_found"], 1);
    assert_eq!(warns[0]["fields"]["not_ready_to_report"], 0);
    assert_eq!(warns[0]["fields"]["unreachable"], 0);

    scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(
        writer.no_snapshot_warns().len(),
        1,
        "a lasting condition WARNs once"
    );

    answer_with(
        pod,
        ResponseTemplate::new(200).set_body_json(idle_snapshot()),
    )
    .await;
    scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(
        writer.no_snapshot_warns().len(),
        1,
        "a reporting pod: no WARN"
    );

    answer_with(pod, ResponseTemplate::new(503)).await;
    scaler.poll_once().await.expect("the API reads succeed");
    let warns = writer.no_snapshot_warns();
    assert_eq!(warns.len(), 2, "re-armed by the successful poll");
    assert_eq!(warns[1]["fields"]["not_ready_to_report"], 1);
}

// ── Cancellation ───────────────────────────────────────────────────────────

/// Wait (up to [`SETTLE`]) until `check` holds.
async fn settle<F, Fut>(what: &str, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    tokio::time::timeout(SETTLE, async {
        while !check().await {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what} did not happen within {SETTLE:?}"));
}

/// Cancelling the loop in the middle of a poll ends it at once, with no
/// scale call: the API server holds its `/scale` answer for [`HELD_ANSWER`],
/// the fleet's backlog would scale up once the poll got it, and `run` must
/// return having sent nothing but the poll's reads. A loop that let the
/// in-flight poll finish would PATCH `/scale` before returning.
#[tokio::test]
async fn cancelling_mid_poll_stops_the_loop_with_no_scale_call() {
    let cluster = Cluster::start(
        &[ready("server-a", with(|s| s.catchups_pending = 10_000))],
        1,
        1,
    )
    .await;
    cluster
        .override_api(
            Mock::given(method("GET"))
                .and(path(SCALE_PATH))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(
                            json!({ "spec": { "replicas": 1 }, "status": { "replicas": 1 } }),
                        )
                        .set_delay(HELD_ANSWER),
                ),
        )
        .await;
    let (scaler, _) = cluster.scaler(config(false));

    let cancel = CancellationToken::new();
    let run = tokio::spawn(scaler.run(cancel.clone()));
    settle("the poll's /scale GET", || async {
        cluster
            .api_requests()
            .await
            .iter()
            .any(|(_, p, _)| p == SCALE_PATH)
    })
    .await;
    cancel.cancel();
    tokio::time::timeout(RUN_EXIT, run)
        .await
        .expect("run returns at once when cancelled mid-poll")
        .expect("the loop does not panic");

    assert_eq!(verbs_and_paths(&cluster.api_requests().await), poll_reads());
    assert_eq!(
        cluster.pods[0].requests().await,
        0,
        "the poll ended before its snapshots"
    );
}

/// Cancelling the loop between polls ends it without another poll and with
/// no scale call of its own.
#[tokio::test]
async fn cancelling_between_polls_stops_the_loop_with_no_scale_call() {
    let cluster = Cluster::start(&[ready("server-a", idle())], 1, 1).await;
    let mut cfg = config(false);
    // One poll at once, the next an hour away.
    cfg.poll_interval = Duration::from_secs(3_600);
    let (scaler, metrics) = cluster.scaler(cfg);

    let cancel = CancellationToken::new();
    let run = tokio::spawn(scaler.run(cancel.clone()));
    settle("the first poll's decision", || async {
        sample(
            &render(&metrics),
            "live_edit_scaler_decisions_total",
            &[("direction", "none"), ("reason", "at_target")],
        ) == Some(1.0)
    })
    .await;
    cancel.cancel();
    tokio::time::timeout(RUN_EXIT, run)
        .await
        .expect("run returns at once when cancelled between polls")
        .expect("the loop does not panic");

    assert_eq!(
        verbs_and_paths(&cluster.api_requests().await),
        poll_reads(),
        "one poll's reads, and no write"
    );
}

// ── Liveness ───────────────────────────────────────────────────────────────

/// Move tokio's clock forward by `by` without waiting, then let it run on in
/// real time. The heartbeat and the scaler's engine read tokio's clock, so
/// this makes a heartbeat stale, or carries the engine across its scale-down
/// window, while every socket in the test keeps real time.
async fn jump_clock(by: Duration) {
    tokio::time::pause();
    tokio::time::advance(by).await;
    tokio::time::resume();
}

/// A poll beats the heartbeat whatever its outcome: here the EndpointSlice
/// list fails, so the poll ends with no decision, and a heartbeat that had
/// gone stale is fresh again. A scaler riding out an API outage therefore
/// keeps passing its liveness probe; only a wedged loop fails it.
#[tokio::test]
async fn a_poll_whose_api_read_fails_still_refreshes_a_stale_heartbeat() {
    let cluster = Cluster::start(&[ready("server-a", idle())], 1, 1).await;
    cluster
        .override_api(
            Mock::given(method("GET"))
                .and(path(SLICES_PATH))
                .respond_with(ResponseTemplate::new(500)),
        )
        .await;
    let heartbeat = Arc::new(Heartbeat::new(Duration::from_secs(60)));
    jump_clock(Duration::from_secs(61)).await;
    assert!(
        !heartbeat.is_fresh(),
        "fixture self-check: stale before the loop runs"
    );

    let mut cfg = config(false);
    // One poll at once, the next an hour away.
    cfg.poll_interval = Duration::from_secs(3_600);
    let (scaler, metrics) = cluster.scaler_with_heartbeat(cfg, Arc::clone(&heartbeat));
    let cancel = CancellationToken::new();
    let run = tokio::spawn(scaler.run(cancel.clone()));
    settle("the failed poll's heartbeat", || async {
        heartbeat.is_fresh()
    })
    .await;
    cancel.cancel();
    tokio::time::timeout(RUN_EXIT, run)
        .await
        .expect("run returns once cancelled")
        .expect("the loop does not panic");

    assert_eq!(
        sample(
            &render(&metrics),
            "live_edit_scaler_poll_errors_total",
            &[("stage", "endpoints")]
        ),
        Some(1.0),
        "the poll that beat was the failed one"
    );
    assert_eq!(verbs_and_paths(&cluster.api_requests().await).len(), 1);
}

/// `/healthz` on the scaler's own port answers 200 while the heartbeat is
/// fresh, 503 once it is stale, and 200 again after a beat; `/metrics`
/// serves the scaler's own families. Served by the production `obs::serve`
/// on a loopback listener.
#[tokio::test]
async fn healthz_follows_the_heartbeat_over_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback port");
    let addr = listener.local_addr().expect("its address");
    let metrics = ScalerMetrics::new();
    let heartbeat = Arc::new(Heartbeat::new(Duration::from_secs(60)));
    let server = tokio::spawn(live_edit_scaler::obs::serve(
        listener,
        metrics.exporter(),
        Arc::clone(&heartbeat),
    ));
    let http = reqwest::Client::new();
    let get = |route: &'static str| {
        let http = http.clone();
        async move {
            let response =
                tokio::time::timeout(SETTLE, http.get(format!("http://{addr}{route}")).send())
                    .await
                    .unwrap_or_else(|_| panic!("GET {route} did not answer within {SETTLE:?}"))
                    .unwrap_or_else(|e| panic!("GET {route} failed: {e}"));
            let status = response.status().as_u16();
            (status, response.text().await.expect("a text body"))
        }
    };

    assert_eq!(get("/healthz").await, (200, "ok".to_owned()), "fresh");
    jump_clock(Duration::from_secs(61)).await;
    assert_eq!(
        get("/healthz").await,
        (503, "poll loop stalled".to_owned()),
        "stale past the bound"
    );
    heartbeat.beat();
    assert_eq!(get("/healthz").await.0, 200, "a beat makes it fresh again");

    let (status, body) = get("/metrics").await;
    assert_eq!(status, 200);
    assert!(
        body.contains("live_edit_scaler_poll_errors_total"),
        "the scaler's own metrics: {body}"
    );
    server.abort();
}

// ── λ̂ gauge ───────────────────────────────────────────────────────────────

/// The scaler's own `/metrics`, served by the production `obs::serve` on a
/// loopback listener for the test's lifetime.
struct MetricsPort {
    addr: std::net::SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl MetricsPort {
    async fn serve(metrics: &ScalerMetrics) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback port");
        let addr = listener.local_addr().expect("its address");
        let task = tokio::spawn(live_edit_scaler::obs::serve(
            listener,
            metrics.exporter(),
            Arc::new(Heartbeat::new(Duration::from_secs(60))),
        ));
        Self { addr, task }
    }

    /// One scrape of `/metrics`, bounded by [`SETTLE`].
    async fn scrape(&self) -> String {
        let response = tokio::time::timeout(
            SETTLE,
            reqwest::get(format!("http://{}/metrics", self.addr)),
        )
        .await
        .expect("GET /metrics answers in time")
        .expect("GET /metrics succeeds");
        assert_eq!(response.status().as_u16(), 200);
        response.text().await.expect("a text body")
    }
}

impl Drop for MetricsPort {
    fn drop(&mut self) {
        self.task.abort();
    }
}

const CATCHUP_RATE: &str = "live_edit_scaler_catchup_rate";

/// `live_edit_scaler_catchup_rate` as scraped: the fleet value (if any) and
/// each pod's value by pod name.
fn catchup_rate_series(text: &str) -> (Option<f64>, HashMap<String, f64>) {
    let fleet = sample(text, CATCHUP_RATE, &[("scope", "fleet")]);
    let pods = crate::common::label_values(text, CATCHUP_RATE, "pod")
        .into_iter()
        .map(|pod| {
            let value = sample(text, CATCHUP_RATE, &[("scope", "pod"), ("pod", &pod)])
                .expect("a pod series has a value");
            (pod, value)
        })
        .collect();
    (fleet, pods)
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{what}: {actual} != {expected}"
    );
}

/// A saturated snapshot, 10 catch-ups pending under no concurrent-catch-up
/// limit (`max_concurrent_catchups` 0, so any pending catch-up saturates),
/// with `completed` catch-ups done.
fn draining(completed: u64) -> ResponseTemplate {
    let mut s = idle_snapshot();
    s.catchups_pending = 10;
    s.catchups_completed_total = completed;
    ResponseTemplate::new(200).set_body_json(s)
}

/// After each decided poll the gauge carries exactly what the decision used:
/// `{scope="fleet"}` the λ̂ it decided with (the prior before any pod has
/// learned one), and one `{scope="pod"}` series per reporting pod with a
/// learned value. The state is replaced whole each poll, so a pod that leaves
/// the EndpointSlices loses its series; and on a poll where no pod reports,
/// both scopes are absent.
#[tokio::test]
async fn catchup_rate_gauge_follows_each_decided_poll_and_drops_what_is_gone() {
    let cluster = Cluster::start(
        &[ready("server-a", idle()), ready("server-b", idle())],
        2,
        2,
    )
    .await;
    let (mut scaler, metrics) = cluster.scaler(config(false));
    let port = MetricsPort::serve(&metrics).await;
    for pod in &cluster.pods {
        answer_with(pod, draining(0)).await;
    }

    let first = scaler.poll_once().await.expect("the API reads succeed");
    let (fleet, pods) = catchup_rate_series(&port.scrape().await);
    assert_close(
        fleet.expect("a fleet series"),
        first.catchup_rate,
        "first poll, fleet",
    );
    assert_close(
        first.catchup_rate,
        live_edit_scaler::decide::CATCHUP_RATE_PRIOR,
        "the prior",
    );
    assert!(pods.is_empty(), "no pod has learned a rate yet: {pods:?}");

    answer_with(&cluster.pods[0], draining(50)).await;
    answer_with(&cluster.pods[1], draining(30)).await;
    let second = scaler.poll_once().await.expect("the API reads succeed");
    let (fleet, pods) = catchup_rate_series(&port.scrape().await);
    assert_eq!(
        second.pod_catchup_rates.len(),
        2,
        "fixture self-check: both pods learned"
    );
    assert_close(
        fleet.expect("a fleet series"),
        second.catchup_rate,
        "second poll, fleet",
    );
    assert_eq!(pods.len(), 2, "{pods:?}");
    for (pod, rate) in &second.pod_catchup_rates {
        assert_close(pods[pod], *rate, pod);
    }

    // server-b leaves the EndpointSlices.
    let only_a = json!({ "items": [{
        "addressType": "IPv4",
        "ports": [{ "name": "metrics", "port": cluster.pods[0].port }],
        "endpoints": [{
            "addresses": ["127.0.0.1"],
            "conditions": { "ready": true, "terminating": false },
            "targetRef": { "kind": "Pod", "name": "server-a", "uid": "uid-server-a" }
        }]
    }] });
    cluster
        .override_api(
            Mock::given(method("GET"))
                .and(path(SLICES_PATH))
                .respond_with(ResponseTemplate::new(200).set_body_json(only_a)),
        )
        .await;
    let third = scaler.poll_once().await.expect("the API reads succeed");
    let (fleet, pods) = catchup_rate_series(&port.scrape().await);
    assert_close(
        fleet.expect("a fleet series"),
        third.catchup_rate,
        "third poll, fleet",
    );
    assert_eq!(
        pods.keys().collect::<Vec<_>>(),
        vec!["server-a"],
        "the pod that left loses its series"
    );

    // No pod reports: nothing was decided with λ̂.
    answer_with(&cluster.pods[0], ResponseTemplate::new(404)).await;
    let fourth = scaler.poll_once().await.expect("the API reads succeed");
    assert_eq!(fourth.action, Action::NoData);
    let text = port.scrape().await;
    assert!(
        !text
            .lines()
            .any(|line| line.starts_with(&format!("{CATCHUP_RATE}{{"))),
        "on a no-data poll both scopes are absent:\n{text}"
    );
}

/// `live_edit_scaler_catchup_rate` (both scopes) and `live_edit_scaler_pods`
/// (by state) as scraped. Parsed, not compared as text: the exporter does
/// not print a series' labels in a fixed order.
#[derive(Debug, PartialEq)]
struct RateAndPods {
    fleet: Option<f64>,
    /// `(pod, λ̂)`, sorted by pod.
    pods: Vec<(String, f64)>,
    /// `(state, count)` for every pod state.
    states: Vec<(&'static str, Option<f64>)>,
}

fn rate_and_pods(text: &str) -> RateAndPods {
    let (fleet, pods) = catchup_rate_series(text);
    let mut pods: Vec<(String, f64)> = pods.into_iter().collect();
    pods.sort_by(|a, b| a.0.cmp(&b.0));
    let states = ["reporting", "not_reporting", "unready", "terminating"]
        .into_iter()
        .map(|state| {
            (
                state,
                sample(text, "live_edit_scaler_pods", &[("state", state)]),
            )
        })
        .collect();
    RateAndPods {
        fleet,
        pods,
        states,
    }
}

/// A poll whose API read fails records no decision, so the λ̂ gauge (both
/// scopes) and the pod-state gauge keep the values of the last decided poll
/// rather than going absent or resetting.
#[tokio::test]
async fn an_api_read_failure_leaves_the_catchup_rate_and_pod_gauges_as_they_were() {
    let cluster = Cluster::start(&[ready("server-a", idle())], 1, 1).await;
    let (mut scaler, metrics) = cluster.scaler(config(false));
    let port = MetricsPort::serve(&metrics).await;
    answer_with(&cluster.pods[0], draining(0)).await;
    scaler.poll_once().await.expect("the API reads succeed");
    answer_with(&cluster.pods[0], draining(40)).await;
    scaler.poll_once().await.expect("the API reads succeed");
    let before = rate_and_pods(&port.scrape().await);
    assert!(
        before.fleet.is_some()
            && !before.pods.is_empty()
            && before.states.iter().all(|(_, count)| count.is_some()),
        "fixture self-check: both scopes and the pod states have series: {before:?}"
    );

    cluster
        .override_api(
            Mock::given(method("GET"))
                .and(path(SCALE_PATH))
                .respond_with(ResponseTemplate::new(500)),
        )
        .await;
    assert!(scaler.poll_once().await.is_none(), "the /scale read fails");
    assert_eq!(
        rate_and_pods(&port.scrape().await),
        before,
        "a failed poll leaves the λ̂ and pod-state gauges as they were"
    );
}

// ── An API read failure restarts the scale-down window ─────────────────────

/// The scale-down window is an unbroken run of observed polls, so a poll
/// whose `/scale` read failed restarts it at the call site
/// (`Scaler::api_read_failed`). With a 600 s window, an idle fleet of three
/// polled at t0 and t0 + 300 s would scale down at t0 + 600 s — the control
/// cluster below does. In the subject, the `/scale` GET fails once at
/// t0 + 600 s: no scale-down PATCH follows at t0 + 600 s or t0 + 900 s, only
/// at t0 + 1 200 s, a full window of observed polls after the failure.
///
/// Polls run in real time; tokio's clock jumps between them ([`jump_clock`])
/// with no request in flight, which moves the engine's clock (it reads
/// tokio's) without letting a request timer fire early.
#[tokio::test]
async fn a_failed_scale_read_restarts_the_scale_down_window() {
    let half_window = Duration::from_secs(300);
    let mut cfg = config(false);
    cfg.scale_down_window = half_window * 2;
    let idle_three = || {
        [
            ready("server-a", idle()),
            ready("server-b", idle()),
            ready("server-c", idle()),
        ]
    };
    let control = Cluster::start(&idle_three(), 3, 3).await;
    let subject = Cluster::start(&idle_three(), 3, 3).await;
    let (mut control_scaler, _) = control.scaler(cfg.clone());
    let (mut subject_scaler, _) = subject.scaler(cfg);
    let scale_patches = |requests: Vec<(String, String, serde_json::Value)>| {
        requests
            .iter()
            .filter(|(verb, path, _)| verb == "PATCH" && path == SCALE_PATH)
            .count()
    };

    // t0 and t0 + 300 s: within the first window, so held `startup`.
    for _ in 0..2 {
        for scaler in [&mut control_scaler, &mut subject_scaler] {
            let d = scaler.poll_once().await.expect("the API reads succeed");
            assert_eq!(d.scale_down_blocked, Some(BlockReason::Startup));
        }
        jump_clock(half_window).await;
    }

    // t0 + 600 s. The control has observed a whole window and scales down.
    let d = control_scaler
        .poll_once()
        .await
        .expect("the API reads succeed");
    assert!(
        matches!(d.action, Action::ScaleDown { to: 2, .. }),
        "{:?}",
        d.action
    );
    assert_eq!(scale_patches(control.api_requests().await), 1, "control");

    // The subject's /scale read fails once, then it polls again at once.
    subject
        .override_api(
            Mock::given(method("GET"))
                .and(path(SCALE_PATH))
                .respond_with(ResponseTemplate::new(500))
                .up_to_n_times(1),
        )
        .await;
    assert!(
        subject_scaler.poll_once().await.is_none(),
        "the /scale read fails"
    );
    for step in ["t0 + 600 s", "t0 + 900 s"] {
        let d = subject_scaler
            .poll_once()
            .await
            .expect("the API reads succeed");
        assert_eq!(
            d.scale_down_blocked,
            Some(BlockReason::Window),
            "{step}: the failed read restarted the window"
        );
        assert_eq!(scale_patches(subject.api_requests().await), 0, "{step}");
        jump_clock(half_window).await;
    }

    // t0 + 1 200 s: a full window of observed polls after the failure.
    let d = subject_scaler
        .poll_once()
        .await
        .expect("the API reads succeed");
    assert!(
        matches!(d.action, Action::ScaleDown { to: 2, .. }),
        "{:?}",
        d.action
    );
    assert_eq!(scale_patches(subject.api_requests().await), 1);
}
