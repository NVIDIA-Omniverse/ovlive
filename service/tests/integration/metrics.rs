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

// Integration tests for the production metrics port:
// `observability::metrics_server::serve_metrics` serving the scrape exporter of
// a real `Metrics::new(..)` on `/metrics`, and
// `serve_metrics_with_scaling` serving the autoscaler's `/scaling/v1`
// snapshot from the production `ServiceScalingSource` over the same `Metrics`.
//
// Every assertion reads what the production handler serves. No test here builds
// its own registry or increments a hand-made counter: a test that did would
// pass whether or not the service wires its instruments to the exporter it
// serves.
//
// No readiness handshake is needed: the listener is bound before the server
// is spawned, so a client connect queues in the socket's backlog until the
// accept loop picks it up.
//
// Not covered here or anywhere: the `/scaling/v1` registration in `main.rs`'s
// `run` (the `ServiceScalingSource` built after the service shutdown token)
// and its field mapping — which config values and gates feed
// `max_subscriptions`, `admission_waiters`, `catchup_read_concurrency`,
// `max_concurrent_catchups` and `cpu_seconds`, and that the source is
// registered at all — nor `run`'s `record_pod_memory_plan` call. No test
// harness runs the binary to serving: the binary tests in
// `tests/misc/startup_log_binary.rs` stop at the gRPC bind, before the
// registration. The tests below build the source by hand, so they pin what a
// registered source serves, not what `main.rs` registers; the pod memory
// plan's test below builds it from the gate `connect` sizes from the plan, as
// `run` does, and records the plan's gauges from the plan, as `run` does.

use std::net::SocketAddr;
use std::sync::Arc;

use live_edit::observability::metrics::test_util::{has_metric, sample_value};
use live_edit::observability::metrics::{
    Metrics, PRESENCE_SWEEP_LEASE_OUTCOMES, PRESENCE_SWEEP_OUTCOME_COMPLETED,
};
use live_edit::observability::metrics_server::serve_metrics;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const SERVICE: &str = "metrics-endpoint-test";

/// Serve `metrics`' own scrape exporter through the production
/// `serve_metrics` on a random port.
async fn spawn_metrics_server(metrics: &Metrics) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");
    let task = tokio::spawn(serve_metrics(listener, metrics.prometheus_exporter()));
    (addr, task)
}

async fn get(addr: SocketAddr, path: &str) -> reqwest::Response {
    reqwest::Client::new()
        .get(format!("http://{addr}{path}"))
        .send()
        .await
        .unwrap_or_else(|e| panic!("GET {path} must not fail at the transport level: {e}"))
}

/// `GET /metrics` answers 200 in the Prometheus text exposition content type,
/// with a UTF-8 body.
#[tokio::test]
async fn serve_metrics_returns_200_with_prometheus_text_content_type() {
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;

    let resp = get(addr, "/metrics").await;
    assert_eq!(resp.status().as_u16(), 200, "GET /metrics must return 200");
    let content_type = resp
        .headers()
        .get("content-type")
        .expect("content-type header must be present")
        .to_str()
        .expect("content-type must be valid UTF-8")
        .to_owned();
    assert_eq!(
        content_type, "text/plain; version=0.0.4",
        "content-type must be the Prometheus text exposition format"
    );
    let bytes = resp.bytes().await.expect("must read body bytes");
    std::str::from_utf8(&bytes).expect("metrics body must be valid UTF-8");

    task.abort();
}

/// The served body is the service's own exposition: the delivery, presence,
/// presence-lease-keeper, command-timeout and presence-sweeper counter
/// families are present at their seeded zero before any traffic, under the
/// service label `Metrics::new` was built with.
#[tokio::test]
async fn serve_metrics_serves_the_seeded_service_families_before_any_traffic() {
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;

    let body = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();

    assert_eq!(
        sample_value(
            &body,
            "live_subscriber_updates_total",
            &[("endpoint", "SubscribeReadLayer"), ("service", SERVICE)],
        ),
        Some(0.0),
        "live_subscriber_updates_total must be served at zero before the first delivery; got:\n{body}"
    );
    for event_type in ["joined", "left", "message"] {
        assert_eq!(
            sample_value(
                &body,
                "live_presence_events_total",
                &[("event_type", event_type), ("service", SERVICE)],
            ),
            Some(0.0),
            "live_presence_events_total{{event_type=\"{event_type}\"}} must be served at zero; \
             got:\n{body}"
        );
    }
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_lease_refresh_failures_total",
            &[("service", SERVICE)],
        ),
        Some(0.0),
        "live_edit_presence_lease_refresh_failures_total must be served at zero before the first \
         keeper failure; got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_rejoins_total",
            &[("service", SERVICE)],
        ),
        Some(0.0),
        "live_edit_presence_rejoins_total must be served at zero before the first self-rejoin; \
         got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_valkey_errors_total",
            &[("pool", "data"), ("kind", "command_timeout")],
        ),
        Some(0.0),
        "the command_timeout cell must be served at zero; got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_stale_removed_total",
            &[("service", SERVICE)],
        ),
        Some(0.0),
        "live_edit_presence_stale_removed_total must be served at zero before the first sweep \
         removal; got:\n{body}"
    );
    for outcome in PRESENCE_SWEEP_LEASE_OUTCOMES {
        assert_eq!(
            sample_value(
                &body,
                "live_edit_presence_sweep_lease_total",
                &[("outcome", outcome), ("service", SERVICE)],
            ),
            Some(0.0),
            "live_edit_presence_sweep_lease_total{{outcome=\"{outcome}\"}} must be served at zero; \
             got:\n{body}"
        );
    }
}

/// The presence sweeper's families that cannot be seeded are absent before
/// the sweeper runs, and served once it records them: the pass-duration and
/// removal-delay histograms on their first observation, and the oldest-stale-
/// age gauge on the replica's first tick that publishes it — a lost leader
/// probe (0) or a completed pass. Until then it is absent, which reads as "no
/// such tick yet", not as 0.
#[tokio::test]
async fn serve_metrics_serves_the_sweeper_families_once_the_sweeper_records_them() {
    const UNSEEDED: [&str; 3] = [
        "live_edit_presence_sweep_duration_seconds",
        "live_edit_presence_stale_removal_delay_seconds",
        "live_edit_presence_sweep_oldest_stale_age_seconds",
    ];
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;

    let before = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    for family in UNSEEDED {
        assert!(
            !has_metric(&before, family),
            "{family} must be absent before the sweeper records it; got:\n{before}"
        );
    }

    metrics.observe_presence_sweep_duration(PRESENCE_SWEEP_OUTCOME_COMPLETED, 0.2);
    metrics.observe_presence_stale_removal_delay(20.0);
    metrics.record_presence_sweep_oldest_stale_age(20.0);
    let body = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();

    assert!(
        !has_metric(&body, "live_edit_presence_peer_left_publish_failures_total"),
        "the path B publish-failure counter must not be registered: the sweep script \
         publishes each left itself; got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_sweep_duration_seconds_count",
            &[("outcome", "completed"), ("service", SERVICE)],
        ),
        Some(1.0),
        "a pass observed after the first scrape must be served on the next; got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_stale_removal_delay_seconds_count",
            &[("service", SERVICE)],
        ),
        Some(1.0),
        "a removal delay observed after the first scrape must be served on the next; \
         got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_presence_sweep_oldest_stale_age_seconds",
            &[("service", SERVICE)],
        ),
        Some(20.0),
        "the oldest stale age recorded after the first scrape must be served on the next; \
         got:\n{body}"
    );
}

/// Each scrape reflects the live instrument state at scrape time: a value
/// recorded after the server started is visible on the next request. This is
/// what separates "the handler serves this `Metrics`' exporter" from "the
/// handler serves some exporter".
#[tokio::test]
async fn serve_metrics_reflects_values_recorded_after_the_server_started() {
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;
    let _ = get(addr, "/metrics").await;

    metrics.add_subscriber_updates(2);
    metrics.observe_join_latency(
        live_edit::observability::metrics::ENDPOINT_JOIN_SESSION,
        std::time::Duration::from_millis(40),
    );

    let body = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();

    assert_eq!(
        sample_value(
            &body,
            "live_subscriber_updates_total",
            &[("endpoint", "SubscribeReadLayer"), ("service", SERVICE)],
        ),
        Some(2.0),
        "a delivery count recorded after the first scrape must be served on the next; got:\n{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_join_latency_seconds_count",
            &[("endpoint", "JoinSession"), ("service", SERVICE)],
        ),
        Some(1.0),
        "a join latency observed after the first scrape must be served on the next; got:\n{body}"
    );
}

/// `serve_metrics` routes `/metrics` and `/scaling/v1` (see
/// [`serve_metrics_scaling_v1_answers_503_with_no_snapshot_source`]); any
/// other path is 404.
#[tokio::test]
async fn serve_metrics_unknown_path_returns_404() {
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;

    let resp = get(addr, "/nonexistent").await;
    assert_eq!(
        resp.status().as_u16(),
        404,
        "GET /nonexistent must return 404"
    );

    task.abort();
}

// ── /scaling/v1 ──────────────────────────────────────────────────────────────

/// The readiness service name the snapshot's `ready` field reads, as
/// `main.rs` registers it.
const READINESS: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// `serve_metrics` (the server without an autoscaler snapshot) still routes
/// `/scaling/v1`, and answers 503 there for its whole life: nothing ever
/// registers a source on its endpoint.
#[tokio::test]
async fn serve_metrics_scaling_v1_answers_503_with_no_snapshot_source() {
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let (addr, task) = spawn_metrics_server(&metrics).await;

    let resp = get(addr, "/scaling/v1").await;
    assert_eq!(resp.status().as_u16(), 503);

    task.abort();
}

/// `/scaling/v1` answers 503 until startup registers the production
/// `ServiceScalingSource`, then 200 with the replica's snapshot as JSON: its
/// counters from `Metrics` (the same calls that move the metrics they
/// mirror), its limits, its readiness from the health reporter, and
/// `shutting_down` once the service shutdown token fires. A second
/// registration is refused and leaves the first in place.
#[tokio::test]
async fn scaling_v1_answers_503_until_its_source_is_registered_then_the_live_snapshot() {
    use live_edit::observability::metrics_server::serve_metrics_with_scaling;
    use live_edit::observability::scaling::{ScalingEndpoint, ServiceScalingSource};
    use live_edit_scaler::snapshot::ScalingSnapshot;
    use std::time::{Duration, Instant};
    use tokio_util::sync::CancellationToken;

    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let endpoint = Arc::new(ScalingEndpoint::new());
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");
    let task = tokio::spawn(serve_metrics_with_scaling(
        listener,
        metrics.prometheus_exporter(),
        Arc::clone(&endpoint),
    ));

    assert_eq!(
        get(addr, "/scaling/v1").await.status().as_u16(),
        503,
        "before registration the route answers 503"
    );

    let (reporter, _) = live_edit::observability::health::create_health_service(READINESS).await;
    let shutdown = CancellationToken::new();
    let source = |admission_waiters: u64| ServiceScalingSource {
        metrics: Arc::clone(&metrics),
        admission_waiters: Box::new(move || admission_waiters),
        max_subscriptions: 4_096,
        health: tonic_health::server::HealthService::from_health_reporter(reporter.clone()),
        readiness_service_name: READINESS.to_owned(),
        shutting_down: {
            let shutdown = shutdown.clone();
            Box::new(move || shutdown.is_cancelled())
        },
        process_started: Instant::now() - Duration::from_secs(90),
        catchup_read_concurrency: 3,
        max_concurrent_catchups: 64,
        cpu_seconds: || Some(12.5),
    };
    assert!(
        endpoint.register(Arc::new(source(4))),
        "the first source registers"
    );
    assert!(
        !endpoint.register(Arc::new(source(999))),
        "a second registration is refused"
    );

    metrics.add_subscriber_admitted(2);
    for _ in 0..3 {
        metrics.begin_pending_catchup();
    }
    metrics.end_pending_catchup(true);

    let snapshot = |label: &'static str| async move {
        let resp = get(addr, "/scaling/v1").await;
        assert_eq!(resp.status().as_u16(), 200, "{label}");
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json"),
            "{label}"
        );
        resp.json::<ScalingSnapshot>()
            .await
            .unwrap_or_else(|e| panic!("{label}: the body must be a snapshot: {e}"))
    };

    let first = snapshot("registered").await;
    assert!(
        first.uptime_secs >= 90,
        "uptime counts from the process start, got {}",
        first.uptime_secs
    );
    assert_eq!(
        first,
        ScalingSnapshot {
            version: 1,
            ready: false,
            shutting_down: false,
            uptime_secs: first.uptime_secs,
            subscriptions_admitted: 2,
            admission_waiters: 4,
            max_subscriptions: 4_096,
            catchups_pending: 2,
            catchups_completed_total: 1,
            cpu_seconds_total: Some(12.5),
            catchup_read_concurrency: 3,
            max_concurrent_catchups: 64,
        }
    );

    reporter
        .set_service_status(READINESS, tonic_health::ServingStatus::Serving)
        .await;
    assert!(
        snapshot("serving").await.ready,
        "readiness follows the reporter"
    );

    shutdown.cancel();
    let last = snapshot("shutting down").await;
    assert!(last.shutting_down, "the shutdown token sets shutting_down");
    assert!(
        last.ready,
        "shutting_down is reported on its own, not as readiness"
    );

    task.abort();
}

// ── The pod memory plan's wiring (b.fj4, W8) ─────────────────────────────────

/// The pod memory limit the wiring test's service is built for: 8 GiB, so
/// every derived budget differs from the 5 GiB reference pod's a `Config`
/// built field by field carries.
const NON_REFERENCE_LIMIT_BYTES: u64 = 8 << 30;

/// A service configured by `Config::from_env` for an 8 GiB pod reports the
/// budgets derived there, not the reference pod's, everywhere an operator or
/// the autoscaler reads them: the `Config` fields its consumers read, the
/// subscription gate and catch-up memory budget `connect` builds, the upload
/// gate and structural cache built from the configuration (as `run` builds
/// them), the four plan gauges and the upload budget gauge on `/metrics`, and
/// `max_subscriptions` in the `/scaling/v1` snapshot. Needs Valkey (the
/// gates are built by a real `connect`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_service_at_a_non_reference_pod_memory_limit_reports_the_budgets_derived_there() {
    use live_edit::config::{Config, PodMemoryLimitSource, PodMemorySource};
    use live_edit::observability::metrics_server::serve_metrics_with_scaling;
    use live_edit::observability::scaling::{ScalingEndpoint, ServiceScalingSource};
    use live_edit_scaler::snapshot::ScalingSnapshot;
    use std::time::Instant;

    const MIB: u64 = 1024 * 1024;
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    let config = temp_env::with_vars(
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_VALKEY_URL", Some(valkey_url.as_str())),
            // Small pools: the plan charges whatever they are, and a test
            // client needs few connections.
            ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("8")),
            ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("2")),
        ],
        || Config::from_env_with(PodMemorySource::Fixed(NON_REFERENCE_LIMIT_BYTES)),
    )
    .expect("an 8 GiB pod's configuration loads");
    let plan = config.pod_memory.clone();
    assert_eq!(plan.limit().bytes, NON_REFERENCE_LIMIT_BYTES);
    assert_eq!(plan.limit().source, PodMemoryLimitSource::CgroupV2);
    // The shares of the limit: 0.8 subscriptions per MiB, 7.5 % and 5 %.
    assert_eq!(
        (
            plan.max_subscriptions(),
            plan.upload_budget_mib(),
            plan.structural_cache_budget_mib()
        ),
        (6_553, 614, 409)
    );
    assert!(
        plan.catchup_budget_mib() > live_edit::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
        "T at 8 GiB ({} MiB) is above the reference pod's",
        plan.catchup_budget_mib()
    );
    assert_eq!(
        (
            config.subscriber_max_subscriptions as u64,
            config.upload_memory_budget_mib,
            config.structural_cache_max_mib,
            config.catchup_build_memory_budget_mib,
        ),
        (
            plan.max_subscriptions(),
            plan.upload_budget_mib(),
            plan.structural_cache_budget_mib(),
            plan.catchup_budget_mib(),
        ),
        "from_env copies the plan's budgets into the fields their consumers read"
    );

    let client = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("connect");
    assert_eq!(
        client.subscriber_admission.capacity(),
        Some(6_553),
        "the subscription gate is the plan's S"
    );
    assert_eq!(
        client.catchup_build_budget.total_bytes(),
        plan.catchup_budget_bytes(),
        "the catch-up memory budget is the plan's T"
    );
    let cache = live_edit::storage::structural_cache::StructuralCache::new(
        config.structural_cache_budget_bytes(),
    );
    assert_eq!(cache.budget_bytes(), 409 * MIB);

    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    let _upload_gate = live_edit::grpc::upload_admission::UploadAdmissionGate::new(
        config.upload_admission_config(),
        Some(Arc::clone(&metrics)),
    );
    metrics.record_pod_memory_plan(&config.pod_memory);

    let endpoint = Arc::new(ScalingEndpoint::new());
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");
    let task = tokio::spawn(serve_metrics_with_scaling(
        listener,
        metrics.prometheus_exporter(),
        Arc::clone(&endpoint),
    ));
    let (reporter, _) = live_edit::observability::health::create_health_service(READINESS).await;
    let admission = Arc::clone(&client.subscriber_admission);
    assert!(endpoint.register(Arc::new(ServiceScalingSource {
        metrics: Arc::clone(&metrics),
        // As `run` maps it: the gate's capacity.
        max_subscriptions: admission.capacity().map_or(0, |c| c as u64),
        admission_waiters: Box::new(move || admission.waiters() as u64),
        health: tonic_health::server::HealthService::from_health_reporter(reporter),
        readiness_service_name: READINESS.to_owned(),
        shutting_down: Box::new(|| false),
        process_started: Instant::now(),
        catchup_read_concurrency: config.catchup_snapshot_read_concurrency,
        max_concurrent_catchups: 8,
        cpu_seconds: || None,
    })));

    let snapshot = get(addr, "/scaling/v1")
        .await
        .json::<ScalingSnapshot>()
        .await
        .expect("a scaling snapshot");
    assert_eq!(snapshot.max_subscriptions, 6_553);

    let body = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();
    let svc = [("service", SERVICE)];
    assert_eq!(
        sample_value(
            &body,
            "live_edit_pod_memory_limit_bytes",
            &[("service", SERVICE), ("source", "cgroup_v2")]
        ),
        Some(NON_REFERENCE_LIMIT_BYTES as f64),
        "{body}"
    );
    assert_eq!(
        sample_value(
            &body,
            "live_edit_pod_memory_limit_bytes",
            &[("service", SERVICE), ("source", "reference")]
        ),
        Some(0.0),
        "{body}"
    );
    // The structural cache's budget is labelled by service, like its weight
    // gauge; the catch-up budget and the subscription ceiling carry no label,
    // like the usage gauges they are read against.
    for (name, expected, labels) in [
        (
            "live_edit_catchup_build_budget_bytes",
            plan.catchup_budget_bytes(),
            &[][..],
        ),
        (
            "live_edit_structural_cache_budget_bytes",
            plan.structural_cache_resident_budget_bytes(),
            &svc[..],
        ),
        ("live_edit_subscriber_max_subscriptions", 6_553, &[][..]),
    ] {
        assert_eq!(
            sample_value(&body, name, labels),
            Some(expected as f64),
            "{name}: {body}"
        );
    }
    // The exact series, label sets included: `sample_value` matches a subset
    // of the labels, so it cannot tell an unlabelled gauge from a labelled one.
    let series = |name: &str| -> Vec<String> {
        let mut lines: Vec<String> = body
            .lines()
            .map(str::trim)
            .filter(|l| {
                l.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(' ') || rest.starts_with('{'))
            })
            .map(str::to_owned)
            .collect();
        lines.sort_unstable();
        lines
    };
    assert_eq!(
        series("live_edit_catchup_build_budget_bytes"),
        vec![format!(
            "live_edit_catchup_build_budget_bytes {}",
            plan.catchup_budget_bytes()
        )]
    );
    assert_eq!(
        series("live_edit_subscriber_max_subscriptions"),
        vec!["live_edit_subscriber_max_subscriptions 6553".to_owned()]
    );
    assert_ne!(
        plan.structural_cache_resident_budget_bytes(),
        409 * MIB,
        "the gauge carries C's resident budget, C × (1 + φ), not C"
    );
    assert_eq!(
        series("live_edit_structural_cache_budget_bytes"),
        vec![format!(
            "live_edit_structural_cache_budget_bytes{{service=\"{SERVICE}\"}} {}",
            plan.structural_cache_resident_budget_bytes()
        )]
    );
    let mut expected_limit: Vec<String> = ["cgroup_v1", "cgroup_v2", "host_memory", "reference"]
        .into_iter()
        .map(|source| {
            let value = if source == "cgroup_v2" {
                NON_REFERENCE_LIMIT_BYTES
            } else {
                0
            };
            format!(
                "live_edit_pod_memory_limit_bytes{{service=\"{SERVICE}\",source=\"{source}\"}} \
                 {value}"
            )
        })
        .collect();
    expected_limit.sort_unstable();
    assert_eq!(series("live_edit_pod_memory_limit_bytes"), expected_limit);
    assert_eq!(
        sample_value(&body, "live_edit_upload_budget_bytes", &[]),
        Some((614 * MIB) as f64),
        "the upload gate's budget gauge: {body}"
    );
    // No backup store is configured: the plan funds no external-value
    // mirror, and its budget gauge reads 0.
    assert_eq!(plan.ext_mirror_bytes(), None);
    assert_eq!(
        series("live_edit_checkpoint_ext_mirror_budget_bytes"),
        vec!["live_edit_checkpoint_ext_mirror_budget_bytes 0".to_owned()]
    );
    client.quit().await;
}

/// W8 with a backup store (b.mhq): a service configured by `Config::from_env`
/// for an 8 GiB pod with checkpointing on and an adapter funds the
/// external-value mirror with the plan's `M_ext`, 204 MiB there (`L / 40`,
/// above its 101 MiB floor), and `/metrics` reports that budget, unlabelled.
/// (That the mirror is built with it, exactly when the plan funds it, is
/// `config`'s
/// `ext_mirror_channel_for_plan_builds_the_mirror_exactly_when_the_plan_funds_it`.)
/// Needs no Valkey.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_service_with_a_backup_store_funds_the_ext_mirror_with_the_budget_derived_at_its_limit() {
    use live_edit::config::{Config, PodMemorySource};

    const MIB: u64 = 1024 * 1024;
    let config = temp_env::with_vars(
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("8")),
            ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("2")),
            ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("true")),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("InMemory")),
        ],
        || Config::from_env_with(PodMemorySource::Fixed(NON_REFERENCE_LIMIT_BYTES)),
    )
    .expect("an 8 GiB pod's configuration with a backup store loads");
    let budget = config
        .pod_memory
        .ext_mirror_bytes()
        .expect("the plan funds the mirror with a backup store");
    assert_eq!(budget, 204 * MIB, "M_ext at 8 GiB");

    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
    metrics.record_pod_memory_plan(&config.pod_memory);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");
    let task = tokio::spawn(live_edit::observability::metrics_server::serve_metrics(
        listener,
        metrics.prometheus_exporter(),
    ));
    let body = get(addr, "/metrics")
        .await
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();
    let series: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("live_edit_checkpoint_ext_mirror_budget_bytes"))
        .collect();
    assert_eq!(
        series,
        vec![format!(
            "live_edit_checkpoint_ext_mirror_budget_bytes {}",
            204 * MIB
        )],
        "{body}"
    );
}
