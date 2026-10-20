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

//! The scaler's own observability: `live_edit_scaler_*` metrics on `/metrics`,
//! and `/healthz` for the liveness probe.
//!
//! | Metric | Labels | Meaning |
//! |---|---|---|
//! | `live_edit_scaler_desired_replicas` | `term` (`cpu`, `catchup`, `occupancy`, `final`) | each term's desired count, and the bounded result, at the last decided poll; a term that is off or has no reading has no series |
//! | `live_edit_scaler_target_replicas` | — | the Deployment's `spec.replicas` after the last poll: what the scaler set, or left |
//! | `live_edit_scaler_decisions_total` | `direction` (`up`, `down`, `none`), `reason` | decisions by direction and cause |
//! | `live_edit_scaler_poll_errors_total` | `stage` | failed API requests and unreachable pods, by poll stage |
//! | `live_edit_scaler_pods` | `state` (`reporting`, `not_reporting`, `unready`, `terminating`) | server pods by state at the last poll that read the cluster state; a poll whose API read failed leaves them as they were |
//! | `live_edit_scaler_scale_down_blocked_total` | `reason` | polls on which a wanted scale-down was held back, by reason |
//! | `live_edit_scaler_catchup_rate` | `scope` (`fleet`; or `pod` with `pod`) | λ̂ at the last poll that read the cluster state: the fleet value the decision used (absent while no pod reports), and each reporting pod's own learned value; a pod's series goes when the pod leaves; a poll whose API read failed leaves both scopes as they were |
//!
//! `/healthz` reports the poll loop's heartbeat, not the API server's health:
//! the loop beats after every poll whatever its outcome, so an unreachable API
//! server (which the scaler rides out by doing nothing) does not get the pod
//! restarted, while a wedged loop does.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use opentelemetry::metrics::{Counter, MeterProvider};
use opentelemetry::KeyValue;
use opentelemetry_prometheus_text_exporter::PrometheusExporter;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tokio::net::TcpListener;

use crate::decide::{BlockReason, Decision, PodCounts, UpReason};

/// `direction` on `live_edit_scaler_decisions_total`.
pub const DIRECTION_UP: &str = "up";
pub const DIRECTION_DOWN: &str = "down";
pub const DIRECTION_NONE: &str = "none";

/// `reason` on a `down` decision.
pub const REASON_BELOW_TARGET: &str = "below_target";
/// `reason` on a `none` decision: the desired count equals the current one.
pub const REASON_AT_TARGET: &str = "at_target";
/// `reason` on a `none` decision: a scale-down was wanted but held back.
pub const REASON_SCALE_DOWN_BLOCKED: &str = "scale_down_blocked";
/// `reason` on a `none` decision: no pod reported, so nothing was decided.
pub const REASON_NO_DATA: &str = "no_data";

/// `stage` on `live_edit_scaler_poll_errors_total`.
pub const STAGE_ENDPOINTS: &str = "endpoints";
pub const STAGE_SCALE: &str = "scale";
pub const STAGE_SNAPSHOT: &str = "snapshot";
pub const STAGE_DELETION_COST: &str = "deletion_cost";
pub const STAGE_PATCH_SCALE: &str = "patch_scale";
pub const POLL_ERROR_STAGES: [&str; 5] = [
    STAGE_ENDPOINTS,
    STAGE_SCALE,
    STAGE_SNAPSHOT,
    STAGE_DELETION_COST,
    STAGE_PATCH_SCALE,
];

/// The last values the observable gauges report.
#[derive(Debug, Default)]
struct GaugeState {
    desired: Vec<(&'static str, u64)>,
    target: Option<u64>,
    pods: Option<PodCounts>,
    /// λ̂ the last decision used (`None` when that poll observed no pod), and
    /// each fleet pod's own (by pod name).
    catchup_rate: Option<f64>,
    pod_catchup_rates: Vec<(String, f64)>,
}

/// The scaler's metric instruments and the exporter `/metrics` renders.
pub struct ScalerMetrics {
    /// Owns the readers; dropped with the metrics.
    _provider: SdkMeterProvider,
    exporter: PrometheusExporter,
    gauges: Arc<Mutex<GaugeState>>,
    decisions: Counter<u64>,
    poll_errors: Counter<u64>,
    blocked: Counter<u64>,
}

impl std::fmt::Debug for ScalerMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalerMetrics").finish_non_exhaustive()
    }
}

impl Default for ScalerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalerMetrics {
    /// Build the instruments on a scrape-only provider, with the counter label
    /// vocabularies pre-registered at zero so an alert on a rate has a series
    /// before the first event.
    pub fn new() -> Self {
        // The same name-mangling settings as the service's exporter: names
        // carry their own `_total`, and no scope or target-info series.
        let exporter = PrometheusExporter::builder()
            .without_counter_suffixes()
            .without_units()
            .without_scope_info()
            .without_target_info()
            .build();
        let provider = SdkMeterProvider::builder()
            .with_reader(exporter.clone())
            .build();
        let meter = provider.meter("live-edit-scaler");
        let gauges = Arc::new(Mutex::new(GaugeState::default()));

        let state = Arc::clone(&gauges);
        let _desired = meter
            .u64_observable_gauge("live_edit_scaler_desired_replicas")
            .with_description(
                "Desired replicas at the scaler's last decided poll, by term: cpu (the CPU \
                 utilization target), catchup (replicas to drain the pending first catch-ups \
                 within the target time), occupancy (replicas to keep subscriptions under the \
                 headroom share of the ceiling), and final (the largest, bounded by the minimum \
                 and maximum replicas). A term that is off or has no reading has no series.",
            )
            .with_callback(move |observer| {
                if let Ok(s) = state.lock() {
                    for (term, value) in &s.desired {
                        observer.observe(*value, &[KeyValue::new("term", *term)]);
                    }
                }
            })
            .build();
        let state = Arc::clone(&gauges);
        let _target = meter
            .u64_observable_gauge("live_edit_scaler_target_replicas")
            .with_description(
                "The service Deployment's spec.replicas after the scaler's last poll: the count \
                 it set, or the count it left in place.",
            )
            .with_callback(move |observer| {
                if let Some(target) = state.lock().ok().and_then(|s| s.target) {
                    observer.observe(target, &[]);
                }
            })
            .build();
        let state = Arc::clone(&gauges);
        let _pods = meter
            .u64_observable_gauge("live_edit_scaler_pods")
            .with_description(
                "Server pods at the scaler's last poll that read the cluster state (a poll whose \
                 API read failed leaves every state as it was), by state: reporting (Ready and \
                 answered /scaling/v1, counted as load), not_reporting (Ready but no snapshot: an \
                 older release, still starting, or unreachable), unready, terminating.",
            )
            .with_callback(move |observer| {
                if let Some(pods) = state.lock().ok().and_then(|s| s.pods) {
                    for (state, n) in [
                        ("reporting", pods.reporting),
                        ("not_reporting", pods.not_reporting),
                        ("unready", pods.unready),
                        ("terminating", pods.terminating),
                    ] {
                        observer.observe(n as u64, &[KeyValue::new("state", state)]);
                    }
                }
            })
            .build();
        let state = Arc::clone(&gauges);
        let _catchup_rate = meter
            .f64_observable_gauge("live_edit_scaler_catchup_rate")
            .with_description(
                "Catch-up drain rate λ̂ at the scaler's last poll that read the cluster state (a \
                 poll whose API read failed leaves both scopes as they were): catch-ups completed \
                 per second per replica, learned only while a replica had at least as many first \
                 catch-ups pending as its concurrent-catch-up limit. scope=\"fleet\" is the value \
                 the decision used (the mean over reporting pods that have a learned value, the \
                 prior while none has one, never below the floor), absent while no pod reports; \
                 scope=\"pod\" is each reporting pod's own learned value, by pod. It drives the \
                 catch-up term (replicas to drain the pending catch-ups within the target time) \
                 and the scale-down feasibility threshold (pending catch-ups one replica drains \
                 within the target time).",
            )
            .with_callback(move |observer| {
                if let Ok(s) = state.lock() {
                    if let Some(rate) = s.catchup_rate {
                        observer.observe(rate, &[KeyValue::new("scope", "fleet")]);
                    }
                    for (pod, rate) in &s.pod_catchup_rates {
                        observer.observe(
                            *rate,
                            &[
                                KeyValue::new("scope", "pod"),
                                KeyValue::new("pod", pod.clone()),
                            ],
                        );
                    }
                }
            })
            .build();

        let decisions = meter
            .u64_counter("live_edit_scaler_decisions_total")
            .with_description(
                "Scaler decisions by direction (up|down|none) and reason: up by the term that set \
                 the target (min|cpu|catchup|occupancy); down below_target; none at_target, \
                 scale_down_blocked or no_data (no pod reported, nothing decided).",
            )
            .build();
        let poll_errors = meter
            .u64_counter("live_edit_scaler_poll_errors_total")
            .with_description(
                "Scaler poll failures by stage: endpoints, scale (an API read failed, so the poll \
                 took no action), snapshot (one pod did not answer /scaling/v1), \
                 deletion_cost (a pod annotation failed, so the scale-down was not made), \
                 patch_scale (the replica change was refused).",
            )
            .build();
        let blocked = meter
            .u64_counter("live_edit_scaler_scale_down_blocked_total")
            .with_description(
                "Polls on which the desired count was below the current count but the scaler did \
                 not remove a pod, by reason: startup (less than one window since the scaler \
                 started), window (a desired count in the window was not lower, or the scaler has \
                 not observed a full window since a poll that observed nothing), unsettled (a \
                 server pod was unready, terminating or not reporting, or the Deployment's pod \
                 count differed from its replica count, as during a rollout), interval (the last \
                 removal was too recent), feasibility (catch-ups were already queued: more \
                 pending than one replica drains within the catch-up target time).",
            )
            .build();

        for reason in [
            UpReason::Min,
            UpReason::Cpu,
            UpReason::Catchup,
            UpReason::Occupancy,
        ] {
            decisions.add(0, &decision_labels(DIRECTION_UP, reason.label()));
        }
        decisions.add(0, &decision_labels(DIRECTION_DOWN, REASON_BELOW_TARGET));
        for reason in [REASON_AT_TARGET, REASON_SCALE_DOWN_BLOCKED, REASON_NO_DATA] {
            decisions.add(0, &decision_labels(DIRECTION_NONE, reason));
        }
        for stage in POLL_ERROR_STAGES {
            poll_errors.add(0, &[KeyValue::new("stage", stage)]);
        }
        for reason in BlockReason::ALL {
            blocked.add(0, &[KeyValue::new("reason", reason.label())]);
        }

        Self {
            _provider: provider,
            exporter,
            gauges,
            decisions,
            poll_errors,
            blocked,
        }
    }

    /// A handle on the exporter `/metrics` renders from.
    pub fn exporter(&self) -> PrometheusExporter {
        self.exporter.clone()
    }

    /// Count one failure at `stage` (a [`POLL_ERROR_STAGES`] value).
    pub fn poll_error(&self, stage: &'static str) {
        self.poll_errors.add(1, &[KeyValue::new("stage", stage)]);
    }

    /// Record one poll's decision: its counters, terms, pod states and λ̂.
    pub fn record_decision(
        &self,
        decision: &Decision,
        direction: &'static str,
        reason: &'static str,
    ) {
        self.decisions.add(1, &decision_labels(direction, reason));
        if let Some(blocked) = decision.scale_down_blocked {
            self.blocked
                .add(1, &[KeyValue::new("reason", blocked.label())]);
        }
        let no_data = direction == DIRECTION_NONE && reason == REASON_NO_DATA;
        if let Ok(mut s) = self.gauges.lock() {
            s.pods = Some(decision.pods);
            // Replaced whole each poll, so a pod that left loses its series.
            // With no pod observed nothing used λ̂ (the decision carries only
            // the prior), so the fleet series goes too, like the pods'.
            s.catchup_rate = (!no_data).then_some(decision.catchup_rate);
            s.pod_catchup_rates = decision.pod_catchup_rates.clone();
            if no_data {
                return;
            }
            let terms = decision.terms;
            s.desired = [
                ("cpu", terms.cpu),
                ("catchup", Some(terms.catchup)),
                ("occupancy", terms.occupancy),
                ("final", Some(decision.desired)),
            ]
            .into_iter()
            .filter_map(|(term, v)| v.map(|v| (term, u64::from(v))))
            .collect();
        }
    }

    /// The Deployment's `spec.replicas` after this poll.
    pub fn set_target(&self, replicas: u32) {
        if let Ok(mut s) = self.gauges.lock() {
            s.target = Some(u64::from(replicas));
        }
    }
}

fn decision_labels(direction: &'static str, reason: &'static str) -> [KeyValue; 2] {
    [
        KeyValue::new("direction", direction),
        KeyValue::new("reason", reason),
    ]
}

/// The poll loop's liveness signal: when it last finished a poll.
#[derive(Debug)]
pub struct Heartbeat {
    origin: tokio::time::Instant,
    last_ms: AtomicU64,
    stale_after: Duration,
}

impl Heartbeat {
    /// Fresh now; stale once `stale_after` passes without a [`Self::beat`].
    pub fn new(stale_after: Duration) -> Self {
        Self {
            origin: tokio::time::Instant::now(),
            last_ms: AtomicU64::new(0),
            stale_after,
        }
    }

    /// The staleness bound for a loop polling every `poll_interval`: six polls,
    /// and never under a minute. One poll makes at most a few API requests of
    /// at most a few seconds each, so a loop that has not finished one in that
    /// long is wedged, not slow.
    pub fn stale_after_for(poll_interval: Duration) -> Duration {
        (poll_interval * 6).max(Duration::from_secs(60))
    }

    pub fn beat(&self) {
        self.last_ms.store(self.elapsed_ms(), Ordering::Release);
    }

    pub fn is_fresh(&self) -> bool {
        let since = self
            .elapsed_ms()
            .saturating_sub(self.last_ms.load(Ordering::Acquire));
        Duration::from_millis(since) <= self.stale_after
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

#[derive(Clone)]
struct HttpState {
    exporter: PrometheusExporter,
    heartbeat: Arc<Heartbeat>,
}

/// Serve `GET /metrics` and `GET /healthz` on `listener` until the task is
/// aborted or the listener fails.
pub async fn serve(listener: TcpListener, exporter: PrometheusExporter, heartbeat: Arc<Heartbeat>) {
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/healthz", get(healthz_handler))
        .with_state(HttpState {
            exporter,
            heartbeat,
        });
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!(error = %e, "scaler HTTP server error");
    }
}

async fn metrics_handler(State(state): State<HttpState>) -> impl IntoResponse {
    let mut buf = Vec::new();
    if let Err(e) = state.exporter.export(&mut buf) {
        tracing::error!(error = %e, "failed to encode scaler metrics");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("content-type", "text/plain")],
            b"failed to encode metrics".to_vec(),
        );
    }
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buf,
    )
}

async fn healthz_handler(State(state): State<HttpState>) -> impl IntoResponse {
    if state.heartbeat.is_fresh() {
        (StatusCode::OK, "ok")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "poll loop stalled")
    }
}
