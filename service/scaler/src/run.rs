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

//! The poll loop, and `live-edit scaler`'s entry point.
//!
//! A poll reads the API server and every pod, asks the engine for a decision
//! and carries it out. Any failed API read ends the poll with no action, so
//! the replica count stays where it is (fail-static). A failed write leaves
//! the engine's state as if the write had not been attempted, so a later poll
//! retries it. Cancellation (SIGTERM) ends the loop between polls or in the
//! middle of one; it never issues a scale call of its own.

use std::sync::Arc;
use std::time::Duration;

use reqwest::StatusCode;
use tokio::net::TcpListener;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::config::ScalerConfig;
use crate::decide::{
    Action, Decision, Engine, EngineConfig, Observation, PodObservation, PodReport,
    FLEET_READ_BUDGET,
};
use crate::kube::{Endpoint, KubeClient};
use crate::obs::{
    self, Heartbeat, ScalerMetrics, DIRECTION_DOWN, DIRECTION_NONE, DIRECTION_UP, REASON_AT_TARGET,
    REASON_BELOW_TARGET, REASON_NO_DATA, REASON_SCALE_DOWN_BLOCKED, STAGE_DELETION_COST,
    STAGE_ENDPOINTS, STAGE_PATCH_SCALE, STAGE_SCALE, STAGE_SNAPSHOT,
};
use crate::snapshot::{ScalingSnapshot, SCALING_SNAPSHOT_PATH};

/// Bound on one pod's `/scaling/v1` request. The pod answers from in-process
/// counters, so anything slower is a pod in trouble, which the poll treats as
/// not reporting rather than waiting on.
pub const SNAPSHOT_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Largest snapshot body read. A real one is a few hundred bytes; the cap only
/// stops a misbehaving endpoint from feeding the scaler an unbounded body.
const SNAPSHOT_MAX_BYTES: usize = 64 * 1024;

/// The scaler: its clients, its engine and its instruments.
pub struct Scaler {
    cfg: ScalerConfig,
    kube: KubeClient,
    pods_http: reqwest::Client,
    engine: Engine,
    metrics: Arc<ScalerMetrics>,
    heartbeat: Arc<Heartbeat>,
    /// The no-snapshot WARN was logged and no poll has reached a Ready pod's
    /// snapshot since; see [`Self::warn_if_no_ready_pod_reports`].
    no_snapshot_warned: bool,
}

impl Scaler {
    /// A scaler whose engine starts now: it will not scale down for one
    /// window.
    pub fn new(
        cfg: ScalerConfig,
        kube: KubeClient,
        metrics: Arc<ScalerMetrics>,
        heartbeat: Arc<Heartbeat>,
    ) -> Result<Self, reqwest::Error> {
        let pods_http = reqwest::Client::builder()
            .timeout(SNAPSHOT_REQUEST_TIMEOUT)
            .build()?;
        let engine = Engine::new(EngineConfig::from(&cfg), now());
        Ok(Self {
            cfg,
            kube,
            pods_http,
            engine,
            metrics,
            heartbeat,
            no_snapshot_warned: false,
        })
    }

    /// Poll every `poll_interval` until `cancel` fires.
    pub async fn run(mut self, cancel: CancellationToken) {
        let mut ticker = tokio::time::interval(self.cfg.poll_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => break,
                _ = ticker.tick() => {}
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => break,
                _ = self.poll_once() => {}
            }
            self.heartbeat.beat();
        }
        tracing::info!("live-edit scaler stopping; the replica count stays where it is");
    }

    /// One poll: one EndpointSlice list and one `/scale` GET against the API,
    /// then each pod's snapshot. Returns the decision, or `None` when an API
    /// read failed and the poll took no action.
    pub async fn poll_once(&mut self) -> Option<Decision> {
        let endpoints = match self
            .kube
            .list_endpoints(&self.cfg.namespace, &self.cfg.service)
            .await
        {
            Ok(e) => e,
            Err(e) => return self.api_read_failed(STAGE_ENDPOINTS, &e),
        };
        let scale = match self
            .kube
            .get_scale(&self.cfg.namespace, &self.cfg.deployment)
            .await
        {
            Ok(s) => s,
            Err(e) => return self.api_read_failed(STAGE_SCALE, &e),
        };
        let replicas = scale.replicas;
        self.metrics.set_target(replicas);

        let (terminating, live): (Vec<Endpoint>, Vec<Endpoint>) =
            endpoints.into_iter().partition(|e| e.terminating);
        let reports =
            futures::future::join_all(live.iter().map(|e| fetch_snapshot(&self.pods_http, e)))
                .await;
        let pods: Vec<PodObservation> = live
            .into_iter()
            .zip(reports)
            .map(|(e, report)| {
                if report == PodReport::Unreachable {
                    self.metrics.poll_error(STAGE_SNAPSHOT);
                }
                PodObservation {
                    uid: e.uid,
                    name: e.name,
                    ready: e.ready,
                    report,
                }
            })
            .collect();
        self.warn_if_no_ready_pod_reports(&pods);

        let decision = self.engine.decide(&Observation {
            now: now(),
            replicas,
            status_replicas: scale.status_replicas,
            pods,
            terminating: terminating.len(),
        });
        if let Some(b) = decision.read_budget_exceeded {
            tracing::warn!(
                max_replicas = b.max_replicas,
                catchup_snapshot_read_concurrency = b.read_concurrency,
                product = u64::from(b.max_replicas) * u64::from(b.read_concurrency),
                budget = FLEET_READ_BUDGET,
                "maxReplicas × LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY exceeds the fleet's \
                 budget of concurrent catch-up snapshot reads: a join wave on a fully scaled-out \
                 fleet can block the backing store for longer than writers tolerate. Keep the \
                 product at or below the budget (lower autoscaling.maxReplicas or the read \
                 concurrency) unless you have re-measured the per-read cost on your store"
            );
        }
        self.apply(&decision, replicas).await;
        Some(decision)
    }

    /// A failed API read: count it, log it, and break the engine's window, so
    /// a scale-down needs a full window of observed polls after the gap.
    fn api_read_failed(
        &mut self,
        stage: &'static str,
        error: &crate::kube::KubeError,
    ) -> Option<Decision> {
        self.engine.observation_gap();
        self.metrics.poll_error(stage);
        tracing::warn!(
            stage,
            error = %error,
            "scaler poll could not read the cluster state; no action this poll"
        );
        None
    }

    /// WARN when Ready server pods exist but none of them answered
    /// `/scaling/v1` with a snapshot: the scaler then decides nothing, and
    /// holds every scale-down, for as long as it lasts. Logged on the first
    /// such poll, and again only after a poll that did reach a Ready pod's
    /// snapshot, so a lasting condition logs once rather than every poll.
    fn warn_if_no_ready_pod_reports(&mut self, pods: &[PodObservation]) {
        let ready: Vec<&PodReport> = pods.iter().filter(|p| p.ready).map(|p| &p.report).collect();
        if ready.iter().any(|r| matches!(r, PodReport::Snapshot(_))) {
            self.no_snapshot_warned = false;
            return;
        }
        if ready.is_empty() || self.no_snapshot_warned {
            return;
        }
        self.no_snapshot_warned = true;
        let count = |kind: &PodReport| ready.iter().filter(|r| **r == kind).count();
        tracing::warn!(
            ready_pods = ready.len(),
            not_found = count(&PodReport::NotServed),
            not_ready_to_report = count(&PodReport::NotRegistered),
            unreachable = count(&PodReport::Unreachable),
            "no Ready server pod answered /scaling/v1, so the scaler is deciding nothing and \
             holding every scale-down. Likely causes: the pods' metrics port is unreachable \
             from the scaler (a NetworkPolicy, for example); the pods run an older release \
             without /scaling/v1 (404); or the pods are stuck starting or unready (503)"
        );
    }

    /// Carry out the decision and record it.
    async fn apply(&mut self, decision: &Decision, replicas: u32) {
        let (direction, reason) = match &decision.action {
            Action::NoData => (DIRECTION_NONE, REASON_NO_DATA),
            Action::Hold if decision.scale_down_blocked.is_some() => {
                (DIRECTION_NONE, REASON_SCALE_DOWN_BLOCKED)
            }
            Action::Hold => (DIRECTION_NONE, REASON_AT_TARGET),
            Action::ScaleUp { reason, .. } => (DIRECTION_UP, reason.label()),
            Action::ScaleDown { .. } => (DIRECTION_DOWN, REASON_BELOW_TARGET),
        };
        self.metrics.record_decision(decision, direction, reason);
        match &decision.action {
            Action::ScaleUp { to, reason } => {
                tracing::info!(
                    from = replicas,
                    to,
                    reason = reason.label(),
                    cpu = ?decision.terms.cpu,
                    catchup = decision.terms.catchup,
                    occupancy = ?decision.terms.occupancy,
                    catchup_rate = decision.catchup_rate,
                    reporting_pods = decision.pods.reporting,
                    "scaling the service up"
                );
                self.set_replicas(*to).await;
            }
            Action::ScaleDown { to, deletion_costs } => {
                if !self.mark_deletion_costs(deletion_costs).await {
                    return;
                }
                tracing::info!(
                    from = replicas,
                    to,
                    desired = decision.desired,
                    catchup_rate = decision.catchup_rate,
                    reporting_pods = decision.pods.reporting,
                    "scaling the service down by one pod"
                );
                if self.set_replicas(*to).await {
                    self.engine.applied(&decision.action, now());
                }
            }
            Action::Hold | Action::NoData => {
                if let Some(blocked) = decision.scale_down_blocked {
                    tracing::debug!(
                        replicas,
                        desired = decision.desired,
                        reason = blocked.label(),
                        "scale-down held back"
                    );
                }
            }
        }
    }

    /// Write every pod's deletion cost, concurrently. `false` if any write
    /// failed: the scale-down is then not made, because without the marks the
    /// ReplicaSet would not necessarily remove the emptiest pod, and the
    /// scale-down could end more streams than it needs to.
    async fn mark_deletion_costs(&self, costs: &[(String, i32)]) -> bool {
        let ns = &self.cfg.namespace;
        let results = futures::future::join_all(
            costs
                .iter()
                .map(|(pod, cost)| self.kube.patch_pod_deletion_cost(ns, pod, *cost)),
        )
        .await;
        let mut ok = true;
        for ((pod, _), result) in costs.iter().zip(results) {
            if let Err(e) = result {
                ok = false;
                self.metrics.poll_error(STAGE_DELETION_COST);
                tracing::warn!(
                    pod = %pod,
                    error = %e,
                    "could not set the pod's deletion cost; not scaling down this poll"
                );
            }
        }
        ok
    }

    /// PATCH `/scale` to `replicas`. `true` on success.
    async fn set_replicas(&self, replicas: u32) -> bool {
        match self
            .kube
            .patch_replicas(&self.cfg.namespace, &self.cfg.deployment, replicas)
            .await
        {
            Ok(()) => {
                self.metrics.set_target(replicas);
                true
            }
            Err(e) => {
                self.metrics.poll_error(STAGE_PATCH_SCALE);
                tracing::warn!(
                    replicas,
                    error = %e,
                    "could not set the service Deployment's replica count"
                );
                false
            }
        }
    }
}

/// One pod's `/scaling/v1`, classified.
async fn fetch_snapshot(http: &reqwest::Client, endpoint: &Endpoint) -> PodReport {
    let Some(url) = endpoint.url(SCALING_SNAPSHOT_PATH) else {
        return PodReport::Unreachable;
    };
    let mut response = match http.get(url).send().await {
        Ok(r) => r,
        Err(_) => return PodReport::Unreachable,
    };
    match response.status() {
        StatusCode::OK => {}
        StatusCode::NOT_FOUND => return PodReport::NotServed,
        StatusCode::SERVICE_UNAVAILABLE => return PodReport::NotRegistered,
        _ => return PodReport::Unreachable,
    }
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) if body.len() + chunk.len() <= SNAPSHOT_MAX_BYTES => {
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Ok(Some(_)) | Err(_) => return PodReport::Unreachable,
        }
    }
    match serde_json::from_slice::<ScalingSnapshot>(&body) {
        Ok(snapshot) => PodReport::Snapshot(snapshot),
        Err(_) => PodReport::Unreachable,
    }
}

/// The engine's clock: tokio's, so a paused test runtime drives it.
fn now() -> std::time::Instant {
    tokio::time::Instant::now().into_std()
}

/// `live-edit scaler`: read the configuration, serve `/metrics` and
/// `/healthz`, and poll until SIGTERM or SIGINT. Returns the process exit
/// code: 1 when the configuration is refused or the scaler cannot start, 0
/// after a signal.
///
/// The caller has initialized tracing. A configuration error goes to stderr,
/// as the service's own does, so no log filter can hide it.
pub async fn main_from_env() -> i32 {
    let cfg = match ScalerConfig::from_env() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("configuration error: {e}");
            return 1;
        }
    };
    cfg.log_startup();

    let kube = match KubeClient::in_cluster() {
        Ok(kube) => kube,
        Err(e) => {
            tracing::error!(error = %e, "live-edit scaler cannot reach the Kubernetes API");
            return 1;
        }
    };
    let listener = match TcpListener::bind(("0.0.0.0", cfg.http_port)).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, port = cfg.http_port, "failed to bind the scaler HTTP port");
            return 1;
        }
    };
    let metrics = Arc::new(ScalerMetrics::new());
    let heartbeat = Arc::new(Heartbeat::new(Heartbeat::stale_after_for(
        cfg.poll_interval,
    )));
    let http = tokio::spawn(obs::serve(
        listener,
        metrics.exporter(),
        Arc::clone(&heartbeat),
    ));

    let cancel = CancellationToken::new();
    let signals = match spawn_signal_handler(cancel.clone()) {
        Ok(handle) => handle,
        Err(e) => {
            tracing::error!(error = %e, "failed to register the scaler's signal handlers");
            return 1;
        }
    };
    let scaler = match Scaler::new(cfg, kube, metrics, heartbeat) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "failed to build the scaler's pod HTTP client");
            return 1;
        }
    };
    scaler.run(cancel).await;
    signals.abort();
    http.abort();
    0
}

/// Cancel `cancel` on the first SIGTERM or SIGINT.
fn spawn_signal_handler(cancel: CancellationToken) -> std::io::Result<tokio::task::JoinHandle<()>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    Ok(tokio::spawn(async move {
        tokio::select! {
            _ = term.recv() => tracing::info!("received SIGTERM"),
            _ = int.recv() => tracing::info!("received SIGINT"),
        }
        cancel.cancel();
    }))
}
