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

//! Held-mode startup (SDD SR-5): what a process runs when its startup hold
//! evaluation finds the fleet held.
//!
//! `main.rs` evaluates the hold right after the Valkey connect and the
//! durability guard, before it builds any store, dispatcher or store- or
//! backup-writing task (SDD SR-5.1a). On `Held` it hands the gRPC socket over
//! from the bootstrap server and calls [`run_held_mode`] instead of carrying
//! on with the normal startup, and the process ends when it returns. Nothing
//! the normal startup builds after that point exists in held mode: no
//! `LiveEditService` or any of its parts (presence router, trackers, the
//! session, layer and external-value stores), no counter seed, and none of the
//! writer inventory
//! (`crate::dr::hold::HOLD_WRITERS`) — so neither the external-value backup
//! repair (`spawn_ext_backup_repair`) nor the checkpoint clean-up
//! (`spawn_checkpoint_cleanup`) is ever spawned in held mode. That holds by construction:
//! held mode spawns exactly the tasks listed on [`run_held_mode`]. Held mode
//! registers nothing on the client's stream multiplexer, whose readers the
//! Valkey connect starts idle and unconnected.
//!
//! Held mode writes nothing to the store or the backup store. Its only store
//! commands are the hold poll's read-only `EVALSHA` and the health
//! supervisor's `PING`, and it never calls the backup store. (When automatic
//! loss detection lands, the poll script gains the one write a held process
//! may make, a fresh marker; SDD SR-4.3.)
//!
//! It serves, behind the same `MetricsLayer` as every server:
//!
//! - the **refuse-all LiveService stub** ([`held_live_service`]): every
//!   LiveService RPC is refused `UNAVAILABLE` `SERVICE_HELD`, naming the
//!   hold poll's latest incident, and logged once at INFO — the hold gate's
//!   refusal, built and logged exactly as on a serving replica that entered
//!   a hold;
//! - health, reflection and the `CapabilitiesService`, unchanged.
//!
//! Readiness reaches `SERVING` (SDD SR-5.4, P1), so the pod passes the
//! readiness-targeted startup probe and is not restarted, while the fleet is
//! held: the counter-seed gate is absent, so its contribution is true, and
//! the readiness inputs are the Valkey probe, the service-identity token,
//! the held server being up (`api_serving`), and a verdict that is not stale.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::Ready;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::CapabilitiesServiceServer;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tonic::server::NamedService;
use tonic::transport::server::TcpIncoming;
use tonic_health::server::HealthReporter;
use tonic_health::ServingStatus;
use tower::Service;

use super::gate::{HoldGate, HoldGateService};
use super::reaction::{spawn_held_mode_reactor, HoldReactorSettings};
use crate::auth::service_token::ServiceTokenProvider;
use crate::dr::hold::{spawn_hold_poll, HoldObservation, HoldPollSettings};
use crate::error::RefusalDetail;
use crate::grpc::capabilities::CapabilitiesServiceImpl;
use crate::grpc::metrics_layer::MetricsLayer;
use crate::grpc::proto::live_service_server::SERVICE_NAME;
use crate::grpc::proto::FILE_DESCRIPTOR_SET;
use crate::grpc::transport::production_server_builder;
use crate::observability::health::{
    health_service_from_reporter, spawn_health_supervisor, HealthSupervisorConfig, OVERALL_SERVICE,
};
use crate::observability::metrics::Metrics;
use crate::observability::scaling::{ScalingEndpoint, ServiceScalingSource};
use crate::storage::client::ValkeyClient;
use crate::storage::keys::KeySchema;

/// How long the held-mode server is given, after the shutdown signal, to
/// finish what it is serving before [`run_held_mode`] stops waiting for it.
/// The same 5 s as the bootstrap server's handoff budget in `main.rs`: there
/// are no LiveService streams to drain, and the probe, discovery and refusal
/// calls are unary and sub-second. A stream on another service — a client
/// holding a reflection stream open across the signal — can outlast it, and
/// that is what this budget bounds: past it, held mode stops waiting and
/// returns; the open stream's connection, which tonic runs as a task of its
/// own, is not ended by that and lasts until the process exits.
pub const HELD_MODE_SERVER_STOP_BUDGET: Duration = Duration::from_secs(5);

/// The LiveService in held mode: every RPC refused by a closed hold gate.
pub type HeldLiveService = HoldGateService<RefuseAllLiveService>;

/// The refuse-all LiveService stub of held mode (SDD SR-5.3): a [`HoldGate`]
/// over `verdicts`, closed for a hold before it serves, around
/// [`RefuseAllLiveService`].
///
/// Every LiveService RPC, and every other method under the LiveService's
/// name, is refused by the gate before anything else runs: `UNAVAILABLE`
/// `SERVICE_HELD` (`RefusalDetail::service_held_status`), naming the
/// incident of the latest verdict on `verdicts` at each refusal — a new
/// incident can replace the old one while held — and logged once at INFO,
/// marked hold-caused (`grpc::rpc_log`). The gate never reopens, so the inner
/// service is never called.
pub fn held_live_service(verdicts: watch::Receiver<HoldObservation>) -> HeldLiveService {
    let gate = HoldGate::new(verdicts);
    gate.close_held();
    gate.wrap(RefuseAllLiveService)
}

/// The inner service of [`held_live_service`]: answers to the LiveService's
/// name with no `LiveEditService` behind it. Never reached through a closed
/// gate, which never reopens; called directly, it refuses every call
/// `SERVICE_HELD` with `incident unknown` and logs nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefuseAllLiveService;

impl NamedService for RefuseAllLiveService {
    const NAME: &'static str = SERVICE_NAME;
}

impl Service<http::Request<tonic::body::Body>> for RefuseAllLiveService {
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = Ready<Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _request: http::Request<tonic::body::Body>) -> Self::Future {
        futures::future::ready(Ok(RefusalDetail::service_held_status(None).into_http()))
    }
}

/// What [`run_held_mode`] runs on. `main.rs` builds it from what exists at
/// the held-mode decision; a test builds it around a test store.
pub struct HeldMode {
    /// The Valkey client: the hold poll's reads and the health probe.
    pub valkey: ValkeyClient,
    /// The key schema the hold marker is read under.
    pub keys: KeySchema,
    /// The hold poll's interval and per-read bound.
    pub poll: HoldPollSettings,
    /// The startup hold evaluation's observation: `Held`. It seeds the poll's
    /// channel, so the stub names its incident until the first poll.
    pub startup: HoldObservation,
    /// The metrics registry `/metrics` serves.
    pub metrics: Arc<Metrics>,
    /// The metrics and deadline layer every server runs behind.
    pub metrics_layer: MetricsLayer,
    /// The health reporter the bootstrap server answered from, so readiness
    /// and `""` carry over the handoff.
    pub health_reporter: HealthReporter,
    /// The readiness service name (the LiveService's).
    pub readiness_service_name: String,
    /// The health supervisor's probe interval (`LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS`).
    pub health_probe_interval: Duration,
    /// The health supervisor's probe timeout (`LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS`).
    pub health_probe_timeout: Duration,
    /// The storage-api service-token provider, when configured: a readiness
    /// input, as in normal mode.
    pub service_token_provider: Option<Arc<ServiceTokenProvider>>,
    /// The `/scaling/v1` slot, still empty: held mode registers its source.
    pub scaling_endpoint: Arc<ScalingEndpoint>,
    /// When the process started, for the scaling snapshot's uptime.
    pub process_started: std::time::Instant,
    /// `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`, echoed on the scaling
    /// snapshot.
    pub catchup_read_concurrency: u32,
    /// `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`: sizes the held server's HTTP/2
    /// connection window, as on every server (`production_server_builder`),
    /// so the window does not change when held mode takes the socket over
    /// from the bootstrap server.
    pub upload_admission_max_waiters: usize,
    /// The held-mode reactor's settings: the heartbeat interval, the exit
    /// seam, and the `verdict_stale` and `held` signals, whose receivers held
    /// mode subscribes for the health supervisor and the scaling source.
    pub reactor: HoldReactorSettings,
}

impl std::fmt::Debug for HeldMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldMode")
            .field("poll", &self.poll)
            .field("startup", &self.startup)
            .field("readiness_service_name", &self.readiness_service_name)
            .field("reactor", &self.reactor)
            .finish_non_exhaustive()
    }
}

/// Run held mode (SDD SR-5.3–5.6) on `incoming`, the gRPC listening socket,
/// until the process exits or `shutdown_signal` resolves.
///
/// In order, before it serves:
///
/// 1. `live_edit_dr_held` set to 1 ([`Metrics::set_dr_held`]; the entry
///    counter is not ticked: no entry while serving happened), and one ERROR
///    naming the incident, reason and evidence;
/// 2. **the hold poll** (`crate::dr::hold::spawn_hold_poll`), seeded with
///    `mode.startup`;
/// 3. **the held-mode reactor** ([`spawn_held_mode_reactor`]), with its held
///    heartbeat: `held` set `true`; exit 0 after the jitter on `Serving`;
///    readiness `NOT_SERVING` and `live_edit_dr_hold_verdict_stale` 1 after
///    30 s with no successful read, reading on with no time limit; exit
///    non-zero at once if the hold poll ends before shutdown;
/// 4. **its own health supervisor** (`spawn_health_supervisor`): the Valkey
///    probe, the service-token input, no counter-seed gate (its contribution
///    is true), `api_serving`, and the reactor's `verdict_stale`;
/// 5. **the scaling source** on `mode.scaling_endpoint`
///    (`ServiceScalingSource`), with the reactor's `held` and a
///    `shutting_down` that this function sets on the shutdown signal;
/// 6. **the server**: [`held_live_service`], health, reflection and the
///    `CapabilitiesService`, behind `mode.metrics_layer`, on `incoming`, with
///    `api_serving` set `true` immediately before it takes the socket.
///
/// Those are all the tasks it spawns, beside the server: the poll, the
/// reactor and its heartbeat, and the health supervisor. The metrics server
/// and the service-token providers `main.rs` started before the held-mode
/// decision keep running.
///
/// Returns when:
///
/// - `shutdown_signal` resolves (in production, the process shutdown
///   request held mode claimed at its entry, cancelled on the first SIGTERM
///   or SIGINT; SDD SR-5.6): the poll and reactor stop first, the scaling
///   snapshot reports `shutting_down`, readiness and `""` go `NOT_SERVING`,
///   and the server
///   stops, waited on for at most [`HELD_MODE_SERVER_STOP_BUDGET`]; there are
///   no LiveService streams to drain. `Ok(())`.
/// - the reactor ends after calling its exit seam (`mode.reactor.exit`).
///   The production seam ends the process, so this happens only under a test
///   seam that returns. Everything is stopped as above. `Ok(())`.
/// - the reactor task fails (it panicked, or was cancelled): one ERROR, then
///   everything is stopped as above and
///   [`HeldModeError::ReactorFailed`] is returned, so the caller exits
///   non-zero rather than run on with nothing acting on the verdict. In
///   production this is the only way this arm is reached.
/// - the server fails: everything is stopped, and
///   [`HeldModeError::Server`] is returned.
///
/// Every task it spawned is aborted when it returns.
pub async fn run_held_mode<F>(
    mode: HeldMode,
    incoming: TcpIncoming,
    shutdown_signal: F,
) -> Result<(), HeldModeError>
where
    F: Future<Output = ()>,
{
    let HeldMode {
        valkey,
        keys,
        poll,
        startup,
        metrics,
        metrics_layer,
        health_reporter,
        readiness_service_name,
        health_probe_interval,
        health_probe_timeout,
        service_token_provider,
        scaling_endpoint,
        process_started,
        catchup_read_concurrency,
        upload_admission_max_waiters,
        reactor,
    } = mode;

    metrics.set_dr_held(true);
    tracing::error!(
        incident = startup.verdict.incident().unwrap_or("unknown"),
        reason = startup.verdict.reason().unwrap_or("unknown"),
        evidence = startup.verdict.evidence().unwrap_or("unknown"),
        "the fleet is held for disaster recovery: this replica starts in held mode. It refuses \
         every LiveService RPC SERVICE_HELD, serves health, reflection and capabilities, runs no \
         store- or backup-writing task, and writes nothing. It stays held until an operator \
         restores the backing store or accepts its loss, then restarts"
    );

    // Stops the poll and the reactor (and its heartbeat): cancelled first
    // when held mode ends.
    let poll_shutdown = CancellationToken::new();
    let (verdicts, poll_task) =
        spawn_hold_poll(valkey.clone(), keys, poll, startup, poll_shutdown.clone());
    let _poll_task = AbortOnDropHandle::new(poll_task);

    let verdict_stale = reactor.verdict_stale.subscribe();
    let held = reactor.held.subscribe();
    let mut reactor_task = AbortOnDropHandle::new(spawn_held_mode_reactor(
        verdicts.clone(),
        Arc::clone(&metrics),
        poll_shutdown.clone(),
        reactor,
    ));

    let (api_serving_tx, api_serving_rx) = watch::channel(false);
    let health_cancel = CancellationToken::new();
    let _health_task = AbortOnDropHandle::new(spawn_health_supervisor(
        health_reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: health_probe_interval,
            probe_timeout: health_probe_timeout,
            readiness_service_name: readiness_service_name.clone(),
            service_token_provider,
            // SDD SR-5.4: no counter seed runs in held mode, and an absent
            // gate contributes true, so a held fleet reports SERVING.
            counter_seed_ready: None,
            api_serving: Some(api_serving_rx),
            verdict_stale: Some(verdict_stale),
        },
        health_cancel.clone(),
    ));

    // Set on the shutdown signal: the scaling snapshot's `shutting_down`.
    let shutting_down = CancellationToken::new();
    {
        let admission = Arc::clone(&valkey.subscriber_admission);
        let shutting_down = shutting_down.clone();
        let registered = scaling_endpoint.register(Arc::new(ServiceScalingSource {
            metrics: Arc::clone(&metrics),
            max_subscriptions: admission.capacity().map_or(0, |c| c as u64),
            admission_waiters: Box::new(move || admission.waiters() as u64),
            health: tonic_health::server::HealthService::from_health_reporter(
                health_reporter.clone(),
            ),
            readiness_service_name: readiness_service_name.clone(),
            shutting_down: Box::new(move || shutting_down.is_cancelled()),
            process_started,
            catchup_read_concurrency,
            max_concurrent_catchups: valkey
                .catchup_gate
                .limit()
                .map_or(0, |l| u32::try_from(l).unwrap_or(u32::MAX)),
            cpu_seconds: crate::observability::process_cpu::cpu_seconds_total,
            held,
        }));
        if !registered {
            tracing::error!(
                "held mode: a /scaling/v1 source was already registered, so the scaler does not \
                 see this replica as held"
            );
        }
    }

    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(
            live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::FILE_DESCRIPTOR_SET,
        )
        .build_v1()
        .expect("failed to build the held-mode reflection service");

    let stop_server = CancellationToken::new();
    let serve = production_server_builder(upload_admission_max_waiters)
        .layer(metrics_layer)
        .add_service(held_live_service(verdicts))
        .add_service(health_service_from_reporter(&health_reporter))
        .add_service(reflection)
        // Bare `.add_service`, outside any auth, as in every server (SDD §5.7).
        .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
        .serve_with_incoming_shutdown(incoming, stop_server.clone().cancelled_owned());
    tokio::pin!(serve);
    tokio::pin!(shutdown_signal);

    // As at the normal serving point: readiness may rise only once the server
    // that answers on the socket is this one.
    let _ = api_serving_tx.send(true);
    tracing::info!("gRPC server serving in held mode: the refuse-all LiveService stub, health, reflection and capabilities");

    // `Some` when the server ended on its own; `reactor_failure` when the
    // reactor task panicked or was cancelled.
    let mut reactor_failure = None;
    let served = tokio::select! {
        result = &mut serve => Some(result),
        () = &mut shutdown_signal => {
            tracing::info!(
                "held mode: stopping on the shutdown signal; there are no LiveService streams \
                 to drain"
            );
            None
        }
        joined = &mut reactor_task => {
            match joined {
                Ok(()) => tracing::info!(
                    "held mode: the hold reactor ended after its exit; stopping"
                ),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "held mode: the hold reactor failed, so this replica no longer acts on \
                         the hold verdict or its age; stopping, and exiting non-zero so it \
                         restarts"
                    );
                    reactor_failure = Some(e);
                }
            }
            None
        }
    };

    poll_shutdown.cancel();
    shutting_down.cancel();
    health_cancel.cancel();
    health_reporter
        .set_service_status(&readiness_service_name, ServingStatus::NotServing)
        .await;
    health_reporter
        .set_service_status(OVERALL_SERVICE, ServingStatus::NotServing)
        .await;

    let served = match served {
        Some(result) => {
            if let Err(e) = &result {
                tracing::error!(error = %e, "held mode: the gRPC server failed");
            }
            result
        }
        None => {
            stop_server.cancel();
            match tokio::time::timeout(HELD_MODE_SERVER_STOP_BUDGET, &mut serve).await {
                Ok(result) => result,
                Err(_) => {
                    tracing::warn!(
                        budget_secs = HELD_MODE_SERVER_STOP_BUDGET.as_secs(),
                        "held mode: the gRPC server did not stop within its budget; leaving it"
                    );
                    Ok(())
                }
            }
        }
    };
    match reactor_failure {
        Some(e) => Err(HeldModeError::ReactorFailed(e)),
        None => served.map_err(HeldModeError::Server),
    }
}

/// Why [`run_held_mode`] ended in failure. Either way the process must exit
/// non-zero.
#[derive(Debug, thiserror::Error)]
pub enum HeldModeError {
    /// The held-mode gRPC server failed.
    #[error("the held-mode gRPC server failed: {0}")]
    Server(#[source] tonic::transport::Error),
    /// The held-mode reactor task panicked or was cancelled, so nothing acts
    /// on the hold verdict or its age any more: no un-hold exit and no stale
    /// readiness.
    #[error("the held-mode hold reactor failed: {0}")]
    ReactorFailed(#[source] tokio::task::JoinError),
}
