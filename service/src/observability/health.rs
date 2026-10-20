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

//! Health-check service for Kubernetes liveness/readiness probes.
//!
//! Two service names are exposed through the gRPC health protocol:
//! - `""` — overall liveness probe target (is the process up?)
//! - `"nvidia.omniverse.live.v1alpha.LiveService"` — readiness probe target
//!   (are we ready to serve traffic?)
//!
//! The two statuses are deliberately asymmetric at construction: `""` is
//! `SERVING` from the moment the reporter exists (the process answering is
//! the liveness contract), while the readiness name starts `NOT_SERVING`
//! and is only raised by the supervisor's first successful probe. That
//! asymmetry is load-bearing since b.rgs — the gRPC socket is bound and a
//! bootstrap health server is answering before any dependency is wired, so
//! a readiness name that started `SERVING` would advertise a replica that
//! has not yet connected to Valkey.
//!
//! The same asymmetry is why the supervisor also gates readiness on the
//! full-API-server startup signal: the bootstrap server carries health,
//! reflection, and the Discovery capabilities service but NOT `LiveService`,
//! so publishing `SERVING` before the handoff would route real RPCs into a
//! terminal `UNIMPLEMENTED`.
//!
//! Post-b.3z7 H-obs-2, the readiness status is no longer pinned to
//! `SERVING` at startup; a supervised background task periodically pings
//! Valkey (plus the service-identity token, counter-seed startup gate, and
//! full-API-server startup gate, when wired) and flips the readiness status
//! to `NOT_SERVING` whenever a
//! dependency becomes unreachable, then back to `SERVING` once it
//! recovers. During an outage, Kubernetes routes
//! traffic away from the affected replica and the autoscaler and Deployment
//! controller handle pod replacement rather than serving errors to clients.
//!
//! The module takes the readiness service name as a `&str` (b.3z7 medium)
//! so it does not depend on concrete `grpc::proto` types. Callers pass the
//! protobuf service name directly (e.g.
//! `"nvidia.omniverse.live.v1alpha.LiveService"`).

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::ClientLike;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic_health::pb::health_server::HealthServer;
use tonic_health::server::{HealthReporter, HealthService};
use tonic_health::ServingStatus;

use crate::auth::service_token::ServiceTokenProvider;
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::ValkeyClient;

/// The overall (liveness) health-check service name. Uses the gRPC health
/// protocol convention that `""` represents the whole server.
pub const OVERALL_SERVICE: &str = "";

/// Create the gRPC health service and reporter.
///
/// Registers two service names:
/// - [`OVERALL_SERVICE`] (liveness probe) as `SERVING`
/// - `readiness_service_name` (readiness probe — typically
///   `"nvidia.omniverse.live.v1alpha.LiveService"`) as `NOT_SERVING`
///
/// Readiness starts `NOT_SERVING` and is raised by
/// [`spawn_health_supervisor`]'s first probe that finds every gated
/// dependency healthy — including the
/// [`HealthSupervisorConfig::api_serving`] gate, which does not open until
/// the full-API server takes over the socket. Registering readiness as
/// `SERVING` would be an outright lie now that the socket answers before any
/// dependency is wired (b.rgs), and it also papered over a latent bug: a
/// failed counter-seed left readiness advertising `SERVING` until the
/// supervisor's first tick corrected it.
///
/// Returns the reporter (used later by the supervisor and the shutdown
/// sequence) and the health service to be added to the tonic `Server`. Use
/// [`health_service_from_reporter`] to build additional service instances
/// sharing the same status map.
pub async fn create_health_service(
    readiness_service_name: &str,
) -> (HealthReporter, HealthServer<HealthService>) {
    let reporter = HealthReporter::new();

    // Liveness: empty string represents the overall server — set SERVING
    // immediately because the process being up is the liveness contract.
    reporter
        .set_service_status(OVERALL_SERVICE, ServingStatus::Serving)
        .await;

    // Readiness: start NOT_SERVING; the supervisor (see
    // [`spawn_health_supervisor`]) raises it once every gated dependency
    // reports healthy, and flips it back during an outage.
    reporter
        .set_service_status(readiness_service_name, ServingStatus::NotServing)
        .await;

    let service = health_service_from_reporter(&reporter);

    (reporter, service)
}

/// Build an additional [`HealthServer`] backed by an existing reporter's
/// status map.
///
/// `HealthReporter` is `Clone` and `HealthService::from_health_reporter`
/// carries the shared status map across, so every service built this way
/// answers from the same state. Startup uses this to run two tonic servers
/// over one listening socket — the bootstrap server (health + reflection +
/// capabilities, up before dependencies are wired) and the full-API server —
/// without needing to clone the service itself.
pub fn health_service_from_reporter(reporter: &HealthReporter) -> HealthServer<HealthService> {
    HealthServer::new(HealthService::from_health_reporter(reporter.clone()))
}

/// Configuration for the health supervisor.
///
/// Readiness is gated on Valkey (hard, one-strike) plus the optional
/// service-identity token provider, counter-seed startup gate, and
/// full-API-server startup gate. Auth
/// backends are deliberately NOT probed for readiness — gating readiness
/// on a shared downstream would flip the entire replica fleet to
/// `NOT_SERVING` at once on a correlated auth-backend outage (b.9f4,
/// supersedes b.53x / b.3z7 H-obs-2's auth-probe design).
#[derive(Clone)]
pub struct HealthSupervisorConfig {
    /// How often to probe each dependency.
    pub probe_interval: Duration,
    /// Per-probe deadline. A probe that doesn't return inside this window
    /// counts as a failure.
    pub probe_timeout: Duration,
    /// Protobuf service name to flip between SERVING/NOT_SERVING.
    pub readiness_service_name: String,
    /// Service-account token provider to consult for readiness. When
    /// `Some`, the supervisor checks `cached_bearer_if_unexpired()` on
    /// every tick and includes its `Some(_)` / `None` state in the
    /// aggregate serving verdict (hard signal — no smoothing). When
    /// `None` (the
    /// `LIVE_EDIT_SERVICE_TOKEN_FILE`-unset wiring from `main.rs`),
    /// the check is omitted entirely, the gauge is not registered, and
    /// the readiness model degrades to the Valkey baseline.
    /// Wired by Task `t2.fgj.dp.s7`.
    ///
    /// The accessor is the *non-withholding* one: `None` means "no token
    /// cached, or the cached token has actually expired", never merely
    /// "the cached token is inside the provider's proactive re-read lead"
    /// (b.e49 — see the probe's comment in [`spawn_health_supervisor`]).
    ///
    /// It is synchronous and lock-free per Task
    /// `t2.fgj.dp.io`'s contract — the supervisor MUST NOT `.await`
    /// the call, wrap it in `tokio::task::spawn_blocking`, or wrap it
    /// in `tokio::time::timeout`. Each tick observes whatever value
    /// the polling task has most recently written; smoothing on this
    /// signal would mask the sidecar-stopped-writing failure mode the
    /// probe exists to detect.
    pub service_token_provider: Option<Arc<ServiceTokenProvider>>,
    /// Startup-gate signal for the counter-seed recovery routine (SR-5.4 /
    /// `t2.fgj.mh.4u`). When `Some(receiver)`, the supervisor consults
    /// `*receiver.borrow()` on every tick and includes its value as another
    /// contributor to the aggregate serving verdict. Hard signal — no
    /// smoothing, same posture as `service_token_provider`. `None` collapses
    /// the contribution to `true` (gate inert).
    ///
    /// Wired by `main.rs` after `seed_object_id_counter` flips a
    /// `tokio::sync::watch::channel` from `false` to `true` on success. On
    /// routine failure the watch stays at `false` and readiness remains
    /// `NOT_SERVING` — Kubernetes pulls the pod and the operator sees both
    /// the routine's error log and the failing readiness probe. When
    /// checkpoints are disabled (`config.checkpoint_enabled == false`) the
    /// routine is skipped and the field is wired as `None` so the gate is
    /// inert and readiness depends only on Valkey / service-token state.
    pub counter_seed_ready: Option<tokio::sync::watch::Receiver<bool>>,
    /// Startup-gate signal for the full-API server (b.rgs). `false` until
    /// `main.rs` is about to call `serve_with_incoming_shutdown` on the
    /// serving `TcpIncoming`; `true` from that point on.
    ///
    /// Load-bearing because the socket is bound and a *bootstrap* server
    /// (health + reflection + capabilities only) answers on it from the top
    /// of `main()`. Without this gate the supervisor's first tick — which
    /// fires immediately — can satisfy Valkey + service-token + counter-seed
    /// and publish `SERVING` while `LiveService` is still unregistered, so
    /// every routed RPC in that window terminates `UNIMPLEMENTED`; and a boot
    /// that wedges after the supervisor spawns leaves the pod Ready, live,
    /// and startup-probe-passed forever with no restart path.
    ///
    /// Hard signal — no smoothing, same posture as `counter_seed_ready`.
    /// `None` collapses the contribution to `true` (gate absent) for callers
    /// that drive the supervisor without a tonic server in front of it.
    ///
    /// The supervisor also `select!`s on this receiver's `changed()`, so the
    /// readiness flip lands on the transition rather than up to one
    /// `probe_interval` after it.
    pub api_serving: Option<tokio::sync::watch::Receiver<bool>>,
}

impl Default for HealthSupervisorConfig {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(2),
            readiness_service_name: String::new(),
            service_token_provider: None,
            counter_seed_ready: None,
            api_serving: None,
        }
    }
}

impl std::fmt::Debug for HealthSupervisorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The service-token provider / watch receiver don't impl Debug;
        // render their presence/absence so log output still distinguishes
        // the wired vs unwired cases.
        f.debug_struct("HealthSupervisorConfig")
            .field("probe_interval", &self.probe_interval)
            .field("probe_timeout", &self.probe_timeout)
            .field("readiness_service_name", &self.readiness_service_name)
            .field(
                "service_token_provider",
                &self.service_token_provider.is_some(),
            )
            .field("counter_seed_ready", &self.counter_seed_ready.is_some())
            .field("api_serving", &self.api_serving.is_some())
            .finish()
    }
}

/// Spawn a background task that periodically probes Valkey (plus the
/// service-identity token, counter-seed startup gate, and full-API-server
/// startup gate, when wired) and
/// flips the gRPC readiness status so Kubernetes routes traffic away from
/// replicas whose backing dependencies are unreachable (b.3z7 H-obs-2;
/// auth-backend probing removed in b.9f4).
///
/// The task:
/// - pings Valkey on every `probe_interval` tick under a `probe_timeout`.
///   Valkey is treated as a hard dependency — a single Valkey ping failure
///   counts as the readiness signal flipping to NOT_SERVING because the
///   service can't serve any layer-state RPC without it.
/// - flips readiness to `NOT_SERVING` on the first failure after a run of
///   successes, and back to `SERVING` on the first success after a run of
///   failures (edge-triggered — the reporter is only written to on
///   transitions). The very first tick always publishes a verdict and logs
///   it, including when that verdict agrees with the `NOT_SERVING` state
///   `create_health_service` registered: otherwise a replica that never
///   becomes ready would produce no log naming the dependency holding it
///   back.
/// - exits when `cancel` fires.
///
/// Besides the periodic tick, the loop wakes on a change to
/// [`HealthSupervisorConfig::api_serving`] (b.rgs) so the readiness flip
/// lands the moment the full-API server takes over the socket rather than up
/// to one `probe_interval` later.
pub fn spawn_health_supervisor(
    reporter: HealthReporter,
    valkey: ValkeyClient,
    metrics: SharedMetrics,
    cfg: HealthSupervisorConfig,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut timer = tokio::time::interval(cfg.probe_interval);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // `None` until the first tick lands. `create_health_service`
        // registers readiness as `NOT_SERVING`, so the first tick is
        // edge-triggered into `SERVING` once every gated dependency reports
        // healthy — the interval's first tick fires immediately, so that
        // costs one probe round-trip of ready-latency, not one probe period.
        //
        // Tracking `Option<bool>` rather than seeding `false` makes the first
        // tick count as a transition in BOTH directions. That matters for the
        // unhealthy-at-boot case: with a plain `false` seed the supervisor
        // would silently agree with the initial state and emit no log at all,
        // so an operator watching a replica that never becomes ready would
        // have nothing naming the dependency that is holding it back.
        let mut last_serving: Option<bool> = None;

        let readiness = Arc::<str>::from(cfg.readiness_service_name.clone());

        // A second receiver used ONLY as a wakeup source (b.rgs). The verdict
        // itself still reads `cfg.api_serving`'s `borrow()`, so a dropped
        // sender never changes the observed value — it only stops the wakeups,
        // which `wait_for_api_serving_change` degrades to `pending()` so the
        // loop parks on the timer instead of spinning on `Err`.
        let mut api_serving_wakeup = cfg.api_serving.clone();

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("health supervisor shutting down");
                    break;
                }
                _ = timer.tick() => {}
                () = wait_for_api_serving_change(&mut api_serving_wakeup) => {}
            }

            let valkey_ok = probe_valkey(&valkey, cfg.probe_timeout).await;
            // b.u96 item 8: tick the Valkey probe-failure counter
            // on every probe failure. The counter gives operators
            // a first-class signal for partial degradation
            // independent of the readiness flip.
            if !valkey_ok {
                metrics.inc_health_probe_failure("valkey");
            }

            // ── Service-identity token availability probe (Task t2.fgj.dp.s7) ──
            //
            // The threshold is **true credential death** (b.e49):
            // `cached_bearer_if_unexpired()` is `None` only when nothing
            // has ever been cached or the cached token is actually past
            // `expires_on`. The provider's refresh lead is a re-read
            // cadence for its polling task, never a withholding window,
            // and nothing here may gate on it: an approaching expiry
            // must not flip readiness. Gating on the lead once did, and
            // it cost a `NOT_SERVING` flip per token lifetime on a
            // replica still holding a working credential — readiness is
            // a routing actuator, so that removes the pod from ALL gRPC
            // traffic, fleet-wide and correlated when token expiry is
            // correlated across a rolling deploy. (The withholding
            // accessor that supplied that rule lost this consumer in
            // b.e49 and was deleted outright in b.mq8.) Early warning is
            // not this signal's job:
            // `live_edit_service_token_expires_in_seconds` and the b.is1
            // rotation-headroom gauge already carry it. This also makes
            // the recorded value match the gauge's documented contract
            // ("a non-expired token is cached").
            //
            // Still a hard signal — flips immediately on each
            // transition with no smoothing. The probe exists to
            // surface a sidecar that has stopped writing the token
            // file, and a consecutive-failure smoother would mask
            // exactly that. The sidecar is per-pod, so a dead one is
            // an *uncorrelated* fault worth routing away from —
            // unlike the shared auth backends b.9f4 removed from the
            // verdict. The accessor is synchronous and lock-free per
            // Task 1's contract — it must NOT be `.await`ed, must NOT
            // be wrapped in `spawn_blocking`, and must NOT be wrapped
            // in `tokio::time::timeout`.
            //
            // When the provider field is `None` (the
            // `LIVE_EDIT_SERVICE_TOKEN_FILE`-unset wiring from
            // `main.rs`) the supervisor omits the check entirely
            // and the contribution is a constant `true` so the
            // readiness model collapses cleanly to the Valkey
            // baseline. `set_service_token_available`
            // is a no-op when the gauge has not been registered,
            // which is exactly the unset-env-var case here.
            let service_token_ok = if let Some(provider) = cfg.service_token_provider.as_deref() {
                let available = provider.cached_bearer_if_unexpired().is_some();
                metrics.set_service_token_available(available);
                available
            } else {
                true
            };

            // ── Counter-seed startup gate (Task t2.fgj.mh.4u / SR-5.4) ──
            //
            // Hard signal — no smoothing, same posture as
            // `service_token_provider`. When `Some(rx)` the supervisor
            // reads the most-recently-published value on every tick;
            // `false` (the default until `seed_object_id_counter`
            // succeeds) holds readiness at `NOT_SERVING` so Kubernetes
            // routes traffic away from a replica whose counter has not
            // yet been seeded. When `None` (checkpoint-disabled wiring
            // from `main.rs`) the contribution is a constant `true`
            // and the gate is inert.
            let counter_seed_ok = if let Some(rx) = cfg.counter_seed_ready.as_ref() {
                *rx.borrow()
            } else {
                true
            };

            // ── Full-API-server startup gate (b.rgs) ──
            //
            // Hard signal — no smoothing, same posture as
            // `counter_seed_ready`. `false` until `main.rs` is about
            // to serve the full API over the handed-off socket; until
            // then only the bootstrap server (health + reflection +
            // capabilities) is answering, so a `SERVING` readiness
            // verdict would route LiveService RPCs into terminal
            // `UNIMPLEMENTED`. When `None` the contribution is a
            // constant `true` and the gate is inert.
            let api_serving_ok = if let Some(rx) = cfg.api_serving.as_ref() {
                *rx.borrow()
            } else {
                true
            };

            let serving = valkey_ok && service_token_ok && counter_seed_ok && api_serving_ok;

            if last_serving != Some(serving) {
                let first_verdict = last_serving.is_none();
                let new_status = if serving {
                    ServingStatus::Serving
                } else {
                    ServingStatus::NotServing
                };
                reporter
                    .set_service_status(readiness.as_ref(), new_status)
                    .await;
                if serving && first_verdict {
                    tracing::info!(
                            service = readiness.as_ref(),
                            "health supervisor: readiness raised to SERVING (all gated dependencies healthy)"
                        );
                } else if serving {
                    tracing::info!(
                        service = readiness.as_ref(),
                        "health supervisor: readiness restored to SERVING (dependencies recovered)"
                    );
                } else {
                    tracing::warn!(
                            service = readiness.as_ref(),
                            valkey_ok,
                            service_token_ok,
                            counter_seed_ok,
                            api_serving_ok,
                            first_verdict,
                            "health supervisor: readiness flipped to NOT_SERVING (dependency unreachable)"
                        );
                }
                last_serving = Some(serving);
            }
        }
    })
}

/// Wait for the next change on the `api_serving` startup gate (b.rgs).
///
/// Resolves when the watch value is republished. When the gate is unwired
/// (`None`) — or once the sender has been dropped, which leaves the last
/// value permanently readable via `borrow()` — this parks forever so the
/// supervisor's `select!` falls back to the periodic timer instead of
/// spinning on an immediately-ready `Err`.
async fn wait_for_api_serving_change(rx: &mut Option<tokio::sync::watch::Receiver<bool>>) {
    match rx.as_mut() {
        Some(rx) => {
            if rx.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
        None => std::future::pending::<()>().await,
    }
}

/// Probe Valkey by issuing a `PING` on the data pool under a timeout.
/// Returns `true` on success, `false` on any error or timeout. Valkey is a
/// hard dependency — one failed ping is enough to trip the readiness flip.
///
/// A `PING` whose own command timeout expired is counted on
/// `valkey_errors_total{pool="data", kind="command_timeout"}`; the probe's
/// outer `probe_timeout` is this supervisor's deadline, not a command's, and is
/// not. That count therefore fires only when `LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS`
/// is at least `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` (on a tie fred's deadline
/// wins: it is armed on the `PING`'s first poll, lands on the same timer tick,
/// and is polled first). At the defaults (2000 ms against 5000 ms) the outer
/// deadline always elapses first. A stalled readiness `PING` shows on
/// `live_edit_health_probe_failures_total`.
async fn probe_valkey(valkey: &ValkeyClient, probe_timeout: Duration) -> bool {
    let ping = valkey.pool.ping::<()>(None);
    match tokio::time::timeout(probe_timeout, ping).await {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            crate::storage::client::FredErrOrigin::client(
                valkey,
                crate::storage::client::POOL_LABEL_DATA,
            )
            .observe_error(&e);
            false
        }
        Err(_elapsed) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tonic_health::pb::health_server::Health;

    // ═══════════════════════════════════════════════════════════════════════════
    // Shared end-to-end supervisor test harness (real Valkey + gRPC health client)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // These tests drive the supervisor against a real Valkey (so the
    // Valkey probe stays SERVING) and observe readiness transitions by
    // watching `HealthReporter` state via `set_service_status` —
    // tonic_health's `HealthReporter` doesn't expose a read API, so we
    // stand up a tiny tonic server and query via the gRPC health client.

    async fn real_valkey() -> Option<ValkeyClient> {
        let valkey_url = std::env::var("TEST_VALKEY_URL")
            .or_else(|_| std::env::var("LIVE_EDIT_VALKEY_ADDR"))
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let cfg = crate::config::Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url,
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: 5,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 500,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: Some("http://unused.test".to_string()),
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: Some("http://unused.test".to_string()),
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 15,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: "test".to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: crate::config::AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: crate::config::UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            session_tokens_sweep_interval_secs: 60,
            checkpoint_enabled: false,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::Disabled,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: 1000,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        };
        match tokio::time::timeout(
            Duration::from_secs(5),
            crate::storage::test_support::connect_test_client(&cfg),
        )
        .await
        {
            Ok(Ok(c)) => Some(c),
            _ => None,
        }
    }

    const READINESS_SERVICE: &str = "nvidia.omniverse.live.v1alpha.LiveService";

    /// Stand up a tonic server exposing the supplied HealthServer on a
    /// random port and return (addr, shutdown).
    async fn spawn_health_server<H>(
        service: HealthServer<H>,
    ) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>)
    where
        H: Health + Send + Sync + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        (addr, tx)
    }

    async fn observe_status(
        client: &mut tonic_health::pb::health_client::HealthClient<tonic::transport::Channel>,
    ) -> tonic_health::pb::health_check_response::ServingStatus {
        let resp = client
            .check(tonic_health::pb::HealthCheckRequest {
                service: READINESS_SERVICE.to_string(),
            })
            .await
            .unwrap()
            .into_inner();
        tonic_health::pb::health_check_response::ServingStatus::try_from(resp.status).unwrap()
    }

    async fn wait_for_status(
        client: &mut tonic_health::pb::health_client::HealthClient<tonic::transport::Channel>,
        target: tonic_health::pb::health_check_response::ServingStatus,
        deadline: Duration,
    ) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < deadline {
            if observe_status(client).await == target {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    async fn connect_health(
        addr: std::net::SocketAddr,
    ) -> tonic_health::pb::health_client::HealthClient<tonic::transport::Channel> {
        let endpoint = format!("http://{addr}");
        let channel = tonic::transport::Channel::from_shared(endpoint)
            .unwrap()
            .connect()
            .await
            .unwrap();
        tonic_health::pb::health_client::HealthClient::new(channel)
    }

    /// Publish `SERVING` on the readiness name *before* the supervisor is
    /// spawned, so that a later `NOT_SERVING` observation is necessarily the
    /// supervisor's own verdict.
    ///
    /// Needed since b.rgs. `create_health_service` now registers readiness as
    /// `NOT_SERVING`, which makes a bare "readiness reaches NOT_SERVING"
    /// assertion vacuous — it holds from t=0 and would pass against a
    /// supervisor that never ran a tick, or one whose dependency classifier
    /// was deleted outright. Seeding `SERVING` first restores the assertion's
    /// teeth: the only writer that can move it back is the supervisor, so the
    /// test once again observes a real `SERVING → NOT_SERVING` transition.
    ///
    /// The seeded `SERVING` is a *detector*, not a claim that production ever
    /// reaches that state with the dependency in question already broken.
    async fn preseed_readiness_serving(reporter: &HealthReporter) {
        reporter
            .set_service_status(READINESS_SERVICE, ServingStatus::Serving)
            .await;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Service-identity token availability tests (Task t2.fgj.dp.s7)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // The supervisor's fourth dependency check consults
    // `ServiceTokenProvider::cached_bearer_if_unexpired()` (b.e49 — before that
    // it read a second accessor that withheld a cached token inside the refresh
    // lead, since deleted in b.mq8). The contract:
    //
    //   - `Some(_)` → backend contributes OK to the aggregate. Including a
    //                 token that is seconds from expiry: it is still a working
    //                 credential, and readiness is a routing actuator, not an
    //                 early-warning channel.
    //   - `None`    → backend contributes NOT OK (flips readiness immediately —
    //                 no consecutive-failure smoother because the probe exists
    //                 to surface a sidecar that has stopped writing the file,
    //                 and a smoother would mask exactly that failure mode).
    //                 `None` means nothing has ever been cached, or the cached
    //                 token has actually expired.
    //   - When the `service_token_provider` config field is itself `None`
    //     (the `LIVE_EDIT_SERVICE_TOKEN_FILE`-unset wiring from `main.rs`)
    //     the supervisor MUST omit the check entirely — no bearer read on the
    //     absent provider, no write to the
    //     `live_edit_service_token_available` gauge, and the aggregate
    //     readiness model collapses cleanly to the pre-Task Valkey + auth
    //     baseline.
    //
    // The supervisor's token-check logic is inlined in the tick body
    // (no public helper function to call directly), so these tests drive
    // `spawn_health_supervisor` end-to-end and observe via the gRPC
    // health client — same shape as the auth-probe tests above. A real
    // `ServiceTokenProvider` is constructed against a `tempdir`-backed
    // token file; rewriting the file mid-test flips the probe's answer and
    // the supervisor picks it up on the next
    // tick. There is no filesystem coupling beyond that: the polling
    // task runs at the same sub-100 ms cadence as the existing
    // `service_token.rs` unit tests so individual cases stay well under
    // five seconds.
    //
    // There is no injectable clock — the accessor calls `Utc::now()` — so the
    // expiry dimension is driven by parameterising the fixture's `expires_on`
    // offset, exactly as `service_token.rs`'s own accessor tests do.

    use tempfile::tempdir;

    /// Build a real `Arc<ServiceTokenProvider>` against a tempdir-backed
    /// token file. The path argument is the file the provider polls; if
    /// it does not yet exist when the test runs, the bearer accessors
    /// return `None` (the SR-6.2 startup-absent path) until a write
    /// lands and the polling task picks it up. The provider's metrics
    /// handle is allocated on a fresh private registry so it cannot
    /// collide with the supervisor's metrics.
    fn build_token_provider(
        path: std::path::PathBuf,
        poll_interval: Duration,
        cancel: CancellationToken,
    ) -> Arc<ServiceTokenProvider> {
        let token_metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        Arc::new(ServiceTokenProvider::new(
            path,
            poll_interval,
            token_metrics,
            cancel,
        ))
    }

    /// Write a valid token file with the given access token expiring an
    /// hour from now. Mirrors `service_token.rs`'s test fixture (which is
    /// a sibling module — duplicating the body here keeps test scope
    /// confined to `src/observability/health.rs` per the Subtask
    /// scoping constraint).
    async fn write_valid_token_file(path: &std::path::Path, token: &str) {
        write_token_file_expiring_in(path, token, 3600).await;
    }

    /// [`write_valid_token_file`] with a caller-chosen lifetime.
    ///
    /// `expires_in_secs` may be small (inside the provider's 60 s default
    /// refresh lead) or negative (already expired). That parameter is the only
    /// lever this module has on the expiry dimension — both bearer accessors
    /// call `Utc::now()` directly and there is no injectable clock, so the
    /// b.e49 boundary ("live but near expiry" vs "actually dead") is expressed
    /// by moving `expires_on`, the same technique `service_token.rs`'s accessor
    /// tests use.
    async fn write_token_file_expiring_in(
        path: &std::path::Path,
        token: &str,
        expires_in_secs: i64,
    ) {
        let expires_on = chrono::Utc::now() + chrono::Duration::seconds(expires_in_secs);
        let body = format!(
            r#"{{"access_token":"{}","expires_on":"{}"}}"#,
            token,
            expires_on.to_rfc3339()
        );
        tokio::fs::write(path, body).await.expect("write fixture");
    }

    /// Poll cadence for the in-test `ServiceTokenProvider`. Matches the
    /// 20 ms used by the existing `service_token.rs` tests so a write /
    /// delete is picked up within roughly one supervisor probe interval
    /// (200 ms).
    const TOKEN_POLL_INTERVAL: Duration = Duration::from_millis(20);

    /// Provider present and the token file holds a valid token → backend
    /// contributes OK; aggregate stays SERVING. This pins the
    /// `Some(token) → OK` half of the classifier (acceptance criterion
    /// #4 (a)).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_provider_some_with_valid_token_stays_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("token.json");
        write_valid_token_file(&token_path, "valid-token-A").await;

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );
        // Wait for the polling task's first tick to populate the cache
        // before the supervisor's first probe lands.
        tokio::time::sleep(Duration::from_millis(80)).await;

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Since b.rgs readiness is registered NOT_SERVING and raised by the
        // supervisor's first tick, so wait for that before sampling — a bare
        // sample would race the first probe rather than test anything.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(5),
            )
            .await,
            "precondition: a valid token + healthy Valkey must raise readiness to SERVING"
        );

        // Sample readiness across several probe cycles — must stay
        // SERVING since `cached_bearer_if_unexpired()` returns Some throughout.
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                "provider with valid token must contribute OK and keep readiness SERVING"
            );
        }

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// Provider present but the token file is missing →
    /// `cached_bearer_if_unexpired()` returns `None` (nothing cached at all)
    /// → backend contributes NOT OK → aggregate flips
    /// NOT_SERVING on the FIRST probe (no consecutive-failure smoother).
    /// Pins the `None → NOT OK` half of the classifier AND the
    /// no-smoother contract for the initial failure (acceptance criteria
    /// #4 (b) and the first transition of #4 (c)).
    ///
    /// Readiness is pre-seeded `SERVING` (see [`preseed_readiness_serving`])
    /// so the assertion is a genuine `SERVING → NOT_SERVING` transition
    /// driven by the supervisor. Without that seed the b.rgs registration
    /// change (readiness starts `NOT_SERVING`) would make the wait below
    /// succeed at t=0 against a supervisor that never evaluated the token
    /// gate at all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_provider_some_with_missing_file_flips_not_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        let dir = tempdir().expect("tempdir");
        // Path that will never be created — startup-absent path leaves
        // `cached_bearer_if_unexpired()` returning None forever.
        let token_path = dir.path().join("never-created.json");

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Readiness was seeded SERVING above, so this is a real transition:
        // even though Valkey would otherwise be OK, the missing token file
        // must pull readiness back down within ONE probe interval — a hard
        // signal with no consecutive-failure smoothing.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "missing token file (cached_bearer_if_unexpired() = None) must drive readiness from the \
             seeded SERVING down to NOT_SERVING on the first probe — no consecutive-failure \
             smoothing"
        );

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// `service_token_provider: None` AND auth clients None → the
    /// supervisor must keep readiness SERVING and must NOT touch the
    /// `live_edit_service_token_available` gauge even when the gauge
    /// has been registered. This is the "no-op when provider absent"
    /// contract (acceptance criterion #4 (d)).
    ///
    /// The sentinel-write trick: we pre-set the gauge to a distinctive
    /// value (42) before starting the supervisor. If the supervisor's
    /// else-branch erroneously called `set_service_token_available` —
    /// in either direction — the sentinel would be replaced by 0 or 1.
    /// Asserting the sentinel survives proves the branch was never
    /// taken across the run's ticks.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_provider_none_does_not_write_gauge_or_flip_readiness() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        // Keep a reader clone of the metrics so we can render `/metrics` and
        // observe whether the supervisor ever recorded the gauge. The
        // `live_edit_service_token_available` instrument is unconditional now
        // (no `register_*` seam); because an OTel gauge emits no Prometheus
        // family until first `record()`, "the supervisor did not write it"
        // manifests as the family staying ABSENT from the scrape output —
        // which is the same observable contract the old sentinel trick pinned.
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let metrics_reader = Arc::clone(&metrics);
        // Precondition: never recorded yet, so the family is absent.
        assert!(
            !crate::observability::metrics::test_util::has_metric(
                &crate::observability::metrics::test_util::render(&metrics_reader),
                "live_edit_service_token_available",
            ),
            "service_token_available must be absent before the supervisor runs"
        );

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                // Explicitly None — the gating field this test pins.
                service_token_provider: None,
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Since b.rgs readiness is registered NOT_SERVING and raised by the
        // supervisor's first tick; wait for that before sampling.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(5),
            )
            .await,
            "precondition: with no service-token gate and a healthy Valkey the supervisor \
             must raise readiness to SERVING"
        );

        // Run multiple probe cycles. Readiness must stay SERVING (the
        // pre-Task baseline) and the gauge sentinel must survive.
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                "supervisor with service_token_provider=None must keep readiness SERVING \
                 (baseline pre-Task behavior preserved)"
            );
        }

        // The supervisor with `service_token_provider: None` must NEVER record
        // the gauge, so the family must still be absent from `/metrics` after
        // several ticks. A regression where the else-branch erroneously called
        // `set_service_token_available` would materialise the family (at 0 or
        // 1), failing this assertion.
        assert!(
            !crate::observability::metrics::test_util::has_metric(
                &crate::observability::metrics::test_util::render(&metrics_reader),
                "live_edit_service_token_available",
            ),
            "supervisor must NOT record live_edit_service_token_available when \
             service_token_provider is None — the family must stay absent from /metrics"
        );

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// No-smoother contract: a `None → Some → None` sequence must flip
    /// readiness on EVERY transition within a single probe interval — the
    /// signal is hard, with no consecutive-failure smoothing (acceptance
    /// criterion #4 (c)).
    ///
    /// The flow:
    ///   1. Start with the token file absent → readiness NOT_SERVING.
    ///   2. Write a valid token file → readiness flips back to SERVING
    ///      on the FIRST tick that sees `Some(_)`.
    ///   3. Write an already-expired token → readiness flips NOT_SERVING
    ///      again on the FIRST tick that sees `None`. (Expired, not
    ///      near-expiry: post-b.e49 the probe's `None` threshold is actual
    ///      credential death — see the comment at the step itself.)
    ///
    /// If the supervisor had introduced a smoother on this signal, step
    /// 3 would require multiple consecutive `None` observations before
    /// flipping. The bounded-wait assertions guarantee single-tick
    /// transitions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_none_to_some_to_none_flips_immediately_on_each_transition() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("flipping.json");
        // Start absent → `cached_bearer_if_unexpired() = None`.

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        // Transition 1 below asserts a downward flip; seed SERVING so it is
        // one (b.rgs registers readiness NOT_SERVING). See
        // `preseed_readiness_serving`.
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Transition 1: None → readiness NOT_SERVING.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "initial absent token file must drive readiness from the seeded SERVING down \
             to NOT_SERVING within one probe"
        );

        // Transition 2: write valid token → readiness flips back to SERVING.
        write_valid_token_file(&token_path, "recovered-token").await;
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(2),
            )
            .await,
            "writing a valid token must restore readiness SERVING on the FIRST tick that \
             observes cached_bearer_if_unexpired() = Some — no smoothing on recovery"
        );

        // Transition 3: drive the probe back to `None` → readiness flips
        // NOT_SERVING again.
        //
        // Deleting the file does not do it: a NotFound read leaves the cache
        // untouched per the previous-good-cache invariant, so the still-live
        // cached token keeps answering. Neither does a malformed file (parse
        // failure, cache untouched). Post-b.e49 the probe reads
        // `cached_bearer_if_unexpired()`, whose only `None` conditions are
        // "nothing cached" and "the cached token has actually expired" — so
        // the fixture writes an *already-expired* token, which the next poll
        // caches and which the probe then rejects on its first look. A
        // near-expiry-but-live token would no longer flip anything, and that
        // is the whole point of the correction: readiness now keys on
        // credential death, not on the provider's re-read lead.
        let expires_on = chrono::Utc::now() - chrono::Duration::seconds(10); // already dead
        let body = format!(
            r#"{{"access_token":"already-expired","expires_on":"{}"}}"#,
            expires_on.to_rfc3339()
        );
        tokio::fs::write(&token_path, body)
            .await
            .expect("write expired fixture");

        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "an actually-expired cached token must flip readiness NOT_SERVING on the FIRST \
             tick that observes it — proves no consecutive-failure smoother gates this signal"
        );

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// Provider present and the token file holds a valid token → the
    /// supervisor writes `1` to the registered
    /// `live_edit_service_token_available` gauge on every tick. Pairs
    /// with the absent-provider gauge test above to pin BOTH halves of
    /// the metrics-write branch (acceptance criterion #4 (d) implies
    /// the positive side: when the provider IS Some, the gauge MUST
    /// reflect availability).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_provider_some_writes_gauge_each_tick() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        // Reader clone: the supervisor records the gauge via the OTel
        // instrument; we observe the recorded value by rendering `/metrics`.
        let metrics_reader = Arc::clone(&metrics);
        let read_gauge = move || {
            crate::observability::metrics::test_util::sample_value(
                &crate::observability::metrics::test_util::render(&metrics_reader),
                "live_edit_service_token_available",
                &[],
            )
        };

        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("token.json");
        write_valid_token_file(&token_path, "gauge-test-token").await;

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );
        // Let the polling task fill the cache before the supervisor's
        // first probe.
        tokio::time::sleep(Duration::from_millis(80)).await;

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let _client = connect_health(addr).await;

        // Wait long enough for at least one supervisor tick to land.
        // After that the gauge must read `1` (token available).
        let start = std::time::Instant::now();
        let mut gauge_value = read_gauge();
        while start.elapsed() < Duration::from_secs(2) && gauge_value != Some(1.0) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            gauge_value = read_gauge();
        }
        assert_eq!(
            gauge_value,
            Some(1.0),
            "with provider present and a cached unexpired token, the supervisor must \
             write 1 to live_edit_service_token_available on each tick. Got {gauge_value:?} \
             — a non-1 value means the supervisor either never ran a tick (timing bug) or \
             wrote the wrong value (classification bug)."
        );

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    // ── b.e49: the readiness threshold is credential *death*, not the lead ──
    //
    // These two are a pair and must be read together. They sit either side of
    // the one boundary the correction moved: a token that is 30 s from expiry
    // (inside the provider's 60 s default refresh lead, and therefore withheld
    // by the accessor the supervisor used to read) versus one that has actually
    // expired. Before b.e49 both produced NOT_SERVING + gauge 0 — which meant a
    // healthy sidecar on a short refresh lead evicted the replica from all gRPC
    // routing once per token lifetime while it held a perfectly good credential,
    // fleet-wide and correlated across a rolling deploy.
    //
    // Both assert the gauge alongside readiness because the gauge's own
    // documented contract is "1 when … parsed into a non-expired token", which
    // only the post-b.e49 accessor actually satisfies.

    /// Near-expiry but LIVE (30 s out, inside the 60 s lead) → readiness stays
    /// SERVING and the gauge stays at 1.
    ///
    /// This is the headline b.e49 assertion: it fails against any accessor that
    /// treats the lead as a withholding window. Early warning is not readiness's
    /// job — that is what `live_edit_service_token_expires_in_seconds` and the
    /// b.is1 rotation-headroom gauge are for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_inside_refresh_lead_but_unexpired_stays_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let metrics_reader = Arc::clone(&metrics);
        let read_gauge = move || {
            crate::observability::metrics::test_util::sample_value(
                &crate::observability::metrics::test_util::render(&metrics_reader),
                "live_edit_service_token_available",
                &[],
            )
        };

        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("near-expiry.json");
        // 30 s of life left: comfortably inside the provider's 60 s default
        // refresh lead, and comfortably longer than this test runs.
        write_token_file_expiring_in(&token_path, "near-expiry-token", 30).await;

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );
        // Precondition: the accessor the supervisor reads serves the token…
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while provider.cached_bearer_if_unexpired().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "precondition: the provider must cache the 30 s token within 5 s"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // …and the fixture really does sit in the band this test names. Asserted
        // arithmetically on the cached expiry because there is no longer an
        // accessor whose `None` would report it (b.mq8 deleted the withholding
        // one); without this the test would silently stop covering the boundary
        // if the fixture's lifetime ever drifted past the lead.
        let remaining = provider
            .cached_token_expires_on()
            .expect("the provider must have cached the fixture token")
            - chrono::Utc::now();
        assert!(
            remaining > chrono::Duration::zero() && remaining < chrono::Duration::seconds(60),
            "fixture precondition: the token must be unexpired AND inside the 60 s default \
             refresh lead; if it drifts outside, this test silently stops covering the boundary \
             it names (remaining: {remaining})"
        );

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(5),
            )
            .await,
            "a cached token that is 30 s from expiry is a working credential: readiness must \
             rise to SERVING. Pre-b.e49 the supervisor read the withholding accessor and this \
             replica was evicted from all gRPC routing for the last 60 s of every token."
        );

        // Hold it across several more ticks — the point is that this is a
        // steady state, not a single lucky sample.
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                "readiness must STAY serving while the cached token is merely near expiry"
            );
        }

        assert_eq!(
            read_gauge(),
            Some(1.0),
            "live_edit_service_token_available documents itself as `1` when a non-expired \
             token is cached — a near-expiry token is non-expired"
        );

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// The other side of the boundary: an ALREADY-EXPIRED cached token →
    /// readiness flips NOT_SERVING and the gauge goes to 0.
    ///
    /// Pins that "ignore the refresh lead" was not implemented as "ignore
    /// expiry". A dead credential is genuine credential death — the sidecar
    /// has stopped delivering — and the replica should be routed away from.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn service_token_actually_expired_flips_not_serving_and_gauge_zero() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let metrics_reader = Arc::clone(&metrics);
        let read_gauge = move || {
            crate::observability::metrics::test_util::sample_value(
                &crate::observability::metrics::test_util::render(&metrics_reader),
                "live_edit_service_token_available",
                &[],
            )
        };

        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("expired.json");
        // Expired 10 s ago. The file parses fine, so the cache genuinely holds
        // it — `None` below is the expiry check, not the never-read state.
        write_token_file_expiring_in(&token_path, "dead-token", -10).await;

        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while provider.cached_token_expires_on().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "precondition: the provider must have parsed and cached the expired token \
                 within 5 s, or the NOT_SERVING below would prove nothing about expiry"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        // Seed SERVING so the assertion below is a real downward transition
        // (b.rgs registers readiness NOT_SERVING).
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(5),
            )
            .await,
            "an actually-expired cached token is not a credential: readiness must fall to \
             NOT_SERVING on the first tick that observes it"
        );

        let start = std::time::Instant::now();
        let mut gauge_value = read_gauge();
        while start.elapsed() < Duration::from_secs(2) && gauge_value != Some(0.0) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            gauge_value = read_gauge();
        }
        assert_eq!(
            gauge_value,
            Some(0.0),
            "the gauge must read 0 when no non-expired token is cached. Got {gauge_value:?}"
        );

        cancel.cancel();
        token_cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Counter-seed readiness gate tests (Task t2.fgj.mh.2w)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // The supervisor's fifth dependency check consults the
    // `counter_seed_ready: Option<tokio::sync::watch::Receiver<bool>>` field
    // wired by `main.rs` after `seed_object_id_counter` flips its
    // `tokio::sync::watch::channel` from `false` to `true` on success
    // (SR-5.4 / Task `t2.fgj.mh.4u`). The contract:
    //
    //   - `Some(rx)` with `*rx.borrow() == true`  → contributes OK; aggregate
    //                                               governed by other deps.
    //   - `Some(rx)` with `*rx.borrow() == false` → contributes NOT OK; flips
    //                                               readiness immediately (no
    //                                               consecutive-failure smoother
    //                                               — same posture as
    //                                               `service_token_provider`).
    //   - `None`                                  → contribution collapses to a
    //                                               constant `true` and the gate
    //                                               is inert (the
    //                                               checkpoint-disabled wiring).
    //
    // The contract is monotone-rising: once the channel flips `false → true`
    // the supervisor must observe SERVING on its next tick. Tests
    // deliberately do NOT exercise the `true → false` direction — over-pinning
    // that direction would prevent future hardening that wants to latch the
    // gate once it flips OK.
    //
    // Mirrors the service-token tests above structurally — the
    // `Option<watch::Receiver<bool>>` hard-signal-no-smoothing posture is
    // identical, so the timing assertions reuse the same 200 ms / 2 s
    // bounded-wait shape.

    /// `counter_seed_ready: None` (the checkpoint-disabled wiring) MUST NOT
    /// gate readiness — the supervisor's `None` branch collapses the
    /// contribution to a constant `true` and the aggregate stays SERVING
    /// for as long as the other deps stay healthy. This pins the
    /// `None → contributes true → gate inert` half of the classifier.
    ///
    /// Mirrors `supervisor_no_auth_clients_never_probes_auth_backends` and
    /// `service_token_provider_none_does_not_write_gauge_or_flip_readiness`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn readiness_probe_counter_seed_none_does_not_gate_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: None,
                // The gating field this test pins.
                counter_seed_ready: None,
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Since b.rgs readiness is registered NOT_SERVING and raised by the
        // supervisor's first tick; wait for that before sampling.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(5),
            )
            .await,
            "precondition: an inert counter-seed gate must let readiness reach SERVING"
        );

        // Sample readiness across several probe cycles (>= 1.1 s at 200 ms
        // probe_interval). With every other dep configured `None` and only
        // Valkey actually probed, the aggregate must stay SERVING for the
        // whole window — a regression where the `None` branch contributed
        // `false` instead of `true` would flip within the first tick.
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                "counter_seed_ready=None must collapse to a constant true contribution \
                 and keep readiness SERVING (gate inert when wired from the \
                 checkpoint-disabled path)"
            );
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// `counter_seed_ready: Some(rx)` with `*rx.borrow() == false` MUST flip
    /// readiness to `NOT_SERVING` within ONE probe interval — no
    /// consecutive-failure smoother. Pins the `Some(false) → contributes
    /// false → readiness gated` half of the classifier AND the no-smoother
    /// contract on the initial failure.
    ///
    /// Readiness is pre-seeded `SERVING` (see [`preseed_readiness_serving`])
    /// so the assertion is a genuine supervisor-driven transition rather than
    /// a restatement of the `NOT_SERVING` state `create_health_service`
    /// registers since b.rgs.
    ///
    /// Mirrors `service_token_provider_some_with_missing_file_flips_not_serving`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn readiness_probe_counter_seed_some_false_flips_not_serving_immediately() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        // Construct the channel BEFORE the config so the Sender stays alive
        // for the duration of the test — a dropped Sender flips
        // `Receiver::has_changed` semantics and could mask the gate's real
        // behavior with an unrelated channel-closed signal.
        let (tx, rx) = tokio::sync::watch::channel(false);
        let _tx_keepalive = tx;

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: None,
                counter_seed_ready: Some(rx),
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // The counter-seed gate is a hard signal — even though Valkey is
        // probing OK and auth + service-token are `None` (each
        // contributing `true`), the `false` gate must pull readiness from
        // the seeded SERVING down to NOT_SERVING within ONE probe interval.
        // A regression that introduced a consecutive-failure smoother on
        // this signal would fail this bounded wait.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "counter_seed_ready=Some(false) must drive readiness from the seeded SERVING \
             down to NOT_SERVING on the first probe — no consecutive-failure smoothing \
             (hard signal, same posture as service_token_provider)"
        );

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// `counter_seed_ready: Some(rx)` with `*rx.borrow() == true` MUST NOT
    /// gate readiness — the gate contributes `true` and the aggregate is
    /// governed by the other deps. Symmetric counterpart to the
    /// `Some(false)` test above; pins the `Some(true) → contributes true →
    /// readiness governed by other deps` half.
    ///
    /// Mirrors `service_token_provider_some_with_valid_token_stays_serving`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn readiness_probe_counter_seed_some_true_does_not_gate_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        // Pre-flip the channel to `true` before wiring it in — this mirrors
        // the steady-state production case where `seed_object_id_counter`
        // has already succeeded by the time the supervisor's first probe
        // tick lands.
        let (_tx, rx) = tokio::sync::watch::channel(true);

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: None,
                counter_seed_ready: Some(rx),
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Since b.rgs readiness is registered NOT_SERVING and raised by the
        // supervisor's first tick; wait for that before sampling.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(5),
            )
            .await,
            "precondition: counter_seed_ready=Some(true) must let readiness reach SERVING"
        );

        // Sample readiness across several probe cycles — must stay SERVING
        // since the gate contributes `true` and the other deps are healthy.
        // A regression that mis-classified `Some(true)` as `false` would
        // flip within the first probe interval.
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                "counter_seed_ready=Some(true) must contribute OK and keep readiness \
                 SERVING (other deps are healthy)"
            );
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// End-to-end test of the monotone-rising contract: a `Some(false)` gate
    /// that flips to `Some(true)` mid-run MUST be observed by the
    /// supervisor on its next tick, lifting the readiness gate from
    /// NOT_SERVING back to SERVING. Pins the SR-5.4 penultimate
    /// Success-Criteria bullet — the production wiring's success path,
    /// where `seed_object_id_counter` succeeds after the supervisor has
    /// already started ticking and readiness must catch up promptly.
    ///
    /// Deliberately does NOT test the `true → false` direction. The
    /// contract is monotone-rising; testing the reverse over-pins the
    /// supervisor and would prevent future hardening that wants to latch
    /// the gate once it flips OK.
    ///
    /// Mirrors the `false → true` half of
    /// `service_token_none_to_some_to_none_flips_immediately_on_each_transition`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn readiness_probe_counter_seed_false_to_true_transition_restores_serving() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        let (tx, rx) = tokio::sync::watch::channel(false);

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        // Phase 1 below is a downward flip; seed SERVING so it is one rather
        // than a restatement of the b.rgs initial registration. See
        // `preseed_readiness_serving`.
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: None,
                counter_seed_ready: Some(rx),
                api_serving: None,
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Phase 1: gate starts `false` → the supervisor must pull readiness
        // from the seeded SERVING down to NOT_SERVING. This half also
        // establishes the precondition Phase 2 needs: without a real
        // NOT_SERVING verdict here, Phase 2's restore would have nothing to
        // restore from.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "initial counter_seed_ready=Some(false) must drive readiness from the seeded \
             SERVING down to NOT_SERVING within one probe interval"
        );

        // Phase 2: flip the gate `false → true` — simulates
        // `seed_object_id_counter` completing successfully. The supervisor
        // must observe the new value on its next tick and restore SERVING.
        tx.send(true).expect("send true");
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                Duration::from_secs(2),
            )
            .await,
            "flipping counter_seed_ready from false to true must restore readiness \
             SERVING on the FIRST tick that observes the new value — pins the \
             monotone-rising contract (SR-5.4)"
        );

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Full-API-server startup gate tests (b.rgs)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // `api_serving: Option<watch::Receiver<bool>>` is the sixth dependency
    // check and the newest. Its contract is structurally identical to
    // `counter_seed_ready` — hard signal, no smoothing, `None` collapses to a
    // constant `true` — with one addition that nothing else in the supervisor
    // has: the loop `select!`s on `changed()` for this gate, so a `false →
    // true` flip is picked up immediately instead of up to one `probe_interval`
    // later. That matters because the gate opens at the bootstrap → full-API
    // handoff, and every millisecond between "the API is live" and "readiness
    // says so" is a millisecond Kubernetes withholds traffic from a replica
    // that is ready to take it.
    //
    // The two properties the gate exists for, one test each:
    //
    //   - `Some(false)` overrides an otherwise fully healthy dependency set
    //     (it is ANDed, not ORed, into the verdict), and the NOT_SERVING WARN
    //     names it as the cause.
    //   - `false → true` is observed WITHOUT waiting for the next periodic
    //     tick.
    //
    // The `true → false` direction is deliberately untested, matching the
    // counter-seed gate's posture: the gate is monotone-rising in production
    // (`main.rs` opens it once, immediately before serving the full API) and
    // pinning the reverse would block future hardening that wants to latch it.

    /// `api_serving: Some(false)` MUST hold readiness at `NOT_SERVING` even
    /// when every other gated dependency is healthy — Valkey pinging, a real
    /// `ServiceTokenProvider` returning a live bearer, and the counter-seed
    /// gate already open. Pins that the new signal is ANDed into the verdict
    /// rather than, say, ORed or consulted only when something else is
    /// already unhealthy.
    ///
    /// The log assertion is the second half: the NOT_SERVING WARN must carry
    /// `api_serving_ok=false` alongside `valkey_ok=true`,
    /// `service_token_ok=true` and `counter_seed_ok=true`. Operators triage
    /// an un-ready replica from exactly these fields, and the full set is
    /// what distinguishes "still in bootstrap" — benign, self-clearing —
    /// from "Valkey is gone", which is not. A WARN that named no cause, or
    /// named the wrong one, would send that triage down the wrong path.
    ///
    /// **Why the default `current_thread` flavor**:
    /// `tracing::subscriber::set_default` installs the dispatcher in a
    /// thread-local, which only reaches `tokio::spawn`-ed tasks running on
    /// the same OS thread as the test body. A multi-thread runtime would put
    /// the supervisor on a worker thread where the dispatcher is invisible
    /// and the captured buffer would stay empty. Same reasoning as
    /// `src/auth/service_token.rs`'s log-capture tests.
    #[tokio::test]
    async fn readiness_probe_api_serving_some_false_holds_not_serving_and_warn_names_it() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        // Every OTHER dependency healthy, so the only thing that can produce
        // a NOT_SERVING verdict is the gate under test.
        let dir = tempdir().expect("tempdir");
        let token_path = dir.path().join("token.json");
        write_valid_token_file(&token_path, "api-serving-gate-token").await;
        let token_cancel = CancellationToken::new();
        let provider = build_token_provider(
            token_path.clone(),
            TOKEN_POLL_INTERVAL,
            token_cancel.clone(),
        );
        // Wait on the observable fact (cache populated) rather than a blanket
        // sleep: if the supervisor's first tick beat the provider's first
        // poll, `service_token_ok` would be false and the WARN-field
        // assertions below would be pinning the wrong cause.
        let cache_deadline = std::time::Instant::now() + Duration::from_secs(5);
        // Same accessor the supervisor's probe reads (b.e49) — a precondition
        // spelled through the *other* accessor could go true (or stay false) on
        // a different rule than the one under test.
        while provider.cached_bearer_if_unexpired().is_none() {
            assert!(
                std::time::Instant::now() < cache_deadline,
                "precondition: the ServiceTokenProvider must cache the fixture token \
                 within 5s so `service_token_ok` is genuinely true for this test"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // Counter-seed gate already open.
        let (_seed_tx, seed_rx) = tokio::sync::watch::channel(true);
        // The gate under test, held shut. Sender kept alive for the whole
        // test so the receiver never observes a channel-closed condition.
        let (_api_tx, api_rx) = tokio::sync::watch::channel(false);

        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let dispatch_guard = tracing::subscriber::set_default(subscriber);

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        // The assertion below is a downward flip; seed SERVING so it is one
        // rather than a restatement of the b.rgs initial registration.
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: Duration::from_millis(200),
                probe_timeout: Duration::from_millis(50),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: Some(provider),
                counter_seed_ready: Some(seed_rx),
                api_serving: Some(api_rx),
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                Duration::from_secs(2),
            )
            .await,
            "api_serving=Some(false) must drive readiness from the seeded SERVING down to \
             NOT_SERVING on the first probe even though Valkey, the service token and the \
             counter-seed gate are all healthy — the gate is ANDed into the verdict"
        );

        // And must HOLD there. A gate consulted only on the first tick, or
        // one whose contribution decayed to `true` after being observed
        // once, would drift back up here while the full-API server is still
        // not answering.
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(220)).await;
            assert_eq!(
                observe_status(&mut client).await,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                "api_serving=Some(false) must keep readiness NOT_SERVING for as long as \
                 the gate stays shut"
            );
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        token_cancel.cancel();
        let _ = shutdown.send(());
        valkey.quit().await;

        drop(dispatch_guard);

        let buf = captured
            .lock()
            .expect("capture buffer must not be poisoned");
        let output = String::from_utf8_lossy(&buf);

        assert!(
            output.contains("readiness flipped to NOT_SERVING"),
            "the gated-shut verdict must be logged; captured: {output}"
        );
        assert!(
            output.contains("api_serving_ok=false"),
            "the NOT_SERVING WARN must carry `api_serving_ok=false` — it is the field an \
             operator reads to tell `still in bootstrap` (benign, self-clearing) from a \
             real dependency outage. Captured: {output}"
        );
        for healthy_field in [
            "valkey_ok=true",
            "service_token_ok=true",
            "counter_seed_ok=true",
        ] {
            assert!(
                output.contains(healthy_field),
                "the WARN must report `{healthy_field}`, exonerating every dependency but \
                 the api_serving gate — otherwise this test would pass with the verdict \
                 driven by the wrong cause. Captured: {output}"
            );
        }
    }

    /// Flipping `api_serving` `false → true` MUST raise readiness in well
    /// under one `probe_interval` — the property that distinguishes the
    /// supervisor's `select!` wakeup arm on `watch::Receiver::changed()`
    /// from plain next-tick pickup.
    ///
    /// The bound is structural, not a wall-clock performance assertion: the
    /// probe interval is set to 30 s, far longer than any scheduling delay a
    /// loaded runner can produce, so a supervisor that only re-evaluated on
    /// its periodic tick could not satisfy the 5 s deadline no matter how
    /// fast the machine is, and one that wakes on `changed()` satisfies it no
    /// matter how slow. Deleting `wait_for_api_serving_change` from the
    /// `select!` is invisible to every other test in this file — they all use
    /// a short probe interval, so next-tick pickup passes them.
    ///
    /// Shape copied from
    /// `readiness_probe_counter_seed_false_to_true_transition_restores_serving`;
    /// the probe interval and the deadline relationship are the difference.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn readiness_probe_api_serving_false_to_true_wakes_supervisor_before_next_tick() {
        let Some(valkey) = real_valkey().await else {
            eprintln!("SKIP: Valkey not reachable");
            return;
        };
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("test"));

        // Deliberately far longer than the post-flip deadline below. The
        // interval's FIRST tick still fires immediately, so Phase 1 does not
        // wait on it.
        const LONG_PROBE_INTERVAL: Duration = Duration::from_secs(30);
        // Post-flip budget. 6x margin over any plausible scheduling delay,
        // and 6x *under* `LONG_PROBE_INTERVAL` — the gap between those two
        // numbers is what makes the assertion structural.
        const FLIP_DEADLINE: Duration = Duration::from_secs(5);

        let (tx, rx) = tokio::sync::watch::channel(false);

        let (reporter, service) = create_health_service(READINESS_SERVICE).await;
        // Phase 1 is a downward flip; seed SERVING so it is one.
        preseed_readiness_serving(&reporter).await;
        let cancel = CancellationToken::new();
        let supervisor = spawn_health_supervisor(
            reporter.clone(),
            valkey.clone(),
            metrics,
            HealthSupervisorConfig {
                probe_interval: LONG_PROBE_INTERVAL,
                probe_timeout: Duration::from_millis(500),
                readiness_service_name: READINESS_SERVICE.to_string(),
                service_token_provider: None,
                counter_seed_ready: None,
                api_serving: Some(rx),
            },
            cancel.clone(),
        );

        let (addr, shutdown) = spawn_health_server(service).await;
        let mut client = connect_health(addr).await;

        // Phase 1: the gate starts shut, and `tokio::time::interval` fires
        // its first tick immediately, so the supervisor publishes a
        // NOT_SERVING verdict without waiting out the 30 s period. This also
        // establishes that the supervisor is parked in its `select!` — the
        // precondition Phase 2's measurement needs.
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::NotServing,
                FLIP_DEADLINE,
            )
            .await,
            "initial api_serving=Some(false) must drive readiness from the seeded SERVING \
             down to NOT_SERVING on the immediate first tick"
        );

        // Phase 2: open the gate — the `main.rs` handoff moment. The next
        // periodic tick is ~30 s away, so observing SERVING inside
        // `FLIP_DEADLINE` is only possible via the `changed()` wakeup arm.
        tx.send(true).expect("send true");
        assert!(
            wait_for_status(
                &mut client,
                tonic_health::pb::health_check_response::ServingStatus::Serving,
                FLIP_DEADLINE,
            )
            .await,
            "opening the api_serving gate must wake the supervisor immediately via its \
             `changed()` select arm: with probe_interval={LONG_PROBE_INTERVAL:?} the next \
             periodic tick could not have landed inside the {FLIP_DEADLINE:?} budget, so \
             a supervisor relying on next-tick pickup would leave readiness NOT_SERVING \
             for ~30 s after the full-API server took over the socket"
        );

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
        let _ = shutdown.send(());
        valkey.quit().await;
    }

    /// A `MakeWriter` that captures everything written to it into a shared
    /// buffer. Mirrors the `CapturingWriter` pattern in
    /// `src/auth/service_token.rs::tests` — duplicated rather than shared
    /// because it lives inside a `#[cfg(test)]` module of a sibling
    /// production module.
    #[derive(Clone)]
    struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl CapturingWriter {
        fn new() -> (Self, Arc<std::sync::Mutex<Vec<u8>>>) {
            let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
            (Self(buf.clone()), buf)
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer must not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
