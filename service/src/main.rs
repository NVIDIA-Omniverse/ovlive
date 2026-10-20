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

// jemalloc global allocator with its retention options compiled in (b.u5k).
// Bin-only on purpose; see the module header for why not `lib.rs`, and the
// `Cargo.toml` target table for why linux-gnu only.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod allocator;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use aws_credential_types::provider::SharedCredentialsProvider;
use dashmap::DashMap;
use fred::interfaces::SetsInterface;
use rand::SeedableRng;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tonic_reflection::server::Builder as ReflectionBuilder;

use live_edit::auth::cache::AuthCache;
use live_edit::auth::service_token::{log_service_token_startup_warnings, ServiceTokenProvider};
use live_edit::auth::{
    build_permissions_client, build_userinfo_client, InstrumentedPermissionsClient,
    InstrumentedUserinfoClient,
};
use live_edit::checkpoint::{
    build_checkpoint_store, register_checkpoint_pacing_gauges_if_enabled, spawn_checkpoint_loop,
    ext_mirror_channel_for_plan, spawn_ext_backup_repair, spawn_ext_mirror_workers,
    CheckpointAdapter, CheckpointLoopConfig, CheckpointStore, ExtBackupRepair, ExtMirrorReceiver,
    ExtMirrorSender, BaseBootstrap, RecoveryHook, SessionRecoveryHook, TailCaptureConfig,
    LAYER_PUBLISH_ARTIFACTS,
};
use live_edit::config::{AuthProtocol, Config, TracingSettings, ValkeyAuthMode};
use live_edit::grpc::capabilities::CapabilitiesServiceImpl;
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::FILE_DESCRIPTOR_SET;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::graceful_shutdown::{GracefulShutdown, StreamPacing};
use live_edit::grpc::transport::{
    drain_stream_tasks, end_streams_after_drain, live_service_server, minimum_termination_grace,
    production_server_builder, serve_with_post_goaway_bound, POST_BUDGET_FLUSH_BOUND,
};
use live_edit::observability::metrics::Metrics;
use live_edit::observability::metrics_server::serve_metrics_with_scaling;
use live_edit::observability::scaling::{ScalingEndpoint, ServiceScalingSource};
use live_edit::observability::trace_export::TraceExportState;
use live_edit::observability::startup::closable_incoming;
use live_edit::observability::{init_tracing, shutdown_tracing};
use live_edit::storage::client::{ValkeyClient, VALKEY_TOKEN_ROTATION_FANOUT_CEILING};
use live_edit::storage::diff_read::{
    run_snapshot_read_self_test, SnapshotSelfTestFailure, SNAPSHOT_SELF_TEST_REMEDIATION,
};
use live_edit::storage::ext_backup::ExtBackupMarks;
use live_edit::storage::external_values::{spawn_gc_task, ValkeyExternalValueStore};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_flights::SnapshotFlights;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::CapabilitiesServiceServer;

#[tokio::main]
async fn main() {
    // The three logging variables first, so the subscriber exists before the
    // rest of the configuration loads: a line logged with no subscriber is
    // silently dropped, and `Config::from_env` logs its startup WARNs while it
    // parses. Configuration errors go to stderr, never through the
    // subscriber, so a log filter cannot hide them.
    let tracing_settings = TracingSettings::from_env().unwrap_or_else(|e| {
        eprintln!("configuration error: {e}");
        std::process::exit(1);
    });

    // `Some` only when `OTEL_EXPORTER_OTLP_ENDPOINT` enabled trace export: the
    // pipeline's loss accounting, registered as metrics in `run`, and the
    // transport the endpoint's scheme selected. An endpoint that is not an
    // `http://` or `https://` URL, or a trace exporter that cannot be built
    // for it, is a configuration error, reported here before any subscriber
    // exists and so in `scaler` mode too.
    let trace_export = init_tracing(
        &tracing_settings.log_level,
        tracing_settings.log_format,
        &tracing_settings.trace_filter,
    )
    .unwrap_or_else(|e| {
        eprintln!("configuration error: {e}");
        std::process::exit(1);
    });

    // `live-edit scaler`: the replica autoscaler, a mode of this same binary
    // (one image, one artifact, one set of build gates). Dispatched before the
    // service configuration loads, so the scaler needs none of the service's
    // variables and reads only `LIVE_EDIT_SCALER_*`. A subcommand rather than
    // an environment variable because server pods load the chart's settings
    // ConfigMap into their environment wholesale: a mode carried there could
    // turn a server pod into a scaler.
    if std::env::args().nth(1).as_deref() == Some("scaler") {
        let code = live_edit_scaler::main_from_env().await;
        shutdown_tracing().await;
        std::process::exit(code);
    }

    let config = Config::from_env().unwrap_or_else(|e| {
        eprintln!("configuration error: {e}");
        std::process::exit(1);
    });

    // First line after configuration loads (only WARNs `from_env` itself
    // logged can precede it): if the test-only override let the
    // no-token-validation shape through `from_env`, say so before every
    // dependency step and INFO line so it cannot be missed in a log
    // scrollback.
    config.log_test_only_auth_override_startup_warning();

    // The service-identity startup WARN, likewise ahead of every dependency
    // step. A helper so integration tests can drive it with a synthetic
    // `Config`; this is the production call site.
    log_service_token_startup_warnings(&config);

    // The pod memory plan `Config::from_env` derived from the container's
    // memory limit: the limit and its source, every budget, the line and the
    // margin, and the smallest limit this configuration closes at (WARN when
    // no limit was readable on Linux). Logged here, ahead of every dependency
    // step, so a pod that never reaches its store or binds its port still
    // shows what it was sized for.
    config.pod_memory.log_startup();

    // The snapshot fan-out's dedicated jemalloc arena, created and its
    // thread start hook registered here, before `run` connects the Valkey
    // client and so before any fan-out thread starts (`allocator`, *The
    // fan-out arena*). Then the allocator options the process actually runs
    // with, the arena among them, WARN if an `_RJEM_MALLOC_CONF` override, or
    // a host THP state jemalloc cannot detect, defeats memory return (b.u5k),
    // or if the arena could not be set up.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    allocator::log_effective_options(allocator::install_fanout_arena());

    // The trace export settings in effect. Deliberately not the endpoint.
    if let Some((state, transport)) = &trace_export {
        live_edit::observability::trace_export::log_trace_export_enabled(
            &config.trace_filter,
            state,
            *transport,
        );
    }

    tracing::info!("live-edit starting");

    run(config, trace_export.map(|(state, _)| state)).await;
}

async fn run(config: Config, trace_export: Option<Arc<TraceExportState>>) {
    // The start the `/scaling/v1` snapshot reports uptime from: the autoscaler
    // ignores a replica's CPU readings while it is still warming up.
    let process_started = std::time::Instant::now();

    // This process's identity in every leader lock (GC, orphan sweep,
    // ref_count reconciler, external-value backup repair, checkpoint loop,
    // presence sweeper). Built ONCE and
    // cloned into each task: the four `le:gc:leader` consumers ride one
    // leadership through the lock's refresh branch, which a per-task instance
    // would break. `service_label` stays the metrics `service` label; only the
    // lock value carries the per-process instance.
    let leader_identity = LeaderIdentity::for_process(config.service_label.clone());

    let grpc_addr: SocketAddr = format!("0.0.0.0:{}", config.port)
        .parse()
        .expect("invalid gRPC bind address");

    let metrics_addr: SocketAddr = format!("0.0.0.0:{}", config.metrics_port)
        .parse()
        .expect("invalid metrics bind address");

    // Build the metrics now so the auth layer can instrument backend calls and
    // record cache hit/miss stats from the first request onward (b.nrt item 4).
    // `Metrics::try_new` builds the dual-reader OpenTelemetry meter provider
    // (OMPE-98174): it owns the always-attached Prometheus scrape exporter (the
    // `/metrics` server clones it out via `metrics.prometheus_exporter()`
    // below), and — when `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` is set — an OTLP
    // push reader is attached alongside it. An endpoint that is not an
    // `http://` or `https://` URL, or a push exporter that cannot be built for
    // it, is a configuration error: reported on stderr, exit 1, before any
    // port is bound.
    //
    // The `live_edit_service_token_available` gauge is an unconditional
    // instrument (Task t2.fgj.dp.s7): because an OTel gauge emits no
    // Prometheus series until first `record()`, and the health supervisor
    // only records it when a `ServiceTokenProvider` exists, the family stays
    // absent from `/metrics` on replicas where `LIVE_EDIT_SERVICE_TOKEN_FILE`
    // is unset — same observable contract as before, no explicit gating.
    //
    // Constructed here — ahead of `ValkeyClient::connect` — because the
    // Redis-scoped access-token-file `ServiceTokenProvider` built just below
    // needs `Metrics`, and its handle is threaded into `connect` (Epic 3
    // t3.tr5.jv). The storage-api provider at Step 1b-v is a *different*
    // instance (different token audience) and stays untouched.
    let metrics = Arc::new(Metrics::try_new(&config.service_label).unwrap_or_else(|e| {
        eprintln!("configuration error: {e}");
        std::process::exit(1);
    }));

    // The pod memory plan's gauges, as soon as the registry exists: the limit
    // by source and the budgets dashboards read against.
    metrics.record_pod_memory_plan(&config.pod_memory);

    // Process and allocator memory gauges (b.u5k). Registered here, before the
    // metrics server spawns at Step 0a-bis, so they are on the first scrape:
    // whether memory returns to the OS after load is only visible as a series
    // that covers the whole process lifetime. The jemalloc pair exists only
    // where jemalloc is the global allocator.
    metrics.register_process_memory_gauge(
        live_edit::observability::process_memory::resident_memory_bytes,
    );
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    metrics.register_allocator_memory_gauges(
        &config.service_label,
        allocator::allocated_bytes,
        allocator::resident_bytes,
    );
    // The allocator's unpurged dirty pages, split between the fan-out arena
    // and the rest (b.fsz d1): the slack the arenas hold beyond the live heap.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    metrics.register_allocator_dirty_gauge(&config.service_label, allocator::dirty_bytes);

    // Trace-export counters, only when trace export is enabled: their
    // presence on `/metrics` is itself that signal.
    if let Some(state) = trace_export {
        metrics.register_trace_export_counters(state);
    }

    // ── Step 0a: Early gRPC bind + bootstrap health server (b.rgs) ───────────
    //
    // The liveness probe targets the gRPC health service's overall (`""`)
    // status. Before this block the socket did not exist until every
    // dependency below had been wired, so the kubelet's liveness probe saw a
    // closed port — reported as `failed to connect service ':50001'` — from
    // t=0, and any startup slower than the liveness window (checkpoint
    // recovery, cold Valkey, DNS, TLS, a credential sidecar race) risked a
    // kill+restart loop even though the process was healthy and progressing.
    //
    // `bind_grpc_listener` binds ONE socket and dups the descriptor into two
    // `TcpIncoming` streams. The bootstrap server below serves the first one
    // — health + reflection + Discovery capabilities, everything that needs
    // no `ValkeyClient` — while the wiring runs; Step 8 drains it and serves
    // the full API over the second. Both streams share one open file
    // description, so the socket never leaves `LISTEN` across the handoff:
    // connections queue in the shared accept backlog rather than being
    // refused. tonic cannot add services to a running `Server`, which is why
    // this is two servers over one socket rather than a reordering.
    //
    // Binding here is also the fail-fast point for `EADDRINUSE`: the process
    // exits immediately instead of ~900 lines later, after every dependency
    // has been connected.

    // The readiness service name is the protobuf fully-qualified name of the
    // LiveService. Passed as a &str so `observability::health` does not
    // depend on concrete `grpc::proto` types (b.3z7 medium).
    const READINESS_SERVICE_NAME: &str = "nvidia.omniverse.live.v1alpha.LiveService";

    // `TCP_KEEPALIVE` for accepted gRPC connections. Applied to the incoming
    // streams rather than the `Server` builder because
    // `serve_with_incoming*` ignores the builder's TCP settings; the value
    // matches the `.tcp_keepalive(...)` the bind-based path used before.
    const GRPC_TCP_KEEPALIVE: Duration = Duration::from_secs(30);

    // Grace for the bootstrap server to finish in-flight requests at handoff.
    // Its only clients are probe/discovery traffic — a kubelet gRPC health
    // check, a Discovery `ListServices` scan, an operator's `grpcurl` — all
    // sub-second unary calls, so 5 s is roughly two orders of magnitude of
    // headroom while staying far inside the probe cadences. On elapse the
    // task is aborted rather than waited on: the second descriptor still
    // holds the socket, so an abort costs at most one probe retry and can
    // never strand the port.
    const BOOTSTRAP_HANDOFF_BUDGET: Duration = Duration::from_secs(5);

    let (bootstrap_incoming, serving_incoming, bound_grpc_addr) =
        live_edit::observability::startup::bind_grpc_listener(grpc_addr, GRPC_TCP_KEEPALIVE)
            .unwrap_or_else(|e| {
                // `grpc_addr` is the *requested* address — the bind failed, so
                // there is no bound address to report. Logged in preference to
                // `config.port`, which is `0` under `LIVE_EDIT_PORT=0`.
                tracing::error!(
                    error = %e,
                    requested_addr = %grpc_addr,
                    "failed to bind gRPC listener"
                );
                std::process::exit(1);
            });

    // The health reporter is built here — ahead of every dependency — so the
    // bootstrap server can answer liveness immediately. `create_health_service`
    // registers `""` as SERVING and the readiness name as NOT_SERVING; the
    // health supervisor (Step 4) raises readiness once Valkey, the
    // service-identity token, and the counter-seed gate all report healthy.
    let (health_reporter, bootstrap_health_service) =
        live_edit::observability::health::create_health_service(READINESS_SERVICE_NAME).await;

    // The Discovery CapabilitiesService (b.hv2) is marked SERVING statically
    // and never flipped: its response is a compile-time constant with no
    // Valkey / service-token / counter-seed dependency, so gating it on the
    // health supervisor's readiness verdict would only hide the service from
    // discovery scans during transient dependency blips (the operator retains
    // prior state on failure). Matches the fleet-wide capabilities reference
    // pattern, which also sets it serving once at startup.
    health_reporter
        .set_serving::<CapabilitiesServiceServer<CapabilitiesServiceImpl>>()
        .await;

    // Built here (rather than at Step 6b) so the bootstrap server can carry a
    // clone: probe and discovery traffic served during startup is measured
    // and deadline-enforced exactly like steady-state traffic.
    let metrics_layer = MetricsLayer::new(
        Arc::clone(&metrics),
        config.service_label.clone(),
        config.rpc_deadline_ms,
        config.session_list_timeout_ms,
    );

    // A second reflection instance for the bootstrap server. The service is
    // stateless (it owns the encoded descriptor sets), so constructing a
    // second one is equivalent to sharing one — and it is not `Clone`.
    let bootstrap_reflection_service = ReflectionBuilder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(
            live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::FILE_DESCRIPTOR_SET,
        )
        .build_v1()
        .expect("failed to build bootstrap reflection service");

    let bootstrap_shutdown = CancellationToken::new();
    let mut bootstrap_server = {
        let shutdown = bootstrap_shutdown.clone();
        tokio::spawn(
            production_server_builder()
                .layer(metrics_layer.clone())
                .add_service(bootstrap_health_service)
                .add_service(bootstrap_reflection_service)
                // Bare `.add_service`, structurally exempt from the auth
                // interceptor exactly as in the full-API server (SDD §5.7).
                .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
                .serve_with_incoming_shutdown(bootstrap_incoming, async move {
                    shutdown.cancelled().await;
                }),
        )
    };
    tracing::info!(
        addr = %bound_grpc_addr,
        "gRPC listener bound (bootstrap: health + reflection + capabilities; readiness NOT_SERVING)"
    );

    // The full-API-server readiness gate. Created here — before anything that
    // could wedge — and flipped to `true` immediately before
    // `serve_with_incoming_shutdown` at Step 8, so readiness can never
    // advertise `SERVING` while only the bootstrap server (health +
    // reflection + capabilities, no `LiveService`) is answering on the socket.
    //
    // Without it the supervisor spawned at Step 4 could publish `SERVING` on
    // its first tick — which fires immediately — and every LiveService RPC
    // routed in the Step 5..7 window, including connections that survive into
    // the bootstrap handoff drain, would terminate `UNIMPLEMENTED`. Worse, a
    // boot that wedges after Step 4 would leave the pod Ready, live, and
    // startupProbe-passed indefinitely with no restart path — strictly worse
    // than the closed socket this early-bind replaced.
    //
    // The `Sender` stays bound in `main`'s scope for the whole process
    // lifetime; the shutdown sequence does not touch it (Step 7 writes
    // `NOT_SERVING` to the reporter directly, and the supervisor is
    // edge-triggered on its own verdict, so it never contradicts that write).
    let (api_serving_tx, api_serving_rx) = tokio::sync::watch::channel(false);

    // ── Step 0a-bis: Metrics HTTP server (SDD §5.5 step 8) ───────────────────
    //
    // Moved ahead of the dependency wiring alongside the gRPC bind (b.rgs):
    // a slow boot must stay diagnosable, and `Metrics` is already
    // constructed. The `live_edit_storage_watch_unwatch_inflight` gauge needs
    // a `ValkeyClient` and so stays registered at its original site below;
    // a gauge registered later is simply absent from earlier scrapes, which
    // is the existing pattern for every dependency-scoped instrument.
    let metrics_listener = TcpListener::bind(metrics_addr).await.unwrap_or_else(|e| {
        tracing::error!(error = %e, port = config.metrics_port, "failed to bind metrics port");
        std::process::exit(1);
    });
    tracing::info!(port = config.metrics_port, "metrics server listening");
    // The autoscaler's `/scaling/v1` route rides on the same server. It
    // answers 503 until Step 6 registers its source, once the admission gate,
    // the health reporter and the service shutdown token all exist.
    let scaling_endpoint = Arc::new(ScalingEndpoint::new());
    let metrics_task = tokio::spawn(serve_metrics_with_scaling(
        metrics_listener,
        metrics.prometheus_exporter(),
        Arc::clone(&scaling_endpoint),
    ));

    // ── Step 0b: Redis-scoped access-token-file provider (Epic 3 t3.tr5.jv) ───
    //
    // A SECOND, independent `ServiceTokenProvider` — distinct from the
    // storage-api provider at Step 1b-v (different token audience) — pointed at
    // the Redis-scoped token file. It only exists in `auth_mode=accessTokenFile`;
    // every other mode passes `None` into `connect` and keeps the pre-auth-mode
    // behavior byte-for-byte. The access-token-file arm of
    // `apply_fred_config_overrides` wraps it in a `TokenFileCredentialProvider`
    // and installs fred's `credential_provider` (SR-4.1 / SR-4.2).
    //
    // Refresh lead: Azure's guidance is to FETCH a fresh token ≥ 3 min before
    // expiry, so `valkey_token_refresh_lead_secs` (default 180 s ⇒ ≥ 3 min) is
    // passed through the lead-configurable constructor to drive the provider's
    // proactive re-read window — the polling task re-reads at its fast cadence
    // over that run-up so the replacement is cached before the old token dies.
    // It is NOT a withholding window on this path: the
    // `TokenFileCredentialProvider` reads `cached_bearer_if_unexpired()`, so a
    // cached token authenticates for its whole real lifetime (b.d6b).
    // `validate_valkey_auth_mode` (Epic 2) already rejects access-token-file
    // mode with a missing token-file path at startup, so no path validation is
    // duplicated here.
    //
    // Cancellation is fired in the shutdown block alongside the storage-api
    // provider's `service_token_cancel` so the polling task exits within the
    // shutdown budget.
    let valkey_token_cancel = CancellationToken::new();
    let valkey_token_provider: Option<Arc<ServiceTokenProvider>> =
        match (config.valkey_auth_mode, &config.valkey_token_file) {
            (ValkeyAuthMode::AccessTokenFile, Some(path)) => {
                // Prolongation floor: `VALKEY_TOKEN_ROTATION_FANOUT_CEILING`
                // (180 s), taken whole from its single definition site in
                // `storage::client` rather than re-derived here. It is the
                // worst-case latency from "the polling task caches a rotated
                // token" to "every pooled connection has re-AUTHed with it" —
                // two provider-tick intervals under b.7td's rotation gate, one
                // for the first client to observe the rotation and one for the
                // last client to take its next tick inside the fan-out window.
                // A rotation picked up with less headroom than this before the
                // outgoing token's expiry cannot be guaranteed to reach every
                // live connection, so the stragglers get dropped and
                // re-established at expiry instead of prolonged. The provider
                // observes the real headroom per rotation and WARNs under this
                // bound (b.is1). Passed in from here — the wiring layer —
                // rather than read inside `auth`, so `auth` gains no dependency
                // edge on `storage`.
                let prolongation_floor = chrono::Duration::from_std(
                    VALKEY_TOKEN_ROTATION_FANOUT_CEILING,
                )
                .expect("rotation fan-out ceiling (180 s) is representable as a chrono::Duration");
                let provider = ServiceTokenProvider::new_with_refresh_lead_and_role(
                    path.clone(),
                    Duration::from_secs(config.service_token_poll_interval_secs),
                    // `try_seconds`, not `seconds`: the latter panics past
                    // chrono's `i64::MAX / 1_000` second bound, and the `as i64`
                    // cast would sign-wrap a `u64` at or above `2^63` into a
                    // negative lead. `Config::from_env` rejects both ranges, so
                    // this cannot fail — the `expect` documents that contract
                    // rather than defending against it (b.is1).
                    chrono::Duration::try_seconds(config.valkey_token_refresh_lead_secs as i64)
                        .expect(
                            "LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS is validated for \
                             chrono representability by Config::from_env",
                        ),
                    Some(prolongation_floor),
                    ServiceTokenProvider::ROLE_VALKEY,
                    Arc::clone(&metrics),
                    valkey_token_cancel.clone(),
                );
                tracing::info!(
                    path = %path.display(),
                    poll_interval_secs = config.service_token_poll_interval_secs,
                    refresh_lead_secs = config.valkey_token_refresh_lead_secs,
                    "redis-scoped access-token-file provider initialized"
                );
                Some(Arc::new(provider))
            }
            _ => None,
        };

    // ── Step 0c: IAM AWS credential source (Epic 4 t3.tr5.we.6w) ──────────────
    //
    // Resolve the AWS credential source for `auth_mode=iam` at startup: the
    // standard AWS provider chain (IRSA in production — an STS
    // AssumeRoleWithWebIdentity round-trip the chain resolves and caches),
    // which needs the tokio runtime this async fn owns. It only exists in
    // `auth_mode=iam`; every other mode passes `None` into `connect` and keeps
    // the pre-auth-mode behavior byte-for-byte. The iam arm of
    // `apply_fred_config_overrides` wraps it in an `IamCredentialProvider` and
    // installs fred's `credential_provider` (SDD §15.5). This is the AWS analog
    // of the Redis-scoped access-token-file `ServiceTokenProvider` built at Step 0b — but it
    // reads no token file (the credential chain owns credential resolution) and
    // needs no `Metrics` / `CancellationToken` (fred owns the refresh task).
    //
    // The credential chain's STS HTTPS connector is pinned to the `ring` rustls
    // crypto provider: Cargo.toml drops aws-config's default `aws-lc-rs` HTTPS
    // client and declares `aws-smithy-http-client` with only `rustls-ring`, so
    // the whole graph shares one rustls `CryptoProvider`. A second provider
    // makes `CryptoProvider::from_crate_features()` return `None` and panic
    // fred's Valkey-TLS startup path.
    //
    // `validate_valkey_auth_mode` (Epic 2) already rejects iam mode with a
    // missing region / username / cluster-host at startup, so the region
    // `expect` below is upheld by that gate.
    let valkey_iam_credentials_provider: Option<SharedCredentialsProvider> =
        match config.valkey_auth_mode {
            ValkeyAuthMode::Iam => {
                let region = config.valkey_iam_region.clone().expect(
                    "validate_valkey_auth_mode guarantees LIVE_EDIT_VALKEY_IAM_REGION \
                     is set in iam auth mode",
                );
                let http_client = aws_smithy_http_client::Builder::new()
                    .tls_provider(aws_smithy_http_client::tls::Provider::Rustls(
                        aws_smithy_http_client::tls::rustls_provider::CryptoMode::Ring,
                    ))
                    .build_https();
                let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
                    .region(aws_config::Region::new(region))
                    .http_client(http_client)
                    .load()
                    .await;
                let provider = sdk_config.credentials_provider().unwrap_or_else(|| {
                    tracing::error!(
                        "iam Valkey auth mode: the AWS default credential chain resolved \
                         no credentials provider (no IRSA / environment / profile source \
                         available); cannot AUTH to Valkey"
                    );
                    std::process::exit(1);
                });
                tracing::info!(
                    service = ?config.valkey_iam_service,
                    "iam Valkey AWS credential source initialized"
                );
                Some(provider)
            }
            _ => None,
        };

    // ── Step 1: Connect to Valkey (SDD §5.5 step 2) ──────────────────────────
    let valkey = ValkeyClient::connect(
        &config,
        valkey_token_provider,
        valkey_iam_credentials_provider,
    )
    .await
    .unwrap_or_else(|e| {
        tracing::error!(error = %e, "failed to connect to Valkey");
        std::process::exit(1);
    });

    // Wire fred's reconnect / error event streams into the metrics pipeline so
    // operators have a first-class "Valkey in trouble" signal
    // (b.3z7 H-obs-3). Must run after `Metrics::new` but before any gRPC
    // traffic so we don't miss the first reconnect/error cycle under a
    // flapping backend.
    valkey
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    // ── Step 1a: Atomic snapshot-read self-test (Spec SDD SR-1.8) ───────────
    //
    // The atomic snapshot read declares only an object's header key and reads
    // the object's other, same-slot keys undeclared, and so does the per-window
    // script of the structural `UpdateLayer` WATCH read
    // (`fanout_window_read.lua`). A store that refuses that, or refuses
    // scripts at all, would fail every catch-up and checkpoint read and every
    // structural edit that reads through the WATCH path at runtime, so it is
    // checked once here, before readiness, and a refusal stops startup.
    //
    // One attempt, bounded as a whole by the per-command timeout and with no
    // retry of its own, so it adds at most one LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS
    // to the boot (`connect` above already proved the store reachable). Only a
    // refusal is a verdict on the store. Everything `Unavailable` covers — no
    // reply in time, a lost connection or other transient IO, the store
    // loading, busy or read-only (`-LOADING` while a restarted primary replays
    // its data), the cluster down or the probe's slot not served or migrating
    // — is logged with its `detail` and startup continues, rather than exiting
    // into a crash loop or growing a second connect-sized retry loop.
    match run_snapshot_read_self_test(
        &valkey,
        &KeySchema::default(),
        Duration::from_millis(config.valkey_command_timeout_ms),
    )
    .await
    {
        Ok(()) => tracing::info!("atomic snapshot-read self-test passed"),
        Err(SnapshotSelfTestFailure::UndeclaredKeyRefused(detail)) => {
            tracing::error!(
                detail = %detail,
                remediation = SNAPSHOT_SELF_TEST_REMEDIATION,
                "startup refused: the backing store refused the atomic snapshot read's \
                 undeclared same-slot key read, which the structural UpdateLayer read's \
                 per-window script relies on too"
            );
            std::process::exit(1);
        }
        Err(SnapshotSelfTestFailure::ScriptRefused(detail)) => {
            tracing::error!(
                detail = %detail,
                "startup refused: the backing store refused the atomic snapshot read script \
                 (object_snapshot_read.lua); every catch-up and checkpoint read, and the \
                 structural UpdateLayer read's per-window script (fanout_window_read.lua), would \
                 fail the same way"
            );
            std::process::exit(1);
        }
        Err(SnapshotSelfTestFailure::Unavailable(detail)) => tracing::warn!(
            detail = %detail,
            "atomic snapshot-read self-test could not run, for a reason that is not a verdict \
             on the store (no reply in time, a lost connection or other transient IO, the store \
             loading, busy or read-only, the cluster down or the probe's slot not served or \
             migrating; detail says which); continuing startup without it. A store that refuses \
             undeclared same-slot keys inside scripts, or scripts at all, will fail catch-up and \
             checkpoint reads and the structural UpdateLayer read at runtime"
        ),
    }

    // ── Step 1b: Session store and auth clients ──────────────────────────────
    //
    // `SessionStore` and `LayerStore` are constructed below — AFTER the
    // checkpoint store + `SessionCheckpointWriter` so each store can be
    // chained with `.with_checkpoint_writer(...)` against the shared
    // dispatcher (Task `t2.fgj.7v.p9`). The reorder is mechanical: both
    // stores depend only on `valkey` + `key_schema` + `config` + `metrics`,
    // all of which are already available above. Nothing between this
    // comment and the construction site below references `session_store`
    // / `layer_store`.
    let key_schema = KeySchema::default();

    // b.pa1: from here on, a store write refused `-OOM` anywhere in this
    // process requests the GC's out-of-memory collection under this replica's
    // label (`storage::store_oom`); before this, a note only wakes the GC loop.
    valkey
        .store_oom_signal()
        .arm(&key_schema, leader_identity.service_label.clone());

    // ── Step 1b-pre: External-value checkpoint mirror channel (SDD §22.17) ─
    //
    // Construct the bounded mpsc channel, its byte budget and the
    // producer-side `ExtMirrorSender` here so the sender can be chained into
    // `ValkeyExternalValueStore` immediately below. The worker pool is
    // spawned later (Step 1e-bis) once the checkpoint store is built; the
    // receiver half is held in `mirror_rx_for_workers` until then.
    //
    // The wiring is `ext_mirror_channel_for_plan`, in the library so its tests
    // pin it: the gate is "the pod memory plan funds the mirror". The plan's
    // `ext_mirror_bytes()` is `Some(M_ext)` exactly when checkpointing is on
    // and the adapter is not `Disabled` (`Config::checkpoint_adapter_wired`),
    // and that budget is what the mirror is built with. The condition matches
    // the checkpoint loop's at Step 1e, which tests `checkpoint_enabled` alone
    // because `validate_durability_mode` refuses enabled with a `Disabled`
    // adapter. The recovery store at Step 1b-iii.4 deliberately tests the
    // adapter alone, independent of `LIVE_EDIT_CHECKPOINT_ENABLED`, so with
    // checkpointing off and an adapter set, recovery is wired and the mirror
    // is not. The not-wired arm logs nothing here (the checkpoint-loop block
    // emits the operator-facing WARN).
    // When the gate does not fire, `mirror_sender` stays `None` so the
    // hot-path send branch in `ValkeyExternalValueStore::store_values`
    // is a complete no-op (Subtask body Acceptance Criterion 1).
    let (mirror_sender, mirror_rx_for_workers): (
        Option<ExtMirrorSender>,
        Option<ExtMirrorReceiver>,
    ) = ext_mirror_channel_for_plan(
        &config.pod_memory,
        config.checkpoint_ext_mirror_queue_capacity,
        metrics.clone(),
    )
    .unzip();
    // The backup repair (Step 1e-bis) re-sends through the same byte budget,
    // so it holds a handle on the same sender; `Some` exactly when the
    // mirror is built.
    let repair_mirror_sender: Option<ExtMirrorSender> = mirror_sender.clone();

    // `ext_value_store` construction moved below to Step 1b-iii.7 so it can
    // chain `.with_checkpoint_store(...)` against the freshly-built
    // `Arc<dyn CheckpointStore>` (SDD SR-3.2; Subtask t3.fgj.zc.p6.78). The
    // checkpoint-store construction at Step 1b-iii also depends on the
    // service-token provider at Step 1b-v, so the dependency chain is now:
    //   mirror_sender (Step 1b-pre)
    //     → service_token_provider (Step 1b-v)
    //       → checkpoint_store (Step 1b-iii)
    //         → ext_value_store (Step 1b-iii.7, below)
    // The startup-recovery GC drain at Step 1c was tightly coupled to
    // `ext_value_store`, so it moved alongside the construction to keep
    // ordering correct.
    let session_tokens = Arc::new(DashMap::new());

    // ── Step 1b-v: Service-identity token provider (Epic t1.fgj.dp) ──────
    //
    // Constructed only when `LIVE_EDIT_SERVICE_TOKEN_FILE` is set. The
    // resulting handle is threaded onto `LiveEditService` so Epic 5's
    // outbound storage-api adapter can read the cached Bearer token, and
    // (Epic t1.fgj.jd) into `build_checkpoint_store` below so the
    // `StorageApi` checkpoint-store arm can own a clone for outbound auth.
    // When unset, `service_token_provider` stays `None`. That never reaches
    // the `StorageApi` adapter: `Config::from_env` already refused that
    // adapter without `LIVE_EDIT_SERVICE_TOKEN_FILE`, and
    // `build_checkpoint_store` refuses to wire it without a provider as
    // defense in depth (see the dispatcher's StorageApi arm).
    // Cancellation is plumbed alongside the other shutdown-block cancels
    // so the polling task exits inside the shutdown budget.
    //
    // Ordering note: this block sits ahead of Step 1b-iii (checkpoint
    // store) and Step 1b-ii (auth backend clients) because
    // `build_checkpoint_store` needs the provider handle as an argument.
    // The block is otherwise self-contained — it depends only on
    // `config.service_token_file`, `config.service_token_poll_interval_secs`,
    // and `metrics` (already built above), so the reorder is safe.
    let service_token_cancel = CancellationToken::new();
    let service_token_provider: Option<Arc<ServiceTokenProvider>> = match &config.service_token_file
    {
        Some(path) => {
            let provider = ServiceTokenProvider::new_with_role(
                path.clone(),
                Duration::from_secs(config.service_token_poll_interval_secs),
                ServiceTokenProvider::ROLE_STORAGE_API,
                Arc::clone(&metrics),
                service_token_cancel.clone(),
            );
            tracing::info!(
                path = %path.display(),
                poll_interval_secs = config.service_token_poll_interval_secs,
                "service token provider initialized"
            );
            Some(Arc::new(provider))
        }
        None => None,
    };

    // ── Step 1b-iii: Checkpoint store (SDD SR-1.1) ────────────────────────
    //
    // Construct the adapter selected by `LIVE_EDIT_CHECKPOINT_ADAPTER`
    // unconditionally — even when `LIVE_EDIT_CHECKPOINT_ENABLED=false` we
    // still hand out the binding (a no-op store in `Disabled` mode) so the
    // periodic checkpoint loop and recovery hook (later subtasks in Epic
    // t1.fgj.ya) thread the trait object through without conditional types.
    // The loop spawn site is what gates on `checkpoint_enabled`; this is
    // pure construction.
    //
    // The `service_token_provider` clone fed in here is in addition to the
    // clone fed to `LiveEditService` later. The `StorageApi` arm of the
    // factory requires it (and always gets it here, since `Config::from_env`
    // refuses that adapter without `LIVE_EDIT_SERVICE_TOKEN_FILE`); the
    // `Disabled` / `InMemory` / `LocalDisk` arms ignore it.
    //
    // Bound to the name `checkpoint_store` so the loop-Task engineer
    // (t2.fgj.ya.9j) can drop in the spawn call without a rename.
    let checkpoint_store: Arc<dyn CheckpointStore> = build_checkpoint_store(
        &config,
        service_token_provider.clone(),
        Some(metrics.clone()),
    )
    .await
    .unwrap_or_else(|e| {
        tracing::error!(error = %e, "failed to construct checkpoint store");
        std::process::exit(1);
    });
    tracing::info!(
        adapter = ?config.checkpoint_adapter,
        enabled = config.checkpoint_enabled,
        "checkpoint store initialized"
    );

    // ── SR-5.5 (Epic 6 t2.fgj.mh.a9): refuse to start when both Valkey AOF
    // and checkpoints are disabled. Best-effort probe — managed Valkey can't
    // answer it (AWS MemoryDB/ElastiCache block CONFIG; AMR does not expose
    // the appendonly parameter), in which case the validator degrades to a
    // startup WARN and returns Ok so managed-platform operators get a softer
    // signal. The validator short-circuits internally when
    // `config.checkpoint_enabled == true`, so the gate is intentionally not
    // duplicated at this call site.
    live_edit::checkpoint::validate_durability_mode(&config, &valkey)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "startup refused by durability guard");
            std::process::exit(1);
        });

    // ── Step 1b-iii.4: External-value store (SDD §11 + SR-3.1 + SR-3.2) ───
    //
    // Constructed AFTER `checkpoint_store` so the `with_checkpoint_store`
    // builder call below can chain against the freshly-built
    // `Arc<dyn CheckpointStore>` (Subtask t3.fgj.zc.p6.78). The
    // `with_ext_mirror_sender` chain consumes the producer half of the
    // SR-3.1 channel constructed at Step 1b-pre. Each adapter-touching
    // builder chain has its own gate: `with_ext_mirror_sender` is attached
    // exactly when the plan funds the mirror (checkpointing on and an
    // adapter set; Step 1b-pre), and `with_checkpoint_store` when the
    // adapter is not `Disabled`, whatever `LIVE_EDIT_CHECKPOINT_ENABLED`
    // says (its comment below). With the adapter `Disabled` neither is
    // attached, preserving the pre-Epic behaviour — wiring a
    // `NoopCheckpointStore` into recovery would burn a singleflight entry +
    // a counter tick per miss for a permanent `adapter_miss` outcome, which
    // is exactly what the pre-Task code path skipped.
    // b.yuo, b.qhk: cancels the store-memory re-read (the release budget and
    // live_edit_store_memory_used_ratio; spawned below when either can use
    // it), at shutdown.
    let ext_release_budget_refresh_cancel = CancellationToken::new();
    let ext_value_store: Arc<dyn live_edit::storage::external_values::ExternalValueStore> = {
        let mut builder = ValkeyExternalValueStore::new(valkey.clone(), key_schema.clone())
            // b.j9j item 4: attach metrics so `gc_underflow_total`
            // increments when the Lua script reports a post-
            // decrement ref_count below zero.
            .with_metrics(metrics.clone())
            // SDD §11.1: the upload content check's mode
            // (LIVE_EDIT_EXT_VALUE_VERIFY), counted on the metrics above.
            .with_ext_value_verify(config.ext_value_verify);
        // b.yuo: the external-value release budget's upload gate. A budget of
        // 0 disables it (and the GC loop's pressure trigger, which reads the
        // budget off the same gate), so no gate is attached then.
        //
        // The configuration answers LIVE_EDIT_EXT_RELEASE_MAX_BYTES or the
        // fallback, min(8 x LIVE_EDIT_MAX_LAYER_SIZE, 100 MB). Unless the
        // budget is explicit, the store's memory replaces the fallback
        // (decider ruling 4, b.qhk): the smallest maxmemory / 8 when every
        // primary reports one above zero; else, with
        // LIVE_EDIT_VALKEY_MEMORY_BYTES (the store's total) set, one shard's
        // share of it / 8 by the cluster map; the fallback otherwise, logged
        // with why — INFO for a store that does not report its size, WARN for
        // a fault (b.1jt).
        //
        // The store is read once at startup whatever is configured, for
        // live_edit_store_memory_used_ratio too. That read decides whether
        // the 60 s refresh runs: unless the budget is explicit, or when the
        // gauge can have a value. The refresh is seeded with what startup
        // read and logged, so it does not log the same reason again and its
        // first replica-acknowledgement comparison starts from startup's.
        let store_memory = live_edit::storage::store_memory::startup_store_memory(
            &valkey,
            config.ext_release_budget(),
            config.valkey_memory_bytes,
        )
        .await;
        let release_budget = store_memory.budget;
        config.log_ext_release_budget_startup(&release_budget, store_memory.problem.as_ref());
        live_edit::storage::store_memory::log_store_memory_startup(
            &store_memory,
            config.valkey_memory_bytes,
        );
        metrics.record_ext_release_budget(release_budget.source.label(), release_budget.bytes);
        metrics.record_store_memory_used_ratio(store_memory.used_ratio);
        let gate = (release_budget.bytes > 0).then(|| {
            Arc::new(live_edit::storage::ext_release_gate::ExtReleaseGate::new(
                valkey.clone(),
                &key_schema,
                release_budget.bytes,
                // The gate waits up to what is left of the call's own
                // deadline minus a margin, capped at the RPC deadline.
                Duration::from_millis(config.rpc_deadline_ms),
                Some(metrics.clone()),
            ))
        });
        if store_memory.refresh_needed {
            // Detached: it holds no state worth draining, and the shutdown
            // sequence cancels it with the other background tokens.
            let _refresh = live_edit::storage::store_memory::spawn_store_memory_refresh(
                valkey.clone(),
                live_edit::storage::store_memory::StoreMemoryRefresh {
                    gate: gate.clone(),
                    initial: release_budget,
                    memory_bytes: config.valkey_memory_bytes,
                    seed: store_memory.seed(),
                    metrics: Some(metrics.clone()),
                },
                ext_release_budget_refresh_cancel.clone(),
            );
        }
        if let Some(gate) = gate {
            builder = builder.with_ext_release_gate(gate);
        }
        if let Some(sender) = mirror_sender {
            builder = builder.with_ext_mirror_sender(sender);
        }
        // SDD SR-3.2: attach the CheckpointStore handle so the
        // `read_batch` miss path can fall back to the adapter for
        // ext-value recovery. Independent of
        // `LIVE_EDIT_CHECKPOINT_ENABLED` — once an adapter is wired,
        // recovery is available even when the periodic checkpoint loop
        // is disabled (e.g. a read-only replica that should still
        // recover from a peer's checkpoints).
        if !matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled) {
            builder = builder.with_checkpoint_store(Arc::clone(&checkpoint_store));
        }
        Arc::new(builder)
    };

    // ── Step 1b-iii.5: Session-checkpoint dispatcher (Task t2.fgj.7v.p9) ──
    //
    // Shared `TaskTracker` + `SessionCheckpointWriter` constructed once and
    // threaded into both `SessionStore` and `LayerStore` below. The tracker
    // is drained during shutdown BEFORE `ValkeyClient::quit()` so in-flight
    // session-checkpoint writes don't observe a torn-down Valkey pool. The
    // drain reuses `shutdown_task_budget` (no new env var); see the
    // shutdown sequence below for the corresponding `close()` + `wait()`
    // mirror of the existing `presence_cleanup_tracker` pattern.
    //
    // The trailing `enabled` flag gates the whole dispatch path on a real
    // adapter being wired, in the same `!matches!(..., Disabled)` form the
    // sibling adapter-touching wirings above use (b.443). Without it a
    // `Disabled` deployment pays a session read per lifecycle event only to
    // discard the payload at the adapter's `NotImplemented`.
    let session_checkpoint_tracker = tokio_util::task::TaskTracker::new();
    let session_checkpoint_writer = Arc::new(live_edit::checkpoint::SessionCheckpointWriter::new(
        Arc::clone(&checkpoint_store),
        valkey.clone(),
        key_schema.clone(),
        session_checkpoint_tracker.clone(),
        Arc::clone(&metrics),
        config.service_label.clone(),
        !matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled),
    ));
    let shutdown_session_checkpoint_tracker = session_checkpoint_tracker.clone();
    tracing::info!(
        adapter = ?config.checkpoint_adapter,
        "session-checkpoint dispatcher initialised"
    );

    // Shared, immutable snapshot of the parsed config. Hoisted above the store
    // wiring because the object-id counter repair below needs it to re-derive
    // its caller-side wait budget per request (b.hka (f)); the recovery hooks
    // further down take the same handle.
    let config_shared = Arc::new(config.clone());

    // ── Step 1b-iii.6: Session store + Layer store ────────────────────────
    //
    // Constructed here (post-writer) so each can be chained with
    // `.with_checkpoint_writer(...)` against the shared dispatcher above.
    let session_store = Arc::new(
        SessionStore::new(valkey.clone(), key_schema.clone())
            .with_retry_max_attempts(config.session_retry_max_attempts)
            .with_retry_initial_delay_ms(config.session_retry_initial_delay_ms)
            .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer)),
    );
    // The shutdown token of the background tasks: the presence and session
    // subscribers and sweepers spawned further down, and the layer store's
    // structural-cache audits. Created here so the layer store can hold it;
    // the shutdown sequence cancels it with the other background tokens.
    let background_tasks_cancel = CancellationToken::new();
    // SDD §5.4: the per-replica upload memory budget and its overflow slot.
    // CreateLayer and UpdateLayer reserve from it after authorization and
    // before reading their content; a queued admission or processing wait ends
    // at the shutdown handle's `begun` token. The gate is built from the
    // validated config, whose budget, derived from the pod memory limit,
    // clears the size-cap floor. Built before
    // the layer store, whose structural-cache audit holds its repair in the
    // same budget (`UploadBudgetAuditCharges`), joined here because
    // `storage` must not depend on `grpc`.
    let upload_admission = Arc::new(live_edit::grpc::upload_admission::UploadAdmissionGate::new(
        config.upload_admission_config(),
        Some(Arc::clone(&metrics)),
    ));
    live_edit::grpc::upload_admission::log_upload_admission_startup(&upload_admission);
    let layer_store = Arc::new({
        let mut builder = LayerStore::new(
            valkey.clone(),
            key_schema.clone(),
            config.layer_stream_maxlen,
        )
        // The per-object relay byte budget (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`;
        // `0` = count-only retention) every UpdateLayer commit folds and trims
        // by, beside the entry cap above.
        .with_layer_stream_max_bytes(config.layer_stream_max_bytes)
        .with_update_layer_retry_max_attempts(config.update_layer_retry_max_attempts)
        .with_update_layer_per_object_concurrency(config.update_layer_per_object_concurrency)
        .with_update_layer_backend(config.update_layer_backend)
        // Every commit path's snapshot-fence waiting, one clock per call, is
        // capped at the RPC deadline and at what is left of the calling RPC's
        // own deadline, so the call answers UNAVAILABLE before the tower
        // deadline drops it. `grpc::deadline::remaining` is injected here, at
        // the composition root, because `storage` must not depend on `grpc`.
        .with_fence_wait_deadline(
            std::time::Duration::from_millis(config.rpc_deadline_ms),
            Arc::new(live_edit::grpc::deadline::remaining),
        )
        // The orphan sweep grace, derived from the RPC deadline (never
        // configured on its own): how long the sweep leaves a new object to the
        // CreateLayer that made it, and how long that call's in-process commit
        // finisher retries. The sweep spawned below reads it from this store.
        .with_orphan_sweep_grace(live_edit::storage::orphan_sweep::orphan_sweep_grace(
            config.rpc_deadline_ms,
        ))
        // The commit release ring's TTL (`le:{oid}:obj:wrel`), set by every
        // commit that writes the ring: the RPC deadline + the Valkey command
        // timeout + 60 s, so it outlives every call that can read an entry.
        .with_commit_release_ring_ttl(live_edit::storage::keys::commit_release_ring_ttl(
            config.rpc_deadline_ms,
            config.valkey_command_timeout_ms,
        ))
        .with_metrics(metrics.clone())
        .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer))
        // The structural write-planning cache at its budget, derived from the
        // pod memory limit (`config::memory_plan`).
        .with_structural_cache(live_edit::storage::structural_cache::StructuralCache::new(
            config.structural_cache_budget_bytes(),
        ))
        // Its sampled audit (`LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE` /
        // `_ABOVE_CEILING`), whose repair holds the upload budget beside the
        // uploads and whose running audit the shutdown cancels.
        .with_structural_cache_audit(
            live_edit::storage::structural_cache_audit::StructuralCacheAudit::new(
                config.structural_cache_audit_rate,
                config.structural_cache_audit_above_ceiling,
                Arc::new(
                    live_edit::grpc::upload_admission::UploadBudgetAuditCharges::new(Arc::clone(
                        &upload_admission,
                    )),
                ),
                background_tasks_cancel.clone(),
            ),
        )
        // The retirement allowance `R_ref` every layer delete, replaced
        // layer's clean-up and orphan retirement reserves from
        // (`storage::retirement`), from the pod memory plan (`L / 500`, a
        // margin term).
        .with_retirement_budget_bytes(config.pod_memory.retirement_budget_bytes())
        // b.vn6: the delete paths record retired object ids on
        // `le:ckpt:{recon}:dead` so the reconciliation tick reaps their
        // checkpoint generations. Gated on the SAME predicate that decides
        // whether the checkpoint loop — the ZSET's only consumer — is spawned
        // at all, a few hundred lines below. Enabling it without the loop would
        // grow the ZSET by one member per DeleteLayer forever against a
        // `noeviction` Valkey.
        .with_dead_object_registration(
            config.checkpoint_enabled
                && !matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled),
        );
        // b.b2j: fail-closed object-id allocation needs a way back. Gated on a
        // real adapter being wired — the same `!matches!(..., Disabled)` form
        // the sibling adapter-touching wirings use — because the repair's only
        // authoritative input is the checkpoint counters sidecar. Under
        // `Disabled` there are no checkpoint blobs to collide with and
        // `create_layer` falls back to establishing the counter at zero.
        //
        // Two budgets, deliberately different: the fetch budget bounds the
        // sidecar read inside the repair's detached task (off any RPC clock),
        // while the wait budget bounds how long the `CreateLayer` handler waits
        // on that task — strictly below `LIVE_EDIT_RPC_DEADLINE_MS` so this, and
        // not the tower deadline layer dropping the future, is what ends the
        // wait. See `Config::object_id_counter_repair_wait_budget`.
        //
        // `.with_deadline_aware_wait_budget` (b.hka (f)) makes that ordering
        // unconditional rather than merely typical: the wait is re-derived per
        // call from what is actually LEFT of the request's deadline, which
        // matters because allocation is the terminal step of
        // `handle_create_layer` and because a client may have set a shorter
        // `grpc-timeout`. `grpc::deadline::remaining` is injected as a closure
        // here, at the composition root, so `checkpoint` never names `grpc` —
        // the module dependency rules in CLAUDE.md do not list that edge, and
        // `grpc → checkpoint` already runs the other way.
        if !matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled) {
            builder = builder.with_object_id_counter_repair(Arc::new(
                live_edit::checkpoint::CheckpointObjectIdCounterRepair::new(
                    Arc::clone(&checkpoint_store),
                    // Metrics wired: a store write refusal of the repair's
                    // counter writes is a `CreateLayer`'s, counted at
                    // `stage="create_commit"` beside its allocation.
                    Arc::new(
                        live_edit::storage::counters::CounterStore::new(
                            valkey.clone(),
                            key_schema.clone(),
                        )
                        .with_metrics(Arc::clone(&metrics)),
                    ),
                    config.checkpoint_counter_safety_margin,
                    Arc::clone(&metrics),
                    config.checkpoint_recovery_fetch_budget(),
                    config.object_id_counter_repair_wait_budget(),
                )
                .with_deadline_aware_wait_budget(
                    Arc::clone(&config_shared),
                    Box::new(live_edit::grpc::deadline::remaining),
                ),
            ));
        }
        builder
    });

    // ── Step 1b-iv.0: Base bootstrap (SDD SR-7.1 creation-time first base,
    // SR-3.6 post-restore gap re-base) ────────────────────────────────────
    //
    // Constructed unconditionally like the hooks below; the gate is resolved
    // here and stored on the unit, so neither `handle_create_layer` nor the
    // recovery hook branches on configuration. The gate is the full three-way
    // conjunction SR-7.1 mandates (`Config::checkpoint_base_bootstrap_enabled`,
    // which the pod memory plan's base-bootstrap term also reads) —
    // deliberately stricter than the session
    // dispatcher's `!Disabled`, because this unit exists to serve the tail path
    // and is off whenever the tail path is. When
    // `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false` (the operator opt-out; the
    // default is `true` since b.3n1) the conjunction is false and restore
    // behaviour is byte-identical to pre-SR-3.6: no task is spawned at either
    // call site.
    //
    // **Constructed HERE, above the recovery hook, and threaded into BOTH call
    // sites as the same `Arc`.** That is a wiring invariant, not an ordering
    // convenience: the unit's in-flight admission bound
    // (`checkpoint::BOOTSTRAP_MAX_INFLIGHT`) is a field, so a second instance
    // would be a second independent bound and "one bound across both sites"
    // would be quietly false. It depends only on `checkpoint_store`, `valkey`,
    // `key_schema`, `session_checkpoint_tracker`, `metrics`, and `config`, all
    // of which exist by this point, so the move costs nothing.
    //
    // It REUSES `session_checkpoint_tracker` rather than creating a second
    // tracker: that one is already `close()`d and `wait()`ed inside the
    // shutdown budget below, so sharing it drains in-flight bootstrap tasks
    // before `ValkeyClient::quit()` with zero new shutdown wiring.
    let base_bootstrap = Arc::new(BaseBootstrap::new(
        Arc::clone(&checkpoint_store),
        valkey.clone(),
        key_schema.clone(),
        session_checkpoint_tracker.clone(),
        Arc::clone(&metrics),
        config.service_label.clone(),
        config.checkpoint_base_bootstrap_enabled(),
    ));

    // ── Step 1b-iv: Recovery hook (SDD SR-2.4 / SR-2.6) ───────────────────
    //
    // Constructed unconditionally; the disabled short-circuit lives inside
    // `try_restore_layer` so the gRPC handlers never need to branch on
    // `checkpoint_enabled` at the call site. The hook holds an
    // `Arc<Config>` clone so it can pass the live `checkpoint_seq_safety_margin`
    // into `apply_checkpoint` on every restore.
    //
    // `.with_ext_value_store(..)` is likewise UNCONDITIONAL (SDD SR-3.4): the
    // composed restore path pre-stores a lineage's tail-carried external values
    // through it, strictly before the single `write_restored_object` MULTI/EXEC. The ext
    // store above is itself constructed unconditionally and depends on none of
    // `layer_store` / the hook, so there is no cycle and no construction-order
    // move. Cloned rather than moved because `ext_value_store` is still wired
    // into the gRPC service and the GC loop below.
    let recovery_hook = Arc::new(
        RecoveryHook::new(
            Arc::clone(&checkpoint_store),
            Arc::clone(&layer_store),
            Arc::clone(&config_shared),
            Arc::clone(&metrics),
        )
        .with_ext_value_store(Arc::clone(&ext_value_store))
        .with_base_bootstrap(Arc::clone(&base_bootstrap)),
    );

    // ── Step 1b-iv.5: Session-recovery hook (SDD SR-2.7, Epic 2) ──────────
    //
    // Mirrors the layer-recovery hook above. Constructed unconditionally —
    // the disabled short-circuit lives inside `try_recover_session` so the
    // session-resolving gRPC handlers never need to branch on
    // `checkpoint_enabled` at the call site. Threaded into
    // `SessionBasePathCache::with_recovery_hook` below so the seven session-
    // resolving handlers (CreateLayer, UpdateLayer, DeleteLayer, StatLayer,
    // DeleteSession, JoinSession, SubscribeReadLayer) all funnel through
    // one wiring point. The hook is ALSO held on `LiveEditService` for
    // diagnostics / future direct call sites.
    let session_recovery_hook = Arc::new(SessionRecoveryHook::new(
        Arc::clone(&checkpoint_store),
        valkey.clone(),
        key_schema.clone(),
        Arc::clone(&config_shared),
        Arc::clone(&metrics),
    ));

    // ── Step 1b-ii: Connect to external auth services ─────────────────────
    let auth_retry_config = live_edit::auth::RetryConfig {
        max_attempts: config.auth_max_attempts,
        initial_delay: std::time::Duration::from_millis(config.auth_retry_initial_delay_ms),
        ..Default::default()
    };

    let raw_permissions_client = build_permissions_client(&config, &auth_retry_config)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to connect to permissions backend");
            std::process::exit(1);
        });

    // With the userinfo service and authorization both on, this is the
    // backend wrapped in the not-found fallback decorator, which counts its
    // fallbacks on `metrics`; it sits beneath the instrumented wrapper and
    // the cache below, so a fallback is a success to both.
    let raw_userinfo_client = build_userinfo_client(&config, &auth_retry_config, metrics.clone())
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to connect to userinfo backend");
            std::process::exit(1);
        });

    // Instrument backend auth clients with duration + error metrics before
    // handing them to the cache (b.nrt item 4). Cache hits bypass these
    // wrappers; only calls that miss the cache — the ones operators care
    // about during an auth incident — contribute to the histogram.
    let auth_protocol_label: &'static str = match config.auth_protocol {
        AuthProtocol::Rest => "rest",
        AuthProtocol::Grpc => "grpc",
    };
    let instrumented_permissions: Arc<dyn live_edit::auth::permissions::PermissionsClient> =
        Arc::new(InstrumentedPermissionsClient::new(
            raw_permissions_client,
            metrics.clone(),
            "storage",
            auth_protocol_label,
        ));
    let instrumented_userinfo: Arc<dyn live_edit::auth::userinfo::UserinfoClient> =
        Arc::new(InstrumentedUserinfoClient::new(
            raw_userinfo_client,
            metrics.clone(),
            "userinfo",
            auth_protocol_label,
        ));

    // Wrap auth clients in AuthCache (SDD §14) so RPC handlers benefit from
    // HMAC-keyed, TTL-expiring caches instead of hitting external services on
    // every request. Negative decisions
    // (Denied/Unauthenticated/PathNotFound/InvalidRequest) use a shorter TTL
    // to bound revocation lag (b.nrt item 1).
    let auth_cache = Arc::new(AuthCache::new_with_negative_ttl(
        instrumented_permissions,
        instrumented_userinfo,
        config.auth_cache_ttl_secs,
        config.auth_cache_negative_ttl_secs,
        config.auth_userinfo_negative_ttl_secs,
        config.auth_cache_max_entries,
        Some(metrics.clone()),
    ));
    let permissions_client: Arc<dyn live_edit::auth::permissions::PermissionsClient> =
        Arc::clone(&auth_cache) as _;
    let userinfo_client: Arc<dyn live_edit::auth::userinfo::UserinfoClient> = auth_cache;

    // ── Step 1c: External-value GC loop (SDD §11.3, §11.4) ───────────────────
    //
    // No startup drain: on every pass the loop's leader re-claims in place its
    // own `ext:inflight` claims older than the re-claim age (b.1jt), so a
    // previous process's crash victims are decided by whichever replica leads,
    // within a pass of it taking the lease, and never go back to `ext:pending`.
    // A re-claimed member whose decrement already applied is answered
    // `replayed` by the decrement script, never decremented twice; an older
    // build's stranded claim is dropped, never decremented.
    let pending_key = key_schema.ext_pending();
    let pending_count: u64 = valkey.pool.scard(&pending_key).await.unwrap_or_else(|e| {
        live_edit::storage::client::FredErrOrigin::client(
            &valkey,
            live_edit::storage::client::POOL_LABEL_DATA,
        )
        .observe_error(&e);
        tracing::warn!(error = %e, "failed to check pending GC set size");
        0
    });

    if pending_count > 0 {
        tracing::info!(
            count = pending_count,
            "found pending GC entries from previous run"
        );
    }

    let gc_cancel = CancellationToken::new();
    let gc_handle = spawn_gc_task(
        Arc::clone(&ext_value_store),
        valkey.clone(),
        key_schema.clone(),
        leader_identity.clone(),
        live_edit::storage::external_values::GcPolicy::PRODUCTION,
        gc_cancel.clone(),
        Some(metrics.clone()),
    );

    // ── Step 1d: Orphan sweep background task (SDD §5.5 step 6) ──────────────
    let orphan_sweep_cancel = CancellationToken::new();
    let orphan_sweep_handle = spawn_orphan_sweep(
        valkey.clone(),
        key_schema.clone(),
        ext_value_store.clone(),
        Arc::clone(&layer_store),
        leader_identity.clone(),
        orphan_sweep_cancel.clone(),
        Some(metrics.clone()),
    );

    // ── Step 1d-bis: External-value ref_count reconciler (b.yda) ─────────────
    //
    // A third consumer of the `le:gc:leader` lease, on a third independent
    // interval — the SDD §22.7.2 shape. Deliberately NOT a step of the GC loop's
    // once-a-second pass: it walks the whole object and blob keyspace, and the
    // residuals it drains accumulate over hours.
    //
    // `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0` is the documented off switch, so
    // the handle is an `Option` and the shutdown join below is conditional. The
    // startup WARN for that case lives in `Config::warn_ext_reconcile_disabled`,
    // beside the value it is warning about.
    let ext_reconcile_cancel = CancellationToken::new();
    let ext_reconcile_handle = if config.ext_reconcile_interval_secs > 0 {
        Some(live_edit::storage::ext_reconcile::spawn_ext_ref_reconcile(
            valkey.clone(),
            key_schema.clone(),
            leader_identity.clone(),
            Duration::from_secs(config.ext_reconcile_interval_secs),
            live_edit::storage::ext_reconcile::ext_reconcile_grace(config.rpc_deadline_ms),
            ext_reconcile_cancel.clone(),
            Some(metrics.clone()),
        ))
    } else {
        None
    };

    // ── Step 1e-pre: Checkpoint pacing config-echo gauges (b.ijf) ─────────
    //
    // Echo this replica's pacing knobs as gauges so alert rules can be
    // written relative to the configuration instead of hardcoding the numbers
    // that only hold at the defaults. Called unconditionally on purpose: the
    // seam itself owns the "iff this replica checkpoints" predicate (the
    // family's presence is that signal), so there is no gate here to drift
    // from the loop-spawn gate below, and none to accidentally hoist the
    // registration out of.
    register_checkpoint_pacing_gauges_if_enabled(&config, &metrics);

    // ── Step 1e: Leader-elected periodic checkpoint loop (SDD SR-2.1) ─────
    //
    // Gated on `LIVE_EDIT_CHECKPOINT_ENABLED=true`. The adapter must be
    // one of `InMemory` / `LocalDisk` / `StorageApi` — the
    // `enabled=true` + `adapter=Disabled` combination has already been
    // refused at startup by the case-2 check in `validate_durability_mode`,
    // so reaching this block with that pairing is impossible. The loop is spawned against the
    // factory-built `Arc<dyn CheckpointStore>`. The
    // `Option<JoinHandle<()>>` binding lets the shutdown sequence
    // conditionally await without branching its `tokio::join!`.
    let (checkpoint_loop_cancel, checkpoint_loop_handle) = if config.checkpoint_enabled {
        match config.checkpoint_adapter {
            CheckpointAdapter::Disabled => unreachable!(
                "checkpoint_enabled=true + adapter=Disabled is refused at startup by \
                 the case-2 check in validate_durability_mode; reaching this arm \
                 indicates a missing or reordered guard call"
            ),
            CheckpointAdapter::InMemory
            | CheckpointAdapter::LocalDisk
            | CheckpointAdapter::StorageApi => {
                let cancel = CancellationToken::new();
                let loop_config = CheckpointLoopConfig {
                    tick: Duration::from_secs(config.checkpoint_tick_secs),
                    interval: Duration::from_secs(config.checkpoint_interval_secs),
                    quiesce: Duration::from_secs(config.checkpoint_quiesce_secs),
                    retain_generations: config.checkpoint_retain_generations as usize,
                    counters_sync: Duration::from_secs(config.checkpoint_counters_sync_secs),
                    reconcile_interval: Duration::from_secs(
                        config.checkpoint_reconcile_interval_secs,
                    ),
                    tail: TailCaptureConfig::from_config(&config),
                    leader: leader_identity.clone(),
                };
                let handle = spawn_checkpoint_loop(
                    valkey.clone(),
                    key_schema.clone(),
                    Arc::clone(&checkpoint_store),
                    loop_config,
                    Some(metrics.clone()),
                    cancel.clone(),
                );
                // Derived fields (b.gg7): the configured knobs alone do not
                // tell an operator how long one call may take, and every
                // ceiling below is a *product* of knobs rather than a knob, so
                // it cannot be read off the deployment manifest. Echoed on the
                // existing line rather than a new one — a second INFO would
                // split the checkpoint startup story across two log records for
                // no gain.
                let adapter_call_ceiling = config.checkpoint_adapter_call_ceiling();
                tracing::info!(
                    adapter = ?config.checkpoint_adapter,
                    tick_secs = config.checkpoint_tick_secs,
                    interval_secs = config.checkpoint_interval_secs,
                    quiesce_secs = config.checkpoint_quiesce_secs,
                    retain_generations = config.checkpoint_retain_generations,
                    counters_sync_secs = config.checkpoint_counters_sync_secs,
                    adapter_call_ceiling_secs = adapter_call_ceiling.as_secs(),
                    layer_publish_ceiling_secs =
                        adapter_call_ceiling.as_secs() * LAYER_PUBLISH_ARTIFACTS,
                    reconcile_grace_secs = config.checkpoint_reconcile_grace().as_secs(),
                    reconcile_interval_secs = config.checkpoint_reconcile_interval_secs,
                    tail_enabled = config.checkpoint_tail_enabled,
                    tail_max_chain = config.checkpoint_tail_max_chain,
                    tail_max_entries = config.checkpoint_tail_max_entries,
                    tail_max_bytes = config.checkpoint_tail_max_bytes,
                    "checkpoint loop spawned",
                );
                (Some(cancel), Some(handle))
            }
        }
    } else {
        tracing::info!("checkpoint loop disabled (LIVE_EDIT_CHECKPOINT_ENABLED=false)");
        (None, None)
    };

    // ── Step 1e-bis: External-value mirror workers and backup repair (SDD §22.17)
    //
    // Gated by the channel construction at Step 1b-pre,
    // `ext_mirror_channel_for_plan`, whose gate is the pod memory plan
    // funding the mirror (`ext_mirror_bytes()` is `Some`, exactly when
    // `checkpoint_enabled == true` AND the adapter is not `Disabled`):
    // workers run only when the receiver exists. When
    // skipped, the mirror's cancel and the repair's handle stay `None` and
    // the handle vec is empty so the shutdown block branches uniformly
    // without having to re-derive the gate.
    //
    // The receiver was constructed at Step 1b-pre and held in
    // `mirror_rx_for_workers`; we move it into `spawn_ext_mirror_workers`
    // here.
    //
    // The external-value backup repair (SDD §22.17) is spawned under the same
    // gate, beside the workers: the workers confirm each copy they deliver
    // under the backup store's identity, and the repair re-sends the copies
    // never confirmed, through the same byte budget. It is the fourth
    // consumer of `le:gc:leader`, on its own fixed interval. Its cancel and
    // join sit beside the reconciler's in the shutdown sequence.
    let ext_backup_repair_cancel = CancellationToken::new();
    let (ext_mirror_cancel, ext_mirror_worker_handles, ext_backup_repair_handle): (
        Option<CancellationToken>,
        Vec<tokio::task::JoinHandle<()>>,
        Option<tokio::task::JoinHandle<()>>,
    ) = match mirror_rx_for_workers.zip(repair_mirror_sender) {
        Some((rx, repair_sender)) => {
            let cancel = CancellationToken::new();
            let backup_store_id = config.checkpoint_store_identity();
            let marks = ExtBackupMarks::new(
                valkey.clone(),
                key_schema.clone(),
                backup_store_id.clone(),
                Some(metrics.clone()),
            );
            let handles = spawn_ext_mirror_workers(
                Arc::clone(&checkpoint_store),
                rx,
                config.checkpoint_ext_mirror_workers,
                metrics.clone(),
                Some(marks),
                cancel.clone(),
            );
            let repair = ExtBackupRepair::new(
                valkey.clone(),
                key_schema.clone(),
                Arc::clone(&checkpoint_store),
                repair_sender,
                backup_store_id.clone(),
                config.checkpoint_adapter_call_ceiling(),
                Some(metrics.clone()),
            );
            let repair_interval_secs = repair.timing().interval.as_secs();
            let repair_handle = spawn_ext_backup_repair(
                repair,
                leader_identity.clone(),
                ext_backup_repair_cancel.clone(),
            );
            tracing::info!(
                workers = config.checkpoint_ext_mirror_workers,
                queue_capacity = config.checkpoint_ext_mirror_queue_capacity,
                budget_bytes = config.pod_memory.ext_mirror_bytes().unwrap_or(0),
                backup_store_id = %backup_store_id,
                repair_interval_secs,
                "checkpoint ext-mirror workers and backup repair spawned",
            );
            (Some(cancel), handles, Some(repair_handle))
        }
        None => (None, Vec::new(), None),
    };

    // ── Step 2: Metrics ──────────────────────────────────────────────────────
    // (Built earlier — moved ahead of auth wiring so auth calls are
    // instrumented from the first request. See Step 1 block above.)

    // ── Step 3: Valkey-scoped metric registration ────────────────────────────
    // (The `/metrics` listener + server were moved ahead of the dependency
    // wiring — see Step 0a-bis. Only the gauge that needs a `ValkeyClient`
    // remains here.)
    //
    // b.k1r H-stor-2: the `live_edit_storage_watch_unwatch_inflight` gauge is
    // an OpenTelemetry observable gauge whose callback runs at each metric
    // collection (every `/metrics` scrape and every OTLP push tick), reading
    // the live `WatchGuard::Drop`-spawned UNWATCH tracker count. This replaces
    // the pre-scrape hook that the pull-only `prometheus`-crate path used.
    {
        let valkey = valkey.clone();
        metrics.register_storage_watch_unwatch_inflight_gauge(move || {
            valkey.watch_unwatch_inflight() as i64
        });
    }

    // ── Step 3b: Counter-seed startup recovery (SDD SR-5.3 / SR-5.4) ──────
    //
    // Task `t2.fgj.mh.4u` — the readiness-gated AOF-off recovery routine.
    // The routine reads the counters-sidecar blob via the configured
    // `CheckpointStore`, computes `seed = max_object_id_observed +
    // safety_margin`, and atomically advances `le:counter:object_id` past
    // that value via Lua (`CounterStore::seed_object_id_if_below`).
    //
    // The `counter_seed_ready_tx` watch handle is bound in this outer scope
    // so it outlives the supervisor spawn — the supervisor's `Receiver`
    // keeps reading whatever value the sender most recently wrote, but a
    // dropped sender does NOT auto-flip the value, so keeping the `Sender`
    // alive is the cleaner contract.
    //
    // Ordering invariants:
    //   - The routine MUST be `await`ed BEFORE `serve_with_shutdown` runs,
    //     so the readiness gate is in its final state by the time the
    //     server accepts traffic.
    //   - When `config.checkpoint_enabled == false`, no checkpoint store
    //     produces a sidecar; the routine is skipped and the supervisor
    //     receives `counter_seed_ready: None` so the gate is inert (the
    //     readiness model collapses to the Valkey + auth + service-token
    //     baseline).
    //   - On `Ok(_)` (Absent / Seeded / NoOp), the gate is flipped to
    //     `true`. `Absent` is the legitimate "no checkpoint sidecar yet"
    //     path (AOF-on durability mode); a future `/quo-execute` iteration
    //     can tighten this further once the per-cluster durability mode
    //     is a first-class config knob.
    //   - On `Err`, the gate stays `false`. Readiness remains
    //     `NOT_SERVING`; the error is logged but the process does NOT exit.
    //     The fetch is wrapped in `seed_object_id_counter_with_retry` so a
    //     transient `Store` failure — most commonly the credential-provisioner
    //     sidecar not having written the service token yet at startup — is
    //     retried with a bounded budget rather than wedging readiness
    //     permanently. (Since b.rgs the liveness probe genuinely does target a
    //     bare gRPC server that is already answering `SERVING` — the bootstrap
    //     server bound at Step 0a — so a one-shot seed failure never triggers a
    //     restart. The in-process retry is one recovery path; the chart's
    //     `startupProbe`, which targets the *readiness* service name, is the
    //     other: it restarts a pod whose seed never completes.)
    let (counter_seed_ready_tx, counter_seed_ready_rx) = if config.checkpoint_enabled {
        let counters = Arc::new(live_edit::storage::counters::CounterStore::new(
            valkey.clone(),
            key_schema.clone(),
        ));
        let (tx, rx) = tokio::sync::watch::channel(false);
        match live_edit::checkpoint::seed_object_id_counter_with_retry(
            Arc::clone(&checkpoint_store),
            counters,
            config.checkpoint_counter_safety_margin,
            Arc::clone(&metrics),
            live_edit::checkpoint::COUNTER_SEED_RETRY_MAX_ATTEMPTS,
            live_edit::checkpoint::COUNTER_SEED_RETRY_DELAY,
        )
        .await
        {
            Ok(outcome) => {
                tracing::info!(?outcome, "counter-seed recovery routine succeeded");
                // Three outcome variants — `Absent`, `Seeded`, `NoOp` —
                // all flip the gate. `Absent` is the legitimate "no
                // counter to seed, AOF is doing the job" path; an
                // operator-visible WARN is already emitted by the
                // routine itself.
                let _ = tx.send(true);
            }
            Err(e) => {
                // Readiness will remain NOT_SERVING; the supervisor reads
                // `*rx.borrow() == false` on every tick until the
                // process exits (Kubernetes liveness restart retries).
                tracing::error!(
                    error = %e,
                    "counter-seed recovery routine failed — readiness will remain NOT_SERVING"
                );
            }
        }
        (Some(tx), Some(rx))
    } else {
        tracing::info!("checkpoint disabled — skipping the startup counter-seed recovery routine");
        (None, None)
    };
    // Keep `counter_seed_ready_tx` alive past the supervisor spawn even
    // though we never write to it again after the routine returns. See
    // SDD SR-5.4 / the Step 3b comment block above for the rationale.
    let _counter_seed_ready_tx = counter_seed_ready_tx;

    // ── Step 3c: Establish `le:counter:object_id` under a Disabled adapter ──
    //
    // b.b2j: `CounterStore::next_object_id` fails closed on an absent counter,
    // and the seed routine's `Absent` arm is what normally establishes the key
    // so that absence unambiguously means "the keyspace was lost". With
    // `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled` that routine may never run, so
    // do it here — safe precisely because a `Disabled` deployment has no
    // checkpoint blobs claiming any object id, and the SR-5.5 durability guard
    // above has already refused startup unless Valkey AOF is providing
    // durability. Without this, the first CreateLayer on every fresh
    // `Disabled` deployment would take the repair path and log a keyspace-loss
    // WARN that is not true.
    //
    // NOT done for a configured adapter: `SET NX 0` there, before the seed
    // routine has proven a floor, would hand a post-wipe allocator a valid
    // counter to INCR from — reopening exactly the reuse window this ticket
    // closes. Under a configured adapter the seed routine owns establishing
    // the key, and the lazy repair covers what it misses.
    if matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled) {
        let counters =
            live_edit::storage::counters::CounterStore::new(valkey.clone(), key_schema.clone());
        match counters.initialize_object_id_counter_if_absent().await {
            Ok(true) => tracing::info!(
                "established le:counter:object_id at 0 (no checkpoint adapter configured)"
            ),
            Ok(false) => tracing::debug!("le:counter:object_id already present"),
            // Non-fatal: the first CreateLayer repairs it lazily through the
            // no-hook fallback, which reaches the same initialize-to-zero
            // conclusion. Exiting here would turn a transient Valkey blip into
            // a crash loop.
            Err(e) => tracing::error!(
                error = %e,
                "failed to establish le:counter:object_id at startup; the first CreateLayer \
                 will initialize it"
            ),
        }
    }

    // ── Step 4: Health service ────────────────────────────────────────────────
    //
    // The reporter, the readiness service name, and the CapabilitiesService
    // SERVING mark were all established at Step 0a so the bootstrap server
    // could answer liveness before any dependency existed. What remains here
    // is (a) a second `HealthServer` for the full-API server, backed by the
    // same status map — `HealthReporter` is `Clone` and the service carries
    // the map across, so both servers answer identically — and (b) the
    // supervisor spawn, which stays at its original position because it needs
    // the `ValkeyClient` and the counter-seed watch.
    let health_service =
        live_edit::observability::health::health_service_from_reporter(&health_reporter);

    // Spawn the health supervisor (b.3z7 H-obs-2). It pings Valkey every
    // `probe_interval` and flips the readiness status so Kubernetes routes
    // traffic away from a replica whose backing store is unreachable,
    // restoring once the dependency recovers. When `authz_enabled` the
    // supervisor also probes the permissions and userinfo backends with a
    // per-backend consecutive-failure threshold so a transient auth blip
    // does not spuriously evict a replica from rotation.
    let health_supervisor_cancel = CancellationToken::new();
    let health_supervisor_handle = live_edit::observability::health::spawn_health_supervisor(
        health_reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        live_edit::observability::health::HealthSupervisorConfig {
            // b.avd item 9: probe interval / timeout now configurable.
            // Defaults match the previous hardcoded 5 s / 2 s.
            probe_interval: std::time::Duration::from_millis(config.health_probe_interval_ms),
            probe_timeout: std::time::Duration::from_millis(config.health_probe_timeout_ms),
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            // Task t2.fgj.dp.s7: pass the provider Arc (or `None` when
            // `LIVE_EDIT_SERVICE_TOKEN_FILE` is unset). The supervisor
            // only consults the provider on each tick when this is
            // `Some`; the gauge registration below is gated on the
            // same `Option` so `/metrics` carries the family iff the
            // provider is configured (the parent Task body contract).
            service_token_provider: service_token_provider.clone(),
            // Task t2.fgj.mh.4u: pass the startup-gate receiver (or
            // `None` when checkpoints are disabled). Hard signal — the
            // supervisor consults `*rx.borrow()` on every tick and gates
            // readiness on its value with no smoothing, same posture as
            // `service_token_provider`. See the Step 3b block above for
            // the readiness-gate rationale.
            counter_seed_ready: counter_seed_ready_rx,
            // b.rgs: gate readiness on the full-API server actually being
            // up. Hard signal, no smoothing — same posture as
            // `counter_seed_ready`. The supervisor also wakes on this
            // receiver's `changed()`, so readiness rises on the Step 8 flip
            // rather than up to one probe interval afterwards.
            api_serving: Some(api_serving_rx),
        },
        health_supervisor_cancel.clone(),
    );

    // ── Step 5: gRPC reflection ───────────────────────────────────────────────
    let reflection_service = ReflectionBuilder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(
            live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::FILE_DESCRIPTOR_SET,
        )
        .build_v1()
        .expect("failed to build reflection service");

    // ── Step 5b: Presence infrastructure (SDD §13) ─────────────────────────────
    let presence_router = Arc::new(live_edit::presence::LocalPresenceRouter::with_capacity(
        config.presence_broadcast_capacity,
    ));

    // Per-replica load gauges (SDD §13.4, §7.5 *Semaphore-map lifecycle*):
    // observable gauges that read live in-process state at every collection,
    // so an entry that outlives its owner shows as a gauge that never returns
    // to zero rather than hiding behind a decrement that never ran.
    //
    // `LayerStore` holds an `Arc<Metrics>`, and the callback is owned by the
    // meter provider inside `Metrics`, so a strong `Arc<LayerStore>` here would
    // close a `Metrics → callback → LayerStore → Metrics` cycle. Capture `Weak`
    // and report 0 once the store is gone. The router holds no `Metrics`, so a
    // strong clone there is cycle-free.
    {
        let sessions_router = Arc::clone(&presence_router);
        let connections_router = Arc::clone(&presence_router);
        let admission_store = Arc::downgrade(&layer_store);
        metrics.register_replica_load_gauges(
            &config.service_label,
            move || sessions_router.session_count() as i64,
            move || connections_router.connection_count() as i64,
            move || {
                admission_store
                    .upgrade()
                    .map_or(0, |store| store.semaphore_count() as i64)
            },
        );
        // The structural write-planning cache's gauges, `Weak` for the same
        // cycle reason.
        let entries_store = Arc::downgrade(&layer_store);
        let weight_store = Arc::downgrade(&layer_store);
        metrics.register_structural_cache_gauges(
            &config.service_label,
            move || {
                entries_store
                    .upgrade()
                    .map_or(0, |store| store.structural_cache().entry_count())
            },
            move || {
                weight_store
                    .upgrade()
                    .map_or(0, |store| store.structural_cache().weight_bytes())
            },
        );
    }

    // Spawn the pub/sub subscriber task — listens for presence events from other
    // replicas and forwards them to local JoinSession streams.
    let presence_subscriber_handle = live_edit::presence::spawn_presence_subscriber(
        valkey.subscriber.clone(),
        Arc::clone(&presence_router),
        key_schema.clone(),
        Arc::clone(&metrics),
        background_tasks_cancel.clone(),
    );

    // Spawn the crash-recovery presence sweeper (SDD §13.3): every
    // LIVE_EDIT_SWEEPER_INTERVAL_SECS the replica holding
    // `le:presence:sweeper:leader` (a lock of its own, so the GC and presence
    // sweeps can run on different replicas) walks the fleet-wide session
    // index and removes connections whose lease has expired, renewing its
    // lease as it goes. It stops at a page or chunk boundary once
    // `background_tasks_cancel` fires.
    let sweeper_handle = live_edit::presence::spawn_presence_sweeper(
        valkey.clone(),
        key_schema.clone(),
        leader_identity.clone(),
        live_edit::presence::SweeperBudget::production(),
        Arc::clone(&metrics),
        Duration::from_secs(config.sweeper_interval_secs),
        background_tasks_cancel.clone(),
    );

    // ── Step 5c: Session-deleted pub/sub subscriber (SDD §7.8) ─────────────
    // Every replica subscribes so DeleteSession on any replica cancels local
    // streams on all others. Also drives cleanup of stale entries in
    // `session_tokens` when DeleteSession is routed to a different replica.
    // Note: this only reacts to explicit DeleteSession events. Entries of
    // sessions that are never deleted by any client (e.g., crashed clients)
    // are reaped by the session_tokens sweeper (Step 5e) once their session
    // is gone from the store.
    live_edit::grpc::session::subscribe_session_deleted_channel(&valkey.subscriber, &key_schema)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to subscribe to session-deleted channel");
            std::process::exit(1);
        });

    // ── Step 5d: Short-TTL base_path cache (b.jr8 mediums) ─────────────────
    // Shared across all handlers + the session-deleted subscriber (so cross-
    // replica invalidations hit it) + the sweeper that ages out stale
    // entries. The session-recovery hook attached via
    // `with_recovery_hook` is the single choke point through which every
    // session-resolving handler triggers lazy on-demand recovery on a
    // cache-miss + session-absent (SDD SR-2.7, Subtask t3.fgj.7v.fp.rd).
    let session_base_path_cache = live_edit::grpc::session_cache::SessionBasePathCache::new()
        .with_recovery_hook(Arc::clone(&session_recovery_hook));
    let session_base_path_cache_sweeper_handle =
        live_edit::grpc::session_cache::spawn_cache_sweeper(
            session_base_path_cache.clone(),
            background_tasks_cancel.clone(),
        );

    // The session-deleted relay requests a deleted-only pass of the
    // session_tokens sweeper (Step 5e) through this handle whenever it may
    // have missed notices: its receiver lagged (a prompt pass), or the pub/sub
    // connection reconnected (a pass after a random delay of up to 5 s, so
    // replicas that reconnected together spread their probes). Requested
    // passes are at least 5 s apart (SDD §7.8).
    let session_revalidation = live_edit::grpc::session::SessionRevalidation::new();
    let session_delete_subscriber_handle =
        live_edit::grpc::session::spawn_session_delete_subscriber(
            valkey.subscriber.clone(),
            Arc::clone(&session_tokens),
            session_base_path_cache.clone(),
            key_schema.clone(),
            Arc::clone(&metrics),
            session_revalidation.clone(),
            background_tasks_cancel.clone(),
        );

    // ── Step 5e: session_tokens crash-recovery sweep (b.j5t item 5 / b.f36 A6) ─
    // Periodically probes `le:{session_id}:sess` for each entry in
    // `session_tokens` and removes + cancels any whose owning session is
    // absent. Necessary because the `le:session:deleted` pub/sub fan-out
    // only fires for explicit `DeleteSession` calls — clients that crash
    // (or never call DeleteSession) leave entries behind — and a replica can
    // miss a notice. On a request from the session-deleted relay it also runs
    // a prompt pass that cancels only deleted (marked) sessions; the periodic
    // pass then bounds only notice loss the relay cannot detect.
    let session_tokens_sweep_handle = live_edit::grpc::session::spawn_session_tokens_sweeper(
        Arc::clone(&session_tokens),
        Arc::clone(&session_store),
        session_base_path_cache.clone(),
        Duration::from_secs(config.session_tokens_sweep_interval_secs),
        session_revalidation,
        background_tasks_cancel.clone(),
    );

    // ── Step 6: Live service ──────────────────────────────────────────────────
    // `live_service_server` builds the one LiveService stack: the auth
    // interceptor around `LiveServiceServer::new(inner)`, and outermost the
    // layer that inserts the end-of-stream record CreateLayer and UpdateLayer
    // commit on (SDD §5.4) and the body-progress record the JoinSession client
    // keepalive reads (SDD §13.2). Every test server that serves the
    // LiveService builds it the same way: through any other stack an upload is
    // refused `INTERNAL`, and an opted-in JoinSession is silently left unarmed
    // and told `keepalive_interval_ms = 0`.
    // The 4 MB request-message limit is tonic's default decode limit: nothing
    // in `src/` calls `.max_decoding_message_size(..)`, so an oversized request
    // is refused by tonic with `OUT_OF_RANGE` before any handler runs, and is
    // logged by the metrics layer's transport backstop. The upload admission's
    // receive allowance is sized from this limit
    // (`grpc::upload_admission::TONIC_DEFAULT_DECODE_LIMIT_BYTES`); configure
    // the one and the other must follow.
    // Clone the layer_store Arc so we can await pending cleanup tasks during
    // shutdown (avoids orphaned Valkey keys from in-flight overwrite cleanup).
    let shutdown_layer_store = Arc::clone(&layer_store);

    // b.w29 item 4: shared TaskTracker for in-flight `run_join_session`
    // futures and each connection's lease keeper. Each JoinSession spawn
    // registers here; the post-server shutdown sequence closes + waits the
    // tracker (bounded) before it fires `background_tasks_cancel` and before
    // Valkey `quit()` tears down the fred pools, so every join's teardown
    // (whose unregister script publishes `left`) lands while the pools are
    // still up.
    let presence_cleanup_tracker = tokio_util::task::TaskTracker::new();
    let shutdown_presence_cleanup_tracker = presence_cleanup_tracker.clone();

    // Every `SubscribeReadLayer` subscription task is spawned on this tracker,
    // which the post-server shutdown sequence drains (bounded) alongside the
    // presence tracker (`drain_stream_tasks`), before Valkey `quit()`, so every
    // subscription ends with its own shutdown status while the pools are still
    // up.
    let subscription_tracker = tokio_util::task::TaskTracker::new();
    let shutdown_subscription_tracker = subscription_tracker.clone();

    // SDD §5.6 / SR-4.7: the graceful-shutdown handle — the process-wide
    // service shutdown token (the only signal graceful shutdown sends the
    // streaming RPCs), its `begun` child and the stream enrollment registry.
    // After the readiness drain, just behind the server's GOAWAY,
    // `end_streams_after_drain` below cancels `begun` and ends the enrolled
    // streams in shuffled batches paced by `stream_pacing`, then cancels the
    // service token; nothing resets it. Each stream ends `UNAVAILABLE "server
    // shutting down"`, which clients retry; this replica has stopped accepting
    // connections (readiness `NOT_SERVING`, GOAWAY sent, listener closed), so
    // the retry reaches another replica. Shutdown never cancels a per-session
    // token: that token means the session is gone.
    let service_shutdown = GracefulShutdown::new();
    let shutdown_service_handle = service_shutdown.clone();
    let stream_pacing = StreamPacing {
        rate_per_sec: config.shutdown_stream_end_rate,
        max_window: Duration::from_secs(u64::from(config.shutdown_stream_end_max_window_secs)),
    };
    let rpc_deadline = Duration::from_millis(config.rpc_deadline_ms);
    let min_termination_grace = minimum_termination_grace(
        Duration::from_secs(config.shutdown_timeout_secs),
        rpc_deadline,
        stream_pacing,
        Duration::from_secs(config.shutdown_overall_budget_secs),
    );
    tracing::info!(
        min_termination_grace_secs = min_termination_grace.as_secs_f64(),
        drain_secs = config.shutdown_timeout_secs,
        stream_end_rate_per_sec = config.shutdown_stream_end_rate,
        stream_end_max_window_secs = config.shutdown_stream_end_max_window_secs,
        rpc_deadline_ms = config.rpc_deadline_ms,
        overall_budget_secs = config.shutdown_overall_budget_secs,
        "graceful shutdown: the pod's termination grace period must be at least \
         min_termination_grace_secs for a shutdown to finish before SIGKILL (the drain, the \
         server's bound after GOAWAY at the longest stream-ending window, the post-server \
         budget and the final flush)"
    );

    // `/scaling/v1` from here on (Step 0a-bis started its server answering
    // 503). Every part it reads now exists: the counters on `metrics`, the
    // admission gate on the Valkey client, the readiness status the health
    // supervisor maintains, and the graceful-shutdown handle, whose paced
    // stream ending having begun the snapshot reports as `shutting_down`.
    {
        let admission = Arc::clone(&valkey.subscriber_admission);
        let registered = scaling_endpoint.register(Arc::new(ServiceScalingSource {
            metrics: Arc::clone(&metrics),
            max_subscriptions: admission.capacity().map_or(0, |c| c as u64),
            admission_waiters: Box::new(move || admission.waiters() as u64),
            health: tonic_health::server::HealthService::from_health_reporter(
                health_reporter.clone(),
            ),
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            shutting_down: {
                let shutdown = service_shutdown.clone();
                Box::new(move || shutdown.begun().is_cancelled())
            },
            process_started,
            catchup_read_concurrency: config.catchup_snapshot_read_concurrency,
            max_concurrent_catchups: valkey
                .catchup_gate
                .limit()
                .map_or(0, |l| u32::try_from(l).unwrap_or(u32::MAX)),
            cpu_seconds: live_edit::observability::process_cpu::cpu_seconds_total,
        }));
        debug_assert!(registered, "the scaling source is registered exactly once");
    }

    let live_edit_service = LiveEditService {
        session_store,
        layer_store,
        ext_value_store,
        session_tokens,
        userinfo_client,
        permissions_client,
        max_layer_size: config.max_layer_size,
        upload_admission,
        valkey: valkey.clone(),
        key_schema: key_schema.clone(),
        presence_router,
        presence_lease_ttl_secs: config.presence_lease_ttl_secs,
        presence_refresh_interval_secs: config.presence_refresh_interval_secs,
        presence_keepalive_interval_ms: config.presence_keepalive_interval_ms,
        presence_cleanup_tracker,
        subscription_tracker,
        shutdown: service_shutdown,
        metrics: Arc::clone(&metrics),
        service_label: config.service_label.clone(),
        catchup_snapshot_enabled: config.catchup_snapshot_enabled,
        session_base_path_cache,
        recovery: recovery_hook,
        session_recovery: session_recovery_hook,
        base_bootstrap: Arc::clone(&base_bootstrap),
        service_token_provider,
        snapshot_flights: SnapshotFlights::new(valkey.clone(), key_schema.clone()),
        #[cfg(feature = "test-fault-injection")]
        catch_up_xrange_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        run_catch_up_inner_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_out_of_window_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_state_read_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_build_turn_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        capacity_budget_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        join_registration_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_replay_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        lease_keeper_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_channel_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        outer_send_deadline_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_chunk_seam_for_test: None,
    };

    let live_service = live_service_server(live_edit_service);

    // ── Step 6b: Metrics + deadline tower layer ───────────────────────────────
    // MetricsLayer wraps the full call stack (including auth) so it measures
    // end-to-end latency and enforces deadlines before auth runs. Constructed
    // at Step 0a (the bootstrap server carries a clone); reused verbatim here.

    // ── Step 7: Graceful shutdown signal ─────────────────────────────────────
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout_secs);
    let shutdown_signal_subscription_tracker = shutdown_subscription_tracker.clone();

    // The full-API server's listener, closable while the server still runs:
    // the drain below closes it, so from then on a connect is refused and
    // retried elsewhere instead of waiting in a backlog nobody accepts, and
    // its open connections are counted for the bound after GOAWAY.
    let (serving_incoming, serving_listener) = closable_incoming(serving_incoming);
    let drain_listener = serving_listener.clone();

    let shutdown_signal = async move {
        // Wait for SIGTERM or SIGINT
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to register SIGTERM handler");

        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM");
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received SIGINT");
            }
        }

        // SDD §5.6 step 1: set health to NOT_SERVING so readiness probes fail.
        tracing::info!("shutdown signal received, setting health to NOT_SERVING");
        health_reporter
            .set_service_status(
                READINESS_SERVICE_NAME,
                tonic_health::ServingStatus::NotServing,
            )
            .await;
        health_reporter
            .set_service_status(
                live_edit::observability::health::OVERALL_SERVICE,
                tonic_health::ServingStatus::NotServing,
            )
            .await;

        // Allow a window for Kubernetes readiness probes to observe
        // NOT_SERVING before the server stops accepting new connections.
        // SDD §5.6 step 2: in-flight unary RPCs complete during this window,
        // and live streams keep serving through it — nothing is cancelled
        // here. `end_streams_after_drain` ends them, paced, once this
        // resolves, starting just behind the server's GOAWAY, so their clients
        // retry against a replica that has stopped accepting connections.
        tracing::info!(
            drain_secs = shutdown_timeout.as_secs(),
            subscriptions = shutdown_signal_subscription_tracker.len(),
            "readiness NOT_SERVING; draining — live streams keep serving until the drain ends, \
             then end UNAVAILABLE, paced, behind the server's GOAWAY"
        );
        tokio::time::sleep(shutdown_timeout).await;

        // Close the listening socket as the drain ends, in the same poll that
        // lets tonic stop accepting: a connect from here on is refused, and
        // the client retries elsewhere.
        drain_listener.close();
        tracing::info!(
            open_connections = drain_listener.open_connections(),
            "drain complete; gRPC listening socket closed"
        );
    };

    // ── Step 8: Start gRPC server (SDD §5.5 step 10) ─────────────────────────
    //
    // Hand the listening socket over from the bootstrap server (Step 0a).
    // Fire its shutdown token, wait a bounded budget for in-flight probe /
    // discovery calls to finish, then abort if it overruns. Aborting is safe
    // and cannot strand the port: `serving_incoming` holds a second
    // descriptor for the same open file description, so the socket stays in
    // `LISTEN` throughout and arriving connections queue in the shared accept
    // backlog instead of being refused.
    bootstrap_shutdown.cancel();
    match tokio::time::timeout(BOOTSTRAP_HANDOFF_BUDGET, &mut bootstrap_server).await {
        Ok(Ok(Ok(()))) => tracing::debug!("bootstrap gRPC server drained"),
        Ok(Ok(Err(e))) => {
            tracing::warn!(error = %e, "bootstrap gRPC server exited with an error")
        }
        Ok(Err(e)) => tracing::warn!(error = %e, "bootstrap gRPC server task failed to join"),
        Err(_) => {
            tracing::warn!(
                budget_secs = BOOTSTRAP_HANDOFF_BUDGET.as_secs(),
                "bootstrap gRPC server did not drain within the handoff budget — aborting it"
            );
            bootstrap_server.abort();
        }
    }

    // Open the readiness gate created at Step 0a. This is the last statement
    // before the full-API server takes the socket: from here the accept
    // backlog is drained by a server that carries `LiveService`, so a
    // `SERVING` readiness verdict is finally truthful. The supervisor is
    // `select!`ing on this receiver's `changed()`, so it re-evaluates now
    // rather than on its next tick. A `send` error is impossible while the
    // supervisor's receiver is alive and is harmless if it is not, so the
    // result is deliberately discarded.
    let _ = api_serving_tx.send(true);

    tracing::info!(addr = %bound_grpc_addr, "gRPC server serving full API");

    // The same HTTP/2 settings as the bootstrap server; see
    // `production_server_builder` for why the TCP options live on the
    // listener instead.
    let serve = production_server_builder()
        .layer(metrics_layer)
        .add_service(live_service)
        .add_service(health_service)
        .add_service(reflection_service)
        // Discovery CapabilitiesService (b.hv2). Bare `.add_service` — auth in
        // this server is attached per-service (inside `live_service_server`,
        // by `LiveServiceServer::with_interceptor`), so the capabilities
        // service is structurally exempt from the auth interceptor, exactly
        // like health and reflection. Discovery-operator scans carry no
        // credentials.
        .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
        // The drain above, then the live streams ended in paced, shuffled
        // batches starting just behind tonic's GOAWAY, then the service
        // shutdown token cancelled (SDD §5.6). tonic returns once every
        // connection has closed.
        .serve_with_incoming_shutdown(
            serving_incoming,
            end_streams_after_drain(
                shutdown_signal,
                shutdown_service_handle.clone(),
                stream_pacing,
                rand::rngs::StdRng::from_rng(&mut rand::rng()),
            ),
        );
    // Bounded after GOAWAY (b.uqx): a server still waiting on a connection
    // past the later of the RPC deadline and the paced ending, plus the close
    // grace, is left to its spawned connection tasks, and the shutdown
    // sequence continues. Either way the shutdown handle is then cancelled,
    // so the drains below never wait on a stream that was never scheduled.
    let served =
        serve_with_post_goaway_bound(serve, &shutdown_service_handle, rpc_deadline, || {
            serving_listener.open_connections()
        })
        .await;
    if let Some(Err(e)) = served {
        tracing::error!(error = %e, "gRPC server error");
        std::process::exit(1);
    }

    // ── Shutdown: bound the entire post-server shutdown sequence (b.3z7 C-2) ──
    //
    // Before this fix each step (layer cleanup + five background-task joins +
    // four sequential Valkey `quit()` calls) had its own `shutdown_timeout_secs`
    // budget, so the worst-case shutdown time was ~10 × `shutdown_timeout_secs`
    // + 4 × 2 s ≈ 58 s — well past the default K8s
    // `terminationGracePeriodSeconds` of 30 s. To bound total cleanup time we
    // (1) trigger all cancellation tokens up front, (2) join the background
    // tasks concurrently via `tokio::join!`, (3) let `ValkeyClient::quit`
    // close all six connection sets (five pools plus the pub/sub client) in
    // parallel internally, and (4) wrap the whole
    // sequence in a single `tokio::time::timeout` sized to fit inside the
    // Helm chart's `terminationGracePeriodSeconds` together with the drain,
    // the bound on the server after GOAWAY and the final flush (80 s by
    // default; the chart refuses a grace below the minimum, which the startup
    // log line reports as `min_termination_grace_secs`).
    let metrics_for_shutdown = Arc::clone(&metrics);
    let shutdown_overall_budget = Duration::from_secs(config.shutdown_overall_budget_secs);
    let shutdown_task_budget = Duration::from_secs(config.shutdown_timeout_secs);
    let layer_store_for_shutdown = shutdown_layer_store;
    let valkey_for_shutdown = valkey.clone();

    let shutdown_sequence = async move {
        tracing::info!("waiting for pending layer cleanup tasks");
        layer_store_for_shutdown
            .wait_for_cleanup(shutdown_task_budget)
            .await;
        tracing::info!("layer cleanup tasks finished");

        // Drain the streaming RPCs' tasks (JoinSession and SubscribeReadLayer)
        // BEFORE we tear down the Valkey pools, concurrently, each bounded by
        // `shutdown_task_budget` so a single stuck task can't monopolise the
        // overall budget. The sequence lives in the library, so the tests
        // that pin it run the same code; see `drain_stream_tasks` for what
        // each drain is for.
        drain_stream_tasks(
            &shutdown_presence_cleanup_tracker,
            &shutdown_subscription_tracker,
            shutdown_task_budget,
        )
        .await;

        // Task t2.fgj.7v.p9: drain the session-checkpoint dispatcher
        // tracker BEFORE the bulk-cancellation block below. The writer's
        // spawned tasks depend only on the Valkey pool and the
        // configured checkpoint adapter; both stay alive until
        // `ValkeyClient::quit()` runs after the join block, so draining
        // here lets in-flight session-checkpoint writes land cleanly.
        // The drain is bounded by `shutdown_task_budget` so a stuck
        // adapter cannot monopolise the overall shutdown window.
        shutdown_session_checkpoint_tracker.close();
        let session_checkpoint_drain = tokio::time::timeout(
            shutdown_task_budget,
            shutdown_session_checkpoint_tracker.wait(),
        )
        .await;
        match session_checkpoint_drain {
            Ok(()) => tracing::info!("session-checkpoint writes drained"),
            Err(_) => tracing::warn!(
                pending = shutdown_session_checkpoint_tracker.len(),
                "timed out waiting for session-checkpoint writes; some lifecycle events may not be reflected in the checkpoint store"
            ),
        }

        // Fire all cancellation tokens so background tasks begin unwinding in
        // parallel — the join below is where we actually wait for them.
        gc_cancel.cancel();
        orphan_sweep_cancel.cancel();
        // b.yda: a mid-pass cancel stops the reconciler at the next drain-round
        // boundary and repairs nothing further. Safe to interrupt anywhere: each
        // repair member is independently owed, and the candidate ZSET is durable
        // with first-observation scores intact, so the next process's first pass
        // resumes exactly where this one stopped.
        ext_reconcile_cancel.cancel();
        // The external-value backup repair: a mid-pass cancel abandons the
        // pass, with no gauge published. Safe anywhere: every check, upload
        // and confirmation it finished stays done, and a value it did not
        // reach stays unconfirmed for the next leader's pass. An abandoned
        // upload's byte charge returns when its bytes drop, which on LocalDisk
        // is when its write ends: that write finishes on its own blocking
        // task, and the runtime waits for it at exit.
        ext_backup_repair_cancel.cancel();
        // b.yuo, b.qhk: the store-memory re-read (release budget and
        // fullness gauge), detached (no join below): it holds nothing to
        // drain.
        ext_release_budget_refresh_cancel.cancel();
        background_tasks_cancel.cancel();
        health_supervisor_cancel.cancel();
        if let Some(cancel) = &checkpoint_loop_cancel {
            cancel.cancel();
        }
        // SDD SR-3.1: fire the ext-mirror workers' shared cancel ahead
        // of the join below so each worker exits its `select!` and runs
        // the best-effort `DRAIN_BUDGET`-bounded drain before returning.
        if let Some(cancel) = &ext_mirror_cancel {
            cancel.cancel();
        }
        // Epic t1.fgj.dp: cancel the service-token polling task. The
        // task observes `cancel.cancelled()` in its `select!` and exits
        // within the existing shutdown budget. No JoinHandle is awaited
        // here — `tokio::spawn`'d handle is detached because the polling
        // task carries no mutable state operators care about preserving
        // across shutdown (the cached token is in-process only and is
        // discarded with the rest of the service state).
        service_token_cancel.cancel();
        // Epic 3 t3.tr5.jv: cancel the Redis-scoped access-token-file provider's
        // polling task, mirroring the storage-api provider cancel above. A
        // no-op in non-access-token-file modes (the token never existed, but the
        // `CancellationToken` is always constructed so this call is
        // unconditional and cheap).
        valkey_token_cancel.cancel();

        // Join the background tasks concurrently. Each individual join still
        // carries a per-task budget so a single runaway task can't monopolise
        // the whole shutdown window.
        let (
            gc_res,
            orphan_res,
            presence_sub_res,
            sweeper_res,
            session_del_res,
            health_sup_res,
            base_path_cache_res,
            session_tokens_sweep_res,
        ) = tokio::join!(
            tokio::time::timeout(shutdown_task_budget, gc_handle),
            tokio::time::timeout(shutdown_task_budget, orphan_sweep_handle),
            tokio::time::timeout(shutdown_task_budget, presence_subscriber_handle),
            tokio::time::timeout(shutdown_task_budget, sweeper_handle),
            tokio::time::timeout(shutdown_task_budget, session_delete_subscriber_handle),
            tokio::time::timeout(shutdown_task_budget, health_supervisor_handle),
            tokio::time::timeout(shutdown_task_budget, session_base_path_cache_sweeper_handle),
            tokio::time::timeout(shutdown_task_budget, session_tokens_sweep_handle),
        );

        report_background_task_result(&metrics_for_shutdown, "gc", gc_res);
        report_background_task_result(&metrics_for_shutdown, "orphan_sweep", orphan_res);
        report_background_task_result(
            &metrics_for_shutdown,
            "presence_subscriber",
            presence_sub_res,
        );
        report_background_task_result(&metrics_for_shutdown, "presence_sweeper", sweeper_res);
        report_background_task_result(&metrics_for_shutdown, "health_supervisor", health_sup_res);
        report_background_task_result(
            &metrics_for_shutdown,
            "session_delete_subscriber",
            session_del_res,
        );
        report_background_task_result(
            &metrics_for_shutdown,
            "session_base_path_cache_sweeper",
            base_path_cache_res,
        );
        report_background_task_result(
            &metrics_for_shutdown,
            "session_tokens_sweeper",
            session_tokens_sweep_res,
        );

        // b.yda: drain the ref_count reconciler only if it was spawned. Same
        // `Option<JoinHandle>` shape and same reason as the checkpoint loop
        // below — the `interval = 0` off switch would otherwise force an
        // always-Some handle into the `tokio::join!` above.
        if let Some(handle) = ext_reconcile_handle {
            let ext_reconcile_res = tokio::time::timeout(shutdown_task_budget, handle).await;
            report_background_task_result(
                &metrics_for_shutdown,
                "ext_ref_reconcile",
                ext_reconcile_res,
            );
        }

        // The backup repair, spawned only with the ext-mirror (Step 1e-bis).
        if let Some(handle) = ext_backup_repair_handle {
            let ext_backup_repair_res = tokio::time::timeout(shutdown_task_budget, handle).await;
            report_background_task_result(
                &metrics_for_shutdown,
                "ext_backup_repair",
                ext_backup_repair_res,
            );
        }

        // Drain the checkpoint loop only if it was spawned. Folding it
        // into the `tokio::join!` above would require an always-Some
        // handle; the `Option` shape lets `enabled=false` builds skip
        // the await without restructuring the join block.
        if let Some(handle) = checkpoint_loop_handle {
            let checkpoint_loop_res = tokio::time::timeout(shutdown_task_budget, handle).await;
            report_background_task_result(
                &metrics_for_shutdown,
                "checkpoint_loop",
                checkpoint_loop_res,
            );
        }

        // SDD SR-3.1: drain the ext-mirror worker pool. The shared
        // cancel fired above, so each worker is already running its
        // `DRAIN_BUDGET`-bounded best-effort drain; we await each
        // handle here under the same per-task budget the other
        // background tasks use. When the last worker exits, the receiver
        // it held discards whatever is still queued, counted as
        // `reason="shutdown"` with one WARN (SDD §22.17); a worker abandoned
        // at its timeout keeps the receiver, and the queue, until exit. One
        // `report_background_task_result`
        // call per worker (rather than aggregating) matches the
        // surrounding per-task reporting style — workers share the
        // `"ext_mirror_worker"` label, and operators disambiguate
        // misbehaving workers via the `background_task_panics_total`
        // counter delta the helper emits per panic.
        // Empty on the skip path (`checkpoint_enabled=false` or
        // adapter=`Disabled`), so this loop is a no-op
        // and matches Acceptance Criterion 1.
        for handle in ext_mirror_worker_handles {
            let res = tokio::time::timeout(shutdown_task_budget, handle).await;
            report_background_task_result(&metrics_for_shutdown, "ext_mirror_worker", res);
        }

        tracing::info!("closing Valkey connections");
        valkey_for_shutdown.quit().await;
    };

    match tokio::time::timeout(shutdown_overall_budget, shutdown_sequence).await {
        Ok(()) => tracing::info!("shutdown complete"),
        Err(_) => tracing::warn!(
            overall_budget_secs = config.shutdown_overall_budget_secs,
            "shutdown overall budget exceeded; exiting anyway to fit terminationGracePeriodSeconds"
        ),
    }

    // Abort the `/metrics` axum server BEFORE the post-budget provider flush,
    // on every path — including the budget-timeout path, where the budgeted
    // `shutdown_sequence` future is dropped mid-execution and its trailing
    // abort would otherwise never run. `metrics.shutdown()` below performs a
    // final export on the meter provider; if the server were still live a
    // concurrent scrape would call `PrometheusExporter::export()` on a provider
    // mid-shutdown (logs an error, returns HTTP 500). `abort()` is idempotent,
    // so this is safe regardless of whether the budget elapsed.
    tracing::info!("aborting metrics server");
    metrics_task.abort();
    let _ = metrics_task.await;

    // Final OTLP flush of the metrics meter provider and the tracer provider
    // runs AFTER the budgeted `shutdown_sequence` above, so it is NOT covered by
    // `shutdown_overall_budget` (30 s default). It is the last term of the
    // minimum termination grace (`minimum_termination_grace`, logged at
    // startup), so this combined flush must be bounded — neither
    // `metrics.shutdown()` (a final OTLP export) nor `shutdown_tracing()` (a
    // 5 s internal batch-processor join) is self-limiting enough against an
    // unreachable collector. Cap the combined flush at
    // `POST_BUDGET_FLUSH_BOUND` (6 s). A reachable collector finishes well
    // inside this ceiling: each OTLP metrics export request is capped at 3 s
    // (`otlp::OTLP_EXPORT_TIMEOUT`), the TLS handshake of an `https://`
    // endpoint, on either exporter, at the same 3 s, and the tracer provider
    // has its own 5 s internal timeout. TCP connect has no bound of its own:
    // only this 6 s cap limits it. The cap only bites when the collector is
    // unreachable, in which case we log and exit rather than risk SIGKILL.
    let post_budget_flush = POST_BUDGET_FLUSH_BOUND;
    let flush = async {
        // Flush + shut down the metrics meter provider so the OTLP
        // `PeriodicReader` performs a final export before exit (OMPE-98174).
        // Offloaded via `spawn_blocking` for the same reason as
        // `shutdown_tracing` — the reader shutdown joins a background worker and
        // must not block a tokio worker.
        {
            let metrics = Arc::clone(&metrics);
            let _ = tokio::task::spawn_blocking(move || metrics.shutdown()).await;
        }

        shutdown_tracing().await;
    };

    if tokio::time::timeout(post_budget_flush, flush)
        .await
        .is_err()
    {
        tracing::warn!(
            post_budget_flush_secs = post_budget_flush.as_secs(),
            "post-budget metrics/tracing flush exceeded its bound; exiting anyway to fit terminationGracePeriodSeconds"
        );
    }
}

/// Classify the outcome of one background-task join during shutdown, emit the
/// appropriate log line, and tick `background_task_panics_total{task}` when
/// the task panicked (b.3z7 H-obs-3). Separating this into a helper keeps the
/// `tokio::join!` block readable.
fn report_background_task_result(
    metrics: &live_edit::observability::metrics::Metrics,
    task: &str,
    result: Result<Result<(), tokio::task::JoinError>, tokio::time::error::Elapsed>,
) {
    match result {
        Ok(Ok(())) => tracing::info!(task = task, "background task stopped cleanly"),
        Ok(Err(join_err)) => {
            if join_err.is_panic() {
                metrics.inc_background_task_panic(task);
            }
            tracing::error!(task = task, error = %join_err, "background task failed during shutdown");
        }
        Err(_elapsed) => {
            tracing::warn!(
                task = task,
                "background task did not stop within per-task shutdown budget"
            );
        }
    }
}
