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

// Shared test helpers for integration tests.
//
// Five test binaries (`tests/integration/main.rs`, `tests/misc/main.rs`,
// `tests/upload_memory_envelope/main.rs` and
// `tests/release_stack_guard/main.rs`, each through
// `#[path = "../common/mod.rs"] mod common;`, and `tests/proptests.rs`
// through `#[path = "common/mod.rs"]`) include this file. Different
// binaries use different subsets, so unused items in any one consumer
// produce noisy dead-code warnings — silence them at the module level.
#![allow(dead_code)]

pub mod owned_delta;
pub mod refusal_detail;
pub mod sibling_model;
pub mod upload_stream;

use std::net::SocketAddr;
use std::time::Duration;

use live_edit::storage::leader_lease::LeaderIdentity;

// ── Leader identity fixtures ─────────────────────────────────────────────────

/// The fixed `instance` of [`test_leader`] identities.
pub const TEST_LEADER_INSTANCE: &str = "integration-test-instance";

/// A [`LeaderIdentity`] for a fixture replica named `service_label`, with a
/// fixed instance. The same label always yields the same lock value, so a
/// fixture that respawns a loop "as the same replica" still recognises its own
/// lock, and two different labels are still two different leaders. A test
/// that needs two same-labelled processes builds them with
/// [`LeaderIdentity::new`] and distinct instances instead.
pub fn test_leader(service_label: &str) -> LeaderIdentity {
    LeaderIdentity::new(service_label, TEST_LEADER_INSTANCE)
}

// ── SR-2.x verbatim error wording ────────────────────────────────────────────
//
// These constants pin the user-facing wording of the SR-2.x JwtUserinfoClient
// error contract. They MUST stay character-for-character identical to the
// strings produced by `src/auth/userinfo_jwt.rs::JwtUserinfoClient::get_current_user`
// — drift here is a Task 2 contract violation. Lifting them into the shared
// `common` module (vs. per-test-binary or per-file duplication) means a reword
// surfaces as a single compile-time fix across both the `integration` and
// `misc` test binaries rather than as silent no-ops in sibling tests.

/// Verbatim SR-2.4 wording — must stay in lockstep with the production
/// `JwtUserinfoClient::get_current_user` opaque-token error string.
pub const SR_2_4_OPAQUE_TOKEN_MSG: &str = "opaque bearer token requires the userinfo service; deploy the userinfo service or use JWT-format bearer tokens";

/// Verbatim SR-2.5 wording — must stay in lockstep with the production
/// `JwtUserinfoClient::get_current_user` missing-display-name-claim error
/// string at `src/auth/userinfo_jwt.rs:137`. Drift here is a Task 2 contract
/// violation; the cross-file pin converts a silent reword into a test-failure
/// signal even though the wire never sees the string.
pub const SR_2_5_MISSING_CLAIM_MSG: &str =
    "JWT payload contains no display-name claim (expected name / preferred_username / sub / oid)";

// ── update_layer header-anomaly sentinels (b.f1v) ────────────────────────────
//
// `src/storage/layer.rs` keeps both literals as module-PRIVATE constants — the
// point of b.f1v was that no production caller may key behaviour off either
// string, so neither is nameable from a test crate. Mirrored here (once, not
// per-file) because both are still cross-backend *parity* contracts: the Lua
// fast path and the WATCH/MULTI/EXEC slow path must produce byte-identical
// messages for the same header anomaly (SDD §7.5), and four test modules assert
// that from both sides. Drift in `src/storage/layer.rs` surfaces as a single
// failing assertion rather than as a quietly weakened `contains(...)`.

/// Mirror of the private `storage::layer::EMPTY_HEADER_SENTINEL` — the message
/// carried by [`live_edit::error::ServiceError::NotFound`] when an object's
/// `obj:hdr` is **absent**. Produced by the txn path's empty-`HGETALL` guard,
/// the Lua path's `NOT_FOUND:` reply mapping, and
/// `LayerStore::disambiguate_stream_collision`'s `header_deleted` verdict.
pub const EMPTY_HEADER_SENTINEL_MSG: &str = "object no longer exists (header deleted)";

/// Mirror of the private `storage::layer::PARTIAL_HEADER_SENTINEL` — the message
/// carried by [`live_edit::error::ServiceError::FailedPrecondition`] when an
/// object's `obj:hdr` is **present** but its `seq` field is missing or
/// non-numeric. Deliberately a different variant from
/// [`EMPTY_HEADER_SENTINEL_MSG`]: a corrupt-but-present object must not be
/// routed into checkpoint recovery **from the commit path**, because a restore
/// there would replace whatever the surviving keyspace still holds with the
/// checkpoint's state — on a request that asked to append to it.
///
/// b.f4t reworded the second half of that reason. It used to read "…whose
/// presence probe is a bare `EXISTS` on the header and would report a restore
/// that never ran", which is no longer true: the probe is now structural
/// (`seq` AND `created_by`), so a partial header reaching the hook drives a
/// REAL restore that repairs it — which is exactly what `StatLayer` and
/// `SubscribeReadLayer` now do with one. The commit path's refusal survives as
/// a scoping decision, not as a workaround for a weak probe.
pub const PARTIAL_HEADER_SENTINEL_MSG: &str = "object header partial (seq missing or malformed)";

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use fred::interfaces::ClientLike;
use fred::types::scan::ScanType;
use futures::StreamExt;
use tokio::sync::oneshot;
use tonic::transport::Server;
use tonic_health::server::HealthReporter;

use live_edit::config::{AuthProtocol, Config, LogFormat, UpdateLayerBackend};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;

/// Proto types for use in integration tests.
pub mod proto {
    tonic::include_proto!("nvidia.omniverse.live.v1alpha");
}

// ── JWT builder ──────────────────────────────────────────────────────────────

/// Build an `alg=none` JWT carrying `claims` as the payload — the integration-
/// test sibling of `src/auth/userinfo_jwt.rs::tests::make_jwt`. The inline
/// production-side helper lives in `#[cfg(test)]` inside the production crate
/// and is invisible to integration-test binaries, while this one is reachable
/// from sibling per-RPC tests via the shared `common` module (re-exported by
/// `tests/integration/grpc/helpers.rs` as `pub use crate::common::make_jwt;`
/// to keep `use super::helpers::*;` call sites unchanged).
///
/// Encoding follows SDD §Test Fixtures: fixed `{"alg":"none"}` header,
/// `URL_SAFE_NO_PAD` base64url over `claims.to_string().as_bytes()` for the
/// payload, empty signature segment, joined with `.`.
pub fn make_jwt(claims: &serde_json::Value) -> String {
    let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
    let payload = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
    let signature = URL_SAFE_NO_PAD.encode(b"");
    format!("{header}.{payload}.{signature}")
}

/// Canonical `Config` used by integration tests (b.ohv item 8).
///
/// Returns a fully-populated [`Config`] with values that are sane for
/// integration tests: a fast Valkey URL (read from `TEST_VALKEY_URL`, with
/// `LIVE_EDIT_VALKEY_ADDR` as legacy fallback, defaulting to
/// `redis://127.0.0.1:6379`), small pools, tight timeouts, `authz_enabled =
/// false`, `userinfo_service_enabled = true` (preserves the pre-b.g3z
/// "external userinfo backend assumed" semantics so tests that exercise
/// REST/gRPC userinfo dispatch keep working without rewrites), the Lua
/// update-layer backend, and the production defaults for everything else.
/// Callers override only the fields their test cares about — e.g.
/// `let mut cfg = common::test_config(); cfg.valkey_pool_size = 10;`.
///
/// **Why this lives here and not in `impl Config`.** The constructor is a
/// test-only convenience that hard-codes a Valkey URL + sensible test
/// defaults. Putting it on `impl Config` would either (a) leak that
/// test-shaped knowledge into the production binary or (b) require a
/// `#[cfg(test)]` block that integration-test crates don't compile under.
/// Living on `tests/common/mod.rs` keeps the helper in the same crate as
/// the integration tests that consume it and out of the production
/// binary. Every field is set explicitly so adding a new `Config` field
/// upstream surfaces a compile error here rather than silently picking up
/// `Default::default()` semantics that may not match what the field
/// actually means in production.
pub fn test_config() -> Config {
    let valkey_url = std::env::var("TEST_VALKEY_URL")
        .or_else(|_| std::env::var("LIVE_EDIT_VALKEY_ADDR"))
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: 4,
        subscriber_pool_size: 4,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
        subscriber_max_concurrent_catchups: 64,
        subscriber_admission_max_waiters: None,
        maintenance_pool_size: 1,
        catchup_snapshot_read_concurrency:
            live_edit::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
        catchup_fenced_read_concurrency: live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
        catchup_build_memory_budget_mib: live_edit::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
        snapshot_pool_size: 1,
        snapshot_atomic_read_max_items: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        snapshot_atomic_read_max_bytes: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
        structural_cache_max_mib: live_edit::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
        structural_cache_audit_rate: live_edit::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
        structural_cache_audit_above_ceiling: false,
        valkey_connect_timeout_secs: 5,
        valkey_command_timeout_ms: 5000,
        valkey_reconnect_max_delay_ms: 30_000,
        valkey_auth_mode: live_edit::config::ValkeyAuthMode::Static,
        valkey_token_file: None,
        valkey_token_username: None,
        valkey_token_refresh_lead_secs: 180,
        valkey_iam_username: None,
        valkey_iam_region: None,
        valkey_iam_cluster_host: None,
        valkey_iam_service: live_edit::config::ValkeyIamService::MemoryDb,
        storage_service_endpoint: Some("http://unused.test".to_string()),
        auth_timeout_ms: 1000,
        auth_cache_ttl_secs: 60,
        auth_cache_negative_ttl_secs: 30,
        auth_userinfo_negative_ttl_secs: 5,
        auth_cache_max_entries: 1_000,
        userinfo_endpoint: Some("http://unused.test".to_string()),
        max_layer_size: 1_048_576,
        upload_memory_budget_mib:
            live_edit::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
        upload_admission_wait_ms:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
        upload_admission_max_waiters:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
        layer_stream_maxlen: 4096,
        layer_stream_max_bytes: live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
        presence_broadcast_capacity: 4096,
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        shutdown_timeout_secs: 1,
        shutdown_overall_budget_secs: 5,
        shutdown_stream_end_rate: 25,
        shutdown_stream_end_max_window_secs: 20,
        log_level: "warn".to_string(),
        log_format: LogFormat::Json,
        trace_filter: live_edit::config::DEFAULT_TRACE_FILTER.to_string(),
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
        auth_protocol: AuthProtocol::Rest,
        update_layer_retry_max_attempts: 128,
        update_layer_per_object_concurrency: 16,
        update_layer_backend: UpdateLayerBackend::Lua,
        ext_value_verify: live_edit::config::ExtValueVerifyMode::Enforce,
        session_retry_max_attempts: 32,
        session_retry_initial_delay_ms: 1,
        // Keep the production default so the test path matches operator
        // semantics. Tests that need a faster sweep override this field
        // directly.
        pod_memory: live_edit::config::PodMemoryPlan::reference(),
        session_tokens_sweep_interval_secs: 60,
        checkpoint_enabled: false,
        checkpoint_adapter: live_edit::checkpoint::CheckpointAdapter::Disabled,
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
    }
}

// ── Cluster-aware keyspace SCAN ──────────────────────────────────────────────

/// Every live key matching `pattern`.
///
/// SCAN, never KEYS: KEYS is O(keyspace) and blocks the server for its whole
/// duration, which on a shared test Valkey shows up as unrelated timeouts in
/// sibling tests.
///
/// The `is_clustered()` fork is load-bearing, not defensive boilerplate.
/// `scan_buffered` walks only the node the pooled client happens to be
/// attached to, so under the `test-cluster` CI job it silently misses every
/// key hashed to another shard — a *false-negative* shape (fewer keys than
/// really exist), which is the direction that turns leak assertions into
/// vacuous passes. `scan_cluster_buffered` fans out across all primaries.
///
/// A SCAN error panics rather than being dropped: a caller that silently
/// skipped errored pages would under-report survivors for exactly the same
/// reason, defeating the assertions built on top of this.
pub async fn scan_keys(client: &ValkeyClient, pattern: &str) -> Vec<fred::types::Key> {
    let node = client.pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if node.is_clustered() {
        Box::pin(node.scan_cluster_buffered(pattern, Some(100), None::<ScanType>))
    } else {
        Box::pin(node.scan_buffered(pattern, Some(100), None::<ScanType>))
    };

    let mut found = Vec::new();
    while let Some(item) = stream.next().await {
        found.push(item.expect("SCAN must succeed"));
    }
    found
}

/// The glob matching every per-object key an object owns:
/// `{prefix}{object_id}:obj:*`. Kept next to [`scan_keys`] so callers that
/// SCAN the object namespace cannot drift on the brace placement (the
/// `{object_id}` hash tag is literal braces in the key, not interpolation).
pub fn obj_key_pattern(prefix: &str, object_id: u64) -> String {
    format!("{prefix}{{{object_id}}}:obj:*")
}

/// Every live `le:{prefix}{object_id}:obj:*` key. See [`scan_keys`] for the
/// cluster-mode caveat this wraps.
pub async fn scan_object_keys(
    client: &ValkeyClient,
    prefix: &str,
    object_id: u64,
) -> Vec<fred::types::Key> {
    scan_keys(client, &obj_key_pattern(prefix, object_id)).await
}

/// Read a numeric `le:{object_id}:obj:hdr` column, `None` when the field is
/// absent.
///
/// Absence and a present-but-zero value are DIFFERENT header states — the
/// checkpoint-cursor columns encode "never checkpointed" / "no restore bump
/// outstanding" as absence and rely on downstream readers preserving the
/// distinction — so the return type is `Option<u64>` rather than a
/// `unwrap_or(0)` fold. A present-but-non-numeric value panics: every writer
/// of these columns emits decimal `u64`, so garbage means the test observed a
/// corrupted header and the assertion built on top would be meaningless.
///
/// Gated on `test-support` because the `TestValkey` fixture it takes is: the
/// `misc` test binary shares this module but never touches Valkey, and must
/// keep compiling without the feature.
#[cfg(feature = "test-support")]
pub async fn read_header_u64(
    tv: &live_edit::storage::test_support::TestValkey,
    object_id: u64,
    field: &str,
) -> Option<u64> {
    use fred::interfaces::HashesInterface;

    let key = tv.key_schema().obj_header(&object_id.to_string());
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(&key, field)
        .await
        .unwrap_or_else(|e| panic!("HGET {field} must succeed: {e}"));
    raw.map(|s| {
        s.parse::<u64>()
            .unwrap_or_else(|e| panic!("obj:hdr {field} must be numeric; got {s:?} ({e})"))
    })
}

// ── Metrics constructor (OTLP-push-safe) ─────────────────────────────────────

/// Env var names for the OTLP-metrics-push gate (mirrors the unit-test
/// constants in `src/observability/metrics.rs`).
const METRICS_OTLP_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
const GENERIC_OTLP_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// Construct a [`Metrics`] with both OTLP endpoint env vars pinned to `None`.
///
/// The OMPE-98174 observability migration made `Metrics::new` consult
/// `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` at construction and attach a live OTLP
/// `PeriodicReader` (a real push to a collector) when it is set. Integration
/// tests only ever want the scrape reader; a leaked endpoint env var (from CI
/// or a sibling test) would otherwise silently attach a live OTLP push during
/// the suite — the scrape assertions still pass, so the hazard is invisible.
///
/// Pinning BOTH `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` and the generic
/// `OTEL_EXPORTER_OTLP_ENDPOINT` to `None` inside a `temp_env` guard mirrors
/// the unit tests' `metrics_scrape_only` and the tracing-side guards (b.5yt).
/// Every integration call site routes through here rather than calling
/// `Metrics::new` directly.
pub fn metrics_scrape_only(label: &str) -> Metrics {
    temp_env::with_vars(
        [
            (METRICS_OTLP_ENDPOINT_VAR, None::<&str>),
            (GENERIC_OTLP_ENDPOINT_VAR, None::<&str>),
        ],
        || Metrics::new(label),
    )
}

// ── First-base bootstrap test builder ────────────────────────────────────────

/// Build a `BaseBootstrap` with the SR-7.1 gate resolved to `false`.
///
/// `spawn_creation_base` short-circuits before allocating, spawning, issuing any
/// Valkey command, or touching the metrics handle in this state, so every field
/// below is inert — the `InMemoryCheckpointStore` is a placeholder that is never
/// called, mirroring [`crate::integration::grpc::helpers::build_disabled_recovery_hook`]'s
/// use of the same store for the same reason. Plumbing this into every
/// `LiveEditService` construction keeps the field present without coupling
/// unrelated tests to the checkpoint subsystem.
///
/// Lives in the shared `common` module (rather than either `helpers.rs`) because
/// `LiveEditService` literals exist in both the `grpc` and `checkpoint`
/// integration trees.
pub fn build_disabled_base_bootstrap(
    client: ValkeyClient,
    keys: live_edit::storage::keys::KeySchema,
    metrics: std::sync::Arc<Metrics>,
) -> std::sync::Arc<live_edit::checkpoint::BaseBootstrap> {
    let store: std::sync::Arc<dyn live_edit::checkpoint::CheckpointStore> =
        std::sync::Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    std::sync::Arc::new(live_edit::checkpoint::BaseBootstrap::new(
        store,
        client,
        keys,
        tokio_util::task::TaskTracker::new(),
        metrics,
        "test".to_string(),
        false,
    ))
}

// ── Health-supervisor readiness pre-seed ─────────────────────────────────────

/// Publish `SERVING` on `readiness_service_name` *before* a health supervisor
/// is spawned, so that a later `NOT_SERVING` observation is necessarily the
/// supervisor's own verdict.
///
/// Needed since b.rgs: `create_health_service` now registers readiness as
/// `NOT_SERVING` (the socket answers before any dependency is wired), which
/// makes a bare "readiness is NOT_SERVING" assertion vacuous — it holds from
/// t=0 and would pass against a supervisor that never ran a tick, or one whose
/// dependency classifier was deleted outright. Seeding `SERVING` first restores
/// the assertion's teeth: the only writer that can move it back down is the
/// supervisor, so the test once again observes a real `SERVING → NOT_SERVING`
/// transition.
///
/// The seeded `SERVING` is a *detector*, not a claim that production ever
/// reaches that state with the dependency in question already broken.
///
/// Shared by every supervisor test binary in this crate. `src/observability/
/// health.rs` keeps its own private copy: `#[cfg(test)]` unit tests inside the
/// production crate cannot reach this module across the crate boundary.
pub async fn preseed_readiness_serving(reporter: &HealthReporter, readiness_service_name: &str) {
    reporter
        .set_service_status(readiness_service_name, tonic_health::ServingStatus::Serving)
        .await;
}

/// A handle that shuts down the test server when dropped or explicitly triggered.
pub struct TestServerHandle {
    /// The local address the server is bound to.
    pub addr: SocketAddr,
    /// A cloned health reporter for driving health transitions in tests.
    pub health_reporter: HealthReporter,
    shutdown_tx: Option<oneshot::Sender<()>>,
    server_task: Option<tokio::task::JoinHandle<()>>,
}

impl TestServerHandle {
    /// Trigger graceful shutdown of the test server and wait for it to stop (max 5 s).
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.server_task.take() {
            tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .expect("server did not stop within 5 seconds")
                .expect("server task panicked");
        }
    }
}

/// Spawn a test gRPC server on a random port (`127.0.0.1:0`).
///
/// Registers both `""` and `"nvidia.omniverse.live.v1alpha.LiveService"` as SERVING.
/// Returns a handle with the bound address and a health reporter.
pub async fn spawn_test_server() -> TestServerHandle {
    use proto::live_service_server::LiveServiceServer;
    use std::pin::Pin;
    use tonic::{Request, Response, Status, Streaming};
    use tonic_health::ServingStatus;

    // ── Inline minimal LiveService implementation ────────────────────────────
    // Integration tests cannot import from the binary crate (no lib.rs).
    // We define just enough to stand up a real tonic server.
    #[derive(Debug, Default)]
    struct TestService;

    #[tonic::async_trait]
    impl proto::live_service_server::LiveService for TestService {
        async fn ping(
            &self,
            request: Request<proto::PingRequest>,
        ) -> Result<Response<proto::PingResponse>, Status> {
            use std::time::{SystemTime, UNIX_EPOCH};
            let client_ts = request
                .into_inner()
                .client_timestamp_microseconds_since_epoch;
            let server_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            // This stand-in implements neither capability.
            Ok(Response::new(proto::PingResponse {
                client_timestamp_microseconds_since_epoch: client_ts,
                server_timestamp_microseconds_since_epoch: server_ts,
                operation_ids: false,
                chained_writes: false,
            }))
        }

        async fn create_session(
            &self,
            _: Request<proto::CreateSessionRequest>,
        ) -> Result<Response<proto::CreateSessionResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn delete_session(
            &self,
            _: Request<proto::DeleteSessionRequest>,
        ) -> Result<Response<proto::DeleteSessionResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn list_sessions(
            &self,
            _: Request<proto::ListSessionsRequest>,
        ) -> Result<Response<proto::ListSessionsResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        type JoinSessionStream = Pin<
            Box<
                dyn futures::Stream<Item = Result<proto::JoinSessionResponse, Status>>
                    + Send
                    + 'static,
            >,
        >;

        async fn join_session(
            &self,
            _: Request<Streaming<proto::JoinSessionRequest>>,
        ) -> Result<Response<Self::JoinSessionStream>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn create_layer(
            &self,
            _: Request<Streaming<proto::CreateLayerRequest>>,
        ) -> Result<Response<proto::CreateLayerResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn update_layer(
            &self,
            _: Request<Streaming<proto::UpdateLayerRequest>>,
        ) -> Result<Response<proto::UpdateLayerResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn delete_layer(
            &self,
            _: Request<proto::DeleteLayerRequest>,
        ) -> Result<Response<proto::DeleteLayerResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        async fn stat_layer(
            &self,
            _: Request<proto::StatLayerRequest>,
        ) -> Result<Response<proto::StatLayerResponse>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }

        type SubscribeReadLayerStream = Pin<
            Box<
                dyn futures::Stream<Item = Result<proto::SubscribeReadLayerResponse, Status>>
                    + Send
                    + 'static,
            >,
        >;

        async fn subscribe_read_layer(
            &self,
            _: Request<proto::SubscribeReadLayerRequest>,
        ) -> Result<Response<Self::SubscribeReadLayerStream>, Status> {
            Err(Status::unimplemented("not yet implemented"))
        }
    }

    // ── Health service setup ──────────────────────────────────────────────────
    let (health_reporter, health_service) = tonic_health::server::health_reporter();

    health_reporter
        .set_service_status("", ServingStatus::Serving)
        .await;
    health_reporter
        .set_serving::<LiveServiceServer<TestService>>()
        .await;

    let reporter_for_handle = health_reporter.clone();

    // ── Bind to random port ───────────────────────────────────────────────────
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");

    let live_service = LiveServiceServer::new(TestService);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    // Listener-ready signal (b.ohv item 5). Replaces a 50 ms blanket sleep
    // that was load-bearing on slow CI machines but unnecessary on fast
    // ones. The server task fires `ready_tx` immediately before calling
    // `serve_with_incoming_shutdown`; once we `await` the receiver here we
    // know the tonic Server has finished its sync setup and the underlying
    // TcpListener is being driven by the spawned task — any subsequent
    // client connect targets a ready accept loop.
    let (ready_tx, ready_rx) = oneshot::channel::<()>();

    let server_task = tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        // Tell the parent the listener stream is wired up. We send before
        // the actual `serve_with_incoming_shutdown` call returns its future
        // — the future drives the listener's accept loop, but the bound
        // TcpListener already accepts SYNs at the kernel level the moment
        // `bind()` returned on the parent. Firing the signal here makes
        // the wait deterministic; any send-error means the parent dropped
        // the receiver (the test is shutting down) and we proceed regardless.
        let _ = ready_tx.send(());
        Server::builder()
            .add_service(live_service)
            .add_service(health_service)
            .serve_with_incoming_shutdown(incoming, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("test server error");
    });

    // Wait for the ready signal — bounded so a buggy server task can't
    // hang test runs indefinitely. 5 s is generous against the historic
    // 50 ms sleep; a real timeout here indicates the task itself failed
    // to spawn.
    let _ = tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("test server spawn did not signal readiness within 5s");

    TestServerHandle {
        addr,
        health_reporter: reporter_for_handle,
        shutdown_tx: Some(shutdown_tx),
        server_task: Some(server_task),
    }
}

// ── Layer-recovery test helpers (b.f4t sub-task 5) ───────────────────────────
//
// Promoted from three near-identical private copies that had already begun to
// diverge: `tests/integration/checkpoint/recovery_hook_test.rs` (the
// originals), `tests/integration/grpc/update_layer_recovery_arm_test.rs`
// (re-stated during b.f1v because the checkpoint tree was another lane's), and
// the newer refusal / residue suites. The `recovery_metric` copies had
// *already* drifted — the grpc one pinned `scope="layer"`, the checkpoint one
// did not — which is exactly the failure mode a shared copy prevents: a metric
// helper reading a label cell nothing writes returns 0 and every assertion
// built on it passes vacuously.
//
// Gated on `test-support` following `read_header_u64` above: the `misc` test
// binary shares this module, never touches Valkey, and must keep compiling
// without the feature.
//
// Also NOT promoted, and unfixable rather than a judgement call: the
// `build_empty_reset_ompf` helper in `src/checkpoint/recovery.rs`'s
// `#[cfg(test)]` module, a duplicate of [`empty_reset_ompf`] below — in-crate
// unit tests cannot reach `tests/common`, so the two copies must be kept in
// sync by hand.
//
// Deliberately NOT promoted: `replace_recovery`. The three copies take
// genuinely different arguments (an explicit `checkpoint_enabled` flag in one,
// an implicit `true` in another) and one of them chains the b.sxc
// post-restore-NotFound latch. Unifying those would mean a builder, which is
// more surface than the duplication costs.

/// Service label every helper below pins.
///
/// Load-bearing rather than cosmetic: [`recovery_metric`] reads a
/// `{service=...}` label cell, and [`replace_metrics`] + `common::test_config`
/// are what write it. A helper reading one label while the hook writes another
/// returns 0 forever and makes every assertion built on it vacuous — the
/// failure this shared copy exists to make impossible.
///
/// "Impossible" is meant literally, and it takes the alias to earn the word.
/// The checkpoint suite's write side —
/// `integration::checkpoint::helpers::SERVICE_LABEL`, which stamps
/// `Config::service_label` and the metrics registry on every E2E recovery
/// fixture — is a `pub use` **of this const**, not a second `"test"` literal
/// that happens to match. Read side and write side are therefore one address:
/// re-typing this value moves both at once, and there is no second copy left to
/// forget. Two same-spelled literals would only have made the drift unlikely,
/// and it would have failed silently (every cell reads 0) rather than loudly.
/// A new fixture that needs this label should alias it the same way rather than
/// re-spelling it.
#[cfg(feature = "test-support")]
pub const RECOVERY_SERVICE_LABEL: &str = "test";

/// Live value of one outcome cell of
/// `live_edit_checkpoint_recovery_attempts_total{result, scope, service}`.
///
/// `scope="layer"` is pinned, not defaulted. The counter carries a session
/// scope too (`SessionRecoveryHook`), and a helper that omitted the label
/// would sum the two — turning "the layer hook attempted one restore" into
/// "something attempted one restore", which is not the assertion any caller
/// wants.
#[cfg(feature = "test-support")]
pub fn recovery_metric(
    metrics: &live_edit::observability::metrics::SharedMetrics,
    result: &str,
) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", RECOVERY_SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Replace a service's metrics with a fresh registry so the test reads a clean
/// baseline.
///
/// MUST be called BEFORE any `replace_recovery`-style rewiring: a
/// `RecoveryHook` captures the metrics handle at construction, so swapping the
/// service's `Arc<Metrics>` afterwards leaves the hook writing to the old
/// registry and every [`recovery_metric`] read at zero.
#[cfg(feature = "test-support")]
pub fn replace_metrics(
    svc: &mut live_edit::grpc::service::LiveEditService,
) -> live_edit::observability::metrics::SharedMetrics {
    let metrics: live_edit::observability::metrics::SharedMetrics =
        std::sync::Arc::new(metrics_scrape_only(RECOVERY_SERVICE_LABEL));
    svc.metrics = std::sync::Arc::clone(&metrics);
    metrics
}

/// The smallest OMPF the checkpoint apply pipeline accepts as a restore
/// source: a reset delta (`is_diff = true`, `base_version = 0`) carrying no
/// content.
///
/// Enough to land an `obj:hdr`, which is all the retry-once and
/// arm-classification fixtures need — they assert which handler arm the RPC
/// took, never the restored content.
///
/// Built through the real `oodf-ffi` builder and a real `OmpfContainer` rather
/// than as a hand-rolled byte string: `decode_checkpoint_ompf` validates
/// exactly that pair of wire knobs, so hand-rolled bytes would be rejected
/// `Permanent` and every hand-put base built on them would restore nothing.
#[cfg(feature = "test-support")]
pub fn empty_reset_ompf() -> bytes::Bytes {
    use live_edit::domain::ompf::OmpfContainer;
    use oodf_ffi::OwnedDelta;

    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        ..OwnedDelta::default()
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta must succeed");
    let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize must succeed");
    bytes::Bytes::from(ompf)
}

/// Seed one layer checkpoint so `RecoveryHook::try_restore_layer` can select
/// and apply it.
///
/// `session_id` / `layer_name` MUST name the LIVE mapping: `apply_checkpoint`'s
/// SR-2.4 identity guard re-reads the blob's own `(session_id, layer_name)`
/// mapping and refuses when it does not resolve to `object_id`, which would
/// silently turn a `Restored` fixture into `NotRecoverable` — green test,
/// nothing proved.
#[cfg(feature = "test-support")]
pub async fn seed_layer_checkpoint(
    store: &live_edit::checkpoint::adapters::InMemoryCheckpointStore,
    object_id: u64,
    session_id: &str,
    layer_name: &str,
    snap_seq: u64,
) {
    use live_edit::checkpoint::{CheckpointMeta, CheckpointStore};

    let meta = CheckpointMeta {
        creator_replica_id: RECOVERY_SERVICE_LABEL.to_string(),
        written_at_unix_ms: 1_700_000_000_000,
        snap_seq: Some(snap_seq),
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: "test-user".to_string(),
        server_ts: 1_700_000_000_000,
        ..CheckpointMeta::default()
    };
    store
        .put_layer_checkpoint(object_id, snap_seq, empty_reset_ompf(), meta)
        .await
        .expect("seeding the layer checkpoint must succeed");
}

/// `UNLINK` every `le:{object_id}:obj:*` key, leaving session metadata and the
/// `le:{sid}:layer:{hex(name)}` mapping intact — the post-loss / pre-recovery
/// keyspace shape the checkpoint apply path runs against.
///
/// The surviving mapping is the discriminator between the two origins of
/// `handle_update_layer`'s merged recovery arm: with it alive the handler's
/// mapping resolve succeeds and the empty header is observed at the COMMIT;
/// wipe it too and the arm is reached from the mapping-resolve loop instead.
/// Callers that want the first origin call `wipe_layer_mapping` as well.
///
/// The shared `{object_id}` hash tag makes the multi-key `UNLINK` slot-safe on
/// Cluster.
///
/// `obj:stream` is wiped along with everything else, and that is deliberate: the
/// restore materializer (`LayerStore::write_restored_object`) does NOT `XADD` a
/// synthetic `1-0` entry the way `CreateLayer` does, so a stream left standing
/// would leak across the wipe into whatever the test plants next.
///
/// # Postcondition, asserted
///
/// The helper **re-scans and asserts no `obj:*` key survived**. This is not
/// belt-and-braces: every caller plants state into the wiped namespace and then
/// asserts something about what the code under test did with it, so a wipe that
/// silently under-deleted would leave the plant sitting beside live state and
/// the assertion built on top would prove nothing — passing for the wrong
/// reason, which is the failure mode a test cannot report on itself. Enumeration
/// goes through [`scan_object_keys`] so the wipe and its proof use the same
/// cluster-aware SCAN (a node-local scan under the `test-cluster` job would
/// under-report both the deletion set and the survivors, in the direction that
/// makes this assertion vacuous).
#[cfg(feature = "test-support")]
pub async fn wipe_object_keys(tv: &live_edit::storage::test_support::TestValkey, object_id: u64) {
    use fred::interfaces::KeysInterface;

    let to_delete = scan_object_keys(tv.client(), tv.prefix(), object_id).await;
    if !to_delete.is_empty() {
        let _: () = tv
            .client()
            .pool
            .unlink(to_delete)
            .await
            .expect("UNLINK of obj keys must succeed");
    }

    let survivors = scan_object_keys(tv.client(), tv.prefix(), object_id).await;
    assert!(
        survivors.is_empty(),
        "the wipe must remove every le:{{{object_id}}}:obj:* key, or the plant below lands \
         beside live state and the test proves nothing; survivors: {survivors:?}"
    );
}

/// A header-only `UpdateLayer` request with a stub bearer token and its
/// end-of-stream record in the extensions map.
///
/// No chunks, then END_STREAM → empty OMPF → `None` delta, which SDD §8.4
/// treats as an empty commit. Sufficient for every caller: they are about
/// which handler arm the commit lands on, not about delta content.
///
/// The request is built from [`upload_stream`] rather than either tree's
/// `streaming_update`, and that is forced rather than chosen: this module is
/// `#[path]`-included by the `misc` test binary too, which has neither
/// `crate::grpc::helpers` nor `crate::checkpoint::helpers` to name.
///
/// The 5-argument shape is the superset of the two promoted copies (one fixed
/// `(1000, "recovery-hook-client")`, the other `(1000, "test-client")`), so no
/// call site silently changes the `client_id` it sends — which self-echo
/// suppression keys off.
///
/// The bearer is the tree-wide `"test-token"`; a caller whose assertion is
/// ABOUT the token (a redaction canary needing a string that appears nowhere
/// else) calls [`update_request_with_bearer`] instead.
#[cfg(feature = "test-support")]
pub fn update_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp_micros: u64,
    client_id: &str,
) -> tonic::Request<tonic::codec::Streaming<live_edit::grpc::proto::UpdateLayerRequest>> {
    update_request_with_bearer(
        session_id,
        layer_name,
        object_id,
        timestamp_micros,
        client_id,
        "test-token",
    )
}

/// [`update_request`] with the bearer chosen by the caller.
///
/// Separate from the 5-argument form rather than folded into it because the
/// token is the one parameter most call sites must NOT think about: the
/// tree-wide `"test-token"` is what the mock auth clients expect, and a test
/// that varied it by accident would fail somewhere unrelated. The callers that
/// need it are the ones whose assertion is about the token — `tests/integration
/// /grpc/rpc_failure_log_test.rs`'s bearer canary needs a value that appears
/// nowhere else in the tree, so that finding it in captured log output is
/// unambiguous evidence of a leak.
#[cfg(feature = "test-support")]
pub fn update_request_with_bearer(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp_micros: u64,
    client_id: &str,
    bearer: &str,
) -> tonic::Request<tonic::codec::Streaming<live_edit::grpc::proto::UpdateLayerRequest>> {
    use live_edit::grpc::proto::{update_layer_request, UpdateLayerHeader, UpdateLayerRequest};
    use upload_stream::{AttestedStreaming, IntoHandlerRequest as _};

    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: timestamp_micros,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
            },
        )),
    };

    // A header-only upload that ends with END_STREAM, carrying its
    // end-of-stream record as the LiveService stack would.
    let mut request = AttestedStreaming::from_messages(&[header]).into_handler_request();
    request
        .extensions_mut()
        .insert(live_edit::grpc::BearerToken::new(bearer));
    request
}
