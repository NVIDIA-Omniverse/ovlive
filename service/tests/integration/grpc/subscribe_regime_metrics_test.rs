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

// Integration tests for the SubscribeReadLayer catch-up regime metrics
// (b.qxe).
//
// The engineer's source-lane change in `src/grpc/subscribe.rs` and
// `src/observability/metrics.rs` introduces two metric families:
//
//   • `live_subscribe_regime_total{regime, service}`  (CounterVec)
//   • `live_subscribe_catch_up_duration_seconds{regime, service}`  (HistogramVec)
//
// Exactly one observation per metric is emitted on the SUCCESS path of
// `run_catch_up`, labelled by the regime the function itself selected:
//
//   1. `brand_new`     — `have_object_id == 0 && have_version == 0` (first
//                         subscribe), OR `have_object_id != current` (the
//                         layer was overwritten on reconnect).
//   2. `up_to_date`    — `have_version >= current_seq`.
//   3. `in_window`     — XRANGE slice from `{have_version+1}-0` succeeds.
//   4. `out_of_window` — falls through after the XRANGE slice misses the
//                         expected start id (stream entry trimmed).
//
// Error paths (e.g. layer-not-found from `get_layer_mapping`) return early
// via `?` BEFORE the regime label is assigned, so no observation fires.
//
// Each test drives one branch end-to-end through `handle_subscribe_read_layer`
// against a real Valkey (PooledTestValkey + LiveEditService), then reads the
// `Metrics` instance the service wires up via `svc.metrics` and asserts:
//
//   • Counter on (regime, service="test") incremented by exactly 1.
//   • Histogram on (regime, service="test") recorded exactly 1 sample.
//   • Histogram sum > 0 (catch-up takes positive wall-clock time).
//   • No other regime's counter / histogram moved (label independence).
//
// All tests build the service via `build_pooled_service` from
// `subscribe_regimes_test`'s module-private fixture; the patterns here mirror
// that sibling file 1:1 so future changes to the test scaffold ripple through
// both. See subscribe_regimes_test.rs for the underlying regime-correctness
// coverage; this file is the metrics-side complement.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::{KeysInterface, StreamsInterface};
use fred::types::streams::XCap;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::config::{AuthProtocol, Config};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;

use crate::common::owned_delta::delta_to_ompf_bytes;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use super::helpers::*;

// ── Per-test fixture (mirrors subscribe_regimes_test's PooledTestValkey) ────

/// Per-test Valkey fixture with its own key-prefix namespace. Kept module-local
/// (rather than promoted to a shared helper) because the sibling
/// `subscribe_regimes_test` already owns the canonical pattern and we want any
/// future evolution of that pattern to touch both files in one pass.
struct MetricsTestValkey {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
}

impl MetricsTestValkey {
    async fn setup() -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let valkey_url = std::env::var("TEST_VALKEY_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let key_schema = KeySchema::new(&prefix);

        let config = Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url,
            valkey_pool_size: 0,
            subscriber_pool_size: 8,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                live_edit::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency:
                live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib:
                live_edit::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items:
                live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes:
                live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: live_edit::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: live_edit::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            // 30 s matches `TestValkey`'s post-b.1kf default — the
            // workspace suite's SYN-burst recovery profile under
            // listen-queue pressure needs that budget to drain
            // reliably (b.1kf, b.pws). Routing through
            // `connect_test_client` below also gates the connect on
            // the shared 16-permit setup-semaphore.
            valkey_connect_timeout_secs: 30,
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
            presence_refresh_interval_secs:
                live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms:
                live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: live_edit::config::LogFormat::Json,
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
            update_layer_backend: live_edit::config::UpdateLayerBackend::Lua,
            ext_value_verify: live_edit::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
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
        };

        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("MetricsTestValkey: connect must succeed");

        Self {
            client,
            key_schema,
            prefix,
        }
    }

    fn client(&self) -> &ValkeyClient {
        &self.client
    }

    fn key_schema(&self) -> &KeySchema {
        &self.key_schema
    }
}

impl Drop for MetricsTestValkey {
    fn drop(&mut self) {
        let pool = self.client.pool.clone();
        let prefix = self.prefix.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let pattern = format!("{prefix}*");
                let iter_client = pool.next();
                let mut stream = iter_client.scan_buffered(&pattern, Some(100), None);
                let mut batch = Vec::with_capacity(100);
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(key) => {
                            batch.push(key);
                            if batch.len() >= 100 {
                                let _ = pool.unlink::<(), _>(std::mem::take(&mut batch)).await;
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !batch.is_empty() {
                    let _ = pool.unlink::<(), _>(batch).await;
                }
            });
        });
    }
}

/// Build a LiveEditService backed by the per-test fixture with a custom
/// `layer_stream_maxlen`. Kept local rather than reusing
/// `subscribe_regimes_test::build_pooled_service` because sibling test files
/// can't directly import each other's module-private fixtures and the
/// duplication is cheap.
///
/// `catchup_snapshot_enabled` selects the catch-up path wired onto the service.
/// Metrics tests exercising the legacy per-commit reconstruction pass `false`;
/// the snapshot-mode coverage passes `true` to exercise the corrected-snapshot
/// path while asserting the regime observation still fires.
fn build_metrics_service(
    tv: &MetricsTestValkey,
    maxlen: u64,
    catchup_snapshot_enabled: bool,
) -> LiveEditService {
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        maxlen,
    ));
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );
    // Inline disabled session-recovery hook — `MetricsTestValkey` isn't
    // a `TestValkey`, so the canonical helper signature doesn't apply.
    let mut session_recovery_config = crate::common::test_config();
    session_recovery_config.checkpoint_enabled = false;
    let session_recovery_store: Arc<dyn live_edit::checkpoint::CheckpointStore> =
        Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    let session_recovery = Arc::new(live_edit::checkpoint::SessionRecoveryHook::new(
        session_recovery_store,
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(session_recovery_config),
        Arc::clone(&metrics),
    ));

    LiveEditService {
        catchup_snapshot_enabled,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        ),
        metrics,
        service_label: "test".to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        service_token_provider: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_xrange_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        outer_send_deadline_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_chunk_seam_for_test: None,
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
    }
}

/// Create a session + layer via the given service; returns (session_id, object_id).
async fn create_session_and_layer_metrics(
    tv: &MetricsTestValkey,
    svc: &LiveEditService,
    session_name: &str,
    layer_name: &str,
) -> (String, u64) {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create(session_name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let ompf_bytes = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    let response = handle_create_layer(svc, request)
        .await
        .expect("create_layer must succeed for test setup");
    (session_id, response.into_inner().object_id)
}

// ── Subscribe / Update helpers (mirror subscribe_regimes_test) ──────────────

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn update_header(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    client_ts: u64,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: client_ts,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
            },
        )),
    }
}

fn update_data(payload: Vec<u8>) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(
            live_edit::grpc::proto::UpdateLayerChunk { data: payload },
        )),
    }
}

/// Non-reset DiffChangeSection delta — mirrors the sibling test file so each
/// update produces a per-commit diff that accumulates real state on the
/// stream.
fn incremental_diff_change_section(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    }
}

async fn apply_unique_update(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    section_id: u64,
) {
    let delta = incremental_diff_change_section(section_id);
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        update_header(
            session_id,
            layer_name,
            object_id,
            client_id,
            section_id * 10,
        ),
        update_data(ompf_bytes),
    ]);
    let request = make_request(stream);
    handle_update_layer(svc, request)
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section_id={section_id}) failed: {e}"));
}

/// Drain the catch-up header (and any data chunks until `total_size` bytes
/// have arrived) so the subscription has fully exited `run_catch_up` — which
/// is the moment the metric observation is recorded — before the asserting
/// test reads counters. Returns the catch-up header for spot checks.
async fn drain_catch_up<S>(stream: &mut S) -> live_edit::grpc::proto::SubscribeReadLayerHeader
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("stream timed out waiting for header")
        .expect("stream closed unexpectedly")
        .expect("stream yielded error");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header, got: {other:?}"),
    };

    let mut buf_len: u64 = 0;
    while buf_len < header.total_size {
        let next = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("stream timed out waiting for chunk")
            .expect("stream closed before total_size reached")
            .expect("stream yielded error");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf_len += c.data.len() as u64;
            }
            other => panic!("expected Chunk while filling body, got: {other:?}"),
        }
    }

    header
}

// ── Metric-reader helpers ───────────────────────────────────────────────────

const TEST_SERVICE: &str = "test";
const REGIMES: &[&str] = &["brand_new", "up_to_date", "in_window", "out_of_window"];

fn read_regime_counter(metrics: &Metrics, regime: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_regime_total",
        &[("regime", regime), ("service", TEST_SERVICE)],
    )
    .unwrap_or(0.0) as u64
}

fn read_regime_histogram_sample_count(metrics: &Metrics, regime: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_duration_seconds_count",
        &[("regime", regime), ("service", TEST_SERVICE)],
    )
    .unwrap_or(0.0) as u64
}

fn read_regime_histogram_sum(metrics: &Metrics, regime: &str) -> f64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_duration_seconds_sum",
        &[("regime", regime), ("service", TEST_SERVICE)],
    )
    .unwrap_or(0.0)
}

// ── Mode-aware readers (Plan b.msb Task 4 additive `catchup_mode` label) ─────
//
// Task 4 adds an ADDITIVE `catchup_mode` label (values `snapshot` | `legacy`)
// to `live_subscribe_regime_total`, `live_subscribe_catch_up_duration_seconds`,
// and the new `live_subscribe_catch_up_delivered_bytes` histogram. These
// helpers mirror `read_regime_counter` / `read_regime_histogram_*` but append
// the `catchup_mode` pair so the two mode series on the SAME metric name are
// read unambiguously (`sample_value` is a subset match, so a mode-less read
// would resolve whichever series prints first — ambiguous when both fire).

/// `live_subscribe_regime_total` on the `(regime, service, catchup_mode)` series.
fn read_regime_counter_mode(metrics: &Metrics, regime: &str, mode: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_regime_total",
        &[
            ("regime", regime),
            ("service", TEST_SERVICE),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_subscribe_catch_up_duration_seconds_count` on the mode series.
fn read_regime_duration_count_mode(metrics: &Metrics, regime: &str, mode: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_duration_seconds_count",
        &[
            ("regime", regime),
            ("service", TEST_SERVICE),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_subscribe_catch_up_duration_seconds_sum` on the mode series.
fn read_regime_duration_sum_mode(metrics: &Metrics, regime: &str, mode: &str) -> f64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_duration_seconds_sum",
        &[
            ("regime", regime),
            ("service", TEST_SERVICE),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0)
}

/// `live_subscribe_catch_up_delivered_bytes_count` on the mode series.
fn read_delivered_bytes_count_mode(metrics: &Metrics, regime: &str, mode: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_delivered_bytes_count",
        &[
            ("regime", regime),
            ("service", TEST_SERVICE),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_subscribe_catch_up_delivered_bytes_sum` on the mode series.
fn read_delivered_bytes_sum_mode(metrics: &Metrics, regime: &str, mode: &str) -> f64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_catch_up_delivered_bytes_sum",
        &[
            ("regime", regime),
            ("service", TEST_SERVICE),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0)
}

/// Poll `read_regime_counter_mode` until it reaches `target` or the 2s deadline
/// expires. Same rationale as `await_counter` but keyed on the mode series so a
/// legacy observation doesn't satisfy a wait for a snapshot one (and vice-versa).
async fn await_counter_mode(metrics: &Metrics, regime: &str, mode: &str, target: u64) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if read_regime_counter_mode(metrics, regime, mode) >= target {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "regime counter {regime}/catchup_mode={mode} did not reach {target} within 2s (final={})",
        read_regime_counter_mode(metrics, regime, mode)
    );
}

/// Poll `read_regime_counter(metrics, regime)` until it reaches `target`
/// (default `1`) or the deadline expires. The metric write happens inside the
/// spawned subscription task, which only runs to the `observe_subscribe_regime`
/// call after `run_catch_up` returns. On the brand_new / in_window / out_of_window
/// paths the test has already drained the catch-up header (which means
/// `run_catch_up` returned and the observation MUST have been recorded), but
/// on `up_to_date` the post-ack send is a single `send_ack_header` that lands
/// on the wire before the observation in `run_catch_up`. Poll briefly to
/// tolerate that single yield-point.
async fn await_counter(metrics: &Metrics, regime: &str, target: u64) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if read_regime_counter(metrics, regime) >= target {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "regime counter {regime} did not reach {target} within 2s (final={})",
        read_regime_counter(metrics, regime)
    );
}

/// Assert that ONE regime label tuple recorded a single counter increment
/// and a single histogram observation, AND no other regime label moved.
/// This guards both the labelling (right label wins) and the dispatch
/// uniqueness (only one regime per RPC).
fn assert_only_regime_observed(metrics: &Metrics, expected: &str) {
    for regime in REGIMES {
        let cnt = read_regime_counter(metrics, regime);
        let samples = read_regime_histogram_sample_count(metrics, regime);
        if *regime == expected {
            assert_eq!(
                cnt, 1,
                "expected regime '{expected}' counter == 1, got {cnt}"
            );
            assert_eq!(
                samples, 1,
                "expected regime '{expected}' histogram sample_count == 1, got {samples}"
            );
            let sum = read_regime_histogram_sum(metrics, expected);
            assert!(
                sum > 0.0,
                "expected regime '{expected}' histogram sample_sum > 0, got {sum}"
            );
        } else {
            assert_eq!(
                cnt, 0,
                "regime '{regime}' counter must remain 0 when '{expected}' is the active branch (got {cnt})"
            );
            assert_eq!(
                samples, 0,
                "regime '{regime}' histogram must have 0 samples when '{expected}' is the active branch (got {samples})"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 1a: brand_new — first-time subscribe (have_object_id=0, have_version=0)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_first_subscribe_increments_metrics_once() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, _object_id) =
        create_session_and_layer_metrics(&tv, &svc, "bn-fresh-session", "bn-fresh-layer").await;

    // First-time subscribe: have_object_id=0, have_version=0 → `is_brand_new` true.
    let sub_req = subscribe_request(&session_id, "bn-fresh-layer", 0, 0, "sub-bn-fresh");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the catch-up header to ensure `run_catch_up` returned.
    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 1,
        "brand_new catch-up should land seq=1 for a freshly-created layer"
    );
    // b.rdr: the brand_new full-state delivery terminates with a total_size==0
    // caught-up ack carrying the same delivered version.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a total_size==0 ack"
    );
    assert_eq!(
        ack.sequence_num, header.sequence_num,
        "caught-up ack must echo the delivered sequence_num"
    );

    await_counter(&svc.metrics, "brand_new", 1).await;
    assert_only_regime_observed(&svc.metrics, "brand_new");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 1a under snapshot catch-up: brand_new metrics still emit when
// LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true reroutes `catch_up_brand_new` to the
// corrected-snapshot builder (Existing-Behavior contract, t3.msb.35.f9.bs).
// ═══════════════════════════════════════════════════════════════════════════════
//
// The snapshot rerouting changes HOW the brand_new full-state body is built,
// not the regime the RPC is classified as: `subscribe.rs` keeps
// `regime = "brand_new"` (only adding a `mode` log field), so the
// `live_subscribe_regime_total{regime="brand_new"}` counter and the
// `live_subscribe_catch_up_duration_seconds` observation must still fire on the
// `brand_new` label and no other regime label may move. This mirrors
// `brand_new_first_subscribe_increments_metrics_once` 1:1, the only difference
// being the `Snapshot` catch-up mode threaded through `build_metrics_service`.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_first_subscribe_increments_metrics_once_snapshot_mode() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, _object_id) =
        create_session_and_layer_metrics(&tv, &svc, "bn-fresh-session-snap", "bn-fresh-layer-snap")
            .await;

    // First-time subscribe: have_object_id=0, have_version=0 → `is_brand_new`
    // true, rerouted to the corrected-snapshot builder under `Snapshot` mode.
    let sub_req = subscribe_request(
        &session_id,
        "bn-fresh-layer-snap",
        0,
        0,
        "sub-bn-fresh-snap",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the full-state header (proves `run_catch_up` returned).
    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 1,
        "brand_new catch-up should land seq=1 for a freshly-created layer"
    );
    // The brand_new full-state delivery still terminates with a total_size==0
    // caught-up ack carrying the same delivered version under snapshot mode.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a total_size==0 ack under snapshot mode"
    );
    assert_eq!(
        ack.sequence_num, header.sequence_num,
        "caught-up ack must echo the delivered sequence_num"
    );

    // The regime observation must still fire on `brand_new` — and only there.
    await_counter(&svc.metrics, "brand_new", 1).await;
    assert_only_regime_observed(&svc.metrics, "brand_new");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 1b: brand_new — overwrite-on-reconnect (have_object_id != current)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_overwrite_reconnect_increments_metrics_once() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, old_object_id) =
        create_session_and_layer_metrics(&tv, &svc, "bn-overwrite-session", "bn-overwrite-layer")
            .await;

    // Overwrite the layer via a fresh CreateLayer — this allocates a NEW
    // object_id while keeping the same (session_id, layer_name) mapping.
    let ompf_bytes = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "bn-overwrite-layer"),
        make_create_data(ompf_bytes),
    ]);
    let create_req = make_request(create_stream);
    let create_resp = handle_create_layer(&svc, create_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = create_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "CreateLayer overwrite must allocate a new object_id"
    );

    // Reconnect carrying the OLD object_id and a non-zero have_version. This
    // hits the `ctx.have_object_id != object_id` branch inside
    // `run_catch_up_inner`, which is still labelled `brand_new`.
    let sub_req = subscribe_request(
        &session_id,
        "bn-overwrite-layer",
        old_object_id,
        5,
        "sub-bn-overwrite",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.object_id, new_object_id,
        "overwrite-on-reconnect catch-up must surface the NEW object_id"
    );
    assert_eq!(
        header.sequence_num, 1,
        "overwrite-on-reconnect catch-up must land at seq=1 on the new object"
    );
    // b.rdr: the overwrite-on-reconnect path also routes through
    // `catch_up_brand_new(NEW)`, so the full-state header is followed by the
    // total_size==0 caught-up ack on the NEW object.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a total_size==0 ack"
    );
    assert_eq!(
        ack.object_id, new_object_id,
        "caught-up ack must echo the NEW object_id"
    );
    assert_eq!(
        ack.sequence_num, 1,
        "caught-up ack must echo the delivered sequence_num (1 on the new object)"
    );

    await_counter(&svc.metrics, "brand_new", 1).await;
    assert_only_regime_observed(&svc.metrics, "brand_new");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 1c: brand_new — STEADY-STATE-phase mid-session-overwrite re-bootstrap
// (SR-2.3 / SDD §12.4, Plan b.msb Task 2 source Subtask t3.msb.xq.um.f9).
// ═══════════════════════════════════════════════════════════════════════════════
//
// Distinct from `brand_new_overwrite_reconnect_increments_metrics_once` above,
// which overwrites BEFORE the subscriber connects so the overwrite is observed
// on the INITIAL subscribe through the `run_catch_up` brand_new-overwrite
// branch. These two tests instead exercise the `run_steady_state` re-bootstrap
// site: the subscriber first completes an `up_to_date` initial catch-up (opening
// the steady-state consumer at `$` on the OLD object's stream), THEN a
// mid-session CreateLayer overwrite XADDs the `deleted` sentinel, and the
// steady-state loop's `LayerNotification::Deleted` arm re-bootstraps onto the new
// generation via `handle_deleted_and_rebootstrap` → `catch_up_brand_new` →
// `send_full_state`. Source Subtask t3.msb.xq.um.f9 added the exactly-once
// `observe_subscribe_regime("brand_new", …, outcome.catchup_mode,
// outcome.delivered_bytes)` emission at that site (src/grpc/subscribe.rs,
// `run_steady_state` Deleted arm); before it the re-bootstrap recorded NO
// regime / mode / delivered-bytes metric at all.
//
// The two tests differ only in the `catchup_snapshot_enabled` mode threaded through
// `build_metrics_service`, mirroring the file's brand_new snapshot/legacy
// pairing: `Snapshot` → the re-bootstrap full-state lands on the
// `catchup_mode="snapshot"` series; `Off` → on `catchup_mode="legacy"` (both
// through `send_full_state`, whose Snapshot arm is the ONLY `"snapshot"`
// emitter). The overwrite CreateLayer carries a NON-EMPTY DiffChangeSection
// delta so the new generation's full-state snapshot delivers > 0 OMPF bytes,
// keeping the delivered-bytes assertion non-vacuous. Without the f9 emission the
// re-bootstrap records nothing, `brand_new` stays 0, and `await_counter_mode`
// times out — so these are clean regression guards.
//
// Note: `assert_only_regime_observed` is deliberately NOT reused — the initial
// `up_to_date` catch-up legitimately fires its own regime observation, so the
// all-other-regimes-zero helper does not apply; instead each test asserts the
// brand_new mode series explicitly and that `in_window` / `out_of_window` stay 0.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_overwrite_rebootstrap_emits_brand_new_metric_snapshot_mode() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, old_object_id) = create_session_and_layer_metrics(
        &tv,
        &svc,
        "ss-rebootstrap-snap-session",
        "ss-rebootstrap-snap-layer",
    )
    .await;

    // Subscribe with have_version=1 == current_seq(=1) → the initial catch-up
    // classifies as `up_to_date`, delivering a zero-body ack and opening the
    // steady-state consumer at `$` on the OLD object's stream. This makes the
    // subscriber "already caught up" (the re-bootstrap precondition) AND keeps
    // `brand_new` at 0 until the re-bootstrap, so the single later `brand_new`
    // observation is unambiguously attributable to the steady-state re-bootstrap.
    let sub_req = subscribe_request(
        &session_id,
        "ss-rebootstrap-snap-layer",
        old_object_id,
        1,
        "sub-ss-rebootstrap-snap",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the up_to_date ack (total_size==0) and settle its regime observation
    // BEFORE the overwrite so the initial catch-up is fully committed.
    let init_ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        init_ack.total_size, 0,
        "up_to_date initial catch-up must be a zero-body ack"
    );
    assert_eq!(
        init_ack.sequence_num, 1,
        "up_to_date ack must echo current server seq (=1)"
    );
    await_counter(&svc.metrics, "up_to_date", 1).await;

    // Mid-session overwrite: a fresh CreateLayer on the SAME (session, layer)
    // carrying a NON-EMPTY DiffChangeSection delta. Allocates a NEW object_id,
    // materializes distinctive new-generation content (so the re-bootstrap
    // snapshot is > 0 bytes), and appends the `deleted` sentinel to the OLD
    // object's stream — which the steady-state `XREAD BLOCK` consumer observes.
    let overwrite_ompf = delta_to_ompf_bytes(&incremental_diff_change_section(7_701), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "ss-rebootstrap-snap-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let create_req = make_request(create_stream);
    let new_object_id = handle_create_layer(&svc, create_req)
        .await
        .expect("mid-session CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "CreateLayer overwrite must allocate a new object_id generation"
    );

    // Observe the re-bootstrap ON THE WIRE: the steady-state Deleted arm
    // re-bootstraps onto the new generation and streams its full-state header
    // (object_id==new, seq==1, total_size>0) followed by the exactly-once
    // total_size==0 caught-up ack on the new object.
    let rebootstrap_header = drain_catch_up(&mut stream).await;
    assert_eq!(
        rebootstrap_header.object_id, new_object_id,
        "steady-state re-bootstrap full-state must surface the NEW object_id"
    );
    assert_eq!(
        rebootstrap_header.sequence_num, 1,
        "re-bootstrap full-state must land at seq=1 on the new generation"
    );
    assert!(
        rebootstrap_header.total_size > 0,
        "non-empty new-generation content must yield a non-empty re-bootstrap body"
    );
    let rebootstrap_ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        rebootstrap_ack.total_size, 0,
        "re-bootstrap trailer must be a total_size==0 caught-up ack"
    );
    assert_eq!(
        rebootstrap_ack.object_id, new_object_id,
        "re-bootstrap caught-up ack must echo the NEW object_id"
    );
    assert_eq!(
        rebootstrap_ack.sequence_num, 1,
        "re-bootstrap caught-up ack must echo the delivered seq (1 on the new object)"
    );

    // EXACTLY ONE brand_new observation on the `catchup_mode="snapshot"` series
    // (the re-bootstrap routes through `send_full_state`'s Snapshot arm).
    await_counter_mode(&svc.metrics, "brand_new", "snapshot", 1).await;
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "steady-state re-bootstrap must increment the brand_new/snapshot counter exactly once"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "re-bootstrap must record exactly one duration sample on the snapshot series"
    );
    assert!(
        read_regime_duration_sum_mode(&svc.metrics, "brand_new", "snapshot") > 0.0,
        "re-bootstrap snapshot duration sum must be > 0"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "re-bootstrap must record exactly one delivered-bytes sample on the snapshot series"
    );
    assert!(
        read_delivered_bytes_sum_mode(&svc.metrics, "brand_new", "snapshot") > 0.0,
        "re-bootstrap snapshot delivered-bytes sum must be > 0 (non-empty new gen)"
    );

    // The opposite (legacy) brand_new series must stay 0 under Snapshot mode.
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "legacy"),
        0,
        "the legacy brand_new series must NOT move under snapshot mode"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "legacy"),
        0,
        "the legacy brand_new delivered-bytes series must NOT move under snapshot mode"
    );

    // The other regimes never fire for a steady-state overwrite re-bootstrap.
    assert_eq!(
        read_regime_counter(&svc.metrics, "in_window"),
        0,
        "in_window must stay 0 for a steady-state overwrite re-bootstrap"
    );
    assert_eq!(
        read_regime_counter(&svc.metrics, "out_of_window"),
        0,
        "out_of_window must stay 0 for a steady-state overwrite re-bootstrap"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_overwrite_rebootstrap_emits_brand_new_metric_legacy_mode() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, old_object_id) = create_session_and_layer_metrics(
        &tv,
        &svc,
        "ss-rebootstrap-legacy-session",
        "ss-rebootstrap-legacy-layer",
    )
    .await;

    // have_version=1 == current_seq(=1) → up_to_date initial catch-up; opens the
    // steady-state consumer at `$` on the OLD object's stream (see the snapshot
    // sibling above for the full rationale).
    let sub_req = subscribe_request(
        &session_id,
        "ss-rebootstrap-legacy-layer",
        old_object_id,
        1,
        "sub-ss-rebootstrap-legacy",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let init_ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        init_ack.total_size, 0,
        "up_to_date initial catch-up must be a zero-body ack"
    );
    assert_eq!(
        init_ack.sequence_num, 1,
        "up_to_date ack must echo current server seq (=1)"
    );
    await_counter(&svc.metrics, "up_to_date", 1).await;

    // Mid-session overwrite with non-empty new-generation content.
    let overwrite_ompf = delta_to_ompf_bytes(&incremental_diff_change_section(7_801), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "ss-rebootstrap-legacy-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let create_req = make_request(create_stream);
    let new_object_id = handle_create_layer(&svc, create_req)
        .await
        .expect("mid-session CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "CreateLayer overwrite must allocate a new object_id generation"
    );

    let rebootstrap_header = drain_catch_up(&mut stream).await;
    assert_eq!(
        rebootstrap_header.object_id, new_object_id,
        "steady-state re-bootstrap full-state must surface the NEW object_id"
    );
    assert_eq!(
        rebootstrap_header.sequence_num, 1,
        "re-bootstrap full-state must land at seq=1 on the new generation"
    );
    assert!(
        rebootstrap_header.total_size > 0,
        "non-empty new-generation content must yield a non-empty re-bootstrap body"
    );
    let rebootstrap_ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        rebootstrap_ack.total_size, 0,
        "re-bootstrap trailer must be a total_size==0 caught-up ack"
    );
    assert_eq!(
        rebootstrap_ack.object_id, new_object_id,
        "re-bootstrap caught-up ack must echo the NEW object_id"
    );
    assert_eq!(
        rebootstrap_ack.sequence_num, 1,
        "re-bootstrap caught-up ack must echo the delivered seq (1 on the new object)"
    );

    // EXACTLY ONE brand_new observation on the `catchup_mode="legacy"` series
    // (under `Off`, `catch_up_brand_new` → `send_full_state`'s non-Snapshot arm
    // emits catchup_mode="legacy").
    await_counter_mode(&svc.metrics, "brand_new", "legacy", 1).await;
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "steady-state re-bootstrap must increment the brand_new/legacy counter exactly once"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "re-bootstrap must record exactly one duration sample on the legacy series"
    );
    assert!(
        read_regime_duration_sum_mode(&svc.metrics, "brand_new", "legacy") > 0.0,
        "re-bootstrap legacy duration sum must be > 0"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "re-bootstrap must record exactly one delivered-bytes sample on the legacy series"
    );
    assert!(
        read_delivered_bytes_sum_mode(&svc.metrics, "brand_new", "legacy") > 0.0,
        "re-bootstrap legacy delivered-bytes sum must be > 0 (non-empty new gen)"
    );

    // The opposite (snapshot) brand_new series must stay 0 under Off mode.
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "snapshot"),
        0,
        "the snapshot brand_new series must NOT move under Off/legacy mode"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "snapshot"),
        0,
        "the snapshot brand_new delivered-bytes series must NOT move under Off/legacy mode"
    );

    // The other regimes never fire for a steady-state overwrite re-bootstrap.
    assert_eq!(
        read_regime_counter(&svc.metrics, "in_window"),
        0,
        "in_window must stay 0 for a steady-state overwrite re-bootstrap"
    );
    assert_eq!(
        read_regime_counter(&svc.metrics, "out_of_window"),
        0,
        "out_of_window must stay 0 for a steady-state overwrite re-bootstrap"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 2: up_to_date — have_version >= current_seq
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn up_to_date_reconnect_increments_metrics_once() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "utd-session", "utd-layer").await;
    // Layer created at seq=1, no further updates → current_seq==1.

    // Reconnect with have_version=1 (matches the server header exactly) →
    // `have_version >= current_seq` branch is taken (the `==` boundary case).
    let sub_req = subscribe_request(&session_id, "utd-layer", object_id, 1, "sub-utd");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // up_to_date sends a single ack header (total_size==0, no chunks) and
    // opens the consumer at $. `drain_catch_up` handles total_size==0
    // correctly (the loop body is skipped) so this still demonstrates that
    // `run_catch_up` returned.
    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 1,
        "up_to_date ack header must echo the current server sequence_num"
    );
    assert_eq!(header.total_size, 0, "up_to_date ack must carry no body");

    await_counter(&svc.metrics, "up_to_date", 1).await;
    assert_only_regime_observed(&svc.metrics, "up_to_date");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 3: in_window — XRANGE slice from {have_version+1}-0 succeeds
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_window_reconnect_increments_metrics_once() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "iw-session", "iw-layer").await;

    // Walk forward to seq=5 (4 updates after the seq=1 create).
    for i in 0..4u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-layer",
            object_id,
            "writer-pre",
            100 + i,
        )
        .await;
    }

    // 3 more updates while "disconnected" — server reaches seq=8. MAXLEN=4096
    // keeps the entries on the stream so the XRANGE slice from 6-0 finds them.
    for i in 0..3u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-layer",
            object_id,
            "writer-post",
            200 + i,
        )
        .await;
    }

    // Reconnect with have_version=5 → XRANGE from 6-0 hits seq=6 → in_window.
    let sub_req = subscribe_request(&session_id, "iw-layer", object_id, 5, "sub-iw");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // in_window replay delivers one header per replayed seq (6, 7, 8). Drain
    // the first one — that's enough to prove `run_catch_up` returned (the
    // observation fires on the same call boundary regardless of how many
    // per-commit headers `catch_up_in_window` later sends).
    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 6,
        "in_window replay must start at have_version + 1 = 6"
    );

    await_counter(&svc.metrics, "in_window", 1).await;
    assert_only_regime_observed(&svc.metrics, "in_window");

    // b.dy5: drain the remaining replay headers (7, 8) and the exactly-once
    // caught-up ack (`total_size==0`, seq=8 = last replayed `tracked_version`)
    // that `run_catch_up` now emits centrally. The regime counter fired on
    // `run_catch_up`'s return regardless of this trailer; this confirms the
    // trailer does not disturb the metric path.
    let replay7 = drain_catch_up(&mut stream).await;
    assert_eq!(replay7.sequence_num, 7, "second in_window replay is seq=7");
    let replay8 = drain_catch_up(&mut stream).await;
    assert_eq!(replay8.sequence_num, 8, "third in_window replay is seq=8");
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "in_window catch-up trailer is a zero-body ack"
    );
    assert_eq!(
        ack.sequence_num, 8,
        "in_window caught-up ack carries the last replayed endpoint (tracked_version=8)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 4: out_of_window — XRANGE slice misses the expected start id
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn out_of_window_reconnect_increments_metrics_once() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4, false); // small MAXLEN, will be force-trimmed below

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "oow-session", "oow-layer").await;

    // Walk forward to seq=5 → then 10 more updates → server seq=15.
    for i in 0..4u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer",
            object_id,
            "writer-pre",
            100 + i,
        )
        .await;
    }
    for i in 0..10u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer",
            object_id,
            "writer-post",
            200 + i,
        )
        .await;
    }

    // Force the stream past the subscriber's start_id deterministically.
    // MAXLEN trimming is approximate ("~"), so we use XTRIM MINID = 15-0 to
    // guarantee the seq=6 entry is gone.
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "15-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    // Reconnect with have_version=5 → XRANGE from 6-0 misses → out_of_window.
    let sub_req = subscribe_request(&session_id, "oow-layer", object_id, 5, "sub-oow");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 15,
        "out_of_window coalesced diff must carry the current server seq"
    );

    await_counter(&svc.metrics, "out_of_window", 1).await;
    assert_only_regime_observed(&svc.metrics, "out_of_window");

    // b.dy5: drain the exactly-once caught-up ack (`total_size==0`, seq=15 =
    // current_seq = tracked_version for Regime 3) that `run_catch_up` now emits
    // after the coalesced diff. The regime counter fired regardless; this
    // confirms the trailer does not disturb the metric path.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "out_of_window catch-up trailer is a zero-body ack"
    );
    assert_eq!(
        ack.sequence_num, 15,
        "out_of_window caught-up ack carries the coalesced endpoint (tracked_version=15)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 4 under snapshot catch-up: out_of_window metrics still emit when
// LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true reroutes the coalesced delivery through
// the faithful-coalesced builder (Plan b.msb Task 2, t3.msb.js.qb.kz).
// ═══════════════════════════════════════════════════════════════════════════════
//
// Task 2 changes only the DELIVERY of the out-of-window regime under `Snapshot`
// mode (the coalesced body is built by the DR-independent faithful-coalesced
// entrypoint instead of the legacy `compute_diff_at_cut` path). The regime
// CLASSIFICATION is chosen by the XRANGE-slice-miss branch in `run_catch_up`
// BEFORE the delivery mode is selected, so the
// `live_subscribe_regime_total{regime="out_of_window"}` counter and the
// `live_subscribe_catch_up_duration_seconds` observation must still fire on the
// `out_of_window` label and no other regime label may move. This mirrors
// `out_of_window_reconnect_increments_metrics_once` 1:1, the only difference
// being the `Snapshot` catch-up mode threaded through `build_metrics_service`.
// (Sibling Task 4 additive `catchup_mode` label has NOT landed at authoring
// time, so this test asserts only the regime dimension.)

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn out_of_window_reconnect_increments_metrics_once_snapshot_mode() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4, true); // small MAXLEN, will be force-trimmed below

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "oow-session-snap", "oow-layer-snap").await;

    // Walk forward to seq=5 → then 10 more updates → server seq=15.
    for i in 0..4u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer-snap",
            object_id,
            "writer-pre",
            100 + i,
        )
        .await;
    }
    for i in 0..10u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer-snap",
            object_id,
            "writer-post",
            200 + i,
        )
        .await;
    }

    // Force the stream past the subscriber's start_id deterministically.
    // MAXLEN trimming is approximate ("~"), so we use XTRIM MINID = 15-0 to
    // guarantee the seq=6 entry is gone → genuine out-of-window reconnect.
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "15-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    // Reconnect with have_version=5 → XRANGE from 6-0 misses → out_of_window,
    // delivered via the faithful-coalesced builder under `Snapshot` mode.
    let sub_req = subscribe_request(&session_id, "oow-layer-snap", object_id, 5, "sub-oow-snap");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 15,
        "out_of_window coalesced diff must carry the current server seq under snapshot mode"
    );

    // The regime observation must still fire on `out_of_window` — and only there.
    await_counter(&svc.metrics, "out_of_window", 1).await;
    assert_only_regime_observed(&svc.metrics, "out_of_window");

    // Drain the exactly-once caught-up ack (`total_size==0`, seq=15 =
    // current_seq = tracked_version) that `run_catch_up` emits after the
    // coalesced diff. The regime counter fired regardless; this confirms the
    // snapshot-mode delivery does not disturb the metric path.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "out_of_window catch-up trailer is a zero-body ack under snapshot mode"
    );
    assert_eq!(
        ack.sequence_num, 15,
        "out_of_window caught-up ack carries the coalesced endpoint (tracked_version=15)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 3 under snapshot catch-up: in_window metrics still emit when
// LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true reroutes the in-window reconnect delivery
// through the DR-independent faithful-coalesced builder (Plan b.msb Task 1,
// t3.msb.xq.so.2e).
// ═══════════════════════════════════════════════════════════════════════════════
//
// Task 1 changes only the DELIVERY of the in-window regime under `Snapshot`
// mode: instead of the legacy per-commit XRANGE-and-relay replay (seq 6/7/8),
// the reconnector is caught up with a SINGLE corrected coalesced snapshot
// (`compute_faithful_coalesced_at_cut`, `base_version = have_version`) at `current_seq`.
// The regime CLASSIFICATION is chosen by the XRANGE-slice-HIT branch in
// `run_catch_up` BEFORE the delivery mode is selected, so the
// `live_subscribe_regime_total{regime="in_window"}` counter and the
// `live_subscribe_catch_up_duration_seconds` observation must still fire on the
// `in_window` label and no other regime label may move. This mirrors the Off-mode
// `in_window_reconnect_increments_metrics_once` on the regime dimension, the
// difference being the `Snapshot` catch-up mode threaded through
// `build_metrics_service` (and the single-coalesced-header delivery shape it
// produces, matching `out_of_window_reconnect_increments_metrics_once_snapshot_mode`).

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_window_reconnect_increments_metrics_once_snapshot_mode() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "iw-session-snap", "iw-layer-snap").await;

    // Walk forward to seq=5 (4 updates after the seq=1 create).
    for i in 0..4u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-layer-snap",
            object_id,
            "writer-pre",
            100 + i,
        )
        .await;
    }

    // 3 more updates while "disconnected" — server reaches seq=8. MAXLEN=4096
    // keeps the entries on the stream so the XRANGE slice from 6-0 finds them →
    // in_window classification (the branch Task 1 leaves in place).
    for i in 0..3u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-layer-snap",
            object_id,
            "writer-post",
            200 + i,
        )
        .await;
    }

    // Reconnect with have_version=5 → XRANGE from 6-0 hits seq=6 → in_window,
    // delivered as a SINGLE corrected coalesced snapshot (base=have_version)
    // under `Snapshot` mode instead of the legacy per-commit 6/7/8 replay.
    let sub_req = subscribe_request(&session_id, "iw-layer-snap", object_id, 5, "sub-iw-snap");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Snapshot-mode in-window delivers ONE coalesced header at current_seq (=8),
    // not the per-commit 6/7/8 replay of the Off-mode legacy arm.
    let header = drain_catch_up(&mut stream).await;
    assert_eq!(
        header.sequence_num, 8,
        "snapshot-mode in_window coalesced snapshot carries current_seq (=8)"
    );

    // The regime observation must still fire on `in_window` — and only there.
    await_counter(&svc.metrics, "in_window", 1).await;
    assert_only_regime_observed(&svc.metrics, "in_window");

    // Drain the exactly-once caught-up ack (`total_size==0`, seq=8 =
    // tracked_version = current_seq) that `run_catch_up` emits after the
    // coalesced snapshot. The regime counter fired on `run_catch_up`'s return
    // regardless; this confirms the snapshot-mode delivery does not disturb the
    // metric path.
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "in_window catch-up trailer is a zero-body ack under snapshot mode"
    );
    assert_eq!(
        ack.sequence_num, 8,
        "in_window caught-up ack carries the coalesced endpoint (tracked_version=8)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regime 3 under snapshot catch-up, additive `catchup_mode` dimension: the
// converged in-window reconnect delivery lands on catchup_mode="legacy".
// ═══════════════════════════════════════════════════════════════════════════════
//
// Task 1 makes the in-window reconnect converge on `compute_faithful_coalesced_at_cut`
// (base=have_version) — the SAME primitive the out_of_window snapshot-mode
// reconnect uses. Per the Plan b.msb `catchup_mode` taxonomy, a coalesced
// `base=have_version` delivery is `"legacy"`; `"snapshot"` is reserved for the
// from-empty full-state (`send_full_state`) path only. Source of truth:
// `catch_up_in_window_snapshot` returns `catchup_mode: "legacy"`
// (src/grpc/subscribe.rs — the comment there pins it explicitly, "matching
// `catch_up_out_of_window`"). So the in-window snapshot-mode observation must
// land on the `catchup_mode="legacy"` series across ALL three catch-up
// instruments and leave the `"snapshot"` series at zero — consistent with the
// out_of_window reconnect precedent. Mirrors
// `brand_new_snapshot_mode_lands_on_snapshot_catchup_mode_series` (but pinned to
// the opposite mode value, because the regime is delivered by the coalesced
// builder, not `send_full_state`'s Snapshot arm).

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_window_snapshot_mode_lands_on_legacy_catchup_mode_series() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "iw-cm-snap-session", "iw-cm-snap-layer").await;

    // Seed 7 in-window updates (seq 2..8 → current_seq=8; MAXLEN=4096 keeps the
    // XRANGE slice from 6-0 alive). Real DiffChangeSection state (sections at
    // seq 6/7/8 have version > have_version=5) so the base=have_version coalesced
    // snapshot delivers > 0 OMPF bytes → the delivered-bytes sum is non-vacuous.
    seed_sections(
        &svc,
        &session_id,
        "iw-cm-snap-layer",
        object_id,
        "seed",
        300,
        7,
    )
    .await;

    // Reconnect with have_version=5 → XRANGE from 6-0 hits → in_window,
    // delivered via the faithful-coalesced primitive under `Snapshot` mode.
    let sub_req = subscribe_request(
        &session_id,
        "iw-cm-snap-layer",
        object_id,
        5,
        "sub-iw-cm-snap",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert!(
        header.total_size > 0,
        "seeded in-window coalesced snapshot must deliver a non-empty body"
    );
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "in_window trailer must be a zero-body ack"
    );

    // The observation lands on the `catchup_mode="legacy"` series across all
    // three catch-up instruments — matching the out_of_window reconnect
    // precedent, NOT the from-empty full-state `"snapshot"` value.
    await_counter_mode(&svc.metrics, "in_window", "legacy", 1).await;
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "in_window", "legacy"),
        1,
        "snapshot-mode in_window must increment the catchup_mode=legacy counter"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "in_window", "legacy"),
        1,
        "snapshot-mode in_window must record exactly one duration sample on the legacy series"
    );
    assert!(
        read_regime_duration_sum_mode(&svc.metrics, "in_window", "legacy") > 0.0,
        "snapshot-mode in_window duration sum must be > 0"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "in_window", "legacy"),
        1,
        "snapshot-mode in_window must record exactly one delivered-bytes sample on the legacy series"
    );
    assert!(
        read_delivered_bytes_sum_mode(&svc.metrics, "in_window", "legacy") > 0.0,
        "snapshot-mode in_window delivered-bytes sum must be > 0 (seeded state)"
    );

    // ...and the opposite `catchup_mode="snapshot"` series stays at zero across
    // all three instruments — this regime is NOT delivered by `send_full_state`.
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "in_window", "snapshot"),
        0,
        "the snapshot series must NOT move for a coalesced in_window reconnect"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "in_window", "snapshot"),
        0,
        "the snapshot duration series must NOT move for a coalesced in_window reconnect"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "in_window", "snapshot"),
        0,
        "the snapshot delivered-bytes series must NOT move for a coalesced in_window reconnect"
    );

    // The mode-less subset-match reader still resolves the series — the additive
    // catchup_mode label does not break the pre-existing mode-less read.
    assert_eq!(
        read_regime_counter(&svc.metrics, "in_window"),
        1,
        "the additive catchup_mode label must not break the mode-less reader"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error path: layer-not-found returns BEFORE regime is assigned → no metric.
// ═══════════════════════════════════════════════════════════════════════════════
//
// `run_catch_up_inner` resolves the layer mapping FIRST via
// `get_layer_mapping(...).await?`; a missing mapping short-circuits with
// `ServiceError::NotFound` before any of the four regime branches are taken.
// The engineer's wrapper records `observe_subscribe_regime` only on the
// success path (after `run_catch_up_inner` returns Ok), so the error path
// must not move any counter.
//
// This is the cleanest error-path probe available without a fault-injection
// seam — `read_object_header` and `xrange_values` failures would require
// reaching past `pub(crate)` boundaries that this test deliberately respects.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn error_path_layer_not_found_records_no_regime_metric() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    // Create a session but no layer; subscribing to a nonexistent layer name
    // hits the `get_layer_mapping(...) -> Ok(None)` branch in
    // `run_catch_up_inner` and short-circuits with NotFound BEFORE any regime
    // is assigned.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("err-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let sub_req = subscribe_request(&session_id, "nonexistent-layer", 0, 0, "sub-err");
    let response = handle_subscribe_read_layer(&svc, sub_req).await.expect(
        "handle_subscribe_read_layer returns Ok(stream) even for layer-not-found; \
             the error surfaces ON the stream from the spawned task",
    );
    let mut stream = response.into_inner();

    // The spawned subscription task discovers the missing layer and forwards
    // a NotFound Status onto the response stream.
    let next = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("stream must yield within 5s")
        .expect("stream must yield a terminal status, not close silently");
    let status = next.expect_err("layer-not-found must surface as Err Status");
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "missing layer must surface as NotFound"
    );

    // No sleep needed here: receiving the terminal `Err(Status::NotFound)`
    // above is itself the event-driven signal that the spawned subscription
    // task hit the early `?` exit in catch-up and tore down before reaching
    // any regime-classification or metric-write call site (those live INSIDE
    // `run_catch_up`, past the failing layer-state read). Adding a
    // precautionary sleep here would re-introduce exactly the timing-coupled
    // pattern that b.ohv item 5 eliminated elsewhere in this file.
    for regime in REGIMES {
        assert_eq!(
            read_regime_counter(&svc.metrics, regime),
            0,
            "no regime counter must move on the layer-not-found error path (regime={regime})"
        );
        assert_eq!(
            read_regime_histogram_sample_count(&svc.metrics, regime),
            0,
            "no regime histogram must record samples on the error path (regime={regime})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Plan b.msb Task 4: additive `catchup_mode` (snapshot|legacy) dimension.
// ═══════════════════════════════════════════════════════════════════════════════
//
// Task 4's source change threads an additive `catchup_mode` label onto ALL
// three catch-up instruments (`live_subscribe_regime_total`,
// `live_subscribe_catch_up_duration_seconds`, and the new
// `live_subscribe_catch_up_delivered_bytes` histogram). `catchup_mode="snapshot"`
// is emitted ONLY by `send_full_state`'s Snapshot arm (brand_new faithful
// full-state under `catchup_snapshot_enabled = true`); every other catch-up —
// including out_of_window even under snapshot mode — is `"legacy"`.
//
// The two tests below drive the SAME brand_new full-state path under the two
// modes and assert each lands on its own `catchup_mode` series while leaving
// the other's series at zero — proving the two modes emit distinct series
// values on one metric NAME (additive, non-breaking).

/// Apply `n` distinct-section updates so the full-state snapshot carries real
/// content (a from-empty full state can encode to zero OMPF bytes, which would
/// make the delivered-bytes sum assertion vacuous). Section ids start at
/// `first_section` and step by 1.
async fn seed_sections(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    writer: &str,
    first_section: u64,
    n: u64,
) {
    for i in 0..n {
        apply_unique_update(
            svc,
            session_id,
            layer_name,
            object_id,
            writer,
            first_section + i,
        )
        .await;
    }
}

// ── (a) snapshot-mode brand_new lands on catchup_mode="snapshot" ────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_snapshot_mode_lands_on_snapshot_catchup_mode_series() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "cm-snap-session", "cm-snap-layer").await;
    // Seed real state so the full-state snapshot delivers > 0 OMPF bytes.
    seed_sections(
        &svc,
        &session_id,
        "cm-snap-layer",
        object_id,
        "seed",
        300,
        3,
    )
    .await;

    // First-time subscribe (have_object_id=0, have_version=0) → brand_new,
    // rerouted to the corrected-snapshot builder under `Snapshot` mode.
    let sub_req = subscribe_request(&session_id, "cm-snap-layer", 0, 0, "sub-cm-snap");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert!(
        header.total_size > 0,
        "seeded snapshot full-state must deliver a non-empty body"
    );
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a zero-body ack"
    );

    // The observation lands on the `catchup_mode="snapshot"` series...
    await_counter_mode(&svc.metrics, "brand_new", "snapshot", 1).await;
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "snapshot brand_new must increment the catchup_mode=snapshot counter"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "snapshot brand_new must record exactly one duration sample on the snapshot series"
    );
    assert!(
        read_regime_duration_sum_mode(&svc.metrics, "brand_new", "snapshot") > 0.0,
        "snapshot brand_new duration sum must be > 0"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "snapshot brand_new must record exactly one delivered-bytes sample on the snapshot series"
    );
    assert!(
        read_delivered_bytes_sum_mode(&svc.metrics, "brand_new", "snapshot") > 0.0,
        "snapshot brand_new delivered-bytes sum must be > 0"
    );

    // ...and NOT on the legacy series.
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "legacy"),
        0,
        "the legacy series must NOT move when snapshot mode fires"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "legacy"),
        0,
        "the legacy duration series must NOT move when snapshot mode fires"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "legacy"),
        0,
        "the legacy delivered-bytes series must NOT move when snapshot mode fires"
    );
}

// ── (a) legacy/Off brand_new lands on catchup_mode="legacy" ─────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_legacy_mode_lands_on_legacy_catchup_mode_series() {
    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, false);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "cm-legacy-session", "cm-legacy-layer").await;
    seed_sections(
        &svc,
        &session_id,
        "cm-legacy-layer",
        object_id,
        "seed",
        300,
        3,
    )
    .await;

    // First-time subscribe under `Off` → brand_new via the legacy full-content
    // path → catchup_mode="legacy".
    let sub_req = subscribe_request(&session_id, "cm-legacy-layer", 0, 0, "sub-cm-legacy");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let header = drain_catch_up(&mut stream).await;
    assert!(
        header.total_size > 0,
        "seeded legacy full-state must deliver a non-empty body"
    );
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a zero-body ack"
    );

    // The observation lands on the `catchup_mode="legacy"` series...
    await_counter_mode(&svc.metrics, "brand_new", "legacy", 1).await;
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "legacy brand_new must increment the catchup_mode=legacy counter"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "legacy brand_new must record exactly one duration sample on the legacy series"
    );
    assert!(
        read_regime_duration_sum_mode(&svc.metrics, "brand_new", "legacy") > 0.0,
        "legacy brand_new duration sum must be > 0"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "legacy"),
        1,
        "legacy brand_new must record exactly one delivered-bytes sample on the legacy series"
    );
    assert!(
        read_delivered_bytes_sum_mode(&svc.metrics, "brand_new", "legacy") > 0.0,
        "legacy brand_new delivered-bytes sum must be > 0"
    );

    // ...and NOT on the snapshot series.
    assert_eq!(
        read_regime_counter_mode(&svc.metrics, "brand_new", "snapshot"),
        0,
        "the snapshot series must NOT move when legacy mode fires"
    );
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "snapshot"),
        0,
        "the snapshot duration series must NOT move when legacy mode fires"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "snapshot"),
        0,
        "the snapshot delivered-bytes series must NOT move when legacy mode fires"
    );

    // The mode-less subset-match reader (used by the pre-existing tests) still
    // resolves the series — the additive label does not break the legacy read.
    assert_eq!(
        read_regime_counter(&svc.metrics, "brand_new"),
        1,
        "the additive catchup_mode label must not break the mode-less reader"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// (b) Guard: existing metric NAMES + the four `regime` label values are neither
// renamed nor removed; the `catchup_mode` label is additive (not a rename).
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn regime_metric_names_and_labels_are_stable_guard() {
    // Pin the regime taxonomy at the source-of-truth const: a future edit that
    // renames or drops a regime value fails to compile-match here.
    assert_eq!(
        REGIMES,
        &["brand_new", "up_to_date", "in_window", "out_of_window"],
        "the pinned regime label set must stay exactly these four values"
    );

    let tv = MetricsTestValkey::setup().await;
    let svc = build_metrics_service(&tv, 4096, true);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "guard-session", "guard-layer").await;
    seed_sections(&svc, &session_id, "guard-layer", object_id, "seed", 300, 2).await;

    let sub_req = subscribe_request(&session_id, "guard-layer", 0, 0, "sub-guard");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();
    drain_catch_up(&mut stream).await;
    drain_catch_up(&mut stream).await;
    await_counter_mode(&svc.metrics, "brand_new", "snapshot", 1).await;

    let text = live_edit::observability::metrics::test_util::render(&svc.metrics);

    // Metric NAMES must still resolve (no rename / removal).
    for name in [
        "live_subscribe_regime_total",
        "live_subscribe_catch_up_duration_seconds",
        "live_subscribe_catch_up_delivered_bytes",
    ] {
        assert!(
            live_edit::observability::metrics::test_util::has_metric(&text, name),
            "metric name `{name}` must be present (guards against rename/removal)"
        );
    }

    // The `regime` label KEY must still resolve on the counter with the expected
    // brand_new VALUE, and the additive `catchup_mode` label must coexist on the
    // SAME series (proving it is additive, not a rename of `regime`).
    assert!(
        live_edit::observability::metrics::test_util::sample_value(
            &text,
            "live_subscribe_regime_total",
            &[("regime", "brand_new"), ("service", TEST_SERVICE)],
        )
        .is_some(),
        "the `regime` label key + brand_new value must still resolve"
    );
    assert!(
        live_edit::observability::metrics::test_util::sample_value(
            &text,
            "live_subscribe_regime_total",
            &[
                ("regime", "brand_new"),
                ("service", TEST_SERVICE),
                ("catchup_mode", "snapshot"),
            ],
        )
        .is_some(),
        "the additive `catchup_mode` label must coexist with the pinned `regime` label"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// (c) Long / trimmed snapshot-mode catch-up cost measurement.
// ═══════════════════════════════════════════════════════════════════════════════
//
// Builds a `Snapshot` service with a small stream MAXLEN, applies a
// representative LONG series of distinct-section commits, force-trims the stream
// (XTRIM MINID) so no live-window history survives, then drives a brand_new
// first-subscribe snapshot-mode catch-up. The full-state snapshot is
// reconstructed from materialized state (NOT the trimmed stream), so it still
// delivers every section. We RECORD the observable catch-up duration and
// delivered size, assert both positive+finite with exactly one histogram sample
// on the snapshot brand_new series, and emit both figures for the downstream
// doc subtask.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn long_trimmed_snapshot_catch_up_records_positive_cost() {
    const SECTION_COUNT: u64 = 40;

    let tv = MetricsTestValkey::setup().await;
    // Small MAXLEN so the stream is aggressively bounded; then force-trim below.
    let svc = build_metrics_service(&tv, 8, true);

    let (session_id, object_id) =
        create_session_and_layer_metrics(&tv, &svc, "long-snap-session", "long-snap-layer").await;

    // Representative long series: tens of distinct sections (seq 1 = create,
    // then SECTION_COUNT updates → server seq = SECTION_COUNT + 1).
    seed_sections(
        &svc,
        &session_id,
        "long-snap-layer",
        object_id,
        "long-writer",
        1_000,
        SECTION_COUNT,
    )
    .await;
    let current_seq = SECTION_COUNT + 1;

    // Force the live window empty: XTRIM MINID past the last id so nothing
    // remains on the stream (the snapshot must reconstruct from materialized
    // state, not the trimmed history).
    let stream_key = tv.key_schema().layer_stream(object_id);
    let minid = format!("{}-0", current_seq + 1);
    let cap: XCap = XCap::try_from(("MINID", "=", minid.as_str())).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force a trimmed stream must succeed");

    // brand_new first-subscribe (have_object_id=0, have_version=0) → snapshot
    // full-state, reconstructed from materialized state despite the trim.
    let sub_req = subscribe_request(&session_id, "long-snap-layer", 0, 0, "sub-long-snap");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the full-state delivery (bounded by the 5s deadline in
    // `drain_catch_up`). `header.total_size` = delivered content-chunk byte
    // total (drain_catch_up fills exactly that many bytes).
    let header = drain_catch_up(&mut stream).await;
    let delivered_wire_bytes = header.total_size;
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand_new trailer must be a zero-body ack under snapshot mode"
    );

    await_counter_mode(&svc.metrics, "brand_new", "snapshot", 1).await;

    // Exactly one histogram sample on the snapshot brand_new series.
    assert_eq!(
        read_regime_duration_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "exactly one duration sample must land on the snapshot brand_new series"
    );
    assert_eq!(
        read_delivered_bytes_count_mode(&svc.metrics, "brand_new", "snapshot"),
        1,
        "exactly one delivered-bytes sample must land on the snapshot brand_new series"
    );

    // Observable cost figures.
    let catch_up_duration_secs =
        read_regime_duration_sum_mode(&svc.metrics, "brand_new", "snapshot");
    let delivered_metric_bytes =
        read_delivered_bytes_sum_mode(&svc.metrics, "brand_new", "snapshot");

    assert!(
        catch_up_duration_secs.is_finite() && catch_up_duration_secs > 0.0,
        "catch-up duration must be positive + finite, got {catch_up_duration_secs}"
    );
    assert!(
        delivered_metric_bytes.is_finite() && delivered_metric_bytes > 0.0,
        "delivered-bytes metric must be positive + finite, got {delivered_metric_bytes}"
    );
    assert!(
        delivered_wire_bytes > 0,
        "delivered wire body must be non-empty, got {delivered_wire_bytes}"
    );

    // Emit the measurement so it is captured for the downstream doc subtask.
    eprintln!(
        "[long_trimmed_snapshot_catch_up] scene: {SECTION_COUNT} sections \
         ({SECTION_COUNT} commits after create, server seq={current_seq}); \
         catch_up_duration_seconds={catch_up_duration_secs:.6}; \
         delivered_bytes(metric OMPF)={delivered_metric_bytes}; \
         delivered_bytes(wire body)={delivered_wire_bytes}"
    );
}
