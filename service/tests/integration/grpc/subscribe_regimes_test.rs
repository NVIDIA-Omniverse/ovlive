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

// Integration tests for the three catch-up regimes (Bee b.ebu) plus the
// Streams-transport regressions called out in Epic 4 Task 1.
//
// Each test calls `handle_subscribe_read_layer` / `handle_update_layer` /
// `handle_create_layer` / `handle_delete_layer` directly against a
// LiveEditService backed by a real Valkey (TestValkey fixture) with per-test
// key prefixes for isolation. See SDD §12.3 / §12.4 / §9 / §10 for the
// authoritative behavioural spec.
//
// Coverage (matching b.ebu Epic 4 Task 1 acceptance criteria):
//   1. Fast-subscriber per-commit (Regime 1 steady-state)
//   2. In-window reconnect (Regime 2, `XRANGE` slice)
//   3. Out-of-window fallback with state-parity (Regime 3, one coalesced diff
//      built from the catch-up's atomic read; above the atomic-read limits,
//      from the layer's shared fenced read)
//   4. CreateLayer overwrite during active subscription
//   5. DeleteLayer during active subscription terminates with NOT_FOUND
//   6. Self-echo suppression in per-commit path

// The `test-fault-injection` gate repeated on the imports below (and on the
// b.vd8 stable-cut helper cluster further down) is load-bearing, not cosmetic:
// CI's `no-default-features:build-lint-test` job compiles this target WITHOUT
// that feature and runs clippy at `-D warnings`, so an item reachable only from
// a `#[cfg(feature = "test-fault-injection")]` test is a hard `unused_imports` /
// `dead_code` build failure there. Gate the item with the same feature its
// consumers carry rather than reaching for `#[allow(dead_code)]`, which would
// also silence a genuinely-orphaned helper under the full-feature build.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
#[cfg(feature = "test-fault-injection")]
use std::pin::Pin;
#[cfg(feature = "test-fault-injection")]
use std::sync::atomic::AtomicBool;
#[cfg(all(
    feature = "test-fault-injection",
    feature = "test-checkpoint-fault-injection"
))]
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use dashmap::DashMap;
#[cfg(feature = "test-fault-injection")]
use fred::interfaces::HashesInterface;
use fred::interfaces::{KeysInterface, StreamsInterface};
use fred::types::streams::XCap;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::config::{AuthProtocol, Config};
use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::grpc::layer::{
    handle_create_layer, handle_delete_layer, handle_stat_layer, handle_update_layer,
};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, DeleteLayerRequest, StatLayerRequest,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::sibling_model::{server_child_lists, SiblingModel, READERS};
use oodf_ffi::{
    CommandCreateSection, CommandDiffChangeSection, CommandDiffDeleteSection, OwnedDelta, SetField,
    StructCommand,
};

/// Anti-hang ceiling for "wait until the subscriber task / stream delivers the
/// next message" awaits. Every such wait is bounded so a genuine logic deadlock
/// fails the test instead of hanging `cargo test` forever. The value is
/// deliberately generous: a `tokio::time::timeout` returns the instant the
/// awaited work completes, so a large ceiling costs nothing on the happy path —
/// it only changes behaviour when something is actually stuck. 30 s clears the
/// worst-case rebootstrap chain (several Valkey round-trips) even under heavy
/// parallel load on a shared Valkey, where a tight 5 s budget flaked. Must stay
/// at least as large as the per-read budget inside `recv_message` so the
/// descriptive inner timeout fires first on a stalled read.
const SUBSCRIBER_DEADLINE: Duration = Duration::from_secs(30);

/// A non-reset delta carrying a single `DiffChangeSection`. The shared
/// fixture `delta_with_diff_change_section` uses `is_diff=true, base_version=0`
/// which triggers the SDD §9.1.3 reset path — that would wipe state between
/// updates and defeat the accumulation the regime tests rely on. This helper
/// uses `is_diff=false` so the command applies against current state without
/// resetting it.
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

use super::helpers::*;

// ── Test-local helpers ──────────────────────────────────────────────────────

/// A per-test Valkey fixture with its own key-prefix namespace AND independent
/// control over the txn pool size AND the subscriber pool size.
///
/// `TestValkey::setup()` only exposes a single `valkey_pool_size` knob (the
/// subscriber_pool is hard-coded to 4). The M2 starvation smoke test
/// (`subscribers_do_not_starve_concurrent_update_layer`) must pin those two
/// pools to different sizes — txn_pool=2, subscriber_pool=4 — to prove that
/// a saturated subscriber pool does not block transactions. This fixture is
/// the only place in the crate that needs that dual-knob control, which is
/// why it remains file-local.
struct PooledTestValkey {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
}

impl PooledTestValkey {
    async fn setup() -> Self {
        // Default: auto-sized data/txn pool, roomy subscriber pool. Suitable
        // for tests that don't need to pin either dimension.
        Self::setup_with_pools(0, 8).await
    }

    /// Explicit pool sizing — `valkey_pool_size` controls the shared data
    /// pool AND the txn `ExclusivePool` (both are sized from the same knob
    /// in production), while `subscriber_pool_size` controls the separate
    /// stream-reader connections. Required for the M2 starvation smoke test
    /// that pins txn to a small size.
    async fn setup_with_pools(valkey_pool_size: u32, subscriber_pool_size: u32) -> Self {
        Self::setup_full(
            valkey_pool_size,
            subscriber_pool_size,
            live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        )
        .await
    }

    /// [`Self::setup`] with an atomic-read item ceiling of one item: every
    /// layer is above it, so every catch-up's atomic read answers `TOO_LARGE`
    /// (or is skipped) and the catch-up reads through the layer's shared
    /// fenced-read flight.
    #[cfg(feature = "test-fault-injection")]
    async fn setup_above_ceiling() -> Self {
        Self::setup_full(0, 8, 1).await
    }

    async fn setup_full(
        valkey_pool_size: u32,
        subscriber_pool_size: u32,
        snapshot_atomic_read_max_items: u64,
    ) -> Self {
        let valkey_url =
            env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        Self::setup_with_url_and_pools(
            valkey_url,
            valkey_pool_size,
            subscriber_pool_size,
            snapshot_atomic_read_max_items,
        )
        .await
    }

    /// [`Self::setup`] against `valkey_url` instead of `TEST_VALKEY_URL` — a
    /// `ScriptReplyProxy` in front of the test store, for instance.
    async fn setup_with_url(valkey_url: String) -> Self {
        Self::setup_with_url_and_pools(
            valkey_url,
            0,
            8,
            live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        )
        .await
    }

    async fn setup_with_url_and_pools(
        valkey_url: String,
        valkey_pool_size: u32,
        subscriber_pool_size: u32,
        snapshot_atomic_read_max_items: u64,
    ) -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let key_schema = KeySchema::new(&prefix);

        let config = Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url,
            valkey_pool_size,
            subscriber_pool_size,
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
            snapshot_atomic_read_max_items,
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
            .expect("PooledTestValkey: connect must succeed");

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

impl Drop for PooledTestValkey {
    fn drop(&mut self) {
        // SCAN + UNLINK all keys under this test's prefix so parallel tests
        // don't observe each other's state.
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

/// Build a LiveEditService backed by the pooled fixture, with a custom
/// `layer_stream_maxlen`.
fn build_pooled_service(
    tv: &PooledTestValkey,
    permissions: Arc<dyn live_edit::auth::permissions::PermissionsClient>,
    maxlen: u64,
) -> LiveEditService {
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        maxlen,
    ));
    build_pooled_service_from_layer_store(tv, permissions, layer_store)
}

/// Like [`build_pooled_service`] but takes a pre-built `LayerStore`, so a test
/// can install a `LayerStore`-level seam (e.g. the overwrite-cleanup hold gate)
/// before the store is shared into the service. All other wiring is identical.
fn build_pooled_service_from_layer_store(
    tv: &PooledTestValkey,
    permissions: Arc<dyn live_edit::auth::permissions::PermissionsClient>,
    layer_store: Arc<LayerStore>,
) -> LiveEditService {
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );
    // Inline disabled session-recovery hook — `PooledTestValkey` isn't a
    // `TestValkey`, so the canonical helper signature doesn't apply.
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
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: permissions,
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

/// Create a session and layer via the given pooled service. Mirrors the shared
/// helper `create_session_and_layer` but doesn't depend on `TestValkey`.
async fn create_session_and_layer_pooled(
    tv: &PooledTestValkey,
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

/// Build an UpdateLayerRequest header.
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

/// Build an UpdateLayerRequest data chunk.
fn update_data(payload: Vec<u8>) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(
            live_edit::grpc::proto::UpdateLayerChunk { data: payload },
        )),
    }
}

/// Issue a single UpdateLayer carrying a `DiffChangeSection` with a fresh
/// section_id. The varying section_id guarantees each commit produces a
/// non-empty per-commit diff (state really mutates).
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

/// Subscribe and return the response stream.
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

/// DeleteLayer request.
fn delete_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    let mut request = Request::new(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Collect a header + any data chunks until the next header, ending the RPC,
/// or a short quiescent timeout. Returns the header and reassembled body bytes.
async fn recv_message<S>(
    stream: &mut S,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    // Time-bound every stream read so a drain that awaits a message the server
    // will never send fails fast with a descriptive message instead of hanging
    // `cargo test` forever (b.rdr). Use the shared `SUBSCRIBER_DEADLINE` (30 s):
    // it converts a genuine logic-error hang into a descriptive failure while
    // staying generous enough not to flake under parallel-Valkey contention,
    // where a tight per-read budget made the multi-round-trip rebootstrap paths
    // (b.1kf) intermittently exceed the timeout on a slow-but-correct run.
    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
        .await
        .expect(
            "recv_message: stream timed out (30s) waiting for the next header — a drain is \
             awaiting a message the server never sent (likely a caught-up-ack count mismatch)",
        )
        .expect("stream closed unexpectedly")
        .expect("stream yielded an error");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header, got: {other:?}"),
    };

    let mut buf = Vec::new();
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
            .await
            .expect(
                "recv_message: stream timed out (30s) waiting for a body chunk before \
                 total_size was reached",
            )
            .expect("stream closed before total_size reached")
            .expect("stream yielded an error");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk while filling body, got: {other:?}"),
        }
    }

    (header, buf)
}

/// Drain the brand-new / rebootstrap catch-up trailer: the `total_size == 0`
/// caught-up ack header that `send_full_state` emits after the full-state
/// content chunks (b.rdr). Every brand-new catch-up — the genuine first-time
/// subscribe (`object_id=0, sequence_num=0`) AND the overwrite-on-reconnect
/// rebootstrap that routes through `catch_up_brand_new(NEW)` — terminates with
/// this marker. Asserts the trailer is a zero-body header carrying the
/// just-delivered content version, then returns it for spot checks.
///
/// `expected_object_id` / `expected_sequence_num` are the object generation and
/// server sequence_num the immediately-preceding full-state content header
/// carried; the caught-up ack must echo both.
async fn recv_caught_up_ack<S>(
    stream: &mut S,
    expected_object_id: u64,
    expected_sequence_num: u64,
) -> live_edit::grpc::proto::SubscribeReadLayerHeader
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let (ack, body) = recv_message(stream).await;
    assert_eq!(
        ack.total_size, 0,
        "brand-new catch-up trailer must be a zero-body caught-up ack header \
         (total_size==0), got total_size={}",
        ack.total_size
    );
    assert!(
        body.is_empty(),
        "caught-up ack header must carry no body bytes"
    );
    assert_eq!(
        ack.object_id, expected_object_id,
        "caught-up ack object_id must match the full-state content header"
    );
    assert_eq!(
        ack.sequence_num, expected_sequence_num,
        "caught-up ack sequence_num must equal the delivered content version"
    );
    ack
}

/// Wait briefly for the subscriber stream to settle — then assert nothing else
/// arrives in that window. Useful for confirming self-echo suppression.
async fn assert_no_next_message<S>(stream: &mut S, timeout: Duration)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    match tokio::time::timeout(timeout, stream.next()).await {
        Err(_elapsed) => {} // no message — good
        Ok(None) => {}      // clean close — also acceptable
        Ok(Some(Ok(msg))) => panic!("expected no next message; got: {msg:?}"),
        Ok(Some(Err(e))) => panic!("expected no next message; got error: {e}"),
    }
}

/// Apply an OMPF-bytes diff to a local ObjectState via the same pipeline the
/// server uses (OmpfContainer::parse → oodf_ffi::parse_delta → apply_delta).
fn apply_ompf_to_state(state: &mut ObjectState, ompf_bytes: &[u8]) {
    let container = OmpfContainer::parse(ompf_bytes).expect("parse OMPF");
    let owned = if container.delta_bytes.is_empty() {
        None
    } else {
        Some(oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta"))
    };
    apply_delta(state, owned).expect("apply_delta");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Fast-subscriber per-commit (Regime 1 steady-state)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fast_subscriber_receives_each_commit_individually() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "fast-sub-session", "fast-sub-layer").await;

    // Subscribe first, drain the Regime-1 full-state header, then tail from $.
    let sub_req = subscribe_request(&session_id, "fast-sub-layer", 0, 0, "subscriber-1");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Regime 1 sends a full-content header (seq=1 for the freshly-created layer).
    let (initial, _body) = recv_message(&mut stream).await;
    assert_eq!(
        initial.sequence_num, 1,
        "initial header should be at seq=1, got {}",
        initial.sequence_num
    );
    assert_eq!(initial.object_id, object_id);

    // b.rdr: the brand-new catch-up terminates with a total_size==0 caught-up
    // ack carrying the delivered version (seq=1) before steady state begins.
    recv_caught_up_ack(&mut stream, object_id, 1).await;

    // Issue N = 10 back-to-back updates from a *different* writer client_id so
    // self-echo suppression doesn't filter them out for our subscriber.
    const N: u64 = 10;
    for i in 0..N {
        apply_unique_update(
            &svc,
            &session_id,
            "fast-sub-layer",
            object_id,
            "writer-client",
            100 + i,
        )
        .await;
    }

    // Expect N distinct per-commit messages with contiguous seqs 2..=N+1.
    for i in 0..N {
        let expected_seq = 2 + i;
        let (header, body) = recv_message(&mut stream).await;
        assert_eq!(
            header.sequence_num, expected_seq,
            "per-commit #{i} should have seq={expected_seq}, got {}",
            header.sequence_num
        );
        assert_eq!(header.object_id, object_id, "object_id should be stable");
        assert!(
            !body.is_empty(),
            "per-commit diff #{i} body must be non-empty (section was added)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. In-window reconnect (Regime 2)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_window_reconnect_replays_individual_per_commit_diffs() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "inwindow-session", "inwindow-layer").await;

    // Advance to sequence_num=5 (4 updates after the create-seq=1).
    for i in 0..4u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "inwindow-layer",
            object_id,
            "writer-initial",
            100 + i,
        )
        .await;
    }

    // Writer issues M = 3 more updates — subscriber is still "disconnected".
    // MAXLEN = 4096 here, so entries 6-0, 7-0, 8-0 are all retained.
    const M: u64 = 3;
    for i in 0..M {
        apply_unique_update(
            &svc,
            &session_id,
            "inwindow-layer",
            object_id,
            "writer-after-disconnect",
            200 + i,
        )
        .await;
    }

    // Reconnect with have_version=5 — expect Regime-2 replay of 6, 7, 8.
    let sub_req = subscribe_request(&session_id, "inwindow-layer", object_id, 5, "sub-2");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let mut observed_seqs = Vec::new();
    for _ in 0..M {
        let (header, body) = recv_message(&mut stream).await;
        observed_seqs.push(header.sequence_num);
        assert_eq!(header.object_id, object_id);
        assert!(
            !body.is_empty(),
            "per-commit diff body must be non-empty — state mutation happened"
        );
        // Regime-2 per-commit diffs carry total_size > 0; the Regime-3 coalesced
        // path in this scenario would deliver ONE message (ruled out by seeing
        // 3 separate headers). Still sanity-check total_size > 0.
        assert!(header.total_size > 0);
    }

    assert_eq!(
        observed_seqs,
        vec![6, 7, 8],
        "in-window replay must deliver contiguous seqs 6, 7, 8"
    );

    // After the per-commit replay, the in-window catch-up now terminates with
    // the exactly-once caught-up ack (`total_size==0`) emitted centrally by
    // `run_catch_up` (b.dy5). Its `tracked_version` is the last *replayed*
    // endpoint — here seq=8, which equals `current_seq` because the replay
    // covered the whole gap. Before b.dy5 the in-window path emitted no
    // terminator and the subscriber never fired the client's `OkLatest`.
    recv_caught_up_ack(&mut stream, object_id, 8).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Out-of-window fallback with state-parity (Regime 3)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn out_of_window_reconnect_coalesces_and_reaches_state_parity() {
    let tv = PooledTestValkey::setup().await;
    // Small MAXLEN so XADD trims aggressively and our XTRIM can force the
    // subscriber cursor out of the window without needing 4097 writes.
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "oow-session", "oow-layer").await;

    // Walk the subscriber forward to seq=5 with a trackable, reproducible
    // sequence of section IDs. Keep a local ObjectState mirror so we can
    // reconstruct the subscriber's view after catch-up.
    let mut local_state = ObjectState::default();

    // Apply the *same* empty create delta the test helper used so local_state
    // matches the server's seq=1 snapshot.
    apply_delta(&mut local_state, None).expect("mirror create_layer (empty delta)");

    for i in 0..4u64 {
        let section_id = 100 + i;
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer",
            object_id,
            "writer-pre-disconnect",
            section_id,
        )
        .await;
        // Mirror locally.
        let delta = incremental_diff_change_section(section_id);
        apply_delta(&mut local_state, Some(delta)).expect("mirror pre-disconnect update");
    }
    assert_eq!(local_state.sequence_num, 5);

    // Writer issues 10 more updates while the subscriber is disconnected —
    // server seq climbs to 15. MAXLEN=4 means XADD trimming leaves only the
    // tail, but trimming is approximate ("~"), so make it deterministic with
    // an explicit XTRIM MINID = (current_seq - 1)-0. After writes complete
    // we force MINID >= 15-0 so nothing the subscriber needs (seqs 6..=15)
    // remains on the stream except the final entry.
    for i in 0..10u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "oow-layer",
            object_id,
            "writer-post-disconnect",
            200 + i,
        )
        .await;
    }

    // Force out-of-window deterministically: trim the stream down so the
    // subscriber's start_id (6-0) is not present.
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "15-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    // Reconnect with have_version=5 — expect ONE coalesced diff covering 6..=15.
    let sub_req = subscribe_request(&session_id, "oow-layer", object_id, 5, "sub-oow");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.sequence_num, 15,
        "coalesced diff must carry the current server sequence_num"
    );
    assert_eq!(header.object_id, object_id);
    assert!(
        !body.is_empty(),
        "coalesced diff must have a non-empty payload"
    );

    // Out-of-window catch-up now terminates with the exactly-once caught-up
    // ack (`total_size==0`) emitted centrally by `run_catch_up` (b.dy5). For
    // Regime 3 the coalesced diff is the whole catch-up, so the terminator's
    // `tracked_version` equals the coalesced header's `current_seq` (15). The
    // client gates stage-apply on this `OkLatest` marker; before b.dy5 the
    // out-of-window path emitted no terminator and the subscriber never left
    // initial-sync. After draining the terminator the consumer opens at $, so
    // no further messages land until new writes (of which there are none here).
    recv_caught_up_ack(&mut stream, object_id, 15).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // Reconstruct the subscriber's view: start at the local mirror (seq=5) and
    // apply the received coalesced diff. `apply_delta` unconditionally bumps
    // sequence_num by exactly 1, so the subscriber's local seq only advances
    // from 5 → 6 after one coalesced diff. This is by design — a coalesced
    // diff discards the per-commit history; what it guarantees is that the
    // resulting structural state (sections, fields, children, time samples)
    // matches what the server has materialized. That's exactly what this
    // state-parity assertion checks, and it's what the load measurements
    // called out as a follow-up.
    apply_ompf_to_state(&mut local_state, &body);

    // Compare structural state against the server's materialized snapshot.
    // Version annotations (first_version, last_version) legitimately differ
    // under coalesced catch-up — a single diff discards the per-commit
    // history, so the subscriber's reconstructed sections all carry the
    // `apply_delta`-bumped version rather than the per-commit versions the
    // server recorded. For state-parity the structural identity of each
    // section (id, parent, name, type, order) is what must match.
    let server_snapshot =
        read_object_state_for_test(&svc.valkey, &svc.key_schema, &object_id.to_string())
            .await
            .expect("read_object_state_for_test");
    let server_state = snapshot_to_object_state(&server_snapshot);

    let section_ids_local: std::collections::HashSet<u64> =
        local_state.sections.keys().copied().collect();
    let section_ids_server: std::collections::HashSet<u64> =
        server_state.sections.keys().copied().collect();
    assert_eq!(
        section_ids_local, section_ids_server,
        "reconstructed section IDs must equal server section IDs"
    );
    for (sid, local_sec) in &local_state.sections {
        let server_sec = server_state
            .sections
            .get(sid)
            .expect("section id present on server");
        assert_eq!(
            local_sec.parent_id, server_sec.parent_id,
            "section {sid} parent must match"
        );
        assert_eq!(
            local_sec.name, server_sec.name,
            "section {sid} name must match"
        );
        assert_eq!(
            local_sec.section_type, server_sec.section_type,
            "section {sid} type must match"
        );
        assert_eq!(
            local_sec.section_order, server_sec.section_order,
            "section {sid} order must match"
        );
    }

    // Children lists structure must match: same child-id set for each parent.
    assert_eq!(
        local_state
            .children
            .keys()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        server_state
            .children
            .keys()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        "reconstructed children-parent IDs must equal server's"
    );
    for (parent, local_kids) in &local_state.children {
        let server_kids = server_state
            .children
            .get(parent)
            .expect("parent present on server");
        assert_eq!(
            local_kids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            server_kids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            "children of parent {parent} must match (as sets)"
        );
    }

    // Fields and time samples: compare (section_id, key_name) / (section_id,
    // time) keys and the stored values; again ignore version annotations.
    let local_field_values: std::collections::HashMap<_, _> = local_state
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), v.value.clone()))
        .collect();
    let server_field_values: std::collections::HashMap<_, _> = server_state
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), v.value.clone()))
        .collect();
    assert_eq!(
        local_field_values, server_field_values,
        "reconstructed fields (key + value) must equal server fields"
    );

    let local_ts_values: std::collections::HashMap<_, _> = local_state
        .time_samples
        .iter()
        .map(|(k, v)| (*k, v.value.clone()))
        .collect();
    let server_ts_values: std::collections::HashMap<_, _> = server_state
        .time_samples
        .iter()
        .map(|(k, v)| (*k, v.value.clone()))
        .collect();
    assert_eq!(
        local_ts_values, server_ts_values,
        "reconstructed time samples must equal server time samples"
    );

    // Subscriber tracks the stream header's sequence_num (from the diff
    // header), not the `ObjectState.sequence_num` that `apply_delta` bumps
    // locally. Assert the stream-header seq matches server seq.
    assert_eq!(
        header.sequence_num, server_state.sequence_num,
        "diff-header seq must equal server seq"
    );

    // Dual-assertion per b.azz N3: honour the letter of Bee b.ebu's
    // acceptance criterion by calling StatLayer after catch-up alongside the
    // (stronger) structural `read_object_state_for_test` comparison above. StatLayer
    // returns *metadata* (proto: created_by_user_name, created_at,
    // last_modified_at) — a different surface from materialized sections and
    // fields. We invoke it from the subscriber's perspective and compare the
    // result against a second, independent StatLayer call; on a single
    // service both calls must return byte-identical metadata for the same
    // (session_id, layer_name), and the values must be internally consistent
    // with the write activity the test performed.
    // Subscriber's perspective: the client that just finished catching up
    // asks for layer metadata.
    let sub_stat = handle_stat_layer(&svc, stat_layer_req(&session_id, "oow-layer"))
        .await
        .expect("subscriber-perspective StatLayer after catch-up must succeed")
        .into_inner();
    // Independent client's perspective: a different caller asks for the same
    // layer's metadata. On a single service this must return byte-identical
    // metadata; the assertion guards against any regression where per-caller
    // state accidentally leaks into the StatLayer response.
    let other_stat = handle_stat_layer(&svc, stat_layer_req(&session_id, "oow-layer"))
        .await
        .expect("independent StatLayer call must succeed")
        .into_inner();
    assert_eq!(
        sub_stat.created_by_user_name, other_stat.created_by_user_name,
        "StatLayer created_by_user_name must be identical across callers"
    );
    assert_eq!(
        sub_stat.created_at, other_stat.created_at,
        "StatLayer created_at must be identical across callers"
    );
    assert_eq!(
        sub_stat.last_modified_at, other_stat.last_modified_at,
        "StatLayer last_modified_at must be identical across callers"
    );
    assert!(
        sub_stat.created_at > 0,
        "StatLayer created_at must be non-zero after create_session_and_layer"
    );
    assert!(
        sub_stat.last_modified_at >= sub_stat.created_at,
        "StatLayer last_modified_at ({}) must be >= created_at ({}) after writes",
        sub_stat.last_modified_at,
        sub_stat.created_at
    );
    // Scenario 3 performed 14 UpdateLayer commits after CreateLayer, so
    // last_modified_at must have been bumped past the creation timestamp.
    assert!(
        sub_stat.last_modified_at > sub_stat.created_at,
        "StatLayer last_modified_at ({}) must exceed created_at ({}) after 14 UpdateLayer writes",
        sub_stat.last_modified_at,
        sub_stat.created_at
    );
}

fn stat_layer_req(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. CreateLayer overwrite during active subscription
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overwrite_during_subscription_rebootstraps_on_new_object() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "ow-session", "ow-layer").await;

    // Subscribe first.
    let sub_req = subscribe_request(&session_id, "ow-layer", 0, 0, "sub-ow");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the Regime-1 initial full-state for old_object_id.
    let (initial, _body) = recv_message(&mut stream).await;
    assert_eq!(initial.object_id, old_object_id);
    assert_eq!(initial.sequence_num, 1);
    // b.rdr: brand-new catch-up trailer for the OLD object.
    recv_caught_up_ack(&mut stream, old_object_id, 1).await;

    // Another client overwrites the layer via CreateLayer. Build a fresh
    // CreateLayerRequest stream carrying an empty delta — same pattern as
    // create_session_and_layer but reusing the existing layer_name to
    // trigger the overwrite path.
    let ompf_bytes = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "ow-layer"),
        make_create_data(ompf_bytes),
    ]);
    let create_req = make_request(create_stream);
    let create_resp = handle_create_layer(&svc, create_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = create_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a new object_id"
    );

    // Subscriber must see a full-state header for the new object — delivered
    // via the `deleted`-sentinel-plus-mapping-re-lookup re-bootstrap path.
    let (rebootstrap, _body) = recv_message(&mut stream).await;
    assert_eq!(
        rebootstrap.object_id, new_object_id,
        "re-bootstrap header must carry the new object_id"
    );
    assert_eq!(rebootstrap.sequence_num, 1, "new object starts at seq=1");
    // b.rdr: the rebootstrap also routes through `catch_up_brand_new(NEW)`, so
    // it terminates with the same total_size==0 caught-up ack on the NEW object.
    recv_caught_up_ack(&mut stream, new_object_id, 1).await;

    // Confirm steady-state continues on the new object: issue an UpdateLayer
    // against new_object_id and expect a per-commit diff.
    apply_unique_update(
        &svc,
        &session_id,
        "ow-layer",
        new_object_id,
        "writer-post-overwrite",
        300,
    )
    .await;

    let (next, next_body) = recv_message(&mut stream).await;
    assert_eq!(next.object_id, new_object_id);
    assert_eq!(next.sequence_num, 2);
    assert!(!next_body.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4b. CreateLayer overwrite during active subscription — SNAPSHOT-mode state
//     parity + strict per-generation catch-up (SR-2.3, SDD §12.4 step 3 / §12.3)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Extension (NOT a rewrite) of
// `overwrite_during_subscription_rebootstraps_on_new_object`, which drives the
// same ordering only in the DEFAULT `catchup_snapshot_enabled = false` mode with EMPTY
// deltas on both generations — it never engages the corrected-snapshot builder
// and never asserts net-state parity or the no-cross-generation-leak invariant.
//
// This test runs in `Snapshot` mode and gives each generation DISTINCTIVE
// content in DISJOINT section-id spaces (old = {1,2}, new = {10,11}). It proves
// that when the shared layer is overwritten mid-session — the overwrite arriving
// via the steady-state `XREAD BLOCK` AFTER the subscriber has already completed
// its initial catch-up, so it exercises the steady-state-phase re-bootstrap site
// (`run_steady_state` → `handle_deleted_and_rebootstrap` → `catch_up_brand_new`
// → `send_full_state` Snapshot arm) — the attached subscriber cleanly
// re-bootstraps onto the NEW `object_id` and receives EXACTLY that generation's
// current state through the corrected catch-up path, with NOTHING from the old
// generation leaking across (catch-up strictly per-generation).
//
// checkpoint_enabled stays false (SR-4.1) — the fixture `Config` sets it and no
// DR path is engaged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_overwrite_rebootstrap_delivers_new_generation_state_no_cross_leak() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-ow-layer";
    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-ow-session", layer).await;

    // ── Establish DISTINCTIVE OLD-generation state via a multi-commit editing
    //    history in the section-id space {1, 2} (disjoint from the new gen). ──
    // Commit 1 (seq=2): root section 1 "old-root" + field "old-marker".
    let old_commit1 = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "old-root".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "old-marker".to_owned(),
            value_or_ext_hash: b"OLD-1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        old_object_id,
        "old-writer",
        100,
        &old_commit1,
    )
    .await;

    // Commit 2 (seq=3): child section 2 "old-branch" under root + field.
    let old_commit2 = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "old-branch".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "old-marker".to_owned(),
            value_or_ext_hash: b"OLD-2".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        old_object_id,
        "old-writer",
        200,
        &old_commit2,
    )
    .await;

    const OLD_ESTABLISHED_SEQ: u64 = 3; // CreateLayer(1) + 2 commits.

    // ── Subscribe a brand-new subscriber (object_id=0, sequence_num=0) and
    //    drain its snapshot catch-up. Reconstructing message #1 from an empty
    //    tree must hold the OLD-generation scene — proving the subscriber
    //    COMPLETED catch-up before the overwrite, so the overwrite lands on the
    //    steady-state-phase re-bootstrap path (not the initial catch-up). ──
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "snap-ow-sub");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (old_header, old_body) = recv_message(&mut stream).await;
    assert_eq!(
        old_header.object_id, old_object_id,
        "initial catch-up message #1 must carry the OLD object_id"
    );
    assert_eq!(
        old_header.sequence_num, OLD_ESTABLISHED_SEQ,
        "snapshot catch-up delivers the OLD generation's current scene at seq={OLD_ESTABLISHED_SEQ}"
    );
    assert!(
        old_header.total_size > 0,
        "OLD-generation catch-up must carry a non-empty full-state body"
    );

    let mut old_reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut old_reconstructed, &old_body);
    let old_ids: std::collections::HashSet<u64> =
        old_reconstructed.sections.keys().copied().collect();
    assert_eq!(
        old_ids,
        [1u64, 2].into_iter().collect(),
        "OLD-generation catch-up must reconstruct exactly sections {{1,2}}"
    );
    // Confirm catch-up COMPLETED: the b.rdr brand-new caught-up ack on the OLD
    // object drains before the overwrite, so what follows is the steady-state
    // (post-catch-up) re-bootstrap, not a still-in-progress initial catch-up.
    recv_caught_up_ack(&mut stream, old_object_id, OLD_ESTABLISHED_SEQ).await;

    // ── From a SEPARATE client, overwrite the layer via CreateLayer reusing the
    //    same (session_id, layer_name) but carrying a NON-EMPTY initial delta
    //    building DISTINCTIVE new-generation state in the DISJOINT section-id
    //    space {10, 11}. ──
    let new_gen_delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 10,
                section_name: "new-root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 10,
                section_id: 11,
                section_name: "new-branch".to_owned(),
                section_type: 0,
                section_order: 7,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 10,
                key_name: "new-marker".to_owned(),
                value_or_ext_hash: b"NEW-10".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 11,
                key_name: "new-marker".to_owned(),
                value_or_ext_hash: b"NEW-11".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
        ],
        ..Default::default()
    };
    let new_gen_ompf = delta_to_ompf_bytes(&new_gen_delta, &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(new_gen_ompf),
    ]);
    let create_req = make_request(create_stream);
    let create_resp = handle_create_layer(&svc, create_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = create_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a new object_id"
    );

    // ── The subscriber must re-bootstrap onto the NEW object via the
    //    steady-state-phase re-bootstrap path: message #1 for the new gen is the
    //    corrected full-state snapshot (SR-1.2). ──
    let (rebootstrap, new_body) = recv_message(&mut stream).await;
    assert_eq!(
        rebootstrap.object_id, new_object_id,
        "re-bootstrap header must carry the new object_id"
    );
    assert_eq!(
        rebootstrap.sequence_num, 1,
        "new generation's from-empty snapshot starts at seq=1"
    );
    assert!(
        rebootstrap.total_size > 0,
        "re-bootstrap must carry a non-empty full-state body"
    );

    let new_decoded = decode_broadcast_delta(&new_body);
    assert!(
        new_decoded.is_diff,
        "re-bootstrap body must be a full-state create (is_diff=true)"
    );
    assert_eq!(
        new_decoded.base_version, 0,
        "re-bootstrap body must be from-empty (base_version=0)"
    );

    // Dense-unique set_order on the emitted body pins that the corrected-snapshot
    // builder produced it (b.bdx). Both new-gen fields are in the one CreateLayer
    // delta (same version), authoring ranks 1 and 2, so they sort by `(version,
    // authoring_set_order)` to dense indices 0 (section 10) and 1 (section 11).
    let new_emitted_set_order = |sid: u64, key: &str| -> u64 {
        new_decoded
            .set_fields
            .iter()
            .find(|sf| sf.section_id == sid && sf.key_name == key)
            .unwrap_or_else(|| panic!("re-bootstrap must emit field ({sid},{key})"))
            .set_order
    };
    assert_eq!(
        new_emitted_set_order(10, "new-marker"),
        0,
        "new-root marker dense set_order (first in (version, authoring) order)"
    );
    assert_eq!(
        new_emitted_set_order(11, "new-marker"),
        1,
        "new-branch marker dense set_order (second in (version, authoring) order)"
    );
    // Global uniqueness across the whole catch-up: no two emitted fields share a
    // set_order (the property the saveCount-keyed Kit client depends on).
    let all_orders: Vec<u64> = new_decoded
        .set_fields
        .iter()
        .map(|sf| sf.set_order)
        .collect();
    let mut uniq = all_orders.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        all_orders.len(),
        "re-bootstrap set_order values must be globally unique"
    );

    // ── Apply the re-bootstrap body to a FRESH empty ObjectState and assert it
    //    reconstructs EXACTLY the new generation's scene — CONTAINS the new-gen
    //    sections/fields {10,11}, and CONTAINS NONE of the old-gen sections/fields
    //    {1,2}. This is the per-generation / no-cross-generation-leak assertion. ──
    let mut new_reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut new_reconstructed, &new_body);

    let new_ids: std::collections::HashSet<u64> =
        new_reconstructed.sections.keys().copied().collect();
    assert_eq!(
        new_ids,
        [10u64, 11].into_iter().collect(),
        "re-bootstrap must reconstruct exactly the NEW generation's sections {{10,11}} — \
         no old-generation section leaked across the object_id boundary"
    );
    // Positive: the new-generation fields are present with their values.
    for (sid, expected) in [(10u64, &b"NEW-10"[..]), (11, &b"NEW-11"[..])] {
        let f = new_reconstructed
            .fields
            .get(&(sid, "new-marker".to_owned()))
            .unwrap_or_else(|| panic!("re-bootstrap must carry field ({sid},new-marker)"));
        assert_eq!(
            f.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},new-marker) must reconstruct its new-generation value"
        );
    }
    // Negative: NO old-generation section survives.
    assert!(
        !new_reconstructed.sections.contains_key(&1)
            && !new_reconstructed.sections.contains_key(&2),
        "old-generation sections {{1,2}} must be ABSENT from the new generation's snapshot"
    );
    // Negative: NO old-generation field survives.
    assert!(
        !new_reconstructed
            .fields
            .contains_key(&(1, "old-marker".to_owned()))
            && !new_reconstructed
                .fields
                .contains_key(&(2, "old-marker".to_owned())),
        "old-generation fields must be ABSENT from the new generation's snapshot (no cross-leak)"
    );

    // ── Exactly-once total_size==0 caught-up terminator re-arming the NEW
    //    generation (SR-1.3), routed through `catch_up_brand_new(NEW)`. ──
    recv_caught_up_ack(&mut stream, new_object_id, 1).await;

    // ── Steady-state relay + self-echo suppression continue on the NEW gen:
    //    a post-overwrite UpdateLayer from a DISTINCT writer client_id is
    //    delivered as a per-commit diff at seq=2 (SR-1.2). ──
    apply_unique_update(
        &svc,
        &session_id,
        layer,
        new_object_id,
        "post-overwrite-writer",
        500,
    )
    .await;

    let (steady, steady_body) = recv_message(&mut stream).await;
    assert_eq!(
        steady.object_id, new_object_id,
        "post-overwrite per-commit diff must be on the new object"
    );
    assert_eq!(
        steady.sequence_num, 2,
        "post-overwrite per-commit diff on the new generation is seq=2"
    );
    assert!(
        !steady_body.is_empty(),
        "post-overwrite per-commit diff must carry a non-empty body"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. DeleteLayer during active subscription terminates with NOT_FOUND
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_layer_during_subscription_terminates_with_not_found() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, _object_id) =
        create_session_and_layer_pooled(&tv, &svc, "del-session", "del-layer").await;

    let sub_req = subscribe_request(&session_id, "del-layer", 0, 0, "sub-del");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the initial full-state header + (possibly empty) body, then the
    // b.rdr brand-new caught-up ack trailer.
    let (init_header, _body) = recv_message(&mut stream).await;
    recv_caught_up_ack(&mut stream, init_header.object_id, init_header.sequence_num).await;

    // Delete the layer from another "client".
    handle_delete_layer(&svc, delete_request(&session_id, "del-layer"))
        .await
        .expect("DeleteLayer must succeed");

    // The subscription must terminate with NOT_FOUND on the response stream —
    // delivered asynchronously per the Epic 3 rewrite.
    let next = tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
        .await
        .expect("stream must yield NOT_FOUND within the subscriber deadline of delete");
    let item = next.expect("stream must yield a terminal status, not close silently");
    let status = item.expect_err("delete must produce an Err terminal status");
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "delete during subscription must terminate with NOT_FOUND, got {}",
        status.code()
    );
    // The code alone does not pin WHICH terminal arm answered. Both
    // `RebootstrapCause`s end this same mapping-absent branch with NOT_FOUND and
    // b.hxk split them by message: only the deleted-sentinel cause is licensed
    // to claim a deletion, and this test drives a real `DeleteLayer`, so it is
    // the arm that must keep saying so. Softening the wording here to the
    // state-loss phrasing (`state was lost and could not be recovered`) would
    // tell an operator that a layer their own DeleteLayer removed had suffered
    // durable state loss — and the code assertion above would stay green
    // through it.
    assert_eq!(
        status.message(),
        "layer 'del-layer' was deleted",
        "the `deleted` sentinel read off the stream plus an absent mapping are two independent \
         records of removal agreeing, which is what licenses the deletion claim (b.hxk, \
         `deleted_not_found` in src/grpc/subscribe.rs). Got: {}",
        status.message()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Self-echo suppression in per-commit path
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn self_echo_suppressed_but_other_clients_delivered() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "echo-session", "echo-layer").await;

    // Subscriber A tracks their own writes.
    let sub_req = subscribe_request(&session_id, "echo-layer", 0, 0, "alice");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Drain the Regime-1 full-state header, then the b.rdr brand-new caught-up
    // ack trailer — both must be consumed before asserting steady-state quiet.
    let (init_header, _body) = recv_message(&mut stream).await;
    recv_caught_up_ack(&mut stream, init_header.object_id, init_header.sequence_num).await;

    // Alice writes — expect this to be suppressed for Alice's subscription.
    apply_unique_update(&svc, &session_id, "echo-layer", object_id, "alice", 400).await;

    // Nothing should arrive within a reasonable window.
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // Bob writes — Alice should receive this.
    apply_unique_update(&svc, &session_id, "echo-layer", object_id, "bob", 401).await;

    let (header, body) = recv_message(&mut stream).await;
    // Alice's own commit bumped seq to 2; Bob's commit is seq=3. Alice should
    // see Bob's per-commit message at seq=3 (the suppressed commit still
    // advanced her tracked_version internally, so the diff covers 2→3).
    assert_eq!(header.sequence_num, 3, "bob's write should arrive at seq=3");
    assert_eq!(header.object_id, object_id);
    assert!(!body.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. M2 smoke — subscribers don't starve concurrent UpdateLayer
// ═══════════════════════════════════════════════════════════════════════════════
//
// Before M2, `SubscribeReadLayer` and `UpdateLayer` shared the same `txn_pool`.
// Each active subscriber held one exclusive client for the lifetime of its
// `XREAD BLOCK` loop, so N concurrent subscribers with N == txn_pool_size could
// starve every incoming UpdateLayer indefinitely. Subscribers now read through
// the stream multiplexer's own reader connections, so UpdateLayer transactions
// are free to proceed however many subscriptions are live.
//
// Scenario: txn_pool=2, subscriber_pool=4. Spawn 3 concurrent subscribers (>
// txn_pool, within subscriber_pool), then concurrently fire 10 UpdateLayers.
// Without M2 the updates would stall forever; with M2 they complete within a
// reasonable timeout and every subscriber receives all 10 per-commit diffs.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribers_do_not_starve_concurrent_update_layer() {
    // txn_pool_size=2 (shared data+txn pool), subscriber_pool_size=4 readers.
    // With 3 live subscribers the txn pool is untouched — UpdateLayer
    // transactions take from their own pool.
    const TXN_POOL: u32 = 2;
    const SUBSCRIBER_POOL: u32 = 4;
    const SUBSCRIBER_COUNT: usize = 3; // > TXN_POOL, <= SUBSCRIBER_POOL
    const UPDATE_COUNT: u64 = 10;

    let tv = PooledTestValkey::setup_with_pools(TXN_POOL, SUBSCRIBER_POOL).await;
    let svc = Arc::new(build_pooled_service(
        &tv,
        Arc::new(AllowAllPermissions),
        4096,
    ));

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "m2-session", "m2-layer").await;

    // Spawn SUBSCRIBER_COUNT subscribers — each one opens a persistent
    // streaming RPC registered with the stream multiplexer for the lifetime
    // of the subscription. Each task collects the per-commit sequence_nums
    // it observes; the main task asserts all subscribers saw all updates.
    let mut subscriber_tasks = Vec::with_capacity(SUBSCRIBER_COUNT);
    for i in 0..SUBSCRIBER_COUNT {
        let svc_clone = svc.clone();
        let session_id_clone = session_id.clone();
        let client_id = format!("sub-{i}");

        let task = tokio::spawn(async move {
            let sub_req = subscribe_request(&session_id_clone, "m2-layer", 0, 0, &client_id);
            let response = handle_subscribe_read_layer(&svc_clone, sub_req)
                .await
                .expect("subscribe must succeed under saturation");
            let mut stream = response.into_inner();

            // Drain the Regime-1 full-state header so subscriber is caught
            // up to seq=1 before writes begin, then the b.rdr caught-up ack.
            let (initial, _body) = recv_message(&mut stream).await;
            assert_eq!(initial.sequence_num, 1);
            recv_caught_up_ack(&mut stream, initial.object_id, 1).await;

            // Collect every per-commit header the subscriber receives.
            let mut observed = Vec::with_capacity(UPDATE_COUNT as usize);
            for _ in 0..UPDATE_COUNT {
                let (header, _body) = recv_message(&mut stream).await;
                observed.push(header.sequence_num);
            }
            observed
        });
        subscriber_tasks.push(task);
    }

    // Give subscribers a moment to all reach the XREAD BLOCK state so they
    // hold their exclusive connections before writes begin — this is the
    // condition that would starve UpdateLayer pre-M2.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Fire UPDATE_COUNT UpdateLayer calls — one at a time but from a writer
    // that is NOT any of the subscribers (so self-echo suppression doesn't
    // drop any message). If subscribers were starving UpdateLayer, these
    // would hang indefinitely and the wall-clock timeout would trip.
    let writer_deadline = Duration::from_secs(30);
    let write_start = std::time::Instant::now();
    tokio::time::timeout(writer_deadline, async {
        for i in 0..UPDATE_COUNT {
            apply_unique_update(
                &svc,
                &session_id,
                "m2-layer",
                object_id,
                "m2-writer",
                500 + i,
            )
            .await;
        }
    })
    .await
    .expect(
        "UPDATE_COUNT updates must complete within the writer deadline \
             — hanging here means subscribers starved the txn pool",
    );

    let write_elapsed = write_start.elapsed();
    // Rough sanity: 10 updates should take well under 10 s on any test host.
    // If we're near the deadline, starvation is contending even if we didn't
    // strictly time out — flag it as a soft regression indicator.
    assert!(
        write_elapsed < Duration::from_secs(15),
        "writes took {write_elapsed:?} — starvation likely re-introduced"
    );

    // Every subscriber must see all UPDATE_COUNT per-commit messages with
    // contiguous seqs 2..=UPDATE_COUNT+1.
    let expected: Vec<u64> = (2..=UPDATE_COUNT + 1).collect();
    for (i, task) in subscriber_tasks.into_iter().enumerate() {
        let observed = tokio::time::timeout(SUBSCRIBER_DEADLINE, task)
            .await
            .unwrap_or_else(|_| {
                panic!("subscriber {i} must finish collecting within {SUBSCRIBER_DEADLINE:?}")
            })
            .unwrap_or_else(|e| panic!("subscriber {i} task panicked: {e}"));
        assert_eq!(
            observed, expected,
            "subscriber {i} must observe all {UPDATE_COUNT} per-commit seqs"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. M3 — per-commit message content is bounded to its declared sequence_num
// ═══════════════════════════════════════════════════════════════════════════════
//
// Before M3, the legacy diff build loaded the current materialized state without an
// upper-version bound. This meant that for each per-commit replay entry, the
// resulting diff carried every section whose `first_version > have_version`
// — which, under a burst of N commits all sitting on the stream before the
// subscriber drained any of them, caused "content bleed-through": the diff
// labelled for seq=K contained sections added at seq=K+1..N as well.
//
// After M3, a per-commit replay's diff is bounded to `upper_version = K`
// (today `compute_per_commit_diff_at_cut`). Sections with `first_version > K`
// are excluded from the seq=K diff
// so each message's content corresponds ONLY to that sequence_num's mutation.
//
// The bug is most reliably triggered by Regime 2 replay: apply N updates
// sequentially (each adding a unique section), then subscribe with
// `have_version = 1` so the subscriber triggers `catch_up_in_window`, which
// computes a per-commit diff for each of the N retained stream entries
// (`replay_commit_diff`: each from its own atomic read of the layer, whose cut
// is the *current* post-N state, built by `compute_per_commit_diff_at_cut`),
// then sends them. Without the bound, every replayed message carries all N
// sections; with it, each replayed message carries only the one section it
// introduced. The bound itself is unchanged since M3: `Some(upper_version)`
// into the same diff planner.

/// Parse an OMPF body from a subscriber message and return the list of
/// (section_id, section_name) tuples carried by its OODF delta.
///
/// Only `DiffChangeSection` commands are surfaced — the bees-ebu update
/// generator uses this variant exclusively (see
/// `incremental_diff_change_section`) so every expected mutation shows up
/// here. Other command types are ignored.
fn extract_diff_change_sections(ompf_bytes: &[u8]) -> Vec<(u64, String)> {
    let container = OmpfContainer::parse(ompf_bytes).expect("parse OMPF");
    if container.delta_bytes.is_empty() {
        return Vec::new();
    }
    let delta = oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF");

    let mut out = Vec::new();
    for cmd in &delta.struct_commands {
        if let StructCommand::DiffChangeSection(cs) = cmd {
            out.push((cs.section_id, cs.section_name.clone()));
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn regime2_replay_bounds_each_per_commit_to_its_own_sequence_num() {
    const UPDATE_COUNT: u64 = 5;

    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "m3-session", "m3-layer").await;

    // Sequential writes — each adds a unique, identifiable section. After
    // this block the server has accumulated all UPDATE_COUNT sections; the
    // stream retains entries 2-0..UPDATE_COUNT+1-0, each labelled with the
    // seq at which it committed.
    for i in 0..UPDATE_COUNT {
        let section_id = 700 + i;
        let client_id = format!("writer-{i}");
        apply_unique_update(
            &svc,
            &session_id,
            "m3-layer",
            object_id,
            &client_id,
            section_id,
        )
        .await;
    }

    // Subscribe with have_version=1 — one past the CreateLayer seq but behind
    // current. This triggers Regime 2 (in-window replay): the server issues
    // XRANGE, gets UPDATE_COUNT entries, and reads and builds each commit's
    // diff in a loop (`replay_commit_diff` → `compute_per_commit_diff_at_cut`)
    // before sending them. Each build takes the M3 bound: without
    // `upper_version = Some(seq)`, the diff built from the post-N atomic read
    // bleeds sections from later seqs into the earlier messages.
    let sub_req = subscribe_request(&session_id, "m3-layer", object_id, 1, "m3-sub");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Collect UPDATE_COUNT per-commit messages (Regime 2 replays each entry
    // as its own per-commit diff). Build seq -> sections map.
    let mut sections_by_seq: std::collections::HashMap<u64, Vec<(u64, String)>> =
        std::collections::HashMap::new();
    for _ in 0..UPDATE_COUNT {
        let (header, body) = recv_message(&mut stream).await;
        let sections = extract_diff_change_sections(&body);
        sections_by_seq.insert(header.sequence_num, sections);
    }

    // Observed seqs must be exactly 2..=UPDATE_COUNT+1, each exactly once.
    let observed_seqs: std::collections::HashSet<u64> = sections_by_seq.keys().copied().collect();
    let expected_seqs: std::collections::HashSet<u64> = (2..=UPDATE_COUNT + 1).collect();
    assert_eq!(
        observed_seqs,
        expected_seqs,
        "expected exactly seqs 2..={} once each, got {observed_seqs:?}",
        UPDATE_COUNT + 1
    );

    // Section i was committed at seq i+2-700 — i.e. section_id 700+i was
    // introduced by the (i+2)th sequence_num. Map seq -> expected section_id.
    //
    // With M3's upper_version bound, each per-commit diff carries ONLY the
    // section introduced at that seq (first_version == seq <= upper_version).
    // Without M3, the earliest replayed seq (=2) would scoop up every section
    // whose first_version > have_version (1), i.e. ALL UPDATE_COUNT sections.
    for (i, section_id) in (0..UPDATE_COUNT).map(|i| (i, 700 + i)) {
        let seq = i + 2;
        let sections = sections_by_seq
            .get(&seq)
            .unwrap_or_else(|| panic!("must have received per-commit for seq={seq}"));
        assert_eq!(
            sections.len(),
            1,
            "M3 bound violated: message at seq={seq} must carry exactly one \
             section (the one introduced at that seq); got {} sections: {sections:?}. \
             Without M3 this test fails — the earlier messages carry a superset \
             of all sections added in seqs > have_version.",
            sections.len()
        );
        let (sid, sname) = &sections[0];
        assert_eq!(
            *sid, section_id,
            "seq={seq} must carry section {section_id}, got {sid}"
        );
        assert_eq!(
            sname,
            &format!("sec-{section_id}"),
            "section_name at seq={seq} must match sec-{section_id}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Natural (unforced) out-of-window fallback — b.776 item 1
// ═══════════════════════════════════════════════════════════════════════════════
//
// The Regime-3 test above (`out_of_window_reconnect_coalesces_and_reaches_state_parity`)
// uses explicit `XTRIM MINID` to force the subscriber's start-id out of the
// retention window. That's deterministic but bypasses the real production path
// — which relies on XADD's `MAXLEN ~ N` approximate trim.
//
// This test exercises the natural path: set a small `layer_stream_maxlen`, then
// issue enough updates that Valkey's approximate trim actually fires, and
// reconnect at an early version. Valkey's approximate trim operates at
// stream-node granularity (roughly 100 entries per node for single-node
// streams), so ≥ ~150 XADDs reliably clips the earliest entries regardless of
// `MAXLEN ~ N` targeting a small N. We explicitly XLEN + XRANGE-verify that
// entry `2-0` is no longer on the stream before subscribing, so the test can't
// mask a regression where trim stops firing.
//
// This test also doubles as the MAXLEN/XLEN assertion case the ticket called
// out — it asserts that after N >> MAXLEN writes, XLEN is bounded well below N
// (confirming approximate trim is engaged) and that the stream does not grow
// unboundedly.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn natural_out_of_window_without_explicit_xtrim_falls_into_regime3() {
    let tv = PooledTestValkey::setup().await;
    // Small MAXLEN. Approximate trim won't clip below ~100 entries for
    // single-node streams, but the key point is that ≥ ~150 XADDs triggers
    // trim at all — the earliest entries get dropped.
    const MAXLEN: u64 = 4;
    // Must exceed Valkey's approximate-trim per-node threshold (~100 for
    // single-node streams) so the initial entries (including 2-0) are
    // actually evicted. 200 writes consistently trims entry 2-0 on
    // redis/valkey 7.x.
    const TOTAL_WRITES: u64 = 200;

    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), MAXLEN);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "natural-oow-session", "natural-oow-layer")
            .await;

    // Walk the server forward far enough that XADD's MAXLEN ~ N approximate
    // trim has had a chance to clip the earliest entries. Use distinct
    // section_ids so each commit produces a non-empty diff (per SDD §9).
    for i in 0..TOTAL_WRITES {
        apply_unique_update(
            &svc,
            &session_id,
            "natural-oow-layer",
            object_id,
            "writer-natural",
            1_000 + i,
        )
        .await;
    }

    let stream_key = tv.key_schema().layer_stream(object_id);

    // Assert 1: approximate MAXLEN trim actually fired. XLEN must be well
    // below TOTAL_WRITES — if it equals TOTAL_WRITES, either MAXLEN trimming
    // stopped working or the test's write volume dropped below the approx
    // threshold. Either way the out-of-window path wouldn't be exercised.
    //
    // Valkey's approximate trim keeps roughly one full macro-node of entries
    // (~100 for single-node streams), so we expect XLEN in the range
    // (MAXLEN, TOTAL_WRITES). We check a generous ceiling below TOTAL_WRITES
    // so future Valkey node-sizing changes don't break the test; the tight
    // assertion below (that entry 2-0 is gone) is what actually proves the
    // out-of-window condition.
    let xlen: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert!(
        xlen < TOTAL_WRITES,
        "approximate MAXLEN trim did not fire: XLEN={xlen} >= TOTAL_WRITES={TOTAL_WRITES}. \
         The natural out-of-window path cannot be exercised if no trim happens. \
         Increase TOTAL_WRITES if Valkey's trim threshold has grown."
    );
    assert!(
        xlen > 0,
        "stream must retain at least one entry (got XLEN=0)"
    );

    // Assert 2: entry `2-0` (the first UpdateLayer commit — CreateLayer uses
    // seq=1 so the first update is seq=2, id 2-0) is no longer on the stream.
    // This is the definitive check that a subscriber with have_version=1 will
    // fall through to Regime 3, independent of exact XLEN bounds.
    // Decode field VALUEs as `Bytes`, not `String`: gRPC-path stream entries
    // carry the binary `ompf` field (b.gze), which is not valid UTF-8 and
    // fails `String` decode. This probe only inspects the entry ID.
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "entry 2-0 must have been trimmed by approximate MAXLEN — got {probe:?}"
    );

    // Reconnect with have_version=1 (one past CreateLayer seq). Without
    // explicit XTRIM: Regime 3 is selected because the first XRANGE entry
    // id will be > 2-0.
    let sub_req = subscribe_request(
        &session_id,
        "natural-oow-layer",
        object_id,
        1,
        "sub-natural",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Regime 3 delivers exactly ONE coalesced diff carrying the current
    // sequence_num. For TOTAL_WRITES updates after CreateLayer's seq=1,
    // the current seq is TOTAL_WRITES + 1.
    let expected_current_seq = TOTAL_WRITES + 1;
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.sequence_num, expected_current_seq,
        "Regime 3 coalesced diff must carry current sequence_num = {expected_current_seq}, got {}",
        header.sequence_num
    );
    assert_eq!(header.object_id, object_id);
    assert!(
        !body.is_empty(),
        "coalesced Regime-3 diff body must be non-empty"
    );

    // Regime 2 would deliver one message per retained entry (i.e. many
    // messages); Regime 3 delivers exactly ONE coalesced diff followed by the
    // exactly-once caught-up ack (`total_size==0`) emitted centrally by
    // `run_catch_up` (b.dy5). For Regime 3 the terminator's `tracked_version`
    // equals the coalesced header's `current_seq` (the whole catch-up was the
    // one coalesced diff). After draining it the consumer opens at $, so a
    // short quiescent window confirms Regime-3 semantics with no extra traffic.
    recv_caught_up_ack(&mut stream, object_id, expected_current_seq).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // Reconstruct state-parity just enough to show the coalesced diff
    // carries substantive content — full parity is already covered by the
    // forced-XTRIM Regime-3 test above, so we keep this light.
    let mut local_state = ObjectState::default();
    apply_delta(&mut local_state, None).expect("seq=1 mirror");
    apply_ompf_to_state(&mut local_state, &body);
    let server_snapshot =
        read_object_state_for_test(&svc.valkey, &svc.key_schema, &object_id.to_string())
            .await
            .expect("read_object_state_for_test");
    let server_state = snapshot_to_object_state(&server_snapshot);
    let local_section_ids: std::collections::HashSet<u64> =
        local_state.sections.keys().copied().collect();
    let server_section_ids: std::collections::HashSet<u64> =
        server_state.sections.keys().copied().collect();
    assert_eq!(
        local_section_ids,
        server_section_ids,
        "Regime-3 coalesced diff must reconstruct every server-side section \
         ({} expected) even under natural approximate-trim",
        server_section_ids.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. have_version > current_seq defensive branch — b.776 item 2
// ═══════════════════════════════════════════════════════════════════════════════
//
// `src/grpc/subscribe.rs` handles the defensive case where a subscriber claims
// a sequence_num ahead of what the server header records. That branch logs a
// warning, acks with the server's current_seq, and tails from $. No production
// flow should hit it (the subscriber can only know seqs the server issued), but
// a misbehaving or offset-corrupted client could — so the handler stays
// forgiving. This test drives that branch end-to-end.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_claiming_version_ahead_of_server_is_acked_and_tails_from_tail() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "ahead-session", "ahead-layer").await;

    // After CreateLayer the server header is at sequence_num=1. Subscribe
    // with a wildly ahead have_version so (have_version > current_seq)
    // holds strictly (not just equal).
    const FUTURE_VERSION: u64 = 99;
    let sub_req = subscribe_request(
        &session_id,
        "ahead-layer",
        object_id,
        FUTURE_VERSION,
        "sub-ahead",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed (the branch is warn-log + ack, not error)");
    let mut stream = response.into_inner();

    // First (and only, for now) message must be the zero-body ack header
    // carrying the SERVER's current_seq (1), NOT the subscriber's claimed
    // FUTURE_VERSION. `send_ack_header` always emits total_size=0 with no
    // following chunks.
    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
        .await
        .expect("stream must yield ack header within the subscriber deadline")
        .expect("stream closed unexpectedly")
        .expect("stream yielded an error");
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header, got: {other:?}"),
    };
    assert_eq!(
        header.sequence_num, 1,
        "ack must carry server's current_seq=1, not the client's claimed FUTURE_VERSION={FUTURE_VERSION}"
    );
    assert_eq!(header.object_id, object_id);
    assert_eq!(
        header.total_size, 0,
        "ack header must have total_size=0 (empty body)"
    );

    // Nothing else should arrive until a new write lands — the consumer was
    // opened at $, so only post-subscription writes get through.
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // Prove the subscription is still live and tailing from $: a new write
    // from a different client must be delivered as a per-commit diff at
    // sequence_num=2. Without the "tail from $" branch semantics, either
    // the subscription would have errored earlier or subsequent writes
    // would not reach this subscriber.
    apply_unique_update(
        &svc,
        &session_id,
        "ahead-layer",
        object_id,
        "writer-post-ack",
        5_000,
    )
    .await;

    let (next, next_body) = recv_message(&mut stream).await;
    assert_eq!(
        next.sequence_num, 2,
        "post-ack write must be delivered at seq=2 (server advanced 1 → 2)"
    );
    assert_eq!(next.object_id, object_id);
    assert!(
        !next_body.is_empty(),
        "post-ack per-commit diff must have a non-empty body"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. XRANGE COUNT boundary — gap > COUNT but still in-window — b.776 item 3
// ═══════════════════════════════════════════════════════════════════════════════
//
// Regime 2's rollback-mode replay (this fixture runs rollback mode) reads the
// in-window entries in pages, capped in all at `layer_stream_maxlen` entries
// (see `in_window_xrange_count` and `read_in_window_replay` in
// `src/grpc/subscribe.rs`). When a subscriber's gap is larger than that cap but
// the stream still retains all entries (approximate `MAXLEN ~ N` trimming lets
// small N coexist with more retained entries under Valkey's per-node
// granularity), the replay under-replays — it reads only the first
// cap's worth of entries. The steady-state consumer, positioned at the last replayed
// entry id, must pick up the remainder via XREAD BLOCK so the subscriber
// eventually reaches state parity.
//
// This test asserts the subscriber observes all M per-commit sequence_nums
// contiguously, even when M > COUNT. Under approximate trim with MAXLEN=5 and
// M=10 XADDs, Valkey does NOT clip any entries (trim only fires around the
// ~100-entry node boundary), so the stream still holds all 10 entries. This
// is the exact gap-> COUNT-but-in-window scenario the ticket calls out.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_window_reconnect_with_gap_larger_than_xrange_count_still_catches_up() {
    let tv = PooledTestValkey::setup().await;
    // XRANGE COUNT = `layer_stream_maxlen` = 5. With M = 10 updates, the
    // subscriber's gap (9 entries: seqs 2..=10 after CreateLayer's seq=1)
    // exceeds COUNT, so the XRANGE slice returns only the first 5 (seqs
    // 2..=6). The remaining 4 (seqs 7..=10) must reach the subscriber via
    // the steady-state consumer.
    const XRANGE_COUNT: u64 = 5;
    // M = XRANGE_COUNT + 5 per the ticket recommendation.
    const M: u64 = XRANGE_COUNT + 5;

    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), XRANGE_COUNT);

    let (session_id, object_id) = create_session_and_layer_pooled(
        &tv,
        &svc,
        "count-boundary-session",
        "count-boundary-layer",
    )
    .await;

    // Apply M updates before subscribing. Each commit produces a unique
    // section so per-commit diffs are non-empty.
    for i in 0..M {
        apply_unique_update(
            &svc,
            &session_id,
            "count-boundary-layer",
            object_id,
            "writer-boundary",
            2_000 + i,
        )
        .await;
    }

    // Precondition: the stream must still retain all entries — otherwise
    // the subscriber would fall into Regime 3 and this test wouldn't
    // exercise the COUNT-boundary. Approximate `MAXLEN ~ 5` only trims at
    // Valkey's per-node boundary (~100 entries), so M+1=11 XADDs
    // (CreateLayer 1-0 plus M=10 UpdateLayer entries) is far below the
    // trim threshold and XLEN must equal M+1.
    let stream_key = tv.key_schema().layer_stream(object_id);
    let xlen: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert_eq!(
        xlen,
        M + 1,
        "approximate MAXLEN ~ {XRANGE_COUNT} must NOT have clipped — M+1={} XADDs \
         (CreateLayer 1-0 + {M} updates) sits well below Valkey's per-node trim \
         threshold (~100). Got XLEN={xlen}. If this assertion fails, the test has \
         drifted out of its designed regime.",
        M + 1
    );
    // Verify entry 2-0 is present and is the second entry — so the
    // subscriber's XRANGE from 2-0 will match and Regime 2 triggers.
    // Decode field VALUEs as `Bytes`, not `String`: gRPC-path stream entries
    // carry the binary `ompf` field (b.gze), which is not valid UTF-8 and
    // fails `String` decode. This probe only inspects the entry ID.
    let probe_2_0: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert_eq!(
        probe_2_0.first().map(|(id, _)| id.as_str()),
        Some("2-0"),
        "entry 2-0 must still be on the stream so Regime 2 is selected"
    );

    // Subscribe with have_version=1 → XRANGE from 2-0 with COUNT=5 returns
    // entries 2-0..6-0. The in-window branch opens the steady-state consumer
    // at entry id 6-0, computes a per-commit diff for each (seqs 2..=6), then
    // sends them. The consumer must then deliver 7-0..(M+1)-0 via XREAD
    // BLOCK.
    let sub_req = subscribe_request(
        &session_id,
        "count-boundary-layer",
        object_id,
        1,
        "sub-boundary",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // GUARDRAIL (b.dy5): the in-window catch-up terminator must carry the
    // last *replayed* endpoint (`tracked_version`), NOT a recomputed
    // `current_seq`. The XRANGE slice under-replays — COUNT=5 returns seqs
    // 2..=6 only — so the catch-up phase ends at the under-replayed endpoint
    // (seq 6), STRICTLY LESS THAN the layer header's current_seq (M+1=11). The
    // exactly-once caught-up ack `run_catch_up` now emits must land BETWEEN the
    // in-window replay and the steady-state remainder, and it must echo seq=6.
    // Acking current_seq (11) here would make the client treat the unreplayed
    // real edits (seqs 7..=11) as already-seen — the precise regression this
    // pins.
    //
    // Phase 1 — Regime-2 in-window replay: exactly XRANGE_COUNT per-commit
    // diffs, contiguous seqs 2..=(XRANGE_COUNT+1) = 2..=6.
    let mut replayed_seqs = Vec::with_capacity(XRANGE_COUNT as usize);
    for _ in 0..XRANGE_COUNT {
        let (header, body) = recv_message(&mut stream).await;
        replayed_seqs.push(header.sequence_num);
        assert_eq!(header.object_id, object_id);
        assert!(
            header.total_size > 0,
            "in-window per-commit diff must carry content (total_size > 0)"
        );
        assert!(
            !body.is_empty(),
            "every in-window per-commit diff must be non-empty"
        );
    }
    let replay_expected: Vec<u64> = (2..=XRANGE_COUNT + 1).collect();
    assert_eq!(
        replayed_seqs, replay_expected,
        "Regime-2 replay must deliver the first XRANGE_COUNT entries (seqs {:?}) \
         before the under-replay terminator",
        replay_expected
    );

    // Phase 2 — the caught-up terminator. Its sequence_num is the last
    // *replayed* endpoint (the XRANGE slice's last entry, seq=6), which is
    // STRICTLY LESS THAN the layer's current_seq (M+1). This inequality is the
    // core of the guardrail: it proves the terminator carries `tracked_version`
    // and not `current_seq`.
    let under_replayed_endpoint = XRANGE_COUNT + 1; // last replayed seq = 6
    let current_seq = M + 1; // layer header's current sequence_num = 11
    assert!(
        under_replayed_endpoint < current_seq,
        "test precondition: under-replayed endpoint ({under_replayed_endpoint}) must be \
         strictly less than current_seq ({current_seq}) for this guardrail to mean anything"
    );
    let ack = recv_caught_up_ack(&mut stream, object_id, under_replayed_endpoint).await;
    assert!(
        ack.sequence_num < current_seq,
        "caught-up terminator must carry the under-replayed endpoint \
         ({under_replayed_endpoint}), strictly less than current_seq ({current_seq}); \
         got {} — acking current_seq would silently skip the unreplayed real edits",
        ack.sequence_num
    );

    // Phase 3 — steady-state remainder: the consumer, positioned at the last
    // replayed entry id, picks up seqs 7..=(M+1) via XREAD BLOCK.
    let mut remainder_seqs = Vec::with_capacity((M - XRANGE_COUNT) as usize);
    for _ in 0..(M - XRANGE_COUNT) {
        let (header, body) = recv_message(&mut stream).await;
        remainder_seqs.push(header.sequence_num);
        assert_eq!(header.object_id, object_id);
        assert!(
            !body.is_empty(),
            "every steady-state per-commit diff must be non-empty"
        );
    }
    let remainder_expected: Vec<u64> = (XRANGE_COUNT + 2..=M + 1).collect();
    assert_eq!(
        remainder_seqs, remainder_expected,
        "steady-state consumer must pick up the remainder (seqs {:?}) AFTER the \
         caught-up terminator, so the subscriber still reaches contiguous parity \
         across the XRANGE boundary",
        remainder_expected
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. b.q2f — overwrite-rebootstrap against a saturated subscriber_pool
// ═══════════════════════════════════════════════════════════════════════════════
//
// An overwrite makes the subscribe handler unregister its consumer and
// register a new one under the same RPC-lifetime admission. Two sites do it:
//
//   * `catch_up_in_window` when an in-window `XRANGE` slice contains a
//     `deleted` sentinel mid-slice.
//   * `run_steady_state` when an already-active subscriber receives
//     `LayerNotification::Deleted` from its consumer.
//
// Both call `consumer.shutdown_and_release()` and then
// `handle_deleted_and_rebootstrap` → `catch_up_brand_new` → `open_consumer`,
// which registers the new generation's stream with the multiplexer — taking
// no second admission, which is what keeps the re-bootstrap from competing at
// admission with new arrivals.
//
// `tests/integration/storage/stream_consumer_test.rs` pins the consumer-level
// halves on their own (re-bootstrap and end-of-subscription churn, the
// mass-drop regression). The existing
// `overwrite_during_subscription_rebootstraps_on_new_object` above covers a
// single subscriber's rebootstrap correctness. What neither covers is N
// concurrent subscribers all rebootstrapping at once.
//
// Scenario:
//   1. N = 3 subscribers already tailing the layer in steady state on a small
//      reader pool (N readers).
//   2. A CreateLayer overwrite fires from a separate client. The mapping
//      flips `old → new`, and `deleted` is XADDed to the old per-object
//      stream. Each subscriber's consumer delivers the sentinel, at which
//      point `run_steady_state` runs `consumer.shutdown_and_release()` and
//      then `handle_deleted_and_rebootstrap` → `catch_up_brand_new` →
//      `open_consumer`.
//   3. Every subscriber must land on the new generation promptly.
//
// Note on the `catch_up_in_window` mid-slice variant: the ticket specifically
// calls out `catch_up_in_window` hitting `deleted` mid-slice as the uncovered
// scenario. Triggering that variant deterministically requires the
// subscriber's `get_layer_mapping` to observe the OLD mapping while the
// subscriber's subsequent `XRANGE` sees the `deleted` sentinel appended by a
// concurrent overwrite — a real-production race but one that cannot be
// reproduced without a fault-injection seam between those two calls (none
// exists in the subscribe handler, and adding one is out-of-scope for this
// test-only ticket). The `catch_up_in_window` and `run_steady_state`
// `Deleted` handlers share the same `shutdown_and_release →
// handle_deleted_and_rebootstrap → catch_up_brand_new → open_consumer`
// chain, so exercising the steady-state variant covers the same chain
// end-to-end.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overwrite_rebootstrap_against_saturated_subscriber_pool_completes_promptly() {
    // Small N: the invariant is qualitative (all N subscribers rebootstrap at
    // once). Going higher adds wall-clock variance and flake risk without
    // strengthening the assertion.
    const SUBSCRIBER_POOL: u32 = 3;
    const SUBSCRIBER_COUNT: usize = SUBSCRIBER_POOL as usize;
    // Unregistering and re-registering is in-memory, so 2 s is orders of
    // magnitude above the happy-path envelope: a bound on the whole
    // end-to-end chain, not a measurement.
    const REBOOTSTRAP_DEADLINE: Duration = Duration::from_secs(2);

    // Roomy txn pool so writes (the overwrite + any post-overwrite update)
    // aren't gated by the subscribe path's pool — this test isolates the
    // subscriber-pool rebootstrap invariant, not txn/sub pool sharing.
    let tv = PooledTestValkey::setup_with_pools(8, SUBSCRIBER_POOL).await;
    let svc = Arc::new(build_pooled_service(
        &tv,
        Arc::new(AllowAllPermissions),
        4096,
    ));

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "q2f-session", "q2f-layer").await;

    // Spawn SUBSCRIBER_COUNT subscribers. Each opens a persistent
    // SubscribeReadLayer RPC, drains its Regime-1 initial full-state header
    // (post-drain it is in steady-state XREAD BLOCK on the OLD object's
    // stream), and then reports back the new object_id + seq it observes
    // once the overwrite-driven rebootstrap lands. All N subscribers share
    // one broadcast channel that fires once the test has confirmed every
    // subscriber has reached steady state — this is the synchronization
    // point that guarantees the overwrite fires against a truly saturated
    // pool.
    let (ready_tx, _) = tokio::sync::broadcast::channel::<()>(SUBSCRIBER_COUNT);
    let mut rebootstrap_tasks = Vec::with_capacity(SUBSCRIBER_COUNT);
    for i in 0..SUBSCRIBER_COUNT {
        let svc_clone = svc.clone();
        let session_id_clone = session_id.clone();
        let client_id = format!("q2f-sub-{i}");
        let mut ready_rx = ready_tx.subscribe();

        let task = tokio::spawn(async move {
            let sub_req = subscribe_request(&session_id_clone, "q2f-layer", 0, 0, &client_id);
            let response = handle_subscribe_read_layer(&svc_clone, sub_req)
                .await
                .expect("initial subscribe must succeed");
            let mut stream = response.into_inner();

            // Drain Regime-1 initial full-state header for the old object,
            // then the b.rdr caught-up ack trailer.
            let (initial, _body) = recv_message(&mut stream).await;
            assert_eq!(initial.sequence_num, 1, "initial header must land at seq=1");
            recv_caught_up_ack(&mut stream, initial.object_id, 1).await;

            // Signal the test we're in steady state; wait for the test to
            // kick off the overwrite.
            let _ = ready_rx.recv().await;

            // The overwrite will XADD `deleted` to the old stream, the
            // steady-state loop will receive `LayerNotification::Deleted`,
            // then `shutdown_and_release` + `handle_deleted_and_rebootstrap` +
            // `catch_up_brand_new` must all complete and the next message we
            // receive is the Regime-1 full-state header for the NEW object,
            // followed by the b.rdr caught-up ack trailer.
            let (rebootstrap, _body) = recv_message(&mut stream).await;
            recv_caught_up_ack(&mut stream, rebootstrap.object_id, rebootstrap.sequence_num).await;
            (rebootstrap.object_id, rebootstrap.sequence_num)
        });
        rebootstrap_tasks.push(task);
    }

    // Give every subscriber time to finish its Regime-1 bootstrap and issue
    // its first XREAD BLOCK. After this point the pool is fully saturated:
    // every slot is held by a subscribe-RPC task parked inside
    // `LayerStreamConsumer`'s blocking XREAD on the old object's stream.
    // `recv` (the `_ = ready_rx.recv().await` above) is what each task
    // awaits before resuming — the sleep bounds the wall-clock window we
    // grant them to park in steady state without coordinating through
    // channel-after-XREAD signals (which would require production-side
    // hooks).
    tokio::time::sleep(Duration::from_millis(300)).await;
    // Release every subscriber at once so the overwrite fires against N
    // concurrently-parked consumers.
    let _ = ready_tx.send(());

    // Separately, fire the CreateLayer overwrite from a different "client".
    // This flips the layer mapping to a new object_id and XADDs `deleted`
    // to the old object's stream, which is what wakes every subscriber's
    // XREAD BLOCK.
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "q2f-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_start = std::time::Instant::now();
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Every subscriber must rebootstrap on the new object within
    // REBOOTSTRAP_DEADLINE measured from the overwrite commit. Failure here
    // means some subscriber's rebootstrap chain stalled under the b.q2f
    // scenario — the incoming consumer never delivered the new generation.
    let mut observed: Vec<(u64, u64)> = Vec::with_capacity(SUBSCRIBER_COUNT);
    for (i, task) in rebootstrap_tasks.into_iter().enumerate() {
        let elapsed_budget = REBOOTSTRAP_DEADLINE.saturating_sub(overwrite_start.elapsed());
        let result = tokio::time::timeout(elapsed_budget.max(Duration::from_millis(1)), task)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "subscriber {i} did not rebootstrap within {REBOOTSTRAP_DEADLINE:?} of \
                     overwrite commit — the rebootstrap chain stalled"
                )
            })
            .unwrap_or_else(|e| panic!("subscriber {i} task panicked: {e}"));
        observed.push(result);
    }

    // Every subscriber must have observed the NEW object_id at seq=1 (the
    // fresh object's create-layer seq). Any subscriber still stuck on the
    // old object_id indicates the `handle_deleted_and_rebootstrap` mapping
    // re-lookup missed the overwrite.
    for (i, (oid, seq)) in observed.iter().enumerate() {
        assert_eq!(
            *oid, new_object_id,
            "subscriber {i} must rebootstrap on the new object_id \
             (got {oid}, expected {new_object_id})"
        );
        assert_eq!(
            *seq, 1,
            "subscriber {i} must see seq=1 on the newly-created object"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 13. Literal `catch_up_in_window` mid-slice `Deleted` handler (b.fdy)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The b.q2f ticket asked for an integration test that exercises the
// `catch_up_in_window` mid-slice `Deleted` handler — the specific scenario
// where an in-window replay's `xrange_values` slice contains the overwrite's
// `deleted` sentinel, the handler calls `consumer.shutdown_and_release()`, and
// re-bootstraps via
// `handle_deleted_and_rebootstrap → catch_up_brand_new → open_consumer`.
//
// b.q2f's test writer could not trigger that path deterministically
// without a production-side fault-injection seam between
// `get_layer_mapping` and `xrange_values`, so it covered the sibling
// steady-state `Deleted` handler instead (see section 12 above). b.fdy
// adds the seam — a 2-party rendezvous `CatchUpXrangeSeam` on
// `LiveEditService` — and this test uses it to interleave an overwrite
// with a parked subscriber at the seam point.
//
// Sequence:
//   1. Create layer at old_object_id; commit two updates so
//      sequence_num advances to 3 (seq=1 create + seq=2/3 updates).
//   2. Install `CatchUpXrangeSeam` on the service.
//   3. Spawn subscriber with `have_object_id=old_object_id`,
//      `have_version=1`. Its `run_catch_up` runs
//      `get_layer_mapping` → OLD, `read_object_header` → current_seq=3,
//      then parks at the seam (before `xrange_values`).
//   4. From the test, arrive at the seam's first rendezvous to confirm
//      the subscriber is parked. Then fire a `CreateLayer` overwrite
//      (new object_id, mapping flips, `XADD deleted` appended to the
//      OLD stream). Then arrive at the seam's second rendezvous to
//      release the subscriber.
//   5. Subscriber's `xrange_values` slice now contains [2-0, 3-0,
//      <deleted>]. The Regime-2 rollback replay parses the slice and scans it
//      for a `LayerNotification::Deleted` sentinel BEFORE it opens a consumer,
//      computes a diff or sends anything. The scan hits the mid-slice
//      sentinel, releases the catch-up slot and runs
//      `handle_deleted_and_rebootstrap → catch_up_brand_new`, so the
//      subscriber's first message is the Regime-1 full-state header for the
//      NEW object, then its caught-up ack — and the OLD object's commits 2
//      and 3 are never sent: the new object's full state supersedes them, and
//      sending them first would put content ahead of the re-bootstrap's
//      capacity waits.
//
// Pre-seam, this test could not be written: there was no way to make
// `get_layer_mapping` observe OLD and `xrange_values` observe `deleted`
// deterministically. The seam closes that gap and lets the test prove
// the literal mid-slice handler — not just its steady-state sibling —
// runs through the full rebootstrap chain correctly.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_in_window_mid_slice_deleted_triggers_rebootstrap() {
    let tv = PooledTestValkey::setup().await;

    // Install a hold gate on the overwrite's spawned `cleanup_old_object`
    // task so the OLD object's data survives until the subscriber has
    // finished. That keeps every OLD commit computable, which is what makes
    // "no OLD message was sent" mean something: were the replay to compute
    // and send before acting on the sentinel, the intact data would let it.
    // Without the gate the cleanup could UNLINK the OLD header first and a
    // compute would answer `NotFound` instead — the Site A path the
    // between-computes test drives. The test releases the gate only after the
    // subscriber has finished.
    let cleanup_gate = Arc::new(tokio::sync::Barrier::new(2));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&cleanup_gate)),
    );
    let mut svc =
        build_pooled_service_from_layer_store(&tv, Arc::new(AllowAllPermissions), layer_store);

    // Install the b.fdy seam BEFORE spawning the subscriber so its
    // `run_catch_up` observes the `Some(seam)` branch on the cfg-gated
    // check. Share the Arc with the test task so both parties can
    // rendezvous.
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "fdy-session", "fdy-layer").await;

    // Commit two updates so the subscriber's in-window slice has real
    // entries at `{have_version+1}-0 = 2-0` and `3-0` BEFORE the `deleted`
    // sentinel appended by the overwrite. Without at least one real entry
    // at `2-0`, `catch_up_in_window`'s first-entry-ID check would miss and
    // the subscriber would fall through to Regime 3 (coalesced diff) — not
    // the path we're exercising.
    apply_unique_update(
        &svc,
        &session_id,
        "fdy-layer",
        old_object_id,
        "fdy-writer",
        42,
    )
    .await;
    apply_unique_update(
        &svc,
        &session_id,
        "fdy-layer",
        old_object_id,
        "fdy-writer",
        43,
    )
    .await;

    // Spawn subscriber with `have_object_id=old` + `have_version=1`. Its
    // `run_catch_up` will resolve the mapping to `old_object_id` (still
    // matches), read the current header (seq=3), then park at the seam.
    let sub_svc = Arc::clone(&svc);
    let sub_session_id = session_id.clone();
    let sub_task = tokio::spawn(async move {
        let sub_req = subscribe_request(
            &sub_session_id,
            "fdy-layer",
            old_object_id,
            1,
            "fdy-subscriber",
        );
        let response = handle_subscribe_read_layer(&sub_svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut stream = response.into_inner();

        // The first message is whatever the catch-up sent first: the
        // Regime-1 full-state header for the NEW object when the sentinel is
        // acted on before any content, an OLD per-commit replay otherwise.
        let (first, _body) = recv_message(&mut stream).await;
        ((first.object_id, first.sequence_num), stream)
    });

    // Rendezvous 1: wait for the subscriber to park at the seam. After
    // `wait_until_parked` returns we KNOW the subscriber has resolved
    // `get_layer_mapping` → OLD and read `read_object_header(OLD)` → seq=3,
    // and is blocked just before `xrange_values`. Firing the overwrite only
    // now guarantees those two reads observed the pre-overwrite state — the
    // subscriber cannot instead take the brand-new / NotFound path and skip
    // the seam (the deadlock the single-barrier design risked).
    seam.wait_until_parked().await;

    // Fire the overwrite while the subscriber is parked at the seam.
    // CreateLayer overwrite:
    //   * allocates a NEW object_id (assigned via counter INCR)
    //   * writes `XADD 1-0 payload update|...` on the NEW stream
    //   * swaps the mapping inside the same MULTI/EXEC (atomic)
    //   * `XADD * payload deleted` on the OLD stream (awaited before the
    //     call returns, so the sentinel is on the OLD stream by the time we
    //     release the subscriber)
    //   * spawns `cleanup_old_object` — which immediately blocks on
    //     `cleanup_gate`, so the OLD object's data is guaranteed intact for
    //     the subscriber's replay until we release the gate below.
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "fdy-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Rendezvous 2: release the subscriber. It issues `xrange_values` on the
    // OLD stream, which sees entries [2-0, 3-0, <deleted>] — the `deleted`
    // sentinel is what the overwrite XADDed above. Because `cleanup_gate`
    // still holds the cleanup task, the OLD object's data is intact, so the
    // only thing that can keep commits 2 and 3 off the wire is the up-front
    // sentinel scan, which re-bootstraps onto the NEW object.
    seam.release().await;

    // The subscriber's first message, bounded by a generous timeout so a
    // stuck rebootstrap shows up as a clear failure (vs the test hanging
    // indefinitely).
    let (first, mut stream) = tokio::time::timeout(SUBSCRIBER_DEADLINE, sub_task)
        .await
        .expect("subscriber must finish within the subscriber deadline")
        .expect("subscriber task panicked");

    // The subscriber is done with the OLD object, so its data is no longer
    // needed. Release the cleanup gate (the test is the barrier's second
    // party) and drain the cleanup task so the OLD keys are torn down
    // deterministically before the test returns.
    let _ = cleanup_gate.wait().await;
    svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;

    // Expected message sequence: the Regime-1 full-state header for the NEW
    // object at seq=1 (produced by `catch_up_brand_new` inside
    // `handle_deleted_and_rebootstrap`), then its caught-up ack — and no
    // OLD-object message at all.
    assert_eq!(
        first,
        (new_object_id, 1),
        "the first message must be the Regime-1 full-state header for the NEW object at seq=1: \
         a `deleted` sentinel in the retained window re-bootstraps before any content. Seeing \
         the OLD object ({old_object_id}) here means its commits were replayed first"
    );
    recv_caught_up_ack(&mut stream, new_object_id, 1).await;

    // Nothing from the OLD object follows either. A commit on the NEW object
    // is the next message, so no OLD replay can sit between the ack and it.
    apply_unique_update(
        &svc,
        &session_id,
        "fdy-layer",
        new_object_id,
        "fdy-writer",
        44,
    )
    .await;
    let (next, _body) = recv_message(&mut stream).await;
    assert_eq!(
        (next.object_id, next.sequence_num),
        (new_object_id, 2),
        "after the NEW object's catch-up, the next message must be its first live commit"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 13b. `catch_up_in_window` after the overwrite's cleanup raced ahead of it (b.jvx)
// ═══════════════════════════════════════════════════════════════════════════════
//
// b.jvx: the failure shape the issue body describes — the OLD object's data
// keys are already UNLINKed when the in-window reconnect reaches them. Section
// 13 holds the cleanup back so the OLD object stays computable; this test
// forces the OPPOSITE ordering — cleanup wins, so the OLD header / section
// indices / etc. are gone by the time the subscriber reads its slice.
// Pre-b.jvx, the per-commit replay's first compute against OLD surfaced that
// as `Status { code: NotFound, message: "object N not found" }` to the client.
//
// The rollback replay now scans the parsed slice for a `deleted` sentinel
// BEFORE it computes any per-commit diff, so here — the slice is
// [2-0, <deleted>] — the scan is what re-bootstraps, and the first compute
// against the cleaned-up OLD object never runs. What this test pins is the
// outcome both orderings owe: the first message is the NEW object's
// full-state header, with no OLD content and no `NotFound` ahead of it. The
// per-commit compute's own `NotFound` classifier (Site A — a slice with no
// sentinel, the OLD state UNLINKed between two computes) is driven by
// `an_in_window_replay_whose_state_is_cleaned_up_between_computes_rebootstraps_before_content`
// in `subscribe_capacity_budget_test.rs`, parked at the between-computes seam.
//
// Ordering trick: between firing the overwrite and releasing the seam,
// call `wait_for_cleanup` on the layer store. `wait_for_cleanup`
// (storage/layer.rs:652) blocks until the spawned `cleanup_old_object`
// task has drained — at which point the OLD object's header has been
// UNLINKed (the per-object stream itself survives 30 s via PEXPIRE, see
// `STREAM_DELETE_GRACE_MS`, so the subscriber's subsequent `xrange_values`
// against OLD still returns the retained entries). Only THEN do we
// release the seam.
//
// `wait_for_cleanup` closes the layer-store's `cleanup_tracker`, so no
// further overwrites can be tracked after this point in the test — we
// only need one overwrite, so this is fine.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_in_window_after_cleanup_raced_ahead_rebootstraps_before_any_content() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    // Install the b.fdy seam so the subscriber parks between
    // `read_object_header(OLD)` and `xrange_values(OLD)` — the window in
    // which the test fires the overwrite + drains cleanup.
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "jvx-a-session", "jvx-a-layer").await;

    // Commit ONE update so the in-window slice starts at the
    // `have_version+1 = 2-0` boundary the subscriber expects. Without at
    // least one real entry at 2-0, `catch_up_in_window`'s first-entry-ID
    // check would miss and Regime 3 (out_of_window) would take over —
    // wrong path. One update is enough: the slice is then [2-0, <deleted>].
    apply_unique_update(
        &svc,
        &session_id,
        "jvx-a-layer",
        old_object_id,
        "jvx-a-writer",
        42,
    )
    .await;

    // Spawn subscriber: have_object_id=OLD, have_version=1. Its
    // `run_catch_up_inner` resolves mapping → OLD, reads header (seq=2),
    // then parks at the seam BEFORE issuing `xrange_values`.
    let sub_svc = Arc::clone(&svc);
    let sub_session_id = session_id.clone();
    let sub_task = tokio::spawn(async move {
        let sub_req = subscribe_request(
            &sub_session_id,
            "jvx-a-layer",
            old_object_id,
            1,
            "jvx-a-subscriber",
        );
        let response = handle_subscribe_read_layer(&sub_svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut stream = response.into_inner();

        // The subscriber's first header is the Regime-1 full-state header
        // for the NEW object delivered by the rebootstrap chain (followed by
        // the b.rdr caught-up ack). A replay that computed against the
        // cleaned-up OLD object and let its `NotFound` through would instead
        // surface `Status::not_found("object N not found")` as a stream error.
        let (header, _body) = recv_message(&mut stream).await;
        let first = (header.object_id, header.sequence_num);

        // b.rdr: the rebootstrap full-state header for the NEW object is
        // followed by the caught-up ack trailer. Drain it before probing
        // for stray messages so the ack is not misread as a leak.
        recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;

        // Actively probe for a further message. If the replay were allowed
        // to send a per-commit diff against OLD before pivoting to
        // rebootstrap, an extra message would land here.
        // `assert_no_next_message` waits the full window and panics on any
        // arrival (header OR stream error), turning the absence-of-leak
        // claim into a real probe instead of a tautology over the loop bound.
        assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

        first
    });

    // Rendezvous 1: wait for the subscriber to park at the seam, having
    // resolved the mapping → OLD and read `read_object_header(OLD)` → seq=2.
    // Firing the overwrite only now guarantees those reads observed the
    // pre-overwrite state (so the subscriber reaches the seam rather than the
    // brand-new / NotFound path), which is what makes the ordering below
    // deterministic.
    seam.wait_until_parked().await;

    // Fire the CreateLayer overwrite while the subscriber is parked at
    // the seam (`wait_until_parked` above guarantees the subscriber has NOT
    // yet issued its `xrange_values` call). The overwrite:
    //   * allocates NEW object_id
    //   * sets the layer mapping → NEW (inside MULTI/EXEC)
    //   * `XADD * payload deleted` on the OLD stream (after MULTI/EXEC)
    //   * spawns `cleanup_old_object` on the layer-store's
    //     `cleanup_tracker` to UNLINK OLD object's data keys
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "jvx-a-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Deterministically drain the spawned `cleanup_old_object` task. By
    // the time this returns, OLD's `obj:hdr` (and sections / fields /
    // time-samples / deleted-sections indices) have been UNLINKed.
    // Critically, the OLD per-object STREAM is NOT UNLINKed — it's
    // PEXPIRE'd with `STREAM_DELETE_GRACE_MS = 30_000` so the subscriber's
    // subsequent `xrange_values(OLD)` still returns [2-0, <deleted>].
    //
    // This is the load-bearing ordering step that distinguishes this
    // test from the section-13 variant: section 13 holds the cleanup back
    // so the OLD object stays computable; here we force cleanup to win, so
    // any per-commit compute against OLD would answer `NotFound`.
    svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;

    // Release the subscriber. Its `xrange_values(OLD)` returns
    // [2-0, <deleted>]. The rollback replay parses the slice and scans it
    // for a `deleted` sentinel before computing anything; the scan hits it,
    // releases the catch-up slot and runs
    // `handle_deleted_and_rebootstrap(ctx, OLD)` → mapping re-lookup returns
    // NEW → `catch_up_brand_new(NEW)` → Regime-1 full-state header for NEW at
    // seq=1. No per-commit compute against the cleaned-up OLD object runs.
    seam.release().await;

    // The subscriber task receives the first header, then runs
    // `assert_no_next_message` for 500 ms to prove no second message
    // (per-commit replay against OLD, stray Status error, etc.) leaks
    // through before returning. The SUBSCRIBER_DEADLINE ceiling covers the
    // first-message wait plus the 500 ms second-message probe plus scheduling
    // slack, with generous headroom under heavy parallel load.
    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, sub_task)
        .await
        .expect(
            "subscriber must finish within the subscriber deadline — a hang here means the \
             re-bootstrap stalled, or a NotFound from the cleaned-up OLD object surfaced as a \
             Status error. Pre-b.jvx, the subscriber's first message would be \
             `Status { code: NotFound, message: \"object N not found\" }` and \
             `recv_message` would panic with `stream yielded an error` rather than this \
             timeout, but either failure shape indicates the fix is gone",
        )
        .expect(
            "subscriber task panicked — either `recv_message` saw a Status error on the first \
             message (a NotFound from the cleaned-up OLD object propagated to the client), or \
             `assert_no_next_message` observed a second message after the rebootstrap header \
             (the replay sent a diff against OLD before pivoting to the re-bootstrap)",
        );

    // The re-bootstrap produces the Regime-1 full-state header for the NEW
    // object at seq=1 (delivered via `handle_deleted_and_rebootstrap →
    // catch_up_brand_new`), followed by the b.rdr total_size==0 caught-up
    // ack on the NEW object. No per-commit replay against OLD is delivered:
    // the sentinel scan re-bootstraps before any compute or send. The
    // "header + ack, nothing else" invariant is enforced inside the spawned
    // task (the explicit `recv_caught_up_ack` drain plus
    // `assert_no_next_message`); here we only need to confirm the full-state
    // header's identity.
    assert_eq!(
        first,
        (new_object_id, 1),
        "subscriber's first message must be the Regime-1 full-state header for the NEW object at \
         seq=1, produced by `catch_up_brand_new` inside `handle_deleted_and_rebootstrap` (b.jvx: \
         an in-window reconnect whose OLD object was cleaned up re-bootstraps rather than \
         propagating `Status::not_found` to the client)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 14. Seam is inert when not installed (release-build parity)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Sanity: with no seam installed, `run_catch_up` runs exactly as release
// builds do — the b.fdy cfg-gated `if let Some(seam) = ...` is a no-op.
// Covers the compile-time contract that the seam doesn't regress the
// default path, separate from the mid-slice Deleted variant above.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_in_window_seam_is_inert_when_not_installed() {
    let tv = PooledTestValkey::setup().await;
    // No seam installed — the default.
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    assert!(
        svc.catch_up_xrange_seam_for_test.is_none(),
        "default service must have no seam installed"
    );

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "fdy-inert-session", "fdy-inert-layer").await;

    apply_unique_update(
        &svc,
        &session_id,
        "fdy-inert-layer",
        object_id,
        "fdy-inert-writer",
        1,
    )
    .await;

    // Regime-2 subscribe with `have_version=1` expects a single per-commit
    // replay for seq=2 — the normal in-window path — with no wait and no
    // rebootstrap. Any hang here would indicate the cfg-gated seam branch
    // somehow fired on a `None` seam.
    let sub_req = subscribe_request(
        &session_id,
        "fdy-inert-layer",
        object_id,
        1,
        "fdy-inert-sub",
    );
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        handle_subscribe_read_layer(&svc, sub_req),
    )
    .await
    .expect("subscribe must not hang when seam is not installed")
    .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, _body) = tokio::time::timeout(Duration::from_secs(2), recv_message(&mut stream))
        .await
        .expect("per-commit replay must not hang when seam is not installed");
    assert_eq!(header.object_id, object_id);
    assert_eq!(header.sequence_num, 2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 15. b.jvx Site B — `run_catch_up_inner` initial header-read NotFound
// ═══════════════════════════════════════════════════════════════════════════════
//
// Mirrors the section-13b Site A test for the b.jvx Site B classifier on
// `run_catch_up_inner`'s same-object `read_object_header(OLD)` call, which runs
// under `read_state_with_recovery`: its `NotFound` takes the recovery hook and,
// failing it, `handle_deleted_and_rebootstrap(...)`. Drives the test through
// the `RunCatchUpInnerSeam` (b.xmk) which parks
// the subscriber between the mapping read at the top of `run_catch_up_inner`
// and the first header read on the same-object_id path, giving the test a
// deterministic window to fire `CreateLayer` overwrite + drain
// `cleanup_old_object`'s UNLINK of OLD's `obj:hdr` before the subscriber's
// header read fires.
//
// This test uses the `PooledTestValkey` fixture flagged by Issue b.pws for
// SYN-burst risk under workspace parallelism; deflate when b.pws lands.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_catch_up_inner_header_notfound_after_cleanup_triggers_rebootstrap() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    // Install the b.xmk Site B seam so the subscriber parks between the
    // mapping read at the top of `run_catch_up_inner` and the
    // `read_object_header(OLD)` call below it — the window in which the
    // test fires the overwrite + drains cleanup.
    let seam = Arc::new(live_edit::grpc::subscribe::RunCatchUpInnerSeam::new());
    svc.run_catch_up_inner_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "jvx-b-session", "jvx-b-layer").await;

    // Spawn subscriber with have_object_id=OLD, have_version=1. This avoids
    // `run_catch_up_inner`'s brand-new short-circuit (which fires only when
    // both fields are zero) AND its overwrite-detection branch (which fires
    // only when `have_object_id != mapping(OLD)`). The subscriber's
    // `run_catch_up_inner` therefore enters the same-object path and parks at
    // the b.xmk seam BEFORE issuing `read_object_header(OLD)`.
    let sub_svc = Arc::clone(&svc);
    let sub_session_id = session_id.clone();
    let sub_task = tokio::spawn(async move {
        let sub_req = subscribe_request(
            &sub_session_id,
            "jvx-b-layer",
            old_object_id,
            1,
            "jvx-b-subscriber",
        );
        let response = handle_subscribe_read_layer(&sub_svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut stream = response.into_inner();

        // With the Site B classifier in place, the subscriber's first header
        // is the Regime-1 full-state header for the NEW object delivered via
        // `handle_deleted_and_rebootstrap → catch_up_brand_new` (followed by
        // the b.rdr caught-up ack). Without the classifier, the first message
        // would instead be a `Status::not_found("object N header not found")`
        // surfaced as a stream error.
        let (header, _body) = recv_message(&mut stream).await;
        let first = (header.object_id, header.sequence_num);

        // b.rdr: drain the brand-new caught-up ack trailer before probing.
        recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;

        // Actively probe for a further message. If the Site B classifier is
        // reverted, the NotFound propagates to the client and
        // `recv_message` panics; if it's only partially reverted (e.g. a
        // helper that loses the structured `ServiceError` variant lands
        // between the call and the classifier), a stray second message
        // could leak through here. `assert_no_next_message` waits the full
        // window and panics on any arrival (header OR stream error),
        // turning the absence-of-leak claim into a real probe instead of a
        // tautology over the loop bound.
        assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

        first
    });

    // Two-phase rendezvous 1: block until the subscriber has parked at the
    // Site B seam. By construction it has already read `mapping == OLD` (the
    // overwrite below fires only after this returns), so the subscriber is
    // committed to the same-object `read_object_header(OLD)` path.
    seam.wait_until_parked().await;

    // Fire the CreateLayer overwrite while the subscriber is parked at the
    // Site B seam (the seam guarantees the subscriber has NOT yet issued
    // its `read_object_header(OLD)` call). The overwrite:
    //   * allocates NEW object_id
    //   * sets the layer mapping → NEW (inside MULTI/EXEC)
    //   * `XADD * payload deleted` on the OLD stream (after MULTI/EXEC)
    //   * spawns `cleanup_old_object` on the layer-store's
    //     `cleanup_tracker` to UNLINK OLD object's data keys (including
    //     `obj:hdr` — the key the parked subscriber will read next).
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "jvx-b-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Deterministically drain the spawned `cleanup_old_object` task. By the
    // time this returns, OLD's `obj:hdr` (and sections / fields /
    // time-samples / deleted-sections indices) have been UNLINKed. The
    // parked subscriber's next `read_object_header(OLD)` will therefore
    // observe `HGETALL → empty hash → ServiceError::NotFound(...)` from
    // `read_object_header`'s single producer in `storage/layer.rs`.
    svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;

    // Release the subscriber. Its `read_object_header(OLD)` returns
    // `ServiceError::NotFound(...)`, and `read_state_with_recovery`'s NotFound
    // arm fires:
    //   1. `try_recover_after_notfound(ctx, OLD)` returns `Ok(false)` because
    //      `build_pooled_service` wires `build_disabled_recovery_hook`
    //      (checkpoint_enabled=false → NotRecoverable short-circuit).
    //   2. `handle_deleted_and_rebootstrap(ctx, OLD)` runs:
    //      mapping re-lookup returns NEW → `catch_up_brand_new(NEW)` →
    //      Regime-1 full-state header for NEW at seq=1.
    seam.release().await;

    // The subscriber task receives the first header, then runs
    // `assert_no_next_message` for 500 ms to prove no second message leaks
    // through before returning. The SUBSCRIBER_DEADLINE ceiling covers the
    // first-message wait plus the 500 ms second-message probe plus scheduling
    // slack, with generous headroom under heavy parallel load.
    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, sub_task)
        .await
        .expect(
            "subscriber must finish within the subscriber deadline — a hang here means the Site B classifier was \
             reverted and the NotFound surfaced as a Status error (or rebootstrap stalled)",
        )
        .expect(
            "subscriber task panicked — either `recv_message` saw a Status error on the first \
             message (Site B classifier reverted, NotFound propagated to the client), or \
             `assert_no_next_message` observed a second message after the rebootstrap header \
             indicating a stray send leaked through (Site B classifier regressed)",
        );

    // The Site B classifier produces EXACTLY ONE message: the Regime-1
    // full-state header for the NEW object at seq=1, delivered via
    // `handle_deleted_and_rebootstrap → catch_up_brand_new`. No
    // header / diff against OLD is delivered (the NotFound on the initial
    // `read_object_header(OLD)` aborts the same-object branch before any
    // send). The single-message invariant is enforced inside the spawned
    // task by `assert_no_next_message`; here we only need to confirm the
    // header's identity.
    assert_eq!(
        first,
        (new_object_id, 1),
        "subscriber's sole message must be the Regime-1 full-state header for the NEW object at \
         seq=1, produced by `catch_up_brand_new` inside `handle_deleted_and_rebootstrap` \
         (b.jvx Site B: NotFound from `read_object_header` against the cleaned-up OLD object \
         routes through the new classifier rather than propagating as `Status::not_found` to \
         the client)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 16. b.jvx Site C arm 1 — `catch_up_out_of_window` NotFound at its atomic read
// ═══════════════════════════════════════════════════════════════════════════════
//
// Mirrors the section-13b Site A test for the b.jvx Site C classifier on
// `catch_up_out_of_window`'s state read — today the catch-up's proven read
// (`catch_up_proven_read`: a read permit, then the atomic read) under
// `read_state_with_recovery`, whose `NotFound` takes the recovery hook and,
// failing it, drops the reader reservation and runs
// `handle_deleted_and_rebootstrap(...)`. Drives the test through
// `CatchUpOutOfWindowSeam::new_before_header_read` (b.xmk), which parks the
// subscriber in `catch_up_out_of_window` after the caller's reader
// reservation (in this rollback-mode fixture, taken under the slot) and before
// the read permit and the atomic read. Nothing is registered and nothing has
// been read from OLD's state when it parks.
//
// Out-of-window regime is forced by the MAXLEN=4 + 5 update + `have_version=1`
// pattern: by the time the subscriber subscribes, the stream's
// `start_id = have_version+1 = 2-0` entry has been evicted by `XADD MAXLEN ~`
// trimming, so `xrange_values` mis-aligns and `run_catch_up_inner` falls
// through to `catch_up_out_of_window` (this mirrors the existing
// out-of-window test `out_of_window_reconnect_coalesces_and_reaches_state_parity`).
//
// This test uses the `PooledTestValkey` fixture flagged by Issue b.pws for
// SYN-burst risk under workspace parallelism; deflate when b.pws lands.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_out_of_window_header_notfound_after_cleanup_triggers_rebootstrap() {
    let tv = PooledTestValkey::setup().await;
    // MAXLEN=4 with ≥5 updates forces eviction of the `2-0` entry the
    // subscriber would land on with `have_version=1`, so `run_catch_up_inner`
    // takes the out-of-window branch into `catch_up_out_of_window` rather
    // than the in-window branch.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);

    // Install the b.xmk Site C arm-1 seam so the subscriber parks in
    // `catch_up_out_of_window` after its reader reservation and before its
    // read permit and atomic read of OLD.
    let seam =
        Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_header_read());
    svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "jvx-c1-session", "jvx-c1-layer").await;

    // 5 updates → server sequence climbs to 6 (1 from CreateLayer + 5 here).
    // MAXLEN=4 + `XADD MAXLEN ~` aggressively trims the early entries,
    // forcing `2-0` (the entry `have_version=1` would land on) out of
    // window. Force MINID deterministically below so we don't depend on
    // `~` approximate trimming for the test's correctness.
    for i in 0..5u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "jvx-c1-layer",
            old_object_id,
            "jvx-c1-writer",
            100 + i,
        )
        .await;
    }

    // Deterministically force `2-0` out of the window so the subscriber's
    // `xrange_values(start=2-0)` in `run_catch_up_inner` mis-aligns and
    // `catch_up_out_of_window` takes over. Mirrors the same XTRIM trick the
    // existing out-of-window test
    // (`out_of_window_reconnect_coalesces_and_reaches_state_parity`) uses.
    let stream_key = tv.key_schema().layer_stream(old_object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "6-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    // Spawn subscriber with have_object_id=OLD, have_version=1. The
    // mapping → OLD path matches, the in-window slice mis-aligns (start_id
    // 2-0 evicted), and `run_catch_up_inner` falls through to
    // `catch_up_out_of_window`. It then reserves its stream reader and parks
    // at the Site C arm-1 seam BEFORE its atomic read of OLD.
    let sub_svc = Arc::clone(&svc);
    let sub_session_id = session_id.clone();
    let sub_task = tokio::spawn(async move {
        let sub_req = subscribe_request(
            &sub_session_id,
            "jvx-c1-layer",
            old_object_id,
            1,
            "jvx-c1-subscriber",
        );
        let response = handle_subscribe_read_layer(&sub_svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut stream = response.into_inner();

        let (header, _body) = recv_message(&mut stream).await;
        let first = (header.object_id, header.sequence_num);

        // b.rdr: drain the brand-new caught-up ack trailer, then probe for a
        // further message — the Site C classifier (`read_state_with_recovery`'s
        // NotFound arm, seen through the first park point) produces the Regime-1
        // header for NEW + its ack and nothing else; any extra is a regression.
        recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;
        assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

        first
    });

    // Wait until the subscriber has actually parked at the Site C arm-1 seam
    // before mutating the layer mapping (b.y3a). Without this the overwrite
    // below races the subscriber's `get_layer_mapping` read in
    // `run_catch_up_inner`: when the overwrite wins, the subscriber reads
    // `mapping == NEW`, takes the overwrite-detect branch into
    // `catch_up_brand_new(NEW)`, never enters `catch_up_out_of_window`, and the
    // test's lone barrier arrival below deadlocks until the 120 s hard-kill.
    // The measured margin was ±1 ms, so the test passed or hung on scheduling
    // noise alone. Same fix b.msb applied to the Site B seam.
    seam.wait_until_parked_pre_header_read().await;

    // Fire the CreateLayer overwrite while the subscriber is parked at the
    // Site C arm-1 seam. Same overwrite shape as the Site B test: the
    // post-MULTI/EXEC `cleanup_old_object` UNLINKs OLD's `obj:hdr`.
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "jvx-c1-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Drain `cleanup_old_object` so OLD's `obj:hdr` is UNLINKed before the
    // subscriber's atomic read of OLD runs.
    svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;

    // Release the subscriber. Its atomic read of OLD inside
    // `catch_up_out_of_window` finds the header absent and answers
    // `NotFound`, and `read_state_with_recovery`'s NotFound arm fires:
    //   1. `try_recover_after_notfound(ctx, OLD) → Ok(false)` (disabled
    //      recovery hook).
    //   2. the reader reservation is dropped; `handle_deleted_and_rebootstrap(ctx, OLD)` →
    //      mapping re-lookup → NEW → `catch_up_brand_new(NEW)` → Regime-1
    //      header at seq=1.
    seam.arrive_at_pre_header_read().await;

    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, sub_task)
        .await
        .expect(
            "subscriber must finish within the subscriber deadline — a hang here means the Site C \
             classifier (`read_state_with_recovery`'s NotFound arm, seen through the first park \
             point) was reverted and the NotFound surfaced as a Status error (or rebootstrap \
             stalled)",
        )
        .expect(
            "subscriber task panicked — either `recv_message` saw a Status error on the first \
             message (the Site C classifier — `read_state_with_recovery`'s NotFound arm, seen \
             through the first park point — reverted, NotFound propagated to the client), or \
             `assert_no_next_message` observed a second message after the rebootstrap header \
             indicating a stray send leaked through (that classifier regressed)",
        );

    assert_eq!(
        first,
        (new_object_id, 1),
        "subscriber's sole message must be the Regime-1 full-state header for the NEW object at \
         seq=1, produced by `catch_up_brand_new` inside `handle_deleted_and_rebootstrap` \
         (b.jvx Site C arm 1: NotFound from the out-of-window atomic read against the \
         cleaned-up OLD object routes through the classifier rather than propagating as \
         `Status::not_found` to the client)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 17. b.jvx Site C arm 2 — `catch_up_out_of_window` NotFound, second park point
// ═══════════════════════════════════════════════════════════════════════════════
//
// Drives the b.jvx Site C classifier through the second of
// `catch_up_out_of_window`'s two park points,
// `CatchUpOutOfWindowSeam::new_before_compute_diff` (b.xmk). The names date
// from when the regime read the OLD header and then ran the since-deleted
// loader-and-build `compute_diff_at_cut` over OLD as
// two reads, and arm 2 parked between them with the header read already
// succeeded. That is no longer so: today the regime's whole state read is one
// proven read (`catch_up_proven_read`: a read permit, then the atomic read,
// under `read_state_with_recovery`), and BOTH park points sit back to back
// after the reader reservation and before the read permit and the atomic read,
// with no read between them. On this fixture this test therefore exercises
// the SAME interleaving as section 16's arm-1 test — overwrite + cleanup
// before the regime's only read of OLD — and differs from it only in which
// park point it holds. It is kept as the guard on the second park point.
//
// The test fires overwrite + drains cleanup, which UNLINKs OLD's `obj:hdr` AND
// the section / field / time-sample / deleted indices. When the subscriber
// wakes, its atomic read of OLD finds the header absent →
// `ServiceError::NotFound(...)` → `read_state_with_recovery`'s NotFound arm
// fires.
//
// This test uses the `PooledTestValkey` fixture flagged by Issue b.pws for
// SYN-burst risk under workspace parallelism; deflate when b.pws lands.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_out_of_window_compute_diff_notfound_after_cleanup_triggers_rebootstrap() {
    let tv = PooledTestValkey::setup().await;
    // Same MAXLEN=4 + 5-update + `have_version=1` shape as the arm-1 test:
    // forces `catch_up_out_of_window` to be the regime under test.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);

    // Install the b.xmk Site C arm-2 seam so the subscriber parks at the
    // second park point in `catch_up_out_of_window` — right after the first,
    // still before its read permit and its atomic read of OLD.
    let seam =
        Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff());
    svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);

    let (session_id, old_object_id) =
        create_session_and_layer_pooled(&tv, &svc, "jvx-c2-session", "jvx-c2-layer").await;

    for i in 0..5u64 {
        apply_unique_update(
            &svc,
            &session_id,
            "jvx-c2-layer",
            old_object_id,
            "jvx-c2-writer",
            100 + i,
        )
        .await;
    }

    // Same deterministic XTRIM as the arm-1 test — force `2-0` out of the
    // window so `catch_up_out_of_window` is the branch taken.
    let stream_key = tv.key_schema().layer_stream(old_object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "6-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    // Spawn subscriber with have_object_id=OLD, have_version=1. By the time
    // the arm-2 seam fires, the subscriber has already executed:
    //   1. its stream-reader reservation (under the slot, in rollback mode)
    //   2. arm-1 seam — auto-completes via a 1-party `Barrier::new(1)`
    //      (b.m52 hardening). `new_before_compute_diff` populates
    //      `park_before_header_read` with a single-party barrier so the
    //      production-side `arrive_at_pre_header_read` arrival completes
    //      immediately on its single arrival; the test never calls into
    //      this barrier. Pre-b.m52 this slot was `None` and the arrival
    //      method silently no-op'd; b.m52 added a `debug_assert!` that
    //      rejects `None` here, so the auto-complete barrier is the
    //      mechanism that preserves the upstream-no-op semantics this
    //      arm-2 test depends on.
    //   3. park at the arm-2 seam (this seam point's `Barrier::wait()`),
    //      having read nothing of OLD's state: no read runs between the two
    //      park points.
    let sub_svc = Arc::clone(&svc);
    let sub_session_id = session_id.clone();
    let sub_task = tokio::spawn(async move {
        let sub_req = subscribe_request(
            &sub_session_id,
            "jvx-c2-layer",
            old_object_id,
            1,
            "jvx-c2-subscriber",
        );
        let response = handle_subscribe_read_layer(&sub_svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut stream = response.into_inner();

        let (header, _body) = recv_message(&mut stream).await;
        let first = (header.object_id, header.sequence_num);

        // b.rdr: drain the brand-new caught-up ack trailer, then probe.
        recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;
        assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

        first
    });

    // Wait until the subscriber has actually parked at the Site C arm-2 seam
    // before mutating the layer mapping (b.y3a) — see the arm-1 test for the
    // full rationale.
    seam.wait_until_parked_pre_compute_diff().await;

    // Fire the CreateLayer overwrite while the subscriber is parked at the
    // Site C arm-2 seam, before its only read of OLD's state.
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "jvx-c2-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Drain `cleanup_old_object`: UNLINKs OLD's `obj:hdr` AND the
    // sections / fields / time-samples / deleted-sections indices, so the
    // subscriber's atomic read of OLD will find the header absent and answer
    // `ServiceError::NotFound(...)`.
    svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;

    // Release the subscriber. Its atomic read of OLD returns NotFound, and
    // `read_state_with_recovery`'s NotFound arm fires:
    //   1. `try_recover_after_notfound → Ok(false)`.
    //   2. the reader reservation is dropped; `handle_deleted_and_rebootstrap(ctx, OLD)` →
    //      mapping → NEW → `catch_up_brand_new(NEW)` → Regime-1 header.
    seam.arrive_at_pre_compute_diff().await;

    let first = tokio::time::timeout(SUBSCRIBER_DEADLINE, sub_task)
        .await
        .expect(
            "subscriber must finish within the subscriber deadline — a hang here means the Site C \
             classifier (`read_state_with_recovery`'s NotFound arm, seen through the second park \
             point) was reverted and the NotFound surfaced as a Status error (or rebootstrap \
             stalled)",
        )
        .expect(
            "subscriber task panicked — either `recv_message` saw a Status error on the first \
             message (the Site C classifier — `read_state_with_recovery`'s NotFound arm, seen \
             through the second park point — reverted, NotFound propagated to the client), or \
             `assert_no_next_message` observed a second message after the rebootstrap header \
             indicating a stray send leaked through (that classifier regressed)",
        );

    assert_eq!(
        first,
        (new_object_id, 1),
        "subscriber's sole message must be the Regime-1 full-state header for the NEW object at \
         seq=1, produced by `catch_up_brand_new` inside `handle_deleted_and_rebootstrap` \
         (b.jvx Site C, second park point: NotFound from the out-of-window atomic read \
         against the cleaned-up OLD object routes through the classifier rather than \
         propagating as `Status::not_found` to the client)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 18. b.rdr — brand-new catch-up emits the total_size==0 caught-up ack trailer
// ═══════════════════════════════════════════════════════════════════════════════
//
// Regression guard for b.rdr: every brand-new catch-up — both callers of
// `send_full_state` — must terminate the full-state delivery with a
// `total_size == 0` caught-up ack header carrying the server's delivered
// `sequence_num`. This is the `OkLatest` marker the Kit client
// gates stage-apply on; pre-fix, a subscriber to a *populated* layer received
// the content chunks and then dropped straight into the steady-state loop with
// no terminator, so the client never left initial-sync and silently stashed
// every subsequent live update.
//
// The two `send_full_state` callers are exercised with one parallel assertion
// each so a future change that re-breaks either path is caught:
//
//   * Caller 1 — genuine brand-new subscribe (`object_id=0, sequence_num=0`)
//     against a layer seeded with real section content (so the full-state
//     header carries `total_size > 0`, distinguishing this from the
//     empty-layer arm that always acked).
//   * Caller 2 — overwrite-on-reconnect rebootstrap, which routes through
//     `handle_deleted_and_rebootstrap → catch_up_brand_new(NEW)`.
//
// Pre-fix this test fails: the brand-new subscriber receives the content
// header + chunks and then never sees a total_size==0 header — `recv_message`
// for the ack would time out (the next message that ever arrives is the first
// post-subscription per-commit, with total_size > 0). Post-fix the ack is the
// message immediately following the full-state content.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_catch_up_emits_caught_up_ack_after_content() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "rdr-session", "rdr-layer").await;

    // Seed the layer with real section content so the brand-new full-state
    // delivery carries non-empty OMPF bytes (total_size > 0) — the populated
    // arm that pre-fix dropped the caught-up ack. After three updates the
    // server header sits at seq=4 (CreateLayer seq=1 + 3 updates).
    const SEED_UPDATES: u64 = 3;
    for i in 0..SEED_UPDATES {
        apply_unique_update(
            &svc,
            &session_id,
            "rdr-layer",
            object_id,
            "rdr-seed-writer",
            900 + i,
        )
        .await;
    }
    let expected_seq = 1 + SEED_UPDATES; // = 4

    // ── Caller 1: genuine brand-new subscribe (object_id=0, sequence_num=0) ──
    let sub_req = subscribe_request(&session_id, "rdr-layer", 0, 0, "rdr-brand-new-sub");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("brand-new subscribe must succeed");
    let mut stream = response.into_inner();

    // Header #1: full-state content for the seeded layer, total_size > 0,
    // carrying the current server sequence_num.
    let (content_header, content_body) = recv_message(&mut stream).await;
    assert_eq!(
        content_header.object_id, object_id,
        "full-state content header must carry the layer's object_id"
    );
    assert_eq!(
        content_header.sequence_num, expected_seq,
        "full-state content header must carry the current server seq ({expected_seq})"
    );
    assert!(
        content_header.total_size > 0,
        "seeded layer's brand-new full-state header must have total_size > 0 \
         (this is the populated arm that pre-fix dropped the caught-up ack), got {}",
        content_header.total_size
    );
    assert!(
        !content_body.is_empty(),
        "full-state content body must be non-empty for a seeded layer"
    );

    // The b.rdr terminator: a total_size==0 caught-up ack carrying the SAME
    // delivered version as the content header. This is the assertion that
    // fails against the pre-fix server (the populated arm never emitted it).
    let ack = recv_caught_up_ack(&mut stream, object_id, expected_seq).await;
    assert_eq!(
        ack.sequence_num, expected_seq,
        "caught-up ack must echo the delivered content version, not 0 or a stale value"
    );

    // ── Caller 2: overwrite-on-reconnect rebootstrap (catch_up_brand_new(NEW)) ──
    //
    // An already-subscribed client whose layer is overwritten re-bootstraps on
    // the new object via `handle_deleted_and_rebootstrap → catch_up_brand_new`,
    // the SECOND caller of `send_full_state`. Drive it on the same live stream:
    // a CreateLayer overwrite flips the mapping and XADDs `deleted` to the old
    // stream, the steady-state loop rebootstraps, and the NEW object's
    // full-state must again be followed by the caught-up ack.
    let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let overwrite_stream = streaming_create(vec![
        make_create_header(&session_id, "rdr-layer"),
        make_create_data(overwrite_ompf),
    ]);
    let overwrite_req = make_request(overwrite_stream);
    let overwrite_resp = handle_create_layer(&svc, overwrite_req)
        .await
        .expect("CreateLayer overwrite must succeed");
    let new_object_id = overwrite_resp.into_inner().object_id;
    assert_ne!(
        new_object_id, object_id,
        "overwrite must allocate a fresh object_id"
    );

    // Rebootstrap full-state header for the NEW object at seq=1.
    let (rebootstrap_header, _rebootstrap_body) = recv_message(&mut stream).await;
    assert_eq!(
        rebootstrap_header.object_id, new_object_id,
        "rebootstrap header must carry the new object_id"
    );
    assert_eq!(
        rebootstrap_header.sequence_num, 1,
        "rebootstrap full-state header must land at seq=1 on the new object"
    );

    // The same b.rdr terminator on the rebootstrap path — guards the SECOND
    // `send_full_state` caller.
    recv_caught_up_ack(&mut stream, new_object_id, 1).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Steady-state verbatim relay (b.gze)
//
// The steady-state path must forward the writer's RAW inbound OMPF byte-for-byte
// rather than reconstructing a diff from materialized state. Reconstruction
// destroys the writer's per-delta `set_order` (folded into a cumulative cursor
// at apply time) and coalesces burst fields to last-writer-wins — neither of
// which the Kit client can apply. Catch-up paths keep reconstructing.
//
// Distinguishing relay from reconstruction:
//   - relay → body decodes to the writer's exact command vocabulary
//     (`CreateSection`, plain `SetField`, `is_diff=false`, the writer's
//     `set_order`).
//   - reconstruction → body decodes to `DiffChangeSection` (the §10 per-commit
//     diff shape) with a rebased/fabricated `set_order`.
// ═══════════════════════════════════════════════════════════════════════════════

/// Issue a single UpdateLayer carrying an arbitrary `delta` from `client_id`,
/// returning the exact OMPF bytes the writer sent (so a steady-state relay test
/// can assert the broadcast body matches them byte-for-byte). Goes through the
/// gRPC `handle_update_layer` path so the writer's OMPF is persisted on the
/// per-object stream's `ompf` field exactly as production does.
async fn apply_delta_update_returning_ompf(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    client_ts: u64,
    delta: &OwnedDelta,
) -> Vec<u8> {
    apply_delta_update_returning_ompf_with_ext(
        svc,
        session_id,
        layer_name,
        object_id,
        client_id,
        client_ts,
        delta,
        &[],
    )
    .await
}

/// As [`apply_delta_update_returning_ompf`], but carries `external_values` as
/// OMPF Parts[1..N] alongside the OODF delta (Part 0). Used by the multi-part
/// relay test to prove inline external-value parts ride inside the relayed
/// container byte-for-byte with no separate ext-value fetch.
#[allow(clippy::too_many_arguments)]
async fn apply_delta_update_returning_ompf_with_ext(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    client_ts: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
) -> Vec<u8> {
    let ompf_bytes = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        update_header(session_id, layer_name, object_id, client_id, client_ts),
        update_data(ompf_bytes.clone()),
    ]);
    let request = make_request(stream);
    handle_update_layer(svc, request)
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer ({client_id}) failed: {e}"));
    ompf_bytes
}

/// Decode a subscriber broadcast body (OMPF bytes) into its `OwnedDelta`.
/// Panics on an empty delta — every relay/per-commit case under test carries a
/// non-empty delta, so an empty body means a coalesced-to-empty broadcast (the
/// b.gze burst symptom) and should fail the test loudly.
fn decode_broadcast_delta(ompf_bytes: &[u8]) -> OwnedDelta {
    let container = OmpfContainer::parse(ompf_bytes).expect("parse broadcast OMPF");
    assert!(
        !container.delta_bytes.is_empty(),
        "broadcast body carried an EMPTY OODF delta — burst-coalesce-to-empty regression (b.gze)"
    );
    oodf_ffi::parse_delta(container.delta_bytes).expect("parse broadcast OODF delta")
}

/// Subscribe a brand-new steady-state subscriber: drain the Regime-1 full-state
/// header + b.rdr caught-up ack so the stream is positioned at the live tail,
/// then return it for the caller to read per-commit broadcasts from.
async fn subscribe_steady_state<'a>(
    svc: &'a LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    create_seq: u64,
    subscriber_client_id: &str,
) -> impl Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin + 'a {
    let sub_req = subscribe_request(session_id, layer_name, 0, 0, subscriber_client_id);
    let response = handle_subscribe_read_layer(svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();
    let (initial, _body) = recv_message(&mut stream).await;
    assert_eq!(initial.object_id, object_id);
    assert_eq!(initial.sequence_num, create_seq);
    recv_caught_up_ack(&mut stream, object_id, create_seq).await;
    stream
}

// (a) New-prim relay: a steady-state subscriber receives the writer's verbatim
//     OMPF (CreateSection + SetField, is_diff=false, set_order preserved) — NOT
//     a reconstructed is_diff=true / DiffChangeSection diff.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_relays_new_prim_verbatim() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "relay-newprim-session", "relay-newprim-layer")
            .await;

    let mut stream = subscribe_steady_state(
        &svc,
        &session_id,
        "relay-newprim-layer",
        object_id,
        1,
        "sub-relay",
    )
    .await;

    // Writer creates a prim (CreateSection) and authors one attribute on it
    // (SetField) with a DISTINCTIVE non-zero set_order. Reconstruction cannot
    // reproduce either the CreateSection command vocabulary or this set_order.
    const DISTINCTIVE_SET_ORDER: u64 = 4242;
    let writer_delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 77,
            section_name: "MySphere".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: 77,
            key_name: "radius".to_owned(),
            value_or_ext_hash: vec![1, 2, 3, 4],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: DISTINCTIVE_SET_ORDER,
        }],
        ..Default::default()
    };

    let writer_ompf = apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-newprim-layer",
        object_id,
        "writer-other",
        100,
        &writer_delta,
    )
    .await;

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.sequence_num, 2, "prim-creation commit is seq=2");
    assert_eq!(header.object_id, object_id);

    // Strongest assertion: the broadcast body is the writer's OMPF byte-for-byte.
    assert_eq!(
        body, writer_ompf,
        "steady-state broadcast must be the writer's RAW OMPF verbatim (b.gze relay), \
         not a reconstructed diff"
    );

    // Decode and assert the relay-faithful shape the client applies.
    let decoded = decode_broadcast_delta(&body);
    assert!(
        !decoded.is_diff,
        "relayed delta must be is_diff=false (writer's shape); a reconstructed per-commit \
         diff would be is_diff=true"
    );
    assert!(
        matches!(
            decoded.struct_commands.as_slice(),
            [StructCommand::CreateSection(cs)] if cs.section_id == 77 && cs.section_name == "MySphere"
        ),
        "relayed delta must carry the writer's CreateSection (not a reconstructed \
         DiffChangeSection); got {:?}",
        decoded.struct_commands
    );
    assert_eq!(decoded.set_fields.len(), 1, "the SetField must ride along");
    assert_eq!(
        decoded.set_fields[0].set_order, DISTINCTIVE_SET_ORDER,
        "the writer's per-delta set_order must be preserved verbatim — reconstruction \
         from materialized state fabricates one (b.gze root cause)"
    );
}

// (b) Field-only move relay: a field-only update on an EXISTING section relays
//     the writer's bytes (plain SetField, no struct command) on steady-state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_relays_field_only_move_verbatim() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "relay-field-session", "relay-field-layer")
            .await;

    // Establish the prim first (seq=2) — NOT what we assert on; just makes the
    // field-only update target an existing section.
    let create_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 88,
            section_name: "Cube".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-field-layer",
        object_id,
        "writer-setup",
        50,
        &create_delta,
    )
    .await;

    // Subscribe AFTER the create so the field-only update is the first thing the
    // steady-state subscriber observes (seq=3).
    let mut stream = {
        let sub_req = subscribe_request(&session_id, "relay-field-layer", 0, 0, "sub-field");
        let response = handle_subscribe_read_layer(&svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut s = response.into_inner();
        // Brand-new catch-up delivers full state at the current seq (=2) then acks.
        let (initial, _body) = recv_message(&mut s).await;
        assert_eq!(initial.object_id, object_id);
        assert_eq!(initial.sequence_num, 2);
        recv_caught_up_ack(&mut s, object_id, 2).await;
        s
    };

    // Field-only "move": a plain SetField on the existing section, no struct cmd.
    const MOVE_SET_ORDER: u64 = 9001;
    let move_delta = OwnedDelta {
        is_diff: false,
        set_fields: vec![SetField {
            section_id: 88,
            key_name: "xformOp:translate".to_owned(),
            value_or_ext_hash: vec![9, 9, 9],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: MOVE_SET_ORDER,
        }],
        ..Default::default()
    };
    let writer_ompf = apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-field-layer",
        object_id,
        "writer-mover",
        100,
        &move_delta,
    )
    .await;

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.sequence_num, 3, "field-only commit is seq=3");
    assert_eq!(
        body, writer_ompf,
        "field-only steady-state broadcast must relay the writer's RAW OMPF verbatim (b.gze)"
    );
    let decoded = decode_broadcast_delta(&body);
    assert!(
        !decoded.is_diff,
        "relayed field-only delta is is_diff=false"
    );
    assert!(
        decoded.struct_commands.is_empty(),
        "a field-only update relays NO struct command (reconstruction would synthesize a \
         DiffChangeSection for the touched section); got {:?}",
        decoded.struct_commands
    );
    assert_eq!(decoded.set_fields.len(), 1);
    assert_eq!(
        decoded.set_fields[0].set_order, MOVE_SET_ORDER,
        "writer's set_order must be preserved verbatim"
    );
}

// (c) Burst-no-coalesce: rapid same-field updates each relay their OWN commit
//     rather than coalescing to an empty broadcast (the prior compute_diff_at_cut
//     window-filter symptom). Each broadcast body must be non-empty and decode
//     to the field set on THAT commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_burst_relays_each_commit_no_coalesce() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "relay-burst-session", "relay-burst-layer")
            .await;

    // Establish the target section first (seq=2).
    let create_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 5,
            section_name: "Dragged".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-burst-layer",
        object_id,
        "writer-setup",
        10,
        &create_delta,
    )
    .await;

    let mut stream = {
        let sub_req = subscribe_request(&session_id, "relay-burst-layer", 0, 0, "sub-burst");
        let response = handle_subscribe_read_layer(&svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut s = response.into_inner();
        let (initial, _body) = recv_message(&mut s).await;
        assert_eq!(initial.sequence_num, 2);
        recv_caught_up_ack(&mut s, object_id, 2).await;
        s
    };

    // A drag: N rapid updates to the SAME field, each with a distinct value and
    // its own set_order. Pre-b.gze, intermediate sequences reconstructed as
    // EMPTY broadcasts because materialized state keeps only the latest version.
    const N: u64 = 6;
    let mut sent_ompf = Vec::new();
    for i in 0..N {
        let drag_delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![SetField {
                section_id: 5,
                key_name: "xformOp:translate".to_owned(),
                value_or_ext_hash: vec![i as u8, (i + 1) as u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1000 + i,
            }],
            ..Default::default()
        };
        let ompf = apply_delta_update_returning_ompf(
            &svc,
            &session_id,
            "relay-burst-layer",
            object_id,
            "writer-dragger",
            100 + i,
            &drag_delta,
        )
        .await;
        sent_ompf.push(ompf);
    }

    // Each of the N commits must arrive as its OWN non-empty broadcast carrying
    // that commit's field value + set_order — none coalesced to empty.
    for i in 0..N {
        let expected_seq = 3 + i;
        let (header, body) = recv_message(&mut stream).await;
        assert_eq!(
            header.sequence_num, expected_seq,
            "burst commit #{i} must arrive at its own seq"
        );
        assert!(
            !body.is_empty(),
            "burst commit #{i} broadcast body must be non-empty — a coalesced empty \
             broadcast is the b.gze burst symptom"
        );
        assert_eq!(
            body, sent_ompf[i as usize],
            "burst commit #{i} must relay that commit's RAW OMPF verbatim"
        );
        let decoded = decode_broadcast_delta(&body);
        assert_eq!(
            decoded.set_fields.len(),
            1,
            "burst commit #{i} must carry exactly its own SetField"
        );
        assert_eq!(
            decoded.set_fields[0].set_order,
            1000 + i,
            "burst commit #{i} must preserve its own set_order"
        );
    }
}

// (d) Catch-up still reconstructs: a brand-new joiner gets the reconstructed
//     full-content diff (is_diff=true / DiffChangeSection), unchanged by b.gze.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_catch_up_still_reconstructs_full_content_diff() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "catchup-recon-session", "catchup-recon-layer")
            .await;

    // Writer creates a prim + field BEFORE any subscriber connects, so the
    // joiner is genuinely brand-new and must receive reconstructed full state.
    let writer_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 33,
            section_name: "Established".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: 33,
            key_name: "size".to_owned(),
            value_or_ext_hash: vec![7],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 555,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "catchup-recon-layer",
        object_id,
        "writer-pre-join",
        100,
        &writer_delta,
    )
    .await;

    // Brand-new subscribe (object_id=0, seq=0) → full-content catch-up at seq=2.
    let sub_req = subscribe_request(&session_id, "catchup-recon-layer", 0, 0, "late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.object_id, object_id);
    assert_eq!(
        header.sequence_num, 2,
        "brand-new catch-up delivers state at the current seq=2"
    );
    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "brand-new catch-up MUST stay reconstructed full-content (is_diff=true) — unchanged \
         by b.gze; got is_diff={}",
        decoded.is_diff
    );
    assert!(
        decoded
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffChangeSection(cs) if cs.section_id == 33)),
        "reconstructed catch-up emits DiffChangeSection for the established section; got {:?}",
        decoded.struct_commands
    );
    recv_caught_up_ack(&mut stream, object_id, 2).await;
}

/// Issue an empty commit (SDD §8.4): an UpdateLayer from `client_id` with a
/// header and no chunks. Its stream entry carries no `ompf` field.
async fn apply_empty_update(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
) {
    let stream = streaming_update(vec![update_header(
        session_id, layer_name, object_id, client_id, 1,
    )]);
    handle_update_layer(svc, make_request(stream))
        .await
        .unwrap_or_else(|e| panic!("the empty UpdateLayer ({client_id}) failed: {e}"));
}

// (e) Empty commit: an UpdateLayer with no chunks has no relay bytes, and a
//     steady-state subscriber receives it as the empty delta at its own
//     sequence, based at the sequence before it, built with no read
//     (SDD §12.2) — never skipped. The Kit client applies contiguous sequences
//     only, so a subscriber that skipped another client's empty commit and
//     then wrote itself would stall on the gap. This is that shape: subscriber
//     A is caught up at 1; B commits empty (2); A commits (3, its own
//     self-echo); B commits for real (4). Then B commits empty (5), A commits
//     (6), B commits empty (7) and for real (8). A receives exactly
//     [2, 4, 5, 7, 8], and each empty delta is based at the sequence before
//     it: 1 after the catch-up, 4 after B's relayed commit, 6 after A's own
//     suppressed write. That base is A's tracked version, so the relay and
//     self-echo arms must each advance it.
//
//     The bound of a per-commit diff to its own commit is pinned on the
//     rollback replay, the one path left that rebuilds a commit from a read
//     (`regime2_replay_bounds_each_per_commit_to_its_own_sequence_num`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_empty_commit_is_delivered_as_the_empty_delta_and_never_skipped() {
    const LAYER: &str = "empty-commit-layer";
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "empty-commit-session", LAYER).await;
    let mut stream =
        subscribe_steady_state(&svc, &session_id, LAYER, object_id, 1, "client-a").await;

    apply_empty_update(&svc, &session_id, LAYER, object_id, "client-b").await;
    apply_unique_update(&svc, &session_id, LAYER, object_id, "client-a", 431).await;
    apply_unique_update(&svc, &session_id, LAYER, object_id, "client-b", 432).await;

    let (empty, body) = recv_message(&mut stream).await;
    assert_eq!(
        (empty.object_id, empty.sequence_num),
        (object_id, 2),
        "B's empty commit reaches A at its own sequence; skipped, A's first message would be 4"
    );
    assert_eq!(
        decode_broadcast_delta(&body),
        OwnedDelta {
            is_diff: true,
            base_version: 1,
            ..Default::default()
        },
        "the empty commit is the empty delta: no change, a diff based at the sequence before it"
    );
    let (real, body) = recv_message(&mut stream).await;
    assert_eq!(
        real.sequence_num, 4,
        "A's own commit (3) is suppressed as self-echo; B's real commit (4) follows"
    );
    assert_eq!(
        extract_diff_change_sections(&body)
            .into_iter()
            .map(|(section_id, _)| section_id)
            .collect::<Vec<_>>(),
        vec![432],
        "B's real commit is relayed as written"
    );

    // The empty delta's base is A's tracked version, the one place the steady
    // state puts it on the wire: after a relayed commit and after A's own
    // suppressed write, each must have advanced it.
    apply_empty_update(&svc, &session_id, LAYER, object_id, "client-b").await;
    apply_unique_update(&svc, &session_id, LAYER, object_id, "client-a", 433).await;
    apply_empty_update(&svc, &session_id, LAYER, object_id, "client-b").await;
    apply_unique_update(&svc, &session_id, LAYER, object_id, "client-b", 434).await;
    for (seq, base, after_what) in [
        (5, 4, "B's relayed commit (4)"),
        (7, 6, "A's own suppressed commit (6)"),
    ] {
        let (empty, body) = recv_message(&mut stream).await;
        assert_eq!(
            empty.sequence_num, seq,
            "nothing else reached A between: the empty commit at {seq} is next"
        );
        assert_eq!(
            decode_broadcast_delta(&body),
            OwnedDelta {
                is_diff: true,
                base_version: base,
                ..Default::default()
            },
            "the empty commit at {seq}, after {after_what}, is based at {base}"
        );
    }
    let (after, _) = recv_message(&mut stream).await;
    assert_eq!(
        after.sequence_num, 8,
        "B's next real commit follows: A received exactly [2, 4, 5, 7, 8]"
    );
}

// (f) Multi-part container relay: a delta whose SetField references an inline
//     external-value blob (OMPF Parts[1..N]) relays the ENTIRE container —
//     OODF delta (Part 0) plus the ext-value parts — byte-for-byte. The
//     production relay branch (src/grpc/subscribe.rs) explicitly relies on the
//     OMPF being self-contained: the delta plus any inline external-value parts
//     ride in the same byte string, so relaying preserves them with NO separate
//     ext-value fetch. A multi-part container is the case most likely to regress
//     if the relay path ever re-serializes instead of forwarding raw bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_relays_external_value_parts_verbatim() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "relay-extval-session", "relay-extval-layer")
            .await;

    let mut stream = subscribe_steady_state(
        &svc,
        &session_id,
        "relay-extval-layer",
        object_id,
        1,
        "sub-extval",
    )
    .await;

    // A large external value carried out-of-line as OMPF Part[1]; the SetField
    // references it via ext_value_size (non-zero) + ext_value_index (1-based ⇒
    // 1 = first ext-value part / OMPF Part[1]). The container therefore has
    // multiple parts: the OODF delta (Part 0) plus this blob (Part 1).
    const EXT_BLOB: &[u8] = b"the-quick-brown-fox-jumps-over-the-lazy-dog-x16-payload-blob";
    const DISTINCTIVE_SET_ORDER: u64 = 31337;
    let writer_delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 99,
            section_name: "BigAsset".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "mesh:points".to_owned(),
            // For an external value, value_or_ext_hash carries the content hash
            // reference rather than the inline bytes; the actual blob rides as
            // the ext-value part indexed by ext_value_index. The upload content
            // check refuses any claim but the blob's own content hash.
            value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(EXT_BLOB)
                .to_vec(),
            ext_value_size: EXT_BLOB.len() as u64,
            ext_value_index: 1,
            set_order: DISTINCTIVE_SET_ORDER,
        }],
        ..Default::default()
    };

    let writer_ompf = apply_delta_update_returning_ompf_with_ext(
        &svc,
        &session_id,
        "relay-extval-layer",
        object_id,
        "writer-extval",
        100,
        &writer_delta,
        &[EXT_BLOB],
    )
    .await;

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.sequence_num, 2, "ext-value commit is seq=2");
    assert_eq!(header.object_id, object_id);

    // The whole multi-part container — Part 0 delta AND Part 1 ext-value blob —
    // must arrive byte-for-byte. If the relay path re-serialized (rebuilding the
    // OMPF from materialized state, fetching the blob separately) the bytes would
    // differ even if the logical content matched.
    assert_eq!(
        body, writer_ompf,
        "steady-state broadcast of a multi-part container must be the writer's RAW OMPF \
         verbatim (b.gze relay) — inline external-value parts ride along with no re-serialize"
    );

    // Independently confirm the relayed container actually carried the ext-value
    // part (so byte-equality above wasn't trivially of a single-part container)
    // and that the writer's shape survived.
    let container = OmpfContainer::parse(&body).expect("parse relayed multi-part OMPF");
    assert_eq!(
        container.external_values.len(),
        1,
        "relayed container must carry the single inline external-value part (Part 1)"
    );
    assert_eq!(
        container.external_values[0], EXT_BLOB,
        "the relayed external-value part must be the writer's blob byte-for-byte"
    );
    let decoded = decode_broadcast_delta(&body);
    assert!(!decoded.is_diff, "relayed delta must be is_diff=false");
    assert_eq!(decoded.set_fields.len(), 1, "the SetField must ride along");
    assert_eq!(
        decoded.set_fields[0].ext_value_size,
        EXT_BLOB.len() as u64,
        "the SetField's ext_value_size must be preserved verbatim"
    );
    assert_eq!(
        decoded.set_fields[0].set_order, DISTINCTIVE_SET_ORDER,
        "the writer's per-delta set_order must be preserved verbatim"
    );
}

// (g) Deletion relays verbatim: a delta bearing a DiffDeleteSection (the
//     fast-path's special-cased deletion shape — a distinct backend dispatch
//     case per CLAUDE.md) relays the writer's bytes byte-for-byte rather than
//     reconstructing a diff. The decoded command vocabulary must still be the
//     writer's DiffDeleteSection, not a reconstructed DiffChangeSection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_relays_deletion_verbatim() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "relay-del-session", "relay-del-layer").await;

    // Establish the section first (seq=2) so the deletion targets an existing
    // section — NOT what we assert on.
    let create_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 7,
            section_name: "Doomed".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-del-layer",
        object_id,
        "writer-setup",
        50,
        &create_delta,
    )
    .await;

    // Subscribe AFTER the create so the deletion is the first thing the
    // steady-state subscriber observes (seq=3).
    let mut stream = {
        let sub_req = subscribe_request(&session_id, "relay-del-layer", 0, 0, "sub-del");
        let response = handle_subscribe_read_layer(&svc, sub_req)
            .await
            .expect("subscribe must succeed");
        let mut s = response.into_inner();
        let (initial, _body) = recv_message(&mut s).await;
        assert_eq!(initial.object_id, object_id);
        assert_eq!(initial.sequence_num, 2);
        recv_caught_up_ack(&mut s, object_id, 2).await;
        s
    };

    // A DiffDeleteSection-only delta — the fast-path's special-cased deletion.
    let delete_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
            section_id: 7,
        })],
        ..Default::default()
    };
    let writer_ompf = apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-del-layer",
        object_id,
        "writer-deleter",
        100,
        &delete_delta,
    )
    .await;

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.sequence_num, 3, "deletion commit is seq=3");
    assert_eq!(header.object_id, object_id);
    assert_eq!(
        body, writer_ompf,
        "deletion steady-state broadcast must relay the writer's RAW OMPF verbatim (b.gze)"
    );
    let decoded = decode_broadcast_delta(&body);
    assert!(
        !decoded.is_diff,
        "relayed deletion delta must be is_diff=false"
    );
    assert!(
        matches!(
            decoded.struct_commands.as_slice(),
            [StructCommand::DiffDeleteSection(cs)] if cs.section_id == 7
        ),
        "relayed delta must carry the writer's DiffDeleteSection (not a reconstructed \
         DiffChangeSection); got {:?}",
        decoded.struct_commands
    );
}

// (h) Self-echo suppression on the relay path: a subscriber sharing the writer's
//     client_id must receive NOTHING for a relay-eligible update. In
//     src/grpc/subscribe.rs the self-echo filter precedes the relay send, so a
//     relay-eligible commit from the subscriber's own client_id is dropped
//     BEFORE the relay branch ever runs. This pins that ordering directly with a
//     bounded non-delivery assertion (rather than relying on the incidental
//     coverage in `self_echo_suppressed_but_other_clients_delivered`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_relay_path_suppresses_self_echo() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    let (session_id, object_id) = create_session_and_layer_pooled(
        &tv,
        &svc,
        "relay-selfecho-session",
        "relay-selfecho-layer",
    )
    .await;

    // Subscribe with the SAME client_id the writer will use ("writer-other").
    let mut stream = subscribe_steady_state(
        &svc,
        &session_id,
        "relay-selfecho-layer",
        object_id,
        1,
        "writer-other",
    )
    .await;

    // A relay-eligible update (CreateSection + SetField, is_diff=false) authored
    // by the SAME client_id. If self-echo suppression did not precede the relay
    // branch, the verbatim OMPF would be broadcast back to this subscriber.
    let writer_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 12,
            section_name: "Echoed".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: 12,
            key_name: "radius".to_owned(),
            value_or_ext_hash: vec![5, 6, 7],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2024,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        "relay-selfecho-layer",
        object_id,
        "writer-other",
        100,
        &writer_delta,
    )
    .await;

    // The subscriber must receive nothing within a bounded window — the commit
    // was self-echo-suppressed before the relay branch could send it.
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Catch-up regime switch (Plan b.msb, Task 3): off = unchanged legacy
//    full-content; snapshot = corrected faithful full-state create.
//
// `LiveEditService.catchup_snapshot_enabled` gates `send_full_state` in
// `src/grpc/subscribe.rs`: `Off` (default) runs today's legacy `compute_diff_at_cut`
// full-content path, `Snapshot` routes to Task 2's `compute_faithful_full_content_at_cut`
// builder. These two tests pin the wiring/dispatch + invariant preservation:
//   - off  → byte-for-byte today's brand-new catch-up (SR-1.2 via reconstruction).
//   - snapshot → still a valid from-empty full-state create (is_diff=true,
//     base_version=0) with the caught-up terminator (total_size==0) preserved
//     (SDD §12.3), AND the corrected builder is proven routed via the faithful
//     authoring `set_order` marker (vs the legacy cumulative cursor).
//
// The heavier end-to-end multi-commit establishment + no-wipe correctness
// assertions (SR-6.3 / SR-3.1) are Task 4 — deliberately NOT duplicated here.
//
// `catchup_snapshot_enabled` is a PUBLIC field and the fixture builds the service by
// value, so mode selection is local to THIS file: build the service, then flip
// the field on the owned instance before serving. The shared `helpers.rs`
// signature is untouched (a sibling test relies on it) and every pre-existing
// test in this file keeps the default `Off`, so none of them change behaviour.
// ═══════════════════════════════════════════════════════════════════════════════

/// Section id the establish-before-join writer creates in the catch-up-switch
/// tests below.
const SWITCH_SECTION_ID: u64 = 33;
/// Authoring `set_order` of the FIRST field the establish writer sets. Applied
/// against a fresh layer (`next_set_order == 0`), so its stored cumulative
/// `set_order` equals this value — the field only exists to advance the cursor
/// so the SECOND field's cumulative and authoring orders diverge.
const SWITCH_FIRST_AUTHORING_SET_ORDER: u64 = 100;
/// Authoring `set_order` of the SECOND ("marker") field. After the first field
/// advanced `next_set_order` to `SWITCH_FIRST_AUTHORING_SET_ORDER + 1 == 101`,
/// this field's STORED cumulative `set_order` becomes `101 + 42 == 143`, while
/// its authoring rank stays `42`. The two builders therefore diverge on the
/// marker field's emitted `set_order`:
///   - legacy `compute_diff_at_cut` (`Off`) clones the stored field → emits the
///     cumulative cursor `143`.
///   - faithful snapshot (`Snapshot`) emits a DENSE-UNIQUE index (b.bdx): the
///     two live fields sort by `(version, authoring_set_order)` to first→0,
///     marker→1, so the marker emits `1` — neither the raw authoring rank (42)
///     nor the cumulative cursor (143).
///
/// That divergence is the observable proving which builder the dispatch routed
/// to.
const SWITCH_MARKER_AUTHORING_SET_ORDER: u64 = 42;
/// The stored cumulative `set_order` the legacy path would emit for the marker
/// field (`101 + 42`). Used only to assert the snapshot path is NOT emitting it.
const SWITCH_MARKER_CUMULATIVE_SET_ORDER: u64 = 143;
/// Server sequence_num after CreateLayer (seq=1) + two establish updates.
const SWITCH_ESTABLISHED_SEQ: u64 = 3;

/// Establish a prim + two fields on `layer_name` BEFORE any subscriber connects,
/// engineered so the marker field's cumulative and authoring `set_order` diverge
/// (see the `SWITCH_*` constants). Two writer updates land: create+first-field
/// (seq=2) then the marker field (seq=3). Returns nothing — the caller knows the
/// resulting seq is `SWITCH_ESTABLISHED_SEQ`.
async fn establish_prim_with_diverging_set_order(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) {
    // Update 1: create the prim + a first field. Advances `next_set_order` from
    // 0 to `SWITCH_FIRST_AUTHORING_SET_ORDER + 1`.
    let create_delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: SWITCH_SECTION_ID,
            section_name: "Established".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: SWITCH_SECTION_ID,
            key_name: "first".to_owned(),
            value_or_ext_hash: vec![1],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: SWITCH_FIRST_AUTHORING_SET_ORDER,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        session_id,
        layer_name,
        object_id,
        "writer-pre-join",
        100,
        &create_delta,
    )
    .await;

    // Update 2: field-only "marker" write on the SAME prim. Now `next_set_order`
    // is non-zero, so the stored cumulative order (143) diverges from the raw
    // authoring rank (42).
    let marker_delta = OwnedDelta {
        is_diff: false,
        set_fields: vec![SetField {
            section_id: SWITCH_SECTION_ID,
            key_name: "marker".to_owned(),
            value_or_ext_hash: vec![2],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: SWITCH_MARKER_AUTHORING_SET_ORDER,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        session_id,
        layer_name,
        object_id,
        "writer-pre-join",
        200,
        &marker_delta,
    )
    .await;
}

// (a) Off = unchanged: with the switch explicitly `Off`, a brand-new joiner
//     gets today's reconstructed full-content diff (is_diff=true) at the current
//     seq, terminated by the total_size==0 caught-up ack — identical to
//     `brand_new_catch_up_still_reconstructs_full_content_diff`. Pins that
//     setting `Off` explicitly is a no-op relative to the default.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catchup_switch_off_delivers_legacy_reconstructed_full_content() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    // Explicit — this is the default, but the test's contract is that `Off` is a
    // no-op relative to the default-behaviour tests, so pin it locally.
    svc.catchup_snapshot_enabled = false;

    let (session_id, object_id) = create_session_and_layer_pooled(
        &tv,
        &svc,
        "catchup-switch-off-session",
        "catchup-switch-off-layer",
    )
    .await;

    establish_prim_with_diverging_set_order(
        &svc,
        &session_id,
        "catchup-switch-off-layer",
        object_id,
    )
    .await;

    // Brand-new subscribe (object_id=0, seq=0) → full-content catch-up.
    let sub_req = subscribe_request(&session_id, "catchup-switch-off-layer", 0, 0, "late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, SWITCH_ESTABLISHED_SEQ,
        "off-mode catch-up delivers state at the current seq"
    );

    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "off-mode brand-new catch-up stays reconstructed full-content (is_diff=true); got {}",
        decoded.is_diff
    );
    // Distinguishing marker: the legacy `compute_diff_at_cut` path clones the stored
    // field, so the marker's emitted set_order is the CUMULATIVE cursor (143),
    // NOT the authoring rank — the exact opposite of the snapshot path below.
    let marker = decoded
        .set_fields
        .iter()
        .find(|sf| sf.key_name == "marker")
        .expect("off-mode reconstructed full content must carry the marker field");
    assert_eq!(
        marker.set_order, SWITCH_MARKER_CUMULATIVE_SET_ORDER,
        "off-mode (legacy compute_diff_at_cut) emits the stored cumulative set_order (143), not the \
         authoring rank"
    );

    // The caught-up terminator (total_size==0) is preserved in off mode.
    recv_caught_up_ack(&mut stream, object_id, SWITCH_ESTABLISHED_SEQ).await;
}

// (b) Snapshot = corrected full-state create dispatched: with the switch
//     `Snapshot`, message #1 is still a valid from-empty full-state create
//     (is_diff=true, base_version=0) that applies cleanly onto an EMPTY tree,
//     the total_size==0 caught-up terminator is still emitted (SDD §12.3), and
//     the faithful authoring `set_order` marker proves the dispatch routed to
//     Task 2's builder rather than the legacy `compute_diff_at_cut` path.
//
// NOTE: deep multi-commit correctness (SR-6.3 / SR-3.1) is Task 4 — not here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catchup_switch_snapshot_dispatches_corrected_full_state_create() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    // Select snapshot mode locally on this owned instance — the shared
    // `helpers.rs` builder signature is untouched.
    svc.catchup_snapshot_enabled = true;

    let (session_id, object_id) = create_session_and_layer_pooled(
        &tv,
        &svc,
        "catchup-switch-snap-session",
        "catchup-switch-snap-layer",
    )
    .await;

    establish_prim_with_diverging_set_order(
        &svc,
        &session_id,
        "catchup-switch-snap-layer",
        object_id,
    )
    .await;

    // Brand-new subscribe (object_id=0, seq=0) → corrected full-state snapshot.
    let sub_req = subscribe_request(
        &session_id,
        "catchup-switch-snap-layer",
        0,
        0,
        "late-joiner",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, SWITCH_ESTABLISHED_SEQ,
        "snapshot-mode catch-up delivers state at the current seq"
    );
    assert!(
        !body.is_empty(),
        "snapshot-mode message #1 must carry a non-empty full-state body"
    );

    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "snapshot-mode message #1 must be a full-state create (is_diff=true); got {}",
        decoded.is_diff
    );
    assert_eq!(
        decoded.base_version, 0,
        "snapshot-mode message #1 must be from-empty (base_version=0) per SR-1.2"
    );

    // Distinguishing marker (b.bdx): the faithful builder emits a DENSE-UNIQUE
    // `set_order`, NOT the raw authoring rank and NOT the cumulative cursor. Two
    // live fields on section 33 — "first" (version 2, authoring 100) and "marker"
    // (version 3, authoring 42) — sort by `(version, authoring_set_order)` to
    // first→0, marker→1. So the marker's dense set_order is 1: still distinct
    // from the legacy cumulative cursor (143, pinned by the off-mode test),
    // proving the dispatch routed to the faithful builder, not `compute_diff_at_cut`.
    const SWITCH_MARKER_DENSE_SET_ORDER: u64 = 1;
    let marker = decoded
        .set_fields
        .iter()
        .find(|sf| sf.key_name == "marker")
        .expect("snapshot full-state create must carry the marker field");
    assert_eq!(
        marker.set_order, SWITCH_MARKER_DENSE_SET_ORDER,
        "snapshot-mode (faithful builder) must emit the dense-unique set_order (1), NOT the raw \
         authoring rank ({SWITCH_MARKER_AUTHORING_SET_ORDER}) and NOT the legacy cumulative cursor \
         ({SWITCH_MARKER_CUMULATIVE_SET_ORDER}) — this proves the mode dispatch routed to \
         compute_faithful_full_content_at_cut"
    );

    // The body must apply cleanly onto an EMPTY ObjectState (it's a valid
    // from-empty create, not a bare diff), reconstructing the established prim +
    // both fields.
    let mut reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut reconstructed, &body);
    let section = reconstructed
        .sections
        .get(&SWITCH_SECTION_ID)
        .expect("established prim must reconstruct on the empty tree");
    assert_eq!(
        section.name, "Established",
        "reconstructed prim name must match the established section"
    );
    assert!(
        reconstructed
            .fields
            .contains_key(&(SWITCH_SECTION_ID, "first".to_owned())),
        "reconstructed tree must carry the 'first' field"
    );
    assert!(
        reconstructed
            .fields
            .contains_key(&(SWITCH_SECTION_ID, "marker".to_owned())),
        "reconstructed tree must carry the 'marker' field"
    );

    // The caught-up terminator (total_size==0) is still emitted in snapshot mode
    // — `run_catch_up` owns it uniformly across both regimes (SDD §12.3).
    recv_caught_up_ack(&mut stream, object_id, SWITCH_ESTABLISHED_SEQ).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SR-6.3 / SR-3.1 — corrected-snapshot late-join establishment + no-wipe (t3.msb.35.7t.ub)
//
// End-to-end in-process coverage of the wired `Snapshot` catch-up mode against a
// real Valkey. Task 4 promotes the switch smoke tests above (which only prove the
// dispatch routed to the faithful builder) into deep multi-commit correctness:
//   1. `snapshot_late_join_catch_up_establishes_correct_scene` — a fresh
//      late-joiner's message #1, applied to an EMPTY tree, reproduces exactly the
//      live scene (created-then-abandoned scratch prim dropped, final field values
//      correct, TRUE authoring `set_order` preserved).
//   2. `snapshot_caught_up_client_write_preserves_prior_state` — after a snapshot-
//      caught-up client issues its next incremental write, the prior writer's scene
//      survives intact (SR-3.1 no-wipe), positively asserted.
//
// Both build the service in `Snapshot` mode by flipping the PUBLIC
// `catchup_snapshot_enabled` field on the owned instance (shared `helpers.rs` untouched;
// `PooledTestValkey`'s default stays `Off`). `checkpoint_enabled` stays `false`
// (SR-4.1) — the fixture `Config` already sets it, and no DR path is engaged.
// ═══════════════════════════════════════════════════════════════════════════════

/// Non-existent parent id the establishment test reparents the scratch prim under
/// to make it net-live-unreachable (abandoned) without deleting it — so it stays
/// materialized in Valkey (an orphan in `read_object_state_for_test`) but is DROPPED by
/// the faithful snapshot builder. This is exactly the divergence the corrected
/// catch-up path fixes (SR-5.2 reachable-from-root pruning).
const ABANDON_ORPHAN_PARENT: u64 = 888_888;

/// 1. Establishment (SR-1.1 / SR-6.3 first clause): build the service in
///    `Snapshot` mode, commit a multi-commit editing history (parent→child
///    sections with distinct `section_order`; a burst of field updates across
///    separate commits so the true last-writer value + authoring rank matter; and
///    a created-then-abandoned scratch prim), then late-join a FRESH subscriber and
///    prove message #1 applied to an EMPTY tree reconstructs the correct CURRENT
///    scene with the abandoned prim absent and authoring order preserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_late_join_catch_up_establishes_correct_scene() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-establish-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-establish-session", layer).await;

    // ── Commit 1 (seq=2): root R(1) + child A(2), each with a first field. ──
    // R.name authoring rank 1; A.posX authoring rank 5 with the INITIAL value.
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;

    // ── Commit 2 (seq=3): child B(3); BURST-update A.posX with the FINAL value ──
    // at authoring rank 2 (last write wins), plus B.color at authoring rank 7.
    // After commit 1 advanced `next_set_order`, A.posX's stored CUMULATIVE
    // set_order (10) now diverges from its authoring rank (2) — the observable
    // that proves the faithful builder emits the authoring rank, not the cursor.
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"final".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;

    // ── Commit 3 (seq=4): create scratch prim S(99) under child A + a field. ──
    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "temp".to_owned(),
            value_or_ext_hash: b"junk".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 3,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    // ── Commit 4 (seq=5): ABANDON scratch S(99) — reparent it under a
    // non-existent parent so it becomes unreachable from root (net-live pruning),
    // WITHOUT deleting it (it stays materialized as an orphan in Valkey). ──
    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        400,
        &commit4,
    )
    .await;

    const ESTABLISHED_SEQ: u64 = 5; // CreateLayer(1) + 4 commits.

    // ── Late-join a FRESH subscriber (object_id=0, seq=0 → brand-new). ──
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, ESTABLISHED_SEQ,
        "snapshot-mode catch-up delivers the current scene at seq={ESTABLISHED_SEQ}"
    );
    assert!(
        header.total_size > 0,
        "message #1 must carry a non-empty full-state body (total_size>0)"
    );

    // Message #1 must be a valid from-empty full-state create (SR-1.2).
    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "message #1 must be a full-state create (is_diff=true)"
    );
    assert_eq!(
        decoded.base_version, 0,
        "message #1 must be from-empty (base_version=0)"
    );

    // ── (b) The created-then-abandoned scratch prim must be ABSENT from the
    //        emitted delta (no DiffChangeSection for 99, no tombstone). ──
    let emitted_section_ids: std::collections::HashSet<u64> = decoded
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        emitted_section_ids,
        [1u64, 2, 3].into_iter().collect(),
        "message #1 must emit exactly the live sections {{1,2,3}} — abandoned prim 99 dropped"
    );
    assert!(
        !decoded
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "faithful snapshot must emit NO tombstones for a from-empty joiner"
    );
    assert!(
        !decoded.set_fields.iter().any(|sf| sf.section_id == 99),
        "the abandoned prim's field must be dropped, not emitted"
    );

    // Emitted section_order must be the REAL per-parent sibling index (b.bdx),
    // NOT the stored `section_order`. Tree: R(1) is root's only child → 0;
    // A(2) and B(3) are R's children in creation order → A=0, B=1. (The
    // abandoned scratch S(99), reparented away, is not a live child of R.)
    let emitted_section_order = |sid: u64| -> u32 {
        decoded
            .struct_commands
            .iter()
            .find_map(|c| match c {
                StructCommand::DiffChangeSection(s) if s.section_id == sid => Some(s.section_order),
                _ => None,
            })
            .unwrap_or_else(|| panic!("section {sid} must be emitted"))
    };
    assert_eq!(emitted_section_order(1), 0, "R sibling index under root");
    assert_eq!(emitted_section_order(2), 0, "A sibling index 0 under R");
    assert_eq!(emitted_section_order(3), 1, "B sibling index 1 under R");

    // ── (d) Field ordering preserved via DENSE-UNIQUE set_order (b.bdx): the
    //        emitted set_order is a dense index 0..N-1 ordered by `(version,
    //        authoring_set_order)`, NOT the raw authoring rank and NOT the
    //        cumulative cursor. R.name (commit 1, authoring 1) → 0; A.posX
    //        (commit 2 last-writer, authoring 2) → 1; B.color (commit 2,
    //        authoring 7) → 2. ──
    let emitted_set_order = |sid: u64, key: &str| -> u64 {
        decoded
            .set_fields
            .iter()
            .find(|sf| sf.section_id == sid && sf.key_name == key)
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be emitted"))
            .set_order
    };
    assert_eq!(emitted_set_order(1, "name"), 0, "R.name dense set_order");
    assert_eq!(
        emitted_set_order(2, "posX"),
        1,
        "A.posX dense set_order (LAST-writer burst survives; ordered after R.name, before B.color)"
    );
    assert_eq!(emitted_set_order(3, "color"), 2, "B.color dense set_order");
    // Global uniqueness across the whole catch-up (the client's saveCount-keyed
    // map drops duplicates): all emitted set_orders are distinct.
    let all_set_orders: Vec<u64> = decoded.set_fields.iter().map(|sf| sf.set_order).collect();
    let mut uniq_orders = all_set_orders.clone();
    uniq_orders.sort_unstable();
    uniq_orders.dedup();
    assert_eq!(
        uniq_orders.len(),
        all_set_orders.len(),
        "emitted set_order values must be globally unique across the catch-up"
    );
    // Relative order across fields (name < posX < color) is preserved.
    assert!(
        emitted_set_order(1, "name") < emitted_set_order(2, "posX")
            && emitted_set_order(2, "posX") < emitted_set_order(3, "color"),
        "emitted set_order must preserve the (version, authoring) apply order"
    );

    // ── Cross-check against the authoritative server state (the mirror). The
    //     server carries A.posX at its CUMULATIVE set_order (10) yet the emitted
    //     rank is the authoring value (2) — proving the divergence. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    let server_posx = server_state
        .fields
        .get(&(2, "posX".to_owned()))
        .expect("server must carry A.posX");
    assert_eq!(
        server_posx.authoring_set_order, 2,
        "server A.posX authoring rank"
    );
    assert_eq!(
        server_posx.set_order, 10,
        "server A.posX cumulative cursor (next_set_order threaded) is 10, distinct from the rank"
    );
    assert_ne!(
        emitted_set_order(2, "posX"),
        server_posx.set_order,
        "emitted rank must NOT be the cumulative cursor the server stores"
    );
    // The abandoned prim IS still materialized server-side (an orphan) — that's
    // exactly the state the faithful snapshot corrects for the from-empty joiner.
    assert!(
        server_state.sections.contains_key(&99),
        "abandoned prim 99 must still exist server-side as an orphan (the divergence the snapshot fixes)"
    );

    // ── (a)+(c) Apply message #1 to an EMPTY ObjectState — exactly what a Kit
    //           client does from an empty tree — and assert the reconstructed
    //           CURRENT scene. ──
    let mut reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut reconstructed, &body);

    let reconstructed_ids: std::collections::HashSet<u64> =
        reconstructed.sections.keys().copied().collect();
    assert_eq!(
        reconstructed_ids,
        [1u64, 2, 3].into_iter().collect(),
        "reconstructed scene must be exactly the live sections {{1,2,3}}"
    );
    assert!(
        !reconstructed.sections.contains_key(&99),
        "abandoned scratch prim 99 must be ABSENT from the reconstructed scene"
    );
    assert!(
        !reconstructed.fields.contains_key(&(99, "temp".to_owned())),
        "the abandoned prim's field must be absent from the reconstructed scene"
    );

    // Structural parity (id/parent/name/type) on the surviving prims. NOTE:
    // `section_order` is deliberately NOT compared against the server here — the
    // corrected snapshot emits the REAL per-parent sibling index (b.bdx), which
    // differs from the arbitrary stored `section_order` the server persists
    // (0/10/20). The reconstructed sibling order is asserted separately below.
    for sid in [1u64, 2, 3] {
        let r = reconstructed
            .sections
            .get(&sid)
            .expect("live section reconstructs");
        let s = server_state
            .sections
            .get(&sid)
            .expect("live section on server");
        assert_eq!(r.section_id, s.section_id, "section {sid} id parity");
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }
    // Reconstructed sibling order (b.bdx): R=0 (only root child), A=0 and B=1
    // (R's children in creation order) — the value a Kit client would use to
    // reconstruct child ordering, distinct from the server's stored value.
    assert_eq!(
        reconstructed.sections[&1].section_order, 0,
        "reconstructed R sibling index"
    );
    assert_eq!(
        reconstructed.sections[&2].section_order, 0,
        "reconstructed A sibling index"
    );
    assert_eq!(
        reconstructed.sections[&3].section_order, 1,
        "reconstructed B sibling index"
    );

    // Final field-VALUE parity on the surviving prims (the burst's last-writer
    // value must win; ordering-cursor differences are irrelevant to value).
    for (sid, key, expected) in [
        (1u64, "name", &b"scene"[..]),
        (2, "posX", &b"final"[..]),
        (3, "color", &b"red"[..]),
    ] {
        let r = reconstructed
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("reconstructed must carry field ({sid},{key})"));
        assert_eq!(
            r.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) must reconstruct its final value"
        );
        let s = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("server must carry field ({sid},{key})"));
        assert_eq!(
            r.value, s.value,
            "field ({sid},{key}) value must match the server (mirror)"
        );
    }

    // The caught-up terminator (total_size==0) is still emitted (SDD §12.3).
    recv_caught_up_ack(&mut stream, object_id, ESTABLISHED_SEQ).await;
}

/// 2. No-wipe (SR-3.1): after a snapshot-caught-up client B issues its next
///    incremental write, writer A's prior sections + field values survive intact
///    AND B's new section is present — asserted POSITIVELY (not inferred from the
///    establishment test). A present-from-start third subscriber observes B's
///    per-commit relay with no delete/reset of A's prims.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_caught_up_client_write_preserves_prior_state() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-nowipe-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-nowipe-session", layer).await;

    // ── Writer A commits a scene of two sections with fields (seq 2 & 3). ──
    let a_commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "a-root".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "a1".to_owned(),
            value_or_ext_hash: b"va1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        100,
        &a_commit1,
    )
    .await;

    let a_commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a-child".to_owned(),
            section_type: 0,
            section_order: 5,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "a2".to_owned(),
            value_or_ext_hash: b"va2".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        200,
        &a_commit2,
    )
    .await;

    const A_ESTABLISHED_SEQ: u64 = 3; // CreateLayer(1) + A's two commits.

    // ── Present-from-start witness C: subscribe BEFORE B's write, capture its
    //     snapshot catch-up body, reconstruct C's local tree. ──
    let c_req = subscribe_request(&session_id, layer, 0, 0, "witness-c");
    let c_response = handle_subscribe_read_layer(&svc, c_req)
        .await
        .expect("witness subscribe must succeed");
    let mut c_stream = c_response.into_inner();
    let (c_header, c_body) = recv_message(&mut c_stream).await;
    assert_eq!(c_header.object_id, object_id);
    assert_eq!(c_header.sequence_num, A_ESTABLISHED_SEQ);
    let mut c_state = ObjectState::default();
    apply_ompf_to_state(&mut c_state, &c_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "witness catch-up must reconstruct A's scene (sections 1 and 2)"
    );
    recv_caught_up_ack(&mut c_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── Client B late-joins, drains its snapshot catch-up + caught-up ack. ──
    let b_req = subscribe_request(&session_id, layer, 0, 0, "client-b");
    let b_response = handle_subscribe_read_layer(&svc, b_req)
        .await
        .expect("client-b subscribe must succeed");
    let mut b_stream = b_response.into_inner();
    let (b_header, _b_body) = recv_message(&mut b_stream).await;
    assert_eq!(b_header.object_id, object_id);
    assert_eq!(b_header.sequence_num, A_ESTABLISHED_SEQ);
    recv_caught_up_ack(&mut b_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── B, now caught-up, issues its next INCREMENTAL (non-reset) write adding
    //     its own section + field (seq=4). Distinct client_id from the witness so
    //     C receives the relay (and B's own stream self-echo-suppresses it). ──
    let b_commit = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 3,
            section_name: "b-root".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "b1".to_owned(),
            value_or_ext_hash: b"vb1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "client-b",
        300,
        &b_commit,
    )
    .await;

    // ── POSITIVE property #1: server state after B's write still contains A's
    //     sections + field values intact, PLUS B's new section. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    for (sid, key, expected) in [
        (1u64, "a1", &b"va1"[..]),
        (2, "a2", &b"va2"[..]),
        (3, "b1", &b"vb1"[..]),
    ] {
        assert!(
            server_state.sections.contains_key(&sid),
            "section {sid} must survive/exist after B's write"
        );
        let f = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be present after B's write"));
        assert_eq!(
            f.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) value must be intact after B's write (no wipe)"
        );
    }

    // ── POSITIVE property #2: the witness receives B's per-commit relay, and its
    //     reconstruction retains A's prims (no delete/reset/wipe arrives). ──
    let (relay_header, relay_body) = recv_message(&mut c_stream).await;
    assert_eq!(
        relay_header.object_id, object_id,
        "relay object_id must be stable (no overwrite)"
    );
    assert_eq!(
        relay_header.sequence_num, 4,
        "witness must observe B's write as a single per-commit relay at seq=4"
    );
    let relay_delta = decode_broadcast_delta(&relay_body);
    assert!(
        !(relay_delta.is_diff && relay_delta.base_version == 0),
        "B's relayed per-commit write must NOT be a reset (would wipe A's prims)"
    );
    assert!(
        !relay_delta
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "B's relayed write must carry no section deletions of A's prims"
    );

    // Apply the relay to the witness's reconstructed tree: A's sections must
    // survive and B's section must be added.
    apply_ompf_to_state(&mut c_state, &relay_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "A's sections must survive in the witness's tree after B's relayed write"
    );
    assert!(
        c_state.sections.contains_key(&3),
        "B's new section must be added to the witness's tree"
    );
    assert_eq!(
        c_state.fields.get(&(1, "a1".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"va1".to_vec()
        )),
        "A's field value must survive in the witness's tree (no wipe)"
    );

    // B's own stream self-echo-suppresses its write — nothing else arrives.
    assert_no_next_message(&mut b_stream, Duration::from_millis(300)).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SR-6.3 / SR-5.2 — corrected-snapshot late-join on a GENUINELY TRIMMED stream
// (t3.msb.js.eg.vx)
//
// The Task-4 snapshot tests above (`snapshot_late_join_catch_up_establishes_correct_scene`,
// `snapshot_caught_up_client_write_preserves_prior_state`) run in Snapshot mode but
// with MAXLEN=4096 and no trim; `natural_out_of_window_without_explicit_xtrim_falls_into_regime3`
// forces a real approximate trim but in Off/legacy mode. Neither exercises the
// intersection: a fresh Snapshot-mode late-join on a stream whose earliest entries
// have provably been evicted by Valkey's approximate `MAXLEN ~ N` trim. Because
// `compute_faithful_full_content_at_cut` reconstructs from MATERIALIZED state + persisted
// `authoring_set_order` and NEVER reads the stream, the established scene must be
// bit-identical whether or not history survives — but nothing asserts that today.
//
// These three tests use the proven forced-trim recipe from the Off-mode test
// (MAXLEN=4, ~200 distinct writes, XLEN/XRANGE-verify entry `2-0` is evicted),
// but drive the scene under `catchup_snapshot_enabled = true`. All run under the
// default DR-disabled `build_pooled_service` wiring. Every stream await is bounded
// by `recv_message` / `SUBSCRIBER_DEADLINE`; no `proptest!` macros are added here.
// ═══════════════════════════════════════════════════════════════════════════════

/// 1. A forced-trim + Snapshot-mode late-join still establishes the correct scene.
///    Build the multi-commit editing history (root + children, a last-writer-wins
///    burst whose authoring rank diverges from the cumulative cursor, and a
///    created-then-abandoned scratch prim), THEN drive a large heartbeat write
///    burst against a small MAXLEN so approximate `MAXLEN ~ N` trim evicts the
///    earliest entries. A fresh late-joiner's message #1 applied to an EMPTY tree
///    must reproduce the current scene exactly — retention-independently.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_late_join_on_trimmed_stream_establishes_correct_scene() {
    let tv = PooledTestValkey::setup().await;
    // Small MAXLEN so XADD's approximate `MAXLEN ~ N` trim clips the earliest
    // entries once the write count crosses Valkey's per-node threshold (~100).
    const MAXLEN: u64 = 4;
    // Distinct heartbeat writes after the scene commits, sized to reliably evict
    // entry `2-0` on redis/valkey 7.x (same threshold the Off-mode test proved).
    const HEARTBEAT_WRITES: u64 = 200;

    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), MAXLEN);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-trim-establish-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-trim-establish-session", layer).await;

    // ── Commit 1 (seq=2): root R(1) + child A(2), each with a first field. ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;

    // ── Commit 2 (seq=3): child B(3); BURST-update A.posX with the FINAL value at
    //    authoring rank 2 (last write wins) — its stored cumulative cursor now
    //    diverges from the authoring rank; plus B.color at authoring rank 7. ──
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"final".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;

    // ── Commit 3 (seq=4): create scratch prim S(99) under child A + a field. ──
    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "temp".to_owned(),
            value_or_ext_hash: b"junk".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 3,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    // ── Commit 4 (seq=5): ABANDON scratch S(99) — reparent under a non-existent
    //    parent so it becomes unreachable from root, WITHOUT deleting it (stays
    //    materialized as an orphan in Valkey). ──
    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        400,
        &commit4,
    )
    .await;

    // ── Commit 5 (seq=6): create a reachable heartbeat prim H(5) whose single
    //    field we then hammer to drive the trim without adding new sections. ──
    let commit5 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 5,
            section_name: "heartbeat".to_owned(),
            section_type: 0,
            section_order: 40,
        })],
        set_fields: vec![SetField {
            section_id: 5,
            key_name: "hb".to_owned(),
            value_or_ext_hash: b"hb-init".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        500,
        &commit5,
    )
    .await;

    // ── Heartbeat burst: HEARTBEAT_WRITES distinct field updates to H.hb so each
    //    commit is a non-empty XADD, driving approximate MAXLEN trim past `2-0`.
    //    These mutate ONLY the heartbeat field — the live-section set stays
    //    {1,2,3,5} (99 abandoned), so the scene assertions below remain exact. ──
    for i in 0..HEARTBEAT_WRITES {
        let beat = OwnedDelta {
            is_diff: false,
            set_fields: vec![SetField {
                section_id: 5,
                key_name: "hb".to_owned(),
                value_or_ext_hash: format!("hb-{i}").into_bytes(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            ..Default::default()
        };
        apply_delta_update_returning_ompf(
            &svc,
            &session_id,
            layer,
            object_id,
            "scene-writer",
            5_000 + i,
            &beat,
        )
        .await;
    }

    // CreateLayer(seq=1) + 5 scene commits + HEARTBEAT_WRITES beats.
    let total_writes = 5 + HEARTBEAT_WRITES;
    let established_seq = total_writes + 1;

    // ── Prove the approximate trim actually fired: XLEN well below the total,
    //    non-zero, and the earliest UpdateLayer commit (entry `2-0`) evicted. ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let xlen: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert!(
        xlen < total_writes,
        "approximate MAXLEN trim did not fire: XLEN={xlen} >= total_writes={total_writes}"
    );
    assert!(
        xlen > 0,
        "stream must retain at least one entry (got XLEN=0)"
    );
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "entry 2-0 must have been trimmed by approximate MAXLEN — got {probe:?}"
    );

    // ── Late-join a FRESH subscriber (object_id=0, seq=0 → brand-new regime). ──
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, established_seq,
        "snapshot-mode catch-up delivers the current scene at the established seq"
    );
    assert!(
        header.total_size > 0,
        "message #1 must carry a non-empty full-state body (total_size>0)"
    );

    // Message #1 must be a valid from-empty full-state create (SR-1.2).
    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "message #1 must be a full-state create (is_diff=true)"
    );
    assert_eq!(
        decoded.base_version, 0,
        "message #1 must be from-empty (base_version=0)"
    );

    // Emitted sections are EXACTLY the live set {1,2,3,5}; the abandoned prim 99
    // is dropped, with no tombstone and no field for it.
    let emitted_section_ids: std::collections::HashSet<u64> = decoded
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        emitted_section_ids,
        [1u64, 2, 3, 5].into_iter().collect(),
        "message #1 must emit exactly the live sections {{1,2,3,5}} — abandoned prim 99 dropped"
    );
    assert!(
        !decoded
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "faithful snapshot must emit NO tombstones for a from-empty joiner"
    );
    assert!(
        !decoded.set_fields.iter().any(|sf| sf.section_id == 99),
        "the abandoned prim's field must be dropped, not emitted"
    );

    // Emitted section_order is the REAL per-parent sibling index (b.bdx), NOT
    // the stored value — even though history was trimmed. Tree: root children in
    // creation order [R(1), H(5)] → R=0, H=1; R's children [A(2), B(3)] → A=0,
    // B=1. (Abandoned S(99) is not a live child of R.)
    let emitted_section_order = |sid: u64| -> u32 {
        decoded
            .struct_commands
            .iter()
            .find_map(|c| match c {
                StructCommand::DiffChangeSection(s) if s.section_id == sid => Some(s.section_order),
                _ => None,
            })
            .unwrap_or_else(|| panic!("section {sid} must be emitted"))
    };
    assert_eq!(emitted_section_order(1), 0, "R sibling index 0 under root");
    assert_eq!(emitted_section_order(5), 1, "H sibling index 1 under root");
    assert_eq!(emitted_section_order(2), 0, "A sibling index 0 under R");
    assert_eq!(emitted_section_order(3), 1, "B sibling index 1 under R");

    // Emitted SetField.set_order is a DENSE-UNIQUE index ordered by `(version,
    // authoring_set_order)` (b.bdx), NOT the raw authoring rank and NOT the
    // cumulative cursor — even though history was trimmed. R.name (commit 1) → 0;
    // A.posX (commit 2 last-writer) → 1; B.color (commit 2) → 2; H.hb (final
    // heartbeat, highest version) → 3.
    let emitted_set_order = |sid: u64, key: &str| -> u64 {
        decoded
            .set_fields
            .iter()
            .find(|sf| sf.section_id == sid && sf.key_name == key)
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be emitted"))
            .set_order
    };
    assert_eq!(emitted_set_order(1, "name"), 0, "R.name dense set_order");
    assert_eq!(
        emitted_set_order(2, "posX"),
        1,
        "A.posX dense set_order (LAST-writer burst survives, ordered after R.name)"
    );
    assert_eq!(emitted_set_order(3, "color"), 2, "B.color dense set_order");
    assert_eq!(
        emitted_set_order(5, "hb"),
        3,
        "H.hb dense set_order (highest version → last)"
    );
    // Global uniqueness across the whole catch-up (client drops duplicates).
    let all_set_orders: Vec<u64> = decoded.set_fields.iter().map(|sf| sf.set_order).collect();
    let mut uniq_orders = all_set_orders.clone();
    uniq_orders.sort_unstable();
    uniq_orders.dedup();
    assert_eq!(
        uniq_orders.len(),
        all_set_orders.len(),
        "emitted set_order values must be globally unique across the catch-up"
    );
    // Relative order across fields (name < posX < color) is preserved.
    assert!(
        emitted_set_order(1, "name") < emitted_set_order(2, "posX")
            && emitted_set_order(2, "posX") < emitted_set_order(3, "color"),
        "emitted set_order must preserve the (version, authoring) apply order"
    );

    // Cross-check against the authoritative server mirror: A.posX's cumulative
    // cursor differs from the emitted authoring rank, and the abandoned prim is
    // still materialized server-side (an orphan) — the exact divergence the
    // faithful snapshot corrects for the from-empty joiner.
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    let server_posx = server_state
        .fields
        .get(&(2, "posX".to_owned()))
        .expect("server must carry A.posX");
    assert_eq!(
        server_posx.authoring_set_order, 2,
        "server A.posX authoring rank"
    );
    assert_ne!(
        emitted_set_order(2, "posX"),
        server_posx.set_order,
        "emitted rank must NOT be the cumulative cursor the server stores"
    );
    assert!(
        server_state.sections.contains_key(&99),
        "abandoned prim 99 must still exist server-side as an orphan (the divergence the snapshot fixes)"
    );

    // ── Apply message #1 to an EMPTY ObjectState — exactly what a Kit client does
    //    from an empty tree — and assert the reconstructed CURRENT scene. ──
    let mut reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut reconstructed, &body);

    let reconstructed_ids: std::collections::HashSet<u64> =
        reconstructed.sections.keys().copied().collect();
    assert_eq!(
        reconstructed_ids,
        [1u64, 2, 3, 5].into_iter().collect(),
        "reconstructed scene must be exactly the live sections {{1,2,3,5}}"
    );
    assert!(
        !reconstructed.sections.contains_key(&99),
        "abandoned scratch prim 99 must be ABSENT from the reconstructed scene"
    );
    assert!(
        !reconstructed.fields.contains_key(&(99, "temp".to_owned())),
        "the abandoned prim's field must be absent from the reconstructed scene"
    );

    // Structural parity (id/parent/name/type) on the surviving prims. NOTE:
    // `section_order` is deliberately NOT compared against the server — the
    // corrected snapshot emits the REAL per-parent sibling index (b.bdx), which
    // differs from the arbitrary stored value. Sibling order asserted separately.
    for sid in [1u64, 2, 3, 5] {
        let r = reconstructed
            .sections
            .get(&sid)
            .expect("live section reconstructs");
        let s = server_state
            .sections
            .get(&sid)
            .expect("live section on server");
        assert_eq!(r.section_id, s.section_id, "section {sid} id parity");
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }
    // Reconstructed sibling order (b.bdx): root children R=0, H=1; R's children
    // A=0, B=1.
    assert_eq!(
        reconstructed.sections[&1].section_order, 0,
        "R sibling index"
    );
    assert_eq!(
        reconstructed.sections[&5].section_order, 1,
        "H sibling index"
    );
    assert_eq!(
        reconstructed.sections[&2].section_order, 0,
        "A sibling index"
    );
    assert_eq!(
        reconstructed.sections[&3].section_order, 1,
        "B sibling index"
    );

    // Final field-VALUE parity on the surviving prims (last-writer value wins;
    // the heartbeat field carries its final burst value).
    for (sid, key, expected) in [
        (1u64, "name", &b"scene"[..]),
        (2, "posX", &b"final"[..]),
        (3, "color", &b"red"[..]),
        (5, "hb", format!("hb-{}", HEARTBEAT_WRITES - 1).as_bytes()),
    ] {
        let r = reconstructed
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("reconstructed must carry field ({sid},{key})"));
        assert_eq!(
            r.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) must reconstruct its final value on the trimmed stream"
        );
        let s = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("server must carry field ({sid},{key})"));
        assert_eq!(
            r.value, s.value,
            "field ({sid},{key}) value must match the server mirror"
        );
    }

    // The caught-up terminator (total_size==0) is still emitted (SDD §12.3).
    recv_caught_up_ack(&mut stream, object_id, established_seq).await;
}

/// 2. No-wipe after a forced trim: once client B is snapshot-caught-up on a
///    trimmed stream, B's next INCREMENTAL write must leave writer A's prior
///    sections + field values intact (asserted positively on both server state
///    and a present-from-start witness C's reconstructed tree), the relay must
///    not be a reset and must carry no deletion of A's prims, and B's own stream
///    must self-echo-suppress. The trimmed-stream analogue of
///    `snapshot_caught_up_client_write_preserves_prior_state`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_caught_up_after_trim_client_write_preserves_prior_state() {
    let tv = PooledTestValkey::setup().await;
    const MAXLEN: u64 = 4;
    const HEARTBEAT_WRITES: u64 = 200;

    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), MAXLEN);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-trim-nowipe-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-trim-nowipe-session", layer).await;

    // ── Writer A commits a scene of two sections with fields (seq 2 & 3). ──
    let a_commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "a-root".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "a1".to_owned(),
            value_or_ext_hash: b"va1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        100,
        &a_commit1,
    )
    .await;

    let a_commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a-child".to_owned(),
            section_type: 0,
            section_order: 5,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "a2".to_owned(),
            value_or_ext_hash: b"va2".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        200,
        &a_commit2,
    )
    .await;

    // ── Create a reachable heartbeat prim H(5) and hammer its field to force a
    //    real approximate trim BEFORE B and C join. ──
    let a_commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 5,
            section_name: "heartbeat".to_owned(),
            section_type: 0,
            section_order: 8,
        })],
        set_fields: vec![SetField {
            section_id: 5,
            key_name: "hb".to_owned(),
            value_or_ext_hash: b"hb-init".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        300,
        &a_commit3,
    )
    .await;

    for i in 0..HEARTBEAT_WRITES {
        let beat = OwnedDelta {
            is_diff: false,
            set_fields: vec![SetField {
                section_id: 5,
                key_name: "hb".to_owned(),
                value_or_ext_hash: format!("hb-{i}").into_bytes(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            ..Default::default()
        };
        apply_delta_update_returning_ompf(
            &svc,
            &session_id,
            layer,
            object_id,
            "writer-a",
            5_000 + i,
            &beat,
        )
        .await;
    }

    // CreateLayer(1) + A's 3 scene commits + HEARTBEAT_WRITES beats.
    let established_seq = 4 + HEARTBEAT_WRITES;

    // ── Prove the trim fired: XLEN bounded, non-zero, entry `2-0` (A's first
    //    commit) evicted. ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let xlen: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    let total_writes = 3 + HEARTBEAT_WRITES;
    assert!(
        xlen < total_writes,
        "approximate MAXLEN trim did not fire: XLEN={xlen} >= total_writes={total_writes}"
    );
    assert!(
        xlen > 0,
        "stream must retain at least one entry (got XLEN=0)"
    );
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "entry 2-0 must have been trimmed — got {probe:?}"
    );

    // ── Present-from-start witness C: subscribe BEFORE B's write, capture its
    //    snapshot catch-up body, reconstruct C's local tree. ──
    let c_req = subscribe_request(&session_id, layer, 0, 0, "witness-c");
    let c_response = handle_subscribe_read_layer(&svc, c_req)
        .await
        .expect("witness subscribe must succeed");
    let mut c_stream = c_response.into_inner();
    let (c_header, c_body) = recv_message(&mut c_stream).await;
    assert_eq!(c_header.object_id, object_id);
    assert_eq!(
        c_header.sequence_num, established_seq,
        "witness catch-up delivers the current scene at the established seq"
    );
    let mut c_state = ObjectState::default();
    apply_ompf_to_state(&mut c_state, &c_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "witness catch-up must reconstruct A's scene (sections 1 and 2) despite the trim"
    );
    recv_caught_up_ack(&mut c_stream, object_id, established_seq).await;

    // ── Client B late-joins on the trimmed stream, drains its snapshot catch-up
    //    + caught-up ack. ──
    let b_req = subscribe_request(&session_id, layer, 0, 0, "client-b");
    let b_response = handle_subscribe_read_layer(&svc, b_req)
        .await
        .expect("client-b subscribe must succeed");
    let mut b_stream = b_response.into_inner();
    let (b_header, _b_body) = recv_message(&mut b_stream).await;
    assert_eq!(b_header.object_id, object_id);
    assert_eq!(b_header.sequence_num, established_seq);
    recv_caught_up_ack(&mut b_stream, object_id, established_seq).await;

    // ── B, now caught-up-after-trim, issues its next INCREMENTAL (non-reset)
    //    write adding its own section + field. ──
    let b_commit = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 3,
            section_name: "b-root".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "b1".to_owned(),
            value_or_ext_hash: b"vb1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "client-b",
        300_000,
        &b_commit,
    )
    .await;

    // ── POSITIVE property #1: server state after B's write still contains A's
    //    sections + field values intact, PLUS B's new section. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    for (sid, key, expected) in [
        (1u64, "a1", &b"va1"[..]),
        (2, "a2", &b"va2"[..]),
        (3, "b1", &b"vb1"[..]),
    ] {
        assert!(
            server_state.sections.contains_key(&sid),
            "section {sid} must survive/exist after B's write on the trimmed stream"
        );
        let f = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be present after B's write"));
        assert_eq!(
            f.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) value must be intact after B's write (no wipe)"
        );
    }

    // ── POSITIVE property #2: the witness receives B's per-commit relay as a
    //    single message at the next seq; it is NOT a reset and carries no deletion
    //    of A's prims; applying it keeps A's sections + values and adds B's. ──
    let (relay_header, relay_body) = recv_message(&mut c_stream).await;
    assert_eq!(
        relay_header.object_id, object_id,
        "relay object_id must be stable (no overwrite)"
    );
    assert_eq!(
        relay_header.sequence_num,
        established_seq + 1,
        "witness must observe B's write as a single per-commit relay at the next seq"
    );
    let relay_delta = decode_broadcast_delta(&relay_body);
    assert!(
        !(relay_delta.is_diff && relay_delta.base_version == 0),
        "B's relayed per-commit write must NOT be a reset (would wipe A's prims)"
    );
    assert!(
        !relay_delta
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "B's relayed write must carry no section deletions of A's prims"
    );

    apply_ompf_to_state(&mut c_state, &relay_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "A's sections must survive in the witness's tree after B's relayed write"
    );
    assert!(
        c_state.sections.contains_key(&3),
        "B's new section must be added to the witness's tree"
    );
    assert_eq!(
        c_state.fields.get(&(1, "a1".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"va1".to_vec()
        )),
        "A's field value must survive in the witness's tree (no wipe)"
    );

    // ── POSITIVE property #3: B's own stream self-echo-suppresses its write. ──
    assert_no_next_message(&mut b_stream, Duration::from_millis(300)).await;
}

/// 3. Retention-independence encoded as a checked property: two fresh
///    Snapshot-mode late-joins — one with full stream history, one after the
///    stream has been forced to near-minimum retention — must reconstruct
///    byte-for-byte identical scenes (same emitted section set, section metadata,
///    final field values, and per-field authoring ranks). This holds because
///    `compute_faithful_full_content_at_cut` reconstructs from MATERIALIZED state +
///    persisted `authoring_set_order` and NEVER reads the stream, so the surviving
///    stream length cannot influence the snapshot output.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_late_join_is_independent_of_surviving_stream_length() {
    // Comparable projection of a snapshot catch-up body, derived purely from the
    // emitted delta + its empty-tree reconstruction — no stream state involved.
    #[derive(Debug, PartialEq)]
    struct CapturedScene {
        emitted_sections: std::collections::BTreeSet<u64>,
        section_meta: std::collections::BTreeMap<u64, (u64, String, u32, u8)>,
        field_values:
            std::collections::BTreeMap<(u64, String), live_edit::domain::types::FieldValue>,
        authoring_ranks: std::collections::BTreeMap<(u64, String), u64>,
    }
    fn capture(body: &[u8]) -> CapturedScene {
        let decoded = decode_broadcast_delta(body);
        let mut reconstructed = ObjectState::default();
        apply_ompf_to_state(&mut reconstructed, body);
        let emitted_sections = decoded
            .struct_commands
            .iter()
            .filter_map(|c| match c {
                StructCommand::DiffChangeSection(s) => Some(s.section_id),
                _ => None,
            })
            .collect();
        let section_meta = reconstructed
            .sections
            .values()
            .map(|s| {
                (
                    s.section_id,
                    (s.parent_id, s.name.clone(), s.section_order, s.section_type),
                )
            })
            .collect();
        let field_values = reconstructed
            .fields
            .iter()
            .map(|((sid, key), f)| ((*sid, key.clone()), f.value.clone()))
            .collect();
        let authoring_ranks = decoded
            .set_fields
            .iter()
            .map(|sf| ((sf.section_id, sf.key_name.clone()), sf.set_order))
            .collect();
        CapturedScene {
            emitted_sections,
            section_meta,
            field_values,
            authoring_ranks,
        }
    }

    let tv = PooledTestValkey::setup().await;
    // Roomy MAXLEN so the scene commits are all retained for the FIRST late-join;
    // an explicit XTRIM later forces the SECOND join onto a minimal stream.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-retention-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-retention-session", layer).await;

    // ── Multi-commit scene: root R(1)+child A(2), burst A.posX + child B(3), a
    //    scratch prim S(99), then abandon it (materialized orphan). ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;

    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"final".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;

    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "temp".to_owned(),
            value_or_ext_hash: b"junk".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 3,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        400,
        &commit4,
    )
    .await;

    const ESTABLISHED_SEQ: u64 = 5; // CreateLayer(1) + 4 commits.

    // ── FIRST fresh late-join, with FULL stream history retained. ──
    let sub1 = subscribe_request(&session_id, layer, 0, 0, "late-joiner-1");
    let response1 = handle_subscribe_read_layer(&svc, sub1)
        .await
        .expect("subscribe #1 must succeed");
    let mut stream1 = response1.into_inner();
    let (header1, body1) = recv_message(&mut stream1).await;
    assert_eq!(header1.object_id, object_id);
    assert_eq!(
        header1.sequence_num, ESTABLISHED_SEQ,
        "join #1 delivers the established scene"
    );
    let scene1 = capture(&body1);
    // Sanity: the abandoned prim is dropped even with full history present.
    assert_eq!(
        scene1.emitted_sections,
        [1u64, 2, 3].into_iter().collect(),
        "join #1 must emit exactly the live sections {{1,2,3}} (abandoned 99 dropped)"
    );
    recv_caught_up_ack(&mut stream1, object_id, ESTABLISHED_SEQ).await;

    // ── Force the stream to near-minimum retention via explicit XTRIM MINID so
    //    only the final entry survives. This mutates ONLY the stream — never the
    //    materialized state the faithful builder reads. ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", "5-0")).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to near-minimum retention must succeed");

    // Verify the historical update entries are gone: XLEN collapsed and the
    // earliest UpdateLayer commit (entry `2-0`) evicted.
    let xlen_after: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert!(
        xlen_after < ESTABLISHED_SEQ,
        "XTRIM must have collapsed the stream well below the {ESTABLISHED_SEQ} historical entries, got XLEN={xlen_after}"
    );
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "entry 2-0 must be gone after XTRIM — got {probe:?}"
    );

    // ── SECOND fresh late-join, now on the near-empty stream. ──
    let sub2 = subscribe_request(&session_id, layer, 0, 0, "late-joiner-2");
    let response2 = handle_subscribe_read_layer(&svc, sub2)
        .await
        .expect("subscribe #2 must succeed");
    let mut stream2 = response2.into_inner();
    let (header2, body2) = recv_message(&mut stream2).await;
    assert_eq!(header2.object_id, object_id);
    assert_eq!(
        header2.sequence_num, ESTABLISHED_SEQ,
        "join #2 must deliver the SAME established seq despite the trimmed stream"
    );
    let scene2 = capture(&body2);
    recv_caught_up_ack(&mut stream2, object_id, ESTABLISHED_SEQ).await;

    // ── The retention-independence property: the faithful snapshot output is
    //    identical regardless of how much stream history survived. ──
    assert_eq!(
        header1.sequence_num, header2.sequence_num,
        "emitted sequence_num must be retention-independent"
    );
    assert_eq!(
        scene1, scene2,
        "faithful snapshot late-join must reconstruct an IDENTICAL scene (section set, \
         section metadata, final field values, and per-field authoring ranks) whether the \
         stream retained full history or was trimmed to near-minimum — because \
         compute_faithful_full_content_at_cut never reads the stream"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// t3.msb.js.qb.33 — corrected-snapshot FAITHFUL COALESCED catch-up for a GENUINE
// out-of-window RECONNECTOR (Regime 3, non-zero stale `have_version`).
//
// The trimmed-stream snapshot tests above (`snapshot_late_join_on_trimmed_stream_*`)
// all late-join FRESH (object_id=0, seq=0 → brand-new / `compute_faithful_full_content_at_cut`).
// These three exercise the OTHER Snapshot-mode entrypoint — `compute_faithful_coalesced_at_cut`
// — reached ONLY by an advanced reconnector whose stale `have_version` is > 0 and whose
// stream slice has been evicted (small MAXLEN + explicit `XTRIM MINID`), so it provably
// lands in Regime 3 (`catch_up_out_of_window`), NOT in_window and NOT brand_new. Unlike
// the from-empty snapshot, the coalesced envelope keeps `base_version = have_version`
// (never 0) and RETRACTS ghosts the reconnector still holds via `DiffDeleteSection`
// (the UNIFIED set: tombstoned deletes AND abandoned-to-unreachable prims).
//
// Every stream await is bounded by `recv_message` / `SUBSCRIBER_DEADLINE`; no
// `proptest!` macros are added here.
// ═══════════════════════════════════════════════════════════════════════════════

/// The out-of-window reconnector's stale version. Both the soon-to-be-abandoned
/// prim and the soon-to-be-deleted section are created at/before this version;
/// the abandon + delete happen strictly after it, so the reconnector still holds
/// both when it reconnects and both must be retracted.
const OOW_HAVE_VERSION: u64 = 3;
/// Server sequence after the shared scene's five commits (CreateLayer(1) + 5).
const OOW_CURRENT_SEQ: u64 = 6;
/// Section explicitly deleted (tombstoned) after `OOW_HAVE_VERSION` — ghost leg (a).
const OOW_TOMBSTONE_SECTION: u64 = 50;
/// Scratch prim reparented under `ABANDON_ORPHAN_PARENT` (net-live-unreachable)
/// after `OOW_HAVE_VERSION` — ghost leg (b).
const OOW_ABANDONED_PRIM: u64 = 99;

/// Root-reachable (net-live) section ids — BFS from the sentinel parent_id=0 via
/// the `children` map, matching `DiffEngine::reachable_sections_in_dfs_order`.
/// Prunes the server's materialized scene (which still carries the abandoned
/// orphan) down to the net-live set the faithful coalesced catch-up targets.
fn reachable_section_ids(state: &ObjectState) -> std::collections::HashSet<u64> {
    let mut reachable = std::collections::HashSet::new();
    let mut stack: Vec<u64> = state.children.get(&0).cloned().unwrap_or_default();
    while let Some(id) = stack.pop() {
        if state.sections.contains_key(&id) && reachable.insert(id) {
            if let Some(kids) = state.children.get(&id) {
                stack.extend(kids.iter().copied());
            }
        }
    }
    reachable
}

/// Cascade-remove a section subtree (the section, its descendants, and their
/// fields / time samples) from a reconstructed `ObjectState`.
fn cascade_remove_section(state: &mut ObjectState, root_id: u64) {
    let mut subtree = vec![root_id];
    let mut i = 0;
    while i < subtree.len() {
        if let Some(kids) = state.children.get(&subtree[i]) {
            let kids = kids.clone();
            subtree.extend(kids);
        }
        i += 1;
    }
    let removal: std::collections::HashSet<u64> = subtree.into_iter().collect();
    for id in &removal {
        state.sections.remove(id);
        state.children.remove(id);
    }
    for kids in state.children.values_mut() {
        kids.retain(|k| !removal.contains(k));
    }
    state.fields.retain(|(sid, _), _| !removal.contains(sid));
    state
        .time_samples
        .retain(|(sid, _), _| !removal.contains(sid));
}

/// Reconstruct a subscriber's scene from a delivered catch-up body, honoring the
/// READ-SIDE semantics of `DiffDeleteSection`. The server INGEST engine
/// (`apply_delta`, used by `apply_ompf_to_state`) silently IGNORES
/// `DiffDeleteSection` (SDD §9.1.4f — a writer deletes via `DeleteSection`, not
/// `DiffDeleteSection`), but a Kit client applying a CATCH-UP diff DOES retract
/// the section from its stage. This helper emulates the client: apply the
/// change/field content via `apply_ompf_to_state`, then cascade-remove every
/// `DiffDeleteSection` target so the reconstruction reflects the retraction.
fn apply_catchup_read_side(state: &mut ObjectState, ompf_bytes: &[u8]) {
    let decoded = decode_broadcast_delta(ompf_bytes);
    let delete_ids: Vec<u64> = decoded
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffDeleteSection(d) => Some(d.section_id),
            _ => None,
        })
        .collect();
    apply_ompf_to_state(state, ompf_bytes);
    for id in delete_ids {
        cascade_remove_section(state, id);
    }
}

/// Section-ids emitted as live content (`DiffChangeSection`) in a catch-up body.
fn emitted_change_section_ids(delta: &OwnedDelta) -> std::collections::HashSet<u64> {
    delta
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect()
}

/// Section-ids a catch-up body names as PRESENT: the DISTINCT ids carried by
/// any NON-delete struct command. Deliberately the same reduction the
/// internal acceptance harness calls `present` (the ids of every struct command
/// that is not a delete), so a
/// count taken here and the count the harness prints for the same payload are
/// directly comparable rather than two similar-looking numbers.
///
/// Wider than `emitted_change_section_ids`, which sees only
/// `DiffChangeSection`: today's coalesced builder emits nothing else on the
/// live-content side, but a builder that started naming sections with
/// `CreateSection` / `MoveSection` / `ReorderChildren` would still be counted
/// here — matching Python, which keys off "not a tombstone" rather than off one
/// command type. The match is exhaustive so a new `StructCommand` variant forces
/// this decision rather than silently landing on one side.
fn emitted_present_section_ids(delta: &OwnedDelta) -> std::collections::HashSet<u64> {
    delta
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::CreateSection(s) => Some(s.section_id),
            StructCommand::MoveSection(s) => Some(s.section_id),
            StructCommand::ReorderChildren(s) => Some(s.section_id),
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            StructCommand::DeleteSection(_) | StructCommand::DiffDeleteSection(_) => None,
        })
        .collect()
}

/// Section-ids retracted (`DiffDeleteSection`) in a catch-up body.
fn emitted_delete_section_ids(delta: &OwnedDelta) -> std::collections::HashSet<u64> {
    delta
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffDeleteSection(d) => Some(d.section_id),
            _ => None,
        })
        .collect()
}

/// Build the shared t3.msb.js.qb.33 editing history on `svc` and force a GENUINE
/// out-of-window trim. Returns `(session_id, object_id, mirror)` where `mirror`
/// is the reconnector's locally-reconstructed `ObjectState` at `OOW_HAVE_VERSION`
/// — holding BOTH the soon-to-be-abandoned prim `OOW_ABANDONED_PRIM` (under root)
/// AND the soon-to-be-deleted section `OOW_TOMBSTONE_SECTION` (under root), plus a
/// burst field (`childA.posX`) whose stored cumulative cursor diverges from its
/// authoring rank and whose value the final burst overwrites. Independent of the
/// `catchup_snapshot_enabled` mode (that only affects the later subscribe), so tests 1
/// and 3 share it.
async fn build_oow_reconnect_scene(
    tv: &PooledTestValkey,
    svc: &LiveEditService,
    name_prefix: &str,
    layer: &str,
) -> (String, u64, ObjectState) {
    let (session_id, object_id) =
        create_session_and_layer_pooled(tv, svc, &format!("{name_prefix}-session"), layer).await;

    // Reconnector mirror — the empty-create (seq=1) plus the two pre-window
    // commits it applied before disconnecting.
    let mut mirror = ObjectState::default();
    apply_delta(&mut mirror, None).expect("mirror empty create (seq=1)");

    // ── Commit 1 (seq=2): root R(1)+name; child A(2)+posX("first", so=5); scratch
    //    SC(99)+temp under root; deletable DL(50)+dlf under root. ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: OOW_ABANDONED_PRIM,
                section_name: "scratch".to_owned(),
                section_type: 0,
                section_order: 30,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: OOW_TOMBSTONE_SECTION,
                section_name: "doomed".to_owned(),
                section_type: 0,
                section_order: 40,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
            SetField {
                section_id: OOW_ABANDONED_PRIM,
                key_name: "temp".to_owned(),
                value_or_ext_hash: b"junk".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 3,
            },
            SetField {
                section_id: OOW_TOMBSTONE_SECTION,
                key_name: "dlf".to_owned(),
                value_or_ext_hash: b"doomed-v".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 4,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;
    apply_delta(&mut mirror, Some(commit1.clone())).expect("mirror commit1 (seq=2)");

    // ── Commit 2 (seq=3): child B(3)+color; BURST-update A.posX="mid" (authoring
    //    rank 2) so its stored cumulative cursor now diverges from the rank. This
    //    is the reconnector's last applied version. ──
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"mid".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;
    apply_delta(&mut mirror, Some(commit2.clone())).expect("mirror commit2 (seq=3)");
    assert_eq!(
        mirror.sequence_num, OOW_HAVE_VERSION,
        "reconnector mirror must be at have_version after the two pre-window commits"
    );

    // Everything below happens WHILE the reconnector is disconnected — its mirror
    // is deliberately NOT advanced past OOW_HAVE_VERSION.

    // ── Commit 3 (seq=4): ABANDON scratch SC(99) — reparent under a non-existent
    //    parent so it becomes unreachable from root WITHOUT deleting it (stays
    //    materialized as an orphan). Ghost-retraction leg (b). ──
    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: OOW_ABANDONED_PRIM,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    // ── Commit 4 (seq=5): DELETE DL(50) — writer-side DeleteSection, which writes
    //    a tombstone and drops the section from the live index. Ghost-retraction
    //    leg (a). ──
    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DeleteSection(
            oodf_ffi::CommandDeleteSection {
                parent_id: 1,
                section_id: OOW_TOMBSTONE_SECTION,
            },
        )],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        400,
        &commit4,
    )
    .await;

    // ── Commit 5 (seq=6): BURST-update A.posX="final" (last-writer wins) AND
    //    STRUCTURALLY re-assert childA(2) under root R(1) via DiffChangeSection,
    //    keeping its DIVERGENT stored section_order (10). Re-asserting the section
    //    bumps its `last_version` to 6 (> have_version=3), so the coalesced
    //    catch-up re-emits childA as a modified section and, in snapshot mode,
    //    restates its unchanged sibling childB(3) with it (R now has a changed
    //    child) — the only two live sections re-emitted in this scene. That
    //    makes the reconnect exercise the b.bdx coalesced sibling-order
    //    override: A's stored order stays 10 but its real per-parent sibling
    //    index under R is 0 (first live child), so the emitted `section_order`
    //    MUST be 0, not 10 (asserted by the Snapshot establish test; the
    //    Off-mode regression pin keeps 10 because the legacy path emits the
    //    stored order verbatim). ──
    let commit5 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "childA".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "posX".to_owned(),
            value_or_ext_hash: b"final".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        500,
        &commit5,
    )
    .await;

    // ── Force a GENUINE out-of-window reconnect: trim the stream so the
    //    reconnector's start id ({have_version+1}-0 = "4-0") is not present, so
    //    the reconnect XRANGE slice misses and falls into Regime 3. ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", format!("{OOW_CURRENT_SEQ}-0").as_str()))
        .expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(
            &stream_key,
            format!("{}-0", OOW_HAVE_VERSION + 1),
            format!("{}-0", OOW_HAVE_VERSION + 1),
            Some(1),
        )
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "reconnector's start id {}-0 must be trimmed so the reconnect lands in Regime 3 — got {probe:?}",
        OOW_HAVE_VERSION + 1
    );

    (session_id, object_id, mirror)
}

/// 1. Snapshot mode: a GENUINE out-of-window reconnector receives a FAITHFUL
///    coalesced catch-up — `base_version = have_version` (never 0), the current
///    server seq in the header, exactly-once caught-up ack — that RETRACTS BOTH
///    ghost legs (tombstoned DL(50) AND abandoned-unreachable SC(99)) via
///    `DiffDeleteSection`, so the reconnector's scene equals the server net-live
///    scene: neither ghost survives and the burst field carries its final value.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_out_of_window_reconnect_establishes_correct_scene() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-oow-establish-layer";
    let (session_id, object_id, mut mirror) =
        build_oow_reconnect_scene(&tv, &svc, "snap-oow-establish", layer).await;

    // ── Server-side sanity: the abandoned prim is still materialized as an orphan;
    //    the deleted section is gone from the live index but present as a tombstone. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    assert!(
        server_state.sections.contains_key(&OOW_ABANDONED_PRIM),
        "abandoned prim {OOW_ABANDONED_PRIM} must still exist server-side as an orphan"
    );
    assert!(
        !server_state.sections.contains_key(&OOW_TOMBSTONE_SECTION),
        "tombstoned section {OOW_TOMBSTONE_SECTION} must be gone from the live index"
    );
    assert!(
        snap.tombstones
            .iter()
            .any(|(sid, ..)| *sid == OOW_TOMBSTONE_SECTION),
        "server must carry a tombstone for {OOW_TOMBSTONE_SECTION}; got {:?}",
        snap.tombstones
    );

    // ── Reconnect presenting the stale have_version on the SAME object_id. ──
    let sub_req = subscribe_request(
        &session_id,
        layer,
        object_id,
        OOW_HAVE_VERSION,
        "reconnector",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, OOW_CURRENT_SEQ,
        "coalesced catch-up header must carry the current server seq"
    );
    assert!(
        !body.is_empty(),
        "message #1 must carry a non-empty coalesced catch-up body"
    );

    // ── Envelope: incremental (is_diff=true) with base_version == have_version —
    //    NEVER a base_version=0 from-empty reset the reconnector would reject. ──
    let decoded = decode_broadcast_delta(&body);
    assert!(decoded.is_diff, "coalesced catch-up must be is_diff=true");
    assert_eq!(
        decoded.base_version, OOW_HAVE_VERSION,
        "coalesced catch-up base_version must equal have_version (never 0)"
    );
    assert_ne!(
        decoded.base_version, 0,
        "an advanced reconnector must NEVER receive a base_version=0 reset"
    );

    // ── set_order contract on the OUT-OF-WINDOW dispatch. It shares the
    //    corrected coalesced builder with the in-window reconnect path, so the
    //    dense-unique emission must hold here too — asserted end-to-end rather
    //    than assumed from the shared call site. ──
    assert_dense_unique_set_orders(&decoded, "out-of-window reconnect");

    // ── Unified ghost retraction: BOTH the tombstoned section AND the
    //    abandoned-to-unreachable prim are retracted via DiffDeleteSection. The
    //    tombstone leg must not silently drop. ──
    let retracted = emitted_delete_section_ids(&decoded);
    assert!(
        retracted.contains(&OOW_TOMBSTONE_SECTION),
        "catch-up must retract the tombstoned section {OOW_TOMBSTONE_SECTION} (leg a); retracted={retracted:?}"
    );
    assert!(
        retracted.contains(&OOW_ABANDONED_PRIM),
        "catch-up must retract the abandoned-unreachable prim {OOW_ABANDONED_PRIM} (leg b); retracted={retracted:?}"
    );
    // Neither ghost is (re)emitted as live content, nor are their fields.
    let live_emitted = emitted_change_section_ids(&decoded);
    assert!(
        !live_emitted.contains(&OOW_ABANDONED_PRIM)
            && !live_emitted.contains(&OOW_TOMBSTONE_SECTION),
        "no ghost may be emitted as a live DiffChangeSection; got {live_emitted:?}"
    );
    assert!(
        !decoded
            .set_fields
            .iter()
            .any(|sf| sf.section_id == OOW_ABANDONED_PRIM || sf.section_id == OOW_TOMBSTONE_SECTION),
        "no ghost's field may be emitted as live content"
    );
    // The burst field's final value is carried.
    let posx = decoded
        .set_fields
        .iter()
        .find(|sf| sf.section_id == 2 && sf.key_name == "posX")
        .expect("catch-up must carry the changed burst field childA.posX");
    assert_eq!(
        posx.value_or_ext_hash, b"final",
        "catch-up must carry the LAST-writer value of the burst field"
    );

    // ── Exactly-once caught-up ack, carrying the current seq. ──
    recv_caught_up_ack(&mut stream, object_id, OOW_CURRENT_SEQ).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // ── Reconstruct the reconnector's scene from its pre-reconnect state + the
    //    delivered catch-up (honoring the read-side DiffDeleteSection retraction),
    //    and assert it equals the server's net-live materialized scene. ──
    apply_catchup_read_side(&mut mirror, &body);

    let reachable = reachable_section_ids(&server_state);
    let reconstructed_ids: std::collections::HashSet<u64> =
        mirror.sections.keys().copied().collect();
    assert_eq!(
        reconstructed_ids, reachable,
        "reconstructed scene must equal the server's net-live (root-reachable) section set"
    );
    assert!(
        !mirror.sections.contains_key(&OOW_ABANDONED_PRIM),
        "abandoned prim {OOW_ABANDONED_PRIM} must be ABSENT from the reconnector's scene (no stale ghost)"
    );
    assert!(
        !mirror.sections.contains_key(&OOW_TOMBSTONE_SECTION),
        "tombstoned section {OOW_TOMBSTONE_SECTION} must be ABSENT from the reconnector's scene (no stale ghost)"
    );
    assert!(
        !mirror
            .fields
            .contains_key(&(OOW_ABANDONED_PRIM, "temp".to_owned()))
            && !mirror
                .fields
                .contains_key(&(OOW_TOMBSTONE_SECTION, "dlf".to_owned())),
        "no ghost's field may survive in the reconnector's scene"
    );

    // Structural parity (parent/name/type) on every net-live section. section_order
    // is asserted SEPARATELY below against the REAL sibling index — deliberately
    // NOT compared to the server's STORED order, which for childA(2) diverges
    // (stored 10 vs sibling index 0). A reconstructed-vs-stored comparison would
    // re-encode the superseded pre-b.bdx contract and pass vacuously.
    for sid in &reachable {
        let r = mirror.sections.get(sid).expect("live section reconstructs");
        let s = server_state
            .sections
            .get(sid)
            .expect("live section on server");
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }

    // ── b.bdx coalesced sibling-order override: the burst section childA(2) is
    //    re-emitted (commit 5 re-asserted it structurally), so the faithful
    //    coalesced catch-up carries its REAL per-parent sibling index — 0, the
    //    first live child of root R(1) — in `section_order`, NOT the divergent
    //    STORED order the server persists (10). Assert the on-the-wire emitted
    //    value AND the reconstructed value are the sibling index, and that both
    //    diverge from the server-stored order. FAILS if the coalesced
    //    sibling-position override is reverted (would emit the stored 10). ──
    assert_eq!(
        server_state.sections[&2].section_order, 10,
        "server must persist childA's divergent STORED authoring order (10)"
    );
    let emitted_childa_order = decoded
        .struct_commands
        .iter()
        .find_map(|c| match c {
            StructCommand::DiffChangeSection(s) if s.section_id == 2 => Some(s.section_order),
            _ => None,
        })
        .expect("childA(2) must be re-emitted as a live DiffChangeSection in the catch-up");
    assert_eq!(
        emitted_childa_order, 0,
        "emitted childA section_order must be the sibling index 0, not the stored order 10"
    );
    assert_eq!(
        mirror.sections[&2].section_order, 0,
        "reconstructed childA section_order must be the sibling index 0 (not the stored 10) — \
         proving the coalesced catch-up delivered the real sibling order (b.bdx)"
    );
    for ((sid, key), s_field) in &server_state.fields {
        if !reachable.contains(sid) {
            continue;
        }
        let r_field = mirror
            .fields
            .get(&(*sid, key.clone()))
            .unwrap_or_else(|| panic!("reconstructed must carry net-live field ({sid},{key})"));
        assert_eq!(
            r_field.value, s_field.value,
            "field ({sid},{key}) value must match the server (last-writer-wins)"
        );
    }
    // The burst field specifically reconstructs its final value.
    assert_eq!(
        mirror.fields.get(&(2, "posX".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"final".to_vec()
        )),
        "childA.posX must reconstruct its final burst value"
    );
}

/// b.bdx (test-review path a): a Snapshot-mode COALESCED (out-of-window reconnect)
/// catch-up must emit each RE-EMITTED live section's REAL per-parent sibling index
/// in `section_order`, NOT the divergent STORED order — the coalesced analogue of
/// the from-empty `snapshot_late_join_catch_up_establishes_correct_scene` coverage,
/// exercising `plan_faithful_coalesced`'s sibling-position override directly.
///
/// Scene: a reconnector that saw only the root parent P(1) disconnects; while it is
/// away three NEW xformOp children are authored under P with NON-MONOTONIC stored
/// section_orders (translate=30, rotate=10, scale=20). Because
/// `insert_child_into_parent` orders siblings by `section_order`, P's children list
/// is [rotate(10), scale(20), translate(30)] → sibling indices rotate=0, scale=1,
/// translate=2, which DIVERGE from every stored order. A trailing field commit
/// advances the seq so trimming evicts the reconnector's start id and the reconnect
/// lands out-of-window (Regime 3 → `plan_faithful_coalesced`). The coalesced
/// catch-up re-emits the three children (new since have_version) and MUST carry the
/// sibling INDEX (0/1/2 by rank), not the stored order — proven both on the wire and
/// in the reconstructed reconnector scene. FAILS if the coalesced sibling-position
/// override is reverted (would emit 30/10/20).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_out_of_window_reconnect_reemits_real_sibling_order() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-oow-sibling-order-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-oow-sibling-order-session", layer).await;

    const HAVE_VERSION: u64 = 2;
    const CURRENT_SEQ: u64 = 4;

    // Reconnector mirror: empty create (seq=1) + commit1 (seq=2) creating root P(1).
    let mut mirror = ObjectState::default();
    apply_delta(&mut mirror, None).expect("mirror empty create (seq=1)");

    // ── Commit 1 (seq=2): root P(1) + name. The reconnector's last applied version. ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "xform".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "name".to_owned(),
            value_or_ext_hash: b"xform".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;
    apply_delta(&mut mirror, Some(commit1.clone())).expect("mirror commit1 (seq=2)");
    assert_eq!(
        mirror.sequence_num, HAVE_VERSION,
        "reconnector mirror must be at have_version after commit1"
    );

    // Everything below happens WHILE the reconnector is disconnected.

    // ── Commit 2 (seq=3): create three NEW xformOp children of P(1) with
    //    NON-MONOTONIC stored section_orders. Authored order [translate, rotate,
    //    scale] but stored orders 30/10/20, so P's section_order-sorted children
    //    list is [rotate(10), scale(20), translate(30)] → sibling indices
    //    rotate=0, scale=1, translate=2 — divergent from every stored order. ──
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 10,
                section_name: "xformOp:translate".to_owned(),
                section_type: 0,
                section_order: 30,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 11,
                section_name: "xformOp:rotateXYZ".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 12,
                section_name: "xformOp:scale".to_owned(),
                section_type: 0,
                section_order: 20,
            }),
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;

    // ── Commit 3 (seq=4): a trailing field write on P so the current seq advances
    //    past have_version+1, giving the trim below an entry to evict. Field-only,
    //    so it does NOT re-emit any section structurally. ──
    let commit3 = OwnedDelta {
        is_diff: false,
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "purpose".to_owned(),
            value_or_ext_hash: b"rig".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    // ── Force a GENUINE out-of-window reconnect: trim so the reconnector's start id
    //    ({have_version+1}-0 = "3-0") is evicted → Regime 3 → plan_faithful_coalesced.
    //    The materialized state still holds the three children (reconstructed, not
    //    stream-replayed). ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", format!("{CURRENT_SEQ}-0").as_str()))
        .expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(
            &stream_key,
            format!("{}-0", HAVE_VERSION + 1),
            format!("{}-0", HAVE_VERSION + 1),
            Some(1),
        )
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "reconnector's start id {}-0 must be trimmed so the reconnect lands in Regime 3 — got {probe:?}",
        HAVE_VERSION + 1
    );

    // ── Reconnect presenting the stale have_version on the SAME object_id. ──
    let sub_req = subscribe_request(&session_id, layer, object_id, HAVE_VERSION, "reconnector");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, CURRENT_SEQ,
        "coalesced catch-up header must carry the current server seq"
    );

    let decoded = decode_broadcast_delta(&body);
    assert!(decoded.is_diff, "coalesced catch-up must be is_diff=true");
    assert_eq!(
        decoded.base_version, HAVE_VERSION,
        "coalesced catch-up base_version must equal have_version (never 0)"
    );

    // ── On the wire: each re-emitted child carries the sibling INDEX (rank), NOT
    //    the stored order (30/10/20). ──
    let emitted_order = |sid: u64| -> u32 {
        decoded
            .struct_commands
            .iter()
            .find_map(|c| match c {
                StructCommand::DiffChangeSection(s) if s.section_id == sid => Some(s.section_order),
                _ => None,
            })
            .unwrap_or_else(|| panic!("child {sid} must be re-emitted as a live DiffChangeSection"))
    };
    assert_eq!(
        emitted_order(11),
        0,
        "rotateXYZ (stored 10) is sibling index 0"
    );
    assert_eq!(emitted_order(12), 1, "scale (stored 20) is sibling index 1");
    assert_eq!(
        emitted_order(10),
        2,
        "translate (stored 30) is sibling index 2"
    );
    // Guard against a vacuous pass: the emitted value must NEVER be the stored order.
    for (sid, stored) in [(10u64, 30u32), (11, 10), (12, 20)] {
        assert_ne!(
            emitted_order(sid),
            stored,
            "child {sid} emitted section_order must be the sibling index, not the stored order {stored}"
        );
    }

    recv_caught_up_ack(&mut stream, object_id, CURRENT_SEQ).await;

    // ── Reconstruct the reconnector's scene; the children carry the sibling index. ──
    apply_catchup_read_side(&mut mirror, &body);
    assert_eq!(
        mirror.sections[&11].section_order, 0,
        "reconstructed rotateXYZ must carry sibling index 0"
    );
    assert_eq!(
        mirror.sections[&12].section_order, 1,
        "reconstructed scale must carry sibling index 1"
    );
    assert_eq!(
        mirror.sections[&10].section_order, 2,
        "reconstructed translate must carry sibling index 2"
    );
    // The server persists the divergent STORED orders unchanged (emit-time-only fix).
    let server_snap =
        read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
            .await
            .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&server_snap);
    assert_eq!(server_state.sections[&10].section_order, 30);
    assert_eq!(server_state.sections[&11].section_order, 10);
    assert_eq!(server_state.sections[&12].section_order, 20);
}

/// 2. Snapshot mode, SR-3.1 no-wipe: after a GENUINE out-of-window reconnector has
///    fully caught up, its next incremental write must leave the prior writer's
///    sections + field values intact (asserted positively on server state and on a
///    present-from-start witness's reconstructed tree); the relay is not a reset
///    and carries no deletion of the prior prims; the reconnector self-echo-suppresses.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_out_of_window_reconnect_caught_up_client_write_preserves_prior_state() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-oow-nowipe-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-oow-nowipe-session", layer).await;

    // ── Writer A commits a scene of two sections with fields (seq 2 & 3). ──
    let a_commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "a-root".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "a1".to_owned(),
            value_or_ext_hash: b"va1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        100,
        &a_commit1,
    )
    .await;

    let a_commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a-child".to_owned(),
            section_type: 0,
            section_order: 5,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "a2".to_owned(),
            value_or_ext_hash: b"va2".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        200,
        &a_commit2,
    )
    .await;

    const A_ESTABLISHED_SEQ: u64 = 3; // CreateLayer(1) + A's two commits.

    // ── Force the out-of-window regime for reconnector B: B's stale have_version=1
    //    (it only saw the empty create); trim so B's start id "2-0" is evicted. ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let cap: XCap = XCap::try_from(("MINID", "=", format!("{A_ESTABLISHED_SEQ}-0").as_str()))
        .expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.is_empty(),
        "reconnector B's start id 2-0 must be trimmed so B lands in Regime 3 — got {probe:?}"
    );

    // ── Present-from-start witness C: subscribe BEFORE B's write, reconstruct C's tree. ──
    let c_req = subscribe_request(&session_id, layer, 0, 0, "witness-c");
    let c_response = handle_subscribe_read_layer(&svc, c_req)
        .await
        .expect("witness subscribe must succeed");
    let mut c_stream = c_response.into_inner();
    let (c_header, c_body) = recv_message(&mut c_stream).await;
    assert_eq!(c_header.object_id, object_id);
    assert_eq!(c_header.sequence_num, A_ESTABLISHED_SEQ);
    let mut c_state = ObjectState::default();
    apply_ompf_to_state(&mut c_state, &c_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "witness catch-up must reconstruct A's scene (sections 1 and 2)"
    );
    recv_caught_up_ack(&mut c_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── Reconnector B: GENUINE out-of-window reconnect (object_id set, stale
    //    have_version=1). Drains its faithful coalesced catch-up + caught-up ack. ──
    let b_req = subscribe_request(&session_id, layer, object_id, 1, "client-b");
    let b_response = handle_subscribe_read_layer(&svc, b_req)
        .await
        .expect("client-b subscribe must succeed");
    let mut b_stream = b_response.into_inner();
    let (b_header, b_body) = recv_message(&mut b_stream).await;
    assert_eq!(b_header.object_id, object_id);
    assert_eq!(
        b_header.sequence_num, A_ESTABLISHED_SEQ,
        "B's out-of-window catch-up must carry the current server seq"
    );
    let b_decoded = decode_broadcast_delta(&b_body);
    assert_eq!(
        b_decoded.base_version, 1,
        "B's coalesced catch-up base_version must equal its stale have_version (never 0)"
    );
    recv_caught_up_ack(&mut b_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── B, now caught-up, issues its next INCREMENTAL (non-reset) write adding its
    //    own section + field (seq=4). Distinct client_id so C receives the relay. ──
    let b_commit = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 3,
            section_name: "b-root".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "b1".to_owned(),
            value_or_ext_hash: b"vb1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "client-b",
        300,
        &b_commit,
    )
    .await;

    // ── POSITIVE property #1: server state after B's write still contains A's
    //    sections + field values intact, PLUS B's new section. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    for (sid, key, expected) in [
        (1u64, "a1", &b"va1"[..]),
        (2, "a2", &b"va2"[..]),
        (3, "b1", &b"vb1"[..]),
    ] {
        assert!(
            server_state.sections.contains_key(&sid),
            "section {sid} must survive/exist after B's write"
        );
        let f = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be present after B's write"));
        assert_eq!(
            f.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) value must be intact after B's write (no wipe)"
        );
    }

    // ── POSITIVE property #2: the witness receives B's per-commit relay; it is not
    //    a reset, carries no deletion of A's prims, and retains A's state. ──
    let (relay_header, relay_body) = recv_message(&mut c_stream).await;
    assert_eq!(
        relay_header.object_id, object_id,
        "relay object_id must be stable"
    );
    assert_eq!(
        relay_header.sequence_num,
        A_ESTABLISHED_SEQ + 1,
        "witness must observe B's write as a single per-commit relay at seq=4"
    );
    let relay_delta = decode_broadcast_delta(&relay_body);
    assert!(
        !(relay_delta.is_diff && relay_delta.base_version == 0),
        "B's relayed per-commit write must NOT be a reset (would wipe A's prims)"
    );
    assert!(
        !relay_delta
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "B's relayed write must carry no section deletions of A's prims"
    );
    apply_ompf_to_state(&mut c_state, &relay_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "A's sections must survive in the witness's tree after B's relayed write"
    );
    assert!(
        c_state.sections.contains_key(&3),
        "B's new section must be added to the witness's tree"
    );
    assert_eq!(
        c_state.fields.get(&(1, "a1".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"va1".to_vec()
        )),
        "A's field value must survive in the witness's tree (no wipe)"
    );

    // ── POSITIVE property #3: B's own stream self-echo-suppresses its write. ──
    assert_no_next_message(&mut b_stream, Duration::from_millis(300)).await;
}

/// 3. Off mode (default): the SAME trimmed out-of-window scenario as test 1, run
///    through the LEGACY coalesced diff. Pins that the Engineer's Snapshot branch
///    does not alter default behavior — Regime 3 still delivers a single coalesced
///    envelope labelled the current seq, retracts ONLY the tombstoned section, and
///    leaves the abandoned-unreachable prim as the known legacy ghost (the exact
///    divergence Snapshot mode corrects). State-parity holds against the FULL
///    materialized state (orphan included), matching `compute_diff_at_cut`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn off_mode_out_of_window_reconnect_unchanged_regression_pin() {
    let tv = PooledTestValkey::setup().await;
    // Legacy mode — the shared fixture pins `catchup_snapshot_enabled = false`
    // (legacy per-commit reconstruction). Assert it so a fixture drift to the
    // corrected-snapshot path can't silently reinterpret this regression pin.
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
    assert!(
        !svc.catchup_snapshot_enabled,
        "regression pin must run in the legacy (snapshot-disabled) mode"
    );

    let layer = "off-oow-regression-layer";
    let (session_id, object_id, mut mirror) =
        build_oow_reconnect_scene(&tv, &svc, "off-oow-regression", layer).await;

    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);

    // ── Reconnect presenting the stale have_version on the SAME object_id. ──
    let sub_req = subscribe_request(
        &session_id,
        layer,
        object_id,
        OOW_HAVE_VERSION,
        "reconnector",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, OOW_CURRENT_SEQ,
        "legacy coalesced envelope must carry the current server seq"
    );
    assert!(
        !body.is_empty(),
        "legacy coalesced envelope must be non-empty"
    );

    let decoded = decode_broadcast_delta(&body);
    assert!(decoded.is_diff, "legacy coalesced diff is is_diff=true");
    assert_eq!(
        decoded.base_version, OOW_HAVE_VERSION,
        "legacy coalesced diff base_version equals have_version"
    );

    // ── Legacy retraction set: ONLY the tombstoned section — the abandoned prim is
    //    NOT retracted (the known legacy ghost that Snapshot mode fixes). ──
    let retracted = emitted_delete_section_ids(&decoded);
    assert!(
        retracted.contains(&OOW_TOMBSTONE_SECTION),
        "legacy coalesced diff must still retract the tombstoned section {OOW_TOMBSTONE_SECTION}; retracted={retracted:?}"
    );
    assert!(
        !retracted.contains(&OOW_ABANDONED_PRIM),
        "legacy coalesced diff must NOT retract the abandoned prim {OOW_ABANDONED_PRIM} (unchanged legacy behavior); retracted={retracted:?}"
    );

    // ── Exactly-once caught-up ack with tracked_version == current seq. ──
    recv_caught_up_ack(&mut stream, object_id, OOW_CURRENT_SEQ).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // ── State-parity: the legacy reconstruction (read-side, honoring the emitted
    //    DiffDeleteSection retractions) equals the FULL materialized state —
    //    including the abandoned orphan the legacy path re-emits as a stale ghost. ──
    apply_catchup_read_side(&mut mirror, &body);
    let reconstructed_ids: std::collections::HashSet<u64> =
        mirror.sections.keys().copied().collect();
    let server_ids: std::collections::HashSet<u64> =
        server_state.sections.keys().copied().collect();
    assert_eq!(
        reconstructed_ids, server_ids,
        "legacy Regime-3 reconstruction must equal the server's full materialized section set \
         (abandoned orphan re-emitted as a stale ghost — unchanged legacy behavior)"
    );
    assert!(
        mirror.sections.contains_key(&OOW_ABANDONED_PRIM),
        "legacy mode keeps the abandoned prim {OOW_ABANDONED_PRIM} as a stale ghost"
    );
    assert!(
        !mirror.sections.contains_key(&OOW_TOMBSTONE_SECTION),
        "legacy mode still retracts the tombstoned section {OOW_TOMBSTONE_SECTION}"
    );
    // Structural section-identity parity against the server materialized state.
    for (sid, s) in &server_state.sections {
        let r = mirror
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("section {sid} must reconstruct in legacy mode"));
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(
            r.section_order, s.section_order,
            "section {sid} order parity"
        );
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Snapshot-mode IN-WINDOW reconnect (Plan b.msb Epic 4 Task 1, t2.msb.xq.so).
//
// The in-window regime differs from the out-of-window analogues above in exactly
// ONE structural respect: the per-object stream is NOT trimmed, so the
// reconnector's start id `{have_version+1}-0` is still present and `run_catch_up`
// selects the `in_window` branch (rather than falling through to Regime 3). Under
// `catchup_snapshot_enabled = true`, Task 1's `catch_up_in_window_snapshot` converges
// on the SAME faithful `compute_faithful_coalesced_at_cut` primitive the out-of-window
// Snapshot arm already uses — so the delivered envelope, ghost retraction, and
// state-parity properties are byte-identical to the out-of-window Snapshot tests;
// the tests below mirror those 1:1 with the trim replaced by an untrimmed
// start-id-PRESENT probe.
//
// SR-6.2 (real multi-section traffic, not a synthetic single-section fixture):
// these reuse the shared FOUR-section / FIVE-commit synthetic history of
// `build_oow_reconnect_scene` — parent→child prims, a burst field whose stored
// cumulative cursor diverges from its authoring rank, a created-then-abandoned
// (reparented-unreachable) scratch prim, and an explicitly tombstoned section.
// That synthetic history covers the STORED-BUT-UNREACHABLE ghost arm
// (`src/domain/diff.rs`) on a scene that is synthetic end to end: the captured
// traffic never reaches that arm by itself, because there the stored section set
// equals the net-live set.
//
// The committed real captured fixture `tools/fixtures/gze_inbound_deltas.json` IS
// replayed in-process too, by `snapshot_catchup_matches_cpp_golden_oracle_offline`,
// `coalesced_reconnect_catchup_set_order_matches_cpp_golden_subsequence_offline`,
// the five-test
// `snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_{1,3,5,6,32}`
// family (one shared body, `run_in_window_fold_case`) — the in-window reconnect
// analogue of the test below driven by that real traffic (b.xvq) — and
// `snapshot_in_window_reconnect_real_traffic_plus_ghost_retraction_discriminates_read_side_semantics`,
// which replays the same capture and then appends two commits authored against
// the captured prims so BOTH ghost arms fire on real prim ids with an explicit
// ignore-the-retraction counterfactual. That hybrid narrows, but does not close,
// the synthetic gap: the SCENE is real Kit traffic, the final two edits are
// test-authored. A true end-to-end Kit capture of a detach/delete session is
// tracked in ticket b.gmb. (The debug-only cross-map-drift assertion that once
// blocked an in-process replay, b.mq1, is resolved.)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the shared Epic-2/3 editing history on `svc` WITHOUT trimming the stream,
/// so the reconnector's start id (`{have_version+1}-0`) remains present and the
/// reconnect lands in the `in_window` regime. The scene contents and the returned
/// `(session_id, object_id, mirror)` are identical to `build_oow_reconnect_scene`
/// — only the terminal step differs (probe start-id PRESENT instead of XTRIM +
/// probe-empty). `svc` MUST be built with a roomy MAXLEN (4096) so none of the six
/// commits are naturally evicted.
async fn build_in_window_reconnect_scene(
    tv: &PooledTestValkey,
    svc: &LiveEditService,
    name_prefix: &str,
    layer: &str,
) -> (String, u64, ObjectState) {
    let (session_id, object_id) =
        create_session_and_layer_pooled(tv, svc, &format!("{name_prefix}-session"), layer).await;

    // Reconnector mirror — the empty-create (seq=1) plus the two pre-window
    // commits it applied before disconnecting.
    let mut mirror = ObjectState::default();
    apply_delta(&mut mirror, None).expect("mirror empty create (seq=1)");

    // ── Commit 1 (seq=2): root R(1)+name; child A(2)+posX("first", so=5); scratch
    //    SC(99)+temp under root; deletable DL(50)+dlf under root. ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: OOW_ABANDONED_PRIM,
                section_name: "scratch".to_owned(),
                section_type: 0,
                section_order: 30,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: OOW_TOMBSTONE_SECTION,
                section_name: "doomed".to_owned(),
                section_type: 0,
                section_order: 40,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
            SetField {
                section_id: OOW_ABANDONED_PRIM,
                key_name: "temp".to_owned(),
                value_or_ext_hash: b"junk".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 3,
            },
            SetField {
                section_id: OOW_TOMBSTONE_SECTION,
                key_name: "dlf".to_owned(),
                value_or_ext_hash: b"doomed-v".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 4,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        100,
        &commit1,
    )
    .await;
    apply_delta(&mut mirror, Some(commit1.clone())).expect("mirror commit1 (seq=2)");

    // ── Commit 2 (seq=3): child B(3)+color; BURST-update A.posX="mid" (authoring
    //    rank 2) so its stored cumulative cursor now diverges from the rank. This
    //    is the reconnector's last applied version. ──
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"mid".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        200,
        &commit2,
    )
    .await;
    apply_delta(&mut mirror, Some(commit2.clone())).expect("mirror commit2 (seq=3)");
    assert_eq!(
        mirror.sequence_num, OOW_HAVE_VERSION,
        "reconnector mirror must be at have_version after the two pre-window commits"
    );

    // Everything below happens WHILE the reconnector is disconnected — its mirror
    // is deliberately NOT advanced past OOW_HAVE_VERSION.

    // ── Commit 3 (seq=4): ABANDON scratch SC(99) — reparent under a non-existent
    //    parent so it becomes unreachable from root WITHOUT deleting it (stays
    //    materialized as an orphan). Ghost-retraction leg (b). ──
    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: OOW_ABANDONED_PRIM,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        300,
        &commit3,
    )
    .await;

    // ── Commit 4 (seq=5): DELETE DL(50) — writer-side DeleteSection, which writes
    //    a tombstone and drops the section from the live index. Ghost-retraction
    //    leg (a). ──
    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DeleteSection(
            oodf_ffi::CommandDeleteSection {
                parent_id: 1,
                section_id: OOW_TOMBSTONE_SECTION,
            },
        )],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        400,
        &commit4,
    )
    .await;

    // ── Commit 5 (seq=6): BURST-update A.posX="final" (last-writer wins) AND
    //    STRUCTURALLY re-assert childA(2) under root R(1) via DiffChangeSection,
    //    keeping its DIVERGENT stored section_order (10). Re-asserting the section
    //    bumps its `last_version` to 6 (> have_version=3), so the coalesced
    //    catch-up re-emits childA as a modified section and, in snapshot mode,
    //    restates its unchanged sibling childB(3) with it (R now has a changed
    //    child) — the only two live sections re-emitted in this scene. That
    //    makes the reconnect exercise the b.bdx coalesced sibling-order
    //    override: A's stored order stays 10 but its real per-parent sibling
    //    index under R is 0 (first live child), so the emitted `section_order`
    //    MUST be 0, not 10 (asserted by the Snapshot establish test; the
    //    Off-mode regression pin is unaffected — it only checks per-commit seq
    //    contiguity). ──
    let commit5 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "childA".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "posX".to_owned(),
            value_or_ext_hash: b"final".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        svc,
        &session_id,
        layer,
        object_id,
        "scene-writer",
        500,
        &commit5,
    )
    .await;

    // ── In-window precondition: do NOT trim. Probe-assert the reconnector's start
    //    id ({have_version+1}-0 = "4-0") is STILL present, so the reconnect XRANGE
    //    slice's first entry matches and `run_catch_up` selects the `in_window`
    //    branch (the mirror of the out-of-window builder's trim-and-probe-empty). ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let start_id = format!("{}-0", OOW_HAVE_VERSION + 1);
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, start_id.clone(), start_id.clone(), Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.first().map(|(id, _)| id == &start_id).unwrap_or(false),
        "reconnector's start id {start_id} must still be present (untrimmed) so the reconnect lands in the in_window regime — got {probe:?}"
    );

    (session_id, object_id, mirror)
}

/// In-window Snapshot mode: an in-window reconnector receives the SAME faithful
/// coalesced catch-up as the out-of-window Snapshot path — `base_version =
/// have_version` (never 0), the current server seq in the header, exactly-once
/// caught-up ack — that RETRACTS BOTH ghost legs (tombstoned DL(50) AND
/// abandoned-unreachable SC(99)) via `DiffDeleteSection`, so the reconnector's
/// scene equals the server net-live scene, the burst field carries its LAST value,
/// and the `in_window` branch is confirmed selected (untrimmed, start-id-present).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_establishes_correct_scene() {
    let tv = PooledTestValkey::setup().await;
    // Roomy MAXLEN so nothing trims — the whole point of the in-window regime.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-inwindow-establish-layer";
    let (session_id, object_id, mut mirror) =
        build_in_window_reconnect_scene(&tv, &svc, "snap-inwindow-establish", layer).await;

    // ── Server-side sanity: the abandoned prim is still materialized as an orphan;
    //    the deleted section is gone from the live index but present as a tombstone. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    assert!(
        server_state.sections.contains_key(&OOW_ABANDONED_PRIM),
        "abandoned prim {OOW_ABANDONED_PRIM} must still exist server-side as an orphan"
    );
    assert!(
        !server_state.sections.contains_key(&OOW_TOMBSTONE_SECTION),
        "tombstoned section {OOW_TOMBSTONE_SECTION} must be gone from the live index"
    );
    assert!(
        snap.tombstones
            .iter()
            .any(|(sid, ..)| *sid == OOW_TOMBSTONE_SECTION),
        "server must carry a tombstone for {OOW_TOMBSTONE_SECTION}; got {:?}",
        snap.tombstones
    );

    // ── Reconnect presenting the stale have_version on the SAME object_id, WITHOUT
    //    trimming (the scene builder already probe-asserted the in_window branch). ──
    let sub_req = subscribe_request(
        &session_id,
        layer,
        object_id,
        OOW_HAVE_VERSION,
        "reconnector",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match"
    );
    assert_eq!(
        header.sequence_num, OOW_CURRENT_SEQ,
        "in-window coalesced snapshot header must carry the current server seq"
    );
    assert!(
        !body.is_empty(),
        "message #1 must carry a single non-empty coalesced catch-up body"
    );

    // ── Envelope: incremental (is_diff=true) with base_version == have_version —
    //    NEVER a base_version=0 from-empty reset the reconnector would reject. ──
    let decoded = decode_broadcast_delta(&body);
    assert!(decoded.is_diff, "coalesced catch-up must be is_diff=true");
    assert_eq!(
        decoded.base_version, OOW_HAVE_VERSION,
        "coalesced catch-up base_version must equal have_version (never 0)"
    );
    assert_ne!(
        decoded.base_version, 0,
        "an advanced reconnector must NEVER receive a base_version=0 reset"
    );

    // ── Unified ghost retraction: BOTH the tombstoned section AND the
    //    abandoned-to-unreachable prim are retracted via DiffDeleteSection. ──
    let retracted = emitted_delete_section_ids(&decoded);
    assert!(
        retracted.contains(&OOW_TOMBSTONE_SECTION),
        "catch-up must retract the tombstoned section {OOW_TOMBSTONE_SECTION} (leg a); retracted={retracted:?}"
    );
    assert!(
        retracted.contains(&OOW_ABANDONED_PRIM),
        "catch-up must retract the abandoned-unreachable prim {OOW_ABANDONED_PRIM} (leg b); retracted={retracted:?}"
    );
    // Neither ghost is (re)emitted as live content, nor are their fields.
    let live_emitted = emitted_change_section_ids(&decoded);
    assert!(
        !live_emitted.contains(&OOW_ABANDONED_PRIM)
            && !live_emitted.contains(&OOW_TOMBSTONE_SECTION),
        "no ghost may be emitted as a live DiffChangeSection; got {live_emitted:?}"
    );
    assert!(
        !decoded
            .set_fields
            .iter()
            .any(|sf| sf.section_id == OOW_ABANDONED_PRIM || sf.section_id == OOW_TOMBSTONE_SECTION),
        "no ghost's field may be emitted as live content"
    );
    // The burst field's final value is carried — proving the faithful builder
    // emits the authoring rank's LAST value, not a stale cursor value.
    let posx = decoded
        .set_fields
        .iter()
        .find(|sf| sf.section_id == 2 && sf.key_name == "posX")
        .expect("catch-up must carry the changed burst field childA.posX");
    assert_eq!(
        posx.value_or_ext_hash, b"final",
        "catch-up must carry the LAST-writer value of the burst field"
    );

    // ── A SINGLE non-empty body then the exactly-once caught-up ack; no trailing
    //    messages (in-window Snapshot delivers ONE coalesced envelope, not a
    //    per-commit replay). ──
    recv_caught_up_ack(&mut stream, object_id, OOW_CURRENT_SEQ).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;

    // ── Reconstruct the reconnector's scene from its pre-reconnect state + the
    //    delivered catch-up (honoring the read-side DiffDeleteSection retraction),
    //    and assert it equals the server's net-live (root-reachable) scene. ──
    apply_catchup_read_side(&mut mirror, &body);

    let reachable = reachable_section_ids(&server_state);
    let reconstructed_ids: std::collections::HashSet<u64> =
        mirror.sections.keys().copied().collect();
    assert_eq!(
        reconstructed_ids, reachable,
        "reconstructed scene must equal the server's net-live (root-reachable) section set"
    );
    assert!(
        !mirror.sections.contains_key(&OOW_ABANDONED_PRIM),
        "abandoned prim {OOW_ABANDONED_PRIM} must be ABSENT from the reconnector's scene (no stale ghost)"
    );
    assert!(
        !mirror.sections.contains_key(&OOW_TOMBSTONE_SECTION),
        "tombstoned section {OOW_TOMBSTONE_SECTION} must be ABSENT from the reconnector's scene (no stale ghost)"
    );
    assert!(
        !mirror
            .fields
            .contains_key(&(OOW_ABANDONED_PRIM, "temp".to_owned()))
            && !mirror
                .fields
                .contains_key(&(OOW_TOMBSTONE_SECTION, "dlf".to_owned())),
        "no ghost's field may survive in the reconnector's scene"
    );

    // Structural parity (parent/name/type) on every net-live section. section_order
    // is asserted SEPARATELY below against the REAL sibling index — deliberately
    // NOT compared to the server's STORED order, which for childA(2) diverges
    // (stored 10 vs sibling index 0). A reconstructed-vs-stored comparison would
    // re-encode the superseded pre-b.bdx contract and pass vacuously.
    for sid in &reachable {
        let r = mirror.sections.get(sid).expect("live section reconstructs");
        let s = server_state
            .sections
            .get(sid)
            .expect("live section on server");
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }

    // ── b.bdx coalesced sibling-order override: the burst section childA(2) is
    //    re-emitted (commit 5 re-asserted it structurally), so the faithful
    //    coalesced catch-up carries its REAL per-parent sibling index — 0, the
    //    first live child of root R(1) — in `section_order`, NOT the divergent
    //    STORED order the server persists (10). Assert the on-the-wire emitted
    //    value AND the reconstructed value are the sibling index, and that both
    //    diverge from the server-stored order. FAILS if the coalesced
    //    sibling-position override is reverted (would emit the stored 10). ──
    assert_eq!(
        server_state.sections[&2].section_order, 10,
        "server must persist childA's divergent STORED authoring order (10)"
    );
    let emitted_childa_order = decoded
        .struct_commands
        .iter()
        .find_map(|c| match c {
            StructCommand::DiffChangeSection(s) if s.section_id == 2 => Some(s.section_order),
            _ => None,
        })
        .expect("childA(2) must be re-emitted as a live DiffChangeSection in the catch-up");
    assert_eq!(
        emitted_childa_order, 0,
        "emitted childA section_order must be the sibling index 0, not the stored order 10"
    );
    assert_eq!(
        mirror.sections[&2].section_order, 0,
        "reconstructed childA section_order must be the sibling index 0 (not the stored 10) — \
         proving the coalesced catch-up delivered the real sibling order (b.bdx)"
    );
    for ((sid, key), s_field) in &server_state.fields {
        if !reachable.contains(sid) {
            continue;
        }
        let r_field = mirror
            .fields
            .get(&(*sid, key.clone()))
            .unwrap_or_else(|| panic!("reconstructed must carry net-live field ({sid},{key})"));
        assert_eq!(
            r_field.value, s_field.value,
            "field ({sid},{key}) value must match the server (last-writer-wins)"
        );
    }
    // The burst field specifically reconstructs its final value.
    assert_eq!(
        mirror.fields.get(&(2, "posX".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"final".to_vec()
        )),
        "childA.posX must reconstruct its final burst value"
    );
}

/// In-window Snapshot mode, SR-3.1 no-wipe: after a Snapshot-mode IN-WINDOW
/// reconnector has caught up, its next incremental write must leave the prior
/// writer's sections + field values intact (asserted positively on server state
/// and on a present-from-start witness's reconstructed tree); the relay is not a
/// reset and carries no deletion of the prior prims; the reconnector
/// self-echo-suppresses. Mirrors the out-of-window analogue but stays in-window
/// (no XTRIM; start-id-present probe).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_caught_up_client_write_preserves_prior_state() {
    let tv = PooledTestValkey::setup().await;
    // Roomy MAXLEN so B's start id is retained and B lands in the in_window regime.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "snap-inwindow-nowipe-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "snap-inwindow-nowipe-session", layer).await;

    // ── Writer A commits a scene of two sections with fields (seq 2 & 3). ──
    let a_commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 1,
            section_name: "a-root".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "a1".to_owned(),
            value_or_ext_hash: b"va1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        100,
        &a_commit1,
    )
    .await;

    let a_commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a-child".to_owned(),
            section_type: 0,
            section_order: 5,
        })],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "a2".to_owned(),
            value_or_ext_hash: b"va2".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "writer-a",
        200,
        &a_commit2,
    )
    .await;

    const A_ESTABLISHED_SEQ: u64 = 3; // CreateLayer(1) + A's two commits.

    // ── In-window precondition for reconnector B: B's stale have_version=1 (it only
    //    saw the empty create) means start id "2-0". Do NOT trim — probe-assert it
    //    is present so B lands in the in_window regime (not Regime 3). ──
    let stream_key = tv.key_schema().layer_stream(object_id);
    let probe: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "2-0", "2-0", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    assert!(
        probe.first().map(|(id, _)| id == "2-0").unwrap_or(false),
        "reconnector B's start id 2-0 must be present (untrimmed) so B lands in the in_window regime — got {probe:?}"
    );

    // ── Present-from-start witness C: subscribe BEFORE B's write, reconstruct C's tree. ──
    let c_req = subscribe_request(&session_id, layer, 0, 0, "witness-c");
    let c_response = handle_subscribe_read_layer(&svc, c_req)
        .await
        .expect("witness subscribe must succeed");
    let mut c_stream = c_response.into_inner();
    let (c_header, c_body) = recv_message(&mut c_stream).await;
    assert_eq!(c_header.object_id, object_id);
    assert_eq!(c_header.sequence_num, A_ESTABLISHED_SEQ);
    let mut c_state = ObjectState::default();
    apply_ompf_to_state(&mut c_state, &c_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "witness catch-up must reconstruct A's scene (sections 1 and 2)"
    );
    recv_caught_up_ack(&mut c_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── Reconnector B: IN-WINDOW reconnect (object_id set, stale have_version=1,
    //    start id "2-0" still present). Drains its single faithful coalesced
    //    catch-up + caught-up ack. ──
    let b_req = subscribe_request(&session_id, layer, object_id, 1, "client-b");
    let b_response = handle_subscribe_read_layer(&svc, b_req)
        .await
        .expect("client-b subscribe must succeed");
    let mut b_stream = b_response.into_inner();
    let (b_header, b_body) = recv_message(&mut b_stream).await;
    assert_eq!(b_header.object_id, object_id);
    assert_eq!(
        b_header.sequence_num, A_ESTABLISHED_SEQ,
        "B's in-window snapshot catch-up must carry the current server seq"
    );
    let b_decoded = decode_broadcast_delta(&b_body);
    assert!(
        b_decoded.is_diff,
        "B's in-window snapshot catch-up must be is_diff=true"
    );
    assert_eq!(
        b_decoded.base_version, 1,
        "B's coalesced catch-up base_version must equal its stale have_version (never 0)"
    );
    recv_caught_up_ack(&mut b_stream, object_id, A_ESTABLISHED_SEQ).await;

    // ── B, now caught-up, issues its next INCREMENTAL (non-reset) write adding its
    //    own section + field (seq=4). Distinct client_id so C receives the relay. ──
    let b_commit = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 3,
            section_name: "b-root".to_owned(),
            section_type: 0,
            section_order: 10,
        })],
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "b1".to_owned(),
            value_or_ext_hash: b"vb1".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "client-b",
        300,
        &b_commit,
    )
    .await;

    // ── POSITIVE property #1: server state after B's write still contains A's
    //    sections + field values intact, PLUS B's new section. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    for (sid, key, expected) in [
        (1u64, "a1", &b"va1"[..]),
        (2, "a2", &b"va2"[..]),
        (3, "b1", &b"vb1"[..]),
    ] {
        assert!(
            server_state.sections.contains_key(&sid),
            "section {sid} must survive/exist after B's write"
        );
        let f = server_state
            .fields
            .get(&(sid, key.to_owned()))
            .unwrap_or_else(|| panic!("field ({sid},{key}) must be present after B's write"));
        assert_eq!(
            f.value,
            live_edit::domain::types::FieldValue::Inline(expected.to_vec()),
            "field ({sid},{key}) value must be intact after B's write (no wipe)"
        );
    }

    // ── POSITIVE property #2: the witness receives B's per-commit relay; it is not
    //    a reset, carries no deletion of A's prims, and retains A's state. ──
    let (relay_header, relay_body) = recv_message(&mut c_stream).await;
    assert_eq!(
        relay_header.object_id, object_id,
        "relay object_id must be stable"
    );
    assert_eq!(
        relay_header.sequence_num,
        A_ESTABLISHED_SEQ + 1,
        "witness must observe B's write as a single per-commit relay at seq=4"
    );
    let relay_delta = decode_broadcast_delta(&relay_body);
    assert!(
        !(relay_delta.is_diff && relay_delta.base_version == 0),
        "B's relayed per-commit write must NOT be a reset (would wipe A's prims)"
    );
    assert!(
        !relay_delta
            .struct_commands
            .iter()
            .any(|c| matches!(c, StructCommand::DiffDeleteSection(_))),
        "B's relayed write must carry no section deletions of A's prims"
    );
    apply_ompf_to_state(&mut c_state, &relay_body);
    assert!(
        c_state.sections.contains_key(&1) && c_state.sections.contains_key(&2),
        "A's sections must survive in the witness's tree after B's relayed write"
    );
    assert!(
        c_state.sections.contains_key(&3),
        "B's new section must be added to the witness's tree"
    );
    assert_eq!(
        c_state.fields.get(&(1, "a1".to_owned())).map(|f| &f.value),
        Some(&live_edit::domain::types::FieldValue::Inline(
            b"va1".to_vec()
        )),
        "A's field value must survive in the witness's tree (no wipe)"
    );

    // ── POSITIVE property #3: B's own stream self-echo-suppresses its write. ──
    assert_no_next_message(&mut b_stream, Duration::from_millis(300)).await;
}

/// Legacy path (this fixture forces `catchup_snapshot_enabled = false`; the
/// product default is now ON): the SAME untrimmed in-window scenario, run through
/// the LEGACY per-commit `XRANGE` replay. Pins that Task 1's Snapshot-gated source
/// edit leaves the legacy in-window path intact — the reconnector still
/// receives the missed commits (seqs 4, 5, 6) as individual per-commit diffs (NOT a
/// single coalesced envelope, NOT a from-empty reset), then the exactly-once
/// caught-up ack at the last replayed seq. Complements the pre-existing
/// `in_window_reconnect_replays_individual_per_commit_diffs` pin by driving the
/// rich multi-section history through the legacy in-window path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn off_mode_in_window_reconnect_unchanged_regression_pin() {
    let tv = PooledTestValkey::setup().await;
    // Legacy mode — the shared fixture pins `catchup_snapshot_enabled = false`
    // (legacy per-commit reconstruction). Assert it so a fixture drift to the
    // corrected-snapshot path can't silently reinterpret this regression pin.
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    assert!(
        !svc.catchup_snapshot_enabled,
        "regression pin must run in the legacy (snapshot-disabled) mode"
    );

    let layer = "off-inwindow-regression-layer";
    let (session_id, object_id, _mirror) =
        build_in_window_reconnect_scene(&tv, &svc, "off-inwindow-regression", layer).await;

    // ── Reconnect presenting the stale have_version on the SAME object_id. Off mode
    //    replays the missed commits per-commit (seqs 4, 5, 6). ──
    let sub_req = subscribe_request(
        &session_id,
        layer,
        object_id,
        OOW_HAVE_VERSION,
        "reconnector",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let mut observed_seqs = Vec::new();
    for _ in OOW_HAVE_VERSION..OOW_CURRENT_SEQ {
        let (header, body) = recv_message(&mut stream).await;
        assert_eq!(header.object_id, object_id);
        assert!(
            !body.is_empty(),
            "each legacy per-commit diff body must be non-empty"
        );
        assert!(
            header.total_size > 0,
            "a per-commit diff carries total_size > 0 (rules out the caught-up ack)"
        );
        // A per-commit replay entry must never be a from-empty reset.
        let decoded = decode_broadcast_delta(&body);
        assert!(
            !(decoded.is_diff && decoded.base_version == 0),
            "a legacy per-commit replay entry must NOT be a base_version=0 reset"
        );
        observed_seqs.push(header.sequence_num);
    }
    assert_eq!(
        observed_seqs,
        vec![4, 5, 6],
        "Off-mode in-window replay must deliver the missed commits as contiguous per-commit seqs 4, 5, 6"
    );

    // ── Exactly-once caught-up ack at the last replayed seq; no trailing messages. ──
    recv_caught_up_ack(&mut stream, object_id, OOW_CURRENT_SEQ).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SR-2.4 — empty-session (no-edits) brand-new establishment (t3.msb.xq.e6.nq)
//
// The PRD "empty-no-edits" edge case: a client joining a session where nothing
// has been edited yet must still establish a valid — possibly empty — scene
// without tripping the client's "empty content is fatal" rule. The server is
// responsible for correctness here: message #1 must be a valid from-empty create
// (well-formed empty create applicable to an empty tree, `is_diff=true`,
// `base_version=0`), NOT a zero-size/absent one, and the only legitimate
// zero-size signal is the caught-up terminator (`total_size==0`) that follows.
//
// Both tests below share the same assertion shape, differing only in catch-up
// mode: the primary drives `Snapshot` (exercising `compute_faithful_full_content_at_cut`
// on an empty net-live scene), the base-case parity test drives the default `Off`
// (exercising the legacy `compute_diff_at_cut` full-content path). Both prove message #1
// is a NON-zero-size content header — never the terminator — so a regression that
// collapses the empty scene into a terminator-only stream is caught in either
// mode. `checkpoint_enabled` stays `false` (SR-4.1) — the fixture `Config` wires
// the disabled recovery hooks, so no DR path is engaged.
//
// The crux distinguished carefully throughout: an ILLEGITIMATE zero-size
// establishment message (message #1) vs. the LEGITIMATE zero-size caught-up
// terminator (the ONE and only zero-size message the stream may carry).
// ═══════════════════════════════════════════════════════════════════════════════

/// Assert the decoded empty-session establishment delta is a valid, well-formed
/// EMPTY from-empty create and that it applies cleanly onto a fresh empty
/// `ObjectState`, leaving that state empty. Shared by both mode variants below so
/// the establishment contract is asserted identically regardless of catch-up
/// mode. `body` is the reassembled message-#1 OMPF payload.
fn assert_empty_establishment_body(body: &[u8]) {
    // decode_broadcast_delta re-asserts the OMPF carries a NON-empty OODF delta
    // (guards the burst-coalesce-to-empty regression b.gze): the empty-create is
    // a well-formed delta buffer, not an absent one.
    let decoded = decode_broadcast_delta(body);
    assert!(
        decoded.is_diff,
        "empty-session message #1 must be a from-empty create (is_diff=true); got is_diff={}",
        decoded.is_diff
    );
    assert_eq!(
        decoded.base_version, 0,
        "empty-session message #1 must be from-empty (base_version=0) per SR-1.2"
    );
    // A well-formed but EMPTY create: no structural commands, no fields, no
    // time-samples — nothing has been edited yet.
    assert!(
        decoded.struct_commands.is_empty(),
        "empty-session establishment must carry NO structural commands, got {}",
        decoded.struct_commands.len()
    );
    assert!(
        decoded.set_fields.is_empty(),
        "empty-session establishment must carry NO set_fields, got {}",
        decoded.set_fields.len()
    );
    assert!(
        decoded.time_samples.is_empty(),
        "empty-session establishment must carry NO time-samples, got {}",
        decoded.time_samples.len()
    );

    // Applies cleanly onto a fresh empty tree and leaves it empty — proving it is
    // a valid establishment payload, NOT a fatal empty-content message.
    let mut reconstructed = ObjectState::default();
    apply_ompf_to_state(&mut reconstructed, body);
    assert!(
        reconstructed.sections.is_empty(),
        "empty-create applied to an empty tree must leave NO sections, got {}",
        reconstructed.sections.len()
    );
    assert!(
        reconstructed.fields.is_empty(),
        "empty-create applied to an empty tree must leave NO fields, got {}",
        reconstructed.fields.len()
    );
    assert!(
        reconstructed.time_samples.is_empty(),
        "empty-create applied to an empty tree must leave NO time-samples, got {}",
        reconstructed.time_samples.len()
    );
}

/// Primary (Snapshot mode): an empty session (session + layer, zero committed
/// edits) subscribed brand-new must yield a message #1 that is a NON-zero-size
/// content header decoding to a valid empty from-empty create, followed by the
/// zero-size caught-up terminator at seq=1 — never a terminator-only stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_session_snapshot_join_establishes_valid_empty_create() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    // Select snapshot mode locally on this owned instance — the shared
    // `helpers.rs` builder signature is untouched. This exercises
    // `compute_faithful_full_content_at_cut` on an EMPTY net-live scene.
    svc.catchup_snapshot_enabled = true;

    // Empty session: create the session + layer (an empty-create delta) and issue
    // NO UpdateLayer calls. Nothing has been edited.
    let layer = "empty-snap-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "empty-snap-session", layer).await;

    // Brand-new subscribe (object_id=0, sequence_num=0).
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "empty-late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Message #1 MUST be the content header — NOT the terminator. The empties-gate
    // in `send_full_state` surfaces an empty payload as INTERNAL rather than
    // letting the stream collapse to the terminator-only sequence.
    let (header, body) = recv_message(&mut stream).await;
    assert!(
        header.total_size > 0,
        "empty-session message #1 MUST be a NON-zero-size content header — an \
         illegitimate zero-size establishment message would be the terminator-only \
         collapse the client treats as fatal; got total_size={}",
        header.total_size
    );
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match the created object"
    );
    assert_eq!(
        header.sequence_num, 1,
        "message #1 must carry the empty-create seq (1)"
    );

    assert_empty_establishment_body(&body);

    // The ONLY legitimate zero-size message: the caught-up terminator at seq=1,
    // emitted exactly once by `run_catch_up` after the establishment content.
    recv_caught_up_ack(&mut stream, object_id, 1).await;
    // Nothing further — no second establishment, no stray terminator.
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
}

/// Base-case parity (legacy path — this fixture leaves `catchup_snapshot_enabled`
/// at the fixture default of `false`, while the product default is now ON): the
/// identical observable establishment contract must hold on the legacy catch-up
/// path, exercising the legacy `compute_diff_at_cut` full-content path. Guards that the
/// empty-session base case holds regardless of catch-up mode and catches any
/// regression collapsing the empty scene into a terminator-only stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_session_off_mode_join_establishes_valid_empty_create() {
    let tv = PooledTestValkey::setup().await;
    // Legacy mode — leave `catchup_snapshot_enabled` at the fixture's `false`.
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    // Empty session: create the session + layer and issue NO UpdateLayer calls.
    let layer = "empty-off-layer";
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "empty-off-session", layer).await;

    // Brand-new subscribe (object_id=0, sequence_num=0).
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "empty-late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    // Message #1 MUST be the content header — NOT the terminator.
    let (header, body) = recv_message(&mut stream).await;
    assert!(
        header.total_size > 0,
        "empty-session (Off mode) message #1 MUST be a NON-zero-size content \
         header — never the terminator-only collapse; got total_size={}",
        header.total_size
    );
    assert_eq!(
        header.object_id, object_id,
        "message #1 object_id must match the created object"
    );
    assert_eq!(
        header.sequence_num, 1,
        "message #1 must carry the empty-create seq (1)"
    );

    assert_empty_establishment_body(&body);

    // The ONLY legitimate zero-size message: the caught-up terminator at seq=1.
    recv_caught_up_ack(&mut stream, object_id, 1).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.yqd re-test — watcher-turned-writer keeps updates flowing (Plan b.msb,
// Epic t1.msb.xq, Task t2.msb.xq.f6, test Subtask t3.msb.xq.f6.yu).
//
// Secondary user-reported symptom b.yqd ("asymmetric writer stall"): a
// participant who was only *watching* a session (an established, caught-up
// `SubscribeReadLayer` stream) then starts *editing* (issues `UpdateLayer`),
// after which live updates freeze — the watcher-turned-writer stops receiving
// other participants' edits, and/or their edits stop reaching peers. Plan b.msb
// folds b.yqd in *only insofar as it stems from the stale-view / edit-wipe
// problem*: Epics 2–3 plus this Epic's Task 1 (in-window reconnect
// corrected-snapshot fix in `catch_up_in_window`) removed the wipe cause. This
// automated scenario is the evidence the Task 4 investigation Subtask
// (`t3.msb.xq.f6.cv`) runs to reach its recorded determination of whether the
// freeze resolved as a consequence or is an independent update-sequencing defect
// that must be tracked on its own.
//
// The watcher's pre-edit catch-up (`subscribe_steady_state`, brand-new subscribe
// at object_id=0/seq=0) routes through the corrected catch-up path Task 1
// rewired. No JoinSession is needed — the freeze is a layer-delivery symptom, and
// the SubscribeReadLayer + concurrent UpdateLayer + same-client self-echo
// combination this scenario needs is exactly what this file already exercises.
// The scenario does not depend on the DR checkpoint subsystem:
// `build_pooled_service` constructs its `Config` with `checkpoint_enabled: false`
// and a disabled recovery hook, so `LIVE_EDIT_CHECKPOINT_ENABLED=false` holds by
// construction.
//
// Branch-dependent authoring:
//   - EXPECTED-RESOLVED (default, this version): after the watcher (Alice)
//     becomes a writer, her stream continues to receive every peer edit at
//     contiguous sequence numbers with no freeze, and her own edits propagate to
//     the peer subscriber (Bob); her own writes are correctly self-echo-suppressed
//     on her own stream (expected, not a freeze). The passing test is the evidence
//     the stall resolved. Every `recv_message` is `SUBSCRIBER_DEADLINE`-bounded,
//     so a genuine freeze fails with a descriptive timeout rather than hanging.
//   - IF THE STALL PERSISTS (only if the Task 4 investigation Subtask determines
//     it survives): this test is adjusted to PIN the currently-observed behaviour
//     (the bounded point at which Alice's stream stops delivering, asserted via
//     `assert_no_next_message` where a message would be expected under correct
//     behaviour) and a doc-comment reference is added to the standalone follow-up
//     Issue the investigation files (cross-referencing b.yqd), so the symptom is
//     never silently closed with the superseded bug tickets (b.zju, b.vvf,
//     b.yqd). Filing that Issue and adjusting this test are the investigation
//     Subtask's job, not this Subtask's.
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watcher_turned_writer_keeps_updates_flowing() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let layer = "watcher-writer-layer";
    // 1. Create session + layer (create_seq = 1).
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "watcher-writer-session", layer).await;

    // 2. Alice — the watcher-turned-writer — subscribes and catches up. This is
    //    her "watching" phase; her catch-up goes through the corrected catch-up
    //    path Task 1 rewired.
    let mut alice_stream =
        subscribe_steady_state(&svc, &session_id, layer, object_id, 1, "alice-watcher").await;

    // 3. Bob — a separate participant — subscribes and catches up the same way.
    //    Bob exists so the test can assert Alice's OWN edits propagate to a peer.
    let mut bob_stream =
        subscribe_steady_state(&svc, &session_id, layer, object_id, 1, "bob-peer").await;

    // 4. Transition: Alice (previously only watching) now begins EDITING. Her
    //    first write lands at server seq 2.
    apply_unique_update(&svc, &session_id, layer, object_id, "alice-watcher", 100).await;

    //    Alice's edit propagates to the peer subscriber Bob.
    let (bob_h, bob_body) = recv_message(&mut bob_stream).await;
    assert_eq!(
        bob_h.sequence_num, 2,
        "Bob must receive Alice's first edit at seq=2"
    );
    assert_eq!(bob_h.object_id, object_id, "Bob seq=2 object_id must match");
    assert!(
        !bob_body.is_empty(),
        "Bob's copy of Alice's edit must carry a non-empty body"
    );

    //    Alice does NOT receive her own edit — self-echo suppression, which is
    //    EXPECTED and is NOT the freeze.
    assert_no_next_message(&mut alice_stream, Duration::from_millis(500)).await;

    // 5. Crux / no-freeze after the transition: Bob writes (seq 3). After Alice
    //    became a writer, her subscription MUST keep receiving peer edits (no
    //    freeze). This is the direct b.yqd assertion — a genuine freeze fails the
    //    `SUBSCRIBER_DEADLINE`-bounded `recv_message` with a descriptive timeout.
    apply_unique_update(&svc, &session_id, layer, object_id, "bob-peer", 101).await;
    let (alice_h, alice_body) = recv_message(&mut alice_stream).await;
    assert_eq!(
        alice_h.sequence_num, 3,
        "after becoming a writer, Alice's stream MUST still deliver Bob's edit at seq=3 \
         (b.yqd no-freeze crux)"
    );
    assert_eq!(
        alice_h.object_id, object_id,
        "Alice seq=3 object_id must match"
    );
    assert!(
        !alice_body.is_empty(),
        "Alice's copy of Bob's edit must carry a non-empty body"
    );
    //    Bob does not receive his own edit — self-echo suppression on his stream.
    assert_no_next_message(&mut bob_stream, Duration::from_millis(500)).await;

    // 6. Sustained bidirectional flow: three more alternating rounds. Each writer's
    //    own commit is self-echo-suppressed on its own stream but advances that
    //    subscriber's tracked_version internally, so each PEER edit lands at the
    //    next contiguous server seq on the receiving stream with no gap. Proving
    //    flow is sustained, not a single lucky message.
    //
    //    Server-seq timeline for the rounds below (each `apply_unique_update`
    //    bumps the server seq by exactly 1):
    //      round 1: Alice writes seq 4 → Bob receives 4; Bob writes seq 5 → Alice receives 5
    //      round 2: Alice writes seq 6 → Bob receives 6; Bob writes seq 7 → Alice receives 7
    //      round 3: Alice writes seq 8 → Bob receives 8; Bob writes seq 9 → Alice receives 9
    let mut next_seq = 4u64;
    let mut section_id = 102u64;
    for round in 0..3u64 {
        // Alice writes; Bob receives at the next contiguous seq.
        apply_unique_update(
            &svc,
            &session_id,
            layer,
            object_id,
            "alice-watcher",
            section_id,
        )
        .await;
        let (h, body) = recv_message(&mut bob_stream).await;
        assert_eq!(
            h.sequence_num, next_seq,
            "round {round}: Bob must receive Alice's edit at contiguous seq={next_seq}"
        );
        assert_eq!(
            h.object_id, object_id,
            "round {round}: Bob object_id must match"
        );
        assert!(
            !body.is_empty(),
            "round {round}: Bob's received Alice edit must be non-empty"
        );
        // Alice must not receive her own write (self-echo suppression holds).
        assert_no_next_message(&mut alice_stream, Duration::from_millis(300)).await;
        next_seq += 1;
        section_id += 1;

        // Bob writes; Alice receives at the next contiguous seq — the watcher-
        // turned-writer keeps flowing.
        apply_unique_update(&svc, &session_id, layer, object_id, "bob-peer", section_id).await;
        let (h, body) = recv_message(&mut alice_stream).await;
        assert_eq!(
            h.sequence_num, next_seq,
            "round {round}: Alice (now a writer) must still receive Bob's edit at contiguous \
             seq={next_seq} — no asymmetric writer stall"
        );
        assert_eq!(
            h.object_id, object_id,
            "round {round}: Alice object_id must match"
        );
        assert!(
            !body.is_empty(),
            "round {round}: Alice's received Bob edit must be non-empty"
        );
        // Bob must not receive his own write.
        assert_no_next_message(&mut bob_stream, Duration::from_millis(300)).await;
        next_seq += 1;
        section_id += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.bdx — Corrected-snapshot catch-up OFFLINE acceptance against the committed
// C++ golden oracle.
//
// This validates the corrected snapshot catch-up (`catchup_snapshot_enabled = true`,
// now the product default) against the committed golden
// `tools/fixtures/gze_cpp_catchup_oracle.json` — the catch-up broadcast the LIVE
// legacy C++ server produced over 36 REAL captured Kit deltas
// (`tools/fixtures/gze_inbound_deltas.json`). It runs fully OFFLINE (no live
// server; the C++ host is unreachable from the sandbox) by replaying the same 36
// deltas through the in-process service and late-joining a fresh Snapshot-mode
// subscriber, then diffing our catch-up against the golden.
//
// THE ACCEPTANCE BAR IS DELIBERATELY NOT BYTE-IDENTICAL `setOrder` INTEGERS.
// The from-empty Kit client keys each catch-up field into an ordered
// `std::map<pair<saveCount, setOrder>>` under ONE shared `saveCount`, then
// RE-BASES every incoming `setOrder` onto its own running counter. So only two
// properties are load-bearing, and both are asserted here:
//   (b) GLOBAL UNIQUENESS — a duplicate `setOrder` collides on the client's map
//       and the field is silently dropped. Our dense `0..N-1` scheme guarantees
//       this; C++'s sparse global counter does too, but by a different mechanism.
//   (c) RELATIVE ORDER — sorting our fields by `set_order` must yield the SAME
//       (sectionId, keyName) sequence as sorting the golden's by its `setOrder`.
// Section `sectionOrder`, by contrast, IS asserted byte-exact (a) — the client
// needs the real sibling order to reconstruct xformOp order (translate→rotate→
// scale), and our real per-parent sibling index matches C++'s exactly.
//
// A regression that reverts to emitting the raw per-delta authoring rank
// (colliding across prims) trips assertion (b); a regression that scrambles the
// (version, authoring) ordering trips (c); a regression that reverts sibling
// order to the hardcoded 0 trips (a).

/// Load and parse a committed fixture under `tools/fixtures/` as JSON.
fn load_fixture_json(name: &str) -> serde_json::Value {
    let path = format!("{}/tools/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse fixture {path}: {e}"))
}

/// Where a (possibly partial) replay of the captured fixture has got to, so a
/// caller can PAUSE the replay at an arbitrary server sequence — observe the
/// service mid-history — and then resume exactly where it stopped.
///
/// `client_ts` is carried across resumes so a paused-and-resumed replay issues
/// the identical header sequence a straight-through replay does.
struct GzeReplayCursor {
    /// The live generation's `object_id`, once a CreateLayer has landed.
    object_id: Option<u64>,
    /// Server sequence on that generation. A CreateLayer overwrite restarts it at 1.
    current_seq: u64,
    /// Monotone client timestamp stamped on each UpdateLayer header.
    client_ts: u64,
}

impl Default for GzeReplayCursor {
    fn default() -> Self {
        Self {
            object_id: None,
            current_seq: 0,
            client_ts: 1,
        }
    }
}

/// The captured fixture's per-delta `kind` strings, in capture order. Lets a
/// test locate the surviving generation's first delta without decoding bodies.
fn gze_delta_kinds() -> Vec<String> {
    load_fixture_json("gze_inbound_deltas.json")
        .as_array()
        .expect("deltas fixture must be a JSON array")
        .iter()
        .map(|d| {
            d["kind"]
                .as_str()
                .expect("delta kind must be a string")
                .to_owned()
        })
        .collect()
}

/// Replay the captured fixture's deltas at `range` through the in-process
/// service, in capture order, advancing `cursor`. See
/// [`replay_gze_captured_deltas`] for the straight-through form.
async fn replay_gze_delta_range(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    range: std::ops::Range<usize>,
    cursor: &mut GzeReplayCursor,
) {
    use base64::{engine::general_purpose::STANDARD as B64, Engine};

    let deltas = load_fixture_json("gze_inbound_deltas.json");
    let deltas = deltas
        .as_array()
        .expect("deltas fixture must be a JSON array");
    assert!(
        range.end <= deltas.len(),
        "replay range {range:?} exceeds the fixture's {} deltas",
        deltas.len()
    );
    for i in range {
        let d = &deltas[i];
        let kind = d["kind"].as_str().expect("delta kind must be a string");
        let body = B64
            .decode(d["b64"].as_str().expect("delta b64 must be a string"))
            .unwrap_or_else(|e| panic!("delta {i} b64 decode: {e}"));
        match kind {
            "CreateLayer" => {
                let stream = streaming_create(vec![
                    make_create_header(session_id, layer),
                    make_create_data(body),
                ]);
                let resp = handle_create_layer(svc, make_request(stream))
                    .await
                    .unwrap_or_else(|e| panic!("delta {i} CreateLayer failed: {e}"));
                cursor.object_id = Some(resp.into_inner().object_id);
                // A CreateLayer overwrite starts a fresh generation whose
                // per-object sequence restarts at 1.
                cursor.current_seq = 1;
            }
            "UpdateLayer" => {
                let oid = cursor
                    .object_id
                    .expect("UpdateLayer before any CreateLayer");
                let stream = streaming_update(vec![
                    update_header(session_id, layer, oid, "gze-writer", cursor.client_ts),
                    update_data(body),
                ]);
                handle_update_layer(svc, make_request(stream))
                    .await
                    .unwrap_or_else(|e| panic!("delta {i} UpdateLayer failed: {e}"));
                cursor.current_seq += 1;
                cursor.client_ts += 1;
            }
            other => panic!("delta {i} unexpected kind {other}"),
        }
    }
}

/// Replay the 36 REAL captured Kit deltas (`tools/fixtures/gze_inbound_deltas.json`)
/// through the in-process service, in capture order: the leading CreateLayer
/// overwrite(s) (which land a fresh generation) then the UpdateLayer deltas —
/// exactly as the C++ oracle was captured (harness `_replay_and_late_join_catchup`).
///
/// Returns `(object_id, current_seq)` for the live generation: `current_seq` is
/// the server sequence after the replay, i.e. 1 (the surviving CreateLayer) plus
/// one per UpdateLayer.
async fn replay_gze_captured_deltas(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
) -> (u64, u64) {
    let total = gze_delta_kinds().len();
    let mut cursor = GzeReplayCursor::default();
    replay_gze_delta_range(svc, session_id, layer, 0..total, &mut cursor).await;
    (
        cursor
            .object_id
            .expect("fixture must contain at least one CreateLayer"),
        cursor.current_seq,
    )
}

/// The golden's `setFields` as an ordered `(sectionId, keyName)` sequence,
/// sorted by the C++ server's emitted `setOrder`.
fn golden_field_sequence(golden: &serde_json::Value) -> Vec<(u64, String)> {
    let mut sorted: Vec<(u64, u64, String)> = golden["normalized_signature"]["setFields"]
        .as_array()
        .expect("golden setFields array")
        .iter()
        .map(|f| {
            (
                f["setOrder"].as_u64().expect("golden setOrder"),
                f["sectionId"].as_u64().expect("golden field sectionId"),
                f["keyName"].as_str().expect("golden keyName").to_string(),
            )
        })
        .collect();
    sorted.sort();
    sorted.into_iter().map(|(_, s, k)| (s, k)).collect()
}

/// Our emitted `setFields` as an ordered `(sectionId, keyName)` sequence, sorted
/// by the `set_order` we put on the wire.
fn emitted_field_sequence(decoded: &OwnedDelta) -> Vec<(u64, String)> {
    let mut ours: Vec<(u64, u64, String)> = decoded
        .set_fields
        .iter()
        .map(|f| (f.set_order, f.section_id, f.key_name.clone()))
        .collect();
    ours.sort();
    ours.into_iter().map(|(_, s, k)| (s, k)).collect()
}

/// Assert the emitted `set_order` values are the dense unique range `0..N-1`.
///
/// Both properties are load-bearing for the Kit client, which keys catch-up
/// fields into an ordered `std::map<pair<saveCount, setOrder>>` under ONE shared
/// `saveCount`: a duplicate `setOrder` collides on `emplace` and the field is
/// silently dropped.
fn assert_dense_unique_set_orders(decoded: &OwnedDelta, context: &str) {
    let emitted: Vec<u64> = decoded.set_fields.iter().map(|f| f.set_order).collect();
    let mut uniq = emitted.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        emitted.len(),
        "{context}: emitted set_order values MUST be globally unique across the whole catch-up \
         message — a duplicate collides on the Kit client's saveCount-keyed map and the field is \
         silently dropped (exactly what a raw-per-delta-authoring-rank emission produces); got \
         {emitted:?}"
    );
    assert_eq!(
        uniq,
        (0..emitted.len() as u64).collect::<Vec<u64>>(),
        "{context}: emitted set_order must be the dense range 0..N-1, got {emitted:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_catchup_matches_cpp_golden_oracle_offline() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "gze-golden-layer";
    // Create the session only; the fixture's leading CreateLayer entries
    // establish (and overwrite to a fresh generation) the layer themselves.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("gze-golden-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ── Replay the 36 real captured deltas in order. ──
    let (object_id, _current_seq) = replay_gze_captured_deltas(&svc, &session_id, layer).await;

    // ── Late-join a FRESH subscriber (object_id=0, seq=0) in Snapshot mode and
    //    capture the single corrected full-state catch-up message. ──
    let sub_req = subscribe_request(&session_id, layer, 0, 0, "gze-late-joiner");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("late-join subscribe must succeed");
    let mut stream = response.into_inner();
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "catch-up must carry the live object_id"
    );
    assert!(
        header.total_size > 0,
        "catch-up must carry a non-empty full-state body"
    );

    let decoded = decode_broadcast_delta(&body);

    // ── Parse the committed C++ golden signature. ──
    let golden = load_fixture_json("gze_cpp_catchup_oracle.json");
    let sig = &golden["normalized_signature"];
    let golden_struct = sig["structCommands"]
        .as_array()
        .expect("golden structCommands array");

    // Golden section_order keyed by section_id.
    let golden_section_order: HashMap<u64, u32> = golden_struct
        .iter()
        .map(|c| {
            (
                c["sectionId"].as_u64().expect("golden sectionId"),
                c["sectionOrder"].as_u64().expect("golden sectionOrder") as u32,
            )
        })
        .collect();

    // Our emitted sections (DiffChangeSection commands).
    let our_sections: Vec<(u64, u32)> = decoded
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffChangeSection(s) => Some((s.section_id, s.section_order)),
            _ => None,
        })
        .collect();

    // ── (a) sectionOrder BYTE-EXACT: every emitted section's sibling order
    //        equals the golden's for that section_id, and the emitted section
    //        SET equals the golden's (all 21 sections). ──
    assert_eq!(
        our_sections.len(),
        golden_struct.len(),
        "emitted section count ({}) must equal the golden's ({})",
        our_sections.len(),
        golden_struct.len()
    );
    for (sid, order) in &our_sections {
        let expected = golden_section_order.get(sid).unwrap_or_else(|| {
            panic!("emitted section {sid} is absent from the C++ golden signature")
        });
        assert_eq!(
            order, expected,
            "section {sid}: emitted sectionOrder {order} must byte-match the C++ golden {expected}"
        );
    }

    // ── (b) set_order GLOBAL UNIQUENESS: all emitted set_orders distinct
    //        (dense 0..N-1). A raw-authoring-rank regression collides here. ──
    assert_dense_unique_set_orders(&decoded, "from-empty snapshot catch-up");

    // ── (c) set_order RELATIVE-ORDER PARITY: sorting our setFields by set_order
    //        yields the SAME (sectionId, keyName) sequence as sorting the
    //        golden's by its setOrder. (NOT byte-identical integers — the client
    //        re-bases; only relative order matters.) All 45 fields match. ──
    let our_seq = emitted_field_sequence(&decoded);
    let golden_seq = golden_field_sequence(&golden);

    assert_eq!(
        our_seq.len(),
        golden_seq.len(),
        "emitted field count ({}) must equal the golden's ({})",
        our_seq.len(),
        golden_seq.len()
    );
    assert_eq!(
        our_seq, golden_seq,
        "sorting our setFields by set_order must reproduce the SAME (sectionId, keyName) sequence \
         as the C++ golden sorted by its setOrder — relative apply order must match (absolute \
         integers deliberately need not, because the Kit client re-bases)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.3pe — RECONNECT coalesced catch-up over the same real captured traffic.
//
// Sibling of `snapshot_catchup_matches_cpp_golden_oracle_offline`. That test
// covers the FROM-EMPTY (`plan_faithful_snapshot`) builder; this one covers the
// RECONNECT (`plan_faithful_coalesced`) builder, which until b.3pe emitted the
// STORED CUMULATIVE `set_order` verbatim. That cursor is bumped in WIRE-ARRIVAL
// order, and the Kit writer serializes each commit's fields out of a
// `std::unordered_map`, so arrival order is hash-bucket order — uncorrelated
// with the writer's declared authoring rank. This fixture is exactly that
// traffic (36 REAL captured Kit deltas), so it is the honest place to catch it;
// a synthetic single-section fixture would not (that is what hid b.gze/b.zju).
//
// WHAT THE GOLDEN CAN AND CANNOT VALIDATE HERE — read before extending:
//
//   CANNOT: the coalesced CONTENT (WHICH fields are included). The legacy C++
//   server has NO coalesced-reconnect analog — it only ever produced the
//   from-empty catch-up captured in the golden — so there is no oracle for the
//   `version > have_version` membership filter. Nothing below asserts our
//   emitted field SET equals any golden subset; the set is only used to select
//   which golden entries to compare order against.
//
//   CAN: the RELATIVE ORDER of the fields we DO include, and their global
//   uniqueness. The golden fixes one total order over all 45 fields. Our
//   coalesced emission is a subset of those same fields ordered by the SAME key
//   the snapshot builder uses — so post-fix it must be a SUBSEQUENCE of the
//   golden order. Pre-fix (cumulative-cursor emission) it is not, because the
//   cursor is hash-bucket ordered.
//
// Both assertions therefore fail against the pre-fix builder and pass after.

/// Mid-window reconnect watermark. The replay lands `current_seq = 35` on the
/// surviving generation (1 CreateLayer + 34 UpdateLayer), so this sits well
/// inside the window: enough commits behind it that the coalesced slice is a
/// PROPER subset of the golden's 45 fields, and enough ahead that the slice is
/// multi-field and its ordering is actually observable.
const GZE_RECONNECT_HAVE_VERSION: u64 = 20;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coalesced_reconnect_catchup_set_order_matches_cpp_golden_subsequence_offline() {
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "gze-coalesced-layer";
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("gze-coalesced-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ── Replay the same 36 real captured deltas the golden was captured over. ──
    let (object_id, current_seq) = replay_gze_captured_deltas(&svc, &session_id, layer).await;
    assert!(
        GZE_RECONNECT_HAVE_VERSION > 0 && GZE_RECONNECT_HAVE_VERSION < current_seq,
        "the reconnect watermark {GZE_RECONNECT_HAVE_VERSION} must be strictly inside the \
         replayed window (0, {current_seq}) — if the fixture's delta count changed, retune it"
    );

    // ── Reconnect at a MID-WINDOW have_version on the SAME object_id. This is a
    //    reconnector, not a late joiner, so it routes through the coalesced
    //    builder (both reconnect regimes converge on it under Snapshot mode). ──
    let sub_req = subscribe_request(
        &session_id,
        layer,
        object_id,
        GZE_RECONNECT_HAVE_VERSION,
        "gze-reconnector",
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("reconnect subscribe must succeed");
    let mut stream = response.into_inner();
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "coalesced catch-up must carry the live object_id"
    );
    assert_eq!(
        header.sequence_num, current_seq,
        "coalesced catch-up header must carry the current server seq"
    );
    assert!(
        header.total_size > 0,
        "coalesced catch-up must carry a non-empty body"
    );

    let decoded = decode_broadcast_delta(&body);
    // Envelope: an advanced reconnector must never get a base_version=0 reset.
    assert!(
        decoded.is_diff,
        "coalesced reconnect catch-up must be is_diff=true"
    );
    assert_eq!(
        decoded.base_version, GZE_RECONNECT_HAVE_VERSION,
        "coalesced reconnect catch-up base_version must equal have_version (never 0)"
    );

    // ── (a) GLOBAL UNIQUENESS / dense 0..N-1 across the delivered coalesced
    //        message. Pre-fix this carried the cumulative cursor, which is a
    //        strictly-increasing per-SetField accumulator and so was ALREADY
    //        unique — the dense re-index exists for ORDERING FIDELITY, not to
    //        repair uniqueness (SDD §12.8). The cursor encodes wire-arrival
    //        (hash-bucket) order, not the writer's declared authoring rank, so
    //        emitting it verbatim re-orders every multi-field commit. Sorting on
    //        the retained rank restores the writer's order and the dense index
    //        then makes uniqueness hold by construction rather than by relying
    //        on the accumulator invariant — which is what this asserts. ──
    assert_dense_unique_set_orders(&decoded, "coalesced reconnect catch-up");

    // ── (b) RELATIVE-ORDER PARITY against the golden, restricted to the
    //        subsequence of golden fields we actually emitted. See the header
    //        comment: the golden cannot adjudicate WHICH fields belong in a
    //        coalesced diff, only the relative order of the ones present. ──
    let our_seq = emitted_field_sequence(&decoded);
    let golden_seq = golden_field_sequence(&load_fixture_json("gze_cpp_catchup_oracle.json"));

    // Non-vacuity guard: a coalesced diff that emitted nothing, or that emitted
    // everything (i.e. degenerated into the full from-empty snapshot), would let
    // the ordering assertion below pass without testing the coalesced path.
    assert!(
        our_seq.len() > 1 && our_seq.len() < golden_seq.len(),
        "the coalesced slice at have_version={GZE_RECONNECT_HAVE_VERSION} must be a PROPER, \
         multi-field subset of the golden's {} fields for this comparison to mean anything — \
         got {} emitted: {our_seq:?}",
        golden_seq.len(),
        our_seq.len()
    );

    // Every field we emit must be one the golden knows about, otherwise the
    // filtered comparison below silently compares against a shorter list.
    let golden_set: HashSet<(u64, String)> = golden_seq.iter().cloned().collect();
    for entry in &our_seq {
        assert!(
            golden_set.contains(entry),
            "emitted field {entry:?} is absent from the C++ golden signature — the fixture and \
             the golden have drifted apart"
        );
    }

    // The comparison: the golden's total order, restricted to the fields our
    // coalesced diff carried, must equal our emitted order.
    let our_set: HashSet<(u64, String)> = our_seq.iter().cloned().collect();
    let golden_subsequence: Vec<(u64, String)> = golden_seq
        .into_iter()
        .filter(|e| our_set.contains(e))
        .collect();
    assert_eq!(
        our_seq, golden_subsequence,
        "sorting our coalesced setFields by set_order must reproduce the C++ golden's relative \
         order restricted to those same fields. A verbatim cumulative-set_order emission (the \
         pre-b.3pe behaviour) reproduces the writer's hash-bucket ARRIVAL order instead, which \
         disagrees with the golden's authoring order"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.xvq — IN-WINDOW reconnect over the REAL captured traffic: the delivered
// coalesced diff, FOLDED onto what the client already holds, reproduces the full
// net-live scene.
//
// Provenance. The acceptance harness check `reconnect_matches_full_net_live_scene`
// failed deterministically on dev (rc.78, clean store) reconnecting at
// have_version=32 against current=35 over THIS fixture: ONE message, is_diff=true,
// base_version=32, carrying only the net-live content with version > 32 (7
// sections / 16 fields) where the harness expected the whole scene (21 / 45). The
// SERVICE is right. SDD §12.10 makes message #1 of an in-window reconnect an
// INCREMENTAL coalesced diff keyed to `have_version`, so the client's scene is
// `fold(state@have_version, message #1)` — never message #1 alone. Comparing the
// payload directly is only correct for the from-empty builder.
//
// Why this test exists. That verdict was reached against a harness run on a
// deployed service, which neither CI nor the sandbox can perform — which is precisely why
// b.xvq had to be its own ticket. This makes the guarantee durable in-process,
// over the SAME real captured traffic (a synthetic single-section fixture is what
// hid b.gze and b.zju):
//
//   • the baseline is what a client that joined via full-state catch-up at
//     `have_version` holds — i.e. the NET-LIVE scene at that version, built by
//     PAUSING the replay there and draining a from-empty late-join into a mirror
//     `ObjectState` (the same oracle the harness builds its reference scene
//     with). It is deliberately NOT the only shape a conforming client can be
//     in: one that streamed per-commit from 0 additionally carries
//     stored-but-unreachable ghosts, which this baseline never holds;
//   • the replay then resumes to the end, that mirror reconnects IN-WINDOW at
//     `have_version`, and the delivered body is folded in with the READ-SIDE
//     `DiffDeleteSection` semantics a Kit client applies (`apply_catchup_read_side`);
//   • the folded scene must equal the full net-live scene, cross-checked against
//     BOTH oracles: a from-empty late-join after the full replay (the harness's
//     reference) and the server's root-reachable materialized state (what
//     `snapshot_in_window_reconnect_establishes_correct_scene` uses).
//
// Path under test: `catch_up_in_window` → `catch_up_in_window_snapshot` →
// `compute_faithful_coalesced_at_cut` → `DiffEngine::plan_faithful_coalesced`, pinned by
// the `live_subscribe_regime_total{regime="in_window", catchup_mode="legacy"}`
// tick asserted per row — the same series that ticked on dev.
//
// DISCRIMINATION — why this cannot pass vacuously. Every row ALSO asserts the
// harness's naive comparison — the delivered body applied to an EMPTY state —
// does NOT reproduce the net-live scene, and covers strictly fewer sections. The
// fold is therefore provably load-bearing: a regression that stopped folding, or
// a builder that silently degenerated into the from-empty full-state snapshot,
// trips one half or the other. That holds unconditionally because every row here
// reconnects with a non-zero lag; a zero-lag row would route through the
// `up_to_date` regime instead and needs a different assertion shape, so none is
// listed.
//
// SET_ORDER — a second, independent property every row checks. The fold model
// upserts fields by `(section_id, key_name)` and is therefore BLIND to
// `set_order`; the Kit client is not. It keys catch-up fields into an ordered
// `std::map<pair<saveCount, setOrder>>` under ONE shared `saveCount`, so a
// duplicate `setOrder` collides on `emplace` and the field is silently dropped
// (SDD §12.8 / §12.9) — real edit loss that a fold-only assertion cannot see,
// because both scenes would still agree on the keys that survived. Every row
// therefore also runs `assert_dense_unique_set_orders` over all THREE bodies it
// decodes: the delivered coalesced body, the from-empty BASELINE body taken at
// that row's `have_version`, and the from-empty REFERENCE body taken at the end
// of the capture. Both builders funnel through
// `DiffEngine::dense_reindex_by_authoring` (`src/domain/diff.rs` —
// `plan_faithful_coalesced` and `plan_faithful_snapshot` each call it), so all
// three owe the dense unique range `0..N-1` and they are checked as one rule
// rather than one being assumed from the other. The baseline body is what makes
// the sweep pay for itself on this property: it is a DIFFERENT mid-history
// snapshot per row (5 distinct bodies), whereas the reference body is the same
// seq-35 snapshot on all five rows — already covered once by
// `snapshot_catchup_matches_cpp_golden_oracle_offline` — and is re-checked here
// only because it is nearly free.
//
// SIBLING ORDER — a third property, which the scene fold is also blind to (it
// compares section sets, never child order). Every row folds the baseline body
// and the delivered body, in wire order, through BOTH client reader models in
// `tests/common/sibling_model.rs` — the Kit client's, which reads a catch-up
// `sectionOrder` as a key with displacement, and the C++ client library's,
// which reads it as an index and renumbers — and requires each to hold the
// server's order in every USD child list, as the same reader does from the
// from-empty reference body. Measured on this capture: all five rows hold that
// order under both readers whether or not a touched parent's other children
// are restated, so this traffic does not reach a shape where the two readings
// part. The discriminating fixtures are synthetic, gapped histories
// (`sibling_order_parity_test.rs`, and the reader tests in
// `src/domain/delta.rs`).
//
// The `have_version` rows: 1 (whole-history collapse — the client holds only the
// CreateLayer's root prim), 3, 5 and 6 (the `DiffDeleteSection` RETRACTION arm:
// the fixture's `CommandDeleteSection`-bearing deltas commit at seq 3, 4, 6 and
// 7, so a reconnector at 3, 5 or 6 is owed retractions for sections it still
// holds), and 32 (b.xvq's exact dev configuration, lag 3 behind the final seq
// 35). Row 5 is the depth the internal acceptance harness reaches at its lag-30
// snapshot-reconnect scenario: SDD
// §12.10 quotes its figures, but they were observed only by that harness against
// a live endpoint, so nothing committed re-derived them until this row.
//
// Each row is its OWN `#[tokio::test]` (`…_have_version_{1,3,5,6,32}`) over one
// shared body, not a loop inside a single test. Three reasons carry it: a loop
// lets row 1's failure MASK rows 3/5/6/32, forfeits nextest's per-test process
// ISOLATION (a process-global left dirty by one row leaks into the next), and
// serializes work nextest would otherwise SCHEDULE in parallel. A fourth,
// stated precisely because the obvious version of it is wrong: the split cuts
// each process's exposure to nextest's 120 s `terminate-after`
// (`.config/nextest.toml`, 3 × 40 s) by 5×, and makes a kill that does happen
// name ONE row. It does NOT put the worst case under that ceiling — a single
// row already waits 6 × `SUBSCRIBER_DEADLINE` = 180 s if every stream read
// times out, and 6 is a floor, since `recv_message` spends a fresh budget per
// body chunk on top of the header. A wedged row is killed either way; the split
// buys attribution and blast radius, not immunity.
//
// Measured here (sections/fields), current seq 35, net-live scene 21/45. EVERY
// column is PINNED per row, not merely bounded — `expected_delivered`,
// `expected_baseline`, the shared `NET_LIVE_SCENE` for the folded column, and
// `expected_retractions` — because the fixture is committed and immutable, so
// these are exact and a table that only documented them could rot silently
// while the fold still passed. `delivered` sections count the DISTINCT section
// ids named by non-delete struct commands (the acceptance harness's `present`);
// `delivered` fields count `SetField` ENTRIES; the `baseline` and `folded`
// columns are STATE counts (sections held, distinct `(section_id, key_name)`
// pairs held). Time samples are absent from this capture throughout (asserted,
// not assumed).
//
//   have_version │ delivered │ baseline │ folded │ retracted
//   ─────────────┼───────────┼──────────┼────────┼───────────
//              1 │     20/45 │      1/0 │  21/45 │         0
//              3 │     20/40 │     7/19 │  21/45 │         6
//              5 │     20/40 │     8/21 │  21/45 │         7
//              6 │     17/28 │     7/19 │  21/45 │         1
//             32 │      8/16 │    14/30 │  21/45 │         0
//
// The `have_version=32` row is the dev failure's configuration. The dev harness
// saw 7/16 delivered against a 21/45 reference from a build that sent only the
// changed sections; this row delivers 8/16, because the coalesced diff also
// restates every other listed child of a parent with a new or changed child,
// and it folds to 21/45. The `have_version=5` row reproduces, in-process, the
// three figures SDD §12.10 attributes to the live-endpoint harness (baseline
// 8/21, delivered 20/40, 7 retractions). Every
// number in this table is quoted by SDD §12.10 ("Observed rows", plus the
// retracted column); re-capturing the fixture means updating this table,
// `IN_WINDOW_FOLD_CASES` / `NET_LIVE_SCENE`, and that SDD sentence together —
// the assertions print the observed values to update them from.
//
// WHAT THIS FAMILY DOES NOT DISCRIMINATE — read-side `DiffDeleteSection`
// semantics. On THIS capture every retracted prim's `(parent, name)` is
// immediately re-taken by a freshly-id'd sibling in the SAME body, so
// `apply_diff_change_section`'s duplicate-sibling-name cascade (SDD §9.1 step 4a)
// evicts the stale id even for a client that ignored the retraction: the fold
// converges either way. And the capture never produces the stored-but-unreachable
// ghost arm (b) of `DiffEngine::plan_faithful_coalesced` at all (there the stored
// section set equals the net-live set — hence the 0 in the retracted column at
// have_version 1 and 32). `snapshot_in_window_reconnect_real_traffic_plus_ghost_
// retraction_discriminates_read_side_semantics` below closes both gaps by
// appending two in-process commits authored against THIS capture's real prim ids.

/// The `folded` / reference column of the banner table, shared by every row:
/// this capture's net-live scene is 21 sections / 45 fields whatever watermark
/// the reconnector comes back at. Asserted per row against the from-empty
/// reference oracle, so "all four fold to 21/45" (SDD §12.10) is checked rather
/// than described. Sections = distinct section ids in the scene; fields =
/// distinct `(section_id, key_name)` pairs.
const NET_LIVE_SCENE: (usize, usize) = (21, 45);

/// One `have_version` row of the in-window fold sweep.
struct InWindowFoldCase {
    /// The reconnector's retained version.
    have_version: u64,
    /// EXACT `(sections, fields)` the delivered coalesced body carries — the
    /// `delivered` column of the banner table.
    ///
    /// SECTIONS counts the DISTINCT section ids named by NON-delete struct
    /// commands in the body (`emitted_present_section_ids`) — deliberately the
    /// same reduction the internal acceptance harness calls `present`, so this
    /// number and the
    /// one the harness prints for the same payload are the same measurement and
    /// stay comparable. Retractions are excluded and counted separately by
    /// `expected_retractions`.
    ///
    /// FIELDS counts `SetField` ENTRIES in the body, not distinct keys: a
    /// builder that emitted one `(section_id, key_name)` twice is a bug a
    /// deduplicated view would hide.
    expected_delivered: (usize, usize),
    /// EXACT `(sections, fields)` of the baseline the reconnector holds at
    /// `have_version` — the `baseline` column of the banner table.
    ///
    /// Counted as STATE, not as a payload: sections present in the mirror, and
    /// distinct `(section_id, key_name)` pairs it holds. The baseline is built
    /// by pausing the replay at `have_version` and draining a from-empty
    /// late-join into the mirror, so this pins WHAT THE CLIENT ALREADY HAS —
    /// the half of `fold(state@have_version, message #1)` that the delivered
    /// body is not allowed to re-send.
    expected_baseline: (usize, usize),
    /// EXACT number of `DiffDeleteSection` entries the delivered body must
    /// carry — not a `> 0` / `== 0` predicate. These are measurements against a
    /// COMMITTED, immutable fixture (`tools/fixtures/gze_inbound_deltas.json`),
    /// so the counts are a property of this traffic and pinning them is what
    /// makes the retraction counts in SDD §12.10's "Observed rows" checkable
    /// rather than an unverified claim. A re-capture of the fixture changes
    /// these numbers: update this column (and the banner table, and those SDD
    /// rows) from the observed values the assertion prints.
    expected_retractions: usize,
}

/// Indexed by the five `#[tokio::test]` wrappers below — one row each, never
/// swept in a loop (see the banner).
const IN_WINDOW_FOLD_CASES: [InWindowFoldCase; 5] = [
    // Whole-history collapse. Even here the delivery is a PROPER subset: the
    // root prim the surviving CreateLayer landed carries version 1, so the
    // `version > have_version` filter excludes it and only the fold restores it
    // (delivered 20 sections, net-live 21). Nothing is retracted: at
    // have_version=1 the client holds only that root prim, and every section
    // the capture later deletes was also CREATED after 1, so the builder drops
    // it rather than retracting something the client never had.
    InWindowFoldCase {
        have_version: 1,
        expected_delivered: (20, 45),
        expected_baseline: (1, 0),
        expected_retractions: 0,
    },
    InWindowFoldCase {
        have_version: 3,
        expected_delivered: (20, 40),
        expected_baseline: (7, 19),
        expected_retractions: 6,
    },
    // The acceptance harness's committed lag-30 depth (`_SNAPSHOT_RECONNECT_LAGS`,
    // 35 − 30 = 5). SDD §12.10 quotes 8/21 baseline, 20/40 delivered and 7
    // retractions here from a LIVE-ENDPOINT harness run; this row is the
    // in-process re-derivation, so those figures stop resting on an environment
    // neither CI nor the sandbox can reach.
    InWindowFoldCase {
        have_version: 5,
        expected_delivered: (20, 40),
        expected_baseline: (8, 21),
        expected_retractions: 7,
    },
    InWindowFoldCase {
        have_version: 6,
        expected_delivered: (17, 28),
        expected_baseline: (7, 19),
        expected_retractions: 1,
    },
    // b.xvq's dev configuration. The capture's last section delete commits at
    // seq 7, far below 32, so the reconnect gap (33..=35) contains none. The
    // failing dev harness run reported 7/16 from a build that sent only the
    // changed sections; the eighth section here is a listed sibling restated
    // with its parent's changed children.
    InWindowFoldCase {
        have_version: 32,
        expected_delivered: (8, 16),
        expected_baseline: (14, 30),
        expected_retractions: 0,
    },
];

/// `live_subscribe_regime_total{regime, service, catchup_mode}`. Read WITH the
/// `catchup_mode` pair: `sample_value` is a subset match, so a mode-less read
/// would resolve whichever series prints first (the baseline late-join in each
/// row ticks `brand_new`/`snapshot` on the same metric name).
fn read_regime_total(metrics: &Metrics, service: &str, regime: &str, mode: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_regime_total",
        &[
            ("regime", regime),
            ("service", service),
            ("catchup_mode", mode),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// A client scene's fields as a comparable `(section_id, key_name) → value` map.
fn client_field_map(state: &ObjectState) -> BTreeMap<(u64, String), FieldValue> {
    state
        .fields
        .iter()
        .map(|((sid, key), f)| ((*sid, key.clone()), f.value.clone()))
        .collect()
}

/// A client scene's time samples as a comparable `(section_id, time) → value`
/// map — the animation-content analogue of `client_field_map`. SDD §12.10:
/// "the builder emits `changed_time_samples` alongside changed fields, so a
/// fold that compares only presence and fields leaves animation content
/// unguarded".
fn client_time_sample_map(
    state: &ObjectState,
) -> BTreeMap<(u64, ordered_float::OrderedFloat<f64>), FieldValue> {
    state
        .time_samples
        .iter()
        .map(|((sid, time), ts)| ((*sid, *time), ts.value.clone()))
        .collect()
}

/// The server's materialized fields restricted to the net-live (root-reachable)
/// sections — the server-side view of the same map.
fn net_live_field_map(
    state: &ObjectState,
    live: &HashSet<u64>,
) -> BTreeMap<(u64, String), FieldValue> {
    state
        .fields
        .iter()
        .filter(|((sid, _), _)| live.contains(sid))
        .map(|((sid, key), f)| ((*sid, key.clone()), f.value.clone()))
        .collect()
}

/// Assert two client scenes hold the same `(section_id, key_name, value)` set,
/// reporting the first divergence rather than dumping both maps.
fn assert_same_field_sets(
    actual: &BTreeMap<(u64, String), FieldValue>,
    expected: &BTreeMap<(u64, String), FieldValue>,
    context: &str,
) {
    let actual_keys: HashSet<&(u64, String)> = actual.keys().collect();
    let expected_keys: HashSet<&(u64, String)> = expected.keys().collect();
    let missing: Vec<_> = expected_keys.difference(&actual_keys).collect();
    let extra: Vec<_> = actual_keys.difference(&expected_keys).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{context}: field KEY sets diverge — missing {} {missing:?}, unexpected {} {extra:?}",
        missing.len(),
        extra.len()
    );
    for (key, expected_value) in expected {
        let actual_value = &actual[key];
        assert_eq!(
            actual_value, expected_value,
            "{context}: field {key:?} value diverges"
        );
    }
}

/// The outcome of a from-empty late-join: the scene a correct client holds, the
/// generation and sequence it observed, and the DECODED body it got there from.
///
/// The decoded body is carried rather than dropped so callers can assert
/// PAYLOAD-shaped properties of the from-empty builder — today
/// `assert_dense_unique_set_orders`, which the folded-scene comparison is blind
/// to by construction (the fold upserts by `(section_id, key_name)` and never
/// looks at `set_order`).
struct FromEmptyJoin {
    scene: ObjectState,
    object_id: u64,
    sequence_num: u64,
    decoded: OwnedDelta,
}

/// Late-join a FRESH subscriber (object_id=0, seq=0) and fold the delivered
/// from-empty full-state catch-up into a new `ObjectState` — exactly what a
/// correct client holds after joining at this moment. This is the harness's
/// reference-scene oracle, and (paused mid-replay) the reconnector's baseline.
async fn gze_from_empty_client_scene(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    client_id: &str,
) -> FromEmptyJoin {
    let response =
        handle_subscribe_read_layer(svc, subscribe_request(session_id, layer, 0, 0, client_id))
            .await
            .expect("from-empty late-join subscribe must succeed");
    let mut stream = response.into_inner();
    let (header, body) = recv_message(&mut stream).await;
    assert!(
        header.total_size > 0,
        "from-empty catch-up must carry a non-empty full-state body"
    );
    let decoded = decode_broadcast_delta(&body);
    assert_eq!(
        decoded.base_version, 0,
        "from-empty catch-up must be base_version=0"
    );
    let mut scene = ObjectState::default();
    apply_catchup_read_side(&mut scene, &body);
    recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;
    drop(stream);
    FromEmptyJoin {
        scene,
        object_id: header.object_id,
        sequence_num: header.sequence_num,
        decoded,
    }
}

/// Look the row up BY `have_version` rather than by table index, so reordering
/// `IN_WINDOW_FOLD_CASES` can never silently point a `…_have_version_N` test at
/// a different row than its name claims.
fn in_window_fold_case(have_version: u64) -> &'static InWindowFoldCase {
    IN_WINDOW_FOLD_CASES
        .iter()
        .find(|c| c.have_version == have_version)
        .unwrap_or_else(|| panic!("no IN_WINDOW_FOLD_CASES row for have_version={have_version}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_1() {
    run_in_window_fold_case(in_window_fold_case(1)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_3() {
    run_in_window_fold_case(in_window_fold_case(3)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_5() {
    run_in_window_fold_case(in_window_fold_case(5)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_6() {
    run_in_window_fold_case(in_window_fold_case(6)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_folds_to_net_live_scene_have_version_32() {
    run_in_window_fold_case(in_window_fold_case(32)).await;
}

async fn run_in_window_fold_case(case: &InWindowFoldCase) {
    let have_version = case.have_version;
    let tv = PooledTestValkey::setup().await;
    // Roomy MAXLEN so the 35-commit history is never trimmed — the reconnector's
    // start id `{have_version+1}-0` must still be present for the in_window branch.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = format!("gze-in-window-fold-layer-{have_version}");
    let layer = layer.as_str();
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create(
            &format!("gze-in-window-fold-session-{have_version}"),
            "/stage",
            "test-user",
        )
        .await
        .expect("session creation must succeed");

    // ── Locate the surviving generation. The fixture opens with CreateLayer
    //    entries, each of which lands a fresh generation and RESETS the sequence
    //    to 1; only the last one's object survives, so the delta index that
    //    leaves the server at `have_version` is `last_create + have_version`. ──
    let kinds = gze_delta_kinds();
    let last_create = kinds
        .iter()
        .rposition(|k| k == "CreateLayer")
        .expect("fixture must contain a CreateLayer");
    assert!(
        kinds[last_create + 1..].iter().all(|k| k == "UpdateLayer"),
        "every delta after the last CreateLayer must be an UpdateLayer for the \
         index→sequence mapping below to hold"
    );
    let pause_at = last_create + have_version as usize;
    assert!(
        pause_at < kinds.len(),
        "have_version={have_version} must sit strictly inside the fixture's replay"
    );

    // ── Replay up to `have_version`, then capture the reconnector's BASELINE:
    //    what a correct client holds at that moment. ──
    let mut cursor = GzeReplayCursor::default();
    replay_gze_delta_range(&svc, &session_id, layer, 0..pause_at, &mut cursor).await;
    assert_eq!(
        cursor.current_seq, have_version,
        "paused replay must leave the server at have_version"
    );
    let baseline_join =
        gze_from_empty_client_scene(&svc, &session_id, layer, "fold-baseline-joiner").await;
    let baseline_object_id = baseline_join.object_id;
    let mut mirror = baseline_join.scene;
    assert_eq!(
        baseline_join.sequence_num, have_version,
        "the baseline late-join must observe exactly the paused sequence"
    );
    let baseline_sections = mirror.sections.len();
    let baseline_fields = mirror.fields.len();
    // The `baseline` column of the banner table, PINNED (see
    // `InWindowFoldCase::expected_baseline`). Counted as STATE: sections held,
    // and distinct `(section_id, key_name)` field pairs held.
    assert_eq!(
        (baseline_sections, baseline_fields),
        case.expected_baseline,
        "have_version={have_version}: the reconnector's baseline must hold exactly {:?} \
         (sections, fields) on this committed capture, observed ({baseline_sections}, \
         {baseline_fields}). These are measurements against the COMMITTED, immutable fixture \
         tools/fixtures/gze_inbound_deltas.json; if it was re-captured, update \
         IN_WINDOW_FOLD_CASES, the banner table above it, and SDD §12.10's \"Observed rows\" \
         sentence TOGETHER from the observed values.",
        case.expected_baseline
    );
    // Dense-unique `set_order` on the BASELINE body. This is the row-specific
    // one of the three `assert_dense_unique_set_orders` calls in this body: it
    // is a from-empty snapshot taken at THIS row's `have_version`, so the five
    // rows check five genuinely different mid-history bodies, whereas the
    // delivered body is the same shape per row and the reference body (below)
    // is the identical seq-35 snapshot on all five. Vacuous by construction on
    // the `have_version=1` row alone — its baseline is `expected_baseline =
    // (1, 0)`, i.e. zero fields — which is pinned right above, so an empty
    // check there is a stated fact rather than an unnoticed gap.
    assert_dense_unique_set_orders(
        &baseline_join.decoded,
        &format!("have_version={have_version}: baseline from-empty body (plan_faithful_snapshot)"),
    );

    // ── Resume the replay to the end of the capture, on the SAME generation. ──
    replay_gze_delta_range(&svc, &session_id, layer, pause_at..kinds.len(), &mut cursor).await;
    let object_id = cursor
        .object_id
        .expect("replay must have created the layer");
    let current_seq = cursor.current_seq;
    assert_eq!(
        object_id, baseline_object_id,
        "the resumed replay must stay on the generation the baseline joined"
    );
    assert!(
        current_seq > have_version,
        "the replay must advance past have_version={have_version}, got {current_seq}"
    );

    // ── Reconnect IN-WINDOW at `have_version` on that object — the dev-observed
    //    shape (see the banner). ──
    let regime_before = read_regime_total(&svc.metrics, &svc.service_label, "in_window", "legacy");
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            layer,
            object_id,
            have_version,
            "fold-reconnector",
        ),
    )
    .await
    .expect("in-window reconnect subscribe must succeed");
    let mut stream = response.into_inner();

    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.object_id, object_id,
        "have_version={have_version}: message #1 must carry the live object_id"
    );
    assert_eq!(
        header.sequence_num, current_seq,
        "have_version={have_version}: message #1 must carry the current server seq"
    );
    assert!(
        !body.is_empty(),
        "have_version={have_version}: message #1 must carry a non-empty coalesced body"
    );

    // Envelope: an INCREMENTAL diff keyed to have_version (SDD §12.10) — never a
    // base_version=0 from-empty reset.
    let decoded = decode_broadcast_delta(&body);
    assert!(
        decoded.is_diff,
        "have_version={have_version}: coalesced catch-up must be is_diff=true"
    );
    assert_eq!(
        decoded.base_version, have_version,
        "have_version={have_version}: coalesced catch-up base_version must equal have_version"
    );

    // Exactly ONE content message, then the caught-up terminator echoing the
    // current sequence, then nothing.
    recv_caught_up_ack(&mut stream, object_id, current_seq).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
    drop(stream);

    // The dev-observed series: the in-window arm under the coalesced (`legacy`
    // catchup_mode) taxonomy. Pins WHICH builder produced the body above.
    assert_eq!(
        read_regime_total(&svc.metrics, &svc.service_label, "in_window", "legacy"),
        regime_before + 1,
        "have_version={have_version}: the reconnect must route through the in_window regime"
    );

    // ── The two net-live oracles, after the full replay. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    let reachable = reachable_section_ids(&server_state);
    let reference_join =
        gze_from_empty_client_scene(&svc, &session_id, layer, "fold-reference-joiner").await;
    let reference = &reference_join.scene;
    assert_eq!(
        reference_join.object_id, object_id,
        "reference joined the live object"
    );
    assert_eq!(
        reference_join.sequence_num, current_seq,
        "reference late-join must observe the final sequence"
    );
    let reference_ids: HashSet<u64> = reference.sections.keys().copied().collect();
    assert_eq!(
        reference_ids, reachable,
        "the two net-live oracles must agree: a from-empty late-join's scene must equal \
         the server's root-reachable section set"
    );
    let reference_fields = client_field_map(reference);
    assert_same_field_sets(
        &reference_fields,
        &net_live_field_map(&server_state, &reachable),
        "oracle cross-check (from-empty late-join vs server net-live state)",
    );
    // The `folded` column of the banner table, PINNED and SHARED by every row
    // (`NET_LIVE_SCENE`). The fold assertions below compare the mirror against
    // this reference, so without this the family would keep passing if the
    // capture's net-live scene shrank — "folds to 21/45" would quietly become
    // "folds to whatever the reference happens to be".
    assert_eq!(
        (reference_ids.len(), reference_fields.len()),
        NET_LIVE_SCENE,
        "have_version={have_version}: this capture's net-live scene must be exactly {:?} \
         (sections, fields), observed ({}, {}). It is a property of the COMMITTED, immutable \
         fixture tools/fixtures/gze_inbound_deltas.json and is the same on every row; if it \
         was re-captured, update NET_LIVE_SCENE, the banner table's folded column, and SDD \
         §12.10's \"Observed rows\" sentence TOGETHER from the observed values.",
        NET_LIVE_SCENE,
        reference_ids.len(),
        reference_fields.len()
    );
    // Animation content. The committed capture is a pure structural/property
    // edit session and carries NO time samples, so this reference map is empty
    // today — asserted explicitly so the equality below is understood as
    // vacuous-by-fixture rather than assumed to be covering something. Both
    // halves stay live: the day a re-capture introduces time samples this
    // emptiness assertion fails (pointing here), and the fold comparison below
    // starts doing real work with no further edit.
    let reference_time_samples = client_time_sample_map(reference);
    assert!(
        reference_time_samples.is_empty(),
        "the committed capture is expected to carry no time samples; the reference scene now \
         holds {} ({reference_time_samples:?}) — the fold's time-sample comparison below is \
         no longer vacuous, so re-read it (and SDD §12.10) before updating this assertion",
        reference_time_samples.len()
    );

    // ── SIBLING ORDER under each client reader, the property the scene fold
    //    is blind to (it compares section sets, never child order). Each
    //    reader model (`tests/common/sibling_model.rs`) applies the baseline
    //    body then the delivered body, in wire order, as that client would,
    //    and must hold the server's order in every USD child list — the order
    //    the same reader builds from the from-empty reference body. The Kit
    //    reader takes a catch-up order as a key with displacement, so
    //    restating only the changed children misplaces them once the
    //    retained keys and the emitted positions disagree; the C++ client
    //    library reads it as an index and renumbers. ──
    let server_lists: BTreeMap<_, _> = server_child_lists(&server_state)
        .into_iter()
        .filter(|((parent, _), _)| *parent == 0 || reachable.contains(parent))
        .collect();
    for reader in READERS {
        let mut reference_model = SiblingModel::new(reader);
        reference_model.apply(&reference_join.decoded);
        assert_eq!(
            reference_model.child_lists(),
            server_lists,
            "have_version={have_version} / {reader:?}: a from-empty joiner must hold the server's \
             order in every child list"
        );
        let mut reconnector = SiblingModel::new(reader);
        reconnector.apply(&baseline_join.decoded);
        reconnector.apply(&decoded);
        assert_eq!(
            reconnector.child_lists(),
            server_lists,
            "have_version={have_version} / {reader:?}: fold(baseline, message #1) must hold the \
             server's order in every child list"
        );
    }

    // ── SET_ORDER, the property the fold is structurally blind to. The folded
    //    and reference SCENES are compared by `(section_id, key_name) → value`,
    //    so a body whose `set_order` values collided would still fold to the
    //    right scene HERE while the Kit client silently dropped the colliding
    //    field (`std::map<pair<saveCount, setOrder>>`, one shared saveCount —
    //    SDD §12.8 / §12.9). Both faithful builders call
    //    `DiffEngine::dense_reindex_by_authoring`, so every body this row
    //    decodes owes the dense unique range `0..N-1` and each is checked
    //    separately — the from-empty bodies' compliance is not inferred from the
    //    coalesced one's. The third body, the row-specific BASELINE snapshot, is
    //    checked at the point it is taken further up. ──
    assert_dense_unique_set_orders(
        &decoded,
        &format!(
            "have_version={have_version}: delivered in-window coalesced body \
             (plan_faithful_coalesced)"
        ),
    );
    assert_dense_unique_set_orders(
        &reference_join.decoded,
        &format!(
            "have_version={have_version}: from-empty reference body \
             (plan_faithful_snapshot)"
        ),
    );

    let delivered_sections = emitted_present_section_ids(&decoded);
    let delivered_fields = decoded.set_fields.len();
    let retracted = emitted_delete_section_ids(&decoded);

    // The `delivered` column of the banner table, PINNED (see
    // `InWindowFoldCase::expected_delivered`). Sections = distinct ids named by
    // NON-delete struct commands (the harness's `present`); fields = `SetField`
    // ENTRIES. Without this the row still passes on a builder that folds
    // correctly while sending materially more than it owes — e.g. one that
    // re-sent fields, or sections outside a touched parent, that the client
    // already holds, which is exactly the degeneration the naive-reading check
    // below only catches once it goes ALL the way to a from-empty snapshot.
    assert_eq!(
        (delivered_sections.len(), delivered_fields),
        case.expected_delivered,
        "have_version={have_version}: the delivered coalesced body must carry exactly {:?} \
         (sections, fields) on this committed capture, observed ({}, {delivered_fields}) — \
         section ids {delivered_sections:?}. These are measurements against the COMMITTED, \
         immutable fixture tools/fixtures/gze_inbound_deltas.json; if it was re-captured, \
         update IN_WINDOW_FOLD_CASES, the banner table above it, and SDD §12.10's \"Observed \
         rows\" sentence TOGETHER from the observed values.",
        case.expected_delivered,
        delivered_sections.len()
    );

    // ── Retraction arm. Note what this does and does NOT pin: on THIS traffic
    //    every retracted prim's (parent, name) is immediately re-taken by a
    //    freshly-id'd prim in the same body, and `apply_diff_change_section`'s
    //    duplicate-sibling-name cascade (SDD §9.1 step 4a) would evict the stale id
    //    even from a client that ignored the retraction — so the fold outcome
    //    here does not discriminate the read-side `DiffDeleteSection` semantics.
    //    Two tests own that discrimination:
    //    `snapshot_in_window_reconnect_establishes_correct_scene` (fully
    //    synthetic scene, both ghost arms, never re-created under the same name)
    //    and `snapshot_in_window_reconnect_real_traffic_plus_ghost_retraction_…`
    //    below (THIS capture's real prims, two appended commits, explicit
    //    ignore-the-retraction counterfactual). What IS pinned here is that the
    //    retraction fires on real traffic, targets only sections the client
    //    actually still holds, never targets a prim that survives into the
    //    net-live scene, and has EXACTLY the size this traffic owes at this
    //    watermark. ──
    let baseline_ids: HashSet<u64> = mirror.sections.keys().copied().collect();

    // The raw `DiffDeleteSection` COUNT, not the deduplicated id set — SDD
    // §12.10 quotes entry counts, and a builder that emitted the same id twice
    // would be a bug the set-shaped view would hide.
    let retraction_entries = decoded
        .struct_commands
        .iter()
        .filter(|c| matches!(c, StructCommand::DiffDeleteSection(_)))
        .count();
    assert_eq!(
        retraction_entries,
        retracted.len(),
        "have_version={have_version}: the retraction set must be duplicate-free — \
         {retraction_entries} DiffDeleteSection entries collapsed to {} distinct ids {retracted:?}",
        retracted.len()
    );
    // EXACT, because the fixture is committed and immutable (see
    // `InWindowFoldCase::expected_retractions`). A `> 0` predicate here would
    // pass a builder that retracted one prim where six are owed.
    assert_eq!(
        retraction_entries, case.expected_retractions,
        "have_version={have_version}: expected exactly {} DiffDeleteSection entries on this \
         committed capture, got {retraction_entries} — ids {retracted:?}. If the fixture was \
         re-captured, update IN_WINDOW_FOLD_CASES and the banner table from these observed ids.",
        case.expected_retractions
    );
    for sid in &retracted {
        assert!(
            !reference_ids.contains(sid),
            "have_version={have_version}: retracted section {sid} must NOT be in the \
             final net-live scene"
        );
        assert!(
            baseline_ids.contains(sid),
            "have_version={have_version}: retraction of section {sid} is SPURIOUS — a \
             client that joined via full-state catch-up at have_version never held it, \
             so the server retracted something this reconnector cannot have"
        );
    }

    // ── DISCRIMINATION: the harness's naive reading — the delivered body IS the
    //    scene — must FAIL on every row, so the fold below is provably what makes
    //    this test pass. Unconditional: every row here reconnects with a non-zero
    //    lag, so the `version > have_version` filter always withholds something
    //    the client already holds. (A zero-lag row would route through the
    //    `up_to_date` regime and need a different shape; none is listed.) ──
    let mut naive = ObjectState::default();
    apply_catchup_read_side(&mut naive, &body);
    let naive_ids: HashSet<u64> = naive.sections.keys().copied().collect();
    assert_ne!(
        naive_ids, reference_ids,
        "have_version={have_version}: the delivered coalesced body alone must NOT reproduce \
         the net-live scene — if it does, this row no longer exercises the incremental \
         builder (it degenerated into a from-empty snapshot) and the fold assertion below \
         is vacuous"
    );
    assert!(
        naive_ids.len() < reference_ids.len(),
        "have_version={have_version}: delivered {} sections must be a PROPER subset of the \
         net-live {} — the fold has to add the sections the client already held",
        naive_ids.len(),
        reference_ids.len()
    );

    // ── THE PROPERTY: fold the delivered diff onto the client's retained state
    //    (read-side DiffDeleteSection semantics) and the result is the full
    //    net-live scene — section-for-section and field-value-for-field-value. ──
    apply_catchup_read_side(&mut mirror, &body);
    let folded_ids: HashSet<u64> = mirror.sections.keys().copied().collect();
    assert_eq!(
        folded_ids,
        reference_ids,
        "have_version={have_version}: fold(state@have_version, message #1) must reproduce the \
         full net-live section set — delivered {} present / baseline {baseline_sections} / \
         folded {} / reference {}",
        delivered_sections.len(),
        folded_ids.len(),
        reference_ids.len()
    );
    assert_same_field_sets(
        &client_field_map(&mirror),
        &reference_fields,
        &format!(
            "have_version={have_version}: folded scene vs the from-empty reference \
             (delivered {} fields / baseline {baseline_fields} / folded {} / reference {})",
            decoded.set_fields.len(),
            mirror.fields.len(),
            reference_fields.len()
        ),
    );
    // …and time-sample-for-time-sample. Same fold, same oracle, third keyspace:
    // `cascade_remove_section` retains on `time_samples`, so a retraction that
    // dropped a section's fields but stranded its animation would show up here.
    assert_eq!(
        client_time_sample_map(&mirror),
        reference_time_samples,
        "have_version={have_version}: folded scene vs the from-empty reference — time samples \
         diverge (delivered {} / folded {} / reference {})",
        decoded.time_samples.len(),
        mirror.time_samples.len(),
        reference_time_samples.len()
    );

    // Structural parity of every net-live section against the server.
    for sid in &reachable {
        let r = mirror
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("net-live section {sid} must be present after the fold"));
        let s = &server_state.sections[sid];
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.xvq — HYBRID: the real captured traffic PLUS two appended commits that force
// BOTH ghost arms on THIS capture's real prim ids, with an explicit
// ignore-the-retraction counterfactual.
//
// The gap this closes. `DiffEngine::plan_faithful_coalesced` (`src/domain/diff.rs`)
// retracts a UNIFIED ghost set from two disjoint sources: (a) tombstoned deletes
// and (b) sections still in the live index but NOT reachable from root. The
// committed capture exercises neither one discriminatingly:
//
//   • arm (b) NEVER fires on it — every stored section is root-reachable, which
//     is why the `…_have_version_{1,32}` rows above are owed ZERO retractions;
//   • arm (a) fires (rows 3 / 5 / 6), but every retracted prim's `(parent, name)`
//     is immediately re-taken by a freshly-id'd sibling in the SAME body, and
//     `apply_diff_change_section`'s duplicate-sibling-name cascade (SDD §9.1 step
//     4a) evicts the stale id even from a client that IGNORED the retraction. The
//     fold converges either way, so those rows cannot tell a correct read-side
//     `DiffDeleteSection` implementation from one that drops the command.
//
// So this test replays the capture in full (seq 35), takes the same from-empty
// baseline the rows above take, then appends TWO commits authored against prims
// the CAPTURE created — ids, names and parents read out of the replayed scene,
// not invented:
//
//   commit A (seq 36) — reparent a real prim (deliberately one WITH descendants)
//     under `ABANDON_ORPHAN_PARENT`, a sentinel id no captured prim uses. The
//     server keeps the section row materialized but it is no longer reachable
//     from root, so it and its whole subtree become arm-(b) ghosts. Asserted
//     directly against the server's materialized state, not assumed. The same
//     commit also touches ONE SURVIVING captured prim twice, so the delivered
//     body is not retractions-only: it writes a new field on that prim AND
//     re-states the prim's own section row verbatim (same parent, name, type,
//     order). Precisely what that buys: the body carries exactly one PRESENT
//     section (`GHOST_CASE_DELIVERED_PRESENT_SECTIONS`) and exactly one field
//     (`GHOST_CASE_DELIVERED_FIELDS`) alongside the 14 retractions.
//
//     The row re-state is load-bearing, not decoration.
//     `plan_faithful_coalesced` selects present sections on the SECTION row's
//     `first_version` / `last_version` (`src/domain/diff.rs`), and nothing else
//     in these two commits advances a SURVIVING section's `last_version` —
//     `apply_set_field` bumps the field row only, and the re-parent's
//     `remove_child_from_parent` / `insert_child_into_parent` rewrite children
//     lists without touching the parent's version tuple (`src/domain/delta.rs`).
//     A field-only commit A would therefore deliver ZERO present sections, and
//     every "the sections it names as present are net-live" check below would
//     pass vacuously over an empty set. The re-state is a legal no-op edit:
//     `apply_diff_change_section` excludes the section itself from the
//     sibling-name conflict scan (`find_child_by_name(..., Some(section_id))`),
//     so re-stating a row's own `(parent, name)` cannot trip
//     `DuplicateSiblingName`, and with parent and order unchanged it rewrites no
//     children list either — it advances `last_version` and nothing else.
//   commit B (seq 37) — `DeleteSection` of a real captured LEAF prim, and
//     deliberately does NOT re-create a same-named sibling. That is the whole
//     point: arm (a) with the duplicate-sibling-name cascade removed, so
//     ignoring the retraction leaves the prim standing.
//
// THE DISCRIMINATOR. The delivered body is folded twice onto the SAME baseline:
// once with `apply_catchup_read_side` (a Kit client that honours
// `DiffDeleteSection`) and once with `apply_ompf_to_state` (the server INGEST
// engine, which silently ignores it — SDD §9.1.4f — standing in for a client
// that dropped the command). The first must equal the reference; the second must
// NOT, and must specifically still hold the detached prim and the deleted prim.
// Read-side retraction semantics are therefore load-bearing on real prim data,
// which no row of the sweep above establishes.
//
// WHAT THIS DOES AND DOES NOT PROVE. It proves the SERVER emits both ghost arms
// correctly over a scene built by real Kit traffic, and that a conforming client
// converges only by honouring the retraction. It does NOT prove Kit itself
// authors these detach/delete shapes — the two commits are synthesised here, so
// the *scene* is real captured traffic while the *final two edits* are not. A
// true end-to-end capture of a Kit session that detaches and deletes prims is
// tracked separately in ticket b.gmb; until it lands, the arm-(b) precondition
// (stored-but-unreachable) is reached through the public write path but under
// test authorship.
// ═══════════════════════════════════════════════════════════════════════════════

/// The number of sections the detached subtree carries (the prim plus its
/// descendants) on the COMMITTED capture. Pinned rather than merely derived so a
/// re-capture that flattened the scene — leaving the "ghost" a bare leaf and
/// silently narrowing what arm (b) is shown to cover — fails here instead of
/// passing quietly.
const GHOST_SUBTREE_SECTIONS: usize = 13;

/// Fields the delivered coalesced body must carry: exactly the one probe field
/// commit A writes. Nothing else in the capture has `version > 35`, and fields
/// anchored to the now-unreachable subtree are dropped by the net-live filter —
/// so this pins the "drop ghost-anchored fields" half of the builder too.
const GHOST_CASE_DELIVERED_FIELDS: usize = 1;

/// PRESENT sections the delivered coalesced body must carry: exactly the one
/// prim commit A re-states (the probe prim). Pinned as a COUNT — the way
/// `IN_WINDOW_FOLD_CASES` pins `expected_delivered.0` — because the
/// "every present section is net-live" loop below quantifies over the emitted
/// set and would pass vacuously if that set were empty. Emitted sections are the
/// net-live rows with `last_version > have_version`, plus every other listed
/// child of such a row's parent (`src/domain/diff.rs`); the detached subtree is
/// excluded as unreachable, the deleted leaf is gone, no captured prim's row is
/// touched by these two commits, and on this capture the probe prim's parent
/// lists no other live child (measured) — so the probe prim's re-stated row is
/// the only one that qualifies.
const GHOST_CASE_DELIVERED_PRESENT_SECTIONS: usize = 1;

/// `(sections, fields)` of the net-live scene AFTER the two appended commits:
/// the pinned pre-edit `NET_LIVE_SCENE` minus the detached subtree and the
/// deleted leaf (and their fields), plus the one probe field.
const GHOST_CASE_NET_LIVE_AFTER: (usize, usize) = (7, 21);

/// `(sections, fields)` a client that IGNORED every `DiffDeleteSection` would be
/// left holding after applying the same body onto the same baseline — pinned so
/// the counterfactual's divergence from `GHOST_CASE_NET_LIVE_AFTER` is a stated
/// magnitude rather than "not equal by some amount". Nothing is retracted, so it
/// is the pre-edit scene plus the probe field.
const GHOST_CASE_IGNORED_RETRACTIONS: (usize, usize) = (21, 46);

/// Key name for the probe field commit A writes onto a surviving captured prim.
const GHOST_PROBE_KEY: &str = "xvq:ghost-probe";

/// Every section id in `root`'s subtree (root included), walked over a client
/// scene's `children` index. This is exactly the set that a single
/// reparent-to-nowhere makes unreachable, and therefore the arm-(b) ghost set
/// the coalesced builder owes.
fn subtree_section_ids(state: &ObjectState, root: u64) -> HashSet<u64> {
    let mut out = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if out.insert(id) {
            if let Some(kids) = state.children.get(&id) {
                stack.extend(kids.iter().copied());
            }
        }
    }
    out
}

/// Pick the two captured prims this test detaches (arm b) and deletes (arm a),
/// BY SHAPE rather than by hardcoded ids, so the choice survives a re-capture.
///
/// The ghost root is the LARGEST subtree among non-root-level prims — richest
/// arm-(b) coverage, since a multi-section ghost proves the builder retracts the
/// detached prim's whole DESCENDANT closure and not just the prim named in the
/// reparent. The delete target is the lowest-id LEAF outside that subtree: a leaf
/// so the delete needs no cascade, outside the subtree so the two arms stay
/// disjoint and separately attributable.
///
/// The two are chosen TOGETHER because the ghost choice constrains the delete
/// choice: candidates are tried largest-subtree-first (lowest id breaking ties)
/// and the first one that still leaves a deletable leaf plus at least one further
/// survivor — to carry the probe field AND the re-stated section row, so the
/// delivered body is not retractions-only — wins.
fn pick_ghost_and_delete_targets(state: &ObjectState) -> (u64, u64) {
    let mut candidates: Vec<(usize, u64)> = state
        .sections
        .keys()
        .copied()
        // Keep the scene's root prim(s) attached: detaching one would take the
        // whole scene with it and leave nothing to survive.
        .filter(|id| state.sections[id].parent_id != 0)
        .map(|id| (subtree_section_ids(state, id).len(), id))
        .filter(|(size, _)| *size >= 2)
        .collect();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (_, ghost_root) in candidates {
        let subtree = subtree_section_ids(state, ghost_root);
        let outside: Vec<u64> = state
            .sections
            .keys()
            .copied()
            .filter(|id| !subtree.contains(id))
            .collect();
        let mut leaves: Vec<u64> = outside
            .iter()
            .copied()
            .filter(|id| state.sections[id].parent_id != 0)
            .filter(|id| state.children.get(id).is_none_or(|kids| kids.is_empty()))
            .collect();
        leaves.sort_unstable();
        if let Some(&delete_target) = leaves.first() {
            if outside.len() >= 2 {
                return (ghost_root, delete_target);
            }
        }
    }
    panic!(
        "the captured scene ({} sections) must contain a non-root prim WITH descendants that \
         still leaves a deletable leaf and one further survivor outside it",
        state.sections.len()
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_reconnect_real_traffic_plus_ghost_retraction_discriminates_read_side_semantics(
) {
    let tv = PooledTestValkey::setup().await;
    // Roomy MAXLEN so nothing trims: the reconnector's start id must survive for
    // the in_window branch to be selected.
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let layer = "gze-ghost-fold-layer";
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("gze-ghost-fold-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ── 1. The real capture, replayed in full. ──
    let (object_id, have_version) = replay_gze_captured_deltas(&svc, &session_id, layer).await;

    // ── 2. The reconnector's baseline: what a correct client holds at seq 35. ──
    let baseline_join =
        gze_from_empty_client_scene(&svc, &session_id, layer, "ghost-baseline-joiner").await;
    assert_eq!(
        baseline_join.object_id, object_id,
        "the baseline must join the generation the replay left live"
    );
    assert_eq!(
        baseline_join.sequence_num, have_version,
        "the baseline late-join must observe the full replay's final sequence"
    );
    let mut mirror = baseline_join.scene;
    assert_eq!(
        (mirror.sections.len(), mirror.fields.len()),
        NET_LIVE_SCENE,
        "the pre-edit baseline is the same net-live scene the fold sweep pins ({:?}), observed \
         ({}, {})",
        NET_LIVE_SCENE,
        mirror.sections.len(),
        mirror.fields.len()
    );
    // The baseline body is a from-empty snapshot in its own right, so it owes
    // the same dense-unique `set_order` range the delivered and reference bodies
    // do (checked at step 8) — and it is the only one of the three built at the
    // PRE-edit watermark.
    assert_dense_unique_set_orders(
        &baseline_join.decoded,
        "ghost hybrid: pre-edit baseline from-empty body (plan_faithful_snapshot)",
    );

    // ── 3. Choose the two victims BY SHAPE out of the replayed scene, so every
    //       id, name and parent below belongs to a prim the capture created. ──
    let (ghost_root, delete_target) = pick_ghost_and_delete_targets(&mirror);
    let ghost_subtree = subtree_section_ids(&mirror, ghost_root);
    assert_eq!(
        ghost_subtree.len(),
        GHOST_SUBTREE_SECTIONS,
        "the detached subtree must carry exactly {GHOST_SUBTREE_SECTIONS} sections on this \
         committed capture, observed {} (root {ghost_root}, ids {ghost_subtree:?}) — if the \
         fixture was re-captured, update GHOST_SUBTREE_SECTIONS from the observed value",
        ghost_subtree.len()
    );
    let delete_parent = mirror.sections[&delete_target].parent_id;
    // The prim commit A writes live content onto — one field plus a verbatim
    // re-state of the prim's own section row: the lowest-id survivor, so the
    // delivered body is not retractions-only (it carries one present section and
    // one field) and the net-live content half of the fold stays exercised.
    let probe_section = {
        let mut survivors: Vec<u64> = mirror
            .sections
            .keys()
            .copied()
            .filter(|id| !ghost_subtree.contains(id) && *id != delete_target)
            .collect();
        survivors.sort_unstable();
        *survivors
            .first()
            .expect("the captured scene must retain a prim to write the probe field onto")
    };

    // ── 4. commit A (seq 36): detach the ghost subtree, write the probe field,
    //       and re-state the probe prim's own row (see the banner: without that
    //       re-state the delivered body would name no present section at all). ──
    //
    // Both re-stated rows are read from the SERVER's materialized state rather
    // than from the client mirror: the catch-up builders overwrite `section_order`
    // with the emitted per-parent sibling index, so a mirror-sourced re-state
    // could carry an order the server never stored — which on the probe prim
    // would take the `old_order != section_order` branch and rebuild its parent's
    // children list, turning a no-op re-state into a real reorder.
    let pre_edit_state = snapshot_to_object_state(
        &read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
            .await
            .expect("read_object_state_for_test must succeed"),
    );
    let ghost_section = pre_edit_state.sections[&ghost_root].clone();
    let probe_row = pre_edit_state.sections[&probe_section].clone();
    let commit_detach = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: ABANDON_ORPHAN_PARENT,
                section_id: ghost_root,
                section_name: ghost_section.name.clone(),
                section_type: ghost_section.section_type,
                section_order: ghost_section.section_order,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: probe_row.parent_id,
                section_id: probe_section,
                section_name: probe_row.name.clone(),
                section_type: probe_row.section_type,
                section_order: probe_row.section_order,
            }),
        ],
        set_fields: vec![SetField {
            section_id: probe_section,
            key_name: GHOST_PROBE_KEY.to_owned(),
            value_or_ext_hash: b"after-detach".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "xvq-ghost-writer",
        1_000,
        &commit_detach,
    )
    .await;

    // ── 5. commit B (seq 37): delete a real leaf, WITHOUT re-creating a
    //       same-named sibling — the cascade that masks arm (a) on the raw
    //       capture is deliberately absent here. ──
    let commit_delete = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DeleteSection(
            oodf_ffi::CommandDeleteSection {
                parent_id: delete_parent,
                section_id: delete_target,
            },
        )],
        ..Default::default()
    };
    apply_delta_update_returning_ompf(
        &svc,
        &session_id,
        layer,
        object_id,
        "xvq-ghost-writer",
        1_001,
        &commit_delete,
    )
    .await;
    let current_seq = have_version + 2;

    // ── 6. The arm-(b) PRECONDITION, verified against the server's materialized
    //       state rather than assumed: the detached subtree is STILL STORED and
    //       NO LONGER REACHABLE, and the deleted leaf is gone from the live index
    //       (tombstoned). If a future server cascade-deleted on reparent, this is
    //       where that shows up — the test would stop being an arm-(b) test and
    //       says so. ──
    let snap = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    let server_state = snapshot_to_object_state(&snap);
    let reachable = reachable_section_ids(&server_state);
    for sid in &ghost_subtree {
        assert!(
            server_state.sections.contains_key(sid),
            "arm-(b) precondition lost: detached section {sid} is no longer STORED — the write \
             path cascade-deleted it, so this test no longer exercises the \
             stored-but-unreachable ghost arm"
        );
        assert!(
            !reachable.contains(sid),
            "arm-(b) precondition lost: detached section {sid} is still REACHABLE from root \
             after the reparent under ABANDON_ORPHAN_PARENT"
        );
    }
    assert!(
        !server_state.sections.contains_key(&delete_target),
        "arm-(a) precondition lost: deleted section {delete_target} is still in the live index"
    );

    // ── 7. Reconnect IN-WINDOW at the pre-edit watermark. ──
    let regime_before = read_regime_total(&svc.metrics, &svc.service_label, "in_window", "legacy");
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            layer,
            object_id,
            have_version,
            "ghost-reconnector",
        ),
    )
    .await
    .expect("in-window reconnect subscribe must succeed");
    let mut stream = response.into_inner();
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(header.object_id, object_id, "message #1 object_id");
    assert_eq!(
        header.sequence_num, current_seq,
        "message #1 must carry the post-edit server sequence"
    );
    let decoded = decode_broadcast_delta(&body);
    assert!(decoded.is_diff, "coalesced catch-up must be is_diff=true");
    assert_eq!(
        decoded.base_version, have_version,
        "coalesced catch-up base_version must equal have_version"
    );
    recv_caught_up_ack(&mut stream, object_id, current_seq).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
    drop(stream);
    assert_eq!(
        read_regime_total(&svc.metrics, &svc.service_label, "in_window", "legacy"),
        regime_before + 1,
        "the reconnect must route through the in_window regime"
    );

    // ── 8. The reference oracles, after the two appended commits. ──
    let reference_join =
        gze_from_empty_client_scene(&svc, &session_id, layer, "ghost-reference-joiner").await;
    let reference = &reference_join.scene;
    assert_eq!(
        reference_join.sequence_num, current_seq,
        "reference late-join must observe the post-edit sequence"
    );
    let reference_ids: HashSet<u64> = reference.sections.keys().copied().collect();
    assert_eq!(
        reference_ids, reachable,
        "the two net-live oracles must agree: a from-empty late-join's scene must equal the \
         server's root-reachable section set"
    );
    let reference_fields = client_field_map(reference);
    assert_same_field_sets(
        &reference_fields,
        &net_live_field_map(&server_state, &reachable),
        "oracle cross-check (from-empty late-join vs server net-live state)",
    );
    // The two edits are subtractive on sections (ghost subtree + one leaf) and
    // additive on fields (the probe) — expressed as arithmetic over the pinned
    // pre-edit scene so the relationship is checked, not just the endpoint.
    assert_eq!(
        reference_ids.len(),
        NET_LIVE_SCENE.0 - ghost_subtree.len() - 1,
        "post-edit net-live scene must lose exactly the {} detached sections plus the one \
         deleted leaf, observed {} sections",
        ghost_subtree.len(),
        reference_ids.len()
    );
    assert_eq!(
        (reference_ids.len(), reference_fields.len()),
        GHOST_CASE_NET_LIVE_AFTER,
        "post-edit net-live scene must be exactly {GHOST_CASE_NET_LIVE_AFTER:?} \
         (sections, fields), observed ({}, {}) — update GHOST_CASE_NET_LIVE_AFTER (and \
         GHOST_CASE_IGNORED_RETRACTIONS) from the observed values if the fixture was \
         re-captured",
        reference_ids.len(),
        reference_fields.len()
    );
    assert!(
        reference_fields.contains_key(&(probe_section, GHOST_PROBE_KEY.to_owned())),
        "the probe field written by commit A must survive into the net-live scene"
    );

    // Both faithful builders owe the dense unique `0..N-1` set_order range — the
    // property the scene comparison below is structurally blind to (SDD §12.8 /
    // §12.9; a duplicate collides on the Kit client's saveCount-keyed map and the
    // field is silently dropped).
    assert_dense_unique_set_orders(
        &decoded,
        "ghost hybrid: delivered in-window coalesced body (plan_faithful_coalesced)",
    );
    assert_dense_unique_set_orders(
        &reference_join.decoded,
        "ghost hybrid: from-empty reference body (plan_faithful_snapshot)",
    );

    // ── 9. The delivered body: BOTH ghost arms, and nothing else. ──
    let retracted = emitted_delete_section_ids(&decoded);
    let retraction_entries = decoded
        .struct_commands
        .iter()
        .filter(|c| matches!(c, StructCommand::DiffDeleteSection(_)))
        .count();
    assert_eq!(
        retraction_entries,
        retracted.len(),
        "the retraction set must be duplicate-free — {retraction_entries} entries collapsed to \
         {} distinct ids {retracted:?}",
        retracted.len()
    );
    let mut expected_retracted = ghost_subtree.clone();
    expected_retracted.insert(delete_target);
    assert_eq!(
        retracted,
        expected_retracted,
        "the delivered body must retract EXACTLY the arm-(b) detached subtree (root \
         {ghost_root}, {} sections) plus the arm-(a) deleted leaf {delete_target}",
        ghost_subtree.len()
    );
    assert_eq!(
        decoded.set_fields.len(),
        GHOST_CASE_DELIVERED_FIELDS,
        "the delivered body must carry exactly the {GHOST_CASE_DELIVERED_FIELDS} probe field \
         (fields anchored to the now-unreachable subtree are dropped by the net-live filter), \
         observed {} — {:?}",
        decoded.set_fields.len(),
        decoded
            .set_fields
            .iter()
            .map(|f| (f.section_id, f.key_name.as_str()))
            .collect::<Vec<_>>()
    );
    // The PRESENT half of the body, pinned as a count BEFORE it is quantified
    // over: the "every present section is net-live" loop below is vacuous on an
    // empty set, and an empty set is exactly what this body would carry if
    // commit A stopped re-stating the probe prim's row (see the banner).
    let delivered_present = emitted_present_section_ids(&decoded);
    assert_eq!(
        delivered_present.len(),
        GHOST_CASE_DELIVERED_PRESENT_SECTIONS,
        "the delivered body must name exactly {GHOST_CASE_DELIVERED_PRESENT_SECTIONS} section as \
         PRESENT (the probe prim {probe_section}, whose row commit A re-states), observed {} — \
         {delivered_present:?}",
        delivered_present.len()
    );
    assert!(
        delivered_present.contains(&probe_section),
        "the one present section must be the probe prim {probe_section}, got {delivered_present:?}"
    );
    for sid in &delivered_present {
        assert!(
            reference_ids.contains(sid),
            "the delivered body named section {sid} as PRESENT, but it is not in the post-edit \
             net-live scene"
        );
    }
    // …and it must describe the prim the way the net-live scene does. The
    // retraction half is checked by id alone, so without this the body could
    // name the probe prim as present under a wrong parent/name/type and every
    // other assertion here would still pass. `section_order` is deliberately
    // NOT compared: the builder overwrites it with the emitted per-parent
    // sibling index, which is not the stored order.
    let delivered_probe_row = decoded
        .struct_commands
        .iter()
        .find_map(|c| match c {
            StructCommand::DiffChangeSection(s) if s.section_id == probe_section => Some(s),
            _ => None,
        })
        .expect("the present section must be emitted as a DiffChangeSection");
    let reference_probe_row = &reference.sections[&probe_section];
    assert_eq!(
        (
            delivered_probe_row.parent_id,
            delivered_probe_row.section_name.as_str(),
            delivered_probe_row.section_type,
        ),
        (
            reference_probe_row.parent_id,
            reference_probe_row.name.as_str(),
            reference_probe_row.section_type,
        ),
        "the present section's structural tuple (parent, name, type) must match the net-live \
         scene's row for prim {probe_section}"
    );

    // ── 10. THE DISCRIMINATOR. Same baseline, same body, two read-side
    //        implementations. The one that IGNORES `DiffDeleteSection` (the
    //        server ingest engine, SDD §9.1.4f) must NOT converge — and must
    //        still be holding both ghosts. On the raw-capture rows this
    //        counterfactual passes anyway, because the duplicate-sibling-name
    //        cascade cleans up after it; here nothing re-takes the names. ──
    let mut ignores_retractions = mirror.clone();
    apply_ompf_to_state(&mut ignores_retractions, &body);
    let ignored_ids: HashSet<u64> = ignores_retractions.sections.keys().copied().collect();
    assert!(
        ignored_ids.contains(&ghost_root),
        "counterfactual is broken: a client ignoring DiffDeleteSection should still hold the \
         detached prim {ghost_root} — if it does not, something else evicted it and this test \
         is no longer proving read-side retraction semantics"
    );
    assert!(
        ignored_ids.contains(&delete_target),
        "counterfactual is broken: a client ignoring DiffDeleteSection should still hold the \
         deleted prim {delete_target} — no same-named sibling is re-created in this body, so \
         the SDD §9.1 step 4a cascade must not be what removes it"
    );
    assert_ne!(
        ignored_ids, reference_ids,
        "read-side DiffDeleteSection semantics are NOT load-bearing on this body — a client \
         that ignored every retraction still converged to the net-live scene, so this test \
         proves nothing"
    );
    // The MAGNITUDE of that divergence, pinned: a bare `assert_ne!` would still
    // pass if the ignoring client were off by a single stranded prim, whereas
    // what this traffic actually costs it is the whole `GHOST_SUBTREE_SECTIONS`
    // detached subtree plus the deleted leaf, and every field anchored to them
    // — 21/46 held against a 7/21 reference.
    assert_eq!(
        (ignored_ids.len(), ignores_retractions.fields.len()),
        GHOST_CASE_IGNORED_RETRACTIONS,
        "a client ignoring every retraction must be left holding exactly \
         {GHOST_CASE_IGNORED_RETRACTIONS:?} (sections, fields) against the reference's \
         {GHOST_CASE_NET_LIVE_AFTER:?}, observed ({}, {})",
        ignored_ids.len(),
        ignores_retractions.fields.len()
    );

    // ── 11. THE PROPERTY: the honouring fold converges, exactly. ──
    apply_catchup_read_side(&mut mirror, &body);
    let folded_ids: HashSet<u64> = mirror.sections.keys().copied().collect();
    assert_eq!(
        folded_ids, reference_ids,
        "fold(state@have_version, message #1) must reproduce the post-edit net-live section \
         set — folded {folded_ids:?} vs reference {reference_ids:?}"
    );
    assert_same_field_sets(
        &client_field_map(&mirror),
        &reference_fields,
        "ghost hybrid: folded scene vs the from-empty reference",
    );
    assert_eq!(
        client_time_sample_map(&mirror),
        client_time_sample_map(reference),
        "ghost hybrid: folded scene vs the from-empty reference — time samples diverge"
    );
    for sid in &reachable {
        let r = mirror
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("net-live section {sid} must be present after the fold"));
        let s = &server_state.sections[sid];
        assert_eq!(r.parent_id, s.parent_id, "section {sid} parent parity");
        assert_eq!(r.name, s.name, "section {sid} name parity");
        assert_eq!(r.section_type, s.section_type, "section {sid} type parity");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.9uq — steady-state duplicate-commit drop (`sequence_num <= tracked_version`)
// ═══════════════════════════════════════════════════════════════════════════════
//
// No catch-up regime hands the literal `$` to XREAD any more. Each registers
// its stream at a durable stream id (`grpc::subscribe::stream_cursor_at`):
// `{header_seq}-0` on the up-to-date path (the header it just read); the cut
// of its proven state read on the brand-new, snapshot-mode in-window and
// out-of-window catch-ups (its own atomic read at or below the atomic-read
// limits, its layer's shared fenced read above them); the last entry id of its
// in-window `XRANGE` slice on the rollback-mode per-commit replay. So every
// later XREAD — including every post-reconnect retry — resumes from that fixed
// ID. That closes the loss window `$` left open. (The interim above-ceiling
// path, now gone, registered at a pin re-read before its state read, which
// moved the race to the DUPLICATE side: a commit landing between the pin and
// the state read was carried by BOTH the state sent during catch-up and the
// replay from the pinned cursor.) On every path a transport replay (fred
// re-issuing an interrupted XREAD from the same cursor) can hand back an entry
// already served.
//
// `run_steady_state` filters those duplicates with a `sequence_num <=
// tracked_version` drop. Without it the subscriber would apply the same commit
// twice, and an entry without relay bytes (an empty commit, or the
// `CreateLayer` seed at sequence 1) would be sent as an empty delta based at or
// above its own sequence. Duplicates are the safe side of the
// trade only BECAUSE this filter exists — hence the test.
//
// Construction, and why it is seam-level: no catch-up path leaves a window
// between its cursor and its state read any more, so the one public-path
// source of a duplicate left is a transport replay, which cannot be provoked
// on demand — and a test that only sometimes constructs its own precondition
// is worse than no test. What this test does instead is inject the exact
// artifact a replay produces: a
// stream entry for an already-served `sequence_num`, byte-shaped like the write
// path's (UTF-8 `payload` + binary `ompf` relay field), appended after the
// subscriber is caught up. Everything downstream of the injection — the
// consumer, `run_steady_state`, the relay/diff branch, the gRPC stream — is the
// production path unmodified.
//
// Discrimination: the injected entries carry relay OMPF bytes, so WITHOUT the
// filter the steady-state arm takes the `Some(ompf_bytes)` branch and relays
// them verbatim to the subscriber. The "no message arrives" assertion below is
// therefore a real signal, not an artifact of the entry being a no-op.

/// XADD a replay of an already-committed `sequence_num` onto an object's layer
/// stream: same field shape the write path uses (UTF-8 `payload` plus the
/// binary `ompf` relay field) and a writer `client_id` distinct from the
/// subscriber's so self-echo suppression cannot be what swallows it.
///
/// `stream_id` is explicit rather than `*` on purpose. The write path assigns
/// entry IDs as `<sequence_num>-0`, and Valkey requires strictly increasing
/// IDs, so a wall-clock `*` ID would sit far above every future commit's ID and
/// make the NEXT real UpdateLayer fail its XADD. Callers pass an ID in the gap
/// between the last committed entry and the next one (e.g. `2-1` between `2-0`
/// and `3-0`).
///
/// Returns the OMPF bytes it stored in the `ompf` field so a caller can assert
/// what the relay branch forwarded.
async fn xadd_replayed_commit(
    tv: &PooledTestValkey,
    object_id: u64,
    sequence_num: u64,
    stream_id: &str,
    writer_client_id: &str,
) -> Vec<u8> {
    let stream_key = tv.key_schema().layer_stream(object_id);
    let payload = format!("update|{object_id}|{sequence_num}|{writer_client_id}|1|2");
    // Real OMPF bytes: the steady-state relay branch forwards the `ompf` field
    // verbatim without parsing, so any non-empty buffer would be delivered —
    // using a genuine one keeps the fixture honest about what a duplicate
    // looks like on the wire.
    let ompf_bytes = delta_to_ompf_bytes(&incremental_diff_change_section(4_242), &[]);
    let fields: Vec<(&str, fred::types::Value)> = vec![
        ("payload", payload.into()),
        (
            "ompf",
            fred::types::Value::Bytes(Bytes::from(ompf_bytes.clone())),
        ),
    ];
    let _: String = tv
        .client()
        .pool
        .xadd(&stream_key, false, None, stream_id, fields)
        .await
        .expect("XADD of the replayed commit must succeed");
    ompf_bytes
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_drops_commits_at_or_below_the_tracked_version() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "dup-drop-session", "dup-drop-layer").await;

    // Catch up (brand-new regime): full state at seq=1, then the caught-up
    // ack. `tracked_version` is 1 when steady state begins.
    let sub_req = subscribe_request(&session_id, "dup-drop-layer", 0, 0, "subscriber-1");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (initial, _body) = recv_message(&mut stream).await;
    assert_eq!(
        initial.sequence_num, 1,
        "brand-new catch-up must deliver the freshly-created layer at seq=1"
    );
    assert_eq!(initial.object_id, object_id);
    recv_caught_up_ack(&mut stream, object_id, 1).await;

    // One real commit from a different writer advances `tracked_version` to 2
    // through the normal steady-state path.
    apply_unique_update(
        &svc,
        &session_id,
        "dup-drop-layer",
        object_id,
        "writer-client",
        100,
    )
    .await;
    let (live, live_body) = recv_message(&mut stream).await;
    assert_eq!(
        live.sequence_num, 2,
        "the first live commit must arrive normally before the duplicate is injected"
    );
    assert!(!live_body.is_empty());

    // Act: replay seq=2 (equal to `tracked_version`) and seq=1 (below it).
    // Both are commits this subscriber has already been served. The `2-1` /
    // `2-2` entry IDs sit above the real `2-0` entry and below the `3-0` the
    // next commit will claim.
    let _ = xadd_replayed_commit(&tv, object_id, 2, "2-1", "writer-client").await;
    let _ = xadd_replayed_commit(&tv, object_id, 1, "2-2", "writer-client").await;

    // Assert: neither reaches the subscriber. Without the drop, the relay
    // branch would forward the injected `ompf` bytes verbatim and this would
    // observe a header + chunk for a commit the subscriber already applied.
    assert_no_next_message(&mut stream, Duration::from_secs(3)).await;

    // Assert the filter does not over-drop: the NEXT genuinely-new commit
    // (seq=3 — the injected stream entries never touched materialized state)
    // must still flow. A filter mis-written as `<` vs `<=`, or one that
    // advanced `tracked_version` from a dropped entry, would strand the
    // subscriber here.
    apply_unique_update(
        &svc,
        &session_id,
        "dup-drop-layer",
        object_id,
        "writer-client",
        101,
    )
    .await;
    let (fresh, fresh_body) = recv_message(&mut stream).await;
    assert_eq!(
        fresh.sequence_num, 3,
        "a commit ABOVE the tracked version must still be delivered — the duplicate drop \
         must filter only what the subscriber has already been served"
    );
    assert_eq!(fresh.object_id, object_id);
    assert!(
        !fresh_body.is_empty(),
        "the post-duplicate commit must carry its payload, not an empty degenerate diff"
    );

    // Positive control for the injection itself (stands in for the negative
    // verification a source edit would give): the SAME synthetic entry shape,
    // differing only in carrying a sequence ABOVE the tracked version, must be
    // relayed verbatim to the subscriber. This rules out the "the drop above
    // passed because the injected entry was inert" reading — the fixture does
    // reach `run_steady_state`'s relay branch, so the `sequence_num <=
    // tracked_version` comparison is the only thing that suppressed it.
    //
    // Runs last because delivering it advances `tracked_version` to 4, which
    // would make a subsequent real seq=4 commit a genuine duplicate.
    let injected_ompf = xadd_replayed_commit(&tv, object_id, 4, "3-1", "writer-client").await;
    let (above, above_body) = recv_message(&mut stream).await;
    assert_eq!(
        above.sequence_num, 4,
        "an injected entry ABOVE the tracked version must be delivered — otherwise the \
         duplicate assertions above prove nothing about the filter"
    );
    assert_eq!(
        above_body, injected_ompf,
        "the steady-state relay branch must forward the entry's `ompf` bytes verbatim — \
         this is exactly what the dropped duplicates would have delivered"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.vd8 → proven reads — catch-up serves, acks and starts its stream at one cut
// ═══════════════════════════════════════════════════════════════════════════════
//
// A catch-up reads the header sequence P before its capacity waits, so a commit
// can land between P and the state read whose result it serves. Post-b.9uq that
// window's failure mode was DUPLICATION: the commit is folded into the
// catch-up snapshot AND re-delivered by the steady-state loop. b.vd8 answered
// it with a stable-cut pin re-read under the slot and a quiescence proof that
// raised the ack over the overlap (pin low, ack high), falling back to the pin —
// and a duplicate — when the proof lost to a busy layer.
//
// That loop is gone (Spec SDD SR-3.1–SR-3.3, SR-6). At or below the
// atomic-read limits the catch-up takes one atomic read under its slot and a
// read permit; above them it attaches, with no slot, to its layer's shared
// fenced-read flight, whose one fenced read is proven at its cut. Either way
// the read's own `seq` is the cut. The catch-up serves the state at that cut,
// stamps the content with the cut and the timestamps the header held at it,
// acks the cut, and registers its stream there, so a commit folded into the
// snapshot is never delivered again and every later commit is. A cut below P
// — a sequence regression, which only a restore with an undersized sequence
// band causes — ends the catch-up `UNAVAILABLE` before any content, recorded
// `unverified`, and never acks P; on the fenced path it is checked per
// attached catch-up, against that catch-up's own P. The checkpoint's twin of
// that regression check (a cut below its pin skips the layer, persisting
// nothing) is pinned in `checkpoint/assembler_test.rs`.
//
// The tests below drive the window deterministically with the
// `CatchUpStateReadSeam` (Regimes 1 and 2), the `CatchUpOutOfWindowSeam`
// (Regime 3) and, for the rollback-mode replay above the limits, the
// `CatchUpXrangeSeam`: each parks the subscriber after P, before the read
// that serves it, and a real commit lands while it is parked. The placement
// half — that the stream is registered at the cut, not at P, so the
// folded-in commits are not even queued — is pinned in
// `subscribe_capacity_budget_test.rs`, where a shrunk backlog cap turns a
// queued commit into a visible ending; here the steady-state
// `sequence_num <= tracked_version` filter would hide a re-delivered commit
// either way.
//
// ── Coverage matrix ─────────────────────────────────────────────────────────
//
// Every regime reaches its read through one shared call
// (`catch_up_proven_read`), but each builds its payload through a different
// `*_at_cut` builder and delivers through its own call site, so a cut threaded
// wrong at one site is invisible at the others.
// `assert_stable_cut_folds_and_acks_window_commit` drives the identical
// park→commit→release→assert sequence at each, at the shipped ceiling (the
// atomic read) and above it (section 6, the shared fenced read), so the matrix
// costs one line per cell:
//
// | regime            | `catchup_snapshot_enabled` | build from the proven snapshot         |
// |-------------------|----------------------------|----------------------------------------|
// | 1 `brand_new`     | `true`  (production)       | `compute_faithful_full_content_at_cut` |
// | 1 `brand_new`     | `false` (rollback knob)    | `compute_diff_at_cut` (full content)   |
// | 2 `in_window`     | `true`  (production)       | `compute_faithful_coalesced_at_cut`    |
// | 3 `out_of_window` | `true`  (production)       | `compute_faithful_coalesced_at_cut`    |
// | 3 `out_of_window` | `false` (rollback knob)    | `compute_diff_at_cut` (coalesced)      |
//
// (Regime 2 with `catchup_snapshot_enabled = false` is deliberately ABSENT at
// the shipped ceiling and is not a gap: that arm replays the retained window
// per-commit, each diff built from its own atomic read and version-bounded to
// its own commit, and acks the last replayed commit's own sequence number,
// never a read's cut (`compute_per_commit_diff_at_cut` documents why). There is
// no P→state-read overlap to close, so there is nothing for these assertions
// to say about it. Above the atomic-read limits that arm hands over to the
// out-of-window route, which reads through the flight and does serve a cut,
// so section 6 adds it as a cell of its own.)

/// Prometheus `service` label the pooled fixture stamps on every observation.
const VD8_SERVICE: &str = "test";

/// `live_subscribe_stable_cut_total{outcome, regime, service}` — one increment
/// per catch-up materialized-state read, `outcome` in `verified` | `unverified`
/// (on the atomic path: `verified` for a cut at or above P, `unverified` for a
/// sequence regression).
///
/// `regime` is NOT optional here even though `sample_value` matches a label
/// SUBSET: the counters are split by regime, so a reader that omitted it would
/// return whichever series the exposition happened to emit first. That is
/// harmless in a single-subscriber test and actively wrong in the matrix cases
/// below, where a cross-matched series would let a case pass on another
/// regime's observation.
fn read_stable_cut_total(metrics: &Metrics, outcome: &str, regime: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_subscribe_stable_cut_total",
        &[
            ("outcome", outcome),
            ("regime", regime),
            ("service", VD8_SERVICE),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_snapshot_read_total{consumer="catchup", path, outcome}`; `0`
/// while absent. Recorded on the client's metrics, so a fixture reading it
/// installs the service's metrics on its client.
#[cfg(feature = "test-fault-injection")]
fn read_catchup_snapshot_reads(metrics: &Metrics, path: &str, outcome: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "catchup"),
            ("path", path),
            ("outcome", outcome),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Section ids named by the struct commands in a catch-up / live payload.
/// Used to assert a specific commit's content is (or is not) carried by a
/// message, without depending on the builder's command flavour.
#[cfg(feature = "test-fault-injection")]
fn struct_command_section_ids(delta: &OwnedDelta) -> HashSet<u64> {
    delta
        .struct_commands
        .iter()
        .map(|cmd| match cmd {
            StructCommand::CreateSection(c) => c.section_id,
            StructCommand::DeleteSection(c) => c.section_id,
            StructCommand::MoveSection(c) => c.section_id,
            StructCommand::ReorderChildren(c) => c.section_id,
            StructCommand::DiffChangeSection(c) => c.section_id,
            StructCommand::DiffDeleteSection(c) => c.section_id,
        })
        .collect()
}

// ── Shared matrix driver ────────────────────────────────────────────────────

/// Section written by the commit that lands INSIDE the pin→state-read window in
/// every matrix case. A fixed id is unambiguous because each case runs against
/// its own key prefix, and it keeps failure messages comparable across cells.
#[cfg(feature = "test-fault-injection")]
const VD8_WINDOW_SECTION: u64 = 990;

/// First section id used by a matrix case's pre-subscribe seed commits.
#[cfg(feature = "test-fault-injection")]
const VD8_SEED_SECTION_BASE: u64 = 900;

/// Which catch-up regime a matrix case drives, and everything that differs
/// between them: the stable-cut `regime` label, how much retained history the
/// layer needs, and how the subscriber presents itself.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum Vd8Regime {
    /// Regime 1 — `have_object_id = 0, have_version = 0`.
    BrandNew,
    /// Regime 2 — reconnector whose start id `{have_version+1}-0` is still
    /// retained, so the reconnect is in window.
    InWindow,
    /// Regime 3 — reconnector whose start id has been trimmed away.
    OutOfWindow,
    /// Regime 2 in rollback mode (`catchup_snapshot_enabled = false`) above
    /// the atomic-read limits only: the per-commit replay's first read
    /// answers `TOO_LARGE` and the replay hands over to the out-of-window
    /// route, which reads through the layer's flight (Spec SDD SR-3.4).
    InWindowRollback,
}

#[cfg(feature = "test-fault-injection")]
impl Vd8Regime {
    /// The regime's name in layer, session and client ids.
    fn label(self) -> &'static str {
        match self {
            Self::BrandNew => "brand_new",
            Self::InWindow => "in_window",
            Self::OutOfWindow => "out_of_window",
            Self::InWindowRollback => "in_window_rollback",
        }
    }

    /// The `regime` label the stable-cut counters carry — the same vocabulary
    /// `observe_subscribe_regime` uses, which is what makes the two joinable.
    /// A rollback replay's hand-over records its read under the out-of-window
    /// route that took it.
    fn stable_cut_label(self) -> &'static str {
        match self {
            Self::InWindowRollback => "out_of_window",
            other => other.label(),
        }
    }

    /// `layer_stream_maxlen` for the case's service.
    ///
    /// Regime 3 needs a MAXLEN small enough that the reconnector's start id can
    /// be evicted (the established `4` + 5-commit shape); the other two need a
    /// roomy stream so nothing trims the start id away.
    fn stream_maxlen(self) -> u64 {
        match self {
            Self::OutOfWindow => 4,
            _ => 4096,
        }
    }

    /// Commits applied after `CreateLayer` (itself seq 1) and before the
    /// subscribe. The header sequence the subscriber pins is therefore
    /// `seed_commits() + 1`.
    fn seed_commits(self) -> u64 {
        match self {
            Self::OutOfWindow => 5,
            _ => 2,
        }
    }

    /// The `sequence_num` the subscriber claims to already hold. Regime 1 is
    /// brand new (0); the two reconnect regimes claim the first commit, which
    /// puts their start id at `2-0` — retained for Regime 2, trimmed away for
    /// Regime 3.
    fn have_version(self) -> u64 {
        match self {
            Self::BrandNew => 0,
            _ => 1,
        }
    }
}

/// Which read serves a matrix case's state.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum Vd8Ceiling {
    /// The shipped atomic-read ceiling, far above every layer here: the
    /// catch-up's own atomic read.
    Shipped,
    /// A one-item ceiling ([`PooledTestValkey::setup_above_ceiling`]): the
    /// layer's shared fenced-read flight.
    Above,
}

#[cfg(feature = "test-fault-injection")]
impl Vd8Ceiling {
    async fn setup(self) -> PooledTestValkey {
        match self {
            Self::Shipped => PooledTestValkey::setup().await,
            Self::Above => PooledTestValkey::setup_above_ceiling().await,
        }
    }

    /// The `path` label of the read that serves the catch-up.
    fn served_path(self) -> &'static str {
        match self {
            Self::Shipped => "atomic",
            Self::Above => "fenced",
        }
    }
}

/// The installed park point, and the arrival the driver task uses to release it.
///
/// One variant per park point rather than one per seam type: the Regime 1 and
/// Regime 2 parks live on the same `CatchUpStateReadSeam` but are selected by
/// which constructor built it, so the variant records the choice the test made
/// and keeps the release call impossible to mismatch.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone)]
enum Vd8Park {
    /// Parked before the brand-new catch-up's proven read (Regime 1).
    FullStateRead(Arc<live_edit::grpc::subscribe::CatchUpStateReadSeam>),
    /// Parked before the in-window snapshot catch-up's proven read (Regime 2).
    CoalescedRead(Arc<live_edit::grpc::subscribe::CatchUpStateReadSeam>),
    /// Parked before the out-of-window catch-up's proven read (Regime 3).
    OutOfWindowDiff(Arc<live_edit::grpc::subscribe::CatchUpOutOfWindowSeam>),
    /// Parked before the rollback-mode in-window replay's `XRANGE` slice and
    /// every read after it (Regime 2, rollback mode).
    ReplaySlice(Arc<live_edit::grpc::subscribe::CatchUpXrangeSeam>),
}

#[cfg(feature = "test-fault-injection")]
impl Vd8Park {
    /// Block until the subscriber has parked. Every park point sits after the
    /// header read P and before the read whose state the catch-up serves (the
    /// atomic read, or above the atomic-read limits its flight's fenced read),
    /// with no registered stream, so a commit issued from here on is above P
    /// and lands before the read that serves it.
    async fn wait_until_parked(&self) {
        match self {
            Self::FullStateRead(seam) => seam.wait_until_parked_pre_full_state().await,
            Self::CoalescedRead(seam) => seam.wait_until_parked_pre_coalesced().await,
            Self::OutOfWindowDiff(seam) => seam.wait_until_parked_pre_compute_diff().await,
            Self::ReplaySlice(seam) => seam.wait_until_parked().await,
        }
    }

    /// Rendezvous with the parked subscriber, letting its state read proceed.
    async fn release(&self) {
        match self {
            Self::FullStateRead(seam) => seam.arrive_at_pre_full_state().await,
            Self::CoalescedRead(seam) => seam.arrive_at_pre_coalesced().await,
            Self::OutOfWindowDiff(seam) => seam.arrive_at_pre_compute_diff().await,
            Self::ReplaySlice(seam) => seam.release().await,
        }
    }
}

/// Install the regime's park point on `svc`, before the read that serves the
/// catch-up (its atomic read, or its flight's fenced read), and return it.
#[cfg(feature = "test-fault-injection")]
fn install_vd8_park(svc: &mut LiveEditService, regime: Vd8Regime) -> Vd8Park {
    match regime {
        Vd8Regime::BrandNew => {
            let seam =
                Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
            svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
            Vd8Park::FullStateRead(seam)
        }
        Vd8Regime::InWindow => {
            let seam =
                Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_coalesced());
            svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
            Vd8Park::CoalescedRead(seam)
        }
        Vd8Regime::OutOfWindow => {
            let seam = Arc::new(
                live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff(),
            );
            svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
            Vd8Park::OutOfWindowDiff(seam)
        }
        Vd8Regime::InWindowRollback => {
            let seam = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
            svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&seam));
            Vd8Park::ReplaySlice(seam)
        }
    }
}

/// A session and `layer` seeded for `regime`: `seed_commits()` commits after
/// `CreateLayer`, so the header the subscriber reads first is
/// `seed_commits() + 1`, and for Regime 3 a deterministic `XTRIM MINID` that
/// evicts the reconnector's start id. Returns `(session_id, object_id)`.
#[cfg(feature = "test-fault-injection")]
async fn seed_vd8_layer(
    tv: &PooledTestValkey,
    svc: &LiveEditService,
    regime: Vd8Regime,
    session_name: &str,
    layer: &str,
) -> (String, u64) {
    let (session_id, object_id) =
        create_session_and_layer_pooled(tv, svc, session_name, layer).await;
    for i in 0..regime.seed_commits() {
        apply_unique_update(
            svc,
            &session_id,
            layer,
            object_id,
            "vd8-writer",
            VD8_SEED_SECTION_BASE + i,
        )
        .await;
    }
    if matches!(regime, Vd8Regime::OutOfWindow) {
        // Force the reconnector's start id (`{have_version+1}-0`) out of the
        // retained window deterministically rather than depending on `XADD
        // MAXLEN ~`'s approximate trimming.
        let stream_key = tv.key_schema().layer_stream(object_id);
        let minid = format!("{}-0", regime.seed_commits() + 1);
        let cap: XCap =
            XCap::try_from(("MINID", "=", minid.as_str())).expect("build MINID xcap must succeed");
        let _: () = tv
            .client()
            .pool
            .xtrim(&stream_key, cap)
            .await
            .expect("XTRIM to force out-of-window must succeed");
    }
    (session_id, object_id)
}

/// Everything a matrix case leaves behind, so a caller can bolt extra legs onto
/// the shared assertions (e.g. the Regime 1 loss-freedom probe).
///
/// `tv` is held — not dropped — because its `Drop` unlinks the case's key
/// prefix, which would pull the layer out from under any follow-up write.
///
/// `#[allow(dead_code)]` on the whole struct: this is a fixture-return bundle,
/// and each cell reads a different subset of it (most read none at all — the
/// shared assertions have already run by the time it is built). Pruning it to
/// the currently-used fields would mean editing it every time a cell grows a
/// follow-up leg.
#[cfg(feature = "test-fault-injection")]
#[allow(dead_code)]
struct Vd8Case {
    tv: PooledTestValkey,
    svc: Arc<LiveEditService>,
    session_id: String,
    layer: String,
    object_id: u64,
    stream: Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>,
    /// The catch-up content message's header.
    content: live_edit::grpc::proto::SubscribeReadLayerHeader,
    /// Its reassembled OMPF body.
    body: Vec<u8>,
    /// Header sequence the subscriber read before its capacity waits (P), and
    /// the sequence the layer held when it parked.
    pin: u64,
    /// Server sequence of the commit that landed while the subscriber was
    /// parked — the cut its proven read had to serve and ack.
    window_seq: u64,
}

/// Drive one `(regime, catchup_snapshot_enabled, ceiling)` cell of the matrix:
/// park the subscriber after its header read P and before the read that serves
/// it, land a real commit, release, and assert the catch-up folded the commit
/// in, ACKED it, stamped it, and delivered it exactly once.
///
/// The assertions fail independently:
///
///   1. **Folded in** — the catch-up payload names `VD8_WINDOW_SECTION`. Without
///      this the overlap never happened and the rest proves nothing, so it is
///      checked first and its failure says so.
///   2. **Acked at the cut** — the content header carries `window_seq`, the
///      proven read's cut, not P. Pre-b.vd8 every regime stamped the pin.
///   3. **Stamped at the cut** — the content header's `server_timestamp` /
///      `client_timestamp` are the header's own at the cut, i.e. the window
///      commit's (Spec SDD SR-3.3). On b.xtb's order they came from the header
///      re-read before the park, so they named the commit before it.
///   4. **Ack ≤ content** — the caught-up ack echoes exactly the content
///      header's sequence (`recv_caught_up_ack` asserts both fields). An ack
///      ABOVE the content is the b.dy5 / b.rdr client-stall shape.
///   5. **Exactly once** — nothing else arrives: the window commit is not
///      delivered again after the ack.
///
/// Plus the observability leg: the case's own `regime` series records one
/// `verified` cut, no `unverified` one and no retry (one pass), and
/// `live_edit_snapshot_read_total` records the catch-up's one read — its
/// atomic read at or below the atomic-read limits, its flight's fenced read
/// above them.
///
/// Parameterised rather than copied per cell because the sequence is genuinely
/// identical at every build site; only the seam, the retained history and the
/// subscriber's `(have_object_id, have_version)` differ, and those are
/// `Vd8Regime`'s job — and, for `ceiling`, which read serves the state.
#[cfg(feature = "test-fault-injection")]
async fn assert_stable_cut_folds_and_acks_window_commit(
    regime: Vd8Regime,
    catchup_snapshot_enabled: bool,
    ceiling: Vd8Ceiling,
) -> Vd8Case {
    let mode = if catchup_snapshot_enabled {
        "snapshot"
    } else {
        "off"
    };
    let layer = format!("vd8-{}-{mode}-layer", regime.label());
    let pin = regime.seed_commits() + 1;
    let window_seq = pin + 1;

    let tv = ceiling.setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), regime.stream_maxlen());
    svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
    let park = install_vd8_park(&mut svc, regime);
    let svc = Arc::new(svc);
    // The snapshot-read counters are recorded on the client's metrics.
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    let (session_id, object_id) = seed_vd8_layer(
        &tv,
        &svc,
        regime,
        &format!("vd8-{}-{mode}", regime.label()),
        &layer,
    )
    .await;

    // Driver: wait until the subscriber is parked — P read, its reader and
    // slot taken (if it needs one), the read that serves it not yet — land the
    // window commit, then release the park so that read runs against the
    // post-commit object. Landed any earlier, the commit could precede P and
    // open no window at all.
    let drv_svc = Arc::clone(&svc);
    let drv_session = session_id.clone();
    let drv_layer = layer.clone();
    let drv_park = park.clone();
    let driver = tokio::spawn(async move {
        drv_park.wait_until_parked().await;
        apply_unique_update(
            &drv_svc,
            &drv_session,
            &drv_layer,
            object_id,
            "vd8-writer",
            VD8_WINDOW_SECTION,
        )
        .await;
        drv_park.release().await;
    });

    let sub_req = subscribe_request(
        &session_id,
        &layer,
        // Regime 1 must present `have_object_id = 0` for `run_catch_up_inner`'s
        // brand-new test; the reconnect regimes must present the real object so
        // they are NOT mistaken for an overwrite-on-reconnect.
        match regime {
            Vd8Regime::BrandNew => 0,
            _ => object_id,
        },
        regime.have_version(),
        &format!("vd8-{}-{mode}-subscriber", regime.label()),
    );
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (content, body) = recv_message(&mut stream).await;
    tokio::time::timeout(SUBSCRIBER_DEADLINE, driver)
        .await
        .expect("driver task must finish once the seam is released")
        .expect("driver task panicked");

    // 1. The snapshot really carries the window commit.
    let delta = decode_broadcast_delta(&body);
    assert!(
        struct_command_section_ids(&delta).contains(&VD8_WINDOW_SECTION),
        "[{} / catchup_snapshot_enabled={catchup_snapshot_enabled}] the catch-up payload must \
         carry the commit that landed after its header read P — without that the duplicate this \
         block is about never existed and the rest of this case proves nothing",
        regime.label()
    );

    // 2. The content is labelled with the proven read's cut, not P.
    assert_eq!(content.object_id, object_id);
    assert_eq!(
        content.sequence_num,
        window_seq,
        "[{} / catchup_snapshot_enabled={catchup_snapshot_enabled}] the catch-up content must be \
         labelled with its proven read's cut ({window_seq}); labelling it with P ({pin}) \
         under-claims a payload that already carries commit {window_seq}",
        regime.label()
    );

    // 3. Stamped at the cut: the header's own timestamps at `window_seq`, which
    //    are the window commit's (nothing has committed since).
    let at_cut = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the header reads");
    assert_eq!(
        at_cut.sequence_num, window_seq,
        "fixture self-check: nothing committed after the window commit"
    );
    assert_eq!(
        (
            content.server_timestamp_microseconds_since_epoch,
            content.client_timestamp_microseconds_since_epoch,
        ),
        (at_cut.last_server_ts, at_cut.last_client_ts),
        "[{} / catchup_snapshot_enabled={catchup_snapshot_enabled}] the content header must carry \
         the timestamps the header held at the cut it serves",
        regime.label()
    );
    assert_eq!(
        content.client_timestamp_microseconds_since_epoch,
        VD8_WINDOW_SECTION * 10,
        "fixture self-check: the cut's client timestamp is the window commit's own \
         (`apply_unique_update` stamps section_id × 10)"
    );

    // 4. Ack-≤-content: the caught-up ack carries exactly the last content
    //    message's sequence (`recv_caught_up_ack` asserts both fields).
    recv_caught_up_ack(&mut stream, content.object_id, content.sequence_num).await;

    // 5. Exactly once: the window commit is not delivered again after the ack.
    assert_no_next_message(&mut stream, Duration::from_secs(3)).await;

    let regime_label = regime.stable_cut_label();
    assert_eq!(
        read_stable_cut_total(&svc.metrics, "verified", regime_label),
        1,
        "[{regime_label} / catchup_snapshot_enabled={catchup_snapshot_enabled} / {ceiling:?}] the \
         catch-up state read must record exactly one verified cut on its OWN regime series"
    );
    assert_eq!(
        read_stable_cut_total(&svc.metrics, "unverified", regime_label),
        0,
        "[{regime_label} / catchup_snapshot_enabled={catchup_snapshot_enabled} / {ceiling:?}] a \
         cut at or above P must not be reported as unverified"
    );
    assert_eq!(
        read_catchup_snapshot_reads(&svc.metrics, "fenced", "fence_broken"),
        0,
        "[{regime_label} / catchup_snapshot_enabled={catchup_snapshot_enabled} / {ceiling:?}] a \
         commit that landed before the read must not cost a fenced read a broken attempt"
    );
    assert_eq!(
        read_catchup_snapshot_reads(&svc.metrics, ceiling.served_path(), "ok"),
        1,
        "[{regime_label} / catchup_snapshot_enabled={catchup_snapshot_enabled} / {ceiling:?}] the \
         catch-up served its one {} read",
        ceiling.served_path()
    );

    Vd8Case {
        tv,
        svc,
        session_id,
        layer,
        object_id,
        stream,
        content,
        body,
        pin,
        window_seq,
    }
}

// ── 1. Regime 1 (brand new): the in-window commit arrives exactly once ───────
//
// THE regression test for b.vd8. A subscriber is parked after its header read
// P and before its atomic read; a real commit lands; on release the catch-up
// must fold the commit in AND ack it, so it is never applied a second time.
//
// Pre-b.vd8 (ack = pin, `tracked_version = header.sequence_num`) this fails
// three ways: the content header carries the pin instead of the folded-in
// commit's sequence, the caught-up ack carries the pin, and the overlapped
// commit comes back as a live delta that `assert_no_next_message` catches.
//
// The tail of the test doubles as the loss-freedom leg: acking the cut must NOT
// cost the subscriber a commit that lands after the state read, which still has
// to arrive live under its own sequence.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_brand_new_commit_inside_pin_window_delivered_exactly_once() {
    /// Section written after the catch-up completes — the loss-freedom probe.
    const POST_CATCH_UP_SECTION: u64 = 604;

    // Production default: the corrected faithful full-state catch-up.
    let mut case = assert_stable_cut_folds_and_acks_window_commit(
        Vd8Regime::BrandNew,
        true,
        Vd8Ceiling::Shipped,
    )
    .await;

    // Loss-freedom leg (Regime 1 only — it is the one regime whose payload is a
    // self-contained full state, so a local mirror built from it can be checked
    // against a later live delta): acking the cut must NOT cost the subscriber
    // a commit that lands AFTER the state read.
    let mut local_state = ObjectState::default();
    apply_ompf_to_state(&mut local_state, &case.body);
    assert!(
        local_state.sections.contains_key(&VD8_WINDOW_SECTION),
        "the full-state catch-up must reconstruct into a mirror containing the window commit's \
         section — the shared assertions check the emitted commands, this checks the applied state"
    );

    apply_unique_update(
        &case.svc,
        &case.session_id,
        &case.layer,
        case.object_id,
        "vd8-writer",
        POST_CATCH_UP_SECTION,
    )
    .await;
    let (live, live_body) = recv_message(&mut case.stream).await;
    assert_eq!(
        live.sequence_num,
        case.window_seq + 1,
        "the first live delta must be the post-catch-up commit — a lower sequence here means \
         the window commit was re-delivered after all, a higher one means a commit was skipped"
    );
    apply_ompf_to_state(&mut local_state, &live_body);
    assert!(
        local_state.sections.contains_key(&POST_CATCH_UP_SECTION),
        "the post-catch-up commit's content must reach the subscriber"
    );
}

// ── 1b. Regime 1 with the rollback knob off: same guarantee, other builder ───
//
// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false` swaps the full-state builder from
// `compute_faithful_full_content_at_cut` to the legacy `compute_diff_at_cut`.
// That is a genuinely separate build site, and it is the arm an operator lands
// on when rolling the catch-up correction back — so the exactly-once handoff
// has to survive there too, or the rollback knob silently reopens this issue.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_brand_new_legacy_mode_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::BrandNew, false, Vd8Ceiling::Shipped)
        .await;
}

// ── 2. Regime 1 (brand new), quiescent layer: verified on the first pass ─────
//
// The common case, and the other half of the ack-≤-content invariant: with no
// writer in flight the cut IS the header sequence read first, and the content
// header, the caught-up ack and that sequence all agree. (The read's cost needs
// no guard here: the catch-up's state read is one atomic `EVALSHA`, which has
// no retry to spend.)

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_brand_new_quiescent_layer_verifies_at_the_pinned_sequence() {
    const LAYER: &str = "vd8-quiescent-layer";
    /// CreateLayer (seq 1) + two commits.
    const CURRENT_SEQ: u64 = 3;

    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;

    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "vd8-quiescent-session", LAYER).await;
    for section_id in [701u64, 702] {
        apply_unique_update(
            &svc,
            &session_id,
            LAYER,
            object_id,
            "vd8-writer",
            section_id,
        )
        .await;
    }

    let sub_req = subscribe_request(&session_id, LAYER, 0, 0, "vd8-quiescent-subscriber");
    let response = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed");
    let mut stream = response.into_inner();

    let (content, _body) = recv_message(&mut stream).await;
    assert_eq!(
        content.sequence_num, CURRENT_SEQ,
        "an untouched layer's catch-up must be labelled with the sequence it pinned"
    );
    recv_caught_up_ack(&mut stream, content.object_id, content.sequence_num).await;

    assert_eq!(
        read_stable_cut_total(&svc.metrics, "verified", "brand_new"),
        1,
        "a quiescent catch-up must record a verified stable cut"
    );
}

// ── 3. Regime 2 (in-window reconnect): the site with no other coverage ───────
//
// A reconnector whose start id is still retained takes
// `catch_up_in_window_snapshot`, which takes its atomic read under its
// catch-up slot and builds coalesced state through
// `compute_faithful_coalesced_at_cut` — a THIRD build and delivery site,
// reached only when `catchup_snapshot_enabled` is true (the `false` arm
// replays the retained window per-commit, each diff version-bounded to its
// commit and acked at that commit, never at a read's cut). Parked via
// `CatchUpStateReadSeam::new_before_coalesced`.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_in_window_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::InWindow, true, Vd8Ceiling::Shipped)
        .await;
}

// ── 4. Regime 3 (out of window): same window, same exactly-once outcome ──────
//
// Drives the identical interleaving on the coalesced out-of-window path through
// the pre-existing `CatchUpOutOfWindowSeam::new_before_compute_diff`, which
// parks before the read permit and the atomic read. The out-of-window regime
// is forced with the established MAXLEN=4 + 5-commit + `have_version=1` +
// deterministic XTRIM shape (see the b.jvx Site C tests).
//
// Both knob positions are covered: the `true` arm builds through
// `compute_faithful_coalesced_at_cut`, the `false` arm through the legacy
// coalesced `compute_diff_at_cut` — again two distinct build sites.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_out_of_window_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(
        Vd8Regime::OutOfWindow,
        true,
        Vd8Ceiling::Shipped,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_out_of_window_legacy_mode_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(
        Vd8Regime::OutOfWindow,
        false,
        Vd8Ceiling::Shipped,
    )
    .await;
}

// ── 5. Sequence regression: end UNAVAILABLE, never ack the pin ───────────────
//
// A catch-up whose proven read's cut — its atomic read's, or its flight's
// shared fenced read's — is below the header sequence P it read
// first saw the layer's sequence move backwards — which only a restore with an
// undersized sequence band does (Spec SDD SR-3.2). There is no proven state at
// or above P to serve, and acking P would claim commits the snapshot does not
// carry, so it ends the stream `UNAVAILABLE` before any content or ack,
// recorded `unverified` under its own regime (SR-8.6) and counted under exit
// reason `error` — never through the NotFound recovery, because the layer
// exists. A retried subscribe re-reads the current header and catches up.
//
// This replaces b.vd8's unverified fallback, which acked the pin and let the
// overlapped commit through as a duplicate the client absorbed: neither the
// atomic read nor the fenced read can lose a proof, so the only `unverified`
// left is the regression.
//
// The regression is staged by rewriting the header's `seq` column below P
// while the subscriber is parked after P and before its proven read. At the
// shipped ceiling, one cell per build/delivery site family: brand new in the
// production mode, the in-window snapshot reconnect, and the rollback-mode
// out-of-window reconnect; above it (section 6), every regime in both modes.
// Each cell also checks that the regression was caught on the read its
// ceiling names: one `ok` read on that path, and one fenced-read flight above
// the ceiling (none at the shipped one).

/// The in-stream status of a catch-up whose cut regressed below P.
#[cfg(feature = "test-fault-injection")]
const SEQUENCE_REGRESSED_MSG: &str =
    "the layer's sequence moved backwards while this subscription \
     caught up (the layer was restored); retry";

#[cfg(feature = "test-fault-injection")]
async fn assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
    regime: Vd8Regime,
    catchup_snapshot_enabled: bool,
    ceiling: Vd8Ceiling,
) {
    let mode = if catchup_snapshot_enabled {
        "snapshot"
    } else {
        "off"
    };
    let name = regime.label();
    let label = regime.stable_cut_label();
    let layer = format!("vd8-regressed-{name}-{mode}-layer");
    let pin = regime.seed_commits() + 1;
    // Below P, as a restore from a checkpoint taken earlier would leave it.
    // The rollback replay's case goes further, to its `have_version` — the
    // case b.h7b left open, where the interim path served zero bytes and
    // acked the pin.
    let regressed = match regime {
        Vd8Regime::InWindowRollback => regime.have_version(),
        _ => pin - 1,
    };

    let tv = ceiling.setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), regime.stream_maxlen());
    svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
    let park = install_vd8_park(&mut svc, regime);
    let svc = Arc::new(svc);
    // The snapshot-read and flight counters are recorded on the client's
    // metrics.
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    let (session_id, object_id) = seed_vd8_layer(
        &tv,
        &svc,
        regime,
        &format!("vd8-regressed-{name}-{mode}"),
        &layer,
    )
    .await;
    let header_key = tv.key_schema().obj_header(&object_id.to_string());

    let have_object_id = match regime {
        Vd8Regime::BrandNew => 0,
        _ => object_id,
    };
    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            &layer,
            have_object_id,
            regime.have_version(),
            &format!("vd8-regressed-{name}-{mode}-subscriber"),
        ),
    )
    .await
    .expect("subscribe must succeed")
    .into_inner();
    tokio::time::timeout(SUBSCRIBER_DEADLINE, park.wait_until_parked())
        .await
        .expect("the subscriber parks after its header read P, before its proven read");
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, ("seq", regressed.to_string()))
        .await
        .expect("rewinding the header sequence must succeed");
    tokio::time::timeout(SUBSCRIBER_DEADLINE, park.release())
        .await
        .expect("the parked subscriber accepts its release");

    // The ending is the stream's FIRST item: no content, no ack at P.
    let status = match tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
        .await
        .expect("the regressed catch-up must end within the deadline")
    {
        Some(Err(status)) => status,
        Some(Ok(msg)) => panic!(
            "[{label} / {mode}] a regressed catch-up must send nothing before its ending — no \
             content and no caught-up ack at P ({pin}); got {msg:?}"
        ),
        None => panic!("[{label} / {mode}] the stream ended with no status: a bare EOS is gRPC OK"),
    };
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "[{label} / {mode}] a regression is retryable, not NOT_FOUND: the layer exists; got \
         {status:?}"
    );
    assert_eq!(
        status.message(),
        SEQUENCE_REGRESSED_MSG,
        "[{label} / {mode}]"
    );
    assert!(
        tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
            .await
            .expect("the stream ends after its status")
            .is_none(),
        "[{label} / {mode}] the status is the stream's terminal item"
    );

    assert_eq!(
        read_stable_cut_total(&svc.metrics, "unverified", label),
        1,
        "[{label} / {mode}] the regression records one `unverified` cut on its own regime series"
    );
    assert_eq!(
        read_stable_cut_total(&svc.metrics, "verified", label),
        0,
        "[{label} / {mode}] nothing was served, so nothing is reported verified"
    );
    let exits_error = || {
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&svc.metrics),
            "live_edit_subscriber_stream_exits_total",
            &[(
                "reason",
                live_edit::observability::metrics::STREAM_EXIT_ERROR,
            )],
        )
        .expect("stream_exits_total{reason=\"error\"} is pre-initialised") as u64
    };
    let deadline = tokio::time::Instant::now() + SUBSCRIBER_DEADLINE;
    while exits_error() != 1 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "[{label} / {mode}] the regression ending is counted once under exit reason `error`"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The regression was caught on the read the cell's ceiling names — the
    // catch-up's own atomic read, or its flight's one fenced read — not
    // earlier on another path.
    assert_eq!(
        read_catchup_snapshot_reads(&svc.metrics, ceiling.served_path(), "ok"),
        1,
        "[{label} / {mode} / {ceiling:?}] the regression is caught on the catch-up's one {} read",
        ceiling.served_path()
    );
    let flight_leaders = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&svc.metrics),
        "live_edit_catchup_snapshot_flight_total",
        &[("role", "leader")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        flight_leaders,
        match ceiling {
            Vd8Ceiling::Shipped => 0,
            Vd8Ceiling::Above => 1,
        },
        "[{label} / {mode} / {ceiling:?}] one fenced-read flight above the ceiling, none at the \
         shipped one"
    );

    // A retried subscribe re-reads the current header and catches up at it.
    // Brand new only: the rewind leaves the reconnect regimes' streams holding
    // entries above the header, which no real restore of the stream would.
    if !matches!(regime, Vd8Regime::BrandNew) {
        return;
    }
    let mut retry_svc =
        build_pooled_service(&tv, Arc::new(AllowAllPermissions), regime.stream_maxlen());
    retry_svc.catchup_snapshot_enabled = catchup_snapshot_enabled;
    let mut retried = handle_subscribe_read_layer(
        &retry_svc,
        subscribe_request(
            &session_id,
            &layer,
            have_object_id,
            regime.have_version(),
            &format!("vd8-regressed-{name}-{mode}-retry"),
        ),
    )
    .await
    .expect("the retried subscribe must succeed")
    .into_inner();
    let (content, _) = recv_message(&mut retried).await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (object_id, regressed),
        "[{label} / {mode}] the retried subscribe serves the layer at its current header"
    );
    recv_caught_up_ack(&mut retried, object_id, regressed).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_brand_new_sequence_regression_ends_unavailable_and_never_acks_the_pin() {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::BrandNew,
        true,
        Vd8Ceiling::Shipped,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_in_window_sequence_regression_ends_unavailable_and_never_acks_the_pin() {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::InWindow,
        true,
        Vd8Ceiling::Shipped,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stable_cut_out_of_window_legacy_mode_sequence_regression_ends_unavailable_and_never_acks_the_pin(
) {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::OutOfWindow,
        false,
        Vd8Ceiling::Shipped,
    )
    .await;
}

// ── 6. Above the atomic-read limits: the same guarantees, from the shared read
//
// With a one-item atomic-read ceiling every layer is above the limits, and
// every regime's state comes from the layer's shared fenced-read flight
// (Epic 3): the catch-up attaches with no slot, checks the flight's cut
// against its own P, registers its stream at that cut and builds from the
// shared snapshot in its build turn. The matrix holds there cell for cell, in
// every regime and both modes — the rollback-mode in-window replay included,
// which hands over to the out-of-window route at its first `TOO_LARGE` and is
// served there — and a sequence regression ends every one of them
// `UNAVAILABLE` before content. The rollback replay's regression is staged
// down to its `have_version`: the case b.h7b left on the interim path, which
// served it zero bytes and acked the pin.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_brand_new_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::BrandNew, true, Vd8Ceiling::Above)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_brand_new_legacy_mode_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::BrandNew, false, Vd8Ceiling::Above)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_in_window_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::InWindow, true, Vd8Ceiling::Above)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_in_window_legacy_mode_replay_hand_over_commit_inside_pin_window_delivered_exactly_once(
) {
    assert_stable_cut_folds_and_acks_window_commit(
        Vd8Regime::InWindowRollback,
        false,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_out_of_window_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(Vd8Regime::OutOfWindow, true, Vd8Ceiling::Above)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_out_of_window_legacy_mode_commit_inside_pin_window_delivered_exactly_once() {
    assert_stable_cut_folds_and_acks_window_commit(
        Vd8Regime::OutOfWindow,
        false,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_brand_new_sequence_regression_ends_unavailable_and_never_acks_the_pin() {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::BrandNew,
        true,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_brand_new_legacy_mode_sequence_regression_ends_unavailable_and_never_acks_the_pin(
) {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::BrandNew,
        false,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_in_window_sequence_regression_ends_unavailable_and_never_acks_the_pin() {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::InWindow,
        true,
        Vd8Ceiling::Above,
    )
    .await;
}

/// The case b.h7b left on the interim path: a rollback-mode replay whose
/// layer's sequence regressed to its own `have_version` was served zero bytes
/// and acked the pin. Through the flight it ends `UNAVAILABLE` before any
/// content or ack.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_in_window_legacy_mode_replay_regression_to_have_version_ends_unavailable_and_never_acks_the_pin(
) {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::InWindowRollback,
        false,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_out_of_window_sequence_regression_ends_unavailable_and_never_acks_the_pin() {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::OutOfWindow,
        true,
        Vd8Ceiling::Above,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cut_out_of_window_legacy_mode_sequence_regression_ends_unavailable_and_never_acks_the_pin(
) {
    assert_sequence_regression_ends_unavailable_and_never_acks_the_pin(
        Vd8Regime::OutOfWindow,
        false,
        Vd8Ceiling::Above,
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.21r — the shared catch-up NotFound recovery policy at consumer-holding sites
// ═══════════════════════════════════════════════════════════════════════════════
//
// b.21r folded the five copies of the catch-up recovery sequence into one
// helper, `read_state_with_recovery`: first read; on `NotFound` one recovery
// attempt; on `Restored` exactly one retry; on a second `NotFound` or
// `NotRecoverable`, shut the site's stream consumer down and then re-bootstrap
// with `RebootstrapCause::StateGone`; any recovery `Err` and any non-`NotFound`
// read error propagates unchanged.
//
// The pre-existing tests drive that policy at the header-read sites with the
// checkpoint hook disabled (NotRecoverable), in the legacy catch-up mode, and
// the recovery-hook suite drives Restored / Transient / Permanent at the one
// site that holds no consumer. The cases below cover what those leave: the
// same outcomes at the sites that DO hold a consumer, the in-window snapshot
// coalesced site's Restored-and-retry, the snapshot-mode out-of-window read,
// and the consumer shutdown that must precede a re-bootstrap on a one-slot
// subscriber pool.
//
// Every case parks the subscriber at an existing b.xmk / b.fdy seam, changes
// the keyspace under it, and releases it. `SUBSCRIBER_DEADLINE` (30 s) bounds
// each wait; the heaviest case awaits one park, one cleanup drain and one
// subscriber task, ~90 s at every budget's full value, inside nextest's 120 s
// hang-killer, and a few hundred milliseconds on the healthy path.
#[cfg(feature = "test-fault-injection")]
mod b21r_recovery_policy {
    use super::*;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointStore, RecoveryHook};

    /// `recovery_metric`'s service label, which `build_pooled_service`'s metrics
    /// are built with.
    const SERVICE: &str = crate::common::RECOVERY_SERVICE_LABEL;

    /// The `snap_seq` the seeded layer checkpoints carry.
    const SNAP_SEQ: u64 = 13;

    /// Replace the service's disabled recovery hook with an enabled one over
    /// `store`, sharing the service's layer store and metrics.
    fn install_recovery(
        svc: &mut LiveEditService,
        store: Arc<dyn CheckpointStore>,
        post_restore_notfound: Option<Arc<AtomicBool>>,
    ) {
        let mut cfg = crate::common::test_config();
        cfg.checkpoint_enabled = true;
        cfg.service_label = SERVICE.to_string();
        let hook = RecoveryHook::new(
            store,
            Arc::clone(&svc.layer_store),
            Arc::new(cfg),
            Arc::clone(&svc.metrics),
        );
        svc.recovery = Arc::new(match post_restore_notfound {
            #[cfg(feature = "test-checkpoint-fault-injection")]
            Some(latch) => hook.with_force_post_restore_notfound_for_test(latch),
            // The latch's seam exists only under the checkpoint feature, and
            // its one caller is gated on it.
            #[cfg(not(feature = "test-checkpoint-fault-injection"))]
            Some(_) => unreachable!(
                "forcing a post-restore NotFound needs `test-checkpoint-fault-injection`"
            ),
            None => hook,
        });
    }

    /// UNLINK the object's `obj:*` keys — optionally sparing its stream, which
    /// the in-window regime needs its start id still retained — leaving the layer
    /// mapping intact: the post-loss keyspace shape recovery runs against.
    async fn wipe_object_state(tv: &PooledTestValkey, object_id: u64, keep_stream: bool) {
        let stream_key = tv.key_schema().layer_stream(object_id);
        let keys: Vec<fred::types::Key> =
            crate::common::scan_object_keys(tv.client(), &tv.prefix, object_id)
                .await
                .into_iter()
                .filter(|k| !keep_stream || k.as_str() != Some(stream_key.as_str()))
                .collect();
        assert!(
            !keys.is_empty(),
            "precondition: the object has state to wipe"
        );
        let _: () = tv
            .client()
            .pool
            .unlink(keys)
            .await
            .expect("UNLINK of the object's keys must succeed");
    }

    /// A layer at seq 6 whose stream no longer holds `2-0`, so a reconnect at
    /// `have_version = 1` is out of window (the XTRIM trick the b.jvx Site C
    /// tests above use).
    async fn out_of_window_scene(
        tv: &PooledTestValkey,
        svc: &LiveEditService,
        tag: &str,
    ) -> (String, u64) {
        let (session_id, object_id) = create_session_and_layer_pooled(
            tv,
            svc,
            &format!("{tag}-session"),
            &format!("{tag}-layer"),
        )
        .await;
        for i in 0..5u64 {
            apply_unique_update(
                svc,
                &session_id,
                &format!("{tag}-layer"),
                object_id,
                &format!("{tag}-writer"),
                100 + i,
            )
            .await;
        }
        let cap: XCap = XCap::try_from(("MINID", "=", "6-0")).expect("build MINID xcap");
        let _: () = tv
            .client()
            .pool
            .xtrim(tv.key_schema().layer_stream(object_id), cap)
            .await
            .expect("XTRIM to force out-of-window must succeed");
        (session_id, object_id)
    }

    /// The first item of the stream, required to be an error status.
    async fn first_status<S>(stream: &mut S) -> Status
    where
        S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
    {
        match tokio::time::timeout(SUBSCRIBER_DEADLINE, stream.next())
            .await
            .expect("the subscription must end within the subscriber deadline")
        {
            Some(Err(status)) => status,
            Some(Ok(msg)) => panic!("expected an error status first, got a message: {msg:?}"),
            None => panic!("expected an error status, got a clean end of stream"),
        }
    }

    /// `(restored, not_recoverable, error)` on the layer recovery-attempt
    /// counter.
    fn recovery_cells(metrics: &Arc<Metrics>) -> (u64, u64, u64) {
        (
            crate::common::recovery_metric(metrics, "restored"),
            crate::common::recovery_metric(metrics, "not_recoverable"),
            crate::common::recovery_metric(metrics, "error"),
        )
    }

    /// At the out-of-window header read — a site that has already opened its
    /// stream consumer — `Restored` is followed by exactly one retry, and a
    /// second `NotFound` on that retry takes the `StateGone` re-bootstrap. With
    /// the mapping still naming the object, that re-bootstrap ends NOT_FOUND
    /// saying the state was lost, not that the layer was deleted.
    ///
    /// The b.sxc post-restore latch re-UNLINKs `obj:hdr` as the restore
    /// returns, which is what makes the retry's `NotFound` deterministic; that
    /// seam exists only under `test-checkpoint-fault-injection`.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_header_restored_then_second_notfound_rebootstraps_as_state_gone() {
        let tv = PooledTestValkey::setup().await;
        let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
        let seam =
            Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_header_read());
        svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, object_id) = out_of_window_scene(&tv, &svc, "b21r-oow-restored").await;

        let store = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &store,
            object_id,
            &session_id,
            "b21r-oow-restored-layer",
            SNAP_SEQ,
        )
        .await;
        let latch = Arc::new(AtomicBool::new(true));
        install_recovery(&mut svc, store, Some(Arc::clone(&latch)));

        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                "b21r-oow-restored-layer",
                object_id,
                1,
                "b21r-sub",
            ),
        )
        .await
        .expect("subscribe must succeed")
        .into_inner();
        tokio::time::timeout(
            SUBSCRIBER_DEADLINE,
            seam.wait_until_parked_pre_header_read(),
        )
        .await
        .expect("the reconnector must park at the out-of-window header-read seam");
        wipe_object_state(&tv, object_id, false).await;
        seam.arrive_at_pre_header_read().await;

        let status = first_status(&mut stream).await;
        assert_eq!(status.code(), tonic::Code::NotFound, "got {status:?}");
        assert!(
            status
                .message()
                .contains("state was lost and could not be recovered"),
            "a StateGone re-bootstrap under a live mapping must say the state was lost, never \
             that the layer was deleted; got {:?}",
            status.message()
        );
        assert_eq!(
            recovery_cells(&svc.metrics),
            (1, 0, 0),
            "exactly one recovery attempt, and it restored: the second NotFound came from the \
             ONE retry and must not trigger a second attempt"
        );
        assert!(
            !latch.load(Ordering::Acquire),
            "precondition: the post-restore latch fired, so the retry really read NotFound"
        );
    }

    /// At the in-window snapshot site's coalesced read (the shipped catch-up
    /// mode), `Restored` is followed by exactly one retry that succeeds, and
    /// the subscriber is served from the restored state on the SAME object —
    /// no re-bootstrap, no error.
    ///
    /// The subscriber parks at the XRANGE seam after its header read; the
    /// object's state is wiped there, sparing its stream so the start id is
    /// still retained and the reconnect still in window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_window_snapshot_coalesced_read_restored_retries_once_and_serves_the_same_object() {
        let tv = PooledTestValkey::setup().await;
        let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
        svc.catchup_snapshot_enabled = true;
        let seam = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
        svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, object_id) = create_session_and_layer_pooled(
            &tv,
            &svc,
            "b21r-iw-restored-session",
            "b21r-iw-restored-layer",
        )
        .await;
        for i in 0..2u64 {
            apply_unique_update(
                &svc,
                &session_id,
                "b21r-iw-restored-layer",
                object_id,
                "b21r-iw-writer",
                100 + i,
            )
            .await;
        }
        let store = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &store,
            object_id,
            &session_id,
            "b21r-iw-restored-layer",
            SNAP_SEQ,
        )
        .await;
        install_recovery(&mut svc, store, None);

        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                "b21r-iw-restored-layer",
                object_id,
                1,
                "b21r-sub",
            ),
        )
        .await
        .expect("subscribe must succeed")
        .into_inner();
        tokio::time::timeout(SUBSCRIBER_DEADLINE, seam.wait_until_parked())
            .await
            .expect("the reconnector must park at the XRANGE seam");
        wipe_object_state(&tv, object_id, true).await;
        seam.release().await;

        // Serve through the caught-up ack; every header is on the original
        // object — a re-bootstrap would name another one, or end the stream.
        loop {
            let (header, _body) = recv_message(&mut stream).await;
            assert_eq!(
                header.object_id, object_id,
                "a Restored retry serves the same object; a re-bootstrap would not"
            );
            if header.total_size == 0 {
                break;
            }
        }
        assert_eq!(
            recovery_cells(&svc.metrics),
            (1, 0, 0),
            "exactly one recovery attempt, which restored, and no further attempt on the retry"
        );
        assert_eq!(
            read_regime_total(&svc.metrics, SERVICE, "in_window", "legacy"),
            1,
            "precondition: the catch-up took the in-window regime (the coalesced snapshot site)"
        );
    }

    /// The snapshot-mode out-of-window read (`compute_faithful_coalesced_at_cut`) under
    /// the overwrite race: the OLD object's keys are cleaned up between the
    /// header read and the coalesced read, recovery is disabled
    /// (`NotRecoverable`), and the subscriber re-bootstraps onto the NEW object.
    /// The legacy (`compute_diff_at_cut`) twin is
    /// `catch_up_out_of_window_compute_diff_notfound_after_cleanup_triggers_rebootstrap`.
    ///
    /// Run with `subscriber_pool` stream readers: with one reader the OLD and
    /// NEW generations' streams are served by the same reader, with eight they
    /// generally are not.
    async fn out_of_window_snapshot_overwrite_rebootstrap(subscriber_pool: u32, tag: &str) {
        let tv = PooledTestValkey::setup_with_pools(8, subscriber_pool).await;
        let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
        svc.catchup_snapshot_enabled = true;
        let seam =
            Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff());
        svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, old_object_id) = out_of_window_scene(&tv, &svc, tag).await;
        let layer = format!("{tag}-layer");

        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(&session_id, &layer, old_object_id, 1, "b21r-sub"),
        )
        .await
        .expect("subscribe must succeed")
        .into_inner();
        tokio::time::timeout(
            SUBSCRIBER_DEADLINE,
            seam.wait_until_parked_pre_compute_diff(),
        )
        .await
        .expect("the reconnector must park at the out-of-window diff-read seam");

        let overwrite_ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
        let new_object_id = handle_create_layer(
            &svc,
            make_request(streaming_create(vec![
                make_create_header(&session_id, &layer),
                make_create_data(overwrite_ompf),
            ])),
        )
        .await
        .expect("CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
        assert_ne!(new_object_id, old_object_id);
        svc.layer_store.wait_for_cleanup(SUBSCRIBER_DEADLINE).await;
        seam.arrive_at_pre_compute_diff().await;

        let (header, _body) = recv_message(&mut stream).await;
        assert_eq!(
            (header.object_id, header.sequence_num),
            (new_object_id, 1),
            "the snapshot-mode coalesced read's NotFound must re-bootstrap onto the NEW object"
        );
        recv_caught_up_ack(&mut stream, header.object_id, header.sequence_num).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_snapshot_coalesced_notfound_after_cleanup_triggers_rebootstrap() {
        out_of_window_snapshot_overwrite_rebootstrap(8, "b21r-oow-snap").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_state_gone_rebootstrap_completes_on_a_one_slot_subscriber_pool() {
        out_of_window_snapshot_overwrite_rebootstrap(1, "b21r-oow-one-slot").await;
    }

    /// A state read that fails with anything other than `NotFound` propagates
    /// unchanged: no recovery attempt, no re-bootstrap. Driven at the
    /// out-of-window coalesced read (snapshot mode, consumer held) by retyping
    /// `obj:hdr` to a string while the subscriber is parked — a permanent
    /// `WRONGTYPE`, which the catch-up surfaces as INTERNAL.
    ///
    /// The recovery hook is ENABLED here, so an attempt would register on its
    /// metric: all three cells staying at zero is the evidence none was made.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_non_notfound_read_error_propagates_without_recovery_or_rebootstrap() {
        let tv = PooledTestValkey::setup().await;
        let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
        svc.catchup_snapshot_enabled = true;
        let seam =
            Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff());
        svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, object_id) = out_of_window_scene(&tv, &svc, "b21r-oow-wrongtype").await;
        install_recovery(&mut svc, Arc::new(InMemoryCheckpointStore::new()), None);

        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                "b21r-oow-wrongtype-layer",
                object_id,
                1,
                "b21r-sub",
            ),
        )
        .await
        .expect("subscribe must succeed")
        .into_inner();
        tokio::time::timeout(
            SUBSCRIBER_DEADLINE,
            seam.wait_until_parked_pre_compute_diff(),
        )
        .await
        .expect("the reconnector must park at the out-of-window diff-read seam");
        let header_key = tv.key_schema().obj_header(&object_id.to_string());
        let _: () = tv
            .client()
            .pool
            .unlink(&header_key)
            .await
            .expect("UNLINK obj:hdr must succeed");
        let _: () = tv
            .client()
            .pool
            .set(&header_key, "not-a-hash", None, None, false)
            .await
            .expect("retyping obj:hdr must succeed");
        seam.arrive_at_pre_compute_diff().await;

        let status = first_status(&mut stream).await;
        assert_eq!(
            status.code(),
            tonic::Code::Internal,
            "a WRONGTYPE state read must propagate as INTERNAL, unchanged; got {status:?}"
        );
        assert_eq!(
            recovery_cells(&svc.metrics),
            (0, 0, 0),
            "a non-NotFound read error must not attempt recovery"
        );
    }

    /// A recovery attempt that ERRORS at a consumer-holding site propagates
    /// its classified status — `Transient` → UNAVAILABLE, anything else →
    /// INTERNAL (b.33e) — rather than re-bootstrapping. The recovery-hook suite
    /// pins the same classification at the one site that holds no consumer.
    ///
    /// Driven at the out-of-window diff read: the object's state is wiped while
    /// the subscriber is parked, and the checkpoint store's restore-source
    /// listing fails with the injected error.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    async fn out_of_window_recovery_error_case(
        error: fn() -> live_edit::checkpoint::CheckpointError,
        expected: tonic::Code,
        tag: &str,
    ) {
        struct ListingFails(fn() -> live_edit::checkpoint::CheckpointError);
        impl live_edit::checkpoint::adapters::CheckpointFaultInjector for ListingFails {
            fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
                if op == "list_layer_checkpoints" {
                    Err((self.0)())
                } else {
                    Ok(())
                }
            }
        }

        let tv = PooledTestValkey::setup().await;
        let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4);
        let seam =
            Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff());
        svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, object_id) = out_of_window_scene(&tv, &svc, tag).await;
        install_recovery(
            &mut svc,
            Arc::new(
                InMemoryCheckpointStore::new().with_fault_injector(Arc::new(ListingFails(error))),
            ),
            None,
        );

        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                &format!("{tag}-layer"),
                object_id,
                1,
                "b21r-sub",
            ),
        )
        .await
        .expect("subscribe must succeed")
        .into_inner();
        tokio::time::timeout(
            SUBSCRIBER_DEADLINE,
            seam.wait_until_parked_pre_compute_diff(),
        )
        .await
        .expect("the reconnector must park at the out-of-window diff-read seam");
        wipe_object_state(&tv, object_id, false).await;
        seam.arrive_at_pre_compute_diff().await;

        let status = first_status(&mut stream).await;
        assert_eq!(status.code(), expected, "got {status:?}");
        assert_eq!(
            recovery_cells(&svc.metrics),
            (0, 0, 1),
            "exactly one recovery attempt, ending in `error`"
        );
    }

    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_recovery_transient_error_ends_the_stream_unavailable() {
        out_of_window_recovery_error_case(
            || live_edit::checkpoint::CheckpointError::Transient {
                source: "injected transient listing failure".into(),
            },
            tonic::Code::Unavailable,
            "b21r-oow-transient",
        )
        .await;
    }

    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_window_recovery_permanent_error_ends_the_stream_internal() {
        out_of_window_recovery_error_case(
            || live_edit::checkpoint::CheckpointError::Permanent {
                source: "injected permanent listing failure".into(),
            },
            tonic::Code::Internal,
            "b21r-oow-permanent",
        )
        .await;
    }
}

/// One stream-reader connection serves subscribers on several layers at once:
/// every `SubscribeReadLayer` catches up and then receives exactly its own
/// layer's live updates, in order, over the single multiplexed XREAD.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_stream_reader_serves_subscribers_on_several_layers() {
    const LAYERS: usize = 3;
    const SUBS_PER_LAYER: usize = 2;
    const UPDATES: u64 = 5;

    let tv = PooledTestValkey::setup_with_pools(2, 1).await;
    let svc = Arc::new(build_pooled_service(
        &tv,
        Arc::new(AllowAllPermissions),
        4096,
    ));
    let mut layers = Vec::new();
    for l in 0..LAYERS {
        let name = format!("mux-layer-{l}");
        let (sid, oid) =
            create_session_and_layer_pooled(&tv, &svc, &format!("mux-{l}"), &name).await;
        layers.push((sid, name, oid));
    }

    let mut streams = Vec::new();
    for (l, (sid, name, oid)) in layers.iter().enumerate() {
        for s in 0..SUBS_PER_LAYER {
            let req = subscribe_request(sid, name, 0, 0, &format!("mux-sub-{l}-{s}"));
            let mut stream = handle_subscribe_read_layer(&svc, req)
                .await
                .expect("subscribe must be admitted")
                .into_inner();
            let (initial, _) = recv_message(&mut stream).await;
            assert_eq!(initial.object_id, *oid);
            recv_caught_up_ack(&mut stream, *oid, 1).await;
            streams.push((l, stream));
        }
    }
    let snap = svc.valkey.stream_mux.snapshot();
    assert_eq!(snap.readers.len(), 1, "fixture: a single reader connection");
    assert_eq!(snap.subscriptions(), LAYERS * SUBS_PER_LAYER);

    for n in 0..UPDATES {
        for (sid, name, oid) in &layers {
            apply_unique_update(&svc, sid, name, *oid, "mux-writer", 100 + n).await;
        }
    }
    let expected: Vec<u64> = (2..2 + UPDATES).collect();
    for (l, stream) in streams.iter_mut() {
        let mut seqs = Vec::new();
        for _ in 0..UPDATES {
            let (header, _) = recv_message(stream).await;
            assert_eq!(header.object_id, layers[*l].2, "no cross-layer delivery");
            seqs.push(header.sequence_num);
        }
        assert_eq!(seqs, expected);
    }
    await_mux_keys(&svc, LAYERS).await;
}

async fn await_mux_keys(svc: &LiveEditService, keys: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while svc.valkey.stream_mux.snapshot().readers[0].keys != keys {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the single reader must read all {keys} layer streams in one XREAD"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.yvh — the in-window regime reads a bounded part of the stream
// ═══════════════════════════════════════════════════════════════════════════════
//
// Every retained entry carries the writer's whole OMPF, so a regime test that
// pulls the in-window slice `XRANGE {have_version+1}-0 + COUNT <MAXLEN>` in
// one reply moves up to a whole retention window per reconnect.
//
// - Snapshot mode (the default) needs only two answers from the stream —
//   whether `{have_version+1}-0` is still there, and whether the last entry is
//   a `deleted` sentinel — so an in-window catch-up reads at most two entries
//   however large the window. Counted off a `MONITOR` feed: every `XRANGE` /
//   `XREVRANGE` a client sent on the layer stream during the catch-up, re-run
//   afterwards on the quiescent stream. The pre-b.yvh slice read all eight.
// - Rollback mode replays every retained commit, so it reads the window in
//   byte-sized pages. A window longer than one page is replayed whole and
//   acked at its last commit; a head trim between two pages (here staged with
//   `ScriptReplyProxy`'s `HoldReplies` on the first page) can leave a hole
//   past the cursor, so the replay re-checks `{have_version+1}-0` after its
//   last page and falls back to Regime 3 rather than replay around the hole.

/// Relay bytes of each large commit in the snapshot-mode test.
const LARGE_RELAY_BYTES: usize = 64 * 1024;

/// Commits past `have_version` in the rollback-mode tests: more than the
/// replay's first page (64, from `stream_mux::DEFAULT_ENTRY_ESTIMATE` against
/// the 1 MiB page target), so the window takes two pages.
const MULTI_PAGE_COMMITS: u64 = 70;

/// Every `XRANGE` / `XREVRANGE` a client (not a script) sent on `stream_key`
/// in `feed`, re-run on the stream now, as `(entries, relay bytes)` returned.
async fn client_stream_reads_returned(
    tv: &PooledTestValkey,
    feed: &[crate::storage::snapshot_read_support::MonitoredCommand],
    stream_key: &str,
) -> (usize, usize) {
    let (mut entries, mut bytes) = (0, 0);
    for cmd in feed {
        let name = cmd.name();
        if cmd.issued_by_script()
            || !matches!(name.as_str(), "XRANGE" | "XREVRANGE")
            || cmd.args.get(1).map(String::as_str) != Some(stream_key)
        {
            continue;
        }
        let count = cmd
            .args
            .iter()
            .position(|a| a.eq_ignore_ascii_case("COUNT"))
            .and_then(|i| cmd.args.get(i + 1))
            .map(|n| n.parse::<u64>().expect("COUNT is numeric"));
        let (first, second) = (cmd.args[2].as_str(), cmd.args[3].as_str());
        let read: Vec<(String, HashMap<String, Bytes>)> = if name == "XRANGE" {
            tv.client()
                .pool
                .xrange_values(stream_key, first, second, count)
                .await
        } else {
            tv.client()
                .pool
                .xrevrange_values(stream_key, first, second, count)
                .await
        }
        .expect("re-running a monitored stream read must succeed");
        entries += read.len();
        bytes += read
            .iter()
            .map(|(_, fields)| fields.get("ompf").map_or(0, Bytes::len))
            .sum::<usize>();
    }
    (entries, bytes)
}

/// Poll `live_subscribe_regime_total{regime, catchup_mode}` until it reaches
/// `target`: the subscription task records it once `run_catch_up` returns.
async fn await_regime(svc: &LiveEditService, regime: &str, mode: &str, target: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while read_regime_total(&svc.metrics, &svc.service_label, regime, mode) < target {
        assert!(
            tokio::time::Instant::now() < deadline,
            "regime {regime}/{mode} did not reach {target}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Snapshot mode, in window, eight 64 KiB commits behind: the catch-up's
/// stream reads return at most two entries — the point read of
/// `{have_version+1}-0` and the last entry — where the slice returned all
/// eight. The regime is still `in_window`, and the content is one coalesced
/// diff to the header sequence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_in_window_catch_up_reads_at_most_two_stream_entries() {
    const BEHIND: u64 = 8;
    let tv = PooledTestValkey::setup().await;
    let mut svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    svc.catchup_snapshot_enabled = true;
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "iw-point-session", "iw-point-layer").await;
    apply_unique_update(
        &svc,
        &session_id,
        "iw-point-layer",
        object_id,
        "writer",
        100,
    )
    .await;
    let have_version = 2;
    for i in 0..BEHIND {
        let delta = incremental_diff_change_section(200 + i);
        let pad = vec![0x5A_u8; LARGE_RELAY_BYTES];
        let relay = Bytes::from(delta_to_ompf_bytes(&delta, &[&pad]));
        svc.layer_store
            .update_layer(
                &session_id,
                "iw-point-layer",
                object_id,
                Some(delta),
                Some(relay),
                "writer",
                1_000 + i,
            )
            .await
            .expect("a large commit must succeed");
    }
    let current = have_version + BEHIND;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let mut tap = crate::storage::snapshot_read_support::MonitorTap::start(
        &crate::storage::snapshot_read_support::test_valkey_addr(),
    )
    .await;

    let sub_req = subscribe_request(
        &session_id,
        "iw-point-layer",
        object_id,
        have_version,
        "sub-iw-point",
    );
    let mut stream = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed")
        .into_inner();
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.sequence_num, current,
        "one coalesced diff to the header"
    );
    assert!(!body.is_empty());
    recv_caught_up_ack(&mut stream, object_id, current).await;
    await_regime(&svc, "in_window", "legacy", 1).await;

    let fence = crate::storage::snapshot_read_support::monitor_fence(&tv.prefix);
    let _: Option<String> = tv
        .client()
        .pool
        .get(&fence)
        .await
        .expect("the fence GET must succeed");
    let feed = tap.drain_until(&fence).await;
    let (entries, bytes) = client_stream_reads_returned(&tv, &feed, &stream_key).await;
    assert!(
        entries <= 2,
        "the in-window catch-up must read at most two stream entries; its reads return \
         {entries} ({bytes} relay bytes) of the {BEHIND}-entry window"
    );
    assert!(
        bytes <= 2 * (LARGE_RELAY_BYTES + 1024),
        "at most two entries' relay bytes; got {bytes}"
    );
}

/// Rollback mode, a window of more than one page: every commit is replayed,
/// in order, and the catch-up acks the last one. The `MONITOR` feed shows the
/// read took two pages and the window-start re-check.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rollback_in_window_replay_longer_than_one_page_replays_every_commit_and_acks_the_last() {
    let tv = PooledTestValkey::setup().await;
    let svc = build_pooled_service(&tv, Arc::new(AllowAllPermissions), 4096);
    assert!(!svc.catchup_snapshot_enabled, "fixture: rollback mode");
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "iw-pages-session", "iw-pages-layer").await;
    for i in 0..MULTI_PAGE_COMMITS {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-pages-layer",
            object_id,
            "writer",
            100 + i,
        )
        .await;
    }
    let current = 1 + MULTI_PAGE_COMMITS;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let mut tap = crate::storage::snapshot_read_support::MonitorTap::start(
        &crate::storage::snapshot_read_support::test_valkey_addr(),
    )
    .await;

    let sub_req = subscribe_request(&session_id, "iw-pages-layer", object_id, 1, "sub-iw-pages");
    let mut stream = handle_subscribe_read_layer(&svc, sub_req)
        .await
        .expect("subscribe must succeed")
        .into_inner();
    let mut seqs = Vec::new();
    for _ in 0..MULTI_PAGE_COMMITS {
        let (header, _) = recv_message(&mut stream).await;
        seqs.push(header.sequence_num);
    }
    assert_eq!(
        seqs,
        (2..=current).collect::<Vec<_>>(),
        "every commit of the window, replayed in order"
    );
    recv_caught_up_ack(&mut stream, object_id, current).await;

    let fence = crate::storage::snapshot_read_support::monitor_fence(&tv.prefix);
    let _: Option<String> = tv
        .client()
        .pool
        .get(&fence)
        .await
        .expect("the fence GET must succeed");
    let feed = tap.drain_until(&fence).await;
    let reads: Vec<Vec<String>> = feed
        .iter()
        .filter(|c| {
            !c.issued_by_script()
                && c.name() == "XRANGE"
                && c.args.get(1).map(String::as_str) == Some(stream_key.as_str())
        })
        .map(|c| c.args[2..].to_vec())
        .collect();
    let pages = reads.iter().filter(|a| a[1] == "+").count();
    assert!(
        pages >= 2,
        "fixture: the replay must have taken more than one page; reads {reads:?}"
    );
    assert!(
        reads.iter().any(|a| a[0] == "2-0" && a[1] == "2-0"),
        "a multi-page read re-checks that the window start is still held; reads {reads:?}"
    );
}

/// Rollback mode, a multi-page window trimmed between its pages: the first
/// page is read, then a head trim (`XTRIM MINID = 68-0`) removes the window
/// start and two entries past the cursor. The next page starts beyond the
/// hole; the replay's window-start re-check finds `2-0` gone and the catch-up
/// falls back to Regime 3 — one coalesced diff to the header sequence — instead
/// of replaying around the hole and acking commits it never delivered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rollback_in_window_replay_trimmed_between_pages_falls_back_to_out_of_window() {
    use crate::storage::snapshot_read_support::{ProxyAction, ReplyGate, ScriptReplyProxy};
    use std::sync::atomic::{AtomicBool, Ordering};

    let gate = ReplyGate::new();
    let armed = Arc::new(AtomicBool::new(false));
    let stream_key_slot: Arc<std::sync::Mutex<String>> =
        Arc::new(std::sync::Mutex::new(String::new()));
    let rule: crate::storage::snapshot_read_support::ProxyRule = {
        let (gate, armed, key) = (
            gate.clone(),
            Arc::clone(&armed),
            Arc::clone(&stream_key_slot),
        );
        Arc::new(move |args: &[Vec<u8>]| {
            let is_first_page = args.len() >= 4
                && args[0].eq_ignore_ascii_case(b"XRANGE")
                && args[1] == key.lock().unwrap().as_bytes()
                && args[2] == b"2-0"
                && args[3] == b"+";
            // Once: the hold covers the first page only.
            (is_first_page && armed.swap(false, Ordering::SeqCst))
                .then(|| ProxyAction::HoldReplies(gate.clone()))
        })
    };
    // Declared before the fixture, so the fixture drops (and cleans up
    // through the proxy) first.
    let proxy = ScriptReplyProxy::start(
        crate::storage::snapshot_read_support::test_valkey_addr(),
        rule,
    )
    .await;
    let tv = PooledTestValkey::setup_with_url(proxy.url()).await;
    // The trim goes through a connection of its own, never the held one.
    let direct = live_edit::storage::test_support::TestValkey::setup().await;
    let svc = Arc::new(build_pooled_service(
        &tv,
        Arc::new(AllowAllPermissions),
        4096,
    ));
    assert!(!svc.catchup_snapshot_enabled, "fixture: rollback mode");
    let (session_id, object_id) =
        create_session_and_layer_pooled(&tv, &svc, "iw-trim-session", "iw-trim-layer").await;
    for i in 0..MULTI_PAGE_COMMITS {
        apply_unique_update(
            &svc,
            &session_id,
            "iw-trim-layer",
            object_id,
            "writer",
            100 + i,
        )
        .await;
    }
    let current = 1 + MULTI_PAGE_COMMITS;
    let stream_key = tv.key_schema().layer_stream(object_id);
    *stream_key_slot.lock().unwrap() = stream_key.clone();
    armed.store(true, Ordering::SeqCst);

    let subscribing = {
        let svc = Arc::clone(&svc);
        let sub_req = subscribe_request(&session_id, "iw-trim-layer", object_id, 1, "sub-iw-trim");
        tokio::spawn(async move { handle_subscribe_read_layer(&svc, sub_req).await })
    };
    // The first page has run on the store; its reply is held at the proxy.
    gate.reply_held().await;
    let cap: XCap = XCap::try_from(("MINID", "=", "68-0")).expect("build MINID xcap");
    let _: () = direct
        .client()
        .pool
        .xtrim(&stream_key, cap)
        .await
        .expect("the staged head trim must succeed");
    gate.open();

    let mut stream = tokio::time::timeout(SUBSCRIBER_DEADLINE, subscribing)
        .await
        .expect("the subscribe must return within the deadline")
        .expect("the subscribe task must not panic")
        .expect("subscribe must succeed")
        .into_inner();
    assert_eq!(proxy.hits(), 1, "fixture: the first page was held once");
    let (header, body) = recv_message(&mut stream).await;
    assert_eq!(
        header.sequence_num, current,
        "a window trimmed between pages must fall back to Regime 3's one coalesced diff to the \
         header ({current}); a per-commit message at {} is the replay acking around the hole",
        header.sequence_num
    );
    assert!(!body.is_empty());
    recv_caught_up_ack(&mut stream, object_id, current).await;
    assert_no_next_message(&mut stream, Duration::from_millis(500)).await;
    await_regime(&svc, "out_of_window", "legacy", 1).await;
    assert_eq!(
        read_regime_total(&svc.metrics, &svc.service_label, "in_window", "legacy"),
        0,
        "the trimmed window is not served in-window"
    );
}
