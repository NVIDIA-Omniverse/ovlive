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

// Integration tests for cross-replica DeleteSession cancellation (bugs b.iou,
// b.ovm sub-task 1).
//
// Two in-process `LiveEditService` instances share a single Valkey namespace.
// When `DeleteSession` runs on replica B, the `le:session:deleted` pub/sub
// fanout must drive replica A to remove its local `session_tokens` entry and
// cancel the associated `CancellationToken`, so that long-lived streams on A
// terminate: a `SubscribeReadLayer` with `NOT_FOUND`, a `JoinSession` with
// `OK` after its teardown.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::BufMut;
use dashmap::DashMap;
use futures::StreamExt;
use prost::Message;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::codec::{BufferSettings, Streaming};
use tonic::Request;
use tonic_prost::ProstCodec;

use live_edit::config::{AuthProtocol, Config};
use live_edit::error::ServiceError;
use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::{
    join_session_request, join_session_response, DeleteSessionRequest, JoinSessionHeader,
    JoinSessionRequest, SubscribeReadLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{
    acquire_session_cancel, handle_create_session, handle_delete_session,
    spawn_session_delete_subscriber, subscribe_session_deleted_channel, SessionCancel,
};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::presence::LocalPresenceRouter;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Request builders ─────────────────────────────────────────────────────────

fn create_session_request(
    path: &str,
    name: &str,
) -> Request<live_edit::grpc::proto::CreateSessionRequest> {
    let mut request = Request::new(live_edit::grpc::proto::CreateSessionRequest {
        path: path.to_owned(),
        name: name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn delete_session_request(session_id: &str) -> Request<DeleteSessionRequest> {
    let mut request = Request::new(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

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

// ── Two-replica fixture ──────────────────────────────────────────────────────

/// Connect a second `ValkeyClient` pointing at the same Valkey instance but
/// with its own subscriber + pool. The returned client shares the *prefix* of
/// the supplied `TestValkey` so both replicas operate on the same keyspace
/// and pub/sub channel names (cross-replica coordination).
async fn connect_peer_client() -> ValkeyClient {
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: 1,
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
        // 30 s matches `TestValkey`'s post-b.1kf default — the
        // workspace suite's SYN-burst recovery profile under
        // listen-queue pressure needs that budget to drain reliably
        // (b.1kf, b.pws). Routing through `connect_test_client`
        // below also gates the connect on the shared 16-permit
        // setup-semaphore.
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
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
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

    live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("peer ValkeyClient must connect")
}

/// Build a `LiveEditService` backed by the given ValkeyClient and KeySchema.
/// Unlike `helpers::build_service`, this variant lets the caller supply the
/// ValkeyClient + KeySchema (so two replicas can share a Valkey + prefix while
/// each holds an independent `session_tokens` DashMap and subscriber client).
fn build_service_with_client(
    valkey: ValkeyClient,
    key_schema: KeySchema,
) -> (LiveEditService, Arc<DashMap<String, SessionCancel>>) {
    let session_store = Arc::new(SessionStore::new(valkey.clone(), key_schema.clone()));
    let layer_store = Arc::new(LayerStore::new(valkey.clone(), key_schema.clone(), 4096));
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );
    // Inline session-recovery hook construction (the helpers' canonical
    // version takes a `&TestValkey`; this builder threads a raw
    // `ValkeyClient` + `KeySchema` so we mirror the disabled-hook recipe
    // here rather than refactoring two helper signatures for one call site).
    let mut session_recovery_config = crate::common::test_config();
    session_recovery_config.checkpoint_enabled = false;
    let session_recovery_store: Arc<dyn live_edit::checkpoint::CheckpointStore> =
        Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    let session_recovery = Arc::new(live_edit::checkpoint::SessionRecoveryHook::new(
        session_recovery_store,
        valkey.clone(),
        key_schema.clone(),
        Arc::new(session_recovery_config),
        Arc::clone(&metrics),
    ));
    // Built before the literal because `valkey` / `key_schema` are moved into
    // it below (this site holds them as locals rather than a `&TestValkey`).
    let base_bootstrap = crate::common::build_disabled_base_bootstrap(
        valkey.clone(),
        key_schema.clone(),
        Arc::clone(&metrics),
    );

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::clone(&session_tokens),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            valkey.clone(),
            key_schema.clone(),
        ),
        valkey,
        key_schema,
        presence_router: Arc::new(LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics,
        service_label: "test".to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        base_bootstrap,
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
    };

    (svc, session_tokens)
}

/// Poll `check` until it returns true or `deadline` elapses.
async fn wait_until<F>(deadline: Duration, mut check: F) -> bool
where
    F: FnMut() -> bool,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if check() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    check()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Regression: cross-replica cleanup of session_tokens entry ────────────
//
// This is the regression test. Before the fix, DeleteSession on replica B did
// not propagate to replica A, so replica A's `session_tokens` entry stayed
// forever (and its `CancellationToken` never fired). With the fix, replica B
// publishes on `le:session:deleted` and replica A's subscriber task removes
// + cancels the token.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_session_on_peer_removes_and_cancels_local_token() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    // ── Replica A: creates the session. ──────────────────────────────────────
    let (svc_a, tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    // ── Replica A subscribes to the session-deleted channel so a DeleteSession
    //    routed to B can cancel A's local token. ───────────────────────────────
    subscribe_session_deleted_channel(&tv.client().subscriber, &key_schema)
        .await
        .expect("subscribe on replica A must succeed");
    let cancel_a = CancellationToken::new();
    let subscriber_a = spawn_session_delete_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&tokens_a),
        live_edit::grpc::session_cache::SessionBasePathCache::new(),
        key_schema.clone(),
        Arc::clone(&svc_a.metrics),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel_a.clone(),
    );

    // ── Replica B: will handle the DeleteSession. Uses its own ValkeyClient. ─
    let peer_client = connect_peer_client().await;
    let (svc_b, _tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());

    // ── Create the session on A. ─────────────────────────────────────────────
    let response = handle_create_session(
        &svc_a,
        create_session_request("/stage", "cross-replica-session"),
    )
    .await
    .expect("CreateSession on A must succeed");
    let session_id = response.into_inner().session_id;

    // Preconditions: A has the token, B does not.
    assert!(
        tokens_a.contains_key(&session_id),
        "replica A must hold the CancellationToken after CreateSession"
    );
    let a_token = tokens_a
        .get(&session_id)
        .expect("A's token must be present")
        .value()
        .clone();
    assert!(
        !a_token.is_cancelled(),
        "A's token must not be cancelled before DeleteSession"
    );

    // ── Delete the session on B (different replica). ─────────────────────────
    handle_delete_session(&svc_b, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on B must succeed");

    // ── Assert: A's subscriber eventually removes the local entry and cancels
    //    the token. Pub/sub delivery is asynchronous so we poll. ──────────────
    let removed = wait_until(Duration::from_secs(2), || {
        !tokens_a.contains_key(&session_id)
    })
    .await;
    assert!(
        removed,
        "replica A must remove its session_tokens entry after pub/sub fanout"
    );

    let cancelled = wait_until(Duration::from_secs(2), || a_token.is_cancelled()).await;
    assert!(
        cancelled,
        "replica A must cancel its CancellationToken after pub/sub fanout"
    );

    // ── Clean shutdown of the subscriber task. ───────────────────────────────
    cancel_a.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_a).await;

    peer_client.quit().await;
}

// ── 2. Cross-replica stream termination ──────────────────────────────────────
//
// A `SubscribeReadLayer` stream on replica A runs with `session_cancel =
// Some(token)` (because A handled CreateSession). When DeleteSession runs on
// replica B, the pub/sub fanout cancels A's token and A's stream terminates.
//
// DeleteSession also cascades through layer deletion, which appends a
// `deleted` sentinel to each layer's stream, so A's subscriber ends on
// whichever signal it sees first — the token (`NOT_FOUND "session was
// deleted"`) or the sentinel (`NOT_FOUND "layer '<name>' was deleted"`). Only
// the code is fixed, so only the code is asserted: a deletion ends the stream
// `NOT_FOUND`, never a clean end (which would be indistinguishable from a
// shutdown the client must retry) and never `UNAVAILABLE` (which it would
// retry into a session that no longer exists). A's session_tokens entry being
// gone afterwards proves the pub/sub fanout ran, not only the sentinel path.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_stream_on_local_replica_terminates_when_peer_deletes() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    let (svc_a, tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    subscribe_session_deleted_channel(&tv.client().subscriber, &key_schema)
        .await
        .expect("subscribe_session_deleted_channel on A must succeed");
    let cancel_a = CancellationToken::new();
    let subscriber_a = spawn_session_delete_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&tokens_a),
        live_edit::grpc::session_cache::SessionBasePathCache::new(),
        key_schema.clone(),
        Arc::clone(&svc_a.metrics),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel_a.clone(),
    );

    let peer_client = connect_peer_client().await;
    let (svc_b, _tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());

    // Create session + layer on A. `create_session_and_layer` bypasses auth for
    // session creation, so insert the CancellationToken manually to match the
    // state after a real CreateSession RPC.
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc_a, "cross-rep-sub", "layer-a").await;
    svc_a
        .session_tokens
        .insert(session_id.clone(), SessionCancel::new());

    // Start the subscribe stream on A. It binds to A's CancellationToken.
    let response = handle_subscribe_read_layer(
        &svc_a,
        subscribe_request(&session_id, "layer-a", object_id, 1, "subscriber-a"),
    )
    .await
    .expect("SubscribeReadLayer on A must succeed");
    let mut stream = response.into_inner();

    // Drain the Case-3 ack header so the stream is in the notification loop.
    let _first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("ack header must arrive")
        .expect("stream must not be empty")
        .expect("ack header must be Ok");

    // DeleteSession on replica B.
    handle_delete_session(&svc_b, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on B must succeed");

    // The stream on A must terminate within a short deadline, with a status.
    let ending = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match stream.next().await {
                None => break None,
                Some(Err(status)) => break Some(status),
                Some(Ok(_)) => continue, // drain any residual chunks
            }
        }
    })
    .await
    .expect("stream must terminate within deadline");
    let status = ending.expect(
        "A's subscribe stream ended cleanly (gRPC OK) after a peer DeleteSession; a deletion \
         must end it NOT_FOUND",
    );
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "a peer DeleteSession must end A's stream NOT_FOUND whichever signal wins; got {status:?}"
    );

    // Prove the pub/sub fanout actually ran: A's session_tokens entry must be
    // gone. This rules out a test that passes purely because of the layer
    // `Deleted` notification path.
    let removed = wait_until(Duration::from_secs(2), || {
        !tokens_a.contains_key(&session_id)
    })
    .await;
    assert!(
        removed,
        "replica A's session_tokens entry must be removed by pub/sub fanout"
    );

    cancel_a.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_a).await;

    peer_client.quit().await;
}

// ── 3. Double-cancel safety: peer receives fanout for a session it never saw ─
//
// Replica A creates + deletes a session; replica B never had a local entry.
// B's subscriber must process the pub/sub message as a silent no-op (no
// panic, no error). This regression-guards the idempotency required by
// `apply_session_deleted`.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peer_subscriber_noop_on_unknown_session_id() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    // Replica A: no subscriber — it handles CreateSession and DeleteSession.
    let (svc_a, tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    // Replica B: installs the subscriber on an *independent* Valkey client.
    let peer_client = connect_peer_client().await;
    let (_svc_b, tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());
    subscribe_session_deleted_channel(&peer_client.subscriber, &key_schema)
        .await
        .expect("subscribe on B must succeed");
    let cancel_b = CancellationToken::new();
    let subscriber_b = spawn_session_delete_subscriber(
        peer_client.subscriber.clone(),
        Arc::clone(&tokens_b),
        live_edit::grpc::session_cache::SessionBasePathCache::new(),
        key_schema.clone(),
        Arc::new(crate::common::metrics_scrape_only("test")),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel_b.clone(),
    );

    // Create + delete on A. B never had an entry for this session.
    let response = handle_create_session(&svc_a, create_session_request("/stage", "noop-session"))
        .await
        .expect("CreateSession on A must succeed");
    let session_id = response.into_inner().session_id;
    assert!(!tokens_b.contains_key(&session_id));
    assert!(tokens_a.contains_key(&session_id));

    handle_delete_session(&svc_a, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on A must succeed");

    // Allow time for the pub/sub to be delivered on B. We can't observe a
    // state change on B (no entry to begin with), so wait a short grace
    // period and then assert the subscriber is still alive by sending a
    // cancel.
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(
        !tokens_b.contains_key(&session_id),
        "replica B must not invent an entry on pub/sub"
    );
    assert!(
        !subscriber_b.is_finished(),
        "replica B's subscriber must still be running (silent no-op, not panic)"
    );

    cancel_b.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_b).await;

    peer_client.quit().await;
}

// ── 4. Regression for b.uue: subscribe on a replica that didn't see CreateSession ─
//
// Bug b.uue: when CreateSession runs on replica A and SubscribeReadLayer runs
// on replica B, replica B's per-process `session_tokens` map has no entry for
// the session, so the subscriber's `session_cancel` is `None`. A subsequent
// DeleteSession on any replica triggers the `le:session:deleted` pub/sub
// fanout, but on B that fanout is a silent no-op (no local entry to remove or
// cancel) — so the subscriber's stream is never told via the session-cancel
// path that the session is gone. CLAUDE.md guarantees "any replica serves any
// RPC", so this gap silently breaks the multi-replica invariant.
//
// The fix: in `handle_subscribe_read_layer`, when the local `session_tokens`
// map has no entry for the session, insert a fresh `CancellationToken` keyed
// by `session_id` before binding it as `session_cancel`. The existing pub/sub
// delete consumer already cancels by `session_id`, so closing this gap needs
// no new wire path.
//
// Regression assertion: the *fix's contract* is that after SubscribeReadLayer
// runs on a replica that did not see CreateSession, the local `session_tokens`
// map MUST contain an entry for that `session_id` and that entry's token MUST
// fire when a peer replica deletes the session. The pre-fix code leaves the
// map empty, so the first assertion fails immediately.
//
// We deliberately assert on the local-map contract (not just stream
// termination) because DeleteSession's per-layer cascade also appends a
// `deleted` sentinel to the layer's stream, which terminates the subscriber
// via NOT_FOUND regardless of the session-cancel path — so a "stream
// terminates" assertion alone passes against the buggy code via the
// deleted-sentinel path. Asserting
// that the local entry exists *and* gets cancelled isolates the
// session-cancel path that the fix actually repairs.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_on_non_create_replica_inserts_local_cancel_token() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    // Replica A handles CreateSession + CreateLayer.
    let (svc_a, tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    // Replica B handles SubscribeReadLayer. Its `session_tokens` is empty —
    // it never saw the CreateSession, so the bug surfaces here.
    let peer_client = connect_peer_client().await;
    let (svc_b, tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());

    // Wire up the session-deleted subscriber on replica B so the cross-replica
    // delete fanout can cancel B's local token (the very entry the fix
    // creates on subscribe).
    subscribe_session_deleted_channel(&peer_client.subscriber, &key_schema)
        .await
        .expect("subscribe_session_deleted_channel on B must succeed");
    let cancel_b_subscriber = CancellationToken::new();
    let subscriber_b = spawn_session_delete_subscriber(
        peer_client.subscriber.clone(),
        Arc::clone(&tokens_b),
        live_edit::grpc::session_cache::SessionBasePathCache::new(),
        key_schema.clone(),
        Arc::clone(&svc_b.metrics),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel_b_subscriber.clone(),
    );

    // Create session + layer on A. `create_session_and_layer` bypasses the
    // CreateSession RPC, so manually insert A's local CancellationToken to
    // mirror the post-CreateSession state on the replica that actually
    // handled it.
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc_a, "buue-cross-rep", "layer-buue").await;
    svc_a
        .session_tokens
        .insert(session_id.clone(), SessionCancel::new());

    // Sanity check: B's session_tokens has NO entry yet — this is the
    // pre-subscribe state that the bug exploits.
    assert!(
        !tokens_b.contains_key(&session_id),
        "precondition: replica B must not hold an entry before SubscribeReadLayer"
    );

    // Subscribe on replica B. The fix inserts a fresh CancellationToken into
    // B's `session_tokens` keyed by session_id. Without the fix, B's map
    // stays empty and the subscriber's `session_cancel` is `None`.
    let response = handle_subscribe_read_layer(
        &svc_b,
        subscribe_request(&session_id, "layer-buue", object_id, 1, "subscriber-b"),
    )
    .await
    .expect("SubscribeReadLayer on B must succeed");
    let mut stream = response.into_inner();

    // Drain the catch-up ack header so the spawned subscriber task has run
    // through `handle_subscribe_read_layer` (and thus through the fix's
    // insert-if-missing branch) and is parked in steady state.
    let _ack = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("ack header must arrive within deadline")
        .expect("stream must not be empty")
        .expect("ack header must be Ok");

    // ── Assertion 1 (regression): replica B has a local CancellationToken
    //    for this session. Pre-fix, this assertion fails (map empty).
    assert!(
        tokens_b.contains_key(&session_id),
        "fix contract: SubscribeReadLayer on a replica that did not handle \
         CreateSession must insert a local CancellationToken so cross-replica \
         DeleteSession fanout can cancel the subscriber"
    );
    let b_token = tokens_b
        .get(&session_id)
        .expect("B's token must be present after subscribe")
        .value()
        .clone();
    assert!(
        !b_token.is_cancelled(),
        "B's freshly-inserted token must not be cancelled before DeleteSession"
    );

    // ── Trigger DeleteSession on replica A (any replica works). ──────────────
    handle_delete_session(&svc_a, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on A must succeed");

    // ── Assertion 2: the cross-replica pub/sub fanout cancels B's token and
    //    removes the entry. Without Assertion 1's contract being honored,
    //    there'd be no entry for the fanout to find on B, so this would also
    //    silently no-op (the original bug behavior). ────────────────────────
    let cancelled = wait_until(Duration::from_secs(2), || b_token.is_cancelled()).await;
    assert!(
        cancelled,
        "B's token must be cancelled by the cross-replica session-deleted \
         pub/sub fanout"
    );
    let removed = wait_until(Duration::from_secs(2), || {
        !tokens_b.contains_key(&session_id)
    })
    .await;
    assert!(
        removed,
        "B's session_tokens entry must be removed after the pub/sub fanout"
    );

    // Replica A's behaviour is exercised by the existing tests in this file;
    // we don't reassert it here. Drain any residual stream messages the
    // subscriber emitted before termination so the receiver doesn't leak.
    let _terminated = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match stream.next().await {
                None => break true,
                Some(Err(_)) => break true,
                Some(Ok(_)) => continue,
            }
        }
    })
    .await
    .expect("stream must terminate within deadline");

    // Drop the subscribe stream so its spawned task can exit cleanly when the
    // sender side is closed.
    drop(stream);

    // Clean shutdown of the pub/sub subscriber task.
    cancel_b_subscriber.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_b).await;

    // Touch tokens_a so the unused-binding warning doesn't fire — the entry
    // is incidental to this test (the fix is on B, not A).
    let _ = tokens_a;

    peer_client.quit().await;
}

// ── 5. Regression for b.uue (JoinSession): bidi stream on a non-create replica ─
//
// Companion to test #4 covering the second consumer of the b.uue fix:
// `handle_join_session` had the same `.get(session_id)`-only pattern on
// `session_tokens` as `handle_subscribe_read_layer`, so a JoinSession landing
// on a replica that did not handle CreateSession ran with `session_cancel ==
// None` and a subsequent DeleteSession (anywhere in the cluster) could not
// terminate the bidi loop. The fix routes both handlers through the shared
// `acquire_session_cancel` helper.
//
// Why the assertions matter for THIS RPC: unlike `SubscribeReadLayer`,
// `DeleteSession`'s per-layer cascade publishes nothing that reaches the
// JoinSession bidi loop — the loop's `tokio::select!` watches client_stream,
// presence pub/sub events, and `session_cancel.cancelled()`. With
// `session_cancel == None` (pre-fix), DeleteSession can't break the loop, so
// the bidi stream stays open until the *client* disconnects. The
// stream-termination assertion is therefore directly diagnostic for this RPC.
// We also assert on the local-map contract for parity with test #4 and to
// pinpoint exactly where the fix takes effect.

/// Build a `Streaming<JoinSessionRequest>` backed by an open `mpsc` channel.
/// Dropping the returned sender closes the client side, which surfaces as
/// `Ok(None)` in `client_stream.message()` inside the bidi loop.
fn open_join_streaming() -> (
    Streaming<JoinSessionRequest>,
    mpsc::UnboundedSender<bytes::Bytes>,
) {
    use http_body_util::StreamBody;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, rx) = mpsc::unbounded_channel::<bytes::Bytes>();
    let rx_stream = UnboundedReceiverStream::new(rx)
        .map(|b| Ok::<_, std::io::Error>(http_body::Frame::data(b)));
    let body = StreamBody::new(rx_stream);
    let decoder = ProstCodec::<(), JoinSessionRequest>::raw_decoder(BufferSettings::default());
    let streaming = Streaming::new_request(decoder, body, None, None);
    (streaming, tx)
}

/// Encode a single JoinSessionRequest as a gRPC frame.
fn encode_join_frame(msg: &JoinSessionRequest) -> bytes::Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0); // no compression
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

fn make_join_request(
    stream: Streaming<JoinSessionRequest>,
) -> Request<Streaming<JoinSessionRequest>> {
    let mut req = Request::new(stream);
    req.extensions_mut().insert(BearerToken::new("test-token"));
    req
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_on_non_create_replica_terminates_when_peer_deletes() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    // Replica A handles CreateSession.
    let (svc_a, _tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    // Replica B handles JoinSession. Its `session_tokens` map is empty —
    // the bug surfaces here.
    let peer_client = connect_peer_client().await;
    let (svc_b, tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());

    // Wire up the session-deleted subscriber on replica B so the cross-replica
    // delete fanout can cancel the freshly-inserted token the fix creates.
    subscribe_session_deleted_channel(&peer_client.subscriber, &key_schema)
        .await
        .expect("subscribe_session_deleted_channel on B must succeed");
    let cancel_b_subscriber = CancellationToken::new();
    let subscriber_b = spawn_session_delete_subscriber(
        peer_client.subscriber.clone(),
        Arc::clone(&tokens_b),
        live_edit::grpc::session_cache::SessionBasePathCache::new(),
        key_schema.clone(),
        Arc::clone(&svc_b.metrics),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel_b_subscriber.clone(),
    );

    // Create the session on A (real handler so A's session_tokens is populated
    // the same way it would be in production).
    let response = handle_create_session(
        &svc_a,
        create_session_request("/stage", "buue-join-session"),
    )
    .await
    .expect("CreateSession on A must succeed");
    let session_id = response.into_inner().session_id;

    // Sanity: B's session_tokens has no entry pre-JoinSession.
    assert!(
        !tokens_b.contains_key(&session_id),
        "precondition: replica B must not hold an entry before JoinSession"
    );

    // Open a JoinSession bidi stream on B. The channel-backed stream stays
    // open after the header, so the bidi loop runs until either
    // session_cancel fires or we drop `client_tx` to close the client side.
    let (client_stream, client_tx) = open_join_streaming();
    client_tx
        .send(encode_join_frame(&JoinSessionRequest {
            request_type: Some(join_session_request::RequestType::Header(
                JoinSessionHeader {
                    session_id: session_id.clone(),
                    user_application: "regression-app".to_owned(),
                    client_keepalive: false,
                },
            )),
        }))
        .expect("client channel must accept the header frame");

    let response = handle_join_session(&svc_b, make_join_request(client_stream))
        .await
        .expect("JoinSession on B must succeed");
    let mut response_stream = response.into_inner();

    // Drain SessionConnected so we know the join task has passed its
    // `acquire_session_cancel` call (it runs after registration and *before*
    // SessionConnected). Pre-fix the same call site simply ran `.get()`.
    let connected = tokio::time::timeout(Duration::from_secs(5), response_stream.next())
        .await
        .expect("SessionConnected must arrive within 5s")
        .expect("response stream must yield SessionConnected")
        .expect("first response must not be an error");
    match connected.response_type {
        Some(join_session_response::ResponseType::Connected(_)) => {}
        other => panic!("expected SessionConnected, got {other:?}"),
    }

    // ── Assertion 1 (regression / fix contract): replica B has a local
    //    CancellationToken for this session after JoinSession. Pre-fix this
    //    fails (B's map is empty). `acquire_session_cancel` (SR-4 step 6)
    //    runs before SessionConnected is sent (step 9), so the entry is
    //    already there once SessionConnected has arrived.
    assert!(
        tokens_b.contains_key(&session_id),
        "fix contract: JoinSession on a replica that did not handle \
         CreateSession must insert a local CancellationToken so cross-replica \
         DeleteSession fanout can cancel the bidi loop"
    );
    let b_token = tokens_b
        .get(&session_id)
        .expect("B's token must be present after JoinSession reaches the bidi loop")
        .value()
        .clone();
    assert!(
        !b_token.is_cancelled(),
        "B's freshly-inserted token must not be cancelled before DeleteSession"
    );

    // ── Trigger DeleteSession on replica A. ──────────────────────────────────
    handle_delete_session(&svc_a, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on A must succeed");

    // ── Assertion 2 (end-to-end): the cross-replica pub/sub fanout cancels
    //    B's token AND breaks the bidi loop within a tight deadline. For
    //    JoinSession this assertion is uniquely diagnostic (no layer-cascade
    //    fallback path can terminate the bidi loop). ────────────────────────
    let cancelled = wait_until(Duration::from_secs(2), || b_token.is_cancelled()).await;
    assert!(
        cancelled,
        "B's token must be cancelled by the cross-replica session-deleted \
         pub/sub fanout"
    );

    // The bidi loop returns Ok(()) when session_cancel fires — the response
    // stream then closes (the spawned task's `tx` is dropped). Drain until
    // end-of-stream so we observe termination directly, not just the token
    // state. Any presence events (PeerJoined echo, PeerLeft from cleanup)
    // are tolerated; we only care that the stream terminates.
    let terminated = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            match response_stream.next().await {
                None => break true,
                Some(Err(_)) => break true,
                Some(Ok(_)) => continue,
            }
        }
    })
    .await
    .expect(
        "JoinSession bidi stream must terminate within deadline after cross-replica DeleteSession",
    );
    assert!(
        terminated,
        "JoinSession bidi stream on B must terminate after peer DeleteSession"
    );

    // The pub/sub consumer also removes the entry by key.
    let removed = wait_until(Duration::from_secs(2), || {
        !tokens_b.contains_key(&session_id)
    })
    .await;
    assert!(
        removed,
        "B's session_tokens entry must be removed after the pub/sub fanout"
    );

    // Drop the client tx so the spawned bidi task can exit cleanly.
    drop(client_tx);

    // Clean shutdown of the pub/sub subscriber task.
    cancel_b_subscriber.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_b).await;

    peer_client.quit().await;
}

// ── 6. Regression for b.uue: race-loser branch in `acquire_session_cancel` ──
//
// The b.uue fix hardens `acquire_session_cancel` against a race: between its
// `entry(...).or_insert_with(...)` and its post-insert `get_base_path`
// re-check, a concurrent `DeleteSession` fan-out may land — leaving a dangling
// `CancellationToken` that no future fan-out can find to cancel (pub/sub is
// one-shot per delete). The race-loser branch inside the helper must:
//   (a) remove the entry we just inserted,
//   (b) cancel our local clone of the token, and
//   (c) return `ServiceError::NotFound`.
//
// Why we call the helper directly: `handle_subscribe_read_layer` does an auth
// pre-check via `session_store.get_base_path` BEFORE `acquire_session_cancel`.
// When the session is truly gone, that pre-check short-circuits the handler
// with `NotFound` and the helper is never reached — so an end-to-end handler
// call cannot exercise the race-loser branch without a `SessionStore` fault-
// injection seam. The helper is `pub`, so the integration-test binary invokes
// it directly at the exact post-delete state that triggers the branch.
//
// Diagnostic specificity: assertion 1 (return `NotFound`) would still hold if
// the branch just `return Err(NotFound)`-ed without cleanup. Assertion 2 (map
// does not contain the entry) is the one that breaks when the cleanup is
// neutered — that's the regression guard for this fix's actual contract.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_on_non_create_replica_returns_not_found_when_session_already_deleted() {
    let tv = TestValkey::setup().await;
    let key_schema = tv.key_schema().clone();

    // Replica A: handles CreateSession + DeleteSession.
    let (svc_a, _tokens_a) = build_service_with_client(tv.client().clone(), key_schema.clone());

    // Replica B: the replica that would have handled Subscribe. We only need
    // its session_store + session_tokens to drive `acquire_session_cancel`.
    let peer_client = connect_peer_client().await;
    let (svc_b, tokens_b) = build_service_with_client(peer_client.clone(), key_schema.clone());

    // Create the session on A, then delete it on A.
    let response = handle_create_session(
        &svc_a,
        create_session_request("/stage", "buue-race-loser-session"),
    )
    .await
    .expect("CreateSession on A must succeed");
    let session_id = response.into_inner().session_id;

    handle_delete_session(&svc_a, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on A must succeed");

    // Wait until B observes the post-delete storage state — `get_base_path`
    // on B must return `NotFound`. This drives the helper into the race-loser
    // branch deterministically: the entry insert succeeds, then the post-
    // insert `get_base_path` sees the deleted state.
    let start = Instant::now();
    let mut session_gone = false;
    while start.elapsed() < Duration::from_secs(2) {
        if svc_b
            .session_store
            .get_base_path(&session_id)
            .await
            .is_err()
        {
            session_gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        session_gone,
        "replica B must observe the deleted session (get_base_path → NotFound) within deadline"
    );

    // Sanity: B's session_tokens must not already have an entry — any entry
    // after the call must be one the helper inserted (and then failed to
    // clean up, were the fix neutered).
    assert!(
        !tokens_b.contains_key(&session_id),
        "precondition: B's session_tokens must not hold an entry before the helper call"
    );

    // ── Drive the race-loser branch directly. ───────────────────────────────
    let result = acquire_session_cancel(
        &tokens_b,
        &svc_b.session_store,
        &svc_b.session_base_path_cache,
        Duration::from_secs(10),
        &session_id,
    )
    .await;

    // ── Assertion 1: helper returns `ServiceError::NotFound`. Without the
    //    `get_base_path` re-check inside the helper, the helper would return
    //    `Ok(token)` with a polluted map — breaking this assertion.
    match result {
        Err(ServiceError::NotFound(_)) => {}
        other => panic!(
            "expected acquire_session_cancel to return Err(NotFound) after the session was \
             deleted; got {other:?}"
        ),
    }

    // ── Assertion 2: the helper cleaned up the entry it inserted. This is
    //    the regression guard for the fix's actual contract — neutering
    //    either the `remove` line or the whole cleanup block breaks this.
    assert!(
        !tokens_b.contains_key(&session_id),
        "race-loser branch must remove the entry it inserted so a future DeleteSession fan-out \
         does not see a stale CancellationToken"
    );

    peer_client.quit().await;
}
