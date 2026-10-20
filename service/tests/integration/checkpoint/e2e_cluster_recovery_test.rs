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

//! Cluster-mode end-to-end recovery test — Epic AC #7 cluster gate
//! (Subtask `t3.fgj.ya.wg.r1` under Task `t2.fgj.ya.wg`, Epic
//! `t1.fgj.ya`, Bee `b.fgj`).
//!
//! Pins the same write → checkpoint → wipe → recover contract the sibling
//! single-node E2E tests exercise (`e2e_stat_layer_recovery_test.rs`,
//! `e2e_update_layer_recovery_test.rs`, etc.) — but against a real
//! 3-shard Valkey Cluster. This is the multi-slot regression surface
//! that the `{object_id}` hash-tag layout protects:
//!
//! - The per-object `le:{object_id}:obj:*` keyspace co-locates on a
//!   single shard so the assembler's SCAN walk + multi-key UNLINK stay
//!   slot-safe.
//! - The layer mapping (`le:{session_id}:layer:{hex(name)}`) and the
//!   session metadata (`le:{session_id}:sess`) live on a potentially
//!   DIFFERENT shard, hash-tagged by `{session_id}`. The recovery hook
//!   reaches across both shards (mapping lookup → object-state restore)
//!   on every invocation, so a regression that lost the cross-shard
//!   independence — e.g. accidentally pinning a Lua script across a
//!   slot-boundary — would surface as `CROSSSLOT` here that the
//!   single-node tests cannot catch.
//! - `wipe_layer_state` uses cluster-aware `scan_cluster_buffered` when
//!   `client.is_clustered()` is true, walking every shard for matching
//!   keys. The `{object_id}` hash tag guarantees the matches land on
//!   one shard, but the cluster-aware SCAN call is exactly what the
//!   production assembler and orphan-sweeper use — so this test
//!   exercises the same code path through the helper.
//!
//! ## Gating
//!
//! Skip-on-missing per the orphan-sweep cluster regression pattern at
//! `tests/integration/storage/orphan_sweep_test.rs:952-957`. The repo
//! does not ship a cluster harness; `bash scripts/dev-valkey.sh
//! cluster-up` brings up a 3-shard cluster on `127.0.0.1:7000..7002`
//! and prints the suggested `LIVE_EDIT_TEST_CLUSTER_URL`. CI's
//! `test-cluster:` job runs the same path.
//!
//! Without the env var set the test compiles clean and skips at
//! runtime — the standard contract for cluster-only regressions in
//! this repo.
//!
//! ## Why StatLayer is the recovery trigger
//!
//! `StatLayer` is the cheapest handler-level recovery trigger in the
//! wired set: no streaming-channel allocation, no OMPF encoding, no
//! delta plumbing — just a single unary RPC that exercises the same
//! `handle_*_layer` recovery branch the sibling E2E tests pin
//! (`recovery.rs` step 5; SDD SR-2.4). Matches the trigger pattern of
//! `.4q` (object-id preservation), `.rh` (live-mapping guard), and
//! `.nx` (StatLayer recovery) sibling tests.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::{ClientLike, HashesInterface};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::grpc::layer::handle_stat_layer;
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
use crate::grpc::helpers::{create_session_and_layer, AllowAllPermissions, StubUserinfo};

const SERVICE_LABEL_LOCAL: &str = SERVICE_LABEL;
const SESSION_NAME: &str = "e2e-cluster-recovery-session";
const LAYER_NAME: &str = "e2e-cluster-recovery-layer";

/// Probe the live label-cell counter value for a given outcome label on
/// the `live_edit_checkpoint_recovery_attempts_total{result, scope, service}`
/// metric at `scope="layer"`. Same shape as the sibling E2E tests'.
fn recovery_metric(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", SERVICE_LABEL_LOCAL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Build a `StatLayerRequest` carrying a stub bearer token — matches the
/// `recovery_hook_test.rs` / `e2e_stat_layer_recovery_test.rs` shape.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Cluster-mode harness bundle. Mirrors [`CheckpointHarness`] but exposes
/// the explicit pieces so the cluster test can drive a clean shutdown
/// without going through the single-node `setup_checkpoint_harness`.
struct ClusterCheckpointHarness {
    tv: TestValkey,
    store: Arc<dyn CheckpointStore>,
    svc: LiveEditService,
    metrics: SharedMetrics,
    loop_handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
    _tempdir: TempDir,
}

/// Construct a cluster-mode E2E recovery harness rooted at the supplied
/// `LIVE_EDIT_TEST_CLUSTER_URL`. Mirrors `setup_checkpoint_harness` 1:1
/// except for the `TestValkey::setup_with_url(cluster_url, 8)` constructor
/// — the single-node `TestValkey::setup()` reads `TEST_VALKEY_URL` and
/// cannot be repointed at a cluster URL.
///
/// Pool size `8` matches the cluster-aware orphan-sweep regression test
/// at `tests/integration/storage/orphan_sweep_test.rs:957` (`4`) with
/// headroom for the concurrent recovery + checkpoint-loop demand.
async fn setup_cluster_checkpoint_harness(cluster_url: String) -> ClusterCheckpointHarness {
    let tv = TestValkey::setup_with_url(cluster_url, 8).await;

    // Sanity-check the URL actually pointed at a clustered deployment.
    // Mirrors the assertion in
    // `sweep_cluster_cleans_orphans_on_non_owner_shards` — a non-cluster
    // URL slipping through here would silently degrade this test to a
    // single-node duplicate of the sibling `e2e_stat_layer_recovery_test`.
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the cluster-aware code paths exercised \
         by this test only run when `is_clustered()` is true"
    );

    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL_LOCAL));

    // Build an ENABLED recovery hook against the real LocalDisk store —
    // same shape as `setup_checkpoint_harness`. The recovery hook plumbs
    // the same `LayerStore` / `Config` / `Metrics` the service uses, so
    // the handler-level recovery branch shares one observation surface.
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL_LOCAL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::LocalDisk;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = 1;
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_local_disk_path = Some(tempdir.path().to_string_lossy().into_owned());

    // Hoisted above the hook build so the SAME `Arc`s reach the hook and the
    // service below — `main.rs` clones one object into each, and a hook/service
    // split would leave writes landing in one ext store while reads come out of
    // the other. The bootstrap is the disabled stub this file already used: this
    // harness runs with the tail flag off and asserts nothing about a re-base.
    let ext_value_store: Arc<dyn ExternalValueStore> = Arc::new(MockExternalValueStore::new());
    let base_bootstrap = crate::common::build_disabled_base_bootstrap(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&metrics),
    );

    let recovery = build_recovery_hook_for_e2e(
        Arc::clone(&layer_store),
        Arc::clone(&store),
        cfg.clone(),
        Arc::clone(&metrics),
        E2eRecoveryHookWiring {
            ext_value_store: Arc::clone(&ext_value_store),
            base_bootstrap: Some(Arc::clone(&base_bootstrap)),
        },
    );
    let session_recovery =
        build_session_recovery_hook_for_e2e(&tv, Arc::clone(&store), cfg, Arc::clone(&metrics));

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store: Arc::clone(&layer_store),
        ext_value_store,
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
        metrics: Arc::clone(&metrics),
        service_label: SERVICE_LABEL_LOCAL.to_string(),
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

    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_secs(1),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; set well above the test runtime
        // so the counters-sidecar timer does not fire during this test
        // (Epic 6 sidecar paths are exercised by their own sibling tests).
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(SERVICE_LABEL_LOCAL),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    ClusterCheckpointHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        _tempdir: tempdir,
    }
}

// =============================================================================
// Cluster-mode E2E recovery — Epic AC #7 cluster gate.
// =============================================================================

/// Pins the Epic AC #7 cluster gate: a write → checkpoint → wipe → recover
/// cycle against a 3-shard Valkey Cluster must succeed end-to-end and
/// restore the layer state under the same `object_id`.
///
/// Why each step is load-bearing on cluster:
///
/// - `TestValkey::setup_with_url(cluster_url, 8)` connects to the cluster
///   URL the operator passed via `LIVE_EDIT_TEST_CLUSTER_URL`. The
///   `is_clustered()` sanity check rejects a misconfigured single-node URL
///   silently degrading the test surface.
/// - `create_session_and_layer` writes BOTH the session metadata
///   (`{session_id}` hash tag) and the layer mapping
///   (`{session_id}:layer:...`) plus the per-object materialized state
///   (`{object_id}` hash tag). On a 3-shard cluster the session keys and
///   the object keys hash to potentially different shards, so the
///   round-trip exercises cross-shard mapping resolution + per-object
///   state restore.
/// - `wait_for_checkpoint` blocks until the leader-elected loop lands at
///   least one generation. The loop's per-object SCAN goes through
///   `scan_cluster_buffered` when `is_clustered()` is true — a regression
///   that lost the cluster branch would surface here as an empty
///   `list_layer_checkpoints` and the deadline panic.
/// - `wipe_layer_state` uses `scan_cluster_buffered` (the helper's
///   `is_clustered()` branch) to walk every shard for matching keys. With
///   the `{object_id}` hash tag all matches land on one shard, but the
///   cluster-aware SCAN call is exactly what the production assembler and
///   orphan-sweeper use.
/// - `handle_stat_layer` triggers the recovery branch from the
///   post-mapping `stat_layer_for_object_id` NotFound (`recovery.rs` step
///   5 / SDD SR-2.4). The recovery hook then reads the checkpoint blob
///   from disk and writes the restored state back through `LayerStore`,
///   which on cluster goes through cluster-aware writes — a regression
///   that lost the cluster branch on the apply path would surface as a
///   `CROSSSLOT` error.
/// - `read_object_state_for_test` post-recovery confirms the materialized state
///   round-tripped through cluster correctly. The per-object hash tag
///   keeps every read on one shard, so the snapshot read is slot-safe.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_recovery_round_trip_against_three_shard_cluster() {
    // Skip-on-missing per the orphan-sweep cluster-test pattern at
    // `tests/integration/storage/orphan_sweep_test.rs:952-957`. The repo
    // does not ship a cluster harness, so this test is opt-in.
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let harness = setup_cluster_checkpoint_harness(cluster_url).await;

    // CreateSession + CreateLayer through the standard helper. Mints a
    // fresh `object_id` whose `{object_id}` hash tag will determine
    // which shard owns the materialized state. On a 3-shard cluster the
    // session and object keys land on potentially different shards.
    let (session_id, object_id) =
        create_session_and_layer(&harness.tv, &harness.svc, SESSION_NAME, LAYER_NAME).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id on cluster; got {object_id}"
    );

    // Wait until the leader-elected loop has landed at least one
    // generation. `tick_secs=1` + `interval=quiesce=0` means the loop
    // produces a generation within one tick window; 10 s deadline
    // absorbs CI jitter and any extra cluster-side latency.
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Pre-call snapshot of the recovery-attempts counter. Fresh harness
    // Metrics start at zero; snapshot for robustness against any future
    // pre-seed behaviour.
    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    // Wipe `le:{object_id}:obj:*` across all shards. The helper's
    // `is_clustered()` branch walks every shard via
    // `scan_cluster_buffered`; the `{object_id}` hash tag guarantees the
    // matches actually land on one shard but the cluster-aware SCAN call
    // is exactly what the production assembler and orphan-sweeper use.
    wipe_layer_state(&harness.tv, object_id).await;

    // Sanity-check the wipe actually cleared the header. A failed wipe
    // would let the next StatLayer succeed without exercising the
    // recovery branch.
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let post_wipe_header: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed on cluster");
    assert!(
        post_wipe_header.is_empty(),
        "wipe_layer_state must leave obj:hdr empty on cluster; got {} fields: {:?}",
        post_wipe_header.len(),
        post_wipe_header.keys().collect::<Vec<_>>()
    );

    // Trigger recovery via StatLayer — the cheapest handler-level
    // recovery trigger (no streaming, no OMPF encoding). The recovery
    // branch in `handle_stat_layer` fires on the post-mapping NotFound,
    // applies the checkpoint, retries the header read, and returns Ok
    // with the recovered metadata.
    let response = handle_stat_layer(&harness.svc, stat_request(&session_id, LAYER_NAME))
        .await
        .expect(
            "enabled-mode StatLayer after cluster wipe MUST recover from checkpoint \
             and return Ok; a NotFound here means the cluster recovery path failed",
        );
    let stat = response.into_inner();
    assert!(
        !stat.created_by_user_name.is_empty(),
        "StatLayer after cluster recovery MUST surface a non-empty \
         created_by_user_name (apply_checkpoint restores the header field from \
         the checkpoint meta); empty here means the apply pipeline ran but \
         failed to land the header field across shards"
    );
    assert!(
        stat.created_at > 0,
        "StatLayer after cluster recovery MUST surface a non-zero created_at; \
         zero here means the apply pipeline lost the timestamp during restore"
    );

    // Exactly ONE recovery attempt landed in the `restored` cell. Pins
    // the at-most-once contract for cluster mode.
    assert_eq!(
        recovery_metric(&harness.metrics, "restored") - restored_pre,
        1,
        "Epic AC #7 cluster gate: exactly ONE recovery attempt must have \
         reported `restored`; zero = recovery never fired, two+ = the \
         `recovered_once` latch is broken on cluster"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` cell MUST stay at the pre-call snapshot — the \
         checkpoint landed pre-wipe so recovery MUST succeed on cluster"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error") - error_pre,
        0,
        "`error` cell MUST stay at the pre-call snapshot — LocalDisk \
         against a healthy tempdir has no transient failures, and the \
         cluster connection is healthy by construction"
    );

    // Post-recovery materialized state. `read_object_state_for_test` is
    // hash-tagged per-`object_id` so the read is slot-safe on cluster.
    // Its success implies the apply pipeline restored the header AND
    // the section index across shards.
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect(
        "post-recovery read_object_state_for_test must succeed on cluster (apply \
         pipeline must have restored the header across shards)",
    );
    assert_eq!(
        snapshot.object_id, object_id,
        "post-recovery snapshot.object_id MUST equal pre-wipe captured \
         object_id ({object_id}) on cluster; got {}. A mismatch would \
         indicate apply re-minted the object instead of restoring it under \
         the same `{{object_id}}` hash-tag slot.",
        snapshot.object_id
    );

    // Graceful shutdown of the checkpoint loop. Drops are sufficient,
    // but cancel + await mirrors the production drain shape and removes
    // the slim window where a still-running tick could race with the
    // test runtime's shutdown.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}
