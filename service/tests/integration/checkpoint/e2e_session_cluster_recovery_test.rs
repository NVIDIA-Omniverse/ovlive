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

//! End-to-end cluster-mode session-recovery round-trip test
//! (Subtask `t3.fgj.7v.o7.v9` under Task `t2.fgj.7v.o7`, Epic `t1.fgj.7v`,
//! Bee `b.fgj`). Gated on `LIVE_EDIT_TEST_CLUSTER_URL`.
//!
//! Pins the SR-2.7 cross-shard contract: the two-step session-recovery
//! protocol (Step A — `path_names_index` + path-index, slot-bound by
//! `{hex(base_path)}`; Step B — session-meta + layer-mappings, slot-bound
//! by `{session_id}`) lands on different Cluster slots whenever the
//! `base_path` and the `session_id` happen to hash differently. Single-
//! node sibling tests (`e2e_session_full_wipe_recovery_test`,
//! `e2e_session_list_sessions_recovery_test`) cannot exercise that cross-
//! shard surface — the slot routing decision evaporates on a single-shard
//! deployment. A regression that lost cluster-aware routing (for
//! instance, accidentally pinning a Lua script or a MULTI/EXEC across a
//! slot boundary) would surface here as `CROSSSLOT`; on the single-node
//! tests it would still pass.
//!
//! Mirrors Epic 1's `e2e_cluster_recovery_test.rs` cluster gate but
//! exercises the session-scope recovery path Task 7 adds, NOT the
//! layer-scope path Epic 1 pinned. The two cluster gates compose:
//! together they cover every recovery code path against a real ≥3-shard
//! cluster.
//!
//! ## Gating
//!
//! Skip-on-missing per the orphan-sweep cluster regression pattern at
//! `tests/integration/storage/orphan_sweep_test.rs:950-957`. The repo
//! does not ship a cluster harness; `bash scripts/dev-valkey.sh
//! cluster-up` brings up a 3-shard cluster on `127.0.0.1:7000..7002`
//! and prints the suggested `LIVE_EDIT_TEST_CLUSTER_URL`. CI's
//! `test-cluster:` job runs the same path. Without the env var set the
//! test compiles clean and skips at runtime — the standard contract for
//! cluster-only regressions in this repo.
//!
//! Sandbox env note: this test is opt-in via `LIVE_EDIT_TEST_CLUSTER_URL`
//! and reads its value verbatim; the `TEST_VALKEY_URL` sandbox workaround
//! is NOT relevant here (this test never falls back to `TestValkey::setup`).
//!
//! ## Test scope
//!
//! Two cases:
//!
//! 1. `cluster_session_recovery_round_trip_against_three_shard_cluster`
//!    — single-session round-trip exercising the two-MULTI/EXEC apply
//!    crossing both shards. Asserts post-recovery `restored{session}`
//!    AND `restored{layer}` both ticked by 1, `handle_list_sessions`
//!    surfaces the recovered session (proves Step A path-index landed
//!    on the `{hex(base_path)}` shard), and `read_object_state_for_test`
//!    returns restored state.
//!
//! 2. `cluster_session_recovery_multi_session_distinct_base_paths_all_recover`
//!    — three sessions on three DISTINCT `base_path`s, each on its own
//!    layer. The path-index keys hash to potentially different shards
//!    by birthday-paradox math; the per-session recovery must land
//!    each Step A apply on the path-index's owning shard while landing
//!    Step B on the session-id's owning shard. Three concurrent
//!    UpdateLayers post-wipe — all three must recover.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::{ClientLike, HashesInterface};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, SessionCheckpointWriter,
    TailCaptureConfig,
};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{handle_create_session, handle_list_sessions};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::helpers::{AllowAllPermissions, StubUserinfo};

const SERVICE_LABEL_LOCAL: &str = SERVICE_LABEL;
const CLIENT_ID: &str = "e2e-cluster-session-recovery-client";

// Single-session test constants.
const SESSION_NAME: &str = "e2e-cluster-session-recovery-session";
const LAYER_NAME: &str = "e2e-cluster-session-recovery-layer";
const SESSION_BASE_PATH: &str = "/stage/cluster-session-recovery";

// Multi-session test constants. Three distinct base_paths so the per-
// base_path `{hex(base_path)}` hash tag potentially routes to different
// shards on a ≥3-shard cluster (birthday-paradox math), exercising the
// Step A cross-shard variance the parent Task body calls out.
const MULTI_SESSION_NAME_A: &str = "e2e-cluster-multi-alpha";
const MULTI_SESSION_NAME_B: &str = "e2e-cluster-multi-beta";
const MULTI_SESSION_NAME_C: &str = "e2e-cluster-multi-gamma";
const MULTI_LAYER_NAME: &str = "e2e-cluster-multi-layer";
const MULTI_BASE_PATH_A: &str = "/stage/cluster-multi-alpha";
const MULTI_BASE_PATH_B: &str = "/stage/cluster-multi-beta";
const MULTI_BASE_PATH_C: &str = "/stage/cluster-multi-gamma";

const SECTION_ID: u64 = 81;

// ── Cluster harness ─────────────────────────────────────────────────────────
//
// The session-recovery code path is wired through TWO places:
//
//   1. `SessionBasePathCache::get_base_path` — funnels the session-meta
//      NotFound on every per-session_id handler (CreateLayer / UpdateLayer
//      / DeleteLayer / StatLayer / DeleteSession / JoinSession /
//      SubscribeReadLayer) into `SessionRecoveryHook::try_recover_session`.
//   2. `reconcile_ghost_members` inside `handle_list_sessions` — for the
//      ListSessions RPC that bypasses the cache.
//
// Both wiring sites need (a) an enabled `SessionRecoveryHook` on
// `LiveEditService.session_recovery` and (b) the cache to carry the hook
// via `with_recovery_hook`. The Epic 1 cluster harness
// (`e2e_cluster_recovery_test.rs::setup_cluster_checkpoint_harness`)
// covers layer-recovery only — it does NOT wire the session-checkpoint
// writer onto `SessionStore`/`LayerStore`, and it does NOT install the
// recovery hook into the cache. Both are required for this Task 7
// session-scope cluster gate.
//
// Re-implement the harness here rather than parametrising the Epic 1
// version — the two cluster gates are sibling tests, not a generic
// shared utility, and parametrising would couple them in a way that
// hurts future independent evolution.

/// Cluster-mode harness bundle. Exposes the same surface as the single-
/// node [`CheckpointHarness`] in `helpers.rs` so per-test bodies can
/// reuse the metric helpers (`session_recovery_metric`,
/// `wait_for_recovery_metric`) and the wipe helper (`full_valkey_wipe`)
/// without modification.
struct ClusterSessionCheckpointHarness {
    tv: TestValkey,
    store: Arc<dyn CheckpointStore>,
    svc: LiveEditService,
    metrics: SharedMetrics,
    loop_handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
    _tempdir: TempDir,
}

/// Construct a cluster-mode E2E session-recovery harness rooted at the
/// supplied `LIVE_EDIT_TEST_CLUSTER_URL`. Mirrors `setup_checkpoint_harness`
/// in `helpers.rs` except for:
///
/// - `TestValkey::setup_with_url(cluster_url, 8)` instead of `setup()` —
///   the single-node constructor reads `TEST_VALKEY_URL` and cannot be
///   repointed at a cluster URL.
/// - `is_clustered()` sanity-check — rejects a misconfigured single-node
///   URL silently degrading this test to a duplicate of the sibling
///   single-node session-recovery suite.
/// - Pool size `8` matches the orphan-sweep cluster regression test
///   (`orphan_sweep_test.rs:957` uses `4`) with headroom for the
///   concurrent multi-session burst.
///
/// Everything else mirrors `setup_checkpoint_harness` 1:1: the session-
/// checkpoint writer is installed on both stores, the
/// `SessionBasePathCache` is wired with `with_recovery_hook`, the
/// enabled `RecoveryHook` and `SessionRecoveryHook` are constructed
/// against the same `LocalDiskCheckpointStore` the loop is using.
async fn setup_cluster_session_harness(cluster_url: String) -> ClusterSessionCheckpointHarness {
    let tv = TestValkey::setup_with_url(cluster_url, 8).await;

    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the cross-shard session-recovery code \
         paths exercised by this test only run when `is_clustered()` is true. \
         A misconfigured single-node URL would silently degrade this test to \
         a duplicate of the sibling single-node `e2e_session_full_wipe_recovery_test`."
    );

    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL_LOCAL));

    // Session-checkpoint writer wired on BOTH stores — load-bearing for
    // this test. Without it, CreateSession / CreateLayer lifecycle hooks
    // would not enqueue session-checkpoint writes, the dispatcher would
    // never land a blob on disk, and the post-wipe `try_recover_session`
    // would surface `NoCheckpoint` instead of `Restored` — the test
    // would fail at the metric assertion (`restored{session} == 0`)
    // rather than the cross-shard wiring it is targeting.
    let session_checkpoint_tracker = TaskTracker::new();
    let session_checkpoint_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        session_checkpoint_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL_LOCAL.to_string(),
        true,
    ));

    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer)),
    );
    let session_store = Arc::new(
        SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer)),
    );

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

    // Cache + recovery-hook wiring. Without `with_recovery_hook`, the
    // cache funnel's `get_base_path` surfaces the raw NotFound from
    // `session_store.get_base_path` and recovery never fires from the
    // per-handler choke point — every cache-funnel handler test would
    // observe `NotFound` instead of `restored=1`.
    let session_base_path_cache = live_edit::grpc::session_cache::SessionBasePathCache::new()
        .with_recovery_hook(Arc::clone(&session_recovery));

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
        session_base_path_cache,
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
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
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

    ClusterSessionCheckpointHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        _tempdir: tempdir,
    }
}

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

// ── Delta fixture builders ───────────────────────────────────────────────────

/// A populated initial delta: one `DiffChangeSection` + one `SetField`.
/// The non-empty initial state is what the post-recovery
/// `read_object_state_for_test` assertion will verify was restored — an empty
/// delta would make the "snapshot has sections" check vacuous (the
/// snapshot would be empty either way).
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "e2e-cluster-session-section".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAB, 0xCD, 0xEF],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

/// Post-wipe UpdateLayer carries a SetField on a NEW key. A regression
/// that lost the new write (e.g. a recovery loop that re-ran apply
/// without re-issuing the commit) would show original fields restored
/// but the marker field missing. `base_version: 1` keeps this as an
/// INCREMENTAL diff (NOT a reset — see
/// `e2e_session_full_wipe_recovery_test.rs` for the same rationale).
fn delta_post_recovery(set_order: u64, key_suffix: &str) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: format!("post-recovery-{key_suffix}"),
            value_or_ext_hash: vec![0xCA, 0xFE, 0xBA, 0xBE],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

// ── Test driver helpers ──────────────────────────────────────────────────────

/// Issue `CreateSession` and wait for the session-checkpoint writer's
/// fire-and-forget put to land on disk. Returns the minted `session_id`.
async fn create_session_and_wait(
    harness: &ClusterSessionCheckpointHarness,
    base_path: &str,
    name: &str,
) -> String {
    let response = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: base_path.to_owned(),
            name: name.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against AllowAllPermissions cluster harness");
    let session_id = response.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;
    session_id
}

/// Issue `CreateLayer` with a populated initial delta. Returns the minted
/// `object_id`.
async fn create_layer_with_initial_state(
    harness: &ClusterSessionCheckpointHarness,
    session_id: &str,
    layer_name: &str,
) -> u64 {
    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(ompf),
    ]);
    let response = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for cluster session-recovery harness setup");
    let object_id = response.into_inner().object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id on cluster; got {object_id}"
    );
    object_id
}

/// Issue `UpdateLayer` and return the produced `sequence_num`. Used both
/// in the pre-wipe burst and as the post-wipe recovery-trigger call.
async fn issue_update(
    harness: &ClusterSessionCheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: OwnedDelta,
    client_timestamp: u64,
) -> Result<u64, tonic::Status> {
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf),
    ]);
    handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .map(|r| r.into_inner().sequence_num)
}

/// Read the live label-cell counter value for the supplied `result` on
/// the `live_edit_checkpoint_recovery_attempts_total{result, scope, service}`
/// metric — variant of `session_recovery_metric` that takes a
/// `SharedMetrics` directly (the helper in `helpers.rs` takes
/// `&CheckpointHarness` which is the single-node bundle type).
fn recovery_metric(metrics: &SharedMetrics, scope: &str, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", scope),
            ("service", SERVICE_LABEL_LOCAL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `full_valkey_wipe` is the public helper in `helpers.rs`, but it takes
/// `&TestValkey` and is cluster-aware (groups keys by slot before
/// UNLINK). Local shim that consumes the cluster harness's `tv` field.
async fn full_wipe(harness: &ClusterSessionCheckpointHarness) {
    full_valkey_wipe(&harness.tv).await;
}

// =============================================================================
// Test 1 — Single-session cross-shard recovery round-trip.
// =============================================================================

/// Pin the cluster-gate AC for Task 7: a single-session
/// CreateSession + CreateLayer + UpdateLayer burst → checkpoint → full
/// Valkey wipe → next UpdateLayer must land BOTH recovery hops across
/// shards and succeed.
///
/// Why each step is load-bearing on cluster:
///
/// - `setup_cluster_session_harness` connects to the cluster URL the
///   operator passed via `LIVE_EDIT_TEST_CLUSTER_URL`. The
///   `is_clustered()` sanity-check rejects a misconfigured single-node
///   URL.
/// - `handle_create_session` writes path-index keys (`le:path:{hex(base_path)}*`
///   — hash-tagged by base_path) AND `le:{session_id}:sess` (hash-tagged
///   by session_id). On a 3-shard cluster the two key families hash to
///   potentially different shards. The session-checkpoint writer
///   captures BOTH families into one blob, which the per-session
///   recovery applier later re-applies via two separate MULTI/EXEC
///   transactions (one per slot).
/// - `wait_for_session_checkpoint` blocks on the fire-and-forget writer's
///   `put_session_checkpoint` landing on disk. Without it, the post-wipe
///   `try_recover_session` would race the writer's put-task and surface
///   `NoCheckpoint` instead of `Restored`.
/// - `handle_create_layer` mints an `object_id` (`{object_id}` hash tag,
///   different shard from both `{session_id}` and `{hex(base_path)}` in
///   the general case). The layer-mapping key is hash-tagged by
///   `{session_id}` — same shard as session-meta — so the apply pipeline
///   for the session checkpoint covers it.
/// - `wait_for_checkpoint` blocks on the layer-checkpoint loop landing
///   the first generation. On cluster the loop's per-object SCAN goes
///   through `scan_cluster_buffered` when `is_clustered()` is true; a
///   regression that lost the cluster branch would surface here as an
///   empty `list_layer_checkpoints` and the deadline panic.
/// - `full_valkey_wipe` SCANs + UNLINKs EVERY `tv.prefix()*` key across
///   every shard. The cluster-aware UNLINK groups by slot before
///   issuing the multi-key command (helpers.rs full_valkey_wipe handles
///   this) — a regression that issued a single multi-slot UNLINK would
///   surface as `CROSSSLOT`.
/// - `invalidate_session_cache` drops the in-process cache entry — see
///   the helper docstring (in the single-node siblings) for why. The
///   pre-wipe CreateLayer call populated the cache with the live
///   `base_path`; without invalidation the post-wipe UpdateLayer would
///   HIT the cache and bypass session-recovery entirely.
/// - `handle_update_layer` MUST succeed. The two recovery hops:
///   a. Cache funnel: `get_base_path` sees NotFound on `le:{sid}:sess`,
///      fires session-recovery. The applier runs Step A on the
///      `{hex(base_path)}` shard and Step B on the `{session_id}` shard
///      — two separate MULTI/EXEC transactions. A regression that
///      lumped them into one would surface as `CROSSSLOT`.
///   b. Layer-recovery: `update_layer_once` HGETALLs the empty
///      `obj:hdr`, surfaces NotFound, the layer-recovery branch fires,
///      `apply_checkpoint` rebuilds object state on the `{object_id}`
///      shard. The commit is re-issued and returns Ok.
/// - Both `restored{session}` AND `restored{layer}` MUST tick by exactly
///   one — pins the cross-shard composition contract.
/// - `handle_list_sessions(base_path)` returns the recovered session —
///   proves the Step A apply landed on the `{hex(base_path)}` shard.
/// - `read_object_state_for_test(object_id)` returns the restored state —
///   proves the layer-recovery apply landed on the `{object_id}` shard.
///
/// Skip-on-missing: this test compiles cleanly without
/// `LIVE_EDIT_TEST_CLUSTER_URL` and skips at runtime per the
/// orphan-sweep cluster-test pattern. CI's `test-cluster:` job sets the
/// env var and runs the real assertions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_session_recovery_round_trip_against_three_shard_cluster() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let harness = setup_cluster_session_harness(cluster_url).await;

    // ── CreateSession + wait for session-checkpoint blob ────────────────
    let session_id = create_session_and_wait(&harness, SESSION_BASE_PATH, SESSION_NAME).await;

    // Capture the pre-CreateLayer session-checkpoint blob so the
    // subsequent CreateLayer-driven write can be observed to mutate
    // the bytes — this guards against the wipe-and-recover step racing
    // the CreateLayer writer's put.
    let pre_create_layer_session_bytes = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint must succeed once a blob has landed")
        .expect("session-checkpoint blob must be present after wait_for_session_checkpoint")
        .bytes;

    // ── CreateLayer + wait for post-CreateLayer session-checkpoint ──────
    let object_id = create_layer_with_initial_state(&harness, &session_id, LAYER_NAME).await;

    wait_for_post_session_checkpoint_bytes_change(
        &harness,
        &session_id,
        pre_create_layer_session_bytes.as_ref(),
        10,
    )
    .await;

    // ── UpdateLayer burst (drive seq above 1 + accumulate state) ────────
    let mut pre_wipe_max_seq: u64 = 0;
    for (i, ts) in (1_001..=1_003).enumerate() {
        let seq = issue_update(
            &harness,
            &session_id,
            LAYER_NAME,
            object_id,
            delta_post_recovery(i as u64 + 1, &format!("burst-{i}")),
            ts,
        )
        .await
        .expect("pre-wipe UpdateLayer must succeed against a healthy cluster");
        pre_wipe_max_seq = seq;
    }
    assert!(
        pre_wipe_max_seq >= 4,
        "pre-wipe max sequence_num must be at least 4 (1 from CreateLayer + 3 \
         UpdateLayers); got {pre_wipe_max_seq}"
    );

    // ── Wait for a layer-checkpoint generation reflecting the burst ─────
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // ── Pre-call metric baselines ───────────────────────────────────────
    //
    // The fresh harness Metrics start at zero, but the snapshot pattern
    // is robust to any future pre-seed change.
    let session_restored_pre = recovery_metric(&harness.metrics, "session", "restored");
    let session_not_recoverable_pre =
        recovery_metric(&harness.metrics, "session", "not_recoverable");
    let session_error_pre = recovery_metric(&harness.metrics, "session", "error");
    let layer_restored_pre = recovery_metric(&harness.metrics, "layer", "restored");
    let layer_not_recoverable_pre = recovery_metric(&harness.metrics, "layer", "not_recoverable");
    let layer_error_pre = recovery_metric(&harness.metrics, "layer", "error");

    // ── Full Valkey wipe — cluster-aware SCAN+UNLINK across every shard ─
    full_wipe(&harness).await;

    // Sanity: session-meta key absent post-wipe.
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let session_meta_after: HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&session_meta_key)
        .await
        .expect("HGETALL session_meta must succeed on cluster");
    assert!(
        session_meta_after.is_empty(),
        "full_valkey_wipe must leave le:{{sid}}:sess empty on cluster; got {} fields: {:?}. \
         A surviving key here means the cluster-aware SCAN missed it and session-recovery \
         would silently no-op via the pre-acquire AlreadyPresent short-circuit.",
        session_meta_after.len(),
        session_meta_after.keys().collect::<Vec<_>>()
    );

    // Sanity: obj:hdr key absent post-wipe (different shard from
    // session-meta in the general case).
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let obj_header_after: HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed on cluster");
    assert!(
        obj_header_after.is_empty(),
        "full_valkey_wipe must leave obj:hdr empty on cluster; got {} fields: {:?}. \
         A surviving header means layer-recovery never fires.",
        obj_header_after.len(),
        obj_header_after.keys().collect::<Vec<_>>()
    );

    // ── Invalidate the cache so the post-wipe call consults session-recovery ─
    harness.svc.session_base_path_cache.invalidate(&session_id);

    // ── Dispatch the post-wipe UpdateLayer. MUST succeed. ───────────────
    //
    // Two recovery hops in series across shards:
    //   1. Session-recovery: Step A on `{hex(base_path)}` shard, Step B
    //      on `{session_id}` shard. A regression that crossed slots in
    //      a single MULTI/EXEC would surface as CROSSSLOT.
    //   2. Layer-recovery: apply on `{object_id}` shard, re-issued
    //      commit lands.
    let post_recovery_seq = issue_update(
        &harness,
        &session_id,
        LAYER_NAME,
        object_id,
        delta_post_recovery(99, "post-recovery"),
        9_999,
    )
    .await
    .expect(
        "post-wipe UpdateLayer MUST succeed on cluster — a CROSSSLOT error here \
         indicates the session-recovery applier issued a multi-key write across \
         shards (regression that lost the cluster-aware two-MULTI/EXEC routing). \
         A NotFound here indicates one of the two recovery hops did not fire.",
    );
    assert!(
        post_recovery_seq > pre_wipe_max_seq,
        "post-recovery sequence_num ({post_recovery_seq}) MUST be strictly greater \
         than pre-wipe max ({pre_wipe_max_seq}); a stale or equal seq indicates the \
         re-issued commit landed against pre-restore state"
    );

    // ── Metric assertions: both scopes ticked by exactly 1 ──────────────
    //
    // Use `wait_for_recovery_metric` for the bounded-poll surface — the
    // layer-recovery metric increment is awaited inside the handler, so
    // by RPC return both increments are visible; the poll covers any
    // future fire-and-forget refactor that would otherwise make the
    // test flaky.
    wait_for_recovery_metric_scope(
        &harness.metrics,
        "session",
        "restored",
        session_restored_pre + 1,
        10,
    )
    .await;
    wait_for_recovery_metric_scope(
        &harness.metrics,
        "layer",
        "restored",
        layer_restored_pre + 1,
        10,
    )
    .await;

    // The other cells stay at baseline — pins exactly-one-restored across
    // both scopes.
    assert_eq!(
        recovery_metric(&harness.metrics, "session", "not_recoverable"),
        session_not_recoverable_pre,
        "post-recovery `not_recoverable{{session}}` MUST equal baseline — the \
         session-checkpoint blob was readable so the apply branch ran"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "session", "error"),
        session_error_pre,
        "post-recovery `error{{session}}` MUST equal baseline — no transient \
         backend failure was injected; cluster connection is healthy"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "layer", "not_recoverable"),
        layer_not_recoverable_pre,
        "post-recovery `not_recoverable{{layer}}` MUST equal baseline — the \
         layer-checkpoint generation was readable"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "layer", "error"),
        layer_error_pre,
        "post-recovery `error{{layer}}` MUST equal baseline — no transient \
         backend failure was injected"
    );

    // ── ListSessions(base_path) returns the recovered session ───────────
    //
    // Proves the Step A path-index restore landed on the
    // `{hex(base_path)}` shard. ListSessions reads `path_names_index`;
    // without the restore it would return an empty list. A CROSSSLOT
    // error inside `reconcile_ghost_members`'s recovery hop would
    // surface here as a tonic Status, not as an empty response.
    let list_resp = handle_list_sessions(
        &harness.svc,
        make_unary_token_request(ListSessionsRequest {
            path: SESSION_BASE_PATH.to_owned(),
        }),
    )
    .await
    .expect(
        "post-recovery ListSessions MUST succeed on cluster — a failure here \
         indicates the path-index was not restored on the {{hex(base_path)}} \
         shard (Step A applier regression) or the ghost-member reconciliation \
         crossed slots",
    );
    let sessions = list_resp.into_inner().sessions;
    assert_eq!(
        sessions.len(),
        1,
        "post-recovery ListSessions MUST return exactly one session under \
         base_path={SESSION_BASE_PATH} on cluster; got {} sessions",
        sessions.len()
    );
    assert_eq!(
        sessions[0].session_id, session_id,
        "post-recovery ListSessions session_id MUST match pre-wipe captured id; \
         got {}, expected {session_id}",
        sessions[0].session_id
    );
    assert_eq!(
        sessions[0].name, SESSION_NAME,
        "post-recovery ListSessions session.name MUST match original; got {}, \
         expected {SESSION_NAME}",
        sessions[0].name
    );

    // ── read_object_state_for_test returns restored state on the {object_id} shard ─
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect(
        "post-recovery read_object_state_for_test MUST succeed on cluster — a NotFound \
         here means the layer-recovery apply did not land the header on the \
         {{object_id}} shard",
    );
    assert_eq!(
        snapshot.object_id, object_id,
        "post-recovery snapshot.object_id MUST equal pre-wipe captured object_id \
         ({object_id}) on cluster; got {}. A mismatch would indicate apply \
         re-minted the object instead of restoring it under the same hash-tag slot.",
        snapshot.object_id
    );
    assert!(
        snapshot.sections.contains_key(&SECTION_ID),
        "post-recovery snapshot MUST contain the original section_id={SECTION_ID} \
         restored from the layer-checkpoint generation; got sections={:?}",
        snapshot.sections.keys().collect::<Vec<_>>()
    );

    // ── Graceful shutdown ───────────────────────────────────────────────
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// =============================================================================
// Test 2 — Multi-session distinct-base_paths cross-shard recovery.
// =============================================================================

/// Three sessions on three DISTINCT `base_path`s → full Valkey wipe →
/// three concurrent UpdateLayers (one per session). On a 3-shard cluster
/// the three `{hex(base_path)}` hash tags route to potentially
/// different shards by birthday-paradox math, so the Step A apply for
/// each session lands on its own shard, while Step B lands on the
/// session's `{session_id}` shard (a fourth distinct slot in the general
/// case). All three UpdateLayers MUST succeed.
///
/// Why this is the multi-session sibling to Test 1:
///
/// - Test 1 exercises ONE cross-shard hop pair (Step A on path-index
///   shard, Step B on session-meta shard). Test 2 exercises THREE
///   independent pairs, with the three path-index shards potentially
///   covering distinct slots. A regression that worked for the
///   "session_id and base_path happen to hash to the same shard" case
///   (which any single-session test can hit by accident) would still
///   pass Test 1's single-session round-trip but fail Test 2's
///   three-distinct-base_path case.
/// - Concurrent dispatch via `tokio::spawn` exercises the
///   per-`session_id` admission discipline against contention. The
///   recovery hook's K=1 admission map is keyed per-session-id, and
///   the three sessions have three distinct ids; the per-session
///   admission mutexes MUST NOT cross-block, and every UpdateLayer
///   must recover independently.
/// - Post-recovery: every layer's `read_object_state_for_test` returns
///   restored state AND every `restored{session}` cell increment
///   matches the per-session count (3 sessions → 3 increments).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_session_recovery_multi_session_distinct_base_paths_all_recover() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let harness = setup_cluster_session_harness(cluster_url).await;

    // ── CreateSession + CreateLayer for three distinct base_paths ───────
    //
    // Each session gets its own layer (same layer_name within each
    // session is fine — the mapping is hash-tagged by `{session_id}`).
    let scenarios = [
        (MULTI_BASE_PATH_A, MULTI_SESSION_NAME_A),
        (MULTI_BASE_PATH_B, MULTI_SESSION_NAME_B),
        (MULTI_BASE_PATH_C, MULTI_SESSION_NAME_C),
    ];

    let mut sessions: Vec<(String, String, String, u64)> = Vec::with_capacity(scenarios.len());
    for (base_path, name) in scenarios {
        let session_id = create_session_and_wait(&harness, base_path, name).await;

        // Capture pre-CreateLayer blob bytes to verify the CreateLayer-
        // driven session-checkpoint write actually landed before the
        // wipe — guards against the wipe racing the CreateLayer writer's
        // put.
        let pre_bytes = harness
            .store
            .get_session_checkpoint(&session_id)
            .await
            .expect("get_session_checkpoint must succeed")
            .expect("session-checkpoint blob must be present")
            .bytes;

        let object_id =
            create_layer_with_initial_state(&harness, &session_id, MULTI_LAYER_NAME).await;

        // Drive one UpdateLayer per session so each has a layer-
        // checkpoint generation reflecting committed state. Single
        // update is sufficient — Test 1 already covers the multi-update
        // burst surface; Test 2's load-bearing axis is multi-session
        // cross-shard, not per-session burst.
        let _ = issue_update(
            &harness,
            &session_id,
            MULTI_LAYER_NAME,
            object_id,
            delta_post_recovery(2, &format!("multi-pre-{name}")),
            2_000,
        )
        .await
        .expect("pre-wipe UpdateLayer must succeed against healthy cluster");

        // Wait for the post-CreateLayer session-checkpoint write to land
        // (bytes-differ against the pre-CreateLayer blob).
        wait_for_post_session_checkpoint_bytes_change(
            &harness,
            &session_id,
            pre_bytes.as_ref(),
            10,
        )
        .await;

        // Wait for at least one layer-checkpoint generation per session.
        let _ = wait_for_checkpoint(&harness.store, object_id, 10).await;

        sessions.push((session_id, base_path.to_owned(), name.to_owned(), object_id));
    }

    // ── Pre-call metric baselines ───────────────────────────────────────
    let session_restored_pre = recovery_metric(&harness.metrics, "session", "restored");
    let layer_restored_pre = recovery_metric(&harness.metrics, "layer", "restored");

    // ── Full wipe + cache invalidation for every session ────────────────
    full_wipe(&harness).await;
    for (session_id, _, _, _) in &sessions {
        harness.svc.session_base_path_cache.invalidate(session_id);
    }

    // ── Three concurrent UpdateLayers, one per session ──────────────────
    //
    // tokio::spawn (NOT join!) so each runs on its own task and the
    // per-session admission mutexes can demonstrate non-blocking
    // independence. JoinSet would also work; spawn + Vec<JoinHandle>
    // matches the sibling concurrent-recovery test's pattern.
    //
    // `LiveEditService` is `#[derive(Clone)]` (every field is `Arc<...>` or
    // similarly Clone-cheap), so each spawned task owns its own clone — no
    // cross-task `&` borrow ceremony required.
    let mut join_handles = Vec::with_capacity(sessions.len());
    for (i, (session_id, _, _, object_id)) in sessions.iter().enumerate() {
        let svc = harness.svc.clone();
        let session_id = session_id.clone();
        let object_id = *object_id;
        let delta = delta_post_recovery((100 + i) as u64, &format!("multi-post-{i}"));
        let handle = tokio::spawn(async move {
            let ompf = delta_to_ompf_bytes(&delta, &[]);
            let stream = streaming_update(vec![
                make_update_header(
                    &session_id,
                    MULTI_LAYER_NAME,
                    object_id,
                    9_900 + i as u64,
                    CLIENT_ID,
                ),
                make_update_data(ompf),
            ]);
            handle_update_layer(&svc, make_streaming_token_request(stream)).await
        });
        join_handles.push(handle);
    }

    let mut succeeded = 0_usize;
    for (i, handle) in join_handles.into_iter().enumerate() {
        let join_result = handle
            .await
            .expect("spawned UpdateLayer task must not panic");
        let response = join_result.unwrap_or_else(|status| {
            panic!(
                "post-wipe UpdateLayer #{i} MUST succeed on cluster — a CROSSSLOT \
                 error indicates the session-recovery applier crossed slots in a \
                 single MULTI/EXEC (Step A and Step B must run as separate \
                 transactions on distinct slot owners). Got status: {status:?}"
            )
        });
        let seq = response.into_inner().sequence_num;
        assert!(
            seq > 0,
            "post-recovery UpdateLayer #{i} sequence_num must be > 0; got {seq}"
        );
        succeeded += 1;
    }
    assert_eq!(
        succeeded,
        sessions.len(),
        "all {} concurrent UpdateLayers MUST succeed; got {}",
        sessions.len(),
        succeeded
    );

    // ── Metric assertions: per-session recovery increments ──────────────
    //
    // EXACTLY three session-recovery `restored` increments (one per
    // session). The K=1 admission map is keyed per-session-id, and the
    // three sessions have distinct ids — so no collapsing across
    // sessions. Bounded poll covers the slim window where the metric
    // increment lags the RPC return.
    wait_for_recovery_metric_scope(
        &harness.metrics,
        "session",
        "restored",
        session_restored_pre + sessions.len() as u64,
        10,
    )
    .await;
    wait_for_recovery_metric_scope(
        &harness.metrics,
        "layer",
        "restored",
        layer_restored_pre + sessions.len() as u64,
        10,
    )
    .await;

    // ── ListSessions per base_path returns the recovered session ────────
    //
    // Proves Step A landed on EACH base_path's `{hex(base_path)}` shard.
    // A CROSSSLOT regression on Step A would surface here on at least
    // one of the three calls.
    for (session_id, base_path, name, _) in &sessions {
        let list_resp = handle_list_sessions(
            &harness.svc,
            make_unary_token_request(ListSessionsRequest {
                path: base_path.clone(),
            }),
        )
        .await
        .unwrap_or_else(|status| {
            panic!(
                "post-recovery ListSessions on base_path={base_path} MUST succeed \
                 on cluster — failure indicates Step A restore did not land on \
                 the {{hex(base_path)}} shard. Status: {status:?}"
            )
        });
        let listed = list_resp.into_inner().sessions;
        assert_eq!(
            listed.len(),
            1,
            "ListSessions on base_path={base_path} MUST return exactly one session; \
             got {} (session_ids: {:?})",
            listed.len(),
            listed.iter().map(|s| &s.session_id).collect::<Vec<_>>()
        );
        assert_eq!(
            &listed[0].session_id, session_id,
            "ListSessions on base_path={base_path} MUST return the recovered \
             session_id; got {}, expected {session_id}",
            listed[0].session_id
        );
        assert_eq!(
            &listed[0].name, name,
            "ListSessions on base_path={base_path} MUST return the recovered name; \
             got {}, expected {name}",
            listed[0].name
        );
    }

    // ── read_object_state_for_test per layer returns restored state ───────────
    //
    // Proves layer-recovery landed on EACH object's `{object_id}` shard
    // independently. Three distinct object_ids → three distinct shard
    // routes by birthday-paradox math on a 3-shard cluster.
    for (_, _, _, object_id) in &sessions {
        let snapshot = read_object_state_for_test(
            harness.tv.client(),
            harness.tv.key_schema(),
            &object_id.to_string(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "post-recovery read_object_state_for_test for object_id={object_id} MUST \
                 succeed on cluster — failure indicates layer-recovery apply did \
                 not land the header. Error: {e}"
            )
        });
        assert_eq!(
            snapshot.object_id, *object_id,
            "post-recovery snapshot.object_id MUST equal pre-wipe object_id ({object_id}) \
             on cluster; got {}",
            snapshot.object_id
        );
        assert!(
            snapshot.sections.contains_key(&SECTION_ID),
            "post-recovery snapshot for object_id={object_id} MUST contain the \
             original section_id={SECTION_ID}; got sections={:?}",
            snapshot.sections.keys().collect::<Vec<_>>()
        );
    }

    // ── Graceful shutdown ───────────────────────────────────────────────
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ── Bounded-poll helpers ─────────────────────────────────────────────────────

/// Bounded-poll variant of `wait_for_recovery_metric` (helpers.rs) that
/// reads `SharedMetrics` directly instead of going through the
/// single-node `CheckpointHarness` bundle. Same shape and semantics —
/// see `helpers.rs` `wait_for_recovery_metric` for the rationale.
async fn wait_for_recovery_metric_scope(
    metrics: &SharedMetrics,
    scope: &str,
    result: &str,
    expected: u64,
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(50);
    loop {
        let observed = recovery_metric(metrics, scope, result);
        if observed == expected {
            return;
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_recovery_metric_scope timed out after {deadline_secs}s \
                 waiting for cell scope={scope} result={result} to equal {expected} \
                 (last observed {observed}). Common causes on cluster: the \
                 recovery branch never fired (cluster-aware wiring regression), \
                 the latch already-fired in an earlier RPC, the metric was read \
                 under the wrong scope label, or a CROSSSLOT error was caught and \
                 mapped to an `error` outcome instead of `restored`."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Poll [`CheckpointStore::get_session_checkpoint`] until the blob's
/// bytes have changed from `pre_bytes`, or panic on deadline expiry.
///
/// Used after CreateLayer to confirm the lifecycle-hook session-
/// checkpoint write landed before we wipe. Without this poll, the wipe
/// could race the CreateLayer writer's put-task and the restored
/// session would be missing the freshly-created layer mapping. Same
/// shape as `wait_for_post_create_layer_session_checkpoint` in
/// `e2e_session_full_wipe_recovery_test.rs`.
async fn wait_for_post_session_checkpoint_bytes_change(
    harness: &ClusterSessionCheckpointHarness,
    session_id: &str,
    pre_bytes: &[u8],
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(50);
    loop {
        match harness.store.get_session_checkpoint(session_id).await {
            Ok(Some(blob)) if blob.bytes.as_ref() != pre_bytes => return,
            Ok(_) => {
                // Bytes have not changed yet — keep polling.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during post-write wait \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_post_session_checkpoint_bytes_change timed out after \
                 {deadline_secs}s — the post-CreateLayer session-checkpoint write \
                 never landed on cluster (bytes did not differ from pre-CreateLayer \
                 blob). Common causes: LayerStore was constructed without \
                 `with_checkpoint_writer` (harness wiring regression), or the \
                 cluster connection back-pressured the writer's spawn."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}
