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

//! End-to-end integration test for the SDD SR-3.2 lazy fetch-on-miss
//! recovery path for external-value blobs (Epic `t1.fgj.zc` Task
//! `t2.fgj.zc.p6`, Subtask `t3.fgj.zc.p6.co`).
//!
//! ## What this file pins
//!
//! Task `t2.fgj.zc.p6` Success Criterion #1: store a layer with an
//! ext-value field, pre-seed the bytes into the checkpoint adapter,
//! wipe ONLY the `le:ext:meta:{hash}` + `le:ext:data:{hash}` keys for
//! those hashes (leaving the object's `le:{oid}:obj:*` keys intact),
//! issue `SubscribeReadLayer` → `compute_diff_at_cut` → `read_batch` →
//! `try_recover_via_checkpoint` falls back to
//! `CheckpointStore::get_ext_blob`, the bytes are refilled into Valkey
//! via `store_recovered_values` — `ExternalValueStore::store_values`'
//! per-hash write, sending no ext-mirror copy since the bytes came from
//! the backup store — and the subscriber observes the original
//! payload byte-identical.
//!
//! Without this fallback the next diff-build for any ext-value-referencing
//! field after a partial Valkey state-loss event would surface
//! `ExternalValueError::MetaNotFound` / `DataNotFound` and the gRPC
//! response would be `INTERNAL` — defeating Epic AC #4 (PRD AC #4).
//!
//! ## How the test wires up
//!
//! The two harnesses below ([`setup_ext_value_recovery_harness`] and its
//! fault-injectable `InMemoryCheckpointStore` twin) are LOCAL, hand-built
//! `LiveEditService` literals. They predate the shared
//! `helpers::setup_checkpoint_harness_with_options` growing the seams
//! they were written to work around, and they have NOT been migrated
//! onto it — so read them as a deliberately narrower harness, not as a
//! statement about how a recovery E2E should be wired today.
//!
//! What has NOT drifted: both build a real `ValkeyExternalValueStore`
//! with `.with_metrics(..)` + `.with_checkpoint_store(..)`, which is the
//! whole point of the file (the SR-3.2 recovery path on `read_batch`
//! must fire against real Valkey + a real checkpoint adapter), and the
//! shared harness now defaults to a real store too — via
//! `CheckpointHarnessOptions::ext_value_store_factory` for the
//! `.with_checkpoint_store(..)` arm.
//!
//! What HAS drifted from the now-production-shaped shared harness, and
//! is safe here only because no test in this file exercises the gap:
//!
//! - Both local `RecoveryHook`s are bare `RecoveryHook::new(..)` — no
//!   `.with_ext_value_store(..)` and no `.with_base_bootstrap(..)`. The
//!   shared builder ([`helpers::recovery_hook_for_e2e`]) makes the first
//!   a required argument and wires the second by default, matching
//!   `main.rs`. No test here drives a composed restore carrying
//!   tail-borne ext values (the SR-3.4 pre-store that would fail closed
//!   without the first builder), and none asserts on an SR-3.6 re-base.
//! - `svc.base_bootstrap` is `build_disabled_base_bootstrap(..)`, and it
//!   is a DIFFERENT unit from the (absent) hook-side one. The shared
//!   harness holds a single real `BaseBootstrap` in both positions.
//! - The `LayerStore` / `SessionStore` are bare: no
//!   `.with_checkpoint_writer(..)`, no `.with_metrics(..)`, no
//!   `.with_object_id_counter_repair(..)` (b.b2j), and the
//!   `SessionBasePathCache` carries no `.with_recovery_hook(..)`. Session
//!   checkpointing and the counter-repair path are therefore invisible
//!   here.
//! - The checkpoint loop's `TailCaptureConfig` is a hard-coded
//!   `::default()` rather than `TailCaptureConfig::from_config(&cfg)`, so
//!   the loop cannot observe a `Config`-level tail knob at all.
//!
//! Migrating these two harnesses onto the shared one is out of scope for
//! the consolidation that removed this file's duplicated metric reader;
//! the divergences are recorded here so a future reader does not mistake
//! the local shape for the current production shape.
//!
//! ## Sibling-Subtask extension points
//!
//! Siblings `t3.fgj.zc.p6.ia` (adapter-missing / Transient / Permanent
//! error classification) and `t3.fgj.zc.p6.wy` (multi-field heavy-layer
//! recovery) will EXTEND this file with additional `#[tokio::test]`
//! functions. The helpers below are sized to be reusable by them:
//!
//! - [`setup_ext_value_recovery_harness`] — LocalDisk-backed; sibling
//!   `p6.wy` reuses it for the heavy-layer test. Sibling `p6.ia`'s
//!   adapter-Transient / adapter-Permanent variants need an
//!   `InMemoryCheckpointStore` with a fault injector — those can build
//!   a parallel local helper rather than re-fitting this one.
//! - [`wipe_ext_namespace_for_hashes`] — surgical UNLINK of
//!   `le:ext:meta:{hash_hex}` + `le:ext:data:{hash_hex}` for an explicit
//!   hash list. Does NOT touch `le:ext:{gc}:*` GC bookkeeping (the
//!   Subtask body explicitly forbids a wildcard that would catch GC
//!   sets) and does NOT touch the object's `le:{oid}:obj:*` keys.
//! - Value hashes are `live_edit::storage::external_values::ext_content_hash`,
//!   the content hash the CreateLayer / UpdateLayer upload check verifies (its
//!   unit tests pin it to the clients' own vectors). A plain SHA-256 claim is
//!   refused by that check. The sibling files in this module import it
//!   directly.
//! - [`recv_bootstrap_message`] — drains a header + N chunks until
//!   `total_size` bytes are gathered. Identical pattern to
//!   `e2e_subscribe_read_layer_recovery_test.rs::recv_message`; siblings
//!   that exercise the same bootstrap-drain pattern reuse it.
//!
//! All helpers are `fn` (not `async fn`) where reasonable per the
//! Subtask body's parallel-agent guidance. The harness builder
//! [`setup_ext_value_recovery_harness`] is `async` because
//! `TestValkey::setup()` is async.
//!
//! ## Sandbox env
//!
//! `TestValkey::setup` reads `TEST_VALKEY_URL`. Under the project
//! sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use dashmap::DashMap;
use fred::interfaces::{HashesInterface, KeysInterface};
use futures::{Stream, StreamExt};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, RecoveryHook,
    SessionRecoveryHook, TailCaptureConfig,
};
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};
use tonic::{Request, Status};

// The one `live_edit_checkpoint_ext_recovery_total{result}` reader. Named
// rather than glob-imported because `helpers` also exports a `SERVICE_LABEL`
// that would collide with this file's own const. This replaces a file-local
// copy that was typed on `&ExtValueRecoveryHarness` — out of reach by
// signature for every other file, which is how a second copy came to exist in
// `composed_restore_ext_matrix_test.rs`. The shared one takes `&Metrics`, so
// `&harness.metrics` (a `SharedMetrics`) deref-coerces at each call site.
use super::helpers::ext_recovery_count;
use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_external};
use crate::grpc::helpers::{
    make_create_data, make_create_header, make_request, streaming_create, AllowAllPermissions,
    StubUserinfo,
};

/// Service label stamped onto the `Metrics` registry and threaded into
/// `LiveEditService::service_label`. Same value the shared
/// [`crate::checkpoint::helpers::SERVICE_LABEL`] constant uses so any
/// future cross-file label-cell read converges on one address.
const SERVICE_LABEL: &str = "test";

/// Bundle returned by [`setup_ext_value_recovery_harness`]. Owns every
/// handle a `compute_diff_at_cut`-misses → adapter-fallback → Valkey-rewrite
/// E2E flow needs end-to-end.
///
/// The `LiveEditService::ext_value_store` field is wired with a REAL
/// `ValkeyExternalValueStore` carrying `.with_checkpoint_store(store)`
/// — this is the load-bearing wiring this file exists to exercise.
/// The `recovery` hook is also REAL (enabled), pointed at the same
/// `LocalDiskCheckpointStore` the periodic loop is using, so any
/// downstream test in this file (or a sibling Subtask's test) that
/// wipes the object header in addition to the ext namespace can also
/// observe the SR-2.4 layer-recovery path firing — though this
/// file's happy-path test deliberately keeps the object header
/// intact and pins the SR-3.2 ext-value path alone.
pub(super) struct ExtValueRecoveryHarness {
    /// Live Valkey connection + key schema shared across the service,
    /// the recovery hook, and the periodic checkpoint loop.
    pub tv: TestValkey,
    /// `Arc<dyn CheckpointStore>` rooted at the harness tempdir. Shared
    /// with the `RecoveryHook` so the hook reads the same generations
    /// the loop writes, AND with `ValkeyExternalValueStore`'s
    /// `with_checkpoint_store` so the SR-3.2 fallback hits the same
    /// adapter.
    pub store: Arc<dyn CheckpointStore>,
    /// Live `LiveEditService` wired against the real (enabled)
    /// `RecoveryHook` and the real `ValkeyExternalValueStore`. Tests
    /// pass this directly to `handle_create_layer` /
    /// `handle_subscribe_read_layer`.
    pub svc: LiveEditService,
    /// Live `SharedMetrics` shared between the service and the loop —
    /// `checkpoint_ext_recovery_total{result="recovered"}` assertions
    /// read this Arc.
    pub metrics: SharedMetrics,
    /// `JoinHandle` for the periodic checkpoint loop. Tests that need
    /// a clean shutdown fire `cancel.cancel()` and `loop_handle.await`.
    pub loop_handle: JoinHandle<()>,
    /// `CancellationToken` paired with `loop_handle`.
    pub cancel: CancellationToken,
    /// LocalDisk adapter root — kept alive for the duration of the
    /// harness. RAII-cleaned on drop. Held so the `TempDir` survives
    /// until the harness itself is dropped (the adapter holds a
    /// `PathBuf` rooted under this directory).
    #[allow(dead_code)]
    pub tempdir: TempDir,
}

/// Construct a fully-wired ext-value-recovery E2E harness backed by
/// `LocalDiskCheckpointStore`.
///
/// Wiring shape:
///
/// - Real `LocalDiskCheckpointStore` rooted at a fresh `TempDir`,
///   shared with the `RecoveryHook` AND the
///   `ValkeyExternalValueStore`'s `.with_checkpoint_store(...)` chain.
/// - Real `ValkeyExternalValueStore` against the live `TestValkey`;
///   this is the LOAD-BEARING departure from `setup_checkpoint_harness`
///   in `helpers.rs` which uses `MockExternalValueStore`. The mock
///   has no `read_batch` miss path and never consults a checkpoint
///   store — useless for SR-3.2 coverage.
/// - Real (enabled) `RecoveryHook` and `SessionRecoveryHook` so any
///   sibling test in this file that wipes the object header in
///   addition to the ext namespace also exercises the layer-recovery
///   path. The happy-path test deliberately keeps the object header
///   intact.
/// - Periodic checkpoint loop spawned with `interval=0` / `quiesce=0`
///   per the same convention `setup_checkpoint_harness` uses. The
///   loop is harmless to the happy-path SR-3.2 test (we pre-seed the
///   adapter directly), but having it running rules out a "we
///   accidentally wired the recovery branch off the loop" false
///   positive.
///
/// `tick_secs` is plumbed into the loop's `CheckpointLoopConfig::tick`
/// — siblings that exercise loop-driven mirror landings will pass
/// `1`; the happy-path SR-3.2 test does NOT depend on the loop
/// landing a generation, so any non-zero value works.
pub(super) async fn setup_ext_value_recovery_harness(tick_secs: u64) -> ExtValueRecoveryHarness {
    let tv = TestValkey::setup().await;

    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    // Real ValkeyExternalValueStore wired with the checkpoint-store
    // handle. This is the production-shape construction chain from
    // `main.rs` Step 1b-iii.4 minus the ext-mirror sender (the SR-3.1
    // mirror pipeline is orthogonal to the SR-3.2 fetch-on-miss path
    // this file exercises; pre-seeding the adapter directly via
    // `store.put_ext_blob(...)` after CreateLayer is exactly what the
    // Subtask body prescribes for test-independence from the mirror's
    // async timing).
    let ext_value_store: Arc<dyn ExternalValueStore> = Arc::new(
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(&metrics))
            .with_checkpoint_store(Arc::clone(&store)),
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

    // Real (enabled) RecoveryHook. The Subtask body's happy-path test
    // keeps the object header intact so this hook never fires, but
    // wiring it identically to `setup_checkpoint_harness` keeps the
    // test harness production-faithful for siblings that may extend
    // this with mixed wipe shapes.
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::LocalDisk;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = tick_secs.max(1);
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_local_disk_path = Some(tempdir.path().to_string_lossy().into_owned());
    let cfg_arc = Arc::new(cfg);

    let recovery = Arc::new(RecoveryHook::new(
        Arc::clone(&store),
        Arc::clone(&layer_store),
        Arc::clone(&cfg_arc),
        Arc::clone(&metrics),
    ));
    let session_recovery = Arc::new(SessionRecoveryHook::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&cfg_arc),
        Arc::clone(&metrics),
    ));

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
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        ),
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

    // Periodic checkpoint loop. Harmless to the SR-3.2 test (we
    // pre-seed the adapter directly via `put_ext_blob`), but matches
    // the production wiring shape.
    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_secs(tick_secs.max(1)),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(SERVICE_LABEL),
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

    ExtValueRecoveryHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        tempdir,
    }
}

/// `UNLINK` `le:ext:meta:{hash_hex}` + `le:ext:data:{hash_hex}` for
/// every supplied hash, simulating a partial Valkey state-loss event
/// that wipes the ext-value namespace but leaves the materialised
/// object structure intact (the SR-3.2 recovery surface).
///
/// Deliberately surgical: per the Subtask body, do NOT use a wildcard
/// pattern that would also catch `le:ext:{gc}:pending` /
/// `le:ext:{gc}:inflight` (the GC bookkeeping sets — wiping those
/// changes the test's semantics) and do NOT touch the
/// `le:{object_id}:obj:*` keys (the object's materialised structure
/// must survive so the subscribe-path runs the normal diff-build
/// against an intact header).
///
/// Returns once every `UNLINK` reply has landed. Per-hash atomicity
/// matches the SR-3.2 fallback's view of the wipe (the recovery
/// branch fires when EITHER meta OR data is missing; wiping both is
/// the most natural simulation of a Valkey namespace loss).
pub(super) async fn wipe_ext_namespace_for_hashes(tv: &TestValkey, hash_hexes: &[String]) {
    for hash_hex in hash_hexes {
        let meta_key = tv.key_schema().ext_meta(hash_hex);
        let data_key = tv.key_schema().ext_data(hash_hex);
        let _: () = tv
            .client()
            .pool
            .unlink((meta_key, data_key))
            .await
            .expect("UNLINK of ext_meta + ext_data must succeed");
    }
}

/// Drain the SubscribeReadLayer bootstrap message: pull the first
/// `Header`, then pull `Chunk`s until `total_size` bytes accumulate.
/// Returns the header (for `object_id` / `sequence_num` assertions)
/// and the concatenated body bytes (the OMPF the subscriber sees).
///
/// 10 s per-step timeout absorbs CI scheduling jitter; a healthy
/// happy-path subscribe lands the bootstrap in sub-millisecond time
/// against the in-memory in-process channel.
pub(super) async fn recv_bootstrap_message<S>(
    stream: &mut S,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("subscribe stream timed out waiting for bootstrap header")
        .expect("subscribe stream closed unexpectedly before bootstrap header")
        .expect("subscribe stream yielded an error before bootstrap header");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected SubscribeReadLayerHeader, got: {other:?}"),
    };

    let mut buf = Vec::new();
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("subscribe stream timed out waiting for chunk")
            .expect("subscribe stream closed before total_size reached")
            .expect("subscribe stream yielded an error during chunk drain");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk while draining bootstrap body, got: {other:?}"),
        }
    }

    (header, buf)
}

/// Build a `Request<SubscribeReadLayerRequest>` carrying a stub
/// `BearerToken` extension so the auth interceptor accepts it.
/// `client_id` is fixed because the happy-path test doesn't care
/// about self-echo suppression — the writer and subscriber are the
/// same task in this test and the bootstrap-message path is not
/// `client_id`-filtered (only the live per-commit path is).
pub(super) fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: "ext-value-recovery-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ════════════════════════════════════════════════════════════════════════════
// Subtask t3.fgj.zc.p6.co — Happy-path SR-3.2 recovery.
//
// Pre-seed the adapter, wipe the ext namespace, subscribe, observe the
// bootstrap body carrying the original payload byte-identical and
// Valkey re-hydrated with the recovered blob. Materialised object
// structure left intact so we are ONLY pinning the SR-3.2 fetch-on-miss
// path (NOT the SR-2.4 layer-recovery path — that has its own e2e
// coverage in `e2e_subscribe_read_layer_recovery_test.rs`).
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_recovers_ext_value_field_after_ext_namespace_wipe() {
    let harness = setup_ext_value_recovery_harness(1).await;

    // Snapshot the metric BEFORE any traffic so the post-subscribe
    // assertion is a clean delta. The metric is process-wide (no
    // service label), so any other test running in the same binary
    // could contribute to the absolute value; the delta is the
    // load-bearing signal.
    let pre_recovered = ext_recovery_count(&harness.metrics, "recovered");

    // ── 1. CreateSession ──────────────────────────────────────────────────
    //
    // Direct SessionStore::create — no auth interceptor in the path, the
    // existing recovery e2e tests in this directory use the same shape.
    const SESSION_NAME: &str = "ext-value-recovery-session";
    const LAYER_NAME: &str = "ext-value-recovery-layer";
    let session_id = harness
        .svc
        .session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ── 2. Pre-store the ext-value blob in Valkey ─────────────────────────
    //
    // The Subtask body's outline step 2 prescribes calling
    // `harness.svc.ext_value_store.store_values(...)` BEFORE the
    // CreateLayer so the ref-count is owned by the layer's later
    // write-through (CreateLayer's `store_values` is idempotent on
    // the SET ext_data side and `HINCRBY ext_meta ref_count 1` on the
    // meta side — calling it twice for the same hash is benign and
    // produces ref_count >= 2; the orphan sweep reconciles drift per
    // SR-3.3, see the SDD §11.1 idempotent-write note).
    //
    // Pre-storing AND CreateLayer-storing the same hash is the
    // production shape: the assembler and the CreateLayer handler
    // would race for any layer that referenced an already-known
    // hash. The wipe step below removes both, so the final ref_count
    // post-recovery is exactly the one bump performed by the recovery
    // path's rewrite.
    let payload: &[u8] = b"sr-3.2-recovery-canonical-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);
    let pair: [(&[u8; 32], &[u8]); 1] = [(&hash, payload)];
    harness
        .svc
        .ext_value_store
        .store_values(&pair)
        .await
        .expect("initial ext-value store_values must succeed");

    // ── 3. CreateLayer with a single SetField referencing the ext blob ────
    //
    // `ext_value_index = 1` (1-based, the first external blob in the
    // OMPF). `ext_value_size` carries the byte length so the consumer-side
    // OODF can size the slot without re-deriving it.
    let delta: OwnedDelta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "ext-field",
        /*hash=*/ hash.to_vec(),
        /*ext_value_size=*/ payload.len() as u64,
        /*ext_value_index=*/ 1,
        /*set_order=*/ 1,
    );
    let create_ompf = delta_to_ompf_bytes(&delta, &[payload]);

    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with ext-value must succeed");
    let object_id = create_resp.into_inner().object_id;

    // ── 4. Pre-seed the adapter ───────────────────────────────────────────
    //
    // Per the Subtask body: "Since Task A's mirror is asynchronous and
    // best-effort, this test bypasses the mirror and pre-seeds the adapter
    // directly via `store.put_ext_blob(hash_hex, bytes)` after `CreateLayer`
    // lands. That makes the test independent of Task A's worker-pool
    // timing and pins purely the fetch-on-miss contract."
    harness
        .store
        .put_ext_blob(&hash_hex, Bytes::copy_from_slice(payload))
        .await
        .expect("put_ext_blob direct pre-seed must succeed");

    // ── 5. Wipe ONLY the ext namespace for our hash ───────────────────────
    //
    // Object header + sections + per-object stream remain intact — the
    // subscribe-path runs the normal diff-build against a healthy
    // materialised object and the FIRST miss it encounters is the
    // ext-value blob.
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&hash_hex)).await;

    // Sanity probe — confirm the wipe actually removed both ext keys.
    // A regression that left either key intact would render the
    // recovery-counter assertion below vacuous (the hot-path would
    // succeed without consulting the adapter).
    let meta_key = harness.tv.key_schema().ext_meta(&hash_hex);
    let data_key = harness.tv.key_schema().ext_data(&hash_hex);
    let meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&meta_key)
        .await
        .expect("EXISTS ext_meta probe must succeed");
    let data_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&data_key)
        .await
        .expect("EXISTS ext_data probe must succeed");
    assert!(
        !meta_exists,
        "wipe must leave ext_meta:{hash_hex} absent (pre-subscribe sanity probe)"
    );
    assert!(
        !data_exists,
        "wipe must leave ext_data:{hash_hex} absent (pre-subscribe sanity probe)"
    );

    // Confirm via read_object_state_for_test that the materialised object
    // state is still readable post-wipe (per the Subtask body's outline
    // step 5: "Confirm via `load_object_snapshot` that the object's
    // materialised state is still readable"; that loader is gone and its
    // test-support successor does the read). This guards against a
    // regression where the wipe pattern accidentally caught the
    // `le:{oid}:obj:*` keys — which would silently flip this test
    // into exercising the SR-2.4 layer-recovery path instead of the
    // SR-3.2 ext-value path.
    let object_id_str = object_id.to_string();
    let post_wipe_snapshot = live_edit::storage::test_support::read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id_str,
    )
    .await
    .expect("read_object_state_for_test must succeed post-wipe — object structure intact");
    assert_eq!(
        post_wipe_snapshot.object_id, object_id,
        "post-wipe snapshot must carry the same object_id"
    );

    // ── 6. SubscribeReadLayer with have_object_id=oid, sequence_num=0 ─────
    //
    // `sequence_num=0` with a non-zero `have_object_id` is the
    // "subscriber wants a full reset from a stale-cache cold start"
    // shape — `DiffEngine::plan` short-circuits to `FullContent`,
    // which emits every section + field. The ext-value field is in
    // section 1, so the diff-build calls `read_batch` for our hash,
    // misses (both ext keys wiped), falls back to
    // `try_recover_via_checkpoint`, which fetches the bytes from the
    // adapter and rewrites them into Valkey via `store_recovered_values`
    // (`store_values`' write, without the ext-mirror copy).
    let sub_req = subscribe_request(&session_id, LAYER_NAME, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("SubscribeReadLayer must return Ok after SR-3.2 recovery");
    let mut stream = response.into_inner();

    let (bootstrap_header, bootstrap_body) = recv_bootstrap_message(&mut stream).await;

    // The bootstrap header carries our object_id and the layer's
    // current seq (which equals 1 here — only CreateLayer ran). The
    // body must be non-empty; we decode it below.
    assert_eq!(
        bootstrap_header.object_id, object_id,
        "bootstrap header object_id must round-trip the CreateLayer id"
    );
    assert!(
        !bootstrap_body.is_empty(),
        "bootstrap body must be non-empty — FullContent diff covers section 1's SetField"
    );

    // ── 7. Decode the bootstrap body, resolve ext-value, assert bytes ─────
    //
    // The bootstrap body is an OMPF. Part 0 is the OODF delta; parts
    // 1..N are external blobs. The SetField's `ext_value_index` is
    // 1-based (index 1 = the first external blob == `external_values[0]`).
    let container = live_edit::domain::ompf::OmpfContainer::parse(&bootstrap_body)
        .expect("bootstrap OMPF must parse");
    assert!(
        !container.delta_bytes.is_empty(),
        "bootstrap OMPF delta part must be non-empty (FullContent reset delta)"
    );
    let decoded_delta =
        oodf_ffi::parse_delta(container.delta_bytes).expect("OODF delta must parse");

    // Find the SetField referencing our hash. The FullContent plan
    // emits every field; for this one-field layer there is exactly one.
    let set_field = decoded_delta
        .set_fields
        .iter()
        .find(|f| f.value_or_ext_hash.as_slice() == hash.as_ref())
        .expect("bootstrap delta must carry a SetField for our ext-value hash");
    assert!(
        set_field.ext_value_index >= 1,
        "ext-value-referencing SetField must carry ext_value_index >= 1 (1-based)"
    );
    let blob_idx_zero_based = (set_field.ext_value_index as usize) - 1;
    let recovered_bytes = container
        .external_values
        .get(blob_idx_zero_based)
        .unwrap_or_else(|| {
            panic!(
                "bootstrap OMPF external_values must contain blob index {} (1-based: {})",
                blob_idx_zero_based, set_field.ext_value_index
            )
        });
    assert_eq!(
        *recovered_bytes, payload,
        "post-recovery bootstrap body's ext-value bytes must equal the original payload byte-for-byte"
    );

    // ── 8. Assert Valkey re-hydration: ext_meta + ext_data present ────────
    //
    // The SR-3.2 fallback's rewrite path calls
    // `store_recovered_values`, which runs `store_values`' per-hash
    // write (`SET ext_data NX; HINCRBY ext_meta ref_count 1`) without
    // the ext-mirror copy. After the subscribe both keys must be back, with
    // `ref_count >= 1`. The exact value of ref_count depends on how
    // many singleflight waiters all rewrote the
    // same hash within this RPC (we only have one missing field, so
    // it should be exactly 1) — `>= 1` is the load-bearing pin.
    let post_data: Bytes = harness
        .tv
        .client()
        .pool
        .get(&data_key)
        .await
        .expect("post-recovery GET ext_data must succeed (Valkey re-hydrated)");
    assert_eq!(
        post_data.as_ref(),
        payload,
        "post-recovery Valkey ext_data bytes must equal the original payload byte-for-byte"
    );

    let ref_count: Option<i64> = harness
        .tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("post-recovery HGET ext_meta ref_count must succeed");
    let ref_count = ref_count.expect(
        "post-recovery ext_meta must carry a ref_count field — store_recovered_values's HINCRBY \
         landed",
    );
    assert!(
        ref_count >= 1,
        "post-recovery ext_meta ref_count must be >= 1, observed {ref_count}"
    );

    // ── 9. Metric pin: exactly one "recovered" tick fired for this RPC ────
    //
    // The singleflight guard collapses concurrent misses on the same
    // hash to one adapter fetch + one rewrite. Our subscribe RPC
    // misses on exactly one hash, so the counter must move by
    // exactly 1.
    let post_recovered = ext_recovery_count(&harness.metrics, "recovered");
    assert_eq!(
        post_recovered.saturating_sub(pre_recovered),
        1,
        "checkpoint_ext_recovery_total{{result=\"recovered\"}} must tick by exactly 1 \
         (observed delta {}, pre={pre_recovered}, post={post_recovered})",
        post_recovered.saturating_sub(pre_recovered),
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_miss"),
        0,
        "no adapter_miss ticks expected — pre-seed populated the adapter"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_transient"),
        0,
        "no adapter_transient ticks expected — LocalDisk store is healthy"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_permanent"),
        0,
        "no adapter_permanent ticks expected — LocalDisk store is healthy"
    );

    // ── 10. Drain the harness ─────────────────────────────────────────────
    //
    // Drop the subscribe stream first so the bridge task can observe
    // its receiver closing and exit cleanly; then cancel the loop and
    // await it under a bounded timeout. Mirrors the shutdown ceremony
    // in `e2e_subscribe_read_layer_recovery_test.rs`.
    drop(stream);
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ════════════════════════════════════════════════════════════════════════════
// Subtask t3.fgj.zc.p6.wy — Multi-field heavy-layer end-to-end recovery
// (Epic AC #3 / PRD AC #4 byte-volume acceptance pin).
//
// The happy-path test above pins the SR-3.2 fetch-on-miss path with a single
// short ext-value payload. This test extends the same path to the realistic
// byte volume the Epic-3 acceptance criterion calls out: a layer whose ext
// content totals well over 100 KB across MULTIPLE fields distributed across
// MULTIPLE sections. Recovery must remain byte-identical for every field —
// no truncation, no payload swap, no per-field skew.
//
// Wiring shape:
//
//   - 4 distinct payloads, each 32 KiB of deterministic pseudo-random bytes
//     (seeded per-index so a regression that swaps payload<->hash produces
//     a clean assert_eq! diff naming the index + hash hex). 4 * 32 KiB =
//     128 KiB total, well above the 100 KB AC floor while staying inside
//     the Subtask's 256 KB memory budget.
//   - Fields distributed across 2 sections (section_ids 1 and 2): payloads
//     0,2 → section 1; payloads 1,3 → section 2. Interleaving by parity
//     ensures the per-section diff-build emits both halves of the payload
//     set and any "first section emits, second section short-circuits"
//     regression surfaces as a byte mismatch on the second-section field.
//   - OMPF: Part 0 = OODF delta, Parts 1..4 = the 4 payloads. SetField
//     `ext_value_index` is 1-based (1 → first external blob → payloads[0]).
//   - Pre-seed the adapter directly for every hash (same `put_ext_blob`
//     pattern the happy-path test uses, mirrored 4×).
//   - Wipe the ext namespace for ALL 4 hashes.
//   - SubscribeReadLayer with `sequence_num=0` → FullContent diff plan
//     → `read_batch` for all 4 hashes misses → SR-3.2 fallback fires for
//     every hash → adapter returns the bytes → Valkey rehydrated →
//     bootstrap body carries every payload byte-identical.
//
// Byte-identical-by-assert_eq is the Subtask's load-bearing contract: the
// diagnostic on mismatch names the field index AND the hash hex so a
// regression that swaps two payloads (e.g. an off-by-one ext_value_index)
// produces a self-explanatory failure.
//
// Does NOT depend on `feature = "test-checkpoint-fault-injection"` — this
// is production happy-path coverage with a healthy LocalDisk adapter.
// ════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ext_value_heavy_layer_recovers_byte_identical_after_full_ext_namespace_wipe() {
    /// Number of distinct ext-value payloads. 4 fields × 32 KiB each =
    /// 128 KiB of ext content, exceeds the Epic AC #3 / PRD AC #4 100 KB
    /// floor and stays inside the Subtask's 256 KB memory budget.
    const N: usize = 4;
    /// Per-payload byte length. 32 KiB is large enough to make the
    /// `read_batch` Valkey roundtrip meaningful while keeping the total
    /// payload-vector allocation well-bounded.
    const PAYLOAD_LEN: usize = 32 * 1024;

    let harness = setup_ext_value_recovery_harness(1).await;

    // Snapshot the recovery metric BEFORE traffic so we can pin the
    // exact post-subscribe delta. Singleflight collapses concurrent
    // misses on the same hash; here every hash is distinct, so the
    // delta must equal N (one tick per recovered hash).
    let pre_recovered = ext_recovery_count(&harness.metrics, "recovered");
    let pre_adapter_miss = ext_recovery_count(&harness.metrics, "adapter_miss");
    let pre_adapter_transient = ext_recovery_count(&harness.metrics, "adapter_transient");
    let pre_adapter_permanent = ext_recovery_count(&harness.metrics, "adapter_permanent");

    // ── 1. CreateSession ──────────────────────────────────────────────────
    const SESSION_NAME: &str = "ext-value-recovery-heavy-session";
    const LAYER_NAME: &str = "ext-value-recovery-heavy-layer";
    let session_id = harness
        .svc
        .session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ── 2. Build N distinct 32 KiB payloads + their hashes ────────────────
    //
    // Deterministic per-index pseudo-random fill: each byte = (i * 257 +
    // j * 31 + 17) mod 256. This produces high-entropy payloads that
    // differ in every byte across index `i`, so a regression that swaps
    // payload `i` for payload `j` (off-by-one ext_value_index, etc.)
    // produces an obvious byte-level diff naming both halves. We do NOT
    // use an actual RNG (proptest is a separate test family and pulling
    // in a `rand` dep here is unnecessary for a fixture-shape test) —
    // the deterministic formula is enough to defeat any byte-equality
    // false positive a regression might produce.
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(N);
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(N);
    let mut hash_hexes: Vec<String> = Vec::with_capacity(N);
    for i in 0..N {
        let mut payload = Vec::with_capacity(PAYLOAD_LEN);
        for j in 0..PAYLOAD_LEN {
            payload.push(((i * 257 + j * 31 + 17) & 0xff) as u8);
        }
        let hash = ext_content_hash(&payload);
        let hash_hex = sha256_to_hex(&hash);
        payloads.push(payload);
        hashes.push(hash);
        hash_hexes.push(hash_hex);
    }
    // Sanity: every payload-hash pair is distinct. The deterministic
    // formula above varies in every byte across `i`, so collisions are
    // not expected, but pinning the assumption protects against an
    // accidental edit that rewrites the formula to repeat across `i`.
    for i in 0..N {
        for j in (i + 1)..N {
            assert_ne!(
                hashes[i], hashes[j],
                "payload hashes must be distinct (i={i}, j={j})"
            );
        }
    }

    // ── 3. Pre-store every payload's ext-value in Valkey ──────────────────
    //
    // Same idempotent-write pattern the happy-path test uses: pre-store
    // → CreateLayer (write-through hits the same `store_values` again,
    // benign) → adapter pre-seed → wipe → subscribe → recovery rewrite.
    // Build the `(hash, payload)` pair slice through borrowed refs so
    // we don't move out of the `payloads` / `hashes` Vecs we still need
    // for the OMPF build below.
    let pairs: Vec<(&[u8; 32], &[u8])> = (0..N)
        .map(|i| (&hashes[i], payloads[i].as_slice()))
        .collect();
    harness
        .svc
        .ext_value_store
        .store_values(&pairs)
        .await
        .expect("initial multi-hash ext_value_store store_values must succeed");

    // ── 4. CreateLayer with N SetField entries across 2 sections ──────────
    //
    // Field-to-section distribution: even-index payloads (0, 2) → section
    // 1; odd-index payloads (1, 3) → section 2. Interleaving by parity
    // guarantees both sections carry multiple fields so the FullContent
    // diff-build's per-section emit-loop is exercised end-to-end. The
    // `ext_value_index` is 1-based per the OMPF contract — field `i`
    // references external blob `i + 1`, which in the OMPF's
    // `external_values: &[&[u8]]` argument is `external_values[i]`.
    // `set_order` is monotonically increasing per the SetField contract
    // (per-section order; uniqueness across the whole layer is harmless
    // — the field's identity is `(section_id, key_name)`, not set_order).
    let set_fields: Vec<SetField> = (0..N)
        .map(|i| SetField {
            section_id: if i % 2 == 0 { 1 } else { 2 },
            key_name: format!("ext-field-{i}"),
            value_or_ext_hash: hashes[i].to_vec(),
            ext_value_size: payloads[i].len() as u64,
            ext_value_index: (i as u32) + 1,
            set_order: (i as u64) + 1,
        })
        .collect();
    let delta = OwnedDelta {
        set_fields,
        ..Default::default()
    };
    let external_values: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
    let create_ompf = delta_to_ompf_bytes(&delta, &external_values);

    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with N ext-value fields must succeed");
    let object_id = create_resp.into_inner().object_id;

    // ── 5. Pre-seed every payload into the adapter ────────────────────────
    //
    // Same bypass-the-async-mirror pattern as the happy-path test, looped
    // over the N hashes. Sequential — the LocalDisk store is fast enough
    // that concurrent puts here would not save meaningful wall-time and
    // would complicate the failure diagnostics.
    for i in 0..N {
        harness
            .store
            .put_ext_blob(&hash_hexes[i], Bytes::copy_from_slice(&payloads[i]))
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "put_ext_blob pre-seed must succeed for i={i} hash={}: {e}",
                    hash_hexes[i]
                )
            });
    }

    // ── 6. Capture pre-wipe ground truth via read_object_state_for_test ─────────
    //
    // Subtask body outline step 5: "Capture the pre-wipe state via
    // `load_object_snapshot` for ground truth." (that loader is gone; its
    // test-support successor does the read). This pins that the
    // materialised object structure carries every expected field BEFORE
    // the wipe — a regression that silently dropped fields on
    // CreateLayer ingestion would surface here, before the wipe is even
    // performed.
    let object_id_str = object_id.to_string();
    let pre_wipe_snapshot = live_edit::storage::test_support::read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id_str,
    )
    .await
    .expect("read_object_state_for_test pre-wipe must succeed");
    assert_eq!(
        pre_wipe_snapshot.object_id, object_id,
        "pre-wipe snapshot object_id must round-trip the CreateLayer id"
    );

    // ── 7. Wipe ALL N ext namespace entries ───────────────────────────────
    //
    // Surgical: every (ext_meta, ext_data) pair for our N hashes. Does
    // NOT touch the materialised object's `le:{oid}:obj:*` keys (the
    // diff-build needs an intact header) and does NOT touch the GC
    // bookkeeping sets (per the Subtask body's wipe contract).
    wipe_ext_namespace_for_hashes(&harness.tv, &hash_hexes).await;

    // Sanity probe: every (meta, data) pair must be absent post-wipe.
    // Vacuum-detection — if the wipe pattern accidentally left even one
    // key intact, the recovery-counter assertion below would not pin
    // the SR-3.2 path firing for that hash (the hot-path would succeed
    // without consulting the adapter).
    for (i, hash_hex) in hash_hexes.iter().enumerate() {
        let meta_key = harness.tv.key_schema().ext_meta(hash_hex);
        let data_key = harness.tv.key_schema().ext_data(hash_hex);
        let meta_exists: bool = harness
            .tv
            .client()
            .pool
            .exists(&meta_key)
            .await
            .unwrap_or_else(|e| panic!("EXISTS ext_meta probe i={i}: {e}"));
        let data_exists: bool = harness
            .tv
            .client()
            .pool
            .exists(&data_key)
            .await
            .unwrap_or_else(|e| panic!("EXISTS ext_data probe i={i}: {e}"));
        assert!(
            !meta_exists,
            "wipe must leave ext_meta:{hash_hex} absent (i={i})"
        );
        assert!(
            !data_exists,
            "wipe must leave ext_data:{hash_hex} absent (i={i})"
        );
    }

    // ── 8. SubscribeReadLayer (FullContent diff plan) ─────────────────────
    //
    // `sequence_num=0` with `have_object_id=object_id` short-circuits
    // `DiffEngine::plan` to `FullContent`, which emits every section's
    // every field. With 4 ext-value-referencing fields across 2 sections,
    // the bootstrap body's OODF delta carries all 4 SetField entries and
    // the OMPF carries all 4 ext blobs. The diff-build's `read_batch`
    // hits 4 distinct misses, fires the SR-3.2 fallback for each, and
    // populates the bootstrap body's external_values from the adapter
    // bytes.
    let sub_req = subscribe_request(&session_id, LAYER_NAME, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("SubscribeReadLayer must return Ok after multi-hash SR-3.2 recovery");
    let mut stream = response.into_inner();

    let (bootstrap_header, bootstrap_body) = recv_bootstrap_message(&mut stream).await;
    assert_eq!(
        bootstrap_header.object_id, object_id,
        "bootstrap header object_id must round-trip the CreateLayer id"
    );
    assert!(
        !bootstrap_body.is_empty(),
        "bootstrap body must be non-empty — FullContent diff covers all N fields"
    );

    // ── 9. Decode OMPF, resolve each SetField, assert byte-identical ──────
    //
    // The bootstrap OMPF carries the FullContent reset delta (part 0) +
    // every ext blob (parts 1..N). For each SetField in the decoded
    // delta, locate its source-side payload by hash (the SetField's
    // `value_or_ext_hash` carries the content hash), then resolve its
    // `ext_value_index` (1-based) to the OMPF external blob and assert
    // the bytes equal the source payload byte-for-byte.
    //
    // The order of SetField entries in the bootstrap delta is NOT
    // guaranteed to match the source-side order (the FullContent
    // diff-build walks sections in section-id order and fields in
    // canonical order within each section). Mapping by hash content
    // rather than by index defeats that ordering assumption.
    let container = live_edit::domain::ompf::OmpfContainer::parse(&bootstrap_body)
        .expect("bootstrap OMPF must parse");
    assert!(
        !container.delta_bytes.is_empty(),
        "bootstrap OMPF delta part must be non-empty (FullContent reset delta)"
    );
    let decoded_delta =
        oodf_ffi::parse_delta(container.delta_bytes).expect("OODF delta must parse");

    // Pin: exactly N SetField entries surface in the bootstrap delta.
    // Fewer = a regression that silently dropped fields; more = a
    // regression that duplicated them. Either is a fault.
    assert_eq!(
        decoded_delta.set_fields.len(),
        N,
        "bootstrap delta must carry exactly {N} SetField entries (one per source field), \
         observed {}",
        decoded_delta.set_fields.len()
    );

    // For each source field, find its matching bootstrap SetField (by
    // hash) and assert byte-identical recovery. The diagnostic on
    // mismatch names BOTH the source index and the hash hex so a swap
    // regression produces a self-explanatory failure naming the offended
    // pair.
    for i in 0..N {
        let bootstrap_field = decoded_delta
            .set_fields
            .iter()
            .find(|f| f.value_or_ext_hash.as_slice() == hashes[i].as_ref())
            .unwrap_or_else(|| {
                panic!(
                    "bootstrap delta must carry a SetField for source field i={i} \
                     hash={}",
                    hash_hexes[i]
                )
            });
        // Sanity: the bootstrap field references the same section as
        // the source. A regression that re-bucketed fields during
        // diff-build would surface here.
        let expected_section_id = if i % 2 == 0 { 1 } else { 2 };
        assert_eq!(
            bootstrap_field.section_id, expected_section_id,
            "bootstrap field for i={i} hash={} must live in section_id={expected_section_id}, \
             observed {}",
            hash_hexes[i], bootstrap_field.section_id
        );
        assert!(
            bootstrap_field.ext_value_index >= 1,
            "bootstrap field for i={i} hash={} must carry ext_value_index >= 1 (1-based), \
             observed {}",
            hash_hexes[i],
            bootstrap_field.ext_value_index
        );
        let blob_idx_zero_based = (bootstrap_field.ext_value_index as usize) - 1;
        let recovered_bytes = container
            .external_values
            .get(blob_idx_zero_based)
            .unwrap_or_else(|| {
                panic!(
                    "bootstrap OMPF external_values must contain blob index {} \
                         (1-based: {}) for source field i={i} hash={}",
                    blob_idx_zero_based, bootstrap_field.ext_value_index, hash_hexes[i]
                )
            });
        assert_eq!(
            *recovered_bytes,
            payloads[i].as_slice(),
            "field i={i} hash={} bytes diverged — recovered payload must equal source \
             payload byte-for-byte (recovered_len={}, source_len={})",
            hash_hexes[i],
            recovered_bytes.len(),
            payloads[i].len()
        );
    }

    // ── 10. Assert Valkey re-hydration: every (meta, data) pair restored ──
    //
    // The SR-3.2 fallback's `store_recovered_values` rewrite path runs
    // per-hash, so every wiped key must be back post-subscribe with
    // `ref_count >= 1`. The exact ref_count depends on how many
    // singleflight waiters all rewrote the same hash
    // (we have N distinct hashes with one miss each, so 1 per hash) —
    // `>= 1` is the load-bearing pin.
    for i in 0..N {
        let meta_key = harness.tv.key_schema().ext_meta(&hash_hexes[i]);
        let data_key = harness.tv.key_schema().ext_data(&hash_hexes[i]);

        let post_data: Bytes = harness
            .tv
            .client()
            .pool
            .get(&data_key)
            .await
            .unwrap_or_else(|e| panic!("post-recovery GET ext_data i={i}: {e}"));
        assert_eq!(
            post_data.as_ref(),
            payloads[i].as_slice(),
            "post-recovery Valkey ext_data bytes for i={i} hash={} must equal the \
             original payload byte-for-byte",
            hash_hexes[i]
        );

        let ref_count: Option<i64> = harness
            .tv
            .client()
            .pool
            .hget(&meta_key, "ref_count")
            .await
            .unwrap_or_else(|e| panic!("post-recovery HGET ext_meta ref_count i={i}: {e}"));
        let ref_count = ref_count.unwrap_or_else(|| {
            panic!(
                "post-recovery ext_meta for i={i} hash={} must carry a ref_count field — \
                 store_recovered_values's HINCRBY landed",
                hash_hexes[i]
            )
        });
        assert!(
            ref_count >= 1,
            "post-recovery ext_meta ref_count for i={i} hash={} must be >= 1, observed {ref_count}",
            hash_hexes[i]
        );
    }

    // ── 11. Metric pin: exactly N "recovered" ticks fired this RPC ────────
    //
    // N distinct hashes × 1 miss-per-hash = N recovered ticks. No other
    // classification arms must tick — the adapter is healthy (no
    // transient or permanent) and the pre-seed populated every blob (no
    // adapter_miss).
    let post_recovered = ext_recovery_count(&harness.metrics, "recovered");
    assert_eq!(
        post_recovered.saturating_sub(pre_recovered),
        N as u64,
        "checkpoint_ext_recovery_total{{result=\"recovered\"}} must tick by exactly {N} \
         (one per recovered hash); observed delta {}, pre={pre_recovered}, post={post_recovered}",
        post_recovered.saturating_sub(pre_recovered),
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_miss"),
        pre_adapter_miss,
        "no adapter_miss ticks expected — pre-seed populated the adapter for every hash"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_transient"),
        pre_adapter_transient,
        "no adapter_transient ticks expected — LocalDisk store is healthy"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_permanent"),
        pre_adapter_permanent,
        "no adapter_permanent ticks expected — LocalDisk store is healthy"
    );

    // ── 12. Drain the harness ─────────────────────────────────────────────
    drop(stream);
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ════════════════════════════════════════════════════════════════════════════
// Subtask t3.fgj.zc.p6.ia — Adapter-empty / Transient / Permanent error
// classification.
//
// The Task Success Criterion #2: the SR-3.2 fetch-on-miss path MUST
// preserve the three distinct error classifications when the adapter
// cannot serve the bytes:
//
//   - Adapter returns `Ok(None)` (blob simply absent): surface the
//     original `ExternalValueError::MetaNotFound` / `DataNotFound` (gRPC
//     `Code::Internal` — the codebase's `ServiceError::Internal` →
//     `Status::internal("internal error")` mapping redacts the message
//     body to a fixed `"internal error"` string per `src/error.rs:206`;
//     the wire `code()` carries the classification, the metric label
//     `adapter_miss` carries the diagnostic distinction).
//   - Adapter returns `Err(CheckpointError::Transient)`: surface gRPC
//     `Code::Unavailable` (retriable per SDD §18.1). The
//     `ServiceError::Unavailable` mapping in `src/error.rs:203`
//     preserves the message body so callers see the underlying
//     transient-failure description.
//   - Adapter returns `Err(CheckpointError::Permanent)`: surface gRPC
//     `Code::Internal` (NOT retriable). Same `Status::internal("internal error")`
//     wire shape as the `Ok(None)` case; the `adapter_permanent` metric
//     label distinguishes the two on the operator side.
//
// The `ServiceError::Internal` → `Status::internal("internal error")`
// redaction is the load-bearing wire-shape contract: it intentionally
// strips the underlying `ExternalValueError::MetaNotFound`'s hash hex
// from the gRPC message so a buggy or malicious caller cannot
// fingerprint stored-blob hashes via probe-and-observe. Tests pin both
// halves of the contract (`Code::Internal` + `"internal error"`)
// because either half alone is a regression — flipping the code to
// `UNAVAILABLE` would trigger client retry storms, and dropping the
// redaction would leak operator-only information to the client.
// ════════════════════════════════════════════════════════════════════════════

/// Drain the subscribe stream until the spawned task surfaces an
/// `Err(Status)` and return it. The bootstrap path of
/// `handle_subscribe_read_layer` returns `Ok(Response::new(stream))`
/// immediately (it only allocates the bridge channel); any
/// `compute_diff_at_cut` failure surfaces inside the spawned `run_subscription`
/// task and arrives on the stream as `Err(Status)`. A subscribe that
/// hits the SR-3.2 miss-path on a layer whose ext-value bytes are
/// gone from BOTH Valkey and the adapter never delivers a bootstrap
/// header — so we drain until either an `Err(Status)` arrives, the
/// stream closes cleanly (regression — should not happen on the error
/// paths this test exercises), or the bounded timeout fires.
///
/// 10 s timeout absorbs CI scheduling jitter; a healthy error path
/// surfaces the status sub-millisecond in-process. Aligns with the
/// happy-path [`recv_bootstrap_message`] timeout budget.
async fn recv_subscribe_error_status<S>(stream: &mut S) -> Status
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    // Single-step receive — every match arm is terminal so this is not
    // a polling loop. The catch-up failure path in `run_subscription`
    // sends `Err(status)` and then drops the mpsc sender, so the very
    // next `stream.next()` carries the classified Status. If the
    // implementation ever changes to interleave Ok messages before the
    // error (which would itself be a regression — the SR-3.2 miss-path
    // currently surfaces before any bootstrap header is sent), this
    // assertion converts that into a clean test failure.
    let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("subscribe stream timed out waiting for error Status");
    match next {
        Some(Err(status)) => status,
        Some(Ok(msg)) => panic!(
            "subscribe stream yielded an Ok response while we were waiting for an \
             error Status — the SR-3.2 miss-path is not surfacing the classified \
             ExternalValueError; got: {msg:?}"
        ),
        None => panic!(
            "subscribe stream closed cleanly while we were waiting for an error \
             Status — the SR-3.2 miss-path must surface the classified \
             ExternalValueError, not silently drain"
        ),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_surfaces_meta_not_found_when_adapter_lacks_ext_value() {
    // Same LocalDisk-backed harness shape as the happy-path test — the
    // adapter wiring is identical (real `ValkeyExternalValueStore` with
    // `with_checkpoint_store(...)`), only the pre-seed step is omitted
    // so the adapter is empty for our hash.
    let harness = setup_ext_value_recovery_harness(1).await;

    // Snapshot the metrics BEFORE traffic so we can assert the exact
    // post-subscribe deltas. The metric is process-wide (no service
    // label, by design — operator alerts care about the aggregate rate,
    // see SDD SR-3.2), so the absolute values may be non-zero from
    // sibling tests running in the same binary; the delta is the
    // load-bearing signal.
    let pre_recovered = ext_recovery_count(&harness.metrics, "recovered");
    let pre_adapter_miss = ext_recovery_count(&harness.metrics, "adapter_miss");
    let pre_adapter_transient = ext_recovery_count(&harness.metrics, "adapter_transient");
    let pre_adapter_permanent = ext_recovery_count(&harness.metrics, "adapter_permanent");

    // ── 1. Standard create-session + create-layer with one ext-value field ─
    const SESSION_NAME: &str = "ext-value-recovery-adapter-missing-session";
    const LAYER_NAME: &str = "ext-value-recovery-adapter-missing-layer";
    let session_id = harness
        .svc
        .session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let payload: &[u8] = b"sr-3.2-adapter-missing-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);
    let pair: [(&[u8; 32], &[u8]); 1] = [(&hash, payload)];
    harness
        .svc
        .ext_value_store
        .store_values(&pair)
        .await
        .expect("initial ext-value store_values must succeed");

    let delta: OwnedDelta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "ext-field",
        /*hash=*/ hash.to_vec(),
        /*ext_value_size=*/ payload.len() as u64,
        /*ext_value_index=*/ 1,
        /*set_order=*/ 1,
    );
    let create_ompf = delta_to_ompf_bytes(&delta, &[payload]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with ext-value must succeed");
    let object_id = create_resp.into_inner().object_id;

    // ── 2. Guarantee the adapter is empty for our hash ────────────────────
    //
    // The Subtask body's outline step 2: "DO NOT pre-seed the adapter;
    // if Task A's mirror has landed it, call
    // `harness.store.delete_ext_blob(&sha256_to_hex(&hash))` to guarantee
    // adapter is empty." The harness's periodic checkpoint loop runs at
    // `interval=0`/`quiesce=0` and could in principle race with us on
    // landing a generation+sidecar; the explicit `delete_ext_blob` makes
    // the adapter-empty invariant non-racy. `delete_ext_blob` is
    // idempotent on missing per the trait contract, so the call is safe
    // regardless of whether the mirror has populated the adapter.
    harness
        .store
        .delete_ext_blob(&hash_hex)
        .await
        .expect("delete_ext_blob (idempotent on missing) must succeed");

    // Sanity probe: the adapter is empty for our hash.
    let pre_subscribe_adapter_blob = harness
        .store
        .get_ext_blob(&hash_hex)
        .await
        .expect("get_ext_blob pre-subscribe probe must succeed");
    assert!(
        pre_subscribe_adapter_blob.is_none(),
        "pre-subscribe: adapter must be empty for hash {hash_hex}, got Some(_)"
    );

    // ── 3. Wipe ONLY the ext namespace ────────────────────────────────────
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&hash_hex)).await;

    // ── 4. SubscribeReadLayer must surface Code::Internal ─────────────────
    //
    // The bootstrap path runs `compute_diff_at_cut` → `read_batch` →
    // `try_recover_via_checkpoint`; the adapter returns `Ok(None)`
    // because the blob was never put. The recovery path ticks
    // `adapter_miss` and re-surfaces the original
    // `ExternalValueError::MetaNotFound` (or `DataNotFound` — the wipe
    // removed both meta and data keys; whichever surfaces first depends
    // on the read-side probe order in `ValkeyExternalValueStore::read_batch`).
    // Both classify as `ServiceError::Internal` per
    // `From<ExternalValueError> for ServiceError`, which the
    // `ServiceError → Status` mapping deliberately redacts to the
    // fixed `"internal error"` message body so backend identifiers
    // (the hash hex) never reach the client.
    let sub_req = subscribe_request(&session_id, LAYER_NAME, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("handle_subscribe_read_layer must return Ok — error surfaces on the stream");
    let mut stream = response.into_inner();
    let status = recv_subscribe_error_status(&mut stream).await;

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "SR-3.2 adapter-empty miss-path must surface Code::Internal (matches the \
         redacted ExternalValueError::MetaNotFound classification); got code={:?}, \
         message={:?}",
        status.code(),
        status.message()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "ServiceError::Internal → Status::internal mapping deliberately redacts the \
         message body to the fixed string 'internal error' (src/error.rs line ~206) \
         so backend identifiers (the hash hex) never leak to the client; got: {:?}",
        status.message()
    );

    // ── 5. Valkey ext keys remain absent (no silent zero-fill) ────────────
    //
    // Subtask body acceptance criterion: "Valkey ext keys remain
    // absent." The SR-3.2 path's only Valkey rewrite is the
    // `store_recovered_values` call on the recovered-bytes arm; on the
    // `adapter_miss` arm no rewrite fires, so the wiped keys MUST stay
    // absent (no panic recovery, no silent zero-fill).
    let meta_key = harness.tv.key_schema().ext_meta(&hash_hex);
    let data_key = harness.tv.key_schema().ext_data(&hash_hex);
    let meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&meta_key)
        .await
        .expect("EXISTS ext_meta post-miss probe must succeed");
    let data_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&data_key)
        .await
        .expect("EXISTS ext_data post-miss probe must succeed");
    assert!(
        !meta_exists,
        "post-miss: ext_meta:{hash_hex} must remain absent — the SR-3.2 adapter_miss \
         path MUST NOT silently rewrite Valkey"
    );
    assert!(
        !data_exists,
        "post-miss: ext_data:{hash_hex} must remain absent — the SR-3.2 adapter_miss \
         path MUST NOT silently rewrite Valkey"
    );

    // ── 6. Metric deltas pin the classification ───────────────────────────
    //
    // `adapter_miss` ticked by exactly 1 (our one missing hash).
    // `recovered` / `adapter_transient` / `adapter_permanent` MUST NOT
    // tick — the classification surface is mutually exclusive.
    let post_adapter_miss = ext_recovery_count(&harness.metrics, "adapter_miss");
    assert_eq!(
        post_adapter_miss.saturating_sub(pre_adapter_miss),
        1,
        "adapter_miss must tick by exactly 1 (one missing hash); observed delta {}, \
         pre={pre_adapter_miss}, post={post_adapter_miss}",
        post_adapter_miss.saturating_sub(pre_adapter_miss),
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "recovered"),
        pre_recovered,
        "no 'recovered' ticks expected — the adapter is empty for this hash"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_transient"),
        pre_adapter_transient,
        "no 'adapter_transient' ticks expected — LocalDisk store is healthy"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_permanent"),
        pre_adapter_permanent,
        "no 'adapter_permanent' ticks expected — LocalDisk store is healthy"
    );

    // ── 7. Drain the harness ──────────────────────────────────────────────
    drop(stream);
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ────────────────────────────────────────────────────────────────────────────
// InMemory + fault-injector harness for the Transient / Permanent variants.
//
// The shared `setup_ext_value_recovery_harness` wires
// `LocalDiskCheckpointStore`, which does NOT consult a
// `CheckpointFaultInjector` in its `with_fault_injector` form unless the
// `test-checkpoint-fault-injection` feature is enabled AND the harness
// constructs it through that builder. Rather than re-fitting the shared
// harness, we build a parallel local helper that mirrors its shape
// exactly but swaps the adapter for `InMemoryCheckpointStore::with_fault_injector(injector)`.
// This is the same parallel-helper pattern Sibling Subtask `p6.co`
// documented in this file's module-level comment as the right escape
// hatch for "the LocalDisk harness can't accept a fault injector".
//
// Gated on the `test-checkpoint-fault-injection` feature because the
// `with_fault_injector` builder is itself gated on that feature — the
// trait + builder are compiled out of production library builds, so
// without the feature the helper would not compile.
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::{CheckpointFaultInjector, InMemoryCheckpointStore};
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::CheckpointError;

/// Op-targeted fault injector — fires the supplied factory-built error
/// only when the calling adapter method's `op` label equals
/// `target_op`. Mirrors the `OpTargetedFaultInjector` in
/// `tests/integration/checkpoint/conformance.rs` (defined locally so
/// this file's contract stays self-contained and we do not import an
/// otherwise-private harness type across module boundaries).
///
/// `error_factory` is a closure rather than a stored `CheckpointError`
/// because `Transient` / `Permanent` carry a boxed `Send + Sync` source
/// that is not `Clone`; the closure rebuilds a fresh error each time
/// `maybe_inject` fires.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct OpTargetedFaultInjector {
    target_op: &'static str,
    error_factory: Box<dyn Fn() -> CheckpointError + Send + Sync>,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl OpTargetedFaultInjector {
    fn new<EF>(target_op: &'static str, error_factory: EF) -> Self
    where
        EF: Fn() -> CheckpointError + Send + Sync + 'static,
    {
        Self {
            target_op,
            error_factory: Box::new(error_factory),
        }
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl CheckpointFaultInjector for OpTargetedFaultInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == self.target_op {
            Err((self.error_factory)())
        } else {
            Ok(())
        }
    }
}

/// Synthetic backend error stuffed into the boxed `source` field on
/// injected `Transient` / `Permanent` errors. Carries a stable string
/// so the `Display` output is observable in test-failure diagnostics
/// (the `Status::unavailable` message body preserves it on the
/// Transient arm).
#[cfg(feature = "test-checkpoint-fault-injection")]
#[derive(Debug)]
struct SyntheticBackendError(&'static str);

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::fmt::Display for SyntheticBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "synthetic backend error: {}", self.0)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::error::Error for SyntheticBackendError {}

/// Construct a fault-injectable ext-value-recovery E2E harness backed
/// by `InMemoryCheckpointStore` instead of `LocalDiskCheckpointStore`.
///
/// Wiring shape mirrors [`setup_ext_value_recovery_harness`] exactly
/// (same real `ValkeyExternalValueStore` with `.with_checkpoint_store(...)`,
/// same enabled `RecoveryHook` / `SessionRecoveryHook`, same
/// `interval=0`/`quiesce=0` periodic loop) so the helpers
/// ([`wipe_ext_namespace_for_hashes`], [`recv_bootstrap_message`],
/// [`ext_recovery_count`], etc.) all stay reusable verbatim. The only
/// substantive difference is the adapter type and the op-targeted
/// fault injector installed via `InMemoryCheckpointStore::with_fault_injector`.
///
/// Returns a harness wrapping the InMemory store as
/// `Arc<dyn CheckpointStore>` so tests can call trait-surface methods
/// (`put_ext_blob`, `get_ext_blob`) via the same `harness.store` field
/// as the LocalDisk variant.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn setup_ext_value_recovery_harness_in_memory(
    injector: Arc<dyn CheckpointFaultInjector>,
) -> ExtValueRecoveryHarness {
    let tv = TestValkey::setup().await;

    // Tempdir is irrelevant for the InMemory store (no disk persistence)
    // but the `ExtValueRecoveryHarness` struct's `tempdir` field is
    // load-bearing for the LocalDisk harness; allocate a fresh empty
    // tempdir so the same struct shape survives.
    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> =
        Arc::new(InMemoryCheckpointStore::new().with_fault_injector(injector));

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let ext_value_store: Arc<dyn ExternalValueStore> = Arc::new(
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(&metrics))
            .with_checkpoint_store(Arc::clone(&store)),
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

    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    // `LIVE_EDIT_CHECKPOINT_ADAPTER=InMemory` matches the wiring shape —
    // a downstream config drift surfaces as a compile error rather than
    // a silent default flip.
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::InMemory;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = 1;
    cfg.checkpoint_retain_generations = 2;
    let cfg_arc = Arc::new(cfg);

    let recovery = Arc::new(RecoveryHook::new(
        Arc::clone(&store),
        Arc::clone(&layer_store),
        Arc::clone(&cfg_arc),
        Arc::clone(&metrics),
    ));
    let session_recovery = Arc::new(SessionRecoveryHook::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&cfg_arc),
        Arc::clone(&metrics),
    ));

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
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        ),
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

    // Periodic checkpoint loop — harmless to the SR-3.2 error-path
    // tests (we drive the adapter directly via `harness.store`), but
    // we still spawn it to match the production wiring shape and
    // exercise the same shutdown ceremony.
    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_secs(1),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(SERVICE_LABEL),
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

    ExtValueRecoveryHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        tempdir,
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_surfaces_unavailable_when_adapter_get_ext_blob_returns_transient() {
    // ── 1. Op-targeted Transient injector for `get_ext_blob` only ─────────
    //
    // Targeting ONLY the `"get_ext_blob"` op lets the pre-seed
    // `put_ext_blob` below run unimpeded — without that scoping the
    // pre-seed itself would fail and the test would observe the
    // wrong-side classification (a put-side Transient leaks through a
    // different gRPC path entirely).
    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(OpTargetedFaultInjector::new("get_ext_blob", || {
            CheckpointError::Transient {
                source: Box::new(SyntheticBackendError("injected transient on get_ext_blob")),
            }
        }));
    let harness = setup_ext_value_recovery_harness_in_memory(Arc::clone(&injector)).await;

    let pre_adapter_transient = ext_recovery_count(&harness.metrics, "adapter_transient");
    let pre_adapter_miss = ext_recovery_count(&harness.metrics, "adapter_miss");
    let pre_adapter_permanent = ext_recovery_count(&harness.metrics, "adapter_permanent");
    let pre_recovered = ext_recovery_count(&harness.metrics, "recovered");

    // ── 2. CreateSession + ext-store + CreateLayer ────────────────────────
    const SESSION_NAME: &str = "ext-value-recovery-transient-session";
    const LAYER_NAME: &str = "ext-value-recovery-transient-layer";
    let session_id = harness
        .svc
        .session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let payload: &[u8] = b"sr-3.2-transient-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);
    let pair: [(&[u8; 32], &[u8]); 1] = [(&hash, payload)];
    harness
        .svc
        .ext_value_store
        .store_values(&pair)
        .await
        .expect("initial ext-value store_values must succeed");

    let delta: OwnedDelta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "ext-field",
        /*hash=*/ hash.to_vec(),
        /*ext_value_size=*/ payload.len() as u64,
        /*ext_value_index=*/ 1,
        /*set_order=*/ 1,
    );
    let create_ompf = delta_to_ompf_bytes(&delta, &[payload]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with ext-value must succeed");
    let object_id = create_resp.into_inner().object_id;

    // ── 3. Pre-seed the adapter ───────────────────────────────────────────
    //
    // The Subtask body's outline step 2: "Pre-seed the InMemory adapter
    // with the blob via `put_ext_blob`, then install the fault injector."
    // The injector is op-targeted on `"get_ext_blob"`, so this put
    // bypasses the injection seam (op label is `"put_ext_blob"`) and
    // lands the bytes. The injector then fires on the subsequent
    // recovery `get_ext_blob` — exercising the
    // `CheckpointError::Transient` → `ServiceError::Unavailable` →
    // `Status::unavailable(msg)` mapping with a payload that exists in
    // the adapter (so the test pins "the injection short-circuits
    // before the underlying store is consulted", not "the underlying
    // store is empty").
    harness
        .store
        .put_ext_blob(&hash_hex, Bytes::copy_from_slice(payload))
        .await
        .expect("pre-seed put_ext_blob must succeed (injector is op-targeted on get)");

    // ── 4. Wipe ONLY the ext namespace ────────────────────────────────────
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&hash_hex)).await;

    // ── 5. SubscribeReadLayer must surface Code::Unavailable ──────────────
    let sub_req = subscribe_request(&session_id, LAYER_NAME, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("handle_subscribe_read_layer must return Ok — error surfaces on the stream");
    let mut stream = response.into_inner();
    let status = recv_subscribe_error_status(&mut stream).await;

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "SR-3.2 adapter-Transient miss-path must surface Code::Unavailable (retriable \
         per SDD §18.1 — Code::Internal would prevent client retry); got code={:?}, \
         message={:?}",
        status.code(),
        status.message()
    );
    // `ServiceError::Unavailable(msg)` → `Status::unavailable(msg)`
    // preserves the message body (operator-facing, not security-sensitive)
    // per `src/error.rs:203` and the existing test in `src/error.rs:585`.
    // The recovery code in `try_recover_via_checkpoint` formats the
    // message as `"checkpoint adapter transient failure during
    // ext-value recovery: <source>"`. We assert the substring
    // `"checkpoint adapter transient failure"` because the surrounding
    // source-display format is implementation detail; the
    // "transient failure" descriptor is the load-bearing contract for
    // operators reading client-side error logs.
    assert!(
        status
            .message()
            .contains("checkpoint adapter transient failure"),
        "Unavailable message must preserve the 'checkpoint adapter transient \
         failure' descriptor (src/storage/external_values.rs line ~550); got: {:?}",
        status.message()
    );

    // ── 6. Metric deltas pin the classification ───────────────────────────
    let post_adapter_transient = ext_recovery_count(&harness.metrics, "adapter_transient");
    assert_eq!(
        post_adapter_transient.saturating_sub(pre_adapter_transient),
        1,
        "adapter_transient must tick by exactly 1; observed delta {}, \
         pre={pre_adapter_transient}, post={post_adapter_transient}",
        post_adapter_transient.saturating_sub(pre_adapter_transient),
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "recovered"),
        pre_recovered,
        "no 'recovered' ticks expected — the injector short-circuited the get"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_miss"),
        pre_adapter_miss,
        "no 'adapter_miss' ticks expected — the injector returns Err, not Ok(None)"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_permanent"),
        pre_adapter_permanent,
        "no 'adapter_permanent' ticks expected — the injection variant is Transient"
    );

    // ── 7. Drain the harness ──────────────────────────────────────────────
    drop(stream);
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_surfaces_meta_not_found_when_adapter_get_ext_blob_returns_permanent()
{
    // ── 1. Op-targeted Permanent injector for `get_ext_blob` only ─────────
    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(OpTargetedFaultInjector::new("get_ext_blob", || {
            CheckpointError::Permanent {
                source: Box::new(SyntheticBackendError("injected permanent on get_ext_blob")),
            }
        }));
    let harness = setup_ext_value_recovery_harness_in_memory(Arc::clone(&injector)).await;

    let pre_adapter_permanent = ext_recovery_count(&harness.metrics, "adapter_permanent");
    let pre_adapter_miss = ext_recovery_count(&harness.metrics, "adapter_miss");
    let pre_adapter_transient = ext_recovery_count(&harness.metrics, "adapter_transient");
    let pre_recovered = ext_recovery_count(&harness.metrics, "recovered");

    // ── 2. CreateSession + ext-store + CreateLayer ────────────────────────
    const SESSION_NAME: &str = "ext-value-recovery-permanent-session";
    const LAYER_NAME: &str = "ext-value-recovery-permanent-layer";
    let session_id = harness
        .svc
        .session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let payload: &[u8] = b"sr-3.2-permanent-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);
    let pair: [(&[u8; 32], &[u8]); 1] = [(&hash, payload)];
    harness
        .svc
        .ext_value_store
        .store_values(&pair)
        .await
        .expect("initial ext-value store_values must succeed");

    let delta: OwnedDelta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "ext-field",
        /*hash=*/ hash.to_vec(),
        /*ext_value_size=*/ payload.len() as u64,
        /*ext_value_index=*/ 1,
        /*set_order=*/ 1,
    );
    let create_ompf = delta_to_ompf_bytes(&delta, &[payload]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with ext-value must succeed");
    let object_id = create_resp.into_inner().object_id;

    // ── 3. Pre-seed the adapter (op-targeted injector won't fire on put) ──
    harness
        .store
        .put_ext_blob(&hash_hex, Bytes::copy_from_slice(payload))
        .await
        .expect("pre-seed put_ext_blob must succeed (injector is op-targeted on get)");

    // ── 4. Wipe ONLY the ext namespace ────────────────────────────────────
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&hash_hex)).await;

    // ── 5. SubscribeReadLayer must surface Code::Internal ─────────────────
    //
    // `CheckpointError::Permanent` routes through
    // `try_recover_via_checkpoint`'s permanent arm, which preserves the
    // original `ExternalValueError::MetaNotFound` and surfaces it as
    // `ServiceError::Internal` — same redaction-to-"internal error"
    // wire shape as the adapter-empty case, but distinguished from it
    // on the operator side by the `adapter_permanent` metric label
    // (not `adapter_miss`).
    let sub_req = subscribe_request(&session_id, LAYER_NAME, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("handle_subscribe_read_layer must return Ok — error surfaces on the stream");
    let mut stream = response.into_inner();
    let status = recv_subscribe_error_status(&mut stream).await;

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "SR-3.2 adapter-Permanent miss-path must surface Code::Internal (NOT retriable \
         per SDD §18.1 — Code::Unavailable would let clients retry a permanent failure \
         and waste resources); got code={:?}, message={:?}",
        status.code(),
        status.message()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "ServiceError::Internal → Status::internal mapping deliberately redacts the \
         message body to 'internal error'; the underlying \
         ExternalValueError::MetaNotFound 'external value meta not found for hash <hex>' \
         shape is preserved in the ServiceError variant but stripped from the wire \
         status to avoid leaking backend identifiers; got: {:?}",
        status.message()
    );

    // ── 6. Metric deltas pin the classification ───────────────────────────
    //
    // The `adapter_permanent` label is the operator-side distinction
    // from `adapter_miss` — both surface as Code::Internal +
    // "internal error" on the wire, but the metric carries the
    // diagnostic information the wire status redacts.
    let post_adapter_permanent = ext_recovery_count(&harness.metrics, "adapter_permanent");
    assert_eq!(
        post_adapter_permanent.saturating_sub(pre_adapter_permanent),
        1,
        "adapter_permanent must tick by exactly 1; observed delta {}, \
         pre={pre_adapter_permanent}, post={post_adapter_permanent}",
        post_adapter_permanent.saturating_sub(pre_adapter_permanent),
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "recovered"),
        pre_recovered,
        "no 'recovered' ticks expected — the injector short-circuited the get"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_miss"),
        pre_adapter_miss,
        "no 'adapter_miss' ticks expected — the injector returns Err, not Ok(None)"
    );
    assert_eq!(
        ext_recovery_count(&harness.metrics, "adapter_transient"),
        pre_adapter_transient,
        "no 'adapter_transient' ticks expected — the injection variant is Permanent"
    );

    // ── 7. Drain the harness ──────────────────────────────────────────────
    drop(stream);
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}
