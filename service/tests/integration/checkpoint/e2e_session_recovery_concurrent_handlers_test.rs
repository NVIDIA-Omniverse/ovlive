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

//! End-to-end concurrent-handler session-recovery test (Subtask
//! `t3.fgj.7v.fp.8n` under Task `t2.fgj.7v.fp`, Epic `t1.fgj.7v`, Bee
//! `b.fgj`).
//!
//! ## What this test pins
//!
//! When N concurrent RPCs against the SAME `session_id` simultaneously hit
//! the session-existence check after a Valkey-side session wipe, the
//! per-`session_id` K=1 admission discipline inside
//! [`SessionRecoveryHook`](live_edit::checkpoint::SessionRecoveryHook) MUST
//! collapse the burst to exactly ONE adapter `get_session_checkpoint` call.
//! The other N−1 callers must short-circuit at the post-acquire EXISTS
//! re-check inside `try_recover_session` (step 4) and observe
//! [`SessionRecoveryOutcome::AlreadyPresent`](live_edit::checkpoint::SessionRecoveryOutcome::AlreadyPresent)
//! without re-issuing the adapter fetch.
//!
//! ## Coverage gap addressed
//!
//! Task 6's inline `#[cfg(test)]` tests verify the per-`session_id` K=1
//! deduplication by direct-mock-counted calls to `try_recover_session`.
//! Task 7's AC #6 exercises the same path via direct hook calls. Neither
//! composes the admission gate with the real RPC dispatch path — auth
//! middleware, [`SessionBasePathCache::get_base_path`], the cache's
//! `recovered_once` latch, [`SessionStore::get_base_path`], the wiring
//! through `LiveEditService`.
//!
//! This test closes that gap by spawning N=8 concurrent
//! [`handle_update_layer`] calls against the same `session_id`, gated on a
//! [`tokio::sync::Barrier`] so admission is genuinely contended. The
//! `CountingCheckpointStore` wrapper records `get_session_checkpoint`
//! invocations on the underlying adapter; after the burst, the counter must
//! read EXACTLY 1.
//!
//! ## Wiring deltas vs. the LocalDisk harness
//!
//! [`setup_checkpoint_harness`] wires `LayerStore` / `SessionStore` /
//! `RecoveryHook` / `SessionRecoveryHook` against the unwrapped
//! `Arc<dyn CheckpointStore>`. We need the SAME store shape but wrapped
//! with a counter on the recovery path. To get that we replicate the
//! harness's wiring locally with a `CountingCheckpointStore` over a
//! [`LocalDiskCheckpointStore`] and feed the WRAPPED `Arc<dyn CheckpointStore>`
//! to every consumer — writer, session-recovery hook, and the loop. The
//! counter therefore captures every adapter call (PUTs from the writer,
//! GETs from the recovery hook, etc.), and we reset it to zero AFTER the
//! pre-wipe setup completes so the post-wipe assertion measures only
//! recovery-driven `get_session_checkpoint` calls.
//!
//! ## Cache TTL trap
//!
//! [`SessionBasePathCache`] has a 2-second positive TTL. The baseline
//! `handle_create_layer` populates the cache, so without explicit
//! invalidation every post-wipe handler would HIT the cache, return the
//! cached `base_path`, and never consult `session_store.get_base_path` —
//! the recovery branch would never fire and the test would silently pass
//! with `restored == 0`. We call [`invalidate_session_cache`] AFTER the
//! wipe so every spawned RPC sees a cache MISS and exercises the cold
//! path through the cache funnel.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::KeysInterface;
use tempfile::TempDir;
use tokio::sync::Barrier;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, RecoveryHook,
    SessionCheckpointWriter, SessionRecoveryHook, TailCaptureConfig,
};
use live_edit::config::Config;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::CreateSessionRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::handle_create_session;
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};
use crate::grpc::helpers::{AllowAllPermissions, StubUserinfo};

const SESSION_NAME: &str = "e2e-session-recovery-concurrent-session";
const SESSION_BASE_PATH: &str = "/stage";
const LAYER_NAME: &str = "concurrent-original-layer";
const CLIENT_ID: &str = "concurrent-update-client";
const CONCURRENCY: usize = 8;

// =============================================================================
// CountingCheckpointStore is the shared `CountingGetSessionCheckpointStore`
// imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 2 deduplication). Aliased to the
// local name so the harness's `counting:` field type stays stable.
// =============================================================================

type CountingCheckpointStore = CountingGetSessionCheckpointStore;

// =============================================================================
// Local harness — mirrors `setup_checkpoint_harness` from helpers.rs but
// substitutes `CountingCheckpointStore` over `LocalDiskCheckpointStore` so the
// counter is wired across every consumer (writer, session-recovery hook, loop).
// =============================================================================

/// All the pieces a concurrent-handler recovery test needs to drive the
/// write → checkpoint → wipe → recover cycle and inspect the adapter-call
/// surface. Owns both the wrapper handle (so the test can read the counter)
/// and the standard `LiveEditService` + harness pieces.
struct CountingHarness {
    tv: TestValkey,
    counting: Arc<CountingCheckpointStore>,
    svc: LiveEditService,
    metrics: SharedMetrics,
    session_recovery: Arc<SessionRecoveryHook>,
    loop_handle: JoinHandle<()>,
    cancel: CancellationToken,
    /// `TaskTracker` shared with the harness's `SessionCheckpointWriter`.
    /// Tests pass it to [`wait_for_writer_idle`] to drain the fire-and-forget
    /// session-checkpoint writes (CreateSession enqueues blob v1, CreateLayer
    /// enqueues blob v2 carrying the layer mapping) before wiping state —
    /// otherwise a cold-cache run can wipe + recover off v1 (no mapping) and
    /// every concurrent UpdateLayer hits NotFound. Mirrors the sibling
    /// `e2e_session_recovery_admission_test` harness.
    session_checkpoint_tracker: TaskTracker,
    /// LocalDisk adapter root, kept alive for the duration of the test.
    /// RAII-cleaned on drop. Never accessed by the test directly; the
    /// counting store holds a `PathBuf` rooted under this directory.
    #[allow(dead_code)]
    tempdir: TempDir,
}

/// Build a `Config` shaped like [`helpers::setup_checkpoint_harness`]'s
/// `harness_config` — same `checkpoint_enabled=true`, `LocalDisk` adapter,
/// `interval=0` / `quiesce=0`, etc. Duplicated locally (vs. re-using the
/// helpers' private fn) because we are intentionally building a parallel
/// harness so we can substitute the store.
fn harness_config(tick_secs: u64, local_disk_path: String) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::LocalDisk;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = tick_secs.max(1);
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_local_disk_path = Some(local_disk_path);
    cfg
}

/// Construct the counting harness. Mirrors `setup_checkpoint_harness` from
/// `helpers.rs` field-for-field, with the single substitution of a
/// `CountingCheckpointStore` wrapping the `LocalDiskCheckpointStore`. The
/// `Arc<dyn CheckpointStore>` exposed to every consumer is the WRAPPED
/// store, so every adapter call is observable through the counter.
async fn setup_counting_harness(tick_secs: u64) -> CountingHarness {
    let tv = TestValkey::setup().await;
    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");

    let inner_store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );
    let counting = Arc::new(CountingCheckpointStore::new(inner_store));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    // Session-checkpoint writer wiring — required so CreateSession +
    // CreateLayer enqueue a session-checkpoint blob the recovery hook can
    // later restore from. Same shape as `setup_checkpoint_harness`.
    let session_checkpoint_tracker = TaskTracker::new();
    let session_checkpoint_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        session_checkpoint_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
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

    let recovery = Arc::new(RecoveryHook::new(
        Arc::clone(&store),
        Arc::clone(&layer_store),
        Arc::new(harness_config(
            tick_secs,
            tempdir.path().to_string_lossy().into_owned(),
        )),
        Arc::clone(&metrics),
    ));
    let session_recovery = Arc::new(SessionRecoveryHook::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(harness_config(
            tick_secs,
            tempdir.path().to_string_lossy().into_owned(),
        )),
        Arc::clone(&metrics),
    ));

    // Wire the session-recovery hook into the cache, matching production.
    let session_base_path_cache =
        SessionBasePathCache::new().with_recovery_hook(Arc::clone(&session_recovery));

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store: Arc::clone(&layer_store),
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
        presence_cleanup_tracker: TaskTracker::new(),
        subscription_tracker: TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics: Arc::clone(&metrics),
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache,
        recovery,
        session_recovery: Arc::clone(&session_recovery),
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

    CountingHarness {
        tv,
        counting,
        svc,
        metrics,
        session_recovery,
        loop_handle,
        cancel,
        session_checkpoint_tracker,
        tempdir,
    }
}

// =============================================================================
// gRPC framing + token helpers — imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication).
// =============================================================================

/// Probe the live label-cell counter value for a given outcome label on the
/// `live_edit_checkpoint_recovery_attempts_total{result, scope="session",
/// service}` metric. `scope="session"` is load-bearing — the per-handler
/// E2E pinning works on the session-recovery cells, not the layer-recovery
/// cells (which fire under a separate `RecoveryHook` for `object_id`-keyed
/// recovery).
fn session_recovery_metric(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "session"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Drop the cached `base_path` entry for `session_id` on the harness's
/// `SessionBasePathCache`. Load-bearing: the baseline CreateLayer call
/// populates the cache with the live `base_path`, and the cache's 2-second
/// positive TTL would let the post-wipe handler dispatch HIT the cache and
/// never consult `session_store.get_base_path` — bypassing the recovery
/// branch and silently passing the test with `restored == 0`. We invalidate
/// after the wipe so every concurrent RPC sees a cache MISS and exercises
/// the cold path through the cache funnel.
fn invalidate_session_cache(svc: &LiveEditService, session_id: &str) {
    svc.session_base_path_cache.invalidate(session_id);
}

/// Build a small incremental UpdateLayer delta carrying ONE `SetField`. Each
/// concurrent task uses its own `set_order` / `marker` pair so the
/// post-recovery `update_layer` writes are independently observable; the
/// test does not assert on those writes' contents (the focus is the
/// admission collapse), but distinct payloads keep the writes from
/// short-circuiting any future write-deduplication change.
///
/// `base_version: 1` (any non-zero) marks the delta as an INCREMENTAL diff
/// — `base_version == 0 && is_diff == true` would take the reset-delta
/// branch and wipe sections/fields, which is irrelevant noise for the
/// recovery-collapse signal. Mirrors the `delta_with_field_update` shape
/// in `e2e_update_layer_recovery_test.rs`.
fn make_update_delta(set_order: u64, marker: u8) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: 0,
            key_name: format!("concurrent-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

// =============================================================================
// Test body.
// =============================================================================

/// Spawn N=8 concurrent `UpdateLayer` RPCs against the same `session_id`
/// after a Valkey-side session wipe; assert the per-`session_id` K=1
/// admission discipline inside `SessionRecoveryHook` collapses the burst
/// to EXACTLY ONE adapter `get_session_checkpoint` call.
///
/// Why each step is load-bearing:
///
/// - `setup_counting_harness(1)` wires a `CountingCheckpointStore` over a
///   `LocalDiskCheckpointStore`. The wrapped store is threaded through
///   every consumer (writer, session-recovery hook, periodic loop) so the
///   counter captures every adapter call. The session-recovery hook is
///   wired into the `SessionBasePathCache` via `with_recovery_hook`,
///   matching the production `main.rs` wiring exactly.
/// - `CreateSession` + `CreateLayer` go through the real RPC entry points.
///   Both lifecycle hooks fire fire-and-forget session-checkpoint writes
///   through the shared dispatcher. `wait_for_session_checkpoint` blocks
///   until the dispatcher's spawned put-task has landed the blob on the
///   `LocalDiskCheckpointStore`.
/// - `counting.reset_count()` is called AFTER the checkpoint has landed
///   and BEFORE the concurrent dispatch. The reset is necessary because
///   `wait_for_session_checkpoint` itself polls `get_session_checkpoint`
///   to detect blob presence — without the reset, the counter would
///   include those polling calls and the exact-1 assertion would fail.
/// - `wipe_session_state(session_id)` UNLINKs every `le:{sid}*` key —
///   session-meta + layer-mapping + layers-index. Path-index left intact
///   (SR-2.7 asymmetric self-healing).
/// - `invalidate_session_cache` drops the cache entry the baseline
///   CreateLayer populated. Without this every spawned RPC would HIT the
///   cache and never consult `session_store.get_base_path` — the recovery
///   branch would never fire.
/// - `Barrier::new(CONCURRENCY)` releases every spawned task at the same
///   instant, so the contention window the K=1 mutex is supposed to
///   collapse is genuinely exercised. Without the barrier the first
///   spawned task could finish its entire admission-acquire +
///   apply_session_checkpoint pipeline before any later task entered
///   `try_recover_session` — letting the post-acquire EXISTS re-check on
///   the followers fire on a no-contention path and vacuously passing
///   the test.
///
/// Assertions:
///
/// - Every spawned RPC returns Ok — none crash, none surface a spurious
///   NotFound from the wiring race.
/// - `counting.get_session_calls()` is EXACTLY 1. The K=1 mutex
///   serializes the bursts; the winner runs the apply pipeline, the
///   losers short-circuit at the post-acquire EXISTS re-check (step 4)
///   BEFORE step 5's `get_session_checkpoint` fetch.
/// - `session_recovery_metric(restored)` is EXACTLY 1 — the winner's
///   apply landed.
/// - `session_recovery_metric(already_present)` is EXACTLY N−1 — the
///   losers' post-acquire EXISTS re-check ticked the `already_present`
///   cell. (Note: the recovery cache's `recovered_once` latch caps
///   per-RPC recovery cost at 1; a single recovery cycle per RPC fires.)
/// - Other recovery cells (`error`, `not_recoverable`) stay at zero.
/// - `session_recovery.admission_len()` drains to 0 after every spawned
///   RPC drops out. A non-zero residual would signal an
///   `AdmissionMapGuard` refcount-decrement bug that would monotonically
///   leak entries under sustained recovery storms.
///
/// Pre-fix regression mode (K=1 mutex removed or weakened): all N
/// restorers would race through `get_session_checkpoint` +
/// `apply_session_checkpoint`, producing somewhere between 1 and N
/// adapter fetches depending on scheduling and the post-acquire EXISTS
/// re-check ordering. The exact-equals-1 assertion makes the K=1
/// invariant a bright-line falsifier rather than a schedule-dependent
/// maybe.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn e2e_session_recovery_concurrent_handlers() {
    let harness = setup_counting_harness(1).await;

    // ── Step 1: CreateSession + CreateLayer through the real RPC entry points.
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: SESSION_NAME.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(ompf_bytes),
    ]);
    let create_layer_resp = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for harness setup");
    let object_id = create_layer_resp.into_inner().object_id;
    assert!(object_id > 0, "CreateLayer must mint a non-zero object_id");

    // ── Step 2: wait for the session-checkpoint blob to land on disk.
    // The writer's enqueue path is fire-and-forget; CreateLayer returns
    // before the spawned put-task lands the blob, so we must poll. The
    // dispatcher writes one blob per lifecycle event (CreateSession +
    // CreateLayer each enqueue one), but the LocalDisk adapter
    // overwrites — the final state is one blob carrying the
    // post-CreateLayer mapping.
    //
    // We need to construct an `Arc<dyn CheckpointStore>` for the helper
    // signature; the counting store implements the trait directly.
    let store_for_wait: Arc<dyn CheckpointStore> = harness.counting.clone();
    wait_for_session_checkpoint(&store_for_wait, &session_id, 10).await;
    // `wait_for_session_checkpoint` returns as soon as ANY blob lands — that is
    // CreateSession's v1 (session metadata, no layer mapping). The layer
    // mapping rides CreateLayer's v2, written by the same fire-and-forget
    // dispatcher. Drain the writer's TaskTracker so v2 is durable before the
    // wipe; without this, a loaded/cold-cache run recovers off v1 and every
    // concurrent UpdateLayer below hits NotFound (no layer_mappings entry).
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    // ── Step 3: reset the counter. `wait_for_session_checkpoint` polls
    // `get_session_checkpoint`, so the counter is currently nonzero. The
    // assertion below measures only post-wipe recovery-driven fetches.
    harness.counting.reset_count();

    // Pre-wipe metric snapshot. The harness should produce a clean zero
    // baseline (no recovery has fired yet), but the snapshot pattern is
    // robust against any future change that pre-seeds a counter cell.
    let restored_pre = session_recovery_metric(&harness.metrics, "restored");
    let already_present_pre = session_recovery_metric(&harness.metrics, "already_present");
    let not_recoverable_pre = session_recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = session_recovery_metric(&harness.metrics, "error");

    // ── Step 4: wipe `le:{sid}*` (session-meta + layer-mapping + layers-index).
    wipe_session_state(&harness.tv, &session_id).await;
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta probe must succeed after wipe");
    assert!(
        !meta_exists,
        "wipe_session_state must UNLINK le:{{session_id}}:sess; a surviving key here \
         means the wipe missed it and the concurrent dispatch would short-circuit \
         on the cache's pre-acquire EXISTS check before ever reaching the recovery \
         branch — silently passing the test."
    );

    // ── Step 5: invalidate the session-base-path cache so every spawned
    // task sees a cache MISS. Without this, the cache TTL (2s) would
    // serve the pre-wipe base_path and recovery would never fire.
    invalidate_session_cache(&harness.svc, &session_id);

    // ── Step 6: spawn N=8 concurrent UpdateLayer RPCs gated on a Barrier.
    // Each task takes its own slot index so its `set_order` / payload is
    // distinct (defensive — the test does not depend on this, but a
    // future write-dedup change should not silently coalesce the writes).
    let barrier = Arc::new(Barrier::new(CONCURRENCY));
    let mut handles = Vec::with_capacity(CONCURRENCY);
    for slot in 0..CONCURRENCY {
        let svc = harness.svc.clone();
        let session_id = session_id.clone();
        let barrier = Arc::clone(&barrier);
        let timestamp = 1_700_000_000_000_000_u64 + slot as u64;
        handles.push(tokio::spawn(async move {
            // Build the per-task delta + framing OUTSIDE the barrier so
            // the barrier release window is as tight as possible. The
            // expensive work (encoding the OMPF, building the
            // `Streaming<UpdateLayerRequest>`) is hoisted; only the
            // handler dispatch runs after `barrier.wait().await`.
            let delta = make_update_delta(slot as u64, slot as u8);
            let ompf = delta_to_ompf_bytes(&delta, &[]);
            let stream = streaming_update(vec![
                make_update_header(
                    &session_id,
                    LAYER_NAME,
                    object_id,
                    timestamp,
                    &format!("{CLIENT_ID}-{slot}"),
                ),
                make_update_data(ompf),
            ]);
            let request = make_streaming_token_request(stream);
            barrier.wait().await;
            handle_update_layer(&svc, request).await
        }));
    }

    // ── Step 7: collect outcomes. Every RPC MUST return Ok — none crash,
    // none surface a spurious NotFound from the wiring race.
    let mut ok_count = 0_usize;
    let mut errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("spawned task must not panic") {
            Ok(resp) => {
                ok_count += 1;
                let seq = resp.into_inner().sequence_num;
                assert!(
                    seq > 0,
                    "post-recovery UpdateLayer must mint a non-zero sequence_num; got {seq}"
                );
            }
            Err(status) => {
                errors.push(format!(
                    "code={:?} message={:?}",
                    status.code(),
                    status.message()
                ));
            }
        }
    }
    assert!(
        errors.is_empty(),
        "every concurrent UpdateLayer MUST succeed via the session-recovery branch in \
         the cache funnel; got {} errors out of {CONCURRENCY}: {errors:?}",
        errors.len()
    );
    assert_eq!(
        ok_count, CONCURRENCY,
        "expected all {CONCURRENCY} RPCs to succeed; got ok_count={ok_count}"
    );

    // ── Step 8: adapter call count is EXACTLY 1. This is the load-bearing
    // K=1 admission assertion. The K=1 mutex serializes the bursts; the
    // winner runs `get_session_checkpoint` + `apply_session_checkpoint`;
    // the losers short-circuit at the post-acquire EXISTS re-check
    // (step 4 of `try_recover_session`) BEFORE step 5's
    // `get_session_checkpoint` fetch. Pre-fix regression (mutex removed
    // or keyed wrong) would produce >1; a re-check ordering bug
    // (fetch-before-check) would also produce >1.
    let get_count = harness.counting.get_session_calls();
    assert_eq!(
        get_count, 1,
        "exactly ONE adapter get_session_checkpoint call must reach the store under \
         N={CONCURRENCY} concurrent RPCs against the same session_id — the K=1 \
         admission discipline inside SessionRecoveryHook collapses the burst to \
         one apply. Got get_session_calls={get_count}; if >1 the per-`session_id` \
         admission mutex is not serializing recoverers and the apply pipeline is \
         duplicating work."
    );

    // ── Step 9: metric assertions. `restored=1` (the winner's apply),
    // `already_present=N-1` (the losers' post-acquire EXISTS re-check),
    // every other cell unchanged.
    let restored_delta = session_recovery_metric(&harness.metrics, "restored") - restored_pre;
    let already_present_delta =
        session_recovery_metric(&harness.metrics, "already_present") - already_present_pre;
    assert_eq!(
        restored_delta, 1,
        "exactly ONE session-recovery attempt must have reported `restored` (the \
         K=1 winner). Got restored_delta={restored_delta}; if >1 the K=1 admission \
         is broken (multiple appliers ran); if 0 the recovery branch never fired \
         (cache HIT, wiring regression, or wipe missed a key)."
    );
    assert_eq!(
        already_present_delta,
        (CONCURRENCY - 1) as u64,
        "exactly N-1 ({}) session-recovery attempts must have reported \
         `already_present` (the K=1 losers' post-acquire EXISTS re-check). Got \
         already_present_delta={already_present_delta}; if <N-1 some losers escaped \
         the re-check and re-ran the apply (would also show up in \
         get_session_calls > 1).",
        CONCURRENCY - 1
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "not_recoverable"),
        not_recoverable_pre,
        "post-recovery `not_recoverable` MUST equal pre-call snapshot — a session \
         checkpoint exists so the NoCheckpoint arm cannot fire."
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "error"),
        error_pre,
        "post-recovery `error` MUST equal pre-call snapshot — no transient backend \
         failure was injected."
    );

    // ── Step 10: admission map drains to zero. The `AdmissionMapGuard`'s
    // RAII drop retires the per-`session_id` entry on the last waiter's
    // drop-out. A residual entry signals a refcount-decrement bug that
    // would monotonically leak memory under sustained recovery storms.
    assert_eq!(
        harness.session_recovery.admission_len(),
        0,
        "admission DashMap must drain to zero after every concurrent recoverer for \
         the shared session_id drops out; got len={}. A non-zero residual means \
         the AdmissionMapGuard's refcount-decrement logic dropped an entry without \
         removing the map slot.",
        harness.session_recovery.admission_len()
    );

    // ── Step 11: graceful shutdown of the checkpoint loop.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
