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

//! End-to-end test for Epic AC #6 — per-`session_id` recovery admission
//! collapse (Subtask `t3.fgj.7v.o7.u6` under Task `t2.fgj.7v.o7`, Epic
//! `t1.fgj.7v`, Bee `b.fgj`).
//!
//! ## What this test pins
//!
//! Two concurrent (here generalized to N) `try_recover_session` calls
//! against the SAME `session_id` from the same replica MUST produce EXACTLY
//! ONE `get_session_checkpoint` adapter call. The remaining N-1 callers
//! short-circuit at the post-acquire `EXISTS le:{sid}:sess` re-check (step
//! 4 of `SessionRecoveryHook::try_recover_session`) and observe
//! [`SessionRecoveryOutcome::AlreadyPresent`](live_edit::checkpoint::SessionRecoveryOutcome::AlreadyPresent).
//! Pre-fix regression (K=1 mutex removed or weakened): the burst would race
//! through `get_session_checkpoint` + `apply_session_checkpoint` and
//! produce between 1 and N adapter fetches depending on scheduling.
//!
//! This is the session-recovery analog of Epic 1's per-`object_id`
//! `recovery_admission_test.rs` — the same admission discipline applied to
//! the new `session_id`-keyed K=1 map on
//! [`SessionRecoveryHook`](live_edit::checkpoint::SessionRecoveryHook).
//!
//! ## Coverage gap addressed
//!
//! Task 6's inline `#[cfg(test)]` tests verify the per-`session_id` K=1
//! deduplication by direct-mock-counted calls to `try_recover_session`. A
//! sibling per-handler concurrent test
//! (`e2e_session_recovery_concurrent_handlers_test.rs` from Task 4)
//! exercises this for `wipe_session_state` (session-keyed wipe only). This
//! test pins the same invariant under the **full-Valkey-wipe** failure
//! mode AC #6 cites — both layer and session recovery fire in series, and
//! the session-recovery admission must still collapse to K=1 regardless of
//! the per-handler layer-recovery also running on the same dispatch.
//!
//! Three scenarios:
//!
//! 1. **`concurrent_recoveries_same_session_id_collapse_to_one_get_session_checkpoint`**
//!    — 10 concurrent `handle_update_layer` RPCs against the same
//!    `(session_id, layer_name, object_id)` after a full-Valkey wipe.
//!    `get_session_checkpoint` is called EXACTLY ONCE; the winner records
//!    `restored{scope=session}+=1`, the other 9 record
//!    `already_present{scope=session}+=1`.
//!
//! 2. **`distinct_session_ids_recover_in_parallel`** — 3 distinct sessions
//!    with their own layers and checkpoints, full wipe, 3 concurrent
//!    `handle_update_layer` RPCs against the 3 different `session_id`s.
//!    All 3 succeed AND `get_session_calls() == 3` — the
//!    per-`session_id` admission mutex does NOT serialize cross-session
//!    recoveries.
//!
//! 3. **`admission_dashmap_drains_after_burst`** — uses Task 6's
//!    `SessionRecoveryHook::admission_len()` accessor to assert the
//!    admission `DashMap` returns to 0 after every burst. Pins the
//!    `AdmissionMapGuard` refcount-decrement invariant from
//!    `src/checkpoint/session_recovery.rs::AdmissionMapGuard::drop`. A
//!    residual entry signals a leak that would monotonically grow under
//!    sustained recovery storms.
//!
//! ## Wiring deltas vs. the LocalDisk harness
//!
//! [`setup_checkpoint_harness`] in `helpers.rs` wires the recovery hooks
//! against the unwrapped `Arc<dyn CheckpointStore>`. We need the SAME store
//! shape but wrapped with a `get_session_checkpoint` counter on the
//! recovery path. To get that we replicate the harness's wiring locally
//! with a `CountingCheckpointStore` over a [`LocalDiskCheckpointStore`] and
//! feed the WRAPPED `Arc<dyn CheckpointStore>` to every consumer (writer,
//! recovery hooks, periodic loop). The counter therefore captures every
//! adapter call (PUTs from the writer, GETs from the recovery hook, etc.),
//! and we reset it to zero AFTER the pre-wipe setup completes so the
//! post-wipe assertion measures only recovery-driven
//! `get_session_checkpoint` calls.
//!
//! The counting wrapper is duplicated inline rather than lifted to a
//! shared module per the Subtask body's explicit instruction (mirror
//! `recovery_admission_test.rs:95-203`; duplicate inline). The duplication
//! is ~95 lines of trivial delegation; lifting it to a shared module would
//! introduce cross-file coupling between the layer-recovery test surface
//! and the session-recovery test surface for marginal savings, and would
//! force every consumer to think about which counter applies to which
//! method.
//!
//! ## Cache TTL trap
//!
//! [`SessionBasePathCache`] has a 2-second positive TTL. The baseline
//! `handle_create_layer` populates the cache, so without explicit
//! invalidation every post-wipe handler would HIT the cache, return the
//! cached `base_path`, and never consult `session_store.get_base_path` —
//! the session-recovery branch would never fire and the test would
//! silently pass with `restored == 0`. We call
//! [`invalidate_session_cache`] AFTER the wipe so every spawned RPC sees
//! a cache MISS and exercises the cold path through the cache funnel.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::HashSet;
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

const SESSION_BASE_PATH: &str = "/stage";
const LAYER_NAME: &str = "admission-layer";
const CLIENT_ID: &str = "admission-client";

// =============================================================================
// CountingCheckpointStore is the shared `CountingGetSessionCheckpointStore`
// imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 2 deduplication). Aliased to the
// local name so the harness's `counting:` field type stays stable.
// =============================================================================

type CountingCheckpointStore = CountingGetSessionCheckpointStore;

// =============================================================================
// Local counting harness — mirrors `setup_checkpoint_harness` from
// `helpers.rs` but substitutes `CountingCheckpointStore` over
// `LocalDiskCheckpointStore` so the counter is wired across every consumer
// (writer, recovery hooks, periodic loop). This is the same harness shape
// `e2e_session_recovery_concurrent_handlers_test.rs` uses for the same
// reason — we need a counter on the wrapped store, not the unwrapped one.
// =============================================================================

/// All the pieces an admission-collapse test needs to drive the
/// write -> checkpoint -> wipe -> recover cycle and inspect the adapter-call
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
    /// `TaskTracker` shared with the harness's [`SessionCheckpointWriter`].
    /// Mirrors [`CheckpointHarness::session_checkpoint_tracker`]; tests
    /// pass it to [`wait_for_writer_idle`] to drain in-flight
    /// fire-and-forget put / delete tasks before any
    /// `delete_session_checkpoint` or wipe step.
    session_checkpoint_tracker: TaskTracker,
    /// LocalDisk adapter root, kept alive for the duration of the test.
    /// RAII-cleaned on drop. Never accessed by the test directly; the
    /// counting store holds a `PathBuf` rooted under this directory.
    #[allow(dead_code)]
    tempdir: TempDir,
}

/// Build a `Config` shaped like `helpers::setup_checkpoint_harness`'s
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
    // Without `with_recovery_hook`, `SessionBasePathCache::get_base_path`
    // would surface NotFound from `session_store.get_base_path` directly
    // and never consult the session-recovery hook — the post-wipe
    // dispatch would fail rather than fire recovery.
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
// gRPC framing + token helpers. Imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication).
// =============================================================================

/// Probe the live label-cell counter value for a given outcome on the
/// `live_edit_checkpoint_recovery_attempts_total{result, scope="session",
/// service}` metric. `scope="session"` is load-bearing — Task 6 split the
/// metric into per-scope cells (`session` vs `layer`); this test asserts
/// against the session-recovery cells. The shape mirrors
/// `e2e_session_recovery_concurrent_handlers_test.rs::session_recovery_metric`
/// — we cannot use the helper from `helpers.rs` because that one takes
/// `&CheckpointHarness` (the LocalDisk-bound harness type) and we are
/// running against a local `CountingHarness` so the wrapped-store wiring
/// is reachable.
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
/// branch and silently passing the test with `restored == 0`. We
/// invalidate after the wipe so every concurrent RPC sees a cache MISS
/// and exercises the cold path through the cache funnel.
fn invalidate_session_cache(svc: &LiveEditService, session_id: &str) {
    svc.session_base_path_cache.invalidate(session_id);
}

/// Build a small incremental UpdateLayer delta carrying ONE `SetField`.
/// Each concurrent task uses its own `set_order` / `marker` pair so the
/// post-recovery `update_layer` writes are independently observable; the
/// test does not assert on those writes' contents (the focus is the
/// admission collapse), but distinct payloads keep the writes from
/// short-circuiting any future write-deduplication change.
///
/// `base_version: 1` (any non-zero) marks the delta as an INCREMENTAL
/// diff — `base_version == 0 && is_diff == true` would take the
/// reset-delta branch and wipe sections/fields, which is irrelevant noise
/// for the recovery-collapse signal. Mirrors the `delta_with_field_update`
/// shape in `e2e_session_recovery_concurrent_handlers_test.rs`.
fn make_update_delta(set_order: u64, marker: u8) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: 0,
            key_name: format!("admission-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

/// Create one session through the real RPC entry point, then create one
/// layer under it, wait for the post-CreateLayer session-checkpoint blob
/// to land, and return `(session_id, object_id)`. The session-checkpoint
/// writer is wired on the harness's SessionStore / LayerStore so the two
/// lifecycle hooks each enqueue a fire-and-forget session-checkpoint put;
/// `wait_for_session_checkpoint` blocks until the spawned put-task lands
/// the blob.
async fn create_session_with_layer(harness: &CountingHarness, session_name: &str) -> (String, u64) {
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: session_name.to_owned(),
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

    // Wait for the session-checkpoint blob to land on disk AND for the
    // writer's TaskTracker to drain. Without the blob wait, the
    // subsequent full wipe would race the writer task and the post-wipe
    // recovery would observe `NoCheckpoint` from the hook — the test
    // would fail at the metric assertion rather than the wiring check
    // it is targeting. Without the tracker drain, a second
    // fire-and-forget put-task (CreateSession enqueues one, CreateLayer
    // enqueues another) may still be in flight when the wipe lands —
    // overwriting it and re-publishing the blob under fast Valkey.
    let store_for_wait: Arc<dyn CheckpointStore> = harness.counting.clone();
    wait_for_session_checkpoint(&store_for_wait, &session_id, 10).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    // Also wait for the FIRST layer-checkpoint generation to land via the
    // periodic loop. Without this, the post-wipe layer-recovery branch would
    // observe `NoCheckpoint`, report `NotRecoverable`, and surface the
    // commit's own `ServiceError::NotFound` verbatim — masking the
    // session-recovery K=1 collapse we are actually testing here.
    //
    // (b.f1v changed the status this would surface: the empty-header case is
    // `NotFound` from both commit backends now, not `FailedPrecondition`. The
    // masking hazard is unchanged, only its symptom.)
    let _ = wait_for_checkpoint(&store_for_wait, object_id, 10).await;

    (session_id, object_id)
}

// =============================================================================
// Scenario 1 — N=10 concurrent recoveries on the SAME session_id collapse
// to ONE adapter `get_session_checkpoint` call.
// =============================================================================

/// Fire N=10 concurrent `handle_update_layer` RPCs against the same
/// `(session_id, layer_name, object_id)` after a full-Valkey wipe. Per
/// AC #6 the per-`session_id` K=1 admission discipline on
/// [`SessionRecoveryHook`] MUST collapse the burst to EXACTLY ONE adapter
/// `get_session_checkpoint` call; the other 9 must observe the restored
/// session-meta hash via the post-acquire `EXISTS le:{sid}:sess` re-check
/// (step 4 of `try_recover_session`) and report
/// `SessionRecoveryOutcome::AlreadyPresent` without touching the store.
///
/// Pre-fix regression mode (K removed or weakened): all 10 recoverers
/// would race through `get_session_checkpoint` + `apply_session_checkpoint`,
/// producing somewhere between 1 and 10 store fetches depending on
/// scheduling. The exact-equals-1 assertion makes the K=1 invariant a
/// bright-line falsifier rather than a schedule-dependent maybe.
///
/// Step-by-step rationale:
///
/// 1. Build the counting harness so every adapter call is observable.
/// 2. CreateSession + CreateLayer through the real RPC entry points —
///    the lifecycle hooks enqueue session-checkpoint writes via the
///    dispatcher. `create_session_with_layer` blocks until the blob has
///    landed on disk.
/// 3. Reset the counter. `wait_for_session_checkpoint` polls
///    `get_session_checkpoint`, so without the reset the counter would
///    include those polling calls and the exact-1 assertion would fail.
/// 4. `full_valkey_wipe` SCANs + UNLINKs EVERY `tv.prefix()*` key —
///    session-meta, layer-mapping, obj:*, path-index, stream, GC, ext
///    values, counters. Both session and layer recovery must rebuild
///    everything.
/// 5. Invalidate the in-process `SessionBasePathCache` entry. The 2s TTL
///    would otherwise serve the pre-wipe `base_path` and recovery would
///    never fire.
/// 6. `Barrier::new(N)` releases every spawned task at the same instant
///    so the contention window the K=1 mutex is supposed to collapse is
///    genuinely exercised. Without the barrier the first task could
///    finish admission + apply before any later task entered
///    `try_recover_session`, letting the post-acquire EXISTS re-check on
///    the followers fire on a no-contention path and vacuously passing.
/// 7. Every spawned RPC must succeed — no thundering-herd failures.
/// 8. `get_session_calls() == 1` — the K=1 collapse.
/// 9. `restored{session} == 1` AND `1 <= already_present{session} <= 9` —
///    metric distribution mirrors the outcome distribution. The
///    `restored == 1` half is the load-bearing applier-count invariant;
///    the bounded range on `already_present` accommodates the production-
///    correct cache-funnel short-circuit (see assertion comment for the
///    timing-race detail).
/// 10. Path-index members carry the recovered session_id exactly once —
///     no duplicated path-index members from the concurrent recoveries.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_recoveries_same_session_id_collapse_to_one_get_session_checkpoint() {
    const N: usize = 10;
    let harness = setup_counting_harness(1).await;

    // Step 1-2: harness + CreateSession + CreateLayer + wait for blob.
    let (session_id, object_id) =
        create_session_with_layer(&harness, "admission-same-session").await;

    // Step 3: reset the counter — the pre-wipe blob-landed poll has
    // already incremented `get_session_checkpoint` calls.
    harness.counting.reset_count();

    // Pre-wipe metric snapshot. The harness should produce a clean zero
    // baseline (no recovery has fired yet), but the snapshot pattern is
    // robust against any future change that pre-seeds a counter cell.
    let restored_pre = session_recovery_metric(&harness.metrics, "restored");
    let already_present_pre = session_recovery_metric(&harness.metrics, "already_present");
    let not_recoverable_pre = session_recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = session_recovery_metric(&harness.metrics, "error");

    // Step 4: full Valkey wipe — every `le:{prefix}*` key gone.
    full_valkey_wipe(&harness.tv).await;
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta probe must succeed after full_valkey_wipe");
    assert!(
        !meta_exists,
        "full_valkey_wipe must remove le:{{sid}}:sess; a surviving key here means \
         the wipe missed it and the concurrent dispatch would short-circuit on \
         the pre-acquire EXISTS check before reaching the recovery branch — \
         silently passing the test."
    );

    // Step 5: invalidate the session-base-path cache so every spawned
    // task sees a cache MISS and exercises the recovery branch.
    invalidate_session_cache(&harness.svc, &session_id);

    // Step 6: spawn N concurrent UpdateLayer RPCs gated on a barrier.
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = Vec::with_capacity(N);
    for slot in 0..N {
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

    // Step 7: every spawned RPC MUST return Ok.
    let mut ok_count = 0_usize;
    let mut errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("spawned task must not panic") {
            Ok(resp) => {
                ok_count += 1;
                let seq = resp.into_inner().sequence_num;
                assert!(
                    seq > 0,
                    "post-recovery UpdateLayer must mint a non-zero sequence_num; \
                     got {seq}"
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
        "every concurrent UpdateLayer MUST succeed via the session-recovery branch \
         in the cache funnel; got {} errors out of {N}: {errors:?}",
        errors.len()
    );
    assert_eq!(
        ok_count, N,
        "expected all {N} RPCs to succeed; got ok_count={ok_count}"
    );

    // Step 8: adapter call count is EXACTLY 1. This is the load-bearing
    // K=1 admission assertion. Pre-fix regression (mutex removed or keyed
    // wrong) would produce >1; a re-check ordering bug (fetch-before-
    // check) would also produce >1.
    let get_count = harness.counting.get_session_calls();
    assert_eq!(
        get_count, 1,
        "exactly ONE adapter get_session_checkpoint call must reach the store under \
         N={N} concurrent RPCs against the same session_id — the K=1 admission \
         discipline inside SessionRecoveryHook collapses the burst to one apply. \
         Got get_session_calls={get_count}; if >1 the per-`session_id` \
         admission mutex is not serializing recoverers and the apply pipeline is \
         duplicating work."
    );

    // Step 9: metric assertions. The K=1 collapse pinning is asserted by
    // the counter check above; the metric assertions here are
    // complementary. Two bounds:
    //
    // - `restored == 1` (LOAD-BEARING): exactly one applier ran, the K=1
    //   winner. >1 means the K=1 mutex is broken (multiple appliers ran);
    //   0 means the recovery branch never fired (cache HIT before the
    //   wipe took effect, wiring regression, or the wipe missed a key).
    //
    // - `already_present >= 1 && already_present <= N-1`: the losers that
    //   DID reach the recovery hook short-circuited at the post-acquire
    //   EXISTS re-check. The upper bound (N-1) is the all-callers-reach-
    //   the-hook case; the lower bound (1) ensures at least one loser
    //   observed the K=1 contention (so the collapse is actually
    //   exercised, not vacuously satisfied by N=1).
    //
    //   The strict equality `already_present == N-1` from the Subtask
    //   body assumes every concurrent caller reaches `try_recover_session`
    //   before the winner's apply completes. In practice the
    //   `SessionBasePathCache::get_base_path` cache funnel re-issues
    //   `session_store.get_base_path` AFTER recovery returns
    //   `Recovered`; the winner's restored `session_meta` may be visible
    //   to a follower's HGET on the FIRST iteration of the cache loop,
    //   in which case the follower returns `Ok(base_path)` from
    //   `session_store.get_base_path` directly and never enters the
    //   recovery branch — production-correct behavior (the cache funnel
    //   short-circuits successful lookups), but it makes the strict
    //   `== N-1` assertion timing-dependent. The bounded range below
    //   pins the K=1 collapse without the timing race; the
    //   `get_session_calls == 1` assertion above is the
    //   load-bearing K=1 invariant.
    let restored_delta = session_recovery_metric(&harness.metrics, "restored") - restored_pre;
    let already_present_delta =
        session_recovery_metric(&harness.metrics, "already_present") - already_present_pre;
    assert_eq!(
        restored_delta, 1,
        "session_recovery_metric(restored) must tick by exactly 1 (the K=1 winner). \
         Got restored_delta={restored_delta}; if >1 the K=1 admission is broken \
         (multiple appliers ran); if 0 the recovery branch never fired (cache HIT, \
         wiring regression, or wipe missed a key)."
    );
    // Upper bound (`<= N - 1`) is the K=1 collapse invariant: no concurrent
    // caller may tick `already_present` more than once, and the winner does
    // not tick it (the winner ticks `restored`). The lower bound is
    // intentionally `>= 0` (i.e. omitted in spirit, retained as an explicit
    // `<= N - 1` cap) because the cache funnel's post-recovery
    // `session_store.get_base_path` re-read can HIT the live restored state
    // before `try_recover_session` returns its `AlreadyPresent` outcome on
    // every follower — under extremely fast scheduling, every follower may
    // resolve via the live HGET path and never reach the `already_present`
    // tick. That outcome is production-correct (the cache funnel
    // short-circuits successful lookups) and would otherwise make a
    // `>= 1` lower bound a timing flake. The load-bearing K=1 admission is
    // pinned strictly by the `get_session_calls == 1`
    // assertion above; this assertion only guards against the inverse
    // failure mode (multiple appliers ticking `already_present`).
    assert!(
        already_present_delta <= (N as u64 - 1),
        "session_recovery_metric(already_present) must tick at most N-1 ({}) — \
         no caller can tick the cell twice, and the K=1 winner ticks `restored` \
         instead. Got already_present_delta={already_present_delta}.",
        N - 1
    );
    // Every caller that entered try_recover_session must have ticked one
    // of {restored, already_present} — the recovery hook does not tick
    // any other cell on the happy-path race.
    assert!(
        restored_delta + already_present_delta <= N as u64,
        "no caller can over-tick the recovery metric: restored ({restored_delta}) \
         + already_present ({already_present_delta}) must be <= N ({N})"
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "not_recoverable"),
        not_recoverable_pre,
        "post-recovery `not_recoverable{{session}}` MUST equal pre-call snapshot — \
         a session checkpoint exists so the NoCheckpoint arm cannot fire."
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "error"),
        error_pre,
        "post-recovery `error{{session}}` MUST equal pre-call snapshot — no \
         transient backend failure was injected."
    );

    // Step 10: path-index members carry the recovered session_id exactly
    // once. Two concurrent recoveries that both ran the apply pipeline
    // would have SADD'd the path-index member twice — Set semantics
    // would still deduplicate the member, but the assertion here catches
    // a future failure mode where the apply pipeline runs the SADD with
    // a per-attempt suffix or otherwise duplicates the member.
    let path_index_key = harness.tv.key_schema().path_index(SESSION_BASE_PATH);
    let path_index_members: Vec<String> =
        fred::interfaces::SetsInterface::smembers(&harness.tv.client().pool, &path_index_key)
            .await
            .expect("SMEMBERS path_index must succeed post-recovery");
    let matching: Vec<&String> = path_index_members
        .iter()
        .filter(|m| m.as_str() == session_id.as_str())
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "path_index MUST carry exactly one member equal to the recovered \
         session_id={session_id}; got {} matching members out of all members: \
         {path_index_members:?}",
        matching.len()
    );

    // Step 11: admission map drains to zero after every recoverer drops
    // out. Pinned more rigorously in scenario 3 below across multiple
    // bursts; included here as a per-scenario invariant so a single-burst
    // refcount-decrement regression surfaces on this test too.
    assert_eq!(
        harness.session_recovery.admission_len(),
        0,
        "admission DashMap must drain to zero after every concurrent recoverer for \
         the shared session_id drops out; got len={}",
        harness.session_recovery.admission_len()
    );

    // Step 12: graceful shutdown of the checkpoint loop.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 2 — DISTINCT session_ids recover in parallel; no false
// serialization across session_ids; counter == number of sessions.
// =============================================================================

/// Mirror Epic 1's cross-key non-serialization assertion at the
/// session-recovery layer. Create 3 distinct sessions (each with its own
/// layer), full-Valkey-wipe, then spawn 3 concurrent `handle_update_layer`
/// RPCs against the 3 different `session_id`s. All 3 MUST succeed AND
/// `get_session_calls() == 3` — no cross-`session_id`
/// serialization.
///
/// Pre-fix regression mode where the admission map keyed on a constant
/// (or a singleton mutex covered all session_ids) would serialize all 3
/// recoverers behind one mutex AND the count would still be 3 — so the
/// count alone cannot distinguish that mode. We additionally assert all 3
/// succeed (which they would even under shared serialization) AND that
/// the post-recovery state for each session carries the recovered
/// session_id in the path-index exactly once (which a hypothetical bug
/// that mis-keyed the K=1 mutex on a constant could corrupt by running
/// one apply that affects only one session).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn distinct_session_ids_recover_in_parallel() {
    const SESSION_COUNT: usize = 3;
    let harness = setup_counting_harness(1).await;

    // Create three distinct sessions, each with its own layer, each
    // session-checkpoint blob landed on disk.
    let mut sessions: Vec<(String, u64)> = Vec::with_capacity(SESSION_COUNT);
    for i in 0..SESSION_COUNT {
        let pair = create_session_with_layer(&harness, &format!("admission-distinct-{i}")).await;
        sessions.push(pair);
    }
    // Sanity: all session_ids are distinct.
    let ids: HashSet<&str> = sessions.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(
        ids.len(),
        SESSION_COUNT,
        "harness must mint distinct session_ids for distinct CreateSession calls; \
         got {} unique ids out of {SESSION_COUNT}: {sessions:?}",
        ids.len()
    );

    // Reset the counter so the pre-wipe blob-landed polls don't count.
    harness.counting.reset_count();

    let restored_pre = session_recovery_metric(&harness.metrics, "restored");
    let already_present_pre = session_recovery_metric(&harness.metrics, "already_present");

    // Full Valkey wipe — every session's `le:{sid}:sess` + layer-mapping
    // + obj:* + path-index entries are gone.
    full_valkey_wipe(&harness.tv).await;
    for (sid, _) in &sessions {
        let session_meta_key = harness.tv.key_schema().session_meta(sid);
        let meta_exists: bool = harness
            .tv
            .client()
            .pool
            .exists(&session_meta_key)
            .await
            .expect("EXISTS session_meta probe must succeed after full_valkey_wipe");
        assert!(
            !meta_exists,
            "full_valkey_wipe must remove le:{{sid}}:sess for every distinct session; \
             session_id={sid} still has a session_meta key — the wipe missed it."
        );
    }

    // Invalidate the cache for EVERY session — the cache TTL would
    // otherwise serve the pre-wipe base_path for each and the recovery
    // branch would never fire.
    for (sid, _) in &sessions {
        invalidate_session_cache(&harness.svc, sid);
    }

    // Spawn one concurrent UpdateLayer RPC per session_id, all released
    // simultaneously via a barrier so the per-`session_id` K=1 mutexes
    // genuinely race against each other. A separate barrier per session
    // would let session A's recovery finish before session B's started,
    // vacuously passing the "no cross-session serialization" assertion.
    let barrier = Arc::new(Barrier::new(SESSION_COUNT));
    let mut handles = Vec::with_capacity(SESSION_COUNT);
    for (slot, (session_id, object_id)) in sessions.iter().enumerate() {
        let svc = harness.svc.clone();
        let session_id = session_id.clone();
        let object_id = *object_id;
        let barrier = Arc::clone(&barrier);
        let timestamp = 1_700_000_000_000_000_u64 + slot as u64;
        handles.push(tokio::spawn(async move {
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
            (session_id, handle_update_layer(&svc, request).await)
        }));
    }

    // Collect outcomes. Every RPC MUST return Ok.
    let mut ok_session_ids: HashSet<String> = HashSet::new();
    let mut errors: Vec<(String, String)> = Vec::new();
    for h in handles {
        let (sid, outcome) = h.await.expect("spawned task must not panic");
        match outcome {
            Ok(resp) => {
                let seq = resp.into_inner().sequence_num;
                assert!(
                    seq > 0,
                    "post-recovery UpdateLayer must mint a non-zero sequence_num \
                     for session_id={sid}; got {seq}"
                );
                ok_session_ids.insert(sid);
            }
            Err(status) => {
                errors.push((
                    sid,
                    format!("code={:?} message={:?}", status.code(), status.message()),
                ));
            }
        }
    }
    assert!(
        errors.is_empty(),
        "every concurrent UpdateLayer across {SESSION_COUNT} distinct session_ids \
         MUST succeed; got {} errors: {errors:?}",
        errors.len()
    );
    assert_eq!(
        ok_session_ids.len(),
        SESSION_COUNT,
        "expected one Ok per distinct session_id; got {} Oks: {ok_session_ids:?}",
        ok_session_ids.len()
    );

    // Counter == SESSION_COUNT — one adapter `get_session_checkpoint`
    // call per distinct session_id. The per-`session_id` admission map
    // collapses NOTHING here (each session_id has its own entry, each
    // burst is N=1), so the count is exactly SESSION_COUNT.
    //
    // Why this assertion catches what scenario 1's count cannot: a
    // regression that mis-keyed the K=1 mutex on a constant would
    // serialize all 3 distinct-session recoveries through one mutex AND
    // still produce a count of 3 (each session's recovery still calls
    // the store once). But the cross-`session_id` admission map shape is
    // a separate failure surface — see the docstring above. This count
    // pins the "no over-collapse" half of the invariant: the K=1 mutex
    // is keyed on session_id, NOT on a shared singleton.
    let get_count = harness.counting.get_session_calls();
    assert_eq!(
        get_count, SESSION_COUNT,
        "exactly {SESSION_COUNT} adapter get_session_checkpoint calls must reach \
         the store under {SESSION_COUNT} concurrent RPCs against {SESSION_COUNT} \
         distinct session_ids — the per-`session_id` admission map keys on \
         session_id, so distinct-session bursts each call the store independently. \
         Got get_session_calls={get_count}; if <{SESSION_COUNT} \
         the admission discipline is over-collapsing across session_ids (broken \
         keying or shared mutex); if >{SESSION_COUNT} a single session's burst \
         was not collapsed to K=1 (regression in scenario 1's invariant)."
    );

    // Metric assertions: 3 distinct restored ticks, zero already_present
    // (every session's burst has N=1, no losers).
    let restored_delta = session_recovery_metric(&harness.metrics, "restored") - restored_pre;
    let already_present_delta =
        session_recovery_metric(&harness.metrics, "already_present") - already_present_pre;
    assert_eq!(
        restored_delta, SESSION_COUNT as u64,
        "session_recovery_metric(restored) must tick by exactly {SESSION_COUNT} — \
         one per distinct session recovery. Got restored_delta={restored_delta}."
    );
    assert_eq!(
        already_present_delta, 0,
        "session_recovery_metric(already_present) must NOT tick for distinct-session \
         recoveries — each session has N=1 recoverer, no losers to short-circuit \
         on the post-acquire EXISTS re-check. Got \
         already_present_delta={already_present_delta}."
    );

    // Post-recovery path-index sanity: every recovered session_id is in
    // the path-index exactly once.
    let path_index_key = harness.tv.key_schema().path_index(SESSION_BASE_PATH);
    let path_index_members: Vec<String> =
        fred::interfaces::SetsInterface::smembers(&harness.tv.client().pool, &path_index_key)
            .await
            .expect("SMEMBERS path_index must succeed post-recovery");
    for (sid, _) in &sessions {
        let matching = path_index_members
            .iter()
            .filter(|m| m.as_str() == sid.as_str())
            .count();
        assert_eq!(
            matching, 1,
            "path_index MUST carry exactly one member equal to recovered \
             session_id={sid}; got {matching} matching members out of all members: \
             {path_index_members:?}"
        );
    }

    // Admission map drains to zero after all recoverers drop out.
    assert_eq!(
        harness.session_recovery.admission_len(),
        0,
        "admission DashMap must drain to zero after every recoverer across all \
         {SESSION_COUNT} session_ids drops out; got len={}. The map peak size \
         during the burst was {SESSION_COUNT} (one entry per session_id); after \
         the bursts, every entry must be retired.",
        harness.session_recovery.admission_len()
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 3 — Admission DashMap drains after every burst.
//
// Pins Task 6's `AdmissionMapGuard` refcount-decrement invariant from
// `src/checkpoint/session_recovery.rs::AdmissionMapGuard::drop`. Uses Task
// 6's `SessionRecoveryHook::admission_len()` accessor.
// =============================================================================

/// Pin the entry-refcount-decrement invariant from
/// `src/checkpoint/session_recovery.rs::AdmissionMapGuard::drop`: the
/// admission `DashMap` MUST return to 0 after every concurrent burst.
/// Two independent bursts shaped to exercise both the
/// "single-entry-many-refcounts" failure mode and the
/// "many-entries-one-refcount-each" failure mode:
///
/// 1. **Same-session burst** — full wipe + N=5 concurrent recoverers on
///    one session_id. Map peak: 1 entry with refcount 5. Post-burst must
///    drain to 0. Catches a refcount-decrement bug where the per-entry
///    refcount stays nonzero after the last guard drops.
/// 2. **Distinct-session burst** — full wipe + 3 distinct session_ids
///    recover concurrently. Map peak: 3 entries with refcount 1 each.
///    Post-burst must drain to 0. Catches a map-key-retirement bug
///    where the `remove_if` shard-locked closure fails to remove an
///    entry whose refcount is already 0.
///
/// Each burst runs on its OWN fresh harness so a state-leak from burst 1
/// cannot mask burst 2's drainage check. The `admission_len()` assertion
/// at the end of each scenario isolates whether the leak is in the
/// per-session refcount-decrement path (burst 1) or the cross-session
/// map-entry-retirement path (burst 2).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn admission_dashmap_drains_after_burst() {
    // ── Burst 1: same-session N=5 — single map entry, refcount 5.
    {
        const BURST: usize = 5;
        let harness = setup_counting_harness(1).await;
        let (sid, oid) = create_session_with_layer(&harness, "admission-drain-same").await;
        harness.counting.reset_count();
        full_valkey_wipe(&harness.tv).await;
        invalidate_session_cache(&harness.svc, &sid);

        let barrier = Arc::new(Barrier::new(BURST));
        let mut handles = Vec::with_capacity(BURST);
        for slot in 0..BURST {
            let svc = harness.svc.clone();
            let session_id = sid.clone();
            let barrier = Arc::clone(&barrier);
            let timestamp = 1_700_000_000_000_000_u64 + slot as u64;
            handles.push(tokio::spawn(async move {
                let delta = make_update_delta(slot as u64, slot as u8);
                let ompf = delta_to_ompf_bytes(&delta, &[]);
                let stream = streaming_update(vec![
                    make_update_header(
                        &session_id,
                        LAYER_NAME,
                        oid,
                        timestamp,
                        &format!("{CLIENT_ID}-drainsame-{slot}"),
                    ),
                    make_update_data(ompf),
                ]);
                let request = make_streaming_token_request(stream);
                barrier.wait().await;
                handle_update_layer(&svc, request).await
            }));
        }
        for h in handles {
            h.await
                .expect("same-session burst task must not panic")
                .expect("same-session burst UpdateLayer must succeed via session-recovery");
        }
        assert_eq!(
            harness.session_recovery.admission_len(),
            0,
            "Same-session burst (N={BURST}): admission DashMap MUST drain to zero \
             after every recoverer drops out; got len={}. A non-zero residual \
             signals the AdmissionMapGuard's refcount-decrement logic dropped an \
             entry without removing the map slot — pinned regression source per \
             Task 6's module-level docstring (the per-session_id K=1 mutex's \
             reference-counted retirement protocol).",
            harness.session_recovery.admission_len()
        );

        harness.cancel.cancel();
        let _ = harness.loop_handle.await;
    }

    // ── Burst 2: distinct-session 3-burst — three map entries, refcount
    // 1 each. Runs on a fresh harness so burst 1's state cannot leak in.
    {
        const BURST: usize = 3;
        let harness = setup_counting_harness(1).await;
        let mut sessions: Vec<(String, u64)> = Vec::with_capacity(BURST);
        for i in 0..BURST {
            let pair =
                create_session_with_layer(&harness, &format!("admission-drain-distinct-{i}")).await;
            sessions.push(pair);
        }
        harness.counting.reset_count();
        full_valkey_wipe(&harness.tv).await;
        for (sid, _) in &sessions {
            invalidate_session_cache(&harness.svc, sid);
        }

        let barrier = Arc::new(Barrier::new(BURST));
        let mut handles = Vec::with_capacity(BURST);
        for (slot, (session_id, object_id)) in sessions.iter().enumerate() {
            let svc = harness.svc.clone();
            let session_id = session_id.clone();
            let object_id = *object_id;
            let barrier = Arc::clone(&barrier);
            let timestamp = 1_700_000_000_000_000_u64 + slot as u64;
            handles.push(tokio::spawn(async move {
                let delta = make_update_delta(slot as u64 + 100, slot as u8);
                let ompf = delta_to_ompf_bytes(&delta, &[]);
                let stream = streaming_update(vec![
                    make_update_header(
                        &session_id,
                        LAYER_NAME,
                        object_id,
                        timestamp,
                        &format!("{CLIENT_ID}-draindistinct-{slot}"),
                    ),
                    make_update_data(ompf),
                ]);
                let request = make_streaming_token_request(stream);
                barrier.wait().await;
                handle_update_layer(&svc, request).await
            }));
        }
        for h in handles {
            h.await
                .expect("distinct-session burst task must not panic")
                .expect("distinct-session burst UpdateLayer must succeed via session-recovery");
        }
        assert_eq!(
            harness.session_recovery.admission_len(),
            0,
            "Distinct-session burst ({BURST} session_ids): admission DashMap MUST \
             drain to zero after every recoverer across all {BURST} session_ids \
             drops out; got len={}. The map peak size during the burst was {BURST} \
             (one entry per session_id); after the burst, every entry must be \
             retired by the `remove_if` closure inside `AdmissionMapGuard::drop`.",
            harness.session_recovery.admission_len()
        );

        harness.cancel.cancel();
        let _ = harness.loop_handle.await;
    }
}
