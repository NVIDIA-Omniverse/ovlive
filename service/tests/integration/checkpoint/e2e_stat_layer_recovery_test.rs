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

//! End-to-end recovery test for the `StatLayer` RPC — Epic AC #7c
//! (Subtask `t3.fgj.ya.wg.nx` under Task `t2.fgj.ya.wg`, Epic
//! `t1.fgj.ya`, Bee `b.fgj`).
//!
//! The contract this file pins is:
//!
//!   "When `LIVE_EDIT_CHECKPOINT_ENABLED=true` and a layer's `obj:*`
//!    materialized state has been wiped from Valkey AFTER a checkpoint
//!    landed, the FIRST `StatLayer` call against that layer MUST
//!    transparently restore the state from the latest checkpoint blob
//!    and return `Ok` with the recovered metadata (the same shape a
//!    pre-wipe `StatLayer` would have returned). The
//!    `live_edit_checkpoint_recovery_attempts_total{result=\"restored\"}`
//!    counter MUST be incremented exactly once."
//!
//! And the inverse:
//!
//!   "When `LIVE_EDIT_CHECKPOINT_ENABLED=false`, the same wipe scenario
//!    MUST surface `Code::NotFound` directly and MUST NOT touch the
//!    checkpoint store, the admission map, or the recovery-attempts
//!    metric — even though a checkpoint blob may exist on disk."
//!
//! Sibling files `e2e_update_layer_recovery_test.rs` (`.rt`) and
//! `e2e_subscribe_read_layer_recovery_test.rs` (`.cm`) pin the same
//! end-to-end recovery contract for the other two wired handlers.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`. Under
//! the project sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tonic::Request;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{CheckpointStore, RecoveryHook};
use live_edit::grpc::layer::handle_stat_layer;
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
// `recovery_metric` is the shared copy in `tests/common`: it pins the same
// `{result, scope="layer", service}` cell this file used to spell locally.
use crate::common::recovery_metric;
use crate::grpc::helpers::{
    build_disabled_recovery_hook, build_disabled_session_recovery_hook, create_session_and_layer,
    AllowAllPermissions, StubUserinfo,
};

/// Build a `StatLayerRequest` with a stub bearer token in the
/// extensions map — same shape `recovery_hook_test.rs` and
/// `postman_regression_test.rs` use.
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

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — Enabled-mode recovery: wipe → StatLayer restores from checkpoint.
// ─────────────────────────────────────────────────────────────────────────────

/// Drives the full happy path the Epic AC #7 contract specifies for
/// `StatLayer`:
///
/// 1. Stand up the harness (real `LocalDiskCheckpointStore`, real
///    leader-elected checkpoint loop, real `RecoveryHook` wired against
///    the same store).
/// 2. CreateSession + CreateLayer (the create handler writes the
///    initial `obj:hdr` with `created_by`, `created_at`, `modified_at`).
/// 3. Wait until the leader-elected loop lands at least one
///    `CheckpointHandle` for the new `object_id` on disk. Capture the
///    handle as proof the checkpoint exists.
/// 4. `wipe_layer_state` UNLINKs every `le:{object_id}:obj:*` key —
///    mirroring the data-loss surface a failover to a fresh replica
///    would observe.
/// 5. Call `handle_stat_layer` with the same `session_id` / `layer_name`.
///    The handler's recovery branch (SDD SR-2.4 / SR-2.6) MUST fire on
///    the post-mapping `stat_layer_for_object_id` NotFound, invoke
///    `RecoveryHook::try_restore_layer(object_id)`, apply the checkpoint,
///    retry the header read, and return `Ok(StatLayerResponse { ... })`
///    populated from the restored header.
///
/// Assertions cover three independent observations of the restored state:
///
/// - The returned `StatLayerResponse` carries non-empty `created_by_user_name`
///   and non-zero `created_at` (the apply-checkpoint pipeline restored the
///   header via `write_object`, which writes both fields verbatim from the
///   checkpoint meta).
/// - The `restored` label cell on `live_edit_checkpoint_recovery_attempts_total`
///   incremented by exactly one — proving exactly ONE recovery attempt
///   landed (at-most-once contract: not zero = recovery never fired; not
///   two = handler retried recovery beyond the latch).
/// - The `not_recoverable` and `error` cells stayed at zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_recovers_layer_state_after_valkey_wipe() {
    let harness = setup_checkpoint_harness(1).await;

    let (session_id, object_id) = create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "e2e-stat-recovery-session",
        "e2e-stat-recovery-layer",
    )
    .await;

    // Wait until the leader-elected checkpoint loop has landed at
    // least one generation for this object_id. The harness configures
    // `interval=0` and `quiesce=0` so the loop's "skip if idle" gate
    // does NOT block the just-created layer; the only latency to absorb
    // is the loop's `tick=1s` schedule. 10 s is comfortably above the
    // worst-case CI tick path.
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Wipe every `le:{object_id}:obj:*` key. Session metadata and the
    // layer-mapping key stay intact so the handler's mapping-resolve
    // succeeds and we end up in the header-read recovery branch
    // (recovery.rs / handle_stat_layer step 5).
    wipe_layer_state(&harness.tv, object_id).await;

    // Snapshot the recovery-attempts counter pre-call so a flaky CI
    // host that pre-incremented the cell (e.g., from a prior leader
    // probe) doesn't poison the post-call delta assertion. In steady
    // state the harness's freshly-built metrics registry starts at
    // zero, but the snapshot pattern is more robust.
    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    let response = handle_stat_layer(
        &harness.svc,
        stat_request(&session_id, "e2e-stat-recovery-layer"),
    )
    .await
    .expect(
        "enabled-mode StatLayer after wipe MUST recover from checkpoint and return Ok; \
         a NotFound here means the recovery branch never fired OR apply_checkpoint \
         failed to restore obj:hdr",
    );

    let stat = response.into_inner();
    assert!(
        !stat.created_by_user_name.is_empty(),
        "StatLayer after recovery MUST surface a non-empty created_by_user_name \
         (apply_checkpoint's write_object pipeline restores the header field from \
         the checkpoint meta `created_by`); empty string here means the apply \
         pipeline ran but never landed the header field"
    );
    assert!(
        stat.created_at > 0,
        "StatLayer after recovery MUST surface a non-zero created_at; \
         zero here means the apply pipeline lost the timestamp during restore"
    );

    // Load-bearing metric assertion — exactly ONE `restored` increment
    // proves the recovery branch fired exactly once for this RPC.
    assert_eq!(
        recovery_metric(&harness.metrics, "restored") - restored_pre,
        1,
        "exactly ONE recovery attempt must have reported `restored` \
         (at-most-once contract: not zero = recovery never fired, \
         not two = handler retried recovery beyond the `recovered_once` latch)"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` cell MUST stay at the pre-call snapshot — \
         the checkpoint blob landed before the wipe, so recovery MUST succeed"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error") - error_pre,
        0,
        "`error` cell MUST stay at the pre-call snapshot — \
         LocalDisk adapter against a healthy tempdir has no transient failures"
    );

    // Graceful shutdown of the checkpoint loop so the spawned task
    // does not outlive the harness drop. The tokio runtime would
    // reap the task either way, but a clean cancel + await mirrors
    // the production drain shape.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — Disabled-mode short-circuit: same wipe MUST surface NotFound.
// ─────────────────────────────────────────────────────────────────────────────

/// Pins the `LIVE_EDIT_CHECKPOINT_ENABLED=false` short-circuit on
/// `StatLayer`:
///
/// 1. Build a `LiveEditService` directly (not the E2E harness) with a
///    `build_disabled_recovery_hook` recovery wiring. This is the same
///    shape `build_service` plumbs in by default; we re-wire here so
///    the metrics handle stays addressable for the post-call counter
///    assertion (the helper-side `build_service` creates its own
///    `Metrics` Arc that we'd otherwise have to thread through).
/// 2. CreateSession + CreateLayer — same shape as test 1.
/// 3. Wipe `obj:*` — same as test 1. (No checkpoint blob exists in
///    this harness; we never spun up the loop. A regression that
///    consulted the store anyway in disabled mode would surface the
///    `NotImplemented` / empty-store path and still skip the recovery
///    increment — which is what we'd then catch via the metric
///    assertion below, in conjunction with the existing
///    `recovery_hook_test.rs::stat_layer_recovery_disabled_returns_not_found_directly`
///    test which uses a `CallCountingStore` wrapper to prove the
///    stronger zero-store-calls property.)
/// 4. Call `handle_stat_layer` — MUST surface `Code::NotFound`.
/// 5. Assert the `live_edit_checkpoint_recovery_attempts_total`
///    counter total stayed at zero across ALL three result labels
///    (no `restored`, no `not_recoverable`, no `error` increments).
///    Disabled-mode short-circuit is the only path that satisfies
///    that property; even a "recovery fired, found nothing" trace
///    would tick `not_recoverable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_returns_not_found_when_recovery_disabled() {
    let tv = TestValkey::setup().await;

    // Build the service directly with a disabled recovery hook. We
    // re-implement the `build_service` shape here (rather than calling
    // it) so the test can observe the same `SharedMetrics` handle the
    // service is using — `build_service` creates its own `Metrics` Arc
    // internally and does not expose it. The wiring matches
    // `grpc/helpers.rs::build_service` 1:1 except for the metrics-Arc
    // hoist.
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let recovery: Arc<RecoveryHook> =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics));
    let session_recovery = build_disabled_session_recovery_hook(&tv, Arc::clone(&metrics));

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

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "e2e-stat-disabled-session",
        "e2e-stat-disabled-layer",
    )
    .await;

    // Also seed a LocalDisk checkpoint store so a regression that
    // accidentally consulted the store anyway in disabled mode (e.g.,
    // a missing short-circuit guard) could find a real blob and
    // restore from it — which would falsify the NotFound assertion
    // below. The seed is technically optional (the disabled-mode hook
    // never reaches the store), but it strengthens the regression
    // surface: a regression that ONLY broke the disabled-mode gate
    // would surface here as a successful recovery → wrong gRPC code.
    let tempdir = tempfile::tempdir().expect("tempdir for disabled-mode probe store");
    let _probe_store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("probe LocalDiskCheckpointStore must construct cleanly"),
    );

    // Snapshot the recovery-attempts counter pre-call. Fresh registry
    // = all cells start at zero, but snapshot the value explicitly so
    // the assertion below is robust to any future Metrics::new wiring
    // change that pre-seeds a cell.
    let restored_pre = recovery_metric(&metrics, "restored");
    let not_recoverable_pre = recovery_metric(&metrics, "not_recoverable");
    let error_pre = recovery_metric(&metrics, "error");

    // Wipe `obj:*` — same data-loss surface as test 1.
    wipe_layer_state(&tv, object_id).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, "e2e-stat-disabled-layer"))
        .await
        .expect_err(
            "disabled-mode StatLayer after wipe MUST surface a gRPC error; \
             Ok here means the recovery branch fired despite the disabled gate",
        );

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "disabled-mode StatLayer MUST surface NotFound (the original header-read \
         NotFound that the disabled short-circuit lets through), got: {}",
        status.code()
    );

    // Counter-total assertion: ZERO increments across all three result
    // labels. The disabled short-circuit returns `NotRecoverable`
    // WITHOUT touching the `with_label_values(...).inc()` call site —
    // a regression that incremented `not_recoverable` from the short-
    // circuit path would tick the cell and fail here.
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        restored_pre,
        "disabled-mode hook MUST NOT increment the `restored` recovery cell"
    );
    assert_eq!(
        recovery_metric(&metrics, "not_recoverable"),
        not_recoverable_pre,
        "disabled-mode hook MUST NOT increment the `not_recoverable` cell — \
         the short-circuit returns NotRecoverable without recording it as an attempt"
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        error_pre,
        "disabled-mode hook MUST NOT increment the `error` cell"
    );

    // Per-`object_id` admission map MUST stay empty — the disabled
    // short-circuit also skips the admission DashMap acquire, which
    // is the other invariant the disabled-mode tests in
    // `recovery_hook_test.rs` pin.
    assert_eq!(
        svc.recovery.admission_len(),
        0,
        "disabled-mode hook MUST NOT touch the per-object_id admission map"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 (b.f4t sub-task 3) — a structurally-PARTIAL header drives a REAL
// restore, not a no-op that claims one.
// ─────────────────────────────────────────────────────────────────────────────

/// A header that is PRESENT but missing `created_by` must drive a real restore
/// through `StatLayer`, repairing the header and answering `Ok`.
///
/// # The defect
///
/// `stat_layer_for_object_id` has always answered `NotFound` on a header
/// missing `created_by` — a `LayerStat` with empty `created_by` would tell a
/// client the layer exists with garbage metadata. `StatLayer` routes that
/// `NotFound` into the recovery hook, whose presence probe was a bare
/// `EXISTS obj:hdr`. So the hook saw the key, reported the layer already
/// restored, ticked
/// `live_edit_checkpoint_recovery_attempts_total{result="restored"}` for a
/// restore that never ran, and handed the handler's one retry the identical
/// error. The layer stayed unreadable and the metric said it had been fixed.
///
/// # Why this fixture is not the same as test 1's
///
/// Test 1 wipes `obj:*` outright — header absent, which the bare `EXISTS`
/// handled correctly. This one **damages** the header instead: `HDEL created_by`
/// and nothing else. The distinction is the whole point of the sub-task, and it
/// is invisible to any fixture that deletes the key.
///
/// # What is asserted, and why each half is needed
///
/// - `Ok` from `StatLayer`, with a non-empty `created_by_user_name`: the RPC
///   the client issued now succeeds, and it succeeds with real metadata.
/// - `restored` ticks **exactly once**: the counter now describes a restore
///   that actually ran. Pre-fix it also read 1 — from a no-op — so the count
///   alone proves nothing; it is the count TOGETHER with the repaired header
///   that separates the two worlds.
/// - `last_checkpoint_seq` present on the header: only `write_object`'s
///   checkpoint-cursor HSET writes that column, so its presence is positive
///   evidence the materializer ran. This is the assertion that a no-op restore
///   cannot fake.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_partial_header_drives_a_real_restore_and_repairs_it() {
    let harness = setup_checkpoint_harness(1).await;

    let layer_name = "e2e-stat-partial-layer";
    let (session_id, object_id) = create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "e2e-stat-partial-session",
        layer_name,
    )
    .await;

    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Damage rather than delete: drop ONE structural field. The header key
    // survives, which is exactly what the pre-fix `EXISTS` probe called
    // "already restored".
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let removed: i64 = {
        use fred::interfaces::HashesInterface;
        harness
            .tv
            .client()
            .pool
            .hdel(&header_key, "created_by")
            .await
            .expect("HDEL created_by must succeed")
    };
    assert_eq!(
        removed, 1,
        "fixture precondition: `created_by` must have been present to remove — a zero here \
         means the header never carried it and the test damages nothing"
    );
    // Non-vacuity: the header key itself must still exist, or this is test 1's
    // absent-header fixture wearing a different name.
    assert!(
        obj_header_exists_via(&harness.tv, object_id).await,
        "fixture precondition: obj:hdr must still be PRESENT — the partial-header case is the \
         one a bare EXISTS probe got wrong, and an absent header does not exercise it"
    );

    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    let response = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "StatLayer against a structurally-partial header MUST restore and return Ok. A \
             NotFound here is the pre-b.f4t behaviour: the hook reported `Restored` without \
             restoring, and the handler's one retry re-observed the same broken header",
        );

    let stat = response.into_inner();
    assert!(
        !stat.created_by_user_name.is_empty(),
        "the restore must have repaired `created_by` — an empty string means the header read \
         succeeded on a header the restore never rewrote"
    );

    assert!(
        crate::common::read_header_u64(&harness.tv, object_id, "last_checkpoint_seq")
            .await
            .is_some(),
        "`last_checkpoint_seq` must be present on the repaired header. Only `write_object`'s \
         checkpoint-cursor HSET writes that column, so this is the assertion a no-op restore \
         cannot fake — and the one that separates 'the metric ticked' from 'a restore ran'"
    );

    assert_eq!(
        recovery_metric(&harness.metrics, "restored") - restored_pre,
        1,
        "exactly ONE recovery attempt must have reported `restored` — not zero (the branch \
         never fired) and not two (the handler retried recovery past the at-most-once latch)"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        0,
        "the checkpoint blob landed before the damage, so recovery MUST succeed"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error") - error_pre,
        0,
        "a healthy LocalDisk tempdir has no transient failures"
    );

    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

/// `EXISTS le:{object_id}:obj:hdr` against the harness fixture.
///
/// Local rather than shared: the sibling `obj_header_exists` in
/// `e2e_live_mapping_guard_test.rs` takes that file's own harness type.
async fn obj_header_exists_via(tv: &TestValkey, object_id: u64) -> bool {
    use fred::interfaces::KeysInterface;
    let n: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("EXISTS obj:hdr must succeed");
    n > 0
}
