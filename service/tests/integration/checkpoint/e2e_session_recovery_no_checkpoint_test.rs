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

//! End-to-end session-recovery integration tests for the `NoCheckpoint`
//! arm (Subtask `t3.fgj.7v.fp.kv` under Task `t2.fgj.7v.fp`, Epic
//! `t1.fgj.7v`, Bee `b.fgj`).
//!
//! Task 6's `try_recover_session` returns
//! [`SessionRecoveryOutcome::NoCheckpoint`](live_edit::checkpoint::SessionRecoveryOutcome::NoCheckpoint)
//! when the adapter has no checkpoint for the `session_id`. This is the
//! "session genuinely never existed → NotFound" branch, NOT the
//! "session existed but Valkey lost its keys → recover-and-proceed"
//! branch covered by sibling Subtask `t3.fgj.7v.fp.xx`'s
//! [`super::e2e_session_recovery_per_handler_test`].
//!
//! The wiring contract under test:
//!
//! - `SessionBasePathCache::get_base_path` consults the
//!   [`SessionRecoveryHook`](live_edit::checkpoint::SessionRecoveryHook)
//!   on a `NotFound` from `session_store.get_base_path`.
//! - On [`SessionRecoveryOutcome::NoCheckpoint`] the cache surfaces the
//!   original `ServiceError::NotFound` verbatim
//!   (`src/grpc/session_cache.rs:200`), which maps to gRPC
//!   `Code::NotFound` (`src/error.rs:195`).
//!
//! This file exercises four session-resolving handlers — `UpdateLayer`,
//! `CreateLayer`, `DeleteLayer`, `DeleteSession` — through their real
//! RPC entry points, each against a session whose Valkey state AND
//! adapter blob have both been deliberately removed. Each test asserts:
//!
//! 1. The handler returns `Err(Status)` with `Code::NotFound`. A
//!    regression mapping `NoCheckpoint` to `Internal` would cause
//!    client retry storms; mapping it to silent success would surface
//!    as either a panic deeper in the handler or — worse — a half-state
//!    where the new operation lands against a non-existent session.
//! 2. The
//!    `live_edit_checkpoint_recovery_attempts_total{scope="session",
//!    result="not_recoverable"}` counter incremented by EXACTLY 1.
//!    `result="restored"` / `result="error"` / `result="already_present"`
//!    stay at their pre-call snapshots — the apply path never ran.
//! 3. Post-call Valkey state stays empty for `le:{sid}*`. A regression
//!    that started a partial apply before discovering the missing
//!    checkpoint would leak phantom session-meta or layer-mapping keys.
//!
//! `ListSessions` is intentionally NOT in this file — its `NoCheckpoint`
//! behavior runs through `reconcile_ghost_members` (a different choke
//! point) and is covered by the ListSessions ghost-walk wiring Subtask
//! (`t3.fgj.7v.fp.ka`).
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::{ClientLike, KeysInterface};
use fred::types::scan::ScanType;
use futures::StreamExt;
use tonic::Code;

use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest};
use live_edit::grpc::session::{handle_create_session, handle_delete_session};
use live_edit::observability::metrics::SharedMetrics;

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

const SESSION_BASE_PATH: &str = "/stage";
const SESSION_NAME_UPDATE_LAYER: &str = "e2e-no-checkpoint-update-layer-session";
const SESSION_NAME_CREATE_LAYER: &str = "e2e-no-checkpoint-create-layer-session";
const SESSION_NAME_DELETE_LAYER: &str = "e2e-no-checkpoint-delete-layer-session";
const SESSION_NAME_DELETE_SESSION: &str = "e2e-no-checkpoint-delete-session-session";
const LAYER_NAME_ORIGINAL: &str = "original-layer";
const LAYER_NAME_NEW_CREATE: &str = "new-layer-after-recovery-attempt";
const UPDATE_CLIENT_ID: &str = "e2e-no-checkpoint-client";

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

/// Probe the live label-cell counter value for a given outcome label on
/// the `live_edit_checkpoint_recovery_attempts_total{result, scope="session",
/// service}` metric. `scope="session"` (NOT `"layer"`) is the load-bearing
/// difference vs. the layer-scope recovery tests — session-recovery and
/// layer-recovery tick disjoint metric cells, so the per-handler tests
/// must inspect the session scope to assert their invariants.
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

/// Snapshot of the four session-scope recovery-metric cells captured
/// before the handler-under-test is dispatched. Each test asserts the
/// `not_recoverable` cell incremented by EXACTLY 1 and every other cell
/// stayed at its pre-call value.
struct RecoveryMetricBaseline {
    restored_pre: u64,
    not_recoverable_pre: u64,
    error_pre: u64,
    already_present_pre: u64,
}

fn capture_baseline(metrics: &SharedMetrics) -> RecoveryMetricBaseline {
    RecoveryMetricBaseline {
        restored_pre: session_recovery_metric(metrics, "restored"),
        not_recoverable_pre: session_recovery_metric(metrics, "not_recoverable"),
        error_pre: session_recovery_metric(metrics, "error"),
        already_present_pre: session_recovery_metric(metrics, "already_present"),
    }
}

// ── Composite harness driver — CreateSession + CreateLayer + wait ────────────
//
// Every test in this file starts from the same baseline shape: bring up
// the harness, CreateSession, CreateLayer, wait for the session-checkpoint
// blob to land. The wipe step (Valkey + adapter) and the handler under
// test are what differ.

struct HarnessBaseline {
    session_id: String,
    object_id: u64,
}

/// CreateSession + CreateLayer + wait_for_session_checkpoint.
///
/// CreateSession goes through `handle_create_session` (the real gRPC
/// entry point) so the wiring through `LiveEditService` is exercised
/// end-to-end. `SessionStore::create`'s success arm enqueues a
/// fire-and-forget session-checkpoint write through the dispatcher;
/// CreateLayer also enqueues one. Both write the same `session_id` —
/// `wait_for_session_checkpoint` only needs to see one blob land.
///
/// The blob existing on disk is what justifies the subsequent
/// `delete_session_checkpoint` call: if the wait short-circuited (e.g.
/// the writer was unwired), the deliberate-delete step would silently
/// no-op against a state that was never populated, and the test would
/// not faithfully exercise the `NoCheckpoint` arm — it would exercise
/// "checkpoint never written" instead of "checkpoint deleted between
/// CreateSession and the recovery attempt," which is the production
/// failure surface SR-2.7's `NoCheckpoint` arm is designed to defend
/// against (a stale RPC reaching a recovered replica after the session
/// was deleted; an adapter blip that dropped the write).
async fn create_session_and_layer_and_wait_for_checkpoint(
    harness: &CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> HarnessBaseline {
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: session_name.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let create_layer_resp = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for harness setup");
    let object_id = create_layer_resp.into_inner().object_id;
    assert!(object_id > 0, "CreateLayer must mint a non-zero object_id");

    // Wait for the dispatcher's spawned put-tasks to land the blob on
    // the adapter AND for the writer's TaskTracker to drain. Both
    // CreateSession and CreateLayer enqueue a fire-and-forget put-task
    // each; observing one blob on disk does NOT prove the second task is
    // also finished. Without the quiesced wait, the subsequent
    // `delete_session_checkpoint` could land first and be silently
    // overwritten by the late put — defeating the test's `NoCheckpoint`
    // arm and turning it into an accidental `Recovered`/`AlreadyPresent`
    // test. Under fast Valkey (sub-millisecond RTT), the race fires
    // deterministically; under slow Valkey it was masked by tasks
    // completing before the polling loop's first 50 ms tick.
    wait_for_session_checkpoint_quiesced(harness, &session_id, 10).await;

    HarnessBaseline {
        session_id,
        object_id,
    }
}

/// Drop the cached `base_path` entry for `session_id` on the harness's
/// `SessionBasePathCache`.
///
/// Load-bearing: the baseline CreateLayer call populates the cache with
/// the live `base_path`, and the cache's 2-second positive TTL is long
/// enough that the post-wipe handler call inside the same test would
/// HIT the cache and never consult `session_store.get_base_path` —
/// bypassing the recovery branch entirely. Without this explicit
/// invalidation, every test in this file would silently observe
/// `not_recoverable == 0` (recovery never fired) AND a successful
/// handler call (the cache HIT returned the pre-wipe `base_path`).
///
/// The corresponding sibling-file helper invalidates for the same
/// reason; per the Subtask description, lifting this would mean moving
/// it to `helpers.rs`, but the sibling's helper is co-located in its
/// own test file and we keep the same shape here for symmetry. A future
/// refactor can lift both at once.
fn invalidate_session_cache(harness: &CheckpointHarness, session_id: &str) {
    harness.svc.session_base_path_cache.invalidate(session_id);
}

/// Confirm that the wipe + delete left the recovery hook's two
/// observation surfaces in the "no session anywhere" state we need.
///
/// Run AFTER `wipe_session_state` + `delete_session_checkpoint` to
/// catch test-infrastructure regressions before the assertion phase
/// obscures them. A surviving `le:{sid}:sess` key would mean the
/// SCAN pattern in `wipe_session_state` missed it (and the next
/// handler call's first `get_base_path` would succeed without going
/// near recovery — test-infrastructure regression). A surviving
/// adapter blob would mean the deliberate `delete_session_checkpoint`
/// failed silently (and recovery would fall into the `Recovered` arm
/// instead of `NoCheckpoint` — different test-infrastructure regression).
async fn assert_session_fully_absent(harness: &CheckpointHarness, session_id: &str) {
    let session_meta_key = harness.tv.key_schema().session_meta(session_id);
    let exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS sess probe must succeed against healthy Valkey");
    assert!(
        !exists,
        "wipe_session_state must UNLINK le:{{session_id}}:sess; surviving key here \
         means the scan-pattern missed it and the handler call would never reach \
         recovery"
    );

    let blob = harness
        .store
        .get_session_checkpoint(session_id)
        .await
        .expect("get_session_checkpoint must succeed against the LocalDisk adapter");
    assert!(
        blob.is_none(),
        "delete_session_checkpoint must remove the session-checkpoint blob; a \
         surviving blob means the test would exercise the `Recovered`/`AlreadyPresent` \
         arm rather than the `NoCheckpoint` arm under test"
    );
}

/// Post-call assertion shared by every test in this file: the
/// `not_recoverable` cell incremented by EXACTLY 1; every other
/// session-scope outcome cell stayed at its pre-call snapshot.
///
/// `not_recoverable == 1` pins the wiring: a regression that bypassed
/// recovery entirely (e.g., cache HIT before the wipe took effect)
/// would show `not_recoverable == 0`. `restored == 0` / `error == 0` /
/// `already_present == 0` pin that the apply pipeline never ran and no
/// transient failure was injected.
fn assert_exactly_one_not_recoverable(
    harness: &CheckpointHarness,
    baseline: &RecoveryMetricBaseline,
) {
    assert_eq!(
        session_recovery_metric(&harness.metrics, "not_recoverable") - baseline.not_recoverable_pre,
        1,
        "exactly ONE session-recovery attempt must have reported `not_recoverable` \
         (the adapter returned `NoCheckpoint` after the deliberate \
         `delete_session_checkpoint`); not_recoverable_pre={}",
        baseline.not_recoverable_pre
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "restored"),
        baseline.restored_pre,
        "post-call `restored` MUST equal pre-call snapshot — the adapter returned \
         `NoCheckpoint` so the apply pipeline never ran. A non-zero increment here \
         means a phantom apply somehow landed."
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "error"),
        baseline.error_pre,
        "post-call `error` MUST equal pre-call snapshot — no transient backend \
         failure was injected. A non-zero increment here means the LocalDisk \
         adapter spuriously surfaced a Transient/Permanent error on a missing \
         session blob (it should return Ok(None) per the trait contract)."
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "already_present"),
        baseline.already_present_pre,
        "post-call `already_present` MUST equal pre-call snapshot — the wipe drove \
         the pre-acquire EXISTS check to false, so the short-circuit cell stays \
         untouched."
    );
}

/// Confirm that Valkey state is still empty for `le:{sid}*` after the
/// failed handler call.
///
/// A regression that started a partial apply before discovering the
/// missing checkpoint — for example, writing the session-meta hash
/// from the in-flight context before the adapter `get` resolved —
/// would leak a phantom key here. Equally, a regression that wrote
/// part of the handler's normal flow (e.g. minted a fresh
/// layer-mapping under the post-recovery branch before realizing the
/// session never existed) would surface as a stray key.
///
/// SCAN + EXISTS check; the `{session_id}` hash tag keeps every
/// matching key on one Cluster slot, so the SCAN is slot-safe.
async fn assert_valkey_still_empty_for_session(harness: &CheckpointHarness, session_id: &str) {
    let pattern = format!("{}{{{session_id}}}*", harness.tv.key_schema().prefix());
    let client = harness.tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
        Box::pin(client.scan_cluster_buffered(pattern.clone(), Some(100), None::<ScanType>))
    } else {
        Box::pin(client.scan_buffered(pattern.clone(), Some(100), None::<ScanType>))
    };
    let mut found = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(key) = item {
            // Convert to a printable string for the diagnostic. Some keys
            // may carry non-UTF-8 bytes in pathological cases; lossy
            // decode is fine here because the assert message is purely
            // diagnostic.
            let key_str = String::from_utf8_lossy(key.as_bytes()).into_owned();
            found.push(key_str);
        }
    }
    assert!(
        found.is_empty(),
        "post-`NoCheckpoint` handler call MUST leave Valkey state empty for \
         le:{{{session_id}}}*; surviving keys ({found:?}) mean the handler ran \
         a partial write before discovering the missing checkpoint. The \
         `NoCheckpoint` contract is `NotFound` AND no side effects."
    );
}

// =============================================================================
// Test 1 — UpdateLayer surfaces NotFound on NoCheckpoint.
// =============================================================================

/// Pin the UpdateLayer choke point's `NoCheckpoint` → `NotFound`
/// surface.
///
/// Why each step is load-bearing:
///
/// - The wipe + adapter delete drive both the cache miss AND the
///   recovery hook's `Ok(None)` from `get_session_checkpoint`. Without
///   the adapter delete, recovery would succeed and the test would
///   pass for the WRONG reason (testing the `Recovered` arm instead).
/// - UpdateLayer is exercised against the ORIGINAL `object_id` from
///   the pre-wipe CreateLayer. A regression that erroneously
///   short-circuited on the object_id check before consulting
///   `get_base_path` would surface as a `FailedPrecondition` instead
///   of `NotFound`, which would also fail the assertion. We pin the
///   `NotFound` surface specifically.
/// - The post-call Valkey-empty assertion catches a regression where
///   the apply path started writing before discovering the missing
///   checkpoint (e.g., re-materializing session-meta from the in-flight
///   request, then realizing too late).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_checkpoint_surfaces_not_found_on_update_layer() {
    let harness = setup_checkpoint_harness(1).await;

    let HarnessBaseline {
        session_id,
        object_id,
    } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_UPDATE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    // Wipe both surfaces: Valkey session-scoped keys AND the adapter blob.
    // The deliberate `delete_session_checkpoint` is what distinguishes this
    // test from the sibling `fp.xx` per-handler recovery test (where the
    // adapter blob is left in place so recovery succeeds).
    wipe_session_state(&harness.tv, &session_id).await;
    harness
        .store
        .delete_session_checkpoint(&session_id)
        .await
        .expect("delete_session_checkpoint on LocalDisk must succeed");
    assert_session_fully_absent(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    let baseline = capture_baseline(&harness.metrics);

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_update(vec![
        make_update_header(
            &session_id,
            LAYER_NAME_ORIGINAL,
            object_id,
            1,
            UPDATE_CLIENT_ID,
        ),
        make_update_data(ompf_bytes),
    ]);
    let status = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect_err(
            "UpdateLayer MUST surface an error when the session is fully absent \
             (both Valkey state and the adapter checkpoint blob); a success here \
             means the cache funnel silently passed through to a non-existent \
             session, or recovery somehow restored from nothing",
        );
    assert_eq!(
        status.code(),
        Code::NotFound,
        "UpdateLayer on a NoCheckpoint session MUST surface `Code::NotFound`. \
         Code::Internal would cause client retry storms; Code::Unavailable would \
         hint at a transient-backend-outage which is misleading for a genuinely \
         missing session. Got code={:?}, message={:?}.",
        status.code(),
        status.message()
    );

    assert_exactly_one_not_recoverable(&harness, &baseline);
    assert_valkey_still_empty_for_session(&harness, &session_id).await;

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Test 2 — CreateLayer surfaces NotFound on NoCheckpoint.
// =============================================================================

/// Pin the CreateLayer choke point's `NoCheckpoint` → `NotFound`
/// surface.
///
/// CreateLayer targets a NEW layer name (not the original) so the test
/// exercises CreateLayer's create-fresh-mapping path — if a regression
/// surfaced `NoCheckpoint` as silent success, the handler would happily
/// mint a fresh `object_id` against a non-existent session and store
/// the layer-mapping under `le:{sid}:layer:{hex(new)}`. The post-call
/// Valkey-empty assertion would then fire on that stray mapping key,
/// pinning the silent-success regression even if it somehow returned
/// `Ok(...)` to the client.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_checkpoint_surfaces_not_found_on_create_layer() {
    let harness = setup_checkpoint_harness(1).await;

    let HarnessBaseline { session_id, .. } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_CREATE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    wipe_session_state(&harness.tv, &session_id).await;
    harness
        .store
        .delete_session_checkpoint(&session_id)
        .await
        .expect("delete_session_checkpoint on LocalDisk must succeed");
    assert_session_fully_absent(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    let baseline = capture_baseline(&harness.metrics);

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME_NEW_CREATE),
        make_create_data(ompf_bytes),
    ]);
    let status = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect_err(
            "CreateLayer MUST surface an error when the session is fully absent; \
             a success here means the cache funnel silently minted a layer-mapping \
             against a non-existent session — the post-Valkey-empty assertion would \
             catch the resulting orphan but we want the handler-level NotFound to \
             surface first",
        );
    assert_eq!(
        status.code(),
        Code::NotFound,
        "CreateLayer on a NoCheckpoint session MUST surface `Code::NotFound`. \
         Got code={:?}, message={:?}.",
        status.code(),
        status.message()
    );

    assert_exactly_one_not_recoverable(&harness, &baseline);
    assert_valkey_still_empty_for_session(&harness, &session_id).await;

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Test 3 — DeleteLayer surfaces NotFound on NoCheckpoint.
// =============================================================================

/// Pin the DeleteLayer choke point's `NoCheckpoint` → `NotFound`
/// surface.
///
/// DeleteLayer targets the original layer name (which used to exist
/// pre-wipe). A regression that surfaced `NoCheckpoint` as silent
/// success would either return `Ok(...)` with no side effects (the
/// session-empty assertion would still pass) or — worse — start a
/// cascade that touched the layer-mapping table from an in-flight
/// recovery context. The Code::NotFound assertion catches the
/// silent-success regression at the client-visible surface.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_checkpoint_surfaces_not_found_on_delete_layer() {
    let harness = setup_checkpoint_harness(1).await;

    let HarnessBaseline { session_id, .. } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    wipe_session_state(&harness.tv, &session_id).await;
    harness
        .store
        .delete_session_checkpoint(&session_id)
        .await
        .expect("delete_session_checkpoint on LocalDisk must succeed");
    assert_session_fully_absent(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    let baseline = capture_baseline(&harness.metrics);

    let status = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME_ORIGINAL.to_owned(),
        }),
    )
    .await
    .expect_err(
        "DeleteLayer MUST surface an error when the session is fully absent; \
         a success here means the cache funnel let the cascade run against a \
         non-existent session",
    );
    assert_eq!(
        status.code(),
        Code::NotFound,
        "DeleteLayer on a NoCheckpoint session MUST surface `Code::NotFound`. \
         Got code={:?}, message={:?}.",
        status.code(),
        status.message()
    );

    assert_exactly_one_not_recoverable(&harness, &baseline);
    assert_valkey_still_empty_for_session(&harness, &session_id).await;

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Test 4 — DeleteSession surfaces NotFound on NoCheckpoint.
// =============================================================================

/// Pin the DeleteSession choke point's `NoCheckpoint` → `NotFound`
/// surface.
///
/// DeleteSession is the most stress-testing case: a regression that
/// surfaced `NoCheckpoint` as silent success could plausibly return
/// `Ok(...)` and tick a "session deleted" log line — semantically
/// closest to "absent" — even though no session existed. Pinning
/// `Code::NotFound` at the client-visible surface forces the cache
/// funnel to distinguish "session never existed" from
/// "session existed and now is gone" through the actual gRPC contract,
/// matching the documented behavior in `src/grpc/session_cache.rs:200`.
///
/// Note: unlike the sibling `recovery_fires_on_delete_session_then_deletes`
/// test, there is NO `wait_for_session_checkpoint_absent` call here.
/// DeleteSession never reached the layer-store cascade — recovery
/// aborted at the cache funnel with `NoCheckpoint` BEFORE the
/// `SessionStore::delete` lifecycle hook ran. So no
/// `enqueue_session_checkpoint_delete` was issued; the adapter blob
/// stays exactly as we left it (already deleted by this test's setup).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_checkpoint_surfaces_not_found_on_delete_session() {
    let harness = setup_checkpoint_harness(1).await;

    let HarnessBaseline { session_id, .. } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_SESSION,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    wipe_session_state(&harness.tv, &session_id).await;
    harness
        .store
        .delete_session_checkpoint(&session_id)
        .await
        .expect("delete_session_checkpoint on LocalDisk must succeed");
    assert_session_fully_absent(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    let baseline = capture_baseline(&harness.metrics);

    let status = handle_delete_session(
        &harness.svc,
        make_unary_token_request(DeleteSessionRequest {
            session_id: session_id.clone(),
        }),
    )
    .await
    .expect_err(
        "DeleteSession MUST surface an error when the session is fully absent; \
         a success here would semantically conflate \"session never existed\" \
         with \"session existed and is now gone\", which is the silent-bug shape \
         the cache funnel's `NoCheckpoint` → `NotFound` mapping is designed to \
         prevent",
    );
    assert_eq!(
        status.code(),
        Code::NotFound,
        "DeleteSession on a NoCheckpoint session MUST surface `Code::NotFound`. \
         Got code={:?}, message={:?}.",
        status.code(),
        status.message()
    );

    assert_exactly_one_not_recoverable(&harness, &baseline);
    assert_valkey_still_empty_for_session(&harness, &session_id).await;

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
