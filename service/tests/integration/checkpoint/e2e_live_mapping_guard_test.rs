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

//! End-to-end tests pinning the SR-2.4 live-mapping guard inside
//! `apply_checkpoint` — Epic AC #10 (Subtask `t3.fgj.ya.wg.rh` under Task
//! `t2.fgj.ya.wg`, Epic `t1.fgj.ya`, Bee `b.fgj`).
//!
//! The contract these tests pin is:
//!
//!   "When a checkpoint blob exists in the adapter for some `object_id`
//!    but no live `le:{sid}:layer:{hex(name)}` mapping resolves to that
//!    `object_id` anymore, `apply_checkpoint` MUST surface
//!    `CheckpointError::NotFound`. The recovery hook MUST translate that
//!    into `RecoveryOutcome::NotRecoverable`, increment the
//!    `live_edit_checkpoint_recovery_attempts_total{result=\"not_recoverable\"}`
//!    counter, and MUST NOT write anything to Valkey. Downstream RPCs MUST
//!    surface `Code::NotFound` to the client."
//!
//! ## The guard has no shipped producer of EMPTY identity any more (b.hka)
//!
//! Worth stating because it is the thing most likely to be misread here. The
//! apply-path guard checks two things: that the blob's `session_id` /
//! `layer_name` are non-empty, and that the live mapping they name resolves to
//! this `object_id`. b.hka closed off every producer of the first: the
//! assembler declines to assemble a generation whose `obj:hdr` vanished or
//! carries empty identity columns, `validate_generation_identity` refuses an
//! identity-empty put at every adapter, and the adapters no longer synthesize
//! an identity-empty placeholder meta for a missing companion.
//!
//! So the emptiness half is a defensive fold with no reachable input, and it is
//! covered where it is now enforced — `conformance::case_39`, the storage-api
//! zero-request pins, and `assembler_test`'s decline cases. What THIS file
//! covers is the half that remains live and is not defensive at all: a blob
//! whose identity is fully populated and **genuinely mismatched** against the
//! live keyspace. Both scenarios below are of that shape, which is why neither
//! needed changing when the empty-identity producers went away.
//!
//! Two scenarios trigger this guard. Both are exercised here:
//!
//! - **DeleteLayer-checkpoint-delete race.** DeleteLayer cleared both the
//!   live `obj:*` keys and the layer mapping, but a checkpoint blob still
//!   sits on disk because the periodic loop landed a generation just
//!   before the delete and the post-delete adapter sweep has not yet run.
//!   Restoring would resurrect a phantom layer.
//!
//! - **CreateLayer-overwrite during state-loss.** Valkey lost `obj:*` for
//!   `object_id_A` (failover, persistence wipe, etc.) AND the layer
//!   mapping was nuked alongside it. A subsequent CreateLayer on the same
//!   `(session_id, layer_name)` minted `object_id_B`; the live mapping
//!   now resolves to B. The checkpoint blob for the stale A is still on
//!   disk; restoring it would resurrect a phantom object disjoint from
//!   the current live mapping.
//!
//! Why we drive these through the recovery hook directly (not through a
//! gRPC handler):
//!
//! The handler-level recovery branch (`handle_stat_layer` step 5 / SDD
//! SR-2.4) is gated on the mapping resolving first. In both scenarios
//! above the live mapping does NOT resolve to `object_id_A`, so the
//! gRPC-level path returns `Code::NotFound` BEFORE the recovery branch
//! even fires — exercising the handler against the deleted layer-name
//! only pins the no-recovery NotFound surfacing, not the guard rejection.
//! To pin the guard rejection itself (the `not_recoverable` metric
//! increment + no-Valkey-side-effects invariant), the tests invoke
//! `RecoveryHook::try_restore_layer` directly with the stale
//! `object_id_A`. This is the same pattern the unit test
//! `try_restore_layer_not_recoverable_on_apply_guard_reject` in
//! `recovery_hook_test.rs` uses; at the E2E level we additionally drive
//! the whole stack (real LocalDisk store, real leader-elected checkpoint
//! loop, real `LayerStore` against a live Valkey) so a regression
//! anywhere in the apply pipeline surfaces here.
//!
//! Each test ALSO issues a `StatLayer` RPC against the now-deleted layer
//! name to pin the orthogonal "missing mapping surfaces NotFound directly
//! without recovery firing" invariant — the gRPC-level contract the
//! orchestrator's note flagged as the right RPC trigger for this Subtask
//! (UpdateLayer's recovery wiring has a known bug an engineer is
//! concurrently fixing on a sibling Subtask).
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`. Under
//! the project sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

use std::time::Duration;

use fred::interfaces::KeysInterface;
use tonic::Request;

use live_edit::checkpoint::RecoveryOutcome;
use live_edit::grpc::layer::{handle_delete_layer, handle_stat_layer};
use live_edit::grpc::proto::{DeleteLayerRequest, StatLayerRequest};
use live_edit::grpc::BearerToken;

use super::helpers::*;
// `recovery_metric` is the shared copy in `tests/common`: it pins the same
// `{result, scope="layer", service}` cell this file used to spell locally.
use crate::common::recovery_metric;
use crate::grpc::helpers::create_session_and_layer;

/// Build a `StatLayerRequest` with a stub bearer token in the extensions
/// map — same shape `e2e_stat_layer_recovery_test.rs` uses.
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

/// Build a `DeleteLayerRequest` with a stub bearer token.
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

/// Probe whether any `le:{object_id}:obj:hdr` key exists in Valkey.
/// Used by both tests to pin the "no side effects" invariant — a
/// regression that bypassed the guard and wrote header bytes during the
/// rejected apply would flip this to `true`.
async fn obj_header_exists(harness: &CheckpointHarness, object_id: u64) -> bool {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    harness
        .tv
        .client()
        .pool
        .exists(&header_key)
        .await
        .expect("EXISTS obj:hdr probe must succeed against healthy Valkey")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — DeleteLayer-checkpoint-delete race.
// ─────────────────────────────────────────────────────────────────────────────

/// Pins the SR-2.4 guard's rejection branch for the
/// DeleteLayer-checkpoint-delete race:
///
/// 1. Stand up the harness (real `LocalDiskCheckpointStore`, real leader-
///    elected checkpoint loop, real `RecoveryHook`).
/// 2. CreateSession + CreateLayer → mint `object_id_A`.
/// 3. Wait until the leader-elected loop lands at least one generation
///    for `object_id_A` on disk. Capture the handle as proof the
///    checkpoint blob landed.
/// 4. Issue `DeleteLayer` via the real handler — this DELs the layer
///    mapping `le:{sid}:layer:{hex(name)}` AND UNLINKs every
///    `le:{object_id_A}:obj:*` key. The checkpoint blob remains on disk
///    (the LocalDisk adapter has no post-delete sweep; that is the GC
///    loop's job, which is not running in this harness).
/// 5. Confirm the checkpoint blob is still listable — proves the test is
///    genuinely exercising the "blob exists but mapping does not" race
///    rather than passing vacuously on an empty store.
/// 6. Issue `StatLayer` against the deleted layer-name. The handler's
///    mapping-resolve gate (`get_layer_mapping`) returns `None` and the
///    handler surfaces `Code::NotFound` directly WITHOUT firing recovery
///    (per `handle_stat_layer` lines 554-561). This pins the gRPC-level
///    contract that a missing mapping is a terminal NotFound, not a
///    recovery trigger.
/// 7. Directly invoke `recovery.try_restore_layer(object_id_A)` to drive
///    the guard. The hook fetches the surviving checkpoint blob, calls
///    `apply_checkpoint`, which runs `live_mapping_resolves_blob_identity`
///    — the blob's own layer mapping is gone → `CheckpointError::NotFound` →
///    `RecoveryOutcome::NotRecoverable`. The `not_recoverable` cell
///    increments exactly once; `restored` and `error` stay at zero.
/// 8. Assert no `obj:hdr` was written for `object_id_A`. A regression
///    that bypassed the guard would write the header before the
///    rejection bubble-up, flipping this to `true`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_rejects_when_no_live_mapping_after_delete_layer() {
    let harness = setup_checkpoint_harness(1).await;
    let layer_name = "e2e-live-mapping-guard-delete-layer";
    let session_name = "e2e-live-mapping-guard-delete-session";

    // CreateSession + CreateLayer via the real handler — the create
    // handler installs the layer mapping and writes `obj:hdr` with the
    // creator identity / timestamps the periodic loop reads when it
    // assembles the checkpoint blob.
    let (session_id, object_id) =
        create_session_and_layer(&harness.tv, &harness.svc, session_name, layer_name).await;

    // Wait until the leader-elected loop has landed at least one
    // generation. With `tick_secs=1` + `interval=quiesce=0` (set by
    // `setup_checkpoint_harness`) the loop produces a generation within
    // a single tick; 10 s is comfortably above CI jitter on the tick.
    let pre_delete_handle = wait_for_checkpoint(&harness.store, object_id, 10).await;
    assert_eq!(
        pre_delete_handle.object_id, object_id,
        "wait_for_checkpoint must hand back a generation for the just-created object"
    );

    // Snapshot the recovery-metric cells pre-RPC. A fresh harness starts
    // every cell at zero, but the snapshot pattern is robust to any
    // future Metrics::new wiring change that pre-seeds a cell.
    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    // Issue DeleteLayer — clears mapping AND obj:* in one cascade.
    handle_delete_layer(&harness.svc, delete_request(&session_id, layer_name))
        .await
        .expect("DeleteLayer against an existing layer must succeed");

    // Confirm the checkpoint blob is still in the store — the LocalDisk
    // adapter has no post-DeleteLayer sweep. Without this assertion the
    // rest of the test could vacuously pass on an empty store.
    let post_delete_handles = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed against a healthy LocalDisk root");
    assert!(
        !post_delete_handles.is_empty(),
        "checkpoint blob for object_id={object_id} MUST survive DeleteLayer \
         (the SR-2.4 guard's whole point is the blob/mapping divergence — \
         no surviving blob = no race to test)"
    );

    // Confirm obj:hdr is gone post-DeleteLayer — the wipe surface the
    // guard rejection is supposed to defend against.
    assert!(
        !obj_header_exists(&harness, object_id).await,
        "DeleteLayer MUST UNLINK le:{{object_id}}:obj:hdr; surviving header here \
         means the cascade was incomplete and recovery would silently no-op \
         (a structurally-complete obj:hdr → the recovery hook's is_layer_present \
         short-circuit; since b.f4t that check is an HGETALL requiring `seq` + \
         `created_by`, not a bare EXISTS, but a header DeleteLayer failed to \
         remove is complete by construction and short-circuits either way)"
    );

    // gRPC-level invariant: StatLayer against the deleted layer name
    // surfaces NotFound directly (mapping-resolve gate, no recovery).
    let stat_status = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect_err(
            "StatLayer against a deleted layer-name MUST surface a gRPC error; \
             Ok here means the mapping somehow re-materialized post-DeleteLayer",
        );
    assert_eq!(
        stat_status.code(),
        tonic::Code::NotFound,
        "StatLayer post-DeleteLayer MUST surface NotFound (mapping-resolve gate \
         returns None → handle_stat_layer's ok_or_else fires before the recovery \
         branch), got: {}",
        stat_status.code()
    );

    // The gRPC-level NotFound MUST NOT have fired recovery — the
    // mapping-resolve gate short-circuits before the header-read recovery
    // branch. Confirm all three cells stayed at their pre-call snapshot.
    assert_eq!(
        recovery_metric(&harness.metrics, "restored"),
        restored_pre,
        "StatLayer mapping-NotFound MUST NOT increment `restored` — the recovery \
         branch is downstream of the mapping resolve and never runs in this path"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable"),
        not_recoverable_pre,
        "StatLayer mapping-NotFound MUST NOT increment `not_recoverable` — \
         the recovery branch never fired"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error"),
        error_pre,
        "StatLayer mapping-NotFound MUST NOT increment `error`"
    );

    // Direct guard exercise: drive `try_restore_layer(object_id)` against
    // the stale `object_id`. The recovery hook fetches the surviving
    // checkpoint blob and calls apply, which runs the SR-2.4 guard. Because
    // DeleteLayer cleared the only live mapping that resolved to this
    // object_id, the guard returns false and apply surfaces
    // `CheckpointError::NotFound` → hook surfaces `NotRecoverable`.
    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::NotRecoverable),
        "guard-rejection arm MUST return NotRecoverable; got {outcome:?} \
         (a `Restored` here means the guard let the apply through — phantom \
         resurrection; an `Error(_)` means the LocalDisk read failed against \
         a healthy tempdir, which is a test infrastructure bug)"
    );

    // Exactly ONE `not_recoverable` increment from the direct call.
    // `restored` and `error` cells stay at the pre-call snapshot — the
    // guard rejection is the only branch that fired.
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        1,
        "exactly ONE recovery attempt must have reported `not_recoverable` \
         (apply_checkpoint's SR-2.4 guard rejected: no live mapping resolves \
         to object_id={object_id})"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "restored"),
        restored_pre,
        "`restored` cell MUST stay at the pre-call snapshot — the guard \
         rejected before write_object ran"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error"),
        error_pre,
        "`error` cell MUST stay at the pre-call snapshot — LocalDisk on a \
         healthy tempdir + guard rejection is the NotFound path, not error"
    );

    // Valkey-state invariant: the rejected apply MUST NOT have written
    // anything for `object_id`. A regression that wrote `obj:hdr` before
    // running the guard (or that re-ordered the steps so the write
    // landed in the rejection path) would flip this to `true`.
    assert!(
        !obj_header_exists(&harness, object_id).await,
        "guard-rejected apply MUST NOT write obj:hdr; got header present \
         (apply step ordering regression — the SR-2.4 guard MUST run before \
         any obj:* write)"
    );

    // Graceful shutdown of the leader-elected checkpoint loop mirrors
    // the production drain shape — the tokio runtime would reap the
    // task on harness drop anyway, but cancelling explicitly avoids a
    // stray tracing emit racing the test's tracing subscriber teardown.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — CreateLayer-overwrite during Valkey state-loss.
// ─────────────────────────────────────────────────────────────────────────────

/// Pins the SR-2.4 guard's rejection branch for the CreateLayer-overwrite-
/// during-Valkey-loss race:
///
/// 1. Harness, CreateSession, CreateLayer → mint `object_id_A`.
/// 2. Wait for the leader-elected loop to land a generation for A.
/// 3. Simulate state loss: UNLINK every `le:{object_id_A}:obj:*` key AND
///    DEL the live `le:{sid}:layer:{hex(name)}` mapping. This is the
///    surface a fresh-replica failover or persistence wipe would expose
///    — the `obj:*` keys and the per-session mapping share a fate (both
///    are stored in Valkey).
/// 4. Re-CreateLayer on the SAME `(session_id, layer_name)`. The
///    CreateLayer handler mints a fresh `object_id_B` (INCR on
///    `le:counter:object_id`, distinct from A) and installs a new
///    mapping `le:{sid}:layer:{hex(name)}` → B.
/// 5. Confirm `object_id_A != object_id_B` — a regression that re-used
///    the old `object_id` in this race window would silently invalidate
///    the rest of the test.
/// 6. Confirm the checkpoint blob for A is still in the store (no
///    post-CreateLayer sweep on the LocalDisk adapter).
/// 7. Issue `StatLayer` against `layer_name` — resolves to B's mapping,
///    obj:hdr for B exists, so this returns Ok with B's metadata. Pins
///    the gRPC-level contract that the new layer is reachable through
///    the same name (no phantom recovery on the stale A confused the
///    handler).
/// 8. Directly invoke `recovery.try_restore_layer(object_id_A)`. The
///    hook fetches A's checkpoint blob and calls apply, which runs the
///    SR-2.4 guard. The only live mapping for `(session_id, layer_name)`
///    resolves to B, NOT to A → guard returns false → apply surfaces
///    `CheckpointError::NotFound` → hook surfaces `NotRecoverable`.
/// 9. Assert the `not_recoverable` cell increments once; `restored`
///    and `error` stay at the pre-call snapshot.
/// 10. Assert no `obj:hdr` exists for A — the rejected apply must not
///    have resurrected the stale object's state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_rejects_when_create_layer_overwrite_during_loss() {
    let harness = setup_checkpoint_harness(1).await;
    let layer_name = "e2e-live-mapping-guard-overwrite-layer";
    let session_name = "e2e-live-mapping-guard-overwrite-session";

    let (session_id, object_id_a) =
        create_session_and_layer(&harness.tv, &harness.svc, session_name, layer_name).await;

    // Land at least one generation for object_id_A.
    let _handle_a = wait_for_checkpoint(&harness.store, object_id_a, 10).await;

    // Simulate Valkey state loss for object_id_A: UNLINK all obj:* keys
    // AND DEL the layer mapping. Both share fate in a real failover.
    wipe_layer_state(&harness.tv, object_id_a).await;
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&session_id, layer_name);
    let _: () = harness
        .tv
        .client()
        .pool
        .del(&mapping_key)
        .await
        .expect("DEL of layer mapping must succeed against healthy Valkey");

    // Re-CreateLayer on the same (session_id, layer_name). The
    // CreateLayer handler mints a fresh object_id; the new mapping
    // overwrites the just-deleted slot.
    let (_session_id_again, object_id_b) =
        create_session_and_layer_reuse(&harness, &session_id, layer_name).await;

    // Sanity: the re-CreateLayer minted a distinct object_id. A
    // regression that reused the old `object_id_A` (e.g. INCR counter
    // miswire) would silently invalidate the rest of the test.
    assert_ne!(
        object_id_a, object_id_b,
        "re-CreateLayer MUST mint a fresh object_id; reusing object_id_A would \
         mean the live mapping STILL resolves to A and the guard would pass"
    );

    // Confirm the checkpoint blob for A is still in the store. Without
    // this, the rest of the test could pass vacuously on an empty store.
    let a_handles = harness
        .store
        .list_layer_checkpoints(object_id_a)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        !a_handles.is_empty(),
        "checkpoint blob for stale object_id_A={object_id_a} MUST survive the \
         state-loss + re-CreateLayer cycle (LocalDisk adapter has no post-loss \
         sweep) — no surviving blob means the test is not exercising the race"
    );

    // gRPC-level invariant: StatLayer against the layer_name resolves to
    // object_id_B (the fresh mapping) and returns Ok — the live mapping
    // is healthy for B, even though A's checkpoint is orphaned on disk.
    let stat_b = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "StatLayer on the re-created layer_name MUST succeed against object_id_B; \
             a NotFound here means the re-CreateLayer mapping install failed",
        )
        .into_inner();
    assert!(
        !stat_b.created_by_user_name.is_empty(),
        "StatLayer on the re-created layer MUST surface a non-empty created_by_user_name \
         from object_id_B's freshly-written header"
    );

    // Snapshot recovery-metric cells AFTER the gRPC StatLayer (which
    // must not have fired recovery — its mapping resolved to B's healthy
    // obj:hdr) so the direct-call assertion below sees a clean baseline.
    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    // Direct guard exercise on the stale A. The hook fetches A's
    // checkpoint blob, runs apply, and the SR-2.4 identity guard GETs the
    // mapping named by the blob's own (sid, layer_name) — which now
    // resolves to B, not A — returns false, apply returns NotFound, hook
    // returns NotRecoverable.
    let outcome = harness.svc.recovery.try_restore_layer(object_id_a).await;
    assert!(
        matches!(outcome, RecoveryOutcome::NotRecoverable),
        "stale-object_id apply MUST return NotRecoverable; got {outcome:?} \
         (a `Restored` here means the guard let the apply through and \
         resurrected a phantom object disjoint from the live mapping)"
    );

    // Exactly ONE `not_recoverable` increment from the direct call.
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        1,
        "exactly ONE recovery attempt must have reported `not_recoverable` \
         for stale object_id_A={object_id_a} (SR-2.4 guard rejected: \
         live mapping resolves to B, not A)"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "restored"),
        restored_pre,
        "`restored` cell MUST stay at the pre-call snapshot — the guard \
         rejected A before write_object ran"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error"),
        error_pre,
        "`error` cell MUST stay at the pre-call snapshot — LocalDisk + guard \
         rejection is a NotFound path, not an error path"
    );

    // Valkey-state invariant: no obj:hdr for A. A regression that
    // bypassed the guard would have resurrected A's header alongside
    // B's (two separate object_ids materialized, only one with a live
    // mapping — the phantom resurrection bug the guard is designed to
    // prevent).
    assert!(
        !obj_header_exists(&harness, object_id_a).await,
        "guard-rejected apply MUST NOT write obj:hdr for stale A={object_id_a}; \
         got header present (phantom resurrection — the SR-2.4 invariant has \
         been violated)"
    );

    // B's header MUST still be present — confirms the rejected apply
    // did not collaterally damage B's live state.
    assert!(
        obj_header_exists(&harness, object_id_b).await,
        "B's obj:hdr MUST survive the stale-A recovery attempt; missing here \
         means the rejected apply somehow wiped B's state"
    );

    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

/// Re-CreateLayer on an existing `session_id` + `layer_name`. Lives in
/// this file (rather than the shared `helpers` module) because only the
/// overwrite-race test needs it — `helpers::create_session_and_layer`
/// always creates a fresh session, and Subtask `t3.fgj.ya.wg.9n`'s
/// scaffolding contract is "no helper deletes / additions without a
/// sibling Subtask need". A future Subtask sharing the same shape can
/// promote this into `helpers` then.
///
/// The session_store / svc path here mirrors
/// `crate::grpc::helpers::create_session_and_layer` but skips the
/// `session_store.create` call — the session already exists from the
/// initial CreateLayer.
async fn create_session_and_layer_reuse(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
) -> (String, u64) {
    use crate::common::upload_stream::{AttestedStreaming, IntoHandlerRequest as _};
    use live_edit::grpc::layer::handle_create_layer;
    use live_edit::grpc::proto::{
        create_layer_request, CreateLayerChunk, CreateLayerHeader, CreateLayerRequest,
    };

    let ompf_bytes = crate::common::owned_delta::delta_to_ompf_bytes(
        &crate::common::owned_delta::empty_delta(),
        &[],
    );

    let messages = vec![
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Header(
                CreateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: layer_name.to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: ompf_bytes,
            })),
        },
    ];

    // The upload ends with END_STREAM and carries its end-of-stream record.
    let mut request = AttestedStreaming::from_messages(&messages).into_handler_request();
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));

    let response = handle_create_layer(&harness.svc, request)
        .await
        .expect("re-CreateLayer on existing (session_id, layer_name) must succeed");

    let object_id = response.into_inner().object_id;
    (session_id.to_owned(), object_id)
}
