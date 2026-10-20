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

// CreateLayer, UpdateLayer, DeleteLayer, StatLayer handlers.

use std::sync::Arc;

use bytes::Bytes;
use tonic::{Request, Response, Status, Streaming};

use super::auth_helpers::{
    check_read_permission, check_write_permission, extract_token, BasePathSource,
};
use super::proto::{
    CreateLayerRequest, CreateLayerResponse, DeleteLayerRequest, DeleteLayerResponse,
    OperationOutcome as WireOperationOutcome, StatLayerRequest, StatLayerResponse,
    UpdateLayerHeader, UpdateLayerRequest, UpdateLayerResponse,
};
use super::rpc_log::{self, RpcLogContext, RpcStage};
use super::service::LiveEditService;
use super::streaming::{
    read_upload_body, read_upload_header, UpdateLayerFirstHeader, UploadStream,
};
use super::transport::server_shutting_down;
use super::upload_admission::{
    log_upload_runs_beyond_budget, object_state_charge, parse_charge, processing_charge,
    CommitPath, DeltaFootprint, UploadCommitCharge, UploadRefusal, UploadReservation, UploadRpc,
};
use crate::checkpoint::RecoveryOutcome;
use crate::domain::ompf::OmpfContainer;
use crate::domain::types::ObjectState;
use crate::domain::validation::{
    validate_client_id, validate_delta_bounds, validate_layer_name, validate_operation_id,
    validate_session_id,
};
use crate::error::{RefusalDetail, ServiceError};
use crate::storage::chain_gate::{ChainGate, ChainLink, GateExit};
use crate::storage::commit_charge::CommitCharge;
use crate::storage::ext_release_gate::ExtReleaseRefusal;
use crate::storage::external_values::{
    delta_external_hashes, prepare_external_values, store_external_values, PreparedExternalValues,
};
use crate::storage::layer::{
    operation_request_digest, outcome_unknown_after_retry, CreateOperation, Operation,
    OperationOutcome,
};

/// The wire value of a storage-side [`OperationOutcome`].
fn wire_outcome(outcome: OperationOutcome) -> i32 {
    match outcome {
        OperationOutcome::Unspecified => WireOperationOutcome::Unspecified,
        OperationOutcome::Applied => WireOperationOutcome::Applied,
        OperationOutcome::AlreadyCommitted => WireOperationOutcome::AlreadyCommitted,
    }
    .into()
}

/// Handle CreateLayer RPC.
///
/// 1. Read the header frame and validate session id, layer name and operation
///    id. No content is read yet. The request must carry its end-of-stream
///    record ([`UploadStream::from_request`]; `INTERNAL` without one).
/// 2. Auth: look up session base_path, check write permission.
/// 3. Admission: reserve from the replica's upload budget
///    ([`UploadAdmissionGate::admit`](super::upload_admission::UploadAdmissionGate::admit)).
///    The reservation lives to the end of the call.
/// 4. Read the content (OMPF chunks) under that reservation. The upload is
///    applied only if its stream ended with END_STREAM: a stream reset, lost
///    or ended any other way — before or after the header — is `CANCELLED`
///    with nothing created or mapped. A header followed by END_STREAM is a
///    valid empty upload.
/// 5. With an operation id: answer from its record if an earlier call already
///    committed it (`ALREADY_COMMITTED`, or `ALREADY_EXISTS` for a different
///    request), before the external-value store and object-id allocation. An
///    `ALREADY_COMMITTED` answer still parses the re-sent bytes, to re-spawn
///    the recorded object's first base (step 8).
/// 6. Parse, in two processing steps: hold the parse charge, parse the OMPF
///    delta, run the per-delta checks on it (§9.4 caps, name characters,
///    sample times, self-parenting; `INVALID_ARGUMENT` before the processing
///    charge), then hold the processing charge its element counts imply for a
///    create; then store its external values — first waiting at the
///    external-value release budget when the container carries external parts
///    (`UNAVAILABLE`, nothing applied, if it stays full for the call's
///    remaining deadline less a margin) — and apply the delta (or create an empty ObjectState for empty
///    content). Neither processing step is refused for size: each waits, up
///    to the RPC deadline, for room in the upload budget or for the overflow
///    slot, where the upload runs beyond the budget.
/// 7. Persist object to Valkey via LayerStore, return object_id and the
///    operation outcome. A session deleted since step 2 is refused at the
///    commit point, `NOT_FOUND "session was deleted"`, with nothing mapped.
///    Once the commit point has made the new object the layer, nothing after
///    it fails the call.
/// 8. Spawn the SR-7.1 creation-time first-base bootstrap for the layer's
///    object — the one this call created, or the recorded one on an
///    `ALREADY_COMMITTED` answer. Post-commit and best-effort: the committed
///    `ObjectState` is *moved* into a task on the shared checkpoint
///    `TaskTracker`, and nothing the bootstrap does can fail this RPC. The gate
///    and every failure arm live inside
///    [`BaseBootstrap`](crate::checkpoint::BaseBootstrap). An overwrite this
///    call could not confirm, completed later by its finisher or by the orphan
///    sweep, gets no creation-time first base.
///
/// Deadline-drop contract (b.yis sub-item 4, see `metrics_layer.rs`): without
/// an `operation_id`, non-idempotent (every call that commits allocates a fresh
/// object_id). With one, a re-send of a create that committed is answered from
/// the `le:{sid}:createop:{operation_id}` record and allocates nothing (see
/// `LayerStore::create_layer_with_operation`). A drop before the commit point
/// leaves external values with non-zero refcounts that the GC sweep reclaims on
/// next pass, and a materialized object no mapping names, which the orphan
/// sweep reclaims. The mapping flip, the layers-index entry and the operation
/// record are one atomic script, so either all or none commit — never a
/// half-registered layer. A client reset after its END_STREAM cancels the call,
/// not the upload: hyper drops this handler the next time it waits, as a
/// deadline does, so the same contract holds.
///
/// **NotFound recovery (SDD SR-2.7).** Step 2's `get_base_path` call funnels
/// through [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache-miss + session-absent triggers lazy on-demand
/// session-state recovery via the attached
/// [`SessionRecoveryHook`](crate::checkpoint::SessionRecoveryHook). A genuinely
/// non-existent / unrecoverable session still surfaces `NotFound` to the
/// client; no behavior change at this handler.
///
/// **Failure logging.** This is a wrapper over [`create_layer_inner`]; every
/// non-OK return flows through [`rpc_log::log_rpc_failure`], which emits
/// exactly one completion line — `RPC failed`, or `internal error` at ERROR
/// for a server fault — and marks the request's completion ledger reported.
/// The signature is deliberately unchanged — the handler is called directly
/// by the `LiveService` trait impl and by ~68 integration-test files, all of
/// which bind `Result<Response<_>, Status>`.
///
/// **A refusal before the commit.** Every failure passes
/// [`refused_before_commit`] before it is logged and answered: a plain
/// `UNAVAILABLE` raised before the commit marker — set immediately before
/// step 7's `LayerStore::create_layer_with_operation` — is answered carrying
/// `UPSTREAM_UNAVAILABLE` (this call applied nothing). Steps 1–6 write no
/// client-visible state: the reads and the permission check write nothing, a
/// lazy session restore on the base-path read restores state this call did
/// not write, the operation-record read writes only on an answer that is not
/// an error, and the external-value pre-store leaves only reference counts
/// and blobs the garbage collector and the reference-count reconciler drain.
/// No step before the marker writes client-visible state; a step that would
/// must move the marker before itself.
pub async fn handle_create_layer(
    svc: &LiveEditService,
    request: Request<Streaming<CreateLayerRequest>>,
) -> Result<Response<CreateLayerResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("CreateLayer", &request);
    let result = create_layer_inner(svc, request, &mut ctx)
        .await
        .map_err(|status| refused_before_commit(&ctx, status));
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn create_layer_inner(
    svc: &LiveEditService,
    request: Request<Streaming<CreateLayerRequest>>,
    ctx: &mut RpcLogContext,
) -> Result<Response<CreateLayerResponse>, Status> {
    let token = extract_token(&request)?;
    // The stream with its end-of-stream record: a request without one is
    // refused `INTERNAL` here, before any read.
    let mut stream = UploadStream::from_request(request)?;

    // Step 1: Read and validate the header frame. No content is read until
    // the call is authorized and admitted (steps 2-3), so neither a bogus
    // header nor an unauthorized caller makes the server buffer content.
    let header = read_upload_header(&mut stream, |h| {
        // b.39x H: full session_id validation (length cap + control-char
        // rejection) before any chunk payload buffers. Replaces the
        // pre-fix `is_empty()`-only check that let a 64KB session_id with
        // ANSI escapes flow into Valkey keys, pub/sub channels, and
        // tracing output.
        //
        // Each `ctx` record sits BELOW its validator for that same reason
        // (b.94g): the failure-boundary log is one more tracing sink, and
        // recording an unvalidated field into it would re-open the hole the
        // validators close.
        validate_session_id(&h.session_id)?;
        ctx.session(&h.session_id);
        validate_layer_name(&h.layer_name)?;
        ctx.layer(&h.layer_name);
        validate_operation_id(&h.operation_id)?;
        ctx.operation(&h.operation_id);
        Ok(())
    })
    .await?;

    let session_id = &header.session_id;
    let layer_name = &header.layer_name;

    // Step 2: Auth checks (BEFORE any content is read or parsed — security
    // requirement).
    ctx.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, session_id)
        .await?;
    ctx.base_path(&base_path);
    check_write_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // Step 3: Admission. The reservation is held to the end of the call —
    // across the parse, the external-value store and the commit — and its
    // drop releases it on every exit.
    ctx.at(RpcStage::Admission);
    let mut reservation = admit_upload(svc, UploadRpc::CreateLayer, session_id, ctx).await?;

    // Step 4: Read the content under the reservation. A stream that did not
    // end with END_STREAM is refused `CANCELLED` here, before anything below
    // stores or commits.
    ctx.at(RpcStage::Body);
    let ompf_bytes = read_upload_body(stream, svc.max_layer_size, &mut reservation).await?;

    // The client operation id, when one was sent, with its request digest over
    // the bytes exactly as streamed. Hashed only after the write-permission
    // check, like every other per-byte cost of the payload.
    let operation = (!header.operation_id.is_empty()).then(|| {
        CreateOperation::new(
            &header.operation_id,
            operation_request_digest(layer_name, &ompf_bytes),
        )
    });

    // Step 5: a re-send of an operation an earlier call already committed is
    // answered from its record here — before userinfo, the external-value
    // store and object-id allocation. `LayerStore::create_layer_with_operation`
    // re-checks atomically for the duplicate that races past this read.
    //
    // The re-send still pays one parse: it re-spawns the recorded object's
    // creation-time first base, because the original call may never have
    // reached its own spawn (its reply was lost after the commit point — the
    // case operation ids exist for). The re-sent bytes are byte-identical to
    // the original's (equal digests), so the state rebuilt here IS the state
    // that object was created with. A duplicate spawn for an object that
    // already has its first base is a no-op by the bootstrap's own guards (see
    // `BaseBootstrap::spawn_creation_base`). A parse failure here — not
    // expected, since the original parsed — only costs the re-spawn, and is
    // logged; the answer stands. So does a refused growth of the reservation
    // for that parse: the upload budget being full costs the re-spawn, not
    // the answer.
    if let Some(op) = operation.as_ref() {
        ctx.at(RpcStage::Resolve);
        if let Some(answered) = svc
            .layer_store
            .resolve_create_layer_operation(session_id, layer_name, op)
            .await
            .map_err(|e| -> Status { e.into() })?
        {
            ctx.at(RpcStage::Parse);
            match creation_state_for_resend(&ompf_bytes, &mut reservation) {
                Ok(state) => svc
                    .base_bootstrap
                    .spawn_creation_base(answered.object_id, state),
                Err(e) => tracing::warn!(
                    session_id = %session_id,
                    layer_name = %layer_name,
                    object_id = answered.object_id,
                    operation_id = op.operation_id(),
                    error = %e,
                    "CreateLayer re-send could not rebuild the creation state; its first-base \
                     re-spawn is skipped (the ALREADY_COMMITTED answer stands)"
                ),
            }
            return Ok(Response::new(CreateLayerResponse {
                object_id: answered.object_id,
                operation_outcome: wire_outcome(answered.outcome),
            }));
        }
    }

    // Resolve user identity for created_by. Transient backend outages flow
    // through `From<AuthError> for ServiceError` so they surface as
    // `UNAVAILABLE` (b.ohw H-grpc-1) rather than `INTERNAL`. Userinfo is an
    // `Auth`-stage call (see `RpcStage::Auth`), so a failure here logs at that
    // stage even though it runs after the content read.
    ctx.at(RpcStage::Auth);
    let user_info = svc
        .userinfo_client
        .get_current_user(&token)
        .await
        .map_err(ServiceError::from)?;

    // Step 6: OMPF + delta pipeline, in two processing steps. The zero-copy
    // container parse gives the delta and external-value part lengths; the
    // reservation holds the parse charge before the delta is parsed, and the
    // processing charge its element counts imply once it is.
    ctx.at(RpcStage::Parse);
    let container = parse_container(&ompf_bytes)?;
    let lengths = UploadLengths::of(&ompf_bytes, &container);
    reserve_processing_step(
        svc,
        &mut reservation,
        lengths.parse_charge(),
        session_id,
        layer_name,
        ctx,
    )
    .await?;
    let mut state = ObjectState::default();
    // Track per-delta observability stats so the gRPC layer can surface
    // counters the pure-domain engine cannot reach (b.39x item 7).
    let mut stats = crate::domain::delta::ApplyDeltaStats::default();
    // b.ot3: CreateLayer materializes the object from `ObjectState` (see
    // `LayerStore::create_layer` / `queue_object_state`), never from the
    // delta's mutation list — it dropped that list on the floor after
    // paying to build it, which is where the measured 244 MiB from a
    // 289 KiB delta went. Discard the mutations at the source instead.
    let mut sink = crate::domain::delta::MutationSink::Discard;
    // The external-value hashes stored (and so pre-incremented) for this call,
    // carried with the operation so a call answered from an earlier call's
    // record can hand them back.
    let mut prestored_ext_hashes: Vec<[u8; 32]> = Vec::new();

    let (owned, external_values) = parse_create_payload(container)?;
    // The per-delta checks (§9.4 caps, name characters, sample times,
    // self-parenting) before the processing charge, so a delta that fails
    // them is refused `INVALID_ARGUMENT` here, at stage `parse`, holding only
    // the parse charge, rather than charged its processing charge — and
    // perhaps queued for or run in the overflow slot — only to be refused by
    // `apply_delta_into`'s own check, which stays as the backstop.
    if let Some(d) = owned.as_ref() {
        validate_delta_bounds(d).map_err(ServiceError::from)?;
    }
    reserve_processing_step(
        svc,
        &mut reservation,
        lengths.processing_charge(CommitPath::Create, DeltaFootprint::of(owned.as_ref())),
        session_id,
        layer_name,
        ctx,
    )
    .await?;

    // Step 6b: Store external values BEFORE object creation (SDD §11.1, §7.9
    // step 2). Must complete before object_id allocation so errors return
    // INVALID_ARGUMENT without allocating orphaned IDs. A no-op for empty
    // content or a delta-less container. An external hash that is not exactly
    // 32 bytes, or an unresolvable `ext_value_index`, is refused by the
    // collector inside this call before any blob is stored.
    //
    // An upload whose container carries external parts first waits at the
    // external-value release budget (b.yuo): it can create more released blobs
    // for the GC, so it is held back, and finally refused UNAVAILABLE with
    // nothing applied, while the store already holds a budget's worth.
    if !external_values.is_empty() {
        await_ext_release_budget(svc, ctx).await?;
    }
    store_external_values(
        owned.as_ref(),
        &external_values,
        svc.ext_value_store.as_ref(),
    )
    .await
    .map_err(|e| -> Status { e.into() })?;
    if operation.is_some() {
        if let Some(d) = owned.as_ref() {
            prestored_ext_hashes = delta_external_hashes(d);
        }
    }

    // Apply delta to fresh ObjectState. Empty content (SDD §8.3) still gets a
    // version increment: `apply_delta_into` on no delta does this.
    crate::domain::delta::apply_delta_into(&mut state, owned, &mut sink, &mut stats)
        .map_err(ServiceError::from)?;

    // Bump the "silently-skipped CreateSection" counter exactly once per
    // RPC. Zero is a no-op inside the helper (b.39x item 7).
    svc.metrics.inc_delta_create_section_skipped_missing_parent(
        stats.create_section_skipped_missing_parent,
    );

    // Step 7: Persist to Valkey and return.
    //
    // No `submitted_object_id` record: CreateLayer *mints* the id, so it does
    // not exist before this call. Once the commit point has made the new
    // object the layer the call answers OK: nothing after it fails the call
    // (see `LayerStore::create_layer_with_operation`).
    ctx.at(RpcStage::Storage);
    // The commit marker ([`refused_before_commit`]): from here on a plain
    // `UNAVAILABLE` is answered as it is, because the commit — allocation,
    // materialization, the mapping flip — can apply part or all of the call,
    // and a replacement whose commit was left in doubt can still take effect
    // after the answer. No step before this marker writes client-visible
    // state; a step that would must move the marker before itself.
    ctx.commit_started();
    let commit = svc
        .layer_store
        .create_layer_with_operation(
            session_id,
            layer_name,
            &user_info.display_name,
            &state,
            svc.ext_value_store.clone(),
            operation.map(|op| op.with_prestored_ext_hashes(prestored_ext_hashes)),
        )
        .await
        .map_err(|e| -> Status { e.into() })?;
    let object_id = commit.object_id;

    // An `AlreadyCommitted` answer is not this call's object: a concurrent
    // call with the same operation id committed first, and the store already
    // logged it. It is not "completed" here, but its first base is still
    // spawned for the RECORDED object: `state` was built from bytes
    // byte-identical to the winner's (equal digests), so it is the state that
    // object was created with, and a duplicate spawn is a no-op by the
    // bootstrap's own guards.
    if commit.outcome != OperationOutcome::AlreadyCommitted {
        tracing::info!(
            session_id = %session_id,
            layer_name = %layer_name,
            object_id,
            "CreateLayer completed"
        );
    }

    // SR-7.1 first-base bootstrap. Post-commit, after the response value is
    // already determined: `state` is moved (never cloned) into a best-effort
    // task on the shared checkpoint `TaskTracker`, so this call adds no
    // synchronous work to the commit path beyond a gate read. The three-way
    // gate and every failure arm live inside the unit.
    svc.base_bootstrap.spawn_creation_base(object_id, state);

    Ok(Response::new(CreateLayerResponse {
        object_id,
        operation_outcome: wire_outcome(commit.outcome),
    }))
}

/// Handle UpdateLayer RPC (SDD §7.12).
///
/// 1. Read the header frame and validate object_id non-zero, layer name,
///    session_id, client_id, operation_id (which requires a non-empty
///    client_id). A chained call's frame is `chained_header`: its `header`
///    must be present and carry an operation_id, and its
///    `previous_operation_id` must be empty or a valid operation_id other
///    than that one. No content is read yet. The request must carry its
///    end-of-stream record ([`UploadStream::from_request`]; `INTERNAL` without
///    one).
/// 2. Auth: look up session base_path, check write permission.
/// 3. Resolve layer mapping — NOT_FOUND if missing, FAILED_PRECONDITION if stale.
/// 4. Admission: reserve from the replica's upload budget
///    ([`UploadAdmissionGate::admit`](super::upload_admission::UploadAdmissionGate::admit)).
///    The reservation lives to the end of the call, the recovery re-issue
///    included.
/// 5. Read the content (OMPF chunks) under that reservation. The upload is
///    applied only if its stream ended with END_STREAM: a stream reset, lost
///    or ended any other way — before or after the header — is `CANCELLED`
///    with nothing committed. A header followed by END_STREAM is a valid
///    empty update. With an operation_id on an unchained call, start the one
///    call-start read of its start record.
///
///    Step 5b: a chained call then waits in the chain gate
///    ([`await_chain_predecessor`], `storage::chain_gate`) until the latest
///    operation recorded for its `(object_id, client_id)` is its predecessor
///    or its own, holding only its reservation; it is refused `ABORTED`
///    `PREDECESSOR_NOT_COMMITTED` at once under `no_wait`, or when only its
///    margin is left before its deadline otherwise (`UNAVAILABLE`
///    `CHAIN_CHECK_UNAVAILABLE` there, with or without `no_wait`, when no
///    probe ever read the record). The margin is measured from the call's
///    deadline budget, read once at this handler's entry, before its first
///    await. The gate's `ready` is advisory: the call's operation carries its
///    predecessor as its writer guard, which every commit attempt checks
///    (step 7).
/// 6. Parse, in two processing steps: hold the parse charge, parse the OMPF
///    delta, run the per-delta checks on it (§9.4 caps, name characters,
///    sample times, self-parenting; `INVALID_ARGUMENT` before the processing
///    charge), decide its commit path (`LayerStore::takes_fast_path`), then
///    hold the processing charge its element counts imply on that path; then
///    store its external values — first waiting at the external-value release
///    budget when the container carries external parts (`UNAVAILABLE`, nothing
///    applied, if it stays full for the call's remaining deadline less a
///    margin). The values are checked once; the recovery re-issue stores
///    them again without a second check or a second release-budget wait.
///    Neither processing
///    step is refused for size: each waits, up to the RPC deadline, for room
///    in the upload budget or for the overflow slot, where the upload runs
///    beyond the budget.
/// 7. Commit through `LayerStore::update_layer_with_summary` — the Lua fast
///    path; under the `lua` backend, for a structural or reset delta, a plan
///    on the per-replica structural cache committed by the script's
///    `seq`-fenced structural mode; or the WATCH/MULTI/EXEC slow path: apply
///    delta, persist, publish the stream entry — or answer a re-sent
///    operation from its record. The call's summary goes on the failure log
///    context.
/// 8. Return sequence_num + server_timestamp + operation_outcome.
///
/// Deadline-drop contract (b.yis sub-item 4, see `metrics_layer.rs`): this RPC
/// runs under the server deadline. Each commit is atomic, so a drop before fred
/// has written the commit leaves nothing and the client's retry is a clean
/// first apply: fred never writes a commit whose call was dropped while it was
/// still queued (the vendored fred's change 9). A drop after fred wrote the
/// `EVALSHA` / `EXEC` does not recall it, so that drop answers
/// `DEADLINE_EXCEEDED` for a write that can land anyway. Neither
/// commit is written a second time by fred after a reconnect (both go out
/// with one attempt). Within one call the in-process retries commit at most
/// once (the per-call commit token;
/// see `LayerStore::update_layer_with_operation`). Across calls it depends on
/// the client:
///
/// - **With an `operation_id`** a client retry after `DEADLINE_EXCEEDED`,
///   after `UNAVAILABLE` on retry exhaustion, or after a lost OK — a NEW call
///   carrying the same id and bytes, sent before that `client_id`'s next
///   write to the object — finds the commit recorded under `op.<id>` and is
///   answered `ALREADY_COMMITTED` with the original sequence and timestamp,
///   applying nothing; a repeated `CreateSection` is answered, not refused.
///   And no attempt of the call applies after that `client_id`'s next commit
///   to the object: an attempt that runs late finds the field moved from the
///   call's start record and answers `UNAVAILABLE` (overtaken), that attempt
///   applying nothing. (An earlier attempt of the call may have committed
///   with its reply lost; once the record has moved the server cannot tell,
///   so a re-send may apply again.) A chained call has no start record and is
///   never overtaken: every attempt commits only while the field names its
///   predecessor, and is otherwise refused `ABORTED`
///   `PREDECESSOR_NOT_COMMITTED` on the call's first attempt, or answered a
///   plain `UNAVAILABLE`, outcome unknown, after a counted retry. Its re-send
///   — same `operation_id`, bytes, `client_id` and predecessor — is always
///   safe.
/// - **Without one** such a retry can re-apply a delta that already landed:
///   field-only deltas converge, a repeated `CreateSection` is refused
///   `INVALID_ARGUMENT`.
///
/// The recovery re-issue below is a new call with a new token, which is
/// correct because the restore it follows wiped the record with the rest of
/// the object's residue — so it matches nothing, with or without an id. It
/// reuses the call's writer guard rather than re-reading anything: a restore
/// that wiped a non-absent start record leaves an unchained re-issue
/// overtaken, and a chained re-issue whose predecessor's record was wiped is
/// refused `PREDECESSOR_NOT_COMMITTED` with `last_record=absent` (a plain
/// `UNAVAILABLE`, outcome unknown, when the first call made more than one
/// attempt).
/// Ext-values stored before the commit are refcount-safe via the GC sweep.
/// Every `update_layer` call of the RPC is preceded by its own pre-store: the
/// re-issue stores the RPC's external values again before it runs
/// ([`prestore_for_recovery_reissue`]; a failed re-pre-store ends the RPC
/// before the re-issue), and each call hands back only its own increments. A
/// call answered from an earlier call's record, or refused overtaken or
/// `PREDECESSOR_NOT_COMMITTED` on its first attempt, hands its own pre-flight
/// increments back — the re-issue included; a call that ends without proof of
/// its outcome after a counted retry or on retry exhaustion instead enqueues
/// the releases its own landed commit owed, if one landed
/// (`LayerStore::update_layer_with_operation`). The first call's increment
/// leaks to the reference-count reconciler, the safe direction, when its
/// commit never landed or the restore kept its reference.
///
/// A client reset after its END_STREAM cancels the call, not the upload: hyper
/// drops this handler the next time it waits, as a deadline does, so the
/// same contract holds.
///
/// A chained call dropped while it waits in the chain gate (step 5b) applies
/// nothing: the gate awaits inside this future, spawns nothing, and precedes
/// the parse, the external-value store and every commit attempt. One dropped
/// after fred wrote its commit can still land, but only directly after its
/// predecessor: the commit's predecessor guard
/// refuses it unless the writer's record still names that predecessor.
///
/// **NotFound recovery (SDD SR-2.4 / SR-2.6 / SR-2.7).** Two recovery surfaces
/// stack here. Step 2's `get_base_path` runs session-state recovery via
/// [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path)
/// on cache-miss + session-absent. Then step 3's layer-mapping lookup runs
/// per-`object_id` layer-state recovery via
/// [`RecoveryHook::try_restore_layer`](crate::checkpoint::RecoveryHook::try_restore_layer)
/// on missing mapping / missing header. The ordering — session-recovery
/// FIRST, layer-recovery SECOND — is load-bearing for the lazy-recovery
/// success criteria and follows by construction from `get_base_path` being
/// step 2 of this handler's flow, before the layer-mapping lookup.
///
/// **Failure logging.** Wrapper over [`update_layer_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`]. An `object_id` mismatch logs at `info`
/// carrying both the submitted and the resolved id. The two recovery-progress
/// `warn!`s inside the inner function are *progress* events, not completions,
/// and stay where they are.
///
/// **A chained call's `INVALID_ARGUMENT`.** Every failure passes
/// [`chained_invalid_argument`] before it is logged and answered: the one
/// choke point that puts `INVALID_UPDATE` on a chained call's
/// `INVALID_ARGUMENT`, whichever step raised it.
///
/// **A refusal before the commit.** Every failure then passes
/// [`refused_before_commit`]: a plain `UNAVAILABLE` raised before the commit
/// marker — set immediately before step 7's first
/// `LayerStore::update_layer_with_summary`, after the call-start read — is
/// answered carrying `UPSTREAM_UNAVAILABLE` (this call applied nothing), on a
/// chained call and an unchained one alike. Steps 1–6 write no client-visible
/// state: the reads, the permission check, the chain gate and the call-start
/// read write nothing, a lazy session or layer restore restores state this
/// call did not write, and the external-value pre-store leaves only reference
/// counts and blobs the garbage collector and the reference-count reconciler
/// drain. Nothing from the marker on is converted — the recovery re-issue
/// included. No step before the marker writes client-visible state; a step
/// that would must move the marker before itself. The two choke points act on
/// disjoint codes (the one `UNAVAILABLE` [`chained_invalid_argument`] answers
/// follows a counted retry, past the marker, and passes unchanged).
pub async fn handle_update_layer(
    svc: &LiveEditService,
    request: Request<Streaming<UpdateLayerRequest>>,
) -> Result<Response<UpdateLayerResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("UpdateLayer", &request);
    let result = update_layer_inner(svc, request, &mut ctx)
        .await
        .map_err(|status| refused_before_commit(&ctx, chained_invalid_argument(&ctx, status)));
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

/// The `INVALID_UPDATE` choke point (SDD §18.1): the status a chained
/// `UpdateLayer` (`UpdateLayerRequest.chained_header`) is answered with in
/// place of `status`.
///
/// - Not a chained call, or not `INVALID_ARGUMENT`: `status` unchanged.
/// - After a counted retry — some `LayerStore::update_layer` call of the RPC
///   made more than one attempt ([`RpcLogContext::update_layer_counted_retry`];
///   the same rule as the recovery re-issue's below) — a plain `UNAVAILABLE`,
///   outcome unknown ([`outcome_unknown_after_retry`]): an earlier attempt may
///   have sent its commit and lost the reply, so the call cannot promise it
///   applied nothing.
/// - A status that already carries a detail keeps it: the size refusals
///   [`DELTA_TOO_LARGE`](crate::error::refusal_reason::DELTA_TOO_LARGE) and
///   [`CHILDREN_LIST_TOO_LONG`](crate::error::refusal_reason::CHILDREN_LIST_TOO_LONG),
///   which the parse or the delta checks raise before any counted retry. Their
///   reason takes the place of `INVALID_UPDATE`; they too mean this call
///   applied nothing.
/// - Otherwise `INVALID_ARGUMENT` with the same message carrying
///   [`INVALID_UPDATE`](crate::error::refusal_reason::INVALID_UPDATE)
///   ([`ServiceError::invalid_update`]): this call applied nothing. So every
///   chained `INVALID_ARGUMENT` carries `INVALID_UPDATE` unless another reason
///   applies.
///
/// Keyed on the status code, not on the `ServiceError` variant, so no
/// `INVALID_ARGUMENT` a chained call meets — the header rules, the body caps,
/// the OODF and delta checks, a planning refusal — can bypass it: every
/// handler `INVALID_ARGUMENT` is a `ServiceError::InvalidArgument` (or, for a
/// size refusal, `InvalidArgumentWithDetail`) converted on its way out.
/// Unchained calls are unchanged, so Kit is unaffected.
fn chained_invalid_argument(ctx: &RpcLogContext, status: Status) -> Status {
    if !ctx.is_chained() || status.code() != tonic::Code::InvalidArgument {
        return status;
    }
    if let Some(attempts) = ctx.update_layer_counted_retry() {
        return outcome_unknown_after_retry(attempts, status.message()).into();
    }
    if !status.details().is_empty() {
        return status;
    }
    ServiceError::invalid_update(status.message()).into()
}

/// The `UPSTREAM_UNAVAILABLE` choke point (SDD §18.1): the status a
/// `CreateLayer` / `UpdateLayer` is answered with in place of `status`.
///
/// - An `UNAVAILABLE` with no detail, raised before the call's commit started
///   ([`RpcLogContext::is_commit_started`]): the same message, carrying
///   [`UPSTREAM_UNAVAILABLE`](crate::error::refusal_reason::UPSTREAM_UNAVAILABLE)
///   and its retry hint ([`RefusalDetail::upstream_unavailable`]). A
///   dependency the call needed before its commit — the backing store, the
///   auth backend, the backup store of a lazy restore — was unavailable, and
///   no step before the commit marker writes client-visible state, so this
///   call applied nothing.
/// - Anything else unchanged: another code; an `UNAVAILABLE` that already
///   carries a detail (an upload-capacity, shutdown, release-budget,
///   store-refusal or chain-gate refusal keeps its own reason); and every
///   `UNAVAILABLE` from the commit on — retry exhaustion, overtaken, a
///   `CreateLayer` commit-stage refusal, an outcome-unknown answer after a
///   counted retry, the recovery re-issue — whose commit may have been sent.
///
/// Keyed on the status code and the marker, not on the `ServiceError`
/// variant or the step, so no pre-commit `UNAVAILABLE` can bypass it, and a
/// new pre-commit step inherits it. The detail is built in `crate::error`,
/// the one builder of refusal details.
fn refused_before_commit(ctx: &RpcLogContext, status: Status) -> Status {
    if ctx.is_commit_started()
        || status.code() != tonic::Code::Unavailable
        || !status.details().is_empty()
    {
        return status;
    }
    RefusalDetail::upstream_unavailable().unavailable(status.message())
}

/// A chained call's link, from its validated `chained_header`.
struct ChainRequest {
    previous_operation_id: String,
    no_wait: bool,
}

/// An `UpdateLayer`'s header, and its link when it is chained. Called only on
/// a header the step-1 closure accepted, which refuses a `chained_header`
/// without its `header`.
fn split_update_header(first: UpdateLayerFirstHeader) -> (UpdateLayerHeader, Option<ChainRequest>) {
    match first {
        UpdateLayerFirstHeader::Plain(header) => (header, None),
        UpdateLayerFirstHeader::Chained(chained) => (
            chained.header.unwrap_or_default(),
            Some(ChainRequest {
                previous_operation_id: chained.previous_operation_id,
                no_wait: chained.no_wait,
            }),
        ),
    }
}

/// The §B rules of a chained call (SDD §23.1), after its header's own: its
/// `operation_id` is non-empty (so, by the header rule, is its `client_id`),
/// and its `previous_operation_id` is empty or a valid operation id other than
/// that one. `INVALID_ARGUMENT` otherwise.
fn validate_chain_link(
    operation_id: &str,
    previous_operation_id: &str,
) -> Result<(), ServiceError> {
    if operation_id.is_empty() {
        return Err(ServiceError::InvalidArgument(
            "chained_header requires a non-empty header.operation_id".to_owned(),
        ));
    }
    validate_operation_id(previous_operation_id).map_err(|e| match e {
        ServiceError::InvalidArgument(msg) => {
            ServiceError::InvalidArgument(format!("previous_operation_id: {msg}"))
        }
        other => other,
    })?;
    if previous_operation_id == operation_id {
        return Err(ServiceError::InvalidArgument(
            "previous_operation_id must differ from header.operation_id".to_owned(),
        ));
    }
    Ok(())
}

async fn update_layer_inner(
    svc: &LiveEditService,
    request: Request<Streaming<UpdateLayerRequest>>,
    ctx: &mut RpcLogContext,
) -> Result<Response<UpdateLayerResponse>, Status> {
    // The call's deadline budget `T` and its deadline `D` (now + `T`), read
    // once, here, before the first await: the sooner of the server deadline
    // and the client's `grpc-timeout`, counted from the arrival of the
    // request's headers, or `LIVE_EDIT_RPC_DEADLINE_MS` when none is in
    // scope. A chained call's gate (step 5b) takes its margin from them
    // (`storage::chain_gate`, the deadline cap).
    let call_budget =
        super::deadline::remaining().unwrap_or_else(|| svc.layer_store.rpc_deadline_cap());
    let call_deadline = tokio::time::Instant::now() + call_budget;
    let token = extract_token(&request)?;
    // The stream with its end-of-stream record: a request without one is
    // refused `INTERNAL` here, before any read.
    let mut stream = UploadStream::from_request(request)?;

    // Step 1: Read and validate the header frame. No content is read until
    // the call is authorized, its target resolved and the upload admitted
    // (steps 2-4), so none of a bogus header, an unauthorized caller, a
    // missing layer or a stale object_id makes the server buffer content.
    let first = read_upload_header(&mut stream, |first| {
        let h = match first {
            UpdateLayerFirstHeader::Plain(h) => h,
            UpdateLayerFirstHeader::Chained(chained) => {
                // Marked before any validator: the arm is the server's
                // reading of the frame, not a client identifier, and every
                // `INVALID_ARGUMENT` below must reach the `INVALID_UPDATE`
                // choke point as a chained call's.
                ctx.chained();
                chained.header.as_ref().ok_or_else(|| {
                    ServiceError::InvalidArgument("chained_header carries no header".to_owned())
                })?
            }
        };
        // b.39x H: full session_id validation (length cap + control-char
        // rejection) — see comment in handle_create_layer for the threat
        // model. Each `ctx` record sits BELOW its validator for the same
        // reason (b.94g) — the failure-boundary log is one more tracing sink.
        validate_session_id(&h.session_id)?;
        ctx.session(&h.session_id);
        if h.object_id == 0 {
            return Err(ServiceError::InvalidArgument(
                "object_id must be non-zero".to_owned(),
            ));
        }
        ctx.submitted_object_id(h.object_id);
        validate_layer_name(&h.layer_name)?;
        ctx.layer(&h.layer_name);
        // Reject pathological client_ids at ingress so they cannot be
        // echoed into the notification payload, layer storage, or
        // tracing output (b.jr8 H-grpc-6).
        validate_client_id(&h.client_id)?;
        ctx.client(&h.client_id);
        validate_operation_id(&h.operation_id)?;
        // An operation id is guarded per (object, client_id): its record and
        // its writer guard are both keyed by `client_id`, so writers sharing
        // an empty one could not be told apart.
        if !h.operation_id.is_empty() && h.client_id.is_empty() {
            return Err(ServiceError::InvalidArgument(
                "operation_id requires a non-empty client_id".to_owned(),
            ));
        }
        ctx.operation(&h.operation_id);
        if let UpdateLayerFirstHeader::Chained(chained) = first {
            validate_chain_link(&h.operation_id, &chained.previous_operation_id)?;
            ctx.chain_link(&chained.previous_operation_id, chained.no_wait);
        }
        Ok(())
    })
    .await?;
    let (header, chain) = split_update_header(first);

    let session_id = &header.session_id;
    let layer_name = &header.layer_name;
    let object_id = header.object_id;
    let client_id = &header.client_id;
    let client_timestamp = header.timestamp_microseconds_since_epoch;

    // Step 2: Auth checks (BEFORE any content is read or parsed — security
    // requirement).
    ctx.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, session_id)
        .await?;
    ctx.base_path(&base_path);
    check_write_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // Step 3: Resolve layer mapping (SDD §7.12 step 3), wrapped in the
    // first half of the at-most-once recovery-retry branch (SDD SR-2.4 /
    // SR-2.6, b.fgj Subtask t3.fgj.ya.qm.th). Done BEFORE admission and the
    // content read, so a request targeting a non-existent layer or a stale
    // object_id fails before the server reserves or buffers potentially
    // 100 MB of OMPF, let alone pays to parse and store it.
    //
    // The single `recovered_once` flag below ALSO gates the post-parse
    // commit's NotFound handler and a chained call's re-resolve after its
    // chain gate found the object replaced (step 5b), so an RPC takes at most
    // ONE recovery
    // attempt total regardless of which surface tripped it.
    //
    // The two NotFound surfaces wrapped:
    //
    // 1. `get_layer_mapping` returns `None` — the
    //    `le:{sid}:layer:{hex(name)}` key is missing.
    // 2. `update_layer_with_operation` returns `ServiceError::NotFound` — the
    //    mapping resolved but the HGETALL of `obj:hdr` inside
    //    `update_layer_once` came back empty (mapping/header replication race, or a partial
    //    state-loss that nuked the header but left the mapping).
    //
    // Outcome contract for both surfaces:
    //
    // - `Restored` → retry the failed sub-sequence ONCE. Second NotFound
    //   surfaces to the client verbatim.
    // - `NotRecoverable` → surface the original NotFound. Also the path
    //   the disabled-checkpoint short-circuit takes
    //   (`Config::checkpoint_enabled=false`), so production behavior is
    //   unchanged from today when checkpointing is off.
    // - `Error(_)` → classify the carried `CheckpointError` through
    //   `ServiceError::from_recovery_error` (b.33e): `Transient` →
    //   UNAVAILABLE so the standard gRPC client retry policy picks it up
    //   (the backend is transiently broken but the request is fine to
    //   re-issue); every other variant → INTERNAL, because retrying a
    //   permanent / inconsistent-store failure cannot succeed.
    ctx.at(RpcStage::Resolve);
    let mut recovered_once = false;
    let current_object_id =
        resolve_layer_object(svc, session_id, layer_name, object_id, &mut recovered_once).await?;

    // Recorded BEFORE the mismatch check, so the b.94g event for a stale
    // client carries both ids — "expected N, got M" is the one fact the dev
    // incident had to be reconstructed from, and it lived only in the
    // client-visible status detail.
    ctx.resolved_object_id(current_object_id);

    if current_object_id != object_id {
        return Err(object_id_mismatch(current_object_id, object_id).into());
    }

    // Step 4: Admission. The reservation is held to the end of the call —
    // across the parse, the external-value store, every storage retry and the
    // recovery re-issue — and its drop releases it on every exit.
    ctx.at(RpcStage::Admission);
    let mut reservation = admit_upload(svc, UploadRpc::UpdateLayer, session_id, ctx).await?;

    // Step 5: Read the content under the reservation. A stream that did not
    // end with END_STREAM is refused `CANCELLED` here, before the start-record
    // read, the chain gate and anything below that stores or commits.
    ctx.at(RpcStage::Body);
    let ompf_bytes = read_upload_body(stream, svc.max_layer_size, &mut reservation).await?;

    // Step 5b: a chained call waits for its predecessor, holding only its
    // reservation (`F` + content). It runs here, awaited inline — no task
    // between this handler and the gate — so a client cancel or a lost
    // connection drops the wait with the handler, before anything is parsed,
    // stored or committed. Its `ready` is advisory: the commit's predecessor
    // guard decides (below).
    if let Some(link) = &chain {
        await_chain_predecessor(
            svc,
            ctx,
            ChainedWrite {
                session_id,
                layer_name,
                object_id,
                client_id,
                operation_id: &header.operation_id,
                link,
                held_bytes: reservation.reserved(),
                budget: call_budget,
                deadline: call_deadline,
            },
            &mut recovered_once,
        )
        .await?;
    }

    // The call's start record (unchained operation ids only): what
    // `le:{oid}:obj:wtok[client_id]` holds right now, as the call starts. Read
    // ONCE — immediately after the whole call is received — and never re-read
    // within the call, the checkpoint-recovery re-issue included (it carries
    // the same `Operation`). Every commit attempt refuses itself as overtaken
    // if the field has moved by then, so no attempt applies after this
    // client's next commit. The read runs as a spawned task that overlaps the
    // parse and the external-value store, and is awaited just before the
    // first commit. An early return drops the handle, which detaches the
    // read; nothing it learns reaches the caller. A chained call issues no
    // such read: its writer guard is its predecessor (below).
    let start_record_read = (chain.is_none() && !header.operation_id.is_empty()).then(|| {
        svc.layer_store
            .spawn_start_record_read(object_id, client_id)
    });

    // The client operation id, when one was sent, with its request digest over
    // the bytes exactly as streamed. Hashed only once the write is authorized
    // and its object resolved; the commit path matches and records it
    // (SDD §7.5.3).
    //
    // A chained call's operation carries its predecessor as its writer guard:
    // every commit attempt, on every route, refuses itself unless the record
    // names that predecessor (or, for a chain's first write, is absent), so
    // the call commits only directly after it, however late its commit runs
    // (`storage::chain_gate`, "The commit-time guard"). An unchained call's
    // guard is its start record, attached once read (step 7).
    let operation = (!header.operation_id.is_empty()).then(|| {
        let operation = Operation::new(
            &header.operation_id,
            operation_request_digest(layer_name, &ompf_bytes),
        );
        match &chain {
            Some(link) => operation.with_predecessor(&link.previous_operation_id),
            None => operation,
        }
    });

    // Step 6: OMPF + delta parse pipeline. Runs AFTER the mapping
    // resolve so a non-existent (and non-recoverable) layer short-
    // circuits before the parse cost is paid. Two processing steps: the
    // zero-copy container parse gives the delta and external-value part
    // lengths, and the reservation holds the parse charge before the delta
    // is parsed, then the processing charge of its commit path once it is.
    ctx.at(RpcStage::Parse);
    let container = parse_container(&ompf_bytes)?;
    let lengths = UploadLengths::of(&ompf_bytes, &container);
    reserve_processing_step(
        svc,
        &mut reservation,
        lengths.parse_charge(),
        session_id,
        layer_name,
        ctx,
    )
    .await?;
    // Shared, never deep-copied by this handler: the commit and the
    // recovery re-issue below each take an `Arc` clone of it.
    let owned = Arc::new(
        if ompf_bytes.is_empty() || container.delta_bytes.is_empty() {
            // SDD §8.4: empty content — skip parsing, treat as empty delta. A
            // container with no delta part carries no delta either.
            None
        } else {
            Some(oodf_ffi::parse_delta(container.delta_bytes).map_err(ServiceError::from)?)
        },
    );
    // The per-delta checks (§9.4 caps, name characters, sample times,
    // self-parenting) before the processing charge, so a delta that fails
    // them is refused `INVALID_ARGUMENT` here, at stage `parse`, holding only
    // the parse charge, rather than charged its processing charge — and
    // perhaps queued for or run in the overflow slot — only to be refused by
    // `update_layer_with_operation`'s own check, which stays as the backstop.
    if let Some(d) = (*owned).as_ref() {
        validate_delta_bounds(d).map_err(ServiceError::from)?;
    }
    // The commit path, decided by the same predicate the commit dispatch
    // uses, so the charge matches the path the commit takes.
    let path = if svc.layer_store.takes_fast_path(&owned) {
        CommitPath::UpdateFast
    } else {
        CommitPath::UpdateSlow
    };
    // The second step also takes a hint of what a structural commit will
    // hold for the object it edits (SDD §5.4): the cached entry, its state
    // and plan indexes, under `lua`, the WATCH attempt's fixed cost on the
    // rollback backends.
    // This step may wait, holding nothing else; the commit's own charge
    // points never do, so they start from the hint. On the WATCH path the
    // first of them (WA0) is then a no-op. On a cache hit the first is the
    // repair's RP0, which grows past the hint by the first repair read's
    // reply term (per entry and per byte at its entry and byte bounds,
    // within 1 MiB by the first-read rule, at `STRUCTURAL_REPAIR_INITIAL_ENTRIES`
    // and `STRUCTURAL_REPAIR_INITIAL_BYTES`) and any growth of the entry
    // since the hint; refused near the budget, the call releases its permits and
    // entry, keeps this step and the hint, and recharges once for the
    // difference. The commit charge's base is the processing charge alone, so
    // the hint is its first object-state term.
    let processing = lengths.processing_charge(path, DeltaFootprint::of((*owned).as_ref()));
    let object_state_hint = svc
        .layer_store
        .object_state_hint(object_id, &owned)
        .map_or(0, |need| object_state_charge(&need));
    reserve_processing_step(
        svc,
        &mut reservation,
        processing.saturating_add(object_state_hint),
        session_id,
        layer_name,
        ctx,
    )
    .await?;

    // Step 6b: Store external values BEFORE the commit, on every route (SDD
    // §11.1). External value keys live on different hash slots, so they cannot
    // ride the commit's script call or transaction. Empty content carries
    // none. An external hash that is not exactly 32 bytes, or an unresolvable
    // `ext_value_index`, is refused INVALID_ARGUMENT by the collector before
    // any blob is stored, so the commit is never reached.
    //
    // Checked once and kept (`PreparedExternalValues`): every `update_layer`
    // call of this RPC is preceded by its own pre-store, and the recovery
    // re-issue below stores the same values again without checking them a
    // second time.
    let prepared = if ompf_bytes.is_empty() {
        None
    } else {
        // An upload whose container carries external parts first waits at the
        // external-value release budget (b.yuo), as on CreateLayer.
        if !container.external_values.is_empty() {
            await_ext_release_budget(svc, ctx).await?;
        }
        let prepared = prepare_external_values(
            (*owned).as_ref(),
            &container.external_values,
            svc.ext_value_store.as_ref(),
        )
        .map_err(|e| -> Status { e.into() })?;
        prepared
            .store(svc.ext_value_store.as_ref())
            .await
            .map_err(|e| -> Status { e.into() })?;
        Some(prepared)
    };

    // Step 7: the commit (SDD §7.12 step 6) — fast path, cached structural
    // route or WATCH/MULTI/EXEC, as `update_layer_with_summary` dispatches
    // it — wrapped in
    // the second half of the at-most-once recovery-retry branch. The
    // `recovered_once` flag is shared with step 3 — if step 3 already
    // triggered a recovery attempt, a NotFound here surfaces directly
    // (no further recovery).
    //
    // Delta ownership: `update_layer_with_summary` takes the delta as a
    // shared `Arc<Option<OwnedDelta>>` and only borrows it (its slow path
    // clones once per slow attempt, the cached structural route once per
    // plan), so the rare recovery re-issue reuses the
    // same parsed delta through another `Arc` clone — no backup copy is held.
    // b.gze: the writer's raw inbound OMPF is relayed verbatim to steady-state
    // subscribers (set_order + burst fields are destroyed by apply-time
    // materialization and cannot be reconstructed — see SDD §7.9). Persist it
    // on the per-object stream entry so `run_steady_state` can forward it.
    // Verbatim includes a time sample's `-0.0` time: the parse above
    // canonicalized it to `0.0` for the stored row and every reconstruction
    // (catch-up, checkpoints), but the relayed bytes keep the writer's bits,
    // so a steady-state subscriber can receive `-0.0` where a late joiner
    // receives `0.0`. The relay is deliberately not rewritten.
    // `Bytes` is refcounted, so the clone for the recovery-retry path is cheap.
    // Empty OMPF (`ompf_bytes.is_empty()`) carries no relay payload — pass
    // `None`, and the entry's missing `ompf` field marks the commit empty: a
    // subscriber sends it as the empty delta at its sequence with no read.
    let relay_ompf: Option<Bytes> = if ompf_bytes.is_empty() {
        None
    } else {
        Some(ompf_bytes.clone())
    };
    // SR-1.2 write-side enforcement. Checked ONCE here, at the point where the
    // two derivations meet and before either value is moved into either of the
    // two `update_layer_with_summary` call sites below — a refactor that
    // decouples them is exactly what this guards. See the fn's doc-comment for why a
    // currently-tautological assert is deliberate rather than dead.
    debug_assert_relay_invariant(&owned, &relay_ompf);
    // Everything from here to the return is the commit against an object the
    // mapping already proved live, including the recovery re-issue: a
    // NOT_FOUND / FAILED_PRECONDITION raised inside it is keyspace damage,
    // not a client mistake, and the b.94g level table reads exactly this
    // pair (`Storage`, resolved) to say so.
    ctx.at(RpcStage::Storage);
    // Resolve the call-start read now, just before the first commit attempt.
    // A failure ends the call before any attempt is sent, as `UNAVAILABLE` —
    // before the commit marker below, so it carries `UPSTREAM_UNAVAILABLE`
    // ([`refused_before_commit`]). (A
    // wrong-typed record key reads as no start record; the first attempt then
    // refuses it as `WRONGTYPE_KEY`, exactly as for a call without an id.)
    // External values stored above are then refcount-safe via the GC sweep, as
    // for any call that fails before committing.
    let operation = match (operation, start_record_read) {
        (Some(op), Some(read)) => {
            let start = svc
                .layer_store
                .resolve_start_record(read)
                .await
                .map_err(|e| -> Status { e.into() })?;
            Some(op.with_start_record(start))
        }
        (operation, _) => operation,
    };
    // The commit charge (SDD §5.4): the reservation as the commit sees it,
    // holding the processing charge as its base and the object-state hint
    // above it. Each commit grows it before the memory a structural commit
    // holds for the object it edits, never waiting, and waits for it only
    // between attempts, holding nothing else. Shared by the first call and
    // the recovery re-issue, and the re-issue's pre-store holds its extra
    // copy on it for that store only. A shutdown during that wait answers the same
    // shutdown status a processing step's does (its detail and retry hint
    // included, dropped only as SDD §18.1 drops any refusal detail after a
    // counted retry) and marks the completion shutdown-caused, as a
    // processing step's does.
    let mut commit_charge = UploadCommitCharge::new(
        &mut reservation,
        processing,
        &svc.shutdown,
        session_id,
        layer_name,
    );
    // Each call's summary (attempts, path, conflicts by kind, commit size) is
    // recorded on `ctx`, so this RPC's single failure line carries the call
    // whose outcome it reports (Spec SDD SR-2.6); the store's own commit lines
    // carry it for a landed commit.
    //
    // The commit marker ([`refused_before_commit`]): from here on — every
    // commit attempt, the recovery and its re-issue — a plain `UNAVAILABLE` is
    // answered as it is, because an attempt may have sent its commit. No step
    // before this marker writes client-visible state; a step that would must
    // move the marker before itself.
    ctx.commit_started();
    let (first_attempt_result, first_call) = svc
        .layer_store
        .update_layer_with_summary(
            session_id,
            layer_name,
            object_id,
            Arc::clone(&owned),
            relay_ompf.clone(),
            client_id,
            client_timestamp,
            operation.clone(),
            &mut commit_charge,
        )
        .await;
    if commit_charge.shut_down() {
        ctx.shutdown_caused();
    }
    let first_call_attempts = first_call.as_ref().map_or(0, |call| call.attempts);
    ctx.update_layer_call(first_call);
    let commit = match first_attempt_result {
        Ok(ok) => ok,
        // Post-commit NotFound recovery arm. Carries BOTH origins of "the
        // object this RPC resolved is gone" (b.f1v merged the second in):
        //
        // 1. the layer mapping vanished mid-RPC — any of the storage layer's
        //    retirement verdicts, carrying "layer '<name>' not found in
        //    session", including `disambiguate_stream_collision`'s
        //    `mapping_deleted`, its `read_failed_verdict`'s `tombstone_entry`
        //    arm, and the EXEC-reconcile tombstone fallback;
        // 2. the mapping still resolves but the object header is gone — the
        //    natural symptom of a partial Valkey state wipe (UNLINK every
        //    `le:{oid}:obj:*` while the mapping survives). Usually the message
        //    "object no longer exists (header deleted)", the storage layer's
        //    shared string across the four sites that guard the header up
        //    front (the txn empty-`HGETALL` guard, the Lua `NOT_FOUND` reply
        //    mapping, the stream-collision disambiguation's `header_deleted`
        //    verdict, and the structural cache's read-side header mapping,
        //    which every structural attempt under the `lua` backend runs
        //    before it plans) — but NOT only those: a header UNLINKed
        //    inside the slow path's commit window, after that guard and before
        //    the snapshot load, surfaces from the snapshot read's own header
        //    check (`read_object_snapshot`) as "object <id> not found" instead.
        //
        // Neither list is closed, and the arm does not need them to be. Both
        // origins want identical handling — one recovery attempt, then
        // re-issue — so they share one arm keyed on the VARIANT, which is what
        // makes the open-endedness safe: ANY producer of `NotFound` for a gone
        // object is routed correctly without this handler having to know the
        // site or the wording. Keying on the variant is also what keeps the arm
        // armed — the deleted b.fgj arm strict-equality matched a message, one
        // cosmetic edit away from silently disarming recovery, and it already
        // missed the snapshot-read shape above. The storage-side
        // constant is module-private for the same reason: "no handler matches
        // this string" is now compiler-enforced, and the messages quoted here
        // name shapes for a reader rather than feeding any dispatch.
        //
        // Origin 1 cannot resurrect a deliberately-deleted layer, and that
        // safety does not live here — it is `apply.rs`'s SR-2.4 identity
        // guard (`live_mapping_resolves_blob_identity`), which re-reads the
        // blob's OWN `(session_id, layer_name)` mapping: absent or pointing
        // at a different object_id → `Ok(false)` → `apply_restore` returns
        // `CheckpointError::NotFound` → `RecoveryOutcome::NotRecoverable` →
        // this arm surfaces the original `NotFound` having written nothing.
        // Both retirement orderings cooperate with that guard by
        // construction: `delete_layer` DELs the mapping BEFORE the header
        // UNLINK, and a CreateLayer overwrite flips the mapping to a fresh
        // object_id — so by the time a commit can observe either retirement,
        // the guard's read already fails. Pre-existing property; stated here
        // because "recover on NotFound" reads alarming without it.
        //
        // Deliberately NOT caught here: every `FailedPrecondition` shape
        // (object_id mismatch, partial header, superseded, SEQ_OVERFLOW /
        // SET_ORDER_OVERFLOW). None is a data-loss condition, and a recovery
        // attempt on them would be wrong — object_id mismatch in particular
        // means the caller holds stale state, not that the server was wiped
        // (pinned by `recovery_hook_test::
        // update_layer_failed_precondition_does_not_invoke_recovery`), and the
        // partial header the commit path refuses is a client-visible
        // PRECONDITION failure, not a data-loss report: the caller is told its
        // write did not apply, and quietly converting that into a restore would
        // replace the live object with checkpoint state on an RPC that never
        // asked for one. Note this is a statement about THIS arm's scope, not
        // about the header itself — since b.f4t a partial header is restorable,
        // and the recovery hook does restore over one when it is reached
        // through `NotFound`. What keeps the two apart is that this match is
        // keyed on the VARIANT: `FailedPrecondition` structurally cannot land
        // in the `NotFound` arm below, whatever its message says.
        Err(ServiceError::NotFound(msg)) => {
            if recovered_once {
                // Step 3 already attempted recovery on this RPC. A
                // NotFound here means the layer was recovered then
                // immediately deleted; surface NotFound directly.
                //
                // Logged at its own site rather than folded into the event
                // below: "a restore landed and the object vanished again
                // inside one RPC" is a different — and more alarming —
                // operational signal than "the object is gone, trying a
                // restore", and it is otherwise invisible on this side (the
                // storage producers log nothing here, and only the hook's
                // metric moves).
                tracing::warn!(
                    session_id = %session_id,
                    layer_name = %layer_name,
                    object_id,
                    storage_message = %msg,
                    "UpdateLayer object gone again after a successful checkpoint recovery on \
                     this RPC; surfacing NOT_FOUND"
                );
                return Err(ServiceError::NotFound(msg).into());
            }
            // Below the latch on purpose: the event claims a recovery attempt
            // is being made, and the early return above makes none. The
            // message is the only thing that now distinguishes the two origins
            // above, and the path is exceptional (a wipe or a concurrent
            // delete), so the log volume is nil.
            tracing::warn!(
                session_id = %session_id,
                layer_name = %layer_name,
                object_id,
                storage_message = %msg,
                "UpdateLayer commit found the object gone; attempting checkpoint recovery"
            );
            match svc.recovery.try_restore_layer(object_id).await {
                RecoveryOutcome::Restored => {
                    // Re-issue the commit ONCE with the same shared delta,
                    // after its own pre-store of the RPC's external values
                    // ([`prestore_for_recovery_reissue`]). The re-issue is an
                    // ordinary call: its increments are its own, so it hands
                    // them back on the same verdicts any call does, and the
                    // first call's increments are never spent or withheld on
                    // its behalf. The first call's `+1` leaks — the safe
                    // direction, drained by the reference-count reconciler —
                    // when that call's commit never landed, or the restore
                    // kept its reference.
                    //
                    // The re-pre-store does not wait at the release budget
                    // again: this RPC already passed it, and the second store
                    // writes new blob bytes only where GC deleted a blob since
                    // the first one, which frees the same bytes (otherwise it
                    // only increments). It holds one more copy of the values
                    // on the upload budget while it runs, given back before
                    // the re-issue's first attempt.
                    //
                    // A failed re-pre-store (the extra-copy charge refused, or
                    // the store itself) ends the RPC here: the re-issue never
                    // runs, and committing on a partial pre-store would spend
                    // increments it did not pay for. Its error is answered as
                    // a re-issue error, by the arms below, and the first
                    // call's summary stays the RPC's.
                    let prestored =
                        prestore_for_recovery_reissue(svc, prepared.as_ref(), &mut commit_charge)
                            .await;
                    if commit_charge.shut_down() {
                        ctx.shutdown_caused();
                    }
                    let (reissue_result, reissue_attempts) = match prestored {
                        Err(e) => (Err(e), 0),
                        Ok(()) => {
                            let (result, reissue_call) = svc
                                .layer_store
                                .update_layer_with_summary(
                                    session_id,
                                    layer_name,
                                    object_id,
                                    owned,
                                    relay_ompf.clone(),
                                    client_id,
                                    client_timestamp,
                                    operation,
                                    &mut commit_charge,
                                )
                                .await;
                            if commit_charge.shut_down() {
                                ctx.shutdown_caused();
                            }
                            let attempts = reissue_call.as_ref().map_or(0, |call| call.attempts);
                            ctx.update_layer_call(reissue_call);
                            (result, attempts)
                        }
                    };
                    match reissue_result {
                        Ok(ok) => ok,
                        // A refusal detail promises this RPC applied nothing.
                        // The re-issue's own tail drops it after its own
                        // counted retries; this drops it after the first
                        // call's: an earlier attempt of that call may have
                        // sent its commit and lost the reply, and it can still
                        // land (SDD §18.1). That covers a chained re-issue's
                        // predecessor refusal at its commit too — after a
                        // restore wiped the record, `last_record=absent` —
                        // and a refused re-pre-store, whose re-issue made no
                        // attempt (`reissue_attempts` is 0).
                        Err(
                            ServiceError::UnavailableWithDetail { message, .. }
                            | ServiceError::PredecessorNotCommitted { message, .. },
                        ) if first_call_attempts > 1 => {
                            return Err(outcome_unknown_after_retry(
                                first_call_attempts + reissue_attempts,
                                &message,
                            )
                            .into());
                        }
                        // Second NotFound (or any other error on retry, or of
                        // the re-pre-store) surfaces verbatim — no further
                        // recovery.
                        Err(e) => return Err(e.into()),
                    }
                }
                RecoveryOutcome::NotRecoverable => {
                    return Err(ServiceError::NotFound(msg).into());
                }
                RecoveryOutcome::Error(err) => {
                    return Err(ServiceError::from_recovery_error(
                        &err,
                        format!("checkpoint recovery failed for object {object_id}: {err}"),
                    )
                    .into());
                }
            }
        }
        Err(other) => return Err(other.into()),
    };

    // Per-commit is debug!: collaborative-editing workloads commit hundreds
    // of updates per second per session, and an info log per commit floods
    // production logs without adding information that metrics don't already
    // surface (b.jr8 lows). Skipped for an `AlreadyCommitted` answer, which
    // completed nothing here (the store already logged it at INFO), matching
    // the CreateLayer handler.
    if commit.outcome != OperationOutcome::AlreadyCommitted {
        tracing::debug!(
            session_id = %session_id,
            layer_name = %layer_name,
            object_id,
            sequence_num = commit.seq,
            client_id = %client_id,
            client_timestamp,
            "UpdateLayer completed"
        );
    }

    Ok(Response::new(UpdateLayerResponse {
        sequence_num: commit.seq,
        server_timestamp_microseconds_since_epoch: commit.server_ts,
        operation_outcome: wire_outcome(commit.outcome),
    }))
}

/// Resolve `layer_name`'s current object (SDD §7.12 step 3), with the
/// at-most-once layer-state recovery (`recovered_once`, shared with the rest
/// of the RPC) when its mapping is missing; `NOT_FOUND` when it stays missing.
/// The caller compares the answer with the submitted `object_id`.
async fn resolve_layer_object(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    recovered_once: &mut bool,
) -> Result<u64, Status> {
    loop {
        match svc
            .layer_store
            .get_layer_mapping(session_id, layer_name)
            .await
            .map_err(|e| -> Status { e.into() })?
        {
            Some(oid) => return Ok(oid),
            None => {
                if *recovered_once {
                    return Err(ServiceError::NotFound(format!(
                        "layer '{layer_name}' not found in session"
                    ))
                    .into());
                }
                match svc.recovery.try_restore_layer(object_id).await {
                    RecoveryOutcome::Restored => {
                        *recovered_once = true;
                        continue;
                    }
                    RecoveryOutcome::NotRecoverable => {
                        return Err(ServiceError::NotFound(format!(
                            "layer '{layer_name}' not found in session"
                        ))
                        .into());
                    }
                    RecoveryOutcome::Error(err) => {
                        return Err(ServiceError::from_recovery_error(
                            &err,
                            format!("checkpoint recovery failed for object {object_id}: {err}"),
                        )
                        .into());
                    }
                }
            }
        }
    }
}

/// The `FAILED_PRECONDITION` of an `UpdateLayer` whose layer names
/// `current_object_id`, not the submitted `object_id`.
fn object_id_mismatch(current_object_id: u64, object_id: u64) -> ServiceError {
    ServiceError::FailedPrecondition(format!(
        "object_id mismatch: expected {current_object_id}, got {object_id}"
    ))
}

/// What the chain gate needs of a chained `UpdateLayer`
/// ([`await_chain_predecessor`]).
struct ChainedWrite<'a> {
    session_id: &'a str,
    layer_name: &'a str,
    object_id: u64,
    client_id: &'a str,
    operation_id: &'a str,
    link: &'a ChainRequest,
    /// The call's reservation as it enters the gate: `F` + its content.
    held_bytes: u64,
    /// `T`: the call's deadline budget, read once at the handler's entry
    /// (`grpc::deadline::remaining`, or `LIVE_EDIT_RPC_DEADLINE_MS` when no
    /// deadline is in scope). The gate's margin is derived from it.
    budget: std::time::Duration,
    /// `D`: the instant of that read plus `budget`.
    deadline: tokio::time::Instant,
}

/// Step 5b of [`handle_update_layer`]: wait in the chain gate
/// (`storage::chain_gate`) until the latest operation recorded for the call's
/// `(object_id, client_id)` is its predecessor or its own operation. Stage
/// `chain`.
///
/// - `ready`: go on to the parse and commit. Advisory: the commit's
///   predecessor guard decides (`storage::chain_gate`, the commit-time
///   guard).
/// - A refusal: `ABORTED` `PREDECESSOR_NOT_COMMITTED` (`no_wait`, or when
///   only the call's margin is left before its deadline), or `UNAVAILABLE`
///   `CHAIN_CHECK_UNAVAILABLE`.
/// - `replaced` (the header gone or superseded): the layer is resolved again,
///   as in step 3 and sharing its one recovery. Another object answers the
///   `object_id` mismatch, no mapping its `NOT_FOUND`; the same object (a
///   `CreateLayer` overwrite between its two steps, a header lost under a live
///   mapping, or the object restored from a checkpoint) keeps the call
///   waiting ([`ChainGate::keep_waiting`]).
/// - Graceful shutdown: the wait races an on-the-spot observation of the
///   shutdown's `begun` token, and ends with the shared shutdown status,
///   marking the completion shutdown-caused (INFO), with nothing applied.
///
/// The bound is the call's deadline `D` ([`ChainedWrite::deadline`]): the
/// handler's entry plus the deadline budget `T` it read there, before its
/// first await ([`ChainedWrite::budget`]: `grpc::deadline::remaining`, the
/// sooner of the server deadline and the client's `grpc-timeout`, or
/// `LIVE_EDIT_RPC_DEADLINE_MS` when none is in scope). The gate's margin is
/// `M = max(T / 10, 50 ms)` (`chain_gate::chain_deadline_margin`): it sleeps
/// no later than `D − M`, and its final probe is the first to start at or
/// after that. Every floor below holds to within the timer's precision. A
/// call that waited until its margin is refused at least `M/2` before `D`. A
/// call that arrives with no more than `M` left is checked once, without
/// waiting, and refused at least half of what it had left when checked. The
/// one exception to both floors is the same-object re-resolve (`replaced`,
/// above), which the gate does not bound: the call is then probed again, a
/// late arrival too. A waiter keeps the `M/2` floor only if the re-resolve
/// finished before `D − M/2`; a late arrival keeps only the bound below,
/// counted from that later probe; a re-resolve that finishes after
/// `D − M/2` leaves the call's next probe a limit already passed, so the call
/// is refused once it finishes (one that runs past `D` is not refused: the
/// call ends unrefused at `D` — `MetricsLayer` ends it `DEADLINE_EXCEEDED`
/// when `LIVE_EDIT_RPC_DEADLINE_MS` set `D`, and when a shorter client
/// `grpc-timeout` did, tonic's `GrpcTimeout` drops it and `RecoverError`
/// answers `CANCELLED "Timeout expired"`; either way the client normally
/// reads its own `DEADLINE_EXCEEDED` or `CANCELLED` first, unsettled). On
/// every path
/// the gate's own refusal (`ABORTED` or `CHAIN_CHECK_UNAVAILABLE`) leaves at
/// least the lesser of `M/2` and half of what was left when the call's last
/// probe started (`storage::chain_gate`, the deadline cap). The margin does
/// not bound the re-resolve's own answer — the `object_id` mismatch, its
/// `NOT_FOUND`, or an error it met — which is sent whenever it finishes.
/// Awaited inline, never spawned: a client cancel or a lost connection drops
/// this future with the handler, and the gate's drop guard records it.
async fn await_chain_predecessor(
    svc: &LiveEditService,
    ctx: &mut RpcLogContext,
    write: ChainedWrite<'_>,
    recovered_once: &mut bool,
) -> Result<(), Status> {
    ctx.at(RpcStage::Chain);
    let mut gate = ChainGate::enter(
        &svc.layer_store,
        Some(Arc::clone(&svc.metrics)),
        write.object_id,
        write.client_id,
        ChainLink {
            operation_id: write.operation_id,
            previous_operation_id: &write.link.previous_operation_id,
            no_wait: write.link.no_wait,
        },
        write.deadline,
        write.budget,
        write.held_bytes,
    );
    loop {
        let exit = tokio::select! {
            biased;
            () = svc.shutdown.begun().cancelled() => None,
            exit = gate.wait() => Some(exit),
        };
        match exit {
            None => {
                gate.finish_shutdown();
                ctx.shutdown_caused();
                return Err(server_shutting_down());
            }
            Some(GateExit::Ready) => return Ok(()),
            Some(GateExit::Refused(refusal)) => return Err(refusal.into()),
            Some(GateExit::Replaced) => {
                let resolved = resolve_layer_object(
                    svc,
                    write.session_id,
                    write.layer_name,
                    write.object_id,
                    recovered_once,
                )
                .await;
                match resolved {
                    Ok(current) if current == write.object_id => gate.keep_waiting(),
                    Ok(current) => {
                        gate.finish_replaced();
                        ctx.resolved_object_id(current);
                        return Err(object_id_mismatch(current, write.object_id).into());
                    }
                    Err(status) => {
                        gate.finish_replaced();
                        return Err(status);
                    }
                }
            }
        }
    }
}

/// Admit one upload of `session_id` into the replica's upload budget.
///
/// A capacity refusal answers its `UNAVAILABLE` (this call applied nothing;
/// [`UploadRefusal::into_service_error`], and for what that promises a client
/// about a re-send, [`REFUSAL_DETAIL_DOMAIN`](crate::error::REFUSAL_DETAIL_DOMAIN)),
/// with the retry hint
/// [`UPLOAD_CAPACITY`](crate::error::refusal_reason::UPLOAD_CAPACITY).
/// The wait races the graceful shutdown's `begun` token: an upload waiting or
/// arriving once the paced stream ending has started is refused with the
/// shared shutdown status, which marks the completion shutdown-caused, so it
/// logs at INFO like every other shutdown ending.
async fn admit_upload(
    svc: &LiveEditService,
    rpc: UploadRpc,
    session_id: &str,
    ctx: &mut RpcLogContext,
) -> Result<UploadReservation, Status> {
    match svc
        .upload_admission
        .admit(rpc, session_id, svc.shutdown.begun())
        .await
    {
        Ok(reservation) => Ok(reservation),
        Err(UploadRefusal::ShuttingDown) => {
            ctx.shutdown_caused();
            Err(server_shutting_down())
        }
        Err(refusal) => Err(refusal.into_service_error().into()),
    }
}

/// Wait at the external-value release budget for an upload whose container
/// carries external parts (b.yuo): proceed while the store's released blobs
/// awaiting garbage collection are below `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` and
/// no out-of-memory GC collection holds uploads back (b.pa1; that hold shares
/// this wait and never refuses on its own), or
/// once both hold within what is left of this call's deadline minus
/// the release gate's own margin (`crate::grpc::deadline::remaining`, capped at
/// `LIVE_EDIT_RPC_DEADLINE_MS`; `ExtReleaseGate::wait_for`); otherwise refuse
/// `UNAVAILABLE` (this call applied nothing; a re-send is safe unless an
/// earlier attempt of the same write is still in doubt — then use
/// `operation_id`; [`ExtReleaseRefusal::into_service_error`]), with the retry
/// hint [`EXT_RELEASE_BUDGET_FULL`](crate::error::refusal_reason::EXT_RELEASE_BUDGET_FULL).
/// The wait races the graceful
/// shutdown's `begun` token, as the two upload-admission waits do: a shutdown
/// while waiting answers the shared shutdown status and marks the completion
/// shutdown-caused. No gate (a disabled budget) admits at once. Runs at stage `parse`, just before
/// the external-value store, so a refusal stores, increments, mirrors and
/// commits nothing. See `storage::ext_release_gate`.
async fn await_ext_release_budget(
    svc: &LiveEditService,
    ctx: &mut RpcLogContext,
) -> Result<(), Status> {
    let Some(gate) = svc.ext_value_store.ext_release_gate() else {
        return Ok(());
    };
    match gate
        .admit(crate::grpc::deadline::remaining(), svc.shutdown.begun())
        .await
    {
        Ok(()) => Ok(()),
        Err(ExtReleaseRefusal::ShuttingDown) => {
            ctx.shutdown_caused();
            Err(server_shutting_down())
        }
        Err(refusal) => Err(refusal.into_service_error().into()),
    }
}

/// The checkpoint-recovery re-issue's own pre-store (SDD §11.1, §11.3): store
/// the RPC's external values again ([`PreparedExternalValues::store`], checked
/// once by step 6b and not again), so the re-issue pays for the increments it
/// may spend and hands back only its own. `None` (empty content) or an empty
/// set stores nothing and charges nothing.
///
/// In order, holding nothing but the call's reservation:
///
/// 1. grow `commit_charge` by one copy of the stored bytes
///    ([`PreparedExternalValues::stored_bytes`]) through its waiting step
///    ([`CommitCharge::hold`] → [`UploadReservation::reserve_processing`]).
///    `e_ext` prices one owned copy per value, but with checkpointing on a
///    copy of the first store handed to the ext-mirror can outlive that store
///    and be alive beside this store's copy and write buffer; this charges
///    that copy, for this store only. (The mirror's own byte budget charges it
///    too while the mirror holds it; counting it twice errs on the safe
///    side.) A refusal for shutdown is recorded on
///    the charge ([`UploadCommitCharge::shut_down`]);
/// 2. store;
/// 3. shrink the charge back to what it held before step 1, on success and
///    on failure ([`CommitCharge::try_hold`] to a lower or equal target, which
///    never waits), so the extra copy never shows in what the re-issue holds
///    ([`CommitCharge::held`] feeds its rewrite allowance and
///    `live_edit_upload_object_state_bytes`). A deadline drop or a shutdown
///    frees it with the reservation.
///
/// No release-budget wait (`storage::ext_release_gate`): the RPC already
/// passed it, and this store writes new blob bytes only where GC deleted a
/// blob since the first store, which freed the same bytes; otherwise it only
/// increments. A second wait would also make the recovering RPC likelier to
/// be refused right after its own restore raised the pending total, with a
/// refusal that promises "applied nothing" although the first call may have
/// landed.
///
/// # Errors
///
/// Step 1's refusal or the store's error. The caller ends the RPC on it,
/// before the re-issue, and answers it as a re-issue error. The hashes the
/// store wrote before failing leak (reconciler-recoverable), as on a first
/// pre-store that fails.
async fn prestore_for_recovery_reissue(
    svc: &LiveEditService,
    prepared: Option<&PreparedExternalValues<'_>>,
    commit_charge: &mut UploadCommitCharge<'_>,
) -> Result<(), ServiceError> {
    let Some(prepared) = prepared.filter(|prepared| !prepared.is_empty()) else {
        return Ok(());
    };
    let held = commit_charge.held();
    let stored = match commit_charge
        .hold(held.saturating_add(prepared.stored_bytes()))
        .await
    {
        Ok(()) => prepared.store(svc.ext_value_store.as_ref()).await,
        Err(refused) => Err(refused),
    };
    let given_back = commit_charge.try_hold(held);
    debug_assert!(
        given_back,
        "a commit charge at or below what the reservation holds is always granted"
    );
    stored
}

/// The zero-copy OMPF container of an upload's content. Empty content (SDD
/// §8.3, §8.4) is an empty container.
fn parse_container(ompf_bytes: &[u8]) -> Result<OmpfContainer<'_>, ServiceError> {
    OmpfContainer::parse(ompf_bytes)
        .map_err(|e| ServiceError::InvalidArgument(format!("malformed OMPF: {e}")))
}

/// An upload's body, delta-part and external-value-part lengths, from the
/// zero-copy container parse: what its processing charges are computed from.
#[derive(Debug, Clone, Copy)]
struct UploadLengths {
    body: u64,
    delta: u64,
    ext: u64,
}

impl UploadLengths {
    fn of(ompf_bytes: &[u8], container: &OmpfContainer<'_>) -> Self {
        let len = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        Self {
            body: len(ompf_bytes.len()),
            delta: len(container.delta_bytes.len()),
            ext: container
                .external_values
                .iter()
                .fold(0u64, |sum, value| sum.saturating_add(len(value.len()))),
        }
    }

    /// The first processing step's charge, held before the delta is parsed.
    fn parse_charge(self) -> u64 {
        parse_charge(self.body, self.delta, self.ext)
    }

    /// The second processing step's charge, once the delta is parsed and its
    /// commit path known.
    fn processing_charge(self, path: CommitPath, footprint: DeltaFootprint) -> u64 {
        processing_charge(path, self.body, self.delta, self.ext, footprint)
    }
}

/// One processing step: hold `target` bytes for the rest of the call, in the
/// upload budget or in its overflow slot
/// ([`UploadReservation::reserve_processing`]). Never refused for size; it
/// waits, up to the RPC deadline, for room or for the slot.
///
/// An upload that enters the overflow slot logs one INFO line, "upload runs
/// beyond the upload budget". The wait races the graceful shutdown's `begun`
/// token: a shutdown while waiting answers the shared shutdown status and
/// marks the completion shutdown-caused (stage `parse`), with nothing applied
/// — the external-value store and the commit have not run.
async fn reserve_processing_step(
    svc: &LiveEditService,
    reservation: &mut UploadReservation,
    target: u64,
    session_id: &str,
    layer_name: &str,
    ctx: &mut RpcLogContext,
) -> Result<(), Status> {
    match reservation
        .reserve_processing(target, svc.shutdown.begun())
        .await
    {
        Ok(grant) => {
            if grant.entered_overflow {
                log_upload_runs_beyond_budget(
                    reservation,
                    session_id,
                    layer_name,
                    target,
                    grant.waited,
                );
            }
            Ok(())
        }
        Err(UploadRefusal::ShuttingDown) => {
            ctx.shutdown_caused();
            Err(server_shutting_down())
        }
        Err(refusal) => Err(refusal.into_service_error().into()),
    }
}

/// Parse a `CreateLayer` payload's container into its delta and its
/// external-value parts.
///
/// Empty content (SDD §8.3) and a container with no delta both yield `None`.
/// Shared by the create path and the re-send path
/// ([`creation_state_for_resend`]) so the state a re-send rebuilds is built
/// exactly as the original's was.
fn parse_create_payload(
    container: OmpfContainer<'_>,
) -> Result<(Option<oodf_ffi::OwnedDelta>, Vec<&[u8]>), ServiceError> {
    let owned = if container.delta_bytes.is_empty() {
        None
    } else {
        Some(oodf_ffi::parse_delta(container.delta_bytes).map_err(ServiceError::from)?)
    };
    Ok((owned, container.external_values))
}

/// Rebuild the creation state of a `CreateLayer` re-send answered
/// `ALREADY_COMMITTED`, for the recorded object's first-base re-spawn.
///
/// The same two charges, parse, per-delta checks and apply as the create path,
/// minus the external-value store: the original call already stored (and
/// mirrored) those values, and storing them again would take references
/// nothing releases. No delta counters are bumped — the original call counted
/// its delta.
///
/// The charges are taken in budget or not at all, without waiting
/// ([`UploadReservation::try_grow_in_budget`]): the re-spawn is best-effort,
/// so a refusal only skips it.
fn creation_state_for_resend(
    ompf_bytes: &[u8],
    reservation: &mut UploadReservation,
) -> Result<ObjectState, ServiceError> {
    let container = parse_container(ompf_bytes)?;
    let lengths = UploadLengths::of(ompf_bytes, &container);
    reservation
        .try_grow_in_budget(lengths.parse_charge())
        .map_err(UploadRefusal::into_service_error)?;
    let (owned, _) = parse_create_payload(container)?;
    // The per-delta checks before the processing charge, as on the create
    // path. Not reachable in practice — the re-sent bytes equal a committed
    // original's, which passed them — but it keeps every path refusing a
    // delta that fails them before its processing charge.
    if let Some(d) = owned.as_ref() {
        validate_delta_bounds(d)?;
    }
    reservation
        .try_grow_in_budget(
            lengths.processing_charge(CommitPath::Create, DeltaFootprint::of(owned.as_ref())),
        )
        .map_err(UploadRefusal::into_service_error)?;
    let mut state = ObjectState::default();
    let mut stats = crate::domain::delta::ApplyDeltaStats::default();
    let mut sink = crate::domain::delta::MutationSink::Discard;
    crate::domain::delta::apply_delta_into(&mut state, owned, &mut sink, &mut stats)
        .map_err(ServiceError::from)?;
    Ok(state)
}

/// Debug-build tripwire for the SR-1.2 ompf-classification invariant:
/// **every non-empty update commit must XADD the `ompf` field.**
///
/// Checkpoint tail capture classifies an `update`-typed `{seq}-0` stream entry
/// with no `ompf` field as a §8.4 *empty* commit — by invariant, not by byte
/// shape (the two are byte-identical on the wire; see `RELAY_OMPF_FIELD` in
/// `crate::storage::notifications`). If a state-changing commit ever reached
/// the stream without its relay OMPF, tail capture would record it as an empty
/// delta and the restore would replay a no-op over a real state change,
/// minting a generation whose claimed coverage exceeds its actual content —
/// the b.pw6 defect class this subsystem exists to forbid. `SubscribeReadLayer`
/// relies on the same classification: its steady state and its rollback-mode
/// in-window replay send such an entry (above sequence 1) as the empty delta
/// with no read, so a live subscriber would silently miss the commit's
/// changes too.
///
/// **This predicate is tautological today, and that is the point.** In
/// `handle_update_layer` both `owned` and `relay_ompf` derive from the same
/// `ompf_bytes.is_empty()` test, so the forbidden pair is unreachable through
/// the handler by construction. The assert is a deliberate tripwire against a
/// future refactor that decouples the two derivations — it is not dead code to
/// be deleted on the grounds that it "can never fire". `handle_update_layer`
/// is the only production caller of `LayerStore::update_layer_with_summary`
/// (which `LayerStore::update_layer_with_operation`, and so its `update_layer`
/// wrapper, wraps), so this one call site covers the whole production surface.
///
/// Compiled out of release builds (`debug_assert!`). The storage API refuses
/// the forbidden pair itself, in every build: `update_layer_with_summary`
/// answers it `Internal` before any store I/O (see
/// `update_layer_with_operation`'s `relay_ompf` parameter contract). This
/// assert names the handler-side derivation as the fault in a debug build
/// rather than leaving it to a refused commit, and the source-scan wiring
/// guard in the ompf-invariant test module fails if this call disappears from
/// the derivation point.
#[inline]
fn debug_assert_relay_invariant(owned: &Option<oodf_ffi::OwnedDelta>, relay_ompf: &Option<Bytes>) {
    debug_assert!(
        owned.is_none() || relay_ompf.is_some(),
        "relay-OMPF invariant: a state-changing UpdateLayer commit must carry relay OMPF; \
         an ompf-less update stream entry is classified as an empty delta by tail capture \
         and would replay as a no-op over a real commit"
    );
}

/// Handle StatLayer RPC (SDD §7.14).
///
/// 1. Validate input (session_id, layer_name).
/// 2. Auth: read permission on session base_path.
/// 3. Read layer mapping → object_id. NOT_FOUND if nil.
/// 4. Read object header → extract created_by, created_at, last_modified_at.
/// 5. Return StatLayerResponse.
///
/// **NotFound recovery (SDD SR-2.7).** Step 2's `get_base_path` funnels
/// through [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache-miss + session-absent triggers lazy on-demand
/// session-state recovery. The header-read NotFound in step 4 is recoverable
/// via the per-`object_id` layer-recovery loop below; the mapping-resolution
/// NotFound in step 3 is not (no `object_id` to key recovery by).
///
/// **Failure logging.** Wrapper over [`stat_layer_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`].
pub async fn handle_stat_layer(
    svc: &LiveEditService,
    request: Request<StatLayerRequest>,
) -> Result<Response<StatLayerResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("StatLayer", &request);
    let result = stat_layer_inner(svc, request, &mut ctx).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn stat_layer_inner(
    svc: &LiveEditService,
    request: Request<StatLayerRequest>,
    ctx: &mut RpcLogContext,
) -> Result<Response<StatLayerResponse>, Status> {
    let token = extract_token(&request)?;
    let req = request.into_inner();

    let session_id = &req.session_id;
    let layer_name = &req.layer_name;

    // b.39x H: full session_id validation (length cap + control-char
    // rejection). Replaces the pre-fix `is_empty()`-only check. The b.94g
    // `ctx` records sit below their validators for that same reason.
    validate_session_id(session_id)?;
    ctx.session(session_id);

    // Validate layer name.
    validate_layer_name(layer_name)?;
    ctx.layer(layer_name);

    // Auth: read permission on the session's base path.
    ctx.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, session_id)
        .await?;
    ctx.base_path(&base_path);
    check_read_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // Resolve layer mapping → object_id (SDD §7.14 step 2). A missing
    // mapping is a true `NotFound` — we have no `object_id` to key
    // recovery by (the recovery hook's checkpoint store is keyed on
    // `object_id`), so this path surfaces NotFound directly without
    // attempting recovery. The header-read NotFound (below) IS
    // recoverable because it happens after the mapping resolved.
    ctx.at(RpcStage::Resolve);
    let object_id = svc
        .layer_store
        .get_layer_mapping(session_id, layer_name)
        .await
        .map_err(|e| -> Status { e.into() })?
        .ok_or_else(|| -> Status {
            ServiceError::NotFound(format!("layer '{layer_name}' not found in session")).into()
        })?;
    ctx.resolved_object_id(object_id);

    // Read header metadata, wrapped in an at-most-once recovery-retry
    // branch (SDD SR-2.4 / SR-2.6, b.fgj Subtask t3.fgj.ya.qm.th). The
    // header may be missing post-mapping if a partial state-loss nuked
    // the `obj:hdr` hash. On NotFound, attempt
    // `RecoveryHook::try_restore_layer(object_id)`:
    //
    // - `Restored` → retry the header read once. Second NotFound surfaces
    //   to the client verbatim — the layer was recovered then immediately
    //   deleted, or apply's live-mapping guard pivoted to a different
    //   `object_id`.
    // - `NotRecoverable` → surface the original NotFound. Also the path
    //   the disabled-checkpoint short-circuit takes.
    // - `Error(_)` → classify the carried `CheckpointError` through
    //   `ServiceError::from_recovery_error` (b.33e): `Transient` →
    //   UNAVAILABLE so standard gRPC client retry policy picks it up (the
    //   backend is transiently broken but the request is fine to re-issue);
    //   every other variant → INTERNAL, because retrying a permanent /
    //   inconsistent-store failure cannot succeed.
    ctx.at(RpcStage::Storage);
    let mut recovered_once = false;
    let stat = loop {
        match svc.layer_store.stat_layer_for_object_id(object_id).await {
            Ok(s) => break s,
            Err(ServiceError::NotFound(msg)) => {
                if recovered_once {
                    return Err(ServiceError::NotFound(msg).into());
                }
                match svc.recovery.try_restore_layer(object_id).await {
                    RecoveryOutcome::Restored => {
                        recovered_once = true;
                        continue;
                    }
                    RecoveryOutcome::NotRecoverable => {
                        return Err(ServiceError::NotFound(msg).into());
                    }
                    RecoveryOutcome::Error(err) => {
                        return Err(ServiceError::from_recovery_error(
                            &err,
                            format!("checkpoint recovery failed for object {object_id}: {err}"),
                        )
                        .into());
                    }
                }
            }
            Err(other) => return Err(other.into()),
        }
    };

    tracing::info!(
        session_id = %session_id,
        layer_name = %layer_name,
        "StatLayer completed"
    );

    Ok(Response::new(StatLayerResponse {
        created_by_user_name: stat.created_by,
        created_at: stat.created_at,
        last_modified_at: stat.last_modified_at,
    }))
}

/// Handle DeleteLayer RPC (SDD §7.11).
///
/// 1. Validate input (session_id, layer_name).
/// 2. Auth: write permission on session base_path.
/// 3. Delegate to LayerStore::delete_layer.
///
/// Deadline-drop contract (b.yis sub-item 4, see `metrics_layer.rs`):
/// idempotent (second call returns NOT_FOUND). `LayerStore::delete_layer`
/// uses MULTI/EXEC for atomic removal of mapping + header + indices, and
/// any ext-value orphans self-heal via refcount GC.
///
/// **NotFound recovery (SDD SR-2.7).** Step 2's `get_base_path` funnels
/// through [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache-miss + session-absent triggers lazy on-demand
/// session-state recovery. A session that has been explicitly deleted (and
/// whose checkpoint has been pruned by the delete-cascade) still surfaces
/// `NotFound` because recovery returns `NoCheckpoint` in that case.
///
/// **Failure logging.** Wrapper over [`delete_layer_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`]. This handler never resolves a mapping of its
/// own, so its context never carries `current_object_id` and its `NOT_FOUND`
/// (delete of an absent layer) always reads as the client-caused shape.
pub async fn handle_delete_layer(
    svc: &LiveEditService,
    request: Request<DeleteLayerRequest>,
) -> Result<Response<DeleteLayerResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("DeleteLayer", &request);
    let result = delete_layer_inner(svc, request, &mut ctx).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn delete_layer_inner(
    svc: &LiveEditService,
    request: Request<DeleteLayerRequest>,
    ctx: &mut RpcLogContext,
) -> Result<Response<DeleteLayerResponse>, Status> {
    let token = extract_token(&request)?;
    let req = request.into_inner();

    let session_id = &req.session_id;
    let layer_name = &req.layer_name;

    // b.39x H: full session_id validation (length cap + control-char
    // rejection). Replaces the pre-fix `is_empty()`-only check. The b.94g
    // `ctx` records sit below their validators for that same reason.
    validate_session_id(session_id)?;
    ctx.session(session_id);

    // Validate layer name.
    validate_layer_name(layer_name)?;
    ctx.layer(layer_name);

    // Auth: write permission on the session's base path.
    ctx.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, session_id)
        .await?;
    ctx.base_path(&base_path);
    check_write_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // Delegate to storage.
    ctx.at(RpcStage::Storage);
    svc.layer_store
        .delete_layer(session_id, layer_name, svc.ext_value_store.clone())
        .await
        .map_err(|e| -> Status { e.into() })?;

    tracing::info!(
        session_id = %session_id,
        layer_name = %layer_name,
        "DeleteLayer completed"
    );

    Ok(Response::new(DeleteLayerResponse {}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state-changing delta: one inline `SetField`. Faithful to the shape
    /// the invariant is about — a commit that mutates object state and would
    /// therefore be replayed as a real change on restore.
    fn state_changing_delta() -> oodf_ffi::OwnedDelta {
        oodf_ffi::OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 7,
                key_name: "radius".to_string(),
                value_or_ext_hash: vec![1, 2, 3, 4],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            ..Default::default()
        }
    }

    /// Forged SR-1.2 violation: a state-changing delta paired with **no**
    /// relay OMPF. `handle_update_layer` cannot produce this pair today (both
    /// values derive from the same `ompf_bytes.is_empty()` test), so the only
    /// way to prove the tripwire is armed is to call the helper directly with
    /// the forbidden pair.
    ///
    /// **Release-build consequence.** `debug_assert!` compiles out under
    /// `cfg(not(debug_assertions))`, so in a release build the helper is a
    /// no-op and this test would not panic — hence the `#[cfg(debug_assertions)]`
    /// gate (a release-profile test run compiles the case out rather than
    /// reporting a spurious failure). The release-era protection is therefore
    /// NOT this test but the source-scan wiring guard in
    /// `tests/integration/grpc/ompf_stream_invariant_test.rs`, which fails if
    /// the call disappears from the derivation point, plus the two storage-site
    /// doc-contracts. Both tiers are needed: this one proves the predicate is
    /// armed and correctly oriented, the guard proves it is still wired in.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "relay-OMPF invariant")]
    fn debug_assert_relay_invariant_state_changing_delta_without_relay_ompf_panics() {
        debug_assert_relay_invariant(&Some(state_changing_delta()), &None);
    }

    /// The other half of the orientation check. The forged test above proves
    /// the assert fires on the forbidden pair; this one proves it does NOT
    /// fire on any of the three legal pairs, so a predicate inverted into
    /// something that panics on legitimate traffic (or that is tautologically
    /// true and never fires) cannot pass both tests at once.
    ///
    /// Ungated: a release build turns every arm into a no-op, which still
    /// satisfies "does not panic", so the case is meaningful (if weaker) in
    /// both profiles.
    #[test]
    fn debug_assert_relay_invariant_permits_every_legal_pair() {
        let relay = Some(Bytes::from_static(b"OMPF-ish bytes"));

        // Non-empty commit carrying its relay OMPF — the production hot path.
        debug_assert_relay_invariant(&Some(state_changing_delta()), &relay);
        // Empty commit, no relay bytes — the §8.4 empty-commit shape.
        debug_assert_relay_invariant(&None, &None);
        // Empty delta that still arrived inside a non-empty OMPF container
        // (`handle_update_layer` yields this when the container parses to a
        // delta with no commands). Legal: the invariant is one-directional.
        debug_assert_relay_invariant(&None, &relay);
    }

    // ---- `refused_before_commit`: the UPSTREAM_UNAVAILABLE choke point ----

    use crate::error::{
        refusal_metadata, refusal_reason, CAPACITY_RETRY_DELAY, REFUSAL_DETAIL_DOMAIN,
        UPSTREAM_UNAVAILABLE_RETRY_DELAY,
    };
    use crate::storage::client::StoreWriteRefusal;
    use crate::storage::layer::OPERATION_OVERTAKEN_MESSAGE;
    use tonic_types::StatusExt;

    /// Draws per converted status: the `RetryInfo` delay is sampled when the
    /// status is built, so one draw could land in range by chance.
    const RETRY_DELAY_DRAWS: usize = 32;

    /// A plain `UNAVAILABLE`'s message, as a pre-commit dependency failure
    /// (the backing store still loading) words it.
    const PLAIN_UNAVAILABLE_MSG: &str =
        "transient I/O: GET layer mapping failed (Unknown): LOADING Valkey is loading the dataset \
         in memory";

    /// A context of `rpc`, its commit marker set or not, chained or not.
    fn ctx(rpc: &'static str, commit_started: bool, chained: bool) -> RpcLogContext {
        let mut ctx = RpcLogContext::new(rpc);
        if chained {
            ctx.chained();
        }
        if commit_started {
            ctx.commit_started();
        }
        ctx
    }

    /// `(code, message, details bytes)`: what a client receives, compared
    /// whole so "unchanged" means byte-identical, the encoded refusal detail
    /// included.
    fn wire(status: &Status) -> (tonic::Code, String, Vec<u8>) {
        (
            status.code(),
            status.message().to_owned(),
            status.details().to_vec(),
        )
    }

    /// (a) A plain `UNAVAILABLE` raised before the commit marker — on either
    /// upload RPC, chained or not — is answered `UNAVAILABLE` with the same
    /// message, an `ErrorInfo` {`UPSTREAM_UNAVAILABLE`, the service's
    /// domain, no metadata} and a `RetryInfo` delay in `[1 s, 3 s)`.
    #[test]
    fn refused_before_commit_plain_unavailable_before_the_marker_carries_upstream_unavailable() {
        for (rpc, chained) in [
            ("CreateLayer", false),
            ("UpdateLayer", false),
            ("UpdateLayer", true),
        ] {
            for _ in 0..RETRY_DELAY_DRAWS {
                let status = refused_before_commit(
                    &ctx(rpc, false, chained),
                    Status::unavailable(PLAIN_UNAVAILABLE_MSG),
                );
                assert_eq!(status.code(), tonic::Code::Unavailable, "{rpc}");
                assert_eq!(
                    status.message(),
                    PLAIN_UNAVAILABLE_MSG,
                    "{rpc}: the message is byte-identical"
                );
                let info = status
                    .get_details_error_info()
                    .unwrap_or_else(|| panic!("{rpc} (chained {chained}): an ErrorInfo"));
                assert_eq!(info.reason, refusal_reason::UPSTREAM_UNAVAILABLE, "{rpc}");
                assert_eq!(info.domain, REFUSAL_DETAIL_DOMAIN, "{rpc}");
                assert!(info.metadata.is_empty(), "{rpc}: no metadata; got {info:?}");
                let delay = status
                    .get_details_retry_info()
                    .and_then(|r| r.retry_delay)
                    .unwrap_or_else(|| panic!("{rpc}: a RetryInfo delay"));
                assert!(
                    UPSTREAM_UNAVAILABLE_RETRY_DELAY.min <= delay
                        && delay < UPSTREAM_UNAVAILABLE_RETRY_DELAY.max,
                    "{rpc}: {delay:?} outside {UPSTREAM_UNAVAILABLE_RETRY_DELAY:?}"
                );
            }
        }
    }

    /// (b) From the commit marker on, every `UNAVAILABLE` passes unchanged —
    /// the overtaken answer, the outcome-unknown answer after a counted retry,
    /// and a plain transient — because the call's commit may have been sent.
    #[test]
    fn refused_before_commit_unavailable_after_the_marker_is_unchanged() {
        let overtaken = Status::from(ServiceError::Unavailable(
            OPERATION_OVERTAKEN_MESSAGE.to_owned(),
        ));
        let outcome_unknown: Status =
            outcome_unknown_after_retry(2, "transient I/O: EVALSHA failed (IO): broken pipe")
                .into();
        let plain = Status::from(ServiceError::Unavailable(PLAIN_UNAVAILABLE_MSG.to_owned()));
        for (rpc, chained) in [
            ("CreateLayer", false),
            ("UpdateLayer", false),
            ("UpdateLayer", true),
        ] {
            for (name, status) in [
                ("overtaken", &overtaken),
                ("outcome unknown", &outcome_unknown),
                ("plain", &plain),
            ] {
                let answered = refused_before_commit(&ctx(rpc, true, chained), status.clone());
                assert_eq!(
                    wire(&answered),
                    wire(status),
                    "{rpc} (chained {chained}), {name}: unchanged after the marker"
                );
                assert!(
                    answered.details().is_empty(),
                    "{rpc}, {name}: no refusal detail"
                );
            }
        }
    }

    /// (c) An `UNAVAILABLE` that already carries a refusal detail keeps it,
    /// before the marker as after: its reason is never replaced by
    /// `UPSTREAM_UNAVAILABLE`.
    #[test]
    fn refused_before_commit_already_detailed_unavailable_keeps_its_reason() {
        let detailed = [
            (
                refusal_reason::UPLOAD_CAPACITY,
                RefusalDetail::new(refusal_reason::UPLOAD_CAPACITY, Some(CAPACITY_RETRY_DELAY))
                    .with_metadata(refusal_metadata::CAUSE, refusal_metadata::WAITER_CAP)
                    .unavailable("this replica's upload admission queue is full"),
            ),
            (
                refusal_reason::SERVER_SHUTTING_DOWN,
                RefusalDetail::server_shutting_down().unavailable("server shutting down"),
            ),
            (
                refusal_reason::STORE_WRITE_REFUSED,
                Status::from(StoreWriteRefusal::OutOfMemory.into_service_error()),
            ),
        ];
        for (rpc, chained) in [
            ("CreateLayer", false),
            ("UpdateLayer", false),
            ("UpdateLayer", true),
        ] {
            for commit_started in [false, true] {
                for (reason, status) in &detailed {
                    let answered =
                        refused_before_commit(&ctx(rpc, commit_started, chained), status.clone());
                    assert_eq!(
                        wire(&answered),
                        wire(status),
                        "{rpc} (chained {chained}, marker {commit_started}): a {reason} \
                         refusal is unchanged"
                    );
                    assert_eq!(
                        answered.get_details_error_info().map(|i| i.reason),
                        Some((*reason).to_owned()),
                        "{rpc}: the reason is not replaced"
                    );
                }
            }
        }
    }

    /// (d) Every code other than `UNAVAILABLE` passes unchanged, before the
    /// marker as after: the hint is for a dependency outage, not for a
    /// refusal, a fault, a missing resource or a deadline.
    #[test]
    fn refused_before_commit_other_codes_are_unchanged() {
        let others = [
            Status::invalid_argument("delta is not valid OODF"),
            Status::internal("internal error"),
            Status::not_found("layer not found"),
            Status::deadline_exceeded("deadline exceeded"),
            Status::cancelled("stream receive error: transport closed"),
        ];
        for (rpc, chained) in [
            ("CreateLayer", false),
            ("UpdateLayer", false),
            ("UpdateLayer", true),
        ] {
            for commit_started in [false, true] {
                for status in &others {
                    let answered =
                        refused_before_commit(&ctx(rpc, commit_started, chained), status.clone());
                    assert_eq!(
                        wire(&answered),
                        wire(status),
                        "{rpc} (chained {chained}, marker {commit_started}): {:?} is unchanged",
                        status.code()
                    );
                    assert!(answered.details().is_empty(), "{:?}", status.code());
                }
            }
        }
    }
}
