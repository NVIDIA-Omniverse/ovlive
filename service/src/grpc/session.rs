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

// CreateSession, DeleteSession, ListSessions gRPC handlers.
//
// Each handler extracts the bearer token, performs auth/userinfo calls,
// delegates to SessionStore, and manages the per-session CancellationToken map.

use std::sync::Arc;

use dashmap::DashMap;
use fred::clients::SubscriberClient;
use fred::interfaces::{EventInterface, PubsubInterface, SetsInterface};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use crate::checkpoint::{SessionRecoveryHook, SessionRecoveryOutcome};
use crate::observability::metrics::{SharedMetrics, PUBSUB_RELAY_SESSION_DELETED};
use crate::presence::{DrainEnd, PubsubRelayReceiver};
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;
use crate::storage::session::{GhostMember, SessionMeta, SessionStore};

use super::auth_helpers::{
    check_read_permission_allowing_path_not_found, check_write_permission, extract_token,
    BasePathSource, ReadDecision,
};
use super::proto::{
    CreateSessionRequest, CreateSessionResponse, DeleteSessionRequest, DeleteSessionResponse,
    ListSessionsRequest, ListSessionsResponse, Session,
};
use super::rpc_log::{self, RpcLogContext, RpcStage};
use super::service::LiveEditService;
use crate::error::ServiceError;

/// Why a per-session token was cancelled. Recorded on the token's
/// [`SessionCancel`] before the cancel, and read by the streaming RPCs'
/// session-cancel arms to choose the stream's ending (SDD §12.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionCancelCause {
    /// An explicit `DeleteSession`: the local handler, its cross-replica
    /// `le:session:deleted` fan-out, or a canceller that found the session
    /// missing from the live store and carrying the deletion marker
    /// `DeleteSession` writes (the `session_tokens` sweeper, or
    /// [`acquire_session_cancel`] before the fan-out reaches this replica).
    /// `SubscribeReadLayer` ends `NOT_FOUND "session was deleted"`
    /// (`reason="session_cancel"`) and `JoinSession` ends `OK`. Neither is
    /// retried.
    Deleted,
    /// The session was found missing from the live store with no deletion
    /// marker (or with a marker read that failed): the `session_tokens`
    /// sweeper (which reads the store only, never the recovery hook), or
    /// [`acquire_session_cancel`] finding it gone even after its recovery
    /// pass. That is what a keyspace loss a checkpoint can restore looks like,
    /// so both streaming RPCs end with the retryable [`session_absent_status`]
    /// (`reason="session_absent"` on `SubscribeReadLayer`). The retry's setup
    /// reaches the session recovery
    /// hook on whichever replica serves it: through a base-path cache miss, or,
    /// when that replica still holds a fresh cache entry, through
    /// [`acquire_session_cancel`]'s recovery pass. So with checkpointing enabled
    /// a session that has a checkpoint is restored, and one that is really
    /// gone is refused `NOT_FOUND` at that retry's setup.
    AbsentFromStore,
}

/// A session's cancellation token together with the cause of its cancel.
///
/// The value type of [`LiveEditService::session_tokens`]. Every stream on the
/// session holds a clone, or a [`Self::child`], and all of them share one cause
/// slot. The canceller records the cause, then cancels. Production cancels go
/// through [`Self::cancel_deleted`] (an explicit `DeleteSession`) or, where the
/// cause is picked at run time from the deletion marker, the module's private
/// `cancel_with`. A stream that observes the cancel reads [`Self::cause`] to
/// choose its ending. The first cause recorded wins; a later cancel with
/// another cause changes nothing, because the token is already cancelled.
///
/// Derefs to the [`CancellationToken`], so `cancelled()` / `is_cancelled()`
/// read as before. A bare `cancel()` through that deref records no cause, and
/// [`Self::cause`] then reads [`SessionCancelCause::Deleted`], the meaning a
/// session-token cancel had before causes existed. Production cancels always
/// record a cause.
#[derive(Clone, Debug, Default)]
pub struct SessionCancel {
    token: CancellationToken,
    cause: Arc<std::sync::OnceLock<SessionCancelCause>>,
}

impl SessionCancel {
    /// A fresh, uncancelled token with no cause recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel for an explicit `DeleteSession` ([`SessionCancelCause::Deleted`]).
    pub fn cancel_deleted(&self) {
        self.cancel_with(SessionCancelCause::Deleted);
    }

    /// Cancel as for a session found missing from the live store with no
    /// deletion marker ([`SessionCancelCause::AbsentFromStore`]).
    ///
    /// A test seam, compiled only for tests and the `test-support` feature: it
    /// lets a test produce that cause without flushing Valkey. Production code
    /// never cancels with a fixed `AbsentFromStore`; it picks the cause from
    /// the deletion marker and cancels through the private `cancel_with`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn cancel_absent_from_store(&self) {
        self.cancel_with(SessionCancelCause::AbsentFromStore);
    }

    /// Record `cause` if none is recorded yet, then cancel. The record is
    /// written before the cancel, so a stream that observes the cancel reads
    /// the cause (the token's cancel synchronizes with its waiters).
    fn cancel_with(&self, cause: SessionCancelCause) {
        let _ = self.cause.set(cause);
        self.token.cancel();
    }

    /// Why the token was cancelled. Meaningful once it is; reads
    /// [`SessionCancelCause::Deleted`] when no cause was recorded (see the
    /// type docs).
    pub fn cause(&self) -> SessionCancelCause {
        self.cause
            .get()
            .copied()
            .unwrap_or(SessionCancelCause::Deleted)
    }

    /// The token itself, for APIs that race a bare [`CancellationToken`].
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// A child of the token that shares this cause slot. Cancelling the parent
    /// cancels the child, and the child's [`Self::cause`] reads the parent's
    /// cause.
    pub fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
            cause: Arc::clone(&self.cause),
        }
    }
}

impl std::ops::Deref for SessionCancel {
    type Target = CancellationToken;

    fn deref(&self) -> &CancellationToken {
        &self.token
    }
}

/// Wrap an existing token with an empty cause slot.
impl From<CancellationToken> for SessionCancel {
    fn from(token: CancellationToken) -> Self {
        Self {
            token,
            cause: Arc::default(),
        }
    }
}

/// The status both streaming RPCs end — or are refused — with when their
/// session token was cancelled because the session is missing from the live
/// store with no deletion marker ([`SessionCancelCause::AbsentFromStore`]):
/// `UNAVAILABLE "session is missing from the live store; retry so it can be
/// recovered"`. A missing session that carries the marker was deleted, and its
/// streams get the deletion endings instead.
///
/// `UNAVAILABLE` because it is the retryable code, and a retry is what recovers
/// a session a checkpoint can restore: its setup runs the session recovery hook
/// on whichever replica serves it, warm base-path cache or not (see
/// [`SessionCancelCause::AbsentFromStore`]). A session that is really gone
/// (no checkpoint, or a marker read that failed) costs its client one retry,
/// which is refused `NOT_FOUND` at setup. Logged at WARN, the default for
/// `UNAVAILABLE`: a session vanishing from under a live stream without a
/// deletion is keyspace loss, which is worth an operator's attention. One
/// constructor, so the two RPCs cannot drift apart.
pub fn session_absent_status() -> Status {
    Status::unavailable("session is missing from the live store; retry so it can be recovered")
}

/// Handle CreateSession RPC.
///
/// 1. Extract bearer token from request extensions.
/// 2. Check write permission on the canonical base_layer_path.
/// 3. Resolve user identity via the userinfo service.
/// 4. Create session via SessionStore.
/// 5. Insert a CancellationToken for the new session.
///
/// Deadline-drop contract (b.yis sub-item 4, see `metrics_layer.rs`):
/// non-idempotent. `SessionStore::create` commits path_index inside MULTI,
/// then HSETs session_meta outside (different Cluster slot). A future drop
/// between EXEC and HSET leaves an orphan path_index entry that
/// `SessionStore::list` opportunistically cleans up on next read.
///
/// **Failure logging.** Wrapper over [`create_session_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`]. `session_id` is minted by storage, so a
/// failure event carries `base_path` (canonical) but no session id.
pub async fn handle_create_session(
    svc: &LiveEditService,
    request: Request<CreateSessionRequest>,
) -> Result<Response<CreateSessionResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("CreateSession", &request);
    let result = create_session_inner(svc, request, &mut ctx).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn create_session_inner(
    svc: &LiveEditService,
    request: Request<CreateSessionRequest>,
    ctx: &mut RpcLogContext,
) -> Result<Response<CreateSessionResponse>, Status> {
    let token = extract_token(&request)?;
    let req = request.into_inner();

    let session_name = &req.name;

    // Validate base_path (length, control chars, null bytes) and canonicalize
    // its percent-encoding before it hits auth, storage key formatting, or logs
    // (b.91j H-grpc-5; b.uoc). Empty-string rejection is covered inside
    // `validate_base_path`, which `canonicalize_base_path` runs on both the raw
    // and the canonical form.
    //
    // Everything downstream of this line uses the canonical spelling — two
    // RFC 3986-equivalent spellings of the same object must land on the same
    // Valkey keys and the same permission decision.
    let base_path = crate::domain::base_path::canonicalize_base_path(&req.path)?;
    // Record the CANONICAL spelling (b.94g), never the caller's: it is the
    // spelling that was authorized and that storage is keyed under.
    ctx.base_path(&base_path);
    // Surface the rewrite: the residual divergences canonicalization
    // deliberately leaves (non-special-scheme host case, userinfo/query
    // percent-encoding, default-port elision, IDNA) still present as a silent
    // empty `ListSessions`, so an operator needs to see which spelling arrived
    // versus which the service used. Base paths are already logged at `info`
    // on session creation, so no redaction concern.
    //
    // `info!`, not `debug!`: `CreateSession` is low-rate, and it is the rewrite
    // that durably fixes the spelling the session is keyed under, so the
    // default log level should record that a rewrite happened.
    if base_path != req.path {
        tracing::info!(
            received = %req.path,
            canonical = %base_path,
            "CreateSession canonicalized base_path"
        );
    }

    // Check write permission on the canonical base path. The path is the
    // client's own field, so a storage-service malformed-request refusal
    // answers `INVALID_ARGUMENT`.
    ctx.at(RpcStage::Auth);
    check_write_permission(svc, &token, &base_path, BasePathSource::Request).await?;

    // Resolve user identity. Transient backend outages flow through the
    // `From<AuthError> for ServiceError` mapping so they surface as
    // `UNAVAILABLE` (b.ohw H-grpc-1) rather than `INTERNAL`, matching the
    // permissions-call path.
    let user_info = svc
        .userinfo_client
        .get_current_user(&token)
        .await
        .map_err(ServiceError::from)?;

    // Create session in storage.
    ctx.at(RpcStage::Storage);
    let session_id = svc
        .session_store
        .create(session_name, &base_path, &user_info.display_name)
        .await?;

    // Insert a cancellation token for the new session.
    svc.session_tokens
        .insert(session_id.clone(), SessionCancel::new());

    Ok(Response::new(CreateSessionResponse { session_id }))
}

/// Handle DeleteSession RPC (SDD §7.8 — session deletion cascade).
///
/// 1. Extract bearer token, validate, auth check.
/// 2. Enumerate all layers and delete each (PUBLISH "deleted", GC ext values, cleanup keys).
/// 3. Delete session metadata from storage.
/// 4. Cancel the per-session CancellationToken (terminates remaining active streams).
///
/// Deadline-drop contract (b.yis sub-item 4, see `metrics_layer.rs`):
/// idempotent — a retry with the same `session_id` returns `NotFound` once
/// the path-index entry is gone. `SessionStore::delete` orders SREM
/// path_index → MULTI{DEL session_meta, DEL layers_index, SET the deletion
/// marker}, so a mid-flight drop leaves at worst orphan session_meta on a
/// session_id no longer in any path-index (invisible to `list`).
///
/// **NotFound recovery (SDD SR-2.7).** Step 1's `get_base_path` funnels
/// through [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache miss when the session is absent triggers lazy on-demand
/// session-state recovery via the attached
/// [`SessionRecoveryHook`](crate::checkpoint::SessionRecoveryHook). A session
/// that has already been deleted (cache invalidated by the delete cascade)
/// still surfaces `NotFound` because recovery returns `NoCheckpoint` in that
/// case. Its deletion marker (`KeySchema::session_deleted_marker`) answers so
/// while the checkpoint still awaits its prune by `delete_session_checkpoint`,
/// and the missing checkpoint answers so after that.
///
/// **Failure logging.** Wrapper over [`delete_session_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`]. A layer-deletion failure in the cascade below
/// logs its own `warn!` with the per-layer detail this boundary cannot carry;
/// the boundary line is the completion record and the two are complementary.
///
/// **Cascade failure code.** The failing layer's class decides the code: a
/// transient failure (`Aborted` / `Unavailable`) answers `UNAVAILABLE` carrying
/// the "after {idx} of {total} layers deleted … retry to continue" progress
/// message, and a retry resumes the cascade through the NotFound-skip; any
/// other class passes through unchanged, so a permanent `Internal` /
/// `PermanentKeyspaceRefusal` failure answers `INTERNAL` (redacted on the wire)
/// and its completion line is the one `internal error` at ERROR, carrying the
/// detail and this request's fields.
pub async fn handle_delete_session(
    svc: &LiveEditService,
    request: Request<DeleteSessionRequest>,
) -> Result<Response<DeleteSessionResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("DeleteSession", &request);
    let result = delete_session_inner(svc, request, &mut ctx).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn delete_session_inner(
    svc: &LiveEditService,
    request: Request<DeleteSessionRequest>,
    ctx: &mut RpcLogContext,
) -> Result<Response<DeleteSessionResponse>, Status> {
    let token = extract_token(&request)?;
    let req = request.into_inner();
    let session_id = &req.session_id;

    // b.39x H: full session_id validation (length cap + control-char
    // rejection) before any storage lookup or auth call. Replaces the
    // pre-fix `is_empty()`-only check that let a 64KB session_id with
    // ANSI escapes flow into Valkey keys and tracing output. The b.94g `ctx`
    // record sits below the validator for that same reason.
    crate::domain::validation::validate_session_id(session_id)?;
    ctx.session(session_id);

    // Look up the base_path for authorization (short-TTL cached).
    ctx.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, session_id)
        .await?;
    ctx.base_path(&base_path);

    // Check write permission on the session's stored base path.
    check_write_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // Cascade: delete all layers in the session (SDD §7.8 step 2).
    //
    // Bail on the first non-NotFound error rather than aggregating across
    // the whole layer list (b.91j H-grpc-7). Collecting every failure
    // before returning turns one Valkey outage into N × (timeout × retry)
    // work — a session with 100 layers under a flapping backend racks up
    // 100 full retry budgets and reliably blows DEADLINE_EXCEEDED. Bailing
    // early preserves the layers index so idempotent retries (NotFound-
    // skip) still work per SDD §7.8.
    ctx.at(RpcStage::Storage);
    let layer_names = svc.layer_store.list_layer_names(session_id).await?;
    let total = layer_names.len();
    for (idx, layer_name) in layer_names.iter().enumerate() {
        // `delete_layer_during_session_delete` (not `delete_layer`): suppress the
        // per-layer session-checkpoint rewrite. The whole session is being torn
        // down, so `session_store.delete` below enqueues the authoritative
        // session-checkpoint DELETE; a per-layer rewrite here would race that
        // delete on the shared writer TaskTracker and could resurrect the
        // tombstoned blob (b: recovery_fires_on_delete_session_then_deletes).
        match svc
            .layer_store
            .delete_layer_during_session_delete(session_id, layer_name, svc.ext_value_store.clone())
            .await
        {
            Ok(()) => {}
            Err(ServiceError::NotFound(_)) => {
                // Already deleted (partial retry) — continue. NotFound is
                // the idempotency marker, not a real failure.
            }
            Err(e) => {
                let remaining = total.saturating_sub(idx + 1);
                // Logged for every failure class: this WARN is the only place
                // the per-layer progress counts survive when the wire message
                // is redacted (an `Internal`-class failure below).
                tracing::warn!(
                    session_id = %session_id,
                    layer_name = %layer_name,
                    layers_deleted_before_failure = idx,
                    layers_remaining = remaining,
                    layers_total = total,
                    error = %e,
                    "DeleteSession cascade aborted on layer deletion failure"
                );
                // The failing layer's class decides the code. A transient
                // failure (`Aborted` — an in-process retry signal, or a
                // retry loop's exhaustion — `Unavailable`, or
                // `UnavailableWithDetail`) answers `UNAVAILABLE` with the
                // progress counts so the client's retry resumes the cascade
                // via the NotFound-skip above; a retry hint the layer's
                // refusal carried is kept. Any other class — notably
                // `Internal` and `PermanentKeyspaceRefusal` — passes through
                // unchanged, so a permanent layer failure is not dressed up as
                // retriable.
                let progress = format!(
                    "DeleteSession cascade failed on layer '{layer_name}' after {idx} of {total} \
                     layers deleted ({remaining} remaining), retry to continue: {e}"
                );
                return Err(match e {
                    ServiceError::Aborted(_) | ServiceError::Unavailable(_) => {
                        ServiceError::Unavailable(progress)
                    }
                    ServiceError::UnavailableWithDetail { detail, .. } => {
                        ServiceError::UnavailableWithDetail {
                            message: progress,
                            detail,
                        }
                    }
                    other => other,
                }
                .into());
            }
        }
    }

    // Delete session metadata + path index + layers index.
    svc.session_store.delete(session_id).await?;

    // Drop any cached base_path so subsequent reads on this replica hit
    // storage (and see `NotFound`) rather than the stale entry.
    svc.session_base_path_cache.invalidate(session_id);

    // Cancel and remove the CancellationToken AFTER layer deletions
    // so subscribers receive "deleted" notifications before stream termination.
    // An explicit deletion: the streams end as deleted, not retryably.
    if let Some((_, token)) = svc.session_tokens.remove(session_id) {
        token.cancel_deleted();
    }

    // Cross-replica fanout: publish the session_id on `le:session:deleted`
    // so every replica removes + cancels its local `session_tokens` entry
    // (SDD §7.8). This replica has already handled its own entry above; the
    // subscriber will simply no-op when the key is already absent.
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: a second notice is a no-op at each receiver.
    let channel = svc.key_schema.session_deleted_channel();
    if let Err(e) = svc
        .valkey
        .pool
        .next()
        .publish::<(), _, _>(&channel, session_id.as_str())
        .await
    {
        FredErrOrigin::client(&svc.valkey, POOL_LABEL_DATA).observe_error(&e);
        tracing::warn!(
            session_id = %session_id,
            error = %e,
            "failed to publish session-deleted event; peer replicas will rely on subsequent signals"
        );
    }

    tracing::info!(
        session_id = %session_id,
        layers_deleted = layer_names.len(),
        "DeleteSession cascade completed"
    );

    Ok(Response::new(DeleteSessionResponse {}))
}

/// Handle ListSessions RPC.
///
/// 1. Validate non-empty base_layer_path (INVALID_ARGUMENT).
/// 2. Check read permission on the base path.
/// 3. List sessions via [`SessionStore::list`] — returns both present sessions
///    and **ghost members** whose `session_meta` HGETALL came back empty.
/// 4. For each ghost, orchestrate the recovery-aware cleanup decision via
///    [`reconcile_ghost_members`]: invoke
///    [`SessionRecoveryHook::try_recover_session`] to discriminate between a
///    recovery candidate and a truly-orphaned member. On `Recovered` /
///    `AlreadyPresent`, re-issue [`SessionStore::get_session_meta`] and
///    append the recovered session to the response. On `NoCheckpoint` (the
///    truly-orphaned-or-disabled-hook arm), SREM the member from both
///    `path_index` and `path_names_index` — preserving today's
///    opportunistic-cleanup behaviour for the non-recovery case. On
///    `Error(_)`, log warn and skip the member from the response without
///    SREM, conservatively deferring to the orphan sweep / next walk.
///
/// **Recovery-aware contract (Epic `t1.fgj.7v`, SDD SR-2.7).** This is the
/// one RPC that bypasses [`crate::grpc::session_cache::SessionBasePathCache::get_base_path`]
/// (the choke point wired by the sibling Subtask for the six per-session_id
/// handlers). Recovery is wired here directly. The `Arc<SessionRecoveryHook>`
/// on [`LiveEditService`] short-circuits to `NoCheckpoint` when
/// `checkpoint_enabled=false`, which preserves "SREM every ghost
/// unconditionally" — the legacy behaviour customers depending on
/// `checkpoint_enabled=false` still see.
///
/// **Failure logging.** Wrapper over [`list_sessions_inner`]; every non-OK
/// return emits exactly one completion line through
/// [`rpc_log::log_rpc_failure`]. The `PathNotFound` arm is an
/// `Ok(empty list)` by design (SDD §7.13), so it stays silent — it is not a
/// failure.
pub async fn handle_list_sessions(
    svc: &LiveEditService,
    request: Request<ListSessionsRequest>,
) -> Result<Response<ListSessionsResponse>, Status> {
    let mut ctx = RpcLogContext::for_request("ListSessions", &request);
    let result = list_sessions_inner(svc, request, &mut ctx).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn list_sessions_inner(
    svc: &LiveEditService,
    request: Request<ListSessionsRequest>,
    ctx: &mut RpcLogContext,
) -> Result<Response<ListSessionsResponse>, Status> {
    let token = extract_token(&request)?;
    let req = request.into_inner();
    // Validate base_path (length, control chars, null bytes) and canonicalize
    // its percent-encoding before it hits auth, storage key formatting, or logs
    // (b.91j H-grpc-5; b.uoc). Empty-string rejection is covered inside
    // `validate_base_path`, which `canonicalize_base_path` runs on both the raw
    // and the canonical form.
    //
    // Listing must use the same canonical spelling `CreateSession` stored under,
    // or a caller whose encoder differs from Kit's reads an empty path index and
    // gets an empty list back — indistinguishable from "no sessions" (b.uoc).
    let base_path = crate::domain::base_path::canonicalize_base_path(&req.path)?;
    // Record the CANONICAL spelling (b.94g) — the one that was authorized and
    // that the path index was read under.
    ctx.base_path(&base_path);
    // See `handle_create_session` — a caller whose spelling gets rewritten here
    // has no other server-side signal that the path it sent is not the path
    // that was listed. `debug!`, not `info!` (unlike `CreateSession`):
    // `ListSessions` is a frequent, polled RPC, so one non-canonical client
    // would emit one event per poll. Set `LIVE_EDIT_LOG_LEVEL=debug` to
    // diagnose an unexpectedly-empty list.
    if base_path != req.path {
        tracing::debug!(
            received = %req.path,
            canonical = %base_path,
            "ListSessions canonicalized base_path"
        );
    }
    let base_path = base_path.as_str();

    // Check read permission on the base path. ListSessions (SDD §7.13)
    // returns an empty list when the base path is unknown, so it shares the
    // `check_read_permission_allowing_path_not_found` helper with any other
    // handler that wants the same semantics. The path is the client's own
    // field, so a storage-service malformed-request refusal answers
    // `INVALID_ARGUMENT` (never an empty list).
    ctx.at(RpcStage::Auth);
    match check_read_permission_allowing_path_not_found(
        svc,
        &token,
        base_path,
        BasePathSource::Request,
    )
    .await?
    {
        ReadDecision::Allowed => {}
        ReadDecision::PathNotFound => {
            // SDD §7.13: if PathNotFound, return empty list.
            return Ok(Response::new(ListSessionsResponse {
                sessions: Vec::new(),
            }));
        }
    }

    // List sessions from storage. `list` returns both `present` (live
    // sessions) and `ghosts` (path-index members whose `HGETALL` came back
    // empty). `list` itself no longer SREMs — that decision is recovery-aware
    // and lives in `reconcile_ghost_members` below.
    ctx.at(RpcStage::Storage);
    let result = svc.session_store.list(base_path).await?;
    let mut sessions = result.present;

    // Reconcile ghosts: for each empty-HGETALL member, decide between
    // recovery (preserve + append) and SREM-as-orphan (today's behaviour
    // for the no-checkpoint / disabled-hook case). See helper docstring
    // for the full state machine.
    let recovered = reconcile_ghost_members(
        &svc.session_recovery,
        &svc.session_store,
        &svc.valkey,
        &svc.key_schema,
        base_path,
        result.ghosts,
    )
    .await;
    sessions.extend(recovered);

    let proto_sessions = sessions
        .into_iter()
        .map(|s| Session {
            session_id: s.session_id,
            name: s.name,
            created_by_user_name: s.created_by,
            created_at: s.created_at,
        })
        .collect();

    Ok(Response::new(ListSessionsResponse {
        sessions: proto_sessions,
    }))
}

/// Recovery-aware reconciliation of `ListSessions` ghost members
/// (Epic `t1.fgj.7v`, SDD SR-2.7).
///
/// A "ghost" is a `path_names_index` member whose `HGETALL le:{sid}:sess`
/// returned empty during [`SessionStore::list`]. Before session-recovery this
/// was unambiguously a true orphan and `list` SREM'd it inline. Now, the
/// member may be either:
///
/// 1. A **recovery candidate** — a session-checkpoint exists for this
///    `session_id` and would restore the missing `session_meta`. Recovery
///    preserves the path-index member; the caller re-reads
///    [`SessionStore::get_session_meta`] and appends the restored session
///    to the response.
/// 2. A **true orphan** — no checkpoint exists (or checkpointing is
///    disabled). The member is SREM'd from both `path_index` and
///    `path_names_index`, matching today's opportunistic-cleanup behaviour.
/// 3. A **transient failure** — recovery returned `Error(_)`. The member is
///    NOT SREM'd; we conservatively defer to the orphan sweep / next walk.
///
/// The disabled-hook case (legacy `checkpoint_enabled=false` deployments and
/// any future tests that construct a hook in disabled mode) flows through
/// arm 2 — [`SessionRecoveryHook::try_recover_session`] returns
/// `NoCheckpoint` without admission churn or store calls, and this helper
/// SREMs the member exactly as the pre-recovery code did. This preserves
/// the `path-index opportunistic SREM` invariant the parent Task body pins.
///
/// Returns the list of [`SessionMeta`] for ghosts that recovered
/// successfully (i.e. `Recovered` / `AlreadyPresent` followed by a non-empty
/// `get_session_meta`). Ghosts that fell through to SREM or failed transiently
/// are excluded from the response.
///
/// **Cluster safety**: the SREMs share the `{hex(base_path)}` hash tag with
/// the `SMEMBERS` that produced the members (both `path_index` and
/// `path_names_index` are derived from the same `KeySchema` helpers and
/// carry the same hash tag), so no new cross-slot operation is introduced.
async fn reconcile_ghost_members(
    recovery: &Arc<SessionRecoveryHook>,
    session_store: &SessionStore,
    valkey: &ValkeyClient,
    keys: &KeySchema,
    base_path: &str,
    ghosts: Vec<GhostMember>,
) -> Vec<SessionMeta> {
    if ghosts.is_empty() {
        return Vec::new();
    }

    let path_names_key = keys.path_names_index(base_path);
    let path_index_key = keys.path_index(base_path);

    let mut recovered = Vec::with_capacity(ghosts.len());

    for ghost in ghosts {
        let outcome = recovery.try_recover_session(&ghost.sid).await;
        match outcome {
            SessionRecoveryOutcome::Recovered | SessionRecoveryOutcome::AlreadyPresent => {
                // Re-issue the targeted HGETALL once. `AlreadyPresent` is
                // the same retry shape: a peer replica (or a previous
                // admission-mutex holder on this replica) restored the
                // session; we re-read and append. On a `Recovered` →
                // `Ok(None)` race (DeleteSession committed between
                // recovery and our re-read), fall through to the
                // SREM-as-orphan arm — the path-index is the eventually-
                // consistent source of truth and `DeleteSession` already
                // SREM'd the legacy path_index entry on its own commit
                // path; we just clean up the names-set composite.
                match session_store.get_session_meta(&ghost.sid).await {
                    Ok(Some(meta)) => {
                        recovered.push(meta);
                    }
                    Ok(None) => {
                        tracing::debug!(
                            session_id = %ghost.sid,
                            base_path = %base_path,
                            "ghost recovered but session_meta missing on re-read (race with DeleteSession); SREMing as orphan",
                        );
                        srem_ghost_member(valkey, &path_names_key, &path_index_key, &ghost).await;
                    }
                    Err(e) => {
                        tracing::warn!(
                            session_id = %ghost.sid,
                            base_path = %base_path,
                            error = %e,
                            "ghost recovered but get_session_meta re-read failed; skipping member from response (no SREM)",
                        );
                    }
                }
            }
            SessionRecoveryOutcome::NoCheckpoint => {
                // Truly-orphaned member (or checkpointing disabled). SREM
                // from both indices — preserves the pre-recovery
                // opportunistic-cleanup behaviour exactly.
                srem_ghost_member(valkey, &path_names_key, &path_index_key, &ghost).await;
            }
            SessionRecoveryOutcome::Error(err) => {
                // Conservative: do NOT SREM on a transient or permanent
                // recovery-side error. The member is excluded from the
                // response (we have no `session_meta` to populate the
                // proto), but the path-index keeps the member so the next
                // walk retries. Permanent errors are rare and best handled
                // by the orphan sweep with full operator visibility.
                tracing::warn!(
                    session_id = %ghost.sid,
                    base_path = %base_path,
                    error = %err,
                    "session recovery failed for ghost member; skipping (no SREM, next walk will retry)",
                );
            }
        }
    }

    recovered
}

/// Issue the two SREMs that retire a truly-orphaned ghost member from both
/// the legacy `path_index` and the authoritative `path_names_index`.
///
/// Errors on either SREM are logged but not surfaced — the indices may
/// already have been cleaned by a peer sweeper or a concurrent
/// `DeleteSession`. The post-fix b.91j C-7 contract treats either entry
/// being absent as a benign race. A command timeout is still counted on
/// `valkey_errors_total{pool="data", kind="command_timeout"}`.
async fn srem_ghost_member(
    valkey: &ValkeyClient,
    path_names_key: &str,
    path_index_key: &str,
    ghost: &GhostMember,
) {
    let pool = &valkey.pool;
    let origin = FredErrOrigin::client(valkey, POOL_LABEL_DATA);
    if let Err(e) = pool
        .srem::<(), _, _>(path_names_key, ghost.composite_member.clone())
        .await
    {
        origin.observe_error(&e);
        tracing::warn!(
            session_id = %ghost.sid,
            member = %ghost.composite_member,
            error = %e,
            "failed to SREM orphaned path_names_index entry",
        );
    }
    if let Err(e) = pool.srem::<(), _, _>(path_index_key, &ghost.sid).await {
        origin.observe_error(&e);
        tracing::warn!(
            session_id = %ghost.sid,
            error = %e,
            "failed to SREM orphaned path_index entry",
        );
    }
}

/// Process one `session_deleted` pub/sub payload: remove the local entry
/// (if any), cancel its token as an explicit deletion
/// ([`SessionCancelCause::Deleted`]: the payload is published only by
/// `DeleteSession`), and drop any cached `base_path`. Absent entries are a
/// no-op, which is the expected case on replicas that never handled
/// `CreateSession` for this session.
pub(crate) fn apply_session_deleted(
    session_tokens: &DashMap<String, SessionCancel>,
    base_path_cache: &super::session_cache::SessionBasePathCache,
    session_id: &str,
) {
    if session_id.is_empty() {
        return;
    }
    if let Some((_, token)) = session_tokens.remove(session_id) {
        token.cancel_deleted();
    }
    base_path_cache.invalidate(session_id);
}

/// The cause to cancel a session's token with once its caller has found the
/// session missing from the live store: [`SessionCancelCause::Deleted`] when
/// the session carries the deletion marker `SessionStore::delete` writes
/// ([`SessionStore::deleted_marker_exists`]), else
/// [`SessionCancelCause::AbsentFromStore`].
///
/// The marker is what tells the two apart. A session a `DeleteSession`
/// removed is marked in the transaction that removed it, so its streams end
/// with the deletion endings (`SubscribeReadLayer` `NOT_FOUND "session was
/// deleted"`, `JoinSession` `OK`) even when this replica cancels before the
/// `le:session:deleted` fan-out reaches it. An unmarked session was lost
/// without a deletion (a keyspace loss), and its streams end retryably so a
/// retry can recover it.
///
/// A failed read falls back to `AbsentFromStore`, the retryable ending: a
/// retry costs a deleted session's client one refused setup, while a
/// deletion ending on a restorable session would lose it for good. The read
/// only picks the cause; the caller cancels either way.
async fn absent_session_cancel_cause(
    session_store: &SessionStore,
    session_id: &str,
) -> SessionCancelCause {
    match session_store.deleted_marker_exists(session_id).await {
        Ok(true) => SessionCancelCause::Deleted,
        Ok(false) => SessionCancelCause::AbsentFromStore,
        Err(e) => {
            tracing::debug!(
                session_id = %session_id,
                error = %e,
                "session deletion-marker read failed; cancelling the session's streams as absent from the store (retryable)"
            );
            SessionCancelCause::AbsentFromStore
        }
    }
}

/// Get-or-insert a per-session [`SessionCancel`] token for streaming handlers
/// (`SubscribeReadLayer`, `JoinSession`) on replicas that did not handle the
/// original `CreateSession` (b.uue).
///
/// Why: `CreateSession` only inserts an entry on the replica that served it.
/// On any other replica, a streaming RPC for the same session would
/// previously see `session_tokens.get(session_id) == None` and run with no
/// session-scoped cancellation — so a `DeleteSession` (which fans out via
/// `le:session:deleted`) would fail to terminate the stream, breaking the
/// "any replica serves any RPC" invariant.
///
/// Race the helper closes: between us reading the session and inserting the
/// token, a `DeleteSession` may have already published `le:session:deleted`.
/// If we just inserted, the pub/sub consumer would no-op (saw nothing to
/// remove) and our fresh token would never be cancelled. To prevent that, we
/// re-check session existence in storage after the insert. `DeleteSession`
/// always runs `SessionStore::delete` (which removes session metadata)
/// before publishing the channel message, so if the publish has already been
/// observed anywhere, `get_base_path` returns `NotFound` here.
///
/// **The token is cancelled only when the session is gone.** Cancelling a
/// session token ends every stream sharing it, so this function removes and
/// cancels the entry on a confirmed `NotFound` and on nothing else:
///
/// * `NotFound` from the re-check — the session is missing from the live
///   store. No `DeleteSession` was seen, and a keyspace loss that a checkpoint
///   can restore looks the same from here, so before cancelling anything the
///   helper drops the session's base-path cache entry and resolves the session
///   once more through `base_path_cache`, which runs the session recovery hook
///   (at most once). The caller got here through that cache, usually from a
///   fresh entry that skipped the hook; without this pass a retry landing on a
///   replica with a warm entry would be refused instead of recovered. The pass
///   never restores a session that a `DeleteSession` removed, even before that
///   session's checkpoint tombstone has landed: the delete marks the session in
///   the transaction that removes it, and the recovery hook answers "no
///   checkpoint" for a marked session (`SessionRecoveryHook::try_recover_session`
///   steps 4a and 6a, which also name the one narrow race the marker leaves).
///   The pass is bounded by `recovery_pass_budget`
///   ([`crate::config::session_recovery_pass_budget`]), because `JoinSession`
///   runs it with its connection registered and its lease not yet renewed.
///   - Resolved — the session was restored (or reappeared meanwhile). Nothing
///     is cancelled, and the token inserted above is returned, as on the
///     found path.
///   - Still `NotFound` — a deleted session, or one with no checkpoint to
///     restore. The entry is removed and its token cancelled, whichever caller
///     inserted it, and the caller itself gets `NotFound`. The cause comes
///     from the session's deletion marker (`absent_session_cancel_cause`):
///     marked, [`SessionCancelCause::Deleted`], so every stream sharing the
///     token gets the deletion endings even though the `le:session:deleted`
///     fan-out has not reached this replica yet (the first recorded cause
///     wins, so the late fan-out could not correct it); unmarked, or the
///     marker unreadable, [`SessionCancelCause::AbsentFromStore`] and the
///     retryable [`session_absent_status`]. A future dropped at the marker
///     read cancels nothing, like one dropped at the other `.await`s; the
///     fan-out or the sweeper then cancels the entry.
///   - The budget elapsed — `Unavailable`, and the entry is left alone, as on
///     the next bullet. The pass already dropped this replica's cache entry, so
///     the client's retry resolves the session at the handler's own lookup.
///   - Any other error — as the next bullet.
/// * A transient lookup error (`Internal`, `Aborted`, `Unavailable`, …), from
///   the re-check or from the recovery pass (a failed recovery maps to
///   `Unavailable` or `Internal`), is surfaced for the client to retry, and the
///   entry is left alone. So is a future dropped at either `.await` (tonic
///   deadline, client disconnect, or `JoinSession`'s shutdown race at step 6).
///   Another caller on the same session may already share the token, and a
///   lookup that merely failed is no evidence the session is gone.
///
/// Leaving an entry behind on those paths cannot grow `session_tokens` without
/// bound (the b.91j H-grpc-3 concern): both callers reach here only after
/// resolving the session through the base-path cache and passing the read
/// permission check, so a caller walking random session ids is refused before
/// it can insert. An entry left for a session that is in fact gone is reaped by
/// the `session_tokens` sweeper, which removes and cancels only on `NotFound`
/// too.
///
/// On `Ok` the entry stays in the map for the lifetime of the session. A
/// future `DeleteSession` for this session_id runs through the existing
/// pub/sub consumer (`apply_session_deleted`), which removes the entry by key.
pub async fn acquire_session_cancel(
    session_tokens: &DashMap<String, SessionCancel>,
    session_store: &SessionStore,
    base_path_cache: &super::session_cache::SessionBasePathCache,
    recovery_pass_budget: std::time::Duration,
    session_id: &str,
) -> Result<SessionCancel, ServiceError> {
    let token = session_tokens
        .entry(session_id.to_owned())
        .or_insert_with(SessionCancel::new)
        .clone();

    // The re-check reads the store itself, not the base-path cache: it must see
    // a `DeleteSession` that ran after the caller's own (possibly cached)
    // lookup.
    match session_store.get_base_path(session_id).await {
        Ok(_) => return Ok(token),
        Err(ServiceError::NotFound(_)) => {}
        // Transient storage errors — surface them so the caller can retry.
        // The entry is left alone: see the rustdoc.
        Err(other) => return Err(other),
    }

    // Missing from the store. The caller resolved the session a moment ago,
    // most often from a still-fresh base-path cache entry, so this is either a
    // deletion or a keyspace loss that a checkpoint can restore. Drop the stale
    // cache entry and resolve once more through the cache, whose recovery hook
    // restores a restorable session and refuses a deleted one (its deletion
    // marker). That resolve runs the hook at most once (its `recovered_once`
    // latch); nothing here loops. It is bounded by `recovery_pass_budget`: see
    // the rustdoc.
    base_path_cache.invalidate(session_id);
    let resolved = match tokio::time::timeout(
        recovery_pass_budget,
        base_path_cache.get_base_path(session_store, session_id),
    )
    .await
    {
        Ok(resolved) => resolved,
        // Elapsed: the recovery future is dropped, which releases the hook's
        // admission entry and abandons its checkpoint fetch. A drop between
        // the apply's two transactions leaves only the path-index half, the
        // partial state that apply is ordered to self-heal from (SR-2.7). No
        // evidence either way, so cancel nothing (as on a transient error,
        // below).
        Err(_elapsed) => {
            tracing::warn!(
                session_id = %session_id,
                budget_ms = recovery_pass_budget.as_millis() as u64,
                "acquire_session_cancel: recovery pass for a session missing from the live store exceeded its budget; returning UNAVAILABLE so the client retries"
            );
            Err(ServiceError::Unavailable(format!(
                "session recovery did not finish within {} ms; retry",
                recovery_pass_budget.as_millis()
            )))
        }
    };
    match resolved {
        // Restored (or recreated meanwhile). Not a session a `DeleteSession`
        // removed: the hook refuses those by their deletion marker, save the
        // narrow race its step 6a names. Proceed as the found path does,
        // with the token inserted before the first re-check: inserting a fresh
        // one now, after the check, would reopen the race the re-check closes.
        // A sweep that cancelled that token during the resolve ends its streams
        // retryably, and their retries find the restored session.
        Ok(_) => {
            tracing::info!(
                session_id = %session_id,
                "acquire_session_cancel: session was missing from the live store at the re-check and resolved again through the recovery pass"
            );
            Ok(token)
        }
        Err(ServiceError::NotFound(msg)) => {
            tracing::debug!(
                session_id = %session_id,
                "acquire_session_cancel: session missing from the live store and not recoverable; cancelled its streams and returning NotFound"
            );
            // The cause first: a deletion if the session carries the deletion
            // marker, else absent from the store (see the rustdoc). This is the
            // cross-replica race the re-check exists for — the store entry is
            // gone and the `le:session:deleted` fan-out has not reached this
            // replica yet — and the first cause recorded wins, so without the
            // marker read the late fan-out could not restore the deletion
            // endings of the streams sharing this token.
            let cause = absent_session_cancel_cause(session_store, session_id).await;
            // Remove whatever entry the map holds for the session now — ours,
            // or one another caller inserted — and cancel it. Then cancel the
            // handle we hold too, in case the entry was already replaced or
            // removed underneath us, so no stream sharing our token outlives
            // the session. A `DeleteSession` that already cancelled the token
            // keeps its cause (the first recorded cause wins).
            if let Some((_, entry)) = session_tokens.remove(session_id) {
                entry.cancel_with(cause);
            }
            token.cancel_with(cause);
            Err(ServiceError::NotFound(msg))
        }
        // A transient store error, or a recovery that failed (`Unavailable` /
        // `Internal`): no evidence the session is gone, so the entry is left
        // alone and the error surfaced, as on the first re-check.
        Err(other) => Err(other),
    }
}

// ── b.j5t item 5 / b.f36 A6: session_tokens crash-recovery sweeper ─────────
//
// `session_tokens` is a per-replica `DashMap<session_id, SessionCancel>`
// keyed by every session that this replica has either (a) created itself, or
// (b) seen through `acquire_session_cancel` for a streaming RPC. Entries are
// pruned by `apply_session_deleted` on the pub/sub fan-out, but that only
// fires when a client calls `DeleteSession`, and only if this replica receives
// the notice. Sessions whose owning client crashes — or that are simply never
// deleted — would otherwise keep their entries for the life of the process:
// on a long-running pod with sporadic client crashes, an unbounded slow leak.
//
// The sweeper iterates the live entries, probes Valkey via
// `SessionStore::get_base_path` (one HGET per entry), and removes + cancels
// any entry whose underlying session is gone (`NotFound`). Before cancelling it
// reads the session's deletion marker (one EXISTS per entry it is about to
// cancel) to pick the cause: a marked session was deleted and its streams get
// the deletion endings (`SessionCancelCause::Deleted`), which matters when this
// replica missed the `le:session:deleted` fan-out; an unmarked one was lost
// with the keyspace, or its marker could not be read, and its streams end
// retryably (`AbsentFromStore`) so a retry can recover it (the retry's setup
// runs the session recovery hook; see `acquire_session_cancel`). The sweeper
// itself reads the store only and never consults the recovery hook.
// Transient Valkey errors are *not* treated as absence — a Valkey blip must not
// mass-cancel every active session's streams. Only an unambiguous `NotFound`
// triggers cleanup.
//
// Besides its periodic pass the sweeper runs a *requested* pass whenever the
// session-deleted relay may have missed notices: its receiver lagged, or the
// pub/sub connection reconnected (the relay re-SUBSCRIBEs first). A request
// goes through `SessionRevalidation`; requests made while one is pending
// collapse into it. A requested pass is deleted-only
// (`revalidate_deleted_sessions`): it cancels, with `Deleted`, only the
// entries whose session is missing *and* marked, and leaves unmarked-missing
// entries — and entries whose probe or marker read fails — to the periodic
// pass. A lost notice can only belong to a marked session — `DeleteSession`
// deletes the metadata and writes the marker in one transaction before it
// publishes — so the pass finds every session whose notice was lost before the
// lag or the re-SUBSCRIBE, except one whose probe or marker read failed in that
// pass; those count as `deferred` on the pass's summary line, and the periodic
// pass ends their streams (as `Deleted` when it reads the marker). Cancelling
// unmarked sessions there too would, after a store that lost its keyspace
// comes back, make every replica cancel every session at the same moment (they
// all reconnect together); the periodic pass spreads that over an interval.
// The periodic pass keeps its full scope and its cadence, which a requested
// pass does not reset; it bounds only notice loss that cannot be detected.
//
// Sized for a low duty-cycle: the default interval is 60 s, which at 1 000
// active sessions per replica is ~17 HGETs/s — well under the data pool's
// hot-path budget. A requested pass costs the same one HGET per entry, run
// back to back, so a relay that keeps lagging (presence traffic alone can do
// that) would otherwise keep the sweeper probing at about one HGET per round
// trip, just when the store is most loaded, and a reconnect would start a
// pass on every replica at once. So after each requested pass the sweeper
// waits `REVALIDATION_MIN_GAP` before it takes the next request; a request
// made meanwhile is kept, never lost, and the periodic pass still runs. And a
// pass a reconnect asked for first waits a random delay of up to
// `REVALIDATION_RECONNECT_JITTER_MAX`, drawn per replica on the sweeper's side
// (the relay never waits); a lag request made during that delay is served by
// the same pass. A lost notice that the relay detects therefore ends its
// streams within the gap, plus that delay after a reconnect, plus one pass.
// Operators tune the interval via
// `LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS`, parsed by
// `Config::from_env` into `Config::session_tokens_sweep_interval_secs` and
// plumbed through `main.rs` to `spawn_session_tokens_sweeper`. The
// `SESSION_TOKENS_SWEEP_INTERVAL_SECS` constant below is the default the
// config field falls back to when the env var is unset.

/// Default sweep interval for the `session_tokens` crash-recovery sweeper,
/// used when `LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS` is unset. Tuned
/// for a low background duty-cycle — see the module-level comment.
pub const SESSION_TOKENS_SWEEP_INTERVAL_SECS: u64 = 60;

/// The least time between the end of one requested (deleted-only) pass of
/// the `session_tokens` sweeper and the start of the next. A code constant,
/// not an operator knob.
///
/// **Why 5 s.** It caps a relay that lags continuously at one requested pass
/// per 5 s plus the pass's own length: at 1 000 sessions on a replica, at most
/// about 200 HGETs/s, about 12 times the periodic sweep's 17 HGETs/s, where an
/// ungapped run of passes probes at one HGET per round trip (1 000/s and more
/// at sub-millisecond latency) for as long as the lag lasts. And it keeps a
/// detected lost notice's bound — this gap, plus
/// [`REVALIDATION_RECONNECT_JITTER_MAX`] after a reconnect, plus one pass — at
/// seconds, well inside the 60 s default periodic interval, which bounds only
/// the loss the relay cannot detect. The gap follows a pass; it never delays
/// the first.
pub const REVALIDATION_MIN_GAP: std::time::Duration = std::time::Duration::from_secs(5);

/// The upper end of the random delay, uniform in `[0, max]`, before a
/// requested pass that a pub/sub reconnect asked for. A code constant, not an
/// operator knob.
///
/// **Why.** A store failover or restart drops every replica's pub/sub
/// connection at once, so without it every replica would start probing all its
/// sessions at the same moment, against a store that has just come back. A
/// delay drawn per replica spreads those passes over 5 s, the same span as
/// [`REVALIDATION_MIN_GAP`], which keeps a detected loss's bound at seconds. A
/// lag request stays immediate (subject to the gap); one made during the delay
/// is served by the same pass.
pub const REVALIDATION_RECONNECT_JITTER_MAX: std::time::Duration =
    std::time::Duration::from_secs(5);

/// A request handle for a prompt deleted-only pass of the `session_tokens`
/// sweeper ([`revalidate_deleted_sessions`]).
///
/// `main` creates one and passes clones to the session-deleted relay
/// ([`spawn_session_delete_subscriber`]), which calls
/// [`request`](Self::request) when its receiver lagged and
/// [`request_after_reconnect`](Self::request_after_reconnect) after a pub/sub
/// reconnect, and to [`spawn_session_tokens_sweeper`], which runs the pass.
/// Requests collapse: any number made before the sweeper starts a pass are
/// served by that one pass. A request made after the sweeper has exited does
/// nothing. Neither request blocks, so the relay never waits on the sweeper.
#[derive(Clone, Debug)]
pub struct SessionRevalidation {
    inner: Arc<RevalidationState>,
}

#[derive(Debug)]
struct RevalidationState {
    /// Wakes the sweeper's wait. A permit stored while no one waits, or left
    /// over once the request was already taken, only makes the wait re-check
    /// `pending`.
    notify: tokio::sync::Notify,
    /// The pending request: [`NO_REQUEST`], [`LAG_REQUEST`] or
    /// [`RECONNECT_REQUEST`]. The requests raise it with `fetch_max`, so a
    /// reconnect's delay is never lost to a lag request; the sweeper takes it
    /// with one `swap` as a pass starts ([`SessionRevalidation::begin_pass`]).
    pending: std::sync::atomic::AtomicU8,
    /// Upper end of that delay: [`REVALIDATION_RECONNECT_JITTER_MAX`], or zero
    /// for a test handle.
    reconnect_jitter_max: std::time::Duration,
}

/// [`RevalidationState::pending`]: no pass wanted.
const NO_REQUEST: u8 = 0;
/// [`RevalidationState::pending`]: a pass wanted at once (a relay lag).
const LAG_REQUEST: u8 = 1;
/// [`RevalidationState::pending`]: a pass wanted after a random delay (a
/// pub/sub reconnect).
const RECONNECT_REQUEST: u8 = 2;

impl Default for SessionRevalidation {
    fn default() -> Self {
        Self::with_reconnect_jitter_max(REVALIDATION_RECONNECT_JITTER_MAX)
    }
}

impl SessionRevalidation {
    /// A handle with no request pending, whose reconnect requests wait a
    /// random delay of up to [`REVALIDATION_RECONNECT_JITTER_MAX`].
    pub fn new() -> Self {
        Self::default()
    }

    /// A handle whose reconnect requests start their pass with no delay, so a
    /// test can drive a reconnect deterministically.
    #[cfg(any(test, feature = "test-support"))]
    pub fn without_reconnect_jitter() -> Self {
        Self::with_reconnect_jitter_max(std::time::Duration::ZERO)
    }

    fn with_reconnect_jitter_max(reconnect_jitter_max: std::time::Duration) -> Self {
        Self {
            inner: Arc::new(RevalidationState {
                notify: tokio::sync::Notify::new(),
                pending: std::sync::atomic::AtomicU8::new(NO_REQUEST),
                reconnect_jitter_max,
            }),
        }
    }

    /// Ask for a deleted-only pass, after a relay lag. Never blocks;
    /// collapses into a pass already pending.
    pub fn request(&self) {
        self.raise(LAG_REQUEST);
    }

    /// Ask for a deleted-only pass after a pub/sub reconnect: like
    /// [`request`](Self::request), but the sweeper waits a random delay of up
    /// to the handle's jitter bound before the pass, so replicas that all
    /// reconnected together spread their passes. Never blocks.
    pub fn request_after_reconnect(&self) {
        self.raise(RECONNECT_REQUEST);
    }

    fn raise(&self, request: u8) {
        self.inner
            .pending
            .fetch_max(request, std::sync::atomic::Ordering::AcqRel);
        self.inner.notify.notify_one();
    }

    /// Wait until a request is pending, without taking it: the sweeper's
    /// request arm. Returns at once if one is already pending. Cancel-safe:
    /// a dropped wait leaves the request pending for the next one.
    pub async fn requested(&self) {
        loop {
            if self
                .inner
                .pending
                .load(std::sync::atomic::Ordering::Acquire)
                != NO_REQUEST
            {
                return;
            }
            self.inner.notify.notified().await;
        }
    }

    /// The delay the sweeper waits before a pending pass: uniform in
    /// `[0, max]` when a reconnect asked for it, else zero.
    fn delay_before_pass(&self) -> std::time::Duration {
        use rand::Rng;
        if self
            .inner
            .pending
            .load(std::sync::atomic::Ordering::Acquire)
            != RECONNECT_REQUEST
        {
            return std::time::Duration::ZERO;
        }
        let max = u64::try_from(self.inner.reconnect_jitter_max.as_nanos()).unwrap_or(u64::MAX);
        std::time::Duration::from_nanos(rand::rng().random_range(0..=max))
    }

    /// Take the pending request as a pass starts. Every request made before
    /// this call is served by the pass; a later one stays pending for the
    /// next. Returns whether a request was pending.
    fn begin_pass(&self) -> bool {
        self.inner
            .pending
            .swap(NO_REQUEST, std::sync::atomic::Ordering::AcqRel)
            != NO_REQUEST
    }
}

/// Which entries a `session_tokens` sweep cancels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SweepScope {
    /// The periodic pass: every entry whose session is missing, with the
    /// cause its deletion marker picks ([`absent_session_cancel_cause`]).
    All,
    /// A requested pass: only entries whose session is missing and carries
    /// the deletion marker, with [`SessionCancelCause::Deleted`].
    DeletedOnly,
}

/// What one `session_tokens` sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepOutcome {
    /// Entries probed against the store (one HGET each).
    pub probed: usize,
    /// Entries removed and cancelled.
    pub cancelled: usize,
    /// Entries left for a later pass because a store read failed: the probe
    /// (in either pass), or, in a requested pass, the deletion-marker read. A
    /// periodic pass cancels on a failed marker read, so it never defers one.
    pub deferred: usize,
}

/// Probe every entry in `session_tokens` against Valkey and remove + cancel
/// any whose owning session is absent (b.j5t item 5), recording
/// [`SessionCancelCause::Deleted`] when the session carries the deletion
/// marker and [`SessionCancelCause::AbsentFromStore`] otherwise, including
/// when the marker read fails (the cancel happens either way). This is the
/// sweeper's periodic pass; [`revalidate_deleted_sessions`] is its requested,
/// deleted-only pass.
///
/// Pure I/O: callers compose this into a tokio interval. Returns the number
/// of entries removed so tests can drive a single sweep and assert.
pub async fn sweep_session_tokens(
    session_tokens: &DashMap<String, SessionCancel>,
    session_store: &SessionStore,
    base_path_cache: &super::session_cache::SessionBasePathCache,
) -> usize {
    sweep_session_tokens_scoped(
        session_tokens,
        session_store,
        base_path_cache,
        SweepScope::All,
    )
    .await
    .cancelled
}

/// The sweeper's requested pass: probe every entry in `session_tokens` like
/// [`sweep_session_tokens`], but remove + cancel, with
/// [`SessionCancelCause::Deleted`], only the entries whose session is absent
/// and carries the deletion marker. Entries whose session is absent without a
/// marker are left for the periodic pass (see the module-level comment for why
/// the scope is deleted-only), and so are entries whose probe or marker read
/// fails: a deleted session among those is found by the periodic pass, not by
/// this one. Those failures count as [`SweepOutcome::deferred`].
///
/// Logs one summary line when it ends, with the entries probed, cancelled and
/// deferred: at INFO when it cancelled or deferred any, else at DEBUG. While
/// presence traffic keeps the session-deleted relay lagging, a replica runs a
/// pass every [`REVALIDATION_MIN_GAP`], and nearly all of them find nothing;
/// at INFO those would be about 720 lines an hour per replica. Each cancelled
/// entry also logs its own INFO line.
pub async fn revalidate_deleted_sessions(
    session_tokens: &DashMap<String, SessionCancel>,
    session_store: &SessionStore,
    base_path_cache: &super::session_cache::SessionBasePathCache,
) -> SweepOutcome {
    let outcome = sweep_session_tokens_scoped(
        session_tokens,
        session_store,
        base_path_cache,
        SweepScope::DeletedOnly,
    )
    .await;
    if outcome.cancelled > 0 || outcome.deferred > 0 {
        tracing::info!(
            probed = outcome.probed,
            cancelled = outcome.cancelled,
            deferred = outcome.deferred,
            "session_tokens sweep: requested pass for deleted sessions done"
        );
    } else {
        tracing::debug!(
            probed = outcome.probed,
            cancelled = outcome.cancelled,
            deferred = outcome.deferred,
            "session_tokens sweep: requested pass for deleted sessions done"
        );
    }
    outcome
}

async fn sweep_session_tokens_scoped(
    session_tokens: &DashMap<String, SessionCancel>,
    session_store: &SessionStore,
    base_path_cache: &super::session_cache::SessionBasePathCache,
    scope: SweepScope,
) -> SweepOutcome {
    // Snapshot keys to avoid holding DashMap shards across `.await`. The
    // snapshot may briefly diverge from the live map if a concurrent
    // CreateSession or DeleteSession runs during the sweep, but every
    // entry's removal is gated on a fresh Valkey probe so we never evict
    // a session that exists at probe time.
    let keys: Vec<String> = session_tokens.iter().map(|e| e.key().clone()).collect();
    let mut outcome = SweepOutcome::default();
    for session_id in keys {
        outcome.probed += 1;
        match session_store.get_base_path(&session_id).await {
            Ok(_) => {
                // Session still present — leave the entry alone.
            }
            Err(ServiceError::NotFound(_)) => {
                let cause = match scope {
                    // Deletion or keyspace loss, by the deletion marker; a
                    // failed marker read falls back to the retryable cause
                    // and still cancels (`absent_session_cancel_cause`).
                    SweepScope::All => {
                        absent_session_cancel_cause(session_store, &session_id).await
                    }
                    // Only a marked session can have lost a deletion notice.
                    SweepScope::DeletedOnly => {
                        match session_store.deleted_marker_exists(&session_id).await {
                            Ok(true) => SessionCancelCause::Deleted,
                            Ok(false) => continue,
                            Err(e) => {
                                tracing::debug!(
                                    session_id = %session_id,
                                    error = %e,
                                    "session_tokens sweep: deletion-marker read failed in a requested pass, leaving entry for the periodic sweep"
                                );
                                outcome.deferred += 1;
                                continue;
                            }
                        }
                    }
                };
                if let Some((_, token)) = session_tokens.remove(&session_id) {
                    token.cancel_with(cause);
                    base_path_cache.invalidate(&session_id);
                    outcome.cancelled += 1;
                    tracing::info!(
                        session_id = %session_id,
                        ?cause,
                        "session_tokens sweep: removed entry for absent session"
                    );
                }
            }
            Err(e) => {
                // Transient Valkey error — log and skip. The next sweep
                // retries; we MUST NOT mass-evict during a Valkey blip.
                outcome.deferred += 1;
                tracing::debug!(
                    session_id = %session_id,
                    error = %e,
                    "session_tokens sweep: probe failed, leaving entry for next sweep"
                );
            }
        }
    }
    outcome
}

/// Spawn the background task that sweeps `session_tokens`: a full pass
/// ([`sweep_session_tokens`]) every `interval`, the first one `interval`
/// after the spawn, and a deleted-only pass ([`revalidate_deleted_sessions`])
/// when `revalidation` is requested, at most one per [`REVALIDATION_MIN_GAP`]
/// after the previous requested pass ends. A request made inside the gap is
/// kept and served when the gap ends; the periodic pass runs meanwhile. A pass
/// a reconnect asked for ([`SessionRevalidation::request_after_reconnect`])
/// first waits a random delay of up to the handle's jitter bound
/// ([`REVALIDATION_RECONNECT_JITTER_MAX`] in production), and the periodic pass
/// waits with it; requests made during that delay are served by the pass. A
/// requested pass does not reset the interval. The task exits when `cancel`
/// fires, including during that delay.
///
/// Mirrors `session_cache::spawn_cache_sweeper`'s shape (b.j5t item 5).
pub fn spawn_session_tokens_sweeper(
    session_tokens: Arc<DashMap<String, SessionCancel>>,
    session_store: Arc<SessionStore>,
    base_path_cache: super::session_cache::SessionBasePathCache,
    interval: std::time::Duration,
    revalidation: SessionRevalidation,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // skip the immediate first tick

        // When the next requested pass may start; a request waits for it.
        let mut next_requested_pass = tokio::time::Instant::now();

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("session_tokens sweeper shutting down");
                    break;
                }
                _ = tick.tick() => {
                    let _removed = sweep_session_tokens(
                        &session_tokens,
                        &session_store,
                        &base_path_cache,
                    ).await;
                }
                // Rebuilt every loop with the same deadline, so a tick that
                // wins during the gap does not restart it. A dropped wait
                // leaves the request pending (`SessionRevalidation::requested`),
                // so none is lost.
                () = async {
                    tokio::time::sleep_until(next_requested_pass).await;
                    revalidation.requested().await;
                } => {
                    // After a reconnect, a random delay spreads the fleet's
                    // passes; a request made during it is served by this pass.
                    let delay = revalidation.delay_before_pass();
                    if !delay.is_zero() {
                        tokio::select! {
                            () = cancel.cancelled() => {
                                tracing::info!("session_tokens sweeper shutting down");
                                break;
                            }
                            () = tokio::time::sleep(delay) => {}
                        }
                    }
                    if !revalidation.begin_pass() {
                        continue;
                    }
                    let _outcome = revalidate_deleted_sessions(
                        &session_tokens,
                        &session_store,
                        &base_path_cache,
                    ).await;
                    next_requested_pass = tokio::time::Instant::now() + REVALIDATION_MIN_GAP;
                }
            }
        }
    })
}

/// Subscribe to the `le:session:deleted` pub/sub channel on the given
/// [`SubscriberClient`]. Callers spawn [`spawn_session_delete_subscriber`]
/// afterwards to drain events; that relay calls this again after every
/// pub/sub reconnect (SUBSCRIBE is idempotent).
pub async fn subscribe_session_deleted_channel(
    subscriber: &SubscriberClient,
    key_schema: &KeySchema,
) -> Result<(), fred::error::Error> {
    let channel = key_schema.session_deleted_channel();
    subscriber.subscribe(&channel).await
}

/// Spawn the background task that consumes `le:session:deleted` messages and
/// drops + cancels matching entries in `session_tokens` on this replica.
///
/// The task exits when `cancel` fires. The caller is responsible for calling
/// [`subscribe_session_deleted_channel`] before spawning so the subscriber
/// client is actually listening on the channel. The task's receivers are taken
/// before it is spawned, so it sees every message published once this function
/// returns.
///
/// The task's receiver is on the subscriber client's message broadcast
/// channel (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY` messages),
/// shared with the presence relay. The task drains everything queued each time
/// it runs ([`PubsubRelayReceiver`]), yielding after one channel capacity's
/// worth of messages. If it still falls more than the capacity behind, the
/// skipped messages are lost for this replica: each drain batch that lost any
/// logs one WARN, adds their sum to
/// `live_edit_pubsub_relay_lagged_total{relay="session_deleted"}` on
/// `metrics`, and requests a deleted-only `session_tokens` pass on
/// `revalidation`. The lost messages may have been on other channels, so a lag
/// means notices *may* have been missed; the pass cancels any session they
/// were for.
///
/// The task also watches the client's reconnect notifications. A notice
/// published while the connection is down is lost, so after each reconnect
/// (or a lag on that notification channel) the task re-sends the SUBSCRIBE —
/// fred's own resubscription runs too — and then requests a pass
/// ([`SessionRevalidation::request_after_reconnect`]: the sweeper runs it after
/// a random delay, so the relay itself never waits). A failed SUBSCRIBE is
/// logged at WARN and the pass is requested anyway; the periodic pass bounds
/// whatever is left.
///
/// The re-SUBSCRIBE is in flight for up to one command timeout, and the task
/// keeps draining messages meanwhile: it polls the SUBSCRIBE as one more arm
/// of its loop rather than awaiting it, so a reconnect under presence traffic
/// does not make the relay lag. At most one re-SUBSCRIBE is in flight. A
/// reconnect notified while one is in flight marks one more as owed, started
/// when the current one completes (its pass is requested first); further
/// reconnects meanwhile collapse into that owed one. So every reconnect is
/// followed by a SUBSCRIBE sent after it and then a pass requested once that
/// SUBSCRIBE completed. The in-flight one is never replaced: dropping fred's
/// SUBSCRIBE does not withdraw a command its router already holds, so
/// restarting would stack SUBSCRIBEs, and under a flapping connection could
/// postpone the pass indefinitely.
///
/// Shutdown (`cancel`) ends the task at once, a SUBSCRIBE in flight included.
/// If the message channel closes while a re-SUBSCRIBE is in flight, the task
/// requests the pass before it ends rather than leave that reconnect without
/// one.
pub fn spawn_session_delete_subscriber(
    subscriber: SubscriberClient,
    session_tokens: Arc<DashMap<String, SessionCancel>>,
    base_path_cache: super::session_cache::SessionBasePathCache,
    key_schema: KeySchema,
    metrics: SharedMetrics,
    revalidation: SessionRevalidation,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    spawn_session_delete_relay(
        subscriber,
        session_tokens,
        base_path_cache,
        key_schema,
        metrics,
        revalidation,
        cancel,
        None,
    )
}

/// [`spawn_session_delete_subscriber`], with every re-SUBSCRIBE the relay
/// starts after a reconnect held on `hold` until the test releases it, so a
/// test can keep one in flight deterministically.
///
/// A test seam, compiled only for tests and the `test-support` feature.
#[cfg(any(test, feature = "test-support"))]
#[allow(clippy::too_many_arguments)]
pub fn spawn_session_delete_subscriber_with_resubscribe_hold(
    subscriber: SubscriberClient,
    session_tokens: Arc<DashMap<String, SessionCancel>>,
    base_path_cache: super::session_cache::SessionBasePathCache,
    key_schema: KeySchema,
    metrics: SharedMetrics,
    revalidation: SessionRevalidation,
    cancel: CancellationToken,
    hold: ResubscribeHold,
) -> tokio::task::JoinHandle<()> {
    spawn_session_delete_relay(
        subscriber,
        session_tokens,
        base_path_cache,
        key_schema,
        metrics,
        revalidation,
        cancel,
        Some(hold),
    )
}

/// A test hold on the session-deleted relay's re-SUBSCRIBEs
/// ([`spawn_session_delete_subscriber_with_resubscribe_hold`]). Each
/// re-SUBSCRIBE the relay starts after a reconnect counts itself as started
/// when first polled, then waits for one release before it sends its
/// SUBSCRIBE. Releases are permits: one given before a re-SUBSCRIBE starts
/// lets it through without waiting. Clones share one hold.
///
/// A test seam, compiled only for tests and the `test-support` feature.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct ResubscribeHold {
    inner: Arc<ResubscribeHoldState>,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
struct ResubscribeHoldState {
    /// Re-SUBSCRIBEs started so far.
    started: tokio::sync::watch::Sender<usize>,
    /// One permit per release not yet taken.
    releases: tokio::sync::Semaphore,
}

#[cfg(any(test, feature = "test-support"))]
impl Default for ResubscribeHold {
    fn default() -> Self {
        Self {
            inner: Arc::new(ResubscribeHoldState {
                started: tokio::sync::watch::channel(0).0,
                releases: tokio::sync::Semaphore::new(0),
            }),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl ResubscribeHold {
    /// A hold with nothing started and no release given.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many re-SUBSCRIBEs the relay has started, held or released.
    pub fn started(&self) -> usize {
        *self.inner.started.borrow()
    }

    /// Wait until the relay has started at least `count` re-SUBSCRIBEs.
    pub async fn wait_started(&self, count: usize) {
        let mut started = self.inner.started.subscribe();
        // The sender lives in `self`, so the wait cannot see it closed.
        let _ = started.wait_for(|n| *n >= count).await;
    }

    /// Let `count` re-SUBSCRIBEs, held now or started later, send their
    /// SUBSCRIBE.
    pub fn release(&self, count: usize) {
        self.inner.releases.add_permits(count);
    }

    /// The relay's side: count this re-SUBSCRIBE as started, then wait for a
    /// release and take it.
    async fn pass(&self) {
        self.inner.started.send_modify(|n| *n += 1);
        if let Ok(permit) = self.inner.releases.acquire().await {
            permit.forget();
        }
    }
}

/// What the session-deleted relay holds its re-SUBSCRIBEs on:
/// [`ResubscribeHold`] under the test seams, uninhabited otherwise, so a
/// production relay carries none.
#[cfg(any(test, feature = "test-support"))]
type RelayResubscribeHold = ResubscribeHold;
/// See the test-seam definition: uninhabited here.
#[cfg(not(any(test, feature = "test-support")))]
type RelayResubscribeHold = std::convert::Infallible;

/// A re-SUBSCRIBE the session-deleted relay has in flight, polled as an arm of
/// its loop ([`resubscribe_completed`]).
type InFlightResubscribe<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<(), fred::error::Error>> + Send + 'a>,
>;

/// Start one re-SUBSCRIBE for the session-deleted relay: the SUBSCRIBE of
/// [`subscribe_session_deleted_channel`], after the test hold's release when
/// there is one.
fn start_resubscribe<'a>(
    subscriber: &'a SubscriberClient,
    key_schema: &'a KeySchema,
    hold: Option<&'a RelayResubscribeHold>,
) -> InFlightResubscribe<'a> {
    Box::pin(async move {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(hold) = hold {
            hold.pass().await;
        }
        #[cfg(not(any(test, feature = "test-support")))]
        if let Some(never) = hold {
            match *never {}
        }
        subscribe_session_deleted_channel(subscriber, key_schema).await
    })
}

/// The session-deleted relay's re-SUBSCRIBE arm: the outcome of the
/// re-SUBSCRIBE in flight, or never when none is. Cancel-safe: a dropped poll
/// leaves the SUBSCRIBE in `in_flight`, to be polled again on the next pass of
/// the loop.
async fn resubscribe_completed(
    in_flight: &mut Option<InFlightResubscribe<'_>>,
) -> Result<(), fred::error::Error> {
    match in_flight {
        Some(subscribe) => subscribe.as_mut().await,
        None => std::future::pending().await,
    }
}

/// The session-deleted relay itself ([`spawn_session_delete_subscriber`]),
/// with the test seam's hold on its re-SUBSCRIBEs when one is given.
#[allow(clippy::too_many_arguments)]
fn spawn_session_delete_relay(
    subscriber: SubscriberClient,
    session_tokens: Arc<DashMap<String, SessionCancel>>,
    base_path_cache: super::session_cache::SessionBasePathCache,
    key_schema: KeySchema,
    metrics: SharedMetrics,
    revalidation: SessionRevalidation,
    cancel: CancellationToken,
    hold: Option<RelayResubscribeHold>,
) -> tokio::task::JoinHandle<()> {
    let mut messages = PubsubRelayReceiver::new(&subscriber);
    let mut reconnects = Some(subscriber.reconnect_rx());
    tokio::spawn(async move {
        let channel_name = key_schema.session_deleted_channel();
        // The re-SUBSCRIBE in flight after a reconnect, if any, and whether a
        // reconnect notified meanwhile owes one more (see
        // `spawn_session_delete_subscriber`).
        let mut resubscribe: Option<InFlightResubscribe<'_>> = None;
        let mut resubscribe_owed = false;

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("session-deleted subscriber shutting down");
                    break;
                }
                first = messages.recv() => {
                    let batch = messages.drain(first, |msg| {
                        apply_session_deleted_message(
                            &session_tokens,
                            &base_path_cache,
                            &channel_name,
                            msg,
                        );
                    });
                    if batch.lagged > 0 {
                        metrics.add_pubsub_relay_lagged(PUBSUB_RELAY_SESSION_DELETED, batch.lagged);
                        tracing::warn!(
                            lagged = batch.lagged,
                            "session-deleted subscriber lagged, events may have been lost"
                        );
                        revalidation.request();
                    }
                    match batch.end {
                        DrainEnd::Empty => {}
                        DrainEnd::YieldDue => {
                            tokio::task::yield_now().await;
                            if cancel.is_cancelled() {
                                tracing::info!("session-deleted subscriber shutting down");
                                break;
                            }
                        }
                        DrainEnd::Closed => {
                            tracing::info!("session-deleted subscriber channel closed");
                            if resubscribe.is_some() {
                                // The task ends without waiting for the
                                // SUBSCRIBE; its reconnect still gets its pass.
                                revalidation.request_after_reconnect();
                            }
                            break;
                        }
                    }
                }
                reconnect = recv_reconnect(&mut reconnects) => {
                    if let Err(tokio::sync::broadcast::error::RecvError::Closed) = reconnect {
                        // fred closes its notification channels together, the
                        // message channel included, so the message arm ends
                        // the task; stop watching this one meanwhile.
                        reconnects = None;
                        continue;
                    }
                    // A reconnect, or a lag on the reconnect channel (at least
                    // one reconnect): notices published during the gap are
                    // lost. SUBSCRIBE again, then re-check every session — or,
                    // with a re-SUBSCRIBE already in flight (it may have gone
                    // out on the dropped connection), owe one more.
                    if resubscribe.is_some() {
                        resubscribe_owed = true;
                    } else {
                        resubscribe = Some(start_resubscribe(&subscriber, &key_schema, hold.as_ref()));
                    }
                }
                result = resubscribe_completed(&mut resubscribe), if resubscribe.is_some() => {
                    if let Err(e) = result {
                        tracing::warn!(
                            error = %e,
                            "session-deleted subscriber: SUBSCRIBE after a pub/sub reconnect failed; requesting a session revalidation anyway"
                        );
                    }
                    revalidation.request_after_reconnect();
                    resubscribe = if std::mem::take(&mut resubscribe_owed) {
                        Some(start_resubscribe(&subscriber, &key_schema, hold.as_ref()))
                    } else {
                        None
                    };
                }
            }
        }
    })
}

/// The session-deleted relay's per-message step: apply `msg` if it is on the
/// session-deleted channel (`channel_name`), else drop it.
///
/// Every message on the pub/sub connection reaches this relay, the
/// per-session presence channels' included, so the filter runs on all of
/// them; it compares the channel name in place and allocates nothing.
fn apply_session_deleted_message(
    session_tokens: &DashMap<String, SessionCancel>,
    base_path_cache: &super::session_cache::SessionBasePathCache,
    channel_name: &str,
    msg: fred::types::Message,
) {
    if *msg.channel != *channel_name {
        return;
    }

    let session_id: String = match msg.value.convert() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                channel = %channel_name,
                error = %e,
                "failed to convert session-deleted payload to string"
            );
            return;
        }
    };

    apply_session_deleted(session_tokens, base_path_cache, &session_id);
}

/// The session-deleted relay's per-message step, for the relay bench
/// (`benches/presence_fanout.rs`): [`apply_session_deleted_message`] itself,
/// which stays private to the relay.
#[cfg(feature = "test-support")]
pub fn apply_session_deleted_message_for_bench(
    session_tokens: &DashMap<String, SessionCancel>,
    base_path_cache: &super::session_cache::SessionBasePathCache,
    channel_name: &str,
    msg: fred::types::Message,
) {
    apply_session_deleted_message(session_tokens, base_path_cache, channel_name, msg);
}

/// The session-deleted relay's reconnect arm: the next reconnect notification,
/// or never once the channel has closed.
async fn recv_reconnect(
    reconnects: &mut Option<tokio::sync::broadcast::Receiver<fred::types::config::Server>>,
) -> Result<fred::types::config::Server, tokio::sync::broadcast::error::RecvError> {
    match reconnects {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::session_cache::SessionBasePathCache;

    #[test]
    fn apply_session_deleted_removes_entry_and_cancels_token() {
        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        let token = CancellationToken::new();
        tokens.insert("sid-1".to_string(), token.clone().into());

        apply_session_deleted(&tokens, &cache, "sid-1");

        assert!(tokens.is_empty(), "entry must be removed");
        assert!(token.is_cancelled(), "token must be cancelled");
    }

    /// The `le:session:deleted` fan-out is published only by `DeleteSession`,
    /// so its cancel records `Deleted` — the cause that ends a subscription
    /// `NOT_FOUND` and a join `OK`.
    ///
    /// Recorded, not merely implied: an empty cause slot also reads `Deleted`,
    /// so the proof is a later store-absence cancel on the same token (the
    /// sweeper, or another setup) finding the slot already taken. A bare
    /// `cancel()` here would let that later cancel turn the deletion into a
    /// retryable ending.
    #[test]
    fn apply_session_deleted_records_the_deleted_cause() {
        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        let token = SessionCancel::new();
        tokens.insert("sid-1".to_string(), token.clone());

        apply_session_deleted(&tokens, &cache, "sid-1");
        token.cancel_absent_from_store();

        assert!(token.is_cancelled());
        assert_eq!(
            token.cause(),
            SessionCancelCause::Deleted,
            "the fan-out must record its cause, so a later store-absence cancel cannot replace it"
        );
    }

    // ── SessionCancel: the cause travels with the token ─────────────────────

    /// Each canceller records its cause, and every clone of the token reads
    /// it: the map entry and each stream's clone share one slot.
    #[test]
    fn session_cancel_cause_is_read_by_every_clone() {
        for (cancel, want) in [
            (
                SessionCancel::cancel_deleted as fn(&SessionCancel),
                SessionCancelCause::Deleted,
            ),
            (
                SessionCancel::cancel_absent_from_store,
                SessionCancelCause::AbsentFromStore,
            ),
        ] {
            let entry = SessionCancel::new();
            let stream_clone = entry.clone();
            cancel(&entry);
            assert!(stream_clone.is_cancelled(), "{want:?}");
            assert_eq!(stream_clone.cause(), want);
        }
    }

    /// The first recorded cause wins. A late `DeleteSession` fan-out after a
    /// store-absence cancel cannot rewrite the ending the streams already
    /// took, and a store-absence cancel after a deletion cannot turn a
    /// deletion into a retryable ending.
    #[test]
    fn session_cancel_first_recorded_cause_wins() {
        let absent_first = SessionCancel::new();
        absent_first.cancel_absent_from_store();
        absent_first.cancel_deleted();
        assert_eq!(absent_first.cause(), SessionCancelCause::AbsentFromStore);

        let deleted_first = SessionCancel::new();
        deleted_first.cancel_deleted();
        deleted_first.cancel_absent_from_store();
        assert_eq!(deleted_first.cause(), SessionCancelCause::Deleted);
    }

    /// A child — what each `JoinSession` polls — is cancelled with its parent
    /// and reads the parent's cause, so a join ends by the same cause as every
    /// other stream on the session.
    #[test]
    fn session_cancel_child_is_cancelled_with_the_parent_and_shares_its_cause() {
        let parent = SessionCancel::new();
        let child = parent.child();
        assert!(!child.is_cancelled());
        parent.cancel_absent_from_store();
        assert!(child.is_cancelled());
        assert_eq!(child.cause(), SessionCancelCause::AbsentFromStore);

        // And a child made from an already-cancelled parent is born cancelled,
        // with the cause.
        let late_child = parent.child();
        assert!(late_child.is_cancelled());
        assert_eq!(late_child.cause(), SessionCancelCause::AbsentFromStore);
    }

    /// A bare `cancel()` through the deref records no cause and reads
    /// `Deleted`, the meaning a session-token cancel had before causes existed.
    #[test]
    fn session_cancel_bare_cancel_reads_as_deleted() {
        let token = SessionCancel::new();
        token.cancel();
        assert!(token.is_cancelled());
        assert_eq!(token.cause(), SessionCancelCause::Deleted);
    }

    /// Both streaming RPCs end a store-absence cancel with this one status:
    /// retryable, and saying why.
    #[test]
    fn session_absent_status_is_retryable_unavailable() {
        let status = session_absent_status();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(
            status.message(),
            "session is missing from the live store; retry so it can be recovered"
        );
    }

    #[test]
    fn apply_session_deleted_absent_session_is_noop() {
        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        let token = CancellationToken::new();
        tokens.insert("other".to_string(), token.clone().into());

        apply_session_deleted(&tokens, &cache, "missing");

        assert_eq!(tokens.len(), 1, "existing entries are untouched");
        assert!(!token.is_cancelled(), "other tokens are not cancelled");
    }

    #[test]
    fn apply_session_deleted_empty_session_id_is_noop() {
        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        let token = CancellationToken::new();
        tokens.insert(String::new(), token.clone().into());

        apply_session_deleted(&tokens, &cache, "");

        assert_eq!(
            tokens.len(),
            1,
            "empty session_id must not drive any removal"
        );
        assert!(!token.is_cancelled());
    }

    #[test]
    fn apply_session_deleted_double_remove_is_safe() {
        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        tokens.insert("sid-1".to_string(), SessionCancel::new());

        apply_session_deleted(&tokens, &cache, "sid-1");
        // Second call on the now-absent entry must not panic.
        apply_session_deleted(&tokens, &cache, "sid-1");

        assert!(tokens.is_empty());
    }

    // ═════════════════════════════════════════════════════════════════════════
    // reconcile_ghost_members — recovery-aware ListSessions ghost reconciliation
    // (Epic `t1.fgj.7v`, SDD SR-2.7, Subtask `t3.fgj.7v.fp.ka`)
    // ═════════════════════════════════════════════════════════════════════════
    //
    // These tests pin the per-ghost state machine implemented in the helper
    // above:
    //
    //   - Recovery-hook-disabled (checkpoint_enabled=false) fallback
    //     -> SREM ghosts unconditionally (today's behaviour).
    //   - All members present -> no recovery attempts, no SREMs.
    //   - Ghost with checkpoint -> Recovered, member preserved, re-read
    //     `session_meta` appended to the response.
    //   - Ghost without checkpoint -> SREM as today; metric ticks
    //     `not_recoverable`.
    //   - Mixed ghost set (recoverable + truly-orphaned + Transient-error) ->
    //     response includes recovered; SREMs no-checkpoint only; warn logged
    //     for transient.
    //   - Empty path-index -> no recovery probes, response empty.
    //   - Ghost with Permanent error -> preserved (not SREM'd), excluded from
    //     response, warn logged.
    //   - Recovered -> Ok(None) race (recovery commits then DeleteSession runs
    //     before re-read) -> fall through to SREM as if NoCheckpoint.
    //   - Cluster slot safety: deferred to Task 7 `t2.fgj.7v.o7` per Subtask
    //     body. The SREMs share the `{hex(base_path)}` hash tag with the
    //     SMEMBERS that produced the members, so no new cross-slot operation
    //     is introduced; verified by inspection at the call site.
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::session_payload::SessionCheckpoint;
    use crate::checkpoint::store::{
        CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    };
    use crate::checkpoint::SessionRecoveryHook;
    use crate::config::Config;
    use crate::observability::metrics::{metrics_scrape_only, SharedMetrics};
    use crate::storage::session::SessionStore;
    use crate::storage::test_support::TestValkey;
    use async_trait::async_trait;
    use bytes::Bytes;
    use fred::interfaces::{HashesInterface, SetsInterface};
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    /// Distinguishing label so test-only metric reads do not collide with
    /// hooks built in another `tokio::test` running concurrently.
    const TEST_SERVICE_LABEL: &str = "test-handle-list-sessions";

    fn make_metrics() -> SharedMetrics {
        Arc::new(metrics_scrape_only(TEST_SERVICE_LABEL))
    }

    /// Probe the live label-cell counter value for a given outcome at
    /// `scope="session"` (matches what `SessionRecoveryHook` ticks), read from
    /// the rendered scrape exposition. The `result × scope × service` grid is
    /// pre-initialised in `Metrics::new`, so every cell renders from boot.
    fn metric_value(metrics: &SharedMetrics, result: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_recovery_attempts_total",
            &[
                ("result", result),
                ("scope", "session"),
                ("service", TEST_SERVICE_LABEL),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    fn test_config(enabled: bool) -> Config {
        Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url: String::new(),
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: 5,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: None,
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: None,
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: TEST_SERVICE_LABEL.to_string(),
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
            auth_protocol: crate::config::AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: crate::config::UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            session_tokens_sweep_interval_secs: super::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: enabled,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::InMemory,
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
        }
    }

    /// Build a `SessionRecoveryHook` backed by the given checkpoint store,
    /// `TestValkey`, and `enabled` flag. The hook short-circuits to
    /// `NoCheckpoint` when `enabled=false` (the load-bearing path the
    /// recovery-hook-disabled fallback test exercises).
    fn build_hook(
        tv: &TestValkey,
        store: Arc<dyn CheckpointStore>,
        metrics: SharedMetrics,
        enabled: bool,
    ) -> Arc<SessionRecoveryHook> {
        Arc::new(SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(enabled)),
            metrics,
        ))
    }

    /// Build a `SessionCheckpoint` whose serialized bytes the
    /// `InMemoryCheckpointStore` will replay through the recovery applier.
    /// `name` / `base_path` / `created_by` / `created_at` match the fields
    /// asserted by the tests' `get_session_meta` re-reads.
    fn sample_session_checkpoint(
        session_id: &str,
        name: &str,
        base_path: &str,
        created_by: &str,
        created_at: u64,
    ) -> SessionCheckpoint {
        let mut sc = SessionCheckpoint {
            schema_version: crate::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.to_owned(),
            name: name.to_owned(),
            base_path: base_path.to_owned(),
            created_by: created_by.to_owned(),
            created_at,
            layer_mappings: BTreeMap::new(),
            layers_index: BTreeSet::new(),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session_id.to_owned(),
        };
        sc.add_path_name_member(name, session_id);
        sc
    }

    /// Seed a `SessionCheckpoint` into an `InMemoryCheckpointStore`. Mirrors
    /// the wire shape `SessionCheckpointWriter::write_session_checkpoint`
    /// uses on the live path.
    async fn seed_checkpoint(store: &InMemoryCheckpointStore, sc: &SessionCheckpoint) {
        let bytes = sc.to_bytes().expect("SessionCheckpoint::to_bytes");
        let meta = CheckpointMeta {
            creator_replica_id: TEST_SERVICE_LABEL.to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: None,
            session_id: sc.session_id.clone(),
            layer_name: String::new(),
            created_by: sc.created_by.clone(),
            server_ts: 0,
            ..CheckpointMeta::default()
        };
        store
            .put_session_checkpoint(&sc.session_id, bytes, meta)
            .await
            .expect("put_session_checkpoint must succeed");
    }

    /// Seed a path-index ghost: the `path_names_index` and `path_index` keys
    /// list `session_id` (under `name`), but `le:{sid}:sess` is absent —
    /// exactly the state `SessionStore::list` surfaces as a `GhostMember`.
    async fn seed_ghost(tv: &TestValkey, base_path: &str, name: &str, session_id: &str) {
        let keys = tv.key_schema();
        let path_index_key = keys.path_index(base_path);
        let path_names_key = keys.path_names_index(base_path);
        let composite = KeySchema::path_name_member(name, session_id);
        let _: () = tv
            .client()
            .pool
            .sadd(&path_index_key, session_id)
            .await
            .expect("SADD path_index");
        let _: () = tv
            .client()
            .pool
            .sadd(&path_names_key, composite)
            .await
            .expect("SADD path_names_index");
    }

    /// Test-only `CheckpointStore` whose `get_session_checkpoint` returns a
    /// classified error on demand. The recovery hook surfaces the error
    /// verbatim through `SessionRecoveryOutcome::Error`, which the
    /// reconcile helper must classify as "skip, do not SREM".
    struct ClassifiedErrorStore {
        err: fn() -> CheckpointError,
    }

    #[async_trait]
    impl CheckpointStore for ClassifiedErrorStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!()
        }
        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Ok(None)
        }
        // `get_layer_checkpoint` deliberately inherits the loud
        // `NotImplemented` trait default: this is a standalone fake with no
        // inner store, so there is nothing to forward to.
        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            Ok(vec![])
        }
        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }
        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!()
        }
        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Err((self.err)())
        }
        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }
        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: Bytes,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }
        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }
        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Ok(())
        }
        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            Ok(())
        }
        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }
        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Ok(vec![])
        }
        async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
            Ok(HashSet::new())
        }
    }

    // ─── Test 1 — Recovery-hook-disabled (checkpoint_enabled=false) ──────
    // Pins the backward-compat invariant: when checkpointing is disabled,
    // `try_recover_session` short-circuits to `NoCheckpoint` and the
    // reconcile helper SREMs every ghost — exactly today's behaviour for
    // `checkpoint_enabled=false` deployments and any future test that
    // builds a hook in disabled mode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_disabled_hook_srems_every_ghost() {
        let tv = TestValkey::setup().await;
        let base_path = "/disabled-stage";
        let sid_orphan = format!("ghost-{}", uuid::Uuid::new_v4());

        // Seed a single ghost (path-index entry without session_meta).
        seed_ghost(&tv, base_path, "ghost-name", &sid_orphan).await;
        let result = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .list(base_path)
            .await
            .expect("list");
        assert_eq!(result.present.len(), 0);
        assert_eq!(result.ghosts.len(), 1);

        // Hook in disabled mode -> NoCheckpoint without admission/store churn.
        let metrics = make_metrics();
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook = build_hook(&tv, store, metrics.clone(), false);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            base_path,
            result.ghosts,
        )
        .await;
        assert!(
            recovered.is_empty(),
            "disabled hook surfaces no recovered sessions",
        );

        // Both indices have been SREM'd of the ghost member.
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_index(base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(
            !path_index_members.contains(&sid_orphan),
            "ghost MUST be SREM'd from path_index on disabled-hook path"
        );

        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_names_index(base_path))
            .await
            .expect("SMEMBERS path_names_index");
        let composite = KeySchema::path_name_member("ghost-name", &sid_orphan);
        assert!(
            !path_names_members.contains(&composite),
            "ghost MUST be SREM'd from path_names_index on disabled-hook path"
        );

        // Disabled hook short-circuits without ticking any cell.
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    // ─── Test 2 — All members present, zero ghosts: no recovery probes ───
    // Pins the "no-checkpoint-needed" / hot-path invariant: when
    // SessionStore::list returns an empty ghosts vec, the reconcile helper
    // is a no-op (no admission churn, no metric tick).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_no_ghosts_is_noop() {
        let tv = TestValkey::setup().await;
        let metrics = make_metrics();
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook = build_hook(&tv, store, metrics.clone(), true);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            "/empty",
            Vec::new(),
        )
        .await;
        assert!(recovered.is_empty());

        // No metric cell ticked.
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(
            hook.admission_len(),
            0,
            "no admission entry created on empty ghosts list"
        );
    }

    // ─── Test 3 — Ghost with checkpoint -> Recovered, member preserved ───
    // Seeds a session-checkpoint, surfaces the ghost from `list`, drives
    // reconcile, asserts the session is materialised + appended to
    // `recovered`, and asserts the path-index member is NOT SREM'd.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_ghost_with_checkpoint_recovers_and_preserves_member() {
        let tv = TestValkey::setup().await;
        let base_path = "/recover-stage";
        let sid = format!("recover-{}", uuid::Uuid::new_v4());

        // Seed the ghost in the path-indices.
        seed_ghost(&tv, base_path, "recover-name", &sid).await;

        // Seed a session-checkpoint so the hook returns `Recovered`.
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let sc = sample_session_checkpoint(
            &sid,
            "recover-name",
            base_path,
            "recover-user",
            1_700_000_000_000_000,
        );
        seed_checkpoint(&in_mem, &sc).await;
        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = build_hook(&tv, store, metrics.clone(), true);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let result = session_store.list(base_path).await.expect("list");
        assert_eq!(result.present.len(), 0, "no live sessions before recovery");
        assert_eq!(result.ghosts.len(), 1);

        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            base_path,
            result.ghosts,
        )
        .await;

        assert_eq!(recovered.len(), 1, "Recovered ghost must be appended");
        assert_eq!(recovered[0].session_id, sid);
        assert_eq!(recovered[0].name, "recover-name");
        assert_eq!(recovered[0].base_path, base_path);
        assert_eq!(recovered[0].created_by, "recover-user");
        assert_eq!(recovered[0].created_at, 1_700_000_000_000_000);

        // session_meta is now present in Valkey (recovery materialised it).
        let session_meta_key = tv.key_schema().session_meta(&sid);
        let stored_name: Option<String> = tv
            .client()
            .pool
            .hget(&session_meta_key, "name")
            .await
            .expect("HGET name");
        assert_eq!(stored_name.as_deref(), Some("recover-name"));

        // Member MUST NOT have been SREM'd from either index — recovery
        // is the "preserve" path.
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_index(base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index_members.contains(&sid),
            "Recovered ghost MUST remain in path_index"
        );
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_names_index(base_path))
            .await
            .expect("SMEMBERS path_names_index");
        assert!(
            path_names_members.contains(&KeySchema::path_name_member("recover-name", &sid)),
            "Recovered ghost MUST remain in path_names_index"
        );

        // Recovery metric ticked exactly once on the `restored` cell.
        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    // ─── Test 4 — Ghost without checkpoint -> SREM + `not_recoverable` tick
    // Pins today's opportunistic-cleanup behaviour on the no-checkpoint
    // arm; metric ticks `not_recoverable` (harmonised vocabulary with
    // Epic 1).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_ghost_without_checkpoint_srems_and_ticks_not_recoverable() {
        let tv = TestValkey::setup().await;
        let base_path = "/no-cp";
        let sid = format!("no-cp-{}", uuid::Uuid::new_v4());

        seed_ghost(&tv, base_path, "no-cp-name", &sid).await;

        let metrics = make_metrics();
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook = build_hook(&tv, store, metrics.clone(), true);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let result = session_store.list(base_path).await.expect("list");
        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            base_path,
            result.ghosts,
        )
        .await;
        assert!(
            recovered.is_empty(),
            "no checkpoint -> no recovered session"
        );

        // Both indices SREM'd of the ghost.
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_index(base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(!path_index_members.contains(&sid));
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_names_index(base_path))
            .await
            .expect("SMEMBERS path_names_index");
        assert!(!path_names_members.contains(&KeySchema::path_name_member("no-cp-name", &sid)));

        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
    }

    // ─── Test 5 — Mixed ghost set: recoverable + truly-orphan + transient
    // Asserts the response contains the recovered session, the
    // truly-orphan member is SREM'd, and the transient-error member is
    // preserved (not SREM'd).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_mixed_ghosts_handles_each_arm_correctly() {
        // We can't easily mix all three arms against a single
        // CheckpointStore (the hook's behaviour is determined by what
        // `get_session_checkpoint` returns). Instead we drive each ghost
        // separately and assert the union of behaviours.

        // Arm A: ghost with checkpoint -> Recovered.
        let tv_a = TestValkey::setup().await;
        let base_a = "/mixed-a";
        let sid_a = format!("mix-a-{}", uuid::Uuid::new_v4());
        seed_ghost(&tv_a, base_a, "mix-a", &sid_a).await;
        let in_mem_a = Arc::new(InMemoryCheckpointStore::new());
        seed_checkpoint(
            &in_mem_a,
            &sample_session_checkpoint(&sid_a, "mix-a", base_a, "mix-user", 1),
        )
        .await;
        let metrics_a = make_metrics();
        let hook_a = build_hook(&tv_a, in_mem_a, metrics_a.clone(), true);
        let session_store_a = SessionStore::new(tv_a.client().clone(), tv_a.key_schema().clone());
        let result_a = session_store_a.list(base_a).await.expect("list a");
        let recovered_a = reconcile_ghost_members(
            &hook_a,
            &session_store_a,
            tv_a.client(),
            tv_a.key_schema(),
            base_a,
            result_a.ghosts,
        )
        .await;
        assert_eq!(recovered_a.len(), 1);
        assert_eq!(recovered_a[0].session_id, sid_a);

        // Arm B: truly-orphan ghost (no checkpoint, hook enabled).
        let tv_b = TestValkey::setup().await;
        let base_b = "/mixed-b";
        let sid_b = format!("mix-b-{}", uuid::Uuid::new_v4());
        seed_ghost(&tv_b, base_b, "mix-b", &sid_b).await;
        let metrics_b = make_metrics();
        let store_b: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook_b = build_hook(&tv_b, store_b, metrics_b.clone(), true);
        let session_store_b = SessionStore::new(tv_b.client().clone(), tv_b.key_schema().clone());
        let result_b = session_store_b.list(base_b).await.expect("list b");
        let recovered_b = reconcile_ghost_members(
            &hook_b,
            &session_store_b,
            tv_b.client(),
            tv_b.key_schema(),
            base_b,
            result_b.ghosts,
        )
        .await;
        assert!(recovered_b.is_empty());
        let path_index_b: Vec<String> = tv_b
            .client()
            .pool
            .smembers(tv_b.key_schema().path_index(base_b))
            .await
            .expect("SMEMBERS b");
        assert!(!path_index_b.contains(&sid_b), "no-checkpoint ghost SREM'd");

        // Arm C: transient-error ghost (store returns Transient).
        let tv_c = TestValkey::setup().await;
        let base_c = "/mixed-c";
        let sid_c = format!("mix-c-{}", uuid::Uuid::new_v4());
        seed_ghost(&tv_c, base_c, "mix-c", &sid_c).await;
        let metrics_c = make_metrics();
        let transient_store: Arc<dyn CheckpointStore> = Arc::new(ClassifiedErrorStore {
            err: || CheckpointError::Transient {
                source: "synthetic transient".into(),
            },
        });
        let hook_c = build_hook(&tv_c, transient_store, metrics_c.clone(), true);
        let session_store_c = SessionStore::new(tv_c.client().clone(), tv_c.key_schema().clone());
        let result_c = session_store_c.list(base_c).await.expect("list c");
        let recovered_c = reconcile_ghost_members(
            &hook_c,
            &session_store_c,
            tv_c.client(),
            tv_c.key_schema(),
            base_c,
            result_c.ghosts,
        )
        .await;
        assert!(
            recovered_c.is_empty(),
            "transient -> excluded from response"
        );
        // Transient -> preserved (not SREM'd) so next walk retries.
        let path_index_c: Vec<String> = tv_c
            .client()
            .pool
            .smembers(tv_c.key_schema().path_index(base_c))
            .await
            .expect("SMEMBERS c");
        assert!(
            path_index_c.contains(&sid_c),
            "transient-error ghost MUST NOT be SREM'd; next walk retries"
        );
        assert_eq!(metric_value(&metrics_c, "error"), 1);
    }

    // ─── Test 6 — Empty path-index: list returns empty present + empty
    // ghosts; reconcile is a no-op. Pins the no-side-effect invariant
    // for the common "no sessions at this base_path" case.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_empty_base_path_yields_no_recovery_probes() {
        let tv = TestValkey::setup().await;
        let metrics = make_metrics();
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook = build_hook(&tv, store, metrics.clone(), true);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let result = session_store.list("/totally-empty").await.expect("list");
        assert!(result.present.is_empty());
        assert!(result.ghosts.is_empty());

        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            "/totally-empty",
            result.ghosts,
        )
        .await;
        assert!(recovered.is_empty());

        // No metric ticked, no admission entry created.
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    // ─── Test 7 — Permanent error: ghost preserved (not SREM'd), excluded
    // from response, metric ticks `error`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_permanent_error_skips_member_without_srem() {
        let tv = TestValkey::setup().await;
        let base_path = "/permanent-stage";
        let sid = format!("perm-{}", uuid::Uuid::new_v4());

        seed_ghost(&tv, base_path, "perm-name", &sid).await;

        let metrics = make_metrics();
        let perm_store: Arc<dyn CheckpointStore> = Arc::new(ClassifiedErrorStore {
            err: || CheckpointError::Permanent {
                source: "synthetic permanent".into(),
            },
        });
        let hook = build_hook(&tv, perm_store, metrics.clone(), true);
        let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let result = session_store.list(base_path).await.expect("list");
        let recovered = reconcile_ghost_members(
            &hook,
            &session_store,
            tv.client(),
            tv.key_schema(),
            base_path,
            result.ghosts,
        )
        .await;
        assert!(recovered.is_empty(), "permanent -> excluded from response");

        let path_index: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_index(base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index.contains(&sid),
            "permanent-error ghost MUST NOT be SREM'd; defer to orphan sweep"
        );
        let path_names: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_names_index(base_path))
            .await
            .expect("SMEMBERS path_names_index");
        assert!(
            path_names.contains(&KeySchema::path_name_member("perm-name", &sid)),
            "permanent-error ghost composite MUST NOT be SREM'd"
        );

        assert_eq!(metric_value(&metrics, "error"), 1);
    }

    // ─── Test 8 — Recovered -> Ok(None) race -> SREM as orphan.
    //
    // Direct unit-test of [`srem_ghost_member`] — the dispatch target for
    // the `Recovered -> Ok(None)` arm of [`reconcile_ghost_members`] (the
    // race where DeleteSession commits between the recovery's
    // session_meta write and the reconcile re-read; see helper match-arm
    // at the call site). The arm is not reachable end-to-end in a unit
    // test without a fault-injection seam on `SessionRecoveryHook`
    // (admission + apply run synchronously inside the helper, so we
    // cannot interpose a DEL between them). The next-best discipline is
    // to assert the SREM target is correct: both indices SREM the
    // composite + bare session_id, and the call is idempotent under
    // the "either index may already be cleaned" benign race.
    //
    // The full end-to-end race is covered by Task 7's `t2.fgj.7v.o7`
    // integration test, per Subtask body's deferral.
    //
    // Inspection invariants pinned by this test:
    //   1. `path_names_index` SREM uses the `name\0session_id` composite.
    //   2. `path_index` SREM uses the bare `session_id`.
    //   3. The helper is tolerant of either index being absent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn srem_ghost_member_clears_both_indices_and_is_idempotent() {
        let tv = TestValkey::setup().await;
        let base_path = "/race-stage";
        let sid = format!("race-{}", uuid::Uuid::new_v4());

        // Seed the ghost in both indices.
        seed_ghost(&tv, base_path, "race-name", &sid).await;

        let composite = KeySchema::path_name_member("race-name", &sid);
        let ghost = GhostMember {
            name: "race-name".to_string(),
            sid: sid.clone(),
            composite_member: composite.clone(),
        };
        let path_names_key = tv.key_schema().path_names_index(base_path);
        let path_index_key = tv.key_schema().path_index(base_path);

        // Drive the SREM helper directly.
        super::srem_ghost_member(tv.client(), &path_names_key, &path_index_key, &ghost).await;

        // Both indices SREM'd of the ghost.
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index");
        assert!(
            !path_index_members.contains(&sid),
            "path_index member MUST be removed by srem_ghost_member"
        );
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names_index");
        assert!(
            !path_names_members.contains(&composite),
            "path_names_index composite MUST be removed by srem_ghost_member"
        );

        // Idempotency: calling again is safe (either index may already
        // be cleaned by a peer sweeper — preserves the "log-and-continue"
        // contract on missing-member SREM).
        super::srem_ghost_member(tv.client(), &path_names_key, &path_index_key, &ghost).await;
    }
}
