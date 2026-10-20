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

// JoinSession bidirectional streaming handler (SDD §13.2).
//
// The handler (`join_session_inner`) reads and validates the header
// (session_id, user_application), decides whether the join is under the
// client keepalive (`JoinKeepalive::arm`), resolves the session and checks
// read permission, resolves the user's identity, mints the connection_id
// (UUID4), refuses the join with `UNAVAILABLE` once graceful shutdown's paced
// stream ending has begun, and spawns the join task on
// `presence_cleanup_tracker`.
//
// The join task (`run_join_session`) runs the SDD SR-4 steps, in this order:
//  1. enroll for this join's shutdown token and check it; an exit here has
//     written nothing, skips teardown and ends `UNAVAILABLE`;
//  2. add the session to the session index (`le:presence:active_sessions`),
//     before the connection is a member;
//  3. join the local presence router and go on only once this replica's
//     Valkey subscription to the session's channel is confirmed: the first
//     local joiner sends `SUBSCRIBE`, and joiners arriving while it is in
//     flight wait for its result (a failure fails them all, `UNAVAILABLE`);
//  4. read the store's clock, take the membership snapshot (one `SMEMBERS`,
//     which step 10 replays), then `storage::presence::register_connection`:
//     lease, info hash and membership in one script, which also publishes
//     `joined`, so a member never exists without its lease. The snapshot
//     comes before the register so that every event carrying this join's
//     own connection_id was published after the snapshot (see
//     `ReplayDelivered`);
//  5. add the session to the index again (the re-add the index-removal
//     protocol relies on);
//  6. acquire the session-cancel token, which can run a bounded session
//     recovery pass, raced against shutdown;
//  7. spawn the lease keeper, which renews the lease from here to teardown
//     and re-registers the connection under the same connection_id if it is
//     evicted, then stop if the session or this join's shutdown token is
//     already cancelled;
//  9. send `SessionConnected`, carrying the join's keepalive interval (0
//     unless the join is armed);
// 10. replay the membership from step 4's snapshot: one member-info pipeline
//     per page of up to `REPLAY_PAGE` other members and one `PeerJoined` per
//     valid entry, every pipeline await and send racing shutdown, the
//     session token and a client disconnect. A completed replay hands the
//     bidi loop the set of ids it delivered (`ReplayDelivered`, the SDD's D);
// 11. observe join latency (only after a completed replay);
// 12. bidirectional loop: publish the client's `PeerMessage`s (a keepalive or
//     any other request type is ignored, never broadcast), forward presence
//     events, and, for a join armed for the client keepalive, end it once its
//     request stream has carried no bytes for three intervals, counted from
//     the loop's start. Because step 3's router subscribe precedes step 4's
//     snapshot, a peer that registers between them reaches this join both
//     live and in the replay; the loop suppresses the live `PeerJoined` of a
//     peer still in D. D ends, freed, at the first of this join's own first
//     live event, a receiver lag, `REPLAY_DUPLICATE_WINDOW_LIMIT`, or the
//     loop's end. Each member is delivered once whenever every event reaches
//     the join and its own event arrives within the limit (see
//     `ReplayDelivered` for where a duplicate remains possible). A receiver
//     lag that lost a peer's `joined` or `left` ends the stream `UNAVAILABLE`
//     after teardown, so the client rejoins with a fresh participant list;
//     one that lost only peer messages, or this join's own events (which the
//     self-filter discards), is counted and the loop goes on;
// 14. teardown (every exit after step 1): stop the keeper, then
//     `storage::presence::unregister_connection` (publishes `left` and arms
//     the `:gone` marker that fences late writes), remove the session from
//     the index if it is now empty, and release the router subscription.
//
// The stream's ending is decided by its cause: graceful shutdown ends it
// `UNAVAILABLE "server shutting down"`, which the client retries. The
// handler's check reads the shutdown handle's `begun` token, cancelled when
// the paced stream ending starts; step 1 enrolls for this join's own shutdown
// token and checks it, and steps 6, 7, 10 and 12 read that token, which the
// paced ending cancels in this join's batch (born cancelled for a join that
// enrolls after the ending started). This replica has stopped
// accepting connections (readiness `NOT_SERVING`, GOAWAY sent), so where the
// rejoin lands is the ingress's choice. A session deletion seen through the
// session token (steps 7, 10 and 12) ends it `OK` (a session found missing
// from the store with `DeleteSession`'s deletion marker is cancelled as a
// deletion). A client that went away is seen at the `SessionConnected` send
// (step 9), or while the replay still has a pipeline or send to make, and
// ends it `CANCELLED`, which it never receives; otherwise the bidi loop sees
// it and ends it `OK` — but only when the stream or the connection is torn
// down at this replica. Behind an HTTP/2-terminating proxy a client that
// vanished without closing its stream is never seen that way: only an armed
// join's keepalive deadline ends it, `UNAVAILABLE` with the retry hint
// `CLIENT_KEEPALIVE_TIMEOUT` (logged at INFO, counted on
// `live_edit_presence_keepalive_timeouts_total`); a join that did not opt in
// is held until the proxy resets its stream or a connection fails. A session
// token cancelled because the session was
// found missing from the live store with no `DeleteSession` seen and no
// marker (the `session_tokens` sweeper, or another stream's setup) ends it
// `UNAVAILABLE`, so the rejoin can recover a session a checkpoint restores.
// A bidi loop whose receiver lagged and lost a peer's `joined` or `left`
// ends it `UNAVAILABLE` too, so the rejoin's replay restores the participant
// list.
// A session this join's own step 6 finds gone, in `acquire_session_cancel`,
// even after that helper's recovery pass, ends it `NOT_FOUND`, after
// teardown. A failure status is sent after teardown, into
// a response-channel slot reserved when the channel was created, so sending
// it never waits on the client.
//
// There is no step 8: `joined` and `left` are published by the membership
// scripts, never from Rust. The only Rust-side presence publish left here is
// the client's `PeerMessage`.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::time::{Duration, Instant};

use fred::interfaces::{ClientLike, SetsInterface};
use futures::Stream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

use super::auth_helpers::{check_read_permission, extract_token, BasePathSource};
use super::proto::{
    JoinSessionRequest, JoinSessionResponse, PeerJoined, PeerLeft, PeerMessage, SessionConnected,
};
use super::request_end::RequestBodyProgress;
use super::rpc_log::{self, RpcLogContext, RpcStage};
use super::service::LiveEditService;
use super::session::{session_absent_status, SessionCancel, SessionCancelCause};
use super::transport::server_shutting_down;
use crate::error::{
    refusal_reason, RefusalDetail, ServiceError, CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY,
};
use crate::observability::metrics::{
    SharedMetrics, ENDPOINT_JOIN_SESSION, PRESENCE_EVENT_JOINED, PRESENCE_EVENT_LEFT,
    PRESENCE_EVENT_MESSAGE,
};
use crate::presence::{
    publish_presence_event, subscribe_presence_channel, unsubscribe_presence_channel, ChannelJoin,
    ConnectedUser, LocalPresenceRouter, PresenceEvent, PresenceReceiver, PresenceRecvError,
};
use crate::storage::client::{
    map_fred_err_for_grpc, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_PUBSUB,
};
use crate::storage::keys::KeySchema;
use crate::storage::presence::{
    add_session_to_index, read_member_infos, refresh_connection, register_connection,
    remove_session_from_index, store_time_ms, unregister_connection, RefreshOutcome,
    RegisterOutcome, REPLAY_PAGE,
};

type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Handle the JoinSession bidirectional streaming RPC.
///
/// **NotFound recovery (SDD SR-2.7).** The `get_base_path` lookup in step 2
/// funnels through
/// [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache-miss + session-absent triggers lazy on-demand
/// session-state recovery via the attached
/// [`SessionRecoveryHook`](crate::checkpoint::SessionRecoveryHook). A
/// genuinely non-existent / unrecoverable session still surfaces `NotFound`
/// to the client; no behavior change at this handler.
///
/// **Failure logging (SDD §17.2.1).** A setup failure — anything before the
/// response stream opens — returns `Err` through this wrapper, which emits
/// the one completion line exactly as the single-response handlers do. A
/// failure after the stream opened is logged by the spawned task, from a
/// clone of the same context, before presence teardown runs (see
/// [`run_join_session`]). `user_name` and `user_application` are never logged.
///
/// **Join latency.** The clock for `live_edit_join_latency_seconds` starts here,
/// at handler entry, so it includes the wait for the client's header message;
/// it is observed once `SessionConnected` and the membership replay have been
/// enqueued (see [`run_join_session_inner`]). A join that fails, or that a
/// shutdown or session cancel ends, before its replay completes is not
/// observed.
pub async fn handle_join_session(
    svc: &LiveEditService,
    request: Request<Streaming<JoinSessionRequest>>,
) -> Result<Response<BoxStream<JoinSessionResponse>>, Status> {
    let started = Instant::now();
    let mut ctx = RpcLogContext::for_request("JoinSession", &request);
    let result = join_session_inner(svc, request, &mut ctx, started).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn join_session_inner(
    svc: &LiveEditService,
    request: Request<Streaming<JoinSessionRequest>>,
    log: &mut RpcLogContext,
    started: Instant,
) -> Result<Response<BoxStream<JoinSessionResponse>>, Status> {
    // ── Extract bearer token ─────────────────────────────────────────────────
    let token = extract_token(&request)?;

    // The request body's progress record (`request_end`), which the client
    // keepalive deadline reads. `live_service_server` inserts it on every
    // request; only a handler-level test request built without it lacks it,
    // and such a join is never armed and is told interval 0.
    let body_progress = request.extensions().get::<RequestBodyProgress>().cloned();

    let mut stream = request.into_inner();

    // ── Step 1: Read header ──────────────────────────────────────────────────
    // A receive error here is tonic's own `Status` — `CANCELLED` for a client
    // that went away, an `INTERNAL` with no fault source for a body that is not
    // valid protobuf — and passes through `?` unchanged, so the wrapper logs it
    // at `stage=header` by the level table.
    let header = match stream.message().await? {
        Some(msg) => match msg.request_type {
            Some(super::proto::join_session_request::RequestType::Header(h)) => h,
            _ => {
                return Err(ServiceError::InvalidArgument(
                    "first message must be a JoinSessionHeader".to_owned(),
                )
                .into());
            }
        },
        None => {
            return Err(
                ServiceError::InvalidArgument("stream closed before header".to_owned()).into(),
            );
        }
    };

    let session_id = header.session_id;
    let user_application = header.user_application;
    let keepalive = JoinKeepalive::arm(
        header.client_keepalive,
        svc.presence_keepalive_interval_ms,
        body_progress,
    );

    // b.39x H: full session_id validation (length cap + control-char
    // rejection) before storage / auth / tracing. Replaces the pre-fix
    // `is_empty()`-only check.
    crate::domain::validation::validate_session_id(&session_id)?;
    log.session(&session_id);

    // b.39x H: validate `user_application` before it reaches the
    // `presence_user` HSET, the `PeerJoined` broadcast, or the
    // membership-replay payload. Pre-fix a multi-MB / control-char
    // user_application poisoned every peer's view; the validator caps
    // length at 256 bytes and rejects control characters.
    crate::domain::validation::validate_user_application(&user_application)?;

    // ── Step 2: Session exists + auth ────────────────────────────────────────
    log.at(RpcStage::Auth);
    let base_path = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await?;
    log.base_path(&base_path);
    check_read_permission(svc, &token, &base_path, BasePathSource::Session).await?;

    // ── Step 3: Resolve user identity ────────────────────────────────────────
    // Transient backend outages flow through `From<AuthError> for
    // ServiceError` so they surface as `UNAVAILABLE` (b.ohw H-grpc-1)
    // rather than `INTERNAL`.
    let user_info = svc
        .userinfo_client
        .get_current_user(&token)
        .await
        .map_err(ServiceError::from)?;

    let user_name = user_info.display_name;

    // ── Step 4: Generate connection_id ───────────────────────────────────────
    let connection_id = uuid::Uuid::new_v4().to_string();
    log.connection_id(&connection_id);

    // ── Shutdown check (SDD SR-4.7) ──────────────────────────────────────────
    // Once the paced ending has begun, refuse the join before any task or
    // store write. `begun` is cancelled after the drain, just behind the
    // server's GOAWAY, and a connection can still open a stream until its
    // GOAWAY takes effect, so a join can arrive after it. Kit retries
    // `UNAVAILABLE` (this replica has stopped accepting connections, so the
    // retry reaches another), whereas an empty `OK` stream is terminal for it.
    // The wrapper logs this once, at `stage=admission` so it does not read as
    // an auth-backend outage, and at INFO because the context is marked
    // shutdown-caused (SDD §17.2.1).
    log.at(RpcStage::Admission);
    if svc.shutdown.begun().is_cancelled() {
        log.shutdown_caused();
        return Err(server_shutting_down());
    }

    // ── Spawn the bidirectional task ─────────────────────────────────────────
    // **33, not 32, and the extra slot is reserved immediately**, as
    // `SubscribeReadLayer` does: the delivery capacity is unchanged at 32, and
    // the 33rd is the final status's. Sending the status into it is
    // synchronous and cannot wait, so a join whose client has stopped reading
    // still finishes its task — teardown included — instead of parking on a
    // full channel after teardown and holding the shutdown sequence's presence
    // drain until its budget runs out.
    let (tx, rx) = mpsc::channel::<Result<JoinSessionResponse, Status>>(33);

    // Taken on a channel that is empty by construction, so it resolves without
    // waiting. `reserve_owned` fails only when the receiver has been dropped,
    // and `rx` is still owned by this function — the arm is unreachable, and
    // is written as an error rather than an `expect` so an unreachable case can
    // never take a connection down.
    let status_permit = tx.clone().reserve_owned().await.map_err(|_| {
        ServiceError::Internal(
            "JoinSession could not reserve its final-status slot on a freshly created \
             response channel"
                .to_owned(),
        )
    })?;

    let svc = svc.clone();
    let session_id_owned = session_id.clone();
    let connection_id_owned = connection_id.clone();
    let user_name_owned = user_name.clone();
    let user_application_owned = user_application.clone();
    let mut tail_log = log.clone();
    tail_log.at(RpcStage::Stream);

    // b.w29 item 4: spawn through `presence_cleanup_tracker` so the
    // shutdown sequence in `main.rs` can `wait()` for every in-flight
    // `run_join_session` to finish its teardown — whose unregister script
    // publishes `left` to peers — before Valkey `quit()` tears down the
    // pools. The connection's lease keeper runs on the same tracker. Without
    // the tracker the bounded shutdown sequence dropped these futures
    // mid-cleanup and peers saw phantom members until the lease TTL expired.
    svc.presence_cleanup_tracker.clone().spawn(async move {
        let result = run_join_session(
            &svc,
            &tx,
            stream,
            &session_id_owned,
            &connection_id_owned,
            &user_name_owned,
            &user_application_owned,
            keepalive,
            &mut tail_log,
            started,
        )
        .await;

        // The final status, after teardown, into the slot reserved for it
        // above. Synchronous: it cannot wait on a client that has stopped
        // reading. A receiver that has gone discards it, which is harmless —
        // there is nobody left to tell. On `Ok` the permit is dropped and the
        // stream ends when `tx` does.
        if let Err(status) = result {
            let _sender = status_permit.send(Err(status));
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Response::new(Box::pin(stream)))
}

/// Parameters for the join session loop, grouped to avoid too_many_arguments.
struct JoinCtx<'a> {
    svc: &'a LiveEditService,
    tx: &'a mpsc::Sender<Result<JoinSessionResponse, Status>>,
    session_id: &'a str,
    connection_id: &'a str,
    user_name: &'a str,
    user_application: &'a str,
    /// Handler entry, the start of `live_edit_join_latency_seconds`.
    started: Instant,
    /// This join's enrolled child of the service shutdown token
    /// ([`GracefulShutdown::enroll_stream`](super::graceful_shutdown::GracefulShutdown::enroll_stream)
    /// on [`LiveEditService::shutdown`]), taken at the task-start check (SDD
    /// SR-4.7). It is cancelled in its batch of the paced ending, or with the
    /// service token.
    ///
    /// Every per-join shutdown check — the task-start check, the race around
    /// `acquire_session_cancel` (SR-4 step 6), the check after it (SR-4 step
    /// 7), the lease keeper's `select!`
    /// (the keeper holds a clone of this token), the membership replay's
    /// pipeline awaits and sends, and the bidi loop's arm and its event send —
    /// reads this token, never a shared one. Polling a token locks that token's own mutex and
    /// `Notify` waiter list. On a process-wide token those locks would be
    /// shared by every join on the replica, on every loop iteration; on the
    /// child they are this join's alone, and the parent is locked only when
    /// the child is created and dropped.
    shutdown: CancellationToken,
    /// This join's client keepalive, `None` when the join is not armed: its
    /// header did not set `client_keepalive`, the keepalive is configured
    /// off, or the request carries no progress record. `SessionConnected`
    /// advertises its interval (0 when `None`), and the bidi loop enforces its
    /// deadline.
    keepalive: Option<JoinKeepalive>,
}

/// How many keepalive intervals an armed join may go without data from its
/// client before the bidi loop ends it. Client contract: the proto's
/// `JoinSession` comment promises three.
pub const CLIENT_KEEPALIVE_DEADLINE_INTERVALS: u32 = 3;

/// An armed join's client keepalive (SDD §13.2): the interval it is told and
/// the deadline the bidi loop holds it to.
struct JoinKeepalive {
    /// `SessionConnected.keepalive_interval_ms`; never 0.
    interval_ms: u64,
    /// [`CLIENT_KEEPALIVE_DEADLINE_INTERVALS`] × the interval.
    deadline: Duration,
    /// When the join's request body last carried bytes.
    progress: RequestBodyProgress,
}

impl JoinKeepalive {
    /// The keepalive of a join, armed only when its header set
    /// `client_keepalive`, the configured interval is not 0 (off), and the
    /// request carries its body's progress record. Otherwise `None`: the join
    /// is told interval 0 and is never ended for silence.
    fn arm(
        client_keepalive: bool,
        interval_ms: u64,
        progress: Option<RequestBodyProgress>,
    ) -> Option<Self> {
        if !client_keepalive || interval_ms == 0 {
            return None;
        }
        Some(Self {
            interval_ms,
            deadline: Duration::from_millis(interval_ms)
                .saturating_mul(CLIENT_KEEPALIVE_DEADLINE_INTERVALS),
            progress: progress?,
        })
    }
}

/// Run the JoinSession flow: registration, replay, bidi loop, teardown.
///
/// **Teardown runs on every exit after the shutdown re-check** (SDD SR-4.1),
/// whatever [`run_join_session_inner`] returned, including after a register
/// that errored — a timed-out script may still have applied. [`teardown`] is
/// the only cleanup path; no step cleans up after itself.
///
/// **Endings by cause.** Graceful shutdown ends the join `UNAVAILABLE "server
/// shutting down"` wherever it is observed — this re-check, the race around
/// SR-4 step 6's `acquire_session_cancel`, the check after registration, the
/// membership replay's pipeline awaits and sends, the bidirectional loop and
/// its event send — so Kit, which never retries `OK`, rejoins; where the
/// rejoin lands is the ingress's choice, since this replica has stopped
/// accepting connections. The session token — observed at the check after
/// registration, in the replay, in the loop and in its event send — ends it by
/// the cancel's cause ([`end_on_session_cancel`]): an explicit `DeleteSession`
/// ends it `OK` (PRD §3.1.2) — including one this replica learns of only from
/// the session's deletion marker; a session found missing from the live store
/// with no `DeleteSession` seen and no marker ends it `UNAVAILABLE`, which the
/// client retries. A client that went away is seen at the `SessionConnected`
/// send, or while the replay still has a pipeline or send to make, and ends it
/// `CANCELLED`; otherwise the loop sees it and ends it `OK` — when the stream
/// or connection is torn down at this replica. A join armed for the client
/// keepalive whose client sent no data for three intervals is ended by the
/// loop, or by its event send, `UNAVAILABLE` with the retry hint
/// `CLIENT_KEEPALIVE_TIMEOUT` ([`client_keepalive_timeout_status`]); this is
/// what ends a client that vanished behind a proxy. A session gone at
/// SR-4 step 6, where this join's own `acquire_session_cancel` finds it
/// missing and its recovery pass cannot restore it, ends it `NOT_FOUND`, after
/// teardown; a recovery pass that exceeds its budget ends it `UNAVAILABLE`. Shutdown is
/// checked first wherever both can be observed. The caller sends the status
/// only once this function has returned, so after teardown.
///
/// **The failure line is logged here, at the point the status is decided,
/// before teardown.** Teardown awaits Valkey, and the shutdown sequence bounds
/// how long it waits for this task, so a line logged after teardown could be
/// lost with the task. Every `Err` is a failure the client sees; a shutdown's
/// is logged at INFO, because its arm marks `log` shutdown-caused, a keepalive
/// timeout's at INFO, because its ending marks `log` with the client-keepalive
/// marker, and a missing session's at WARN.
#[allow(clippy::too_many_arguments)]
async fn run_join_session(
    svc: &LiveEditService,
    tx: &mpsc::Sender<Result<JoinSessionResponse, Status>>,
    mut client_stream: Streaming<JoinSessionRequest>,
    session_id: &str,
    connection_id: &str,
    user_name: &str,
    user_application: &str,
    keepalive: Option<JoinKeepalive>,
    log: &mut RpcLogContext,
    started: Instant,
) -> Result<(), Status> {
    // ── Shutdown re-check (SDD SR-4 step 1, SR-4.7) ──────────────────────────
    // Enroll for this join's shutdown token, then check it: closes the race
    // between the handler's check and this task starting. Once the paced
    // ending has started the token is born cancelled, so no start between the
    // two checks is missed; an enrolled one is cancelled in its batch. The
    // enrollment is held until this function returns, after teardown.
    // Nothing has been written yet, so there is nothing to tear down
    // (SR-4.1); the stream ends `UNAVAILABLE`, as a shutdown ends a join
    // everywhere else, and the line is logged here because no teardown
    // follows to log it.
    let (shutdown, _shutdown_enrollment) = svc.shutdown.enroll_stream();
    if shutdown.is_cancelled() {
        log.shutdown_caused();
        let status = server_shutting_down();
        rpc_log::log_rpc_failure(log, &status);
        return Err(status);
    }

    let ctx = JoinCtx {
        svc,
        tx,
        session_id,
        connection_id,
        user_name,
        user_application,
        started,
        shutdown,
        keepalive,
    };

    let mut presence = JoinPresence::default();
    let result = run_join_session_inner(&ctx, &mut client_stream, &mut presence, log).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(log, status);
    }

    // ── SR-4 step 14: Teardown (every exit from here) ────────────────────────
    teardown(&ctx, presence).await;

    result
}

/// What the join task has done to the store and the local router, read by
/// [`teardown`] to undo exactly that (SDD SR-4.3).
#[derive(Default)]
struct JoinPresence {
    /// Set as soon as the router join returns (SR-4 step 3), before the
    /// channel subscription is confirmed. The join took a router count however
    /// the subscription then goes, so teardown releases it exactly once when
    /// this is set — and never when it is not, so an exit before step 3 never
    /// decrements another local connection's router count.
    subscribed: bool,
    /// Set immediately before `register_connection` is sent (SR-4 step 4), so
    /// it is true even when the register then errors: a timed-out script may
    /// still have applied, and teardown must unregister it.
    register_issued: bool,
    /// The lease keeper, `None` until SR-4 step 7 spawns it.
    keeper: Option<KeeperHandle>,
}

/// The join task's side of a running lease keeper.
struct KeeperHandle {
    /// The keeper's teardown token. Cancelled once, at the start of
    /// [`teardown`].
    teardown: CancellationToken,
    /// Awaited by [`teardown`], bounded, before the connection is unregistered.
    handle: JoinHandle<()>,
}

/// Inner body of JoinSession — SR-4 steps 2 to 12. Records in `presence` what
/// it did, for the teardown that follows it on every exit.
///
/// `log` is the task's completion context; the shutdown arms mark it
/// shutdown-caused before they return, so the caller logs their `UNAVAILABLE`
/// at INFO.
async fn run_join_session_inner(
    ctx: &JoinCtx<'_>,
    client_stream: &mut Streaming<JoinSessionRequest>,
    presence: &mut JoinPresence,
    log: &mut RpcLogContext,
) -> Result<(), Status> {
    let valkey = &ctx.svc.valkey;
    let keys = &ctx.svc.key_schema;
    // The storage wrappers count a timeout on the origin they are given, and
    // `map_fred_err_for_grpc` counts on its own, so a wrapper whose error is
    // mapped for the client gets an origin that counts nothing (SR-1.7). It
    // still notes an `-OOM` on the client's signal (b.pa1; noting twice is
    // harmless).
    let mapped = FredErrOrigin::metrics(None, POOL_LABEL_DATA, valkey.store_oom_signal());
    let map_data_err = |op: &str, e: fred::error::Error| -> Status {
        map_fred_err_for_grpc(FredErrOrigin::client(valkey, POOL_LABEL_DATA), op, e).into()
    };

    // ── SR-4 step 2: Index the session, before the connection is a member ────
    // A crash after this leaves at most an index entry for a session with no
    // members, which the sweeper removes — never a member the sweeper cannot
    // find.
    add_session_to_index(valkey, mapped, keys, ctx.session_id)
        .await
        .map_err(|e| map_data_err("SADD presence active_sessions", e))?;

    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.join_registration_seam_for_test.as_ref() {
        seam.after_index_add.arrive().await;
    }

    // ── SR-4 step 3: Join the local presence router; channel subscribed ──────
    // Before the register below, so this connection is subscribed before
    // peers can learn of it from the register script's `joined` and react.
    // The router count is taken here whatever follows, so `subscribed` is set
    // at once and teardown releases the count exactly once. The receiver is
    // tied to this connection, so a lag over this join's own `joined` (or
    // any own event), which the self-filter would discard, is not counted as
    // membership loss and does not end the stream for a resync.
    let (mut event_rx, channel) = ctx
        .svc
        .presence_router
        .join_channel(ctx.session_id, Some(ctx.connection_id));
    presence.subscribed = true;

    // This join goes on only once the replica's Valkey subscription to the
    // session's channel is confirmed, so no announcement published from here
    // on can miss it (see `LocalPresenceRouter`).
    match channel {
        ChannelJoin::Subscribed => {}
        // Another local join's `SUBSCRIBE` is in flight: share its result. A
        // failure fails this join too, `UNAVAILABLE` so the client retries;
        // the error was counted where the subscribing join received it.
        ChannelJoin::Wait(wait) => {
            if let Err(reason) = wait.wait().await {
                return Err(ServiceError::Unavailable(format!(
                    "SUBSCRIBE presence channel failed in a concurrent join: {reason}"
                ))
                .into());
            }
        }
        // This join subscribes, for itself and any join that arrives while
        // its `SUBSCRIBE` is in flight.
        ChannelJoin::Subscribe(mut subscribe) => {
            // After the last local leaver's in-flight `UNSUBSCRIBE`, if any,
            // so the channel ends subscribed.
            subscribe.wait_for_prior_unsubscribe().await;

            #[cfg(any(test, feature = "test-fault-injection"))]
            let injected = match ctx.svc.presence_channel_seam_for_test.as_ref() {
                Some(seam) => {
                    seam.before_subscribe.arrive().await;
                    seam.take_injected_subscribe_failure()
                }
                None => None,
            };
            #[cfg(not(any(test, feature = "test-fault-injection")))]
            let injected: Option<fred::error::Error> = None;

            let subscribed = match injected {
                Some(e) => Err(e),
                None => {
                    subscribe_presence_channel(
                        &ctx.svc.valkey.subscriber,
                        &ctx.svc.key_schema,
                        ctx.session_id,
                    )
                    .await
                }
            };
            match subscribed {
                Ok(()) => subscribe.succeeded(),
                Err(e) => {
                    subscribe.failed(&e.to_string());
                    // b.fd9: transient Valkey blips during pub/sub subscribe
                    // surface as `Code::Unavailable` (retriable) rather than
                    // `Internal`, by the uniform classifier.
                    return Err(map_fred_err_for_grpc(
                        FredErrOrigin::client(&ctx.svc.valkey, POOL_LABEL_PUBSUB),
                        "SUBSCRIBE presence channel",
                        e,
                    )
                    .into());
                }
            }
        }
    }

    // ── SR-4 step 4: Clock, snapshot, register — lease, info hash, membership,
    //    `joined` ───────────────────────────────────────────────────────────
    // The register's deadline is one lease TTL past the store's clock, read on
    // the `{sid}` node now, while this connection's teardown cannot have
    // started: teardown runs only after this function returns. That is the
    // obligation on `register_connection` that lets its teardown marker and
    // its deadline together refuse this register however late it reaches the
    // store. A failed clock read leaves the register unissued.
    let store_now_ms = store_time_ms(valkey, mapped, keys, ctx.session_id)
        .await
        .map_err(|e| map_data_err("TIME presence register deadline", e))?;

    // Inside the subscribe-to-snapshot window: subscribed (step 3), not yet
    // snapshotted, not a member.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.presence_replay_seam_for_test.as_ref() {
        seam.before_snapshot.arrive().await;
    }

    // The membership snapshot step 10 replays: one `SMEMBERS`, an atomic,
    // duplicate-free set of member ids. Taken before the register, and its
    // reply awaited before the register is issued (both on the `{sid}`
    // primary), so every event carrying this join's connection_id — its own
    // `joined` first — is published after the snapshot. That makes this
    // join's first own live event the point after which no announcement the
    // snapshot already reflects can still arrive, where the bidi loop ends
    // its duplicate suppression (see `ReplayDelivered`). Not raced: it is one
    // command, and a shutdown begun meanwhile is seen at step 6 or 7. A
    // failure ends the join here, before `SessionConnected`, with the
    // register never issued.
    let mut member_ids: Vec<String> = valkey
        .pool
        .next()
        .smembers(keys.presence_members(ctx.session_id))
        .await
        .map_err(|e| map_data_err("SMEMBERS presence", e))?;
    // Defensive only: this join is not yet a member, so the snapshot cannot
    // hold its own id (minted fresh by `join_session_inner`). Kept so that the
    // replay never sends a joiner its own `PeerJoined`, whatever the store
    // holds.
    member_ids.retain(|cid| cid != ctx.connection_id);

    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.join_registration_seam_for_test.as_ref() {
        seam.before_register.arrive().await;
    }

    let not_after_ms =
        store_now_ms.saturating_add(ctx.svc.presence_lease_ttl_secs.saturating_mul(1000));
    let joined_payload = PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: ctx.connection_id.to_string(),
        user_name: ctx.user_name.to_string(),
        user_application: ctx.user_application.to_string(),
    })
    .encode();
    presence.register_issued = true;
    let registered = register_connection(
        valkey,
        mapped,
        keys,
        ctx.session_id,
        ctx.connection_id,
        ctx.user_name,
        ctx.user_application,
        ctx.svc.presence_lease_ttl_secs,
        not_after_ms,
        &joined_payload,
    )
    .await
    .map_err(|e| map_data_err("EVALSHA presence_register", e))?;
    match registered {
        // `added: false` cannot come from a first register of a fresh
        // connection_id, but can from fred re-sending a register whose first
        // attempt applied and lost its reply; the connection is registered
        // and announced once either way.
        RegisterOutcome::Registered { .. } => {}
        // A teardown marker cannot exist for a connection_id this join just
        // minted. The deadline passes only if the register reached the store
        // more than one lease TTL after the clock read above (a re-send after
        // a long reconnect, say). Both are internal errors; teardown follows,
        // as for any failure after step 2.
        RegisterOutcome::Gone => {
            return Err(ServiceError::Internal(
                "presence register refused: the new connection already has a teardown marker"
                    .to_owned(),
            )
            .into());
        }
        RegisterOutcome::Expired => {
            return Err(ServiceError::Internal(
                "presence register refused: it reached the store after its deadline".to_owned(),
            )
            .into());
        }
    }

    // ── SR-4 step 5: Re-add the session to the index ─────────────────────────
    // Issued only after the register completed. An index removal racing this
    // join either finds the member when it rechecks the member set, or ran
    // before this re-add, which then restores the entry
    // (`remove_session_from_index`).
    add_session_to_index(valkey, mapped, keys, ctx.session_id)
        .await
        .map_err(|e| map_data_err("SADD presence active_sessions", e))?;

    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.join_registration_seam_for_test.as_ref() {
        seam.before_session_cancel.arrive().await;
    }

    // ── SR-4 step 6: Acquire the session-cancel token ────────────────────────
    // Get-or-insert the per-session CancellationToken. CreateSession only
    // populates it on the replica that handled the create RPC, so any
    // JoinSession that lands on a different replica must insert its own
    // entry — otherwise a subsequent DeleteSession (which fans out via
    // `le:session:deleted`) would fail to terminate this stream (b.uue).
    // The helper also re-checks session existence after the insert to close
    // the race where the delete fan-out arrived before our insert; a session
    // missing from the store is resolved once more through the recovery hook,
    // and one still missing ends the join with `NotFound`.
    //
    // That recovery pass is the slow case: a checkpoint read, serialized per
    // session by the hook's admission mutex. It runs with the connection
    // registered (step 4) and the lease keeper not yet started (step 7), so
    // nothing renews the lease meanwhile. The helper bounds it by
    // `config::session_recovery_pass_budget`, which is derived from the lease
    // TTL and refresh interval so the lease outlives it (without that bound
    // only the hook's `checkpoint_recovery_fetch_budget`, 60 s by default,
    // would apply). And it races the shutdown token, checked first as
    // everywhere in this join, so a graceful shutdown during a slow pass ends
    // the join `UNAVAILABLE` and runs its teardown, as for any failure after
    // step 2, rather than holding the task past the shutdown budget. Dropping
    // the helper's future at either `.await` cancels nothing (see its rustdoc).
    let session_cancel = tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => {
            log.shutdown_caused();
            return Err(server_shutting_down());
        }
        acquired = super::session::acquire_session_cancel(
            &ctx.svc.session_tokens,
            &ctx.svc.session_store,
            &ctx.svc.session_base_path_cache,
            crate::config::session_recovery_pass_budget(
                ctx.svc.presence_refresh_interval_secs,
                ctx.svc.presence_lease_ttl_secs,
            ),
            ctx.session_id,
        ) => acquired?,
    };
    // This join's child of the session token: what the lease keeper, the
    // membership replay and the bidi loop poll, for the reason
    // `JoinCtx::shutdown` is a child of the service token. Polling a token
    // locks that token's own mutex and `Notify` waiter list. The session token
    // is shared by every local join in the session, so polling it directly
    // would take one lock shared by all of them on every replay step, loop
    // iteration and keeper step; the child's locks are this join's alone.
    // Cancelling the session token — which happens only once the session is
    // gone from the store, never at shutdown — cancels the child, and a child
    // made from an already-cancelled token is born cancelled. The child shares
    // the token's cause, which decides how the join ends
    // (`end_on_session_cancel`).
    let join_session_cancel = session_cancel.child();

    // ── SR-4 step 7: Spawn the lease keeper, then check the tokens ───────────
    // The keeper renews the lease from here until teardown stops it, through
    // the replay and the bidi loop alike, and re-registers the connection if
    // it is evicted meanwhile. It is handed all three tokens now, so a session
    // delete or a shutdown stops it, rejoins included, even before this task
    // notices.
    let keeper_teardown = CancellationToken::new();
    let handle = spawn_lease_keeper(
        &ctx.svc.presence_cleanup_tracker,
        LeaseKeeper {
            valkey: valkey.clone(),
            key_schema: keys.clone(),
            metrics: ctx.svc.metrics.clone(),
            presence_router: std::sync::Arc::clone(&ctx.svc.presence_router),
            session_id: ctx.session_id.to_owned(),
            connection_id: ctx.connection_id.to_owned(),
            user_name: ctx.user_name.to_owned(),
            user_application: ctx.user_application.to_owned(),
            lease_ttl_secs: ctx.svc.presence_lease_ttl_secs,
            refresh_interval: Duration::from_secs(ctx.svc.presence_refresh_interval_secs),
            teardown: keeper_teardown.clone(),
            session_cancel: join_session_cancel.token().clone(),
            shutdown: ctx.shutdown.clone(),
            #[cfg(any(test, feature = "test-fault-injection"))]
            seam: ctx.svc.lease_keeper_seam_for_test.clone(),
        },
    );
    presence.keeper = Some(KeeperHandle {
        teardown: keeper_teardown,
        handle,
    });
    // A shutdown begun, or a session token cancelled, while this join was
    // registering ends it here, before `SessionConnected`, straight to
    // teardown — each as it ends a join in its replay and its loop: a shutdown
    // `UNAVAILABLE`, checked first, then the session token by its cause (a
    // deletion `OK`, a session missing from the live store `UNAVAILABLE`). The
    // shutdown read is this join's child token (see `JoinCtx::shutdown`). From
    // here on the replay's pipeline awaits and sends and the bidi loop watch
    // both tokens.
    if ctx.shutdown.is_cancelled() {
        log.shutdown_caused();
        return Err(server_shutting_down());
    }
    if session_cancel.is_cancelled() {
        return end_on_session_cancel(&session_cancel);
    }

    // ── SR-4 step 9: Send SessionConnected ───────────────────────────────────
    // A non-zero keepalive interval means exactly "this join is under the
    // keepalive deadline"; every other join is told 0 and sends none.
    let connected_msg = JoinSessionResponse {
        response_type: Some(
            super::proto::join_session_response::ResponseType::Connected(SessionConnected {
                connection_id: ctx.connection_id.to_string(),
                keepalive_interval_ms: ctx.keepalive.as_ref().map_or(0, |k| k.interval_ms),
            }),
        ),
    };
    ctx.tx
        .send(Ok(connected_msg))
        .await
        .map_err(|_| Status::cancelled("client disconnected"))?;

    // ── SR-4 step 10: Membership replay ──────────────────────────────────────
    // Replays step 4's snapshot, which it consumes. Paged, and every pipeline
    // await and send races shutdown, the session
    // token and a client disconnect (SR-4.4). The keeper keeps the lease alive however
    // long it takes. A replay ended by a token skips steps 11–12 and ends the
    // join as the bidi loop's arm for that token does: a shutdown marks `log`
    // shutdown-caused and ends `UNAVAILABLE`, logged at INFO before teardown;
    // a session cancel ends by its cause (a deletion `OK`, with no failure
    // line). A client disconnect or a store error returns its status, logged
    // once by `run_join_session`. Teardown follows every exit. Only a
    // completed replay yields its delivered-id set D, which the bidi loop
    // takes over; on every other exit there is no D.
    let delivered = match replay_membership(ctx, member_ids, &join_session_cancel).await? {
        ReplayOutcome::Complete(delivered) => delivered,
        ReplayOutcome::ShuttingDown => {
            log.shutdown_caused();
            return Err(server_shutting_down());
        }
        ReplayOutcome::SessionCancelled => return end_on_session_cancel(&join_session_cancel),
    };

    // ── SR-4 step 11 ─────────────────────────────────────────────────────────
    // The joiner now holds its first complete state: SessionConnected and the
    // full membership are enqueued. A completed replay only — every failure,
    // and every replay a token ended, returns before this line.
    ctx.svc
        .metrics
        .observe_join_latency(ENDPOINT_JOIN_SESSION, ctx.started.elapsed());

    // ── SR-4 step 12: Bidirectional loop ─────────────────────────────────────
    // The loop owns D, and ends it at the first of this join's own first live
    // event, a lag, the duplicate-window limit, or the loop's end.
    run_bidi_loop(
        ctx,
        client_stream,
        &mut event_rx,
        delivered,
        &join_session_cancel,
        log,
    )
    .await
}

/// How long [`teardown`] waits for the lease keeper beyond one store command
/// timeout. A keeper step already under way runs to completion, so the wait
/// covers a step's refresh plus scheduling slack. A step that takes longer —
/// a refresh retried by fred, a renewal's index re-add on top, or a rejoin's
/// clock read, register and index re-add after the refresh — outlives the
/// wait; teardown logs that and goes on. A late refresh or rejoin register is
/// then fenced by the teardown marker and the register's deadline, and a late
/// index re-add can leave only an index entry for an empty session, which the
/// sweeper removes.
const KEEPER_STOP_SLACK: Duration = Duration::from_secs(1);

/// A connection's lease keeper (SDD SR-4.2): what it owns.
///
/// Spawned at SR-4 step 7 on `presence_cleanup_tracker`, so the shutdown drain
/// waits for it. Every refresh interval it runs one [`LeaseKeeper::step`]. It
/// stops when any of its three tokens is cancelled, or when a refresh or a
/// rejoin register reports the connection torn down. Besides the presence
/// sweeper, only the keeper and the join task write this connection's
/// presence, and only the keeper ever adds the connection back after an
/// eviction.
struct LeaseKeeper {
    valkey: ValkeyClient,
    key_schema: KeySchema,
    /// Counts a failed step on `live_edit_presence_lease_refresh_failures_total`,
    /// a rejoin on `live_edit_presence_rejoins_total` and, through the storage
    /// wrappers, a command timeout once on `valkey_errors_total{pool="data"}`.
    metrics: SharedMetrics,
    /// This replica's presence router. Its
    /// [`LocalPresenceRouter::index_renewal_due`] gates the renewal's index
    /// re-add to once per session per replica per refresh interval (SDD
    /// SR-7.2b).
    presence_router: std::sync::Arc<LocalPresenceRouter>,
    session_id: String,
    connection_id: String,
    /// The connection's user name and application exactly as the join
    /// registered them, so a rejoin writes the same info hash and its `joined`
    /// carries the same values. A rejoin never substitutes a value: Kit drops
    /// a `PeerJoined` whose `user_application` is empty, so a rejoin with a
    /// blanked application would leave the connection hidden from Kit peers.
    user_name: String,
    user_application: String,
    lease_ttl_secs: u64,
    refresh_interval: Duration,
    /// The connection's teardown token; [`teardown`] cancels it first.
    teardown: CancellationToken,
    /// The join's child of the session's cancel token, cancelled with it when
    /// the session is gone. The keeper only stops on it, so it holds the bare
    /// token, not the cause. A child, so that the keeper's polls lock this
    /// join's token alone, not the one every local join in the session shares.
    session_cancel: CancellationToken,
    /// The join's enrolled child of the service shutdown token
    /// (`JoinCtx::shutdown`).
    shutdown: CancellationToken,
    #[cfg(any(test, feature = "test-fault-injection"))]
    seam: Option<std::sync::Arc<LeaseKeeperSeam>>,
}

/// How long after spawn a lease keeper's first renewal comes: uniform in
/// [`interval`/2, `interval`] (see [`LeaseKeeper::run`]).
fn first_renewal_delay(interval: Duration) -> Duration {
    use rand::Rng;
    let nanos = u64::try_from(interval.as_nanos()).unwrap_or(u64::MAX);
    Duration::from_nanos(rand::rng().random_range(nanos / 2..=nanos))
}

/// Spawn `keeper` on `tracker` (the service's `presence_cleanup_tracker`).
fn spawn_lease_keeper(
    tracker: &tokio_util::task::TaskTracker,
    keeper: LeaseKeeper,
) -> JoinHandle<()> {
    tracker.spawn(keeper.run())
}

impl LeaseKeeper {
    /// The keeper loop. Registration has just set the lease, so there is no
    /// renewal at spawn. The first renewal comes after a random delay, uniform
    /// in [interval/2, interval] ([`first_renewal_delay`]), and every later one
    /// an interval after the one before.
    ///
    /// The jitter spreads the renewals of connections that joined together,
    /// after a rolling restart or a mass reconnect, over the whole interval.
    /// Without it they would keep their join-time phase, `Delay` preserving
    /// it, and hit the `{sid}` shards as a periodic burst indefinitely.
    ///
    /// The first renewal still lands inside the lease: it comes at most one
    /// interval after spawn, and configuration requires the TTL to be at least
    /// twice the interval (`Config::validate_presence_lease_budgets`).
    ///
    /// Missed ticks are delayed, not burst, so a slow step is followed by one
    /// step, not a catch-up run. The tokens are checked only between steps: a
    /// step, once started, runs to completion, so a store command is never
    /// abandoned mid-flight.
    async fn run(self) {
        let stopped = async {
            tokio::select! {
                () = self.teardown.cancelled() => {}
                () = self.session_cancel.cancelled() => {}
                () = self.shutdown.cancelled() => {}
            }
        };
        tokio::pin!(stopped);
        let first = tokio::time::Instant::now() + first_renewal_delay(self.refresh_interval);
        let mut interval = tokio::time::interval_at(first, self.refresh_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = &mut stopped => return,
                tick = interval.tick() => {
                    if self.step(tick).await.is_break() {
                        return;
                    }
                }
            }
        }
    }

    /// One keeper step, run at interval tick `tick` (the instant the tick was
    /// scheduled for): renew the lease if the connection is still fully
    /// registered, and re-register it if it is not. `Break` stops the keeper.
    ///
    /// - `Renewed` → re-add the session to the session index, but only if
    ///   [`LocalPresenceRouter::index_renewal_due`] says so: at most once per
    ///   session per replica per refresh interval (SDD SR-7.2b). An entry lost
    ///   by any other route is healed within one refresh interval while the
    ///   connection whose renewal last re-added the session keeps renewing,
    ///   otherwise within two (see `index_renewal_due`). One keeper's ticks
    ///   are at least one interval apart, so a session whose only local
    ///   connection this is is re-added on every renewal.
    /// - `Gone` → the connection has been torn down: stop.
    /// - `Evicted` / `InfoMissing` → [`Self::rejoin`]: re-register the
    ///   connection under the same `connection_id`, name and application,
    ///   unless its teardown (or a session delete or shutdown) has begun.
    /// - A store error → counted and logged; the next tick tries again.
    ///
    /// A step counts at most one failure on
    /// `live_edit_presence_lease_refresh_failures_total`, whichever command
    /// failed.
    async fn step(&self, tick: tokio::time::Instant) -> ControlFlow<()> {
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = self.seam.as_ref() {
            seam.before_refresh.arrive().await;
        }

        let origin = FredErrOrigin::metrics(
            Some(&self.metrics),
            POOL_LABEL_DATA,
            self.valkey.store_oom_signal(),
        );
        let refreshed = refresh_connection(
            &self.valkey,
            origin,
            &self.key_schema,
            &self.session_id,
            &self.connection_id,
            self.lease_ttl_secs,
        )
        .await;
        match refreshed {
            Ok(RefreshOutcome::Renewed) => {
                // Only this re-add is gated. The re-adds after a rejoin or an
                // info repair (in `rejoin`) and the join's own re-adds are not.
                if self.presence_router.index_renewal_due(
                    &self.session_id,
                    tick,
                    self.refresh_interval,
                ) {
                    self.reindex(origin, "renewing the lease").await;
                }
                ControlFlow::Continue(())
            }
            Ok(RefreshOutcome::Gone) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    "presence lease keeper: connection torn down; stopping"
                );
                ControlFlow::Break(())
            }
            Ok(outcome @ (RefreshOutcome::Evicted | RefreshOutcome::InfoMissing)) => {
                self.rejoin(origin, outcome).await
            }
            Err(e) => {
                self.metrics.inc_presence_lease_refresh_failure();
                tracing::warn!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    error = %e,
                    "presence lease keeper: lease refresh failed; retrying at the next tick"
                );
                ControlFlow::Continue(())
            }
        }
    }

    /// Self-rejoin (SDD SR-4.2 path A, keeper step 2): the refresh found the
    /// connection evicted (`Evicted`) or missing its info hash
    /// (`InfoMissing`), so re-register it with
    /// [`register_connection`] under the same `connection_id`, user name and
    /// application. Only this keeper ever adds its connection back, and only
    /// before the connection's teardown begins.
    ///
    /// **Check, read the clock, check again, register.** The register's
    /// deadline is the store's clock plus one lease TTL, and it fences a
    /// register that arrives late only if the clock read ran before
    /// teardown's unregister (the caller obligation on `register_connection`).
    /// So the stop check runs both before the clock read and after it, and the
    /// register is issued only if neither saw the teardown token cancelled.
    /// Teardown cancels that token before it unregisters, so a read that
    /// passed the second check preceded the unregister. Both checks also stop
    /// on a cancelled session or shutdown token: the keeper would stop at its
    /// next `select!` anyway, and teardown follows either, so a rejoin then
    /// would only announce a `joined` that is immediately undone. Stopping
    /// writes nothing.
    ///
    /// The register script publishes `joined` iff it re-added the membership,
    /// in the same execution. So an eviction, which the evicting script
    /// announced as `left`, is followed by exactly one `joined`, while an info
    /// repair of a current member is announced to no one. A rejoin that runs
    /// just before teardown is announced, and teardown's unregister then
    /// announces `left`: peers see `left`, `joined`, `left`, in store order.
    ///
    /// By register result:
    /// - `Registered { added: true }` → count one rejoin, then re-add the
    ///   session to the index (the connection may be the session's only
    ///   member, and the session may have left the index with it).
    /// - `Registered { added: false }` → the connection was already a member:
    ///   an info repair after `InfoMissing`, or, after `Evicted`, a member an
    ///   earlier timed-out rejoin register had re-added (and announced) by
    ///   the time this one ran. Nothing was published and no rejoin is
    ///   counted. The register renewed the lease, so the session is re-added
    ///   to the index, ungated: only a plain renewal's re-add goes through
    ///   [`LocalPresenceRouter::index_renewal_due`].
    /// - `Gone` → the teardown marker exists: stop.
    /// - `Expired` → the register reached the store after its deadline and
    ///   wrote nothing: count a failure; the next tick reads a fresh clock.
    /// - A store error (the clock read or the register) → count a failure;
    ///   the next tick tries again. A register that timed out but still
    ///   applies announces itself atomically in the same script, so peers
    ///   still see exactly one `joined`:
    ///   - If it applied before the next step's refresh, that refresh sees
    ///     `Renewed`.
    ///   - If it applies only after that refresh, which is likely because a
    ///     step that ran past its command timeout is followed at once by the
    ///     next (`MissedTickBehavior::Delay`), the refresh sees `Evicted`. That
    ///     step then issues a second rejoin register, which re-adds, announces
    ///     and counts the rejoin, and the late register lands as
    ///     `Registered { added: false }`, publishing nothing.
    ///
    ///   A rejoin whose re-adding register lost its reply is never counted on
    ///   `live_edit_presence_rejoins_total`: the keeper never saw its
    ///   `added: true`.
    async fn rejoin(&self, origin: FredErrOrigin<'_>, found: RefreshOutcome) -> ControlFlow<()> {
        if self.stopping() {
            return self.stop_before_rejoin(found);
        }

        let store_now_ms =
            match store_time_ms(&self.valkey, origin, &self.key_schema, &self.session_id).await {
                Ok(now_ms) => now_ms,
                Err(e) => {
                    self.metrics.inc_presence_lease_refresh_failure();
                    tracing::warn!(
                        session_id = %self.session_id,
                        connection_id = %self.connection_id,
                        found = ?found,
                        error = %e,
                        "presence lease keeper: failed to read the store clock for a rejoin; \
                         retrying at the next tick"
                    );
                    return ControlFlow::Continue(());
                }
            };

        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = self.seam.as_ref() {
            seam.before_rejoin_register.arrive().await;
        }

        if self.stopping() {
            return self.stop_before_rejoin(found);
        }

        let not_after_ms = store_now_ms.saturating_add(self.lease_ttl_secs.saturating_mul(1000));
        let joined_payload = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: self.connection_id.clone(),
            user_name: self.user_name.clone(),
            user_application: self.user_application.clone(),
        })
        .encode();
        let registered = register_connection(
            &self.valkey,
            origin,
            &self.key_schema,
            &self.session_id,
            &self.connection_id,
            &self.user_name,
            &self.user_application,
            self.lease_ttl_secs,
            not_after_ms,
            &joined_payload,
        )
        .await;
        match registered {
            Ok(RegisterOutcome::Registered { added }) => {
                if added {
                    self.metrics.inc_presence_rejoin();
                }
                tracing::warn!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    found = ?found,
                    announced = added,
                    "presence lease keeper: connection was no longer fully registered; \
                     re-registered it"
                );
                self.reindex(origin, "re-registering the connection").await;
                ControlFlow::Continue(())
            }
            Ok(RegisterOutcome::Gone) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    "presence lease keeper: rejoin refused, connection torn down; stopping"
                );
                ControlFlow::Break(())
            }
            Ok(RegisterOutcome::Expired) => {
                self.metrics.inc_presence_lease_refresh_failure();
                tracing::warn!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    found = ?found,
                    "presence lease keeper: rejoin register reached the store after its \
                     deadline and wrote nothing; retrying at the next tick"
                );
                ControlFlow::Continue(())
            }
            Err(e) => {
                self.metrics.inc_presence_lease_refresh_failure();
                tracing::warn!(
                    session_id = %self.session_id,
                    connection_id = %self.connection_id,
                    found = ?found,
                    error = %e,
                    "presence lease keeper: rejoin register failed; retrying at the next tick"
                );
                ControlFlow::Continue(())
            }
        }
    }

    /// Whether any of the keeper's three tokens is cancelled: the connection's
    /// teardown has begun, its session is gone (deleted, or found missing from
    /// the live store), or the service is shutting down.
    fn stopping(&self) -> bool {
        self.teardown.is_cancelled()
            || self.session_cancel.is_cancelled()
            || self.shutdown.is_cancelled()
    }

    /// Stop the keeper at one of [`Self::rejoin`]'s checks, having written
    /// nothing.
    fn stop_before_rejoin(&self, found: RefreshOutcome) -> ControlFlow<()> {
        tracing::debug!(
            session_id = %self.session_id,
            connection_id = %self.connection_id,
            found = ?found,
            "presence lease keeper: connection is stopping; not re-registering it"
        );
        ControlFlow::Break(())
    }

    /// Re-add the session to the session index after the keeper wrote the
    /// lease (SDD SR-4.5). A failure counts one failed step. It is retried by
    /// the next re-add this replica makes for the session. A failed renewal
    /// re-add still counts as the session's re-add for its interval (see
    /// [`LocalPresenceRouter::index_renewal_due`]), so the retry comes within
    /// one refresh interval if this keeper's next refresh renews, and
    /// otherwise from a sibling connection's renewal within two intervals.
    /// `after` names what the step just did, for the log.
    async fn reindex(&self, origin: FredErrOrigin<'_>, after: &str) {
        if let Err(e) =
            add_session_to_index(&self.valkey, origin, &self.key_schema, &self.session_id).await
        {
            self.metrics.inc_presence_lease_refresh_failure();
            tracing::warn!(
                session_id = %self.session_id,
                connection_id = %self.connection_id,
                error = %e,
                "presence lease keeper: failed to re-add the session to the session index \
                 after {after}; retrying at the session's next renewal re-add \
                 (within one or two refresh intervals)"
            );
        }
    }
}

/// How a membership replay that did not fail ended (SDD SR-4.4).
#[derive(Debug)]
enum ReplayOutcome {
    /// Every other member of the snapshot with a valid info hash was enqueued
    /// as `PeerJoined`; the set carries exactly those connection ids (D). The
    /// join goes on to observe its latency and run the bidi loop, which takes
    /// D over.
    Complete(ReplayDelivered),
    /// The join's shutdown token was cancelled. The join ends `UNAVAILABLE
    /// "server shutting down"` after teardown, as a shutdown in the loop does.
    ShuttingDown,
    /// The join's session token was cancelled. The join ends by the cancel's
    /// cause after teardown ([`end_on_session_cancel`]), as a session cancel in
    /// the loop does.
    SessionCancelled,
}

/// Replay the membership to the joiner, every other member as one
/// `PeerJoined` (SDD SR-4 step 10, SR-4.4).
///
/// `member_ids` is step 4's `SMEMBERS` snapshot — atomic and duplicate-free,
/// taken before this join registered, with the joiner's own id already
/// dropped — which the replay consumes. The ids are read in pages of up to
/// [`REPLAY_PAGE`]: one [`read_member_infos`] pipeline per page, then one send
/// per valid entry. An entry the read returns as `None` (an empty or malformed
/// info hash, a member being torn down) is skipped, and the replay goes on. So
/// at most one pipeline is in flight, and at most one page of info is held
/// between sends.
///
/// A completed replay returns, with [`ReplayOutcome::Complete`], the set D of
/// the connection ids it actually enqueued as `PeerJoined`: a skipped `None`
/// entry is not in D, and neither is an entry whose send a cancel or a
/// disconnect ended. The snapshot was taken after this join's router subscribe
/// (SR-4 step 3), so a peer that registered in between is both in the
/// snapshot and, live, in the join's presence receiver; [`run_bidi_loop`]
/// uses D to suppress that live duplicate (see [`ReplayDelivered`]). D is
/// built from the snapshot's own strings: a UUID id is stored as its 128-bit
/// value and its string moves into the `PeerJoined`, so nothing is copied for
/// it; only a non-UUID id is copied, once, for the `PeerJoined`.
///
/// Every pipeline await and every send races the join's endings, in the bidi
/// loop's order ([`race_replay_endings`]): shutdown, then the session token,
/// then a client disconnect. A cancel therefore stops the replay before its
/// next pipeline or send.
///
/// `session_cancel` is the join's child of the session token, the one the bidi
/// loop watches. Nothing here cleans up: every exit, failures included, goes
/// to the join's [`teardown`]. A replay that does not complete drops the ids
/// it had collected, so there is no D on those exits.
///
/// Errors: a store error, mapped by `map_fred_err_for_grpc`, and
/// `CANCELLED "client disconnected"` for a client that went away.
async fn replay_membership(
    ctx: &JoinCtx<'_>,
    member_ids: Vec<String>,
    session_cancel: &SessionCancel,
) -> Result<ReplayOutcome, Status> {
    let result = replay_member_pages(ctx, member_ids, session_cancel).await;
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.presence_replay_seam_for_test.as_ref() {
        seam.record_result(&result);
    }
    result
}

/// The body of [`replay_membership`].
async fn replay_member_pages(
    ctx: &JoinCtx<'_>,
    mut member_ids: Vec<String>,
    session_cancel: &SessionCancel,
) -> Result<ReplayOutcome, Status> {
    let valkey = &ctx.svc.valkey;
    let keys = &ctx.svc.key_schema;
    let map_data_err = |op: &str, e: fred::error::Error| -> Status {
        map_fred_err_for_grpc(FredErrOrigin::client(valkey, POOL_LABEL_DATA), op, e).into()
    };

    #[cfg(any(test, feature = "test-fault-injection"))]
    let seam = ctx.svc.presence_replay_seam_for_test.as_deref();
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = seam {
        seam.before_first_send.arrive().await;
    }

    // D, sized for every snapshot entry being delivered; trimmed at the end
    // when enough were skipped to free buckets.
    let mut delivered = ReplayDelivered::with_capacity(member_ids.len());
    // This replay's pipelines in flight, for the seam's high-water mark.
    #[cfg(any(test, feature = "test-fault-injection"))]
    let in_flight = std::sync::atomic::AtomicUsize::new(0);

    // `map_data_err` counts a timeout itself, so the read counts nothing
    // (SR-1.7); a read is never refused `-OOM`, so the signal is moot here.
    let mapped = FredErrOrigin::metrics(None, POOL_LABEL_DATA, valkey.store_oom_signal());
    // `chunks_mut`, so a delivered id's string can be moved out of the
    // snapshot once its page has been read.
    for (page_index, page) in member_ids.chunks_mut(REPLAY_PAGE).enumerate() {
        // After the previous page's sends, before this page's pipeline.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = seam.filter(|_| page_index > 0) {
            seam.between_pages.arrive().await;
        }
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let _ = page_index;

        let read = async {
            // Inside the raced future, so a pipeline a cancel arm drops before
            // its first poll is never counted as issued.
            #[cfg(any(test, feature = "test-fault-injection"))]
            let _in_flight = seam.map(|seam| seam.pipeline_issued(page.len(), &in_flight));
            read_member_infos(valkey, mapped, keys, ctx.session_id, page).await
        };
        let infos = match race_replay_endings(ctx, session_cancel, read).await {
            ControlFlow::Continue(read) => {
                read.map_err(|e| map_data_err("HGETALL presence user info replay page", e))?
            }
            ControlFlow::Break(ending) => return ending,
        };

        for (cid, info) in page.iter_mut().zip(infos) {
            let Some((user_name, user_application)) = info else {
                continue;
            };
            // The page has been read, so its strings are free to move: a UUID
            // id's string goes into the `PeerJoined` and D will keep its
            // 128-bit value; a non-UUID id is copied for the `PeerJoined` and
            // its own string is D's key (`split_for_delivery`).
            let (connection_id, key) = ReplayDelivered::split_for_delivery(std::mem::take(cid));
            let peer_joined = JoinSessionResponse {
                response_type: Some(
                    super::proto::join_session_response::ResponseType::PeerJoined(PeerJoined {
                        connection_id,
                        user_name,
                        user_application,
                    }),
                ),
            };
            match race_replay_endings(ctx, session_cancel, ctx.tx.send(Ok(peer_joined))).await {
                // Delivered: enqueued to this join's stream. Only here does an
                // id enter D.
                ControlFlow::Continue(Ok(())) => delivered.record_delivered(key),
                ControlFlow::Continue(Err(_)) => {
                    return Err(Status::cancelled("client disconnected"));
                }
                ControlFlow::Break(ending) => return ending,
            }
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(seam) = seam {
                seam.peer_joined_sent();
            }
        }
    }

    delivered.shrink_to_fit();
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.presence_replay_seam_for_test.as_ref() {
        delivered.observe(std::sync::Arc::clone(seam), ctx.connection_id);
    }
    Ok(ReplayOutcome::Complete(delivered))
}

/// `cid` as a key of [`ReplayDelivered`]'s 128-bit set: `Some` only when `cid`
/// is exactly the lowercase hyphenated text of a UUID, the form
/// `join_session_inner` mints (`Uuid::new_v4().to_string()`). Any other
/// spelling, an upper-case or braced UUID included, is `None` and keyed by its
/// string. Each 128-bit value therefore stands for exactly one string, so a
/// lookup matches exactly the ids string equality would, in either set.
fn delivered_uuid_key(cid: &str) -> Option<u128> {
    if cid.len() != uuid::fmt::Hyphenated::LENGTH {
        return None;
    }
    let uuid = uuid::Uuid::try_parse(cid).ok()?;
    let mut canonical = [0u8; uuid::fmt::Hyphenated::LENGTH];
    let canonical: &str = uuid.hyphenated().encode_lower(&mut canonical);
    (canonical == cid).then(|| uuid.as_u128())
}

/// How long, from the start of a join's bidi loop, its replay-delivered set D
/// ([`ReplayDelivered`]) waits for the join's own first live event before it
/// ends anyway (SDD SR-4.4). Normally that event is already queued when the
/// replay finishes, so D ends within the loop's first few events; the limit
/// bounds D's lifetime when the event is delayed or lost upstream. It applies
/// only to a D that held ids when the loop started: an empty D allocates
/// nothing and never gains an id, so no timer is created for it. A
/// pre-snapshot `joined` arriving after the limit is forwarded as a duplicate
/// (R2 on [`ReplayDelivered`]), which the client's view absorbs. Tests shorten
/// it through `PresenceReplaySeam::set_duplicate_window_limit`.
pub const REPLAY_DUPLICATE_WINDOW_LIMIT: Duration = Duration::from_secs(60);

/// The set D of connection ids a completed membership replay delivered as
/// `PeerJoined` (SDD SR-4.4), owned by the join's bidi loop, which uses it to
/// deliver each other member to this joiner exactly once (PRD FR-3).
///
/// **Why.** The join's router subscribe (SR-4 step 3) precedes its `SMEMBERS`
/// snapshot (step 4, after the store-clock read and before the register), so
/// a peer X that registers between the two reaches the joiner twice: in the
/// replay, and as a live `joined X` from its presence receiver. So does an X
/// that registered earlier whose `joined` was still on its way to this
/// replica when the join's receiver was attached to an already-subscribed
/// local channel.
///
/// **The rule** ([`Self::suppresses`]), applied to each live event after the
/// self-filter, while D is live:
/// - a live `PeerJoined` for X ∈ D is suppressed — neither sent nor counted in
///   `live_presence_events_total` — and X leaves D;
/// - a live `PeerLeft` for X is forwarded, and X leaves D;
/// - every other event is forwarded.
///
/// **When D ends** ([`Self::end`]): at the first of
/// - (a) the join's own first live event — the first event of any type
///   carrying this join's connection_id, seen at the self-filter;
/// - (b) [`PresenceRecvError::Lagged`], any lag. A lag that lost a peer's
///   join or leave also ends the stream (resync), so a stream that continues
///   lost only `PeerMessage`s or its own events; D ends on that lag too, by
///   choice. Keeping D live after such a lag (the fan-out's membership-loss
///   flag would allow it) would change D's contract and its tests only to
///   avoid a duplicate `PeerJoined` that clients already ignore (R1);
///   ending D on every lag keeps D's correctness independent of that flag;
/// - (c) [`REPLAY_DUPLICATE_WINDOW_LIMIT`] after the bidi loop started — only
///   for a D that held ids when the loop started; an empty D allocates
///   nothing and never gains an id, so no timer is created for it and it
///   ends by (a), (b) or (d);
/// - (d) the bidi loop's end.
///
/// An ended D is freed and never re-created: nothing is suppressed for the
/// rest of the connection.
///
/// **Why (a) is the boundary.** The connection_id is minted fresh for this
/// join, so every event carrying it was published after the join's register,
/// its own `joined` first. The register is issued only after the `SMEMBERS`
/// reply arrived, both on the `{sid}` primary (no replica reads), so every
/// such event was published after the snapshot. A session's events leave one
/// origin node in execution order and every subscriber receives them in that
/// order (SDD SR-1.P, SR-1.M; §13.1), so by the time this join receives its
/// own first event it has received every event published before the
/// snapshot, and each has already gone through D.
///
/// **Safety.** X is in D only while the last X event delivered to this client
/// is the replay's `joined X`, with no `left X` delivered since. Every `joined`
/// for a connection carries the same payload (a keeper rejoin re-registers
/// with the same user name and application), so suppressing a `joined X`
/// while X ∈ D never changes what the client shows, under any upstream loss
/// or reordering.
///
/// **Exactness.** When every event reaches the join and its own first event
/// arrives within the limit, no member is delivered twice:
/// - before the boundary, a live `joined X` for X ∈ D is either an
///   announcement published before the snapshot, which the replay already
///   covered (suppressed), or follows a `left X` published after the
///   snapshot, which took X out of D first (forwarded). So a peer that
///   leaves through an in-script `left` (the unregister or eviction script)
///   and rejoins reaches the client as joined, left, joined, and the rejoin
///   is never suppressed;
/// - after the boundary, a `joined X` for X in the snapshot needs a `left X`
///   published after the snapshot, so forwarding it is correct.
///
/// **Where a duplicate `PeerJoined` is still possible.** Except in a
/// mixed-version fleet (R4), the duplicate is for a peer the client already
/// shows as present, so it does not change the client's view. A client
/// reconnect gets a new `connection_id` and is never a duplicate.
/// - R1: a joiner lag that lost only peer messages or its own events — D
///   ends (b) and the stream continues. (A lag that lost a peer's join or
///   leave ends the stream.)
/// - R2: the join's own event is delayed past setup, replay and
///   [`REPLAY_DUPLICATE_WINDOW_LIMIT`], and a pre-snapshot `joined` arrives
///   after D ended (c).
/// - R3: upstream loss of a `left X` the joiner never sees — a pub/sub
///   reconnect gap, the replica-wide relay's own lag (counted on
///   `live_edit_pubsub_relay_lagged_total{relay="presence"}`), cluster-bus
///   loss — followed, after D ended, by X's lease keeper re-registering it
///   under the same connection_id. Such a loss neither ends D, nor ends the
///   stream, nor moves `live_edit_presence_broadcast_dropped_total`.
/// - R4: a mixed-version fleet, where pre-feature replicas publish `joined`
///   from the channel-name slot's node rather than the `{sid}` node, so the
///   one-origin order does not hold across them. A pre-feature replica's
///   sweeper also removes a stale member in separate commands, and a keeper
///   rejoin landing inside that removal is published `joined` before the
///   sweeper's `left`: a joiner whose D has already ended gets the `joined X`
///   forwarded as a duplicate, then the late `left`, and ends showing a
///   present X as absent.
/// - R5: a `{sid}` slot migration or primary failover between the join's
///   snapshot and its register.
///
/// **Lifetime.** Built during the replay from the snapshot's own ids, and
/// handed over only by a completed replay ([`ReplayOutcome::Complete`]); a
/// replay that ends any other way drops it. While live it only shrinks — on a
/// suppression or a live `PeerLeft` — until it ends, freed, at its boundary.
/// Normally the join's own `joined` (published at step 4, before
/// `SessionConnected`) is already queued when the replay finishes, so D
/// outlives the replay only by the time the loop takes to drain the events
/// ahead of it.
///
/// **Memory.** Held only from the replay to the boundary; nothing after it.
/// At most n − 1 ids per joiner in a session of n members, and never more
/// than the replay delivered.
///
/// Per entry, on a 64-bit target with 16-byte control groups (x86-64 SSE2):
/// - a UUID id (every id this service mints) is a `u128` in a
///   `HashSet<u128>`: 16 B of value plus 1 control byte per bucket, with
///   the bucket count `next_power_of_two(⌊8n/7⌋)` for n entries
///   (hashbrown's 7/8 maximum load), so between 8/7 × 17 ≈ 19.4 B and
///   16/7 × 17 ≈ 38.9 B per entry, as requested from the allocator. At
///   10,000 entries that is 16,384 buckets and a request of
///   16,384 × 17 + 16 = 278,544 B, about 27.9 B per entry. The production
///   allocator, jemalloc (`src/allocator.rs`), serves a large request from
///   its size class (four per doubling), so that request takes the
///   327,680 B (320 KiB) class: about 32.8 B per entry of reserved address
///   space, while resident pages stay near the 278,544 B requested;
/// - any other id is a `String` in a `HashSet<String>`: a 24 B `String` and
///   1 control byte per bucket plus its own heap allocation (the id's length,
///   rounded up to the allocator's size class). Empty unless some member id is
///   not a canonical UUID, and an empty set allocates nothing.
///
/// Fixed per joiner: the two sets' inline headers and hashers (about 96 B),
/// plus one 16-byte trailing control group per allocated table.
struct ReplayDelivered {
    /// Ids in their canonical UUID form ([`delivered_uuid_key`]).
    uuids: HashSet<u128>,
    /// Every other id, by its string.
    others: HashSet<String>,
    /// Why D ended, `None` while it is live.
    ended: Option<DeliveredEnd>,
    /// Test-only observation of this D, attached when the replay completes.
    #[cfg(any(test, feature = "test-fault-injection"))]
    observer: Option<DeliveredSetObserver>,
}

/// The key a delivered id is recorded under in [`ReplayDelivered`], from
/// [`ReplayDelivered::split_for_delivery`].
enum DeliveredKey {
    /// A canonical UUID ([`delivered_uuid_key`]), kept as its 128-bit value.
    Uuid(u128),
    /// Any other id, kept as its own string.
    Other(String),
}

/// Why a [`ReplayDelivered`] ended before its bidi loop did. The loop's own
/// end, (d), simply drops it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeliveredEnd {
    /// (a) The join's own first live event arrived.
    OwnEvent,
    /// (b) The join's receiver lagged.
    Lag,
    /// (c) [`REPLAY_DUPLICATE_WINDOW_LIMIT`] elapsed first.
    TimeLimit,
}

impl std::fmt::Debug for ReplayDelivered {
    /// The size and state only: D can hold thousands of ids.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayDelivered")
            .field("len", &self.len())
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

impl ReplayDelivered {
    /// An empty, live D whose 128-bit set has room for `capacity` ids.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            uuids: HashSet::with_capacity(capacity),
            others: HashSet::new(),
            ended: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            observer: None,
        }
    }

    /// Whether D has not yet ended; see the type's docs.
    fn is_live(&self) -> bool {
        self.ended.is_none()
    }

    /// End D for `cause`, if it is still live: free its tables and suppress
    /// nothing for the rest of the connection. A later cause changes nothing;
    /// the first one is D's end.
    fn end(&mut self, cause: DeliveredEnd) {
        if !self.is_live() {
            return;
        }
        self.uuids = HashSet::new();
        self.others = HashSet::new();
        self.ended = Some(cause);
        #[cfg(any(test, feature = "test-fault-injection"))]
        self.observe_end(cause.into());
    }

    /// Split snapshot id `cid`, about to be delivered, into the string its
    /// `PeerJoined` carries and the key D records it under once that
    /// `PeerJoined` is enqueued ([`Self::record_delivered`]). The one place
    /// a delivered id's recording key is chosen, for the replay and for
    /// `replay_delivered_measure` alike. Lookups ([`Self::remove`]) key the
    /// id through the same [`delivered_uuid_key`], so the two must change
    /// together:
    /// - a canonical UUID ([`delivered_uuid_key`]): the string goes to the
    ///   `PeerJoined` whole and D keys the id by its 128-bit value, so nothing
    ///   is copied;
    /// - any other id: a copy goes to the `PeerJoined` and the id's own string
    ///   is D's key.
    fn split_for_delivery(cid: String) -> (String, DeliveredKey) {
        match delivered_uuid_key(&cid) {
            Some(uuid) => (cid, DeliveredKey::Uuid(uuid)),
            None => (cid.clone(), DeliveredKey::Other(cid)),
        }
    }

    /// Record a delivered id under the key [`Self::split_for_delivery`] gave
    /// it. Call it only once the id's `PeerJoined` has been enqueued: only a
    /// delivered id enters D.
    fn record_delivered(&mut self, key: DeliveredKey) {
        match key {
            DeliveredKey::Uuid(uuid) => self.uuids.insert(uuid),
            DeliveredKey::Other(cid) => self.others.insert(cid),
        };
    }

    /// Release the capacity reserved for snapshot entries the replay skipped.
    /// Reallocates only when that frees at least one bucket-count step.
    fn shrink_to_fit(&mut self) {
        self.uuids.shrink_to_fit();
    }

    fn len(&self) -> usize {
        self.uuids.len() + self.others.len()
    }

    /// Remove `cid` from D; whether it was there.
    fn remove(&mut self, cid: &str) -> bool {
        match delivered_uuid_key(cid) {
            Some(uuid) => self.uuids.remove(&uuid),
            None => self.others.remove(cid),
        }
    }

    /// Apply the rule (see the type's docs) to one live event that passed the
    /// self-filter: `true` when the event must be suppressed.
    fn suppresses(&mut self, event: &PresenceEvent) -> bool {
        // An ended D is empty, so once D has ended — or every id in it has
        // left or been suppressed — no event costs a lookup; a message never
        // does.
        if self.len() == 0 {
            return false;
        }
        match event {
            PresenceEvent::PeerJoined(user) => {
                let suppressed = self.remove(&user.connection_id);
                #[cfg(any(test, feature = "test-fault-injection"))]
                self.observe_change(suppressed);
                suppressed
            }
            PresenceEvent::PeerLeft { connection_id } => {
                self.remove(connection_id);
                #[cfg(any(test, feature = "test-fault-injection"))]
                self.observe_change(false);
                false
            }
            PresenceEvent::PeerMessage { .. } => false,
        }
    }
}

/// Race one membership-replay operation, a page's pipeline or a send, against
/// the join's endings (SDD SR-4.4), in the bidi loop's order:
///
/// 1. the join's shutdown token → [`ReplayOutcome::ShuttingDown`];
/// 2. the join's session token → [`ReplayOutcome::SessionCancelled`];
/// 3. the client gone (`tx.closed()`) → `CANCELLED "client disconnected"`;
/// 4. the operation, whose output is returned as `Continue`.
///
/// `biased`, so a replay that sees a shutdown and a deletion together ends as
/// a shutdown, which the client retries, and an already-cancelled token wins
/// before the operation is first polled. A send to a client that has stopped
/// reading, with the channel full, therefore cannot park the join past any
/// ending. Dropping the operation abandons a pipeline already sent; its reply
/// is discarded, and it only reads.
async fn race_replay_endings<T>(
    ctx: &JoinCtx<'_>,
    session_cancel: &SessionCancel,
    operation: impl std::future::Future<Output = T>,
) -> ControlFlow<Result<ReplayOutcome, Status>, T> {
    tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => ControlFlow::Break(Ok(ReplayOutcome::ShuttingDown)),
        () = session_cancel.cancelled() => {
            ControlFlow::Break(Ok(ReplayOutcome::SessionCancelled))
        }
        () = ctx.tx.closed() => {
            ControlFlow::Break(Err(Status::cancelled("client disconnected")))
        }
        output = operation => ControlFlow::Continue(output),
    }
}

/// How a join ends once it has observed its session token cancelled, by the
/// cancel's cause: `OK` for an explicit `DeleteSession` (PRD §3.1.2), whether
/// seen directly or through its deletion marker, and the retryable
/// [`session_absent_status`] for a session found missing from the live store
/// without that marker, so the client's rejoin can recover a session a
/// checkpoint restores. Every arm that observes the session token ends through here.
fn end_on_session_cancel(session_cancel: &SessionCancel) -> Result<(), Status> {
    match session_cancel.cause() {
        SessionCancelCause::Deleted => Ok(()),
        SessionCancelCause::AbsentFromStore => Err(session_absent_status()),
    }
}

/// The status a JoinSession stream ends with when it fell more than the
/// per-session presence capacity behind and lost a participant's join or
/// leave (the bidi loop's lag arm). `UNAVAILABLE`, so the client rejoins and
/// the rejoin's replay restores the participant list. Sent to the client, so
/// it names no internals.
pub fn presence_resync_status() -> Status {
    Status::unavailable(
        "this connection fell too far behind and missed a participant joining or leaving; \
         rejoin the session to restore the participant list",
    )
}

/// The status a JoinSession stream ends with when its client opted in to
/// keepalives and then sent no data for [`CLIENT_KEEPALIVE_DEADLINE_INTERVALS`]
/// intervals (the bidi loop's keepalive deadline). `UNAVAILABLE` carrying the
/// refusal detail [`refusal_reason::CLIENT_KEEPALIVE_TIMEOUT`] with a
/// `RetryInfo` drawn from [`CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY`] — a minimum
/// wait, the client keeping its own backoff — so a client that is still alive
/// rejoins. Sent to the client, so it names no internals.
pub fn client_keepalive_timeout_status() -> Status {
    RefusalDetail::new(
        refusal_reason::CLIENT_KEEPALIVE_TIMEOUT,
        Some(CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY),
    )
    .unavailable("no data from the client for three keepalive intervals; rejoin the session")
}

/// End a join at its client keepalive deadline: count it once on
/// `live_edit_presence_keepalive_timeouts_total`, mark `log` so the one
/// completion line `run_join_session` logs is INFO, and return
/// [`client_keepalive_timeout_status`]. Teardown, which publishes the one
/// `left` peers see, follows as for every exit.
fn end_on_keepalive_timeout(ctx: &JoinCtx<'_>, log: &mut RpcLogContext) -> Status {
    ctx.svc.metrics.inc_presence_keepalive_timeout();
    log.client_keepalive_timed_out();
    client_keepalive_timeout_status()
}

/// What the keepalive deadline decides when its timer fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeepaliveExpiry {
    /// No data from the client for the whole deadline: end the join.
    Expired,
    /// Data arrived since the timer was set: wait again, until this instant.
    RearmAt(tokio::time::Instant),
}

/// The keepalive rule, pure: a join whose bidi loop started at `loop_start`,
/// whose request body last carried bytes at `last_data` (`None` if never), has
/// expired at `now` once `deadline` has passed since the later of the two.
/// The loop's start counts as data, so neither the membership replay (during
/// which the stream is not read) nor the setup before it counts against the
/// deadline.
fn keepalive_expiry(
    loop_start: tokio::time::Instant,
    last_data: Option<tokio::time::Instant>,
    deadline: Duration,
    now: tokio::time::Instant,
) -> KeepaliveExpiry {
    let reference = last_data.map_or(loop_start, |at| at.max(loop_start));
    let expires_at = reference + deadline;
    if now >= expires_at {
        KeepaliveExpiry::Expired
    } else {
        KeepaliveExpiry::RearmAt(expires_at)
    }
}

/// An armed join's running keepalive deadline, owned by the bidi loop from
/// its start. The timer is re-armed lazily: it is set once per deadline, not
/// reset per frame, and when it fires [`keepalive_expiry`] either ends the
/// join or sets it again from the latest data.
struct KeepaliveDeadline {
    loop_start: tokio::time::Instant,
    deadline: Duration,
    progress: RequestBodyProgress,
    timer: Pin<Box<tokio::time::Sleep>>,
}

impl KeepaliveDeadline {
    /// Start the deadline now, at the bidi loop's start.
    fn start(keepalive: &JoinKeepalive) -> Self {
        let loop_start = tokio::time::Instant::now();
        Self {
            loop_start,
            deadline: keepalive.deadline,
            progress: keepalive.progress.clone(),
            timer: Box::pin(tokio::time::sleep_until(loop_start + keepalive.deadline)),
        }
    }

    /// Resolves once the client has sent no data for the whole deadline.
    /// Cancel-safe: all state is in `self`, so a `select!` may drop it and
    /// call it again.
    async fn expired(&mut self) {
        loop {
            self.timer.as_mut().await;
            match keepalive_expiry(
                self.loop_start,
                self.progress.last_data_at(),
                self.deadline,
                tokio::time::Instant::now(),
            ) {
                KeepaliveExpiry::Expired => return,
                KeepaliveExpiry::RearmAt(at) => self.timer.as_mut().reset(at),
            }
        }
    }
}

/// [`KeepaliveDeadline::expired`] for a join that has one; never resolves for
/// a join that does not.
async fn keepalive_expired(deadline: &mut Option<KeepaliveDeadline>) {
    match deadline {
        Some(deadline) => deadline.expired().await,
        None => std::future::pending().await,
    }
}

/// Handle one read of the client's request stream in the bidi loop: publish a
/// `Message` to the session's peers and ignore every other request type — a
/// keepalive, a duplicate header, or a type this server does not know.
/// `Break` when the client is gone (its stream ended or errored), which ends
/// the join `OK`. The keepalive deadline needs nothing from here: any bytes
/// the read consumed were stamped on the body's progress record.
async fn handle_client_message(
    ctx: &JoinCtx<'_>,
    client_msg: Result<Option<JoinSessionRequest>, Status>,
) -> ControlFlow<()> {
    match client_msg {
        Ok(Some(msg)) => {
            if let Some(super::proto::join_session_request::RequestType::Message(m)) =
                msg.request_type
            {
                // Publish PeerMessage to Valkey.
                let event = PresenceEvent::PeerMessage {
                    connection_id: ctx.connection_id.to_string(),
                    data: m.data,
                };
                if let Err(e) = publish_presence_event(
                    &ctx.svc.valkey.pool,
                    &ctx.svc.key_schema,
                    ctx.session_id,
                    &event,
                )
                .await
                {
                    FredErrOrigin::client(&ctx.svc.valkey, POOL_LABEL_DATA).observe_error(&e);
                    tracing::warn!(
                        session_id = %ctx.session_id,
                        connection_id = %ctx.connection_id,
                        error = %e,
                        "failed to publish PeerMessage"
                    );
                }
            }
            // Every other request type is ignored and never broadcast: a
            // keepalive, a duplicate header, or a type unknown to this server.
            ControlFlow::Continue(())
        }
        // Client stream closed gracefully, or the client disconnected
        // unexpectedly.
        Ok(None) | Err(_) => ControlFlow::Break(()),
    }
}

/// Bidirectional loop: read client messages and forward events.
///
/// `delivered` is the completed replay's D. The loop owns it and, while it is
/// live, applies [`ReplayDelivered`]'s rule to every live event that passes
/// the self-filter — suppressing a live `PeerJoined` the replay already
/// delivered. It ends D at the first of this join's own first live event, a
/// lag, or [`REPLAY_DUPLICATE_WINDOW_LIMIT`] from the loop's start (logged at
/// DEBUG), and drops it on every exit. The time limit exists only for a D that
/// holds ids when the loop starts: an empty D allocates nothing and can never
/// gain an id, so no timer is created for it, and it ends at the join's own
/// event, a lag, or the loop's end — never at the time limit.
///
/// `session_cancel` is the join's child of the session token (created at SR-4
/// step 6), so the per-iteration poll of its arm locks this join's token
/// alone, not the one every local join in the session shares. It shares the
/// session token's cause.
///
/// **Client keepalive (SDD §13.2).** For a join armed for it
/// (`ctx.keepalive`), the loop starts a [`KeepaliveDeadline`] of
/// [`CLIENT_KEEPALIVE_DEADLINE_INTERVALS`] × the interval at its own start,
/// so the replay never counts, and ends the join once its request stream has
/// carried no bytes for that long ([`keepalive_expiry`]; the bytes are read
/// off the body's [`RequestBodyProgress`], so a large message still arriving
/// counts). Its arm comes after the client-message arm, so it fires only when
/// the stream has just been read to `Pending`. The event send of an armed join
/// also races a read of the client's stream and the deadline, so a client
/// that stops reading in a busy session is still ended, and one that keeps
/// sending while its responses are back-pressured is not. An unarmed join
/// has neither.
///
/// Returns `Err` for a shutdown — `UNAVAILABLE "server shutting down"`, after
/// marking `log` shutdown-caused — for a session token cancelled because the
/// session was found missing from the live store with no deletion marker
/// ([`end_on_session_cancel`]), for a receiver lag that lost a peer's
/// `PeerJoined` or `PeerLeft` ([`presence_resync_status`], counted on
/// `live_edit_presence_resyncs_total`), and for an armed join's keepalive
/// deadline ([`client_keepalive_timeout_status`], after marking `log` with the
/// client-keepalive marker, counted on
/// `live_edit_presence_keepalive_timeouts_total`). A lag that lost only
/// `PeerMessage`s, or this join's own events, is counted and the loop
/// continues. Every other exit (session deleted, client gone, event channel
/// closed) returns `Ok`.
async fn run_bidi_loop(
    ctx: &JoinCtx<'_>,
    client_stream: &mut Streaming<JoinSessionRequest>,
    event_rx: &mut PresenceReceiver,
    mut delivered: ReplayDelivered,
    session_cancel: &SessionCancel,
    log: &mut RpcLogContext,
) -> Result<(), Status> {
    // `live_presence_events_total` for this stream, batched; see
    // [`PresenceEventTally`]. Its `Drop` flushes on every exit from this loop.
    let mut tally = PresenceEventTally::new(&ctx.svc.metrics);
    let flush_timer = tokio::time::sleep(PRESENCE_TALLY_FLUSH_INTERVAL);
    tokio::pin!(flush_timer);
    // D's time limit (end cause (c)), from the loop's start. Created only for
    // a D that holds ids now: an empty D (no peers delivered) holds no
    // allocation — the replay's `shrink_to_fit` freed it — and never gains an
    // id, so it has nothing for a limit to bound. Its arm is polled only while
    // D is live; a D emptied in the loop keeps its table until it ends, so the
    // limit still applies to it.
    let window_armed = delivered.len() > 0;
    #[cfg(any(test, feature = "test-fault-injection"))]
    let duplicate_window_limit = ctx
        .svc
        .presence_replay_seam_for_test
        .as_ref()
        .and_then(|seam| seam.duplicate_window_limit())
        .unwrap_or(REPLAY_DUPLICATE_WINDOW_LIMIT);
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let duplicate_window_limit = REPLAY_DUPLICATE_WINDOW_LIMIT;
    let duplicate_window: futures::future::OptionFuture<_> = window_armed
        .then(|| tokio::time::sleep(duplicate_window_limit))
        .into();
    tokio::pin!(duplicate_window);
    // The client keepalive deadline of an armed join, started here at the
    // loop's start: the membership replay, during which the client's stream
    // is not read, never counts against it. `None` for an unarmed join.
    let mut keepalive_deadline = ctx.keepalive.as_ref().map(KeepaliveDeadline::start);

    loop {
        tokio::select! {
            biased;

            // Service shutdown (SDD §5.6, SR-4.7): teardown runs, then the
            // stream ends `UNAVAILABLE`, so the client rejoins on another
            // replica. First, so a join that sees a shutdown and a deletion
            // together ends as a shutdown. Reads the join's enrolled child
            // token, never a shared one (see `JoinCtx::shutdown`).
            () = ctx.shutdown.cancelled() => {
                log.shutdown_caused();
                return Err(server_shutting_down());
            }

            // Session token cancelled (through the join's child): teardown
            // runs, and the stream ends by the cause — `OK` for a deletion
            // (PRD §3.1.2), `UNAVAILABLE` for a session missing from the live
            // store.
            () = session_cancel.cancelled() => {
                return end_on_session_cancel(session_cancel);
            }

            // Client disconnected: `tx.closed()` fires when the gRPC
            // response stream's receiver is dropped. Wake immediately so
            // the presence-router subscriber slot (+ pub/sub subscription
            // if last subscriber for this session on this replica) is
            // released without waiting for the next event or client
            // message (b.jr8 H-grpc-2).
            () = ctx.tx.closed() => {
                return Ok(());
            }

            // Pending deliveries reach the counter at most one interval
            // after they were recorded; armed only while some are pending.
            () = &mut flush_timer, if tally.has_pending() => {
                tally.flush();
                flush_timer
                    .as_mut()
                    .reset(tokio::time::Instant::now() + PRESENCE_TALLY_FLUSH_INTERVAL);
            }

            // D's time limit (end cause (c)): this join's own first live
            // event has not arrived within the limit, so D ends, freed, and
            // suppression stops. Armed only for a D that held ids at the
            // loop's start (`window_armed`; for any other D the timer does not
            // exist), and only while D is live; once D has ended this arm is
            // never polled again. `remaining` can be 0 for a D every id has
            // left, whose table this frees.
            Some(()) = &mut duplicate_window, if window_armed && delivered.is_live() => {
                tracing::debug!(
                    session_id = %ctx.session_id,
                    connection_id = %ctx.connection_id,
                    remaining = delivered.len(),
                    "replay duplicate suppression ended at its time limit before the join's own presence event arrived"
                );
                delivered.end(DeliveredEnd::TimeLimit);
            }

            // Client sends a message, a keepalive, or anything else: see
            // `handle_client_message`. The read stamps the body's progress
            // record for every DATA frame it consumes.
            client_msg = client_stream.message() => {
                if handle_client_message(ctx, client_msg).await.is_break() {
                    return Ok(());
                }
            }

            // Client keepalive deadline (SDD §13.2), armed joins only: no
            // data from the client for three intervals ends the join
            // `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT`, after teardown has
            // published its one `left`. After the client-message arm, so this
            // arm is reached only in a pass where tonic has just polled the
            // request body to `Pending`: the join is never ended while bytes
            // of its stream wait unread at the server. Before the event arm,
            // so a busy session cannot starve it.
            () = keepalive_expired(&mut keepalive_deadline), if keepalive_deadline.is_some() => {
                return Err(end_on_keepalive_timeout(ctx, log));
            }

            // Presence event from Valkey pub/sub (via local router).
            event = event_rx.recv() => {
                match event {
                    Ok(presence_event) => {
                        // Self-filter: don't send events originating from this connection.
                        let sender_cid = match &*presence_event {
                            PresenceEvent::PeerJoined(user) => &user.connection_id,
                            PresenceEvent::PeerLeft { connection_id } => connection_id,
                            PresenceEvent::PeerMessage { connection_id, .. } => connection_id,
                        };
                        if sender_cid == ctx.connection_id {
                            // This join's own first live event ends D (end
                            // cause (a)): it was published after the snapshot,
                            // so every event published before the snapshot has
                            // already arrived. Later own events find D already
                            // ended. See `ReplayDelivered`.
                            delivered.end(DeliveredEnd::OwnEvent);
                            continue;
                        }

                        // Replay duplicate (SDD SR-4.4), while D is live: a
                        // live `PeerJoined` for a peer the replay delivered,
                        // with no `left` for it delivered since, is dropped
                        // here — neither sent nor tallied. A `PeerLeft` takes
                        // its id out of D and is forwarded. See
                        // `ReplayDelivered`.
                        if delivered.suppresses(&presence_event) {
                            continue;
                        }

                        // The send races the loop's own terminal arms, in the
                        // loop's order — shutdown, deletion, client gone — so a
                        // client that has stopped reading, with the channel
                        // full, cannot park the join here past any of them.
                        // Without this a shutdown could not reach a stalled
                        // join, and the shutdown sequence's presence drain
                        // would wait out its whole budget for it.
                        //
                        // An armed join's send also races, after those, the
                        // client's request stream and the keepalive deadline:
                        // a vanished client in a busy session parks the loop
                        // here once the channel and the proxy's window are
                        // full, and without the deadline arm it would never
                        // end. The read arm keeps consuming — and stamping —
                        // what a live client sends while its responses are
                        // back-pressured, publishing its messages as the outer
                        // arm does, so such a client is not ended. The send
                        // stays pinned across those arms, keeping its place
                        // in the channel's queue. For an unarmed join both
                        // arms are disabled, and the send races exactly the
                        // three terminal arms.
                        let armed = keepalive_deadline.is_some();
                        let send = ctx.tx.send(Ok(presence_event_to_proto(&presence_event)));
                        tokio::pin!(send);
                        loop {
                            tokio::select! {
                                biased;
                                () = ctx.shutdown.cancelled() => {
                                    log.shutdown_caused();
                                    return Err(server_shutting_down());
                                }
                                () = session_cancel.cancelled() => {
                                    return end_on_session_cancel(session_cancel);
                                }
                                () = ctx.tx.closed() => {
                                    return Ok(());
                                }
                                client_msg = client_stream.message(), if armed => {
                                    if handle_client_message(ctx, client_msg).await.is_break() {
                                        return Ok(());
                                    }
                                }
                                () = keepalive_expired(&mut keepalive_deadline), if armed => {
                                    return Err(end_on_keepalive_timeout(ctx, log));
                                }
                                sent = &mut send => {
                                    if sent.is_err() {
                                        // Client disconnected.
                                        return Ok(());
                                    }
                                    break;
                                }
                            }
                        }
                        // Counted per recipient, only once the event is
                        // enqueued to this stream.
                        tally.record(&presence_event);
                    }
                    Err(PresenceRecvError::Lagged { dropped, membership_lost }) => {
                        // A flapping receiver can lag thousands of times per
                        // second; logging each at WARN would flood log
                        // aggregators and drown the incident signal. The
                        // line stays at DEBUG, and alerting keys on
                        // `live_edit_presence_broadcast_dropped_total`,
                        // which sums lost events across all sessions.
                        ctx.svc
                            .metrics
                            .inc_presence_broadcast_dropped(&ctx.svc.service_label, dropped);
                        tracing::debug!(
                            session_id = %ctx.session_id,
                            connection_id = %ctx.connection_id,
                            lagged = dropped,
                            membership_lost,
                            "presence event receiver lagged — see live_edit_presence_broadcast_dropped_total"
                        );
                        // D ends on every lag (end cause (b)), a
                        // message-only one included, by choice: D's
                        // correctness then does not depend on
                        // `membership_lost`, at the cost of R1's duplicate
                        // `PeerJoined`, which clients ignore. Nothing is
                        // suppressed from here on. See `ReplayDelivered`.
                        delivered.end(DeliveredEnd::Lag);
                        // A lost peer join or leave would leave the client's
                        // participant list wrong with no later event to
                        // correct it, and an in-stream repair could not
                        // produce the `PeerLeft` for a peer that left in the
                        // lost window without a per-connection record of
                        // what the client was shown. So the stream ends
                        // retryable: `run_join_session` logs it once and
                        // tears down (peers see this connection leave), and
                        // the client's rejoin replays the membership afresh.
                        // A loss of peer messages alone continues: they are
                        // best effort (SDD §13.2). So does a loss of this
                        // join's own joins and leaves, which the self-filter
                        // would discard: the receiver is tied to this
                        // connection and leaves them out of
                        // `membership_lost`.
                        if membership_lost {
                            ctx.svc.metrics.inc_presence_resync();
                            return Err(presence_resync_status());
                        }
                        continue;
                    }
                    Err(PresenceRecvError::Closed) => {
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// The `event_type` labels of `live_presence_events_total`, in
/// [`PresenceEventTally`]'s slot order.
const PRESENCE_TALLY_SLOTS: [&str; 3] = [
    PRESENCE_EVENT_JOINED,
    PRESENCE_EVENT_LEFT,
    PRESENCE_EVENT_MESSAGE,
];

/// `event`'s slot in [`PRESENCE_TALLY_SLOTS`].
fn presence_tally_slot(event: &PresenceEvent) -> usize {
    match event {
        PresenceEvent::PeerJoined(_) => 0,
        PresenceEvent::PeerLeft { .. } => 1,
        PresenceEvent::PeerMessage { .. } => 2,
    }
}

/// Deliveries a [`PresenceEventTally`] holds, across all event types, before
/// flushing them at once.
const PRESENCE_TALLY_FLUSH_AT: u64 = 64;

/// How often `run_bidi_loop` flushes a pending [`PresenceEventTally`].
const PRESENCE_TALLY_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// One `JoinSession` stream's pending `live_presence_events_total` counts, by
/// `event_type`.
///
/// The counter is per recipient per event, so a per-delivery `add(1)` puts
/// every recipient task in a large session on the same counter cell (see
/// `Metrics::add_presence_events` for the measured cost). The tally
/// accumulates locally and flushes each type with one `add(n)` when
/// [`PRESENCE_TALLY_FLUSH_AT`] deliveries are pending, when `run_bidi_loop`'s
/// flush timer fires (every [`PRESENCE_TALLY_FLUSH_INTERVAL`]), and on `Drop` —
/// every exit from the loop, errors included. Every delivery is counted
/// exactly once; only when it reaches the counter is batched. A delivery
/// therefore reaches the counter within about a second while the stream is
/// open, unless the loop is parked on a send to a slow reader, and all of them
/// by the time the stream ends.
struct PresenceEventTally<'a> {
    metrics: &'a crate::observability::metrics::Metrics,
    pending: [u64; 3],
}

impl<'a> PresenceEventTally<'a> {
    fn new(metrics: &'a crate::observability::metrics::Metrics) -> Self {
        PresenceEventTally {
            metrics,
            pending: [0; 3],
        }
    }

    /// `event` was enqueued to this stream.
    fn record(&mut self, event: &PresenceEvent) {
        self.pending[presence_tally_slot(event)] += 1;
        if self.pending.iter().sum::<u64>() >= PRESENCE_TALLY_FLUSH_AT {
            self.flush();
        }
    }

    fn has_pending(&self) -> bool {
        self.pending.iter().any(|&n| n > 0)
    }

    fn flush(&mut self) {
        for (slot, event_type) in PRESENCE_TALLY_SLOTS.iter().enumerate() {
            let n = std::mem::take(&mut self.pending[slot]);
            if n > 0 {
                self.metrics.add_presence_events(event_type, n);
            }
        }
    }
}

impl Drop for PresenceEventTally<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Convert a PresenceEvent to a JoinSessionResponse.
fn presence_event_to_proto(event: &PresenceEvent) -> JoinSessionResponse {
    match event {
        PresenceEvent::PeerJoined(user) => JoinSessionResponse {
            response_type: Some(
                super::proto::join_session_response::ResponseType::PeerJoined(PeerJoined {
                    connection_id: user.connection_id.clone(),
                    user_name: user.user_name.clone(),
                    user_application: user.user_application.clone(),
                }),
            ),
        },
        PresenceEvent::PeerLeft { connection_id } => JoinSessionResponse {
            response_type: Some(super::proto::join_session_response::ResponseType::PeerLeft(
                PeerLeft {
                    connection_id: connection_id.clone(),
                },
            )),
        },
        PresenceEvent::PeerMessage {
            connection_id,
            data,
        } => JoinSessionResponse {
            response_type: Some(
                super::proto::join_session_response::ResponseType::PeerMessage(PeerMessage {
                    connection_id: connection_id.clone(),
                    data: data.clone(),
                }),
            ),
        },
    }
}

/// Tear the connection down (SDD SR-4.3). Runs after every exit from
/// [`run_join_session_inner`], so on every exit after the shutdown re-check,
/// and undoes exactly what `presence` records was done:
///
/// 1. cancel the lease keeper's teardown token;
/// 2. await the keeper, if one was spawned, bounded by the store command
///    timeout plus [`KEEPER_STOP_SLACK`]. On timeout it logs a WARN and goes
///    on; a keeper panic surfaces here as a `JoinError`, is logged, and
///    teardown goes on;
/// 3. if the register was issued — even if it errored — unregister: one
///    script arms the `:gone` marker, removes the member, deletes the info
///    hash and lease, and publishes `left` iff it removed the member. A
///    connection a sweep already removed (and announced) is therefore not
///    announced again;
/// 4. if the unregister left the session with no members, or the register
///    was never issued, remove the session from the index by
///    remove-then-recheck;
/// 5. release the router count only if this join took one — and
///    `UNSUBSCRIBE` from Valkey when it was the session's last local one,
///    with the router holding off any joiner's `SUBSCRIBE` until that
///    completes.
///
/// Late writes are fenced by the marker and the register's deadline, not by
/// step 2's bound. A keeper rejoin register that reaches the store after step
/// 3 finds the marker, or once the marker has expired is past its deadline:
/// the keeper read the clock for that deadline before step 1 cancelled its
/// teardown token. A keeper refresh after step 3 finds the marker, or once it
/// has expired no member. Either writes nothing. Store
/// errors are logged, and a timeout is counted once where received; nothing
/// here fails the stream.
async fn teardown(ctx: &JoinCtx<'_>, presence: JoinPresence) {
    let valkey = &ctx.svc.valkey;
    let keys = &ctx.svc.key_schema;
    let origin = FredErrOrigin::client(valkey, POOL_LABEL_DATA);

    // ── SR-4.3 steps 1–2: Stop the lease keeper ──────────────────────────────
    if let Some(keeper) = presence.keeper {
        keeper.teardown.cancel();
        // `last()`, not `next()`: this only reads the configured command
        // timeout, and `next()` would advance the pool's round-robin cursor.
        let bound = valkey.pool.last().perf_config().default_command_timeout + KEEPER_STOP_SLACK;
        match tokio::time::timeout(bound, keeper.handle).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::error!(
                    session_id = %ctx.session_id,
                    connection_id = %ctx.connection_id,
                    error = %e,
                    "presence lease keeper task failed; continuing teardown"
                );
            }
            Err(_) => {
                tracing::warn!(
                    session_id = %ctx.session_id,
                    connection_id = %ctx.connection_id,
                    bound_ms = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX),
                    "presence lease keeper did not stop within its bound; continuing teardown"
                );
            }
        }
    }

    // ── SR-4.3 step 3: Unregister ────────────────────────────────────────────
    let mut remove_from_index = !presence.register_issued;
    if presence.register_issued {
        let left_payload = PresenceEvent::PeerLeft {
            connection_id: ctx.connection_id.to_string(),
        }
        .encode();
        match unregister_connection(
            valkey,
            origin,
            keys,
            ctx.session_id,
            ctx.connection_id,
            ctx.svc.presence_lease_ttl_secs,
            &left_payload,
        )
        .await
        {
            Ok(unregistered) => remove_from_index = unregistered.remaining == 0,
            Err(e) => {
                tracing::warn!(
                    session_id = %ctx.session_id,
                    connection_id = %ctx.connection_id,
                    error = %e,
                    "failed to unregister presence on teardown"
                );
            }
        }
    }

    // ── SR-4.3 step 4: Remove the session from the index if it is empty ──────
    if remove_from_index {
        if let Err(e) = remove_session_from_index(valkey, origin, keys, ctx.session_id).await {
            tracing::warn!(
                session_id = %ctx.session_id,
                connection_id = %ctx.connection_id,
                error = %e,
                "failed to remove the session from the presence session index on teardown"
            );
        }
    }

    // ── SR-4.3 step 5: Release the router subscription ───────────────────────
    // Only if this join joined the router: an exit before that must not
    // decrement another local connection's count. A join whose channel
    // `SUBSCRIBE` failed (its own or one it waited on) still holds a count and
    // releases it here. `leave_channel` hands back the duty to unsubscribe
    // only on the last-disconnect (1 -> 0) transition for this session on
    // this replica; the router keeps the session's entry until `finish`, so a
    // join arriving meanwhile sends its `SUBSCRIBE` only after this
    // `UNSUBSCRIBE` and the channel ends subscribed.
    if presence.subscribed {
        if let Some(unsubscribe) = ctx.svc.presence_router.leave_channel(ctx.session_id) {
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(seam) = ctx.svc.presence_channel_seam_for_test.as_ref() {
                seam.before_unsubscribe.arrive().await;
            }
            if let Err(e) = unsubscribe_presence_channel(
                &ctx.svc.valkey.subscriber,
                &ctx.svc.key_schema,
                ctx.session_id,
            )
            .await
            {
                FredErrOrigin::client(&ctx.svc.valkey, POOL_LABEL_PUBSUB).observe_error(&e);
                tracing::warn!(
                    session_id = %ctx.session_id,
                    connection_id = %ctx.connection_id,
                    error = %e,
                    "failed to unsubscribe presence channel on teardown"
                );
            }
            unsubscribe.finish();
        }
    }
}

/// One park point of the JoinSession presence test seams
/// ([`JoinRegistrationSeam`], [`PresenceReplaySeam`], [`LeaseKeeperSeam`],
/// [`PresenceChannelSeam`]); defined in `presence`, which the presence sweeper's
/// seam shares, and re-exported here.
#[cfg(any(test, feature = "test-fault-injection"))]
pub use crate::presence::PresenceSeamGate;

/// Test-only park points in a join's registration (SDD SR-4 steps 2–6), for
/// races that close too fast to hit without one. Installed through
/// `LiveEditService::join_registration_seam_for_test`; every gate starts
/// disarmed (see [`PresenceSeamGate`]).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
pub struct JoinRegistrationSeam {
    /// After the join's first index add (step 2), before its router subscribe
    /// (step 3). The session is indexed; the join is neither subscribed nor
    /// registered. For an index removal gated between the join's first add
    /// and its register, or a Valkey `SUBSCRIBE` made to fail.
    pub after_index_add: PresenceSeamGate,
    /// Inside step 4, after the store-clock read that fixes the register's
    /// deadline and the membership snapshot, and before the register is
    /// issued: the join is indexed and subscribed but not a member, its
    /// deadline is already set, and its snapshot already taken — so a peer
    /// that registers while the join is parked here is not in the snapshot
    /// and reaches the join live only. For an index removal gated just before
    /// the register; a stall armed on the register command alone (the clock
    /// read and the snapshot have already completed); or a hold longer than
    /// the lease TTL, after which the register reaches the store past its
    /// deadline and returns `Expired`. For a park inside the
    /// subscribe-to-snapshot window, see
    /// [`PresenceReplaySeam::before_snapshot`].
    pub before_register: PresenceSeamGate,
    /// After the register and the index re-add (steps 4–5), before
    /// `acquire_session_cancel` (step 6). The join is a member and no keeper
    /// runs yet. For a shutdown or session delete that lands mid-join, which
    /// the join must see at its step-7 check.
    pub before_session_cancel: PresenceSeamGate,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl JoinRegistrationSeam {
    /// Every gate disarmed.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Test-only park points in, and observations of, the membership snapshot
/// and replay (SDD SR-4 steps 4 and 10, SR-4.4) and of the replay-delivered
/// set D the replay hands the bidi loop ([`ReplayDelivered`]), plus an
/// override of D's time limit. Installed through
/// `LiveEditService::presence_replay_seam_for_test`; every gate starts
/// disarmed (see [`PresenceSeamGate`]).
///
/// **Cancelling a parked replay.** The gates are not cancellation-aware, so a
/// test fires its trigger (a shutdown, a session delete, a client disconnect)
/// while the replay is parked, then releases the gate. The replay sees the
/// ending at its next raced step — the next pipeline, or the next send — before
/// that step's operation is polled.
///
/// **Observations.** Recorded by every replay on the service while the seam is
/// installed, and accumulated across them; [`Self::reset_observations`]
/// clears them, so a test that sets up other connections first resets before
/// driving the join it measures. The result is the last replay's. D is
/// observed per joining connection ([`Self::delivered_set`]).
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct PresenceReplaySeam {
    /// Inside the subscribe-to-snapshot window: in SR-4 step 4, after the
    /// join's router subscribe (step 3) and its store-clock read, just before
    /// its `SMEMBERS` snapshot and, after that, its register
    /// ([`JoinRegistrationSeam::before_register`]). The join is NOT a member
    /// here and no lease keeper runs, and the register's deadline is already
    /// fixed by the clock read, so a hold must stay under one lease TTL — a
    /// longer one makes the register return `Expired` and fails the join.
    /// While the join is parked here:
    /// - a peer that registers is in the snapshot and also reaches the join
    ///   live, ahead of the join's own `joined`; its live `PeerJoined` is the
    ///   duplicate D suppresses;
    /// - a peer that leaves (an in-script `left`) and rejoins reaches the join
    ///   as the replay's `PeerJoined` (it is a member at the snapshot), then
    ///   the live `PeerLeft` and the live `PeerJoined`. The `left` takes it
    ///   out of D first, so nothing is suppressed.
    ///
    /// A failure injected here (a stalled data pool, say) fails the snapshot
    /// read, which ends the join before `SessionConnected` with the register
    /// never issued.
    pub before_snapshot: PresenceSeamGate,
    /// At the replay's start (SR-4 step 10), after step 4's member-set
    /// snapshot, before its first info read and first send — reached whether
    /// or not there are peers to replay. `SessionConnected` has been sent and
    /// the lease keeper is running, so a test can hold the replay past the
    /// lease TTL.
    pub before_first_send: PresenceSeamGate,
    /// Between pages: after one page's sends, before the next page's pipeline.
    /// Reached only by a replay of more than one page, once per page after the
    /// first.
    pub between_pages: PresenceSeamGate,
    /// Pipelines issued; see [`Self::pipelines_issued`].
    pipelines_issued: std::sync::atomic::AtomicUsize,
    /// Each issued pipeline's page size, in issue order.
    page_sizes: std::sync::Mutex<Vec<usize>>,
    /// The most pipelines one replay has had in flight at once.
    max_pipelines_in_flight: std::sync::atomic::AtomicUsize,
    /// Completed replay `PeerJoined` sends.
    peer_joined_sent: tokio::sync::watch::Sender<usize>,
    /// The last replay's result, `None` until one ends.
    result: tokio::sync::watch::Sender<Option<ObservedReplayResult>>,
    /// Each joining connection's D, by its connection id, from the moment its
    /// replay completes.
    delivered_sets:
        tokio::sync::watch::Sender<std::collections::HashMap<String, DeliveredSetObservation>>,
    /// D's time limit for bidi loops that start from now on, in place of
    /// [`REPLAY_DUPLICATE_WINDOW_LIMIT`]; see
    /// [`Self::set_duplicate_window_limit`].
    duplicate_window_limit: std::sync::Mutex<Option<std::time::Duration>>,
}

/// One joining connection's replay-delivered set D ([`ReplayDelivered`]), as
/// [`PresenceReplaySeam`] observes it. Recorded from the moment its replay
/// completes; a replay that does not complete creates no D, so records none.
///
/// **Test-only.** Gated with the seam.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveredSetObservation {
    /// D's size when the replay completed: the `PeerJoined` it delivered.
    pub created_len: usize,
    /// D's current size while it is live, `None` once it has ended (and been
    /// freed). Never above `created_len`: D only shrinks.
    pub len: Option<usize>,
    /// Live `PeerJoined` the bidi loop suppressed because their id was in D.
    pub suppressed_peer_joined: usize,
    /// Why D ended, `None` while it is live.
    pub ended_by: Option<DeliveredSetEnd>,
}

/// Why a join's D ended, as [`PresenceReplaySeam`] observes it — the first of
/// the four causes on [`ReplayDelivered`].
///
/// **Test-only.** Gated with the seam.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveredSetEnd {
    /// (a) The join's own first live event arrived.
    OwnEvent,
    /// (b) The join's presence receiver lagged.
    Lag,
    /// (c) D's time limit elapsed first ([`REPLAY_DUPLICATE_WINDOW_LIMIT`],
    /// or the seam's override). Never for a D whose `created_len` is 0: no
    /// timer exists for it.
    TimeLimit,
    /// (d) The bidi loop ended while D was still live.
    LoopEnd,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl From<DeliveredEnd> for DeliveredSetEnd {
    fn from(cause: DeliveredEnd) -> Self {
        match cause {
            DeliveredEnd::OwnEvent => DeliveredSetEnd::OwnEvent,
            DeliveredEnd::Lag => DeliveredSetEnd::Lag,
            DeliveredEnd::TimeLimit => DeliveredSetEnd::TimeLimit,
        }
    }
}

/// Where a [`ReplayDelivered`] reports its changes, in test builds.
#[cfg(any(test, feature = "test-fault-injection"))]
struct DeliveredSetObserver {
    seam: std::sync::Arc<PresenceReplaySeam>,
    connection_id: String,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReplayDelivered {
    /// The replay completed: start reporting this D to `seam`, under
    /// `connection_id`.
    fn observe(&mut self, seam: std::sync::Arc<PresenceReplaySeam>, connection_id: &str) {
        let len = self.len();
        seam.delivered_sets.send_modify(|sets| {
            sets.insert(
                connection_id.to_owned(),
                DeliveredSetObservation {
                    created_len: len,
                    len: Some(len),
                    suppressed_peer_joined: 0,
                    ended_by: None,
                },
            );
        });
        self.observer = Some(DeliveredSetObserver {
            seam,
            connection_id: connection_id.to_owned(),
        });
    }

    /// Report D's current size, plus one suppression if `suppressed`.
    fn observe_change(&self, suppressed: bool) {
        let len = self.len();
        self.update_observation(|observed| {
            observed.len = Some(len);
            observed.suppressed_peer_joined += usize::from(suppressed);
        });
    }

    /// Report that D ended for `cause`.
    fn observe_end(&self, cause: DeliveredSetEnd) {
        self.update_observation(|observed| {
            observed.len = None;
            observed.ended_by = Some(cause);
        });
    }

    /// Apply `update` to this D's observation, if it is observed and the
    /// observation has not been reset since.
    fn update_observation(&self, update: impl FnOnce(&mut DeliveredSetObservation)) {
        let Some(observer) = self.observer.as_ref() else {
            return;
        };
        observer.seam.delivered_sets.send_if_modified(|sets| {
            match sets.get_mut(&observer.connection_id) {
                Some(observed) => {
                    update(observed);
                    true
                }
                None => false,
            }
        });
    }
}

/// An observed D is dropped: the bidi loop ended, which is D's end (d) if no
/// earlier cause ended it. (An unobserved one — the replay's own, dropped on
/// an exit before completion — reports nothing.)
#[cfg(any(test, feature = "test-fault-injection"))]
impl Drop for ReplayDelivered {
    fn drop(&mut self) {
        if self.is_live() {
            self.observe_end(DeliveredSetEnd::LoopEnd);
        }
    }
}

/// How a membership replay ended, as [`PresenceReplaySeam`] records it.
///
/// **Test-only.** Gated with the seam.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedReplayResult {
    /// Every other member with a valid info hash was sent.
    Complete,
    /// The join's shutdown token ended it.
    ShuttingDown,
    /// The join's session token ended it.
    SessionCancelled,
    /// It returned an error with this code: `CANCELLED` for a client
    /// disconnect, the mapped code for a store error.
    Failed(tonic::Code),
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Default for PresenceReplaySeam {
    fn default() -> Self {
        Self {
            before_snapshot: PresenceSeamGate::new(),
            before_first_send: PresenceSeamGate::new(),
            between_pages: PresenceSeamGate::new(),
            pipelines_issued: std::sync::atomic::AtomicUsize::new(0),
            page_sizes: std::sync::Mutex::new(Vec::new()),
            max_pipelines_in_flight: std::sync::atomic::AtomicUsize::new(0),
            peer_joined_sent: tokio::sync::watch::channel(0).0,
            result: tokio::sync::watch::channel(None).0,
            delivered_sets: tokio::sync::watch::channel(std::collections::HashMap::new()).0,
            duplicate_window_limit: std::sync::Mutex::new(None),
        }
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl PresenceReplaySeam {
    /// Every gate disarmed, nothing observed, D's time limit the default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Test side: use `limit` instead of [`REPLAY_DUPLICATE_WINDOW_LIMIT`] as
    /// D's time limit (end cause (c)) for every bidi loop on the service that
    /// starts after this call with a D that holds ids — a loop whose D is
    /// empty at its start has no time limit to override; a loop already
    /// running keeps the limit it started with. Not an observation, so
    /// [`Self::reset_observations`] leaves it.
    pub fn set_duplicate_window_limit(&self, limit: std::time::Duration) {
        *self
            .duplicate_window_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(limit);
    }

    /// Production side: the overridden time limit, if a test set one.
    fn duplicate_window_limit(&self) -> Option<std::time::Duration> {
        *self
            .duplicate_window_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Test side: clear every observation. The gates and D's time-limit
    /// override are untouched. A D that exists at the reset is no longer
    /// observed; only replays that complete after it are.
    pub fn reset_observations(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        self.pipelines_issued.store(0, SeqCst);
        self.page_sizes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.max_pipelines_in_flight.store(0, SeqCst);
        self.peer_joined_sent.send_replace(0);
        self.result.send_replace(None);
        self.delivered_sets
            .send_modify(std::collections::HashMap::clear);
    }

    /// Test side: member-info pipelines issued. A pipeline is counted when it
    /// is sent, inside the future its cancel race polls, so one a cancel arm
    /// drops before its first poll is not counted.
    pub fn pipelines_issued(&self) -> usize {
        self.pipelines_issued
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Test side: each issued pipeline's page size (the connection ids it
    /// reads), in issue order.
    pub fn page_sizes(&self) -> Vec<usize> {
        self.page_sizes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Test side: the most pipelines any one replay has had in flight at
    /// once.
    pub fn max_pipelines_in_flight(&self) -> usize {
        self.max_pipelines_in_flight
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Test side: completed replay `PeerJoined` sends — enqueued, not merely
    /// started. A replay whose count holds while a send is due is blocked on
    /// that send.
    pub fn peer_joined_sent_count(&self) -> usize {
        *self.peer_joined_sent.borrow()
    }

    /// Test side: block until at least `n` replay `PeerJoined` sends have
    /// completed.
    pub async fn wait_for_peer_joined_sent(&self, n: usize) {
        let mut sent = self.peer_joined_sent.subscribe();
        // Errs only once the sender is dropped, and `self` holds it.
        let _ = sent.wait_for(|&sent| sent >= n).await;
    }

    /// Test side: the last replay's result, `None` until one ends.
    pub fn result(&self) -> Option<ObservedReplayResult> {
        *self.result.borrow()
    }

    /// Test side: block until a replay has ended, and return its result.
    pub async fn wait_for_result(&self) -> ObservedReplayResult {
        let mut result = self.result.subscribe();
        let observed = result
            .wait_for(Option::is_some)
            .await
            .expect("the seam holds the result sender, so the channel cannot close");
        observed.expect("wait_for returned a recorded result")
    }

    /// Test side: the D of the join whose connection id is `connection_id`,
    /// `None` until that join's replay completes (or if it never does).
    pub fn delivered_set(&self, connection_id: &str) -> Option<DeliveredSetObservation> {
        self.delivered_sets.borrow().get(connection_id).copied()
    }

    /// Test side: block until the D of `connection_id`'s join exists and
    /// satisfies `until`, and return it. Level-triggered: returns at once if it
    /// already does.
    pub async fn wait_for_delivered_set(
        &self,
        connection_id: &str,
        mut until: impl FnMut(&DeliveredSetObservation) -> bool,
    ) -> DeliveredSetObservation {
        let mut sets = self.delivered_sets.subscribe();
        let sets = sets
            .wait_for(|sets| sets.get(connection_id).is_some_and(&mut until))
            .await
            .expect("the seam holds the delivered-set sender, so the channel cannot close");
        *sets
            .get(connection_id)
            .expect("wait_for returned a recorded observation")
    }

    /// Production side: a page's pipeline is being sent. Counts it, records
    /// its size, and holds one of this replay's in-flight slots (`in_flight`)
    /// until the returned guard drops — when the read completes, or a cancel
    /// arm drops it.
    fn pipeline_issued<'a>(
        &self,
        page_size: usize,
        in_flight: &'a std::sync::atomic::AtomicUsize,
    ) -> PipelineInFlight<'a> {
        use std::sync::atomic::Ordering::SeqCst;
        self.pipelines_issued.fetch_add(1, SeqCst);
        self.page_sizes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(page_size);
        let now = in_flight.fetch_add(1, SeqCst) + 1;
        self.max_pipelines_in_flight.fetch_max(now, SeqCst);
        PipelineInFlight(in_flight)
    }

    /// Production side: one replay `PeerJoined` send completed.
    fn peer_joined_sent(&self) {
        self.peer_joined_sent.send_modify(|sent| *sent += 1);
    }

    /// Production side: the replay ended with `result`.
    fn record_result(&self, result: &Result<ReplayOutcome, Status>) {
        let observed = match result {
            Ok(ReplayOutcome::Complete(_)) => ObservedReplayResult::Complete,
            Ok(ReplayOutcome::ShuttingDown) => ObservedReplayResult::ShuttingDown,
            Ok(ReplayOutcome::SessionCancelled) => ObservedReplayResult::SessionCancelled,
            Err(status) => ObservedReplayResult::Failed(status.code()),
        };
        self.result.send_replace(Some(observed));
    }
}

/// One replay pipeline in flight, for [`PresenceReplaySeam`]'s high-water
/// mark; releases its slot on drop.
#[cfg(any(test, feature = "test-fault-injection"))]
struct PipelineInFlight<'a>(&'a std::sync::atomic::AtomicUsize);

#[cfg(any(test, feature = "test-fault-injection"))]
impl Drop for PipelineInFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Test-only park points in a connection's lease keeper (SDD SR-4.2).
/// Installed through `LiveEditService::lease_keeper_seam_for_test`; both gates
/// start disarmed (see [`PresenceSeamGate`]).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
pub struct LeaseKeeperSeam {
    /// Inside a keeper step, before its refresh. The step has started, so it
    /// runs its refresh once released whatever has been cancelled meanwhile —
    /// a test can hold a refresh while teardown starts. Teardown's bounded
    /// wait for the keeper runs while the step is parked. If the refresh then
    /// reports the connection evicted, the rejoin's first stop check sees the
    /// cancellation and the step writes nothing more.
    pub before_refresh: PresenceSeamGate,
    /// Inside a rejoin (a step whose refresh reported `Evicted` or
    /// `InfoMissing`), after the first stop check and the store-clock read
    /// that fixes the register's deadline, before the second stop check and
    /// the register. Teardown's bounded wait for the keeper runs while the
    /// step is parked.
    ///
    /// - Released after teardown has begun: the second check stops the step,
    ///   and no register is issued.
    /// - Released before teardown, with no stall: the register is issued with
    ///   its deadline already set. A park held longer than one lease TTL
    ///   makes the register reach the store past that deadline, and the
    ///   keeper itself gets `Expired`.
    /// - Released before teardown, with the register stalled: a stall armed
    ///   while parked lands on the register command alone (the clock read has
    ///   completed). The register can then be held past teardown's
    ///   unregister, where the teardown marker refuses it, or past the
    ///   marker's expiry, where its deadline refuses it.
    ///
    ///   The keeper never sees either refusal. The register was issued before
    ///   teardown cancelled the token, so it times out (at issue plus the
    ///   command timeout) before teardown's bounded wait (from cancel, the
    ///   command timeout plus [`KEEPER_STOP_SLACK`]) runs out. The keeper gets
    ///   `Err`, and its
    ///   loop stops on the cancelled token. Only then does teardown send the
    ///   unregister. A register held past the unregister therefore always
    ///   shows up in the keeper as its command timeout. The marker (`Gone`) or
    ///   deadline (`Expired`) refusal shows up only as the store writing
    ///   nothing once the stall is released.
    ///
    ///   Such a test must make sure teardown's unregister is not sent on the
    ///   stalled data-pool connection. Teardown takes the unregister's
    ///   connection with `pool.next()`, advancing the round-robin cursor once
    ///   (its keeper-stop bound reads the command timeout through
    ///   `pool.last()`, which does not advance it). An unregister landing on
    ///   the stalled connection queues behind the register, so the fence is
    ///   never exercised.
    pub before_rejoin_register: PresenceSeamGate,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl LeaseKeeperSeam {
    /// Both gates disarmed.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Test-only park points and fault injection around a session's Valkey
/// channel subscription on this replica (SDD SR-4 step 3 and SR-4.3 step 5).
/// Installed through `LiveEditService::presence_channel_seam_for_test`; both
/// gates start disarmed (see [`PresenceSeamGate`]) and no failure is queued.
///
/// Holding the Valkey command itself is not an option: a server-side stall of
/// the data pool does not reach the dedicated subscriber connection, so the
/// joiner is held just before it sends the command instead.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
pub struct PresenceChannelSeam {
    /// In the join that subscribes the channel — the first local joiner, or
    /// the first after a failed `SUBSCRIBE` — after the router has marked the
    /// session subscribing and after any prior local `UNSUBSCRIBE` has
    /// completed, just before `SUBSCRIBE` is sent. While it is parked every
    /// other local join into the session waits on its result, so a test can
    /// start them here and then release it.
    pub before_subscribe: PresenceSeamGate,
    /// In the teardown of the session's last local connection, after the
    /// router has marked the session unsubscribing, just before `UNSUBSCRIBE`
    /// is sent. A join that arrives while it is parked becomes the subscribing
    /// join, and sends its `SUBSCRIBE` only once this `UNSUBSCRIBE` has
    /// completed — which a test can see by also arming
    /// [`Self::before_subscribe`].
    pub before_unsubscribe: PresenceSeamGate,
    /// Set by [`Self::fail_next_subscribe`]; consumed by the next subscribing
    /// join.
    fail_next_subscribe: std::sync::atomic::AtomicBool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl PresenceChannelSeam {
    /// Both gates disarmed, no failure queued.
    pub fn new() -> Self {
        Self::default()
    }

    /// Make the next subscribing join's `SUBSCRIBE` fail without sending it,
    /// with a transient (`IO`) error — as a connection failure would — so
    /// that join and every join waiting on it end `UNAVAILABLE`. One-shot:
    /// the join after it subscribes for real. Checked after
    /// [`Self::before_subscribe`], so it may be queued while a join is parked
    /// there.
    pub fn fail_next_subscribe(&self) {
        self.fail_next_subscribe
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The injected `SUBSCRIBE` failure, if one is queued; consumes it.
    fn take_injected_subscribe_failure(&self) -> Option<fred::error::Error> {
        self.fail_next_subscribe
            .swap(false, std::sync::atomic::Ordering::SeqCst)
            .then(|| {
                fred::error::Error::new(
                    fred::error::ErrorKind::IO,
                    "injected SUBSCRIBE failure (test seam)",
                )
            })
    }
}

/// **Test-support only.** D ([`ReplayDelivered`]) built as a completed replay
/// builds it, for the replay's allocation measurement (a local measurement
/// tool outside the published source). A child module, so the type stays
/// private to this one.
#[cfg(feature = "test-support")]
pub mod replay_delivered_measure;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::metrics_scrape_only;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::Metrics;
    use crate::presence::ConnectedUser;

    // ── Endings by the session token's cause ─────────────────────────────

    /// A join that observes its session token ends by the cancel's cause: `OK`
    /// for an explicit deletion — including a bare cancel, which records no
    /// cause — and the retryable `UNAVAILABLE` for a session found missing
    /// from the live store, so the client rejoins and the rejoin can recover
    /// the session. Every arm that observes the token ends through this one
    /// function.
    #[test]
    fn end_on_session_cancel_maps_each_cause_to_its_join_ending() {
        let deleted = SessionCancel::new();
        deleted.cancel_deleted();
        assert!(
            end_on_session_cancel(&deleted).is_ok(),
            "a deletion ends the join OK (PRD §3.1.2)"
        );

        let bare = SessionCancel::new();
        bare.cancel();
        assert!(end_on_session_cancel(&bare).is_ok());

        let absent = SessionCancel::new();
        absent.cancel_absent_from_store();
        let status = end_on_session_cancel(&absent.child())
            .expect_err("a session missing from the live store must not end the join OK");
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), session_absent_status().message());
    }

    // ── ReplayDelivered (D) and its key ─────────────────────────────────
    //
    // D keys a canonical UUID id by its 128-bit value and any other id by its
    // string. A lookup must match exactly the ids string equality matches, or
    // a different spelling of a delivered id would be suppressed as its
    // duplicate. The integration tests (`presence_replay_dedup_test`) cover D
    // inside a real join; these pin the key, the rule and `end`.

    /// One spelling of a fixed UUID.
    const UUID_LOWER: &str = "67e55044-10b1-426f-9247-bb680e5fe0c8";

    /// Record `cid` in `d` as a completed replay send does.
    fn deliver(d: &mut ReplayDelivered, cid: &str) {
        let (_peer_joined, key) = ReplayDelivered::split_for_delivery(cid.to_owned());
        d.record_delivered(key);
    }

    fn joined_by(cid: &str) -> PresenceEvent {
        PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: cid.to_owned(),
            user_name: "user".to_owned(),
            user_application: "app".to_owned(),
        })
    }

    fn left_by(cid: &str) -> PresenceEvent {
        PresenceEvent::PeerLeft {
            connection_id: cid.to_owned(),
        }
    }

    /// Only the exact lowercase hyphenated text of a UUID — the form
    /// `join_session_inner` mints — has a 128-bit key; every other spelling,
    /// including ones `Uuid::try_parse` accepts, is keyed by its string.
    #[test]
    fn delivered_uuid_key_accepts_only_the_lowercase_hyphenated_form() {
        let uuid = uuid::Uuid::parse_str(UUID_LOWER).expect("a valid UUID");
        assert_eq!(delivered_uuid_key(UUID_LOWER), Some(uuid.as_u128()));
        let minted = uuid::Uuid::new_v4();
        assert_eq!(
            delivered_uuid_key(&minted.to_string()),
            Some(minted.as_u128()),
            "a minted connection id has a 128-bit key"
        );

        let mixed_case = format!("{}{}", &UUID_LOWER[..9], UUID_LOWER[9..].to_uppercase());
        for other in [
            UUID_LOWER.to_uppercase(),
            mixed_case,
            format!("{{{UUID_LOWER}}}"),
            UUID_LOWER.replace('-', ""),
            format!("urn:uuid:{UUID_LOWER}"),
            format!(" {UUID_LOWER}"),
            format!("{UUID_LOWER} "),
            // 36 characters, hyphens misplaced.
            format!("{}-{}", UUID_LOWER.replace('-', ""), "abc"),
            "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz".to_owned(),
            "seeded-0001".to_owned(),
            String::new(),
        ] {
            assert_eq!(
                delivered_uuid_key(&other),
                None,
                "{other:?} is not the lowercase hyphenated form, so it is keyed by its string"
            );
        }
    }

    /// `split_for_delivery` copies nothing for a canonical UUID: the id's own
    /// string goes to the `PeerJoined` and D keys it by its 128-bit value.
    /// Any other id costs one copy, which goes to the `PeerJoined`, while the
    /// id's own string becomes D's key. Splitting records nothing; once
    /// `record_delivered` records each key, D suppresses each id's live
    /// duplicate, and nothing else.
    #[test]
    fn split_for_delivery_moves_a_uuid_whole_and_copies_any_other_id_once() {
        let uuid = UUID_LOWER.to_owned();
        let uuid_buffer = uuid.as_ptr();
        let other = "seeded-0001".to_owned();
        let other_buffer = other.as_ptr();
        let mut d = ReplayDelivered::with_capacity(2);

        let (uuid_peer_joined, uuid_key) = ReplayDelivered::split_for_delivery(uuid);
        let (other_peer_joined, other_key) = ReplayDelivered::split_for_delivery(other);

        assert_eq!(uuid_peer_joined, UUID_LOWER);
        assert_eq!(
            uuid_peer_joined.as_ptr(),
            uuid_buffer,
            "a UUID's own string goes to the PeerJoined, not a copy"
        );
        let expected_uuid = uuid::Uuid::parse_str(UUID_LOWER).expect("a valid UUID");
        assert!(
            matches!(uuid_key, DeliveredKey::Uuid(k) if k == expected_uuid.as_u128()),
            "a UUID is keyed by its 128-bit value"
        );

        assert_eq!(other_peer_joined, "seeded-0001");
        assert_ne!(
            other_peer_joined.as_ptr(),
            other_buffer,
            "any other id's PeerJoined carries a copy"
        );
        match &other_key {
            DeliveredKey::Other(key) => {
                assert_eq!(key, "seeded-0001");
                assert_eq!(key.as_ptr(), other_buffer, "the id's own string is D's key");
            }
            DeliveredKey::Uuid(_) => panic!("a non-UUID id is keyed by its string"),
        }
        assert_eq!(d.len(), 0, "splitting records nothing");

        d.record_delivered(uuid_key);
        d.record_delivered(other_key);

        assert_eq!(d.len(), 2);
        assert!(
            !d.suppresses(&joined_by("never-delivered")),
            "an id never recorded is forwarded"
        );
        assert!(
            d.suppresses(&joined_by(&uuid_peer_joined)),
            "the UUID's live duplicate is suppressed"
        );
        assert!(
            d.suppresses(&joined_by(&other_peer_joined)),
            "the other id's live duplicate is suppressed"
        );
    }

    /// Whatever spelling was delivered and whatever spelling arrives live, a
    /// live `PeerJoined` is suppressed exactly when the two strings are equal.
    #[test]
    fn replay_delivered_suppresses_exactly_the_ids_string_equality_matches() {
        let spellings = [
            UUID_LOWER.to_owned(),
            UUID_LOWER.to_uppercase(),
            format!("{{{UUID_LOWER}}}"),
            UUID_LOWER.replace('-', ""),
            format!("urn:uuid:{UUID_LOWER}"),
            "0e5b2b9a-2c7c-4e9f-9d3a-1b2c3d4e5f60".to_owned(),
            "seeded-0001".to_owned(),
        ];
        for delivered_id in &spellings {
            for live_id in &spellings {
                let mut d = ReplayDelivered::with_capacity(1);
                deliver(&mut d, delivered_id);
                assert_eq!(
                    d.suppresses(&joined_by(live_id)),
                    delivered_id == live_id,
                    "delivered {delivered_id:?}, live joined {live_id:?}"
                );
            }
        }
    }

    /// The rule: a suppressed `PeerJoined` takes its id out of D, so a second
    /// one is forwarded; a `PeerLeft` is forwarded and takes its id out, so
    /// the rejoin after it is forwarded; a `PeerMessage` is forwarded and
    /// leaves D as it was.
    #[test]
    fn replay_delivered_rule_suppresses_one_joined_and_forwards_leaves_and_messages() {
        let (x, y) = (UUID_LOWER, "seeded-0001");
        let mut d = ReplayDelivered::with_capacity(2);
        deliver(&mut d, x);
        deliver(&mut d, y);
        assert_eq!(d.len(), 2);

        assert!(d.suppresses(&joined_by(x)), "the replay's duplicate");
        assert!(
            !d.suppresses(&joined_by(x)),
            "a second joined has no replayed copy left to duplicate"
        );

        let message = PresenceEvent::PeerMessage {
            connection_id: y.to_owned(),
            data: vec![1],
        };
        assert!(!d.suppresses(&message), "a message is never suppressed");
        assert_eq!(d.len(), 1, "a message leaves D as it was");
        assert!(!d.suppresses(&left_by(y)), "a leave is forwarded");
        assert_eq!(d.len(), 0, "a leave takes its id out of D");
        assert!(
            !d.suppresses(&joined_by(y)),
            "the rejoin after a leave is forwarded"
        );
        assert!(d.is_live(), "emptying D does not end it");
    }

    /// `end` frees D's tables and keeps its first cause: later causes change
    /// nothing, and an ended D suppresses nothing. The seam observes the first
    /// cause only, and dropping an ended D records no loop end.
    #[test]
    fn replay_delivered_end_is_idempotent_and_keeps_its_first_cause() {
        let seam = std::sync::Arc::new(PresenceReplaySeam::new());
        let mut d = ReplayDelivered::with_capacity(2);
        deliver(&mut d, UUID_LOWER);
        deliver(&mut d, "seeded-0001");
        d.observe(std::sync::Arc::clone(&seam), "joiner");
        assert_eq!(
            seam.delivered_set("joiner"),
            Some(DeliveredSetObservation {
                created_len: 2,
                len: Some(2),
                suppressed_peer_joined: 0,
                ended_by: None,
            })
        );

        d.end(DeliveredEnd::OwnEvent);
        assert!(!d.is_live());
        assert_eq!(d.ended, Some(DeliveredEnd::OwnEvent));
        assert_eq!(d.len(), 0);
        assert_eq!(
            (d.uuids.capacity(), d.others.capacity()),
            (0, 0),
            "an ended D holds no table"
        );

        d.end(DeliveredEnd::Lag);
        d.end(DeliveredEnd::TimeLimit);
        assert_eq!(
            d.ended,
            Some(DeliveredEnd::OwnEvent),
            "the first cause is D's end"
        );
        assert!(
            !d.suppresses(&joined_by(UUID_LOWER)),
            "an ended D suppresses nothing, a delivered id included"
        );

        let ended = Some(DeliveredSetObservation {
            created_len: 2,
            len: None,
            suppressed_peer_joined: 0,
            ended_by: Some(DeliveredSetEnd::OwnEvent),
        });
        assert_eq!(seam.delivered_set("joiner"), ended);
        drop(d);
        assert_eq!(
            seam.delivered_set("joiner"),
            ended,
            "dropping an ended D records no loop end"
        );
    }

    /// A D still live when it is dropped — the bidi loop ended first — is
    /// observed as ended by the loop's end.
    #[test]
    fn replay_delivered_dropped_while_live_is_observed_ended_by_the_loop_end() {
        let seam = std::sync::Arc::new(PresenceReplaySeam::new());
        let mut d = ReplayDelivered::with_capacity(1);
        deliver(&mut d, UUID_LOWER);
        d.observe(std::sync::Arc::clone(&seam), "joiner");
        assert!(d.suppresses(&joined_by(UUID_LOWER)));
        drop(d);
        assert_eq!(
            seam.delivered_set("joiner"),
            Some(DeliveredSetObservation {
                created_len: 1,
                len: None,
                suppressed_peer_joined: 1,
                ended_by: Some(DeliveredSetEnd::LoopEnd),
            })
        );
    }

    /// `reset_observations` clears the observation of a D that already exists,
    /// and that D reports nothing afterwards: a suppression, its end and its
    /// drop each find no observation to update, and none re-creates one.
    #[test]
    fn replay_delivered_observed_before_a_reset_reports_nothing_after_it() {
        let seam = std::sync::Arc::new(PresenceReplaySeam::new());
        let mut d = ReplayDelivered::with_capacity(1);
        deliver(&mut d, UUID_LOWER);
        d.observe(std::sync::Arc::clone(&seam), "joiner");
        assert!(
            seam.delivered_set("joiner").is_some(),
            "precondition: the D is observed"
        );

        seam.reset_observations();
        assert_eq!(
            seam.delivered_set("joiner"),
            None,
            "a reset clears the observation of a D that already exists"
        );

        assert!(
            d.suppresses(&joined_by(UUID_LOWER)),
            "the D itself is unaffected by the reset"
        );
        assert_eq!(
            seam.delivered_set("joiner"),
            None,
            "a suppression after the reset records nothing"
        );
        d.end(DeliveredEnd::OwnEvent);
        assert_eq!(
            seam.delivered_set("joiner"),
            None,
            "an end after the reset records nothing"
        );
        drop(d);
        assert_eq!(
            seam.delivered_set("joiner"),
            None,
            "a drop after the reset records nothing"
        );
    }

    // ── PresenceEventTally ──────────────────────────────────────────────
    //
    // The tally batches `live_presence_events_total` per stream. Batching may
    // delay a delivery's arrival on the counter; it must never lose one, count
    // one twice, or credit it to the wrong `event_type`.

    /// `(joined, left, message)` as served.
    fn presence_events(metrics: &Metrics) -> (u64, u64, u64) {
        let text = render(metrics);
        let read = |event_type: &str| {
            sample_value(
                &text,
                "live_presence_events_total",
                &[("event_type", event_type)],
            )
            .expect("every event_type cell is seeded in Metrics::new") as u64
        };
        (
            read(PRESENCE_EVENT_JOINED),
            read(PRESENCE_EVENT_LEFT),
            read(PRESENCE_EVENT_MESSAGE),
        )
    }

    fn joined() -> PresenceEvent {
        PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: "peer".to_owned(),
            user_name: "user".to_owned(),
            user_application: "app".to_owned(),
        })
    }

    fn left() -> PresenceEvent {
        PresenceEvent::PeerLeft {
            connection_id: "peer".to_owned(),
        }
    }

    fn message() -> PresenceEvent {
        PresenceEvent::PeerMessage {
            connection_id: "peer".to_owned(),
            data: vec![1],
        }
    }

    /// Below the threshold nothing reaches the counter; the delivery that
    /// reaches it — counted across all event types — publishes the whole
    /// batch, each type on its own cell.
    #[test]
    fn presence_tally_flushes_every_type_at_the_shared_threshold() {
        let metrics = metrics_scrape_only("presence-tally");
        let mut tally = PresenceEventTally::new(&metrics);
        let (j, l) = (10, 20);
        let m = PRESENCE_TALLY_FLUSH_AT - j - l;
        for _ in 0..j {
            tally.record(&joined());
        }
        for _ in 0..l {
            tally.record(&left());
        }
        for _ in 0..m - 1 {
            tally.record(&message());
        }
        assert!(tally.has_pending());
        assert_eq!(
            presence_events(&metrics),
            (0, 0, 0),
            "below the threshold every delivery is still pending"
        );
        tally.record(&message());
        assert_eq!(
            presence_events(&metrics),
            (j, l, m),
            "the threshold delivery flushes every type, each onto its own event_type cell"
        );
        assert!(!tally.has_pending(), "a flush empties every slot");
    }

    /// An explicit flush publishes what is pending, a second adds nothing, and
    /// `Drop` publishes only what was recorded after the last flush: the timer
    /// flush followed by the stream-end `Drop` must not count a delivery twice.
    #[test]
    fn presence_tally_flush_then_drop_counts_each_delivery_once() {
        let metrics = metrics_scrape_only("presence-tally");
        {
            let mut tally = PresenceEventTally::new(&metrics);
            tally.record(&joined());
            tally.record(&message());
            tally.record(&message());
            tally.flush();
            assert_eq!(presence_events(&metrics), (1, 0, 2));
            tally.flush();
            assert_eq!(
                presence_events(&metrics),
                (1, 0, 2),
                "an empty flush adds nothing"
            );
            tally.record(&left());
            tally.record(&message());
        }
        assert_eq!(
            presence_events(&metrics),
            (1, 1, 3),
            "Drop must publish the deliveries recorded after the last flush, and only those"
        );
    }

    /// `run_bidi_loop` leaves by `return` — an error exit included — and the
    /// tally's `Drop` is the only flush on that path, so deliveries pending
    /// when the loop errors out must still be published.
    #[test]
    fn presence_tally_drop_publishes_pending_deliveries_on_an_error_exit() {
        fn loop_that_errors(metrics: &Metrics) -> Result<(), Status> {
            let mut tally = PresenceEventTally::new(metrics);
            tally.record(&joined());
            tally.record(&left());
            tally.record(&message());
            Err(Status::internal("the stream failed"))
        }
        let metrics = metrics_scrape_only("presence-tally");
        assert!(loop_that_errors(&metrics).is_err());
        assert_eq!(presence_events(&metrics), (1, 1, 1));
    }

    // ── Lease keeper failure counting ───────────────────────────────────

    /// `live_edit_presence_lease_refresh_failures_total` for `metrics`.
    fn lease_refresh_failures(metrics: &Metrics, service: &str) -> f64 {
        sample_value(
            &render(metrics),
            "live_edit_presence_lease_refresh_failures_total",
            &[("service", service)],
        )
        .expect("the lease refresh failure cell is seeded in Metrics::new")
    }

    /// The lease TTL the keeper tests register with and renew to.
    const KEEPER_TEST_LEASE_TTL_SECS: u64 = 30;

    /// A lease TTL planted so a later read tells a renewal (back near
    /// [`KEEPER_TEST_LEASE_TTL_SECS`]) from no renewal.
    const KEEPER_TEST_SHORTENED_SECS: i64 = 5;

    /// The lowest PTTL a lease just renewed to [`KEEPER_TEST_LEASE_TTL_SECS`]
    /// may read back, with slack for runner load; far above the planted TTL.
    const KEEPER_TEST_RENEWED_FLOOR_MS: i64 = 20_000;

    /// Register `cid` in `sid` the way a join does, with a deadline of the
    /// store's clock plus one lease TTL.
    async fn register_for_keeper(
        tv: &crate::storage::test_support::TestValkey,
        sid: &str,
        cid: &str,
    ) {
        let origin = FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            crate::storage::store_oom::StoreOomSignal::unwired(),
        );
        let now_ms = store_time_ms(tv.client(), origin, tv.key_schema(), sid)
            .await
            .expect("store_time_ms must succeed");
        let joined = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: cid.to_owned(),
            user_name: "user".to_owned(),
            user_application: "app".to_owned(),
        })
        .encode();
        let outcome = register_connection(
            tv.client(),
            origin,
            tv.key_schema(),
            sid,
            cid,
            "user",
            "app",
            KEEPER_TEST_LEASE_TTL_SECS,
            now_ms + KEEPER_TEST_LEASE_TTL_SECS * 1000,
            &joined,
        )
        .await
        .expect("register_connection must succeed");
        assert_eq!(outcome, RegisterOutcome::Registered { added: true });
    }

    /// Shorten `lease_key` to [`KEEPER_TEST_SHORTENED_SECS`] through the
    /// maintenance pool, so the write never queues behind a data-pool stall.
    async fn shorten_lease(tv: &crate::storage::test_support::TestValkey, lease_key: &str) {
        use fred::interfaces::KeysInterface;
        let shortened: bool = tv
            .client()
            .maintenance_pool
            .next()
            .expire(lease_key, KEEPER_TEST_SHORTENED_SECS, None)
            .await
            .expect("EXPIRE must succeed");
        assert!(shortened, "the registered lease must exist to be shortened");
    }

    async fn lease_pttl_ms(tv: &crate::storage::test_support::TestValkey, lease_key: &str) -> i64 {
        use fred::interfaces::KeysInterface;
        tv.client()
            .maintenance_pool
            .next()
            .pttl(lease_key)
            .await
            .expect("PTTL must succeed")
    }

    /// A lease keeper refresh that gets no reply within the command timeout
    /// counts exactly one `valkey_errors_total{pool="data",
    /// kind="command_timeout"}` and exactly one
    /// `live_edit_presence_lease_refresh_failures_total` on the metrics handle
    /// the keeper was given, and the keeper goes on renewing the lease. The
    /// connection is registered before the refresh script is held behind a
    /// server-side `BLPOP` on every data-pool connection, so the `Timeout` is
    /// fred's real one and every refresh that is answered renews.
    ///
    /// The stalls are released as soon as the timeout is counted. At most two
    /// refreshes sat behind them — the failed step's, which still runs once
    /// released, and the step that followed at once on the missed tick, which
    /// is answered well inside its own deadline — and either may renew the
    /// lease after the release. So the lease is shortened and its renewal
    /// awaited three times, which includes at least one renewal by a keeper
    /// step issued after the release, before the keeper is stopped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_timed_out_lease_refresh_counts_one_data_command_timeout_and_one_keeper_failure() {
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, release_all, TestValkey,
        };
        /// First refresh at 1 s plus the fixture's 5 s command timeout, with
        /// slack; shorter than a second refresh's own timeout would take.
        const REFRESH_TICK_WAIT: std::time::Duration = std::time::Duration::from_secs(9);
        /// Bound on each awaited renewal: one refresh interval, with slack.
        const RENEWAL_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

        const SERVICE: &str = "presence-refresh";
        const SID: &str = "stalled-session";
        const CID: &str = "stalled-connection";

        let tv = TestValkey::setup().await;
        let metrics: crate::observability::metrics::SharedMetrics =
            std::sync::Arc::new(metrics_scrape_only(SERVICE));
        assert_eq!(lease_refresh_failures(&metrics, SERVICE), 0.0);
        register_for_keeper(&tv, SID, CID).await;
        let lease_key = tv.key_schema().presence_lease(SID, CID);

        let stalls = tv.stall_data_pool().await;
        let cancel = CancellationToken::new();
        let tracker = tokio_util::task::TaskTracker::new();
        let keeper = spawn_lease_keeper(
            &tracker,
            LeaseKeeper {
                valkey: tv.client().clone(),
                key_schema: tv.key_schema().clone(),
                metrics: std::sync::Arc::clone(&metrics),
                presence_router: std::sync::Arc::new(LocalPresenceRouter::new()),
                session_id: SID.to_owned(),
                connection_id: CID.to_owned(),
                user_name: "user".to_owned(),
                user_application: "app".to_owned(),
                lease_ttl_secs: KEEPER_TEST_LEASE_TTL_SECS,
                refresh_interval: std::time::Duration::from_secs(1),
                teardown: cancel.clone(),
                session_cancel: CancellationToken::new(),
                shutdown: CancellationToken::new(),
                seam: None,
            },
        );
        let deadline = tokio::time::Instant::now() + REFRESH_TICK_WAIT;
        while command_timeout_row(&metrics) == command_timeouts_only_on("data", 0)
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        release_all(stalls).await;

        for round in 1..=3 {
            shorten_lease(&tv, &lease_key).await;
            let renewal_deadline = tokio::time::Instant::now() + RENEWAL_WAIT;
            while lease_pttl_ms(&tv, &lease_key).await <= KEEPER_TEST_RENEWED_FLOOR_MS {
                assert!(
                    tokio::time::Instant::now() < renewal_deadline,
                    "renewal round {round}: the keeper must renew the lease after the failed \
                     step"
                );
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(10), keeper)
            .await
            .expect("the keeper must stop once its teardown token is cancelled")
            .expect("the keeper must not panic");

        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "the timed-out refresh must count once, on the data pool"
        );
        assert_eq!(
            lease_refresh_failures(&metrics, SERVICE),
            1.0,
            "the failed keeper step must count once as a lease refresh failure"
        );
    }

    /// A keeper step whose refresh renews the lease but whose index re-add
    /// after it fails counts one lease refresh failure and does not stop the
    /// keeper. The re-add is made to fail by planting a string at the session
    /// index key, so its `SADD` answers `WRONGTYPE` (the refresh script does
    /// not touch the index). The step is driven directly, once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_keeper_step_whose_index_re_add_fails_counts_one_failure_and_continues() {
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, TestValkey,
        };
        use fred::interfaces::KeysInterface;

        const SERVICE: &str = "presence-reindex";
        const SID: &str = "reindex-session";
        const CID: &str = "reindex-connection";

        let tv = TestValkey::setup().await;
        let metrics: crate::observability::metrics::SharedMetrics =
            std::sync::Arc::new(metrics_scrape_only(SERVICE));
        register_for_keeper(&tv, SID, CID).await;
        let lease_key = tv.key_schema().presence_lease(SID, CID);
        let _: () = tv
            .client()
            .pool
            .next()
            .set(
                tv.key_schema().presence_active_sessions(),
                "not-a-set",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string at the index key must succeed");
        shorten_lease(&tv, &lease_key).await;

        let keeper = LeaseKeeper {
            valkey: tv.client().clone(),
            key_schema: tv.key_schema().clone(),
            metrics: std::sync::Arc::clone(&metrics),
            presence_router: std::sync::Arc::new(LocalPresenceRouter::new()),
            session_id: SID.to_owned(),
            connection_id: CID.to_owned(),
            user_name: "user".to_owned(),
            user_application: "app".to_owned(),
            lease_ttl_secs: KEEPER_TEST_LEASE_TTL_SECS,
            refresh_interval: std::time::Duration::from_secs(1),
            teardown: CancellationToken::new(),
            session_cancel: CancellationToken::new(),
            shutdown: CancellationToken::new(),
            seam: None,
        };
        assert_eq!(
            keeper.step(tokio::time::Instant::now()).await,
            ControlFlow::Continue(()),
            "a failed index re-add must not stop the keeper"
        );

        assert!(
            lease_pttl_ms(&tv, &lease_key).await > KEEPER_TEST_RENEWED_FLOOR_MS,
            "the step's refresh must have renewed the lease before the re-add failed"
        );
        assert_eq!(
            lease_refresh_failures(&metrics, SERVICE),
            1.0,
            "the failed re-add must count once as a lease refresh failure"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 0),
            "a WRONGTYPE reply is not a command timeout"
        );
    }

    // ── Renewal index re-add gate (SDD SR-7.2b) ─────────────────────────
    //
    // Keepers of one session's local connections share the replica's router,
    // and a renewal re-adds the session to the session index only when
    // `LocalPresenceRouter::index_renewal_due` says so. Most tests drive each
    // step directly at an explicit tick, so which step re-adds is exact, with
    // no timing involved. A step's re-add is observed by removing the session
    // from the index just before the step and reading it back after: the
    // refresh and register scripts never touch the index, so the session is
    // indexed after the step iff the step issued its `SADD`.
    //
    // The last tests cover the keeper's own loop (`LeaseKeeper::run`), run and
    // parked at its seams: which instant `run` hands a step, and when its
    // first renewal comes (with `first_renewal_delay` itself).

    /// The refresh interval the gated keepers are built with. Steps driven at
    /// explicit ticks never wait it out; the `run` tests do.
    const GATE_TEST_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

    /// Local connections of one session on one replica: each registered in the
    /// store and joined to the shared router, with a keeper sharing that
    /// router.
    struct GatedKeepers {
        tv: crate::storage::test_support::TestValkey,
        router: std::sync::Arc<LocalPresenceRouter>,
        /// The keepers' shared metrics, served under [`GATE_SERVICE`].
        metrics: SharedMetrics,
        keepers: Vec<LeaseKeeper>,
        _receivers: Vec<PresenceReceiver>,
    }

    /// The session every gated-keeper test uses; each test has its own key
    /// prefix.
    const GATE_SID: &str = "gated-session";

    /// The `service` label of the gated keepers' metrics.
    const GATE_SERVICE: &str = "presence-gate";

    async fn gated_keepers(cids: &[&str]) -> GatedKeepers {
        let tv = crate::storage::test_support::TestValkey::setup().await;
        let router = std::sync::Arc::new(LocalPresenceRouter::new());
        let metrics: SharedMetrics = std::sync::Arc::new(metrics_scrape_only(GATE_SERVICE));
        let mut keepers = Vec::with_capacity(cids.len());
        let mut receivers = Vec::with_capacity(cids.len());
        for cid in cids {
            register_for_keeper(&tv, GATE_SID, cid).await;
            receivers.push(router.subscribe(GATE_SID).0);
            keepers.push(LeaseKeeper {
                valkey: tv.client().clone(),
                key_schema: tv.key_schema().clone(),
                metrics: std::sync::Arc::clone(&metrics),
                presence_router: std::sync::Arc::clone(&router),
                session_id: GATE_SID.to_owned(),
                connection_id: (*cid).to_owned(),
                user_name: "user".to_owned(),
                user_application: "app".to_owned(),
                lease_ttl_secs: KEEPER_TEST_LEASE_TTL_SECS,
                refresh_interval: GATE_TEST_INTERVAL,
                teardown: CancellationToken::new(),
                session_cancel: CancellationToken::new(),
                shutdown: CancellationToken::new(),
                seam: None,
            });
        }
        GatedKeepers {
            tv,
            router,
            metrics,
            keepers,
            _receivers: receivers,
        }
    }

    impl GatedKeepers {
        async fn is_indexed(&self) -> bool {
            self.tv
                .client()
                .maintenance_pool
                .next()
                .sismember(self.tv.key_schema().presence_active_sessions(), GATE_SID)
                .await
                .expect("SISMEMBER active_sessions must succeed")
        }

        /// Run keeper `i`'s step at `tick` with the session first removed from
        /// the index; returns whether the step re-added it.
        async fn step_re_adds(&self, i: usize, tick: tokio::time::Instant) -> bool {
            let _: i64 = self
                .tv
                .client()
                .maintenance_pool
                .next()
                .srem(self.tv.key_schema().presence_active_sessions(), GATE_SID)
                .await
                .expect("SREM active_sessions must succeed");
            assert!(
                !self.is_indexed().await,
                "precondition: the session is out of the index before the step"
            );
            assert_eq!(
                self.keepers[i].step(tick).await,
                ControlFlow::Continue(()),
                "keeper {i}'s step must not stop the keeper"
            );
            self.is_indexed().await
        }
    }

    /// Several local connections of one session, their keepers ticking at
    /// fixed phases inside each interval, over several intervals: in every
    /// interval exactly one renewal re-adds the session — the interval's first
    /// — and every other renewal skips its `SADD`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn renewal_index_re_add_is_issued_once_per_session_per_interval_across_local_connections()
    {
        const CIDS: [&str; 3] = ["c0", "c1", "c2"];
        const INTERVALS: u32 = 4;
        let fx = gated_keepers(&CIDS).await;
        let base = tokio::time::Instant::now();

        for interval in 0..INTERVALS {
            let mut re_added_by = Vec::new();
            for i in 0..CIDS.len() {
                // Phases 0, 1/4 and 1/2 of an interval: each keeper's ticks
                // are exactly one interval apart, as its `tokio` interval's are.
                let tick = base + GATE_TEST_INTERVAL * interval + GATE_TEST_INTERVAL / 4 * i as u32;
                if fx.step_re_adds(i, tick).await {
                    re_added_by.push(i);
                }
            }
            assert_eq!(
                re_added_by,
                vec![0],
                "interval {interval}: only the interval's first renewal may re-add the session"
            );
        }
    }

    /// A session's only local connection re-adds the session on every renewal:
    /// its ticks are exactly one interval apart, so the gate never skips it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn renewal_index_re_add_of_a_sessions_only_local_connection_happens_on_every_renewal() {
        const RENEWALS: u32 = 4;
        let fx = gated_keepers(&["sole"]).await;
        let base = tokio::time::Instant::now();

        for k in 0..RENEWALS {
            assert!(
                fx.step_re_adds(0, base + GATE_TEST_INTERVAL * k).await,
                "renewal {k} of the session's only local connection must re-add the session"
            );
        }
    }

    /// When the connection whose renewal re-added the session leaves, a
    /// sibling's first renewal at or after one interval past that re-add takes
    /// the re-add over; the record stayed (the leave was not the last), so a
    /// renewal before that point still skips.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn renewal_index_re_add_passes_to_a_sibling_once_the_recording_connection_leaves() {
        let fx = gated_keepers(&["c0", "c1", "c2"]).await;
        let base = tokio::time::Instant::now();
        assert!(fx.step_re_adds(0, base).await, "precondition: c0 re-adds");
        assert!(
            !fx.step_re_adds(1, base + GATE_TEST_INTERVAL / 4).await,
            "precondition: c1 skips inside c0's interval"
        );

        assert!(
            !fx.router.unsubscribe(GATE_SID),
            "precondition: c0's leave is not the last"
        );
        assert!(
            !fx.step_re_adds(2, base + GATE_TEST_INTERVAL / 2).await,
            "c2's renewal inside the recorded interval must still skip after c0 left"
        );
        assert!(
            fx.step_re_adds(1, base + GATE_TEST_INTERVAL + GATE_TEST_INTERVAL / 4)
                .await,
            "c1's next renewal, past one interval since c0's re-add, must re-add the session"
        );
        assert!(
            !fx.step_re_adds(2, base + GATE_TEST_INTERVAL + GATE_TEST_INTERVAL / 2)
                .await,
            "c2's renewal inside c1's new interval must skip"
        );
    }

    /// A renewal re-add whose `SADD` fails still takes the interval: a
    /// sibling's renewal inside it skips its re-add, and the retry is the
    /// failing keeper's own renewal one interval later. The `SADD` is made to
    /// fail by planting a string at the session index key (`WRONGTYPE`), which
    /// is deleted again before the sibling's step.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn renewal_index_re_add_that_fails_still_takes_the_interval_and_its_keeper_retries_next_renewal(
    ) {
        use fred::interfaces::KeysInterface;
        let fx = gated_keepers(&["c0", "c1"]).await;
        let index_key = fx.tv.key_schema().presence_active_sessions();
        let _: () = fx
            .tv
            .client()
            .maintenance_pool
            .next()
            .set(&index_key, "not-a-set", None, None, false)
            .await
            .expect("planting a string at the index key must succeed");
        let base = tokio::time::Instant::now();

        assert_eq!(
            fx.keepers[0].step(base).await,
            ControlFlow::Continue(()),
            "precondition: a failed re-add does not stop the keeper"
        );
        assert_eq!(
            lease_refresh_failures(&fx.metrics, GATE_SERVICE),
            1.0,
            "precondition: c0's renewal issued its re-add, and the re-add failed"
        );
        let deleted: i64 = fx
            .tv
            .client()
            .maintenance_pool
            .next()
            .del(&index_key)
            .await
            .expect("DEL of the planted string must succeed");
        assert_eq!(deleted, 1, "precondition: the planted string existed");

        assert!(
            !fx.step_re_adds(1, base + GATE_TEST_INTERVAL / 4).await,
            "c1's renewal inside the interval must skip: c0's failed re-add still took it"
        );
        assert!(
            fx.step_re_adds(0, base + GATE_TEST_INTERVAL).await,
            "c0's next renewal, one interval later, must retry the re-add"
        );
    }

    /// How a gated keeper's connection is made to need re-registering.
    #[derive(Clone, Copy, Debug)]
    enum NeedsRegister {
        /// Evicted from the member set: the refresh reports `Evicted`, and the
        /// rejoin's register re-adds the membership (`added: true`).
        Evicted,
        /// Info hash deleted: the refresh reports `InfoMissing`, and the repair
        /// register finds it still a member (`added: false`).
        InfoDeleted,
    }

    /// Connection c0's renewal records the interval, and c1's plain renewal in
    /// that interval skips its re-add. c1 is then made to re-register
    /// (`found`): its rejoin or info-repair register re-adds the session to
    /// the index in the same interval, because only the plain renewal's re-add
    /// is gated.
    async fn re_register_re_add_is_not_gated_by_a_siblings_record(found: NeedsRegister) {
        let fx = gated_keepers(&["c0", "c1"]).await;
        let base = tokio::time::Instant::now();
        assert!(
            fx.step_re_adds(0, base).await,
            "{found:?}: precondition: c0's renewal re-adds and records the interval"
        );
        assert!(
            !fx.step_re_adds(1, base + GATE_TEST_INTERVAL / 4).await,
            "{found:?}: precondition: c1's plain renewal in c0's interval skips its re-add"
        );

        let origin = FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            crate::storage::store_oom::StoreOomSignal::unwired(),
        );
        let ks = fx.tv.key_schema();
        match found {
            NeedsRegister::Evicted => {
                let eviction = crate::storage::presence::evict_connection_for_test(
                    fx.tv.client(),
                    origin,
                    ks,
                    GATE_SID,
                    "c1",
                    &PresenceEvent::PeerLeft {
                        connection_id: "c1".to_owned(),
                    }
                    .encode(),
                )
                .await
                .expect("the eviction must succeed");
                assert!(eviction.removed, "precondition: c1 was evicted");
            }
            NeedsRegister::InfoDeleted => {
                use fred::interfaces::KeysInterface;
                let deleted: i64 = fx
                    .tv
                    .client()
                    .maintenance_pool
                    .next()
                    .del(ks.presence_user(GATE_SID, "c1"))
                    .await
                    .expect("DEL info hash must succeed");
                assert_eq!(deleted, 1, "precondition: c1's info hash existed");
            }
        }

        assert!(
            fx.step_re_adds(1, base + GATE_TEST_INTERVAL / 2).await,
            "{found:?}: c1's re-register must re-add the session although c0 recorded the interval"
        );
        let members: Vec<String> = fx
            .tv
            .client()
            .maintenance_pool
            .next()
            .smembers(ks.presence_members(GATE_SID))
            .await
            .expect("SMEMBERS must succeed");
        assert!(
            members.iter().any(|m| m == "c1"),
            "{found:?}: the step must have re-registered c1; members are {members:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rejoin_index_re_add_is_not_gated_by_a_siblings_renewal_in_the_same_interval() {
        re_register_re_add_is_not_gated_by_a_siblings_record(NeedsRegister::Evicted).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn info_repair_index_re_add_is_not_gated_by_a_siblings_renewal_in_the_same_interval() {
        re_register_re_add_is_not_gated_by_a_siblings_record(NeedsRegister::InfoDeleted).await;
    }

    /// Bound on each wait for a parked keeper or a stopped one; unreachable on
    /// the healthy path.
    const RUN_TEST_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

    /// Take the fixture's keepers out to be spawned, each with its own seam.
    fn seamed_keepers(
        fx: &mut GatedKeepers,
    ) -> Vec<(LeaseKeeper, std::sync::Arc<LeaseKeeperSeam>)> {
        std::mem::take(&mut fx.keepers)
            .into_iter()
            .map(|mut keeper| {
                let seam = std::sync::Arc::new(LeaseKeeperSeam::new());
                keeper.seam = Some(std::sync::Arc::clone(&seam));
                (keeper, seam)
            })
            .collect()
    }

    async fn parked(gate: &PresenceSeamGate, what: &str) {
        tokio::time::timeout(RUN_TEST_WAIT, gate.wait_until_parked())
            .await
            .unwrap_or_else(|_| panic!("{what} did not park within {RUN_TEST_WAIT:?}"));
    }

    /// `run` hands a step the scheduled instant of its tick, not the time it
    /// calls the step. The two differ for a tick the keeper missed: with
    /// `MissedTickBehavior::Delay`, `interval.tick()` returns the missed tick's
    /// scheduled instant at once, whenever it is called.
    ///
    /// The keeper's first step is a rejoin (the connection is evicted before
    /// spawn), parked before its register at instant `P`: its scheduled tick
    /// `t1` is at most `P`, so the next tick `t2` is at most `P + I`. A renewal
    /// re-add is recorded on the router at `P + I/5`, and the parked step is
    /// held until `P + 3I/2`, past `t2`. The second step, parked before its
    /// refresh, is therefore the missed tick `t2`, called at least `3I/2`
    /// after `P`. The session is removed from the index and the keeper stopped
    /// after that step. Handed `t2`, its renewal is at most `4I/5` after the
    /// record and skips its re-add; handed the call time, it would be at least
    /// `13I/10` after it and re-add. Every bound holds by construction, however
    /// late the runner schedules anything.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_hands_a_missed_tick_its_scheduled_instant_so_its_renewal_is_gated_by_it() {
        let mut fx = gated_keepers(&["a"]).await;
        let (keeper, seam) = seamed_keepers(&mut fx).pop().expect("one keeper");
        let stop = keeper.teardown.clone();
        let eviction = crate::storage::presence::evict_connection_for_test(
            fx.tv.client(),
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            fx.tv.key_schema(),
            GATE_SID,
            "a",
            &PresenceEvent::PeerLeft {
                connection_id: "a".to_owned(),
            }
            .encode(),
        )
        .await
        .expect("the eviction must succeed");
        assert!(eviction.removed, "precondition: the connection was evicted");

        seam.before_rejoin_register.arm();
        let tracker = tokio_util::task::TaskTracker::new();
        let handle = spawn_lease_keeper(&tracker, keeper);
        parked(
            &seam.before_rejoin_register,
            "the first step's rejoin, before its register",
        )
        .await;
        let parked_at = tokio::time::Instant::now();
        assert!(
            fx.router.index_renewal_due(
                GATE_SID,
                parked_at + GATE_TEST_INTERVAL / 5,
                GATE_TEST_INTERVAL
            ),
            "precondition: recording a renewal re-add on the router"
        );
        seam.before_refresh.arm();
        // The lateness under test, not a synchronisation: the next tick is
        // missed while the first step is held.
        tokio::time::sleep_until(parked_at + GATE_TEST_INTERVAL * 3 / 2).await;
        seam.before_rejoin_register.release();

        parked(&seam.before_refresh, "the second step, before its refresh").await;
        let _: i64 = fx
            .tv
            .client()
            .maintenance_pool
            .next()
            .srem(fx.tv.key_schema().presence_active_sessions(), GATE_SID)
            .await
            .expect("SREM active_sessions must succeed");
        assert!(
            !fx.is_indexed().await,
            "precondition: the session is out of the index before the second step's renewal"
        );
        stop.cancel();
        seam.before_refresh.release();
        tokio::time::timeout(RUN_TEST_WAIT, handle)
            .await
            .expect("the keeper must stop after the second step")
            .expect("the keeper must not panic");

        assert!(
            !fx.is_indexed().await,
            "the missed tick's renewal must be gated by its scheduled instant, which is inside the \
             recorded interval"
        );
    }

    /// `first_renewal_delay` stays within [interval/2, interval] and spreads
    /// over the range, for intervals of several sizes, down to one nanosecond.
    #[test]
    fn first_renewal_delay_is_between_half_an_interval_and_one_interval() {
        const DRAWS: usize = 10_000;
        for interval in [
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(7),
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(15),
            std::time::Duration::from_millis(15_001),
        ] {
            let draws: Vec<std::time::Duration> =
                (0..DRAWS).map(|_| first_renewal_delay(interval)).collect();
            for d in &draws {
                assert!(
                    interval / 2 <= *d && *d <= interval,
                    "a delay for interval {interval:?} must be in [interval/2, interval]; got {d:?}"
                );
            }
            if interval >= std::time::Duration::from_secs(1) {
                let (min, max) = (
                    draws.iter().min().expect("draws"),
                    draws.iter().max().expect("draws"),
                );
                assert!(
                    *min < interval / 2 + interval / 8 && *max > interval - interval / 8,
                    "{DRAWS} delays for interval {interval:?} must spread over the range; got \
                     {min:?}..={max:?}"
                );
            }
        }
    }

    /// `run`'s first renewal comes after `first_renewal_delay`, so no keeper
    /// steps before half an interval after its spawn, and not every keeper
    /// waits the whole interval, as a keeper whose first renewal came one
    /// interval after spawn would. Each keeper's first step parks before its
    /// refresh, and the time a waiter sees it parked is an upper bound on the
    /// step's start.
    ///
    /// The lower bound holds by construction: a step seen parked at `s` started
    /// by `s`, and a step starts no earlier than half an interval after spawn.
    /// The other bound fails only if every one of [`KEEPERS`] uniform draws
    /// lands within the scheduling delay of the interval's end: even with a
    /// 50 ms delay on the 1 s interval, 0.1 per draw and 10⁻¹⁰ for all ten.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_first_renewal_comes_between_half_an_interval_and_one_interval_after_spawn() {
        const KEEPERS: usize = 10;
        let cids: Vec<String> = (0..KEEPERS).map(|i| format!("k{i}")).collect();
        let cid_refs: Vec<&str> = cids.iter().map(String::as_str).collect();
        let mut fx = gated_keepers(&cid_refs).await;
        let tracker = tokio_util::task::TaskTracker::new();
        let mut running = Vec::with_capacity(KEEPERS);
        for (keeper, seam) in seamed_keepers(&mut fx) {
            seam.before_refresh.arm();
            let stop = keeper.teardown.clone();
            let spawned_at = tokio::time::Instant::now();
            let handle = spawn_lease_keeper(&tracker, keeper);
            running.push((spawned_at, seam, stop, handle));
        }

        // Every keeper watched at once, so each is seen parked promptly.
        let seen_after_spawn = futures::future::join_all(running.iter().enumerate().map(
            |(i, (spawned_at, seam, _, _))| async move {
                parked(&seam.before_refresh, &format!("keeper {i}'s first step")).await;
                tokio::time::Instant::now() - *spawned_at
            },
        ))
        .await;
        for (i, (_, seam, stop, handle)) in running.into_iter().enumerate() {
            stop.cancel();
            seam.before_refresh.release();
            tokio::time::timeout(RUN_TEST_WAIT, handle)
                .await
                .unwrap_or_else(|_| panic!("keeper {i} must stop after its first step"))
                .expect("the keeper must not panic");
        }

        for (i, seen) in seen_after_spawn.iter().enumerate() {
            assert!(
                *seen >= GATE_TEST_INTERVAL / 2,
                "keeper {i}'s first step must not start before half an interval after spawn; \
                 seen parked {seen:?} after spawn"
            );
        }
        let earliest = seen_after_spawn.iter().min().expect("keepers");
        assert!(
            *earliest < GATE_TEST_INTERVAL,
            "some keeper's first step must come before one whole interval after spawn; the \
             earliest was seen {earliest:?} after spawn"
        );
    }

    // ── Client keepalive (SDD §13.2) ─────────────────────────────────────
    //
    // The arming rule, the pure expiry rule, the running deadline under a
    // paused clock, and the ending's status. The integration tests
    // (`tests/integration/grpc/presence_keepalive_test.rs`) cover the rule
    // inside a real join: the bidi loop's arms, the blocked send, the replay.

    /// A request body's progress record, with the body that stamps it: each
    /// `frame()` of the body yields one non-empty DATA frame, stamped at
    /// tokio's `now`.
    fn stamping_body() -> (
        impl http_body::Body<Data = bytes::Bytes, Error = std::convert::Infallible> + Unpin,
        RequestBodyProgress,
    ) {
        let frames: Vec<Result<http_body::Frame<bytes::Bytes>, std::convert::Infallible>> = (0..16)
            .map(|_| Ok(http_body::Frame::data(bytes::Bytes::from_static(b"k"))))
            .collect();
        let (body, _end, progress) = crate::grpc::request_end::attest_request(
            http_body_util::StreamBody::new(futures::stream::iter(frames)),
        );
        (body, progress)
    }

    /// A progress record no DATA frame has stamped.
    fn unstamped_progress() -> RequestBodyProgress {
        let (_body, _end, progress) = crate::grpc::request_end::attest_request(());
        progress
    }

    /// A join is armed only when all three hold: its header set
    /// `client_keepalive`, the configured interval is not 0 (off), and the
    /// request carries its body's progress record. An armed join keeps the
    /// configured interval (what `SessionConnected` advertises) and is held
    /// to [`CLIENT_KEEPALIVE_DEADLINE_INTERVALS`] (3, the client contract) ×
    /// that interval.
    #[test]
    fn join_keepalive_arms_only_an_opted_in_join_with_an_interval_and_a_progress_record() {
        assert_eq!(
            CLIENT_KEEPALIVE_DEADLINE_INTERVALS, 3,
            "the proto's JoinSession comment promises clients three intervals"
        );

        let armed = JoinKeepalive::arm(true, 1_000, Some(unstamped_progress()))
            .expect("an opted-in join with an interval and a progress record is armed");
        assert_eq!(armed.interval_ms, 1_000, "the advertised interval");
        assert_eq!(
            armed.deadline,
            Duration::from_secs(3),
            "the deadline is three intervals"
        );

        for (client_keepalive, interval_ms, progress, why) in [
            (false, 1_000, true, "the header did not opt in"),
            (true, 0, true, "the keepalive is configured off"),
            (true, 1_000, false, "the request carries no progress record"),
            (false, 0, false, "nothing holds"),
        ] {
            assert!(
                JoinKeepalive::arm(
                    client_keepalive,
                    interval_ms,
                    progress.then(unstamped_progress)
                )
                .is_none(),
                "a join must not be armed when {why}"
            );
        }
    }

    /// The largest configurable interval arms without overflow, and the bidi
    /// loop can start and check its deadline: `Config` accepts any `u64`.
    #[tokio::test(start_paused = true)]
    async fn join_keepalive_with_the_largest_interval_starts_without_overflow() {
        let armed = JoinKeepalive::arm(true, u64::MAX, Some(unstamped_progress()))
            .expect("an opted-in join with an interval and a progress record is armed");
        assert_eq!(
            armed.deadline,
            Duration::from_millis(u64::MAX) * CLIENT_KEEPALIVE_DEADLINE_INTERVALS
        );
        let mut deadline = KeepaliveDeadline::start(&armed);
        let not_yet = tokio::time::timeout(Duration::from_secs(3600), deadline.expired()).await;
        assert!(
            not_yet.is_err(),
            "a deadline of ~1.7 billion years has not expired after an hour"
        );
    }

    /// The pure rule: with no data since the loop started, the join expires
    /// exactly one deadline after the loop's start, and until then the timer
    /// is re-armed for that instant.
    #[test]
    fn keepalive_expiry_without_data_expires_one_deadline_after_the_loop_start() {
        let deadline = Duration::from_secs(3);
        let loop_start = tokio::time::Instant::now();
        let expires_at = loop_start + deadline;

        assert_eq!(
            keepalive_expiry(
                loop_start,
                None,
                deadline,
                expires_at - Duration::from_nanos(1)
            ),
            KeepaliveExpiry::RearmAt(expires_at),
            "one nanosecond before the deadline the join is still alive"
        );
        assert_eq!(
            keepalive_expiry(loop_start, None, deadline, expires_at),
            KeepaliveExpiry::Expired,
            "the join expires at the deadline itself"
        );
        assert_eq!(
            keepalive_expiry(
                loop_start,
                None,
                deadline,
                expires_at + Duration::from_secs(60)
            ),
            KeepaliveExpiry::Expired
        );
    }

    /// Data the client sent before the loop started — the header, or a
    /// keepalive that waited unread through the membership replay — counts
    /// from the loop's start, never from when it was stamped, so the replay
    /// never counts against the deadline.
    #[test]
    fn keepalive_expiry_counts_data_from_before_the_loop_from_the_loop_start() {
        let deadline = Duration::from_secs(3);
        let before_the_loop = tokio::time::Instant::now();
        let loop_start = before_the_loop + Duration::from_secs(10);

        assert_eq!(
            keepalive_expiry(
                loop_start,
                Some(before_the_loop),
                deadline,
                loop_start + Duration::from_secs(2)
            ),
            KeepaliveExpiry::RearmAt(loop_start + deadline),
            "a stamp older than the loop must not expire a join the loop started 2 s ago"
        );
        assert_eq!(
            keepalive_expiry(
                loop_start,
                Some(before_the_loop),
                deadline,
                loop_start + deadline
            ),
            KeepaliveExpiry::Expired
        );
    }

    /// Data after the loop started moves the deadline: the join expires one
    /// whole deadline after its client's latest data, and a timer that fires
    /// at the old deadline is re-armed for the new one.
    #[test]
    fn keepalive_expiry_counts_from_the_latest_data_after_the_loop_start() {
        let deadline = Duration::from_secs(3);
        let loop_start = tokio::time::Instant::now();
        let last_data = loop_start + Duration::from_secs(2);

        assert_eq!(
            keepalive_expiry(loop_start, Some(last_data), deadline, loop_start + deadline),
            KeepaliveExpiry::RearmAt(last_data + deadline),
            "data 2 s into the loop keeps the join alive at the loop's first deadline, and the \
             timer is re-armed one deadline after that data"
        );
        assert_eq!(
            keepalive_expiry(
                loop_start,
                Some(last_data),
                deadline,
                last_data + deadline - Duration::from_nanos(1)
            ),
            KeepaliveExpiry::RearmAt(last_data + deadline)
        );
        assert_eq!(
            keepalive_expiry(loop_start, Some(last_data), deadline, last_data + deadline),
            KeepaliveExpiry::Expired,
            "one whole deadline after the latest data the join expires"
        );
    }

    /// The running deadline under a paused clock: data stamped 2 s into a
    /// 3 s deadline keeps the join alive past the loop's first deadline, and
    /// the deadline resolves exactly one deadline after that data. The first
    /// wait is dropped at 4 s and the deadline awaited again, which a bidi
    /// loop's `select!` does on every pass.
    #[tokio::test(start_paused = true)]
    async fn keepalive_deadline_resolves_one_deadline_after_the_latest_data() {
        use http_body_util::BodyExt;

        let (mut body, progress) = stamping_body();
        let armed = JoinKeepalive::arm(true, 1_000, Some(progress))
            .expect("an opted-in join with an interval and a progress record is armed");
        let loop_start = tokio::time::Instant::now();
        let mut deadline = KeepaliveDeadline::start(&armed);

        tokio::time::advance(Duration::from_secs(2)).await;
        let data_at = tokio::time::Instant::now();
        body.frame()
            .await
            .expect("a frame")
            .expect("an infallible body");

        let at_four_seconds =
            tokio::time::timeout_at(loop_start + Duration::from_secs(4), deadline.expired()).await;
        assert!(
            at_four_seconds.is_err(),
            "data 2 s into the loop keeps the join alive past the loop's first deadline (3 s)"
        );
        deadline.expired().await;
        assert_eq!(
            tokio::time::Instant::now(),
            data_at + Duration::from_secs(3),
            "the deadline resolves one whole deadline after the latest data"
        );
    }

    /// The keepalive ending is `UNAVAILABLE` carrying the refusal detail
    /// `CLIENT_KEEPALIVE_TIMEOUT` in the service's domain, with no metadata,
    /// and a `RetryInfo` delay under 100 ms: a minimum wait, so a live client
    /// that was ended rejoins at its own first backoff.
    #[test]
    fn client_keepalive_timeout_status_carries_its_reason_and_a_short_retry_delay() {
        use tonic_types::StatusExt;

        let status = client_keepalive_timeout_status();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(
            status.message(),
            "no data from the client for three keepalive intervals; rejoin the session"
        );
        let info = status
            .get_details_error_info()
            .expect("the ending must carry an ErrorInfo");
        assert_eq!(
            (info.reason.as_str(), info.domain.as_str()),
            ("CLIENT_KEEPALIVE_TIMEOUT", "nvidia.omniverse.live.v1alpha")
        );
        assert!(info.metadata.is_empty(), "no metadata: {:?}", info.metadata);
        for _ in 0..100 {
            let delay = client_keepalive_timeout_status()
                .get_details_retry_info()
                .and_then(|r| r.retry_delay)
                .expect("the ending must carry a RetryInfo delay");
            assert!(
                delay < Duration::from_millis(100),
                "retry_delay {delay:?} must be in [0, 100 ms)"
            );
        }
    }
}
