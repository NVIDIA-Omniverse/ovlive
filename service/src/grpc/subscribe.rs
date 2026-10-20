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

//! `SubscribeReadLayer`: admission, catch-up, then the steady-state live feed.
//! The client contract (regimes, acks, endings, what is retried) is SDD §12.3
//! and §12.4; this is the flow that implements it.
//!
//! **Setup (the handler).** Validate, authorize, take the per-session token
//! and enroll for the shutdown token (`GracefulShutdown::enroll_stream`), then
//! a subscription admission (`SubscriberAdmissionGate`).
//! A refusal is Trailers-Only and spawns nothing. Everything after runs in one
//! task on `subscription_tracker`, which holds the admission and the
//! enrollment for the RPC's
//! whole lifetime, re-bootstraps included.
//!
//! **Catch-up (`run_catch_up`).** Resolve the layer mapping, pick a regime,
//! deliver its content, then send exactly one caught-up ack (`total_size = 0`):
//!
//! - *Regime 1* (no prior object, or a different `object_id`; checked first):
//!   the full state (`catch_up_brand_new`).
//! - *Up to date* (same `object_id`, `have_version` at or above the header
//!   sequence): register the stream at the header sequence and ack it. No
//!   slot, no content.
//! - *Regime 2* (the stream still holds `{have_version+1}-0`): one coalesced
//!   diff based at `have_version` (`catch_up_in_window_snapshot`), decided by
//!   two single-entry reads (`read_in_window_point`); in rollback mode
//!   (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) one reconstructed diff per
//!   retained commit (`catch_up_in_window`), read in byte-sized pages
//!   (`read_in_window_replay`).
//! - *Regime 3* (otherwise): one coalesced diff based at `have_version`
//!   (`catch_up_out_of_window`).
//!
//! A state-reading regime reads the header (P), reserves a stream reader
//! (`reserve_stream`), takes a catch-up slot (`acquire_catchup_slot`), then a
//! read permit inside the slot (`acquire_snapshot_read_permit`) for one atomic
//! snapshot read (`read_catch_up_snapshot`), released as the read returns. It
//! registers its stream at the read's cut, builds from that snapshot under the
//! slot, releases the slot, and only then sends, acking and stamping the cut
//! (`catch_up_proven_read`, `deliver_catch_up_content`). A cut below P (a
//! sequence regression) ends the catch-up `UNAVAILABLE` before any content.
//! Every build goes through `build_catch_up`, the external-value miss
//! protocol: a blob the snapshot references that the GC collected since (a
//! later commit released it) is not corruption — the catch-up takes a
//! per-holder `{gc}` read hold, swaps its stream registration for a fresh
//! reservation, reads the layer again at a fresh proven cut (its own atomic
//! read under a new permit inside its slot, or, above the atomic-read limits,
//! the object's fenced-read flight — a successor of the stale one, on the
//! fenced path — which it attaches to with no slot), registers its
//! stream at that cut, builds and acks there, and releases the hold when the
//! rebuild ends; only a blob still missing under a hold the store confirms
//! live takes fetch-on-miss and, failing that, `INTERNAL`.
//!
//! A layer above the atomic-read limits is read once per replica, by its
//! shared fenced-read flight (`storage::snapshot_flights`, held on
//! `LiveEditService::snapshot_flights`). A catch-up that finds its object's
//! flight registered attaches to it with no slot and no atomic attempt;
//! otherwise its atomic read answers `TOO_LARGE`, and it gives its slot back
//! and attaches then (`catch_up_flight_read`). Attached, it holds no slot,
//! permit or memory of its own: it waits for the flight's read
//! (`read_through_flight`), checks the flight's cut against its own P, which
//! it read before attaching, registers its stream at that cut, waits for a
//! build turn the flight serves (`take_build_turn`), builds its own payload
//! from the shared snapshot and gives the turn back before it sends. The
//! reader, slot and permit waits, and an attached catch-up's share of its
//! flight's slot, fenced-permit and memory waits, draw on one
//! `CapacityBudget`. (Rollback mode reserves its reader under the slot, and
//! its in-window replay registers at the last entry it read and reads each
//! commit atomically under a permit of its own inside the slot; on a layer
//! above the atomic-read limits it hands over to the out-of-window route,
//! which attaches to the flight.) So no commit is lost between the state read
//! and the stream, and since the cursor is the acked cut, no commit the
//! snapshot folded in is delivered again. A `NotFound` on a state read runs
//! the layer recovery hook, then the mapping re-read
//! (`read_state_with_recovery`, `handle_deleted_and_rebootstrap`); so does one
//! on an external-value miss's re-read, once its read hold is released
//! (`recover_after_reread_not_found`). The rollback-mode in-window replay's
//! per-commit reads re-bootstrap on a `NotFound` without the hook.
//!
//! **Steady state (`run_steady_state`).** An `Update` at or below the tracked
//! version is dropped; a self-echo advances the tracked version and sends
//! nothing; any other is relayed verbatim from the entry's `ompf` field, or,
//! when the entry has none, is an empty commit and gets the empty delta at its
//! sequence, built with no read (`empty_commit_payload`). The steady state
//! never reads layer state. A `Deleted` entry re-reads the mapping: a new
//! `object_id` re-bootstraps as Regime 1 and sends its own caught-up ack;
//! otherwise the RPC ends `NOT_FOUND`.
//!
//! **Endings.** Every ending but a client disconnect returns `Err(status)`,
//! written into a response-channel slot reserved before anything was sent
//! (`send_terminal_status`), so no server-driven ending reads as `OK`. Each
//! ending after admission records one exit reason
//! (`live_edit_subscriber_stream_exits_total`); an admission refusal is logged
//! by the handler and not counted there.

use bytes::Bytes;
use fred::interfaces::StreamsInterface;
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use super::auth_helpers::{check_read_permission, extract_token, BasePathSource};
use super::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use super::rpc_log::{self, RpcLogContext, RpcStage};
use super::service::LiveEditService;
use super::session::{session_absent_status, SessionCancel, SessionCancelCause};
use super::streaming::{chunk_ompf_for_stream, OmpfStreamHeader};
use super::subscriber_delivery::{
    DeliveryTally, LiveDelivery, Outbound, SendKind, SendStall, TransportLagStream,
    DELIVERY_TALLY_FLUSH_INTERVAL,
};
use super::transport::server_shutting_down;
use crate::checkpoint::RecoveryOutcome;
use crate::error::{
    refusal_metadata, refusal_reason, RefusalDetail, ServiceError, CAPACITY_RETRY_DELAY,
};
use crate::observability::metrics::{
    CATCHUP_EXT_VALUE_MISS_OUTCOME_GENUINE, CATCHUP_EXT_VALUE_MISS_OUTCOME_STALE_RETRIED,
    CATCHUP_EXT_VALUE_MISS_READ_ATOMIC, CATCHUP_EXT_VALUE_MISS_READ_SHARED_FENCED,
    ENDPOINT_SUBSCRIBE_READ_LAYER, STREAM_EXIT_CATCHUP_CAPACITY, STREAM_EXIT_CLIENT_DISCONNECT,
    STREAM_EXIT_ERROR, STREAM_EXIT_FENCED_READ_WAIT_CAP, STREAM_EXIT_INNER_WATCHDOG,
    STREAM_EXIT_READER_CAPACITY, STREAM_EXIT_SESSION_ABSENT, STREAM_EXIT_SESSION_CANCEL,
    STREAM_EXIT_SHUTDOWN, STREAM_EXIT_SLOW_READER_DEADLINE, STREAM_EXIT_STREAM_GAP,
};
use crate::storage::diff_read::{
    compute_diff_at_cut, compute_faithful_coalesced_at_cut, compute_faithful_full_content_at_cut,
    compute_per_commit_diff_at_cut, empty_commit_payload, read_object_snapshot, AtomicSnapshot,
    Built, DiffCut, DiffResponse, ExtValueReads, PerCommitDiff, SnapshotRead, SnapshotReadConsumer,
    SnapshotTooLarge, SnapshotTooLargeAt, SubscriberState,
};
use crate::storage::external_values::{ReadHold, ReadHoldRenewal};
use crate::storage::notifications::{LayerNotification, StreamCursor, OUTER_SEND_DEADLINE};
use crate::storage::snapshot_fence::{probe_fence, FenceProbe, LayerSize};
use crate::storage::snapshot_flights::{BuildTurn, FlightRead, FlightWaitError};
use crate::storage::subscriber_admission::{
    cancelled_or_never, AdmissionRefusal, CapacityBudget, CatchUpPermit, CatchUpReadPermit,
    CatchUpRefusal,
};
use crate::storage::{reserve_layer_stream, ConsumeError, LayerStreamConsumer, ReservedStream};

type BoxStream<T> =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Default entry cap for the rollback-mode in-window replay when the
/// LayerStore has no configured MAXLEN. MAXLEN is the natural ceiling (a
/// subscriber can't possibly be more entries behind than the stream retains),
/// but when MAXLEN is 0 (no entry cap) we still need a guardrail so a
/// very-behind subscriber doesn't replay an unbounded window.
const IN_WINDOW_XRANGE_SOFT_CAP: u64 = 8192;

/// Handler for the `SubscribeReadLayer` RPC.
///
/// Validates, authenticates, takes a bounded subscription admission, then
/// spawns a task that runs the three-regime catch-up followed by a
/// steady-state loop. See the module docstring for the lifecycle overview; see
/// SDD §12.3 and §12.4 for the authoritative spec.
///
/// **Admission (b.f4f).** The per-replica ceiling is the subscription
/// ceiling, derived from the pod memory limit (`config::memory_plan`). Past it
/// a subscribe waits up to
/// `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS` (racing server shutdown, session
/// deletion and client disconnect) and is then refused with `UNAVAILABLE` —
/// rather than queueing indefinitely behind streams that may never end. The
/// admission is held for the RPC's whole lifetime, so a generation-transition
/// re-bootstrap re-uses it instead of competing for admission again.
/// Subscriptions hold no Valkey connection: they share the stream
/// multiplexer's readers (`LIVE_EDIT_SUBSCRIBER_POOL_SIZE` of them).
///
/// **One capacity budget.** `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS` is the
/// subscription's whole capacity-wait budget ([`CapacityBudget`]): what
/// admission leaves of it is all the task may then spend waiting for a stream
/// reader (cluster mode, [`reserve_stream`]), for catch-up slots
/// ([`acquire_catchup_slot`]), for snapshot read permits
/// ([`acquire_snapshot_read_permit`]) and, attached to a shared fenced-read
/// flight, for its share of the flight's slot, fenced-permit and memory waits
/// ([`read_through_flight`]). A catch-up waiting for capacity holds at most a
/// reader reservation, which queues nothing: its layer stream is registered
/// only at its proven read's cut, after every capacity wait
/// ([`catch_up_proven_read`]). Those waits come after the stream opened; when
/// the budget runs out they end it in-stream with `UNAVAILABLE` under
/// `reason="reader_capacity"` or `reason="catchup_capacity"`: before any
/// content during catch-up, in both catch-up modes; after the caught-up ack
/// only for a steady-state overwrite re-bootstrap. A re-bootstrap onto a new
/// object refills the budget. Nothing else in the steady state waits for
/// capacity: it never reads layer state.
///
/// **Endings by cause (SDD §12.6).** Graceful shutdown ends the RPC
/// `UNAVAILABLE "server shutting down"` in every phase — admission, catch-up,
/// delivery and the idle wait — which the client retries; this replica has
/// stopped accepting connections, so where the retry lands is the ingress's
/// choice. A
/// deleted session ends it `NOT_FOUND`, which the client does not retry: `NOT_FOUND
/// "session was deleted"` with `reason="session_cancel"` when the session token
/// is observed, in any phase, but `NOT_FOUND "layer '<name>' was deleted"` with
/// `reason="error"` when the deletion's `deleted` sentinel is read off the
/// layer stream first. Only the code is fixed; the message and reason depend on
/// which signal wins. Shutdown is checked first wherever both a shutdown and a
/// deletion can be observed. A session token cancelled because the session was
/// found missing from the live store, with no `DeleteSession` seen and no
/// deletion marker (the `session_tokens` sweeper, or another stream's setup
/// finding it gone; a marked session is cancelled as a deletion), ends the RPC
/// `UNAVAILABLE` instead, with `reason="session_absent"`
/// ([`session_absent_status`]), so the client's retry can recover a session a
/// checkpoint restores.
///
/// **NotFound recovery (SDD SR-2.7).** The pre-spawn `get_base_path` lookup
/// funnels through
/// [`SessionBasePathCache::get_base_path`](super::session_cache::SessionBasePathCache::get_base_path),
/// which on a cache-miss + session-absent triggers lazy on-demand
/// session-state recovery via the attached
/// [`SessionRecoveryHook`](crate::checkpoint::SessionRecoveryHook). A
/// genuinely non-existent / unrecoverable session still surfaces `NotFound`
/// to the client; no behavior change at this handler.
///
/// **Failure logging (SDD §17.2.1).** A setup failure — anything before the
/// response stream opens, admission refusals included — returns `Err` through
/// this wrapper, which emits the one completion line exactly as the
/// single-response handlers do. A failure after the stream opened is logged by
/// the subscription task's tail instead (see [`teardown_event`]), from a clone
/// of the same context. The signature is unchanged because the `LiveService`
/// trait impl and the integration tests bind it directly.
///
/// **Join latency.** The clock for `live_edit_join_latency_seconds` starts here,
/// at handler entry, and is observed once, when the first caught-up ack is
/// enqueued (see [`run_catch_up`]).
pub async fn handle_subscribe_read_layer(
    service: &LiveEditService,
    request: Request<SubscribeReadLayerRequest>,
) -> Result<Response<BoxStream<SubscribeReadLayerResponse>>, Status> {
    let started = Instant::now();
    let mut ctx = RpcLogContext::for_request("SubscribeReadLayer", &request);
    let result = subscribe_read_layer_inner(service, request, &mut ctx, started).await;
    if let Err(status) = &result {
        rpc_log::log_rpc_failure(&ctx, status);
    }
    result
}

async fn subscribe_read_layer_inner(
    service: &LiveEditService,
    request: Request<SubscribeReadLayerRequest>,
    log: &mut RpcLogContext,
    started: Instant,
) -> Result<Response<BoxStream<SubscribeReadLayerResponse>>, Status> {
    let token = extract_token(&request)?;

    let req = request.into_inner();

    // Validate client_id before anything else so a malformed identifier
    // never reaches storage / auth / tracing output (b.jr8 H-grpc-6). Every
    // `log` record below sits after its validator for the same reason.
    crate::domain::validation::validate_client_id(&req.client_id)?;
    log.client(&req.client_id);

    // b.39x H: full session_id validation (length cap + control-char
    // rejection) before storage / auth / tracing — same threat model as
    // the layer-handler suite.
    crate::domain::validation::validate_session_id(&req.session_id)?;
    log.session(&req.session_id);

    // The same rule CreateLayer / UpdateLayer / StatLayer / DeleteLayer
    // enforce, so no layer that can exist becomes unreachable. It also makes
    // the name safe to log: the mid-stream NOT_FOUND message echoes it, and the
    // teardown line carries that message.
    crate::domain::validation::validate_layer_name(&req.layer_name)?;
    log.layer(&req.layer_name);

    let session_id = req.session_id;
    let layer_name = req.layer_name;
    let have_object_id = req.object_id;
    let have_version = req.sequence_num;
    let client_id = req.client_id;
    if have_object_id != 0 {
        log.submitted_object_id(have_object_id);
    }

    // Auth: look up session base_path (short-TTL cached), check read permission.
    log.at(RpcStage::Auth);
    let base_path = service
        .session_base_path_cache
        .get_base_path(&service.session_store, &session_id)
        .await?;
    log.base_path(&base_path);
    check_read_permission(service, &token, &base_path, BasePathSource::Session).await?;

    // Get-or-insert the per-session CancellationToken. CreateSession only
    // populates it on the replica that handled the create RPC, so any
    // SubscribeReadLayer that lands on a different replica must insert its
    // own entry — otherwise a subsequent DeleteSession (which fans out via
    // `le:session:deleted`) would fail to cancel this stream (b.uue). The
    // helper also re-checks session existence after the insert to close the
    // race where the delete fan-out arrived before our insert, and, if the
    // session is missing, resolves it once more through the recovery hook
    // before treating it as gone. That pass is bounded by the budget JoinSession
    // needs (derived from the presence lease knobs; see
    // `config::session_recovery_pass_budget`): the helper is shared.
    let session_cancel = Some(
        super::session::acquire_session_cancel(
            &service.session_tokens,
            &service.session_store,
            &service.session_base_path_cache,
            crate::config::session_recovery_pass_budget(
                service.presence_refresh_interval_secs,
                service.presence_lease_ttl_secs,
            ),
            &session_id,
        )
        .await?,
    );

    // This subscription's enrolled child of the service shutdown token (SDD
    // §5.6): what admission, every send and the idle wait race, and the only
    // way graceful shutdown reaches the subscription — cancelled in its batch of
    // the paced ending. A child rather than a shared token for the reason
    // `JoinCtx::shutdown` is one (`presence.rs`): polling a token locks that
    // token's own mutex and waiter list, and on a process-wide token those
    // locks would be shared by every stream on the replica on every send; the
    // child's are this subscription's alone. Enrolled before admission, so it
    // is born cancelled if the paced ending has already started. The
    // enrollment moves into the subscription task, which holds it for the
    // RPC's whole life; a refusal before the spawn drops it here.
    let (shutdown, shutdown_enrollment) = service.shutdown.enroll_stream();

    // Bounded, cancellation-aware subscription admission (the subscription
    // ceiling derived from the pod memory limit, b.f4f).
    //
    // Placement is the whole point, and every neighbour above and below is load
    // bearing:
    //
    // - AFTER `check_read_permission`, so per-pod saturation is not observable
    //   to a caller who is not allowed to read this layer in the first place.
    // - AFTER `acquire_session_cancel` and the shutdown child above, so the wait
    //   can race both: shutdown first (`UNAVAILABLE`, retried), then the
    //   session token (`NOT_FOUND` for a deletion, `UNAVAILABLE` for a session
    //   found missing from the live store). A client that disconnects mid-wait
    //   needs no arm
    //   of its own: hyper's `H2Stream` drops this handler future on
    //   RST_STREAM (it polls for the reset while the handler is pending),
    //   which cancels the semaphore acquire.
    // - BEFORE the response channel and the spawn, so a refusal returns
    //   `Err(Status)` straight out of the handler and tonic emits
    //   **Trailers-Only**: no response message precedes the status, and no task
    //   is spawned behind a request that will be refused.
    //
    // One acknowledged residual: a refused request has already done a
    // get-or-insert on `service.session_tokens`. That entry is per-session,
    // idempotent, and would have been created by any other subscribe on the same
    // session, so it is not a leak.
    //
    // A refusal is logged by the handler wrapper at `stage=admission`, with
    // session, layer, client and `base_path`, the cause in `status_message`:
    // WARN `UNAVAILABLE` for saturation, INFO `UNAVAILABLE` for a shutdown (the
    // context is marked shutdown-caused), INFO `NOT_FOUND` for a deletion, and
    // WARN `UNAVAILABLE` for a session found missing from the live store.
    log.at(RpcStage::Admission);
    let admission_started = tokio::time::Instant::now();
    let admission = match service
        .valkey
        .admit_subscriber_until_shutdown(
            &shutdown,
            session_cancel.as_ref().map(SessionCancel::token),
        )
        .await
    {
        Ok(admission) => admission,
        Err(refusal) => {
            if refusal == AdmissionRefusal::ShuttingDown {
                log.shutdown_caused();
            }
            let cause = session_cancel.as_ref().map(SessionCancel::cause);
            return Err(refusal_to_status(refusal, cause));
        }
    };

    // The subscription's single capacity-wait budget (see the rustdoc above):
    // `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`, less what admission just waited.
    // Every later wait for capacity — a stream reader, a catch-up slot — draws
    // on what is left.
    let budget = CapacityBudget::new(service.valkey.subscriber_admission.policy().wait);
    let admission_waited = admission_started.elapsed();
    budget.charge(admission_waited);
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = service.capacity_budget_seam_for_test.as_ref() {
        seam.record(BudgetCharge {
            client_id: client_id.clone(),
            wait: BudgetWait::Admission,
            budget_before: budget.total(),
            started: admission_started,
            deadline: None,
            charged: admission_waited,
            budget_after: budget.remaining(),
        });
    }

    // Bridge the spawned task to the gRPC response stream.
    //
    // **33, not 32, and the extra slot is reserved immediately** (b.f4f). The
    // delivery capacity for data is unchanged at 32; the 33rd exists so the
    // terminal trailer always has somewhere to go. Without it the trailer
    // competes with the backlog it is usually reporting on — the
    // `slow_reader_deadline` tear-down is *reached* because the channel is full
    // — and a trailer that cannot be enqueued leaves the stream ending with a
    // bare EOS, i.e. gRPC `OK`. The Kit client does not retry `OK` (its retry
    // policy retries only DEADLINE_EXCEEDED, UNAVAILABLE, RESOURCE_EXHAUSTED
    // and INTERNAL; the inspected revision and line anchors:
    // `docs-internal/cross-repo-debugging.md` § Kit client retry and reconnect
    // behaviour), so a server-driven tear-down would kill the
    // subscription without a reconnect: exactly what the b.c7a contract
    // forbids, reached by a timeout instead of by an exit reason.
    let (tx, rx) = mpsc::channel::<Outbound>(33);

    // Taken here, on a channel that is empty by construction, so it resolves
    // without waiting. `reserve_owned` fails only when the receiver has been
    // dropped, and `rx` is still owned by this function two lines up — the arm
    // is unreachable, and is written as an error rather than an `expect` so an
    // unreachable case can never take a connection down.
    let trailer_permit = tx.clone().reserve_owned().await.map_err(|_| {
        ServiceError::Internal(
            "SubscribeReadLayer could not reserve its terminal-trailer slot on a freshly created \
             response channel"
                .to_owned(),
        )
    })?;

    let svc = service.clone();
    let session_id_owned = session_id.clone();
    let layer_name_owned = layer_name.clone();
    let client_id_owned = client_id.clone();
    // The setup context, handed to the task: its tail is the completion point
    // for every status decided after this handler returns.
    let mut tail_log = log.clone();
    tail_log.at(RpcStage::Stream);

    // Spawned on `subscription_tracker` so the post-server shutdown sequence
    // (`grpc::transport::drain_stream_tasks`) can wait (bounded) for every
    // subscription to finish — dropping its consumer and so its stream
    // multiplexer registration — before the pools quit.
    service.subscription_tracker.spawn(async move {
        // The admission permit lives HERE, in the subscription task, and not in
        // the handler: `handle_subscribe_read_layer` returns its response
        // stream the moment this task is spawned, so a handler-held
        // permit would be released before any work happened. This task IS the
        // RPC's lifetime, re-bootstraps included. The shutdown enrollment too:
        // it leaves the registry when this task ends, however it ends.
        let _shutdown_enrollment = shutdown_enrollment;
        let exit = StreamExitCell::new();
        let outcome = SubscriptionOutcomeCells::new();
        let result = {
            let ctx = SubscriberCtx {
                svc: &svc,
                tx: &tx,
                shutdown: &shutdown,
                session_cancel: session_cancel.as_ref(),
                session_id: &session_id_owned,
                layer_name: &layer_name_owned,
                client_id: &client_id_owned,
                have_object_id,
                have_version,
                budget: &budget,
                exit: &exit,
                outcome: &outcome,
                started,
            };
            run_subscription(&ctx).await
        };

        // Release the admission before the terminal trailer goes out; the
        // consumer (if one was open) already unregistered when
        // `run_subscription`'s frame unwound.
        drop(admission);

        let reason = exit.get();
        let caught_up = outcome.caught_up();

        // The completion line, logged synchronously BEFORE the trailer is
        // enqueued: logging marks the request's completion ledger, and the
        // metrics layer reads that mark when the trailer streams past, so the
        // order is what stops its backstop from logging the same status again.
        // What gets logged is `teardown_event`'s decision; this block changes
        // neither which status is sent, nor whether, nor the exit metric.
        outcome.record_into(&mut tail_log);
        tail_log.exit_reason(reason);
        tail_log.caught_up(caught_up);
        // A shutdown ending is expected on every rolling restart: its
        // `UNAVAILABLE` line is INFO, not WARN (see `rpc_log::failure_level`).
        if reason == STREAM_EXIT_SHUTDOWN {
            tail_log.shutdown_caused();
        }
        if let (TeardownEvent::CompletionLine, Err(status)) =
            (teardown_event(&result, reason, caught_up), &result)
        {
            rpc_log::log_rpc_failure(&tail_log, status);
        }

        if let Err(status) = result {
            // **The exhaustive list of exit reasons that end the stream with no
            // trailer is exactly one: `client_disconnect`.** An `Ok` ending is
            // always a client disconnect too (see `teardown_event`), so this
            // holds for every ending, not only for this arm.
            //
            // It is honest because it is not really a suppression: the receiver
            // has been dropped, so the trailer would be discarded whatever we
            // passed it. Nothing is hidden, because there is nobody left to hide
            // it from. (Whether the ending still gets a log line is a separate
            // question, answered by `teardown_event` above: a client that left
            // before its first caught-up ack does.)
            //
            // Every other reason surfaces its status, and it matters that they
            // do:
            //   * `slow_reader_deadline` and `inner_watchdog` — server-driven
            //     tear-downs. `UNAVAILABLE`, per the b.c7a contract, so the
            //     client re-subscribes and catches up from the position it
            //     holds. An `inner_watchdog` (or any other ending of the
            //     subscription's stream registration) recorded before the
            //     catch-up's first message ends the catch-up before any
            //     content (`end_catch_up_if_stream_ended`).
            //   * `shutdown` — graceful shutdown, in any phase: mid-catch-up,
            //     mid-delivery or idle. `UNAVAILABLE "server shutting down"`,
            //     so the client re-subscribes with the position it holds and
            //     catches up in-window; where the retry lands is the
            //     ingress's choice, since this replica has stopped accepting
            //     connections. The Kit client
            //     never retries `OK`, so a clean end here would silently end its
            //     live feed on every rolling restart.
            //   * `session_cancel` — the session token was observed cancelled
            //     for an explicit `DeleteSession`, in any phase. `NOT_FOUND
            //     "session was deleted"`: there is nothing to reconnect to, and
            //     the client must not retry. Mid-catch-up this also keeps a
            //     truncated OMPF sequence from reading as a complete one. A
            //     deletion that reaches the subscriber as the layer's `deleted`
            //     sentinel first is not this reason: it ends `NOT_FOUND "layer
            //     '<name>' was deleted"` under `error`, below.
            //   * `session_absent` — the session token was observed cancelled
            //     because the session was found missing from the live store
            //     with no `DeleteSession` seen and no deletion marker (a marked
            //     one counts as `session_cancel`), in any phase. `UNAVAILABLE`, so
            //     the client retries and a session a checkpoint can restore is
            //     recovered at the retry's setup, which runs the session
            //     recovery hook on whichever replica serves it
            //     (`acquire_session_cancel`).
            //   * `reader_capacity` and `catchup_capacity` — the capacity budget
            //     ran out waiting for a stream reader (cluster mode), or for a
            //     catch-up slot, a read permit, or the capacity a shared
            //     fenced-read flight waited for while the catch-up was attached
            //     to it. `UNAVAILABLE`, so the client retries, here or on
            //     another replica. Before any content during catch-up; after
            //     the caught-up ack only for a steady-state overwrite
            //     re-bootstrap.
            //   * `fenced_read_wait_cap` — the catch-up left its shared
            //     fenced-read flight because the read's work passed
            //     `reader_wait_max(L)`. `UNAVAILABLE`, before any content; the
            //     client retries.
            //   * `stream_gap` — the layer stream trimmed a commit this
            //     subscription was owed, one above its caught-up ack and above
            //     the last commit it was sent, before its stream reader read
            //     it, or the stream was removed and recreated by a later commit
            //     (a restore after a keyspace loss, a stream-key removal) above
            //     the subscription's position. `UNAVAILABLE "layer stream was
            //     trimmed past this subscription's position; reconnect to catch
            //     up"`, after the caught-up ack; the client's reconnect catches
            //     up in full (after a removal, onto the layer's current,
            //     restored content). A trim the catch-up covered never ends a
            //     subscription this way.
            //   * `error` — the default. Any backend failure, a stream reader
            //     that gave up, a `NOT_FOUND` on a deleted layer, a stream that
            //     ended without a recorded reason, or an unclassified `?` return
            //     lands here, which is why the match is written as "clean end
            //     iff client_disconnect" rather than as a list of codes to
            //     drop: a new failure mode cannot accidentally join the
            //     suppressed set.
            //
            // No cancellation ends the stream cleanly: every one of
            // `run_steady_state`'s shutdown and deletion arms returns `Err` too,
            // so each reaches here with its status.
            //
            // Reaching `send_terminal_status` means the status IS enqueued: it
            // writes into a slot reserved before this subscription sent anything,
            // so there is no arm on which a server-driven tear-down degrades into
            // a bare EOS the client would read as `OK`.
            if reason != STREAM_EXIT_CLIENT_DISCONNECT {
                send_terminal_status(trailer_permit, status);
            }
        }
        // Exactly one exit observation per established subscription — that is
        // what makes `sum by (reason)` the total (b.f4f).
        svc.metrics.inc_subscriber_stream_exit(reason);
    });

    // The channel's items, unchanged, with the `transport` delivery lag
    // measured as tonic polls each stamped one.
    let stream = TransportLagStream::new(rx, service.metrics.clone());
    Ok(Response::new(Box::pin(stream)))
}

/// Enqueue the terminal trailer into the slot reserved for it when the response
/// channel was created (b.f4f).
///
/// **Synchronous, infallible and unconditional**, and all three matter.
/// `OwnedPermit::send` writes into capacity this subscription has held since
/// before it did any work, so there is no await, nothing to race, and no way for
/// a full channel to refuse it. The status is therefore *always* enqueued ahead
/// of the stream's end.
///
/// The obvious alternatives are both wrong, and the reason is the same in each
/// case — a trailer is not an ordinary message and must not queue behind the
/// backlog it is reporting on:
///
/// * A bare `tx.send(Err(status)).await` parks for as long as the client stays
///   away, because the `slow_reader_deadline` exit is *reached* by the channel
///   being full. The task would never return.
/// * Racing that send against a deadline and giving up (what b.f4f shipped
///   first) bounds the task but not the contract: abandoning the trailer lets
///   `tx` drop with nothing in it, the response stream ends normally, and a
///   client that resumes draining a moment later reads the backlog and then EOS
///   with gRPC **`OK`**. The Kit client does not retry `OK` (its retry policy
///   retries only DEADLINE_EXCEEDED, UNAVAILABLE, RESOURCE_EXHAUSTED and
///   INTERNAL, and a subscription that finishes `OK` does not reconnect; the
///   inspected revision and line anchors: `docs-internal/cross-repo-debugging.md`
///   § Kit client retry and reconnect behaviour), so a
///   server-driven tear-down would end that subscription for good — the exact
///   outcome the b.c7a contract exists to prevent, arrived at by timeout rather
///   than by exit reason.
///
/// What the client does with the trailer once it is enqueued is the client's
/// business: a receiver that is never drained never sees it, and a dropped
/// receiver discards it. Neither can turn a server-driven tear-down into an
/// `OK`, which is the property this function exists to hold.
fn send_terminal_status(permit: mpsc::OwnedPermit<Outbound>, status: Status) {
    // Returns the `Sender` the permit was reserved from; dropping it here is
    // correct — this is the last thing the subscription sends. The trailer
    // ends no live delivery, so it carries no commit stamp.
    let _sender = permit.send(Outbound::unstamped(Err(status)));
}

/// Map an admission refusal to the gRPC status the client sees.
///
/// The three retryable refusals — saturation, a full waiter queue, and a
/// shutdown — are `UNAVAILABLE` (b.f4f policy decision 1), each with a retry
/// hint: [`SUBSCRIPTION_CAPACITY`](refusal_reason::SUBSCRIPTION_CAPACITY)
/// (metadata `cause` = `saturated` / `waiter_cap`) or
/// [`SERVER_SHUTTING_DOWN`](refusal_reason::SERVER_SHUTTING_DOWN). It is the one
/// saturation code this codebase already uses — the b.c7a terminal-error
/// contract — and the Kit client's retry policy treats `UNAVAILABLE` and
/// `RESOURCE_EXHAUSTED` identically anyway, mapping both to a single
/// client-visible result, with retries bounded at 10 retries whose backoff
/// sums to ~7.5-15 s (the inspected revision and line anchors:
/// `docs-internal/cross-repo-debugging.md` § Kit client retry and reconnect
/// behaviour). A shutdown refusal carries
/// the same status as every other shutdown ending
/// ([`server_shutting_down`]).
///
/// A cancelled session token maps by its cause (`session_cause`, the
/// handler's [`SessionCancel::cause`]), to the same status an established
/// subscription ends with when it observes the token
/// ([`session_cancel_status`]): [`session_deleted`] — `NOT_FOUND "session was
/// deleted"`, retrying cannot help — for an explicit `DeleteSession`, and the
/// retryable [`session_absent_status`] for a session found missing from the
/// live store.
///
/// The causes are told apart by `live_edit_subscriber_admission_total{result}`
/// and by these messages, which name them: the handler's refusal line
/// (`stage=admission`) carries the message as `status_message`. Both session
/// causes record `result="cancelled"`.
fn refusal_to_status(
    refusal: AdmissionRefusal,
    session_cause: Option<SessionCancelCause>,
) -> Status {
    let capacity = |cause: &str| {
        RefusalDetail::new(
            refusal_reason::SUBSCRIPTION_CAPACITY,
            Some(CAPACITY_RETRY_DELAY),
        )
        .with_metadata(refusal_metadata::CAUSE, cause)
    };
    match refusal {
        AdmissionRefusal::Saturated => capacity(refusal_metadata::SATURATED).unavailable(
            "this replica is at its concurrent-subscription ceiling (its memory budget, sized \
             from the pod's memory limit); retry",
        ),
        AdmissionRefusal::WaiterCap => capacity(refusal_metadata::WAITER_CAP).unavailable(
            "this replica's subscription admission queue is full \
             (LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS); retry",
        ),
        AdmissionRefusal::ShuttingDown => server_shutting_down(),
        AdmissionRefusal::Cancelled => session_cancel_status(session_cause),
    }
}

/// The status for an observed session-token cancel, by its cause. No cause
/// (no token) and [`SessionCancelCause::Deleted`] both read as a deletion.
fn session_cancel_status(cause: Option<SessionCancelCause>) -> Status {
    match cause {
        Some(SessionCancelCause::AbsentFromStore) => session_absent_status(),
        Some(SessionCancelCause::Deleted) | None => session_deleted(),
    }
}

/// The status a `SubscribeReadLayer` ends — or is refused — with when it
/// observes its session token cancelled for an explicit `DeleteSession`, in any
/// phase: `NOT_FOUND "session was deleted"`, with `reason="session_cancel"`
/// (SDD §12.6). A token cancelled because the session was found missing from
/// the live store ends with [`session_absent_status`] instead.
///
/// `NOT_FOUND` because the session is gone and a retry cannot bring it back.
/// A session deletion reaches a subscriber by one of two signals, and only the
/// **code** is the same on both: a subscriber that reads the deletion's
/// `deleted` sentinel off its layer stream before it sees the token ends
/// through `handle_deleted_and_rebootstrap` → [`deleted_not_found`] instead —
/// `NOT_FOUND "layer '<name>' was deleted"`, with the exit reason left at its
/// `error` default. The message and the `reason` label therefore depend on
/// which signal wins. One constructor, so every token-observing phase says the
/// same thing.
fn session_deleted() -> Status {
    Status::not_found("session was deleted")
}

/// Which terminal arm a subscription ended on, for
/// `live_edit_subscriber_stream_exits_total{reason}` (b.f4f).
///
/// A cell rather than a return value because the classification is made deep
/// inside the catch-up / steady-state call tree — `send_or_disconnect` knows it
/// hit the slow-reader deadline; `run_steady_state` knows the consumer handed it
/// a watchdog tear-down — while the single observation point is the spawned
/// task's tail. `AtomicU8` (not `Cell`) so `&SubscriberCtx` stays `Sync` and the
/// subscription future stays `Send`.
struct StreamExitCell(std::sync::atomic::AtomicU8);

impl StreamExitCell {
    /// Defaults to [`STREAM_EXIT_ERROR`], which is therefore the catch-all for
    /// any ending this file could not classify: a `?`-propagated failure — a
    /// catch-up error, a backend fault, a `NOT_FOUND` on a deleted layer — or
    /// a consumer that ended with no recorded reason (see `run_steady_state`).
    /// Every one of them delivers its status to the client, and gets its
    /// completion line.
    ///
    /// Every ending that *can* be classified sets the cell before returning.
    fn new() -> Self {
        Self(std::sync::atomic::AtomicU8::new(Self::ERROR))
    }

    const ERROR: u8 = 0;
    const SLOW_READER_DEADLINE: u8 = 1;
    const INNER_WATCHDOG: u8 = 2;
    const SESSION_CANCEL: u8 = 3;
    const CLIENT_DISCONNECT: u8 = 4;
    const SHUTDOWN: u8 = 5;
    const SESSION_ABSENT: u8 = 6;
    const READER_CAPACITY: u8 = 7;
    const CATCHUP_CAPACITY: u8 = 8;
    const STREAM_GAP: u8 = 9;
    const FENCED_READ_WAIT_CAP: u8 = 10;

    fn set(&self, code: u8) {
        self.0.store(code, std::sync::atomic::Ordering::Release);
    }

    /// Set from a `&'static str` the consumer already classified
    /// ([`ConsumerTerminalExit::reason`]).
    fn set_reason(&self, reason: &'static str) {
        self.set(match reason {
            STREAM_EXIT_INNER_WATCHDOG => Self::INNER_WATCHDOG,
            STREAM_EXIT_SLOW_READER_DEADLINE => Self::SLOW_READER_DEADLINE,
            STREAM_EXIT_SESSION_CANCEL => Self::SESSION_CANCEL,
            STREAM_EXIT_SESSION_ABSENT => Self::SESSION_ABSENT,
            STREAM_EXIT_SHUTDOWN => Self::SHUTDOWN,
            STREAM_EXIT_CLIENT_DISCONNECT => Self::CLIENT_DISCONNECT,
            STREAM_EXIT_READER_CAPACITY => Self::READER_CAPACITY,
            STREAM_EXIT_CATCHUP_CAPACITY => Self::CATCHUP_CAPACITY,
            STREAM_EXIT_FENCED_READ_WAIT_CAP => Self::FENCED_READ_WAIT_CAP,
            STREAM_EXIT_STREAM_GAP => Self::STREAM_GAP,
            _ => Self::ERROR,
        });
    }

    fn get(&self) -> &'static str {
        match self.0.load(std::sync::atomic::Ordering::Acquire) {
            Self::SLOW_READER_DEADLINE => STREAM_EXIT_SLOW_READER_DEADLINE,
            Self::INNER_WATCHDOG => STREAM_EXIT_INNER_WATCHDOG,
            Self::SESSION_CANCEL => STREAM_EXIT_SESSION_CANCEL,
            Self::SESSION_ABSENT => STREAM_EXIT_SESSION_ABSENT,
            Self::SHUTDOWN => STREAM_EXIT_SHUTDOWN,
            Self::CLIENT_DISCONNECT => STREAM_EXIT_CLIENT_DISCONNECT,
            Self::READER_CAPACITY => STREAM_EXIT_READER_CAPACITY,
            Self::CATCHUP_CAPACITY => STREAM_EXIT_CATCHUP_CAPACITY,
            Self::FENCED_READ_WAIT_CAP => STREAM_EXIT_FENCED_READ_WAIT_CAP,
            Self::STREAM_GAP => STREAM_EXIT_STREAM_GAP,
            _ => STREAM_EXIT_ERROR,
        }
    }
}

/// What the subscription had reached when it ended, for its teardown line
/// (SDD §17.2.1): the object it was serving, the catch-up branch last entered,
/// and whether the first caught-up ack was enqueued.
///
/// Written only at catch-up entry points and at the caught-up ack — never in
/// the steady-state `Update` arm, so the per-message path is untouched — and
/// read once, by the task's tail. Atomics for the same reason as
/// [`StreamExitCell`]: `&SubscriberCtx` must stay `Sync`.
///
/// `regime` is "the catch-up branch that was running when the subscription
/// ended", in the `live_subscribe_regime_total{regime}` vocabulary. It can
/// differ from that metric's label for the same subscription — a same-object
/// reconnect that re-bootstraps is `in_window` in the metric and `brand_new`
/// here — which is harmless because the metric only records successful
/// catch-ups and this only reaches teardown lines.
///
/// The cells also hold the subscription's place on
/// `live_edit_subscribe_catchups_pending` (and the `/scaling/v1` pending and
/// completed counts): [`Self::begin_first_catchup`] counts it in when its first
/// state-reading catch-up is selected; [`Self::set_caught_up`] moves it to
/// completed; and if the subscription ends before that, dropping the cells —
/// which the task owns, so a task abort drops them too — counts it out.
struct SubscriptionOutcomeCells {
    regime: std::sync::atomic::AtomicU8,
    /// `0` = unset: no layer mapping resolved yet. `0` is never a real object
    /// id — the protocol uses it for "no object" (`have_object_id == 0` is a
    /// brand-new subscriber).
    current_object_id: std::sync::atomic::AtomicU64,
    caught_up: std::sync::atomic::AtomicBool,
    /// `FIRST_CATCHUP_*`: whether the first state-reading catch-up is
    /// counted pending. Moves only forward, `NONE` → `PENDING` → `DONE` or
    /// `NONE` → `DONE`, so the subscription is counted in at most once and
    /// out exactly once.
    first_catchup: std::sync::atomic::AtomicU8,
    /// The metrics the pending count was taken on, set with `PENDING`.
    first_catchup_metrics: std::sync::OnceLock<crate::observability::metrics::SharedMetrics>,
}

impl SubscriptionOutcomeCells {
    const REGIME_UNSET: u8 = 0;
    const REGIME_BRAND_NEW: u8 = 1;
    const REGIME_UP_TO_DATE: u8 = 2;
    const REGIME_IN_WINDOW: u8 = 3;
    const REGIME_OUT_OF_WINDOW: u8 = 4;

    const FIRST_CATCHUP_NONE: u8 = 0;
    const FIRST_CATCHUP_PENDING: u8 = 1;
    const FIRST_CATCHUP_DONE: u8 = 2;

    fn new() -> Self {
        Self {
            regime: std::sync::atomic::AtomicU8::new(Self::REGIME_UNSET),
            current_object_id: std::sync::atomic::AtomicU64::new(0),
            caught_up: std::sync::atomic::AtomicBool::new(false),
            first_catchup: std::sync::atomic::AtomicU8::new(Self::FIRST_CATCHUP_NONE),
            first_catchup_metrics: std::sync::OnceLock::new(),
        }
    }

    /// A catch-up that reads layer state was selected: count the
    /// subscription pending, once. A no-op after the first caught-up ack
    /// (a steady-state re-bootstrap is not a first catch-up) and on a second
    /// call before it (a catch-up-phase re-bootstrap continues the same one).
    fn begin_first_catchup(&self, metrics: &crate::observability::metrics::SharedMetrics) {
        if self
            .first_catchup
            .compare_exchange(
                Self::FIRST_CATCHUP_NONE,
                Self::FIRST_CATCHUP_PENDING,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
        {
            let _ = self
                .first_catchup_metrics
                .set(std::sync::Arc::clone(metrics));
            metrics.begin_pending_catchup();
        }
    }

    /// End a pending first catch-up, as completed or not. Does nothing unless
    /// it is pending.
    fn finish_first_catchup(&self, completed: bool) {
        if self.first_catchup.swap(
            Self::FIRST_CATCHUP_DONE,
            std::sync::atomic::Ordering::AcqRel,
        ) == Self::FIRST_CATCHUP_PENDING
        {
            if let Some(metrics) = self.first_catchup_metrics.get() {
                metrics.end_pending_catchup(completed);
            }
        }
    }

    fn set_regime(&self, regime: u8) {
        self.regime
            .store(regime, std::sync::atomic::Ordering::Release);
    }

    fn set_object_id(&self, object_id: u64) {
        self.current_object_id
            .store(object_id, std::sync::atomic::Ordering::Release);
    }

    /// The set point is `run_catch_up`, once `send_ack_header` returned `Ok`
    /// — the moment the subscriber's first caught-up ack is enqueued. A
    /// pending first catch-up completes here; an up-to-date subscriber, never
    /// counted pending, is only marked so that no later catch-up counts.
    fn set_caught_up(&self) {
        self.caught_up
            .store(true, std::sync::atomic::Ordering::Release);
        self.finish_first_catchup(true);
    }

    fn caught_up(&self) -> bool {
        self.caught_up.load(std::sync::atomic::Ordering::Acquire)
    }

    fn regime_label(&self) -> Option<&'static str> {
        match self.regime.load(std::sync::atomic::Ordering::Acquire) {
            Self::REGIME_BRAND_NEW => Some("brand_new"),
            Self::REGIME_UP_TO_DATE => Some("up_to_date"),
            Self::REGIME_IN_WINDOW => Some("in_window"),
            Self::REGIME_OUT_OF_WINDOW => Some("out_of_window"),
            _ => None,
        }
    }

    /// Copy the object id and regime into the teardown line's context. Unset
    /// cells stay absent from the line.
    fn record_into(&self, log: &mut RpcLogContext) {
        let object_id = self
            .current_object_id
            .load(std::sync::atomic::Ordering::Acquire);
        if object_id != 0 {
            log.resolved_object_id(object_id);
        }
        if let Some(regime) = self.regime_label() {
            log.regime(regime);
        }
    }
}

impl Drop for SubscriptionOutcomeCells {
    /// A subscription that ends before its first caught-up ack — an error, a
    /// disconnect, a refusal, a task abort — leaves the pending count without
    /// completing.
    fn drop(&mut self) {
        self.finish_first_catchup(false);
    }
}

/// What a subscription's tail logs; see [`teardown_event`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TeardownEvent {
    /// No line.
    Silent,
    /// The completion line for the status the subscription ended with.
    CompletionLine,
}

/// Decide what a subscription's tail logs, from how it ended.
///
/// | outcome | exit reason | `caught_up` | event |
/// |---|---|---|---|
/// | `Err` | anything but `client_disconnect` | any | completion line — the status is enqueued for the client |
/// | `Err` | `client_disconnect` | `false` | completion line — the client left before its first caught-up ack was enqueued (`CANCELLED`, INFO) |
/// | `Err` | `client_disconnect` | `true` | silent |
/// | `Ok` | `client_disconnect` (the only reason an `Ok` ending carries) | any | silent |
///
/// Silent endings are the ones that are not failures: a client leaving an
/// established stream, whether observed mid-send (an `Err` carrying
/// `client_disconnect`) or at a steady-state loop boundary (the `OK` ending).
/// Every other ending returns `Err`: a shutdown or a session-token cancel in
/// every phase, and a consumer that ended with no recorded reason too
/// (`run_steady_state`). So each gets its completion line — INFO for a
/// shutdown (`UNAVAILABLE` marked shutdown-caused) and for a deletion
/// (`NOT_FOUND`), WARN for a session found missing from the live store
/// (`UNAVAILABLE`, `reason="session_absent"`).
///
/// Pure, so the whole table is unit-testable without a subscription.
fn teardown_event(
    outcome: &Result<(), Status>,
    reason: &'static str,
    caught_up: bool,
) -> TeardownEvent {
    match outcome {
        Err(_) if reason != STREAM_EXIT_CLIENT_DISCONNECT => TeardownEvent::CompletionLine,
        Err(_) if !caught_up => TeardownEvent::CompletionLine,
        Err(_) => TeardownEvent::Silent,
        Ok(()) => {
            debug_assert_eq!(
                reason, STREAM_EXIT_CLIENT_DISCONNECT,
                "a SubscribeReadLayer that ends `Ok` must have ended on a client disconnect; \
                 every other ending returns its status"
            );
            TeardownEvent::Silent
        }
    }
}

/// Shared context for the spawned subscription task.
struct SubscriberCtx<'a> {
    svc: &'a LiveEditService,
    tx: &'a mpsc::Sender<Outbound>,
    /// This subscription's enrolled child of the service shutdown token.
    /// Cancelled only by graceful shutdown — in its batch of the paced ending,
    /// or with the service token; every observation ends the RPC with
    /// [`server_shutting_down`] and `reason="shutdown"`. Checked before
    /// `session_cancel` everywhere both are observed.
    shutdown: &'a CancellationToken,
    /// The per-session token, cancelled only once the session is gone from the
    /// store. Every observation ends the RPC by the cancel's cause
    /// ([`SubscriberCtx::end_on_session_cancel`]): [`session_deleted`] with
    /// `reason="session_cancel"` for an explicit `DeleteSession`,
    /// [`session_absent_status`] with `reason="session_absent"` for a session
    /// found missing from the live store.
    session_cancel: Option<&'a SessionCancel>,
    session_id: &'a str,
    layer_name: &'a str,
    client_id: &'a str,
    have_object_id: u64,
    have_version: u64,
    /// This subscription's single capacity-wait budget, drawn on by every
    /// stream-reader wait ([`reserve_stream`]), catch-up slot wait
    /// ([`acquire_catchup_slot`]), read-permit wait
    /// ([`acquire_snapshot_read_permit`]) and share of a shared fenced-read
    /// flight's capacity waits ([`read_through_flight`]); refilled by a
    /// re-bootstrap onto a new object ([`handle_deleted_and_rebootstrap`]).
    budget: &'a CapacityBudget,
    /// Where each terminal arm records why the subscription ended, for the one
    /// `live_edit_subscriber_stream_exits_total{reason}` observation the spawned
    /// task emits after `run_subscription` returns.
    exit: &'a StreamExitCell,
    /// What the subscription had reached, for the tail's teardown line.
    outcome: &'a SubscriptionOutcomeCells,
    /// Handler entry, the start of `live_edit_join_latency_seconds`.
    started: Instant,
}

impl SubscriberCtx<'_> {
    /// The session token, for APIs that race a bare token.
    fn session_token(&self) -> Option<&CancellationToken> {
        self.session_cancel.map(SessionCancel::token)
    }

    /// End the subscription for an observed session-token cancel: record the
    /// exit reason for the cancel's cause and return its status
    /// ([`session_cancel_status`]). Every arm that observes the session token
    /// ends through here, so the phases cannot disagree.
    fn end_on_session_cancel(&self) -> Status {
        let cause = self.session_cancel.map(SessionCancel::cause);
        self.exit.set(match cause {
            Some(SessionCancelCause::AbsentFromStore) => StreamExitCell::SESSION_ABSENT,
            Some(SessionCancelCause::Deleted) | None => StreamExitCell::SESSION_CANCEL,
        });
        session_cancel_status(cause)
    }
}

/// Tracked subscriber position after catch-up. Feeds the steady-state loop.
struct CatchUpOutcome {
    /// The object generation the subscriber is now tracking.
    tracked_object_id: u64,
    /// Last sequence number that has been delivered (or suppressed via echo).
    tracked_version: u64,
    /// Live stream consumer for the steady-state loop, positioned at the
    /// correct cursor for the regime we just finished.
    consumer: LayerStreamConsumer,
    /// Catch-up delivery mode for the `catchup_mode` metric attribute (Plan
    /// b.msb): `"snapshot"` ONLY for the corrected faithful full-state path
    /// (`send_full_state`'s `Snapshot` arm); `"legacy"` for every other outcome.
    catchup_mode: &'static str,
    /// OMPF bytes delivered to the subscriber during this catch-up regime, fed
    /// to the `live_subscribe_catch_up_delivered_bytes` histogram (Plan b.msb).
    delivered_bytes: u64,
}

/// What a catch-up regime's content delivery produced, for the caller to fold
/// into its [`CatchUpOutcome`].
///
/// **Ack-≤-content invariant (b.vd8, cross-repo).** `effective_version` is
/// simultaneously the `sequence_num` stamped on the catch-up phase's LAST
/// content message and the `tracked_version` the caught-up ack carries. The two
/// must move together and the ack must NEVER exceed the content: the Kit client
/// suppresses the caught-up header only when its sequence number is at or below
/// the one the client already holds, so an ack above the last content message's
/// sequence dispatches an empty-content update, which the Kit client rejects as
/// fatal — the pending-queue stall of the b.dy5 / b.rdr failure class. Returning
/// one value used at both sites is what makes that structural rather than a
/// convention each call site has to remember.
///
/// All three content-serving regimes (`brand_new`, `in_window`,
/// `out_of_window`) construct this exclusively via
/// [`deliver_catch_up_content`], which is the only place `effective_version` is
/// computed and the only place a content message is stamped with it — so the
/// invariant holds by construction rather than per regime. (`up_to_date` serves
/// no content and acks the header sequence it read directly.)
struct CatchUpContent {
    /// OMPF bytes delivered (Plan b.msb catch-up-bytes metric).
    delivered_bytes: u64,
    /// `"snapshot"` for the corrected faithful full-state path, `"legacy"`
    /// otherwise (Plan b.msb `catchup_mode` attribute).
    catchup_mode: &'static str,
    /// The version this catch-up actually served: the proven cut of the
    /// snapshot its payload was built from — its own atomic read's at or below
    /// the atomic-read limits, its flight's shared fenced read's above them.
    effective_version: u64,
}

/// Record one catch-up state read's cut outcome (b.vd8; Spec SDD SR-8.6):
/// `verified` for a served proven cut — an atomic read's, or, once per
/// attached catch-up, its flight's shared fenced read's; `unverified` for a
/// read whose cut regressed below the catch-up's first header read, which
/// ends the catch-up `UNAVAILABLE` ([`catch_up_regressed`]), or for a
/// rollback-mode in-window replay's per-commit read whose cut is below the
/// commit it rebuilds, which ends it the same way
/// ([`per_commit_regressed`]; the only replay read recorded here), or for a
/// re-read after an external-value miss whose cut is below the first read's
/// ([`reread_and_rebuild`]).
///
/// `regime` is the literal of the catch-up path that took the read (`brand_new`
/// | `in_window` | `out_of_window`). It matches what the caller's regime reports
/// to `observe_subscribe_regime`, so the stable-cut counters join against
/// `subscribe_regime_total`, with one deliberate exception: a rollback-mode
/// in-window replay that hands over above the ceiling
/// ([`OutOfWindowEntry::AboveCeiling`]) records `out_of_window` here, because
/// the read is Regime 3's, while `subscribe_regime_total` still counts the
/// catch-up `in_window` (SDD §12.3).
fn record_stable_cut(ctx: &SubscriberCtx<'_>, verified: bool, regime: &'static str) {
    ctx.svc
        .metrics
        .observe_stable_cut(verified, &ctx.svc.service_label, regime);
}

/// Deliver one catch-up regime's computed payload and report what it served.
///
/// The single place any catch-up regime turns a [`DiffResponse`] into wire
/// messages. Centralising it is what makes the ack-≤-content invariant on
/// [`CatchUpContent`] structural instead of a convention three regimes have to
/// re-derive: `effective_version` is computed once here, stamped on every
/// content chunk, and returned for the caller to hand to
/// `CatchUpOutcome.tracked_version`. There is no path that can raise one without
/// the other.
///
/// What it serves comes from the payload's cut (Spec SDD SR-3.1, SR-3.3): a
/// catch-up payload is built from a snapshot proven at its cut — its own
/// atomic read's, or its flight's shared fenced read's — so it carries
/// [`DiffCut::Atomic`], and serves that cut, stamped with the timestamps the
/// header held at it. Any other cut is an `INTERNAL` ending, before anything
/// is sent: a catch-up payload with no proven cut cannot say what it serves.
///
/// **Empty payloads.** No regime reaches here with zero OMPF bytes. A
/// coalesced diff is an incremental delta — always from the faithful builder,
/// and from the legacy `compute_diff_at_cut` whenever the state it read is above
/// `have_version`, which every catch-up's is: a regime that serves content
/// read its header sequence P above `have_version`, and a cut below P ends the
/// catch-up before it builds, in every regime and in both modes — and an
/// incremental delta is never zero bytes. So a reconnector whose gap held
/// only empty commits still receives one delta, empty of changes, based at
/// `have_version` and labelled with the served version, which is what keeps
/// the client's sequence contiguous.
///
/// Regime 1 enforces the same rule *before* calling here: a from-empty
/// establishment payload must never be empty, so `send_full_state` fails it as
/// `INTERNAL` (SR-1.2) rather than reaching this helper with zero bytes.
async fn deliver_catch_up_content(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    diff_response: &DiffResponse,
    regime: &'static str,
    catchup_mode: &'static str,
) -> Result<CatchUpContent, Status> {
    // The version this catch-up actually serves, and the timestamps its
    // content header carries.
    let (effective_version, server_timestamp, client_timestamp) = match diff_response.cut {
        DiffCut::Atomic {
            cut,
            last_server_ts,
            last_client_ts,
        } => {
            record_stable_cut(ctx, true, regime);
            (cut, last_server_ts, last_client_ts)
        }
        cut => {
            return Err(ServiceError::Internal(format!(
                "catch-up payload for object_id={object_id} built at {cut:?} was delivered \
                     with no proven cut; it cannot say what it serves"
            ))
            .into());
        }
    };

    if !diff_response.ompf_bytes.is_empty() {
        let stream_header = OmpfStreamHeader {
            object_id,
            sequence_num: effective_version,
            server_timestamp,
            client_timestamp,
        };
        // Counter and park both live entirely inside the cfg, so a release build
        // compiles the loop exactly as it did before the seam existed — there is
        // no dead local for the optimiser to have to remove.
        #[cfg(any(test, feature = "test-fault-injection"))]
        let mut chunks_sent = 0usize;
        for msg in chunk_ompf_for_stream(&diff_response.ompf_bytes, &stream_header) {
            send_or_disconnect(ctx, msg).await?;
            // Test-only park, after the FIRST chunk only: content has been
            // delivered, the caught-up ack has not, and the test can now fire a
            // cancellation into exactly that window (b.f4f). A no-op when no
            // seam is installed.
            #[cfg(any(test, feature = "test-fault-injection"))]
            {
                chunks_sent += 1;
                if chunks_sent == 1 {
                    if let Some(seam) = ctx.svc.catch_up_chunk_seam_for_test.as_ref() {
                        seam.arrive_and_wait().await;
                    }
                }
            }
        }
    }

    Ok(CatchUpContent {
        delivered_bytes: diff_response.ompf_bytes.len() as u64,
        catchup_mode,
        effective_version,
    })
}

/// Top-level subscription driver: catch up, then steady state.
async fn run_subscription(ctx: &SubscriberCtx<'_>) -> Result<(), Status> {
    let outcome = run_catch_up(ctx).await?;
    run_steady_state(ctx, outcome).await
}

// ── Catch-up phase ──────────────────────────────────────────────────────────

/// Pick one of the four regimes and catch the subscriber up to current state.
///
/// Returns the post-catch-up position (tracked object_id, tracked version, and
/// a live stream consumer positioned at the right cursor for steady state).
///
/// On the success path, exactly one `observe_subscribe_regime` observation is
/// emitted before returning (`brand_new` | `up_to_date` | `in_window` |
/// `out_of_window`), labelled by the regime `run_catch_up` itself picked.
/// In-window dispatches that fall through to `handle_deleted_and_rebootstrap`
/// (and from there back into `catch_up_brand_new`) stay labelled `in_window`
/// — that's the branch `run_catch_up` selected. See b.qxe.
///
/// Takes no catch-up slot itself: each regime takes one around its
/// backing-store reads and build (and, in rollback mode on a same-object
/// reconnect, its cluster stream-reader wait) only, and has released it before
/// its first send, so the caught-up ack below never waits behind, or holds, a
/// slot.
async fn run_catch_up(ctx: &SubscriberCtx<'_>) -> Result<CatchUpOutcome, Status> {
    let start = Instant::now();
    let (outcome, regime) = run_catch_up_inner(ctx).await?;
    ctx.svc.metrics.observe_subscribe_regime(
        regime,
        &ctx.svc.service_label,
        start.elapsed(),
        outcome.catchup_mode,
        outcome.delivered_bytes,
    );
    // Single exactly-once caught-up terminator for the entire catch-up phase.
    // Covers all four regimes (`brand_new` | `up_to_date` | `in_window` |
    // `out_of_window`) AND the catch-up-phase rebootstrap, because every one of
    // those paths funnels its `CatchUpOutcome` up through `run_catch_up_inner`'s
    // `Ok` return. The Kit client gates stage-apply on this
    // `total_size == 0` ack (it fires the `OkLatest` callback that marks the
    // client caught up); without it a subscriber to a populated layer in any regime
    // never leaves initial-sync and silently stashes subsequent live updates
    // (b.dy5, sibling of b.rdr). Placed AFTER `observe_subscribe_regime` so the
    // regime metric stays exactly-once even if this send fails. MUST carry
    // `outcome.tracked_version` (the last replayed/delivered endpoint per
    // regime), NOT a recomputed `current_seq`: for the in-window under-replay
    // case `tracked_version` is the last *replayed* sequence while `current_seq`
    // is the header's value, and acking `current_seq` there would make the
    // client treat the unreplayed real edits as already-seen.
    send_ack_header(ctx, outcome.tracked_object_id, outcome.tracked_version).await?;
    // The first caught-up ack is enqueued: from here a client disconnect is a
    // client leaving an established stream, which the tail does not log. A
    // first catch-up counted pending completes here.
    ctx.outcome.set_caught_up();
    // The same point closes the join: handler entry to first delivered state,
    // once per subscription. The steady-state re-bootstrap's ack does not
    // observe again.
    ctx.svc
        .metrics
        .observe_join_latency(ENDPOINT_SUBSCRIBE_READ_LAYER, ctx.started.elapsed());
    Ok(outcome)
}

/// Inner body of `run_catch_up`. Returns the catch-up outcome alongside the
/// regime label so the outer wrapper can record exactly one metric observation
/// at the end of the catch-up phase.
async fn run_catch_up_inner(
    ctx: &SubscriberCtx<'_>,
) -> Result<(CatchUpOutcome, &'static str), Status> {
    // Resolve the current object mapped to this layer.
    let object_id = ctx
        .svc
        .layer_store
        .get_layer_mapping(ctx.session_id, ctx.layer_name)
        .await?
        .ok_or_else(|| {
            ServiceError::NotFound(format!(
                "layer '{}' not found in session '{}'",
                ctx.layer_name, ctx.session_id
            ))
        })?;
    ctx.outcome.set_object_id(object_id);

    // Stack (b.9qr): every regime future below is boxed at its dispatch, so
    // this function's future carries pointers, not the regimes' state
    // machines, and a debug build (which gives each inline future its own
    // non-overlapping slot) does not sum them. The fence is
    // `tests::run_catch_up_future_stays_under_its_recorded_debug_ceiling`.

    // Regime 1: brand new subscriber — no prior state to reconcile. Send full
    // content and open the consumer at the header-derived cursor `{seq}-0`.
    let is_brand_new = ctx.have_object_id == 0 && ctx.have_version == 0;
    if is_brand_new {
        let outcome = Box::pin(catch_up_brand_new(ctx, object_id)).await?;
        return Ok((outcome, "brand_new"));
    }

    // Known subscriber — compare have_object_id to current mapping.
    if ctx.have_object_id != object_id {
        // The layer was overwritten between the subscriber's last session and
        // now. Treat as a brand-new subscription on the new object.
        let outcome = Box::pin(catch_up_brand_new(ctx, object_id)).await?;
        return Ok((outcome, "brand_new"));
    }

    // Test-only: synchronization seam (b.xmk / b.msb). If installed, park
    // here — AFTER the mapping read + overwrite-detect branch above, and
    // BEFORE the same-object `read_object_header(OLD)` call below — so a test
    // can deterministically interleave a `create_layer` overwrite +
    // `cleanup_old_object_retrying` UNLINK against the parked subscriber, exercising
    // the b.jvx Site B `ServiceError::NotFound → handle_deleted_and_rebootstrap`
    // classifier on the initial `read_object_header(OLD)`.
    //
    // Two-phase `parked`/`released` gate (b.msb): the subscriber has already
    // read `mapping == OLD` at this point (the test fires its overwrite only
    // AFTER `wait_until_parked` returns, so the mapping read cannot observe
    // NEW), so the subscriber is guaranteed to reach this park — no race, no
    // deadlock. The pre-b.msb single 2-party barrier fired concurrently with
    // the overwrite: a subscriber that lost the race read `mapping == NEW`,
    // took the overwrite-detect branch above, and never arrived here,
    // hanging the test's lone barrier arrival. The two-phase handshake
    // removes that race while keeping the park at this Site-B position so the
    // intended `read_object_header(OLD)` NotFound path is the one exercised.
    // Gated behind `test-fault-injection` so release builds never see this
    // branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.run_catch_up_inner_seam_for_test.as_ref() {
        seam.arrive_and_wait().await;
    }

    // Same object_id — read header to learn the current sequence number.
    //
    // b.jvx Site B: a `CreateLayer` overwrite committing between the mapping
    // read above and this header read can UNLINK the OLD header (and other
    // data keys) via its cleanup body's background task (`cleanup_old_object_retrying`)
    // before we issue HGETALL. `read_object_header` returns
    // `ServiceError::NotFound("object {oid} header not found")` in that case
    // (single producer in `storage/layer.rs`).
    //
    // SR-2.4 (b.fgj): the same NotFound symptom can also indicate
    // layer-state loss in Valkey for an `object_id` whose live mapping
    // still resolves here. Try lazy on-demand recovery first; on
    // `Restored` retry the header read exactly once (the retry-once
    // contract). On `NotRecoverable` (no checkpoint exists, apply guard
    // rejected, or checkpointing disabled) fall through to the existing
    // mapping-re-lookup path so b.jvx Site B's overwrite-detection
    // semantics are preserved. On `Error(_)` classify per the project's
    // transient-error contract (b.33e): `Transient` → UNAVAILABLE,
    // everything else → INTERNAL.
    //
    // The outer wrapper records this catch-up as `in_window` for the
    // metric (b.qxe — preserve the branch that was selected) regardless
    // of which recovery arm fires.
    //
    // No consumer is open yet at this site, so there is none to shut down, and
    // no catch-up slot is held: this header read comes before the slot on
    // every path (see the placement note below).
    // A second NotFound after a `Restored` means a CreateLayer overwrite raced
    // the recovery window; the mapping re-lookup inside the re-bootstrap is
    // what detects it.
    let header = match read_state_with_recovery(ctx, object_id, None, &mut None, move || {
        ctx.svc.layer_store.read_object_header(object_id)
    })
    .await?
    {
        StateRead::Value(h) => h,
        StateRead::Rebootstrapped(outcome) => return Ok((outcome, "in_window")),
    };
    let current_seq = header.sequence_num;

    if ctx.have_version >= current_seq {
        ctx.outcome
            .set_regime(SubscriptionOutcomeCells::REGIME_UP_TO_DATE);
        if ctx.have_version > current_seq {
            // Subscriber claims a version ahead of what the server has
            // committed. This shouldn't happen under normal operation — the
            // subscriber can only know a sequence_num the server issued — so
            // log it for observability but stay forgiving (ack + tail from the
            // header-derived cursor).
            tracing::warn!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                object_id = object_id,
                have_version = ctx.have_version,
                current_seq = current_seq,
                "client reported version ahead of server header — treating as up-to-date"
            );
        }
        // Already up to date. Nothing to replay; the caught-up terminator is
        // emitted once for the whole catch-up phase by `run_catch_up`. Tail
        // from the header sequence we just read — `stream_cursor_at`, not `$`
        // (b.9uq round 4; see that helper). No catch-up slot: this path reads
        // and builds nothing.
        let consumer = open_consumer(ctx, object_id, stream_cursor_at(current_seq)).await?;
        return Ok((
            CatchUpOutcome {
                tracked_object_id: object_id,
                tracked_version: current_seq,
                consumer,
                catchup_mode: "legacy",
                delivered_bytes: 0,
            },
            "up_to_date",
        ));
    }

    // The regime reads layer state from here on (in-window or out-of-window):
    // the subscription's first catch-up is now pending until its ack.
    ctx.outcome.begin_first_catchup(&ctx.svc.metrics);

    // Catch-up slot placement. The slot covers this catch-up's backing-store
    // reads — the in-window stream read included — its read permit and its
    // build, and is released before anything is sent, before any recovery
    // attempt and before any re-bootstrap; no path waits for a slot while
    // holding one, or while holding a read permit. Above the atomic-read
    // limits the in-window read is the only read it covers: the catch-up
    // gives the slot back before it attaches to its object's shared
    // fenced-read flight, whose own slot covers the read and the build turns
    // (`catch_up_proven_read`).
    //
    // Snapshot mode (the default) RESERVES the stream reader HERE, before the
    // slot, so a cluster-mode wait for a reader never holds a slot. The
    // reservation queues nothing: each regime registers the stream only at
    // its proven read's cut (`catch_up_proven_read`), after every capacity
    // wait — so nothing that lands during the capacity waits is queued only
    // to be dropped as already covered by the ack.
    //
    // The rollback mode (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) instead
    // takes its reader under the slot, after the in-window read: the legacy
    // in-window replay registers at the last entry it read, which only that
    // read names, and the out-of-window regime reserves there and registers
    // where snapshot mode does. Its reader wait is still bounded by the
    // capacity budget, and runs holding the slot.
    let reserved = if ctx.svc.catchup_snapshot_enabled {
        Some(reserve_stream(ctx, object_id).await?)
    } else {
        None
    };
    let slot = acquire_catchup_slot(ctx).await?;

    // The regime test: the subscriber is in-window when the stream still
    // holds exactly `{have_version+1}-0`, the first commit it has not seen.
    // Otherwise fall through to Regime 3. Both modes read under the slot, and
    // neither pulls the retained window into one reply:
    //
    // - Snapshot mode (the default) needs only two answers, never an entry's
    //   bytes: is `{have_version+1}-0` there, and is the stream's last entry a
    //   `deleted` sentinel. Two single-entry reads in one round trip
    //   (`read_in_window_point`), so a catch-up moves at most two entries
    //   whatever the window holds.
    // - Rollback mode replays every retained commit, so it reads the window
    //   in byte-sized pages, each parsed and dropped before the next, and
    //   falls through to Regime 3 when `{have_version+1}-0` is gone after a
    //   multi-page read (a trim between pages may have left a hole)
    //   (`read_in_window_replay`).
    let start_id = format!("{}-0", ctx.have_version + 1);
    let stream_key = ctx.svc.key_schema.layer_stream(object_id);

    // Test-only: synchronization seam (b.fdy). If installed, pause here —
    // holding the catch-up slot — so a test can deterministically interleave
    // a `create_layer` overwrite
    // between the header read above and the in-window stream read below —
    // exactly the production race that makes the literal `catch_up_in_window`
    // mid-window `Deleted` variant untestable without a seam. Gated behind
    // `test-fault-injection` so release builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_xrange_seam_for_test.as_ref() {
        seam.arrive_and_wait().await;
    }

    if let Some(reserved) = reserved {
        let point = read_in_window_point(ctx, &stream_key, &start_id).await?;
        if point.in_window {
            let outcome = Box::pin(catch_up_in_window_snapshot(
                ctx,
                object_id,
                point.deleted_sentinel,
                current_seq,
                reserved,
                slot,
            ))
            .await?;
            return Ok((outcome, "in_window"));
        }
        // Regime 3: out of window. Fall back to the coalesced diff + tail from
        // the proven read's cut.
        let outcome = Box::pin(catch_up_out_of_window(
            ctx,
            object_id,
            current_seq,
            reserved,
            slot,
            OutOfWindowEntry::ProvenRead,
        ))
        .await?;
        return Ok((outcome, "out_of_window"));
    }

    if let Some(replay) = read_in_window_replay(ctx, &stream_key, &start_id).await? {
        let outcome = Box::pin(catch_up_in_window(
            ctx,
            object_id,
            replay,
            current_seq,
            slot,
        ))
        .await?;
        return Ok((outcome, "in_window"));
    }

    // Regime 3, rollback mode: the reader is reserved under the slot; the
    // stream is registered where snapshot mode registers it.
    let reserved = reserve_stream(ctx, object_id).await?;
    let outcome = Box::pin(catch_up_out_of_window(
        ctx,
        object_id,
        current_seq,
        reserved,
        slot,
        OutOfWindowEntry::ProvenRead,
    ))
    .await?;
    Ok((outcome, "out_of_window"))
}

/// Snapshot mode's in-window regime test ([`read_in_window_point`]).
struct InWindowPoint {
    /// The stream holds `{have_version+1}-0`.
    in_window: bool,
    /// The id of the stream's last entry when it is a `deleted` sentinel.
    deleted_sentinel: Option<String>,
}

/// Snapshot mode's in-window regime test, under the caller's catch-up slot:
/// `XRANGE key {h+1}-0 {h+1}-0` and `XREVRANGE key + {h+1}-0 COUNT 1`, pipelined
/// in one round trip on one connection (the same key, so the same slot).
///
/// - In-window exactly when the first returns its entry: the stream still
///   holds `{h+1}-0`.
/// - A `deleted` sentinel can only be the stream's LAST entry: its auto id
///   (`{ms}-{n}`) sorts above every `{seq}-0` commit, and no commit can be
///   appended after it (the next commit's `XADD {seq}-0` is refused below the
///   stream top). So "a sentinel anywhere in the retained window from
///   `{h+1}-0`" is "the last entry at or above `{h+1}-0` is a sentinel", which
///   the second read answers — the same race the b.fdy seam covers.
///
/// At most two entries are transferred, whatever the retained window holds;
/// their `ompf` bytes are dropped with the replies. Single-attempt with no
/// in-process retry: a transient failure answers `UNAVAILABLE` (the client
/// retries the subscribe), a permanent one `INTERNAL`.
async fn read_in_window_point(
    ctx: &SubscriberCtx<'_>,
    stream_key: &str,
    start_id: &str,
) -> Result<InWindowPoint, Status> {
    let map_err = |e: fred::error::Error| -> Status {
        crate::storage::client::map_fred_err_for_grpc(
            crate::storage::client::FredErrOrigin::client(
                &ctx.svc.valkey,
                crate::storage::client::POOL_LABEL_DATA,
            ),
            "XRANGE / XREVRANGE in-window point read",
            e,
        )
        .into()
    };
    let pipeline = ctx.svc.valkey.pool.next().pipeline();
    let _: () = pipeline
        .xrange(stream_key, start_id, start_id, None)
        .await
        .map_err(map_err)?;
    let _: () = pipeline
        .xrevrange(stream_key, "+", start_id, Some(1))
        .await
        .map_err(map_err)?;
    let mut replies = pipeline.try_all::<fred::types::Value>().await;
    if replies.len() != 2 {
        return Err(map_err(
            replies
                .into_iter()
                .find_map(Result::err)
                .unwrap_or_else(|| {
                    fred::error::Error::new(
                        fred::error::ErrorKind::Protocol,
                        "in-window point read returned a short reply set",
                    )
                }),
        ));
    }
    // Field VALUES decode as `Bytes` (not `String`): an entry carries the
    // binary `ompf` field (b.gze) beside the UTF-8 `payload`, and a `String`
    // decode would fail on the OMPF bytes' invalid UTF-8.
    let last: Vec<(String, HashMap<String, Bytes>)> = replies
        .pop()
        .expect("length checked")
        .and_then(|value| value.into_xread_value())
        .map_err(map_err)?;
    let point: Vec<(String, HashMap<String, Bytes>)> = replies
        .pop()
        .expect("length checked")
        .and_then(|value| value.into_xread_value())
        .map_err(map_err)?;
    let deleted_sentinel = last.into_iter().next().and_then(|(id, fields)| {
        let payload = fields.get(crate::storage::notifications::NOTIFICATION_PAYLOAD_FIELD)?;
        let payload = std::str::from_utf8(payload).ok()?;
        matches!(
            LayerNotification::parse(payload),
            Ok(LayerNotification::Deleted)
        )
        .then_some(id)
    });
    Ok(InWindowPoint {
        in_window: point.first().is_some_and(|(id, _)| id == start_id),
        deleted_sentinel,
    })
}

/// Rollback mode's in-window window, as [`read_in_window_replay`] read it.
struct InWindowReplay {
    /// The id of the last entry read, the steady-state consumer's start.
    last_entry_id: String,
    /// Every entry read whose `payload` parsed, in stream order: its id, its
    /// notification, and whether it carried relay bytes (an `ompf` field).
    replay: Vec<(String, LayerNotification, bool)>,
}

/// Rollback mode's in-window read, under the caller's catch-up slot: the
/// stream from `{h+1}-0` to its end, `None` when the stream no longer holds
/// `{h+1}-0` (out of window; nothing past the first page is read).
///
/// Paged with `XRANGE key <cursor> + COUNT n`, each page's `n` byte-sized from
/// the largest entry read so far ([`crate::storage::stream_mux::paged_xrange_count`],
/// the soft rule of the stream multiplexer's reads), the cursor advanced to
/// the exclusive `({last id}`. Each page is parsed into `(id, notification,
/// has_ompf)` and dropped, `ompf` bytes included, before the next is read, so
/// a catch-up never holds more than one page of relay payload. The relay byte
/// budget bounds the whole window at about one budget's bytes.
///
/// Paging makes the read non-atomic, so a read of more than one page ends with
/// one more `XRANGE key {h+1}-0 {h+1}-0`, and answers `None` (out of window)
/// when `{h+1}-0` is gone. A head trim between two pages (`MAXLEN ~` on a
/// concurrent `XADD`, or the relay byte budget's `XTRIM MINID ~`) can remove
/// entries past the cursor, and the next page would then start beyond the
/// hole, acking commits it never replays. Trimming is head-only and monotone,
/// so any trim that removed an entry past `{h+1}-0` removed `{h+1}-0` first,
/// and a check issued after the last page sees it. A one-page read is one
/// atomic `XRANGE`, which cannot hold a hole, so it skips the check.
///
/// The read stops at the stream's end (a page shorter than its `COUNT`) or
/// after [`in_window_xrange_count`] entries. Past that cap the replay
/// under-replays, which is safe: the steady-state consumer opens at the last
/// entry read and relays the remainder live. The loop runs to the end rather
/// than stopping after one page because the live remainder lands in the
/// subscription's bounded backlog, which a window of large entries would
/// overflow.
///
/// An entry with no `payload`, a non-UTF-8 one, or one that does not parse is
/// skipped with a WARN, as the replay always has, but still moves
/// `last_entry_id`. Single-attempt reads with no in-process retry: a
/// transient failure answers `UNAVAILABLE`, a permanent one `INTERNAL`.
async fn read_in_window_replay(
    ctx: &SubscriberCtx<'_>,
    stream_key: &str,
    start_id: &str,
) -> Result<Option<InWindowReplay>, Status> {
    let entry_cap = in_window_xrange_count(ctx.svc);
    let map_err = |what: &'static str, e: fred::error::Error| -> Status {
        crate::storage::client::map_fred_err_for_grpc(
            crate::storage::client::FredErrOrigin::client(
                &ctx.svc.valkey,
                crate::storage::client::POOL_LABEL_DATA,
            ),
            what,
            e,
        )
        .into()
    };
    let mut cursor = start_id.to_owned();
    let mut read: u64 = 0;
    let mut pages: u32 = 0;
    let mut largest_entry_bytes: Option<u64> = None;
    let mut last_entry_id: Option<String> = None;
    let mut replay: Vec<(String, LayerNotification, bool)> = Vec::new();
    loop {
        let count = crate::storage::stream_mux::paged_xrange_count(largest_entry_bytes)
            .min(entry_cap - read);
        // Field VALUES decode as `Bytes` (not `String`): see
        // `read_in_window_point`. The replay re-validates `payload` as UTF-8,
        // and of `ompf` uses only its presence.
        let page: Vec<(String, HashMap<String, Bytes>)> = ctx
            .svc
            .valkey
            .pool
            .xrange_values(stream_key, cursor.as_str(), "+", Some(count))
            .await
            .map_err(|e| map_err("XRANGE in-window replay page", e))?;
        pages = pages.saturating_add(1);
        if last_entry_id.is_none() && page.first().is_none_or(|(id, _)| id != start_id) {
            // Out of window: the stream no longer holds `{h+1}-0`.
            return Ok(None);
        }
        let page_len = page.len() as u64;
        read += page_len;
        for (entry_id, fields) in page {
            let entry_bytes = crate::storage::stream_mux::stream_entry_bytes(&entry_id, &fields);
            largest_entry_bytes =
                Some(largest_entry_bytes.map_or(entry_bytes, |largest| largest.max(entry_bytes)));
            if let Some(notification) = parse_replay_entry(ctx, &entry_id, &fields) {
                replay.push((
                    entry_id.clone(),
                    notification,
                    fields.contains_key(crate::storage::notifications::RELAY_OMPF_FIELD),
                ));
            }
            last_entry_id = Some(entry_id);
        }
        let Some(last) = last_entry_id.as_deref() else {
            // Unreachable: the in-window check above saw a first entry.
            return Ok(None);
        };
        if page_len < count || read >= entry_cap {
            if pages > 1 {
                // A head trim between two pages can have removed entries past
                // the cursor; any such trim removed `{h+1}-0` first (head-only,
                // monotone), and this check runs after the last page.
                let still_held: Vec<(String, HashMap<String, Bytes>)> = ctx
                    .svc
                    .valkey
                    .pool
                    .xrange_values(stream_key, start_id, start_id, Some(1))
                    .await
                    .map_err(|e| map_err("XRANGE in-window replay window-start check", e))?;
                if still_held.first().is_none_or(|(id, _)| id != start_id) {
                    // Trimmed mid-replay: the pages may hold a hole, so the
                    // window is out, never replayed with the hole acked.
                    return Ok(None);
                }
            }
            return Ok(Some(InWindowReplay {
                last_entry_id: last.to_owned(),
                replay,
            }));
        }
        cursor = format!("({last}");
    }
}

/// One in-window entry's notification, or `None` — with the WARN the replay
/// has always logged — when the entry has no `payload`, a non-UTF-8 one, or
/// one that does not parse.
fn parse_replay_entry(
    ctx: &SubscriberCtx<'_>,
    entry_id: &str,
    fields: &HashMap<String, Bytes>,
) -> Option<LayerNotification> {
    let Some(payload_bytes) = fields.get(crate::storage::notifications::NOTIFICATION_PAYLOAD_FIELD)
    else {
        tracing::warn!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            entry_id = %entry_id,
            "in-window entry missing payload field; skipping"
        );
        return None;
    };
    let Ok(payload) = std::str::from_utf8(payload_bytes) else {
        tracing::warn!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            entry_id = %entry_id,
            "in-window entry payload field is not valid UTF-8; skipping"
        );
        return None;
    };
    match LayerNotification::parse(payload) {
        Ok(notification) => Some(notification),
        Err(e) => {
            tracing::warn!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                entry_id = %entry_id,
                payload = %payload,
                error = %e,
                "failed to parse in-window stream entry; skipping"
            );
            None
        }
    }
}

/// Regime 1: brand-new subscriber (or overwrite-on-reconnect).
///
/// Reads the current header (P), reserves a stream reader (its cluster wait
/// bounded by the capacity budget), then takes its proven read
/// ([`catch_up_proven_read`]). At or below the atomic-read limits that is a
/// catch-up slot, a read permit inside it, the atomic read, the permit
/// released and the stream registered at the read's cut. Above them it is the
/// object's shared fenced-read flight, attached with no slot of its own
/// ([`catch_up_flight_read`]): a catch-up that finds the flight registered
/// skips the slot wait here altogether. It builds the full state — under its
/// slot, or in its build turn in the flight ([`take_build_turn`]) — gives the
/// slot or the turn back, and only then sends the full state as a single
/// diff. Before the send it ends the catch-up if its stream registration
/// already ended ([`end_catch_up_if_stream_ended`]).
async fn catch_up_brand_new(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
) -> Result<CatchUpOutcome, Status> {
    // Every path onto a new object — first subscription, overwrite on
    // reconnect, both re-bootstraps — enters here, so this one set point keeps
    // the teardown line's object and regime current, and counts a first
    // catch-up pending (a no-op for a steady-state re-bootstrap, which comes
    // after the first ack).
    ctx.outcome
        .set_regime(SubscriptionOutcomeCells::REGIME_BRAND_NEW);
    ctx.outcome.set_object_id(object_id);
    ctx.outcome.begin_first_catchup(&ctx.svc.metrics);
    // P: only the regression pin of the proven read below (Spec SDD SR-3.2),
    // never where the stream registers. Read before any flight lookup or
    // attach: a catch-up attaching to a fenced flight must have read its pin
    // first (`SnapshotFlights::attach_or_create`'s precondition).
    let header = ctx.svc.layer_store.read_object_header(object_id).await?;

    // Reserve the stream reader BEFORE the slot wait, so a cluster-mode wait
    // for a reader never holds a slot and the reader stays bound through the
    // slot wait. The reservation queues nothing: commits that land during the
    // capacity waits are at or below the cut the proven read takes, so the
    // snapshot folds them in and the stream starts after them.
    let mut reserved = Some(reserve_stream(ctx, object_id).await?);

    // The catch-up slot covers the read permit, the state read and the build
    // below, and nothing else: it is taken after the reader wait above and
    // released before the first send. A catch-up whose object already has a
    // shared fenced-read flight registered — attachable, or closed and still
    // building — takes none: it will attach to that flight, whose slot is the
    // unit of work, and an attached catch-up never waits for a slot (Spec SDD
    // SR-7.5). `catch_up_proven_read` looks again, and takes the slot after
    // all should the flight have finished meanwhile.
    let mut slot = if ctx.svc.snapshot_flights.lookup(object_id).is_some() {
        None
    } else {
        Some(acquire_catchup_slot(ctx).await?)
    };

    // Test-only: synchronization seam (b.vd8). If installed with
    // `new_before_full_state`, park here — after P and the capacity waits for
    // the reader and the slot, holding the slot (none when the object's flight
    // is registered), before the read permit and the state read — so a test
    // can deterministically land a commit after P and assert that the catch-up
    // serves and acks it exactly once. Parks once per catch-up, before its
    // proven read (the atomic attempt or the attach). Gated behind
    // `test-fault-injection` so release builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_state_read_seam_for_test.as_ref() {
        seam.arrive_at_pre_full_state().await;
    }

    let read = Box::pin(catch_up_proven_read(
        ctx,
        object_id,
        header.sequence_num,
        "brand_new",
        &mut reserved,
        &mut slot,
    ))
    .await?;
    let (mut consumer, snapshot) = match read {
        CatchUpRead::Proven { snapshot, consumer } => (consumer, snapshot),
        CatchUpRead::Rebootstrapped(outcome) => return Ok(outcome),
    };
    let snapshot = Box::pin(take_build_turn(ctx, snapshot, &consumer)).await?;
    // The build, under the external-value miss protocol (b.yuo). It consumes
    // the snapshot handle, so the handle (on the fenced path, the build turn
    // and the place in the flight with it) goes back as the build returns. A
    // re-read after a miss replaces `consumer` in place, registered at the
    // re-read's cut, and gives the slot back if it attaches to a flight; the
    // slot, if still held, goes back here, before the ending check and the
    // send.
    let built = Box::pin(build_catch_up(
        ctx,
        object_id,
        CatchUpBuild::FullState,
        snapshot,
        "brand_new",
        &mut slot,
        Some(&mut consumer),
    ))
    .await;
    drop(slot);
    let Some((diff_response, regime_mode)) = cut_based_payload(built?)? else {
        // The miss protocol's re-read answered `NotFound`, and its read hold
        // is released: stop the stream (a no-op: the re-read released its
        // registration), then the recovery the first read would have run,
        // still before any content.
        consumer.shutdown_and_release().await;
        return Box::pin(recover_after_reread_not_found(
            ctx,
            object_id,
            RestoredRerun::BrandNew,
        ))
        .await;
    };
    // Released; end the catch-up before its first message if its stream
    // registration ended during the build.
    end_catch_up_if_stream_ended(ctx, &consumer)?;
    let content = send_full_state(ctx, object_id, &diff_response, regime_mode).await?;

    Ok(CatchUpOutcome {
        tracked_object_id: object_id,
        tracked_version: content.effective_version,
        consumer,
        catchup_mode: content.catchup_mode,
        delivered_bytes: content.delivered_bytes,
    })
}

/// Regime 2, rollback mode (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`): known
/// subscriber with the gap fully inside the retention window, replayed as the
/// legacy per-commit reconstruction.
///
/// Replays each retained entry as an individual per-commit diff, applying
/// self-echo suppression per entry, and opens the steady-state consumer from
/// the last replayed stream ID so nothing is duplicated on the steady-state
/// loop. The default mode takes [`catch_up_in_window_snapshot`] instead
/// (Plan b.msb, SR-2.1), which delivers a SINGLE re-established coalesced diff
/// based at `have_version` — converging on the same primitive the out-of-window
/// regime already uses; the only surviving distinction there is the regime
/// metric label (`in_window`).
///
/// Entered holding the catch-up `slot` the caller took around the in-window
/// read ([`read_in_window_replay`]), and every read and build of the replay
/// runs under that one slot; only then is anything sent. `current_seq` is the
/// header sequence the caller read before its capacity waits, P. In order:
///
/// 1. Take the window the paged read parsed, page by page, into
///    `(id, notification, has_ompf)`; no `ompf` bytes are held.
/// 2. Scan for a `deleted` sentinel. If there is one, release the slot and
///    re-bootstrap before any consumer is opened or anything is sent — the
///    check [`catch_up_in_window_snapshot`] makes from its point read.
/// 3. Open the steady-state consumer at the window's last entry (its cluster
///    reader wait bounded by the capacity budget).
/// 4. Compute every commit's diff in order ([`replay_commit_diff`]): each from
///    its own atomic read, under its own read permit taken inside the slot and
///    charged to the capacity budget (Spec SDD SR-3.4, SR-7.1) — except an
///    empty commit (an entry without relay bytes above sequence 1), whose
///    empty delta is built with no read and no permit
///    ([`empty_commit_payload`]). A read whose cut is below the commit it
///    rebuilds (a sequence regression) is refused by the build itself
///    ([`compute_per_commit_diff_at_cut`]) and ends the catch-up `UNAVAILABLE`
///    ([`per_commit_regressed`]). Each commit's build runs under the
///    external-value miss protocol ([`build_catch_up`]): a blob its read
///    references that was collected since is re-read, under a new permit
///    inside the slot, before it counts as missing. A `NotFound`
///    (the OLD object's state gone, b.jvx Site A) releases the slot, stops the
///    consumer and re-bootstraps, still before any content. On the first
///    `TOO_LARGE` (the layer is above the atomic-read limits) — or, before a
///    commit's read, on finding the object's shared fenced-read flight
///    registered, which skips that atomic attempt — the replay stops: it
///    drops the diffs computed so far, stops the consumer, reserves a reader
///    under the same slot and hands the slot and P to the out-of-window route
///    ([`catch_up_out_of_window`] with [`OutOfWindowEntry::AboveCeiling`]),
///    which gives the slot back, attaches to the object's flight
///    ([`catch_up_flight_read`]), registers at the flight's cut and delivers
///    one coalesced diff — so the rollback switch never issues one fenced read
///    per replayed commit.
/// 5. Release the slot, end the catch-up if its stream registration already
///    ended ([`end_catch_up_if_stream_ended`]), then send the stored diffs in
///    order ([`send_per_commit_payload`]), freeing each after its send.
///
/// So every capacity wait of this catch-up — the slot, the reader, every read
/// permit, and those of the out-of-window route or of a re-bootstrap it
/// enters — comes before its first content message.
async fn catch_up_in_window(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    window: InWindowReplay,
    current_seq: u64,
    slot: CatchUpPermit,
) -> Result<CatchUpOutcome, Status> {
    ctx.outcome
        .set_regime(SubscriptionOutcomeCells::REGIME_IN_WINDOW);

    // 1. The window as `read_in_window_replay` read it: each page was parsed
    // into the replay's `(id, notification, has_ompf)` triples and dropped,
    // relay `ompf` bytes included, so none stays pinned across the computes
    // and sends; an entry with no payload, one that is not UTF-8, or one that
    // does not parse was skipped there with the warning the replay has always
    // logged for it. `last_entry_id` is the last entry read, skipped or not:
    // the stream ID we'll hand off to the steady-state consumer.
    let InWindowReplay {
        last_entry_id,
        replay,
    } = window;

    // 2. Overwrite handling: a `deleted` sentinel anywhere in the entries
    // the in-window read returned means the OLD object is gone — re-resolve via the mapping and
    // either terminate or re-bootstrap on the new object's stream. Scanned
    // up-front, before any consumer is opened or anything is sent, so the
    // OLD object's commits are not replayed ahead of the new object's state
    // (which supersedes them) and the re-bootstrap's capacity waits come
    // before any content.
    if let Some((entry_id, _, _)) = replay
        .iter()
        .find(|(_, notification, _)| matches!(notification, LayerNotification::Deleted))
    {
        tracing::debug!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            object_id = object_id,
            entry_id = %entry_id,
            "in-window replay: deleted sentinel in the in-window read; re-bootstrapping"
        );
        drop(replay);
        drop(slot);
        return handle_deleted_and_rebootstrap(ctx, object_id, RebootstrapCause::DeletedSentinel)
            .await;
    }

    // 3. Open the consumer BEFORE we deliver anything — if more entries land
    // while we're replaying the window, the consumer picks them up from the
    // last replayed ID onward. Position it at the last replayed stream ID.
    let mut consumer =
        open_consumer(ctx, object_id, StreamCursor::From(last_entry_id.clone())).await?;

    // 4. Compute every commit's diff, in order, under the same slot. Each is
    // stored with what its send needs; self-echo entries advance the tracked
    // version and store nothing.
    let mut tracked_version = ctx.have_version;
    let mut diffs: Vec<ReplayDiff> = Vec::with_capacity(replay.len());
    #[cfg(any(test, feature = "test-fault-injection"))]
    let mut computed = 0usize;
    for (_entry_id, notification, has_relay_ompf) in replay {
        let LayerNotification::Update {
            object_id: notif_oid,
            sequence_num,
            client_id: notif_client_id,
            client_ts,
            server_ts,
            // b.gze: in-window catch-up keeps reconstructing per-commit
            // diffs (the relay path is steady-state only; in-window's
            // relay-vs-reconstruct question is a separately-filed deferred
            // refinement). It uses only whether the entry carried relay bytes
            // (`has_relay_ompf`, captured at parse), never the bytes.
            relay_ompf: _,
        } = notification
        else {
            // The scan above returned on any `Deleted`.
            continue;
        };
        if notif_oid != object_id {
            // Should not happen — the stream is per-object. Log and skip so
            // we don't misattribute a diff.
            tracing::warn!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                tracked_object_id = object_id,
                notif_object_id = notif_oid,
                "in-window entry object_id mismatch; skipping"
            );
            continue;
        }

        if is_self_echo(ctx.client_id, &notif_client_id) {
            tracked_version = sequence_num;
            continue;
        }

        // An entry without relay bytes above sequence 1 is an empty commit
        // (SDD §12.2): store the empty delta based at the tracked version,
        // built with no read and no permit, which is byte-identical to what
        // its read would have built ([`empty_commit_payload`]). Sequence 1 is
        // always the `CreateLayer` seed entry, which carries no relay bytes
        // but is not empty (a restore appends no stream entry and seeds the
        // sequence far above 1), so it keeps its read below and still
        // delivers the full content. An empty commit performs no read, so it
        // detects no sequence regression and is not counted by the
        // between-computes seam below. Nor does it discover `TOO_LARGE` or the
        // OLD object's state gone (b.jvx Site A's `NotFound`): an all-empty
        // in-window replay racing an overwrite sends its empty deltas and the caught-up
        // ack for the OLD object, and the stream's `deleted` entry then
        // re-bootstraps through the steady-state overwrite path, whose
        // capacity refusal, if any, comes after content. Closest
        // deterministic test (no seam can park an all-empty replay):
        // `a_rollback_replay_whose_empty_commits_race_an_overwrite_sends_them_then_rebootstraps_in_steady_state`
        // in `tests/integration/grpc/subscribe_capacity_budget_test.rs`,
        // window `[2, 3 empty, 4 empty]`.
        if !has_relay_ompf && sequence_num > 1 {
            // At least 1: the replay starts at `have_version + 1`, so with
            // `have_version` 0 its first entry is the seed, which raised the
            // tracked version to 1 before any later entry.
            debug_assert!(
                tracked_version >= 1,
                "an empty commit's delta must be based at a sequence of at least 1"
            );
            diffs.push(ReplayDiff {
                sequence_num,
                from_version: tracked_version,
                client_ts,
                server_ts,
                diff_response: empty_commit_payload(tracked_version)?,
            });
            tracked_version = sequence_num;
            continue;
        }

        // b.jvx Site A: route `ServiceError::NotFound` from the OLD object's
        // per-commit read into the same rebootstrap path the explicit
        // `Deleted`-sentinel arm uses. The atomic read answers `NotFound` only
        // for an absent header, or one without a numeric `seq` — what a
        // concurrent `cleanup_old_object_retrying` UNLINK leaves — so matching the
        // variant alone is safe. The background cleanup can UNLINK the OLD
        // data keys while the replay computes, even when the in-window read
        // returned no `deleted` sentinel (SDD §12.3 Regime 2). All other `ServiceError`
        // variants propagate to the client via `From<ServiceError> for
        // Status` — a transient read failure (`Aborted` or `Unavailable`) as
        // `UNAVAILABLE`, which Kit retries — and so does a read-permit wait's
        // ending (a capacity refusal, shutdown, session cancellation or client
        // disconnect) and the regression ending of a read whose cut is below
        // this commit ([`per_commit_regressed`]), which is never
        // routed through NotFound recovery. Nothing has been sent yet on any
        // of these endings.
        //
        // Boxed (b.9qr, Spec SDD SR-3.5): the permit wait, the atomic read and
        // the build would otherwise sit inline in this regime's future.
        match Box::pin(replay_commit_diff(
            ctx,
            object_id,
            tracked_version,
            sequence_num,
        ))
        .await
        {
            Ok(Ok(ReplayCommit::Built(diff_response))) => diffs.push(ReplayDiff {
                sequence_num,
                from_version: tracked_version,
                client_ts,
                server_ts,
                diff_response,
            }),
            Ok(Ok(ReplayCommit::AboveCeiling(size))) => {
                // The layer is above the atomic-read limits: replaying it one
                // read per commit would multiply the fenced reads, and the
                // writer pauses each one costs, by the replay length, so take
                // the out-of-window route's one coalesced read through the
                // object's shared fenced-read flight instead (Spec SDD
                // SR-3.4). Its delivery is based at `have_version`, so the
                // diffs computed so far are dropped unsent, and the consumer
                // opened at the last entry the in-window read returned is
                // stopped before the route registers its own stream — never a
                // second registration on this one. The route reserves a reader under
                // the slot this replay holds, as the rollback-mode
                // out-of-window regime does, and gives the slot back before it
                // attaches; it never waits for a slot.
                tracing::debug!(
                    session_id = %ctx.session_id,
                    layer_name = %ctx.layer_name,
                    object_id = object_id,
                    sequence_num,
                    pin = current_seq,
                    "in-window replay: layer is above the atomic-read limits; delivering one \
                     coalesced diff through the out-of-window route instead"
                );
                drop(diffs);
                consumer.shutdown_and_release().await;
                let reserved = reserve_stream(ctx, object_id).await?;
                return Box::pin(catch_up_out_of_window(
                    ctx,
                    object_id,
                    current_seq,
                    reserved,
                    slot,
                    OutOfWindowEntry::AboveCeiling(size),
                ))
                .await;
            }
            Ok(Err(ending)) => return Err(ending),
            Err(ServiceError::NotFound(_)) => {
                drop(diffs);
                drop(slot);
                consumer.shutdown_and_release().await;
                return handle_deleted_and_rebootstrap(ctx, object_id, RebootstrapCause::StateGone)
                    .await;
            }
            Err(other) => return Err(other.into()),
        }
        tracked_version = sequence_num;

        // Test-only: synchronization seam. If installed with
        // `new_between_replay_computes`, park here — after the FIRST
        // per-commit read and build (an empty commit takes no read and never
        // reaches here), holding the catch-up slot, with nothing sent —
        // so a test can land an overwrite and its cleanup between two
        // computes. Gated behind `test-fault-injection` so release builds
        // never see this branch.
        #[cfg(any(test, feature = "test-fault-injection"))]
        {
            computed += 1;
            if computed == 1 {
                if let Some(seam) = ctx.svc.catch_up_state_read_seam_for_test.as_ref() {
                    seam.arrive_between_replay_computes().await;
                }
            }
        }
    }

    // 5. Everything is read and built: release the slot, end the catch-up
    // before its first message if its stream registration ended meanwhile,
    // then send.
    drop(slot);
    end_catch_up_if_stream_ended(ctx, &consumer)?;
    // Accumulate the per-commit OMPF bytes delivered across the replay, the
    // empty commits' deltas included, for the
    // `live_subscribe_catch_up_delivered_bytes` metric (Plan b.msb).
    // Self-echo-suppressed entries deliver nothing. Consuming
    // the vector frees each diff after its send; a send that ends the
    // subscription drops the rest on return. Its status is the subscription's
    // ending, returned as-is.
    let mut delivered_bytes: u64 = 0;
    for diff in diffs {
        delivered_bytes += send_per_commit_payload(
            ctx,
            object_id,
            diff.from_version,
            diff.sequence_num,
            diff.client_ts,
            diff.server_ts,
            &diff.diff_response,
            SendKind::CatchUp,
        )
        .await?;
    }

    Ok(CatchUpOutcome {
        tracked_object_id: object_id,
        tracked_version,
        consumer,
        catchup_mode: "legacy",
        delivered_bytes,
    })
}

/// One commit of a rollback-mode in-window replay, computed under the
/// catch-up slot and sent after it is released ([`catch_up_in_window`]).
struct ReplayDiff {
    /// The commit's sequence number, which labels its payload.
    sequence_num: u64,
    /// The tracked version the diff is based at.
    from_version: u64,
    client_ts: u64,
    server_ts: i64,
    diff_response: DiffResponse,
}

/// Regime 2 `Snapshot` arm (Plan b.msb, SR-2.1): deliver one re-established
/// coalesced diff based at `have_version` to an in-window reconnector.
///
/// NOT a full-state snapshot, despite the `Snapshot` mode name (which selects
/// the corrected builders across all regimes, not a delivery shape): this arm
/// emits `is_diff=true`, `base_version=have_version`, carrying only net-live
/// content newer than `have_version` plus the unified retraction set. The
/// client's resulting scene is the FOLD of it onto the state it retained. That
/// distinction is load-bearing — reading it as a full-state payload and
/// comparing its content against the whole scene is a category error that only
/// holds when `have_version` precedes every edit (Issue b.xvq).
///
/// Structurally mirrors `catch_up_out_of_window`'s `Snapshot` arm: takes its
/// proven read ([`catch_up_proven_read`]; above the atomic-read limits, its
/// share of the object's shared fenced-read flight's read) and registers the
/// stream on the reader the caller reserved, at the read's cut, so no
/// committed entry is lost or delivered twice; builds through the always-on
/// (DR-independent, SR-4.1) faithful coalesced builder
/// (`base_version = have_version`, net-live prims only + true authoring
/// `set_order`); and preserves overwrite / OLD-object-gone handling — a
/// `deleted` sentinel at the stream's end (the caller's point read), or a
/// `NotFound` from the state read (retry-once, b.jvx / b.fgj), both route to
/// `handle_deleted_and_rebootstrap`, under their own [`RebootstrapCause`]
/// (only the sentinel observed a deletion).
///
/// Entered holding the catch-up `slot` the caller took around the in-window
/// point read ([`read_in_window_point`]), and the stream reader the caller
/// `reserved` before the slot wait;
/// the proven read (its read permit inside the slot), the stream registration
/// and the build run under the slot, and it is released before the delivery
/// and before any re-bootstrap. Above the atomic-read limits the slot is
/// given back before the catch-up attaches to the flight, and the build runs
/// in its build turn instead ([`take_build_turn`]). `current_seq` is the
/// header sequence read before the capacity waits, P — the proven read's
/// regression pin. Before the delivery it ends the catch-up if its stream
/// registration already ended ([`end_catch_up_if_stream_ended`]).
async fn catch_up_in_window_snapshot(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    deleted_sentinel: Option<String>,
    current_seq: u64,
    reserved: ReservedStream,
    slot: CatchUpPermit,
) -> Result<CatchUpOutcome, Status> {
    ctx.outcome
        .set_regime(SubscriptionOutcomeCells::REGIME_IN_WINDOW);
    let mut slot = Some(slot);

    // Overwrite handling: a `deleted` sentinel at or above `{have_version+1}-0`
    // means the OLD object is gone — re-resolve via the mapping and either
    // terminate or re-bootstrap on the new object's stream. Mirrors the legacy
    // arm's `Deleted` branch, decided up-front from the point read
    // (`read_in_window_point`: a sentinel can only be the stream's last
    // entry), since this arm reconstructs no per-commit diffs.
    if let Some(entry_id) = deleted_sentinel {
        tracing::debug!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            object_id = object_id,
            entry_id = %entry_id,
            "in-window snapshot: deleted sentinel at the stream's end; re-bootstrapping"
        );
        // Release the slot and the OLD stream's reader reservation before the
        // re-bootstrap reserves the new object's (it takes a slot of its own).
        drop(slot);
        drop(reserved);
        return handle_deleted_and_rebootstrap(ctx, object_id, RebootstrapCause::DeletedSentinel)
            .await;
    }

    // Test-only: synchronization seam (b.vd8). If installed with
    // `new_before_coalesced`, park here — after P, the point read and the capacity
    // waits, holding the catch-up slot, before the read permit and the state
    // read — so a test can land a commit after P on the in-window regime and
    // assert that the catch-up serves and acks it exactly once. Parks once per
    // catch-up, before its proven read (the atomic attempt or the attach).
    // Gated behind `test-fault-injection` so release builds never see this
    // branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_state_read_seam_for_test.as_ref() {
        seam.arrive_at_pre_coalesced().await;
    }

    // The proven read, and the stream registered at its cut. Do NOT register
    // at a legacy `From(<last retained entry>)` cursor here: it sits at or
    // below the header, so it would replay commits the coalesced diff already
    // folded in. The cut is at or ahead of everything the point read saw — it
    // was read before the state read — and strictly ahead when commits landed
    // during the capacity waits (the reservation queued none of those; the
    // coalesced diff folds them in). A `NotFound` from the read (the OLD object's keys
    // UNLINKed by `cleanup_old_object_retrying`, b.jvx / SR-2.4 b.fgj) takes the shared
    // recovery policy inside.
    let mut reserved = Some(reserved);
    let read = Box::pin(catch_up_proven_read(
        ctx,
        object_id,
        current_seq,
        "in_window",
        &mut reserved,
        &mut slot,
    ))
    .await?;
    let (mut consumer, snapshot) = match read {
        CatchUpRead::Proven { snapshot, consumer } => (consumer, snapshot),
        CatchUpRead::Rebootstrapped(outcome) => return Ok(outcome),
    };
    // This regime runs in snapshot mode only, so the build is the faithful
    // coalesced one, under the external-value miss protocol (b.yuo).
    let snapshot = Box::pin(take_build_turn(ctx, snapshot, &consumer)).await?;
    let built = Box::pin(build_catch_up(
        ctx,
        object_id,
        CatchUpBuild::Coalesced,
        snapshot,
        "in_window",
        &mut slot,
        Some(&mut consumer),
    ))
    .await;
    // Built: the build gave back the snapshot handle (on the fenced path, the
    // build turn and the place in the flight with it) as it returned, and a
    // re-read after a miss replaced `consumer` in place (and gave the slot
    // back if it attached to a flight); give back the slot, if still held,
    // before the delivery, and end the catch-up before its first message if
    // its stream registration ended meanwhile.
    drop(slot);
    let Some((diff_response, _)) = cut_based_payload(built?)? else {
        // The miss protocol's re-read answered `NotFound` (the OLD object's
        // keys UNLINKed, or the state lost), and its read hold is released:
        // stop the stream (a no-op: the re-read released its registration),
        // then the recovery the first read would have run, before any
        // content. A restored layer is caught up again through Regime 3,
        // which builds the same coalesced diff (`RestoredRerun`).
        consumer.shutdown_and_release().await;
        return Box::pin(recover_after_reread_not_found(
            ctx,
            object_id,
            RestoredRerun::OutOfWindow { current_seq },
        ))
        .await;
    };
    end_catch_up_if_stream_ended(ctx, &consumer)?;

    // b.vd8: content header and ack both come from ONE `CatchUpContent` — see
    // the ack-≤-content invariant there and on `deliver_catch_up_content`.
    // A coalesced `base=have_version` delivery is `"legacy"` for the
    // `catchup_mode` attribute — `"snapshot"` is reserved for the from-empty
    // full-state path only (Plan b.msb taxonomy, matching
    // `catch_up_out_of_window`).
    let content =
        deliver_catch_up_content(ctx, object_id, &diff_response, "in_window", "legacy").await?;

    tracing::info!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        case = diff_response.case.as_label(),
        sections_count = diff_response.sections_count,
        ompf_size = diff_response.ompf_bytes.len(),
        regime = "in_window",
        // The DELIVERY shape, which is a coalesced diff based at
        // `have_version` — not the `Snapshot` mode name that selected this arm,
        // and deliberately not the `catchup_mode` METRIC label (`legacy`, set
        // above), which is a separate taxonomy this log line must not restate.
        mode = "coalesced",
        "catch-up coalesced diff sent"
    );

    Ok(CatchUpOutcome {
        tracked_object_id: object_id,
        tracked_version: content.effective_version,
        consumer,
        catchup_mode: content.catchup_mode,
        delivered_bytes: content.delivered_bytes,
    })
}

/// How [`catch_up_out_of_window`] reaches the layer's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutOfWindowEntry {
    /// Through its own proven read ([`catch_up_proven_read`]): the regime as
    /// `run_catch_up` dispatches it.
    ProvenRead,
    /// Straight onto the object's shared fenced-read flight
    /// ([`catch_up_flight_read`]), sized as [`FlightSizing`] says: a
    /// rollback-mode in-window replay whose per-commit atomic read answered
    /// `TOO_LARGE`, or that found the flight registered, under the slot it
    /// hands over (Spec SDD SR-3.4). No second atomic attempt of a layer
    /// already known to be above the limits.
    AboveCeiling(FlightSizing),
}

/// Regime 3: out-of-window — the stream no longer holds `{have_version+1}-0`
/// (trimmed before the in-window read, or, in rollback mode, during its
/// paging).
///
/// Builds one coalesced diff from the subscriber's last version to current.
/// Takes its proven read ([`catch_up_proven_read`]; above the atomic-read
/// limits, its share of the object's shared fenced-read flight's read) and
/// registers the stream on the reader the caller reserved at the read's cut,
/// so no commit falls between snapshot and stream and none lands in both.
///
/// Entered holding the catch-up `slot` the caller took around the in-window
/// `XRANGE` probe, and the stream reader the caller `reserved` (before the
/// slot wait in snapshot mode, under the slot in rollback mode); the proven
/// read (its read permit inside the slot), the stream registration and the
/// diff build run under the slot, and it is released before the delivery and
/// before any re-bootstrap. Above the atomic-read limits the slot is given
/// back before the catch-up attaches to the flight, and the build runs in its
/// build turn instead ([`take_build_turn`]). `current_seq` is the header
/// sequence the caller read before the capacity waits, P — the proven read's
/// regression pin. Before the delivery it ends the catch-up if its stream
/// registration already ended ([`end_catch_up_if_stream_ended`]).
///
/// `entry` says how the state is reached: through the proven read, or — when
/// a rollback-mode in-window replay ([`catch_up_in_window`]) hands over after
/// its per-commit atomic read answered `TOO_LARGE`, or after it found the
/// object's flight registered, under this same slot — straight onto the
/// flight ([`catch_up_flight_read`]), with no second atomic read of a layer
/// already known to be above the limits.
async fn catch_up_out_of_window(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    current_seq: u64,
    reserved: ReservedStream,
    slot: CatchUpPermit,
    entry: OutOfWindowEntry,
) -> Result<CatchUpOutcome, Status> {
    ctx.outcome
        .set_regime(SubscriptionOutcomeCells::REGIME_OUT_OF_WINDOW);
    let mut slot = Some(slot);
    let mut reserved = Some(reserved);

    // Test-only: synchronization seam (b.xmk). If installed with
    // `new_before_header_read`, pause here — holding the catch-up slot and the
    // reader reservation, with no stream registered — so a test can
    // deterministically interleave a `create_layer` overwrite between the
    // caller's reader reservation and this regime's first read of the OLD
    // object's state (the atomic read, or above the atomic-read limits the
    // shared fenced read it attaches to) — exactly the production race the
    // b.jvx Site C arm 1 `ServiceError::NotFound →
    // handle_deleted_and_rebootstrap` classifier guards. Gated behind
    // `test-fault-injection` so release builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_out_of_window_seam_for_test.as_ref() {
        seam.arrive_at_pre_header_read().await;
    }

    // Test-only: synchronization seam (b.xmk). If installed with
    // `new_before_compute_diff`, pause here — still before the read permit and
    // the state read, holding the slot and the reservation — so a test can
    // interleave a `create_layer` overwrite and its cleanup before the state
    // read: the b.jvx Site C arm 2 `ServiceError::NotFound →
    // handle_deleted_and_rebootstrap` classifier. The header and the state are
    // one read (atomic or fenced), so this point and the one above sit back to
    // back; both park once per catch-up, before its proven read (the atomic
    // attempt or the attach). Gated behind `test-fault-injection` so release
    // builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_out_of_window_seam_for_test.as_ref() {
        seam.arrive_at_pre_compute_diff().await;
    }

    // b.jvx Site C: the OLD-object-gone race (a `CreateLayer` overwrite's
    // `cleanup_old_object_retrying` UNLINKing the OLD object's keys) can hit this
    // regime's state read, so the proven read runs under the shared recovery
    // policy — SR-2.4 (b.fgj): on `Restored` retry the read exactly once; on
    // `NotRecoverable` drop the reservation and fall through to the
    // re-bootstrap path; on `Error(_)` classify per b.33e (`Transient` →
    // UNAVAILABLE, everything else → INTERNAL; SDD §18.1). The structure is
    // identical across both modes — the race (§7.9/§7.11 asymmetric grace)
    // hits this path regardless of mode. A replay's hand-over gives the slot
    // back and attaches to the object's flight directly, under the same
    // recovery policy.
    let read = match entry {
        OutOfWindowEntry::ProvenRead => {
            Box::pin(catch_up_proven_read(
                ctx,
                object_id,
                current_seq,
                "out_of_window",
                &mut reserved,
                &mut slot,
            ))
            .await?
        }
        OutOfWindowEntry::AboveCeiling(size) => {
            // An attached catch-up holds no slot (Spec SDD SR-7.5).
            slot = None;
            Box::pin(catch_up_flight_read(
                ctx,
                object_id,
                current_seq,
                "out_of_window",
                size,
                &mut reserved,
            ))
            .await?
        }
    };

    // Out-of-window (Regime 3): deliver everything in one coalesced diff up
    // to the current materialized state. No upper bound — the client will
    // receive a single envelope labelled with the version it serves, the
    // proven cut of the snapshot it was built from.
    //
    // Mode branch (Plan b.msb, Task 2): `Off` (rollback knob;
    // `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) runs today's legacy
    // `compute_diff_at_cut` coalesced path byte-for-byte; `Snapshot` (the shipped
    // default) routes through Task 2's DR-independent faithful coalesced
    // entrypoint, which corrects the CONTENT (net-live prims only, unified
    // ghost retraction, dense-unique `set_order`) while keeping the
    // SAME `is_diff=true` / `base_version=have_version` envelope the legacy path
    // emits. No `base_version=0` reset is ever emitted to an advanced
    // reconnector, so the client never rejects message #1.
    //
    // The in-place `base=have_version` reconciliation was validated live
    // against two real Kit clients on 2026-08-24 (b.vsw, closed PASSED; SDD
    // §12.3 Regime 3 / §12.9) — a pass that predates the b.3pe dense
    // `set_order` re-index and does not cover it, with the post-b.3pe live
    // re-check intentionally untracked. The clean-reboot fallback is
    // deliberately not implemented on this regime.
    let (mut consumer, snapshot) = match read {
        CatchUpRead::Proven { snapshot, consumer } => (consumer, snapshot),
        CatchUpRead::Rebootstrapped(outcome) => return Ok(outcome),
    };
    let snapshot = Box::pin(take_build_turn(ctx, snapshot, &consumer)).await?;
    let built = Box::pin(build_catch_up(
        ctx,
        object_id,
        CatchUpBuild::OutOfWindow,
        snapshot,
        "out_of_window",
        &mut slot,
        Some(&mut consumer),
    ))
    .await;
    // Built: the build gave back the snapshot handle (on the fenced path, the
    // build turn and the place in the flight with it) as it returned, and a
    // re-read after a miss replaced `consumer` in place (and gave the slot
    // back if it attached to a flight); give back the slot, if still held,
    // before the delivery, and end the catch-up before its first message if
    // its stream registration ended meanwhile.
    drop(slot);
    let Some((diff_response, regime_mode)) = cut_based_payload(built?)? else {
        // The miss protocol's re-read answered `NotFound` (b.jvx Site C's
        // race, or state loss, met on the re-read), and its read hold is
        // released: stop the stream (a no-op: the re-read released its
        // registration), then the recovery this regime's own read would have
        // run, before any content.
        consumer.shutdown_and_release().await;
        return Box::pin(recover_after_reread_not_found(
            ctx,
            object_id,
            RestoredRerun::OutOfWindow { current_seq },
        ))
        .await;
    };
    end_catch_up_if_stream_ended(ctx, &consumer)?;

    // b.vd8: content header and ack both come from ONE `CatchUpContent` — see
    // the ack-≤-content invariant there and on `deliver_catch_up_content`, which
    // also documents why a payload here is never empty (a gap of empty commits
    // still gets a non-empty, change-free delta).
    //
    // Both `Off` and Task 2's `Snapshot` coalesced deliveries are classified
    // `legacy`: the out-of-window path is not the from-empty faithful full-state
    // snapshot (Plan b.msb taxonomy).
    let content =
        deliver_catch_up_content(ctx, object_id, &diff_response, "out_of_window", "legacy").await?;

    tracing::info!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        case = diff_response.case.as_label(),
        sections_count = diff_response.sections_count,
        ompf_size = diff_response.ompf_bytes.len(),
        regime = "out_of_window",
        mode = regime_mode,
        "catch-up diff sent"
    );

    Ok(CatchUpOutcome {
        tracked_object_id: object_id,
        tracked_version: content.effective_version,
        consumer,
        catchup_mode: content.catchup_mode,
        delivered_bytes: content.delivered_bytes,
    })
}

// ── Steady-state loop ───────────────────────────────────────────────────────

/// Steady-state loop: forwards per-commit diffs until the stream ends or the
/// subscriber disconnects. Handles overwrites via `Deleted` + mapping re-lookup.
async fn run_steady_state(ctx: &SubscriberCtx<'_>, outcome: CatchUpOutcome) -> Result<(), Status> {
    let CatchUpOutcome {
        mut tracked_object_id,
        mut tracked_version,
        mut consumer,
        // The catch-up-phase metric was already emitted by `run_catch_up`; the
        // steady-state loop does not re-observe, so these are unused here.
        catchup_mode: _,
        delivered_bytes: _,
    } = outcome;
    // The caught-up floor: every commit at or below the ack reached the client
    // in its catch-up, so a stream trim that took only such commits is covered
    // and must not end the subscription `stream_gap`. Set before the first
    // poll, which is the first time the consumer looks at a trim marker.
    consumer.set_caught_up_floor(tracked_version);

    // `live_subscriber_updates_total`, the `enqueued` delivery lag and the
    // backlog depth for this subscription, batched; see [`DeliveryTally`]. Its
    // `Drop` flushes on every exit from this function.
    let mut tally = DeliveryTally::new(
        &ctx.svc.metrics,
        ctx.session_id,
        ctx.layer_name,
        ctx.client_id,
    );
    // The caught-up ack is enqueued (`run_catch_up`): from here on, time
    // counts toward the lagging line's trigger.
    tally.enter_steady_state();
    let flush_timer = tokio::time::sleep(DELIVERY_TALLY_FLUSH_INTERVAL);
    tokio::pin!(flush_timer);

    loop {
        // Early exit on shutdown, then on session deletion — the same order as
        // every other place both are observed, so a subscription that sees both
        // ends as a shutdown, which its client retries.
        //
        // **Neither ends the stream `OK`, and that is the point.** The Kit
        // client never retries `OK`: its subscription's finish handling runs
        // only after the client's retry classification has already decided not
        // to retry, and `OK` is not in the set its retry policy retries
        // (DEADLINE_EXCEEDED / UNAVAILABLE / RESOURCE_EXHAUSTED / INTERNAL; the
        // inspected revision and line anchors:
        // `docs-internal/cross-repo-debugging.md` § Kit client retry and
        // reconnect behaviour).
        // Once the client is caught up, the live read simply completes. So:
        //
        //   * **Graceful shutdown** ends `UNAVAILABLE "server shutting down"`.
        //     The session lives on in Valkey; the client re-subscribes with
        //     the position it holds and catches up in-window, on whichever
        //     replica the ingress picks (this one has stopped accepting
        //     connections). An `OK` here silently ended every idle subscriber's
        //     live feed on every rolling restart.
        //   * **Session deletion**, seen through the token here, ends
        //     `NOT_FOUND "session was deleted"` with `reason="session_cancel"`.
        //     A subscriber that reads the deletion's `deleted` sentinel off its
        //     stream first ends through `handle_deleted_and_rebootstrap`
        //     instead: `NOT_FOUND "layer '<name>' was deleted"`, reason
        //     `error`. The code is `NOT_FOUND` either way, and the client does
        //     not retry it; the message and the reason depend on which signal
        //     wins.
        //   * **A session found missing from the live store** (the token
        //     cancelled by the sweeper or by another stream's setup, with no
        //     `DeleteSession` seen and no deletion marker) ends `UNAVAILABLE` with
        //     `reason="session_absent"`, so the retry can recover a session a
        //     checkpoint restores (`SubscriberCtx::end_on_session_cancel`).
        //
        // The idle wait below has the same two arms in the same order; these
        // pre-checks catch a token that fired while the loop was busy
        // delivering.
        if ctx.shutdown.is_cancelled() {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            return Err(server_shutting_down());
        }
        if ctx
            .session_token()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(ctx.end_on_session_cancel());
        }
        // Early exit if the client already disconnected: `tx.closed()`
        // resolves immediately when every receiver has been dropped. Cheap
        // polling catches the case where the client went away between the
        // last send and this poll without waiting for the next entry
        // (b.jr8 H-grpc-2).
        if ctx.tx.is_closed() {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            return Ok(());
        }

        // `consumer.recv()` is the subscription's IDLE await — a quiet layer
        // parks here, on its stream multiplexer registration, for as long as
        // nothing is committed.
        // It carries no deadline, and must not: `OUTER_SEND_DEADLINE` bounds a
        // *blocked delivery*, not waiting for work (b.f4f). The three arms above
        // it are the only things that may end this wait — shutdown, deletion,
        // client disconnect, in that order. An absent session token's arm never
        // resolves. The tally's flush arm may interrupt the wait — only while
        // deliveries are pending, at most once per
        // `DELIVERY_TALLY_FLUSH_INTERVAL` — and then loops straight back.
        let maybe_pair = tokio::select! {
            biased;
            () = ctx.shutdown.cancelled() => {
                ctx.exit.set(StreamExitCell::SHUTDOWN);
                return Err(server_shutting_down());
            }
            () = cancelled_or_never(ctx.session_token()) => {
                return Err(ctx.end_on_session_cancel());
            }
            // Client disconnect while idle: end immediately, so the admission
            // and the stream multiplexer registration held by `consumer` are
            // released now rather than when the next entry arrives
            // (b.jr8 H-grpc-2).
            () = ctx.tx.closed() => {
                ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
                return Ok(());
            }
            () = &mut flush_timer, if tally.has_pending() => {
                tally.flush();
                flush_timer.as_mut().reset(
                    tokio::time::Instant::now() + DELIVERY_TALLY_FLUSH_INTERVAL,
                );
                continue;
            }
            n = consumer.recv() => n,
        };

        let Some((_entry_id, notification)) = maybe_pair else {
            // Consumer ended. Per `LayerStreamConsumer`'s ending contract, an
            // empty terminal slot is reserved for the consumer closing itself,
            // which this loop never does before reading it; every
            // server-driven ending records a `ConsumerTerminalExit`
            // (`Unavailable`, so the client re-subscribes) with the reason for
            // `stream_exits_total{reason}`:
            //   - `inner_watchdog`: the subscription's backlog, in entries or
            //     bytes, stayed at its soft threshold past
            //     `CONSUMER_SEND_WATCHDOG` once this loop had started polling
            //     it, or an entry would have taken it past a hard cap;
            //   - `shutdown`: the stream multiplexer shut down;
            //   - `stream_gap`: the consumer itself found the stream trimmed
            //     past a commit above both the floor set above (the caught-up
            //     ack) and the last entry it received;
            //   - `error`: its stream reader gave up (transient-retry budget
            //     exhausted, a permanent XREAD error, an unconvertible reply,
            //     a key that is not a stream), or the reader ended with no
            //     reason recorded — a panic, for one — in which case
            //     `take_terminal_exit` supplies `Unavailable` itself.
            //
            // An ending recorded before the catch-up's first message does not
            // reach this arm: the catch-up already ended with it, before any
            // content (`end_catch_up_if_stream_ended`). One recorded while the
            // catch-up's content was being sent does, after that content. A
            // `stream_gap` is only ever decided here, by this loop's polls.
            //
            // A layer deletion / overwrite does NOT reach this arm: `Deleted`
            // arrives as a notification on the match below, and
            // `handle_deleted_and_rebootstrap` either returns
            // `Err(Status::not_found)` or re-bootstraps onto the new object's
            // stream with a fresh consumer.
            if let Some(exit) = consumer.take_terminal_exit() {
                ctx.exit.set_reason(exit.reason);
                return Err(exit.error.into());
            }
            // Unreachable: this loop never closes the consumer before reading
            // it, so `take_terminal_exit` always yields an exit here. Should
            // that ever change, the ending still follows the one contract every
            // other server-driven ending does — a retryable status under
            // `error`, never a silent `OK` — rather than inventing a second one.
            debug_assert!(
                false,
                "a LayerStreamConsumer ended with no terminal exit while its subscription was \
                 still reading it"
            );
            return Err(ServiceError::Unavailable(
                crate::storage::notifications::READER_ENDED_WITHOUT_REASON_MSG.to_owned(),
            )
            .into());
        };

        match notification {
            LayerNotification::Deleted => {
                // Publish this generation's pending deliveries before the
                // re-bootstrap's full catch-up, which can hold the loop away
                // from its idle flush for as long as that catch-up takes.
                tally.flush();
                // Re-resolve the layer mapping. Either the layer is truly gone
                // (NOT_FOUND) or it's an overwrite (re-bootstrap on new stream).
                consumer.shutdown_and_release().await;
                let old_tracked_object_id = tracked_object_id;
                // Time the rebootstrap so the steady-state-phase overwrite regime
                // emits its own catch-up metric (Plan b.msb, Task 2). The
                // catch-up-phase `run_catch_up` observation does NOT cover this
                // fresh `CatchUpOutcome` (that path already returned before this
                // loop), so without the emission below the canonical
                // mid-session-overwrite-while-caught-up regime would record no
                // catch-up regime / mode / bytes at all.
                let rebootstrap_start = Instant::now();
                let outcome = handle_deleted_and_rebootstrap(
                    ctx,
                    tracked_object_id,
                    RebootstrapCause::DeletedSentinel,
                )
                .await?;
                tracked_object_id = outcome.tracked_object_id;
                tracked_version = outcome.tracked_version;
                // Emit exactly one catch-up regime observation for this
                // rebootstrap transition, mirroring `run_catch_up`'s single
                // exactly-once emission. Labelled `brand_new` to match the branch
                // `handle_deleted_and_rebootstrap` selects on overwrite
                // (`catch_up_brand_new` → `send_full_state`), and carries the
                // rebootstrap outcome's `catchup_mode` + `delivered_bytes`. Only
                // the rebootstrap transition observes — steady-state `Update`s do
                // not.
                ctx.svc.metrics.observe_subscribe_regime(
                    "brand_new",
                    &ctx.svc.service_label,
                    rebootstrap_start.elapsed(),
                    outcome.catchup_mode,
                    outcome.delivered_bytes,
                );
                consumer = outcome.consumer;
                // The new object's caught-up floor, before its first poll.
                consumer.set_caught_up_floor(tracked_version);
                // Defensive cross-generation guard (release no-op): a mid-session
                // overwrite rebootstrap always lands on a DISTINCT new generation,
                // and the reassigned `tracked_object_id` must equal the outcome's
                // — catch-up never spans generations (SR-2.3). Old-generation
                // state / sequence must never leak across the boundary.
                debug_assert_ne!(
                    tracked_object_id, old_tracked_object_id,
                    "steady-state rebootstrap must land on a new object_id generation"
                );
                debug_assert_eq!(
                    tracked_object_id, outcome.tracked_object_id,
                    "tracked_object_id must match the rebootstrap outcome's generation"
                );
                // Steady-state-phase rebootstrap (mid-session overwrite arriving
                // via XREAD BLOCK). The catch-up phase already returned before we
                // reached this loop, so its `run_catch_up` terminator does NOT
                // cover this fresh `CatchUpOutcome` — emit the caught-up
                // terminator here so the re-bootstrapped subscriber re-arms the
                // client's `OkLatest` caught-up handshake on the new object
                // (b.dy5). Carries the rebootstrap outcome's `tracked_version`.
                send_ack_header(ctx, tracked_object_id, tracked_version).await?;
                // Steady state again: the re-bootstrap's catch-up does not
                // count toward the lagging line's trigger.
                tally.enter_steady_state();
            }
            LayerNotification::Update {
                object_id: notif_oid,
                sequence_num,
                client_id: notif_client_id,
                client_ts,
                server_ts,
                relay_ompf,
            } => {
                if notif_oid != tracked_object_id {
                    // Defensive: the stream is per-object so this shouldn't
                    // happen. Log and skip — we trust the consumer's stream
                    // key, not the payload, for object identity.
                    tracing::warn!(
                        session_id = %ctx.session_id,
                        layer_name = %ctx.layer_name,
                        tracked_object_id = tracked_object_id,
                        notif_object_id = notif_oid,
                        "steady-state Update object_id mismatch; skipping"
                    );
                    continue;
                }

                // Duplicate-commit drop (b.9uq). `tracked_version` is the
                // highest sequence this subscriber has been served — set from
                // the catch-up outcome, re-set from the rebootstrap outcome,
                // and advanced to `sequence_num` on every delivered or
                // self-echo-suppressed Update below. Sequences are per-object
                // and strictly increasing, so anything at or below it has
                // already been accounted for and must not be replayed.
                //
                // Every catch-up registers its stream at its proven read's cut
                // (its own atomic read's, or its flight's shared fenced
                // read's) and acks that same cut, so no commit the snapshot
                // folded in is re-delivered live and this filter has no
                // catch-up overlap to close.
                //
                // It covers at-least-once re-delivery from the transport:
                // fred replaying an interrupted `XREAD` (the
                // `Blocking::Interrupt` / transient-retry paths re-issue from
                // the same durable cursor) and any out-of-band entry at or
                // below the tracked sequence. And it keeps the empty-commit
                // delta below (an entry with no relay bytes) from being based
                // at or above its own sequence, and drops the `CreateLayer`
                // seed entry (sequence 1) before it could be mistaken for an
                // empty commit.
                //
                // Cross-generation safety: the `object_id` mismatch check above
                // runs first, so an entry from a stale generation never reaches
                // here; and the `Deleted` rebootstrap arm reassigns
                // `tracked_object_id` and `tracked_version` together from the
                // new generation's `CatchUpOutcome` before any of its Updates
                // are processed. The comparison is therefore always within one
                // generation's sequence space.
                if sequence_num <= tracked_version {
                    tracing::debug!(
                        session_id = %ctx.session_id,
                        layer_name = %ctx.layer_name,
                        object_id = notif_oid,
                        sequence_num,
                        tracked_version,
                        "steady-state Update at or below the tracked version; \
                         dropping as a duplicate"
                    );
                    continue;
                }

                tracing::debug!(
                    session_id = %ctx.session_id,
                    layer_name = %ctx.layer_name,
                    object_id = notif_oid,
                    sequence_num,
                    subscriber_client_id = %ctx.client_id,
                    writer_client_id = %notif_client_id,
                    self_echo = is_self_echo(ctx.client_id, &notif_client_id),
                    has_relay_ompf = relay_ompf.is_some(),
                    "steady-state notification for subscriber"
                );

                if is_self_echo(ctx.client_id, &notif_client_id) {
                    tracing::debug!(
                        session_id = %ctx.session_id,
                        object_id = notif_oid,
                        sequence_num,
                        subscriber_client_id = %ctx.client_id,
                        writer_client_id = %notif_client_id,
                        "self-echo suppressed; NOT delivering to subscriber"
                    );
                    tracked_version = sequence_num;
                    continue;
                }

                // b.gze: forward the writer's raw inbound OMPF VERBATIM on the
                // steady-state path. Reconstruction from materialized state
                // destroys the writer's per-delta `set_order` (folded into a
                // cumulative cursor at apply time, SDD §7.1) and coalesces
                // burst fields to last-writer-wins, which the Kit client
                // cannot apply. The OMPF container is self-contained — the
                // OODF delta plus any inline external-value parts ride in the
                // same byte string — so relaying it preserves `is_diff` /
                // `base_version` / `set_order` exactly with no separate
                // ext-value fetch.
                //
                // An entry with no relay bytes is an empty commit (SDD
                // §12.2), and the steady-state loop never reads layer state
                // for it. The two other kinds of `ompf`-less entry cannot
                // reach here: the `CreateLayer` seed entry is sequence 1, at
                // or below every caught-up subscriber's tracked version and
                // dropped above; a pre-relay entry exists only during a
                // rollout off a build older than the relay field, which is
                // an upgrade precondition, not a case handled here. Checkpoint
                // tail capture relies on the same classification.
                //
                // Steady-state propagates every `ServiceError` to the client
                // unchanged (since nothing here reads, only the empty delta's
                // build can produce one).
                match relay_ompf {
                    Some(ompf_bytes) => {
                        let stream_header = OmpfStreamHeader {
                            object_id: tracked_object_id,
                            sequence_num,
                            server_timestamp: server_ts as u64,
                            client_timestamp: client_ts,
                        };
                        // A live delivery: its last chunk carries the commit
                        // stamp the transport stage is measured against.
                        let kind = SendKind::Live {
                            commit_end: Some(server_ts),
                        };
                        let mut chunks =
                            chunk_ompf_for_stream(&ompf_bytes, &stream_header).peekable();
                        while let Some(msg) = chunks.next() {
                            let is_last = chunks.peek().is_none();
                            send_or_disconnect_as(ctx, msg, kind.for_chunk(is_last)).await?;
                        }
                        // Every chunk of this commit is enqueued.
                        tally.record(LiveDelivery {
                            object_id: tracked_object_id,
                            sequence_num,
                            server_ts,
                            backlog: consumer.backlog(),
                        });
                        tracing::debug!(
                            session_id = %ctx.session_id,
                            object_id = tracked_object_id,
                            sequence_num,
                            subscriber_client_id = %ctx.client_id,
                            relay_bytes = ompf_bytes.len(),
                            "pushed relayed writer OMPF to subscriber"
                        );
                    }
                    None => {
                        // An empty commit: send the empty delta at its
                        // sequence, built from the tracked version with no
                        // read ([`empty_commit_payload`]). Never skip the
                        // sequence — the Kit client applies contiguous
                        // sequences only, so a gap followed by the client's
                        // own write would stall it.
                        //
                        // `tracked_version` is at least 1 here: every
                        // catch-up acks at or above the `CreateLayer` seed's
                        // sequence 1, and an entry at or below the tracked
                        // version was dropped above. At 0 a read would have
                        // planned full content, not an empty delta.
                        debug_assert!(
                            tracked_version >= 1,
                            "an empty commit's delta must be based at a sequence of at least 1"
                        );
                        let payload = empty_commit_payload(tracked_version)?;
                        let delivered = send_per_commit_payload(
                            ctx,
                            tracked_object_id,
                            tracked_version,
                            sequence_num,
                            client_ts,
                            server_ts,
                            &payload,
                            SendKind::Live {
                                commit_end: Some(server_ts),
                            },
                        )
                        .await?;
                        // Every chunk of this commit is enqueued. The empty
                        // delta is never zero bytes, so it is always counted;
                        // a zero-byte payload would reach the defensive arm,
                        // which sends nothing, and nothing sent is not a
                        // delivery.
                        if delivered > 0 {
                            tally.record(LiveDelivery {
                                object_id: tracked_object_id,
                                sequence_num,
                                server_ts,
                                backlog: consumer.backlog(),
                            });
                        }
                        tracing::debug!(
                            session_id = %ctx.session_id,
                            object_id = tracked_object_id,
                            sequence_num,
                            base_version = tracked_version,
                            subscriber_client_id = %ctx.client_id,
                            "no relay bytes: an empty commit; pushed the empty delta at its \
                             sequence with no read"
                        );
                    }
                }
                tracked_version = sequence_num;
            }
        }
    }
}

/// What sent a subscriber down [`handle_deleted_and_rebootstrap`].
///
/// The mapping re-read and the overwrite branch are identical on both causes —
/// the layer really may have been overwritten under either. What differs is
/// both **terminal** arms, mapping-present and mapping-absent alike: only
/// [`Self::DeletedSentinel`] is evidence that the layer was **deleted**, and
/// telling a subscriber its layer was deleted when its state was merely lost
/// and unrestorable is a wrong claim about durable data, not a wording nit
/// (b.hxk). An absent mapping does not supply that evidence on
/// [`Self::StateGone`] — the keyspace loss that emptied the state can equally
/// have taken the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RebootstrapCause {
    /// A `deleted` sentinel was actually read off the object's stream — the
    /// affirmative signal that `DeleteLayer`, a session cascade, or a
    /// `CreateLayer` overwrite retired this object.
    DeletedSentinel,
    /// A state read answered `NotFound` and recovery could not — or would not,
    /// with checkpointing disabled — restore the layer; or, in a rollback-mode
    /// in-window replay, a per-commit read (or its external-value miss
    /// re-read) answered `NotFound`, which re-bootstraps at once without
    /// asking recovery (b.jvx Site A). Nothing observed says the layer was
    /// deleted; its materialized state is simply gone.
    StateGone,
}

/// The terminal NOT_FOUND for a [`RebootstrapCause::StateGone`] subscriber.
///
/// Both spellings say the same two things — the layer's materialized state is
/// gone and it was not recovered — and neither claims a deletion, because on
/// this cause nothing observed one. Neither mentions a *backup* either: under
/// `LIVE_EDIT_CHECKPOINT_ENABLED=false` (the binary default) no backup
/// subsystem is consulted at all, so "no usable backup could be restored" would
/// be a claim this arm cannot make in every configuration; what is known in all
/// of them is that the state is gone and nothing brought it back.
/// `mapping_present` selects which:
/// `true` when the session mapping still names the tracked object, `false` when
/// the mapping is gone too (state loss that took the mapping with it, or a
/// `DeleteLayer` that raced the state read — indistinguishable from here, which
/// is precisely why neither wording asserts either).
///
/// Kept as one function so the two messages cannot drift apart (b.hxk).
fn state_gone_not_found(layer_name: &str, mapping_present: bool) -> Status {
    if mapping_present {
        Status::not_found(format!(
            "layer '{layer_name}' state was lost and could not be recovered"
        ))
    } else {
        Status::not_found(format!(
            "layer '{layer_name}' state and its session mapping are gone and could not be \
             recovered"
        ))
    }
}

/// The terminal NOT_FOUND for a [`RebootstrapCause::DeletedSentinel`]
/// subscriber.
///
/// Both terminal arms on that cause — mapping-absent and same-`object_id` —
/// make the identical claim, that the layer was deleted, and both are licensed
/// to make it because a `deleted` sentinel was actually read off the stream.
/// Kept as one function beside [`state_gone_not_found`] so the two spellings
/// cannot drift apart (b.hxk), and so the deletion claim has exactly one home
/// per cause.
///
/// The message text is pinned by the subscribe integration tests — change it
/// here and they change with it, deliberately.
fn deleted_not_found(layer_name: &str) -> Status {
    Status::not_found(format!("layer '{layer_name}' was deleted"))
}

/// Re-read the layer mapping after the tracked object stopped answering, then
/// either terminate with NOT_FOUND (nothing left to serve) or re-bootstrap on
/// the new object's stream (overwrite detected).
///
/// `cause` carries what brought the subscriber here; see [`RebootstrapCause`].
/// It changes the two terminal arms — mapping-absent and same-`object_id` —
/// and nothing else: never the mapping re-read, never the overwrite branch.
///
/// Every caller has released its catch-up slot before calling, so the
/// re-bootstrap's own catch-up takes a slot without holding one. The
/// overwrite branch refills the capacity budget first: every catch-up entered
/// here waits for capacity as long as a first catch-up could, in the catch-up
/// phase and in steady state alike.
async fn handle_deleted_and_rebootstrap(
    ctx: &SubscriberCtx<'_>,
    tracked_object_id: u64,
    cause: RebootstrapCause,
) -> Result<CatchUpOutcome, Status> {
    let mapping = ctx
        .svc
        .layer_store
        .get_layer_mapping(ctx.session_id, ctx.layer_name)
        .await?;

    match mapping {
        None => match cause {
            RebootstrapCause::DeletedSentinel => {
                // Layer truly deleted (DeleteLayer or session cascade): a
                // `deleted` sentinel was read off the stream AND the mapping is
                // gone — two independent records of removal agreeing.
                Err(deleted_not_found(ctx.layer_name))
            }
            RebootstrapCause::StateGone => {
                // Nothing observed a deletion, and an absent mapping is not
                // independent evidence of one on this cause: the keyspace loss
                // that made the state read answer NotFound can equally have
                // taken the mapping (the DR scenario). Report what is actually
                // known — state and mapping both gone, and nothing brought the
                // state back — and leave the deletion claim to the arm that has
                // a sentinel behind it (b.hxk).
                //
                // Traced at debug rather than WARN, unlike the mapping-present
                // arm below: there this outcome is anomalous under a live
                // mapping, whereas here a plain DeleteLayer racing the state
                // read produces exactly this shape routinely, and paging on a
                // routine race is the false-page the level split exists to
                // avoid. Silence is not the alternative, though — the arm ends
                // a subscription NOT_FOUND, so it leaves a debug-level record
                // naming the object an operator would otherwise have to infer
                // from the client's error alone.
                tracing::debug!(
                    session_id = %ctx.session_id,
                    layer_name = %ctx.layer_name,
                    tracked_object_id = tracked_object_id,
                    "layer state is gone and the session mapping is gone too; ending the \
                     subscription NOT_FOUND without claiming the layer was deleted"
                );
                Err(state_gone_not_found(ctx.layer_name, false))
            }
        },
        Some(new_oid) if new_oid != tracked_object_id => {
            // Overwrite detected. Bootstrap on the new object's stream.
            tracing::info!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                old_object_id = tracked_object_id,
                new_object_id = new_oid,
                "overwrite detected via mapping re-lookup; re-bootstrapping on new stream"
            );
            ctx.budget.refill();
            // Boxed (b.9qr): every caller would otherwise carry a whole
            // brand-new catch-up's state machine inline for this rare arm.
            Box::pin(catch_up_brand_new(ctx, new_oid)).await
        }
        Some(_same) => match cause {
            RebootstrapCause::DeletedSentinel => {
                // Same object_id. Per SDD §12.4 step 5, a `deleted` entry on an
                // object_id that the mapping still points at shouldn't happen in
                // practice (DeleteLayer removes the mapping before XADD deleted,
                // and CreateLayer overwrite only writes `deleted` to the OLD
                // object's stream). Exit NOT_FOUND defensively.
                tracing::warn!(
                    session_id = %ctx.session_id,
                    layer_name = %ctx.layer_name,
                    tracked_object_id = tracked_object_id,
                    "received Deleted on stream whose mapping still points at tracked object_id; treating as NOT_FOUND"
                );
                Err(deleted_not_found(ctx.layer_name))
            }
            RebootstrapCause::StateGone => {
                // Nothing observed a deletion. A state read answered NotFound,
                // recovery could not restore the layer, and the mapping still
                // names this object — so the layer exists as far as the session
                // is concerned and its materialized state does not. Saying "was
                // deleted" here would be a false claim about durable data, and
                // the deleted-sentinel arm's WARN would fire as a routine
                // outcome rather than the anomaly it was written for (b.hxk).
                //
                // WARN, like the arm above, for its own reason: materialized
                // layer state vanishing under a *live* mapping is anomalous in
                // every configuration, and under
                // `LIVE_EDIT_CHECKPOINT_ENABLED=false` (the binary default) it
                // is the only operator signal there is — recovery
                // short-circuits without touching the admission map, the store,
                // or the restore metrics, and there is no subscribe-termination
                // metric. What this ticket fixes on this arm is the *message*,
                // not the level. Status stays NOT_FOUND — there is no state to
                // serve and no other object to bootstrap onto — so only the
                // claim changes, not the client's control flow.
                tracing::warn!(
                    session_id = %ctx.session_id,
                    layer_name = %ctx.layer_name,
                    tracked_object_id = tracked_object_id,
                    "layer state is gone and was not recovered; the mapping still names this \
                     object, so ending the subscription NOT_FOUND without claiming the layer \
                     was deleted"
                );
                Err(state_gone_not_found(ctx.layer_name, true))
            }
        },
    }
}

// ── Diff + send helpers ─────────────────────────────────────────────────────

/// One commit of a rollback-mode in-window replay, as
/// [`replay_commit_diff`] leaves it.
enum ReplayCommit {
    /// The commit's diff, built from its atomic read.
    Built(DiffResponse),
    /// The layer is above the atomic-read limits — the atomic read answered
    /// `TOO_LARGE`, or the object's shared fenced-read flight was found
    /// registered — and the replay switches to the coalesced route, which
    /// attaches to the flight sized as this says.
    AboveCeiling(FlightSizing),
}

/// Read and build one commit of a rollback-mode in-window replay
/// ([`catch_up_in_window`]; Spec SDD SR-3.4): its atomic read under its own
/// read permit inside the replay's slot ([`read_replay_commit_snapshot`]),
/// then the diff covering `(from_version, to_version]` built from it
/// ([`build_catch_up`] with [`CatchUpBuild::PerCommit`], under the
/// external-value miss protocol). A `TOO_LARGE` answer builds nothing
/// and returns [`ReplayCommit::AboveCeiling`] with the answer's SR-11.1
/// sizing ([`FlightSizing::from_too_large`]; an early-out's fence probe is
/// left to the attach, [`read_through_flight`]); so does finding the object's
/// shared fenced-read flight registered before the read (sized as that flight
/// knows the layer), with no atomic attempt, and so does the miss protocol's
/// re-read meeting either.
/// Only a commit that owes content comes here: an empty commit takes no read
/// ([`empty_commit_payload`]).
///
/// Error layers as on [`read_catch_up_snapshot`]: the outer
/// `Err(ServiceError)` is the read's (the replay routes its `NotFound`, which
/// the miss protocol's re-read reports as well), the inner `Err(Status)` an
/// ending of the subscription, returned as-is — the permit wait's, a build's
/// failure, the miss protocol's own ending, or the regression ending
/// ([`per_commit_regressed`]) when the read's cut is below the commit.
async fn replay_commit_diff(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    from_version: u64,
    to_version: u64,
) -> Result<Result<ReplayCommit, Status>, ServiceError> {
    // A registered flight means the layer is above the atomic-read limits:
    // hand over without an atomic attempt that would only answer `TOO_LARGE`
    // (Spec SDD SR-7.5, a catch-up attaches without an atomic attempt).
    if let Some(found) = ctx.svc.snapshot_flights.lookup(object_id) {
        return Ok(Ok(ReplayCommit::AboveCeiling(FlightSizing::Known(
            found.size,
        ))));
    }
    let read = match read_replay_commit_snapshot(ctx, object_id).await? {
        Ok(read) => read,
        Err(ending) => return Ok(Err(ending)),
    };
    let commit = match read {
        SnapshotRead::Snapshot(snapshot) => {
            let built = Box::pin(build_catch_up(
                ctx,
                object_id,
                CatchUpBuild::PerCommit {
                    from_version,
                    to_version,
                },
                BuildSnapshot::Atomic(snapshot),
                "in_window",
                // A per-commit build never gives a slot back (above the
                // limits it answers `AboveCeiling`, and the replay hands over
                // with the slot it holds) and never swaps a registration (the
                // replay acks each commit at its own sequence).
                &mut None,
                None,
            ))
            .await;
            match built {
                Ok(CatchUpBuilt::Diff { diff_response, .. }) => ReplayCommit::Built(diff_response),
                Ok(CatchUpBuilt::Regressed { cut, to_version }) => {
                    return Ok(Err(per_commit_regressed(
                        ctx,
                        object_id,
                        from_version,
                        cut,
                        to_version,
                    )));
                }
                Ok(CatchUpBuilt::AboveCeiling(size)) => ReplayCommit::AboveCeiling(size),
                // The re-read answered `NotFound` (the OLD object's state
                // gone): handled as the replay's own read's `NotFound`, by its
                // arm (b.jvx Site A), which re-bootstraps.
                Ok(CatchUpBuilt::RereadNotFound) => {
                    return Err(ServiceError::NotFound(
                        "the layer's state was gone when the in-window replay re-read it"
                            .to_owned(),
                    ));
                }
                Err(ending) => return Ok(Err(ending)),
            }
        }
        SnapshotRead::TooLarge(too_large) => {
            ReplayCommit::AboveCeiling(FlightSizing::from_too_large(&too_large))
        }
    };
    Ok(Ok(commit))
}

/// The ending of a rollback-mode in-window replay whose per-commit read's
/// `cut` is below the commit it rebuilds, `to_version`.
///
/// **A cut below `to_version` ends the subscription (Spec SDD SR-3.2's rule,
/// applied to a per-commit read).** The read was taken after the commit was
/// streamed, so its cut is at or above the commit — unless the layer's
/// sequence regressed in between, as a restore with an undersized sequence
/// band does. The state then lacks changes the commit made, and a diff built
/// from it would be acked at `to_version` while missing them. The check is
/// [`compute_per_commit_diff_at_cut`]'s own, so no caller can skip it: it
/// builds nothing and answers [`PerCommitDiff::Regressed`], which
/// [`build_catch_up`] passes on as [`CatchUpBuilt::Regressed`]. This function
/// maps that refusal to the subscription's ending: it WARNs,
/// records `unverified` on `live_subscribe_stable_cut_total` under
/// `regime="in_window"` (a replay adds to that counter only when a read
/// regresses: a replay that succeeds adds +1 to
/// `live_subscribe_regime_total{regime="in_window"}` and nothing to the
/// stable-cut counter), and answers [`sequence_regressed`] — `UNAVAILABLE`,
/// before any content, since the replay sends only after every commit is
/// built. It never goes through NotFound recovery: the layer exists. The exit
/// reason stays `error`.
fn per_commit_regressed(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    from_version: u64,
    cut: u64,
    to_version: u64,
) -> Status {
    tracing::warn!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        from_version = from_version,
        upper_version = to_version,
        cut = cut,
        regime = "in_window",
        "in-window replay's per-commit atomic read's cut is below the commit it rebuilds \
         (the layer's sequence regressed, as a restore with an undersized sequence band \
         does); ending the subscription UNAVAILABLE before any content so the client \
         re-subscribes from the current header"
    );
    record_stable_cut(ctx, false, "in_window");
    sequence_regressed()
}

/// Deliver one per-commit payload labelled `to_version`: the steady-state
/// loop's empty delta for an empty commit ([`empty_commit_payload`]), or a
/// legacy in-window replay's commit ([`replay_commit_diff`], or the empty
/// delta there too). Returns the OMPF bytes delivered — the legacy in-window
/// replay sums them for the `live_subscribe_catch_up_delivered_bytes` metric
/// (Plan b.msb) — or the send's status, verbatim: it already says how the
/// subscription ends (`CANCELLED` for a client disconnect, `UNAVAILABLE` for a
/// shutdown or the slow-reader deadline, `NOT_FOUND` for a deleted session),
/// and callers return it as-is.
///
/// `kind` is what the payload is: [`SendKind::Live`] for the steady-state
/// empty delta, whose stamp goes on the payload's last chunk, and
/// [`SendKind::CatchUp`] for a legacy in-window replay's commit.
#[allow(clippy::too_many_arguments)]
async fn send_per_commit_payload(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    from_version: u64,
    to_version: u64,
    client_ts: u64,
    server_ts: i64,
    diff_response: &DiffResponse,
    kind: SendKind,
) -> Result<u64, Status> {
    if diff_response.ompf_bytes.is_empty() {
        // Unreachable: every payload handed here is an incremental delta, or
        // the seed's full content at base 0; neither is zero bytes. The
        // empty-commit delta is incremental by construction. A replayed
        // commit's diff is built from a read whose cut is at or above the
        // commit (`compute_per_commit_diff_at_cut` refuses to build otherwise,
        // and the subscription ends), above its base, so the plan is never
        // `UpToDate`: it is
        // incremental, or full content when its base is 0 — the seed entry
        // replayed for a client that sent the current `have_object_id` with
        // `have_version` 0, which is routed in-window rather than brand-new.
        // Kept as a defensive arm: should a zero-byte payload ever
        // arrive, send nothing rather than an empty message, and surface it.
        // The caller still advances its tracked version past the commit.
        tracing::warn!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            object_id = object_id,
            sequence_num = to_version,
            from_version = from_version,
            "per-commit payload was empty, which no caller builds; skipping delivery"
        );
        return Ok(0);
    }

    let delivered_bytes = diff_response.ompf_bytes.len() as u64;
    let stream_header = OmpfStreamHeader {
        object_id,
        sequence_num: to_version,
        server_timestamp: server_ts as u64,
        client_timestamp: client_ts,
    };

    let mut chunks = chunk_ompf_for_stream(&diff_response.ompf_bytes, &stream_header).peekable();
    while let Some(msg) = chunks.next() {
        let is_last = chunks.peek().is_none();
        // The send's status goes back verbatim: it already says how the
        // subscription ends (see this function's rustdoc).
        send_or_disconnect_as(ctx, msg, kind.for_chunk(is_last)).await?;
    }
    Ok(delivered_bytes)
}

// ── Catch-up builds and the external-value miss protocol (b.yuo) ───────────
//
// Every catch-up build — Regime 1's full state, a reconnector's coalesced diff
// in either regime, the rollback-mode legacy diffs and the rollback-mode
// in-window replay's per-commit diffs — goes through [`build_catch_up`], and
// [`build_catch_up_once`] is the one place a catch-up calls a
// `storage::diff_read` builder.

/// Which payload a catch-up builds from a proven snapshot: the one thing that
/// differs between the catch-up build sites ([`build_catch_up`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatchUpBuild {
    /// Regime 1 / overwrite-on-reconnect: the full state, built with no read
    /// and no send. `catchup_snapshot_enabled` (default `true`, Plan b.msb)
    /// builds the corrected late-join snapshot — net-live sections only,
    /// fields carrying true authoring `set_order`, no tombstones, real
    /// ext-value blobs, `base_version = 0` so message #1 is a valid
    /// from-empty create (SR-1.2), always-on with no
    /// `LIVE_EDIT_CHECKPOINT_ENABLED` dependency (SR-4.1); `false` builds the
    /// legacy `compute_diff_at_cut` full content. Both feed the same
    /// chunk/send tail in [`send_full_state`]. The payload's cut is the
    /// snapshot's ([`DiffCut::Atomic`]).
    FullState,
    /// A reconnector's faithful coalesced diff based at `have_version`
    /// (Regime 2's snapshot-mode arm, [`catch_up_in_window_snapshot`]).
    Coalesced,
    /// Regime 3 ([`catch_up_out_of_window`]): the faithful coalesced diff
    /// based at `have_version` in snapshot mode, the legacy version diff
    /// (`compute_diff_at_cut`) in rollback mode.
    OutOfWindow,
    /// One commit of a rollback-mode in-window replay
    /// ([`replay_commit_diff`]): the single-version diff covering
    /// `(from_version, to_version]`, version-bounded by `to_version` so
    /// commits the read's cut holds above it are left out, and acked at the
    /// commit's own sequence number.
    PerCommit { from_version: u64, to_version: u64 },
}

/// What a catch-up build came to ([`build_catch_up`]).
enum CatchUpBuilt {
    /// The payload, with the tracing mode (`"snapshot"` | `"off"`) the
    /// regimes that switch on `catchup_snapshot_enabled` log.
    Diff {
        diff_response: DiffResponse,
        mode: &'static str,
    },
    /// [`CatchUpBuild::PerCommit`] only: the read's cut is below the commit
    /// ([`PerCommitDiff::Regressed`]); nothing was built
    /// ([`per_commit_regressed`]).
    Regressed { cut: u64, to_version: u64 },
    /// The miss protocol's re-read answered `NotFound`: the layer's state was
    /// gone at the re-read (an overwrite's cleanup, or keyspace loss). Nothing
    /// was built, and no recovery was attempted — the re-read met it under
    /// the read hold, which no restore may run under. The caller, after the
    /// hold is released, handles it as its own read's `NotFound`, before any
    /// content: a cut-based caller through [`recover_after_reread_not_found`],
    /// a rollback-mode replay through its own `NotFound` arm.
    RereadNotFound,
    /// [`CatchUpBuild::PerCommit`] only: the miss protocol's re-read found
    /// the layer above the atomic-read limits (`TOO_LARGE`, or the object's
    /// shared fenced-read flight registered), sized as on
    /// [`ReplayCommit::AboveCeiling`]; the replay hands over to the coalesced
    /// route as it does for its own read.
    AboveCeiling(FlightSizing),
}

/// A cut-based build's payload and mode ([`CatchUpBuilt::Diff`]), or `None`
/// when the miss protocol's re-read answered `NotFound`
/// ([`CatchUpBuilt::RereadNotFound`]) and the caller must run its recovery
/// ([`recover_after_reread_not_found`]). The other two outcomes come only
/// from a per-commit build, so a cut-based caller meeting one is a bug,
/// answered `INTERNAL`.
fn cut_based_payload(built: CatchUpBuilt) -> Result<Option<(DiffResponse, &'static str)>, Status> {
    match built {
        CatchUpBuilt::Diff {
            diff_response,
            mode,
        } => Ok(Some((diff_response, mode))),
        CatchUpBuilt::RereadNotFound => Ok(None),
        CatchUpBuilt::Regressed { .. } | CatchUpBuilt::AboveCeiling(_) => {
            Err(ServiceError::Internal(
                "a catch-up's cut-based build answered a per-commit outcome".to_owned(),
            )
            .into())
        }
    }
}

/// Build `build` from `snapshot` once, reading its external values as
/// `ext_reads` says: the one place a catch-up calls a `storage::diff_read`
/// builder. Answers [`Built::Missing`] only under [`ExtValueReads::PROBE`],
/// and only [`CatchUpBuilt::Diff`] or (per-commit) [`CatchUpBuilt::Regressed`]
/// otherwise. Each builder is boxed (b.9qr).
async fn build_catch_up_once(
    ctx: &SubscriberCtx<'_>,
    snapshot: &AtomicSnapshot,
    build: CatchUpBuild,
    ext_reads: ExtValueReads,
) -> Result<Built<CatchUpBuilt>, ServiceError> {
    let ext_store = ctx.svc.ext_value_store.as_ref();
    let diff = |mode: &'static str| {
        move |diff_response: DiffResponse| CatchUpBuilt::Diff {
            diff_response,
            mode,
        }
    };
    let faithful = ctx.svc.catchup_snapshot_enabled;
    let built = match build {
        CatchUpBuild::FullState if faithful => Box::pin(compute_faithful_full_content_at_cut(
            snapshot,
            ext_store,
            ext_reads,
            Some(&ctx.svc.metrics),
            &ctx.svc.service_label,
        ))
        .await?
        .map(diff("snapshot")),
        CatchUpBuild::FullState => {
            // have_object_id=0 forces Case 1 (full content) in the diff plan.
            let subscriber_state = SubscriberState {
                have_object_id: 0,
                have_version: 0,
            };
            Box::pin(compute_diff_at_cut(
                snapshot,
                &subscriber_state,
                ext_store,
                ext_reads,
                None,
                "subscribe_read_layer",
            ))
            .await?
            .map(diff("off"))
        }
        CatchUpBuild::Coalesced => Box::pin(compute_faithful_coalesced_at_cut(
            snapshot,
            ctx.have_version,
            ext_store,
            ext_reads,
            Some(&ctx.svc.metrics),
            &ctx.svc.service_label,
        ))
        .await?
        .map(diff("snapshot")),
        CatchUpBuild::OutOfWindow if faithful => Box::pin(compute_faithful_coalesced_at_cut(
            snapshot,
            ctx.have_version,
            ext_store,
            ext_reads,
            Some(&ctx.svc.metrics),
            &ctx.svc.service_label,
        ))
        .await?
        .map(diff("snapshot")),
        CatchUpBuild::OutOfWindow => {
            let subscriber_state = SubscriberState {
                have_object_id: ctx.have_object_id,
                have_version: ctx.have_version,
            };
            Box::pin(compute_diff_at_cut(
                snapshot,
                &subscriber_state,
                ext_store,
                ext_reads,
                None,
                "subscribe_read_layer",
            ))
            .await?
            .map(diff("off"))
        }
        CatchUpBuild::PerCommit {
            from_version,
            to_version,
        } => {
            let subscriber_state = SubscriberState {
                have_object_id: snapshot.state.object_id,
                have_version: from_version,
            };
            Box::pin(compute_per_commit_diff_at_cut(
                snapshot,
                &subscriber_state,
                to_version,
                ext_store,
                ext_reads,
                None,
                "subscribe_read_layer",
            ))
            .await?
            .map(|commit| match commit {
                PerCommitDiff::Built(diff_response) => CatchUpBuilt::Diff {
                    diff_response,
                    mode: "off",
                },
                PerCommitDiff::Regressed { cut, to_version } => {
                    CatchUpBuilt::Regressed { cut, to_version }
                }
            })
        }
    };
    Ok(built)
}

/// Build a catch-up's payload from its proven snapshot under the
/// external-value miss protocol (b.yuo; SDD §11.2): the one call through
/// which every catch-up build site builds.
///
/// A blob a snapshot references can be gone by the time the build reads it,
/// with nothing damaged: a commit after the snapshot released it and the GC
/// collected it, which it does from 5 s after the release
/// (`GcPolicy::PRODUCTION`'s age floor), and with no age floor at all for the
/// excess over half the release budget while the release total is above it.
/// So a miss is first treated as a stale read, never as corruption:
///
/// 1. Build with [`ExtValueReads::PROBE`]: a missing blob is an answer, not
///    an error, with nothing logged and no checkpoint adapter consulted. With
///    every blob present, that is the payload, and nothing is written: only a
///    miss takes a hold.
/// 2. On a miss, count `stale_retried` on
///    `live_edit_catchup_ext_value_miss_total`, under the `read` of the
///    snapshot it was found in — `atomic`, the catch-up's own read, or
///    `shared_fenced`, its flight's — give back the stale snapshot (on the
///    fenced path, the build turn and the place in the old flight with it,
///    before anything waits on `{gc}`), and rebuild under a per-holder `{gc}`
///    read hold ([`rebuild_under_read_hold`]): take the hold
///    ([`ReadHold::take`]; a failure ends the catch-up
///    [`read_hold_refused`]) and, renewing it every few seconds throughout,
///    read the layer again at a fresh proven cut and build again
///    ([`reread_and_rebuild`]). A cut-based build first swaps its stream
///    registration for a fresh reservation, and registers it at the fresh
///    cut once that cut is checked against the first read's, so its stream
///    cursor, its ack and its payload's cut are the one fresh cut, as on
///    every catch-up. The fresh cut is the catch-up's own atomic read's,
///    under a new read permit inside its slot; or, above the atomic-read
///    limits — the stale snapshot a flight's, the object's flight
///    registered, or the atomic re-read answering `TOO_LARGE` — the object's
///    fenced-read flight's (a successor of the stale one, on the fenced
///    path), attached to with no slot and built from in a build turn. A
///    renewal the store answers lapsed (or that fails) ends the catch-up
///    [`read_hold_refused`].
/// 3. A hash still missing after the re-read is genuine only if the store
///    confirms the hold never lapsed (one more renewal, answered live —
///    [`ReadHold::confirm_live`], whose proof is the only way to build the
///    fetch-on-miss mode): then count `genuine` (under the same `read`) and
///    build the re-read's snapshot with [`ExtValueReads::recover`], today's
///    fetch-on-miss through the checkpoint adapter, `INTERNAL` when that
///    cannot restore it — still under the hold's renewal. A hold the store reports lapsed or gone ends
///    the catch-up [`read_hold_refused`], never `INTERNAL`.
///
/// The hold is released (`HDEL` of its field) on every return once the last
/// blob read is done, so a miss stops the collection of newer releases only
/// while its re-read and its rebuild run: on the atomic path that includes
/// its read-permit wait; through a flight, the flight's waits, its read and
/// the build turn wait. A catch-up cancelled meanwhile drops the
/// hold unreleased, which spawns the release; a dead process's field lapses
/// by its TTL.
///
/// Entered as the build would have been: holding the catch-up slot (in
/// `slot`) on the atomic path, or the build turn on the fenced one (`slot`
/// `None`), with the stream registered (`consumer`). Consumes the snapshot
/// handle, so a fenced catch-up's turn and place in its flight go back as
/// this returns, or, on a miss, before its hold is taken. A re-read
/// replaces the registration in `consumer` in place, and a re-read that
/// attaches to a flight gives the slot back (`slot` `None`); the caller
/// releases whatever slot is left. A per-commit build
/// ([`CatchUpBuild::PerCommit`]) passes no `consumer` and no slot: its re-read
/// swaps nothing and, above the limits, answers
/// [`CatchUpBuilt::AboveCeiling`] to its replay, which hands over with the
/// slot it holds.
///
/// [`CatchUpBuilt::RereadNotFound`] and, for a per-commit build,
/// [`CatchUpBuilt::AboveCeiling`] come only from the re-read; the caller
/// handles them as it handles its own read's `NotFound` and `TOO_LARGE`, the
/// `NotFound`'s recovery attempt included, once the hold is released.
async fn build_catch_up(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    build: CatchUpBuild,
    snapshot: BuildSnapshot,
    regime: &'static str,
    slot: &mut Option<CatchUpPermit>,
    consumer: Option<&mut LayerStreamConsumer>,
) -> Result<CatchUpBuilt, Status> {
    // Test-only: park between the snapshot and the build's first blob read
    // (the probe), so a test can land a commit that releases a blob the
    // snapshot references and let the GC collect it — the stale-miss race.
    // Parks once per installed seam, at its first build. Gated behind
    // `test-fault-injection` so release builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_state_read_seam_for_test.as_ref() {
        seam.arrive_at_pre_ext_probe().await;
    }

    let first_cut = snapshot.cut;
    let missing = match Box::pin(build_catch_up_once(
        ctx,
        &snapshot,
        build,
        ExtValueReads::PROBE,
    ))
    .await?
    {
        Built::Ready(built) => return Ok(built),
        Built::Missing(missing) => missing,
    };
    // The stale snapshot goes back here, on both paths, before the hold is
    // taken: all the re-read needs of it is its cut (copied above) and, on
    // the fenced path, its measured size, which sizes the successor flight
    // exactly, with no fence probe. On the fenced path that gives back the
    // build turn, the snapshot handle and the place in the old flight
    // together, and it comes before the re-attach, which it must: a
    // successor waits for the old flight's `done`, which a place still held
    // would keep from firing. Nothing of the old flight waits on the `{gc}`
    // round trip. A successor may therefore fix its cut before the hold is
    // acknowledged. Joining it is still sound, for the reason a pin read is:
    // attach closes before the fence release, and no commit lands inside the
    // epoch (`SnapshotFlights::attach_or_create`). The atomic path keeps its
    // slot across the take: its re-read runs in it.
    let (route, read) = match snapshot {
        BuildSnapshot::Atomic(snapshot) => {
            drop(snapshot);
            (RereadRoute::Own, CATCHUP_EXT_VALUE_MISS_READ_ATOMIC)
        }
        BuildSnapshot::Turn(turn) => {
            let sizing = FlightSizing::Known(turn.measured_size());
            drop(turn);
            (
                RereadRoute::Flight(sizing),
                CATCHUP_EXT_VALUE_MISS_READ_SHARED_FENCED,
            )
        }
    };
    ctx.svc
        .metrics
        .inc_catchup_ext_value_miss(CATCHUP_EXT_VALUE_MISS_OUTCOME_STALE_RETRIED, read);
    tracing::debug!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        regime = regime,
        read = read,
        cut = first_cut,
        missing = missing.len(),
        "catch-up: external values its snapshot references were collected after it was read; \
         re-reading under the GC read hold"
    );
    Box::pin(rebuild_under_read_hold(
        ctx,
        object_id,
        build,
        MissRun {
            first_cut,
            read,
            regime,
        },
        route,
        CatchUpHeld { slot, consumer },
    ))
    .await
}

/// One run of a catch-up's miss protocol: what its re-read is checked
/// against and labelled with ([`build_catch_up`]).
#[derive(Clone, Copy)]
struct MissRun {
    /// The first read's cut. A cut-based re-read's cut below it means the
    /// layer's sequence regressed between the two reads.
    first_cut: u64,
    /// The `read` label of the snapshot the miss was found in
    /// (`CATCHUP_EXT_VALUE_MISS_READ_*`): the miss counter's cells and the
    /// hold's DEBUG line carry it.
    read: &'static str,
    /// The catch-up's regime, for its log lines.
    regime: &'static str,
}

/// What a catch-up build holds that its miss protocol's re-read may change
/// ([`build_catch_up`]).
struct CatchUpHeld<'a> {
    /// The catch-up slot, `None` when the catch-up holds none (attached to a
    /// flight). Given back, set to `None`, when the re-read attaches to a
    /// flight.
    slot: &'a mut Option<CatchUpPermit>,
    /// A cut-based build's stream registration, swapped for a fresh
    /// reservation before the re-read and replaced in place by its
    /// registration at the re-read's cut. `None` for a per-commit build.
    consumer: Option<&'a mut LayerStreamConsumer>,
}

/// A build with [`ExtValueReads::recover`] answered [`Built::Missing`],
/// which its builders never do.
fn unreachable_recover_miss() -> Status {
    ServiceError::Internal(
        "a catch-up build with fetch-on-miss answered a missing external value".to_owned(),
    )
    .into()
}

/// The status a catch-up ends with when it could not hold the `{gc}` read
/// hold across its re-read after an external-value miss ([`build_catch_up`]):
/// the take or a renewal failed, or the store answered that the holder's field
/// had lapsed (a renewal that came too late, or a field lost to a `{gc}`
/// failover), so the re-read's blobs may have been collected meanwhile.
/// Before any content. Retryable: a re-subscribe reads afresh. Never
/// `INTERNAL`: nothing shows the layer is damaged. It carries no refusal
/// detail (no retry hint), like the other fault endings `sequence_regressed`
/// and `fenced_read_wait_cap`: it is not load-shedding, and a prompt
/// re-subscribe usually reads with no miss at all.
fn read_hold_refused() -> Status {
    Status::unavailable(
        "an external value this catch-up's snapshot referenced was collected after the \
         snapshot was read, and the catch-up could not hold garbage collection off while it \
         read the layer again; retry",
    )
}

/// What the re-read and rebuild under the read hold came to
/// ([`reread_and_rebuild`]).
enum Reread {
    /// Built, or an outcome the caller handles ([`CatchUpBuilt`]).
    Done(CatchUpBuilt),
    /// External values the re-read references are still not stored. Genuine
    /// only if the store confirms the hold never lapsed, which the caller
    /// checks. Carries the re-read's snapshot as the build had it — its own,
    /// or its build turn in the object's flight — for the fetch-on-miss build
    /// on it.
    StillMissing(BuildSnapshot),
}

/// Where the miss protocol's re-read gets its fresh cut
/// ([`reread_and_rebuild`]).
enum RereadRoute {
    /// The build's own read: a cut-based build's atomic read — or, above the
    /// atomic-read limits (the object's flight registered, or the read
    /// answering `TOO_LARGE`), the object's flight — and a per-commit build's
    /// replay read.
    Own,
    /// The object's fenced-read flight — a successor of the flight whose
    /// shared snapshot missed, which the catch-up has already left — sized
    /// from that snapshot's measured tallies (so no fence probe is needed).
    Flight(FlightSizing),
}

/// Steps 2 and 3 of [`build_catch_up`], entered once the caller has given
/// back the stale snapshot (on the fenced path, its build turn and its place
/// in the old flight with it), so nothing of an earlier flight is held across
/// the `{gc}` round trip: take a `{gc}` read hold ([`ReadHold::take`]; a
/// failure ends the catch-up [`read_hold_refused`]), run everything after it
/// under the hold's renewal ([`under_read_hold`]) — the re-read and rebuild
/// along `route` ([`reread_and_rebuild`]) and, for a hash still missing, the
/// server-side genuine check and the fetch-on-miss build
/// ([`genuine_miss_rebuild`]) — then release the hold, on every return. A
/// catch-up on the atomic path still holds its slot here: its re-read runs
/// in it.
///
/// The hold is released ([`ReadHold::release`]) as soon as the last blob
/// read is done, whatever the outcome, so it stops the collection of newer
/// releases only for its re-read and its rebuild. On the atomic path that is
/// the wait for a read permit (first in, first out behind at most one atomic
/// read per other catch-up slot: about a quarter second at the defaults,
/// never past the subscribe's capacity budget), one atomic read and one build
/// — sub-second at the defaults. The hold is taken before the permit wait,
/// not after: a permit is held across its atomic read alone, never across a
/// `{gc}` round trip. Above the atomic-read limits it is the object's
/// fenced-read flight's waits (for a successor of a closed flight, its
/// predecessor's remaining builds; then its capacity waits, charged to the
/// subscribe's capacity budget), its fenced read and the wait for a build
/// turn, then one build: typically seconds. The hold is acknowledged before
/// the attach, as the attach requires (`SnapshotFlights::attach_or_create`).
/// Each released hold logs one DEBUG
/// line with its `read` and `held_ms`, from before the take to after the
/// release. A cancelled catch-up drops it unreleased, which spawns the
/// release (and logs no line); a dead process's field lapses one TTL after
/// its last write.
async fn rebuild_under_read_hold(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    build: CatchUpBuild,
    run: MissRun,
    route: RereadRoute,
    held: CatchUpHeld<'_>,
) -> Result<CatchUpBuilt, Status> {
    let started = std::time::Instant::now();
    let hold = match ReadHold::take(&ctx.svc.valkey, &ctx.svc.key_schema, object_id).await {
        Ok(hold) => hold,
        Err(error) => {
            tracing::warn!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                object_id = object_id,
                error = %error,
                "catch-up: could not take the GC read hold it needs to re-read the layer after an \
                 external-value miss; ending the subscription UNAVAILABLE before any content"
            );
            return Err(read_hold_refused());
        }
    };
    let built = Box::pin(under_read_hold(
        ctx,
        object_id,
        &hold,
        rebuild_and_check(ctx, object_id, &hold, build, run, route, held),
    ))
    .await;
    hold.release().await;
    tracing::debug!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        read = run.read,
        held_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        ok = built.is_ok(),
        "catch-up: released the GC read hold it took after an external-value miss"
    );
    built
}

/// Run `work` while renewing `hold` every [`ReadHold::renew_every`], so a
/// re-read or rebuild slower than the hold's TTL keeps it
/// ([`renewing_while`]). A renewal that fails, or that the store answers
/// [`ReadHoldRenewal::Lapsed`], ends the catch-up [`read_hold_refused`] at
/// once ([`read_hold_lost`]); the caller releases the hold either way.
async fn under_read_hold<F>(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    hold: &ReadHold,
    work: F,
) -> Result<CatchUpBuilt, Status>
where
    F: std::future::Future<Output = Result<CatchUpBuilt, Status>>,
{
    match renewing_while(hold.renew_every(), || hold.renew(), work).await {
        Ok(built) => built,
        Err(error) => Err(read_hold_lost(ctx, object_id, error.as_ref())),
    }
}

/// [`under_read_hold`]'s loop: poll `work` to completion while starting a
/// `renew` every `every`, and return `work`'s output. The first renewal
/// starts one interval in. A renewal answering [`ReadHoldRenewal::Lapsed`]
/// ends the loop `Err(None)`, a failed one `Err(Some(error))`, at once.
///
/// A renewal runs concurrently with `work`, never instead of it: `work` keeps
/// being polled while a renewal is in flight, however long the store takes
/// to answer (up to `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` per renewal — twice
/// that when a `NOSCRIPT` reply makes it reload the script — with no bound
/// when that is 0). That matters for any `work` that drives a read
/// other futures wait on — a stalled poll would stall the read itself. At
/// most one renewal is in flight: the interval is not polled while one is,
/// and once it resolves the next starts when the interval next ticks (at
/// once, if the tick fell due during the renewal).
///
/// If `work` finishes while a renewal is in flight, the renewal is dropped
/// unawaited and the caller releases the hold as usual. That is safe: the
/// renewal's write may still reach the store, but a renewal only extends a
/// field that is still live and never revives a missing one
/// (`ext_read_hold.lua`). Landing before the release's `HDEL`, it extends a
/// field the `HDEL` then deletes; landing after it, it finds the field gone
/// and writes nothing; if the `HDEL` failed, the field still lapses one TTL
/// after its last write.
async fn renewing_while<T, W, R, RF>(
    every: std::time::Duration,
    mut renew: R,
    work: W,
) -> Result<T, Option<ServiceError>>
where
    W: std::future::Future<Output = T>,
    R: FnMut() -> RF,
    RF: std::future::Future<Output = Result<ReadHoldRenewal, ServiceError>>,
{
    use futures::future::{FusedFuture, FutureExt};

    let mut ticks = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut work = std::pin::pin!(work);
    // The renewal in flight, if any: terminated when none is.
    let mut renewal = std::pin::pin!(futures::future::Fuse::terminated());
    loop {
        tokio::select! {
            biased;
            done = &mut work => return Ok(done),
            renewed = &mut renewal, if !renewal.is_terminated() => match renewed {
                Ok(ReadHoldRenewal::Live) => {}
                Ok(ReadHoldRenewal::Lapsed) => return Err(None),
                Err(error) => return Err(Some(error)),
            },
            _ = ticks.tick(), if renewal.is_terminated() => renewal.set(renew().fuse()),
        }
    }
}

/// The re-read and rebuild ([`reread_and_rebuild`]), then, for a hash still
/// missing, the genuine-miss build ([`genuine_miss_rebuild`]). Runs under
/// [`under_read_hold`].
async fn rebuild_and_check(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    hold: &ReadHold,
    build: CatchUpBuild,
    run: MissRun,
    route: RereadRoute,
    held: CatchUpHeld<'_>,
) -> Result<CatchUpBuilt, Status> {
    match Box::pin(reread_and_rebuild(ctx, object_id, build, run, route, held)).await? {
        Reread::Done(built) => Ok(built),
        // A re-read through a flight keeps its build turn in that flight
        // across the genuine check and the fetch-on-miss build, and gives it
        // back as this returns.
        Reread::StillMissing(snapshot) => {
            Box::pin(genuine_miss_rebuild(
                ctx, object_id, hold, build, &snapshot, run,
            ))
            .await
        }
    }
}

/// Step 3 of [`build_catch_up`]: a hash the re-read references is still not
/// stored. It is genuine only if the hold has protected the re-read without a
/// gap, which the store decides: one renewal, [`ReadHold::confirm_live`] — the
/// clocks and the field are the `{gc}` primary's, so a delayed write or a
/// failover cannot pass for a live hold. Live: its proof builds the
/// fetch-on-miss mode, so count `genuine` (under the miss's `read`) and build
/// the re-read's `snapshot` with [`ExtValueReads::recover`] (fetch-on-miss
/// through the checkpoint adapter, then `INTERNAL`), still under the hold's
/// renewal. Lapsed, missing or failed: [`read_hold_refused`], never
/// `INTERNAL`.
async fn genuine_miss_rebuild(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    hold: &ReadHold,
    build: CatchUpBuild,
    snapshot: &AtomicSnapshot,
    run: MissRun,
) -> Result<CatchUpBuilt, Status> {
    let recover = match hold.confirm_live().await {
        Ok(Some(live)) => ExtValueReads::recover(&live),
        Ok(None) => return Err(read_hold_lost(ctx, object_id, None)),
        Err(error) => return Err(read_hold_lost(ctx, object_id, Some(&error))),
    };
    ctx.svc
        .metrics
        .inc_catchup_ext_value_miss(CATCHUP_EXT_VALUE_MISS_OUTCOME_GENUINE, run.read);
    tracing::warn!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        regime = run.regime,
        read = run.read,
        cut = snapshot.cut,
        "catch-up: an external value the layer references is still not stored after a re-read \
         under a GC read hold the store confirmed live; building with fetch-on-miss through the \
         checkpoint adapter"
    );
    match Box::pin(build_catch_up_once(ctx, snapshot, build, recover)).await? {
        Built::Ready(built) => Ok(built),
        Built::Missing(_) => Err(unreachable_recover_miss()),
    }
}

/// The ending of a catch-up whose read hold could not be shown to have lived
/// through its re-read: a renewal failed (`error`), or the store answered
/// that the field had lapsed or was gone (`None`). WARNs and answers
/// [`read_hold_refused`].
fn read_hold_lost(ctx: &SubscriberCtx<'_>, object_id: u64, error: Option<&ServiceError>) -> Status {
    match error {
        Some(error) => tracing::warn!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            object_id = object_id,
            error = %error,
            "catch-up: renewing the GC read hold failed, so a re-read's external values may have \
             been collected; ending the subscription UNAVAILABLE before any content"
        ),
        None => tracing::warn!(
            session_id = %ctx.session_id,
            layer_name = %ctx.layer_name,
            object_id = object_id,
            "catch-up: the store answered that the GC read hold had lapsed (a late renewal, or a \
             hold lost to a failover), so a re-read's external values may have been collected; \
             ending the subscription UNAVAILABLE before any content"
        ),
    }
    read_hold_refused()
}

/// The re-read and rebuild of [`rebuild_under_read_hold`], run while the
/// `{gc}` read hold lives.
///
/// **A per-commit build** ([`CatchUpBuild::PerCommit`]) re-reads as
/// `replay_commit_diff` reads: the object's flight registered, or its replay
/// read ([`read_replay_commit_snapshot`], under a new read permit inside the
/// replay's slot) answering `TOO_LARGE`, answers
/// [`CatchUpBuilt::AboveCeiling`] for the replay to hand over; it swaps no
/// registration and checks its own cut against its commit.
///
/// **A cut-based build** first swaps its stream registration (`held.consumer`)
/// for a fresh reservation: [`reserve_stream`] first — it never waits, since
/// the registration it replaces keeps the reader bound — then
/// `shutdown_and_release` of the old registration, so no backlog queues
/// behind the first read's cut while the re-read waits. Then it gets a fresh
/// proven cut:
///
/// - With `route` [`RereadRoute::Own`] and no flight registered for the
///   object: its own atomic read ([`read_catch_up_snapshot`]), under a new
///   read permit inside the catch-up slot it still holds — a wait charged to
///   the capacity budget like every permit wait, ending only at the budget's
///   expiry. A `TOO_LARGE` answer gives the slot back and goes to the flight
///   below, sized from the answer ([`FlightSizing::from_too_large`]).
/// - With the object's flight registered: the slot is given back and the
///   flight read below, sized as that flight knows the layer.
/// - With `route` [`RereadRoute::Flight`] (the stale snapshot was a flight's):
///   the flight read below, sized as given.
/// - **The flight read**: attach to the object's flight (on the fenced path, a
///   successor of the stale one; otherwise an open flight joined, or a new one
///   created) and wait for its read ([`read_through_flight`]) with no slot,
///   its waits charged to the capacity budget; then wait for a build turn in
///   it ([`take_build_turn`], which ends the catch-up if the new
///   registration reaches the hard backlog cap meanwhile).
///
/// A cut below `run.first_cut` means the layer's sequence regressed between
/// the two reads (Spec SDD SR-3.2): the payload could not cover the commits
/// between them, so the catch-up ends [`sequence_regressed`], through
/// [`catch_up_regressed`] under the `read` the re-read took (`atomic` or
/// `shared_fenced`). Otherwise the reservation is registered at the cut and
/// replaces the caller's consumer in place, so the catch-up's ack, its
/// payload's cut and its stream cursor are the one cut.
///
/// Every read's ending (a capacity refusal, `fenced_read_wait_cap`, shutdown,
/// session cancellation, client disconnect) is returned as-is, dropping the
/// reservation; a `NotFound` is [`CatchUpBuilt::RereadNotFound`], with no
/// recovery attempted under the hold; any other read error is its status.
/// Then the build again with [`ExtValueReads::PROBE`]: its payload, or
/// [`Reread::StillMissing`].
async fn reread_and_rebuild(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    build: CatchUpBuild,
    run: MissRun,
    route: RereadRoute,
    held: CatchUpHeld<'_>,
) -> Result<Reread, Status> {
    let CatchUpHeld { slot, consumer } = held;
    if matches!(build, CatchUpBuild::PerCommit { .. }) {
        debug_assert!(
            matches!(route, RereadRoute::Own),
            "a per-commit build reads atomically, never through a flight's shared snapshot"
        );
        // As `replay_commit_diff`: a registered flight means the layer is
        // above the atomic-read limits, so hand over with no atomic attempt.
        if let Some(found) = ctx.svc.snapshot_flights.lookup(object_id) {
            return Ok(Reread::Done(CatchUpBuilt::AboveCeiling(
                FlightSizing::Known(found.size),
            )));
        }
        let snapshot = match read_replay_commit_snapshot(ctx, object_id).await {
            Ok(Ok(SnapshotRead::Snapshot(snapshot))) => snapshot,
            // As `replay_commit_diff`: an early-out's fence probe is left to
            // the attach (`read_through_flight`).
            Ok(Ok(SnapshotRead::TooLarge(too_large))) => {
                return Ok(Reread::Done(CatchUpBuilt::AboveCeiling(
                    FlightSizing::from_too_large(&too_large),
                )));
            }
            Ok(Err(ending)) => return Err(ending),
            Err(ServiceError::NotFound(_)) => {
                return Ok(Reread::Done(CatchUpBuilt::RereadNotFound));
            }
            Err(other) => return Err(other.into()),
        };
        return match Box::pin(build_catch_up_once(
            ctx,
            &snapshot,
            build,
            ExtValueReads::PROBE,
        ))
        .await?
        {
            Built::Ready(built) => Ok(Reread::Done(built)),
            Built::Missing(_) => Ok(Reread::StillMissing(BuildSnapshot::Atomic(snapshot))),
        };
    }
    let Some(consumer) = consumer else {
        return Err(ServiceError::Internal(
            "a catch-up's cut-based re-read was given no stream registration to swap".to_owned(),
        )
        .into());
    };

    // The swap: reserve first, while the registration being replaced keeps
    // the reader bound, then release the old one.
    let reserved = reserve_stream(ctx, object_id).await?;
    consumer.shutdown_and_release().await;

    let sizing = match route {
        RereadRoute::Flight(sizing) => sizing,
        RereadRoute::Own => match ctx.svc.snapshot_flights.lookup(object_id) {
            Some(found) => {
                // An attached catch-up holds no slot (Spec SDD SR-7.5).
                *slot = None;
                FlightSizing::Known(found.size)
            }
            None => match read_catch_up_snapshot(ctx, object_id).await {
                Ok(Ok(SnapshotRead::Snapshot(snapshot))) => {
                    if snapshot.cut < run.first_cut {
                        return Err(catch_up_regressed(
                            ctx,
                            object_id,
                            run.first_cut,
                            snapshot.cut,
                            run.regime,
                            "atomic",
                        ));
                    }
                    *consumer = register_stream(ctx, reserved, &stream_cursor_at(snapshot.cut))?;
                    return match Box::pin(build_catch_up_once(
                        ctx,
                        &snapshot,
                        build,
                        ExtValueReads::PROBE,
                    ))
                    .await?
                    {
                        Built::Ready(built) => Ok(Reread::Done(built)),
                        Built::Missing(_) => {
                            Ok(Reread::StillMissing(BuildSnapshot::Atomic(snapshot)))
                        }
                    };
                }
                Ok(Ok(SnapshotRead::TooLarge(too_large))) => {
                    // The layer grew past the atomic-read limits since the
                    // first read. `read_object_snapshot` already counted the
                    // answer; an attached catch-up holds no slot.
                    *slot = None;
                    FlightSizing::from_too_large(&too_large)
                }
                Ok(Err(ending)) => return Err(ending),
                Err(ServiceError::NotFound(_)) => {
                    return Ok(Reread::Done(CatchUpBuilt::RereadNotFound));
                }
                Err(other) => return Err(other.into()),
            },
        },
    };

    // Above the atomic-read limits: the flight. The hold was acknowledged
    // before this attach, and the catch-up holds no slot and no place in an
    // earlier flight of the object.
    tracing::debug!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        pin = run.first_cut,
        read = run.read,
        regime = run.regime,
        "catch-up: re-reading the layer through its shared fenced-read flight after an \
         external-value miss"
    );
    let read = match Box::pin(read_through_flight(ctx, object_id, sizing)).await {
        Ok(Ok(read)) => read,
        Ok(Err(ending)) => return Err(ending),
        Err(ServiceError::NotFound(_)) => return Ok(Reread::Done(CatchUpBuilt::RereadNotFound)),
        Err(other) => return Err(other.into()),
    };
    let cut = read.snapshot().cut;
    if cut < run.first_cut {
        // Dropping `read` on the way out leaves the flight.
        return Err(catch_up_regressed(
            ctx,
            object_id,
            run.first_cut,
            cut,
            run.regime,
            "shared_fenced",
        ));
    }
    *consumer = register_stream(ctx, reserved, &stream_cursor_at(cut))?;
    let turn = Box::pin(take_build_turn(
        ctx,
        CatchUpSnapshot::Shared(read),
        consumer,
    ))
    .await?;
    match Box::pin(build_catch_up_once(ctx, &turn, build, ExtValueReads::PROBE)).await? {
        Built::Ready(built) => Ok(Reread::Done(built)),
        Built::Missing(_) => Ok(Reread::StillMissing(turn)),
    }
}

/// Send the full-content snapshot [`build_catch_up`] built for
/// Regime 1 / overwrite-on-reconnect. Called with no catch-up slot or build
/// turn held.
///
/// Returns the delivered byte count and `catchup_mode` for the Plan b.msb
/// catch-up metrics, plus the `effective_version` this catch-up actually
/// serves — see [`CatchUpContent`].
async fn send_full_state(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    diff_response: &DiffResponse,
    regime_mode: &'static str,
) -> Result<CatchUpContent, Status> {
    // Enforce the brand-new-establishment invariant (SR-2.4 / SR-1.2): the
    // from-empty full-state message #1 must ALWAYS be a valid non-empty OMPF
    // create. The `FullContent` builder guarantees this even for an empty
    // net-live scene (a zero-command OODF delta still serializes to a non-empty
    // OMPF container), so an empty payload here is an invariant violation, not a
    // legitimate outcome. Surfacing it as `INTERNAL` — rather than silently
    // skipping the send and letting `run_catch_up` emit only its `total_size==0`
    // terminator — keeps the establishment stream from collapsing to the
    // terminator-only sequence the client treats as fatal empty content. The
    // terminator remains the ONE legitimate zero-size signal and is still
    // emitted exactly once by `run_catch_up`, never here.
    if diff_response.ompf_bytes.is_empty() {
        return Err(ServiceError::Internal(format!(
            "brand-new full-state establishment produced an empty OMPF payload for object_id={object_id} (mode={regime_mode}); \
             message #1 must be a valid non-empty from-empty create"
        ))
        .into());
    }
    // Map the tracing "off"/"snapshot" mode to the metric taxonomy: only the
    // corrected faithful full-state path is `"snapshot"` (Plan b.msb).
    let catchup_mode = if regime_mode == "snapshot" {
        "snapshot"
    } else {
        "legacy"
    };

    // Label the content with the cut the snapshot actually represents and
    // return that same value for the ack — see the ack-≤-content invariant
    // on `CatchUpContent` / `deliver_catch_up_content`. Deliberately AFTER the
    // empty-payload guard above: an establishment that fails as `INTERNAL` never
    // produces a `CatchUpOutcome`, so `observe_subscribe_regime` does not fire
    // for it, and recording the stable cut only past the guard means such a
    // failure records no `verified` cut either — a build that served nothing
    // never counts as a served cut. (The two counters are not 1:1 in general;
    // see `live_subscribe_stable_cut_total`'s doc.)
    let content =
        deliver_catch_up_content(ctx, object_id, diff_response, "brand_new", catchup_mode).await?;

    // Content-only: this helper delivers the full-state snapshot but does NOT
    // emit the `total_size == 0` caught-up terminator. The terminator is sent
    // exactly once for the entire catch-up phase by `run_catch_up` (covering
    // every regime uniformly via `CatchUpOutcome.tracked_version`), and once
    // for the steady-state-phase rebootstrap by `run_steady_state`. Emitting it
    // here too would double-emit on the brand-new / overwrite-on-reconnect path
    // (b.dy5).
    tracing::info!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        case = diff_response.case.as_label(),
        sections_count = diff_response.sections_count,
        ompf_size = diff_response.ompf_bytes.len(),
        regime = "brand_new",
        mode = regime_mode,
        "full-state catch-up sent"
    );
    Ok(content)
}

async fn send_ack_header(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    sequence_num: u64,
) -> Result<(), Status> {
    let ack = SubscribeReadLayerResponse {
        response_type: Some(subscribe_read_layer_response::ResponseType::Header(
            SubscribeReadLayerHeader {
                object_id,
                sequence_num,
                total_size: 0,
                server_timestamp_microseconds_since_epoch: 0,
                client_timestamp_microseconds_since_epoch: 0,
            },
        )),
    };
    send_or_disconnect(ctx, ack).await
}

/// Hand one message to the gRPC response channel, racing the send against every
/// reason it might never complete (b.f4f).
///
/// Pre-b.f4f this was a bare `tx.send(...).await` with no timeout and no
/// cancellation arm, which meant: a client that stopped reading parked this task
/// on a full 32-slot channel indefinitely, holding its catch-up buffers and its
/// stream registration; and a `DeleteSession` or a graceful shutdown was not
/// observed until the send eventually completed. Both are now exits.
///
/// The race itself is [`send_racing_endings`], which takes no
/// [`SubscriberCtx`] so it can be tested on a bare channel. This wrapper
/// resolves the deadline, records the exit reason and maps each outcome to the
/// status the client sees:
///
/// * [`SendOutcome::ShuttingDown`] — graceful shutdown. `reason="shutdown"`,
///   [`server_shutting_down`]: `UNAVAILABLE`, so the client re-subscribes,
///   wherever the ingress routes it. Reaching it means a message was in flight, so the client
///   holds a truncated OMPF sequence and, mid-catch-up, never received its
///   caught-up ack; the retry re-establishes it.
/// * [`SendOutcome::SessionCancelled`] — the session token fired, ending by
///   its cause as in every other phase
///   ([`SubscriberCtx::end_on_session_cancel`]). For an explicit
///   `DeleteSession`: `reason="session_cancel"`, [`session_deleted`]:
///   `NOT_FOUND "session was deleted"` (a deletion read first as the layer's
///   `deleted` sentinel ends `NOT_FOUND` too, with its own message and
///   `reason="error"`), not retried — there is nothing to reconnect to. For a
///   session found missing from the live store: `reason="session_absent"`,
///   [`session_absent_status`], retried.
/// * [`SendOutcome::Disconnected`] — the client went away. `Code::Cancelled`;
///   the receiver is gone, so nothing is delivered either way, and this is the
///   one exit the task's tail treats as a clean end.
/// * [`SendOutcome::SlowReader`] — [`OUTER_SEND_DEADLINE`] elapsed. Routes
///   through the b.c7a terminal-error contract as `Code::Unavailable`, never a
///   clean `Ok(())`, so the client's standard retry policy re-subscribes.
///
/// **The deadline arms only around a send that cannot complete.** It is not a
/// timer on waiting for new stream entries — that await lives in
/// `consumer.recv()`, not here, and a quiet layer is a healthy one. It is not a
/// whole-phase budget on catch-up either. Every chunk loop in this file calls
/// this, or [`send_or_disconnect_as`], once per chunk, so each chunk gets its
/// own deadline and a large catch-up
/// to a client that is reading takes as long as it takes. There are three, and
/// naming them is the point — the claim is per-chunk, not per-loop:
///
/// * `deliver_catch_up_content` — the chunk loop for the full-state,
///   in-window-snapshot and out-of-window regimes.
/// * `send_per_commit_payload` — the chunk loop for the legacy in-window replay
///   (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`, the documented rollback knob)
///   and for the steady-state empty delta of a commit with no relay bytes to
///   forward.
/// * `run_steady_state`'s verbatim relay — the b.gze path that forwards the
///   writer's own OMPF.
///
/// Reaching the slow-reader outcome therefore means one specific thing: the
/// channel's 32 delivery slots are full and tonic has not drained them. (The
/// channel is built with 33; the 33rd is reserved for the terminal trailer
/// before any data is sent and is not available here — see
/// `send_terminal_status`.)
///
/// **Stall accounting.** A send that finds those 32 slots already full is a
/// stall ([`SendStall`]): it is counted on
/// `live_edit_subscriber_send_stalls_total{phase}` when it begins, and how long
/// it waited is recorded on `live_edit_subscriber_send_wait_seconds{phase}`
/// whatever ends it — sent, the deadline, shutdown, session cancel,
/// disconnect, or the task being dropped mid-send. A send that finds room
/// records nothing. This wrapper sends as [`SendKind::CatchUp`], labelled
/// `phase="catchup"`: catch-up content and every ack. The other chunk loops
/// call [`send_or_disconnect_as`] with their kind: the steady-state relay and
/// empty delta as [`SendKind::Live`], labelled `phase="live"`, stamping their
/// last message for the `transport` delivery lag; the legacy in-window replay
/// as [`SendKind::CatchUp`].
async fn send_or_disconnect(
    ctx: &SubscriberCtx<'_>,
    msg: SubscribeReadLayerResponse,
) -> Result<(), Status> {
    send_or_disconnect_as(ctx, msg, SendKind::CatchUp).await
}

/// [`send_or_disconnect`] for a send of the given `kind`: its stall `phase`,
/// and the commit stamp a live delivery's last message carries
/// ([`Outbound::commit_end`]).
async fn send_or_disconnect_as(
    ctx: &SubscriberCtx<'_>,
    msg: SubscribeReadLayerResponse,
    kind: SendKind,
) -> Result<(), Status> {
    let deadline = outer_send_deadline(ctx.svc);
    let outbound = Outbound {
        item: Ok(msg),
        commit_end: kind.commit_end(),
    };
    // Held across the send, so its wait is recorded however the send ends,
    // the task being dropped included.
    let stall = SendStall::begin(ctx.tx, &ctx.svc.metrics, kind.phase());
    let outcome = send_racing_endings(
        ctx.tx,
        outbound,
        ctx.shutdown,
        ctx.session_token(),
        deadline,
    )
    .await;
    drop(stall);

    match outcome {
        SendOutcome::Sent => Ok(()),
        SendOutcome::Disconnected => {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            Err(Status::cancelled("client disconnected"))
        }
        SendOutcome::ShuttingDown => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            // `Unavailable`, not `Cancelled`: the Kit client retries the former
            // and not the latter (its retry policy covers DEADLINE_EXCEEDED /
            // UNAVAILABLE / RESOURCE_EXHAUSTED / INTERNAL; the inspected
            // revision and line anchors: `docs-internal/cross-repo-debugging.md`
            // § Kit client retry and reconnect behaviour), and
            // a shutdown caught mid-delivery has left it with a partial stream
            // it must re-establish, wherever the ingress routes the retry.
            Err(server_shutting_down())
        }
        SendOutcome::SessionCancelled => Err(ctx.end_on_session_cancel()),
        SendOutcome::SlowReader => {
            ctx.exit.set(StreamExitCell::SLOW_READER_DEADLINE);
            tracing::warn!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                client_id = %ctx.client_id,
                deadline_ms = deadline.as_millis() as u64,
                outer_capacity = 32,
                "subscriber stopped reading its gRPC stream for longer than the outer send \
                 deadline; tearing the subscription down so its buffers, task and \
                 stream registration are released"
            );
            Err(ServiceError::Unavailable(format!(
                "subscriber did not read its stream within {}ms; tearing down",
                deadline.as_millis()
            ))
            .into())
        }
    }
}

/// Which arm of [`send_racing_endings`]'s `select!` won.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendOutcome {
    /// The message is enqueued.
    Sent,
    /// The receiver is gone — before the send, or it refused the send.
    Disconnected,
    /// The subscription's shutdown token fired.
    ShuttingDown,
    /// The session token fired; its cause decides the ending.
    SessionCancelled,
    /// The deadline elapsed with the channel still full.
    SlowReader,
}

/// Race one send on `tx` against every reason it might never complete, and
/// report which won. Takes nothing but the channel, the message, the two
/// tokens and the deadline, so it is testable on a bare `mpsc` channel; its one
/// production caller is [`send_or_disconnect_as`], which maps the outcome to
/// the client's status.
///
/// Five arms, `biased` so the terminal conditions are checked before the send
/// can win a race it is already losing — without `biased`, a send into a
/// channel with room and a token that has already fired are both ready, and
/// tokio would pick between them at random:
///
/// 1. `shutdown` — [`SendOutcome::ShuttingDown`]. First, so a subscription that
///    sees a shutdown and a deletion together ends as a shutdown, which its
///    client retries.
/// 2. `session_cancel` — [`SendOutcome::SessionCancelled`]. An absent token's
///    arm never resolves.
/// 3. `tx.closed()` — [`SendOutcome::Disconnected`].
/// 4. the send itself — [`SendOutcome::Sent`], or
///    [`SendOutcome::Disconnected`] if the receiver dropped under it.
/// 5. `deadline` — [`SendOutcome::SlowReader`]. It arms only around this one
///    send; see [`send_or_disconnect`].
async fn send_racing_endings<T>(
    tx: &mpsc::Sender<T>,
    msg: T,
    shutdown: &CancellationToken,
    session_cancel: Option<&CancellationToken>,
    deadline: std::time::Duration,
) -> SendOutcome {
    let send = tx.send(msg);
    tokio::pin!(send);
    tokio::select! {
        biased;
        () = shutdown.cancelled() => SendOutcome::ShuttingDown,
        () = cancelled_or_never(session_cancel) => SendOutcome::SessionCancelled,
        () = tx.closed() => SendOutcome::Disconnected,
        result = &mut send => match result {
            Ok(()) => SendOutcome::Sent,
            Err(_) => SendOutcome::Disconnected,
        },
        () = tokio::time::sleep(deadline) => SendOutcome::SlowReader,
    }
}

/// Test-only two-phase seam inside [`deliver_catch_up_content`]'s chunk loop
/// (b.f4f) — the loop that serves the full-state, in-window-snapshot and
/// out-of-window regimes.
///
/// Deliberately not installed on the other two chunk loops on this path
/// (`send_per_commit_payload`, which owns the legacy in-window replay's sends,
/// and `run_steady_state`'s verbatim relay): the window this seam exists to
/// open is "content delivered, caught-up ack not yet sent", which only the
/// catch-up phase has.
///
/// A subscriber parked here holds no concurrent-catch-up slot: every catch-up
/// releases its slot before its first send.
///
/// Installed via `LiveEditService::catch_up_chunk_seam_for_test`. The subscriber
/// parks after its FIRST content chunk has been handed to
/// [`send_or_disconnect`]; the test then knows content is on the wire and the
/// caught-up ack is not, fires its cancellation, and releases the park. The next
/// `send_or_disconnect` — the following chunk, or `send_ack_header` when the
/// payload was a single chunk — observes the cancellation and the RPC ends
/// `UNAVAILABLE` (a shutdown) or `NOT_FOUND` (a deletion) rather than with a
/// truncated-but-`OK` stream.
///
/// **Why a rendezvous rather than a chunk-size override.** Shrinking
/// `OMPF_CHUNK_SIZE` so a small layer produces >32 chunks would have to thread a
/// size parameter through `chunk_ompf_for_stream` — a `pub` helper with three
/// production call sites, only one of which is the catch-up path — and it would
/// still leave *when* the cancellation lands to wall-clock luck. This seam is one
/// cfg-gated field, changes no shared signature, and makes the interleaving
/// deterministic. It also works for a single-chunk payload, where the ack send is
/// the one that observes the cancel; a size override would need a payload big
/// enough to chunk at all.
///
/// Two `Barrier`s rather than one, for the same reason `RunCatchUpInnerSeam` uses
/// two: with a single rendezvous the test cannot tell "the subscriber has parked"
/// from "the subscriber is about to park", and firing the cancellation into that
/// gap is a race.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct CatchUpChunkSeam {
    /// Rendezvous 1: the subscriber announces the first chunk is away.
    parked: tokio::sync::Barrier,
    /// Rendezvous 2: the test releases the parked subscriber.
    released: tokio::sync::Barrier,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CatchUpChunkSeam {
    /// Build a seam expecting exactly one subscriber and one test task at each
    /// of the two rendezvous points.
    pub fn new() -> Self {
        Self {
            parked: tokio::sync::Barrier::new(2),
            released: tokio::sync::Barrier::new(2),
        }
    }

    /// Subscriber side (called from inside `deliver_catch_up_content`): announce
    /// the first chunk has been sent (rendezvous 1), then block until the test
    /// releases it (rendezvous 2).
    ///
    /// **Each underlying `Barrier` is 2-party, so a third arrival hangs** — and
    /// the `chunks_sent == 1` guard at the call site bounds arrivals *per
    /// invocation of `deliver_catch_up_content`*, not per service. Two shapes
    /// reach it a second time, and a test must account for both:
    ///
    /// * **A re-bootstrap.** A subscription that re-bootstraps mid-catch-up runs
    ///   the chunk loop again on the new generation.
    /// * **A second concurrent subscription.** The seam is installed on the
    ///   `LiveEditService`, which every subscriber on that service shares, so any
    ///   other subscription whose catch-up delivers at least one content chunk
    ///   parks here too — even one the test is not thinking about, e.g. a
    ///   fixture's warm-up subscriber. Same caveat the stream multiplexer's
    ///   shared per-site fault queues (`StreamMux::push_faults_for_test`) carry.
    ///
    /// Either way: install a fresh seam per subscription, or keep exactly one
    /// content-delivering subscription alive for the seam's lifetime.
    pub async fn arrive_and_wait(&self) {
        let _ = self.parked.wait().await;
        let _ = self.released.wait().await;
    }

    /// Test side: block until the subscriber has parked (rendezvous 1). After
    /// this returns, exactly one content chunk has been handed to the response
    /// channel and no caught-up ack has been sent; the test may now cancel the
    /// session with no race against the send. Must be paired with a later
    /// [`Self::release`].
    pub async fn wait_until_parked(&self) {
        let _ = self.parked.wait().await;
    }

    /// Test side: release the parked subscriber (rendezvous 2). Call only after
    /// [`Self::wait_until_parked`] has returned and the cancellation is in place.
    pub async fn release(&self) {
        let _ = self.released.wait().await;
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Default for CatchUpChunkSeam {
    fn default() -> Self {
        Self::new()
    }
}

/// The effective [`OUTER_SEND_DEADLINE`], honouring the test-only override.
///
/// The production constant is 5 s, which would cost every slow-reader test that
/// much wall clock per run — the same reason `StreamMuxKnobs` carries a
/// test-overridable `send_watchdog`. Release builds do not compile the field or
/// this branch.
fn outer_send_deadline(svc: &LiveEditService) -> std::time::Duration {
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(override_deadline) = svc.outer_send_deadline_for_test {
        return override_deadline;
    }
    let _ = svc;
    OUTER_SEND_DEADLINE
}

/// Build the steady-state consumer's start cursor from a layer-header sequence
/// number (b.9uq round 4).
///
/// Per-object stream entry IDs for commits are **deterministically**
/// `{sequence_num}-0` — `storage::layer` XADDs the seed as the literal `"1-0"`,
/// the WATCH/MULTI/EXEC path as `format!("{new_seq}-0")`, and
/// `update_layer_fast.lua` as `new_seq_str .. "-0"` — so a header sequence is
/// directly usable as an XREAD resume point without asking Valkey what the tail
/// is. `XREAD` starts *strictly after* the given ID, so `{seq}-0` means "every
/// commit the subscriber has not been served".
///
/// This replaces the `StreamCursor::Latest` (`"$"`) opens every catch-up regime
/// used through b.9uq round 3, and the `XREVRANGE` tail-fetch the consumer ran
/// to pin them. `$` was silently lossy on any XREAD re-issue (Valkey re-resolves
/// it per execution); the consumer-side pin fixed that but resolved the cursor
/// one RTT *after* `open_consumer` returned — i.e. after the caller had already
/// started, and sometimes finished, its catch-up state read — leaving a
/// commit-loss window that stretched from ~1 RTT healthy to minutes whenever the
/// probe was force-reconnecting a wedged client. Deriving the cursor from the
/// header the caller already holds moves the pin *before* the state read, which
/// closes the window instead of narrowing it.
///
/// Non-commit entries (the overwrite `deleted` sentinel, XADDed with an auto
/// wall-clock ID) sort far above any `{seq}-0`, so a header-pinned consumer
/// still receives them.
fn stream_cursor_at(sequence_num: u64) -> StreamCursor {
    StreamCursor::From(format!("{sequence_num}-0"))
}

/// End a catch-up whose own stream registration has already ended — for
/// example its backlog reached a hard cap while the catch-up built, or the
/// multiplexer shut down — with that ending, before the catch-up sends its
/// first message.
///
/// A catch-up's stream is registered at its proven read's cut (its own atomic
/// read's, or its flight's shared fenced read's), after every capacity wait
/// ([`catch_up_proven_read`]), so this runs where a registered stream has had
/// time to end: after the slot (or build turn) is released and before the
/// first message, and, attached to a flight, all along its wait for a build
/// turn ([`take_build_turn`]), so a registration that reached a hard backlog
/// cap ends the catch-up there rather than after a build it can no longer
/// send. Before the registration there is only a [`ReservedStream`], which
/// queues nothing and cannot end.
///
/// Without this check the ending would surface only in `run_steady_state`,
/// after the catch-up's content and caught-up ack, which reset the clients'
/// retry counts: at a saturated replica that is a reconnect loop after
/// content. It records the consumer's exit reason and returns its status
/// (`UNAVAILABLE`) exactly as `run_steady_state` does on `recv() == None`; the
/// caller's return drops the consumer (unregistering it and freeing its queued
/// entries) and any catch-up slot.
///
/// It reads only the recorded terminal exit. It never calls `recv` or
/// `try_recv`, so it drains nothing and is not the consumer's first poll: the
/// consumer is not marked draining, and only the hard backlog caps apply.
fn end_catch_up_if_stream_ended(
    ctx: &SubscriberCtx<'_>,
    consumer: &LayerStreamConsumer,
) -> Result<(), Status> {
    if let Some(exit) = consumer.take_terminal_exit() {
        ctx.exit.set_reason(exit.reason);
        return Err(exit.error.into());
    }
    Ok(())
}

/// The status a subscription ends with when its capacity budget ran out
/// waiting for a stream reader: in cluster mode every reader stayed bound to
/// another hash slot (`reason="reader_capacity"`). Retryable, with the retry
/// hint [`READER_CAPACITY`](refusal_reason::READER_CAPACITY).
fn reader_capacity_refused() -> Status {
    RefusalDetail::new(refusal_reason::READER_CAPACITY, Some(CAPACITY_RETRY_DELAY)).unavailable(
        "this replica is serving as many distinct layers as it has stream readers \
         (LIVE_EDIT_SUBSCRIBER_POOL_SIZE); retry",
    )
}

/// The status a subscription ends with when its capacity budget ran out
/// waiting for a catch-up slot (`reason="catchup_capacity"`). Retryable, with
/// the retry hint [`CATCHUP_CAPACITY`](refusal_reason::CATCHUP_CAPACITY),
/// metadata `wait` = `slot`.
fn catchup_capacity_refused() -> Status {
    RefusalDetail::new(refusal_reason::CATCHUP_CAPACITY, Some(CAPACITY_RETRY_DELAY))
        .with_metadata(refusal_metadata::WAIT, refusal_metadata::SLOT)
        .unavailable(
            "this replica is at its concurrent catch-up limit \
             (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS); retry",
        )
}

/// The status a catch-up attached to a shared fenced-read flight ends with
/// when its capacity budget ran out while the flight waited for capacity
/// ([`FlightWaitError::CapacityExhausted`]; `reason="catchup_capacity"`). It
/// names the flight's three waits, since the catch-up was charged for
/// whichever of them the flight was in while it was attached. Retryable, with
/// the retry hint [`CATCHUP_CAPACITY`](refusal_reason::CATCHUP_CAPACITY) and no
/// `wait` metadata: the flight does not record which of its three waits the
/// budget ran out in, and its memory wait has no `wait` value.
fn flight_capacity_refused() -> Status {
    RefusalDetail::new(refusal_reason::CATCHUP_CAPACITY, Some(CAPACITY_RETRY_DELAY)).unavailable(
        "this replica's catch-up capacity for layers above the atomic-read limits is in use: \
         the shared snapshot read this catch-up joined was still waiting for a catch-up slot \
         (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS), a fenced read permit \
         (LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY) or catch-up memory (its memory budget, \
         sized from the pod's memory limit) when its capacity budget ran out; retry",
    )
}

/// The status a catch-up attached to a shared fenced-read flight ends with
/// when the flight's work — fence acquires, `RETRY_AFTER` sleeps and the
/// fenced read — passed `reader_wait_max(L)` before its read resolved
/// ([`FlightWaitError::ReaderWaitCap`]; `reason="fenced_read_wait_cap"`, Spec
/// SDD SR-5.5). Before any content. Retryable.
fn fenced_read_wait_refused() -> Status {
    Status::unavailable(
        "this layer is above the atomic-read limits, and the shared fenced snapshot read this \
         catch-up joined did not complete within the time a catch-up waits for one \
         (reader_wait_max); retry",
    )
}

/// Takes a `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS` slot for one
/// catch-up's backing-store reads and build (and, in rollback mode
/// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) on a same-object reconnect, its
/// cluster stream-reader wait). The caller drops the returned
/// permit before its first send, before any recovery attempt and before any
/// re-bootstrap, and never calls this while it holds another.
///
/// The wait ends at the subscription's capacity-budget deadline and is charged
/// to the budget. It races the endings first, in the order every other wait
/// observes them: shutdown, session cancellation, client disconnect. A budget
/// that runs out ends the subscription `UNAVAILABLE` with
/// `reason="catchup_capacity"` and records the wait as `timed_out`. Every
/// catch-up takes its slot before its first content message, in both catch-up
/// modes, so this refusal comes before any content — except on a steady-state
/// overwrite re-bootstrap, where it comes after the subscription's caught-up
/// ack.
async fn acquire_catchup_slot(ctx: &SubscriberCtx<'_>) -> Result<CatchUpPermit, Status> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    let budget_before = ctx.budget.remaining();
    let started = tokio::time::Instant::now();
    let deadline = ctx.budget.deadline();
    let acquired = tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            return Err(server_shutting_down());
        }
        () = cancelled_or_never(ctx.session_token()) => {
            return Err(ctx.end_on_session_cancel());
        }
        () = ctx.tx.closed() => {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            return Err(Status::cancelled("client disconnected"));
        }
        acquired = ctx.svc.valkey.catchup_gate.acquire(deadline) => acquired,
    };
    let waited = started.elapsed();
    ctx.budget.charge(waited);
    #[cfg(any(test, feature = "test-fault-injection"))]
    record_budget_charge(
        ctx,
        BudgetWait::CatchUpSlot,
        budget_before,
        started,
        Some(deadline),
        waited,
    );
    match acquired {
        Ok(permit) => Ok(permit),
        // `ValkeyClient::quit` closed the gate: the replica is shutting down.
        Err(CatchUpRefusal::Closed) => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            Err(server_shutting_down())
        }
        Err(CatchUpRefusal::TimedOut) => {
            ctx.exit.set(StreamExitCell::CATCHUP_CAPACITY);
            Err(catchup_capacity_refused())
        }
    }
}

/// The status a subscription ends with when its capacity budget ran out
/// waiting for a snapshot read permit (`reason="catchup_capacity"`).
/// Retryable, with the retry hint
/// [`CATCHUP_CAPACITY`](refusal_reason::CATCHUP_CAPACITY), metadata `wait` =
/// `read_permit`.
fn snapshot_read_capacity_refused() -> Status {
    RefusalDetail::new(refusal_reason::CATCHUP_CAPACITY, Some(CAPACITY_RETRY_DELAY))
        .with_metadata(refusal_metadata::WAIT, refusal_metadata::READ_PERMIT)
        .unavailable(
            "this replica is at its concurrent catch-up snapshot-read limit \
             (LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY); retry",
        )
}

/// The status a catch-up ends with when a read's cut shows the layer's
/// sequence regressed (Spec SDD SR-3.2), before any content: a regime's proven
/// read — its own atomic read, or its flight's shared fenced read, checked
/// per attached catch-up — whose cut is below the header sequence it read
/// first ([`catch_up_regressed`]), or a rollback-mode in-window replay's
/// per-commit read whose cut is below the commit it rebuilds, which
/// [`compute_per_commit_diff_at_cut`] refuses and
/// [`per_commit_regressed`] maps to this status, or a re-read after an
/// external-value miss whose cut is below the first read's
/// ([`reread_and_rebuild`]). The layer exists, so
/// this is no `NOT_FOUND`, and a retried subscribe re-reads the current header
/// and succeeds. Retryable.
fn sequence_regressed() -> Status {
    Status::unavailable(
        "the layer's sequence moved backwards while this subscription caught up (the layer \
         was restored); retry",
    )
}

/// Takes a `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` read permit for one
/// catch-up atomic snapshot read (Spec SDD SR-7.1), observed on
/// `live_edit_catchup_snapshot_read_wait_seconds` under `consumer`: a regime's
/// proven read (`catchup`, [`read_catch_up_snapshot`]) or one commit of a
/// rollback-mode in-window replay (`catchup_legacy_per_commit`,
/// [`read_replay_commit_snapshot`]). The caller holds its catch-up slot across
/// this call and the read, and drops the permit as soon as the read returns; it
/// never waits for a slot while holding a permit.
///
/// The only place an atomic-class read permit is taken: the steady state
/// never reads layer state, so every atomic permit wait is a catch-up's, with
/// a deadline. (Fenced-class permits are taken only by a shared fenced-read
/// flight, `storage::snapshot_flights`, with no deadline: each catch-up
/// attached to it is charged its share of the wait instead.)
///
/// A capacity wait like [`acquire_catchup_slot`]'s, and handled the same way
/// (Spec SDD SR-7.3): it ends at the subscription's capacity-budget deadline
/// and is charged to the budget; it races the endings first, in the order
/// every other wait observes them (shutdown, session cancellation, client
/// disconnect); a closed gate ends the subscription `shutdown`; and a budget
/// that runs out ends it `UNAVAILABLE` with `reason="catchup_capacity"`. It
/// runs inside the catch-up slot, so it comes before any content — except on
/// a steady-state overwrite re-bootstrap, where it comes after the
/// subscription's caught-up ack, as the slot wait does.
async fn acquire_snapshot_read_permit(
    ctx: &SubscriberCtx<'_>,
    consumer: SnapshotReadConsumer,
) -> Result<CatchUpReadPermit, Status> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    let budget_before = ctx.budget.remaining();
    let started = tokio::time::Instant::now();
    let deadline = ctx.budget.deadline();
    let acquired = tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            return Err(server_shutting_down());
        }
        () = cancelled_or_never(ctx.session_token()) => {
            return Err(ctx.end_on_session_cancel());
        }
        () = ctx.tx.closed() => {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            return Err(Status::cancelled("client disconnected"));
        }
        acquired = ctx
            .svc
            .valkey
            .catchup_read_gate
            .acquire(deadline, consumer) => acquired,
    };
    let waited = started.elapsed();
    ctx.budget.charge(waited);
    #[cfg(any(test, feature = "test-fault-injection"))]
    record_budget_charge(
        ctx,
        BudgetWait::ReadPermit,
        budget_before,
        started,
        Some(deadline),
        waited,
    );
    match acquired {
        Ok(permit) => Ok(permit),
        // `ValkeyClient::quit` closed the gate: the replica is shutting down.
        Err(CatchUpRefusal::Closed) => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            Err(server_shutting_down())
        }
        Err(CatchUpRefusal::TimedOut) => {
            ctx.exit.set(StreamExitCell::CATCHUP_CAPACITY);
            Err(snapshot_read_capacity_refused())
        }
    }
}

/// The one read call through which every catch-up regime reaches its atomic
/// snapshot read (Spec SDD SR-3.1; Epic 4's shared reads wrap exactly this):
/// take a read permit ([`acquire_snapshot_read_permit`]), run
/// [`read_object_snapshot`] for the `catchup` consumer, and release the
/// permit as the read returns, whatever it returned — before a `TOO_LARGE`
/// answer's attach to the object's shared fenced-read flight, the regression
/// check, the registration, the build or a `NotFound` recovery.
///
/// **Two error layers, one per stage, so neither is mistaken for the other.**
/// The outer `Err(ServiceError)` is the read's own. It stays a `ServiceError`
/// (not `Status`) so [`read_state_with_recovery`] can match on the structured
/// `ServiceError::NotFound` variant to detect the OLD-object-gone race (b.jvx)
/// and route it into `handle_deleted_and_rebootstrap` per SDD §12.3 / §12.4.
/// The inner `Err(Status)` is the permit wait's ending (a capacity refusal,
/// shutdown, session cancellation or client disconnect), returned as-is by
/// the caller. Folding it into the outer layer would be lossy both ways — a
/// deleted session's `NOT_FOUND` would read as the OLD-object-gone race, and a
/// recoding by status code would turn one ending into another.
///
/// Called only while the caller holds its catch-up slot. Both awaits are
/// boxed (b.9qr), so the snapshot read's state machine is not carried inline
/// by every regime.
async fn read_catch_up_snapshot(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
) -> Result<Result<SnapshotRead, Status>, ServiceError> {
    let permit = match Box::pin(acquire_snapshot_read_permit(
        ctx,
        SnapshotReadConsumer::Catchup,
    ))
    .await
    {
        Ok(permit) => permit,
        Err(ending) => return Ok(Err(ending)),
    };
    let oid = object_id.to_string();
    let read = Box::pin(read_object_snapshot(
        &ctx.svc.valkey,
        &ctx.svc.key_schema,
        &oid,
        SnapshotReadConsumer::Catchup,
    ))
    .await;
    drop(permit);
    read.map(Ok)
}

/// One commit's atomic read in a rollback-mode in-window replay
/// ([`catch_up_in_window`]; Spec SDD SR-3.4, SR-7.1): take a read permit inside
/// the replay's catch-up slot ([`acquire_snapshot_read_permit`], charged to the
/// capacity budget like every catch-up wait), run [`read_object_snapshot`] for
/// the `catchup_legacy_per_commit` consumer, and release the permit as the read
/// returns. One permit per read, never one across the replay.
///
/// Error layers as on [`read_catch_up_snapshot`]: the outer `Err(ServiceError)`
/// is the read's own (the replay routes its `NotFound`), the inner
/// `Err(Status)` the permit wait's ending, returned as-is. Deliberately not
/// [`read_catch_up_snapshot`]: per-commit reads are never shared between
/// catch-ups (Spec SDD SR-3.4), so they stay off the regimes' one read call.
async fn read_replay_commit_snapshot(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
) -> Result<Result<SnapshotRead, Status>, ServiceError> {
    let permit = match Box::pin(acquire_snapshot_read_permit(
        ctx,
        SnapshotReadConsumer::CatchupLegacyPerCommit,
    ))
    .await
    {
        Ok(permit) => permit,
        Err(ending) => return Ok(Err(ending)),
    };
    let oid = object_id.to_string();
    let read = Box::pin(read_object_snapshot(
        &ctx.svc.valkey,
        &ctx.svc.key_schema,
        &oid,
        SnapshotReadConsumer::CatchupLegacyPerCommit,
    ))
    .await;
    drop(permit);
    read.map(Ok)
}

/// What a catch-up's proven read ([`catch_up_proven_read`]) leaves it to build
/// from, with its stream registered.
enum CatchUpRead {
    /// The layer's state proven at a cut, with the stream registered at that
    /// cut: the catch-up's own atomic read's at or below the atomic-read
    /// limits, its flight's shared fenced read's above them.
    Proven {
        snapshot: CatchUpSnapshot,
        consumer: LayerStreamConsumer,
    },
    /// The layer's state was gone and the subscriber was re-bootstrapped (see
    /// [`StateRead::Rebootstrapped`]); the caller returns this as its own.
    Rebootstrapped(CatchUpOutcome),
}

/// The proven snapshot a catch-up builds from, and how it holds it.
enum CatchUpSnapshot {
    /// Its own atomic read's, built from under the catch-up slot it holds.
    Atomic(Box<AtomicSnapshot>),
    /// Its flight's shared fenced read's, with the catch-up still attached to
    /// the flight; it builds in a build turn the flight serves
    /// ([`take_build_turn`]).
    Shared(FlightRead),
}

/// The snapshot a catch-up builds from once it may build ([`take_build_turn`]):
/// its own, or its flight's in a build turn. Derefs to the snapshot.
///
/// Drop it as the build returns, before the stream-ending check and the first
/// send: on the fenced path that gives back, together, the snapshot handle
/// (the flight's memory grant is repaid when the last one drops), the build
/// turn (handed to the next catch-up, or an extra build's slot and part
/// returned) and the catch-up's place in its flight ([`BuildTurn`]).
enum BuildSnapshot {
    Atomic(Box<AtomicSnapshot>),
    Turn(BuildTurn),
}

impl std::ops::Deref for BuildSnapshot {
    type Target = AtomicSnapshot;

    fn deref(&self) -> &AtomicSnapshot {
        match self {
            BuildSnapshot::Atomic(snapshot) => snapshot,
            BuildSnapshot::Turn(turn) => turn,
        }
    }
}

/// A catch-up's proven read and stream registration, shared by every regime
/// that serves state (brand-new in both modes, in-window snapshot,
/// out-of-window in both modes; Spec SDD SR-3.1–SR-3.3, SR-3.6, SR-7.5).
///
/// Entered holding the reader `reserved` for the object's stream (`Some`) and,
/// in `slot`, the catch-up slot — `None` only for a brand-new catch-up that
/// found its object's flight registered and so took none. `first_read` is P,
/// the header sequence the catch-up read before its capacity waits and before
/// any flight lookup — only the regression pin.
///
/// 1. **A flight registered for the object** — attachable, or closed and
///    still building (`SnapshotFlights::lookup`) — means the layer is above
///    the atomic-read limits: give the slot back and attach to it
///    ([`catch_up_flight_read`]) with no atomic attempt, sized as the flight
///    knows the layer: the predecessor's measured size, or what the fence's
///    replies named. Otherwise take the slot now if none is held (the flight
///    a brand-new catch-up saw has finished since).
/// 2. The atomic read through [`read_catch_up_snapshot`] (read permit inside
///    the slot, released as the read returns), under the regime's NotFound
///    recovery ([`read_state_with_recovery`], with the stream still only
///    reserved: a header that is actually gone runs the recovery hook and,
///    failing it, drops the reservation and re-bootstraps).
/// 3. On a snapshot: if its cut is below `first_read`, the header sequence
///    regressed between the two reads (only a restore with an undersized band
///    does that, Spec SDD SR-3.2): end `UNAVAILABLE` before any content or ack
///    ([`catch_up_regressed`]) — never through NotFound recovery, because the
///    layer exists. Otherwise register the stream at the cut
///    ([`stream_cursor_at`]): every commit the snapshot holds is at or below
///    it and every later one is delivered live, so nothing lands in both and
///    nothing is lost.
/// 4. On `TOO_LARGE`: give the slot back (the read permit went back as the
///    read returned) and attach to the object's flight
///    ([`catch_up_flight_read`]), sized from the answer (Spec SDD SR-11.1;
///    [`FlightSizing::from_too_large`]).
///
/// On a value return the slot, when this catch-up still holds one (the atomic
/// path), is back in `slot` for the caller to release after its build; on the
/// fenced path `slot` is `None`. The reservation has become the returned
/// consumer.
async fn catch_up_proven_read(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    first_read: u64,
    regime: &'static str,
    reserved: &mut Option<ReservedStream>,
    slot: &mut Option<CatchUpPermit>,
) -> Result<CatchUpRead, Status> {
    if let Some(found) = ctx.svc.snapshot_flights.lookup(object_id) {
        // An attached catch-up holds no slot (Spec SDD SR-7.5).
        *slot = None;
        return Box::pin(catch_up_flight_read(
            ctx,
            object_id,
            first_read,
            regime,
            FlightSizing::Known(found.size),
            reserved,
        ))
        .await;
    }
    if slot.is_none() {
        *slot = Some(acquire_catchup_slot(ctx).await?);
    }
    let read = match read_state_with_recovery(
        ctx,
        object_id,
        Some(CatchUpStream::Reserved(&mut *reserved)),
        slot,
        move || read_catch_up_snapshot(ctx, object_id),
    )
    .await?
    {
        StateRead::Value(read) => read?,
        StateRead::Rebootstrapped(outcome) => return Ok(CatchUpRead::Rebootstrapped(outcome)),
    };

    match read {
        SnapshotRead::Snapshot(snapshot) => {
            if snapshot.cut < first_read {
                return Err(catch_up_regressed(
                    ctx,
                    object_id,
                    first_read,
                    snapshot.cut,
                    regime,
                    "atomic",
                ));
            }
            let reserved = reserved
                .take()
                .expect("only a re-bootstrap drops the reservation, and it returned");
            let consumer = register_stream(ctx, reserved, &stream_cursor_at(snapshot.cut))?;
            Ok(CatchUpRead::Proven {
                snapshot: CatchUpSnapshot::Atomic(snapshot),
                consumer,
            })
        }
        SnapshotRead::TooLarge(too_large) => {
            // `read_object_snapshot` already counted the answer
            // (`outcome="too_large"`) and WARNed if it was the item-tally
            // anomaly; this line only records the route.
            tracing::debug!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                object_id = object_id,
                pin = first_read,
                item_bound = too_large.item_bound,
                early_out = matches!(too_large.at, SnapshotTooLargeAt::EarlyOut),
                regime = regime,
                "catch-up: layer is above the atomic-read limits; reading it through its \
                 shared fenced-read flight"
            );
            // An attached catch-up holds no slot (Spec SDD SR-7.5).
            *slot = None;
            Box::pin(catch_up_flight_read(
                ctx,
                object_id,
                first_read,
                regime,
                FlightSizing::from_too_large(&too_large),
                reserved,
            ))
            .await
        }
    }
}

/// How a catch-up sizes the flight it creates, should it create one when it
/// attaches (Spec SDD SR-7.2, SR-11.1). The flight's combined memory
/// reservation is taken from this size before its first fence reply, so it
/// must already be the design's `b`. Resolved to a [`LayerSize`] only at the
/// attach ([`read_through_flight`]), and only when no flight is registered
/// for the object then: a registered flight's own size wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlightSizing {
    /// A size already in hand: a registered flight's, as found; a mid-read
    /// abort's — the answer's own tallies scaled to the whole layer (the
    /// fitted `bpi` if it tallied no item); or, for a re-attach after an
    /// external-value miss, the missed flight's snapshot's measured tallies
    /// ([`AtomicSnapshot::measured_size`]). No Valkey round trip.
    Known(LayerSize),
    /// An early-out answer: `e × bpi`, with the `bpi` the fence key stored at
    /// the layer's last fenced read when the key holds one, else the fitted
    /// value ([`flight_size`]). The probe for that `bpi` is sent at the
    /// attach, after the registry lookup there misses, so a flight that left
    /// the registry between an earlier lookup and the attach never makes its
    /// successor size from the fitted value unprobed.
    EarlyOut(SnapshotTooLarge),
}

impl FlightSizing {
    /// How the flight is sized after an atomic read answered `too_large`.
    fn from_too_large(too_large: &SnapshotTooLarge) -> Self {
        match too_large.at {
            SnapshotTooLargeAt::EarlyOut => FlightSizing::EarlyOut(*too_large),
            SnapshotTooLargeAt::MidRead { .. } => {
                FlightSizing::Known(LayerSize::from_too_large(too_large, None))
            }
        }
    }
}

/// The size an early-out `too_large` answer gives a flight this catch-up is
/// about to create ([`FlightSizing::EarlyOut`]): `e × bpi`, with the `bpi`
/// the fence key stored at the layer's last fenced read ([`probe_fence`])
/// when the key holds one, else the fitted value. Called only by
/// [`read_through_flight`], after its registry lookup found no flight, so
/// only a catch-up that may create the flight pays the round trip.
///
/// The probe writes nothing and runs after the pin P was read and before
/// `attach_or_create`, holding no catch-up slot or read permit and waiting
/// for nothing: a catch-up has given its slot back before it attaches, and so
/// has a rollback-mode replay handing over ([`ReplayCommit::AboveCeiling`]).
/// It only refines an estimate, so it never ends the catch-up: a probe that
/// fails, or finds the key absent, wrong-typed or holding no `bpi`, sizes from
/// the fitted value — the flight's fence acquire meets and reports any real
/// fault itself.
async fn flight_size(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    too_large: &SnapshotTooLarge,
) -> LayerSize {
    let recorded_bpi = match probe_fence(&ctx.svc.valkey, &ctx.svc.key_schema, object_id).await {
        Ok(FenceProbe::Present { stored_bpi, .. }) => stored_bpi,
        Ok(FenceProbe::Absent | FenceProbe::WrongType { .. }) => None,
        Err(error) => {
            tracing::debug!(
                session_id = %ctx.session_id,
                layer_name = %ctx.layer_name,
                object_id = object_id,
                error = %error,
                "catch-up: fence probe for the layer's recorded bytes per item failed; sizing \
                 its fenced-read flight from the fitted bytes per item"
            );
            None
        }
    };
    LayerSize::from_too_large(too_large, recorded_bpi)
}

/// The fenced path of a catch-up's proven read (Spec SDD SR-3.1, SR-3.2,
/// SR-6.4, SR-7.5): attach to the object's shared fenced-read flight, wait
/// for its read, check its cut against this catch-up's own pin, and register
/// the stream at that cut. Reached from [`catch_up_proven_read`], and from
/// [`catch_up_out_of_window`] when a rollback-mode in-window replay hands over
/// ([`OutOfWindowEntry::AboveCeiling`]).
///
/// Entered holding the reader `reserved` for the object's stream and no
/// catch-up slot or read permit: an attached catch-up holds none; the
/// flight's slot is the unit of work. `first_read` is P, read before this
/// call — `SnapshotFlights::attach_or_create`'s precondition: a catch-up may
/// attach between the fence's grant and its release, and its pin is at most
/// the cut only because it was read first. `sizing` sizes the flight should
/// this catch-up create it (Spec SDD SR-11.1; [`FlightSizing`]).
///
/// 1. Attach and wait for the flight's read ([`read_through_flight`]), under
///    the regime's NotFound recovery ([`read_state_with_recovery`]): a shared
///    `NotFound` reaches each attached catch-up as its own copy and runs its
///    own recovery, and a restored layer is read again by attaching anew,
///    with no slot.
/// 2. If the flight's cut is below `first_read`, the layer's sequence
///    regressed under this catch-up's pin: end it alone, `UNAVAILABLE`,
///    before any content ([`catch_up_regressed`]); the flight goes on for the
///    others. Checked per catch-up, in every regime and in both modes.
/// 3. Register the stream at the flight's cut. From here until its build
///    turn the stream's backlog queues outside the catch-up memory budget, and
///    a registration that reaches the multiplexer's hard backlog cap ends the
///    catch-up before content ([`take_build_turn`]).
///
/// The returned [`CatchUpSnapshot::Shared`] keeps the catch-up attached; it
/// leaves the flight when that, or the build turn it becomes, drops.
async fn catch_up_flight_read(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    first_read: u64,
    regime: &'static str,
    sizing: FlightSizing,
    reserved: &mut Option<ReservedStream>,
) -> Result<CatchUpRead, Status> {
    let mut no_slot = None;
    let read = match read_state_with_recovery(
        ctx,
        object_id,
        Some(CatchUpStream::Reserved(&mut *reserved)),
        &mut no_slot,
        move || read_through_flight(ctx, object_id, sizing),
    )
    .await?
    {
        StateRead::Value(read) => read?,
        StateRead::Rebootstrapped(outcome) => return Ok(CatchUpRead::Rebootstrapped(outcome)),
    };
    let cut = read.snapshot().cut;
    if cut < first_read {
        // Dropping `read` on the way out leaves the flight.
        return Err(catch_up_regressed(
            ctx,
            object_id,
            first_read,
            cut,
            regime,
            "shared_fenced",
        ));
    }
    let reserved = reserved
        .take()
        .expect("only a re-bootstrap drops the reservation, and it returned");
    let consumer = register_stream(ctx, reserved, &stream_cursor_at(cut))?;
    tracing::debug!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        pin = first_read,
        cut = cut,
        role = ?read.role(),
        regime = regime,
        "catch-up: served by its layer's shared fenced snapshot read"
    );
    Ok(CatchUpRead::Proven {
        snapshot: CatchUpSnapshot::Shared(read),
        consumer,
    })
}

/// One attach to `object_id`'s shared fenced-read flight and the wait for its
/// read (Spec SDD SR-6, SR-7.3, SR-5.5): `SnapshotFlights::attach_or_create`
/// — sized as a registered flight knows the layer, else from `sizing`, an
/// early-out's fence probe sent here, on that lookup's miss ([`flight_size`])
/// — then
/// [`FlightWaiter::read`](crate::storage::snapshot_flights::FlightWaiter::read)
/// on this subscription's capacity budget, raced against the subscription's
/// own endings (shutdown, session cancellation, client disconnect, in that
/// order), whose arm leaves the flight by dropping the waiter.
///
/// While attached the catch-up is charged, on its capacity budget, for the
/// time the flight waits for its slot, its fenced read permit or its memory,
/// and counts the flight's work (fence acquires, `RETRY_AFTER` sleeps, the
/// read; not the wait for a predecessor flight) against `reader_wait_max(L)`.
///
/// Two error layers, as on [`read_catch_up_snapshot`]: the outer
/// `Err(ServiceError)` is the shared read's own failure, this catch-up's copy
/// with its classification kept (a `NotFound` routes through the caller's
/// recovery); the inner `Err(Status)` is this catch-up's own ending, its exit
/// reason recorded, always before any content:
///
/// | ending | status | `reason` |
/// |---|---|---|
/// | [`FlightWaitError::CapacityExhausted`] | [`flight_capacity_refused`] | `catchup_capacity` |
/// | [`FlightWaitError::ReaderWaitCap`] | [`fenced_read_wait_refused`] | `fenced_read_wait_cap` |
/// | [`FlightWaitError::ShuttingDown`], or the shutdown token | [`server_shutting_down`] | `shutdown` |
/// | the session token | by its cause ([`SubscriberCtx::end_on_session_cancel`]) | `session_cancel` / `session_absent` |
/// | client disconnect | `CANCELLED` | `client_disconnect` |
async fn read_through_flight(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    sizing: FlightSizing,
) -> Result<Result<FlightRead, Status>, ServiceError> {
    let flights = &ctx.svc.snapshot_flights;
    // One lookup decides the size, immediately before the attach: a flight
    // that leaves the registry after an earlier lookup (the caller's) is
    // missed here, and an early-out answer is probed for then. A flight
    // another catch-up registers during the probe is handled as usual by
    // `attach_or_create`: joined while open (this size unused), or, closed,
    // succeeded by a flight of this size.
    let size = match flights.lookup(object_id) {
        Some(found) => found.size,
        None => match sizing {
            FlightSizing::Known(size) => size,
            FlightSizing::EarlyOut(too_large) => {
                Box::pin(flight_size(ctx, object_id, &too_large)).await
            }
        },
    };
    let waiter = flights.attach_or_create(object_id, size);
    #[cfg(any(test, feature = "test-fault-injection"))]
    let budget_before = ctx.budget.remaining();
    #[cfg(any(test, feature = "test-fault-injection"))]
    let started = tokio::time::Instant::now();
    let read = tokio::select! {
        biased;
        ending = subscription_ending(ctx) => return Ok(Err(ending)),
        read = Box::pin(waiter.read(ctx.budget)) => read,
    };
    #[cfg(any(test, feature = "test-fault-injection"))]
    record_budget_charge(
        ctx,
        BudgetWait::FlightRead,
        budget_before,
        started,
        None,
        budget_before.saturating_sub(ctx.budget.remaining()),
    );
    match read {
        Ok(read) => Ok(Ok(read)),
        Err(FlightWaitError::CapacityExhausted) => {
            ctx.exit.set(StreamExitCell::CATCHUP_CAPACITY);
            Ok(Err(flight_capacity_refused()))
        }
        Err(FlightWaitError::ReaderWaitCap { .. }) => {
            // The flight WARNed with the work time and the cap.
            ctx.exit.set(StreamExitCell::FENCED_READ_WAIT_CAP);
            Ok(Err(fenced_read_wait_refused()))
        }
        Err(FlightWaitError::ShuttingDown) => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            Ok(Err(server_shutting_down()))
        }
        Err(FlightWaitError::Read(error)) => Err(error),
    }
}

/// How often a catch-up waiting for its build turn checks whether its own
/// stream registration has ended (a hard backlog cap reached, the multiplexer
/// shutting down). That ending wakes nothing a catch-up can await without
/// draining its stream, so the turn wait polls it with
/// [`end_catch_up_if_stream_ended`]. The interval sets only how soon such a
/// catch-up gives up its place in the turn queue and frees its backlog: it
/// ends before content either way, since the same check runs again after its
/// build and before its first message.
const BUILD_TURN_STREAM_CHECK: std::time::Duration = std::time::Duration::from_millis(250);

/// Wait until `snapshot` may be built from (Spec SDD SR-6.3): at once for a
/// catch-up's own atomic snapshot, which it builds under the slot it holds;
/// for a shared one, until its build turn in its flight
/// ([`FlightRead::build_turn`]) — the flight's own, in the order the attached
/// catch-ups asked, or at once as an extra build when one more catch-up slot
/// and one more build part are free.
///
/// The turn wait is work: never charged to the capacity budget and not capped
/// by the server. It is bounded by the builds ahead of this catch-up and ends
/// only on the subscription's own endings (shutdown, session cancellation,
/// client disconnect, in that order) or on its stream registration ending,
/// which a multiplexer hard backlog cap reached by the backlog queueing while
/// it waits does — checked every [`BUILD_TURN_STREAM_CHECK`]. All before
/// content. Leaving the wait gives up the catch-up's place and leaves the
/// flight.
///
/// The returned turn counts on `live_edit_catchup_builds_in_progress` until
/// it drops; the caller drops it as its build returns, before its ending
/// check and first send, so the flight's slot and grant never wait for a
/// send.
async fn take_build_turn(
    ctx: &SubscriberCtx<'_>,
    snapshot: CatchUpSnapshot,
    consumer: &LayerStreamConsumer,
) -> Result<BuildSnapshot, Status> {
    let read = match snapshot {
        CatchUpSnapshot::Atomic(snapshot) => return Ok(BuildSnapshot::Atomic(snapshot)),
        CatchUpSnapshot::Shared(read) => read,
    };
    let mut turn = Box::pin(read.build_turn());
    let mut stream_check = tokio::time::interval(BUILD_TURN_STREAM_CHECK);
    stream_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let turn = loop {
        tokio::select! {
            biased;
            ending = subscription_ending(ctx) => return Err(ending),
            turn = &mut turn => break turn,
            _ = stream_check.tick() => end_catch_up_if_stream_ended(ctx, consumer)?,
        }
    };

    // Test-only: park inside the build turn, holding the turn guard, the
    // snapshot handle and the place in the flight, for a test that selected
    // this catch-up (`catch_up_build_turn_seam_for_test`). The park races the
    // subscription's endings, so a parked catch-up still ends by its cause,
    // giving the turn back. Gated behind `test-fault-injection` so release
    // builds never see this branch.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(seam) = ctx.svc.catch_up_build_turn_seam_for_test.as_ref() {
        tokio::select! {
            biased;
            ending = subscription_ending(ctx) => return Err(ending),
            () = seam.park(ctx.client_id, turn.is_extra()) => {}
        }
    }

    Ok(BuildSnapshot::Turn(turn))
}

/// Resolves when one of the subscription's own endings fires — shutdown, then
/// session cancellation, then client disconnect, in the order every wait
/// observes them — with the exit reason recorded and the status it ends with.
/// For a wait that has no other way to observe them; race it first, under
/// `biased`.
async fn subscription_ending(ctx: &SubscriberCtx<'_>) -> Status {
    tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            server_shutting_down()
        }
        () = cancelled_or_never(ctx.session_token()) => ctx.end_on_session_cancel(),
        () = ctx.tx.closed() => {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            Status::cancelled("client disconnected")
        }
    }
}

/// End a catch-up whose proven read's cut is below its pin P (Spec SDD SR-3.2):
/// the layer's sequence went backwards between P and the read, which only a
/// restore with an undersized sequence band causes. P is the header sequence
/// read before the capacity waits, or, for a re-read after an external-value
/// miss ([`reread_and_rebuild`]: its atomic re-read or its re-attach to a
/// flight), the first read's cut. WARNs
/// with the pin and the cut, records `unverified` on
/// `live_subscribe_stable_cut_total` under `regime`, and returns
/// [`sequence_regressed`], before any content or ack. It is never routed
/// through NotFound recovery — the layer exists — and the exit reason stays
/// `error`.
///
/// `read` names the read whose cut it is: `atomic`, the catch-up's own; or
/// `shared_fenced`, its flight's, checked against each attached catch-up's own
/// pin, which ends only that catch-up while the flight goes on for the others.
fn catch_up_regressed(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    pin: u64,
    cut: u64,
    regime: &'static str,
    read: &'static str,
) -> Status {
    tracing::warn!(
        session_id = %ctx.session_id,
        layer_name = %ctx.layer_name,
        object_id = object_id,
        pin = pin,
        cut = cut,
        regime = regime,
        read = read,
        "catch-up snapshot read's cut is below its pin, the header sequence read before the \
         capacity waits (or, on a re-read after an external-value miss, the first read's cut): \
         the layer's sequence regressed, as a restore with an undersized sequence band does; \
         ending the subscription UNAVAILABLE before any content so the client re-subscribes \
         from the current header"
    );
    record_stable_cut(ctx, false, regime);
    sequence_regressed()
}

/// Register this subscription's stream with the multiplexer at once, for a
/// path with no capacity wait after it (up-to-date, and the rollback-mode
/// in-window replay, which registers under its slot at the last entry its
/// in-window read returned): [`reserve_stream`] then [`register_stream`].
async fn open_consumer(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    cursor: StreamCursor,
) -> Result<LayerStreamConsumer, Status> {
    let reserved = reserve_stream(ctx, object_id).await?;
    register_stream(ctx, reserved, &cursor)
}

/// Reserve a stream reader for this subscription, without registering its
/// stream ([`ReservedStream`]): nothing is queued for it until
/// [`register_stream`]. A catch-up reserves before its capacity waits, so the
/// reader — on a cluster, its hash slot — stays bound to it through them, and
/// registers at its proven read's cut, after every capacity wait
/// ([`catch_up_proven_read`]).
///
/// In cluster mode the reservation may wait for a free stream reader; that
/// wait ends at the subscription's capacity-budget deadline and is charged to
/// the budget. It races the endings first, in the order every other wait
/// observes them — shutdown, session cancellation, client disconnect — so it
/// ends by its cause rather than only on the multiplexer's own close. Each
/// `ConsumeError` maps to its ending ([`consume_error_status`]).
async fn reserve_stream(ctx: &SubscriberCtx<'_>, object_id: u64) -> Result<ReservedStream, Status> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    let budget_before = ctx.budget.remaining();
    let started = tokio::time::Instant::now();
    let deadline = ctx.budget.deadline();
    let reserved = tokio::select! {
        biased;
        () = ctx.shutdown.cancelled() => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            return Err(server_shutting_down());
        }
        () = cancelled_or_never(ctx.session_token()) => {
            return Err(ctx.end_on_session_cancel());
        }
        () = ctx.tx.closed() => {
            ctx.exit.set(StreamExitCell::CLIENT_DISCONNECT);
            return Err(Status::cancelled("client disconnected"));
        }
        reserved = reserve_layer_stream(
            &ctx.svc.valkey,
            &ctx.svc.key_schema,
            object_id,
            deadline,
        ) => reserved,
    };
    let waited = started.elapsed();
    ctx.budget.charge(waited);
    #[cfg(any(test, feature = "test-fault-injection"))]
    record_budget_charge(
        ctx,
        BudgetWait::StreamReader,
        budget_before,
        started,
        Some(deadline),
        waited,
    );
    reserved.map_err(|err| consume_error_status(ctx, err))
}

/// Register a reserved stream at `cursor` and return its consumer. No wait:
/// the reader was reserved by [`reserve_stream`]. A multiplexer that closed
/// meanwhile ends the subscription `shutdown` ([`consume_error_status`]); the
/// reservation is dropped either way on failure, releasing the reader.
fn register_stream(
    ctx: &SubscriberCtx<'_>,
    reserved: ReservedStream,
    cursor: &StreamCursor,
) -> Result<LayerStreamConsumer, Status> {
    reserved
        .register(cursor)
        .map_err(|err| consume_error_status(ctx, err))
}

/// The ending of a failed stream reservation or registration:
///
/// | `ConsumeError` | exit reason | status |
/// |---|---|---|
/// | `ShuttingDown` | `shutdown` | [`server_shutting_down`] |
/// | `Saturated` | `reader_capacity` | `UNAVAILABLE` ([`reader_capacity_refused`]) |
/// | `InvalidCursor` | `error` | `INTERNAL` — unreachable: every cursor here is built from a stream id |
fn consume_error_status(ctx: &SubscriberCtx<'_>, err: ConsumeError) -> Status {
    match err {
        ConsumeError::ShuttingDown => {
            ctx.exit.set(StreamExitCell::SHUTDOWN);
            server_shutting_down()
        }
        ConsumeError::Saturated => {
            ctx.exit.set(StreamExitCell::READER_CAPACITY);
            reader_capacity_refused()
        }
        // The exit reason stays at its `error` default.
        err @ ConsumeError::InvalidCursor(_) => ServiceError::Internal(err.to_string()).into(),
    }
}

/// Attempt lazy on-demand recovery (SDD SR-2.4) after a state-read returns
/// `NotFound` from somewhere inside `subscribe.rs`.
///
/// Returns:
///
/// - `Ok(true)` — `RecoveryOutcome::Restored`: a checkpoint existed and
///   the layer was materialized (or a peer already landed it). The caller
///   SHOULD retry the originating state-read exactly once. Because
///   recovery does NOT `XADD` to `le:{oid}:obj:stream`, any subscriber
///   that was previously in-window will now fall through to the
///   out-of-window (Regime 3) coalesced-diff path on the retry — exactly
///   what SDD §12.3 requires for a post-recovery bootstrap.
///
/// - `Ok(false)` — `RecoveryOutcome::NotRecoverable`: either no checkpoint
///   exists for this object_id, or the restore-source selection adjudicated
///   every full generation the store holds and none of them is usable, or the
///   apply's SR-2.4 live-mapping guard rejected because the mapping retired or
///   now resolves to a different object_id. What this arm does NOT cover is a
///   selection that ran out of budget before it could read every full
///   generation: that is `Exhaustion::BudgetInconclusive` →
///   `CheckpointError::Transient` → `RecoveryOutcome::Error(_)`, i.e. the `Err`
///   arm below and a retriable `UNAVAILABLE`, because a backup the walk never
///   read is not evidence that no backup exists (b.hxk). So `Ok(false)` is an
///   affirmative "nothing here can be restored", not "the restore did not
///   happen".
///
///   The caller SHOULD fall through to [`handle_deleted_and_rebootstrap`] with
///   [`RebootstrapCause::StateGone`], which preserves the b.jvx
///   overwrite-detection semantics: re-read the layer mapping and either
///   terminate with NotFound or re-bootstrap on the new object_id (CreateLayer
///   overwrite during state-loss). `StateGone` is what keeps the terminal arm
///   from telling the subscriber its layer "was deleted" — nothing on this path
///   observed a deletion.
///
/// - `Err(Status)` — `RecoveryOutcome::Error(_)`: classified backend
///   failure during the recovery attempt, routed through
///   [`ServiceError::from_recovery_error`] (b.33e). A `Transient` failure
///   surfaces as gRPC `Code::Unavailable` per the project's transient-error
///   contract, so the client's standard retry policy reconnects.
///   `Permanent` / `NotFound` / `Conflict` /
///   `NotImplemented` surface as `Code::Internal` — a retry cannot heal a
///   corrupt blob or an inconsistent store, and INTERNAL is what gets the
///   full message into the operator's ERROR log.
///
/// **Disabled short-circuit.** When `LIVE_EDIT_CHECKPOINT_ENABLED=false`,
/// `RecoveryHook::try_restore_layer` returns `NotRecoverable` immediately
/// without touching the admission map, the store, or the metric — so this
/// helper degenerates to `Ok(false)` and the call site behaves bit-for-bit
/// like pre-recovery code (fall through to
/// `handle_deleted_and_rebootstrap`). This is the other cause that must not
/// report "the layer was deleted": with checkpointing off the service has no
/// opinion about the layer at all, which is exactly
/// [`RebootstrapCause::StateGone`].
async fn try_recover_after_notfound(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
) -> Result<bool, Status> {
    match ctx.svc.recovery.try_restore_layer(object_id).await {
        RecoveryOutcome::Restored => Ok(true),
        RecoveryOutcome::NotRecoverable => Ok(false),
        RecoveryOutcome::Error(err) => Err(ServiceError::from_recovery_error(
            &err,
            format!("layer-state recovery failed for object {object_id}: {err}"),
        )
        .into()),
    }
}

/// Which catch-up a cut-based build site runs again once
/// [`recover_after_reread_not_found`] has restored its layer: the one its own
/// read's recovery would have continued with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestoredRerun {
    /// The brand-new site ([`catch_up_brand_new`], also every re-bootstrap):
    /// a fresh brand-new catch-up of the object, whose from-empty full state
    /// is what this site sends.
    BrandNew,
    /// A reconnector's site — Regime 2's snapshot arm or Regime 3: a fresh
    /// Regime 3 catch-up ([`catch_up_out_of_window`] through its proven
    /// read), whose coalesced diff is based at the client's `have_version`,
    /// never a `base_version = 0` reset an advanced reconnector would refuse.
    /// Regime 2's snapshot arm builds the same faithful coalesced diff, and a
    /// restore appends no stream entry, which is what sends a restored
    /// layer's reconnector to Regime 3 ([`try_recover_after_notfound`]).
    /// `current_seq` is the site's pin P.
    OutOfWindow { current_seq: u64 },
}

/// A cut-based catch-up whose external-value miss re-read answered `NotFound`
/// ([`CatchUpBuilt::RereadNotFound`]): the recovery its own first read would
/// have run (SDD §11.2), run here because the re-read met the `NotFound`
/// under the `{gc}` read hold, and no restore may run under it.
///
/// Entered with the hold released, no catch-up slot or read permit held and
/// the stream registration released (the re-read swapped it for a
/// reservation, which its `NotFound` dropped), before any content. Then, as
/// [`read_state_with_recovery`] does for a first read:
///
/// - [`try_recover_after_notfound`]'s `Err` (classified per b.33e) is
///   returned as-is;
/// - `NotRecoverable` (or checkpointing disabled) goes to
///   [`handle_deleted_and_rebootstrap`] with [`RebootstrapCause::StateGone`],
///   which re-bootstraps onto an overwrite's new object or ends `NOT_FOUND`;
/// - `Restored` re-bootstraps on the restored layer: the catch-up `rerun`
///   names runs again from its start, with its own reservation, slot and
///   proven read, charged to the same capacity budget (a restore is not an
///   overwrite, so the budget is not refilled). A first read's recovery reads
///   once more in place; a re-read cannot, because the regime's registration,
///   slot and first cut have already been given up for it. The re-run has
///   its own miss protocol, so another round of this needs another miss,
///   another `NotFound` and another restore.
async fn recover_after_reread_not_found(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    rerun: RestoredRerun,
) -> Result<CatchUpOutcome, Status> {
    if !Box::pin(try_recover_after_notfound(ctx, object_id)).await? {
        return Box::pin(handle_deleted_and_rebootstrap(
            ctx,
            object_id,
            RebootstrapCause::StateGone,
        ))
        .await;
    }
    match rerun {
        RestoredRerun::BrandNew => Box::pin(catch_up_brand_new(ctx, object_id)).await,
        RestoredRerun::OutOfWindow { current_seq } => {
            // Reader first, then the slot, as every snapshot-mode catch-up
            // takes them: a cluster-mode reader wait never holds a slot.
            let reserved = reserve_stream(ctx, object_id).await?;
            let slot = acquire_catchup_slot(ctx).await?;
            Box::pin(catch_up_out_of_window(
                ctx,
                object_id,
                current_seq,
                reserved,
                slot,
                OutOfWindowEntry::ProvenRead,
            ))
            .await
        }
    }
}

/// What a catch-up state read under [`read_state_with_recovery`] produced.
enum StateRead<T> {
    /// The read succeeded, first time or on its one retry after a restore.
    Value(T),
    /// The state is gone and the subscriber was re-bootstrapped onto the
    /// layer's current object. (A re-bootstrap that ends the subscription
    /// NOT_FOUND arrives as the helper's `Err`.) The caller returns this
    /// outcome as its own.
    Rebootstrapped(CatchUpOutcome),
}

/// The catch-up's layer stream at a recovering read
/// ([`read_state_with_recovery`]).
enum CatchUpStream<'a> {
    /// The reader reserved before the capacity waits, its stream not
    /// registered yet: every recovering read precedes the registration — the
    /// atomic read, or the wait for a flight's shared fenced read, whose cut
    /// the stream then registers at. The recovery empties the `Option` —
    /// dropping the reservation and releasing the reader — only on its way
    /// into a re-bootstrap.
    Reserved(&'a mut Option<ReservedStream>),
}

/// Run one catch-up state read under the NotFound recovery policy (SDD SR-2.4,
/// b.jvx, b.fgj) that every recovering catch-up read site shares:
///
/// 1. Read.
/// 2. On `NotFound`, release the catch-up slot (if `slot` holds one) and
///    attempt [`try_recover_after_notfound`]. Its `Err` (classified per b.33e)
///    is returned as-is; the site's `stream` is then dropped by the caller's
///    return rather than shut down.
/// 3. On `Restored`, re-acquire a catch-up slot if the read held one
///    ([`acquire_catchup_slot`], charged to the same capacity budget), then
///    read again — exactly once. The site's stream is at most reserved, which
///    has no queue and no recorded ending, so there is nothing to check.
/// 4. On a second `NotFound`, or on `NotRecoverable`: release the slot and
///    drop the site's reservation (if it has one), so the old object's reader
///    is released before the new object's is reserved, then
///    [`handle_deleted_and_rebootstrap`] with [`RebootstrapCause::StateGone`].
/// 5. Any other error propagates unchanged through `From<ServiceError> for
///    Status`.
///
/// `slot` is the catch-up slot the site's reads run under: `Some` at every
/// site but the same-object header read, which precedes the slot, and the
/// wait for a flight's shared fenced read ([`catch_up_flight_read`]), which
/// holds none — its retry after a restore attaches again with no slot. A slot
/// is never held across a recovery attempt or a re-bootstrap; on
/// [`StateRead::Value`] the site's slot (re-acquired after a restore) is back
/// in `slot` for the caller to release before it sends. A read that takes a
/// snapshot read permit ([`read_catch_up_snapshot`]) takes it inside the read
/// and releases it before the read returns, so no permit is held across a
/// recovery attempt, a slot wait or a re-bootstrap either, and the retried
/// read takes its permit again under the re-acquired slot.
///
/// Callers keep what differs between them: the read itself, its mode and
/// regime labels, and any test seam, which fires once before the first
/// attempt.
///
/// Stack (b.9qr): `read` is awaited at one point for both attempts, and the
/// rarely taken recovery attempt and re-bootstrap — the latter inlines
/// `catch_up_brand_new`'s whole state machine — are boxed, so a debug build
/// carries one copy of each future per call site rather than several.
async fn read_state_with_recovery<T, F, Fut>(
    ctx: &SubscriberCtx<'_>,
    object_id: u64,
    stream: Option<CatchUpStream<'_>>,
    slot: &mut Option<CatchUpPermit>,
    mut read: F,
) -> Result<StateRead<T>, Status>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ServiceError>>,
{
    let reads_under_slot = slot.is_some();
    let mut retried = false;
    loop {
        match read().await {
            Ok(value) => return Ok(StateRead::Value(value)),
            Err(ServiceError::NotFound(_)) if !retried => {
                // No slot across the recovery attempt.
                *slot = None;
                if !Box::pin(try_recover_after_notfound(ctx, object_id)).await? {
                    break;
                }
                if reads_under_slot {
                    *slot = Some(acquire_catchup_slot(ctx).await?);
                }
                retried = true;
            }
            Err(ServiceError::NotFound(_)) => break,
            Err(other) => return Err(other.into()),
        }
    }
    // No slot across the re-bootstrap, which takes its own, and no hold on the
    // old object's reader.
    *slot = None;
    if let Some(CatchUpStream::Reserved(reserved)) = stream {
        drop(reserved.take());
    }
    let outcome = Box::pin(handle_deleted_and_rebootstrap(
        ctx,
        object_id,
        RebootstrapCause::StateGone,
    ))
    .await?;
    Ok(StateRead::Rebootstrapped(outcome))
}

/// Pick the entry cap of the rollback-mode in-window replay
/// ([`read_in_window_replay`], which pages up to it). Uses the LayerStore's
/// configured MAXLEN when available (a subscriber can't be more entries
/// behind than the stream retains), falling back to a soft cap so a stream
/// with no entry cap doesn't yield an unbounded replay.
fn in_window_xrange_count(svc: &LiveEditService) -> u64 {
    let maxlen = svc.layer_store.layer_stream_maxlen();
    if maxlen == 0 {
        IN_WINDOW_XRANGE_SOFT_CAP
    } else {
        maxlen
    }
}

/// Self-echo suppression predicate (SDD §12.5).
///
/// Returns `true` if the notification should be suppressed. Both client IDs
/// must be non-empty and equal for suppression to trigger.
fn is_self_echo(subscriber_client_id: &str, notification_client_id: &str) -> bool {
    !subscriber_client_id.is_empty()
        && !notification_client_id.is_empty()
        && subscriber_client_id == notification_client_id
}

/// Test-only synchronization seam for the `catch_up_in_window` mid-window
/// `Deleted` variant (b.fdy). When installed on a `LiveEditService` via
/// `catch_up_xrange_seam_for_test`, the subscriber's `run_catch_up` pauses
/// between the `read_object_header` call and the in-window stream read
/// long enough for a test to fire a `create_layer` overwrite against
/// a parked subscriber.
///
/// **Two-phase rendezvous.** The seam separates "the subscriber has parked"
/// from "release the subscriber" using two 2-party barriers, because a
/// single barrier cannot express both without a race: the test must be able
/// to observe that the subscriber has *already* passed its mapping + header
/// reads on the OLD object (so an overwrite fired next cannot be seen by
/// those reads) BEFORE it mutates state, and must NOT release the subscriber
/// until that mutation is complete and visible. Concretely:
///
///   * The subscriber side calls [`CatchUpXrangeSeam::arrive_and_wait`],
///     which announces arrival (rendezvous 1) and then blocks until the test
///     releases it (rendezvous 2).
///   * The test side calls [`CatchUpXrangeSeam::wait_until_parked`] to await
///     rendezvous 1 — after it returns the subscriber is guaranteed parked
///     just before the in-window stream read, having already resolved the mapping and
///     read the header against the OLD object — then mutates state (fires the
///     overwrite, and either holds or drains the async cleanup depending on
///     which post-overwrite ordering it is exercising), then calls
///     [`CatchUpXrangeSeam::release`] to complete rendezvous 2.
///
/// A single-barrier design (the pre-b.tr5 version) conflated the two phases:
/// the test fired the overwrite before any confirmation the subscriber had
/// parked, so a slow subscriber could observe the *post*-overwrite mapping,
/// take the brand-new / NotFound path, and never reach the seam — deadlocking
/// the test's lone barrier arrival. The split closes that race.
///
/// **Test-only.** Production code never constructs this — it is gated
/// behind `#[cfg(any(test, feature = "test-fault-injection"))]` and the
/// field that holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct CatchUpXrangeSeam {
    /// Rendezvous 1: the subscriber announces it has parked at the seam.
    parked: tokio::sync::Barrier,
    /// Rendezvous 2: the test releases the parked subscriber.
    released: tokio::sync::Barrier,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CatchUpXrangeSeam {
    /// Build a seam expecting exactly one subscriber and one test task at
    /// each of the two rendezvous points.
    pub fn new() -> Self {
        Self {
            parked: tokio::sync::Barrier::new(2),
            released: tokio::sync::Barrier::new(2),
        }
    }

    /// Subscriber side (called from inside `run_catch_up`): announce arrival
    /// at the seam (rendezvous 1), then block until the test releases it
    /// (rendezvous 2). Call exactly once — each underlying `Barrier` is
    /// 2-party and a third arrival would hang.
    pub async fn arrive_and_wait(&self) {
        let _ = self.parked.wait().await;
        let _ = self.released.wait().await;
    }

    /// Test side: block until the subscriber has parked at the seam
    /// (rendezvous 1). After this returns, the subscriber has passed its
    /// mapping + header reads on the OLD object and is blocked just before
    /// the in-window stream read; the test may now mutate state with no race against
    /// those reads. Must be paired with a later [`Self::release`].
    pub async fn wait_until_parked(&self) {
        let _ = self.parked.wait().await;
    }

    /// Test side: release the parked subscriber (rendezvous 2), letting its
    /// in-window stream read run. Call only after [`Self::wait_until_parked`] has
    /// returned and the interleaved state mutation is complete and visible.
    pub async fn release(&self) {
        let _ = self.released.wait().await;
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Default for CatchUpXrangeSeam {
    fn default() -> Self {
        Self::new()
    }
}

/// Test-only synchronization seam for `run_catch_up_inner` (b.xmk / b.msb).
/// When installed on a `LiveEditService` via `run_catch_up_inner_seam_for_test`,
/// the subscriber parks between the layer-mapping read at the top of
/// `run_catch_up_inner` (past the overwrite-detect branch
/// `ctx.have_object_id != object_id`) and the same-object
/// `read_object_header(OLD)` call — the "Site B" position. This lets a test
/// interleave a `create_layer` overwrite + `cleanup_old_object_retrying` UNLINK against
/// the parked subscriber, covering the b.jvx Site B classifier (the
/// `ServiceError::NotFound → handle_deleted_and_rebootstrap` arm on the
/// initial `read_object_header(OLD)`).
///
/// **Why the two-phase gate keeps the Site-B park (b.msb).** The pre-b.msb
/// seam used a single 2-party barrier fired *concurrently* with the overwrite:
/// a test hoped the subscriber's mapping read won, but if the overwrite
/// committed the mapping → NEW first, the subscriber read NEW, took the
/// overwrite-detect branch, and never reached the seam — so the lone barrier
/// arrival deadlocked (120 s hard-kill). The fix is the two-phase handshake
/// below, NOT moving the park: the test fires its overwrite only AFTER
/// [`wait_until_parked`](Self::wait_until_parked) returns, so the subscriber's
/// mapping read is guaranteed to observe OLD (overwrite not yet fired) and the
/// subscriber is guaranteed to reach this Site-B park. Moving the park *before*
/// the mapping read would instead make the post-release mapping read observe
/// NEW and route the subscriber down `catch_up_brand_new(NEW)` — never calling
/// `read_object_header(OLD)`, so the injected header-NotFound fault would never
/// fire and the test would false-pass on the wrong path.
///
/// **Two-phase rendezvous.** Mirrors [`CatchUpXrangeSeam`]: a single barrier
/// cannot express both "the subscriber has parked" and "release the
/// subscriber" without a race. The `parked` barrier (rendezvous 1) lets the
/// test observe that the subscriber is blocked at the Site-B seam — having
/// already read `mapping == OLD` — before it mutates state; the `released`
/// barrier (rendezvous 2) holds the subscriber until that mutation (overwrite
/// + cleanup drain) is complete and visible.
///
/// **Test-only.** Production code never constructs this — it is gated
/// behind `#[cfg(any(test, feature = "test-fault-injection"))]` and the
/// field that holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct RunCatchUpInnerSeam {
    /// Rendezvous 1: the subscriber announces it has parked at the Site-B seam
    /// (after the mapping read, before `read_object_header(OLD)`).
    parked: tokio::sync::Barrier,
    /// Rendezvous 2: the test releases the parked subscriber.
    released: tokio::sync::Barrier,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl RunCatchUpInnerSeam {
    /// Build a seam expecting exactly one subscriber and one test task at
    /// each of the two rendezvous points.
    pub fn new() -> Self {
        Self {
            parked: tokio::sync::Barrier::new(2),
            released: tokio::sync::Barrier::new(2),
        }
    }

    /// Subscriber side (called from inside `run_catch_up_inner`, after the
    /// layer-mapping read + overwrite-detect branch and before the same-object
    /// `read_object_header(OLD)` call): announce arrival at the Site-B seam
    /// (rendezvous 1), then block until the test releases it (rendezvous 2).
    /// Call exactly once — each underlying `Barrier` is 2-party and a third
    /// arrival would hang.
    pub async fn arrive_and_wait(&self) {
        let _ = self.parked.wait().await;
        let _ = self.released.wait().await;
    }

    /// Test side: block until the subscriber has parked at the Site-B seam
    /// (rendezvous 1). After this returns, the subscriber has already read
    /// `mapping == OLD` and is blocked just before `read_object_header(OLD)`;
    /// the test may now fire its overwrite and drain its cleanup body
    /// (`cleanup_old_object_retrying`) with
    /// no race against that header read. Must be paired with a later
    /// [`Self::release`].
    pub async fn wait_until_parked(&self) {
        let _ = self.parked.wait().await;
    }

    /// Test side: release the parked subscriber (rendezvous 2), letting its
    /// layer-mapping read (and the subsequent `read_object_header`) fire.
    /// Call only after [`Self::wait_until_parked`] has returned and the
    /// interleaved overwrite + cleanup drain is complete and visible.
    pub async fn release(&self) {
        let _ = self.released.wait().await;
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Default for RunCatchUpInnerSeam {
    fn default() -> Self {
        Self::new()
    }
}

/// Test-only synchronization seam for `catch_up_out_of_window` (b.xmk).
/// When installed on a `LiveEditService` via
/// `catch_up_out_of_window_seam_for_test`, the subscriber pauses at one of
/// two distinct points inside `catch_up_out_of_window`:
///
///   * **Pre-header-read** — right after entry, holding the caller's
///     stream-reader reservation, before the regime's first read of the OLD
///     object's state. Covers the b.jvx Site C arm 1 classifier
///     (`ServiceError::NotFound → handle_deleted_and_rebootstrap`).
///   * **Pre-compute_diff** — immediately after it, still before the read
///     permit and the state read. Covers the b.jvx Site C arm 2 classifier
///     (`ServiceError::NotFound → handle_deleted_and_rebootstrap` on the state
///     read).
///
/// Since the catch-up's proven read reads the header and the state in one
/// read — atomic, or above the atomic-read limits the shared fenced read of
/// the flight it attaches to — the two points sit back to back, both before
/// that read: a subscriber parked at either holds a reader reservation and no
/// registered stream (it registers at the read's cut), and either point's
/// NotFound arrives from the same read. Each fires once per catch-up, before
/// its atomic attempt or its attach.
///
/// Each instance installs exactly one of the two park points — selected by
/// the constructor used — so a test can scope its parked window to the
/// classifier under test without parking the other arm. Both sit after the
/// catch-up has taken its concurrent-catch-up slot, so a subscriber parked at
/// either holds that slot (and no read permit).
///
/// Each park point is a 2-party rendezvous mirroring [`CatchUpXrangeSeam`].
/// The subscriber task arrives from inside `catch_up_out_of_window`, and
/// the test task arrives after firing the overwrite. Both arrivals must
/// complete for either side to proceed.
///
/// **Parked latch — required before a test mutates the layer mapping (b.y3a).**
/// The barrier alone expresses "release the subscriber"; it cannot express "the
/// subscriber has *reached* the park". A test that fires a `CreateLayer`
/// overwrite without first observing the park is racing the subscriber's
/// layer-mapping read in `run_catch_up_inner` — and when the overwrite wins, the
/// subscriber reads `mapping == NEW`, takes the overwrite-detect branch into
/// `catch_up_brand_new(NEW)`, and never enters `catch_up_out_of_window` at all.
/// The test's lone barrier arrival then blocks forever (120 s hard-kill). This
/// is the same failure [`RunCatchUpInnerSeam`] documents for Site B, where it was
/// fixed by a two-phase handshake; the measured margin on the Site C tests was
/// ±1 ms, so any timing perturbation flipped it. Each park point therefore
/// raises a level-triggered `parked` latch immediately before blocking, which
/// [`wait_until_parked_pre_header_read`](Self::wait_until_parked_pre_header_read)
/// / [`wait_until_parked_pre_compute_diff`](Self::wait_until_parked_pre_compute_diff)
/// let the test await first.
///
/// The latch is level-triggered, not edge-triggered, so it is order-independent:
/// a test that calls `wait_until_parked_*` after the subscriber already parked
/// returns immediately rather than missing the signal.
///
/// **The park points stay where they are — downstream of the mapping read.**
/// The latch is the fix for the overwrite race; moving a park point upstream of
/// `run_catch_up_inner`'s mapping read is not. Park earlier and the post-release
/// mapping read observes `NEW`, routing the subscriber into
/// `catch_up_brand_new(NEW)` — it never reads OLD's header or state, so the
/// OLD-object `NotFound` classifier under test never
/// fires and the test false-passes on the wrong path. Same warning
/// [`RunCatchUpInnerSeam`] carries.
///
/// **Test-only.** Production code never constructs this — it is gated
/// behind `#[cfg(any(test, feature = "test-fault-injection"))]` and the
/// field that holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct CatchUpOutOfWindowSeam {
    park_before_header_read: Option<SeamParkPoint>,
    park_before_compute_diff: Option<SeamParkPoint>,
}

/// One park point: the release rendezvous plus the `parked` latch that lets a
/// test observe the subscriber reached it (b.y3a).
///
/// Split out rather than inlined because [`CatchUpOutOfWindowSeam`]'s two slots
/// and [`CatchUpStateReadSeam`]'s pre-full-state and pre-coalesced slots all
/// need the identical pairing, and the auto-complete (1-party) slot must behave
/// exactly like the real one minus the waiting.
#[cfg(any(test, feature = "test-fault-injection"))]
struct SeamParkPoint {
    /// Release rendezvous. 2 parties for the slot under test, 1 for the
    /// auto-complete slot the other constructor installs.
    barrier: tokio::sync::Barrier,
    /// Level-triggered latch raised by the subscriber immediately before it
    /// blocks on `barrier`. `watch` (not `Notify`) so a test that arrives late
    /// still observes the raised state instead of waiting for an edge that
    /// already passed.
    parked: tokio::sync::watch::Sender<bool>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl SeamParkPoint {
    fn new(parties: usize) -> Self {
        Self {
            barrier: tokio::sync::Barrier::new(parties),
            parked: tokio::sync::watch::channel(false).0,
        }
    }

    /// Subscriber side: raise the `parked` latch, then block for release.
    async fn arrive(&self) {
        // `send_replace`, not `send`: the latter reports an error and skips the
        // update when no receiver exists, and no receiver exists whenever the
        // test never calls `wait_until_parked_*` (every pre-b.y3a call site).
        self.parked.send_replace(true);
        let _ = self.barrier.wait().await;
    }

    /// Test side: block until the subscriber has raised the `parked` latch.
    async fn wait_until_parked(&self) {
        let mut rx = self.parked.subscribe();
        while !*rx.borrow_and_update() {
            if rx.changed().await.is_err() {
                return;
            }
        }
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CatchUpOutOfWindowSeam {
    /// Build a seam that parks the subscriber at the pre-header-read point
    /// inside `catch_up_out_of_window` (b.jvx Site C arm 1): after its
    /// stream-reader reservation, before its first read of the OLD object's
    /// state.
    ///
    /// Both `Option<SeamParkPoint>` slots are populated even though only the
    /// `park_before_header_read` slot is the actual rendezvous. The
    /// downstream `park_before_compute_diff` slot holds a single-party
    /// `Barrier::new(1)` so the production-side `arrive_at_pre_compute_diff`
    /// arrival completes immediately on its single arrival — preserving
    /// the downstream-no-op semantics the production unconditional call
    /// site needs, while satisfying the
    /// `debug_assert!(park_before_compute_diff.is_some())` safety net the
    /// b.m52 hardening installed on the arrival method. Symmetric mirror
    /// of the dual-barrier pattern in
    /// [`Self::new_before_compute_diff`] — see that constructor for the
    /// full b.m52 rationale block; this constructor inherits the identical
    /// rationale. Without the auto-complete barrier here, the
    /// `debug_assert!` would fire under `cargo test` (which compiles with
    /// debug-asserts enabled) the moment production hit the downstream
    /// seam call, panicking every arm-1 test that reaches the
    /// pre-compute_diff site before it could reach its real rendezvous.
    pub fn new_before_header_read() -> Self {
        Self {
            park_before_header_read: Some(SeamParkPoint::new(2)),
            // 1-party barrier: completes on the single production-side
            // arrival from `arrive_at_pre_compute_diff`. The test never
            // calls into this barrier.
            park_before_compute_diff: Some(SeamParkPoint::new(1)),
        }
    }

    /// Build a seam that parks the subscriber at the pre-compute_diff point
    /// inside `catch_up_out_of_window` (b.jvx Site C arm 2): just before its
    /// read permit and its state read.
    ///
    /// Both `Option<SeamParkPoint>` slots are populated even though only the
    /// `park_before_compute_diff` slot is the actual rendezvous. The
    /// upstream `park_before_header_read` slot holds a single-party
    /// `Barrier::new(1)` so the production-side `arrive_at_pre_header_read`
    /// arrival completes immediately on its single arrival — preserving
    /// the upstream-no-op semantics the production unconditional call site
    /// needs, while satisfying the
    /// `debug_assert!(park_before_header_read.is_some())` safety net the
    /// b.m52 hardening installed on the arrival method. b.m52 record:
    /// without the auto-complete barrier here, the `debug_assert!` would
    /// fire under `cargo test` (which compiles with debug-asserts enabled)
    /// the moment production hit the upstream seam call, panicking every
    /// arm-2 test before it could reach its real rendezvous.
    pub fn new_before_compute_diff() -> Self {
        Self {
            // 1-party barrier: completes on the single production-side
            // arrival from `arrive_at_pre_header_read`. The test never
            // calls into this barrier.
            park_before_header_read: Some(SeamParkPoint::new(1)),
            park_before_compute_diff: Some(SeamParkPoint::new(2)),
        }
    }

    /// Rendezvous at the pre-header-read park point. Both named
    /// constructors populate this slot (`new_before_header_read` with
    /// a 2-party real-rendezvous barrier; `new_before_compute_diff` with
    /// a 1-party auto-complete barrier — see that constructor for the
    /// rationale), so calling this is always safe today. The call site
    /// can unconditionally invoke it without first pattern-matching on
    /// the constructor.
    ///
    /// The `debug_assert!` is a fail-loud safety net for future
    /// constructors: any new constructor that leaves
    /// `park_before_header_read` as `None` would silently no-op the
    /// rendezvous, racing past the intended park point. Production
    /// never constructs `CatchUpOutOfWindowSeam` (the type is gated out
    /// by `cfg(any(test, feature = "test-fault-injection"))`), so the
    /// assertion has zero release-build cost. The `if let Some(...)`
    /// block is retained so the production call site's unconditional
    /// invocation pattern compiles regardless of debug-asserts.
    pub async fn arrive_at_pre_header_read(&self) {
        debug_assert!(
            self.park_before_header_read.is_some(),
            "arrive_at_pre_header_read called on a seam whose \
             park_before_header_read slot is None; rendezvous would \
             silently no-op"
        );
        if let Some(p) = self.park_before_header_read.as_ref() {
            p.arrive().await;
        }
    }

    /// Test side: block until the subscriber has parked at the pre-header-read
    /// point (b.y3a).
    ///
    /// After this returns, the subscriber has already read `mapping == OLD`,
    /// resolved the out-of-window regime, and reserved its stream reader — so
    /// the test may now fire a `CreateLayer` overwrite and drain
    /// its cleanup body (`cleanup_old_object_retrying`) with no race against
    /// the mapping read that would
    /// otherwise divert the subscriber into `catch_up_brand_new(NEW)`. Pair with
    /// a later [`Self::arrive_at_pre_header_read`], which performs the release.
    pub async fn wait_until_parked_pre_header_read(&self) {
        if let Some(p) = self.park_before_header_read.as_ref() {
            p.wait_until_parked().await;
        }
    }

    /// Rendezvous at the pre-compute_diff park point. Both named
    /// constructors populate this slot (`new_before_compute_diff` with
    /// a 2-party real-rendezvous barrier; `new_before_header_read` with
    /// a 1-party auto-complete barrier — see that constructor for the
    /// rationale), so calling this is always safe today. The call site
    /// can unconditionally invoke it without first pattern-matching on
    /// the constructor.
    ///
    /// See [`Self::arrive_at_pre_header_read`] for the rationale behind
    /// the `debug_assert!` + retained `if let Some(...)` pattern (this
    /// method mirrors that one).
    pub async fn arrive_at_pre_compute_diff(&self) {
        debug_assert!(
            self.park_before_compute_diff.is_some(),
            "arrive_at_pre_compute_diff called on a seam whose \
             park_before_compute_diff slot is None; rendezvous would \
             silently no-op"
        );
        if let Some(p) = self.park_before_compute_diff.as_ref() {
            p.arrive().await;
        }
    }

    /// Test side: block until the subscriber has parked at the pre-compute_diff
    /// point (b.y3a).
    ///
    /// After this returns the subscriber has read `mapping == OLD` *and* the
    /// OLD object's header (P, in `run_catch_up_inner`), so a test may fire its
    /// overwrite and cleanup drain knowing both that a header read already
    /// succeeded (the arm-2 precondition) and that the subscriber cannot be
    /// diverted onto the overwrite-detect branch. Pair with a later
    /// [`Self::arrive_at_pre_compute_diff`], which performs the release.
    pub async fn wait_until_parked_pre_compute_diff(&self) {
        if let Some(p) = self.park_before_compute_diff.as_ref() {
            p.wait_until_parked().await;
        }
    }
}

/// Test-only synchronization seam for the catch-up **state read** (b.vd8).
/// When installed on a `LiveEditService` via
/// `catch_up_state_read_seam_for_test`, the subscriber pauses at up to four
/// distinct points. The first two sit in the gap between the header sequence
/// the catch-up read first (P) and the state read whose result it serves —
/// after the capacity waits for the reader and the slot, before the read
/// permit, the state read and the stream registration
/// ([`catch_up_proven_read`]), so a subscriber parked there holds a reader
/// reservation and queues nothing:
///
///   * **Pre-full-state** — inside `catch_up_brand_new` (Regime 1).
///   * **Pre-coalesced** — inside `catch_up_in_window_snapshot` (Regime 2,
///     default `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true` arm).
///
///   Each parks once per catch-up, before its proven read: the atomic attempt
///   or, above the atomic-read limits, the attach to the object's shared
///   fenced-read flight. Each is a rendezvous the test releases by arriving
///   itself
///   ([`Self::arrive_at_pre_full_state`] / [`Self::arrive_at_pre_coalesced`]);
///   a test that must act while the subscriber is parked — land a commit
///   after P — first awaits
///   [`Self::wait_until_parked_pre_full_state`] /
///   [`Self::wait_until_parked_pre_coalesced`].
///   * **Between replay computes** — inside `catch_up_in_window` (Regime 2,
///     rollback `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false` arm), after its
///     FIRST per-commit read (and the diff built from it) and before the
///     next, with nothing sent. Empty commits are skipped: their stored delta
///     is built with no read and does not trigger the park, so in a window
///     `[s empty, s+1, …]` it parks after the second stored diff. Lets a test
///     land an overwrite and drain its cleanup so a later read answers
///     `NotFound` (b.jvx Site A). A two-phase park
///     ([`Self::wait_until_parked_between_replay_computes`] /
///     [`Self::release_between_replay_computes`]): the subscriber stays
///     parked until the test releases it.
///   * **Pre-ext-probe** (b.yuo) — inside [`build_catch_up`], which every
///     catch-up build goes through: after the catch-up's state read (its own
///     atomic read, or its flight's shared fenced read, in its build turn)
///     and before the build reads any external-value blob, so a test can land
///     a commit that releases a blob the snapshot references and let the GC
///     collect it — the stale-miss race the external-value miss protocol
///     resolves. Parks only the FIRST build to reach it in the seam's life (a
///     rollback-mode replay's first per-commit build that takes a read; the
///     build a re-bootstrap enters passes). A two-phase park
///     ([`Self::wait_until_parked_pre_ext_probe`] /
///     [`Self::release_pre_ext_probe`]).
///
/// Regime 3 already has the equivalent park point on
/// [`CatchUpOutOfWindowSeam::arrive_at_pre_compute_diff`], so the three catch-up
/// regimes are now symmetrically drivable: park the subscriber, commit, release,
/// and assert on what the resulting catch-up acked. A commit landed while parked
/// at the first two points is above P, folded into the snapshot, and at or below
/// the cut the stream registers at — so it must be delivered exactly once, in
/// the snapshot, and acked.
///
/// The four park points sit after the catch-up has taken its
/// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS` slot, so a subscriber parked
/// at any of them holds that slot (unlike one parked at [`CatchUpChunkSeam`],
/// which released it before its first send) — except a brand-new catch-up
/// whose object's shared fenced-read flight was registered when it looked,
/// which takes no slot and so parks at the pre-full-state point holding none,
/// and a catch-up building from its flight's shared snapshot, which parks at
/// the pre-ext-probe point holding its build turn instead.
///
/// Each instance installs exactly one of the four park points — selected by
/// the constructor used — so a test can scope its parked window to the
/// regime under test. Mirrors [`CatchUpOutOfWindowSeam`]'s multi-slot shape,
/// including the 1-party auto-complete barriers in the non-selected slots: the
/// park points live in different functions, but an in-window catch-up that
/// re-bootstraps does so through `catch_up_brand_new` and would otherwise trip the
/// `debug_assert!` safety net on the unselected arrival.
///
/// **Test-only.** Production code never constructs this — it is gated behind
/// `#[cfg(any(test, feature = "test-fault-injection"))]` and the field that
/// holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct CatchUpStateReadSeam {
    park_before_full_state: Option<SeamParkPoint>,
    park_before_coalesced: Option<SeamParkPoint>,
    /// `(parked, released)`: two 2-party barriers when selected, two 1-party
    /// auto-complete barriers otherwise.
    park_between_replay_computes: Option<(tokio::sync::Barrier, tokio::sync::Barrier)>,
    /// The pre-ext-probe point (b.yuo): a 2-party park when selected, a
    /// 1-party auto-complete one otherwise.
    park_before_ext_probe: Option<SeamParkPoint>,
    /// Set by the first build to reach the pre-ext-probe point; every later
    /// build passes it, so a seam selecting it parks exactly one build.
    ext_probe_reached: std::sync::atomic::AtomicBool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CatchUpStateReadSeam {
    /// Build a seam that parks the subscriber inside `catch_up_brand_new`
    /// (Regime 1) after its slot wait, before its read permit and state read.
    ///
    /// The downstream `park_before_coalesced` slot holds a single-party
    /// `Barrier::new(1)` that auto-completes on the production-side arrival —
    /// see the type-level docs for why both slots are always populated.
    pub fn new_before_full_state() -> Self {
        Self {
            park_before_full_state: Some(SeamParkPoint::new(2)),
            park_before_coalesced: Some(SeamParkPoint::new(1)),
            park_between_replay_computes: Some(Self::auto_complete_two_phase()),
            park_before_ext_probe: Some(SeamParkPoint::new(1)),
            ext_probe_reached: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Build a seam that parks the subscriber inside
    /// `catch_up_in_window_snapshot` (Regime 2) after its in-window point
    /// read and its slot
    /// wait, before its read permit and state read.
    ///
    /// The `park_before_full_state` slot holds a single-party `Barrier::new(1)`
    /// that auto-completes on the production-side arrival — see the type-level
    /// docs for why both slots are always populated.
    pub fn new_before_coalesced() -> Self {
        Self {
            park_before_full_state: Some(SeamParkPoint::new(1)),
            park_before_coalesced: Some(SeamParkPoint::new(2)),
            park_between_replay_computes: Some(Self::auto_complete_two_phase()),
            park_before_ext_probe: Some(SeamParkPoint::new(1)),
            ext_probe_reached: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Build a seam that parks the subscriber inside `catch_up_in_window`
    /// (Regime 2, rollback mode) after its FIRST per-commit read (and the diff
    /// built from it), holding the catch-up slot, with nothing sent. Empty
    /// commits are skipped: they take no read and do not trigger the park.
    /// The subscriber stays parked until
    /// [`Self::release_between_replay_computes`].
    ///
    /// The other three slots hold single-party auto-complete barriers — see
    /// the type-level docs — so a re-bootstrap the release leads to passes
    /// `catch_up_brand_new`'s arrival without waiting.
    pub fn new_between_replay_computes() -> Self {
        Self {
            park_before_full_state: Some(SeamParkPoint::new(1)),
            park_before_coalesced: Some(SeamParkPoint::new(1)),
            park_between_replay_computes: Some((
                tokio::sync::Barrier::new(2),
                tokio::sync::Barrier::new(2),
            )),
            park_before_ext_probe: Some(SeamParkPoint::new(1)),
            ext_probe_reached: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Build a seam that parks the FIRST catch-up build to reach
    /// [`build_catch_up`] (b.yuo): after its state read, holding its catch-up
    /// slot (or, on a shared fenced snapshot, its build turn), with its stream
    /// registered, before the build reads any external-value blob. The
    /// subscriber stays parked until [`Self::release_pre_ext_probe`]; every
    /// later build passes.
    ///
    /// The other three slots hold single-party auto-complete barriers — see
    /// the type-level docs.
    pub fn new_before_ext_probe() -> Self {
        Self {
            park_before_full_state: Some(SeamParkPoint::new(1)),
            park_before_coalesced: Some(SeamParkPoint::new(1)),
            park_between_replay_computes: Some(Self::auto_complete_two_phase()),
            park_before_ext_probe: Some(SeamParkPoint::new(2)),
            ext_probe_reached: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn auto_complete_two_phase() -> (tokio::sync::Barrier, tokio::sync::Barrier) {
        (tokio::sync::Barrier::new(1), tokio::sync::Barrier::new(1))
    }

    /// Rendezvous at the Regime 1 pre-full-state park point. Safe to call
    /// unconditionally: every constructor populates the slot (one with the real
    /// 2-party rendezvous, the others with a 1-party auto-complete barrier). The
    /// `debug_assert!` is a fail-loud safety net for any future constructor that
    /// leaves the slot `None` and would silently no-op the rendezvous.
    ///
    /// The subscriber calls it, and so does the test to release the
    /// subscriber: the point is a 2-party rendezvous, so the subscriber stays
    /// parked until the test arrives. A test that must know the subscriber has
    /// parked before it acts — to land a commit after the header read P —
    /// first awaits [`Self::wait_until_parked_pre_full_state`].
    pub async fn arrive_at_pre_full_state(&self) {
        debug_assert!(
            self.park_before_full_state.is_some(),
            "arrive_at_pre_full_state called on a seam whose \
             park_before_full_state slot is None; rendezvous would \
             silently no-op"
        );
        if let Some(p) = self.park_before_full_state.as_ref() {
            p.arrive().await;
        }
    }

    /// Test side: block until the subscriber has parked at the pre-full-state
    /// point — holding its catch-up slot (none when its object's shared
    /// fenced-read flight was registered), with P already read and no read
    /// permit or state read yet. Pair with a later
    /// [`Self::arrive_at_pre_full_state`], which releases it.
    pub async fn wait_until_parked_pre_full_state(&self) {
        if let Some(p) = self.park_before_full_state.as_ref() {
            p.wait_until_parked().await;
        }
    }

    /// Rendezvous at the Regime 2 pre-coalesced park point. See
    /// [`Self::arrive_at_pre_full_state`] for the unconditional-call rationale
    /// and for how a test uses it.
    pub async fn arrive_at_pre_coalesced(&self) {
        debug_assert!(
            self.park_before_coalesced.is_some(),
            "arrive_at_pre_coalesced called on a seam whose \
             park_before_coalesced slot is None; rendezvous would \
             silently no-op"
        );
        if let Some(p) = self.park_before_coalesced.as_ref() {
            p.arrive().await;
        }
    }

    /// Test side: block until the subscriber has parked at the pre-coalesced
    /// point — holding its catch-up slot, with P already read and no read
    /// permit or state read yet. Pair with a later
    /// [`Self::arrive_at_pre_coalesced`], which releases it.
    pub async fn wait_until_parked_pre_coalesced(&self) {
        if let Some(p) = self.park_before_coalesced.as_ref() {
            p.wait_until_parked().await;
        }
    }

    /// Subscriber side of the between-replay-computes park point: announce
    /// the park (rendezvous 1), then block until the test releases it
    /// (rendezvous 2). Called once per `catch_up_in_window` invocation, after
    /// its first per-commit read (an empty commit takes none and does not
    /// count); a no-op wait on a seam built by another constructor.
    /// Each barrier is 2-party when selected, so a second rollback-mode replay
    /// on the same service (a second subscription, or a retry) would hang:
    /// install a fresh seam per replay.
    pub async fn arrive_between_replay_computes(&self) {
        debug_assert!(
            self.park_between_replay_computes.is_some(),
            "arrive_between_replay_computes called on a seam whose \
             park_between_replay_computes slot is None; rendezvous would \
             silently no-op"
        );
        if let Some((parked, released)) = self.park_between_replay_computes.as_ref() {
            let _ = parked.wait().await;
            let _ = released.wait().await;
        }
    }

    /// Test side: block until the subscriber has parked after its first replay
    /// read (rendezvous 1). It then holds its catch-up slot and has sent
    /// nothing. Pair with a later [`Self::release_between_replay_computes`].
    pub async fn wait_until_parked_between_replay_computes(&self) {
        if let Some((parked, _)) = self.park_between_replay_computes.as_ref() {
            let _ = parked.wait().await;
        }
    }

    /// Test side: release the parked subscriber (rendezvous 2).
    pub async fn release_between_replay_computes(&self) {
        if let Some((_, released)) = self.park_between_replay_computes.as_ref() {
            let _ = released.wait().await;
        }
    }

    /// Subscriber side of the pre-ext-probe park point (b.yuo), called by
    /// every catch-up build ([`build_catch_up`]) before its first blob read:
    /// the first call raises the `parked` latch and blocks until the test
    /// releases it (a pass-through on a seam built by another constructor);
    /// every later call returns at once.
    pub async fn arrive_at_pre_ext_probe(&self) {
        debug_assert!(
            self.park_before_ext_probe.is_some(),
            "arrive_at_pre_ext_probe called on a seam whose park_before_ext_probe slot is None; \
             rendezvous would silently no-op"
        );
        if self
            .ext_probe_reached
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        if let Some(p) = self.park_before_ext_probe.as_ref() {
            p.arrive().await;
        }
    }

    /// Test side: block until the first catch-up build has parked at the
    /// pre-ext-probe point, its snapshot read and no blob read yet. Pair with
    /// a later [`Self::release_pre_ext_probe`].
    pub async fn wait_until_parked_pre_ext_probe(&self) {
        if let Some(p) = self.park_before_ext_probe.as_ref() {
            p.wait_until_parked().await;
        }
    }

    /// Test side: release the build parked at the pre-ext-probe point.
    pub async fn release_pre_ext_probe(&self) {
        if let Some(p) = self.park_before_ext_probe.as_ref() {
            let _ = p.barrier.wait().await;
        }
    }
}

/// Which capacity wait a [`BudgetCharge`] records.
///
/// **Test-only.** Gated with [`CapacityBudgetSeam`].
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetWait {
    /// The admission wait, charged once, by the handler, to the budget it has
    /// just created. Admission is bounded by the gate's own wait, not by a
    /// budget deadline: the budget does not exist yet.
    Admission,
    /// A stream-reader reservation ([`reserve_stream`]), made before every
    /// stream registration. It waits only in cluster mode, so on standalone
    /// Valkey its charge is only the time the reservation itself took.
    StreamReader,
    /// A catch-up slot wait ([`acquire_catchup_slot`]).
    CatchUpSlot,
    /// A snapshot read-permit wait ([`acquire_snapshot_read_permit`]), made
    /// inside the catch-up slot before each atomic read.
    ReadPermit,
    /// A catch-up's wait for its shared fenced-read flight's read
    /// ([`read_through_flight`]): charged the catch-up's share of the
    /// flight's slot, fenced-permit and memory waits — the time it was
    /// attached while the flight waited in them — and recorded once, when the
    /// wait returns (read, refused or the flight's read failed). The flight's
    /// own waits have no deadline, so `deadline` is `None`; `charged` can be
    /// less than the time from `started`, since the flight's work and its wait
    /// for a predecessor are not charged.
    FlightRead,
}

/// One charge to a subscription's [`CapacityBudget`], as recorded by
/// [`CapacityBudgetSeam`].
///
/// The fields are the values the charge site itself used, so a test can state
/// the budget arithmetic exactly instead of racing a sleep against a deadline:
///
/// - `budget_before - charged == budget_after` (saturating at zero): the wait
///   was charged what it waited.
/// - A later record's `budget_before` equals the previous record's
///   `budget_after` for the same `client_id`, unless a re-bootstrap onto a new
///   object refilled the budget in between, in which case it equals the
///   budget's full size. This holds only when that `client_id` belongs to one
///   subscription on the service: `validate_client_id` accepts the empty id
///   and does not enforce uniqueness, so a test that reuses a `client_id`, or
///   sends the empty one, must tell the subscriptions apart itself.
/// - For a wait given a deadline, `started + budget_before <= deadline <=
///   started + charged + budget_before`: the wait was bounded by what was
///   left of the budget, not by the full configured wait. Both bounds are
///   exact: the deadline is computed after `started` and before the wait ends
///   (barring a budget too large for the clock, whose deadline is decades
///   away instead).
///
/// **Test-only.** Gated with [`CapacityBudgetSeam`].
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug)]
pub struct BudgetCharge {
    /// The subscribing client's `client_id`, to tell subscriptions on one
    /// service apart — when each uses its own non-empty id (nothing enforces
    /// that; see the chain bullet above).
    pub client_id: String,
    /// Which wait was charged.
    pub wait: BudgetWait,
    /// What was left of the budget when the wait started. For
    /// [`BudgetWait::Admission`] this is the budget's full size
    /// (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`).
    pub budget_before: std::time::Duration,
    /// When the wait started.
    pub started: tokio::time::Instant,
    /// The deadline the wait was given ([`CapacityBudget::deadline`]); `None`
    /// for [`BudgetWait::Admission`] and [`BudgetWait::FlightRead`].
    pub deadline: Option<tokio::time::Instant>,
    /// What the budget was charged: for a reader, slot or read-permit wait,
    /// the time from `started` to the end of the wait; for
    /// [`BudgetWait::FlightRead`], only the catch-up's share of its flight's
    /// capacity waits. Every reader, slot or read-permit wait that the
    /// multiplexer or its gate ended is charged and recorded — capacity found,
    /// the deadline passed, or the multiplexer or gate closed — and every
    /// flight-read wait that returned. A wait ended first by the
    /// subscription's shutdown token, its session's cancellation or its
    /// client's disconnect ends the subscription and records nothing (a
    /// flight-read wait has by then charged its share so far). The admission
    /// wait is recorded only when it admits: a refused subscribe never creates
    /// a budget.
    pub charged: std::time::Duration,
    /// What was left of the budget right after the charge.
    pub budget_after: std::time::Duration,
}

/// Test-only recorder of every [`CapacityBudget`] charge on a
/// `LiveEditService`, in the order the charges were made. Installed via
/// `capacity_budget_seam_for_test`.
///
/// Each charge site — the handler's admission charge, [`reserve_stream`],
/// [`acquire_catchup_slot`], [`acquire_snapshot_read_permit`] and
/// [`read_through_flight`] (a shared fenced-read flight charges the catch-up
/// inside `FlightWaiter::read`; the site records the total when that wait
/// returns) — records one [`BudgetCharge`] right after it charges the budget.
/// Recording only observes: it never changes what is
/// charged, the deadline a wait is given, or how a wait ends. A test that
/// holds a capacity wait to spend budget can therefore assert what each wait
/// was charged and what the next wait was given, instead of timing a release
/// against the next wait's deadline.
///
/// **Test-only.** Production code never constructs this — it is gated behind
/// `#[cfg(any(test, feature = "test-fault-injection"))]` and the field that
/// holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct CapacityBudgetSeam {
    charges: std::sync::Mutex<Vec<BudgetCharge>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CapacityBudgetSeam {
    /// An empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every charge recorded so far, in charge order, across every
    /// subscription on the service.
    pub fn charges(&self) -> Vec<BudgetCharge> {
        self.lock().clone()
    }

    /// The charges recorded so far for `client_id`, in charge order. They are
    /// one subscription's charges only when that `client_id` belongs to one
    /// subscription on the service: `validate_client_id` accepts the empty id
    /// and does not enforce uniqueness, so a test that reuses a `client_id`,
    /// or sends the empty one, gets every such subscription's charges
    /// interleaved and must tell them apart itself.
    pub fn charges_for(&self, client_id: &str) -> Vec<BudgetCharge> {
        self.lock()
            .iter()
            .filter(|charge| charge.client_id == client_id)
            .cloned()
            .collect()
    }

    fn record(&self, charge: BudgetCharge) {
        self.lock().push(charge);
    }

    /// The record list; a panic elsewhere while it was held leaves it intact
    /// (a push is the only mutation), so a poisoned lock is still read.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<BudgetCharge>> {
        self.charges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Record a wait's budget charge on the service's [`CapacityBudgetSeam`], if
/// one is installed. Call right after the wait's charge to `ctx.budget`.
#[cfg(any(test, feature = "test-fault-injection"))]
fn record_budget_charge(
    ctx: &SubscriberCtx<'_>,
    wait: BudgetWait,
    budget_before: std::time::Duration,
    started: tokio::time::Instant,
    deadline: Option<tokio::time::Instant>,
    charged: std::time::Duration,
) {
    if let Some(seam) = ctx.svc.capacity_budget_seam_for_test.as_ref() {
        seam.record(BudgetCharge {
            client_id: ctx.client_id.to_owned(),
            wait,
            budget_before,
            started,
            deadline,
            charged,
            budget_after: ctx.budget.remaining(),
        });
    }
}

/// Test-only park point inside a catch-up's build turn in its shared
/// fenced-read flight ([`take_build_turn`]), installed via
/// `catch_up_build_turn_seam_for_test`.
///
/// A selected catch-up parks once it holds its turn — the flight's own, or an
/// extra build's — and before it builds, holding the turn guard, its snapshot
/// handle and its place in the flight, so the flight's slot and memory grant
/// (and an extra build's own slot and part) stay held and every catch-up
/// queued behind it keeps waiting. That makes a build chain deterministic: a
/// second wave forming behind a parked first, a burst's extra builds taken
/// while the flight's own turn is parked, a catch-up cancelled inside its
/// turn (the park races the subscription's endings, so shutdown, session
/// cancellation or a client disconnect still end it and give the turn back).
///
/// Which catch-ups park is chosen at construction: those of given
/// `client_id`s ([`Self::for_clients`]), or the first `n` to reach a build
/// turn on the service ([`Self::for_first`]). A parked catch-up stays parked
/// until the test releases it ([`Self::release`], [`Self::release_all`]);
/// [`Self::wait_until_parked`] waits, level-triggered, for a number of parks,
/// and [`Self::parked`] lists the catch-ups parked now, with whether each holds
/// an extra build's turn.
///
/// Catch-ups at or below the atomic-read limits build under their own slot,
/// take no build turn and never reach it.
///
/// **Test-only.** Production code never constructs this — it is gated behind
/// `#[cfg(any(test, feature = "test-fault-injection"))]` and the field that
/// holds it on `LiveEditService` is also gated.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct CatchUpBuildTurnSeam {
    select: BuildTurnSelect,
    /// Build turns that reached the seam, selected or not; what
    /// [`BuildTurnSelect::First`] counts against.
    arrivals: std::sync::atomic::AtomicUsize,
    state: tokio::sync::watch::Sender<BuildTurnSeamState>,
}

/// Which catch-ups a [`CatchUpBuildTurnSeam`] parks.
#[cfg(any(test, feature = "test-fault-injection"))]
enum BuildTurnSelect {
    /// Those whose `client_id` is in the set.
    Clients(std::collections::HashSet<String>),
    /// The first `n` build turns to reach the seam.
    First(usize),
}

/// One catch-up parked in its build turn ([`CatchUpBuildTurnSeam::parked`]).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParkedBuildTurn {
    /// The parked catch-up's `client_id`.
    pub client_id: String,
    /// Whether its turn is an extra build's (its own slot and build part)
    /// rather than the flight's own.
    pub extra: bool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
struct BuildTurnSeamState {
    /// Catch-ups parked now, in arrival order.
    parked: Vec<ParkedBuildTurn>,
    /// Parks so far, released or not: what [`CatchUpBuildTurnSeam::wait_until_parked`]
    /// waits on.
    parks: usize,
    /// Every parked catch-up released, now and from here on.
    release_all: bool,
    /// `client_id`s released.
    released: std::collections::HashSet<String>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl CatchUpBuildTurnSeam {
    fn new(select: BuildTurnSelect) -> Self {
        Self {
            select,
            arrivals: std::sync::atomic::AtomicUsize::new(0),
            state: tokio::sync::watch::channel(BuildTurnSeamState::default()).0,
        }
    }

    /// Park the catch-ups with these `client_id`s in their build turns.
    pub fn for_clients<I, S>(client_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(BuildTurnSelect::Clients(
            client_ids.into_iter().map(Into::into).collect(),
        ))
    }

    /// Park the first `n` catch-ups to reach a build turn on the service.
    pub fn for_first(n: usize) -> Self {
        Self::new(BuildTurnSelect::First(n))
    }

    /// Test side: wait until at least `count` catch-ups have parked since the
    /// seam was built (released ones included). Level-triggered: returns at
    /// once if they already have.
    pub async fn wait_until_parked(&self, count: usize) {
        let mut rx = self.state.subscribe();
        // `send_modify` never closes the channel while `self` holds the
        // sender, so this wait ends only on the count.
        let _ = rx.wait_for(|state| state.parks >= count).await;
    }

    /// The catch-ups parked now, in the order they parked.
    pub fn parked(&self) -> Vec<ParkedBuildTurn> {
        self.state.borrow().parked.clone()
    }

    /// Test side: release the catch-up parked (or yet to park) with
    /// `client_id`.
    pub fn release(&self, client_id: &str) {
        self.state.send_modify(|state| {
            state.released.insert(client_id.to_owned());
        });
    }

    /// Test side: release every parked catch-up, and let every later one
    /// through without parking.
    pub fn release_all(&self) {
        self.state.send_modify(|state| state.release_all = true);
    }

    /// Subscriber side: park `client_id`'s catch-up, holding its build turn
    /// (`extra` for an extra build's), if the seam selects it, until the test
    /// releases it. Returns at once for a catch-up it does not select.
    /// Dropping the future (the subscription ended) unparks it.
    pub(crate) async fn park(&self, client_id: &str, extra: bool) {
        let arrival = self
            .arrivals
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let selected = match &self.select {
            BuildTurnSelect::Clients(ids) => ids.contains(client_id),
            BuildTurnSelect::First(n) => arrival < *n,
        };
        if !selected {
            return;
        }
        let released =
            |state: &BuildTurnSeamState| state.release_all || state.released.contains(client_id);
        let mut rx = self.state.subscribe();
        if released(&rx.borrow_and_update()) {
            return;
        }
        let parked = ParkedBuildTurn {
            client_id: client_id.to_owned(),
            extra,
        };
        self.state.send_modify(|state| {
            state.parked.push(parked.clone());
            state.parks += 1;
        });
        let _unpark = UnparkOnDrop {
            state: &self.state,
            parked,
        };
        let _ = rx.wait_for(released).await;
    }
}

/// Takes a parked catch-up off [`BuildTurnSeamState::parked`] when its park
/// ends, released or dropped.
#[cfg(any(test, feature = "test-fault-injection"))]
struct UnparkOnDrop<'a> {
    state: &'a tokio::sync::watch::Sender<BuildTurnSeamState>,
    parked: ParkedBuildTurn,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Drop for UnparkOnDrop<'_> {
    fn drop(&mut self) {
        self.state.send_modify(|state| {
            if let Some(at) = state.parked.iter().position(|p| *p == self.parked) {
                state.parked.remove(at);
            }
        });
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl std::fmt::Debug for CatchUpBuildTurnSeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUpBuildTurnSeam")
            .field("parked", &self.parked())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Catch-up future-size fence (b.9qr, Spec SDD SR-3.5) ──────────────────

    /// Debug builds only: the ceilings are debug-build sizes, and a release
    /// build overlaps inline futures' slots, so its sizes are not comparable.
    #[cfg(debug_assertions)]
    mod future_size_fence {
        use super::*;

        /// The size of the future `f` returns, without calling `f`: only the
        /// future's type is measured, so no service, context or store is needed.
        fn future_size_of<'a, F: std::future::Future>(
            _f: impl FnOnce(&'a SubscriberCtx<'a>) -> F,
        ) -> usize {
            std::mem::size_of::<F>()
        }

        /// A value of any type for an argument of a function whose future is
        /// measured but never created: [`future_size_of`] never calls the closure
        /// this appears in.
        fn never<T>() -> T {
            unreachable!("never called: only the future's type is measured")
        }

        fn measured_catch_up_futures() -> [(&'static str, usize); 5] {
            [
                ("run_catch_up", future_size_of(run_catch_up)),
                (
                    "catch_up_brand_new",
                    future_size_of(|c| catch_up_brand_new(c, never())),
                ),
                (
                    "catch_up_in_window",
                    future_size_of(|c| catch_up_in_window(c, never(), never(), never(), never())),
                ),
                (
                    "catch_up_in_window_snapshot",
                    future_size_of(|c| {
                        catch_up_in_window_snapshot(c, never(), never(), never(), never(), never())
                    }),
                ),
                (
                    "catch_up_out_of_window",
                    future_size_of(|c| {
                        catch_up_out_of_window(c, never(), never(), never(), never(), never())
                    }),
                ),
            ]
        }

        /// The debug-build ceilings
        /// [`run_catch_up_future_stays_under_its_recorded_debug_ceiling`]
        /// enforces, in bytes, beside the size each future measured when the
        /// ceiling was recorded.
        ///
        /// | future | measured | ceiling |
        /// |---|---|---|
        /// | `run_catch_up` | 7,296 | 8,192 |
        /// | `catch_up_brand_new` | 7,304 | 8,192 |
        /// | `catch_up_in_window` | 7,768 | 8,192 |
        /// | `catch_up_in_window_snapshot` | 7,840 | 8,192 |
        /// | `catch_up_out_of_window` | 1,952 | 4,096 |
        ///
        /// Measured on a debug `--lib` test build (`cfg(test)`, so the test
        /// seams' awaits are counted) once every catch-up regime, the
        /// rollback-mode per-commit replay included, took its snapshot from
        /// its proven read — its own atomic read, or above the atomic-read
        /// limits its layer's shared fenced-read flight — with the read-permit
        /// wait, the atomic read, the flight-size probe, the attach and wait
        /// for the flight's read, the wait for a build turn, the build from
        /// the snapshot, the replay's per-commit read and build and the
        /// replay's hand-over to the out-of-window route each boxed where it
        /// is awaited (b.y3m Epic 3 Task 7; before it, with the interim
        /// above-ceiling fallback boxed in the same place, the four regime
        /// futures measured 7,032 / 7,744 / 7,576 / 2,016).
        ///
        /// History: before any boxing `run_catch_up` measured 23,352 bytes,
        /// because it carried every regime inline and each regime carried a
        /// whole `catch_up_brand_new` through its re-bootstrap. Boxing the
        /// regimes at their dispatch in `run_catch_up_inner`, the re-bootstrap
        /// and the recovery attempt brought it to 7,296, with each regime still
        /// 22.2–22.6 KiB from its inline snapshot-load chain; the fence was
        /// first recorded there (24,576-byte regime ceilings), and the
        /// proven-read rework was measured against it before these ceilings
        /// were tightened. `catch_up_in_window` kept its 24,576-byte ceiling
        /// (22,576 measured) while it still carried the per-commit load chain
        /// (the since-deleted `compute_diff`) inline; boxing its per-commit read
        /// brought it to 7,744, and its ceiling was tightened to 8,192.
        ///
        /// Headroom: about 4–12 % on the ~7 KiB futures, room for a few small
        /// locals; 4,096 bytes on
        /// the 2 KiB `catch_up_out_of_window`, so a small local does not trip
        /// it. Every ceiling sits well under the smallest thing that must never
        /// be inlined again: a snapshot-load chain (about 20 KiB) or the
        /// recovery attempt (about 9.5 KiB). A change that trips the fence
        /// boxes the new await (`Box::pin`) rather than raising the ceiling;
        /// raise one only with a measurement, recorded here.
        ///
        /// Why a fence on future size: in a debug build the catch-up path once
        /// needed about 1.7 MiB of the 2 MiB tokio worker stack, so any large
        /// async fn added under it aborted `grpc::subscribe_regimes_test` with
        /// a stack overflow far from the code that caused it (b.9qr). A debug
        /// build does not overlap the slots of inline futures, so their sizes
        /// add up; this fence fails at the code that grew instead.
        const CATCH_UP_FUTURE_CEILINGS: [(&str, usize); 5] = [
            ("run_catch_up", 8_192),
            ("catch_up_brand_new", 8_192),
            ("catch_up_in_window", 8_192),
            ("catch_up_in_window_snapshot", 8_192),
            ("catch_up_out_of_window", 4_096),
        ];

        /// The b.9qr future-size fence (Spec SDD SR-3.5): the `run_catch_up`
        /// future, and each regime future it dispatches to, stay at or under the
        /// ceiling recorded in [`CATCH_UP_FUTURE_CEILINGS`] in a debug build.
        #[test]
        fn run_catch_up_future_stays_under_its_recorded_debug_ceiling() {
            let measured = measured_catch_up_futures();
            eprintln!("catch-up future sizes (debug, bytes): {measured:?}");
            for ((name, size), (ceiling_name, ceiling)) in
                measured.iter().zip(CATCH_UP_FUTURE_CEILINGS.iter())
            {
                assert_eq!(name, ceiling_name, "the two tables list the same futures");
                assert!(
                    size <= ceiling,
                    "the {name} future is {size} bytes in a debug build, above its recorded \
                 {ceiling}-byte ceiling (b.9qr): `Box::pin` the await that grew it rather than \
                 raising the ceiling; see CATCH_UP_FUTURE_CEILINGS"
                );
            }
        }
    }

    // Self-echo predicate tests.

    #[test]
    fn self_echo_matching_client_ids_returns_true() {
        assert!(is_self_echo("client-A", "client-A"));
    }

    #[test]
    fn self_echo_different_client_ids_returns_false() {
        assert!(!is_self_echo("client-A", "client-B"));
    }

    #[test]
    fn self_echo_empty_subscriber_id_returns_false() {
        assert!(!is_self_echo("", "client-B"));
    }

    #[test]
    fn self_echo_empty_notification_id_returns_false() {
        assert!(!is_self_echo("client-A", ""));
    }

    #[test]
    fn self_echo_both_empty_returns_false() {
        assert!(!is_self_echo("", ""));
    }

    #[test]
    fn self_echo_prefix_mismatch_returns_false() {
        assert!(!is_self_echo("client-A", "client-AB"));
    }

    #[test]
    fn self_echo_case_sensitive() {
        assert!(!is_self_echo("Client-A", "client-a"));
    }

    // Chunking helper sanity (used by both catch-up and steady-state paths).

    #[test]
    fn chunk_ompf_sends_header_and_data_chunks() {
        use crate::grpc::proto::subscribe_read_layer_response::ResponseType;

        let bytes = vec![0u8; 2_500_000];
        let header = OmpfStreamHeader {
            object_id: 7,
            sequence_num: 3,
            server_timestamp: 0,
            client_timestamp: 0,
        };
        let msgs: Vec<_> = chunk_ompf_for_stream(&bytes, &header).collect();
        // 1 header + ceil(2.5 MiB / 1 MiB) = 1 + 3 = 4 messages.
        assert_eq!(msgs.len(), 4);
        assert!(matches!(
            msgs[0].response_type.as_ref().unwrap(),
            ResponseType::Header(_)
        ));
        for msg in &msgs[1..] {
            assert!(matches!(
                msg.response_type.as_ref().unwrap(),
                ResponseType::Chunk(_)
            ));
        }
    }

    // ── `send_racing_endings`: which ending wins a send (SDD §12.6) ──────────
    //
    // `send_or_disconnect` is a thin wrapper over this helper, which takes only
    // the channel, the message, the two tokens and the deadline, so its arm
    // order is pinned here behaviourally on a real `mpsc` channel.
    //
    // **Why the ready-at-once cases repeat.** Several cases below make two arms
    // ready on the first poll — a channel with room (so the send completes at
    // once) and a token that has already fired. Under `biased;` the earlier arm
    // always wins; without it tokio picks uniformly at random, so one round
    // would pass half the time. `ROUNDS` rounds turn "passes by luck" into a
    // 2^-`ROUNDS` event, so deleting `biased;` or reordering the arms fails
    // here.

    /// Rounds for every case whose outcome depends on `biased;` arm order.
    const ROUNDS: usize = 64;

    /// A deadline no case here should reach unless it is the slow-reader case.
    const UNREACHED_DEADLINE: std::time::Duration = std::time::Duration::from_secs(3600);

    /// One `send_racing_endings` call on a fresh channel of `capacity`, with
    /// the receiver kept alive, after firing the requested tokens.
    async fn race_one_send(
        capacity: usize,
        shutdown_fired: bool,
        session: Option<bool>,
    ) -> (SendOutcome, mpsc::Receiver<u8>) {
        let (tx, rx) = mpsc::channel::<u8>(capacity);
        let shutdown = CancellationToken::new();
        if shutdown_fired {
            shutdown.cancel();
        }
        let session_token = session.map(|fired| {
            let token = CancellationToken::new();
            if fired {
                token.cancel();
            }
            token
        });
        let outcome = send_racing_endings(
            &tx,
            7,
            &shutdown,
            session_token.as_ref(),
            UNREACHED_DEADLINE,
        )
        .await;
        (outcome, rx)
    }

    /// Nothing has fired and the channel has room: the message is enqueued.
    /// The baseline every other case differs from by one condition.
    #[tokio::test]
    async fn send_racing_endings_with_room_and_no_ending_sends_the_message() {
        for session in [None, Some(false)] {
            let (outcome, mut rx) = race_one_send(4, false, session).await;
            assert_eq!(outcome, SendOutcome::Sent, "session token {session:?}");
            assert_eq!(
                rx.try_recv().ok(),
                Some(7),
                "a `Sent` outcome must mean the message is in the channel"
            );
        }
    }

    /// A fired shutdown token beats a send that could complete at once, and
    /// the message is NOT enqueued: the client must not receive one more
    /// message after the server decided the stream ends `UNAVAILABLE`.
    #[tokio::test]
    async fn send_racing_endings_shutdown_with_room_is_shutting_down_not_sent() {
        for round in 0..ROUNDS {
            let (outcome, mut rx) = race_one_send(4, true, Some(false)).await;
            assert_eq!(
                outcome,
                SendOutcome::ShuttingDown,
                "round {round}: a send into a channel with room won against an already-fired \
                 shutdown token. Graceful shutdown would then be observed only by luck, and a \
                 subscription whose next send is its last would end without the retryable \
                 `UNAVAILABLE` the client needs to reconnect"
            );
            assert!(
                rx.try_recv().is_err(),
                "round {round}: the shutdown arm won, so nothing may have been enqueued"
            );
        }
    }

    /// A fired session token beats a send that could complete at once.
    #[tokio::test]
    async fn send_racing_endings_session_cancel_with_room_is_session_cancelled_not_sent() {
        for round in 0..ROUNDS {
            let (outcome, mut rx) = race_one_send(4, false, Some(true)).await;
            assert_eq!(
                outcome,
                SendOutcome::SessionCancelled,
                "round {round}: a send into a channel with room won against an already-fired \
                 session token, so a deletion would be observed only by luck"
            );
            assert!(
                rx.try_recv().is_err(),
                "round {round}: the session arm won, so nothing may have been enqueued"
            );
        }
    }

    /// Both tokens fired: shutdown wins, so a subscription that sees a
    /// shutdown and a deletion together ends as a shutdown — the ending its
    /// client retries.
    #[tokio::test]
    async fn send_racing_endings_shutdown_and_session_cancel_together_is_shutting_down() {
        for round in 0..ROUNDS {
            let (outcome, _rx) = race_one_send(4, true, Some(true)).await;
            assert_eq!(
                outcome,
                SendOutcome::ShuttingDown,
                "round {round}: with both tokens fired the session arm won. The shutdown arm \
                 must come first, or a replica going away reports a deletion (`NOT_FOUND`, not \
                 retried) to a client whose session still exists"
            );
        }
    }

    /// The receiver is gone: `Disconnected`, whether or not the channel had
    /// room. Neither token has fired.
    #[tokio::test]
    async fn send_racing_endings_with_the_receiver_dropped_is_disconnected() {
        for session in [None, Some(false)] {
            let (tx, rx) = mpsc::channel::<u8>(4);
            drop(rx);
            let shutdown = CancellationToken::new();
            let session_token = session.map(|_| CancellationToken::new());
            let outcome = send_racing_endings(
                &tx,
                7,
                &shutdown,
                session_token.as_ref(),
                UNREACHED_DEADLINE,
            )
            .await;
            assert_eq!(
                outcome,
                SendOutcome::Disconnected,
                "session token {session:?}"
            );
        }
    }

    /// A fired token beats a dropped receiver too: the terminal arms are
    /// ordered shutdown, deletion, disconnect.
    #[tokio::test]
    async fn send_racing_endings_a_fired_token_beats_a_dropped_receiver() {
        for round in 0..ROUNDS {
            for (shutdown_fired, session_fired, want) in [
                (true, false, SendOutcome::ShuttingDown),
                (false, true, SendOutcome::SessionCancelled),
            ] {
                let (tx, rx) = mpsc::channel::<u8>(4);
                drop(rx);
                let shutdown = CancellationToken::new();
                if shutdown_fired {
                    shutdown.cancel();
                }
                let session = CancellationToken::new();
                if session_fired {
                    session.cancel();
                }
                let outcome =
                    send_racing_endings(&tx, 7, &shutdown, Some(&session), UNREACHED_DEADLINE)
                        .await;
                assert_eq!(outcome, want, "round {round}");
            }
        }
    }

    /// A full channel whose reader never drains it, with nothing fired, ends
    /// on the deadline as `SlowReader`. Paused time, so the deadline elapses
    /// without the test waiting for it.
    #[tokio::test(start_paused = true)]
    async fn send_racing_endings_full_channel_past_the_deadline_is_slow_reader() {
        let (tx, _rx) = mpsc::channel::<u8>(1);
        tx.try_send(0)
            .expect("fixture: the one slot takes the first message");
        let shutdown = CancellationToken::new();
        let session = CancellationToken::new();
        let outcome = send_racing_endings(
            &tx,
            7,
            &shutdown,
            Some(&session),
            std::time::Duration::from_millis(50),
        )
        .await;
        assert_eq!(outcome, SendOutcome::SlowReader);
    }

    /// A shutdown that fires while the send is parked on a full channel ends
    /// it at once as `ShuttingDown` — not on the slow-reader deadline, and
    /// not whenever the reader next drains. This is the mid-delivery case: a
    /// client that has stopped reading cannot hold a subscription past a
    /// shutdown.
    #[tokio::test]
    async fn send_racing_endings_shutdown_while_parked_on_a_full_channel_is_shutting_down() {
        let (tx, _rx) = mpsc::channel::<u8>(1);
        tx.try_send(0)
            .expect("fixture: the one slot takes the first message");
        let shutdown = CancellationToken::new();
        let session = CancellationToken::new();
        let race = send_racing_endings(&tx, 7, &shutdown, Some(&session), UNREACHED_DEADLINE);
        tokio::pin!(race);
        assert!(
            futures::poll!(race.as_mut()).is_pending(),
            "fixture: with the channel full and nothing fired, the send must park"
        );
        shutdown.cancel();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), race)
            .await
            .expect("a parked send must observe a shutdown that fires under it");
        assert_eq!(outcome, SendOutcome::ShuttingDown);
    }

    // Confirms the default COUNT bound when no MAXLEN is configured.

    #[test]
    fn soft_cap_is_sensible_default() {
        const _: () = assert!(IN_WINDOW_XRANGE_SOFT_CAP >= 1024);
        const _: () = assert!(IN_WINDOW_XRANGE_SOFT_CAP <= 1_000_000);
    }

    // ── Teardown line decision (SDD §17.2.1) ────────────────────────────────
    //
    // `teardown_event` decides what the subscription task's tail logs. It is
    // pure precisely so the whole table can be pinned here: the integration
    // tests in `tests/integration/grpc/stream_completion_log_test.rs` drive the
    // reachable rows through real subscriptions.

    /// The documented table, restated independently of the implementation.
    fn expected_teardown(is_err: bool, reason: &'static str, caught_up: bool) -> TeardownEvent {
        match (is_err, reason == STREAM_EXIT_CLIENT_DISCONNECT, caught_up) {
            // A status is enqueued for the client: always a completion line.
            (true, false, _) => TeardownEvent::CompletionLine,
            // The client left before its first caught-up ack: logged.
            (true, true, false) => TeardownEvent::CompletionLine,
            // The client left an established stream: silent.
            (true, true, true) => TeardownEvent::Silent,
            // An `Ok` ending is a client disconnect at a loop boundary: silent.
            (false, _, _) => TeardownEvent::Silent,
        }
    }

    /// Every reachable (outcome × exit reason × caught_up) cell agrees with the
    /// documented table: every `Err` cell (11 × 2), and the `Ok` cells, which
    /// only `client_disconnect` reaches.
    ///
    /// Exhaustive over `STREAM_EXIT_REASONS`, the metric's own vocabulary, so
    /// a reason added there is covered here without editing this test. Two
    /// cells worth naming: an `Err` under `shutdown` gets its completion line
    /// whether or not the subscription was caught up — an idle subscriber on a
    /// draining pod is the ending that used to be silent — and so does an
    /// `Err` under `session_absent`, the retryable ending of a session found
    /// missing from the live store.
    #[test]
    fn teardown_event_matches_the_documented_table_for_every_cell() {
        assert_eq!(
            crate::observability::metrics::STREAM_EXIT_REASONS.len(),
            11,
            "precondition: the matrix is sized for the eleven documented exit reasons"
        );
        for &reason in crate::observability::metrics::STREAM_EXIT_REASONS {
            for caught_up in [false, true] {
                for is_err in [false, true] {
                    if !is_err && reason != STREAM_EXIT_CLIENT_DISCONNECT {
                        continue;
                    }
                    let outcome: Result<(), Status> = if is_err {
                        Err(Status::unavailable("teardown matrix"))
                    } else {
                        Ok(())
                    };
                    assert_eq!(
                        teardown_event(&outcome, reason, caught_up),
                        expected_teardown(is_err, reason, caught_up),
                        "teardown disagreement for outcome={} reason={reason} \
                         caught_up={caught_up}",
                        if is_err { "Err" } else { "Ok" }
                    );
                }
            }
        }
    }

    /// The no-status set is exactly `{client_disconnect}`: an `Ok` ending under
    /// any other reason is a broken invariant, not a silent end, and a debug
    /// build says so rather than logging nothing for it. The `Ok` cells the
    /// table test above skips are these.
    #[cfg(debug_assertions)]
    #[test]
    fn teardown_event_debug_asserts_that_only_a_client_disconnect_ends_ok() {
        for &reason in crate::observability::metrics::STREAM_EXIT_REASONS {
            for caught_up in [false, true] {
                let asserted =
                    std::panic::catch_unwind(|| teardown_event(&Ok(()), reason, caught_up))
                        .is_err();
                assert_eq!(
                    asserted,
                    reason != STREAM_EXIT_CLIENT_DISCONNECT,
                    "reason={reason} caught_up={caught_up}: only `client_disconnect` may end a \
                     subscription `Ok`"
                );
            }
        }
    }

    /// Every `Err` except an established client leaving produces a line, so no
    /// status the client receives can go unlogged — whatever its code.
    #[test]
    fn teardown_event_logs_every_err_the_client_can_receive() {
        use tonic::Code;
        for &reason in crate::observability::metrics::STREAM_EXIT_REASONS {
            if reason == STREAM_EXIT_CLIENT_DISCONNECT {
                continue;
            }
            for caught_up in [false, true] {
                for code in [
                    Code::Unavailable,
                    Code::NotFound,
                    Code::Internal,
                    Code::Cancelled,
                ] {
                    assert_eq!(
                        teardown_event(&Err(Status::new(code, "x")), reason, caught_up),
                        TeardownEvent::CompletionLine,
                        "code={code:?} reason={reason} caught_up={caught_up}: a status is \
                         enqueued for the client, so it must be logged"
                    );
                }
            }
        }
    }

    // ── Endings by cause: the statuses and the exit cell ─────────────────────

    /// Each admission refusal maps to the status the client sees: the three
    /// retryable ones `UNAVAILABLE`, a deleted session `NOT_FOUND`. A shutdown
    /// refusal carries exactly the status every other shutdown ending carries,
    /// and a deletion refusal exactly the one the session token produces
    /// mid-stream.
    #[test]
    fn refusal_to_status_maps_shutdown_to_unavailable_and_deletion_to_not_found() {
        let shutting_down = refusal_to_status(AdmissionRefusal::ShuttingDown, None);
        assert_eq!(shutting_down.code(), tonic::Code::Unavailable);
        assert_eq!(shutting_down.message(), server_shutting_down().message());
        assert_eq!(shutting_down.message(), "server shutting down");

        let deleted = refusal_to_status(
            AdmissionRefusal::Cancelled,
            Some(SessionCancelCause::Deleted),
        );
        assert_eq!(
            deleted.code(),
            tonic::Code::NotFound,
            "a deleted session must not be retried, so its refusal is NOT_FOUND, not \
             UNAVAILABLE"
        );
        assert_eq!(deleted.message(), session_deleted().message());
        assert_eq!(deleted.message(), "session was deleted");

        for refusal in [AdmissionRefusal::Saturated, AdmissionRefusal::WaiterCap] {
            assert_eq!(
                refusal_to_status(refusal, None).code(),
                tonic::Code::Unavailable,
                "{refusal:?} is a capacity refusal and must stay retryable"
            );
        }
    }

    /// A status's refusal detail as a client decodes it.
    struct DecodedDetail {
        reason: String,
        domain: String,
        /// `ErrorInfo.metadata`, sorted by key.
        metadata: Vec<(String, String)>,
        /// `RetryInfo.retry_delay`.
        delay: Option<std::time::Duration>,
    }

    /// The `ErrorInfo` a client decodes from `status`, and the `RetryInfo`
    /// delay; `None` when it carries no refusal detail.
    fn decoded_detail(status: &Status) -> Option<DecodedDetail> {
        use tonic_types::StatusExt;
        let info = status.get_details_error_info()?;
        let mut metadata: Vec<(String, String)> = info.metadata.into_iter().collect();
        metadata.sort();
        let delay = status.get_details_retry_info().and_then(|r| r.retry_delay);
        Some(DecodedDetail {
            reason: info.reason,
            domain: info.domain,
            metadata,
            delay,
        })
    }

    /// One refusal the test decodes: what it is, its status, and the reason,
    /// metadata and message it must carry.
    struct RefusalCase {
        what: &'static str,
        status: Status,
        reason: &'static str,
        metadata: Vec<(&'static str, &'static str)>,
        message: &'static str,
    }

    /// Every load-shedding refusal a subscription can meet carries its retry
    /// hint, with its message unchanged: admission saturated and at the
    /// waiter cap (`SUBSCRIPTION_CAPACITY`, `cause`), a reader wait and a
    /// catch-up slot or read-permit wait that outran the capacity budget
    /// (`READER_CAPACITY`; `CATCHUP_CAPACITY`, `wait`), each with a delay in
    /// `[500 ms, 1500 ms)`; and a shutdown refusal (`SERVER_SHUTTING_DOWN`,
    /// under 100 ms).
    #[test]
    fn every_subscribe_capacity_and_shutdown_refusal_carries_its_retry_hint() {
        use crate::error::{CAPACITY_RETRY_DELAY, REFUSAL_DETAIL_DOMAIN, SHUTDOWN_RETRY_DELAY};
        let cases = [
            RefusalCase {
                what: "admission saturated",
                status: refusal_to_status(AdmissionRefusal::Saturated, None),
                reason: "SUBSCRIPTION_CAPACITY",
                metadata: vec![("cause", "saturated")],
                message: "this replica is at its concurrent-subscription ceiling (its memory \
                          budget, sized from the pod's memory limit); retry",
            },
            RefusalCase {
                what: "admission waiter cap",
                status: refusal_to_status(AdmissionRefusal::WaiterCap, None),
                reason: "SUBSCRIPTION_CAPACITY",
                metadata: vec![("cause", "waiter_cap")],
                message: "this replica's subscription admission queue is full \
                          (LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS); retry",
            },
            RefusalCase {
                what: "reader capacity",
                status: reader_capacity_refused(),
                reason: "READER_CAPACITY",
                metadata: vec![],
                message: "this replica is serving as many distinct layers as it has stream \
                          readers (LIVE_EDIT_SUBSCRIBER_POOL_SIZE); retry",
            },
            RefusalCase {
                what: "catch-up slot",
                status: catchup_capacity_refused(),
                reason: "CATCHUP_CAPACITY",
                metadata: vec![("wait", "slot")],
                message: "this replica is at its concurrent catch-up limit \
                          (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS); retry",
            },
            RefusalCase {
                what: "catch-up read permit",
                status: snapshot_read_capacity_refused(),
                reason: "CATCHUP_CAPACITY",
                metadata: vec![("wait", "read_permit")],
                message: "this replica is at its concurrent catch-up snapshot-read limit \
                          (LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY); retry",
            },
            RefusalCase {
                what: "admission shutdown",
                status: refusal_to_status(AdmissionRefusal::ShuttingDown, None),
                reason: "SERVER_SHUTTING_DOWN",
                metadata: vec![],
                message: "server shutting down",
            },
        ];
        for case in cases {
            let what = case.what;
            let status = &case.status;
            assert_eq!(status.code(), tonic::Code::Unavailable, "{what}");
            assert_eq!(
                status.message(),
                case.message,
                "{what}: the message is unchanged"
            );
            let got = decoded_detail(status)
                .unwrap_or_else(|| panic!("{what}: no refusal detail on {status:?}"));
            assert_eq!(got.reason, case.reason, "{what}");
            assert_eq!(got.domain, REFUSAL_DETAIL_DOMAIN, "{what}");
            assert_eq!(
                got.metadata,
                case.metadata
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect::<Vec<_>>(),
                "{what}"
            );
            let range = if case.reason == "SERVER_SHUTTING_DOWN" {
                SHUTDOWN_RETRY_DELAY
            } else {
                CAPACITY_RETRY_DELAY
            };
            let delay = got
                .delay
                .unwrap_or_else(|| panic!("{what}: no RetryInfo delay"));
            assert!(
                range.min <= delay && delay < range.max,
                "{what}: retry_delay {delay:?} outside {range:?}"
            );
        }
    }

    /// The endings that are not load-shedding refusals carry no refusal
    /// detail: a session missing from the live store (`UNAVAILABLE`, which a
    /// recovery may answer) and a deleted session (`NOT_FOUND`).
    #[test]
    fn session_absent_and_session_deleted_refusals_carry_no_refusal_detail() {
        for (what, status) in [
            (
                "session absent",
                refusal_to_status(
                    AdmissionRefusal::Cancelled,
                    Some(SessionCancelCause::AbsentFromStore),
                ),
            ),
            (
                "session deleted",
                refusal_to_status(
                    AdmissionRefusal::Cancelled,
                    Some(SessionCancelCause::Deleted),
                ),
            ),
            ("session absent status", session_absent_status()),
        ] {
            assert!(
                decoded_detail(&status).is_none() && status.details().is_empty(),
                "{what} must carry no refusal detail; got {status:?}"
            );
        }
    }

    /// The catch-up's fault endings are not load-shedding refusals and carry
    /// no refusal detail (no retry hint): `read_hold_refused` (the `{gc}` read
    /// hold could not be taken or kept across a re-read after an
    /// external-value miss), `sequence_regressed` and `fenced_read_wait_cap`.
    /// Each is a retryable `UNAVAILABLE`.
    #[test]
    fn catch_up_fault_endings_carry_no_refusal_detail() {
        for (what, status) in [
            ("read_hold_refused", read_hold_refused()),
            ("sequence_regressed", sequence_regressed()),
            ("fenced_read_wait_cap", fenced_read_wait_refused()),
        ] {
            assert_eq!(status.code(), tonic::Code::Unavailable, "{what}");
            assert!(
                decoded_detail(&status).is_none() && status.details().is_empty(),
                "{what} must carry no refusal detail; got {status:?}"
            );
        }
    }

    /// A session token cancelled because the session went missing from the
    /// live store with no deletion marker refuses admission with the same
    /// retryable status an established subscription ends with — never the
    /// deletion's `NOT_FOUND`, which the client does not retry and which would
    /// lose a session a checkpoint could restore.
    #[test]
    fn refusal_to_status_maps_a_session_absent_from_the_store_to_unavailable() {
        let absent = refusal_to_status(
            AdmissionRefusal::Cancelled,
            Some(SessionCancelCause::AbsentFromStore),
        );
        assert_eq!(
            absent.code(),
            tonic::Code::Unavailable,
            "a session missing from the live store must be refused retryably, so the retry's \
             setup can recover it"
        );
        assert_eq!(absent.message(), session_absent_status().message());
        assert_eq!(
            absent.message(),
            "session is missing from the live store; retry so it can be recovered"
        );
    }

    /// `session_cancel_status` by cause: an explicit deletion and a token with
    /// no cause read `NOT_FOUND "session was deleted"`; a session absent from
    /// the store reads the retryable `UNAVAILABLE`. Every phase of the
    /// subscription ends through this mapping, so it is the one place the two
    /// causes can be crossed.
    #[test]
    fn session_cancel_status_maps_each_cause_to_its_ending() {
        for cause in [None, Some(SessionCancelCause::Deleted)] {
            let status = session_cancel_status(cause);
            assert_eq!(status.code(), tonic::Code::NotFound, "cause {cause:?}");
            assert_eq!(status.message(), "session was deleted", "cause {cause:?}");
        }
        let absent = session_cancel_status(Some(SessionCancelCause::AbsentFromStore));
        assert_eq!(absent.code(), tonic::Code::Unavailable);
        assert_eq!(absent.message(), session_absent_status().message());
    }

    /// Every exit reason survives a round trip through the cell: a reason the
    /// cell could not store would be reported as `error`, and
    /// `stream_exits_total{reason}` would lose it.
    #[test]
    fn stream_exit_cell_round_trips_every_exit_reason() {
        for &reason in crate::observability::metrics::STREAM_EXIT_REASONS {
            let cell = StreamExitCell::new();
            cell.set_reason(reason);
            assert_eq!(cell.get(), reason);
        }
        let cell = StreamExitCell::new();
        cell.set(StreamExitCell::SHUTDOWN);
        assert_eq!(cell.get(), STREAM_EXIT_SHUTDOWN);
        let cell = StreamExitCell::new();
        cell.set(StreamExitCell::SESSION_ABSENT);
        assert_eq!(cell.get(), STREAM_EXIT_SESSION_ABSENT);
    }

    // ── Outcome cells ────────────────────────────────────────────────────────

    /// Fresh cells put nothing on the line: no object, no regime, not caught
    /// up. An absent field means "never reached", which is the truth for a
    /// subscription that failed before its mapping resolved.
    #[test]
    fn outcome_cells_fresh_record_nothing_and_are_not_caught_up() {
        let cells = SubscriptionOutcomeCells::new();
        let mut log = RpcLogContext::new("SubscribeReadLayer");
        cells.record_into(&mut log);
        assert!(!cells.caught_up());
        assert_eq!(cells.regime_label(), None);
        assert_eq!(
            format!("{log:?}"),
            format!("{:?}", RpcLogContext::new("SubscribeReadLayer")),
            "unset cells must leave the context untouched"
        );
    }

    /// Each regime constant maps to its `live_subscribe_regime_total{regime}`
    /// label, and the last one set wins — the branch that was running when the
    /// subscription ended.
    #[test]
    fn outcome_cells_regime_labels_match_the_regime_metric_vocabulary_and_last_set_wins() {
        let cases = [
            (SubscriptionOutcomeCells::REGIME_BRAND_NEW, "brand_new"),
            (SubscriptionOutcomeCells::REGIME_UP_TO_DATE, "up_to_date"),
            (SubscriptionOutcomeCells::REGIME_IN_WINDOW, "in_window"),
            (
                SubscriptionOutcomeCells::REGIME_OUT_OF_WINDOW,
                "out_of_window",
            ),
        ];
        let cells = SubscriptionOutcomeCells::new();
        for (regime, label) in cases {
            cells.set_regime(regime);
            assert_eq!(cells.regime_label(), Some(label));
        }
        // An in-window catch-up that re-bootstraps onto a new object ends in
        // `catch_up_brand_new`: the line must say so.
        cells.set_regime(SubscriptionOutcomeCells::REGIME_IN_WINDOW);
        cells.set_regime(SubscriptionOutcomeCells::REGIME_BRAND_NEW);
        assert_eq!(cells.regime_label(), Some("brand_new"));
    }

    /// `record_into` copies the object id into `current_object_id` (never the
    /// submitted `object_id`) and the regime label, and leaves `caught_up` to
    /// the tail.
    #[test]
    fn outcome_cells_record_the_object_and_regime_into_the_context() {
        let cells = SubscriptionOutcomeCells::new();
        cells.set_object_id(41);
        cells.set_object_id(42);
        cells.set_regime(SubscriptionOutcomeCells::REGIME_OUT_OF_WINDOW);
        cells.set_caught_up();

        let mut expected = RpcLogContext::new("SubscribeReadLayer");
        expected.resolved_object_id(42);
        expected.regime("out_of_window");

        let mut log = RpcLogContext::new("SubscribeReadLayer");
        cells.record_into(&mut log);
        assert_eq!(format!("{log:?}"), format!("{expected:?}"));
        assert!(cells.caught_up());
    }

    /// Counts a renewal's lifetime for [`renewing_while`]'s tests: started on
    /// creation, in flight until dropped (resolved or abandoned).
    struct RenewalProbe {
        started: std::sync::atomic::AtomicUsize,
        in_flight: std::sync::atomic::AtomicUsize,
        max_in_flight: std::sync::atomic::AtomicUsize,
    }

    impl RenewalProbe {
        fn new() -> Self {
            Self {
                started: std::sync::atomic::AtomicUsize::new(0),
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                max_in_flight: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        /// A renewal that answers `answer` after `takes`, or never when
        /// `takes` is `None`, counted in flight until it is dropped.
        fn renewal(
            &self,
            takes: Option<std::time::Duration>,
            answer: ReadHoldRenewal,
        ) -> impl std::future::Future<Output = Result<ReadHoldRenewal, ServiceError>> + '_ {
            use std::sync::atomic::Ordering;
            self.started.fetch_add(1, Ordering::SeqCst);
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            let in_flight = scopeguard::guard(&self.in_flight, |n| {
                n.fetch_sub(1, Ordering::SeqCst);
            });
            async move {
                let _in_flight = in_flight;
                match takes {
                    Some(takes) => tokio::time::sleep(takes).await,
                    None => std::future::pending::<()>().await,
                }
                Ok(answer)
            }
        }

        fn started(&self) -> usize {
            self.started.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn in_flight(&self) -> usize {
            self.in_flight.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn max_in_flight(&self) -> usize {
            self.max_in_flight.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    /// `work` keeps running while a renewal is in flight: a renewal the store
    /// never answers does not stop `work` from completing, no second renewal
    /// starts beside it, and the unanswered one is dropped when `work` ends.
    /// Awaiting the renewal in place of polling `work` never completes.
    #[tokio::test(start_paused = true)]
    async fn renewing_while_keeps_polling_work_while_a_renewal_is_in_flight() {
        let probe = RenewalProbe::new();
        let every = std::time::Duration::from_secs(1);
        let work = async {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            "built"
        };
        let ended = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            renewing_while(every, || probe.renewal(None, ReadHoldRenewal::Live), work),
        )
        .await
        .expect("work must complete while the first renewal is still unanswered");
        assert!(matches!(ended, Ok("built")), "work's output is returned");
        assert_eq!(probe.started(), 1, "no renewal starts beside one in flight");
        assert_eq!(probe.in_flight(), 0, "the unanswered renewal is dropped");
    }

    /// A slow renewal is followed by the next one only once it resolves, at
    /// once if the tick fell due meanwhile; live renewals keep `work` going.
    #[tokio::test(start_paused = true)]
    async fn renewing_while_runs_one_renewal_at_a_time_on_the_interval() {
        let probe = RenewalProbe::new();
        // Renewals start at 1 s and (due ticks delayed) 3.5 s, each taking
        // 2.5 s; `work` ends at 5 s with the second still in flight.
        let ended = renewing_while(
            std::time::Duration::from_secs(1),
            || {
                probe.renewal(
                    Some(std::time::Duration::from_millis(2500)),
                    ReadHoldRenewal::Live,
                )
            },
            tokio::time::sleep(std::time::Duration::from_secs(5)),
        )
        .await;
        assert!(ended.is_ok());
        assert_eq!(probe.started(), 2);
        assert_eq!(probe.max_in_flight(), 1);
        assert_eq!(probe.in_flight(), 0);
    }

    /// A renewal answering LAPSED ends the loop `Err(None)`, and a failed one
    /// `Err(Some(error))`, before `work` completes.
    #[tokio::test(start_paused = true)]
    async fn renewing_while_ends_on_a_lapsed_or_failed_renewal() {
        let probe = RenewalProbe::new();
        let lapsed = renewing_while(
            std::time::Duration::from_secs(1),
            || {
                probe.renewal(
                    Some(std::time::Duration::from_millis(100)),
                    ReadHoldRenewal::Lapsed,
                )
            },
            std::future::pending::<()>(),
        )
        .await;
        assert!(matches!(lapsed, Err(None)), "{lapsed:?}");

        let failed = renewing_while(
            std::time::Duration::from_secs(1),
            || async { Err(ServiceError::Unavailable("store down".into())) },
            std::future::pending::<()>(),
        )
        .await;
        assert!(
            matches!(failed, Err(Some(ServiceError::Unavailable(_)))),
            "{failed:?}"
        );
    }
}
