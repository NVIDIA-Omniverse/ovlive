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

// Completion logging for every RPC on both gRPC servers (SDD §17.2.1).
//
// **The invariant: every non-OK completion emits exactly one completion line,
// from exactly one place.** A failure used to be visible only as a `+1` on
// `live_endpoint_calls_total{grpc_code=…}` — no `session_id`, no `layer_name`,
// no message — and support had to infer the cause from surrounding lines.
//
// There is no single point that sees every completion, so each completion
// point reports through the one vocabulary in this module:
//
// - **Handler boundary.** Each single-response handler splits into a thin
//   wrapper plus a private `*_inner` threading an [`RpcLogContext`]; the
//   wrapper calls [`log_rpc_failure`] on `Err`. An early return added to a
//   handler therefore cannot forget to log. The two streaming handlers use the
//   same wrapper for their setup phase.
// - **Streaming tails.** `SubscribeReadLayer` and `JoinSession` decide their
//   final status in spawned tasks, after the wrapper has returned. Each tail
//   logs at the point its status is decided, from a clone of the setup context
//   advanced to [`RpcStage::Stream`].
// - **The auth interceptor** rejects before any handler runs.
// - **`super::metrics_layer`**, the outermost layer this code owns, keeps a
//   per-request [`RpcCompletion`] ledger. Every emission here marks it
//   reported; the layer's backstop logs, at [`RpcStage::Transport`], a non-OK
//   status that nothing marked (tonic's decode, message-size and unknown-method
//   rejections), and its drop guard logs [`RPC_ABANDONED_MSG`] for a call torn
//   down before any response existed. The server-side deadline keeps its own
//   `RPC deadline exceeded` line there.
//
// A server fault is the one ERROR: `From<ServiceError> for Status`
// (`src/error.rs`) attaches the un-redacted detail to the `Status` as an
// [`InternalFault`] source, and [`log_rpc_failure`] emits it as `internal error`
// with the request fields. A fault no completion point reports is logged by
// the fault's own `Drop`, without context, so it is never lost.
//
// ## Why the level is chosen structurally, not by message match
//
// The interesting split is between a *client* mistake and a *server-side*
// keyspace-damage sentinel that happen to share a gRPC code. `NOT_FOUND` at
// mapping-resolution time means "you named a layer that does not exist"
// (client, info); `NOT_FOUND` raised by the commit path against an object
// whose mapping already resolved means the header was wiped underneath a live
// object (server, warn). Same for `FAILED_PRECONDITION`: an `object_id`
// mismatch is a stale client (info), while the partial-header / superseded
// refusals from the commit path are keyspace damage (warn).
//
// The discriminator is therefore the pair (`stage` reached, was the object
// proven live) — NOT the status message. Two reasons, both load-bearing:
//
// 1. The sentinel strings live in `src/storage` and are deliberately
//    module-private, so this module could not match on them even if it wanted
//    to.
// 2. `src/grpc/layer.rs` records at length (see the post-commit NotFound
//    recovery arm) why this codebase does not key behaviour on those messages:
//    the deleted arm strict-equality matched one, and was one cosmetic edit
//    away from silently disarming itself.

use std::error::Error as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tonic::Code;
use tracing::Level;

use crate::error::{InternalFault, INTERNAL_ERROR_MSG};
use crate::observability::metrics::{
    UPDATE_LAYER_CONFLICT_CACHE_DIVERGED, UPDATE_LAYER_CONFLICT_SEQ_FENCE,
    UPDATE_LAYER_CONFLICT_SNAPSHOT_FENCE_WAIT, UPDATE_LAYER_CONFLICT_WATCH_CONFLICT,
    UPDATE_LAYER_CONFLICT_WATCH_READ_MOVED,
};
use crate::storage::layer::UpdateLayerCallSummary;

/// How far into its flow the request got before it failed.
///
/// Half of the level discriminator (the other half is whether an `object_id`
/// was ever resolved). Also emitted as its own field so an operator can tell a
/// request that died at ingress from one that died against storage without
/// parsing the message.
///
/// The variants are ordered by the flow every handler follows; a handler that
/// has no work for a given stage simply never enters it (session RPCs run
/// `Header` → `Auth` → `Storage`; `SubscribeReadLayer` runs `Header` → `Auth`
/// → `Admission` → `Stream`; `JoinSession` runs `Header` → `Auth` →
/// `Admission` → `Stream`). The two uploads enter `Resolve` out of that
/// order: `UpdateLayer` runs `Header` → `Auth` → `Resolve` → `Admission` →
/// `Body` → `Chain` (a chained call only) → `Parse` → `Storage`, and
/// `CreateLayer` runs `Header` → `Auth` →
/// `Admission` → `Body`, then `Resolve` (its operation-record check, with an
/// operation id) → `Auth` (userinfo) → `Parse` → `Storage`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RpcStage {
    /// Field ingest — for the uploads, the header frame — and the input
    /// validation that precedes any storage or auth work. The default: a
    /// request that fails before the handler records anything is reported
    /// here, with no identifiers. Auth-interceptor rejections are reported
    /// here too, and so are an upload stream that ended before its first
    /// message without END_STREAM (`CANCELLED`) and an upload request with no
    /// end-of-stream record (`INTERNAL`).
    #[default]
    Header,
    /// Session `base_path` resolution, permission check, userinfo resolution.
    Auth,
    /// An admission gate between auth and the stream: `SubscribeReadLayer`'s
    /// subscription admission (the subscription ceiling),
    /// `JoinSession`'s service-shutdown refusal (SDD SR-4.7), and the upload
    /// admission of `CreateLayer` / `UpdateLayer` (the upload memory budget),
    /// both derived from the pod memory limit.
    Admission,
    /// Layer-mapping resolution to an `object_id`, and the `object_id`
    /// precondition check that follows it.
    Resolve,
    /// An admitted upload reading its content: the chunk-count and size caps
    /// (`INVALID_ARGUMENT`), a second header (`INVALID_ARGUMENT`), a receive
    /// error or a stream that ended without END_STREAM — reset, lost or ended
    /// early, so nothing is applied (`CANCELLED`) — and a refused receive
    /// growth of the upload's reservation (`UNAVAILABLE`).
    Body,
    /// A chained `UpdateLayer` (`UpdateLayerRequest.chained_header`) waiting
    /// in the chain gate (`storage::chain_gate`) for its predecessor: its
    /// `ABORTED` predecessor refusal (`no_wait`, or at the deadline cap), the
    /// `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` and shutdown refusals, and the
    /// `FAILED_PRECONDITION` / `NOT_FOUND` of a layer replaced or deleted
    /// while it waited. An unchained call never enters it.
    Chain,
    /// OMPF/OODF parse, the per-delta checks that follow it (§9.4 caps, name
    /// characters, sample times, self-parenting; `INVALID_ARGUMENT`, after the
    /// first processing step's parse charge and before the second's processing
    /// charge), the upload's two processing steps (a shutdown while one waits
    /// ends here), the external-value release-budget wait (b.yuo) and the
    /// external-value store that precede the commit.
    Parse,
    /// The primary storage operation against the resolved object.
    Storage,
    /// The final status of a streaming RPC, decided after its response stream
    /// opened.
    Stream,
    /// A non-OK status that no handler reported: tonic's own decode,
    /// message-size and unknown-method rejections, or a response-encode
    /// failure. Emitted only by the `super::metrics_layer` backstop.
    Transport,
}

impl RpcStage {
    /// Stable lowercase label for the `stage` log field.
    pub(crate) fn label(self) -> &'static str {
        match self {
            RpcStage::Header => "header",
            RpcStage::Auth => "auth",
            RpcStage::Admission => "admission",
            RpcStage::Resolve => "resolve",
            RpcStage::Body => "body",
            RpcStage::Chain => "chain",
            RpcStage::Parse => "parse",
            RpcStage::Storage => "storage",
            RpcStage::Stream => "stream",
            RpcStage::Transport => "transport",
        }
    }
}

/// Per-request completion ledger, carried as an `Arc<RpcCompletion>` request
/// extension.
///
/// `super::metrics_layer` inserts it before calling the inner service, so it
/// travels through tonic's interceptor into every handler's request
/// extensions. Every emission of [`log_rpc_failure`] whose context carries it
/// marks it reported; the metrics layer's backstop logs a non-OK status only
/// while it is still unreported. That is what makes the backstop exactly the
/// complement of the other completion points, rather than a second line.
#[derive(Debug)]
pub(crate) struct RpcCompletion {
    endpoint: &'static str,
    reported: AtomicBool,
}

impl RpcCompletion {
    /// `endpoint` is the metrics layer's bucketed method name — one of
    /// [`super::metrics_layer::KNOWN_ENDPOINTS`] or
    /// [`super::metrics_layer::OTHER_ENDPOINT`].
    pub(crate) fn new(endpoint: &'static str) -> Self {
        Self {
            endpoint,
            reported: AtomicBool::new(false),
        }
    }

    pub(crate) fn endpoint(&self) -> &'static str {
        self.endpoint
    }

    /// Release pairs with the Acquire in [`Self::is_reported`]: a streaming
    /// tail marks the ledger on its own task before enqueueing the status the
    /// metrics layer later reads off the trailer, on another thread.
    pub(crate) fn mark_reported(&self) {
        self.reported.store(true, Ordering::Release);
    }

    pub(crate) fn is_reported(&self) -> bool {
        self.reported.load(Ordering::Acquire)
    }
}

/// Request identifiers accumulated as a handler progresses, read once at a
/// completion point if the RPC fails.
///
/// **Every field is populated only after the corresponding validator has
/// returned `Ok`.** That ordering is load-bearing: `session_id`, `layer_name`,
/// `client_id` and `operation_id` are attacker-controlled until validated, and
/// an unvalidated 64 KB `session_id` carrying ANSI escapes once reached tracing
/// output.
/// Recording before validation would re-open exactly that hole on this sink,
/// so the populate calls sit *below* their validators at every site. The
/// server-minted and closed-vocabulary fields (`connection_id`, `regime`,
/// `reason`, `caught_up`, `rejection`, the `update_layer` call summary) need
/// no validator.
///
/// No bearer material is ever placed here — [`BearerToken`](super::BearerToken)
/// is not a field, and the `PermissionDenied` / `Unauthenticated` statuses
/// carry an empty message by construction (`src/error.rs`).
///
/// `Clone` so a streaming handler can hand its setup context to the spawned
/// task that decides the stream's final status.
#[derive(Clone, Debug, Default)]
pub(crate) struct RpcLogContext {
    /// RPC method name; one of [`super::metrics_layer::KNOWN_ENDPOINTS`], or
    /// [`super::metrics_layer::OTHER_ENDPOINT`] on the transport backstop.
    /// Emitted under the field name `endpoint`, deliberately matching both the
    /// `live_endpoint_calls_total{endpoint}` metric label and the deadline
    /// warn in [`super::metrics_layer`] — field name *and* value join, so a
    /// log line and a metric series need no translation in either direction.
    endpoint: &'static str,
    stage: RpcStage,
    session_id: Option<String>,
    layer_name: Option<String>,
    base_path: Option<String>,
    /// The object id the client submitted (`UpdateLayer`, and a non-zero
    /// `SubscribeReadLayer` `have_object_id`). `CreateLayer` never records one
    /// — it submits no id; the one it mints is recorded as
    /// `current_object_id` when a failure after its commit point names
    /// it (see the comment at that site in `super::layer`).
    object_id: Option<u64>,
    /// The object id the layer mapping actually resolved to — recorded by
    /// `UpdateLayer` and `StatLayer` once the mapping resolves, and by a
    /// `SubscribeReadLayer` tail for the object it was serving, and by
    /// `CreateLayer` for the object its commit point made the layer when a
    /// later step then failed. Never recorded by `DeleteLayer` / the session
    /// RPCs. `Some(..)` is exactly the claim "this object was proven live
    /// during this RPC", which is what separates a stale-client rejection from
    /// keyspace damage.
    current_object_id: Option<u64>,
    client_id: Option<String>,
    /// `UpdateLayer` / `CreateLayer`: the client-supplied `operation_id`, when
    /// the header carried a non-empty one. Not sensitive: it names a write
    /// intent, not a credential.
    operation_id: Option<String>,
    /// `UpdateLayer`: the call's first message was the `chained_header` arm.
    /// Server-derived from the oneof arm, not from any field's content, so it
    /// is recorded before the header's validators run: every
    /// `INVALID_ARGUMENT` a chained call is answered with — its header
    /// validation's included — passes `super::layer`'s `INVALID_UPDATE`
    /// choke point. Emitted as `chained = true` on every chained call's line,
    /// one refused by its header's or the chain's validation included (it
    /// then carries no `previous_operation_id` or `no_wait`); absent on an
    /// unchained call's.
    chained: bool,
    /// `UpdateLayer`, a chained call: its `previous_operation_id`, empty for
    /// the first write of a chain. Emitted as `previous_operation_id`. Not
    /// sensitive, as `operation_id`.
    previous_operation_id: Option<String>,
    /// `UpdateLayer`, a chained call: its `no_wait`. Emitted as `no_wait`.
    no_wait: Option<bool>,
    /// `CreateLayer` / `UpdateLayer`: the call reached its commit — set
    /// immediately before the handler hands the write to
    /// `LayerStore::create_layer_with_operation` /
    /// `LayerStore::update_layer_with_summary` ([`Self::commit_started`]).
    /// False from [`Self::for_request`] until then, and never reset. Read by
    /// `super::layer`'s `UPSTREAM_UNAVAILABLE` choke point: a plain
    /// `UNAVAILABLE` raised before it is a refusal that applied nothing. Not
    /// emitted: the line's `stage` and the `update_layer_call` summary
    /// already say how far the call got.
    commit_started: bool,
    /// `UpdateLayer`: the commit attempts of every `LayerStore::update_layer`
    /// call the RPC made, summed, and whether any of those calls made more
    /// than one — the counted retry after which a chained call's refusal is a
    /// plain `UNAVAILABLE`, outcome unknown
    /// ([`Self::update_layer_counted_retry`]). Not emitted: the
    /// `update_layer_call` summary's `attempt` is the line's attempt count.
    update_layer_attempts: u32,
    update_layer_retried: bool,
    /// `SubscribeReadLayer`: the catch-up branch that was running when the
    /// subscription ended, in the `live_subscribe_regime_total{regime}`
    /// vocabulary.
    regime: Option<&'static str>,
    /// `SubscribeReadLayer`: the stream-exit classification, in the
    /// `live_edit_subscriber_stream_exits_total{reason}` vocabulary. Emitted
    /// as `reason`, matching that label's name.
    exit_reason: Option<&'static str>,
    /// `SubscribeReadLayer`: whether the first caught-up ack was enqueued
    /// before the subscription ended.
    caught_up: Option<bool>,
    /// `JoinSession`: the server-minted connection id.
    connection_id: Option<String>,
    /// Auth interceptor: which header check refused the request
    /// (`super::auth_interceptor::AuthRejection`).
    rejection: Option<&'static str>,
    /// `UpdateLayer`: what its last `LayerStore::update_layer` call did — its
    /// attempts, commit path, conflicts by kind and commit size. Emitted as
    /// the same fields a landed commit's log line carries (`attempt`, `path`,
    /// one count per conflict kind, `records`, `bytes`), so a failed write
    /// joins to the commit lines by those fields and, having no sequence
    /// number, by the session, layer and client ids beside them. Absent when
    /// the RPC failed before calling the store or the call made no attempt.
    update_layer_call: Option<UpdateLayerCallSummary>,
    /// The RPC ended — or was refused — because the service is shutting down
    /// gracefully. Set only by the shutdown arms of the two streaming RPCs,
    /// their two shutdown refusals, and an upload's shutdown refusal at
    /// admission, while a processing step waits, at the external-value release
    /// budget, or while a chained `UpdateLayer` waits in the chain gate (see
    /// [`Self::shutdown_caused`]); read only by
    /// [`failure_level`], which logs
    /// the `UNAVAILABLE` those arms produce at INFO. Not emitted: the
    /// `SubscribeReadLayer` line already carries `reason=shutdown`, and the
    /// level is the whole effect.
    shutdown_caused: bool,
    /// The `JoinSession` ended because its client opted in to keepalives and
    /// then sent no data for three intervals (see
    /// [`Self::client_keepalive_timed_out`]); read only by [`failure_level`],
    /// which logs that `UNAVAILABLE` at INFO. Not emitted: the line's
    /// `status_message` names the cause, and the level is the whole effect.
    client_keepalive_timeout: bool,
    /// The request's completion ledger, when the request came through
    /// `super::metrics_layer`. Not emitted.
    completion: Option<Arc<RpcCompletion>>,
}

impl RpcLogContext {
    /// A context with no ledger, for unit tests. Test-only because a
    /// production context built this way would never mark the request's
    /// ledger reported, and the metrics layer's backstop would log the same
    /// failure a second time; production uses [`Self::for_request`].
    #[cfg(test)]
    pub(crate) fn new(rpc: &'static str) -> Self {
        Self {
            endpoint: rpc,
            ..Default::default()
        }
    }

    /// A context carrying the request's [`RpcCompletion`] ledger, when the
    /// request came through `super::metrics_layer`.
    pub(crate) fn for_request<T>(rpc: &'static str, request: &tonic::Request<T>) -> Self {
        Self {
            endpoint: rpc,
            completion: request.extensions().get::<Arc<RpcCompletion>>().cloned(),
            ..Default::default()
        }
    }

    /// A context for a completion point that has only the ledger — the
    /// metrics layer's backstop. The endpoint is the ledger's.
    pub(crate) fn for_completion(completion: Arc<RpcCompletion>) -> Self {
        Self {
            endpoint: completion.endpoint(),
            completion: Some(completion),
            ..Default::default()
        }
    }

    /// Advance the recorded stage. Monotone in practice, but not enforced —
    /// the field is a report of where the handler was, not a state machine.
    pub(crate) fn at(&mut self, stage: RpcStage) {
        self.stage = stage;
    }

    /// Record the session id. Call ONLY after `validate_session_id` returned
    /// `Ok`.
    pub(crate) fn session(&mut self, session_id: &str) {
        self.session_id = Some(session_id.to_owned());
    }

    /// Record the layer name. Call ONLY after `validate_layer_name` returned
    /// `Ok`.
    pub(crate) fn layer(&mut self, layer_name: &str) {
        self.layer_name = Some(layer_name.to_owned());
    }

    /// Record the client id. Call ONLY after `validate_client_id` returned
    /// `Ok`.
    pub(crate) fn client(&mut self, client_id: &str) {
        self.client_id = Some(client_id.to_owned());
    }

    /// Record the operation id. Call ONLY after `validate_operation_id`
    /// returned `Ok`. An empty id is the field's absent value and is not
    /// recorded.
    pub(crate) fn operation(&mut self, operation_id: &str) {
        if !operation_id.is_empty() {
            self.operation_id = Some(operation_id.to_owned());
        }
    }

    /// Mark the call chained: its first message was `chained_header`. Call
    /// before the header's validators — the arm is server-derived, not an
    /// identifier — so that every `INVALID_ARGUMENT` of the call is
    /// recognised as a chained call's.
    pub(crate) fn chained(&mut self) {
        self.chained = true;
    }

    /// Whether the call is chained ([`Self::chained`]).
    pub(crate) fn is_chained(&self) -> bool {
        self.chained
    }

    /// Mark the call's commit started: `CreateLayer` / `UpdateLayer` call it
    /// immediately before handing the write to the store's commit, and no
    /// step before that call writes client-visible state. Never undone: once
    /// set, an `UNAVAILABLE` cannot be shown to have applied nothing.
    pub(crate) fn commit_started(&mut self) {
        self.commit_started = true;
    }

    /// Whether the call's commit started ([`Self::commit_started`]).
    pub(crate) fn is_commit_started(&self) -> bool {
        self.commit_started
    }

    /// Record a chained call's `previous_operation_id` (empty for a chain's
    /// first write) and `no_wait`. Call ONLY after `previous_operation_id`
    /// passed its validation.
    pub(crate) fn chain_link(&mut self, previous_operation_id: &str, no_wait: bool) {
        self.previous_operation_id = Some(previous_operation_id.to_owned());
        self.no_wait = Some(no_wait);
    }

    /// The commit attempts of the RPC's `LayerStore::update_layer` calls,
    /// summed, when any of those calls made more than one: a counted
    /// in-process retry, after which an earlier attempt's commit may still
    /// land, so a refusal can no longer promise the call applied nothing.
    /// `None` without one.
    pub(crate) fn update_layer_counted_retry(&self) -> Option<u32> {
        self.update_layer_retried
            .then_some(self.update_layer_attempts)
    }

    /// Record the base path. Always the **canonical** spelling — the value
    /// the service authorized and keyed storage under, not the caller's
    /// spelling.
    pub(crate) fn base_path(&mut self, base_path: &str) {
        self.base_path = Some(base_path.to_owned());
    }

    /// Record the object id the caller submitted.
    pub(crate) fn submitted_object_id(&mut self, object_id: u64) {
        self.object_id = Some(object_id);
    }

    /// Record the object id the layer mapping resolved to. Setting this also
    /// promotes `NOT_FOUND` / `FAILED_PRECONDITION` failures raised later in
    /// the flow from `info` to `warn` — see [`failure_level`].
    pub(crate) fn resolved_object_id(&mut self, object_id: u64) {
        self.current_object_id = Some(object_id);
    }

    /// Record the catch-up regime (`SubscribeReadLayer`).
    pub(crate) fn regime(&mut self, regime: &'static str) {
        self.regime = Some(regime);
    }

    /// Record the stream-exit reason (`SubscribeReadLayer`).
    pub(crate) fn exit_reason(&mut self, reason: &'static str) {
        self.exit_reason = Some(reason);
    }

    /// Record whether the first caught-up ack was enqueued
    /// (`SubscribeReadLayer`).
    pub(crate) fn caught_up(&mut self, caught_up: bool) {
        self.caught_up = Some(caught_up);
    }

    /// Record the server-minted connection id (`JoinSession`).
    pub(crate) fn connection_id(&mut self, connection_id: &str) {
        self.connection_id = Some(connection_id.to_owned());
    }

    /// Record which header check refused the request (auth interceptor).
    pub(crate) fn rejection(&mut self, rejection: &'static str) {
        self.rejection = Some(rejection);
    }

    /// Record the summary of the RPC's latest `LayerStore::update_layer` call
    /// (`UpdateLayer`), replacing an earlier call's: the failure line describes
    /// the call whose outcome the RPC reports. `None` — the call made no
    /// attempt — clears it. Every call's attempts also count toward
    /// [`Self::update_layer_counted_retry`].
    pub(crate) fn update_layer_call(&mut self, summary: Option<UpdateLayerCallSummary>) {
        if let Some(call) = &summary {
            self.update_layer_attempts = self.update_layer_attempts.saturating_add(call.attempts);
            self.update_layer_retried |= call.attempts > 1;
        }
        self.update_layer_call = summary;
    }

    /// One field of the recorded `update_layer` summary, `None` without one.
    fn update_layer_field<T>(&self, field: impl FnOnce(&UpdateLayerCallSummary) -> T) -> Option<T> {
        self.update_layer_call.as_ref().map(field)
    }

    /// Mark this completion as caused by graceful shutdown. Call ONLY from a
    /// streaming RPC's shutdown arm or shutdown refusal — the ending every
    /// stream on a draining pod takes on every rolling restart — or from an
    /// upload refused for shutdown at admission, while a processing step
    /// waits, at the external-value release budget, or while a chained
    /// `UpdateLayer` waits in the chain gate. It is
    /// an expected event, so
    /// [`failure_level`] logs its `UNAVAILABLE` at INFO rather than the WARN a
    /// dependency outage gets.
    pub(crate) fn shutdown_caused(&mut self) {
        self.shutdown_caused = true;
    }

    /// Mark this completion as a `JoinSession` ended by its client keepalive
    /// deadline. Call ONLY from the join's keepalive-timeout ending. A client
    /// whose laptop sleeps or whose network drops is an expected event, like a
    /// shutdown ending, so [`failure_level`] logs its `UNAVAILABLE` at INFO;
    /// the rate is on `live_edit_presence_keepalive_timeouts_total`.
    pub(crate) fn client_keepalive_timed_out(&mut self) {
        self.client_keepalive_timeout = true;
    }

    fn mark_reported(&self) {
        if let Some(completion) = &self.completion {
            completion.mark_reported();
        }
    }
}

/// The message every non-fault completion line carries. Stable and greppable:
/// one `grep 'RPC failed'` over a pod log yields every client-visible
/// rejection on either server, whichever completion point reported it — a
/// handler boundary, a streaming tail, the auth interceptor, or the metrics
/// layer's transport backstop. Three other messages complete the vocabulary,
/// each for a completion this one does not describe:
///
/// - `internal error` ([`INTERNAL_ERROR_MSG`]) at ERROR, for a server fault —
///   see [`log_rpc_failure`];
/// - `RPC deadline exceeded`, the server-side deadline in
///   `super::metrics_layer`, which drops the handler future before any
///   wrapper runs;
/// - [`RPC_ABANDONED_MSG`], a call torn down before it produced a response.
pub(crate) const RPC_FAILURE_MSG: &str = "RPC failed";

/// The message of the metrics layer's drop-guard line: a call torn down —
/// client reset, client `grpc-timeout`, connection loss — before any response
/// existed, so no completion point ever saw a status. Counted as `CANCELLED`.
pub(crate) const RPC_ABANDONED_MSG: &str = "RPC abandoned before completion";

/// Level for a non-OK completion, or `None` for `OK`.
///
/// `object_resolved` is `ctx.current_object_id.is_some()`; `shutdown_caused` is
/// the context's shutdown marker ([`RpcLogContext::shutdown_caused`]);
/// `keepalive_timed_out` is its client-keepalive marker
/// ([`RpcLogContext::client_keepalive_timed_out`]).
///
/// | code | condition | level |
/// |---|---|---|
/// | `NOT_FOUND` / `FAILED_PRECONDITION` | reached `Storage` with the object resolved | warn |
/// | `NOT_FOUND` / `FAILED_PRECONDITION` | otherwise | info |
/// | `INVALID_ARGUMENT`, `ALREADY_EXISTS`, `RESOURCE_EXHAUSTED`, `OUT_OF_RANGE`, `CANCELLED`, `ABORTED` | — | info |
/// | `PERMISSION_DENIED`, `UNAUTHENTICATED` | — | warn |
/// | `UNAVAILABLE` | shutdown-caused, at any stage | info |
/// | `UNAVAILABLE` | client-keepalive timeout, at any stage | info |
/// | `UNAVAILABLE` | otherwise | warn |
/// | `INTERNAL` | — | warn |
/// | anything else (incl. `UNKNOWN`) | — | warn |
/// | `OK` | — | none |
///
/// The table never yields `error`. A server fault is still an ERROR, but that
/// decision is made before the table: [`log_rpc_failure`] routes an
/// `INTERNAL` carrying an [`InternalFault`] to its own arm. The `INTERNAL` row
/// here is therefore an `INTERNAL` no `ServiceError` produced — tonic's own,
/// such as a request body that is not valid protobuf, which a client can cause
/// at will and so must not page anyone.
///
/// `PERMISSION_DENIED` / `UNAUTHENTICATED` are the one client-caused family
/// raised to `warn`, because PRD §7.2 prescribes WARN for authorization
/// denials (a denial burst is a security signal, not routine traffic).
/// `OUT_OF_RANGE` sits with `RESOURCE_EXHAUSTED`: a client that exceeded a
/// limit, most often the request-message size. No handler answers
/// `RESOURCE_EXHAUSTED` itself — an upload over `LIVE_EDIT_MAX_LAYER_SIZE`
/// answers `INVALID_ARGUMENT`, which clients do not retry — so the row covers
/// one raised by a layer below the handlers.
///
/// `ABORTED` is `UpdateLayer`'s chained-write `PREDECESSOR_NOT_COMMITTED`,
/// the only `ABORTED` any handler builds (`src/error.rs`). It logs at INFO,
/// as does a chained call's `INVALID_ARGUMENT` `INVALID_UPDATE`: the
/// client-caused chained refusals (SDD §17.2.1). A predecessor refusal can
/// come from the client's own duplicate or cancelled calls of one write as
/// well as from a restore, and the server cannot tell them apart. A chained
/// call's `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` (no read of its record
/// succeeded) is not client-caused and logs at WARN, like every `UNAVAILABLE`
/// a shutdown or a client-keepalive timeout did not cause.
/// `PREDECESSOR_NOT_COMMITTED` has two producers: the chain gate, at stage
/// `chain`, and the commit path's predecessor guard on a call's first
/// attempt, at stage `storage` (after a counted retry that refusal is a plain
/// `UNAVAILABLE` instead, at stage `storage`, at WARN like every other such
/// `UNAVAILABLE`).
///
/// The two marker rows are the only places the markers matter. Graceful
/// shutdown ends every stream on the draining pod `UNAVAILABLE`, on every
/// rolling restart, so at WARN it would bury real dependency outages under
/// routine deploys. The client-keepalive timeout ends a `JoinSession` whose
/// client went silent — a laptop that slept, a network that dropped — which is
/// routine at any scale, so it is INFO too, and its rate is read from
/// `live_edit_presence_keepalive_timeouts_total`. Neither marker changes the
/// level of any other code.
pub(crate) fn failure_level(
    code: Code,
    stage: RpcStage,
    object_resolved: bool,
    shutdown_caused: bool,
    keepalive_timed_out: bool,
) -> Option<Level> {
    match code {
        Code::Ok => None,
        // The two codes whose meaning depends on how far the request got.
        // Reaching `Storage` with a resolved object means the mapping proved
        // the object live and the *commit path* then refused it — the
        // EMPTY_HEADER / PARTIAL_HEADER / SUPERSEDED sentinel family, i.e.
        // damaged or concurrently-retired keyspace. Anywhere earlier (or with
        // no object resolved) it is the ordinary client shape: an unknown
        // layer name, or a stale `object_id`.
        Code::NotFound | Code::FailedPrecondition => {
            if stage == RpcStage::Storage && object_resolved {
                Some(Level::WARN)
            } else {
                Some(Level::INFO)
            }
        }
        Code::InvalidArgument
        | Code::AlreadyExists
        | Code::ResourceExhausted
        | Code::OutOfRange
        | Code::Cancelled => Some(Level::INFO),
        // A chained `UpdateLayer`'s predecessor refusal, the only `ABORTED`
        // (see the doc comment).
        Code::Aborted => Some(Level::INFO),
        // PRD §7.2: authorization denials are WARN.
        Code::PermissionDenied | Code::Unauthenticated => Some(Level::WARN),
        // A graceful shutdown's retryable ending: expected on every rolling
        // restart, so not a warning.
        Code::Unavailable if shutdown_caused => Some(Level::INFO),
        // A `JoinSession` whose client went silent past its keepalive
        // deadline: expected whenever a client sleeps or loses its network.
        Code::Unavailable if keepalive_timed_out => Some(Level::INFO),
        // Server-side or dependency conditions the client did not cause.
        Code::Unavailable => Some(Level::WARN),
        // An `INTERNAL` without an `InternalFault` — see the doc comment.
        Code::Internal => Some(Level::WARN),
        // Defensive: a code no handler is expected to produce still gets a
        // line, and gets it loudly enough to notice.
        _ => Some(Level::WARN),
    }
}

/// One event carrying every identifier an [`RpcLogContext`] holds, plus the
/// caller's extra fields in front.
///
/// `tracing` levels are compile-time metadata, so each level is its own macro
/// call site and the field list is shared through this macro (idiom: `emit_at!`
/// in `src/storage/client.rs`).
macro_rules! emit_with_context {
    ($level:ident, $ctx:expr, $msg:expr $(, $extra:ident = $extra_val:expr)*) => {
        tracing::$level!(
            $($extra = $extra_val,)*
            endpoint = $ctx.endpoint,
            stage = $ctx.stage.label(),
            session_id = $ctx.session_id.as_deref(),
            layer_name = $ctx.layer_name.as_deref(),
            base_path = $ctx.base_path.as_deref(),
            object_id = $ctx.object_id,
            current_object_id = $ctx.current_object_id,
            client_id = $ctx.client_id.as_deref(),
            operation_id = $ctx.operation_id.as_deref(),
            chained = $ctx.chained.then_some(true),
            previous_operation_id = $ctx.previous_operation_id.as_deref(),
            no_wait = $ctx.no_wait,
            connection_id = $ctx.connection_id.as_deref(),
            regime = $ctx.regime,
            reason = $ctx.exit_reason,
            caught_up = $ctx.caught_up,
            rejection = $ctx.rejection,
            attempt = $ctx.update_layer_field(|s| s.attempts),
            path = $ctx.update_layer_field(|s| s.path),
            watch_conflict =
                $ctx.update_layer_field(|s| s.conflicts(UPDATE_LAYER_CONFLICT_WATCH_CONFLICT)),
            watch_read_moved =
                $ctx.update_layer_field(|s| s.conflicts(UPDATE_LAYER_CONFLICT_WATCH_READ_MOVED)),
            seq_fence = $ctx.update_layer_field(|s| s.conflicts(UPDATE_LAYER_CONFLICT_SEQ_FENCE)),
            cache_diverged =
                $ctx.update_layer_field(|s| s.conflicts(UPDATE_LAYER_CONFLICT_CACHE_DIVERGED)),
            snapshot_fence_wait = $ctx
                .update_layer_field(|s| s.conflicts(UPDATE_LAYER_CONFLICT_SNAPSHOT_FENCE_WAIT)),
            records = $ctx.update_layer_field(|s| s.records).flatten(),
            bytes = $ctx.update_layer_field(|s| s.bytes).flatten(),
            "{}",
            $msg
        )
    };
}

/// Emit the single completion line for a failed RPC, and mark the request's
/// ledger reported.
///
/// Called once per non-OK completion, by whichever completion point decided
/// the status (see the module comment). Silent for `OK`.
///
/// **Server faults first.** An `INTERNAL` whose `source()` is an
/// [`InternalFault`] emits exactly one `error!` with message
/// [`INTERNAL_ERROR_MSG`], the fault's un-redacted detail as `error`, and the
/// standard request fields; it then marks the fault reported, which silences
/// the fault's own fallback line. Every other status follows
/// [`failure_level`] and emits [`RPC_FAILURE_MSG`].
///
/// Fields are attached to the **event**, not a span: the log layer sees
/// events only, so a field on a span would never reach the line, and carrying
/// the request identifiers on a span would cost the hot success path for a
/// payload only the failure path reads. The call's one server span
/// (`MetricsLayer`) exists for trace export alone; a line emitted inside it
/// that the trace filter admits is also exported as one of its span events.
///
/// Optional identifiers are passed as `Option`, which `tracing` records only
/// when `Some` — an absent field is simply not present in the JSON, rather
/// than present-and-null.
pub(crate) fn log_rpc_failure(ctx: &RpcLogContext, status: &tonic::Status) {
    // Same string the `grpc_code` metric label uses, so a log line and a
    // `live_endpoint_calls_total` series join without translation. Built
    // through the metrics module's own mapper rather than a local copy —
    // a second mapping would be free to drift.
    let grpc_code =
        crate::observability::metrics::grpc_code_slug(Some(&(status.code() as i32).to_string()));
    // Not named `message`: `tracing` reserves that field name for the event's
    // own message. Matches the `storage_message` idiom already used by
    // `handle_update_layer`'s recovery arm.
    let status_message = status.message();

    if status.code() == Code::Internal {
        if let Some(fault) = status
            .source()
            .and_then(|source| source.downcast_ref::<InternalFault>())
        {
            emit_with_context!(
                error,
                ctx,
                INTERNAL_ERROR_MSG,
                error = tracing::field::display(fault.detail()),
                grpc_code = grpc_code,
                status_message = status_message
            );
            fault.mark_reported();
            ctx.mark_reported();
            return;
        }
    }

    let Some(level) = failure_level(
        status.code(),
        ctx.stage,
        ctx.current_object_id.is_some(),
        ctx.shutdown_caused,
        ctx.client_keepalive_timeout,
    ) else {
        return;
    };

    // Total by construction: the `failure_level` table and this emitter are
    // far apart, so a row added there with a third level must fail loudly in
    // debug rather than be silently downgraded to `info` by an `else` arm.
    if level == Level::WARN {
        emit_with_context!(
            warn,
            ctx,
            RPC_FAILURE_MSG,
            grpc_code = grpc_code,
            status_message = status_message
        );
    } else if level == Level::INFO {
        emit_with_context!(
            info,
            ctx,
            RPC_FAILURE_MSG,
            grpc_code = grpc_code,
            status_message = status_message
        );
    } else {
        debug_assert!(false, "failure_level returned an unhandled level {level:?}");
        emit_with_context!(
            warn,
            ctx,
            RPC_FAILURE_MSG,
            grpc_code = grpc_code,
            status_message = status_message
        );
    }
    ctx.mark_reported();
}

#[cfg(test)]
mod tests {
    //! Level matrix, context-population, ledger and emitter unit tests for the
    //! shared completion-logging vocabulary.
    //!
    //! The end-to-end assertions — that a real completion point emits this
    //! event with these fields for a real rejection — live in
    //! `tests/integration/grpc/rpc_failure_log_test.rs` (the single-response
    //! handlers), `stream_completion_log_test.rs` (the two streaming RPCs) and
    //! `completion_log_server_test.rs` (the interceptor, the metrics layer's
    //! backstop and drop guard, behind a real tonic server). What lives here is
    //! the part an integration test cannot cover exhaustively: every
    //! `tonic::Code` against every [`RpcStage`] against both resolution states,
    //! which is 272 combinations and one Valkey round trip each if driven
    //! through a handler, plus the `InternalFault` arm and the ledger marking
    //! that every completion point relies on.
    //!
    //! These tests deliberately DO assert on the level, the field set, and the
    //! event message, which the test-writing guide's "what NOT to test" list
    //! otherwise warns off as observability rather than contract. This is the
    //! case where that inverts: the log line *is* the deliverable — an operator
    //! grepping `RPC failed` and reading `object_id` / `current_object_id` off
    //! it is the entire remediation this module exists to provide, and a
    //! silent downgrade to `debug!` or a dropped identifier would restore the
    //! exact blindness the logging was added to cure while every status-code
    //! test in the tree stayed green.

    use super::*;

    /// Every `tonic::Code`, including the ones no handler is expected to
    /// produce.
    ///
    /// Enumerated by hand because `tonic::Code` is a plain C-like enum with no
    /// iterator. [`all_codes_covers_every_tonic_code`] is what keeps the list
    /// honest against a tonic upgrade that adds a variant.
    const ALL_CODES: &[Code] = &[
        Code::Ok,
        Code::Cancelled,
        Code::Unknown,
        Code::InvalidArgument,
        Code::DeadlineExceeded,
        Code::NotFound,
        Code::AlreadyExists,
        Code::PermissionDenied,
        Code::ResourceExhausted,
        Code::FailedPrecondition,
        Code::Aborted,
        Code::OutOfRange,
        Code::Unimplemented,
        Code::Internal,
        Code::Unavailable,
        Code::DataLoss,
        Code::Unauthenticated,
    ];

    /// Every [`RpcStage`], in flow order.
    /// [`rpc_stage_label_is_a_distinct_stable_lowercase_string_per_variant`]
    /// pairs each with its label and fails if one is listed here without a
    /// label expectation.
    const ALL_STAGES: &[RpcStage] = &[
        RpcStage::Header,
        RpcStage::Auth,
        RpcStage::Admission,
        RpcStage::Resolve,
        RpcStage::Body,
        RpcStage::Chain,
        RpcStage::Parse,
        RpcStage::Storage,
        RpcStage::Stream,
        RpcStage::Transport,
    ];

    /// Every combination of the two context markers, `(shutdown_caused,
    /// keepalive_timed_out)`.
    const MARKERS: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

    /// Compile-time half of [`ALL_STAGES`]'s completeness: a new [`RpcStage`]
    /// variant fails to compile here until it is matched, which is the prompt
    /// to list it in [`ALL_STAGES`] and give it a label expectation. The
    /// length check in the label test cannot catch a variant missing from both
    /// lists.
    #[allow(dead_code)]
    fn every_stage_is_listed(stage: RpcStage) {
        match stage {
            RpcStage::Header
            | RpcStage::Auth
            | RpcStage::Admission
            | RpcStage::Resolve
            | RpcStage::Body
            | RpcStage::Chain
            | RpcStage::Parse
            | RpcStage::Storage
            | RpcStage::Stream
            | RpcStage::Transport => {}
        }
    }

    /// The documented table, restated independently of the implementation.
    ///
    /// Deliberately NOT a call into `failure_level` with the arms rearranged —
    /// that would be the implementation asserting about itself. This is the
    /// doc-comment table on `failure_level` transcribed, so a behaviour change
    /// there has to be reflected here (and, being a table, read) rather than
    /// absorbed silently.
    ///
    /// `INTERNAL` is WARN here because this table only ever sees an `INTERNAL`
    /// with no `InternalFault` source — tonic's own, which a client can cause
    /// with a malformed body. The server-fault ERROR is decided before the
    /// table, in `log_rpc_failure`.
    ///
    /// `shutdown_caused` is the context's shutdown marker: it moves
    /// `UNAVAILABLE` to INFO at every stage and resolution, and nothing else.
    /// Graceful shutdown ends every stream on a draining pod that way on every
    /// rolling restart, so at WARN it would bury real dependency outages under
    /// routine deploys; without the marker `UNAVAILABLE` stays WARN.
    ///
    /// `keepalive_timed_out` is the context's client-keepalive marker, with the
    /// same effect: a `JoinSession` ended because its client went silent past
    /// its keepalive deadline is routine, so its `UNAVAILABLE` is INFO.
    fn expected_level(
        code: Code,
        stage: RpcStage,
        resolved: bool,
        shutdown_caused: bool,
        keepalive_timed_out: bool,
    ) -> Option<Level> {
        match code {
            Code::Ok => None,
            Code::Unavailable if shutdown_caused => Some(Level::INFO),
            Code::Unavailable if keepalive_timed_out => Some(Level::INFO),
            Code::NotFound | Code::FailedPrecondition => {
                if stage == RpcStage::Storage && resolved {
                    Some(Level::WARN)
                } else {
                    Some(Level::INFO)
                }
            }
            Code::InvalidArgument
            | Code::AlreadyExists
            | Code::ResourceExhausted
            | Code::OutOfRange
            | Code::Cancelled
            | Code::Aborted => Some(Level::INFO),
            _ => Some(Level::WARN),
        }
    }

    // ── Level matrix ─────────────────────────────────────────────────────────

    /// The full 17 × 10 × 2 × 2 × 2 matrix — code, stage, resolution, shutdown
    /// marker, client-keepalive marker — agrees with the documented table.
    ///
    /// Exhaustive by construction, so the individually interesting rows need no
    /// separate tests — they are cells in here. Two worth naming, because they
    /// are the ones a reader will look for: `(FAILED_PRECONDITION, Resolve,
    /// resolved)` stays INFO, which is the exact shape of the `object_id`
    /// mismatch this ticket was filed for (the mapping resolved — that is what
    /// puts both ids on the line — but the refusal happened before the commit,
    /// so it is a stale client, not damage); and every other pre-`Storage`
    /// stage with a resolved object stays INFO for the same reason.
    #[test]
    fn failure_level_matches_the_documented_table_for_every_code_stage_and_resolution() {
        for &code in ALL_CODES {
            for &stage in ALL_STAGES {
                for resolved in [false, true] {
                    for shutdown in [false, true] {
                        for keepalive in [false, true] {
                            assert_eq!(
                                failure_level(code, stage, resolved, shutdown, keepalive),
                                expected_level(code, stage, resolved, shutdown, keepalive),
                                "level table disagreement for code={code:?} stage={} \
                                 resolved={resolved} shutdown_caused={shutdown} \
                                 keepalive_timed_out={keepalive}",
                                stage.label()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The shutdown marker changes the level of `UNAVAILABLE` and of no other
    /// code, anywhere in the matrix. A marker that downgraded, say, an
    /// `INTERNAL` or an `ABORTED` would hide a real failure that happened to
    /// land during a shutdown.
    #[test]
    fn failure_level_shutdown_marker_changes_no_other_code() {
        for &code in ALL_CODES {
            for &stage in ALL_STAGES {
                for resolved in [false, true] {
                    let changes = failure_level(code, stage, resolved, true, false)
                        != failure_level(code, stage, resolved, false, false);
                    assert_eq!(
                        changes,
                        code == Code::Unavailable,
                        "code={code:?} stage={} resolved={resolved}: the shutdown marker may \
                         change the level of UNAVAILABLE only",
                        stage.label()
                    );
                }
            }
        }
    }

    /// The client-keepalive marker, like the shutdown marker, changes the
    /// level of `UNAVAILABLE` and of no other code, anywhere in the matrix —
    /// with or without the shutdown marker beside it. A marker that
    /// downgraded, say, an `INTERNAL` would hide a real failure that happened
    /// to end a join whose client also went silent.
    #[test]
    fn failure_level_client_keepalive_marker_changes_no_other_code() {
        for &code in ALL_CODES {
            for &stage in ALL_STAGES {
                for resolved in [false, true] {
                    for shutdown in [false, true] {
                        let changes = failure_level(code, stage, resolved, shutdown, true)
                            != failure_level(code, stage, resolved, shutdown, false);
                        assert_eq!(
                            changes,
                            code == Code::Unavailable && !shutdown,
                            "code={code:?} stage={} resolved={resolved} \
                             shutdown_caused={shutdown}: the client-keepalive marker may change \
                             the level of UNAVAILABLE only (and not one the shutdown marker has \
                             already made INFO)",
                            stage.label()
                        );
                    }
                }
            }
        }
    }

    /// No input, anywhere in the matrix, yields `error`.
    ///
    /// The hard constraint: a client-caused rejection must never page anyone.
    /// ERROR is reserved for a server fault, which `log_rpc_failure` recognises
    /// by its `InternalFault` source *before* consulting this table. The table
    /// also sees tonic's own `INTERNAL` (a request body that is not valid
    /// protobuf), which any client can produce at will — an `error` here would
    /// turn a fuzzer into an alert.
    #[test]
    fn failure_level_never_returns_error_for_any_code_stage_or_resolution() {
        for &code in ALL_CODES {
            for &stage in ALL_STAGES {
                for resolved in [false, true] {
                    for (shutdown, keepalive) in MARKERS {
                        let level = failure_level(code, stage, resolved, shutdown, keepalive);
                        assert!(
                            level != Some(Level::ERROR),
                            "code={code:?} stage={} resolved={resolved} \
                             shutdown_caused={shutdown} keepalive_timed_out={keepalive} logs \
                             at ERROR; the level table must never do that for ANY code — \
                             ERROR belongs to the InternalFault arm of log_rpc_failure",
                            stage.label()
                        );
                        assert!(
                            matches!(level, None | Some(Level::WARN) | Some(Level::INFO)),
                            "code={code:?} stage={} resolved={resolved} \
                             shutdown_caused={shutdown} keepalive_timed_out={keepalive} \
                             produced {level:?}; the emitter only dispatches WARN and INFO, so \
                             any other level would be silently re-routed to WARN by its \
                             fallback arm",
                            stage.label()
                        );
                    }
                }
            }
        }
    }

    /// `OK` is the one silent code: every other code gets a line regardless of
    /// how far the request got.
    ///
    /// `OK` because a successful RPC is not a failure and the per-commit success
    /// line stays at `debug`. Every other code — `INTERNAL` included — must log,
    /// because the invariant is "every non-OK completion emits exactly one
    /// line": a `None` for any non-OK code is a completion that vanishes.
    #[test]
    fn failure_level_ok_is_the_only_silent_code_in_every_stage_and_resolution() {
        for &code in ALL_CODES {
            for &stage in ALL_STAGES {
                for resolved in [false, true] {
                    for (shutdown, keepalive) in MARKERS {
                        assert_eq!(
                            failure_level(code, stage, resolved, shutdown, keepalive).is_none(),
                            code == Code::Ok,
                            "code={code:?} stage={} resolved={resolved} \
                             shutdown_caused={shutdown} keepalive_timed_out={keepalive}: only \
                             OK may be silent",
                            stage.label()
                        );
                    }
                }
            }
        }
    }

    /// `INTERNAL` without a fault source, and `OUT_OF_RANGE`, land on the levels
    /// the table documents in every stage.
    ///
    /// Named out of the matrix because they are the two rows this table
    /// changed, and both are reachable only from tonic itself: `INTERNAL` is
    /// what tonic answers for a request body that is not valid protobuf (so it
    /// is WARN, never ERROR — a client can cause it), and `OUT_OF_RANGE` is its
    /// answer to a request over the 4 MB decode limit (a client that exceeded a
    /// limit, so INFO beside `RESOURCE_EXHAUSTED`).
    #[test]
    fn failure_level_tonic_internal_is_warn_and_out_of_range_is_info_in_every_stage() {
        for &stage in ALL_STAGES {
            for resolved in [false, true] {
                assert_eq!(
                    failure_level(Code::Internal, stage, resolved, false, false),
                    Some(Level::WARN),
                    "an INTERNAL reaching the table has no fault source and is client-causable; \
                     stage={} resolved={resolved}",
                    stage.label()
                );
                assert_eq!(
                    failure_level(Code::OutOfRange, stage, resolved, false, false),
                    Some(Level::INFO),
                    "OUT_OF_RANGE is a client over a limit; stage={} resolved={resolved}",
                    stage.label()
                );
            }
        }
    }

    /// The `(Storage, resolved)` escalation applies to exactly two codes.
    ///
    /// Stated as a difference rather than as two more table rows: the property
    /// that matters is that reaching the commit path against a proven-live
    /// object changes the reading of `NOT_FOUND` / `FAILED_PRECONDITION` (client
    /// mistake → keyspace damage) and changes *nothing else*. A refactor that
    /// made the escalation general would still satisfy the per-row table for
    /// those two codes.
    ///
    /// The `(Storage, not resolved)` half asserted below is the `DeleteLayer`
    /// shape: it runs its storage delete without resolving a mapping of its
    /// own, so its NOT_FOUND (a delete of an absent layer) is the ordinary
    /// client shape even though the recorded stage is `Storage`. The pair is
    /// the discriminator, never either half alone — escalating on the stage
    /// by itself would make every idempotent re-delete WARN.
    #[test]
    fn failure_level_storage_and_resolved_escalates_only_not_found_and_failed_precondition() {
        for &code in ALL_CODES {
            let escalates = failure_level(code, RpcStage::Storage, true, false, false)
                != failure_level(code, RpcStage::Storage, false, false, false);
            let should_escalate = matches!(code, Code::NotFound | Code::FailedPrecondition);
            assert_eq!(
                escalates, should_escalate,
                "code={code:?}: escalation on (Storage, resolved) must be exclusive to NOT_FOUND \
                 and FAILED_PRECONDITION"
            );
            if should_escalate {
                assert_eq!(
                    failure_level(code, RpcStage::Storage, true, false, false),
                    Some(Level::WARN),
                    "code={code:?} at the commit path against a live object is keyspace damage \
                     and must be WARN"
                );
                assert_eq!(
                    failure_level(code, RpcStage::Storage, false, false, false),
                    Some(Level::INFO),
                    "code={code:?} with no object ever resolved is the client shape and must be \
                     INFO"
                );
            }
        }
    }

    /// Guard against a tonic upgrade adding a `Code` variant [`ALL_CODES`] does
    /// not list.
    ///
    /// Without this the matrix tests above would keep passing while covering
    /// one fewer code than they claim. `Code::from_i32` is tonic's own
    /// numeric-to-variant mapping, so round-tripping `0..=16` and checking the
    /// set membership catches both an added variant and a renumbering.
    #[test]
    fn all_codes_covers_every_tonic_code() {
        for raw in 0..=16i32 {
            let code = Code::from_i32(raw);
            assert!(
                ALL_CODES.contains(&code),
                "tonic::Code::from_i32({raw}) = {code:?} is missing from ALL_CODES; the level \
                 matrix is covering fewer codes than it claims"
            );
        }
        assert_eq!(
            ALL_CODES.len(),
            17,
            "ALL_CODES must hold each of the 17 gRPC codes exactly once"
        );
    }

    // ── Stage labels ─────────────────────────────────────────────────────────

    /// Every stage carries the stable lowercase label operators filter on, and
    /// no two stages share one.
    #[test]
    fn rpc_stage_label_is_a_distinct_stable_lowercase_string_per_variant() {
        let pairs = [
            (RpcStage::Header, "header"),
            (RpcStage::Auth, "auth"),
            (RpcStage::Admission, "admission"),
            (RpcStage::Resolve, "resolve"),
            (RpcStage::Body, "body"),
            (RpcStage::Chain, "chain"),
            (RpcStage::Parse, "parse"),
            (RpcStage::Storage, "storage"),
            (RpcStage::Stream, "stream"),
            (RpcStage::Transport, "transport"),
        ];
        assert_eq!(
            pairs.len(),
            ALL_STAGES.len(),
            "a stage was added without a label expectation"
        );
        for (stage, want) in pairs {
            assert_eq!(stage.label(), want, "label drift for {stage:?}");
        }
        let mut labels: Vec<&str> = ALL_STAGES.iter().map(|s| s.label()).collect();
        labels.sort_unstable();
        let distinct = labels.len();
        labels.dedup();
        assert_eq!(
            labels.len(),
            distinct,
            "two stages share a label, so the `stage` field cannot discriminate them"
        );
    }

    /// `Header` is the default, so a request that fails before the handler
    /// records anything reports the ingress stage rather than an arbitrary one.
    #[test]
    fn rpc_stage_default_is_header() {
        assert_eq!(RpcStage::default(), RpcStage::Header);
    }

    // ── Context population ───────────────────────────────────────────────────

    /// How many optional emitted fields a context currently carries. The
    /// ledger is deliberately not counted: it is never emitted.
    fn populated_count(ctx: &RpcLogContext) -> usize {
        [
            ctx.session_id.is_some(),
            ctx.layer_name.is_some(),
            ctx.base_path.is_some(),
            ctx.object_id.is_some(),
            ctx.current_object_id.is_some(),
            ctx.client_id.is_some(),
            ctx.regime.is_some(),
            ctx.exit_reason.is_some(),
            ctx.caught_up.is_some(),
            ctx.connection_id.is_some(),
            ctx.rejection.is_some(),
        ]
        .iter()
        .filter(|set| **set)
        .count()
    }

    /// A fresh context carries the endpoint and nothing else.
    ///
    /// This is the populate-after-validate floor: an RPC rejected by its first
    /// validator must log no attacker-controlled identifier at all, and that
    /// holds only if the context starts empty.
    #[test]
    fn rpc_log_context_new_records_the_endpoint_and_no_identifiers() {
        let ctx = RpcLogContext::new("UpdateLayer");
        assert_eq!(ctx.endpoint, "UpdateLayer");
        assert_eq!(ctx.stage, RpcStage::Header);
        assert_eq!(
            populated_count(&ctx),
            0,
            "a fresh context must carry no identifiers"
        );
        assert!(ctx.completion.is_none(), "`new` must carry no ledger");
    }

    /// `for_request` picks the ledger up from the request extensions and starts
    /// otherwise empty; a request with no ledger yields a context with none.
    ///
    /// The ledger is what stops the metrics layer's backstop from logging a
    /// failure a second time, so a production context that silently dropped
    /// it would double every handler failure line behind a real server.
    #[test]
    fn rpc_log_context_for_request_carries_the_request_ledger_and_no_identifiers() {
        let completion = Arc::new(RpcCompletion::new("StatLayer"));
        let mut request = tonic::Request::new(());
        request.extensions_mut().insert(Arc::clone(&completion));

        let ctx = RpcLogContext::for_request("StatLayer", &request);
        assert_eq!(ctx.endpoint, "StatLayer");
        assert_eq!(ctx.stage, RpcStage::Header);
        assert_eq!(populated_count(&ctx), 0);
        let carried = ctx
            .completion
            .as_ref()
            .expect("a request carrying a ledger must yield a context carrying it");
        assert!(
            Arc::ptr_eq(carried, &completion),
            "the context must carry the request's own ledger, not a fresh one"
        );

        let bare = RpcLogContext::for_request("StatLayer", &tonic::Request::new(()));
        assert!(
            bare.completion.is_none(),
            "a request that never passed the metrics layer has no ledger to carry"
        );
    }

    /// `for_completion` takes its endpoint from the ledger — the backstop knows
    /// nothing else about the request.
    #[test]
    fn rpc_log_context_for_completion_takes_the_ledger_endpoint() {
        let completion = Arc::new(RpcCompletion::new("CreateLayer"));
        let ctx = RpcLogContext::for_completion(Arc::clone(&completion));
        assert_eq!(ctx.endpoint, "CreateLayer");
        assert_eq!(ctx.stage, RpcStage::Header);
        assert_eq!(populated_count(&ctx), 0);
        assert!(Arc::ptr_eq(
            ctx.completion.as_ref().expect("ledger carried"),
            &completion
        ));
    }

    /// A cloned context shares the ledger: the streaming tails log from a
    /// clone handed to a spawned task, and that clone's emission must mark the
    /// request's ledger or the metrics layer's backstop logs the same status
    /// again off the trailer.
    #[test]
    fn rpc_log_context_clone_marks_the_same_ledger() {
        let completion = Arc::new(RpcCompletion::new("SubscribeReadLayer"));
        let ctx = RpcLogContext::for_completion(Arc::clone(&completion));
        let mut tail = ctx.clone();
        tail.at(RpcStage::Stream);
        drop(ctx);

        let _ = capture_json(|| log_rpc_failure(&tail, &tonic::Status::unavailable("gone")));
        assert!(
            completion.is_reported(),
            "the clone's emission must mark the original request's ledger"
        );
    }

    /// Each setter populates exactly one field and leaves the rest untouched.
    ///
    /// A setter that also wrote a sibling field would put an identifier on the
    /// line before its own validator ran — the populate-after-validate hole,
    /// re-opened on this sink.
    #[test]
    fn rpc_log_context_setters_each_populate_exactly_one_field() {
        type Setter = (&'static str, fn(&mut RpcLogContext));
        let setters: &[Setter] = &[
            ("session", |c| c.session("s-1")),
            ("layer", |c| c.layer("l-1")),
            ("base_path", |c| c.base_path("/stage")),
            ("client", |c| c.client("c-1")),
            ("submitted_object_id", |c| c.submitted_object_id(7)),
            ("resolved_object_id", |c| c.resolved_object_id(9)),
            ("regime", |c| c.regime("in_window")),
            ("exit_reason", |c| c.exit_reason("slow_reader_deadline")),
            ("caught_up", |c| c.caught_up(false)),
            ("connection_id", |c| c.connection_id("conn-1")),
            ("rejection", |c| c.rejection("missing_header")),
        ];
        for (name, apply) in setters {
            let mut ctx = RpcLogContext::new("UpdateLayer");
            apply(&mut ctx);
            assert_eq!(
                populated_count(&ctx),
                1,
                "`{name}` must populate exactly one field"
            );
            assert_eq!(
                ctx.stage,
                RpcStage::Header,
                "`{name}` must not advance the stage"
            );
        }
    }

    /// The setters record the values they were handed, under the right fields.
    #[test]
    fn rpc_log_context_setters_record_the_submitted_values() {
        let mut ctx = RpcLogContext::new("UpdateLayer");
        ctx.session("sess-1");
        ctx.layer("layer-1");
        ctx.base_path("/canonical/stage");
        ctx.client("client-1");
        ctx.submitted_object_id(41);
        ctx.resolved_object_id(42);
        ctx.regime("out_of_window");
        ctx.exit_reason("consumer_fault");
        ctx.caught_up(true);
        ctx.connection_id("conn-7");
        ctx.rejection("not_bearer");

        assert_eq!(ctx.session_id.as_deref(), Some("sess-1"));
        assert_eq!(ctx.layer_name.as_deref(), Some("layer-1"));
        assert_eq!(ctx.base_path.as_deref(), Some("/canonical/stage"));
        assert_eq!(ctx.client_id.as_deref(), Some("client-1"));
        assert_eq!(
            (ctx.object_id, ctx.current_object_id),
            (Some(41), Some(42)),
            "the submitted and resolved ids must occupy DIFFERENT fields — collapsing them \
             loses the `expected N, got M` fact the dev incident had to be reconstructed from"
        );
        assert_eq!(ctx.regime, Some("out_of_window"));
        assert_eq!(ctx.exit_reason, Some("consumer_fault"));
        assert_eq!(ctx.caught_up, Some(true));
        assert_eq!(ctx.connection_id.as_deref(), Some("conn-7"));
        assert_eq!(ctx.rejection, Some("not_bearer"));
    }

    /// `at` moves the stage and touches nothing else.
    #[test]
    fn rpc_log_context_at_advances_only_the_stage() {
        let mut ctx = RpcLogContext::new("StatLayer");
        ctx.session("sess-1");
        ctx.at(RpcStage::Storage);
        assert_eq!(ctx.stage, RpcStage::Storage);
        assert_eq!(ctx.endpoint, "StatLayer");
        assert_eq!(populated_count(&ctx), 1);
    }

    // ── Emitter ──────────────────────────────────────────────────────────────

    /// Capture sink for [`capture_json`]. Local copy of the tree-wide
    /// `CapturingWriter` idiom (`src/checkpoint/assembler.rs::tests`).
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `f` under a JSON-formatting capture subscriber and return every
    /// event it emitted, parsed.
    ///
    /// JSON rather than the human format so field *absence* can be asserted
    /// structurally: `tracing` records nothing for a `None`, so an absent
    /// identifier is a missing key, not a `null` — and a substring search for
    /// `session_id` could never tell "absent" from "present but empty".
    ///
    /// `set_default` (thread-local) is sufficient here and `WithSubscriber` is
    /// not needed: these are synchronous calls with no `.await` inside, so the
    /// emitter cannot migrate to another thread mid-event. The integration
    /// tests, which drive real async handlers, use `WithSubscriber` for lines
    /// emitted inside the awaited handler, and a thread-local capture on a
    /// current-thread runtime for lines emitted by a spawned task.
    fn capture_json(f: impl FnOnce()) -> Vec<serde_json::Value> {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(std::sync::Arc::clone(&buf)))
            .with_max_level(Level::TRACE)
            .with_ansi(false)
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            f();
        }
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect()
    }

    /// The events whose message is [`RPC_FAILURE_MSG`].
    fn rpc_failure_events(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
        events
            .iter()
            .filter(|e| e["fields"]["message"] == RPC_FAILURE_MSG)
            .collect()
    }

    fn ctx_with_everything() -> RpcLogContext {
        let mut ctx = RpcLogContext::new("UpdateLayer");
        ctx.session("sess-1");
        ctx.layer("layer-1");
        ctx.base_path("/stage");
        ctx.client("client-1");
        ctx.submitted_object_id(41);
        ctx.resolved_object_id(42);
        ctx.at(RpcStage::Resolve);
        ctx
    }

    /// The mismatch shape emits one INFO line carrying every identifier the
    /// handler had, including BOTH object ids.
    #[test]
    fn log_rpc_failure_object_id_mismatch_emits_one_info_event_with_both_object_ids() {
        let events = capture_json(|| {
            log_rpc_failure(
                &ctx_with_everything(),
                &tonic::Status::failed_precondition("object_id mismatch: expected 42, got 41"),
            );
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(
            failures.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        let event = failures[0];
        assert_eq!(event["level"], "INFO");
        assert_eq!(event["fields"]["endpoint"], "UpdateLayer");
        assert_eq!(event["fields"]["stage"], "resolve");
        assert_eq!(event["fields"]["grpc_code"], "FAILED_PRECONDITION");
        assert_eq!(event["fields"]["session_id"], "sess-1");
        assert_eq!(event["fields"]["layer_name"], "layer-1");
        assert_eq!(event["fields"]["client_id"], "client-1");
        assert_eq!(event["fields"]["object_id"], 41);
        assert_eq!(event["fields"]["current_object_id"], 42);
        assert_eq!(
            event["fields"]["status_message"],
            "object_id mismatch: expected 42, got 41"
        );
    }

    /// A context marked shutdown-caused emits its `UNAVAILABLE` as one INFO
    /// `RPC failed` line; the identical context without the marker emits WARN.
    /// The marker itself is not a field on the line.
    #[test]
    fn log_rpc_failure_shutdown_caused_unavailable_emits_one_info_event() {
        let unmarked = || {
            let mut ctx = RpcLogContext::new("SubscribeReadLayer");
            ctx.session("sess-1");
            ctx.at(RpcStage::Stream);
            ctx
        };
        let mut marked = unmarked();
        marked.shutdown_caused();
        let status = tonic::Status::unavailable("server shutting down");

        let events = capture_json(|| log_rpc_failure(&marked, &status));
        let failures = rpc_failure_events(&events);
        assert_eq!(
            failures.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        assert_eq!(failures[0]["level"], "INFO");
        assert_eq!(failures[0]["fields"]["grpc_code"], "UNAVAILABLE");
        assert_eq!(failures[0]["fields"]["stage"], "stream");
        assert!(
            failures[0]["fields"].get("shutdown_caused").is_none(),
            "the marker selects the level and is not emitted; got {:?}",
            failures[0]
        );

        let events = capture_json(|| log_rpc_failure(&unmarked(), &status));
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1);
        assert_eq!(
            failures[0]["level"], "WARN",
            "without the marker the same UNAVAILABLE is WARN"
        );
    }

    /// A `JoinSession` context marked with the client-keepalive marker emits
    /// the keepalive ending's `UNAVAILABLE` as one INFO `RPC failed` line,
    /// carrying the ending's message; the identical context without the
    /// marker emits WARN. The marker itself is not a field on the line.
    #[test]
    fn log_rpc_failure_client_keepalive_timed_out_unavailable_emits_one_info_event() {
        let unmarked = || {
            let mut ctx = RpcLogContext::new("JoinSession");
            ctx.session("sess-1");
            ctx.at(RpcStage::Stream);
            ctx
        };
        let mut marked = unmarked();
        marked.client_keepalive_timed_out();
        let status = crate::grpc::presence::client_keepalive_timeout_status();

        let events = capture_json(|| log_rpc_failure(&marked, &status));
        let failures = rpc_failure_events(&events);
        assert_eq!(
            failures.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        assert_eq!(failures[0]["level"], "INFO");
        assert_eq!(failures[0]["fields"]["endpoint"], "JoinSession");
        assert_eq!(failures[0]["fields"]["grpc_code"], "UNAVAILABLE");
        assert_eq!(failures[0]["fields"]["stage"], "stream");
        assert_eq!(failures[0]["fields"]["status_message"], status.message());
        assert!(
            failures[0]["fields"]
                .get("client_keepalive_timeout")
                .is_none(),
            "the marker selects the level and is not emitted; got {:?}",
            failures[0]
        );

        let events = capture_json(|| log_rpc_failure(&unmarked(), &status));
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1);
        assert_eq!(
            failures[0]["level"], "WARN",
            "without the marker the same UNAVAILABLE is WARN"
        );
    }

    /// Every optional field an event can carry. `error` is the fault arm's
    /// detail and is present only on that arm.
    const OPTIONAL_FIELDS: &[&str] = &[
        "session_id",
        "layer_name",
        "base_path",
        "object_id",
        "current_object_id",
        "client_id",
        "connection_id",
        "regime",
        "reason",
        "caught_up",
        "rejection",
        "error",
    ];

    /// A `None` identifier is ABSENT from the event, not present-and-null.
    ///
    /// Load-bearing for the populate-after-validate ordering: "the handler
    /// never validated this field" and "the field was empty" must not read the
    /// same way in a log aggregator, and a `null` would let an unvalidated
    /// field appear to have been recorded.
    #[test]
    fn log_rpc_failure_omits_unrecorded_identifiers_rather_than_emitting_null() {
        let events = capture_json(|| {
            log_rpc_failure(
                &RpcLogContext::new("UpdateLayer"),
                &tonic::Status::invalid_argument("session_id must not be empty"),
            );
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1);
        let fields = &failures[0]["fields"];
        for absent in OPTIONAL_FIELDS {
            assert!(
                fields.get(absent).is_none(),
                "`{absent}` must be absent from a context that never recorded it; got {fields:?}"
            );
        }
        assert_eq!(failures[0]["level"], "INFO");
        assert_eq!(fields["stage"], "header");
        assert_eq!(fields["grpc_code"], "INVALID_ARGUMENT");
    }

    /// Each streaming / interceptor field is emitted under its documented
    /// name, with its recorded value.
    ///
    /// The names are the join keys: `regime` and `reason` match the
    /// `live_subscribe_regime_total{regime}` and
    /// `live_edit_subscriber_stream_exits_total{reason}` label names, so a
    /// renamed field silently breaks every query that joins the line to the
    /// series. `caught_up` must be a JSON boolean, not the string `"true"`.
    #[test]
    fn log_rpc_failure_emits_each_streaming_field_under_its_documented_name() {
        let events = capture_json(|| {
            let mut ctx = RpcLogContext::new("SubscribeReadLayer");
            ctx.at(RpcStage::Stream);
            ctx.regime("in_window");
            ctx.exit_reason("slow_reader_deadline");
            ctx.caught_up(true);
            ctx.connection_id("conn-9");
            ctx.rejection("empty_token");
            log_rpc_failure(&ctx, &tonic::Status::unavailable("slow reader"));
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1, "got {events:?}");
        let fields = &failures[0]["fields"];
        assert_eq!(fields["stage"], "stream");
        assert_eq!(fields["regime"], "in_window");
        assert_eq!(fields["reason"], "slow_reader_deadline");
        assert_eq!(fields["caught_up"], serde_json::Value::Bool(true));
        assert_eq!(fields["connection_id"], "conn-9");
        assert_eq!(fields["rejection"], "empty_token");
        assert!(
            fields.get("exit_reason").is_none(),
            "the exit reason is emitted as `reason`, never under its Rust field name; got \
             {fields:?}"
        );
    }

    /// An `UpdateLayer` call summary recorded on the context is emitted as the
    /// commit line's field set: `attempt`, `path`, each conflict kind under its
    /// own kind's field, `records`, `bytes`.
    ///
    /// Every kind carries a distinct non-zero count, so two kinds' fields
    /// swapped in the emitter — or one kind read as its neighbour — reads a
    /// wrong number on at least one field. A failed write is joined to the
    /// large-read fixtures by these names (Spec SDD SR-2.6, SR-1.2).
    #[test]
    fn log_rpc_failure_emits_each_update_layer_summary_field_under_its_own_name() {
        use crate::observability::metrics::UPDATE_LAYER_CONFLICT_KINDS;

        let conflicts: Vec<(&str, u64)> = UPDATE_LAYER_CONFLICT_KINDS
            .iter()
            .enumerate()
            .map(|(i, kind)| (*kind, i as u64 + 1))
            .collect();
        let summary =
            UpdateLayerCallSummary::for_test(5, "watch", &conflicts, Some(13), Some(4_321));
        let events = capture_json(|| {
            let mut ctx = ctx_with_everything();
            ctx.at(RpcStage::Storage);
            ctx.update_layer_call(Some(summary));
            log_rpc_failure(&ctx, &tonic::Status::unavailable("attempts exhausted"));
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1, "got {events:?}");
        let fields = &failures[0]["fields"];
        assert_eq!(fields["attempt"], 5, "{fields}");
        assert_eq!(fields["path"], "watch", "{fields}");
        assert_eq!(fields["records"], 13, "{fields}");
        assert_eq!(fields["bytes"], 4_321, "{fields}");
        for (kind, want) in &conflicts {
            assert_eq!(
                fields[*kind], *want,
                "the failure line must carry `{kind}`'s count under the `{kind}` field; {fields}"
            );
        }
    }

    /// `caught_up = false` is emitted as `false`, not dropped: absent means
    /// "not a subscription tail", false means "the client left before its
    /// first caught-up ack", and the second is the load-test shape the field
    /// exists to show.
    #[test]
    fn log_rpc_failure_emits_caught_up_false_rather_than_omitting_it() {
        let events = capture_json(|| {
            let mut ctx = RpcLogContext::new("SubscribeReadLayer");
            ctx.caught_up(false);
            log_rpc_failure(&ctx, &tonic::Status::cancelled("client disconnected"));
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1, "got {events:?}");
        assert_eq!(
            failures[0]["fields"]["caught_up"],
            serde_json::Value::Bool(false)
        );
    }

    /// The two statuses the conversion builds for a server fault, one per
    /// `ServiceError` variant that maps to `INTERNAL`. Both must carry the
    /// same `InternalFault` source; each is paired with its detail.
    fn server_fault_statuses() -> Vec<(&'static str, tonic::Status)> {
        const INTERNAL_DETAIL: &str = "rpc-log-internal-detail WRONGTYPE le:{7}:obj:hdr";
        const REFUSAL_DETAIL: &str = "rpc-log-refusal-detail WRONGTYPE_KEY: le:{7}:obj:secs";
        vec![
            (
                INTERNAL_DETAIL,
                crate::error::ServiceError::Internal(INTERNAL_DETAIL.to_owned()).into(),
            ),
            (
                REFUSAL_DETAIL,
                crate::error::ServiceError::PermanentKeyspaceRefusal {
                    refusal: crate::error::KeyspaceRefusal::WrongType,
                    detail: REFUSAL_DETAIL.to_owned(),
                }
                .into(),
            ),
        ]
    }

    /// A server fault emits exactly ONE line — ERROR `internal error` with the
    /// un-redacted detail and every request field — and no `RPC failed`; and
    /// dropping the status afterwards adds nothing.
    ///
    /// Three regressions in one: the fault arm skipped (the table's WARN `RPC
    /// failed` would appear and the detail would be lost), the fault not marked
    /// reported (its `Drop` fallback would add a second, context-free ERROR when
    /// the status goes away — the drop is inside the capture for exactly that
    /// reason), and the context fields missing from the ERROR (the line
    /// operators page on would again be the one with the least context).
    #[test]
    fn log_rpc_failure_server_fault_emits_one_error_with_detail_and_context_and_no_fallback() {
        for (detail, status) in server_fault_statuses() {
            let completion = Arc::new(RpcCompletion::new("UpdateLayer"));
            let events = capture_json(|| {
                let mut ctx = ctx_with_everything();
                ctx.completion = Some(Arc::clone(&completion));
                ctx.at(RpcStage::Storage);
                log_rpc_failure(&ctx, &status);
                drop(status);
            });

            assert!(
                rpc_failure_events(&events).is_empty(),
                "a server fault must not also log `{RPC_FAILURE_MSG}`; got {events:?}"
            );
            assert_eq!(
                events.len(),
                1,
                "a server fault must produce exactly one line, including after the status drops \
                 (a second one is the unreported-fault fallback); got {events:?}"
            );
            let event = &events[0];
            assert_eq!(event["level"], "ERROR");
            let fields = &event["fields"];
            assert_eq!(fields["message"], INTERNAL_ERROR_MSG);
            assert_eq!(
                fields["error"], detail,
                "the un-redacted detail rides the line"
            );
            assert_eq!(fields["grpc_code"], "INTERNAL");
            assert_eq!(
                fields["status_message"], INTERNAL_ERROR_MSG,
                "status_message is the redacted wire text, never the detail"
            );
            assert_eq!(fields["endpoint"], "UpdateLayer");
            assert_eq!(fields["stage"], "storage");
            assert_eq!(fields["session_id"], "sess-1");
            assert_eq!(fields["layer_name"], "layer-1");
            assert_eq!(fields["base_path"], "/stage");
            assert_eq!(fields["client_id"], "client-1");
            assert_eq!(fields["object_id"], 41);
            assert_eq!(fields["current_object_id"], 42);
            assert!(
                completion.is_reported(),
                "the fault arm must mark the ledger, or the backstop logs the INTERNAL again"
            );
        }
    }

    /// An `INTERNAL` with no fault source — tonic's own — follows the table:
    /// exactly one WARN `RPC failed`, no ERROR, no `error` field.
    #[test]
    fn log_rpc_failure_internal_without_a_fault_emits_one_warn_rpc_failed() {
        let events = capture_json(|| {
            log_rpc_failure(
                &ctx_with_everything(),
                &tonic::Status::internal("failed to decode Protobuf message"),
            );
        });
        assert_eq!(events.len(), 1, "exactly one line; got {events:?}");
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1, "got {events:?}");
        assert_eq!(failures[0]["level"], "WARN");
        assert_eq!(failures[0]["fields"]["grpc_code"], "INTERNAL");
        assert_eq!(
            failures[0]["fields"]["status_message"],
            "failed to decode Protobuf message"
        );
        assert!(
            failures[0]["fields"].get("error").is_none(),
            "only the fault arm carries `error`"
        );
    }

    /// A source that is not an `InternalFault` does not reach the fault arm.
    ///
    /// The arm keys on the concrete type by downcast, so an `INTERNAL` carrying
    /// some other error as its source (tonic attaches transport errors that
    /// way) is still client-causable and must stay off ERROR.
    #[test]
    fn log_rpc_failure_internal_with_a_foreign_source_emits_one_warn_rpc_failed() {
        let mut status = tonic::Status::internal("h2 protocol error");
        status.set_source(Arc::new(std::io::Error::other("foreign")));
        let events = capture_json(|| log_rpc_failure(&ctx_with_everything(), &status));
        assert_eq!(events.len(), 1, "got {events:?}");
        assert_eq!(events[0]["level"], "WARN");
        assert_eq!(events[0]["fields"]["message"], RPC_FAILURE_MSG);
    }

    /// Every emission marks the ledger; `OK`, which emits nothing, does not.
    ///
    /// The metrics layer's backstop logs a non-OK status exactly when the
    /// ledger is unmarked, so this is the whole of the "never a second line"
    /// guarantee on the logging side: a code that emitted without marking
    /// would be logged again at `stage=transport`.
    #[test]
    fn log_rpc_failure_marks_the_ledger_exactly_when_it_emits() {
        for &code in ALL_CODES {
            let completion = Arc::new(RpcCompletion::new("StatLayer"));
            let ctx = RpcLogContext::for_completion(Arc::clone(&completion));
            let events = capture_json(|| log_rpc_failure(&ctx, &tonic::Status::new(code, "msg")));
            assert_eq!(
                completion.is_reported(),
                !events.is_empty(),
                "code={code:?}: the ledger must be marked iff a line was emitted; got {events:?}"
            );
            assert_eq!(
                completion.is_reported(),
                code != Code::Ok,
                "code={code:?}: every non-OK code emits and marks"
            );
        }
    }

    /// A context with no ledger emits normally: a handler driven directly (no
    /// metrics layer, as in every integration test and the interceptor's unit
    /// tests) must still log.
    #[test]
    fn log_rpc_failure_without_a_ledger_still_emits() {
        let events = capture_json(|| {
            log_rpc_failure(
                &RpcLogContext::new("DeleteLayer"),
                &tonic::Status::not_found("layer gone"),
            )
        });
        assert_eq!(rpc_failure_events(&events).len(), 1, "got {events:?}");
    }

    /// The commit-path sentinel shape is WARN, and says so at `stage=storage`.
    #[test]
    fn log_rpc_failure_commit_path_sentinel_emits_one_warn_event() {
        let events = capture_json(|| {
            let mut ctx = ctx_with_everything();
            ctx.at(RpcStage::Storage);
            log_rpc_failure(
                &ctx,
                &tonic::Status::failed_precondition(
                    "object header partial (seq missing or malformed)",
                ),
            );
        });
        let failures = rpc_failure_events(&events);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0]["level"], "WARN");
        assert_eq!(failures[0]["fields"]["stage"], "storage");
        assert_eq!(failures[0]["fields"]["grpc_code"], "FAILED_PRECONDITION");
    }

    /// The `grpc_code` field carries the same slug the metric label does, for
    /// every code that produces an event.
    ///
    /// The join between `grep 'RPC failed'` and
    /// `live_endpoint_calls_total{grpc_code=…}` is the whole reason the field
    /// is a slug rather than a `Debug` rendering of `Code`.
    #[test]
    fn log_rpc_failure_grpc_code_field_matches_the_metric_label_vocabulary() {
        for &code in ALL_CODES {
            if failure_level(code, RpcStage::Header, false, false, false).is_none() {
                continue;
            }
            let want =
                crate::observability::metrics::grpc_code_slug(Some(&(code as i32).to_string()));
            let events = capture_json(|| {
                log_rpc_failure(
                    &RpcLogContext::new("StatLayer"),
                    &tonic::Status::new(code, "msg"),
                );
            });
            let failures = rpc_failure_events(&events);
            assert_eq!(
                failures.len(),
                1,
                "code={code:?} must emit exactly one event"
            );
            assert_eq!(
                failures[0]["fields"]["grpc_code"], want,
                "code={code:?} must be labelled with the metric slug"
            );
        }
    }

    /// One call emits at most one event of ANY message, for every code in the
    /// matrix — exactly one for every non-OK code.
    ///
    /// The regression fence against a future second emit site (e.g. a wrapper
    /// added around an already-wrapped handler): double emission would silently
    /// double every `RPC failed` count an operator greps. Counted over all
    /// events, not only `RPC failed`, so a second line under a different
    /// message is caught too.
    #[test]
    fn log_rpc_failure_emits_at_most_one_event_per_call_for_every_code() {
        for &code in ALL_CODES {
            let events = capture_json(|| {
                log_rpc_failure(&ctx_with_everything(), &tonic::Status::new(code, "msg"));
            });
            let want =
                usize::from(failure_level(code, RpcStage::Resolve, true, false, false).is_some());
            assert_eq!(
                events.len(),
                want,
                "code={code:?} must emit exactly {want} event(s); got {events:?}"
            );
            assert_eq!(
                rpc_failure_events(&events).len(),
                want,
                "code={code:?}: that event must be `{RPC_FAILURE_MSG}`; got {events:?}"
            );
        }
    }
}
