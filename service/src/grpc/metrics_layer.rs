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

// Tower Layer + Service that wraps every gRPC RPC to:
//   1. Increment the endpoint call counter.
//   2. Record endpoint duration in microseconds.
//   3. Enforce a per-RPC deadline (SDD §5.3).
//   4. Keep the per-request completion ledger (`rpc_log::RpcCompletion`, SDD
//      §17.2.1): inserted into the request extensions before the inner
//      service runs, marked by whichever completion point logs a failure.
//      This layer is the outermost one the service owns, so it logs, exactly
//      once, what nothing inside reported — a non-OK status tonic produced
//      itself (the transport backstop), and a call torn down before any
//      response existed (the drop guard, `RPC abandoned before completion`).
//   5. Open the call's one INFO server span (`grpc.server`) for every
//      `LiveService` RPC, and no other request. It joins the caller's trace
//      when the request carries a W3C `traceparent`, covers the auth
//      interceptor and the handler future, and ends where the call's metric
//      is recorded, carrying the same status — so a stream's span ends with
//      the stream. Only trace export sees it: the log layer never sees spans.
//      See `rpc_server_span`.
//
// tonic's own `RecoverError` and `GrpcTimeout` layers wrap this one inside
// `Server` and cannot be moved inside it. When a client's `grpc-timeout` fires
// first, `GrpcTimeout` drops this layer's future and `RecoverError` answers
// `CANCELLED "Timeout expired"`; when
// the client resets the stream first, hyper drops the whole chain and nothing
// answers. The drop guard sees both, so both count as `CANCELLED`.
//
// Deadline rules:
//   - ListSessions            → config.session_list_timeout_ms (2 s default)
//   - JoinSession             → exempt (bidirectional streaming)
//   - SubscribeReadLayer      → exempt (server streaming)
//   - All other RPCs          → config.rpc_deadline_ms (30 s default)
//
// ── Mid-flight cancellation contract (b.yis sub-item 4) ──────────────────────
//
// When the deadline fires, `tokio::time::timeout` drops the wrapped handler
// future. A dropped future cancels all in-progress `.await`s without running
// any unwind code, so any Valkey mutations pending at the point of drop are
// abandoned in whatever state the server actually observed. This wrapper
// does NOT propagate a `CancellationToken` into handlers (the alternative
// would require plumbing a token through every unary handler and every
// storage method — invasive, high-churn, and unjustified given the audit
// below).
//
// Instead, every unary mutating RPC is engineered so that a mid-flight drop
// leaves at worst a self-healing inconsistency — never a client-visible
// half-commit. The contract each handler honors:
//
//   CreateSession — Non-idempotent (generates a fresh UUID4). Writes
//     path_index inside MULTI/EXEC, then session_meta outside (on a different
//     Cluster slot). A drop between EXEC and HSET leaves an orphan
//     path_index entry; `SessionStore::list` opportunistically SREMs orphans
//     on read, so the inconsistency self-heals on the next `ListSessions`
//     for that base_path.
//
//   DeleteSession — Idempotent (second call returns NOT_FOUND). Post-b.yis
//     ordering is SREM path_index → MULTI{DEL session_meta, DEL
//     layers_index}, so a drop leaves at worst orphan session_meta +
//     layers_index on a session_id no longer in any path-index — invisible
//     to `list`. Retries are safe because the empty-HGETALL case returns
//     NOT_FOUND.
//
//   CreateLayer — Without a client operation_id, non-idempotent (every
//     call that commits allocates a fresh object_id). With one, a re-send of
//     a create that committed is answered ALREADY_COMMITTED from the
//     `le:{sid}:createop:{operation_id}` record and allocates nothing.
//     External values are stored first (ref-counted, GC'd by the background
//     sweep if the layer is never committed), then the new object is
//     materialized in one MULTI/EXEC; the mapping flip, the layers-index
//     entry and the operation record are one atomic script after it. A drop
//     before that script leaves ext-values the refcount GC sweep reclaims and
//     an object no mapping names, which the orphan sweep reclaims.
//
//   UpdateLayer — Without a client operation_id, non-idempotent across calls
//     (every landed commit advances sequence_num); with one, exactly-once
//     across a client's re-sends (below). Under this deadline: only
//     JoinSession and SubscribeReadLayer are exempt. Each commit is atomic on
//     both backends (one Lua script, or one WATCH/MULTI/EXEC): the header
//     update, the XADD stream entry and the field/sample writes land as a unit, or
//     nothing is persisted (save the slow path's torn-EXEC residual, SDD
//     §7.5.2, which answers non-retriable INTERNAL). Within ONE call the
//     in-process retries commit at most once: every attempt first checks the
//     call's commit-token record (`le:{oid}:obj:wtok`, written with the
//     commit) and reports an already-landed commit instead of re-applying
//     it. A drop that fires before fred has written the commit leaves
//     nothing behind, so a client retry is a clean first apply: a commit
//     still queued in fred — in its command channel, behind a reconnect, or
//     in a pipeline the router has not finished writing — is never written
//     once its call is dropped (the vendored fred's change 9). Dropping the
//     future does not recall a commit fred already wrote, though: a drop
//     after the `EVALSHA` / `EXEC` was written can still land, and so can one
//     that fires after the commit landed (fred never writes either a second
//     time: both commits go out with one attempt). Either way the client
//     sees DEADLINE_EXCEEDED for a write that is (or will be) in the store,
//     and its retry is a new call. A retry carrying the same client
//     operation_id (and bytes, before that client_id's next write to the
//     object) finds the commit recorded under `op.<operation_id>` and is
//     answered ALREADY_COMMITTED, applying nothing; and no attempt of an
//     unchained operation_id call applies after that client_id's next commit
//     — a late attempt finds the field moved from the call's start record and
//     answers UNAVAILABLE (overtaken), itself applying nothing (a chained call
//     is never overtaken; see below). A retry without one applies
//     the delta again at the next
//     sequence_num: field-only deltas converge; a repeated CreateSection is
//     refused INVALID_ARGUMENT for an edit that landed. Ext-values stored
//     before the commit are refcount-safe via the GC sweep.
//
//     A chained call (`UpdateLayerRequest.chained_header`) dropped while it
//     waits in the chain gate for its predecessor — a client cancel, a lost
//     connection or this deadline — applies nothing and never will: the gate
//     runs inside the handler future, spawns nothing, and precedes the parse,
//     the external-value store and every commit attempt. One whose commit
//     fred had already written when it was dropped can still land (one fred
//     had not yet written never is), but only directly after its
//     predecessor: the commit's own
//     predecessor guard refuses it otherwise, writing nothing
//     (`storage::chain_gate`, "The commit-time guard").
//
//   DeleteLayer — Idempotent (returns NOT_FOUND on second call). Atomic
//     MULTI/EXEC; any ext-value orphans self-heal via refcount GC.
//
// Retry exhaustion. No retry exhaustion reaches the wire as ABORTED
// (`src/error.rs`; the service's one ABORTED is a chained UpdateLayer's
// predecessor refusal, from the chain gate before any attempt or from the
// commit on the call's first attempt — never after an in-process retry,
// which answers it a plain UNAVAILABLE instead): when an in-process
// retry budget runs out — or a single-attempt site returns the in-loop
// `Aborted` signal — the call answers UNAVAILABLE, which the Kit client
// retries like DEADLINE_EXCEEDED (its retry policy retries DEADLINE_EXCEEDED,
// UNAVAILABLE, RESOURCE_EXHAUSTED and INTERNAL; the inspected revision and
// line anchors: `docs-internal/cross-repo-debugging.md` § Kit client retry and
// reconnect behaviour). What that client retry means for each mutating RPC the
// exhaustion reaches:
//
//   UpdateLayer — without an operation_id, can re-apply a landed delta.
//     Inside the call, an attempt whose outcome was unknowable (a transient
//     `EVALSHA` failure on the Lua fast path, a lost `EXEC` reply whose
//     reconciling XRANGE also failed on the txn path) is retried, and the
//     retry's commit-token check reports the commit if it landed, so the
//     in-process loop commits at most once.
//     But exhaustion can still follow a landed commit — the attempt that
//     landed lost its reply, and every later attempt failed before reading
//     the token record — and the UNAVAILABLE it answers sends the client
//     round again as a NEW call, with a new token, which applies the delta a
//     second time at the next sequence_num. Field-only deltas converge; a
//     repeated CreateSection is refused INVALID_ARGUMENT. A NEW call carrying
//     the same operation_id is instead answered ALREADY_COMMITTED from the
//     landed commit's record.
//
//   CreateSession — no duplicate, but a name can stay held. Failures after
//     the commit was sent undo the attempt first, so the client's retry
//     (the Kit client retries CreateSession through the same retry policy as
//     every other RPC) normally starts clean. Three residuals can
//     make it answer ALREADY_EXISTS:
//       - the undo could not release the name and the commit had landed: the
//         name stays held until the next ListSessions of that base path;
//       - a late `HSET` plus a failed names `SREM`: a complete, never-removed
//         session under an id the client was never told;
//       - a late `EXEC`: a ghost the in-process retry ignores but a later
//         call sees until the next ListSessions.
//     See `undo_create_attempt` (`src/storage/session.rs`) and SDD §7.10.
//
//   DeleteLayer, DeleteSession — idempotent: a retry after a commit that
//     landed answers NOT_FOUND, and a partly-run DeleteSession cascade
//     resumes through its NotFound-skip (`src/grpc/session.rs`).
//
// The invariant sustaining this contract: **every mutating storage path is
// either MULTI/EXEC-atomic or refcount-GC-reclaimable.** Handlers that
// violate this — introducing a new non-atomic, non-self-healing write path
// — must either restructure into MULTI/EXEC or introduce a
// CancellationToken seam. See the per-handler comments in
// `src/grpc/session.rs` and `src/grpc/layer.rs` for the drop-behavior note
// at each call site.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use http::Request;
use http_body::{Body, Frame, SizeHint};
use tonic::{Code, Status};
use tower::{Layer, Service};
use tracing::field::Empty;
use tracing::Instrument as _;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use crate::grpc::deadline;
use crate::grpc::proto::live_service_server::SERVICE_NAME as LIVE_SERVICE_NAME;
use crate::grpc::rpc_log::{
    log_rpc_failure, RpcCompletion, RpcLogContext, RpcStage, RPC_ABANDONED_MSG,
};
use crate::observability::metrics::Metrics;

/// The two long-lived streaming RPCs exempt from the deadline. The
/// client-streaming uploads (`CreateLayer`, `UpdateLayer`) are NOT exempt:
/// they run under the deadline exactly like a unary call.
const EXEMPT_FROM_DEADLINE: &[&str] = &["JoinSession", "SubscribeReadLayer"];

/// The endpoint that uses the shorter ListSessions deadline.
const LIST_SESSIONS_ENDPOINT: &str = "ListSessions";

/// Allowlist of known LiveService RPC method names (b.3z7 H-obs-1).
///
/// The `endpoint_name` function buckets anything outside this list under
/// `"other"` so an unauthenticated attacker probing arbitrary HTTP/2 paths
/// — which reach the metrics layer *before* the auth interceptor — cannot
/// explode the `endpoint` label set and DoS the Prometheus scraper by OOM.
/// Keep this list in sync with the `rpc` declarations in
/// `proto/nvidia/omniverse/live/v1alpha/live_service.proto`; a missing
/// entry collapses an otherwise-distinct label into `"other"` and will
/// only show up as a dashboard regression.
///
/// `pub(crate)` so `observability::metrics::new()` can pre-init every
/// `(endpoint, service, grpc_code)` permutation against the exact same
/// allowlist used at runtime — eliminating the b.avd-medium duplication
/// between the metrics layer's runtime classification and the metrics
/// constructor's pre-init pass.
pub(crate) const KNOWN_ENDPOINTS: &[&str] = &[
    "CreateSession",
    "DeleteSession",
    "ListSessions",
    "JoinSession",
    "CreateLayer",
    "UpdateLayer",
    "DeleteLayer",
    "StatLayer",
    "SubscribeReadLayer",
    "Ping",
];

/// Bucket label applied to any RPC method not in [`KNOWN_ENDPOINTS`].
///
/// `pub(crate)` so the metrics constructor's pre-init pass shares the
/// exact same "other" label name as the runtime classifier — preventing
/// a future rename from silently splitting the bucket across two series.
pub(crate) const OTHER_ENDPOINT: &str = "other";

// ── Layer ────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that installs [`MetricsService`] around each wrapped service.
#[derive(Clone)]
pub struct MetricsLayer {
    metrics: Arc<Metrics>,
    service_label: String,
    rpc_deadline_ms: u64,
    session_list_timeout_ms: u64,
}

impl MetricsLayer {
    pub fn new(
        metrics: Arc<Metrics>,
        service_label: impl Into<String>,
        rpc_deadline_ms: u64,
        session_list_timeout_ms: u64,
    ) -> Self {
        MetricsLayer {
            metrics,
            service_label: service_label.into(),
            rpc_deadline_ms,
            session_list_timeout_ms,
        }
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService {
            inner,
            metrics: Arc::clone(&self.metrics),
            service_label: self.service_label.clone(),
            rpc_deadline_ms: self.rpc_deadline_ms,
            session_list_timeout_ms: self.session_list_timeout_ms,
        }
    }
}

// ── Service ──────────────────────────────────────────────────────────────────

/// Tower [`Service`] that records metrics and enforces RPC deadlines.
#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
    metrics: Arc<Metrics>,
    service_label: String,
    rpc_deadline_ms: u64,
    session_list_timeout_ms: u64,
}

/// Extract the RPC method name from the gRPC request URI path, bucketed
/// against the allowlist (b.3z7 H-obs-1).
///
/// gRPC paths follow the pattern `/<package>.<Service>/<Method>`. The method
/// name is the last path segment. If the extracted segment is in
/// [`KNOWN_ENDPOINTS`] it is returned verbatim; otherwise the function
/// returns [`OTHER_ENDPOINT`] so Prometheus sees a bounded label set even
/// under an attacker-controlled path flood.
///
/// `path.rsplit('/').next()` always returns `Some` — even an empty string
/// yields `Some("")` — so the fallback branch is unreachable. We keep the
/// match arms on the split's output so `""` is classified as `"other"`
/// alongside everything else unknown.
fn endpoint_name(path: &str) -> &'static str {
    let Some(method) = path.rsplit('/').next() else {
        return OTHER_ENDPOINT;
    };
    for known in KNOWN_ENDPOINTS {
        if *known == method {
            return known;
        }
    }
    OTHER_ENDPOINT
}

// ── Per-RPC server span ──────────────────────────────────────────────────────

/// `<service>/<method>` for every [`KNOWN_ENDPOINTS`] method of the
/// `LiveService`: the server span's name and `rpc.method`. Built once so a
/// span costs no allocation.
static LIVE_SERVICE_RPC_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    KNOWN_ENDPOINTS
        .iter()
        .map(|method| format!("{LIVE_SERVICE_NAME}/{method}"))
        .collect()
});

/// The `<service>/<method>` name of a `LiveService` RPC path, or `None` for
/// any other path — health, reflection, the CapabilitiesService, an unknown
/// method, or a known method name under another service.
fn live_service_rpc_name(path: &str) -> Option<&'static str> {
    let name = path.strip_prefix('/')?;
    LIVE_SERVICE_RPC_NAMES
        .iter()
        .find(|known| known.as_str() == name)
        .map(String::as_str)
}

/// gRPC codes that mark a server span as an error, per the OpenTelemetry gRPC
/// semantic conventions for server spans. Every other code, including the
/// caller-caused ones (`INVALID_ARGUMENT`, `NOT_FOUND`, `UNAUTHENTICATED`,
/// ...), leaves the span status unset.
const SERVER_ERROR_CODES: &[&str] = &[
    "UNKNOWN",
    "DEADLINE_EXCEEDED",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
];

/// Open the server span for one request: an INFO `grpc.server` span for a
/// `LiveService` RPC, `Span::none()` for every other path.
///
/// The span is a root (`parent: None`) unless the request carries a valid W3C
/// `traceparent`, whose context then becomes its parent, sampled flag
/// included. It is set before the span is first entered, the only point at
/// which a parent can still be set; a span the trace filter disables ignores
/// it.
///
/// **One span per RPC, whatever the auth outcome.** The span is opened here,
/// before the auth interceptor runs, so an RPC that fails authentication
/// still has exactly its one span. `traceparent` only chooses which trace
/// that span joins and whether it is sampled: it never creates another span,
/// a link or an event, so an unauthenticated caller cannot use it to raise
/// export volume beyond one span per RPC it sends.
///
/// Attributes follow the OpenTelemetry RPC conventions: `rpc.system.name`
/// and `rpc.method` at creation; `rpc.status_code`, and for a server-error
/// code `error.type` and an error status, when the call's metric is recorded
/// ([`record_span_status`]).
pub(crate) fn rpc_server_span(path: &str, headers: &http::HeaderMap) -> tracing::Span {
    let Some(rpc_name) = live_service_rpc_name(path) else {
        return tracing::Span::none();
    };
    let span = tracing::info_span!(
        parent: None,
        "grpc.server",
        otel.name = rpc_name,
        otel.kind = "server",
        rpc.system.name = "grpc",
        rpc.method = rpc_name,
        rpc.status_code = Empty,
        "error.type" = Empty,
        otel.status_code = Empty,
    );
    if !span.is_disabled() {
        if let Some(parent) = crate::observability::remote_parent_context(headers) {
            // The only failure is a span the trace layer does not track (it
            // is filtered out or the layer is absent): nothing to join.
            let _ = span.set_parent(parent);
        }
    }
    span
}

/// Record the call's resolved status on its server span.
///
/// The span is entered for the recording because the trace layer takes a
/// span's end timestamp from its last exit. Without it, a stream's span —
/// closed only when its body finishes — would carry the time its handler
/// future returned the response headers. With it, the end is the recording
/// (or, when the recording runs inside the handler future, that future's
/// return a moment later).
fn record_span_status(span: &tracing::Span, code: &str) {
    if span.is_disabled() {
        return;
    }
    let _entered = span.enter();
    span.record("rpc.status_code", code);
    if SERVER_ERROR_CODES.contains(&code) {
        span.record("error.type", code);
        span.record("otel.status_code", "error");
    }
}

impl<S, ReqBody> Service<Request<ReqBody>> for MetricsService<S>
where
    S: Service<Request<ReqBody>, Response = http::Response<tonic::body::Body>>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
    ReqBody: Send + 'static,
{
    type Response = http::Response<tonic::body::Body>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, mut request: Request<ReqBody>) -> Self::Future {
        let endpoint: &'static str = endpoint_name(request.uri().path());
        let metrics = Arc::clone(&self.metrics);
        let service_label = self.service_label.clone();
        let rpc_deadline_ms = self.rpc_deadline_ms;
        let session_list_timeout_ms = self.session_list_timeout_ms;
        // b.hka (f): read the client's `grpc-timeout` BEFORE the request is
        // moved into the inner service. tonic's own `GrpcTimeout` layer sits
        // outside this one and passes the request through unmodified, so the
        // header is still here and parsing it observes exactly the bound tonic
        // enforces. Purely informational for us — we do not enforce it, we
        // publish it (see `deadline::scope` below) and report it on the drop
        // guard's abandoned line.
        let client_timeout = deadline::parse_grpc_timeout(request.headers());

        // The completion ledger (SDD §17.2.1). Inserted before the inner
        // service runs, so it rides the request extensions through tonic's
        // interceptor into every handler; this layer keeps one clone to read
        // at the backstop.
        let completion = Arc::new(RpcCompletion::new(endpoint));
        request.extensions_mut().insert(Arc::clone(&completion));

        // The call's server span (`rpc_server_span`). Opened before the
        // inner service is called so the auth interceptor's synchronous work
        // runs inside it. One clone instruments the response future; another
        // rides the recording context to whichever exit records the call,
        // so the span closes only once both the future and that exit are
        // done.
        let span = rpc_server_span(request.uri().path(), request.headers());

        // b.avd item 5: tick the inflight gauge up here and pair it with a
        // RAII guard whose Drop decrements. We must decrement on EVERY exit
        // shape: success (unary), unary error, deadline-exceeded synthetic
        // response, body error propagation, future-drop, panic, AND
        // streaming completion (which can outlive the response-headers
        // future).
        //
        // The inflight guard and the per-call recording context start life
        // inside a `CallGuard`, built here rather than in the async block so
        // it exists even if the future is dropped before its first poll. Every
        // exit below disarms it, taking both pieces out; a `CallGuard` that
        // drops still armed is a call abandoned before any response existed,
        // and records itself (see `CallGuard`).
        //
        // Exit-shape ownership map, after disarming:
        //   - Timeout (DEADLINE_EXCEEDED)               → drop in async block
        //   - Inner-service error (no response)         → drop in async block
        //   - Trailers-only / synthetic-headers status  → drop in async block
        //   - Response body already end-of-stream       → drop in async block
        //   - Wrapped streaming/unary body              → move into body
        //
        // The call counter waits until the response trailer / synthetic
        // status resolves — `grpc_code` is not known yet here.
        metrics.inc_inflight_rpc(endpoint, &service_label);
        let inflight = InflightGuard {
            metrics: Arc::clone(&metrics),
            endpoint,
            service_label: service_label.clone(),
        };
        let guard = CallGuard {
            call: Some((
                MetricsBodyCtx {
                    metrics,
                    endpoint,
                    service_label,
                    start: Instant::now(),
                    completion,
                    span: span.clone(),
                },
                inflight,
            )),
            client_timeout,
        };

        let fut = {
            let _entered = span.enter();
            self.inner.call(request)
        };

        let call = async move {
            let result = if EXEMPT_FROM_DEADLINE.contains(&endpoint) {
                // The two exempt streaming RPCs: no timeout — and,
                // deliberately, no deadline scope either. There is no bound
                // to publish, so
                // `deadline::remaining()` answers `None` inside these handlers,
                // which every consumer reads as "unbounded".
                fut.await.map_err(Into::into)
            } else {
                let deadline_ms = if endpoint == LIST_SESSIONS_ENDPOINT {
                    session_list_timeout_ms
                } else {
                    rpc_deadline_ms
                };
                let duration = Duration::from_millis(deadline_ms);

                // b.hka (f): publish the effective deadline to the handler.
                // `min(server, client)` mirrors what tonic's `GrpcTimeout`
                // layer enforces, so a call site sizing a wait against
                // `deadline::remaining()` cannot outlast whichever bound
                // actually fires. The enforcement below is UNCHANGED — it
                // still times out on the server-side `duration` alone; the
                // client half is tonic's to enforce and ours only to observe.
                let effective = match client_timeout {
                    Some(client) => duration.min(client),
                    None => duration,
                };
                let scoped = deadline::scope(tokio::time::Instant::now() + effective, fut);

                match tokio::time::timeout(duration, scoped).await {
                    Ok(inner_result) => inner_result.map_err(Into::into),
                    Err(_elapsed) => {
                        // Timeout: record duration with the synthetic
                        // DEADLINE_EXCEEDED status as `grpc_code`, then
                        // return the same status to the caller. The call
                        // counter ticks on the timeout path too, which
                        // preserves the "every accepted RPC counts"
                        // invariant. `inflight` drops right after the
                        // record, decrementing the gauge.
                        let (ctx, inflight) = guard.disarm();
                        let elapsed_us = ctx.start.elapsed().as_micros() as f64;
                        // The timeout DROPS the handler future, so its wrapper
                        // never runs and no `RpcLogContext` is ever read — the
                        // completion line has to be emitted here, and it
                        // carries only what this layer honestly knows (no
                        // request fields; the layer has not parsed the request
                        // body). It is this path's own line, so the transport
                        // backstop does not run here. Derived from the same
                        // `elapsed_us` sample the histogram observes below, not
                        // a second `start.elapsed()` — the logged latency and
                        // the metric cannot disagree.
                        tracing::warn!(
                            endpoint,
                            deadline_ms,
                            elapsed_ms = (elapsed_us / 1000.0) as u64,
                            "RPC deadline exceeded"
                        );
                        ctx.record_at("DEADLINE_EXCEEDED", elapsed_us);
                        drop(inflight);
                        let status = Status::deadline_exceeded("rpc deadline exceeded");
                        return Ok(status.into_http());
                    }
                }
            };

            // Two shapes for the eventual `grpc_code` label:
            //
            // 1. The response carries `grpc-status` in headers — that
            //    happens for synthetic error responses (e.g.
            //    `Status::into_http()` from a Tower interceptor failing
            //    before tonic's body adapter ever materialised) and for
            //    `trailers-only` responses where tonic writes the status
            //    into headers + END_STREAM. Read it synchronously here.
            // 2. The response carries trailers at the end of the body
            //    (the common path for unary success and streaming RPCs).
            //    Wrap the body in `MetricsTrailerBody` so the trailer
            //    frame's grpc-status records the call counter and
            //    duration when the body completes.
            //
            // If the inner service errored before producing a response at
            // all (case `Err(_)`), tonic's outer adapter will synthesise
            // INTERNAL. Record `INTERNAL` here so the counter still ticks
            // — operators expect every accepted RPC to count.
            let (ctx, inflight) = guard.disarm();
            match result {
                Ok(mut response) => {
                    let elapsed_us = ctx.start.elapsed().as_micros() as f64;
                    if let Some(slug) = grpc_code_from_headers(response.headers()) {
                        // Synthetic / trailers-only response. Record now
                        // and drop the inflight guard — the RPC is done
                        // (no body to stream).
                        log_unreported_status(&ctx.completion, response.headers());
                        ctx.record_at(slug, elapsed_us);
                        drop(inflight);
                        return Ok(response);
                    }

                    // Body-end short-circuit: if the response body already
                    // signals end-of-stream (empty `tonic::body::Body`,
                    // unit-tested mock responses, certain interceptor
                    // shapes), no trailer will ever arrive — record `OK`
                    // synchronously so the call counter still ticks. Skip
                    // wrapping in this case since `MetricsTrailerBody`
                    // would just sit there waiting for a frame that never
                    // comes. Drop the inflight guard inline.
                    if response.body().is_end_stream() {
                        ctx.record_at("OK", elapsed_us);
                        drop(inflight);
                        return Ok(response);
                    }

                    // Stream the body to capture the trailing grpc-status.
                    // Pull existing body out and wrap it. The inflight
                    // guard is moved into the body so the gauge stays
                    // incremented until the body completes — critical for
                    // streaming RPCs (`SubscribeReadLayer`, `JoinSession`)
                    // where the response future returns at headers but
                    // the body continues to stream for the subscription
                    // lifetime.
                    let body = std::mem::replace(response.body_mut(), tonic::body::Body::empty());
                    let wrapped = MetricsTrailerBody {
                        inner: body,
                        ctx: Some(ctx),
                        // Move the guard into the body so its Drop releases
                        // the gauge on true stream end (success trailer,
                        // body error, panic, or caller dropping the body
                        // without consuming).
                        inflight,
                    };
                    let wrapped_body = tonic::body::Body::new(wrapped);
                    *response.body_mut() = wrapped_body;
                    Ok(response)
                }
                Err(e) => {
                    let elapsed_us = ctx.start.elapsed().as_micros() as f64;
                    // No response exists, so no completion point saw a
                    // status. Report it unless something already did.
                    if !ctx.completion.is_reported() {
                        let mut log_ctx =
                            RpcLogContext::for_completion(Arc::clone(&ctx.completion));
                        log_ctx.at(RpcStage::Transport);
                        log_rpc_failure(&log_ctx, &Status::internal(e.to_string()));
                    }
                    ctx.record_at("INTERNAL", elapsed_us);
                    drop(inflight);
                    Err(e)
                }
            }
        };

        // Dropping the instrumented future enters the span first, so an
        // abandoned call's `CallGuard` line lands inside it too.
        Box::pin(call.instrument(span))
    }
}

// ── Abandoned-call drop guard ────────────────────────────────────────────────

/// Owns a call's recording context and inflight guard until the call reaches
/// an exit that records it, and records the call itself if it never does.
///
/// Built in `call()`, before the future exists, and moved into it. Every exit
/// of the future — timeout, trailers-only, end-of-stream, the body handoff,
/// inner-service error — calls [`Self::disarm`]. A guard that drops still
/// armed means the future was dropped before any response existed: the client
/// reset the stream, tonic's `GrpcTimeout` fired on the client's
/// `grpc-timeout`, or the connection went away. No completion point saw a
/// status, so this records one — a `CANCELLED` counter tick and duration
/// sample, the same slug `MetricsTrailerBody::drop` records for an abandoned
/// body — and emits one INFO [`RPC_ABANDONED_MSG`] line.
///
/// It is `CANCELLED` whatever the elapsed time: a client reset can land just
/// before its own deadline, so comparing elapsed time against the client's
/// timeout would be a guess. `client_timeout_ms` is on the line so a reader can
/// make that comparison with the caveat in view.
///
/// Once the response exists this guard is gone; a response body dropped later
/// is `MetricsTrailerBody::drop`'s, which records `CANCELLED` and logs nothing.
struct CallGuard {
    /// `Some` while armed. `Drop` records the counter before dropping the
    /// inflight guard — the order the drop-ordering note on
    /// `MetricsTrailerBody` explains.
    call: Option<(MetricsBodyCtx, InflightGuard)>,
    /// The request's `grpc-timeout`, when it carried one.
    client_timeout: Option<Duration>,
}

impl CallGuard {
    /// Take the recording context and inflight guard out; the guard then
    /// drops silently.
    fn disarm(mut self) -> (MetricsBodyCtx, InflightGuard) {
        match self.call.take() {
            Some(call) => call,
            // `call` is only taken here and in `Drop`, and `disarm` consumes
            // the guard, so it cannot run twice.
            None => unreachable!("CallGuard::disarm ran on a disarmed guard"),
        }
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        let Some((ctx, inflight)) = self.call.take() else {
            return;
        };
        let elapsed_us = ctx.start.elapsed().as_micros() as f64;
        tracing::info!(
            endpoint = ctx.endpoint,
            grpc_code = "CANCELLED",
            elapsed_ms = (elapsed_us / 1000.0) as u64,
            client_timeout_ms = self.client_timeout.map(|t| t.as_millis() as u64),
            "{}",
            RPC_ABANDONED_MSG
        );
        ctx.record_at("CANCELLED", elapsed_us);
        drop(inflight);
    }
}

/// The transport backstop: log a non-OK `grpc-status` in `headers` that no
/// completion point reported (SDD §17.2.1).
///
/// What reaches it is what no handler, streaming tail or the auth interceptor
/// ever saw — tonic's own request-decode (`INTERNAL`), message-size
/// (`OUT_OF_RANGE`) and unknown-method (`UNIMPLEMENTED`) rejections, and a
/// response-encode failure. Logged at [`RpcStage::Transport`] through the
/// shared emitter, so the level comes from the same table as every other line.
///
/// The status is rebuilt from `grpc-status` and the percent-decoded
/// `grpc-message` only. `Status::from_header_map` is deliberately not used: it
/// panics on a malformed `grpc-status-details-bin`.
fn log_unreported_status(completion: &Arc<RpcCompletion>, headers: &http::HeaderMap) {
    let Some(raw) = headers.get("grpc-status") else {
        return;
    };
    let code = Code::from_bytes(raw.as_bytes());
    if code == Code::Ok || completion.is_reported() {
        return;
    }
    let message = headers
        .get("grpc-message")
        .map(|value| {
            percent_encoding::percent_decode(value.as_bytes())
                .decode_utf8_lossy()
                .into_owned()
        })
        .unwrap_or_default();
    let mut ctx = RpcLogContext::for_completion(Arc::clone(completion));
    ctx.at(RpcStage::Transport);
    log_rpc_failure(&ctx, &Status::new(code, message));
}

// ── Inflight RAII guard ──────────────────────────────────────────────────────

/// Decrement the in-flight RPC gauge on drop (b.avd item 5). The drop runs
/// regardless of how the future exits — success, deadline-exceeded synthetic
/// response, body error, future-drop on caller cancellation, or panic — so
/// the gauge always rebalances.
struct InflightGuard {
    metrics: Arc<Metrics>,
    endpoint: &'static str,
    service_label: String,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.metrics
            .dec_inflight_rpc(self.endpoint, &self.service_label);
    }
}

// ── Trailer-capturing body wrapper ───────────────────────────────────────────

/// Per-call context the body wrapper consumes when it observes the trailer
/// frame (or end-of-stream without trailers).
struct MetricsBodyCtx {
    metrics: Arc<Metrics>,
    endpoint: &'static str,
    service_label: String,
    start: Instant,
    /// The request's completion ledger, read by the transport backstop when
    /// the trailer frame streams past.
    completion: Arc<RpcCompletion>,
    /// The call's server span (`Span::none()` outside the `LiveService`).
    /// Consuming the context records the status on it and drops this clone,
    /// so the span ends where the call's metric is recorded.
    span: tracing::Span,
}

impl MetricsBodyCtx {
    /// Record the call counter + duration with the resolved `grpc_code`.
    /// Consumed on first call so we never double-count if a misbehaving
    /// body yields trailers more than once.
    fn record(self, code: &str) {
        let elapsed_us = self.start.elapsed().as_micros() as f64;
        self.record_at(code, elapsed_us);
    }

    /// [`Self::record`] with a duration sample the caller already took, for
    /// a path that also logs the same latency. Also records `code` on the
    /// server span ([`record_span_status`]).
    fn record_at(self, code: &str, elapsed_us: f64) {
        self.metrics
            .inc_calls(self.endpoint, &self.service_label, code);
        self.metrics
            .observe_duration_us(self.endpoint, &self.service_label, code, elapsed_us);
        record_span_status(&self.span, code);
    }
}

/// `http_body::Body` wrapper that records the `grpc-status` trailer when
/// it streams past (b.avd item 4) AND keeps the inflight-RPC gauge ticked
/// up for the full body lifetime (b.avd HIGH follow-up). Falls back to
/// `"OK"` (success) when the body ends with no trailers, which matches
/// gRPC's wire-level convention that omitted `grpc-status` means code 0.
///
/// The struct is `Unpin` because both `tonic::body::Body` (used by tonic
/// 0.14 as an internal type) and the per-call context types are `Unpin`.
/// `poll_frame` therefore projects via the safe `self.get_mut()` API —
/// any future change that introduces a `!Unpin` inner type fails to
/// compile loudly rather than silently relying on an `unsafe` projection.
///
/// Drop ordering note (load-bearing): Rust drops the manually-implemented
/// `Drop for MetricsTrailerBody` FIRST, then drops the fields in
/// declaration order: `inner`, `ctx`, `inflight`. The manual `Drop`
/// records the call counter (under `CANCELLED`) for any caller-cancelled
/// streaming body that was dropped before `poll_frame` saw end-of-stream;
/// the `inflight` field's own `Drop` then runs last and decrements the
/// gauge. Keeping `inflight` as the last field guarantees the gauge
/// dec strictly follows the counter record so a Prometheus scrape can
/// never observe an inflight=0, count-not-yet-incremented gap.
struct MetricsTrailerBody {
    inner: tonic::body::Body,
    /// Per-call context. `Some` until the metric is recorded; `None`
    /// afterwards so a buggy body yielding trailers twice does not
    /// double-count.
    ///
    /// On caller cancellation (body dropped before `poll_frame` reaches
    /// `Ready(None)` / `Ready(Some(Err(_)))`) the field stays `Some`
    /// and `Drop for MetricsTrailerBody` records it under `CANCELLED`
    /// before the field's auto-drop fires.
    ctx: Option<MetricsBodyCtx>,
    /// Inflight-RPC gauge guard moved here from the response-headers
    /// future (b.avd HIGH follow-up). Its `Drop` impl decrements the
    /// gauge — for unary responses Drop fires when the consumer drops
    /// the body, and for streaming responses it fires when the
    /// subscription ends (either trailer, error, caller cancellation,
    /// or panic).
    ///
    /// Plain (non-`Option`) because no path takes ownership before
    /// drop — the b.avd second-follow-up flatten removed the
    /// `Option<InflightGuard>` indirection. `#[allow(dead_code)]` is
    /// still required because Rust's dead-code lint flags any never-read
    /// named field, even when its `Drop` impl is the entire reason it
    /// exists. The drop ordering is still load-bearing (see struct-level
    /// note).
    #[allow(dead_code)]
    inflight: InflightGuard,
}

impl Drop for MetricsTrailerBody {
    /// Caller-cancellation safety net (b.avd second follow-up).
    ///
    /// If a streaming-RPC consumer (`SubscribeReadLayer`, `JoinSession`)
    /// drops the body before `poll_frame` ever resolves to
    /// `Ready(None)` / `Ready(Some(Err(_)))`, the `ctx` field still
    /// holds the per-call recording context — meaning `inc_calls` and
    /// `observe_duration_us` would never fire and the RPC would
    /// silently disappear from the call counter. That violates the
    /// "every accepted RPC counts" invariant the metrics layer commits
    /// to on every other exit, `CallGuard`'s abandoned-call path included.
    ///
    /// Record under `"CANCELLED"` (the canonical slug in
    /// [`crate::observability::metrics::GRPC_CODES`]) when we still own
    /// the context. The `inflight` field's own `Drop` impl runs AFTER
    /// this method, decrementing the gauge — see the struct-level
    /// drop-ordering note.
    ///
    /// Deliberately silent: the response existed, so this is a client
    /// leaving an established stream (a closed health `Watch`, a finished
    /// `JoinSession`), not a failure.
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            ctx.record("CANCELLED");
        }
    }
}

impl Body for MetricsTrailerBody {
    type Data = bytes::Bytes;
    type Error = tonic::Status;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // Safe Pin projection — `MetricsTrailerBody` is `Unpin` because
        // every field's type is `Unpin` (tonic::body::Body, Option<_>),
        // so `self.get_mut()` is sound. A future struct change that
        // introduces a `!Unpin` field would fail to compile here.
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(trailers) = frame.trailers_ref() {
                    if let Some(ctx) = this.ctx.take() {
                        let slug = grpc_code_from_headers(trailers).unwrap_or("OK");
                        log_unreported_status(&ctx.completion, trailers);
                        ctx.record(slug);
                    }
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(e))) => {
                if let Some(ctx) = this.ctx.take() {
                    // Body error: surface as INTERNAL. The downstream
                    // transport will synthesise a status from this error
                    // anyway, but our metric needs a recorded code.
                    ctx.record("INTERNAL");
                }
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if let Some(ctx) = this.ctx.take() {
                    // End of stream with no trailers ⇒ OK.
                    ctx.record("OK");
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// Look up the `grpc-status` value in a header/trailer map and translate
/// it via [`grpc_code_slug`]. Returns `None` if the header is absent or
/// not UTF-8; that signals "no recorded status yet" to the caller (which
/// then either records `OK` on clean stream end or falls back to the
/// per-shape default).
fn grpc_code_from_headers(headers: &http::HeaderMap) -> Option<&'static str> {
    let raw = headers.get("grpc-status")?.to_str().ok()?;
    Some(crate::observability::metrics::grpc_code_slug(Some(raw)))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::time::Duration;

    use http::Request;
    use tower::{Layer, Service};

    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};

    use super::MetricsLayer;

    // ── Mock inner service ────────────────────────────────────────────────────

    /// A mock tower Service that sleeps for `delay` then returns an empty OK response.
    #[derive(Clone)]
    struct DelayedOkService {
        delay: Duration,
    }

    impl DelayedOkService {
        fn instant() -> Self {
            DelayedOkService {
                delay: Duration::ZERO,
            }
        }

        fn with_delay(delay: Duration) -> Self {
            DelayedOkService { delay }
        }
    }

    impl<ReqBody> Service<Request<ReqBody>> for DelayedOkService
    where
        ReqBody: Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
            let delay = self.delay;
            Box::pin(async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let response = http::Response::builder()
                    .status(200)
                    .body(tonic::body::Body::empty())
                    .expect("response construction failed");
                Ok(response)
            })
        }
    }

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_metrics_layer(
        metrics: Arc<Metrics>,
        rpc_deadline_ms: u64,
        session_list_timeout_ms: u64,
    ) -> MetricsLayer {
        MetricsLayer::new(
            metrics,
            "test-pod",
            rpc_deadline_ms,
            session_list_timeout_ms,
        )
    }

    fn make_request(grpc_method: &str) -> Request<()> {
        let path = format!("/nvidia.omniverse.live.v1alpha.LiveService/{grpc_method}");
        Request::builder()
            .uri(&path)
            .body(())
            .expect("request construction failed")
    }

    fn make_metrics() -> Arc<Metrics> {
        Arc::new(metrics_scrape_only("test-pod"))
    }

    fn counter_value(metrics: &Metrics, endpoint: &str) -> u64 {
        // b.avd item 4: `endpoint_calls` now carries a `grpc_code` label.
        // The mock inner service in this file returns HTTP 200 with no
        // body / no trailers, so the metrics body wrapper records the
        // call under `grpc_code = "OK"`. Sum across every code so we can
        // assert "this endpoint was called N times" without coupling to
        // the eventual status code each path takes.
        sum_endpoint_calls(metrics, endpoint)
    }

    fn sum_endpoint_calls(metrics: &Metrics, endpoint: &str) -> u64 {
        let text = render(metrics);
        crate::observability::metrics::GRPC_CODES
            .iter()
            .map(|code| {
                sample_value(
                    &text,
                    "live_endpoint_calls_total",
                    &[
                        ("endpoint", endpoint),
                        ("service", "test-pod"),
                        ("grpc_code", code),
                    ],
                )
                .unwrap_or(0.0) as u64
            })
            .sum()
    }

    fn histogram_sample_count(metrics: &Metrics, endpoint: &str) -> u64 {
        let text = render(metrics);
        crate::observability::metrics::GRPC_CODES
            .iter()
            .map(|code| {
                sample_value(
                    &text,
                    "live_endpoint_duration_microseconds_count",
                    &[
                        ("endpoint", endpoint),
                        ("service", "test-pod"),
                        ("grpc_code", code),
                    ],
                )
                .unwrap_or(0.0) as u64
            })
            .sum()
    }

    /// Read a single `live_endpoint_calls_total{endpoint,service,grpc_code}`
    /// cell from the rendered scrape exposition.
    fn endpoint_call_cell(metrics: &Metrics, endpoint: &str, code: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_endpoint_calls_total",
            &[
                ("endpoint", endpoint),
                ("service", "test-pod"),
                ("grpc_code", code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Read a single duration-histogram `_count` cell from rendered exposition.
    fn endpoint_duration_count(metrics: &Metrics, endpoint: &str, code: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_endpoint_duration_microseconds_count",
            &[
                ("endpoint", endpoint),
                ("service", "test-pod"),
                ("grpc_code", code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Read the `live_edit_inflight_rpcs{endpoint,service}` gauge value from the
    /// rendered scrape exposition. The cell is pre-initialised at zero in
    /// `Metrics::new`, so it is present from the first render.
    fn inflight(metrics: &Metrics, endpoint: &str) -> i64 {
        sample_value(
            &render(metrics),
            "live_edit_inflight_rpcs",
            &[("endpoint", endpoint), ("service", "test-pod")],
        )
        .unwrap_or(0.0) as i64
    }

    // ── Counter tests ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn metrics_layer_call_counter_increments_once_per_call() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        let req = make_request("Ping");
        svc.call(req).await.expect("call failed");

        assert_eq!(counter_value(&metrics, "Ping"), 1);
    }

    #[tokio::test]
    async fn metrics_layer_call_counter_increments_n_times_for_n_calls() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        for _ in 0..5 {
            let req = make_request("CreateSession");
            svc.call(req).await.expect("call failed");
        }

        assert_eq!(
            counter_value(&metrics, "CreateSession"),
            5,
            "counter must equal the number of calls made"
        );
    }

    #[tokio::test]
    async fn metrics_layer_distinct_endpoints_have_separate_counter_labels() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        svc.call(make_request("Ping")).await.expect("call failed");
        svc.call(make_request("Ping")).await.expect("call failed");
        svc.call(make_request("CreateSession"))
            .await
            .expect("call failed");

        assert_eq!(counter_value(&metrics, "Ping"), 2, "Ping counter must be 2");
        assert_eq!(
            counter_value(&metrics, "CreateSession"),
            1,
            "CreateSession counter must be 1"
        );
    }

    // ── Histogram tests ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn metrics_layer_histogram_has_observation_after_call() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        svc.call(make_request("Ping")).await.expect("call failed");

        assert_eq!(
            histogram_sample_count(&metrics, "Ping"),
            1,
            "histogram must have exactly one sample after one call"
        );
    }

    #[tokio::test]
    async fn metrics_layer_histogram_sample_count_matches_call_count() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        for _ in 0..3 {
            svc.call(make_request("StatLayer"))
                .await
                .expect("call failed");
        }

        assert_eq!(
            histogram_sample_count(&metrics, "StatLayer"),
            3,
            "histogram sample count must match call count"
        );
    }

    // ── Default deadline tests ────────────────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_default_deadline_exceeded_returns_deadline_exceeded_status() {
        // Default deadline is 30s. The inner service takes 31s of virtual time.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let response = svc
            .call(make_request("CreateSession"))
            .await
            .expect("outer service must not error");

        // tonic encodes DEADLINE_EXCEEDED as HTTP 408 or grpc-status trailer.
        // Status::into_http() sets the grpc-status trailer; the HTTP status is 200.
        // Read the grpc-status header from the response to verify the error.
        let grpc_status = response
            .headers()
            .get("grpc-status")
            .expect("grpc-status header must be present on timeout response")
            .to_str()
            .expect("grpc-status must be valid ASCII");

        // gRPC status code 4 = DEADLINE_EXCEEDED
        assert_eq!(
            grpc_status, "4",
            "expected grpc-status 4 (DEADLINE_EXCEEDED), got {grpc_status}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_default_deadline_not_exceeded_returns_ok() {
        // Default deadline is 30s. The inner service takes 29s of virtual time.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(29_000)));

        let response = svc
            .call(make_request("CreateSession"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            response.status(),
            200,
            "response within deadline must be HTTP 200"
        );
        // No grpc-status header means OK (code 0 is omitted by tonic).
        assert!(
            response.headers().get("grpc-status").is_none(),
            "successful response must not have a grpc-status header"
        );
    }

    // ── ListSessions 2s deadline tests ────────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_list_sessions_2001ms_handler_returns_deadline_exceeded() {
        // ListSessions deadline is 2s. The inner service takes 2001ms.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(2_001)));

        let response = svc
            .call(make_request("ListSessions"))
            .await
            .expect("outer service must not error");

        let grpc_status = response
            .headers()
            .get("grpc-status")
            .expect("grpc-status header must be present when ListSessions exceeds 2s deadline")
            .to_str()
            .expect("grpc-status must be valid ASCII");

        assert_eq!(
            grpc_status, "4",
            "expected grpc-status 4 (DEADLINE_EXCEEDED) for ListSessions at 2001ms, got {grpc_status}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_list_sessions_1999ms_handler_returns_ok() {
        // ListSessions deadline is 2s. The inner service takes 1999ms.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(1_999)));

        let response = svc
            .call(make_request("ListSessions"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            response.status(),
            200,
            "ListSessions within 2s deadline must return HTTP 200"
        );
        assert!(
            response.headers().get("grpc-status").is_none(),
            "ListSessions within deadline must not have a grpc-status error header"
        );
    }

    // ── Streaming RPC deadline exemption tests ────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_join_session_exempt_from_deadline_31s_handler_returns_ok() {
        // JoinSession is exempt. The inner service takes 31s of virtual time.
        // With a 30s default deadline that would normally trigger DEADLINE_EXCEEDED.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let response = svc
            .call(make_request("JoinSession"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            response.status(),
            200,
            "JoinSession must be exempt from deadline and return HTTP 200 even after 31s"
        );
        assert!(
            response.headers().get("grpc-status").is_none(),
            "JoinSession exempt response must not have a grpc-status error header"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_subscribe_read_layer_exempt_from_deadline_31s_handler_returns_ok() {
        // SubscribeReadLayer is exempt. The inner service takes 31s of virtual time.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let response = svc
            .call(make_request("SubscribeReadLayer"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            response.status(),
            200,
            "SubscribeReadLayer must be exempt from deadline and return HTTP 200 even after 31s"
        );
        assert!(
            response.headers().get("grpc-status").is_none(),
            "SubscribeReadLayer exempt response must not have a grpc-status error header"
        );
    }

    // ── Counter still increments on timed-out calls ───────────────────────────

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_counter_increments_even_when_deadline_exceeded() {
        // The call counter is incremented before awaiting the future, so it must
        // be recorded even when the handler exceeds the deadline.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        svc.call(make_request("DeleteSession"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            counter_value(&metrics, "DeleteSession"),
            1,
            "call counter must be incremented even when the handler exceeds the deadline"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn metrics_layer_histogram_records_observation_on_deadline_exceeded() {
        // Duration is recorded in both the success and timeout paths.
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        svc.call(make_request("DeleteLayer"))
            .await
            .expect("outer service must not error");

        assert_eq!(
            histogram_sample_count(&metrics, "DeleteLayer"),
            1,
            "histogram must record a duration sample even when the deadline is exceeded"
        );
    }

    // ── Endpoint-label cardinality (b.3z7 H-obs-1) ────────────────────────────
    //
    // Before the fix, `endpoint_name` returned the last URI path segment
    // verbatim, which meant any unauthenticated attacker could flood the
    // server with arbitrary `/foo/<garbage>` paths and explode the
    // `endpoint` label on `live_endpoint_calls_total` / latency histograms
    // — each series eats hundreds of bytes in Prometheus and is kept
    // forever, so an attacker-controlled unbounded set OOMs the scraper.
    // Post-fix: unknown methods collapse to the `"other"` bucket.

    /// Every known LiveService RPC must round-trip through `endpoint_name`
    /// unchanged. A missing entry would collapse the endpoint into `"other"`
    /// and a perfectly normal client would have its calls misattributed —
    /// this test pins the allowlist so that accidentally removing a method
    /// fails CI before it ships.
    #[test]
    fn endpoint_name_every_known_rpc_round_trips_verbatim() {
        let cases = [
            "CreateSession",
            "DeleteSession",
            "ListSessions",
            "JoinSession",
            "CreateLayer",
            "UpdateLayer",
            "DeleteLayer",
            "StatLayer",
            "SubscribeReadLayer",
            "Ping",
        ];
        for method in cases {
            let path = format!("/nvidia.omniverse.live.v1alpha.LiveService/{method}");
            assert_eq!(
                super::endpoint_name(&path),
                method,
                "known RPC {method} must preserve its method name as the metric label"
            );
        }
    }

    /// Arbitrary attacker-supplied paths collapse to `"other"`.
    #[test]
    fn endpoint_name_unknown_paths_bucket_to_other() {
        // A mix of shapes an attacker might send: empty trailing segment,
        // random garbage, wrong service, and a path with no slashes at all.
        let cases = [
            "/nvidia.omniverse.live.v1alpha.LiveService/DoesNotExist",
            "/foo",
            "/bar/baz/AttackerSuppliedMethod",
            "",                        // empty path → rsplit yields Some("")
            "/",                       // trailing slash → last segment is ""
            "no-leading-slash",        // no slashes → whole string is the segment
            "/..%2F..%2Fetc%2Fpasswd", // url-encoded traversal attempt
        ];
        for path in cases {
            let label = super::endpoint_name(path);
            assert_eq!(
                label, "other",
                "attacker-supplied path {path:?} must bucket to \"other\" (got {label:?})"
            );
        }
    }

    /// End-to-end through the metrics layer: a flood of attacker-controlled
    /// paths must not create a distinct label per path. After N requests
    /// against N different unknown paths, the `other` counter is N and no
    /// other label appears anywhere in the registry's gathered output.
    #[tokio::test]
    async fn metrics_layer_attacker_path_flood_buckets_under_other() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        // Fire N requests against N distinct synthetic paths. N=50 is
        // enough to distinguish "one bucket" from "one series per path" but
        // stays fast.
        const ATTACK_COUNT: usize = 50;
        for i in 0..ATTACK_COUNT {
            let path = format!("/{i}");
            let req = http::Request::builder()
                .uri(&path)
                .body(())
                .expect("request construction failed");
            svc.call(req).await.expect("call failed");
        }

        // All ATTACK_COUNT calls must have landed on the `other` bucket.
        assert_eq!(
            counter_value(&metrics, "other"),
            ATTACK_COUNT as u64,
            "every attacker-supplied path must count under `other`"
        );

        // No known-RPC label should have been incremented by the attack —
        // the counter for every real RPC must still be zero.
        for known in &[
            "CreateSession",
            "DeleteSession",
            "ListSessions",
            "JoinSession",
            "CreateLayer",
            "UpdateLayer",
            "DeleteLayer",
            "StatLayer",
            "SubscribeReadLayer",
            "Ping",
        ] {
            assert_eq!(
                counter_value(&metrics, known),
                0,
                "attack flood must not tick the counter for known RPC {known}"
            );
        }

        // Belt and suspenders: gather the whole registry and assert the
        // `endpoint_calls` series set contains only known labels + `other`.
        // This proves the attacker-controlled paths never became series
        // names in Prometheus — which is the actual DoS shape the fix
        // prevents. b.avd item 4 added a third `grpc_code` label; the
        // attacker-flood assertion still keys on `endpoint` because that
        // is the attacker-controlled dimension. `grpc_code` cardinality
        // is bounded by `GRPC_CODES` regardless of input.
        let text = render(&metrics);
        let allowed: std::collections::HashSet<&str> = [
            "CreateSession",
            "DeleteSession",
            "ListSessions",
            "JoinSession",
            "CreateLayer",
            "UpdateLayer",
            "DeleteLayer",
            "StatLayer",
            "SubscribeReadLayer",
            "Ping",
            "other",
        ]
        .into_iter()
        .collect();
        // Every rendered series line for the two endpoint-keyed families must
        // carry an `endpoint="..."` label whose value is inside the allowlist —
        // i.e. no attacker-controlled path became a Prometheus series.
        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            if !(line.starts_with("live_endpoint_calls_total")
                || line.starts_with("live_endpoint_duration_microseconds"))
            {
                continue;
            }
            let Some(start) = line.find("endpoint=\"") else {
                continue;
            };
            let rest = &line[start + "endpoint=\"".len()..];
            let value = &rest[..rest.find('"').expect("endpoint label must be quoted")];
            assert!(
                allowed.contains(value),
                "series label value {value:?} on line {line:?} is outside the allowlist — \
                 attacker-controlled paths escaped into Prometheus",
            );
        }
    }

    // ── b.avd item 4: grpc_code label on endpoint metrics ──────────────

    /// First successful call must record under `grpc_code = "OK"` on
    /// `live_endpoint_calls_total`. Acceptance criterion 3 from b.avd:
    /// "First gRPC call's metrics show `grpc_code` label."
    #[tokio::test]
    async fn metrics_layer_first_call_records_grpc_code_label_ok() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        svc.call(make_request("Ping")).await.expect("call failed");

        let ok = endpoint_call_cell(&metrics, "Ping", "OK");
        assert_eq!(
            ok, 1,
            "first Ping call must record under grpc_code=OK (b.avd item 4)"
        );
    }

    /// A deadline-exceeded synthetic response must record under
    /// `grpc_code = "DEADLINE_EXCEEDED"`. This is the timeout path through
    /// the metrics layer; pre-fix it shared the unlabelled call counter
    /// with success traffic.
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_timeout_records_grpc_code_deadline_exceeded() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        svc.call(make_request("CreateSession"))
            .await
            .expect("outer service must not error");

        let de = endpoint_call_cell(&metrics, "CreateSession", "DEADLINE_EXCEEDED");
        assert_eq!(
            de, 1,
            "timeout path must record under grpc_code=DEADLINE_EXCEEDED (b.avd item 4)"
        );
        // The OK label cell for the same endpoint must NOT be ticked.
        let ok = endpoint_call_cell(&metrics, "CreateSession", "OK");
        assert_eq!(
            ok, 0,
            "timeout path must not tick the OK cell — operator dashboards rely on this split"
        );
    }

    // ── b.94g: the deadline completion log line ────────────────────────
    //
    // DEADLINE_EXCEEDED is the one non-OK completion the handler-side failure
    // boundary (`super::super::rpc_log`) structurally cannot report: the
    // timeout DROPS the handler future, so its wrapper never runs. Before
    // b.94g the timeout was therefore a metric tick and nothing else — the
    // same operability hole the ticket was filed for, one layer up. These
    // tests pin the replacement line and, just as importantly, pin that the
    // handler boundary stays silent here rather than double-reporting.

    /// Sink for [`capture_json_events`]. Local copy of the tree-wide
    /// `CapturingWriter` idiom (`src/checkpoint/assembler.rs::tests`).
    #[derive(Clone)]
    struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

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

    /// Drive `f` under a JSON-formatting capture subscriber and return every
    /// event it emitted, parsed.
    ///
    /// JSON so field presence can be asserted structurally rather than by
    /// substring — `tracing` records nothing for an absent field, so a missing
    /// key and an empty value are different things and must stay so.
    ///
    /// `WithSubscriber` rather than `set_default`: the layer's future awaits
    /// across a `tokio::time::timeout`, and a thread-local dispatcher can stop
    /// capturing at an `.await` that resumes elsewhere — the failure mode that
    /// reads as "the WARN was never emitted".
    async fn capture_json_events<F, T>(f: F) -> (T, Vec<serde_json::Value>)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let value = f.with_subscriber(tracing::Dispatch::new(subscriber)).await;
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let events = String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect();
        (value, events)
    }

    /// Events whose message equals `msg`.
    fn events_with_message<'a>(
        events: &'a [serde_json::Value],
        msg: &str,
    ) -> Vec<&'a serde_json::Value> {
        events
            .iter()
            .filter(|e| e["fields"]["message"] == msg)
            .collect()
    }

    /// A timed-out RPC emits exactly one WARN naming the endpoint and the
    /// deadline it blew, alongside exactly one counter tick.
    ///
    /// Neither half alone is the operability contract: the metric says an
    /// endpoint is timing out, the line says which budget and by how much. A
    /// change that emitted the line without the tick — or either of them twice
    /// per timeout — would desynchronise a dashboard from the log an operator
    /// drills into from it, so both are counted here in one place.
    ///
    /// The `elapsed_ms` *value* is deliberately not asserted — it is a latency,
    /// and the test-writing guide bars wall-clock assertions from the gating
    /// suite (this test runs on a paused clock, so the value is near zero
    /// regardless). Its presence is the contract: without it the line says an
    /// RPC timed out but not by how much, which is the difference between "the
    /// backend is slow" and "the backend is wedged".
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_timeout_emits_one_deadline_exceeded_warn_with_endpoint_and_budget() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let (result, events) = capture_json_events(svc.call(make_request("CreateSession"))).await;
        result.expect("outer service must not error");

        let warns = events_with_message(&events, "RPC deadline exceeded");
        assert_eq!(
            warns.len(),
            1,
            "a timed-out RPC must emit exactly one deadline line; got {events:?}"
        );
        assert_eq!(warns[0]["level"], "WARN");
        assert_eq!(
            warns[0]["fields"]["endpoint"], "CreateSession",
            "the `endpoint` field must carry the same value as the \
             live_endpoint_calls_total{{endpoint}} label so the line and the series join"
        );
        assert_eq!(
            warns[0]["fields"]["deadline_ms"], 30_000,
            "the line must name the budget that was blown, not just that one was"
        );
        assert!(
            warns[0]["fields"]["elapsed_ms"].is_u64(),
            "`elapsed_ms` must be present and numeric; got {:?}",
            warns[0]["fields"]
        );
        assert_eq!(
            endpoint_call_cell(&metrics, "CreateSession", "DEADLINE_EXCEEDED"),
            1,
            "exactly one DEADLINE_EXCEEDED tick must accompany the one line"
        );
    }

    /// The timeout emits the deadline line and nothing else — not `RPC
    /// failed` (from the backstop reading the synthetic `DEADLINE_EXCEEDED`),
    /// and not `RPC abandoned before completion` (from the drop guard, whose
    /// inner handler future the timeout drops).
    ///
    /// Both neighbours sit on this path: the timeout returns a trailers-only
    /// `DEADLINE_EXCEEDED` with the ledger unmarked, which is exactly the
    /// backstop's trigger condition, and it drops the handler future, which is
    /// exactly the drop guard's. The server-deadline path owning its one line
    /// is what keeps "exactly one line per failure" true here.
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_timeout_emits_only_its_own_deadline_line() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let (result, events) = capture_json_events(svc.call(make_request("ListSessions"))).await;
        result.expect("outer service must not error");

        assert_eq!(
            events_with_message(&events, "RPC deadline exceeded").len(),
            1,
            "precondition: the timeout path must have run"
        );
        assert_eq!(
            events.len(),
            1,
            "the server-deadline path must emit its own line and nothing else — no `RPC failed` \
             from the backstop, no abandoned line from the drop guard; got {events:?}"
        );
        assert_eq!(
            endpoint_call_cell(&metrics, "ListSessions", "CANCELLED"),
            0,
            "the drop guard must not also count the timed-out call as CANCELLED"
        );
    }

    /// An RPC that completes inside its budget emits no deadline line.
    ///
    /// The fence against the line becoming per-call noise: `RPC deadline
    /// exceeded` is emitted at WARN, and a copy on every successful call would
    /// swamp the log the ticket exists to make greppable.
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_call_within_deadline_emits_no_deadline_warn() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(29_000)));

        let (result, events) = capture_json_events(svc.call(make_request("CreateSession"))).await;
        result.expect("outer service must not error");

        assert!(
            events_with_message(&events, "RPC deadline exceeded").is_empty(),
            "a call inside its budget must not log a deadline line; got {events:?}"
        );
        assert_eq!(
            endpoint_call_cell(&metrics, "CreateSession", "DEADLINE_EXCEEDED"),
            0,
            "precondition: the call must not have timed out"
        );
    }

    // ── b.avd item 5: inflight RPC gauge ───────────────────────────────

    /// The gauge ticks up on entry and back down on response completion.
    /// Acceptance criterion 4: "Inflight RPCs gauge appears." We pin the
    /// balance contract here — pre-call value 0, during-call value 1
    /// (visible to a tokio-paused observer), post-call value 0.
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_inflight_gauge_balances_for_unary() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(500)));

        let read = || inflight(&metrics, "Ping");
        assert_eq!(read(), 0, "pre-call inflight must be 0");

        // Pin the future, drive it forward until the inner service is
        // poll-pending (the delay sleep). At that point the inflight
        // gauge must be 1.
        let mut fut = std::pin::pin!(svc.call(make_request("Ping")));
        // Poll once to enter the future and hit the await.
        std::future::poll_fn::<(), _>(|cx| {
            // Drive fut forward; expect Pending.
            match fut.as_mut().poll(cx) {
                Poll::Pending => Poll::Ready(()),
                Poll::Ready(_) => Poll::Ready(()),
            }
        })
        .await;
        assert_eq!(read(), 1, "inflight must be 1 while the call is in-flight");

        // Finish the call.
        fut.await.expect("call failed");
        assert_eq!(read(), 0, "post-call inflight must rebalance to 0");
    }

    // ── b.avd HIGH follow-up: inflight balance for streaming bodies ─────
    //
    // Pre-fix the inflight gauge dropped when the response-headers future
    // returned, but for streaming RPCs the body continues to stream long
    // after that — the gauge under-counted the actual live RPCs. Post-fix
    // the guard is moved into the `MetricsTrailerBody` so the gauge stays
    // incremented for the body's lifetime.

    /// `BodyChannelService` — inner Tower service that returns a response
    /// whose body is fed by a `tokio::sync::mpsc` channel. The test holds
    /// the sender alive to keep the stream open, then drops it to signal
    /// end-of-stream. Mirrors a `SubscribeReadLayer` / `JoinSession` shape
    /// where the response headers fly immediately but the body streams
    /// for the subscription lifetime.
    #[derive(Clone)]
    struct BodyChannelService {
        body: Arc<std::sync::Mutex<Option<tonic::body::Body>>>,
    }

    impl BodyChannelService {
        fn new() -> (Self, tokio::sync::mpsc::UnboundedSender<bytes::Bytes>) {
            use futures::stream::StreamExt;
            use http_body_util::StreamBody;

            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
            let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
                .map(|b| Ok::<_, tonic::Status>(http_body::Frame::data(b)));
            let stream_body = StreamBody::new(stream);
            let body = tonic::body::Body::new(stream_body);
            (
                BodyChannelService {
                    body: Arc::new(std::sync::Mutex::new(Some(body))),
                },
                tx,
            )
        }
    }

    impl<ReqBody> Service<Request<ReqBody>> for BodyChannelService
    where
        ReqBody: Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
            let body_slot = Arc::clone(&self.body);
            Box::pin(async move {
                let body = body_slot
                    .lock()
                    .expect("body slot poisoned")
                    .take()
                    .expect("BodyChannelService::call may only be called once");
                let response = http::Response::builder()
                    .status(200)
                    .body(body)
                    .expect("response construction failed");
                Ok(response)
            })
        }
    }

    /// The inflight gauge must stay incremented for the full body lifetime
    /// of a streaming response. Pre-fix the gauge dropped when the
    /// response-headers future returned, so `SubscribeReadLayer` /
    /// `JoinSession` were silently under-counted; post-fix the
    /// `MetricsTrailerBody` owns the inflight guard and its `Drop` only
    /// fires when the body is consumed or dropped.
    #[tokio::test]
    async fn metrics_layer_inflight_gauge_stays_up_until_streaming_body_completes() {
        use http_body_util::BodyExt;

        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let (inner, tx) = BodyChannelService::new();
        let mut svc = layer.layer(inner);

        let read = || inflight(&metrics, "SubscribeReadLayer");
        assert_eq!(read(), 0, "pre-call inflight must be 0");

        // Drive the call to get the response (with the wrapped body).
        let response = svc
            .call(make_request("SubscribeReadLayer"))
            .await
            .expect("call failed");

        // The response future has completed (headers + wrapped body
        // returned), but the body has not yet streamed any frame. Inflight
        // must STILL be 1 — pre-fix this assertion was 0 because the guard
        // dropped at the future boundary.
        assert_eq!(
            read(),
            1,
            "inflight must remain 1 after response future returns but before stream ends \
             (streaming RPCs hold the gauge until the body completes)"
        );

        // Pull a single data frame to prove the body is alive and the
        // gauge stays at 1 while data is in flight.
        let (_parts, mut body) = response.into_parts();
        tx.send(bytes::Bytes::from_static(b"frame-1"))
            .expect("send frame failed");
        let _frame = body
            .frame()
            .await
            .expect("frame must arrive")
            .expect("frame ok");
        assert_eq!(
            read(),
            1,
            "inflight must remain 1 while the body is mid-stream"
        );

        // Close the stream — drop the sender so the receiver yields None
        // → the body ends → MetricsTrailerBody records OK on the
        // Poll::Ready(None) branch. The inflight guard's Drop fires when
        // the body itself is dropped, so we drain to end-of-stream first
        // and then drop the body to release the gauge.
        drop(tx);
        // Drain to end-of-stream.
        while let Some(frame) = body.frame().await {
            let _ = frame.expect("frame error");
        }
        // Body has streamed to end; the inflight guard still lives
        // inside the body. Drop the body to release the guard.
        drop(body);

        assert_eq!(
            read(),
            0,
            "inflight must rebalance to 0 once the streaming body is dropped after completion"
        );
    }

    /// Dropping the body without consuming it to end-of-stream must still
    /// release the inflight gauge. Covers the caller-cancellation /
    /// dead-subscriber shape where the consumer simply walks away from a
    /// streaming RPC.
    #[tokio::test]
    async fn metrics_layer_inflight_gauge_decrements_when_body_dropped_without_trailers() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let (inner, _tx) = BodyChannelService::new();
        let mut svc = layer.layer(inner);

        let read = || inflight(&metrics, "JoinSession");
        assert_eq!(read(), 0);

        let response = svc
            .call(make_request("JoinSession"))
            .await
            .expect("call failed");
        assert_eq!(read(), 1, "inflight must be 1 while body is held");

        // Drop the response (and therefore its body) without ever calling
        // `.frame()` — emulates a client that walks away from a streaming
        // subscription mid-flight.
        drop(response);
        assert_eq!(
            read(),
            0,
            "inflight must rebalance to 0 when a streaming body is dropped without trailers"
        );
    }

    /// Caller-cancellation safety net (b.avd second follow-up).
    ///
    /// Pre-fix, dropping a streaming-RPC body before `poll_frame` ever
    /// resolved to `Ready(None)` / `Ready(Some(Err(_)))` let `ctx` stay
    /// `Some` and the field-by-field auto-drop silently scrubbed it —
    /// `inc_calls` and `observe_duration_us` never fired for that RPC,
    /// violating the "every accepted RPC counts" invariant. Post-fix
    /// `Drop for MetricsTrailerBody` records the call under the
    /// canonical `"CANCELLED"` slug (matches
    /// `crate::observability::metrics::GRPC_CODES`) before the inflight
    /// guard's own Drop decrements the gauge.
    ///
    /// Pairs with `metrics_layer_inflight_gauge_decrements_when_body_dropped_without_trailers`
    /// (the gauge half of the same scenario): together they pin both
    /// the counter and the gauge under caller cancellation.
    #[tokio::test]
    async fn metrics_layer_body_dropped_without_trailers_records_cancelled_call() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let (inner, _tx) = BodyChannelService::new();
        let mut svc = layer.layer(inner);

        // No call observations yet — every counter cell is at 0.
        assert_eq!(
            counter_value(&metrics, "SubscribeReadLayer"),
            0,
            "pre-call counter for SubscribeReadLayer must be 0"
        );

        // Drive the response future; receive the wrapped body.
        let response = svc
            .call(make_request("SubscribeReadLayer"))
            .await
            .expect("call failed");

        // Pre-Drop the counter is still 0 — neither
        // `poll_frame::Ready(None)` (records OK) nor a trailer (records
        // resolved code) nor a body-error (records INTERNAL) has fired.
        assert_eq!(
            counter_value(&metrics, "SubscribeReadLayer"),
            0,
            "no frame observed yet ⇒ no counter tick yet"
        );

        // Drop without consuming — caller-cancellation shape.
        drop(response);

        // The new `Drop for MetricsTrailerBody` must have ticked the
        // CANCELLED cell exactly once. Reading the specific cell pins
        // the slug: any future change that records under a different
        // label (e.g. `"UNKNOWN"`, `"INTERNAL"`) will fail this
        // assertion and surface the drift loudly.
        let cancelled = endpoint_call_cell(&metrics, "SubscribeReadLayer", "CANCELLED");
        assert_eq!(
            cancelled, 1,
            "dropped-body must record under grpc_code=CANCELLED \
             (b.avd second follow-up — every accepted RPC counts)"
        );

        // And the duration histogram must have observed exactly one
        // sample under the same code — operator dashboards expect the
        // duration series to track the counter 1:1.
        let cancelled_samples =
            endpoint_duration_count(&metrics, "SubscribeReadLayer", "CANCELLED");
        assert_eq!(
            cancelled_samples, 1,
            "cancelled duration histogram must record a paired sample"
        );

        // Belt-and-suspenders: no other code label for this endpoint
        // ticked. The OK / INTERNAL / DEADLINE_EXCEEDED cells must
        // still be at 0 — pre-fix the body-end branches silently
        // recorded nothing, and a regression could plausibly mis-record
        // a body-drop as one of those.
        for code in &["OK", "INTERNAL", "DEADLINE_EXCEEDED"] {
            let cell = endpoint_call_cell(&metrics, "SubscribeReadLayer", code);
            assert_eq!(
                cell, 0,
                "body-dropped path must not tick grpc_code={code} on SubscribeReadLayer"
            );
        }
    }

    /// Deadline-exceeded path also decrements the inflight gauge. The
    /// RAII guard fires when the response future is dropped on the
    /// timeout branch.
    #[tokio::test(start_paused = true)]
    async fn metrics_layer_inflight_gauge_decrements_on_deadline_exceeded() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::with_delay(Duration::from_millis(31_000)));

        let read = || inflight(&metrics, "DeleteSession");
        assert_eq!(read(), 0);
        svc.call(make_request("DeleteSession"))
            .await
            .expect("outer service must not error");
        assert_eq!(
            read(),
            0,
            "timeout path must rebalance the inflight gauge (b.avd item 5)"
        );
    }

    /// Malformed paths with no slashes still pass through safely — no
    /// panic, no `unwrap` unreachable branch hit — and bucket under
    /// `other`. Covers the "defensive `unwrap_or`" rationale documented
    /// in `endpoint_name`.
    #[tokio::test]
    async fn metrics_layer_malformed_path_does_not_panic_and_buckets_to_other() {
        let metrics = make_metrics();
        let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
        let mut svc = layer.layer(DelayedOkService::instant());

        // Path with no slash at all — `rsplit('/').next()` still returns
        // Some("no-slash"), which is not in the allowlist → `other`.
        let req = http::Request::builder()
            .uri("no-slash")
            .body(())
            .expect("request construction failed");
        svc.call(req).await.expect("call must not panic");

        assert_eq!(
            counter_value(&metrics, "other"),
            1,
            "slashless path must tick the `other` bucket without panicking"
        );
    }

    // ── The ambient request deadline (b.hka (f)) ─────────────────────────────
    //
    // This layer is the only thing that publishes `grpc::deadline`, so it is
    // the only place the publication can be pinned. What a *consumer* then does
    // with the value lives with the consumer
    // (`Config::object_id_counter_repair_wait_budget_within`,
    // `checkpoint::counter_seed`), and the composed payoff — a `CreateLayer` on
    // a short client deadline answering a deliberate `UNAVAILABLE` instead of a
    // bare `DEADLINE_EXCEEDED` — in
    // `tests/integration/checkpoint/counter_repair_test.rs`.

    /// An inner service that reports what `deadline::remaining()` answers
    /// *inside* the wrapped future — i.e. from where a handler would read it.
    #[derive(Clone)]
    struct DeadlineProbeService {
        observed: Arc<std::sync::Mutex<Option<Option<Duration>>>>,
    }

    impl DeadlineProbeService {
        fn new() -> (Self, Arc<std::sync::Mutex<Option<Option<Duration>>>>) {
            let observed = Arc::new(std::sync::Mutex::new(None));
            (
                Self {
                    observed: Arc::clone(&observed),
                },
                observed,
            )
        }
    }

    impl<ReqBody> Service<Request<ReqBody>> for DeadlineProbeService
    where
        ReqBody: Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
            let observed = Arc::clone(&self.observed);
            Box::pin(async move {
                *observed.lock().expect("probe mutex poisoned") =
                    Some(crate::grpc::deadline::remaining());
                Ok(http::Response::builder()
                    .status(200)
                    .body(tonic::body::Body::empty())
                    .expect("response construction failed"))
            })
        }
    }

    /// Build a request for `grpc_method`, optionally carrying a client
    /// `grpc-timeout`.
    fn make_request_with_timeout(grpc_method: &str, timeout: Option<&str>) -> Request<()> {
        let mut request = make_request(grpc_method);
        if let Some(timeout) = timeout {
            request.headers_mut().insert(
                "grpc-timeout",
                http::HeaderValue::from_str(timeout).expect("test header must be valid ASCII"),
            );
        }
        request
    }

    /// Drive one request through the layer and return what the handler saw.
    async fn observe_deadline(
        rpc_deadline_ms: u64,
        grpc_method: &str,
        client_timeout: Option<&str>,
    ) -> Option<Duration> {
        let layer = make_metrics_layer(make_metrics(), rpc_deadline_ms, 2_000);
        let (probe, observed) = DeadlineProbeService::new();
        let mut svc = layer.layer(probe);
        svc.call(make_request_with_timeout(grpc_method, client_timeout))
            .await
            .expect("call must succeed");
        let seen = *observed.lock().expect("probe mutex poisoned");
        seen.expect("the inner service must have run")
    }

    /// With no client `grpc-timeout`, a non-exempt RPC is scoped with the
    /// server deadline.
    ///
    /// The upper bound is the contract: a handler sizing a wait as a fraction
    /// of this value must not be able to outlast the `tokio::time::timeout`
    /// this same layer applies. The lower bound is deliberately slack — it only
    /// distinguishes "the configured deadline" from "some much smaller
    /// number", never measures how fast the machine is, so it cannot convert
    /// runner load into a red release gate.
    #[tokio::test]
    async fn deadline_scope_uses_the_server_deadline_when_the_client_sets_none() {
        let observed = observe_deadline(30_000, "CreateLayer", None)
            .await
            .expect("a non-exempt RPC MUST carry an ambient deadline");

        assert!(
            observed <= Duration::from_millis(30_000),
            "the published remainder ({observed:?}) MUST NOT exceed the server deadline the \
             layer itself enforces — over-reporting lets a handler size a wait past the timeout \
             that will drop it"
        );
        assert!(
            observed > Duration::from_millis(15_000),
            "and it MUST be the server deadline, not some smaller default: a handler that reads \
             a fraction of a wrong-by-orders-of-magnitude remainder gives up long before it \
             needs to. Got {observed:?}"
        );
    }

    /// A client `grpc-timeout` SHORTER than the server deadline wins.
    ///
    /// This is the case b.hka (f) exists for. tonic's own `GrpcTimeout` layer —
    /// which sits outside this one and is what actually enforces the client
    /// bound — will drop the handler future at 500 ms, so publishing the
    /// server's 30 s here would let a handler size a 20 s wait against a
    /// request that has half a second to live: a bare `DEADLINE_EXCEEDED` with
    /// nothing recorded about what it was waiting on.
    #[tokio::test]
    async fn deadline_scope_takes_the_client_timeout_when_it_is_shorter() {
        let observed = observe_deadline(30_000, "CreateLayer", Some("500m"))
            .await
            .expect("a non-exempt RPC MUST carry an ambient deadline");

        assert!(
            observed <= Duration::from_millis(500),
            "the published remainder ({observed:?}) MUST respect the client's 500ms \
             `grpc-timeout`, which is the bound tonic will actually enforce"
        );
    }

    /// A client `grpc-timeout` LONGER than the server deadline is ignored.
    ///
    /// The pairing matters: this is a `min`, so a client cannot buy itself more
    /// server-side patience than `LIVE_EDIT_RPC_DEADLINE_MS` allows just by
    /// asking for it, and a handler cannot be told it has ten minutes when this
    /// layer will time it out at thirty seconds.
    #[tokio::test]
    async fn deadline_scope_ignores_a_client_timeout_longer_than_the_server_deadline() {
        let observed = observe_deadline(30_000, "CreateLayer", Some("600S"))
            .await
            .expect("a non-exempt RPC MUST carry an ambient deadline");

        assert!(
            observed <= Duration::from_millis(30_000),
            "a 600s client timeout MUST NOT widen the published remainder past the server \
             deadline this layer enforces; got {observed:?}"
        );
    }

    /// A malformed `grpc-timeout` degrades to the server deadline.
    ///
    /// Fail-open, matching tonic's parser: the header is only ever used to make
    /// a budget tighter, so an unparseable value must leave the pre-b.hka
    /// behaviour in place rather than collapsing the deadline to nothing.
    #[tokio::test]
    async fn deadline_scope_falls_back_to_the_server_deadline_on_a_malformed_client_timeout() {
        let observed = observe_deadline(30_000, "CreateLayer", Some("not-a-timeout"))
            .await
            .expect("a non-exempt RPC MUST carry an ambient deadline");

        assert!(
            observed > Duration::from_millis(15_000),
            "a malformed `grpc-timeout` MUST read as absent, leaving the server deadline in \
             force — treating it as zero would fail every request carrying a typo'd header. Got \
             {observed:?}"
        );
    }

    /// Deadline-exempt streaming RPCs are NOT scoped at all.
    ///
    /// `None` here means "unbounded", and that is the honest answer: these RPCs
    /// have no deadline, so there is nothing to report. Publishing a synthetic
    /// one would put a bound on waits inside `JoinSession` /
    /// `SubscribeReadLayer` that nothing enforces and nobody asked for.
    #[tokio::test]
    async fn deadline_scope_is_absent_for_deadline_exempt_streaming_rpcs() {
        for endpoint in super::EXEMPT_FROM_DEADLINE {
            assert_eq!(
                observe_deadline(30_000, endpoint, Some("500m")).await,
                None,
                "{endpoint} is exempt from the RPC deadline, so no ambient deadline may be \
                 published — not even the client's `grpc-timeout`, which this layer does not \
                 enforce for it either"
            );
        }
    }

    // ── Completion ledger, transport backstop, drop guard (SDD §17.2.1) ──────
    //
    // This layer is the outermost one the service owns, so it is where a
    // completion nobody inside reported gets its one line: the backstop logs a
    // non-OK `grpc-status` whose ledger is unmarked, and the drop guard logs a
    // call torn down before any response existed. Both are pinned here with
    // mock inner services; `tests/integration/grpc/completion_log_server_test.rs`
    // checks the same premises behind a real tonic server.
    mod completion {
        use super::super::super::rpc_log::{RpcCompletion, RPC_ABANDONED_MSG, RPC_FAILURE_MSG};
        use super::*;

        /// Inner service answering a trailers-only status — exactly what
        /// tonic produces for a status returned before any response message,
        /// and for its own pre-handler rejections. With `mark_ledger` it
        /// first marks the request's ledger, standing in for a handler,
        /// streaming tail or interceptor that already logged the failure.
        #[derive(Clone)]
        struct TrailersOnlyService {
            code: tonic::Code,
            message: &'static str,
            mark_ledger: bool,
            /// Extra response headers, for the malformed-details case.
            extra: Option<(&'static str, &'static str)>,
        }

        impl TrailersOnlyService {
            fn unreported(code: tonic::Code, message: &'static str) -> Self {
                Self {
                    code,
                    message,
                    mark_ledger: false,
                    extra: None,
                }
            }
        }

        impl<ReqBody> Service<Request<ReqBody>> for TrailersOnlyService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
                if self.mark_ledger {
                    request
                        .extensions()
                        .get::<Arc<RpcCompletion>>()
                        .expect("the metrics layer must have inserted the ledger")
                        .mark_reported();
                }
                let mut response: http::Response<tonic::body::Body> =
                    tonic::Status::new(self.code, self.message).into_http();
                if let Some((name, value)) = self.extra {
                    response.headers_mut().insert(
                        http::HeaderName::from_static(name),
                        http::HeaderValue::from_static(value),
                    );
                }
                Box::pin(async move { Ok(response) })
            }
        }

        /// Inner service whose response body streams one data frame and then
        /// a trailer frame carrying `grpc-status` — the ordinary shape of a
        /// unary or streaming response that failed after its head was sent.
        #[derive(Clone)]
        struct TrailerFrameService {
            code: tonic::Code,
            message: &'static str,
            mark_ledger: bool,
        }

        impl<ReqBody> Service<Request<ReqBody>> for TrailerFrameService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
                let ledger = request
                    .extensions()
                    .get::<Arc<RpcCompletion>>()
                    .cloned()
                    .expect("the metrics layer must have inserted the ledger");
                let mark_ledger = self.mark_ledger;
                let mut trailers = http::HeaderMap::new();
                tonic::Status::new(self.code, self.message)
                    .add_header(&mut trailers)
                    .expect("status must encode");
                Box::pin(async move {
                    use futures::stream::StreamExt;
                    // The mark happens as the trailer is produced, the way a
                    // streaming tail marks before it enqueues its status.
                    let frames = futures::stream::iter(vec![
                        Ok::<_, tonic::Status>(http_body::Frame::data(bytes::Bytes::from_static(
                            b"payload",
                        ))),
                        Ok(http_body::Frame::trailers(trailers)),
                    ])
                    .map(move |frame| {
                        if mark_ledger {
                            ledger.mark_reported();
                        }
                        frame
                    });
                    let body = tonic::body::Body::new(http_body_util::StreamBody::new(frames));
                    Ok(http::Response::builder()
                        .status(200)
                        .body(body)
                        .expect("response construction failed"))
                })
            }
        }

        /// Inner service that fails without producing any response.
        #[derive(Clone)]
        struct ErrService;

        impl<ReqBody> Service<Request<ReqBody>> for ErrService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Box<dyn std::error::Error + Send + Sync>;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
                Box::pin(async { Err(std::io::Error::other("inner service exploded").into()) })
            }
        }

        /// Inner service that never answers — a handler still working when
        /// the client resets the stream, or when tonic's `GrpcTimeout` fires.
        #[derive(Clone)]
        struct PendingService;

        impl<ReqBody> Service<Request<ReqBody>> for PendingService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
                Box::pin(std::future::pending())
            }
        }

        /// Inner service that records the ledger it was handed.
        #[derive(Clone)]
        struct LedgerProbeService {
            seen: Arc<std::sync::Mutex<Option<Arc<RpcCompletion>>>>,
        }

        impl<ReqBody> Service<Request<ReqBody>> for LedgerProbeService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
                *self.seen.lock().expect("probe mutex poisoned") =
                    request.extensions().get::<Arc<RpcCompletion>>().cloned();
                Box::pin(async {
                    Ok(http::Response::builder()
                        .status(200)
                        .body(tonic::body::Body::empty())
                        .expect("response construction failed"))
                })
            }
        }

        /// Drive `svc` for `method` to its response and drain the body, under
        /// capture. Draining is what streams the trailer frame past the
        /// layer's body wrapper, where the trailer-frame backstop runs.
        async fn call_and_drain<S>(svc: &mut S, method: &str) -> Vec<serde_json::Value>
        where
            S: Service<
                Request<()>,
                Response = http::Response<tonic::body::Body>,
                Error = Box<dyn std::error::Error + Send + Sync>,
            >,
        {
            let fut = svc.call(make_request(method));
            let ((), events) = capture_json_events(async move {
                use http_body_util::BodyExt;
                let response = fut.await.expect("outer service must not error");
                let mut body = response.into_body();
                while let Some(frame) = body.frame().await {
                    frame.expect("body frame must not error");
                }
            })
            .await;
            events
        }

        /// The ledger rides the request extensions into the inner service,
        /// named with the same bucketed endpoint the metric label uses.
        #[tokio::test]
        async fn the_ledger_reaches_the_inner_service_with_the_bucketed_endpoint() {
            let seen = Arc::new(std::sync::Mutex::new(None));
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(LedgerProbeService {
                seen: Arc::clone(&seen),
            });

            for (method, want) in [("StatLayer", "StatLayer"), ("NoSuchMethod", "other")] {
                svc.call(make_request(method))
                    .await
                    .expect("call must succeed");
                let ledger = seen
                    .lock()
                    .expect("probe mutex poisoned")
                    .take()
                    .expect("the inner service must see an `Arc<RpcCompletion>` extension");
                assert_eq!(ledger.endpoint(), want);
                assert!(!ledger.is_reported(), "a fresh ledger starts unmarked");
            }
        }

        /// An unreported trailers-only `OUT_OF_RANGE` — tonic's answer to a
        /// request over the decode limit — logs one INFO `RPC failed` at
        /// `stage=transport`, naming the endpoint and tonic's message.
        #[tokio::test]
        async fn backstop_logs_one_info_line_for_an_unreported_trailers_only_out_of_range() {
            let metrics = make_metrics();
            let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
            let mut svc = layer.layer(TrailersOnlyService::unreported(
                tonic::Code::OutOfRange,
                "Error, decoded message length too large: found 5242880 bytes",
            ));

            let events = call_and_drain(&mut svc, "UpdateLayer").await;

            assert_eq!(events.len(), 1, "exactly one line; got {events:?}");
            let event = &events[0];
            assert_eq!(event["level"], "INFO");
            let fields = &event["fields"];
            assert_eq!(fields["message"], RPC_FAILURE_MSG);
            assert_eq!(fields["endpoint"], "UpdateLayer");
            assert_eq!(fields["stage"], "transport");
            assert_eq!(fields["grpc_code"], "OUT_OF_RANGE");
            assert_eq!(
                fields["status_message"],
                "Error, decoded message length too large: found 5242880 bytes",
                "the percent-encoded `grpc-message` must be decoded back to tonic's text"
            );
            assert_eq!(
                endpoint_call_cell(&metrics, "UpdateLayer", "OUT_OF_RANGE"),
                1,
                "the backstop logs alongside the existing counter tick, never instead of it"
            );
        }

        /// An unreported trailers-only `INTERNAL` with no fault source —
        /// tonic's answer to an undecodable body — logs one WARN, never ERROR.
        #[tokio::test]
        async fn backstop_logs_one_warn_line_for_an_unreported_trailers_only_internal() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailersOnlyService::unreported(
                tonic::Code::Internal,
                "failed to decode Protobuf message",
            ));

            let events = call_and_drain(&mut svc, "StatLayer").await;

            assert_eq!(events.len(), 1, "exactly one line; got {events:?}");
            assert_eq!(
                events[0]["level"], "WARN",
                "a client can cause this INTERNAL at will, so it must never page anyone"
            );
            assert_eq!(events[0]["fields"]["message"], RPC_FAILURE_MSG);
            assert_eq!(events[0]["fields"]["grpc_code"], "INTERNAL");
            assert_eq!(events[0]["fields"]["stage"], "transport");
        }

        /// A trailers-only status whose ledger is already marked logs nothing
        /// here: whoever marked it logged it, with more context than this
        /// layer has.
        #[tokio::test]
        async fn backstop_is_silent_for_a_reported_trailers_only_status() {
            let metrics = make_metrics();
            let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
            let mut svc = layer.layer(TrailersOnlyService {
                code: tonic::Code::NotFound,
                message: "layer gone",
                mark_ledger: true,
                extra: None,
            });

            let events = call_and_drain(&mut svc, "StatLayer").await;

            assert!(
                events.is_empty(),
                "a reported failure must not be logged twice; got {events:?}"
            );
            assert_eq!(
                endpoint_call_cell(&metrics, "StatLayer", "NOT_FOUND"),
                1,
                "precondition: the call was recorded under its status"
            );
        }

        /// A trailers-only `OK` logs nothing.
        #[tokio::test]
        async fn backstop_is_silent_for_a_trailers_only_ok() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailersOnlyService::unreported(tonic::Code::Ok, ""));
            let events = call_and_drain(&mut svc, "Ping").await;
            assert!(events.is_empty(), "OK is not a failure; got {events:?}");
        }

        /// A malformed `grpc-status-details-bin` does not panic the backstop.
        ///
        /// The reason the backstop rebuilds the status from `grpc-status` and
        /// `grpc-message` alone: `Status::from_header_map` panics on a
        /// malformed details header, and this path runs on every failed
        /// response.
        #[tokio::test]
        async fn backstop_survives_a_malformed_status_details_header() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailersOnlyService {
                code: tonic::Code::Unavailable,
                message: "busy",
                mark_ledger: false,
                extra: Some(("grpc-status-details-bin", "!!!not-base64!!!")),
            });

            let events = call_and_drain(&mut svc, "DeleteLayer").await;

            assert_eq!(events.len(), 1, "got {events:?}");
            assert_eq!(events[0]["fields"]["grpc_code"], "UNAVAILABLE");
            assert_eq!(events[0]["fields"]["status_message"], "busy");
        }

        /// An unreported non-OK status in a trailer frame logs one line when
        /// the trailer streams past.
        #[tokio::test]
        async fn backstop_logs_one_line_for_an_unreported_trailer_frame() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailerFrameService {
                code: tonic::Code::Internal,
                message: "Error, failed to encode response",
                mark_ledger: false,
            });

            let events = call_and_drain(&mut svc, "SubscribeReadLayer").await;

            assert_eq!(events.len(), 1, "got {events:?}");
            assert_eq!(events[0]["level"], "WARN");
            assert_eq!(events[0]["fields"]["endpoint"], "SubscribeReadLayer");
            assert_eq!(events[0]["fields"]["stage"], "transport");
            assert_eq!(events[0]["fields"]["grpc_code"], "INTERNAL");
        }

        /// A trailer frame whose ledger was marked before the trailer streamed
        /// — a streaming tail that logged and then enqueued its status — logs
        /// nothing here.
        #[tokio::test]
        async fn backstop_is_silent_for_a_reported_trailer_frame() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailerFrameService {
                code: tonic::Code::Unavailable,
                message: "slow reader",
                mark_ledger: true,
            });
            let events = call_and_drain(&mut svc, "SubscribeReadLayer").await;
            assert!(events.is_empty(), "got {events:?}");
        }

        /// An OK trailer frame logs nothing.
        #[tokio::test]
        async fn backstop_is_silent_for_an_ok_trailer_frame() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let mut svc = layer.layer(TrailerFrameService {
                code: tonic::Code::Ok,
                message: "",
                mark_ledger: false,
            });
            let events = call_and_drain(&mut svc, "StatLayer").await;
            assert!(events.is_empty(), "got {events:?}");
        }

        /// An inner service that fails with no response logs one WARN `RPC
        /// failed` `INTERNAL` at `stage=transport`, beside the existing
        /// `INTERNAL` tick.
        #[tokio::test]
        async fn inner_service_error_logs_one_internal_line_at_the_transport_stage() {
            let metrics = make_metrics();
            let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
            let mut svc = layer.layer(ErrService);

            let fut = svc.call(make_request("CreateLayer"));
            let (result, events) = capture_json_events(fut).await;

            assert!(result.is_err(), "the inner error must propagate unchanged");
            assert_eq!(events.len(), 1, "got {events:?}");
            let fields = &events[0]["fields"];
            assert_eq!(events[0]["level"], "WARN");
            assert_eq!(fields["message"], RPC_FAILURE_MSG);
            assert_eq!(fields["endpoint"], "CreateLayer");
            assert_eq!(fields["stage"], "transport");
            assert_eq!(fields["grpc_code"], "INTERNAL");
            assert_eq!(endpoint_call_cell(&metrics, "CreateLayer", "INTERNAL"), 1);
        }

        /// Abandon a call the way hyper does on a client reset or tonic's
        /// `GrpcTimeout` does on a client `grpc-timeout`: drop the layer's
        /// future before any response exists. `tokio::time::timeout` is the
        /// stand-in (it drops the future it wraps when it fires).
        async fn abandon(
            metrics: &Arc<Metrics>,
            method: &str,
            grpc_timeout: Option<&str>,
        ) -> Vec<serde_json::Value> {
            let layer = make_metrics_layer(Arc::clone(metrics), 30_000, 2_000);
            let mut svc = layer.layer(PendingService);
            let fut = svc.call(make_request_with_timeout(method, grpc_timeout));
            let ((), events) = capture_json_events(async move {
                let outcome = tokio::time::timeout(Duration::from_millis(100), fut).await;
                assert!(
                    outcome.is_err(),
                    "precondition: the inner service never answers, so the stand-in must fire"
                );
            })
            .await;
            events
        }

        /// A call dropped before any response existed logs one INFO
        /// `RPC abandoned before completion` naming the endpoint, the
        /// `CANCELLED` code and the elapsed time, and records exactly one
        /// `CANCELLED` tick and one duration sample; the inflight gauge
        /// balances.
        ///
        /// Before the drop guard this call left no line AND no tick — an
        /// accepted RPC that simply vanished from `live_endpoint_calls_total`.
        #[tokio::test(start_paused = true)]
        async fn abandoned_call_logs_one_info_line_and_one_cancelled_tick() {
            let metrics = make_metrics();
            let events = abandon(&metrics, "CreateLayer", None).await;

            assert_eq!(events.len(), 1, "exactly one line; got {events:?}");
            let event = &events[0];
            assert_eq!(event["level"], "INFO");
            let fields = &event["fields"];
            assert_eq!(fields["message"], RPC_ABANDONED_MSG);
            assert_eq!(fields["endpoint"], "CreateLayer");
            assert_eq!(fields["grpc_code"], "CANCELLED");
            assert!(
                fields["elapsed_ms"].is_u64(),
                "`elapsed_ms` must be present and numeric; got {fields:?}"
            );
            assert!(
                fields.get("client_timeout_ms").is_none(),
                "a request with no `grpc-timeout` has no client bound to report; got {fields:?}"
            );

            assert_eq!(endpoint_call_cell(&metrics, "CreateLayer", "CANCELLED"), 1);
            assert_eq!(
                endpoint_duration_count(&metrics, "CreateLayer", "CANCELLED"),
                1,
                "the duration series must track the counter 1:1"
            );
            assert_eq!(
                counter_value(&metrics, "CreateLayer"),
                1,
                "exactly one tick in total — CANCELLED and nothing else"
            );
            assert_eq!(
                inflight(&metrics, "CreateLayer"),
                0,
                "the gauge must balance"
            );
        }

        /// `client_timeout_ms` is on the abandoned line exactly when the
        /// request carried a `grpc-timeout`, with the parsed bound.
        #[tokio::test(start_paused = true)]
        async fn abandoned_call_reports_the_client_timeout_when_the_request_carried_one() {
            let metrics = make_metrics();
            let events = abandon(&metrics, "UpdateLayer", Some("250m")).await;

            assert_eq!(events.len(), 1, "got {events:?}");
            assert_eq!(events[0]["fields"]["message"], RPC_ABANDONED_MSG);
            assert_eq!(
                events[0]["fields"]["client_timeout_ms"], 250,
                "a `250m` grpc-timeout is a 250 ms client bound"
            );
            assert_eq!(
                events[0]["fields"]["grpc_code"], "CANCELLED",
                "classified CANCELLED whatever the elapsed time — no deadline heuristic"
            );
        }

        /// A future dropped before its first poll is still recorded: the
        /// guard is built in `call()`, not inside the async block.
        #[tokio::test]
        async fn call_dropped_before_its_first_poll_is_still_recorded_as_abandoned() {
            let metrics = make_metrics();
            let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
            let mut svc = layer.layer(PendingService);
            let fut = svc.call(make_request("StatLayer"));
            assert_eq!(
                inflight(&metrics, "StatLayer"),
                1,
                "precondition: the call is in flight"
            );

            let ((), events) = capture_json_events(async move { drop(fut) }).await;

            assert_eq!(events.len(), 1, "got {events:?}");
            assert_eq!(events[0]["fields"]["message"], RPC_ABANDONED_MSG);
            assert_eq!(endpoint_call_cell(&metrics, "StatLayer", "CANCELLED"), 1);
            assert_eq!(inflight(&metrics, "StatLayer"), 0);
        }

        /// A completed call adds no abandoned line and no `CANCELLED` tick —
        /// every exit disarms the guard.
        #[tokio::test]
        async fn completed_call_logs_no_abandoned_line_and_ticks_no_cancelled() {
            let metrics = make_metrics();
            let layer = make_metrics_layer(Arc::clone(&metrics), 30_000, 2_000);
            let mut svc = layer.layer(DelayedOkService::instant());

            let fut = svc.call(make_request("Ping"));
            let (result, events) = capture_json_events(fut).await;
            result.expect("call must succeed");

            assert!(
                events.is_empty(),
                "a completed call is silent; got {events:?}"
            );
            assert_eq!(endpoint_call_cell(&metrics, "Ping", "CANCELLED"), 0);
            assert_eq!(endpoint_call_cell(&metrics, "Ping", "OK"), 1);
            assert_eq!(inflight(&metrics, "Ping"), 0);
        }

        /// A response body dropped after the response existed logs nothing.
        ///
        /// This is the client leaving an established stream — a closed health
        /// `Watch`, a finished `JoinSession` — which probes do continuously. It
        /// keeps its existing `CANCELLED` tick (pinned elsewhere in this file)
        /// and must never reach the drop guard's abandoned line.
        #[tokio::test]
        async fn body_dropped_after_the_response_existed_logs_nothing() {
            let layer = make_metrics_layer(make_metrics(), 30_000, 2_000);
            let (inner, _tx) = BodyChannelService::new();
            let mut svc = layer.layer(inner);

            let fut = svc.call(make_request("JoinSession"));
            let ((), events) = capture_json_events(async move {
                let response = fut.await.expect("call must succeed");
                drop(response);
            })
            .await;

            assert!(
                events.is_empty(),
                "leaving an established stream is not a failure; got {events:?}"
            );
        }
    }

    // ── The per-RPC server span (`rpc_server_span`) ──────────────────────────
    //
    // A collecting trace pipeline, composed the way `build_registry` composes
    // it (default trace filter), installed as the thread-local default on a
    // current-thread runtime: the span opened in `call()` and every future and
    // body that later closes it run on the thread holding the guard. The
    // exporter is synchronous, so a span is in the capture the moment it ends
    // and absent until then.
    mod server_span {
        use opentelemetry::trace::{SpanId, SpanKind, Status, TraceId};
        use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData};

        use crate::config::{LogFormat, DEFAULT_TRACE_FILTER};
        use crate::observability::compose_registry;
        use crate::observability::trace_export::test_support::{
            collecting_tracer, keep_a_second_dispatcher_registered, CollectingExporter,
        };

        use super::super::KNOWN_ENDPOINTS;
        use super::*;

        const LIVE_SERVICE: &str = "nvidia.omniverse.live.v1alpha.LiveService";
        const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
        const PARENT_SPAN_ID: &str = "00f067aa0ba902b7";

        /// The exported spans of everything run while it is alive.
        struct SpanCapture {
            exporter: CollectingExporter,
            _provider: SdkTracerProvider,
            _guard: tracing::subscriber::DefaultGuard,
        }

        impl SpanCapture {
            fn start() -> Self {
                assert_eq!(
                    tokio::runtime::Handle::current().runtime_flavor(),
                    tokio::runtime::RuntimeFlavor::CurrentThread,
                    "a thread-local capture only sees work polled on this thread"
                );
                keep_a_second_dispatcher_registered();
                let (provider, tracer, exporter) = collecting_tracer();
                let subscriber = compose_registry(
                    "info",
                    LogFormat::Json,
                    DEFAULT_TRACE_FILTER,
                    std::io::sink,
                    Some(tracer),
                );
                Self {
                    exporter,
                    _provider: provider,
                    _guard: tracing::subscriber::set_default(subscriber),
                }
            }

            fn spans(&self) -> Vec<SpanData> {
                self.exporter.spans()
            }

            /// The one exported span, asserting there is exactly one.
            fn only_span(&self) -> SpanData {
                let spans = self.spans();
                assert_eq!(
                    spans.len(),
                    1,
                    "expected exactly one exported span; got {:?}",
                    spans.iter().map(|s| s.name.as_ref()).collect::<Vec<_>>()
                );
                spans.into_iter().next().expect("length checked")
            }
        }

        fn attr(span: &SpanData, key: &str) -> Option<String> {
            span.attributes
                .iter()
                .find(|kv| kv.key.as_str() == key)
                .map(|kv| kv.value.as_str().into_owned())
        }

        fn request_at(path: &str) -> Request<()> {
            Request::builder()
                .uri(path)
                .body(())
                .expect("request construction failed")
        }

        fn request_with_headers(method: &str, headers: &[(&'static str, String)]) -> Request<()> {
            let mut request = make_request(method);
            for (name, value) in headers {
                request.headers_mut().insert(
                    *name,
                    http::HeaderValue::from_str(value).expect("valid header value"),
                );
            }
            request
        }

        fn traceparent(parent_span_id: &str, flags: &str) -> (&'static str, String) {
            (
                "traceparent",
                format!("00-{TRACE_ID}-{parent_span_id}-{flags}"),
            )
        }

        fn layered<S>(inner: S) -> super::super::MetricsService<S> {
            make_metrics_layer(make_metrics(), 30_000, 2_000).layer(inner)
        }

        async fn drain(response: http::Response<tonic::body::Body>) {
            use http_body_util::BodyExt;
            let mut body = response.into_body();
            while let Some(frame) = body.frame().await {
                let _frame = frame.expect("body frame must not error");
            }
        }

        /// Inner service whose body streams one data frame and then a
        /// trailer frame carrying `code` — a response that failed (or
        /// succeeded) after its head was sent.
        #[derive(Clone)]
        struct TrailerStatusService {
            code: tonic::Code,
        }

        impl<ReqBody> Service<Request<ReqBody>> for TrailerStatusService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
                let mut trailers = http::HeaderMap::new();
                tonic::Status::new(self.code, "trailer status")
                    .add_header(&mut trailers)
                    .expect("status must encode");
                Box::pin(async move {
                    let frames = futures::stream::iter(vec![
                        Ok::<_, tonic::Status>(http_body::Frame::data(bytes::Bytes::from_static(
                            b"payload",
                        ))),
                        Ok(http_body::Frame::trailers(trailers)),
                    ]);
                    let body = tonic::body::Body::new(http_body_util::StreamBody::new(frames));
                    Ok(http::Response::builder()
                        .status(200)
                        .body(body)
                        .expect("response construction failed"))
                })
            }
        }

        /// Inner service answering a trailers-only `code` before any
        /// message — the shape of the auth interceptor's rejection.
        #[derive(Clone)]
        struct TrailersOnlyStatusService {
            code: tonic::Code,
        }

        impl<ReqBody> Service<Request<ReqBody>> for TrailersOnlyStatusService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
                let response = tonic::Status::new(self.code, "rejected").into_http();
                Box::pin(async move { Ok(response) })
            }
        }

        /// Inner service that never answers.
        #[derive(Clone)]
        struct NeverService;

        impl<ReqBody> Service<Request<ReqBody>> for NeverService
        where
            ReqBody: Send + 'static,
        {
            type Response = http::Response<tonic::body::Body>;
            type Error = Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _request: Request<ReqBody>) -> Self::Future {
                Box::pin(std::future::pending())
            }
        }

        // ── Name, kind, attributes ──────────────────────────────────────────

        #[tokio::test]
        async fn server_span_is_named_for_the_rpc_with_server_kind_and_rpc_attributes() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            svc.call(make_request("StatLayer"))
                .await
                .expect("call failed");

            let span = capture.only_span();
            let rpc = format!("{LIVE_SERVICE}/StatLayer");
            assert_eq!(span.name, rpc);
            assert_eq!(span.span_kind, SpanKind::Server);
            assert_eq!(attr(&span, "rpc.system.name").as_deref(), Some("grpc"));
            assert_eq!(attr(&span, "rpc.method").as_deref(), Some(rpc.as_str()));
            assert_eq!(attr(&span, "rpc.status_code").as_deref(), Some("OK"));
            assert_eq!(attr(&span, "error.type"), None, "OK is not an error");
            assert_eq!(span.status, Status::Unset);
            assert_eq!(
                span.parent_span_id,
                SpanId::INVALID,
                "without a traceparent the span is a root"
            );
        }

        /// One span per call for every `LiveService` method, named after it.
        #[tokio::test]
        async fn server_span_every_live_service_method_gets_one_span_named_after_it() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            for method in KNOWN_ENDPOINTS {
                svc.call(make_request(method)).await.expect("call failed");
            }

            let exported: Vec<String> = capture
                .spans()
                .iter()
                .map(|span| span.name.to_string())
                .collect();
            let expected: Vec<String> = KNOWN_ENDPOINTS
                .iter()
                .map(|method| format!("{LIVE_SERVICE}/{method}"))
                .collect();
            assert_eq!(exported, expected);
        }

        /// Health, reflection, the CapabilitiesService, an unknown method and a
        /// known method name under another service open no span. A positive
        /// control at the end shows the capture was live throughout.
        #[tokio::test]
        async fn server_span_is_not_opened_outside_the_live_service_methods() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            for path in [
                "/grpc.health.v1.Health/Check",
                "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
                "/nvidia.omniverse.discovery.capabilities.v2alpha.CapabilitiesService/ListServices",
                "/nvidia.omniverse.live.v1alpha.LiveService/NoSuchMethod",
                "/foo.Bar/UpdateLayer",
            ] {
                svc.call(request_at(path)).await.expect("call failed");
                assert!(
                    capture.spans().is_empty(),
                    "{path} must not open a server span; got {:?}",
                    capture.spans()
                );
            }

            svc.call(make_request("Ping")).await.expect("call failed");
            assert_eq!(
                capture.only_span().name,
                format!("{LIVE_SERVICE}/Ping"),
                "positive control: a LiveService call in the same capture is exported"
            );
        }

        // ── Status by exit ──────────────────────────────────────────────────

        /// A stream's span ends with the stream: nothing is exported when the
        /// response head returns or while data flows, and the span is exported
        /// with `OK` once the body ends — carrying an end time from the stream's
        /// end, not from when the handler future returned the response head.
        ///
        /// The end-time check is an ordering, not a speed: a marker taken while
        /// the stream is still open must not be later than the span's end.
        #[tokio::test]
        async fn server_span_of_a_stream_is_exported_ok_only_when_the_body_ends() {
            use http_body_util::BodyExt;

            let capture = SpanCapture::start();
            let (inner, tx) = BodyChannelService::new();
            let mut svc = layered(inner);

            let response = svc
                .call(make_request("SubscribeReadLayer"))
                .await
                .expect("call failed");
            assert!(
                capture.spans().is_empty(),
                "the span must not end when the response head returns"
            );

            let mut body = response.into_body();
            tx.send(bytes::Bytes::from_static(b"frame-1"))
                .expect("send frame failed");
            let _frame = body
                .frame()
                .await
                .expect("frame must arrive")
                .expect("frame ok");
            assert!(
                capture.spans().is_empty(),
                "the span must not end while the stream is open"
            );
            let stream_still_open = std::time::SystemTime::now();

            drop(tx);
            while let Some(frame) = body.frame().await {
                let _frame = frame.expect("frame error");
            }

            let span = capture.only_span();
            assert_eq!(attr(&span, "rpc.status_code").as_deref(), Some("OK"));
            assert_eq!(span.status, Status::Unset);
            assert!(
                span.end_time >= stream_still_open,
                "the span's end time must come from the stream's end, not from the handler \
                 returning the response head: end {:?} precedes a moment the stream was still \
                 open ({stream_still_open:?})",
                span.end_time
            );
        }

        /// Every gRPC code against the OpenTelemetry gRPC server-span table:
        /// the code is always recorded, and only UNKNOWN, DEADLINE_EXCEEDED,
        /// UNIMPLEMENTED, INTERNAL, UNAVAILABLE and DATA_LOSS mark the span an
        /// error with `error.type`. The table is written out here rather than
        /// read from `SERVER_ERROR_CODES`, so a change to that list fails.
        #[tokio::test]
        async fn server_span_status_follows_the_otel_grpc_server_table_for_every_code() {
            use tonic::Code;

            /// (code sent in the trailer, its slug, whether it is a server error)
            const TABLE: [(Code, &str, bool); 17] = [
                (Code::Ok, "OK", false),
                (Code::Cancelled, "CANCELLED", false),
                (Code::Unknown, "UNKNOWN", true),
                (Code::InvalidArgument, "INVALID_ARGUMENT", false),
                (Code::DeadlineExceeded, "DEADLINE_EXCEEDED", true),
                (Code::NotFound, "NOT_FOUND", false),
                (Code::AlreadyExists, "ALREADY_EXISTS", false),
                (Code::PermissionDenied, "PERMISSION_DENIED", false),
                (Code::ResourceExhausted, "RESOURCE_EXHAUSTED", false),
                (Code::FailedPrecondition, "FAILED_PRECONDITION", false),
                (Code::Aborted, "ABORTED", false),
                (Code::OutOfRange, "OUT_OF_RANGE", false),
                (Code::Unimplemented, "UNIMPLEMENTED", true),
                (Code::Internal, "INTERNAL", true),
                (Code::Unavailable, "UNAVAILABLE", true),
                (Code::DataLoss, "DATA_LOSS", true),
                (Code::Unauthenticated, "UNAUTHENTICATED", false),
            ];

            let capture = SpanCapture::start();
            for (i, (code, slug, server_error)) in TABLE.into_iter().enumerate() {
                let mut svc = layered(TrailerStatusService { code });
                let response = svc
                    .call(make_request("StatLayer"))
                    .await
                    .expect("call failed");
                drain(response).await;

                let spans = capture.spans();
                assert_eq!(spans.len(), i + 1, "{slug}: one span per call");
                let span = &spans[i];
                assert_eq!(
                    attr(span, "rpc.status_code").as_deref(),
                    Some(slug),
                    "{code:?} must be recorded as {slug}"
                );
                if server_error {
                    assert_eq!(
                        attr(span, "error.type").as_deref(),
                        Some(slug),
                        "{slug} is a server error and must name itself in error.type"
                    );
                    assert!(
                        matches!(span.status, Status::Error { .. }),
                        "{slug} is a server error; got {:?}",
                        span.status
                    );
                } else {
                    assert_eq!(
                        attr(span, "error.type"),
                        None,
                        "{slug} is not a server error"
                    );
                    assert_eq!(span.status, Status::Unset, "{slug} is not a server error");
                }
            }
        }

        /// The server deadline ends the span `DEADLINE_EXCEEDED`, an error.
        #[tokio::test(start_paused = true)]
        async fn server_span_on_the_server_deadline_is_deadline_exceeded_with_error_status() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::with_delay(Duration::from_millis(31_000)));

            svc.call(make_request("CreateSession"))
                .await
                .expect("outer service must not error");

            let span = capture.only_span();
            assert_eq!(
                attr(&span, "rpc.status_code").as_deref(),
                Some("DEADLINE_EXCEEDED")
            );
            assert_eq!(
                attr(&span, "error.type").as_deref(),
                Some("DEADLINE_EXCEEDED")
            );
            assert!(
                matches!(span.status, Status::Error { .. }),
                "DEADLINE_EXCEEDED is a server error; got {:?}",
                span.status
            );
        }

        /// A call dropped before any response existed ends its span
        /// `CANCELLED` — the client left, not a server error.
        #[tokio::test]
        async fn server_span_of_a_dropped_call_is_cancelled_without_error_status() {
            let capture = SpanCapture::start();
            let mut svc = layered(NeverService);

            let mut fut = svc.call(make_request("CreateSession"));
            std::future::poll_fn(|cx| {
                assert!(fut.as_mut().poll(cx).is_pending(), "the call never answers");
                Poll::Ready(())
            })
            .await;
            assert!(
                capture.spans().is_empty(),
                "an in-flight call's span has not ended"
            );
            drop(fut);

            let span = capture.only_span();
            assert_eq!(attr(&span, "rpc.status_code").as_deref(), Some("CANCELLED"));
            assert_eq!(attr(&span, "error.type"), None);
            assert_eq!(span.status, Status::Unset);
        }

        /// A response body dropped before it ended (a client leaving a stream)
        /// ends the span `CANCELLED`, not an error.
        #[tokio::test]
        async fn server_span_of_a_dropped_body_is_cancelled_without_error_status() {
            let capture = SpanCapture::start();
            let (inner, _tx) = BodyChannelService::new();
            let mut svc = layered(inner);

            let response = svc
                .call(make_request("JoinSession"))
                .await
                .expect("call failed");
            assert!(capture.spans().is_empty(), "the stream is still open");
            drop(response);

            let span = capture.only_span();
            assert_eq!(attr(&span, "rpc.status_code").as_deref(), Some("CANCELLED"));
            assert_eq!(attr(&span, "error.type"), None);
            assert_eq!(span.status, Status::Unset);
        }

        // ── traceparent ─────────────────────────────────────────────────────

        /// A sampled `traceparent` makes the span a child of the caller's span
        /// in the caller's trace — and adds nothing else: no link, no event.
        #[tokio::test]
        async fn server_span_joins_the_trace_of_a_sampled_traceparent() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            svc.call(request_with_headers(
                "StatLayer",
                &[traceparent(PARENT_SPAN_ID, "01")],
            ))
            .await
            .expect("call failed");

            let span = capture.only_span();
            assert_eq!(
                span.span_context.trace_id(),
                TraceId::from_hex(TRACE_ID).expect("valid trace id")
            );
            assert_eq!(
                span.parent_span_id,
                SpanId::from_hex(PARENT_SPAN_ID).expect("valid span id")
            );
            assert!(span.parent_span_is_remote);
            assert!(span.links.is_empty(), "a traceparent adds no link");
            assert!(span.events.is_empty(), "a traceparent adds no event");
        }

        /// An unsampled `traceparent` keeps the caller's decision: nothing is
        /// exported. A sampled call in the same capture is the positive
        /// control.
        #[tokio::test]
        async fn server_span_unsampled_traceparent_exports_nothing() {
            const SAMPLED_PARENT: &str = "1111111111111111";

            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            svc.call(request_with_headers(
                "StatLayer",
                &[traceparent(PARENT_SPAN_ID, "00")],
            ))
            .await
            .expect("call failed");
            assert!(
                capture.spans().is_empty(),
                "an unsampled caller's decision must be kept; got {:?}",
                capture.spans()
            );

            svc.call(request_with_headers(
                "StatLayer",
                &[traceparent(SAMPLED_PARENT, "01")],
            ))
            .await
            .expect("call failed");
            assert_eq!(
                capture.only_span().parent_span_id,
                SpanId::from_hex(SAMPLED_PARENT).expect("valid span id"),
                "positive control: the sampled call in the same capture is exported"
            );
        }

        /// A malformed `traceparent` — here one missing its flags — is
        /// ignored and the span is a root of its own trace.
        #[tokio::test]
        async fn server_span_with_malformed_traceparent_is_a_root() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            svc.call(request_with_headers(
                "StatLayer",
                &[("traceparent", format!("00-{TRACE_ID}-{PARENT_SPAN_ID}"))],
            ))
            .await
            .expect("call failed");

            let span = capture.only_span();
            assert_eq!(span.parent_span_id, SpanId::INVALID);
            assert!(!span.parent_span_is_remote);
            assert_ne!(
                span.span_context.trace_id(),
                TraceId::from_hex(TRACE_ID).expect("valid trace id"),
                "a malformed header must not lend the span its trace id"
            );
        }

        /// An inbound `tracestate` is not copied onto the span, so a caller
        /// cannot attach its own bytes to every exported span.
        #[tokio::test]
        async fn server_span_ignores_inbound_tracestate() {
            let capture = SpanCapture::start();
            let mut svc = layered(DelayedOkService::instant());

            svc.call(request_with_headers(
                "StatLayer",
                &[
                    traceparent(PARENT_SPAN_ID, "01"),
                    ("tracestate", "vendor=caller-chosen-bytes".to_string()),
                ],
            ))
            .await
            .expect("call failed");

            let span = capture.only_span();
            assert_eq!(
                span.parent_span_id,
                SpanId::from_hex(PARENT_SPAN_ID).expect("valid span id"),
                "precondition: the traceparent was honoured"
            );
            assert_eq!(
                span.span_context.trace_state().header(),
                "",
                "the inbound tracestate must not reach the exported span"
            );
        }

        /// An RPC refused before its handler — the auth interceptor's
        /// trailers-only `UNAUTHENTICATED` — still has exactly one span, in
        /// the caller's trace: never zero, and a traceparent never adds more.
        #[tokio::test]
        async fn server_span_of_a_rejected_call_with_traceparent_is_exactly_one_span() {
            let capture = SpanCapture::start();
            let mut svc = layered(TrailersOnlyStatusService {
                code: tonic::Code::Unauthenticated,
            });

            svc.call(request_with_headers(
                "StatLayer",
                &[traceparent(PARENT_SPAN_ID, "01")],
            ))
            .await
            .expect("outer service must not error");

            let span = capture.only_span();
            assert_eq!(
                span.parent_span_id,
                SpanId::from_hex(PARENT_SPAN_ID).expect("valid span id")
            );
            assert_eq!(
                attr(&span, "rpc.status_code").as_deref(),
                Some("UNAUTHENTICATED")
            );
            assert_eq!(span.status, Status::Unset, "a caller-caused code");
            assert!(span.links.is_empty(), "a traceparent adds no link");
        }
    }
}
