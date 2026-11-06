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

//! The production HTTP/2 server settings, LiveService stack and shutdown
//! sequencing, in one place.
//!
//! Both production servers — the bootstrap server that answers probes during
//! dependency wiring and the full-API server that takes the socket over from it
//! (SDD §5.5) — start from [`production_server_builder`], and so do the
//! integration tests that pin HTTP/2 transport behaviour, so a test server
//! cannot drift from the one the service runs. The LiveService is built only by
//! [`live_service_server`] — the auth interceptor, the disaster-recovery hold
//! gate in front of it, the end-of-stream record the uploads commit on, and
//! the body-progress record the JoinSession client keepalive reads (SDD §5.1,
//! §5.4, §13.2, SR-6.1) — in `main.rs` and in every test server that serves
//! it, for the same reason. The full-API server's graceful
//! shutdown signal is built by [`end_streams_after_drain`], the wait for the
//! server after GOAWAY is bounded by [`serve_with_post_goaway_bound`], and
//! everything after the server returns, up to and including
//! `ValkeyClient::quit()`, is [`run_post_server_sequence`]: the streaming RPCs'
//! task drains ([`drain_stream_tasks`]) and the other waits under one shared
//! deadline that keeps `quit()`'s headroom, then the background-task join. The
//! tests that pin shutdown behaviour call them too, for the same reason.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;
use tonic::{Request, Status};

use super::graceful_shutdown::{end_streams_paced, GracefulShutdown, StreamPacing};
use super::hold::{HoldGate, HoldGateService, HoldWriterAborts};
use super::proto::live_service_server::LiveServiceServer;
use super::request_end::AttestRequestEnd;
use super::service::LiveEditService;
use crate::observability::metrics::Metrics;

/// The auth interceptor's type, as [`live_service_server`] names it.
pub type AuthInterceptorFn = fn(Request<()>) -> Result<Request<()>, Status>;

/// The LiveService as the service serves it: built by [`live_service_server`].
pub type LiveServiceStack = AttestRequestEnd<
    HoldGateService<InterceptedService<LiveServiceServer<LiveEditService>, AuthInterceptorFn>>,
>;

/// The LiveService as every server serves it: the auth interceptor
/// ([`auth_interceptor`](super::auth_interceptor::auth_interceptor)) inside;
/// around it the hold gate (`gate`, [`HoldGate`]), which refuses every RPC
/// `SERVICE_HELD` while this replica is held, before auth runs (SDD SR-6.1);
/// and outermost [`AttestRequestEnd`](super::request_end::AttestRequestEnd),
/// which inserts two records: the end-of-stream record that `CreateLayer` and
/// `UpdateLayer` commit on (SDD §5.4), and the
/// [`RequestBodyProgress`](super::request_end::RequestBodyProgress) record the
/// JoinSession client keepalive reads (SDD §13.2).
///
/// The one constructor: `main.rs` and every in-repo test server that serves the
/// LiveService use it, so none can drift from the production stack. `main.rs`
/// passes the gate the hold reactor closes; a test server passes
/// [`HoldGate::always_open`], or a gate it closes itself. Through a stack
/// built any other way, an upload carries no end-of-stream record and is
/// refused `INTERNAL` (`grpc::streaming::UploadStream::from_request`), and a
/// JoinSession that opted in to the client keepalive carries no progress
/// record and is silently left unarmed: it is told `keepalive_interval_ms = 0`
/// and is never ended for silence.
pub fn live_service_server(svc: LiveEditService, gate: HoldGate) -> LiveServiceStack {
    AttestRequestEnd::new(gate.wrap(LiveServiceServer::with_interceptor(
        svc,
        super::auth_interceptor::auth_interceptor as AuthInterceptorFn,
    )))
}

/// HTTP/2 keepalive PING interval.
///
/// Public so the upload end-to-end keepalive test
/// (`tests/integration/grpc/upload_admission_test.rs`, `cancelled_upload`)
/// can assert at compile time that its server's RPC deadline outlasts this
/// interval plus [`HTTP2_KEEPALIVE_TIMEOUT`], so its call meets the keepalive
/// ending rather than the deadline.
pub const HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);

/// How long a keepalive PING may go unanswered before the connection is closed.
///
/// Public for the same compile-time assertion as
/// [`HTTP2_KEEPALIVE_INTERVAL`].
pub const HTTP2_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(5);

/// The HTTP/2 receive window of each inbound stream, 1 MiB: what the server
/// advertises as `SETTINGS_INITIAL_WINDOW_SIZE`.
///
/// The same value as hyper's server default (`DEFAULT_STREAM_WINDOW` in
/// hyper's `proto/h2/server.rs`), set explicitly by
/// [`production_server_builder`] because two figures depend on it: the
/// unread content one queued upload waiter can hold, the pod memory plan's
/// waiter term (`config::memory_plan::UPLOAD_WAITER_WINDOW_BYTES`, SDD §21.2),
/// and the connection window [`http2_connection_window_bytes`] derives from
/// it. A change to hyper's default must not move either silently.
pub const HTTP2_STREAM_WINDOW_BYTES: u32 = 1_048_576;

/// The most stream errors h2 itself detects and answers with RST_STREAM on one
/// inbound connection, over the connection's whole life: 10,240. The next one
/// closes the connection with GOAWAY `ENHANCE_YOUR_CALM`, debug data
/// `too_many_internal_resets`, and h2 logs a WARN, "locally-reset streams
/// reached limit (10240)" (h2 0.4.19 `proto/streams/streams.rs`).
///
/// Set by [`production_server_builder`] because tonic's server builder
/// defaults this limit to `None` and passes that through, and hyper and h2
/// read `None` as no limit at all: any peer that completes an HTTP/2
/// handshake could then make the service reset streams without end on one
/// connection, the CPU and memory pattern of RUSTSEC-2024-0003.
///
/// The value is 10 × h2's and hyper's own default of 1,024
/// (`DEFAULT_LOCAL_RESET_COUNT_MAX` in h2's `proto/mod.rs`). h2 never
/// decrements this count, and it also counts conforming late client DATA:
/// DATA on a stream the service ended early that arrives after h2 has stopped
/// remembering the reset stream, which it does for 1 s and for at most 50
/// streams at a time (`DEFAULT_RESET_STREAM_SECS`, `DEFAULT_RESET_STREAM_MAX`).
/// Neither of those can be changed through hyper's server builder, which
/// builds its own `h2::server::Builder` (hyper 1.11.0 `proto/h2/server.rs`).
/// On a long-lived upstream connection behind Envoy, shared by many clients,
/// ordinary early-ended uploads and joins would build up to the library
/// default over days, and the GOAWAY would then fail every stream on that
/// connection. 10,240 makes that 10 × rarer and still bounds a reset flood on
/// one connection: a flooder has to reconnect. See SDD §5.1.
pub const HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS: usize = 10_240;

/// The largest frame payload the service accepts, 16,384 bytes: what the
/// server advertises as `SETTINGS_MAX_FRAME_SIZE`.
///
/// The same value as hyper's server default (`DEFAULT_MAX_FRAME_SIZE` in
/// hyper's `proto/h2/server.rs`), set explicitly by
/// [`production_server_builder`] so a hyper upgrade cannot move it silently.
/// Frames the service sends follow the client's own setting, not this one.
pub const HTTP2_MAX_FRAME_BYTES: u32 = 16_384;

/// h2's largest receive window, 2³¹ − 1 (RFC 9113 §6.9.1; h2's
/// `MAX_WINDOW_SIZE`). h2 asserts a connection window at or below it in every
/// connection's handshake, so a larger one would panic every connection.
const H2_MAX_WINDOW_SIZE_BYTES: u32 = (1 << 31) - 1;

/// The largest `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS` whose connection window
/// ([`http2_connection_window_bytes`]) fits h2's largest window: 2,046, since
/// 2,047 × 1 MiB = 2,146,435,072 B is within 2³¹ − 1 and 2,048 × 1 MiB is not.
/// Configuration refuses a larger waiter cap at startup
/// (`Config::validate_upload_admission`).
pub const UPLOAD_ADMISSION_MAX_WAITERS_CEILING: usize =
    (H2_MAX_WINDOW_SIZE_BYTES / HTTP2_STREAM_WINDOW_BYTES) as usize - 1;

/// The HTTP/2 receive window of each inbound connection, shared by every
/// stream on it: (`upload_admission_max_waiters` + 1) ×
/// [`HTTP2_STREAM_WINDOW_BYTES`] — 68,157,440 B (65 MiB) at the default 64
/// waiters.
///
/// An upload waiting for admission has not read its content, so up to one
/// stream window of it sits unread in h2, and h2 gives connection capacity
/// back only as the application reads. With a connection window no larger
/// than one stream window, one waiting upload could hold its connection's
/// whole window and stall every other stream on it. This window is sized from
/// the admission queue's bound, the one bound the service sets on how many
/// uploads can wait at once per replica, so the parked queue alone never fills
/// a connection's window: even with every queued upload parked on one
/// connection, each holding a full stream window (behind Envoy, all of a
/// replica's waiters can share one upstream connection), one stream window
/// stays free.
///
/// That spare window is shared, not reserved. The connection stalls once its
/// unread streams together hold the whole window, (waiter cap + 1) full stream
/// windows: waiters, uploads not yet admitted (past their header, in the
/// session lookup, the write-permission check or `UpdateLayer`'s target
/// resolution), and `JoinSession`s that are not reading. A not-yet-admitted
/// upload holds its window for milliseconds while the permission decision is
/// cached or the auth service is healthy; while the auth service times out,
/// up to about 15.3 s at the defaults (`LIVE_EDIT_AUTH_TIMEOUT_MS` ×
/// `LIVE_EDIT_AUTH_MAX_ATTEMPTS`, plus at most 300 ms of backoff); and an
/// `UpdateLayer` whose target resolution runs its checkpoint recovery, up to
/// `LIVE_EDIT_RPC_DEADLINE_MS`. Joins that are not reading are the sustained
/// case.
///
/// The window is also what one connection's unread request content can reach
/// across all its streams; the service bounds neither how many connections a
/// replica accepts nor how many streams one carries, so across connections
/// that content is connections × this window (SDD §5.1, §21.2).
///
/// # Panics
///
/// When `upload_admission_max_waiters` is above
/// [`UPLOAD_ADMISSION_MAX_WAITERS_CEILING`]: the window would exceed h2's
/// largest. Configuration refuses such a waiter cap at startup.
pub const fn http2_connection_window_bytes(upload_admission_max_waiters: usize) -> u32 {
    assert!(
        upload_admission_max_waiters <= UPLOAD_ADMISSION_MAX_WAITERS_CEILING,
        "the upload admission waiter cap is above UPLOAD_ADMISSION_MAX_WAITERS_CEILING"
    );
    (upload_admission_max_waiters as u32 + 1) * HTTP2_STREAM_WINDOW_BYTES
}

/// h2's floor on its small-DATA-frame budget: 25,600 bytes
/// (`DEFAULT_DATA_FRAME_BUDGET` in h2's `proto/mod.rs`).
const H2_DATA_FRAME_BUDGET_FLOOR_BYTES: usize = 25_600;

/// The small-DATA-frame budget h2 0.4.19 keeps on a connection whose initial
/// receive window is `connection_window`: half of it, floored at 25,600 bytes
/// (h2's `DataFrameBudget::Auto`). Overdrawing it makes h2 close the
/// connection with GOAWAY `ENHANCE_YOUR_CALM` (SDD §5.1). Neither tonic nor
/// hyper exposes the budget, so this restates h2's rule for the startup log
/// and the tests.
pub const fn h2_small_frame_budget_bytes(connection_window: u32) -> usize {
    let half = connection_window as usize / 2;
    if half > H2_DATA_FRAME_BUDGET_FLOOR_BYTES {
        half
    } else {
        H2_DATA_FRAME_BUDGET_FLOOR_BYTES
    }
}

/// A `tonic` server builder carrying the service's HTTP/2 settings: a keepalive
/// PING every 30 s, the connection closed when one goes unacknowledged for 5 s,
/// the receive windows below, the per-connection limit on streams h2 resets for
/// a protocol error ([`HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS`]) and the maximum
/// frame size ([`HTTP2_MAX_FRAME_BYTES`]). `upload_admission_max_waiters` is
/// the waiter cap of the upload admission gate the server's uploads queue at
/// (`LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`); it sizes the connection window.
///
/// Callers add their layers and services and serve from a pre-bound listener.
/// TCP options (`TCP_NODELAY`, `TCP_KEEPALIVE`) are deliberately not set here:
/// `serve_with_incoming*` ignores the builder's TCP settings, so
/// `observability::startup::bind_grpc_listener` applies them to each accepted
/// connection instead.
///
/// The HTTP/2 flow-control windows, passed to the h2 handshake as its initial
/// windows: [`HTTP2_STREAM_WINDOW_BYTES`] (1 MiB) per stream, and
/// [`http2_connection_window_bytes`] of the waiter cap per connection — one
/// stream window per upload that can wait for admission, plus one, so uploads
/// parked at admission with their content unread cannot fill their
/// connection's window on their own. The spare window is shared with every
/// other unread stream on the connection — uploads not yet admitted and
/// `JoinSession`s that are not reading — so together with the waiters those
/// can still stall it ([`http2_connection_window_bytes`]). Per connection,
/// unread request content is bounded by that window; across connections it is
/// connections × the window, and the service bounds neither connections nor
/// streams per connection.
///
/// h2 sizes its small-DATA-frame budget from the connection window
/// ([`h2_small_frame_budget_bytes`]: half of it, from h2 0.4.19), so every
/// inbound connection may hold 32.5 MiB of small-frame overhead unread at the
/// default waiter cap before h2 closes it with GOAWAY `ENHANCE_YOUR_CALM`. Only
/// the connection window sets that budget: the waiter cap moves it, and so
/// would enabling `http2_adaptive_window`, which makes hyper drop the
/// connection window to 65,535 and so the budget to 32,767 bytes. The stream
/// window bounds how many unread frames one stream can hold but leaves the
/// budget alone. See SDD §5.1.
///
/// Two limits are set because tonic's builder would otherwise leave them
/// unset. The local-error-reset limit, [`HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS`]
/// (10,240 over a connection's life): tonic defaults it to `None`, which
/// hyper and h2 read as no limit (RUSTSEC-2024-0003); past it h2 closes the
/// connection with GOAWAY `ENHANCE_YOUR_CALM` "too_many_internal_resets". The
/// maximum frame size, [`HTTP2_MAX_FRAME_BYTES`] (16,384), is hyper's default
/// pinned, so the advertised value no longer rests on that default. The
/// pending-accept reset limit is deliberately not set: under tonic's `None`
/// hyper leaves h2's default of 20 in place, so a connection on which the
/// client resets more than 20 streams the service has not yet accepted, all
/// pending at once, is closed with GOAWAY `ENHANCE_YOUR_CALM`
/// "too_many_resets". Nor is `SETTINGS_MAX_CONCURRENT_STREAMS`: none is
/// advertised.
///
/// # Panics
///
/// When `upload_admission_max_waiters` is above
/// [`UPLOAD_ADMISSION_MAX_WAITERS_CEILING`] ([`http2_connection_window_bytes`]).
pub fn production_server_builder(upload_admission_max_waiters: usize) -> Server {
    Server::builder()
        .http2_keepalive_interval(Some(HTTP2_KEEPALIVE_INTERVAL))
        .http2_keepalive_timeout(Some(HTTP2_KEEPALIVE_TIMEOUT))
        .initial_stream_window_size(HTTP2_STREAM_WINDOW_BYTES)
        .initial_connection_window_size(http2_connection_window_bytes(upload_admission_max_waiters))
        .http2_max_local_error_reset_streams(Some(HTTP2_MAX_LOCAL_ERROR_RESET_STREAMS))
        .max_frame_size(HTTP2_MAX_FRAME_BYTES)
}

/// How long after the drain ends [`end_streams_after_drain`] waits before it
/// starts the paced ending (t0) — the head start the server's GOAWAY gets over
/// the streams' `UNAVAILABLE` endings.
///
/// Once the shutdown signal resolves, tonic stops accepting and signals every
/// connection task, each of which calls `graceful_shutdown()` — sending GOAWAY —
/// the next time it is polled. That is normally well under a millisecond, so
/// this only has to cover scheduling under load. It is a named constant and not
/// a setting because the ordering it buys is best effort either way: a stream
/// that ends before its connection's GOAWAY is written costs its client one
/// retry, which is refused `UNAVAILABLE` by the draining server and retried
/// again.
pub const STREAM_END_DELAY_AFTER_GOAWAY: Duration = Duration::from_millis(250);

/// E: how long past the later of the RPC deadline and the paced ending the
/// full-API server is given to close its connections before
/// [`serve_with_post_goaway_bound`] stops waiting for it.
pub const STREAM_CLOSE_GRACE: Duration = Duration::from_secs(5);

/// The cap on the final metrics and tracing flush `main.rs` runs after the
/// post-server shutdown sequence, outside its overall budget, and at the end
/// of a stop during startup.
pub const POST_BUDGET_FLUSH_BOUND: Duration = Duration::from_secs(6);

/// B_post: the longest [`serve_with_post_goaway_bound`] can wait after t0, for
/// a paced ending of `window` — max(`rpc_deadline`, `window`) +
/// [`STREAM_CLOSE_GRACE`]. By then every call in flight at t0 has ended:
/// uploads and unary calls by their deadline, streams by the end of the
/// window.
pub fn post_goaway_bound(rpc_deadline: Duration, window: Duration) -> Duration {
    rpc_deadline.max(window).saturating_add(STREAM_CLOSE_GRACE)
}

/// The shortest termination grace period under which a graceful shutdown
/// finishes before the kubelet's SIGKILL, in the worst case: the drain
/// (`drain`), [`STREAM_END_DELAY_AFTER_GOAWAY`], the bound after GOAWAY at the
/// longest window `pacing` allows ([`post_goaway_bound`]), the post-server
/// sequence's `overall_budget`, and [`POST_BUDGET_FLUSH_BOUND`]. 76.25 s at the
/// defaults.
pub fn minimum_termination_grace(
    drain: Duration,
    rpc_deadline: Duration,
    pacing: StreamPacing,
    overall_budget: Duration,
) -> Duration {
    drain
        .saturating_add(STREAM_END_DELAY_AFTER_GOAWAY)
        .saturating_add(post_goaway_bound(rpc_deadline, pacing.window(usize::MAX)))
        .saturating_add(overall_budget)
        .saturating_add(POST_BUDGET_FLUSH_BOUND)
}

/// The message of every shutdown ending and refusal.
pub(crate) const SERVER_SHUTTING_DOWN_MSG: &str = "server shutting down";

/// The status every streaming RPC ends — or is refused — with when the service
/// shuts down gracefully, and every upload refused by it: `UNAVAILABLE
/// "server shutting down"`, in every phase of `SubscribeReadLayer` and
/// `JoinSession` (SDD §5.6, §12.6).
///
/// `UNAVAILABLE` because it is the retryable code: the Kit client retries it
/// (and never retries `OK`). It carries the refusal detail
/// [`SERVER_SHUTTING_DOWN`](crate::error::refusal_reason::SERVER_SHUTTING_DOWN)
/// with a `retry_delay` in [`SHUTDOWN_RETRY_DELAY`](crate::error::SHUTDOWN_RETRY_DELAY),
/// near zero: the server already paces its stream endings
/// ([`end_streams_after_drain`]), and this replica has stopped accepting
/// connections (readiness `NOT_SERVING`, GOAWAY sent, listener closed), so the
/// retry reaches another replica. One constructor, so the RPCs cannot drift
/// apart.
pub(crate) fn server_shutting_down() -> Status {
    crate::error::RefusalDetail::server_shutting_down().unavailable(SERVER_SHUTTING_DOWN_MSG)
}

/// The message of every refusal and stream ending of a replica that has lost
/// contact with the store it reads the disaster-recovery hold from (SDD
/// SR-6.5).
pub(crate) const HOLD_VERDICT_STALE_MSG: &str =
    "this replica lost contact with the backing store and stopped serving until it restarts; \
     retry";

/// The status every refusal and stream ending of a replica whose hold verdict
/// went stale carries (SDD SR-6.5): `UNAVAILABLE` [`HOLD_VERDICT_STALE_MSG`],
/// with [`UPSTREAM_UNAVAILABLE`](crate::error::refusal_reason::UPSTREAM_UNAVAILABLE)
/// and its retry delay. Never `SERVICE_HELD`: no fleet hold is in force, so
/// there is no incident. The streams' counterpart of a fleet hold's
/// [`RefusalDetail::service_held_status`](crate::error::RefusalDetail::service_held_status).
pub(crate) fn hold_verdict_stale() -> Status {
    crate::error::RefusalDetail::upstream_unavailable().unavailable(HOLD_VERDICT_STALE_MSG)
}

/// The full-API server's graceful-shutdown signal: resolves once `drained`
/// does, and ends the live streams, paced, starting just after the server's
/// GOAWAY.
///
/// `drained` is the readiness drain — `main.rs` flips readiness to
/// `NOT_SERVING`, sleeps `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` and closes the
/// listening socket. Streams keep serving through it: nothing ends one until
/// it resolves (SDD §5.6).
///
/// When `drained` resolves this spawns the paced ending and returns. Returning
/// is what lets tonic's `serve_with_incoming_shutdown` stop accepting and start
/// the HTTP/2 graceful shutdown of every connection, so the GOAWAY goes out
/// first. [`STREAM_END_DELAY_AFTER_GOAWAY`] later (t0) the spawned task runs
/// [`end_streams_paced`] with `pacing` and `rng`: new streaming RPCs and
/// waiting uploads are refused from t0, and the live `SubscribeReadLayer` and
/// `JoinSession` streams end `UNAVAILABLE "server shutting down"` in random
/// order over at most `pacing.max_window`, at max(`pacing.rate_per_sec`,
/// N / `pacing.max_window`) per second for N streams, so their reconnects
/// reach the other replicas as a ramp.
/// Then the service shutdown token is cancelled, and tonic, which waits for
/// every connection to close, can return.
///
/// The spawn is what puts the endings *behind* tonic's graceful shutdown:
/// tonic polls this future to completion before it starts, so ending inline
/// here would end the streams on connections that have not been told to go
/// away, and a client's immediate retry would reach this same draining server.
pub async fn end_streams_after_drain<F, R>(
    drained: F,
    shutdown: GracefulShutdown,
    pacing: StreamPacing,
    mut rng: R,
) where
    F: Future<Output = ()>,
    R: rand::Rng + Send + 'static,
{
    drained.await;
    tokio::spawn(async move {
        tokio::time::sleep(STREAM_END_DELAY_AFTER_GOAWAY).await;
        end_streams_paced(&shutdown, pacing, &mut rng).await;
    });
}

/// Run the full-API server's `serve` future, bounded after GOAWAY, then
/// [`GracefulShutdown::cancel`].
///
/// The bound: from t0 (when the paced ending starts, [`GracefulShutdown::begun`])
/// the server is given the later of `rpc_deadline` and the end of the paced
/// ending, plus [`STREAM_CLOSE_GRACE`] — [`post_goaway_bound`] of the window
/// it ran. By then every call in flight at t0 has ended (uploads and unary
/// calls by their deadline, streams by the end of the window), so a server
/// still waiting is waiting on a connection that cannot finish closing, such
/// as one whose client stopped reading. Then this logs one WARN, with
/// `open_connections()` — the connections still open — and drops `serve`.
/// tonic's connection tasks are spawned, so they are not torn down: a healthy
/// one finishes flushing, and a stalled one ends with the process.
///
/// Returns `serve`'s output, or `None` when the bound fired. Either way it
/// then cancels `shutdown` (idempotent), so the post-server drains never wait
/// on a stream that was never scheduled.
pub async fn serve_with_post_goaway_bound<S, T, C>(
    serve: S,
    shutdown: &GracefulShutdown,
    rpc_deadline: Duration,
    open_connections: C,
) -> Option<T>
where
    S: Future<Output = T>,
    C: Fn() -> usize,
{
    let bound = async {
        shutdown.begun().cancelled().await;
        let t0 = tokio::time::Instant::now();
        tokio::join!(
            tokio::time::sleep(rpc_deadline),
            shutdown.service_shutdown_token().cancelled()
        );
        tokio::time::sleep(STREAM_CLOSE_GRACE).await;
        t0.elapsed()
    };
    let outcome = tokio::select! {
        biased;
        served = serve => Some(served),
        waited = bound => {
            tracing::warn!(
                open_connections = open_connections(),
                waited_ms = u64::try_from(waited.as_millis()).unwrap_or(u64::MAX),
                rpc_deadline_ms = u64::try_from(rpc_deadline.as_millis()).unwrap_or(u64::MAX),
                close_grace_ms = u64::try_from(STREAM_CLOSE_GRACE.as_millis()).unwrap_or(u64::MAX),
                "the gRPC server did not close every connection within its bound after GOAWAY \
                 (the later of LIVE_EDIT_RPC_DEADLINE_MS and the end of the paced stream ending, \
                 plus the close grace); continuing the shutdown sequence — the connections still \
                 open finish flushing on their own or end with the process"
            );
            None
        }
    };
    shutdown.cancel();
    outcome
}

/// Drain the streaming RPCs' tasks after the server has returned, before the
/// Valkey pools quit (SDD §5.6): the `JoinSession` tasks on
/// `presence_cleanup` (`LiveEditService::presence_cleanup_tracker`) and the
/// `SubscribeReadLayer` subscription tasks on `subscriptions`
/// (`LiveEditService::subscription_tracker`).
///
/// Both trackers are closed, then waited concurrently, each bounded by `budget`
/// ([`run_post_server_sequence`] passes the per-task shutdown budget,
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`, or less when its shared deadline is
/// nearer), so one stuck task cannot use up the overall shutdown budget. Each outcome is logged: INFO when the tracker
/// drained, WARN with the number of tasks still pending when it did not. The
/// trackers' `is_empty()` tells a caller the same thing afterwards.
///
/// What the drain is for:
///
/// * **JoinSession** (b.w29 item 4), so every connection's `PeerLeft` reaches
///   peers via pub/sub while the pools are still up. The join's enrolled
///   shutdown token — cancelled in its batch of the paced ending
///   ([`end_streams_after_drain`]), or by [`GracefulShutdown::cancel`] once
///   the server has returned — ends each join's `run_bidi_loop` — or its
///   event send — through the shutdown arm. A join not yet in its loop stops
///   at its check after `acquire_session_cancel`, or leaves the loop on entry.
///   Each join task then runs its teardown, which stops its lease keeper (the
///   keeper also stops on the shutdown token by itself, and runs on this
///   tracker too). It then puts its `UNAVAILABLE` into a slot reserved for it,
///   so a client that stopped reading cannot hold the task.
/// * **SubscribeReadLayer**, so every subscription ends with its own
///   shutdown status before the pools quit. The same endings end every
///   subscription `UNAVAILABLE`. Each task drops its consumer on the way out,
///   which unregisters it from the stream multiplexer. Subscriptions hold no
///   Valkey connection; the readers' blocked `XREAD BLOCK`s are woken by
///   `StreamMux::shutdown` inside `ValkeyClient::quit`, which also ends any
///   subscription still registered, before the reader connections' `QUIT`s.
///
/// `close()` does not fence spawns (a closed `TaskTracker` still accepts them);
/// it only lets `wait()` finish once the tracker is empty. New streams stop
/// because the server has stopped serving, and an enrollment after the paced
/// ending started is born cancelled, which refuses any stream that opened on a
/// connection before its GOAWAY took effect.
pub async fn drain_stream_tasks(
    presence_cleanup: &TaskTracker,
    subscriptions: &TaskTracker,
    budget: Duration,
) {
    presence_cleanup.close();
    subscriptions.close();
    let (cleanup_drain_outcome, subscription_drain_outcome) = tokio::join!(
        tokio::time::timeout(budget, presence_cleanup.wait()),
        tokio::time::timeout(budget, subscriptions.wait()),
    );
    match cleanup_drain_outcome {
        Ok(()) => tracing::info!("presence cleanup futures drained"),
        Err(_) => tracing::warn!(
            pending = presence_cleanup.len(),
            "timed out waiting for presence cleanup futures; some peers may not have received PeerLeft"
        ),
    }
    match subscription_drain_outcome {
        Ok(()) => tracing::info!("SubscribeReadLayer subscription tasks drained"),
        Err(_) => tracing::warn!(
            pending = subscriptions.len(),
            "timed out waiting for SubscribeReadLayer subscription tasks; those still \
             registered with the stream multiplexer are ended UNAVAILABLE by its shutdown \
             before the pools quit"
        ),
    }
}

/// What [`run_post_server_sequence`]'s join report reads besides each join's
/// result: the registry for the panic counter, and the writer inventory's
/// abort set, to tell a task a hold or stale-verdict entry aborted from one
/// cancelled by anything else.
pub struct BackgroundTaskJoinReport {
    /// Where `background_task_panics_total{task}` is counted.
    pub metrics: Arc<Metrics>,
    /// The writer inventory's abort set (`grpc::hold`).
    pub hold_writers: Arc<HoldWriterAborts>,
}

/// Classify the outcome of one background-task join during shutdown, emit the
/// appropriate log line, and tick `background_task_panics_total{task}` when
/// the task panicked (b.3z7 H-obs-3).
///
/// `bound` is the join's bound. A task still running when a zero bound
/// elapses was polled once and never waited for, so it is reported as such,
/// not as one that outlived the per-task budget.
fn report_background_task_result(
    report: &BackgroundTaskJoinReport,
    task: &str,
    bound: Duration,
    result: Result<Result<(), tokio::task::JoinError>, tokio::time::error::Elapsed>,
) {
    match result {
        Ok(Ok(())) => tracing::info!(task = task, "background task stopped cleanly"),
        // Cancelled after a disaster-recovery hold's entry, or a stale hold
        // verdict's, aborted the writer inventory (`grpc::hold`; both abort
        // the same set through the same abort set), before this join: no
        // failure. Any other cancellation takes the failure arm below.
        Ok(Err(join_err)) if join_err.is_cancelled() && report.hold_writers.is_aborted() => {
            tracing::info!(
                task = task,
                "background task was aborted earlier, at hold or stale-verdict entry"
            )
        }
        Ok(Err(join_err)) => {
            if join_err.is_panic() {
                report.metrics.inc_background_task_panic(task);
            }
            tracing::error!(task = task, error = %join_err, "background task failed during shutdown");
        }
        Err(_elapsed) if bound.is_zero() => {
            tracing::warn!(
                task = task,
                "background task not waited for: the shared shutdown deadline was already spent"
            );
        }
        Err(_elapsed) => {
            tracing::warn!(
                task = task,
                "background task did not stop within per-task shutdown budget"
            );
        }
    }
}

/// The bound of a wait that starts now: the per-task budget, or what is left
/// before `deadline` when that is less. Zero once the deadline has passed:
/// `tokio::time::timeout` polls its future before its timer, so a zero-bound
/// wait still sees a future that is already done.
fn pre_quit_bound(per_task: Duration, deadline: Instant) -> Duration {
    per_task.min(deadline.saturating_duration_since(Instant::now()))
}

/// How far below the per-task budget the shared deadline must cut a wait's
/// bound before [`run_post_server_sequence`] logs its cut WARN: 100 ms.
///
/// A wait that times out returns at its bound plus however long the runtime
/// takes to wake it and run the next step, a few milliseconds in real time.
/// With every wait stuck at an overall budget of exactly 3T + 15 (the
/// defaults), those milliseconds put the join's bound just under T although
/// nothing material was cut, and a WARN then would contradict the rule that
/// such a budget gives every wait its full per-task budget. A shortfall
/// within the tolerance is that scheduling slack, not a cut.
pub const SHUTDOWN_CUT_WARN_TOLERANCE: Duration = Duration::from_millis(100);

/// Whether the shared deadline cut `bound` materially below `per_task`: by
/// more than [`SHUTDOWN_CUT_WARN_TOLERANCE`].
fn cut_materially(bound: Duration, per_task: Duration) -> bool {
    bound.saturating_add(SHUTDOWN_CUT_WARN_TOLERANCE) < per_task
}

/// The WARN for a wait the shared deadline cut: logged only when `bound` is
/// materially below the per-task budget ([`cut_materially`]), so a wait cut
/// by the per-task budget alone logs only its own timeout line.
fn warn_if_cut(wait: &'static str, bound: Duration, per_task: Duration, pending: usize) {
    if cut_materially(bound, per_task) {
        tracing::warn!(
            wait,
            pending,
            bound_ms = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX),
            "shutdown wait cut short to keep Valkey quit()'s headroom"
        );
    }
}

/// The post-server shutdown sequence (SDD §5.6 steps 4–5): everything between
/// the full-API server returning and the final metrics and tracing flush,
/// bounded by `overall_budget` (`LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS`, O)
/// and ending with `ValkeyClient::quit()`.
///
/// # The shared deadline
///
/// Every wait before `quit()` is bounded by the earlier of `task_budget`
/// (`LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`, T) and a deadline computed once, when
/// the sequence starts: the start plus O minus
/// `config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS` (15 s). So `quit()` always
/// starts with at least that headroom left, which the `const` assertion in
/// `storage::client` shows covers its own sequential bounds. There are three
/// such waits, one after another, so they run to their full T whenever
/// O ≥ 3T + 15 (the defaults sit exactly there), short only of the few
/// milliseconds each earlier timed-out wait overran its bound by; below that
/// they share O − 15 in order and the later ones are cut, the join first. A
/// wait whose bound was cut below T by more than
/// [`SHUTDOWN_CUT_WARN_TOLERANCE`] and that timed out logs one more WARN,
/// `shutdown wait cut short to keep Valkey quit()'s headroom`, naming the
/// wait, what was still pending (for the join, the count and the tasks'
/// names) and the bound.
///
/// # In order
///
/// 1. **W1, concurrently:** the layer store's tracked cleanups
///    (`layer_cleanup`, handed the bound; `main.rs` passes
///    `LayerStore::wait_for_cleanup`, then `abort_cleanup_tasks`, and it
///    answers the number of cleanups still pending) beside the streaming RPCs'
///    task drains ([`drain_stream_tasks`]). Neither feeds the other: the
///    layer-cleanup tracker is fed only by `CreateLayer` and `UpdateLayer`
///    and its own finisher, and no layer cleanup spawns a stream task.
/// 2. **W2:** the session-checkpoint and base-bootstrap tracker
///    (`session_checkpoints`), closed and waited, then
///    `session_checkpoint_abort` cancelled, so only what outlived the drain
///    is aborted. It stays after W1 because both W1 trackers feed it: an
///    overwrite's tail enqueues a session-checkpoint write, and a stream's
///    recovery-hook restore spawns a base bootstrap on it.
/// 3. Every token in `cancels`, fired together, so the background tasks
///    unwind in parallel.
/// 4. **W3:** one concurrent join of every task in `tasks`, each under the
///    same bound, computed once at the join's start, and reported by name
///    (`report`). A task that outlives its join is detached, not aborted.
///    When that bound is zero (below O = 3T + 15, the earlier waits spent
///    the shared deadline), each task is polled once and one still running
///    is reported `background task not waited for: the shared shutdown
///    deadline was already spent`, which includes a task that was about to
///    end on its cancel; the per-task line, `background task did not stop
///    within per-task shutdown budget`, is kept for a non-zero bound.
/// 5. `quit`.
///
/// The outer `timeout(overall_budget)` stays as a backstop for the steps with
/// no timeout of their own (the token fires, the tracker close and the
/// reporting between the waits): it logs `shutdown overall budget exceeded`
/// and drops the sequence where it stands. On a sequence that finishes, it logs
/// `shutdown complete`.
#[allow(clippy::too_many_arguments)]
pub async fn run_post_server_sequence<L, LF, Q>(
    overall_budget: Duration,
    task_budget: Duration,
    layer_cleanup: L,
    presence_cleanup: &TaskTracker,
    subscriptions: &TaskTracker,
    session_checkpoints: &TaskTracker,
    session_checkpoint_abort: &CancellationToken,
    cancels: &[CancellationToken],
    tasks: Vec<(&'static str, JoinHandle<()>)>,
    report: &BackgroundTaskJoinReport,
    quit: Q,
) where
    L: FnOnce(Duration) -> LF,
    LF: Future<Output = usize>,
    Q: Future<Output = ()>,
{
    let pre_quit_deadline = Instant::now()
        + overall_budget.saturating_sub(Duration::from_secs(
            crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS,
        ));

    let sequence = async {
        // W1: the layer cleanups beside the stream drains.
        tracing::info!("waiting for pending layer cleanup tasks");
        let w1_bound = pre_quit_bound(task_budget, pre_quit_deadline);
        let (layer_pending, ()) = tokio::join!(
            layer_cleanup(w1_bound),
            drain_stream_tasks(presence_cleanup, subscriptions, w1_bound),
        );
        tracing::info!("layer cleanup tasks finished");
        // A defensive guard: W1 is the first wait, so its bound is below T
        // only when O < T + 15. `Config` refuses that, so in the binary these
        // never fire; they cover a library caller that passes such an O.
        if layer_pending > 0 {
            warn_if_cut("layer_cleanup", w1_bound, task_budget, layer_pending);
        }
        if !presence_cleanup.is_empty() {
            warn_if_cut(
                "presence_cleanup",
                w1_bound,
                task_budget,
                presence_cleanup.len(),
            );
        }
        if !subscriptions.is_empty() {
            warn_if_cut("subscriptions", w1_bound, task_budget, subscriptions.len());
        }

        // W2: the session-checkpoint writes and base bootstraps. Their writer
        // tasks need only the Valkey pools and the backup store, both alive
        // until `quit()`, so a write in flight lands cleanly within the bound.
        session_checkpoints.close();
        let w2_bound = pre_quit_bound(task_budget, pre_quit_deadline);
        match tokio::time::timeout(w2_bound, session_checkpoints.wait()).await {
            Ok(()) => tracing::info!("session-checkpoint writes drained"),
            Err(_) => {
                tracing::warn!(
                    pending = session_checkpoints.len(),
                    "timed out waiting for session-checkpoint writes; some lifecycle events may \
                     not be reflected in the checkpoint store"
                );
                warn_if_cut(
                    "session_checkpoints",
                    w2_bound,
                    task_budget,
                    session_checkpoints.len(),
                );
            }
        }
        // After the drain: abort the session-checkpoint writes and base
        // bootstraps that outlived it.
        session_checkpoint_abort.cancel();

        // Every background task's token at once, so they unwind in parallel;
        // the join below is where they are waited for.
        for cancel in cancels {
            cancel.cancel();
        }

        // W3: one concurrent join. All were cancelled at the same instant, so
        // a task that obeys its cancel costs only its own unwind time, and one
        // stuck task costs one bound, not one bound per task after it. A zero
        // bound waits for nothing, so a task still running then is reported
        // as not waited for, not as one that outlived its budget (step 4).
        let w3_bound = pre_quit_bound(task_budget, pre_quit_deadline);
        let joined =
            futures::future::join_all(tasks.into_iter().map(|(task, handle)| async move {
                (task, tokio::time::timeout(w3_bound, handle).await)
            }))
            .await;
        let mut still_running: Vec<&'static str> = Vec::new();
        for (task, result) in joined {
            if result.is_err() {
                still_running.push(task);
            }
            report_background_task_result(report, task, w3_bound, result);
        }
        if !still_running.is_empty() && cut_materially(w3_bound, task_budget) {
            tracing::warn!(
                wait = "background_tasks",
                pending = still_running.len(),
                pending_tasks = ?still_running,
                bound_ms = u64::try_from(w3_bound.as_millis()).unwrap_or(u64::MAX),
                "shutdown wait cut short to keep Valkey quit()'s headroom"
            );
        }

        tracing::info!("closing Valkey connections");
        quit.await;
    };

    match tokio::time::timeout(overall_budget, sequence).await {
        Ok(()) => tracing::info!("shutdown complete"),
        Err(_) => tracing::warn!(
            overall_budget_secs = overall_budget.as_secs(),
            "shutdown overall budget exceeded; exiting anyway to fit terminationGracePeriodSeconds"
        ),
    }
}

#[cfg(test)]
mod tests {
    //! The shutdown sequencing contract of [`end_streams_after_drain`] — the
    //! drain, the GOAWAY head start and the paced ending's schedule — the
    //! bound [`serve_with_post_goaway_bound`] puts on the server after GOAWAY,
    //! the grace arithmetic, what [`drain_stream_tasks`] tells the operator
    //! when its budget runs out, and the shared pre-`quit()` deadline of
    //! [`run_post_server_sequence`] with its cut WARN, on paused time: every wait is a
    //! virtual-clock deadline, so the ordering is asserted exactly and nothing
    //! here depends on how fast the machine is.
    //! The enrollment registry and the end order are pinned in
    //! `graceful_shutdown.rs`'s tests; the real-server end-to-end checks
    //! (GOAWAY, then the paced `UNAVAILABLE` endings on live streams, the
    //! closed listener, the bound on a stalled connection) are
    //! `tests/integration/grpc/shutdown_ending_test.rs`.
    //!
    //! Also the arithmetic of the HTTP/2 receive windows
    //! [`production_server_builder`] sets and of the small-frame budget h2
    //! derives from the connection window; what a parked upload's unread
    //! content does to its connection's other streams is
    //! `tests/integration/grpc/upload_window_test.rs`.

    use super::*;
    use crate::grpc::graceful_shutdown::STREAM_END_BATCH_INTERVAL;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    /// Streams keep serving through the drain: nothing cancels the service
    /// shutdown token, and the signal does not resolve, until `drained` does —
    /// however long the drain runs.
    #[tokio::test(start_paused = true)]
    async fn end_streams_after_drain_cancels_nothing_until_the_drain_resolves() {
        let shutdown = GracefulShutdown::new();
        let (drained_tx, drained_rx) = tokio::sync::oneshot::channel::<()>();
        let signal = tokio::spawn(end_streams_after_drain(
            async move {
                let _ = drained_rx.await;
            },
            shutdown.clone(),
            StreamPacing::DEFAULT,
            seeded_rng(),
        ));

        // Far longer than any production drain.
        tokio::time::sleep(Duration::from_secs(3600)).await;
        assert!(
            !shutdown.is_cancelled() && !shutdown.begun().is_cancelled(),
            "the service shutdown token was cancelled while the drain was still running; live \
             streams must keep serving until it ends"
        );
        assert!(
            !signal.is_finished(),
            "the shutdown signal resolved before the drain did, so tonic would stop accepting \
             and send GOAWAY mid-drain"
        );

        drained_tx
            .send(())
            .expect("the signal task still holds the receiver");
        tokio::time::timeout(Duration::from_secs(1), signal)
            .await
            .expect("the signal must resolve once the drain has")
            .expect("the signal task must not panic");
    }

    /// Once the drain resolves the signal returns at once — which is what
    /// lets tonic start its graceful shutdown and send GOAWAY — and the token
    /// is cancelled [`STREAM_END_DELAY_AFTER_GOAWAY`] later, not inline. A
    /// cancel inline would end the streams on connections that have not yet
    /// been told to go away, and a client's immediate retry would land on this
    /// same draining server.
    #[tokio::test(start_paused = true)]
    async fn end_streams_after_drain_returns_at_once_and_cancels_the_token_after_the_goaway_delay()
    {
        let shutdown = GracefulShutdown::new();
        let returned_at = {
            end_streams_after_drain(
                async {},
                shutdown.clone(),
                StreamPacing::DEFAULT,
                seeded_rng(),
            )
            .await;
            tokio::time::Instant::now()
        };
        assert!(
            !shutdown.is_cancelled() && !shutdown.begun().is_cancelled(),
            "the token was cancelled inline, before tonic could send GOAWAY"
        );

        tokio::time::sleep(STREAM_END_DELAY_AFTER_GOAWAY - Duration::from_millis(1)).await;
        assert!(
            !shutdown.is_cancelled() && !shutdown.begun().is_cancelled(),
            "the token was cancelled before the GOAWAY head start had elapsed"
        );

        // No stream is enrolled, so the paced ending is one batch at t0.
        tokio::time::timeout(Duration::from_secs(1), shutdown.cancelled())
            .await
            .expect("the token must be cancelled once the GOAWAY head start has elapsed");
        assert!(
            returned_at.elapsed() >= STREAM_END_DELAY_AFTER_GOAWAY,
            "on the virtual clock the cancel landed {:?} after the signal returned, inside the \
             {STREAM_END_DELAY_AFTER_GOAWAY:?} GOAWAY head start",
            returned_at.elapsed()
        );
    }

    /// Sink for [`capture_json_events`]. Local copy of the tree-wide
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

    /// Drive `f` under a JSON capture and return every event it emitted,
    /// parsed. `WithSubscriber`, so the capture follows the future across its
    /// awaits.
    async fn capture_json_events<F, T>(f: F) -> (T, Vec<serde_json::Value>)
    where
        F: Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(std::sync::Arc::clone(&buf)))
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

    /// A subscription drain that outruns its budget returns at the budget and
    /// tells the operator so: one WARN, with the number of subscription tasks
    /// still pending, saying the stream multiplexer's shutdown ends the ones
    /// still registered before the pools quit. The presence drain, which
    /// finished, logs its INFO line and no WARN.
    #[tokio::test(start_paused = true)]
    async fn a_subscription_drain_past_its_budget_logs_one_warn_with_the_pending_count() {
        const BUDGET: Duration = Duration::from_secs(5);
        const DRAIN_TIMEOUT_MSG: &str =
            "timed out waiting for SubscribeReadLayer subscription tasks; those still registered \
             with the stream multiplexer are ended UNAVAILABLE by its shutdown before the pools \
             quit";
        let presence_cleanup = TaskTracker::new();
        let subscriptions = TaskTracker::new();
        for _ in 0..2 {
            subscriptions.spawn(std::future::pending::<()>());
        }

        let started = tokio::time::Instant::now();
        let ((), events) = capture_json_events(drain_stream_tasks(
            &presence_cleanup,
            &subscriptions,
            BUDGET,
        ))
        .await;
        assert_eq!(
            started.elapsed(),
            BUDGET,
            "on the virtual clock the drain gives up exactly at its budget"
        );

        let with_message = |msg: &str| -> Vec<&serde_json::Value> {
            events
                .iter()
                .filter(|e| e["fields"]["message"] == msg)
                .collect()
        };
        let warns = with_message(DRAIN_TIMEOUT_MSG);
        assert_eq!(warns.len(), 1, "one drain-timeout WARN; got {events:#?}");
        assert_eq!(warns[0]["level"], "WARN", "{}", warns[0]);
        assert_eq!(
            warns[0]["fields"]["pending"], 2,
            "the WARN counts the subscription tasks still pending; {}",
            warns[0]
        );
        let presence = with_message("presence cleanup futures drained");
        assert_eq!(presence.len(), 1, "got {events:#?}");
        assert_eq!(presence[0]["level"], "INFO");
        assert!(
            with_message("SubscribeReadLayer subscription tasks drained").is_empty(),
            "a drain that timed out must not also report it drained; got {events:#?}"
        );
    }

    /// The one shutdown status both streaming RPCs end with: retryable
    /// `UNAVAILABLE`, which the Kit client retries and `OK` would not be,
    /// carrying the refusal detail `SERVER_SHUTTING_DOWN` in the service's
    /// domain, no metadata, and a `retry_delay` under 100 ms — the server has
    /// already paced its endings, so the client reconnects at once.
    #[test]
    fn server_shutting_down_is_retryable_unavailable_with_the_shutdown_retry_hint() {
        use tonic_types::StatusExt;
        for _ in 0..32 {
            let status = server_shutting_down();
            assert_eq!(status.code(), tonic::Code::Unavailable);
            assert_eq!(status.message(), "server shutting down");
            let info = status
                .get_details_error_info()
                .expect("the shutdown status carries an ErrorInfo");
            assert_eq!(info.reason, "SERVER_SHUTTING_DOWN");
            assert_eq!(info.domain, "nvidia.omniverse.live.v1alpha");
            assert!(info.metadata.is_empty(), "{info:?}");
            let delay = status
                .get_details_retry_info()
                .and_then(|r| r.retry_delay)
                .expect("the shutdown status carries a RetryInfo delay");
            assert!(
                delay < Duration::from_millis(100),
                "the shutdown retry delay is drawn from [0, 100 ms); got {delay:?}"
            );
        }
    }

    // ── The paced ending's schedule, through `end_streams_after_drain` ───────

    /// The readiness drain every schedule test runs before the paced ending.
    const DRAIN: Duration = Duration::from_secs(5);

    /// The virtual instant `token` is cancelled at.
    fn cancelled_at(
        token: &tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<tokio::time::Instant> {
        let token = token.clone();
        tokio::spawn(async move {
            token.cancelled().await;
            tokio::time::Instant::now()
        })
    }

    /// One shutdown of `streams` enrolled streams, as `main.rs` runs it: a
    /// [`DRAIN`]-long drain, then [`end_streams_after_drain`] with `pacing`.
    struct PacedRun {
        /// The drain's end plus [`STREAM_END_DELAY_AFTER_GOAWAY`]: when the
        /// paced ending must start.
        t0: tokio::time::Instant,
        begun_at: tokio::time::Instant,
        parent_at: tokio::time::Instant,
        /// Each enrolled stream's cancel instant, in enrollment order.
        endings: Vec<tokio::time::Instant>,
    }

    impl PacedRun {
        /// The endings grouped by batch: each distinct instant, as an offset
        /// from `t0`, with the number of streams ended at it, in time order.
        fn batches(&self) -> Vec<(Duration, usize)> {
            let mut grouped = std::collections::BTreeMap::new();
            for at in &self.endings {
                *grouped.entry(at.duration_since(self.t0)).or_insert(0) += 1;
            }
            grouped.into_iter().collect()
        }
    }

    async fn paced_shutdown(streams: usize, pacing: StreamPacing) -> PacedRun {
        let shutdown = GracefulShutdown::new();
        let mut enrollments = Vec::new();
        let mut watchers = Vec::new();
        for _ in 0..streams {
            let (token, enrollment) = shutdown.enroll_stream();
            watchers.push(cancelled_at(&token));
            enrollments.push(enrollment);
        }
        let begun = cancelled_at(shutdown.begun());
        let parent = tokio::spawn({
            let shutdown = shutdown.clone();
            async move {
                shutdown.cancelled().await;
                tokio::time::Instant::now()
            }
        });
        end_streams_after_drain(
            tokio::time::sleep(DRAIN),
            shutdown.clone(),
            pacing,
            seeded_rng(),
        )
        .await;
        let t0 = tokio::time::Instant::now() + STREAM_END_DELAY_AFTER_GOAWAY;
        let parent_at = parent.await.expect("watcher");
        let begun_at = begun.await.expect("watcher");
        let mut endings = Vec::new();
        for watcher in watchers {
            endings.push(watcher.await.expect("watcher"));
        }
        drop(enrollments);
        PacedRun {
            t0,
            begun_at,
            parent_at,
            endings,
        }
    }

    /// **The paced ending's schedule.** 23 streams at 50/s (cap 20 s): a
    /// window of 460 ms, so T = 5 batches, Δ = 100 ms apart. The first
    /// batch runs exactly at the drain's end plus the GOAWAY head start (t0),
    /// the last at t0 + 4Δ; every batch ends ⌊23/5⌋ or ⌈23/5⌉ streams and every
    /// stream ends in exactly one; `begun` fires at t0; and the parent is
    /// cancelled only with the last batch. Today's single cancel would end all
    /// 23 at t0.
    #[tokio::test(start_paused = true)]
    async fn end_streams_after_drain_ends_the_streams_in_paced_batches_starting_at_t0() {
        const STREAMS: usize = 23;
        let pacing = StreamPacing {
            rate_per_sec: 50,
            max_window: Duration::from_secs(20),
        };
        assert_eq!(pacing.batches(STREAMS), 5, "fixture: five batches");
        let run = paced_shutdown(STREAMS, pacing).await;

        assert_eq!(
            run.begun_at, run.t0,
            "the paced ending starts (`begun`) exactly at the drain's end plus the GOAWAY head \
             start"
        );
        let batches = run.batches();
        let offsets: Vec<Duration> = batches.iter().map(|(at, _)| *at).collect();
        assert_eq!(
            offsets,
            (0..5u32)
                .map(|k| STREAM_END_BATCH_INTERVAL * k)
                .collect::<Vec<_>>(),
            "one batch every {STREAM_END_BATCH_INTERVAL:?} from t0, the first at t0 and the last \
             at t0 + (T−1)Δ; got {batches:?}"
        );
        for (at, ended) in &batches {
            assert!(
                (STREAMS / 5..=STREAMS.div_ceil(5)).contains(ended),
                "the batch at t0 + {at:?} ended {ended} streams; each batch ends ⌊N/T⌋ or \
                 ⌈N/T⌉. Batches: {batches:?}"
            );
        }
        assert_eq!(
            batches.iter().map(|(_, n)| n).sum::<usize>(),
            STREAMS,
            "every stream ends in exactly one batch"
        );
        assert_eq!(
            run.parent_at.duration_since(run.t0),
            STREAM_END_BATCH_INTERVAL * 4,
            "the service shutdown token is cancelled with the last batch, not before"
        );
    }

    /// **The switches.** The cap binds when N/R exceeds it: 50 streams at
    /// 10/s would take 5 s, but the 2 s cap makes 20 batches ending at
    /// t0 + 1.9 s. A rate of 0, or a cap of 0, ends every stream — and
    /// cancels the parent — at t0: today's single cancel, the rollback
    /// switch. With no stream enrolled the parent is cancelled at t0 too.
    #[tokio::test(start_paused = true)]
    async fn end_streams_after_drain_caps_the_window_and_either_switch_ends_everything_at_t0() {
        let capped = paced_shutdown(
            50,
            StreamPacing {
                rate_per_sec: 10,
                max_window: Duration::from_secs(2),
            },
        )
        .await;
        let batches = capped.batches();
        assert_eq!(
            batches.len(),
            20,
            "the 2 s cap: 20 batches; got {batches:?}"
        );
        assert_eq!(
            batches.last().map(|(at, _)| *at),
            Some(Duration::from_millis(1_900)),
            "the last batch is at t0 + 1.9 s, inside the 2 s cap"
        );
        assert!(
            batches.iter().all(|(_, n)| (2..=3).contains(n)),
            "50 streams over 20 batches: 2 or 3 each; got {batches:?}"
        );
        assert_eq!(
            capped.parent_at.duration_since(capped.t0),
            Duration::from_millis(1_900)
        );

        for (what, pacing) in [
            (
                "rate 0",
                StreamPacing {
                    rate_per_sec: 0,
                    max_window: Duration::from_secs(20),
                },
            ),
            (
                "cap 0",
                StreamPacing {
                    rate_per_sec: 25,
                    max_window: Duration::ZERO,
                },
            ),
        ] {
            let run = paced_shutdown(30, pacing).await;
            assert_eq!(
                run.batches(),
                vec![(Duration::ZERO, 30)],
                "{what}: every stream ends at t0"
            );
            assert_eq!(run.begun_at, run.t0, "{what}");
            assert_eq!(
                run.parent_at, run.t0,
                "{what}: the parent is cancelled at t0"
            );
        }

        let empty = paced_shutdown(0, StreamPacing::DEFAULT).await;
        assert_eq!(
            empty.parent_at, empty.t0,
            "no stream enrolled: one empty batch at t0, then the parent"
        );
    }

    // ── The bound after GOAWAY (b.uqx) ───────────────────────────────────────

    const RPC_DEADLINE: Duration = Duration::from_secs(30);

    const BOUND_WARN_MSG: &str = "the gRPC server did not close every connection within its \
                                  bound after GOAWAY (the later of LIVE_EDIT_RPC_DEADLINE_MS \
                                  and the end of the paced stream ending, plus the close \
                                  grace); continuing the shutdown sequence — the connections \
                                  still open finish flushing on their own or end with the \
                                  process";

    /// Start the paced ending on `shutdown` at once (no drain), with
    /// `pacing`.
    async fn begin_now(shutdown: &GracefulShutdown, pacing: StreamPacing) {
        end_streams_after_drain(async {}, shutdown.clone(), pacing, seeded_rng()).await;
    }

    /// A server that returns on its own is returned from, its output passed
    /// through, with no WARN; the shutdown handle is cancelled after it, so
    /// the post-server drains never wait on a stream that was never
    /// scheduled.
    #[tokio::test(start_paused = true)]
    async fn serve_with_post_goaway_bound_passes_through_a_server_that_returns() {
        let shutdown = GracefulShutdown::new();
        let (straggler, _straggler) = shutdown.enroll_stream();
        let (served, events) = capture_json_events(serve_with_post_goaway_bound(
            async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                "served"
            },
            &shutdown,
            RPC_DEADLINE,
            || 0,
        ))
        .await;
        assert_eq!(served, Some("served"));
        assert!(
            events
                .iter()
                .all(|e| e["fields"]["message"] != BOUND_WARN_MSG),
            "a server that returned logs no bound WARN; got {events:#?}"
        );
        assert!(
            shutdown.is_cancelled() && straggler.is_cancelled(),
            "the handle is cancelled once the server has returned"
        );
    }

    /// No bound runs before the paced ending starts: a server waiting through
    /// the drain, however long, is waited for.
    #[tokio::test(start_paused = true)]
    async fn serve_with_post_goaway_bound_waits_without_limit_until_the_paced_ending_starts() {
        let shutdown = GracefulShutdown::new();
        let bounded = serve_with_post_goaway_bound(
            std::future::pending::<()>(),
            &shutdown,
            RPC_DEADLINE,
            || 0,
        );
        assert!(
            tokio::time::timeout(Duration::from_secs(3600), bounded)
                .await
                .is_err(),
            "the bound counts from t0 (`begun`), not from the call"
        );
        assert!(!shutdown.is_cancelled());
    }

    /// **The bound.** A server that never returns — a connection that cannot
    /// finish closing — is given up on exactly B_post after t0: the later of
    /// the RPC deadline and the end of the paced ending, plus
    /// [`STREAM_CLOSE_GRACE`]. Then one WARN with the open-connection count
    /// and the numbers the bound came from, `None`, and the handle cancelled.
    /// Twice: the RPC deadline the later term (an empty registry, so the paced
    /// ending ends at t0), and the paced ending the later term (40 streams at
    /// 2/s, capped at 20 s, so its last batch is at t0 + 19.9 s against a 1 s
    /// deadline).
    #[tokio::test(start_paused = true)]
    async fn serve_with_post_goaway_bound_gives_up_b_post_after_t0_with_one_warn() {
        let deadline_later = GracefulShutdown::new();
        let window_later = GracefulShutdown::new();
        let mut enrollments = Vec::new();
        for _ in 0..40 {
            enrollments.push(window_later.enroll_stream());
        }
        for (what, shutdown, pacing, rpc_deadline, expected) in [
            (
                "the RPC deadline",
                &deadline_later,
                StreamPacing::DEFAULT,
                RPC_DEADLINE,
                RPC_DEADLINE + STREAM_CLOSE_GRACE,
            ),
            (
                "the paced ending",
                &window_later,
                StreamPacing {
                    rate_per_sec: 2,
                    max_window: Duration::from_secs(20),
                },
                Duration::from_secs(1),
                Duration::from_millis(19_900) + STREAM_CLOSE_GRACE,
            ),
        ] {
            let bounded = capture_json_events(serve_with_post_goaway_bound(
                std::future::pending::<()>(),
                shutdown,
                rpc_deadline,
                || 3,
            ));
            let ((served, events), t0) = tokio::join!(bounded, async {
                begin_now(shutdown, pacing).await;
                tokio::time::sleep(STREAM_END_DELAY_AFTER_GOAWAY).await;
                tokio::time::Instant::now()
            });
            assert_eq!(served, None, "{what}: the bound fired");
            assert_eq!(
                t0.elapsed(),
                expected,
                "{what}: the bound fires exactly B_post after t0"
            );
            let warns: Vec<_> = events
                .iter()
                .filter(|e| e["fields"]["message"] == BOUND_WARN_MSG)
                .collect();
            assert_eq!(warns.len(), 1, "{what}: one WARN; got {events:#?}");
            let warn = warns[0];
            assert_eq!(warn["level"], "WARN", "{warn}");
            assert_eq!(
                warn["fields"]["open_connections"], 3,
                "{what}: the WARN counts the connections still open; {warn}"
            );
            assert_eq!(
                warn["fields"]["waited_ms"],
                u64::try_from(expected.as_millis()).expect("fits"),
                "{warn}"
            );
            assert_eq!(
                warn["fields"]["rpc_deadline_ms"],
                u64::try_from(rpc_deadline.as_millis()).expect("fits"),
                "{warn}"
            );
            assert_eq!(warn["fields"]["close_grace_ms"], 5_000, "{warn}");
            assert!(shutdown.is_cancelled(), "{what}: the handle is cancelled");
        }
    }

    /// B_post = max(RPC deadline, window) + 5 s, and the minimum termination
    /// grace adds the drain, the GOAWAY head start, the post-server budget and
    /// the final flush to the bound at the longest window the pacing allows:
    /// 5 + 0.25 + (max(30, 20) + 5) + 30 + 6 = 76.25 s at the defaults, the
    /// figure the startup line reports and the chart's 80 s default covers.
    /// A window longer than the deadline takes its place.
    ///
    /// Only the arithmetic is pinned; the startup INFO line that reports it
    /// (`min_termination_grace_secs`, `main.rs`) is not. That line is logged
    /// after the store connect, which the binary-level startup harness
    /// (`tests/misc/startup_log_binary.rs`, which stops the child at the gRPC
    /// bind, before any dependency) cannot reach, and `main.rs` has no seam
    /// that runs that step without a reachable store and auth backends.
    #[test]
    fn minimum_termination_grace_is_76_25_s_at_the_defaults() {
        assert_eq!(STREAM_CLOSE_GRACE, Duration::from_secs(5));
        assert_eq!(POST_BUDGET_FLUSH_BOUND, Duration::from_secs(6));
        assert_eq!(
            post_goaway_bound(RPC_DEADLINE, Duration::from_secs(20)),
            Duration::from_secs(35)
        );
        assert_eq!(
            post_goaway_bound(Duration::from_secs(10), Duration::from_secs(20)),
            Duration::from_secs(25)
        );
        assert_eq!(
            minimum_termination_grace(
                Duration::from_secs(5),
                RPC_DEADLINE,
                StreamPacing::DEFAULT,
                Duration::from_secs(30),
            ),
            Duration::from_millis(76_250)
        );
        assert_eq!(
            minimum_termination_grace(
                Duration::from_secs(5),
                Duration::from_secs(10),
                StreamPacing {
                    rate_per_sec: 25,
                    max_window: Duration::from_secs(40),
                },
                Duration::from_secs(30),
            ),
            Duration::from_millis(5_000 + 250 + 45_000 + 30_000 + 6_000),
            "a window above the RPC deadline sets the bound"
        );
        assert_eq!(
            minimum_termination_grace(
                Duration::from_secs(5),
                RPC_DEADLINE,
                StreamPacing::ALL_AT_ONCE,
                Duration::from_secs(30),
            ),
            Duration::from_millis(76_250),
            "with pacing off the RPC deadline alone sets the bound"
        );
    }

    // ── HTTP/2 receive windows ──────────────────────────────────────────────

    /// h2's largest window, 2³¹ − 1, restated so the tests below do not lean
    /// on the private constant they check.
    const RFC_MAX_WINDOW: u64 = (1 << 31) - 1;

    /// Each stream's window is 1 MiB, the value hyper's server default had,
    /// and the pod memory plan's per-waiter term is that same window.
    #[test]
    fn http2_stream_window_is_1_mib_and_is_the_memory_plans_waiter_term() {
        assert_eq!(HTTP2_STREAM_WINDOW_BYTES, 1_048_576);
        assert_eq!(
            crate::config::memory_plan::UPLOAD_WAITER_WINDOW_BYTES,
            u64::from(HTTP2_STREAM_WINDOW_BYTES),
            "a queued waiter's unread content is bounded by one stream window"
        );
    }

    /// The connection window holds one stream window per upload that can wait
    /// for admission, plus one: (n + 1) × 1 MiB, 65 MiB at the default 64.
    #[test]
    fn http2_connection_window_is_one_stream_window_per_waiter_plus_one() {
        use crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;

        for waiters in [1_usize, 3, 64, 1_000, UPLOAD_ADMISSION_MAX_WAITERS_CEILING] {
            assert_eq!(
                u64::from(http2_connection_window_bytes(waiters)),
                (waiters as u64 + 1) * u64::from(HTTP2_STREAM_WINDOW_BYTES),
                "the connection window at {waiters} waiters"
            );
        }
        assert_eq!(
            http2_connection_window_bytes(UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT),
            68_157_440
        );
    }

    /// The waiter ceiling is the largest cap whose connection window fits
    /// h2's largest window: 2,046 does, one more would not.
    #[test]
    fn upload_admission_waiter_ceiling_is_the_largest_cap_whose_window_h2_accepts() {
        assert_eq!(UPLOAD_ADMISSION_MAX_WAITERS_CEILING, 2_046);
        assert!(
            u64::from(http2_connection_window_bytes(
                UPLOAD_ADMISSION_MAX_WAITERS_CEILING
            )) <= RFC_MAX_WINDOW
        );
        assert!(
            (UPLOAD_ADMISSION_MAX_WAITERS_CEILING as u64 + 2)
                * u64::from(HTTP2_STREAM_WINDOW_BYTES)
                > RFC_MAX_WINDOW,
            "one waiter more would pass h2's largest window"
        );
    }

    /// A waiter cap above the ceiling has no window h2 accepts: the function
    /// panics rather than wrap or hand h2 a window it would assert on.
    #[test]
    #[should_panic(expected = "UPLOAD_ADMISSION_MAX_WAITERS_CEILING")]
    fn http2_connection_window_above_the_waiter_ceiling_panics() {
        let _ = http2_connection_window_bytes(UPLOAD_ADMISSION_MAX_WAITERS_CEILING + 1);
    }

    /// h2 0.4.19's small-frame budget is half the connection window, floored
    /// at 25,600 bytes: 32.5 MiB at the default waiter cap, 512 KiB at the
    /// 1 MiB connection window the service ran with before the window
    /// followed the waiter cap, 32,767 at the 65,535-byte window
    /// `http2_adaptive_window` would leave.
    #[test]
    fn h2_small_frame_budget_is_half_the_connection_window_floored_at_25_600() {
        use crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;

        assert_eq!(
            h2_small_frame_budget_bytes(http2_connection_window_bytes(
                UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT
            )),
            34_078_720
        );
        assert_eq!(
            h2_small_frame_budget_bytes(HTTP2_STREAM_WINDOW_BYTES),
            524_288
        );
        assert_eq!(h2_small_frame_budget_bytes(65_535), 32_767);
        assert_eq!(h2_small_frame_budget_bytes(51_200), 25_600);
        assert_eq!(
            h2_small_frame_budget_bytes(40_000),
            25_600,
            "below twice the floor the floor holds"
        );
    }

    // ── run_post_server_sequence: the shared pre-quit() deadline ─────────
    //
    // Every wait the sequence makes is a virtual-clock timeout, so on paused
    // time each run below lands its steps at exact instants: a stuck wait ends
    // exactly at its bound, and the bounds are read back off the clock.

    /// A stand-in for `ValkeyClient::quit()` at its worst: 11.5 s, under the
    /// 15 s headroom its sequential bounds must fit (the `const` assertion in
    /// `storage::client`), so a `quit()` that starts with the headroom left
    /// ends inside the overall budget.
    const QUIT_WORST_CASE: Duration = Duration::from_millis(11_500);

    /// The cut WARN's message.
    const CUT_WARN: &str = "shutdown wait cut short to keep Valkey quit()'s headroom";

    /// The overall budget's WARN, which drops the sequence where it stands.
    const OVERALL_BUDGET_WARN: &str =
        "shutdown overall budget exceeded; exiting anyway to fit terminationGracePeriodSeconds";

    /// The chart's background tasks with an adapter wired, one entry per
    /// join, as `main.rs` lists them: the eight always present, the
    /// reconciler (R = 1), the backup repair and the checkpoint clean-up
    /// (E = C = 1), the checkpoint loop, and two mirror workers (M = 2).
    const CHART_TASKS: [&str; 14] = [
        "gc",
        "orphan_sweep",
        "presence_subscriber",
        "presence_sweeper",
        "health_supervisor",
        "session_delete_subscriber",
        "session_base_path_cache_sweeper",
        "session_tokens_sweeper",
        "ext_ref_reconcile",
        "ext_backup_repair",
        "checkpoint_cleanup",
        "checkpoint_loop",
        "ext_mirror_worker",
        "ext_mirror_worker",
    ];

    /// Which of the sequence's waits are stuck in one run: a stuck wait never
    /// finishes on its own, so only its bound ends it.
    #[derive(Debug, Clone, Copy)]
    struct Shape {
        /// O, `LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS`.
        overall: Duration,
        /// T, `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`.
        task: Duration,
        /// W1: the layer store's tracked cleanups.
        layer_stuck: bool,
        /// W1: one presence cleanup and one subscription task.
        streams_stuck: bool,
        /// W2: one session-checkpoint write.
        session_stuck: bool,
        /// W3: the first this many of [`CHART_TASKS`] (cycled) never stop;
        /// none others are joined.
        stuck_tasks: usize,
    }

    impl Shape {
        /// The defaults (T = 5 s, O = 30 s = 3T + 15) with every wait stuck,
        /// for the chart's fourteen background tasks.
        fn chart_all_stuck() -> Self {
            Self {
                overall: Duration::from_secs(30),
                task: Duration::from_secs(5),
                layer_stuck: true,
                streams_stuck: true,
                session_stuck: true,
                stuck_tasks: CHART_TASKS.len(),
            }
        }
    }

    /// When each step of one run happened, on the paused clock, measured from
    /// the sequence's start, and every event it logged.
    #[derive(Debug)]
    struct SequenceRun {
        /// The bound W1's layer-cleanup wait was handed.
        w1_bound: Duration,
        /// When W1 ended: the later of its layer-cleanup wait and its stream
        /// drains.
        w1_end: Duration,
        /// When `session_checkpoint_abort` was cancelled, which is when W2
        /// ended; `None` if it never was.
        w2_end: Option<Duration>,
        /// When the background tasks' tokens fired, which is when W3 began.
        cancels_fired: Option<Duration>,
        /// When `quit()` started, which is when W3 ended.
        quit_started: Option<Duration>,
        /// When the sequence returned.
        returned: Duration,
        events: Vec<serde_json::Value>,
    }

    impl SequenceRun {
        /// W2's bound, read off a stuck W2: it ran from W1's end to its own.
        fn w2_bound(&self) -> Duration {
            self.w2_end.expect("W2 ended") - self.w1_end
        }

        /// W3's bound, read off a stuck W3: it ran from the token fire to
        /// `quit()`.
        fn w3_bound(&self) -> Duration {
            self.quit_started.expect("quit() started") - self.cancels_fired.expect("tokens fired")
        }

        fn with_message(&self, message: &str) -> Vec<&serde_json::Value> {
            self.events
                .iter()
                .filter(|e| e["fields"]["message"] == message)
                .collect()
        }

        /// The cut WARNs, by the wait each names.
        fn cut_warns(&self) -> Vec<&serde_json::Value> {
            self.with_message(CUT_WARN)
        }

        fn cut_warn_for(&self, wait: &str) -> Option<&serde_json::Value> {
            self.cut_warns()
                .into_iter()
                .find(|e| e["fields"]["wait"] == wait)
        }
    }

    /// Run [`run_post_server_sequence`] on paused time over `shape`, with a
    /// `quit()` that takes [`QUIT_WORST_CASE`].
    async fn run_sequence(shape: Shape) -> SequenceRun {
        use std::sync::Mutex;

        let start = tokio::time::Instant::now();
        let since_start = move || start.elapsed();

        let w1 = Arc::new(Mutex::new(None::<(Duration, Duration)>));
        let layer_cleanup = {
            let w1 = Arc::clone(&w1);
            let stuck = shape.layer_stuck;
            move |bound: Duration| async move {
                if stuck {
                    // `LayerStore::wait_for_cleanup` with a cleanup that never
                    // finishes: it waits out its whole bound.
                    tokio::time::sleep(bound).await;
                }
                *w1.lock().unwrap() = Some((bound, since_start()));
                usize::from(stuck)
            }
        };
        let presence_cleanup = TaskTracker::new();
        let subscriptions = TaskTracker::new();
        if shape.streams_stuck {
            presence_cleanup.spawn(std::future::pending::<()>());
            subscriptions.spawn(std::future::pending::<()>());
        }
        let session_checkpoints = TaskTracker::new();
        if shape.session_stuck {
            session_checkpoints.spawn(std::future::pending::<()>());
        }

        let session_checkpoint_abort = CancellationToken::new();
        let w2_end = tokio::spawn({
            let abort = session_checkpoint_abort.clone();
            async move {
                abort.cancelled().await;
                since_start()
            }
        });
        let cancels: Vec<CancellationToken> = (0..3).map(|_| CancellationToken::new()).collect();
        let cancels_fired = tokio::spawn({
            let first = cancels[0].clone();
            async move {
                first.cancelled().await;
                since_start()
            }
        });
        let tasks: Vec<(&'static str, JoinHandle<()>)> = (0..shape.stuck_tasks)
            .map(|i| {
                (
                    CHART_TASKS[i % CHART_TASKS.len()],
                    tokio::spawn(std::future::pending::<()>()),
                )
            })
            .collect();
        let report = BackgroundTaskJoinReport {
            metrics: Arc::new(crate::observability::metrics::metrics_scrape_only(
                "transport-shutdown-test",
            )),
            hold_writers: Arc::new(HoldWriterAborts::new()),
        };
        let quit_started = Arc::new(Mutex::new(None::<Duration>));
        let quit = {
            let quit_started = Arc::clone(&quit_started);
            async move {
                *quit_started.lock().unwrap() = Some(since_start());
                tokio::time::sleep(QUIT_WORST_CASE).await;
            }
        };

        let ((), events) = capture_json_events(run_post_server_sequence(
            shape.overall,
            shape.task,
            layer_cleanup,
            &presence_cleanup,
            &subscriptions,
            &session_checkpoints,
            &session_checkpoint_abort,
            &cancels,
            tasks,
            &report,
            quit,
        ))
        .await;
        let returned = since_start();

        // A watcher woken by its token ran at the token's instant: nothing
        // advances a paused clock while a task is ready.
        let w2_end = if session_checkpoint_abort.is_cancelled() {
            Some(w2_end.await.expect("the W2 watcher does not panic"))
        } else {
            w2_end.abort();
            None
        };
        let cancels_fired = if cancels[0].is_cancelled() {
            Some(cancels_fired.await.expect("the W3 watcher does not panic"))
        } else {
            cancels_fired.abort();
            None
        };
        let (w1_bound, layer_end) = w1.lock().unwrap().expect("W1 ran");
        // W1 ends when the later of its two halves does: the layer cleanups
        // (recorded) and the stream drains, which, stuck, run to the same
        // bound from the same start.
        let w1_end = if shape.streams_stuck {
            layer_end.max(w1_bound)
        } else {
            layer_end
        };
        let quit_started = *quit_started.lock().unwrap();
        SequenceRun {
            w1_bound,
            w1_end,
            w2_end,
            cancels_fired,
            quit_started,
            returned,
            events,
        }
    }

    /// **Regression: the chart's shape with every wait stuck.** At the
    /// defaults (T = 5 s, O = 30 s) with the chart's fourteen background
    /// tasks, every pre-`quit()` wait stuck — the layer cleanups, both stream
    /// drains, the session-checkpoint drain and every task — `quit()` still
    /// starts by O − 15 s, so it runs to its end inside the overall budget and
    /// the overall budget's WARN does not fire. On the sequential shape this
    /// replaced (each wait and each join after the first eight a full T in
    /// turn, 50 s before `quit()`), the budget fired before `quit()` started.
    /// O sits exactly at 3T + 15, so no wait is cut: each gets its full T and
    /// no cut WARN is logged.
    #[tokio::test(start_paused = true)]
    async fn the_chart_shape_with_every_wait_stuck_starts_quit_with_its_headroom_left() {
        let shape = Shape::chart_all_stuck();
        let run = run_sequence(shape).await;

        let headroom = Duration::from_secs(crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS);
        let quit_started = run
            .quit_started
            .expect("quit() starts: the overall budget dropped the sequence before it");
        assert!(
            quit_started <= shape.overall - headroom,
            "quit() started {quit_started:?} in, past O − 15 s = {:?}",
            shape.overall - headroom
        );
        assert!(
            run.with_message(OVERALL_BUDGET_WARN).is_empty(),
            "the overall budget never fires: {:#?}",
            run.events
        );
        assert_eq!(
            run.with_message("shutdown complete").len(),
            1,
            "the sequence ran to its end"
        );
        assert_eq!(
            run.returned,
            quit_started + QUIT_WORST_CASE,
            "quit() ran to its end"
        );
        assert_eq!(
            (run.w1_bound, run.w2_bound(), run.w3_bound()),
            (shape.task, shape.task, shape.task),
            "at O = 3T + 15 every wait gets its full per-task budget"
        );
        assert!(
            run.cut_warns().is_empty(),
            "nothing was cut: {:#?}",
            run.cut_warns()
        );
        assert_eq!(
            run.with_message("background task did not stop within per-task shutdown budget")
                .len(),
            CHART_TASKS.len(),
            "every stuck task is reported once"
        );
    }

    /// **At the floor** (T = 5 s, O = T + 15 = 20 s) with every wait stuck,
    /// the shared deadline is 5 s in: W1 gets its full T, which spends it, so
    /// `quit()` starts at 5 s, W2 and the join get a zero bound, and each of
    /// those two logs the cut WARN, naming the wait, what was still pending
    /// and the bound — for the join, the count and the stuck tasks' names.
    #[tokio::test(start_paused = true)]
    async fn at_the_floor_quit_starts_after_one_per_task_budget_and_the_cut_waits_are_named() {
        let tasks = &CHART_TASKS[..2];
        let shape = Shape {
            overall: Duration::from_secs(20),
            stuck_tasks: tasks.len(),
            ..Shape::chart_all_stuck()
        };
        let run = run_sequence(shape).await;

        assert_eq!(run.quit_started, Some(shape.task));
        assert!(run.with_message(OVERALL_BUDGET_WARN).is_empty());
        assert_eq!(
            run.w1_bound, shape.task,
            "W1 runs before the deadline bites"
        );
        assert!(
            run.cut_warn_for("layer_cleanup").is_none()
                && run.cut_warn_for("presence_cleanup").is_none()
                && run.cut_warn_for("subscriptions").is_none(),
            "W1 was not cut: {:#?}",
            run.cut_warns()
        );

        let session = run
            .cut_warn_for("session_checkpoints")
            .unwrap_or_else(|| panic!("the session drain was cut: {:#?}", run.events));
        assert_eq!(session["level"], "WARN");
        assert_eq!(session["fields"]["pending"], 1, "{session}");
        assert_eq!(session["fields"]["bound_ms"], 0, "{session}");

        let join = run
            .cut_warn_for("background_tasks")
            .unwrap_or_else(|| panic!("the join was cut: {:#?}", run.events));
        assert_eq!(join["level"], "WARN");
        assert_eq!(join["fields"]["pending"], tasks.len(), "{join}");
        assert_eq!(join["fields"]["bound_ms"], 0, "{join}");
        let names = join["fields"]["pending_tasks"]
            .as_str()
            .unwrap_or_else(|| panic!("the join's WARN names the tasks: {join}"));
        for task in tasks {
            assert!(names.contains(task), "{task} is named: {names}");
        }
        assert_eq!(run.cut_warns().len(), 2, "{:#?}", run.cut_warns());
    }

    /// Run [`run_post_server_sequence`] at the floor (T = 5 s, O = T + 15 =
    /// 20 s) with only W1's layer-cleanup wait stuck: it spends the whole
    /// shared deadline, so W2 and the join run with a zero bound. The session
    /// tracker is empty, the stream trackers are empty, the join is over
    /// `tasks` and the token fire over `cancels`. Returns every event logged
    /// and how long the sequence took.
    async fn run_at_the_floor_with_w1_stuck(
        tasks: Vec<(&'static str, JoinHandle<()>)>,
        cancels: &[CancellationToken],
    ) -> (Vec<serde_json::Value>, Duration) {
        let task_budget = Duration::from_secs(5);
        let report = BackgroundTaskJoinReport {
            metrics: Arc::new(crate::observability::metrics::metrics_scrape_only(
                "transport-floor-test",
            )),
            hold_writers: Arc::new(HoldWriterAborts::new()),
        };
        let empty = TaskTracker::new();
        let session_checkpoints = TaskTracker::new();
        let started = tokio::time::Instant::now();
        let ((), events) = capture_json_events(run_post_server_sequence(
            task_budget + Duration::from_secs(crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS),
            task_budget,
            |bound: Duration| async move {
                tokio::time::sleep(bound).await;
                1
            },
            &empty,
            &empty,
            &session_checkpoints,
            &CancellationToken::new(),
            cancels,
            tasks,
            &report,
            async {},
        ))
        .await;
        (events, started.elapsed())
    }

    /// The `message` of the one event that names `task`.
    fn join_line_for<'a>(events: &'a [serde_json::Value], task: &str) -> &'a serde_json::Value {
        let lines: Vec<&serde_json::Value> = events
            .iter()
            .filter(|e| e["fields"]["task"] == task)
            .collect();
        assert_eq!(lines.len(), 1, "one line for {task}: {events:#?}");
        &lines[0]["fields"]["message"]
    }

    /// **A zero bound still polls once.** At the floor with W1 stuck, the
    /// session drain and the join get a zero bound, yet a wait already done
    /// passes: `tokio::time::timeout` polls its future before its timer. The
    /// empty session tracker is reported drained, with no timeout WARN, and a
    /// joined task that had already finished is reported stopped cleanly; no
    /// cut WARN is logged, since nothing timed out.
    #[tokio::test(start_paused = true)]
    async fn at_the_floor_a_zero_bound_still_passes_a_drained_tracker_and_a_finished_task() {
        let finished = tokio::spawn(async {});
        while !finished.is_finished() {
            tokio::task::yield_now().await;
        }

        let (events, took) = run_at_the_floor_with_w1_stuck(vec![("gc", finished)], &[]).await;

        assert_eq!(
            took,
            Duration::from_secs(5),
            "fixture: W1 spent the whole shared deadline, so W2 and the join had a zero bound"
        );
        let with_message = |msg: &str| {
            events
                .iter()
                .filter(|e| e["fields"]["message"] == msg)
                .count()
        };
        assert_eq!(
            with_message("session-checkpoint writes drained"),
            1,
            "{events:#?}"
        );
        assert_eq!(
            with_message(
                "timed out waiting for session-checkpoint writes; some lifecycle events may not \
                 be reflected in the checkpoint store"
            ),
            0,
            "{events:#?}"
        );
        assert_eq!(with_message(CUT_WARN), 0, "{events:#?}");
        assert_eq!(
            join_line_for(&events, "gc"),
            "background task stopped cleanly"
        );
    }

    /// **A zero join bound waits for nothing, and its line says so.** At the
    /// floor with W1 stuck the join's bound is zero, so each task is polled
    /// once and a task still running then is reported at WARN as not waited
    /// for, never as one that did not stop within the per-task budget: no
    /// budget was spent waiting for it. That holds for a task that never
    /// stops and for one that ends as soon as its token fires, since the
    /// join polls each task in the same poll that fired the tokens, before a
    /// woken task has run, and a zero `timeout` fires on that first poll: the
    /// paused clock sits on the 5 s tick the wheel has already processed, and
    /// tokio rounds the deadline up to that same tick
    /// (`runtime/time/source.rs`, `deadline_to_tick`), so registering it finds
    /// it already elapsed (`runtime/time/wheel/mod.rs`, `insert`) and fires
    /// it. A tokio whose zero timer returned Pending there would let `gc` run
    /// and see it reported "stopped cleanly". The join's cut WARN still names
    /// both, with a zero bound.
    #[tokio::test(start_paused = true)]
    async fn at_the_floor_a_task_still_running_is_reported_not_waited_for() {
        const NOT_WAITED_FOR: &str =
            "background task not waited for: the shared shutdown deadline was already spent";
        const OVER_BUDGET: &str = "background task did not stop within per-task shutdown budget";

        let cancel = CancellationToken::new();
        let obeys = tokio::spawn({
            let cancel = cancel.clone();
            async move { cancel.cancelled().await }
        });
        // Its first poll parks it on the token.
        tokio::task::yield_now().await;
        assert!(
            !obeys.is_finished(),
            "fixture: the task waits for its cancel"
        );
        let stuck = tokio::spawn(std::future::pending::<()>());

        let (events, took) = run_at_the_floor_with_w1_stuck(
            vec![("gc", obeys), ("orphan_sweep", stuck)],
            std::slice::from_ref(&cancel),
        )
        .await;

        assert_eq!(took, Duration::from_secs(5), "fixture: a zero join bound");
        for task in ["gc", "orphan_sweep"] {
            let lines: Vec<&serde_json::Value> = events
                .iter()
                .filter(|e| e["fields"]["task"] == task)
                .collect();
            assert_eq!(lines.len(), 1, "one line for {task}: {events:#?}");
            assert_eq!(
                (&lines[0]["level"], &lines[0]["fields"]["message"]),
                (
                    &serde_json::json!("WARN"),
                    &serde_json::json!(NOT_WAITED_FOR)
                ),
                "{task}: {}",
                lines[0]
            );
        }
        assert!(
            !events.iter().any(|e| e["fields"]["message"] == OVER_BUDGET),
            "no task is reported as having outlived a budget nobody waited out: {events:#?}"
        );
        let cut: Vec<&serde_json::Value> = events
            .iter()
            .filter(|e| e["fields"]["message"] == CUT_WARN)
            .collect();
        assert_eq!(cut.len(), 1, "only the join was cut: {events:#?}");
        let join = cut[0];
        assert_eq!(join["level"], "WARN", "{join}");
        assert_eq!(join["fields"]["wait"], "background_tasks", "{join}");
        assert_eq!(join["fields"]["bound_ms"], 0, "{join}");
        assert_eq!(join["fields"]["pending"], 2, "{join}");
        let names = join["fields"]["pending_tasks"]
            .as_str()
            .unwrap_or_else(|| panic!("the cut WARN names the tasks: {join}"));
        for task in ["gc", "orphan_sweep"] {
            assert!(names.contains(task), "{task} is named: {names}");
        }
    }

    /// **The session drain is never squeezed before the join.** Over a grid
    /// of per-task budgets T, overall budgets O from the floor (T + 15) to
    /// above 3T + 15, and W1 stuck or not (W2 and W3 always stuck, so each
    /// runs to its bound and the bound can be read off the clock), the join's
    /// bound is never above the session drain's, and `quit()` always starts
    /// with the 15 s headroom left.
    #[tokio::test(start_paused = true)]
    async fn the_joins_bound_is_never_above_the_session_drains_and_quit_keeps_its_headroom() {
        let headroom = Duration::from_secs(crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS);
        for task_secs in [1u64, 5, 8, 20] {
            let task = Duration::from_secs(task_secs);
            for overall in [
                task + headroom,
                task + task / 2 + headroom,
                2 * task + headroom,
                3 * task + headroom - Duration::from_millis(1),
                3 * task + headroom,
                4 * task + headroom,
            ] {
                for (layer_stuck, streams_stuck) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let shape = Shape {
                        overall,
                        task,
                        layer_stuck,
                        streams_stuck,
                        session_stuck: true,
                        stuck_tasks: 3,
                    };
                    let run = run_sequence(shape).await;
                    let quit_started = run
                        .quit_started
                        .unwrap_or_else(|| panic!("{shape:?}: quit() never started"));
                    assert!(
                        quit_started <= overall - headroom,
                        "{shape:?}: quit() started {quit_started:?} in, past O − 15 s"
                    );
                    assert!(
                        run.w3_bound() <= run.w2_bound(),
                        "{shape:?}: the join's bound {:?} is above the session drain's {:?}",
                        run.w3_bound(),
                        run.w2_bound()
                    );
                    assert!(
                        run.w1_bound <= task && run.w2_bound() <= task && run.w3_bound() <= task,
                        "{shape:?}: no wait runs past T: {run:?}"
                    );
                }
            }
        }
    }

    /// A wait cut by the per-task budget alone logs only its own timeout
    /// line, never the cut WARN: with O far above 3T + 15 every stuck wait
    /// times out at T. And a shortfall within [`SHUTDOWN_CUT_WARN_TOLERANCE`]
    /// is not a cut either: half the tolerance below 3T + 15 leaves the join
    /// that much under T and logs nothing, while one and a half times the
    /// tolerance below logs the join's cut WARN with its bound.
    #[tokio::test(start_paused = true)]
    async fn a_wait_cut_by_the_per_task_budget_or_within_the_tolerance_logs_no_cut_warn() {
        let defaults = Shape::chart_all_stuck();
        let roomy = run_sequence(Shape {
            overall: Duration::from_secs(120),
            ..defaults
        })
        .await;
        assert!(
            roomy.cut_warns().is_empty(),
            "a wait cut by T alone: {:#?}",
            roomy.cut_warns()
        );
        assert_eq!(
            roomy
                .with_message(
                    "timed out waiting for session-checkpoint writes; some lifecycle \
                     events may not be reflected in the checkpoint store"
                )
                .len(),
            1,
            "the stuck waits still log their own timeout lines"
        );

        let at_defaults = defaults.overall;
        let within = run_sequence(Shape {
            overall: at_defaults - SHUTDOWN_CUT_WARN_TOLERANCE / 2,
            ..defaults
        })
        .await;
        assert_eq!(
            within.w3_bound(),
            defaults.task - SHUTDOWN_CUT_WARN_TOLERANCE / 2,
            "fixture: the join is cut by half the tolerance"
        );
        assert!(
            within.cut_warns().is_empty(),
            "a shortfall within the tolerance is not a cut: {:#?}",
            within.cut_warns()
        );

        let beyond = run_sequence(Shape {
            overall: at_defaults - SHUTDOWN_CUT_WARN_TOLERANCE * 3 / 2,
            ..defaults
        })
        .await;
        let cut = defaults.task - SHUTDOWN_CUT_WARN_TOLERANCE * 3 / 2;
        assert_eq!(beyond.w3_bound(), cut, "fixture");
        let warn = beyond
            .cut_warn_for("background_tasks")
            .unwrap_or_else(|| panic!("a cut past the tolerance is logged: {:#?}", beyond.events));
        assert_eq!(warn["fields"]["bound_ms"], cut.as_millis() as u64, "{warn}");
        assert_eq!(
            beyond.cut_warns().len(),
            1,
            "only the join was cut: {:#?}",
            beyond.cut_warns()
        );
    }

    /// N stuck background tasks cost one per-task budget together, not one
    /// each: they are all cancelled at one instant and joined concurrently,
    /// so with nothing else stuck the join ends exactly T after the tokens
    /// fire, and `quit()` starts then. Each stuck task is reported once.
    #[tokio::test(start_paused = true)]
    async fn n_stuck_background_tasks_end_in_one_per_task_budget() {
        const STUCK: usize = 16;
        let shape = Shape {
            overall: Duration::from_secs(120),
            task: Duration::from_secs(5),
            layer_stuck: false,
            streams_stuck: false,
            session_stuck: false,
            stuck_tasks: STUCK,
        };
        let run = run_sequence(shape).await;

        assert_eq!(
            run.cancels_fired,
            Some(Duration::ZERO),
            "nothing before the join waited"
        );
        assert_eq!(
            run.quit_started,
            Some(shape.task),
            "{STUCK} stuck tasks end in one T"
        );
        assert_eq!(
            run.with_message("background task did not stop within per-task shutdown budget")
                .len(),
            STUCK
        );
    }

    /// The join reports each task by how it ended: INFO for a task that
    /// stopped on its token;
    /// ERROR for one that panicked, counted on
    /// `live_edit_background_task_panics_total{task}`; INFO "aborted earlier"
    /// for one cancelled after a hold or stale-verdict entry aborted the
    /// writer inventory; and ERROR, uncounted, for one cancelled when no such
    /// entry did.
    #[tokio::test(start_paused = true)]
    async fn the_join_reports_each_task_by_how_it_ended_and_counts_a_panic() {
        use crate::observability::metrics::test_util::{render, sample_value};

        const FAILED: &str = "background task failed during shutdown";
        const ABORTED: &str = "background task was aborted earlier, at hold or stale-verdict entry";

        for hold_entered in [true, false] {
            let report = BackgroundTaskJoinReport {
                metrics: Arc::new(crate::observability::metrics::metrics_scrape_only(
                    "transport-join-report-test",
                )),
                hold_writers: Arc::new(HoldWriterAborts::new()),
            };
            let cancel = CancellationToken::new();
            let clean = tokio::spawn({
                let cancel = cancel.clone();
                async move { cancel.cancelled().await }
            });
            let panicked = tokio::spawn(async { panic!("deliberate panic for the join report") });
            let aborted = tokio::spawn(std::future::pending::<()>());
            aborted.abort();
            if hold_entered {
                report.hold_writers.abort_all();
            }
            let empty = TaskTracker::new();

            let ((), events) = capture_json_events(run_post_server_sequence(
                Duration::from_secs(60),
                Duration::from_secs(5),
                |_| async { 0 },
                &empty,
                &empty,
                &empty,
                &CancellationToken::new(),
                std::slice::from_ref(&cancel),
                vec![
                    ("gc", clean),
                    ("orphan_sweep", panicked),
                    ("checkpoint_cleanup", aborted),
                ],
                &report,
                async {},
            ))
            .await;

            let line_for = |task: &str| -> &serde_json::Value {
                let lines: Vec<&serde_json::Value> = events
                    .iter()
                    .filter(|e| e["fields"]["task"] == task)
                    .collect();
                assert_eq!(lines.len(), 1, "one line for {task}: {events:#?}");
                lines[0]
            };
            let gc = line_for("gc");
            assert_eq!(
                (&gc["level"], &gc["fields"]["message"]),
                (
                    &serde_json::json!("INFO"),
                    &serde_json::json!("background task stopped cleanly")
                ),
                "{gc}"
            );
            let panic_line = line_for("orphan_sweep");
            assert_eq!(
                (&panic_line["level"], &panic_line["fields"]["message"]),
                (&serde_json::json!("ERROR"), &serde_json::json!(FAILED)),
                "{panic_line}"
            );
            let cancelled = line_for("checkpoint_cleanup");
            let expected = if hold_entered {
                ("INFO", ABORTED)
            } else {
                ("ERROR", FAILED)
            };
            assert_eq!(
                (&cancelled["level"], &cancelled["fields"]["message"]),
                (
                    &serde_json::json!(expected.0),
                    &serde_json::json!(expected.1)
                ),
                "hold entered: {hold_entered}; {cancelled}"
            );

            let panics = |task: &str| {
                sample_value(
                    &render(&report.metrics),
                    "live_edit_background_task_panics_total",
                    &[("task", task)],
                )
                .unwrap_or(0.0)
            };
            assert_eq!(panics("orphan_sweep"), 1.0, "the panic is counted");
            assert_eq!(
                panics("checkpoint_cleanup"),
                0.0,
                "a cancellation is not a panic"
            );
        }
    }

    /// A stuck layer-cleanup wait and stuck stream drains cost one per-task
    /// budget together, not two: W1 runs them concurrently, so it ends exactly
    /// T in, and with nothing else stuck `quit()` starts then.
    #[tokio::test(start_paused = true)]
    async fn a_stuck_layer_drain_and_stuck_stream_drains_end_in_one_per_task_budget() {
        let shape = Shape {
            overall: Duration::from_secs(120),
            task: Duration::from_secs(5),
            layer_stuck: true,
            streams_stuck: true,
            session_stuck: false,
            stuck_tasks: 0,
        };
        let run = run_sequence(shape).await;

        assert_eq!(
            run.quit_started,
            Some(shape.task),
            "the layer drain and the stream drains beside it all end at T"
        );
        assert_eq!(
            run.with_message(
                "timed out waiting for SubscribeReadLayer subscription tasks; those still \
                 registered with the stream multiplexer are ended UNAVAILABLE by its shutdown \
                 before the pools quit"
            )
            .len(),
            1,
            "fixture: the subscription drain was stuck"
        );
    }
}
