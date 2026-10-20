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
//! [`live_service_server`] — the auth interceptor, the end-of-stream record
//! the uploads commit on, and the body-progress record the JoinSession client
//! keepalive reads (SDD §5.1, §5.4, §13.2) — in `main.rs` and in every test
//! server that serves it, for the same reason. The full-API server's graceful
//! shutdown signal is built by [`end_streams_after_drain`], the wait for the
//! server after GOAWAY is bounded by [`serve_with_post_goaway_bound`], and the
//! streaming RPCs' tasks are drained after the server returns by
//! [`drain_stream_tasks`]. The tests that pin shutdown behaviour call them
//! too, for the same reason.

use std::future::Future;
use std::time::Duration;

use tokio_util::task::TaskTracker;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;
use tonic::{Request, Status};

use super::graceful_shutdown::{end_streams_paced, GracefulShutdown, StreamPacing};
use super::proto::live_service_server::LiveServiceServer;
use super::request_end::AttestRequestEnd;
use super::service::LiveEditService;

/// The auth interceptor's type, as [`live_service_server`] names it.
pub type AuthInterceptorFn = fn(Request<()>) -> Result<Request<()>, Status>;

/// The LiveService as the service serves it: built by [`live_service_server`].
pub type LiveServiceStack =
    AttestRequestEnd<InterceptedService<LiveServiceServer<LiveEditService>, AuthInterceptorFn>>;

/// The LiveService as every server serves it: the auth interceptor
/// ([`auth_interceptor`](super::auth_interceptor::auth_interceptor)) inside,
/// and outermost [`AttestRequestEnd`](super::request_end::AttestRequestEnd),
/// which inserts two records: the end-of-stream record that `CreateLayer` and
/// `UpdateLayer` commit on (SDD §5.4), and the
/// [`RequestBodyProgress`](super::request_end::RequestBodyProgress) record the
/// JoinSession client keepalive reads (SDD §13.2).
///
/// The one constructor: `main.rs` and every in-repo test server that serves the
/// LiveService use it, so none can drift from the production stack. Through a
/// stack built any other way, an upload carries no end-of-stream record and is
/// refused `INTERNAL` (`grpc::streaming::UploadStream::from_request`), and a
/// JoinSession that opted in to the client keepalive carries no progress
/// record and is silently left unarmed: it is told `keepalive_interval_ms = 0`
/// and is never ended for silence.
pub fn live_service_server(svc: LiveEditService) -> LiveServiceStack {
    AttestRequestEnd::new(LiveServiceServer::with_interceptor(
        svc,
        super::auth_interceptor::auth_interceptor as AuthInterceptorFn,
    ))
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

/// A `tonic` server builder carrying the service's HTTP/2 settings: a keepalive
/// PING every 30 s, and the connection closed when one goes unacknowledged for
/// 5 s.
///
/// Callers add their layers and services and serve from a pre-bound listener.
/// TCP options (`TCP_NODELAY`, `TCP_KEEPALIVE`) are deliberately not set here:
/// `serve_with_incoming*` ignores the builder's TCP settings, so
/// `observability::startup::bind_grpc_listener` applies them to each accepted
/// connection instead.
///
/// The HTTP/2 flow-control windows are deliberately left at hyper's defaults of
/// 1 MiB per connection and 1 MiB per stream (`DEFAULT_CONN_WINDOW` /
/// `DEFAULT_STREAM_WINDOW` in hyper's `proto/h2/server.rs`, passed to the h2
/// handshake as its initial windows). h2 sizes its small-DATA-frame budget from
/// the connection window — half of it, from h2 0.4.19 — so every inbound
/// connection may hold 512 KiB of small-frame overhead unread before h2 closes
/// it with GOAWAY `ENHANCE_YOUR_CALM`. Only the connection window sets that
/// budget: changing it here changes the budget, and so does enabling
/// `http2_adaptive_window`, which makes hyper drop the connection window to
/// 65,535 and so the budget to 32,767 bytes. The stream window bounds how many
/// unread frames one stream can hold but leaves the budget alone. See SDD §5.1.
pub fn production_server_builder() -> Server {
    Server::builder()
        .http2_keepalive_interval(Some(HTTP2_KEEPALIVE_INTERVAL))
        .http2_keepalive_timeout(Some(HTTP2_KEEPALIVE_TIMEOUT))
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
/// post-server shutdown sequence, outside its overall budget.
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
/// (`main.rs` passes the per-task shutdown budget,
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`), so one stuck task cannot use up the
/// overall shutdown budget. Each outcome is logged: INFO when the tracker
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

#[cfg(test)]
mod tests {
    //! The shutdown sequencing contract of [`end_streams_after_drain`] — the
    //! drain, the GOAWAY head start and the paced ending's schedule — the
    //! bound [`serve_with_post_goaway_bound`] puts on the server after GOAWAY,
    //! the grace arithmetic, and what [`drain_stream_tasks`] tells the
    //! operator when its budget runs out, on paused time: every wait is a
    //! virtual-clock deadline, so the ordering is asserted exactly and nothing
    //! here depends on how fast the machine is.
    //! The enrollment registry and the end order are pinned in
    //! `graceful_shutdown.rs`'s tests; the real-server end-to-end checks
    //! (GOAWAY, then the paced `UNAVAILABLE` endings on live streams, the
    //! closed listener, the bound on a stalled connection) are
    //! `tests/integration/grpc/shutdown_ending_test.rs`.

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
}
