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

//! Early gRPC listener bind for the bootstrap health server (b.rgs), and the
//! process's shutdown signal and startup claim (below).
//!
//! The liveness probe targets the gRPC health service's overall (`""`)
//! status, so a closed `:50001` socket is indistinguishable from a wedged
//! process: the kubelet reports `failed to connect service ':50001'` and
//! restarts the pod. Before this module the socket did not exist until every
//! dependency (Valkey, checkpoint store, counter-seed recovery, auth clients)
//! had been wired, so *any* slow-but-progressing startup burned the liveness
//! window.
//!
//! [`bind_grpc_listener`] moves the bind to the very top of startup and hands
//! back **two** `TcpIncoming` streams over one dup'ed file descriptor:
//!
//! 1. the bootstrap server (health + reflection + capabilities) serves the
//!    first one while dependencies are wired, with readiness `NOT_SERVING`;
//! 2. the full-API server serves the second one after the handoff.
//!
//! Because `try_clone` dups the descriptor rather than opening a new socket,
//! both streams refer to the *same* open file description. The socket
//! therefore never leaves `LISTEN` across the handoff — connections queue in
//! the shared accept backlog instead of being refused, so there is no unbind /
//! rebind window and no `ECONNREFUSED` race with the kubelet.
//!
//! `Server::serve_with_incoming*` documents that the builder's `tcp_nodelay`
//! and `tcp_keepalive` settings are ignored, so both settings are re-applied
//! per-connection here via [`TcpIncoming::with_nodelay`] /
//! [`TcpIncoming::with_keepalive`], byte-identical to what tonic's own
//! `bind_incoming` does for the bind-based `serve_with_shutdown` path.
//!
//! The full-API server serves its stream through [`closable_incoming`], so
//! graceful shutdown can close the socket when the readiness drain ends rather
//! than when the server returns, and count the connections still open.
//!
//! # Shutdown signals from the first statement (SDD §5.5, §5.6)
//!
//! The binary runs as PID 1 in its container, and the kernel drops a signal
//! sent to PID 1 that has no handler, so a handler installed only when the
//! server starts serving would lose every `SIGTERM` that arrives while the
//! replica boots. [`ShutdownSignal::install`] registers the `SIGTERM` and
//! `SIGINT` handlers at once; `main.rs` calls it as the first statement of
//! every mode but the scaler (which keeps its own, equally eager, listener),
//! and it is the process's only listener.
//!
//! [`StartupClaim`] decides who acts on the first signal. Its watcher
//! ([`StartupClaim::watch`]) turns the signal into the process shutdown
//! request and, while the replica is still booting, runs the startup stop it
//! is handed; once the serving path has claimed the startup parts
//! ([`StartupClaim::claim_serving`]) the watcher only cancels the request,
//! which the serving path drains on. The two outcomes are exclusive: the
//! startup parts go to exactly one side.

use std::future::Future;
use std::io;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio::signal::unix::{signal, Signal, SignalKind};
use tokio_util::sync::CancellationToken;
use tonic::transport::server::{Connected, TcpConnectInfo, TcpIncoming};

/// What [`ShutdownSignal::recv`] resolves to on a `SIGTERM`.
pub const SIGTERM: &str = "SIGTERM";

/// What [`ShutdownSignal::recv`] resolves to on a `SIGINT`.
pub const SIGINT: &str = "SIGINT";

/// The process's one `SIGTERM` and `SIGINT` listener.
///
/// Both handlers are registered by [`Self::install`], when it is called, not
/// when [`Self::recv`] is first polled: a tokio signal listener sees every
/// signal delivered after its creation, and its OS handler is never
/// uninstalled, so no signal is lost from the install on, however long the
/// process runs before anything waits on it.
pub struct ShutdownSignal {
    terminate: Signal,
    interrupt: Signal,
}

impl ShutdownSignal {
    /// Register the `SIGTERM` and `SIGINT` handlers now.
    ///
    /// Must be called from inside a tokio runtime context. An error is the
    /// OS refusing a handler; `main.rs` reports it on stderr and exits 1.
    pub fn install() -> io::Result<Self> {
        Ok(Self {
            terminate: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }

    /// The next `SIGTERM` or `SIGINT`: [`SIGTERM`] or [`SIGINT`]. A signal
    /// delivered before this is first polled resolves it at once.
    pub async fn recv(&mut self) -> &'static str {
        tokio::select! {
            _ = self.terminate.recv() => SIGTERM,
            _ = self.interrupt.recv() => SIGINT,
        }
    }
}

/// Where startup stands, for [`StartupClaim`].
enum StartupPhase<P> {
    /// Booting: the startup parts are unclaimed.
    Booting(P),
    /// The serving path claimed the parts ([`StartupClaim::claim_serving`]).
    Serving,
    /// A signal arrived first, and the watcher took the parts to stop startup.
    StoppingStartup,
}

struct ClaimState<P> {
    phase: Mutex<StartupPhase<P>>,
    /// The process shutdown request: cancelled by the watcher on the first
    /// signal, after it records the cause, and by nothing else.
    request: CancellationToken,
    /// The first signal's name, set once, before `request` is cancelled.
    cause: OnceLock<&'static str>,
}

/// Which side acts on the first shutdown signal: the startup stop while the
/// replica boots, or the serving path once it has claimed the startup parts
/// `P` (in `main.rs`, the bootstrap server, its shutdown token and the
/// `/metrics` server).
///
/// Whichever side wins takes `P`; the other gets nothing, so a signal is
/// never handled twice. Clones share one claim.
///
/// - [`Self::watch`] waits for the signal, records its name
///   ([`Self::cause`]) and cancels the shutdown request; then, if the parts
///   are still unclaimed, it takes them and runs the startup stop.
/// - [`Self::claim_serving`] takes the parts for the serving path, or answers
///   `None` when the startup stop already has them.
/// - [`Self::request`] hands out children of the shutdown request, cancelled
///   on the first signal in every phase.
pub struct StartupClaim<P> {
    state: Arc<ClaimState<P>>,
}

impl<P> Clone for StartupClaim<P> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<P> StartupClaim<P> {
    /// A claim over `parts`, unclaimed, with no shutdown requested.
    pub fn new(parts: P) -> Self {
        Self {
            state: Arc::new(ClaimState {
                phase: Mutex::new(StartupPhase::Booting(parts)),
                request: CancellationToken::new(),
                cause: OnceLock::new(),
            }),
        }
    }

    fn phase(&self) -> MutexGuard<'_, StartupPhase<P>> {
        // Every operation under the lock is a `mem::replace`; a panic in one
        // leaves the phase consistent.
        self.state.phase.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A child of the process shutdown request: cancelled on the first
    /// signal, whether startup or the serving path acts on it. Cancelling the
    /// child does not cancel the request.
    pub fn request(&self) -> CancellationToken {
        self.state.request.child_token()
    }

    /// The first signal's name ([`SIGTERM`] or [`SIGINT`]), once one has
    /// arrived. Set before the shutdown request is cancelled, so it is `Some`
    /// for anyone woken by the request.
    pub fn cause(&self) -> Option<&'static str> {
        self.state.cause.get().copied()
    }

    /// Claim the startup parts for the serving path: `Some` with the parts
    /// and a child of the shutdown request ([`Self::request`]). The child is
    /// already cancelled when a signal's watcher cancelled the request just
    /// before this claim took the parts; the serving path then drains at
    /// once.
    ///
    /// `None` when a signal claimed startup first: its startup stop owns the
    /// parts and ends the process, so the caller waits for that and does not
    /// serve. Called once; a second call finds no parts and also answers
    /// `None`.
    pub fn claim_serving(&self) -> Option<(P, CancellationToken)> {
        let mut phase = self.phase();
        match std::mem::replace(&mut *phase, StartupPhase::Serving) {
            StartupPhase::Booting(parts) => Some((parts, self.request())),
            other => {
                *phase = other;
                None
            }
        }
    }

    /// The watcher: wait for `signal`, record its name, cancel the shutdown
    /// request, and, if the startup parts are unclaimed, take them and run
    /// `stop` with the signal's name and the parts. Once the serving path
    /// has claimed them, cancelling the request is all it does.
    ///
    /// A signal that arrives before this is first polled (`signal` resolves
    /// at once) is acted on as soon as it is. `main.rs` spawns this with the
    /// process's [`ShutdownSignal`] and its startup stop, which ends the
    /// process; under a `stop` that returns, this returns after it.
    pub async fn watch<S, F, Fut>(self, signal: S, stop: F)
    where
        S: Future<Output = &'static str>,
        F: FnOnce(&'static str, P) -> Fut,
        Fut: Future<Output = ()>,
    {
        let name = signal.await;
        let _ = self.state.cause.set(name);
        self.state.request.cancel();
        let parts = {
            let mut phase = self.phase();
            match std::mem::replace(&mut *phase, StartupPhase::StoppingStartup) {
                StartupPhase::Booting(parts) => Some(parts),
                other => {
                    *phase = other;
                    None
                }
            }
        };
        if let Some(parts) = parts {
            stop(name, parts).await;
        }
    }
}

/// Bind the gRPC listening socket once and return two `TcpIncoming` streams
/// over the same open file description, plus the address actually bound
/// (which differs from `addr` when the caller asked for port 0).
///
/// The first stream is for the bootstrap health server, the second for the
/// full-API server. Errors are returned rather than logged so the caller owns
/// the fail-fast exit — an `EADDRINUSE` now surfaces at the top of startup
/// instead of ~900 lines later, after all dependency wiring has run.
///
/// `keepalive` is the `TCP_KEEPALIVE` time applied to accepted connections on
/// both streams; `TCP_NODELAY` is always enabled, matching tonic's default.
///
/// Must be called from inside a tokio runtime context —
/// `tokio::net::TcpListener::from_std` registers the descriptor with the
/// current reactor.
pub fn bind_grpc_listener(
    addr: SocketAddr,
    keepalive: Duration,
) -> io::Result<(TcpIncoming, TcpIncoming, SocketAddr)> {
    let listener = StdTcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    let bound_addr = listener.local_addr()?;

    // `try_clone` dups the descriptor: the clone refers to the *same* open
    // file description as the original, so it shares the listen backlog and —
    // because `O_NONBLOCK` is a property of the description, not of the
    // descriptor — inherits the `set_nonblocking(true)` applied above. That
    // sharing is exactly what makes the bootstrap → full-API handoff seamless,
    // and it is also why no second `set_nonblocking` call is needed here.
    let cloned = listener.try_clone()?;

    let bootstrap = into_incoming(listener, keepalive)?;
    let serving = into_incoming(cloned, keepalive)?;

    Ok((bootstrap, serving, bound_addr))
}

fn into_incoming(listener: StdTcpListener, keepalive: Duration) -> io::Result<TcpIncoming> {
    let listener = tokio::net::TcpListener::from_std(listener)?;
    Ok(TcpIncoming::from(listener)
        .with_nodelay(Some(true))
        .with_keepalive(Some(keepalive)))
}

/// Wrap the full-API server's `TcpIncoming` so the listening socket can be
/// closed while the server is still running, and its accepted connections
/// counted.
///
/// tonic keeps the incoming stream it serves until `serve_with_incoming*`
/// returns — after every connection has closed, which in a graceful shutdown
/// is after the paced stream ending. A socket still in `LISTEN` that long lets
/// a stale client's connect sit in a backlog nobody accepts.
/// [`ListenerControl::close`] drops the listener at once instead, so a connect
/// is refused and retried elsewhere; `main.rs` calls it when the readiness
/// drain ends. The bootstrap server's duplicated descriptor is gone by then
/// (it is dropped at the handoff), so this closes the socket.
///
/// Each accepted connection is a [`CountedTcpStream`], counted in
/// [`ListenerControl::open_connections`] until tonic's connection task drops
/// it.
pub fn closable_incoming(incoming: TcpIncoming) -> (ClosableIncoming, ListenerControl) {
    let shared = Arc::new(ListenerShared {
        incoming: Mutex::new(Some(incoming)),
        open_connections: Arc::new(AtomicUsize::new(0)),
    });
    (
        ClosableIncoming {
            shared: Arc::clone(&shared),
        },
        ListenerControl { shared },
    )
}

struct ListenerShared {
    /// `None` once closed.
    incoming: Mutex<Option<TcpIncoming>>,
    open_connections: Arc<AtomicUsize>,
}

impl ListenerShared {
    fn incoming(&self) -> MutexGuard<'_, Option<TcpIncoming>> {
        // Every operation under the lock is a poll or a `take`; a panic in
        // one leaves the slot consistent.
        self.incoming.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// The serving half of [`closable_incoming`]: what the server serves from.
/// Ends (yields `None`) once the listener is closed.
pub struct ClosableIncoming {
    shared: Arc<ListenerShared>,
}

/// The control half of [`closable_incoming`]. Cloning shares it.
#[derive(Clone)]
pub struct ListenerControl {
    shared: Arc<ListenerShared>,
}

impl ListenerControl {
    /// Close the listening socket now: no further connection is accepted,
    /// and a connect is refused. Idempotent. Connections already accepted are
    /// not touched.
    pub fn close(&self) {
        let listener = self.shared.incoming().take();
        drop(listener);
    }

    /// Whether [`Self::close`] has run.
    pub fn is_closed(&self) -> bool {
        self.shared.incoming().is_none()
    }

    /// The number of accepted connections still open.
    pub fn open_connections(&self) -> usize {
        self.shared.open_connections.load(Ordering::Acquire)
    }
}

impl Stream for ClosableIncoming {
    type Item = io::Result<CountedTcpStream>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut slot = self.shared.incoming();
        let Some(incoming) = slot.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(incoming).poll_next(cx) {
            Poll::Ready(Some(Ok(stream))) => {
                self.shared.open_connections.fetch_add(1, Ordering::AcqRel);
                Poll::Ready(Some(Ok(CountedTcpStream {
                    inner: stream,
                    open_connections: Arc::clone(&self.shared.open_connections),
                })))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// An accepted connection of a [`ClosableIncoming`], counted open until
/// dropped. Reads, writes and connection info pass straight through to the
/// `TcpStream`.
pub struct CountedTcpStream {
    inner: TcpStream,
    open_connections: Arc<AtomicUsize>,
}

impl Drop for CountedTcpStream {
    fn drop(&mut self) {
        self.open_connections.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Connected for CountedTcpStream {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        self.inner.connect_info()
    }
}

impl AsyncRead for CountedTcpStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for CountedTcpStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    //! [`StartupClaim`]: which side acts on the first shutdown signal, the
    //! startup stop or the serving path, with a fake signal future in place of
    //! [`ShutdownSignal::recv`], so a test chooses when the signal "arrives".
    //! The real handlers, and a signal to the binary while it is blocked in
    //! startup, are `tests/misc/startup_signal_binary.rs`.

    use std::sync::atomic::AtomicBool;

    use tokio::sync::oneshot;

    use super::*;

    /// The startup parts every test claim carries, so a test can tell which
    /// side took them.
    const PARTS: &str = "the bootstrap and /metrics servers";

    /// Bound on every wait here. A deadline, not a timing claim: each wait
    /// ends on its own in microseconds.
    const WAIT: Duration = Duration::from_secs(5);

    /// What a startup stop was handed: the signal's name and the parts.
    type StopLog = Arc<Mutex<Vec<(&'static str, &'static str)>>>;

    /// A startup stop that records what it was handed and returns (the
    /// production stop ends the process instead).
    fn recording_stop(
        log: &StopLog,
    ) -> impl FnOnce(&'static str, &'static str) -> std::future::Ready<()> {
        let log = Arc::clone(log);
        move |signal, parts| {
            log.lock().unwrap().push((signal, parts));
            std::future::ready(())
        }
    }

    /// A signal delivered before the watcher starts — between the listener's
    /// install at the top of `main` and the watcher's spawn — is acted on
    /// once the watcher runs: the claim does nothing on its own, and the
    /// watcher then records the cause, cancels the request and runs the
    /// startup stop exactly once, with the signal's name and the parts.
    #[tokio::test]
    async fn a_signal_delivered_before_the_watcher_starts_stops_startup_once_it_runs() {
        let claim = StartupClaim::new(PARTS);
        let request = claim.request();
        let (delivered, recorded) = oneshot::channel::<&'static str>();
        delivered
            .send(SIGTERM)
            .expect("the fake listener still holds its receiver");
        assert!(
            !request.is_cancelled() && claim.cause().is_none(),
            "nothing acts on a signal before the watcher runs"
        );

        let stopped = StopLog::default();
        let watcher = tokio::spawn(claim.clone().watch(
            async move { recorded.await.expect("the signal was recorded") },
            recording_stop(&stopped),
        ));
        tokio::time::timeout(WAIT, watcher)
            .await
            .expect("the watcher acts on the recorded signal at once")
            .expect("the watcher does not panic");

        assert_eq!(
            *stopped.lock().unwrap(),
            vec![(SIGTERM, PARTS)],
            "the startup stop ran once, with the signal's name and the startup parts"
        );
        assert_eq!(claim.cause(), Some(SIGTERM));
        assert!(request.is_cancelled(), "the shutdown request is cancelled");
    }

    /// Once a signal has started the startup stop, the serving path's claim
    /// answers `None`, so it never takes the bootstrap server over, reaches the
    /// serving point or opens readiness: the stop owns the parts, and ends the
    /// process. Here the stop parks, as the production one does until its
    /// `exit(0)`.
    #[tokio::test]
    async fn the_serving_claim_answers_none_once_a_signal_has_started_the_startup_stop() {
        let claim = StartupClaim::new(PARTS);
        let (started, stop_started) = oneshot::channel::<(&'static str, &'static str)>();
        let _watcher = tokio::spawn(claim.clone().watch(
            std::future::ready(SIGINT),
            move |signal, parts| {
                let _ = started.send((signal, parts));
                std::future::pending::<()>()
            },
        ));
        let handed = tokio::time::timeout(WAIT, stop_started)
            .await
            .expect("the startup stop starts")
            .expect("the stop reports what it was handed");
        assert_eq!(handed, (SIGINT, PARTS));

        assert!(
            claim.claim_serving().is_none(),
            "the serving path gets nothing once the startup stop has the parts"
        );
        assert_eq!(claim.cause(), Some(SIGINT));
        assert!(claim.request().is_cancelled());
    }

    /// Once the serving path has claimed the parts, a signal only records its
    /// cause and cancels the request — the startup stop never runs — and the
    /// serving path, woken by its request, reads the cause. A second claim
    /// finds no parts.
    #[tokio::test]
    async fn a_signal_after_the_serving_claim_only_cancels_the_request() {
        let claim = StartupClaim::new(PARTS);
        let (deliver, delivered) = oneshot::channel::<&'static str>();
        let stop_ran = Arc::new(AtomicBool::new(false));
        let watcher = tokio::spawn(claim.clone().watch(
            async move { delivered.await.expect("the test delivers a signal") },
            {
                let stop_ran = Arc::clone(&stop_ran);
                move |_, _| {
                    stop_ran.store(true, Ordering::SeqCst);
                    std::future::ready(())
                }
            },
        ));

        let (parts, request) = claim
            .claim_serving()
            .expect("unclaimed parts go to the serving path");
        assert_eq!(parts, PARTS);
        assert!(
            !request.is_cancelled() && claim.cause().is_none(),
            "no signal yet"
        );
        let woken_with = tokio::spawn({
            let claim = claim.clone();
            let request = request.clone();
            async move {
                request.cancelled().await;
                claim.cause()
            }
        });

        deliver
            .send(SIGINT)
            .expect("the watcher still waits for its signal");
        tokio::time::timeout(WAIT, watcher)
            .await
            .expect("the watcher returns once it has cancelled the request")
            .expect("the watcher does not panic");

        assert_eq!(
            tokio::time::timeout(WAIT, woken_with)
                .await
                .expect("the serving path is woken by the request")
                .expect("the reader does not panic"),
            Some(SIGINT),
            "the cause is recorded before the request is cancelled"
        );
        assert!(
            !stop_ran.load(Ordering::SeqCst),
            "the startup stop never runs once the serving path has the parts"
        );
        assert!(
            claim.claim_serving().is_none(),
            "a second claim finds no parts"
        );
    }

    /// A child of the shutdown request — what `main.rs` stops the hold poll
    /// and the hold reactor on — is cancelled by the first signal in both
    /// phases: while booting, when the startup stop acts, and once serving,
    /// when only the request is cancelled. Cancelling a child alone leaves the
    /// request and its other children as they were.
    #[tokio::test]
    async fn a_child_of_the_request_is_cancelled_by_a_signal_while_booting_and_while_serving() {
        for serving in [false, true] {
            let phase = if serving { "serving" } else { "booting" };
            let claim = StartupClaim::new(PARTS);
            let child = claim.request();
            let sibling = claim.request();
            sibling.cancel();
            assert!(
                !child.is_cancelled() && !claim.request().is_cancelled(),
                "{phase}: cancelling one child cancels neither the request nor another child"
            );
            if serving {
                claim.claim_serving().expect("the parts are unclaimed");
            }

            let stopped = StopLog::default();
            tokio::time::timeout(
                WAIT,
                claim
                    .clone()
                    .watch(std::future::ready(SIGTERM), recording_stop(&stopped)),
            )
            .await
            .unwrap_or_else(|_| panic!("{phase}: the watcher returns"));

            assert!(
                child.is_cancelled(),
                "{phase}: the first signal cancels a child of the request"
            );
            assert_eq!(
                stopped.lock().unwrap().len(),
                usize::from(!serving),
                "{phase}: the startup stop runs only while booting"
            );
        }
    }
}
