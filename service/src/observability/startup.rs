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

//! Early gRPC listener bind for the bootstrap health server (b.rgs).
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

use std::io;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tonic::transport::server::{Connected, TcpConnectInfo, TcpIncoming};

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
