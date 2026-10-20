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

//! The live-edit service's changes to fred 10.1.0.
//!
//! fred is vendored by the live-edit repository as the published 10.1.0 crate plus one patch file. The patch edits
//! existing lines only, so every file keeps its line count; new code lives in this module, and each hook in an
//! existing file carries a `live-edit patch` comment. The record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! Shared state is one settings field on fred's config (`ConnectionConfig::live_edit`, a [`Settings`]), one
//! per-client field (`ClientInner::live_edit`, a [`ClientState`]), one per-connection field
//! (`Connection::live_edit`, a [`ConnectionState`]), one per-pool field (`PoolInner::live_edit`, a
//! [`pool::PoolState`]) and one field on each connection's codec (`Codec::live_edit`, a [`CodecState`]). Hooks in
//! existing files call the functions here, which dispatch to the submodules:
//!
//! - [`abandon`]: a command, pipeline or transaction is never written after its caller gave up on it (its timeout
//!   fired, fred answered it with an error, or its future was dropped); a pipeline cut during the router step that
//!   writes it, and a fred `Transaction` cut partway, never leave a `MULTI` open. A pipeline replayed from the retry
//!   buffer is not covered; the service sends every `MULTI` pipeline at one write attempt, so it never replays one
//!   (the entry points, `process_pipeline` and the transaction router call it directly).
//! - [`ask`]: following a cluster `-ASK` redirect to its target node, with a pipelined `ASKING` and no synchronous
//!   read (`process_ask` calls [`ask::follow`] directly).
//! - [`liveness`]: the unresponsive check, bounded router waits, closing a dead connection without waiting on it,
//!   flushing fed frames before the router waits, connects inline or writes where the write can reach a socket, all
//!   connections at once, the command-queue counter's resync, pubsub classification and the state a verdict records.
//! - [`pool`]: a `Pool`'s logical members, each re-pointable to a fresh client.
//! - [`retention`]: what a connection keeps once its traffic drains (connection buffer limits).

pub mod abandon;
pub mod ask;
pub mod liveness;
pub mod pool;
pub mod retention;

pub use retention::CodecState;

use crate::{
  error::Error,
  modules::inner::{ClientInner, CommandReceiver},
  protocol::{
    codec::Codec,
    command::Command,
    connection::{Connection, ExclusiveConnection},
    types::{ProtocolFrame, Server},
  },
  router::Router,
  runtime::RefCount,
};
use redis_protocol::resp3::types::BytesFrame as Resp3Frame;
use std::{
  collections::{HashMap, VecDeque},
  sync::atomic::AtomicUsize,
  time::{Duration, Instant},
};

pub(crate) use liveness::Classify;

/// Settings for the live-edit additions, on `ConnectionConfig::live_edit`. The default keeps fred's stock
/// behaviour apart from the fixes that always apply (see each submodule).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Settings {
  /// Connection liveness settings.
  pub liveness:  liveness::Settings,
  /// Per-connection buffer limits ([`retention`]). `None`, the default, keeps stock fred's buffers.
  pub retention: Option<retention::RetentionLimits>,
}

/// Per-client state for the live-edit additions, on `ClientInner::live_edit`.
#[derive(Clone, Debug, Default)]
pub struct ClientState {
  pub(crate) liveness:  liveness::ClientState,
  /// The test-only retention report, sweep mode, and sweep and write-check counters ([`retention::test_support`]),
  /// compiled only with the `live-edit-retention-report` feature.
  #[cfg(feature = "live-edit-retention-report")]
  pub(crate) retention: std::sync::Arc<retention::test_support::ReportState>,
}

/// Per-connection state for the live-edit additions, on `Connection::live_edit`.
#[derive(Debug, Default)]
pub struct ConnectionState {
  pub(crate) liveness: liveness::ConnectionState,
  pub(crate) abandon:  abandon::ConnectionState,
}

/// Make a pipelined connection from an initialized one (stock `into_pipelined`) and set up its state.
pub(crate) fn pipelined(inner: &RefCount<ClientInner>, transport: ExclusiveConnection, replica: bool) -> Connection {
  let mut conn = transport.into_pipelined(replica);
  liveness::attach(inner, &mut conn);
  conn
}

/// `Connection::push_command`: record the command; returns whether it has no responses.
pub(crate) fn on_push(conn: &mut Connection, cmd: &Command) -> bool {
  liveness::on_push(conn, cmd)
}

/// `Connection::write`: count a tracked write.
pub(crate) fn on_write(conn: &mut Connection) {
  liveness::count_unsent(conn);
}

/// A reply was matched to `command`, popped from `conn`'s in-flight queue. Returns `command`.
pub(crate) fn on_reply(inner: &RefCount<ClientInner>, conn: &mut Connection, command: Command) -> Command {
  let command = liveness::on_reply(inner, conn, command);
  retention::on_reply(conn);
  command
}

/// The read loop took a frame, or the end of the stream, off `conn`.
pub(crate) fn on_frame(conn: &mut Connection, now: Instant) {
  liveness::on_frame(conn, now);
  retention::on_read(conn);
}

/// Write and flush a frame.
pub(crate) async fn send(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
  liveness::send(conn, frame).await?;
  retention::on_flushed(conn);
  Ok(())
}

/// Buffer a frame.
pub(crate) async fn feed(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
  liveness::feed(conn, frame).await
}

/// Flush the connection.
pub(crate) async fn flush(conn: &mut Connection) -> Result<(), Error> {
  liveness::flush(conn).await?;
  retention::on_flushed(conn);
  Ok(())
}

/// Read the next frame.
pub(crate) async fn next(conn: &mut Connection) -> Result<Option<Result<ProtocolFrame, Error>>, Error> {
  let item = liveness::next(conn).await?;
  // Read rule here: a fred `Transaction`'s `exec` (CreateLayer's `client.multi()`) reads its EXEC reply via `next`.
  retention::on_read(conn);
  Ok(item)
}

/// `Connection::close`: close the transport, unless the connection was judged dead or is closed under an abandoned
/// transaction ([`abandon::close_transaction`]); either returns at once, with no flush and no shutdown.
pub(crate) async fn close(conn: &mut Connection) -> Result<(), Error> {
  if abandon::closes_at_once(conn) {
    return Ok(());
  }
  liveness::close(conn).await
}

/// Convert a frame read from `conn` to RESP3.
pub(crate) fn into_resp3(conn: &Connection, frame: ProtocolFrame) -> Resp3Frame {
  liveness::into_resp3(conn, frame)
}

/// How frames read from `conn` are classified (pubsub or data).
pub(crate) fn classify(conn: &Connection) -> Classify {
  liveness::classify(conn)
}

/// How frames read from the connection to `server` are classified (pubsub or data).
pub(crate) fn subscribed(router: &mut Router, server: &Server) -> Classify {
  liveness::subscribed(router, server)
}

/// Whether `frame` may be classified as a pubsub frame under `classify`.
pub(crate) fn pubsub_classified(classify: &Classify, frame: &Resp3Frame) -> bool {
  liveness::pubsub_classified(classify, frame)
}

/// [`pubsub_classified`], once per frame, counting a pubsub-shaped frame it does not classify as pubsub.
pub(crate) fn classify_frame(classify: &Classify, frame: &Resp3Frame, subscription_response: bool) -> bool {
  liveness::classify_frame(classify, frame, subscription_response)
}

/// The read loop found `conn` pending: when its current stall began, if it has one to judge.
pub(crate) fn stall_start(inner: &RefCount<ClientInner>, conn: &mut Connection, now: Instant) -> Option<Instant> {
  liveness::stall_start(inner, conn, now)
}

/// The error that closes `conn` as unresponsive, when the stall that began at `since` is a verdict.
pub(crate) fn verdict(
  inner: &RefCount<ClientInner>,
  conn: &mut Connection,
  now: Instant,
  since: Instant,
  slow: Duration,
) -> Option<Error> {
  liveness::verdict(inner, conn, now, since, slow)
}

/// The read loop has polled every connection of a clustered (or replica) client: log the verdicts it reached.
pub(crate) fn log_verdicts_all(inner: &RefCount<ClientInner>, connections: &mut HashMap<Server, Connection>) {
  if connections.values().any(liveness::has_pending) {
    liveness::log_verdicts(inner, connections.values_mut());
  }
}

/// The read loop has polled a centralized client's connection: log the verdict it reached, if any.
pub(crate) fn log_verdicts_one(inner: &RefCount<ClientInner>, connection: &mut Connection) {
  if liveness::has_pending(connection) {
    liveness::log_verdicts(inner, std::iter::once(connection));
  }
}

/// The top of the router's `read_or_write`: the connection-buffer sweep, then flush fed frames when no queued command
/// follows.
pub(crate) async fn before_wait(inner: &RefCount<ClientInner>, router: &mut Router, rx: &CommandReceiver) {
  // Test-only: the retention report request pending before this pass's sweep begins, if any. Read first so that the
  // sweep below runs under every sweep mode a test set before that request.
  #[cfg(feature = "live-edit-retention-report")]
  let report_request = retention::test_support::pending_report(inner);
  retention::sweep(inner, router);
  // Test-only: answer that request with what this pass's sweep left. Deliberately outside `sweep`, so a build with
  // the sweep call removed still answers reports, and that perturbation fails on what the report shows rather than
  // on a report that never comes.
  #[cfg(feature = "live-edit-retention-report")]
  if let Some(request) = report_request {
    retention::test_support::answer_report(inner, router, request);
  }
  liveness::before_wait(inner, router, rx).await;
}

/// Before every inline connect, the one entry point of all three sites: the top of every pass of
/// `reconnect_with_policy` and `sync_cluster_with_policy`, before the pass's backoff sleep and its attempt, and
/// [`ask::follow`], before it opens a connection to an `-ASK` target. On a clustered client, flush every connection
/// holding fed but unflushed frames, all at once and whatever the command channel holds
/// ([`liveness::flush_fed_concurrently`]; [`liveness`]'s *Never waiting on unsent frames*), so the connect waits at
/// most one bound for it however many connections hold fed frames. The connect runs inline, polling no connection,
/// and keeps the connections it does not replace, so a frame left fed on one would otherwise wait through it.
///
/// Any other client does nothing here: its reconnect attempt closes its one connection before it connects, and that
/// close flushes it (or, when the connection was judged dead, returns at once without flushing). Flushing first would
/// only replace that close's flush, bounded by fred's 5 s close timeout, with one bounded by the slow liveness bound
/// (`UnresponsiveConfig::max_timeout`). `ask::follow` runs only on a clustered client, so the gate never skips it.
pub(crate) async fn before_inline_connect(inner: &RefCount<ClientInner>, router: &mut Router) {
  if inner.config.server.is_clustered() {
    let site = liveness::FlushSite::InlineConnect(inner);
    let _ = liveness::flush_fed_concurrently(liveness::connections_mut(router), site).await;
  }
}

/// Before the router writes `command` to the one connection it routes to: the top of the single-connection branch of
/// `write_command` (`src/router/commands.rs`), which every single command, every pipeline command, the retry buffer's
/// replay after a reconnect or cluster sync, and a `-MOVED` retry's write go through. On a clustered client, when the
/// write can reach a socket and another connection holds fed but unflushed frames, flush every connection holding
/// them, the destination included, all at once ([`liveness::before_socket_write`]; [`liveness`]'s *Never waiting on
/// unsent frames*), so no frame fed elsewhere waits behind this write's own socket wait.
///
/// Any other client has one connection, so no other connection's frames can wait behind the write: it returns at
/// once.
pub(crate) async fn before_socket_write(inner: &RefCount<ClientInner>, router: &mut Router, command: &Command) {
  if inner.config.server.is_clustered() {
    liveness::before_socket_write(inner, router, command).await;
  }
}

/// Before one of this patch's own writes to `server` outside `write_command`: [`ask::follow`]'s `ASKING` and command
/// to an `-ASK` target, and [`abandon::after_pipeline`]'s `DISCARD`. On a clustered client, when another connection
/// holds fed but unflushed frames, flush every connection holding them, all at once, as [`before_socket_write`] does,
/// without its write predicate: these paths are rare ([`liveness::before_write_to`]). Any other client returns at
/// once, as there.
pub(crate) async fn before_write_to(inner: &RefCount<ClientInner>, router: &mut Router, server: &Server) {
  if inner.config.server.is_clustered() {
    liveness::before_write_to(inner, router, server).await;
  }
}

/// fred's router-wide flush of a clustered client's connections (`Connections::flush`, which `Router::flush` runs
/// after a successful send, after the retry buffer's replay and at the start of `drain_all`): flush every connection
/// holding fed but unflushed frames, all at once and each to its own end, and return the first error, which the
/// callers log or return as they did stock's ([`liveness::flush_fed_concurrently`]). Stock flushed every connection
/// through `try_join_all`, which allocated on every call and dropped the other flushes at the first error; this
/// flushes only connections holding fed frames, allocates only when two or more do, and lets a fast error on one
/// never cut another's flush short. A centralized or sentinel client keeps stock's flush of its one connection.
pub(crate) async fn flush_router_wide<'a>(connections: impl Iterator<Item = &'a mut Connection>) -> Result<(), Error> {
  liveness::flush_fed_concurrently(connections, liveness::FlushSite::RouterWide).await
}

/// The router took a command off its channel: resync the command-queue counter.
pub(crate) fn on_receive(inner: &RefCount<ClientInner>, rx: &CommandReceiver) {
  liveness::on_receive(inner, rx);
}

/// Decrement the command-queue counter, saturating at zero.
pub(crate) fn saturating_decr(counter: &AtomicUsize) -> usize {
  liveness::saturating_decr(counter)
}

/// The retention state for a new codec of the client `inner`.
pub(crate) fn codec_state(inner: &RefCount<ClientInner>) -> CodecState {
  retention::codec_state(inner)
}

/// The codec encoded a frame, leaving `buffered` bytes in the write buffer.
pub(crate) fn on_encode(codec: &Codec, buffered: usize) {
  retention::on_encode(codec, buffered);
}

/// The codec decoded a frame of `size` bytes.
pub(crate) fn on_decode(codec: &Codec, size: usize) {
  retention::on_decode(codec, size);
}

/// The router replayed its retry buffer after a reconnect.
pub(crate) fn on_retry_replayed(inner: &RefCount<ClientInner>, retry_buffer: &mut VecDeque<Command>) {
  retention::on_retry_replayed(inner, retry_buffer);
}
