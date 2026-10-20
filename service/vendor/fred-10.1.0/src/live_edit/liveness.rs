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

//! Connection liveness: how fred decides a pooled connection has gone silent, how long any router wait on one
//! connection's I/O may last (closing a connection judged dead waits on nothing), that the router never waits idle,
//! connects inline or waits on one connection's write while frames it has not sent sit on another, which replies count
//! as pubsub frames, and what fred records when it closes a connection as unresponsive.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! # The unresponsive check
//!
//! Stock fred stamps `last_write` on every non-pubsub write, clears it whenever the read loop takes a frame, and
//! closes a connection whose stamp is older than `UnresponsiveConfig::max_timeout`. That measures the *newest*
//! write: a connection that keeps receiving writes never trips it, however long its oldest command has waited, and
//! a reply read outside the read loop (a drain) leaves a stale stamp that can trip it on a healthy connection.
//!
//! This module replaces the rule. For each connection with at least one outstanding non-pubsub command that is not
//! `blocked`, the *stall* is the time since the later of:
//!
//! - when the oldest outstanding non-pubsub command was written: handed to the socket by a send or flush that
//!   completed (`Command::network_start`, re-stamped here when its frame is flushed, so a large frame's own upload
//!   time does not count as silence) — or, while it is still only fed, the later of its prepare time and the
//!   connection's last completed write, feed or flush, so a large frame being uploaded ahead of it is not silence
//!   either; and
//! - when the connection last made read progress: a frame decoded, or its read buffer grew.
//!
//! Two verdicts close the connection through fred's existing path; in-flight commands with attempts left are re-sent
//! on the replacement connection, and a command with none fails with the verdict's error:
//!
//! - **fast**: the stall exceeds [`Settings::fast_timeout`] *and* some other connection in the process received a
//!   reply from the same cluster slot range at least [`Settings::peer_evidence_lag`] after this connection's stall
//!   began ([`PeerReplies`]). The shard answers others, so this socket is the dead one. A command is judged by its
//!   slot's range only when the client's cluster routing gives that slot to the connection's node, whether the slot
//!   comes from a key, a `ClusterHash::Custom` pin or a hashed argument: that slot's shard runs it. Any other — a
//!   command with nothing to hash, one whose hashed slot another node owns (sent to a named node), or any sent
//!   before the client has cluster routing — is judged at node level: any reply from the connection's node counts. A
//!   centralized client's node owns every slot. Its error text is [`UNRESPONSIVE_FAST`].
//! - **slow**: the stall exceeds `UnresponsiveConfig::max_timeout`, whatever peers are doing. This covers a shard
//!   that is silent for everyone. Its error text is stock fred's, [`UNRESPONSIVE`].
//!
//! [`verdict_of`] tells the two apart from an error, so a caller never compares the texts itself.
//!
//! The check runs where stock fred's ran, in the read loop's `poll_connection`, which `read_or_write` re-polls at
//! least every `UnresponsiveConfig::interval`. The peer evidence (a read of the slot range's reply times) is
//! evaluated at most once per interval per connection, and only once a stall has passed the evidence lag.
//!
//! # Bounded waits
//!
//! Every wait on one connection's I/O outside the read loop (a write, a flush, a frame read by a drain, a broadcast
//! reply read, an `ASKING` read, a transaction's `EXEC` and skip reads) is bounded by `max_timeout`, the slow bound.
//! When the bound expires that connection is treated as failed: the wait returns an IO error with the [`UNRESPONSIVE`]
//! text into the caller's existing error path, and the connection is marked so the read loop also reports it, which
//! covers the callers that ignore a flush error. With `max_timeout: None` (fred's default) no wait is bounded.
//!
//! From then on every write, feed, flush or read on the marked connection fails at once with the same error, before
//! any I/O and without a second log line, until fred closes it. The expiry already decided the connection is dead and
//! nothing recovers it, so a second wait could only delay what each caller then does anyway, while holding the router
//! and every command queued behind it: callers that clean up on the connection they just failed on (a transaction's
//! `DISCARD`, a re-flush of frames left fed, a drain, an `ASKING` retry) would otherwise each wait up to another bound
//! (the `DISCARD` up to the lesser of that and fred's `internal_command_timeout`). The marker is never cleared; it is
//! dropped with the connection.
//!
//! The close is the last wait on a connection, on every path that closes one. Stock `Connection::close` first flushes
//! fred's write buffer and shuts the transport down (under TLS, sending `close_notify` first), under fred's 5 s
//! `CONNECTION_CLOSE_TIMEOUT_MS`. After an expired write or flush the bytes still buffered are, by construction, bytes
//! the kernel would not take, so that close always waits out the 5 s; after a read-loop verdict it waits whenever the
//! socket's send queue refuses what fred still holds, which on a fast verdict can push the re-send past the command
//! timeout. So a connection judged dead — an expired wait, or a verdict the read loop reached (`closed_by`) — is
//! closed without a flush or a shutdown ([`close`]): its in-flight commands are returned as before, and dropping the
//! connection then closes the socket without blocking. So one silent connection holds the router for at most one bound
//! per wait chain, or per attempt of a transaction, its close included. Every other close keeps fred's stock close.
//! A frame fed on one connection does not wait through other connections' waits: before any of them, every fed frame
//! is flushed, all connections at once (*Never waiting on unsent frames*, which lists the exceptions), so the frame
//! reaches its socket within one bound, or its connection is failed, however many of the client's connections stall.
//!
//! # Never waiting on unsent frames
//!
//! fred writes a command without flushing it when its command-queue counter says another command follows, and
//! relies on a later write to flush it. Two paths leave such a frame fed but never flushed even with a balanced
//! counter (a timed-out command skipped at write flushes only its own connection; a skipped `Reconnect` returns
//! without flushing), and a counter that drifted above the channel's length would hold every write back. And while a
//! frame sits fed, the router can wait on other connections' bounded I/O, each wait up to the slow bound; waited one
//! after another, k silent connections would hold the frame for k bounds.
//!
//! The rule, on a clustered client: the router never waits on one connection's I/O while another connection holds fed
//! but unflushed frames, except inside a flush that runs every such connection's flush at once, each to its own end
//! ([`flush_fed_concurrently`]). That flush holds the router for at most one bound however many connections stall in
//! it, a fast error on one never cuts another's short, and it flushes only connections holding fed frames: with none
//! it is one walk of the connections and with one a plain flush, so neither adds an allocation to what the flush
//! itself does; two or more are joined (`join_all`), which allocates the join. A centralized or sentinel client has
//! one connection, so no frame of it can wait behind another connection's I/O. So:
//!
//! - **before the router waits** ([`before_wait`], at the top of `read_or_write`): when its command channel is empty,
//!   every connection holding fed but unflushed frames is flushed, all at once. The channel itself is checked, not the
//!   counter: this is fred's own rule, "flush when no queued command follows", applied by construction;
//! - **before every inline connect** of a clustered client (`live_edit::before_inline_connect`, called at the top of
//!   every pass of `reconnect_with_policy` and `sync_cluster_with_policy`, before its backoff sleep and its attempt,
//!   and by `live_edit::ask::follow`, before it opens a connection to an `-ASK` target): every connection holding
//!   fed but unflushed frames is flushed whatever the channel holds, all at once. The router command that leads to
//!   the connect (a `Reconnect`, `SyncCluster`, `Moved` or `Ask`) is itself queued and counted, so the command routed
//!   just ahead of it is fed, not flushed, and the queued command keeps the channel non-empty, so [`before_wait`]
//!   skips that frame. The connect then runs inline, polling no other connection, so neither a write nor a read-loop
//!   verdict reaches a frame left fed on a healthy node's connection until the connect ends and a later flush sends
//!   it (after a reconnect or sync, the retry buffer's). Flushing at the top of each pass, before the backoff sleep and
//!   not only before the attempt, also lets the sleep's reads take that connection's reply, so no verdict closes a
//!   healthy connection over a frame that was only unsent. The cost is at most one bound per pass or `-ASK` connect,
//!   however many connections hold fed frames: the same wait fred's own router-wide flush after a successful attempt
//!   would have spent, only earlier. A centralized or sentinel client's attempt closes its one connection first, and
//!   that close flushes it (or, on one judged dead, drops it), so this flush is not run there;
//! - **before a write that can reach a socket** ([`before_socket_write`], through `live_edit::before_socket_write` at
//!   the top of the single-connection branch of the router's `write_command`, which every single command, every
//!   pipeline command, the retry buffer's replay and a `-MOVED` retry's write go through): the command is routed to its
//!   destination, and when the write can reach that connection's socket and another connection holds fed frames,
//!   every connection holding them, the destination included, is flushed all at once before the write. The write can
//!   reach the socket ([`write_reaches_socket`]) when fred's own flush rule would send it — the command-queue counter
//!   at zero, the destination's feed count past `max_feed_count`, or a kind fred always sends (`should_flush`, with
//!   custom commands and scripts among them) — or when it is a feed onto a write buffer at tokio-util's
//!   8 KiB backpressure boundary, which flushes that buffer first. A feed below the boundary only encodes into the
//!   buffer and cannot block, so it triggers nothing, which keeps fred's batching across connections below the
//!   boundary; and when only the destination holds fed frames nothing is flushed first, so its own write batches them
//!   as stock does. This patch's rare writes outside `write_command` (`ask::follow`'s, `abandon::after_pipeline`'s)
//!   flush whenever another connection holds fed frames, without the predicate ([`before_write_to`]). Per command the
//!   check costs one more `Router::route` and a few atomic loads, plus a walk of the connections when the write can
//!   reach a socket. It flushes only connections holding fed frames, allocating only when two or more do. For the
//!   counter and command-kind terms the write stays a send, and the flush sends only what fred's router-wide flush
//!   right after it would have sent; that flush then finds nothing left and allocates nothing, where stock's allocated
//!   on every flushed write. For the feed-count term the flush also sends the destination's fed frames and resets its
//!   feed count, so every fed frame leaves as stock's send and router-wide flush would have sent it, but the write
//!   itself is then only fed, and leaves at the next flush point. For a feed at the 8 KiB boundary, which stock follows
//!   with no router-wide flush, it sends other connections' fed frames earlier than stock's next flush point (the next
//!   send, or `before_wait`) would. The router also waits earlier: a silent connection holding fed frames is waited
//!   on, up to one bound, before a write elsewhere, where that write could once fail first and a verdict then drop it
//!   unwaited, so a multi-node outage can recover up to one bound later; the hold stays at most one bound per silent
//!   connection, as an expiry fails later I/O on it at once. A healthy connection whose one outstanding command is fed
//!   now sends it first, so is not judged silent;
//! - **in fred's router-wide flush** (`Connections::flush`, through `live_edit::flush_router_wide`: after a successful
//!   send, after the retry buffer's replay, and at the start of `drain_all`): only connections holding fed frames are
//!   flushed, all at once and each to its own end, and the first error is returned, which the callers log or return
//!   as before. Stock's `try_join_all` stopped at the first error, dropping the other flushes; so now a `drain_all`
//!   whose flush meets a fast error waits for the others, up to one bound, before it returns that error;
//! - **at each receive** ([`on_receive`]): the counter is set to the channel's length plus one (the command just
//!   received, which `process_command` then subtracts), so a drift lasts at most until the next command. Every
//!   sender increments the counter before it sends (`send_to_router`), and the router reads the channel's length
//!   before the counter, so a send in flight at that moment shows as a counter one above the channel; the reset then
//!   drops that sender's increment, so once its command lands the counter reads one below, which the next receive
//!   resets, whichever command it takes. Commands that leave the channel without being counted down show the same
//!   way: fred's `drain_command_rx` cancels the commands still queued when a router ends without decrementing, and
//!   the same channel goes back to the next router, whose first receive finds one above per cancelled command. A
//!   difference of one either way is usually that race, or a single command cancelled when a router ended, and logs
//!   at DEBUG; a larger one (several sends in flight at once, or several commands cancelled) logs one WARN per client
//!   per [`DRIFT_LOG_INTERVAL`] with its size. Either way the reset has already corrected it.
//! - the counter's decrement saturates at zero ([`saturating_decr`]) instead of wrapping.
//!
//! Not covered, each a wait on one connection that can still run while another holds fed frames:
//!
//! - the flush before a `CLIENT UNBLOCK` under `Blocking::Interrupt` (`write_command`, and its copy in
//!   `live_edit::ask::follow`), after a blocking command is written. Only the service's stream readers set
//!   `Blocking::Interrupt`, and they write nothing to the keyspace;
//! - closes that keep fred's stock close, each bounded by its 5 s close timeout. Its flush can wait only on a peer that
//!   half-closed and stopped reading; an errored or closed socket does not hold it. `drop_connection`'s close, as after
//!   a read error or the end of the stream, is one such wait. `disconnect_all` closes a client's connections one after
//!   another, so a frame fed on one late in that walk can wait up to 5 s per such peer; only a router's end reaches it
//!   with fed frames, since a failed cluster-wide command runs `drain_all` first;
//! - a `Moved`'s wait of `cluster_cache_update_delay` (`delay_cluster_sync`; fred's default 0, unset by the service);
//! - the `replicas` feature (the replica loops and routing), which the service does not build: [`before_socket_write`]
//!   skips a command routed to a replica, because routing it twice would advance the replica round-robin.
//!
//! # Pubsub classification
//!
//! Stock fred classifies any reply shaped like a pubsub frame (a three-element array starting `"message"`, a three-
//! or four-element array starting `"subscribe"`, ...) as pubsub on every connection and drops it, so a data reply of
//! that shape is never matched to its command and every later caller on the connection gets the previous caller's
//! reply. Here a connection classifies non-`Push` frames as pubsub only once it has sent a subscription command
//! (`subscribe`, `unsubscribe` and their pattern and shard forms); every other connection matches them to their
//! commands, and counts them. `Push` frames, which are pushes by construction, keep the stock handling.
//!
//! # What a verdict records
//!
//! Each fast or slow verdict, and each expired bounded wait, logs one WARN (target `fred::live_edit::liveness`)
//! starting `Unresponsive connection to <node>`, with these fields:
//!
//! - **the event**: the client id (the line's prefix), the node, the verdict (`fast`, `slow`, or `expired <write|
//!   flush|read>`), the stall, and the age of the newest reply from the same slot range on another connection (from
//!   the connection's node, for a command judged at node level);
//! - **this connection**: the in-flight count (replies owed); the oldest outstanding command's kind, whether its
//!   frame reached the socket (`sent`, with the age since it did) or is still only in fred's write buffer (`fed, not
//!   yet sent`, with the age since fred prepared it); the commands fed since the last completed send or flush
//!   (`unsent`), the write buffer's bytes and fred's feed count; the read buffer's bytes and the time since the last
//!   read progress; whether the connection has sent a subscription command, and how many pubsub-shaped frames it has
//!   received in its life (on one that never subscribed, frames stock fred would have dropped); the longest single
//!   write, flush or read wait on it since the stall began; and the longest time a command sent since the stall began
//!   sat fed before it was sent (the send re-stamps `network_start`, so "sent X ago" alone would hide a frame that
//!   sat unsent);
//! - **this client**: the command-queue counter, the command channel's length when the router last waited, the time
//!   since the router last took a command, and one summary per other connection of the client (node, in flight,
//!   oldest outstanding age, unsent, write-buffer bytes, longest wait). An expired wait runs outside the read loop,
//!   where the other connections are not reachable: its line says so;
//! - **the kernel's view of the socket**, from the `/proc/self/net/tcp` (or `tcp6`) row of the connection's local and
//!   remote addresses: `tx_queue` (bytes the kernel holds unsent or unacknowledged), `rx_queue` (bytes received, not
//!   yet read), the active timer (`retransmit`, `zero-window probe`, `keepalive`) and `retrnsmt` (consecutive
//!   unrecovered retransmit timeouts). Read only at a verdict, with std file I/O; `unavailable` when there is no such
//!   file or no matching row (not Linux, a unix socket).
//!
//! Where the stuck bytes sit names the cause: in fred's write buffer (`unsent` or write-buffer bytes above zero), a
//! client flush problem; a counter that differs from the channel length, a counter leak; a kernel `tx_queue` above
//! zero with a retransmit timer or `retrnsmt` above zero and fred's buffers empty, the network path; both kernel
//! queues empty and no reply, the server or a proxy; an `rx_queue` above zero, the client's read side; a long wait on
//! another connection, the client's shared router; a pubsub-shaped count above zero, the RESP2 shape (handled).

use crate::{
  error::{Error, ErrorKind},
  modules::inner::{ClientInner, CommandReceiver},
  protocol::{
    command::Command,
    connection::{Connection, ConnectionKind},
    types::{ClusterRouting, ProtocolFrame, Server},
  },
  router::{connections::Connections, Router},
  runtime::RefCount,
  utils as client_utils,
};
use futures::{SinkExt, StreamExt};
use redis_protocol::{
  resp2::types::BytesFrame as Resp2Frame,
  resp3::types::{BytesFrame as Resp3Frame, Resp3Frame as _Resp3Frame},
};
use std::{
  collections::HashMap,
  fmt,
  fmt::Write as _,
  future::Future,
  net::SocketAddr,
  sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
    Mutex,
  },
  time::{Duration, Instant},
};

/// The error text of a slow verdict and of every expired bounded wait: stock fred's text, unchanged, so callers that
/// match on it keep working.
pub const UNRESPONSIVE: &str = "Unresponsive connection.";

/// The error text of a fast verdict: the connection was silent while its shard answered other connections.
pub const UNRESPONSIVE_FAST: &str = "Unresponsive connection (fast verdict).";

/// Which unresponsive verdict closed a connection ([`verdict_of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
  /// The fast verdict ([`UNRESPONSIVE_FAST`]): this connection was silent while the same slot range answered others.
  Fast,
  /// The slow verdict or an expired bounded wait ([`UNRESPONSIVE`]): no progress for the slow bound, whatever peers
  /// were doing.
  Slow,
}

impl Verdict {
  /// `fast` or `slow`.
  pub fn as_str(self) -> &'static str {
    match self {
      Verdict::Fast => "fast",
      Verdict::Slow => "slow",
    }
  }
}

/// Which unresponsive verdict `error` reports, if any: an IO error with the [`UNRESPONSIVE_FAST`] text is a fast
/// verdict, one with the [`UNRESPONSIVE`] text a slow verdict or an expired bounded wait (the two share stock fred's
/// text). Every other error is `None`. Classify through this rather than by comparing texts.
pub fn verdict_of(error: &Error) -> Option<Verdict> {
  if *error.kind() != ErrorKind::IO {
    return None;
  }
  match error.details() {
    UNRESPONSIVE_FAST => Some(Verdict::Fast),
    UNRESPONSIVE => Some(Verdict::Slow),
    _ => None,
  }
}

fn verdict_error(verdict: Verdict) -> Error {
  Error::new(ErrorKind::IO, match verdict {
    Verdict::Fast => UNRESPONSIVE_FAST,
    Verdict::Slow => UNRESPONSIVE,
  })
}

/// The number of cluster hash slots.
const SLOTS: usize = 16_384;

/// The minimum spacing of near-miss INFO lines per [`PeerReplies`], unless a near miss is longer than every one
/// logged in the current window (so the longest is never suppressed).
const NEAR_MISS_LOG_INTERVAL: Duration = Duration::from_secs(10);

/// The minimum spacing of command-queue counter drift WARN lines per client ([`on_receive`]).
pub const DRIFT_LOG_INTERVAL: Duration = Duration::from_secs(60);

/// Liveness settings, on `ConnectionConfig::live_edit.liveness`. The default is stock fred's rule set apart from
/// the fixes this module always applies: no fast verdict and no peer registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
  /// The fast verdict's stall bound. `None` disables the fast verdict: only the slow bound
  /// (`UnresponsiveConfig::max_timeout`) applies.
  ///
  /// Default: `None`
  pub fast_timeout:      Option<Duration>,
  /// How long after a connection's stall began a peer's reply from the same slot range must arrive to count as
  /// evidence that the shard is answering.
  ///
  /// Default: 1 sec
  pub peer_evidence_lag: Duration,
  /// The registry of reply times shared by every client of one service. Every reply a connection receives stamps
  /// it; the fast verdict reads it. `None` stamps nothing and disables the fast verdict.
  ///
  /// Default: `None`
  pub peer_replies:      Option<PeerReplies>,
}

impl Default for Settings {
  fn default() -> Self {
    Settings {
      fast_timeout:      None,
      peer_evidence_lag: Duration::from_secs(1),
      peer_replies:      None,
    }
  }
}

/// When each cluster hash slot, and each node for commands judged at node level, last answered a command on any
/// connection that shares this registry. Cloning shares the registry; it is dropped with its last clone.
///
/// Stamping a reply is a read of the client's cluster routing (to place the slot on the connection's node) and one
/// relaxed atomic store; a stalled connection reads its slot range only when it judges a stall.
#[derive(Clone)]
pub struct PeerReplies {
  shared: Arc<Registry>,
}

struct Registry {
  epoch:       Instant,
  /// Per slot: nanoseconds since `epoch`, plus one, of the newest reply; `0` = none yet.
  slots:       Box<[AtomicU64]>,
  /// Per node, for replies to commands judged at node level, encoded as `slots`.
  nodes:       Mutex<HashMap<Server, Arc<AtomicU64>>>,
  near_misses: Mutex<NearMisses>,
}

#[derive(Default)]
struct NearMisses {
  window_start: Option<Instant>,
  longest:      Duration,
  suppressed:   u64,
}

impl PeerReplies {
  /// A new, empty registry.
  pub fn new() -> Self {
    PeerReplies {
      shared: Arc::new(Registry {
        epoch:       Instant::now(),
        slots:       (0 .. SLOTS).map(|_| AtomicU64::new(0)).collect(),
        nodes:       Mutex::new(HashMap::new()),
        near_misses: Mutex::new(NearMisses::default()),
      }),
    }
  }

  /// Whether `self` and `other` are the same registry.
  pub fn same_as(&self, other: &PeerReplies) -> bool {
    Arc::ptr_eq(&self.shared, &other.shared)
  }

  fn encode(&self, at: Instant) -> u64 {
    (at.saturating_duration_since(self.shared.epoch).as_nanos() as u64).saturating_add(1)
  }

  fn decode(&self, value: u64) -> Option<Instant> {
    value
      .checked_sub(1)
      .map(|nanos| self.shared.epoch + Duration::from_nanos(nanos))
  }

  fn node(&self, server: &Server) -> Arc<AtomicU64> {
    let mut nodes = self.shared.nodes.lock().unwrap_or_else(|e| e.into_inner());
    nodes.entry(server.clone()).or_default().clone()
  }

  fn stamp(&self, cell: &AtomicU64, at: Instant) {
    cell.store(self.encode(at), Ordering::Relaxed);
  }

  /// The newest reply time over the inclusive slot `ranges` and, when given, `node`.
  fn newest(&self, ranges: &[(u16, u16)], node: Option<&AtomicU64>) -> Option<Instant> {
    let mut newest = node.map(|n| n.load(Ordering::Relaxed)).unwrap_or(0);
    for &(start, end) in ranges {
      let (start, end) = (start as usize, (end as usize).min(SLOTS - 1));
      if start > end {
        continue;
      }
      for cell in self.shared.slots[start ..= end].iter() {
        newest = newest.max(cell.load(Ordering::Relaxed));
      }
    }
    self.decode(newest)
  }
}

impl Default for PeerReplies {
  fn default() -> Self {
    PeerReplies::new()
  }
}

impl fmt::Debug for PeerReplies {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("PeerReplies").finish_non_exhaustive()
  }
}

impl PartialEq for PeerReplies {
  fn eq(&self, other: &Self) -> bool {
    self.same_as(other)
  }
}

impl Eq for PeerReplies {}

/// Per-client liveness state, on `ClientInner::live_edit`, shared with every connection of the client. Written only
/// by the client's router task; read when a verdict is logged.
#[derive(Clone, Debug)]
pub struct ClientState {
  shared: Arc<ClientShared>,
}

#[derive(Debug)]
struct ClientShared {
  epoch:        Instant,
  /// When the router last took a command off its channel: nanoseconds since `epoch`, plus one; `0` = never.
  last_command: AtomicU64,
  /// The command channel's length when the router last waited ([`before_wait`]).
  queued:       AtomicUsize,
  drift:        Mutex<DriftLog>,
}

#[derive(Debug, Default)]
struct DriftLog {
  logged:     Option<Instant>,
  suppressed: u64,
}

impl Default for ClientState {
  fn default() -> Self {
    ClientState {
      shared: Arc::new(ClientShared {
        epoch:        Instant::now(),
        last_command: AtomicU64::new(0),
        queued:       AtomicUsize::new(0),
        drift:        Mutex::new(DriftLog::default()),
      }),
    }
  }
}

impl ClientState {
  fn mark_command(&self, now: Instant) {
    let nanos = now.saturating_duration_since(self.shared.epoch).as_nanos() as u64;
    self.shared.last_command.store(nanos.saturating_add(1), Ordering::Relaxed);
  }

  fn last_command(&self) -> Option<Instant> {
    self
      .shared
      .last_command
      .load(Ordering::Relaxed)
      .checked_sub(1)
      .map(|nanos| self.shared.epoch + Duration::from_nanos(nanos))
  }
}

/// Per-connection liveness state. Created with the connection (`crate::live_edit::pipelined`), dropped with it; a
/// reconnect starts fresh.
#[derive(Debug, Default)]
pub struct ConnectionState {
  /// The bound on every wait on this connection's I/O outside the read loop: the client's
  /// `UnresponsiveConfig::max_timeout` when the connection was created.
  wait_bound:       Option<Duration>,
  /// Whether this connection has sent a subscription command, so pubsub-shaped frames on it are pubsub.
  pub(crate) subscribed: bool,
  /// How many pubsub-shaped frames this connection has received in its life (on one that never subscribed: frames it
  /// matched to their commands, which stock fred would have dropped). Shared with the frame classification, which
  /// runs where only the router is at hand.
  pubsub_shaped:    Arc<AtomicU64>,
  /// The last read progress: a frame decoded or the read buffer grew.
  last_progress:    Option<Instant>,
  /// The read buffer's length at the last observation.
  read_len:         usize,
  /// Tracked commands written since the last completed send or flush.
  unsent:           usize,
  /// Which bounded wait expired, once one has: the connection is failed, every later write, feed, flush or read on it
  /// fails at once, and its close does not flush ([`close`]). Never cleared.
  expired:          Option<&'static str>,
  /// The read-loop verdict that is closing this connection: set where the read loop reaches a fast or slow verdict
  /// ([`verdict`]), so the close that follows does not flush ([`close`]). Never cleared; dropped with the connection.
  /// Unlike `expired` it fails no later I/O: the router closes the connection in the step that reached the verdict
  /// ([`defer`]).
  closed_by:        Option<Verdict>,
  /// The current stall, while the connection has an outstanding command.
  episode:          Option<Episode>,
  /// When the peer evidence for the current stall was last read.
  evidence_checked: Option<Instant>,
  /// This connection's node cell in the peer registry, for replies to commands judged at node level.
  node_stamp:       Option<Arc<AtomicU64>>,
  /// The longest single write, flush or read wait on this connection that ended since its current stall began.
  longest_wait:     Option<Wait>,
  /// The longest time a tracked command sat fed (prepared, in fred's write buffer) before a send or flush put it on
  /// the socket, among the commands sent since the current stall began. [`on_sent`] re-stamps a command's
  /// `network_start` when it is sent, so this is the only record of that delay.
  longest_fed:      Option<Wait>,
  /// When the connection's last write, feed or flush completed: a fed command's frame cannot have waited on the
  /// socket for longer than since then ([`stall_start`]).
  last_write:       Option<Instant>,
  /// A verdict reached in the read loop, logged once every connection of the client has been polled.
  pending:          Option<Pending>,
  /// The owning client, set when the connection is created, for a log line written where only the connection is at
  /// hand (an expired wait). The router task owns both, so this makes no cycle.
  inner:            Option<ClientHandle>,
}

/// The owning client of a connection ([`ConnectionState`]); `Debug` prints its id.
struct ClientHandle(RefCount<ClientInner>);

impl fmt::Debug for ClientHandle {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(&self.0.id)
  }
}

#[derive(Debug, Clone, Copy)]
struct Episode {
  /// When the stall began.
  since:    Instant,
  /// Whether a peer answered the same slot range at least the evidence lag after `since`.
  evidence: bool,
}

/// One completed wait on a connection's I/O.
#[derive(Debug, Clone, Copy)]
struct Wait {
  what:  &'static str,
  took:  Duration,
  ended: Instant,
}

#[derive(Debug, Clone, Copy)]
struct Pending {
  verdict: Verdict,
  stall:   Duration,
  since:   Instant,
}

/// How a frame read from one connection is classified: its `subscribed` flag, and the connection's pubsub-shaped
/// frame count (absent when the connection is gone).
#[derive(Debug, Clone)]
pub(crate) struct Classify {
  subscribed: bool,
  shaped:     Option<Arc<AtomicU64>>,
}

/// Initialize a new pipelined connection's state from its client's settings.
pub(crate) fn attach(inner: &RefCount<ClientInner>, conn: &mut Connection) {
  conn.live_edit.liveness.wait_bound = inner.connection.unresponsive.max_timeout;
  conn.live_edit.liveness.inner = Some(ClientHandle(inner.clone()));
}

macro_rules! per_transport {
  ($transport:expr, $framed:ident => $body:expr) => {
    match $transport {
      ConnectionKind::Tcp(ref t) => {
        let $framed = t.get_ref();
        $body
      },
      #[cfg(feature = "unix-sockets")]
      ConnectionKind::Unix(ref t) => {
        let $framed = t.get_ref();
        $body
      },
      #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
      ConnectionKind::Rustls(ref t) => {
        let $framed = t.get_ref();
        $body
      },
      #[cfg(feature = "enable-native-tls")]
      ConnectionKind::NativeTls(ref t) => {
        let $framed = t.get_ref();
        $body
      },
    }
  };
}

fn read_buffer_len(transport: &ConnectionKind) -> usize {
  per_transport!(transport, framed => framed.read_buffer().len())
}

fn write_buffer_len(transport: &ConnectionKind) -> usize {
  per_transport!(transport, framed => framed.write_buffer().len())
}

/// The owning client's id (`fred-...`), which the codec carries.
fn client_name(transport: &ConnectionKind) -> &str {
  per_transport!(transport, framed => &*framed.codec().name)
}

/// The TCP socket's local and remote addresses, when the transport is TCP (plain or rustls).
fn socket_addrs(transport: &ConnectionKind) -> Option<(SocketAddr, SocketAddr)> {
  let tcp = match transport {
    ConnectionKind::Tcp(ref t) => t.get_ref().get_ref(),
    #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
    ConnectionKind::Rustls(ref t) => t.get_ref().get_ref().get_ref().0,
    #[allow(unreachable_patterns)]
    _ => return None,
  };
  Some((tcp.local_addr().ok()?, tcp.peer_addr().ok()?))
}

fn on_progress(conn: &mut Connection, now: Instant) {
  let read_len = read_buffer_len(&conn.transport);
  let state = &mut conn.live_edit.liveness;
  state.last_progress = Some(now);
  state.read_len = read_len;
}

/// When `conn` last made read progress (a frame taken off it, or its read buffer grew), if it has.
pub(crate) fn last_progress(conn: &Connection) -> Option<Instant> {
  conn.live_edit.liveness.last_progress
}

/// The read loop took a frame (or the end of the stream) off the connection.
pub(crate) fn on_frame(conn: &mut Connection, now: Instant) {
  on_progress(conn, now);
}

/// Count a tracked write (`write`'s `check_unresponsive`); it is re-stamped when its frame is flushed.
pub(crate) fn count_unsent(conn: &mut Connection) {
  conn.live_edit.liveness.unsent += 1;
}

/// A send or flush completed: every tracked command written since the previous one is now on the socket, so its
/// wait starts now.
///
/// Before re-stamping, each command's fed-to-sent delay (from its prepare-time `network_start`) is kept when it is the
/// longest since the current stall began ([`ConnectionState`]'s `longest_fed`).
fn on_sent(conn: &mut Connection, now: Instant) {
  let mut left = std::mem::take(&mut conn.live_edit.liveness.unsent);
  let mut longest = Duration::ZERO;
  for command in conn.buffer.iter_mut().rev() {
    if left == 0 {
      break;
    }
    if command.kind.is_pubsub() {
      continue;
    }
    if let Some(prepared) = command.network_start {
      longest = longest.max(now.saturating_duration_since(prepared));
    }
    command.network_start = Some(now);
    left -= 1;
  }
  let state = &mut conn.live_edit.liveness;
  let since = state.episode.map(|episode| episode.since);
  let fed = Wait {
    what:  "fed",
    took:  longest,
    ended: now,
  };
  state.longest_fed = match state.longest_fed {
    Some(kept) if since.map_or(true, |since| kept.ended >= since) && kept.took >= fed.took => Some(kept),
    _ => Some(fed),
  };
}

/// Record one completed wait on the connection's I/O, keeping the longest that ended since the current stall began.
fn record_wait(conn: &mut Connection, what: &'static str, started: Instant) {
  let ended = Instant::now();
  let wait = Wait {
    what,
    took: ended.saturating_duration_since(started),
    ended,
  };
  let state = &mut conn.live_edit.liveness;
  if what != "read" {
    state.last_write = Some(ended);
  }
  let since = state.episode.map(|episode| episode.since);
  state.longest_wait = match state.longest_wait {
    Some(kept) if since.map_or(true, |since| kept.ended >= since) && kept.took >= wait.took => Some(kept),
    _ => Some(wait),
  };
}

/// Mark the connection failed after a bounded wait expired, log it, and return the error for the caller's path.
fn expire(conn: &mut Connection, what: &'static str, started: Instant) -> Error {
  record_wait(conn, what, started);
  conn.live_edit.liveness.expired = Some(what);
  if log::log_enabled!(log::Level::Warn) {
    let now = Instant::now();
    let bound = conn.live_edit.liveness.wait_bound.unwrap_or_default();
    let newest_peer = match conn.live_edit.liveness.inner.as_ref() {
      Some(ClientHandle(inner)) => describe_newest_peer(inner, conn, now),
      None => "unknown".to_owned(),
    };
    let mut line = format!(
      "{}: Unresponsive connection to {} (expired {}): a {} on it made no progress for {:?}; failing the \
       connection. Newest reply from the same slot range on another connection: {}. ",
      client_name(&conn.transport),
      conn.server,
      what,
      what,
      bound,
      newest_peer,
    );
    describe_connection(&mut line, conn, now, Some(bound));
    describe_client(&mut line, conn, now);
    line.push_str("Other connections: not captured (the wait ran outside the read loop). ");
    describe_kernel(&mut line, conn);
    log::warn!("{}", line);
  }
  verdict_error(Verdict::Slow)
}

async fn within<T, F>(bound: Option<Duration>, ft: F) -> Option<T>
where
  F: Future<Output = T>,
{
  match bound {
    Some(bound) => tokio::time::timeout(bound, ft).await.ok(),
    None => Some(ft.await),
  }
}

/// `SinkExt::send` (write and flush), bounded. Fails at once, without writing, on a connection whose bounded wait
/// already expired (see the module's *Bounded waits*).
pub(crate) async fn send(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
  if conn.live_edit.liveness.expired.is_some() {
    return Err(verdict_error(Verdict::Slow));
  }
  let bound = conn.live_edit.liveness.wait_bound;
  let started = Instant::now();
  match within(bound, conn.transport.send(frame)).await {
    Some(Ok(())) => {
      record_wait(conn, "write", started);
      on_sent(conn, Instant::now());
      Ok(())
    },
    Some(Err(e)) => {
      record_wait(conn, "write", started);
      Err(e)
    },
    None => Err(expire(conn, "write", started)),
  }
}

/// `SinkExt::feed` (buffer, flushing only past the sink's backpressure boundary), bounded. Fails at once, without
/// buffering, on a connection whose bounded wait already expired.
pub(crate) async fn feed(conn: &mut Connection, frame: ProtocolFrame) -> Result<(), Error> {
  if conn.live_edit.liveness.expired.is_some() {
    return Err(verdict_error(Verdict::Slow));
  }
  let bound = conn.live_edit.liveness.wait_bound;
  let started = Instant::now();
  match within(bound, conn.transport.feed(frame)).await {
    Some(result) => {
      record_wait(conn, "write", started);
      result
    },
    None => Err(expire(conn, "write", started)),
  }
}

/// `SinkExt::flush`, bounded. Fails at once, without flushing, on a connection whose bounded wait already expired,
/// even when it has nothing to flush.
pub(crate) async fn flush(conn: &mut Connection) -> Result<(), Error> {
  if conn.live_edit.liveness.expired.is_some() {
    return Err(verdict_error(Verdict::Slow));
  }
  let bound = conn.live_edit.liveness.wait_bound;
  let started = Instant::now();
  match within(bound, conn.transport.flush()).await {
    Some(Ok(())) => {
      record_wait(conn, "flush", started);
      on_sent(conn, Instant::now());
      Ok(())
    },
    Some(Err(e)) => {
      record_wait(conn, "flush", started);
      Err(e)
    },
    None => Err(expire(conn, "flush", started)),
  }
}

/// `StreamExt::next` (one frame), bounded; the outer error is an expired bound, this one's or, failing at once
/// without reading, an earlier one on the same connection.
pub(crate) async fn next(conn: &mut Connection) -> Result<Option<Result<ProtocolFrame, Error>>, Error> {
  if conn.live_edit.liveness.expired.is_some() {
    return Err(verdict_error(Verdict::Slow));
  }
  let bound = conn.live_edit.liveness.wait_bound;
  let started = Instant::now();
  match within(bound, conn.transport.next()).await {
    Some(item) => {
      record_wait(conn, "read", started);
      on_progress(conn, Instant::now());
      Ok(item)
    },
    None => Err(expire(conn, "read", started)),
  }
}

/// `Connection::close`'s transport close (`SinkExt::close`: flush fred's write buffer, then shut the transport down,
/// under TLS after queueing `close_notify`), which `Connection::close` bounds by `CONNECTION_CLOSE_TIMEOUT_MS`.
///
/// On a connection judged dead — a bounded wait expired (`expired`), or the read loop reached a verdict on it
/// (`closed_by`) — returns at once instead: no flush, no `close_notify`, no shutdown, no log and no metric (see the
/// module's *Bounded waits*). The peer is not reading or not acknowledging, so the close could only wait; and were it
/// only slow, flushing would hand it frames that fred re-sends or fails anyway. The caller still drains and returns
/// the in-flight commands, and dropping the connection afterwards closes the socket without blocking.
pub(crate) async fn close(conn: &mut Connection) -> Result<(), Error> {
  let state = &conn.live_edit.liveness;
  if state.expired.is_some() || state.closed_by.is_some() {
    return Ok(());
  }
  conn.transport.close().await
}

/// Whether `conn` holds frames fred fed but has not flushed.
fn has_unflushed(conn: &Connection) -> bool {
  client_utils::read_atomic(&conn.counters.feed_count) > 0 || write_buffer_len(&conn.transport) > 0
}

/// Every connection of the router, primaries and (with the `replicas` feature) replicas: the primaries' (a
/// centralized or sentinel client's one connection, or the cluster map's), then the replicas'. An iterator over the
/// router's own collections, so walking it allocates nothing; the connection buffer sweep walks it twice on every
/// router pass. Two walks with no connection added or removed between them visit the connections in the same order.
pub(crate) fn connections_mut(router: &mut Router) -> impl Iterator<Item = &mut Connection> {
  let primaries = match router.connections {
    Connections::Centralized { ref mut connection } | Connections::Sentinel { ref mut connection } => {
      Primaries::One(connection.iter_mut())
    },
    Connections::Clustered {
      ref mut connections, ..
    } => Primaries::Map(connections.values_mut()),
  };
  #[cfg(feature = "replicas")]
  let primaries = primaries.chain(router.replicas.connections.values_mut());
  primaries
}

/// The primaries' connections, for [`connections_mut`]: the one connection of a centralized or sentinel client, or a
/// cluster client's connection map.
enum Primaries<'a> {
  One(std::option::IterMut<'a, Connection>),
  Map(std::collections::hash_map::ValuesMut<'a, Server, Connection>),
}

impl<'a> Iterator for Primaries<'a> {
  type Item = &'a mut Connection;

  fn next(&mut self) -> Option<Self::Item> {
    match self {
      Primaries::One(connection) => connection.next(),
      Primaries::Map(connections) => connections.next(),
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    match self {
      Primaries::One(connection) => connection.size_hint(),
      Primaries::Map(connections) => connections.size_hint(),
    }
  }
}

/// The top of the router's `read_or_write`: record the command channel's length, and when it is empty — no queued
/// command will flush what an earlier write fed — flush every connection holding fed but unflushed frames, all of
/// them at once ([`flush_fed_concurrently`]). The channel is checked, not the counter: fred's own rule, "flush when no
/// queued command follows", applied by construction.
pub(crate) async fn before_wait(inner: &RefCount<ClientInner>, router: &mut Router, rx: &CommandReceiver) {
  let queued = rx.len();
  inner.live_edit.liveness.shared.queued.store(queued, Ordering::Relaxed);
  if queued > 0 {
    return;
  }
  let _ = flush_fed_concurrently(connections_mut(router), FlushSite::Wait(inner)).await;
}

/// Before the router writes `command` to the one connection it routes to, on a clustered client
/// (`live_edit::before_socket_write`, the top of the single-connection branch of the router's `write_command`): when
/// that write can reach the connection's socket ([`write_reaches_socket`]) and another connection holds fed but
/// unflushed frames, flush every connection holding them, the destination included, all at once
/// ([`flush_fed_concurrently`]). The write can then wait on its own connection's socket for up to a bound, and a frame
/// fed elsewhere would otherwise wait through that, and through each later such write, before any flush sent it. When
/// only the destination holds fed frames nothing is flushed here, so its own write batches them as stock does.
///
/// The check is one more `Router::route` (a slot lookup and a map probe, as the write's own), a few atomic loads and,
/// when the write can reach the socket, one walk of the client's connections ([`has_unflushed`] on each), and allocates
/// nothing; its flush adds an allocation only when two or more connections need one. With the `replicas` feature (not
/// built by the service) a command routed to a replica is not checked: routing it twice would advance the replica
/// round-robin, so the check could name another replica than the write's.
pub(crate) async fn before_socket_write(inner: &RefCount<ClientInner>, router: &mut Router, command: &Command) {
  // Test-only: count this check and time it (`retention::test_support::write_check_stats`); compiled out otherwise.
  #[cfg(feature = "live-edit-retention-report")]
  let mut check = super::retention::test_support::WriteCheck::begin(inner);
  #[cfg(feature = "replicas")]
  if command.use_replica {
    return;
  }
  let target = match router.route(command) {
    Some(conn) if write_reaches_socket(inner, conn, command) => address(conn),
    _ => return,
  };
  if !others_fed(router, target) {
    return;
  }
  #[cfg(feature = "live-edit-retention-report")]
  check.flushing();
  flush_before_write(inner, router, target).await;
}

/// Before one of the live-edit patch's own writes to `server` on a clustered client (`live_edit::before_write_to`:
/// [`super::ask::follow`]'s `ASKING` and command, [`super::abandon::after_pipeline`]'s `DISCARD`): when another
/// connection holds fed but unflushed frames, flush every connection holding them, the destination included, all at
/// once, as [`before_socket_write`] does. These paths are rare, so the write predicate is not applied: any write there
/// is treated as one that can reach the socket. Nothing is done when the client has no connection to `server`.
pub(crate) async fn before_write_to(inner: &RefCount<ClientInner>, router: &mut Router, server: &Server) {
  // Test-only: count this check and time it, as in `before_socket_write`.
  #[cfg(feature = "live-edit-retention-report")]
  let mut check = super::retention::test_support::WriteCheck::begin(inner);
  let target = match router.get_connection_mut(server) {
    Some(conn) => address(conn),
    None => return,
  };
  if !others_fed(router, target) {
    return;
  }
  #[cfg(feature = "live-edit-retention-report")]
  check.flushing();
  flush_before_write(inner, router, target).await;
}

/// Whether writing `command` to `conn` can reach `conn`'s socket: fred's own flush rule for the write
/// (`prepare_command`, `src/router/utils.rs:56`), or a feed that must first flush what `conn` holds because its write
/// buffer is at the backpressure boundary. A feed below the boundary only encodes the frame into the write buffer and
/// cannot block, so it is not such a write: below the boundary fred's batching across connections is kept. Every term
/// reads state only the router changes, apart from the command-queue counter, which senders can only raise while the
/// router runs (a sender lowers it again only when its send finds the channel closed): so whenever fred's write, or
/// the flush after it, touches the socket, this was true. Each term is its own function, so each can be judged, and
/// reverted, alone.
fn write_reaches_socket(inner: &RefCount<ClientInner>, conn: &Connection, command: &Command) -> bool {
  queue_empty(conn) || feed_count_full(inner, conn) || flushes_by_kind(command) || at_backpressure(conn)
}

/// The counter-at-zero term (`Counters::should_send`'s "no queued command follows"): the write is a send. It also
/// covers `write_command`'s flush of the destination after a write that did not flush
/// (`src/router/commands.rs:164-166`), which runs only when the counter is zero: after a command the router skipped
/// because its caller gave up, or could not encode.
fn queue_empty(conn: &Connection) -> bool {
  client_utils::read_atomic(&conn.counters.cmd_buffer_len) == 0
}

/// The feed-count term (`Counters::should_send`'s other half): the destination has fed more than `max_feed_count`
/// frames (fred's default 200, unset by the service): a send, unless the flush before it resets the count first.
fn feed_count_full(inner: &RefCount<ClientInner>, conn: &Connection) -> bool {
  client_utils::read_atomic(&conn.counters.feed_count) as u64 > inner.max_feed_count()
}

/// The command-kind term (`CommandKind::should_flush`): the kinds fred always sends, every custom command and script
/// among them. fred's third term, a command to every cluster node, never reaches a single-connection write.
fn flushes_by_kind(command: &Command) -> bool {
  command.kind.should_flush()
}

/// The backpressure term: the destination's write buffer is at tokio-util's backpressure boundary, so a feed first
/// flushes what it holds (`FramedImpl::poll_ready`, tokio-util 0.7.19 `codec/framed_impl.rs:262-268`).
fn at_backpressure(conn: &Connection) -> bool {
  write_buffer_len(&conn.transport) >= super::retention::BACKPRESSURE_BYTES
}

/// A connection's address, to tell the write's destination apart from the other connections of one walk. A plain
/// integer, unlike a pointer, so it can be held across an `await`; it is only compared, never turned back into a
/// reference.
fn address(conn: &Connection) -> usize {
  conn as *const Connection as usize
}

/// Whether a connection other than the one at `target` ([`address`]) holds fed but unflushed frames. Walking the
/// connections allocates nothing.
fn others_fed(router: &mut Router, target: usize) -> bool {
  connections_mut(router).any(|conn| address(conn) != target && has_unflushed(conn))
}

/// The flush of [`before_socket_write`] and [`before_write_to`]: every connection holding fed but unflushed frames,
/// the destination at `target` included, all at once. The destination's name is taken first, for the DEBUG line a
/// failed flush logs.
async fn flush_before_write(inner: &RefCount<ClientInner>, router: &mut Router, target: usize) {
  let to = match connections_mut(router).find(|conn| address(conn) == target) {
    Some(conn) => conn.server.clone(),
    None => return,
  };
  let _ = flush_fed_concurrently(connections_mut(router), FlushSite::Write { inner, to: &to }).await;
}

/// Where [`flush_fed_concurrently`] runs: it names the DEBUG line a failed flush logs.
#[derive(Clone, Copy)]
pub(crate) enum FlushSite<'a> {
  /// [`before_wait`]: `Failed to flush fed frames to <node> before waiting`.
  Wait(&'a RefCount<ClientInner>),
  /// `live_edit::before_inline_connect`: `Failed to flush fed frames to <node> before an inline connect`.
  InlineConnect(&'a RefCount<ClientInner>),
  /// [`before_socket_write`] and [`before_write_to`], before a write to `to`: `Failed to flush fed frames to <node>
  /// before a write to <to>`.
  Write {
    inner: &'a RefCount<ClientInner>,
    to:    &'a Server,
  },
  /// fred's router-wide flush (`Connections::flush`, through `live_edit::flush_router_wide`): no line, as in stock;
  /// its callers log or return the error.
  RouterWide,
}

/// Flush every connection of `connections` holding fed but unflushed frames, all of them at once, each through the
/// bounded flush and to its own end, and return the first error in the walk's order, if any. Every flush of a
/// clustered client's connections taken together runs here: [`before_wait`]'s, the inline-connect flush
/// (`live_edit::before_inline_connect`), the flush before a write ([`before_socket_write`], [`before_write_to`]) and
/// fred's router-wide flush (`Connections::flush`, after a send, after the retry buffer's replay and in `drain_all`).
///
/// Run concurrently, a flush holds the router for at most one bound however many connections stall in it; one after
/// another, k silent connections would hold it for k bounds. Each connection's flush runs to its own end, so a fast
/// error on one never cuts another's short, as stock's `try_join_all` would. Only connections holding fed frames are
/// flushed: with none the walk is all, and with one its flush runs inline, so neither adds an allocation to what the
/// flush itself does. Two or more are joined (`join_all`), which allocates the join. A failed flush logs at DEBUG, as
/// `site` names, and is left to the connection's read side, or returned to the router-wide flush's caller; an expired
/// flush has marked the connection failed, and a connection already marked fails at once, with no I/O.
pub(crate) async fn flush_fed_concurrently<'c, I>(connections: I, site: FlushSite<'_>) -> Result<(), Error>
where
  I: Iterator<Item = &'c mut Connection>,
{
  let mut fed = connections.filter(|conn| has_unflushed(conn));
  let first = match fed.next() {
    Some(conn) => conn,
    None => return Ok(()),
  };
  let second = match fed.next() {
    Some(conn) => conn,
    None => return flush_one(first, site).await,
  };
  let flushes = [first, second].into_iter().chain(fed).map(|conn| flush_one(conn, site));
  let results = futures::future::join_all(flushes).await;
  results.into_iter().find(Result::is_err).unwrap_or(Ok(()))
}

/// One connection's flush for [`flush_fed_concurrently`], logging a failure as `site` names.
async fn flush_one(conn: &mut Connection, site: FlushSite<'_>) -> Result<(), Error> {
  let result = conn.flush().await;
  if let Err(ref e) = result {
    match site {
      FlushSite::Wait(inner) => {
        _debug!(inner, "Failed to flush fed frames to {} before waiting: {:?}", conn.server, e);
      },
      FlushSite::InlineConnect(inner) => {
        _debug!(inner, "Failed to flush fed frames to {} before an inline connect: {:?}", conn.server, e);
      },
      FlushSite::Write { inner, to } => {
        _debug!(inner, "Failed to flush fed frames to {} before a write to {}: {:?}", conn.server, to, e);
      },
      FlushSite::RouterWide => {},
    }
  }
  result
}

/// The router took a command off its channel: set the command-queue counter to the channel's length plus one (the
/// command just taken, which `process_command` subtracts next), and log a drift found doing so. The reset is a
/// compare-and-set against the value read, so a sender that increments between that read and the reset is never
/// overwritten; one that incremented before the read and has not sent yet has its increment dropped, and once its
/// command lands the next receive, whichever command it takes, finds the counter one below and resets it again.
///
/// A difference of one either way is usually that race, or a single command cancelled when a router ended (see the
/// module's *Never waiting on unsent frames*), and logs at DEBUG. A larger one logs a WARN, rate-limited per client to
/// one per [`DRIFT_LOG_INTERVAL`].
pub(crate) fn on_receive(inner: &RefCount<ClientInner>, rx: &CommandReceiver) {
  inner.live_edit.liveness.mark_command(Instant::now());
  let counter = &inner.counters.cmd_buffer_len;
  let expected = rx.len().saturating_add(1);
  let observed = client_utils::read_atomic(counter);
  if observed == expected
    || counter
      .compare_exchange(observed, expected, Ordering::AcqRel, Ordering::Acquire)
      .is_err()
  {
    return;
  }
  let drift = observed as i128 - expected as i128;
  if drift.abs() == 1 {
    _debug!(
      inner,
      "Command-queue counter off by one: the counter read {} where the command channel holds {} commands plus the \
       one just taken ({:+}), usually a send in flight as it was read or that send's command landing after a reset, \
       or a single command cancelled when a router ended; reset it to {}.",
      observed,
      expected - 1,
      drift,
      expected
    );
    return;
  }
  let now = Instant::now();
  let mut log = inner.live_edit.liveness.shared.drift.lock().unwrap_or_else(|e| e.into_inner());
  let due = log
    .logged
    .map_or(true, |logged| now.saturating_duration_since(logged) >= DRIFT_LOG_INTERVAL);
  if due {
    _warn!(
      inner,
      "Command-queue counter drift: the counter read {} where the command channel holds {} commands plus the one \
       just taken ({:+}); reset it to {}. More than one send in flight as it was read, or commands that left the \
       channel without being counted down, such as those cancelled when a router ended ({} further drifts larger \
       than one not logged since the previous line).",
      observed,
      expected - 1,
      drift,
      expected,
      log.suppressed
    );
    log.logged = Some(now);
    log.suppressed = 0;
  } else {
    log.suppressed += 1;
  }
}

/// Decrement the command-queue counter, saturating at zero instead of wrapping as stock fred's `decr_atomic` does.
/// Returns the new value.
pub(crate) fn saturating_decr(counter: &AtomicUsize) -> usize {
  match counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| Some(value.saturating_sub(1))) {
    Ok(previous) | Err(previous) => previous.saturating_sub(1),
  }
}

/// Record a written command: a subscription command marks the connection a pubsub connection. Returns
/// `cmd.has_no_responses()`, the condition stock `push_command` tests.
pub(crate) fn on_push(conn: &mut Connection, cmd: &Command) -> bool {
  if cmd.kind.is_pubsub() {
    conn.live_edit.liveness.subscribed = true;
  }
  cmd.has_no_responses()
}

/// How frames read from `conn` are classified.
pub(crate) fn classify(conn: &Connection) -> Classify {
  Classify {
    subscribed: conn.live_edit.liveness.subscribed,
    shaped:     Some(Arc::clone(&conn.live_edit.liveness.pubsub_shaped)),
  }
}

/// How frames read from the connection to `server` are classified. A missing connection keeps the stock
/// classification.
pub(crate) fn subscribed(router: &mut Router, server: &Server) -> Classify {
  router.get_connection_mut(server).map(|conn| classify(conn)).unwrap_or(Classify {
    subscribed: true,
    shaped:     None,
  })
}

/// Whether `frame` may be classified as a pubsub frame under `classify`.
pub(crate) fn pubsub_classified(classify: &Classify, frame: &Resp3Frame) -> bool {
  classify.subscribed || matches!(frame, Resp3Frame::Push { .. })
}

/// [`pubsub_classified`], called once per frame (`subscription_response` is whether the frame is shaped like a
/// subscription response): a pubsub-shaped frame it does not classify as pubsub is counted on the connection.
pub(crate) fn classify_frame(classify: &Classify, frame: &Resp3Frame, subscription_response: bool) -> bool {
  let classified = pubsub_classified(classify, frame);
  if !classified
    && (subscription_response
      || frame.is_normal_pubsub_message()
      || frame.is_pattern_pubsub_message()
      || frame.is_shard_pubsub_message())
  {
    if let Some(shaped) = classify.shaped.as_ref() {
      shaped.fetch_add(1, Ordering::Relaxed);
    }
  }
  classified
}

/// Convert a decoded frame to RESP3. On a connection that never subscribed, a RESP2 array stays an array at every
/// depth: stock conversion turns any pubsub-shaped RESP2 array, nested ones included, into a `Push` frame.
pub(crate) fn into_resp3(conn: &Connection, frame: ProtocolFrame) -> Resp3Frame {
  match frame {
    ProtocolFrame::Resp2(frame) if !conn.live_edit.liveness.subscribed => resp2_as_data(frame),
    frame => frame.into_resp3(),
  }
}

fn resp2_as_data(frame: Resp2Frame) -> Resp3Frame {
  match frame {
    Resp2Frame::SimpleString(data) => Resp3Frame::SimpleString { data, attributes: None },
    Resp2Frame::Error(data) => Resp3Frame::SimpleError { data, attributes: None },
    Resp2Frame::Integer(data) => Resp3Frame::Number { data, attributes: None },
    Resp2Frame::BulkString(data) => Resp3Frame::BlobString { data, attributes: None },
    Resp2Frame::Array(frames) => Resp3Frame::Array {
      data:       frames.into_iter().map(resp2_as_data).collect(),
      attributes: None,
    },
    Resp2Frame::Null => Resp3Frame::Null,
  }
}

/// The slot a command on the connection to `server` is stamped and judged by: its hash slot when the cluster routing
/// gives that slot to `server`, `None` when the command is judged at node level.
///
/// fred hashes a command's first string argument whatever it names (`ClusterHash::FirstKey`), and a hash slot
/// says where a command was sent only when the router sent it by that slot. A command with nothing to hash (`PING`,
/// `TIME`) has no slot. A command sent to a named node (`Options::cluster_node`: `with_cluster_node`, a `SCAN` page
/// continuing on its node) can hash to a slot another node owns, as can one the router sent to an arbitrary
/// connection because it holds none to the slot's owner. Judging either by that slot's range would count another
/// node's replies as evidence for this one, so both are judged by their connection's node.
fn judged_slot(state: &ClusterRouting, server: &Server, command: &Command) -> Option<u16> {
  command
    .cluster_hash()
    .filter(|slot| state.get_server(*slot) == Some(server))
}

/// A reply was matched to `command` (popped from `conn`'s in-flight queue): stamp the peer registry with its slot,
/// or with the connection's node for a command judged at node level ([`judged_slot`]). Returns `command`.
pub(crate) fn on_reply(inner: &RefCount<ClientInner>, conn: &mut Connection, command: Command) -> Command {
  if let Some(peers) = inner.connection.live_edit.liveness.peer_replies.as_ref() {
    let now = Instant::now();
    let slot = if inner.config.server.is_clustered() {
      inner
        .with_cluster_state(|state| Ok(judged_slot(state, &conn.server, &command)))
        .unwrap_or(None)
    } else {
      command.cluster_hash()
    };
    match slot {
      Some(slot) => peers.stamp(&peers.shared.slots[slot as usize % SLOTS], now),
      None => {
        let server = &conn.server;
        let node = conn
          .live_edit
          .liveness
          .node_stamp
          .get_or_insert_with(|| peers.node(server));
        peers.stamp(node, now);
      },
    }
  }
  command
}

fn oldest_outstanding(conn: &Connection) -> Option<&Command> {
  conn.buffer.iter().find(|command| !command.kind.is_pubsub())
}

/// Whether the oldest outstanding command's frame is still only fed (one of the last `unsent` tracked commands).
fn oldest_is_unsent(conn: &Connection) -> bool {
  let tracked = conn.buffer.iter().filter(|command| !command.kind.is_pubsub()).count();
  tracked > 0 && conn.live_edit.liveness.unsent >= tracked
}

/// The newest reply on any connection sharing `peers` from the slot range of `conn`'s oldest outstanding command,
/// or from `conn`'s node for a command judged at node level ([`judged_slot`]) or a centralized client. Without the
/// client's cluster routing no slot can be placed on the node, so only the node's own cell counts.
fn newest_peer_reply(inner: &RefCount<ClientInner>, conn: &Connection, peers: &PeerReplies) -> Option<Instant> {
  if !inner.config.server.is_clustered() {
    return peers.newest(&[(0, (SLOTS - 1) as u16)], Some(&peers.node(&conn.server)));
  }
  let oldest = oldest_outstanding(conn);
  let (ranges, by_node) = inner
    .with_cluster_state(|state| {
      let slot = oldest.and_then(|command| judged_slot(state, &conn.server, command));
      let ranges = state
        .slots()
        .iter()
        .filter(|range| match slot {
          Some(slot) => range.start <= slot && slot <= range.end,
          None => range.primary == conn.server,
        })
        .map(|range| (range.start, range.end))
        .collect::<Vec<_>>();
      Ok((ranges, slot.is_none()))
    })
    .unwrap_or_else(|_| (Vec::new(), true));
  if by_node {
    peers.newest(&ranges, Some(&peers.node(&conn.server)))
  } else {
    peers.newest(&ranges, None)
  }
}

/// Update the connection's progress and return when its current stall began, or `None` when it has nothing to
/// judge (no outstanding non-pubsub command, or `blocked`). A connection whose bounded wait expired is always
/// judged.
///
/// Ends the previous stall when it changed: one that had peer evidence and recovered by itself is a near miss.
pub(crate) fn stall_start(inner: &RefCount<ClientInner>, conn: &mut Connection, now: Instant) -> Option<Instant> {
  if conn.live_edit.liveness.expired.is_some() {
    return Some(now);
  }
  let read_len = read_buffer_len(&conn.transport);
  let written = if conn.blocked {
    None
  } else {
    // A command still only fed is measured from the later of its prepare time and the connection's last completed
    // write, feed or flush: a large frame being uploaded ahead of it is progress, not silence.
    let unsent = oldest_is_unsent(conn);
    let last_write = conn.live_edit.liveness.last_write;
    oldest_outstanding(conn).map(|command| {
      let written = command.network_start.unwrap_or(now);
      match last_write {
        Some(last_write) if unsent && last_write > written => last_write,
        _ => written,
      }
    })
  };
  let state = &mut conn.live_edit.liveness;
  if read_len > state.read_len {
    state.last_progress = Some(now);
  }
  state.read_len = read_len;
  let since = written.map(|written| match state.last_progress {
    Some(progress) if progress > written => progress,
    _ => written,
  });

  if let Some(ended) = state.episode.filter(|episode| Some(episode.since) != since) {
    state.episode = None;
    state.evidence_checked = None;
    if ended.evidence {
      let end = state
        .last_progress
        .filter(|progress| *progress > ended.since)
        .unwrap_or(now);
      near_miss(inner, &conn.server, end.saturating_duration_since(ended.since));
    }
  }
  if let Some(since) = since {
    if state.episode.is_none() {
      state.episode = Some(Episode { since, evidence: false });
      if state.longest_wait.is_some_and(|wait| wait.ended < since) {
        state.longest_wait = None;
      }
      if state.longest_fed.is_some_and(|fed| fed.ended < since) {
        state.longest_fed = None;
      }
    }
  }
  since
}

/// Whether the stall that began at `since` is a verdict (fast or slow, see the module docs), and if so the error
/// that closes the connection. `slow` is the client's `UnresponsiveConfig::max_timeout`. A verdict is logged by
/// [`log_verdicts`] once every connection of the client has been polled; an expired wait was logged when it expired.
pub(crate) fn verdict(
  inner: &RefCount<ClientInner>,
  conn: &mut Connection,
  now: Instant,
  since: Instant,
  slow: Duration,
) -> Option<Error> {
  if conn.live_edit.liveness.expired.is_some() {
    return Some(verdict_error(Verdict::Slow));
  }
  let settings = &inner.connection.live_edit.liveness;
  let stall = now.saturating_duration_since(since);
  if stall > slow {
    return Some(defer(conn, Verdict::Slow, stall, since));
  }

  let (fast, peers) = match (settings.fast_timeout, settings.peer_replies.as_ref()) {
    (Some(fast), Some(peers)) => (fast, peers),
    _ => return None,
  };
  if stall <= settings.peer_evidence_lag {
    return None;
  }
  let has_evidence = conn.live_edit.liveness.episode.map(|e| e.evidence).unwrap_or(false);
  if !has_evidence {
    let due = conn
      .live_edit
      .liveness
      .evidence_checked
      .map(|checked| now.saturating_duration_since(checked) >= inner.connection.unresponsive.interval)
      .unwrap_or(true);
    if !due {
      return None;
    }
    conn.live_edit.liveness.evidence_checked = Some(now);
    let answered = newest_peer_reply(inner, conn, peers)
      .map(|newest| newest >= since + settings.peer_evidence_lag)
      .unwrap_or(false);
    if !answered {
      return None;
    }
    if let Some(episode) = conn.live_edit.liveness.episode.as_mut() {
      episode.evidence = true;
    }
  }

  if stall > fast {
    Some(defer(conn, Verdict::Fast, stall, since))
  } else {
    None
  }
}

/// Record a fresh fast or slow verdict on `conn` — the one place the read loop reaches one — and return its error.
/// The verdict is logged by [`log_verdicts`] (`pending`) and marks the connection judged dead (`closed_by`), so its
/// close does not flush ([`close`]).
///
/// The read loop returns that error as the connection's poll result, and the router closes the connection in the same
/// step: `process_response` drops a connection whose result is an error (`router/utils.rs` `disconnect!`). Neither
/// `select!` that polls the read loop drops a ready result: `read_or_write` is biased toward it, and in
/// `read_and_sleep` a verdict exists only once the read branch is polled and ready. An error from an earlier result of
/// the same batch can leave it unprocessed: in `read_or_write` that error ends the router, whose `disconnect_all`
/// closes the connection next; `read_and_sleep` returns early only when fred will not reconnect (a client that is
/// quitting, or whose reconnect policy is spent), and the connection is then closed at the router's end or judged
/// again at its next poll.
fn defer(conn: &mut Connection, verdict: Verdict, stall: Duration, since: Instant) -> Error {
  conn.live_edit.liveness.pending = Some(Pending { verdict, stall, since });
  conn.live_edit.liveness.closed_by = Some(verdict);
  verdict_error(verdict)
}

/// Whether the read loop reached a verdict on `conn` in this poll that is not logged yet.
pub(crate) fn has_pending(conn: &Connection) -> bool {
  conn.live_edit.liveness.pending.is_some()
}

/// Log every verdict the read loop reached on `connections` in this poll, each with a summary of the client's other
/// connections. Called after every connection has been polled (`ReadAllFuture`, `ReadFuture`), once
/// [`has_pending`] found one.
pub(crate) fn log_verdicts<'a, I>(inner: &RefCount<ClientInner>, connections: I)
where
  I: IntoIterator<Item = &'a mut Connection>,
{
  let mut all: Vec<&mut Connection> = connections.into_iter().collect();
  let pending: Vec<(usize, Pending)> = all
    .iter_mut()
    .enumerate()
    .filter_map(|(i, conn)| conn.live_edit.liveness.pending.take().map(|pending| (i, pending)))
    .collect();
  if !log::log_enabled!(log::Level::Warn) {
    return;
  }
  let now = Instant::now();
  for (i, pending) in pending {
    let conn = &*all[i];
    let newest_peer = describe_newest_peer(inner, conn, now);
    let mut line = format!(
      "Unresponsive connection to {} ({} verdict): no reply progress for {:?}; newest reply from the same slot \
       range on another connection: {}. ",
      conn.server,
      pending.verdict.as_str(),
      pending.stall,
      newest_peer
    );
    let since = Some(now.saturating_duration_since(pending.since));
    describe_connection(&mut line, conn, now, since);
    describe_client(&mut line, conn, now);
    line.push_str("Other connections: ");
    let mut others = 0;
    for (j, other) in all.iter().enumerate() {
      if j == i {
        continue;
      }
      if others > 0 {
        line.push_str("; ");
      }
      others += 1;
      describe_other(&mut line, other, now);
    }
    if others == 0 {
      line.push_str("none");
    }
    line.push_str(". ");
    describe_kernel(&mut line, conn);
    line.push_str(" Closing it and reconnecting.");
    _warn!(inner, "{}", line);
  }
}

/// The age of a command's `network_start`: since fred prepared its frame (`write_command`) while it is only fed,
/// since the send or flush that put it on the socket once sent ([`on_sent`] re-stamps it). Without fred's `metrics`
/// feature a command carries no other timestamp.
fn command_age(command: &Command, now: Instant) -> String {
  ago(now, command.network_start)
}

/// The age of the newest reply on another connection from the slot range of `conn`'s oldest outstanding command, or
/// from `conn`'s node ([`newest_peer_reply`]).
fn describe_newest_peer(inner: &RefCount<ClientInner>, conn: &Connection, now: Instant) -> String {
  inner
    .connection
    .live_edit
    .liveness
    .peer_replies
    .as_ref()
    .and_then(|peers| newest_peer_reply(inner, conn, peers))
    .map(|newest| format!("{:?} ago", now.saturating_duration_since(newest)))
    .unwrap_or_else(|| "none recorded".to_owned())
}

fn ago(now: Instant, at: Option<Instant>) -> String {
  match at {
    Some(at) => format!("{:?} ago", now.saturating_duration_since(at)),
    None => "never".to_owned(),
  }
}

fn in_stall(wait: Option<Wait>, stall: Option<Duration>, now: Instant) -> Option<Wait> {
  wait.filter(|wait| stall.map_or(true, |stall| now.saturating_duration_since(wait.ended) <= stall))
}

fn describe_wait(wait: Option<Wait>, stall: Option<Duration>, now: Instant) -> String {
  match in_stall(wait, stall, now) {
    Some(wait) => format!("{} {:?}", wait.what, wait.took),
    None => "none".to_owned(),
  }
}

/// The longest fed-to-sent delay of a command sent since the stall began: how long a frame sat in fred's write buffer
/// before it reached the socket, which `network_start`'s re-stamp at the send hides.
fn describe_fed(fed: Option<Wait>, stall: Option<Duration>, now: Instant) -> String {
  match in_stall(fed, stall, now) {
    Some(fed) => format!("{:?}", fed.took),
    None => "none".to_owned(),
  }
}

/// This connection's fields of a verdict line. `stall` limits the longest wait to the waits that ended within it.
fn describe_connection(line: &mut String, conn: &Connection, now: Instant, stall: Option<Duration>) {
  let state = &conn.live_edit.liveness;
  let oldest = match oldest_outstanding(conn) {
    Some(command) => format!(
      "{}, {} {}",
      command.kind.to_str_debug(),
      if oldest_is_unsent(conn) {
        "fed, not yet sent; prepared"
      } else {
        "sent"
      },
      command_age(command, now)
    ),
    None => "none".to_owned(),
  };
  let _ = write!(
    line,
    "This connection: {} in flight; oldest outstanding command {}; {} commands fed since the last send or flush, \
     {} bytes in the write buffer, feed count {}; {} bytes in the read buffer, last read progress {}; subscribed {}, \
     {} pubsub-shaped frames received; longest wait since the stall began: {}; longest time a command sent since \
     the stall began sat fed before it was sent: {}. ",
    conn.buffer.len(),
    oldest,
    state.unsent,
    write_buffer_len(&conn.transport),
    client_utils::read_atomic(&conn.counters.feed_count),
    read_buffer_len(&conn.transport),
    ago(now, state.last_progress),
    state.subscribed,
    state.pubsub_shaped.load(Ordering::Relaxed),
    describe_wait(state.longest_wait, stall, now),
    describe_fed(state.longest_fed, stall, now),
  );
}

/// The client's fields of a verdict line, read through `conn`.
fn describe_client(line: &mut String, conn: &Connection, now: Instant) {
  let (queued, last_command) = match conn.live_edit.liveness.inner.as_ref() {
    Some(ClientHandle(inner)) => (
      inner.live_edit.liveness.shared.queued.load(Ordering::Relaxed).to_string(),
      ago(now, inner.live_edit.liveness.last_command()),
    ),
    None => ("unknown".to_owned(), "unknown".to_owned()),
  };
  let _ = write!(
    line,
    "This client: command-queue counter {}, command channel length {} when the router last waited, last command \
     taken by the router {}. ",
    client_utils::read_atomic(&conn.counters.cmd_buffer_len),
    queued,
    last_command,
  );
}

/// One other connection's summary in a verdict line.
fn describe_other(line: &mut String, conn: &Connection, now: Instant) {
  let state = &conn.live_edit.liveness;
  let stall = state.episode.map(|episode| now.saturating_duration_since(episode.since));
  let _ = write!(
    line,
    "{}: {} in flight, oldest {}, {} unsent, {} bytes in the write buffer, longest wait {}, longest fed before \
     sent {}",
    conn.server,
    conn.buffer.len(),
    oldest_outstanding(conn)
      .map(|command| command_age(command, now))
      .unwrap_or_else(|| "none".to_owned()),
    state.unsent,
    write_buffer_len(&conn.transport),
    describe_wait(state.longest_wait, stall, now),
    describe_fed(state.longest_fed, stall, now),
  );
}

/// The kernel's view of the connection's socket, from `/proc/self/net/tcp` or `tcp6`; `unavailable` with no such file
/// or no matching row.
fn describe_kernel(line: &mut String, conn: &Connection) {
  line.push_str("Kernel socket: ");
  match socket_addrs(&conn.transport).and_then(|(local, remote)| kernel_row(local, remote)) {
    Some(row) => line.push_str(&row),
    None => line.push_str("unavailable"),
  }
  line.push('.');
}

/// A socket address as `/proc/net/tcp{,6}` prints it: the address as the native-endian 32-bit words of its network
/// byte order bytes, then the port, all in upper-case hex.
fn proc_address(addr: SocketAddr) -> String {
  let mut out = String::new();
  match addr {
    SocketAddr::V4(v4) => {
      let _ = write!(out, "{:08X}", u32::from_ne_bytes(v4.ip().octets()));
    },
    SocketAddr::V6(v6) => {
      let octets = v6.ip().octets();
      for word in 0 .. 4 {
        let bytes = [
          octets[word * 4],
          octets[word * 4 + 1],
          octets[word * 4 + 2],
          octets[word * 4 + 3],
        ];
        let _ = write!(out, "{:08X}", u32::from_ne_bytes(bytes));
      }
    },
  }
  let _ = write!(out, ":{:04X}", addr.port());
  out
}

fn timer_name(code: u64) -> String {
  match code {
    0 => "none".to_owned(),
    1 => "retransmit".to_owned(),
    2 => "keepalive".to_owned(),
    3 => "time-wait".to_owned(),
    4 => "zero-window probe".to_owned(),
    other => format!("code {other}"),
  }
}

/// The `tx_queue`, `rx_queue`, timer and `retrnsmt` fields of the `/proc/self/net/tcp{,6}` row whose local and remote
/// addresses are `local` and `remote`.
fn kernel_row(local: SocketAddr, remote: SocketAddr) -> Option<String> {
  let path = match (local, remote) {
    (SocketAddr::V4(_), SocketAddr::V4(_)) => "/proc/self/net/tcp",
    (SocketAddr::V6(_), SocketAddr::V6(_)) => "/proc/self/net/tcp6",
    _ => return None,
  };
  let (local, remote) = (proc_address(local), proc_address(remote));
  let table = std::fs::read_to_string(path).ok()?;
  let hex = |field: &str| u64::from_str_radix(field, 16).ok();
  for row in table.lines().skip(1) {
    let fields: Vec<&str> = row.split_whitespace().collect();
    if fields.len() < 7 || !fields[1].eq_ignore_ascii_case(&local) || !fields[2].eq_ignore_ascii_case(&remote) {
      continue;
    }
    let (tx, rx) = fields[4].split_once(':')?;
    let (timer, _) = fields[5].split_once(':')?;
    return Some(format!(
      "tx_queue {} bytes, rx_queue {} bytes, timer {}, retrnsmt {}",
      hex(tx)?,
      hex(rx)?,
      timer_name(hex(timer)?),
      hex(fields[6])?
    ));
  }
  None
}

/// Log a near miss: a stall that had peer evidence and recovered by itself. Rate-limited per registry to one line
/// per [`NEAR_MISS_LOG_INTERVAL`], except that a near miss longer than every one logged in the current window is
/// always logged.
fn near_miss(inner: &RefCount<ClientInner>, server: &Server, stall: Duration) {
  let peers = match inner.connection.live_edit.liveness.peer_replies.as_ref() {
    Some(peers) => peers,
    None => return,
  };
  let now = Instant::now();
  let mut log = peers.shared.near_misses.lock().unwrap_or_else(|e| e.into_inner());
  let window_over = log
    .window_start
    .map(|start| now.saturating_duration_since(start) >= NEAR_MISS_LOG_INTERVAL)
    .unwrap_or(true);
  if window_over || stall > log.longest {
    _info!(
      inner,
      "Near miss on the connection to {}: no reply progress for {:?} while the same slot range answered other \
       connections, then it recovered by itself ({} shorter near misses not logged since the previous line).",
      server,
      stall,
      log.suppressed
    );
    if window_over {
      log.window_start = Some(now);
    }
    log.longest = stall;
    log.suppressed = 0;
  } else {
    log.suppressed += 1;
  }
}
