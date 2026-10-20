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

//! Connection buffer retention: a cap on what a connection keeps once its traffic drains.
//!
//! Part of the live-edit patch to fred 10.1.0; the record of every change, with the reason for each, is
//! `vendor/fred-10.1.0.SOURCE.md` in the live-edit repository.
//!
//! # Why
//!
//! A stock connection has four holders whose allocations only ever grow, so what it keeps is the largest load it
//! has ever carried, freed only when the connection itself is replaced:
//!
//! - **the in-flight queue** (`Connection::buffer`, 64 slots to start): the router writes a whole pipeline before it
//!   reads a reply, so the queue grows to the longest pipeline in flight. A slot is [`IN_FLIGHT_SLOT_BYTES`].
//! - **the write buffer** (tokio-util's `Framed`, 8 KiB to start): a frame is encoded whole, so it grows to the
//!   largest frame plus what was buffered ahead of it.
//! - **the read buffer** (8 KiB to start): fred decodes only complete frames, so it grows to hold the largest frame
//!   read.
//! - **the router's retry buffer** (`Router::retry_buffer`, empty and unallocated to start): it takes a lost
//!   connection's in-flight commands for replay.
//!
//! Draining a byte buffer does not free its allocation: a flush advances the buffer and a decode splits the frame
//! off, and the next `reserve` reclaims the whole allocation for reuse.
//!
//! A client also holds a backchannel connection (`Backchannel::transport`), opened lazily beside its pipelined
//! connections: for `CLIENT UNBLOCK` under `Blocking::Interrupt` (from a reader's first interrupt), and for
//! `CLUSTER SLOTS` at a cluster client's topology sync (re-created at every sync of a healthy client). Its buffers
//! grow like any other connection's, and nothing in stock fred cuts them back.
//!
//! # The rule
//!
//! With `Settings::retention` (on `ConnectionConfig::live_edit`) set to a [`RetentionLimits`], each holder is cut
//! back once its traffic drains. Every check runs in the router task that owns the holder.
//!
//! A connection is **drained** when its in-flight queue is empty and its write and read buffers hold nothing (no
//! unflushed frame, no partial frame). Each client has at most one **warm** connection, the only one allowed to keep
//! the class limits once drained; every other drained connection is cut to the **idle shape**: an in-flight queue
//! with no allocation, an unallocated write buffer and a fresh 8 KiB read buffer ([`cut`]).
//!
//! The class rules, which every connection gets as its traffic passes and the warm one keeps once drained:
//!
//! - **in-flight queue**, after a reply is matched to the command at its head ([`on_reply`]): when its capacity is
//!   above `in_flight_commands` and at most half that many commands are still queued, it shrinks to
//!   `in_flight_commands`. The half keeps a queue that hovers near the limit from reallocating on every reply.
//! - **write buffer**, after a write-and-flush or a flush leaves it empty ([`on_flushed`]): when an encode since it
//!   was last replaced left more than `write_buffer_bytes` in it, it is replaced by a fresh 8 KiB buffer.
//! - **read buffer**, when a frame is taken off the connection and the buffer is left empty ([`on_read`]): when a
//!   frame decoded since it was last replaced was larger than `read_buffer_bytes`, it is replaced by a fresh 8 KiB
//!   buffer. Every way a frame is taken off a pipelined connection runs the check before the frame's caller is
//!   answered: the read loop (`live_edit::on_frame`), the reply pop ([`on_reply`], which also covers replies a drain
//!   reads), and `Connection::read` through its `live_edit::next` hook, which covers the frames fred reads outside
//!   the read loop, such as a `Transaction`'s EXEC reply.
//!
//! The client-wide rules:
//!
//! - **the sweep** ([`sweep`]), the first thing the router does on each pass of its loop (`live_edit::before_wait`,
//!   at the top of `read_or_write`): it elects the warm connection ([`elect`]), applies the class rules to the warm
//!   connection once drained (a backstop: every read and flush is also checked when it happens), and cuts every other
//!   drained connection that carried traffic since its last cut ([`action`]). The router passes once per command,
//!   pipeline or batch of reads and, with the unresponsive check configured, at least once per its interval while it
//!   waits. The sweep's own work allocates nothing (*The sweep allocates nothing*, below).
//! - **the cut at the reply pop** ([`on_reply`]): a reply that leaves a non-warm connection drained cuts it there,
//!   before the reply's caller is answered.
//! - **retry buffer**, after the router replays it ([`on_retry_replayed`]): once empty, it is replaced by an
//!   unallocated `VecDeque::new()`, stock fred's starting state. It is used only during a replay, so it keeps
//!   nothing between replays.
//! - **backchannel**, in the sweep ([`cut_backchannel`]): once it holds nothing (both buffers empty) and it carried
//!   anything since its last cut (its `idle` flag is clear), both its buffers are replaced by fresh 8 KiB ones and
//!   `idle` is set.
//! - **cluster routing table**, at each rebuild from a `CLUSTER SLOTS` reply ([`own_slot_ranges`]): every string the
//!   table keeps is copied out of the reply, so the reply's read-buffer allocation is freed when the sync ends.
//! - **sentinel primary**, when a sentinel names the primary to connect to ([`own_server`]): its host is copied out
//!   of the sentinel's reply before fred connects, so the sentinel connection's read buffer is not kept alive.
//!
//! Only capacity changes. Command order, write attempts, flush points, what is written and when are untouched, and a
//! byte buffer is replaced only while it holds nothing. A decoded frame holds its own reference to its bytes, so
//! replacing the read buffer never touches a reply on its way to a caller: the old allocation is freed when the
//! last frame cut from it is dropped, which a frame kept past its call would prevent. fred keeps two such values, the
//! cluster routing table and the primary `Server` a sentinel names, and [`own_slot_ranges`] and [`own_server`] make
//! both own their strings, so on every topology no value fred keeps past a call holds a view into a read buffer.
//! The cost is one reallocation per drain after traffic above a limit, and on a client with several connections, up
//! to three small allocations when a cut connection is used again.
//!
//! # Why one warm connection, and how it is chosen
//!
//! A cluster client holds a connection to every primary, and its traffic reaches all of them over time, so a limit
//! per connection alone lets a client keep its class limits once per primary. One warm connection per client makes
//! what a client keeps independent of the primary count: one connection at the class limits, the rest at the idle
//! shape.
//!
//! - **Read progress chooses it.** The warm connection should be the one the client's traffic is using now, so that
//!   the connection kept ready is the one the next call most likely lands on. The newest read progress (a frame taken
//!   off the connection, or its read buffer grew) marks it, and it is already recorded per connection by the liveness
//!   check (`liveness::last_progress`) on every read, so choosing by it adds nothing to the command path. Write
//!   recency would need a hook in fred's routing path or a clock read per write.
//! - **Warmth moves only when the warm connection is drained**, and only to a connection with strictly newer read
//!   progress. A connection is never demoted while a call it carries is in flight, so that call's buffers are left
//!   alone until it ends, and two busy connections cannot trade warmth back and forth on every frame. A tie keeps it.
//! - **A client with one connection is always warm**, from the first sweep after the connection is made. That
//!   connection carries all of the client's traffic, and one connection at the class limits is already what the
//!   bound allows per client: cutting it would only make every call regrow. So, at rest, the rule changes nothing
//!   for a standalone or sentinel store, for every single-node client, and for a one-primary cluster. The one
//!   exception is a reply popped in the same router pass as the reconnect, before any sweep: a fred `Transaction`
//!   retried after a reconnect (`retry!` → `reconnect_with_policy` → `drain_all`,
//!   `router/transactions.rs:122-142`) drains on a connection not yet marked warm, and can cut it once. That
//!   changes allocations only; the next sweep marks it warm.
//! - **The idle read buffer stays at 8 KiB**, tokio-util's own starting size. tokio-util reserves one byte before
//!   each read, so an unallocated read buffer would grow by doubling from a few bytes and take several reads for
//!   even a 1 KiB reply. At 8 KiB, a connection used again reads a reply of up to 8 KiB in one read.
//! - **The backchannel is taken with `try_write`.** Its lock is a tokio `RwLock` that a caller task holds while it
//!   uses the backchannel (to interrupt a blocked command, or for a topology sync). The router task must never wait
//!   on a caller, so the sweep takes the lock only when it is free; a busy lock skips the backchannel for that pass,
//!   and a later pass tries again.
//! - **The backchannel is cut after any traffic**, not after traffic above a size. That makes "8 KiB read + 8 KiB
//!   write" exact by construction: a size threshold would have to prove, from bytes' growth rules, that no frame
//!   under it can grow the buffer (one from 4 to 8 KiB can double it to 16 KiB behind a setup reply's prefix), and
//!   that a credential in its HELLO never grows the write buffer. The cost is two 8 KiB allocations per use, and use
//!   is rare: one per cluster sync, which already opens a new connection, and one per reader interrupt.
//!
//! # Why the cluster routing table and the sentinel primary own their strings
//!
//! fred parses `CLUSTER SLOTS` without copying: every string in the reply frame, and so each `SlotRange`'s `id`,
//! primary `host` and TLS server name, is a view into the read-buffer allocation the reply was read into. Those
//! strings outlive the sync that read them. They are kept by the routing state (in `ClientInner` and in the
//! router's cache), by every connection-map key and `Connection::server` added at that sync, by the next
//! backchannel's `server` (`connect_any` takes it from the old routing), and by each `Server` a caller clones from
//! the routing, such as the live-edit service's stream readers, whose client config keeps their owner's `Server`
//! for life. Each one would pin a whole read-buffer allocation, however the backchannel's buffer is cut, and the
//! pins would grow with the primary count and the sync history. [`own_slot_ranges`], called at the end of
//! `ClusterRouting::rebuild`, copies them out, so the reply's allocation is freed when the sync ends and every
//! `Server` cloned later shares the small owned strings.
//!
//! A sentinel store has the same problem once per connection. fred converts the `SENTINEL GET-MASTER-ADDR-BY-NAME`
//! reply to the primary's host `Str` without copying (`FromValue for Str`), so the `Server` it builds views the
//! sentinel connection's read buffer, and it lives on as the primary connection's `server` and `default_host`, its
//! codec's server, the client's recorded sentinel primary and the backchannel's connection-id key. Every client of
//! a sentinel store, the live-edit service's stream readers included, discovers its own primary this way.
//! [`own_server`], called on that `Server` before fred connects to it (`router/sentinel.rs`,
//! `discover_primary_node`), copies its strings out. The sentinel node list fred refreshes from `SENTINEL sentinels`
//! is converted through owned `String`s and needs no copy. These two, the routing table and the sentinel primary,
//! are the only reply strings fred keeps past the call that read them.
//!
//! The election, the per-connection decision, the backchannel decision and the two string copies are pure
//! functions ([`elect`], [`elect_streaming`], [`action`], [`plan`], [`cut_backchannel`], [`own_slot_ranges`],
//! [`own_server`]), so they can be tested without connections.
//!
//! # The sweep allocates nothing
//!
//! The sweep runs on every router pass, about once per command on a busy client, so its own work allocates nothing:
//! no list of the connections, their views or their actions. It walks the router's connections twice, through an
//! iterator over the router's own collections (`liveness::connections_mut`). The first walk folds the election over
//! the connections' views ([`elect_streaming`]), keeping in locals only the current warm connection's index and view
//! and the newest read progress's index and time; the second sets each connection's warm flag from the result and
//! applies its [`action`], reading its view again. Nothing adds or removes a connection between the walks, so both
//! visit the connections in the same order, and what the second walk does to one connection changes no other
//! connection's view. The decisions are [`plan`]'s: [`elect`] is the same fold over a slice, and [`plan`] then
//! applies [`action`] to each connection with its elected flag, as the second walk does.
//!
//! The only allocations in a sweep are those its cuts and class rules make: a cut's fresh read buffer, a class rule's
//! shrunk in-flight queue or replacement byte buffer, and a backchannel cut's two fresh buffers. A pass that changes
//! no buffer, which is every pass of a client at rest once its connections are cut back, allocates nothing.
//!
//! # Byte buffers are judged by what they held, never by `capacity()`
//!
//! `BytesMut::capacity()` does not show how large a drained buffer's allocation is: `advance` (a flush) and
//! `split_to` (a decode) lower it by the bytes consumed while the allocation stays whole, for the next `reserve` to
//! reclaim. A drained write buffer whose allocation is 16 MiB reports a capacity near zero, so a check keyed on
//! `capacity()` would never fire. The codec instead records, per connection, the write buffer's length after each
//! encode and the size of each decoded frame ([`on_encode`], [`on_decode`]): what the allocation grew to hold.
//!
//! # What a drained connection keeps
//!
//! A limit caps what a buffer may have *held* before it is replaced, not its allocation, which `BytesMut` grows by
//! at least doubling. [`RetentionLimits::kept_queue_bytes`], [`RetentionLimits::kept_write_bytes`] and
//! [`RetentionLimits::kept_read_bytes`] bound what the warm connection keeps once drained; every other connection
//! keeps the idle shape, [`FRESH_BUFFER_BYTES`] of read buffer; a replayed retry buffer keeps nothing; a backchannel
//! keeps two [`FRESH_BUFFER_BYTES`] buffers once the sweep has seen it hold nothing. The bounds are derived from fred
//! 10.1.0, tokio-util 0.7.19
//! (`FramedImpl`) and bytes 1.12.1 (`BytesMut::split_to` and `BytesMut::reserve`), and must be re-derived when any
//! of them changes.
//!
//! # Test support
//!
//! With fred's `live-edit-retention-report` feature, which only the live-edit service's `test-support` feature
//! enables, `test_support` lets a test see and break the warm rule on a running client:
//!
//! - **a report** per client: `request_report` makes a request, and the first router pass whose sweep began after
//!   it answers right after that sweep (`live_edit::before_wait`, which reads the pending request before the sweep)
//!   with one `ConnectionReport` per connection (its flags, read progress, in-flight command count and buffer
//!   capacities); `take_report` collects it. A report shows exactly what that pass's sweep left, and a sweep mode
//!   set before the request is the mode that sweep ran. Its byte-buffer capacities are `BytesMut::capacity()`, so
//!   they are exact only for a buffer nothing has been consumed from, which is what a cut leaves; the rule itself
//!   still never reads `capacity()` (above).
//! - **sweep modes** per client (`set_sweep_for_test`): `Skipped` models a build without the sweep (no warm
//!   connection, every drained connection cut at its reply pop), `EveryConnectionWarm` the per-connection rule alone
//!   (every connection keeps the class limits), and `AsBuilt`, the default, the sweep as built.
//! - **sweep counters** per client (`sweep_stats`): how many sweeps ran and the wall time they took, in nanoseconds,
//!   counted from the client's creation in every sweep mode, for a test that reports what the sweep costs per pass.
//!   A client without limits runs no sweep and counts none.
//! - **write-check counters** per client (`write_check_stats`): how many checks before a write
//!   (`liveness::before_socket_write`, `liveness::before_write_to`, which run only on a clustered client) ran, the
//!   wall time of their decisions in nanoseconds, and how many went on to flush, with those flushes' time, counted
//!   from the client's creation, for a test that reports what the check costs per write. Unlike the sweep counters
//!   these count whether or not the client has limits.
//! - **an in-sweep flag** per thread (`in_sweep`): true only while a sweep runs on the calling thread. The sweep is
//!   synchronous, so the flag is set and cleared on the router's thread within one pass; a test's counting allocator
//!   reads it to count the allocations made inside the sweep. It is a const-initialised thread-local `Cell<bool>`,
//!   so reading or setting it never allocates.
//!
//! None of it exists in a build without the feature, the release build included: no state on the client, no check,
//! clock read or thread-local in the router's pass or in its writes.

use crate::{
  live_edit::liveness,
  modules::inner::ClientInner,
  protocol::{
    codec::Codec,
    command::Command,
    connection::{Connection, ConnectionKind},
    types::{Server, SlotRange},
  },
  router::Router,
  runtime::RefCount,
};
use bytes::BytesMut;
use bytes_utils::Str;
use std::{
  collections::VecDeque,
  sync::atomic::{AtomicBool, AtomicUsize, Ordering},
  time::Instant,
};

/// The bytes one in-flight queue or retry buffer slot takes: one `Command`.
pub const IN_FLIGHT_SLOT_BYTES: usize = std::mem::size_of::<Command>();

/// The size of a replacement read buffer, and of the backchannel's buffers: tokio-util's `Framed` starts both buffers
/// at 8 KiB (`INITIAL_CAPACITY`, `codec/framed_impl.rs:25`), so a replaced buffer is the same as a new connection's.
pub const FRESH_BUFFER_BYTES: usize = 8 * 1024;

/// tokio-util's write backpressure boundary (`FramedImpl::poll_ready`, `codec/framed_impl.rs:262-268`): a write
/// starts only while less than this is buffered; at or past it, a feed first flushes the buffer (the liveness
/// module's flush before a write that can reach a socket reads it too).
pub(crate) const BACKPRESSURE_BYTES: usize = 8 * 1024;

/// The largest capacity bytes 1.12.1 carries over when a shared buffer reallocates (`original_capacity_from_repr`,
/// `MAX_ORIGINAL_CAPACITY_WIDTH = 17`).
const MAX_CARRIED_CAPACITY: usize = 64 * 1024;

/// What one connection may keep once its traffic drains, on `ConnectionConfig::live_edit.retention`. Read once per
/// connection, when its codec is created; a reconnect reads it again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionLimits {
  /// The in-flight queue's capacity, in commands, once at most half that many are queued.
  pub in_flight_commands: usize,
  /// The most the write buffer may have held since it was last replaced without being replaced once it drains.
  pub write_buffer_bytes: usize,
  /// The largest frame the read buffer may have held since it was last replaced without being replaced once it
  /// drains.
  pub read_buffer_bytes:  usize,
}

impl RetentionLimits {
  /// The most a drained in-flight queue keeps: `in_flight_commands` slots. Once the queue is empty its next check
  /// shrinks any larger capacity to the limit.
  pub const fn kept_queue_bytes(&self) -> usize {
    self.in_flight_commands * IN_FLIGHT_SLOT_BYTES
  }

  /// An upper bound on the allocation a drained write buffer keeps when nothing it held exceeded
  /// `write_buffer_bytes`: `2 × (write_buffer_bytes + 8 KiB)`. An encode reserves the frame's length once. A
  /// reserve that does not fit grows the allocation to the larger of twice itself and what it needs: the contents
  /// plus a consumed prefix that is either reclaimed first or shorter than what is already buffered, which is under
  /// 8 KiB when a write starts (tokio-util flushes once 8 KiB is buffered).
  pub const fn kept_write_bytes(&self) -> usize {
    2 * (self.write_buffer_bytes + BACKPRESSURE_BYTES)
  }

  /// An upper bound on the allocation a drained read buffer keeps when no frame it held exceeded
  /// `read_buffer_bytes`: just under `4 × max(read_buffer_bytes, 64 KiB)`.
  ///
  /// The decoder takes each frame off the buffer with `split_to`, which makes the allocation shared and leaves the
  /// consumed frame behind as an offset into it, and tokio-util reserves one byte before each read. A `reserve` on
  /// the buffer then takes one of two branches:
  ///
  /// - **unique** (every frame cut from the allocation was dropped): it reuses the space after the contents, or
  ///   copies the partial frame to the front when the consumed prefix is at least as long as it; it allocates only
  ///   when the partial frame is longer than the prefix, and then at twice the old allocation, which the prefix
  ///   and the partial frame filled, so at under four times the partial frame;
  /// - **shared** (a frame cut from it is still held): it allocates the larger of the partial frame plus one byte
  ///   and the buffer's original capacity, which bytes carries over only up to 64 KiB, and grows from there by the
  ///   unique rule.
  ///
  /// A partial frame is shorter than its frame, so either branch leaves the allocation under four times the larger
  /// of the largest frame and 64 KiB, however the frames before it were sized or held.
  pub const fn kept_read_bytes(&self) -> usize {
    let held = if self.read_buffer_bytes > MAX_CARRIED_CAPACITY {
      self.read_buffer_bytes
    } else {
      MAX_CARRIED_CAPACITY
    };
    4 * held
  }
}

/// Per-connection retention state, on the connection's codec (`Codec::live_edit`): the limits, the largest
/// write-buffer length and decoded frame since the matching buffer was last replaced, and the warm and idle flags.
/// Created with the codec, dropped with it; a reconnect starts fresh (not warm, not idle).
///
/// It lives on the codec because only the codec sees each encode's and decode's size, and the connection reaches it
/// through its transport. The fields are atomics only because a codec must be `Sync`; they are touched by one task
/// at a time: the router task that owns the connection or, for a backchannel, whichever task holds its lock.
#[derive(Debug, Default)]
pub struct CodecState {
  limits:        Option<RetentionLimits>,
  largest_write: AtomicUsize,
  largest_frame: AtomicUsize,
  /// Whether this is its client's warm connection, the one that keeps the class limits once drained. Set by the
  /// sweep's election.
  warm:          AtomicBool,
  /// Whether the connection is in the idle shape (for a backchannel: fresh 8 KiB buffers) and has carried nothing
  /// since: set by a cut, cleared by every encode and decode, so a connection is cut only if it carried something
  /// since its last cut.
  idle:          AtomicBool,
}

/// The retention state for a new connection of the client `inner`.
pub(crate) fn codec_state(inner: &RefCount<ClientInner>) -> CodecState {
  CodecState {
    limits: inner.connection.live_edit.retention,
    ..CodecState::default()
  }
}

fn raise(mark: &AtomicUsize, value: usize) {
  if value > mark.load(Ordering::Relaxed) {
    mark.store(value, Ordering::Relaxed);
  }
}

/// The codec encoded a frame, leaving `buffered` bytes in the write buffer.
pub(crate) fn on_encode(codec: &Codec, buffered: usize) {
  if codec.live_edit.limits.is_some() {
    raise(&codec.live_edit.largest_write, buffered);
    codec.live_edit.idle.store(false, Ordering::Relaxed);
  }
}

/// The codec decoded a frame of `size` bytes.
pub(crate) fn on_decode(codec: &Codec, size: usize) {
  if codec.live_edit.limits.is_some() {
    raise(&codec.live_edit.largest_frame, size);
    codec.live_edit.idle.store(false, Ordering::Relaxed);
  }
}

/// `$body` with `$framed` bound to the transport's `Framed`, whose type differs per transport kind.
macro_rules! with_framed {
  ($transport:expr, $framed:ident => $body:expr) => {
    match $transport {
      ConnectionKind::Tcp(ref mut t) => {
        let $framed = t.get_mut();
        $body
      },
      #[cfg(feature = "unix-sockets")]
      ConnectionKind::Unix(ref mut t) => {
        let $framed = t.get_mut();
        $body
      },
      #[cfg(any(feature = "enable-rustls", feature = "enable-rustls-ring"))]
      ConnectionKind::Rustls(ref mut t) => {
        let $framed = t.get_mut();
        $body
      },
      #[cfg(feature = "enable-native-tls")]
      ConnectionKind::NativeTls(ref mut t) => {
        let $framed = t.get_mut();
        $body
      },
    }
  };
}

/// What the warm election and the cut decision read about one connection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConnectionView {
  /// Whether the connection is its client's warm connection now.
  pub warm:           bool,
  /// Whether the connection is in the idle shape and has carried nothing since its last cut.
  pub idle:           bool,
  /// Commands in its in-flight queue.
  pub queued:         usize,
  /// Bytes in its write buffer (fed but unflushed frames).
  pub write_buffered: usize,
  /// Bytes in its read buffer (a partial frame).
  pub read_buffered:  usize,
  /// Its newest read progress, if any.
  pub last_progress:  Option<Instant>,
}

impl ConnectionView {
  /// Whether the connection is drained: nothing queued, nothing in either buffer.
  pub fn drained(&self) -> bool {
    self.queued == 0 && self.write_buffered == 0 && self.read_buffered == 0
  }
}

/// What the sweep does to one connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
  /// Nothing: the connection is not drained, or is already in the idle shape and carried nothing since.
  Keep,
  /// Apply the class rules: the drained warm connection.
  ClassRules,
  /// Cut it to the idle shape: a drained connection that is not warm and carried something since its last cut.
  Cut,
}

/// The index of the warm connection among a client's `connections`, given each one's current state: the one
/// connection of a one-connection client; else the current warm connection `W`, unless `W` is drained and another
/// connection has strictly newer read progress, which then becomes warm; else, with no `W`, the connection with the
/// newest read progress (the first of a tie). `None` when there is no connection, or no `W` and no connection has
/// made read progress yet.
///
/// [`elect_streaming`] over the slice: the fold the sweep runs over the router's connections.
pub fn elect(connections: &[ConnectionView]) -> Option<usize> {
  elect_streaming(connections.iter().copied())
}

/// [`elect`] as a fold over a client's connection views in order, keeping only the count, the first connection
/// marked warm (`W`: its index and view) and the newest read progress (its index and time; the first of a tie). The
/// sweep folds it over the views of the router's connections as it walks them, so the election allocates nothing.
/// The index it returns counts `views` from 0, in the order given.
pub fn elect_streaming(views: impl IntoIterator<Item = ConnectionView>) -> Option<usize> {
  let mut count = 0;
  let mut current: Option<(usize, ConnectionView)> = None;
  let mut newest: Option<(usize, Instant)> = None;
  for (i, view) in views.into_iter().enumerate() {
    count = i + 1;
    if current.is_none() && view.warm {
      current = Some((i, view));
    }
    if let Some(progress) = view.last_progress {
      if newest.map_or(true, |(_, newest)| progress > newest) {
        newest = Some((i, progress));
      }
    }
  }
  if count <= 1 {
    return if count == 0 { None } else { Some(0) };
  }
  match current {
    None => newest.map(|(n, _)| n),
    Some((w, current)) => match newest {
      Some((n, progress)) if n != w && current.drained() && Some(progress) > current.last_progress => Some(n),
      _ => Some(w),
    },
  }
}

/// What the sweep does to one connection whose `warm` flag reflects the election.
pub fn action(view: &ConnectionView) -> Action {
  if !view.drained() {
    Action::Keep
  } else if view.warm {
    Action::ClassRules
  } else if view.idle {
    Action::Keep
  } else {
    Action::Cut
  }
}

/// The sweep's decisions for one client's `connections`: the warm connection ([`elect`]) and, with each connection's
/// `warm` flag set from it, what is done to each ([`action`]). The sweep makes the same two passes over the router's
/// connections without collecting either result: the election's fold, then each connection's action as it is
/// reached.
pub fn plan(connections: &[ConnectionView]) -> (Option<usize>, Vec<Action>) {
  let warm = elect(connections);
  let actions = connections
    .iter()
    .enumerate()
    .map(|(i, view)| elected_action(view, warm == Some(i)))
    .collect();
  (warm, actions)
}

/// [`action`] for a connection with its `warm` flag set from the election: `warm` is whether it was elected.
fn elected_action(view: &ConnectionView, warm: bool) -> Action {
  action(&ConnectionView { warm, ..*view })
}

/// Whether the sweep replaces a backchannel's read and write buffers with fresh [`FRESH_BUFFER_BYTES`] ones: when it
/// holds nothing (`read_buffered` and `write_buffered` both 0) and it carried anything since its last cut (`idle`
/// clear: a cut sets it, every encode and decode clears it, and a new connection starts with it clear).
pub fn cut_backchannel(read_buffered: usize, write_buffered: usize, idle: bool) -> bool {
  read_buffered == 0 && write_buffered == 0 && !idle
}

/// Make every string `ranges` keeps an owned copy: each range's `id`, its primary's `host` and TLS server name, and,
/// with the `replicas` feature, the same strings of each replica.
///
/// fred parses `CLUSTER SLOTS` without copying, so each of these strings is a view into the read-buffer allocation
/// the reply was read into, and anything that keeps one (the routing state, connection keys, a backchannel's
/// `server`, a `Server` a caller cloned from the routing) keeps that whole allocation alive. Called at the end of
/// `ClusterRouting::rebuild`, the only place a running client builds its routing table, so the reply's allocation
/// is freed when its sync ends. The contents are unchanged, so every comparison, hash and lookup is unchanged.
pub fn own_slot_ranges(ranges: &mut [SlotRange]) {
  for range in ranges.iter_mut() {
    range.id = owned(&range.id);
    own_server(&mut range.primary);
    #[cfg(feature = "replicas")]
    for replica in range.replicas.iter_mut() {
      own_server(replica);
    }
  }
}

/// An owned copy of `value`, in an allocation of its own.
fn owned(value: &Str) -> Str {
  Str::from(String::from(&**value))
}

/// Make the strings `server` keeps owned copies: its `host` and, with TLS, its TLS server name. The contents are
/// unchanged, so every comparison, hash and lookup is unchanged.
///
/// Used for each `Server` a cluster routing table keeps ([`own_slot_ranges`]), and called on the primary a sentinel
/// names (`router/sentinel.rs`, `discover_primary_node`) before fred connects to it. fred converts the
/// `SENTINEL GET-MASTER-ADDR-BY-NAME` reply to a `Str` without copying, so that host is a view into the sentinel
/// connection's read buffer; the `Server` built from it lives on as the primary connection's `server` and
/// `default_host`, its codec's server, the client's recorded sentinel primary and the backchannel's connection-id
/// key, each of which would keep that whole allocation alive for the connection's life.
pub fn own_server(server: &mut Server) {
  server.host = owned(&server.host);
  #[cfg(any(
    feature = "enable-rustls",
    feature = "enable-native-tls",
    feature = "enable-rustls-ring"
  ))]
  if let Some(ref name) = server.tls_server_name {
    server.tls_server_name = Some(owned(name));
  }
}

/// The view of `conn` the decisions read.
fn view(conn: &mut Connection) -> ConnectionView {
  let queued = conn.buffer.len();
  let last_progress = liveness::last_progress(conn);
  with_framed!(conn.transport, framed => {
    let state = &framed.codec().live_edit;
    ConnectionView {
      warm: state.warm.load(Ordering::Relaxed),
      idle: state.idle.load(Ordering::Relaxed),
      queued,
      write_buffered: framed.write_buffer().len(),
      read_buffered: framed.read_buffer().len(),
      last_progress,
    }
  })
}

/// The limits of `conn`'s client, when it has any.
fn limits_of(conn: &mut Connection) -> Option<RetentionLimits> {
  with_framed!(conn.transport, framed => framed.codec().live_edit.limits)
}

fn set_warm(conn: &mut Connection, warm: bool) {
  with_framed!(conn.transport, framed => framed.codec().live_edit.warm.store(warm, Ordering::Relaxed))
}

/// Shrink `queue` to `limit` slots when its capacity is above the limit and at most half the limit is queued.
fn shrink_queue(queue: &mut VecDeque<Command>, limit: usize) {
  if queue.capacity() > limit && queue.len() <= limit / 2 {
    queue.shrink_to(limit);
  }
}

/// Apply the class rules to `conn`: the queue rule, the read-buffer rule and the write-buffer rule.
fn class_rules(conn: &mut Connection, limits: RetentionLimits) {
  shrink_queue(&mut conn.buffer, limits.in_flight_commands);
  on_read(conn);
  on_flushed(conn);
}

/// Cut a drained `conn` to the idle shape: its in-flight queue's allocation freed, an unallocated write buffer and a
/// fresh 8 KiB read buffer. The caller has checked that it is drained.
fn cut(conn: &mut Connection) {
  conn.buffer.shrink_to(0);
  with_framed!(conn.transport, framed => {
    *framed.write_buffer_mut() = BytesMut::new();
    *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
    let state = &framed.codec().live_edit;
    state.largest_write.store(0, Ordering::Relaxed);
    state.largest_frame.store(0, Ordering::Relaxed);
    state.idle.store(true, Ordering::Relaxed);
  })
}

/// The first statement of each router pass (`live_edit::before_wait`): elect the client's warm connection, apply
/// the class rules to it once drained, cut every other drained connection that carried something since its last
/// cut, and cut the backchannel's buffers when it qualifies.
///
/// Two walks over the router's connections (`liveness::connections_mut`), with no collection of them: the first
/// folds the election over their views ([`elect_streaming`]), the second sets each one's warm flag and applies its
/// [`action`]. Nothing adds or removes a connection between the walks, so both visit them in the same order and the
/// election's index names the same connection in each; what the second walk does to one connection changes no other
/// connection's view. Apart from the buffers a cut or a class rule installs, the sweep allocates nothing (the module
/// docs' *The sweep allocates nothing*).
pub(crate) fn sweep(inner: &RefCount<ClientInner>, router: &mut Router) {
  let limits = match inner.connection.live_edit.retention {
    Some(limits) => limits,
    None => return,
  };
  // Test-only: mark this thread as inside the sweep and time the pass, until the guard drops as `sweep` returns.
  #[cfg(feature = "live-edit-retention-report")]
  let _pass = test_support::SweepPass::begin(inner);
  #[cfg(feature = "live-edit-retention-report")]
  match test_support::sweep_mode(inner) {
    test_support::SweepForTest::AsBuilt => {},
    test_support::SweepForTest::Skipped => {
      for conn in liveness::connections_mut(router) {
        set_warm(conn, false);
      }
      return;
    },
    test_support::SweepForTest::EveryConnectionWarm => {
      for conn in liveness::connections_mut(router) {
        set_warm(conn, true);
        if action(&ConnectionView {
          warm: true,
          ..view(conn)
        }) == Action::ClassRules
        {
          class_rules(conn, limits);
        }
      }
      sweep_backchannel(inner);
      return;
    },
  }
  let warm = elect_streaming(liveness::connections_mut(router).map(view));
  for (i, conn) in liveness::connections_mut(router).enumerate() {
    let elected = warm == Some(i);
    let seen = view(conn);
    set_warm(conn, elected);
    match elected_action(&seen, elected) {
      Action::Keep => {},
      Action::ClassRules => class_rules(conn, limits),
      Action::Cut => cut(conn),
    }
  }
  sweep_backchannel(inner);
}

/// Cut the backchannel's buffers to fresh 8 KiB ones when [`cut_backchannel`] says so, and mark it idle; a held lock
/// skips it for this pass.
fn sweep_backchannel(inner: &RefCount<ClientInner>) {
  let mut guard = match inner.backchannel.transport.try_write() {
    Ok(guard) => guard,
    Err(_) => return,
  };
  if let Some(ref mut transport) = *guard {
    with_framed!(transport.transport, framed => {
      let idle = framed.codec().live_edit.idle.load(Ordering::Relaxed);
      if cut_backchannel(framed.read_buffer().len(), framed.write_buffer().len(), idle) {
        *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
        *framed.write_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
        let state = &framed.codec().live_edit;
        state.largest_write.store(0, Ordering::Relaxed);
        state.largest_frame.store(0, Ordering::Relaxed);
        state.idle.store(true, Ordering::Relaxed);
      }
    })
  }
}

/// A reply was matched to the command popped from the head of `conn`'s in-flight queue: apply the queue rule and the
/// read-buffer rule (this also covers replies read outside the read loop, by a drain), then, when the pop left a
/// connection that is not warm drained, cut it before the reply's caller is answered.
pub(crate) fn on_reply(conn: &mut Connection) {
  if let Some(limits) = limits_of(conn) {
    shrink_queue(&mut conn.buffer, limits.in_flight_commands);
    on_read(conn);
    if action(&view(conn)) == Action::Cut {
      cut(conn);
    }
  }
}

/// A frame was taken off `conn`: replace its read buffer when it is empty and held a frame above the limit.
pub(crate) fn on_read(conn: &mut Connection) {
  with_framed!(conn.transport, framed => {
    let state = &framed.codec().live_edit;
    let over = match state.limits {
      Some(limits) => state.largest_frame.load(Ordering::Relaxed) > limits.read_buffer_bytes,
      None => false,
    };
    if over && framed.read_buffer().is_empty() {
      framed.codec().live_edit.largest_frame.store(0, Ordering::Relaxed);
      *framed.read_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
    }
  })
}

/// A write-and-flush or a flush completed on `conn`: replace its write buffer when it is empty and held more than
/// the limit.
pub(crate) fn on_flushed(conn: &mut Connection) {
  with_framed!(conn.transport, framed => {
    let state = &framed.codec().live_edit;
    let over = match state.limits {
      Some(limits) => state.largest_write.load(Ordering::Relaxed) > limits.write_buffer_bytes,
      None => false,
    };
    if over && framed.write_buffer().is_empty() {
      framed.codec().live_edit.largest_write.store(0, Ordering::Relaxed);
      *framed.write_buffer_mut() = BytesMut::with_capacity(FRESH_BUFFER_BYTES);
    }
  })
}

/// The router replayed its retry buffer after a reconnect: once it is empty, return it to stock's unallocated
/// `VecDeque::new()`. A replay that stopped on a write error leaves its remaining commands in it, and then it is
/// left as is.
pub(crate) fn on_retry_replayed(inner: &RefCount<ClientInner>, retry_buffer: &mut VecDeque<Command>) {
  if inner.connection.live_edit.retention.is_some() && retry_buffer.is_empty() {
    *retry_buffer = VecDeque::new();
  }
}

/// Test support: a per-client report of each connection's retention state, test sweep modes, per-client sweep and
/// write-check counters and a per-thread in-sweep flag. Compiled only with the `live-edit-retention-report` feature
/// (see the module docs' *Test support*).
#[cfg(feature = "live-edit-retention-report")]
pub mod test_support {
  use super::{limits_of, liveness, view, Connection, ConnectionKind, RetentionLimits};
  use crate::{interfaces::ClientLike, modules::inner::ClientInner, router::Router, runtime::RefCount};
  use std::{
    cell::Cell,
    sync::{
      atomic::{AtomicU64, AtomicU8, Ordering},
      Mutex,
      PoisonError,
      TryLockError,
    },
    time::Instant,
  };

  /// How a client's sweep runs, set per client with [`set_sweep_for_test`]. Each mode other than `AsBuilt` models a
  /// broken warm rule, so a test can show its check tells the two apart.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub enum SweepForTest {
    /// The sweep as built.
    #[default]
    AsBuilt,
    /// The sweep clears every connection's warm flag and does nothing else: no class rules, no cuts, no backchannel.
    /// The state a build without the sweep reaches: no warm connection, and every drained connection cut at its
    /// next reply pop.
    Skipped,
    /// The sweep marks every connection warm, applies the class rules to each drained one and sweeps the
    /// backchannel: the per-connection rule alone, with no single warm connection per client.
    EveryConnectionWarm,
  }

  impl SweepForTest {
    fn from_u8(value: u8) -> Self {
      match value {
        1 => SweepForTest::Skipped,
        2 => SweepForTest::EveryConnectionWarm,
        _ => SweepForTest::AsBuilt,
      }
    }

    fn as_u8(self) -> u8 {
      match self {
        SweepForTest::AsBuilt => 0,
        SweepForTest::Skipped => 1,
        SweepForTest::EveryConnectionWarm => 2,
      }
    }
  }

  /// One connection of a client, as the router saw it right after a sweep.
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct ConnectionReport {
    /// The connection's server, `host:port`.
    pub server:         String,
    /// The limits on the connection's codec: its client's class limits, or `None` without limits.
    pub limits:         Option<RetentionLimits>,
    /// Whether it is its client's warm connection.
    pub warm:           bool,
    /// Whether it is in the idle shape and has carried nothing since its last cut.
    pub idle:           bool,
    /// Whether it is drained: nothing queued, nothing in either buffer.
    pub drained:        bool,
    /// Whether it has made read progress (a frame taken off it, or its read buffer grew) since it connected.
    pub read_progress:  bool,
    /// The commands in its in-flight queue (`Connection::buffer`, the commands written and not yet answered) when
    /// the report was taken: exact (`VecDeque::len`).
    pub queued:         usize,
    /// The in-flight queue's capacity, in commands (`VecDeque::capacity`): exact.
    pub queue_capacity: usize,
    /// The write buffer's `BytesMut::capacity()`. Exact only for a buffer nothing has been consumed from, such as
    /// the idle shape's unallocated `BytesMut::new()` (0); otherwise a lower bound on the allocation, since a flush
    /// lowers `capacity()` by the bytes it consumed while the allocation stays whole (the module docs' *Byte
    /// buffers are judged by what they held*).
    pub write_capacity: usize,
    /// The read buffer's `BytesMut::capacity()`. Exact only for a buffer nothing has been consumed from, such as
    /// the idle shape's fresh `with_capacity(FRESH_BUFFER_BYTES)` (8,192); otherwise a lower bound on the
    /// allocation, since a decode lowers `capacity()` by the frame it split off.
    pub read_capacity:  usize,
  }

  /// The per-client report state, on `live_edit::ClientState`: the request count, the request last answered, the
  /// report slot, the sweep mode, and the sweep and write-check counters. Created with the client's `ClientInner` and
  /// dropped with it.
  ///
  /// Requests are numbered rather than flagged so that a request made while a pass is between its pending check and
  /// its answer is not answered by that pass, whose sweep began before it.
  #[derive(Debug, Default)]
  pub struct ReportState {
    /// How many reports have been requested: [`request_report`] adds one, under the slot's lock, with `Release`.
    requests:     AtomicU64,
    /// The request the router last answered. Written only by the client's router.
    answered:     AtomicU64,
    report:       Mutex<Option<Vec<ConnectionReport>>>,
    mode:         AtomicU8,
    /// How many sweeps the client's routers ran. Written only by the router, as each sweep ends ([`SweepPass`]).
    sweep_passes: AtomicU64,
    /// The wall time those sweeps took, in nanoseconds. Written with `sweep_passes`.
    sweep_nanos:  AtomicU64,
    /// How many write checks the client's routers ran ([`WriteCheck`]). Written only by the router, as each check
    /// ends.
    write_checks: AtomicU64,
    /// The wall time of those checks' decisions, in nanoseconds. Written with `write_checks`.
    check_nanos:  AtomicU64,
    /// How many of those checks went on to flush. Written with `write_checks`.
    flushes:      AtomicU64,
    /// The wall time of those flushes, in nanoseconds. Written with `write_checks`.
    flush_nanos:  AtomicU64,
  }

  /// A client's sweep counters ([`sweep_stats`]): how many sweeps its routers ran since the client was created, and
  /// the wall time they took. A test takes two readings and divides the differences for the time per pass.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub struct SweepStats {
    /// Sweeps run, in every sweep mode.
    pub passes: u64,
    /// Their total wall time, in nanoseconds: from just after the sweep finds the client's limits to its return,
    /// the backchannel's cut included.
    pub nanos:  u64,
  }

  /// A client's write-check counters ([`write_check_stats`]): how many checks before a write
  /// (`liveness::before_socket_write`, `liveness::before_write_to`) its routers ran on a clustered client since the
  /// client was created, the wall time of their decisions, and how many went on to flush, with those flushes' time. A
  /// test takes two readings and divides the differences for the time per check.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub struct WriteCheckStats {
    /// Checks run: every call on a clustered client, the early returns included (no route, a write that cannot
    /// reach the socket, no other connection holding fed frames).
    pub checks:      u64,
    /// Their decisions' total wall time, in nanoseconds: from the check's first line to its return, or, for a check
    /// that flushes, to the start of that flush. Routing, the write predicate and the walk of the other connections
    /// are all inside it; the flush is not.
    pub nanos:       u64,
    /// Checks that went on to flush every connection holding fed frames.
    pub flushes:     u64,
    /// Those flushes' total wall time, in nanoseconds: finding the destination's name, then the concurrent flush
    /// with its socket waits.
    pub flush_nanos: u64,
  }

  /// One write check in progress, held by `liveness::before_socket_write` and `liveness::before_write_to` from their
  /// first line (each runs only on a clustered client): reads the clock when made. [`WriteCheck::flushing`] ends the
  /// decision's time where the check goes on to flush; when dropped, as the check returns or its future is dropped,
  /// it counts the check, its decision's time and, when it flushed, the flush and its time ([`write_check_stats`]).
  /// Neither end allocates.
  pub(crate) struct WriteCheck<'a> {
    state:         &'a ReportState,
    started:       Instant,
    flush_started: Option<Instant>,
  }

  impl<'a> WriteCheck<'a> {
    /// Begin a write check on the client `inner`.
    pub(crate) fn begin(inner: &'a RefCount<ClientInner>) -> Self {
      WriteCheck {
        state:         &inner.live_edit.retention,
        started:       Instant::now(),
        flush_started: None,
      }
    }

    /// The check decided to flush: its decision ends here, and the flush's time starts.
    pub(crate) fn flushing(&mut self) {
      let now = Instant::now();
      add_nanos(&self.state.check_nanos, now.saturating_duration_since(self.started));
      self.flush_started = Some(now);
    }
  }

  impl Drop for WriteCheck<'_> {
    fn drop(&mut self) {
      match self.flush_started {
        Some(flush_started) => {
          add_nanos(&self.state.flush_nanos, flush_started.elapsed());
          self.state.flushes.fetch_add(1, Ordering::Relaxed);
        },
        None => add_nanos(&self.state.check_nanos, self.started.elapsed()),
      }
      self.state.write_checks.fetch_add(1, Ordering::Relaxed);
    }
  }

  fn add_nanos(counter: &AtomicU64, took: std::time::Duration) {
    counter.fetch_add(u64::try_from(took.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
  }

  thread_local! {
    /// Whether a sweep is running on this thread ([`in_sweep`]). Const-initialised and without a destructor, so
    /// reading or setting it never allocates or registers anything, even from inside an allocator.
    static IN_SWEEP: Cell<bool> = const { Cell::new(false) };
  }

  /// Whether a sweep is running on the calling thread: true from just after the sweep finds the client's limits to
  /// its return, on the router's thread, and false everywhere else. The sweep is synchronous, so no other task runs
  /// on that thread in between. For a test's counting global allocator, which reads it on every allocation to count
  /// those the sweep makes; it never allocates or panics.
  pub fn in_sweep() -> bool {
    IN_SWEEP.try_with(Cell::get).unwrap_or(false)
  }

  /// One sweep in progress, held by `sweep` from just after it finds the client's limits: sets the in-sweep flag
  /// ([`in_sweep`]) and reads the clock when made; when dropped, as `sweep` returns or unwinds, counts the pass and
  /// its time on the client ([`sweep_stats`]) and clears the flag. Neither end allocates.
  pub(crate) struct SweepPass<'a> {
    state:   &'a ReportState,
    started: Instant,
  }

  impl<'a> SweepPass<'a> {
    /// Begin a sweep of the client `inner`.
    pub(crate) fn begin(inner: &'a RefCount<ClientInner>) -> Self {
      IN_SWEEP.with(|flag| flag.set(true));
      SweepPass {
        state:   &inner.live_edit.retention,
        started: Instant::now(),
      }
    }
  }

  impl Drop for SweepPass<'_> {
    fn drop(&mut self) {
      let nanos = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
      self.state.sweep_nanos.fetch_add(nanos, Ordering::Relaxed);
      self.state.sweep_passes.fetch_add(1, Ordering::Relaxed);
      IN_SWEEP.with(|flag| flag.set(false));
    }
  }

  fn state<C: ClientLike>(client: &C) -> &ReportState {
    &client.inner().live_edit.retention
  }

  /// Ask `client`'s router for a report: clears any report not yet taken and makes a new request, which supersedes
  /// any earlier one still pending. Take the answer with [`take_report`].
  ///
  /// The answer comes from the first router pass whose sweep began after this call, and shows what that sweep left.
  /// So a mode set with [`set_sweep_for_test`] before this call is the mode that sweep ran: the router reads the
  /// request (`Acquire`) before its sweep reads the mode, and this call's `Release` store comes after the mode's. A
  /// pass that finds the report slot held by a test thread leaves the request for the next pass, whose sweep also
  /// began after it, so the router never waits on a test thread.
  ///
  /// The router passes at least once per unresponsive-check interval while it waits, so a request is answered within
  /// about one interval when the check is configured, with or without traffic. A router that has ended never answers.
  ///
  /// Pass a client (a `Client`, a pool member, a `SubscriberClient`), never a `Pool`: a `Pool`'s `ClientLike::inner`
  /// picks a member round-robin, so the request and the take could reach different clients.
  pub fn request_report<C: ClientLike>(client: &C) {
    let state = state(client);
    let mut slot = state.report.lock().unwrap_or_else(PoisonError::into_inner);
    *slot = None;
    state.requests.fetch_add(1, Ordering::Release);
  }

  /// The report `client`'s router stored for the last [`request_report`], if it has answered; taking it empties the
  /// slot. Pass a client, never a `Pool` (see [`request_report`]).
  pub fn take_report<C: ClientLike>(client: &C) -> Option<Vec<ConnectionReport>> {
    state(client).report.lock().unwrap_or_else(PoisonError::into_inner).take()
  }

  /// `client`'s sweep counters ([`SweepStats`]), read now. They count from the client's creation, across
  /// reconnects and in every sweep mode, so a test reads them before and after the work it measures. Pass a client,
  /// never a `Pool` (see [`request_report`]).
  pub fn sweep_stats<C: ClientLike>(client: &C) -> SweepStats {
    let state = state(client);
    SweepStats {
      passes: state.sweep_passes.load(Ordering::Relaxed),
      nanos:  state.sweep_nanos.load(Ordering::Relaxed),
    }
  }

  /// `client`'s write-check counters ([`WriteCheckStats`]), read now. They count from the client's creation, across
  /// reconnects, so a test reads them before and after the work it measures. Pass a client, never a `Pool` (see
  /// [`request_report`]).
  pub fn write_check_stats<C: ClientLike>(client: &C) -> WriteCheckStats {
    let state = state(client);
    WriteCheckStats {
      checks:      state.write_checks.load(Ordering::Relaxed),
      nanos:       state.check_nanos.load(Ordering::Relaxed),
      flushes:     state.flushes.load(Ordering::Relaxed),
      flush_nanos: state.flush_nanos.load(Ordering::Relaxed),
    }
  }

  /// Set how `client`'s sweep runs ([`SweepForTest`]); it stays until set again, across reconnects, for the life of
  /// the client. Every sweep that begins after a later [`request_report`] call runs under it, so a report requested
  /// after this call shows a sweep in this mode; a pass already running may still sweep under the previous one. Pass a
  /// client, never a `Pool` (see [`request_report`]).
  pub fn set_sweep_for_test<C: ClientLike>(client: &C, mode: SweepForTest) {
    state(client).mode.store(mode.as_u8(), Ordering::Release);
  }

  /// The sweep mode of the client `inner`, read at the top of `sweep`.
  pub(crate) fn sweep_mode(inner: &RefCount<ClientInner>) -> SweepForTest {
    SweepForTest::from_u8(inner.live_edit.retention.mode.load(Ordering::Acquire))
  }

  /// The request pending at the top of a router pass (`live_edit::before_wait`, before the sweep), if any. Its
  /// `Acquire` load makes every mode set before that request visible to the sweep that follows. Two atomic loads and
  /// no allocation when none is pending.
  pub(crate) fn pending_report(inner: &RefCount<ClientInner>) -> Option<u64> {
    let state = &inner.live_edit.retention;
    let request = state.requests.load(Ordering::Acquire);
    (request != state.answered.load(Ordering::Relaxed)).then_some(request)
  }

  /// Answer `request`, which [`pending_report`] found before this pass's sweep (`live_edit::before_wait`, right after
  /// the sweep): one [`ConnectionReport`] per connection of the router. Left pending for the next pass when a test
  /// thread holds the report slot, or when a newer request was made since the pass began (its sweep began before that
  /// request).
  pub(crate) fn answer_report(inner: &RefCount<ClientInner>, router: &mut Router, request: u64) {
    let state = &inner.live_edit.retention;
    let mut slot = match state.report.try_lock() {
      Ok(slot) => slot,
      Err(TryLockError::WouldBlock) => return,
      Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
    };
    // Requests are counted under this lock, so this read sees the latest one.
    if state.requests.load(Ordering::Relaxed) != request {
      return;
    }
    let report = liveness::connections_mut(router).map(report_of).collect();
    *slot = Some(report);
    state.answered.store(request, Ordering::Relaxed);
  }

  fn report_of(conn: &mut Connection) -> ConnectionReport {
    let seen = view(conn);
    let limits = limits_of(conn);
    let server = conn.server.to_string();
    let queue_capacity = conn.buffer.capacity();
    let (write_capacity, read_capacity) =
      with_framed!(conn.transport, framed => (framed.write_buffer().capacity(), framed.read_buffer().capacity()));
    ConnectionReport {
      server,
      limits,
      warm: seen.warm,
      idle: seen.idle,
      drained: seen.drained(),
      read_progress: seen.last_progress.is_some(),
      queued: seen.queued,
      queue_capacity,
      write_capacity,
      read_capacity,
    }
  }
}
