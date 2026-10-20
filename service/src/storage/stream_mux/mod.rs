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

//! XREAD multiplexer for layer-stream subscriptions.
//!
//! A fixed set of readers (`LIVE_EDIT_SUBSCRIBER_POOL_SIZE`) each run one
//! `XREAD BLOCK … STREAMS k1..kn` over every stream its subscriptions need and
//! fan entries out in-process, so subscriptions cost memory, not connections.
//! Keys are routed to readers by [`router::RouterCore`]; each reader's
//! per-subscription state is the pure [`core::ReaderCore`]; `reader.rs` is the
//! async loop.
//!
//! **Reader connections are lazy, and on a cluster single-node.** A reader
//! holds no connection until it first serves a subscription: startup opens
//! none, so a new replica's connect and `AUTH` burst does not grow with the
//! reader count. Each connection is a single-node fred client built by
//! [`ReaderConnector`](crate::storage::client::ReaderConnector). On a cluster a
//! reader serves one hash slot at a time, so its connection goes to that slot's
//! primary alone — the slot owner's `Server` taken whole from the readers'
//! routing view (data-pool member 0's current client's, the only view the
//! readers read and sync) — not to every primary. The connection must target the bound
//! slot's current owner, so the reader moves it (a **retarget**: the old
//! client is retired, a new one connected) when:
//! - it binds a slot another primary owns (checked before the first XREAD of a
//!   binding, with the idle→active probe);
//! - an XREAD, head check or probe fails `-MOVED`
//!   ([`is_moved_redirect`](crate::storage::client::is_moved_redirect)): the
//!   reader re-resolves the owner — syncing the readers' routing view once
//!   if that view still names the current node; the sync is single-flight
//!   across the readers, so readers redirected together cost one (one per
//!   bound period while syncs outlive their bound) — and
//!   retargets when the owner changed, else backs off as for any transient
//!   failure. The first retarget since the reader's last successful read is
//!   immediate and free; each later one before a read succeeds is charged to
//!   the transient budget and backed off first, so a routing view flipping
//!   during a failover cannot spin a reader. A single-node client cannot
//!   follow a redirect, and without this it would retry the same node until
//!   its transient budget ran out;
//! - a forced reconnect is due on every phase but the idle keepalive and the
//!   slot's owner has changed (a failover): it retargets instead of
//!   reconnecting the old node.
//!
//! An idle reader whose node is no longer a primary retires its client instead
//! of reconnecting it. `-ASK` and `-TRYAGAIN` stay transient, through the
//! shared classifier (`storage::client::is_transient_fred_error`). Off a cluster
//! (standalone, sentinel, unix socket) a reader connects to the URL's endpoint;
//! only the laziness applies. Router node affinity
//! ([`router::RouterCore::route_near`]) prefers a free reader already connected
//! to a new binding's owner, so most bindings need no retarget.
//!
//! A reader's connection lives in its slot on the shared state, outside the
//! reader loop's panic boundary, so a restarted reader keeps it and
//! [`StreamMux::quit_readers`] reaches it at shutdown. Each connection's router
//! is supervised from its first successful connect and its metric hooks are
//! armed then; both end when the reader retires it (`storage::router_supervisor`).

pub mod core;
mod reader;
pub mod router;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use fred::clients::{Client, Pool};
use fred::types::config::Server;
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::error::ServiceError;
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{ReaderConn, ReaderConnector};
use crate::storage::keys::KeySchema;
use crate::storage::notifications::{
    ConsumerBacklog, ConsumerTerminalExit, LayerNotification, LayerStreamConsumer, StreamCursor,
    CONSUMER_BACKLOG_BYTES, CONSUMER_CHANNEL_CAPACITY, CONSUMER_SEND_WATCHDOG, XREAD_BLOCK_MS,
};

use self::core::{Failure, Offer, OverflowBound, StreamId, SubId, SubSink};
use self::router::{RouterCore, Routing};

/// The most entries requested per stream per XREAD: the ceiling of the
/// byte-sized `COUNT` ([`core::ReadSizing`]).
pub const XREAD_COUNT: u64 = 512;
/// Bytes one reader's XREAD reply aims at. Each read's `COUNT` is sized before
/// it is issued so that `COUNT × Σ(per-key entry estimate)` stays within this
/// ([`core::ReadSizing`]); a replica's reply memory is then about readers ×
/// this (256 MiB at the Helm chart's `LIVE_EDIT_SUBSCRIBER_POOL_SIZE` of 256).
/// Soft: an entry larger than its key's estimate over-commits one reply.
pub const XREAD_REPLY_TARGET_BYTES: u64 = 1024 * 1024;
/// Entry-size estimate for a key the reader has no non-empty reply for yet,
/// unless a key it already reads has a larger one. Deliberately above the
/// captured Kit traffic (`tools/fixtures/gze_inbound_deltas.json`: 128 B to
/// 1.6 KB of OMPF per commit), so a new key's first read is not oversized.
pub const DEFAULT_ENTRY_ESTIMATE: u64 = 16 * 1024;

/// The most entries one paged `XRANGE` of a layer stream requests
/// ([`paged_xrange_count`]): the ceiling the checkpoint tail capture always
/// paged at, and the multiplexer's own [`XREAD_COUNT`].
pub const PAGED_XRANGE_MAX_COUNT: u64 = 512;

/// The `COUNT` for the next page of a paged `XRANGE` over one layer stream —
/// the shared helper of the bounded stream reads outside the multiplexer
/// (`SubscribeReadLayer`'s rollback-mode in-window replay,
/// `storage::tail_slice::read_tail_slice`).
///
/// The multiplexer's soft [`core::ReadSizing`] rule for one key: one page aims
/// at [`XREAD_REPLY_TARGET_BYTES`], sized from `largest_entry_bytes`, the
/// largest entry the read has seen so far as [`stream_entry_bytes`] counts it
/// (`None` before the first page, which then uses [`DEFAULT_ENTRY_ESTIMATE`]),
/// clamped to `1..=`[`PAGED_XRANGE_MAX_COUNT`]. Soft: a page of entries larger
/// than any seen before over-commits that one page, after which the estimate
/// follows them.
///
/// This helper sizes pages; it does not make a paged read atomic. Each page is
/// one atomic `XRANGE`, but a head trim between two pages (`MAXLEN ~` on any
/// concurrent `XADD`, or the relay byte budget's `XTRIM MINID ~`) can remove
/// entries past the cursor, and the next page then starts beyond a hole. More
/// pages leave more such windows, so a caller that needs the whole range must
/// detect the hole itself: the tail capture by its classifier's `+1`
/// progression check, the in-window replay by re-checking that its first entry
/// is still held after its last page.
pub fn paged_xrange_count(largest_entry_bytes: Option<u64>) -> u64 {
    core::ReadSizing {
        target_bytes: XREAD_REPLY_TARGET_BYTES,
        max_count: PAGED_XRANGE_MAX_COUNT,
        default_entry_estimate: DEFAULT_ENTRY_ESTIMATE,
    }
    .count_for(largest_entry_bytes.unwrap_or(DEFAULT_ENTRY_ESTIMATE))
}

/// One layer-stream entry's size as the read-size estimates count it: its id
/// plus the values of its `payload` and `ompf` fields.
pub fn stream_entry_bytes(
    id: &str,
    fields: &std::collections::HashMap<String, bytes::Bytes>,
) -> u64 {
    use crate::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};
    let field = |name: &str| fields.get(name).map_or(0, bytes::Bytes::len);
    (id.len() + field(NOTIFICATION_PAYLOAD_FIELD) + field(RELAY_OMPF_FIELD)) as u64
}
/// A reader whose XREAD has not returned this long after its wake XADD
/// **landed** is treated as wedged: the no-answer verdict shares the reader's
/// timeout streak with XREAD timeouts, and force-reconnects the reader
/// (`valkey_consumer_forced_reconnects_total{phase="wake_ack"}`) on the first
/// and every 8th verdict of a streak; otherwise it is a WARN. The clock starts
/// when the wake task reports the landing back to the reader, not when the
/// wake is sent, and a wake that never lands arms no deadline at all.
///
/// Also the liveness age ([`StreamMuxKnobs::liveness_age`]): a joining
/// subscription the in-flight read already covers still wakes it when the
/// reader's connection has not answered within this, or has failed since —
/// at the join, or, for a join made between reads, as the next read is
/// issued. So every join either comes at most this long after the
/// connection's last answer with nothing failing since, or is checked: the
/// woken read must return within this of the wake landing (a wake that fails
/// to land gives no verdict; the XREAD's own timeout still applies). That
/// verdict shares the reader's timeout streak with XREAD timeouts: it
/// force-reconnects on the first and every 8th no-answer verdict of a streak
/// and is otherwise a WARN.
pub const WAKE_ACK_TIMEOUT: Duration = Duration::from_secs(3);
/// How often a connected reader with nothing to read (no subscription, no
/// XREAD in flight) `PING`s its one connection, so the backing store's
/// idle-connection timeout never closes it:
/// a third of Azure Managed Redis's 10-minute idle timeout, so a connection
/// survives two missed ticks. Each reader starts at its own random offset
/// within the first interval and then ticks on that phase. A failed tick
/// force-reconnects the reader
/// (`valkey_consumer_forced_reconnects_total{phase="keepalive"}`), or on a
/// cluster retires its connection when its node is no longer a primary. A
/// reader with an XREAD in flight is never pinged: a reader sends its other
/// commands — this `PING`, its probes, head checks and credential re-AUTH —
/// only between reads, since a command queued behind a live read waits out its
/// `BLOCK` (or, under `Blocking::Interrupt`, `CLIENT UNBLOCK`s it). A reader
/// with no connection has nothing to keep alive and does not tick.
pub const READER_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(200);
/// Bound on the `PING` a reader sends before its first XREAD after it had
/// nothing to read, and on a cluster before its first XREAD bound to a
/// different hash slot than its previous read (the idle→active probe), on its
/// connection — after it connected, or retargeted to the slot's owner, if it
/// had to. A probe that does not answer in time
/// force-reconnects the reader
/// (`valkey_consumer_forced_reconnects_total{phase="probe"}`) and is retried
/// under the reader's transient-failure budget.
pub const READER_PROBE_TIMEOUT: Duration = Duration::from_secs(3);
/// Multiple of the soft backlog thresholds (`CONSUMER_CHANNEL_CAPACITY`
/// entries, `CONSUMER_BACKLOG_BYTES` bytes) at which a subscription is ended
/// immediately instead of after the watchdog: 16,384 entries or 32 MiB. The
/// hard caps apply from registration; the watchdog only once the consumer
/// drains.
pub const BACKLOG_HARD_CAP_FACTOR: usize = 4;
/// TTL of a reader's wake stream; set atomically with every wake XADD by
/// `scripts/stream_mux_wake.lua`, so the key never exists without one.
pub const WAKE_KEY_TTL_MS: i64 = 3_600_000;

/// Tunables. Production uses [`StreamMuxKnobs::new`]; tests override.
///
/// How long a subscription may wait for a free reader (cluster mode) is not a
/// knob: each [`StreamMux::reserve`] (and so [`StreamMux::consume`]) call
/// carries its own deadline, taken from the subscription's capacity budget.
#[derive(Debug, Clone, Copy)]
pub struct StreamMuxKnobs {
    pub block_ms: u64,
    /// The most entries requested per stream per XREAD ([`XREAD_COUNT`]).
    pub count: u64,
    /// Bytes one XREAD reply aims at ([`XREAD_REPLY_TARGET_BYTES`]).
    pub reply_target_bytes: u64,
    /// Soft backlog threshold in entries (`CONSUMER_CHANNEL_CAPACITY`).
    pub channel_capacity: usize,
    pub send_watchdog: Duration,
    /// Hard backlog cap in entries.
    pub backlog_hard_cap: usize,
    /// Soft backlog threshold in bytes (`CONSUMER_BACKLOG_BYTES`).
    pub backlog_bytes: u64,
    /// Hard backlog cap in bytes.
    pub backlog_hard_bytes: u64,
    pub max_transient_failures: u32,
    pub backoff_initial: Duration,
    pub backoff_cap: Duration,
    pub wake_ack_timeout: Duration,
    /// One overall bound on a reader's concurrent probes of its keys, and on
    /// its keepalive: the `TYPE` probes of a poison-key isolation, the head
    /// checks of one reply, and the `PING` of one idle keepalive tick
    /// (`PROBE_TIMEOUT`, 2 s). Also bounds each step of a credential re-AUTH:
    /// the tick's `fetch(None)`, and the `AUTH`. The idle→active connection
    /// probe has its own bound, [`Self::reader_probe_timeout`].
    pub probe_timeout: Duration,
    /// Bound on the idle→active connection probe ([`READER_PROBE_TIMEOUT`],
    /// 3 s).
    pub reader_probe_timeout: Duration,
    /// Interval of the idle keepalive ([`READER_KEEPALIVE_INTERVAL`]); `None`
    /// turns the keepalive off. A zero interval is read as 1 ms.
    pub keepalive_interval: Option<Duration>,
    /// How recently the reader's connection must have answered, with nothing
    /// failing since, for a joining subscription the in-flight XREAD already
    /// covers not to wake it. Past that, the join wakes the read so it gets a
    /// connection verdict through the wake-ack watchdog (the liveness wake;
    /// [`WAKE_ACK_TIMEOUT`], 3 s).
    pub liveness_age: Duration,
}

impl StreamMuxKnobs {
    pub fn new() -> Self {
        Self {
            block_ms: XREAD_BLOCK_MS,
            count: XREAD_COUNT,
            reply_target_bytes: XREAD_REPLY_TARGET_BYTES,
            channel_capacity: CONSUMER_CHANNEL_CAPACITY,
            send_watchdog: CONSUMER_SEND_WATCHDOG,
            backlog_hard_cap: CONSUMER_CHANNEL_CAPACITY * BACKLOG_HARD_CAP_FACTOR,
            backlog_bytes: CONSUMER_BACKLOG_BYTES,
            backlog_hard_bytes: CONSUMER_BACKLOG_BYTES * BACKLOG_HARD_CAP_FACTOR as u64,
            max_transient_failures: reader::MAX_CONSECUTIVE_TRANSIENT_FAILURES,
            backoff_initial: reader::XREAD_TRANSIENT_BACKOFF_INITIAL,
            backoff_cap: reader::XREAD_TRANSIENT_BACKOFF_CAP,
            wake_ack_timeout: WAKE_ACK_TIMEOUT,
            probe_timeout: reader::PROBE_TIMEOUT,
            reader_probe_timeout: READER_PROBE_TIMEOUT,
            keepalive_interval: Some(READER_KEEPALIVE_INTERVAL),
            liveness_age: WAKE_ACK_TIMEOUT,
        }
    }
}

impl Default for StreamMuxKnobs {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConsumeError {
    #[error(
        "every stream reader is bound to another hash slot \
         (raise LIVE_EDIT_SUBSCRIBER_POOL_SIZE)"
    )]
    Saturated,
    #[error("the stream multiplexer is shutting down")]
    ShuttingDown,
    #[error("invalid stream cursor {0:?}")]
    InvalidCursor(String),
}

/// Where a test-injected [`MuxFault`] lands. Each site has its own FIFO queue
/// on the multiplexer ([`StreamMux::push_faults_for_test`]); the next
/// operation any reader performs at that site consumes the queue's head, and
/// an empty queue means the real command runs.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MuxFaultSite {
    /// A reader's multiplexed `XREAD BLOCK`.
    Xread,
    /// One attempt of a wake — one `stream_mux_wake.lua` EVALSHA — whether it
    /// wakes an in-flight read for a new subscription or wakes it at shutdown.
    /// A wake makes up to three attempts, so three faults fail one wake.
    Wake,
    /// One key's head check (`XRANGE key - + COUNT 1`), run when a reply's
    /// first entry for that key is past the entry after the id it was read
    /// from. The head checks of one reply run concurrently, one per key, so
    /// each consumes one fault. A `Hang` is bounded by the knobs'
    /// `probe_timeout`, after which the check counts as failed.
    HeadCheck,
    /// The idle→active connection probe: the one `PING` a reader sends before
    /// its first XREAD after it had nothing to read, and on a cluster before
    /// its first XREAD bound to a different hash slot. A `Hang` is bounded by
    /// the knobs' `reader_probe_timeout`, after which the probe counts as
    /// timed out.
    Probe,
    /// The `PING` of an idle keepalive tick: one per tick, on the reader's one
    /// connection, so each tick consumes one fault. A reader with no
    /// connection does not tick. A `Hang` is bounded by the knobs'
    /// `probe_timeout`, after which the `PING` counts as failed.
    Keepalive,
    /// The `AUTH` of a reader's credential re-AUTH (b.19i), sent on its one
    /// connection between reads once a tick fetched a credential: one per
    /// `AUTH`, so each consumes one fault. Not consumed by a tick that fetched
    /// no credential, nor when the reader's client is not connected (the
    /// `AUTH` is skipped). A `Hang` is bounded by the knobs' `probe_timeout`,
    /// after which the `AUTH` counts as failed.
    Reauth,
    /// One attempt to connect a reader (`ReaderConnector::connect`, from the
    /// connection step before a binding's first XREAD): one per attempt, so
    /// each consumes one fault, whether its `cause` is `bind` or `retarget`.
    /// Only a reader's own connect consumes it — no pooled or pub/sub
    /// connection does. The reader's real client is still built and its
    /// router started; the fault replaces, or (a `Hold`) delays, only what
    /// the connect then reports about its connection's setup, so everything
    /// after that runs as for a real connect: the cancellation race, the
    /// `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` bound, the router aborted on
    /// a failed, cut or cancelled connect,
    /// `live_edit_stream_mux_reader_connects_total{cause, result}`, the
    /// steady-state credential hint for a `WRONGPASS`, and the reader's
    /// classification and WARN. An `Err` is that setup's error, classified
    /// as a real one is (`is_transient_fred_error`: an `IO`, `Canceled` or
    /// `Timeout` kind is transient; `Auth`, `Tls` or `Config` is permanent),
    /// and an `Auth` error whose details start `WRONGPASS` — not `NOAUTH` —
    /// logs the hint, as a server's reply would. A `Hang` is a setup that
    /// never reports back, cut at the connect bound with the same `Timeout`
    /// a real stalled setup gets. Under either, the real connection's setup
    /// runs on underneath and may complete; the router's abort closes it. A
    /// `Hold` reports the real setup's own outcome, but only once the test
    /// releases it; a hold that outlasts the connect bound is cut as a
    /// `Timeout`, as a `Hang` is.
    ///
    /// On a cluster, an attempt that fails before its setup begins — its
    /// routing view names no owner for the bound slot — consumes its fault
    /// unused.
    Connect,
}

/// One test-injected outcome at a [`MuxFaultSite`]. `Pass` runs the real
/// command.
///
/// `Reply` and `Panic` apply to [`MuxFaultSite::Xread`] only, and `Hold` to
/// [`MuxFaultSite::Connect`] only; [`StreamMux::push_faults_for_test`]
/// refuses each at any other site.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
pub enum MuxFault {
    Pass,
    Err(fred::error::Error),
    /// Nothing is sent to Valkey and the operation does not resolve on its
    /// own — except at [`MuxFaultSite::Connect`], where the reader's real
    /// client is still built and its router started, and that router runs
    /// the real connection handshake (and may finish it) until the connect
    /// bound or the reader's cancellation aborts it; only the connect's
    /// report of the setup never resolves.
    ///
    /// - At [`MuxFaultSite::Xread`] it models a read on a wedged reader
    ///   connection **without** the client-side timeout a real read carries
    ///   (`block_ms + 5 s`, after which fred fails it with `Timeout` and the
    ///   reader takes the `phase="xread"` wedge path). Test that path with
    ///   `MuxFault::Err` of `ErrorKind::Timeout` instead. A hung read ends only
    ///   when that reader force-reconnects (any phase) — or, where the forced
    ///   reconnect finds its slot's owner moved, retires its connection to
    ///   retarget — failing with the error
    ///   fred 10.1.0 returns for an in-flight `max_attempts = 1` command
    ///   replayed after a reconnect — `BUDGET_EXHAUSTED_SHAPES[0]`,
    ///   `ErrorKind::Unknown`, `"Too many failed write attempts."`
    ///   (`protocol/command.rs:1778`, reached from `Router::retry_buffer` →
    ///   `write_command`), which the reader retries as transient — or at
    ///   shutdown (the reader's cancellation).
    /// - At [`MuxFaultSite::Wake`] the attempt never ends; only the shutdown
    ///   wake's timeout bounds it. A hung wake never reports back, so its read
    ///   keeps an unlanded wake until it returns: no wake-ack deadline and no
    ///   second wake.
    /// - At [`MuxFaultSite::HeadCheck`] the check never answers; the head
    ///   checks' `probe_timeout` ends it as a failed check.
    /// - At [`MuxFaultSite::Probe`] the probe never answers; its
    ///   `reader_probe_timeout` ends it as timed out.
    /// - At [`MuxFaultSite::Keepalive`] the tick's `PING` never answers; its
    ///   `probe_timeout` ends it as failed.
    /// - At [`MuxFaultSite::Reauth`] the `AUTH` never answers; its
    ///   `probe_timeout` ends it as failed.
    /// - At [`MuxFaultSite::Connect`] the connection's setup never reports
    ///   back; `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` cuts it as a `Timeout`
    ///   (or the reader's cancellation, at shutdown, if that comes first),
    ///   and either way the router the connect started is aborted.
    Hang,
    Reply(fred::types::Value),
    Panic,
    /// At [`MuxFaultSite::Connect`] only: the connect reports its real
    /// connection setup's outcome, success or failure, but only once the test
    /// releases the hold by sending on, or dropping, the paired
    /// [`tokio::sync::oneshot::Sender`]. Until then the reader stays inside
    /// its connection step, so control messages queued meanwhile wait for it
    /// to come out. Everything else runs as for a real connect: a release
    /// that does not come within `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` is
    /// cut as a `Timeout`, the reader's cancellation wins at shutdown, the
    /// router is aborted on a failed, cut or cancelled connect, and the
    /// outcome is counted on `live_edit_stream_mux_reader_connects_total` as
    /// a real connect's is.
    Hold(tokio::sync::oneshot::Receiver<()>),
}

pub(crate) enum Ctl {
    Add {
        sub: SubId,
        key: String,
        cursor: StreamId,
        sink: ChannelSink,
        key_schema: KeySchema,
    },
    Remove {
        sub: SubId,
    },
    /// The wake sent for the reader's read number `read` landed at `at`; that
    /// read must now return within [`StreamMuxKnobs::wake_ack_timeout`] of `at`.
    /// Ignored when `read` is no longer the read in flight.
    WakeLanded {
        read: u64,
        at: tokio::time::Instant,
    },
    /// Every attempt of the wake sent for read number `read` failed. The read
    /// is not woken and draws no wedge verdict; the reader forgets the wake,
    /// so the next subscription that joins that read sends a fresh one.
    /// Ignored when `read` is no longer the read in flight.
    WakeFailed {
        read: u64,
    },
}

type TerminalSlot = Arc<StdMutex<Option<ConsumerTerminalExit>>>;

/// One item queued for a subscription.
pub(crate) enum Delivery {
    /// A stream entry: its stream id, its notification, and the size its sink
    /// counted for it ([`core::Entry::bytes`]), which the consumer subtracts
    /// when it receives the entry — exactly what was added.
    Entry {
        id: String,
        notification: LayerNotification,
        bytes: u64,
    },
    /// A trim marker ([`SubSink::offer_trimmed`]) carrying `resumes_at`: the
    /// id of the key's first entry in the same reply, whether or not it can be
    /// delivered (an entry that cannot be converted is skipped, and the
    /// subscription receives the next one), queued ahead of the entries that
    /// reply delivers to the subscription. The stream no longer held the ids
    /// between the subscription's cursor and `resumes_at` when the reader read
    /// them. It counts against no backlog limit, in entries or bytes, and
    /// never starts or clears the watchdog clock. The consumer decides from it
    /// whether the subscription ends `stream_gap`
    /// ([`LayerStreamConsumer::set_caught_up_floor`]).
    Trimmed(StreamId),
}

/// Production sink: an unbounded channel whose backlog is bounded in entries
/// and in bytes by the sink's own limits rather than by blocking the reader.
///
/// - **Hard caps** (`hard_cap` entries, `hard_bytes` bytes) apply from
///   registration: an entry that would take the queue past either is refused
///   and the subscription ends. The queue therefore never holds more than
///   either cap — except that an empty queue always accepts one entry, so a
///   single oversized update never ends a caught-up subscriber.
/// - **Soft thresholds** (`capacity` entries, `soft_bytes` bytes) apply only
///   once the consumer is `draining` (its first `recv`/`try_recv`): the
///   watchdog clock starts at the first offer that finds the queue at or above
///   either, and a queue still there `watchdog` later ends the subscription. A
///   queue below both clears the clock. Before the first poll — a catch-up
///   building while its stream buffers — only the hard caps apply.
///
/// Either ending is an [`Offer::Overflow`] whose [`OverflowBound`] is `Bytes`
/// only when a byte limit was reached and the entry limit was not. Trim
/// markers ([`Delivery::Trimmed`]) are outside both rules.
pub(crate) struct ChannelSink {
    tx: mpsc::UnboundedSender<Delivery>,
    backlog: Arc<AtomicUsize>,
    backlog_bytes: Arc<AtomicU64>,
    draining: Arc<AtomicBool>,
    terminal: TerminalSlot,
    over_since: Option<Instant>,
    capacity: usize,
    hard_cap: usize,
    soft_bytes: u64,
    hard_bytes: u64,
    watchdog: Duration,
}

impl ChannelSink {
    /// A sink sized by `knobs`, and the consumer's end of it: the channel's
    /// receiver, the shared backlog counters and the terminal-exit slot.
    fn new(
        knobs: &StreamMuxKnobs,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<Delivery>,
        ConsumerBacklog,
        TerminalSlot,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let backlog = ConsumerBacklog {
            entries: Arc::new(AtomicUsize::new(0)),
            bytes: Arc::new(AtomicU64::new(0)),
            draining: Arc::new(AtomicBool::new(false)),
        };
        let terminal: TerminalSlot = Arc::new(StdMutex::new(None));
        let sink = ChannelSink {
            tx,
            backlog: backlog.entries.clone(),
            backlog_bytes: backlog.bytes.clone(),
            draining: backlog.draining.clone(),
            terminal: terminal.clone(),
            over_since: None,
            capacity: knobs.channel_capacity,
            hard_cap: knobs.backlog_hard_cap,
            soft_bytes: knobs.backlog_bytes,
            hard_bytes: knobs.backlog_hard_bytes,
            watchdog: knobs.send_watchdog,
        };
        (sink, rx, backlog, terminal)
    }
}

impl SubSink<LayerNotification> for ChannelSink {
    fn offer(&mut self, id: &str, payload: &LayerNotification, bytes: u64) -> Offer {
        if self.tx.is_closed() {
            return Offer::Closed;
        }
        let held = self.backlog.load(Ordering::Acquire);
        let held_bytes = self.backlog_bytes.load(Ordering::Acquire);
        if held > 0 {
            if held >= self.hard_cap {
                return Offer::Overflow(OverflowBound::Entries);
            }
            if held_bytes.saturating_add(bytes) > self.hard_bytes {
                return Offer::Overflow(OverflowBound::Bytes);
            }
        }
        if self.draining.load(Ordering::Acquire) {
            let over_entries = held >= self.capacity;
            if over_entries || held_bytes >= self.soft_bytes {
                let since = *self.over_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= self.watchdog {
                    return Offer::Overflow(if over_entries {
                        OverflowBound::Entries
                    } else {
                        OverflowBound::Bytes
                    });
                }
            } else {
                self.over_since = None;
            }
        }
        self.backlog.fetch_add(1, Ordering::AcqRel);
        self.backlog_bytes.fetch_add(bytes, Ordering::AcqRel);
        let delivery = Delivery::Entry {
            id: id.to_owned(),
            notification: payload.clone(),
            bytes,
        };
        if self.tx.send(delivery).is_err() {
            self.backlog.fetch_sub(1, Ordering::AcqRel);
            self.backlog_bytes.fetch_sub(bytes, Ordering::AcqRel);
            return Offer::Closed;
        }
        Offer::Accepted
    }

    /// Queues the marker (`resumes_at`: the id of the key's first entry in the
    /// same reply, whether or not it can be delivered) outside the backlog: no
    /// count, no
    /// cap, no watchdog clock, so a marker can neither end a subscription nor
    /// change when one ends. At most one is queued per reply, ahead of that
    /// reply's entries.
    fn offer_trimmed(&mut self, resumes_at: StreamId) -> Offer {
        if self.tx.send(Delivery::Trimmed(resumes_at)).is_err() {
            return Offer::Closed;
        }
        Offer::Accepted
    }

    fn fail(self, failure: Failure) {
        // The multiplexer's shutdown ending (the reader's `shutdown`, the only
        // failure with the `shutdown` reason) is a shutdown ending like every
        // other: it carries the shutdown retry hint, as
        // `grpc::transport::server_shutting_down` does. No other failure —
        // a stream gap, a slow reader, a watchdog, a reader error — carries
        // one.
        let error = if failure.reason == crate::observability::metrics::STREAM_EXIT_SHUTDOWN {
            crate::error::RefusalDetail::server_shutting_down()
                .into_unavailable_error(failure.message)
        } else {
            ServiceError::Unavailable(failure.message)
        };
        if let Ok(mut slot) = self.terminal.lock() {
            *slot = Some(ConsumerTerminalExit {
                reason: failure.reason,
                error,
            });
        }
        drop(self.tx);
    }
}

pub(crate) struct ReaderStats {
    pub keys: AtomicUsize,
    pub xreads: AtomicU64,
    pub wakes: AtomicU64,
    /// `COUNT` of the reader's latest XREAD; 0 before its first.
    pub last_count: AtomicU64,
    /// Mirrors of the reader loop's re-AUTH state (b.19i), for
    /// `StreamMux::reauth_for_test`: whether a tick is scheduled, whether a
    /// fetched credential awaits its `AUTH`, and the ticks taken.
    pub reauth_scheduled: AtomicBool,
    pub reauth_pending: AtomicBool,
    pub reauth_ticks: AtomicU64,
}

/// A reader's re-AUTH state, as [`StreamMux::reauth_for_test`] reports it.
#[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReauthSnapshot {
    /// A re-AUTH tick is scheduled: the reader holds a connection and a
    /// credential provider schedules ticks (never under `static` auth).
    pub scheduled: bool,
    /// A tick fetched a credential whose `AUTH` has not been sent yet: it
    /// waits for the in-flight XREAD to return.
    pub pending: bool,
    /// Re-AUTH ticks the reader has taken (each one `fetch(None)`).
    pub ticks: u64,
}

struct RouterState {
    core: RouterCore,
    ctl: Vec<mpsc::UnboundedSender<Ctl>>,
    closed: bool,
}

/// One reader's connection, kept on the shared state rather than in the
/// reader loop so it outlives a reader panic and [`StreamMux::quit_readers`]
/// can reach it.
#[derive(Default)]
pub(crate) struct ReaderSlot {
    conn: Option<ReaderConn>,
    /// Set by [`StreamMux::quit_readers`]: a connection that comes up after it
    /// is retired at once rather than published.
    closed: bool,
}

pub(crate) struct MuxShared {
    router: StdMutex<RouterState>,
    freed: Notify,
    next_sub: AtomicU64,
    knobs: StdMutex<StreamMuxKnobs>,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    stats: Vec<ReaderStats>,
    /// One per reader: its connection, if it holds one.
    conns: Vec<StdMutex<ReaderSlot>>,
    #[cfg(any(test, feature = "test-fault-injection"))]
    faults: StdMutex<std::collections::HashMap<MuxFaultSite, std::collections::VecDeque<MuxFault>>>,
    /// One per reader: its forced reconnects, which end a [`MuxFault::Hang`]
    /// XREAD on that reader.
    #[cfg(any(test, feature = "test-fault-injection"))]
    reconnects: Vec<TestSignal>,
    /// Every [`StreamMux::set_knobs_for_test`] call, so a reader waiting with
    /// nothing to read re-reads the knobs (a new keepalive interval, or the
    /// keepalive turned off) instead of sleeping out the old schedule.
    #[cfg(any(test, feature = "test-fault-injection"))]
    knobs_changed: TestSignal,
}

/// Test-only: counts events — one reader's forced reconnects, or knob changes
/// — so a task can wait for the next one without missing one that landed
/// before it started waiting. An injected hung XREAD ends when its reader
/// reconnects, as a real read on a wedged connection does.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
pub(crate) struct TestSignal {
    generation: AtomicU64,
    notify: Notify,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl TestSignal {
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Records one event.
    pub(crate) fn bump(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.notify.notify_waiters();
    }

    /// Resolves once an event has been recorded since generation `seen`.
    pub(crate) async fn changed_since(&self, seen: u64) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.generation() != seen {
                return;
            }
            notified.await;
        }
    }
}

impl MuxShared {
    fn knobs(&self) -> StreamMuxKnobs {
        *self.knobs.lock().expect("knobs lock")
    }

    fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|g| g.clone())
    }

    fn slot(&self, reader: usize) -> std::sync::MutexGuard<'_, ReaderSlot> {
        self.conns[reader]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Reader `reader`'s client and, on a cluster, the node it targets;
    /// `None` while it holds no connection.
    fn conn(&self, reader: usize) -> Option<(Client, Option<Server>)> {
        self.slot(reader)
            .conn
            .as_ref()
            .map(|conn| (conn.client().clone(), conn.node().cloned()))
    }

    fn is_connected(&self, reader: usize) -> bool {
        self.slot(reader).conn.is_some()
    }

    /// Makes `conn` reader `reader`'s connection and records its node on the
    /// router. Hands it back (`Some`) when [`StreamMux::quit_readers`] has
    /// closed the slot, for the caller to retire.
    fn publish(&self, reader: usize, conn: ReaderConn) -> Option<ReaderConn> {
        let node = conn.node().map(ToString::to_string);
        {
            let mut slot = self.slot(reader);
            if slot.closed {
                return Some(conn);
            }
            debug_assert!(slot.conn.is_none(), "a reader publishes over no connection");
            slot.conn = Some(conn);
        }
        self.set_router_node(reader, node);
        None
    }

    /// Arms reader `reader`'s published connection's metric hooks, and counts
    /// it on the connected-readers gauge if not yet counted
    /// (`ReaderConnector::arm_hooks`), under the slot's lock: a connection a
    /// retire has already taken out of the slot is left alone, so nothing is
    /// armed or counted on a connection after its retire disarmed it. No-op
    /// while the reader holds no connection.
    fn arm_published(&self, reader: usize, connector: &ReaderConnector) {
        if let Some(conn) = self.slot(reader).conn.as_mut() {
            connector.arm_hooks(conn);
        }
    }

    /// Takes reader `reader`'s connection out of its slot, for the caller to
    /// retire, and clears its node on the router.
    fn take_conn(&self, reader: usize) -> Option<ReaderConn> {
        let conn = self.slot(reader).conn.take();
        if conn.is_some() {
            self.set_router_node(reader, None);
        }
        conn
    }

    fn set_router_node(&self, reader: usize, node: Option<String>) {
        if let Ok(mut r) = self.router.lock() {
            r.core.set_node(reader, node);
        }
    }

    #[cfg(any(test, feature = "test-fault-injection"))]
    fn pop_fault(&self, site: MuxFaultSite) -> Option<MuxFault> {
        self.faults
            .lock()
            .expect("faults lock")
            .get_mut(&site)
            .and_then(std::collections::VecDeque::pop_front)
    }

    #[cfg(any(test, feature = "test-fault-injection"))]
    fn reconnect_signal(&self, reader: usize) -> &TestSignal {
        &self.reconnects[reader]
    }

    /// The knob-change count a reader reads before it reads the knobs; `0`
    /// when test seams are compiled out (the knobs never change).
    fn knobs_generation(&self) -> u64 {
        #[cfg(any(test, feature = "test-fault-injection"))]
        return self.knobs_changed.generation();
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        0
    }

    /// Resolves once the knobs have changed since generation `seen`; never
    /// when test seams are compiled out.
    async fn knobs_changed_since(&self, seen: u64) {
        #[cfg(any(test, feature = "test-fault-injection"))]
        self.knobs_changed.changed_since(seen).await;
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        {
            let _ = seen;
            std::future::pending::<()>().await;
        }
    }
}

/// Keeps one subscription's reader reference, from its reservation
/// ([`StreamMux::reserve`]) until its [`ReservedStream`] or
/// [`LayerStreamConsumer`] is dropped; dropping it releases the reference and
/// unregisters the stream if one was registered.
pub(crate) struct Registration {
    shared: Arc<MuxShared>,
    reader: usize,
    sub: SubId,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let freed = match self.shared.router.lock() {
            Ok(mut r) => {
                let freed = r.core.release(self.reader);
                let _ = r.ctl[self.reader].send(Ctl::Remove { sub: self.sub });
                freed
            }
            Err(_) => false,
        };
        if let Some(m) = self.shared.metrics() {
            m.add_stream_mux_subscriptions(-1);
        }
        if freed {
            self.shared.freed.notify_waiters();
        }
    }
}

/// A stream reader reserved for one subscription whose stream is not
/// registered yet ([`StreamMux::reserve`]).
///
/// It holds the reader's router reference — on a cluster the reader stays
/// bound to the key's hash slot — and counts on
/// `live_edit_stream_mux_subscriptions` and in [`MuxSnapshot::subscriptions`].
/// The reader neither reads nor queues anything for it: a reservation has no
/// channel, no backlog and no recorded ending, and it cannot be polled. Only
/// [`Self::register`] yields the [`LayerStreamConsumer`] that can, so a stream
/// is never polled before it is registered.
///
/// Dropping it — on any ending while reserved, a task abort included —
/// releases the reader exactly as dropping a consumer does: the reference is
/// released (a slot-bound reader whose last reference this was is unbound and
/// a waiting subscription is woken), the gauge is decremented, and the reader
/// is sent a `Remove` that finds nothing to remove.
pub struct ReservedStream {
    registration: Registration,
    key: String,
    key_schema: KeySchema,
}

impl std::fmt::Debug for ReservedStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedStream")
            .field("reader", &self.registration.reader)
            .field("key", &self.key)
            .finish()
    }
}

impl ReservedStream {
    /// Registers the stream on the reserved reader, delivering entries
    /// strictly after `cursor`, and returns its consumer. In-memory: no Valkey
    /// round trip and no wait. The consumer's backlog limits are the
    /// multiplexer's knobs at this call.
    ///
    /// Fails [`ConsumeError::ShuttingDown`] when the multiplexer has closed or
    /// the reader's control channel is gone, and
    /// [`ConsumeError::InvalidCursor`] on a cursor that is not a stream id.
    /// Either way the reservation is dropped, releasing the reader.
    pub fn register(self, cursor: &StreamCursor) -> Result<LayerStreamConsumer, ConsumeError> {
        let start = parse_cursor(cursor)?;
        let shared = self.registration.shared.clone();
        let (sink, rx, backlog, terminal) = ChannelSink::new(&shared.knobs());
        let sent = {
            let r = shared.router.lock().expect("router lock");
            !r.closed
                && r.ctl[self.registration.reader]
                    .send(Ctl::Add {
                        sub: self.registration.sub,
                        key: self.key,
                        cursor: start,
                        sink,
                        key_schema: self.key_schema,
                    })
                    .is_ok()
        };
        if !sent {
            // Dropping the registration (outside the router lock) releases
            // the reader.
            drop(self.registration);
            return Err(ConsumeError::ShuttingDown);
        }
        Ok(LayerStreamConsumer::new(
            rx,
            backlog,
            terminal,
            self.registration,
            start,
        ))
    }
}

fn parse_cursor(cursor: &StreamCursor) -> Result<StreamId, ConsumeError> {
    let raw = cursor.as_xread_id();
    StreamId::parse(raw).ok_or_else(|| ConsumeError::InvalidCursor(raw.to_owned()))
}

/// Snapshot for tests and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuxSnapshot {
    pub routing: Routing,
    pub readers: Vec<ReaderSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderSnapshot {
    pub subscriptions: usize,
    pub slot: Option<u16>,
    /// Stream keys named by the reader's latest XREAD (wake key excluded).
    pub keys: usize,
    pub xreads: u64,
    pub wakes: u64,
    /// Whether the reader holds a connection.
    pub connected: bool,
    /// On a cluster, the node (`host:port`) the reader's connection targets;
    /// `None` without a connection, and off a cluster.
    pub node: Option<String>,
}

impl MuxSnapshot {
    pub fn subscriptions(&self) -> usize {
        self.readers.iter().map(|r| r.subscriptions).sum()
    }
    pub fn xreads(&self) -> u64 {
        self.readers.iter().map(|r| r.xreads).sum()
    }
    pub fn active_readers(&self) -> usize {
        self.readers.iter().filter(|r| r.subscriptions > 0).count()
    }
    /// Readers holding a connection.
    pub fn connected_readers(&self) -> usize {
        self.readers.iter().filter(|r| r.connected).count()
    }
}

pub struct StreamMux {
    shared: Arc<MuxShared>,
    /// Also resolves a new binding's slot owner for the router's node
    /// affinity, from the readers' routing view.
    connector: Arc<ReaderConnector>,
    cluster: bool,
    cancel: CancellationToken,
    tasks: StdMutex<Vec<JoinHandle<bool>>>,
}

impl std::fmt::Debug for StreamMux {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamMux")
            .field("readers", &self.shared.conns.len())
            .field("cluster", &self.cluster)
            .finish()
    }
}

impl StreamMux {
    /// Spawns `readers` reader tasks, none of them connected: each connects
    /// through `connector` when it first serves a subscription, and the
    /// connector's routing view names slot owners. `data` carries the
    /// non-blocking commands (wake XADDs, key probes). `cluster` selects slot
    /// affinity. Must be called inside a Tokio runtime.
    pub(crate) fn start(
        connector: Arc<ReaderConnector>,
        data: Pool,
        cluster: bool,
        readers: usize,
        knobs: StreamMuxKnobs,
        metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    ) -> Self {
        let n = readers;
        let routing = if cluster {
            Routing::SlotAffinity
        } else {
            Routing::Hash
        };
        let mut senders = Vec::with_capacity(n);
        let mut receivers = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = mpsc::unbounded_channel();
            senders.push(tx);
            receivers.push(rx);
        }
        // Each reader also holds a sender to its own control channel, which the
        // wake tasks it spawns report their result on.
        let own_ctl = senders.clone();
        let shared = Arc::new(MuxShared {
            router: StdMutex::new(RouterState {
                core: RouterCore::new(routing, n),
                ctl: senders,
                closed: false,
            }),
            freed: Notify::new(),
            next_sub: AtomicU64::new(1),
            knobs: StdMutex::new(knobs),
            metrics,
            stats: (0..n)
                .map(|_| ReaderStats {
                    keys: AtomicUsize::new(0),
                    xreads: AtomicU64::new(0),
                    wakes: AtomicU64::new(0),
                    last_count: AtomicU64::new(0),
                    reauth_scheduled: AtomicBool::new(false),
                    reauth_pending: AtomicBool::new(false),
                    reauth_ticks: AtomicU64::new(0),
                })
                .collect(),
            conns: (0..n)
                .map(|_| StdMutex::new(ReaderSlot::default()))
                .collect(),
            #[cfg(any(test, feature = "test-fault-injection"))]
            faults: StdMutex::new(std::collections::HashMap::new()),
            #[cfg(any(test, feature = "test-fault-injection"))]
            reconnects: (0..n).map(|_| TestSignal::default()).collect(),
            #[cfg(any(test, feature = "test-fault-injection"))]
            knobs_changed: TestSignal::default(),
        });
        let cancel = CancellationToken::new();
        let instance: Arc<str> = uuid::Uuid::new_v4().simple().to_string().into();
        let tasks = receivers
            .into_iter()
            .zip(own_ctl)
            .enumerate()
            .map(|(idx, (rx, ctl))| {
                tokio::spawn(reader::run(
                    reader::ReaderCtx {
                        idx,
                        connector: connector.clone(),
                        data: data.clone(),
                        cluster,
                        instance: instance.clone(),
                        shared: shared.clone(),
                        cancel: cancel.clone(),
                        ctl,
                    },
                    rx,
                ))
            })
            .collect();
        Self {
            shared,
            connector,
            cluster,
            cancel,
            tasks: StdMutex::new(tasks),
        }
    }

    /// The fred client reader `reader` currently holds, or `None` while it
    /// holds no connection — every reader until it first serves a
    /// subscription. To borrow a reader's client, subscribe first, find the
    /// reader bound to it ([`Self::snapshot`]), then take its client here;
    /// fred's cached `connection_ids()` identifies the connection without
    /// sending a command.
    ///
    /// Never issue a command on it: a reader is parked in `XREAD BLOCK` with
    /// `Blocking::Interrupt`, so any other command would `CLIENT UNBLOCK` the
    /// reader's live read. An idle connected reader is not quiet either: its
    /// keepalive `PING`s the connection (`keepalive_interval`, from a random
    /// offset), so a test that borrows an idle reader's connection turns the
    /// keepalive off first. Under a credential provider (never in the static
    /// fixtures) a connected reader also sends `AUTH` on it between reads when
    /// a re-AUTH tick fetches a credential. The client may also be replaced at
    /// any moment by a retarget.
    ///
    /// # Panics
    ///
    /// When `reader` is not below the reader count.
    #[cfg(any(test, feature = "test-support"))]
    pub fn reader_client_for_test(&self, reader: usize) -> Option<Client> {
        self.shared.conn(reader).map(|(client, _)| client)
    }

    /// The fred clients of every reader holding a connection now, in reader
    /// order — empty until the first subscription, since readers connect
    /// lazily. Same caveats as [`Self::reader_client_for_test`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn reader_clients_for_test(&self) -> Vec<Client> {
        (0..self.shared.conns.len())
            .filter_map(|reader| self.shared.conn(reader).map(|(client, _)| client))
            .collect()
    }

    /// Readers holding a connection now.
    pub fn connected_readers(&self) -> usize {
        (0..self.shared.conns.len())
            .filter(|&reader| self.shared.is_connected(reader))
            .count()
    }

    /// Arms the metric hooks of every reader connected now (startup, from
    /// `ValkeyClient::install_observability_hooks`), and counts on
    /// `live_edit_stream_mux_readers_connected` each such connection its
    /// connect could not count (it came up before the metrics were
    /// installed), so the gauge counts every held connection exactly once and
    /// each retire subtracts exactly what was added. A reader that connects
    /// later arms (and, if need be, counts) its own once its connection is
    /// published ([`MuxShared::arm_published`]).
    pub(crate) fn arm_reader_hooks(&self) {
        for reader in 0..self.shared.conns.len() {
            self.shared.arm_published(reader, &self.connector);
        }
    }

    /// How many routing-view syncs the readers' connector has issued and seen
    /// end — answered, or cut off by their bound — not answered syncs only
    /// (see `ReaderConnector::sync_routing`, which is single-flight across
    /// the readers).
    #[cfg(any(test, feature = "test-support"))]
    pub fn routing_syncs_finished_for_test(&self) -> u64 {
        self.connector.routing_syncs_finished_for_test()
    }

    /// The fred client id of the client whose routing table is the readers'
    /// routing view: data-pool member 0's current client, which a member
    /// replacement moves to the fresh client.
    #[cfg(any(test, feature = "test-support"))]
    pub fn routing_client_id_for_test(&self) -> String {
        self.connector.routing_client_id_for_test()
    }

    /// Reserves a reader for a subscription to `object_id`'s layer stream,
    /// without registering the stream: the reader reads and queues nothing for
    /// it until [`ReservedStream::register`]. A catch-up reserves before its
    /// capacity waits and registers at its proven read's cut (its own atomic
    /// read's, or the shared fenced read's of the flight it is attached to),
    /// so while it waits it holds a reader and no queue.
    ///
    /// The reservation takes the reader's router reference — on a cluster that
    /// binds the reader to the key's hash slot — and counts on
    /// `live_edit_stream_mux_subscriptions` (not on
    /// `live_edit_stream_mux_streams`).
    ///
    /// In cluster mode, when every reader is bound to another hash slot, waits
    /// for one to free until `deadline` — the subscription's capacity-budget
    /// deadline — and is then refused [`ConsumeError::Saturated`] (counted on
    /// `live_edit_stream_mux_bind_refusals_total`). A deadline already passed
    /// takes a free reader if there is one and never waits. Standalone routing
    /// never waits. The wait does not observe the caller's endings (shutdown of
    /// the service, session deletion, client disconnect); the caller races it
    /// against them. Only the multiplexer's own shutdown ends it early, as
    /// [`ConsumeError::ShuttingDown`].
    pub async fn reserve(
        &self,
        key_schema: &KeySchema,
        object_id: u64,
        deadline: tokio::time::Instant,
    ) -> Result<ReservedStream, ConsumeError> {
        let key = key_schema.layer_stream(object_id);
        let slot = fred::util::redis_keyslot(key.as_bytes());
        // The slot's owner, for node affinity, once this reservation needs a
        // new binding: resolved outside the router lock (it clones the readers'
        // routing view, so not on every subscribe — a slot already bound keeps
        // its reader and needs none), and at most once per call. `None` until
        // resolved; `Some(None)` off a cluster or while the routing view does
        // not name an owner, and the router then takes the lowest-index free
        // reader.
        let mut owner: Option<Option<String>> = None;
        loop {
            let freed = self.shared.freed.notified();
            tokio::pin!(freed);
            freed.as_mut().enable();
            {
                let mut r = self.shared.router.lock().expect("router lock");
                if r.closed {
                    return Err(ConsumeError::ShuttingDown);
                }
                if owner.is_none() && self.cluster && !r.core.is_bound(slot) {
                    drop(r);
                    owner = Some(self.connector.slot_owner(slot).map(|s| s.to_string()));
                    // Re-take the lock and re-check: the slot may have been
                    // bound, or the multiplexer closed, meanwhile.
                    continue;
                }
                let near = owner.as_ref().and_then(Option::as_deref);
                if let Some(reader) = r.core.route_near(object_id, slot, near) {
                    drop(r);
                    let sub = self.shared.next_sub.fetch_add(1, Ordering::Relaxed);
                    let registration = Registration {
                        shared: self.shared.clone(),
                        reader,
                        sub,
                    };
                    if let Some(m) = self.shared.metrics() {
                        m.add_stream_mux_subscriptions(1);
                    }
                    return Ok(ReservedStream {
                        registration,
                        key,
                        key_schema: key_schema.clone(),
                    });
                }
            }
            if tokio::time::timeout_at(deadline, freed).await.is_err() {
                if let Some(m) = self.shared.metrics() {
                    m.inc_stream_mux_bind_refusal();
                }
                return Err(ConsumeError::Saturated);
            }
        }
    }

    /// Registers a subscription to `object_id`'s layer stream, delivering
    /// entries strictly after `cursor`: [`Self::reserve`] then
    /// [`ReservedStream::register`], for a caller with no capacity wait
    /// between the two. An invalid `cursor` is refused
    /// ([`ConsumeError::InvalidCursor`]) before any reader wait.
    pub async fn consume(
        &self,
        key_schema: &KeySchema,
        object_id: u64,
        cursor: &StreamCursor,
        deadline: tokio::time::Instant,
    ) -> Result<LayerStreamConsumer, ConsumeError> {
        parse_cursor(cursor)?;
        self.reserve(key_schema, object_id, deadline)
            .await?
            .register(cursor)
    }

    pub fn snapshot(&self) -> MuxSnapshot {
        let r = self.shared.router.lock().expect("router lock");
        MuxSnapshot {
            routing: r.core.routing(),
            readers: r
                .core
                .readers()
                .iter()
                .zip(&self.shared.stats)
                .enumerate()
                .map(|(idx, (b, s))| ReaderSnapshot {
                    subscriptions: b.refs,
                    slot: b.slot,
                    keys: s.keys.load(Ordering::Acquire),
                    xreads: s.xreads.load(Ordering::Acquire),
                    wakes: s.wakes.load(Ordering::Acquire),
                    connected: self.shared.is_connected(idx),
                    node: r.core.node(idx).map(str::to_owned),
                })
                .collect(),
        }
    }

    /// Refuses new subscriptions, ends live ones with `Unavailable`, wakes
    /// blocked reads, and waits up to `grace` for the readers to stop. The
    /// data pool must still be connected (wakes use it).
    pub async fn shutdown(&self, grace: Duration) {
        if let Ok(mut r) = self.shared.router.lock() {
            r.closed = true;
        }
        self.shared.freed.notify_waiters();
        self.cancel.cancel();
        let tasks: Vec<JoinHandle<bool>> = self
            .tasks
            .lock()
            .map(|mut t| t.drain(..).collect())
            .unwrap_or_default();
        let readers = tasks.len();
        match tokio::time::timeout(grace, futures::future::join_all(tasks)).await {
            Ok(exits) => {
                let woken = exits.iter().filter(|e| matches!(e, Ok(true))).count();
                // "reader pool" is historical wording (each reader quits its
                // own connection now); the text is kept because tests match
                // it verbatim.
                tracing::debug!(
                    readers,
                    woken,
                    "woke blocked stream reads before the reader pool QUIT"
                );
            }
            Err(_) => tracing::warn!(
                grace_ms = grace.as_millis() as u64,
                "stream multiplexer readers did not stop within the shutdown grace"
            ),
        }
    }

    /// Closes every reader's connection slot and retires the connections
    /// held now, all concurrently: each retire ends its router's
    /// supervision, disarms its hooks and sends `QUIT` bounded by
    /// `QUIT_TIMEOUT` (none to a client that is not connected), then aborts
    /// the router. A reader still
    /// connecting finds its slot closed and retires its new connection itself;
    /// one cancelled mid-connect aborts its half-started router. Call after
    /// [`Self::shutdown`], so no XREAD is blocking the `QUIT`s.
    ///
    /// Returns how many `QUIT`s overran their bound.
    pub async fn quit_readers(&self) -> usize {
        let conns: Vec<ReaderConn> = (0..self.shared.conns.len())
            .filter_map(|reader| {
                let conn = {
                    let mut slot = self.shared.slot(reader);
                    slot.closed = true;
                    slot.conn.take()
                };
                if conn.is_some() {
                    self.shared.set_router_node(reader, None);
                }
                conn
            })
            .collect();
        let outcomes =
            futures::future::join_all(conns.into_iter().map(|conn| self.connector.retire(conn)))
                .await;
        outcomes.iter().filter(|o| o.quit_overran).count()
    }

    /// Replaces the knobs. Readers apply them from their next step; a reader
    /// waiting with nothing to read is woken, so a new keepalive interval (or
    /// the keepalive turned off) takes effect at once rather than after the
    /// old schedule's next tick.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn set_knobs_for_test(&self, knobs: StreamMuxKnobs) {
        *self.shared.knobs.lock().expect("knobs lock") = knobs;
        self.shared.knobs_changed.bump();
    }

    pub fn knobs(&self) -> StreamMuxKnobs {
        self.shared.knobs()
    }

    /// The `COUNT` reader `reader` put on its latest XREAD, or `None` before
    /// its first. Covers every read the reader issues, including one a
    /// [`MuxFault`] then replaces.
    ///
    /// # Panics
    ///
    /// When `reader` is not below the reader count.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn last_planned_count_for_test(&self, reader: usize) -> Option<u64> {
        match self.shared.stats[reader].last_count.load(Ordering::Acquire) {
            0 => None,
            count => Some(count),
        }
    }

    /// Reader `reader`'s re-AUTH state (b.19i): whether a tick is scheduled,
    /// whether a fetched credential awaits its `AUTH`, and how many ticks it
    /// has taken. Mirrors the reader loop's own state, so it can trail it by
    /// the loop's current step.
    ///
    /// # Panics
    ///
    /// When `reader` is not below the reader count.
    #[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
    pub fn reauth_for_test(&self, reader: usize) -> ReauthSnapshot {
        let stats = &self.shared.stats[reader];
        ReauthSnapshot {
            scheduled: stats.reauth_scheduled.load(Ordering::Acquire),
            pending: stats.reauth_pending.load(Ordering::Acquire),
            ticks: stats.reauth_ticks.load(Ordering::Acquire),
        }
    }

    /// Queues faults for the next operations any reader performs at `site`, in
    /// order. The queue is shared by every reader on this multiplexer: at
    /// [`MuxFaultSite::Connect`], for instance, the next reader to connect
    /// takes the head, whichever reader that is.
    ///
    /// # Panics
    ///
    /// On a [`MuxFault::Reply`] or [`MuxFault::Panic`] at any site other than
    /// [`MuxFaultSite::Xread`], or a [`MuxFault::Hold`] at any site other than
    /// [`MuxFaultSite::Connect`]: none has a meaning there.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn push_faults_for_test(
        &self,
        site: MuxFaultSite,
        faults: impl IntoIterator<Item = MuxFault>,
    ) {
        let faults: Vec<MuxFault> = faults.into_iter().collect();
        if site != MuxFaultSite::Xread {
            assert!(
                faults
                    .iter()
                    .all(|f| !matches!(f, MuxFault::Reply(_) | MuxFault::Panic)),
                "MuxFault::Reply and MuxFault::Panic apply to MuxFaultSite::Xread only, not {site:?}"
            );
        }
        if site != MuxFaultSite::Connect {
            assert!(
                faults.iter().all(|f| !matches!(f, MuxFault::Hold(_))),
                "MuxFault::Hold applies to MuxFaultSite::Connect only, not {site:?}"
            );
        }
        self.shared
            .faults
            .lock()
            .expect("faults lock")
            .entry(site)
            .or_default()
            .extend(faults);
    }

    /// Faults still queued at `site`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn pending_faults_for_test(&self, site: MuxFaultSite) -> usize {
        self.shared
            .faults
            .lock()
            .expect("faults lock")
            .get(&site)
            .map_or(0, std::collections::VecDeque::len)
    }
}

impl Drop for StreamMux {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// The production sink's backlog rule, and its consumer's rule for trim
/// markers, driven through a real registration ([`ReservedStream::register`])
/// on an in-memory multiplexer with no reader task: the test holds the reader's
/// control channel and takes the sink out of the `Ctl::Add` the registration
/// sends, so the consumer it offers into is the one the registration returned.
#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    /// What one offer did (`Offer` itself is not comparable).
    #[derive(Debug, PartialEq, Eq)]
    enum Got {
        Accepted,
        Closed,
        Overflow(OverflowBound),
    }

    struct Registered {
        sink: ChannelSink,
        consumer: LayerStreamConsumer,
        /// The reader's end of its control channel: keeps it open.
        _ctl: mpsc::UnboundedReceiver<Ctl>,
    }

    impl Registered {
        fn offer(&mut self, bytes: u64) -> Got {
            match self.sink.offer("1-0", &LayerNotification::Deleted, bytes) {
                Offer::Accepted => Got::Accepted,
                Offer::Closed => Got::Closed,
                Offer::Overflow(bound) => Got::Overflow(bound),
            }
        }

        /// Offers `n` entries of `bytes` each, all of which must be accepted.
        fn accept(&mut self, n: u64, bytes: u64) {
            for i in 0..n {
                assert_eq!(
                    self.offer(bytes),
                    Got::Accepted,
                    "entry {i} of {n} ({bytes} B) must be accepted"
                );
            }
        }

        /// `(entries, bytes)` queued, as the sink and consumer share them.
        fn held(&self) -> (usize, u64) {
            (
                self.sink.backlog.load(Ordering::Acquire),
                self.sink.backlog_bytes.load(Ordering::Acquire),
            )
        }
    }

    /// A stream registered on a one-reader in-memory multiplexer with `knobs`,
    /// from `0-0`.
    fn register(knobs: StreamMuxKnobs) -> Registered {
        register_from(knobs, "0-0")
    }

    /// [`register`] from `cursor`.
    fn register_from(knobs: StreamMuxKnobs, cursor: &str) -> Registered {
        let (tx, mut ctl) = mpsc::unbounded_channel();
        let mut core = RouterCore::new(Routing::Hash, 1);
        let reader = core.route(1, 0).expect("hash routing always routes");
        let shared = Arc::new(MuxShared {
            router: StdMutex::new(RouterState {
                core,
                ctl: vec![tx],
                closed: false,
            }),
            freed: Notify::new(),
            next_sub: AtomicU64::new(2),
            knobs: StdMutex::new(knobs),
            metrics: Arc::new(StdMutex::new(None)),
            stats: Vec::new(),
            conns: vec![StdMutex::new(ReaderSlot::default())],
            faults: StdMutex::new(std::collections::HashMap::new()),
            reconnects: vec![TestSignal::default()],
            knobs_changed: TestSignal::default(),
        });
        let reserved = ReservedStream {
            registration: Registration {
                shared,
                reader,
                sub: 1,
            },
            key: "k".to_owned(),
            key_schema: KeySchema::default(),
        };
        let consumer = reserved
            .register(&StreamCursor::From(cursor.to_owned()))
            .expect("an open multiplexer registers");
        let Ok(Ctl::Add { sink, .. }) = ctl.try_recv() else {
            panic!("registering sends the reader its sink");
        };
        Registered {
            sink,
            consumer,
            _ctl: ctl,
        }
    }

    /// Production knobs with the soft watchdog at zero: once the consumer is
    /// draining, the first offer over a soft threshold ends the subscription,
    /// so the soft rule is observable without waiting.
    fn instant_watchdog(knobs: StreamMuxKnobs) -> StreamMuxKnobs {
        StreamMuxKnobs {
            send_watchdog: Duration::ZERO,
            ..knobs
        }
    }

    /// At the production caps, before the first poll, entries of 2 KiB (the
    /// largest captured Kit entry, rounded up) reach the entry cap first: the
    /// queue holds 16,384 of them — exactly 32 MiB — and the next is refused
    /// `entries`. This is what keeps Kit-size traffic ending exactly as the
    /// entry-only backlog did.
    #[tokio::test]
    async fn at_the_production_caps_kit_size_entries_reach_the_entry_cap_first() {
        let mut r = register(StreamMuxKnobs::new());
        r.accept(16_384, 2 * KIB);
        assert_eq!(r.held(), (16_384, 32 * MIB));
        assert_eq!(
            r.offer(2 * KIB),
            Got::Overflow(OverflowBound::Entries),
            "both limits are reached; the entry limit names the bound"
        );
        assert_eq!(
            r.held(),
            (16_384, 32 * MIB),
            "a refused entry is not queued"
        );
    }

    /// At the production caps, before the first poll, 16 KiB entries reach the
    /// byte cap first: 2,048 of them fill 32 MiB and the next is refused
    /// `bytes`, far below the 16,384-entry cap.
    #[tokio::test]
    async fn at_the_production_caps_16_kib_entries_reach_the_byte_cap_first() {
        let mut r = register(StreamMuxKnobs::new());
        r.accept(2_048, 16 * KIB);
        assert_eq!(r.held(), (2_048, 32 * MIB));
        assert_eq!(r.offer(16 * KIB), Got::Overflow(OverflowBound::Bytes));
    }

    /// An empty queue always accepts one entry, so a single update larger than
    /// the whole hard byte cap never ends a caught-up subscriber; the next
    /// offer while it is still queued is refused `bytes`. Once drained, the
    /// queue is empty again and accepts the next oversized entry.
    #[tokio::test]
    async fn an_oversized_entry_into_an_empty_queue_is_accepted_and_the_next_is_refused_bytes() {
        let mut r = register(StreamMuxKnobs::new());
        assert_eq!(r.offer(40 * MIB), Got::Accepted);
        assert_eq!(r.held(), (1, 40 * MIB));
        assert_eq!(r.offer(1), Got::Overflow(OverflowBound::Bytes));

        assert!(r.consumer.try_recv().await.is_some());
        assert_eq!(r.held(), (0, 0));
        assert_eq!(
            r.offer(40 * MIB),
            Got::Accepted,
            "drained, the queue is empty again"
        );
    }

    /// The consumer subtracts exactly the size each entry was offered with, so
    /// the byte count returns exactly to zero however the sizes mix.
    #[tokio::test]
    async fn draining_returns_the_byte_count_exactly_to_zero() {
        let mut r = register(StreamMuxKnobs::new());
        let sizes = [1, 777, 16 * KIB, 5 * MIB + 3, 2 * KIB];
        for &bytes in &sizes {
            assert_eq!(r.offer(bytes), Got::Accepted);
        }
        let mut left: u64 = sizes.iter().sum();
        assert_eq!(r.held(), (sizes.len(), left));
        for (i, &bytes) in sizes.iter().enumerate() {
            let received = if i % 2 == 0 {
                r.consumer.recv().await
            } else {
                r.consumer.try_recv().await
            };
            assert!(received.is_some(), "entry {i} is delivered");
            left -= bytes;
            assert_eq!(r.held(), (sizes.len() - i - 1, left));
        }
        assert_eq!(r.held(), (0, 0));
    }

    /// Before the consumer's first poll only the hard caps apply: with the soft
    /// watchdog at zero, a queue past both soft thresholds (4,096 entries and
    /// 8 MiB) keeps accepting. The first poll — here a `try_recv` — marks it
    /// draining, and the next offer over a soft threshold ends it.
    #[tokio::test]
    async fn before_the_first_poll_the_soft_watchdog_never_fires() {
        let mut r = register(instant_watchdog(StreamMuxKnobs::new()));
        r.accept(5_000, 2 * KIB);
        let (entries, bytes) = r.held();
        assert!(
            entries > 4_096 && bytes > 8 * MIB,
            "past both soft thresholds"
        );
        assert!(
            r.consumer.take_terminal_exit().is_none(),
            "the ending check is not a poll"
        );
        assert_eq!(r.offer(2 * KIB), Got::Accepted, "still not draining");

        assert!(r.consumer.try_recv().await.is_some());
        assert_eq!(
            r.offer(2 * KIB),
            Got::Overflow(OverflowBound::Entries),
            "draining: the soft watchdog applies"
        );
    }

    /// Once draining, a queue held at 8 MiB past the watchdog ends the
    /// subscription `bytes`, well below the 4,096-entry soft threshold. The
    /// first poll counts whatever it returns: here it finds the queue empty.
    #[tokio::test]
    async fn once_draining_bytes_held_past_the_watchdog_end_it_on_bytes() {
        let mut r = register(instant_watchdog(StreamMuxKnobs::new()));
        assert!(
            r.consumer.try_recv().await.is_none(),
            "an empty poll still marks the consumer draining"
        );
        r.accept(512, 16 * KIB);
        assert_eq!(r.held(), (512, 8 * MIB));
        assert_eq!(r.offer(16 * KIB), Got::Overflow(OverflowBound::Bytes));
    }

    /// `Bytes` only when a byte limit was reached and the entry limit was not;
    /// every other overflow, both reached included, is `Entries` — for the hard
    /// caps (from registration) and the soft thresholds (once draining) alike.
    #[tokio::test]
    async fn the_bound_is_bytes_only_when_a_byte_limit_was_reached_and_the_entry_limit_was_not() {
        // Soft: 2 entries or 200 B; hard: 4 entries or 400 B.
        let small = StreamMuxKnobs {
            channel_capacity: 2,
            backlog_hard_cap: 4,
            backlog_bytes: 200,
            backlog_hard_bytes: 400,
            send_watchdog: Duration::ZERO,
            ..StreamMuxKnobs::new()
        };
        use OverflowBound::{Bytes, Entries};
        // (case, draining, queued sizes, next size, bound)
        let cases: [(&str, bool, &[u64], u64, OverflowBound); 6] = [
            (
                "hard, both reached",
                false,
                &[100, 100, 100, 100],
                100,
                Entries,
            ),
            ("hard, entries only", false, &[1, 1, 1, 1], 1, Entries),
            ("hard, bytes only", false, &[100, 100, 100], 101, Bytes),
            ("soft, both reached", true, &[100, 100], 1, Entries),
            ("soft, entries only", true, &[1, 1], 1, Entries),
            ("soft, bytes only", true, &[200], 1, Bytes),
        ];
        for (case, draining, queued, next, bound) in cases {
            let mut r = register(small);
            if draining {
                assert!(r.consumer.try_recv().await.is_none());
            }
            for &bytes in queued {
                assert_eq!(r.offer(bytes), Got::Accepted, "{case}: queued {bytes} B");
            }
            assert_eq!(r.offer(next), Got::Overflow(bound), "{case}");
        }
    }

    /// A consumer that is gone refuses the offer as closed, which the reader
    /// drops silently, and nothing is counted as queued.
    #[tokio::test]
    async fn an_offer_to_a_dropped_consumer_is_closed_and_counts_nothing() {
        let mut r = register(StreamMuxKnobs::new());
        r.consumer.shutdown_and_release().await;
        assert_eq!(r.offer(10), Got::Closed);
        assert_eq!(r.held(), (0, 0));
    }

    // ── Trim markers: the consumer's floor rule ─────────────────────────────

    use super::core::{Entry, Head, Heads, ReaderCore};
    use crate::observability::metrics::{STREAM_EXIT_ERROR, STREAM_EXIT_STREAM_GAP};
    use crate::storage::notifications::STREAM_GAP_MSG;

    fn update(seq: u64) -> LayerNotification {
        LayerNotification::Update {
            object_id: 1,
            sequence_num: seq,
            client_id: "writer".to_owned(),
            client_ts: 0,
            server_ts: 0,
            relay_ompf: None,
        }
    }

    fn stream_id(ms: u64) -> StreamId {
        StreamId { ms, seq: 0 }
    }

    fn got(offer: Offer) -> Got {
        match offer {
            Offer::Accepted => Got::Accepted,
            Offer::Closed => Got::Closed,
            Offer::Overflow(bound) => Got::Overflow(bound),
        }
    }

    impl Registered {
        /// Offers the update entry `{seq}-0` of `bytes` bytes.
        fn entry(&mut self, seq: u64, bytes: u64) -> Got {
            got(self.sink.offer(&format!("{seq}-0"), &update(seq), bytes))
        }

        /// Offers a trim marker resuming at `{ms}-0`.
        fn trim(&mut self, ms: u64) -> Got {
            got(self.sink.offer_trimmed(stream_id(ms)))
        }
    }

    /// Asserts the consumer's ending is the `stream_gap` one, with its status.
    fn assert_stream_gap(consumer: &LayerStreamConsumer, case: &str) {
        let exit = consumer
            .take_terminal_exit()
            .unwrap_or_else(|| panic!("{case}: an uncovered trim must record an ending"));
        assert_eq!(exit.reason, STREAM_EXIT_STREAM_GAP, "{case}");
        match exit.error {
            ServiceError::Unavailable(message) => assert_eq!(message, STREAM_GAP_MSG, "{case}"),
            other => panic!("{case}: stream_gap is a retryable UNAVAILABLE; got {other:?}"),
        }
    }

    /// The multiplexer's shutdown ending — the reader's `shutdown` failure,
    /// reason `shutdown` — is the shared shutdown status: `UNAVAILABLE
    /// "server shutting down"` with the `SERVER_SHUTTING_DOWN` retry hint.
    /// Every other reader-side failure (a stream gap, a slow reader, a
    /// watchdog, a reader error) stays a plain `UNAVAILABLE` with no refusal
    /// detail: none of them is a nothing-applied load-shedding refusal.
    #[tokio::test]
    async fn only_the_shutdown_failure_carries_the_shutdown_retry_hint() {
        use super::core::Failure;
        use crate::observability::metrics::{
            STREAM_EXIT_INNER_WATCHDOG, STREAM_EXIT_SHUTDOWN, STREAM_EXIT_SLOW_READER_DEADLINE,
        };

        let r = register(StreamMuxKnobs::new());
        let Registered { sink, consumer, .. } = r;
        sink.fail(Failure {
            reason: STREAM_EXIT_SHUTDOWN,
            message: "server shutting down".to_owned(),
        });
        let exit = consumer
            .take_terminal_exit()
            .expect("the shutdown failure records an ending");
        assert_eq!(exit.reason, STREAM_EXIT_SHUTDOWN);
        match exit.error {
            ServiceError::UnavailableWithDetail { message, detail } => {
                assert_eq!(message, "server shutting down");
                assert_eq!(*detail, crate::error::RefusalDetail::server_shutting_down());
            }
            other => panic!("the shutdown ending carries the shutdown retry hint; got {other:?}"),
        }

        for reason in [
            STREAM_EXIT_STREAM_GAP,
            STREAM_EXIT_ERROR,
            STREAM_EXIT_SLOW_READER_DEADLINE,
            STREAM_EXIT_INNER_WATCHDOG,
        ] {
            let Registered { sink, consumer, .. } = register(StreamMuxKnobs::new());
            sink.fail(Failure {
                reason,
                message: format!("{reason} ending"),
            });
            let exit = consumer
                .take_terminal_exit()
                .unwrap_or_else(|| panic!("{reason}: the failure records an ending"));
            match exit.error {
                ServiceError::Unavailable(message) => {
                    assert_eq!(message, format!("{reason} ending"))
                }
                other => panic!(
                    "{reason}: a reader-side ending other than shutdown carries no refusal \
                     detail; got {other:?}"
                ),
            }
        }
    }

    /// A marker the floor covers is dropped silently by both polls: each
    /// returns the entry queued behind it and records no ending.
    #[tokio::test]
    async fn a_covered_marker_is_dropped_by_recv_and_by_try_recv() {
        for blocking in [true, false] {
            let mut r = register_from(StreamMuxKnobs::new(), "5-0");
            r.consumer.set_caught_up_floor(20);
            assert_eq!(r.trim(21), Got::Accepted);
            assert_eq!(r.entry(22, 10), Got::Accepted);
            let next = if blocking {
                r.consumer.recv().await
            } else {
                r.consumer.try_recv().await
            };
            let (id, _) = next.unwrap_or_else(|| panic!("blocking={blocking}: the entry"));
            assert_eq!(id, "22-0", "blocking={blocking}");
            assert!(
                r.consumer.take_terminal_exit().is_none(),
                "blocking={blocking}: a covered marker records no ending"
            );
        }
    }

    /// `resumes_at` is compared with `covered_to = max(floor, last entry
    /// received)`, the registration cursor standing in before any entry: at
    /// or below `covered_to`'s successor the marker is dropped, one above it
    /// ends the subscription `stream_gap`.
    #[tokio::test]
    async fn a_marker_is_covered_up_to_the_successor_of_the_higher_of_floor_and_position() {
        // Registered at `5-0`: (case, floor, entries received first,
        // `covered_to`). Each is checked with a marker at `covered_to`'s
        // successor (dropped) and one above it (`stream_gap`).
        let scenarios: [(&str, Option<u64>, &[u64], u64); 5] = [
            ("the registration cursor, no floor", None, &[], 5),
            ("a floor above the cursor", Some(20), &[], 20),
            ("a floor below the cursor", Some(3), &[], 5),
            ("an entry received above the floor", Some(20), &[30], 30),
            ("a floor above the entry received", Some(40), &[30], 40),
        ];
        for (scenario, floor, received, covered_to) in scenarios {
            for (resumes_at, covered) in [(covered_to + 1, true), (covered_to + 2, false)] {
                let case = format!("{scenario}, marker at {resumes_at}");
                let mut r = register_from(StreamMuxKnobs::new(), "5-0");
                if let Some(floor) = floor {
                    r.consumer.set_caught_up_floor(floor);
                }
                for &seq in received {
                    assert_eq!(r.entry(seq, 1), Got::Accepted, "{case}");
                    assert!(r.consumer.try_recv().await.is_some(), "{case}");
                }
                assert_eq!(r.trim(resumes_at), Got::Accepted, "{case}");
                assert_eq!(r.entry(resumes_at, 1), Got::Accepted, "{case}");
                let next = r.consumer.try_recv().await;
                if covered {
                    assert_eq!(
                        next.map(|(id, _)| id),
                        Some(format!("{resumes_at}-0")),
                        "{case}"
                    );
                    assert!(r.consumer.take_terminal_exit().is_none(), "{case}");
                } else {
                    assert!(next.is_none(), "{case}: an uncovered marker ends the poll");
                    assert_stream_gap(&r.consumer, &case);
                }
            }
        }
    }

    /// An uncovered marker ends the stream: the poll returns `None`, the ending
    /// is `stream_gap` with its client-visible status, and every later poll
    /// returns `None` though an entry is still queued behind the marker.
    #[tokio::test]
    async fn an_uncovered_marker_ends_the_subscription_stream_gap_and_every_later_poll() {
        let mut r = register_from(StreamMuxKnobs::new(), "5-0");
        r.consumer.set_caught_up_floor(20);
        assert_eq!(r.trim(25), Got::Accepted);
        assert_eq!(r.entry(25, 1), Got::Accepted);
        assert!(r.consumer.recv().await.is_none());
        assert_stream_gap(&r.consumer, "uncovered");
        assert!(r.consumer.recv().await.is_none(), "recv after the gap");
        assert!(
            r.consumer.try_recv().await.is_none(),
            "try_recv after the gap"
        );
    }

    /// The ending a poll reported is the one the handler takes: a reader ending
    /// recorded after the gap does not replace it.
    #[tokio::test]
    async fn the_gap_wins_over_a_reader_ending_recorded_after_it() {
        let Registered {
            mut sink,
            consumer,
            _ctl,
        } = register_from(StreamMuxKnobs::new(), "5-0");
        assert_eq!(got(sink.offer_trimmed(stream_id(9))), Got::Accepted);
        assert!(consumer.try_recv().await.is_none());
        sink.fail(Failure {
            reason: STREAM_EXIT_ERROR,
            message: "a reader ending after the gap".to_owned(),
        });
        assert_stream_gap(&consumer, "gap, then a reader ending");
    }

    /// A marker counts against neither backlog limit: it is accepted with the
    /// entry hard cap full, adds nothing to the counters, and is received
    /// without subtracting anything, so they return exactly to zero.
    #[tokio::test]
    async fn a_marker_counts_against_no_backlog_limit_and_leaves_the_counters_unchanged() {
        let small = StreamMuxKnobs {
            backlog_hard_cap: 2,
            ..StreamMuxKnobs::new()
        };
        let mut r = register_from(small, "5-0");
        r.consumer.set_caught_up_floor(1_000);
        assert_eq!(r.entry(6, 100), Got::Accepted);
        assert_eq!(r.entry(7, 200), Got::Accepted);
        assert_eq!(r.trim(9), Got::Accepted, "a marker with the hard cap full");
        assert_eq!(r.held(), (2, 300), "a marker adds nothing");
        assert_eq!(
            r.entry(10, 1),
            Got::Overflow(OverflowBound::Entries),
            "the entry cap is still full"
        );
        assert!(r.consumer.try_recv().await.is_some());
        assert!(r.consumer.try_recv().await.is_some());
        assert_eq!(r.held(), (0, 0));
        assert!(
            r.consumer.try_recv().await.is_none(),
            "the covered marker is dropped"
        );
        assert_eq!(r.held(), (0, 0), "a marker subtracts nothing");
        assert!(r.consumer.take_terminal_exit().is_none());
    }

    /// A marker never starts or clears the soft watchdog's clock: only entries
    /// do.
    #[tokio::test]
    async fn a_marker_never_starts_or_clears_the_watchdog_clock() {
        let knobs = StreamMuxKnobs {
            channel_capacity: 1,
            send_watchdog: Duration::from_secs(3_600),
            ..StreamMuxKnobs::new()
        };
        let mut r = register_from(knobs, "5-0");
        r.consumer.set_caught_up_floor(1_000);
        assert!(r.consumer.try_recv().await.is_none(), "start draining");
        assert_eq!(r.entry(6, 1), Got::Accepted);
        assert_eq!(r.trim(8), Got::Accepted);
        assert_eq!(
            r.sink.over_since, None,
            "over the soft threshold, a marker does not start the clock"
        );
        assert_eq!(r.entry(7, 1), Got::Accepted, "an entry starts it");
        let started = r
            .sink
            .over_since
            .expect("an entry over the threshold starts the clock");
        assert_eq!(r.trim(9), Got::Accepted);
        while r.consumer.try_recv().await.is_some() {}
        assert_eq!(r.held(), (0, 0));
        assert_eq!(r.trim(10), Got::Accepted);
        assert_eq!(
            r.sink.over_since,
            Some(started),
            "under the threshold, a marker does not clear the clock"
        );
    }

    /// T4.8, FR-2.6 through the real sink and consumer: a catch-up whose floor
    /// covers the trim never ends `stream_gap`, even when a second trim after
    /// the read moved the head past that floor. Cursor 5, floor 20, reply
    /// `18..=30` read from 5, head 25 at the check: the marker carries 18,
    /// which the floor covers, and the consumer yields 18..=30. A marker valued
    /// at the head (25) would end it.
    #[tokio::test]
    async fn a_trim_the_floor_covers_never_ends_the_subscription_whatever_the_head() {
        let Registered {
            sink,
            consumer,
            _ctl,
        } = register_from(StreamMuxKnobs::new(), "5-0");
        consumer.set_caught_up_floor(20);
        let mut reader = ReaderCore::<LayerNotification, ChannelSink>::new();
        reader.add(1, "k".to_owned(), stream_id(5), sink);
        reader.plan_read("w").expect("a registered key is read");
        let reply: Vec<Entry<LayerNotification>> = (18..=30)
            .map(|seq| Entry {
                id: stream_id(seq),
                id_str: format!("{seq}-0"),
                payload: Some(update(seq)),
                bytes: 1,
            })
            .collect();
        let out = reader.on_read_ok(
            vec![("k".to_owned(), reply)],
            &Heads::from([("k".to_owned(), Head::First(Some(stream_id(25))))]),
        );
        assert_eq!(
            out.trimmed, 1,
            "the head passed the read position: one marker"
        );
        let mut ids = Vec::new();
        while let Some((id, _)) = consumer.try_recv().await {
            ids.push(id);
        }
        let expected: Vec<String> = (18..=30).map(|seq| format!("{seq}-0")).collect();
        assert_eq!(ids, expected);
        assert!(
            consumer.take_terminal_exit().is_none(),
            "a trim the catch-up covered never ends the subscription"
        );
    }
}
