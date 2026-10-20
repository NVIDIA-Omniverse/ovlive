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

//! Layer-stream notifications (SDD §12).
//!
//! Writers XADD one entry per commit to `le:{object_id}:obj:stream` inside the
//! commit's MULTI/EXEC; entry ids are `{sequence_num}-0`. Subscribers consume
//! through the shared XREAD multiplexer ([`crate::storage::stream_mux`]) via
//! [`consume_layer_stream`], or in two steps via [`reserve_layer_stream`] and
//! [`ReservedStream::register`]; either returns a [`LayerStreamConsumer`]
//! yielding `(entry_id, LayerNotification)` pairs.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::mpsc;

use crate::error::ServiceError;
use crate::observability::metrics::{STREAM_EXIT_ERROR, STREAM_EXIT_STREAM_GAP};
use crate::storage::client::ValkeyClient;
use crate::storage::keys::KeySchema;
use crate::storage::stream_mux::core::StreamId;
use crate::storage::stream_mux::{Delivery, Registration};

pub use crate::storage::stream_mux::{ConsumeError, ReservedStream};

/// How long each multiplexed XREAD blocks before returning empty.
///
/// A reader is woken early only by a new subscription, or by the
/// multiplexer's shutdown (so each reader connection's QUIT does not wait out the
/// block). A new subscription wakes it when it needs something the in-flight
/// read does not cover (a key the read does not name, or a cursor behind the
/// id its key is read from), or when the reader's connection has not answered
/// within the liveness age (`stream_mux::WAKE_ACK_TIMEOUT`) or has failed
/// since, so the join gets a connection verdict. A removal or a cursor advance
/// never wakes it. So this bounds idle round trips, and how long a new subscription
/// can wait for its first live update when its wake fails. With no join, the
/// read's client-side timeout (this plus 5 s, 35 s) is what detects a wedged
/// reader connection.
pub const XREAD_BLOCK_MS: u64 = 30_000;

/// Stream-entry field name carrying the pipe-delimited notification metadata
/// (`update|oid|seq|cid|cts|sts` or `deleted`). Present on every entry.
pub const NOTIFICATION_PAYLOAD_FIELD: &str = "payload";

/// Value of [`NOTIFICATION_PAYLOAD_FIELD`] on the layer-retirement tombstone
/// appended by `delete_layer` and by the `CreateLayer`-overwrite path.
///
/// Named because three unrelated sites must agree on it byte-for-byte: the
/// two writers, [`LayerNotification::parse`] on the subscriber side, and the
/// stream-collision disambiguation in `storage::layer` (b.ph9), which reads it
/// to tell a retired layer apart from a corrupted stream. The mirror copy in
/// `scripts/stream_collision_probe.lua` is the one place it is spelled out
/// again — Lua cannot see this constant.
pub const DELETED_STREAM_PAYLOAD: &str = "deleted";

/// Stream-entry field name carrying the writer's raw inbound OMPF (binary),
/// relayed verbatim to steady-state subscribers (b.gze). Present only on
/// `update` entries written after this feature landed AND whose writer
/// supplied non-empty OMPF; absent on `deleted` sentinels, on CreateLayer's
/// `1-0` seed entry, and on pre-b.gze entries still inside the retention
/// window. So on a current build an `update` entry without this field above
/// sequence 1 is an empty commit: a subscriber sends it as the empty delta at
/// its sequence with no read, in steady state and in the rollback-mode
/// in-window replay alike (`grpc::subscribe`), as checkpoint tail capture
/// classifies it. Pre-b.gze entries are excluded by upgrade precondition, not
/// told apart here.
pub const RELAY_OMPF_FIELD: &str = "ompf";

/// A parsed notification from a layer stream entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNotification {
    /// A layer was updated (or created).
    Update {
        object_id: u64,
        sequence_num: u64,
        /// The client that performed the write. Empty string for CreateLayer.
        client_id: String,
        /// Microsecond timestamp from the client (0 for CreateLayer).
        ///
        /// `u64`, matching the proto's unvalidated `u64` field and what every
        /// writer emits, so any value a client sends parses, including one
        /// above `i64::MAX`.
        client_ts: u64,
        /// Microsecond timestamp assigned by the server.
        server_ts: i64,
        /// The writer's raw inbound OMPF, relayed verbatim to steady-state
        /// subscribers (b.gze). `None` when the entry carries no `ompf` field
        /// (CreateLayer seed entry, empty-OMPF update, or a pre-b.gze entry) —
        /// a steady-state subscriber, which never sees the seed, then sends
        /// the empty delta at this commit's sequence with no read (see
        /// [`RELAY_OMPF_FIELD`]).
        ///
        /// NOT part of `parse` (which works only on the pipe-delimited
        /// `payload` text); the consumer attaches it from the separate binary
        /// `ompf` stream field via [`LayerNotification::with_relay_ompf`].
        relay_ompf: Option<Bytes>,
    },
    /// The layer was deleted (DeleteLayer or session cascade).
    Deleted,
}

impl LayerNotification {
    /// Attach relay OMPF bytes to an `Update` notification (b.gze). No-op on
    /// `Deleted`. Called by the stream consumer after `parse` once it has
    /// extracted the separate binary `ompf` field from the same stream entry.
    pub fn with_relay_ompf(self, relay_ompf: Option<Bytes>) -> Self {
        match self {
            LayerNotification::Update {
                object_id,
                sequence_num,
                client_id,
                client_ts,
                server_ts,
                ..
            } => LayerNotification::Update {
                object_id,
                sequence_num,
                client_id,
                client_ts,
                server_ts,
                relay_ompf,
            },
            other => other,
        }
    }
}

/// Errors that can occur when parsing a notification payload.
#[derive(Debug, thiserror::Error)]
pub enum NotificationParseError {
    #[error("unknown notification type: {0}")]
    UnknownType(String),

    #[error("wrong field count: expected {expected}, got {got}")]
    WrongFieldCount { expected: usize, got: usize },

    #[error("failed to parse integer field '{field}': {source}")]
    ParseInt {
        field: &'static str,
        source: std::num::ParseIntError,
    },
}

impl LayerNotification {
    /// Parse a pipe-delimited notification payload string.
    ///
    /// Formats:
    /// - `update|{object_id}|{sequence_num}|{client_id}|{client_ts}|{server_ts}`
    /// - `deleted`
    pub fn parse(payload: &str) -> Result<Self, NotificationParseError> {
        if payload == DELETED_STREAM_PAYLOAD {
            return Ok(LayerNotification::Deleted);
        }

        let parts: Vec<&str> = payload.splitn(6, '|').collect();

        if parts.first() != Some(&"update") {
            return Err(NotificationParseError::UnknownType(
                parts.first().unwrap_or(&"").to_string(),
            ));
        }

        if parts.len() != 6 {
            return Err(NotificationParseError::WrongFieldCount {
                expected: 6,
                got: parts.len(),
            });
        }

        let object_id = parts[1]
            .parse::<u64>()
            .map_err(|e| NotificationParseError::ParseInt {
                field: "object_id",
                source: e,
            })?;

        let sequence_num =
            parts[2]
                .parse::<u64>()
                .map_err(|e| NotificationParseError::ParseInt {
                    field: "sequence_num",
                    source: e,
                })?;

        let client_id = parts[3].to_string();

        let client_ts = parts[4]
            .parse::<u64>()
            .map_err(|e| NotificationParseError::ParseInt {
                field: "client_ts",
                source: e,
            })?;

        let server_ts = parts[5]
            .parse::<i64>()
            .map_err(|e| NotificationParseError::ParseInt {
                field: "server_ts",
                source: e,
            })?;

        Ok(LayerNotification::Update {
            object_id,
            sequence_num,
            client_id,
            client_ts,
            server_ts,
            // `parse` works only on the pipe-delimited `payload` text; the
            // binary relay OMPF rides on a separate stream field and is
            // attached by the consumer via `with_relay_ompf` (b.gze).
            relay_ompf: None,
        })
    }
}

/// Starting position for a consumer: strictly after the given stream id.
///
/// There is deliberately no `$` ("latest") variant: Valkey re-resolves `$` at
/// every XREAD, so a re-issued read would silently skip entries appended in
/// between (SDD §12.3). Callers pin `From("{seq}-0")` from the header they hold.
#[derive(Debug, Clone)]
pub enum StreamCursor {
    /// Start strictly after this stream ID.
    From(String),
}

impl StreamCursor {
    pub(crate) fn as_xread_id(&self) -> &str {
        match self {
            StreamCursor::From(id) => id.as_str(),
        }
    }
}

/// Undelivered entries at which a draining consumer's delivery watchdog arms
/// (the soft entry threshold; the hard cap is
/// `stream_mux::BACKLOG_HARD_CAP_FACTOR` times this). Matches the default
/// `LIVE_EDIT_LAYER_STREAM_MAXLEN`, the stream's entry cap, so a fresh
/// subscriber's in-window window cannot by itself trip the entry threshold.
/// In bytes the retained window can exceed [`CONSUMER_BACKLOG_BYTES`] (the
/// relay byte budget, `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`, defaults to 128 MiB),
/// which is why the rollback-mode in-window replay reads the window itself, in
/// pages, rather than leaving it to the backlog.
pub const CONSUMER_CHANNEL_CAPACITY: usize = 4096;

/// Undelivered bytes at which a draining consumer's delivery watchdog arms
/// (the soft byte threshold, 8 MiB; the hard cap is
/// `stream_mux::BACKLOG_HARD_CAP_FACTOR` times this, 32 MiB). Each queued entry
/// counts its id plus its `payload` and `ompf` field values, the measure that
/// also sizes the multiplexer's reads. It is [`CONSUMER_CHANNEL_CAPACITY`] ×
/// 2 KiB — the largest captured Kit entry (≈1.7 KB,
/// `tools/fixtures/gze_inbound_deltas.json`) rounded up — so a backlog of
/// entries up to 2 KiB reaches the entry limits first and ends exactly as an
/// entry-only backlog would; only larger entries reach a byte limit first.
pub const CONSUMER_BACKLOG_BYTES: u64 = 8 * 1024 * 1024;

/// How long a draining consumer's backlog may stay at or above
/// [`CONSUMER_CHANNEL_CAPACITY`] entries or [`CONSUMER_BACKLOG_BYTES`] before
/// the subscription is ended with `STREAM_EXIT_INNER_WATCHDOG`. The clock
/// starts at the first offer over either threshold after the consumer's first
/// poll; before that only the hard caps apply.
pub const CONSUMER_SEND_WATCHDOG: Duration = Duration::from_secs(5);

/// Per-send bound on the **outer** gRPC hand-off in `grpc::subscribe` (the
/// response channel's 32 delivery slots; the 33rd is reserved for the terminal
/// trailer). It must not exceed [`CONSUMER_SEND_WATCHDOG`], or the inner
/// watchdog would always fire first and `slow_reader_deadline` would be
/// unreachable. Tests override it via
/// `LiveEditService::outer_send_deadline_for_test`.
pub const OUTER_SEND_DEADLINE: Duration = Duration::from_secs(5);

const _: () = assert!(
    OUTER_SEND_DEADLINE.as_millis() <= CONSUMER_SEND_WATCHDOG.as_millis(),
    "OUTER_SEND_DEADLINE must not exceed CONSUMER_SEND_WATCHDOG: the outer send parks only while \
     the RPC task is not draining, which is the same stall that backs the consumer's backlog up; \
     a longer outer deadline would make slow_reader_deadline unreachable."
);

const _: () = assert!(
    OUTER_SEND_DEADLINE.as_millis() == 5_000
        && CONSUMER_CHANNEL_CAPACITY == 4_096
        && CONSUMER_BACKLOG_BYTES == 8 * 1024 * 1024
        && crate::storage::stream_mux::BACKLOG_HARD_CAP_FACTOR == 4,
    "observability::metrics copies these limits: SUBSCRIBER_SEND_WAIT_BUCKETS' top boundary is \
     OUTER_SEND_DEADLINE (5 s), and the live_edit_subscriber_send_wait_seconds help text says a \
     slow-reader deadline lands above the 5 s bucket; SUBSCRIBER_BACKLOG_ENTRIES_BUCKETS brackets \
     CONSUMER_CHANNEL_CAPACITY (4096) and its hard cap (16384), and \
     SUBSCRIBER_BACKLOG_BYTES_BUCKETS brackets CONSUMER_BACKLOG_BYTES (8 MiB) and its hard cap \
     (32 MiB, BACKLOG_HARD_CAP_FACTOR 4). Change those buckets and that help text with these \
     values."
);

/// Client-visible `UNAVAILABLE` message for a layer stream that ended with no
/// recorded reason: synthesized by [`LayerStreamConsumer::take_terminal_exit`],
/// and the handler's own fallback should the consumer ever yield none
/// (`grpc::subscribe::run_steady_state`). One constant, so the two cannot say
/// different things.
pub const READER_ENDED_WITHOUT_REASON_MSG: &str =
    "layer stream reader ended without a recorded reason; reconnect";

/// Client-visible `UNAVAILABLE` message of a `stream_gap` ending: the layer
/// stream trimmed entries after this subscription's position that its catch-up
/// did not cover ([`LayerStreamConsumer::set_caught_up_floor`]), or the stream
/// was removed and recreated by a later commit (a restore after a keyspace
/// loss, a stream-key removal) above that position. The reconnect catches up
/// in full — after a removal, onto the layer's current, restored content.
pub const STREAM_GAP_MSG: &str =
    "layer stream was trimmed past this subscription's position; reconnect to catch up";

/// Why a consumer's stream ended, recorded before end-of-stream is observable.
/// `reason` classifies `live_edit_subscriber_stream_exits_total{reason}`;
/// `error` (always `Unavailable`) is what the handler returns.
pub struct ConsumerTerminalExit {
    /// `STREAM_EXIT_INNER_WATCHDOG`, `STREAM_EXIT_ERROR`,
    /// `STREAM_EXIT_SHUTDOWN` (the multiplexer shutting down), or
    /// `STREAM_EXIT_STREAM_GAP` (recorded by the consumer itself, from a trim
    /// marker its catch-up floor does not cover).
    pub reason: &'static str,
    /// The status the handler ends the RPC with.
    pub error: ServiceError,
}

/// A live subscription to one layer stream.
///
/// **Ending contract.** `recv() == None` followed by an empty
/// [`Self::take_terminal_exit`] happens only after this handle closed itself
/// (`unsubscribe` / `shutdown_and_release`). Every other ending — reader
/// failure, watchdog, shutdown, a trim past the subscription's position, even
/// a reader panic — yields a terminal exit so the handler returns
/// `Unavailable` and the client re-subscribes.
///
/// A `Deleted` entry is delivered like any other; the RPC layer decides what
/// it means.
///
/// **Backlog.** The multiplexer queues entries into this consumer without
/// blocking its reader; the queue's size is tracked in entries and in bytes
/// (each entry carries its own size, which is subtracted exactly when it is
/// received). The first [`Self::recv`] or
/// [`Self::try_recv`] marks the consumer **draining**, whatever it returns;
/// only from then does the soft delivery watchdog apply
/// ([`CONSUMER_SEND_WATCHDOG`]). Before it — a catch-up building while its
/// stream buffers — only the hard caps bound the queue.
/// [`Self::take_terminal_exit`] is not a poll and never marks it.
///
/// **Trimmed stream.** When the reader finds the stream's head past its read
/// position and the key's first entry in a reply is more than one id above
/// this consumer's cursor, it queues a trim marker ahead of that reply's
/// entries. The marker carries `resumes_at`, the id of the key's first entry
/// in that reply, whether or not it can be delivered (an entry that cannot be
/// converted is skipped, and the consumer receives the next one): the stream
/// no longer held the ids between the cursor and it when the reader read
/// them. `recv` and `try_recv` never yield a
/// marker: they compare `resumes_at` with `covered_to` = the higher of the
/// caught-up floor ([`Self::set_caught_up_floor`]) and the id of the last
/// entry received (the registration cursor before any), and drop it when
/// `resumes_at` is at or below `covered_to`'s successor — nothing the
/// subscription was owed is missing. Otherwise they record a `stream_gap`
/// ending (`Unavailable`, [`STREAM_GAP_MSG`]) and return `None`, as for any
/// other ending; [`Self::take_terminal_exit`] then yields it, and every later
/// poll returns `None`.
pub struct LayerStreamConsumer {
    rx: tokio::sync::Mutex<Queue>,
    backlog: Arc<AtomicUsize>,
    backlog_bytes: Arc<AtomicU64>,
    draining: Arc<AtomicBool>,
    terminal_error: Arc<StdMutex<Option<ConsumerTerminalExit>>>,
    ended: AtomicBool,
    registration: StdMutex<Option<Registration>>,
    /// The caught-up ack's sequence: the floor is `{floor}-0`. `0` until set,
    /// which is no floor: every position is at or above `0-0`.
    floor: AtomicU64,
    /// Set when a poll found a trim its floor does not cover; every poll after
    /// it returns `None`.
    gapped: AtomicBool,
    /// Set with `gapped`, and cleared when [`Self::take_terminal_exit`] hands
    /// out the `stream_gap` ending. Kept apart from `terminal_error`, which the
    /// reader writes, so a reader ending recorded after the gap cannot replace
    /// the ending the poll already reported.
    gap_unreported: AtomicBool,
}

/// The consumer's end of its channel, and how far it has read.
struct Queue {
    rx: mpsc::UnboundedReceiver<Delivery>,
    /// The id of the last entry received, or the registration cursor before
    /// any.
    position: StreamId,
}

/// What one received item means to a poll.
enum Received {
    Entry(String, LayerNotification),
    /// A trim marker the floor covers; the poll moves on to the next item.
    Covered,
    /// The stream ended, or a trim marker recorded `stream_gap`.
    End,
}

/// A consumer's undelivered backlog at one instant
/// ([`LayerStreamConsumer::backlog`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BacklogDepth {
    /// Undelivered entries (trim markers not counted).
    pub entries: u64,
    /// Undelivered bytes, as the multiplexer's sink counts them.
    pub bytes: u64,
}

/// The backlog counters a consumer shares with its multiplexer sink.
pub(crate) struct ConsumerBacklog {
    /// Undelivered entries (trim markers not counted).
    pub entries: Arc<AtomicUsize>,
    /// Undelivered bytes, as each [`Delivery::Entry`]'s `bytes` counts them
    /// (a trim marker counts none).
    pub bytes: Arc<AtomicU64>,
    /// Set by the consumer's first poll; arms the soft watchdog.
    pub draining: Arc<AtomicBool>,
}

impl LayerStreamConsumer {
    /// `start` is the registration cursor: the consumer's position until it
    /// receives its first entry.
    pub(crate) fn new(
        rx: mpsc::UnboundedReceiver<Delivery>,
        backlog: ConsumerBacklog,
        terminal_error: Arc<StdMutex<Option<ConsumerTerminalExit>>>,
        registration: Registration,
        start: StreamId,
    ) -> Self {
        Self {
            rx: tokio::sync::Mutex::new(Queue {
                rx,
                position: start,
            }),
            backlog: backlog.entries,
            backlog_bytes: backlog.bytes,
            draining: backlog.draining,
            terminal_error,
            ended: AtomicBool::new(false),
            registration: StdMutex::new(Some(registration)),
            floor: AtomicU64::new(0),
            gapped: AtomicBool::new(false),
            gap_unreported: AtomicBool::new(false),
        }
    }

    /// Sets the caught-up floor to `{seq}-0`: `seq` is the version the
    /// subscription's caught-up ack carried, so every commit at or below it
    /// reached the client in its catch-up. A trim marker's `resumes_at` is
    /// compared with the higher of this floor and the last entry received, so
    /// a trim the catch-up covered never ends the subscription `stream_gap`,
    /// while one that took a commit above both does. `run_steady_state` sets it on entry
    /// and after each re-bootstrap, before its first poll of the consumer.
    pub fn set_caught_up_floor(&self, seq: u64) {
        self.floor.store(seq, Ordering::Release);
    }

    /// The entries and bytes queued for this consumer and not yet received:
    /// two loads of the counters the multiplexer's sink already maintains, so
    /// the reader's offer path is untouched. The two loads are not one atomic
    /// snapshot; each is exact at its own instant.
    ///
    /// Read by the subscription task, not at offer: `grpc::subscribe`'s
    /// `run_steady_state` samples it for each live delivery once the
    /// delivery's last chunk is enqueued on the response channel. Each of the
    /// delivery's messages (a header and at least one chunk) may wait up to
    /// `OUTER_SEND_DEADLINE` for a slot, so the sample can trail the dequeue
    /// by a multiple of it. The queue
    /// shrinks only when that task dequeues and only grows in between, so a
    /// sample taken before the next dequeue is close to the queue's peak.
    pub fn backlog(&self) -> BacklogDepth {
        BacklogDepth {
            entries: self.backlog.load(Ordering::Acquire) as u64,
            bytes: self.backlog_bytes.load(Ordering::Acquire),
        }
    }

    fn observe(&self, queue: &mut Queue, item: Option<Delivery>) -> Received {
        match item {
            Some(Delivery::Entry {
                id,
                notification,
                bytes,
            }) => {
                self.backlog.fetch_sub(1, Ordering::AcqRel);
                self.backlog_bytes.fetch_sub(bytes, Ordering::AcqRel);
                // The reader offers only entries whose id parsed.
                if let Some(entry) = StreamId::parse(&id) {
                    queue.position = entry;
                }
                Received::Entry(id, notification)
            }
            Some(Delivery::Trimmed(resumes_at)) => {
                let floor = StreamId {
                    ms: self.floor.load(Ordering::Acquire),
                    seq: 0,
                };
                let covered_to = floor.max(queue.position);
                if resumes_at <= covered_to.successor() {
                    return Received::Covered;
                }
                self.gap_unreported.store(true, Ordering::Release);
                self.gapped.store(true, Ordering::Release);
                Received::End
            }
            None => {
                self.ended.store(true, Ordering::Release);
                Received::End
            }
        }
    }

    /// Next `(stream_entry_id, notification)`, or `None` once the stream ended
    /// — including on a trim its floor does not cover (see the type's
    /// "Trimmed stream"). Marks the consumer draining. Cancel-safe: an item is
    /// consumed only when this returns it, except a covered trim marker, which
    /// is dropped either way.
    pub async fn recv(&self) -> Option<(String, LayerNotification)> {
        self.draining.store(true, Ordering::Release);
        let mut queue = self.rx.lock().await;
        loop {
            if self.gapped.load(Ordering::Acquire) {
                return None;
            }
            let item = queue.rx.recv().await;
            match self.observe(&mut queue, item) {
                Received::Entry(id, notification) => return Some((id, notification)),
                Received::Covered => continue,
                Received::End => return None,
            }
        }
    }

    /// Non-blocking variant of [`Self::recv`]; `None` when nothing is queued
    /// or the stream ended. Marks the consumer draining, whatever it returns.
    pub async fn try_recv(&self) -> Option<(String, LayerNotification)> {
        self.draining.store(true, Ordering::Release);
        let mut queue = self.rx.lock().await;
        loop {
            if self.gapped.load(Ordering::Acquire) {
                return None;
            }
            let item = match queue.rx.try_recv() {
                Ok(item) => Some(item),
                Err(mpsc::error::TryRecvError::Empty) => return None,
                Err(mpsc::error::TryRecvError::Disconnected) => None,
            };
            match self.observe(&mut queue, item) {
                Received::Entry(id, notification) => return Some((id, notification)),
                Received::Covered => continue,
                Received::End => return None,
            }
        }
    }

    /// Takes the recorded terminal exit. After an end this handle did not
    /// cause, synthesizes an `Unavailable` one if none was recorded, so an
    /// abnormal ending can never read as a clean close. A `stream_gap` the
    /// consumer recorded comes first: it is the ending its poll reported.
    pub fn take_terminal_exit(&self) -> Option<ConsumerTerminalExit> {
        if self.gap_unreported.swap(false, Ordering::AcqRel) {
            return Some(ConsumerTerminalExit {
                reason: STREAM_EXIT_STREAM_GAP,
                error: ServiceError::Unavailable(STREAM_GAP_MSG.to_owned()),
            });
        }
        if let Some(exit) = self.terminal_error.lock().ok()?.take() {
            return Some(exit);
        }
        let self_closed = self
            .registration
            .lock()
            .map(|r| r.is_none())
            .unwrap_or(true);
        if self.ended.swap(false, Ordering::AcqRel) && !self_closed {
            return Some(ConsumerTerminalExit {
                reason: STREAM_EXIT_ERROR,
                error: ServiceError::Unavailable(READER_ENDED_WITHOUT_REASON_MSG.to_owned()),
            });
        }
        None
    }

    /// Unregisters from the multiplexer.
    pub async fn unsubscribe(mut self) {
        self.shutdown_and_release().await;
    }

    /// Unregisters from the multiplexer and closes the channel. After this
    /// returns no further entry is delivered to this consumer.
    pub async fn shutdown_and_release(&mut self) {
        let registration = self.registration.lock().ok().and_then(|mut r| r.take());
        drop(registration);
        self.rx.lock().await.rx.close();
    }
}

/// Reserves a stream reader for `object_id`'s layer stream without registering
/// the stream: nothing is read or queued for it until
/// [`ReservedStream::register`]. In-memory: no Valkey round trip.
///
/// `deadline` bounds the cluster-mode wait for a free stream reader; see
/// [`StreamMux::reserve`](crate::storage::stream_mux::StreamMux::reserve).
pub async fn reserve_layer_stream(
    client: &ValkeyClient,
    key_schema: &KeySchema,
    object_id: u64,
    deadline: tokio::time::Instant,
) -> Result<ReservedStream, ConsumeError> {
    client
        .stream_mux
        .reserve(key_schema, object_id, deadline)
        .await
}

/// Subscribes to `object_id`'s layer stream, delivering entries strictly after
/// `start_cursor`: [`reserve_layer_stream`] then [`ReservedStream::register`].
/// Registration is in-memory: no Valkey round trip.
///
/// `deadline` bounds the cluster-mode wait for a free stream reader; see
/// [`StreamMux::consume`](crate::storage::stream_mux::StreamMux::consume).
pub async fn consume_layer_stream(
    client: &ValkeyClient,
    key_schema: &KeySchema,
    object_id: u64,
    start_cursor: StreamCursor,
    deadline: tokio::time::Instant,
) -> Result<LayerStreamConsumer, ConsumeError> {
    client
        .stream_mux
        .consume(key_schema, object_id, &start_cursor, deadline)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LayerNotification::parse tests (unchanged from pre-stream transport) ──

    #[test]
    fn parse_valid_update_returns_all_fields() {
        let payload = "update|42|7|client-abc|1000000|2000000";
        let result = LayerNotification::parse(payload).unwrap();
        assert_eq!(
            result,
            LayerNotification::Update {
                object_id: 42,
                sequence_num: 7,
                client_id: "client-abc".to_string(),
                client_ts: 1_000_000,
                server_ts: 2_000_000,
                relay_ompf: None,
            }
        );
    }

    #[test]
    fn parse_update_empty_client_id_succeeds() {
        let payload = "update|1|2||0|0";
        let result = LayerNotification::parse(payload).unwrap();
        assert_eq!(
            result,
            LayerNotification::Update {
                object_id: 1,
                sequence_num: 2,
                client_id: String::new(),
                client_ts: 0,
                server_ts: 0,
                relay_ompf: None,
            }
        );
    }

    #[test]
    fn parse_update_zero_timestamps_succeeds() {
        let payload = "update|99|1|creator|0|0";
        let result = LayerNotification::parse(payload).unwrap();
        assert_eq!(
            result,
            LayerNotification::Update {
                object_id: 99,
                sequence_num: 1,
                client_id: "creator".to_string(),
                client_ts: 0,
                server_ts: 0,
                relay_ompf: None,
            }
        );
    }

    /// G4 (SDD SR-5.4): the proto's `client_timestamp` is an unvalidated
    /// `uint64` that writers format into the payload verbatim, so every `u64`
    /// must parse — including values above `i64::MAX`, which the earlier
    /// `i64` parse rejected.
    #[test]
    fn parse_update_client_ts_above_i64_max_parses() {
        for client_ts in [i64::MAX as u64 + 1, u64::MAX] {
            let payload = format!("update|42|7|client-abc|{client_ts}|2000000");

            let result = LayerNotification::parse(&payload)
                .unwrap_or_else(|e| panic!("client_ts {client_ts} must parse, got {e:?}"));

            assert!(
                matches!(result, LayerNotification::Update { client_ts: ts, .. } if ts == client_ts),
                "client_ts {client_ts} must survive the parse verbatim, got {result:?}"
            );
        }
    }

    /// The other side of the `u64` domain: a value no writer can emit is a
    /// `client_ts` parse error, not a silently coerced timestamp.
    #[test]
    fn parse_update_client_ts_outside_u64_returns_client_ts_parse_int_error() {
        for client_ts in ["-1", "18446744073709551616"] {
            let payload = format!("update|42|7|client-abc|{client_ts}|2000000");

            let err = LayerNotification::parse(&payload).unwrap_err();

            assert!(
                matches!(
                    err,
                    NotificationParseError::ParseInt {
                        field: "client_ts",
                        ..
                    }
                ),
                "client_ts {client_ts}: expected ParseInt for client_ts, got: {err:?}"
            );
        }
    }

    #[test]
    fn parse_deleted_returns_deleted_variant() {
        let result = LayerNotification::parse("deleted").unwrap();
        assert_eq!(result, LayerNotification::Deleted);
    }

    #[test]
    fn parse_unknown_type_returns_unknown_type_error() {
        let err = LayerNotification::parse("refresh|1|2|c|3|4").unwrap_err();
        assert!(
            matches!(err, NotificationParseError::UnknownType(ref t) if t == "refresh"),
            "expected UnknownType(\"refresh\"), got: {err:?}"
        );
    }

    #[test]
    fn parse_wrong_field_count_returns_error() {
        // "update" with only 4 fields instead of 6
        let err = LayerNotification::parse("update|1|2|c").unwrap_err();
        assert!(
            matches!(err, NotificationParseError::WrongFieldCount { expected: 6, got } if got < 6),
            "expected WrongFieldCount, got: {err:?}"
        );
    }

    #[test]
    fn parse_non_numeric_object_id_returns_parse_int_error() {
        let err = LayerNotification::parse("update|abc|2|c|3|4").unwrap_err();
        assert!(
            matches!(
                err,
                NotificationParseError::ParseInt {
                    field: "object_id",
                    ..
                }
            ),
            "expected ParseInt for object_id, got: {err:?}"
        );
    }

    #[test]
    fn parse_empty_string_returns_error() {
        let err = LayerNotification::parse("").unwrap_err();
        assert!(
            matches!(err, NotificationParseError::UnknownType(ref t) if t.is_empty()),
            "expected UnknownType with empty string, got: {err:?}"
        );
    }

    #[test]
    fn parse_garbage_data_returns_error() {
        let err = LayerNotification::parse("!@#$%^&*()").unwrap_err();
        assert!(err.to_string().contains("unknown notification type"));
    }

    // ── StreamCursor tests ─────────────────────────────────────────────

    #[test]
    fn stream_cursor_from_serializes_to_id() {
        let cursor = StreamCursor::From("1234-0".to_string());
        assert_eq!(cursor.as_xread_id(), "1234-0");
    }

    // `$` is re-resolved by Valkey at every XREAD, so a cursor must always be a
    // concrete id; the single-variant enum makes `$` unrepresentable.
    #[test]
    fn every_stream_cursor_serializes_to_a_concrete_id_never_dollar() {
        for id in ["0-0", "1-0", "7-0", "9007199254740991-0"] {
            let cursor = StreamCursor::From(id.to_string());
            assert_eq!(
                cursor.as_xread_id(),
                id,
                "the cursor must reach XREAD as the caller's durable ID, verbatim"
            );
            assert_ne!(
                cursor.as_xread_id(),
                "$",
                "no `StreamCursor` may serialize to `$` — Valkey re-resolves `$` at every \
                 XREAD execution, so a post-wedge retry skips every entry appended while \
                 the connection was down (b.9uq round 4, SDD §12.3)"
            );
        }
    }
}
