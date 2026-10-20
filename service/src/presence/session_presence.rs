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

// In-memory presence routing and presence event types.
//
// LocalPresenceRouter multiplexes presence events from Valkey pub/sub to local
// JoinSession gRPC streams on this replica. It does NOT track membership —
// that is Valkey's responsibility (SDD §13.1).
//
// Presence events are published to `le:presence:events:{session_id}` as
// pipe-delimited strings (SDD §13.1):
//   PeerJoined:  joined|{esc(connection_id)}|{esc(user_name)}|{esc(user_application)}
//   PeerLeft:    left|{connection_id}
//   PeerMessage: msg|{connection_id}|{base64(data)}
// esc() writes `%` as `%25` and `|` as `%7C` and changes nothing else, so a
// `joined` payload always holds exactly three `|`. The parser decodes only
// `%25` and uppercase `%7C` and copies any other `%` (lowercase `%7c`
// included) as is. The escaping is wire-only: the presence info hash stores
// raw values.

use std::borrow::Cow;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use fred::clients::SubscriberClient;
use fred::interfaces::{ClientLike, PubsubInterface};
use fred::types::config::Options;
use fred::types::{ClusterHash, Message};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::fanout::{PresenceReceiver, SessionFanout};
use super::relay::{DrainEnd, PubsubRelayReceiver};
use crate::observability::metrics::{SharedMetrics, PUBSUB_RELAY_PRESENCE};
use crate::storage::keys::KeySchema;

/// Default capacity of each per-session fan-out queue: how many presence
/// events a local JoinSession stream's queue cursor may fall behind before
/// it loses any. Up to `RECV_BATCH - 1` more events may already be taken
/// into the stream's read-ahead buffer past that cursor, and those can no
/// longer be lost (SDD §13.1).
///
/// A lag bound, not a reservation: beyond a small fixed floor, a queue holds
/// memory only for the events some receiver has not read yet (see
/// `presence::fanout`), so raising it costs nothing while streams keep up.
/// Overridable at construction time via
/// [`LocalPresenceRouter::with_capacity`], which `main` feeds from
/// `LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY`. A receiver that falls further
/// behind sees [`PresenceRecvError::Lagged`](super::PresenceRecvError::Lagged);
/// the JoinSession loop counts the loss on
/// `live_edit_presence_broadcast_dropped_total`.
///
/// `Config::from_env` reads this constant as that variable's default, so the
/// code default and the environment default cannot drift.
pub const DEFAULT_BROADCAST_CHANNEL_CAPACITY: usize = 4096;

/// A user connected to a session via JoinSession.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedUser {
    pub connection_id: String,
    pub user_name: String,
    pub user_application: String,
}

/// Events delivered to JoinSession streams via the LocalPresenceRouter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresenceEvent {
    PeerJoined(ConnectedUser),
    PeerLeft {
        connection_id: String,
    },
    PeerMessage {
        connection_id: String,
        data: Vec<u8>,
    },
}

/// Errors when parsing a presence event wire-format string.
///
/// No variant's `Display` renders any byte of the payload: the relay logs
/// the error on a parse failure, and the payload carries user names
/// (SDD §13.1).
#[derive(Debug, thiserror::Error)]
pub enum PresenceEventParseError {
    /// The payload's first field is not a known event tag. Only the tag's
    /// length is kept, never its bytes.
    #[error("unknown event type ({tag_bytes}-byte tag)")]
    UnknownType { tag_bytes: usize },

    #[error("wrong field count for {event_type}: expected {expected}, got {got}")]
    WrongFieldCount {
        event_type: &'static str,
        expected: usize,
        got: usize,
    },

    /// The `msg` data field is not valid base64. The decoder's error stays
    /// the `source` but is kept out of the message, because its `Display`
    /// prints the rejected byte's value.
    #[error("invalid base64 in PeerMessage data")]
    Base64Decode(#[from] base64::DecodeError),
}

/// Escape one `joined` field for the wire: `%` becomes `%25` and `|`
/// becomes `%7C`; every other character is copied unchanged, so a field
/// holding neither is returned as is, byte for byte. Not
/// `utf8_percent_encode`, which would also encode every non-ASCII byte and
/// so change ordinary names on the wire.
fn escape_field(field: &str) -> Cow<'_, str> {
    if !field.contains(['%', '|']) {
        return Cow::Borrowed(field);
    }
    let mut out = String::with_capacity(field.len() + 8);
    for c in field.chars() {
        match c {
            '%' => out.push_str("%25"),
            '|' => out.push_str("%7C"),
            other => out.push(other),
        }
    }
    Cow::Owned(out)
}

/// Undo [`escape_field`] on one received `joined` field, in one
/// left-to-right pass: `%25` becomes `%`, and `%7C` becomes `|`, matched
/// only in the uppercase form the encoder writes. Any other `%`, lowercase
/// `%7c` included, is copied as is and scanning resumes at the byte after
/// it, so the decode cannot fail: an older replica's raw `100%`, `%FF` or
/// `%7c` passes through unchanged. One pass, not sequential `replace`
/// calls, so an escaped literal `%7C` (`%257C`) decodes back to `%7C`,
/// never to `|`.
fn unescape_field(field: &str) -> Cow<'_, str> {
    if !field.contains('%') {
        return Cow::Borrowed(field);
    }
    let mut out = String::with_capacity(field.len());
    let mut rest = field;
    while let Some(i) = rest.find('%') {
        out.push_str(&rest[..i]);
        // `%` is one byte, so the byte after it starts a character.
        let after = &rest[i + 1..];
        if let Some(r) = after.strip_prefix("25") {
            out.push('%');
            rest = r;
        } else if let Some(r) = after.strip_prefix("7C") {
            out.push('|');
            rest = r;
        } else {
            out.push('%');
            rest = after;
        }
    }
    out.push_str(rest);
    Cow::Owned(out)
}

impl PresenceEvent {
    /// Encode a presence event to the wire format published on Valkey pub/sub.
    ///
    /// Each `joined` field is escaped ([`escape_field`]: `%` → `%25`,
    /// `|` → `%7C`), so the payload always holds exactly three `|` and a
    /// name or application containing `|` reaches every peer whole. A field
    /// with neither character is written byte for byte, as every
    /// server-minted `connection_id` is, so `left` and `msg`, which are not
    /// escaped, carry the same `connection_id` bytes as `joined`. The
    /// escaping exists only on this wire: the presence info hash stores the
    /// raw values and the membership replay reads them raw (SDD §13.1).
    pub fn encode(&self) -> String {
        match self {
            PresenceEvent::PeerJoined(user) => {
                format!(
                    "joined|{}|{}|{}",
                    escape_field(&user.connection_id),
                    escape_field(&user.user_name),
                    escape_field(&user.user_application)
                )
            }
            PresenceEvent::PeerLeft { connection_id } => {
                format!("left|{connection_id}")
            }
            PresenceEvent::PeerMessage {
                connection_id,
                data,
            } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                format!("msg|{connection_id}|{b64}")
            }
        }
    }

    /// Parse a pipe-delimited presence event from the wire format.
    ///
    /// A `joined` payload must split into four fields, so one with fewer than
    /// three `|` is rejected; one with more is not, since the split stops at
    /// the third `|` and everything after it stays in `user_application`.
    /// Only an older replica's raw payload can hold such an extra `|`. Fields
    /// 1–3 are then decoded by [`unescape_field`], which decodes only `%25`
    /// and uppercase `%7C`, exactly what the encoder writes, and copies any
    /// other `%` as is. The set is that narrow because an older replica
    /// publishes names raw, and a wider decode (any `%HH`, or lowercase
    /// `%7c`) would alter more of those names; the decode is lenient, with
    /// no failure path, so no raw name can make a join fail to parse. An
    /// older replica's literal `%25` or `%7C` still decodes, which is the
    /// accepted mixed-version residual (SDD §13.1).
    pub fn parse(payload: &str) -> Result<Self, PresenceEventParseError> {
        // Split into at most 4 parts (joined has 4 fields).
        let parts: Vec<&str> = payload.splitn(4, '|').collect();

        match parts.first().copied() {
            Some("joined") => {
                if parts.len() != 4 {
                    return Err(PresenceEventParseError::WrongFieldCount {
                        event_type: "joined",
                        expected: 4,
                        got: parts.len(),
                    });
                }
                Ok(PresenceEvent::PeerJoined(ConnectedUser {
                    connection_id: unescape_field(parts[1]).into_owned(),
                    user_name: unescape_field(parts[2]).into_owned(),
                    user_application: unescape_field(parts[3]).into_owned(),
                }))
            }
            Some("left") => {
                if parts.len() != 2 {
                    return Err(PresenceEventParseError::WrongFieldCount {
                        event_type: "left",
                        expected: 2,
                        got: parts.len(),
                    });
                }
                Ok(PresenceEvent::PeerLeft {
                    connection_id: parts[1].to_string(),
                })
            }
            Some("msg") => {
                if parts.len() != 3 {
                    return Err(PresenceEventParseError::WrongFieldCount {
                        event_type: "msg",
                        expected: 3,
                        got: parts.len(),
                    });
                }
                let data = base64::engine::general_purpose::STANDARD.decode(parts[2])?;
                Ok(PresenceEvent::PeerMessage {
                    connection_id: parts[1].to_string(),
                    data,
                })
            }
            Some(other) => Err(PresenceEventParseError::UnknownType {
                tag_bytes: other.len(),
            }),
            None => Err(PresenceEventParseError::UnknownType { tag_bytes: 0 }),
        }
    }
}

/// Per-session entry in `LocalPresenceRouter.sessions`.
///
/// Holds the fan-out queue that carries the session's events to its local
/// JoinSession streams, and — under one mutex — how many local connections
/// hold the entry and where this replica's Valkey subscription to the
/// session's channel stands.
pub struct SessionEntry {
    fanout: Arc<SessionFanout>,
    state: Mutex<EntryState>,
}

/// The mutable part of a [`SessionEntry`].
struct EntryState {
    /// Local connections that joined and have not yet left. Every
    /// [`LocalPresenceRouter::join_channel`] adds one and every
    /// [`LocalPresenceRouter::leave_channel`] removes one, whatever became of
    /// the join's subscription.
    connections: usize,
    channel: ChannelState,
}

/// Where this replica's Valkey subscription to a session's presence channel
/// stands. At most one `SUBSCRIBE` or `UNSUBSCRIBE` for the channel is in
/// flight at any time, and a join proceeds only once the channel is subscribed.
enum ChannelState {
    /// No subscription is held or being made: the entry is new, or the last
    /// `SUBSCRIBE` failed. The next joiner issues `SUBSCRIBE`.
    Unsubscribed,
    /// One joiner is issuing `SUBSCRIBE`; every other joiner waits on
    /// `outcome` for its result.
    Subscribing {
        outcome: watch::Receiver<SubscribeOutcome>,
    },
    /// The channel is subscribed.
    Subscribed,
    /// The last local connection left and is issuing `UNSUBSCRIBE`; `done`
    /// turns true once that command has completed. Only ever with zero
    /// connections: a joiner that arrives meanwhile moves the entry to
    /// `Subscribing` and sends its `SUBSCRIBE` only after `done`.
    Unsubscribing { done: watch::Receiver<bool> },
}

/// The result of an in-flight `SUBSCRIBE`, as its waiters see it.
#[derive(Clone)]
enum SubscribeOutcome {
    Pending,
    Subscribed,
    /// The `SUBSCRIBE` failed; the reason is the error as the issuing joiner
    /// saw it.
    Failed(Arc<str>),
}

impl SessionEntry {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            fanout: Arc::new(SessionFanout::new(capacity)),
            state: Mutex::new(EntryState {
                connections: 0,
                channel: ChannelState::Unsubscribed,
            }),
        }
    }

    /// Lock the entry's state. A panic while it was held cannot leave it
    /// half-updated (every critical section is a few field writes), so a
    /// poisoned lock is recovered rather than propagated.
    fn lock(&self) -> MutexGuard<'_, EntryState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What a joiner must do about the session's Valkey channel subscription
/// before it may go on (see [`LocalPresenceRouter::join_channel`]).
pub enum ChannelJoin {
    /// The channel is already subscribed: go on.
    Subscribed,
    /// Another local joiner's `SUBSCRIBE` is in flight: wait for its result
    /// with [`ChannelWait::wait`], and go on only if it succeeded.
    Wait(ChannelWait),
    /// This joiner must subscribe the channel: wait for any in-flight
    /// `UNSUBSCRIBE` ([`ChannelSubscribe::wait_for_prior_unsubscribe`]), send
    /// `SUBSCRIBE`, and report the result with [`ChannelSubscribe::succeeded`]
    /// or [`ChannelSubscribe::failed`].
    Subscribe(ChannelSubscribe),
}

/// A joiner's wait on another local joiner's in-flight `SUBSCRIBE`.
pub struct ChannelWait {
    outcome: watch::Receiver<SubscribeOutcome>,
}

impl ChannelWait {
    /// Wait for the in-flight `SUBSCRIBE`. `Ok` once it succeeded; `Err` with
    /// the issuing joiner's error if it failed. Bounded by that joiner's own
    /// commands: its wait for a prior `UNSUBSCRIBE` and its `SUBSCRIBE`, each
    /// under the store command timeout.
    pub async fn wait(mut self) -> Result<(), Arc<str>> {
        loop {
            let seen = self.outcome.borrow_and_update().clone();
            match seen {
                SubscribeOutcome::Subscribed => return Ok(()),
                SubscribeOutcome::Failed(reason) => return Err(reason),
                SubscribeOutcome::Pending => {}
            }
            if self.outcome.changed().await.is_err() {
                // The issuer reports before it drops its sender, so this is
                // unreachable in practice; treat it as a failure, not a hang.
                return Err(Arc::from("the subscribing join ended without a result"));
            }
        }
    }
}

/// The one joiner's duty to subscribe a session's Valkey channel.
///
/// Waiters see the result as soon as it is reported. Dropped without a
/// report — the join's future dropped mid-`SUBSCRIBE` — it reports a failure,
/// so no waiter is left waiting and the next joiner subscribes again.
///
/// **Report (or drop) this duty before the join leaves.** A duty is tied to
/// the subscribing round it opened, and its report changes the entry's state
/// only while that round is still the entry's current one. If its join left
/// first anyway and so took the count to zero, [`LocalPresenceRouter::leave_channel`]
/// has already started an unsubscribe round, and the late report only tells
/// any waiters the result; it cannot overwrite a newer round's state. The
/// Valkey channel itself can still end subscribed, or tracked by fred, with no
/// router entry in that case, depending on how the commands interleave; see
/// [`LocalPresenceRouter::leave_channel`].
pub struct ChannelSubscribe {
    entry: Arc<SessionEntry>,
    outcome: watch::Sender<SubscribeOutcome>,
    /// The last local leaver's in-flight `UNSUBSCRIBE`, when this joiner
    /// arrived during it.
    prior_unsubscribe: Option<watch::Receiver<bool>>,
    reported: bool,
}

impl ChannelSubscribe {
    /// Wait until an `UNSUBSCRIBE` the last local leaver had in flight when
    /// this joiner arrived has completed, so this joiner's `SUBSCRIBE` is sent
    /// after it and the channel ends subscribed. Returns at once if there was
    /// none. Bounded by the leaver's `UNSUBSCRIBE`, under the store command
    /// timeout.
    pub async fn wait_for_prior_unsubscribe(&mut self) {
        let Some(mut done) = self.prior_unsubscribe.take() else {
            return;
        };
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                // The leaver reports before it drops its sender.
                return;
            }
        }
    }

    /// The `SUBSCRIBE` succeeded: the channel is subscribed, and every waiter
    /// goes on.
    pub fn succeeded(mut self) {
        self.report(SubscribeOutcome::Subscribed);
    }

    /// The `SUBSCRIBE` failed with `reason`: every waiter fails too, and the
    /// session's channel is left unsubscribed, so the next joiner sends
    /// `SUBSCRIBE` again. The failed joiners still hold their router counts;
    /// each releases it through [`LocalPresenceRouter::leave_channel`].
    pub fn failed(mut self, reason: &str) {
        self.report(SubscribeOutcome::Failed(Arc::from(reason)));
    }

    fn report(&mut self, outcome: SubscribeOutcome) {
        if self.reported {
            return;
        }
        self.reported = true;
        {
            let mut state = self.entry.lock();
            // Only this duty's own round is moved on. While the join holds its
            // count nothing else leaves `Subscribing`: other joiners wait, and
            // no leaver can take the count to zero. If the join left before
            // reporting, a later round owns the state and is left alone.
            let own_round = matches!(
                &state.channel,
                ChannelState::Subscribing { outcome } if outcome.same_channel(&self.outcome.subscribe())
            );
            if own_round {
                state.channel = match outcome {
                    SubscribeOutcome::Subscribed => ChannelState::Subscribed,
                    _ => ChannelState::Unsubscribed,
                };
            }
        }
        self.outcome.send_replace(outcome);
    }
}

impl Drop for ChannelSubscribe {
    fn drop(&mut self) {
        self.report(SubscribeOutcome::Failed(Arc::from(
            "the subscribing join ended before its SUBSCRIBE completed",
        )));
    }
}

/// The last local leaver's duty to unsubscribe a session's Valkey channel.
///
/// Send `UNSUBSCRIBE`, then call [`Self::finish`] whatever its result. Until
/// then the entry stays in the router in the `Unsubscribing` state, so a joiner
/// that arrives meanwhile queues its `SUBSCRIBE` behind this `UNSUBSCRIBE`.
/// Dropped unfinished, it finishes.
///
/// A duty is tied to the unsubscribing round it opened: `finish` removes the
/// entry only while that round is still the entry's current state. A later
/// round replacing this one before it finishes is left for its own duty to
/// finish, so the entry is never removed while a newer `UNSUBSCRIBE` a later
/// joiner must wait for is still in flight. Such a later round arises only
/// when a subscribing join that took this entry over breaks the caller
/// contract — leaves while its duty is unreported (see
/// [`LocalPresenceRouter::leave_channel`]), or reports or drops the duty
/// without first waiting in [`ChannelSubscribe::wait_for_prior_unsubscribe`]
/// and then leaves. The JoinSession task does neither: a join whose
/// `SUBSCRIBE` fails has already waited for this `UNSUBSCRIBE` to finish.
pub struct ChannelUnsubscribe<'a> {
    router: &'a LocalPresenceRouter,
    session_id: String,
    entry: Arc<SessionEntry>,
    done: watch::Sender<bool>,
    finished: bool,
}

impl ChannelUnsubscribe<'_> {
    /// The `UNSUBSCRIBE` has completed (or failed). Removes the session's
    /// entry if it is still in this duty's unsubscribing round with no
    /// connections — not if a joiner has taken it over meanwhile (that joiner
    /// now sends its own `SUBSCRIBE`), nor if a later round has replaced this
    /// one, which happens only on a caller-contract breach (see
    /// [`ChannelUnsubscribe`]) — then releases any joiner waiting on this
    /// `UNSUBSCRIBE`.
    pub fn finish(mut self) {
        self.finish_inner();
    }

    fn finish_inner(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Entry::Occupied(occupied) = self.router.sessions.entry(self.session_id.clone()) {
            if Arc::ptr_eq(occupied.get(), &self.entry) {
                let idle = {
                    let state = self.entry.lock();
                    state.connections == 0
                        && matches!(
                            &state.channel,
                            ChannelState::Unsubscribing { done } if done.same_channel(&self.done.subscribe())
                        )
                };
                if idle {
                    occupied.remove();
                    // No send reaches the queue once the entry is gone; a
                    // receiver that outlives it ends `Closed` rather than
                    // waiting forever.
                    self.entry.fanout.close();
                }
            }
        }
        self.done.send_replace(true);
    }
}

impl Drop for ChannelUnsubscribe<'_> {
    fn drop(&mut self) {
        self.finish_inner();
    }
}

/// Per-replica in-memory router that multiplexes presence events from Valkey
/// pub/sub to local JoinSession streams.
///
/// Each session with at least one local connection gets a `SessionEntry` whose
/// fan-out queue (`presence::fanout`) carries events to the
/// [`PresenceReceiver`]s handed back by [`Self::join_channel`]; the queue
/// holds memory for events a receiver has not read yet and, once drained,
/// only a small fixed floor (see `presence::fanout`), whatever the
/// capacity. The entry also
/// counts its local connections and tracks the replica's Valkey subscription
/// to the session's channel as a single-flight state (unsubscribed,
/// subscribing, subscribed, unsubscribing):
///
/// - the first local joiner sends `SUBSCRIBE`, and every joiner that arrives
///   while it is in flight waits for its result, so no join goes on — to
///   registration, `SessionConnected` and the replay — before the channel is
///   subscribed and a peer's announcement can reach it;
/// - a failed `SUBSCRIBE` fails every waiter too and leaves the channel
///   unsubscribed, so the next joiner tries again;
/// - the last local leaver sends `UNSUBSCRIBE`, and the entry stays until that
///   completes, so a joiner that arrives meanwhile sends its `SUBSCRIBE` only
///   after it and the channel ends subscribed.
///
/// Locks are taken in one order: the `DashMap` shard lock for the session,
/// then the entry's state mutex, then the entry's fan-out mutex, each step
/// optional. Joining, leaving and `finish` take the shard lock first, then
/// the state mutex (a join then subscribes to the fan-out under both, and
/// `finish` closes it after releasing them); a `SUBSCRIBE` result is
/// recorded under the state mutex alone; [`Self::broadcast`] takes the shard
/// lock, then the fan-out mutex; a receiver takes the fan-out mutex alone,
/// which takes no other lock. No lock is held across an `.await`: the Valkey
/// commands themselves are sent by the callers, between the calls that hand
/// out and settle these duties.
pub struct LocalPresenceRouter {
    /// session_id -> per-session state shared with subscribers of that session.
    sessions: DashMap<String, Arc<SessionEntry>>,
    /// Capacity of each new per-session fan-out queue: the lag bound, in
    /// events (see [`DEFAULT_BROADCAST_CHANNEL_CAPACITY`]).
    channel_capacity: usize,
    /// session_id -> the keeper tick at which a lease renewal on this replica
    /// last re-added the session to the session index (SDD SR-7.2b). Read and
    /// written only by [`Self::index_renewal_due`], and only while the
    /// session has local connections. [`Self::leave_channel`] removes the
    /// entry on the last local connection's leave, so the map is bounded by
    /// the replica's live sessions.
    index_renewals: DashMap<String, tokio::time::Instant>,
}

impl Default for LocalPresenceRouter {
    fn default() -> Self {
        Self {
            sessions: DashMap::new(),
            channel_capacity: DEFAULT_BROADCAST_CHANNEL_CAPACITY,
            index_renewals: DashMap::new(),
        }
    }
}

impl LocalPresenceRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a router with a non-default capacity: the lag bound, in
    /// events, of every per-session fan-out queue `join_channel()` creates,
    /// applied exactly (see [`DEFAULT_BROADCAST_CHANNEL_CAPACITY`]).
    ///
    /// Panics if `capacity` is 0: a queue that may hold no event would lose
    /// every event before any stream could read it.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "LocalPresenceRouter capacity must be > 0; with 0 every presence event would be lost",
        );
        Self {
            sessions: DashMap::new(),
            channel_capacity: capacity,
            index_renewals: DashMap::new(),
        }
    }

    /// Return the fan-out capacity (lag bound, in events) this router uses
    /// for new per-session entries. Exposed for tests and debug logging.
    pub fn channel_capacity(&self) -> usize {
        self.channel_capacity
    }

    /// Join a session locally: count one more local connection, hand back a
    /// receiver for the session's events (live from this call on), and say
    /// what this joiner must do about the Valkey channel subscription before
    /// it goes on (see [`ChannelJoin`]).
    ///
    /// `connection_id` is the joining connection's id: its receiver does not
    /// count that connection's own lost joins and leaves as membership loss,
    /// since its stream discards them anyway (see `presence::fanout`). `None`
    /// for a receiver with no connection of its own; every lost join or leave
    /// then counts.
    ///
    /// Every call must be paired with exactly one [`Self::leave_channel`] —
    /// including when the subscription then fails, since the count is taken
    /// here regardless.
    pub fn join_channel(
        &self,
        session_id: &str,
        connection_id: Option<&str>,
    ) -> (PresenceReceiver, ChannelJoin) {
        let capacity = self.channel_capacity;
        let shard = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(SessionEntry::with_capacity(capacity)));
        let entry = Arc::clone(shard.value());
        let mut state = entry.lock();
        state.connections += 1;
        let rx = entry.fanout.subscribe(connection_id);
        let prior_unsubscribe = match &state.channel {
            ChannelState::Subscribed => return (rx, ChannelJoin::Subscribed),
            ChannelState::Subscribing { outcome } => {
                return (
                    rx,
                    ChannelJoin::Wait(ChannelWait {
                        outcome: outcome.clone(),
                    }),
                );
            }
            ChannelState::Unsubscribed => None,
            ChannelState::Unsubscribing { done } => Some(done.clone()),
        };
        let (outcome_tx, outcome_rx) = watch::channel(SubscribeOutcome::Pending);
        state.channel = ChannelState::Subscribing {
            outcome: outcome_rx,
        };
        drop(state);
        drop(shard);
        (
            rx,
            ChannelJoin::Subscribe(ChannelSubscribe {
                entry,
                outcome: outcome_tx,
                prior_unsubscribe,
                reported: false,
            }),
        )
    }

    /// Leave a session locally: count one fewer local connection. Returns the
    /// duty to unsubscribe the session's Valkey channel when this was the last
    /// local connection, and `None` otherwise — including for a session with
    /// no entry or no connections left (a double leave, logged at debug).
    ///
    /// The entry is not removed here: the returned [`ChannelUnsubscribe`]
    /// removes it once the `UNSUBSCRIBE` has completed, unless a joiner has
    /// arrived in the meantime. The session's index-renewal record (see
    /// [`Self::index_renewal_due`]) is removed here, on the last local leave.
    ///
    /// **Caller contract:** a join that holds an unreported
    /// [`ChannelSubscribe`] must report or drop it before leaving. The
    /// JoinSession task meets this by construction: the duty lives in
    /// `run_join_session_inner`'s step-3 match arm and is dropped before that
    /// function returns, and `run_join_session` calls `teardown` — the only
    /// production caller of this function — only after it has returned.
    ///
    /// Breaking the contract panics a debug build, at a `debug_assert!` here,
    /// so the release-only handling below is never exercised by the debug test
    /// profile. In a release build the router's own state stays consistent —
    /// the unsubscribe round starts regardless, and the duty's later report
    /// leaves it alone (see [`ChannelSubscribe`]) — but the Valkey channel's
    /// end state depends on how the duty's `SUBSCRIBE` interleaves with this
    /// round's `UNSUBSCRIBE`: it can end subscribed, or left in fred's
    /// tracked-channel set (re-subscribed on reconnect), with no router entry,
    /// until a later last leaver of the session unsubscribes it.
    pub fn leave_channel(&self, session_id: &str) -> Option<ChannelUnsubscribe<'_>> {
        let Entry::Occupied(occupied) = self.sessions.entry(session_id.to_string()) else {
            tracing::debug!(
                session_id = %session_id,
                "leave_channel called for session with no entry"
            );
            return None;
        };
        let entry = Arc::clone(occupied.get());
        let mut state = entry.lock();
        if state.connections == 0 {
            tracing::debug!(
                session_id = %session_id,
                "leave_channel called with no local connections left"
            );
            return None;
        }
        state.connections -= 1;
        if state.connections > 0 {
            return None;
        }
        // No local connection is left, so no local lease keeper renews this
        // session: drop its index-renewal record (SDD SR-7.2b). Under the
        // session's shard lock and entry mutex, the same locks
        // `index_renewal_due` records under, so no record can outlive this.
        self.index_renewals.remove(session_id);
        // The last local connection. The state is `Subscribing` here only if
        // the subscribing join broke the caller contract and left before
        // reporting, which the JoinSession task cannot do. Replacing it keeps
        // the router consistent: no waiters are left (each holds a
        // connection), and the duty's report will no longer match the current
        // round. What the Valkey channel ends up as is described in this
        // function's rustdoc. The assert below panics a debug build on such a
        // breach, so the replacement that follows it is exercised only in
        // release builds, never by the debug test profile.
        debug_assert!(
            !matches!(state.channel, ChannelState::Subscribing { .. }),
            "leave_channel: the last local connection left while its session's \
             SUBSCRIBE duty was unreported"
        );
        let (done_tx, done_rx) = watch::channel(false);
        state.channel = ChannelState::Unsubscribing { done: done_rx };
        drop(state);
        drop(occupied);
        Some(ChannelUnsubscribe {
            router: self,
            session_id: session_id.to_string(),
            entry,
            done: done_tx,
            finished: false,
        })
    }

    /// Whether a lease renewal at keeper tick `tick` should re-add
    /// `session_id` to the session index (SDD SR-4.5, SR-7.2b): at most once
    /// per session per replica per `interval`, measured in keeper tick
    /// instants. Returns `true` and records `tick` when no renewal re-add is
    /// recorded for the session or the recorded one is at least `interval`
    /// before `tick`; returns `false` otherwise.
    ///
    /// Keeper ticks are used rather than the time of the call so that a
    /// session's only connection, whose keeper ticks exactly one interval
    /// apart, is re-added on every renewal: the index heal it backs stays
    /// within one refresh interval. With several local connections, the first
    /// renewal of each interval re-adds and the others skip. If the connection
    /// that recorded then stops renewing (it leaves, or its next refresh does
    /// not renew), the next re-add comes from a sibling's first tick at or
    /// after the record plus one interval. That is up to
    /// almost two intervals after the last re-add. SR-7.2b accepts this: it
    /// only thins the second layer behind the index-removal protocol.
    ///
    /// Only the keeper's plain renewal is gated. A re-add after a rejoin or an
    /// info repair, and the join's own re-adds, are never passed through here.
    ///
    /// A session with no router entry or no local connections left is not
    /// recorded, and the call returns `true`: the keeper asking belongs to a
    /// connection whose teardown has already left the router, so the re-add
    /// is ungated, as it was before this gate, and no record is left behind
    /// for [`Self::leave_channel`] to miss. The check and the record happen
    /// under the session's shard lock and entry mutex, the locks
    /// [`Self::leave_channel`] removes the record under.
    pub fn index_renewal_due(
        &self,
        session_id: &str,
        tick: tokio::time::Instant,
        interval: std::time::Duration,
    ) -> bool {
        let Some(shard) = self.sessions.get(session_id) else {
            return true;
        };
        let state = shard.value().lock();
        if state.connections == 0 {
            return true;
        }
        match self.index_renewals.entry(session_id.to_string()) {
            Entry::Occupied(mut last) => {
                if tick.saturating_duration_since(*last.get()) < interval {
                    return false;
                }
                last.insert(tick);
                true
            }
            Entry::Vacant(vacant) => {
                vacant.insert(tick);
                true
            }
        }
    }

    /// **Test-only.** Join a session locally with the channel treated as
    /// subscribed — no Valkey `SUBSCRIBE` is sent and none is awaited. Returns
    /// a receiver and whether this call was the one that would have had to
    /// subscribe. For tests that need a local receiver or exercise the counts;
    /// production joins go through [`Self::join_channel`]. The receiver has
    /// no connection of its own, so every lost join or leave counts as
    /// membership loss for it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn subscribe(&self, session_id: &str) -> (PresenceReceiver, bool) {
        let (rx, join) = self.join_channel(session_id, None);
        match join {
            ChannelJoin::Subscribe(subscribe) => {
                subscribe.succeeded();
                (rx, true)
            }
            ChannelJoin::Subscribed | ChannelJoin::Wait(_) => (rx, false),
        }
    }

    /// **Test-only.** Leave a session locally with the channel treated as
    /// unsubscribed at once — no Valkey `UNSUBSCRIBE` is sent. Returns whether
    /// this was the last local connection. Pairs with [`Self::subscribe`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn unsubscribe(&self, session_id: &str) -> bool {
        match self.leave_channel(session_id) {
            Some(unsubscribe) => {
                unsubscribe.finish();
                true
            }
            None => false,
        }
    }

    /// Broadcast a presence event to all local subscribers for a session.
    ///
    /// Returns the number of receivers the event was queued for, or 0 if there
    /// are no local subscribers (the event is then dropped).
    pub fn broadcast(&self, session_id: &str, event: PresenceEvent) -> usize {
        if let Some(entry) = self.sessions.get(session_id) {
            entry.value().fanout.send(event)
        } else {
            0
        }
    }

    /// Test-only helper: forcibly remove the DashMap entry for a session,
    /// regardless of its connection count, and close its fan-out as
    /// `ChannelUnsubscribe::finish` does. Production code must go through
    /// `leave_channel()` so that the Valkey pub/sub subscription is released
    /// in the same transition.
    #[cfg(test)]
    pub fn remove_session(&self, session_id: &str) {
        if let Some((_, entry)) = self.sessions.remove(session_id) {
            entry.fanout.close();
        }
    }

    /// **Test-only.** The slots the session's fan-out queue has allocated,
    /// or `None` for a session with no entry. Lets a test see the queue give
    /// its memory back once drained.
    #[cfg(any(test, feature = "test-support"))]
    pub fn fanout_allocated_slots(&self, session_id: &str) -> Option<usize> {
        self.sessions
            .get(session_id)
            .map(|entry| entry.value().fanout.allocated_slots())
    }

    /// Check if a session has an entry: at least one local connection, or the
    /// last one's `UNSUBSCRIBE` still in flight.
    pub fn has_session(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Return the number of sessions with an entry (see [`Self::has_session`]).
    /// The source of the `live_active_sessions` gauge.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Return the number of JoinSession connections currently joined across
    /// all local sessions (the sum of every entry's connection count). Source
    /// of the `live_active_connections` gauge. O(sessions) per call; only
    /// invoked at metric collection.
    pub fn connection_count(&self) -> usize {
        self.sessions
            .iter()
            .map(|entry| entry.value().lock().connections)
            .sum()
    }

    /// Return the IDs of all sessions with an entry: at least one local
    /// connection, or the last one's `UNSUBSCRIBE` still in flight (see
    /// [`Self::has_session`]).
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

/// Publish a presence event to the Valkey pub/sub channel for a session.
///
/// This is the only Rust-side presence publish, and in production it carries
/// only a connection's `PeerMessage`s, from the JoinSession bidi loop. `joined`
/// and `left` — including the presence sweeper's — are published by the
/// membership scripts in `storage::presence`, from the node
/// that owns the session's `{sid}` slot; so that every event of a session
/// leaves that same node, in the order it executed them, this `PUBLISH` is
/// routed there too, pinned to `presence_publish_pin`'s slot (SDD SR-1.M).
/// Left to fred's default routing it would go to the node owning the channel
/// name's slot — the channel carries no hash tag — and a peer on a cluster
/// could then see a connection's message before its `joined`. `PUBLISH` is
/// accepted on any cluster node, so the pin changes only where the event
/// originates. On a standalone store it is a no-op.
///
/// The pool's default retries apply, so a connection lost after the `PUBLISH`
/// ran can deliver one `PeerMessage` to the session's peers twice (fred
/// re-sends it on the new connection). A command timeout is not counted here;
/// the caller counts it.
pub async fn publish_presence_event(
    pool: &fred::clients::Pool,
    key_schema: &KeySchema,
    session_id: &str,
    event: &PresenceEvent,
) -> Result<(), fred::error::Error> {
    let channel = key_schema.presence_events_channel(session_id);
    let payload = event.encode();
    let pin = Options {
        cluster_hash: Some(presence_publish_pin(key_schema, session_id)),
        ..Default::default()
    };
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: the peers receive the `PeerMessage` twice.
    pool.next()
        .with_options(&pin)
        .publish::<(), _, _>(&channel, &payload)
        .await
}

/// The cluster hash every Rust-side presence `PUBLISH` for `session_id` is
/// routed by: the slot of the session's member set
/// ([`KeySchema::presence_members`]), which is the `{sid}` slot the membership
/// scripts run in. See [`publish_presence_event`].
pub(crate) fn presence_publish_pin(key_schema: &KeySchema, session_id: &str) -> ClusterHash {
    ClusterHash::from(key_schema.presence_members(session_id).as_str())
}

/// Spawn a background task that relays presence events from Valkey pub/sub
/// to the LocalPresenceRouter.
///
/// The task reads the fred SubscriberClient's message stream. For each
/// message on a `le:presence:events:{session_id}` channel whose session has
/// local subscribers, it parses the event and broadcasts it to them via the
/// router.
///
/// That stream is a broadcast channel of
/// `storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY` messages, shared with
/// every other receiver on the connection. The receiver is taken before the
/// task is spawned, so the relay sees every message published once this
/// function returns. The task drains everything queued each time it runs
/// ([`PubsubRelayReceiver`]), yielding after one channel capacity's worth of
/// messages. If it still falls more than the capacity behind, the skipped
/// messages are lost for every local stream: each drain batch that lost any
/// logs one WARN and adds their sum to
/// `live_edit_pubsub_relay_lagged_total{relay="presence"}` on `metrics`.
///
/// The task exits when the cancellation token fires.
pub fn spawn_presence_subscriber(
    subscriber: SubscriberClient,
    router: std::sync::Arc<LocalPresenceRouter>,
    key_schema: KeySchema,
    metrics: SharedMetrics,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let mut messages = PubsubRelayReceiver::new(&subscriber);
    tokio::spawn(async move {
        // Channel subscriptions are managed by the JoinSession join tasks, through
        // the `SUBSCRIBE` / `UNSUBSCRIBE` duties `LocalPresenceRouter::join_channel`
        // and `leave_channel` hand out; this task only forwards what arrives.
        let prefix = key_schema.presence_events_channel("");

        loop {
            tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("presence subscriber shutting down");
                    break;
                }
                first = messages.recv() => {
                    let batch = messages.drain(first, |msg| {
                        relay_presence_message(&router, &prefix, msg);
                    });
                    if batch.lagged > 0 {
                        metrics.add_pubsub_relay_lagged(PUBSUB_RELAY_PRESENCE, batch.lagged);
                        tracing::warn!(
                            lagged = batch.lagged,
                            "presence subscriber lagged, events may have been lost"
                        );
                    }
                    match batch.end {
                        DrainEnd::Empty => {}
                        DrainEnd::YieldDue => {
                            tokio::task::yield_now().await;
                            if cancel.is_cancelled() {
                                tracing::info!("presence subscriber shutting down");
                                break;
                            }
                        }
                        DrainEnd::Closed => {
                            tracing::info!("presence subscriber channel closed");
                            break;
                        }
                    }
                }
            }
        }
    })
}

/// The presence relay's per-message step, for the relay bench
/// (`benches/presence_fanout.rs`): [`relay_presence_message`] itself, which
/// stays private to the relay.
#[cfg(feature = "test-support")]
pub fn relay_presence_message_for_bench(router: &LocalPresenceRouter, prefix: &str, msg: Message) {
    relay_presence_message(router, prefix, msg);
}

/// The presence relay's per-message step: forward one pub/sub message to the
/// router if it is a presence event for a session with local subscribers.
/// `prefix` is the presence-events channel prefix
/// (`KeySchema::presence_events_channel("")`).
///
/// The channel filter reads the message's channel name in place; nothing is
/// allocated for a message the filter drops.
fn relay_presence_message(router: &LocalPresenceRouter, prefix: &str, msg: Message) {
    // Extract session_id from channel name.
    let channel: &str = &msg.channel;
    let session_id = match channel.strip_prefix(prefix) {
        Some(sid) if !sid.is_empty() => sid,
        _ => return,
    };

    // Only forward events for sessions with local subscribers.
    if !router.has_session(session_id) {
        return;
    }

    let payload: String = match msg.value.convert() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                channel = %channel,
                error = %e,
                "failed to convert presence event value to string"
            );
            return;
        }
    };

    match PresenceEvent::parse(&payload) {
        Ok(event) => {
            router.broadcast(session_id, event);
        }
        Err(e) => {
            // The payload carries user names, so no byte of it is logged:
            // only its length, and an error whose text holds none of it.
            tracing::warn!(
                channel = %channel,
                payload_bytes = payload.len(),
                error = %e,
                "failed to parse presence event"
            );
        }
    }
}

/// Subscribe the Valkey subscriber client to the presence events channel for a
/// session. Sent by the join that holds the router's subscribe duty
/// ([`ChannelJoin::Subscribe`]): the first local JoinSession stream for the
/// session, or the first after a failed `SUBSCRIBE`.
pub async fn subscribe_presence_channel(
    subscriber: &SubscriberClient,
    key_schema: &KeySchema,
    session_id: &str,
) -> Result<(), fred::error::Error> {
    let channel = key_schema.presence_events_channel(session_id);
    subscriber.subscribe(&channel).await
}

/// Unsubscribe the Valkey subscriber client from the presence events channel
/// for a session. Called when the last local JoinSession stream disconnects.
pub async fn unsubscribe_presence_channel(
    subscriber: &SubscriberClient,
    key_schema: &KeySchema,
    session_id: &str,
) -> Result<(), fred::error::Error> {
    let channel = key_schema.presence_events_channel(session_id);
    subscriber.unsubscribe(&channel).await
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;

    use super::*;

    // ── 0. Gauge sources: session_count / connection_count ─────────────

    #[test]
    fn session_and_connection_counts_track_subscribe_and_unsubscribe() {
        let router = LocalPresenceRouter::with_capacity(4);
        assert_eq!((router.session_count(), router.connection_count()), (0, 0));

        let (_rx_a, first_a) = router.subscribe("s1");
        let (_rx_b, first_b) = router.subscribe("s1");
        let (_rx_c, first_c) = router.subscribe("s2");
        assert!(first_a && !first_b && first_c);
        // Two sessions hold entries; three connections are registered.
        assert_eq!((router.session_count(), router.connection_count()), (2, 3));

        assert!(!router.unsubscribe("s1"), "s1 still has one connection");
        assert_eq!((router.session_count(), router.connection_count()), (2, 2));

        assert!(
            router.unsubscribe("s1"),
            "last s1 connection removes the entry"
        );
        assert_eq!((router.session_count(), router.connection_count()), (1, 1));

        assert!(router.unsubscribe("s2"));
        assert_eq!((router.session_count(), router.connection_count()), (0, 0));

        // Double-unsubscribe must not drive either count negative / phantom.
        assert!(!router.unsubscribe("s2"));
        assert_eq!((router.session_count(), router.connection_count()), (0, 0));
    }

    // ── A. Wire format encode/decode round-trip ────────────────────────

    #[test]
    fn encode_decode_peer_joined_round_trips() {
        let event = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: "conn-1".to_string(),
            user_name: "Alice".to_string(),
            user_application: "Omniverse".to_string(),
        });
        let wire = event.encode();
        assert_eq!(wire, "joined|conn-1|Alice|Omniverse");
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn encode_decode_peer_left_round_trips() {
        let event = PresenceEvent::PeerLeft {
            connection_id: "conn-2".to_string(),
        };
        let wire = event.encode();
        assert_eq!(wire, "left|conn-2");
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn encode_decode_peer_message_round_trips() {
        let data = vec![0x00, 0x01, 0xff, 0xfe, 0x42];
        let event = PresenceEvent::PeerMessage {
            connection_id: "conn-3".to_string(),
            data: data.clone(),
        };
        let wire = event.encode();
        assert!(wire.starts_with("msg|conn-3|"));
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn encode_decode_peer_message_empty_data_round_trips() {
        let event = PresenceEvent::PeerMessage {
            connection_id: "conn-4".to_string(),
            data: vec![],
        };
        let wire = event.encode();
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn encode_decode_peer_joined_with_pipes_in_app_name() {
        let event = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: "conn-5".to_string(),
            user_name: "Bob".to_string(),
            user_application: "App|With|Pipes".to_string(),
        });
        let wire = event.encode();
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn encode_decode_peer_message_large_binary_round_trips() {
        let data: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let event = PresenceEvent::PeerMessage {
            connection_id: "conn-6".to_string(),
            data,
        };
        let wire = event.encode();
        let parsed = PresenceEvent::parse(&wire).unwrap();
        assert_eq!(parsed, event);
    }

    // ── A2. `joined` field escaping (SDD §13.1) ───────────────────────

    /// A `PeerJoined` event with the given fields.
    fn joined(connection_id: &str, user_name: &str, user_application: &str) -> PresenceEvent {
        PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: connection_id.to_owned(),
            user_name: user_name.to_owned(),
            user_application: user_application.to_owned(),
        })
    }

    /// A name or application holding `|` or `%` comes back from `parse`
    /// exactly as `encode` was given it. `|` no longer cuts the name short
    /// (`Jane Doe|Kit`, an Auth0 subject); a name holding the literal text
    /// `%25` or `%7C` is not decoded a second time; a `%` that starts no
    /// escape and an empty name survive too.
    #[test]
    fn encode_then_parse_peer_joined_with_pipes_and_percents_returns_the_input_exactly() {
        let cases = [
            ("Jane Doe|Kit", "Omniverse"),
            ("auth0|5f8a1b2c", "Kit"),
            ("100% Kit", "Omniverse"),
            ("50%25 off %7C x", "Kit"),
            ("", "Kit"),
            ("Alice", "Kit|Omniverse 100%"),
        ];

        // Every case runs before the assertion, so a failure lists each
        // case that broke, not only the first.
        let mut failures = Vec::new();
        for (user_name, user_application) in cases {
            let event = joined("c1", user_name, user_application);
            let wire = event.encode();
            match PresenceEvent::parse(&wire) {
                Ok(parsed) if parsed == event => {}
                other => failures.push(format!(
                    "({user_name:?}, {user_application:?}) sent as {wire:?} parsed to {other:?}"
                )),
            }
        }
        assert!(
            failures.is_empty(),
            "joined round trip changed the fields:\n{}",
            failures.join("\n")
        );
    }

    /// `encode` writes `|` as `%7C` and `%` as `%25`, so the payload keeps
    /// exactly three `|`. An older replica, whose parse is this
    /// `splitn(4, '|')` with no decode (every release from 0.1.0 on), still
    /// gets four fields with the connection_id intact, and shows the
    /// escaped name and application as they are on the wire.
    #[test]
    fn encode_peer_joined_escapes_pipe_and_percent_so_an_older_split_shows_the_escaped_fields() {
        let cases = [
            (
                ("c1", "Jane Doe|Kit", "Omniverse"),
                "joined|c1|Jane Doe%7CKit|Omniverse",
                ["joined", "c1", "Jane Doe%7CKit", "Omniverse"],
            ),
            (
                ("c1", "100% Kit", "Kit|Omni"),
                "joined|c1|100%25 Kit|Kit%7COmni",
                ["joined", "c1", "100%25 Kit", "Kit%7COmni"],
            ),
        ];

        for ((connection_id, user_name, user_application), expected_wire, older_fields) in cases {
            let wire = joined(connection_id, user_name, user_application).encode();
            assert_eq!(wire, expected_wire);
            let split: Vec<&str> = wire.splitn(4, '|').collect();
            assert_eq!(
                split, older_fields,
                "what an older replica's parse reads from {wire:?}"
            );
        }
    }

    /// An older replica publishes `joined` fields raw, so a name holding `|`
    /// arrives with one extra `|` per `|` in it. The split still stops at
    /// the third `|` and keeps the rest in `user_application`, however many
    /// `|` that rest holds: the older encoder's cut, unchanged by the fix,
    /// with nothing rejected and nothing decoded.
    #[test]
    fn parse_older_replicas_raw_joined_with_extra_pipes_keeps_the_rest_in_user_application() {
        let cases = [
            (
                "joined|c1|Grace Hopper|Kit|Omniverse",
                "Grace Hopper",
                "Kit|Omniverse",
            ),
            (
                "joined|c1|Ada|Lovelace|Kit|Omniverse",
                "Ada",
                "Lovelace|Kit|Omniverse",
            ),
        ];

        for (payload, user_name, user_application) in cases {
            let parsed = PresenceEvent::parse(payload)
                .unwrap_or_else(|e| panic!("{payload:?} must parse, got {e:?}"));
            assert_eq!(
                parsed,
                joined("c1", user_name, user_application),
                "{payload:?}"
            );
        }
    }

    /// The parser decodes `%25` and uppercase `%7C`, exactly what the
    /// encoder writes, and nothing else. An older replica's raw name holding
    /// one of those two literals is decoded (the accepted mixed-version
    /// residual, SDD §13.1), while any other `%HH`, such as `%41`, `%20`,
    /// `%2F` or lowercase `%7c`, is copied as is.
    #[test]
    fn parse_raw_joined_decodes_only_percent_25_and_uppercase_7c() {
        let cases = [
            ("joined|c1|50%25 off %7C x|Kit", "50% off | x", "Kit"),
            ("joined|c1|%41%20%7c%2F|a", "%41%20%7c%2F", "a"),
        ];

        for (payload, user_name, user_application) in cases {
            let parsed = PresenceEvent::parse(payload)
                .unwrap_or_else(|e| panic!("{payload:?} must parse, got {e:?}"));
            assert_eq!(
                parsed,
                joined("c1", user_name, user_application),
                "{payload:?}"
            );
        }
    }

    /// A `%` that starts neither escape, an older replica's raw
    /// `100%` or `%FF` included, is copied as is: the decode has no failure
    /// path, so no name makes a join fail to parse.
    #[test]
    fn parse_joined_with_a_percent_starting_no_escape_copies_it_unchanged() {
        for user_name in ["100%", "%", "%2", "%7", "%ZZ", "%FF"] {
            let payload = format!("joined|c1|{user_name}|Kit");
            let parsed = PresenceEvent::parse(&payload)
                .unwrap_or_else(|e| panic!("{payload:?} must parse, got {e:?}"));
            assert_eq!(parsed, joined("c1", user_name, "Kit"), "{payload:?}");
        }
    }

    // ── B. Parse error cases ──────────────────────────────────────────

    #[test]
    fn parse_unknown_type_returns_error() {
        let err = PresenceEvent::parse("unknown|data").unwrap_err();
        assert!(matches!(
            err,
            PresenceEventParseError::UnknownType { tag_bytes: 7 }
        ));
    }

    #[test]
    fn parse_empty_string_returns_error() {
        let err = PresenceEvent::parse("").unwrap_err();
        assert!(matches!(err, PresenceEventParseError::UnknownType { .. }));
    }

    #[test]
    fn parse_joined_too_few_fields_returns_error() {
        let err = PresenceEvent::parse("joined|conn-1").unwrap_err();
        assert!(matches!(
            err,
            PresenceEventParseError::WrongFieldCount {
                event_type: "joined",
                expected: 4,
                ..
            }
        ));
    }

    #[test]
    fn parse_msg_missing_data_field_returns_error() {
        let err = PresenceEvent::parse("msg|conn-1").unwrap_err();
        assert!(matches!(
            err,
            PresenceEventParseError::WrongFieldCount {
                event_type: "msg",
                ..
            }
        ));
    }

    #[test]
    fn parse_msg_invalid_base64_returns_error() {
        let err = PresenceEvent::parse("msg|conn-1|!!!invalid!!!").unwrap_err();
        assert!(matches!(err, PresenceEventParseError::Base64Decode(_)));
    }

    // ── B2. The relay's parse-failure WARN (SDD §13.1) ────────────────

    /// Capture sink for [`relay_log_output`]: the tree-wide
    /// `CapturingWriter` idiom (`src/grpc/rpc_log.rs::tests`).
    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

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
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Everything the relay logs, as JSON lines at every level, while it
    /// handles one pub/sub message carrying `payload` on `session_id`'s
    /// presence channel.
    fn relay_log_output(router: &LocalPresenceRouter, session_id: &str, payload: &str) -> String {
        use fred::types::config::Server;
        use fred::types::{MessageKind, Value};

        let prefix = KeySchema::default().presence_events_channel("");
        let msg = Message {
            channel: format!("{prefix}{session_id}").into(),
            value: Value::String(payload.to_owned().into()),
            kind: MessageKind::Message,
            server: Server::new("test", 6379),
        };
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            relay_presence_message(router, &prefix, msg);
        });
        let raw = buf.lock().unwrap_or_else(PoisonError::into_inner).clone();
        String::from_utf8(raw).expect("captured tracing output must be UTF-8")
    }

    /// A payload the relay cannot parse gives one WARN naming the channel,
    /// the payload's length and the error, and no byte of the payload
    /// reaches the log: not in a payload field, and not in the error text
    /// (a wrong field count, an unknown tag, invalid base64). Each error
    /// text is pinned whole, because a payload byte can leak through it
    /// without spelling the name: base64's own error prints the rejected
    /// byte's value (`Invalid symbol 33` for the `!`).
    #[test]
    fn relay_parse_failure_warns_once_with_channel_and_length_and_no_payload_bytes() {
        const NAME: &str = "Zelda";
        let router = LocalPresenceRouter::with_capacity(4);
        let (_rx, _first) = router.subscribe("s1");

        for (payload, error) in [
            (
                "joined|c1|Zelda",
                "wrong field count for joined: expected 4, got 3",
            ),
            ("Zelda|x", "unknown event type (5-byte tag)"),
            ("msg|c1|Zelda!", "invalid base64 in PeerMessage data"),
        ] {
            assert!(
                PresenceEvent::parse(payload).is_err(),
                "precondition: {payload:?} must not parse"
            );
            let output = relay_log_output(&router, "s1", payload);

            let warns: Vec<serde_json::Value> = output
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
                .filter(|line: &serde_json::Value| line["level"] == "WARN")
                .collect();
            assert_eq!(
                warns.len(),
                1,
                "{payload:?}: one WARN expected, got {warns:?}"
            );
            let fields = &warns[0]["fields"];
            assert_eq!(fields["message"], "failed to parse presence event");
            assert_eq!(
                fields["channel"],
                KeySchema::default().presence_events_channel("s1"),
                "{payload:?}"
            );
            assert_eq!(fields["payload_bytes"], payload.len(), "{payload:?}");
            assert_eq!(fields["error"], error, "{payload:?}");
            assert!(
                !output.contains(NAME),
                "{payload:?}: the log must carry no byte of the payload's name, got {output}"
            );
        }
    }

    // ── C. LocalPresenceRouter tests ──────────────────────────────────

    #[test]
    fn router_new_has_zero_sessions() {
        let router = LocalPresenceRouter::new();
        assert_eq!(router.session_count(), 0);
    }

    // ── b.k2t: configurable broadcast capacity ────────────────────────

    #[test]
    fn default_router_uses_default_capacity() {
        let router = LocalPresenceRouter::new();
        assert_eq!(
            router.channel_capacity(),
            DEFAULT_BROADCAST_CHANNEL_CAPACITY
        );
    }

    #[test]
    fn router_with_capacity_applies_to_new_sessions() {
        // Custom capacity must be honoured, exactly — otherwise
        // LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY would have no effect, or a
        // rounded-up one. 10 is not a power of two, so a queue that rounded
        // it up (to 16) would lose nothing below.
        let router = LocalPresenceRouter::with_capacity(10);
        assert_eq!(router.channel_capacity(), 10);

        // Flood capacity+1 events. The 11th send must drop the oldest,
        // which the receiver surfaces as PresenceRecvError::Lagged on next recv.
        let (mut rx, _) = router.subscribe("s-cap");
        for i in 0..11 {
            let _ = router.broadcast(
                "s-cap",
                PresenceEvent::PeerLeft {
                    connection_id: format!("c-{i}"),
                },
            );
        }
        // The first recv should surface Lagged(1) — one event dropped because
        // the capacity was 10 — and, the dropped event being a leave, a
        // membership loss.
        match rx.recv().now_or_never() {
            Some(Err(crate::presence::PresenceRecvError::Lagged {
                dropped,
                membership_lost,
            })) => {
                assert_eq!(
                    (dropped, membership_lost),
                    (1, true),
                    "expected one dropped PeerLeft with capacity=10 and 11 sends"
                );
            }
            other => panic!("expected Lagged(_), got {other:?}"),
        }
    }

    #[test]
    fn join_channel_with_a_connection_id_does_not_count_that_connections_lost_join() {
        // The connection id `join_channel` is given reaches the receiver:
        // a lag over only that connection's own `joined` (as a JoinSession
        // gets when its first read lags) is no membership loss for it. The
        // test-only `subscribe`, which passes none, counts the same loss.
        let router = LocalPresenceRouter::with_capacity(2);
        let (mut joiner, join) = router.join_channel("s1", Some("me"));
        expect_subscribe(join, "the first join").succeeded();
        let (mut observer, _) = router.subscribe("s1");
        let own_joined = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: "me".to_string(),
            user_name: "Me".to_string(),
            user_application: "App".to_string(),
        });
        router.broadcast("s1", own_joined);
        for i in 0..2 {
            router.broadcast(
                "s1",
                PresenceEvent::PeerMessage {
                    connection_id: "peer".to_string(),
                    data: vec![i],
                },
            );
        }

        assert!(
            matches!(
                joiner.recv().now_or_never(),
                Some(Err(crate::presence::PresenceRecvError::Lagged {
                    dropped: 1,
                    membership_lost: false,
                }))
            ),
            "the joiner lost only its own joined"
        );
        assert!(
            matches!(
                observer.recv().now_or_never(),
                Some(Err(crate::presence::PresenceRecvError::Lagged {
                    dropped: 1,
                    membership_lost: true,
                }))
            ),
            "a receiver of no connection counts it"
        );
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn router_with_capacity_zero_panics() {
        // A zero capacity would lose every event; the constructor must fail
        // fast with a clear message rather than build a router that delivers
        // nothing.
        let _router = LocalPresenceRouter::with_capacity(0);
    }

    #[test]
    fn router_subscribe_creates_session_and_returns_first_true() {
        let router = LocalPresenceRouter::new();
        let (_rx, first) = router.subscribe("s1");
        assert!(first);
        assert_eq!(router.session_count(), 1);
        assert!(router.has_session("s1"));
    }

    #[test]
    fn router_second_subscribe_returns_first_false() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first1) = router.subscribe("s1");
        let (_rx2, first2) = router.subscribe("s1");
        assert!(first1);
        assert!(!first2);
        assert_eq!(router.session_count(), 1);
    }

    #[test]
    fn router_broadcast_reaches_subscriber() {
        let router = LocalPresenceRouter::new();
        let (mut rx, _) = router.subscribe("s1");

        let event = PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: "c1".to_string(),
            user_name: "Alice".to_string(),
            user_application: "App".to_string(),
        });

        let count = router.broadcast("s1", event.clone());
        assert_eq!(count, 1);
        assert_eq!(*rx.recv().now_or_never().unwrap().unwrap(), event);
    }

    #[test]
    fn router_broadcast_fans_out_to_multiple() {
        let router = LocalPresenceRouter::new();
        let (mut rx1, _) = router.subscribe("s1");
        let (mut rx2, _) = router.subscribe("s1");

        let event = PresenceEvent::PeerLeft {
            connection_id: "c1".to_string(),
        };

        let count = router.broadcast("s1", event.clone());
        assert_eq!(count, 2);
        assert_eq!(*rx1.recv().now_or_never().unwrap().unwrap(), event);
        assert_eq!(*rx2.recv().now_or_never().unwrap().unwrap(), event);
    }

    #[test]
    fn router_broadcast_to_absent_session_returns_zero() {
        let router = LocalPresenceRouter::new();
        let count = router.broadcast(
            "no-such",
            PresenceEvent::PeerLeft {
                connection_id: "c".to_string(),
            },
        );
        assert_eq!(count, 0);
    }

    #[test]
    fn router_sessions_are_isolated() {
        let router = LocalPresenceRouter::new();
        let (mut rx1, _) = router.subscribe("s1");
        let (_rx2, _) = router.subscribe("s2");

        router.broadcast(
            "s2",
            PresenceEvent::PeerLeft {
                connection_id: "c".to_string(),
            },
        );

        assert!(rx1.recv().now_or_never().is_none());
    }

    #[test]
    fn router_remove_session_clears_channel() {
        let router = LocalPresenceRouter::new();
        let (mut rx, _) = router.subscribe("s1");
        router.remove_session("s1");
        assert_eq!(router.session_count(), 0);
        assert!(!router.has_session("s1"));
        assert!(
            matches!(
                rx.recv().now_or_never(),
                Some(Err(crate::presence::PresenceRecvError::Closed))
            ),
            "a receiver of a removed entry ends Closed rather than waiting forever"
        );
    }

    #[test]
    fn last_unsubscribe_closes_a_receiver_still_held_after_it_reads_what_was_queued() {
        // The production removal path: the last leave's `finish` removes the
        // entry and closes its fan-out. A receiver the join still holds
        // reads what was queued for it, then `Closed`.
        let router = LocalPresenceRouter::new();
        let (mut rx, _) = router.subscribe("s1");
        let event = PresenceEvent::PeerLeft {
            connection_id: "c1".to_string(),
        };
        assert_eq!(router.broadcast("s1", event.clone()), 1);
        assert!(router.unsubscribe("s1"), "the only connection was the last");

        assert_eq!(*rx.recv().now_or_never().unwrap().unwrap(), event);
        assert!(matches!(
            rx.recv().now_or_never(),
            Some(Err(crate::presence::PresenceRecvError::Closed))
        ));
    }

    #[test]
    fn a_receiver_of_a_removed_entry_does_not_see_the_next_entrys_events() {
        // A rejoin after the entry was removed gets a fresh entry and queue;
        // the old receiver is closed and never reads the new queue's events.
        let router = LocalPresenceRouter::new();
        let (mut old_rx, _) = router.subscribe("s1");
        assert!(router.unsubscribe("s1"));
        let (mut new_rx, first) = router.subscribe("s1");
        assert!(first, "precondition: a fresh entry");
        let event = PresenceEvent::PeerLeft {
            connection_id: "c1".to_string(),
        };
        assert_eq!(router.broadcast("s1", event.clone()), 1);

        assert!(matches!(
            old_rx.recv().now_or_never(),
            Some(Err(crate::presence::PresenceRecvError::Closed))
        ));
        assert_eq!(*new_rx.recv().now_or_never().unwrap().unwrap(), event);
    }

    #[test]
    fn fanout_allocated_slots_is_none_without_an_entry_and_zero_for_an_idle_one() {
        // The test-support accessor the memory tests read: no entry, no
        // figure; a session with receivers but no event yet allocates no slot.
        let router = LocalPresenceRouter::new();
        assert_eq!(router.fanout_allocated_slots("s1"), None);
        let (_rx, _) = router.subscribe("s1");
        assert_eq!(router.fanout_allocated_slots("s1"), Some(0));
    }

    #[test]
    fn router_session_ids_returns_all_sessions() {
        let router = LocalPresenceRouter::new();
        let (_rx1, _) = router.subscribe("s1");
        let (_rx2, _) = router.subscribe("s2");
        let mut ids = router.session_ids();
        ids.sort();
        assert_eq!(ids, vec!["s1", "s2"]);
    }

    #[test]
    fn router_has_session_false_for_absent() {
        let router = LocalPresenceRouter::new();
        assert!(!router.has_session("nope"));
    }

    // ── D. Subscribe/unsubscribe lifecycle (b.i2e) ────────────────────

    #[test]
    fn subscribe_first_then_second_only_first_is_first_subscriber() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first1) = router.subscribe("s1");
        let (_rx2, first2) = router.subscribe("s1");
        assert!(
            first1,
            "0 -> 1 transition must return first_subscriber=true"
        );
        assert!(
            !first2,
            "1 -> 2 transition must return first_subscriber=false"
        );
        assert_eq!(router.session_count(), 1);
    }

    #[test]
    fn unsubscribe_with_remaining_subscribers_keeps_entry() {
        let router = LocalPresenceRouter::new();
        let (_rx1, _) = router.subscribe("s1");
        let (_rx2, _) = router.subscribe("s1");

        let last = router.unsubscribe("s1");
        assert!(
            !last,
            "unsubscribe with remaining subscribers must return last_subscriber=false"
        );
        assert!(
            router.has_session("s1"),
            "entry must persist while subscribers remain"
        );
        assert_eq!(router.session_count(), 1);
    }

    #[test]
    fn unsubscribe_last_subscriber_removes_entry() {
        let router = LocalPresenceRouter::new();
        let (_rx, _) = router.subscribe("s1");

        let last = router.unsubscribe("s1");
        assert!(last, "1 -> 0 transition must return last_subscriber=true");
        assert!(
            !router.has_session("s1"),
            "entry must be removed after last unsubscribe"
        );
        assert_eq!(router.session_count(), 0);
    }

    #[test]
    fn subscribe_after_full_unsubscribe_is_first_subscriber_again() {
        let router = LocalPresenceRouter::new();
        let (_rx, first_a) = router.subscribe("s1");
        assert!(first_a);
        let last = router.unsubscribe("s1");
        assert!(last);

        // Regression: reconnect on a session that was previously fully
        // released must create a fresh entry and trigger Valkey re-SUBSCRIBE.
        let (_rx2, first_b) = router.subscribe("s1");
        assert!(
            first_b,
            "resubscribe after full release must return first_subscriber=true"
        );
        assert!(router.has_session("s1"));
    }

    #[test]
    fn unsubscribe_on_absent_session_is_noop_returns_false() {
        let router = LocalPresenceRouter::new();
        let last = router.unsubscribe("no-such");
        assert!(!last);
        assert_eq!(router.session_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_subscribe_unsubscribe_never_leaks_or_orphans() {
        // Stress test: N tasks each do subscribe/unsubscribe loops on the
        // same session_id. At the end, the router state must be consistent:
        // either session absent (count==0) or present with a matching count
        // of live receivers. We can assert the absence/presence invariant
        // by ensuring that after all receivers are dropped and all
        // unsubscribes are awaited, the entry is gone.
        let router = Arc::new(LocalPresenceRouter::new());
        let tasks = 16;
        let iterations = 200;

        let mut handles = Vec::with_capacity(tasks);
        for _ in 0..tasks {
            let r = router.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..iterations {
                    let (rx, _first) = r.subscribe("s1");
                    // Hold the receiver briefly so the session is observed as
                    // populated by other threads, then drop it and unsubscribe.
                    tokio::task::yield_now().await;
                    drop(rx);
                    let _ = r.unsubscribe("s1");
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // Invariant: after every subscribe has a matching unsubscribe, the
        // counter must be at zero and the DashMap entry must have been
        // removed under the shard lock on the final 1 -> 0 transition.
        assert!(
            !router.has_session("s1"),
            "entry must be absent after all subscribe/unsubscribe pairs complete"
        );
        assert_eq!(router.session_count(), 0);
    }

    // ── E. Channel subscription single-flight (join_channel / leave_channel) ──
    //
    // Driven through the production API — `join_channel`, `leave_channel` and
    // the `ChannelSubscribe` / `ChannelUnsubscribe` duties they hand out — not
    // the test-only `subscribe` / `unsubscribe`, which settle every duty at
    // once. No Valkey: a test plays the join task's part by settling the
    // duties itself, and "the Valkey command completed" is the call that
    // settles them.

    /// Bound on awaiting a wait the test has just unblocked; unreachable on
    /// the healthy path.
    const UNBLOCKED: std::time::Duration = std::time::Duration::from_secs(5);

    fn expect_subscribe(join: ChannelJoin, what: &str) -> ChannelSubscribe {
        match join {
            ChannelJoin::Subscribe(subscribe) => subscribe,
            ChannelJoin::Wait(_) => panic!("{what}: expected the subscribe duty, got Wait"),
            ChannelJoin::Subscribed => {
                panic!("{what}: expected the subscribe duty, got Subscribed")
            }
        }
    }

    fn expect_wait(join: ChannelJoin, what: &str) -> ChannelWait {
        match join {
            ChannelJoin::Wait(wait) => wait,
            ChannelJoin::Subscribe(_) => panic!("{what}: expected Wait, got the subscribe duty"),
            ChannelJoin::Subscribed => panic!("{what}: expected Wait, got Subscribed"),
        }
    }

    fn assert_subscribed(join: ChannelJoin, what: &str) {
        match join {
            ChannelJoin::Subscribed => {}
            ChannelJoin::Wait(_) => panic!("{what}: expected Subscribed, got Wait"),
            ChannelJoin::Subscribe(_) => {
                panic!("{what}: expected Subscribed, got the subscribe duty")
            }
        }
    }

    /// A joiner that arrives while the first joiner's `SUBSCRIBE` is in flight
    /// waits, and goes on (`Ok`) once that `SUBSCRIBE` is reported successful;
    /// a joiner after that finds the channel subscribed.
    #[tokio::test]
    async fn join_channel_wait_resolves_ok_once_the_subscribe_succeeds() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        let subscribe = expect_subscribe(first, "first joiner");
        let (_rx2, second) = router.join_channel("s1", None);
        let mut wait = Box::pin(expect_wait(second, "joiner during SUBSCRIBE").wait());
        assert!(
            futures::poll!(wait.as_mut()).is_pending(),
            "the waiter must stay waiting while the SUBSCRIBE is unreported"
        );

        subscribe.succeeded();
        let outcome = tokio::time::timeout(UNBLOCKED, wait)
            .await
            .expect("the waiter must wake once the SUBSCRIBE is reported");
        assert!(
            outcome.is_ok(),
            "a successful SUBSCRIBE lets the waiter go on"
        );
        let (_rx3, third) = router.join_channel("s1", None);
        assert_subscribed(third, "joiner after the SUBSCRIBE succeeded");
        assert_eq!(router.connection_count(), 3);
    }

    /// A failed `SUBSCRIBE` fails every waiter with the subscribing joiner's
    /// reason and leaves the channel unsubscribed: the next joiner gets the
    /// subscribe duty again. The failed joiners still hold their counts until
    /// they leave.
    #[tokio::test]
    async fn join_channel_wait_resolves_err_with_the_reason_when_the_subscribe_fails() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        let subscribe = expect_subscribe(first, "first joiner");
        let (_rx2, second) = router.join_channel("s1", None);
        let (_rx3, third) = router.join_channel("s1", None);
        let waits = [
            expect_wait(second, "second joiner"),
            expect_wait(third, "third joiner"),
        ];

        subscribe.failed("connection reset");
        for wait in waits {
            let outcome = tokio::time::timeout(UNBLOCKED, wait.wait())
                .await
                .expect("a waiter must wake once the SUBSCRIBE is reported");
            let reason = outcome.expect_err("a failed SUBSCRIBE must fail every waiter");
            assert!(
                reason.contains("connection reset"),
                "the waiter must see the subscribing joiner's reason; got {reason:?}"
            );
        }
        assert_eq!(
            router.connection_count(),
            3,
            "the failed joiners hold their counts until they leave"
        );
        let (_rx4, fourth) = router.join_channel("s1", None);
        expect_subscribe(fourth, "the joiner after a failed SUBSCRIBE").succeeded();
    }

    /// A subscribe duty dropped unreported — its join's future dropped
    /// mid-`SUBSCRIBE` — reports a failure: its waiter wakes with an error
    /// instead of waiting forever, and the next joiner subscribes again.
    #[tokio::test]
    async fn dropping_an_unreported_subscribe_duty_fails_its_waiters_and_lets_the_next_join_subscribe(
    ) {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        let subscribe = expect_subscribe(first, "first joiner");
        let (_rx2, second) = router.join_channel("s1", None);
        let wait = expect_wait(second, "joiner during SUBSCRIBE");

        drop(subscribe);
        let outcome = tokio::time::timeout(UNBLOCKED, wait.wait())
            .await
            .expect("the waiter must wake once the duty is dropped");
        assert!(
            outcome.is_err(),
            "a dropped duty must fail its waiter, not let it go on unsubscribed"
        );
        let (_rx3, third) = router.join_channel("s1", None);
        expect_subscribe(third, "the joiner after a dropped duty").succeeded();
    }

    /// A joiner that arrives while the last leaver's `UNSUBSCRIBE` is in flight
    /// gets the subscribe duty, but its wait for that prior `UNSUBSCRIBE` holds
    /// it until the leaver finishes; a joiner arriving meanwhile waits on its
    /// `SUBSCRIBE` in turn.
    #[tokio::test]
    async fn join_during_an_unsubscribe_waits_for_it_before_subscribing() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        expect_subscribe(first, "first joiner").succeeded();
        let unsubscribe = router
            .leave_channel("s1")
            .expect("the last leaver gets the unsubscribe duty");
        assert!(
            router.has_session("s1"),
            "the entry stays while the UNSUBSCRIBE is in flight"
        );

        let (_rx2, second) = router.join_channel("s1", None);
        let mut subscribe = expect_subscribe(second, "joiner during UNSUBSCRIBE");
        let (_rx3, third) = router.join_channel("s1", None);
        let wait = expect_wait(third, "joiner behind the new SUBSCRIBE");
        {
            let mut prior = Box::pin(subscribe.wait_for_prior_unsubscribe());
            assert!(
                futures::poll!(prior.as_mut()).is_pending(),
                "the new joiner must wait while the prior UNSUBSCRIBE is in flight"
            );
            unsubscribe.finish();
            tokio::time::timeout(UNBLOCKED, prior)
                .await
                .expect("the new joiner must go on once the UNSUBSCRIBE finishes");
        }
        subscribe.succeeded();
        tokio::time::timeout(UNBLOCKED, wait.wait())
            .await
            .expect("the waiter must wake once the SUBSCRIBE is reported")
            .expect("the SUBSCRIBE succeeded");
    }

    /// The last leaver racing a new joiner: the leaver's `finish`, arriving
    /// after the joiner took the entry over, leaves the entry in place, and
    /// once the joiner's `SUBSCRIBE` succeeds the session is subscribed with
    /// the joiner counted — later joiners find it subscribed, and events reach
    /// the joiner.
    #[tokio::test]
    async fn last_leaver_racing_a_new_joiner_ends_subscribed_with_the_new_joiner() {
        let router = LocalPresenceRouter::new();
        let (rx1, first) = router.join_channel("s1", None);
        expect_subscribe(first, "first joiner").succeeded();
        // The leaver's stream goes with it.
        drop(rx1);
        let unsubscribe = router
            .leave_channel("s1")
            .expect("the last leaver gets the unsubscribe duty");
        let (mut rx2, second) = router.join_channel("s1", None);
        let mut subscribe = expect_subscribe(second, "joiner during UNSUBSCRIBE");

        unsubscribe.finish();
        assert!(
            router.has_session("s1"),
            "the leaver's finish must not remove an entry a joiner has taken over"
        );
        tokio::time::timeout(UNBLOCKED, subscribe.wait_for_prior_unsubscribe())
            .await
            .expect("the prior UNSUBSCRIBE has finished");
        subscribe.succeeded();

        let (_rx3, third) = router.join_channel("s1", None);
        assert_subscribed(third, "joiner after the race");
        assert_eq!(router.connection_count(), 2);
        let event = PresenceEvent::PeerLeft {
            connection_id: "peer".to_owned(),
        };
        assert_eq!(router.broadcast("s1", event.clone()), 2);
        assert_eq!(
            *rx2.recv()
                .now_or_never()
                .expect("the new joiner receives events")
                .expect("the new joiner receives events"),
            event
        );
    }

    /// A stale `finish` — from an unsubscribe round a newer round has since
    /// replaced — leaves the newer round's entry in place; the newer round's
    /// own `finish` removes it. The newer round arises when the joiner that
    /// took the entry over drops its subscribe duty without first waiting for
    /// the prior `UNSUBSCRIBE`, then leaves: a breach of the documented caller
    /// contract that no `debug_assert!` guards, so the release-safe handling
    /// is reachable here. (A stale subscribe `report` needs a join to leave
    /// with its duty unreported, which panics a debug build instead; see
    /// `leaving_the_last_connection_with_an_unreported_subscribe_duty_panics_in_debug_builds`.)
    #[test]
    fn a_stale_unsubscribe_finish_leaves_a_newer_unsubscribe_round_in_place() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        expect_subscribe(first, "first joiner").succeeded();
        let stale = router
            .leave_channel("s1")
            .expect("the last leaver gets the unsubscribe duty");
        let (_rx2, second) = router.join_channel("s1", None);
        drop(expect_subscribe(second, "joiner during UNSUBSCRIBE"));
        let current = router
            .leave_channel("s1")
            .expect("the joiner, leaving last, gets a new unsubscribe duty");

        stale.finish();
        assert!(
            router.has_session("s1"),
            "a stale finish must not remove the entry while a newer UNSUBSCRIBE is in flight"
        );
        current.finish();
        assert!(
            !router.has_session("s1"),
            "the current round's finish removes the idle entry"
        );
    }

    /// Leaving the session's last connection while its subscribe duty is
    /// unreported breaks `leave_channel`'s caller contract, which a debug build
    /// catches with a `debug_assert!`. Compiled only with debug assertions.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "SUBSCRIBE duty was unreported")]
    fn leaving_the_last_connection_with_an_unreported_subscribe_duty_panics_in_debug_builds() {
        let router = LocalPresenceRouter::new();
        let (_rx, first) = router.join_channel("s1", None);
        let _subscribe = expect_subscribe(first, "first joiner");
        let _ = router.leave_channel("s1");
    }

    // ── F. Index-renewal gate (SDD SR-7.2b) ────────────────────────────
    //
    // `index_renewal_due` decides whether a keeper renewal re-adds its session
    // to the session index: at most once per session per interval, measured
    // in keeper tick instants. The gate reads no clock and every tick passed to
    // it is an explicit offset from one instant, so these tests are exact at
    // the interval boundary.

    /// The refresh interval the gate is asked about.
    const RENEWAL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

    /// A step far below [`RENEWAL_INTERVAL`], for "a moment later".
    const RENEWAL_EPSILON: std::time::Duration = std::time::Duration::from_millis(1);

    /// The first renewal of an interval is due and records its tick; a later
    /// renewal of the same session inside that interval, however close to its
    /// end, is not.
    #[test]
    fn index_renewal_due_first_renewal_records_and_later_ones_in_the_interval_are_not_due() {
        let router = LocalPresenceRouter::new();
        let (_rx1, _) = router.subscribe("s1");
        let (_rx2, _) = router.subscribe("s1");
        let t0 = tokio::time::Instant::now();

        assert!(
            router.index_renewal_due("s1", t0, RENEWAL_INTERVAL),
            "the session's first renewal is due"
        );
        assert!(
            !router.index_renewal_due("s1", t0 + RENEWAL_INTERVAL / 2, RENEWAL_INTERVAL),
            "a sibling's renewal half an interval later is not due"
        );
        assert!(
            !router.index_renewal_due(
                "s1",
                t0 + RENEWAL_INTERVAL - RENEWAL_EPSILON,
                RENEWAL_INTERVAL
            ),
            "a renewal just short of one interval after the record is not due"
        );
    }

    /// A session's only connection ticks exactly one interval apart, and every
    /// one of its renewals is due: the boundary is inclusive, and each due
    /// renewal moves the record to its own tick. A renewal just after one of
    /// them is not due, so the record did move.
    #[test]
    fn index_renewal_due_ticks_exactly_one_interval_apart_are_each_due() {
        let router = LocalPresenceRouter::new();
        let (_rx, _) = router.subscribe("s1");
        let t0 = tokio::time::Instant::now();

        for k in 0..5u32 {
            let tick = t0 + RENEWAL_INTERVAL * k;
            assert!(
                router.index_renewal_due("s1", tick, RENEWAL_INTERVAL),
                "the sole connection's renewal {k}, one interval after the last, must be due"
            );
            assert!(
                !router.index_renewal_due("s1", tick + RENEWAL_EPSILON, RENEWAL_INTERVAL),
                "renewal {k} must have moved the record to its own tick"
            );
        }
    }

    /// Records are per session: one session's renewal does not gate another's.
    #[test]
    fn index_renewal_due_records_are_per_session() {
        let router = LocalPresenceRouter::new();
        let (_rx1, _) = router.subscribe("s1");
        let (_rx2, _) = router.subscribe("s2");
        let t0 = tokio::time::Instant::now();

        assert!(router.index_renewal_due("s1", t0, RENEWAL_INTERVAL));
        assert!(
            router.index_renewal_due("s2", t0, RENEWAL_INTERVAL),
            "s1's record must not gate s2's renewal"
        );
    }

    /// A session with no router entry is always due, and the call records
    /// nothing: the session's first local renewal after it joins is due at the
    /// very same tick.
    #[test]
    fn index_renewal_due_for_a_session_with_no_router_entry_is_due_and_records_nothing() {
        let router = LocalPresenceRouter::new();
        let t0 = tokio::time::Instant::now();

        assert!(
            router.index_renewal_due("s1", t0, RENEWAL_INTERVAL),
            "a session with no router entry is due"
        );
        let (_rx, _) = router.subscribe("s1");
        assert!(
            router.index_renewal_due("s1", t0, RENEWAL_INTERVAL),
            "the call with no router entry must not have recorded its tick"
        );
    }

    /// The last local connection's leave removes the session's record: a
    /// connection that joins afterwards is due inside the interval the old
    /// record covered.
    #[test]
    fn leave_channel_of_the_last_local_connection_removes_the_index_renewal_record() {
        let router = LocalPresenceRouter::new();
        let (rx, _) = router.subscribe("s1");
        let t0 = tokio::time::Instant::now();
        assert!(router.index_renewal_due("s1", t0, RENEWAL_INTERVAL));

        drop(rx);
        assert!(router.unsubscribe("s1"), "precondition: the last leave");
        let (_rx, _) = router.subscribe("s1");
        assert!(
            router.index_renewal_due("s1", t0 + RENEWAL_EPSILON, RENEWAL_INTERVAL),
            "the last leave must have removed the record, so a new connection's renewal is due"
        );
    }

    /// A leave that leaves another local connection behind keeps the record:
    /// the remaining connection's renewal inside the interval is still not due.
    #[test]
    fn leave_channel_of_a_non_last_connection_keeps_the_index_renewal_record() {
        let router = LocalPresenceRouter::new();
        let (_rx1, _) = router.subscribe("s1");
        let (rx2, _) = router.subscribe("s1");
        let t0 = tokio::time::Instant::now();
        assert!(router.index_renewal_due("s1", t0, RENEWAL_INTERVAL));

        drop(rx2);
        assert!(
            !router.unsubscribe("s1"),
            "precondition: not the last leave"
        );
        assert!(
            !router.index_renewal_due("s1", t0 + RENEWAL_EPSILON, RENEWAL_INTERVAL),
            "a non-last leave must keep the record"
        );
    }

    /// The record goes at the last leave itself, not when its `UNSUBSCRIBE`
    /// finishes: while that `UNSUBSCRIBE` is in flight the session has an entry
    /// but no local connection, so a renewal is due and records nothing, and a
    /// connection joining meanwhile starts with no record.
    #[test]
    fn a_joiner_during_the_last_leavers_unsubscribe_starts_with_no_index_renewal_record() {
        let router = LocalPresenceRouter::new();
        let (_rx1, first) = router.join_channel("s1", None);
        expect_subscribe(first, "first joiner").succeeded();
        let t0 = tokio::time::Instant::now();
        assert!(router.index_renewal_due("s1", t0, RENEWAL_INTERVAL));

        let unsubscribe = router
            .leave_channel("s1")
            .expect("the last leaver gets the unsubscribe duty");
        assert!(
            router.has_session("s1"),
            "precondition: the entry stays while the UNSUBSCRIBE is in flight"
        );
        assert!(
            router.index_renewal_due("s1", t0 + RENEWAL_EPSILON, RENEWAL_INTERVAL),
            "with no local connection left, a renewal is due"
        );

        let (_rx2, second) = router.join_channel("s1", None);
        let subscribe = expect_subscribe(second, "joiner during UNSUBSCRIBE");
        assert!(
            router.index_renewal_due("s1", t0 + RENEWAL_EPSILON * 2, RENEWAL_INTERVAL),
            "a joiner during the in-flight UNSUBSCRIBE must start with no record: the leave \
             removed it and the renewal with no connection left recorded nothing"
        );
        unsubscribe.finish();
        subscribe.succeeded();
    }

    // ── G. PUBLISH routing pin ─────────────────────────────────────────

    /// For a session whose member-set slot differs from its channel name's
    /// slot, `presence_publish_pin` routes by the member-set (`{sid}`) slot —
    /// where the membership scripts publish — and not by the channel's slot,
    /// fred's default for an untagged `PUBLISH`.
    #[test]
    fn presence_publish_pin_hashes_to_the_member_set_slot_not_the_channel_slot() {
        let ks = KeySchema::new("le:");
        let slot = |key: String| fred::util::redis_keyslot(key.as_bytes());
        let session_id = (0..)
            .map(|i| format!("pin-session-{i}"))
            .find(|sid| slot(ks.presence_members(sid)) != slot(ks.presence_events_channel(sid)))
            .expect("some session's two slots differ");
        let members_slot = slot(ks.presence_members(&session_id));
        let channel_slot = slot(ks.presence_events_channel(&session_id));

        let pin = presence_publish_pin(&ks, &session_id);
        assert_eq!(pin, ClusterHash::Custom(members_slot));
        assert_ne!(pin, ClusterHash::Custom(channel_slot));
    }
}
