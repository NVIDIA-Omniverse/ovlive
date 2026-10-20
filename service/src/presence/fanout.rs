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

//! Per-session fan-out of presence events to a replica's local JoinSession
//! streams.
//!
//! [`SessionFanout`] is one session's queue, owned by the
//! [`LocalPresenceRouter`](super::LocalPresenceRouter) entry; each local join
//! holds a [`PresenceReceiver`] on it. Every event is wrapped in one `Arc`
//! and queued once, whatever the number of receivers, with a count of the
//! receivers that have still to read it.
//!
//! **Memory follows undelivered events, not the capacity.** An entry leaves
//! the queue as soon as every receiver has read it (or dropped), freeing its
//! shared event. The slot array (16 B a slot) grows with the queued count,
//! doubling, but never past the capacity: a full queue drops its oldest entry
//! before it takes a new one. As the queue drains, the array is given back
//! whenever the queued count falls to a quarter of it or below, down to twice
//! the queued count, and an emptied queue keeps [`RETAINED_SLOTS`]. So an
//! idle session holds no more than that floor, plus one connection id once a
//! join or leave has passed through it (see below), whatever the capacity; a busy
//! one holds one shared event per queued event, and at most four times as
//! many slots as queued events or the floor, whichever is larger, and never
//! more than the capacity. Shrinking only at a quarter keeps a queue that
//! hovers around one size from reallocating on every event. A tokio broadcast
//! channel, the obvious alternative, allocates all `capacity` slots for every
//! session up front and cannot grow, so its lag bound and its idle memory
//! are one setting.
//!
//! Each receiver is a fixed 96 bytes inline (see [`PresenceReceiver`]) plus
//! its connection id on the heap. When it is behind it takes up to
//! [`RECV_BATCH`] events under one lock, returning one and keeping the other
//! `RECV_BATCH - 1`, so it can keep up to that many already-read events
//! alive after the queue has let them go: a bounded
//! per-receiver cost that buys one lock acquisition per batch instead of per
//! event (see [`PresenceReceiver::recv`]).
//!
//! **The capacity is a lag bound, applied exactly.** A send to a queue that
//! already holds `capacity` events removes the oldest first; a receiver whose next
//! event was removed sees [`PresenceRecvError::Lagged`] with the exact count
//! it missed, then resumes at the oldest event still queued.
//!
//! **Membership loss is told apart from message loss.** Entries leave in
//! position order, and an entry a receiver has not read can leave only by
//! overflow (it still counts that receiver otherwise), so a lagging receiver
//! missed a join or leave exactly when some `PeerJoined` or `PeerLeft` at or
//! after its cursor has been removed: `Lagged::membership_lost`. The
//! JoinSession loop ends the stream for a resync on that, and only counts a
//! lag that lost messages alone (SDD §13.2).
//!
//! A receiver's own connection's joins and leaves do not count: its stream
//! discards them at the self-filter, so losing one loses nothing the client
//! would have seen. Without this, a join's own `joined`, which reaches its
//! own queue just after it subscribes, would turn a lag over it into a
//! resync, and a slow client in a busy session could rejoin over and over.
//! So the queue keeps, exactly and in constant space, the position and
//! connection id of the newest removed join or leave, and the position of
//! the newest removed one whose connection id differs from that. For a
//! receiver of connection X the newest removed join or leave not about X is
//! the first if it is not about X, and otherwise the second; the loss counts
//! when that position is at or after the receiver's cursor.
//!
//! **Order.** One queue per session, read in position order by every
//! receiver, so each local stream sees the relay's arrival order (SDD §13.1).
//!
//! **Locking.** The queue's mutex takes no other lock and is never held
//! across an `.await`. The router calls in with its `DashMap` shard lock and,
//! on a join, the entry's state mutex already held, so the order is shard,
//! entry state, fan-out.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::Notify;

use super::session_presence::PresenceEvent;

/// Slots a drained queue keeps allocated rather than give back: enough that
/// a session with steady light traffic does not reallocate on every event,
/// and small next to the per-session state the router already holds.
pub(crate) const RETAINED_SLOTS: usize = 8;

/// The fewest slots a growing queue allocates, as `VecDeque` itself would
/// for a 16-byte element.
const MIN_GROWN_SLOTS: usize = 4;

/// One session's fan-out queue. See the module docs.
pub(crate) struct SessionFanout {
    state: Mutex<FanoutState>,
    /// Woken (`notify_waiters`) after every send and on close. A receiver
    /// creates its `Notified` before it looks for an event, and tokio's
    /// `Notified` counts every `notify_waiters` call from its creation on,
    /// even before it is first polled, so no wakeup falls between the look
    /// and the wait.
    notify: Notify,
    /// Test-only hook run by [`PresenceReceiver::recv`] after a look that
    /// found nothing, with the state lock released and before it awaits its
    /// `Notified`: the window in which a send must still wake it. See
    /// [`SessionFanout::set_after_empty_look`].
    #[cfg(test)]
    after_empty_look: Mutex<Option<AfterEmptyLook>>,
}

/// The hook type of [`SessionFanout::set_after_empty_look`].
#[cfg(test)]
pub(crate) type AfterEmptyLook = Arc<dyn Fn(&SessionFanout) + Send + Sync>;

struct FanoutState {
    /// Events not yet read by every receiver they were sent to, oldest
    /// first. `queue[i]` is at position `front + i`.
    queue: VecDeque<Slot>,
    /// Position of `queue[0]`; the next send is queued at `front + queue.len()`.
    front: u64,
    /// Live [`PresenceReceiver`]s. A send is queued only when this is nonzero,
    /// and counts each of them as a reader of the new entry.
    receivers: usize,
    /// Most entries the queue holds; a send beyond it removes the oldest.
    capacity: usize,
    /// Set by [`SessionFanout::close`]: the router has removed the session's
    /// entry, so nothing more is sent here.
    closed: bool,
    /// The `PeerJoined` and `PeerLeft` entries ever removed from the queue,
    /// read or not, as far as `Lagged::membership_lost` needs them.
    membership_removed: MembershipRemoved,
}

/// The removed joins and leaves a lagging receiver's `membership_lost` is
/// decided from (module docs): enough to find, for any connection X, the
/// newest removed join or leave about a connection other than X.
#[derive(Default)]
struct MembershipRemoved {
    /// The newest removed join or leave: its position and connection id.
    newest: Option<(u64, String)>,
    /// The position of the newest removed join or leave whose connection id
    /// differs from `newest`'s.
    newest_other: Option<u64>,
}

impl MembershipRemoved {
    /// Record the join or leave about `connection_id` removed at `position`,
    /// newer than every one recorded before.
    fn record(&mut self, position: u64, connection_id: String) {
        if let Some((previous, previous_id)) = &self.newest {
            if *previous_id != connection_id {
                // The previous newest is now the newest about a connection
                // other than the new newest's. If it was about the same
                // connection, `newest_other` already is.
                self.newest_other = Some(*previous);
            }
        }
        self.newest = Some((position, connection_id));
    }

    /// The position of the newest removed join or leave about a connection
    /// other than `own` (any connection, for `None`).
    fn newest_not_about(&self, own: Option<&str>) -> Option<u64> {
        match &self.newest {
            Some((_, id)) if own == Some(id.as_str()) => self.newest_other,
            Some((position, _)) => Some(*position),
            None => None,
        }
    }
}

struct Slot {
    event: Arc<PresenceEvent>,
    /// Receivers that have still to read this entry. The entries at zero
    /// always form a prefix of the queue (a receiver counted on an entry is
    /// counted on every later one, and reads them in order), and are removed
    /// at once.
    remaining: usize,
}

impl FanoutState {
    fn next(&self) -> u64 {
        self.front + self.queue.len() as u64
    }

    /// Remove the front entry, recording it on `membership_removed` when it
    /// is a join or a leave. The connection id is moved out of the event only
    /// when nothing else still holds it, and copied otherwise. An entry
    /// removed by its last read is always still held (`recv` clones the event
    /// before it pops). One removed by an overflow in `send` or a receiver's
    /// drop is often not, but may be: another receiver's read-ahead buffer or
    /// a JoinSession loop handling it can still hold it. Joins and leaves are
    /// rare next to messages, so the copy is not worth avoiding.
    fn pop_front(&mut self) {
        let Some(slot) = self.queue.pop_front() else {
            return;
        };
        let position = self.front;
        self.front += 1;
        let connection_id = match Arc::try_unwrap(slot.event) {
            Ok(PresenceEvent::PeerJoined(user)) => user.connection_id,
            Ok(PresenceEvent::PeerLeft { connection_id }) => connection_id,
            Ok(PresenceEvent::PeerMessage { .. }) => return,
            Err(shared) => match &*shared {
                PresenceEvent::PeerJoined(user) => user.connection_id.clone(),
                PresenceEvent::PeerLeft { connection_id } => connection_id.clone(),
                PresenceEvent::PeerMessage { .. } => return,
            },
        };
        self.membership_removed.record(position, connection_id);
    }

    /// Remove the entries every receiver has read, then give slots back once
    /// the queued count has fallen to a quarter of the array or below: down
    /// to twice the queued count, never below [`RETAINED_SLOTS`] (an emptied
    /// queue keeps exactly that floor, or what it had if less).
    fn pop_read(&mut self) {
        while self.queue.front().is_some_and(|slot| slot.remaining == 0) {
            self.pop_front();
        }
        let len = self.queue.len();
        if len <= self.queue.capacity() / 4 {
            self.queue.shrink_to((2 * len).max(RETAINED_SLOTS));
        }
    }

    /// Make room for one more entry when the slot array is full: double it,
    /// as `VecDeque` would, but never past the capacity. Called only with
    /// fewer than `capacity` entries queued.
    fn reserve_one(&mut self) {
        let len = self.queue.len();
        if len == self.queue.capacity() {
            let target = (2 * len).max(MIN_GROWN_SLOTS).min(self.capacity);
            self.queue.reserve_exact(target - len);
        }
    }
}

impl SessionFanout {
    /// An empty queue holding at most `capacity` events. Allocates no slots.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(FanoutState {
                queue: VecDeque::new(),
                front: 0,
                receivers: 0,
                capacity,
                closed: false,
                membership_removed: MembershipRemoved::default(),
            }),
            notify: Notify::new(),
            #[cfg(test)]
            after_empty_look: Mutex::new(None),
        }
    }

    /// Lock the queue. No critical section awaits or calls out, so the only
    /// panic one can raise is a broken count invariant or a collection
    /// capacity overflow; a poisoned lock is recovered rather than propagated, as
    /// `SessionEntry::lock` does, so that one panicking stream does not panic
    /// every other stream of the session too.
    fn lock(&self) -> MutexGuard<'_, FanoutState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A receiver of every event sent from this call on, for the local
    /// connection `own`, whose own joins and leaves its `membership_lost`
    /// leaves out (module docs). `None` for a receiver with no connection of
    /// its own, for which every lost join or leave counts.
    pub(crate) fn subscribe(self: &Arc<Self>, own: Option<&str>) -> PresenceReceiver {
        let mut state = self.lock();
        state.receivers += 1;
        let cursor = state.next();
        drop(state);
        PresenceReceiver {
            fanout: Arc::clone(self),
            cursor,
            own: own.map(Box::from),
            batch: [const { None }; RECV_BATCH - 1],
            next: 0,
            len: 0,
        }
    }

    /// Queue `event` for every live receiver and return how many there are.
    /// With none it is dropped and 0 returned, as a broadcast with no
    /// receivers would. A full queue removes its oldest entry first, so the
    /// slot array never has to grow past the capacity to take the new one.
    pub(crate) fn send(&self, event: PresenceEvent) -> usize {
        let receivers = {
            let mut state = self.lock();
            if state.receivers == 0 {
                return 0;
            }
            let receivers = state.receivers;
            if state.queue.len() == state.capacity {
                state.pop_front();
            }
            state.reserve_one();
            state.queue.push_back(Slot {
                event: Arc::new(event),
                remaining: receivers,
            });
            receivers
        };
        self.notify.notify_waiters();
        receivers
    }

    /// **Test-only.** Run `hook` with this queue in every
    /// [`PresenceReceiver::recv`] after a look that found no event, with the
    /// state lock released and before the receiver awaits its wakeup, so a
    /// test can `send` exactly inside that window; `None` removes it. The
    /// hook is cloned out of its own mutex before it runs, so it may call
    /// any `&self` method of the queue (`send`, `close`, this setter)
    /// without deadlocking. It runs on every empty look, so a test that
    /// should act once keeps its own count.
    #[cfg(test)]
    pub(crate) fn set_after_empty_look(&self, hook: Option<AfterEmptyLook>) {
        *self
            .after_empty_look
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = hook;
    }

    /// Run the [`Self::set_after_empty_look`] hook, if one is set.
    #[cfg(test)]
    fn run_after_empty_look(&self) {
        let hook = self
            .after_empty_look
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(hook) = hook {
            hook(self);
        }
    }

    /// The router has removed this session's entry: a receiver still held
    /// reads what is queued for it, then [`PresenceRecvError::Closed`].
    pub(crate) fn close(&self) {
        self.lock().closed = true;
        self.notify.notify_waiters();
    }

    /// The slots the queue has allocated. Test-only.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn allocated_slots(&self) -> usize {
        self.lock().queue.capacity()
    }
}

/// Why [`PresenceReceiver::recv`] returned no event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PresenceRecvError {
    /// The receiver fell more than the router's capacity behind, and the
    /// `dropped` oldest events it had not read were removed. It resumes at
    /// the oldest event still queued. `membership_lost` is true exactly when
    /// a `PeerJoined` or `PeerLeft` about a connection other than the
    /// receiver's own was among them; false when they were all
    /// `PeerMessage`s or the receiver's own joins and leaves, which its
    /// stream discards anyway.
    #[error(
        "presence receiver lagged: {dropped} events dropped (membership lost: {membership_lost})"
    )]
    Lagged { dropped: u64, membership_lost: bool },
    /// The session's router entry was removed, and every event queued for
    /// this receiver has been read.
    #[error("presence fan-out closed")]
    Closed,
}

/// One local JoinSession stream's place in its session's fan-out queue,
/// handed out by [`LocalPresenceRouter::join_channel`](super::LocalPresenceRouter::join_channel).
///
/// Dropping it releases every entry it had not read; an entry no other
/// receiver still has to read leaves the queue then.
///
/// 96 bytes on a 64-bit target, all inline: the queue handle, the cursor,
/// the connection id's box, and the `RECV_BATCH - 1`-event read-ahead
/// buffer.
/// The connection id is the only heap allocation a receiver makes.
pub struct PresenceReceiver {
    fanout: Arc<SessionFanout>,
    /// Position of the next event this receiver takes from the queue: one
    /// past the last event it took, buffered or returned.
    cursor: u64,
    /// The local connection this receiver serves, whose own joins and
    /// leaves do not count as membership loss; `None` for none.
    own: Option<Box<str>>,
    /// Events taken from the queue but not yet returned, in order, at
    /// `batch[next..len]`; every slot outside that range is `None`. They
    /// precede `cursor`, count as read in the queue (their `remaining` was
    /// decremented when they were taken), and so can never be lost to a lag.
    batch: [Option<Arc<PresenceEvent>>; RECV_BATCH - 1],
    next: u8,
    len: u8,
}

// The size the `PresenceReceiver` rustdoc states, checked at compile time so
// a new field cannot grow every JoinSession connection's receiver unnoticed.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<PresenceReceiver>() == 96);

/// Most events one [`PresenceReceiver::recv`] takes from the queue under a
/// single lock: the one it returns and up to `RECV_BATCH - 1` more, kept in
/// the receiver for its next calls, which then take no lock. A receiver that
/// keeps up finds one event at a time and gains nothing; one that is behind
/// locks the session's queue once per this many events instead of once per
/// event, which is what keeps hundreds of local receivers of a saturated
/// session from contending on that one mutex for every event. 8 cuts those
/// acquisitions eightfold and grows each receiver from 32 to 96 bytes, 56 of
/// them the 7-slot buffer (one pointer a slot); going higher would grow
/// every JoinSession connection's receiver for a smaller further gain, and a
/// larger batch also keeps more read events alive in the receiver after the
/// queue has released them.
///
/// Exported to integration tests under `test-support` (see `presence`).
pub const RECV_BATCH: usize = 8;

impl PresenceReceiver {
    /// The next event for this receiver, waiting for one if none is queued.
    ///
    /// Events already taken into the receiver's read-ahead buffer come first,
    /// without a lock. Otherwise one lock acquisition takes the event at the
    /// cursor and up to `RECV_BATCH - 1` after it, returns the first and
    /// buffers the rest, so a receiver that is behind locks the queue once
    /// per batch rather than once per event. It never waits to fill a batch:
    /// it takes only the events already queued at that look, so a single
    /// event arriving on its own is returned at once, with no added latency.
    /// A lag is reported only for events at or after the cursor, never for
    /// buffered ones, and a closed queue reports [`PresenceRecvError::Closed`]
    /// only once the buffer and the queue are both drained.
    ///
    /// Cancel-safe: the receiver's state, its buffer included, changes only
    /// inside the call that returns, so a future dropped while it waits loses
    /// nothing, and the next call picks up where it would have.
    pub async fn recv(&mut self) -> Result<Arc<PresenceEvent>, PresenceRecvError> {
        if self.next < self.len {
            let event = self.batch[usize::from(self.next)]
                .take()
                .expect("batch[next..len] holds events");
            self.next += 1;
            return Ok(event);
        }
        let Self {
            fanout,
            cursor,
            own,
            batch,
            next,
            len,
        } = self;
        loop {
            // Created before the look below, so a `notify_waiters` (a send
            // or a close) between the look and the `.await` still wakes it:
            // tokio guarantees a `Notified` receives `notify_waiters`
            // wakeups from its creation on, even before it is first polled.
            // Created after the look, a send landing in that gap would be
            // missed and the receiver would sleep with an event queued. The
            // test-only `after_empty_look` seam lets a test send inside that
            // gap.
            let notified = fanout.notify.notified();
            {
                let mut state = fanout.lock();
                if *cursor < state.front {
                    let dropped = state.front - *cursor;
                    let membership_lost = state
                        .membership_removed
                        .newest_not_about(own.as_deref())
                        .is_some_and(|position| position >= *cursor);
                    *cursor = state.front;
                    return Err(PresenceRecvError::Lagged {
                        dropped,
                        membership_lost,
                    });
                }
                let index = (*cursor - state.front) as usize;
                if let Some(slot) = state.queue.get_mut(index) {
                    let event = Arc::clone(&slot.event);
                    slot.remaining -= 1;
                    // Read ahead under the same lock, into the (empty)
                    // buffer: each taken event counts as read in the queue
                    // now, exactly as if it had been returned.
                    let mut taken = 0;
                    for (buffered, slot) in
                        batch.iter_mut().zip(state.queue.iter_mut().skip(index + 1))
                    {
                        *buffered = Some(Arc::clone(&slot.event));
                        slot.remaining -= 1;
                        taken += 1;
                    }
                    *cursor += 1 + taken as u64;
                    *next = 0;
                    *len = taken;
                    state.pop_read();
                    return Ok(event);
                }
                if state.closed {
                    return Err(PresenceRecvError::Closed);
                }
            }
            #[cfg(test)]
            fanout.run_after_empty_look();
            notified.await;
        }
    }
}

impl Drop for PresenceReceiver {
    fn drop(&mut self) {
        let mut state = self.fanout.lock();
        let start = self.cursor.saturating_sub(state.front) as usize;
        for slot in state.queue.iter_mut().skip(start) {
            slot.remaining -= 1;
        }
        state.receivers -= 1;
        state.pop_read();
    }
}

#[cfg(test)]
mod tests {
    //! The fan-out queue on its own: exact lag, the membership-loss record,
    //! release on read and on drop, memory given back, order, cancel safety,
    //! wakeups and close. No I/O; the router-level tests are in
    //! `session_presence`.

    use std::future::Future;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    use futures::task::{waker, ArcWake};
    use futures::FutureExt;

    use super::*;
    use crate::presence::ConnectedUser;

    fn message(seq: u64) -> PresenceEvent {
        PresenceEvent::PeerMessage {
            connection_id: "peer".to_owned(),
            data: seq.to_be_bytes().to_vec(),
        }
    }

    fn joined(cid: &str) -> PresenceEvent {
        PresenceEvent::PeerJoined(ConnectedUser {
            connection_id: cid.to_owned(),
            user_name: "User".to_owned(),
            user_application: "app".to_owned(),
        })
    }

    fn left(cid: &str) -> PresenceEvent {
        PresenceEvent::PeerLeft {
            connection_id: cid.to_owned(),
        }
    }

    fn lagged(dropped: u64, membership_lost: bool) -> PresenceRecvError {
        PresenceRecvError::Lagged {
            dropped,
            membership_lost,
        }
    }

    fn fanout(capacity: usize) -> Arc<SessionFanout> {
        Arc::new(SessionFanout::new(capacity))
    }

    /// One poll of `recv`: `None` when nothing is queued for `rx`.
    fn poll_once(rx: &mut PresenceReceiver) -> Option<Result<PresenceEvent, PresenceRecvError>> {
        rx.recv()
            .now_or_never()
            .map(|result| result.map(|event| (*event).clone()))
    }

    /// Everything `rx` can read now, lags included, up to the first poll that
    /// would wait or a `Closed`.
    fn read_now(rx: &mut PresenceReceiver) -> Vec<Result<PresenceEvent, PresenceRecvError>> {
        let mut out = Vec::new();
        while let Some(result) = poll_once(rx) {
            let closed = result == Err(PresenceRecvError::Closed);
            out.push(result);
            if closed {
                break;
            }
        }
        out
    }

    /// A waker that counts its wakes.
    #[derive(Default)]
    struct CountingWaker(AtomicUsize);

    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ── The capacity is a lag bound, applied exactly ─────────────────────────

    /// Capacity 5, six sends to an idle receiver: exactly one is lost, then
    /// the newest five arrive in order. A queue that rounded the capacity up
    /// (as a tokio broadcast channel does, to 8) would lose nothing here.
    #[test]
    fn recv_past_a_capacity_of_five_reports_exactly_one_dropped_then_the_newest_five() {
        let fanout = fanout(5);
        let mut rx = fanout.subscribe(None);
        for seq in 0..6 {
            assert_eq!(fanout.send(message(seq)), 1);
        }

        let expected: Vec<_> = std::iter::once(Err(lagged(1, false)))
            .chain((1..6).map(|seq| Ok(message(seq))))
            .collect();
        assert_eq!(
            read_now(&mut rx),
            expected,
            "one event past an exact capacity of 5 is lost, and the newest 5 follow in order"
        );
    }

    // ── Membership loss ──────────────────────────────────────────────────────

    /// A lag that lost only messages is no membership loss, even with a join
    /// queued right behind the lost ones (it survived).
    #[test]
    fn lag_that_lost_only_messages_reports_no_membership_loss() {
        let fanout = fanout(3);
        let mut rx = fanout.subscribe(None);
        fanout.send(message(0));
        fanout.send(message(1));
        fanout.send(joined("x"));
        fanout.send(message(2));

        assert_eq!(
            read_now(&mut rx),
            vec![
                Err(lagged(1, false)),
                Ok(message(1)),
                Ok(joined("x")),
                Ok(message(2)),
            ],
            "only message 0 was lost, so the lag is message-only and the join is delivered"
        );
    }

    /// One `PeerJoined` or one `PeerLeft` among the lost events is a
    /// membership loss.
    #[test]
    fn lag_that_lost_a_join_or_a_leave_reports_membership_loss() {
        for (kind, membership) in [("PeerJoined", joined("x")), ("PeerLeft", left("x"))] {
            let fanout = fanout(2);
            let mut rx = fanout.subscribe(None);
            fanout.send(message(0));
            fanout.send(membership);
            for seq in 1..4 {
                fanout.send(message(seq));
            }

            assert_eq!(
                read_now(&mut rx),
                vec![Err(lagged(3, true)), Ok(message(2)), Ok(message(3))],
                "a lost {kind} among three lost events is a membership loss"
            );
        }
    }

    /// Membership loss is judged per receiver: a join that A read before
    /// overflow removed it is no loss to A, which then lags only on a
    /// message, but is a loss to B, which had not read it.
    #[test]
    fn a_join_one_receiver_read_before_overflow_removed_it_is_lost_only_to_the_other() {
        let fanout = fanout(3);
        let mut a = fanout.subscribe(None);
        let mut b = fanout.subscribe(None);
        fanout.send(joined("x"));
        assert_eq!(poll_once(&mut a), Some(Ok(joined("x"))));
        // Positions 1..=4: the third and fourth overflow the join (position
        // 0, which B still counts on) and then message 1.
        for seq in 1..5 {
            fanout.send(message(seq));
        }

        assert_eq!(
            poll_once(&mut a),
            Some(Err(lagged(1, false))),
            "A read the join, so it lost only message 1"
        );
        assert_eq!(
            poll_once(&mut b),
            Some(Err(lagged(2, true))),
            "B never read the join, so it lost it with message 1"
        );
    }

    /// A join every receiver read leaves the queue at once; a later lag of
    /// messages only is not reported as losing it.
    #[test]
    fn a_join_every_receiver_read_is_not_reported_by_a_later_message_only_lag() {
        let fanout = fanout(2);
        let mut a = fanout.subscribe(None);
        let mut b = fanout.subscribe(None);
        fanout.send(joined("x"));
        assert_eq!(poll_once(&mut a), Some(Ok(joined("x"))));
        assert_eq!(poll_once(&mut b), Some(Ok(joined("x"))));
        for seq in 0..4 {
            fanout.send(message(seq));
        }

        assert_eq!(
            poll_once(&mut b),
            Some(Err(lagged(2, false))),
            "B lost messages 0 and 1 only; the join it read is no loss"
        );
    }

    // ── A receiver's own joins and leaves are no membership loss ─────────────

    /// Each receiver's first poll after `events` were sent and then pushed
    /// out of the queue, for receivers given as `(own, read_first)`: the
    /// receiver's own connection, and how many of `events` it read before
    /// they were pushed out, which is its cursor. A read takes every event
    /// queued at its cursor, up to [`RECV_BATCH`], so reading after all were
    /// sent would move the cursor past `read_first`. The events are sent one
    /// at a time instead, and a receiver still within its `read_first` reads
    /// each as it arrives, finding exactly that one queued and buffering
    /// nothing (self-checked). The queue holds every event some receiver
    /// has not yet read until all are sent (one every receiver has read
    /// leaves at once, and is recorded on the membership-loss record the same
    /// way); a message per event then pushes the rest out.
    fn lag_per_receiver(
        events: &[PresenceEvent],
        receivers: &[(Option<&str>, usize)],
    ) -> Vec<Option<Result<PresenceEvent, PresenceRecvError>>> {
        let fanout = fanout(events.len());
        let mut rxs: Vec<_> = receivers
            .iter()
            .map(|(own, _)| fanout.subscribe(*own))
            .collect();
        for (i, event) in events.iter().enumerate() {
            fanout.send(event.clone());
            for (rx, (_, read_first)) in rxs.iter_mut().zip(receivers) {
                if i < *read_first {
                    assert_eq!(
                        poll_once(rx),
                        Some(Ok(event.clone())),
                        "fixture self-check: the receiver reads event {i}"
                    );
                    assert_eq!(
                        poll_once(rx),
                        None,
                        "fixture self-check: nothing buffered past event {i}"
                    );
                }
            }
        }
        for seq in 0..events.len() as u64 {
            fanout.send(message(seq));
        }
        rxs.iter_mut().map(poll_once).collect()
    }

    /// The regression gate: a join's own `joined` reaches its own queue just
    /// after it subscribes, so a first read that lags lands on it. Losing it
    /// (and messages) is no membership loss for that receiver, which would
    /// discard it at the self-filter anyway. A receiver of no connection, on
    /// the same queue, still counts it.
    #[test]
    fn a_lag_over_only_its_own_join_is_no_membership_loss() {
        let fanout = fanout(2);
        let mut mine = fanout.subscribe(Some("me"));
        let mut anyone = fanout.subscribe(None);
        fanout.send(joined("me"));
        fanout.send(message(0));
        fanout.send(message(1));

        assert_eq!(
            read_now(&mut mine),
            vec![Err(lagged(1, false)), Ok(message(0)), Ok(message(1))],
            "the receiver lost only its own join, which is no membership loss to it"
        );
        assert_eq!(
            poll_once(&mut anyone),
            Some(Err(lagged(1, true))),
            "a receiver of no connection counts the same lost join"
        );
    }

    /// The same for its own leave after its own join: two own events lost,
    /// still no membership loss.
    #[test]
    fn a_lag_over_its_own_join_and_leave_is_no_membership_loss() {
        assert_eq!(
            lag_per_receiver(&[joined("me"), left("me")], &[(Some("me"), 0), (None, 0)]),
            vec![Some(Err(lagged(2, false))), Some(Err(lagged(2, true)))],
            "own join and leave lost: none for its own receiver, a loss for one of no connection"
        );
    }

    /// A peer's join or leave lost by a receiver of some other connection is
    /// still a membership loss.
    #[test]
    fn a_lag_over_a_peers_join_or_leave_is_a_membership_loss_for_another_connection() {
        for (kind, membership) in [("PeerJoined", joined("peer")), ("PeerLeft", left("peer"))] {
            assert_eq!(
                lag_per_receiver(&[membership], &[(Some("me"), 0)]),
                vec![Some(Err(lagged(1, true)))],
                "a lost {kind} of a peer is a membership loss to `me`"
            );
        }
    }

    /// The newest removed join or leave is the receiver's own, and the newest
    /// one about anyone else sits just before or just at its cursor. Peer A
    /// joins (position 0), then `me` joins (1), and all are pushed out. A
    /// receiver of `me` that had read A's join (cursor 1) lost only its own:
    /// no loss. One that had not (cursor 0) lost A's join too: a loss. So the
    /// position of the newest peer event, not of the newest event, is what is
    /// compared with the cursor.
    #[test]
    fn its_own_join_newest_and_a_peers_older_join_count_by_where_the_cursor_is() {
        assert_eq!(
            lag_per_receiver(
                &[joined("a"), joined("me")],
                &[(Some("me"), 1), (Some("me"), 0), (None, 1)],
            ),
            vec![
                Some(Err(lagged(1, false))),
                Some(Err(lagged(2, true))),
                Some(Err(lagged(1, true))),
            ],
            "cursor past A's join: own join only; cursor at it: A's join lost too; no \
             connection: own join counts"
        );
    }

    /// Interleaved joins and leaves of two connections, A and B: A joins (0),
    /// B joins (1), A leaves (2), B leaves (3), A rejoins (4), then all are
    /// pushed out. The newest removed is A's (4); the newest about anyone
    /// other than A is B's leave (3). Receivers of A, of B and of no
    /// connection, each having read the first three events (cursor 3) or the
    /// first four (cursor 4):
    ///
    /// | receiver | cursor 3 (lost 3, 4)   | cursor 4 (lost 4)      |
    /// |----------|------------------------|------------------------|
    /// | A        | B's leave lost: true   | own rejoin only: false |
    /// | B        | A's rejoin lost: true  | A's rejoin lost: true  |
    /// | none     | true                   | true                   |
    #[test]
    fn interleaved_joins_and_leaves_count_for_each_receiver_but_its_own() {
        let events = [joined("a"), joined("b"), left("a"), left("b"), joined("a")];
        let receivers = [
            (Some("a"), 3),
            (Some("a"), 4),
            (Some("b"), 3),
            (Some("b"), 4),
            (None, 3),
            (None, 4),
        ];
        assert_eq!(
            lag_per_receiver(&events, &receivers),
            vec![
                Some(Err(lagged(2, true))),
                Some(Err(lagged(1, false))),
                Some(Err(lagged(2, true))),
                Some(Err(lagged(1, true))),
                Some(Err(lagged(2, true))),
                Some(Err(lagged(1, true))),
            ],
            "per (own, cursor): {receivers:?}"
        );
    }

    /// A connection's lease keeper re-registers it after a sweep evicted it:
    /// the sweeper's `left` for `me` (position 1) and the keeper's `joined`
    /// for `me` (2) follow a peer A's join (0), and all are pushed out. The
    /// newest removed is `me`'s rejoin, and two in a row about `me` must keep
    /// A's join as the newest about anyone else. A receiver of `me` that had
    /// not read A's join (cursor 0) lost it; one that had (cursor 1) lost only
    /// its own leave and rejoin; one of no connection lost them.
    #[test]
    fn its_own_leave_then_rejoin_keep_the_older_peer_join_as_the_newest_other() {
        assert_eq!(
            lag_per_receiver(
                &[joined("a"), left("me"), joined("me")],
                &[(Some("me"), 0), (Some("me"), 1), (None, 1)],
            ),
            vec![
                Some(Err(lagged(3, true))),
                Some(Err(lagged(2, false))),
                Some(Err(lagged(2, true))),
            ],
            "cursor 0: A's join lost; cursor 1: own leave and rejoin only; no connection: lost"
        );
    }

    /// The pattern a busy session gives a joiner: a peer, its own, another
    /// peer, its own again — A joins (0), `me` joins (1), B joins (2), `me`
    /// leaves (3). The newest removed is `me`'s leave; the newest about
    /// anyone else is B's join (2). A receiver of `me` past B's join (cursor
    /// 3) lost only its own leave; at B's join (cursor 2) it lost B's join.
    /// Receivers of A and of no connection lost `me`'s leave either way.
    #[test]
    fn peer_own_peer_own_counts_the_peers_by_where_the_cursor_is() {
        let events = [joined("a"), joined("me"), joined("b"), left("me")];
        let receivers = [(Some("me"), 3), (Some("me"), 2), (Some("a"), 3), (None, 3)];
        assert_eq!(
            lag_per_receiver(&events, &receivers),
            vec![
                Some(Err(lagged(1, false))),
                Some(Err(lagged(2, true))),
                Some(Err(lagged(1, true))),
                Some(Err(lagged(1, true))),
            ],
            "per (own, cursor): {receivers:?}"
        );
    }

    // ── Release on read and on drop ─────────────────────────────────────────

    /// An event leaves the queue once every receiver has read it or dropped:
    /// dropping a receiver with unread events releases them, so the queue's
    /// share of the shared event is freed.
    #[test]
    fn dropping_a_receiver_with_unread_events_releases_them() {
        let fanout = fanout(16);
        let mut a = fanout.subscribe(None);
        let b = fanout.subscribe(None);
        for seq in 0..3 {
            assert_eq!(fanout.send(message(seq)), 2);
        }
        let read: Vec<Arc<PresenceEvent>> = (0..3)
            .map(|_| a.recv().now_or_never().expect("queued").expect("an event"))
            .collect();
        assert!(
            read.iter().all(|event| Arc::strong_count(event) == 2),
            "B has not read them, so the queue still holds each event"
        );

        drop(b);

        assert!(
            read.iter().all(|event| Arc::strong_count(event) == 1),
            "dropping B released every event it had not read"
        );
        assert_eq!(
            fanout.send(message(3)),
            1,
            "only A is counted on a new send"
        );
    }

    /// The same for a receiver that had lagged (its position is behind the
    /// queue's front when it drops).
    #[test]
    fn dropping_a_lagged_receiver_releases_the_events_still_queued_for_it() {
        let fanout = fanout(2);
        let mut a = fanout.subscribe(None);
        let b = fanout.subscribe(None);
        let mut last = None;
        for seq in 0..4 {
            fanout.send(message(seq));
            last = Some(a.recv().now_or_never().expect("queued").expect("an event"));
        }
        let last = last.expect("four events were read");
        assert_eq!(
            Arc::strong_count(&last),
            2,
            "B, two events behind the front, still holds the newest"
        );

        drop(b);

        assert_eq!(Arc::strong_count(&last), 1, "dropping lagged B released it");
    }

    /// With no receiver a send is dropped and queues nothing, and a receiver
    /// sees only what is sent after it subscribed.
    #[test]
    fn send_with_no_receiver_queues_nothing_and_a_later_receiver_does_not_see_it() {
        let fanout = fanout(16);
        assert_eq!(fanout.send(message(0)), 0);
        assert_eq!(fanout.allocated_slots(), 0, "nothing was queued");

        let mut rx = fanout.subscribe(None);
        assert_eq!(
            poll_once(&mut rx),
            None,
            "the earlier send is not delivered"
        );
        fanout.send(message(1));
        assert_eq!(poll_once(&mut rx), Some(Ok(message(1))));
    }

    // ── Memory follows undelivered events ────────────────────────────────────

    /// A new queue allocates no slot, and once every receiver has read a
    /// burst the slot array is back at the retained floor.
    #[test]
    fn allocation_returns_to_the_retained_floor_once_every_receiver_has_read() {
        const BURST: u64 = 1000;
        let fanout = fanout(4096);
        let mut a = fanout.subscribe(None);
        let mut b = fanout.subscribe(None);
        assert_eq!(
            fanout.allocated_slots(),
            0,
            "an idle queue allocates nothing"
        );
        for seq in 0..BURST {
            fanout.send(message(seq));
        }
        assert!(
            fanout.allocated_slots() >= BURST as usize,
            "precondition: the queue grew to hold the burst ({} slots)",
            fanout.allocated_slots()
        );

        assert_eq!(read_now(&mut a).len(), BURST as usize);
        assert_eq!(read_now(&mut b).len(), BURST as usize);

        assert!(
            fanout.allocated_slots() <= RETAINED_SLOTS,
            "a drained queue keeps at most {RETAINED_SLOTS} slots, not {}",
            fanout.allocated_slots()
        );
    }

    /// A receiver that falls ever further behind holds the queue at its
    /// capacity: the slot array never grows past it. (A queue that pushed
    /// before it popped would push onto a full 100-slot array, which
    /// `VecDeque` doubles to 200 itself, past the capacity cap `reserve_one`
    /// applies.)
    #[test]
    fn an_overflowing_lag_never_allocates_more_slots_than_the_capacity() {
        const CAPACITY: usize = 100;
        let fanout = fanout(CAPACITY);
        let _never_reads = fanout.subscribe(None);
        for seq in 0..10 * CAPACITY as u64 {
            fanout.send(message(seq));
        }

        assert!(
            fanout.allocated_slots() <= CAPACITY,
            "the slot array must stay within the capacity of {CAPACITY}, got {}",
            fanout.allocated_slots()
        );
    }

    /// Draining part of a full queue gives slots back: at most four per
    /// queued event (the array shrinks once the queued count falls to a
    /// quarter of it), and the full drain leaves the floor.
    #[test]
    fn a_partial_drain_gives_slots_back() {
        const CAPACITY: usize = 4096;
        const LEFT_QUEUED: usize = 512;
        let fanout = fanout(CAPACITY);
        let mut rx = fanout.subscribe(None);
        for seq in 0..CAPACITY as u64 {
            fanout.send(message(seq));
        }
        assert_eq!(
            fanout.allocated_slots(),
            CAPACITY,
            "precondition: a full queue holds the capacity"
        );

        for _ in 0..CAPACITY - LEFT_QUEUED {
            assert!(matches!(poll_once(&mut rx), Some(Ok(_))));
        }
        let partial = fanout.allocated_slots();
        assert!(
            (LEFT_QUEUED..=4 * LEFT_QUEUED).contains(&partial),
            "with {LEFT_QUEUED} queued the array holds between {LEFT_QUEUED} and {} slots, got \
             {partial}",
            4 * LEFT_QUEUED
        );

        assert_eq!(read_now(&mut rx).len(), LEFT_QUEUED);
        assert!(fanout.allocated_slots() <= RETAINED_SLOTS);
    }

    // ── Order ────────────────────────────────────────────────────────────────

    /// Every receiver reads the same sequence, in the order the sends took
    /// the queue, even with several threads sending at once.
    #[test]
    fn every_receiver_reads_the_same_order_under_concurrent_senders() {
        const SENDERS: u64 = 4;
        const PER_SENDER: u64 = 250;
        let fanout = fanout((SENDERS * PER_SENDER) as usize);
        let mut receivers: Vec<_> = (0..3).map(|_| fanout.subscribe(None)).collect();

        std::thread::scope(|scope| {
            for sender in 0..SENDERS {
                let fanout = &fanout;
                scope.spawn(move || {
                    for i in 0..PER_SENDER {
                        fanout.send(message(sender * PER_SENDER + i));
                    }
                });
            }
        });

        let sequences: Vec<_> = receivers.iter_mut().map(read_now).collect();
        assert_eq!(sequences[0].len(), (SENDERS * PER_SENDER) as usize);
        assert!(
            sequences.iter().all(|seq| *seq == sequences[0]),
            "every receiver reads the same order"
        );
        let mut last_per_sender = vec![None; SENDERS as usize];
        for result in &sequences[0] {
            let Ok(PresenceEvent::PeerMessage { data, .. }) = result else {
                panic!("expected only messages, got {result:?}");
            };
            let seq = u64::from_be_bytes(data.as_slice().try_into().expect("8 bytes"));
            let slot = &mut last_per_sender[(seq / PER_SENDER) as usize];
            assert!(
                slot.is_none_or(|last| last < seq),
                "one sender's events keep its order"
            );
            *slot = Some(seq);
        }
    }

    // ── Read-ahead batches ───────────────────────────────────────────────────

    /// A receiver more than a batch behind reads every event, in order,
    /// across several batches.
    #[test]
    fn a_receiver_behind_by_more_than_a_batch_reads_every_event_in_order() {
        const SENT: u64 = 3 * RECV_BATCH as u64 + 3;
        let fanout = fanout(64);
        let mut rx = fanout.subscribe(None);
        for seq in 0..SENT {
            fanout.send(message(seq));
        }

        assert_eq!(
            read_now(&mut rx),
            (0..SENT).map(|seq| Ok(message(seq))).collect::<Vec<_>>()
        );
    }

    /// A receiver that lagged and is then more than a batch behind accounts
    /// for every event sent: the lag's `dropped` plus the events read, which
    /// arrive in order across batches.
    #[test]
    fn a_lag_then_several_batches_account_for_every_event_sent() {
        const CAPACITY: usize = 2 * RECV_BATCH + 3;
        const SENT: u64 = 3 * CAPACITY as u64;
        let fanout = fanout(CAPACITY);
        let mut rx = fanout.subscribe(None);
        for seq in 0..SENT {
            fanout.send(message(seq));
        }

        let lost = SENT - CAPACITY as u64;
        let expected: Vec<_> = std::iter::once(Err(lagged(lost, false)))
            .chain((lost..SENT).map(|seq| Ok(message(seq))))
            .collect();
        assert_eq!(read_now(&mut rx), expected);
    }

    /// A read never waits to fill a batch: a lone queued event is returned
    /// at once, with nothing left buffered, and events sent one at a time are
    /// each returned as soon as they are sent.
    #[test]
    fn a_read_returns_the_events_already_queued_without_waiting_for_more() {
        let fanout = fanout(64);
        let mut rx = fanout.subscribe(None);
        fanout.send(message(0));
        assert_eq!(poll_once(&mut rx), Some(Ok(message(0))));
        assert_eq!(
            poll_once(&mut rx),
            None,
            "nothing more is queued or buffered"
        );

        for seq in 1..=2 * RECV_BATCH as u64 {
            fanout.send(message(seq));
            assert_eq!(
                poll_once(&mut rx),
                Some(Ok(message(seq))),
                "event {seq} is returned as soon as it is sent"
            );
        }
    }

    /// The events after the first of a batch come from the receiver's own
    /// buffer, without the queue's lock: with the lock held by another
    /// thread, the rest of a full batch is still read. (A receiver that took
    /// one event per lock would block on the held lock here.)
    #[test]
    fn the_rest_of_a_batch_is_read_without_the_queues_lock() {
        let fanout = fanout(64);
        let mut rx = fanout.subscribe(None);
        for seq in 0..RECV_BATCH as u64 {
            fanout.send(message(seq));
        }
        assert_eq!(poll_once(&mut rx), Some(Ok(message(0))));

        let guard = fanout.lock();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let rx = &mut rx;
        std::thread::scope(|scope| {
            let reader = scope.spawn(move || {
                let rest: Vec<_> = (1..RECV_BATCH).map(|_| poll_once(rx)).collect();
                let _ = done_tx.send(());
                rest
            });
            let finished = done_rx.recv_timeout(BUFFERED_READ_DEADLINE);
            drop(guard);
            let rest = reader.join().expect("the reader thread finished");
            assert!(
                finished.is_ok(),
                "reading the rest of the batch waited on the queue's lock for \
                 {BUFFERED_READ_DEADLINE:?}: it was not read ahead under the first lock"
            );
            assert_eq!(
                rest,
                (1..RECV_BATCH as u64)
                    .map(|seq| Some(Ok(message(seq))))
                    .collect::<Vec<_>>()
            );
        });
    }

    /// A receiver holding a read-ahead batch while the sender overflows the
    /// queue returns its buffered events first: they were taken, so they
    /// cannot be lost. The lag that follows counts only the events past the
    /// batch, and its `membership_lost` is judged against that cursor: a join
    /// inside the batch (read) is no loss, one past it (lost) is.
    #[test]
    fn a_buffered_batch_comes_before_a_lag_that_counts_only_events_past_it() {
        const CAPACITY: usize = RECV_BATCH + 2;
        for (join_at, membership_lost) in [(RECV_BATCH - 1, false), (RECV_BATCH + 1, true)] {
            let fanout = fanout(CAPACITY);
            let mut rx = fanout.subscribe(None);
            let first: Vec<_> = (0..CAPACITY)
                .map(|i| {
                    if i == join_at {
                        joined("x")
                    } else {
                        message(i as u64)
                    }
                })
                .collect();
            for event in &first {
                fanout.send(event.clone());
            }
            // Takes the first batch, positions 0..RECV_BATCH.
            assert_eq!(poll_once(&mut rx), Some(Ok(first[0].clone())));
            for seq in 0..CAPACITY as u64 {
                fanout.send(message(100 + seq));
            }

            let expected: Vec<_> = first[1..RECV_BATCH]
                .iter()
                .map(|event| Ok(event.clone()))
                .chain(std::iter::once(Err(lagged(
                    (CAPACITY - RECV_BATCH) as u64,
                    membership_lost,
                ))))
                .chain((0..CAPACITY as u64).map(|seq| Ok(message(100 + seq))))
                .collect();
            assert_eq!(
                read_now(&mut rx),
                expected,
                "join at position {join_at}: the buffered events, then a lag of the {} events \
                 past the batch",
                CAPACITY - RECV_BATCH
            );
        }
    }

    /// `close` while events are buffered: the buffered events, then the rest
    /// of the queue, then `Closed`.
    #[test]
    fn close_with_a_buffered_batch_returns_the_buffered_events_then_closed() {
        const SENT: u64 = RECV_BATCH as u64 + 2;
        let fanout = fanout(64);
        let mut rx = fanout.subscribe(None);
        for seq in 0..SENT {
            fanout.send(message(seq));
        }
        assert_eq!(poll_once(&mut rx), Some(Ok(message(0))));
        fanout.close();

        let expected: Vec<_> = (1..SENT)
            .map(|seq| Ok(message(seq)))
            .chain(std::iter::once(Err(PresenceRecvError::Closed)))
            .collect();
        assert_eq!(read_now(&mut rx), expected);
    }

    /// Dropping a receiver with buffered events drops its hold on them: once
    /// every other holder is gone, so are the events.
    #[test]
    fn dropping_a_receiver_with_a_buffered_batch_releases_the_buffered_events() {
        const SENT: usize = 5;
        let fanout = fanout(64);
        let mut a = fanout.subscribe(None);
        let mut b = fanout.subscribe(None);
        for seq in 0..SENT as u64 {
            fanout.send(message(seq));
        }
        drop(a.recv().now_or_never().expect("queued").expect("an event"));
        let read_by_b: Vec<Arc<PresenceEvent>> = (0..SENT)
            .map(|_| b.recv().now_or_never().expect("queued").expect("an event"))
            .collect();
        assert_eq!(
            fanout.lock().queue.len(),
            0,
            "precondition: the queue released every event, A having taken them into its buffer"
        );
        assert!(
            read_by_b[1..]
                .iter()
                .all(|event| Arc::strong_count(event) == 2),
            "precondition: A's buffer and B hold each event after the first"
        );

        drop(a);

        assert!(
            read_by_b.iter().all(|event| Arc::strong_count(event) == 1),
            "dropping A released the events it had buffered"
        );
    }

    /// Dropping a receiver with buffered events leaves every event for the
    /// other receivers: the buffered ones were counted as read when they
    /// were taken, so `Drop` releases only the events past its cursor. The
    /// queue then drains as usual.
    #[test]
    fn dropping_a_receiver_with_a_buffered_batch_leaves_the_events_for_the_others() {
        const SENT: u64 = RECV_BATCH as u64 + 3;
        let fanout = fanout(64);
        let mut a = fanout.subscribe(None);
        let mut b = fanout.subscribe(None);
        for seq in 0..SENT {
            fanout.send(message(seq));
        }
        assert_eq!(poll_once(&mut a), Some(Ok(message(0))));

        drop(a);

        assert_eq!(
            read_now(&mut b),
            (0..SENT).map(|seq| Ok(message(seq))).collect::<Vec<_>>(),
            "B still reads every event"
        );
        assert!(
            fanout.allocated_slots() <= RETAINED_SLOTS,
            "and the queue drained once B had read them"
        );
        assert_eq!(
            fanout.send(message(SENT)),
            1,
            "only B is counted on a new send"
        );
    }

    // ── Cancel safety and wakeups ────────────────────────────────────────────

    /// Bound on a `recv` the test expects to return at once; reached only if
    /// a wakeup was lost.
    const WAKE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

    /// Bound on reading the rest of a read-ahead batch while another thread
    /// holds the queue's lock; reached only if those reads took the lock,
    /// that is, if the events were not read ahead under the first one.
    const BUFFERED_READ_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

    /// Install a hook on `fanout` that runs `action` in the first empty look
    /// of a `recv`, inside the gap between the look and the wait, and does
    /// nothing in later ones. Returns the count of empty looks the hook saw.
    fn on_first_empty_look(
        fanout: &SessionFanout,
        action: impl Fn(&SessionFanout) + Send + Sync + 'static,
    ) -> Arc<AtomicUsize> {
        let looks = Arc::new(AtomicUsize::new(0));
        let hook: AfterEmptyLook = {
            let looks = Arc::clone(&looks);
            Arc::new(move |fanout: &SessionFanout| {
                if looks.fetch_add(1, Ordering::SeqCst) == 0 {
                    action(fanout);
                }
            })
        };
        fanout.set_after_empty_look(Some(hook));
        looks
    }

    /// The deterministic gate for lost wakeups: a send that lands after a
    /// receiver found its queue empty and before it awaits its wakeup still
    /// wakes it, because its `Notified` was created before the look. The
    /// seam sends exactly inside that gap. Were the `Notified` created after
    /// the look, the send's wakeup would be lost and the receiver would sleep
    /// with the event queued, past the deadline.
    ///
    /// Its limit: the seam runs at one point of the gap, after the state
    /// block, so the test proves the `Notified` exists by then, not that it
    /// was created before the look. A `notified()` moved to between the end
    /// of the state block and the hook call would still pass here, though a
    /// send landing just before it would be lost.
    #[tokio::test(flavor = "current_thread")]
    async fn a_send_between_an_empty_look_and_the_wait_still_wakes_the_receiver() {
        let fanout = fanout(16);
        let mut rx = fanout.subscribe(None);
        let looks = on_first_empty_look(&fanout, |fanout| {
            assert_eq!(fanout.send(message(7)), 1, "the receiver is counted");
        });

        let received = tokio::time::timeout(WAKE_DEADLINE, rx.recv())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "the send inside the look-to-wait gap did not wake the receiver within \
                     {WAKE_DEADLINE:?}: a lost wakeup"
                )
            })
            .expect("an event, not an error");

        assert_eq!(*received, message(7));
        assert_eq!(
            looks.load(Ordering::SeqCst),
            1,
            "self-check: one empty look, whose gap took the send, then the event"
        );
    }

    /// The same for `close`: a close inside the look-to-wait gap wakes the
    /// receiver, which then ends `Closed`.
    #[tokio::test(flavor = "current_thread")]
    async fn a_close_between_an_empty_look_and_the_wait_still_ends_the_receiver_closed() {
        let fanout = fanout(16);
        let mut rx = fanout.subscribe(None);
        let looks = on_first_empty_look(&fanout, SessionFanout::close);

        let result = tokio::time::timeout(WAKE_DEADLINE, rx.recv())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "the close inside the look-to-wait gap did not wake the receiver within \
                     {WAKE_DEADLINE:?}: a lost wakeup"
                )
            });

        assert_eq!(result.err(), Some(PresenceRecvError::Closed));
        assert_eq!(
            looks.load(Ordering::SeqCst),
            1,
            "self-check: one empty look"
        );
    }

    /// A `recv` dropped while it waits — after events were sent and woke it,
    /// before it was polled again — loses nothing: the next calls return
    /// every event in order, and nothing else. Run with one event and with a
    /// burst of more than a batch, so the call after the cancelled wait also
    /// forms a read-ahead batch and reads the rest across batches.
    #[test]
    fn a_recv_dropped_after_being_woken_loses_nothing() {
        for burst in [1, RECV_BATCH as u64 + 2] {
            let fanout = fanout(64);
            let mut rx = fanout.subscribe(None);
            let wakes = Arc::new(CountingWaker::default());
            let waker = waker(Arc::clone(&wakes));
            let mut cx = Context::from_waker(&waker);

            let mut pending = Box::pin(rx.recv());
            assert!(pending.as_mut().poll(&mut cx).is_pending());
            for seq in 0..burst {
                fanout.send(message(seq));
            }
            assert!(
                wakes.0.load(Ordering::SeqCst) >= 1,
                "burst {burst}: the sends woke the waiter"
            );
            drop(pending);

            assert_eq!(
                read_now(&mut rx),
                (0..burst).map(|seq| Ok(message(seq))).collect::<Vec<_>>(),
                "burst {burst}: every event, in order, after the cancelled wait"
            );
        }
    }

    /// `close` wakes a waiting receiver, which then reports `Closed`.
    #[test]
    fn close_wakes_a_waiting_receiver_with_closed() {
        let fanout = fanout(16);
        let mut rx = fanout.subscribe(None);
        let wakes = Arc::new(CountingWaker::default());
        let waker = waker(Arc::clone(&wakes));
        let mut cx = Context::from_waker(&waker);

        let mut pending = Box::pin(rx.recv());
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        fanout.close();

        assert_eq!(wakes.0.load(Ordering::SeqCst), 1, "close woke the waiter");
        assert!(matches!(
            pending.as_mut().poll(&mut cx),
            Poll::Ready(Err(PresenceRecvError::Closed))
        ));
    }

    /// After `close` a receiver reads what was queued for it, then `Closed`,
    /// and `Closed` again on every later call.
    #[test]
    fn close_lets_receivers_read_what_is_queued_then_reports_closed() {
        let fanout = fanout(16);
        let mut rx = fanout.subscribe(None);
        fanout.send(message(0));
        fanout.send(message(1));
        fanout.close();

        assert_eq!(
            read_now(&mut rx),
            vec![
                Ok(message(0)),
                Ok(message(1)),
                Err(PresenceRecvError::Closed)
            ]
        );
        assert_eq!(poll_once(&mut rx), Some(Err(PresenceRecvError::Closed)));
    }

    /// The sequence number a test [`message`] carries.
    fn seq_of(event: &PresenceEvent) -> u64 {
        let PresenceEvent::PeerMessage { data, .. } = event else {
            panic!("expected only messages, got {event:?}");
        };
        u64::from_be_bytes(data.as_slice().try_into().expect("8 bytes"))
    }

    /// Every send wakes a receiver waiting on an empty queue, under
    /// multi-thread scheduling. The sender runs in lockstep with the
    /// receivers: after each send it waits until every receiver has read that
    /// event, so each receiver is back on an empty queue, looking or waiting,
    /// when the next send lands. Each read therefore finds exactly one event
    /// queued, so no read-ahead batch forms and every read goes through the
    /// queue's lock and, between events, its wait. A missed wakeup leaves that receiver parked
    /// and the sender waiting for it, so the deadline fails the test. (Without
    /// lockstep the next send would wake the receiver and hide the miss.) The
    /// capacity holds every event, so nothing can lag.
    ///
    /// Its sensitivity to the narrow gap between a receiver's empty look and
    /// its wait is statistical: the sender, woken through a `Notify` by the
    /// last receiver's read, almost always sends after that receiver is
    /// already waiting. With `notified()` moved after the look it passed 10
    /// runs of 10. [`a_send_between_an_empty_look_and_the_wait_still_wakes_the_receiver`]
    /// is the deterministic gate for that gap; this test covers the same
    /// invariant across threads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_send_wakes_a_receiver_waiting_on_an_empty_queue() {
        const RECEIVERS: usize = 4;
        const SENT: u64 = 20_000;
        const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
        let fanout = fanout(SENT as usize);
        // Per receiver, the events it has read; `acked` is woken after each.
        let read: Arc<Vec<AtomicU64>> =
            Arc::new((0..RECEIVERS).map(|_| AtomicU64::new(0)).collect());
        let acked = Arc::new(Notify::new());
        let mut tasks: Vec<_> = (0..RECEIVERS)
            .map(|i| {
                let mut rx = fanout.subscribe(None);
                let (read, acked) = (Arc::clone(&read), Arc::clone(&acked));
                tokio::spawn(async move {
                    loop {
                        match rx.recv().await {
                            Ok(event) => {
                                assert_eq!(
                                    seq_of(&event),
                                    read[i].load(Ordering::SeqCst),
                                    "receiver {i}: events arrive in order"
                                );
                                read[i].fetch_add(1, Ordering::SeqCst);
                                acked.notify_waiters();
                            }
                            Err(PresenceRecvError::Closed) => return,
                            Err(lag) => {
                                panic!("receiver {i}: {lag:?} at a capacity that cannot lag")
                            }
                        }
                    }
                })
            })
            .collect();
        let sent = Arc::new(AtomicU64::new(0));
        let mut sender = {
            let (fanout, read, acked, sent) = (
                Arc::clone(&fanout),
                Arc::clone(&read),
                Arc::clone(&acked),
                Arc::clone(&sent),
            );
            tokio::spawn(async move {
                for seq in 0..SENT {
                    fanout.send(message(seq));
                    sent.store(seq + 1, Ordering::SeqCst);
                    loop {
                        // Created before the check, so an ack between the
                        // check and the wait still wakes it.
                        let notified = acked.notified();
                        if read.iter().all(|n| n.load(Ordering::SeqCst) > seq) {
                            break;
                        }
                        notified.await;
                    }
                }
                fanout.close();
            })
        };

        let finished = tokio::time::timeout(DEADLINE, async {
            (&mut sender).await.expect("the sender finished");
            for task in tasks.iter_mut() {
                task.await.expect("a receiver finished");
            }
        })
        .await;

        if finished.is_err() {
            // A receiver that panicked also stalls the sender: report its
            // panic, not a lost wakeup. While the sender is unfinished no
            // receiver handle has been awaited yet, so each can be awaited
            // here; once it has finished, a stalled receiver missed the
            // `close` wakeup.
            if !sender.is_finished() {
                for task in tasks.iter_mut().filter(|task| task.is_finished()) {
                    if let Err(err) = task.await {
                        if err.is_panic() {
                            std::panic::resume_unwind(err.into_panic());
                        }
                    }
                }
            }
            let reads: Vec<u64> = read.iter().map(|n| n.load(Ordering::SeqCst)).collect();
            panic!(
                "event {} of {SENT} was not read by every receiver within {DEADLINE:?} (events \
                 read per receiver: {reads:?}): a send's wakeup was lost while a receiver waited \
                 on an empty queue",
                sent.load(Ordering::SeqCst)
            );
        }
    }

    /// Accounting and order under lag, on a multi-thread runtime. With a
    /// capacity of 8 and a sender that does not wait for the receivers,
    /// receivers lag often. Each accounts for every event sent (received plus
    /// reported dropped), reads the survivors in order, and ends `Closed`
    /// after `close`. It does not check for lost wakeups: a wakeup missed
    /// mid-stream is covered by the next send. The deterministic check for
    /// the look-to-wait gap is the two seam tests,
    /// [`a_send_between_an_empty_look_and_the_wait_still_wakes_the_receiver`]
    /// and [`a_close_between_an_empty_look_and_the_wait_still_ends_the_receiver_closed`];
    /// [`every_send_wakes_a_receiver_waiting_on_an_empty_queue`] covers
    /// wakeups across threads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lagging_receivers_account_for_every_event_sent_and_read_the_rest_in_order() {
        const CAPACITY: usize = 8;
        const RECEIVERS: usize = 4;
        const SENT: u64 = 20_000;
        const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
        let fanout = fanout(CAPACITY);
        let tasks: Vec<_> = (0..RECEIVERS)
            .map(|_| {
                let mut rx = fanout.subscribe(None);
                tokio::spawn(async move {
                    let (mut received, mut dropped, mut next) = (0u64, 0u64, 0u64);
                    loop {
                        match rx.recv().await {
                            Ok(event) => {
                                assert_eq!(seq_of(&event), next, "events arrive in order");
                                next += 1;
                                received += 1;
                            }
                            Err(PresenceRecvError::Lagged { dropped: n, .. }) => {
                                dropped += n;
                                next += n;
                            }
                            Err(PresenceRecvError::Closed) => return (received, dropped),
                        }
                    }
                })
            })
            .collect();
        let sender = {
            let fanout = Arc::clone(&fanout);
            tokio::spawn(async move {
                for seq in 0..SENT {
                    fanout.send(message(seq));
                    if seq % 64 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
                fanout.close();
            })
        };

        tokio::time::timeout(DEADLINE, async {
            sender.await.expect("the sender finished");
            for task in tasks {
                let (received, dropped) = task.await.expect("a receiver finished");
                assert_eq!(
                    received + dropped,
                    SENT,
                    "received {received} + dropped {dropped}"
                );
            }
        })
        .await
        .unwrap_or_else(|_| panic!("a receiver did not reach Closed within {DEADLINE:?}"));
    }
}
