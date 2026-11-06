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

//! A `SubscribeReadLayer` response queue: the queue between a subscription's
//! task and the response stream tonic polls (SDD §12.6).
//!
//! It holds up to a fixed number of **data** messages — 32 on a
//! subscription — and, apart from them, **one status slot**, reserved when
//! the queue is created. So the terminal status always has somewhere to go:
//! [`ResponseSender::send_status`] is synchronous and infallible, never waits
//! behind the backlog it is usually reporting on, and a server-driven ending
//! can never degrade into a bare end of stream, which a client reads as `OK`.
//!
//! **Discard.** [`ResponseSender::clear_data`] drops every queued data
//! message at once. A subscription the server ends — anything but the
//! client's own disconnect — clears its data before it enqueues its status,
//! so the status is the next message a client that resumes reading gets, and
//! the memory the queued messages held (their relay copies' and catch-up
//! pages' charges) is released at the ending, not when a stalled peer drains.
//!
//! The receiver yields every data message in order, then the status, then
//! the end of stream once the sender is gone. Dropping the receiver — the
//! client went away — is what [`ResponseSender::closed`] observes, and it
//! drops whatever is still queued. Storage grows with the messages queued,
//! never ahead of them.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};

use futures::task::AtomicWaker;
use tokio::sync::Notify;

/// The data messages a subscription's response queue holds.
pub const RESPONSE_QUEUE_DATA_SLOTS: usize = 32;

struct State<T> {
    data: VecDeque<T>,
    status: Option<T>,
    data_slots: usize,
    receiver_dropped: bool,
    sender_done: bool,
}

struct Shared<T> {
    state: StdMutex<State<T>>,
    /// Woken when a data slot frees or the receiver drops.
    space: Notify,
    /// Woken when the receiver drops.
    closed: Notify,
    /// The receiver's task.
    receiver: AtomicWaker,
}

impl<T> Shared<T> {
    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // Nothing panics while holding the lock and every update leaves the
        // state consistent, so a poisoned lock's state is still sound.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A response queue of `data_slots` data messages plus the reserved status
/// slot (module docs).
pub fn response_queue<T>(data_slots: usize) -> (ResponseSender<T>, ResponseReceiver<T>) {
    let shared = Arc::new(Shared {
        state: StdMutex::new(State {
            data: VecDeque::new(),
            status: None,
            data_slots: data_slots.max(1),
            receiver_dropped: false,
            sender_done: false,
        }),
        space: Notify::new(),
        closed: Notify::new(),
        receiver: AtomicWaker::new(),
    });
    (
        ResponseSender {
            shared: Arc::clone(&shared),
        },
        ResponseReceiver { shared },
    )
}

/// The subscription task's end of a response queue.
pub struct ResponseSender<T> {
    shared: Arc<Shared<T>>,
}

impl<T> std::fmt::Debug for ResponseSender<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.shared.lock();
        f.debug_struct("ResponseSender")
            .field("queued", &st.data.len())
            .field("data_slots", &st.data_slots)
            .field("receiver_dropped", &st.receiver_dropped)
            .finish()
    }
}

impl<T> ResponseSender<T> {
    /// The data slots free now: 0 when a send would wait.
    pub fn capacity(&self) -> usize {
        let st = self.shared.lock();
        st.data_slots.saturating_sub(st.data.len())
    }

    /// Whether the receiver has been dropped: the client went away.
    pub fn is_closed(&self) -> bool {
        self.shared.lock().receiver_dropped
    }

    /// Resolves once the receiver has been dropped.
    pub async fn closed(&self) {
        loop {
            let notified = self.shared.closed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_closed() {
                return;
            }
            notified.await;
        }
    }

    /// Queues `item` as a data message, waiting for a free data slot. `Err`
    /// hands it back when the receiver has been dropped. Dropping the future
    /// mid-wait drops `item` unsent.
    pub async fn send(&self, item: T) -> Result<(), T> {
        let mut item = Some(item);
        loop {
            let space = self.shared.space.notified();
            tokio::pin!(space);
            space.as_mut().enable();
            {
                let mut st = self.shared.lock();
                if st.receiver_dropped {
                    return Err(item.take().expect("an unsent item is held"));
                }
                if st.data.len() < st.data_slots {
                    st.data
                        .push_back(item.take().expect("an unsent item is held"));
                    drop(st);
                    self.shared.receiver.wake();
                    return Ok(());
                }
            }
            space.await;
        }
    }

    /// Queues `item` as a data message if a data slot is free now; otherwise,
    /// or when the receiver has been dropped, hands it back.
    pub fn try_send(&self, item: T) -> Result<(), T> {
        let mut st = self.shared.lock();
        if st.receiver_dropped || st.data.len() >= st.data_slots {
            return Err(item);
        }
        st.data.push_back(item);
        drop(st);
        self.shared.receiver.wake();
        Ok(())
    }

    /// Drops every queued data message at once (module docs, *Discard*), and
    /// returns how many there were. The messages are dropped after the
    /// queue's lock is released.
    pub fn clear_data(&self) -> usize {
        let cleared = std::mem::take(&mut self.shared.lock().data);
        let n = cleared.len();
        drop(cleared);
        self.shared.space.notify_waiters();
        n
    }

    /// Puts `status` in the reserved status slot and ends the queue: the
    /// receiver yields it after the data still queued, then the end of
    /// stream. Synchronous, infallible and unconditional: the slot was
    /// reserved when the queue was made, so nothing can refuse or delay it. A
    /// dropped receiver discards it.
    pub fn send_status(self, status: T) {
        {
            let mut st = self.shared.lock();
            if !st.receiver_dropped {
                st.status = Some(status);
            }
        }
        // Dropping `self` marks the sender done and wakes the receiver.
    }
}

impl<T> Drop for ResponseSender<T> {
    fn drop(&mut self) {
        self.shared.lock().sender_done = true;
        self.shared.receiver.wake();
    }
}

/// The response stream's end of a response queue.
pub struct ResponseReceiver<T> {
    shared: Arc<Shared<T>>,
}

impl<T> std::fmt::Debug for ResponseReceiver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseReceiver").finish()
    }
}

impl<T> ResponseReceiver<T> {
    /// The next message if one is ready now ([`Self::poll_recv`]'s order);
    /// `None` when nothing is queued or the queue has ended.
    pub fn try_recv(&mut self) -> Option<T> {
        let mut st = self.shared.lock();
        if let Some(item) = st.data.pop_front() {
            drop(st);
            self.shared.space.notify_waiters();
            return Some(item);
        }
        st.status.take()
    }

    /// The next message: a data message, in order; then the status; then
    /// `None` once the sender is gone and nothing is left.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.shared.receiver.register(cx.waker());
        let mut st = self.shared.lock();
        if let Some(item) = st.data.pop_front() {
            drop(st);
            self.shared.space.notify_waiters();
            return Poll::Ready(Some(item));
        }
        if let Some(status) = st.status.take() {
            return Poll::Ready(Some(status));
        }
        if st.sender_done {
            return Poll::Ready(None);
        }
        Poll::Pending
    }
}

impl<T> Drop for ResponseReceiver<T> {
    fn drop(&mut self) {
        let (data, status) = {
            let mut st = self.shared.lock();
            st.receiver_dropped = true;
            (std::mem::take(&mut st.data), st.status.take())
        };
        drop(data);
        drop(status);
        self.shared.closed.notify_waiters();
        self.shared.space.notify_waiters();
    }
}

/// The queue's contract on its own (module docs): the reserved status slot,
/// the discard, order, waiting for a data slot, and the receiver's drop. The
/// subscription's use of it — which endings discard, and the status a client
/// then reads — is driven end to end in `tests/integration/grpc`
/// (`subscribe_admission_test`, `delivery_budget_test`).
#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use futures::FutureExt;

    /// Bound on any wait here; unreachable on the healthy path.
    const SETTLE: Duration = Duration::from_secs(5);

    /// A message that counts its drops, standing for one holding memory (a
    /// relay copy's slice, a catch-up page): its drop is its release.
    struct Counted(u32, Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Everything the receiver yields until the end of stream, as the
    /// messages' numbers.
    async fn drain(rx: &mut ResponseReceiver<Counted>) -> Vec<u32> {
        let mut out = Vec::new();
        tokio::time::timeout(SETTLE, async {
            while let Some(item) = std::future::poll_fn(|cx| rx.poll_recv(cx)).await {
                out.push(item.0);
            }
        })
        .await
        .expect("the queue ends once its sender is gone");
        out
    }

    /// The status slot is apart from the data slots: with every data slot
    /// full, a data send has to wait, and the status still goes in at once;
    /// without a discard the receiver yields the data in order, then the
    /// status, then the end of stream.
    #[tokio::test]
    async fn the_status_goes_in_past_full_data_slots_and_comes_out_after_the_data() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = response_queue::<Counted>(3);
        for n in 0..3 {
            tx.try_send(Counted(n, Arc::clone(&drops)))
                .map_err(|_| ())
                .expect("a data slot is free");
        }
        assert_eq!(tx.capacity(), 0);
        assert!(
            tx.try_send(Counted(99, Arc::clone(&drops))).is_err(),
            "every data slot is full"
        );
        assert!(
            tx.send(Counted(98, Arc::clone(&drops)))
                .now_or_never()
                .is_none(),
            "a data send waits for a slot"
        );
        tx.send_status(Counted(1_000, Arc::clone(&drops)));
        assert_eq!(drain(&mut rx).await, vec![0, 1, 2, 1_000]);
    }

    /// The discard: `clear_data` drops every queued data message at once —
    /// releasing what each held, with no reader — and the status enqueued
    /// after it is the next and last message.
    #[tokio::test]
    async fn clear_data_drops_every_queued_message_at_once_and_the_status_is_next() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = response_queue::<Counted>(RESPONSE_QUEUE_DATA_SLOTS);
        for n in 0..RESPONSE_QUEUE_DATA_SLOTS as u32 {
            tx.try_send(Counted(n, Arc::clone(&drops)))
                .map_err(|_| ())
                .expect("a data slot is free");
        }
        assert_eq!(tx.clear_data(), RESPONSE_QUEUE_DATA_SLOTS);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            RESPONSE_QUEUE_DATA_SLOTS,
            "every queued message is dropped by the clear, not when a reader drains it"
        );
        assert_eq!(tx.capacity(), RESPONSE_QUEUE_DATA_SLOTS);
        tx.send_status(Counted(1_000, Arc::clone(&drops)));
        assert_eq!(drain(&mut rx).await, vec![1_000]);
    }

    /// A send parked on a full queue is woken by the clear and goes in.
    #[tokio::test]
    async fn a_send_parked_on_a_full_queue_goes_in_once_the_queue_is_cleared() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = response_queue::<Counted>(1);
        tx.try_send(Counted(0, Arc::clone(&drops)))
            .map_err(|_| ())
            .expect("the slot is free");
        let send = tx.send(Counted(1, Arc::clone(&drops)));
        tokio::pin!(send);
        assert!(futures::poll!(send.as_mut()).is_pending());
        assert_eq!(tx.clear_data(), 1);
        assert!(
            tokio::time::timeout(SETTLE, send)
                .await
                .expect("woken")
                .is_ok(),
            "the parked send goes in"
        );
        assert_eq!(rx.try_recv().map(|c| c.0), Some(1));
    }

    /// The receiver taking a message frees a slot for a parked send.
    #[tokio::test]
    async fn a_take_frees_a_data_slot_for_a_parked_send() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = response_queue::<Counted>(1);
        tx.try_send(Counted(0, Arc::clone(&drops)))
            .map_err(|_| ())
            .expect("the slot is free");
        let send = tx.send(Counted(1, Arc::clone(&drops)));
        tokio::pin!(send);
        assert!(futures::poll!(send.as_mut()).is_pending());
        assert_eq!(rx.try_recv().map(|c| c.0), Some(0));
        assert!(tokio::time::timeout(SETTLE, send)
            .await
            .expect("woken by the take")
            .is_ok());
        assert_eq!(rx.try_recv().map(|c| c.0), Some(1));
    }

    /// The client went away: `closed` resolves, a parked send and every later
    /// one hand their message back, whatever was queued is dropped with the
    /// receiver, and a status has nowhere to go.
    #[tokio::test]
    async fn a_dropped_receiver_closes_the_queue_and_drops_what_it_held() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = response_queue::<Counted>(1);
        tx.try_send(Counted(0, Arc::clone(&drops)))
            .map_err(|_| ())
            .expect("the slot is free");
        let returned = {
            let send = tx.send(Counted(1, Arc::clone(&drops)));
            tokio::pin!(send);
            assert!(futures::poll!(send.as_mut()).is_pending());
            assert!(!tx.is_closed());
            drop(rx);
            assert!(tx.is_closed());
            tokio::time::timeout(SETTLE, tx.closed())
                .await
                .expect("closed resolves once the receiver is gone");
            tokio::time::timeout(SETTLE, send)
                .await
                .expect("the parked send is woken")
                .expect_err("and hands its message back")
        };
        assert_eq!(returned.0, 1);
        drop(returned);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "the queued message went with it"
        );
        assert!(tx.try_send(Counted(2, Arc::clone(&drops))).is_err());
        tx.send_status(Counted(1_000, Arc::clone(&drops)));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            4,
            "a status for a receiver that is gone is discarded"
        );
    }

    /// A queue asked for no data slot still has one: a zero would make every
    /// data send wait for ever.
    #[tokio::test]
    async fn a_queue_of_zero_data_slots_has_one() {
        let (tx, mut rx) = response_queue::<u8>(0);
        assert_eq!(tx.capacity(), 1);
        tx.try_send(7).expect("one data slot");
        assert_eq!(rx.try_recv(), Some(7));
    }
}
