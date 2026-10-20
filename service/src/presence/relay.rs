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

//! The receive side shared by the replica-wide pub/sub relays: the presence
//! relay (`presence::spawn_presence_subscriber`) and the session-deleted relay
//! (`grpc::session::spawn_session_delete_subscriber`).
//!
//! fred delivers every pub/sub message on the `SubscriberClient`'s connection
//! through one tokio `broadcast` channel
//! (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY` slots), and each
//! relay holds its own receiver on it. fred's router task pushes with
//! `Sender::send`, which is not subject to tokio's cooperative budget, but
//! `Receiver::recv` is: after 128 ready operations in one task poll it
//! returns `Pending` and puts the task at the back of the run queue. A relay
//! that only ever awaited `recv()` therefore took at most about 128 messages
//! per turn. Under the 30,000-connection join and close waves of the
//! keeper-load runs, relays fell more than the channel's capacity behind and
//! lost messages for the whole replica; that this per-turn cap, rather than
//! the capacity, is what let them fall behind is inferred from the code and
//! those runs' logs (lags milliseconds apart, far inside the time a wave fills
//! the channel). The drain below is not measured at that scale, so this stays
//! inferred there (SDD §13.3, *Headroom at extreme scale is untested*); at
//! 1,000 streams under 1,000 messages/s neither relay lagged
//! (docs-internal/validation/relay-drain-20261005/report.md §2).
//!
//! [`PubsubRelayReceiver`] lets a relay take everything already queued each
//! time it runs: the relay awaits one message with
//! [`recv`](PubsubRelayReceiver::recv), then hands it to
//! [`drain`](PubsubRelayReceiver::drain), which takes the rest with
//! `try_recv` (not cooperative) until the channel is empty. A drain reports
//! the lag it saw as one sum, so the relay adds to its lag counter and logs
//! its WARN once per batch rather than once per `Lagged`; a WARN per lag left
//! a relay sitting at the channel's tail paying a log write for every few
//! messages it lost.
//!
//! **Yield bound.** So that one relay cannot hold a worker thread for as long
//! as messages keep arriving, a drain stops with [`DrainEnd::YieldDue`] once
//! the relay has handled one channel capacity's worth of messages since it
//! last yielded at that boundary. The relay then yields
//! (`tokio::task::yield_now`) and checks its cancellation token before it
//! reads again. The count carries across drains that end
//! [`DrainEnd::Empty`], so a relay that suspended in `recv()` in between
//! yields at most early, never late.

use fred::clients::SubscriberClient;
use fred::interfaces::{ClientLike, EventInterface};
use fred::types::Message;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::sync::broadcast::Receiver;

/// A replica-wide relay's receiver on the pub/sub client's message channel,
/// drained in batches. See the module docs.
pub struct PubsubRelayReceiver {
    rx: Receiver<Message>,
    /// Messages handled before the relay must yield: the channel capacity the
    /// client was built with, at least 1.
    yield_after: usize,
    /// Messages handled since the relay last yielded at the bound.
    handled_since_yield: usize,
}

/// What one [`PubsubRelayReceiver::drain`] call saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainBatch {
    /// The sum of every `Lagged(n)` the batch met: messages lost for this
    /// relay. The relay adds it to its lag counter and logs its WARN once,
    /// when it is non-zero.
    pub lagged: u64,
    /// Why the batch ended.
    pub end: DrainEnd,
}

/// Why a [`PubsubRelayReceiver::drain`] call returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainEnd {
    /// The channel is empty: await [`PubsubRelayReceiver::recv`] again.
    Empty,
    /// The relay has handled one channel capacity's worth of messages since
    /// it last yielded: yield, check cancellation, then read again. Messages
    /// may still be queued.
    YieldDue,
    /// fred closed the channel (the client quit): the relay ends.
    Closed,
}

impl PubsubRelayReceiver {
    /// Take a new receiver on `subscriber`'s message channel. It sees every
    /// message published from this call on, so a relay takes it before it
    /// spawns its task.
    pub fn new(subscriber: &SubscriberClient) -> Self {
        let yield_after = subscriber.perf_config().broadcast_channel_capacity.max(1);
        Self {
            rx: subscriber.message_rx(),
            yield_after,
            handled_since_yield: 0,
        }
    }

    /// A relay receiver on a bare broadcast receiver, yielding after
    /// `yield_after` messages (at least 1): [`new`](Self::new) without a
    /// client, so a bench or this module's tests can drive the drain from a
    /// channel they fill themselves.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_receiver(rx: Receiver<Message>, yield_after: usize) -> Self {
        Self {
            rx,
            yield_after: yield_after.max(1),
            handled_since_yield: 0,
        }
    }

    /// Await the next message. Cancel-safe (tokio's `Receiver::recv`), so it
    /// can sit in a `select!` beside the relay's other arms; pass its result
    /// to [`drain`](Self::drain).
    pub async fn recv(&mut self) -> Result<Message, RecvError> {
        self.rx.recv().await
    }

    /// Hand `first` (a [`recv`](Self::recv) result) and then every message
    /// already queued to `handle`, in order, until the channel is empty, the
    /// yield bound is reached, or the channel closes. A `Lagged(n)` from
    /// either `recv` or `try_recv` is summed into [`DrainBatch::lagged`] and
    /// the drain goes on.
    ///
    /// Synchronous: it never awaits, so a relay's `select!` arm runs the
    /// whole batch and its lag report before any other arm is polled.
    pub fn drain(
        &mut self,
        first: Result<Message, RecvError>,
        mut handle: impl FnMut(Message),
    ) -> DrainBatch {
        let mut lagged = 0u64;
        match first {
            Ok(msg) => {
                handle(msg);
                self.handled_since_yield += 1;
            }
            Err(RecvError::Lagged(n)) => lagged = n,
            Err(RecvError::Closed) => {
                return DrainBatch {
                    lagged,
                    end: DrainEnd::Closed,
                }
            }
        }
        let end = loop {
            if self.handled_since_yield >= self.yield_after {
                self.handled_since_yield = 0;
                break DrainEnd::YieldDue;
            }
            match self.rx.try_recv() {
                Ok(msg) => {
                    handle(msg);
                    self.handled_since_yield += 1;
                }
                Err(TryRecvError::Lagged(n)) => lagged = lagged.saturating_add(n),
                Err(TryRecvError::Empty) => break DrainEnd::Empty,
                Err(TryRecvError::Closed) => break DrainEnd::Closed,
            }
        };
        DrainBatch { lagged, end }
    }
}

#[cfg(test)]
mod tests {
    //! The drain on a bare broadcast channel ([`PubsubRelayReceiver::from_receiver`]):
    //! the batch, lag-sum, yield-bound and closed-channel semantics. The real
    //! relays' drains, on a subscriber client, are driven in
    //! `tests/integration/grpc/pubsub_relay_lag_test.rs`.

    use super::*;
    use fred::types::config::Server;
    use fred::types::{MessageKind, Value};
    use tokio::sync::broadcast;

    /// A pub/sub message whose payload is `payload`.
    fn message(payload: &str) -> Message {
        Message {
            channel: "relay-drain-test".into(),
            value: Value::String(payload.into()),
            kind: MessageKind::Message,
            server: Server::new("relay-drain-test", 6379),
        }
    }

    /// `message`'s payload.
    fn payload(message: &Message) -> String {
        message.value.clone().convert().expect("a string payload")
    }

    /// One drain of everything queued after the awaited message: every message
    /// reaches the handler in order, and the batch ends `Empty`. Then a drain that
    /// meets a lag on its awaited receive and another on a `try_recv` (the handler
    /// publishes past the capacity mid-batch) reports the two as one sum, and
    /// still hands over every message the channel kept.
    #[tokio::test]
    async fn relay_drain_hands_over_every_queued_message_and_sums_its_lags() {
        let (tx, rx) = broadcast::channel(4);
        let mut relay = PubsubRelayReceiver::from_receiver(rx, 64);

        for i in 0..3 {
            tx.send(message(&format!("m{i}")))
                .expect("a receiver is open");
        }
        let mut handled = Vec::new();
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| handled.push(payload(&msg)));
        assert_eq!(
            handled,
            ["m0", "m1", "m2"],
            "one drain takes the whole queue"
        );
        assert_eq!(batch.lagged, 0);
        assert_eq!(batch.end, DrainEnd::Empty);

        // Ten into a channel of four: the awaited receive reports Lagged(6) and
        // the channel keeps a6..a9.
        for i in 0..10 {
            tx.send(message(&format!("a{i}")))
                .expect("a receiver is open");
        }
        let mut handled = Vec::new();
        let mut published_mid_batch = false;
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| {
            handled.push(payload(&msg));
            if !published_mid_batch {
                // Six more while a7..a9 are still unread: the channel keeps
                // b2..b5, so the next try_recv reports Lagged(5) (a7..a9, b0, b1).
                published_mid_batch = true;
                for i in 0..6 {
                    tx.send(message(&format!("b{i}")))
                        .expect("a receiver is open");
                }
            }
        });
        assert_eq!(
            handled,
            ["a6", "b2", "b3", "b4", "b5"],
            "a drain goes on past each lag with what the channel kept"
        );
        assert_eq!(
            batch.lagged,
            6 + 5,
            "a drain reports every lag it met as one sum"
        );
        assert_eq!(batch.end, DrainEnd::Empty);
    }

    /// A drain ends `YieldDue` once the relay has handled its yield bound of
    /// messages since it last yielded, leaving the rest queued, and the count
    /// carries across drains that end `Empty`, so the bound is never exceeded.
    #[tokio::test]
    async fn relay_drain_ends_for_a_yield_after_its_bound_counted_across_batches() {
        const YIELD_AFTER: usize = 4;
        let (tx, rx) = broadcast::channel(16);
        let mut relay = PubsubRelayReceiver::from_receiver(rx, YIELD_AFTER);
        let send = |range: std::ops::Range<usize>| {
            for i in range {
                tx.send(message(&format!("m{i}")))
                    .expect("a receiver is open");
            }
        };

        send(0..3);
        let mut handled = Vec::new();
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| handled.push(payload(&msg)));
        assert_eq!(handled, ["m0", "m1", "m2"]);
        assert_eq!(batch.end, DrainEnd::Empty, "three is under the bound");

        send(3..6);
        let mut handled = Vec::new();
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| handled.push(payload(&msg)));
        assert_eq!(
            handled,
            ["m3"],
            "the fourth message since the last yield reaches the bound"
        );
        assert_eq!(batch.end, DrainEnd::YieldDue);
        assert_eq!(batch.lagged, 0);

        let mut handled = Vec::new();
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| handled.push(payload(&msg)));
        assert_eq!(
            handled,
            ["m4", "m5"],
            "after the yield the relay reads what was left queued"
        );
        assert_eq!(batch.end, DrainEnd::Empty);
    }

    /// A drain hands over what is still queued when the sender closes, then ends
    /// `Closed`; a closed, empty channel ends the next drain `Closed` at once.
    #[tokio::test]
    async fn relay_drain_reports_a_closed_channel_after_its_last_message() {
        let (tx, rx) = broadcast::channel(4);
        let mut relay = PubsubRelayReceiver::from_receiver(rx, 64);
        tx.send(message("m0")).expect("a receiver is open");
        tx.send(message("m1")).expect("a receiver is open");
        drop(tx);

        let mut handled = Vec::new();
        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| handled.push(payload(&msg)));
        assert_eq!(handled, ["m0", "m1"]);
        assert_eq!(batch.end, DrainEnd::Closed);

        let first = relay.recv().await;
        let batch = relay.drain(first, |msg| panic!("no message is left: {msg:?}"));
        assert_eq!(batch.end, DrainEnd::Closed);
        assert_eq!(batch.lagged, 0);
    }
}
