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

//! Criterion A/B of the per-session presence fan-out: the production path
//! against the tokio broadcast channel it replaced. Report-only: it asserts
//! no ratio.
//!
//! ## The two arms
//!
//! - **`fanout`**, the production path: a [`LocalPresenceRouter`], `R`
//!   receivers from `join_channel(sid, Some(cid))` (the first join's
//!   subscribe duty reported as succeeded, as a successful `SUBSCRIBE`
//!   would), and one `broadcast(sid, event)` per event. Each event is
//!   wrapped in one `Arc` and queued once; `recv` hands every receiver a
//!   clone of that `Arc`, and the last reader frees the event, on whichever
//!   thread it runs.
//! - **`tokio_broadcast`**, the replaced path as it stood before the
//!   per-session fan-out queue: a `DashMap` from session id to an entry
//!   holding a `tokio::sync::broadcast::Sender<PresenceEvent>` of the same
//!   capacity, receivers from `sender.subscribe()`, and a broadcast that
//!   looks the entry up and calls `sender.send(event)`, as the old
//!   `LocalPresenceRouter::broadcast` did. tokio's `recv` clones the
//!   `PresenceEvent` (its `String` and 200-byte `Vec`) out of the slot for
//!   every receiver, which is what the old JoinSession loop received.
//!
//! Downstream of `recv` both JoinSession loops did the same thing with the
//! event (`presence_event_to_proto(&event)`, by reference), so the sink here
//! is the same for both: `black_box` of a reference to the event.
//!
//! ## The scenario
//!
//! One session, `R` receiver tasks, each looping on `recv` on a multi-thread
//! tokio runtime with a fixed [`WORKERS`] worker threads (not the service's
//! per-core default, so the figure does not move with the host's core
//! count). One sender, on the thread driving `block_on`, publishes without
//! pacing.
//!
//! The measured unit is one **round**: the sender publishes `N` events, then
//! waits until every receiver has read all of them; the last receiver to
//! finish a round wakes the sender through a `Notify`. Each criterion
//! iteration is one round, and throughput is `N × R` delivered events, so
//! criterion's element rate inverts to the cost per delivered event. Four
//! groups, `presence_fanout/{R}_receivers/{N}_per_round`, cross two
//! receiver counts with two regimes:
//!
//! - `R` = 400, a large session under a heavy message load, and `R` = 5, a
//!   small editing session;
//! - **`N` = 1, lockstep**: every send lands on parked receivers, so a round
//!   exercises the wake path (`notify_waiters` waking `R` parked receivers,
//!   each re-registering its `Notified` in the `Notify`'s waiter list, the
//!   tasks' rescheduling). The 400-receiver load's 1,000 messages/s, about
//!   1 ms apart, and the 5-receiver session's ~1 update/s are this regime.
//!   The round barrier (`R` atomic increments and one `notify_one` per
//!   round) costs the same in both arms and is charged to every event here,
//!   so it pulls the lockstep ratio towards 1: the arms' own paths differ by
//!   at least what it shows;
//! - **`N` = 2048, saturated drain**: the sender runs ahead and receivers
//!   drain bursts, the barrier amortised over 2,048 events.
//!
//! The event mix follows the 400-receiver load (16 senders at 1,000
//! messages/s, 8 join/leave churners at about 39 cycles/s): out of every
//! 26 events, 24 are a `PeerMessage` of 200 bytes from one of 16 sender
//! ids, one a `PeerJoined` and one a `PeerLeft` of a churner id. Events are
//! taken in turn from a prebuilt cycle of [`MAX_ROUND`] events, so lockstep
//! rounds see the same mix, and each is cloned before it is broadcast, the
//! same in both arms, standing in for the relay's parse of the pub/sub
//! payload.
//!
//! ## What is measured: process CPU time
//!
//! On unix each round is timed as process CPU time
//! (`clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`: every thread, sender, workers
//! and the blocked main thread alike), and `iter_custom` returns the sum, so
//! the element rate inverts to **CPU per delivered event**, what the
//! 400-receiver load run measured from `/proc/<pid>/stat`. Parked threads
//! add nothing. The returned CPU durations feed criterion's statistics and
//! throughput only; it plans iteration counts from its own wall-clock
//! bracket around `iter_custom`. The `clock_gettime` pair around each round,
//! like the round barrier, is charged equally to both arms and pulls the
//! ratio towards 1. On other targets the bench falls back to wall time,
//! which is not comparable with that load run.
//!
//! ## Allocator
//!
//! On linux-gnu the bench runs on jemalloc, the service's global allocator
//! there (`src/allocator.rs`): the old arm's per-receiver cost is mostly
//! allocator work (two allocations and two frees per receiver per event,
//! for the event's `String` and `Vec`), and the new arm frees each event on
//! whichever thread reads it last, so glibc malloc would give a ratio that
//! is not production's. The service's compiled-in `_rjem_malloc_conf`
//! options (`background_thread:true`, `thp:never`) are not reproduced; they
//! govern purging and huge pages, not the per-allocation path.
//!
//! ## Neither arm may lag
//!
//! A lagging receiver skips events and would report a figure for the lag
//! path, not delivery. Every round is followed by a full drain, and
//! `2 × MAX_ROUND` is at most the capacity (checked at compile time), so a
//! round cannot overflow the queue. Lagged events are still counted, and
//! the bench panics after any round that lagged, before criterion can print
//! a figure. A startup self-check builds each arm with a capacity below its
//! round size and asserts that lag *is* counted, so the zero-lag check
//! cannot pass vacuously.
//!
//! ## Running it
//!
//! ```text
//! cargo bench -j 4 -p live-edit --bench presence_fanout
//! ```
//!
//! No `required-features` and no Valkey: `join_channel`, `broadcast` and
//! `PresenceReceiver` are all `pub`. Criterion's windows are set in the
//! `criterion_group!` config at the bottom (3 s warm-up, 20 s measurement,
//! 30 samples); command-line flags override them. Each id runs for about
//! its warm-up plus measurement window in wall time, longer only when
//! criterion's minimum iteration count for 30 samples outlasts the window
//! (it warns when it does), so the eight ids finish in a few minutes.
//!
//! **Trust the ratio of the two arms within one run, and re-run until
//! consecutive runs agree.** Criterion runs each arm's window one after the
//! other, never interleaved, so host drift between the windows lands on one
//! arm. Not gated in CI: no ceiling applies to any id here.
//!
//! ## The replica-wide relays (`--features test-support`)
//!
//! With `test-support` the bench adds the cost of the replica-wide pub/sub
//! relays upstream of the fan-out: the work one relay turn does per message,
//! on the worker it runs on. Without the feature these groups are compiled
//! out and the fan-out groups above run alone, as before.
//!
//! ```text
//! cargo bench -j 4 -p live-edit --features test-support --bench presence_fanout -- relay
//! ```
//!
//! - **`presence_relay/{R}_receivers/{N}_per_round`**, `R` in 1, 50, 200 and
//!   400 parked receivers: the presence relay's turn, through the production
//!   [`PubsubRelayReceiver`] and the relay's own per-message step
//!   (`relay_presence_message_for_bench`): the `recv`/`try_recv` clone of the
//!   fred `Message` out of the broadcast slot, the channel filter, the payload
//!   conversion and parse, the router lookup, and `SessionFanout::send`,
//!   including the wake of every receiver parked on the session.
//! - **`session_deleted_relay/{N}_per_round`**: the session-deleted relay's
//!   turn over the same presence traffic, which its channel filter drops
//!   (`apply_session_deleted_message_for_bench`): the receive and the filter,
//!   the whole per-message cost presence traffic puts on that relay.
//!
//! `N` = 1 is one message per turn onto parked receivers, the wake path at its
//! dearest. `N` = `RELAY_DRAIN`, 4096, the pub/sub client's channel
//! capacity (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY`), is one
//! full drain: the relay's yield bound, so the longest a relay turn holds a
//! worker before it yields.
//!
//! Each round fills a 4096-slot broadcast channel with `N` messages (the
//! pub/sub payloads of the event mix above, on the session's
//! `le:presence:events:` channel) outside the measurement, then runs one
//! relay turn — an awaited `recv` and the synchronous `drain` — in a task on
//! the bench runtime, as the relay runs in the service. The turn is timed on
//! the **thread CPU clock** of the worker running it
//! (`CLOCK_THREAD_CPUTIME_ID`). The channel is filled before the turn starts,
//! so its `recv` is ready at once, and the drain is synchronous, so the turn
//! never yields part way: the receivers it wakes run on other workers or on
//! this one after the turn ends, and are not charged to it either way. The
//! bench asserts that the turn ends on the thread it started on.
//!
//! Criterion's element rate inverts to CPU per relayed message. The longest
//! wall-clock turn of each id is printed after it as `longest turn held its
//! worker for …`: for `N` = 4096 that is the worst-case time one full drain
//! held a worker in the run. A round then waits, outside the measurement,
//! until every receiver has read the round's events.
//!
//! Neither the relay nor any receiver may lag: a round of at most the
//! channel's and the fan-out's capacity, drained fully before the next,
//! cannot overflow either, and the bench panics on any lag. A startup
//! self-check drains a channel filled past its capacity and asserts the drain
//! *does* report the lag, then checks that the presence step delivers a
//! message to a receiver and that the session-deleted step applies a notice on
//! its channel and drops one on another, so neither group times a step that
//! does nothing.

use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dashmap::DashMap;
use live_edit::presence::{
    ChannelJoin, ConnectedUser, LocalPresenceRouter, PresenceEvent, PresenceRecvError,
    DEFAULT_BROADCAST_CHANNEL_CAPACITY,
};
use tokio::runtime::Runtime;
use tokio::sync::{broadcast, Notify};
use tokio::task::JoinHandle;

#[cfg(feature = "test-support")]
use live_edit::presence::PubsubRelayReceiver;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Worker threads of the bench runtime.
const WORKERS: usize = 4;
/// Events per round, one regime each: lockstep and saturated drain.
const ROUND_SIZES: [usize; 2] = [1, 2048];
/// The largest round size, and the length of the event cycle.
const MAX_ROUND: usize = 2048;
/// Receiver counts: the 400-receiver session under a heavy message load and
/// the 5-receiver editing session.
const RECEIVER_COUNTS: [usize; 2] = [400, 5];
/// `PeerMessage` payload bytes, as in the 400-receiver load.
const MESSAGE_BYTES: usize = 200;
/// The session every arm uses.
const SID: &str = "bench-session";

// A round drains fully before the next, so a round no larger than half the
// capacity can never overflow the queue: no receiver can lag.
const _: () = assert!(2 * MAX_ROUND <= DEFAULT_BROADCAST_CHANNEL_CAPACITY);
const _: () = assert!(ROUND_SIZES[0] <= MAX_ROUND && ROUND_SIZES[1] <= MAX_ROUND);

/// The self-check's capacity and round size: a round four times the
/// capacity, sent before any receiver runs, so every receiver lags.
const SELF_CHECK_CAPACITY: usize = 16;
const SELF_CHECK_ROUND: usize = 64;

/// The clock rounds are measured on: process CPU time on unix.
#[cfg(unix)]
fn clock() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec` for the call's duration,
    // and CLOCK_PROCESS_CPUTIME_ID is supported on every unix target built.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    Duration::new(
        u64::try_from(ts.tv_sec).expect("non-negative seconds"),
        u32::try_from(ts.tv_nsec).expect("nanoseconds below one second"),
    )
}

/// The clock rounds are measured on: wall time where no process CPU clock
/// is wired (see the module docs).
#[cfg(not(unix))]
fn clock() -> Duration {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed()
}

/// One cycle of the event mix (see the module docs).
fn event_cycle() -> Vec<PresenceEvent> {
    (0..MAX_ROUND)
        .map(|i| match i % 26 {
            12 => PresenceEvent::PeerJoined(ConnectedUser {
                connection_id: format!("churner-{}", i % 8),
                user_name: "bench-user".to_string(),
                user_application: "bench-app".to_string(),
            }),
            25 => PresenceEvent::PeerLeft {
                connection_id: format!("churner-{}", i % 8),
            },
            _ => PresenceEvent::PeerMessage {
                connection_id: format!("member-{}", i % 16),
                data: vec![b'x'; MESSAGE_BYTES],
            },
        })
        .collect()
}

/// Round bookkeeping shared by the sender and the receivers.
struct Rounds {
    /// Receivers that have accounted for every event of the current round.
    done: AtomicUsize,
    /// Woken by the last of them.
    all_done: Notify,
    /// Events lost to lag, summed over every receiver.
    lagged: AtomicU64,
    receivers: usize,
    /// Events per round.
    round_size: usize,
}

impl Rounds {
    fn new(receivers: usize, round_size: usize) -> Arc<Self> {
        Arc::new(Self {
            done: AtomicUsize::new(0),
            all_done: Notify::new(),
            lagged: AtomicU64::new(0),
            receivers,
            round_size,
        })
    }

    /// A receiver has accounted for `n` more events (read or lost); `seen` is
    /// its running total. Ends its round each time the total reaches a
    /// multiple of the round size.
    fn account(&self, seen: &mut usize, n: usize) {
        let before = *seen / self.round_size;
        *seen += n;
        for _ in before..*seen / self.round_size {
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == self.receivers {
                self.all_done.notify_one();
            }
        }
    }

    /// Wait for every receiver to finish the round, then reset for the next.
    /// The next round's events are sent only after the reset, so no receiver
    /// counts towards a round twice.
    async fn wait_all(&self) {
        self.all_done.notified().await;
        self.done.store(0, Ordering::Release);
    }

    fn lagged(&self) -> u64 {
        self.lagged.load(Ordering::Relaxed)
    }
}

/// The arm under test.
enum Arm {
    Fanout(LocalPresenceRouter),
    TokioBroadcast(DashMap<String, Arc<OldEntry>>),
}

/// The replaced router entry: one broadcast sender per session.
struct OldEntry {
    sender: broadcast::Sender<PresenceEvent>,
}

impl Arm {
    /// Broadcast to the session's receivers and return their count (0 with
    /// none); for the old arm, as the old `LocalPresenceRouter::broadcast`
    /// did.
    fn broadcast(&self, event: PresenceEvent) -> usize {
        match self {
            Arm::Fanout(router) => router.broadcast(SID, event),
            Arm::TokioBroadcast(sessions) => sessions
                .get(SID)
                .map_or(0, |entry| entry.value().sender.send(event).unwrap_or(0)),
        }
    }
}

/// One arm with its receiver tasks running.
struct Harness {
    arm: Arc<Arm>,
    rounds: Arc<Rounds>,
    tasks: Vec<JoinHandle<()>>,
    cycle: Vec<PresenceEvent>,
    /// Index into `cycle` of the next event to send.
    next: Cell<usize>,
}

/// Builds one arm's [`Harness`]: runtime, receivers, round size, capacity.
type Build = fn(&Runtime, usize, usize, usize) -> Harness;

impl Harness {
    fn fanout(rt: &Runtime, receivers: usize, round_size: usize, capacity: usize) -> Self {
        let router = LocalPresenceRouter::with_capacity(capacity);
        let rounds = Rounds::new(receivers, round_size);
        let mut tasks = Vec::with_capacity(receivers);
        for i in 0..receivers {
            let cid = format!("member-{i}");
            let (mut rx, join) = router.join_channel(SID, Some(&cid));
            if let ChannelJoin::Subscribe(subscribe) = join {
                subscribe.succeeded();
            }
            let rounds = Arc::clone(&rounds);
            tasks.push(rt.spawn(async move {
                let mut seen = 0;
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            black_box(&*event);
                            rounds.account(&mut seen, 1);
                        }
                        Err(PresenceRecvError::Lagged { dropped, .. }) => {
                            rounds.lagged.fetch_add(dropped, Ordering::Relaxed);
                            rounds.account(&mut seen, dropped as usize);
                        }
                        Err(PresenceRecvError::Closed) => return,
                    }
                }
            }));
        }
        Self::new(Arm::Fanout(router), rounds, tasks)
    }

    fn tokio_broadcast(rt: &Runtime, receivers: usize, round_size: usize, capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        let sessions = DashMap::new();
        sessions.insert(SID.to_string(), Arc::new(OldEntry { sender }));
        let rounds = Rounds::new(receivers, round_size);
        let mut tasks = Vec::with_capacity(receivers);
        for _ in 0..receivers {
            let mut rx = sessions
                .get(SID)
                .expect("session entry")
                .value()
                .sender
                .subscribe();
            let rounds = Arc::clone(&rounds);
            tasks.push(rt.spawn(async move {
                let mut seen = 0;
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            black_box(&event);
                            rounds.account(&mut seen, 1);
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            rounds.lagged.fetch_add(n, Ordering::Relaxed);
                            rounds.account(&mut seen, n as usize);
                        }
                        Err(broadcast::error::RecvError::Closed) => return,
                    }
                }
            }));
        }
        Self::new(Arm::TokioBroadcast(sessions), rounds, tasks)
    }

    fn new(arm: Arm, rounds: Arc<Rounds>, tasks: Vec<JoinHandle<()>>) -> Self {
        Self {
            arm: Arc::new(arm),
            rounds,
            tasks,
            cycle: event_cycle(),
            next: Cell::new(0),
        }
    }

    /// One round: publish the round's events unpaced, taken in turn from the
    /// event cycle, then wait until every receiver has accounted for them.
    async fn round(&self) {
        for _ in 0..self.rounds.round_size {
            let i = self.next.get();
            self.next.set((i + 1) % self.cycle.len());
            let queued = self.arm.broadcast(self.cycle[i].clone());
            debug_assert_eq!(queued, self.rounds.receivers);
        }
        self.rounds.wait_all().await;
    }

    /// Close the session so every receiver ends `Closed`, wait for the
    /// tasks, and return the events lost to lag.
    fn finish(self, rt: &Runtime) -> u64 {
        match &*self.arm {
            Arm::Fanout(router) => {
                for _ in 0..self.rounds.receivers {
                    if let Some(unsubscribe) = router.leave_channel(SID) {
                        unsubscribe.finish();
                    }
                }
            }
            Arm::TokioBroadcast(sessions) => {
                sessions.remove(SID);
            }
        }
        rt.block_on(async {
            for task in self.tasks {
                task.await.expect("receiver task");
            }
        });
        self.rounds.lagged()
    }
}

/// The arms, by benchmark id.
const ARMS: [(&str, Build); 2] = [
    ("fanout", Harness::fanout),
    ("tokio_broadcast", Harness::tokio_broadcast),
];

/// Prove the zero-lag check can fail: on a current-thread runtime the sender
/// queues a whole round, four times the capacity, before any receiver runs,
/// so every receiver must report lag.
fn lag_self_check() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    for (name, build) in ARMS {
        let harness = build(&rt, 5, SELF_CHECK_ROUND, SELF_CHECK_CAPACITY);
        rt.block_on(harness.round());
        let lagged = harness.finish(&rt);
        assert!(
            lagged > 0,
            "self-check: {name} with capacity {SELF_CHECK_CAPACITY} and a round of \
             {SELF_CHECK_ROUND} must count lag, but counted none"
        );
        eprintln!("lag self-check: {name} counted {lagged} lagged events");
    }
}

fn presence_fanout(c: &mut Criterion) {
    lag_self_check();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKERS)
        .enable_all()
        .build()
        .expect("tokio runtime");
    for receivers in RECEIVER_COUNTS {
        for round_size in ROUND_SIZES {
            let mut group = c.benchmark_group(format!(
                "presence_fanout/{receivers}_receivers/{round_size}_per_round"
            ));
            group.throughput(Throughput::Elements((round_size * receivers) as u64));
            for (name, build) in ARMS {
                let harness = build(
                    &rt,
                    receivers,
                    round_size,
                    DEFAULT_BROADCAST_CHANNEL_CAPACITY,
                );
                group.bench_function(BenchmarkId::from_parameter(name), |b| {
                    b.iter_custom(|iters| {
                        rt.block_on(async {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let start = clock();
                                harness.round().await;
                                total += clock() - start;
                                // Stop before criterion prints a figure
                                // measured on the lag path.
                                let lagged = harness.rounds.lagged();
                                assert_eq!(
                                    lagged, 0,
                                    "{name}: a receiver lagged by {lagged} events"
                                );
                            }
                            total
                        })
                    });
                });
                let lagged = harness.finish(&rt);
                assert_eq!(lagged, 0, "{name}: a receiver lagged by {lagged} events");
            }
            group.finish();
        }
    }
}

/// The replica-wide relay groups (see the module docs).
#[cfg(feature = "test-support")]
mod relay {
    use std::time::Instant;

    use dashmap::DashMap;
    use fred::types::config::Server;
    use fred::types::{Message, MessageKind, Value};
    use live_edit::grpc::session::{apply_session_deleted_message_for_bench, SessionCancel};
    use live_edit::grpc::session_cache::SessionBasePathCache;
    use live_edit::presence::relay_presence_message_for_bench;
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// Receivers on the session: one, a small session, a keeper-load session
    /// (30,000 connections in sessions of 200) and the 400-receiver session.
    const RELAY_RECEIVER_COUNTS: [usize; 4] = [1, 50, 200, 400];
    /// The pub/sub client's message channel capacity
    /// (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY`, crate-private),
    /// which is also the relay's yield bound: one full drain.
    pub(super) const RELAY_DRAIN: usize = 4096;
    /// Messages per relay turn: one, and one full drain.
    const RELAY_ROUND_SIZES: [usize; 2] = [1, RELAY_DRAIN];
    /// The service's presence-events channel prefix (`KeySchema::default()`).
    const PRESENCE_PREFIX: &str = "le:presence:events:";
    /// The service's session-deleted channel (`KeySchema::default()`).
    const SESSION_DELETED_CHANNEL: &str = "le:session:deleted";
    /// How long a round may wait for its receivers before the bench fails
    /// rather than hangs (a step that delivered nothing).
    const ROUND_DEADLINE: Duration = Duration::from_secs(30);

    // A full drain fits the per-session fan-out queue, so no receiver can lag.
    const _: () = assert!(RELAY_DRAIN <= DEFAULT_BROADCAST_CHANNEL_CAPACITY);

    /// The clock a relay turn is measured on: the CPU time of the thread
    /// running it, on unix.
    #[cfg(unix)]
    fn thread_clock() -> Duration {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid, writable `timespec` for the call's
        // duration, and CLOCK_THREAD_CPUTIME_ID is supported on every unix
        // target built.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
        Duration::new(
            u64::try_from(ts.tv_sec).expect("non-negative seconds"),
            u32::try_from(ts.tv_nsec).expect("nanoseconds below one second"),
        )
    }

    /// The clock a relay turn is measured on: wall time where no thread CPU
    /// clock is wired.
    #[cfg(not(unix))]
    fn thread_clock() -> Duration {
        super::clock()
    }

    /// The pub/sub message a presence event arrives as, on the bench
    /// session's channel.
    fn presence_message(event: &PresenceEvent) -> Message {
        Message {
            channel: format!("{PRESENCE_PREFIX}{SID}").into(),
            value: Value::String(event.encode().into()),
            kind: MessageKind::Message,
            server: Server::new("bench", 6379),
        }
    }

    /// One full drain's worth of pub/sub messages, the event mix in turn.
    fn message_cycle() -> Vec<Message> {
        let events = event_cycle();
        (0..RELAY_DRAIN)
            .map(|i| presence_message(&events[i % events.len()]))
            .collect()
    }

    /// A relay receiver on a fresh channel of the pub/sub client's capacity.
    fn relay_channel(capacity: usize) -> (broadcast::Sender<Message>, PubsubRelayReceiver) {
        let (tx, rx) = broadcast::channel(capacity);
        (tx, PubsubRelayReceiver::from_receiver(rx, capacity))
    }

    /// One relay turn as the service runs it: an awaited `recv`, then the
    /// synchronous drain, in a task on `rt`. Returns the receiver, the turn's
    /// thread CPU time and wall time, and the drain's lag.
    fn relay_turn(
        rt: &Runtime,
        mut relay: PubsubRelayReceiver,
        mut step: impl FnMut(Message) + Send + 'static,
    ) -> (PubsubRelayReceiver, Duration, Duration, u64) {
        rt.block_on(async {
            rt.spawn(async move {
                let thread = std::thread::current().id();
                let cpu = thread_clock();
                let wall = Instant::now();
                let first = relay.recv().await;
                let batch = relay.drain(first, &mut step);
                let (cpu, wall) = (thread_clock() - cpu, wall.elapsed());
                assert_eq!(
                    std::thread::current().id(),
                    thread,
                    "the relay turn moved between workers, so its thread CPU time is not the \
                     turn's: its recv must be ready at once (fill the channel first)"
                );
                assert_ne!(
                    batch.end,
                    live_edit::presence::DrainEnd::Closed,
                    "the bench channel closed mid-run"
                );
                (relay, cpu, wall, batch.lagged)
            })
            .await
            .expect("relay turn task")
        })
    }

    /// Prove the checks can fail: a drain of a channel filled past its
    /// capacity reports the lag; the presence step delivers to a receiver;
    /// the session-deleted step applies a notice on its channel and drops one
    /// on another.
    fn relay_self_check() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let cycle = message_cycle();

        let (tx, relay) = relay_channel(SELF_CHECK_CAPACITY);
        for msg in cycle.iter().take(SELF_CHECK_ROUND) {
            tx.send(msg.clone()).expect("a receiver is open");
        }
        let (_, _, _, lagged) = relay_turn(&rt, relay, |msg| {
            black_box(msg);
        });
        assert!(
            lagged > 0,
            "self-check: a drain of {SELF_CHECK_ROUND} messages through a channel of \
             {SELF_CHECK_CAPACITY} must report lag, but reported none"
        );
        eprintln!("relay self-check: the drain reported {lagged} lagged messages");

        let router = LocalPresenceRouter::new();
        let (mut rx, join) = router.join_channel(SID, Some("member-0"));
        if let ChannelJoin::Subscribe(subscribe) = join {
            subscribe.succeeded();
        }
        relay_presence_message_for_bench(&router, PRESENCE_PREFIX, cycle[0].clone());
        let delivered = rt
            .block_on(async { tokio::time::timeout(ROUND_DEADLINE, rx.recv()).await })
            .expect("self-check: the presence step must deliver the event")
            .expect("self-check: the receiver must read the event, not lag");
        black_box(&*delivered);

        let tokens: DashMap<String, SessionCancel> = DashMap::new();
        let cache = SessionBasePathCache::new();
        let token = CancellationToken::new();
        tokens.insert("deleted-session".to_string(), token.clone().into());
        apply_session_deleted_message_for_bench(
            &tokens,
            &cache,
            SESSION_DELETED_CHANNEL,
            cycle[0].clone(),
        );
        assert!(
            !token.is_cancelled() && tokens.len() == 1,
            "self-check: the session-deleted step must drop a presence message"
        );
        let notice = Message {
            channel: SESSION_DELETED_CHANNEL.into(),
            value: Value::String("deleted-session".into()),
            kind: MessageKind::Message,
            server: Server::new("bench", 6379),
        };
        apply_session_deleted_message_for_bench(&tokens, &cache, SESSION_DELETED_CHANNEL, notice);
        assert!(
            token.is_cancelled() && tokens.is_empty(),
            "self-check: the session-deleted step must apply a notice on its channel"
        );
        eprintln!("relay self-check: both relay steps act on their own messages");
    }

    /// Fill the channel with the round's messages, outside the measurement.
    fn fill(tx: &broadcast::Sender<Message>, cycle: &[Message], next: &mut usize, n: usize) {
        for _ in 0..n {
            tx.send(cycle[*next].clone()).expect("the relay is open");
            *next = (*next + 1) % cycle.len();
        }
    }

    /// Print the longest wall-clock turn an id measured; nothing for an id
    /// criterion's filter skipped.
    fn report_longest(id: &str, longest: Duration) {
        if longest > Duration::ZERO {
            eprintln!("{id}: longest turn held its worker for {longest:?} (wall clock)");
        }
    }

    pub(super) fn presence_relay(c: &mut Criterion) {
        relay_self_check();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(WORKERS)
            .enable_all()
            .build()
            .expect("tokio runtime");
        let cycle = message_cycle();
        for receivers in RELAY_RECEIVER_COUNTS {
            for round_size in RELAY_ROUND_SIZES {
                let id = format!("presence_relay/{receivers}_receivers/{round_size}_per_round");
                let mut group = c.benchmark_group(&id);
                group.throughput(Throughput::Elements(round_size as u64));
                let harness = Harness::fanout(
                    &rt,
                    receivers,
                    round_size,
                    DEFAULT_BROADCAST_CHANNEL_CAPACITY,
                );
                let arm = Arc::clone(&harness.arm);
                let (tx, relay) = relay_channel(RELAY_DRAIN);
                let mut relay = Some(relay);
                let mut next = 0;
                let mut longest = Duration::ZERO;
                group.bench_function(BenchmarkId::from_parameter("relay_turn"), |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            fill(&tx, &cycle, &mut next, round_size);
                            let arm = Arc::clone(&arm);
                            let (back, cpu, wall, lagged) = relay_turn(
                                &rt,
                                relay.take().expect("relay receiver"),
                                move |msg| {
                                    let Arm::Fanout(router) = &*arm else {
                                        unreachable!("the relay feeds the fan-out arm");
                                    };
                                    relay_presence_message_for_bench(router, PRESENCE_PREFIX, msg);
                                },
                            );
                            relay = Some(back);
                            assert_eq!(lagged, 0, "the presence relay lagged by {lagged}");
                            total += cpu;
                            longest = longest.max(wall);
                            rt.block_on(async {
                                tokio::time::timeout(ROUND_DEADLINE, harness.rounds.wait_all())
                                    .await
                                    .expect("every receiver must read the round's events");
                            });
                            let lagged = harness.rounds.lagged();
                            assert_eq!(lagged, 0, "a receiver lagged by {lagged} events");
                        }
                        total
                    });
                });
                group.finish();
                report_longest(&id, longest);
                drop(arm);
                let lagged = harness.finish(&rt);
                assert_eq!(lagged, 0, "a receiver lagged by {lagged} events");
            }
        }
    }

    pub(super) fn session_deleted_relay(c: &mut Criterion) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(WORKERS)
            .enable_all()
            .build()
            .expect("tokio runtime");
        let cycle = message_cycle();
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
        let cache = SessionBasePathCache::new();
        for round_size in RELAY_ROUND_SIZES {
            let id = format!("session_deleted_relay/{round_size}_per_round");
            let mut group = c.benchmark_group(&id);
            group.throughput(Throughput::Elements(round_size as u64));
            let (tx, relay) = relay_channel(RELAY_DRAIN);
            let mut relay = Some(relay);
            let mut next = 0;
            let mut longest = Duration::ZERO;
            group.bench_function(BenchmarkId::from_parameter("relay_turn"), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        fill(&tx, &cycle, &mut next, round_size);
                        let tokens = Arc::clone(&tokens);
                        let cache = cache.clone();
                        let (back, cpu, wall, lagged) =
                            relay_turn(&rt, relay.take().expect("relay receiver"), move |msg| {
                                apply_session_deleted_message_for_bench(
                                    &tokens,
                                    &cache,
                                    SESSION_DELETED_CHANNEL,
                                    msg,
                                );
                            });
                        relay = Some(back);
                        assert_eq!(lagged, 0, "the session-deleted relay lagged by {lagged}");
                        total += cpu;
                        longest = longest.max(wall);
                    }
                    total
                });
            });
            group.finish();
            report_longest(&id, longest);
        }
    }
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20))
        .sample_size(30)
}

#[cfg(not(feature = "test-support"))]
criterion_group! {
    name = benches;
    config = config();
    targets = presence_fanout
}
#[cfg(feature = "test-support")]
criterion_group! {
    name = benches;
    config = config();
    targets = presence_fanout, relay::presence_relay, relay::session_deleted_relay
}
criterion_main!(benches);
