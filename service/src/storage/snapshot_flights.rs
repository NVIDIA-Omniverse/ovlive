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

//! Shared snapshot reads: the per-replica [`SnapshotFlights`] registry (Spec
//! SDD SR-6), with its fenced arm.
//!
//! Catch-ups of one layer above the atomic-read ceiling on one replica share
//! one fenced read ([`read_object_snapshot_fenced`]) instead of each taking
//! the fence: they attach to the object's **flight** ([`FlightSlot`]), which
//! reads once and hands every attached catch-up (a **waiter**,
//! [`FlightWaiter`]) the same immutable `Arc<AtomicSnapshot>`. Each waiter
//! then builds its own payload from it, in a **build turn** the flight serves.
//!
//! # A flight is one unit of catch-up work
//!
//! It holds one catch-up slot (`CatchUpGate`) from its read through its last
//! waiter's build, and one combined memory grant (`MemoryBudget`: the snapshot
//! plus one build) from its read until the last holder of the snapshot drops
//! it. Attached waiters hold no slot and no memory of their own.
//!
//! - **The read stage** ([`FlightSlot`]'s `Shared<BoxFuture>`, driven by
//!   every waiter that waits on it) waits for its predecessor flight's `done`
//!   (below), takes the flight's slot — once, kept through every fence sleep
//!   — and runs the fenced read under a [`FencedReadHook`] that implements
//!   Spec SDD SR-5.6's acquire cycle: per acquire attempt the fenced read
//!   permit (`CatchUpReadGate`, class `fenced`), then the combined
//!   reservation ([`flight_grant_bytes`]); on `RETRY_AFTER` it gives back the
//!   reservation, its top-up and the permit, sleeps the hint and takes them
//!   again. The reservation is sized from an estimate — the fitted bytes per
//!   item, or the fence key's stored size while that key lingers — which a
//!   byte-dense layer (large inline arrays, posed or recorded rigs) far
//!   outweighs, so **as the read goes** the hook tops it up: after the
//!   fan-out's first round and after each later window
//!   ([`FencedReadHook::on_read_progress`]), never waiting, to
//!   `SNAPSHOT_MEMORY_FACTOR` × the reply bytes read so far (the snapshot;
//!   no build runs during the read), clamped so that the grant and its one
//!   top-up together never exceed `T` (`MemoryGrant::top_up_to`). It holds
//!   the fenced permit throughout, so the top-up is inside the over-commit
//!   `T × (1 + P_f)` funds, and every read of a layer whose weight is within
//!   `T` is accounted while it reads, the first included. A broken attempt
//!   keeps the reservation and its top-up for the retry. After a verified
//!   read it sets the grant and its top-up to the measured weight
//!   ([`flight_weight_bytes`]; `MemoryGrant::adjust_with_top_up`: the grant
//!   shrunk and the top-up dropped, or the one top-up resized to the
//!   difference, never waiting) **before** giving the permit back, and hands
//!   the grant and its top-up to the snapshot (`AtomicSnapshot::memory`). It
//!   runs under `catch_unwind`: a panic is an `Internal` failure for every
//!   waiter, a panic in the read's fan-out pass included (the pass runs on
//!   the `fanout-read` thread, `storage::fanout_thread`, and its panic is
//!   re-raised in the read stage). The hook, the permit and the grant stay
//!   with the read stage: the pass only reports its progress back, and
//!   issues its next window once the hook has applied the report.
//! - **Build turns.** After the read the flight's slot passes to its turn
//!   server, which hands the flight's one build turn to its waiters one at a
//!   time, in the order they asked ([`FlightRead::build_turn`]). When a
//!   waiter would queue, and one more catch-up slot (`CatchUpGate::try_acquire`)
//!   and one more build part (`MemoryBudget::try_grant`, sized
//!   [`extra_build_part_bytes`]) are free at once, the head of the queue builds
//!   at once under them instead (an **extra build**); its slot and part are
//!   its turn guard. No build waits for a slot or memory while holding the
//!   snapshot, so there is no slot/memory wait cycle.
//! - A waiter gives back its turn, its snapshot handle and its place in the
//!   flight together, when its [`BuildTurn`] drops — before it sends anything,
//!   so neither the flight's slot nor its grant ever waits for a send. The
//!   flight's slot is released when its last waiter leaves, and its grant
//!   when the last holder of the snapshot drops it.
//!
//! Nesting (Spec SDD SR-7.5): a fenced permit is taken only while holding the
//! flight's slot, memory only while holding the permit, and nothing waits for
//! a slot or a permit while holding memory.
//!
//! # Attach, flip and close, under the object's entry lock
//!
//! Every map operation for one object runs under its `DashMap` entry lock
//! (Spec SDD SR-6.2): attach, creating a successor, the `Queued → Reading`
//! flip, closing to attach, and removal.
//!
//! - A flight is attachable (`open`) while [`FlightState::Queued`] — through
//!   its predecessor wait, its capacity waits and every fence sleep — and,
//!   once acquire granted the fence, while [`FlightState::Reading`] until the
//!   release script is sent: no writer commits during the epoch, so a pin read
//!   then equals the cut. The flip happens on the grant, before the fan-out's
//!   first command; attach closes before the release is sent, since a pin
//!   read after it could exceed the cut. A broken read reopens the flight,
//!   `Queued`, for its retry.
//! - A newcomer finding the object's flight closed creates its **successor**,
//!   which takes the predecessor's place in the map, collects the next wave
//!   while `Queued`, and waits for the predecessor's `done` before taking any
//!   slot, permit or memory. **The whole flight is the predecessor:** `done`
//!   fires from a drop guard owned by the turn server, so it fires after the
//!   flight's last build — or on a failure, a panic, or when every waiter
//!   left.
//! - A flight leaves the map when its last waiter leaves (its last build
//!   ended, or every waiter was cancelled) or when its read fails, each time
//!   only if the map still holds this flight (`Arc::ptr_eq`): a predecessor
//!   never removes its successor. A failed flight is never reused.
//! - **A displaced predecessor goes back.** Each flight keeps a weak link to
//!   the flight it succeeded. When a flight leaves the map, its entry goes back
//!   to its nearest predecessor that is still running (whose `done` has not
//!   fired), found by walking those links; only when none is left does the
//!   entry go empty. So a successor whose every waiter left — say all timed
//!   out while the predecessor's build chain ran — does not leave the entry
//!   empty while that predecessor still builds: the closed predecessor is
//!   registered again, [`SnapshotFlights::lookup`] finds it, and the next
//!   catch-up becomes its successor and waits for its `done`, never reading
//!   beside it. Each flight waits only on its own predecessor's `done`. A
//!   flight is displaced only once it is closed, which it is only after its
//!   own predecessor wait ended, so a running predecessor's predecessor is
//!   always done and the walk stops at the first link in practice; it walks
//!   on past finished links regardless, and never puts a finished flight
//!   back.
//! - **The reopen window.** A catch-up arriving after a flight closed for its
//!   release but before a `BROKEN` answer reopened it creates a successor that
//!   takes the flight's place. The reopened flight is then unreachable: no
//!   later catch-up attaches to its retry, and the successor's wave waits for
//!   it and pays a read of its own. It becomes reachable again only if that
//!   successor's waiters all leave (it goes back, as above, still open). The
//!   window is one release round trip followed by a broken read.
//!
//! # Per-waiter charging and the reader cap
//!
//! The flight's own capacity waits carry no deadline ([`no_deadline`] for the
//! two gates, none for the budget): they serve every waiter and end only when
//! the last one leaves. The flight publishes its [`FlightPhase`] on a `watch`
//! channel, and each waiter, in [`FlightWaiter::read`], is charged on its own
//! `CapacityBudget` for the time it is attached during `Capacity` phases, and
//! counts `Work` time (fence sleeps, acquires, the read) against its own
//! `reader_wait_max(L)`. When either runs out it leaves alone, refused, and
//! the flight continues for the others. The `Predecessor` phase is neither
//! charged nor capped, nor is the wait for a build turn.
//!
//! # Placement
//!
//! In `storage`: everything a flight touches — the client, the gates, the
//! budget, the fenced read, `build_diff_payload` for Epic 4's shared build —
//! is storage's, and the module knows nothing of catch-up regimes, statuses or
//! streams. The caller (`grpc::subscribe`) maps a [`FlightWaitError`] to its
//! ending and races every wait here against its own endings (shutdown,
//! session cancellation, client disconnect); dropping a waiter at any point
//! leaves the flight cleanly.
//!
//! # Test seam
//!
//! `FlightLeaveSeam` (`test-fault-injection` and `cfg(test)` builds only;
//! compiled out of release builds) parks one waiter's leaving inside
//! `Registry::leave`, after its count-out and with the object's entry lock
//! released, before `leave` returns to [`FlightWaiter`]'s drop. Armed per
//! object with `SnapshotFlights::arm_leave_seam_for_test`. Because it sits
//! inside `leave`, not in the drop, it stays between a waiter's count-out and
//! whatever that drop does after `leave` returns, whatever order the drop
//! uses — so a test can hold one waiter there while another counts out last
//! and fires `done`. Used by the post-completion review F4 (i) regression in
//! `tests/integration/storage/snapshot_flights_test.rs`
//! (`flight_cancelled_by_two_waiters_at_once_frees_slot_permit_and_grant_before_its_successor_wakes`).

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError, Weak};
use std::time::Duration;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use futures::future::{BoxFuture, FutureExt, Shared};
use tokio::sync::{oneshot, watch};
use tokio::time::Instant;

use crate::error::ServiceError;
use crate::observability::metrics::{
    SharedMetrics, CATCHUP_SNAPSHOT_FLIGHT_ROLE_JOINED_READ, CATCHUP_SNAPSHOT_FLIGHT_ROLE_LEADER,
};
use crate::storage::client::ValkeyClient;
use crate::storage::diff_read::{AtomicSnapshot, SnapshotReadConsumer};
use crate::storage::keys::KeySchema;
use crate::storage::memory_budget::{
    MemoryBudget, MemoryGrant, MemoryTopUp, SnapshotMemory, BUILD_MEMORY_FACTOR,
    SNAPSHOT_MEMORY_FACTOR,
};
use crate::storage::snapshot_fence::read::{
    read_object_snapshot_fenced, FenceAttempt, FenceAttemptOutcome, FenceGrant, FenceReleased,
    FenceRetry, FencedReadHook,
};
use crate::storage::snapshot_fence::{coefficients, ms_to_duration, FenceCoefficients, LayerSize};
use crate::storage::subscriber_admission::{
    no_deadline, CapacityBudget, CatchUpGate, CatchUpPermit, CatchUpReadPermit, CatchUpRefusal,
};

/// The `tracing` target of the flight events (`flight_attach`, `flight_read`),
/// at DEBUG.
const TRACE_TARGET: &str = "live_edit::snapshot_flight";

/// The combined memory grant a fenced flight reserves before each fence
/// acquire, for a layer of `b` reply bytes (Spec SDD SR-7.2 step 1, fenced
/// path): the snapshot plus one build, `b` × ([`SNAPSHOT_MEMORY_FACTOR`] +
/// [`BUILD_MEMORY_FACTOR`]), plus one fan-out window's raw replies, at most
/// min(`FANOUT_WINDOW` × the window's largest entry, `b`). The largest entry
/// is unknown before the read, so the window term is its bound, `b`. The
/// budget clamps the total to `T`.
///
/// The window term models a window's memory as its reply bytes. Measured
/// (Task 10 of `docs-internal/validation/fenced-snapshot-reads-20261002`), a
/// window's client memory scales with its commands instead: a full window of
/// the uncapped 7,131 commands peaked at about 50 MiB allocated (115–130 MiB
/// RSS) whatever `b` was, so a small fenced layer's read stage runs over its
/// grant. That transient, `X`, is funded beside the budget rather than
/// reserved here (resident memory ≤ `T × (1 + P_f) + P_f × X`, see
/// `memory_budget`), and the window cap (`config::FANOUT_WINDOW_MAX`) shrinks
/// it about 70 %. Epic 5 changes this term to `FANOUT_WINDOW` × the measured
/// per-command client memory, reserved before the read and dropped at the
/// post-read adjust as `b` is today.
pub fn flight_grant_bytes(b: u64) -> u64 {
    b.saturating_mul(SNAPSHOT_MEMORY_FACTOR + BUILD_MEMORY_FACTOR)
        .saturating_add(b)
}

/// The measured weight a verified flight's grant is adjusted to (Spec SDD
/// SR-7.2 step 1): the snapshot plus one build at the read's `reply_bytes`.
/// The window's raw replies are decoded and gone by then.
pub fn flight_weight_bytes(reply_bytes: u64) -> u64 {
    reply_bytes.saturating_mul(SNAPSHOT_MEMORY_FACTOR + BUILD_MEMORY_FACTOR)
}

/// The build part an extra build takes (Spec SDD SR-6.3): one build at the
/// read's `reply_bytes`, an ordinary non-blocking admission, never topped up.
pub fn extra_build_part_bytes(reply_bytes: u64) -> u64 {
    reply_bytes.saturating_mul(BUILD_MEMORY_FACTOR)
}

/// The per-replica registry of shared snapshot-read flights, keyed by
/// `object_id` (Spec SDD SR-6.1); see the module docs. Cheap to clone; clones
/// share one registry.
#[derive(Clone)]
pub struct SnapshotFlights {
    registry: Arc<Registry>,
}

impl std::fmt::Debug for SnapshotFlights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotFlights")
            .field("flights", &self.registry.slots.len())
            .finish()
    }
}

struct Registry {
    slots: DashMap<u64, Arc<FlightSlot>>,
    client: ValkeyClient,
    keys: KeySchema,
    /// The armed [`FlightLeaveSeam`] and the object it parks a leave of;
    /// taken by the first leave of that object to reach it (one-shot).
    #[cfg(any(test, feature = "test-fault-injection"))]
    leave_seam: StdMutex<Option<(u64, Arc<FlightLeaveSeam>)>>,
}

/// Where a flight's read stands (Spec SDD SR-6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightState {
    /// No cut is fixed: waiting for a predecessor, for capacity, or between
    /// fence acquire attempts; or reopened for the retry of a broken read.
    Queued,
    /// Acquire granted the fence and the cut is fixed; the fenced read runs.
    /// Still attachable until the release script is sent.
    Reading,
}

/// What a flight is doing, as its waiters are charged for it (Spec SDD
/// SR-7.3); published on a `watch` channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightPhase {
    /// Waiting for the predecessor flight's `done`: work, neither charged nor
    /// capped.
    Predecessor,
    /// Waiting for the flight's slot, its fenced read permit or its memory
    /// reservation: charged to every attached waiter's capacity budget.
    Capacity,
    /// Fence acquires, `RETRY_AFTER` sleeps, the fenced read: work, counted
    /// against each waiter's `reader_wait_max(L)`.
    Work,
}

/// A catch-up's part in its flight (`live_edit_catchup_snapshot_flight_total`,
/// Spec SDD SR-8.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightRole {
    /// It created the flight (the first, or a closed flight's successor).
    Leader,
    /// It attached to a flight another catch-up created.
    JoinedRead,
}

impl FlightRole {
    fn label(self) -> &'static str {
        match self {
            FlightRole::Leader => CATCHUP_SNAPSHOT_FLIGHT_ROLE_LEADER,
            FlightRole::JoinedRead => CATCHUP_SNAPSHOT_FLIGHT_ROLE_JOINED_READ,
        }
    }
}

/// What [`SnapshotFlights::lookup`] found for an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlightLookup {
    /// Whether a catch-up attaching now joins this flight's read. `false`: the
    /// flight is closed (its fence released, or its read resolved and its
    /// builds still running), and [`SnapshotFlights::attach_or_create`]
    /// creates its successor.
    pub attachable: bool,
    /// The flight's read state.
    pub state: FlightState,
    /// The best size known for the layer: the estimate the flight was created
    /// with, then what the fence replies named, then — once read — the
    /// measured size. What a catch-up joining or succeeding it sizes from.
    pub size: LayerSize,
}

/// Why [`FlightWaiter::read`] returned without a snapshot. In every case the
/// waiter has left its flight, which carries on for the others.
#[derive(Debug)]
pub enum FlightWaitError {
    /// The waiter's capacity budget ran out while it was attached during the
    /// flight's capacity waits (slot, fenced permit, memory). Retryable, and
    /// before any content.
    CapacityExhausted,
    /// The work the waiter waited for — fence acquires, `RETRY_AFTER` sleeps,
    /// the fenced read, counted from the end of the predecessor wait — passed
    /// `reader_wait_max(L)` (Spec SDD SR-5.5, SR-11.4). Logged at WARN with the
    /// work time; the caller counts the ending (a catch-up's stream exit,
    /// `reason="fenced_read_wait_cap"`). Not observed on
    /// `live_edit_snapshot_fence_reader_wait_seconds`, whose observations are
    /// per read attempt. Retryable, and before any content.
    ReaderWaitCap {
        /// The work time counted.
        waited: Duration,
        /// The cap it passed, for the size the flight last knew.
        cap: Duration,
    },
    /// A gate or the memory budget is closed: the replica is shutting down.
    ShuttingDown,
    /// The shared read failed. The error is this waiter's own copy of the one
    /// the read ended with, its classification kept (Spec SDD SR-6.6):
    /// `NotFound`, transient (`Aborted` / `Unavailable`, `Unavailable` after
    /// three broken attempts) or permanent (`Internal`, a panic included).
    Read(ServiceError),
}

impl SnapshotFlights {
    /// An empty registry reading through `client` and `keys`.
    pub fn new(client: ValkeyClient, keys: KeySchema) -> Self {
        Self {
            registry: Arc::new(Registry {
                slots: DashMap::new(),
                client,
                keys,
                #[cfg(any(test, feature = "test-fault-injection"))]
                leave_seam: StdMutex::new(None),
            }),
        }
    }

    /// The flight registered for `object_id`, if any: attachable, or closed
    /// and still building (the predecessor of whatever catch-up comes next).
    /// Either way the object's catch-ups go through
    /// [`Self::attach_or_create`].
    pub fn lookup(&self, object_id: u64) -> Option<FlightLookup> {
        let entry = self.registry.slots.get(&object_id)?;
        let inner = entry.lock();
        Some(FlightLookup {
            attachable: inner.open,
            state: inner.state,
            size: inner.size,
        })
    }

    /// Flights currently registered (one per object at most).
    pub fn len(&self) -> usize {
        self.registry.slots.len()
    }

    /// Whether no flight is registered.
    pub fn is_empty(&self) -> bool {
        self.registry.slots.is_empty()
    }

    /// Test-only: arm a one-shot [`FlightLeaveSeam`] for `object_id`. The
    /// next waiter of any flight of that object to leave parks inside
    /// `Registry::leave` (module docs, # Test seam) until the returned seam
    /// is released; later leaves pass through. Replaces a seam still armed.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn arm_leave_seam_for_test(&self, object_id: u64) -> Arc<FlightLeaveSeam> {
        let seam = Arc::new(FlightLeaveSeam::default());
        *self
            .registry
            .leave_seam
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some((object_id, Arc::clone(&seam)));
        seam
    }

    /// Attach to `object_id`'s flight, or create one, under the object's
    /// entry lock: attach when the registered flight is open; otherwise create
    /// a flight — the successor of a closed one, which waits for its `done` —
    /// sized from `size_estimate` (Spec SDD SR-11.1; the fence key's stored
    /// size wins once a fence reply names it). Counts
    /// `live_edit_catchup_snapshot_flight_total{role}`.
    ///
    /// The caller must already be admitted and authorized, and must hold no
    /// catch-up slot or read permit: the flight's slot is the unit of work.
    /// It must also have read its pin — the header sequence its `cut ≥ pin`
    /// check uses — **before** calling this. A catch-up may attach between the
    /// fence's grant and its release, and the share is sound only because its
    /// pin was read before that release, when no commit could land, so the pin
    /// is at most the cut; a pin read after attaching could postdate the
    /// release, exceed the cut, and refuse a sound read as a regression. In
    /// general, anything that must precede every commit after the cut — its
    /// pin read, a GC read hold — a waiter does before calling this: attach
    /// closes before the fence release, so that suffices even when the
    /// flight's cut is already fixed. A catch-up re-attaching after an
    /// external-value miss has its `{gc}` read hold acknowledged first, so
    /// every commit after the cut lands after the hold began and the GC
    /// collects none of the blobs the cut references. A waiter re-attaching
    /// must also have left any earlier flight of the object: a successor
    /// waits for its predecessor's `done`, which the waiter's own place would
    /// hold off. Nothing is waited for here; [`FlightWaiter::read`] waits.
    pub fn attach_or_create(&self, object_id: u64, size_estimate: LayerSize) -> FlightWaiter {
        let registry = &self.registry;
        let (flight, read_stage, role, predecessor) = match registry.slots.entry(object_id) {
            Entry::Occupied(mut occupied) => {
                let current = Arc::clone(occupied.get());
                let attached = {
                    let mut inner = current.lock();
                    match inner.read_stage.clone() {
                        Some(stage) if inner.open => {
                            inner.waiters += 1;
                            Some(stage)
                        }
                        _ => None,
                    }
                };
                match attached {
                    Some(stage) => (current, stage, FlightRole::JoinedRead, None),
                    None => {
                        let (successor, stage) =
                            registry.new_flight(object_id, size_estimate, Some(&current));
                        occupied.insert(Arc::clone(&successor));
                        // `current` lives on through its waiters; it is
                        // dropped here only after the entry lock is released.
                        (successor, stage, FlightRole::Leader, Some(current))
                    }
                }
            }
            Entry::Vacant(vacant) => {
                let (flight, stage) = registry.new_flight(object_id, size_estimate, None);
                vacant.insert(Arc::clone(&flight));
                (flight, stage, FlightRole::Leader, None)
            }
        };
        drop(predecessor);
        if let Some(m) = &flight.metrics {
            m.inc_catchup_snapshot_flight(role.label());
        }
        tracing::debug!(
            target: TRACE_TARGET,
            object_id,
            role = role.label(),
            "flight_attach"
        );
        FlightWaiter {
            read_stage: Some(read_stage),
            flight,
            registry: Arc::clone(registry),
            role,
        }
    }
}

impl Registry {
    /// A new flight for `object_id`, counting its creator as its one waiter,
    /// with its read stage built (not yet polled): the successor of
    /// `predecessor` when given, waiting for its `done` and linked back to it.
    /// Called under the object's entry lock; touches nothing else in the map.
    fn new_flight(
        self: &Arc<Self>,
        object_id: u64,
        estimate: LayerSize,
        predecessor: Option<&Arc<FlightSlot>>,
    ) -> (Arc<FlightSlot>, ReadStage) {
        let (done_tx, done_rx) = watch::channel(false);
        let predecessor_link = predecessor.map(Arc::downgrade);
        let predecessor = predecessor.map(|p| p.done.clone());
        let initial_phase = if predecessor.is_some() {
            FlightPhase::Predecessor
        } else {
            FlightPhase::Capacity
        };
        let (phase, _) = watch::channel(initial_phase);
        let flight = Arc::new(FlightSlot {
            object_id,
            coefficients: coefficients(&self.client),
            gate: Arc::clone(&self.client.catchup_gate),
            budget: Arc::clone(&self.client.catchup_build_budget),
            metrics: self.client.metrics_snapshot(),
            phase,
            done: done_rx,
            predecessor: predecessor_link,
            inner: StdMutex::new(SlotInner {
                state: FlightState::Queued,
                open: true,
                waiters: 1,
                size: estimate,
                read_stage: None,
                turns: Some(TurnServer {
                    slot: None,
                    busy: false,
                    queue: VecDeque::new(),
                    extra_part_bytes: 0,
                    _done: DoneSignal(done_tx),
                }),
            }),
        });
        let stage = read_stage(
            Arc::downgrade(self),
            Arc::clone(&flight),
            self.client.clone(),
            self.keys.clone(),
            predecessor,
            estimate,
        );
        flight.lock().read_stage = Some(stage.clone());
        (flight, stage)
    }

    /// One waiter of `flight` left: count it out under the object's entry
    /// lock and, if it was the last, take out everything the flight still
    /// holds — the registry's clone of its read stage and its turn server (the
    /// flight's slot after the read, and the `done` signal) — and, when the
    /// map still holds the flight, hand its entry back to its nearest running
    /// predecessor or empty it ([`Self::vacate`]). The parts are returned, not
    /// dropped, so the caller drops them after the locks and in the order
    /// [`FlightWaiter`]'s drop needs: every read-stage clone first (cancelling
    /// an unresolved read: its slot, permit, grant and lease guard drop), the
    /// turn server last (`done` fires).
    ///
    /// The entry lock is held whether or not the map holds this flight: a
    /// flight's turn server is only ever taken under it, so a predecessor that
    /// [`Self::vacate`] finds running cannot finish between that check and
    /// its going back into the map.
    fn leave(&self, flight: &Arc<FlightSlot>) -> (Option<ReadStage>, Option<TurnServer>) {
        let (released, vacated) = match self.slots.entry(flight.object_id) {
            Entry::Occupied(occupied) if Arc::ptr_eq(occupied.get(), flight) => {
                match flight.count_out() {
                    Some(released) => (Some(released), Some(Self::vacate(occupied, flight))),
                    None => (None, None),
                }
            }
            // Not in the map (a successor took its place, or its read
            // failed): no one can attach to it. Counted out under the entry
            // lock all the same, the rule above.
            other => {
                let released = flight.count_out();
                drop(other);
                (released, None)
            }
        };
        // The map's handle is never the flight's last: the leaving waiter
        // holds one. Dropped after the entry lock.
        drop(vacated);
        // Test-only: park here — counted out, the entry lock released, the
        // released parts not yet handed back — when a seam is armed for this
        // object (module docs, # Test seam).
        #[cfg(any(test, feature = "test-fault-injection"))]
        self.park_at_leave_seam(flight.object_id, released.is_some());
        released.unwrap_or((None, None))
    }

    /// Test-only: take the [`FlightLeaveSeam`] armed for `object_id`, if any,
    /// and park this leave on it. `was_last` is whether this leave counted
    /// out the flight's last waiter. Holds no lock while parked.
    #[cfg(any(test, feature = "test-fault-injection"))]
    fn park_at_leave_seam(&self, object_id: u64, was_last: bool) {
        let seam = {
            let mut armed = self
                .leave_seam
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match armed.as_ref() {
                Some((armed_for, _)) if *armed_for == object_id => {
                    armed.take().map(|(_, seam)| seam)
                }
                _ => None,
            }
        };
        if let Some(seam) = seam {
            seam.park(was_last);
        }
    }

    /// `flight`, which the map holds at `occupied`, is leaving it: put back
    /// its nearest predecessor still running, or remove the entry when none
    /// is ([`FlightSlot::running_predecessor`]). Returns the handle the map
    /// gave up, for the caller to drop after the entry lock.
    fn vacate(
        mut occupied: dashmap::mapref::entry::OccupiedEntry<'_, u64, Arc<FlightSlot>>,
        flight: &FlightSlot,
    ) -> Arc<FlightSlot> {
        match flight.running_predecessor() {
            Some(predecessor) => occupied.insert(predecessor),
            None => occupied.remove(),
        }
    }
}

/// The read stage's shared future: every waiter of the flight holds a clone
/// and drives it; it resolves to the flight's snapshot or a cloneable
/// failure.
type ReadStage = Shared<BoxFuture<'static, Result<Arc<AtomicSnapshot>, FlightFailure>>>;

/// How a read stage failed, cloneable so every waiter gets it (the
/// `SharedCheckpointError` pattern: the error itself is shared behind an
/// `Arc`, and each waiter takes its own copy).
#[derive(Debug, Clone)]
enum FlightFailure {
    /// A gate or the memory budget was closed at shutdown.
    Closed,
    /// The read failed with this error.
    Failed(Arc<ServiceError>),
}

impl FlightFailure {
    fn from_refusal(refusal: CatchUpRefusal) -> Self {
        match refusal {
            CatchUpRefusal::Closed => FlightFailure::Closed,
            // A flight's waits carry no deadline, so they cannot time out;
            // should one, it is a transient failure, never a hang.
            CatchUpRefusal::TimedOut => FlightFailure::Failed(Arc::new(ServiceError::Unavailable(
                "a shared snapshot read's capacity wait ended without capacity; retry".to_owned(),
            ))),
        }
    }
}

/// One object's flight (Spec SDD SR-6.1): its read stage, its state and its
/// build-turn server; see the module docs.
pub struct FlightSlot {
    object_id: u64,
    /// The fence coefficients `reader_wait_max(L)` is computed from.
    coefficients: FenceCoefficients,
    gate: Arc<CatchUpGate>,
    budget: Arc<MemoryBudget>,
    metrics: Option<SharedMetrics>,
    phase: watch::Sender<FlightPhase>,
    /// Becomes `true` when the flight is over (its turn server dropped); a
    /// successor waits for it.
    done: watch::Receiver<bool>,
    /// The flight this one succeeded, if any: where the map entry goes back
    /// when this flight leaves the map while its predecessor still runs
    /// (module docs). Weak, so a successor never keeps a finished predecessor
    /// alive.
    predecessor: Option<Weak<FlightSlot>>,
    inner: StdMutex<SlotInner>,
}

impl std::fmt::Debug for FlightSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("FlightSlot")
            .field("object_id", &self.object_id)
            .field("state", &inner.state)
            .field("open", &inner.open)
            .field("waiters", &inner.waiters)
            .field("phase", &*self.phase.borrow())
            .field("done", &*self.done.borrow())
            .finish_non_exhaustive()
    }
}

struct SlotInner {
    state: FlightState,
    /// Whether a catch-up may attach (module docs).
    open: bool,
    /// Attached waiters: counted in at attach or creation, out when a
    /// [`FlightWaiter`] drops.
    waiters: usize,
    /// The best size known for the layer ([`FlightLookup::size`]).
    size: LayerSize,
    /// The registry's clone of the read stage, handed to attaching waiters;
    /// cleared once the read resolved (by the first waiter to see it) or when
    /// the last waiter leaves, so the map never keeps a resolved snapshot
    /// alive.
    read_stage: Option<ReadStage>,
    /// `None` once the flight is over: its last waiter left, or its read
    /// failed. Taken only under the object's entry lock, so `Some` is what
    /// "still running" means to [`FlightSlot::running_predecessor`].
    turns: Option<TurnServer>,
}

impl SlotInner {
    /// Close the flight and take out what it holds, for the caller to drop
    /// outside the locks.
    fn release_all(&mut self) -> (Option<ReadStage>, Option<TurnServer>) {
        self.open = false;
        (self.read_stage.take(), self.turns.take())
    }
}

/// The build-turn server (Spec SDD SR-6.3, fenced arm): the flight's slot
/// after the read, its one build turn, the waiters queued for it, and the
/// `done` signal.
struct TurnServer {
    /// The flight's catch-up slot, from the read's resolution until the
    /// flight is over.
    slot: Option<CatchUpPermit>,
    /// Whether the flight's own build turn is held.
    busy: bool,
    /// Waiters waiting for a turn, oldest first.
    queue: VecDeque<oneshot::Sender<TurnGuard>>,
    /// An extra build's part ([`extra_build_part_bytes`] at the read's
    /// `reply_bytes`).
    extra_part_bytes: u64,
    /// Fires `done` when the server is dropped: after the flight's last
    /// build, on a failed read, or when every waiter left. Declared last, so
    /// the slot above is released before `done` fires.
    _done: DoneSignal,
}

/// Sets the flight's `done` when dropped (Spec SDD SR-6.5).
struct DoneSignal(watch::Sender<bool>);

impl Drop for DoneSignal {
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}

impl FlightSlot {
    fn lock(&self) -> MutexGuard<'_, SlotInner> {
        // Nothing panics while holding the lock, and every update leaves the
        // state consistent, so a poisoned lock's state is still sound.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The object this flight reads.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// The flight's read state.
    pub fn state(&self) -> FlightState {
        self.lock().state
    }

    /// What the flight is doing now.
    pub fn phase(&self) -> FlightPhase {
        *self.phase.borrow()
    }

    /// Whether a catch-up attaching now would join this flight.
    pub fn is_attachable(&self) -> bool {
        self.lock().open
    }

    /// Catch-ups attached now (counted in at attach, out when their
    /// [`FlightWaiter`] drops).
    pub fn waiters(&self) -> usize {
        self.lock().waiters
    }

    /// Whether the flight is over and its `done` has fired: its last build
    /// ended, its read failed, or every waiter left.
    pub fn is_done(&self) -> bool {
        *self.done.borrow()
    }

    /// Run `f` on the flight's state under the flight's own lock and, when
    /// the map has an entry for `object_id` — this flight or another (its
    /// successor, or a predecessor put back) — a read guard on that entry's
    /// shard (`DashMap::get`). The guard serializes `f` with every operation
    /// that takes the object's entry for writing: attach, creating a
    /// successor, [`Registry::leave`], [`Registry::vacate`] and a failed
    /// read's removal. It does not serialize `f` with other callers of this
    /// function; the flight's own lock does.
    ///
    /// When the map has no entry for the object (or the registry is gone), no
    /// map lock is held, only the flight's own. Nothing can attach to the
    /// flight then: it is out of the map, a newcomer creates a fresh flight
    /// with no link to it, and [`Registry::vacate`] puts back only a
    /// predecessor of a flight the map still holds.
    fn with_entry_lock<R>(
        &self,
        registry: &Weak<Registry>,
        f: impl FnOnce(&mut SlotInner) -> R,
    ) -> R {
        let registry = registry.upgrade();
        let _entry = registry.as_ref().and_then(|r| r.slots.get(&self.object_id));
        let mut inner = self.lock();
        f(&mut inner)
    }

    fn set_phase(&self, phase: FlightPhase) {
        self.phase.send_if_modified(|current| {
            let changed = *current != phase;
            *current = phase;
            changed
        });
    }

    fn set_size(&self, size: LayerSize) {
        self.lock().size = size;
    }

    /// Count one waiter out; when it was the last, close the flight and take
    /// out what it holds ([`SlotInner::release_all`]). Called under the
    /// object's entry lock ([`Registry::leave`]).
    fn count_out(&self) -> Option<(Option<ReadStage>, Option<TurnServer>)> {
        let mut inner = self.lock();
        inner.waiters = inner.waiters.saturating_sub(1);
        (inner.waiters == 0).then(|| inner.release_all())
    }

    /// The nearest flight up this flight's predecessor links that is still
    /// running (its turn server not yet taken, so its `done` has not fired),
    /// if any: what the map entry goes back to when this flight leaves it.
    /// Called under the object's entry lock, which every taking of a turn
    /// server holds, so the answer stays true until that lock is released.
    /// Locks one flight at a time, never this one.
    fn running_predecessor(&self) -> Option<Arc<FlightSlot>> {
        let mut next = self.predecessor.as_ref().and_then(Weak::upgrade);
        while let Some(candidate) = next {
            if candidate.lock().turns.is_some() {
                return Some(candidate);
            }
            next = candidate.predecessor.as_ref().and_then(Weak::upgrade);
        }
        None
    }

    /// `reader_wait_max(L)` for the size the flight knows now.
    fn reader_wait_max(&self) -> Duration {
        let size = self.lock().size;
        ms_to_duration(self.coefficients.timing(size).reader_wait_max_ms)
    }

    /// The first waiter to see the read resolved drops the registry's clone of
    /// the read stage (module docs). Dropped after the lock.
    fn clear_read_stage(&self) {
        let stage = self.lock().read_stage.take();
        drop(stage);
    }

    /// The read stage ended: close the flight to attach under the entry lock
    /// and record the outcome. On success the flight's slot passes to the turn
    /// server and the flight stays in the map, closed, until its last build
    /// ends. On failure the flight leaves the map (when it still holds it) and
    /// its turn server is dropped, firing `done`; it is never reused.
    fn resolve(
        self: &Arc<Self>,
        registry: &Weak<Registry>,
        staged: Result<(AtomicSnapshot, CatchUpPermit), FlightFailure>,
    ) -> Result<Arc<AtomicSnapshot>, FlightFailure> {
        match staged {
            Ok((snapshot, slot)) => {
                let snapshot = Arc::new(snapshot);
                let measured = snapshot.measured_size();
                let extra_part_bytes = extra_build_part_bytes(snapshot.reply_bytes);
                // `Some` only when no turn server is left to take it — every
                // waiter gone, which cannot happen while one drives this
                // future; dropped after the lock either way.
                let unclaimed = self.with_entry_lock(registry, |inner| {
                    inner.open = false;
                    inner.size = measured;
                    match inner.turns.as_mut() {
                        Some(turns) => {
                            turns.slot = Some(slot);
                            turns.extra_part_bytes = extra_part_bytes;
                            None
                        }
                        None => Some(slot),
                    }
                });
                drop(unclaimed);
                Ok(snapshot)
            }
            Err(failure) => {
                let close = |flight: &FlightSlot| {
                    let mut inner = flight.lock();
                    inner.open = false;
                    inner.turns.take()
                };
                // Under the entry lock, like every taking of a turn server
                // (`Registry::leave`), whether or not the map holds it.
                let (released, vacated) = match registry.upgrade() {
                    Some(registry) => match registry.slots.entry(self.object_id) {
                        Entry::Occupied(occupied) if Arc::ptr_eq(occupied.get(), self) => {
                            let released = close(self);
                            (released, Some(Registry::vacate(occupied, self)))
                        }
                        other => {
                            let released = close(self);
                            drop(other);
                            (released, None)
                        }
                    },
                    // Every waiter holds the registry, and one is driving
                    // this read; unreachable, and nothing to lock if it were.
                    None => (close(self), None),
                };
                drop(vacated);
                drop(released);
                Err(failure)
            }
        }
    }

    /// Ask for a build turn: the flight's own when it is free and no one is
    /// queued, otherwise a place in the queue — and then an extra build for
    /// the queue's head, if a slot and a build part are free at once.
    fn request_turn(self: &Arc<Self>) -> TurnRequest {
        let rx = {
            let mut inner = self.lock();
            let Some(turns) = inner.turns.as_mut() else {
                return TurnRequest::Ready(TurnGuard::Unserved);
            };
            if !turns.busy && turns.queue.is_empty() {
                turns.busy = true;
                return TurnRequest::Ready(TurnGuard::Flight(FlightTurn {
                    flight: Arc::clone(self),
                    armed: true,
                }));
            }
            let (tx, rx) = oneshot::channel();
            turns.queue.push_back(tx);
            rx
        };
        self.try_extras();
        TurnRequest::Queued(rx)
    }

    /// The flight's own build turn came back: hand it to the oldest waiter
    /// still waiting, or mark it free. Having handed it on, start extra
    /// builds for the waiters still queued behind it ([`Self::try_extras`]):
    /// a slot or memory freed elsewhere on the replica since they queued is
    /// noticed here, not only at the next request or extra build's end.
    ///
    /// Called from [`FlightTurn`]'s drop with no lock held; the flight's lock
    /// is released before the send and before `try_extras`, which takes it
    /// again, and neither a refused send (disarmed) nor `try_extras` calls
    /// back here.
    fn hand_on_flight_turn(self: &Arc<Self>) {
        loop {
            let next = {
                let mut inner = self.lock();
                let Some(turns) = inner.turns.as_mut() else {
                    return;
                };
                match turns.queue.pop_front() {
                    Some(tx) => tx,
                    None => {
                        turns.busy = false;
                        return;
                    }
                }
            };
            let turn = TurnGuard::Flight(FlightTurn {
                flight: Arc::clone(self),
                armed: true,
            });
            match next.send(turn) {
                Ok(()) => break,
                // That waiter left while queued: try the next.
                Err(returned) => returned.disarm(),
            }
        }
        // Returns at once when no one is left queued.
        self.try_extras();
    }

    /// Start extra builds (Spec SDD SR-6.3) for the queue's head, one at a
    /// time, while a catch-up slot and an extra build part are both free at
    /// once; both are non-blocking, so this never waits.
    fn try_extras(self: &Arc<Self>) {
        loop {
            let (tx, slot, part) = {
                let mut inner = self.lock();
                let Some(turns) = inner.turns.as_mut() else {
                    return;
                };
                // Skip waiters that left, so a free slot and part go to one
                // that will build.
                while turns.queue.front().is_some_and(oneshot::Sender::is_closed) {
                    turns.queue.pop_front();
                }
                if turns.queue.is_empty() {
                    return;
                }
                let Some(slot) = self.gate.try_acquire() else {
                    return;
                };
                let Some(part) = self.budget.try_grant(turns.extra_part_bytes) else {
                    return;
                };
                let Some(tx) = turns.queue.pop_front() else {
                    return;
                };
                (tx, slot, part)
            };
            let turn = TurnGuard::Extra(ExtraTurn {
                flight: Arc::clone(self),
                slot: Some(slot),
                part: Some(part),
                armed: true,
            });
            if let Err(returned) = tx.send(turn) {
                returned.disarm();
            }
        }
    }
}

/// The read stage's future for `flight` (module docs): the fenced read under
/// `catch_unwind`, then [`FlightSlot::resolve`].
fn read_stage(
    registry: Weak<Registry>,
    flight: Arc<FlightSlot>,
    client: ValkeyClient,
    keys: KeySchema,
    predecessor: Option<watch::Receiver<bool>>,
    estimate: LayerSize,
) -> ReadStage {
    async move {
        let staged = AssertUnwindSafe(run_read_stage(
            &registry,
            &flight,
            &client,
            &keys,
            predecessor,
            estimate,
        ))
        .catch_unwind()
        .await;
        let staged = staged.unwrap_or_else(|_panic| {
            // The object id is on this line; each attached catch-up's own
            // failure line carries its request's fields.
            tracing::error!(
                object_id = flight.object_id,
                "shared snapshot read panicked; failing every catch-up attached to it"
            );
            Err(FlightFailure::Failed(Arc::new(ServiceError::Internal(
                "a shared snapshot read panicked".to_owned(),
            ))))
        });
        flight.resolve(&registry, staged)
    }
    .boxed()
    .shared()
}

/// The read stage proper: the predecessor wait, the flight's slot, and the
/// fenced read under the flight's [`FlightHook`]. Returns the snapshot, now
/// owning the flight's grant and top-up, and the slot for the turn server.
async fn run_read_stage(
    registry: &Weak<Registry>,
    flight: &Arc<FlightSlot>,
    client: &ValkeyClient,
    keys: &KeySchema,
    predecessor: Option<watch::Receiver<bool>>,
    estimate: LayerSize,
) -> Result<(AtomicSnapshot, CatchUpPermit), FlightFailure> {
    if let Some(mut done) = predecessor {
        flight.set_phase(FlightPhase::Predecessor);
        // An error means the predecessor's signal is gone without firing,
        // which its drop guard rules out; either way its flight is over.
        let _ = done.wait_for(|over| *over).await;
    }
    flight.set_phase(FlightPhase::Capacity);
    let slot = client
        .catchup_gate
        .acquire(no_deadline())
        .await
        .map_err(FlightFailure::from_refusal)?;
    flight.set_phase(FlightPhase::Work);

    let mut hook = FlightHook {
        registry,
        flight,
        client,
        permit: None,
        grant: None,
        top_up: None,
        refusal: None,
    };
    let read = Box::pin(read_object_snapshot_fenced(
        client,
        keys,
        flight.object_id,
        SnapshotReadConsumer::Catchup,
        estimate,
        &mut hook,
    ))
    .await;
    // Released already after a verified read's adjust; on any other ending,
    // now.
    hook.permit = None;
    let mut snapshot = match read {
        Ok(snapshot) => snapshot,
        Err(error) => {
            return Err(match hook.refusal.take() {
                Some(refusal) => FlightFailure::from_refusal(refusal),
                None => FlightFailure::Failed(Arc::new(error)),
            })
        }
    };
    if let Some(grant) = hook.grant.take() {
        snapshot.memory = SnapshotMemory::new(grant, hook.top_up.take());
    }
    Ok((snapshot, slot))
}

/// The flight's [`FencedReadHook`]: Spec SDD SR-5.6's acquire cycle, the
/// flip, the close before release and the adjust after it.
struct FlightHook<'a> {
    registry: &'a Weak<Registry>,
    flight: &'a Arc<FlightSlot>,
    client: &'a ValkeyClient,
    permit: Option<CatchUpReadPermit>,
    grant: Option<MemoryGrant>,
    top_up: Option<MemoryTopUp>,
    /// Why a capacity wait ended the read, if one did.
    refusal: Option<CatchUpRefusal>,
}

impl FlightHook<'_> {
    /// Take what the attempt needs and is not already held: the fenced read
    /// permit, then the combined reservation sized for `size`. Both waits
    /// carry no deadline and are the flight's `Capacity` phase.
    async fn take_capacity(&mut self, size: LayerSize) -> Result<(), ServiceError> {
        if self.permit.is_none() {
            // Never wait for a permit while holding memory (Spec SDD SR-7.5).
            self.top_up = None;
            self.grant = None;
            self.flight.set_phase(FlightPhase::Capacity);
            let permit = self
                .client
                .catchup_fenced_read_gate
                .acquire(no_deadline(), SnapshotReadConsumer::Catchup)
                .await;
            self.permit = Some(permit.map_err(|refusal| self.refuse(refusal))?);
        }
        if self.grant.is_none() {
            self.flight.set_phase(FlightPhase::Capacity);
            let grant = self
                .client
                .catchup_build_budget
                .reserve(flight_grant_bytes(size.bytes), None)
                .await;
            self.grant = Some(grant.map_err(|refusal| self.refuse(refusal))?);
        }
        self.flight.set_phase(FlightPhase::Work);
        Ok(())
    }

    /// End the read on a capacity refusal, remembering why.
    fn refuse(&mut self, refusal: CatchUpRefusal) -> ServiceError {
        self.refusal = Some(refusal);
        ServiceError::Unavailable(
            "this replica is shutting down; its catch-up capacity is closed".to_owned(),
        )
    }
}

impl FencedReadHook for FlightHook<'_> {
    async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
        if attempt.follows_broken_read() {
            // A broken read returns the flight to `Queued`, attachable, for
            // its retry: no cut is fixed.
            self.flight.with_entry_lock(self.registry, |inner| {
                inner.state = FlightState::Queued;
                inner.open = true;
            });
        }
        self.flight.set_size(attempt.size);
        self.take_capacity(attempt.size).await
    }

    async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
        // Nothing held idle through the sleep: the reservation and its
        // top-up (a broken attempt's, kept for this retry), then the permit.
        // The slot stays, so the next acquire runs at the wake.
        self.top_up = None;
        self.grant = None;
        self.permit = None;
        self.flight.set_size(retry.size);
        self.flight.set_phase(FlightPhase::Work);
        tokio::time::sleep(retry.hint).await;
        self.take_capacity(retry.size).await
    }

    async fn on_grant(&mut self, grant: FenceGrant) {
        // The cut is fixed. Attach stays open until the release is sent: no
        // commit lands during the epoch, so a pin read now equals the cut.
        self.flight.with_entry_lock(self.registry, |inner| {
            inner.state = FlightState::Reading;
            inner.size = grant.size;
        });
        tracing::debug!(
            target: TRACE_TARGET,
            object_id = self.flight.object_id,
            cut = grant.cut,
            kind = grant.kind.as_str(),
            "flight_read"
        );
    }

    fn on_read_progress(&mut self, bytes_read: u64) {
        // The read holds its fenced permit throughout (asserted). A top-up
        // records memory already resident, so it is taken whenever a grant
        // is held: withholding it would only hide that memory from
        // admission. The snapshot alone is resident while the read runs: its
        // factor, not the flight's combined one.
        debug_assert!(self.permit.is_some(), "a fenced read holds its permit");
        if let Some(grant) = &self.grant {
            grant.top_up_to(
                &mut self.top_up,
                bytes_read.saturating_mul(SNAPSHOT_MEMORY_FACTOR),
            );
        }
    }

    async fn before_release(&mut self) {
        // A pin read after the release could exceed the cut.
        self.flight.with_entry_lock(self.registry, |inner| {
            inner.open = false;
        });
    }

    async fn after_release(&mut self, released: FenceReleased) {
        if released.outcome != FenceAttemptOutcome::Verified {
            // A broken attempt keeps its permit, grant and top-up for the
            // retry that follows at once; an ended read drops them with the
            // hook.
            return;
        }
        // Set the grant and its one top-up to the measured weight, and only
        // then give the permit back: at most `P_f` top-ups are outstanding
        // past `T`.
        if let (Some(grant), Some(measured)) = (self.grant.as_mut(), released.measured) {
            grant.adjust_with_top_up(&mut self.top_up, flight_weight_bytes(measured.bytes));
        }
        self.permit = None;
    }
}

/// One catch-up attached to a flight (Spec SDD SR-6.1's ref-counted waiter):
/// it holds its place in the flight's waiter count and its clone of the read
/// stage. Dropping it at any point leaves the flight; the last waiter to
/// leave ends the flight (module docs).
pub struct FlightWaiter {
    /// Dropped before the waiter is counted out, so the last waiter to count
    /// out holds no clone of its own and the registry's clone it takes out is
    /// the last one; a cancelled read's state goes with it before `done`
    /// fires (see the `Drop` impl).
    read_stage: Option<ReadStage>,
    flight: Arc<FlightSlot>,
    registry: Arc<Registry>,
    role: FlightRole,
}

impl std::fmt::Debug for FlightWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlightWaiter")
            .field("flight", &self.flight)
            .field("role", &self.role)
            .finish_non_exhaustive()
    }
}

impl Drop for FlightWaiter {
    fn drop(&mut self) {
        // Invariant (Spec SDD SR-6.5): a successor never wakes to find this
        // flight's slot, permit or memory still held. So every waiter drops
        // its own clone of the read stage BEFORE it counts out. Waiters that
        // leave together (a session deletion, a drain) then cannot leave a
        // clone alive past the last count-out: whichever counts out last, the
        // others' clones are already gone. Dropped outside the locks; a clone
        // that is not the last only decrements a count.
        drop(self.read_stage.take());
        let (registry_stage, turns) = self.registry.leave(&self.flight);
        // The last waiter to count out takes the registry's clone, the last
        // one left: dropping it drops the read itself (or its resolved
        // snapshot), which gives back its slot, fenced permit and grant at
        // once and spawns its lease's release. Only then does the turn server
        // drop and fire `done`.
        drop(registry_stage);
        drop(turns);
    }
}

impl FlightWaiter {
    /// This waiter's part in the flight.
    pub fn role(&self) -> FlightRole {
        self.role
    }

    /// The flight this waiter is attached to.
    pub fn flight(&self) -> &FlightSlot {
        &self.flight
    }

    /// Wait for the flight's shared read, driving it, and charge this
    /// waiter for its share of the flight's waits (module docs, # Per-waiter
    /// charging):
    ///
    /// - while the flight is in [`FlightPhase::Capacity`], the time passes on
    ///   `budget`; when the budget is out, the waiter leaves with
    ///   [`FlightWaitError::CapacityExhausted`] (a zero budget therefore
    ///   leaves at the first capacity wait that is not satisfied at once);
    /// - while it is in [`FlightPhase::Work`], the time counts against
    ///   `reader_wait_max(L)` for the size the flight knows; past it the
    ///   waiter leaves with [`FlightWaitError::ReaderWaitCap`], with a WARN;
    /// - [`FlightPhase::Predecessor`] is neither charged nor capped.
    ///
    /// On success the waiter is still attached, inside the returned
    /// [`FlightRead`]; on any error it has left. The caller races this against
    /// its own endings; dropping it leaves the flight.
    pub async fn read(mut self, budget: &CapacityBudget) -> Result<FlightRead, FlightWaitError> {
        let mut phases = self.flight.phase.subscribe();
        let mut phases_open = true;
        let mut work = Duration::ZERO;
        let waited = {
            let Some(stage) = self.read_stage.as_mut() else {
                // Unreachable: a waiter holds its read stage until it reads.
                return Err(FlightWaitError::Read(ServiceError::Internal(
                    "snapshot flight waiter read twice".to_owned(),
                )));
            };
            loop {
                let phase = *phases.borrow_and_update();
                let started = Instant::now();
                let limit = match phase {
                    FlightPhase::Predecessor => None,
                    FlightPhase::Capacity => started.checked_add(budget.remaining()),
                    FlightPhase::Work => {
                        started.checked_add(self.flight.reader_wait_max().saturating_sub(work))
                    }
                };
                let expired = async move {
                    match limit {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending().await,
                    }
                };
                let step = tokio::select! {
                    biased;
                    out = &mut *stage => Step::Resolved(out),
                    changed = phases.changed(), if phases_open => {
                        // The sender lives in the flight this waiter holds,
                        // so it is never gone; stop polling it if it were.
                        phases_open = changed.is_ok();
                        Step::PhaseChanged
                    }
                    () = expired => Step::Expired,
                };
                let elapsed = started.elapsed();
                match phase {
                    FlightPhase::Capacity => budget.charge(elapsed),
                    FlightPhase::Work => work = work.saturating_add(elapsed),
                    FlightPhase::Predecessor => {}
                }
                match step {
                    Step::Resolved(out) => break Ok(out),
                    Step::PhaseChanged => {}
                    Step::Expired => break Err(phase),
                }
            }
        };

        // The clone polled to completion holds nothing more; once the read
        // resolved, the registry's clone is cleared too, so the map keeps no
        // resolved snapshot alive.
        self.read_stage = None;
        if waited.is_ok() {
            self.flight.clear_read_stage();
        }
        match waited {
            Ok(Ok(snapshot)) => Ok(FlightRead {
                snapshot,
                waiter: self,
            }),
            Ok(Err(FlightFailure::Closed)) => Err(FlightWaitError::ShuttingDown),
            Ok(Err(FlightFailure::Failed(error))) => {
                Err(FlightWaitError::Read(copy_service_error(&error)))
            }
            Err(FlightPhase::Work) => {
                let cap = self.flight.reader_wait_max();
                let size = self.flight.lock().size;
                tracing::warn!(
                    object_id = self.flight.object_id,
                    waited_ms = work.as_millis() as u64,
                    reader_wait_max_ms = cap.as_millis() as u64,
                    size_items = size.items,
                    size_bytes = size.bytes,
                    "catch-up gave up waiting for its layer's shared fenced snapshot read: the \
                     read's work passed reader_wait_max(L); the read continues for any other \
                     catch-up attached to it"
                );
                // Counted by the caller as this catch-up's ending; the
                // reader-wait histogram stays per read attempt.
                Err(FlightWaitError::ReaderWaitCap { waited: work, cap })
            }
            Err(_) => Err(FlightWaitError::CapacityExhausted),
        }
    }
}

/// One iteration's outcome in [`FlightWaiter::read`].
enum Step {
    Resolved(Result<Arc<AtomicSnapshot>, FlightFailure>),
    PhaseChanged,
    Expired,
}

/// A waiter whose flight's read succeeded: the shared snapshot, with the
/// waiter still attached. Check the cut against the waiter's pin and register
/// its stream from [`Self::snapshot`], then build in
/// [`Self::build_turn`]. Dropping it leaves the flight.
pub struct FlightRead {
    // Field order is drop order, and it is load-bearing (Spec SDD SR-6.5):
    // `snapshot` — possibly the last handle, which owns the flight's grant —
    // must drop before `waiter`, whose drop may count out the flight's last
    // waiter and fire `done`. Keep `snapshot` first, and never move the two
    // out into locals that live across an await ([`Self::build_turn`]).
    snapshot: Arc<AtomicSnapshot>,
    waiter: FlightWaiter,
}

impl std::fmt::Debug for FlightRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlightRead")
            .field("cut", &self.snapshot.cut)
            .field("waiter", &self.waiter)
            .finish()
    }
}

impl FlightRead {
    /// The shared snapshot, proven at its `cut`.
    pub fn snapshot(&self) -> &AtomicSnapshot {
        &self.snapshot
    }

    /// This waiter's part in the flight.
    pub fn role(&self) -> FlightRole {
        self.waiter.role
    }

    /// Wait for this waiter's build turn (uncharged, uncapped work: bounded by
    /// the builds ahead of it) and return it. The turn is the flight's own, or
    /// an extra build's when a slot and a build part were free. The caller
    /// races this against its own endings; dropping it gives the place up.
    pub async fn build_turn(self) -> BuildTurn {
        // `self` stays whole across the await: cancelled while queued, it
        // drops as a `FlightRead`, snapshot before waiter (the field-order
        // invariant on the struct). Split only after the last await.
        let turn = match self.waiter.flight.request_turn() {
            TurnRequest::Ready(turn) => turn,
            // A queued turn's sender is dropped unsent only with the turn
            // server, which outlives every waiter; build unserved if it were.
            TurnRequest::Queued(rx) => rx.await.unwrap_or(TurnGuard::Unserved),
        };
        let FlightRead { snapshot, waiter } = self;
        BuildTurn::new(snapshot, turn, waiter)
    }
}

/// A waiter's build turn: the shared snapshot to build from, the turn's
/// guard, and the waiter's place in the flight. Derefs to the snapshot.
///
/// Drop it as soon as the build returns, before checking the stream or
/// sending anything: dropping it gives back, in order, the snapshot handle
/// (its memory is repaid when the last one drops), the turn (handed to the
/// next waiter, or an extra build's slot and part returned) and the waiter's
/// place (the last one ends the flight, releasing its slot and firing
/// `done`). Copy out what the send needs (the cut, the timestamps) first.
pub struct BuildTurn {
    // Field order is the drop order the docs above give, and it is
    // load-bearing (Spec SDD SR-6.5): the snapshot handle and the turn go
    // before the waiter, whose drop may count out the flight's last waiter
    // and fire `done`.
    snapshot: Arc<AtomicSnapshot>,
    turn: TurnGuard,
    _building: BuildingGauge,
    _waiter: FlightWaiter,
}

impl std::fmt::Debug for BuildTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildTurn")
            .field("cut", &self.snapshot.cut)
            .field("extra", &self.is_extra())
            .finish_non_exhaustive()
    }
}

impl BuildTurn {
    fn new(snapshot: Arc<AtomicSnapshot>, turn: TurnGuard, waiter: FlightWaiter) -> Self {
        Self {
            snapshot,
            turn,
            _building: BuildingGauge::start(waiter.flight.metrics.clone()),
            _waiter: waiter,
        }
    }

    /// Whether this is an extra build, under its own slot and build part,
    /// rather than the flight's own turn.
    pub fn is_extra(&self) -> bool {
        matches!(self.turn, TurnGuard::Extra(_))
    }
}

impl std::ops::Deref for BuildTurn {
    type Target = AtomicSnapshot;

    fn deref(&self) -> &AtomicSnapshot {
        &self.snapshot
    }
}

/// Counts one build on `live_edit_catchup_builds_in_progress` while alive.
struct BuildingGauge(Option<SharedMetrics>);

impl BuildingGauge {
    fn start(metrics: Option<SharedMetrics>) -> Self {
        if let Some(m) = &metrics {
            m.add_catchup_builds_in_progress(1);
        }
        Self(metrics)
    }
}

impl Drop for BuildingGauge {
    fn drop(&mut self) {
        if let Some(m) = &self.0 {
            m.add_catchup_builds_in_progress(-1);
        }
    }
}

/// What [`FlightSlot::request_turn`] gave.
enum TurnRequest {
    Ready(TurnGuard),
    Queued(oneshot::Receiver<TurnGuard>),
}

/// A held build turn.
enum TurnGuard {
    /// The flight's own turn, under its slot and the build part of its grant.
    Flight(FlightTurn),
    /// An extra build, under its own slot and build part.
    Extra(ExtraTurn),
    /// No turn server was left to serve one; see [`FlightRead::build_turn`].
    Unserved,
}

impl TurnGuard {
    /// Drop a turn that could not be handed over without passing it on again
    /// (the caller is already passing turns on): an extra build's slot and
    /// part are still given back.
    fn disarm(self) {
        match self {
            TurnGuard::Flight(mut turn) => turn.armed = false,
            TurnGuard::Extra(mut turn) => turn.armed = false,
            TurnGuard::Unserved => {}
        }
    }
}

/// The flight's own build turn; passed to the next waiter when dropped.
struct FlightTurn {
    flight: Arc<FlightSlot>,
    armed: bool,
}

impl Drop for FlightTurn {
    fn drop(&mut self) {
        if self.armed {
            self.flight.hand_on_flight_turn();
        }
    }
}

/// An extra build's turn: its catch-up slot and build part, given back when
/// dropped, which may let the next queued waiter build at once too.
struct ExtraTurn {
    flight: Arc<FlightSlot>,
    slot: Option<CatchUpPermit>,
    part: Option<MemoryGrant>,
    armed: bool,
}

impl Drop for ExtraTurn {
    fn drop(&mut self) {
        drop(self.part.take());
        drop(self.slot.take());
        if self.armed {
            self.flight.try_extras();
        }
    }
}

/// Test-only park point inside a waiter's leaving (module docs, # Test seam),
/// armed with [`SnapshotFlights::arm_leave_seam_for_test`].
///
/// The leave that takes it parks the **thread it runs on** — `leave` runs
/// synchronously inside [`FlightWaiter`]'s drop — on a `Condvar`, after its
/// count-out and with no lock held, until [`Self::release`]. A test must
/// therefore drop the waiter it means to park on a plain OS thread (e.g. a
/// `std::thread::scope` thread that enters the test's runtime with
/// `Handle::enter`, so a dropped lease guard can still spawn its release),
/// never on a runtime worker or the test's own task: a parked worker stalls
/// every task scheduled on it. Wait for the park with [`Self::is_parked`]
/// polled from async code, or [`Self::wait_until_parked`] from a thread that
/// may block. A park that is never released ends on its own after
/// [`LEAVE_SEAM_PARK_LIMIT`], so a failed test cannot hang its parked
/// thread for ever.
///
/// Usage, two waiters `w1` and `w2` of one flight of `object_id`:
/// arm (`let seam = flights.arm_leave_seam_for_test(object_id)`); drop `w1`
/// on a spawned OS thread; wait until `seam.is_parked()` (`w1` counted out,
/// not last: `seam.parked_was_last() == Some(false)`); drop `w2` (it passes
/// through, counts out last, `done` fires); assert; then `seam.release()`
/// and join the thread.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct FlightLeaveSeam {
    state: StdMutex<LeaveSeamState>,
    changed: std::sync::Condvar,
}

/// How long a [`FlightLeaveSeam`] park lasts at most without a release.
#[cfg(any(test, feature = "test-fault-injection"))]
pub const LEAVE_SEAM_PARK_LIMIT: Duration = Duration::from_secs(60);

#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
struct LeaveSeamState {
    /// `Some(was_last)` while a leave is parked.
    parked: Option<bool>,
    /// Parks so far, ended or not.
    parks: usize,
    /// Released: a parked leave returns, and a later arrival passes through.
    released: bool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl FlightLeaveSeam {
    fn lock(&self) -> MutexGuard<'_, LeaveSeamState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Park the calling thread until released or [`LEAVE_SEAM_PARK_LIMIT`].
    fn park(&self, was_last: bool) {
        let deadline = std::time::Instant::now() + LEAVE_SEAM_PARK_LIMIT;
        let mut state = self.lock();
        state.parks += 1;
        state.parked = Some(was_last);
        self.changed.notify_all();
        while !state.released {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                break;
            }
            state = self
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        state.parked = None;
        self.changed.notify_all();
    }

    /// Whether a leave is parked now.
    pub fn is_parked(&self) -> bool {
        self.lock().parked.is_some()
    }

    /// While a leave is parked, whether it counted out the flight's last
    /// waiter; `None` when none is parked.
    pub fn parked_was_last(&self) -> Option<bool> {
        self.lock().parked
    }

    /// How many leaves have parked here (0 or 1: the seam is one-shot).
    pub fn parks(&self) -> usize {
        self.lock().parks
    }

    /// Block the calling thread until a leave is parked, at most `limit`;
    /// returns whether one is. Not for a runtime worker (type docs).
    pub fn wait_until_parked(&self, limit: Duration) -> bool {
        let state = self.lock();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, limit, |state| state.parked.is_none())
            .unwrap_or_else(PoisonError::into_inner);
        state.parked.is_some()
    }

    /// Let the parked leave return (and any later arrival pass through).
    pub fn release(&self) {
        self.lock().released = true;
        self.changed.notify_all();
    }
}

/// A waiter's own copy of a shared read's error, classification kept.
fn copy_service_error(error: &ServiceError) -> ServiceError {
    match error {
        ServiceError::InvalidArgument(m) => ServiceError::InvalidArgument(m.clone()),
        ServiceError::InvalidArgumentWithDetail { message, detail } => {
            ServiceError::InvalidArgumentWithDetail {
                message: message.clone(),
                detail: detail.clone(),
            }
        }
        ServiceError::NotFound(m) => ServiceError::NotFound(m.clone()),
        ServiceError::AlreadyExists(m) => ServiceError::AlreadyExists(m.clone()),
        ServiceError::PermissionDenied => ServiceError::PermissionDenied,
        ServiceError::Unauthenticated => ServiceError::Unauthenticated,
        ServiceError::FailedPrecondition(m) => ServiceError::FailedPrecondition(m.clone()),
        ServiceError::Aborted(m) => ServiceError::Aborted(m.clone()),
        ServiceError::PredecessorNotCommitted { message, detail } => {
            ServiceError::PredecessorNotCommitted {
                message: message.clone(),
                detail: detail.clone(),
            }
        }
        ServiceError::Cancelled(m) => ServiceError::Cancelled(m.clone()),
        ServiceError::Unavailable(m) => ServiceError::Unavailable(m.clone()),
        ServiceError::UnavailableWithDetail { message, detail } => {
            ServiceError::UnavailableWithDetail {
                message: message.clone(),
                detail: detail.clone(),
            }
        }
        ServiceError::Internal(m) => ServiceError::Internal(m.clone()),
        ServiceError::PermanentKeyspaceRefusal { refusal, detail } => {
            ServiceError::PermanentKeyspaceRefusal {
                refusal: *refusal,
                detail: detail.clone(),
            }
        }
    }
}

/// Compile-time check that the flight's futures and handles can live in a
/// subscription's task and be shared across tasks.
#[allow(dead_code)]
fn _assert_send(waiter: FlightWaiter, budget: CapacityBudget, read: FlightRead) {
    fn send<T: Send>(_: &T) {}
    fn send_sync<T: Send + Sync>() {}
    send(&async move { waiter.read(&budget).await });
    send(&async move { read.build_turn().await });
    send_sync::<SnapshotFlights>();
    send_sync::<BuildTurn>();
}

#[cfg(test)]
mod tests {
    //! The flight's [`FlightHook`] and its progress top-ups, driven directly
    //! against a real client's gate and budget on a flight registered nowhere
    //! and never driven: a real fenced read of a byte-dense layer, the hook
    //! wrapped to record what it holds at each progress report and before the
    //! release (`storage::test_support::fenced_read_progress`), and single
    //! events. The registry, attach, turns and charging are
    //! `tests/integration/storage/snapshot_flights_test.rs`; catch-ups
    //! through a flight end to end, `subscribe_fenced_catch_up_test.rs`.

    use std::future::Future;
    use std::pin::pin;
    use std::task::Poll;

    use super::*;
    use crate::storage::memory_budget::kib_for_bytes;
    use crate::storage::snapshot_fence::RetryReason;
    use crate::storage::test_support::fenced_read_progress::{
        as_held, assert_accounted_while_reading, lowered_client, posed_rig_layer, read_fenced,
        Held, ObservedHook,
    };
    use crate::storage::test_support::TestValkey;

    /// The bytes `hook` holds for its read: its grant and its one top-up.
    fn hook_memory(hook: &FlightHook<'_>) -> u64 {
        hook.grant.as_ref().map_or(0, MemoryGrant::bytes)
            + hook.top_up.as_ref().map_or(0, MemoryTopUp::bytes)
    }

    /// A flight for `object_id` that is in no map and never driven: what a
    /// [`FlightHook`] needs beside the client. Its read stage is dropped
    /// unpolled, the flight's own clone included.
    fn idle_flight(
        flights: &SnapshotFlights,
        object_id: u64,
        estimate: LayerSize,
    ) -> Arc<FlightSlot> {
        let (flight, stage) = flights.registry.new_flight(object_id, estimate, None);
        drop(stage);
        flight.clear_read_stage();
        flight
    }

    fn flight_hook<'a>(
        registry: &'a Weak<Registry>,
        flight: &'a Arc<FlightSlot>,
        client: &'a ValkeyClient,
    ) -> FlightHook<'a> {
        FlightHook {
            registry,
            flight,
            client,
            permit: None,
            grant: None,
            top_up: None,
            refusal: None,
        }
    }

    /// What the client's fenced gate and budget hold: `(P_f held, reserved
    /// bytes, over-committed bytes)`.
    fn held(client: &ValkeyClient) -> (usize, u64, u64) {
        let gate = &client.catchup_fenced_read_gate;
        (
            gate.limit() - gate.available(),
            client.catchup_build_budget.reserved_bytes(),
            client.catchup_build_budget.overcommit_bytes(),
        )
    }

    /// `bytes` as `client`'s budget grants it: whole KiB, clamped to `T`.
    fn granted(client: &ValkeyClient, bytes: u64) -> u64 {
        let total_kib = client.catchup_build_budget.total_bytes() / 1024;
        kib_for_bytes(bytes).min(total_kib) * 1024
    }

    /// Poll `fut` once.
    async fn poll_once<F: Future>(fut: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        let mut fut = fut;
        std::future::poll_fn(|cx| Poll::Ready(fut.as_mut().poll(cx))).await
    }

    /// A test layer's estimate.
    const ESTIMATE: LayerSize = LayerSize {
        items: 5_000,
        bytes: 80_000,
    };

    /// One fenced read of a posed-rig layer by a flight's hook, from the
    /// estimate a catch-up takes from the layer's `TOO_LARGE` answer (the
    /// fitted bytes per item), observed. Checks [`assert_accounted_while_reading`]
    /// against `client`'s `T`, then that the verified adjust left exactly the
    /// flight's measured weight, and returns what the hook held then and the
    /// end of the read's figure.
    async fn rig_flight_read(tv: &TestValkey, client: &ValkeyClient, layer: &str) -> (u64, u64) {
        let total = client.catchup_build_budget.total_bytes();
        let (oid, answer) = posed_rig_layer(tv, client, layer).await;
        let estimate = LayerSize::from_too_large(&answer, None);
        let flights = SnapshotFlights::new(client.clone(), tv.key_schema().clone());
        let flight = idle_flight(&flights, oid, estimate);
        let registry = Arc::downgrade(&flights.registry);
        let mut hook = ObservedHook::new(
            flight_hook(&registry, &flight, client),
            hook_memory,
            Arc::clone(&client.catchup_build_budget),
        );
        read_fenced(
            client,
            tv,
            oid,
            SnapshotReadConsumer::Catchup,
            estimate,
            &mut hook,
        )
        .await;

        let end = assert_accounted_while_reading(&hook, total);
        let measured = hook.measured.expect("measured");
        let weight = as_held(flight_weight_bytes(measured.bytes), total);
        assert_eq!(
            hook.held(),
            Held {
                hook: weight,
                budget: weight
            },
            "the verified adjust leaves exactly the flight's measured weight: no top-up \
             counted twice"
        );
        assert!(
            hook.inner.permit.is_none(),
            "the permit given back after it"
        );
        drop(hook);
        assert_eq!(held(client), (0, 0, 0), "dropping the hook frees all");
        (weight, end)
    }

    /// R1 — a byte-dense layer (posed-rig shaped: few items, large inline
    /// arrays, above the item ceiling) under-sizes the flight's grant from
    /// the fitted bytes per item, so as the read goes the hook tops the grant
    /// up to `SNAPSHOT_MEMORY_FACTOR` × the bytes read so far: at the end of
    /// the read, before the adjust, the read holds at least that factor of
    /// its measured bytes, all of it accounted, where the grant alone was
    /// below it; the verified adjust then leaves the snapshot-plus-build
    /// weight in one grant and one top-up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_byte_dense_flight_reads_grant_is_topped_up_to_the_snapshot_factor_of_the_bytes_read()
    {
        let tv = TestValkey::setup().await;
        let client = lowered_client(&tv, None);
        Box::pin(rig_flight_read(&tv, &client, "rig-flight")).await;
    }

    /// R3b — the cumulative clamp: with a `T` well below the dense layer's
    /// weight, the grant and its top-up never exceed `T` at any report and
    /// equal it by the end of the read, and the adjust clamps the same way.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_progress_top_ups_never_take_grant_and_top_up_past_a_small_t_and_reach_it() {
        const SMALL_T_KIB: u64 = 1024;
        let tv = TestValkey::setup().await;
        let client = lowered_client(&tv, Some(SMALL_T_KIB));
        let (weight, end) = Box::pin(rig_flight_read(&tv, &client, "rig-flight-small-t")).await;
        assert_eq!(
            end,
            SMALL_T_KIB * 1024,
            "the end of the read holds T exactly"
        );
        assert_eq!(weight, SMALL_T_KIB * 1024, "and so does the adjust");
    }

    /// R3a — a progress report's top-up never waits: with the rest of `T`
    /// reserved by another grant and a reservation queued behind it, the
    /// report (synchronous) returns having taken the top-up past the total,
    /// and the queued reservation is still queued.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_flight_progress_top_up_never_waits_on_a_fully_reserved_budget() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let budget = &client.catchup_build_budget;
        let flights = SnapshotFlights::new(client.clone(), tv.key_schema().clone());
        let flight = idle_flight(&flights, 7, ESTIMATE);
        let registry = Arc::downgrade(&flights.registry);
        let mut hook = flight_hook(&registry, &flight, client);
        hook.before_attempt(FenceAttempt {
            attempt: 1,
            size: ESTIMATE,
        })
        .await
        .expect("free capacity is taken");
        let grant = granted(client, flight_grant_bytes(ESTIMATE.bytes));
        assert_eq!(held(client), (1, grant, 0), "P_f and the combined grant");
        let others = budget
            .try_grant(budget.total_bytes() - grant)
            .expect("fixture: the rest of T");
        let mut queued = pin!(budget.reserve(1024, None));
        assert!(poll_once(queued.as_mut()).await.is_pending(), "T is full");

        let bytes_read = ESTIMATE.bytes * 3;
        hook.on_read_progress(bytes_read);

        let top_up = granted(client, bytes_read * SNAPSHOT_MEMORY_FACTOR) - grant;
        assert!(top_up > 0, "fixture: the report needs a top-up");
        assert_eq!(
            held(client),
            (1, budget.total_bytes(), top_up),
            "taken at once, past a full T"
        );
        assert!(
            poll_once(queued.as_mut()).await.is_pending(),
            "the queued reservation is still queued"
        );
        assert_eq!(budget.waiters(), 1);
        drop(others);
    }

    /// The `RETRY_AFTER` path gives memory back with its top-up: a broken
    /// attempt keeps the grant and the progress top-up for its retry, and a
    /// `RETRY_AFTER` on that retry's acquire gives both back, and the permit,
    /// before its sleep.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_retry_after_gives_a_kept_progress_top_up_back_with_the_grant() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let flights = SnapshotFlights::new(client.clone(), tv.key_schema().clone());
        let flight = idle_flight(&flights, 7, ESTIMATE);
        let registry = Arc::downgrade(&flights.registry);
        let mut hook = flight_hook(&registry, &flight, client);
        hook.before_attempt(FenceAttempt {
            attempt: 1,
            size: ESTIMATE,
        })
        .await
        .expect("free capacity is taken");
        hook.on_read_progress(ESTIMATE.bytes * 3);
        hook.after_release(FenceReleased {
            verdict: Some(crate::storage::snapshot_fence::ReleaseVerdict::Broken),
            outcome: FenceAttemptOutcome::Broken,
            measured: Some(ESTIMATE),
        })
        .await;
        let kept = held(client);
        assert!(
            kept.2 > 0,
            "a broken attempt keeps its top-up for the retry"
        );
        hook.before_attempt(FenceAttempt {
            attempt: 2,
            size: ESTIMATE,
        })
        .await
        .expect("the retry's attempt");
        assert_eq!(held(client), kept, "the retry takes nothing more");

        let mut sleeping = pin!(hook.on_retry_after(FenceRetry {
            hint: Duration::from_secs(3_600),
            reason: RetryReason::Epoch,
            size: ESTIMATE,
        }));
        assert!(poll_once(sleeping.as_mut()).await.is_pending(), "it sleeps");
        assert_eq!(held(client), (0, 0, 0), "nothing held through the sleep");
    }
}
