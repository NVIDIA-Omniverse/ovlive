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

//! The presence fan-out's memory (SDD §13.4, §21.4): a session's fan-out
//! memory follows its undelivered events, at most the capacity, and does not
//! depend on the capacity while it is idle.
//!
//! A dedicated binary because it installs a counting `#[global_allocator]`,
//! which applies to the whole binary — in `integration` it would count every
//! other test's allocations too. It wraps the system allocator and counts the
//! bytes Rust requests, net of frees; glibc's own per-block overhead is not
//! counted. No Valkey: the router is driven directly, through the
//! `test-support` `subscribe`, and each measured span opens after its inputs
//! are built.
//!
//! **Assertions**, each against a bound derived from the measurements
//! (a local measurement tool outside the published source, in jemalloc's
//! usable sizes: about
//! 357 B per idle session of 5 receivers at capacity 4096 and 356 B at
//! 1<<20, 112 B per undelivered event, 128 B
//! of slots kept after a drain; this binary, in requested bytes: about
//! 310 B, 104 B and 128 B, and about 394 B for a lone session, which pays
//! for the router table's first allocation alone. The fan-out's record of
//! the newest removed join or leave added about 32 B to the idle figure,
//! from 278 B):
//!
//! - an idle session with 5 receivers holds at most
//!   [`IDLE_BOUND_PER_SESSION`] bytes at the default capacity (a per-session
//!   ring preallocated to the capacity held about 393 KB);
//! - its idle heap is the same at capacity 4096 and 1<<20, within
//!   [`CAPACITY_TOLERANCE_PER_SESSION`]: the router's `DashMap` places
//!   sessions by a random hash, so its table growth differs a little between
//!   two routers, and only that may differ. One session at 1<<20 is first
//!   screened against the idle bound, so a capacity-sized reservation fails
//!   on one session's allocation rather than 64 sessions';
//! - a burst of [`BURST`] events one receiver leaves unread grows the heap by
//!   at most one shared event and two slots per event, and once that receiver
//!   drains, only the retained slots are left.
//!
//! Every test takes [`MEASURE`] first, so under `cargo test` (one process,
//! parallel threads) no two measure at once; under nextest each runs in its
//! own process anyway.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Mutex;

use futures::FutureExt;
use live_edit::presence::{
    LocalPresenceRouter, PresenceEvent, PresenceReceiver, PresenceRecvError,
    DEFAULT_BROADCAST_CHANNEL_CAPACITY,
};

// ═════════════════════════════════════════════════════════════════════════════
// The counting allocator
// ═════════════════════════════════════════════════════════════════════════════

/// Bytes allocated and not yet freed, process-wide.
static LIVE: AtomicIsize = AtomicIsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates the counter on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded verbatim; the caller upholds `alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::SeqCst);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::SeqCst);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as `alloc`; `ptr` came from this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as `dealloc`.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_add(new_size as isize - layout.size() as isize, Ordering::SeqCst);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::SeqCst)
}

/// Serializes the measuring tests within one process.
static MEASURE: Mutex<()> = Mutex::new(());

fn measure_lock() -> std::sync::MutexGuard<'static, ()> {
    MEASURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ═════════════════════════════════════════════════════════════════════════════
// Bounds
// ═════════════════════════════════════════════════════════════════════════════

/// Local receivers per session: the Engineer's measured shape.
const RECEIVERS: usize = 5;

/// Most heap an idle session with [`RECEIVERS`] receivers may hold: the
/// measured ~310 B per session over many sessions, ~394 B for a lone one
/// (its router entry, its fan-out queue with no slot, its key, and its share
/// of the router's table), with at least a quarter again as margin. A ring
/// preallocated to the default capacity is ~1,000× this.
const IDLE_BOUND_PER_SESSION: f64 = 512.0;

/// Sessions the idle bound is measured over, so the router's table growth
/// is spread thin.
const IDLE_SESSIONS: usize = 1000;

/// Sessions the capacity comparison is measured over: enough to keep the
/// router's table growth within [`CAPACITY_TOLERANCE_PER_SESSION`]. The
/// comparison runs only after one session at [`LARGE_CAPACITY`] has passed
/// [`IDLE_BOUND_PER_SESSION`]. A defect that reserves memory by the capacity
/// therefore fails on that one session's allocation, with the assertion
/// message, and never asks for 64 sessions' worth: a ring preallocated to
/// the capacity, about 96 B a slot, is about 100 MB per session at 1<<20,
/// and 64 of them would be about 6.4 GB.
const COMPARE_SESSIONS: usize = 64;

/// The largest capacity compared: 256× the default.
const LARGE_CAPACITY: usize = 1 << 20;

/// How much the idle heap per session may differ between two capacities:
/// the router's `DashMap` table growth, which depends on where a random hash
/// puts each session (measured about 2.6 B per session over 1,000 sessions;
/// a few times that over [`COMPARE_SESSIONS`]). A per-session reservation
/// that scales with the capacity differs by megabytes.
const CAPACITY_TOLERANCE_PER_SESSION: f64 = 64.0;

/// One slot of the fan-out's array: the shared event's pointer and its count
/// of receivers still to read it.
const SLOT_BYTES: usize = 2 * size_of::<usize>();

/// Slots a drained queue keeps (`presence::fanout::RETAINED_SLOTS`, which is
/// crate-private).
const RETAINED_SLOTS: usize = 8;

/// Undelivered events in the burst: under the capacity, and not a power of
/// two, so the slot array's doubling leaves some slots spare.
const BURST: usize = 1000;

/// Most heap one undelivered event may add: its shared allocation (two
/// reference counts and the event) and two slots, the most the array's
/// doubling can leave per queued event. The burst's events own no heap of
/// their own, so this is the queue's cost alone (measured 104 B in requested
/// bytes, 112 B in jemalloc's usable sizes).
const PER_EVENT_BOUND: usize = 2 * size_of::<usize>() + size_of::<PresenceEvent>() + 2 * SLOT_BYTES;

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The net heap per session of `sessions` idle sessions with [`RECEIVERS`]
/// receivers each, on a fresh router of `capacity`. The receivers are held in
/// a vector allocated before the span, so only what the router holds counts.
fn idle_heap_per_session(capacity: usize, sessions: usize) -> f64 {
    let router = LocalPresenceRouter::with_capacity(capacity);
    let ids: Vec<String> = (0..sessions)
        .map(|i| format!("presence-memory-{i:06}"))
        .collect();
    let mut held: Vec<PresenceReceiver> = Vec::with_capacity(sessions * RECEIVERS);

    let before = live();
    for id in &ids {
        for _ in 0..RECEIVERS {
            held.push(router.subscribe(id).0);
        }
    }
    let after = live();

    assert_eq!(router.session_count(), sessions, "fixture self-check");
    (after - before) as f64 / sessions as f64
}

/// Fail unless `per_session`, measured at `capacity`, is within
/// [`IDLE_BOUND_PER_SESSION`].
fn assert_within_idle_bound(per_session: f64, capacity: usize) {
    assert!(
        per_session <= IDLE_BOUND_PER_SESSION,
        "an idle session with {RECEIVERS} receivers holds {per_session:.1} B at capacity \
         {capacity}, past the {IDLE_BOUND_PER_SESSION} B bound: the fan-out reserves memory \
         before any event is sent"
    );
}

/// A `PeerMessage` that owns no heap: an empty id and payload allocate
/// nothing, so a burst of them measures the queue's own cost.
fn heapless_message() -> PresenceEvent {
    PresenceEvent::PeerMessage {
        connection_id: String::new(),
        data: Vec::new(),
    }
}

/// Read everything queued for `rx`, a lag included, until none is left. A
/// read takes up to a batch of events ahead into the receiver, which keeps
/// them alive until returned; reading to the end leaves that buffer empty,
/// so no measurement after a drain counts events a receiver still holds.
fn drain(rx: &mut PresenceReceiver) -> usize {
    let mut read = 0;
    while let Some(result) = rx.recv().now_or_never() {
        match result {
            Ok(_) => read += 1,
            Err(PresenceRecvError::Lagged { .. }) => {}
            Err(PresenceRecvError::Closed) => break,
        }
    }
    read
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

/// An idle session holds a few hundred bytes at the default capacity, not a
/// buffer sized by it.
#[test]
fn idle_session_with_five_receivers_holds_at_most_the_idle_bound() {
    let _serial = measure_lock();
    let per_session = idle_heap_per_session(DEFAULT_BROADCAST_CHANNEL_CAPACITY, IDLE_SESSIONS);
    eprintln!(
        "== idle: {per_session:.1} B per session ({IDLE_SESSIONS} sessions x {RECEIVERS} \
         receivers, capacity {DEFAULT_BROADCAST_CHANNEL_CAPACITY}); bound \
         {IDLE_BOUND_PER_SESSION} B =="
    );
    assert_within_idle_bound(per_session, DEFAULT_BROADCAST_CHANNEL_CAPACITY);
}

/// Raising the capacity 256-fold costs an idle session nothing. A single
/// session at [`LARGE_CAPACITY`] is screened against the idle bound first,
/// so a capacity-sized reservation fails there on one session's allocation
/// (see [`COMPARE_SESSIONS`]).
#[test]
fn idle_session_heap_does_not_depend_on_the_capacity() {
    let _serial = measure_lock();
    let screened = idle_heap_per_session(LARGE_CAPACITY, 1);
    eprintln!(
        "== idle screen: {screened:.1} B for one session at capacity {LARGE_CAPACITY}; bound \
         {IDLE_BOUND_PER_SESSION} B =="
    );
    assert_within_idle_bound(screened, LARGE_CAPACITY);
    let at_default = idle_heap_per_session(DEFAULT_BROADCAST_CHANNEL_CAPACITY, COMPARE_SESSIONS);
    let at_large = idle_heap_per_session(LARGE_CAPACITY, COMPARE_SESSIONS);
    eprintln!(
        "== idle by capacity: {at_default:.1} B per session at \
         {DEFAULT_BROADCAST_CHANNEL_CAPACITY}, {at_large:.1} B at {LARGE_CAPACITY} \
         ({COMPARE_SESSIONS} sessions); tolerance {CAPACITY_TOLERANCE_PER_SESSION} B =="
    );
    assert!(
        (at_large - at_default).abs() <= CAPACITY_TOLERANCE_PER_SESSION,
        "an idle session holds {at_default:.1} B at capacity {DEFAULT_BROADCAST_CHANNEL_CAPACITY} \
         but {at_large:.1} B at {LARGE_CAPACITY}: its memory depends on the capacity"
    );
}

/// A burst one receiver leaves unread costs at most one shared event and two
/// slots per event; once that receiver drains, the session is back to the
/// retained slots alone. The burst is of messages: a queue that has removed a
/// join or leave also keeps the newest one's connection id (for
/// `membership_lost`), which messages never leave behind, so the drained
/// bound stays the slots alone.
#[test]
fn burst_grows_by_at_most_the_per_event_bound_and_drains_to_the_retained_slots() {
    const SESSION: &str = "presence-memory-burst";
    let _serial = measure_lock();
    let router = LocalPresenceRouter::with_capacity(DEFAULT_BROADCAST_CHANNEL_CAPACITY);
    let mut receivers: Vec<PresenceReceiver> = (0..RECEIVERS)
        .map(|_| router.subscribe(SESSION).0)
        .collect();
    let mut events: Vec<PresenceEvent> = (0..BURST).map(|_| heapless_message()).collect();
    let (slow, readers) = receivers
        .split_first_mut()
        .expect("fixture self-check: receivers");

    let before = live();
    for event in events.drain(..) {
        router.broadcast(SESSION, event);
    }
    for rx in readers.iter_mut() {
        assert_eq!(
            drain(rx),
            BURST,
            "fixture self-check: a reader reads the burst"
        );
    }
    let after_burst = live();
    let unread_slots = router.fanout_allocated_slots(SESSION);
    assert_eq!(drain(slow), BURST, "fixture self-check: no event was lost");
    let after_drain = live();
    let drained_slots = router.fanout_allocated_slots(SESSION);

    let per_event = (after_burst - before) as f64 / BURST as f64;
    let kept = after_drain - before;
    eprintln!(
        "== burst: {per_event:.1} B per undelivered event ({BURST} events, {unread_slots:?} \
         slots; bound {PER_EVENT_BOUND} B); after the drain {kept} B kept ({drained_slots:?} \
         slots; bound {} B) ==",
        RETAINED_SLOTS * SLOT_BYTES
    );
    assert!(
        per_event <= PER_EVENT_BOUND as f64,
        "{BURST} undelivered events cost {per_event:.1} B each, past {PER_EVENT_BOUND} B (one \
         shared event and two slots)"
    );
    assert!(
        kept <= (RETAINED_SLOTS * SLOT_BYTES) as isize,
        "after every receiver drained, the session still holds {kept} B more than before the \
         burst ({drained_slots:?} slots), past the {RETAINED_SLOTS} retained slots"
    );
}
