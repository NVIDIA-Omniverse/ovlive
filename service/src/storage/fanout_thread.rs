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

//! The snapshot fan-out's dedicated threads: `fanout-read`, which polls every
//! fenced snapshot fan-out pass; the [`WATCH_READ_THREADS`] `watch-read-<i>`
//! threads, which poll every unfenced one; and `snapshot-io`, which hosts the
//! Valkey routers of the snapshot lane.
//!
//! # What runs here
//!
//! The bounded snapshot fan-out (`storage::diff_read`) runs as passes, each
//! spawned by [`FanoutThreads::run_pass`] with owned clones of its inputs on
//! the thread its [`PassLane`] names. Nothing else is spawned on those
//! threads: a new fan-out reader goes through `run_pass`.
//!
//! - **`fanout-read`** ([`PassLane::Fenced`]) polls the fenced pass of every
//!   fenced read (`storage::snapshot_fence::read` — catch-up flights,
//!   structural cache cold starts and audits, checkpoint reads).
//! - **`watch-read-0` … `watch-read-3`** ([`PassLane::Unfenced`]) poll the
//!   unfenced passes: the structural update's charged WATCH read
//!   (`storage::diff_read`'s `read_object_snapshot_unfenced`) and
//!   external-value clean-up's reference walk
//!   (`storage::retirement::walk_object_references`). Each
//!   pass is assigned one of the threads when it starts (*Assignment*,
//!   below) and stays on it to its end.
//! - **`snapshot-io`** hosts the router task of every snapshot lane member
//!   (`ValkeyClient::snapshot_pool`): started on it at connect
//!   (`MemberRouters`), by a router supervisor restart and by a member
//!   replacement ([`connect_router_on`]). The routers parse the replies the
//!   passes read. No other pool's router runs here.
//!
//! Each is a tokio runtime of one worker thread
//! (`Builder::new_multi_thread().worker_threads(1)`), so it drives itself:
//! nothing ever blocks a thread to run it, and a task spawned on it never
//! moves to another thread. All [`FANOUT_RUNTIMES`] are owned by one
//! [`FanoutThreads`] behind an `Arc` on `ValkeyClient`, built in
//! `ValkeyClient::connect`, the client every test builds included.
//!
//! # Why: the allocator keeps what a moving read freed
//!
//! A plain fan-out window — every window of a whole-row read, and the
//! structural WATCH read's round 1, lone long children lists and windows after
//! a script fallback — builds the per-command future of each of its up to
//! 2,141 commands at once (`try_join_all`); a script window of the structural
//! WATCH read holds one future for the whole window. On the build the diagnosis
//! measured (the record below), whose whole-row read still carried the
//! value-row read's per-command future, a release build's heap swung
//! 25.1–26.4 MiB per window (median per arm, cold start and catch-up). The
//! whole-row read now has a per-command future of its own, about half that
//! size (7,216 B against 13,768 B in a debug build: the `issue_window_command`
//! future around `storage::diff_read`'s whole-row issuer,
//! `issue_fanout_command`, against the same around the one issuer with the
//! value-row read inline); its release swing per window is not measured.
//! Polled on the service's runtime, the read task runs on whichever worker the
//! replying router woke, so successive windows allocate on different worker
//! threads. jemalloc gives each worker thread its own arena; freed pages
//! return to the arena that allocated them and stay dirty until that arena's
//! background thread purges them, and only the arena a thread is allocating in
//! has its purge woken early. So a read that moved across N workers leaves
//! dirty pages in up to N arenas, and its resident memory beyond its live heap
//! grew with the worker count — that is, with the pod CPU limit, a customer
//! setting. A 10x real-id structural cache cold start measured X (peak
//! resident growth less its grant) = −42.9 + 43.6 MiB × workers, its heap flat
//! at +51–56 MiB; with one arena the same read held 10.5–28.0 MiB. Here every
//! pass and every reply frame allocates on one of a fixed set of threads
//! whatever the CPU limit, and in the service binary every one of those
//! threads allocates from one arena (*One arena*, below), so the window term
//! `X` that the catch-up memory budget's derivation funds
//! (`storage::memory_budget`) grows neither with the worker count nor with the
//! number of fan-out threads.
//! Measurements: `docs-internal/validation/fanout-executor-b26e-s2-20261009/`.
//!
//! # One arena: the binary's start hook
//!
//! A fixed set of threads alone still left a slack per thread: with each
//! fan-out thread in an arena of its own, four concurrent structural WATCH
//! writers' reads, spread over the four `watch-read` threads, left their
//! freed pages dirty in four arenas until each arena's 10 s decay purged
//! them. At 2 CPUs a release build's peak resident growth beyond the memory
//! its budgets held at that peak exceeded the window term in every round
//! (225.2–330.8 MiB), and it grew with the CPUs, where more WATCH reads
//! overlapped [measured; the per-arena attribution inferred].
//!
//! So in the service binary every thread of the [`FANOUT_RUNTIMES`] runtimes
//! — `fanout-read`, the `watch-read` threads, `snapshot-io`, and any
//! blocking-pool thread one of them starts — allocates from one dedicated
//! jemalloc arena whose freed pages are purged after a short decay rather
//! than jemalloc's default 10 s: the slack is one arena's, purged promptly,
//! whatever the thread count or the CPUs. The arena, its decay and the reason
//! for that value are the binary's (`src/allocator.rs`, *The fan-out arena*);
//! the measured basis is the record above.
//!
//! The binding also moves the threads' largest buffers. On jemalloc's
//! automatic arenas every allocation of `oversize_threshold` (8 MiB) or more
//! comes from jemalloc's huge arena, which purges eagerly (decay 0); a thread
//! bound to a manual arena takes them from that arena instead (the vendored
//! jemalloc 5.3.0: `arena_choose_maybe_huge`, `arena_inlines_b.h:21-39`;
//! the huge arena's decay, `arena.c:1725-1738`). Those buffers' freed pages
//! now wait out the fan-out arena's decay, and the measurement that chose it
//! includes them.
//!
//! The allocator is the binary's alone, so this library never calls it.
//! Each thread runs, as it starts and before it polls anything, the
//! [`FanoutThreadStartHook`] the binary registered
//! ([`register_fanout_thread_start_hook`]) before its Valkey client
//! connected; that hook binds the thread to the arena. The runtimes take
//! the hook when they are built, in `ValkeyClient::connect`: a runtime built
//! before the registration never runs it, on any thread it ever starts. With
//! no hook registered — the library's test binaries, a target whose binary
//! has no jemalloc, any other library consumer, or a binary whose arena
//! could not be created — a thread starts with nothing run and allocates
//! from the arena its allocator gives any new thread.
//!
//! # Why these threads, and why constants
//!
//! Every count here is a compiled constant, never a setting: a knob would
//! make the bound depend on a customer value again.
//!
//! - **`fanout-read` and `snapshot-io` apart.** With the pass and the
//!   routers on one thread a 10x cold start's fence hold grew from 1.27–1.38
//!   s to 1.65–1.69 s: the routers' reply parsing overlaps usefully with the
//!   pass's conversion and decode.
//! - **One thread for the fenced passes.** A fenced pass is round-trip bound,
//!   and at most `P_f` budgeted fenced reads (plus a checkpoint read) run at
//!   once; they interleave on `fanout-read`, each slower when they share it.
//! - **[`WATCH_READ_THREADS`] = 4 for the unfenced passes.** An unfenced pass
//!   on plain windows is CPU bound on its thread: a 4x structural WATCH read
//!   whose windows are plain (the read after a script fallback) costs about
//!   0.62–0.65 s of its thread's CPU, so one thread completes about 1.5 such
//!   reads a second whatever the replica's CPUs (measured, §3.1 of
//!   `docs-internal/validation/fanout-executor-b26e-s2-20261009/report.md`);
//!   read through window scripts, as it normally is, the same read costs
//!   about 0.09 s of that thread and waits on its round trips instead
//!   (modelled: the plain figure scaled by the flattened Kitchen_set's
//!   script-to-plain `watch-read` CPU, 0.43 / 3.13 s, M2 of
//!   `docs-internal/validation/watch-window-script-b-gvf-20261010/`). On
//!   `fanout-read` beside the fenced passes, four concurrent WATCH writers'
//!   latency (each read then plain) rose by half at 2 CPUs and tripled at 6,
//!   and a fenced catch-up queued behind them (its grant hold 3.3 → 4.6 s at
//!   2 CPUs). So the unfenced passes have threads of
//!   their own, and the fenced passes never queue behind them. Four lets four
//!   unfenced passes progress at once, about four times one thread's
//!   throughput where the replica has the CPUs for them. In arenas of their
//!   own the four left four arenas' slack, over the window term (*One
//!   arena*, above); in the binary's one fan-out arena they leave that
//!   arena's, the same whatever their number.
//!
//! # Assignment: least-loaded
//!
//! An unfenced pass goes to the `watch-read` thread with the fewest unfenced
//! passes in flight, the lowest index on a tie ([`FanoutThreads::run_pass`]).
//! The choice and the count's increment are one step under a lock, and the
//! count falls when the pass's future is dropped — completed, aborted, or
//! ended by the runtime's shutdown — so with at most [`WATCH_READ_THREADS`]
//! unfenced passes in flight no two share a thread. Not round-robin: passes
//! differ in length (a 10x read takes about 2.5 times a 4x one, an ext-GC
//! reference walk runs beside the writers), so a rotating index sends a new pass to a
//! thread still busy with a long one while another thread is idle — the
//! queueing these threads exist to remove. The lowest index on a tie keeps
//! the passes on as few threads as their concurrency needs. A pass never
//! moves: its thread's runtime has one worker, so every allocation it makes
//! comes from that thread's arena — in the service binary, the one fan-out
//! arena (*One arena*, above).
//!
//! # The handshake: the caller's seams answered in order
//!
//! A pass reaches its caller at the seams it had when it ran inline, each
//! now a [`PassRequest`] over a channel of one, answered by the caller loop
//! of [`FanoutThreads::run_pass`] before the pass goes on: the fenced read's
//! progress report after each window (the caller's hook tops its memory
//! grant up), and the charged WATCH read's allowance seams (WA1's count
//! charge, WA2's growth before and after each window and index page, and the
//! retained name bytes at the end). The pass issues nothing until the answer
//! arrives, so a report is applied before the next window is issued and a
//! refused growth stops the read before the window it would have covered,
//! exactly as when the seams were synchronous calls. The caller's state —
//! the fence hook, the commit charge — never leaves the caller's task.
//!
//! # Cancellation, panics, shutdown
//!
//! - **Cancellation.** The spawned pass is held by an abort-on-drop guard:
//!   dropping the caller's future (an RPC deadline, a client cancel, a lost
//!   fence lease, graceful shutdown) aborts the pass at its next await, and
//!   its window memory is freed on the pass's own thread. A pass waiting on
//!   an answer whose caller has gone waits there until that abort. An
//!   aborted unfenced pass leaves its `watch-read` thread's in-flight count
//!   with its future's drop.
//! - **Panics.** A panic in the pass is re-raised on the caller
//!   (`std::panic::resume_unwind`), so a catch-up flight's `catch_unwind`
//!   still sees it.
//! - **Tracing.** The pass runs inside the caller's span and emits to the
//!   caller's default subscriber, so the per-window `fan-out window` debug
//!   event and trace export keep its context, as when it ran inline.
//! - **Shutdown: the threads live as long as the client's owners.** An
//!   owner is a `ValkeyClient` clone held from outside the client: the
//!   service's, its loops' and handlers', a pass's, a test's. When the last
//!   owner drops, quit or not, `ValkeyClient`'s `ClientOwners` shuts every
//!   one of the [`FANOUT_RUNTIMES`] runtimes down at once
//!   ([`FanoutThreads::shut_down`]) with `Runtime::shutdown_background`,
//!   which never blocks the dropping thread, whatever thread that is; every
//!   task on them, a pass or a snapshot member's router, is dropped. At process exit that is after
//!   `ValkeyClient::quit`, whose `QUIT`s the `snapshot-io` routers serve:
//!   `main.rs` quits through an owner it still holds. The clones the
//!   client's own machinery keeps — the `txn` slots' router restart, held by
//!   the member replacer and by each `txn` slot's router watcher — are not
//!   owners (`ValkeyClient::internal_clone`): as owners they would close a
//!   cycle through the client and keep the threads, the pools, the router
//!   supervisor and the fred clients alive for the rest of the process once
//!   the last real owner had gone. The same drop releases the member
//!   replacer and the router supervisor, which lets those internal clones
//!   go. A caller of `run_pass` holds an owner, so the runtimes outlive
//!   every pass they run; a pass a runtime nonetheless cancelled would
//!   answer `ServiceError::Unavailable` (retryable).

use std::future::Future;
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::AtomicUsize;
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use fred::interfaces::ClientLike;
use fred::types::ConnectHandle;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::instrument::WithSubscriber;
use tracing::Instrument;

use crate::error::ServiceError;

/// The name of the thread that polls every fenced snapshot fan-out pass.
pub const FANOUT_READ_THREAD_NAME: &str = "fanout-read";

/// The name of the thread that hosts the snapshot lane's Valkey routers.
pub const SNAPSHOT_IO_THREAD_NAME: &str = "snapshot-io";

/// The `watch-read` threads, which poll every unfenced pass (the structural
/// update's WATCH read and the ext-GC reference walk): a compiled
/// constant, never a setting, and four for the reasons the module docs give
/// (*Why these threads, and why constants*).
pub const WATCH_READ_THREADS: usize = 4;

/// The prefix of each `watch-read` thread's name: thread `i` is
/// `watch-read-<i>` ([`watch_read_thread_name`]).
pub const WATCH_READ_THREAD_NAME_PREFIX: &str = "watch-read";

/// Every runtime [`FanoutThreads`] owns, each of one worker thread:
/// `fanout-read`, `snapshot-io` and the [`WATCH_READ_THREADS`] `watch-read`
/// threads.
pub const FANOUT_RUNTIMES: usize = 2 + WATCH_READ_THREADS;

/// The name of `watch-read` thread `index`: `watch-read-<index>`.
pub fn watch_read_thread_name(index: usize) -> String {
    format!("{WATCH_READ_THREAD_NAME_PREFIX}-{index}")
}

/// The stack of each of the fan-out threads: 2 MiB, the stack every worker
/// of the service's own runtime has (Rust's default thread stack, which
/// `main.rs`'s `#[tokio::main]` leaves as it is), and the stack the release
/// stack guard (`tests/release_stack_guard`, SDD §15.1) measures a pass's
/// depth against. Explicit, so no thread inherits a larger
/// `RUST_MIN_STACK` and hides a pass that outgrew the workers' stack.
pub const FANOUT_THREAD_STACK_BYTES: usize = 2 * 1024 * 1024;

/// What every thread of the [`FANOUT_RUNTIMES`] runtimes runs as it starts,
/// before it polls anything: the service binary binds the thread to the
/// fan-out's one dedicated allocator arena (module docs, *One arena*). A plain
/// `fn`, so it carries no state and the library needs no allocator API.
pub type FanoutThreadStartHook = fn();

/// The hook [`register_fanout_thread_start_hook`] registered, if any.
static FANOUT_THREAD_START_HOOK: OnceLock<FanoutThreadStartHook> = OnceLock::new();

/// Register the hook every fan-out thread runs as it starts (module docs,
/// *One arena*). Set once per process: a second registration is refused and
/// returns the hook it carried.
///
/// The cutoff is when the runtimes are built, not when a thread starts:
/// `FanoutThreads::new`, run by `ValkeyClient::connect`, reads the
/// registered hook once and gives it to every runtime it builds. Runtimes
/// built before the registration never run it, including blocking-pool
/// threads they start after it. The service binary registers it at startup,
/// before its Valkey client connects, so every fan-out thread the service
/// starts runs it.
///
/// Without a registration the fan-out threads start with no hook — the
/// library's test binaries, a target whose service binary has no jemalloc,
/// any other consumer of the library — and allocate from whatever arena
/// their allocator gives a new thread.
pub fn register_fanout_thread_start_hook(
    hook: FanoutThreadStartHook,
) -> Result<(), FanoutThreadStartHook> {
    FANOUT_THREAD_START_HOOK.set(hook)
}

/// The registered hook, or `None` when nothing registered one.
fn registered_fanout_thread_start_hook() -> Option<FanoutThreadStartHook> {
    FANOUT_THREAD_START_HOOK.get().copied()
}

/// Which thread a fan-out pass runs on ([`FanoutThreads::run_pass`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassLane {
    /// A fenced read's pass (`storage::snapshot_fence::read`): on
    /// `fanout-read`.
    Fenced,
    /// An unfenced pass — the structural update's WATCH read
    /// (`storage::diff_read`'s `read_object_snapshot_unfenced`) or the ext-GC
    /// reference walk (`storage::retirement::walk_object_references`): on the
    /// least-loaded `watch-read` thread.
    Unfenced,
}

/// Unfenced passes in flight per `watch-read` thread: the assignment's load
/// (module docs, *Assignment*).
type WatchReadLoad = Arc<Mutex<[usize; WATCH_READ_THREADS]>>;

/// The runtimes of the snapshot fan-out (module docs), owned by an `Arc` on
/// `ValkeyClient` and shared by its clones.
pub struct FanoutThreads {
    /// `fanout-read`, `snapshot-io`, then `watch-read-0` …, in that order;
    /// `None` once [`Self::shut_down`] or the drop has shut them all down.
    runtimes: Mutex<Option<Vec<Runtime>>>,
    fanout_read_handle: Handle,
    snapshot_io_handle: Handle,
    watch_read_handles: [Handle; WATCH_READ_THREADS],
    watch_read_load: WatchReadLoad,
    /// Passes spawned and not yet dropped (a pass is dropped when it
    /// completes, when it is aborted, or when its runtime shuts down), per
    /// lane: fenced, unfenced. Test-only observability.
    #[cfg(any(test, feature = "test-support"))]
    live_passes: [Arc<AtomicUsize>; 2],
    /// Passes ever spawned, per lane: fenced, unfenced. Test-only
    /// observability.
    #[cfg(any(test, feature = "test-support"))]
    passes_started: [AtomicUsize; 2],
    /// Unfenced passes ever assigned to each `watch-read` thread. Test-only
    /// observability.
    #[cfg(any(test, feature = "test-support"))]
    watch_read_passes_started: [AtomicUsize; WATCH_READ_THREADS],
    /// Threads of any of the runtimes (its worker and any blocking-pool
    /// thread) that have started and not yet stopped. Test-only
    /// observability.
    #[cfg(any(test, feature = "test-support"))]
    alive_threads: Arc<AtomicUsize>,
}

impl std::fmt::Debug for FanoutThreads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FanoutThreads")
            .field("fanout_read", &FANOUT_READ_THREAD_NAME)
            .field("snapshot_io", &SNAPSHOT_IO_THREAD_NAME)
            .field("watch_read_threads", &WATCH_READ_THREADS)
            .finish()
    }
}

/// The index of a [`PassLane`] in the per-lane test counters.
#[cfg(any(test, feature = "test-support"))]
fn lane_index(lane: PassLane) -> usize {
    match lane {
        PassLane::Fenced => 0,
        PassLane::Unfenced => 1,
    }
}

impl FanoutThreads {
    /// Build every runtime, each starting its one worker thread. Fails only
    /// when the OS refuses a thread or the I/O driver cannot be created;
    /// the runtimes built before the failure are then shut down in the
    /// background, never dropped plainly (this runs inside `connect`, an
    /// async context, where a plain runtime drop panics).
    ///
    /// Every thread of every runtime runs, as it starts, the hook
    /// [`register_fanout_thread_start_hook`] had registered when this ran, if
    /// any: the hook is read here, once, so one registered later never runs
    /// on these runtimes' threads (module docs, *One arena*).
    pub(crate) fn new() -> std::io::Result<Self> {
        Self::with_thread_start_hook(registered_fanout_thread_start_hook())
    }

    /// [`Self::new`] with the start hook given rather than the registered
    /// one: `None` starts every thread with no hook.
    fn with_thread_start_hook(start_hook: Option<FanoutThreadStartHook>) -> std::io::Result<Self> {
        #[cfg(any(test, feature = "test-support"))]
        let alive_threads = Arc::new(AtomicUsize::new(0));
        let build = |name: String| {
            let mut builder = tokio::runtime::Builder::new_multi_thread();
            builder
                .worker_threads(1)
                .thread_name(name)
                .thread_stack_size(FANOUT_THREAD_STACK_BYTES)
                .enable_all();
            // Runs on the new thread before it polls anything: its worker,
            // and any blocking-pool thread the runtime starts later. The
            // hook runs first, so a thread counted alive has run it.
            #[cfg(any(test, feature = "test-support"))]
            let started = Arc::clone(&alive_threads);
            builder.on_thread_start(move || {
                if let Some(hook) = start_hook {
                    hook();
                }
                #[cfg(any(test, feature = "test-support"))]
                started.fetch_add(1, Ordering::SeqCst);
            });
            #[cfg(any(test, feature = "test-support"))]
            {
                let stopped = Arc::clone(&alive_threads);
                builder.on_thread_stop(move || {
                    stopped.fetch_sub(1, Ordering::SeqCst);
                });
            }
            builder.build()
        };
        let names = [FANOUT_READ_THREAD_NAME, SNAPSHOT_IO_THREAD_NAME]
            .into_iter()
            .map(str::to_owned)
            .chain((0..WATCH_READ_THREADS).map(watch_read_thread_name));
        let mut runtimes = Vec::with_capacity(FANOUT_RUNTIMES);
        for name in names {
            match build(name) {
                Ok(runtime) => runtimes.push(runtime),
                Err(e) => {
                    for runtime in runtimes {
                        runtime.shutdown_background();
                    }
                    return Err(e);
                }
            }
        }
        let handle = |index: usize| runtimes[index].handle().clone();
        Ok(Self {
            fanout_read_handle: handle(0),
            snapshot_io_handle: handle(1),
            watch_read_handles: std::array::from_fn(|i| handle(2 + i)),
            watch_read_load: Arc::new(Mutex::new([0; WATCH_READ_THREADS])),
            runtimes: Mutex::new(Some(runtimes)),
            #[cfg(any(test, feature = "test-support"))]
            live_passes: std::array::from_fn(|_| Arc::new(AtomicUsize::new(0))),
            #[cfg(any(test, feature = "test-support"))]
            passes_started: std::array::from_fn(|_| AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-support"))]
            watch_read_passes_started: std::array::from_fn(|_| AtomicUsize::new(0)),
            #[cfg(any(test, feature = "test-support"))]
            alive_threads,
        })
    }

    /// The `snapshot-io` runtime, which the snapshot lane's routers are
    /// started on ([`connect_router_on`]).
    pub(crate) fn snapshot_io(&self) -> &Handle {
        &self.snapshot_io_handle
    }

    /// Assign an unfenced pass its `watch-read` thread: the one with the
    /// fewest unfenced passes in flight, the lowest index on a tie (module
    /// docs, *Assignment*). Choosing and counting are one step under the
    /// lock; the returned slot uncounts the pass when it drops.
    fn assign_watch_read(&self) -> WatchReadSlot {
        let mut load = self
            .watch_read_load
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // `min_by_key` returns the first of equal minima: the lowest index.
        let index = (0..WATCH_READ_THREADS)
            .min_by_key(|&i| load[i])
            .unwrap_or(0);
        load[index] += 1;
        WatchReadSlot {
            load: Arc::clone(&self.watch_read_load),
            index,
        }
    }

    /// Run one fan-out pass on the thread `lane` names — `fanout-read` for a
    /// fenced pass, the least-loaded `watch-read` thread for an unfenced one
    /// (module docs, *Assignment*) — and answer its requests here, in the
    /// caller's task, in the order it sends them; its result. The pass stays
    /// on that thread to its end.
    ///
    /// `pass` builds the pass from the [`PassLink`] it reaches its caller
    /// through; it must own everything it reads (a `ValkeyClient` clone, a
    /// `KeySchema` clone, the object id). `answer` answers each
    /// [`PassRequest`] synchronously, never waiting; the pass issues nothing
    /// more until its answer arrives (module docs, *The handshake*).
    ///
    /// The pass runs inside the caller's span, emitting to the caller's
    /// default subscriber. Dropping the returned future aborts the pass at
    /// its next await. A panic in the pass is re-raised here. A pass the
    /// runtime cancelled answers `ServiceError::Unavailable`, which callers
    /// retry; defensive only, since the runtimes shut down only once the
    /// client's last owner is gone, and a caller here holds one.
    pub(crate) async fn run_pass<T, P, F, A>(
        &self,
        lane: PassLane,
        pass: P,
        mut answer: A,
    ) -> Result<T, ServiceError>
    where
        P: FnOnce(PassLink) -> F,
        F: Future<Output = Result<T, ServiceError>> + Send + 'static,
        T: Send + 'static,
        A: FnMut(PassRequest) -> PassAnswer,
    {
        let (requests_tx, mut requests) = mpsc::channel(1);
        // The pass's future goes to the heap as it is built
        // ([`build_pass_on_heap`]), so what follows (the wrappers, the
        // instrumenting, `spawn`) moves a pointer, not the future. Built
        // here, the future and each of its moves took a stack slot in this
        // frame, and the release stack guard measured every route that reads
        // a snapshot, pass or no pass, about 91–154 KiB deeper on its worker
        // for it (`tests/release_stack_guard/`).
        let pass = build_pass_on_heap(
            pass,
            PassLink {
                requests: requests_tx,
            },
        );
        let (runtime, slot) = match lane {
            PassLane::Fenced => (&self.fanout_read_handle, None),
            PassLane::Unfenced => {
                let slot = self.assign_watch_read();
                (&self.watch_read_handles[slot.index], Some(slot))
            }
        };
        #[cfg(any(test, feature = "test-support"))]
        let pass = {
            self.passes_started[lane_index(lane)].fetch_add(1, Ordering::SeqCst);
            if let Some(slot) = &slot {
                self.watch_read_passes_started[slot.index].fetch_add(1, Ordering::SeqCst);
            }
            let live = LivePass::new(Arc::clone(&self.live_passes[lane_index(lane)]));
            async move {
                let _live = live;
                pass.await
            }
        };
        // The slot is the pass's future's: it uncounts the pass when that
        // future drops, completed, aborted or ended by the shutdown.
        let pass = async move {
            let _slot = slot;
            pass.await
        };
        // Inside the caller's span, and emitting to the caller's subscriber:
        // a thread's default subscriber is its own, so without this a
        // subscriber set for the caller's scope alone would not see the
        // pass's events.
        let mut pinned = PinnedPass(
            runtime.spawn(
                pass.instrument(tracing::Span::current())
                    .with_current_subscriber(),
            ),
        );
        let mut open = true;
        loop {
            tokio::select! {
                biased;
                request = requests.recv(), if open => match request {
                    Some((request, reply)) => {
                        // The pass waits on this reply; a pass aborted
                        // meanwhile has dropped its end, which is fine.
                        let _ = reply.send(answer(request));
                    }
                    None => open = false,
                },
                joined = &mut pinned.0 => {
                    return match joined {
                        Ok(result) => result,
                        Err(error) if error.is_panic() => {
                            std::panic::resume_unwind(error.into_panic())
                        }
                        Err(_) => Err(ServiceError::Unavailable(
                            "the snapshot fan-out threads of this replica have stopped; the \
                             snapshot read did not run"
                                .to_owned(),
                        )),
                    };
                }
            }
        }
    }

    /// Build the threads outside a `ValkeyClient`, as `ValkeyClient::connect`
    /// builds them: every thread runs the registered start hook, if any
    /// ([`register_fanout_thread_start_hook`]). For a test that reads what
    /// the hook did on each thread, through [`Self::fanout_read_handle`],
    /// [`Self::snapshot_io_handle`] and [`Self::watch_read_handles`] — in the
    /// service binary's own tests, the arena each thread is bound to.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_for_test() -> std::io::Result<Self> {
        Self::new()
    }

    /// [`Self::new_for_test`] with `start_hook` in place of the registered
    /// hook; `None` starts every thread with no hook, as in a process that
    /// registered none.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_thread_start_hook_for_test(
        start_hook: Option<FanoutThreadStartHook>,
    ) -> std::io::Result<Self> {
        Self::with_thread_start_hook(start_hook)
    }

    /// The `fanout-read` runtime: every fenced snapshot fan-out pass runs
    /// here.
    #[cfg(any(test, feature = "test-support"))]
    pub fn fanout_read_handle(&self) -> &Handle {
        &self.fanout_read_handle
    }

    /// The `snapshot-io` runtime: the snapshot lane's routers run here.
    #[cfg(any(test, feature = "test-support"))]
    pub fn snapshot_io_handle(&self) -> &Handle {
        &self.snapshot_io_handle
    }

    /// The `watch-read` runtimes, by index (`watch-read-<i>` is entry `i`):
    /// every unfenced snapshot fan-out pass runs on one of them.
    #[cfg(any(test, feature = "test-support"))]
    pub fn watch_read_handles(&self) -> &[Handle; WATCH_READ_THREADS] {
        &self.watch_read_handles
    }

    /// Passes spawned by [`Self::run_pass`] and not yet dropped (completed,
    /// aborted, or ended by the runtime's shutdown), both lanes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn live_passes(&self) -> usize {
        self.live_passes_on(PassLane::Fenced) + self.live_passes_on(PassLane::Unfenced)
    }

    /// [`Self::live_passes`] of `lane` alone.
    #[cfg(any(test, feature = "test-support"))]
    pub fn live_passes_on(&self, lane: PassLane) -> usize {
        self.live_passes[lane_index(lane)].load(Ordering::SeqCst)
    }

    /// Passes [`Self::run_pass`] ever spawned, both lanes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn passes_started(&self) -> usize {
        self.passes_started_on(PassLane::Fenced) + self.passes_started_on(PassLane::Unfenced)
    }

    /// [`Self::passes_started`] of `lane` alone.
    #[cfg(any(test, feature = "test-support"))]
    pub fn passes_started_on(&self, lane: PassLane) -> usize {
        self.passes_started[lane_index(lane)].load(Ordering::SeqCst)
    }

    /// Unfenced passes ever assigned to each `watch-read` thread, by index.
    #[cfg(any(test, feature = "test-support"))]
    pub fn watch_read_passes_started(&self) -> [usize; WATCH_READ_THREADS] {
        std::array::from_fn(|i| self.watch_read_passes_started[i].load(Ordering::SeqCst))
    }

    /// The assignment's load: unfenced passes in flight on each `watch-read`
    /// thread, by index (module docs, *Assignment*). A pass leaves the count
    /// when its future drops: on its `watch-read` thread, or, for a pass
    /// spawned after [`Self::shut_down`], at once on the caller's thread,
    /// since a runtime already shut down drops a spawned future unpolled.
    #[cfg(any(test, feature = "test-support"))]
    pub fn watch_read_load(&self) -> [usize; WATCH_READ_THREADS] {
        *self
            .watch_read_load
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Threads of any of the [`FANOUT_RUNTIMES`] runtimes started and not
    /// yet stopped: each worker, and any blocking-pool thread a runtime has
    /// (a router's DNS lookup runs on one). Shared, so it can be read after
    /// the runtimes are dropped: 0 once all have shut down and their threads
    /// exited.
    #[cfg(any(test, feature = "test-support"))]
    pub fn alive_threads(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.alive_threads)
    }
}

impl FanoutThreads {
    /// Shut every runtime down now, from whatever thread this runs on,
    /// without blocking it (module docs, *Shutdown*). Every task on any of
    /// them — a pass, a snapshot member's router — is dropped, and a pass
    /// started after this answers `ServiceError::Unavailable`. Run when the
    /// client's last owner drops (`ValkeyClient`'s `ClientOwners`), and by the
    /// drop. Idempotent.
    pub(crate) fn shut_down(&self) {
        let runtimes = self
            .runtimes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        // Never a plain runtime drop: that waits for the runtime's blocking
        // threads and panics when the dropping thread is in an async
        // context — the service's workers, or this very runtime's worker
        // when a pass held the client's last owner.
        for runtime in runtimes.into_iter().flatten() {
            runtime.shutdown_background();
        }
    }
}

impl Drop for FanoutThreads {
    fn drop(&mut self) {
        self.shut_down();
    }
}

/// Start `client`'s router on `spawn_on` when given (`snapshot-io`, for a
/// snapshot lane member), else on the calling runtime, and return its
/// handle. `connect()` spawns the router task and does not await, so the
/// runtime is entered only around that spawn.
pub(crate) fn connect_router_on<C: ClientLike>(
    spawn_on: Option<&Handle>,
    client: &C,
) -> ConnectHandle {
    match spawn_on {
        Some(runtime) => {
            let _entered = runtime.enter();
            client.connect()
        }
        None => client.connect(),
    }
}

/// What a pass running on a fan-out thread asks its caller, at the seams it
/// had when it ran inline (module docs, *The handshake*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassRequest {
    /// The fenced pass has read `bytes_read` reply bytes so far: after its
    /// first round and after each later window. The caller's fence hook
    /// tops its memory grant up (`FencedReadHook::on_read_progress`).
    Progress { bytes_read: u64 },
    /// WA1 of the charged read: the count windows' member counts are in
    /// (`FanoutAllowance::counted`).
    Counted {
        counts: [u64; 5],
        max_commands: u64,
        max_bytes: u64,
    },
    /// WA2 of the charged read: grow the charge, at least, to the need with
    /// `items`, `window_bytes` and `name_bytes` (`FanoutAllowance::grow`),
    /// before a window or index page is issued and after it. `plain_commands`
    /// is the wire commands of a plain window about to be issued (an index
    /// page, a lone long children list, a window after the read's script
    /// fallback), each holding its own future; 0 for a script window and
    /// after a window.
    Grow {
        items: u64,
        window_bytes: u64,
        name_bytes: u64,
        plain_commands: u64,
    },
    /// The charged read completed: the name bytes it retained, for the
    /// attempt's later charge points.
    NameBytes { name_bytes: u64 },
}

/// The caller's answer to a [`PassRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassAnswer {
    /// Applied; the pass goes on.
    GoOn,
    /// A refused growth: the pass stops before its next window, with the
    /// best-known whole target the caller's allowance recorded.
    Stop { target: u64 },
}

/// A pass's end of its caller's request channel ([`FanoutThreads::run_pass`]).
pub(crate) struct PassLink {
    requests: mpsc::Sender<(PassRequest, oneshot::Sender<PassAnswer>)>,
}

impl PassLink {
    /// Ask the caller and wait for its answer. When the caller has gone, its
    /// drop is aborting this pass, so this never returns: the pass issues
    /// nothing more and ends at this await.
    pub(crate) async fn ask(&self, request: PassRequest) -> PassAnswer {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.requests.send((request, reply_tx)).await.is_ok() {
            if let Ok(answer) = reply_rx.await {
                return answer;
            }
        }
        std::future::pending().await
    }
}

/// Build a pass's future from `pass` and box it: the future
/// [`FanoutThreads::run_pass`] spawns.
///
/// Never inlined, so the future's own stack slot, while it is built and
/// moved to the heap, is this call's alone: it is on the caller's thread
/// only for the call, and never in the frame of `run_pass` or of the
/// snapshot read that awaits it.
#[inline(never)]
fn build_pass_on_heap<P, F>(pass: P, link: PassLink) -> std::pin::Pin<Box<F>>
where
    P: FnOnce(PassLink) -> F,
    F: Future,
{
    Box::pin(pass(link))
}

/// The spawned pass, aborted when dropped (a no-op once it finished).
struct PinnedPass<T>(JoinHandle<T>);

impl<T> Drop for PinnedPass<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// One unfenced pass counted on its `watch-read` thread's load from its
/// assignment ([`FanoutThreads::assign_watch_read`]) to the drop of its
/// future, whichever way it ends — completed, aborted, or dropped by the
/// runtime's shutdown (or at once, when spawned on a runtime already shut
/// down).
struct WatchReadSlot {
    load: WatchReadLoad,
    index: usize,
}

impl Drop for WatchReadSlot {
    fn drop(&mut self) {
        let mut load = self.load.lock().unwrap_or_else(PoisonError::into_inner);
        load[self.index] = load[self.index].saturating_sub(1);
    }
}

/// Counts one spawned pass in [`FanoutThreads::live_passes`] from its spawn
/// to the drop of its future, whichever way it ends.
#[cfg(any(test, feature = "test-support"))]
struct LivePass(Arc<AtomicUsize>);

#[cfg(any(test, feature = "test-support"))]
impl LivePass {
    fn new(live: Arc<AtomicUsize>) -> Self {
        live.fetch_add(1, Ordering::SeqCst);
        Self(live)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Drop for LivePass {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    //! The fan-out threads' contract (b.26e Stage 2, r14 B.1), pinned on the
    //! real readers against a real store and, for `run_pass` alone, on
    //! synthetic passes:
    //!
    //! | Area | What it pins |
    //! |---|---|
    //! | Thread identity | every window of a fenced cold start (twice) and a fenced catch-up flight runs on `fanout-read`, and every window of the structural WATCH read and of the ext-GC reference walk (`storage::retirement`) on a `watch-read` thread, never on `fanout-read` or a service worker, on a 4-worker runtime |
    //! | Lane isolation | a fenced pass runs to completion on `fanout-read` while every `watch-read` thread is held by a busy unfenced pass |
    //! | Assignment | parked unfenced passes take the `watch-read` threads one each, lowest index first; a fifth goes to the least loaded, the lowest index on a tie; a pass that ended frees its thread for the next; the load returns to 0 once a pass completes, is aborted or panics |
    //! | Progress order | no fenced-pass window is issued while its caller applies a progress report; the window after the first report that tops the grant up waits until the hook holds that top-up |
    //! | Allowance order | no WATCH-read window is built while its caller answers a growth; the read's last window is charged before it is built; a refused WA2 growth stops the read before the window it would have covered; the retained name bytes reach the caller's allowance |
    //! | Cancellation | a cold start's read, or a flight whose every waiter left, dropped mid-read leaves no live pass within 1 s and releases its fence lease; a dropped `run_pass` aborts its pass on either lane |
    //! | Panic | a panic in a flight's pass reaches every waiter as `Internal`; `run_pass` re-raises a pass's panic on its caller, on either lane |
    //! | Shutdown | dropping the threads inside a runtime never panics and stops every one of the [`FANOUT_RUNTIMES`] threads; a pass on either lane after the shutdown answers `Unavailable` |
    //! | Start hook | a hook given runs exactly once on every thread the runtimes start, each runtime's worker and blocking-pool threads, and on no other; with none, the threads start, run passes on both lanes and run no hook, even beside threads built with one. That the service binary's hook binds each thread to its arena is pinned in the binary's own tests (`src/allocator.rs`) |
    //!
    //! The window recorder is the fan-out seam's
    //! (`ObjectSnapshotReadFaultInjector::record_window_threads`): it names
    //! the thread each window's commands were built on, as the window is
    //! built. The layer is the posed-rig fixture (`test_support::
    //! fenced_read_progress`) on a client whose atomic-read ceilings are
    //! lowered, so each read runs many short windows.
    //!
    //! The `run_pass`-alone tests (lane isolation, assignment, the dropped,
    //! panicking and post-shutdown passes) and the start-hook tests need no
    //! store: their passes are synthetic, each reporting the thread it runs
    //! on.
    //!
    //! Each wait is bounded: a read by [`READ_BUDGET`], a signal by
    //! [`SIGNAL_BUDGET`], a live pass's end by [`PASS_END_BUDGET`]. The
    //! progress-order test watches each report for [`QUIET`], and the
    //! allowance-order test each growth for [`GROWTH_QUIET`]: a window built
    //! meanwhile fails them, and a correct build passes whatever the host's
    //! load, since its pass builds nothing until the answer; only a broken
    //! build's failure depends on its pass reaching its next window inside
    //! that time.

    use std::sync::Mutex as StdMutex;
    use std::time::{Duration, Instant};

    use futures::FutureExt;

    use super::*;
    use crate::domain::delta::MutationAllowance;
    use crate::storage::client::ValkeyClient;
    use crate::storage::commit_charge::{ObjectStateNeed, SharedCommitCharge};
    use crate::storage::diff_read::{
        read_object_snapshot_slim, read_object_snapshot_slim_charged, ChargedSnapshotRead,
        FanoutAllowance, FanoutWindowThread, ObjectSnapshotReadFaultInjector,
        ObjectSnapshotReadFaultTarget, ObjectSnapshotReader, SlimSnapshotReadConsumer,
        SnapshotReadOf, SnapshotTooLarge,
    };
    use crate::storage::keys::KeySchema;
    use crate::storage::memory_budget::SNAPSHOT_MEMORY_FACTOR;
    use crate::storage::snapshot_fence::read::{
        read_object_snapshot_fenced, FenceAttempt, FenceGrant, FenceReleased, FenceRetry,
        FencedReadHook,
    };
    use crate::storage::snapshot_fence::LayerSize;
    use crate::storage::snapshot_flights::{FlightWaitError, SnapshotFlights};
    use crate::storage::structural_cache::StructuralCache;
    use crate::storage::structural_cache_cold_start::STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM;
    use crate::storage::structural_cache_fenced::ColdStartFenceHook;
    use crate::storage::subscriber_admission::CapacityBudget;
    use crate::storage::test_support::fenced_read_progress::{
        as_held, lowered_client, posed_rig_layer, RIG_SECTIONS,
    };
    use crate::storage::test_support::TestValkey;

    /// Bound on one read.
    const READ_BUDGET: Duration = Duration::from_secs(30);
    /// Bound on a signal: a hold parked, a lease released, a flight left.
    const SIGNAL_BUDGET: Duration = Duration::from_secs(10);
    /// How soon a dropped read's pass must be gone (r14 B.1 proof plan).
    const PASS_END_BUDGET: Duration = Duration::from_secs(1);
    /// How long the progress-order hook watches for a window after each
    /// report (module docs).
    const QUIET: Duration = Duration::from_millis(200);
    /// How long the allowance-order charge watches for a window while it
    /// answers each growth (module docs): shorter than [`QUIET`], since the
    /// read asks many more growths than it reports.
    const GROWTH_QUIET: Duration = Duration::from_millis(50);
    /// The structural cache's budget: no read here comes near it.
    const CACHE_BUDGET: u64 = 64 * 1024 * 1024;
    /// The capacity budget of a flight's waiter: no wait here runs it out.
    const ROOMY: Duration = Duration::from_secs(60);
    /// The name of a service runtime's worker thread (tokio 1.53's default,
    /// `runtime::Builder`'s `thread_name`), the thread a pass ran on before
    /// the fan-out threads.
    const SERVICE_WORKER: &str = "tokio-rt-worker";

    /// The cold start's probed estimate from the layer's `TOO_LARGE` answer:
    /// its item bound × the route predictor's bytes per item.
    fn cold_start_estimate(answer: &SnapshotTooLarge) -> LayerSize {
        LayerSize {
            items: answer.item_bound,
            bytes: answer.item_bound * STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM,
        }
    }

    /// A posed-rig layer on a lowered client of `tv`'s with its own budget,
    /// and an injector installed on it: `(client, injector, object id, its
    /// TOO_LARGE answer)`.
    async fn rig(
        tv: &TestValkey,
        layer: &str,
    ) -> (
        ValkeyClient,
        Arc<ObjectSnapshotReadFaultInjector>,
        u64,
        SnapshotTooLarge,
    ) {
        let client = lowered_client(tv, None);
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        client.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        let (oid, answer) = Box::pin(posed_rig_layer(tv, &client, layer)).await;
        (client, injector, oid, answer)
    }

    /// One fenced cold-start read of `oid` through the cold start's own hook,
    /// as `structural_cache_cold_start` runs it above the ceilings.
    async fn cold_start_read(
        client: &ValkeyClient,
        keys: &KeySchema,
        oid: u64,
        answer: &SnapshotTooLarge,
    ) {
        let cache = StructuralCache::new(CACHE_BUDGET);
        let estimate = cold_start_estimate(answer);
        let mut hook = ColdStartFenceHook::new(client, &cache, estimate, None);
        tokio::time::timeout(
            READ_BUDGET,
            Box::pin(read_object_snapshot_fenced(
                client,
                keys,
                oid,
                SlimSnapshotReadConsumer::StructuralCache,
                estimate,
                &mut hook,
            )),
        )
        .await
        .expect("the cold start's read ends in time")
        .expect("the cold start's fenced read of a quiet layer verifies");
    }

    /// One fenced catch-up of `oid` through a flight of its own: attach,
    /// read, take the build turn and drop it.
    async fn flight_read(
        client: &ValkeyClient,
        keys: &KeySchema,
        oid: u64,
        answer: &SnapshotTooLarge,
    ) {
        let flights = SnapshotFlights::new(client.clone(), keys.clone());
        let waiter = flights.attach_or_create(oid, LayerSize::from_too_large(answer, None));
        let budget = CapacityBudget::new(ROOMY);
        let read = tokio::time::timeout(READ_BUDGET, Box::pin(waiter.read(&budget)))
            .await
            .expect("the flight's read ends in time")
            .unwrap_or_else(|e| panic!("the flight's waiter is served: {e:?}"));
        let turn = tokio::time::timeout(SIGNAL_BUDGET, read.build_turn())
            .await
            .expect("the waiter gets its build turn");
        drop(turn);
    }

    /// The windows `injector` recorded, by reader.
    fn windows_of(
        windows: &[FanoutWindowThread],
        reader: ObjectSnapshotReader,
    ) -> Vec<&FanoutWindowThread> {
        windows.iter().filter(|w| w.reader == reader).collect()
    }

    /// Thread identity (r14 B.1 proof plan, the CI pin; PD-U1): on a
    /// 4-worker runtime, every window of two consecutive fenced cold starts
    /// and of a fenced catch-up flight is built on `fanout-read`, and every
    /// window of the two unfenced passes — the structural update's WATCH read
    /// and the ext-GC reference walk (`storage::retirement`) — on a
    /// `watch-read` thread: never on
    /// `fanout-read`, where they would queue the fenced passes behind them,
    /// and never on one of the runtime's workers, which is where each window
    /// ran while the pass was polled in its caller's task, and where its
    /// allocations then landed in whichever worker's allocator arena the
    /// reply's router had woken.
    ///
    /// The reference walk (b.fsz MR-D4, D4-T8) consults the window recorder
    /// as the unfenced fan-out's reader for each of its window-opening
    /// commands — the header `EXISTS`, each index page, each `HSCAN obj:ext`
    /// step — so its windows are recorded like the WATCH read's. (A
    /// retirement route's size read, issued on the caller's thread before the
    /// walk's pass, is no window and is not recorded; this test runs the walk
    /// alone.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_snapshot_fan_out_window_runs_on_its_lanes_fan_out_thread_b26e() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, injector, oid, answer) = rig(&tv, "fanout-thread-identity").await;
        injector.record_window_threads();

        let mut per_read = Vec::new();
        for read in [
            "cold start 1",
            "cold start 2",
            "catch-up flight",
            "WATCH read",
            "ext-GC reference walk",
        ] {
            let before = injector.windows_recorded();
            // In a task of its own, on the runtime's workers, as an RPC
            // handler runs.
            let (client, keys) = (client.clone(), keys.clone());
            let task = tokio::spawn(async move {
                match read {
                    "catch-up flight" => flight_read(&client, &keys, oid, &answer).await,
                    "WATCH read" => {
                        let read = Box::pin(read_object_snapshot_slim(
                            &client,
                            &keys,
                            &oid.to_string(),
                            SlimSnapshotReadConsumer::StructuralUpdate,
                        ))
                        .await
                        .expect("the WATCH read of the rig succeeds");
                        assert!(matches!(read, SnapshotReadOf::Snapshot(_)));
                    }
                    "ext-GC reference walk" => {
                        Box::pin(crate::storage::retirement::walk_object_references(
                            &client,
                            &keys,
                            &oid.to_string(),
                            &crate::storage::retirement::RetirementWalk::unreserved(),
                        ))
                        .await
                        .expect("the ext-GC reference walk of the rig succeeds");
                    }
                    _ => cold_start_read(&client, &keys, oid, &answer).await,
                }
            });
            task.await.expect("the read's task completes");
            per_read.push((read, injector.windows_recorded() - before));
        }

        let windows = injector.window_threads();
        for (read, count) in &per_read {
            assert!(
                *count >= 3,
                "fixture: the {read} runs several windows on the lowered ceilings; {per_read:?}"
            );
        }
        let fenced = windows_of(&windows, ObjectSnapshotReader::FencedFanout).len();
        let unfenced = windows_of(&windows, ObjectSnapshotReader::UnfencedFanout).len();
        assert_eq!(
            (fenced, unfenced),
            (
                per_read[..3].iter().map(|(_, n)| n).sum::<usize>(),
                per_read[3..].iter().map(|(_, n)| n).sum::<usize>(),
            ),
            "the three fenced reads' windows on the fenced fan-out, the WATCH read's and the \
             ext-GC reference walk's on the unfenced one"
        );
        let threads = client.fanout_threads();
        assert_eq!(
            (
                threads.passes_started_on(PassLane::Unfenced),
                threads.watch_read_passes_started().iter().sum::<usize>()
            ),
            (2, 2),
            "the WATCH read and the ext-GC reference walk each ran one pass on the unfenced \
             lane, each assigned a `watch-read` thread"
        );
        assert!(
            threads.passes_started_on(PassLane::Fenced) >= 3,
            "fixture: each fenced read ran at least one pass on the fenced lane"
        );
        let watch_read: Vec<String> = (0..WATCH_READ_THREADS)
            .map(watch_read_thread_name)
            .collect();
        let elsewhere: Vec<&FanoutWindowThread> = windows
            .iter()
            .filter(|w| match w.reader {
                ObjectSnapshotReader::UnfencedFanout => !watch_read
                    .iter()
                    .any(|name| w.thread.as_ref() == Some(name)),
                _ => w.thread.as_deref() != Some(FANOUT_READ_THREAD_NAME),
            })
            .collect();
        assert!(
            elsewhere.is_empty(),
            "every fenced fan-out window runs on `{FANOUT_READ_THREAD_NAME}` and every unfenced \
             one on a `{WATCH_READ_THREAD_NAME_PREFIX}` thread; {} of {} ran elsewhere ({} on a \
             `{SERVICE_WORKER}`): {elsewhere:?}",
            elsewhere.len(),
            windows.len(),
            elsewhere
                .iter()
                .filter(|w| w.thread.as_deref() == Some(SERVICE_WORKER))
                .count()
        );
        assert_eq!(
            client.fanout_threads().live_passes(),
            0,
            "every pass ended with its read"
        );
    }

    /// One report as [`ProgressOrderHook`] saw it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SeenReport {
        bytes_read: u64,
        /// Windows recorded when the report reached the hook.
        windows_at_entry: usize,
        /// Windows recorded once the hook had applied the report and then
        /// watched [`QUIET`] more.
        windows_at_exit: usize,
        /// What the read's budget accounted once the inner hook took it.
        held: u64,
    }

    /// The cold start's hook, every event forwarded, which records each
    /// progress report against the window recorder, holds the window after
    /// the first report that tops the grant up (held > grant) on the seam, and
    /// watches for a window issued while it is still applying a report.
    struct ProgressOrderHook<'a> {
        inner: ColdStartFenceHook<'a>,
        client: &'a ValkeyClient,
        injector: Arc<ObjectSnapshotReadFaultInjector>,
        reports: Arc<StdMutex<Vec<SeenReport>>>,
        /// The hold on the first window after the first report that tops the
        /// grant up (held > grant), armed by that report.
        hold: Arc<StdMutex<Option<Arc<crate::storage::diff_read::SnapshotReadHold>>>>,
        /// What the budget held once the lease was granted: the up-front
        /// grant.
        at_grant: Arc<StdMutex<Option<u64>>>,
    }

    impl ProgressOrderHook<'_> {
        fn held(&self) -> u64 {
            let budget = &self.client.catchup_build_budget;
            budget.reserved_bytes() + budget.overcommit_bytes()
        }
    }

    impl FencedReadHook for ProgressOrderHook<'_> {
        async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
            self.inner.before_attempt(attempt).await
        }

        async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
            self.inner.on_retry_after(retry).await
        }

        async fn on_grant(&mut self, grant: FenceGrant) {
            self.inner.on_grant(grant).await;
            *self.at_grant.lock().expect("never poisoned") = Some(self.held());
        }

        fn on_read_progress(&mut self, bytes_read: u64) {
            let windows_at_entry = self.injector.windows_recorded();
            self.inner.on_read_progress(bytes_read);
            let held = self.held();
            let grant = self
                .at_grant
                .lock()
                .expect("never poisoned")
                .unwrap_or(u64::MAX);
            let mut hold = self.hold.lock().expect("never poisoned");
            if hold.is_none() && held > grant {
                // The first report that topped the grant up. The pass is
                // waiting for its answer, so the next window it reaches is
                // the one after this report.
                *hold = Some(
                    self.injector
                        .push_hold_at(ObjectSnapshotReadFaultTarget::next_read()),
                );
            }
            drop(hold);
            // A test hook may wait here: no window may be issued meanwhile.
            let watched = Instant::now();
            while self.injector.windows_recorded() == windows_at_entry && watched.elapsed() < QUIET
            {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.reports
                .lock()
                .expect("never poisoned")
                .push(SeenReport {
                    bytes_read,
                    windows_at_entry,
                    windows_at_exit: self.injector.windows_recorded(),
                    held,
                });
        }

        async fn before_release(&mut self) {
            self.inner.before_release().await;
        }

        async fn after_release(&mut self, released: FenceReleased) {
            self.inner.after_release(released).await;
        }
    }

    /// Progress order (r14 B.1 item 3, proof plan "Ordering pin"): the
    /// fenced pass, on `fanout-read`, issues no window while its caller is
    /// applying a progress report — each report is answered before the pass
    /// goes on. Seen two ways on a fenced cold start:
    ///
    /// - every report's hook watches [`QUIET`] after applying it, and no
    ///   window is recorded meanwhile;
    /// - the window after the first report that tops the up-front grant up
    ///   is parked on the seam; while it is parked, the read's budget already
    ///   holds that report's top-up — the grant raised to
    ///   `SNAPSHOT_MEMORY_FACTOR` × the bytes it reported — and no later
    ///   report has been applied.
    ///
    /// A pass that sent its report and went on without the answer issues its
    /// next window while the hook is still watching.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fenced_pass_issues_no_window_before_its_caller_applied_the_report_b26e() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, injector, oid, answer) = rig(&tv, "fanout-progress-order").await;
        let total = client.catchup_build_budget.total_bytes();
        injector.record_window_threads();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let estimate = cold_start_estimate(&answer);
        let reports = Arc::new(StdMutex::new(Vec::new()));
        let hold = Arc::new(StdMutex::new(None));
        let at_grant = Arc::new(StdMutex::new(None));
        let mut hook = ProgressOrderHook {
            inner: ColdStartFenceHook::new(&client, &cache, estimate, None),
            client: &client,
            injector: Arc::clone(&injector),
            reports: Arc::clone(&reports),
            hold: Arc::clone(&hold),
            at_grant: Arc::clone(&at_grant),
        };
        let read = Box::pin(read_object_snapshot_fenced(
            &client,
            keys,
            oid,
            SlimSnapshotReadConsumer::StructuralCache,
            estimate,
            &mut hook,
        ));
        let at_park = async {
            let deadline = tokio::time::Instant::now() + SIGNAL_BUDGET;
            let parked = loop {
                if let Some(hold) = hold.lock().expect("never poisoned").clone() {
                    break hold;
                }
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "no report topped the grant up"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            };
            tokio::time::timeout_at(deadline, parked.parked())
                .await
                .expect("the window after the first top-up parks on the seam");
            let seen = (
                reports.lock().expect("never poisoned").clone(),
                injector.windows_recorded(),
                client.catchup_build_budget.reserved_bytes()
                    + client.catchup_build_budget.overcommit_bytes(),
            );
            parked.release();
            seen
        };
        let (read, (reports_at_park, windows_at_park, held_at_park)) =
            tokio::time::timeout(READ_BUDGET, async { tokio::join!(read, at_park) })
                .await
                .expect("the read ends in time");
        read.expect("the fenced read of a quiet layer verifies");
        drop(hook);

        let reports = reports.lock().expect("never poisoned").clone();
        assert!(
            reports.len() >= 3,
            "fixture: round 1 and several windows reported; {reports:?}"
        );
        for (i, report) in reports.iter().enumerate() {
            assert_eq!(
                report.windows_at_exit, report.windows_at_entry,
                "report {i} ({} B): no window may be issued while the caller applies the report \
                 (the pass waits for its answer); {reports:?}",
                report.bytes_read
            );
        }
        let grant = at_grant
            .lock()
            .expect("never poisoned")
            .expect("the lease was granted");
        let topping = reports
            .iter()
            .position(|r| r.held > grant)
            .unwrap_or_else(|| {
                panic!("fixture: a report tops the grant ({grant} B) up; {reports:?}")
            });
        assert!(
            topping + 1 < reports.len(),
            "fixture: windows follow the first top-up; {reports:?}"
        );
        assert_eq!(
            reports_at_park,
            reports[..=topping],
            "parked on the window after the first report that topped the grant up: every report \
             up to it applied, none after"
        );
        let topping = reports[topping];
        assert_eq!(
            windows_at_park,
            topping.windows_at_entry + 1,
            "the parked window is the first one after that report"
        );
        let topped_up = grant.max(as_held(topping.bytes_read * SNAPSHOT_MEMORY_FACTOR, total));
        assert_eq!(
            (topping.held, held_at_park),
            (topped_up, topped_up),
            "while the next window is parked the read already holds that report's top-up: the \
             grant raised to {SNAPSHOT_MEMORY_FACTOR} × its {} B",
            topping.bytes_read
        );
    }

    /// A charge for the WATCH read's allowance that grants every growth but
    /// the `refuse_at`-th, and records, at each, how many windows the seam
    /// had recorded.
    struct ScriptedGrowth {
        injector: Arc<ObjectSnapshotReadFaultInjector>,
        refuse_at: Option<usize>,
        held: StdMutex<u64>,
        /// The windows built when each growth was asked.
        windows_at_growth: StdMutex<Vec<usize>>,
        /// The windows built while each growth was being answered, over
        /// [`GROWTH_QUIET`].
        windows_during_growth: StdMutex<Vec<usize>>,
    }

    impl ScriptedGrowth {
        fn new(injector: &Arc<ObjectSnapshotReadFaultInjector>, refuse_at: Option<usize>) -> Self {
            Self {
                injector: Arc::clone(injector),
                refuse_at,
                held: StdMutex::new(0),
                windows_at_growth: StdMutex::new(Vec::new()),
                windows_during_growth: StdMutex::new(Vec::new()),
            }
        }

        fn windows_at_growth(&self) -> Vec<usize> {
            self.windows_at_growth
                .lock()
                .expect("never poisoned")
                .clone()
        }

        fn windows_during_growth(&self) -> Vec<usize> {
            self.windows_during_growth
                .lock()
                .expect("never poisoned")
                .clone()
        }
    }

    impl SharedCommitCharge for ScriptedGrowth {
        fn held(&self) -> u64 {
            *self.held.lock().expect("never poisoned")
        }

        fn target(&self, need: &ObjectStateNeed) -> u64 {
            crate::grpc::upload_admission::object_state_charge(need)
        }

        fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
            crate::grpc::upload_admission::object_state_rewrite_allowance(self.held(), need)
        }

        fn try_hold(&self, target: u64) -> bool {
            *self.held.lock().expect("never poisoned") = target;
            true
        }

        fn try_hold_at_least(&self, target: u64) -> bool {
            let windows = self.injector.windows_recorded();
            let growth = {
                let mut seen = self.windows_at_growth.lock().expect("never poisoned");
                seen.push(windows);
                seen.len() - 1
            };
            // The pass waits for this answer: no window may be built while
            // it is being decided.
            let watched = Instant::now();
            while watched.elapsed() < GROWTH_QUIET {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.windows_during_growth
                .lock()
                .expect("never poisoned")
                .push(self.injector.windows_recorded() - windows);
            if self.refuse_at == Some(growth) {
                return false;
            }
            let mut held = self.held.lock().expect("never poisoned");
            *held = (*held).max(target);
            true
        }
    }

    /// The charged WATCH read of `oid` under `charge`: whether it stopped,
    /// and the name bytes its allowance holds after it.
    async fn charged_read(
        client: &ValkeyClient,
        keys: &KeySchema,
        oid: u64,
        charge: &ScriptedGrowth,
    ) -> (bool, u64) {
        let mut allowance = FanoutAllowance::new(charge);
        let read = tokio::time::timeout(
            READ_BUDGET,
            Box::pin(read_object_snapshot_slim_charged(
                client,
                keys,
                &oid.to_string(),
                &mut allowance,
            )),
        )
        .await
        .expect("the charged read ends in time")
        .expect("the charged read of a quiet layer succeeds or stops");
        let stopped = matches!(read, ChargedSnapshotRead::Stopped { .. });
        (stopped, allowance.name_bytes())
    }

    /// Allowance order (PD-U1; r14 Decisions for writers 3): the charged
    /// WATCH read's pass, on a `watch-read` thread, waits for its caller's answer to
    /// each allowance growth, so a growth refused before a window stops the
    /// read before that window is built — as when the allowance was called
    /// inline. A run that grants everything records, at each growth, the
    /// windows built so far: none is built while a growth is being answered
    /// (each watched for [`GROWTH_QUIET`]), and the read's last window is
    /// charged before it is built. A second read of the same layer refuses
    /// the last growth taken right before a window, and builds no window past
    /// it. The granted read's allowance ends holding the name bytes the read
    /// retained: every field index member's and every section name's.
    ///
    /// A pass that issued its window before the answer arrived builds it
    /// while the growth is still being answered, or, never waiting, is not
    /// stopped by the refusal; one that charged a window only after building
    /// it asks its read's last growths with that window already built.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_refused_watch_read_growth_stops_the_read_before_its_next_window_b26e() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, injector, oid, _) = rig(&tv, "fanout-allowance-order").await;

        injector.record_window_threads();
        let granted = ScriptedGrowth::new(&injector, None);
        let (stopped, name_bytes) = charged_read(&client, keys, oid, &granted).await;
        assert!(
            !stopped,
            "a charge that grants everything never stops the read"
        );
        let expected_names: u64 = (1..=RIG_SECTIONS)
            .map(|joint| {
                (KeySchema::field_member(&joint.to_string(), "xformOp:transform").len()
                    + format!("Joint{joint}").len()) as u64
            })
            .sum();
        assert_eq!(
            name_bytes, expected_names,
            "the read's retained name bytes reach the caller's allowance"
        );
        let at_growth = granted.windows_at_growth();
        let all_windows = injector.windows_recorded();
        assert!(
            granted.windows_during_growth().iter().all(|&n| n == 0),
            "no window may be built while the caller answers a growth (the pass waits for the \
             answer): windows built during each growth {:?}",
            granted.windows_during_growth()
        );
        // The read's last window is charged before it is built: its growth
        // is asked one window short of the read's total, and the growth
        // after it — the last — once the window is in.
        let last = at_growth.len() - 1;
        assert_eq!(
            (at_growth[last - 1], at_growth[last]),
            (all_windows - 1, all_windows),
            "the read's last window is charged before it is built: {at_growth:?}"
        );
        // A growth right before a window: the windows built are the same as
        // at the growth before it, and one more by the growth after it.
        let before_a_window = (1..at_growth.len().saturating_sub(1))
            .rev()
            .find(|&g| at_growth[g] == at_growth[g - 1] && at_growth[g + 1] == at_growth[g] + 1)
            .unwrap_or_else(|| {
                panic!("fixture: the read grows its charge before a window; {at_growth:?}")
            });
        assert!(
            at_growth[before_a_window] < all_windows,
            "fixture: windows follow the refused growth"
        );

        injector.record_window_threads();
        let refusing = ScriptedGrowth::new(&injector, Some(before_a_window));
        let (stopped, _) = charged_read(&client, keys, oid, &refusing).await;
        assert!(stopped, "the refused growth stops the read");
        assert_eq!(
            refusing.windows_at_growth().len(),
            before_a_window + 1,
            "no growth after the refused one"
        );
        assert_eq!(
            injector.windows_recorded(),
            at_growth[before_a_window],
            "the read stopped before the window its refused growth would have covered: \
             growth {before_a_window} of {at_growth:?}"
        );
        assert_eq!(client.fanout_threads().live_passes(), 0);
    }

    /// Wait, within [`PASS_END_BUDGET`], until `client` has no live pass.
    async fn no_live_pass_within_the_budget(client: &ValkeyClient, what: &str) {
        let deadline = Instant::now() + PASS_END_BUDGET;
        while client.fanout_threads().live_passes() != 0 {
            assert!(
                Instant::now() < deadline,
                "{what}: a pass still runs {PASS_END_BUDGET:?} after its read was dropped"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Wait, within [`SIGNAL_BUDGET`], until `oid`'s fence key holds no
    /// reader lease (`r:<token>` field): the dropped read's lease guard
    /// released it.
    async fn lease_released(tv: &TestValkey, oid: u64, what: &str) {
        use fred::interfaces::HashesInterface;
        let fence = tv.key_schema().obj_snapshot_fence(&oid.to_string());
        let deadline = Instant::now() + SIGNAL_BUDGET;
        loop {
            let fields: std::collections::HashMap<String, String> = tv
                .client()
                .pool
                .hgetall(&fence)
                .await
                .expect("HGETALL the fence key");
            if !fields.keys().any(|k| k.starts_with("r:")) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{what}: the lease was not released within {SIGNAL_BUDGET:?}: {fields:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// The first field row's command of the fenced fan-out (round 2), held:
    /// a point mid-read, after the lease was granted.
    fn hold_mid_read(
        injector: &ObjectSnapshotReadFaultInjector,
    ) -> Arc<crate::storage::diff_read::SnapshotReadHold> {
        injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_of_family(
            ObjectSnapshotReader::FencedFanout,
            "field",
            0,
        ))
    }

    /// Cancellation (r14 B.1 item 7, proof plan "Cancellation pin"): a cold
    /// start's fenced read dropped mid-read — its pass parked on a field row
    /// — aborts its pass on `fanout-read` (no live pass within
    /// [`PASS_END_BUDGET`]) and releases its fence lease. A pass whose
    /// handle were detached rather than aborted would stay parked.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cold_start_dropped_mid_read_leaves_no_live_pass_and_releases_its_lease_b26e() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, injector, oid, answer) = rig(&tv, "fanout-cold-start-dropped").await;
        let hold = hold_mid_read(&injector);
        let cache = StructuralCache::new(CACHE_BUDGET);
        let estimate = cold_start_estimate(&answer);
        let mut hook = ColdStartFenceHook::new(&client, &cache, estimate, None);
        {
            let mut read = Box::pin(read_object_snapshot_fenced(
                &client,
                keys,
                oid,
                SlimSnapshotReadConsumer::StructuralCache,
                estimate,
                &mut hook,
            ));
            tokio::time::timeout(SIGNAL_BUDGET, async {
                tokio::select! {
                    read = &mut read => panic!("the read ended before its hold: {:?}", read.err()),
                    () = hold.parked() => {}
                }
            })
            .await
            .expect("the read parks mid-read on the seam");
            assert_eq!(
                client.fanout_threads().live_passes(),
                1,
                "precondition: the pass is parked, live"
            );
        }
        drop(hook);
        no_live_pass_within_the_budget(&client, "the dropped cold start").await;
        lease_released(&tv, oid, "the dropped cold start").await;
        hold.release();
    }

    /// Cancellation, a flight (r14 Decisions for writers 4): a flight whose
    /// every waiter left mid-read — its pass parked on a field row — aborts
    /// its pass and releases its lease, and leaves the registry.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_flight_whose_waiters_all_left_mid_read_leaves_no_live_pass_and_releases_its_lease_b26e(
    ) {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, injector, oid, answer) = rig(&tv, "fanout-flight-dropped").await;
        let hold = hold_mid_read(&injector);
        let flights = SnapshotFlights::new(client.clone(), keys.clone());
        let waiters = [
            flights.attach_or_create(oid, LayerSize::from_too_large(&answer, None)),
            flights.attach_or_create(oid, LayerSize::from_too_large(&answer, None)),
        ];
        let tasks: Vec<_> = waiters
            .into_iter()
            .map(|waiter| {
                tokio::spawn(async move {
                    let budget = CapacityBudget::new(ROOMY);
                    waiter.read(&budget).await.map(drop)
                })
            })
            .collect();
        tokio::time::timeout(SIGNAL_BUDGET, hold.parked())
            .await
            .expect("the flight's read parks mid-read on the seam");
        assert_eq!(client.fanout_threads().live_passes(), 1, "precondition");
        for task in tasks {
            task.abort();
            let joined = tokio::time::timeout(SIGNAL_BUDGET, task)
                .await
                .expect("the aborted waiter settles");
            assert!(
                matches!(&joined, Err(e) if e.is_cancelled()),
                "the waiter was cancelled mid-read: {joined:?}"
            );
        }
        no_live_pass_within_the_budget(&client, "the abandoned flight").await;
        lease_released(&tv, oid, "the abandoned flight").await;
        let deadline = Instant::now() + SIGNAL_BUDGET;
        while !flights.is_empty() {
            assert!(
                Instant::now() < deadline,
                "the abandoned flight leaves the map"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        hold.release();
    }

    /// A `tracing` subscriber that panics, once, on the fan-out's per-window
    /// `fan-out window` event — which the pass emits on `fanout-read`, under
    /// its caller's subscriber — so the pass itself panics, with no seam in
    /// the code under test.
    struct PanicAtFanoutWindow {
        armed: Arc<std::sync::atomic::AtomicBool>,
    }

    impl tracing::Subscriber for PanicAtFanoutWindow {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            let meta = event.metadata();
            if meta.target() == "live_edit::storage::diff_read"
                && meta.fields().field("replies_us").is_some()
                && self.armed.swap(false, Ordering::SeqCst)
            {
                panic!("test: the fan-out pass panics at its first window");
            }
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// Panic (r14 B.1 item 7; Decisions for writers 5): a panic in a
    /// flight's pass, on `fanout-read`, is re-raised in the flight's read
    /// stage, whose `catch_unwind` makes it an `Internal` failure for every
    /// waiter; the flight leaves the map, its grant goes back, and no pass
    /// is left. A pass whose panic were turned into a plain error, or never
    /// reached its caller, fails it.
    ///
    /// Runs alongside other tests in one process under plain `cargo test`,
    /// so it keeps a second dispatcher registered before it installs its
    /// subscriber (below).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_panic_in_a_flights_pass_fails_every_waiter_internal_b26e() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let (client, _injector, oid, answer) = rig(&tv, "fanout-flight-panic").await;
        let flights = SnapshotFlights::new(client.clone(), keys.clone());
        let estimate = LayerSize::from_too_large(&answer, None);
        let (w1, w2) = (
            flights.attach_or_create(oid, estimate),
            flights.attach_or_create(oid, estimate),
        );
        // Before `PanicAtFanoutWindow` is installed: tracing-core's
        // process-wide callsite interest cache would otherwise let a sibling
        // test's pass, on a thread with no subscriber, cache the `fan-out
        // window` event as `never` for this pass too (see the helper's doc).
        crate::observability::trace_export::test_support::keep_a_second_dispatcher_registered();
        let armed = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let budget = CapacityBudget::new(ROOMY);
        let (e1, e2) = tokio::time::timeout(
            READ_BUDGET,
            async { tokio::join!(w1.read(&budget), w2.read(&budget)) }.with_subscriber(
                PanicAtFanoutWindow {
                    armed: Arc::clone(&armed),
                },
            ),
        )
        .await
        .expect("both waiters end in time");
        assert!(
            !armed.load(Ordering::SeqCst),
            "fixture: the pass reached its first window and panicked"
        );
        for (who, ended) in [("w1", e1.map(drop)), ("w2", e2.map(drop))] {
            assert!(
                matches!(&ended, Err(FlightWaitError::Read(ServiceError::Internal(m))) if m.contains("panicked")),
                "{who}: a panic in the flight's pass is an Internal failure for every waiter: \
                 {ended:?}"
            );
        }
        assert!(flights.is_empty(), "a failed flight leaves the map");
        assert_eq!(client.fanout_threads().live_passes(), 0, "no pass is left");
        let budget = &client.catchup_build_budget;
        assert_eq!(
            (budget.reserved_bytes(), budget.overcommit_bytes()),
            (0, 0),
            "the flight's grant is back"
        );
    }

    /// The name of the thread the calling code runs on.
    fn this_thread() -> Option<String> {
        std::thread::current().name().map(str::to_owned)
    }

    /// The `watch-read` load with one pass on each of the first `n` threads:
    /// what least-loaded assignment, lowest index on a tie, leaves after `n`
    /// unfenced passes on idle threads.
    fn first_n_loaded(n: usize) -> [usize; WATCH_READ_THREADS] {
        std::array::from_fn(|i| usize::from(i < n))
    }

    /// Wait, within [`SIGNAL_BUDGET`], until `threads`' `watch-read` load is
    /// `expected`: a pass leaves its count when its future drops, on its
    /// own thread, so an abort's or a panic's reaches the count a moment
    /// after its caller has moved on.
    async fn load_settles(
        threads: &FanoutThreads,
        expected: [usize; WATCH_READ_THREADS],
        what: &str,
    ) {
        let deadline = Instant::now() + SIGNAL_BUDGET;
        while threads.watch_read_load() != expected {
            assert!(
                Instant::now() < deadline,
                "{what}: the `watch-read` load is {:?}, not {expected:?}, after {SIGNAL_BUDGET:?}",
                threads.watch_read_load()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// The thread the next pass to start reported over `ran_on`, within
    /// [`SIGNAL_BUDGET`].
    async fn next_ran_on(
        ran_on: &mut tokio::sync::mpsc::UnboundedReceiver<Option<String>>,
        what: &str,
    ) -> Option<String> {
        tokio::time::timeout(SIGNAL_BUDGET, ran_on.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: the pass did not start within {SIGNAL_BUDGET:?}"))
            .expect("the report channel stays open")
    }

    /// A pass that reports the thread it runs on over `ran_on`, then stays
    /// pending until its caller's drop aborts it.
    async fn parked_pass(
        ran_on: tokio::sync::mpsc::UnboundedSender<Option<String>>,
    ) -> Result<(), ServiceError> {
        let _ = ran_on.send(this_thread());
        std::future::pending().await
    }

    /// An unfenced [`parked_pass`]'s `run_pass` future, polled once: its
    /// first poll assigns the pass its `watch-read` thread, counts it there
    /// and spawns it, all before its first await, so the load reads the
    /// assignment as soon as this returns.
    fn start_parked<'a>(
        threads: &'a FanoutThreads,
        ran_on: &tokio::sync::mpsc::UnboundedSender<Option<String>>,
    ) -> std::pin::Pin<Box<impl Future<Output = Result<(), ServiceError>> + 'a>> {
        let ran_on = ran_on.clone();
        let mut pass = Box::pin(threads.run_pass(
            PassLane::Unfenced,
            move |_link| parked_pass(ran_on),
            |_| PassAnswer::GoOn,
        ));
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        assert!(
            pass.as_mut().poll(&mut cx).is_pending(),
            "the parked pass is pending"
        );
        pass
    }

    /// `run_pass` on its own: a pass that panics re-raises its panic, with
    /// its payload, on the caller, on either lane — a fenced pass ran on
    /// `fanout-read`, an unfenced one on a `watch-read` thread (the first, all
    /// being idle) — so the panic reached the caller's task rather than a
    /// plain error. The panicked unfenced pass leaves the `watch-read` load
    /// as it found it: a pass whose count were taken off only on completion
    /// would hold its thread's count for good.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_pass_re_raises_a_pass_panic_on_its_caller_on_either_lane_b26e() {
        let threads = FanoutThreads::new().expect("the fan-out threads start");
        for (lane, on) in [
            (PassLane::Fenced, FANOUT_READ_THREAD_NAME.to_owned()),
            (PassLane::Unfenced, watch_read_thread_name(0)),
        ] {
            let expected = format!("test: a pass panics on {on}");
            let caught = std::panic::AssertUnwindSafe(threads.run_pass(
                lane,
                move |_link| async move {
                    let thread = this_thread();
                    if thread.as_deref() == Some(on.as_str()) {
                        panic!("test: a pass panics on {on}");
                    }
                    Ok::<_, ServiceError>(thread)
                },
                |_| PassAnswer::GoOn,
            ))
            .catch_unwind()
            .await;
            let payload = match caught {
                Err(payload) => payload,
                Ok(ran) => panic!("{lane:?}: the pass must have panicked; it returned {ran:?}"),
            };
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                message, expected,
                "{lane:?}: the pass's own panic payload reaches the caller"
            );
            load_settles(
                &threads,
                [0; WATCH_READ_THREADS],
                &format!("{lane:?}: after the panicked pass"),
            )
            .await;
        }
        assert_eq!(threads.live_passes(), 0);
    }

    /// Lane isolation: with every `watch-read` thread held by a busy unfenced
    /// pass — each blocking its thread, as a CPU-bound WATCH read does — a
    /// fenced pass still runs to completion, on `fanout-read`. An unfenced
    /// pass routed to `fanout-read` would hold it, and the fenced pass would
    /// queue behind it. Once released, the busy passes complete, and the
    /// load returns to 0.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fenced_pass_completes_while_every_watch_read_thread_is_busy_b26e() {
        let threads = FanoutThreads::new().expect("the fan-out threads start");
        let (ran_on_tx, mut ran_on) = tokio::sync::mpsc::unbounded_channel();
        // Dropped on any exit, which unblocks every busy pass.
        let mut releases = Vec::new();
        let mut busy = Vec::new();
        for _ in 0..WATCH_READ_THREADS {
            let (release, released) = std::sync::mpsc::channel::<()>();
            releases.push(release);
            let ran_on_tx = ran_on_tx.clone();
            let mut pass = Box::pin(threads.run_pass(
                PassLane::Unfenced,
                move |_link| async move {
                    let _ = ran_on_tx.send(this_thread());
                    // Holds its thread until released: nothing else runs
                    // on it meanwhile.
                    let _ = released.recv();
                    Ok::<_, ServiceError>(())
                },
                |_| PassAnswer::GoOn,
            ));
            assert!(futures::poll!(pass.as_mut()).is_pending());
            busy.push(pass);
        }
        let mut busy_on = std::collections::BTreeSet::new();
        for i in 0..WATCH_READ_THREADS {
            busy_on.insert(next_ran_on(&mut ran_on, &format!("busy pass {i}")).await);
        }
        let watch_read: std::collections::BTreeSet<Option<String>> = (0..WATCH_READ_THREADS)
            .map(|i| Some(watch_read_thread_name(i)))
            .collect();
        assert_eq!(
            busy_on, watch_read,
            "fixture: one busy pass holds each `watch-read` thread"
        );

        let fenced = tokio::time::timeout(
            SIGNAL_BUDGET,
            threads.run_pass(
                PassLane::Fenced,
                |_link| async { Ok::<_, ServiceError>(this_thread()) },
                |_| PassAnswer::GoOn,
            ),
        )
        .await
        .expect("a fenced pass completes while every `watch-read` thread is busy")
        .expect("the fenced pass succeeds");
        assert_eq!(
            fenced.as_deref(),
            Some(FANOUT_READ_THREAD_NAME),
            "the fenced pass ran on `{FANOUT_READ_THREAD_NAME}`"
        );

        drop(releases);
        for pass in busy {
            tokio::time::timeout(SIGNAL_BUDGET, pass)
                .await
                .expect("a released busy pass completes")
                .expect("the busy pass succeeds");
        }
        load_settles(
            &threads,
            [0; WATCH_READ_THREADS],
            "every completed pass left its count",
        )
        .await;
        assert_eq!(threads.live_passes(), 0);
    }

    /// Assignment, least-loaded (module docs, *Assignment*): the first
    /// [`WATCH_READ_THREADS`] parked unfenced passes take a thread each,
    /// lowest index first; the next, every thread then holding one, goes to
    /// the lowest index, `watch-read-0`. A pass aborted on `watch-read-2`
    /// leaves its count there, and the next pass goes to that least-loaded
    /// thread — where a rotating index would pick `watch-read-1` and a fixed
    /// one `watch-read-0`. Once every pass is aborted, the load is 0.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_passes_go_to_the_least_loaded_watch_read_thread_lowest_index_on_a_tie_b26e() {
        const {
            assert!(
                WATCH_READ_THREADS >= 3,
                "fixture: the test frees `watch-read-2`"
            )
        };
        let threads = FanoutThreads::new().expect("the fan-out threads start");
        let (ran_on_tx, mut ran_on) = tokio::sync::mpsc::unbounded_channel();
        let mut parked = Vec::new();
        for i in 0..WATCH_READ_THREADS {
            parked.push(start_parked(&threads, &ran_on_tx));
            assert_eq!(
                threads.watch_read_load(),
                first_n_loaded(i + 1),
                "pass {i} is counted on `watch-read-{i}`, the lowest-index thread with no pass"
            );
            assert_eq!(
                next_ran_on(&mut ran_on, &format!("pass {i}")).await,
                Some(watch_read_thread_name(i)),
                "pass {i} runs on `watch-read-{i}`"
            );
        }
        assert_eq!(threads.watch_read_passes_started(), [1; WATCH_READ_THREADS]);
        assert_eq!(
            (
                threads.live_passes_on(PassLane::Unfenced),
                threads.passes_started_on(PassLane::Fenced)
            ),
            (WATCH_READ_THREADS, 0),
            "every parked pass is live on the unfenced lane, none ran on the fenced one"
        );

        let tie = start_parked(&threads, &ran_on_tx);
        let mut expected = [1; WATCH_READ_THREADS];
        expected[0] = 2;
        assert_eq!(
            threads.watch_read_load(),
            expected,
            "every thread holds one pass: the next goes to the lowest index"
        );
        assert_eq!(
            next_ran_on(&mut ran_on, "the pass on a tie").await,
            Some(watch_read_thread_name(0))
        );

        drop(parked.remove(2));
        expected[2] = 0;
        load_settles(
            &threads,
            expected,
            "the pass aborted on `watch-read-2` left its count",
        )
        .await;
        let least_loaded = start_parked(&threads, &ran_on_tx);
        expected[2] = 1;
        assert_eq!(
            threads.watch_read_load(),
            expected,
            "the next pass is counted on the least-loaded thread, `watch-read-2`"
        );
        assert_eq!(
            next_ran_on(&mut ran_on, "the pass after the abort").await,
            Some(watch_read_thread_name(2)),
            "the next pass runs on the least-loaded thread, `watch-read-2`"
        );
        let mut started = [1; WATCH_READ_THREADS];
        started[0] = 2;
        started[2] = 2;
        assert_eq!(threads.watch_read_passes_started(), started);

        drop((parked, tie, least_loaded));
        load_settles(
            &threads,
            [0; WATCH_READ_THREADS],
            "every aborted pass left its count",
        )
        .await;
        no_live_pass_on(&threads, "every parked pass was aborted").await;
    }

    /// Wait, within [`PASS_END_BUDGET`], until `threads` has no live pass.
    async fn no_live_pass_on(threads: &FanoutThreads, what: &str) {
        let deadline = Instant::now() + PASS_END_BUDGET;
        while threads.live_passes() != 0 {
            assert!(
                Instant::now() < deadline,
                "{what}: a pass still runs {PASS_END_BUDGET:?} after its caller was dropped"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Shutdown, then a pass: once the threads are shut down (as the
    /// client's last owner's drop shuts them down), a pass on either lane
    /// answers `Unavailable`, retryable, rather than running — every
    /// runtime is down, the `watch-read` ones included — and the unfenced
    /// pass leaves no count on the `watch-read` load. Through the client
    /// this is the internal clone's pass after the last owner dropped
    /// (`storage::client`'s `client_owners` tests); a public clone is an
    /// owner, so no public caller reaches it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pass_after_the_shutdown_answers_unavailable_on_either_lane_b26e() {
        let threads = FanoutThreads::new().expect("the fan-out threads start");
        threads.shut_down();
        for lane in [PassLane::Fenced, PassLane::Unfenced] {
            let ran = tokio::time::timeout(
                SIGNAL_BUDGET,
                threads.run_pass(
                    lane,
                    |_link| async { Ok::<_, ServiceError>(this_thread()) },
                    |_| PassAnswer::GoOn,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{lane:?}: the pass answers in time"));
            assert!(
                matches!(ran, Err(ServiceError::Unavailable(_))),
                "{lane:?}: a pass after the shutdown answers Unavailable: {ran:?}"
            );
        }
        load_settles(
            &threads,
            [0; WATCH_READ_THREADS],
            "the cancelled unfenced pass left its count",
        )
        .await;
        assert_eq!(threads.live_passes(), 0);
    }

    /// `run_pass` on its own: dropping its future aborts a pass parked on an
    /// await — gone within [`PASS_END_BUDGET`] — while its caller holds the
    /// threads, on either lane, and the aborted unfenced pass leaves the
    /// `watch-read` load; and dropping the threads inside a runtime (where a
    /// plain runtime drop would panic) stops every one of their
    /// [`FANOUT_RUNTIMES`] threads, the `watch-read` ones included.
    ///
    /// Each pass is dropped only once it has started on its thread
    /// ([`parked_pass`]): a pass aborted before its first poll drops its
    /// captures unstarted, which would hide a count released only when a
    /// started pass completes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_run_pass_aborts_its_pass_on_either_lane_and_dropped_threads_stop_b26e() {
        let threads = FanoutThreads::new().expect("the fan-out threads start");
        let alive = threads.alive_threads();
        let (ran_on_tx, mut ran_on) = tokio::sync::mpsc::unbounded_channel();
        for lane in [PassLane::Fenced, PassLane::Unfenced] {
            {
                let ran_on_tx = ran_on_tx.clone();
                let pass = threads.run_pass(
                    lane,
                    move |_link| parked_pass(ran_on_tx),
                    |_| PassAnswer::GoOn,
                );
                let mut pass = std::pin::pin!(pass);
                assert!(
                    futures::poll!(pass.as_mut()).is_pending(),
                    "{lane:?}: the pass is pending"
                );
                let on = next_ran_on(&mut ran_on, &format!("{lane:?}: the parked pass")).await;
                let expected = match lane {
                    PassLane::Fenced => FANOUT_READ_THREAD_NAME.to_owned(),
                    PassLane::Unfenced => watch_read_thread_name(0),
                };
                assert_eq!(
                    on.as_deref(),
                    Some(expected.as_str()),
                    "{lane:?}: the pass started on its lane's thread"
                );
                assert_eq!(
                    threads.live_passes_on(lane),
                    1,
                    "{lane:?}: the pass is live"
                );
                if lane == PassLane::Unfenced {
                    assert_eq!(
                        threads.watch_read_load(),
                        first_n_loaded(1),
                        "the parked unfenced pass is counted on `watch-read-0`"
                    );
                }
            }
            no_live_pass_on(&threads, &format!("{lane:?}: the dropped pass")).await;
            load_settles(
                &threads,
                [0; WATCH_READ_THREADS],
                &format!("{lane:?}: the aborted pass"),
            )
            .await;
        }
        // Bounded, not a single check: tokio runs `on_thread_start` in the
        // spawned worker, not before `build()` returns.
        let deadline = Instant::now() + SIGNAL_BUDGET;
        while alive.load(Ordering::SeqCst) < FANOUT_RUNTIMES {
            assert!(
                Instant::now() < deadline,
                "all {FANOUT_RUNTIMES} runtimes' worker threads start within {SIGNAL_BUDGET:?}: \
                 {} alive",
                alive.load(Ordering::SeqCst)
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        drop(threads);
        let deadline = Instant::now() + SIGNAL_BUDGET;
        while alive.load(Ordering::SeqCst) != 0 {
            assert!(
                Instant::now() < deadline,
                "dropping the threads stops them: {} still alive",
                alive.load(Ordering::SeqCst)
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    thread_local! {
        /// Calls of a start hook below on the calling thread.
        static START_HOOK_CALLS_HERE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Every call of [`counting_start_hook`], on any thread. Only
    /// [`a_start_hook_runs_once_on_every_fanout_thread_blocking_pool_threads_included_b26e`]
    /// builds threads with that hook, so nothing else moves it.
    static COUNTING_START_HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

    /// A start hook that counts its calls, on its thread and in total.
    fn counting_start_hook() {
        START_HOOK_CALLS_HERE.with(|calls| calls.set(calls.get() + 1));
        COUNTING_START_HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    /// A start hook that counts its calls on its thread only.
    fn marking_start_hook() {
        START_HOOK_CALLS_HERE.with(|calls| calls.set(calls.get() + 1));
    }

    /// One thread of a [`FanoutThreads`] as a task on it saw itself.
    #[derive(Debug)]
    struct ThreadSeen {
        /// The runtime the task was spawned on, by its thread name.
        runtime: String,
        /// The name of the thread the task ran on.
        ran_on: Option<String>,
        id: std::thread::ThreadId,
        /// Start-hook calls on that thread ([`START_HOOK_CALLS_HERE`]).
        hook_calls: usize,
    }

    /// What a task sees of the thread it runs on.
    fn see_this_thread(runtime: String) -> ThreadSeen {
        ThreadSeen {
            runtime,
            ran_on: this_thread(),
            id: std::thread::current().id(),
            hook_calls: START_HOOK_CALLS_HERE.with(std::cell::Cell::get),
        }
    }

    /// Every runtime of `threads`, each by its thread name, in
    /// [`FanoutThreads`]' order.
    fn named_runtimes(threads: &FanoutThreads) -> Vec<(String, &Handle)> {
        let mut runtimes = vec![
            (
                FANOUT_READ_THREAD_NAME.to_owned(),
                threads.fanout_read_handle(),
            ),
            (
                SNAPSHOT_IO_THREAD_NAME.to_owned(),
                threads.snapshot_io_handle(),
            ),
        ];
        runtimes.extend(
            threads
                .watch_read_handles()
                .iter()
                .enumerate()
                .map(|(i, handle)| (watch_read_thread_name(i), handle)),
        );
        runtimes
    }

    /// See every runtime's worker thread, and one blocking-pool thread each
    /// runtime starts for it here, within [`SIGNAL_BUDGET`] each. Each
    /// blocking task stays on its thread until the returned senders drop,
    /// so every thread seen is still alive while the caller holds them.
    async fn see_every_thread(
        threads: &FanoutThreads,
    ) -> (Vec<ThreadSeen>, Vec<std::sync::mpsc::Sender<()>>) {
        let mut seen = Vec::new();
        let mut holds = Vec::new();
        for (runtime, handle) in named_runtimes(threads) {
            let on_worker = runtime.clone();
            let worker = tokio::time::timeout(
                SIGNAL_BUDGET,
                handle.spawn(async move { see_this_thread(on_worker) }),
            )
            .await
            .unwrap_or_else(|_| panic!("{runtime}: its worker answers in time"))
            .expect("the worker task completes");
            let (hold, held) = std::sync::mpsc::channel::<()>();
            holds.push(hold);
            let (report, reported) = tokio::sync::oneshot::channel();
            let on_blocking = runtime.clone();
            drop(handle.spawn_blocking(move || {
                let _ = report.send(see_this_thread(on_blocking));
                // Ends when the caller drops its sender.
                let _ = held.recv();
            }));
            let blocking = tokio::time::timeout(SIGNAL_BUDGET, reported)
                .await
                .unwrap_or_else(|_| panic!("{runtime}: its blocking pool answers in time"))
                .expect("the blocking task reports");
            seen.push(worker);
            seen.push(blocking);
        }
        (seen, holds)
    }

    /// A start hook given (`with_thread_start_hook_for_test`) runs exactly
    /// once on every thread the [`FANOUT_RUNTIMES`] runtimes start: each
    /// runtime's worker and the blocking-pool thread it starts later, before
    /// either runs a task. Its total calls equal the threads started across
    /// the runtimes, so it ran on no other thread and on none twice.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_start_hook_runs_once_on_every_fanout_thread_blocking_pool_threads_included_b26e() {
        let calls_before = COUNTING_START_HOOK_CALLS.load(Ordering::SeqCst);
        let threads = FanoutThreads::with_thread_start_hook_for_test(Some(counting_start_hook))
            .expect("the fan-out threads start");
        let alive = threads.alive_threads();
        let (seen, holds) = see_every_thread(&threads).await;

        for thread in &seen {
            assert_eq!(
                thread.ran_on.as_deref(),
                Some(thread.runtime.as_str()),
                "fixture: the task ran on a thread of `{}`",
                thread.runtime
            );
            assert_eq!(
                thread.hook_calls, 1,
                "a thread of `{}` must have run the start hook exactly once before its task: \
                 {thread:?}",
                thread.runtime
            );
        }
        let distinct: std::collections::HashSet<_> = seen.iter().map(|t| t.id).collect();
        assert_eq!(
            distinct.len(),
            2 * FANOUT_RUNTIMES,
            "fixture: each runtime's worker and its blocking-pool thread are distinct threads"
        );
        // Every thread seen is held alive, and none of them can be counted
        // before its hook ran (`on_thread_start` counts it after the hook).
        let started = alive.load(Ordering::SeqCst);
        assert_eq!(
            started,
            2 * FANOUT_RUNTIMES,
            "fixture: the runtimes started exactly the threads seen"
        );
        assert_eq!(
            COUNTING_START_HOOK_CALLS.load(Ordering::SeqCst) - calls_before,
            started,
            "the start hook's calls must equal the threads the runtimes started"
        );
        drop(holds);
    }

    /// No start hook (`None`, as in a process that registered none, such as
    /// this library's test binaries): the runtimes start, a pass runs on
    /// each lane on its thread, and no thread of theirs runs a hook — while
    /// threads built with a hook in the same process run it on each of
    /// theirs, so a hook reaching the threads of another
    /// [`FanoutThreads`] fails here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_threads_without_a_start_hook_start_run_passes_and_run_no_hook_b26e() {
        let hooked = FanoutThreads::with_thread_start_hook_for_test(Some(marking_start_hook))
            .expect("the hooked fan-out threads start");
        let plain = FanoutThreads::with_thread_start_hook_for_test(None)
            .expect("the fan-out threads start with no hook");
        let (hooked_seen, hooked_holds) = see_every_thread(&hooked).await;
        let (plain_seen, plain_holds) = see_every_thread(&plain).await;

        for thread in &hooked_seen {
            assert_eq!(
                thread.hook_calls, 1,
                "fixture: the hooked threads run their hook: {thread:?}"
            );
        }
        assert_eq!(plain_seen.len(), 2 * FANOUT_RUNTIMES, "fixture");
        for thread in &plain_seen {
            assert_eq!(
                thread.ran_on.as_deref(),
                Some(thread.runtime.as_str()),
                "fixture: the task ran on a thread of `{}`",
                thread.runtime
            );
            assert_eq!(
                thread.hook_calls, 0,
                "a thread of fan-out threads built with no start hook must run none: {thread:?}"
            );
        }
        for (lane, expected) in [
            (PassLane::Fenced, FANOUT_READ_THREAD_NAME.to_owned()),
            (PassLane::Unfenced, watch_read_thread_name(0)),
        ] {
            let ran = tokio::time::timeout(
                SIGNAL_BUDGET,
                plain.run_pass(
                    lane,
                    |_link| async { Ok::<_, ServiceError>(this_thread()) },
                    |_| PassAnswer::GoOn,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{lane:?}: the pass answers in time"))
            .unwrap_or_else(|e| panic!("{lane:?}: the pass succeeds: {e:?}"));
            assert_eq!(
                ran.as_deref(),
                Some(expected.as_str()),
                "{lane:?}: with no start hook the pass runs on its lane's thread"
            );
        }
        drop((hooked_holds, plain_holds));
    }
}
