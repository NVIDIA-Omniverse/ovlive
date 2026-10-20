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

//! The measured thread and the test side's handle on it.
//!
//! The measured thread B is spawned with an explicit
//! [`PRODUCT_WORKER_STACK`] and runs a current-thread tokio runtime whose
//! blocking pool gets the same explicit size, so neither inherits
//! `RUST_MIN_STACK`. B serves the service the way `main.rs` builds the
//! full-API server — `production_server_builder()`, `MetricsLayer`, then
//! `live_service_server` — so every task the server spawns for an RPC (the
//! connection, each HTTP/2 stream, a subscription's task, an overwrite's
//! cleanup) is polled on B's stack. B's protocol, from its entry frame:
//!
//! 1. Check its own stack is worker-sized ([`check_worker_stack`]).
//! 2. `block_on(setup)`: get a worker-sized blocking-pool thread, bind and
//!    spawn the server, report the address, and await "open"; the test side
//!    connects and does its unmeasured warm-up in the meantime.
//! 3. Paint the unused stack below the entry frame.
//! 4. `block_on(window)`: report "painted", then await "close" (or its
//!    sender's drop, so a test that fails mid-window cannot hang B).
//! 5. Scan for the deepest write: the measurement.
//! 6. `block_on(teardown)`: cancel the service shutdown token, so live
//!    streams end, then shut the server down gracefully under a bound.
//!
//! **A requested stack size is not always the size a thread gets.** glibc
//! keeps the stacks of exited threads and hands one to a new thread whose
//! request it fits, so a thread asking for 2 MiB can be given the 8 MiB
//! stack of a thread that inherited `RUST_MIN_STACK` and has exited (seen
//! here: a blocking-pool thread spawned after two joined 8 MiB threads got
//! 8 MiB). B, and the blocking-pool thread `setup` reads, therefore check
//! their own stacks, and one that is not worker-sized holds its stack —
//! parked, so the next spawn cannot be handed it again — while another is
//! spawned, up to [`SPAWN_ATTEMPTS`] times. A size that never takes — an
//! explicit size dropped from the code, so every spawn inherits
//! `RUST_MIN_STACK` — fails the case with the check's message.
//!
//! The test side ([`Measurement`]) runs on the test's own runtime; every wait
//! there is bounded by [`SETTLE`].
//!
//! **The fan-out threads.** Every snapshot fan-out pass an RPC starts runs on
//! one of the served client's pass threads (`storage::fanout_thread`), not
//! on B: a fenced read's on `fanout-read`, the structural WATCH read's and an
//! ext-GC reference walk's on one of the `watch-read-<i>` threads. So each pass
//! thread's depth is measured too ([`FanoutMeasure`], one [`ThreadMeasure`]
//! per thread, named by its role): when the window opens, a task on each
//! paints that thread's unused stack below the task's own frame (the depth
//! every task's poll starts at, the runtime's scheduler loop above it); when
//! the window closes, another scans it. The threads are the service's, built
//! with an explicit 2 MiB stack, so each is checked worker-sized but cannot
//! be re-spawned: a glibc-cached larger stack handed to one fails the case
//! with the check's message.

use std::hint::black_box;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tonic::transport::Channel;

use live_edit::config::RPC_DEADLINE_MS_DEFAULT;
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::{live_service_server, production_server_builder};
use live_edit::storage::fanout_thread::{
    watch_read_thread_name, PassLane, FANOUT_READ_THREAD_NAME, WATCH_READ_THREADS,
};

use crate::probe::{
    calibration_frame, check_worker_stack, deepest_write, paint_below, Painted, StackBounds,
    PRODUCT_WORKER_STACK,
};

/// Bound on every single wait on the test side: B's start, its address, the
/// paint, one RPC or drain, B's join. Unreachable on the healthy path.
pub const SETTLE: Duration = Duration::from_secs(20);

/// Bound on B's graceful server shutdown. Past it B drops its runtime, which
/// drops whatever the server still runs.
const TEARDOWN: Duration = Duration::from_secs(10);

/// How many threads B's spawn, and the blocking-pool thread `setup` reads,
/// may try before a worker-sized stack: above the number of 8 MiB stacks
/// glibc's default stack cache (40 MiB) can hold.
pub const SPAWN_ATTEMPTS: usize = 8;

/// The `service` label of the measured service's metrics (the one the
/// fixtures' registries carry).
pub const SERVICE_LABEL: &str = "test";

/// `LIVE_EDIT_SESSION_LIST_TIMEOUT_MS`'s default, which `main.rs` hands the
/// metrics layer.
const SESSION_LIST_TIMEOUT_MS_DEFAULT: u64 = 2_000;

/// What B runs.
#[derive(Clone)]
pub enum Work {
    /// Serve `LiveEditService` through the production server stack.
    Serve(Box<LiveEditService>),
    /// Serve nothing; run [`calibration_frame`] in a spawned task inside the
    /// window (the probe's calibration).
    Calibrate,
}

/// One measurement's result, read on B.
#[derive(Debug)]
pub struct Probe {
    /// B's own stack.
    pub thread: StackBounds,
    /// The stack of the blocking-pool thread of B's runtime.
    pub blocking: StackBounds,
    /// The address of the anchor in B's entry frame.
    pub entry: usize,
    pub painted: Painted,
    /// The deepest word written inside the window.
    pub deepest: Option<usize>,
    /// The served client's pass threads over the same window; `None` for
    /// [`Work::Calibrate`], which serves nothing.
    pub fanout: Option<FanoutMeasure>,
}

/// One of the served client's pass threads over one window
/// (`storage::fanout_thread`): its role, its depth below the frame a task's
/// poll starts at, its stack, and the passes it ran.
#[derive(Debug, Clone)]
pub struct ThreadMeasure {
    /// The thread's name, which is its role: `fanout-read` or
    /// `watch-read-<i>`.
    pub role: String,
    /// The thread's stack.
    pub thread: StackBounds,
    /// The address of the anchor in the painting task's frame.
    pub entry: usize,
    pub painted: Painted,
    /// The deepest word written inside the window.
    pub deepest: Option<usize>,
    /// Fan-out passes started on this thread inside the window.
    pub passes: usize,
}

impl ThreadMeasure {
    /// Bytes of stack used below the painting task's frame during the
    /// window: what a pass's poll took below the scheduler's.
    pub fn depth(&self) -> usize {
        self.entry - self.deepest.unwrap_or(self.painted.hi)
    }
}

/// The served client's pass threads over one window, and `snapshot-io`'s
/// stack.
#[derive(Debug, Clone)]
pub struct FanoutMeasure {
    /// `fanout-read`, then `watch-read-0` … in index order.
    pub threads: Vec<ThreadMeasure>,
    /// `snapshot-io`'s stack.
    pub snapshot_io: StackBounds,
}

impl FanoutMeasure {
    /// Fenced passes started inside the window: those `fanout-read` ran.
    pub fn fenced_passes(&self) -> usize {
        self.threads[0].passes
    }

    /// Unfenced passes started inside the window: those the `watch-read`
    /// threads ran.
    pub fn unfenced_passes(&self) -> usize {
        self.threads[1..].iter().map(|t| t.passes).sum()
    }
}

/// One pass thread's side of an open window.
struct ThreadWindow {
    role: String,
    handle: tokio::runtime::Handle,
    entry: usize,
    painted: Painted,
    thread: StackBounds,
}

/// The pass threads' side of an open window ([`Measurement::open`]).
struct FanoutWindow {
    client: live_edit::storage::client::ValkeyClient,
    threads: Vec<ThreadWindow>,
    snapshot_io: StackBounds,
    /// Fenced passes, then each `watch-read` thread's, started so far.
    passes_before: Vec<usize>,
}

/// Paint the calling thread's unused stack below this frame: run inside a
/// task on a pass thread, so the frame sits where every task's poll starts.
/// `#[inline(never)]` so the anchor is this frame's own.
#[inline(never)]
fn paint_from_this_frame() -> (usize, Painted, StackBounds) {
    let anchor = 0u64;
    let entry = black_box(&anchor) as *const u64 as usize;
    let thread = StackBounds::of_current_thread();
    let painted = paint_below(entry, &thread);
    (entry, painted, thread)
}

/// The passes started so far on each of `client`'s pass threads:
/// `fanout-read`'s (the fenced lane's), then each `watch-read` thread's.
fn passes_per_thread(threads: &live_edit::storage::fanout_thread::FanoutThreads) -> Vec<usize> {
    std::iter::once(threads.passes_started_on(PassLane::Fenced))
        .chain(threads.watch_read_passes_started())
        .collect()
}

impl FanoutWindow {
    /// Open the window on `client`'s pass threads: paint each, and read
    /// their stacks, `snapshot-io`'s and the passes started so far.
    async fn open(client: live_edit::storage::client::ValkeyClient) -> Self {
        let fanout = client.fanout_threads();
        let roles = std::iter::once((
            FANOUT_READ_THREAD_NAME.to_owned(),
            fanout.fanout_read_handle().clone(),
        ))
        .chain(
            fanout
                .watch_read_handles()
                .iter()
                .enumerate()
                .map(|(i, handle)| (watch_read_thread_name(i), handle.clone())),
        );
        let mut threads = Vec::with_capacity(WATCH_READ_THREADS + 1);
        for (role, handle) in roles {
            let (entry, painted, thread) =
                tokio::time::timeout(SETTLE, handle.spawn(async { paint_from_this_frame() }))
                    .await
                    .unwrap_or_else(|_| panic!("the {role} thread paints in time"))
                    .unwrap_or_else(|e| panic!("the {role} thread's paint task must finish: {e}"));
            threads.push(ThreadWindow {
                role,
                handle,
                entry,
                painted,
                thread,
            });
        }
        let snapshot_io = tokio::time::timeout(
            SETTLE,
            fanout
                .snapshot_io_handle()
                .spawn(async { StackBounds::of_current_thread() }),
        )
        .await
        .expect("the snapshot-io thread answers in time")
        .expect("the snapshot-io thread's task must finish");
        let passes_before = passes_per_thread(fanout);
        FanoutWindow {
            client,
            threads,
            snapshot_io,
            passes_before,
        }
    }

    /// Close the window: scan each pass thread's painted stack.
    async fn finish(self) -> FanoutMeasure {
        let passes_after = passes_per_thread(self.client.fanout_threads());
        let mut threads = Vec::with_capacity(self.threads.len());
        for (i, window) in self.threads.into_iter().enumerate() {
            let painted = window.painted;
            let deepest = tokio::time::timeout(
                SETTLE,
                window.handle.spawn(async move { deepest_write(&painted) }),
            )
            .await
            .unwrap_or_else(|_| panic!("the {} thread scans in time", window.role))
            .unwrap_or_else(|e| panic!("the {} thread's scan task must finish: {e}", window.role));
            threads.push(ThreadMeasure {
                role: window.role,
                thread: window.thread,
                entry: window.entry,
                painted: window.painted,
                deepest,
                passes: passes_after[i] - self.passes_before[i],
            });
        }
        FanoutMeasure {
            threads,
            snapshot_io: self.snapshot_io,
        }
    }
}

impl Probe {
    /// Bytes of stack used below the entry frame during the window. With no
    /// write in the painted range, the most it can have been: the unpainted
    /// reserve.
    pub fn depth(&self) -> usize {
        self.entry - self.deepest.unwrap_or(self.painted.hi)
    }

    /// Bytes of stack below the entry frame: what the window could have used
    /// before the guard.
    pub fn usable(&self) -> usize {
        self.entry - self.thread.lo
    }
}

/// How B's start went.
enum Started {
    /// B's own stack is not worker-sized: the check's message. B holds its
    /// stack until released.
    Unfit(String),
    /// No blocking-pool thread got a worker-sized stack: the check's
    /// message.
    BlockingUnfit(String),
    /// Serving at the address (none for [`Work::Calibrate`]), after this
    /// many blocking-pool threads.
    Serving {
        addr: Option<SocketAddr>,
        blocking_spawns: usize,
    },
}

/// The channels between the test side and B.
struct Control {
    started_tx: oneshot::Sender<Started>,
    release_rx: oneshot::Receiver<()>,
    open_rx: oneshot::Receiver<()>,
    painted_tx: oneshot::Sender<()>,
    close_rx: oneshot::Receiver<()>,
}

/// The server B spawned.
struct Served {
    shutdown_tx: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

/// What `setup` hands back to B's body.
struct SetUp {
    served: Option<Served>,
    blocking: StackBounds,
    /// Releases the blocking-pool threads that were not worker-sized.
    held: Vec<std::sync::mpsc::Sender<()>>,
}

/// B's body: `None` when B gave up its place (its stack was not
/// worker-sized, or no blocking-pool thread's was). `#[inline(never)]` so
/// its frame — the entry frame every measured frame lies below — is its
/// own. Everything it keeps across the paint is small or boxed, and every
/// future it blocks on is boxed, so nothing large sits in the frames the
/// measurement counts beyond what a worker's scheduler loop would hold.
#[inline(never)]
fn measured_thread(work: Work, control: Control) -> Option<Probe> {
    let anchor = 0u64;
    let entry = black_box(&anchor) as *const u64 as usize;
    let thread = StackBounds::of_current_thread();
    let Control {
        started_tx,
        release_rx,
        open_rx,
        painted_tx,
        close_rx,
    } = control;
    if let Err(unfit) = check_worker_stack("the measured thread", &thread) {
        let _ = started_tx.send(Started::Unfit(unfit));
        let _ = release_rx.blocking_recv();
        return None;
    }
    let runtime = Box::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .thread_stack_size(PRODUCT_WORKER_STACK)
            .build()
            .expect("the measured runtime must build"),
    );

    let set_up = runtime.block_on(Box::pin(setup(&work, started_tx, open_rx)))?;
    let painted = paint_below(entry, &thread);
    runtime.block_on(Box::pin(window(&work, painted_tx, close_rx)));
    let deepest = deepest_write(&painted);
    runtime.block_on(Box::pin(teardown(&work, set_up.served)));
    // Before the runtime's drop, which waits for its blocking threads.
    drop(set_up.held);
    drop(runtime);

    Some(Probe {
        thread,
        blocking: set_up.blocking,
        entry,
        painted,
        deepest,
        fanout: None,
    })
}

/// A blocking-pool thread of the current runtime with a worker-sized stack,
/// left idle in the pool for the window's blocking work, and the releases of
/// the threads tried before it, each kept busy (holding its stack) until
/// released. `Err` with the check's message when none of
/// [`SPAWN_ATTEMPTS`] is worker-sized.
async fn worker_sized_blocking_thread(
) -> Result<(StackBounds, Vec<std::sync::mpsc::Sender<()>>), String> {
    let mut held = Vec::new();
    let mut unfit = String::new();
    for _ in 0..SPAWN_ATTEMPTS {
        let (bounds_tx, bounds_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        tokio::task::spawn_blocking(move || {
            let bounds = StackBounds::of_current_thread();
            let fits = check_worker_stack("", &bounds).is_ok();
            let _ = bounds_tx.send(bounds);
            if !fits {
                let _ = release_rx.recv();
            }
        });
        let bounds = bounds_rx
            .await
            .expect("a blocking-pool thread must report its bounds");
        match check_worker_stack("the measured runtime's blocking-pool thread", &bounds) {
            Ok(()) => return Ok((bounds, held)),
            Err(e) => {
                held.push(release_tx);
                unfit = e;
            }
        }
    }
    Err(unfit)
}

async fn setup(
    work: &Work,
    started_tx: oneshot::Sender<Started>,
    open_rx: oneshot::Receiver<()>,
) -> Option<SetUp> {
    let (blocking, held) = match worker_sized_blocking_thread().await {
        Ok(found) => found,
        Err(unfit) => {
            let _ = started_tx.send(Started::BlockingUnfit(unfit));
            return None;
        }
    };
    let blocking_spawns = held.len() + 1;
    let served = match work {
        Work::Serve(svc) => {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("the measured server must bind");
            let addr = listener.local_addr().expect("the listener has an address");
            // As `main.rs` builds the full-API server: the metrics layer over
            // the service's own registry, the production deadlines.
            let metrics_layer = MetricsLayer::new(
                Arc::clone(&svc.metrics),
                SERVICE_LABEL,
                RPC_DEADLINE_MS_DEFAULT,
                SESSION_LIST_TIMEOUT_MS_DEFAULT,
            );
            let router = production_server_builder()
                .layer(metrics_layer)
                .add_service(live_service_server(LiveEditService::clone(svc)));
            let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
            let task = tokio::spawn(router.serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async move {
                    let _ = shutdown_rx.await;
                },
            ));
            let _ = started_tx.send(Started::Serving {
                addr: Some(addr),
                blocking_spawns,
            });
            Some(Served { shutdown_tx, task })
        }
        Work::Calibrate => {
            let _ = started_tx.send(Started::Serving {
                addr: None,
                blocking_spawns,
            });
            None
        }
    };
    // Dropped unsent when the test side failed: proceed, measure nothing it
    // asks for, and exit.
    let _ = open_rx.await;
    Some(SetUp {
        served,
        blocking,
        held,
    })
}

async fn window(work: &Work, painted_tx: oneshot::Sender<()>, close_rx: oneshot::Receiver<()>) {
    let _ = painted_tx.send(());
    if let Work::Calibrate = work {
        tokio::spawn(async { black_box(calibration_frame()) })
            .await
            .expect("the calibration task must finish");
    }
    let _ = close_rx.await;
}

async fn teardown(work: &Work, served: Option<Served>) {
    if let Work::Serve(svc) = work {
        svc.shutdown.cancel();
    }
    if let Some(served) = served {
        let _ = served.shutdown_tx.send(());
        match tokio::time::timeout(TEARDOWN, served.task).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => eprintln!("stack-guard: the measured server ended with {e}"),
            Ok(Err(e)) => eprintln!("stack-guard: the measured server task failed: {e}"),
            Err(_) => eprintln!(
                "stack-guard: the measured server did not stop within {TEARDOWN:?}; dropping it"
            ),
        }
    }
}

/// The test side's handle on one measurement.
pub struct Measurement {
    pub case: &'static str,
    /// Threads spawned to get a worker-sized B, B included.
    pub thread_spawns: usize,
    /// Blocking-pool threads B's runtime spawned to get a worker-sized one.
    pub blocking_spawns: usize,
    addr: Option<SocketAddr>,
    open_tx: Option<oneshot::Sender<()>>,
    painted_rx: Option<oneshot::Receiver<()>>,
    close_tx: Option<oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<Option<Probe>>>,
    /// The served client, whose pass threads the window also measures;
    /// `None` for [`Work::Calibrate`], and once the window opened.
    served: Option<live_edit::storage::client::ValkeyClient>,
    /// The pass threads' side of the open window.
    fanout: Option<FanoutWindow>,
    /// Releases the spawns whose stacks were not worker-sized, on drop.
    _held: Vec<oneshot::Sender<()>>,
}

impl Measurement {
    /// Spawn B for `case`, named `stack-guard:<case>` — the name std's
    /// overflow abort prints — and wait for it to serve.
    pub async fn start(case: &'static str, work: Work) -> Self {
        let served = match &work {
            Work::Serve(svc) => Some(svc.valkey.clone()),
            Work::Calibrate => None,
        };
        let mut held = Vec::new();
        let mut unfit = String::new();
        for thread_spawns in 1..=SPAWN_ATTEMPTS {
            let (started_tx, started_rx) = oneshot::channel();
            let (release_tx, release_rx) = oneshot::channel();
            let (open_tx, open_rx) = oneshot::channel();
            let (painted_tx, painted_rx) = oneshot::channel();
            let (close_tx, close_rx) = oneshot::channel();
            let control = Control {
                started_tx,
                release_rx,
                open_rx,
                painted_tx,
                close_rx,
            };
            let attempt = work.clone();
            let thread = std::thread::Builder::new()
                .name(format!("stack-guard:{case}"))
                .stack_size(PRODUCT_WORKER_STACK)
                .spawn(move || measured_thread(attempt, control))
                .expect("the measured thread must spawn");
            let started = match tokio::time::timeout(SETTLE, started_rx).await {
                Ok(Ok(started)) => started,
                Ok(Err(_)) => join_failed(case, thread, "before it started").await,
                Err(_) => panic!("{case}: the measured thread did not start within {SETTLE:?}"),
            };
            match started {
                Started::Unfit(e) => {
                    held.push(release_tx);
                    unfit = e;
                }
                Started::BlockingUnfit(e) => {
                    panic!("{case}: thread-bounds self-check: after {SPAWN_ATTEMPTS} tries: {e}")
                }
                Started::Serving {
                    addr,
                    blocking_spawns,
                } => {
                    return Measurement {
                        case,
                        thread_spawns,
                        blocking_spawns,
                        addr,
                        open_tx: Some(open_tx),
                        painted_rx: Some(painted_rx),
                        close_tx: Some(close_tx),
                        thread: Some(thread),
                        served,
                        fanout: None,
                        _held: held,
                    }
                }
            }
        }
        panic!("{case}: thread-bounds self-check: after {SPAWN_ATTEMPTS} tries: {unfit}")
    }

    /// A client of the measured server, warmed up with one `Ping` so the
    /// connection's setup is not inside the window.
    pub async fn client(&self) -> LiveServiceClient<Channel> {
        let addr = self.addr.expect("this measurement serves nothing");
        let channel = tokio::time::timeout(
            SETTLE,
            Channel::from_shared(format!("http://{addr}"))
                .expect("a valid endpoint")
                .connect(),
        )
        .await
        .unwrap_or_else(|_| panic!("{}: the client did not connect in time", self.case))
        .unwrap_or_else(|e| panic!("{}: the client could not connect: {e}", self.case));
        let mut client = LiveServiceClient::new(channel);
        tokio::time::timeout(
            SETTLE,
            client.ping(crate::fixtures::authed(
                live_edit::grpc::proto::PingRequest {
                    client_timestamp_microseconds_since_epoch: 0,
                },
            )),
        )
        .await
        .unwrap_or_else(|_| panic!("{}: the warm-up ping did not answer in time", self.case))
        .unwrap_or_else(|s| panic!("{}: the warm-up ping failed: {s:?}", self.case));
        client
    }

    /// Open the window: B paints, then each of the served client's pass
    /// threads does, and from the return on everything any of them runs is
    /// measured.
    pub async fn open(&mut self) {
        let _ = self.open_tx.take().expect("the window opens once").send(());
        let painted = self.painted_rx.take().expect("the window opens once");
        match tokio::time::timeout(SETTLE, painted).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => {
                let thread = self.thread.take().expect("joined once");
                join_failed(self.case, thread, "before painting").await
            }
            Err(_) => panic!("{}: the measured thread did not paint in time", self.case),
        }
        if let Some(client) = self.served.take() {
            self.fanout = Some(FanoutWindow::open(client).await);
        }
    }

    /// Close the window and collect the measurement: the pass threads' scans
    /// first, then B's.
    pub async fn finish(mut self) -> Probe {
        let fanout = match self.fanout.take() {
            Some(window) => Some(window.finish().await),
            None => None,
        };
        let mut probe = self.finish_measured_thread().await;
        probe.fanout = fanout;
        probe
    }

    /// Close B's window and collect B's measurement.
    async fn finish_measured_thread(mut self) -> Probe {
        let _ = self
            .close_tx
            .take()
            .expect("the window closes once")
            .send(());
        let thread = self.thread.take().expect("joined once");
        let case = self.case;
        match tokio::time::timeout(SETTLE, tokio::task::spawn_blocking(move || thread.join())).await
        {
            Ok(Ok(Ok(Some(probe)))) => probe,
            Ok(Ok(Ok(None))) => panic!("{case}: the measured thread measured nothing"),
            Ok(Ok(Err(panic))) => std::panic::resume_unwind(panic),
            Ok(Err(e)) => panic!("{case}: joining the measured thread failed: {e}"),
            Err(_) => panic!("{case}: the measured thread did not finish within {SETTLE:?}"),
        }
    }
}

/// B ended early: surface its panic.
async fn join_failed(case: &str, thread: std::thread::JoinHandle<Option<Probe>>, when: &str) -> ! {
    match tokio::time::timeout(SETTLE, tokio::task::spawn_blocking(move || thread.join())).await {
        Ok(Ok(Err(panic))) => std::panic::resume_unwind(panic),
        _ => panic!("{case}: the measured thread ended {when}"),
    }
}
