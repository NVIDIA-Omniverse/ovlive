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

//! The release-build stack guard (SDD §15.1): in the build that ships, every
//! covered `LiveService` RPC route uses at most [`STACK_BUDGET`] (512 KiB) of
//! the [`PRODUCT_WORKER_STACK`] (2 MiB) a tokio worker of the service has.
//!
//! # Why release, and why a guard
//!
//! The service's workers get Rust's default 2 MiB stack: `main.rs`'s
//! `#[tokio::main]` sets no `thread_stack_size`, and nothing sets
//! `RUST_MIN_STACK` in the deployment. Stack depth in a debug build is not a
//! product signal: a debug build gives each inline future its own stack
//! slot, so nesting adds up to many times the release depth, unevenly by
//! path. The suite's debug tests run with an 8 MiB `RUST_MIN_STACK`
//! (`.cargo/config.toml`), so a debug overflow is a test-lane problem, fixed
//! in the test. This binary measures the release build instead, and fails
//! with a number when a route's depth grows past the budget.
//!
//! Every test here is `#[ignore]`d in a debug build (the `test:` job lists
//! them as skipped). They run with
//!
//! ```text
//! cargo nextest run --cargo-profile release -p live-edit --features test-support \
//!     --test release_stack_guard
//! ```
//!
//! which the must-pass `test-release-stack-guard:` CI job runs (with
//! `--no-tests=fail`). `TEST_VALKEY_URL` must name a reachable Valkey. The
//! binary is built for linux-gnu only (the probe reads its thread's stack
//! with `pthread_getattr_np`); elsewhere it is empty.
//!
//! # What is measured
//!
//! Each case serves the service through the production server stack —
//! `production_server_builder()`, `MetricsLayer`, the auth interceptor and
//! `live_service_server`, as `main.rs` builds the full-API server — on a
//! thread spawned with an explicit [`PRODUCT_WORKER_STACK`], running a
//! current-thread runtime whose blocking pool gets the same explicit size
//! (`harness`). Everything the server runs for an RPC is polled on that
//! thread: the connection, each HTTP/2 stream, the handler, a
//! subscription's spawned task (where catch-up and checkpoint recovery
//! run), an overwrite's cleanup task. The production log subscriber is
//! installed (JSON at `info`, into a sink), so event formatting is counted,
//! and the global allocator is jemalloc, as in the service binary. The
//! client, the Valkey connections and the fixtures' setup live on the test's
//! own runtime.
//!
//! The one exception is the snapshot fan-out: every fan-out pass an RPC
//! starts runs on one of the served client's pass threads — a fenced read's
//! on `fanout-read`, the structural WATCH read's and an ext-GC reference walk's on one
//! of the `watch-read-<i>` threads — and the snapshot lane's routers on its
//! `snapshot-io` thread (`storage::fanout_thread`), each with an explicit
//! 2 MiB stack. Every serving case measures each pass thread over the same
//! window, each against the same budget, and checks every one of them and
//! `snapshot-io` is worker-sized (`harness::FanoutMeasure`). The cases whose
//! route reads by a fan-out pass also check that a pass ran on the thread
//! its lane names: on `fanout-read` for the fenced-read cases and the first
//! subscribes at `MAX_SECTION_TREE_DEPTH`, on a `watch-read` thread for the
//! WATCH cases and the overwrite (its cleanup's ext-GC reference walk).
//!
//! The probe (`probe`) paints the measured thread's unused stack from its
//! entry frame, opens a window in which the case's RPC runs, and reports how
//! deep the deepest write went. Each case checks the route it took — a
//! metric delta or the RPC's result — so it cannot pass on another path, and
//! prints one line, on a pass and on a failure: the case, the KiB used, the
//! budget, the usable stack below the entry frame, the headroom, the KiB each
//! pass thread used and the fan-out passes it ran (`fanout-read … over N
//! pass(es), watch-read-0 … over N pass(es), …`), whether the route was
//! taken, and `PASS` or `FAIL`.
//! `.config/nextest.toml` keeps that line in the log and the JUnit XML on a
//! pass.
//!
//! Two self-tests run with the cases: a 256 KiB calibration frame must
//! measure within its bounds (a probe reporting about zero fails it), and
//! the thread-bounds check must reject an 8 MiB stack, explicit or
//! inherited from `RUST_MIN_STACK`, while the measured thread and its
//! runtime's blocking-pool thread pass it. Every case runs that check on
//! both. glibc can hand a 2 MiB request the cached stack of an exited 8 MiB
//! thread, so the harness re-spawns either thread until its stack is
//! worker-sized; an explicit size dropped from the code fails every case.
//!
//! One relative-depth check runs with them
//! (`subscribe_snapshot_reads_go_no_deeper_than_the_up_to_date_catch_up`):
//! in one process, it serves the up-to-date catch-up (no read, no build) and
//! then every subscribe case that reads a snapshot without a fan-out pass
//! (the first subscribe, the in-window and out-of-window reconnects, each in
//! both catch-up modes, and the session recovery's first subscribe), and
//! fails a read case more than 32 KiB deeper on its worker than the
//! up-to-date one. The budget alone cannot see a growth of that size: it
//! pins that the snapshot read's frames stay on the worker at about the
//! depth of a catch-up that reads nothing, which a fan-out pass future built
//! inline in `FanoutThreads::run_pass` instead of boxed as it is built
//! (`storage::fanout_thread::build_pass_on_heap`) broke by 91 to 98 KiB.
//! Both sides come from one run of one build, so the check holds on any
//! architecture and toolchain. Each of its cases checks its route and the
//! budget as the per-case tests do and prints its line under a `relative-`
//! name, then one `stack-guard relative-… against the up-to-date catch-up`
//! line with its difference.
//!
//! # Coverage
//!
//! | RPC | Cases |
//! |---|---|
//! | `CreateLayer` | new; overwrite (and its cleanup); new at `MAX_SECTION_TREE_DEPTH` |
//! | `UpdateLayer` | fast path; header-only; cached structural cold and warm; reset; WATCH (`txn`); layer recovery from the commit (`update-layer-layer-recovery`: object gone, mapping kept); layer recovery from the mapping resolve (`update-layer-mapping-recovery`: mapping and object gone, ends `not_recoverable`); cached cold, cached warm (the plan in place's incremental depth check) and WATCH at `MAX_SECTION_TREE_DEPTH` |
//! | `SubscribeReadLayer` | `brand_new`, `in_window` and `out_of_window` in each catch-up mode (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED` on and off); `brand_new` at `MAX_SECTION_TREE_DEPTH` in each mode; `brand_new` above the atomic-read limits (the fenced-read flight); `up_to_date`; the live relay; layer recovery; session recovery |
//!
//! The other `LiveService` RPCs (`Ping`, the session RPCs, `DeleteLayer`,
//! `StatLayer`, `JoinSession`) have no case.
//!
//! Configurations run: the `lua` and `txn` update backends (`lua-watch-
//! structural` commits structural deltas through the WATCH code the `txn`
//! cases measure), both catch-up modes, and checkpoint recovery enabled, on
//! a single-node Valkey with trace export off. A Valkey Cluster topology and
//! OTLP trace export are not run (SDD §15.1).
//!
//! # What is not measured
//!
//! SDD §15.1 lists what the guard does not cover: the RPCs and routes it
//! leaves out, work that runs independently of an RPC, and the
//! configurations it does not run. Two limits are the guard's own mechanics:
//!
//! - **The deepest frame's tail.** A frame allocated but not written to its
//!   bottom is under-reported by that tail; the budget's margin covers it.
//! - **Not exactly the production build.** `--features test-support`
//!   compiles the `cfg(any(test, feature = "test-support"))` test seams in
//!   `src/`, some of them on measured paths (e.g. the hook awaits on the
//!   `UpdateLayer` structural read in `storage::structural_cache_read`).
//!   They add state to the measured futures and never remove any, so the
//!   figures slightly over-count the shipped build's depth.
//!
//! # The budget
//!
//! One budget for every case, a quarter of the worker stack. The other three
//! quarters are the margin for frames that nest inside an RPC on the worker
//! but are not counted here: the real auth clients' (the cases use mocks),
//! codegen differences between this binary and the service binary and
//! between toolchains and architectures (CI runs x86_64), the probe's blind
//! tail, and the difference between the guard's current-thread scheduler
//! loop and production's multi-thread workers'. Work that runs independently
//! of an RPC — background tasks, fred's router tasks, blocking-pool closures
//! — adds nothing to an RPC's depth and is not covered by this budget at all
//! (Issue b.w7c). A breach
//! is fixed by moving the frame that grew off the stack — `Box::pin` the new
//! inline await, or heap-allocate the large local — never by raising the
//! budget, which moves only with the worker stack size. Growth past the
//! whole 2 MiB aborts the process with std's `thread 'stack-guard:<case>'
//! (<tid>) has overflowed its stack`, the thread name naming the case, as a
//! product worker would crash at that depth; under nextest only that case's
//! process dies.
//!
//! # Figures
//!
//! Measured locally on aarch64 (linux-gnu, rustc 1.94, `--cargo-profile
//! release`), KiB of the 512 KiB budget, all from one build (2026-10-10:
//! the window script, b.gvf, merged with the structural plan made in place,
//! main a02fd3c2), read from the lines of its runs. Two earlier
//! measurements, made the same way, are in the window script's record
//! (`docs-internal/validation/watch-window-script-b-gvf-20261010/`,
//! `report.md` § Release stack depth and `raw/stack-guard-main-f199e302.md`):
//! main at f199e302, before the window script and the plan in place; and
//! the window script with the fan-out pass's future built inline in
//! `FanoutThreads::run_pass`, before it was boxed as it is built
//! (`storage::fanout_thread::build_pass_on_heap`), when every route that
//! reads a snapshot, with a fan-out pass or without one, measured 91–154 KiB
//! deeper than the boxed build. CI's x86_64 figures are the lines the
//! `test-release-stack-guard` job stores in its JUnit XML. Runs of one build
//! differ by up to about 0.2 KiB. The
//! `fanout-read` column is that thread's depth over the same window and the
//! fan-out passes it ran there; the `watch-read` column the same for the
//! `watch-read` thread that ran the case's unfenced pass (in every case here
//! `watch-read-0`, the least-loaded thread's lowest index when all are idle).
//! Every pass thread that ran no pass reads 16.0 KiB, its depth without one.
//!
//! | Case | KiB | `fanout-read` KiB (passes) | `watch-read` KiB (passes) |
//! |---|---|---|---|
//! | `create-layer-new` | 54.2 | 16.0 (0) | 16.0 (0) |
//! | `create-layer-new-deep-tree` | 54.4 | 16.0 (0) | 16.0 (0) |
//! | `create-layer-overwrite` | 89.8 | 16.0 (0) | 33.5 (1) |
//! | `update-layer-fast` | 49.5 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-header-only` | 49.5 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-cached-cold` | 102.1 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-cached-cold-deep-tree` | 102.1 | 38.9 (1) | 16.0 (0) |
//! | `update-layer-cached-warm` | 82.4 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-cached-warm-deep-tree` | 82.4 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-reset` | 102.1 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-watch` | 79.4 | 16.0 (0) | 31.2 (1) |
//! | `update-layer-watch-deep-tree` | 79.4 | 16.0 (0) | 30.6 (1) |
//! | `update-layer-layer-recovery` | 171.2 | 16.0 (0) | 16.0 (0) |
//! | `update-layer-mapping-recovery` | 117.7 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-brand-new-snapshot` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-brand-new-legacy` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-brand-new-deep-tree-snapshot` | 114.5 | 37.8 (1) | 16.0 (0) |
//! | `subscribe-brand-new-deep-tree-legacy` | 114.5 | 37.8 (1) | 16.0 (0) |
//! | `subscribe-brand-new-above-atomic-read-limits` | 114.5 | 19.1 (1) | 16.0 (0) |
//! | `subscribe-in-window-snapshot` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-in-window-legacy` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-out-of-window-snapshot` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-out-of-window-legacy` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-up-to-date` | 96.5 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-live-relay` | 39.0 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-layer-recovery` | 196.0 | 16.0 (0) | 16.0 (0) |
//! | `subscribe-session-recovery` | 96.5 | 16.0 (0) | 16.0 (0) |
//!
//! The subscribe cases that read or build (brand-new, in- and out-of-window,
//! session recovery) measure 96.5 KiB, the depth of the `up_to_date`
//! catch-up (no read, no build); a regime's own work goes deeper on the deep
//! tree and above the atomic-read limits (114.5 KiB), and layer recovery is
//! the deepest case. The cached warm route on the deep tree goes no deeper
//! than on the flat base: its incremental depth check walks the chain in a
//! loop. The structural WATCH read's pass issues round 1 as plain commands
//! and each window of its rounds 2 and 3 as one window script `EVALSHA`
//! (b.gvf; no window of these cases is a lone children list over τ or
//! follows a script fallback, the other plain windows): its `watch-read`
//! thread reads about 31 KiB. The calibration frame measures 257.0 KiB for
//! its 256 KiB, and the usable stack below the entry frame is 2037.1 KiB.
//!
//! # Lifecycle
//!
//! Each case's measured server shuts down gracefully, under a bound, when
//! its window closes. Each case's `TestValkey` purges its keys when the case
//! ends, normally or by a panic. A stack-overflow abort or a nextest kill
//! skips the purge, leaving that case's keys under its unique prefix; in CI
//! the Valkey belongs to the job and goes with it.

#![cfg(all(target_os = "linux", target_env = "gnu"))]

#[path = "../common/mod.rs"]
mod common;

/// The integration suite's gRPC helpers; this binary uses a few of them, so
/// the rest (and their re-exports) are unused here.
#[path = "../integration/grpc/helpers.rs"]
#[allow(dead_code, unused_imports)]
mod helpers;

mod fixtures;
mod harness;
mod probe;

use std::future::Future;
use std::time::Duration;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::limits::MAX_SECTION_TREE_DEPTH;
use live_edit::observability::metrics::{
    UPDATE_LAYER_PATH_FAST, UPDATE_LAYER_PATH_RESET, UPDATE_LAYER_PATH_STRUCTURAL,
    UPDATE_LAYER_PATH_STRUCTURAL_FULL, UPDATE_LAYER_PATH_WATCH,
};
use live_edit::storage::fanout_thread::PassLane;

use fixtures::{
    base_content, cache_lookups, cached_depth_ok, catch_ups, chain, commits_on, create_layer,
    drain_catch_up, faithful_builds, field_update, flights_led, layer_checkpoints,
    low_ceiling_client, read_relay, recoveries, reset_update, restore_lineages, restores,
    session_checkpoints, structural_update, subscribe, update_layer, update_uploads_admitted,
    CatchUp, Client, Fx, Opts, BASE_SECTIONS, FIELD_SECTION,
};
use harness::{Measurement, Probe, ThreadMeasure, Work};
use probe::{
    check_worker_stack, StackBounds, CALIBRATION_FRAME, CALIBRATION_SLACK, KIB, MIB,
    PRODUCT_WORKER_STACK, STACK_BUDGET,
};

// The fan-out threads' explicit stack is the workers' stack the budget is a
// quarter of; the thread-bounds check then confirms the threads got it.
const _: () =
    assert!(PRODUCT_WORKER_STACK == live_edit::storage::fanout_thread::FANOUT_THREAD_STACK_BYTES);

/// The service binary's allocator on linux-gnu, so the measured paths
/// allocate through the same code.
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The one deadline over a whole case, inside nextest's 120 s hang-killer,
/// so a stuck case fails with a message rather than being killed silently,
/// however many steps it takes. Each wait inside it is bounded by
/// [`harness::SETTLE`] (20 s) and fails with its own message first.
const TEST_BUDGET: Duration = Duration::from_secs(100);

/// Install the production log subscriber once per process: JSON at `info`,
/// written to a sink, so event formatting runs on the measured stack as it
/// does on a worker (`observability::init_tracing`, minus the OTLP exporter
/// the default deployment does not enable).
fn install_production_log_subscriber() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let subscriber = live_edit::observability::compose_registry_for_tests(
            "info",
            live_edit::config::LogFormat::Json,
            live_edit::config::DEFAULT_TRACE_FILTER,
            std::io::sink,
            None,
        );
        tracing::subscriber::set_global_default(subscriber)
            .expect("no other global subscriber in this binary");
    });
}

/// Run one test's body under [`TEST_BUDGET`], boxed so its state is on the
/// heap rather than in the test thread's frame, and hand back its output.
async fn run<T>(body: impl Future<Output = T> + Send) -> T {
    install_production_log_subscriber();
    tokio::time::timeout(TEST_BUDGET, Box::pin(body))
        .await
        .unwrap_or_else(|_| panic!("the case did not finish within {TEST_BUDGET:?}"))
}

/// Start B for `case` over `svc`, and a warmed-up client of it.
async fn serve(
    case: &'static str,
    svc: &live_edit::grpc::service::LiveEditService,
) -> (Measurement, Client) {
    let measurement = Measurement::start(case, Work::Serve(Box::new(svc.clone()))).await;
    let client = measurement.client().await;
    (measurement, client)
}

fn kib(bytes: usize) -> f64 {
    bytes as f64 / KIB as f64
}

/// Print the case's one line — on a pass and on a failure — then fail on a
/// missed route, a measured thread that is not worker-sized, or a breach.
/// The served client's pass threads — `fanout-read` and each `watch-read`
/// thread — are measured too, each against the same budget: the fan-out
/// passes an RPC starts run there (`storage::fanout_thread`), and they and
/// `snapshot-io` must be worker-sized.
fn verdict(case: &str, probe: &Probe, route: Result<(), String>) {
    let depth = probe.depth();
    let pass_threads = probe
        .fanout
        .as_ref()
        .map(|fanout| fanout.threads.as_slice())
        .unwrap_or_default();
    let bounds = check_worker_stack(&format!("the measured thread of {case}"), &probe.thread)
        .and_then(|()| {
            check_worker_stack(
                &format!("the measured runtime's blocking pool of {case}"),
                &probe.blocking,
            )
        })
        .and_then(|()| {
            pass_threads.iter().try_for_each(|t| {
                check_worker_stack(&format!("the {} thread of {case}", t.role), &t.thread)
            })
        })
        .and_then(|()| match &probe.fanout {
            Some(fanout) => check_worker_stack(
                &format!("the snapshot-io thread of {case}"),
                &fanout.snapshot_io,
            ),
            None => Ok(()),
        });
    let within = depth <= STACK_BUDGET;
    let breached: Vec<&ThreadMeasure> = pass_threads
        .iter()
        .filter(|t| t.depth() > STACK_BUDGET)
        .collect();
    println!(
        "stack-guard {case}: {:.1} KiB used, budget {} KiB, usable stack {:.1} KiB, headroom \
         {:.1} KiB, {}, route {}, {}",
        kib(depth),
        STACK_BUDGET / KIB,
        kib(probe.usable()),
        kib(STACK_BUDGET) - kib(depth),
        if pass_threads.is_empty() {
            "pass threads not measured".to_owned()
        } else {
            pass_threads
                .iter()
                .map(|t| {
                    format!(
                        "{} {:.1} KiB over {} pass(es)",
                        t.role,
                        kib(t.depth()),
                        t.passes
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        },
        if route.is_ok() { "taken" } else { "MISSED" },
        if within && breached.is_empty() && route.is_ok() && bounds.is_ok() {
            "PASS"
        } else {
            "FAIL"
        },
    );
    if let Err(e) = route {
        panic!("route check: {case}: {e}");
    }
    if let Err(e) = bounds {
        panic!("thread-bounds self-check: {e}");
    }
    assert!(
        within,
        "release-build stack use of {:.1} KiB serving {case}, above the {} KiB budget (a \
         quarter of the {} KiB tokio worker stack). Move the frame that grew off the stack — \
         `Box::pin` the new inline await, or heap-allocate the large local — rather than \
         raising the budget, which changes only with the worker stack size (SDD §15.1).",
        kib(depth),
        STACK_BUDGET / KIB,
        PRODUCT_WORKER_STACK / KIB,
    );
    if let Some(t) = breached.first() {
        panic!(
            "release-build stack use of {:.1} KiB on the {} thread serving {case}, above the {} \
             KiB budget (a quarter of the thread's {} KiB stack): a fan-out pass grew. Move the \
             frame that grew off the stack rather than raising the budget (SDD §15.1).",
            kib(t.depth()),
            t.role,
            STACK_BUDGET / KIB,
            PRODUCT_WORKER_STACK / KIB,
        );
    }
}

/// `Ok` when at least one fan-out pass of `lane` ran inside the window on
/// the served client's thread for it: `fanout-read` for a fenced pass (a
/// fenced read's: a cold start's, a flight's, a deep-tree first
/// subscribe's), a `watch-read` thread for an unfenced one (the structural
/// WATCH read's, the overwrite's cleanup's ext-GC reference walk's).
fn ran_a_pass_on(probe: &Probe, lane: PassLane) -> Result<(), String> {
    let Some(fanout) = &probe.fanout else {
        return Err("the pass threads were not measured".to_owned());
    };
    let (passes, threads) = match lane {
        PassLane::Fenced => (fanout.fenced_passes(), "the fanout-read thread"),
        PassLane::Unfenced => (fanout.unfenced_passes(), "a watch-read thread"),
    };
    if passes > 0 {
        Ok(())
    } else {
        Err(format!("no {lane:?} fan-out pass ran on {threads}"))
    }
}

/// `Ok` when `got == want`, else what differed.
fn expect_eq<T: PartialEq + std::fmt::Debug>(what: &str, got: T, want: T) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{what}: expected {want:?}, got {got:?}"))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Probe self-tests
// ═════════════════════════════════════════════════════════════════════════════

/// A spawned task running a sync frame of exactly [`CALIBRATION_FRAME`]
/// bytes measures at least that, and at most [`CALIBRATION_SLACK`] more (the
/// runtime's own frames). A probe that reports about zero — a paint range
/// the window never reaches, a scan that stops early — fails the lower
/// bound, so it cannot pass every case vacuously; one that counts frames
/// outside the window fails the upper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn probe_measures_a_256_kib_calibration_frame_within_its_bounds() {
    run(async {
        let mut measurement = Measurement::start("probe-calibration", Work::Calibrate).await;
        measurement.open().await;
        let probe = measurement.finish().await;
        let depth = probe.depth();
        println!(
            "stack-guard probe-calibration: {:.1} KiB measured for a {} KiB frame (accepted: \
             {}..={} KiB), usable stack {:.1} KiB",
            kib(depth),
            CALIBRATION_FRAME / KIB,
            CALIBRATION_FRAME / KIB,
            (CALIBRATION_FRAME + CALIBRATION_SLACK) / KIB,
            kib(probe.usable()),
        );
        assert!(
            (CALIBRATION_FRAME..=CALIBRATION_FRAME + CALIBRATION_SLACK).contains(&depth),
            "probe self-check: a {} KiB frame measured {:.1} KiB; the probe does not measure \
             what the window uses",
            CALIBRATION_FRAME / KIB,
            kib(depth),
        );
    })
    .await;
}

/// The thread-bounds check rejects a stack that is not the product
/// worker's — an explicit 8 MiB, and, when the environment sets
/// `RUST_MIN_STACK` as `.cargo/config.toml` does for every cargo-run test, a
/// thread that sets no size and inherits it — and the measured thread and
/// its runtime's blocking-pool thread pass it. Those two are measured right
/// after the 8 MiB threads exit, so glibc's stack cache holds their stacks
/// and may hand one to a 2 MiB request: the harness's re-spawns must still
/// end on worker-sized stacks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn thread_bounds_check_rejects_other_stacks_and_passes_the_measured_threads() {
    run(async {
        let large = std::thread::Builder::new()
            .stack_size(8 * MIB)
            .spawn(StackBounds::of_current_thread)
            .expect("spawn")
            .join()
            .expect("join");
        assert!(
            check_worker_stack("an explicit 8 MiB thread", &large).is_err(),
            "the check must reject an 8 MiB stack; it reported {} KiB and passed",
            large.size / KIB,
        );

        let inherited = std::thread::spawn(StackBounds::of_current_thread)
            .join()
            .expect("join");
        let rust_min_stack = std::env::var("RUST_MIN_STACK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        if let Some(min) = rust_min_stack.filter(|&m| m > 2 * PRODUCT_WORKER_STACK) {
            assert!(
                inherited.size >= min,
                "a default-size thread must inherit RUST_MIN_STACK={min}; it has {} bytes",
                inherited.size,
            );
            assert!(
                check_worker_stack("a thread inheriting RUST_MIN_STACK", &inherited).is_err(),
                "the check must reject a thread sized by the inherited RUST_MIN_STACK={min}",
            );
        }

        let mut measurement = Measurement::start("thread-bounds", Work::Calibrate).await;
        let (thread_spawns, blocking_spawns) =
            (measurement.thread_spawns, measurement.blocking_spawns);
        measurement.open().await;
        let probe = measurement.finish().await;
        println!(
            "stack-guard thread-bounds: explicit 8 MiB thread {} KiB, default-size thread {} KiB \
             (RUST_MIN_STACK={rust_min_stack:?}), measured thread {} KiB after {thread_spawns} \
             spawn(s), its blocking-pool thread {} KiB after {blocking_spawns} spawn(s)",
            large.size / KIB,
            inherited.size / KIB,
            probe.thread.size / KIB,
            probe.blocking.size / KIB,
        );
        check_worker_stack("the measured thread", &probe.thread)
            .expect("the measured thread must have the product worker's stack");
        check_worker_stack(
            "the measured runtime's blocking-pool thread",
            &probe.blocking,
        )
        .expect("the measured runtime's blocking-pool thread must have the product worker's stack");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// A new layer: the upload, the parse and the create commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn create_layer_new_stays_within_the_stack_budget() {
    run(create_new("create-layer-new", base_content())).await;
}

/// A new layer whose section tree is [`MAX_SECTION_TREE_DEPTH`] deep: the
/// write path's depth check at the cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn create_layer_new_deep_tree_stays_within_the_stack_budget() {
    run(create_new(
        "create-layer-new-deep-tree",
        chain(MAX_SECTION_TREE_DEPTH),
    ))
    .await;
}

async fn create_new(case: &'static str, content: oodf_ffi::OwnedDelta) {
    let fx = Fx::new().await;
    let session_id = fx.session().await;
    let svc = fx.service(Opts::default()).await;
    let (mut m, mut client) = serve(case, &svc).await;
    m.open().await;
    let created = create_layer(&mut client, &session_id, &content).await;
    let probe = m.finish().await;
    let route = match created {
        Ok(r) if r.object_id > 0 => Ok(()),
        Ok(r) => Err(format!("CreateLayer answered object_id {}", r.object_id)),
        Err(s) => Err(format!("CreateLayer failed: {s:?}")),
    };
    verdict(case, &probe, route);
}

/// A second `CreateLayer` of the same name: the overwrite, and the old
/// object's cleanup task, which the window waits for. The cleanup's release
/// reads the old object by the paged reference walk (`storage::retirement`),
/// one unfenced pass on a `watch-read` thread, then tears it down in pages;
/// the case requires the pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn create_layer_overwrite_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "create-layer-overwrite";
        let fx = Fx::new().await;
        let (session_id, old) = fx.session_with_layer(&base_content()).await;
        let svc = fx.service(Opts::default()).await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        m.open().await;
        let created = create_layer(&mut client, &session_id, &base_content()).await;
        // The overwrite's cleanup of the old object runs on the measured
        // runtime; keep the window open until it is done.
        svc.layer_store.wait_for_cleanup(harness::SETTLE).await;
        let probe = m.finish().await;
        let route = match created {
            Ok(r) if r.object_id != old => {
                if fx.header_exists(old).await {
                    Err(format!("the old object {old} was not cleaned up"))
                } else {
                    ran_a_pass_on(&probe, PassLane::Unfenced)
                }
            }
            Ok(r) => Err(format!("the overwrite kept object_id {}", r.object_id)),
            Err(s) => Err(format!("CreateLayer failed: {s:?}")),
        };
        verdict(CASE, &probe, route);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// One `UpdateLayer` on a fresh service, checked to commit on `path` and,
/// with a `pass` lane, to have run a fan-out pass of that lane on its
/// thread: the WATCH path's unfenced read on a `watch-read` thread, or a
/// fenced cold start's on `fanout-read`. A warmed case (`warm`) is also
/// checked to have taken the cached warm route: its structural cache lookup
/// answered `hit` once in the window — a result only that route produces —
/// and the entry it planned on held plan indexes with the depth flag set
/// before the window and after it, the flag under which the plan in place
/// checks its depth incrementally from the ids it listed instead of walking
/// the whole tree ([`cached_depth_ok`]).
async fn update_on(
    case: &'static str,
    opts: Opts,
    base: oodf_ffi::OwnedDelta,
    warm: Option<oodf_ffi::OwnedDelta>,
    update: Option<oodf_ffi::OwnedDelta>,
    path: &str,
    pass: Option<PassLane>,
) {
    let fx = Fx::new().await;
    let (session_id, object_id) = fx.session_with_layer(&base).await;
    let svc = fx.service(opts).await;
    let warmed = warm.is_some();
    if let Some(warm) = warm {
        // Unmeasured: one structural commit through the measured store loads
        // the object into its structural cache.
        Fx::commit_through(&svc.layer_store, &session_id, object_id, warm).await;
    }
    // Read outside the window: a checkout that puts the entry back unchanged.
    let depth_ok_before = warmed.then(|| cached_depth_ok(&svc, object_id));
    let (mut m, mut client) = serve(case, &svc).await;
    let before = commits_on(&svc, path);
    let hits_before = cache_lookups(&svc, "hit");
    m.open().await;
    let updated = update_layer(&mut client, &session_id, object_id, update.as_ref()).await;
    let probe = m.finish().await;
    let route = match updated {
        Ok(_) => expect_eq(
            &format!("UpdateLayer commits on path {path}"),
            commits_on(&svc, path) - before,
            1.0,
        )
        .and(match pass {
            Some(lane) => ran_a_pass_on(&probe, lane),
            None => Ok(()),
        })
        .and(match depth_ok_before {
            Some(depth_ok) => expect_eq(
                "the cached warm route's lookups answered hit",
                cache_lookups(&svc, "hit") - hits_before,
                1.0,
            )
            .and(
                if depth_ok == Some(true) && cached_depth_ok(&svc, object_id) == Some(true) {
                    Ok(())
                } else {
                    Err(format!(
                    "the entry's depth flag must be set before and after the plan, so the plan \
                     checks its depth incrementally (before: {depth_ok:?}, after: {:?})",
                    cached_depth_ok(&svc, object_id)
                ))
                },
            ),
            None => Ok(()),
        }),
        Err(s) => Err(format!("UpdateLayer failed: {s:?}")),
    };
    verdict(case, &probe, route);
}

/// A field-only update: the Lua fast path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_fast_path_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-fast",
        Opts::default(),
        base_content(),
        None,
        Some(field_update(1)),
        UPDATE_LAYER_PATH_FAST,
        None,
    ))
    .await;
}

/// A header-only upload: an empty commit, on the fast path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_header_only_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-header-only",
        Opts::default(),
        base_content(),
        None,
        None,
        UPDATE_LAYER_PATH_FAST,
        None,
    ))
    .await;
}

/// A structural update on a cold structural cache: the cached route's cold
/// start, which reads the object into the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_cached_cold_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-cached-cold",
        Opts::default(),
        base_content(),
        None,
        Some(structural_update(FIELD_SECTION, BASE_SECTIONS + 1)),
        UPDATE_LAYER_PATH_STRUCTURAL_FULL,
        None,
    ))
    .await;
}

/// The cached cold start on a base whose section tree is one short of
/// [`MAX_SECTION_TREE_DEPTH`], the update adding the section at the cap. The
/// chain is above the cold start's route predictor, so the cold start reads
/// it by the fenced read and this case measures that read's pass on
/// `fanout-read`; the case requires the pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_cached_cold_deep_tree_stays_within_the_stack_budget() {
    let deepest = MAX_SECTION_TREE_DEPTH as u64;
    run(update_on(
        "update-layer-cached-cold-deep-tree",
        Opts::default(),
        chain(MAX_SECTION_TREE_DEPTH - 1),
        None,
        Some(structural_update(deepest - 1, deepest)),
        UPDATE_LAYER_PATH_STRUCTURAL_FULL,
        Some(PassLane::Fenced),
    ))
    .await;
}

/// A structural update with the object already in the structural cache: the
/// cached route's plan in place on the held entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_cached_warm_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-cached-warm",
        Opts::default(),
        base_content(),
        Some(structural_update(0, BASE_SECTIONS + 100)),
        Some(structural_update(FIELD_SECTION, BASE_SECTIONS + 1)),
        UPDATE_LAYER_PATH_STRUCTURAL,
        None,
    ))
    .await;
}

/// The cached warm route on a base whose section tree is one short of
/// [`MAX_SECTION_TREE_DEPTH`], the update adding the section at the cap: the
/// plan made in place on the cached entry, whose incremental depth check
/// walks up the whole chain (iteratively) from the new section's listing.
/// The warm-up (a section under the root) loads the entry outside the
/// window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_cached_warm_deep_tree_stays_within_the_stack_budget() {
    let deepest = MAX_SECTION_TREE_DEPTH as u64;
    run(update_on(
        "update-layer-cached-warm-deep-tree",
        Opts::default(),
        chain(MAX_SECTION_TREE_DEPTH - 1),
        Some(structural_update(0, deepest + 100)),
        Some(structural_update(deepest - 1, deepest)),
        UPDATE_LAYER_PATH_STRUCTURAL,
        None,
    ))
    .await;
}

/// A reset (`is_diff`, `base_version` 0) replacing the layer's content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_reset_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-reset",
        Opts::default(),
        base_content(),
        None,
        Some(reset_update()),
        UPDATE_LAYER_PATH_RESET,
        None,
    ))
    .await;
}

/// The WATCH/MULTI/EXEC commit, under the `txn` backend. The same commit
/// code takes structural deltas under `lua-watch-structural` and the `lua`
/// backend's fallbacks. The WATCH path reads the whole layer by the unfenced
/// fan-out, whose pass runs on a `watch-read` thread; the case requires the
/// pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_watch_stays_within_the_stack_budget() {
    run(update_on(
        "update-layer-watch",
        Opts {
            backend: UpdateLayerBackend::Txn,
            ..Opts::default()
        },
        base_content(),
        None,
        Some(structural_update(FIELD_SECTION, BASE_SECTIONS + 1)),
        UPDATE_LAYER_PATH_WATCH,
        Some(PassLane::Unfenced),
    ))
    .await;
}

/// The WATCH commit on a base one section short of
/// [`MAX_SECTION_TREE_DEPTH`], the update adding the section at the cap; its
/// unfenced read's pass on a `watch-read` thread is required.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_watch_deep_tree_stays_within_the_stack_budget() {
    let deepest = MAX_SECTION_TREE_DEPTH as u64;
    run(update_on(
        "update-layer-watch-deep-tree",
        Opts {
            backend: UpdateLayerBackend::Txn,
            ..Opts::default()
        },
        chain(MAX_SECTION_TREE_DEPTH - 1),
        None,
        Some(structural_update(deepest - 1, deepest)),
        UPDATE_LAYER_PATH_WATCH,
        Some(PassLane::Unfenced),
    ))
    .await;
}

/// A commit whose object is gone from the store while its layer mapping
/// survives, with a layer checkpoint to restore: the handler resolves the
/// mapping, admits the upload, and its commit-path recovery arm restores the
/// layer and re-issues the commit. The admitted upload is what places the
/// restore after the mapping resolve (see
/// `update_layer_mapping_recovery_stays_within_the_stack_budget`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_layer_recovery_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "update-layer-layer-recovery";
        let fx = Fx::new().await;
        let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
        let checkpoints = layer_checkpoints(&session_id, object_id, 3).await;
        fx.wipe_object(object_id).await;
        let svc = fx
            .service(Opts {
                layer_checkpoints: Some(checkpoints),
                ..Opts::default()
            })
            .await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        let (restores_before, admitted_before) =
            (restores(&svc, "layer"), update_uploads_admitted(&svc));
        m.open().await;
        let updated = update_layer(&mut client, &session_id, object_id, None).await;
        let probe = m.finish().await;
        let route = match updated {
            Ok(r) if r.sequence_num > 0 => expect_eq(
                "layer restores",
                restores(&svc, "layer") - restores_before,
                1.0,
            )
            .and(expect_eq(
                "uploads admitted",
                update_uploads_admitted(&svc) - admitted_before,
                1.0,
            )),
            Ok(r) => Err(format!(
                "the re-issued commit answered sequence {}",
                r.sequence_num
            )),
            Err(s) => Err(format!("UpdateLayer failed: {s:?}")),
        };
        verdict(CASE, &probe, route);
    })
    .await;
}

/// A commit whose layer mapping and object are both gone from the store,
/// with a layer checkpoint seeded: the handler's recovery arm at the
/// mapping resolve, before admission, attempts the restore. It cannot
/// restore here — the restore writes object state, never the mapping, and
/// its identity guard refuses a blob whose layer mapping no longer resolves
/// — so the attempt ends `not_recoverable` once the checkpoint is selected
/// and handed to the apply, whose guard refuses it before any write, and
/// the RPC answers that arm's `NOT_FOUND`.
/// A restore through this arm lands only if the mapping is back before the
/// apply's identity guard reads it (another replica's session recovery,
/// say): a race. Back any later, the attempt still ends `not_recoverable`.
/// A landed restore's hook chain runs at the same handler depth as
/// `update-layer-layer-recovery`'s, so its frames are measured there. That
/// argument rests on both `RecoveryHook::try_restore_layer` awaits — this
/// arm's at the mapping resolve and the commit-path arm's — being direct
/// awaits in `grpc::layer::update_layer_inner`; a refactor that moves either
/// into a nested future breaks it, and must add a case for the moved one.
///
/// The route check tells this arm from the commit-path one by the upload
/// gate: this arm returns before the upload is admitted, the commit-path arm
/// runs after (`update_layer_layer_recovery_stays_within_the_stack_budget`
/// checks its admitted upload). It tells a refused apply from a selection
/// that found no checkpoint — which also ends `not_recoverable`, without
/// reaching the apply — by the base-only lineage the selection gathered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn update_layer_mapping_recovery_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "update-layer-mapping-recovery";
        let fx = Fx::new().await;
        let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
        let checkpoints = layer_checkpoints(&session_id, object_id, 3).await;
        fx.wipe_object(object_id).await;
        fx.wipe_mapping(&session_id).await;
        let svc = fx
            .service(Opts {
                layer_checkpoints: Some(checkpoints),
                ..Opts::default()
            })
            .await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        let (refused_before, restores_before, admitted_before, gathered_before) = (
            recoveries(&svc, "not_recoverable", "layer"),
            restores(&svc, "layer"),
            update_uploads_admitted(&svc),
            restore_lineages(&svc, "gathered_base"),
        );
        m.open().await;
        let updated = update_layer(&mut client, &session_id, object_id, None).await;
        let probe = m.finish().await;
        let route = match updated {
            Err(s) if s.code() == tonic::Code::NotFound => expect_eq(
                "layer recoveries ending not_recoverable",
                recoveries(&svc, "not_recoverable", "layer") - refused_before,
                1.0,
            )
            .and(expect_eq(
                "layer restores",
                restores(&svc, "layer") - restores_before,
                0.0,
            ))
            .and(expect_eq(
                "base-only lineages the restore's selection gathered (the seeded checkpoint)",
                restore_lineages(&svc, "gathered_base") - gathered_before,
                1.0,
            ))
            .and(expect_eq(
                "uploads admitted (the commit-path arm runs after admission)",
                update_uploads_admitted(&svc) - admitted_before,
                0.0,
            ))
            .and(expect_eq(
                "the status message",
                s.message(),
                format!("layer '{}' not found in session", fixtures::LAYER).as_str(),
            )),
            Err(s) => Err(format!("UpdateLayer failed with {s:?}, not NOT_FOUND")),
            Ok(r) => Err(format!(
                "UpdateLayer committed sequence {} with its mapping gone",
                r.sequence_num
            )),
        };
        verdict(CASE, &probe, route);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// SubscribeReadLayer
// ═════════════════════════════════════════════════════════════════════════════

/// The catch-up `mode` label and faithful-build count a catch-up in `regime`
/// records under `snapshot`: the faithful builds are snapshot mode's alone,
/// and only a from-empty full state is labelled `snapshot`.
fn mode_of(regime: &str, snapshot: bool) -> (&'static str, f64) {
    let label = if snapshot && regime == "brand_new" {
        "snapshot"
    } else {
        "legacy"
    };
    let builds = if snapshot && regime != "up_to_date" {
        1.0
    } else {
        0.0
    };
    (label, builds)
}

/// What a catch-up case checks its catch-up did: served in `regime`, with
/// `contents` content deliveries, and, when `runs_a_pass`, with a fenced
/// fan-out pass on the `fanout-read` thread.
struct Served {
    regime: &'static str,
    contents: usize,
    runs_a_pass: bool,
}

/// One catch-up, measured from the subscribe to its caught-up ack, on a
/// fresh service; checked to be served as `served` says by the `snapshot`
/// mode's code. Returns the measured thread's depth, for
/// [`subscribe_snapshot_reads_go_no_deeper_than_the_up_to_date_catch_up`].
async fn catch_up_case(
    case: &'static str,
    fx: Fx,
    opts: Opts,
    session_id: String,
    have: (u64, u64),
    served: Served,
) -> usize {
    let Served {
        regime,
        contents,
        runs_a_pass,
    } = served;
    let snapshot = opts.snapshot;
    let svc = fx.service(opts).await;
    let (label, builds) = mode_of(regime, snapshot);
    let (mut m, mut client) = serve(case, &svc).await;
    let (regimes_before, builds_before) = (catch_ups(&svc, regime, label), faithful_builds(&svc));
    m.open().await;
    let mut stream = subscribe(&mut client, &session_id, have.0, have.1).await;
    let caught_up = drain_catch_up(&mut stream).await;
    let probe = m.finish().await;
    let route = expect_eq(
        &format!("{regime} catch-ups labelled {label}"),
        catch_ups(&svc, regime, label) - regimes_before,
        1.0,
    )
    .and(expect_eq(
        "faithful (snapshot-mode) builds",
        faithful_builds(&svc) - builds_before,
        builds,
    ))
    .and(expect_eq(
        "content deliveries",
        caught_up.contents,
        contents,
    ))
    .and(if runs_a_pass {
        ran_a_pass_on(&probe, PassLane::Fenced)
    } else {
        Ok(())
    });
    drop(stream);
    verdict(case, &probe, route);
    probe.depth()
}

/// A first subscribe to a layer of `content`, checked, when `runs_a_pass`,
/// to have read it by a fenced fan-out pass on `fanout-read`. Returns the
/// measured thread's depth.
async fn brand_new(
    case: &'static str,
    snapshot: bool,
    content: oodf_ffi::OwnedDelta,
    runs_a_pass: bool,
) -> usize {
    let fx = Fx::new().await;
    let (session_id, _) = fx.session_with_layer(&content).await;
    let opts = Opts {
        snapshot,
        ..Opts::default()
    };
    let served = Served {
        regime: "brand_new",
        contents: 1,
        runs_a_pass,
    };
    catch_up_case(case, fx, opts, session_id, (0, 0), served).await
}

/// A first subscribe: the full state at a proven cut, snapshot mode's
/// faithful build.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_brand_new_snapshot_mode_stays_within_the_stack_budget() {
    run(brand_new(
        "subscribe-brand-new-snapshot",
        true,
        base_content(),
        false,
    ))
    .await;
}

/// A first subscribe under the rollback mode
/// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`): the legacy full-content diff.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_brand_new_legacy_mode_stays_within_the_stack_budget() {
    run(brand_new(
        "subscribe-brand-new-legacy",
        false,
        base_content(),
        false,
    ))
    .await;
}

/// A first subscribe to a layer whose section tree is
/// [`MAX_SECTION_TREE_DEPTH`] deep, snapshot mode. The catch-up reads the
/// deep chain by a fan-out pass on `fanout-read`, so this case measures that
/// pass; the case requires it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_brand_new_deep_tree_snapshot_mode_stays_within_the_stack_budget() {
    run(brand_new(
        "subscribe-brand-new-deep-tree-snapshot",
        true,
        chain(MAX_SECTION_TREE_DEPTH),
        true,
    ))
    .await;
}

/// The deep-tree first subscribe under the rollback mode; its read's pass on
/// `fanout-read` is required.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_brand_new_deep_tree_legacy_mode_stays_within_the_stack_budget() {
    run(brand_new(
        "subscribe-brand-new-deep-tree-legacy",
        false,
        chain(MAX_SECTION_TREE_DEPTH),
        true,
    ))
    .await;
}

/// A first subscribe to a layer above the client's atomic-read limits: the
/// catch-up creates its object's shared fenced-read flight.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_brand_new_above_atomic_read_limits_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "subscribe-brand-new-above-atomic-read-limits";
        let fx = Fx::new().await;
        let (session_id, _) = fx.session_with_layer(&base_content()).await;
        let svc = fx
            .service(Opts {
                client: Some(low_ceiling_client().await),
                ..Opts::default()
            })
            .await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        let (regimes_before, flights_before) =
            (catch_ups(&svc, "brand_new", "snapshot"), flights_led(&svc));
        m.open().await;
        let mut stream = subscribe(&mut client, &session_id, 0, 0).await;
        let caught_up = drain_catch_up(&mut stream).await;
        let probe = m.finish().await;
        let route = expect_eq(
            "brand_new catch-ups",
            catch_ups(&svc, "brand_new", "snapshot") - regimes_before,
            1.0,
        )
        .and(expect_eq(
            "fenced-read flights created",
            flights_led(&svc) - flights_before,
            1.0,
        ))
        .and(expect_eq("content deliveries", caught_up.contents, 1))
        .and(ran_a_pass_on(&probe, PassLane::Fenced));
        drop(stream);
        verdict(CASE, &probe, route);
    })
    .await;
}

/// A reconnect whose gap is still on the layer stream: three missed
/// structural commits. Snapshot mode delivers one coalesced diff; the
/// rollback mode replays each commit. Returns the measured thread's depth.
async fn in_window(case: &'static str, snapshot: bool) -> usize {
    let fx = Fx::new().await;
    let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
    for i in 1..=3 {
        fx.update(
            &session_id,
            object_id,
            &structural_update(FIELD_SECTION, BASE_SECTIONS + i),
        )
        .await;
    }
    let contents = if snapshot { 1 } else { 3 };
    let opts = Opts {
        snapshot,
        ..Opts::default()
    };
    catch_up_case(
        case,
        fx,
        opts,
        session_id,
        (object_id, 1),
        Served {
            regime: "in_window",
            contents,
            runs_a_pass: false,
        },
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_in_window_snapshot_mode_stays_within_the_stack_budget() {
    run(in_window("subscribe-in-window-snapshot", true)).await;
}

/// The rollback mode's per-commit replay: the largest catch-up future.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_in_window_legacy_mode_stays_within_the_stack_budget() {
    run(in_window("subscribe-in-window-legacy", false)).await;
}

/// A reconnect whose gap was trimmed off the layer stream: one coalesced
/// diff from the proven read. Returns the measured thread's depth.
async fn out_of_window(case: &'static str, snapshot: bool) -> usize {
    let fx = Fx::new().await;
    let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
    let mut seq = 1;
    for i in 1..=4 {
        seq = fx
            .update(
                &session_id,
                object_id,
                &structural_update(FIELD_SECTION, BASE_SECTIONS + i),
            )
            .await;
    }
    // The subscriber holds 2; the entry after it, 3, is trimmed away.
    fx.trim_stream_below(object_id, seq).await;
    let opts = Opts {
        snapshot,
        ..Opts::default()
    };
    catch_up_case(
        case,
        fx,
        opts,
        session_id,
        (object_id, 2),
        Served {
            regime: "out_of_window",
            contents: 1,
            runs_a_pass: false,
        },
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_out_of_window_snapshot_mode_stays_within_the_stack_budget() {
    run(out_of_window("subscribe-out-of-window-snapshot", true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_out_of_window_legacy_mode_stays_within_the_stack_budget() {
    run(out_of_window("subscribe-out-of-window-legacy", false)).await;
}

/// A reconnect already at the head: no read, no build, the ack only. The
/// same code in both modes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_up_to_date_stays_within_the_stack_budget() {
    run(up_to_date("subscribe-up-to-date")).await;
}

/// The up-to-date reconnect of
/// [`subscribe_up_to_date_stays_within_the_stack_budget`], measured as
/// `case`. Returns the measured thread's depth.
async fn up_to_date(case: &'static str) -> usize {
    let fx = Fx::new().await;
    let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
    let seq = fx.update(&session_id, object_id, &field_update(1)).await;
    catch_up_case(
        case,
        fx,
        Opts::default(),
        session_id,
        (object_id, seq),
        Served {
            regime: "up_to_date",
            contents: 0,
            runs_a_pass: false,
        },
    )
    .await
}

/// The steady state: a commit another writer lands, relayed to a caught-up
/// subscriber. The subscription and its catch-up are set up before the
/// window; the window holds the relay alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_live_relay_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "subscribe-live-relay";
        let fx = Fx::new().await;
        let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
        let svc = fx.service(Opts::default()).await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        let mut stream = subscribe(&mut client, &session_id, 0, 0).await;
        let caught_up: CatchUp = drain_catch_up(&mut stream).await;
        m.open().await;
        // Written on the test's runtime, as another replica's writer: only
        // the relay runs on the measured thread.
        let seq =
            Fx::commit_through(&svc.layer_store, &session_id, object_id, field_update(2)).await;
        read_relay(&mut stream, seq).await;
        let probe = m.finish().await;
        let route = expect_eq("the relayed commit", seq, caught_up.ack_seq + 1);
        drop(stream);
        verdict(CASE, &probe, route);
    })
    .await;
}

/// A reconnect whose object is gone from the store, with a layer checkpoint
/// to restore: the catch-up's header read finds nothing, the recovery hook
/// restores the layer (the only restore entry a subscribe reaches without a
/// race; see SDD §15.1), and the catch-up goes on out of window (a restore
/// writes no stream entries).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_layer_recovery_stays_within_the_stack_budget() {
    run(async {
        const CASE: &str = "subscribe-layer-recovery";
        let fx = Fx::new().await;
        let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
        let checkpoints = layer_checkpoints(&session_id, object_id, 3).await;
        fx.wipe_object(object_id).await;
        let svc = fx
            .service(Opts {
                layer_checkpoints: Some(checkpoints),
                ..Opts::default()
            })
            .await;
        let (mut m, mut client) = serve(CASE, &svc).await;
        let (restores_before, regimes_before) = (
            restores(&svc, "layer"),
            catch_ups(&svc, "out_of_window", "legacy"),
        );
        m.open().await;
        let mut stream = subscribe(&mut client, &session_id, object_id, 0).await;
        drain_catch_up(&mut stream).await;
        let probe = m.finish().await;
        let route = expect_eq(
            "layer restores",
            restores(&svc, "layer") - restores_before,
            1.0,
        )
        .and(expect_eq(
            "out_of_window catch-ups",
            catch_ups(&svc, "out_of_window", "legacy") - regimes_before,
            1.0,
        ));
        drop(stream);
        verdict(CASE, &probe, route);
    })
    .await;
}

/// A first subscribe to a session gone from the store, with a session
/// checkpoint to restore: the handler's base-path lookup recovers the
/// session, then the catch-up serves the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_session_recovery_stays_within_the_stack_budget() {
    run(session_recovery("subscribe-session-recovery")).await;
}

/// The session-recovery subscribe of
/// [`subscribe_session_recovery_stays_within_the_stack_budget`], measured as
/// `case`. Returns the measured thread's depth.
async fn session_recovery(case: &'static str) -> usize {
    let fx = Fx::new().await;
    let (session_id, object_id) = fx.session_with_layer(&base_content()).await;
    let checkpoints =
        session_checkpoints(&fx.session_checkpoint(&session_id, object_id).await).await;
    fx.flush_session(&session_id).await;
    let svc = fx
        .service(Opts {
            session_checkpoints: Some(checkpoints),
            ..Opts::default()
        })
        .await;
    let (mut m, mut client) = serve(case, &svc).await;
    let (restores_before, regimes_before) = (
        restores(&svc, "session"),
        catch_ups(&svc, "brand_new", "snapshot"),
    );
    m.open().await;
    let mut stream = subscribe(&mut client, &session_id, 0, 0).await;
    drain_catch_up(&mut stream).await;
    let probe = m.finish().await;
    let route = expect_eq(
        "session restores",
        restores(&svc, "session") - restores_before,
        1.0,
    )
    .and(expect_eq(
        "brand_new catch-ups",
        catch_ups(&svc, "brand_new", "snapshot") - regimes_before,
        1.0,
    ));
    drop(stream);
    verdict(case, &probe, route);
    probe.depth()
}

/// How much deeper than the up-to-date catch-up, in one run, a subscribe
/// that reads a snapshot may go on its worker
/// ([`subscribe_snapshot_reads_go_no_deeper_than_the_up_to_date_catch_up`]).
/// Over these cases the read's own frames measured +0 to +5.2 KiB on main at
/// f199e302 and +0 on the window script's tree (aarch64); a pass future
/// built inline in `FanoutThreads::run_pass`, not boxed as it is built,
/// measured +91 to +98 KiB. 32 KiB sits between them, with about 27 KiB of
/// room above main's figures for codegen drift and about 59 KiB below the
/// inline build's.
const SNAPSHOT_READ_OVER_UP_TO_DATE_MARGIN: usize = 32 * KIB;

/// The relative-depth check: in one run of one build, every subscribe case
/// that reads a snapshot without a fan-out pass — the first subscribe, the
/// in-window and out-of-window reconnects, each in both catch-up modes, and
/// the session recovery's first subscribe — goes at most
/// [`SNAPSHOT_READ_OVER_UP_TO_DATE_MARGIN`] deeper on its worker than the
/// up-to-date catch-up, which reads nothing. It pins that no frame the
/// snapshot read keeps on the worker grows by tens of KiB, a growth well
/// inside the 512 KiB budget the per-case tests check: a fan-out pass future
/// built inline in `FanoutThreads::run_pass` (instead of
/// `storage::fanout_thread::build_pass_on_heap`) deepened every one of these
/// routes, pass or no pass, by 91 to 98 KiB.
///
/// One test serves the baseline and every read case in sequence, each on its
/// own measured thread and fresh service, as the per-case tests do: both
/// sides come from one process of one build, so the comparison does not
/// depend on the architecture, the toolchain or another process's figures.
/// The baseline is measured first. Each case still checks its route and the
/// budget ([`verdict`]); its line is printed under a `relative-` name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    debug_assertions,
    ignore = "release-build stack guard: run with --cargo-profile release (CI: test-release-stack-guard)"
)]
async fn subscribe_snapshot_reads_go_no_deeper_than_the_up_to_date_catch_up() {
    run(async {
        let baseline = up_to_date("relative-subscribe-up-to-date").await;
        let reads = [
            (
                "relative-subscribe-brand-new-snapshot",
                brand_new(
                    "relative-subscribe-brand-new-snapshot",
                    true,
                    base_content(),
                    false,
                )
                .await,
            ),
            (
                "relative-subscribe-brand-new-legacy",
                brand_new(
                    "relative-subscribe-brand-new-legacy",
                    false,
                    base_content(),
                    false,
                )
                .await,
            ),
            (
                "relative-subscribe-in-window-snapshot",
                in_window("relative-subscribe-in-window-snapshot", true).await,
            ),
            (
                "relative-subscribe-in-window-legacy",
                in_window("relative-subscribe-in-window-legacy", false).await,
            ),
            (
                "relative-subscribe-out-of-window-snapshot",
                out_of_window("relative-subscribe-out-of-window-snapshot", true).await,
            ),
            (
                "relative-subscribe-out-of-window-legacy",
                out_of_window("relative-subscribe-out-of-window-legacy", false).await,
            ),
            (
                "relative-subscribe-session-recovery",
                session_recovery("relative-subscribe-session-recovery").await,
            ),
        ];
        let limit = baseline + SNAPSHOT_READ_OVER_UP_TO_DATE_MARGIN;
        let mut over = Vec::new();
        for (case, depth) in reads {
            let pass = depth <= limit;
            println!(
                "stack-guard {case} against the up-to-date catch-up: {:.1} KiB used, up-to-date \
                 {:.1} KiB, {:+.1} KiB, margin {} KiB, {}",
                kib(depth),
                kib(baseline),
                kib(depth) - kib(baseline),
                SNAPSHOT_READ_OVER_UP_TO_DATE_MARGIN / KIB,
                if pass { "PASS" } else { "FAIL" },
            );
            if !pass {
                over.push(format!(
                    "{case} {:.1} KiB (+{:.1} KiB)",
                    kib(depth),
                    kib(depth) - kib(baseline)
                ));
            }
        }
        assert!(
            over.is_empty(),
            "release-build stack use of a subscribe that reads a snapshot went more than {} KiB \
             deeper than the up-to-date catch-up's {:.1} KiB in the same run: {over:?}. A frame \
             the snapshot read keeps on the worker grew; a fan-out pass future built inline in \
             `FanoutThreads::run_pass` is one such growth. Move the frame off the stack (SDD \
             §15.1).",
            SNAPSHOT_READ_OVER_UP_TO_DATE_MARGIN / KIB,
            kib(baseline),
        );
    })
    .await;
}
