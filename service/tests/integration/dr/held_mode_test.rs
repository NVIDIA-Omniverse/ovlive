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

//! Held-mode startup (`live_edit::grpc::hold::run_held_mode`; SDD SR-5.1–5.6,
//! SR-6.4, SR-6.5, SR-7, SR-7a): what a replica runs when its startup hold
//! evaluation finds the fleet held, in place of the normal startup. These
//! tests drive the held branch in process — the same function `main.rs`
//! calls, over a real store, a real TCP socket, and the real `/metrics` and
//! `/scaling/v1` server — so every branch of it can be reached and its
//! signals read. `tests/misc/held_mode_binary.rs` runs the real binary
//! through held mode under a test-only key prefix (readiness, one refusal,
//! SIGTERM, the un-hold exit), and `tests/misc/hold_wiring_scan.rs` pins the
//! `main.rs` glue neither reaches: the non-zero exit of a failed held mode,
//! the SIGTERM handler registered before the handoff, and where held mode
//! sits in the startup order.
//!
//! - **The held server** ([`held_mode_refuses_every_live_service_rpc_and_serves_health_reflection_and_capabilities`]).
//!   Every LiveService RPC, credentialed or not, is refused `UNAVAILABLE`
//!   `SERVICE_HELD` naming the marker's incident, with one INFO completion
//!   line each, and names a new incident once the marker is replaced;
//!   health, reflection and the CapabilitiesService answer; readiness and
//!   `""` are `SERVING`; `/scaling/v1` reports `held` and `ready`; and
//!   `/metrics` shows `live_edit_dr_held 1` from the first scrape held mode
//!   answers, with no hold entry counted.
//! - **Its HTTP/2 connection window**
//!   ([`held_mode_advertises_the_connection_window_its_upload_waiter_cap_sizes`]):
//!   sized from `HeldMode::upload_admission_max_waiters`, as on the serving
//!   server (SDD §5.1), read off the wire with a non-default cap.
//! - **It writes nothing** ([`held_mode_leaves_the_store_keyspace_unchanged`]):
//!   every key under the test's prefix is byte-identical, and no TTL grew,
//!   across a run with ten hold reads and refused RPCs. That held mode never
//!   reaches the backup store holds by construction — it is handed none —
//!   and is pinned by `hold_wiring_scan.rs`.
//! - **The exits.** The marker removed: exit 0 through the seam, `held`
//!   cleared, and the function returns
//!   ([`held_mode_exits_zero_when_the_marker_is_removed`]). The shutdown
//!   signal: it returns promptly with no exit, readiness and `""`
//!   `NOT_SERVING`, `/scaling/v1` `shutting_down`, and the socket closed
//!   ([`held_mode_stops_on_the_shutdown_signal_without_exiting`]); with a
//!   call still open on the server, it stops waiting for the server after
//!   its 5 s budget, says so in one WARN, and returns `Ok`
//!   ([`held_mode_with_a_stream_open_stops_waiting_for_its_server_after_the_stop_budget`]).
//!   The reactor task failing: it returns `ReactorFailed` and logs one ERROR
//!   ([`held_mode_returns_reactor_failed_when_its_hold_reactor_panics`]).
//! - **A `/scaling/v1` source already registered**
//!   ([`held_mode_with_a_scaling_source_already_registered_logs_one_error_and_keeps_it`]):
//!   one ERROR saying the scaler does not see this replica as held; the
//!   first source stays and held mode serves on.
//! - **A dead hold poll** ([`held_mode_with_failing_hold_reads_reports_not_serving_stays_held_and_recovers`];
//!   defer-16): about 30 s after the last read that answered, readiness goes
//!   `NOT_SERVING` and the stale gauge 1 while every RPC is still refused
//!   `SERVICE_HELD` and the replica does not exit; a read that answers `Held`
//!   clears both, and the marker's removal then exits 0.
//! - **The held-mode reactor on a paused clock**
//!   (`held_mode_reactor_*`): `held` set at spawn; a `Held` verdict changes
//!   nothing; `Serving` exits 0 after its drawn jitter; with no read,
//!   `NOT_SERVING` at exactly 30 s with no exit, and still no exit an hour
//!   later, when a `Serving` read exits 0; a fresh `Held` read clears the
//!   stale signals and restarts the stale clock; a verdict channel that
//!   closes without the shutdown token exits non-zero at once; the shutdown
//!   token stops it with no exit.
//!
//! PM B1 — a marker that appears after the startup evaluation but before the
//! serving point exits 0 at once through the seam, entering no hold — is the
//! normal reactor's: `hold_reactor_test.rs`
//! (`reactor_hold_arriving_before_the_serving_point_exits_zero_at_once`) and
//! `hold_entry_test.rs`
//! (`a_marker_before_the_serving_point_exits_zero_without_entering_the_hold`).
//! The scaler's freeze with every pod held is `scaler/tests/integration/engine.rs`'s.
//!
//! # Clocks and waits
//!
//! The reactor tests run on a current-thread runtime with tokio's clock
//! paused, so the jitter and the 30 s clock are pinned on virtual time: no
//! assertion there depends on how fast the runner is. The 30 s clock is the
//! production constant (not injectable).
//!
//! The held-server tests run on real time, on a current-thread runtime so a
//! [`ThreadCapture`] sees what the spawned tasks log, and assert no upper
//! bound on a duration; the open-stream test asserts its wait lasted at least
//! the stop budget. Every wait is a deadline a healthy run never reaches —
//! [`SETTLE`] (10 s)
//! per awaited event (a connect, a readiness change, a metric tick, the
//! function's return), [`EXIT_DEADLINE`] (20 s) for an un-hold exit (one
//! poll interval plus the 5 s jitter bound), and [`STALE_DEADLINE`] (45 s)
//! for the stale readiness. The heaviest of them other than the dead-poll
//! test, the first, waits eight [`SETTLE`]s at every budget's full value (the
//! connect, `/metrics`, readiness, `/scaling/v1`, every refusal's metric tick
//! under one deadline, reflection, the replaced incident, the return): 80 s,
//! inside nextest's 120 s hang-killer. A healthy run pays none of them, except
//! the open-stream test, which pays the server's 5 s stop budget in every run
//! (that budget running out is what it checks).
//!
//! The dead-poll test pays the production 30 s threshold in every run, and up
//! to 5 s of jitter at its exit: 30 to 40 s in all. At every budget's full
//! value it waits about 160 s: [`STALE_DEADLINE`], [`EXIT_DEADLINE`] and
//! about ten [`SETTLE`]s (connect, two readiness changes, the scrapes and
//! the scaling snapshot, the return), so `.config/nextest.toml` gives it its
//! own 240 s deadline. Its 30 s cannot be paid on a paused clock: the real
//! store's replies and the client's own timers share the runtime's clock,
//! and a paused clock would fire them early.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::StreamExt;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Request, Status};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_reflection::pb::v1::server_reflection_client::ServerReflectionClient;
use tonic_reflection::pb::v1::server_reflection_request::MessageRequest;
use tonic_reflection::pb::v1::ServerReflectionRequest;

use live_edit::dr::hold::{HoldObservation, STALE_VERDICT_THRESHOLD, UNHOLD_EXIT_JITTER_MAX};
use live_edit::grpc::hold::{
    spawn_held_mode_reactor, HeldModeError, HoldExit, HoldReactorSettings,
    HELD_MODE_SERVER_STOP_BUDGET,
};
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{
    CreateLayerRequest, CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest,
    JoinSessionRequest, ListSessionsRequest, PingRequest, StatLayerRequest,
    SubscribeReadLayerRequest, UpdateLayerRequest,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, DR_HOLD_REASON_KEYSPACE_LOST};
use live_edit::observability::scaling::{ScalingEndpoint, ScalingSnapshotSource};
use live_edit::storage::client::STORE_NOT_SERVING_SHAPES;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_client::CapabilitiesServiceClient;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::ListServicesRequest;
use live_edit_scaler::snapshot::ScalingSnapshot;

use super::hold_support::{startup_held, HeldModeRun, EXIT_DEADLINE, POLL_INTERVAL, SETTLE};
use super::support::{
    health_status, held, hold_entries, keyspace, keyspace_writes, lines_starting, serving,
    until_readiness, HealthSource, RecordingExit, WriteRules, READINESS,
};
use crate::common::refusal_detail::assert_service_held;
use crate::grpc::log_capture::{
    assert_common_fields, purge_keys, Captured, ThreadCapture, RPC_FAILURE_MSG,
};
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on the stale readiness after the hold reads start failing: the 30 s
/// threshold, counted from the last read that answered, and slack.
const STALE_DEADLINE: Duration = Duration::from_secs(45);

/// The bearer on the credentialed calls. Nothing checks it: the stub refuses
/// before any auth.
const BEARER: &str = "held-mode-bearer";

/// The incident the hold marker names.
const INCIDENT: &str = "20261015t2210-a1b2";

/// The incident a second marker names while the replica is held.
const REPLACED_INCIDENT: &str = "20261015t2245-c3d4";

/// The heartbeat interval the paused-clock reactor tests run with.
const TEST_HEARTBEAT: Duration = Duration::from_secs(7);

/// Virtual-time bound on every wait of the paused-clock tests. Reached only
/// if the awaited event never comes.
const VIRTUAL_SETTLE: Duration = Duration::from_secs(60);

/// One step of virtual time: a reaction due at an instant has happened one
/// step after it and not one step before.
const STEP: Duration = Duration::from_millis(1);

/// The startup ERROR of held mode, matched on its fixed start.
const HELD_MODE_START_PREFIX: &str =
    "the fleet is held for disaster recovery: this replica starts in held mode.";

/// The held server's INFO once it takes the socket.
const HELD_SERVING_LINE: &str = "gRPC server serving in held mode: the refuse-all LiveService stub, health, reflection and capabilities";

/// The INFO on the shutdown signal.
const HELD_SIGNAL_LINE: &str =
    "held mode: stopping on the shutdown signal; there are no LiveService streams to drain";

/// The WARN when the server has not stopped within its stop budget.
const HELD_STOP_BUDGET_PREFIX: &str =
    "held mode: the gRPC server did not stop within its budget; leaving it";

/// The ERROR when the `/scaling/v1` slot already held a source.
const SCALING_ALREADY_REGISTERED_PREFIX: &str =
    "held mode: a /scaling/v1 source was already registered";

/// The INFO when the reactor ends after its exit seam returned.
const HELD_REACTOR_ENDED_LINE: &str = "held mode: the hold reactor ended after its exit; stopping";

/// The ERROR when the reactor task fails.
const HELD_REACTOR_FAILED_PREFIX: &str = "held mode: the hold reactor failed";

/// The un-hold INFO line, matched on its fixed start.
const UNHOLD_PREFIX: &str = "the disaster-recovery hold cleared";

/// The WARN of a held replica whose verdict goes stale.
const HELD_STALE_PREFIX: &str =
    "this held replica has had no successful disaster-recovery hold read";

/// The INFO of a held replica whose stale verdict is fresh again.
const HELD_STALE_CLEARED_PREFIX: &str =
    "the disaster-recovery hold read succeeded again; this held replica's verdict is no longer \
     stale";

/// The held heartbeat's WARN, matched on its fixed start.
const HEARTBEAT_PREFIX: &str = "this replica is still held for disaster recovery";

// ═════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═════════════════════════════════════════════════════════════════════════════

fn marker(incident: &str) -> HoldMarkerSeed {
    HoldMarkerSeed::default()
        .with_incident(incident)
        .with_reason(DR_HOLD_REASON_KEYSPACE_LOST)
}

fn gauge(metrics: &Metrics, name: &str) -> u64 {
    sample_value(&render(metrics), name, &[])
        .unwrap_or_else(|| panic!("{name} is recorded at construction")) as u64
}

/// An exit seam that panics, so the reactor task fails.
struct PanickingExit;

impl HoldExit for PanickingExit {
    fn exit(&self, code: i32) -> BoxFuture<'static, ()> {
        panic!("test exit seam: refusing to exit {code}");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The held-mode reactor, on a paused clock
// ═════════════════════════════════════════════════════════════════════════════

/// A running held-mode reactor, the channel and token the test drives it
/// through, and receivers of the two signals it publishes.
struct HeldReactor {
    verdicts: watch::Sender<HoldObservation>,
    metrics: Arc<Metrics>,
    shutdown: CancellationToken,
    exits: mpsc::UnboundedReceiver<(i32, Instant)>,
    task: JoinHandle<()>,
    /// `HoldReactorSettings::verdict_stale`: the readiness input.
    verdict_stale: watch::Receiver<bool>,
    /// `HoldReactorSettings::held`: the scaling snapshot's `held`.
    held: watch::Receiver<bool>,
}

impl HeldReactor {
    /// A held-mode reactor whose channel starts at `initial`.
    fn start(initial: HoldObservation) -> Self {
        let (verdicts, rx) = watch::channel(initial);
        let metrics = Arc::new(crate::common::metrics_scrape_only("held-mode-reactor"));
        let shutdown = CancellationToken::new();
        let (calls, exits) = mpsc::unbounded_channel();
        let stale_tx = watch::Sender::new(false);
        let held_tx = watch::Sender::new(false);
        let verdict_stale = stale_tx.subscribe();
        let held = held_tx.subscribe();
        let task = spawn_held_mode_reactor(
            rx,
            Arc::clone(&metrics),
            shutdown.clone(),
            HoldReactorSettings {
                heartbeat_interval: TEST_HEARTBEAT,
                exit: Arc::new(RecordingExit { calls }),
                verdict_stale: stale_tx,
                held: held_tx,
            },
        );
        Self {
            verdicts,
            metrics,
            shutdown,
            exits,
            task,
            verdict_stale,
            held,
        }
    }

    fn is_stale(&self) -> bool {
        *self.verdict_stale.borrow()
    }

    fn is_held_signal(&self) -> bool {
        *self.held.borrow()
    }

    fn stale_gauge(&self) -> u64 {
        gauge(&self.metrics, "live_edit_dr_hold_verdict_stale")
    }

    /// Publish `observation`, as a successful hold read does.
    fn publish(&self, observation: HoldObservation) {
        self.verdicts.send_replace(observation);
    }

    /// The next exit-seam call, within [`VIRTUAL_SETTLE`] plus `extra` of
    /// virtual time.
    async fn next_exit(&mut self, extra: Duration, what: &str) -> (i32, Instant) {
        tokio::time::timeout(VIRTUAL_SETTLE + extra, self.exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: the exit seam was not called"))
            .unwrap_or_else(|| panic!("{what}: the reactor dropped its exit seam uncalled"))
    }

    /// Nothing has called the exit seam so far.
    fn assert_no_exit_yet(&mut self, what: &str) {
        if let Ok(call) = self.exits.try_recv() {
            panic!("{what}: the exit seam must not be called; got {call:?}");
        }
    }

    /// Wait for the reactor task to return, without failing.
    async fn stopped(&mut self, what: &str) {
        tokio::time::timeout(VIRTUAL_SETTLE, &mut self.task)
            .await
            .unwrap_or_else(|_| panic!("{what}: the reactor did not stop"))
            .unwrap_or_else(|e| panic!("{what}: the reactor task failed: {e}"));
    }
}

/// Poll `cond` until it holds, within [`VIRTUAL_SETTLE`] of virtual time.
async fn wait_virtual(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + VIRTUAL_SETTLE;
    while !cond() {
        assert!(Instant::now() < deadline, "{what} did not happen");
        tokio::time::sleep(STEP).await;
    }
}

/// The jitter the un-hold INFO line drew: the only un-hold line captured.
fn drawn_jitter(captured: &Captured) -> Duration {
    let lines = lines_starting(captured, UNHOLD_PREFIX);
    assert_eq!(
        lines.len(),
        1,
        "one un-hold line per exit:\n{}",
        captured.text
    );
    assert_eq!(lines[0]["level"], "INFO");
    let ms = lines[0]["fields"]["jitter_ms"]
        .as_u64()
        .expect("the un-hold line carries its jitter");
    Duration::from_millis(ms)
}

/// Assert an exit at `at` came `jitter` after `from`: the jitter is drawn at
/// sub-millisecond precision and logged in whole milliseconds, so the wait
/// is within one millisecond above the logged value, and never past the
/// bound.
fn assert_exit_after_jitter(from: Instant, at: Instant, jitter: Duration, what: &str) {
    let waited = at - from;
    assert!(
        waited >= jitter && waited < jitter + STEP,
        "{what}: the exit waited {waited:?}, not the {jitter:?} the un-hold line drew"
    );
    assert!(
        waited <= UNHOLD_EXIT_JITTER_MAX,
        "{what}: the exit waited {waited:?}, past the {UNHOLD_EXIT_JITTER_MAX:?} jitter bound"
    );
}

/// The held-mode reactor reports `held` the moment it is spawned, before it
/// is first polled, so the scaler freezes from held mode's start (SDD SR-7a).
/// A `Held` verdict, re-read for 150 s, changes nothing: no exit, no stale
/// signal, no hold entry (the gauge and the entry counter are held mode's
/// start's business, not the reactor's), and the heartbeat names the latest
/// incident.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_reports_held_at_spawn_and_acts_on_no_held_verdict() {
    let capture = ThreadCapture::start();
    let mut reactor = HeldReactor::start(held(INCIDENT));
    assert!(
        reactor.is_held_signal(),
        "held is set at spawn, before the reactor task first runs"
    );

    let started = Instant::now();
    while Instant::now() < started + Duration::from_secs(150) {
        tokio::time::sleep(Duration::from_secs(10)).await;
        reactor.publish(held(REPLACED_INCIDENT));
        reactor.assert_no_exit_yet("a Held verdict");
        assert!(!reactor.is_stale(), "a fresh Held read is not stale");
    }

    assert!(reactor.is_held_signal(), "still held");
    assert!(!reactor.task.is_finished(), "the reactor runs on");
    assert_eq!(reactor.stale_gauge(), 0);
    assert_eq!(
        gauge(&reactor.metrics, "live_edit_dr_held"),
        0,
        "the reactor sets no hold gauge: held mode's start does"
    );
    assert!(
        hold_entries(&render(&reactor.metrics))
            .iter()
            .all(|(_, n)| *n == 0),
        "the reactor runs no hold entry"
    );
    let captured = capture.snapshot();
    let beats = lines_starting(&captured, HEARTBEAT_PREFIX);
    assert!(
        beats.len() >= 2,
        "the held heartbeat runs from spawn:\n{}",
        captured.text
    );
    assert_eq!(
        beats.last().expect("non-empty")["fields"]["incident"],
        REPLACED_INCIDENT,
        "the heartbeat names the latest incident"
    );
    reactor.shutdown.cancel();
    reactor.stopped("after the shutdown token").await;
}

/// `Serving` exits 0 after the jitter the un-hold line draws, within the
/// 5 s bound, with `held` cleared first; once, and the reactor then returns.
/// A `Serving` first value — the startup evaluation's — is acted on the same
/// way.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_exits_zero_after_its_jitter_when_the_hold_clears() {
    for first_value_serving in [false, true] {
        let what = if first_value_serving {
            "a Serving first value"
        } else {
            "a Serving read after Held"
        };
        let capture = ThreadCapture::start();
        let spawned = Instant::now();
        let mut reactor = HeldReactor::start(if first_value_serving {
            serving()
        } else {
            held(INCIDENT)
        });
        assert!(reactor.is_held_signal(), "{what}: held at spawn");
        let cleared = if first_value_serving {
            spawned
        } else {
            tokio::time::sleep(Duration::from_secs(3)).await;
            reactor.assert_no_exit_yet("held");
            reactor.publish(serving());
            Instant::now()
        };

        let (code, at) = reactor.next_exit(Duration::ZERO, what).await;

        assert_eq!(code, 0, "{what}: an un-held replica exits 0");
        assert_exit_after_jitter(cleared, at, drawn_jitter(&capture.snapshot()), what);
        assert!(
            !reactor.is_held_signal(),
            "{what}: held is cleared before the exit"
        );
        reactor.stopped(what).await;
        reactor.assert_no_exit_yet(&format!("{what}: it exits once"));
        assert!(!reactor.is_stale(), "{what}: never stale");
    }
}

/// With no successful read — a dead hold poll — the held-mode reactor stays
/// held: at exactly 30 s it sets the readiness input and the stale gauge,
/// logs one WARN, and does not exit (the normal reactor's pre-serving-point
/// arm would exit non-zero there).
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_with_no_read_reports_stale_at_30_s() {
    let capture = ThreadCapture::start();
    let last_read = Instant::now();
    let mut reactor = HeldReactor::start(held(INCIDENT));

    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD - STEP).await;
    assert!(!reactor.is_stale(), "fresh until 30 s");
    assert_eq!(reactor.stale_gauge(), 0);

    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(
        reactor.is_stale(),
        "readiness NOT_SERVING 30 s after the last read"
    );
    assert_eq!(reactor.stale_gauge(), 1);
    assert!(reactor.is_held_signal(), "a stale verdict stays held");
    reactor.assert_no_exit_yet("a held-mode verdict going stale");
    assert!(!reactor.task.is_finished(), "the reactor runs on, stale");

    let captured = capture.snapshot();
    let stale = lines_starting(&captured, HELD_STALE_PREFIX);
    assert_eq!(stale.len(), 1, "one stale WARN:\n{}", captured.text);
    assert_eq!(stale[0]["level"], "WARN");
    assert_eq!(stale[0]["fields"]["incident"], INCIDENT);
    reactor.shutdown.cancel();
    reactor.stopped("after the shutdown token").await;
}

/// A fresh `Held` read after the verdict went stale clears the readiness
/// input and the stale gauge with one INFO, exits nothing, and restarts the
/// stale clock from that read: stale again 30 s after it.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_fresh_held_read_clears_the_stale_signals_and_restarts_its_stale_clock() {
    let capture = ThreadCapture::start();
    let first_read = Instant::now();
    let mut reactor = HeldReactor::start(held(INCIDENT));
    tokio::time::sleep_until(first_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(reactor.is_stale(), "precondition: stale");

    let fresh_read = Instant::now();
    reactor.publish(held(INCIDENT));
    wait_virtual("the stale signals to clear", || !reactor.is_stale()).await;
    assert_eq!(reactor.stale_gauge(), 0, "the gauge clears with the signal");
    reactor.assert_no_exit_yet("a fresh Held read");
    let captured = capture.snapshot();
    let cleared = lines_starting(&captured, HELD_STALE_CLEARED_PREFIX);
    assert_eq!(cleared.len(), 1, "one INFO:\n{}", captured.text);
    assert_eq!(cleared[0]["level"], "INFO");
    assert_eq!(cleared[0]["fields"]["held"], true);

    tokio::time::sleep_until(fresh_read + STALE_VERDICT_THRESHOLD - STEP).await;
    assert!(!reactor.is_stale(), "the stale clock restarts at the read");
    tokio::time::sleep_until(fresh_read + STALE_VERDICT_THRESHOLD + STEP).await;
    reactor.assert_no_exit_yet("stale again after the fresh read");
    assert!(reactor.is_stale(), "stale again 30 s after the fresh read");
    reactor.shutdown.cancel();
    reactor.stopped("after the shutdown token").await;
}

/// A `Serving` read after the verdict went stale clears the stale signals,
/// clears `held`, and exits 0 after the drawn jitter.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_serving_read_while_stale_exits_zero_after_its_jitter() {
    let capture = ThreadCapture::start();
    let mut reactor = HeldReactor::start(held(INCIDENT));
    tokio::time::sleep(STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(reactor.is_stale(), "precondition: stale");

    let cleared = Instant::now();
    reactor.publish(serving());
    let (code, at) = reactor
        .next_exit(Duration::ZERO, "Serving while stale")
        .await;

    assert_eq!(code, 0);
    let captured = capture.snapshot();
    assert_exit_after_jitter(cleared, at, drawn_jitter(&captured), "Serving while stale");
    assert!(!reactor.is_stale(), "the readiness input is cleared");
    assert_eq!(reactor.stale_gauge(), 0, "the stale gauge is cleared");
    assert!(!reactor.is_held_signal(), "held is cleared");
    assert_eq!(
        lines_starting(&captured, HELD_STALE_CLEARED_PREFIX).len(),
        1,
        "{}",
        captured.text
    );
}

/// A held-mode replica whose verdict went stale waits on no clock: an hour
/// with no read exits nothing, and it stays held and stale (readiness input,
/// gauge, `held`). A `Serving` read then exits 0 after its drawn jitter.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_stale_for_an_hour_exits_nothing_then_zero_on_a_serving_read() {
    let capture = ThreadCapture::start();
    let last_read = Instant::now();
    let mut reactor = HeldReactor::start(held(INCIDENT));
    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(reactor.is_stale(), "precondition: stale at 30 s");

    tokio::time::sleep(Duration::from_secs(3600)).await;

    reactor.assert_no_exit_yet("an hour with no read");
    assert!(reactor.is_stale(), "still NOT_SERVING");
    assert_eq!(reactor.stale_gauge(), 1);
    assert!(reactor.is_held_signal(), "still held");
    assert!(!reactor.task.is_finished(), "the reactor runs on");

    let read = Instant::now();
    reactor.publish(serving());
    let (code, at) = reactor
        .next_exit(Duration::ZERO, "a Serving read after an hour")
        .await;
    assert_eq!(code, 0);
    assert_exit_after_jitter(
        read,
        at,
        drawn_jitter(&capture.snapshot()),
        "a Serving read after an hour",
    );
    assert!(!reactor.is_held_signal(), "held is cleared");
    reactor.stopped("after the exit").await;
}

/// The hold poll's channel closing without the shutdown token exits
/// `NO_VERDICT_EXIT_CODE` at the instant it closes, fresh or stale, with one
/// ERROR naming the replica held and whether it was stale, and the reactor
/// stops. After the shutdown token, a closed channel exits nothing.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_closed_verdict_channel_exits_non_zero_at_once() {
    use live_edit::grpc::hold::NO_VERDICT_EXIT_CODE;
    const CHANNEL_CLOSED_PREFIX: &str = "hold: the hold poll ended before shutdown";
    let capture = ThreadCapture::start();
    for (case, go_stale) in [("fresh", false), ("stale", true)] {
        let last_read = Instant::now();
        let reactor = HeldReactor::start(held(INCIDENT));
        if go_stale {
            tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
            assert!(reactor.is_stale(), "{case}: precondition");
        } else {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        let errors_before = lines_starting(&capture.snapshot(), CHANNEL_CLOSED_PREFIX).len();

        let HeldReactor {
            verdicts,
            mut exits,
            task,
            ..
        } = reactor;
        let closed = Instant::now();
        drop(verdicts);
        let (code, at) = tokio::time::timeout(VIRTUAL_SETTLE, exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{case}: a closed channel exited nothing"))
            .unwrap_or_else(|| panic!("{case}: the reactor dropped its exit seam uncalled"));

        assert_eq!(code, NO_VERDICT_EXIT_CODE, "{case}");
        assert_eq!(at, closed, "{case}: at once");
        let captured = capture.snapshot();
        let errors = lines_starting(&captured, CHANNEL_CLOSED_PREFIX);
        assert_eq!(
            errors.len(),
            errors_before + 1,
            "{case}: one ERROR:\n{}",
            captured.text
        );
        let error = errors.last().expect("just counted");
        assert_eq!(error["level"], "ERROR");
        assert_eq!(error["fields"]["held"], true, "{case}");
        assert_eq!(error["fields"]["stale"], go_stale, "{case}");
        tokio::time::timeout(VIRTUAL_SETTLE, task)
            .await
            .unwrap_or_else(|_| panic!("{case}: the reactor did not stop"))
            .unwrap_or_else(|e| panic!("{case}: the reactor task failed: {e}"));
    }

    let reactor = HeldReactor::start(held(INCIDENT));
    let HeldReactor {
        verdicts,
        mut exits,
        task,
        shutdown,
        ..
    } = reactor;
    shutdown.cancel();
    drop(verdicts);
    tokio::time::sleep(Duration::from_secs(3600)).await;
    assert!(
        exits.try_recv().is_err(),
        "a channel closed after the shutdown token exits nothing"
    );
    tokio::time::timeout(VIRTUAL_SETTLE, task)
        .await
        .expect("the reactor stops on the token")
        .expect("and does not fail");
}

/// The shutdown token (held mode's SIGTERM path) stops the reactor: a later
/// `Serving` exits nothing, and the stale clock does not fire after it.
#[tokio::test(start_paused = true)]
async fn held_mode_reactor_stops_on_the_shutdown_token_and_exits_nothing() {
    let mut reactor = HeldReactor::start(held(INCIDENT));
    reactor.shutdown.cancel();
    reactor.publish(serving());
    tokio::time::sleep(Duration::from_secs(120) + UNHOLD_EXIT_JITTER_MAX).await;

    reactor.assert_no_exit_yet("after the shutdown token");
    assert!(
        !reactor.is_stale(),
        "no stale verdict after the shutdown token"
    );
    assert!(reactor.task.is_finished(), "the reactor stopped");
    reactor.stopped("after the shutdown token").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The held server, in process, on real time
// ═════════════════════════════════════════════════════════════════════════════

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {BEARER}")).expect("valid metadata"),
    );
    request
}

/// Every LiveService RPC, each refused: the metrics layer's endpoint name
/// and the status. Every call but `Ping` carries a bearer; the stub refuses
/// before any auth either way.
async fn call_every_live_service_rpc(channel: &Channel) -> Vec<(&'static str, Status)> {
    let mut client = LiveServiceClient::new(channel.clone());
    let refused = |what: &str| format!("{what} must be refused in held mode");
    vec![
        (
            "CreateSession",
            client
                .create_session(with_bearer(CreateSessionRequest::default()))
                .await
                .expect_err(&refused("CreateSession")),
        ),
        (
            "DeleteSession",
            client
                .delete_session(with_bearer(DeleteSessionRequest::default()))
                .await
                .expect_err(&refused("DeleteSession")),
        ),
        (
            "ListSessions",
            client
                .list_sessions(with_bearer(ListSessionsRequest::default()))
                .await
                .expect_err(&refused("ListSessions")),
        ),
        (
            "JoinSession",
            client
                .join_session(with_bearer(futures::stream::iter(vec![
                    JoinSessionRequest::default(),
                ])))
                .await
                .expect_err(&refused("JoinSession")),
        ),
        (
            "CreateLayer",
            client
                .create_layer(with_bearer(futures::stream::iter(vec![
                    CreateLayerRequest::default(),
                ])))
                .await
                .expect_err(&refused("CreateLayer")),
        ),
        (
            "UpdateLayer",
            client
                .update_layer(with_bearer(futures::stream::iter(vec![
                    UpdateLayerRequest::default(),
                ])))
                .await
                .expect_err(&refused("UpdateLayer")),
        ),
        (
            "DeleteLayer",
            client
                .delete_layer(with_bearer(DeleteLayerRequest::default()))
                .await
                .expect_err(&refused("DeleteLayer")),
        ),
        (
            "StatLayer",
            client
                .stat_layer(with_bearer(StatLayerRequest::default()))
                .await
                .expect_err(&refused("StatLayer")),
        ),
        (
            "SubscribeReadLayer",
            client
                .subscribe_read_layer(with_bearer(SubscribeReadLayerRequest::default()))
                .await
                .expect_err(&refused("SubscribeReadLayer")),
        ),
        (
            "Ping",
            client
                .ping(PingRequest {
                    client_timestamp_microseconds_since_epoch: 1,
                })
                .await
                .expect_err(&refused("Ping")),
        ),
    ]
}

/// One `StatLayer` call's refusal.
async fn stat_refusal(channel: &Channel) -> Status {
    LiveServiceClient::new(channel.clone())
        .stat_layer(with_bearer(StatLayerRequest::default()))
        .await
        .expect_err("StatLayer is refused in held mode")
}

/// A proxy rule acting on the hold poll's script frames only: `EVALSHA` or
/// `EVAL` naming `marker`, the test's prefixed marker key, which no other
/// client sends. `act` decides each such frame's action.
fn hold_read_rule(
    marker: String,
    act: impl Fn() -> Option<ProxyAction> + Send + Sync + 'static,
) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        if name != b"EVALSHA" && name != b"EVAL" {
            return None;
        }
        if !args.iter().any(|a| a.as_slice() == marker.as_bytes()) {
            return None;
        }
        act()
    })
}

/// A replica started with the marker present runs held mode: from the first
/// scrape it answers, `/metrics` shows `live_edit_dr_held 1` and no hold
/// entry; readiness and `""` are `SERVING` with no counter seed;
/// `/scaling/v1` reports `held` and `ready`; every LiveService RPC is
/// refused `UNAVAILABLE` `SERVICE_HELD` naming the incident, with one INFO
/// completion line each; health, reflection and the CapabilitiesService
/// answer; a replaced marker's incident is named by the next refusal; and
/// it never calls its exit seam. It logs one ERROR naming the hold at its
/// start and one INFO once it serves.
#[tokio::test]
async fn held_mode_refuses_every_live_service_rpc_and_serves_health_reflection_and_capabilities() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let mut run = HeldModeRun::start(tv.client().clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;

    // The gauge, from the first scrape held mode answers.
    let first_scrape = run.scrape().await;
    assert_eq!(
        sample_value(&first_scrape, "live_edit_dr_held", &[]),
        Some(1.0),
        "a replica started held reports the hold from its first scrape"
    );
    assert!(
        hold_entries(&first_scrape).iter().all(|(_, n)| *n == 0),
        "held mode counts no hold entry: none happened while serving"
    );

    // Readiness, liveness, the scaling snapshot.
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;
    assert_eq!(
        health_status(HealthSource::Wire(&channel), "").await,
        ServingStatus::Serving as i32
    );
    let snapshot = run.scaling().await;
    assert!(snapshot.held, "/scaling/v1 reports held");
    assert!(snapshot.ready, "a held replica stays ready");
    assert!(!snapshot.shutting_down);

    // Every LiveService RPC, refused SERVICE_HELD with one INFO line each.
    let calls = call_every_live_service_rpc(&channel).await;
    for (_, status) in &calls {
        assert_service_held(status, Some(INCIDENT));
    }
    let endpoints: Vec<&str> = calls.iter().map(|(endpoint, _)| *endpoint).collect();
    run.await_refused(&endpoints).await;
    let captured = capture.snapshot();
    let completions = captured.completion_lines();
    assert_eq!(
        completions.len(),
        calls.len(),
        "one completion line per refused call:\n{}",
        captured.text
    );
    for (endpoint, _) in &calls {
        let line = completions
            .iter()
            .find(|e| e["fields"]["endpoint"] == *endpoint)
            .unwrap_or_else(|| panic!("no completion line for {endpoint}:\n{}", captured.text));
        assert_eq!(line["fields"]["message"], RPC_FAILURE_MSG);
        assert_common_fields(line, "INFO", endpoint, "header", "UNAVAILABLE");
    }

    // Reflection and the CapabilitiesService answer.
    {
        use tonic_reflection::pb::v1::server_reflection_response::MessageResponse;
        let mut responses = ServerReflectionClient::new(channel.clone())
            .server_reflection_info(futures::stream::iter(vec![list_services_request()]))
            .await
            .expect("reflection answers in held mode")
            .into_inner();
        let answer = tokio::time::timeout(SETTLE, responses.next())
            .await
            .expect("the reflection answer within the deadline")
            .expect("one reflection answer")
            .expect("a reflection answer, not an error");
        let Some(MessageResponse::ListServicesResponse(list)) = answer.message_response else {
            panic!("a ListServices answer, got {answer:?}");
        };
        assert!(
            list.service.iter().any(|s| s.name == READINESS),
            "reflection lists the LiveService: {list:?}"
        );
    }
    CapabilitiesServiceClient::new(channel.clone())
        .list_services(ListServicesRequest::default())
        .await
        .expect("the CapabilitiesService answers in held mode");

    // A replaced marker: the next refusals name its incident.
    tv.write_hold_marker(&marker(REPLACED_INCIDENT)).await;
    let deadline = Instant::now() + SETTLE;
    loop {
        let status = stat_refusal(&channel).await;
        if status.message().contains(REPLACED_INCIDENT) {
            assert_service_held(&status, Some(REPLACED_INCIDENT));
            break;
        }
        assert_service_held(&status, Some(INCIDENT));
        assert!(
            Instant::now() < deadline,
            "the replaced incident was not named within {SETTLE:?}"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }

    run.assert_no_exit_yet("held");
    let captured = capture.snapshot();
    let start = lines_starting(&captured, HELD_MODE_START_PREFIX);
    assert_eq!(start.len(), 1, "one held-mode ERROR:\n{}", captured.text);
    assert_eq!(start[0]["level"], "ERROR");
    assert_eq!(start[0]["fields"]["incident"], INCIDENT);
    assert_eq!(start[0]["fields"]["reason"], DR_HOLD_REASON_KEYSPACE_LOST);
    assert_eq!(captured.with_message(HELD_SERVING_LINE).len(), 1);

    drop(channel);
    run.signal();
    run.returned(SETTLE, "the shutdown signal")
        .await
        .expect("held mode stops cleanly on the signal");
    drop(capture);
    purge_keys(&tv).await;
}

/// Held mode writes nothing to the store: across a run of at least ten hold
/// reads (counted at a pass-through proxy in front of the held client),
/// health probes and refused RPCs, every key under the test's prefix keeps
/// its exact bytes, no key is added or removed, and no TTL grows.
#[tokio::test]
async fn held_mode_leaves_the_store_keyspace_unchanged() {
    const READS: usize = 10;
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_read_rule(tv.key_schema().dr_hold_marker(), || {
            Some(ProxyAction::Delay(Duration::ZERO))
        }),
    )
    .await;
    let client = client_through(&proxy).await;
    let startup = startup_held(&tv).await;
    let before = keyspace(&tv).await;
    assert!(
        before.contains_key(&tv.key_schema().dr_hold_marker()),
        "precondition: the snapshot sees the marker"
    );

    let mut run = HeldModeRun::start(client.clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;
    for (_, status) in call_every_live_service_rpc(&channel).await {
        assert_service_held(&status, Some(INCIDENT));
    }
    let deadline = Instant::now() + SETTLE;
    while proxy.hits() < READS {
        assert!(
            Instant::now() < deadline,
            "the hold poll made {} reads within {SETTLE:?}",
            proxy.hits()
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    drop(channel);
    run.signal();
    run.returned(SETTLE, "the shutdown signal")
        .await
        .expect("held mode stops cleanly");

    let after = keyspace(&tv).await;
    let writes = keyspace_writes(&before, &after, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "held mode adds, removes and changes no key, and grows no TTL: {writes:#?}"
    );
    drop(client);
    purge_keys(&tv).await;
}

/// The marker removed: held mode's reactor calls the exit seam with 0 within
/// one poll interval plus the jitter bound (a deadline here; the virtual-time
/// bound is the reactor tests'), the scaling snapshot's `held` is cleared,
/// and with the test seam returning, `run_held_mode` stops and returns `Ok`.
#[tokio::test]
async fn held_mode_exits_zero_when_the_marker_is_removed() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let mut run = HeldModeRun::start(tv.client().clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;
    drop(channel);

    assert!(tv.remove_hold_marker().await, "fixture: the marker existed");
    let code = run.next_exit(EXIT_DEADLINE, "the marker removed").await;

    assert_eq!(code, 0, "an un-held replica exits 0 to restart normally");
    assert!(EXIT_DEADLINE > POLL_INTERVAL + UNHOLD_EXIT_JITTER_MAX);
    run.returned(SETTLE, "after the exit seam returned")
        .await
        .expect("held mode stops cleanly after its exit");
    assert!(
        !run.scaling().await.held,
        "held is cleared before the un-hold exit"
    );
    let captured = capture.snapshot();
    assert_eq!(
        captured.with_message(HELD_REACTOR_ENDED_LINE).len(),
        1,
        "{}",
        captured.text
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// The shutdown signal (in production, the process shutdown request held mode
/// claimed at its entry, cancelled on the first SIGTERM or SIGINT):
/// `run_held_mode` returns `Ok` within its server stop budget, with no
/// exit-seam call; readiness and `""` are `NOT_SERVING`; `/scaling/v1`
/// reports `shutting_down`; and the socket no longer accepts.
#[tokio::test]
async fn held_mode_stops_on_the_shutdown_signal_without_exiting() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let mut run = HeldModeRun::start(tv.client().clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;

    run.signal();
    run.returned(HELD_MODE_SERVER_STOP_BUDGET + SETTLE, "the shutdown signal")
        .await
        .expect("held mode stops cleanly on the signal");

    run.assert_no_exit_yet("the shutdown signal");
    assert_eq!(
        health_status(HealthSource::Reporter(&run.reporter), READINESS).await,
        ServingStatus::NotServing as i32
    );
    assert_eq!(
        health_status(HealthSource::Reporter(&run.reporter), "").await,
        ServingStatus::NotServing as i32
    );
    let snapshot = run.scaling().await;
    assert!(snapshot.shutting_down, "/scaling/v1 reports shutting_down");
    assert!(!snapshot.ready);
    let reconnect = tokio::time::timeout(
        SETTLE,
        Channel::from_shared(format!("http://{}", run.grpc))
            .expect("valid endpoint")
            .connect(),
    )
    .await
    .expect("a connect attempt ends within the deadline");
    assert!(
        reconnect.is_err(),
        "the held server's socket is closed once it returns"
    );
    let captured = capture.snapshot();
    assert_eq!(
        captured.with_message(HELD_SIGNAL_LINE).len(),
        1,
        "{}",
        captured.text
    );
    assert!(
        lines_starting(&captured, HELD_STOP_BUDGET_PREFIX).is_empty(),
        "with nothing open the server stops within its budget:\n{}",
        captured.text
    );
    drop(channel);
    drop(capture);
    purge_keys(&tv).await;
}

/// The HTTP/2 client connection preface (RFC 9113 §3.4).
const H2_CLIENT_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// An empty SETTINGS frame on stream 0: the client preface's second half.
const H2_EMPTY_SETTINGS: [u8; 9] = [0, 0, 0, 0x4, 0, 0, 0, 0, 0];

/// The HTTP/2 WINDOW_UPDATE frame type.
const H2_WINDOW_UPDATE: u8 = 0x8;

/// Every HTTP/2 connection's receive window before any WINDOW_UPDATE
/// (RFC 9113 §6.9.2); SETTINGS cannot change it.
const H2_DEFAULT_CONNECTION_WINDOW: u32 = 65_535;

/// The upload waiter cap the window test runs held mode with: not the default
/// (64), so a held server that ignored the field, or took the default in its
/// place, advertises a different connection window.
const NON_DEFAULT_UPLOAD_WAITERS: usize = 3;

/// The connection-level receive window `addr` advertises: open a raw TCP
/// connection, send the client preface and an empty SETTINGS frame, and read
/// frames until the first WINDOW_UPDATE on stream 0, whose increment is added
/// to 65,535. Panics if none arrives within [`SETTLE`] or the server closes.
async fn advertised_connection_window(addr: std::net::SocketAddr) -> u32 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut tcp = tokio::time::timeout(SETTLE, tokio::net::TcpStream::connect(addr))
        .await
        .expect("TCP connect within the deadline")
        .expect("TCP connect to the held server");
    tcp.write_all(H2_CLIENT_PREFACE)
        .await
        .expect("write the client preface");
    tcp.write_all(&H2_EMPTY_SETTINGS)
        .await
        .expect("write the client SETTINGS");
    tokio::time::timeout(SETTLE, async {
        loop {
            let mut header = [0u8; 9];
            tcp.read_exact(&mut header)
                .await
                .expect("the held server sends a WINDOW_UPDATE on stream 0 before it closes");
            let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
            let kind = header[3];
            let stream =
                u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
            let mut payload = vec![0u8; len];
            tcp.read_exact(&mut payload)
                .await
                .expect("read a frame payload");
            if kind == H2_WINDOW_UPDATE && stream == 0 {
                assert_eq!(len, 4, "a WINDOW_UPDATE payload is 4 bytes");
                let increment =
                    u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]])
                        & 0x7fff_ffff;
                return H2_DEFAULT_CONNECTION_WINDOW + increment;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no connection WINDOW_UPDATE within {SETTLE:?}"))
}

/// The held server sizes its HTTP/2 connection window from
/// `HeldMode::upload_admission_max_waiters`, as the serving server does
/// (SDD §5.1): run with a waiter cap of 3, it advertises
/// `http2_connection_window_bytes(3)` (4 MiB), not the default cap's 65 MiB.
/// The fixture's other tests pass the default, so they would not notice the
/// field ignored.
#[tokio::test]
async fn held_mode_advertises_the_connection_window_its_upload_waiter_cap_sizes() {
    use live_edit::grpc::transport::http2_connection_window_bytes;
    use live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT;

    assert_ne!(
        http2_connection_window_bytes(NON_DEFAULT_UPLOAD_WAITERS),
        http2_connection_window_bytes(UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT),
        "the test's cap must size a window the default does not"
    );
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let mut run = HeldModeRun::start_with_upload_waiters(
        tv.client().clone(),
        tv.key_schema(),
        startup,
        NON_DEFAULT_UPLOAD_WAITERS,
    )
    .await;

    assert_eq!(
        advertised_connection_window(run.grpc).await,
        http2_connection_window_bytes(NON_DEFAULT_UPLOAD_WAITERS),
        "the held server's connection window is (waiter cap + 1) x 1 MiB of the cap it was given"
    );

    run.signal();
    run.returned(HELD_MODE_SERVER_STOP_BUDGET + SETTLE, "the shutdown signal")
        .await
        .expect("held mode stops cleanly on the signal");
    purge_keys(&tv).await;
}

/// A `ListServices` reflection request.
fn list_services_request() -> ServerReflectionRequest {
    ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::ListServices(String::new())),
    }
}

/// The shutdown signal while a call is still open on the held server — a
/// reflection stream whose client keeps its side open — so the server cannot
/// stop gracefully: `run_held_mode` waits [`HELD_MODE_SERVER_STOP_BUDGET`]
/// for it, logs one WARN naming the budget, leaves it, and returns `Ok` with
/// no exit-seam call. A held pod sent SIGTERM ends within its budget whatever
/// a client holds open, rather than waiting for the kubelet's SIGKILL.
#[tokio::test]
async fn held_mode_with_a_stream_open_stops_waiting_for_its_server_after_the_stop_budget() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let mut run = HeldModeRun::start(tv.client().clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;

    // The stream stays open while `requests` lives; one answer shows the
    // server is serving it.
    let (requests, request_stream) = mpsc::channel(1);
    requests
        .send(list_services_request())
        .await
        .expect("the request stream is open");
    let mut responses = ServerReflectionClient::new(channel.clone())
        .server_reflection_info(ReceiverStream::new(request_stream))
        .await
        .expect("reflection answers in held mode")
        .into_inner();
    tokio::time::timeout(SETTLE, responses.next())
        .await
        .expect("the reflection answer within the deadline")
        .expect("one reflection answer")
        .expect("a reflection answer, not an error");

    let signalled = Instant::now();
    run.signal();
    run.returned(HELD_MODE_SERVER_STOP_BUDGET + SETTLE, "the shutdown signal")
        .await
        .expect("held mode returns Ok when it leaves its server");

    assert!(
        signalled.elapsed() >= HELD_MODE_SERVER_STOP_BUDGET,
        "the open stream held the server past the signal, so held mode waited out its budget"
    );
    run.assert_no_exit_yet("the shutdown signal");
    let captured = capture.snapshot();
    let warns = lines_starting(&captured, HELD_STOP_BUDGET_PREFIX);
    assert_eq!(warns.len(), 1, "one WARN:\n{}", captured.text);
    assert_eq!(warns[0]["level"], "WARN");
    assert_eq!(
        warns[0]["fields"]["budget_secs"],
        HELD_MODE_SERVER_STOP_BUDGET.as_secs()
    );
    drop(requests);
    drop(responses);
    drop(channel);
    drop(capture);
    purge_keys(&tv).await;
}

/// A `/scaling/v1` source that reports one fixed snapshot.
struct FixedScalingSource(ScalingSnapshot);

#[async_trait]
impl ScalingSnapshotSource for FixedScalingSource {
    async fn snapshot(&self) -> ScalingSnapshot {
        self.0.clone()
    }
}

/// Held mode handed a `/scaling/v1` slot that already holds a source cannot
/// register its own, so the scaler would not see this replica as held: it
/// logs one ERROR saying so, the first source stays in place, and held mode
/// serves on, refusing `SERVICE_HELD`.
#[tokio::test]
async fn held_mode_with_a_scaling_source_already_registered_logs_one_error_and_keeps_it() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let first = ScalingSnapshot::default();
    assert!(
        !first.held,
        "fixture: the first source does not report held"
    );
    let scaling = Arc::new(ScalingEndpoint::new());
    assert!(
        scaling.register(Arc::new(FixedScalingSource(first.clone()))),
        "fixture: the slot was empty"
    );
    let capture = ThreadCapture::start();
    let (calls, exits) = mpsc::unbounded_channel();
    let mut run = HeldModeRun::start_on_endpoint(
        tv.client().clone(),
        tv.key_schema(),
        startup,
        Arc::new(RecordingExit { calls }),
        exits,
        scaling,
    )
    .await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;

    assert_eq!(
        run.scaling().await,
        first,
        "/scaling/v1 still answers from the first source"
    );
    assert_service_held(&stat_refusal(&channel).await, Some(INCIDENT));
    let captured = capture.snapshot();
    let errors = lines_starting(&captured, SCALING_ALREADY_REGISTERED_PREFIX);
    assert_eq!(errors.len(), 1, "one ERROR:\n{}", captured.text);
    assert_eq!(errors[0]["level"], "ERROR");

    drop(channel);
    run.signal();
    run.returned(SETTLE, "the shutdown signal")
        .await
        .expect("held mode stops cleanly on the signal");
    drop(capture);
    purge_keys(&tv).await;
}

/// A held-mode reactor task that fails — here, an exit seam that panics at
/// the un-hold exit — leaves nothing acting on the verdict, so
/// `run_held_mode` logs one ERROR, runs its stop sequence (readiness
/// `NOT_SERVING`), and returns `HeldModeError::ReactorFailed`, which makes
/// `main.rs` exit non-zero.
#[tokio::test]
async fn held_mode_returns_reactor_failed_when_its_hold_reactor_panics() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let (_calls, exits) = mpsc::unbounded_channel();
    let mut run = HeldModeRun::start_with_exit(
        tv.client().clone(),
        tv.key_schema(),
        startup,
        Arc::new(PanickingExit),
        exits,
    )
    .await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;
    drop(channel);

    assert!(tv.remove_hold_marker().await, "fixture: the marker existed");
    let result = run
        .returned(EXIT_DEADLINE, "the reactor's panicking exit")
        .await;

    match result {
        Err(HeldModeError::ReactorFailed(e)) => {
            assert!(e.is_panic(), "the reactor task panicked: {e}");
        }
        other => panic!("a failed reactor must be ReactorFailed, got {other:?}"),
    }
    assert_eq!(
        health_status(HealthSource::Reporter(&run.reporter), READINESS).await,
        ServingStatus::NotServing as i32,
        "the stop sequence ran"
    );
    let captured = capture.snapshot();
    let failed = lines_starting(&captured, HELD_REACTOR_FAILED_PREFIX);
    assert_eq!(failed.len(), 1, "one ERROR:\n{}", captured.text);
    assert_eq!(failed[0]["level"], "ERROR");
    assert!(
        failed[0]["fields"]["error"].is_string(),
        "the ERROR names the join error"
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// The hold-read failures that make a dead hold poll (defer-16), staged as
/// the store's `-LOADING` refusal on the held client's hold reads only,
/// while its health `PING`s keep answering: about 30 s after the last read
/// that answered, readiness goes `NOT_SERVING` and
/// `live_edit_dr_hold_verdict_stale` 1, while the replica stays up, stays
/// held (`/metrics` `live_edit_dr_held 1`, `/scaling/v1` `held`), and still
/// refuses `SERVICE_HELD` naming the incident. The reads answering `Held`
/// again clear both, and the marker's removal then exits 0.
///
/// The 30 s threshold is the production constant, so this test runs 30 to
/// 40 s in a healthy run: the end-to-end proof that held mode's health
/// supervisor reads its reactor's stale signal. Every reaction's timing is
/// pinned on virtual time by the reactor tests above.
#[tokio::test]
async fn held_mode_with_failing_hold_reads_reports_not_serving_stays_held_and_recovers() {
    let tv = TestValkey::setup().await;
    tv.write_hold_marker(&marker(INCIDENT)).await;
    let failing = Arc::new(Mutex::new(false));
    let staged = Arc::clone(&failing);
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_read_rule(tv.key_schema().dr_hold_marker(), move || {
            staged
                .lock()
                .unwrap()
                .then(|| ProxyAction::Eval(error_reply_script(STORE_NOT_SERVING_SHAPES[0])))
        }),
    )
    .await;
    let client = client_through(&proxy).await;
    let startup = startup_held(&tv).await;
    let capture = ThreadCapture::start();
    let mut run = HeldModeRun::start(client.clone(), tv.key_schema(), startup).await;
    let channel = run.channel().await;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;

    // The hold reads start failing; the health probe keeps passing.
    *failing.lock().unwrap() = true;
    let armed = Instant::now();
    while health_status(HealthSource::Reporter(&run.reporter), READINESS).await
        != ServingStatus::NotServing as i32
    {
        assert!(
            Instant::now() < armed + STALE_DEADLINE,
            "readiness did not go NOT_SERVING within {STALE_DEADLINE:?} of the hold reads failing"
        );
        run.assert_no_exit_yet("failing hold reads");
        assert!(!run.run.is_finished(), "held mode stays up");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let hits_at_stale = proxy.hits();
    assert!(
        hits_at_stale > 0,
        "the staged refusal was taken by the hold reads"
    );

    // Stale, still held, still up, still refusing SERVICE_HELD.
    assert_eq!(
        health_status(HealthSource::Wire(&channel), READINESS).await,
        ServingStatus::NotServing as i32,
        "the held server answers readiness NOT_SERVING"
    );
    let scrape = run.scrape().await;
    assert_eq!(
        sample_value(&scrape, "live_edit_dr_hold_verdict_stale", &[]),
        Some(1.0)
    );
    assert_eq!(sample_value(&scrape, "live_edit_dr_held", &[]), Some(1.0));
    let snapshot = run.scaling().await;
    assert!(snapshot.held && !snapshot.ready, "{snapshot:?}");
    assert_service_held(&stat_refusal(&channel).await, Some(INCIDENT));
    run.assert_no_exit_yet("a stale held-mode verdict");
    assert!(!run.run.is_finished(), "held mode stays up while stale");

    // The reads answer Held again: both signals clear, no exit.
    *failing.lock().unwrap() = false;
    until_readiness(HealthSource::Wire(&channel), ServingStatus::Serving, SETTLE).await;
    let scrape = run.scrape().await;
    assert_eq!(
        sample_value(&scrape, "live_edit_dr_hold_verdict_stale", &[]),
        Some(0.0)
    );
    run.assert_no_exit_yet("a fresh Held read");

    // The marker removed: exit 0.
    assert!(tv.remove_hold_marker().await, "fixture: the marker existed");
    assert_eq!(run.next_exit(EXIT_DEADLINE, "the marker removed").await, 0);

    let captured = capture.snapshot();
    let stale = lines_starting(&captured, HELD_STALE_PREFIX);
    assert_eq!(stale.len(), 1, "one stale WARN:\n{}", captured.text);
    assert_eq!(stale[0]["fields"]["incident"], INCIDENT);
    assert_eq!(
        lines_starting(&captured, HELD_STALE_CLEARED_PREFIX).len(),
        1,
        "one cleared INFO:\n{}",
        captured.text
    );
    run.returned(SETTLE, "after the exit seam returned")
        .await
        .expect("held mode stops cleanly");
    drop(channel);
    drop(client);
    drop(capture);
    purge_keys(&tv).await;
}

/// `Metrics::set_dr_held` sets `live_edit_dr_held` alone: no hold entry is
/// counted under any reason.
#[test]
fn set_dr_held_sets_the_gauge_and_counts_no_hold_entry() {
    let metrics = crate::common::metrics_scrape_only("held-mode-gauge");
    assert_eq!(gauge(&metrics, "live_edit_dr_held"), 0, "0 at construction");

    metrics.set_dr_held(true);
    let text = render(&metrics);
    assert_eq!(sample_value(&text, "live_edit_dr_held", &[]), Some(1.0));
    assert!(
        hold_entries(&text).iter().all(|(_, n)| *n == 0),
        "the entered counter is not ticked"
    );

    metrics.set_dr_held(false);
    assert_eq!(gauge(&metrics, "live_edit_dr_held"), 0);
}
