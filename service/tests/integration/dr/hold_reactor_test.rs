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

//! What a replica does with each hold verdict and with its age
//! (`live_edit::grpc::hold`; SDD SR-6.1–6.5, SR-7, SR-7a): the hold reactor,
//! the hold and stale-verdict entry routines, the writer abort set, the
//! LiveService gate and the two signals the reactor publishes, driven through
//! a verdict channel the test owns. No store is involved: the poll that feeds
//! the channel in production is `hold_poll_test.rs`'s, and the same reactions
//! behind a real poll and a real server are `hold_entry_test.rs`'s.
//!
//! - **Before the serving point** a `Held` verdict — the startup observation
//!   or a later one — makes the reactor call the exit seam with 0 at once,
//!   with no jitter, and run nothing of the entry routine: the gate stays
//!   open, nothing is aborted, `live_edit_dr_held` stays 0. A verdict
//!   [`STALE_VERDICT_THRESHOLD`] old there exits non-zero at once.
//! - **On a serving replica** a `Held` verdict runs the entry routine once:
//!   the gate closes, a stream enrolling afterwards is born ended with the
//!   hold's cause, every registered writer is aborted, `live_edit_dr_held` is
//!   1 and `live_edit_dr_hold_entered_total{reason}` ticks once under the
//!   bounded label, and the `held` signal is set. Neither `begun` nor the
//!   shutdown parent is touched. A replaced incident while held enters
//!   nothing again.
//! - **Un-hold.** `Serving` after the entry clears the `held` signal and
//!   calls the exit seam with 0 after a wait in `[0, 5 s]`; the gate never
//!   reopens.
//! - **A stale verdict (SR-6.5).** A serving replica that is not held, with
//!   no publication for [`STALE_VERDICT_THRESHOLD`], runs the stale-verdict
//!   entry: the gate refuses `UPSTREAM_UNAVAILABLE`, never `SERVICE_HELD`,
//!   every enrolled and later stream ends with the cause `StaleVerdict`, the
//!   writers are aborted, `live_edit_dr_hold_verdict_stale` is 1 and the
//!   readiness signal is set; it is no hold and no shutdown. Its next
//!   publication, whatever it says, exits 0 after the jitter. A held replica
//!   whose verdict goes stale stays held (defer-7): no second stream ending,
//!   so a late stream still carries `Held`, and only the readiness signal and
//!   the gauge record it; a later `Held` publication clears them and a
//!   `Serving` one exits 0 after the jitter.
//! - **No exit because reads fail.** Once stale, held or serving, a replica
//!   waits on no clock: an hour with no read exits nothing, and its next
//!   read still exits 0 after the jitter.
//! - **A closed verdict channel** (the poll ended without the shutdown
//!   token) exits [`NO_VERDICT_EXIT_CODE`] at once with one ERROR, before or
//!   after the serving point, held or serving, fresh or stale; after the
//!   shutdown token it exits nothing.
//! - **A failed reactor task** exits [`NO_VERDICT_EXIT_CODE`] through the
//!   watcher `spawn_hold_reactor` returns, with one ERROR; after the shutdown
//!   token it exits nothing, and the returned handle still resolves only once
//!   the reactor has ended.
//! - **Shutdown is read from the shutdown token only (defer-1).** SIGTERM
//!   while held stops the reactor and its heartbeat, and a later `Serving`
//!   exits nothing; SIGTERM during the exit wait cancels the exit; a
//!   shutdown that outlasts the stale threshold enters no stale verdict, so
//!   its streams still end `server shutting down`.
//! - **The heartbeat** repeats one WARN per interval while held, naming the
//!   latest incident (interval shortened through `HoldReactorSettings`).
//! - **The writer inventory** is the spec's list, in one place, and both
//!   entry routines stop every entry of it; the abort set reports what is not
//!   registered, and the serving point logs it.
//! - **The gate** refuses a hold with the incident of the latest verdict,
//!   never a copy taken at entry, and a stale verdict `UPSTREAM_UNAVAILABLE`;
//!   a stale closing never replaces a hold's.
//!
//! # Clock and capture
//!
//! Every reactor test runs on a current-thread runtime with tokio's clock
//! paused, so the jitter, the heartbeat and the 30 s clock are
//! measured on virtual time — no assertion depends on how fast the runner
//! is — and a [`ThreadCapture`] sees the lines the reactor's spawned tasks
//! log. The 30 s clock is the production constant (it is not injectable;
//! `dr::hold::verdict`'s tests pin its value), reached on
//! the paused clock in no wall time. Every wait is a virtual deadline
//! ([`SETTLE`]); a paused clock reaches it only when nothing else can run, so
//! a healthy test never pays it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use futures::future::BoxFuture;
use live_edit::dr::hold::{
    HoldObservation, HoldWriter, HOLD_WRITERS, STALE_VERDICT_THRESHOLD, UNHOLD_EXIT_JITTER_MAX,
};
use live_edit::grpc::graceful_shutdown::{GracefulShutdown, HoldCause};
use live_edit::grpc::hold::{
    enter_hold, enter_stale_verdict, spawn_hold_reactor, HoldEntryParts, HoldExit, HoldGate,
    HoldGateState, HoldReactorSettings, HoldServingPoint, HoldWriterAborts,
    HOLD_HEARTBEAT_INTERVAL, NO_VERDICT_EXIT_CODE,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    dr_hold_reason_label, Metrics, DR_HOLD_REASONS, DR_HOLD_REASON_KEYSPACE_LOST,
    DR_HOLD_REASON_MAPPING_LOST, DR_HOLD_REASON_OTHER, DR_HOLD_REASON_RETIRED_OBJECT_MAPPED,
    DR_HOLD_REASON_SESSION_LOST, DR_HOLD_REASON_SLOT_LOST, DR_HOLD_REASON_SUCCESSOR_LOST,
};

use super::support::{self, held, lines_starting, serving, RecordingExit};
use crate::common::refusal_detail::{
    assert_service_held, assert_shutdown_status, assert_stale_verdict,
};
use crate::grpc::log_capture::ThreadCapture;

/// Virtual-time bound on every wait. Reached only if the awaited event never
/// comes.
const SETTLE: Duration = Duration::from_secs(60);

/// The heartbeat interval the reactor tests run with: short, so several beats
/// fit in a test, and far from every other interval here.
const TEST_HEARTBEAT: Duration = Duration::from_secs(7);

/// The incident the first hold names, in the shape of SDD SR-4.1's ids.
const INCIDENT: &str = "20261015t0347-b1c2";

/// The incident a second marker names while the replica is held.
const REPLACED_INCIDENT: &str = "20261015t0412-d3e4";

/// The reason every hold here names (`support::held`'s): one of the six, so
/// its label is itself.
const REASON: &str = DR_HOLD_REASON_KEYSPACE_LOST;

/// The entry routine's ERROR line, matched on its fixed start.
const ENTRY_ERROR_PREFIX: &str = "the fleet is held for disaster recovery: this replica refuses";

/// The heartbeat's WARN line, matched on its fixed start.
const HEARTBEAT_PREFIX: &str = "this replica is still held for disaster recovery";

/// The un-hold INFO line, matched on its fixed start.
const UNHOLD_PREFIX: &str = "the disaster-recovery hold cleared";

/// The WARN of a replica held before its serving point.
const HELD_BEFORE_SERVING_PREFIX: &str =
    "the fleet is held for disaster recovery and this replica has not started serving";

/// The stale-verdict entry routine's WARN, on a serving replica that is not
/// held.
const STALE_ENTRY_PREFIX: &str = "this replica has had no successful disaster-recovery hold read";

/// The WARN of a held replica whose verdict goes stale.
const HELD_STALE_PREFIX: &str =
    "this held replica has had no successful disaster-recovery hold read";

/// The INFO of a stale replica whose read succeeded again, before its exit.
const STALE_RECOVERED_PREFIX: &str =
    "the disaster-recovery hold read succeeded again after this replica's verdict went stale";

/// The INFO of a held replica whose stale verdict is fresh again.
const HELD_STALE_CLEARED_PREFIX: &str =
    "the disaster-recovery hold read succeeded again; this held replica's verdict is no longer \
     stale";

/// The ERROR of a replica whose verdict went stale before its serving point.
const STALE_BEFORE_SERVING_PREFIX: &str =
    "no successful disaster-recovery hold read for 30 s before this replica started serving";

/// The ERROR of a reactor whose verdict channel closed before shutdown.
const CHANNEL_CLOSED_PREFIX: &str = "hold: the hold poll ended before shutdown";

/// The ERROR of the watcher whose reactor task failed before shutdown.
const REACTOR_FAILED_PREFIX: &str = "hold: the hold reactor failed";

/// A span of virtual time far past the 120 s no-read exit an older build
/// had: a replica that cannot read waits on no clock through all of it.
const LONG_OUTAGE: Duration = Duration::from_secs(3600);

/// One step of virtual time: a reaction due at an instant has happened one
/// step after it and not one step before.
const STEP: Duration = Duration::from_millis(1);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A running reactor, the parts its entry routine acts on, the channel and
/// tokens the test drives it through, and receivers of the two signals it
/// publishes.
struct Reactor {
    verdicts: watch::Sender<HoldObservation>,
    point: HoldServingPoint,
    parts: HoldEntryParts,
    shutdown: CancellationToken,
    exits: mpsc::UnboundedReceiver<(i32, Instant)>,
    task: JoinHandle<()>,
    /// `HoldReactorSettings::verdict_stale`: the readiness input.
    verdict_stale: watch::Receiver<bool>,
    /// `HoldReactorSettings::held`: the scaling snapshot's `held`.
    held: watch::Receiver<bool>,
}

impl Reactor {
    /// A reactor whose channel starts at `initial`, its serving point reached
    /// with fresh parts when `reached`.
    fn start(initial: HoldObservation, reached: bool) -> Self {
        let (calls, exits) = mpsc::unbounded_channel();
        Self::start_with_exit(initial, reached, Arc::new(RecordingExit { calls }), exits)
    }

    /// [`Self::start`] with `exit` as the exit seam, and `exits` as the
    /// receiver of what it records.
    fn start_with_exit(
        initial: HoldObservation,
        reached: bool,
        exit: Arc<dyn HoldExit>,
        exits: mpsc::UnboundedReceiver<(i32, Instant)>,
    ) -> Self {
        let (verdicts, rx) = watch::channel(initial);
        let parts = HoldEntryParts {
            gate: HoldGate::new(rx.clone()),
            streams: GracefulShutdown::new(),
            writers: Arc::new(HoldWriterAborts::new()),
            metrics: Arc::new(crate::common::metrics_scrape_only("hold-reactor")),
        };
        let point = HoldServingPoint::new();
        if reached {
            assert!(
                point.reach(parts.clone()),
                "a fresh serving point is reached once"
            );
        }
        let shutdown = CancellationToken::new();
        let stale_tx = watch::Sender::new(false);
        let held_tx = watch::Sender::new(false);
        let verdict_stale = stale_tx.subscribe();
        let held = held_tx.subscribe();
        let task = spawn_hold_reactor(
            rx,
            point.clone(),
            shutdown.clone(),
            HoldReactorSettings {
                heartbeat_interval: TEST_HEARTBEAT,
                exit,
                verdict_stale: stale_tx,
                held: held_tx,
            },
        );
        Self {
            verdicts,
            point,
            parts,
            shutdown,
            exits,
            task,
            verdict_stale,
            held,
        }
    }

    /// The readiness input the reactor publishes now.
    fn is_verdict_stale(&self) -> bool {
        *self.verdict_stale.borrow()
    }

    /// The scaling snapshot's `held` the reactor publishes now.
    fn is_held_signal(&self) -> bool {
        *self.held.borrow()
    }

    /// Publish `observation`, as a successful poll does.
    fn publish(&self, observation: HoldObservation) {
        self.verdicts.send_replace(observation);
    }

    /// The next exit-seam call, within [`SETTLE`] of virtual time.
    async fn next_exit(&mut self, what: &str) -> (i32, Instant) {
        tokio::time::timeout(SETTLE, self.exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: the exit seam was not called within {SETTLE:?}"))
            .unwrap_or_else(|| panic!("{what}: the reactor dropped its exit seam uncalled"))
    }

    /// Assert the exit seam is not called while `span` of virtual time passes
    /// (or before the reactor returns, dropping the seam uncalled).
    async fn assert_no_exit_for(&mut self, span: Duration, what: &str) {
        if let Ok(Some(call)) = tokio::time::timeout(span, self.exits.recv()).await {
            panic!("{what}: the exit seam must not be called; got {call:?}");
        }
    }

    /// Wait until the gate is closed: the entry routine has run.
    async fn until_held(&self) {
        wait_until("the gate to close", || {
            self.parts.gate.state() == HoldGateState::Held
        })
        .await;
    }

    /// Wait for the reactor task to return.
    async fn stopped(self, what: &str) {
        tokio::time::timeout(SETTLE, self.task)
            .await
            .unwrap_or_else(|_| panic!("{what}: the reactor did not stop within {SETTLE:?}"))
            .unwrap_or_else(|e| panic!("{what}: the reactor task failed: {e}"));
    }
}

/// Poll `cond` until it holds, within [`SETTLE`] of virtual time.
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + SETTLE;
    while !cond() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

fn dr_held(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_dr_held", &[])
        .expect("live_edit_dr_held is recorded at construction") as u64
}

fn hold_entries(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_dr_hold_entered_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("the reason={reason} cell is materialised at construction"))
        as u64
}

fn all_hold_entries(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    support::hold_entries(&render(metrics))
}

fn verdict_stale_gauge(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_dr_hold_verdict_stale", &[])
        .expect("live_edit_dr_hold_verdict_stale is recorded at construction") as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// Before the serving point (PM B1)
// ═════════════════════════════════════════════════════════════════════════════

/// A replica whose hold poll starts out `Held` — the marker appeared between
/// the startup evaluation and the poll's first publication — exits 0 at once
/// through the seam, and its entry routine never runs.
///
/// Before the serving point the reactor holds no entry parts at all (they
/// are handed over by `HoldServingPoint::reach`), so the evidence that the
/// routine did not run is its ERROR line's absence and the point still
/// unreached, not the state of parts it never had.
#[tokio::test(start_paused = true)]
async fn reactor_held_before_the_serving_point_exits_zero_at_once_without_entering_the_hold() {
    let capture = ThreadCapture::start();
    let started = Instant::now();
    let mut reactor = Reactor::start(held(INCIDENT), false);

    let (code, at) = reactor.next_exit("held before the serving point").await;

    assert_eq!(
        code, 0,
        "a replica held before serving exits 0, to restart held"
    );
    assert_eq!(
        at - started,
        Duration::ZERO,
        "the exit is at once: no jitter, no wait of any kind before the serving point"
    );
    assert!(!reactor.point.is_reached());
    let captured = capture.snapshot();
    assert_eq!(
        lines_starting(&captured, HELD_BEFORE_SERVING_PREFIX).len(),
        1,
        "one WARN names the incident before the exit; capture:\n{}",
        captured.text
    );
    assert!(
        lines_starting(&captured, ENTRY_ERROR_PREFIX).is_empty(),
        "the entry routine's ERROR must not be logged before the serving point"
    );
    reactor.stopped("after the exit").await;
}

/// The same when the marker appears while the replica is still starting: a
/// `Serving` start, then `Held` before the serving point is reached. The
/// entry routine's ERROR is not logged and the point stays unreached.
#[tokio::test(start_paused = true)]
async fn reactor_hold_arriving_before_the_serving_point_exits_zero_at_once() {
    let capture = ThreadCapture::start();
    let mut reactor = Reactor::start(serving(), false);
    reactor
        .assert_no_exit_for(Duration::from_secs(10), "serving, before the serving point")
        .await;

    let published = Instant::now();
    reactor.publish(held(INCIDENT));
    let (code, at) = reactor.next_exit("hold arriving before serving").await;

    assert_eq!(code, 0);
    assert_eq!(at - published, Duration::ZERO, "the exit is at once");
    assert!(!reactor.point.is_reached());
    assert!(
        lines_starting(&capture.snapshot(), ENTRY_ERROR_PREFIX).is_empty(),
        "the entry routine's ERROR must not be logged before the serving point"
    );
    reactor.stopped("after the exit").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Hold entry on a serving replica
// ═════════════════════════════════════════════════════════════════════════════

/// `Held` on a serving replica runs the entry routine once: the gate closes,
/// later enrollments are born ended with the hold's cause, the registered
/// writers are aborted, the two signals are recorded and one ERROR names the
/// incident, reason and evidence. It is not a shutdown, and it does not exit.
/// A second `Held` verdict naming another incident enters nothing again.
#[tokio::test(start_paused = true)]
async fn reactor_held_on_a_serving_replica_runs_the_entry_routine_once() {
    let capture = ThreadCapture::start();
    let mut reactor = Reactor::start(serving(), true);
    let token = CancellationToken::new();
    reactor
        .parts
        .writers
        .register_token(HoldWriter::CheckpointLoop, token.clone());
    assert_eq!(dr_held(&reactor.parts.metrics), 0, "0 before any hold");

    reactor.publish(held(INCIDENT));
    reactor.until_held().await;

    // The gate, the streams and the writers.
    let refusal = reactor.parts.gate.refusal().expect("a closed gate refuses");
    assert_service_held(&refusal, Some(INCIDENT));
    let (late, _enrollment) = reactor.parts.streams.enroll_stream();
    assert!(
        late.is_cancelled(),
        "a stream enrolling after entry is born ended"
    );
    assert_eq!(
        late.hold_cause(),
        Some(&HoldCause::Held {
            incident: Some(INCIDENT.to_owned())
        }),
        "and carries the hold's cause and incident"
    );
    assert!(reactor.parts.writers.is_aborted());
    assert!(
        token.is_cancelled(),
        "a registered writer's token is cancelled"
    );

    // The signals.
    assert_eq!(dr_held(&reactor.parts.metrics), 1);
    assert_eq!(
        all_hold_entries(&reactor.parts.metrics),
        DR_HOLD_REASONS
            .iter()
            .map(|r| (*r, u64::from(*r == REASON)))
            .collect::<Vec<_>>(),
        "one entry, under the marker's reason"
    );
    let captured = capture.snapshot();
    let entry = lines_starting(&captured, ENTRY_ERROR_PREFIX);
    assert_eq!(
        entry.len(),
        1,
        "one ERROR at entry; capture:\n{}",
        captured.text
    );
    assert_eq!(entry[0]["level"], "ERROR");
    assert_eq!(entry[0]["fields"]["incident"], INCIDENT);
    assert_eq!(entry[0]["fields"]["reason"], REASON);
    assert_eq!(
        entry[0]["fields"]["evidence"],
        "le:counter:object_id absent on an established store"
    );

    // Not a shutdown.
    assert!(!reactor.parts.streams.begun().is_cancelled());
    assert!(!reactor.parts.streams.is_cancelled());
    assert!(!reactor.shutdown.is_cancelled());

    // A replaced incident is still the same hold.
    reactor.publish(held(REPLACED_INCIDENT));
    reactor
        .assert_no_exit_for(Duration::from_secs(1), "a replaced incident")
        .await;
    assert_eq!(
        hold_entries(&reactor.parts.metrics, REASON),
        1,
        "a held replica enters once"
    );
    assert_eq!(
        lines_starting(&capture.snapshot(), ENTRY_ERROR_PREFIX).len(),
        1,
        "and logs its entry once"
    );
    assert_service_held(
        &reactor.parts.gate.refusal().expect("still closed"),
        Some(REPLACED_INCIDENT),
    );

    reactor.shutdown.cancel();
    reactor.stopped("after shutdown").await;
}

/// A serving replica that keeps reading `Serving` does nothing: no exit, the
/// gate open, nothing recorded.
#[tokio::test(start_paused = true)]
async fn reactor_serving_verdicts_on_a_serving_replica_change_nothing() {
    let mut reactor = Reactor::start(serving(), true);
    for _ in 0..5 {
        reactor.publish(serving());
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    reactor
        .assert_no_exit_for(Duration::from_secs(10), "serving throughout")
        .await;
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Open);
    assert!(!reactor.parts.writers.is_aborted());
    assert_eq!(dr_held(&reactor.parts.metrics), 0);
    reactor.shutdown.cancel();
    reactor.stopped("after shutdown").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Un-hold (SDD SR-6.4)
// ═════════════════════════════════════════════════════════════════════════════

/// Runs of the un-hold exit in [`reactor_serving_after_a_hold_exits_zero_within_the_jitter_bound_and_never_reopens`]:
/// enough that a jitter that is always 0, or always the same, cannot pass.
const UNHOLD_RUNS: usize = 32;

/// `Serving` after a hold entry calls the exit seam with 0 after a wait in
/// `[0, UNHOLD_EXIT_JITTER_MAX]`, and the gate never reopens meanwhile or
/// after. Over [`UNHOLD_RUNS`] runs the wait is not always the same, so it is
/// drawn, not fixed.
#[tokio::test(start_paused = true)]
async fn reactor_serving_after_a_hold_exits_zero_within_the_jitter_bound_and_never_reopens() {
    let mut waits = Vec::with_capacity(UNHOLD_RUNS);
    for run in 0..UNHOLD_RUNS {
        let mut reactor = Reactor::start(serving(), true);
        reactor.publish(held(INCIDENT));
        reactor.until_held().await;

        let cleared = Instant::now();
        reactor.publish(serving());
        let (code, at) = reactor.next_exit("un-hold").await;

        assert_eq!(code, 0, "run {run}: an un-held replica exits 0");
        let waited = at - cleared;
        assert!(
            waited <= UNHOLD_EXIT_JITTER_MAX,
            "run {run}: the exit waited {waited:?}, past the {UNHOLD_EXIT_JITTER_MAX:?} jitter bound"
        );
        assert_eq!(
            reactor.parts.gate.state(),
            HoldGateState::Held,
            "run {run}: the gate never reopens in place"
        );
        reactor.stopped("after the un-hold exit").await;
        waits.push(waited);
    }
    assert!(
        waits.iter().any(|w| *w > Duration::ZERO),
        "the un-hold exit must wait a jitter, not exit at once every time: {waits:?}"
    );
    let first = waits[0];
    assert!(
        waits.iter().any(|w| *w != first),
        "the jitter must be drawn, not fixed: {waits:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Shutdown, and what is not a shutdown (defer-1)
// ═════════════════════════════════════════════════════════════════════════════

/// SIGTERM while held (the shutdown token `main.rs` cancels first thing)
/// stops the reactor and its heartbeat, so the normal paced shutdown runs
/// alone: a later `Serving` verdict exits nothing, no heartbeat follows, and
/// the hold left `begun` and the shutdown parent for that shutdown to cancel.
#[tokio::test(start_paused = true)]
async fn reactor_sigterm_while_held_stops_the_reactor_and_its_heartbeat_and_exits_nothing() {
    let capture = ThreadCapture::start();
    let mut reactor = Reactor::start(serving(), true);
    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    tokio::time::sleep(TEST_HEARTBEAT + Duration::from_millis(1)).await;
    let beats_before = lines_starting(&capture.snapshot(), HEARTBEAT_PREFIX).len();
    assert_eq!(beats_before, 1, "precondition: the heartbeat is running");

    reactor.shutdown.cancel();
    reactor.publish(serving());
    reactor
        .assert_no_exit_for(
            UNHOLD_EXIT_JITTER_MAX * 3,
            "a Serving verdict after SIGTERM",
        )
        .await;
    tokio::time::sleep(TEST_HEARTBEAT * 3).await;
    assert_eq!(
        lines_starting(&capture.snapshot(), HEARTBEAT_PREFIX).len(),
        beats_before,
        "the heartbeat stops with the shutdown token"
    );
    assert!(
        !reactor.parts.streams.begun().is_cancelled() && !reactor.parts.streams.is_cancelled(),
        "the hold left the shutdown for SIGTERM's paced ending to start"
    );
    let streams = reactor.parts.streams.clone();
    reactor.stopped("after SIGTERM").await;
    streams.cancel();
    assert!(
        streams.begun().is_cancelled(),
        "the normal shutdown still begins after a hold"
    );
}

/// SIGTERM during the un-hold exit's jitter cancels the exit: shutdown wins.
/// A run whose jitter came out at or below the test's 1 ms probe step can
/// exit before the cancel lands, so the test retries until a run's jitter
/// (read off its INFO line) is above that.
#[tokio::test(start_paused = true)]
async fn reactor_sigterm_during_the_unhold_jitter_cancels_the_exit() {
    let capture = ThreadCapture::start();
    for _ in 0..8 {
        let mut reactor = Reactor::start(serving(), true);
        reactor.publish(held(INCIDENT));
        reactor.until_held().await;
        let unholds_before = lines_starting(&capture.snapshot(), UNHOLD_PREFIX).len();
        reactor.publish(serving());
        wait_until("the un-hold INFO line", || {
            lines_starting(&capture.snapshot(), UNHOLD_PREFIX).len() > unholds_before
        })
        .await;
        let captured = capture.snapshot();
        let line = *lines_starting(&captured, UNHOLD_PREFIX)
            .last()
            .expect("just waited for it");
        let jitter_ms = line["fields"]["jitter_ms"]
            .as_u64()
            .expect("the un-hold line carries its jitter");
        if jitter_ms <= 2 {
            // Exits about when the probe wakes: the cancel may not land
            // first, so this run shows nothing either way.
            continue;
        }
        reactor.shutdown.cancel();
        reactor
            .assert_no_exit_for(UNHOLD_EXIT_JITTER_MAX * 2, "SIGTERM during the jitter")
            .await;
        reactor.stopped("after SIGTERM during the jitter").await;
        return;
    }
    panic!("eight un-hold runs in a row drew a jitter of at most 2 ms; it is not being drawn");
}

/// A shutdown that outlasts the stale threshold is never read as a stale
/// verdict (defer-1): once the shutdown token is cancelled the reactor acts
/// on nothing, and the poll stops on the same token, so its verdict only
/// ages. On a serving replica no stale entry follows — the gate stays open,
/// no writer is aborted, no signal is set — and the stream still enrolled
/// when the paced ending finishes ends `server shutting down`, never
/// `UPSTREAM_UNAVAILABLE`. On a held replica no stale signal and no
/// non-zero exit follow either.
#[tokio::test(start_paused = true)]
async fn reactor_shutdown_that_outlasts_the_stale_threshold_enters_no_stale_verdict() {
    for held_first in [false, true] {
        let case = if held_first { "held" } else { "serving" };
        let mut reactor = Reactor::start(serving(), true);
        let last_read = Instant::now();
        if held_first {
            reactor.publish(held(INCIDENT));
            reactor.until_held().await;
        }
        let (stream, _enrollment) = reactor.parts.streams.enroll_stream();
        let writer = CancellationToken::new();
        reactor
            .parts
            .writers
            .register_token(HoldWriter::CheckpointLoop, writer.clone());

        // SIGTERM 1 s in; the paced ending runs long past the stale threshold.
        tokio::time::sleep(Duration::from_secs(1)).await;
        reactor.shutdown.cancel();
        tokio::time::sleep_until(last_read + Duration::from_secs(150)).await;
        assert!(
            reactor.exits.try_recv().is_err(),
            "{case}: a shutdown past the stale threshold exits nothing"
        );
        assert!(
            !reactor.is_verdict_stale(),
            "{case}: a shutdown is not a stale verdict: readiness is not touched"
        );
        assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 0, "{case}");
        if held_first {
            assert_eq!(reactor.parts.gate.state(), HoldGateState::Held, "{case}");
        } else {
            assert_eq!(
                reactor.parts.gate.state(),
                HoldGateState::Open,
                "{case}: no stale entry closes the gate"
            );
            assert!(
                !writer.is_cancelled() && !reactor.parts.writers.is_aborted(),
                "{case}: no stale entry aborts the writers; the shutdown stops them"
            );
            assert!(
                !stream.is_cancelled(),
                "{case}: no stale entry ends the streams"
            );
            // The paced ending's last step ends what is still enrolled.
            reactor.parts.streams.cancel();
            assert!(stream.is_cancelled());
            assert_eq!(stream.hold_cause(), None, "{case}");
            assert_shutdown_status(&stream.ending_status());
        }
        reactor.stopped(&format!("{case}: after SIGTERM")).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A stale verdict (SDD SR-6.5)
// ═════════════════════════════════════════════════════════════════════════════

/// A serving replica that is not held, with no publication for the stale
/// threshold, fails closed on its own, exactly at the threshold: the gate
/// refuses `UNAVAILABLE` `UPSTREAM_UNAVAILABLE` (never `SERVICE_HELD`), the
/// stream enrolled before ends with the cause `StaleVerdict` and its status,
/// a stream enrolling after is born ended with it, the registered writer is
/// aborted, `live_edit_dr_hold_verdict_stale` is 1 and the readiness signal
/// is set, with one WARN naming the aborted writers. It is no hold — the
/// `held` signal, `live_edit_dr_held` and the hold entries stay as they were
/// — and no shutdown, and it exits nothing.
#[tokio::test(start_paused = true)]
async fn reactor_serving_replica_without_a_read_for_the_stale_threshold_fails_closed_upstream_unavailable(
) {
    let capture = ThreadCapture::start();
    let last_read = Instant::now();
    let mut reactor = Reactor::start(serving(), true);
    let writer = CancellationToken::new();
    reactor
        .parts
        .writers
        .register_token(HoldWriter::CheckpointLoop, writer.clone());
    let (open_stream, _enrollment) = reactor.parts.streams.enroll_stream();
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 0, "0 before");

    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD - STEP).await;
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Open);
    assert!(!reactor.is_verdict_stale(), "fresh until the threshold");
    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;

    // The gate.
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Stale);
    assert_stale_verdict(&reactor.parts.gate.refusal().expect("a closed gate refuses"));
    // The streams.
    // The status first: a cause that maps to SERVICE_HELD fails here.
    assert!(open_stream.is_cancelled(), "an open stream is ended");
    assert_stale_verdict(&open_stream.ending_status());
    assert_eq!(open_stream.hold_cause(), Some(&HoldCause::StaleVerdict));
    let (late, _late_enrollment) = reactor.parts.streams.enroll_stream();
    assert!(late.is_cancelled(), "a later stream is born ended");
    assert_stale_verdict(&late.ending_status());
    assert_eq!(late.hold_cause(), Some(&HoldCause::StaleVerdict));
    // The writers.
    assert!(reactor.parts.writers.is_aborted());
    assert!(
        writer.is_cancelled(),
        "a registered writer's token is cancelled"
    );
    // The signals.
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 1);
    assert!(reactor.is_verdict_stale(), "readiness goes NOT_SERVING");
    assert!(!reactor.is_held_signal(), "a stale verdict is not a hold");
    assert_eq!(dr_held(&reactor.parts.metrics), 0, "not a hold");
    assert!(
        all_hold_entries(&reactor.parts.metrics)
            .iter()
            .all(|(_, n)| *n == 0),
        "no hold entry is counted"
    );
    // Not a shutdown.
    assert!(!reactor.parts.streams.begun().is_cancelled());
    assert!(!reactor.parts.streams.is_cancelled());
    assert!(!reactor.shutdown.is_cancelled());
    // One WARN.
    let captured = capture.snapshot();
    let warns = lines_starting(&captured, STALE_ENTRY_PREFIX);
    assert_eq!(
        warns.len(),
        1,
        "one WARN at stale entry:\n{}",
        captured.text
    );
    assert_eq!(warns[0]["level"], "WARN");
    assert!(
        warns[0]["fields"]["aborted_writers"]
            .as_str()
            .is_some_and(|w| w.contains("checkpoint_loop")),
        "the WARN names the writers it aborted: {}",
        warns[0]
    );
    assert!(lines_starting(&captured, ENTRY_ERROR_PREFIX).is_empty());

    reactor
        .assert_no_exit_for(Duration::from_secs(10), "stale, no read yet")
        .await;
    reactor.shutdown.cancel();
    reactor.stopped("after shutdown").await;
}

/// After a stale entry, the next publication — whatever it says — exits 0
/// after a wait within the jitter bound, so the replica restarts into normal
/// service, or held if the fleet is held. The gate never reopens in place,
/// and a `Held` publication runs no hold entry first.
#[tokio::test(start_paused = true)]
async fn reactor_stale_serving_replica_exits_zero_within_the_jitter_on_its_next_read() {
    let capture = ThreadCapture::start();
    fn held_incident() -> HoldObservation {
        held(INCIDENT)
    }
    for (case, next) in [
        ("Serving", serving as fn() -> HoldObservation),
        ("Held", held_incident),
    ] {
        let mut reactor = Reactor::start(serving(), true);
        let last_read = Instant::now();
        tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
        assert_eq!(
            reactor.parts.gate.state(),
            HoldGateState::Stale,
            "{case}: precondition"
        );
        let recovered_before = lines_starting(&capture.snapshot(), STALE_RECOVERED_PREFIX).len();

        let read = Instant::now();
        reactor.publish(next());
        let (code, at) = reactor.next_exit(case).await;

        assert_eq!(
            code, 0,
            "{case}: a stale replica whose read succeeds exits 0"
        );
        assert!(
            at - read <= UNHOLD_EXIT_JITTER_MAX,
            "{case}: the exit waited {:?}, past the jitter bound",
            at - read
        );
        assert_eq!(
            reactor.parts.gate.state(),
            HoldGateState::Stale,
            "{case}: the gate never reopens in place"
        );
        assert_eq!(
            dr_held(&reactor.parts.metrics),
            0,
            "{case}: no hold is entered on the way out"
        );
        let captured = capture.snapshot();
        let recovered = lines_starting(&captured, STALE_RECOVERED_PREFIX);
        assert_eq!(recovered.len(), recovered_before + 1, "{case}");
        assert!(
            recovered
                .last()
                .expect("just counted")
                .get("fields")
                .and_then(|f| f.get("jitter_ms"))
                .is_some_and(serde_json::Value::is_u64),
            "{case}: the INFO carries the jitter"
        );
        reactor.stopped(&format!("{case}: after the exit")).await;
    }
    assert!(
        lines_starting(&capture.snapshot(), ENTRY_ERROR_PREFIX).is_empty(),
        "a stale replica's Held read exits; it does not enter the hold first"
    );
}

/// A held replica whose verdict goes stale stays held (PM N8, defer-7): the
/// gate keeps refusing `SERVICE_HELD`, no second stream ending runs, so a
/// stream enrolling after the stale threshold is still born with the hold's
/// cause and ends `SERVICE_HELD`, never `UPSTREAM_UNAVAILABLE`. Only its
/// staleness is recorded: the readiness signal, the gauge and one WARN
/// naming the incident. A later `Held` publication clears both signals and
/// exits nothing; the replica stays held.
#[tokio::test(start_paused = true)]
async fn reactor_held_replica_going_stale_stays_held_with_no_second_stream_ending() {
    let capture = ThreadCapture::start();
    let mut reactor = Reactor::start(serving(), true);
    let (before, _enrollment) = reactor.parts.streams.enroll_stream();
    let held_read = Instant::now();
    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    assert!(reactor.is_held_signal(), "the held signal is set at entry");
    assert_eq!(
        before.hold_cause(),
        Some(&HoldCause::Held {
            incident: Some(INCIDENT.to_owned())
        })
    );

    tokio::time::sleep_until(held_read + STALE_VERDICT_THRESHOLD - STEP).await;
    assert!(!reactor.is_verdict_stale());
    tokio::time::sleep_until(held_read + STALE_VERDICT_THRESHOLD + STEP).await;

    assert!(reactor.is_verdict_stale(), "readiness goes NOT_SERVING");
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 1);
    assert!(
        reactor.is_held_signal(),
        "a stale verdict leaves held as it is"
    );
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Held);
    assert_service_held(
        &reactor.parts.gate.refusal().expect("still closed"),
        Some(INCIDENT),
    );
    let (late, _late_enrollment) = reactor.parts.streams.enroll_stream();
    assert!(late.is_cancelled());
    assert_eq!(
        late.hold_cause(),
        Some(&HoldCause::Held {
            incident: Some(INCIDENT.to_owned())
        }),
        "a late stream keeps the hold's cause: no stale ending replaced it"
    );
    assert_service_held(&late.ending_status(), Some(INCIDENT));
    assert_eq!(
        before.hold_cause(),
        Some(&HoldCause::Held {
            incident: Some(INCIDENT.to_owned())
        }),
        "the stream the hold ended keeps its cause"
    );
    let captured = capture.snapshot();
    let warns = lines_starting(&captured, HELD_STALE_PREFIX);
    assert_eq!(warns.len(), 1, "one WARN:\n{}", captured.text);
    assert_eq!(warns[0]["level"], "WARN");
    assert_eq!(warns[0]["fields"]["incident"], INCIDENT);
    assert!(
        lines_starting(&captured, STALE_ENTRY_PREFIX).is_empty(),
        "the stale-verdict entry routine does not run on a held replica"
    );
    assert_eq!(
        hold_entries(&reactor.parts.metrics, REASON),
        1,
        "entered once"
    );

    // A Held read again: both signals clear, the replica stays held.
    tokio::time::sleep(Duration::from_secs(10)).await;
    reactor.publish(held(INCIDENT));
    wait_until("the stale signal to clear", || !reactor.is_verdict_stale()).await;
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 0);
    assert_eq!(
        lines_starting(&capture.snapshot(), HELD_STALE_CLEARED_PREFIX).len(),
        1
    );
    reactor
        .assert_no_exit_for(
            Duration::from_secs(10),
            "a Held read on a held stale replica",
        )
        .await;
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Held);
    assert!(reactor.is_held_signal());

    reactor.shutdown.cancel();
    reactor.stopped("after shutdown").await;
}

/// A held replica whose verdict went stale and whose next read says
/// `Serving` clears both signals and exits 0 after the jitter: the hold
/// cleared while it could not read.
#[tokio::test(start_paused = true)]
async fn reactor_held_stale_replica_exits_zero_within_the_jitter_when_a_read_says_serving() {
    let mut reactor = Reactor::start(serving(), true);
    let held_read = Instant::now();
    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    tokio::time::sleep_until(held_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(reactor.is_verdict_stale(), "precondition: stale while held");

    let read = Instant::now();
    reactor.publish(serving());
    let (code, at) = reactor.next_exit("a Serving read after a stale hold").await;

    assert_eq!(code, 0);
    assert!(at - read <= UNHOLD_EXIT_JITTER_MAX);
    assert!(!reactor.is_verdict_stale(), "the stale signal is cleared");
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 0);
    assert!(
        !reactor.is_held_signal(),
        "the held signal is cleared before the exit"
    );
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Held);
    reactor.stopped("after the exit").await;
}

/// Before the serving point, a verdict that reaches the stale threshold
/// exits non-zero at once (Task 3's rule B1: nothing a stale entry stops
/// exists yet), with one ERROR, and the point stays unreached.
#[tokio::test(start_paused = true)]
async fn reactor_before_the_serving_point_exits_non_zero_at_the_stale_threshold() {
    let capture = ThreadCapture::start();
    let last_read = Instant::now();
    let mut reactor = Reactor::start(serving(), false);

    let (code, at) = reactor.next_exit("stale before the serving point").await;

    assert_eq!(code, NO_VERDICT_EXIT_CODE);
    assert_eq!(at - last_read, STALE_VERDICT_THRESHOLD, "at once, at 30 s");
    assert!(!reactor.point.is_reached());
    let captured = capture.snapshot();
    let errors = lines_starting(&captured, STALE_BEFORE_SERVING_PREFIX);
    assert_eq!(errors.len(), 1, "one ERROR:\n{}", captured.text);
    assert_eq!(errors[0]["level"], "ERROR");
    assert!(lines_starting(&captured, STALE_ENTRY_PREFIX).is_empty());
    reactor.stopped("after the exit").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// No exit because reads fail (Ruling 2)
// ═════════════════════════════════════════════════════════════════════════════

/// A serving replica that goes stale at exactly the threshold waits on no
/// clock after it: an hour with no read exits nothing, and it stays stale
/// (gate `Stale`, readiness signal and gauge set). Its next read then exits
/// 0 within the jitter, as at any time.
#[tokio::test(start_paused = true)]
async fn reactor_stale_serving_replica_exits_nothing_for_an_hour_then_zero_on_its_next_read() {
    let last_read = Instant::now();
    let mut reactor = Reactor::start(serving(), true);
    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD - STEP).await;
    assert_eq!(reactor.parts.gate.state(), HoldGateState::Open);
    tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert_eq!(
        reactor.parts.gate.state(),
        HoldGateState::Stale,
        "stale entry at exactly the threshold"
    );

    reactor
        .assert_no_exit_for(LONG_OUTAGE, "a stale serving replica with no read")
        .await;

    assert_eq!(reactor.parts.gate.state(), HoldGateState::Stale);
    assert!(reactor.is_verdict_stale(), "still NOT_SERVING");
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 1);
    assert!(!reactor.task.is_finished(), "the reactor still runs");

    let read = Instant::now();
    reactor.publish(serving());
    let (code, at) = reactor.next_exit("the first read after an hour").await;
    assert_eq!(code, 0);
    assert!(
        at - read <= UNHOLD_EXIT_JITTER_MAX,
        "the exit waited {:?}, past the jitter bound",
        at - read
    );
    reactor.stopped("after the exit").await;
}

/// A held replica whose verdict goes stale stays held with no clock: an
/// hour with no read exits nothing, the gate still refuses `SERVICE_HELD`,
/// and the readiness signal and gauge stay set. A `Serving` read then clears
/// them and exits 0 within the jitter.
#[tokio::test(start_paused = true)]
async fn reactor_held_stale_replica_exits_nothing_for_an_hour_then_zero_on_a_serving_read() {
    let mut reactor = Reactor::start(serving(), true);
    let held_read = Instant::now();
    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    tokio::time::sleep_until(held_read + STALE_VERDICT_THRESHOLD + STEP).await;
    assert!(reactor.is_verdict_stale(), "precondition: stale while held");

    reactor
        .assert_no_exit_for(LONG_OUTAGE, "a stale held replica with no read")
        .await;

    assert_eq!(reactor.parts.gate.state(), HoldGateState::Held);
    assert_service_held(
        &reactor.parts.gate.refusal().expect("still closed"),
        Some(INCIDENT),
    );
    assert!(reactor.is_verdict_stale());
    assert!(reactor.is_held_signal(), "still held");
    assert_eq!(verdict_stale_gauge(&reactor.parts.metrics), 1);

    let read = Instant::now();
    reactor.publish(serving());
    let (code, at) = reactor.next_exit("a Serving read after an hour").await;
    assert_eq!(code, 0);
    assert!(at - read <= UNHOLD_EXIT_JITTER_MAX);
    assert!(!reactor.is_verdict_stale(), "the stale signal is cleared");
    assert!(!reactor.is_held_signal(), "held is cleared before the exit");
    reactor.stopped("after the exit").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A closed verdict channel, and a failed reactor (PD-D, S-B)
// ═════════════════════════════════════════════════════════════════════════════

/// The hold poll's channel closing without the shutdown token — the poll
/// task ended — exits [`NO_VERDICT_EXIT_CODE`] at the instant it closes, with
/// one ERROR naming whether the replica was held and stale, and the reactor
/// stops. Before the serving point, serving fresh, serving stale, held fresh
/// and held stale alike: a replica that can no longer see the verdict
/// restarts with a new poll, at once rather than on any clock.
#[tokio::test(start_paused = true)]
async fn reactor_closed_verdict_channel_exits_non_zero_at_once_with_one_error() {
    let capture = ThreadCapture::start();
    for (case, reached, hold, go_stale) in [
        ("before the serving point", false, false, false),
        ("serving, fresh", true, false, false),
        ("serving, stale", true, false, true),
        ("held, fresh", true, true, false),
        ("held, stale", true, true, true),
    ] {
        let reactor = Reactor::start(serving(), reached);
        let mut last_read = Instant::now();
        if hold {
            last_read = Instant::now();
            reactor.publish(held(INCIDENT));
            reactor.until_held().await;
        }
        if go_stale {
            tokio::time::sleep_until(last_read + STALE_VERDICT_THRESHOLD + STEP).await;
            assert!(reactor.is_verdict_stale(), "{case}: precondition: stale");
        } else {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        let errors_before = lines_starting(&capture.snapshot(), CHANNEL_CLOSED_PREFIX).len();

        let Reactor {
            verdicts,
            mut exits,
            task,
            ..
        } = reactor;
        let closed = Instant::now();
        drop(verdicts);
        let (code, at) = tokio::time::timeout(SETTLE, exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{case}: a closed channel exited nothing"))
            .unwrap_or_else(|| panic!("{case}: the reactor dropped its exit seam uncalled"));

        assert_eq!(code, NO_VERDICT_EXIT_CODE, "{case}");
        assert_eq!(at, closed, "{case}: at once, on no clock");
        let captured = capture.snapshot();
        let errors = lines_starting(&captured, CHANNEL_CLOSED_PREFIX);
        assert_eq!(
            errors.len(),
            errors_before + 1,
            "{case}: one ERROR:\n{}",
            captured.text
        );
        let error = errors.last().expect("just counted");
        assert_eq!(error["level"], "ERROR", "{case}");
        assert_eq!(error["fields"]["held"], hold, "{case}");
        assert_eq!(error["fields"]["stale"], go_stale, "{case}");
        tokio::time::timeout(SETTLE, task)
            .await
            .unwrap_or_else(|_| panic!("{case}: the reactor did not stop"))
            .unwrap_or_else(|e| panic!("{case}: the reactor task failed: {e}"));
        assert!(exits.try_recv().is_err(), "{case}: it exits once");
    }
}

/// The shutdown token is read before the channel: a channel that closes
/// after it (the poll stops on the same token) exits nothing, then or an
/// hour later.
#[tokio::test(start_paused = true)]
async fn reactor_closed_verdict_channel_after_the_shutdown_token_exits_nothing() {
    for held_first in [false, true] {
        let reactor = Reactor::start(serving(), true);
        if held_first {
            reactor.publish(held(INCIDENT));
            reactor.until_held().await;
        }
        let Reactor {
            verdicts,
            mut exits,
            task,
            shutdown,
            ..
        } = reactor;
        shutdown.cancel();
        drop(verdicts);
        tokio::time::sleep(LONG_OUTAGE).await;
        assert!(
            exits.try_recv().is_err(),
            "held first: {held_first}: a channel closed after the shutdown token exits nothing"
        );
        tokio::time::timeout(SETTLE, task)
            .await
            .expect("the reactor stops on the token")
            .expect("and does not fail");
    }
}

/// An exit seam whose `exit(0)` panics inside the reactor task, so the task
/// fails; every other code is recorded. With `gate`, the panic waits until
/// the gate is notified.
struct FailingExit {
    calls: mpsc::UnboundedSender<(i32, Instant)>,
    gate: Option<Arc<tokio::sync::Notify>>,
}

impl HoldExit for FailingExit {
    fn exit(&self, code: i32) -> BoxFuture<'static, ()> {
        if code != 0 {
            let _ = self.calls.send((code, Instant::now()));
            return Box::pin(async {});
        }
        match &self.gate {
            None => panic!("test exit seam: the reactor task fails at exit(0)"),
            Some(gate) => {
                let gate = Arc::clone(gate);
                Box::pin(async move {
                    gate.notified().await;
                    panic!("test exit seam: the reactor task fails after the shutdown token");
                })
            }
        }
    }
}

/// A normal-mode reactor whose task fails (here it panics inside its
/// `exit(0)`, the held-before-serving exit) makes the replica exit
/// [`NO_VERDICT_EXIT_CODE`] at once, through the same exit seam, with one
/// ERROR naming the failure: a replica whose reactor died would never react
/// to a hold. The handle `spawn_hold_reactor` returns is the watcher's, and
/// it resolves without failing.
#[tokio::test(start_paused = true)]
async fn reactor_task_failing_before_shutdown_exits_non_zero_through_the_watcher() {
    let capture = ThreadCapture::start();
    let (calls, exits) = mpsc::unbounded_channel();
    let started = Instant::now();
    let mut reactor = Reactor::start_with_exit(
        held(INCIDENT),
        false,
        Arc::new(FailingExit { calls, gate: None }),
        exits,
    );

    let (code, at) = reactor.next_exit("a failed reactor").await;

    assert_eq!(code, NO_VERDICT_EXIT_CODE);
    assert_eq!(at, started, "at once");
    let captured = capture.snapshot();
    let errors = lines_starting(&captured, REACTOR_FAILED_PREFIX);
    assert_eq!(errors.len(), 1, "one ERROR:\n{}", captured.text);
    assert_eq!(errors[0]["level"], "ERROR");
    assert!(
        errors[0]["fields"]["error"]
            .as_str()
            .is_some_and(|e| e.contains("panic")),
        "the ERROR names the failure: {}",
        errors[0]
    );
    reactor.stopped("the watcher, after the exit").await;
}

/// After the shutdown token, a failing reactor task exits nothing, and the
/// handle `spawn_hold_reactor` returns resolves only once the reactor task
/// has ended: here the reactor is still inside its exit seam when the token
/// is cancelled, the handle stays pending, and it resolves, without failing,
/// once that task has failed.
#[tokio::test(start_paused = true)]
async fn reactor_task_failing_after_shutdown_exits_nothing_and_the_handle_waits_for_it() {
    let (calls, exits) = mpsc::unbounded_channel();
    let gate = Arc::new(tokio::sync::Notify::new());
    let mut reactor = Reactor::start_with_exit(
        held(INCIDENT),
        false,
        Arc::new(FailingExit {
            calls,
            gate: Some(Arc::clone(&gate)),
        }),
        exits,
    );
    tokio::time::sleep(Duration::from_secs(1)).await;
    reactor.shutdown.cancel();
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert!(
        !reactor.task.is_finished(),
        "the handle resolves only once the reactor task has ended"
    );

    gate.notify_one();
    reactor
        .assert_no_exit_for(Duration::from_secs(10), "a reactor failing after shutdown")
        .await;
    reactor.stopped("after the reactor failed").await;
}

/// The `held` signal the scaling snapshot reports (SR-7a) is set at hold
/// entry and cleared once the latest verdict is `Serving` again, before the
/// un-hold exit; it starts unset.
#[tokio::test(start_paused = true)]
async fn reactor_held_signal_runs_from_hold_entry_to_the_unhold_exit() {
    let mut reactor = Reactor::start(serving(), true);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!reactor.is_held_signal(), "unset while serving");

    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    assert!(reactor.is_held_signal(), "set at hold entry");

    reactor.publish(serving());
    wait_until("the held signal to clear", || !reactor.is_held_signal()).await;
    let (code, _) = reactor.next_exit("un-hold").await;
    assert_eq!(code, 0);
    assert!(!reactor.is_held_signal());
    reactor.stopped("after the exit").await;
}

/// The stale-verdict entry routine stops every entry of the inventory, the
/// same set a hold stops: with a running task and a token registered for
/// each, every task ends aborted, every token is cancelled, and the WARN
/// names every writer in inventory order.
#[tokio::test(start_paused = true)]
async fn enter_stale_verdict_stops_every_writer_in_the_inventory() {
    let capture = ThreadCapture::start();
    let (_tx, rx) = watch::channel(serving());
    let parts = HoldEntryParts {
        gate: HoldGate::new(rx),
        streams: GracefulShutdown::new(),
        writers: Arc::new(HoldWriterAborts::new()),
        metrics: Arc::new(crate::common::metrics_scrape_only("stale-inventory")),
    };
    let mut running = Vec::new();
    for writer in HOLD_WRITERS {
        let task = tokio::spawn(std::future::pending::<()>());
        let token = CancellationToken::new();
        parts.writers.register_task(writer, task.abort_handle());
        parts.writers.register_token(writer, token.clone());
        running.push((writer, task, token));
    }

    enter_stale_verdict(&parts);

    for (writer, task, token) in running {
        let ended = tokio::time::timeout(SETTLE, task)
            .await
            .unwrap_or_else(|_| panic!("{}: its task was left running", writer.name()));
        assert!(
            ended
                .expect_err("a pending task only ends by abort")
                .is_cancelled(),
            "{}: its task must end aborted",
            writer.name()
        );
        assert!(token.is_cancelled(), "{}: its token", writer.name());
    }
    assert_eq!(parts.gate.state(), HoldGateState::Stale);
    assert_eq!(verdict_stale_gauge(&parts.metrics), 1);
    let captured = capture.snapshot();
    let warns = lines_starting(&captured, STALE_ENTRY_PREFIX);
    assert_eq!(warns.len(), 1);
    let named: Vec<&str> = warns[0]["fields"]["aborted_writers"]
        .as_str()
        .expect("the WARN lists the writers it aborted")
        .split(|c: char| !(c.is_ascii_lowercase() || c == '_'))
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(
        named,
        HOLD_WRITERS.iter().map(|w| w.name()).collect::<Vec<_>>(),
        "the WARN names every writer it aborted, in inventory order"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The heartbeat (SDD SR-7, PM N1)
// ═════════════════════════════════════════════════════════════════════════════

/// While held, one WARN per heartbeat interval, each naming the latest
/// verdict's incident; none before the hold.
#[tokio::test(start_paused = true)]
async fn heartbeat_warns_once_per_interval_while_held_naming_the_latest_incident() {
    let capture = ThreadCapture::start();
    let reactor = Reactor::start(serving(), true);
    tokio::time::sleep(TEST_HEARTBEAT * 3).await;
    assert!(
        lines_starting(&capture.snapshot(), HEARTBEAT_PREFIX).is_empty(),
        "no heartbeat before the hold"
    );

    reactor.publish(held(INCIDENT));
    reactor.until_held().await;
    let entered = Instant::now();
    tokio::time::sleep_until(entered + TEST_HEARTBEAT * 3 + TEST_HEARTBEAT / 2).await;
    let captured = capture.snapshot();
    let beats = lines_starting(&captured, HEARTBEAT_PREFIX);
    assert_eq!(
        beats.len(),
        3,
        "one WARN per interval; capture:\n{}",
        captured.text
    );
    for beat in &beats {
        assert_eq!(beat["level"], "WARN");
        assert_eq!(beat["fields"]["incident"], INCIDENT);
        assert_eq!(beat["fields"]["reason"], REASON);
    }

    reactor.publish(held(REPLACED_INCIDENT));
    tokio::time::sleep(TEST_HEARTBEAT).await;
    let captured = capture.snapshot();
    let beats = lines_starting(&captured, HEARTBEAT_PREFIX);
    assert_eq!(beats.len(), 4);
    assert_eq!(
        beats[3]["fields"]["incident"], REPLACED_INCIDENT,
        "the heartbeat reads the latest verdict, never a copy taken at entry"
    );

    reactor.shutdown.cancel();
    reactor.stopped("after shutdown").await;
}

/// The production heartbeat is every 5 minutes, and production settings use
/// it.
#[test]
fn production_heartbeat_interval_is_five_minutes() {
    assert_eq!(HOLD_HEARTBEAT_INTERVAL, Duration::from_secs(5 * 60));
    assert_eq!(
        HoldReactorSettings::production().heartbeat_interval,
        HOLD_HEARTBEAT_INTERVAL
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The writer inventory and the abort set (SDD SR-6.2)
// ═════════════════════════════════════════════════════════════════════════════

/// The inventory is the spec's list (Task 3, "The writer inventory"), in one
/// place, each entry with a distinct stable name.
#[test]
fn hold_writers_inventory_is_the_specs_list() {
    let names: Vec<&str> = HOLD_WRITERS.iter().map(|w| w.name()).collect();
    assert_eq!(
        names,
        [
            "ext_gc",
            "orphan_sweep",
            "ext_ref_reconcile",
            "checkpoint_loop",
            "session_checkpoints",
            "ext_mirror_workers",
            "ext_backup_repair",
            "checkpoint_cleanup",
            "layer_cleanups",
            "presence_and_session_tasks",
            "on_access_repairs",
        ],
        "a writer added to or dropped from the inventory must be added to or dropped from the \
         spec's list"
    );
}

/// The entry routine stops every entry of the inventory: with a running task
/// and a token registered for each, every task ends aborted and every token
/// is cancelled, and the ERROR names every writer.
#[tokio::test(start_paused = true)]
async fn enter_hold_stops_every_writer_in_the_inventory() {
    let capture = ThreadCapture::start();
    let (_tx, rx) = watch::channel(held(INCIDENT));
    let parts = HoldEntryParts {
        gate: HoldGate::new(rx.clone()),
        streams: GracefulShutdown::new(),
        writers: Arc::new(HoldWriterAborts::new()),
        metrics: Arc::new(crate::common::metrics_scrape_only("hold-inventory")),
    };
    let mut running = Vec::new();
    for writer in HOLD_WRITERS {
        let task = tokio::spawn(std::future::pending::<()>());
        let token = CancellationToken::new();
        parts.writers.register_task(writer, task.abort_handle());
        parts.writers.register_token(writer, token.clone());
        assert!(parts.writers.is_registered(writer));
        running.push((writer, task, token));
    }
    assert!(parts.writers.unregistered().is_empty());

    let verdict = rx.borrow().verdict.clone();
    enter_hold(&parts, &verdict);

    for (writer, task, token) in running {
        let ended = tokio::time::timeout(SETTLE, task)
            .await
            .unwrap_or_else(|_| panic!("{}: its task was left running", writer.name()));
        assert!(
            ended
                .expect_err("a pending task only ends by abort")
                .is_cancelled(),
            "{}: its task must end aborted",
            writer.name()
        );
        assert!(
            token.is_cancelled(),
            "{}: its token must be cancelled",
            writer.name()
        );
    }
    let captured = capture.snapshot();
    let entry = lines_starting(&captured, ENTRY_ERROR_PREFIX);
    assert_eq!(entry.len(), 1);
    let named: Vec<&str> = entry[0]["fields"]["aborted_writers"]
        .as_str()
        .expect("the ERROR lists the writers it aborted")
        .split(|c: char| !(c.is_ascii_lowercase() || c == '_'))
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(
        named,
        HOLD_WRITERS.iter().map(|w| w.name()).collect::<Vec<_>>(),
        "the ERROR names every writer it aborted, in inventory order"
    );
}

/// A task or token registered after the abort — a writer spawned late — is
/// aborted or cancelled at once.
#[tokio::test(start_paused = true)]
async fn writer_registered_after_the_abort_is_stopped_at_once() {
    let aborts = HoldWriterAborts::new();
    assert!(
        aborts.abort_all().is_empty(),
        "nothing was registered to abort"
    );
    assert!(aborts.is_aborted());

    let task = tokio::spawn(std::future::pending::<()>());
    let token = CancellationToken::new();
    aborts.register_task(HoldWriter::LayerCleanups, task.abort_handle());
    aborts.register_token(HoldWriter::LayerCleanups, token.clone());

    let ended = tokio::time::timeout(SETTLE, task)
        .await
        .expect("a task registered after the abort must be aborted at once");
    assert!(ended.expect_err("aborted").is_cancelled());
    assert!(token.is_cancelled());
}

/// A registered task that would drain on its token's cancel (the
/// external-value mirror workers drain their queue on cancel) ends aborted
/// without running the drain. On a current-thread runtime the task cannot run
/// between the abort and the cancel, so this pins the outcome for a task whose
/// handle is registered; it cannot tell the order of the two steps apart. The
/// quiescence test shows the drain landing writes when a worker's handle is
/// not aborted (`hold_quiescence_test.rs`).
#[tokio::test(start_paused = true)]
async fn abort_all_leaves_no_registered_task_running_its_cancel_drain() {
    let aborts = HoldWriterAborts::new();
    let token = CancellationToken::new();
    let drained = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn({
        let token = token.clone();
        let drained = Arc::clone(&drained);
        async move {
            token.cancelled().await;
            drained.store(true, Ordering::SeqCst);
        }
    });
    tokio::task::yield_now().await;
    aborts.register_task(HoldWriter::ExtMirrorWorkers, task.abort_handle());
    aborts.register_token(HoldWriter::ExtMirrorWorkers, token.clone());

    assert_eq!(aborts.abort_all(), vec![HoldWriter::ExtMirrorWorkers]);

    let ended = tokio::time::timeout(SETTLE, task)
        .await
        .expect("the task ended");
    assert!(ended.expect_err("aborted").is_cancelled());
    assert!(token.is_cancelled());
    assert!(
        !drained.load(Ordering::SeqCst),
        "the aborted task must not have run its cancel drain"
    );
}

/// `unregistered` names every inventory entry with neither a task, a token
/// nor a not-spawned record — what a hold would leave running — and the
/// serving point logs them in one ERROR.
#[tokio::test(start_paused = true)]
async fn unregistered_writers_are_reported_and_logged_at_the_serving_point() {
    let capture = ThreadCapture::start();
    let aborts = Arc::new(HoldWriterAborts::new());
    assert_eq!(aborts.unregistered(), HOLD_WRITERS.to_vec());

    aborts.register_token(HoldWriter::ExtGc, CancellationToken::new());
    let task = tokio::spawn(std::future::pending::<()>());
    aborts.register_task(HoldWriter::OrphanSweep, task.abort_handle());
    aborts.register_not_spawned(HoldWriter::CheckpointLoop, "checkpointing off");
    let expected: Vec<HoldWriter> = HOLD_WRITERS
        .into_iter()
        .filter(|w| {
            ![
                HoldWriter::ExtGc,
                HoldWriter::OrphanSweep,
                HoldWriter::CheckpointLoop,
            ]
            .contains(w)
        })
        .collect();
    assert_eq!(aborts.unregistered(), expected);
    assert!(
        !aborts.is_registered(HoldWriter::CheckpointLoop),
        "a not-spawned record is not a registration"
    );

    let (_tx, rx) = watch::channel(serving());
    let parts = HoldEntryParts {
        gate: HoldGate::new(rx),
        streams: GracefulShutdown::new(),
        writers: Arc::clone(&aborts),
        metrics: Arc::new(crate::common::metrics_scrape_only("hold-serving-point")),
    };
    let point = HoldServingPoint::new();
    assert!(point.reach(parts.clone()));
    let captured = capture.snapshot();
    assert!(!point.reach(parts), "reached once, never twice");

    let errors: Vec<_> = captured
        .events
        .iter()
        .filter(|e| e["level"] == "ERROR")
        .collect();
    assert_eq!(
        errors.len(),
        1,
        "one ERROR at the serving point:\n{}",
        captured.text
    );
    let listed = errors[0]["fields"]["writers"]
        .as_str()
        .expect("the ERROR lists the unregistered writers");
    for writer in &expected {
        assert!(
            listed.contains(&format!("\"{}\"", writer.name())),
            "{} must be named: {listed}",
            writer.name()
        );
    }
    for writer in [
        HoldWriter::ExtGc,
        HoldWriter::OrphanSweep,
        HoldWriter::CheckpointLoop,
    ] {
        assert!(
            !listed.contains(&format!("\"{}\"", writer.name())),
            "{} is covered and must not be named: {listed}",
            writer.name()
        );
    }
    task.abort();
}

// ═════════════════════════════════════════════════════════════════════════════
// The gate's refusal (SDD SR-6.1)
// ═════════════════════════════════════════════════════════════════════════════

/// An open gate refuses nothing. A closed one refuses `SERVICE_HELD` with the
/// incident of the latest verdict on its channel, read at each refusal: a
/// replaced incident is named, and a cleared hold (the replica about to exit)
/// reads `incident unknown` while the gate stays closed.
#[test]
fn gate_refusal_names_the_latest_verdicts_incident() {
    let (tx, rx) = watch::channel(serving());
    let gate = HoldGate::new(rx);
    assert_eq!(gate.state(), HoldGateState::Open);
    assert!(gate.refusal().is_none(), "an open gate refuses nothing");

    tx.send_replace(held(INCIDENT));
    assert!(
        gate.refusal().is_none(),
        "a verdict alone does not close the gate: the entry routine does"
    );
    gate.close_held();
    assert_service_held(&gate.refusal().expect("closed"), Some(INCIDENT));

    tx.send_replace(held(REPLACED_INCIDENT));
    assert_service_held(&gate.refusal().expect("closed"), Some(REPLACED_INCIDENT));

    tx.send_replace(serving());
    assert_eq!(gate.state(), HoldGateState::Held, "never reopened in place");
    assert_service_held(&gate.refusal().expect("still closed"), None);

    let clone = gate.clone();
    assert_eq!(clone.state(), HoldGateState::Held, "clones share the gate");
}

/// A gate with no verdict channel — the test servers' — refuses `incident
/// unknown` once closed.
#[test]
fn always_open_gate_closed_refuses_with_incident_unknown() {
    let gate = HoldGate::always_open();
    assert!(gate.refusal().is_none());
    gate.close_held();
    assert_service_held(&gate.refusal().expect("closed"), None);
}

/// A gate closed for a stale verdict (SDD SR-6.5) is in its own state and
/// refuses `UPSTREAM_UNAVAILABLE` with no incident, whatever the verdict on
/// its channel says — never `SERVICE_HELD`, since no fleet hold is known.
/// It is never reopened, and a hold entered afterwards replaces it: the
/// fleet hold is the more specific answer.
#[test]
fn gate_closed_for_a_stale_verdict_refuses_upstream_unavailable_until_a_hold_replaces_it() {
    let (tx, rx) = watch::channel(serving());
    let gate = HoldGate::new(rx);
    gate.close_stale();
    assert_eq!(gate.state(), HoldGateState::Stale);
    assert_stale_verdict(&gate.refusal().expect("closed"));

    tx.send_replace(held(INCIDENT));
    assert_eq!(gate.state(), HoldGateState::Stale);
    assert_stale_verdict(&gate.refusal().expect("still closed"));
    gate.close_stale();
    assert_eq!(gate.state(), HoldGateState::Stale, "idempotent");

    gate.close_held();
    assert_eq!(gate.state(), HoldGateState::Held);
    assert_service_held(&gate.refusal().expect("closed"), Some(INCIDENT));
}

/// A stale closing never replaces a hold's: a held gate whose replica's
/// verdict goes stale keeps refusing `SERVICE_HELD` (PM N8).
#[test]
fn gate_closed_for_a_hold_keeps_refusing_service_held_after_a_stale_closing() {
    let (_tx, rx) = watch::channel(held(INCIDENT));
    let gate = HoldGate::new(rx);
    gate.close_held();
    gate.close_stale();
    assert_eq!(gate.state(), HoldGateState::Held);
    assert_service_held(&gate.refusal().expect("closed"), Some(INCIDENT));
}

/// The production settings publish both signals unset, and the no-verdict
/// exit code is 1: non-zero, so the restart reads as a failure. The 30 s
/// clock's value is pinned where it is defined (`dr::hold::verdict`'s tests).
#[test]
fn production_settings_start_both_signals_unset_and_the_no_verdict_exit_code_is_one() {
    let settings = HoldReactorSettings::production();
    assert!(!*settings.verdict_stale.borrow());
    assert!(!*settings.held.borrow());
    assert_eq!(NO_VERDICT_EXIT_CODE, 1);
}

// ═════════════════════════════════════════════════════════════════════════════
// The signals' vocabulary (SDD SR-7)
// ═════════════════════════════════════════════════════════════════════════════

/// The `reason` label is the marker's reason when it is one of the six, and
/// `other` for none or anything else, matched exactly.
#[test]
fn dr_hold_reason_label_keeps_the_six_reasons_and_buckets_everything_else() {
    for reason in [
        DR_HOLD_REASON_KEYSPACE_LOST,
        DR_HOLD_REASON_SLOT_LOST,
        DR_HOLD_REASON_MAPPING_LOST,
        DR_HOLD_REASON_SESSION_LOST,
        DR_HOLD_REASON_SUCCESSOR_LOST,
        DR_HOLD_REASON_RETIRED_OBJECT_MAPPED,
    ] {
        assert_eq!(dr_hold_reason_label(Some(reason)), reason);
    }
    for unbounded in [
        None,
        Some(""),
        Some("counter lost"),
        Some("KEYSPACE_LOST"),
        Some("keyspace_lost "),
        Some(DR_HOLD_REASON_OTHER),
    ] {
        assert_eq!(
            dr_hold_reason_label(unbounded),
            DR_HOLD_REASON_OTHER,
            "{unbounded:?}"
        );
    }
    assert_eq!(
        DR_HOLD_REASONS,
        [
            "keyspace_lost",
            "slot_lost",
            "mapping_lost",
            "session_lost",
            "successor_lost",
            "retired_object_mapped",
            "other",
        ]
    );
}

/// A fresh registry reports `live_edit_dr_held` 0 and every reason cell at 0;
/// an entry with an unknown reason is counted under `other` and sets the
/// gauge.
#[test]
fn dr_hold_signals_start_at_zero_and_count_an_unknown_reason_as_other() {
    let metrics = crate::common::metrics_scrape_only("hold-signals");
    assert_eq!(dr_held(&metrics), 0);
    assert!(all_hold_entries(&metrics).iter().all(|(_, n)| *n == 0));

    metrics.record_dr_hold_entered(Some("counter lost"));

    assert_eq!(dr_held(&metrics), 1);
    assert_eq!(
        all_hold_entries(&metrics),
        DR_HOLD_REASONS
            .iter()
            .map(|r| (*r, u64::from(*r == DR_HOLD_REASON_OTHER)))
            .collect::<Vec<_>>()
    );
}
