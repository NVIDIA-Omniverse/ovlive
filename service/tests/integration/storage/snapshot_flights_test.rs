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

//! The shared fenced read — `storage::snapshot_flights::SnapshotFlights`, the
//! flight registry's fenced arm (b.y3m Epic 3 Task 6; Spec SDD SR-6.1–SR-6.6,
//! SR-7.3–SR-7.5, SR-8.4). Every test here drives the registry directly, as
//! catch-up does (Task 7): attach, `read`, take a build turn, drop it. How a
//! catch-up maps the registry's outcomes to its endings, and the regimes
//! built on it, are pinned end to end in `grpc::subscribe_fenced_catch_up_test`
//! and the forced-low-ceiling tests of the other `grpc::subscribe_*` files.
//!
//! | Area | What it proves |
//! |---|---|
//! | Coalescing | waiters attached while the flight is `Queued` share one fenced acquire and one read (one `Arc<AtomicSnapshot>`); a waiter attaching between grant and release joins, at a pin equal to the cut, and the wave costs one epoch; a `BROKEN` read reopens its flight and retries in it |
//! | Attach closes before release | a newcomer after the release creates a successor (never joins the closed flight, whose cut can be below its pin) that reads after the predecessor's `done`, at or past its pin; a predecessor's last waiter leaving never removes its successor (`ptr_eq`) |
//! | Put-back | a successor abandoned while it waits puts its still-building predecessor back, closed, and the next catch-up succeeds it and waits for its `done`; a predecessor reopened by a broken read goes back open and is joined; a failed (finished) predecessor is never put back, and failing behind its successor leaves the successor in place |
//! | Drop order | a cancelled predecessor returns its slot, fenced permit and grant before its `done` wakes the successor (code review r1 F1) — its last waiter cancelled alone, two waiters cancelled together (the first parked by the flights' leave seam after its count-out while the second counts out last), and its last waiter cancelled while queued for its build turn, holding the last snapshot handle (post-completion F4) |
//! | Unit of work | one catch-up slot serves every waiter, one at a time in FIFO order; extra builds start for the queue's head when a slot and a build part are free — also when they freed while the waiters queued and the flight's turn is handed on (post-completion F18) — and a cancelled one returns both; a grant clamped to `T` runs alone; a slot wait is observed once per flight |
//! | Two waves | a wave arriving after the first flight's release succeeds it, waits for its last build uncharged and uncapped (a chain longer than `reader_wait_max(L)` included), and shares one read; `lookup` reports the measured size once the first read resolved |
//! | Ownership | a built payload left unsent holds nothing; a waiter cancelled in its own turn passes it on, and the slot and grant go after the last build |
//! | Progress | two above-ceiling layers, two slots, `P_f` = 1, `T` = one combined grant: both flights complete |
//! | Charging | each capacity wait (slot, fenced permit, memory) refuses the poorer waiter alone; `Work` (a fence sleep, the read itself) is not charged; a waiter past `reader_wait_max(L)` leaves alone, and the reader-wait histogram observes only the flight's read attempt, never the waiter's ending (code review r1 F3; defer-13) |
//! | Shutdown | the slot gate, the fenced read gate or the budget closed while a flight waits on it: every waiter ends `ShuttingDown`, `done` fires, the flight leaves the map, nothing is held |
//! | Isolation | an overwrite's new object gets a flight of its own; a delete under the read fails every waiter `NotFound`; a restore under the read is retried and every waiter is handed the restored cut to check against its own pin |
//! | Failure | a failed read leaves the map at once and is never reused; a panicking read stage fails its waiters `Internal`, fires `done` and leaves the registry usable |
//! | Cancellation (SR-7.4) | one waiter or every waiter cancelled in the predecessor wait, the slot wait, the fenced-permit wait, the memory wait, a `RETRY_AFTER` sleep and the read: slots, permits and the budget back in full, no flight left, no live lease after the lease |
//!
//! The arms that hold a read open with the fan-out's delay seam (to join it,
//! break it, check it is not charged, or delete, overwrite or restore its
//! object under it), or fail it, and the two-waiter drop-order arm that
//! parks a leave with the flights' leave seam, sit in the inner
//! `#[cfg(feature = "test-fault-injection")] mod seams`, compiled out without
//! that feature. Where a test must stop a flight at an
//! exact step — just after its release was sent, or as a predecessor's `done`
//! fires — it polls the waiter's future by hand (a no-op waker, or a waker
//! that records the capacity held when it is woken). A stop between two store
//! awaits (`drive_until`) is deterministic only on a current-thread runtime
//! that also hosts the flights' client (`hand_polled`, on
//! `snapshot_read_support`'s `on_current_thread`): no reply then lands inside
//! the poll that sent its request, so that poll ends `Pending` at the step.
//! The five tests that stop a read that way run there; every other test runs
//! on a multi-thread runtime, but for the panicking-read-stage test (a
//! current-thread `#[tokio::test]`, for its thread-local trace subscriber).
//! So the attach-closes-before-release family (its row above: the stale-cut
//! newcomer and the drop-order regression's `ptr_eq` check) has no
//! multi-thread test: each of its tests stops a read by hand at its release.
//!
//! # Time
//!
//! Nothing here asserts how long anything took. A few tests need real time to
//! pass: `reader_wait_max(L)` at these layers is ~27 s (W_wave dominates it),
//! so the reader-cap test (~31 s) and the long-chain two-wave test (~29 s)
//! each take about that long, and the fence-sleep tests sleep a staged
//! writers' gap of a few seconds. Every other test takes well under a second
//! or, holding a read open with the delay seam, a few seconds. A wait that
//! must *not* end is checked over [`QUIET`] (300 ms), which can only miss a
//! defect, never fail a correct build.
//!
//! # Hang budget
//!
//! Every test body runs under [`TEST_BUDGET`] (90 s), below nextest's 120 s
//! hang-killer, so a stuck test fails with a message rather than being
//! killed silently. For the five hand-polled tests ([`hand_polled`]) that
//! one deadline covers both the seeding, on the test's runtime, and the
//! scenario, which runs under it inside its own current-thread runtime: a
//! timeout from outside could not stop the scenario's thread. Inside it each
//! wait has its own budget, so a stuck wait panics with its own message
//! first: one read [`READ_BUDGET`] (30 s), a waiter that must end at the cap
//! [`CAP_READ_BUDGET`] (60 s), a probe, a build turn or an aborted task
//! settling [`SIGNAL_BUDGET`] (10 s), a lease lapsing [`LAPSE_BUDGET`]
//! (15 s). The two-waiter drop-order test blocks
//! its thread waiting for the flights' leave seam to park ([`SIGNAL_BUDGET`]);
//! when no park came it releases the seam before the second waiter leaves,
//! so that leave passes through, and otherwise it releases the park before
//! it joins the parked thread. It waits at most that long there, outside
//! [`TEST_BUDGET`]'s reach while it blocks; the seam's own 60 s limit
//! (`LEAVE_SEAM_PARK_LIMIT`) is a backstop it never reaches. Summed at their
//! full budgets the serial waits of the longer tests (the reader-cap and
//! long-chain tests, the cancellation and two-wave tests) exceed 90 s;
//! [`TEST_BUDGET`] cuts them there, still inside the hang-killer.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface};
use fred::types::scripts::Script;
use futures::task::{noop_waker_ref, ArcWake};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{AtomicSnapshot, SnapshotReadConsumer};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::memory_budget::{kib_for_bytes, MemoryBudget, MemoryGrant};
use live_edit::storage::snapshot_fence::read::CappedFenceWait;
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::snapshot_flights::{
    extra_build_part_bytes, flight_grant_bytes, BuildTurn, FlightPhase, FlightRead, FlightRole,
    FlightState, FlightWaitError, FlightWaiter, SnapshotFlights,
};
use live_edit::storage::subscriber_admission::{
    CapacityBudget, CatchUpGate, CatchUpPermit, CatchUpReadGate, CatchUpReadPermit,
};
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_atomic_test::{
    commit, create, create_session, every_shape_state, layer_store, SIBLING,
};
use crate::storage::snapshot_read_support::{on_current_thread, read_too_large};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The flights' client's atomic-read item ceiling: far below the fixture
/// object, so it is above the ceiling and a catch-up of it takes a flight.
const LOW_ITEMS: u64 = 16;

/// The flights' client's atomic-read byte ceiling.
const LOW_BYTES: u64 = 256;

/// Bound on one flight's read.
const READ_BUDGET: Duration = Duration::from_secs(30);

/// Bound on a waiter that must end at `reader_wait_max(L)` (~27 s here).
const CAP_READ_BUDGET: Duration = Duration::from_secs(60);

/// Bound on a probe of observable state, a build turn, or an aborted task
/// settling. Unreachable on a healthy run.
const SIGNAL_BUDGET: Duration = Duration::from_secs(10);

/// Bound on a lease nobody releases lapsing: one lease (~2.2 s) plus slack.
const LAPSE_BUDGET: Duration = Duration::from_secs(15);

/// The one deadline over a whole test. See the module's hang budget.
const TEST_BUDGET: Duration = Duration::from_secs(90);

/// How long a wait that must not end is watched (see the module docs).
const QUIET: Duration = Duration::from_millis(300);

/// A capacity budget no waiter here runs out of.
const ROOMY: Duration = Duration::from_secs(60);

/// How long the fan-out's delay seam holds a read open between grant and
/// release (renewals keep the ~2.2 s lease live through it).
#[cfg(feature = "test-fault-injection")]
const HOLD: Duration = Duration::from_secs(2);

/// A staged writers' gap past every `reader_wait_max(L)` here.
const FAR_MS: i64 = 600_000;

/// The reader-cap test's second waiter starts this long after the first.
const OFFSET: Duration = Duration::from_secs(8);

/// The reader-cap test's writers' gap ends this long after the first
/// waiter's `reader_wait_max(L)`: inside the second waiter's (it started
/// [`OFFSET`] later).
const CAP_MARGIN: Duration = Duration::from_secs(4);

/// A staged writers' gap a flight sleeps through and then reads.
const SHORT_GAP_MS: i64 = 2_500;

/// Probe interval of the bounded polls.
const PROBE: Duration = Duration::from_millis(5);

/// The `tracing` target of the flight events.
const FLIGHT_TRACE_TARGET: &str = "live_edit::snapshot_flight";

/// Run a test's body under [`TEST_BUDGET`], boxed so the body's future lives
/// on the heap rather than in the test thread's stack frame.
fn within_test_budget<'a>(
    body: impl Future<Output = ()> + 'a,
) -> std::pin::Pin<Box<dyn Future<Output = ()> + 'a>> {
    within_test_deadline(Instant::now() + TEST_BUDGET, body)
}

/// Run a test's body until `deadline`, the end of its [`TEST_BUDGET`], boxed
/// as [`within_test_budget`] does.
fn within_test_deadline<'a>(
    deadline: Instant,
    body: impl Future<Output = ()> + 'a,
) -> std::pin::Pin<Box<dyn Future<Output = ()> + 'a>> {
    let body = Box::pin(body);
    Box::pin(async move {
        tokio::time::timeout_at(deadline, body)
            .await
            .unwrap_or_else(|_| {
                panic!("the test did not finish within TEST_BUDGET ({TEST_BUDGET:?})")
            });
    })
}

/// Run `scenario` on a fixture whose flights' client lives on a current-thread
/// runtime of its own (`snapshot_read_support`'s [`on_current_thread`]): the
/// premise of [`drive_until`]'s stops. The store side ([`Seeded`]: the
/// `TestValkey`, the `LayerStore`, the session and the object) is created
/// here, on the test's multi-thread runtime, and its last clone is dropped
/// here after the scenario has returned, so `TestValkey`'s `Drop` runs on that
/// runtime. The fixture itself — the flights' client, its registry and the
/// estimate — is built and dropped inside the scenario.
///
/// [`TEST_BUDGET`] covers the whole test: the seeding here, and the scenario
/// under the same deadline inside its own runtime, since the scenario's thread
/// cannot be cancelled from out here.
async fn hand_polled<S, Fut>(layer: &str, shape: Shape, scenario: S)
where
    S: FnOnce(Fx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    let deadline = Instant::now() + TEST_BUDGET;
    let seeded = tokio::time::timeout_at(deadline, Seeded::new(layer))
        .await
        .unwrap_or_else(|_| panic!("the test did not finish within TEST_BUDGET ({TEST_BUDGET:?})"));
    let theirs = seeded.clone();
    on_current_thread(move || {
        within_test_deadline(deadline, async move {
            scenario(Fx::on(theirs, shape).await).await;
        })
    })
    .await;
    drop(seeded);
}

/// The store clock, as the fence scripts read it: `TIME` in whole ms.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The capacity a fixture's client is built with.
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`.
    slots: usize,
    /// `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY` (`P_f`).
    fenced: u32,
    /// `T` in KiB, when a test needs one finer than the config's MiB.
    budget_kib: Option<u64>,
}

impl Shape {
    const DEFAULT: Shape = Shape {
        slots: 8,
        fenced: 1,
        budget_kib: None,
    };

    const fn slots(slots: usize) -> Shape {
        Shape {
            slots,
            ..Shape::DEFAULT
        }
    }
}

type ReadTask = JoinHandle<Result<FlightRead, FlightWaitError>>;

/// One above-ceiling object on a fresh store: the store side of a fixture.
/// Cheap to clone; [`hand_polled`] keeps a clone on the test's multi-thread
/// runtime so the last of it drops there, never on the scenario's
/// current-thread runtime.
#[derive(Clone)]
struct Seeded {
    tv: Arc<TestValkey>,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
}

impl Seeded {
    async fn new(layer: &str) -> Self {
        let tv = TestValkey::setup().await;
        let ls = layer_store(&tv);
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
        Seeded {
            tv: Arc::new(tv),
            ls,
            session_id,
            layer: layer.to_owned(),
            oid,
        }
    }
}

/// One above-ceiling object on a fresh store, a client whose capacity is
/// [`Shape`], with a metrics registry on it, and a registry on that client.
struct Fx {
    tv: Arc<TestValkey>,
    client: ValkeyClient,
    metrics: SharedMetrics,
    flights: SnapshotFlights,
    shape: Shape,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
    /// The catch-up's size for the object, from its `TOO_LARGE` answer.
    estimate: LayerSize,
}

impl Fx {
    async fn new(layer: &str, shape: Shape) -> Self {
        Fx::on(Seeded::new(layer).await, shape).await
    }

    /// The fixture on `seeded`: its flights' client is connected here, on
    /// the runtime this runs on.
    async fn on(seeded: Seeded, shape: Shape) -> Self {
        let Seeded {
            tv,
            ls,
            session_id,
            layer,
            oid,
        } = seeded;
        let mut config = crate::common::test_config();
        config.snapshot_atomic_read_max_items = LOW_ITEMS;
        config.snapshot_atomic_read_max_bytes = LOW_BYTES;
        config.subscriber_max_concurrent_catchups = shape.slots;
        config.catchup_fenced_read_concurrency = shape.fenced;
        let mut client = connect_test_client(&config)
            .await
            .expect("the flights' client must connect");
        let metrics: SharedMetrics =
            Arc::new(crate::common::metrics_scrape_only("snapshot-flights"));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        if let Some(kib) = shape.budget_kib {
            client.catchup_build_budget = Arc::new(MemoryBudget::new(
                kib,
                Arc::new(StdMutex::new(Some(Arc::clone(&metrics)))),
            ));
        }
        let estimate = estimate_of(&client, tv.key_schema(), oid).await;
        let flights = SnapshotFlights::new(client.clone(), tv.key_schema().clone());
        Fx {
            tv,
            client,
            metrics,
            flights,
            shape,
            ls,
            session_id,
            layer,
            oid,
            estimate,
        }
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    fn gate(&self) -> &Arc<CatchUpGate> {
        &self.client.catchup_gate
    }

    fn fenced(&self) -> &Arc<CatchUpReadGate> {
        &self.client.catchup_fenced_read_gate
    }

    fn budget(&self) -> &Arc<MemoryBudget> {
        &self.client.catchup_build_budget
    }

    /// A catch-up of the fixture's object attaching (or creating its flight).
    fn attach(&self) -> FlightWaiter {
        self.flights.attach_or_create(self.oid, self.estimate)
    }

    /// The fixture object's flight's `(attachable, state)`, if one is
    /// registered.
    fn registered(&self) -> Option<(bool, FlightState)> {
        self.flights
            .lookup(self.oid)
            .map(|l| (l.attachable, l.state))
    }

    fn header_key(&self, oid: u64) -> String {
        self.keys().obj_header(&oid.to_string())
    }

    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid.to_string())
    }

    #[cfg(feature = "test-fault-injection")]
    fn intent_key(&self) -> String {
        self.keys().obj_snapshot_intent(&self.oid.to_string())
    }

    /// A catch-up's pin: the header's `seq` now.
    async fn seq_of(&self, oid: u64) -> u64 {
        let seq: Option<String> = self
            .tv
            .client()
            .pool
            .hget(self.header_key(oid), "seq")
            .await
            .expect("HGET seq must succeed");
        seq.expect("the header has a seq")
            .parse()
            .expect("seq is numeric")
    }

    async fn seq(&self) -> u64 {
        self.seq_of(self.oid).await
    }

    /// Set the header's `seq` directly (a non-commit mutator's write).
    #[cfg(feature = "test-fault-injection")]
    async fn set_seq(&self, seq: u64) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(self.header_key(self.oid), [("seq", seq.to_string())])
            .await
            .expect("staging HSET seq must succeed");
    }

    /// One real commit on the object (a fast-path inline field).
    async fn commit_one(&self) {
        commit(
            &self.ls,
            &self.session_id,
            &self.layer,
            self.oid,
            delta_with_set_field_inline(SIBLING, "label", vec![0x2A], 5),
        )
        .await;
    }

    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    /// `live_edit_snapshot_read_total{consumer="catchup", path="fenced",
    /// outcome}`: one per fenced read attempt.
    fn reads(&self, outcome: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "catchup"),
                ("path", "fenced"),
                ("outcome", outcome),
            ],
        ) as u64
    }

    fn acquires(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_fence_acquire_total",
            &[("result", result)],
        ) as u64
    }

    /// `live_edit_catchup_snapshot_flight_total{role}`.
    fn roles(&self, role: &str) -> u64 {
        self.metric("live_edit_catchup_snapshot_flight_total", &[("role", role)]) as u64
    }

    fn builds_in_progress(&self) -> i64 {
        self.metric("live_edit_catchup_builds_in_progress", &[]) as i64
    }

    fn slot_waiters(&self) -> i64 {
        self.metric("live_edit_subscribe_catchups_waiting", &[]) as i64
    }

    /// `live_edit_snapshot_fence_reader_wait_seconds_count{consumer="catchup"}`:
    /// one observation per fenced read attempt's lease wait, the flight's own,
    /// however many waiters share it — never one per waiter, and none for a
    /// waiter that leaves at `reader_wait_max(L)`.
    fn reader_waits(&self) -> u64 {
        self.metric(
            "live_edit_snapshot_fence_reader_wait_seconds_count",
            &[("consumer", "catchup")],
        ) as u64
    }

    /// Stage a writers' gap ending `ahead_ms` after the store's clock, sized
    /// at the estimate: acquire answers `RETRY_AFTER gap` until then.
    async fn stage_gap(&self, ahead_ms: i64) {
        let until = self.now().await + ahead_ms;
        let _: () = self
            .tv
            .client()
            .pool
            .hset(
                self.fence_key(),
                [
                    ("gap_until", until.to_string()),
                    ("size_e", self.estimate.items.to_string()),
                    ("size_b", self.estimate.bytes.to_string()),
                ],
            )
            .await
            .expect("staging the gap must succeed");
        let _: i64 = self
            .tv
            .client()
            .pool
            .pexpire(self.fence_key(), FAR_MS * 2, None)
            .await
            .expect("staging the key expiry must succeed");
    }

    async fn now(&self) -> i64 {
        CLOCK_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                self.tv.client().pool.next(),
                vec![self.fence_key()],
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must answer")
    }

    /// The fence key's reader fields (`r:<token>` → deadline).
    async fn readers(&self) -> Vec<i64> {
        let fields: std::collections::HashMap<String, String> = self
            .tv
            .client()
            .pool
            .hgetall(self.fence_key())
            .await
            .expect("HGETALL fence must succeed");
        fields
            .into_iter()
            .filter(|(k, _)| k.starts_with("r:"))
            .map(|(_, v)| v.parse().expect("a deadline is an integer"))
            .collect()
    }

    /// Poll until no reader field is live on the store's clock.
    async fn until_no_live_lease(&self, what: &str) {
        let wait = async {
            loop {
                let now = self.now().await;
                if self.readers().await.iter().all(|deadline| *deadline <= now) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::time::timeout(LAPSE_BUDGET, wait)
            .await
            .unwrap_or_else(|_| panic!("{what}: a lease was still live after {LAPSE_BUDGET:?}"));
    }

    /// Poll until the fence key holds no reader field: a lease was released.
    async fn until_released(&self, what: &str) {
        let wait = async {
            while !self.readers().await.is_empty() {
                tokio::time::sleep(PROBE).await;
            }
        };
        tokio::time::timeout(SIGNAL_BUDGET, wait)
            .await
            .unwrap_or_else(|_| {
                panic!("{what}: the lease was not released within {SIGNAL_BUDGET:?}")
            });
    }

    /// Everything a flight takes is back: no flight registered, every slot
    /// and fenced permit free, nothing reserved or over-committed, no build
    /// counted, and no lease live once leases have had time to lapse.
    async fn assert_idle(&self, what: &str) {
        until(&format!("{what}: the registry emptied"), || {
            self.flights.is_empty()
        })
        .await;
        assert_eq!(
            self.gate().available(),
            self.shape.slots,
            "{what}: every catch-up slot is free"
        );
        assert_eq!(
            self.fenced().available(),
            self.shape.fenced as usize,
            "{what}: every fenced read permit is free"
        );
        assert_eq!(
            (
                self.budget().reserved_bytes(),
                self.budget().overcommit_bytes()
            ),
            (0, 0),
            "{what}: reserved == 0 && overcommit == 0"
        );
        assert_eq!(self.builds_in_progress(), 0, "{what}: no build is counted");
        self.until_no_live_lease(what).await;
    }

    /// Poll `fut` (a waiter's `read`) by hand until its flight has closed to
    /// attach while still `Reading` — the fan-out ended and the release
    /// script was sent — and leave it there, unresolved: nothing advances it
    /// until it is polled again. The stop lasts one release round trip; it is
    /// deterministic only because, on [`hand_polled`]'s current-thread
    /// runtime, the release's reply cannot be delivered inside the poll that
    /// sent it (see [`drive_until`]).
    async fn drive_until_closed<F>(&self, fut: Pin<&mut F>)
    where
        F: Future + ?Sized,
        F::Output: std::fmt::Debug,
    {
        drive_until(fut, "its release was sent", || {
            self.registered() == Some((false, FlightState::Reading))
        })
        .await;
    }

    #[cfg(feature = "test-fault-injection")]
    fn injector(&self) -> Arc<live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector> {
        let injector =
            Arc::new(live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector::new());
        self.client
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        injector
    }
}

/// A catch-up's size for `oid`, from its atomic read's `TOO_LARGE` answer on
/// `client` (Spec SDD SR-11.1); also the precondition that `oid` is above
/// `client`'s ceiling.
async fn estimate_of(client: &ValkeyClient, keys: &KeySchema, oid: u64) -> LayerSize {
    LayerSize::from_too_large(&read_too_large(client, keys, oid).await, None)
}

/// Poll `cond` until it holds, within [`SIGNAL_BUDGET`].
async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let wait = async {
        while !cond() {
            tokio::time::sleep(PROBE).await;
        }
    };
    tokio::time::timeout(SIGNAL_BUDGET, wait)
        .await
        .unwrap_or_else(|_| panic!("{what} did not happen within {SIGNAL_BUDGET:?}"));
}

/// Poll `fut` (a waiter's `read`) by hand, every millisecond, until `cond`
/// holds, and leave it there unresolved: nothing advances it until it is
/// polled again. `step` names the point, for the panics.
///
/// A stop between two store awaits is deterministic only on a current-thread
/// runtime that also hosts the flights' client ([`hand_polled`]): no other
/// task runs while the read is polled, so a reply cannot be delivered inside
/// the poll that sent its request, and that poll ends `Pending` with `cond`
/// observable. On a multi-thread runtime a worker can deliver the reply while
/// the polling thread is descheduled, and the read then runs past the stop in
/// one poll — so this asserts the runtime's flavor first. Each poll runs
/// under `tokio::task::coop::unconstrained`, so an exhausted coop budget
/// never answers `Pending` in place of a step.
async fn drive_until<F>(mut fut: Pin<&mut F>, step: &str, mut cond: impl FnMut() -> bool)
where
    F: Future + ?Sized,
    F::Output: std::fmt::Debug,
{
    assert_eq!(
        tokio::runtime::Handle::current().runtime_flavor(),
        tokio::runtime::RuntimeFlavor::CurrentThread,
        "fixture: a hand-polled stop needs a current-thread runtime hosting the flights' client \
         (`hand_polled`); on a multi-thread runtime a store reply can land inside the poll that \
         sent its request, and the read runs past the point where {step}"
    );
    let deadline = Instant::now() + READ_BUDGET;
    loop {
        let polled = tokio::task::coop::unconstrained(std::future::poll_fn(|_| {
            Poll::Ready(poll_now(fut.as_mut()))
        }))
        .await;
        if let Poll::Ready(out) = polled {
            panic!("fixture: the read resolved before it could be stopped when {step}: {out:?}");
        }
        if cond() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "fixture: the flight never reached the point where {step} within {READ_BUDGET:?}"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// Poll `fut` once with a no-op waker.
fn poll_now<F: Future + ?Sized>(fut: Pin<&mut F>) -> Poll<F::Output> {
    fut.poll(&mut Context::from_waker(noop_waker_ref()))
}

/// `waiter.read` with a capacity budget of `budget`, in a task of its own.
fn spawn_read(waiter: FlightWaiter, budget: Duration) -> ReadTask {
    tokio::spawn(async move {
        let budget = CapacityBudget::new(budget);
        waiter.read(&budget).await
    })
}

/// A read task's outcome, within `bound`.
async fn joined_within(
    task: ReadTask,
    bound: Duration,
    what: &str,
) -> Result<FlightRead, FlightWaitError> {
    tokio::time::timeout(bound, task)
        .await
        .unwrap_or_else(|_| panic!("{what}: the read did not end within {bound:?}"))
        .unwrap_or_else(|e| panic!("{what}: the read task failed: {e}"))
}

async fn joined(task: ReadTask, what: &str) -> Result<FlightRead, FlightWaitError> {
    joined_within(task, READ_BUDGET, what).await
}

/// A read task that must be served.
async fn served(task: ReadTask, what: &str) -> FlightRead {
    joined(task, what)
        .await
        .unwrap_or_else(|e| panic!("{what}: the waiter must be served; got {e:?}"))
}

/// Abort a read task and assert it was cancelled mid-wait, not finished.
async fn cancel(task: ReadTask, what: &str) {
    task.abort();
    let joined = tokio::time::timeout(SIGNAL_BUDGET, task)
        .await
        .unwrap_or_else(|_| panic!("{what}: the aborted read did not settle"));
    assert!(
        matches!(&joined, Err(e) if e.is_cancelled()),
        "{what}: the read must have been cancelled mid-wait, not finished: {joined:?}"
    );
}

/// A waiter's build turn, within [`SIGNAL_BUDGET`].
async fn turn(read: FlightRead, what: &str) -> BuildTurn {
    tokio::time::timeout(SIGNAL_BUDGET, read.build_turn())
        .await
        .unwrap_or_else(|_| panic!("{what}: no build turn within {SIGNAL_BUDGET:?}"))
}

/// A queued build turn that must now be handed over.
fn ready_turn<F: Future<Output = BuildTurn> + ?Sized>(fut: Pin<&mut F>, what: &str) -> BuildTurn {
    match poll_now(fut) {
        Poll::Ready(turn) => turn,
        Poll::Pending => panic!("{what}: the build turn must have been handed over"),
    }
}

/// A build turn that must still be queued.
fn assert_queued<F: Future<Output = BuildTurn> + ?Sized>(fut: Pin<&mut F>, what: &str) {
    assert!(poll_now(fut).is_pending(), "{what}: must still be queued");
}

fn snapshot_ptr(read: &FlightRead) -> *const AtomicSnapshot {
    read.snapshot()
}

/// Assert every read shares one snapshot (one `Arc`), and return its cut.
fn assert_one_snapshot(reads: &[&FlightRead], what: &str) -> u64 {
    let first = snapshot_ptr(reads[0]);
    for (i, r) in reads.iter().enumerate() {
        assert!(
            std::ptr::eq(snapshot_ptr(r), first),
            "{what}: waiter {i} must be handed the flight's one shared snapshot"
        );
    }
    reads[0].snapshot().cut
}

/// Build every read in turn, dropping each turn as its "build" returns.
async fn build_all(reads: Vec<FlightRead>, what: &str) {
    for r in reads {
        drop(turn(r, what).await);
    }
}

/// `reader_wait_max(L)` for `size` on `client`'s coefficients — the cap a
/// flight's waiters count their work against.
fn reader_wait_max(client: &ValkeyClient, size: LayerSize) -> Duration {
    CappedFenceWait::new(client, 0, SnapshotReadConsumer::Catchup).cap(size)
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Coalescing (SR-6.1, SR-6.2)
// ═════════════════════════════════════════════════════════════════════════════

/// Waiters attached while the flight waits for its slot (`Queued`) — the
/// last one attaching during that wait — share one fenced acquire, one read
/// and one snapshot. With one catch-up slot, every waiter completes and they
/// build one at a time under the flight's slot (none is an extra build); the
/// flight's slot wait is observed once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiters_attached_while_queued_share_one_fenced_read_and_one_slot() {
    within_test_budget(async {
        let fx = Fx::new("sf-coalesce", Shape::slots(1)).await;
        let held = fx
            .gate()
            .try_acquire()
            .expect("the test takes the only slot");

        let waiters: Vec<FlightWaiter> = (0..3).map(|_| fx.attach()).collect();
        assert_eq!(
            waiters.iter().map(FlightWaiter::role).collect::<Vec<_>>(),
            vec![
                FlightRole::Leader,
                FlightRole::JoinedRead,
                FlightRole::JoinedRead
            ]
        );
        let mut tasks: Vec<ReadTask> = waiters.into_iter().map(|w| spawn_read(w, ROOMY)).collect();
        until("the flight parks in its slot wait", || {
            fx.slot_waiters() == 1
        })
        .await;
        assert_eq!(fx.registered(), Some((true, FlightState::Queued)));
        let late = fx.attach();
        assert_eq!(
            late.role(),
            FlightRole::JoinedRead,
            "a flight waiting for capacity is still attachable"
        );
        tasks.push(spawn_read(late, ROOMY));
        assert_eq!(fx.flights.len(), 1);
        drop(held);

        let mut reads = Vec::new();
        for (i, t) in tasks.into_iter().enumerate() {
            reads.push(served(t, &format!("waiter {i}")).await);
        }
        assert_one_snapshot(&reads.iter().collect::<Vec<_>>(), "four waiters");
        assert_eq!(
            (fx.acquires("started_epoch"), fx.acquires("joined_epoch")),
            (1, 0),
            "one fenced acquire for the whole wave"
        );
        assert_eq!(fx.reads("ok"), 1, "one fenced read for the whole wave");
        assert_eq!(
            (fx.roles("leader"), fx.roles("joined_read")),
            (1, 3),
            "one leader, the rest joined its read"
        );
        assert_eq!(
            fx.metric(
                "live_edit_subscribe_catchup_wait_seconds_count",
                &[("result", "acquired")]
            ),
            1.0,
            "one slot wait per flight; the attached waiters take none"
        );

        // Build concurrently, as the catch-ups would: with one slot, one at a
        // time, all under the flight's slot.
        let building = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let builds: Vec<JoinHandle<bool>> = reads
            .into_iter()
            .map(|r| {
                let (building, most) = (Arc::clone(&building), Arc::clone(&most));
                tokio::spawn(async move {
                    let turn = r.build_turn().await;
                    let now = building.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    building.fetch_sub(1, Ordering::SeqCst);
                    let extra = turn.is_extra();
                    drop(turn);
                    extra
                })
            })
            .collect();
        for b in builds {
            let extra = tokio::time::timeout(SIGNAL_BUDGET, b)
                .await
                .expect("every build completes")
                .expect("the build task must not fail");
            assert!(!extra, "no slot is free for an extra build");
        }
        assert_eq!(
            most.load(Ordering::SeqCst),
            1,
            "the flight's waiters build one at a time"
        );
        fx.assert_idle("after the wave").await;
    })
    .await;
}

/// The attach rule after the release (Spec SDD SR-6.2, PM review r2 F25): a
/// flight stopped just after its release was sent is closed to attach. A
/// commit lands after that release; a newcomer pinned at the new `seq`
/// creates a successor — it must never join the closed flight, whose cut is
/// the old `seq` — and the successor reads, after the predecessor's last
/// build, at or past the newcomer's pin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_closed_at_its_release_is_succeeded_never_joined_by_a_newcomer_pinned_past_its_cut()
{
    hand_polled("sf-stale-cut", Shape::DEFAULT, |fx| async move {
        let before = fx.seq().await;
        let budget = CapacityBudget::new(ROOMY);
        let mut leader = Box::pin(fx.attach().read(&budget));
        fx.drive_until_closed(leader.as_mut()).await;
        fx.until_released("the stopped leader's release").await;

        fx.commit_one().await;
        let pin = fx.seq().await;
        assert!(pin > before, "fixture: the commit moved the header");
        let newcomer = fx.attach();

        assert_eq!(
            newcomer.role(),
            FlightRole::Leader,
            "a newcomer after the release must create a successor, not join a read whose cut \
             ({before}) is below its pin ({pin})"
        );
        assert_eq!(fx.flights.len(), 1, "the successor replaced it in the map");
        assert_eq!(fx.registered(), Some((true, FlightState::Queued)));
        let successor = spawn_read(newcomer, ROOMY);

        let first = tokio::time::timeout(READ_BUDGET, leader)
            .await
            .expect("the predecessor's read resumes and ends")
            .expect("the predecessor is served");
        assert_eq!(first.snapshot().cut, before);
        let held = turn(first, "the predecessor's build").await;
        assert!(
            tokio::time::timeout(QUIET, async {
                while !successor.is_finished() {
                    tokio::time::sleep(PROBE).await;
                }
            })
            .await
            .is_err(),
            "the successor waits for the predecessor's last build"
        );
        drop(held);

        let second = served(successor, "the successor").await;
        assert!(
            second.snapshot().cut >= pin,
            "the successor's cut {} is at or past the newcomer's pin {pin}",
            second.snapshot().cut
        );
        assert_eq!(fx.reads("ok"), 2, "one read per flight");
        drop(turn(second, "the successor's build").await);
        fx.assert_idle("after both flights").await;
    })
    .await;
}

/// What a waker saw held when it was woken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Held {
    free_slots: usize,
    free_fenced_permits: usize,
    reserved: u64,
    overcommit: u64,
}

/// A waker that records the catch-up capacity held at the moment it is
/// woken. Polled with it, a successor's read registers it on its
/// predecessor's `done`, so it records exactly what the predecessor still
/// held when `done` fired — synchronously, inside the predecessor's drop.
struct WakeProbe {
    gate: Arc<CatchUpGate>,
    fenced: Arc<CatchUpReadGate>,
    budget: Arc<MemoryBudget>,
    seen: StdMutex<Vec<Held>>,
}

impl ArcWake for WakeProbe {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let held = Held {
            free_slots: arc_self.gate.available(),
            free_fenced_permits: arc_self.fenced.available(),
            reserved: arc_self.budget.reserved_bytes(),
            overcommit: arc_self.budget.overcommit_bytes(),
        };
        arc_self.seen.lock().expect("probe").push(held);
    }
}

impl WakeProbe {
    /// A probe on `fx`'s gate, fenced read gate and budget.
    fn new(fx: &Fx) -> Arc<Self> {
        Arc::new(WakeProbe {
            gate: Arc::clone(fx.gate()),
            fenced: Arc::clone(fx.fenced()),
            budget: Arc::clone(fx.budget()),
            seen: StdMutex::new(Vec::new()),
        })
    }

    /// Every wake so far, oldest first.
    fn seen(&self) -> Vec<Held> {
        self.seen.lock().expect("probe").clone()
    }
}

/// What a successor must find held when its predecessor's `done` wakes it on
/// a one-slot, `P_f` = 1 fixture: every slot and fenced permit free, nothing
/// reserved or over-committed.
const ALL_FREE_AT_ONE_SLOT: Held = Held {
    free_slots: 1,
    free_fenced_permits: 1,
    reserved: 0,
    overcommit: 0,
};

/// A successor of `fx`'s flight, created now (the flight must be closed to
/// attach), polled once with a [`WakeProbe`] so its predecessor's `done`
/// wakes the probe. Returns the probe and the successor's pending read.
fn probed_successor<'b>(
    fx: &Fx,
    budget: &'b CapacityBudget,
    what: &str,
) -> (
    Arc<WakeProbe>,
    Pin<Box<impl Future<Output = Result<FlightRead, FlightWaitError>> + 'b>>,
) {
    let successor = fx.attach();
    assert_eq!(
        successor.role(),
        FlightRole::Leader,
        "{what}: a newcomer to a closed flight creates its successor"
    );
    let probe = WakeProbe::new(fx);
    let waker = futures::task::waker(Arc::clone(&probe));
    let mut next = Box::pin(successor.read(budget));
    assert!(
        next.as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending(),
        "{what}: the successor waits for its predecessor's done"
    );
    assert!(probe.seen().is_empty(), "{what}: not woken yet");
    (probe, next)
}

/// Regression (code review r1 F1, path (b)): a flight cancelled after its
/// release was sent — holding its slot, its fenced permit and its grant —
/// gives all three back before its `done` fires, so the successor woken by
/// `done` finds them free. With one slot and `P_f` = 1, the successor's read
/// stage is polled with a [`WakeProbe`]; the predecessor's last waiter is
/// dropped; the probe's first wake must see everything free. The
/// predecessor's leaving does not touch the successor's map entry (`ptr_eq`):
/// the successor is still registered, open, and a newcomer joins it. Then the
/// successor (the predecessor dropped, not finished) proceeds and is served,
/// to both its waiters.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_cancelled_after_its_release_frees_slot_permit_and_grant_before_its_successor_wakes()
{
    hand_polled("sf-drop-order", Shape::slots(1), |fx| async move {
        let budget = CapacityBudget::new(ROOMY);
        let mut leader = Box::pin(fx.attach().read(&budget));
        fx.drive_until_closed(leader.as_mut()).await;
        assert_eq!(
            (
                fx.gate().available(),
                fx.fenced().available(),
                fx.budget().reserved_bytes() > 0
            ),
            (0, 0, true),
            "fixture: stopped after its release was sent, the flight holds its slot, its fenced \
             permit and its grant"
        );

        let successor_budget = CapacityBudget::new(ROOMY);
        let (probe, next) = probed_successor(&fx, &successor_budget, "the successor");

        drop(leader);

        let seen = probe.seen();
        assert_eq!(
            seen.first().copied(),
            Some(ALL_FREE_AT_ONE_SLOT),
            "when the predecessor's done woke the successor, its slot, fenced permit and grant \
             must already have been returned (every wake: {seen:?})"
        );
        assert_eq!(
            fx.registered(),
            Some((true, FlightState::Queued)),
            "a predecessor's last waiter leaving never removes its successor"
        );
        let joiner = fx.attach();
        assert_eq!(
            joiner.role(),
            FlightRole::JoinedRead,
            "the successor is still the object's flight"
        );
        let joiner = spawn_read(joiner, ROOMY);

        let read = tokio::time::timeout(READ_BUDGET, next)
            .await
            .expect("the successor proceeds once its predecessor is dropped")
            .expect("the successor is served");
        let joined = served(joiner, "the successor's joiner").await;
        assert_one_snapshot(&[&read, &joined], "the successor");
        build_all(vec![read, joined], "the successor").await;
        fx.assert_idle("after the successor").await;
    })
    .await;
}

/// Regression (post-completion review F4 (ii)): the last waiter of a flight
/// cancelled while queued for its build turn — the flight's turn already
/// handed to it, unreceived — holds the flight's last snapshot handle, which
/// owns the grant. Its cancellation gives the grant back before its leaving
/// fires `done`: the successor woken by `done` finds the slot, the fenced
/// permit and the memory all back. The first waiter builds and drops its
/// turn (handing it on to the second, unpolled), then the second's queued
/// `build_turn` is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiter_cancelled_while_queued_as_the_last_frees_its_grant_before_its_successor_wakes(
) {
    within_test_budget(async {
        let fx = Fx::new("sf-drop-order-queued", Shape::slots(1)).await;
        let (w1, w2) = (fx.attach(), fx.attach());
        let (t1, t2) = (spawn_read(w1, ROOMY), spawn_read(w2, ROOMY));
        let (r1, r2) = (served(t1, "w1").await, served(t2, "w2").await);
        let b1 = turn(r1, "w1").await;
        assert!(!b1.is_extra(), "w1 holds the flight's own turn");
        let mut f2 = Box::pin(r2.build_turn());
        assert_queued(f2.as_mut(), "w2 while w1 builds");
        let successor_budget = CapacityBudget::new(ROOMY);
        let (probe, next) = probed_successor(&fx, &successor_budget, "the successor");

        drop(b1);
        assert!(
            probe.seen().is_empty(),
            "w2 is still the flight's waiter, so its done has not fired"
        );
        drop(f2);

        let seen = probe.seen();
        assert_eq!(
            seen.first().copied(),
            Some(ALL_FREE_AT_ONE_SLOT),
            "when the predecessor's done woke the successor, the cancelled last waiter's snapshot \
             handle — the grant's owner — must already have been dropped (every wake: {seen:?})"
        );
        let read = tokio::time::timeout(READ_BUDGET, next)
            .await
            .expect("the successor proceeds once its predecessor is done")
            .expect("the successor is served");
        build_all(vec![read], "the successor").await;
        fx.assert_idle("after the successor").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Build turns: FIFO, extra builds, the clamp (SR-6.3, SR-7.2, SR-7.5)
// ═════════════════════════════════════════════════════════════════════════════

/// With no slot free for an extra build, a flight's waiters build one at a
/// time, in the order they asked: the first takes the flight's turn at once,
/// the others queue, and each finished build hands the turn to the oldest.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiters_build_one_at_a_time_in_the_order_they_asked_when_nothing_is_free() {
    within_test_budget(async {
        let fx = Fx::new("sf-fifo", Shape::slots(1)).await;
        let (w1, w2, w3) = (fx.attach(), fx.attach(), fx.attach());
        let (t1, t2, t3) = (
            spawn_read(w1, ROOMY),
            spawn_read(w2, ROOMY),
            spawn_read(w3, ROOMY),
        );
        let (r1, r2, r3) = (
            served(t1, "w1").await,
            served(t2, "w2").await,
            served(t3, "w3").await,
        );

        let b1 = turn(r1, "w1").await;
        assert!(!b1.is_extra(), "the first takes the flight's own turn");
        let mut f2 = Box::pin(r2.build_turn());
        let mut f3 = Box::pin(r3.build_turn());
        assert_queued(f2.as_mut(), "w2 while w1 builds");
        assert_queued(f3.as_mut(), "w3 while w1 builds");
        assert_eq!(fx.builds_in_progress(), 1);

        drop(b1);
        assert_queued(f3.as_mut(), "w3, behind w2");
        let b2 = ready_turn(f2.as_mut(), "w2, the oldest queued");
        assert!(!b2.is_extra());
        drop(b2);
        let b3 = ready_turn(f3.as_mut(), "w3");
        assert!(!b3.is_extra());
        drop(b3);
        fx.assert_idle("after the builds").await;
    })
    .await;
}

/// Extra builds (Spec SDD SR-6.3, PM review r3 F29 path (b), r4 F41): with
/// two slots, while one waiter holds the flight's turn the next asking builds
/// at once under the second slot and a build part of its own. With both
/// slots busy the next two queue. When the extra build is cancelled mid-build
/// its slot and part go to the queue's head — not the later waiter — and the
/// queue otherwise stands: the later waiter gets the flight's own turn when
/// it comes back. A cancelled extra with nobody queued returns its slot and
/// part outright. No extra build records a slot wait; `reserved == 0`
/// afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_extra_builds_go_to_the_queues_head_and_a_cancelled_one_returns_its_slot_and_part() {
    within_test_budget(async {
        let fx = Fx::new("sf-extras", Shape::slots(2)).await;
        let waiters: Vec<FlightWaiter> = (0..4).map(|_| fx.attach()).collect();
        let tasks: Vec<ReadTask> = waiters.into_iter().map(|w| spawn_read(w, ROOMY)).collect();
        let mut reads = Vec::new();
        for (i, t) in tasks.into_iter().enumerate() {
            reads.push(served(t, &format!("w{}", i + 1)).await);
        }
        let mut reads = reads.into_iter();
        let (r1, r2, r3, r4) = (
            reads.next().unwrap(),
            reads.next().unwrap(),
            reads.next().unwrap(),
            reads.next().unwrap(),
        );

        let b1 = turn(r1, "w1").await;
        assert!(!b1.is_extra());
        let grant = b1.memory.bytes();
        let part = kib_for_bytes(extra_build_part_bytes(b1.reply_bytes)) * 1024;
        let held = || fx.budget().reserved_bytes() + fx.budget().overcommit_bytes();
        assert_eq!(held(), grant, "the flight's grant, owned by its snapshot");

        let b2 = turn(r2, "w2").await;
        assert!(
            b2.is_extra(),
            "a slot and a build part are free: w2 builds at once"
        );
        assert_eq!(fx.gate().available(), 0);
        assert_eq!(
            held(),
            grant + part,
            "the extra build holds a part of its own"
        );
        assert_eq!(fx.builds_in_progress(), 2, "two builds at once");

        let mut f3 = Box::pin(r3.build_turn());
        let mut f4 = Box::pin(r4.build_turn());
        assert_queued(f3.as_mut(), "w3 with both slots busy");
        assert_queued(f4.as_mut(), "w4 with both slots busy");

        drop(b2);
        assert_queued(f4.as_mut(), "w4, behind the queue's head");
        let b3 = ready_turn(f3.as_mut(), "w3, the queue's head");
        assert!(b3.is_extra(), "the freed slot and part went to the head");
        assert_eq!(held(), grant + part);

        drop(b1);
        let b4 = ready_turn(f4.as_mut(), "w4, when the flight's turn came back");
        assert!(
            !b4.is_extra(),
            "the cancelled extra left the queue standing: w4 waited for the flight's own turn"
        );

        drop(b3);
        assert_eq!(
            fx.gate().available(),
            1,
            "a cancelled extra with nobody queued returns its slot"
        );
        assert_eq!(held(), grant, "and its part");
        drop(b4);
        assert_eq!(
            fx.metric(
                "live_edit_subscribe_catchup_wait_seconds_count",
                &[("result", "acquired")]
            ),
            1.0,
            "the flight's slot wait is observed; an extra build's slot records none"
        );
        fx.assert_idle("after the builds").await;
    })
    .await;
}

/// What kept an extra build from starting while the waiters queued.
#[derive(Debug, Clone, Copy)]
enum ExtrasBlockedBy {
    /// Every catch-up slot was taken.
    Slot,
    /// The build-memory budget was full.
    Memory,
}

/// Extra builds after a hand-on (post-completion review F18): with two
/// slots, the first waiter builds on the flight's turn while the second and
/// third queue with no extra build possible — the replica's other slot, or
/// all of its build memory, taken elsewhere. That capacity is then freed,
/// which by itself starts nothing (no turn is asked for and no extra build
/// ends). When the first waiter's turn is handed on, the second gets the
/// flight's turn and the third starts at once as an extra build, rather than
/// waiting for the second's build to end.
async fn extras_start_when_the_flight_turn_is_handed_on(blocked_by: ExtrasBlockedBy) {
    let what = format!("{blocked_by:?}");
    let fx = Fx::new("sf-extras-at-hand-on", Shape::slots(2)).await;
    let waiters: Vec<FlightWaiter> = (0..3).map(|_| fx.attach()).collect();
    let tasks: Vec<ReadTask> = waiters.into_iter().map(|w| spawn_read(w, ROOMY)).collect();
    let mut reads = Vec::new();
    for (i, t) in tasks.into_iter().enumerate() {
        reads.push(served(t, &format!("{what}: w{}", i + 1)).await);
    }
    let mut reads = reads.into_iter();
    let (r1, r2, r3) = (
        reads.next().unwrap(),
        reads.next().unwrap(),
        reads.next().unwrap(),
    );
    let b1 = turn(r1, &what).await;
    assert!(!b1.is_extra(), "{what}: w1 holds the flight's own turn");

    let (held_slot, held_memory) = match blocked_by {
        ExtrasBlockedBy::Slot => (
            Some(
                fx.gate()
                    .try_acquire()
                    .expect("fixture: the replica's second slot is free"),
            ),
            None,
        ),
        ExtrasBlockedBy::Memory => {
            let budget = fx.budget();
            let free = budget.total_bytes() - budget.reserved_bytes() - budget.overcommit_bytes();
            (
                None,
                Some(
                    budget
                        .try_grant(free)
                        .expect("fixture: the rest of the budget is free to take"),
                ),
            )
        }
    };
    let mut f2 = Box::pin(r2.build_turn());
    let mut f3 = Box::pin(r3.build_turn());
    assert_queued(
        f2.as_mut(),
        &format!("{what}: w2, nothing free for an extra"),
    );
    assert_queued(
        f3.as_mut(),
        &format!("{what}: w3, nothing free for an extra"),
    );

    drop((held_slot, held_memory));
    assert_queued(
        f3.as_mut(),
        &format!("{what}: w3 — freeing capacity elsewhere asks for nothing by itself"),
    );

    drop(b1);
    let b2 = ready_turn(f2.as_mut(), &format!("{what}: w2, the oldest queued"));
    assert!(!b2.is_extra(), "{what}: w2 gets the flight's own turn");
    let b3 = ready_turn(
        f3.as_mut(),
        &format!("{what}: w3, once the turn was handed on with a slot and a part free"),
    );
    assert!(
        b3.is_extra(),
        "{what}: w3 builds at once beside w2, under the freed slot and a part of its own"
    );
    assert_eq!(fx.builds_in_progress(), 2, "{what}: two builds at once");
    drop((b2, b3));
    fx.assert_idle(&what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiters_queued_with_no_slot_free_start_an_extra_build_when_the_turn_is_handed_on() {
    within_test_budget(extras_start_when_the_flight_turn_is_handed_on(
        ExtrasBlockedBy::Slot,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiters_queued_with_no_memory_free_start_an_extra_build_when_the_turn_is_handed_on()
{
    within_test_budget(extras_start_when_the_flight_turn_is_handed_on(
        ExtrasBlockedBy::Memory,
    ))
    .await;
}

/// A layer whose snapshot plus one build exceeds `T` (Spec SDD SR-7.2, PM
/// review r2 F18): the combined grant is clamped to `T` and the flight runs
/// alone — no extra build fits beside it — and every waiter completes, none
/// refused for size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_whose_snapshot_and_build_exceed_t_runs_clamped_and_alone_with_no_refusal() {
    within_test_budget(async {
        const T_KIB: u64 = 4;
        let fx = Fx::new(
            "sf-clamped",
            Shape {
                budget_kib: Some(T_KIB),
                ..Shape::DEFAULT
            },
        )
        .await;
        assert!(
            kib_for_bytes(flight_grant_bytes(fx.estimate.bytes)) > T_KIB,
            "fixture: the combined grant must exceed T"
        );
        let tasks: Vec<ReadTask> = (0..3).map(|_| spawn_read(fx.attach(), ROOMY)).collect();
        let mut reads = Vec::new();
        for (i, t) in tasks.into_iter().enumerate() {
            reads.push(served(t, &format!("waiter {i} of a clamped flight")).await);
        }
        assert_eq!(fx.reads("ok"), 1);

        for r in reads {
            let b = turn(r, "a clamped flight's build").await;
            assert!(!b.is_extra(), "nothing fits beside a grant of the whole T");
            assert_eq!(
                b.memory.bytes(),
                T_KIB * 1024,
                "the clamped grant holds exactly T"
            );
            drop(b);
        }
        fx.assert_idle("after the clamped flight").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Two waves (PM review r3 F30, r4 F38)
// ═════════════════════════════════════════════════════════════════════════════

/// A second wave arriving after the first flight's release: its first
/// catch-up creates the successor, the rest join it (open while `Queued`);
/// none starts reading while the first flight's builds go on; each wave
/// shares one read. Once the first flight's read resolved, `lookup` reports
/// the layer's measured size (what the next catch-up sizes from) in place of
/// the estimate it was created with.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_second_wave_after_the_release_succeeds_the_first_and_shares_one_read() {
    within_test_budget(async {
        let fx = Fx::new("sf-two-waves", Shape::DEFAULT).await;
        let (a1, a2) = (fx.attach(), fx.attach());
        let (ta1, ta2) = (spawn_read(a1, ROOMY), spawn_read(a2, ROOMY));
        let (ra1, ra2) = (served(ta1, "a1").await, served(ta2, "a2").await);
        let measured = ra1.snapshot().measured_size();
        assert_ne!(
            measured, fx.estimate,
            "fixture: the read measures a size other than the TOO_LARGE estimate"
        );
        assert_eq!(
            fx.flights.lookup(fx.oid).map(|l| l.size),
            Some(measured),
            "a resolved flight reports the size its read measured"
        );
        let held = turn(ra1, "a1").await;
        assert_eq!(
            fx.registered(),
            Some((false, FlightState::Reading)),
            "fixture: the first flight is closed and still building"
        );

        let wave: Vec<FlightWaiter> = (0..3).map(|_| fx.attach()).collect();
        assert_eq!(
            wave.iter().map(FlightWaiter::role).collect::<Vec<_>>(),
            vec![
                FlightRole::Leader,
                FlightRole::JoinedRead,
                FlightRole::JoinedRead
            ],
            "the wave's first creates the successor and the rest join it"
        );
        assert_eq!(fx.registered(), Some((true, FlightState::Queued)));
        let tasks: Vec<ReadTask> = wave.into_iter().map(|w| spawn_read(w, ROOMY)).collect();
        tokio::time::sleep(QUIET).await;
        assert!(
            tasks.iter().all(|t| !t.is_finished()),
            "the second wave waits for the first flight's builds"
        );
        assert_eq!(fx.reads("ok"), 1, "and has not read");

        drop(held);
        drop(turn(ra2, "a2").await);
        let mut reads = Vec::new();
        for (i, t) in tasks.into_iter().enumerate() {
            reads.push(served(t, &format!("second wave {i}")).await);
        }
        assert_one_snapshot(&reads.iter().collect::<Vec<_>>(), "the second wave");
        assert_eq!(fx.reads("ok"), 2, "one read per wave");
        assert_eq!(
            (fx.roles("leader"), fx.roles("joined_read")),
            (2, 3),
            "a leader per wave"
        );
        build_all(reads, "second wave").await;
        fx.assert_idle("after both waves").await;
    })
    .await;
}

/// The predecessor wait is uncharged and uncapped (PM review r4 F38): the
/// first flight's build chain is held longer than `reader_wait_max(L)`, with
/// the second wave's capacity budgets far shorter than that; the second wave
/// still completes, refused for neither, sharing one read.
///
/// Slow by necessity: the chain is held for `reader_wait_max(L)` + 2 s
/// (~29 s) of real time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_second_wave_waits_uncharged_and_uncapped_for_a_build_chain_past_reader_wait_max() {
    within_test_budget(async {
        const SHORT_BUDGET: Duration = Duration::from_secs(1);
        let fx = Fx::new("sf-long-chain", Shape::DEFAULT).await;
        let cap = reader_wait_max(&fx.client, fx.estimate);
        assert!(
            cap < Duration::from_secs(45),
            "fixture: reader_wait_max(L) {cap:?} must fit the test budget"
        );
        let first = served(spawn_read(fx.attach(), ROOMY), "the first wave").await;
        let chain = turn(first, "the first wave's build").await;

        let (s1, s2) = (fx.attach(), fx.attach());
        assert_eq!(
            (s1.role(), s2.role()),
            (FlightRole::Leader, FlightRole::JoinedRead)
        );
        let (t1, t2) = (spawn_read(s1, SHORT_BUDGET), spawn_read(s2, SHORT_BUDGET));
        // The build chain runs past the cap (a long build, held by the test).
        tokio::time::sleep(cap + Duration::from_secs(2)).await;
        assert!(
            !t1.is_finished() && !t2.is_finished(),
            "a predecessor's builds are neither charged nor capped: the second wave is still \
             waiting, not refused"
        );
        drop(chain);

        let (r1, r2) = (
            served(t1, "second wave, s1").await,
            served(t2, "second wave, s2").await,
        );
        assert_one_snapshot(&[&r1, &r2], "the second wave");
        assert_eq!(fx.reads("ok"), 2, "one read per wave");
        build_all(vec![r1, r2], "second wave").await;
        fx.assert_idle("after both waves").await;
    })
    .await;
}

/// A displaced predecessor goes back (PM review F1 path (a)): a successor
/// whose every waiter is cancelled while it waits for its predecessor puts
/// the still-building predecessor back in the map — `lookup` finds it,
/// closed — so the next catch-up becomes its successor and waits in the
/// `Predecessor` phase, uncharged, for its `done`, never reading beside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_whose_successor_is_abandoned_goes_back_and_its_next_successor_waits_for_its_done() {
    within_test_budget(async {
        const SHORT_BUDGET: Duration = Duration::from_millis(300);
        let fx = Fx::new("sf-put-back", Shape::DEFAULT).await;
        let first = served(spawn_read(fx.attach(), ROOMY), "the first flight").await;
        let chain = turn(first, "the first flight's build").await;
        let abandoned = fx.attach();
        assert_eq!(
            (abandoned.role(), abandoned.flight().phase()),
            (FlightRole::Leader, FlightPhase::Predecessor),
            "fixture: a successor waiting for the still-building first flight"
        );
        let abandoned = spawn_read(abandoned, ROOMY);
        tokio::task::yield_now().await;
        cancel(abandoned, "the successor's only waiter").await;

        assert_eq!(
            fx.registered(),
            Some((false, FlightState::Reading)),
            "the still-building predecessor is registered again, closed"
        );
        let next = fx.attach();
        assert_eq!(
            (next.role(), next.flight().phase()),
            (FlightRole::Leader, FlightPhase::Predecessor),
            "the next catch-up becomes the predecessor's successor and waits for its done"
        );
        let next = spawn_read(next, SHORT_BUDGET);
        tokio::time::sleep(SHORT_BUDGET * 3).await;
        assert!(
            !next.is_finished(),
            "the predecessor wait is not charged: a 300 ms budget outlasts it"
        );
        assert_eq!(fx.reads("ok"), 1, "nothing reads beside the predecessor");
        drop(chain);

        let read = served(next, "the next successor").await;
        assert_eq!(fx.reads("ok"), 2);
        drop(turn(read, "the next successor").await);
        fx.assert_idle("after both flights").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Ownership (PM review r3 F31)
// ═════════════════════════════════════════════════════════════════════════════

/// A waiter drops its build turn when its build returns, before its send: a
/// payload left unsent behind a full channel neither delays the next turn nor
/// keeps the flight's slot or memory — after the last build, with that
/// payload still unsent, `reserved == 0 && overcommit == 0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_build_turn_dropped_before_a_blocked_send_holds_up_neither_the_next_turn_nor_memory()
{
    within_test_budget(async {
        let fx = Fx::new("sf-unsent", Shape::slots(1)).await;
        let (w1, w2) = (fx.attach(), fx.attach());
        let (t1, t2) = (spawn_read(w1, ROOMY), spawn_read(w2, ROOMY));
        let (r1, r2) = (served(t1, "w1").await, served(t2, "w2").await);

        let b1 = turn(r1, "w1").await;
        let payload = (b1.cut, b1.state.sections.len());
        drop(b1);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(payload).expect("fill the channel");
        let blocked = tokio::spawn(async move { tx.send(payload).await });
        tokio::task::yield_now().await;
        assert!(!blocked.is_finished(), "fixture: w1's send is blocked");

        let mut f2 = Box::pin(r2.build_turn());
        let b2 = ready_turn(f2.as_mut(), "w2, while w1's payload is unsent");
        drop(b2);
        assert!(!blocked.is_finished(), "w1's payload is still unsent");
        fx.assert_idle("after the last build, a payload still unsent")
            .await;
        blocked.abort();
        drop(rx);
    })
    .await;
}

/// A waiter cancelled during its own turn (its task aborted mid-build) passes
/// the turn to the next; one cancelled while queued leaves the queue; the
/// flight's slot and grant stay held for the remaining build and go with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiter_cancelled_in_its_own_turn_passes_it_on_and_the_last_build_frees_slot_and_grant(
) {
    within_test_budget(async {
        let fx = Fx::new("sf-cancel-turn", Shape::slots(1)).await;
        let (w1, w2, w3) = (fx.attach(), fx.attach(), fx.attach());
        let (t1, t2, t3) = (
            spawn_read(w1, ROOMY),
            spawn_read(w2, ROOMY),
            spawn_read(w3, ROOMY),
        );
        let (r1, r2, r3) = (
            served(t1, "w1").await,
            served(t2, "w2").await,
            served(t3, "w3").await,
        );

        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let builder = tokio::spawn(async move {
            let _turn = r1.build_turn().await;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        tokio::time::timeout(SIGNAL_BUDGET, started_rx)
            .await
            .expect("w1 starts building")
            .expect("w1's builder signals");
        let mut f2 = Box::pin(r2.build_turn());
        let mut f3 = Box::pin(r3.build_turn());
        assert_queued(f2.as_mut(), "w2");
        assert_queued(f3.as_mut(), "w3");

        builder.abort();
        assert!(builder.await.unwrap_err().is_cancelled());
        let b2 = ready_turn(f2.as_mut(), "w2, after w1 was cancelled mid-build");
        drop(f3);
        assert_eq!(
            fx.gate().available(),
            0,
            "the flight's slot stays for the remaining build"
        );
        assert!(
            fx.budget().reserved_bytes() > 0,
            "and so does its grant, owned by the snapshot w2 holds"
        );
        drop(b2);
        fx.assert_idle("after the last remaining build").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Progress (PM review r3 F37)
// ═════════════════════════════════════════════════════════════════════════════

/// The progress rule: two above-ceiling layers on one replica, two slots,
/// `P_f` = 1 and `T` exactly one combined grant. One flight reads while the
/// other waits for the permit and then for memory, charged; the first
/// flight's short builds free its grant, and both flights complete with no
/// refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_above_ceiling_flights_with_two_slots_one_fenced_permit_and_one_grant_both_complete() {
    within_test_budget(async {
        let probe = Fx::new("sf-progress-probe", Shape::DEFAULT).await;
        let t_kib = kib_for_bytes(flight_grant_bytes(probe.estimate.bytes));
        drop(probe);
        let fx = Fx::new(
            "sf-progress",
            Shape {
                slots: 2,
                fenced: 1,
                budget_kib: Some(t_kib),
            },
        )
        .await;
        let other = create(
            &fx.ls,
            &fx.session_id,
            "sf-progress-b",
            &every_shape_state(),
        )
        .await;
        let other_estimate = estimate_of(&fx.client, fx.keys(), other).await;
        assert!(
            kib_for_bytes(flight_grant_bytes(other_estimate.bytes)) <= t_kib
                && kib_for_bytes(flight_grant_bytes(fx.estimate.bytes)) <= t_kib,
            "fixture: T fits one combined grant of either layer"
        );

        let mut waiters = Vec::new();
        for _ in 0..2 {
            waiters.push(fx.attach());
            waiters.push(fx.flights.attach_or_create(other, other_estimate));
        }
        assert_eq!(fx.flights.len(), 2);
        let tasks: Vec<JoinHandle<Result<u64, FlightWaitError>>> = waiters
            .into_iter()
            .map(|w| {
                tokio::spawn(async move {
                    let budget = CapacityBudget::new(ROOMY);
                    let read = w.read(&budget).await?;
                    let turn = read.build_turn().await;
                    let cut = turn.cut;
                    tokio::task::yield_now().await;
                    drop(turn);
                    Ok(cut)
                })
            })
            .collect();
        for (i, t) in tasks.into_iter().enumerate() {
            let out = tokio::time::timeout(READ_BUDGET, t)
                .await
                .unwrap_or_else(|_| panic!("catch-up {i} did not complete: no progress"))
                .expect("the catch-up task must not fail");
            assert!(out.is_ok(), "catch-up {i} must not be refused: {out:?}");
        }
        assert_eq!(fx.reads("ok"), 2, "one read per layer");
        fx.assert_idle("after both flights").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Per-waiter charging (SR-7.3, PM review r2 F20; SR-5.5, SR-11.4)
// ═════════════════════════════════════════════════════════════════════════════

/// Two waiters on one flight held in a capacity wait (`at`: its slot, its
/// fenced permit or its memory), with different capacity budgets: the poorer
/// runs out and leaves alone, `CapacityExhausted` (the caller's retryable
/// `UNAVAILABLE`); the flight carries on, attachable, and serves the other
/// once the capacity frees. Each of the flight's three capacity waits is
/// charged, not just the first.
async fn poorer_waiter_refused_at(at: At) {
    let fx = Fx::new("sf-charge", shape_for(at)).await;
    let what = format!("a capacity wait at {at:?}");
    let blocker = block_at(&fx, at).await;
    let (poor, rich) = (fx.attach(), fx.attach());
    let poor = spawn_read(poor, Duration::from_millis(300));
    let rich = spawn_read(rich, ROOMY);
    until_waiting_at(&fx, at, &what).await;

    let refused = joined(poor, &what).await;
    assert!(
        matches!(refused, Err(FlightWaitError::CapacityExhausted)),
        "{what}: the poorer waiter's budget ran out in the flight's capacity wait: {refused:?}"
    );
    assert_eq!(
        fx.registered(),
        Some((true, FlightState::Queued)),
        "{what}: the flight carries on for the other"
    );
    drop(blocker);
    let read = served(rich, &what).await;
    drop(turn(read, &what).await);
    fx.assert_idle(&what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_held_in_its_slot_wait_refuses_its_poorer_waiter_alone_and_serves_the_other() {
    within_test_budget(poorer_waiter_refused_at(At::SlotWait)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_held_in_its_fenced_permit_wait_refuses_its_poorer_waiter_alone_and_serves_the_other(
) {
    within_test_budget(poorer_waiter_refused_at(At::PermitWait)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_held_in_its_memory_wait_refuses_its_poorer_waiter_alone_and_serves_the_other() {
    within_test_budget(poorer_waiter_refused_at(At::MemoryWait)).await;
}

/// A fence sleep is work, not a capacity wait: waiters whose capacity
/// budgets are far shorter than the writers' gap the flight sleeps through
/// are served, not refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiters_are_not_charged_for_the_fence_sleeps_the_flight_works_through() {
    within_test_budget(async {
        const BUDGET: Duration = Duration::from_millis(200);
        let fx = Fx::new("sf-work-uncharged", Shape::DEFAULT).await;
        fx.stage_gap(SHORT_GAP_MS).await;
        let (t1, t2) = (
            spawn_read(fx.attach(), BUDGET),
            spawn_read(fx.attach(), BUDGET),
        );
        let (r1, r2) = (served(t1, "w1").await, served(t2, "w2").await);
        assert!(
            fx.acquires("retry_after_gap") >= 1,
            "fixture: the flight slept through the gap"
        );
        build_all(vec![r1, r2], "uncharged").await;
        fx.assert_idle("after the flight").await;
    })
    .await;
}

/// The reader cap (Spec SDD SR-5.5, SR-11.4; code review r1 F3): a flight
/// sleeps through a writers' gap staged to end [`CAP_MARGIN`] after the first
/// waiter's `reader_wait_max(L)`. The first waiter leaves alone at the cap
/// with `ReaderWaitCap`, which tells it the work it counted and the cap. Its
/// ending is NOT observed on
/// `live_edit_snapshot_fence_reader_wait_seconds{consumer="catchup"}`
/// (defer-13's units decision: that histogram is one observation per read
/// attempt; the caller counts the ending — a catch-up's
/// `stream_exits_total{reason="fenced_read_wait_cap"}`, pinned in
/// `grpc::subscribe_fenced_catch_up_test`), and the flight's own attempt is
/// still waiting and has recorded nothing either. The second waiter, attached
/// [`OFFSET`] later, stays; the gap ends inside its cap and it is served, and
/// the flight's attempt is then observed once — the histogram's only
/// observation.
///
/// Slow by necessity: the first waiter runs out a real `reader_wait_max(L)`
/// (~27 s), and the gap ends a few seconds after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiter_past_reader_wait_max_leaves_alone_and_only_the_read_attempt_is_observed() {
    within_test_budget(async {
        let fx = Fx::new("sf-reader-cap", Shape::DEFAULT).await;
        let cap = reader_wait_max(&fx.client, fx.estimate);
        assert!(
            cap < Duration::from_secs(45),
            "fixture: reader_wait_max(L) {cap:?} must fit the test budget"
        );
        fx.stage_gap((cap + CAP_MARGIN).as_millis() as i64).await;
        let first = spawn_read(fx.attach(), ROOMY);
        // The second waiter's work starts later, so it is still inside its
        // cap when the first runs out.
        tokio::time::sleep(OFFSET).await;
        let late = fx.attach();
        assert_eq!(late.role(), FlightRole::JoinedRead);
        let second = spawn_read(late, ROOMY);

        let ended = joined_within(first, CAP_READ_BUDGET, "the first waiter").await;
        match ended {
            Err(FlightWaitError::ReaderWaitCap { waited, cap: told }) => {
                assert_eq!(told, cap, "the cap for the flight's size");
                assert!(waited >= cap, "it left at the cap: waited {waited:?}");
            }
            other => panic!("the first waiter must leave at reader_wait_max(L); got {other:?}"),
        }
        assert_eq!(
            fx.reader_waits(),
            0,
            "the waiter's cap ending is not a read attempt's lease wait, so it is not observed; \
             the flight's attempt is still waiting and has recorded nothing"
        );
        assert_eq!(
            fx.registered(),
            Some((true, FlightState::Queued)),
            "the flight carries on for the second waiter"
        );

        let read = served(second, "the second waiter").await;
        drop(turn(read, "the second waiter").await);
        assert_eq!(
            fx.reader_waits(),
            1,
            "the flight's one read attempt, observed at its grant, is the only observation"
        );
        fx.assert_idle("after the flight").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Failure and panic isolation (SR-6.5, SR-6.6)
// ═════════════════════════════════════════════════════════════════════════════

/// A `tracing` subscriber that panics, once, on the flight's `flight_read`
/// event — which the read stage emits from its fence-grant hook, inside its
/// `catch_unwind` — so a test can make the read stage panic with no seam in
/// the code under test.
struct PanicAtFlightRead {
    armed: Arc<AtomicBool>,
}

impl tracing::Subscriber for PanicAtFlightRead {
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
        if meta.target() == FLIGHT_TRACE_TARGET
            && meta.fields().field("cut").is_some()
            && self.armed.swap(false, Ordering::SeqCst)
        {
            panic!("test: the shared read stage panics at its fence grant");
        }
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// A read stage that panics (at its fence grant, holding a lease, its slot,
/// its permit and its grant) fails every attached waiter `Internal`, each
/// with its own copy; fires `done`; leaves nothing held and no live lease;
/// and the registry stays usable — the next catch-up creates a new flight and
/// is served.
///
/// On a current-thread runtime with the panicking subscriber installed
/// thread-locally, so the spawned reads run under it.
#[tokio::test(flavor = "current_thread")]
async fn flight_whose_read_stage_panics_fails_its_waiters_internal_and_leaves_the_registry_usable()
{
    let armed = Arc::new(AtomicBool::new(true));
    let _subscriber = tracing::subscriber::set_default(PanicAtFlightRead {
        armed: Arc::clone(&armed),
    });
    within_test_budget(async {
        let fx = Fx::new("sf-panic", Shape::DEFAULT).await;
        let (w1, w2, observer) = (fx.attach(), fx.attach(), fx.attach());
        let (t1, t2) = (spawn_read(w1, ROOMY), spawn_read(w2, ROOMY));

        let (e1, e2) = (joined(t1, "w1").await, joined(t2, "w2").await);
        assert!(!armed.load(Ordering::SeqCst), "fixture: the read stage panicked");
        for (who, e) in [("w1", &e1), ("w2", &e2)] {
            assert!(
                matches!(e, Err(FlightWaitError::Read(ServiceError::Internal(m))) if m.contains("panicked")),
                "{who}: a panicking read stage is an Internal failure for every waiter: {e:?}"
            );
        }
        assert!(
            observer.flight().is_done(),
            "a panicked flight fires done"
        );
        assert!(fx.flights.is_empty(), "a failed flight leaves the map");
        drop(observer);
        fx.assert_idle("after the panic").await;

        let next = fx.attach();
        assert_eq!(next.role(), FlightRole::Leader, "a new flight, not the panicked one");
        let read = served(spawn_read(next, ROOMY), "after the panic").await;
        drop(turn(read, "after the panic").await);
        fx.assert_idle("after the next flight").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. Cancellation at every await (SR-7.4)
// ═════════════════════════════════════════════════════════════════════════════

/// Where a flight is when its waiters are cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    /// Waiting for its predecessor's `done` (the predecessor's build held).
    Predecessor,
    /// Waiting for its catch-up slot (the only slot held by the test).
    SlotWait,
    /// Holding its slot, waiting for its fenced permit (held by the test).
    PermitWait,
    /// Holding its slot and permit, waiting for memory (`T` held by the
    /// test).
    MemoryWait,
    /// In a `RETRY_AFTER` sleep (a writers' gap staged), holding only its
    /// slot.
    RetrySleep,
    /// Reading between grant and release (the fan-out held by the delay
    /// seam), holding a lease, its slot, permit and grant.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    Reading,
}

/// The fixture shape a flight needs to wait `at` a step.
fn shape_for(at: At) -> Shape {
    match at {
        At::SlotWait | At::PermitWait => Shape::slots(1),
        At::MemoryWait => Shape {
            budget_kib: Some(1024),
            ..Shape::DEFAULT
        },
        _ => Shape::DEFAULT,
    }
}

/// The capacity the test holds so a flight waits at one of its capacity
/// steps ([`block_at`]); dropping it lets the flight on.
#[derive(Default)]
struct Blocker {
    _slot: Option<CatchUpPermit>,
    _permit: Option<CatchUpReadPermit>,
    _memory: Option<MemoryGrant>,
}

/// Take the capacity a flight then waits for at `at`: the only slot
/// (`SlotWait`), the only fenced permit (`PermitWait`) or the whole budget
/// (`MemoryWait`). Nothing for the other steps.
async fn block_at(fx: &Fx, at: At) -> Blocker {
    let mut blocker = Blocker::default();
    match at {
        At::SlotWait => {
            blocker._slot = Some(
                fx.gate()
                    .try_acquire()
                    .expect("the test takes the only slot"),
            )
        }
        At::PermitWait => {
            blocker._permit = Some(
                fx.fenced()
                    .acquire(
                        Instant::now() + Duration::from_secs(3_600),
                        SnapshotReadConsumer::Catchup,
                    )
                    .await
                    .expect("the test takes the only fenced permit"),
            );
        }
        At::MemoryWait => {
            blocker._memory = Some(
                fx.budget()
                    .reserve(u64::MAX, None)
                    .await
                    .expect("the test takes the whole budget"),
            );
        }
        _ => {}
    }
    blocker
}

/// Poll until the fixture's flight is parked in the capacity wait `at`
/// names (after [`block_at`]).
async fn until_waiting_at(fx: &Fx, at: At, what: &str) {
    match at {
        At::SlotWait => until(what, || fx.slot_waiters() == 1).await,
        At::PermitWait => {
            until(what, || {
                fx.gate().available() == 0
                    && fx.metric(
                        "live_edit_subscribe_catchup_wait_seconds_count",
                        &[("result", "acquired")],
                    ) == 1.0
            })
            .await
        }
        At::MemoryWait => until(what, || fx.budget().waiters() == 1).await,
        other => panic!("{what}: {other:?} is not a capacity wait"),
    }
}

/// Two waiters on a flight that is `at` the step named. `every_waiter`:
/// cancel both — the flight ends; otherwise cancel the leader, and the other
/// drives the flight on to its snapshot and build. Then everything is
/// released, and nothing may be left held.
async fn cancelled_at(at: At, every_waiter: bool) {
    let fx = Fx::new("sf-cancel", shape_for(at)).await;
    let what = format!(
        "{} cancelled at {at:?}",
        if every_waiter {
            "every waiter"
        } else {
            "one waiter"
        }
    );

    // What the flight will wait at.
    let blocker = block_at(&fx, at).await;
    let mut predecessor = None;
    match at {
        At::Predecessor => {
            let read = served(spawn_read(fx.attach(), ROOMY), "the predecessor").await;
            predecessor = Some(turn(read, "the predecessor").await);
        }
        At::RetrySleep => {
            fx.stage_gap(if every_waiter { FAR_MS } else { SHORT_GAP_MS })
                .await
        }
        At::Reading => {
            #[cfg(feature = "test-fault-injection")]
            fx.injector().push_delay_at(
                live_edit::storage::diff_read::ObjectSnapshotReadFaultTarget::next_read(),
                HOLD,
            );
        }
        At::SlotWait | At::PermitWait | At::MemoryWait => {}
    }

    let (w1, w2) = (fx.attach(), fx.attach());
    assert_eq!(
        (w1.role(), w2.role()),
        (FlightRole::Leader, FlightRole::JoinedRead),
        "{what}: fixture"
    );
    let (t1, t2) = (spawn_read(w1, ROOMY), spawn_read(w2, ROOMY));

    // Reach the step.
    match at {
        At::Predecessor => {
            tokio::time::sleep(QUIET).await;
            assert!(
                !t1.is_finished() && !t2.is_finished(),
                "{what}: waiting for the predecessor"
            );
        }
        At::SlotWait | At::PermitWait | At::MemoryWait => until_waiting_at(&fx, at, &what).await,
        At::RetrySleep => {
            until(&what, || {
                fx.acquires("retry_after_gap") >= 1
                    && fx.fenced().available() == 1
                    && fx.budget().reserved_bytes() == 0
            })
            .await
        }
        At::Reading => {
            until(&what, || {
                fx.registered() == Some((true, FlightState::Reading))
            })
            .await
        }
    }

    cancel(t1, &what).await;
    let survivor = if every_waiter {
        cancel(t2, &what).await;
        None
    } else {
        Some(t2)
    };
    if every_waiter {
        if at != At::Predecessor {
            until(&format!("{what}: the flight left the map"), || {
                fx.flights.is_empty()
            })
            .await;
        }
        // Nothing the cancelled flight took is still held once only the
        // test's own capacity is.
        if at == At::MemoryWait {
            assert_eq!(
                fx.budget().waiters(),
                0,
                "{what}: the memory wait left the queue"
            );
        }
    }

    drop((blocker, predecessor));
    if let Some(t2) = survivor {
        let read = served(t2, &what).await;
        drop(turn(read, &what).await);
    }
    fx.assert_idle(&what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_one_waiter_cancelled_in_the_predecessor_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::Predecessor, false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_every_waiter_cancelled_in_the_predecessor_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::Predecessor, true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_one_waiter_cancelled_in_the_slot_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::SlotWait, false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_every_waiter_cancelled_in_the_slot_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::SlotWait, true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_one_waiter_cancelled_in_the_fenced_permit_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::PermitWait, false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_every_waiter_cancelled_in_the_fenced_permit_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::PermitWait, true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_one_waiter_cancelled_in_the_memory_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::MemoryWait, false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_every_waiter_cancelled_in_the_memory_wait_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::MemoryWait, true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_one_waiter_cancelled_in_a_retry_after_sleep_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::RetrySleep, false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_every_waiter_cancelled_in_a_retry_after_sleep_leaves_nothing_held() {
    within_test_budget(cancelled_at(At::RetrySleep, true)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. Replica shutdown (`ValkeyClient::quit` closes the gates and the budget)
// ═════════════════════════════════════════════════════════════════════════════

/// A flight waiting `at` a capacity step when shutdown closes what it waits
/// for — the catch-up gate, the fenced read gate or the memory budget, as
/// `ValkeyClient::quit` does — ends `ShuttingDown` for every waiter; it fires
/// `done`, leaves the map, and leaves nothing held.
async fn shut_down_at(at: At) {
    let fx = Fx::new("sf-shutdown", shape_for(at)).await;
    let what = format!("shutdown at {at:?}");
    let blocker = block_at(&fx, at).await;
    let (w1, w2, observer) = (fx.attach(), fx.attach(), fx.attach());
    let (t1, t2) = (spawn_read(w1, ROOMY), spawn_read(w2, ROOMY));
    until_waiting_at(&fx, at, &what).await;

    match at {
        At::SlotWait => fx.gate().close(),
        At::PermitWait => fx.fenced().close(),
        At::MemoryWait => fx.budget().close(),
        other => panic!("{what}: {other:?} is not a capacity wait"),
    }

    for (who, t) in [("w1", t1), ("w2", t2)] {
        let ended = joined(t, &format!("{what}: {who}")).await;
        assert!(
            matches!(ended, Err(FlightWaitError::ShuttingDown)),
            "{what}: {who} ends ShuttingDown: {ended:?}"
        );
    }
    assert!(observer.flight().is_done(), "{what}: the flight fires done");
    assert!(fx.flights.is_empty(), "{what}: the flight leaves the map");
    drop((observer, blocker));
    fx.assert_idle(&what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiting_for_its_slot_at_shutdown_ends_shutting_down_for_every_waiter() {
    within_test_budget(shut_down_at(At::SlotWait)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiting_for_its_fenced_permit_at_shutdown_ends_shutting_down_for_every_waiter() {
    within_test_budget(shut_down_at(At::PermitWait)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_waiting_for_memory_at_shutdown_ends_shutting_down_for_every_waiter() {
    within_test_budget(shut_down_at(At::MemoryWait)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 10. Seams (`test-fault-injection`): a read held open, broken, or failed
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod seams {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::{ObjectSnapshotReadFaultTarget, ObjectSnapshotReader};

    use crate::storage::snapshot_read_atomic_test::mock_ext_store;

    /// Hold the fixture flight's next read open for [`HOLD`] after its grant:
    /// a delay on the first command of the next fenced fan-out, which only
    /// the flight's read issues. Not `next_read`, which a `DeleteLayer` or
    /// overwrite cleanup started while the flight reads would also match: its
    /// retirement size read consults the seam as a lone first command (b.fsz
    /// MR-D4), and taking the delay there would let the flight's read run
    /// unheld past the retirement the test stages under it.
    fn hold_next_read(fx: &Fx) {
        fx.injector().push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            HOLD,
        );
    }

    async fn until_reading(fx: &Fx, what: &str) {
        until(what, || {
            fx.registered() == Some((true, FlightState::Reading))
        })
        .await;
    }

    /// Regression (post-completion review F4 (i)): two waiters of one flight
    /// cancelled together — as a session deletion or a replica drain ends
    /// both catch-ups — while the flight is stopped just after its release
    /// was sent, holding its slot, fenced permit and grant. The flights'
    /// leave seam (`SnapshotFlights::arm_leave_seam_for_test`) parks the
    /// first waiter's drop after its count-out, on an OS thread of its own;
    /// the second then counts out last and fires `done`. The successor woken
    /// by `done` must find all three back: the parked waiter's own clone of
    /// the read stage was dropped before it counted out, so the second
    /// waiter's drop held the last clone. With the drop dropping its clone
    /// after counting out, the parked waiter still holds it as `done` fires.
    ///
    /// The parked thread enters the scenario's runtime, so a lease guard
    /// dropped there could still spawn its release; the test's thread, that
    /// runtime's only one, blocks meanwhile, so the release runs after the
    /// join. Results are read before the seam is released and asserted after
    /// the thread is joined, so a failed assertion never leaves the thread
    /// parked.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_cancelled_by_two_waiters_at_once_frees_slot_permit_and_grant_before_its_successor_wakes(
    ) {
        hand_polled("sf-drop-order-two", Shape::slots(1), |fx| async move {
            let budget = CapacityBudget::new(ROOMY);
            let (first, second) = (fx.attach(), fx.attach());
            assert_eq!(
                (first.role(), second.role()),
                (FlightRole::Leader, FlightRole::JoinedRead),
                "both waiters share one flight"
            );
            let mut leader = Box::pin(first.read(&budget));
            fx.drive_until_closed(leader.as_mut()).await;
            let successor_budget = CapacityBudget::new(ROOMY);
            let (probe, next) = probed_successor(&fx, &successor_budget, "the successor");
            let seam = fx.flights.arm_leave_seam_for_test(fx.oid);
            let runtime = tokio::runtime::Handle::current();

            let (parked, parked_was_last, woken_while_parked, seen) = std::thread::scope(|scope| {
                let runtime = &runtime;
                let first_leaving = scope.spawn(move || {
                    let _runtime = runtime.enter();
                    drop(leader);
                });
                let parked = seam.wait_until_parked(SIGNAL_BUDGET);
                let parked_was_last = seam.parked_was_last();
                let woken_while_parked = probe.seen();
                // A missed park leaves the seam armed: spend it now, so the
                // second waiter's leave below passes through instead of
                // parking the test's own thread with nothing to release it.
                if !parked {
                    seam.release();
                }
                drop(second);
                let seen = probe.seen();
                seam.release();
                first_leaving
                    .join()
                    .expect("the parked drop returns once released");
                (parked, parked_was_last, woken_while_parked, seen)
            });

            assert!(parked, "the first waiter's leave parked at the seam within {SIGNAL_BUDGET:?}");
            assert_eq!(
                parked_was_last,
                Some(false),
                "the parked waiter counted out, but not as the flight's last"
            );
            assert!(
                woken_while_parked.is_empty(),
                "done has not fired while the second waiter is still in: {woken_while_parked:?}"
            );
            assert_eq!(
                seen.first().copied(),
                Some(ALL_FREE_AT_ONE_SLOT),
                "when the second waiter's leaving fired done, the slot, fenced permit and grant must \
                 already have been returned, with the first waiter still parked after its count-out \
                 (every wake: {seen:?})"
            );
            assert_eq!(seam.parks(), 1, "the seam is one-shot");
            drop(next);
            until("the registry emptied", || fx.flights.is_empty()).await;
            fx.assert_idle("after both waiters and the successor").await;
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_one_waiter_cancelled_during_the_read_leaves_nothing_held() {
        within_test_budget(cancelled_at(At::Reading, false)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_every_waiter_cancelled_during_the_read_leaves_nothing_held() {
        within_test_budget(cancelled_at(At::Reading, true)).await;
    }

    /// The round-5 finding (Spec SDD SR-6.2): a waiter arriving between the
    /// fence grant and the release attaches — no commit lands during the
    /// epoch, so its pin equals the cut — and the wave costs one epoch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_waiter_arriving_between_grant_and_release_joins_and_the_wave_costs_one_epoch() {
        within_test_budget(async {
            let fx = Fx::new("sf-join-reading", Shape::DEFAULT).await;
            hold_next_read(&fx);
            let first = spawn_read(fx.attach(), ROOMY);
            until_reading(&fx, "the read holds its lease").await;

            let pin = fx.seq().await;
            let late = fx.attach();
            assert_eq!(
                late.role(),
                FlightRole::JoinedRead,
                "a waiter arriving while the flight reads under its fence joins it"
            );
            let late = spawn_read(late, ROOMY);
            let (r1, r2) = (served(first, "first").await, served(late, "late").await);

            assert_eq!(
                assert_one_snapshot(&[&r1, &r2], "grant-to-release join"),
                pin,
                "the late waiter's pin, read during the epoch, equals the cut"
            );
            assert_eq!(
                (
                    fx.acquires("started_epoch"),
                    fx.acquires("joined_epoch"),
                    fx.reads("ok")
                ),
                (1, 0, 1),
                "one epoch, one read for the wave"
            );
            build_all(vec![r1, r2], "grant-to-release join").await;
            fx.assert_idle("after the flight").await;
        })
        .await;
    }

    /// A `BROKEN` read (the header moved under the pass) is retried in the
    /// same flight, which reopens — `Queued`, attachable — for the retry: a
    /// waiter arriving after the broken attempt joins it, and both are served
    /// the retry's snapshot at the new `seq`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_broken_read_reopens_its_flight_and_retries_it_for_every_waiter() {
        within_test_budget(async {
            let fx = Fx::new("sf-broken", Shape::DEFAULT).await;
            // Hold the first attempt's pass, and the retry's, open: command 0
            // of each attempt.
            let target =
                ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0);
            fx.injector()
                .push_delay_at(target, HOLD)
                .push_delay_at(target, HOLD);
            let first = spawn_read(fx.attach(), ROOMY);
            until_reading(&fx, "the first attempt holds its lease").await;
            fx.set_seq(fx.seq().await + 1).await;
            let moved = fx.seq().await;

            until("the broken attempt ends and the flight reopens", || {
                fx.reads("fence_broken") == 1
                    && fx.registered().is_some_and(|(attachable, _)| attachable)
            })
            .await;
            let late = fx.attach();
            assert_eq!(
                late.role(),
                FlightRole::JoinedRead,
                "a broken read reopens its flight for the retry"
            );
            let late = spawn_read(late, ROOMY);
            let (r1, r2) = (served(first, "first").await, served(late, "late").await);

            assert_eq!(
                assert_one_snapshot(&[&r1, &r2], "after the retry"),
                moved,
                "the retry is proven at the moved seq"
            );
            assert_eq!(
                (fx.reads("fence_broken"), fx.reads("ok"), fx.roles("leader")),
                (1, 1, 1),
                "one flight, one broken attempt and its retry"
            );
            build_all(vec![r1, r2], "after the retry").await;
            fx.assert_idle("after the flight").await;
        })
        .await;
    }

    /// A failed read (a transient fan-out failure) is delivered to every
    /// waiter as its own copy of the one error; the failed flight leaves the
    /// map at once — while a waiter still holds it — and is never reused: the
    /// next catch-up creates a new flight, which is served.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_failed_read_leaves_the_map_at_once_and_is_never_reused() {
        within_test_budget(async {
            let fx = Fx::new("sf-failed", Shape::DEFAULT).await;
            let seam = fx.injector();
            seam.push(FredError::new(
                FredErrorKind::Timeout,
                "injected fenced timeout",
            ));
            let (w1, w2) = (fx.attach(), fx.attach());

            let e1 = joined(spawn_read(w1, ROOMY), "w1").await;
            assert_eq!(seam.remaining(), 0, "fixture: the read consumed the fault");
            assert!(
                matches!(&e1, Err(FlightWaitError::Read(ServiceError::Aborted(_)))),
                "the transient failure, classification kept: {e1:?}"
            );
            assert!(w2.flight().is_done(), "a failed flight fires done");
            assert!(
                fx.flights.is_empty(),
                "a failed flight leaves the map at once, while w2 still holds it"
            );

            let next = fx.attach();
            assert_eq!(
                next.role(),
                FlightRole::Leader,
                "a failed flight is never reused"
            );
            let read = served(spawn_read(next, ROOMY), "the retry").await;

            let e2 = joined(spawn_read(w2, ROOMY), "w2").await;
            match (&e1, &e2) {
                (
                    Err(FlightWaitError::Read(ServiceError::Aborted(m1))),
                    Err(FlightWaitError::Read(ServiceError::Aborted(m2))),
                ) => assert_eq!(m1, m2, "w2 gets its own copy of the same error"),
                _ => panic!("w2 must get the failed flight's error, not the retry: {e2:?}"),
            }
            drop(turn(read, "the retry").await);
            fx.assert_idle("after the retry").await;
        })
        .await;
    }

    /// Overwrite during a fenced flight: the layer's new object (a new
    /// `object_id`) gets a flight of its own — a catch-up of it never joins
    /// the old object's — and is served its own state at or past its pin.
    /// The old flight's waiter gets the old object's state, or `NotFound` if
    /// the overwrite's cleanup took it first; never the new object's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_of_an_overwritten_layer_is_not_joined_by_a_catch_up_of_its_new_object() {
        within_test_budget(async {
            let fx = Fx::new("sf-overwrite", Shape::DEFAULT).await;
            hold_next_read(&fx);
            let old = spawn_read(fx.attach(), ROOMY);
            until_reading(&fx, "the old object's read").await;

            let new_oid = create(&fx.ls, &fx.session_id, &fx.layer, &every_shape_state()).await;
            assert_ne!(new_oid, fx.oid, "fixture: the overwrite made a new object");
            let new_estimate = estimate_of(&fx.client, fx.keys(), new_oid).await;
            let pin = fx.seq_of(new_oid).await;
            let newcomer = fx.flights.attach_or_create(new_oid, new_estimate);
            assert_eq!(
                newcomer.role(),
                FlightRole::Leader,
                "the new object has a flight of its own"
            );
            assert_eq!(fx.flights.len(), 2);

            let new_read = served(spawn_read(newcomer, ROOMY), "the new object").await;
            assert_eq!(new_read.snapshot().state.object_id, new_oid);
            assert!(
                new_read.snapshot().cut >= pin,
                "its own cut ≥ pin check holds"
            );
            match joined(old, "the old object").await {
                Ok(read) => {
                    assert_eq!(read.snapshot().state.object_id, fx.oid);
                    drop(turn(read, "the old object").await);
                }
                Err(FlightWaitError::Read(ServiceError::NotFound(_))) => {}
                Err(e) => panic!("the old object's waiter: its state or NotFound; got {e:?}"),
            }
            drop(turn(new_read, "the new object").await);
            fx.assert_idle("after both flights").await;
        })
        .await;
    }

    /// Delete and recreate during a fenced flight: the delete under the read
    /// ends it `NotFound` for every waiter (each its own copy); the flight
    /// leaves the map; the recreated layer's object gets a new flight.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_of_a_layer_deleted_and_recreated_under_it_fails_not_found_and_the_new_object_reads_fresh(
    ) {
        within_test_budget(async {
            let fx = Fx::new("sf-delete", Shape::DEFAULT).await;
            hold_next_read(&fx);
            let w1 = spawn_read(fx.attach(), ROOMY);
            until_reading(&fx, "the read").await;
            let w2 = spawn_read(fx.attach(), ROOMY);

            fx.ls
                .delete_layer(&fx.session_id, &fx.layer, mock_ext_store())
                .await
                .expect("DeleteLayer must succeed under the read");
            let new_oid = create(&fx.ls, &fx.session_id, &fx.layer, &every_shape_state()).await;
            assert_ne!(new_oid, fx.oid);

            for (who, task) in [("w1", w1), ("w2", w2)] {
                let e = joined(task, who).await;
                assert!(
                    matches!(e, Err(FlightWaitError::Read(ServiceError::NotFound(_)))),
                    "{who}: the deleted object's read is NotFound: {e:?}"
                );
            }
            assert!(fx.flights.is_empty(), "the failed flight left the map");

            let new_estimate = estimate_of(&fx.client, fx.keys(), new_oid).await;
            let pin = fx.seq_of(new_oid).await;
            let fresh = fx.flights.attach_or_create(new_oid, new_estimate);
            assert_eq!(fresh.role(), FlightRole::Leader);
            let read = served(spawn_read(fresh, ROOMY), "the recreated object").await;
            assert_eq!(read.snapshot().state.object_id, new_oid);
            assert!(read.snapshot().cut >= pin);
            drop(turn(read, "the recreated object").await);
            fx.assert_idle("after the recreated object's flight").await;
        })
        .await;
    }

    /// A DR restore at the same `object_id` during a fenced flight — staged
    /// as the restore residue wipe does it (header, fence and intent keys
    /// unlinked) and the header written back at an older `seq` — breaks the
    /// read, which is retried and served at the restored `seq`. Every waiter
    /// is handed that one cut to check against its own pin: a waiter pinned
    /// before the restore finds it below its pin (its check refuses it,
    /// `UNAVAILABLE` in catch-up), one pinned after finds it at its pin.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_across_a_restore_at_the_same_id_hands_every_waiter_the_restored_cut_for_its_own_pin_check(
    ) {
        within_test_budget(async {
            const BEFORE: u64 = 5;
            let fx = Fx::new("sf-restore", Shape::DEFAULT).await;
            fx.set_seq(BEFORE).await;
            let pin_before = fx.seq().await;
            hold_next_read(&fx);
            let early = spawn_read(fx.attach(), ROOMY);
            until_reading(&fx, "the read").await;

            let header: std::collections::HashMap<String, String> = fx
                .tv
                .client()
                .pool
                .hgetall(fx.header_key(fx.oid))
                .await
                .expect("HGETALL header");
            let _: i64 = fx
                .tv
                .client()
                .pool
                .unlink(vec![fx.header_key(fx.oid), fx.fence_key(), fx.intent_key()])
                .await
                .expect("the restore wipe");
            let mut restored = header;
            restored.insert("seq".to_owned(), (BEFORE - 1).to_string());
            let _: () = fx
                .tv
                .client()
                .pool
                .hset(fx.header_key(fx.oid), restored)
                .await
                .expect("the restore writes the header back");
            let pin_after = fx.seq().await;
            let late = fx.attach();
            assert_eq!(late.role(), FlightRole::JoinedRead);
            let late = spawn_read(late, ROOMY);

            let (r_early, r_late) = (served(early, "early").await, served(late, "late").await);
            let cut = assert_one_snapshot(&[&r_early, &r_late], "across the restore");
            assert_eq!(cut, BEFORE - 1, "served at the restored seq");
            assert_eq!(
                (fx.reads("fence_broken"), fx.reads("ok")),
                (1, 1),
                "the wipe broke the first attempt; the retry served"
            );
            assert!(
                cut < pin_before,
                "the waiter pinned before the restore finds the cut below its pin ({cut} < \
                 {pin_before}): its own check refuses it"
            );
            assert!(
                cut >= pin_after,
                "the waiter pinned after the restore finds the cut at its pin"
            );
            build_all(vec![r_early, r_late], "across the restore").await;
            fx.assert_idle("after the flight").await;
        })
        .await;
    }

    /// The read itself is work, never charged: a waiter whose capacity
    /// budget (200 ms) is far shorter than the read it waits for — held open
    /// for [`HOLD`] between grant and release — is served.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_waiter_is_not_charged_for_the_read_it_waits_for() {
        within_test_budget(async {
            let fx = Fx::new("sf-read-uncharged", Shape::DEFAULT).await;
            hold_next_read(&fx);
            let task = spawn_read(fx.attach(), Duration::from_millis(200));
            until_reading(&fx, "the read holds its lease").await;
            let read = served(task, "a waiter whose budget the read outlasts").await;
            drop(turn(read, "the uncharged waiter").await);
            fx.assert_idle("after the flight").await;
        })
        .await;
    }

    /// The reopen window, closed by a put-back (PM review F1 path (a)): a
    /// flight closed for its release is succeeded by a newcomer's flight
    /// before its `BROKEN` answer reopens it for the retry. When that
    /// successor's only waiter is cancelled, the reopened predecessor goes
    /// back into the map still open, `Queued`: the next catch-up joins its
    /// retry, and every waiter shares that one retried read.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_reopened_by_a_broken_read_goes_back_open_when_its_successor_is_abandoned() {
        hand_polled("sf-put-back-open", Shape::DEFAULT, |fx| async move {
            fx.injector().push_delay_at(
                ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
                HOLD,
            );
            let budget = CapacityBudget::new(ROOMY);
            let mut leader = Box::pin(fx.attach().read(&budget));
            drive_until(leader.as_mut(), "the pass is held open", || {
                fx.registered() == Some((true, FlightState::Reading))
            })
            .await;
            // A second waiter: a handle on this flight once it is displaced.
            let observer = fx.attach();
            assert_eq!(observer.role(), FlightRole::JoinedRead);
            fx.set_seq(fx.seq().await + 1).await;
            fx.drive_until_closed(leader.as_mut()).await;

            let successor = fx.attach();
            assert_eq!(
                successor.role(),
                FlightRole::Leader,
                "fixture: a newcomer after the release takes the entry"
            );
            drive_until(leader.as_mut(), "the broken read reopened it", || {
                observer.flight().state() == FlightState::Queued
                    && observer.flight().is_attachable()
            })
            .await;
            assert_eq!(
                fx.reads("fence_broken"),
                1,
                "fixture: the first attempt broke"
            );

            drop(successor);
            assert_eq!(
                fx.registered(),
                Some((true, FlightState::Queued)),
                "the reopened predecessor goes back, still open"
            );
            let joiner = fx.attach();
            assert_eq!(
                joiner.role(),
                FlightRole::JoinedRead,
                "the next catch-up joins the predecessor's retry"
            );
            let (joiner, observer) = (spawn_read(joiner, ROOMY), spawn_read(observer, ROOMY));
            let first = tokio::time::timeout(READ_BUDGET, leader)
                .await
                .expect("the predecessor's retry ends")
                .expect("the predecessor is served");
            let (joined, observed) = (
                served(joiner, "the joiner").await,
                served(observer, "the observer").await,
            );
            assert_one_snapshot(&[&first, &joined, &observed], "the retried read");
            assert_eq!(
                (fx.reads("fence_broken"), fx.reads("ok")),
                (1, 1),
                "one broken attempt and its retry; no second read"
            );
            build_all(vec![first, joined, observed], "the retried read").await;
            fx.assert_idle("after the flight").await;
        })
        .await;
    }

    /// A flight that fails while its successor holds the map entry leaves the
    /// successor in place (code review r3, PM F1): the entry still names the
    /// successor, and a newcomer joins it. A failed predecessor is over, so
    /// when the successor's waiters all leave it is never put back — even
    /// while a waiter of it is still alive — and the entry empties.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flight_failing_behind_its_successor_leaves_it_in_place_and_is_never_put_back() {
        hand_polled("sf-fail-behind", Shape::DEFAULT, |fx| async move {
            fx.injector().push(FredError::new(
                FredErrorKind::Timeout,
                "injected fenced timeout",
            ));
            let budget = CapacityBudget::new(ROOMY);
            let mut leader = Box::pin(fx.attach().read(&budget));
            // Never polled: keeps the failed flight alive after it is over.
            let held_open = fx.attach();
            assert_eq!(held_open.role(), FlightRole::JoinedRead);
            fx.drive_until_closed(leader.as_mut()).await;
            let successor = fx.attach();
            assert_eq!(successor.role(), FlightRole::Leader);

            let failed = tokio::time::timeout(READ_BUDGET, leader)
                .await
                .expect("the predecessor's read ends");
            assert!(
                matches!(
                    &failed,
                    Err(FlightWaitError::Read(ServiceError::Aborted(_)))
                ),
                "fixture: the predecessor's read failed: {failed:?}"
            );
            assert!(
                held_open.flight().is_done(),
                "fixture: the failed flight is over"
            );
            assert_eq!(
                fx.registered(),
                Some((true, FlightState::Queued)),
                "a flight failing behind its successor leaves the successor registered"
            );
            let joiner = fx.attach();
            assert_eq!(
                joiner.role(),
                FlightRole::JoinedRead,
                "a newcomer joins the successor"
            );

            drop((joiner, successor));
            assert!(
                fx.flights.is_empty(),
                "the successor's last waiter left; its failed predecessor is never put back"
            );
            drop(held_open);
            fx.assert_idle("after both flights").await;
        })
        .await;
    }
}
