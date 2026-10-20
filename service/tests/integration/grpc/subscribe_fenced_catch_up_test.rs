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

//! b.y3m Epic 3 Task 7 — catch-ups of a layer above the atomic-read limits
//! served through the layer's shared fenced-read flight
//! (`storage::snapshot_flights`), driven end-to-end through
//! `handle_subscribe_read_layer` (Spec SDD SR-3.1, SR-3.2, SR-5.5, SR-6, SR-7.3,
//! SR-7.5, SR-11.1).
//!
//! Every layer here is above the client's atomic-read ceiling ([`LOW_ITEMS`] /
//! [`LOW_BYTES`]), so every catch-up's atomic read answers `TOO_LARGE` (or is
//! skipped, when the object's flight is already registered) and the catch-up
//! attaches to the flight.
//!
//! | Area | What it proves |
//! |---|---|
//! | Two waves | at one catch-up slot, a second wave arriving during the first wave's build chain takes no slot until the first flight is done, is never charged for that chain, and is served with no refusal: one fenced read per wave |
//! | Same-replica burst | more joiners than catch-up slots share one epoch and one read, build one at a time under the flight's slot (plus an extra build when a slot and a build part are free), and are all served; one slot-wait observation per slot acquire, none for an extra build |
//! | Joinability | a catch-up whose fenced read is held open longer than its capacity budget is served (the read is work); one attaching between the fence's grant and its release, its pin read first, is served at the cut, not refused as a regression (defer-16 (i)) |
//! | Reader cap | a catch-up whose flight sleeps past `reader_wait_max(L)` ends `UNAVAILABLE` before content under `reason="fenced_read_wait_cap"`, while the flight goes on for a later one; the reader-wait histogram observes the read attempt only (defer-13) |
//! | Capacity | the flight's slot, fenced-permit and memory waits are charged to each attached catch-up and refuse it `catchup_capacity` before content, with a status naming all three waits (defer-16 (ii)), on one `FlightRead` budget charge with no deadline; waiting for memory holds no fence; a `RETRY_AFTER` sleep keeps the flight's slot, frees its permit and memory, and is not charged |
//! | Build turn | a catch-up queued for its build turn whose stream registration is killed by the multiplexer's hard cap ends before content without waiting for its turn |
//! | Sizing (code review r2) | an early-out `TOO_LARGE` sizes the flight from the fence key's stored `bpi`; a registered flight suppresses the probe; a flight that leaves the registry after an early-out answer is probed for at the next attach, the one lookup that decides the size; a rollback-mode replay handing over on an early-out probes at the attach, its slot given back; an absent, wrong-typed or failed probe falls back to the fitted bytes per item and the catch-up is still served; a mid-read abort sizes from its own tallies and sends no probe |
//! | Readers under writers (T14, INV-9) | a catch-up and a checkpoint assembly of the layer complete — served at a proven cut, written, no `fence_broken` read — while a writer commits continuously on the fast path, the cached structural route or the uncacheable WATCH fallback, each fenced read held open far longer than the writer's commit gap; the writer records its fence waits on its own path |
//! | Cancellation and failure (SR-7.4) | a catch-up ended by its client in the predecessor wait, the flight's slot, fenced-permit or memory wait, a `RETRY_AFTER` sleep, the read, the turn queue or its build turn; by shutdown in the read and in its build turn; by its session's deletion in a `RETRY_AFTER` sleep and in its build turn; and by the fenced read gate closing (`ValkeyClient::quit`'s step) while its flight waits for a permit, with the shutdown token untouched — and one whose shared read or build fails — leaves every slot, fenced and atomic permit and the memory budget at full capacity (`reserved == 0 && overcommit == 0`), no flight and no build registered, and no live reader lease once leases lapse; graceful shutdown with a flight reading under its fence releases the lease before `ValkeyClient::quit` closes the pools |
//!
//! Elsewhere: the registry itself, driven directly, in
//! `storage::snapshot_flights_test`; the regimes at the fenced cut and the
//! sequence regression on the fenced path in `subscribe_regimes_test.rs`; the
//! structural double-apply regression on the fenced path in
//! `subscribe_proven_read_test.rs`; the rollback replay's hand-over and a shared
//! read's `NotFound` recovery in `subscribe_capacity_budget_test.rs`.
//!
//! # Holding a flight still
//!
//! A flight is held where a test needs it by holding what it waits for: the
//! catch-up slot, the fenced read permit, the whole memory budget, a staged
//! writers' gap on the fence key (the acquire answers `RETRY_AFTER gap`), the
//! fan-out's delay seam (holding the read open between grant and release), a
//! flight the test registers itself and never drives, or a catch-up parked
//! inside its build turn (`CatchUpBuildTurnSeam`). Where a test must show a
//! wait is NOT charged, it spends real time past the capacity budget
//! ([`SPEND`]) — a spend, never a synchronisation: the outcome is the same
//! however long it runs past.
//!
//! Three tests rest on a window instead, each seconds wide against steps that
//! take milliseconds, and each fails loudly, never green, if a step misses it:
//! the joinability test's late joiner must attach inside the [`HOLD`] its
//! flight's read is held open (else its one-epoch assertion fails), the
//! left-flight sizing test's layer loss must land inside the [`HOLD`] its
//! joined flight's read is held open (else that read succeeds and its
//! recovery assertion fails), and the reader-cap test's gap must end between
//! the first catch-up's cap and the second's ([`CAP_MARGIN`] after one,
//! [`OFFSET`] − [`CAP_MARGIN`] before the other).
//!
//! # A second, silent activation condition
//!
//! Every test here uses a test seam, so the module exists only under the
//! **`test-fault-injection`** cargo feature (`grpc/mod.rs` gates it): without
//! it the run reports these tests absent, not failed.
//!
//! ```text
//! cargo nextest run -p live-edit --test integration \
//!     --features test-support,test-fault-injection grpc::subscribe_fenced_catch_up_test
//! ```
//!
//! # Wait budget
//!
//! Every test body runs under [`TEST_BUDGET`] (100 s), inside nextest's 120 s
//! hang-killer, so a stuck test fails with a message. Inside it every wait is a
//! deadline: [`SETTLE`] (10 s) per message, park or polled condition, [`DRAIN`]
//! (20 s) per catch-up drain, [`LAPSE_BUDGET`] (15 s) for a lease to lapse and
//! [`CAP_BUDGET`] (60 s) for a catch-up that must end at `reader_wait_max(L)`.
//! The healthy path pays none of them. Paid on purpose: [`SPEND`] (1.5 s) in the
//! two-wave and burst tests, [`SHORT_BUDGET_MS`] (300 ms) where a refusal is
//! the event under test, [`HOLD`] (2 s) where a read is held open, a staged
//! writers' gap of [`SHORT_GAP_MS`] (2.5 s) where a flight must sleep, and the
//! reader-cap test's real `reader_wait_max(L)` (~27 s) plus [`CAP_MARGIN`] —
//! about 31 s, the longest test here.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface};
use fred::types::scripts::Script;
use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::{
    handle_subscribe_read_layer, BudgetWait, CapacityBudgetSeam, CatchUpBuildTurnSeam,
    CatchUpStateReadSeam,
};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_CATCHUP_CAPACITY, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_ERROR,
    STREAM_EXIT_FENCED_READ_WAIT_CAP, STREAM_EXIT_INNER_WATCHDOG, STREAM_EXIT_SESSION_CANCEL,
    STREAM_EXIT_SHUTDOWN,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{
    ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    SnapshotReadConsumer,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::memory_budget::MemoryGrant;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::cost::FITTED_BYTES_PER_ITEM;
use live_edit::storage::snapshot_fence::read::CappedFenceWait;
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::snapshot_flights::{FlightState, FlightWaiter, SnapshotFlights};
use live_edit::storage::subscriber_admission::{CatchUpPermit, CatchUpReadPermit};
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::storage::snapshot_read_support::{
    error_reply_script, read_too_large, ProxyAction, ProxyRule, ScriptReplyProxy,
};

use super::helpers::{
    build_disabled_recovery_hook, build_service, make_create_data, make_create_header,
    make_request, streaming_create, streaming_update, AllowAllPermissions,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// `Metrics` / `service` label of every fixture here.
const SERVICE: &str = "test";

/// The fixture client's atomic-read item ceiling: far below every layer here,
/// so each is above the ceiling and its catch-ups take a flight. The same
/// ceiling as `snapshot_flights_test`, so `reader_wait_max(L)` is the ~27 s
/// that test documents.
const LOW_ITEMS: u64 = 16;

/// The fixture client's atomic-read byte ceiling.
const LOW_BYTES: u64 = 256;

/// Sections the layer is created with: about eight items each, so the layer's
/// item bound is far above [`LOW_ITEMS`].
const LAYER_SECTIONS: u64 = 6;

/// Bound on one awaited message, park or polled condition.
const SETTLE: Duration = Duration::from_secs(10);

/// One budget for a whole catch-up drain, however many messages it holds.
const DRAIN: Duration = Duration::from_secs(20);

/// Bound on a lease nobody releases lapsing: one lease (~2.2 s) plus slack.
const LAPSE_BUDGET: Duration = Duration::from_secs(15);

/// Bound on a catch-up that must end at `reader_wait_max(L)` (~27 s here).
const CAP_BUDGET: Duration = Duration::from_secs(60);

/// The one deadline over a whole test; see the module's wait budget.
const TEST_BUDGET: Duration = Duration::from_secs(100);

/// A capacity budget short enough that a refusal under test costs little.
const SHORT_BUDGET_MS: u64 = 300;

/// The capacity budget of the tests that show a wait is NOT charged: below
/// [`SPEND`], so a charged wait of that length would refuse the catch-up.
const SPENT_PAST_BUDGET_MS: u64 = 1_000;

/// How long the two-wave and burst tests hold a build chain, to spend real
/// time past [`SPENT_PAST_BUDGET_MS`]. A spend, not a synchronisation.
const SPEND: Duration = Duration::from_millis(1_500);

/// A capacity budget no wait here runs out of.
const ROOMY_BUDGET_MS: u64 = 60_000;

/// How long the fan-out's delay seam holds a read open between its grant and
/// its release (renewals keep the ~2.2 s lease live through it).
const HOLD: Duration = Duration::from_secs(2);

/// A staged writers' gap a flight sleeps through and then reads.
const SHORT_GAP_MS: i64 = 2_500;

/// A staged writers' gap past every `reader_wait_max(L)` here.
const FAR_MS: i64 = 600_000;

/// The reader-cap test's second catch-up subscribes this long after the
/// first.
const OFFSET: Duration = Duration::from_secs(8);

/// The reader-cap test's writers' gap ends this long after the first
/// catch-up's `reader_wait_max(L)`: inside the second's (it started
/// [`OFFSET`] later).
const CAP_MARGIN: Duration = Duration::from_secs(4);

/// The shrunk per-subscription backlog hard cap of the build-turn test.
const TINY_HARD_CAP: usize = 8;

/// Commits written into a subscription's sink to kill it at [`TINY_HARD_CAP`].
const OVERFLOW_COMMITS: u64 = 12;

/// The writer of every commit here: no subscriber uses it, so self-echo never
/// hides a commit.
const WRITER: &str = "fenced-writer";

/// The in-stream status of a catch-up refused while its flight waited for
/// capacity (`grpc::subscribe::flight_capacity_refused`).
const FLIGHT_CAPACITY_MSG: &str =
    "this replica's catch-up capacity for layers above the atomic-read limits is in use: the \
     shared snapshot read this catch-up joined was still waiting for a catch-up slot \
     (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS), a fenced read permit \
     (LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY) or catch-up memory (its memory budget, \
     sized from the pod's memory limit) when its capacity budget ran out; retry";

/// The in-stream status of a catch-up that left its flight at
/// `reader_wait_max(L)` (`grpc::subscribe::fenced_read_wait_refused`).
const FENCED_READ_WAIT_MSG: &str =
    "this layer is above the atomic-read limits, and the shared fenced snapshot read this \
     catch-up joined did not complete within the time a catch-up waits for one \
     (reader_wait_max); retry";

/// The fence scripts' clock: `TIME` in whole ms, as they read it.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

/// `snapshot_fence_probe.lua`'s SHA, to recognise the probe's `EVALSHA` on a
/// proxied connection.
fn probe_sha() -> String {
    let script = Script::from_lua(include_str!(
        "../../../src/storage/scripts/snapshot_fence_probe.lua"
    ));
    let sha: &str = script.sha1();
    sha.to_owned()
}

/// Run a test's body under [`TEST_BUDGET`], boxed so the body's future lives
/// on the heap rather than in the test thread's stack frame.
fn within_test_budget<'a>(
    body: impl std::future::Future<Output = ()> + Send + 'a,
) -> Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    let body = Box::pin(body);
    Box::pin(async move {
        tokio::time::timeout(TEST_BUDGET, body)
            .await
            .unwrap_or_else(|_| panic!("the test did not finish within {TEST_BUDGET:?}"));
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The capacity a fixture's client is built with.
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`.
    slots: usize,
    /// `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`: each subscription's capacity
    /// budget.
    budget_ms: u64,
    /// The atomic-read item ceiling.
    items: u64,
    /// The atomic-read byte ceiling.
    bytes: u64,
}

impl Shape {
    const fn new(slots: usize, budget_ms: u64) -> Shape {
        Shape {
            slots,
            budget_ms,
            items: LOW_ITEMS,
            bytes: LOW_BYTES,
        }
    }
}

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// One above-ceiling layer on a fresh key prefix, a client of [`Shape`] with a
/// metrics registry installed on both it and the service, and a service over
/// that client (snapshot mode; a test flips `catchup_snapshot_enabled` before
/// subscribing).
struct Fx {
    tv: TestValkey,
    client: ValkeyClient,
    metrics: Arc<Metrics>,
    svc: LiveEditService,
    shape: Shape,
    layer: String,
    session_id: String,
    oid: u64,
}

impl Fx {
    async fn new(layer: &str, shape: Shape) -> Self {
        Self::with_url(layer, shape, None).await
    }

    /// [`Self::new`] with the client connected to `url` (a proxy in front of
    /// the test store) instead of `TEST_VALKEY_URL`.
    async fn with_url(layer: &str, shape: Shape, url: Option<String>) -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let tv = TestValkey::setup().await;
        let mut config = crate::common::test_config();
        if let Some(url) = url {
            config.valkey_url = url;
        }
        config.snapshot_atomic_read_max_items = shape.items;
        config.snapshot_atomic_read_max_bytes = shape.bytes;
        config.subscriber_max_concurrent_catchups = shape.slots;
        config.subscriber_admission_wait_ms = shape.budget_ms;
        config.subscriber_admission_max_waiters = None;
        config.valkey_connect_timeout_secs = 30;
        let client = connect_test_client(&config)
            .await
            .expect("the fixture's client must connect");
        let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let svc = Self::service(&tv, &client, &metrics);
        let session_id = SessionStore::new(client.clone(), tv.key_schema().clone())
            .create(&format!("{layer}-session"), "/stage", "test-user")
            .await
            .expect("session creation must succeed");
        let sections: Vec<u64> = (1..=LAYER_SECTIONS).collect();
        let oid = handle_create_layer(
            &svc,
            make_request(streaming_create(vec![
                make_create_header(&session_id, layer),
                make_create_data(delta_to_ompf_bytes(&sections_delta(&sections), &[])),
            ])),
        )
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;
        Fx {
            tv,
            client,
            metrics,
            svc,
            shape,
            layer: layer.to_owned(),
            session_id,
            oid,
        }
    }

    /// A service over `client`, everything else as `helpers::build_service`
    /// builds it (so a field added to `LiveEditService` is added there only).
    fn service(tv: &TestValkey, client: &ValkeyClient, metrics: &Arc<Metrics>) -> LiveEditService {
        let keys = tv.key_schema().clone();
        let layer_store = Arc::new(LayerStore::new(client.clone(), keys.clone(), 4096));
        let recovery = build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(metrics));
        LiveEditService {
            catchup_snapshot_enabled: true,
            session_store: Arc::new(SessionStore::new(client.clone(), keys.clone())),
            layer_store,
            snapshot_flights: SnapshotFlights::new(client.clone(), keys),
            valkey: client.clone(),
            metrics: Arc::clone(metrics),
            recovery,
            ..build_service(tv, Arc::new(AllowAllPermissions))
        }
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    // ── Subscribing and committing ──────────────────────────────────────────

    async fn subscribe(
        &self,
        have_object_id: u64,
        have_version: u64,
        client_id: &str,
    ) -> SubStream {
        let mut request = Request::new(SubscribeReadLayerRequest {
            session_id: self.session_id.clone(),
            layer_name: self.layer.clone(),
            object_id: have_object_id,
            sequence_num: have_version,
            client_id: client_id.to_owned(),
        });
        request
            .extensions_mut()
            .insert(BearerToken::new("test-token"));
        handle_subscribe_read_layer(&self.svc, request)
            .await
            .unwrap_or_else(|s| panic!("{client_id}: the subscribe must be admitted: {s:?}"))
            .into_inner()
    }

    /// A brand-new subscriber.
    async fn join(&self, client_id: &str) -> SubStream {
        self.subscribe(0, 0, client_id).await
    }

    /// One commit adding `section_id` under the root, from [`WRITER`].
    async fn commit(&self, section_id: u64) {
        update(
            &self.svc,
            &self.session_id,
            &self.layer,
            self.oid,
            &sections_delta(&[section_id]),
            &[],
        )
        .await;
    }

    /// The layer's header sequence now.
    async fn seq(&self) -> u64 {
        self.svc
            .layer_store
            .read_object_header(self.oid)
            .await
            .expect("the header reads")
            .sequence_num
    }

    /// The layer's size from its atomic read's `TOO_LARGE` answer on the
    /// fixture's client, with `recorded_bpi` (Spec SDD SR-11.1) — what a
    /// catch-up's flight is sized from.
    async fn size_with(&self, recorded_bpi: Option<f64>) -> LayerSize {
        LayerSize::from_too_large(
            &read_too_large(&self.client, self.keys(), self.oid).await,
            recorded_bpi,
        )
    }

    /// Install `seam` as the service's build-turn seam and return it.
    fn park_build_turns(&mut self, seam: CatchUpBuildTurnSeam) -> Arc<CatchUpBuildTurnSeam> {
        let seam = Arc::new(seam);
        self.svc.catch_up_build_turn_seam_for_test = Some(Arc::clone(&seam));
        seam
    }

    /// Install a capacity-budget recorder and return it.
    fn record_charges(&mut self) -> Arc<CapacityBudgetSeam> {
        let seam = Arc::new(CapacityBudgetSeam::new());
        self.svc.capacity_budget_seam_for_test = Some(Arc::clone(&seam));
        seam
    }

    /// Wait until `client_id`'s wait for its flight's read has returned — its
    /// `FlightRead` charge is recorded as the wait returns, before it checks
    /// its cut — so it is about to register its stream at the flight's cut
    /// and go on to wait for its build turn. A caller that needs the
    /// registration itself awaits it separately. Needs
    /// [`Self::record_charges`].
    async fn until_read_returned_for(&self, charges: &CapacityBudgetSeam, client_id: &str) {
        await_condition(
            || {
                charges
                    .charges_for(client_id)
                    .iter()
                    .any(|c| c.wait == BudgetWait::FlightRead)
            },
            &format!("{client_id}'s shared read returned"),
        )
        .await;
    }

    fn injector(&self) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        self.client
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        injector
    }

    /// Hold the fixture flight's fenced read open for [`HOLD`] between its
    /// grant and its release.
    fn hold_next_fenced_read(&self) {
        self.injector().push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            HOLD,
        );
    }

    // ── Holding what a flight waits for ─────────────────────────────────────

    async fn hold_slot(&self) -> CatchUpPermit {
        self.client
            .catchup_gate
            .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600))
            .await
            .unwrap_or_else(|r| panic!("the test's own slot must be free: {r:?}"))
    }

    async fn hold_fenced_permit(&self) -> CatchUpReadPermit {
        self.client
            .catchup_fenced_read_gate
            .acquire(
                tokio::time::Instant::now() + Duration::from_secs(3_600),
                SnapshotReadConsumer::Catchup,
            )
            .await
            .unwrap_or_else(|r| panic!("the test's own fenced permit must be free: {r:?}"))
    }

    /// Reserve the whole catch-up memory budget.
    async fn hold_memory(&self) -> MemoryGrant {
        let budget = &self.client.catchup_build_budget;
        budget
            .reserve(budget.total_bytes(), None)
            .await
            .unwrap_or_else(|r| panic!("the whole budget must be free: {r:?}"))
    }

    /// Register a flight for the fixture object, sized `size`, that nothing
    /// drives until a catch-up attaches to it and reads.
    fn hold_flight(&self, size: LayerSize) -> FlightWaiter {
        self.svc.snapshot_flights.attach_or_create(self.oid, size)
    }

    /// The fixture object's flight's `(attachable, state)`, if registered.
    fn registered(&self) -> Option<(bool, FlightState)> {
        self.svc
            .snapshot_flights
            .lookup(self.oid)
            .map(|l| (l.attachable, l.state))
    }

    // ── The fence key ───────────────────────────────────────────────────────

    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid.to_string())
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

    /// Stage a writers' gap ending `ahead_ms` after the store's clock, at
    /// `size`: acquire answers `RETRY_AFTER gap` until then.
    async fn stage_gap(&self, ahead_ms: i64, size: LayerSize) {
        let until = self.now().await + ahead_ms;
        let _: () = self
            .tv
            .client()
            .pool
            .hset(
                self.fence_key(),
                [
                    ("gap_until", until.to_string()),
                    ("size_e", size.items.to_string()),
                    ("size_b", size.bytes.to_string()),
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

    /// Write `bpi` on the fence key, as a past fenced read records it.
    async fn store_bpi(&self, bpi: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(self.fence_key(), [("bpi", bpi.to_owned())])
            .await
            .expect("staging the stored bpi must succeed");
    }

    /// The fence key's reader fields' deadlines (`r:<token>` → deadline).
    async fn readers(&self) -> Vec<i64> {
        let kind: String = self
            .tv
            .client()
            .pool
            .r#type(self.fence_key())
            .await
            .expect("TYPE fence must succeed");
        if kind != "hash" {
            return Vec::new();
        }
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

    // ── Metrics ─────────────────────────────────────────────────────────────

    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    /// `live_edit_subscriber_stream_exits_total{reason}`.
    fn exits(&self, reason: &str) -> u64 {
        self.metric(
            "live_edit_subscriber_stream_exits_total",
            &[("reason", reason)],
        ) as u64
    }

    /// `live_edit_snapshot_read_total{consumer="catchup", path, outcome}`.
    fn reads(&self, path: &str, outcome: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "catchup"),
                ("path", path),
                ("outcome", outcome),
            ],
        ) as u64
    }

    /// The structural cache's fenced cold-start reads that verified: a
    /// structural writer's first edit of a layer above the ceilings
    /// cold-starts it fenced, which takes a fence epoch of its own.
    fn cold_start_fenced_reads(&self) -> u64 {
        self.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "structural_cache"),
                ("path", "fenced"),
                ("outcome", "ok"),
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

    /// `live_edit_subscribe_catchup_wait_seconds_count{result}`: slot acquires.
    fn slot_waits(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_subscribe_catchup_wait_seconds_count",
            &[("result", result)],
        ) as u64
    }

    fn reader_waits(&self) -> u64 {
        self.metric(
            "live_edit_snapshot_fence_reader_wait_seconds_count",
            &[("consumer", "catchup")],
        ) as u64
    }

    fn gauge(&self, name: &str) -> i64 {
        self.metric(name, &[]) as i64
    }

    fn stable_cuts(&self, outcome: &str, regime: &str) -> u64 {
        self.metric(
            "live_subscribe_stable_cut_total",
            &[
                ("outcome", outcome),
                ("regime", regime),
                ("service", SERVICE),
            ],
        ) as u64
    }

    // ── Idle ────────────────────────────────────────────────────────────────

    /// Everything a catch-up or a flight takes is back: no flight registered,
    /// every catch-up slot, fenced and atomic read permit free, nothing
    /// reserved or over-committed, no build counted, no subscription
    /// admitted, and no reader lease live once leases have had time to lapse.
    async fn assert_idle(&self, what: &str) {
        await_condition(
            || {
                self.svc.snapshot_flights.is_empty()
                    && self.client.catchup_gate.available() == self.shape.slots
                    && self.gauge("live_edit_subscriber_admitted") == 0
            },
            &format!("{what}: no flight registered, every slot free, nothing admitted"),
        )
        .await;
        let fenced = &self.client.catchup_fenced_read_gate;
        let atomic = &self.client.catchup_read_gate;
        assert_eq!(
            (fenced.available(), atomic.available()),
            (fenced.limit(), atomic.limit()),
            "{what}: every fenced and atomic read permit is free"
        );
        let budget = &self.client.catchup_build_budget;
        assert_eq!(
            (budget.reserved_bytes(), budget.overcommit_bytes()),
            (0, 0),
            "{what}: reserved == 0 && overcommit == 0"
        );
        assert_eq!(
            (
                self.gauge("live_edit_catchup_builds_in_progress"),
                self.gauge("live_edit_subscribe_catchups_inflight"),
            ),
            (0, 0),
            "{what}: no build counted and no slot counted held"
        );
        self.until_no_live_lease(what).await;
    }
}

/// A delta adding each of `section_ids` under the root.
fn sections_delta(section_ids: &[u64]) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: section_ids
            .iter()
            .map(|&section_id| {
                StructCommand::DiffChangeSection(CommandDiffChangeSection {
                    parent_id: 0,
                    section_id,
                    section_name: format!("sec-{section_id}"),
                    section_type: 0,
                    section_order: 0,
                })
            })
            .collect(),
        ..Default::default()
    }
}

/// One `UpdateLayer` of `delta` (with `external_values` as its OMPF parts)
/// from [`WRITER`].
async fn update(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
) {
    let messages = [
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: layer.to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: 10,
                    client_id: WRITER.to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk {
                    data: delta_to_ompf_bytes(delta, external_values),
                },
            )),
        },
    ];
    handle_update_layer(svc, make_request(streaming_update(messages.to_vec())))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer must succeed: {e}"));
}

// ── Stream helpers ──────────────────────────────────────────────────────────

/// The next item, which must arrive within [`SETTLE`] and not be end of
/// stream.
async fn next_item(
    stream: &mut SubStream,
    what: &str,
) -> Result<SubscribeReadLayerResponse, Status> {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("the stream ended cleanly before {what}"))
}

/// The stream's first item, which must be its terminal status: nothing — no
/// content, no ack — may precede it.
async fn first_status(stream: &mut SubStream, what: &str) -> Status {
    first_status_within(stream, SETTLE, what).await
}

/// [`first_status`], arriving within `bound`.
async fn first_status_within(stream: &mut SubStream, bound: Duration, what: &str) -> Status {
    let item = tokio::time::timeout(bound, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {bound:?}"))
        .unwrap_or_else(|| panic!("the stream ended cleanly before {what}"));
    match item {
        Err(status) => status,
        Ok(msg) => panic!(
            "{what} must be the FIRST item on the stream, before any content or caught-up ack; \
             got a message first: {msg:?}"
        ),
    }
}

/// One header with its reassembled body.
async fn recv_message(stream: &mut SubStream, what: &str) -> (SubscribeReadLayerHeader, Vec<u8>) {
    let header = match next_item(stream, what).await {
        Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
        }) => h,
        other => panic!("{what}: expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match next_item(stream, what).await {
            Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
            }) => body.extend_from_slice(&c.data),
            other => panic!("{what}: expected a body chunk, got {other:?}"),
        }
    }
    (header, body)
}

/// A brand-new catch-up's content and its caught-up ack, under one [`DRAIN`]
/// deadline: the content at `(object_id, cut)` and the ack at the same cut.
/// Returns the cut.
async fn recv_served(stream: &mut SubStream, object_id: u64, who: &str) -> u64 {
    tokio::time::timeout(DRAIN, async {
        let (content, _) = recv_message(stream, &format!("{who}'s content")).await;
        assert!(
            content.total_size > 0,
            "{who}: the first message must be content, not the ack: {content:?}"
        );
        let (ack, _) = recv_message(stream, &format!("{who}'s caught-up ack")).await;
        assert_eq!(
            ack.total_size, 0,
            "{who}: expected the caught-up ack: {ack:?}"
        );
        assert_eq!(
            (content.object_id, ack.object_id, ack.sequence_num),
            (object_id, object_id, content.sequence_num),
            "{who}: the ack carries the cut the content serves"
        );
        content.sequence_num
    })
    .await
    .unwrap_or_else(|_| panic!("{who}: the catch-up did not finish within {DRAIN:?}"))
}

/// Poll `condition` until it holds or [`SETTLE`] elapses. The sleep is the
/// probe interval, not a coordination primitive.
async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `reader_wait_max(L)` for `size` on `client`'s coefficients.
fn reader_wait_max(client: &ValkeyClient, size: LayerSize) -> Duration {
    CappedFenceWait::new(client, 0, SnapshotReadConsumer::Catchup).cap(size)
}

/// Gauges of held and queued catch-up slots.
const INFLIGHT: &str = "live_edit_subscribe_catchups_inflight";
const WAITING: &str = "live_edit_subscribe_catchups_waiting";
/// Registered streams the multiplexer reads; a reader reservation is not one.
const STREAMS: &str = "live_edit_stream_mux_streams";

// ═════════════════════════════════════════════════════════════════════════════
// 1. Two waves and a burst: one read per wave, builds in turn, no refusal
// ═════════════════════════════════════════════════════════════════════════════

/// Two waves at the catch-up level (PM review r4 F38), at one catch-up slot:
/// a second wave arriving while the first wave's flight is still building
/// takes no slot — it succeeds the closed flight and waits for its `done`,
/// uncharged — and both waves are served with no refusal, one fenced read per
/// wave.
///
/// Wave 1 (two joiners) attaches while the test holds the fenced permit, so
/// both share the first flight. Its first build is parked in its turn
/// (`CatchUpBuildTurnSeam::for_first(1)`), holding the flight's turn and the
/// replica's only slot. Wave 2 (two joiners) then arrives: the first creates
/// the successor flight, the second joins it, and neither queues for a slot.
/// The test spends [`SPEND`] there, past every joiner's capacity budget
/// ([`SPENT_PAST_BUDGET_MS`]), then lets wave 1 build. Had wave 2 waited for
/// the slot it would queue (`catchups_waiting` = 1) and be refused
/// `catchup_capacity`; had it been charged for wave 1's whole build chain it
/// would be refused too, and for any [`SHORT_BUDGET_MS`] of it its
/// `FlightRead` charge would reach that bound — its own flight's capacity
/// waits, all free once wave 1 is done, cost it next to nothing.
///
/// The successor is sized from its predecessor (PM review r4 F38): the size
/// the registry reports for wave 2's flight as it registers is the size wave
/// 1's flight measured once its read resolved, not the estimate wave 1's
/// flight was created with.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_wave_takes_no_slot_until_the_first_flight_is_done_and_both_are_served_with_one_read_each(
) {
    within_test_budget(async {
        let mut fx = Fx::new("fenced-two-waves", Shape::new(1, SPENT_PAST_BUDGET_MS)).await;
        let seam = fx.park_build_turns(CatchUpBuildTurnSeam::for_first(1));
        let charges = fx.record_charges();

        let permit = fx.hold_fenced_permit().await;
        let mut wave1 = vec![fx.join("wave1-a").await];
        await_condition(
            || fx.roles("leader") == 1 && fx.gauge(INFLIGHT) == 1,
            "wave 1's flight is created and holds the only slot",
        )
        .await;
        wave1.push(fx.join("wave1-b").await);
        await_condition(
            || fx.roles("joined_read") == 1,
            "the second joiner attaches to wave 1's flight",
        )
        .await;
        let flight_size = || {
            fx.svc
                .snapshot_flights
                .lookup(fx.oid)
                .expect("a flight is registered")
                .size
        };
        // The fenced permit is held, so the flight has had no fence reply:
        // this is still the size it was created with.
        let estimate = flight_size();
        drop(permit);
        tokio::time::timeout(SETTLE, seam.wait_until_parked(1))
            .await
            .expect("wave 1's first build parks in its turn");
        assert_eq!(
            fx.registered(),
            Some((false, FlightState::Reading)),
            "fixture: wave 1's flight has read and is closed to attach, still building"
        );
        let measured = flight_size();
        assert_ne!(
            measured, estimate,
            "fixture: wave 1's measured size differs from its creation estimate, so the \
             successor's size tells the two apart"
        );

        let mut wave2 = vec![fx.join("wave2-a").await];
        await_condition(
            || fx.roles("leader") == 2,
            "wave 2's first joiner succeeds the closed flight",
        )
        .await;
        assert_eq!(
            fx.registered(),
            Some((true, FlightState::Queued)),
            "fixture: the registered flight is now wave 2's successor"
        );
        assert_eq!(
            flight_size(),
            measured,
            "the successor is sized from its predecessor's measured snapshot (PM review r4 F38)"
        );
        wave2.push(fx.join("wave2-b").await);
        await_condition(
            || fx.roles("joined_read") == 2,
            "wave 2's second joiner joins the successor",
        )
        .await;
        assert_eq!(
            (fx.gauge(INFLIGHT), fx.gauge(WAITING)),
            (1, 0),
            "while wave 1 builds, wave 2 holds no slot and queues for none"
        );
        tokio::time::sleep(SPEND).await;
        assert_eq!(fx.exits(STREAM_EXIT_CATCHUP_CAPACITY), 0);
        seam.release_all();

        let mut cuts = Vec::new();
        for (who, stream) in ["wave1-a", "wave1-b", "wave2-a", "wave2-b"]
            .iter()
            .zip(wave1.iter_mut().chain(wave2.iter_mut()))
        {
            cuts.push(recv_served(stream, fx.oid, who).await);
        }
        let seq = fx.seq().await;
        assert!(
            cuts.iter().all(|c| *c == seq),
            "every joiner is served the layer at its sequence {seq}: {cuts:?}"
        );
        assert_eq!(
            (fx.reads("fenced", "ok"), fx.reads("atomic", "too_large")),
            (2, 1),
            "one fenced read per wave; only the very first joiner tried an atomic read (every \
             later one found a flight registered)"
        );
        assert_eq!(fx.exits(STREAM_EXIT_CATCHUP_CAPACITY), 0, "no refusal");
        for who in ["wave2-a", "wave2-b"] {
            let flight_read = charges
                .charges_for(who)
                .into_iter()
                .find(|c| c.wait == BudgetWait::FlightRead)
                .unwrap_or_else(|| panic!("{who}: its flight wait is recorded"));
            assert!(
                flight_read.charged < Duration::from_millis(SHORT_BUDGET_MS),
                "{who}: the wait for the predecessor's builds is not charged: {flight_read:?}"
            );
            assert!(
                charges
                    .charges_for(who)
                    .iter()
                    .all(|c| c.wait != BudgetWait::CatchUpSlot),
                "{who}: a joiner that found a flight registered never waits for a slot"
            );
        }
        drop((wave1, wave2));
        fx.assert_idle("after both waves").await;
    })
    .await;
}

/// The decider's same-replica burst: more joiners of one above-ceiling layer
/// than the replica has catch-up slots share one epoch and one read, and are
/// all served with no refusal, building in turn.
///
/// Two slots, five joiners, all attached while the test holds the fenced
/// permit. After the read, the first two builds park in their turns: one in
/// the flight's own turn (under the flight's slot), one as an extra build
/// (under the second slot and its own build part); the other three wait for a
/// turn, holding no slot and queueing for none, for [`SPEND`] — past their
/// capacity budget. Released, every joiner is served the same cut. Slot-wait
/// observations follow the contract: one per slot acquire — the first
/// joiner's atomic attempt and the flight's — and none for the extra build's
/// non-blocking `try_acquire`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_of_more_joiners_than_slots_shares_one_epoch_builds_in_turn_and_is_never_refused() {
    within_test_budget(async {
        const JOINERS: usize = 5;
        let mut fx = Fx::new("fenced-burst", Shape::new(2, SPENT_PAST_BUDGET_MS)).await;
        let seam = fx.park_build_turns(CatchUpBuildTurnSeam::for_first(2));

        let permit = fx.hold_fenced_permit().await;
        let mut streams = vec![fx.join("burst-0").await];
        await_condition(
            || fx.roles("leader") == 1,
            "the first joiner creates the flight",
        )
        .await;
        for i in 1..JOINERS {
            streams.push(fx.join(&format!("burst-{i}")).await);
        }
        await_condition(
            || fx.roles("joined_read") == (JOINERS - 1) as u64,
            "every later joiner attaches to the one flight",
        )
        .await;
        drop(permit);
        tokio::time::timeout(SETTLE, seam.wait_until_parked(2))
            .await
            .expect("the flight's turn and one extra build park");
        let mut extra: Vec<bool> = seam.parked().iter().map(|p| p.extra).collect();
        extra.sort_unstable();
        assert_eq!(
            extra,
            vec![false, true],
            "one build in the flight's own turn and one extra build, under the free slot"
        );
        assert_eq!(
            (fx.gauge(INFLIGHT), fx.gauge(WAITING)),
            (2, 0),
            "the flight's slot and the extra build's; the other joiners queue for no slot"
        );
        tokio::time::sleep(SPEND).await;
        seam.release_all();

        let mut cuts = Vec::new();
        for (i, stream) in streams.iter_mut().enumerate() {
            cuts.push(recv_served(stream, fx.oid, &format!("burst-{i}")).await);
        }
        assert!(
            cuts.windows(2).all(|w| w[0] == w[1]),
            "one read serves every joiner the same cut: {cuts:?}"
        );
        assert_eq!(
            (
                fx.acquires("started_epoch"),
                fx.acquires("joined_epoch"),
                fx.reads("fenced", "ok"),
            ),
            (1, 0, 1),
            "the burst costs one epoch and one read"
        );
        assert_eq!(fx.exits(STREAM_EXIT_CATCHUP_CAPACITY), 0, "no refusal");
        assert_eq!(
            fx.slot_waits("acquired"),
            2,
            "one slot-wait observation per slot acquire: the first joiner's atomic attempt and \
             the flight's; the extra build's try_acquire records none"
        );
        drop(streams);
        fx.assert_idle("after the burst").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Joinability: the read is work; a pin read first is never a regression
// ═════════════════════════════════════════════════════════════════════════════

/// A catch-up behind a fenced read that lasts longer than its capacity budget
/// still joins it, and one that attaches between the fence's grant and its
/// release, with its pin read first, is served at the cut — not refused as a
/// sequence regression (Spec SDD SR-6.2; Task 6 PM review r2, defer-16 (i)).
///
/// The first joiner's read is held open for [`HOLD`] between its grant and its
/// release — far past the [`SHORT_BUDGET_MS`] budget, which the read, being
/// work, is not charged against. While it is held, a second brand-new joiner
/// reads its header (P), parks before its proven read holding no slot (it
/// found the flight registered, so it takes none), and is released to attach:
/// no commit can land during the epoch, so its P equals the cut. A writer
/// started during the read commits after the release; both joiners are served
/// the cut and then that commit, exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_joiner_attaching_between_grant_and_release_with_its_pin_read_first_is_served_not_refused(
) {
    within_test_budget(async {
        let mut fx = Fx::new("fenced-join-reading", Shape::new(2, SHORT_BUDGET_MS)).await;
        fx.hold_next_fenced_read();
        let mut first = fx.join("first").await;
        await_condition(
            || fx.registered() == Some((true, FlightState::Reading)),
            "the first joiner's flight holds its fence and reads",
        )
        .await;
        let cut = fx.seq().await;

        let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
        fx.svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
        let mut late = fx.join("late").await;
        tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
            .await
            .expect("the late joiner parks after its pin, before attaching");
        assert_eq!(
            fx.gauge(INFLIGHT),
            1,
            "parked before its proven read, the late joiner holds no slot: only the flight's is \
             held"
        );
        let writer = {
            let (svc, session, layer, oid) = (
                fx.svc.clone(),
                fx.session_id.clone(),
                fx.layer.clone(),
                fx.oid,
            );
            tokio::spawn(async move {
                update(&svc, &session, &layer, oid, &sections_delta(&[700]), &[]).await;
            })
        };
        tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
            .await
            .expect("release the late joiner");

        assert_eq!(recv_served(&mut first, fx.oid, "first").await, cut);
        assert_eq!(
            recv_served(&mut late, fx.oid, "late").await,
            cut,
            "the late joiner's pin, read during the epoch, equals the cut: it is served there"
        );
        tokio::time::timeout(SETTLE, writer)
            .await
            .expect("the writer commits after the release")
            .expect("the writer must not panic");
        for (who, stream) in [("first", &mut first), ("late", &mut late)] {
            let (live, _) = recv_message(stream, &format!("{who}'s first live commit")).await;
            assert_eq!(
                live.sequence_num,
                cut + 1,
                "{who}: the commit after the release arrives live, once"
            );
        }
        assert_eq!(
            (
                fx.roles("leader"),
                fx.roles("joined_read"),
                fx.reads("fenced", "ok"),
                // The writer's own fenced cold start takes one more grant —
                // an epoch of its own, or a join of the catch-ups' if its
                // acquire lands inside the join window — so the catch-ups'
                // grants are what is left: one grant between them, the
                // flight's. (Whether the writer's grant shared their epoch is
                // not pinned.)
                fx.acquires("started_epoch") + fx.acquires("joined_epoch")
                    - fx.cold_start_fenced_reads(),
            ),
            (1, 1, 1, 1),
            "the late joiner joined the read under the fence: one flight, one read, one fence \
             grant for the two catch-ups (the writer's cold-start grant excluded)"
        );
        assert_eq!(
            (
                fx.exits(STREAM_EXIT_CATCHUP_CAPACITY),
                fx.stable_cuts("unverified", "brand_new"),
                fx.stable_cuts("verified", "brand_new"),
            ),
            (0, 0, 2),
            "neither joiner is refused, by its budget or as a regression"
        );
        drop((first, late));
        fx.assert_idle("after the shared read").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. The reader cap (SR-5.5): UNAVAILABLE before content, its own exit reason
// ═════════════════════════════════════════════════════════════════════════════

/// A catch-up whose flight's work outlasts `reader_wait_max(L)` ends
/// `UNAVAILABLE` before any content, with no refusal detail (no retry hint, so
/// a prompt retry can re-attach to the still-open flight), counted under
/// `reason="fenced_read_wait_cap"`; the flight goes on for a catch-up that
/// attached later, which is served. The test holds the fence off with a staged
/// writers' gap ending [`CAP_MARGIN`] after the first catch-up's cap; the
/// second subscribes [`OFFSET`] later, so the gap ends inside its cap.
///
/// The fence sleep is work, never charged: both catch-ups run on a
/// [`SHORT_BUDGET_MS`] budget. And defer-13's units: the cap ending is not
/// observed on `live_edit_snapshot_fence_reader_wait_seconds` — it reads zero
/// after the ending, while the flight's attempt still sleeps — and the
/// flight's one read attempt is its only observation.
///
/// Slow by necessity: the first catch-up runs out a real `reader_wait_max(L)`
/// (~27 s).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_shared_read_outlasts_reader_wait_max_ends_unavailable_before_content() {
    within_test_budget(async {
        let fx = Fx::new("fenced-reader-cap", Shape::new(2, SHORT_BUDGET_MS)).await;
        let size = fx.size_with(None).await;
        let cap = reader_wait_max(&fx.client, size);
        assert!(
            cap < Duration::from_secs(45),
            "fixture: reader_wait_max(L) {cap:?} must fit the test budget"
        );
        fx.stage_gap((cap + CAP_MARGIN).as_millis() as i64, size)
            .await;
        let mut early = fx.join("early").await;
        tokio::time::sleep(OFFSET).await;
        let mut late = fx.join("late").await;
        await_condition(
            || fx.roles("joined_read") == 1,
            "the second catch-up attaches to the sleeping flight",
        )
        .await;

        let status = first_status_within(&mut early, CAP_BUDGET, "the cap ending").await;
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert_eq!(status.message(), FENCED_READ_WAIT_MSG);
        crate::common::refusal_detail::assert_no_refusal_detail(&status);
        await_condition(
            || fx.exits(STREAM_EXIT_FENCED_READ_WAIT_CAP) == 1,
            "the ending is counted under fenced_read_wait_cap",
        )
        .await;
        assert_eq!(
            fx.reader_waits(),
            0,
            "the cap ending is not observed on the reader-wait histogram"
        );
        assert_eq!(
            fx.registered(),
            Some((true, FlightState::Queued)),
            "the flight carries on for the later catch-up"
        );

        recv_served(&mut late, fx.oid, "late").await;
        assert_eq!(
            fx.reader_waits(),
            1,
            "the flight's one read attempt is the histogram's only observation"
        );
        assert_eq!(
            (
                fx.exits(STREAM_EXIT_FENCED_READ_WAIT_CAP),
                fx.exits(STREAM_EXIT_CATCHUP_CAPACITY),
            ),
            (1, 0),
            "the fence sleep is not charged: no capacity refusal on a 300 ms budget"
        );
        drop(late);
        fx.assert_idle("after the flight").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Capacity: the flight's waits are charged per attached catch-up
// ═════════════════════════════════════════════════════════════════════════════

/// Which of its flight's capacity waits a refused catch-up is attached
/// through.
#[derive(Debug, Clone, Copy)]
enum FlightWait {
    /// The flight's catch-up slot: the test holds the only slot and has
    /// registered the flight itself, so the subject finds it registered,
    /// takes no slot of its own and attaches.
    Slot,
    /// The fenced read permit (`P_f` = 1), which the test holds.
    FencedPermit,
    /// The combined memory reservation: the test holds the whole budget.
    Memory,
}

/// A catch-up attached to a flight that waits for capacity is charged, on its
/// own budget, for the time the flight waits; when the budget runs out it
/// leaves the flight and ends in-stream `UNAVAILABLE` under
/// `reason="catchup_capacity"`, as its FIRST item, with a status naming the
/// flight's three waits (defer-16 (ii)) and the retry hint `CATCHUP_CAPACITY`
/// with no `wait` metadata, whichever wait it was. The charge is recorded once, as a
/// `FlightRead` wait with no deadline that ran out the whole remaining budget.
///
/// Waiting for memory, the flight holds the fenced permit and no fence: the
/// fence key holds no reader (a fenced read never waits for budget while
/// holding its fence).
async fn a_flight_capacity_wait_past_the_budget_refuses_the_catch_up_before_content(
    wait: FlightWait,
) {
    let slots = match wait {
        FlightWait::Slot => 1,
        _ => 2,
    };
    let mut fx = Fx::new("fenced-capacity", Shape::new(slots, SHORT_BUDGET_MS)).await;
    let charges = fx.record_charges();
    let size = fx.size_with(None).await;
    let held_slot = match wait {
        FlightWait::Slot => Some(fx.hold_slot().await),
        _ => None,
    };
    let held_flight = matches!(wait, FlightWait::Slot).then(|| fx.hold_flight(size));
    let held_permit = match wait {
        FlightWait::FencedPermit => Some(fx.hold_fenced_permit().await),
        _ => None,
    };
    let held_memory = match wait {
        FlightWait::Memory => Some(fx.hold_memory().await),
        _ => None,
    };

    let mut stream = fx.join("refused").await;
    if matches!(wait, FlightWait::Memory) {
        await_condition(
            || fx.client.catchup_build_budget.waiters() == 1,
            "the flight waits for its memory reservation",
        )
        .await;
        assert_eq!(
            fx.client.catchup_fenced_read_gate.available(),
            0,
            "it waits for memory holding the fenced permit"
        );
        assert!(
            fx.readers().await.is_empty(),
            "and no fence: no reader lease is on the fence key while it waits for memory"
        );
    }
    let status = first_status(&mut stream, "the flight capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{wait:?}: {status:?}");
    assert_eq!(status.message(), FLIGHT_CAPACITY_MSG, "{wait:?}");
    crate::common::refusal_detail::assert_refusal_detail(
        &status,
        live_edit::error::refusal_reason::CATCHUP_CAPACITY,
        &[],
        live_edit::error::CAPACITY_RETRY_DELAY,
    );
    await_condition(
        || fx.exits(STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted under catchup_capacity",
    )
    .await;

    let recorded = charges.charges_for("refused");
    let last = recorded.last().expect("the subject's charges are recorded");
    assert_eq!(
        (last.wait, last.deadline, last.budget_after),
        (BudgetWait::FlightRead, None, Duration::ZERO),
        "{wait:?}: one flight-read charge, with no deadline, that ran out what was left: \
         {recorded:#?}"
    );
    assert!(
        last.charged > Duration::ZERO && last.charged == last.budget_before,
        "{wait:?}: the flight's wait was charged: {last:?}"
    );
    if matches!(wait, FlightWait::Slot) {
        assert!(
            recorded.iter().all(|c| c.wait != BudgetWait::CatchUpSlot),
            "a catch-up that found the flight registered took no slot of its own: {recorded:#?}"
        );
    }

    drop((held_slot, held_flight, held_permit, held_memory));
    fx.assert_idle(&format!("{wait:?}: after the refusal"))
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_attached_while_its_flight_waits_for_a_slot_past_its_budget_is_refused_before_content(
) {
    within_test_budget(
        a_flight_capacity_wait_past_the_budget_refuses_the_catch_up_before_content(
            FlightWait::Slot,
        ),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_attached_while_its_flight_waits_for_a_fenced_permit_past_its_budget_is_refused_before_content(
) {
    within_test_budget(
        a_flight_capacity_wait_past_the_budget_refuses_the_catch_up_before_content(
            FlightWait::FencedPermit,
        ),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_attached_while_its_flight_waits_for_memory_past_its_budget_is_refused_before_content(
) {
    within_test_budget(
        a_flight_capacity_wait_past_the_budget_refuses_the_catch_up_before_content(
            FlightWait::Memory,
        ),
    )
    .await;
}

/// A `RETRY_AFTER` sleep is work: a flight sleeping out a staged writers' gap
/// keeps its catch-up slot (so it re-acquires at the wake), gives back its
/// fenced permit and its memory reservation for the sleep, holds no fence,
/// and charges its catch-up nothing — served on a [`SHORT_BUDGET_MS`] budget
/// far shorter than the [`SHORT_GAP_MS`] sleep. The flight takes its slot
/// once: two slot-wait observations in all, the catch-up's atomic attempt's
/// and the flight's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flight_sleeping_out_a_retry_after_keeps_its_slot_frees_its_permit_and_memory_and_charges_nothing(
) {
    within_test_budget(async {
        let mut fx = Fx::new("fenced-retry-after", Shape::new(1, SHORT_BUDGET_MS)).await;
        let charges = fx.record_charges();
        let size = fx.size_with(None).await;
        fx.stage_gap(SHORT_GAP_MS, size).await;
        let mut stream = fx.join("sleeper").await;
        await_condition(
            || fx.acquires("retry_after_gap") >= 1,
            "the flight's acquire is told to sleep out the gap",
        )
        .await;
        assert_eq!(
            fx.gauge(INFLIGHT),
            1,
            "the flight keeps its slot through the sleep"
        );
        let fenced = &fx.client.catchup_fenced_read_gate;
        await_condition(
            || {
                fenced.available() == fenced.limit()
                    && fx.client.catchup_build_budget.reserved_bytes() == 0
            },
            "the flight gives back its fenced permit and its reservation for the sleep",
        )
        .await;
        assert!(
            fx.readers().await.is_empty(),
            "no fence is held through the sleep"
        );

        recv_served(&mut stream, fx.oid, "sleeper").await;
        assert_eq!(
            fx.exits(STREAM_EXIT_CATCHUP_CAPACITY),
            0,
            "the sleep is not charged"
        );
        let flight_read = charges
            .charges_for("sleeper")
            .into_iter()
            .find(|c| c.wait == BudgetWait::FlightRead)
            .expect("the flight wait is recorded");
        assert!(
            flight_read.charged < Duration::from_millis(SHORT_BUDGET_MS),
            "the gap's sleep is not charged: {flight_read:?}"
        );
        assert_eq!(
            fx.slot_waits("acquired"),
            2,
            "the catch-up's atomic attempt and the flight's one slot acquire, kept through the \
             sleep"
        );
        drop(stream);
        fx.assert_idle("after the sleep").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. The build-turn wait ends on its stream's hard cap
// ═════════════════════════════════════════════════════════════════════════════

/// A catch-up waiting for its build turn holds a registered stream whose
/// backlog grows outside the memory budget; when the multiplexer's hard cap
/// kills that registration, the catch-up ends with that ending, before
/// content, without waiting for its turn (`take_build_turn` checks every
/// `BUILD_TURN_STREAM_CHECK`).
///
/// Two joiners share one flight at one slot; whichever reaches a build turn
/// first parks in it (the holder), so the other queues (no extra build: the
/// flight holds the only slot). With a [`TINY_HARD_CAP`]-entry hard cap,
/// [`OVERFLOW_COMMITS`] commits kill both registrations. The queued joiner's
/// first item is the watchdog ending while the holder is still parked; the
/// holder ends the same way once released.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_queued_for_its_build_turn_whose_stream_hits_the_hard_cap_ends_before_content_without_waiting(
) {
    within_test_budget(async {
        let mut fx = Fx::new("fenced-turn-hard-cap", Shape::new(1, ROOMY_BUDGET_MS)).await;
        let seam = fx.park_build_turns(CatchUpBuildTurnSeam::for_first(1));
        let charges = fx.record_charges();
        let knobs = fx.client.stream_mux.knobs();
        fx.client
            .stream_mux
            .set_knobs_for_test(live_edit::storage::stream_mux::StreamMuxKnobs {
                channel_capacity: TINY_HARD_CAP / 2,
                backlog_hard_cap: TINY_HARD_CAP,
                send_watchdog: Duration::from_secs(600),
                ..knobs
            });

        let permit = fx.hold_fenced_permit().await;
        let a = fx.join("a").await;
        await_condition(
            || fx.roles("leader") == 1,
            "the first joiner creates the flight",
        )
        .await;
        let b = fx.join("b").await;
        await_condition(
            || fx.roles("joined_read") == 1,
            "the second joiner attaches",
        )
        .await;
        drop(permit);
        tokio::time::timeout(SETTLE, seam.wait_until_parked(1))
            .await
            .expect("the first joiner to reach a build turn parks in it");
        let ((mut holder, _), (mut queued, queued_id)) = match seam.parked()[0].client_id.as_str() {
            "a" => ((a, "a"), (b, "b")),
            _ => ((b, "b"), (a, "a")),
        };
        fx.until_read_returned_for(&charges, queued_id).await;
        assert_eq!(
            fx.gauge(STREAMS),
            1,
            "the layer's one stream is registered; the queued joiner registers on it at the \
             flight's cut as its read returns, then waits for its turn"
        );

        for section_id in 0..OVERFLOW_COMMITS {
            fx.commit(800 + section_id).await;
        }
        let status = first_status(&mut queued, "the queued joiner's ending").await;
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert!(
            status
                .message()
                .contains("fell behind its delivery watchdog"),
            "the queued joiner ends with its registration's ending: {status:?}"
        );
        assert_eq!(
            seam.parked().len(),
            1,
            "it ended while the holder still held the turn: it did not wait for its own"
        );
        assert_eq!(
            fx.gauge("live_edit_catchup_builds_in_progress"),
            1,
            "only the holder's build turn was ever taken"
        );

        seam.release_all();
        let status = first_status(&mut holder, "the holder's ending").await;
        assert!(
            status
                .message()
                .contains("fell behind its delivery watchdog"),
            "the holder's registration was killed too, and it ends before content: {status:?}"
        );
        await_condition(
            || fx.exits(STREAM_EXIT_INNER_WATCHDOG) == 2,
            "both endings are counted under inner_watchdog",
        )
        .await;
        fx.assert_idle("after both endings").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Sizing the flight (Spec SDD SR-11.1; code review r2)
// ═════════════════════════════════════════════════════════════════════════════
//
// A flight reserves its memory from its size before its first fence reply, so
// the size a catch-up creates it with must already be the design's `b`. The
// tests read that size from the registry (`SnapshotFlights::lookup`) while the
// test holds the fenced permit, so the flight has not yet acquired and the
// size is still the creator's. The fixture's client runs through a proxy that
// counts the probe's `EVALSHA` (`snapshot_fence_probe.lua`, preloaded so the
// count is one per probe) and, for the failed-probe case, answers it with an
// error.

/// The bytes per item a fence key here stores: far from the fitted value, so
/// a size computed from it is told apart from the fitted one.
const STORED_BPI: f64 = 100.0;

/// Called on every fence probe the proxy sees, as it passes through.
type OnProbe = Arc<dyn Fn() + Send + Sync>;

/// A client-to-store proxy rule matching the fence probe's `EVALSHA`: each one
/// is counted on `probes` and reported to `on_probe`, and answered with an
/// error when `fail`, else forwarded.
fn probe_rule(probes: Arc<AtomicUsize>, fail: bool, on_probe: Option<OnProbe>) -> ProxyRule {
    let sha = probe_sha();
    Arc::new(move |args: &[Vec<u8>]| {
        let evalsha = args
            .first()
            .is_some_and(|a| a.eq_ignore_ascii_case(b"EVALSHA"));
        if !evalsha || args.get(1).map(Vec::as_slice) != Some(sha.as_bytes()) {
            return None;
        }
        probes.fetch_add(1, Ordering::SeqCst);
        if let Some(on_probe) = &on_probe {
            on_probe();
        }
        fail.then(|| ProxyAction::Eval(error_reply_script("ERR injected fence probe failure")))
    })
}

/// A fixture whose client runs through a [`probe_rule`] proxy, with the probe
/// script preloaded on the store. Returns the fixture, the proxy (kept alive
/// for the test) and the probe count.
async fn proxied_fx(
    layer: &str,
    shape: Shape,
    fail: bool,
) -> (Fx, ScriptReplyProxy, Arc<AtomicUsize>) {
    proxied_fx_observing(layer, shape, fail, None).await
}

/// [`proxied_fx`], with `on_probe` called on every probe.
async fn proxied_fx_observing(
    layer: &str,
    shape: Shape,
    fail: bool,
    on_probe: Option<OnProbe>,
) -> (Fx, ScriptReplyProxy, Arc<AtomicUsize>) {
    use fred::interfaces::LuaInterface;
    let probes = Arc::new(AtomicUsize::new(0));
    let proxy = ScriptReplyProxy::start(
        crate::storage::snapshot_read_support::test_valkey_addr(),
        probe_rule(Arc::clone(&probes), fail, on_probe),
    )
    .await;
    // The proxy's URL carries `TEST_VALKEY_URL`'s database path, so the
    // fixture's client reads the database its `TestValkey` writes.
    let fx = Fx::with_url(layer, shape, Some(proxy.url())).await;
    let _: String = fx
        .tv
        .client()
        .pool
        .next()
        .script_load(include_str!(
            "../../../src/storage/scripts/snapshot_fence_probe.lua"
        ))
        .await
        .expect("preloading the probe script must succeed");
    (fx, proxy, probes)
}

/// The flight registered for the fixture object, its size read while the
/// test holds the fenced permit.
async fn flight_size_while_held(fx: &Fx) -> LayerSize {
    await_condition(
        || fx.registered().is_some(),
        "the catch-up's flight is registered",
    )
    .await;
    fx.svc
        .snapshot_flights
        .lookup(fx.oid)
        .expect("the flight is registered")
        .size
}

/// An early-out `TOO_LARGE` sizes the flight from the `bpi` the fence key
/// stored at the layer's last fenced read: `e × bpi`, from one probe.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_early_out_catch_up_sizes_its_flight_from_the_fence_keys_stored_bpi() {
    within_test_budget(async {
        let (fx, _proxy, probes) =
            proxied_fx("fenced-size-stored", Shape::new(2, ROOMY_BUDGET_MS), false).await;
        fx.store_bpi(&STORED_BPI.to_string()).await;
        let stored = fx.size_with(Some(STORED_BPI)).await;
        assert_ne!(
            stored,
            fx.size_with(None).await,
            "fixture: the stored bpi sizes the layer differently from the fitted one"
        );
        let permit = fx.hold_fenced_permit().await;
        let mut stream = fx.join("sized").await;
        assert_eq!(
            flight_size_while_held(&fx).await,
            stored,
            "the flight is sized e × the stored bpi"
        );
        assert_eq!(probes.load(Ordering::SeqCst), 1, "one probe sized it");
        drop(permit);
        recv_served(&mut stream, fx.oid, "sized").await;
        drop(stream);
        fx.assert_idle("after the flight").await;
    })
    .await;
}

/// A registered flight suppresses the probe: a catch-up whose atomic read
/// answers `TOO_LARGE` after the object's flight was registered sends no probe
/// and attaches to that flight, at the size the flight knows.
///
/// The catch-up is held inside its atomic read (the snapshot lane stalled)
/// while the test registers the flight itself, at a size of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_finding_the_flight_registered_after_its_atomic_read_sends_no_probe() {
    within_test_budget(async {
        let (fx, _proxy, probes) = proxied_fx(
            "fenced-size-suppressed",
            Shape::new(2, ROOMY_BUDGET_MS),
            false,
        )
        .await;
        fx.store_bpi(&STORED_BPI.to_string()).await;
        let observer = fx.client.maintenance_pool.next().clone();
        let stalls = live_edit::storage::test_support::stall_pool(
            &fx.client.snapshot_pool,
            &observer,
            &format!("{}conn-stall:snapshot:", fx.tv.prefix()),
        )
        .await;
        let mut stream = fx.join("late-sizer").await;
        await_condition(
            || {
                fx.metric(
                    "live_edit_catchup_snapshot_reads_in_flight",
                    &[("class", "atomic")],
                ) == 1.0
            },
            "the catch-up is inside its atomic read",
        )
        .await;
        let held_size = LayerSize {
            items: 1_000,
            bytes: 77_777,
        };
        let held = fx.hold_flight(held_size);
        live_edit::storage::test_support::release_all(stalls).await;

        recv_served(&mut stream, fx.oid, "late-sizer").await;
        assert_eq!(
            fx.reads("atomic", "too_large"),
            1,
            "fixture: the catch-up's atomic read answered TOO_LARGE"
        );
        assert_eq!(
            probes.load(Ordering::SeqCst),
            0,
            "a flight registered when the answer came suppresses the probe"
        );
        assert_eq!(
            (fx.roles("leader"), fx.roles("joined_read")),
            (1, 1),
            "the catch-up joined the registered flight"
        );
        drop((stream, held));
        fx.assert_idle("after the flight").await;
    })
    .await;
}

/// Installs an enabled recovery hook on `fx`'s service with a checkpoint of
/// its layer (at sequence 13) to restore from.
async fn enable_recovery(fx: &mut Fx) {
    let store = Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    crate::common::seed_layer_checkpoint(&store, fx.oid, &fx.session_id, &fx.layer, 13).await;
    let mut recovery_config = crate::common::test_config();
    recovery_config.checkpoint_enabled = true;
    recovery_config.service_label = SERVICE.to_string();
    fx.svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
        store,
        Arc::clone(&fx.svc.layer_store),
        Arc::new(recovery_config),
        Arc::clone(&fx.svc.metrics),
    ));
}

/// Remove every key of `fx`'s object but its stream — a keyspace loss under a
/// read, which the read then answers `NotFound`. The fence key goes with the
/// rest.
async fn lose_layer_state(fx: &Fx) {
    let stream_key = fx.keys().layer_stream(fx.oid);
    let state_keys: Vec<fred::types::Key> =
        crate::common::scan_object_keys(fx.tv.client(), fx.tv.prefix(), fx.oid)
            .await
            .into_iter()
            .filter(|k| k.as_str() != Some(stream_key.as_str()))
            .collect();
    assert!(
        !state_keys.is_empty(),
        "fixture: the object has state to lose"
    );
    let _: () = fx
        .tv
        .client()
        .pool
        .unlink(state_keys)
        .await
        .expect("UNLINK of the object's state must succeed");
}

/// Regression (post-completion review F17): a catch-up whose early-out
/// `TOO_LARGE` answer arrives while the object's flight is registered, and
/// which finds no flight registered when it next attaches, probes the fence
/// key at that attach — the one registry lookup that decides a flight's size
/// is the attach's own.
///
/// The catch-up is held inside its atomic read (the snapshot lane stalled)
/// while the test registers a flight; freed, its early-out answer arrives
/// with that flight registered, so it joins it and sends no probe. The
/// flight's read is held under its fence ([`HOLD`]) while the layer's state
/// is lost, so the read answers `NotFound` and the flight leaves the
/// registry. The catch-up recovers the layer (a restore) and attaches again:
/// the registry misses, so it probes then — once — and creates the next
/// flight. Sizing from the earlier answer with no probe at the miss was the
/// defect: a flight that left between the answer and the attach was
/// succeeded unprobed. The probe count is the regression's pin. The size
/// check only confirms consistency: the restore wiped the old fence key,
/// `bpi` included, so the probe answers with no `bpi` and the flight is
/// sized `e` (from the catch-up's own answer) × the fitted bytes per item —
/// the size the old code also lands on here, unprobed. The test also rests
/// on the [`HOLD`] window (the loss, one `UNLINK`, must land inside it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_flight_left_after_its_early_out_answer_probes_at_its_next_attach() {
    within_test_budget(async {
        let (mut fx, _proxy, probes) =
            proxied_fx("fenced-size-left", Shape::new(2, ROOMY_BUDGET_MS), false).await;
        fx.store_bpi(&STORED_BPI.to_string()).await;
        let fitted = fx.size_with(None).await;
        // `size_with` reads through the fixture's client too.
        let too_large_before = fx.reads("atomic", "too_large");
        enable_recovery(&mut fx).await;
        let observer = fx.client.maintenance_pool.next().clone();
        let stalls = live_edit::storage::test_support::stall_pool(
            &fx.client.snapshot_pool,
            &observer,
            &format!("{}conn-stall:snapshot:", fx.tv.prefix()),
        )
        .await;
        let mut stream = fx.join("left-behind").await;
        await_condition(
            || {
                fx.metric(
                    "live_edit_catchup_snapshot_reads_in_flight",
                    &[("class", "atomic")],
                ) == 1.0
            },
            "the catch-up is inside its atomic read",
        )
        .await;
        fx.hold_next_fenced_read();
        let held = fx.hold_flight(LayerSize {
            items: 1_000,
            bytes: 77_777,
        });
        live_edit::storage::test_support::release_all(stalls).await;
        await_condition(
            || {
                fx.roles("joined_read") == 1
                    && fx.registered() == Some((true, FlightState::Reading))
            },
            "the catch-up joined the registered flight, whose read is held under its fence",
        )
        .await;
        assert_eq!(
            (
                fx.reads("atomic", "too_large") - too_large_before,
                probes.load(Ordering::SeqCst)
            ),
            (1, 0),
            "fixture: the catch-up's early-out answer came with the flight registered, so no \
             probe yet"
        );
        // Queue for the only fenced permit behind the held read, so the next
        // flight waits for it and its size can be read.
        let mut permit = std::pin::pin!(fx.client.catchup_fenced_read_gate.acquire(
            tokio::time::Instant::now() + Duration::from_secs(3_600),
            SnapshotReadConsumer::Catchup,
        ));
        assert!(
            futures::poll!(permit.as_mut()).is_pending(),
            "fixture: the test queues behind the held read for the fenced permit"
        );
        drop(held);
        lose_layer_state(&fx).await;

        let permit = tokio::time::timeout(SETTLE, permit)
            .await
            .expect("the failed flight gives its fenced permit back")
            .expect("the test takes the fenced permit");
        await_condition(
            || {
                crate::common::recovery_metric(&fx.metrics, "restored") == 1
                    && fx.roles("leader") == 2
            },
            "the catch-up restored the layer and created the next flight",
        )
        .await;
        assert_eq!(
            probes.load(Ordering::SeqCst),
            1,
            "the attach that found no flight registered probed the fence key (the regression's \
             pin: the old code sized the next flight from the earlier answer, unprobed)"
        );
        assert_eq!(
            flight_size_while_held(&fx).await,
            fitted,
            "consistency only: the probe found no bpi (the restore wiped it), so e × the fitted \
             bpi, which the old code also reaches unprobed"
        );
        drop(permit);
        recv_served(&mut stream, fx.oid, "left-behind").await;
        assert!(
            fx.reads("fenced", "not_found") >= 1,
            "fixture: the held flight's read found the layer gone"
        );
        drop(stream);
        fx.assert_idle("after the recovered catch-up").await;
    })
    .await;
}

/// A rollback-mode in-window replay whose per-commit atomic read answers an
/// early-out `TOO_LARGE` hands over to the layer's shared fenced read and
/// probes for the stored `bpi` at that attach (post-completion review F17):
/// after giving its catch-up slot back, never while holding it. The flight is
/// sized `e × stored bpi` from that one probe.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_replay_handing_over_on_an_early_out_probes_at_the_attach_with_its_slot_given_back(
) {
    within_test_budget(async {
        const SLOTS: usize = 2;
        let gate: Arc<
            std::sync::OnceLock<Arc<live_edit::storage::subscriber_admission::CatchUpGate>>,
        > = Arc::new(std::sync::OnceLock::new());
        let free_at_probe = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let on_probe: OnProbe = {
            let (gate, free_at_probe) = (Arc::clone(&gate), Arc::clone(&free_at_probe));
            Arc::new(move || {
                if let Some(gate) = gate.get() {
                    free_at_probe
                        .lock()
                        .expect("probe record")
                        .push(gate.available());
                }
            })
        };
        let (mut fx, _proxy, probes) = proxied_fx_observing(
            "fenced-size-replay",
            Shape::new(SLOTS, ROOMY_BUDGET_MS),
            false,
            Some(on_probe),
        )
        .await;
        fx.svc.catchup_snapshot_enabled = false;
        let have_version = fx.seq().await;
        fx.commit(101).await;
        fx.commit(102).await;
        // The commits probe the fence on their own path; count from here.
        let probes_before = probes.load(Ordering::SeqCst);
        gate.set(Arc::clone(&fx.client.catchup_gate))
            .unwrap_or_else(|_| panic!("the gate is recorded once"));
        fx.store_bpi(&STORED_BPI.to_string()).await;
        let stored = fx.size_with(Some(STORED_BPI)).await;
        assert_ne!(
            stored,
            fx.size_with(None).await,
            "fixture: the stored bpi sizes the layer differently from the fitted one"
        );
        let held = fx.hold_fenced_permit().await;

        let mut stream = fx.subscribe(fx.oid, have_version, "replayer").await;
        assert_eq!(
            flight_size_while_held(&fx).await,
            stored,
            "the hand-over's flight is sized e × the stored bpi"
        );
        assert_eq!(
            probes.load(Ordering::SeqCst) - probes_before,
            1,
            "one probe sized the flight"
        );
        assert_eq!(
            free_at_probe.lock().expect("probe record").clone(),
            vec![SLOTS],
            "the one probe was sent at the attach, with the replay's slot already given back"
        );
        drop(held);
        recv_served(&mut stream, fx.oid, "replayer").await;
        drop(stream);
        fx.assert_idle("after the hand-over").await;
    })
    .await;
}

/// What the probe of a sizing test finds.
#[derive(Debug, Clone, Copy)]
enum Probe {
    /// No fence key.
    Absent,
    /// The fence key holds a string; the test removes it before the flight
    /// acquires.
    WrongType,
    /// The probe fails (the proxy answers it with an error); the key stores a
    /// `bpi` the probe never returns.
    Failed,
}

/// A probe that returns no `bpi` — the key absent or wrong-typed, or the
/// probe itself failing — sizes the flight from the fitted bytes per item
/// (`FITTED_BYTES_PER_ITEM`), and never ends the catch-up: it is served.
async fn a_probe_with_no_bpi_sizes_the_flight_from_the_fitted_bpi_and_the_catch_up_is_served(
    probe: Probe,
) {
    let (fx, _proxy, probes) = proxied_fx(
        "fenced-size-fitted",
        Shape::new(2, ROOMY_BUDGET_MS),
        matches!(probe, Probe::Failed),
    )
    .await;
    match probe {
        Probe::Absent => {}
        Probe::WrongType => {
            let _: () = fx
                .tv
                .client()
                .pool
                .set(fx.fence_key(), "not-a-hash", None, None, false)
                .await
                .expect("staging a wrong-typed fence key must succeed");
        }
        Probe::Failed => fx.store_bpi(&STORED_BPI.to_string()).await,
    }
    let fitted = fx.size_with(None).await;
    assert_eq!(
        fitted,
        fx.size_with(Some(FITTED_BYTES_PER_ITEM)).await,
        "fixture: the fitted size is e × FITTED_BYTES_PER_ITEM"
    );
    assert_ne!(fitted, fx.size_with(Some(STORED_BPI)).await);

    let permit = fx.hold_fenced_permit().await;
    let mut stream = fx.join("fitted").await;
    assert_eq!(
        flight_size_while_held(&fx).await,
        fitted,
        "{probe:?}: the flight is sized from the fitted bytes per item"
    );
    assert_eq!(
        probes.load(Ordering::SeqCst),
        1,
        "{probe:?}: one probe was sent"
    );
    if matches!(probe, Probe::WrongType) {
        let _: i64 = fx
            .tv
            .client()
            .pool
            .del(fx.fence_key())
            .await
            .expect("removing the wrong-typed key must succeed");
    }
    drop(permit);
    recv_served(&mut stream, fx.oid, "fitted").await;
    assert_eq!(
        fx.exits(STREAM_EXIT_ERROR),
        0,
        "{probe:?}: the probe never ended the catch-up"
    );
    drop(stream);
    fx.assert_idle(&format!("{probe:?}: after the flight"))
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_probe_finds_no_fence_key_sizes_from_the_fitted_bpi_and_is_served() {
    within_test_budget(
        a_probe_with_no_bpi_sizes_the_flight_from_the_fitted_bpi_and_the_catch_up_is_served(
            Probe::Absent,
        ),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_probe_finds_a_wrong_typed_fence_key_sizes_from_the_fitted_bpi_and_is_served(
) {
    within_test_budget(
        a_probe_with_no_bpi_sizes_the_flight_from_the_fitted_bpi_and_the_catch_up_is_served(
            Probe::WrongType,
        ),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_probe_fails_sizes_from_the_fitted_bpi_and_is_served() {
    within_test_budget(
        a_probe_with_no_bpi_sizes_the_flight_from_the_fitted_bpi_and_the_catch_up_is_served(
            Probe::Failed,
        ),
    )
    .await;
}

/// A mid-read abort sizes the flight from its own tallies, scaled to the whole
/// layer, and sends no probe, even with a `bpi` stored on the fence key.
///
/// The client's byte ceiling is low and its item ceiling high, and the layer
/// carries three large inline values, so the atomic read passes its early-out
/// and stops mid-read on the byte tally.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_mid_read_too_large_sizes_its_flight_from_its_tallies_and_sends_no_probe() {
    within_test_budget(async {
        const VALUE_LEN: usize = 3_000;
        let shape = Shape {
            items: 10_000,
            bytes: 5_000,
            ..Shape::new(2, ROOMY_BUDGET_MS)
        };
        let (fx, _proxy, probes) = proxied_fx("fenced-size-mid-read", shape, false).await;
        let values = OwnedDelta {
            is_diff: false,
            base_version: 0,
            set_fields: ["a", "b", "c"]
                .iter()
                .map(|key| SetField {
                    section_id: 1,
                    key_name: (*key).to_owned(),
                    value_or_ext_hash: vec![0x5A; VALUE_LEN],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                })
                .collect(),
            ..Default::default()
        };
        update(&fx.svc, &fx.session_id, &fx.layer, fx.oid, &values, &[]).await;
        fx.store_bpi(&STORED_BPI.to_string()).await;
        let answer = read_too_large(&fx.client, fx.keys(), fx.oid).await;
        assert!(
            matches!(
                answer.at,
                live_edit::storage::diff_read::SnapshotTooLargeAt::MidRead { .. }
            ),
            "fixture: the atomic read stops mid-read: {answer:?}"
        );
        let tallied = LayerSize::from_too_large(&answer, None);

        let permit = fx.hold_fenced_permit().await;
        let mut stream = fx.join("mid-read").await;
        assert_eq!(
            flight_size_while_held(&fx).await,
            tallied,
            "the flight is sized from the answer's tallies"
        );
        assert_eq!(
            probes.load(Ordering::SeqCst),
            0,
            "a mid-read abort sends no probe"
        );
        drop(permit);
        recv_served(&mut stream, fx.oid, "mid-read").await;
        drop(stream);
        fx.assert_idle("after the flight").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Cancellation at every await, and injected failures (SR-7.4)
// ═════════════════════════════════════════════════════════════════════════════

/// Where in its flight a cancelled catch-up is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    /// Its flight succeeds one still building (a holder parked in its build
    /// turn) and waits for that one's `done`.
    Predecessor,
    /// Attached to a flight the test registered, which waits for the slot the
    /// test holds.
    SlotWait,
    /// Its flight holds its slot and waits for the fenced permit the test
    /// holds.
    FencedPermitWait,
    /// Its flight holds its slot and the fenced permit and waits for memory
    /// the test holds.
    MemoryWait,
    /// Its flight sleeps out a staged writers' gap.
    RetryAfterSleep,
    /// Its flight reads under the fence, held open by the delay seam.
    Reading,
    /// Read, registered, and queued for its build turn behind a holder parked
    /// in the flight's turn (one slot: no extra build).
    TurnQueue,
    /// Parked inside its own build turn.
    InTurn,
}

/// How the cancelled catch-up ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    ClientDisconnect,
    Shutdown,
    SessionDeletion,
}

/// A catch-up ended at `at` by `ending` ends before any content, counted
/// under its own reason, and once the test lets go of what it held (and any
/// holder catch-up has finished) every catch-up slot, fenced and atomic
/// permit and the memory budget are back at full capacity, no flight or build
/// is left, and no reader lease outlives its lease.
async fn a_catch_up_ended_at(at: At, ending: Ending) {
    let slots = match at {
        At::Predecessor | At::SlotWait | At::TurnQueue => 1,
        _ => 2,
    };
    let mut fx = Fx::new("fenced-cancel", Shape::new(slots, ROOMY_BUDGET_MS)).await;
    let charges = fx.record_charges();
    let seam = match at {
        At::Predecessor => Some(fx.park_build_turns(CatchUpBuildTurnSeam::for_clients(["holder"]))),
        At::TurnQueue => Some(fx.park_build_turns(CatchUpBuildTurnSeam::for_first(1))),
        At::InTurn => Some(fx.park_build_turns(CatchUpBuildTurnSeam::for_clients(["subject"]))),
        _ => None,
    };
    let size = fx.size_with(None).await;
    let (mut held_slot, mut held_flight, mut held_permit, mut held_memory, mut holder) =
        (None, None, None, None, None);
    match at {
        At::Predecessor => {
            holder = Some(fx.join("holder").await);
            tokio::time::timeout(
                SETTLE,
                seam.as_ref().expect("installed").wait_until_parked(1),
            )
            .await
            .expect("the holder parks in its build turn");
        }
        At::SlotWait => {
            held_slot = Some(fx.hold_slot().await);
            held_flight = Some(fx.hold_flight(size));
        }
        At::FencedPermitWait => held_permit = Some(fx.hold_fenced_permit().await),
        At::MemoryWait => held_memory = Some(fx.hold_memory().await),
        At::RetryAfterSleep => fx.stage_gap(FAR_MS, size).await,
        At::Reading => fx.hold_next_fenced_read(),
        At::TurnQueue => {
            held_permit = Some(fx.hold_fenced_permit().await);
            holder = Some(fx.join("holder").await);
            await_condition(|| fx.roles("leader") == 1, "the holder creates the flight").await;
        }
        At::InTurn => {}
    }

    let mut subject = fx.join("subject").await;
    match at {
        At::Predecessor => {
            await_condition(
                || fx.roles("leader") == 2,
                "the subject succeeds the holder's flight",
            )
            .await
        }
        At::SlotWait => {
            await_condition(
                || fx.roles("joined_read") == 1 && fx.gauge(WAITING) == 1,
                "the subject attached and its flight queues for the held slot",
            )
            .await
        }
        At::FencedPermitWait => {
            await_condition(
                || fx.registered().is_some() && fx.gauge(INFLIGHT) == 1,
                "the subject's flight holds its slot, waiting for the held fenced permit",
            )
            .await
        }
        At::MemoryWait => {
            await_condition(
                || fx.client.catchup_build_budget.waiters() == 1,
                "the subject's flight waits for the held memory",
            )
            .await
        }
        At::RetryAfterSleep => {
            await_condition(
                || fx.acquires("retry_after_gap") >= 1,
                "the subject's flight sleeps out the gap",
            )
            .await
        }
        At::Reading => {
            await_condition(
                || fx.registered() == Some((true, FlightState::Reading)),
                "the subject's flight reads under its fence",
            )
            .await
        }
        At::TurnQueue => {
            await_condition(|| fx.roles("joined_read") == 1, "the subject attaches").await;
            drop(held_permit.take());
            let seam = seam.as_ref().expect("installed");
            tokio::time::timeout(SETTLE, seam.wait_until_parked(1))
                .await
                .expect("the first joiner to reach the flight's turn parks in it");
            // Whichever reached the turn first is the holder; the other is
            // the subject, queued behind it.
            if seam.parked()[0].client_id == "subject" {
                let other = holder.take().expect("the holder joined");
                holder = Some(std::mem::replace(&mut subject, other));
            }
            let queued = if seam.parked()[0].client_id == "subject" {
                "holder"
            } else {
                "subject"
            };
            fx.until_read_returned_for(&charges, queued).await;
        }
        At::InTurn => tokio::time::timeout(
            SETTLE,
            seam.as_ref().expect("installed").wait_until_parked(1),
        )
        .await
        .expect("the subject parks inside its build turn"),
    }

    let what = format!("{at:?} / {ending:?}");
    match ending {
        Ending::ClientDisconnect => {
            drop(subject);
            await_condition(
                || fx.exits(STREAM_EXIT_CLIENT_DISCONNECT) == 1,
                &format!("{what}: the ending is counted under client_disconnect"),
            )
            .await;
        }
        Ending::Shutdown => {
            fx.svc.shutdown.cancel();
            let status = first_status(&mut subject, &what).await;
            crate::common::refusal_detail::assert_shutdown_status(&status);
            await_condition(
                || fx.exits(STREAM_EXIT_SHUTDOWN) == 1,
                &format!("{what}: the ending is counted under shutdown"),
            )
            .await;
        }
        Ending::SessionDeletion => {
            fx.svc
                .session_tokens
                .get(&fx.session_id)
                .expect("the subscribe registered its session token")
                .cancel_deleted();
            let status = first_status(&mut subject, &what).await;
            assert_eq!(
                (status.code(), status.message()),
                (Code::NotFound, "session was deleted"),
                "{what}"
            );
            await_condition(
                || fx.exits(STREAM_EXIT_SESSION_CANCEL) == 1,
                &format!("{what}: the ending is counted under session_cancel"),
            )
            .await;
        }
    }

    drop((held_slot, held_flight, held_permit, held_memory));
    if let Some(seam) = &seam {
        seam.release_all();
    }
    if let Some(mut holder) = holder {
        recv_served(&mut holder, fx.oid, "the holder").await;
    }
    fx.assert_idle(&what).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_waiting_for_its_predecessor_flight_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(
        At::Predecessor,
        Ending::ClientDisconnect,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_its_flight_waits_for_a_slot_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::SlotWait, Ending::ClientDisconnect)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_its_flight_waits_for_a_fenced_permit_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(
        At::FencedPermitWait,
        Ending::ClientDisconnect,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_its_flight_waits_for_memory_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(
        At::MemoryWait,
        Ending::ClientDisconnect,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_its_flight_sleeps_out_a_retry_after_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(
        At::RetryAfterSleep,
        Ending::ClientDisconnect,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_its_flight_reads_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::Reading, Ending::ClientDisconnect)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_while_queued_for_its_build_turn_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::TurnQueue, Ending::ClientDisconnect)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_disconnected_inside_its_build_turn_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::InTurn, Ending::ClientDisconnect)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_shutdown_while_its_flight_reads_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::Reading, Ending::Shutdown)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_shutdown_inside_its_build_turn_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::InTurn, Ending::Shutdown)).await;
}

/// Graceful shutdown with a catch-up's flight reading under its fence gives
/// the fence lease back to the store before the pools close
/// (post-completion review F12), as `main.rs` runs it: the service shutdown
/// token ends the subscription `UNAVAILABLE`, its task drops the flight's
/// read (its last waiter), `drain_stream_tasks` waits for that task, and
/// `ValkeyClient::quit` waits for the dropped lease's tracked release before
/// it closes the pools. When `quit` returns, nothing is in flight and the
/// fence holds no reader field: released, not left to lapse (a lapse removes
/// nothing). The read is held far past the test, so only the drop can have
/// released it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shutdown_while_a_flight_reads_under_its_fence_releases_the_lease_before_the_pools_close()
{
    within_test_budget(async {
        /// How long the flight's first fan-out window is held back: far past
        /// this test's budget.
        const HELD_BACK: Duration = Duration::from_secs(600);

        let fx = Fx::new("fenced-shutdown-release", Shape::new(2, ROOMY_BUDGET_MS)).await;
        fx.injector().push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            HELD_BACK,
        );
        let mut subject = fx.join("subject").await;
        await_condition(
            || fx.registered() == Some((true, FlightState::Reading)),
            "the subject's flight reads under its fence",
        )
        .await;
        assert_eq!(
            fx.readers().await.len(),
            1,
            "fixture: the flight's read holds a lease on the fence"
        );

        fx.svc.shutdown.cancel();
        let status = first_status(&mut subject, "the shutdown").await;
        crate::common::refusal_detail::assert_shutdown_status(&status);
        live_edit::grpc::transport::drain_stream_tasks(
            &fx.svc.presence_cleanup_tracker,
            &fx.svc.subscription_tracker,
            SETTLE,
        )
        .await;
        fx.client.quit().await;

        assert_eq!(
            fx.client.fence_guard_cleanups_inflight(),
            0,
            "quit waited for the dropped read's lease release"
        );
        assert_eq!(
            fx.readers().await,
            Vec::<i64>::new(),
            "the lease was released before the pools closed — its reader field removed — rather \
             than left to lapse"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_session_deletion_while_its_flight_sleeps_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(
        At::RetryAfterSleep,
        Ending::SessionDeletion,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_session_deletion_inside_its_build_turn_leaves_nothing_held() {
    within_test_budget(a_catch_up_ended_at(At::InTurn, Ending::SessionDeletion)).await;
}

/// A capacity gate that closes under a flight — `ValkeyClient::quit`'s first
/// step at shutdown — ends each attached catch-up `shutdown`: in-stream
/// `UNAVAILABLE "server shutting down"` as its first item, counted under
/// `reason="shutdown"` and not `catchup_capacity`, with nothing left held.
/// The fenced twin of the capacity suite's quit-closes-the-read-gate test.
///
/// The service's shutdown token is never cancelled, so the subscription's own
/// shutdown ending cannot fire first: the closed gate, reaching the catch-up
/// as its flight's shutdown error, is the only thing that ends it (the
/// [`ROOMY_BUDGET_MS`] budget is never reached). The test holds the only
/// fenced permit and the subject's flight holds its slot, so the flight's next
/// wait is the permit's; closing the fenced read gate refuses it whether it is
/// already queued or about to queue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_whose_flight_waits_for_a_fenced_permit_when_the_gate_closes_ends_shutdown_before_content(
) {
    within_test_budget(async {
        let fx = Fx::new("fenced-gate-closed", Shape::new(2, ROOMY_BUDGET_MS)).await;
        let held = fx.hold_fenced_permit().await;
        let mut stream = fx.join("subject").await;
        await_condition(
            || fx.registered().is_some() && fx.gauge(INFLIGHT) == 1,
            "the subject's flight holds its slot, waiting for the held fenced permit",
        )
        .await;

        fx.client.catchup_fenced_read_gate.close();
        let status = first_status(&mut stream, "the closed gate's ending").await;
        crate::common::refusal_detail::assert_shutdown_status(&status);
        await_condition(
            || fx.exits(STREAM_EXIT_SHUTDOWN) == 1,
            "the ending is counted under shutdown",
        )
        .await;
        assert_eq!(
            fx.exits(STREAM_EXIT_CATCHUP_CAPACITY),
            0,
            "a closed gate is shutdown, not a capacity refusal"
        );
        assert!(
            !fx.svc.shutdown.is_cancelled(),
            "fixture self-check: the closed gate, not the shutdown token, ended the wait"
        );
        drop(held);
        fx.assert_idle("after the closed gate's ending").await;
    })
    .await;
}

/// A shared read that fails — a transient fault injected on the fenced
/// fan-out's first command — reaches every attached catch-up as its own copy
/// of the error: each ends `UNAVAILABLE` before content, under
/// `reason="error"`, and nothing is left held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shared_read_that_fails_ends_every_attached_catch_up_before_content_and_leaves_nothing_held(
) {
    within_test_budget(async {
        let fx = Fx::new("fenced-read-fails", Shape::new(2, ROOMY_BUDGET_MS)).await;
        fx.injector().push_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "injected fenced fan-out timeout",
            ),
        );
        let permit = fx.hold_fenced_permit().await;
        let mut first = fx.join("first").await;
        await_condition(
            || fx.roles("leader") == 1,
            "the first joiner creates the flight",
        )
        .await;
        let mut second = fx.join("second").await;
        await_condition(
            || fx.roles("joined_read") == 1,
            "the second joiner attaches",
        )
        .await;
        drop(permit);
        for (who, stream) in [("first", &mut first), ("second", &mut second)] {
            let status = first_status(stream, &format!("{who}'s ending")).await;
            assert_eq!(status.code(), Code::Unavailable, "{who}: {status:?}");
        }
        await_condition(
            || fx.exits(STREAM_EXIT_ERROR) == 2,
            "both endings are counted under error",
        )
        .await;
        assert_eq!(fx.reads("fenced", "ok"), 0);
        fx.assert_idle("after the failed read").await;
    })
    .await;
}

/// The blob of the external value a failing build resolves.
const EXT_BLOB: &[u8] = b"fenced-test-external-value-blob";

/// An external-value store that fails every `read_batch*` and `probe_values`
/// read once armed, and is otherwise [`MockExternalValueStore`]: a catch-up
/// build resolving the layer's external value then fails.
struct FailingExtStore {
    inner: MockExternalValueStore,
    failing: AtomicBool,
}

impl FailingExtStore {
    fn check(&self) -> Result<(), live_edit::error::ServiceError> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(live_edit::error::ServiceError::Unavailable(
                "FailingExtStore: injected read failure".to_owned(),
            ));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl live_edit::storage::external_values::ExternalValueStore for FailingExtStore {
    async fn store_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), live_edit::error::ServiceError> {
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), live_edit::error::ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> live_edit::storage::external_values::ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, live_edit::error::ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, live_edit::error::ServiceError> {
        self.check()?;
        self.inner.read_batch(hashes).await
    }

    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        recovery_counter: Option<&AtomicUsize>,
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, live_edit::error::ServiceError> {
        self.check()?;
        self.inner
            .read_batch_with_recovery_counter(hashes, recovery_counter)
            .await
    }

    async fn probe_values(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<live_edit::storage::external_values::ExtValueProbe, live_edit::error::ServiceError>
    {
        self.check()?;
        self.inner.probe_values(hashes).await
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[live_edit::storage::external_values::ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<
        live_edit::storage::external_values::ReleaseAttemptOutcome,
        live_edit::error::ServiceError,
    > {
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn process_pending(
        &self,
        policy: &live_edit::storage::external_values::GcPolicy,
        state: &mut live_edit::storage::external_values::GcPassState,
    ) -> Result<live_edit::storage::external_values::GcBatchResult, live_edit::error::ServiceError>
    {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(
        &self,
    ) -> Option<&std::sync::Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// A build that fails in its build turn ends its catch-up before content and
/// gives the turn back: at one slot, two catch-ups sharing a flight whose
/// layer carries an external value the builds cannot resolve each build in
/// turn, each ends `UNAVAILABLE` before content, and nothing is left held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_build_that_fails_in_its_turn_ends_its_catch_up_before_content_and_passes_the_turn_on() {
    within_test_budget(async {
        let mut fx = Fx::new("fenced-build-fails", Shape::new(1, ROOMY_BUDGET_MS)).await;
        let store = Arc::new(FailingExtStore {
            inner: MockExternalValueStore::new(),
            failing: AtomicBool::new(false),
        });
        fx.svc.ext_value_store = Arc::clone(&store) as _;
        let with_ext = OwnedDelta {
            is_diff: false,
            base_version: 0,
            set_fields: vec![SetField {
                section_id: 1,
                key_name: "ext:value".to_owned(),
                // The value's content hash: `UpdateLayer` checks an upload's
                // external values against it.
                value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(EXT_BLOB)
                    .to_vec(),
                ext_value_size: EXT_BLOB.len() as u64,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        };
        update(
            &fx.svc,
            &fx.session_id,
            &fx.layer,
            fx.oid,
            &with_ext,
            &[EXT_BLOB],
        )
        .await;
        store.failing.store(true, Ordering::SeqCst);

        let permit = fx.hold_fenced_permit().await;
        let mut first = fx.join("first").await;
        await_condition(
            || fx.roles("leader") == 1,
            "the first joiner creates the flight",
        )
        .await;
        let mut second = fx.join("second").await;
        await_condition(
            || fx.roles("joined_read") == 1,
            "the second joiner attaches",
        )
        .await;
        drop(permit);
        for (who, stream) in [("first", &mut first), ("second", &mut second)] {
            let status = first_status(stream, &format!("{who}'s ending")).await;
            assert_eq!(status.code(), Code::Unavailable, "{who}: {status:?}");
        }
        assert_eq!(
            fx.reads("fenced", "ok"),
            1,
            "fixture: the shared read served both; their builds failed"
        );
        fx.assert_idle("after the failed builds").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. Readers complete under writers on every writer path (T14; INV-9)
// ═════════════════════════════════════════════════════════════════════════════

/// The writer paths T14 drives, each through its own `LayerStore` on the
/// shipped-ceiling client (so the structural cache can hold the layer where
/// the path needs it to) reporting to the fixture's metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterPath {
    /// One inline `SetField` per commit: the update script's fast mode.
    Fast,
    /// One new section per commit, cached: the structural route
    /// (`structural_full` for the cold start, `structural` after).
    CachedStructural,
    /// One new section per commit on a store whose structural cache cannot
    /// hold the layer (a 1-byte budget: `OverWeight`): the WATCH fallback.
    UncacheableFallback,
}

impl WriterPath {
    /// The `backend` label the writer's fence waits are observed under.
    fn backend(self) -> &'static str {
        match self {
            WriterPath::UncacheableFallback => "txn",
            _ => "lua",
        }
    }

    /// The `path` labels the writer's refused attempts are counted on.
    fn paths(self) -> &'static [&'static str] {
        match self {
            WriterPath::Fast => &["fast"],
            WriterPath::CachedStructural => &["structural", "structural_full"],
            WriterPath::UncacheableFallback => &["watch"],
        }
    }

    fn delta(self, n: u64) -> OwnedDelta {
        match self {
            WriterPath::Fast => OwnedDelta {
                set_fields: vec![SetField {
                    section_id: 1,
                    key_name: "t14".to_owned(),
                    value_or_ext_hash: n.to_le_bytes().to_vec(),
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                ..Default::default()
            },
            _ => sections_delta(&[10_000 + n]),
        }
    }
}

/// T14 (INV-9): readers of a layer above the atomic-read ceiling complete
/// while a writer commits continuously on `path`. A brand-new subscriber
/// joins through `handle_subscribe_read_layer` and a checkpoint assembly reads
/// the layer, each through a fenced read whose window the fan-out's delay seam
/// holds open for [`HOLD`] — far longer than the gap between two of the
/// writer's commits. The catch-up is served at its cut and ends with its
/// caught-up ack, the assembly is written (a proven cut), neither consumer
/// records a `fence_broken` read, and the writer records at least one fence
/// wait, on its own path's backend and conflict cell: the writer waited at
/// the fence rather than breaking the reads.
///
/// Without the writer side of the fence — the update script's step 1d for
/// the fast and structural paths, the WATCH path's fence handling (B6) for the
/// fallback — the writer's commits land during each hold, every release finds
/// the header moved, and the catch-up ends `UNAVAILABLE` after
/// `FENCE_READ_ATTEMPTS` broken reads (and the assembly with an error).
async fn readers_complete_under_a_writer_on(path: WriterPath) {
    use std::sync::atomic::AtomicU64;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::assembler::{
        assemble_layer_checkpoint, AssemblyDecision, NoLeaseToKeep,
    };
    use live_edit::storage::structural_cache::StructuralCache;
    use live_edit::storage::test_support::relay::RelayedUpdates;

    let label = match path {
        WriterPath::Fast => "fast",
        WriterPath::CachedStructural => "structural",
        WriterPath::UncacheableFallback => "fallback",
    };
    let fx = Fx::new(&format!("t14-{label}"), Shape::new(2, ROOMY_BUDGET_MS)).await;
    let mut writer_store = LayerStore::new(fx.tv.client().clone(), fx.keys().clone(), 4096)
        .with_metrics(Arc::clone(&fx.metrics));
    if path == WriterPath::UncacheableFallback {
        writer_store = writer_store.with_structural_cache(StructuralCache::new(1));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let made = Arc::new(AtomicU64::new(0));
    let writer = {
        let (stop, made) = (Arc::clone(&stop), Arc::clone(&made));
        let (session, layer, oid) = (fx.session_id.clone(), fx.layer.clone(), fx.oid);
        tokio::spawn(async move {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                writer_store
                    .update_layer_relayed(
                        &session,
                        &layer,
                        oid,
                        Some(path.delta(n)),
                        "t14-writer",
                        n,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{path:?}: the writer's commit {n} failed: {e:?}"));
                n += 1;
                made.store(n, Ordering::Relaxed);
            }
            n
        })
    };
    await_condition(
        || made.load(Ordering::Relaxed) >= 3,
        "the writer's first commits",
    )
    .await;

    // The catch-up: its flight's fenced read held open for HOLD.
    fx.hold_next_fenced_read();
    let mut joiner = fx.join("t14-joiner").await;
    let cut = recv_served(&mut joiner, fx.oid, "t14-joiner").await;

    // The checkpoint assembly: its fenced read held open for HOLD too.
    fx.hold_next_fenced_read();
    let checkpoints = InMemoryCheckpointStore::new();
    let decision = tokio::time::timeout(
        DRAIN,
        assemble_layer_checkpoint(
            &fx.client,
            fx.keys(),
            fx.oid,
            &checkpoints,
            SERVICE,
            &mut NoLeaseToKeep,
        ),
    )
    .await
    .unwrap_or_else(|_| panic!("{path:?}: the assembly did not finish within {DRAIN:?}"))
    .unwrap_or_else(|e| panic!("{path:?}: the assembly must not fail; got {e:?}"));
    let snap_seq = match decision {
        AssemblyDecision::Written(outcome) => outcome.snap_seq,
        other => panic!("{path:?}: the assembly must write a proven generation; got {other:?}"),
    };

    stop.store(true, Ordering::Relaxed);
    let commits = tokio::time::timeout(DRAIN, writer)
        .await
        .expect("the writer stops")
        .expect("the writer must not panic");
    let seq = fx.seq().await;
    assert!(
        commits >= 3 && seq > snap_seq && snap_seq >= cut,
        "{path:?}: the writer kept committing through both reads (commits {commits}, cut \
         {cut}, snap_seq {snap_seq}, seq now {seq})"
    );

    let broken = |consumer: &str| {
        fx.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", consumer),
                ("path", "fenced"),
                ("outcome", "fence_broken"),
            ],
        ) as u64
    };
    let fenced_ok = |consumer: &str| {
        fx.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", consumer),
                ("path", "fenced"),
                ("outcome", "ok"),
            ],
        ) as u64
    };
    assert_eq!(
        (broken("catchup"), broken("checkpoint")),
        (0, 0),
        "{path:?}: no fenced read was broken by the writer"
    );
    assert!(
        fenced_ok("catchup") >= 1 && fenced_ok("checkpoint") >= 1,
        "{path:?}: both readers took the fenced read (vacuity guard)"
    );
    assert_eq!(
        fx.stable_cuts("verified", "brand_new"),
        1,
        "{path:?}: the catch-up served a proven cut"
    );
    assert!(
        fx.metric(
            "live_edit_update_layer_fence_wait_seconds_count",
            &[("backend", path.backend())],
        ) >= 1.0,
        "{path:?}: the writer waited at the fence (backend {})",
        path.backend()
    );
    let refused: f64 = path
        .paths()
        .iter()
        .map(|p| {
            fx.metric(
                "live_edit_update_layer_conflicts_total",
                &[("path", p), ("kind", "snapshot_fence_wait")],
            )
        })
        .sum();
    assert!(
        refused >= 1.0,
        "{path:?}: the writer's refused attempts are counted on its path {:?}",
        path.paths()
    );
    drop(joiner);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_complete_under_a_fast_path_writer() {
    within_test_budget(readers_complete_under_a_writer_on(WriterPath::Fast)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_complete_under_a_cached_structural_writer() {
    within_test_budget(readers_complete_under_a_writer_on(
        WriterPath::CachedStructural,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_complete_under_a_writer_on_the_uncacheable_watch_fallback() {
    within_test_budget(readers_complete_under_a_writer_on(
        WriterPath::UncacheableFallback,
    ))
    .await;
}
