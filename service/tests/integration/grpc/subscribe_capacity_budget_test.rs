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

//! b.xtb stages 2–3 and b.y3m Epic 2 — `SubscribeReadLayer`'s single capacity
//! budget, the catch-up slot and read-permit placement, and the reader
//! reservation a catch-up holds through its capacity waits, driven end-to-end
//! through `handle_subscribe_read_layer`.
//!
//! What is pinned here:
//!
//! * **One capacity budget** (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`) bounds
//!   the admission, cluster stream-reader, catch-up slot and snapshot
//!   read-permit waits. A slot, permit or reader wait that outlasts it ends
//!   the stream in-stream `UNAVAILABLE` (`catchup_capacity` /
//!   `reader_capacity`) **before any content or ack** — the only exception
//!   being a steady-state overwrite re-bootstrap, after its ack; a zero budget
//!   takes a free permit and is otherwise refused at once. The waits are each
//!   **charged** to the budget, so a later wait gets only what they left, and
//!   an overwrite re-bootstrap **refills** it. The charges and the refill are
//!   asserted on what the service recorded (`CapacityBudgetSeam`, checked by
//!   `assert_budget_chain`): what each wait was charged, what it started
//!   from, and the deadline it was handed — exact values, not a held slot
//!   raced against a deadline. In every regime, every read-permit wait comes
//!   before the first content message.
//! * **The slot covers reads and the build only; the permit, the atomic read
//!   only.** An up-to-date subscribe takes neither; a re-bootstrap nested
//!   inside a catch-up never waits for a slot while holding one. A read permit
//!   is taken only inside the slot, so
//!   a catch-up waiting for its slot holds no permit and one in its atomic
//!   read holds both; it is returned as the read returns, so the concurrent
//!   builds are bounded by the slot count, not the permits. It comes back to
//!   full capacity after an injected read failure and after every ending at
//!   the permit wait or during the read. The gate holds
//!   `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` permits, and
//!   `ValkeyClient::quit` closes it, ending a catch-up waiting at it
//!   `shutdown` before content.
//! * **An empty commit takes no read.** An entry without relay bytes above
//!   sequence 1 is an empty commit (SDD §12.2), sent as the empty delta at its
//!   sequence, based at the one before, with no read and no read permit by a
//!   caught-up subscriber, while a catch-up holds the replica's only permit and
//!   at any layer size.
//! * **Reserve, then register at the cut.** A catch-up reserves its stream
//!   reader before its capacity waits and registers its stream only at its
//!   proven read's cut — the cut it acks: at or below the atomic-read limits,
//!   under its slot at its atomic read's cut; above them, at its flight's cut
//!   holding no slot (next bullet). Through the waits it
//!   holds a reader reservation and queues nothing: commits landed while it
//!   waits for its slot and its permit are folded into its snapshot and never
//!   delivered again, and more than its backlog hard cap of them cannot end
//!   it. On a cluster the reservation keeps the reader bound to the layer's
//!   hash slot through the waits, and every ending while reserved — shutdown,
//!   session deletion, disconnect, a `catchup_capacity` refusal, the
//!   in-window `Deleted` scan, a `NotFound` re-bootstrap at the atomic
//!   read — releases the reader (and unbinds its slot) and leaves the read
//!   gate full. A registration the multiplexer refuses because it shut down
//!   meanwhile ends `shutdown` before content.
//! * **Above the atomic-read limits** (a forced-low ceiling here): after
//!   `TOO_LARGE` the catch-up returns its read permit and its slot and attaches
//!   to the layer's shared fenced-read flight, registering at the flight's
//!   cut, so commits landed during the waits are folded in and never
//!   delivered again. A shared read broken on every attempt ends the catch-up
//!   `UNAVAILABLE` before content — nothing unproven is served. A layer lost
//!   under the shared read is recovered by each attached catch-up on its own;
//!   each attaches again with no slot, and registers only at the retried
//!   read's cut, so commits landed during the recovery queue nothing and are
//!   delivered exactly once. (The flight's own waits, sizing, reader cap and
//!   endings are pinned in `subscribe_fenced_catch_up_test.rs`.)
//! * **The ending check.** A catch-up whose own stream registration ended
//!   after it registered — here, killed by the backlog hard cap while its
//!   build is held, or while it waits for the catch-up payload budget —
//!   ends with that ending before its first message: after its slot is
//!   released, or as soon as its payload-budget wait's periodic check sees
//!   it, never at that wait's capacity-budget deadline.
//!
//! The concurrent-catch-up and admission basics (FIFO order, endings of a
//! queued wait, the reserved trailer slot) live in
//! `subscribe_admission_test.rs`; the read gate's own FIFO, deadline,
//! zero-budget, drop and close semantics in `storage::subscriber_admission`'s
//! unit tests; the fan-out-window matrix and the proven read's sequence
//! regression in `subscribe_regimes_test.rs`, beside the steady state's
//! relay, self-echo and empty-commit order.
//!
//! # Holding a catch-up where no seam is
//!
//! There is no seam between a catch-up's registration and its build, nor
//! inside its atomic read. Two test-side stand-ins hold it there instead, with
//! no production hook: [`ParkingExtStore`], an external-value store whose
//! ranged reads park while armed — a catch-up's build reads its layer's
//! external values through it, after the registration, holding its slot — and
//! `test_support::stall_pool` over the snapshot lane, which parks a catch-up
//! inside its atomic read (behind a `BLPOP` on the lane's one connection),
//! holding its slot and its read permit.
//!
//! # A second, silent activation condition
//!
//! The tests that park a catch-up at a seam, record the budget's charges
//! (`CapacityBudgetSeam`), inject a snapshot-read fault, shrink the stream
//! multiplexer's sink caps (`StreamMux::set_knobs_for_test`) or override the
//! outer send deadline exist only under the **`test-fault-injection`** cargo
//! feature and are `#[cfg]`-gated on it: without it the run reports fewer
//! tests, not a pass. The tests that only hold catch-up slots and read permits
//! themselves (`CatchUpGate::acquire`, `CatchUpReadGate::acquire`) run in every
//! feature shape. The `cluster` module additionally self-skips unless
//! `LIVE_EDIT_TEST_CLUSTER_URL` is set.
//!
//! ```text
//! cargo nextest run -p live-edit --test integration \
//!     --features test-support,test-fault-injection grpc::subscribe_capacity_budget_test
//! ```
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a measurement: `SETTLE` (10 s) per awaited
//! message, park or polled condition, `DRAIN` (20 s) once per catch-up drain
//! (including `assert_serves_every_commit_once`'s content, ack and three live
//! commits). The healthy path pays none of them. Three budgets ARE paid,
//! deliberately, once per test that asserts on them: `SHORT_BUDGET_MS`
//! (300 ms), where the refusal under test is the budget running out,
//! `REACQUIRE_REFUSAL_BUDGET_MS` (3 s, less the promptly-released waits
//! before it) for the refused re-acquire after a `Restored` recovery, and
//! `CLUSTER_SHORT_BUDGET_MS` (500 ms) for a cluster reader refusal (by the
//! budget, or by the deadline a storage-level subscription is given). The
//! two budget-charge tests also
//! pay their spends, which exist to spend budget and are never
//! synchronisation: 1 s standalone (`ADMISSION_SPEND` + `FIRST_SLOT_SPEND`)
//! and 0.5 s on a cluster (`READER_SPEND`). They run on `PARKED_BUDGET_MS`
//! and assert on the charges the service recorded, so no outcome there turns
//! on a spend racing a deadline. `PARKED_BUDGET_MS` (60 s) is never paid: it
//! only keeps a queued wait from refusing itself before the event under test,
//! and every await on such a wait is bounded by `SETTLE`. The heaviest tests —
//! the steady-state overwrite refusal, the layer lost under a shared fenced
//! read and the refused re-acquire after a restored recovery — each sum to
//! ≈ 90 s at every budget's full value (the layer lost under a shared read:
//! five polled conditions or slot waits at `SETTLE` + two drains; the refused
//! re-acquire: nine `SETTLE`s, the budget running out inside one of them).
//! The fenced catch-up whose reads are all broken sums to ≈ 70 s
//! (seven `SETTLE`-bounded waits). The slot-and-permit wait sums to ≈ 60 s (three `SETTLE`s, one drain
//! and the polled conditions after it, plus its twelve commits); the
//! concurrent-builds test to ≈ 70 s (five polled conditions and one drain over
//! all five joiners); the stalled-read cancellation and every permit-wait
//! ending to ≤ 60 s (at most six `SETTLE`-bounded waits each); the standalone
//! budget-charge test to ≈ 81 s (six `SETTLE`s, one drain and 1 s of spends).
//! Of the empty-commit tests, the empty commits delivered to three caught-up
//! subscribers sum to ≈ 70 s (three acks and two drains). Of the `/scaling/v1`
//! counter tests, the re-bootstrap before the first ack sums to ≈ 60 s (a
//! park, the cleanup wait and a release at `SETTLE`, one drain, one polled
//! condition) and the steady-state re-bootstrap after it to ≈ 70 s (two
//! drains, one polled condition, one header-and-body message 20 s). The rest
//! sum to ≤ 50 s. All are inside nextest's 120 s hang-killer. On the healthy path the
//! budget-charge tests take their spends plus a few hundred milliseconds, the
//! rest a few hundred milliseconds.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::subscribe::{BudgetCharge, BudgetWait, CapacityBudgetSeam};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, CATCHUP_WAIT_RESULT_ACQUIRED, CATCHUP_WAIT_RESULT_CANCELLED,
    CATCHUP_WAIT_RESULT_TIMED_OUT, STREAM_EXIT_CATCHUP_CAPACITY,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::SnapshotReadConsumer;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::snapshot_flights::FlightState;
use live_edit::storage::subscriber_admission::{CatchUpPermit, CatchUpReadPermit};
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;

use super::helpers::{
    make_create_data, make_create_header, make_request, streaming_create, streaming_update,
    AllowAllPermissions, StubUserinfo,
};

// ── Budgets and constants ───────────────────────────────────────────────────

/// Upper bound on one awaited message, park or polled condition.
const SETTLE: Duration = Duration::from_secs(10);

/// One budget for a whole catch-up drain, however many messages it holds.
const DRAIN: Duration = Duration::from_secs(20);

/// A capacity budget short enough that a refusal under test costs little.
const SHORT_BUDGET_MS: u64 = 300;

/// A capacity budget no queued wait here reaches: the wait ends by the event
/// the test fires, not by the budget.
const PARKED_BUDGET_MS: u64 = 60_000;

/// `Metrics` / `service` label of every fixture here. The same value as
/// `crate::common::RECOVERY_SERVICE_LABEL`, so the recovery-attempt counter
/// reads through `crate::common::recovery_metric`.
const SERVICE: &str = "test";

/// The in-stream status of a catch-up refused its slot.
const CATCHUP_CAPACITY_MSG: &str = "this replica is at its concurrent catch-up limit \
     (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS); retry";

/// The in-stream status of a catch-up refused its snapshot read permit.
const CATCHUP_READ_CAPACITY_MSG: &str = "this replica is at its concurrent catch-up snapshot-read \
     limit (LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY); retry";

/// `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` for a forced-low ceiling: below
/// the item bound of any layer here, so every catch-up's atomic read answers
/// `TOO_LARGE` at its early-out and the catch-up attaches to the layer's
/// shared fenced-read flight.
const FORCED_LOW_ITEMS: u64 = 1;

/// Commits written into a subscription's sink to kill it at the shrunk hard
/// cap ([`TINY_HARD_CAP`]): comfortably more than the cap.
#[cfg(feature = "test-fault-injection")]
const OVERFLOW_COMMITS: u64 = 12;

/// The shrunk per-subscription backlog hard cap the ending-check tests
/// install. The soft watchdog is pushed out of reach, so only this cap kills.
#[cfg(feature = "test-fault-injection")]
const TINY_HARD_CAP: usize = 8;

// ── Fixture ─────────────────────────────────────────────────────────────────

/// A client with a caller-chosen catch-up limit and capacity budget, its own
/// key prefix, and one `Metrics` handle installed on both the client (gates,
/// readers) and the service (stream exits, stable cut), so one scrape sees
/// every series.
struct CapacityFixture {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
    metrics: Arc<Metrics>,
    /// Set by [`Self::purge_and_quit`]: the keys are gone and the client's
    /// connections are closed, so `Drop` has nothing to purge and no
    /// connection to purge it on.
    quit: bool,
}

/// The atomic-read item ceiling a fixture's client runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ceiling {
    /// The shipped default: every layer here is far below it, so catch-ups
    /// take the atomic read.
    Shipped,
    /// [`FORCED_LOW_ITEMS`]: every catch-up reads through its layer's shared
    /// fenced-read flight.
    ForcedLow,
}

impl CapacityFixture {
    /// Single-node Valkey, two stream readers, unlimited subscriptions, one
    /// read permit (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` default).
    async fn setup(max_catchups: usize, budget_ms: u64) -> Self {
        Self::setup_with(None, 2, 0, max_catchups, budget_ms).await
    }

    /// [`Self::setup`] with a forced-low atomic-read ceiling: every layer is
    /// above it, so every catch-up reads through a shared fenced-read flight.
    #[cfg(feature = "test-fault-injection")]
    async fn setup_above_ceiling(max_catchups: usize, budget_ms: u64) -> Self {
        Self::setup_full(None, 2, 0, max_catchups, budget_ms, Ceiling::ForcedLow).await
    }

    /// `max_subscriptions` is the subscription ceiling
    /// (`Config::subscriber_max_subscriptions`); 0 is unlimited, which only a
    /// `Config` built field by field, as here, can hold.
    async fn setup_with(
        url: Option<String>,
        readers: u32,
        max_subscriptions: usize,
        max_catchups: usize,
        budget_ms: u64,
    ) -> Self {
        Self::setup_full(
            url,
            readers,
            max_subscriptions,
            max_catchups,
            budget_ms,
            Ceiling::Shipped,
        )
        .await
    }

    /// [`Self::setup`] with `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` set
    /// to `read_concurrency` read permits.
    async fn setup_read_concurrency(
        max_catchups: usize,
        budget_ms: u64,
        read_concurrency: u32,
    ) -> Self {
        let mut config = Self::config(None, 2, 0, max_catchups, budget_ms, Ceiling::Shipped);
        config.catchup_snapshot_read_concurrency = read_concurrency;
        Self::connect(config).await
    }

    async fn setup_full(
        url: Option<String>,
        readers: u32,
        max_subscriptions: usize,
        max_catchups: usize,
        budget_ms: u64,
        ceiling: Ceiling,
    ) -> Self {
        Self::connect(Self::config(
            url,
            readers,
            max_subscriptions,
            max_catchups,
            budget_ms,
            ceiling,
        ))
        .await
    }

    /// The fixture's client configuration.
    fn config(
        url: Option<String>,
        readers: u32,
        max_subscriptions: usize,
        max_catchups: usize,
        budget_ms: u64,
        ceiling: Ceiling,
    ) -> live_edit::config::Config {
        let mut config = crate::common::test_config();
        if let Some(url) = url {
            config.valkey_url = url;
        }
        config.subscriber_pool_size = readers;
        config.subscriber_admission_wait_ms = budget_ms;
        config.subscriber_max_subscriptions = max_subscriptions;
        config.subscriber_max_concurrent_catchups = max_catchups;
        config.subscriber_admission_max_waiters = None;
        match ceiling {
            Ceiling::Shipped => {}
            Ceiling::ForcedLow => config.snapshot_atomic_read_max_items = FORCED_LOW_ITEMS,
        }
        config.valkey_connect_timeout_secs = 30;
        config
    }

    /// Connect a client with `config` under a fresh key prefix.
    async fn connect(config: live_edit::config::Config) -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("CapacityFixture: connect must succeed");
        let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        Self {
            key_schema: KeySchema::new(&prefix),
            client,
            prefix,
            metrics,
            quit: false,
        }
    }

    /// Delete every key under the prefix, one `UNLINK` per key so it is
    /// cluster-safe. `Drop` does this on a multi-thread runtime; a
    /// current-thread test calls it itself.
    async fn purge(&self) {
        use fred::interfaces::KeysInterface;
        for key in crate::common::scan_keys(&self.client, &format!("{}*", self.prefix)).await {
            let _ = self.client.pool.unlink::<(), _>(key).await;
        }
    }

    /// Purge, then run the client's shutdown step, `ValkeyClient::quit`: it
    /// closes the admission, catch-up slot and read gates, then the
    /// connections. The service's `valkey` is a clone of this client, so it
    /// shares those gates and connections.
    async fn purge_and_quit(&mut self) {
        self.purge().await;
        self.quit = true;
        self.client.quit().await;
    }
}

impl Drop for CapacityFixture {
    fn drop(&mut self) {
        if self.quit {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }
        tokio::task::block_in_place(|| handle.block_on(self.purge()));
    }
}

/// A service over `fx`, with no seam installed. Tests install the seams they
/// drive on the returned value.
fn build_service(fx: &CapacityFixture, maxlen: u64) -> LiveEditService {
    let layer_store = Arc::new(LayerStore::new(
        fx.client.clone(),
        fx.key_schema.clone(),
        maxlen,
    ));
    let metrics = Arc::clone(&fx.metrics);
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );
    let mut session_recovery_config = crate::common::test_config();
    session_recovery_config.checkpoint_enabled = false;
    let session_recovery = Arc::new(live_edit::checkpoint::SessionRecoveryHook::new(
        Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new()),
        fx.client.clone(),
        fx.key_schema.clone(),
        Arc::new(session_recovery_config),
        Arc::clone(&metrics),
    ));
    LiveEditService {
        session_store: Arc::new(SessionStore::new(fx.client.clone(), fx.key_schema.clone())),
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            fx.client.clone(),
            fx.key_schema.clone(),
        ),
        valkey: fx.client.clone(),
        key_schema: fx.key_schema.clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            fx.client.clone(),
            fx.key_schema.clone(),
            Arc::clone(&metrics),
        ),
        metrics,
        service_label: SERVICE.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        service_token_provider: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_xrange_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        outer_send_deadline_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_chunk_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        run_catch_up_inner_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_out_of_window_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_state_read_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_build_turn_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        capacity_budget_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        join_registration_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_replay_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        lease_keeper_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_channel_seam_for_test: None,
    }
}

// ── Layer and commit builders ───────────────────────────────────────────────

/// A session and a layer on it; returns `(session_id, object_id)`. The layer
/// is created at sequence 1.
async fn create_layer(fx: &CapacityFixture, svc: &LiveEditService, layer: &str) -> (String, u64) {
    create_layer_carrying(fx, svc, layer, &crate::common::owned_delta::empty_delta()).await
}

/// [`create_layer`] whose `CreateLayer` content is `delta`: the layer's state
/// at sequence 1, which its seed stream entry `1-0` stands for.
async fn create_layer_carrying(
    fx: &CapacityFixture,
    svc: &LiveEditService,
    layer: &str,
    delta: &OwnedDelta,
) -> (String, u64) {
    // The session create and every handler call these helpers await are
    // boxed, as tonic polls a handler from the heap. Inline, each future is
    // held in every helper future above it, and in a debug build each of
    // their poll frames on the test thread's stack holds copies of it; the
    // out-of-window slot-and-permit tests then came within 256 KiB of
    // std's default 2 MiB test-thread stack.
    let session_id = Box::pin(
        SessionStore::new(fx.client.clone(), fx.key_schema.clone()).create(
            &format!("{layer}-session"),
            "/stage",
            "test-user",
        ),
    )
    .await
    .expect("session creation must succeed");
    let ompf = delta_to_ompf_bytes(delta, &[]);
    let object_id = Box::pin(handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(&session_id, layer),
            make_create_data(ompf),
        ])),
    ))
    .await
    .expect("CreateLayer must succeed")
    .into_inner()
    .object_id;
    (session_id, object_id)
}

/// One commit adding section `section_id` under the root, written by a client
/// no subscriber here uses (so self-echo never hides it). Advances the layer's
/// sequence by one and relays one steady-state message.
async fn commit(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    section_id: u64,
) {
    commit_sections(svc, session_id, layer, object_id, &[section_id]).await;
}

/// A delta adding each of `section_ids` under the root, as [`commit`] writes.
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

/// One commit adding every section of `section_ids` under the root, written
/// like [`commit`]. Advances the layer's sequence by one.
async fn commit_sections(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    section_ids: &[u64],
) {
    let section_id = section_ids[0];
    let delta = sections_delta(section_ids);
    let messages = vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: layer.to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: section_id * 10,
                    client_id: "capacity-writer".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk {
                    data: delta_to_ompf_bytes(&delta, &[]),
                },
            )),
        },
    ];
    // Boxed: see `create_layer_carrying`.
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(messages)),
    ))
    .await
    .unwrap_or_else(|e| panic!("UpdateLayer (section {section_id}) must succeed: {e}"));
}

/// The blob [`commit_with_external_value`] writes out of line.
#[cfg(feature = "test-fault-injection")]
const EXT_BLOB: &[u8] = b"capacity-test-external-value-blob";

/// One commit creating section `section_id` under the root with one field
/// whose value is held out of line ([`EXT_BLOB`], OMPF Part 1), so every
/// catch-up build that serves the field resolves it through the service's
/// `ExternalValueStore` ([`ParkingExtStore`]). Advances the sequence by one.
#[cfg(feature = "test-fault-injection")]
async fn commit_with_external_value(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    section_id: u64,
) {
    use oodf_ffi::{CommandCreateSection, SetField};
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: format!("ext-{section_id}"),
            section_type: 0,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: "ext:value".to_owned(),
            // The blob's own content hash: the upload content check refuses
            // any other claim.
            value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(EXT_BLOB)
                .to_vec(),
            ext_value_size: EXT_BLOB.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        ..Default::default()
    };
    let messages = vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: layer.to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: section_id * 10,
                    client_id: "capacity-writer".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk {
                    data: delta_to_ompf_bytes(&delta, &[EXT_BLOB]),
                },
            )),
        },
    ];
    // Boxed: see `create_layer_carrying`.
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(messages)),
    ))
    .await
    .unwrap_or_else(|e| panic!("UpdateLayer (external value, section {section_id}): {e}"));
}

/// The writer of every [`empty_commit`]: no subscriber here uses it, so
/// self-echo never hides one.
const EMPTY_WRITER: &str = "empty-writer";

/// One empty commit (SDD §8.4): an `UpdateLayer` with a header and no chunks,
/// from [`EMPTY_WRITER`], as a client sends it. Advances the layer's sequence
/// by one; its stream entry carries no `ompf` field, which is what marks it
/// empty to a subscriber.
async fn empty_commit(svc: &LiveEditService, session_id: &str, layer: &str, object_id: u64) {
    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: 1,
                client_id: EMPTY_WRITER.to_owned(),
                operation_id: String::new(),
            },
        )),
    };
    // Boxed: see `create_layer_carrying`.
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(vec![header])),
    ))
    .await
    .expect("an empty UpdateLayer (header only) must succeed");
}

/// The delta an empty commit is delivered as: no change at all, a diff based
/// at `base_version`.
fn the_empty_delta(base_version: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version,
        ..Default::default()
    }
}

/// A layer whose stream no longer holds `2-0`, so a reconnect at
/// `have_version = 1` is out of window: five commits (sequence 6) and a
/// deterministic `XTRIM MINID 6-0`. Pair with a service built with a small
/// `maxlen`. Returns `(session_id, object_id)`.
async fn out_of_window_layer(
    fx: &CapacityFixture,
    svc: &LiveEditService,
    layer: &str,
) -> (String, u64) {
    use fred::interfaces::StreamsInterface;
    use fred::types::streams::XCap;
    let (session_id, object_id) = create_layer(fx, svc, layer).await;
    for section_id in 101..=105 {
        commit(svc, &session_id, layer, object_id, section_id).await;
    }
    let cap = XCap::try_from(("MINID", "=", "6-0")).expect("build MINID xcap");
    let _: () = fx
        .client
        .pool
        .xtrim(fx.key_schema.layer_stream(object_id), cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");
    (session_id, object_id)
}

/// Overwrite `layer` with an empty new object; returns its object id.
#[cfg(feature = "test-fault-injection")]
async fn overwrite(svc: &LiveEditService, session_id: &str, layer: &str) -> u64 {
    let ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    // Boxed: see `create_layer_carrying`.
    Box::pin(handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(session_id, layer),
            make_create_data(ompf),
        ])),
    ))
    .await
    .expect("the CreateLayer overwrite must succeed")
    .into_inner()
    .object_id
}

/// Shrink every new subscription's sink: a hard cap of [`TINY_HARD_CAP`]
/// entries and a soft watchdog out of reach, so only the hard cap can kill.
/// Applies to registrations made after the call.
#[cfg(feature = "test-fault-injection")]
fn install_tiny_hard_cap(svc: &LiveEditService) {
    let knobs = svc.valkey.stream_mux.knobs();
    svc.valkey
        .stream_mux
        .set_knobs_for_test(live_edit::storage::stream_mux::StreamMuxKnobs {
            channel_capacity: TINY_HARD_CAP / 2,
            backlog_hard_cap: TINY_HARD_CAP,
            send_watchdog: Duration::from_secs(600),
            ..knobs
        });
}

// ── Stream helpers ──────────────────────────────────────────────────────────

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    // Boxed: see `create_layer_carrying`.
    Box::pin(handle_subscribe_read_layer(svc, request))
        .await
        .unwrap_or_else(|s| panic!("the subscribe must be admitted: {s:?}"))
        .into_inner()
}

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
    match next_item(stream, what).await {
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

/// The caught-up ack, which must carry `(object_id, sequence_num)`.
async fn recv_ack(stream: &mut SubStream, object_id: u64, sequence_num: u64) {
    let (ack, _) = recv_message(stream, "the caught-up ack").await;
    assert_eq!(
        ack.total_size, 0,
        "expected the caught-up ack, got content: {ack:?}"
    );
    assert_eq!(
        (ack.object_id, ack.sequence_num),
        (object_id, sequence_num),
        "the caught-up ack must carry the version the catch-up served"
    );
}

/// Every content header of a catch-up (in order) and its caught-up ack, under
/// one [`DRAIN`] deadline.
async fn recv_catch_up(
    stream: &mut SubStream,
) -> (Vec<SubscribeReadLayerHeader>, SubscribeReadLayerHeader) {
    tokio::time::timeout(DRAIN, async {
        let mut contents = Vec::new();
        loop {
            let (header, _) = recv_message(stream, "a catch-up message").await;
            if header.total_size == 0 {
                return (contents, header);
            }
            contents.push(header);
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not finish within {DRAIN:?}"))
}

/// Section ids named by an OMPF payload's struct commands.
fn section_ids(ompf: &[u8]) -> HashSet<u64> {
    let container = live_edit::domain::ompf::OmpfContainer::parse(ompf)
        .expect("the payload is an OMPF container");
    if container.delta_bytes.is_empty() {
        return HashSet::new();
    }
    oodf_ffi::parse_delta(container.delta_bytes)
        .expect("the payload carries an OODF delta")
        .struct_commands
        .iter()
        .map(|cmd| match cmd {
            StructCommand::CreateSection(c) => c.section_id,
            StructCommand::DeleteSection(c) => c.section_id,
            StructCommand::MoveSection(c) => c.section_id,
            StructCommand::ReorderChildren(c) => c.section_id,
            StructCommand::DiffChangeSection(c) => c.section_id,
            StructCommand::DiffDeleteSection(c) => c.section_id,
        })
        .collect()
}

// ── Metric readers ──────────────────────────────────────────────────────────

/// `live_edit_subscriber_stream_exits_total{reason}`; pre-initialised, so an
/// absent series is a misspelled reason and fails loudly.
fn exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("stream_exits_total{{reason=\"{reason}\"}} is pre-initialised"))
        as u64
}

/// `live_edit_subscribe_catchup_wait_seconds_count{result}`; `0` while absent
/// (a histogram series appears with its first observation).
fn catchup_waits(metrics: &Metrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscribe_catchup_wait_seconds_count",
        &[("result", result)],
    )
    .unwrap_or(0.0) as u64
}

/// A pre-initialised unlabelled gauge or counter.
fn series(metrics: &Metrics, name: &str) -> i64 {
    sample_value(&render(metrics), name, &[])
        .unwrap_or_else(|| panic!("{name} is pre-initialised in Metrics::new")) as i64
}

const INFLIGHT: &str = "live_edit_subscribe_catchups_inflight";
const WAITING: &str = "live_edit_subscribe_catchups_waiting";
const ADMITTED: &str = "live_edit_subscriber_admitted";
/// Registered streams the multiplexer reads; a reader reservation is not one.
const STREAMS: &str = "live_edit_stream_mux_streams";

/// `live_edit_stream_mux_overflow_kills_total{bound}` — `entries` or `bytes`,
/// each cell read on its own (both are pre-initialised, so an unlabelled read
/// would return whichever renders first).
#[cfg(feature = "test-fault-injection")]
fn overflow_kills(metrics: &Metrics, bound: &str) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_stream_mux_overflow_kills_total",
        &[("bound", bound)],
    )
    .unwrap_or_else(|| panic!("overflow_kills_total{{bound=\"{bound}\"}} is pre-initialised"))
        as i64
}

/// Diff builds completed on the fixture's `Metrics`
/// (`live_diff_case_total`, summed over its labels): one per catch-up state
/// read that got as far as building a payload.
#[cfg(feature = "test-fault-injection")]
fn diff_builds(metrics: &Metrics) -> u64 {
    render(metrics)
        .lines()
        .filter_map(|line| line.strip_prefix("live_diff_case_total{"))
        .filter_map(|rest| rest.rsplit_once(' '))
        .map(|(_, value)| {
            value
                .trim()
                .parse::<f64>()
                .expect("a counter sample is a number") as u64
        })
        .sum()
}

/// `live_subscribe_stable_cut_total{outcome, regime, service}`.
fn stable_cuts(metrics: &Metrics, outcome: &str, regime: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_subscribe_stable_cut_total",
        &[
            ("outcome", outcome),
            ("regime", regime),
            ("service", SERVICE),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_catchup_snapshot_reads_in_flight{class="atomic"}`: catch-up
/// atomic read permits held. Pre-registered.
fn reads_in_flight(metrics: &Metrics) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_catchup_snapshot_reads_in_flight",
        &[("class", "atomic")],
    )
    .expect("reads_in_flight{class=\"atomic\"} is pre-registered") as i64
}

/// `live_edit_catchup_snapshot_read_wait_seconds_count{consumer="catchup",
/// class="atomic"}`: atomic read-permit acquisitions that ended, however they
/// ended (the test's own included); `0` while absent (not pre-seeded).
fn read_permit_waits(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_catchup_snapshot_read_wait_seconds_count",
        &[("consumer", "catchup"), ("class", "atomic")],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_snapshot_read_total{consumer="catchup", path, outcome}`; `0`
/// while absent.
fn catchup_snapshot_reads(metrics: &Metrics, path: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "catchup"),
            ("path", path),
            ("outcome", outcome),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// The sum of every sample of the counter family `metric` (a `_total` or a
/// histogram's `_count`), over all its label sets; `0` while none is present.
fn sum_of_samples(metrics: &Metrics, metric: &str) -> u64 {
    render(metrics)
        .lines()
        .filter(|line| line.starts_with(&format!("{metric}{{")))
        .filter_map(|line| line.rsplit_once(' '))
        .map(|(_, value)| {
            value
                .trim()
                .parse::<f64>()
                .expect("a counter sample is a number") as u64
        })
        .sum()
}

/// Every snapshot read (`live_edit_snapshot_read_total`), by any consumer, on
/// any path, however it ended.
fn all_snapshot_reads(metrics: &Metrics) -> u64 {
    sum_of_samples(metrics, "live_edit_snapshot_read_total")
}

/// Every catch-up read-permit acquisition that ended
/// (`live_edit_catchup_snapshot_read_wait_seconds_count`), by any consumer,
/// in any class, however it ended (the test's own included).
fn all_read_permit_waits(metrics: &Metrics) -> u64 {
    sum_of_samples(
        metrics,
        "live_edit_catchup_snapshot_read_wait_seconds_count",
    )
}

/// `live_edit_catchup_snapshot_flight_total{role}` (`leader` | `joined_read`):
/// catch-ups that created, or attached to, a shared fenced-read flight.
#[cfg(feature = "test-fault-injection")]
fn flight_roles(metrics: &Metrics, role: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_catchup_snapshot_flight_total",
        &[("role", role)],
    )
    .unwrap_or(0.0) as u64
}

/// The OODF delta of a content message's OMPF body.
fn delta_of(ompf: &[u8]) -> OwnedDelta {
    let container = live_edit::domain::ompf::OmpfContainer::parse(ompf)
        .expect("the payload is an OMPF container");
    oodf_ffi::parse_delta(container.delta_bytes).expect("the payload carries an OODF delta")
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
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Take one catch-up slot for the test itself, with a deadline it never
/// reaches.
async fn hold_catchup_slot(svc: &LiveEditService) -> CatchUpPermit {
    svc.valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600))
        .await
        .unwrap_or_else(|r| panic!("the test's own slot must be free: {r:?}"))
}

/// Take one catch-up read permit for the test itself — the replica's only one
/// at the default read concurrency — with a deadline it never reaches.
async fn hold_read_permit(svc: &LiveEditService) -> CatchUpReadPermit {
    svc.valkey
        .catchup_read_gate
        .acquire(
            tokio::time::Instant::now() + Duration::from_secs(3_600),
            SnapshotReadConsumer::Catchup,
        )
        .await
        .unwrap_or_else(|r| panic!("the test's own read permit must be free: {r:?}"))
}

/// Whether the read gate holds every permit it has: none taken, none leaked.
fn read_gate_is_full(svc: &LiveEditService) -> bool {
    let gate = &svc.valkey.catchup_read_gate;
    gate.available() == gate.limit()
}

/// Everything a shared fenced-read flight takes is back (polled within
/// [`SETTLE`]): no flight registered, every catch-up slot and fenced read
/// permit free, nothing reserved or over-committed on the catch-up memory
/// budget, no build counted.
#[cfg(feature = "test-fault-injection")]
async fn assert_flights_idle(svc: &LiveEditService, what: &str) {
    let client = &svc.valkey;
    await_condition(
        || {
            let budget = &client.catchup_build_budget;
            svc.snapshot_flights.is_empty()
                && client.catchup_gate.limit() == Some(client.catchup_gate.available())
                && client.catchup_fenced_read_gate.available()
                    == client.catchup_fenced_read_gate.limit()
                && budget.reserved_bytes() == 0
                && budget.overcommit_bytes() == 0
                && series(&svc.metrics, "live_edit_catchup_builds_in_progress") == 0
        },
        &format!(
            "{what}: no flight registered, every slot and fenced permit free, the memory budget \
             back in full and no build counted"
        ),
    )
    .await;
}

// ── A catch-up held inside its build ────────────────────────────────────────

/// An external-value store that, while armed, parks every `read_value_range`
/// call until the test releases it, and otherwise is
/// [`MockExternalValueStore`] (whose length probe, restore and ranged read are
/// the trait's defaults, over its `probe_values` and `read_batch`).
///
/// A catch-up's build lays its payload out from its external values' probed
/// lengths, reserves its catch-up payload memory, then reads each value in
/// ranges (b.ntm), after its stream registration and holding its catch-up
/// slot, so a layer carrying one small value
/// ([`commit_with_external_value`], one range) holds a catch-up at a point no
/// production seam reaches, once per build. (It parks the ranged read only:
/// the build probes its values' lengths twice, before the layout and at its
/// closing check, so parking the probe would park each build twice.)
///
/// It counts the calls parked at once and the most ever parked at once, which
/// is the number of catch-up builds running together.
#[cfg(feature = "test-fault-injection")]
struct ParkingExtStore {
    inner: MockExternalValueStore,
    armed: std::sync::atomic::AtomicBool,
    parked: std::sync::atomic::AtomicUsize,
    max_parked: std::sync::atomic::AtomicUsize,
    /// One permit per parked call released.
    release: tokio::sync::Semaphore,
}

#[cfg(feature = "test-fault-injection")]
impl ParkingExtStore {
    fn new() -> Self {
        Self {
            inner: MockExternalValueStore::new(),
            armed: std::sync::atomic::AtomicBool::new(false),
            parked: std::sync::atomic::AtomicUsize::new(0),
            max_parked: std::sync::atomic::AtomicUsize::new(0),
            release: tokio::sync::Semaphore::new(0),
        }
    }

    fn arm(&self) {
        self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn disarm(&self) {
        self.armed.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn parked(&self) -> usize {
        self.parked.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn max_parked(&self) -> usize {
        self.max_parked.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Wait (bounded by [`SETTLE`]) until exactly `n` calls are parked.
    async fn wait_until_parked(&self, n: usize) {
        await_condition(|| self.parked() == n, "the expected builds park").await;
    }

    /// Let one parked call go on to the mock.
    fn release_one(&self) {
        self.release.add_permits(1);
    }

    /// Park while armed, until released.
    async fn park(&self) -> Result<(), live_edit::error::ServiceError> {
        use std::sync::atomic::Ordering::SeqCst;
        if !self.armed.load(SeqCst) {
            return Ok(());
        }
        let now = self.parked.fetch_add(1, SeqCst) + 1;
        self.max_parked.fetch_max(now, SeqCst);
        self.release
            .acquire()
            .await
            .expect("the release semaphore is never closed")
            .forget();
        self.parked.fetch_sub(1, SeqCst);
        Ok(())
    }
}

#[cfg(feature = "test-fault-injection")]
#[async_trait::async_trait]
impl live_edit::storage::external_values::ExternalValueStore for ParkingExtStore {
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
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<live_edit::storage::external_values::ExtValueProbe, live_edit::error::ServiceError>
    {
        self.inner.probe_values(hashes).await
    }

    async fn read_value_range(
        &self,
        hash: &[u8; 32],
        start: u64,
        len: usize,
    ) -> Result<bytes::Bytes, live_edit::error::ServiceError> {
        self.park().await?;
        self.inner.read_value_range(hash, start, len).await
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

/// Install a [`ParkingExtStore`] on `svc` and return it.
#[cfg(feature = "test-fault-injection")]
fn install_parking_ext_store(svc: &mut LiveEditService) -> Arc<ParkingExtStore> {
    let store = Arc::new(ParkingExtStore::new());
    svc.ext_value_store = Arc::clone(&store) as _;
    store
}

// ═══════════════════════════════════════════════════════════════════════════
// The budget bounds the slot wait; the refusal comes before content
// ═══════════════════════════════════════════════════════════════════════════

/// Which catch-up a slot-timeout test refuses. Each reserves its stream
/// reader, then waits for the slot.
#[derive(Clone, Copy, Debug)]
enum RefusedCatchUp {
    BrandNew,
    InWindow,
}

/// A catch-up whose slot wait outlasts the capacity budget ends in-stream
/// `UNAVAILABLE` naming `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`, under
/// `reason="catchup_capacity"`, as the stream's FIRST item — no content, no
/// ack — records its wait as `timed_out`, and gives back its admission and
/// any reader reservation it held while waiting.
async fn a_slot_wait_that_outlasts_the_budget_is_refused_before_content(case: RefusedCatchUp) {
    let fx = CapacityFixture::setup(1, SHORT_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "slot-timeout";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 11).await;
    commit(&svc, &session_id, layer, object_id, 12).await;
    let held = hold_catchup_slot(&svc).await;

    let (have_object, have_version) = match case {
        RefusedCatchUp::BrandNew => (0, 0),
        _ => (object_id, 1),
    };
    let mut stream = subscribe(
        &svc,
        &session_id,
        layer,
        have_object,
        have_version,
        "refused",
    )
    .await;
    let status = first_status(&mut stream, "the catch-up capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{case:?}: {status:?}");
    assert_eq!(status.message(), CATCHUP_CAPACITY_MSG, "{case:?}");
    crate::common::refusal_detail::assert_catchup_slot_refusal(&status);

    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted under catchup_capacity",
    )
    .await;
    assert_eq!(
        catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
        1,
        "{case:?}: the slot wait that ran out the budget records one `timed_out`"
    );
    assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_CANCELLED), 0);
    assert_eq!(
        series(&fx.metrics, WAITING),
        0,
        "the refused wait left the queue"
    );
    assert_eq!(
        series(&fx.metrics, INFLIGHT),
        1,
        "only the test's own slot is held"
    );
    await_condition(
        || series(&fx.metrics, ADMITTED) == 0,
        "the refused subscription returns its admission",
    )
    .await;
    await_condition(
        || svc.valkey.stream_mux.snapshot().subscriptions() == 0,
        "a reader reserved before the wait is released",
    )
    .await;
    drop(held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_slot_wait_that_outlasts_the_budget_is_refused_catchup_capacity_before_content()
{
    a_slot_wait_that_outlasts_the_budget_is_refused_before_content(RefusedCatchUp::BrandNew).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnect_slot_wait_that_outlasts_the_budget_is_refused_before_content() {
    a_slot_wait_that_outlasts_the_budget_is_refused_before_content(RefusedCatchUp::InWindow).await;
}

/// An up-to-date subscribe takes no catch-up slot: with the only slot held and
/// a short budget it still gets its caught-up ack at once, records no slot
/// wait, and then relays live commits. Had it waited for a slot it would have
/// been refused `catchup_capacity` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_up_to_date_subscribe_takes_no_catch_up_slot() {
    let fx = CapacityFixture::setup(1, SHORT_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "up-to-date";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 21).await;
    let held = hold_catchup_slot(&svc).await;
    let acquired_before = catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED);

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 2, "up-to-date").await;
    recv_ack(&mut stream, object_id, 2).await;
    assert_eq!(
        catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
        acquired_before,
        "an up-to-date catch-up reads and builds nothing, so it takes no slot"
    );
    assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT), 0);
    assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_CANCELLED), 0);

    commit(&svc, &session_id, layer, object_id, 22).await;
    let (live, _) = recv_message(&mut stream, "the first live commit").await;
    assert_eq!(live.sequence_num, 3);
    assert_eq!(exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY), 0);
    drop(held);
}

// ═══════════════════════════════════════════════════════════════════════════
// Commits landed during a slot wait: the atomic cut, and the fenced cut
// ═══════════════════════════════════════════════════════════════════════════

/// Which catch-up regime a waited-catch-up case drives.
#[derive(Clone, Copy, Debug)]
enum WaitedRegime {
    BrandNew,
    InWindow,
    OutOfWindow,
}

impl WaitedRegime {
    /// The `regime` label the stable-cut counters carry.
    fn label(self) -> &'static str {
        match self {
            Self::BrandNew => "brand_new",
            Self::InWindow => "in_window",
            Self::OutOfWindow => "out_of_window",
        }
    }
}

/// A catch-up that waited for its slot while K commits landed verifies its
/// cut on the FIRST attempt, its content and its ack carry P + K, the K
/// commits are folded in, and none of them is delivered again: the next
/// message is the next commit.
///
/// - [`Ceiling::Shipped`]: the atomic read under the slot reads P + K, one
///   pass, recorded `verified`, and the stream registers there.
/// - [`Ceiling::ForcedLow`] (the fenced path): the atomic read answers
///   `TOO_LARGE`; the catch-up gives its slot back and attaches to the layer's
///   flight, whose one fenced read — taking the slot the catch-up gave back —
///   reads P + K, recorded `verified`, and the stream registers there. One
///   pass: no broken attempt (`live_edit_snapshot_read_total{path="fenced",
///   outcome="fence_broken"}` = 0).
///
/// The test holds the only slot itself, so the subject's wait is certain, and
/// lands the commits once the subject is queued (`catchups_waiting` = 1),
/// which is after its pre-wait header read in every regime.
async fn a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
    regime: WaitedRegime,
    ceiling: Ceiling,
) {
    const K: u64 = 3;
    let fx = CapacityFixture::setup_full(None, 2, 0, 1, PARKED_BUDGET_MS, ceiling).await;
    let maxlen = match regime {
        WaitedRegime::OutOfWindow => 4,
        _ => 4096,
    };
    let svc = build_service(&fx, maxlen);
    let layer = format!("waited-{}-{ceiling:?}", regime.label());
    // The header sequence before the waits (P).
    let (session_id, object_id, first_read) = match regime {
        WaitedRegime::OutOfWindow => {
            let (sid, oid) = out_of_window_layer(&fx, &svc, &layer).await;
            (sid, oid, 6)
        }
        _ => {
            let (sid, oid) = create_layer(&fx, &svc, &layer).await;
            commit(&svc, &sid, &layer, oid, 41).await;
            commit(&svc, &sid, &layer, oid, 42).await;
            (sid, oid, 3)
        }
    };
    let held = hold_catchup_slot(&svc).await;

    let (have_object, have_version) = match regime {
        WaitedRegime::BrandNew => (0, 0),
        _ => (object_id, 1),
    };
    let mut stream = subscribe(
        &svc,
        &session_id,
        &layer,
        have_object,
        have_version,
        "waited",
    )
    .await;
    await_condition(
        || series(&fx.metrics, WAITING) == 1,
        "the subject queues for the only slot",
    )
    .await;
    assert_eq!(
        svc.valkey.stream_mux.snapshot().subscriptions(),
        1,
        "fixture self-check: the catch-up holds a reader reservation through its slot wait"
    );
    let wait_sections: Vec<u64> = (0..K).map(|i| 500 + i).collect();
    for &section_id in &wait_sections {
        commit(&svc, &session_id, &layer, object_id, section_id).await;
    }
    drop(held);

    let last = first_read + K;
    let (content, body) = recv_message(&mut stream, "the catch-up content").await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (object_id, last),
        "[{} / {ceiling:?}] the content must carry the last commit made during the wait",
        regime.label()
    );
    let carried = section_ids(&body);
    for section_id in &wait_sections {
        assert!(
            carried.contains(section_id),
            "the content must fold in the commit made during the wait (section {section_id})"
        );
    }
    recv_ack(&mut stream, object_id, last).await;

    // Exactly once: the next message is the next commit, so none of the wait
    // commits the content folded in comes back after the ack.
    commit(&svc, &session_id, &layer, object_id, 599).await;
    let (next, _) = recv_message(&mut stream, "the first live commit").await;
    assert_eq!(
        next.sequence_num,
        last + 1,
        "a commit made during the wait was delivered again after the ack"
    );

    let label = regime.label();
    let case = format!("[{label} / {ceiling:?}]");
    assert_eq!(stable_cuts(&fx.metrics, "verified", label), 1, "{case}");
    assert_eq!(stable_cuts(&fx.metrics, "unverified", label), 0, "{case}");
    assert_eq!(
        catchup_snapshot_reads(&fx.metrics, "fenced", "fence_broken"),
        0,
        "{case} commits made during the slot wait must not cost the read a second attempt: the \
         fenced read's first attempt is proven, never broken and retried"
    );
    let (atomic, too_large, fenced) = match ceiling {
        Ceiling::Shipped => (1, 0, 0),
        Ceiling::ForcedLow => (0, 1, 1),
    };
    assert_eq!(
        (
            catchup_snapshot_reads(&fx.metrics, "atomic", "ok"),
            catchup_snapshot_reads(&fx.metrics, "atomic", "too_large"),
            catchup_snapshot_reads(&fx.metrics, "fenced", "ok"),
        ),
        (atomic, too_large, fenced),
        "{case} the catch-up's atomic read served its state, or answered TOO_LARGE and the \
         layer's flight served it from one fenced read"
    );
    assert_eq!(reads_in_flight(&fx.metrics), 0, "{case} the permit is back");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_brand_new_catch_up_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::BrandNew,
        Ceiling::Shipped,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_in_window_catch_up_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::InWindow,
        Ceiling::Shipped,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_out_of_window_catch_up_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::OutOfWindow,
        Ceiling::Shipped,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_brand_new_catch_up_on_the_fenced_path_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::BrandNew,
        Ceiling::ForcedLow,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_in_window_catch_up_on_the_fenced_path_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::InWindow,
        Ceiling::ForcedLow,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waited_out_of_window_catch_up_on_the_fenced_path_verifies_on_the_first_attempt() {
    a_waited_catch_up_verifies_its_stable_cut_on_the_first_attempt(
        WaitedRegime::OutOfWindow,
        Ceiling::ForcedLow,
    )
    .await;
}

/// Above the atomic-read limits no catch-up is served an unproven cut: a
/// catch-up whose shared fenced read is broken on every attempt ends
/// `UNAVAILABLE` before any content or ack, never serving an unverified state.
/// (The interim path this replaced acked its re-read pin after a slot wait and
/// served a state its quiescence loop could not prove.)
///
/// The subject (brand new) waits for the only slot while K
/// commits land, then parks at its pre-full-state point, holding its slot (no
/// flight is registered yet), before its atomic attempt. A hammer then moves
/// the header sequence with no commit — which the fence does not hold off — so
/// each fenced read's release finds the header moved and answers `BROKEN`.
/// After `FENCE_READ_ATTEMPTS` (3) broken reads the shared read fails
/// `UNAVAILABLE`: that is the subject's first item, under `reason="error"`,
/// and no stable cut is recorded, verified or not. Its flight leaves nothing
/// held.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fenced_catch_up_whose_every_read_is_broken_ends_unavailable_before_content_never_serving_an_unproven_cut(
) {
    use fred::interfaces::HashesInterface;
    use live_edit::observability::metrics::STREAM_EXIT_ERROR;
    use std::sync::atomic::{AtomicBool, Ordering};

    const K: u64 = 3;
    let fx = CapacityFixture::setup_above_ceiling(1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let layer = "broken-after-wait";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 51).await;
    let held = hold_catchup_slot(&svc).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "broken").await;
    await_condition(
        || series(&fx.metrics, WAITING) == 1,
        "the subject queues for the only slot",
    )
    .await;
    for section_id in 520..520 + K {
        commit(&svc, &session_id, layer, object_id, section_id).await;
    }
    drop(held);
    tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("the subject parks under its slot, before its atomic attempt");

    let running = Arc::new(AtomicBool::new(true));
    let hammer = {
        let running = Arc::clone(&running);
        let pool = fx.client.pool.clone();
        let header_key = fx.key_schema.obj_header(&object_id.to_string());
        tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                let _: i64 = pool
                    .hincrby(&header_key, "seq", 1)
                    .await
                    .expect("the header-sequence hammer must succeed");
            }
        })
    };
    tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
        .await
        .expect("the parked subject must accept its release");
    let status = first_status(&mut stream, "the broken reads' ending").await;
    running.store(false, Ordering::Relaxed);
    tokio::time::timeout(SETTLE, hammer)
        .await
        .expect("the hammer stops")
        .expect("the hammer must not panic");

    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("broken 3 times in a row"),
        "the shared read's own failure, never an unverified state: {status:?}"
    );
    assert_eq!(
        (
            catchup_snapshot_reads(&fx.metrics, "atomic", "too_large"),
            catchup_snapshot_reads(&fx.metrics, "fenced", "fence_broken"),
            catchup_snapshot_reads(&fx.metrics, "fenced", "ok"),
        ),
        (1, 3, 0),
        "fixture self-check: the forced-low ceiling sends the catch-up to its flight, whose \
         three fenced reads the moving header broke"
    );
    assert_eq!(
        (
            stable_cuts(&fx.metrics, "verified", "brand_new"),
            stable_cuts(&fx.metrics, "unverified", "brand_new"),
        ),
        (0, 0),
        "no cut was served, so none is recorded"
    );
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_ERROR) == 1,
        "the ending is counted under error",
    )
    .await;
    assert_flights_idle(&svc, "after the broken reads").await;
}

// ═══════════════════════════════════════════════════════════════════════════
// The slot covers reads and the build only
// ═══════════════════════════════════════════════════════════════════════════

/// At a limit of one, a re-bootstrap nested inside a catch-up completes: the
/// in-window catch-up that finds its layer overwritten (a `deleted` sentinel
/// in its retained window) releases its slot before the re-bootstrap takes
/// one. Entered holding it, the re-bootstrap would wait on itself and be
/// refused `catchup_capacity` when the 300 ms budget ran out.
///
/// The XRANGE seam parks the reconnect after its header read, holding the
/// slot, while the test overwrites the layer.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_nested_overwrite_rebootstrap_completes_at_a_limit_of_one() {
    let fx = CapacityFixture::setup(1, SHORT_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&seam));
    let layer = "nested-rebootstrap";
    let (session_id, old_object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, old_object_id, 61).await;
    commit(&svc, &session_id, layer, old_object_id, 62).await;

    let mut stream = subscribe(&svc, &session_id, layer, old_object_id, 1, "nested").await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked())
        .await
        .expect("the reconnect parks before its XRANGE, holding the only slot");
    let new_object_id = overwrite(&svc, &session_id, layer).await;
    tokio::time::timeout(SETTLE, seam.release())
        .await
        .expect("release the reconnect");

    let (content, _) = recv_message(&mut stream, "the re-bootstrap's full state").await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (new_object_id, 1),
        "the nested re-bootstrap must serve the NEW object's full state first"
    );
    recv_ack(&mut stream, new_object_id, 1).await;
    assert_eq!(
        exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY),
        0,
        "the nested re-bootstrap was refused its slot"
    );
    assert_eq!(
        catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
        2,
        "one slot for the reconnect's catch-up, one for its re-bootstrap"
    );
    await_condition(|| series(&fx.metrics, INFLIGHT) == 0, "both slots are back").await;
}

/// The one capacity refusal that may come after content: a caught-up
/// subscription whose layer is overwritten re-bootstraps in steady state, and
/// when that re-bootstrap's slot wait runs out the budget, its stream shows
/// the earlier ack and THEN `UNAVAILABLE catchup_capacity`, counted once. Its
/// reconnect — carrying the OLD object's position, with the slot still held —
/// is refused before content: the status is its first item.
///
/// The re-bootstrap starts a fresh budget: the charges the service recorded
/// show its reservation starting from the full budget, not from what the
/// first open's admission and reservation left ([`assert_budget_chain`]'s
/// refill clause), and its slot wait running all of it out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_steady_state_overwrite_refused_its_slot_ends_after_the_ack_and_its_reconnect_is_refused_first(
) {
    let fx = CapacityFixture::setup(1, SHORT_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let (session_id, old_object_id) = create_layer(&fx, &svc, "overwritten").await;
    let (holder_session, _) = create_layer(&fx, &svc, "slot-holder").await;

    // Up to date (no catch-up slot, and no pass through the seam).
    let mut subject = subscribe(
        &svc,
        &session_id,
        "overwritten",
        old_object_id,
        1,
        "subject",
    )
    .await;
    recv_ack(&mut subject, old_object_id, 1).await;

    let mut holder = subscribe(&svc, &holder_session, "slot-holder", 0, 0, "holder").await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("another catch-up parks holding the only slot");

    overwrite(&svc, &session_id, "overwritten").await;
    let status = match next_item(&mut subject, "the steady-state re-bootstrap's ending").await {
        Err(status) => status,
        Ok(msg) => panic!("the refused re-bootstrap must send nothing; got {msg:?}"),
    };
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_CAPACITY_MSG);
    crate::common::refusal_detail::assert_catchup_slot_refusal(&status);
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the steady-state refusal is counted once",
    )
    .await;

    // Read before the reconnect, which reuses the `subject` client id. Waits:
    // the first open's admission and reservation (up to date, so no slot),
    // then the re-bootstrap's reservation on the new object and its slot.
    let charges = assert_budget_chain(
        &budget_seam,
        "subject",
        SHORT_BUDGET_MS,
        &[
            BudgetWait::Admission,
            BudgetWait::StreamReader,
            BudgetWait::StreamReader,
            BudgetWait::CatchUpSlot,
        ],
        &[2],
    );
    assert!(
        charges[0].charged + charges[1].charged > Duration::ZERO,
        "precondition: the first open spent some budget, so a missing refill would leave the \
         re-bootstrap less than the full budget: {charges:#?}"
    );
    let refused = &charges[3];
    assert!(
        refused.charged >= refused.budget_before && refused.budget_after == Duration::ZERO,
        "the refused slot wait ran out everything the refilled budget had left: {refused:?}"
    );

    let mut reconnect = subscribe(
        &svc,
        &session_id,
        "overwritten",
        old_object_id,
        1,
        "subject",
    )
    .await;
    let status = first_status(&mut reconnect, "the reconnect's capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_CAPACITY_MSG);
    crate::common::refusal_detail::assert_catchup_slot_refusal(&status);
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 2,
        "the reconnect's refusal is counted",
    )
    .await;
    assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT), 2);

    tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
        .await
        .expect("release the holder");
    let (_, ack) = recv_catch_up(&mut holder).await;
    assert_eq!(ack.sequence_num, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Reserved through the waits: a waiting catch-up queues nothing
// ═══════════════════════════════════════════════════════════════════════════

/// What a catch-up must serve, for [`assert_serves_every_commit_once`].
#[cfg(feature = "test-fault-injection")]
struct Served<'a> {
    svc: &'a LiveEditService,
    session_id: &'a str,
    layer: &'a str,
    object_id: u64,
    /// The version its content and ack must carry.
    last: u64,
    /// Sections its content must fold in.
    folded: &'a [u64],
    /// Sections of the live commits the check makes start after this one.
    next_sections: u64,
}

/// Under one [`DRAIN`] deadline: the catch-up's content, carrying
/// `(object_id, last)` and folding in every section of `folded`; its ack at
/// `last`; then, for each of three commits made after the ack, the next live
/// message — `last + 1`, `+ 2`, `+ 3`: each commit exactly once, in order, and
/// nothing the content folded in delivered again.
#[cfg(feature = "test-fault-injection")]
async fn assert_serves_every_commit_once(served: Served<'_>, stream: &mut SubStream) {
    let Served {
        svc,
        session_id,
        layer,
        object_id,
        last,
        folded,
        next_sections,
    } = served;
    tokio::time::timeout(DRAIN, async {
        let (content, body) = recv_message(stream, "the catch-up content").await;
        assert_eq!(
            (content.object_id, content.sequence_num),
            (object_id, last),
            "the content carries the last commit made before the catch-up read its state"
        );
        let carried = section_ids(&body);
        for section_id in folded {
            assert!(
                carried.contains(section_id),
                "the content folds in section {section_id}"
            );
        }
        recv_ack(stream, object_id, last).await;
        for n in 1..=3 {
            commit(svc, session_id, layer, object_id, next_sections + n).await;
            let (next, _) = recv_message(stream, "a live commit").await;
            assert_eq!(
                next.sequence_num,
                last + n,
                "every later commit arrives exactly once, in order"
            );
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up and three live commits took over {DRAIN:?}"));
}

/// Which catch-up a capacity-waiting case drives: every regime that takes the
/// atomic read.
#[derive(Clone, Copy, Debug)]
enum WaitingCatchUp {
    BrandNew,
    InWindow,
    OutOfWindow,
}

impl WaitingCatchUp {
    /// The service's `layer_stream_maxlen`: small for the out-of-window layer
    /// ([`out_of_window_layer`]), roomy otherwise.
    fn stream_maxlen(self) -> u64 {
        match self {
            Self::OutOfWindow => 4,
            _ => 4096,
        }
    }
}

/// A seam that parks a [`WaitingCatchUp`] under its slot, holding its reader
/// reservation, before its read permit and atomic read.
#[cfg(feature = "test-fault-injection")]
enum PrePermitPark {
    State(Arc<live_edit::grpc::subscribe::CatchUpStateReadSeam>),
    OutOfWindow(Arc<live_edit::grpc::subscribe::CatchUpOutOfWindowSeam>),
}

#[cfg(feature = "test-fault-injection")]
impl PrePermitPark {
    fn install(svc: &mut LiveEditService, case: WaitingCatchUp) -> Self {
        use live_edit::grpc::subscribe::{CatchUpOutOfWindowSeam, CatchUpStateReadSeam};
        match case {
            WaitingCatchUp::BrandNew => {
                let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
                svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
                Self::State(seam)
            }
            WaitingCatchUp::InWindow => {
                let seam = Arc::new(CatchUpStateReadSeam::new_before_coalesced());
                svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
                Self::State(seam)
            }
            WaitingCatchUp::OutOfWindow => {
                let seam = Arc::new(CatchUpOutOfWindowSeam::new_before_compute_diff());
                svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
                Self::OutOfWindow(seam)
            }
        }
    }

    async fn wait_until_parked(&self, case: WaitingCatchUp) {
        let parked = async {
            match (self, case) {
                (Self::State(seam), WaitingCatchUp::InWindow) => {
                    seam.wait_until_parked_pre_coalesced().await
                }
                (Self::State(seam), _) => seam.wait_until_parked_pre_full_state().await,
                (Self::OutOfWindow(seam), _) => seam.wait_until_parked_pre_compute_diff().await,
            }
        };
        tokio::time::timeout(SETTLE, parked)
            .await
            .expect("the catch-up takes its slot and parks before its read permit");
    }

    async fn release(&self, case: WaitingCatchUp) {
        let release = async {
            match (self, case) {
                (Self::State(seam), WaitingCatchUp::InWindow) => {
                    seam.arrive_at_pre_coalesced().await
                }
                (Self::State(seam), _) => seam.arrive_at_pre_full_state().await,
                (Self::OutOfWindow(seam), _) => seam.arrive_at_pre_compute_diff().await,
            }
        };
        tokio::time::timeout(SETTLE, release)
            .await
            .expect("the parked catch-up accepts its release");
    }
}

/// A layer and the subscriber position for `case`: brand new and in window on
/// a layer at sequence 3 (`have_version` 1 for the reconnect), out of window
/// on [`out_of_window_layer`] (sequence 6). Returns
/// `(session_id, object_id, have_object, have_version, P)`.
async fn waiting_catch_up_layer(
    fx: &CapacityFixture,
    svc: &LiveEditService,
    layer: &str,
    case: WaitingCatchUp,
) -> (String, u64, u64, u64, u64) {
    match case {
        WaitingCatchUp::OutOfWindow => {
            let (sid, oid) = out_of_window_layer(fx, svc, layer).await;
            (sid, oid, oid, 1, 6)
        }
        _ => {
            let (sid, oid) = create_layer(fx, svc, layer).await;
            commit(svc, &sid, layer, oid, 81).await;
            commit(svc, &sid, layer, oid, 82).await;
            let (have_object, have_version) = match case {
                WaitingCatchUp::BrandNew => (0, 0),
                _ => (oid, 1),
            };
            (sid, oid, have_object, have_version, 3)
        }
    }
}

/// Placement on the atomic path (Spec SDD SR-3.6, SR-7.4): a catch-up holds
/// only a reader reservation while it waits for its slot and then for its read
/// permit, so the commits that land meanwhile — more than its backlog hard cap
/// — queue nothing and cannot end it. It registers its stream at its atomic
/// read's cut, which folds them all in, and serves the lot: its content
/// carries the last commit made during the waits and folds in every one, its
/// ack carries the same, and every later commit arrives exactly once, never one
/// of the folded commits again. Nothing overflows and nothing ends
/// `inner_watchdog`.
///
/// Registered at the header sequence it read before its waits (P) instead, its
/// stream would start below every commit made during the waits, queue all of
/// them past the shrunk hard cap and be killed — so this is the test that fails
/// if registration happens at P instead of at the cut: the registration cursor
/// is the acked cut.
///
/// The test holds the only slot while the first half of the commits land, then
/// takes the only read permit and frees the slot; the subject parks under its
/// slot before its permit (still only reserved), and, released, waits for the
/// permit the test holds while the second half land.
#[cfg(feature = "test-fault-injection")]
async fn a_catch_up_waiting_for_its_slot_and_permit_queues_nothing_and_serves_every_commit_once(
    case: WaitingCatchUp,
) {
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;

    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, case.stream_maxlen());
    let park = PrePermitPark::install(&mut svc, case);
    install_tiny_hard_cap(&svc);
    let layer = "waiting-slot-and-permit";
    let (session_id, object_id, have_object, have_version, first_read) =
        waiting_catch_up_layer(&fx, &svc, layer, case).await;
    let held_slot = hold_catchup_slot(&svc).await;

    let mut stream = subscribe(
        &svc,
        &session_id,
        layer,
        have_object,
        have_version,
        "waiting",
    )
    .await;
    await_condition(
        || {
            series(&fx.metrics, WAITING) == 1
                && svc.valkey.stream_mux.snapshot().subscriptions() == 1
        },
        "the subject reserves its reader and queues for the only slot",
    )
    .await;
    assert_eq!(
        series(&fx.metrics, STREAMS),
        0,
        "{case:?}: a slot wait registers no stream"
    );
    let sections: Vec<u64> = (0..OVERFLOW_COMMITS).map(|i| 800 + i).collect();
    let (during_slot_wait, during_permit_wait) = sections.split_at(sections.len() / 2);
    for &section_id in during_slot_wait {
        commit(&svc, &session_id, layer, object_id, section_id).await;
    }

    let held_permit = hold_read_permit(&svc).await;
    drop(held_slot);
    park.wait_until_parked(case).await;
    assert_eq!(
        (
            svc.valkey.stream_mux.snapshot().subscriptions(),
            series(&fx.metrics, STREAMS),
        ),
        (1, 0),
        "{case:?}: under its slot, before its read, the subject holds a reservation and no queue"
    );
    park.release(case).await;
    for &section_id in during_permit_wait {
        commit(&svc, &session_id, layer, object_id, section_id).await;
    }
    assert_eq!(
        (
            series(&fx.metrics, STREAMS),
            overflow_kills(&fx.metrics, "entries")
        ),
        (0, 0),
        "{case:?}: waiting for its read permit, the subject still queues nothing"
    );
    drop(held_permit);

    assert_serves_every_commit_once(
        Served {
            svc: &svc,
            session_id: &session_id,
            layer,
            object_id,
            last: first_read + OVERFLOW_COMMITS,
            folded: &sections,
            next_sections: 900,
        },
        &mut stream,
    )
    .await;
    assert_eq!(overflow_kills(&fx.metrics, "entries"), 0, "{case:?}");
    assert_eq!(overflow_kills(&fx.metrics, "bytes"), 0, "{case:?}");
    assert_eq!(
        exits(&fx.metrics, STREAM_EXIT_INNER_WATCHDOG),
        0,
        "{case:?}"
    );
    assert!(read_gate_is_full(&svc), "{case:?}: the read permit is back");
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_catch_up_waiting_for_its_slot_and_permit_queues_nothing() {
    a_catch_up_waiting_for_its_slot_and_permit_queues_nothing_and_serves_every_commit_once(
        WaitingCatchUp::BrandNew,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_catch_up_waiting_for_its_slot_and_permit_queues_nothing() {
    a_catch_up_waiting_for_its_slot_and_permit_queues_nothing_and_serves_every_commit_once(
        WaitingCatchUp::InWindow,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_catch_up_waiting_for_its_slot_and_permit_queues_nothing() {
    a_catch_up_waiting_for_its_slot_and_permit_queues_nothing_and_serves_every_commit_once(
        WaitingCatchUp::OutOfWindow,
    )
    .await;
}

/// What ends a catch-up while it holds only a reader reservation, in
/// [`an_ending_while_reserved_releases_the_reader`].
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum ReservedEnding {
    /// Graceful shutdown during the slot wait.
    Shutdown,
    /// `DeleteSession` during the slot wait.
    SessionDeletion,
    /// The client leaves during the slot wait.
    ClientDisconnect,
    /// The capacity budget runs out during the slot wait.
    CatchUpCapacity,
    /// An in-window reconnect's point read finds a `deleted`
    /// sentinel as the stream's last entry, and the reconnect re-bootstraps
    /// onto the layer's new object.
    DeletedScan,
    /// A brand-new catch-up's atomic read answers `NotFound` (an overwrite's
    /// cleanup landed during its slot wait) and it re-bootstraps onto the
    /// layer's new object.
    NotFoundAtTheAtomicRead,
}

/// Every ending of a catch-up that holds only a reader reservation releases
/// the reader, and leaves the read gate full — the dropped catch-up held no
/// permit, or returned the one its `NotFound` read took: the multiplexer's
/// subscription count returns to zero — or, for
/// a re-bootstrap, to the one the new object's catch-up holds — and on a
/// cluster the reader's hash slot unbinds (or rebinds to the new object's).
/// Held on past an ending, the reservation would keep the only cluster reader
/// bound, and a re-bootstrap onto an object in another slot would wait for a
/// reader until its budget ran out.
///
/// `fx` is a standalone or a one-reader cluster fixture with one catch-up
/// slot; `CatchUpCapacity` needs a `SHORT_BUDGET_MS` budget, every other case
/// `PARKED_BUDGET_MS`.
#[cfg(feature = "test-fault-injection")]
async fn an_ending_while_reserved_releases_the_reader(fx: CapacityFixture, ending: ReservedEnding) {
    use live_edit::observability::metrics::{
        STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_SESSION_CANCEL, STREAM_EXIT_SHUTDOWN,
    };
    use live_edit::storage::stream_mux::router::Routing;

    let deleted_scan = matches!(ending, ReservedEnding::DeletedScan);
    let mut svc = build_service(&fx, 4096);
    let xrange = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    if deleted_scan {
        svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&xrange));
    }
    let layer = "reserved-ending";
    let (session_id, old_object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, old_object_id, 161).await;
    commit(&svc, &session_id, layer, old_object_id, 162).await;
    let slot_of = |oid: u64| fred::util::redis_keyslot(fx.key_schema.layer_stream(oid).as_bytes());
    let cluster = fx.client.stream_mux.snapshot().routing == Routing::SlotAffinity;

    // The subject reserves its reader, then waits holding the reservation:
    // for the slot the test holds, or — the `Deleted` scan — parked before its
    // XRANGE, holding the slot itself.
    let mut held = None;
    let mut stream = if deleted_scan {
        let stream = subscribe(&svc, &session_id, layer, old_object_id, 1, "reserved").await;
        tokio::time::timeout(SETTLE, xrange.wait_until_parked())
            .await
            .expect("the reconnect parks before its XRANGE, reserved and holding the slot");
        stream
    } else {
        held = Some(hold_catchup_slot(&svc).await);
        subscribe(&svc, &session_id, layer, 0, 0, "reserved").await
    };
    // The budget refusal ends the wait on its own within 300 ms, so only the
    // waits the test ends itself are inspected mid-wait.
    if !matches!(ending, ReservedEnding::CatchUpCapacity) {
        if !deleted_scan {
            await_condition(
                || series(&fx.metrics, WAITING) == 1,
                "the subject reserved its reader and queued for the only slot",
            )
            .await;
        }
        let snap = fx.client.stream_mux.snapshot();
        assert_eq!(
            snap.subscriptions(),
            1,
            "{ending:?}: the subject holds its reservation"
        );
        if cluster {
            assert_eq!(
                snap.readers[0].slot,
                Some(slot_of(old_object_id)),
                "{ending:?}: the reservation binds the only reader to the layer's slot"
            );
        }
        assert_eq!(
            series(&fx.metrics, STREAMS),
            0,
            "{ending:?}: and registers no stream"
        );
    }

    // End it while it is reserved. A re-bootstrap returns the object it moved
    // to, whose catch-up then holds the one remaining subscription.
    let new_object_id = match ending {
        ReservedEnding::Shutdown => {
            svc.shutdown.cancel();
            let status = first_status(&mut stream, "the shutdown ending").await;
            assert_eq!(status.code(), Code::Unavailable, "{status:?}");
            assert_eq!(status.message(), "server shutting down");
            await_condition(
                || exits(&fx.metrics, STREAM_EXIT_SHUTDOWN) == 1,
                "the ending is counted under shutdown",
            )
            .await;
            None
        }
        ReservedEnding::SessionDeletion => {
            svc.session_tokens
                .get(&session_id)
                .expect("the subscribe registered its session token")
                .cancel_deleted();
            let status = first_status(&mut stream, "the deletion ending").await;
            assert_eq!(status.code(), Code::NotFound, "{status:?}");
            assert_eq!(status.message(), "session was deleted");
            await_condition(
                || exits(&fx.metrics, STREAM_EXIT_SESSION_CANCEL) == 1,
                "the ending is counted under session_cancel",
            )
            .await;
            None
        }
        ReservedEnding::ClientDisconnect => {
            drop(stream);
            await_condition(
                || exits(&fx.metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
                "the ending is counted under client_disconnect",
            )
            .await;
            None
        }
        ReservedEnding::CatchUpCapacity => {
            let status = first_status(&mut stream, "the catch-up capacity refusal").await;
            assert_eq!(status.code(), Code::Unavailable, "{status:?}");
            assert_eq!(status.message(), CATCHUP_CAPACITY_MSG);
            crate::common::refusal_detail::assert_catchup_slot_refusal(&status);
            await_condition(
                || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
                "the refusal is counted under catchup_capacity",
            )
            .await;
            None
        }
        ReservedEnding::DeletedScan => {
            let new_object_id = overwrite(&svc, &session_id, layer).await;
            tokio::time::timeout(SETTLE, xrange.release())
                .await
                .expect("release the reconnect");
            let (content, _) = recv_message(&mut stream, "the re-bootstrap's full state").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (new_object_id, 1)
            );
            recv_ack(&mut stream, new_object_id, 1).await;
            Some(new_object_id)
        }
        ReservedEnding::NotFoundAtTheAtomicRead => {
            let new_object_id = overwrite(&svc, &session_id, layer).await;
            svc.layer_store.wait_for_cleanup(SETTLE).await;
            drop(held.take());
            let (content, _) = recv_message(&mut stream, "the re-bootstrap's full state").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (new_object_id, 1)
            );
            recv_ack(&mut stream, new_object_id, 1).await;
            Some(new_object_id)
        }
    };

    match new_object_id {
        None => {
            await_condition(
                || fx.client.stream_mux.snapshot().subscriptions() == 0,
                "the ending released the reader reservation",
            )
            .await;
            if cluster {
                assert_eq!(
                    fx.client.stream_mux.snapshot().readers[0].slot,
                    None,
                    "{ending:?}: the released reservation unbound the reader's slot"
                );
            }
        }
        Some(new_object_id) => {
            let snap = fx.client.stream_mux.snapshot();
            assert_eq!(
                snap.subscriptions(),
                1,
                "{ending:?}: only the new object's catch-up holds a reader; the old object's \
                 reservation was released before the re-bootstrap"
            );
            if cluster {
                assert_eq!(
                    snap.readers[0].slot,
                    Some(slot_of(new_object_id)),
                    "{ending:?}: the reader is bound to the new object's slot"
                );
                if slot_of(new_object_id) == slot_of(old_object_id) {
                    eprintln!(
                        "note: objects {old_object_id} and {new_object_id} share a hash slot, so \
                         this run did not need the old reservation released to rebind"
                    );
                }
            }
        }
    }
    await_condition(
        || read_gate_is_full(&svc) && reads_in_flight(&fx.metrics) == 0,
        "the ended (or re-bootstrapped) catch-up holds no read permit",
    )
    .await;
    drop(held);
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_ended_by_shutdown_releases_its_reader() {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::Shutdown).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_ended_by_session_deletion_releases_its_reader() {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::SessionDeletion).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_ended_by_client_disconnect_releases_its_reader() {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::ClientDisconnect).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_refused_catchup_capacity_releases_its_reader() {
    let fx = CapacityFixture::setup(1, SHORT_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::CatchUpCapacity).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_ended_by_its_deleted_scan_releases_its_reader_before_the_rebootstrap()
{
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::DeletedScan).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reserved_catch_up_whose_atomic_read_is_not_found_releases_its_reader_before_the_rebootstrap(
) {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    an_ending_while_reserved_releases_the_reader(fx, ReservedEnding::NotFoundAtTheAtomicRead).await;
}

/// A catch-up parked while reserved — at the out-of-window pre-header-read
/// point, holding its slot and a reader reservation — whose multiplexer shuts
/// down meanwhile cannot register its stream: its first item is `UNAVAILABLE
/// "server shutting down"` under `shutdown`, with no content and no ack, and
/// the refused registration releases the reader. The service itself is not
/// shutting down: the ending comes from the registration alone.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_registration_refused_by_a_multiplexer_that_shut_down_while_reserved_ends_shutdown_before_content(
) {
    use live_edit::observability::metrics::STREAM_EXIT_SHUTDOWN;

    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4);
    let seam =
        Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_header_read());
    svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
    let layer = "mux-shut-while-reserved";
    let (session_id, object_id) = out_of_window_layer(&fx, &svc, layer).await;

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1, "reserved").await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_header_read())
        .await
        .expect("the out-of-window catch-up parks before its header read, reserved");
    assert_eq!(svc.valkey.stream_mux.snapshot().subscriptions(), 1);
    assert_eq!(series(&fx.metrics, STREAMS), 0, "reserved, not registered");

    svc.valkey.stream_mux.shutdown(Duration::from_secs(1)).await;
    tokio::time::timeout(SETTLE, seam.arrive_at_pre_header_read())
        .await
        .expect("the parked catch-up accepts its release");

    let status = first_status(&mut stream, "the refused registration's ending").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), "server shutting down");
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_SHUTDOWN) == 1,
        "the ending is counted under shutdown",
    )
    .await;
    assert!(
        !svc.shutdown.is_cancelled(),
        "fixture self-check: only the multiplexer shut down"
    );
    await_condition(
        || svc.valkey.stream_mux.snapshot().subscriptions() == 0,
        "the refused registration released the reader",
    )
    .await;
    await_condition(
        || series(&fx.metrics, ADMITTED) == 0 && series(&fx.metrics, INFLIGHT) == 0,
        "the ended catch-up returns its admission and its slot",
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════
// The ending check: a registration that ended ends the catch-up before content
// ═══════════════════════════════════════════════════════════════════════════

/// Which catch-up a registration-killed-while-held case holds, and where: each
/// inside its build from the atomic read, under its slot, with its stream
/// registered, before the check that follows its slot release.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum KilledWhileHeld {
    /// Brand new.
    BrandNew,
    /// Out-of-window reconnect.
    OutOfWindow,
    /// In-window reconnect.
    InWindow,
}

/// A catch-up whose stream registration is killed by the backlog hard cap
/// while it builds — registered under its slot, before its release — ends
/// with that ending once its slot is released, before its first message: the
/// first item is the `UNAVAILABLE`, with no content and no ack. Without the
/// post-release check it would send its content and ack first, and only its
/// steady state would see the ending.
///
/// Each case is held inside its build by [`ParkingExtStore`]: the layer
/// carries an external value the build resolves, after the registration at
/// the cut.
#[cfg(feature = "test-fault-injection")]
async fn a_registration_killed_while_held_ends_the_catch_up_before_content(case: KilledWhileHeld) {
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;

    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let maxlen = match case {
        KilledWhileHeld::OutOfWindow => 4,
        _ => 4096,
    };
    let mut svc = build_service(&fx, maxlen);
    let ext = install_parking_ext_store(&mut svc);
    install_tiny_hard_cap(&svc);

    let layer = "killed-held";
    let (session_id, object_id) = match case {
        KilledWhileHeld::OutOfWindow => {
            let (sid, oid) = out_of_window_layer(&fx, &svc, layer).await;
            commit_with_external_value(&svc, &sid, layer, oid, 93).await;
            (sid, oid)
        }
        _ => {
            let (sid, oid) = create_layer(&fx, &svc, layer).await;
            commit_with_external_value(&svc, &sid, layer, oid, 91).await;
            commit(&svc, &sid, layer, oid, 92).await;
            (sid, oid)
        }
    };
    let (have_object, have_version) = match case {
        KilledWhileHeld::BrandNew => (0, 0),
        _ => (object_id, 1),
    };
    ext.arm();
    let mut stream = subscribe(
        &svc,
        &session_id,
        layer,
        have_object,
        have_version,
        "killed",
    )
    .await;
    ext.wait_until_parked(1).await;
    assert_eq!(
        (series(&fx.metrics, STREAMS), series(&fx.metrics, INFLIGHT)),
        (1, 1),
        "{case:?}: held under its slot, with its stream registered"
    );

    for section_id in 0..OVERFLOW_COMMITS {
        commit(&svc, &session_id, layer, object_id, 900 + section_id).await;
    }
    await_condition(
        || overflow_kills(&fx.metrics, "entries") == 1,
        "the held catch-up's backlog hits the hard cap",
    )
    .await;

    ext.disarm();
    ext.release_one();

    let status = first_status(&mut stream, "the registration's ending").await;
    assert_eq!(status.code(), Code::Unavailable, "{case:?}: {status:?}");
    assert!(
        status
            .message()
            .contains("fell behind its delivery watchdog"),
        "{case:?}: got {status:?}"
    );
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_INNER_WATCHDOG) == 1,
        "the ending is counted under inner_watchdog",
    )
    .await;
    await_condition(
        || series(&fx.metrics, ADMITTED) == 0,
        "the ended catch-up returns its admission",
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_registration_killed_during_the_build_ends_before_content() {
    a_registration_killed_while_held_ends_the_catch_up_before_content(KilledWhileHeld::BrandNew)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_registration_killed_during_the_build_ends_before_content() {
    a_registration_killed_while_held_ends_the_catch_up_before_content(KilledWhileHeld::OutOfWindow)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_registration_killed_during_the_build_ends_before_content() {
    a_registration_killed_while_held_ends_the_catch_up_before_content(KilledWhileHeld::InWindow)
        .await;
}

/// A catch-up waiting for the catch-up payload budget (`D_c`) — under its
/// slot, with its stream registered — whose registration is killed by the
/// backlog hard cap meanwhile ends with that ending, before its first
/// message, as soon as the wait's periodic stream check sees it: not with
/// `catchup_capacity` when its capacity budget runs out, which
/// [`PARKED_BUDGET_MS`] puts out of reach. It leaves the budget's queue and
/// reserves nothing.
///
/// Fails without the wait's stream check (in-place perturbation): the
/// catch-up waits on for `D_c`, and no first item arrives within `SETTLE`.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_registration_killed_while_waiting_for_payload_memory_ends_the_catch_up_before_content() {
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;

    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    install_tiny_hard_cap(&svc);
    let layer = "killed-waiting-for-payload";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 91).await;
    let budget = Arc::clone(&svc.valkey.catchup_payload_budget);
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("the whole payload budget is free");

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "killed-waiting").await;
    await_condition(
        || budget.waiters() == 1,
        "the catch-up waits for payload memory",
    )
    .await;
    assert_eq!(
        (series(&fx.metrics, STREAMS), series(&fx.metrics, INFLIGHT)),
        (1, 1),
        "it waits under its slot, with its stream registered"
    );

    for section_id in 0..OVERFLOW_COMMITS {
        commit(&svc, &session_id, layer, object_id, 900 + section_id).await;
    }
    await_condition(
        || overflow_kills(&fx.metrics, "entries") == 1,
        "the waiting catch-up's backlog hits the hard cap",
    )
    .await;

    let status = first_status(&mut stream, "the registration's ending").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status
            .message()
            .contains("fell behind its delivery watchdog"),
        "the registration's ending, not the capacity refusal: got {status:?}"
    );
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_INNER_WATCHDOG) == 1,
        "the ending is counted under inner_watchdog",
    )
    .await;
    assert_eq!(exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY), 0);
    await_condition(
        || {
            budget.waiters() == 0
                && series(&fx.metrics, ADMITTED) == 0
                && series(&fx.metrics, INFLIGHT) == 0
        },
        "the ended catch-up leaves the budget's queue and returns its admission and slot",
    )
    .await;
    assert_eq!(
        budget.reserved_bytes(),
        budget.total_bytes(),
        "it reserved nothing: only the test's grant is held"
    );
    drop(held);
    assert_eq!(budget.reserved_bytes(), 0);
}

// ── Across a `Restored` recovery ─────────────────────────────────────────────

/// Installs an enabled recovery hook on `svc` with a checkpoint of `object_id`
/// (at sequence 13) to restore from.
#[cfg(feature = "test-fault-injection")]
async fn enable_recovery(svc: &mut LiveEditService, object_id: u64, session_id: &str, layer: &str) {
    let store = Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    crate::common::seed_layer_checkpoint(&store, object_id, session_id, layer, 13).await;
    let mut recovery_config = crate::common::test_config();
    recovery_config.checkpoint_enabled = true;
    recovery_config.service_label = SERVICE.to_string();
    svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
        store,
        Arc::clone(&svc.layer_store),
        Arc::new(recovery_config),
        Arc::clone(&svc.metrics),
    ));
}

/// Loses `object_id`'s state but not its stream, so a reconnect stays in
/// window and its next state read answers `NotFound`.
#[cfg(feature = "test-fault-injection")]
async fn lose_state_keep_stream(fx: &CapacityFixture, object_id: u64) {
    use fred::interfaces::KeysInterface;
    let stream_key = fx.key_schema.layer_stream(object_id);
    let state_keys: Vec<fred::types::Key> =
        crate::common::scan_object_keys(&fx.client, &fx.prefix, object_id)
            .await
            .into_iter()
            .filter(|k| k.as_str() != Some(stream_key.as_str()))
            .collect();
    assert!(
        !state_keys.is_empty(),
        "precondition: the object has state to lose"
    );
    let _: () = fx
        .client
        .pool
        .unlink(state_keys)
        .await
        .expect("UNLINK of the object's state must succeed");
}

/// A layer lost under a shared fenced read — its state removed while the read
/// holds its fence, as a keyspace loss leaves it — is recovered by each catch-up
/// attached to the read, on its own (one regime: brand new):
/// the shared `NotFound` reaches each as its own copy, each runs the recovery
/// hook (two attempts, both `restored`: the first restores, the second finds
/// the layer back), and each retries by attaching again, with no slot of its
/// own, to a new flight.
///
/// The new flight waits for the replica's only slot, which the test took as the
/// failed flight gave it back, while more commits than the shrunk backlog hard
/// cap land on the restored layer. A catch-up registers its stream only at its
/// read's cut, after that read, so nothing queues for it meanwhile and nothing
/// ends it: each is served the restored layer at the last of those commits,
/// folding them all in, and every later commit arrives exactly once. (There is
/// no stream registered before a fenced catch-up's state read, so no
/// registration can end across its recovery; the interim path's
/// registered-then-recovered order this replaces is gone.)
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_layer_lost_under_a_shared_fenced_read_is_recovered_by_each_catch_up_which_attaches_again_with_no_slot(
) {
    use crate::storage::snapshot_read_support::HeldCommand;
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };

    let fx = CapacityFixture::setup_above_ceiling(1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    install_tiny_hard_cap(&svc);
    let layer = "lost-under-read";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 141).await;
    commit(&svc, &session_id, layer, object_id, 142).await;
    enable_recovery(&mut svc, object_id, &session_id, layer).await;
    // The first flight's read is held under its fence, its lease
    // lengthened so no late renewal can end the held attempt, until the
    // test has lost the layer's state under it.
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.lengthen_fence_leases(true);
    fx.client
        .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let hold = HeldCommand::new(
        injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_command(
            ObjectSnapshotReader::FencedFanout,
            0,
        )),
    );

    let mut first = subscribe(&svc, &session_id, layer, 0, 0, "lost-first").await;
    hold.parked_within(SETTLE, "the first catch-up's flight's read")
        .await;
    await_condition(
        || {
            svc.snapshot_flights
                .lookup(object_id)
                .is_some_and(|l| l.attachable && l.state == FlightState::Reading)
        },
        "the first catch-up's flight reads under its fence",
    )
    .await;
    let mut second = subscribe(&svc, &session_id, layer, 0, 0, "lost-second").await;
    await_condition(
        || flight_roles(&fx.metrics, "joined_read") == 1,
        "the second catch-up joins the read",
    )
    .await;

    // The test queues for the only slot, held by the reading flight, so it
    // takes it as that flight fails.
    let mut taken = std::pin::pin!(svc
        .valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600)));
    assert!(
        futures::poll!(taken.as_mut()).is_pending(),
        "the test queues behind the reading flight for the slot"
    );
    live_edit::storage::test_support::assert_fence_lease_lengthened(
        &fx.client,
        &fx.key_schema,
        object_id,
        &fx.client,
    )
    .await;
    lose_state_keep_stream(&fx, object_id).await;
    hold.release();
    let permit = tokio::time::timeout(SETTLE, taken)
        .await
        .expect("the failed flight gives its slot back")
        .expect("the test takes the released slot");
    await_condition(
        || {
            crate::common::recovery_metric(&fx.metrics, "restored") == 2
                && flight_roles(&fx.metrics, "leader") == 2
                && flight_roles(&fx.metrics, "joined_read") == 2
                && series(&fx.metrics, WAITING) == 1
        },
        "each catch-up ran its own recovery and attached again, to one new flight waiting for \
         the slot",
    )
    .await;
    assert_eq!(
        series(&fx.metrics, STREAMS),
        0,
        "neither catch-up registered a stream before its read"
    );
    let restored = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the restored header reads")
        .sequence_num;
    let sections: Vec<u64> = (0..OVERFLOW_COMMITS).map(|i| 1_500 + i).collect();
    for &section_id in &sections {
        commit(&svc, &session_id, layer, object_id, section_id).await;
    }
    drop(permit);

    let last = restored + OVERFLOW_COMMITS;
    assert_serves_every_commit_once(
        Served {
            svc: &svc,
            session_id: &session_id,
            layer,
            object_id,
            last,
            folded: &sections,
            next_sections: 1_600,
        },
        &mut first,
    )
    .await;
    // The second catch-up: the same content and ack, then the three commits
    // the first check made, each once.
    tokio::time::timeout(DRAIN, async {
        let (content, body) = recv_message(&mut second, "the second's content").await;
        assert_eq!((content.object_id, content.sequence_num), (object_id, last));
        let carried = section_ids(&body);
        assert!(sections.iter().all(|s| carried.contains(s)));
        recv_ack(&mut second, object_id, last).await;
        for n in 1..=3 {
            let (next, _) = recv_message(&mut second, "a live commit").await;
            assert_eq!(
                next.sequence_num,
                last + n,
                "each later commit once, in order"
            );
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the second catch-up took over {DRAIN:?}"));

    assert_eq!(
        (
            overflow_kills(&fx.metrics, "entries"),
            overflow_kills(&fx.metrics, "bytes"),
            exits(&fx.metrics, STREAM_EXIT_INNER_WATCHDOG),
        ),
        (0, 0, 0),
        "nothing queued for either catch-up before its read, so nothing ended it"
    );
    assert!(
        catchup_snapshot_reads(&fx.metrics, "fenced", "not_found") >= 1,
        "fixture self-check: the first flight's read found the layer gone"
    );
    drop((first, second));
    assert_flights_idle(&svc, "after the recovered catch-ups").await;
}

/// The sibling, on the atomic path: the `NotFound` comes at the atomic read,
/// while the catch-up holds only a reader reservation. More commits than its
/// hard cap land on the restored layer while it waits to re-acquire a slot; it
/// queues none of them, so nothing ends it. It registers at its retried atomic
/// read's cut after the restore and catches up: its content carries the last
/// of those commits, and every later commit arrives exactly once.
///
/// Registered before its waits instead, its stream would have queued those
/// commits past the shrunk hard cap and been killed before content.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_restored_while_reserved_queues_nothing_through_its_reacquire_and_catches_up() {
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;

    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let xrange = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&xrange));
    install_tiny_hard_cap(&svc);
    let layer = "restored-reserved";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 121).await;
    commit(&svc, &session_id, layer, object_id, 122).await;
    enable_recovery(&mut svc, object_id, &session_id, layer).await;

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1, "restored").await;
    tokio::time::timeout(SETTLE, xrange.wait_until_parked())
        .await
        .expect("the reconnect parks before its XRANGE, reserved and holding the only slot");
    lose_state_keep_stream(&fx, object_id).await;

    let mut taken = std::pin::pin!(svc
        .valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600)));
    assert!(
        futures::poll!(taken.as_mut()).is_pending(),
        "the test queues behind the reconnect for its slot"
    );
    tokio::time::timeout(SETTLE, xrange.release())
        .await
        .expect("release the reconnect");
    let permit = tokio::time::timeout(SETTLE, taken)
        .await
        .expect("the reconnect releases its slot for its recovery attempt")
        .expect("the test takes the released slot");
    await_condition(
        || {
            crate::common::recovery_metric(&fx.metrics, "restored") == 1
                && series(&fx.metrics, WAITING) == 1
        },
        "the layer is restored and the reconnect waits to re-acquire a slot",
    )
    .await;
    assert_eq!(
        (
            svc.valkey.stream_mux.snapshot().subscriptions(),
            series(&fx.metrics, STREAMS)
        ),
        (1, 0),
        "the reconnect still holds only its reader reservation"
    );
    let restored = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the restored header reads")
        .sequence_num;
    let sections: Vec<u64> = (0..OVERFLOW_COMMITS).map(|i| 1_300 + i).collect();
    for &section_id in &sections {
        commit(&svc, &session_id, layer, object_id, section_id).await;
    }
    drop(permit);

    // The content carries the last commit made during the re-acquire and folds
    // them all in.
    assert_serves_every_commit_once(
        Served {
            svc: &svc,
            session_id: &session_id,
            layer,
            object_id,
            last: restored + OVERFLOW_COMMITS,
            folded: &sections,
            next_sections: 1_400,
        },
        &mut stream,
    )
    .await;
    assert_eq!(overflow_kills(&fx.metrics, "entries"), 0);
    assert_eq!(overflow_kills(&fx.metrics, "bytes"), 0);
    assert_eq!(exits(&fx.metrics, STREAM_EXIT_INNER_WATCHDOG), 0);
    assert_eq!(
        crate::common::recovery_metric(&fx.metrics, "restored"),
        1,
        "exactly one recovery attempt"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// The budget is charged for waiting
// ═══════════════════════════════════════════════════════════════════════════

/// How long the standalone budget-charge test keeps the subject waiting for
/// admission, to spend budget.
#[cfg(feature = "test-fault-injection")]
const ADMISSION_SPEND: Duration = Duration::from_millis(500);

/// How long the standalone budget-charge test keeps the subject waiting for
/// its first slot, to spend budget.
#[cfg(feature = "test-fault-injection")]
const FIRST_SLOT_SPEND: Duration = Duration::from_millis(500);

/// Every charge `seam` recorded for `client_id`, in charge order, checked
/// against the budget arithmetic every charge site must keep, and returned.
/// The subscription's waits must be `waits`, against a budget of `total_ms`
/// that was refilled (an overwrite re-bootstrap) just before each wait whose
/// index is in `refilled_before`:
///
/// 1. Each wait was charged what it waited:
///    `budget_before - charged == budget_after` (saturating at zero).
/// 2. Each wait started from what the one before it left, or, after a refill,
///    from the full budget.
/// 3. The first wait is the admission wait, from the full budget, with no
///    deadline (the budget did not exist before it).
/// 4. So each wait started from the full budget less every charge since the
///    last refill (or since admission).
/// 5. Each later wait was handed what was left, not a fresh full wait:
///    `started + budget_before <= deadline <= started + charged +
///    budget_before`. The upper bound tells the two apart when the wait was
///    charged less than the waits before it since the last refill.
///
/// Each record carries the values its charge site itself used, so these hold
/// exactly on correct code, however the scheduler ran. A dropped `charge`
/// call fails 1 (its record has `budget_after == budget_before`) or 2 and 4
/// (the next wait's `budget_before` is read from the real budget), as long as
/// the dropped charge was above zero; a dropped refill fails 4 (the wait after
/// it starts below the full budget), as long as the waits before it were
/// charged above zero.
#[cfg(feature = "test-fault-injection")]
fn assert_budget_chain(
    seam: &CapacityBudgetSeam,
    client_id: &str,
    total_ms: u64,
    waits: &[BudgetWait],
    refilled_before: &[usize],
) -> Vec<BudgetCharge> {
    let total = Duration::from_millis(total_ms);
    let charges = seam.charges_for(client_id);
    assert_eq!(
        charges.iter().map(|c| c.wait).collect::<Vec<_>>(),
        waits,
        "the subscription's capacity waits, in charge order; records: {charges:#?}"
    );
    // 3.
    assert_eq!(charges[0].wait, BudgetWait::Admission);
    assert!(
        charges[0].deadline.is_none(),
        "admission has no budget deadline: {:?}",
        charges[0]
    );
    let mut since_refill = Duration::ZERO;
    for (i, c) in charges.iter().enumerate() {
        // 1.
        assert_eq!(
            c.budget_before.saturating_sub(c.charged),
            c.budget_after,
            "wait {i} must be charged exactly what it waited: {c:?}"
        );
        // 2. and 4.
        let refilled = i == 0 || refilled_before.contains(&i);
        if refilled {
            since_refill = Duration::ZERO;
        } else {
            assert_eq!(
                c.budget_before,
                charges[i - 1].budget_after,
                "wait {i} must start from what wait {} left: {charges:#?}",
                i - 1
            );
        }
        assert_eq!(
            c.budget_before,
            total.saturating_sub(since_refill),
            "wait {i} must start from the {}budget less every charge since: {charges:#?}",
            if refilled { "full (refilled) " } else { "" }
        );
        since_refill += c.charged;
        // 5.
        if let Some(deadline) = c.deadline {
            assert!(
                c.started + c.budget_before <= deadline
                    && deadline <= c.started + c.charged + c.budget_before,
                "wait {i}'s deadline must be when it started plus what was left of the budget: \
                 {c:?}"
            );
        }
    }
    charges
}

/// The admission wait and the slot wait are charged to the subscription's one
/// capacity budget, so a later slot wait — the re-acquire after a `Restored`
/// recovery — gets only what they left: its deadline is the full budget less
/// every earlier charge. Asserted on the charges the service recorded
/// ([`assert_budget_chain`]) rather than by racing a held slot against the
/// re-acquire's deadline.
///
/// The subject (an in-window reconnect) waits for admission
/// while the test holds the only admission, waits for its first slot while
/// the test holds the only slot, parks at its `XRANGE`, finds its state gone
/// at its atomic read, gives its slot up for the recovery attempt, and,
/// restored, waits to re-acquire one. Its charges are, in order: admission,
/// its stream-reader reservation (on standalone Valkey only the reservation's
/// own time), the first slot, the read permit for the read that answered
/// `NotFound`, the re-acquire, and the read permit for the retried read. The
/// `Restored` recovery does not refill the budget, so the chain runs unbroken.
///
/// `ADMISSION_SPEND` and `FIRST_SLOT_SPEND` exist to SPEND budget; they are
/// not synchronisation. Each starts only after the subject is observed
/// waiting (`waiters` = 1, `catchups_waiting` = 1), so each wait is charged
/// at least its spend however the scheduler runs. Every other step is
/// established by polling under `SETTLE`, and the test frees the re-acquire's
/// slot as soon as the subject queues for it. The budget is
/// `PARKED_BUDGET_MS` (60 s), which no wait here comes near, so the subject
/// is never refused, and it catches up on the restored layer.
///
/// The deadline bracket tells "handed what was left" from "handed a fresh
/// full wait" on the reservation and the re-acquire, each normally charged
/// far less than the spends before it. The refusal a run-out budget ends a wait with
/// is pinned by `a_slot_wait_that_outlasts_the_budget_is_refused_before_content`.
/// The cluster stream-reader wait's charge is pinned by
/// `cluster::a_reader_wait_is_charged_to_the_budget_and_the_later_slot_wait_gets_only_what_it_left`:
/// on standalone Valkey that wait never waits.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn waits_are_charged_to_the_one_budget_and_a_later_slot_wait_gets_only_what_they_left() {
    const SUBJECT: &str = "charged";
    let fx = CapacityFixture::setup_with(None, 2, 1, 1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let xrange = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&xrange));
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let layer = "charged";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 131).await;
    commit(&svc, &session_id, layer, object_id, 132).await;
    enable_recovery(&mut svc, object_id, &session_id, layer).await;

    // The test holds the only admission and the only catch-up slot.
    let admission_gate = Arc::clone(&svc.valkey.subscriber_admission);
    let admission = admission_gate
        .admit(None)
        .await
        .expect("the test's own admission must be free");
    let first_slot = hold_catchup_slot(&svc).await;

    // Spend 1: the admission wait.
    let (mut stream, ()) = tokio::join!(
        subscribe(&svc, &session_id, layer, object_id, 1, SUBJECT),
        async move {
            await_condition(
                || admission_gate.waiters() == 1,
                "the subject queues for the only admission",
            )
            .await;
            tokio::time::sleep(ADMISSION_SPEND).await;
            drop(admission);
        },
    );

    // Spend 2: the first slot wait.
    await_condition(
        || series(&fx.metrics, WAITING) == 1,
        "the subject queues for the only slot",
    )
    .await;
    tokio::time::sleep(FIRST_SLOT_SPEND).await;
    drop(first_slot);
    tokio::time::timeout(SETTLE, xrange.wait_until_parked())
        .await
        .expect("the reconnect takes the slot and parks before its XRANGE");

    // The layer's state is lost; its stream survives, so the reconnect is
    // still in window and its read answers NotFound.
    lose_state_keep_stream(&fx, object_id).await;

    let mut taken = std::pin::pin!(svc
        .valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600)));
    assert!(
        futures::poll!(taken.as_mut()).is_pending(),
        "the test queues behind the reconnect for its slot"
    );
    tokio::time::timeout(SETTLE, xrange.release())
        .await
        .expect("release the reconnect");
    let reacquire_slot = tokio::time::timeout(SETTLE, taken)
        .await
        .expect("the reconnect releases its slot for its recovery attempt")
        .expect("the test takes the released slot");
    await_condition(
        || {
            crate::common::recovery_metric(&fx.metrics, "restored") == 1
                && series(&fx.metrics, WAITING) == 1
        },
        "the layer is restored and the reconnect waits to re-acquire a slot",
    )
    .await;

    let restored = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the restored header reads")
        .sequence_num;
    drop(reacquire_slot);

    // The re-acquire takes the freed slot and the subject catches up on the
    // restored layer; its last charge was recorded before it sent anything.
    let (_, ack) = recv_catch_up(&mut stream).await;
    assert_eq!(
        (ack.object_id, ack.sequence_num),
        (object_id, restored),
        "the subject catches up on the restored layer"
    );

    let charges = assert_budget_chain(
        &budget_seam,
        SUBJECT,
        PARKED_BUDGET_MS,
        &[
            BudgetWait::Admission,
            BudgetWait::StreamReader,
            BudgetWait::CatchUpSlot,
            BudgetWait::ReadPermit,
            BudgetWait::CatchUpSlot,
            BudgetWait::ReadPermit,
            // The built payload's reservation of the catch-up payload budget
            // (b.ntm), inside the re-acquired slot, before any content.
            BudgetWait::CatchUpPayload,
        ],
        &[],
    );
    let (admission, first_slot, reacquire) = (&charges[0], &charges[2], &charges[4]);
    assert!(
        admission.charged >= ADMISSION_SPEND,
        "the admission wait, observed in progress before the spend, is charged at least the \
         spend: {admission:?}"
    );
    assert!(
        first_slot.charged >= FIRST_SLOT_SPEND,
        "the first slot wait, observed in progress before the spend, is charged at least the \
         spend: {first_slot:?}"
    );
    assert!(
        reacquire.budget_before
            <= Duration::from_millis(PARKED_BUDGET_MS) - ADMISSION_SPEND - FIRST_SLOT_SPEND,
        "the re-acquire after the `Restored` recovery gets only what the earlier waits left: \
         {charges:#?}"
    );
    assert_eq!(
        crate::common::recovery_metric(&fx.metrics, "restored"),
        1,
        "exactly one recovery attempt"
    );
    assert_eq!(
        catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
        0,
        "no wait came near the budget"
    );
}

/// The capacity budget of
/// `a_reacquire_after_a_restored_recovery_that_outlasts_the_budget_is_refused_before_content`:
/// paid once, by the re-acquire the test never frees, and far more than the
/// promptly-released waits before it take.
#[cfg(feature = "test-fault-injection")]
const REACQUIRE_REFUSAL_BUDGET_MS: u64 = 3_000;

/// The re-acquire after a `Restored` recovery is the one slot wait a catch-up
/// makes after a recovery (`read_state_with_recovery`'s
/// `acquire_catchup_slot`) — here holding only its reader reservation, its
/// atomic read having answered `NotFound`. When it outlasts what is left of
/// the budget, the subscription ends in-stream `UNAVAILABLE`
/// `catchup_capacity` as its first item, counted once with one `timed_out`
/// slot wait. It then returns its admission and holds no slot or read permit,
/// and the re-acquire's charge ran out everything that was left.
///
/// The same scenario as
/// `waits_are_charged_to_the_one_budget_and_a_later_slot_wait_gets_only_what_they_left`,
/// with no deliberate spends: the test frees the admission and the first slot
/// as soon as the subject is seen queued for each, and never frees the slot
/// the re-acquire waits for until the refusal has arrived, so on correct code
/// the refusal is certain.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reacquire_after_a_restored_recovery_that_outlasts_the_budget_is_refused_before_content()
{
    const SUBJECT: &str = "reacquire-refused";
    let fx = CapacityFixture::setup_with(None, 2, 1, 1, REACQUIRE_REFUSAL_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let xrange = Arc::new(live_edit::grpc::subscribe::CatchUpXrangeSeam::new());
    svc.catch_up_xrange_seam_for_test = Some(Arc::clone(&xrange));
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let layer = "reacquire-refused";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 151).await;
    commit(&svc, &session_id, layer, object_id, 152).await;
    enable_recovery(&mut svc, object_id, &session_id, layer).await;

    // The test holds the only admission and the only catch-up slot, and frees
    // each as soon as the subject queues for it.
    let admission_gate = Arc::clone(&svc.valkey.subscriber_admission);
    let admission = admission_gate
        .admit(None)
        .await
        .expect("the test's own admission must be free");
    let first_slot = hold_catchup_slot(&svc).await;
    let (mut stream, ()) = tokio::join!(
        subscribe(&svc, &session_id, layer, object_id, 1, SUBJECT),
        async move {
            await_condition(
                || admission_gate.waiters() == 1,
                "the subject queues for the only admission",
            )
            .await;
            drop(admission);
        },
    );
    await_condition(
        || series(&fx.metrics, WAITING) == 1,
        "the subject queues for the only slot",
    )
    .await;
    drop(first_slot);
    tokio::time::timeout(SETTLE, xrange.wait_until_parked())
        .await
        .expect("the reconnect takes the slot and parks before its XRANGE");

    // Its read answers NotFound; it gives its slot up for the recovery
    // attempt, and the test takes that slot and keeps it.
    lose_state_keep_stream(&fx, object_id).await;
    let mut taken = std::pin::pin!(svc
        .valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600)));
    assert!(
        futures::poll!(taken.as_mut()).is_pending(),
        "the test queues behind the reconnect for its slot"
    );
    tokio::time::timeout(SETTLE, xrange.release())
        .await
        .expect("release the reconnect");
    let reacquire_slot = tokio::time::timeout(SETTLE, taken)
        .await
        .expect("the reconnect releases its slot for its recovery attempt")
        .expect("the test takes the released slot");
    await_condition(
        || {
            crate::common::recovery_metric(&fx.metrics, "restored") == 1
                && series(&fx.metrics, WAITING) == 1
        },
        "the layer is restored and the reconnect waits to re-acquire a slot",
    )
    .await;

    // The slot stays held: the re-acquire can only run out the budget.
    let status = first_status(&mut stream, "the re-acquire's capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_CAPACITY_MSG);
    crate::common::refusal_detail::assert_catchup_slot_refusal(&status);
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted under catchup_capacity",
    )
    .await;
    assert_eq!(
        catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
        1,
        "the re-acquire is the one slot wait that ran out the budget"
    );
    assert_eq!(
        crate::common::recovery_metric(&fx.metrics, "restored"),
        1,
        "exactly one recovery attempt"
    );

    // The read permit is the one the atomic read that answered `NotFound` took.
    let charges = assert_budget_chain(
        &budget_seam,
        SUBJECT,
        REACQUIRE_REFUSAL_BUDGET_MS,
        &[
            BudgetWait::Admission,
            BudgetWait::StreamReader,
            BudgetWait::CatchUpSlot,
            BudgetWait::ReadPermit,
            BudgetWait::CatchUpSlot,
        ],
        &[],
    );
    let reacquire = &charges[4];
    assert!(
        reacquire.charged >= reacquire.budget_before && reacquire.budget_after == Duration::ZERO,
        "the refused re-acquire ran out everything the budget had left: {reacquire:?}"
    );

    drop(reacquire_slot);
    await_condition(
        || {
            series(&fx.metrics, ADMITTED) == 0
                && series(&fx.metrics, INFLIGHT) == 0
                && read_gate_is_full(&svc)
        },
        "the refused catch-up returns its admission and holds no slot or read permit",
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════
// The snapshot read permit: inside the slot, around the atomic read only
// ═══════════════════════════════════════════════════════════════════════════

/// With the replica's only read permit held, a catch-up that has its slot
/// waits for a permit, and when the wait outlasts its capacity budget it ends
/// in-stream `UNAVAILABLE` naming `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`,
/// under `reason="catchup_capacity"`, as the stream's FIRST item — no content,
/// no ack (Spec SDD SR-7.3). Its slot was free (one `acquired` slot wait, none
/// `timed_out`), so the refusal is the permit's; the refused wait is observed
/// on the read-wait histogram; and it gives back its slot, its admission and
/// its reader reservation, while the test's own permit is the only one held.
async fn a_read_permit_wait_that_outlasts_the_budget_is_refused_before_content(
    case: WaitingCatchUp,
) {
    let fx = CapacityFixture::setup(64, SHORT_BUDGET_MS).await;
    let svc = build_service(&fx, case.stream_maxlen());
    let layer = "permit-timeout";
    let (session_id, _object_id, have_object, have_version, _) =
        waiting_catch_up_layer(&fx, &svc, layer, case).await;
    let held = hold_read_permit(&svc).await;
    let waits_before = read_permit_waits(&fx.metrics);

    let mut stream = subscribe(
        &svc,
        &session_id,
        layer,
        have_object,
        have_version,
        "permit-refused",
    )
    .await;
    let status = first_status(&mut stream, "the read-permit capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{case:?}: {status:?}");
    assert_eq!(status.message(), CATCHUP_READ_CAPACITY_MSG, "{case:?}");
    crate::common::refusal_detail::assert_catchup_read_permit_refusal(&status);

    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted under catchup_capacity",
    )
    .await;
    assert_eq!(
        (
            catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
        ),
        (1, 0),
        "{case:?}: the slot was taken at once, so the budget ran out on the permit"
    );
    assert_eq!(
        read_permit_waits(&fx.metrics),
        waits_before + 1,
        "{case:?}: the refused permit wait is observed once"
    );
    assert_eq!(
        reads_in_flight(&fx.metrics),
        1,
        "{case:?}: only the test's own permit is held"
    );
    await_condition(
        || {
            series(&fx.metrics, INFLIGHT) == 0
                && series(&fx.metrics, ADMITTED) == 0
                && svc.valkey.stream_mux.snapshot().subscriptions() == 0
        },
        "the refused catch-up gives back its slot, its admission and its reader",
    )
    .await;
    drop(held);
    assert!(read_gate_is_full(&svc), "{case:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_read_permit_wait_that_outlasts_the_budget_is_refused_catchup_capacity_before_content(
) {
    a_read_permit_wait_that_outlasts_the_budget_is_refused_before_content(WaitingCatchUp::BrandNew)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_read_permit_wait_that_outlasts_the_budget_is_refused_before_content() {
    a_read_permit_wait_that_outlasts_the_budget_is_refused_before_content(WaitingCatchUp::InWindow)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_read_permit_wait_that_outlasts_the_budget_is_refused_before_content() {
    a_read_permit_wait_that_outlasts_the_budget_is_refused_before_content(
        WaitingCatchUp::OutOfWindow,
    )
    .await;
}

/// A zero capacity budget means "never wait for capacity" (Spec SDD SR-7.3):
/// a catch-up takes a free read permit and catches up, and with the only
/// permit held it is refused `UNAVAILABLE` before content at once — the test
/// never frees its permit before the refusal arrives, so a refusal that
/// queued would never come. Its recorded permit wait started from an empty
/// budget.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_budget_catch_up_takes_a_free_read_permit_and_is_otherwise_refused_at_once() {
    const SUBJECT: &str = "zero-budget-refused";
    let fx = CapacityFixture::setup(64, 0).await;
    let mut svc = build_service(&fx, 4096);
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let layer = "zero-budget";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 181).await;

    let mut free = subscribe(&svc, &session_id, layer, 0, 0, "zero-budget-free").await;
    let (contents, ack) = recv_catch_up(&mut free).await;
    assert_eq!(
        (contents.len(), ack.sequence_num),
        (1, 2),
        "with no budget, a free read permit is still taken and the catch-up served"
    );

    let held = hold_read_permit(&svc).await;
    let mut refused = subscribe(&svc, &session_id, layer, 0, 0, SUBJECT).await;
    let status = first_status(&mut refused, "the zero-budget read-permit refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_READ_CAPACITY_MSG);
    crate::common::refusal_detail::assert_catchup_read_permit_refusal(&status);
    let charges = assert_budget_chain(
        &budget_seam,
        SUBJECT,
        0,
        &[
            BudgetWait::Admission,
            BudgetWait::StreamReader,
            BudgetWait::CatchUpSlot,
            BudgetWait::ReadPermit,
        ],
        &[],
    );
    assert_eq!(
        charges[3].budget_before,
        Duration::ZERO,
        "the permit wait had no budget to spend: {:?}",
        charges[3]
    );
    drop(held);
    assert!(read_gate_is_full(&svc));
}

/// `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` sizes the read gate the
/// catch-ups wait on. Every other fixture here runs the default of one permit;
/// at two, the gate holds two, so with one of them held by the test a
/// brand-new catch-up still takes the other and is served — at the default it
/// would be refused — and with both held the next one is refused its permit
/// `UNAVAILABLE` before content once its short budget runs out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_read_permit_count_is_the_configured_catch_up_snapshot_read_concurrency() {
    const READ_CONCURRENCY: u32 = 2;
    let fx = CapacityFixture::setup_read_concurrency(64, SHORT_BUDGET_MS, READ_CONCURRENCY).await;
    let svc = build_service(&fx, 4096);
    assert_eq!(
        (
            svc.valkey.catchup_read_gate.limit(),
            svc.valkey.catchup_read_gate.available(),
        ),
        (READ_CONCURRENCY as usize, READ_CONCURRENCY as usize),
        "the gate holds the configured number of read permits, all free"
    );
    let layer = "read-concurrency";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 186).await;

    let first = hold_read_permit(&svc).await;
    let mut served = subscribe(&svc, &session_id, layer, 0, 0, "beside-a-held-permit").await;
    let (contents, ack) = recv_catch_up(&mut served).await;
    assert_eq!(
        (contents.len(), ack.sequence_num),
        (1, 2),
        "with one of two permits held, a catch-up takes the other and is served"
    );
    await_condition(
        || reads_in_flight(&fx.metrics) == 1,
        "the served catch-up returned its permit",
    )
    .await;

    let second = hold_read_permit(&svc).await;
    let mut refused = subscribe(&svc, &session_id, layer, 0, 0, "both-permits-held").await;
    let status = first_status(&mut refused, "the read-permit capacity refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_READ_CAPACITY_MSG);
    crate::common::refusal_detail::assert_catchup_read_permit_refusal(&status);
    drop((first, second));
    assert!(read_gate_is_full(&svc));
}

/// An injected failure of the catch-up's atomic read (a transient fred
/// `Timeout` in place of its `EVALSHA` result) ends the stream `UNAVAILABLE`
/// before content and returns the read permit: the gate is back to full
/// capacity, the in-flight gauge to zero, and the next catch-up takes the
/// permit and is served.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_read_gate_returns_to_full_capacity_after_an_injected_read_failure() {
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::observability::metrics::STREAM_EXIT_ERROR;
    use live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector;

    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "permit-after-failure";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 191).await;
    // No checkpoint loop runs on this fixture, so only the catch-up's read
    // can consume the queued error.
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.push(FredError::new(
        FredErrorKind::Timeout,
        "injected catch-up atomic snapshot read timeout",
    ));
    fx.client
        .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));

    let mut failed = subscribe(&svc, &session_id, layer, 0, 0, "failed-read").await;
    let status = first_status(&mut failed, "the failed read's ending").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("EVALSHA object_snapshot_read"),
        "the ending came from the atomic read; got {status:?}"
    );
    assert_eq!(injector.remaining(), 0, "the injected error was consumed");
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_ERROR) == 1,
        "the failure is counted under error",
    )
    .await;
    await_condition(
        || {
            read_gate_is_full(&svc)
                && reads_in_flight(&fx.metrics) == 0
                && series(&fx.metrics, INFLIGHT) == 0
        },
        "the failed read returned its permit and its slot",
    )
    .await;

    let mut next = subscribe(&svc, &session_id, layer, 0, 0, "after-failure").await;
    let (_, ack) = recv_catch_up(&mut next).await;
    assert_eq!(ack.sequence_num, 2, "the permit is usable again");
}

/// What ends a catch-up at its read-permit wait in
/// [`a_catch_up_ended_at_its_read_permit_wait_leaves_the_read_gate_full`].
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum PermitWaitEnding {
    Shutdown,
    SessionDeletion,
    ClientDisconnect,
}

/// A catch-up waiting for its read permit is ended by its cause — graceful
/// shutdown, session deletion or the client leaving — promptly, against a
/// 60 s budget, and its dropped wait takes nothing: once the test frees its
/// own permit the gate is full and the in-flight gauge zero. The subscription
/// gives back its slot, its reader and its admission; a wait ended by its
/// cause ends before the budget is charged, so no permit charge is recorded.
///
/// The state-read seam parks the brand-new subject under its slot, before its
/// permit; the test holds the only permit, releases the subject into its
/// permit wait and ends it there — the permit wait is the subject's next
/// await, and it observes its endings first.
#[cfg(feature = "test-fault-injection")]
async fn a_catch_up_ended_at_its_read_permit_wait_leaves_the_read_gate_full(
    ending: PermitWaitEnding,
) {
    use live_edit::observability::metrics::{
        STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_SESSION_CANCEL, STREAM_EXIT_SHUTDOWN,
    };
    const SUBJECT: &str = "ended-at-permit";

    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let layer = "ended-at-permit";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 201).await;
    let held = hold_read_permit(&svc).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0, SUBJECT).await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("the subject parks under its slot, before its read permit");
    tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
        .await
        .expect("release the subject into its read-permit wait");

    let reason = match ending {
        PermitWaitEnding::Shutdown => {
            svc.shutdown.cancel();
            let status = first_status(&mut stream, "the shutdown ending").await;
            assert_eq!(status.code(), Code::Unavailable, "{status:?}");
            assert_eq!(status.message(), "server shutting down");
            STREAM_EXIT_SHUTDOWN
        }
        PermitWaitEnding::SessionDeletion => {
            svc.session_tokens
                .get(&session_id)
                .expect("the subscribe registered its session token")
                .cancel_deleted();
            let status = first_status(&mut stream, "the deletion ending").await;
            assert_eq!(status.code(), Code::NotFound, "{status:?}");
            assert_eq!(status.message(), "session was deleted");
            STREAM_EXIT_SESSION_CANCEL
        }
        PermitWaitEnding::ClientDisconnect => {
            drop(stream);
            STREAM_EXIT_CLIENT_DISCONNECT
        }
    };
    await_condition(
        || exits(&fx.metrics, reason) == 1,
        "the ending is counted under its cause's reason",
    )
    .await;
    assert_eq!(
        exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY),
        0,
        "{ending:?}"
    );
    assert!(
        budget_seam
            .charges_for(SUBJECT)
            .iter()
            .all(|c| c.wait != BudgetWait::ReadPermit),
        "{ending:?}: a permit wait ended by its cause is not charged"
    );
    await_condition(
        || {
            series(&fx.metrics, INFLIGHT) == 0
                && series(&fx.metrics, ADMITTED) == 0
                && svc.valkey.stream_mux.snapshot().subscriptions() == 0
        },
        "the ended catch-up gives back its slot, its admission and its reader",
    )
    .await;
    assert_eq!(
        reads_in_flight(&fx.metrics),
        1,
        "{ending:?}: only the test's own permit is held"
    );
    drop(held);
    assert!(
        read_gate_is_full(&svc) && reads_in_flight(&fx.metrics) == 0,
        "{ending:?}: the dropped wait took nothing"
    );
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_shutdown_at_its_read_permit_wait_leaves_the_read_gate_full() {
    a_catch_up_ended_at_its_read_permit_wait_leaves_the_read_gate_full(PermitWaitEnding::Shutdown)
        .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_session_deletion_at_its_read_permit_wait_leaves_the_read_gate_full() {
    a_catch_up_ended_at_its_read_permit_wait_leaves_the_read_gate_full(
        PermitWaitEnding::SessionDeletion,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_ended_by_client_disconnect_at_its_read_permit_wait_leaves_the_read_gate_full() {
    a_catch_up_ended_at_its_read_permit_wait_leaves_the_read_gate_full(
        PermitWaitEnding::ClientDisconnect,
    )
    .await;
}

/// `ValkeyClient::quit` — the shutdown step that closes the store — closes the
/// read gate, and a catch-up waiting for a read permit when it does ends
/// `shutdown`: in-stream `UNAVAILABLE "server shutting down"` as its first
/// item, with no content or ack, counted under `shutdown` and not
/// `catchup_capacity`, and its slot and admission given back.
///
/// The service's shutdown token is never cancelled here, so the closed gate
/// is the only thing that can end the wait (the 60 s budget is never reached).
/// The test holds the only permit; the subject is past its slot (its slot wait
/// recorded `acquired`), so its next wait is the permit's — closing the gate
/// refuses it whether it is already queued or about to queue. Once closed, the
/// gate refuses every later acquisition too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_waiting_for_its_read_permit_when_the_client_quits_ends_shutdown_before_content()
{
    use live_edit::observability::metrics::STREAM_EXIT_SHUTDOWN;
    use live_edit::storage::subscriber_admission::CatchUpRefusal;

    let mut fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "permit-wait-at-quit";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 206).await;
    let held = hold_read_permit(&svc).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "waiting-at-quit").await;
    await_condition(
        || {
            catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED) == 1
                && series(&fx.metrics, INFLIGHT) == 1
        },
        "the subject takes its slot and goes on to its read-permit wait",
    )
    .await;

    fx.purge_and_quit().await;
    let status = first_status(&mut stream, "the closed read gate's ending").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), "server shutting down");
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_SHUTDOWN) == 1,
        "the ending is counted under shutdown",
    )
    .await;
    assert_eq!(
        exits(&fx.metrics, STREAM_EXIT_CATCHUP_CAPACITY),
        0,
        "a closed gate is shutdown, not a capacity refusal"
    );
    assert!(
        !svc.shutdown.is_cancelled(),
        "fixture self-check: the closed gate, not the shutdown token, ended the wait"
    );
    await_condition(
        || series(&fx.metrics, INFLIGHT) == 0 && series(&fx.metrics, ADMITTED) == 0,
        "the ended catch-up gives back its slot and its admission",
    )
    .await;
    assert_eq!(
        svc.valkey
            .catchup_read_gate
            .acquire(
                tokio::time::Instant::now() + Duration::from_secs(3_600),
                SnapshotReadConsumer::Catchup,
            )
            .await
            .err(),
        Some(CatchUpRefusal::Closed),
        "quit closed the read gate: every later acquisition is refused"
    );
    drop(held);
}

/// The read permit nests inside the catch-up slot (Spec SDD SR-7.5), and a
/// catch-up cancelled during its atomic read returns it.
///
/// With the snapshot lane stalled (`stall_pool`: its one connection parked in
/// a `BLPOP`, so the atomic read's `EVALSHA` queues behind it): while the
/// subject waits for the only slot it holds no permit; once it has the slot it
/// takes the permit and sits in its read holding both — and only its reader
/// reservation, no registered stream, since it registers at the read's cut.
/// The client then leaves; the lane is released, the read completes, and the
/// subscription ends `client_disconnect` with its permit, slot, reader and
/// admission all given back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_read_permit_is_taken_only_inside_the_slot_and_returned_by_a_catch_up_cancelled_in_its_read(
) {
    use live_edit::observability::metrics::STREAM_EXIT_CLIENT_DISCONNECT;
    use live_edit::storage::test_support::{release_all, stall_pool};

    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "cancelled-in-read";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 211).await;
    let observer = fx.client.maintenance_pool.next().clone();
    let stalls = stall_pool(
        &fx.client.snapshot_pool,
        &observer,
        &format!("{}conn-stall:snapshot:", fx.prefix),
    )
    .await;
    let held_slot = hold_catchup_slot(&svc).await;

    let stream = subscribe(&svc, &session_id, layer, 0, 0, "in-read").await;
    await_condition(
        || series(&fx.metrics, WAITING) == 1,
        "the subject queues for the only slot",
    )
    .await;
    assert!(
        read_gate_is_full(&svc) && reads_in_flight(&fx.metrics) == 0,
        "waiting for its slot, the subject holds no read permit"
    );
    drop(held_slot);
    await_condition(
        || reads_in_flight(&fx.metrics) == 1,
        "the subject takes its read permit and sits in its stalled atomic read",
    )
    .await;
    assert_eq!(
        (
            series(&fx.metrics, INFLIGHT),
            svc.valkey.stream_mux.snapshot().subscriptions(),
            series(&fx.metrics, STREAMS),
        ),
        (1, 1, 0),
        "in its read the subject holds its slot (the permit is inside it) and its reader \
         reservation, and has registered nothing"
    );

    drop(stream);
    release_all(stalls).await;
    await_condition(
        || exits(&fx.metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "the cancelled catch-up ends client_disconnect",
    )
    .await;
    await_condition(
        || {
            read_gate_is_full(&svc)
                && reads_in_flight(&fx.metrics) == 0
                && series(&fx.metrics, INFLIGHT) == 0
                && series(&fx.metrics, ADMITTED) == 0
                && svc.valkey.stream_mux.snapshot().subscriptions() == 0
        },
        "the cancelled catch-up gave back its permit, its slot, its reader and its admission",
    )
    .await;
}

/// The catch-up slot, not the read permit, bounds the catch-up builds running
/// at once (Spec SDD SR-7.5): with two slots and one permit, five brand-new
/// joiners of an external-value layer build two at a time — their reads take
/// the one permit in turn and return it as each read returns — and never
/// three. Each build is held in [`ParkingExtStore`] while it resolves the
/// layer's external value; at every step the two held builds hold the two
/// slots and no permit, and the rest wait for a slot. Released one at a time,
/// every joiner catches up.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_catch_up_builds_never_exceed_the_slot_count() {
    const SLOTS: usize = 2;
    const JOINERS: usize = 5;
    let fx = CapacityFixture::setup(SLOTS, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let ext = install_parking_ext_store(&mut svc);
    let layer = "concurrent-builds";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit_with_external_value(&svc, &session_id, layer, object_id, 221).await;

    ext.arm();
    let mut streams = Vec::with_capacity(JOINERS);
    for i in 0..JOINERS {
        streams.push(subscribe(&svc, &session_id, layer, 0, 0, &format!("joiner-{i}")).await);
    }
    for released in 0..JOINERS {
        let building = SLOTS.min(JOINERS - released);
        let waiting = (JOINERS - released - building) as i64;
        await_condition(
            || ext.parked() == building && series(&fx.metrics, WAITING) == waiting,
            "the held builds fill the slots and the rest wait for one",
        )
        .await;
        assert_eq!(
            (series(&fx.metrics, INFLIGHT), reads_in_flight(&fx.metrics)),
            (building as i64, 0),
            "{building} builds hold {building} slots and no read permit (released {released})"
        );
        ext.release_one();
    }
    assert_eq!(
        ext.max_parked(),
        SLOTS,
        "never more catch-up builds at once than slots, and as many as slots — more than the \
         one read permit"
    );
    ext.disarm();
    tokio::time::timeout(DRAIN, async {
        for stream in &mut streams {
            let (contents, ack) = recv_catch_up(stream).await;
            assert_eq!((contents.len(), ack.sequence_num), (1, 2));
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the joiners' catch-ups did not all finish within {DRAIN:?}"));
}

/// In every regime, every read-permit wait a catch-up makes comes before its
/// first content message (the Epic's "no capacity refusal after the first
/// content message"; Spec SDD SR-7.3): every charge the service recorded for
/// the subscription — read permits included — was recorded by the time its
/// first content message arrived, and none after, through the rest of its
/// catch-up and two live commits.
///
/// Each regime records exactly one permit wait, after the reader reservation
/// and the slot wait, and then its payload's reservation of the catch-up
/// payload budget (b.ntm), still before content: the charge order pins the
/// nesting.
#[cfg(feature = "test-fault-injection")]
async fn every_read_permit_wait_comes_before_the_first_content_message(case: WaitingCatchUp) {
    const SUBJECT: &str = "charged-before-content";
    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, case.stream_maxlen());
    let budget_seam = Arc::new(CapacityBudgetSeam::new());
    svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
    let layer = "charged-before-content";
    let (session_id, object_id, have_object, have_version, _) =
        waiting_catch_up_layer(&fx, &svc, layer, case).await;

    let mut stream = subscribe(&svc, &session_id, layer, have_object, have_version, SUBJECT).await;
    let (first, _) = recv_message(&mut stream, "the first content message").await;
    assert!(
        first.total_size > 0,
        "{case:?}: fixture self-check: the catch-up sends content"
    );
    let at_first_content = budget_seam.charges_for(SUBJECT);
    let mut last = first;
    while last.total_size > 0 {
        last = recv_message(&mut stream, "the rest of the catch-up")
            .await
            .0;
    }
    for n in 1..=2 {
        commit(&svc, &session_id, layer, object_id, 240 + n).await;
        let (live, _) = recv_message(&mut stream, "a live commit").await;
        assert_eq!(live.sequence_num, last.sequence_num + n, "{case:?}");
    }
    let at_end = budget_seam.charges_for(SUBJECT);
    assert_eq!(
        at_end.len(),
        at_first_content.len(),
        "{case:?}: no capacity wait — read permits included — after the first content message: \
         {at_end:#?}"
    );

    let chain = [
        BudgetWait::Admission,
        BudgetWait::StreamReader,
        BudgetWait::CatchUpSlot,
        BudgetWait::ReadPermit,
        BudgetWait::CatchUpPayload,
    ];
    assert_budget_chain(&budget_seam, SUBJECT, PARKED_BUDGET_MS, &chain, &[]);
    assert!(
        read_gate_is_full(&svc) && reads_in_flight(&fx.metrics) == 0,
        "{case:?}: every read permit is back"
    );
}

/// [`every_read_permit_wait_comes_before_the_first_content_message`] for
/// `case`, boxed. Every caller goes through here: that helper's future is
/// large, and a `#[tokio::test]` would otherwise hold it by value in the
/// debug build's `block_on` frames on std's default 2 MiB test thread, which the
/// out-of-window cases already overflowed; boxing puts it on the heap.
#[cfg(feature = "test-fault-injection")]
fn charged_before_content(case: WaitingCatchUp) -> Pin<Box<impl std::future::Future<Output = ()>>> {
    Box::pin(every_read_permit_wait_comes_before_the_first_content_message(case))
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_catch_up_waits_for_its_read_permit_only_before_content() {
    charged_before_content(WaitingCatchUp::BrandNew).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_catch_up_waits_for_its_read_permit_only_before_content() {
    charged_before_content(WaitingCatchUp::InWindow).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_catch_up_waits_for_its_read_permit_only_before_content() {
    charged_before_content(WaitingCatchUp::OutOfWindow).await;
}

// ═══════════════════════════════════════════════════════════════════════════
// Empty commits, which take no read
// ═══════════════════════════════════════════════════════════════════════════

/// How many caught-up subscribers
/// [`caught_up_subscribers_receive_empty_commits_as_empty_deltas_while_the_read_permit_is_held`]
/// runs.
const EMPTY_SUBSCRIBERS: usize = 3;

/// How many empty commits land close together in that test.
const EMPTY_COMMITS: u64 = 3;

/// Every caught-up subscriber receives an empty commit — an `UpdateLayer`
/// with no chunks, whose stream entry carries no relay bytes — as the empty
/// delta at its own sequence, based at the sequence before it, with no read
/// and no read permit (SDD §12.2): while the test holds the replica's only
/// read permit, as a catch-up's atomic read does, [`EMPTY_SUBSCRIBERS`]
/// caught-up subscribers each receive [`EMPTY_COMMITS`] empty commits landed
/// close together, in order, and stay live. No subscriber waited for a read
/// permit or read the layer.
///
/// The regression it pins: a subscriber that took a read permit to rebuild an
/// empty commit parked behind a catch-up's permit, and at the design rate a
/// few empty commits close together ended every caught-up subscription of the
/// replica. One that skipped the empty commit instead would send nothing here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn caught_up_subscribers_receive_empty_commits_as_empty_deltas_while_the_read_permit_is_held()
{
    let fx = CapacityFixture::setup(64, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "steady-empty-commits";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 341).await;
    let caught_up = 2;
    let mut streams = Vec::with_capacity(EMPTY_SUBSCRIBERS);
    for i in 0..EMPTY_SUBSCRIBERS {
        // Up to date: acked at once, with no catch-up read.
        let mut stream = subscribe(
            &svc,
            &session_id,
            layer,
            object_id,
            caught_up,
            &format!("steady-empty-{i}"),
        )
        .await;
        recv_ack(&mut stream, object_id, caught_up).await;
        streams.push(stream);
    }
    let held = hold_read_permit(&svc).await;
    let (waits_before, reads_before) = (
        all_read_permit_waits(&fx.metrics),
        all_snapshot_reads(&fx.metrics),
    );

    for _ in 0..EMPTY_COMMITS {
        empty_commit(&svc, &session_id, layer, object_id).await;
    }
    tokio::time::timeout(DRAIN, async {
        for (i, stream) in streams.iter_mut().enumerate() {
            for seq in caught_up + 1..=caught_up + EMPTY_COMMITS {
                let (header, body) =
                    recv_message(stream, "an empty commit, with the read permit held").await;
                assert_eq!(
                    (header.object_id, header.sequence_num),
                    (object_id, seq),
                    "subscriber {i}: every empty commit is delivered at its own sequence, in order"
                );
                assert_eq!(
                    delta_of(&body),
                    the_empty_delta(seq - 1),
                    "subscriber {i}: the empty commit at {seq} is the empty delta based at {}",
                    seq - 1
                );
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the empty commits were not all delivered within {DRAIN:?}"));
    assert_eq!(
        (
            all_read_permit_waits(&fx.metrics) - waits_before,
            all_snapshot_reads(&fx.metrics) - reads_before,
            reads_in_flight(&fx.metrics),
        ),
        (0, 0, 1),
        "no subscriber waited for a read permit or read the layer; only the test's own permit \
         is held"
    );
    drop(held);

    commit(&svc, &session_id, layer, object_id, 342).await;
    let next = caught_up + EMPTY_COMMITS + 1;
    tokio::time::timeout(DRAIN, async {
        for (i, stream) in streams.iter_mut().enumerate() {
            let (live, _) = recv_message(stream, "the next live commit").await;
            assert_eq!(live.sequence_num, next, "subscriber {i} is still live");
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the next live commit was not delivered within {DRAIN:?}"));
}

/// An empty commit on a layer above the atomic-read limits (a forced-low
/// ceiling) is delivered the same way: the empty delta at its sequence, with
/// no read at all — no atomic read, so none answers `TOO_LARGE` (the answer
/// that sends a read above the limits on to read the data pool), no read
/// permit, and no diff build. The subscriber then goes on relaying.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_commit_above_the_atomic_read_limits_is_delivered_with_no_read() {
    let fx = CapacityFixture::setup_above_ceiling(64, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "steady-empty-above-ceiling";
    let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, object_id, 351).await;
    let mut stream = subscribe(&svc, &session_id, layer, object_id, 2, "steady-empty-large").await;
    recv_ack(&mut stream, object_id, 2).await;
    let (builds_before, reads_before, waits_before) = (
        diff_builds(&fx.metrics),
        all_snapshot_reads(&fx.metrics),
        all_read_permit_waits(&fx.metrics),
    );

    empty_commit(&svc, &session_id, layer, object_id).await;
    let (content, body) = recv_message(&mut stream, "the empty commit").await;
    assert_eq!((content.object_id, content.sequence_num), (object_id, 3));
    assert_eq!(delta_of(&body), the_empty_delta(2));
    assert_eq!(
        (
            all_snapshot_reads(&fx.metrics) - reads_before,
            all_read_permit_waits(&fx.metrics) - waits_before,
            diff_builds(&fx.metrics) - builds_before,
        ),
        (0, 0, 0),
        "the empty commit took no snapshot read, no read permit and no diff build"
    );

    commit(&svc, &session_id, layer, object_id, 352).await;
    let (live, _) = recv_message(&mut stream, "the next live commit").await;
    assert_eq!(live.sequence_num, 4, "the subscription is still live");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cluster mode: the stream-reader wait
// ═══════════════════════════════════════════════════════════════════════════

/// The cluster-mode stream-reader wait: every reader is bound to one hash slot
/// at a time, so a subscription on another slot waits for a free reader under
/// the capacity budget, and a catch-up's reader reservation keeps its reader
/// bound through its slot wait. Self-skips without
/// `LIVE_EDIT_TEST_CLUSTER_URL` (`bash scripts/dev-valkey.sh cluster-up`); the
/// reserved-ending cases and the reader-wait charge also need
/// `test-fault-injection`, the rest no cargo feature.
mod cluster {
    use super::*;

    use live_edit::observability::metrics::{
        STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_READER_CAPACITY, STREAM_EXIT_SESSION_CANCEL,
        STREAM_EXIT_SHUTDOWN,
    };
    use live_edit::storage::stream_mux::router::Routing;
    use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};

    use super::super::log_capture::{
        assert_common_fields, assert_field, ThreadCapture, RPC_FAILURE_MSG,
    };

    /// A reader-capacity refusal's budget, paid once by the test asserting it.
    const CLUSTER_SHORT_BUDGET_MS: u64 = 500;

    /// The in-stream status of a subscription refused a stream reader.
    const READER_CAPACITY_MSG: &str = "this replica is serving as many distinct layers as it has \
         stream readers (LIVE_EDIT_SUBSCRIBER_POOL_SIZE); retry";

    /// A one-reader cluster fixture with `max_catchups` catch-up slots, or
    /// `None` (with a `SKIP:` line) when no cluster is configured.
    async fn one_reader_cluster(max_catchups: usize, budget_ms: u64) -> Option<CapacityFixture> {
        let Some(url) = live_edit::storage::test_support::test_cluster_url() else {
            eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
            return None;
        };
        let fx = CapacityFixture::setup_with(Some(url), 1, 0, max_catchups, budget_ms).await;
        assert_eq!(
            fx.client.stream_mux.snapshot().routing,
            Routing::SlotAffinity,
            "fixture self-check: cluster routing binds each reader to one hash slot"
        );
        Some(fx)
    }

    /// Bind the only reader to a hash slot other than `object_id`'s stream's,
    /// so a subscription on `object_id` must wait for a reader.
    async fn bind_the_reader_elsewhere(
        fx: &CapacityFixture,
        object_id: u64,
    ) -> LayerStreamConsumer {
        let slot_of =
            |oid: u64| fred::util::redis_keyslot(fx.key_schema.layer_stream(oid).as_bytes());
        let other = (1_000_000u64..)
            .find(|&oid| slot_of(oid) != slot_of(object_id))
            .expect("some object id hashes to another slot");
        consume_layer_stream(
            &fx.client,
            &fx.key_schema,
            other,
            StreamCursor::From("0-0".into()),
            tokio::time::Instant::now() + SETTLE,
        )
        .await
        .expect("the only reader is free")
    }

    fn bind_refusals(metrics: &Metrics) -> i64 {
        series(metrics, "live_edit_stream_mux_bind_refusals_total")
    }

    /// A reader wait that outlasts the capacity budget ends in-stream
    /// `UNAVAILABLE` naming `LIVE_EDIT_SUBSCRIBER_POOL_SIZE`, under
    /// `reason="reader_capacity"`, as the stream's first item, with one WARN
    /// completion line. The subscription never waited for, or held, a catch-up
    /// slot during the reader wait (`catchups_inflight` 0, no slot-wait
    /// observation at all), and its admission is returned.
    ///
    /// Brand new, and a same-object reconnect: both reserve their stream
    /// reader before their slot wait.
    async fn a_reader_wait_that_outlasts_the_budget_is_refused(reconnect: bool) {
        let Some(fx) = one_reader_cluster(64, CLUSTER_SHORT_BUDGET_MS).await else {
            return;
        };
        let svc = build_service(&fx, 4096);
        let layer = "reader-capacity";
        let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
        commit(&svc, &session_id, layer, object_id, 121).await;
        let holder = bind_the_reader_elsewhere(&fx, object_id).await;
        let capture = ThreadCapture::start();

        let (have_object, have_version) = if reconnect { (object_id, 1) } else { (0, 0) };
        let mut stream = subscribe(
            &svc,
            &session_id,
            layer,
            have_object,
            have_version,
            "no-reader",
        )
        .await;
        let status = first_status(&mut stream, "the reader capacity refusal").await;
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert_eq!(status.message(), READER_CAPACITY_MSG);
        crate::common::refusal_detail::assert_reader_capacity_refusal(&status);
        await_condition(
            || exits(&fx.metrics, STREAM_EXIT_READER_CAPACITY) == 1,
            "the refusal is counted under reader_capacity",
        )
        .await;
        assert_eq!(bind_refusals(&fx.metrics), 1);
        assert_eq!(series(&fx.metrics, INFLIGHT), 0);
        for result in [
            CATCHUP_WAIT_RESULT_ACQUIRED,
            CATCHUP_WAIT_RESULT_CANCELLED,
            CATCHUP_WAIT_RESULT_TIMED_OUT,
        ] {
            assert_eq!(
                catchup_waits(&fx.metrics, result),
                0,
                "no catch-up slot was waited for during the reader wait ({result})"
            );
        }
        await_condition(
            || series(&fx.metrics, ADMITTED) == 0,
            "the refused subscription returns its admission",
        )
        .await;

        let captured = capture.snapshot();
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
        assert_field(event, "reason", STREAM_EXIT_READER_CAPACITY.into());
        drop(capture);
        drop(holder);
        fx.purge().await;
    }

    #[tokio::test]
    async fn a_brand_new_reader_wait_that_outlasts_the_budget_is_refused_reader_capacity() {
        a_reader_wait_that_outlasts_the_budget_is_refused(false).await;
    }

    #[tokio::test]
    async fn a_reconnect_reader_wait_that_outlasts_the_budget_is_refused_reader_capacity() {
        a_reader_wait_that_outlasts_the_budget_is_refused(true).await;
    }

    /// What ends a reader wait in [`a_reader_wait_ends_by_its_cause`].
    #[derive(Clone, Copy, Debug)]
    enum ReaderWaitEnding {
        Shutdown,
        SessionDeletion,
        ClientDisconnect,
    }

    /// A reader wait ends by its cause, promptly — within `SETTLE`, against a
    /// 60 s budget — not only when the budget or the multiplexer's own
    /// shutdown ends it: graceful shutdown (`UNAVAILABLE "server shutting
    /// down"`, `shutdown`, logged at INFO), session deletion (`NOT_FOUND`,
    /// `session_cancel`) or the client leaving (`client_disconnect`). Its
    /// admission comes back at once, and it never took a catch-up slot.
    async fn a_reader_wait_ends_by_its_cause(ending: ReaderWaitEnding) {
        let Some(fx) = one_reader_cluster(64, PARKED_BUDGET_MS).await else {
            return;
        };
        let svc = build_service(&fx, 4096);
        let layer = "reader-wait-ending";
        let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
        let holder = bind_the_reader_elsewhere(&fx, object_id).await;
        let capture = ThreadCapture::start();

        let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "waiting").await;
        await_condition(
            || series(&fx.metrics, ADMITTED) == 1,
            "the subscription is admitted and its task started",
        )
        .await;
        assert_eq!(
            fx.client.stream_mux.snapshot().subscriptions(),
            1,
            "fixture self-check: only the holder holds a reader; the subject waits for one"
        );

        let reason = match ending {
            ReaderWaitEnding::Shutdown => {
                svc.shutdown.cancel();
                let status = first_status(&mut stream, "the shutdown ending").await;
                assert_eq!(status.code(), Code::Unavailable, "{status:?}");
                assert_eq!(status.message(), "server shutting down");
                STREAM_EXIT_SHUTDOWN
            }
            ReaderWaitEnding::SessionDeletion => {
                svc.session_tokens
                    .get(&session_id)
                    .expect("the subscribe registered its session token")
                    .cancel_deleted();
                let status = first_status(&mut stream, "the deletion ending").await;
                assert_eq!(status.code(), Code::NotFound, "{status:?}");
                assert_eq!(status.message(), "session was deleted");
                STREAM_EXIT_SESSION_CANCEL
            }
            ReaderWaitEnding::ClientDisconnect => {
                drop(stream);
                STREAM_EXIT_CLIENT_DISCONNECT
            }
        };
        await_condition(
            || exits(&fx.metrics, reason) == 1,
            "the reader wait's exit carries its cause's reason",
        )
        .await;
        assert_eq!(
            bind_refusals(&fx.metrics),
            0,
            "{ending:?}: the wait ended by its cause, not by its budget running out"
        );
        assert_eq!(exits(&fx.metrics, STREAM_EXIT_READER_CAPACITY), 0);
        assert_eq!(series(&fx.metrics, INFLIGHT), 0);
        assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_ACQUIRED), 0);
        await_condition(
            || series(&fx.metrics, ADMITTED) == 0,
            "the ended subscription returns its admission promptly",
        )
        .await;

        if matches!(ending, ReaderWaitEnding::Shutdown) {
            let captured = capture.snapshot();
            let event = captured.only(RPC_FAILURE_MSG);
            assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "UNAVAILABLE");
            assert_field(event, "reason", STREAM_EXIT_SHUTDOWN.into());
        }
        drop(capture);
        drop(holder);
        fx.purge().await;
    }

    #[tokio::test]
    async fn a_reader_wait_ends_unavailable_at_info_when_the_replica_shuts_down() {
        a_reader_wait_ends_by_its_cause(ReaderWaitEnding::Shutdown).await;
    }

    #[tokio::test]
    async fn a_reader_wait_ends_not_found_when_its_session_is_deleted() {
        a_reader_wait_ends_by_its_cause(ReaderWaitEnding::SessionDeletion).await;
    }

    #[tokio::test]
    async fn a_reader_wait_ends_client_disconnect_when_its_client_goes_away() {
        a_reader_wait_ends_by_its_cause(ReaderWaitEnding::ClientDisconnect).await;
    }

    /// How long
    /// `a_reader_wait_is_charged_to_the_budget_and_the_later_slot_wait_gets_only_what_it_left`
    /// keeps the only stream reader bound elsewhere, to spend budget.
    #[cfg(feature = "test-fault-injection")]
    const READER_SPEND: Duration = Duration::from_millis(500);

    /// The cluster stream-reader wait is charged to the subscription's one
    /// capacity budget, so the catch-up slot wait after it gets only what the
    /// admission and reader waits left: its deadline is the full budget less
    /// both charges. Asserted on the charges the service recorded
    /// ([`assert_budget_chain`]) rather than by racing a held slot against the
    /// slot wait's deadline.
    ///
    /// A brand-new subscriber waits for a reader while the test holds the only
    /// one on another hash slot, then for the only catch-up slot, which the
    /// test frees as soon as the subject queues for it. Its charges are, in
    /// order: admission, the reader wait, the slot, the read permit for its
    /// atomic read. The budget is
    /// `PARKED_BUDGET_MS` (60 s), which no wait here comes near, so the
    /// subject is never refused and catches up.
    ///
    /// `READER_SPEND` exists to SPEND budget; it is not synchronisation. The
    /// reader wait has no in-progress signal of its own — the spend starts once
    /// the subject is admitted, before the layer-mapping and header reads that
    /// precede its reader wait — so its charge is certain only to be above
    /// zero, which is all the chain needs to catch a dropped reader charge.
    /// The slot wait, freed at once and so normally charged far less than the
    /// reader wait before it, is the one whose deadline bracket tells "handed
    /// what was left" from "handed a fresh full wait".
    ///
    /// Needs `LIVE_EDIT_TEST_CLUSTER_URL`: on standalone Valkey the reader wait
    /// never waits, so nothing else pins this charge. Needs
    /// `test-fault-injection` for the budget seam; without the feature it is
    /// compiled out, not skipped. CI's `test-cluster:` job has both.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test]
    async fn a_reader_wait_is_charged_to_the_budget_and_the_later_slot_wait_gets_only_what_it_left()
    {
        const SUBJECT: &str = "reader-charged";
        let Some(fx) = one_reader_cluster(1, PARKED_BUDGET_MS).await else {
            return;
        };
        let mut svc = build_service(&fx, 4096);
        let budget_seam = Arc::new(CapacityBudgetSeam::new());
        svc.capacity_budget_seam_for_test = Some(Arc::clone(&budget_seam));
        let layer = "reader-charged";
        let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
        commit(&svc, &session_id, layer, object_id, 141).await;
        let holder = bind_the_reader_elsewhere(&fx, object_id).await;
        let slot = hold_catchup_slot(&svc).await;

        let mut stream = subscribe(&svc, &session_id, layer, 0, 0, SUBJECT).await;
        await_condition(
            || series(&fx.metrics, ADMITTED) == 1,
            "the subscription is admitted and its task started",
        )
        .await;
        tokio::time::sleep(READER_SPEND).await;
        drop(holder);
        await_condition(
            || series(&fx.metrics, WAITING) == 1,
            "the subject takes the freed reader and queues for the only slot",
        )
        .await;
        drop(slot);

        let (contents, ack) = recv_catch_up(&mut stream).await;
        assert_eq!(
            contents
                .iter()
                .map(|h| (h.object_id, h.sequence_num))
                .collect::<Vec<_>>(),
            [(object_id, 2)],
            "the subject catches up once it has its reader and its slot"
        );
        assert_eq!((ack.object_id, ack.sequence_num), (object_id, 2));

        let charges = assert_budget_chain(
            &budget_seam,
            SUBJECT,
            PARKED_BUDGET_MS,
            &[
                BudgetWait::Admission,
                BudgetWait::StreamReader,
                BudgetWait::CatchUpSlot,
                BudgetWait::ReadPermit,
                BudgetWait::CatchUpPayload,
            ],
            &[],
        );
        let (reader, slot) = (&charges[1], &charges[2]);
        assert!(
            reader.charged > Duration::ZERO,
            "the reader wait is charged what it waited: {reader:?}"
        );
        assert_eq!(
            slot.budget_before,
            Duration::from_millis(PARKED_BUDGET_MS) - charges[0].charged - reader.charged,
            "the slot wait gets only what the admission and reader waits left: {charges:#?}"
        );
        assert_eq!(
            bind_refusals(&fx.metrics),
            0,
            "the reader wait ended by a freed reader, not by the budget"
        );
        assert_eq!(catchup_waits(&fx.metrics, CATCHUP_WAIT_RESULT_TIMED_OUT), 0);
        drop(stream);
        fx.purge().await;
    }

    /// A catch-up waiting for its slot holds its reader reservation, and the
    /// reservation keeps the only reader bound to the layer's hash slot
    /// through the wait: a layer in another slot, asking for a reader
    /// meanwhile, is refused `Saturated` — the refusal the gRPC layer ends
    /// `reader_capacity` — counted once on the bind refusals, while the
    /// subject is still queued for its slot with no stream registered. Freed,
    /// the subject catches up on the reader it kept.
    ///
    /// The other layer asks at the storage level, with its own 500 ms
    /// deadline: a second `SubscribeReadLayer` would draw on the same
    /// capacity budget as the subject, whose earlier slot wait would then run
    /// out first and release the reader before the other's refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_catch_up_waiting_for_its_slot_keeps_the_only_reader_bound_to_its_layers_slot() {
        let Some(fx) = one_reader_cluster(1, PARKED_BUDGET_MS).await else {
            return;
        };
        let svc = build_service(&fx, 4096);
        let layer = "reserved-binds";
        let (session_id, object_id) = create_layer(&fx, &svc, layer).await;
        commit(&svc, &session_id, layer, object_id, 171).await;
        let slot_of =
            |oid: u64| fred::util::redis_keyslot(fx.key_schema.layer_stream(oid).as_bytes());
        let held = hold_catchup_slot(&svc).await;

        let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "reserved").await;
        await_condition(
            || series(&fx.metrics, WAITING) == 1,
            "the subject reserved its reader and queued for the only slot",
        )
        .await;
        let snap = fx.client.stream_mux.snapshot();
        assert_eq!(snap.subscriptions(), 1);
        assert_eq!(
            snap.readers[0].slot,
            Some(slot_of(object_id)),
            "the reservation binds the only reader to the subject's slot"
        );
        assert_eq!(series(&fx.metrics, STREAMS), 0, "no stream is registered");

        let other = (object_id + 1..)
            .find(|&oid| slot_of(oid) != slot_of(object_id))
            .expect("some object id hashes to another slot");
        let refused = consume_layer_stream(
            &fx.client,
            &fx.key_schema,
            other,
            StreamCursor::From("0-0".into()),
            tokio::time::Instant::now() + Duration::from_millis(CLUSTER_SHORT_BUDGET_MS),
        )
        .await;
        assert!(
            matches!(refused, Err(live_edit::storage::ConsumeError::Saturated)),
            "a layer in another slot is refused a reader while the subject waits reserved"
        );
        assert_eq!(bind_refusals(&fx.metrics), 1);
        assert_eq!(
            series(&fx.metrics, WAITING),
            1,
            "the subject is still waiting for its slot"
        );
        assert_eq!(
            fx.client.stream_mux.snapshot().readers[0].slot,
            Some(slot_of(object_id))
        );

        drop(held);
        let (contents, ack) = recv_catch_up(&mut stream).await;
        assert_eq!(
            contents
                .iter()
                .map(|h| (h.object_id, h.sequence_num))
                .collect::<Vec<_>>(),
            [(object_id, 2)],
            "the subject catches up on the reader it kept"
        );
        assert_eq!((ack.object_id, ack.sequence_num), (object_id, 2));
        assert_eq!(exits(&fx.metrics, STREAM_EXIT_READER_CAPACITY), 0);
    }

    // Every ending while reserved releases the reader and unbinds its slot
    // (`an_ending_while_reserved_releases_the_reader`), on a one-reader
    // cluster. These need `test-fault-injection` as well as
    // `LIVE_EDIT_TEST_CLUSTER_URL`: without the feature they are compiled
    // out, not skipped.

    #[cfg(feature = "test-fault-injection")]
    async fn on_a_cluster(ending: ReservedEnding) {
        let budget_ms = match ending {
            ReservedEnding::CatchUpCapacity => SHORT_BUDGET_MS,
            _ => PARKED_BUDGET_MS,
        };
        let Some(fx) = one_reader_cluster(1, budget_ms).await else {
            return;
        };
        an_ending_while_reserved_releases_the_reader(fx, ending).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_ended_by_shutdown_unbinds_its_readers_slot() {
        on_a_cluster(ReservedEnding::Shutdown).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_ended_by_session_deletion_unbinds_its_readers_slot() {
        on_a_cluster(ReservedEnding::SessionDeletion).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_ended_by_client_disconnect_unbinds_its_readers_slot() {
        on_a_cluster(ReservedEnding::ClientDisconnect).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_refused_catchup_capacity_unbinds_its_readers_slot() {
        on_a_cluster(ReservedEnding::CatchUpCapacity).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_ended_by_its_deleted_scan_rebinds_the_reader_to_the_new_object() {
        on_a_cluster(ReservedEnding::DeletedScan).await;
    }

    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reserved_catch_up_whose_atomic_read_is_not_found_rebinds_the_reader_to_the_new_object(
    ) {
        on_a_cluster(ReservedEnding::NotFoundAtTheAtomicRead).await;
    }
}

// ── /scaling/v1 counters across re-bootstraps (b.wnb) ───────────────────────
//
// The autoscaler's pending and completed counts take each subscription's
// FIRST state-reading catch-up exactly once. A re-bootstrap continues that
// catch-up when it comes before the first ack, and is not a first catch-up
// at all when it comes after it.

/// `(admitted, pending, completed)` as `/scaling/v1` would serve them.
#[cfg(feature = "test-fault-injection")]
fn scaling_counts(metrics: &Metrics) -> (u64, u64, u64) {
    let c = metrics.scaling_counters();
    (
        c.subscriptions_admitted(),
        c.catchups_pending(),
        c.catchups_completed(),
    )
}

/// A catch-up that re-bootstraps before its first ack is still ONE first
/// catch-up. An in-window reconnect is counted pending when it starts;
/// parked under its slot before its atomic read, the layer is overwritten and
/// cleaned up, so its read answers `NotFound` and it re-bootstraps through
/// `catch_up_brand_new`, which counts a first catch-up too — a no-op here.
/// Once the NEW object's state and its ack arrive: one admitted, none
/// pending, one completed.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_that_rebootstraps_before_its_first_ack_completes_one_first_catch_up() {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    let mut svc = build_service(&fx, 4096);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_coalesced());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let layer = "scaling-rebootstrap-in-catch-up";
    let (session_id, old_object_id) = create_layer(&fx, &svc, layer).await;
    for section_id in 91..=93 {
        commit(&svc, &session_id, layer, old_object_id, section_id).await;
    }

    let mut stream = subscribe(&svc, &session_id, layer, old_object_id, 1, "scaling-a").await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_coalesced())
        .await
        .expect("the reconnect parks under its slot, before its atomic read");
    assert_eq!(
        scaling_counts(&fx.metrics),
        (1, 1, 0),
        "the in-window catch-up is pending from its start"
    );

    let new_object_id = overwrite(&svc, &session_id, layer).await;
    svc.layer_store.wait_for_cleanup(SETTLE).await;
    tokio::time::timeout(SETTLE, seam.arrive_at_pre_coalesced())
        .await
        .expect("release the reconnect");

    let (contents, ack) = recv_catch_up(&mut stream).await;
    assert_eq!(
        contents
            .iter()
            .map(|h| (h.object_id, h.sequence_num))
            .collect::<Vec<_>>(),
        vec![(new_object_id, 1)],
        "fixture self-check: the reconnect re-bootstrapped onto the NEW object before any content"
    );
    assert_eq!((ack.object_id, ack.sequence_num), (new_object_id, 1));
    await_condition(
        || scaling_counts(&fx.metrics) == (1, 0, 1),
        "the re-bootstrapped catch-up completes once and leaves nothing pending",
    )
    .await;
}

/// A re-bootstrap after the first ack is not a first catch-up: a caught-up
/// subscription whose layer is overwritten re-bootstraps in steady state
/// onto the NEW object's full state and its ack, and the counts stay at one
/// completed and none pending.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_steady_state_rebootstrap_after_the_first_ack_is_not_counted() {
    let fx = CapacityFixture::setup(1, PARKED_BUDGET_MS).await;
    let svc = build_service(&fx, 4096);
    let layer = "scaling-rebootstrap-steady";
    let (session_id, old_object_id) = create_layer(&fx, &svc, layer).await;
    commit(&svc, &session_id, layer, old_object_id, 95).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0, "scaling-b").await;
    let (_, ack) = recv_catch_up(&mut stream).await;
    assert_eq!((ack.object_id, ack.sequence_num), (old_object_id, 2));
    await_condition(
        || scaling_counts(&fx.metrics) == (1, 0, 1),
        "the first catch-up completes",
    )
    .await;

    let new_object_id = overwrite(&svc, &session_id, layer).await;
    let (contents, ack) = recv_catch_up(&mut stream).await;
    assert_eq!(
        contents
            .iter()
            .map(|h| (h.object_id, h.sequence_num))
            .collect::<Vec<_>>(),
        vec![(new_object_id, 1)],
        "fixture self-check: the steady-state re-bootstrap serves the NEW object's state"
    );
    assert_eq!((ack.object_id, ack.sequence_num), (new_object_id, 1));
    // A live commit after the re-bootstrap's ack: everything the
    // re-bootstrap did has run by the time it arrives.
    commit(&svc, &session_id, layer, new_object_id, 96).await;
    let (live, _) = recv_message(&mut stream, "the first live commit").await;
    assert_eq!((live.object_id, live.sequence_num), (new_object_id, 2));
    assert_eq!(
        scaling_counts(&fx.metrics),
        (1, 0, 1),
        "the steady-state re-bootstrap is neither pending nor a second completion"
    );
}
