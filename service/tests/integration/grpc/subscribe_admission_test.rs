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

// b.f4f — bounded subscription admission, cancellation-aware delivery, and
// the reserved terminal-trailer slot, driven end-to-end through
// `handle_subscribe_read_layer`.
//
// Three properties the pre-b.f4f service did not have, and which this file is
// organised around:
//
//   1. **Admission is bounded and happens before any work.** An RPC that cannot
//      get an admission slot (the subscription ceiling) is refused
//      with a Trailers-Only
//      `UNAVAILABLE` — no response message precedes it and no background task is
//      spawned behind it. Before the fix the RPC parked on `ExclusivePool::
//      acquire` indefinitely, after having already spawned its catch-up.
//
//   2. **A stalled reader is torn down, and the tear-down is delivered.** The
//      outer `tx.send` now races server shutdown, session deletion,
//      `tx.closed()` and `OUTER_SEND_DEADLINE`. The terminal status goes into a
//      channel slot reserved before the subscription sent anything, so a
//      server-driven tear-down can never degrade into a bare EOS — which the
//      Kit client reads as gRPC `OK` and does not retry
//      (its retry policy retries only DEADLINE_EXCEEDED, UNAVAILABLE,
//      RESOURCE_EXHAUSTED and INTERNAL).
//
//   3. **The clean ending is exactly one case.** `client_disconnect` sends no
//      trailer because there is nobody to send it to. Every other ending —
//      slow-reader deadline, inner watchdog, shutdown (`UNAVAILABLE`), session
//      deletion (`NOT_FOUND`), a session found missing from the live store
//      (`UNAVAILABLE`), any backend failure — delivers its status, in every
//      phase: admission, catch-up, delivery and the idle wait (b.e1h).
//
// ── Wait budget ──────────────────────────────────────────────────────────────
//
// Every wait below is a **deadline, not a measurement**: the healthy path
// resolves in milliseconds and pays none of them. Worst case per test, summed at
// full budget, against nextest's 120 s hang-killer:
//
//   * `SETTLE_DEADLINE` (5 s) — one per catch-up message and per polled
//     condition. Bounded by construction: a catch-up here is one content header,
//     one or two body chunks, and the caught-up ack.
//   * `DRAIN_DEADLINE` (20 s) — ONE budget for a whole steady-state drain,
//     however long the backlog is. These tests deliberately fill all 32 delivery
//     slots, so a per-message deadline would multiply by 33 and put the worst
//     case past the hang-killer — where a SIGKILLed test prints nothing and
//     every diagnostic panic below is lost at exactly the moment a real failure
//     needs reading.
//   * `TEST_OUTER_SEND_DEADLINE` (300 ms) — the shrunk `OUTER_SEND_DEADLINE`.
//   * `TEST_ADMISSION_WAIT_MS` (300 ms) — the shrunk admission wait.
//   * `MID_DELIVERY_SEND_DEADLINE` (20 s) — an `OUTER_SEND_DEADLINE` override
//     that is RAISED rather than shrunk, local to the three mid-delivery tests
//     (`cancellation_mid_delivery`) and the three in-window replay tests
//     (`cancellation_mid_in_window_replay`). Payable in full: the exit those
//     tests assert on is racing it.
//   * `STALL_SEND_DEADLINE` (30 s) — the other raised override, local to
//     `a_stalled_live_send_is_counted_at_once_and_its_wait_recorded_when_the_reader_resumes`,
//     which keeps a send parked while it checks the stall counter. NOT
//     payable: nothing in that test awaits the deadline, and the resumed
//     drain that ends the stall is bounded by `DRAIN_DEADLINE`.
//   * `STALL_HELD` (1.5 s) — the one sleep, in the same test: a lower bound on
//     how long its send stays parked, paid in full.
//   * `PARKED_WAIT_MS` (60 s) — the raised admission wait local to
//     `a_client_that_disconnects_while_queued_for_admission_leaves_no_waiting_work`
//     and to the two queued-refusal logging tests (`queued_refusal_logs`).
//     Inventoried here because a per-test constant is exactly what goes
//     unnoticed in this arithmetic, but NOT payable: in each test the only
//     await on the waiter it governs is a `timeout(SETTLE_DEADLINE, …)` join
//     after the abort or the cancel, so each spends at most 5 s on it. The
//     60 s exists solely so the waiter cannot refuse itself before the event
//     under test lands.
//   * `PARKED_CATCH_UP_BUDGET_MS` (60 s) — the capacity budget of the
//     concurrent-catch-up tests, NOT payable for the same reason: it bounds
//     the slot wait of a catch-up queued behind a parked one, and every await
//     on that catch-up is itself bounded by `SETTLE_DEADLINE`.
//
// The heaviest tests are the three mid-delivery ones
// (`shutdown_mid_delivery_ends_the_stream_unavailable` and its deletion and
// session-absence twins): one catch-up drain (≤ 4 messages at
// `SETTLE_DEADLINE`) 20 s + one condition poll 5 s + one `next_ok` 5 s +
// `MID_DELIVERY_SEND_DEADLINE` 20 s + one full drain (`DRAIN_DEADLINE`) 20 s +
// four condition polls 20 s ≈ 90 s at every budget's full value; the
// shutdown and session-absence ones — the endings a client retries — add
// their retry's catch-up, under one more `DRAIN_DEADLINE` (20 s), for ≈ 110 s
// — inside nextest's 120 s hang-killer, where a SIGKILL would print nothing.
// Next heaviest, both ≈ 65 s,
// are `a_reconnect_after_a_slow_reader_teardown_converges_with_no_lost_commits`
// (two catch-up drains + one full drain + one condition poll) and
// `a_client_that_disconnects_while_queued_for_admission_leaves_no_waiting_work`
// (two catch-up drains + four condition polls + one bounded join); then the
// three in-window replay tests, ≈ 60 s (one `next_ok` 5 s +
// `MID_DELIVERY_SEND_DEADLINE` 20 s + one full drain 20 s + three condition
// polls 15 s). The
// rendezvous-driven tests — the three mid-catch-up ones (whose two extra
// release polls, the catch-up slot and the admitted gauge, add 10 s) and the
// tracker-drain one, which adds its `SUBSCRIPTION_DRAIN_BUDGET` (5 s), two
// more condition polls (its reader's busy gauge and its reader's read, 10 s)
// and a pool quit the client bounds itself (a few seconds at most) — sit at
// ≈ 55–80 s, and their seam
// rendezvous is itself
// wrapped in `SETTLE_DEADLINE` rather than left bare — a barrier whose other
// party never arrives would otherwise be bounded only by the hang-killer,
// which prints nothing. No assertion here is about speed: the failure the
// tracker-drain test's pool quit rules out — a `QUIT` queued behind a blocked
// `XREAD` — overruns the 2 s `QUIT_TIMEOUT` and logs the reader pool's
// quit-timeout WARN, so the test asserts that WARN's absence, not the quit's
// duration.
//
// ── A second, silent activation condition ────────────────────────────────────
//
// Several tests here drive a test seam — `outer_send_deadline_for_test` (the
// slow-reader tests), the stream multiplexer's fault queue (the backend-failure
// test), `catch_up_chunk_seam_for_test` (the mid-catch-up cancellation) and
// `catch_up_state_read_seam_for_test` (the concurrent-catch-up tests, which
// park a catch-up holding its slot) — and all four exist only under the
// **`test-fault-injection`** cargo
// feature. Without it the fields are compiled out, `build_admission_service`
// silently ignores whatever [`Seams`] carried, and each of those tests would run
// against production behaviour instead: the slow-reader ones would race this
// file's own 5 s `SETTLE_DEADLINE` against the production 5 s deadline, the
// backend-failure one would inject nothing, and the catch-up ones would wait on a
// rendezvous nobody ever reaches. All of them are therefore
// `#[cfg(feature = "test-fault-injection")]` rather than left to run degraded: a
// missing feature compiles them **out**, so the run reports fewer tests instead
// of reporting false confidence. Run with:
//
// ```text
// cargo nextest run -p live-edit --test integration \
//     --features test-support,test-fault-injection grpc::subscribe_admission_test
// ```

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::KeysInterface;
use futures::{Stream, StreamExt};
use tokio::time::Instant;
use tonic::{Code, Request, Status};

use live_edit::config::Config;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::subscriber_delivery::LAGGING_LOG_THRESHOLD;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::{
    test_util::{render, sample_value},
    Metrics, ADMISSION_RESULTS, ADMISSION_RESULT_ADMITTED, ADMISSION_RESULT_CANCELLED,
    ADMISSION_RESULT_REFUSED_SATURATED, ADMISSION_RESULT_REFUSED_WAITER_CAP,
    CONSUMER_FORCED_RECONNECT_PHASES, CONSUMER_PERMANENT_EXIT_PHASES, CONSUMER_PHASE_KEEPALIVE,
    CONSUMER_PHASE_PERMANENT, CONSUMER_PHASE_PROBE, CONSUMER_PHASE_WAKE_ACK, CONSUMER_PHASE_XREAD,
    HEAD_CHECK_RESULTS, HEAD_CHECK_RESULT_ERROR, HEAD_CHECK_RESULT_GAP, HEAD_CHECK_RESULT_NO_GAP,
    KEEPALIVE_RESULTS, KEEPALIVE_RESULT_FAILED, KEEPALIVE_RESULT_OK, OVERFLOW_BOUNDS,
    OVERFLOW_BOUND_BYTES, OVERFLOW_BOUND_ENTRIES, REAUTH_RESULTS, REAUTH_RESULT_FAILED,
    REAUTH_RESULT_OK, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_ERROR,
    STREAM_EXIT_FENCED_READ_WAIT_CAP, STREAM_EXIT_REASONS, STREAM_EXIT_SESSION_ABSENT,
    STREAM_EXIT_SESSION_CANCEL, STREAM_EXIT_SHUTDOWN, STREAM_EXIT_SLOW_READER_DEADLINE,
    STREAM_EXIT_STREAM_GAP, WAKE_OUTCOMES, WAKE_OUTCOME_ERROR, WAKE_OUTCOME_OK,
};
#[cfg(feature = "test-fault-injection")]
use live_edit::observability::metrics::{
    DELIVERY_LAG_STAGE_ENQUEUED, SEND_PHASE_CATCHUP, SEND_PHASE_LIVE,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::refusal_detail::{
    assert_no_refusal_detail, assert_refusal_detail, assert_shutdown_status,
};
use live_edit::error::{refusal_metadata, refusal_reason, CAPACITY_RETRY_DELAY};
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use super::helpers::*;

// ── Budgets and shared constants ────────────────────────────────────────────

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE_DEADLINE: Duration = Duration::from_secs(5);

/// Shrunk [`live_edit::storage::notifications::OUTER_SEND_DEADLINE`], installed
/// via `LiveEditService::outer_send_deadline_for_test`. The production constant
/// is 5 s and every slow-reader test would otherwise spend it.
///
/// The override field — and therefore every test that depends on it — exists
/// only under `test-fault-injection`. See the module header on why those tests
/// are `cfg`-gated rather than left to run with the production constant.
#[cfg(feature = "test-fault-injection")]
const TEST_OUTER_SEND_DEADLINE: Duration = Duration::from_millis(300);

/// Shrunk `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`.
const TEST_ADMISSION_WAIT_MS: u64 = 300;

/// The capacity budget of the concurrent-catch-up tests, whose queued
/// catch-ups must stay queued until the test releases or ends them: the budget
/// bounds the catch-up slot wait too, and a shrunk one would refuse the queued
/// catch-up `catchup_capacity` before the event under test lands. Not payable:
/// every wait on such a catch-up is itself bounded by `SETTLE_DEADLINE`. Gated
/// with its users, which park a catch-up at a seam (module header).
#[cfg(feature = "test-fault-injection")]
const PARKED_CATCH_UP_BUDGET_MS: u64 = 60_000;

/// Delivery capacity of the `SubscribeReadLayer` response channel.
///
/// The channel is built with **33**: 32 for data plus one reserved with an
/// `OwnedPermit` for the terminal trailer before any data is sent. The tests
/// that count buffered messages assert against this number, never 33 — a data
/// message landing in the 33rd slot is the starvation this reservation exists
/// to prevent.
/// Gated with its users: every test that counts delivery slots drives the
/// `outer_send_deadline_for_test` seam (module header).
#[cfg(feature = "test-fault-injection")]
const OUTER_CHANNEL_DELIVERY_SLOTS: usize = 32;

/// Steady-state commits needed to fill those 32 slots.
///
/// **Two messages per relayed commit**, so sixteen commits fill the channel and
/// the seventeenth parks. The relay emits a content header followed by one body
/// chunk (`chunk_ompf_for_stream`, `src/grpc/streaming.rs:65`), and these test
/// deltas are far below `OMPF_CHUNK_SIZE`, so the body is always a single chunk.
///
/// **Measured, not derived from the code.** A probe that wrote one commit at a
/// time and waited out the deadline after each reported the tear-down firing on
/// commit #17 — i.e. sixteen commits' worth of messages fit exactly. Reading the
/// ratio off the source instead would be a second, unverified derivation of the
/// same number; `a_full_response_channel_alone_does_not_tear_the_subscription_down`
/// re-checks it from both sides on every run.
///
/// Only that test uses it, and it is gated on `test-fault-injection` (module
/// header), so without the feature this would be a dead constant and
/// `-D warnings` would refuse the build.
#[cfg(feature = "test-fault-injection")]
const COMMITS_TO_FILL_THE_CHANNEL: usize = OUTER_CHANNEL_DELIVERY_SLOTS / 2;

/// One budget for a whole drain, however many messages the backlog holds. See
/// [`drain_to_end`] on why this is not a per-message deadline.
/// Gated with [`drain_to_end`], its only user.
#[cfg(feature = "test-fault-injection")]
const DRAIN_DEADLINE: Duration = Duration::from_secs(20);

const TEST_SERVICE: &str = "test";

// ── Fixture ─────────────────────────────────────────────────────────────────

/// Per-test Valkey fixture with an isolated key prefix and a caller-chosen
/// subscription cap and admission policy.
///
/// `TestValkey` uses the production subscription cap (4096) and a 2 s
/// capacity budget, which is right for every other suite and useless here:
/// these tests need a cap of one or two so "saturated" is reachable, and a
/// sub-second wait so a refusal does not cost 2 s each time. Two stream
/// readers serve every subscription.
///
/// **One `Metrics` handle for the whole fixture**, and that is load-bearing.
/// The b.f4f series are emitted from two different owners: `admission_total`,
/// `valkey_pool_waiters` and `valkey_pool_in_use` come from the admission gate
/// and the stream readers inside `ValkeyClient` (so they only exist once
/// `install_observability_hooks` has stashed a handle there — without that call
/// every `PoolUseToken` is a no-op and the gauge stays flat at zero), while
/// `subscriber_stream_exits_total` is emitted by the gRPC handler off
/// `svc.metrics`. Installing the same handle in both places is what lets one
/// `render()` see all of them.
struct AdmissionFixture {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
    metrics: Arc<Metrics>,
    /// Set by [`Self::purge_keys_now`], after which `Drop` has nothing to do —
    /// and must not try: a test that quits the client afterwards leaves a pool
    /// on which a cleanup `SCAN` would never complete.
    purged: bool,
}

impl AdmissionFixture {
    /// The common shape: the shrunk [`TEST_ADMISSION_WAIT_MS`], so a refusal
    /// costs milliseconds rather than the production 20 s.
    async fn setup(max_subscriptions: usize, max_waiters: Option<usize>) -> Self {
        Self::setup_with_wait(max_subscriptions, max_waiters, TEST_ADMISSION_WAIT_MS).await
    }

    /// As [`Self::setup`], with the admission wait chosen by the caller.
    ///
    /// Exists for the one test that needs a waiter to stay parked rather than
    /// time out: a short wait would let the waiter refuse itself as
    /// `refused_saturated` before the test could abort it, and the abort is the
    /// thing under test. A separate constructor rather than a third positional
    /// argument on `setup`, so the twenty-odd existing call sites read unchanged.
    async fn setup_with_wait(
        max_subscriptions: usize,
        max_waiters: Option<usize>,
        admission_wait_ms: u64,
    ) -> Self {
        Self::setup_full(max_subscriptions, max_waiters, admission_wait_ms, 64).await
    }

    /// Unlimited subscriptions, at most `max_catchups` catch-ups at once, and a
    /// capacity budget of `budget_ms`.
    ///
    /// The budget is the caller's because it now bounds a catch-up's slot wait
    /// too: a test whose catch-up must stay queued until an event it fires uses
    /// [`PARKED_CATCH_UP_BUDGET_MS`], so the wait cannot refuse itself
    /// `catchup_capacity` first.
    async fn setup_with_catchups(max_catchups: usize, budget_ms: u64) -> Self {
        Self::setup_full(0, None, budget_ms, max_catchups).await
    }

    async fn setup_full(
        max_subscriptions: usize,
        max_waiters: Option<usize>,
        admission_wait_ms: u64,
        max_catchups: usize,
    ) -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let key_schema = KeySchema::new(&prefix);

        let config = Config {
            subscriber_pool_size: 2,
            subscriber_admission_wait_ms: admission_wait_ms,
            subscriber_max_subscriptions: max_subscriptions,
            subscriber_max_concurrent_catchups: max_catchups,
            subscriber_admission_max_waiters: max_waiters,
            // 30 s matches `TestValkey`'s post-b.1kf default: the workspace
            // suite's SYN-burst recovery profile needs that budget to drain
            // reliably under listen-queue pressure.
            valkey_connect_timeout_secs: 30,
            ..crate::common::test_config()
        };

        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("AdmissionFixture: connect must succeed");

        // Must happen before any subscription exists: it is what makes the
        // gate's and readers' metrics (gauge, waiters, admission counters)
        // non-no-op.
        let metrics = Arc::new(crate::common::metrics_scrape_only(TEST_SERVICE));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        Self {
            client,
            key_schema,
            prefix,
            metrics,
            purged: false,
        }
    }

    /// Delete the fixture's keys now rather than on `Drop`, for a test that
    /// goes on to quit the client itself.
    #[cfg(feature = "test-fault-injection")]
    async fn purge_keys_now(&mut self) {
        purge_prefix(&self.client, &self.prefix).await;
        self.purged = true;
    }

    fn client(&self) -> &ValkeyClient {
        &self.client
    }

    fn key_schema(&self) -> &KeySchema {
        &self.key_schema
    }

    fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }
}

/// Delete every key under `prefix` through `client`'s data pool.
async fn purge_prefix(client: &ValkeyClient, prefix: &str) {
    let pool = &client.pool;
    let pattern = format!("{prefix}*");
    let iter_client = pool.next();
    let mut stream = iter_client.scan_buffered(&pattern, Some(100), None);
    let mut batch = Vec::with_capacity(100);
    while let Some(result) = stream.next().await {
        match result {
            Ok(key) => {
                batch.push(key);
                if batch.len() >= 100 {
                    let _ = pool.unlink::<(), _>(std::mem::take(&mut batch)).await;
                }
            }
            Err(_) => break,
        }
    }
    if !batch.is_empty() {
        let _ = pool.unlink::<(), _>(batch).await;
    }
}

impl Drop for AdmissionFixture {
    fn drop(&mut self) {
        if self.purged {
            return;
        }
        let client = self.client.clone();
        let prefix = self.prefix.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(purge_prefix(&client, &prefix));
        });
    }
}

/// The test-only seams a fixture may install on its `LiveEditService`.
///
/// A struct rather than a growing parameter list, because every b.f4f seam so far
/// has arrived as "one more `Option` nobody but two tests set": with
/// `..Default::default()` at the call sites, the next one costs one line here and
/// one at the builder, and touches no existing test. `Seams::default()` is "no
/// seams installed", which is what all but four tests in this file want.
///
/// Every field is `#[cfg(feature = "test-fault-injection")]` because the
/// `LiveEditService` fields they feed are — see the module header on why the
/// tests that set them are `cfg`-gated too rather than left to run degraded.
#[derive(Default)]
struct Seams {
    /// Shrinks `OUTER_SEND_DEADLINE`. `None` keeps the production 5 s.
    #[cfg(feature = "test-fault-injection")]
    outer_send_deadline: Option<Duration>,
    /// Parks `deliver_catch_up_content`'s chunk loop after its first chunk. A
    /// catch-up parked there has released its catch-up slot.
    #[cfg(feature = "test-fault-injection")]
    catch_up_chunk: Option<Arc<live_edit::grpc::subscribe::CatchUpChunkSeam>>,
    /// Parks a catch-up at one of its state-read points, where it holds its
    /// catch-up slot.
    #[cfg(feature = "test-fault-injection")]
    catch_up_state_read: Option<Arc<live_edit::grpc::subscribe::CatchUpStateReadSeam>>,
}

/// Build a service over `fx`, returning the fixture's shared `Metrics` handle so
/// the test can read the b.f4f series (`build_service` keeps its own private
/// handle, which is why these tests do not use it).
fn build_admission_service(
    fx: &AdmissionFixture,
    permissions: Arc<dyn live_edit::auth::permissions::PermissionsClient>,
    seams: Seams,
) -> (LiveEditService, Arc<Metrics>) {
    let session_store = Arc::new(SessionStore::new(
        fx.client().clone(),
        fx.key_schema().clone(),
    ));
    let layer_store = Arc::new(LayerStore::new(
        fx.client().clone(),
        fx.key_schema().clone(),
        4096,
    ));
    // The fixture's handle, not a fresh one: the pool's own series are emitted
    // against the handle installed on the `ValkeyClient`, so a service-private
    // `Metrics` would render a permanently-zero gauge.
    let metrics = Arc::clone(fx.metrics());
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );

    let mut session_recovery_config = crate::common::test_config();
    session_recovery_config.checkpoint_enabled = false;
    let session_recovery_store: Arc<dyn live_edit::checkpoint::CheckpointStore> =
        Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new());
    let session_recovery = Arc::new(live_edit::checkpoint::SessionRecoveryHook::new(
        session_recovery_store,
        fx.client().clone(),
        fx.key_schema().clone(),
        Arc::new(session_recovery_config),
        Arc::clone(&metrics),
    ));

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: permissions,
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            fx.client().clone(),
            fx.key_schema().clone(),
        ),
        valkey: fx.client().clone(),
        key_schema: fx.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            fx.client().clone(),
            fx.key_schema().clone(),
            Arc::clone(&metrics),
        ),
        metrics: Arc::clone(&metrics),
        service_label: TEST_SERVICE.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        service_token_provider: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_xrange_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        outer_send_deadline_for_test: seams.outer_send_deadline,
        #[cfg(feature = "test-fault-injection")]
        catch_up_chunk_seam_for_test: seams.catch_up_chunk,
        #[cfg(feature = "test-fault-injection")]
        run_catch_up_inner_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_out_of_window_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_state_read_seam_for_test: seams.catch_up_state_read,
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
    };
    // Without `test-fault-injection` every field `Seams` carries is compiled out
    // and the argument is genuinely unused. Consuming it here keeps one signature
    // across both feature shapes, so every caller reads the same; the tests that
    // actually depend on a seam are `cfg`-gated instead (module header).
    let _ = seams;
    (svc, metrics)
}

// ── Request builders ────────────────────────────────────────────────────────

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn incremental_diff_change_section(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    }
}

async fn create_session_and_layer_for(
    fx: &AdmissionFixture,
    svc: &LiveEditService,
    session_name: &str,
    layer_name: &str,
) -> (String, u64) {
    let session_store = SessionStore::new(fx.client().clone(), fx.key_schema().clone());
    let session_id = session_store
        .create(session_name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let ompf_bytes = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let response = handle_create_layer(svc, make_request(stream))
        .await
        .expect("create_layer must succeed for test setup");
    (session_id, response.into_inner().object_id)
}

/// Apply one commit that produces exactly one relayed steady-state message.
async fn apply_unique_update(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    section_id: u64,
) {
    let ompf_bytes = delta_to_ompf_bytes(&incremental_diff_change_section(section_id), &[]);
    let stream = streaming_update(vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: layer_name.to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: section_id * 10,
                    client_id: "writer".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk { data: ompf_bytes },
            )),
        },
    ]);
    handle_update_layer(svc, make_request(stream))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section_id={section_id}) failed: {e}"));
}

/// Await the next message, requiring it to be present and `Ok`.
async fn next_ok<S>(stream: &mut S, what: &str) -> SubscribeReadLayerResponse
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE_DEADLINE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE_DEADLINE:?}"))
        .unwrap_or_else(|| panic!("the stream ended before {what}"))
        .unwrap_or_else(|e| panic!("{what} was an error status: {e:?}"))
}

/// Drain the whole catch-up phase — every content header with its body chunks,
/// and then the caught-up ack — leaving the subscription parked in steady state
/// with an empty response channel.
///
/// The ack is the last message of catch-up and is the only header carrying
/// `total_size == 0` (`send_ack_header`), so it is an unambiguous terminator
/// across all four regimes: `up_to_date` sends it alone, the three
/// content-serving regimes send it after their payload. **Draining the payload
/// but not the ack would leave one message in the channel**, which silently
/// shifts every "the client is not reading, so the channel fills at N" count in
/// this file by one and lets a leftover ack masquerade as a relayed update.
///
/// Returns the caught-up ack, whose `sequence_num` is the version the server
/// just told this subscriber it is at — the value a reconnect would carry back
/// as `have_version`.
async fn drain_catch_up<S>(stream: &mut S) -> live_edit::grpc::proto::SubscribeReadLayerHeader
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    loop {
        let msg = next_ok(stream, "a catch-up message").await;
        let header = match msg.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
            other => panic!("expected a catch-up Header, got: {other:?}"),
        };
        if header.total_size == 0 {
            // The caught-up ack: catch-up is over.
            return header;
        }
        let mut seen: u64 = 0;
        while seen < header.total_size {
            let next = next_ok(stream, "a catch-up body chunk").await;
            match next.response_type {
                Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                    seen += c.data.len() as u64
                }
                other => panic!("expected a Chunk while filling the catch-up body, got: {other:?}"),
            }
        }
    }
}

// ── Metric readers ──────────────────────────────────────────────────────────

/// `live_edit_valkey_pool_in_use{pool}`. Strict for the same reason as
/// [`read_admission`]: this file's most common assertion is that the gauge
/// returned to **zero** after a tear-down, and a defaulted zero on a mistyped
/// pool label would satisfy every one of them while measuring nothing.
fn read_pool_in_use(metrics: &Metrics, pool: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_in_use",
        &[("pool", pool)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_valkey_pool_in_use{{pool=\"{pool}\"}} is absent from the scrape. The \
             pool label set is pre-initialised at zero in `Metrics::new`, so this is a \
             misspelled label rather than a pool that has not been used yet"
        )
    }) as u64
}

/// `live_edit_valkey_pool_waiters{pool}` — the EXPORTED gauge, as distinct from
/// the `waiters()` atomic the admission cap enforces against.
///
/// Strict for the same reason as [`read_admission`]: the label set is
/// pre-initialised, so an absent series is a misspelled label rather than a
/// queue that has never formed, and a defaulted zero would satisfy exactly the
/// `== 0` assertions this file uses it for.
fn read_pool_waiters_gauge(metrics: &Metrics, pool: &str) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_waiters",
        &[("pool", pool)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_valkey_pool_waiters{{pool=\"{pool}\"}} is absent from the scrape. The \
             pool label set is pre-initialised at zero in `Metrics::new`, so this is a \
             misspelled label rather than a queue that has not formed yet"
        )
    }) as i64
}

/// `live_edit_subscriber_admission_total{result}`.
///
/// **`expect`, not `unwrap_or(0.0)`.** Every value of both vocabularies is
/// pre-initialised in `Metrics::new` — which
/// `admission_and_stream_exit_label_values_export_at_zero_before_any_event`
/// asserts directly — so an absent series means the label was misspelled or its
/// constant drifted, not that the event has yet to happen. Defaulting to zero
/// there is precisely what would let an `== 0` assertion pass while measuring
/// nothing, so the callers below pass the `ADMISSION_RESULT_*` /
/// `STREAM_EXIT_*` constants rather than raw strings AND the lookup fails loudly
/// if one of them stops resolving.
fn read_admission(metrics: &Metrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_admission_total",
        &[("result", result)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_subscriber_admission_total{{result=\"{result}\"}} is absent from the \
             scrape. Every `result` value is pre-initialised at zero in `Metrics::new`, so this \
             is a misspelled or drifted label, not an event that has not happened yet — and an \
             `== 0` assertion on it would have passed vacuously"
        )
    }) as u64
}

/// `live_edit_subscriber_stream_exits_total{reason}`. See [`read_admission`] on
/// why this fails loudly on an absent series instead of reading zero.
fn read_stream_exit(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_subscriber_stream_exits_total{{reason=\"{reason}\"}} is absent from the \
             scrape. Every `reason` value is pre-initialised at zero in `Metrics::new`, so this \
             is a misspelled or drifted label, not an exit that has not happened yet"
        )
    }) as u64
}

/// `live_edit_subscriber_admitted`: the subscriptions admitted right now. See
/// [`read_admission`] on why this fails loudly on an absent series.
fn read_admitted(metrics: &Metrics) -> i64 {
    sample_value(&render(metrics), "live_edit_subscriber_admitted", &[]).unwrap_or_else(|| {
        panic!(
            "live_edit_subscriber_admitted is absent from the scrape; it is pre-initialised at \
             zero in `Metrics::new`"
        )
    }) as i64
}

/// Poll `condition` until it holds or `SETTLE_DEADLINE` elapses.
///
/// A bounded poll on observable state rather than a settle sleep, per the test
/// guide's synchronisation rules. The sleep between probes is the probe
/// interval, not a coordination primitive.
async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = Instant::now() + SETTLE_DEADLINE;
    loop {
        if condition() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what} did not hold within {SETTLE_DEADLINE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// One drained subscription: every `Ok` message in arrival order, the terminal
/// status if there was one, and how many messages arrived after it.
/// Gated with [`drain_to_end`], which is its only constructor.
#[cfg(feature = "test-fault-injection")]
struct Drained {
    messages: Vec<SubscribeReadLayerResponse>,
    terminal: Option<Status>,
    after_terminal: usize,
}

#[cfg(feature = "test-fault-injection")]
impl Drained {
    /// The `sequence_num` of every message that carried a header, in arrival
    /// order.
    ///
    /// (The whole `impl` is gated with [`Drained`]; see there.)
    fn header_sequences(&self) -> Vec<u64> {
        self.messages
            .iter()
            .filter_map(|msg| match &msg.response_type {
                Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
                    Some(h.sequence_num)
                }
                _ => None,
            })
            .collect()
    }
}

/// Drain a subscription to the end of its stream under **one** deadline.
///
/// A per-message `timeout` would multiply by the backlog: these tests
/// deliberately fill all 32 delivery slots, so 33 messages at `SETTLE_DEADLINE`
/// each has a worst case of ~165 s against nextest's 120 s hang-killer — and a
/// SIGKILLed test prints nothing, losing every diagnostic panic below at exactly
/// the moment a real failure needs reading. One deadline over the whole drain
/// keeps the worst case equal to the budget whatever the backlog turns out to be
/// (test-writing guide, § Valkey Integration Tests).
/// Gated with its users: every test that drains a torn-down subscription to
/// its end is seam-driven (module header).
#[cfg(feature = "test-fault-injection")]
async fn drain_to_end<S>(stream: &mut S, what: &str) -> Drained
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let mut drained = Drained {
        messages: Vec::new(),
        terminal: None,
        after_terminal: 0,
    };
    tokio::time::timeout(DRAIN_DEADLINE, async {
        while let Some(item) = stream.next().await {
            match item {
                Ok(msg) => {
                    if drained.terminal.is_some() {
                        drained.after_terminal += 1;
                    }
                    drained.messages.push(msg);
                }
                Err(status) => drained.terminal = Some(status),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what} did not finish draining within {DRAIN_DEADLINE:?}"));
    drained
}

// ═══════════════════════════════════════════════════════════════════════════
// Admission
// ═══════════════════════════════════════════════════════════════════════════

/// A refusal is **Trailers-Only**: `Err(Status)` out of the handler, so tonic
/// emits the status with no response message ahead of it, and nothing was
/// spawned behind it.
///
/// Asserting only "the status is UNAVAILABLE" would have passed on the shape
/// b.f4f replaced, where the handler returned `Ok(stream)` and the spawned task
/// discovered the problem afterwards — the client would then have seen a
/// response head, possibly a message, and only then the failure. The two
/// discriminating halves are therefore: the handler returned `Err` (so there is
/// no stream to hold a message at all), and `valkey_pool_in_use` never rose
/// above the subscription that legitimately holds the slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admission_refusal_is_trailers_only_with_no_task_spawned() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());

    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-admit-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;

    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;
    let admitted_before = read_admission(&metrics, ADMISSION_RESULT_ADMITTED);

    // The cap is one and it is taken.
    let (other_session, other_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-admit-refused", "layer-refused").await;
    let refused = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &other_session,
            "layer-refused",
            other_object,
            0,
            "client-refused",
        ),
    )
    .await;

    let status = match refused {
        Err(status) => status,
        Ok(_) => panic!(
            "a second SubscribeReadLayer was ADMITTED onto a cap-of-one pod whose slot is held. \
             Either admission is not bounded by the pool size, or the handler returned a stream \
             and deferred the failure into the spawned task — which is the post-headers shape \
             b.f4f replaced"
        ),
    };
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a saturated pod must refuse retryably: the Kit client retries UNAVAILABLE and does not \
         retry OK or INVALID_ARGUMENT. Got: {status:?}"
    );
    assert_refusal_detail(
        &status,
        refusal_reason::SUBSCRIPTION_CAPACITY,
        &[(refusal_metadata::CAUSE, refusal_metadata::SATURATED)],
        CAPACITY_RETRY_DELAY,
    );

    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        1,
        "the refusal must be recorded as `refused_saturated` — the series operators alert a \
         pod-capacity page on"
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_ADMITTED),
        admitted_before,
        "a refused request must not also count as admitted"
    );
    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        1,
        "the refused request must not have spawned a task that took a slot: the gauge must still \
         show only the one established subscription"
    );
    assert_eq!(
        svc.shutdown.enrolled(),
        1,
        "the refused request enrolled before its admission and must have dropped that \
         enrollment with its refusal: only the established subscription stays enrolled"
    );

    drop(held);
}

/// Pod saturation is not observable to a caller who is not allowed to read the
/// layer: admission sits *after* `check_read_permission`.
///
/// Discriminating because the natural place to put a cheap capacity check is at
/// the top of the handler, and doing so turns per-pod occupancy into an
/// unauthenticated side channel.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unauthorized_caller_against_a_saturated_pod_sees_the_permission_failure() {
    let fx = AdmissionFixture::setup(1, None).await;
    // One service to establish the holding subscription (auth allowed)…
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-authz-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // …and a second service over the SAME pool whose permission client denies.
    let (denying, denying_metrics) =
        build_admission_service(&fx, Arc::new(DenyAllPermissions), Seams::default());
    let status = handle_subscribe_read_layer(
        &denying,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-denied"),
    )
    .await;
    let status = expect_err_status(status, "a denied caller was nevertheless admitted");

    assert_eq!(
        status.code(),
        Code::PermissionDenied,
        "an unauthorized caller must get the permission failure. UNAVAILABLE here would tell an \
         unauthenticated caller how full this pod is, and would send the Kit client into its \
         retry loop against a layer it will never be allowed to read. Got: {status:?}"
    );
    assert_eq!(
        read_admission(&denying_metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        0,
        "admission must not even have been attempted for a caller the permission check rejected"
    );

    drop(held);
}

/// At the waiter cap the refusal is `refused_waiter_cap`, not
/// `refused_saturated`, and it is immediate.
///
/// The two are different operational events — the pool is full vs. the queue in
/// front of the full pool is full — and only the label tells them apart, since
/// both are `UNAVAILABLE` on the wire. A test asserting the status code alone
/// cannot distinguish them at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admission_at_the_waiter_cap_records_refused_waiter_cap() {
    // A cap of one, one waiter allowed.
    let fx = AdmissionFixture::setup(1, Some(1)).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());

    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-cap-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // One waiter fills the cap. It will be refused `refused_saturated` when its
    // own wait elapses; what matters here is that it occupies the queue first.
    let svc_for_waiter = svc.clone();
    let waiter_session = held_session.clone();
    let queued = tokio::spawn(async move {
        handle_subscribe_read_layer(
            &svc_for_waiter,
            subscribe_request(
                &waiter_session,
                "layer-held",
                held_object,
                0,
                "client-queued",
            ),
        )
        .await
        .map(|_| ())
    });
    await_condition(
        || fx.client().subscriber_admission.waiters() >= 1,
        "the first waiter joins the admission queue",
    )
    .await;

    let status = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &held_session,
            "layer-held",
            held_object,
            0,
            "client-overflow",
        ),
    )
    .await;
    let status = expect_err_status(status, "a request past the waiter cap was admitted");
    assert_eq!(status.code(), Code::Unavailable);
    // The wire tells the two refusals apart too: `cause` names the full queue.
    assert_refusal_detail(
        &status,
        refusal_reason::SUBSCRIPTION_CAPACITY,
        &[(refusal_metadata::CAUSE, refusal_metadata::WAITER_CAP)],
        CAPACITY_RETRY_DELAY,
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP),
        1,
        "the overflow request must record `refused_waiter_cap`. `refused_saturated` here would \
         report a full QUEUE as a full POOL, and an operator reading the two series would size \
         the pool instead of the queue"
    );

    // The cap-occupying waiter's own outcome, not discarded: a `JoinError` here
    // would mean the handler PANICKED while queued, and swallowing it would let
    // this test pass while the path it depends on was crashing. Its verdict is
    // pinned too — it occupied the cap and then timed out, which is
    // `refused_saturated`, not the `refused_waiter_cap` the overflow request
    // above got.
    let queued_status = tokio::time::timeout(SETTLE_DEADLINE, queued)
        .await
        .expect("the queued waiter must resolve once its admission wait elapses")
        .expect("the queued waiter's handler must not panic");
    let queued_status = expect_err_status(
        queued_status,
        "the waiter occupying the cap was admitted, so it was not in fact occupying the cap and \
         the overflow refusal above proves nothing",
    );
    assert_eq!(
        queued_status.code(),
        Code::Unavailable,
        "a queued waiter whose wait elapses is refused retryably; got {queued_status:?}"
    );
    assert_refusal_detail(
        &queued_status,
        refusal_reason::SUBSCRIPTION_CAPACITY,
        &[(refusal_metadata::CAUSE, refusal_metadata::SATURATED)],
        CAPACITY_RETRY_DELAY,
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        1,
        "the waiter that occupied the cap must be scored `refused_saturated` — it joined the \
         queue and its wait ran out. Only the request that could not join at all is \
         `refused_waiter_cap`, which is the distinction this whole test exists to draw"
    );
    drop(held);
}

/// Refusing new arrivals must not disturb the subscriptions already running.
///
/// The failure this rules out is an admission path that takes something the
/// established stream needs — a slot it probes, a lock it holds — so that a
/// burst of refusals degrades the very streams the ceiling exists to protect.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn established_streams_keep_delivering_while_admission_refuses() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());

    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-survive", "layer-survive").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-survive", object_id, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // Three refusals in a row.
    for attempt in 0..3 {
        let status = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                "layer-survive",
                object_id,
                0,
                &format!("client-refused-{attempt}"),
            ),
        )
        .await;
        let status = expect_err_status(status, "an arrival past the ceiling was admitted");
        assert_eq!(status.code(), Code::Unavailable);
    }
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        3
    );

    // The established stream still delivers. The catch-up phase (including its
    // ack) is fully drained, so the next message can only be the relay of the
    // commit below.
    apply_unique_update(&svc, &session_id, "layer-survive", object_id, 7).await;
    let relayed = next_ok(
        &mut held,
        "the relay of a commit written while admission was refusing new arrivals \
         (a refusal must not touch a running stream's slot or its consumer)",
    )
    .await;
    match relayed.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => assert!(
            h.total_size > 0 && h.sequence_num >= 2,
            "the established stream delivered a payload-free header (total_size={}, \
             sequence_num={}) rather than the relayed commit",
            h.total_size,
            h.sequence_num
        ),
        other => panic!("expected the relayed commit's header, got: {other:?}"),
    }

    drop(held);
}

/// Every `result` and every `reason` value exports at zero before the first
/// event.
///
/// Without pre-initialisation a dashboard's `rate(...)` panel reads
/// absent-metric semantics on a fresh pod, and an alert on
/// `refused_saturated > 0` has no series to evaluate until the pod is already
/// over capacity — i.e. it is silent exactly when it is first needed.
#[tokio::test]
async fn admission_and_stream_exit_label_values_export_at_zero_before_any_event() {
    let metrics = crate::common::metrics_scrape_only(TEST_SERVICE);
    let rendered = render(&metrics);

    for result in ADMISSION_RESULTS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_subscriber_admission_total",
                &[("result", result)]
            ),
            Some(0.0),
            "live_edit_subscriber_admission_total{{result=\"{result}\"}} must export at zero \
             before the first admission"
        );
    }
    for reason in STREAM_EXIT_REASONS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_subscriber_stream_exits_total",
                &[("reason", reason)]
            ),
            Some(0.0),
            "live_edit_subscriber_stream_exits_total{{reason=\"{reason}\"}} must export at zero \
             before the first stream exit"
        );
    }
    for pool in ["txn", "subscriber_pool"] {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_valkey_pool_waiters",
                &[("pool", pool)]
            ),
            Some(0.0),
            "live_edit_valkey_pool_waiters{{pool=\"{pool}\"}} must export at zero before the \
             first waiter"
        );
    }

    // `valkey_pool_in_use{pool}` too — and this one is load-bearing beyond the
    // dashboard argument above. [`read_pool_in_use`] hard-fails on an absent
    // series across two dozen call sites, on the stated premise that the pool
    // label set is pre-initialised; without this loop that premise is an
    // assumption the suite makes about itself. The maintenance pool and the
    // snapshot lane are included even though nothing in this file reads them:
    // the premise is about the whole label set, and a reader pointed at a pool
    // this file does not exercise must still fail loudly rather than read a
    // defaulted zero.
    for pool in ["txn", "subscriber_pool", "maintenance", "snapshot"] {
        assert_eq!(
            sample_value(&rendered, "live_edit_valkey_pool_in_use", &[("pool", pool)]),
            Some(0.0),
            "live_edit_valkey_pool_in_use{{pool=\"{pool}\"}} must export at zero before the \
             pool is first used. `read_pool_in_use` panics on an absent series precisely so an \
             `== 0` assertion cannot pass vacuously, and this is the assertion that makes that \
             panic's premise true"
        );
    }

    // The stream multiplexer's series and the subscription gauges, which an
    // operator watches for a reader that wedges, a join storm or capacity
    // creep: each must be plottable from the first scrape, before its first
    // (often rare) event.
    for (name, labels) in [
        ("live_edit_subscriber_admitted", &[][..]),
        ("live_edit_subscribe_catchups_inflight", &[][..]),
        ("live_edit_subscribe_catchups_waiting", &[][..]),
        ("live_edit_stream_mux_subscriptions", &[][..]),
        ("live_edit_stream_mux_streams", &[][..]),
        ("live_edit_stream_mux_xreads_total", &[][..]),
        ("live_edit_stream_mux_bind_refusals_total", &[][..]),
        ("live_edit_stream_mux_poison_keys_total", &[][..]),
        ("live_edit_valkey_consumer_unblocked_retries_total", &[][..]),
    ] {
        assert_eq!(
            sample_value(&rendered, name, labels),
            Some(0.0),
            "{name}{labels:?} must export at zero before its first event"
        );
    }
    // The overflow endings by `bound`: each cell on its own, so a reader of
    // one cannot be satisfied by the other rendering first.
    for bound in OVERFLOW_BOUNDS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_stream_mux_overflow_kills_total",
                &[("bound", bound)]
            ),
            Some(0.0),
            "live_edit_stream_mux_overflow_kills_total{{bound=\"{bound}\"}} must export at zero"
        );
    }
    assert_eq!(
        OVERFLOW_BOUNDS,
        [OVERFLOW_BOUND_ENTRIES, OVERFLOW_BOUND_BYTES],
        "the overflow bounds are entries / bytes"
    );
    // The head checks by `result`: a `gap` is rare and an `error` rarer, and
    // both must be plottable before the first.
    for result in HEAD_CHECK_RESULTS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_stream_mux_head_checks_total",
                &[("result", result)]
            ),
            Some(0.0),
            "live_edit_stream_mux_head_checks_total{{result=\"{result}\"}} must export at zero"
        );
    }
    assert_eq!(
        HEAD_CHECK_RESULTS,
        [
            HEAD_CHECK_RESULT_GAP,
            HEAD_CHECK_RESULT_NO_GAP,
            HEAD_CHECK_RESULT_ERROR
        ],
        "the head-check results are gap / no_gap / error"
    );
    for outcome in WAKE_OUTCOMES {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_stream_mux_wakes_total",
                &[("outcome", outcome)]
            ),
            Some(0.0),
            "live_edit_stream_mux_wakes_total{{outcome=\"{outcome}\"}} must export at zero"
        );
    }
    for phase in CONSUMER_FORCED_RECONNECT_PHASES {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_valkey_consumer_forced_reconnects_total",
                &[("phase", phase)]
            ),
            Some(0.0),
            "live_edit_valkey_consumer_forced_reconnects_total{{phase=\"{phase}\"}} must export \
             at zero"
        );
    }
    for phase in CONSUMER_PERMANENT_EXIT_PHASES {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_valkey_consumer_permanent_exits_total",
                &[("phase", phase)]
            ),
            Some(0.0),
            "live_edit_valkey_consumer_permanent_exits_total{{phase=\"{phase}\"}} must export at \
             zero"
        );
    }
    // The idle keepalive's ticks by `result`: `failed` is the rare one an
    // alert keys on, so it must be plottable before the first.
    for result in KEEPALIVE_RESULTS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_stream_mux_keepalives_total",
                &[("result", result)]
            ),
            Some(0.0),
            "live_edit_stream_mux_keepalives_total{{result=\"{result}\"}} must export at zero"
        );
    }
    // The readers' own credential re-AUTHs by `result` (b.19i): `failed` is
    // the one an alert keys on, and under `static` auth neither cell ever
    // moves — both must still be plottable from the first scrape.
    for result in REAUTH_RESULTS {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_stream_mux_reauths_total",
                &[("result", result)]
            ),
            Some(0.0),
            "live_edit_stream_mux_reauths_total{{result=\"{result}\"}} must export at zero"
        );
    }
    assert_eq!(
        REAUTH_RESULTS,
        [REAUTH_RESULT_OK, REAUTH_RESULT_FAILED],
        "the re-AUTH results are ok / failed"
    );
    assert_eq!((REAUTH_RESULT_OK, REAUTH_RESULT_FAILED), ("ok", "failed"));
    assert_eq!(
        CONSUMER_FORCED_RECONNECT_PHASES,
        [
            CONSUMER_PHASE_XREAD,
            CONSUMER_PHASE_WAKE_ACK,
            CONSUMER_PHASE_PROBE,
            CONSUMER_PHASE_KEEPALIVE,
            CONSUMER_PHASE_PERMANENT
        ],
        "the forced-reconnect phases are xread / wake_ack / probe / keepalive / permanent"
    );
    assert_eq!(
        (CONSUMER_PHASE_PROBE, CONSUMER_PHASE_KEEPALIVE),
        ("probe", "keepalive"),
        "the label values the published metrics reference documents"
    );
    assert_eq!(
        CONSUMER_PERMANENT_EXIT_PHASES,
        [CONSUMER_PHASE_XREAD, CONSUMER_PHASE_PROBE],
        "the permanent-exit phases are xread / probe"
    );
    assert_eq!(
        KEEPALIVE_RESULTS,
        [KEEPALIVE_RESULT_OK, KEEPALIVE_RESULT_FAILED],
        "the keepalive results are ok / failed"
    );
    assert_eq!(
        (KEEPALIVE_RESULT_OK, KEEPALIVE_RESULT_FAILED),
        ("ok", "failed")
    );
    assert_eq!(WAKE_OUTCOMES, [WAKE_OUTCOME_OK, WAKE_OUTCOME_ERROR]);

    // Non-vacuity: the vocabularies are non-empty and hold the values the code
    // actually emits. An empty slice would satisfy every loop above.
    assert_eq!(
        ADMISSION_RESULTS.len(),
        4,
        "the admission vocabulary is admitted / refused_saturated / refused_waiter_cap / \
         cancelled; got {ADMISSION_RESULTS:?}"
    );
    assert_eq!(
        STREAM_EXIT_REASONS.len(),
        11,
        "the exit vocabulary is slow_reader_deadline / inner_watchdog / session_cancel / \
         session_absent / shutdown / client_disconnect / reader_capacity / catchup_capacity / \
         fenced_read_wait_cap / stream_gap / error; got {STREAM_EXIT_REASONS:?}"
    );
    for expected in [
        ADMISSION_RESULT_ADMITTED,
        ADMISSION_RESULT_REFUSED_SATURATED,
        ADMISSION_RESULT_REFUSED_WAITER_CAP,
        ADMISSION_RESULT_CANCELLED,
    ] {
        assert!(ADMISSION_RESULTS.contains(&expected), "{expected} missing");
    }
    // `shutdown` in particular: without it in the vocabulary, the one exit a
    // rolling restart produces on every idle subscriber would have no
    // pre-initialised series, and `sum by (reason)` would undercount exits.
    assert!(
        STREAM_EXIT_REASONS.contains(&STREAM_EXIT_SHUTDOWN),
        "`{STREAM_EXIT_SHUTDOWN}` missing from {STREAM_EXIT_REASONS:?}"
    );
    // And `session_absent`: the exit a keyspace loss produces on every
    // subscriber of every lost session at once, which is exactly when an
    // operator's `rate(...)` panel needs a series to already exist.
    assert!(
        STREAM_EXIT_REASONS.contains(&STREAM_EXIT_SESSION_ABSENT),
        "`{STREAM_EXIT_SESSION_ABSENT}` missing from {STREAM_EXIT_REASONS:?}"
    );
    // And `stream_gap`: after a restore or a stream-key removal it ends every
    // attached subscription on the layer at once.
    assert!(
        STREAM_EXIT_REASONS.contains(&STREAM_EXIT_STREAM_GAP),
        "`{STREAM_EXIT_STREAM_GAP}` missing from {STREAM_EXIT_REASONS:?}"
    );
    // And `fenced_read_wait_cap`: a catch-up that gave up on its layer's
    // shared fenced read at `reader_wait_max(L)`. Rare by design, so the alert
    // on any increase needs its series from the first scrape.
    assert!(
        STREAM_EXIT_REASONS.contains(&STREAM_EXIT_FENCED_READ_WAIT_CAP),
        "`{STREAM_EXIT_FENCED_READ_WAIT_CAP}` missing from {STREAM_EXIT_REASONS:?}"
    );
    assert_eq!(STREAM_EXIT_FENCED_READ_WAIT_CAP, "fenced_read_wait_cap");
}

// ═══════════════════════════════════════════════════════════════════════════
// The reserved terminal-trailer slot, and tear-down while the client is away
// ═══════════════════════════════════════════════════════════════════════════

/// Stall the reader, write more commits than the channel can hold, and return
/// the subscription's still-unread response stream once the server has torn it
/// down.
///
/// The write count is deliberately `OUTER_CHANNEL_DELIVERY_SLOTS + 8`: enough
/// that the channel provably fills and a send provably parks, without making
/// the backlog so long that the assertions have to reason about which commit was
/// in flight when the deadline fired.
#[cfg(feature = "test-fault-injection")]
async fn stall_until_slow_reader_teardown(
    fx: &AdmissionFixture,
    svc: &LiveEditService,
    metrics: &Metrics,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>> {
    let mut stream = handle_subscribe_read_layer(
        svc,
        subscribe_request(session_id, layer_name, object_id, 0, "client-stalled"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(metrics, "subscriber_pool") == 1,
        "the stalled subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // From here the client reads NOTHING. Each commit relays TWO messages (a
    // content header and one body chunk — see [`COMMITS_TO_FILL_THE_CHANNEL`]),
    // so the 32 delivery slots fill after sixteen of them and the next send
    // parks, at which point the deadline fires. This count is comfortably past
    // that on purpose: it is not trying to land on the boundary, which is what
    // `a_full_response_channel_alone_does_not_tear_the_subscription_down` does.
    for section_id in 1..=(OUTER_CHANNEL_DELIVERY_SLOTS as u64 + 8) {
        apply_unique_update(svc, session_id, layer_name, object_id, section_id).await;
    }

    await_condition(
        || read_stream_exit(metrics, STREAM_EXIT_SLOW_READER_DEADLINE) == 1,
        "the stalled subscriber is torn down on the outer send deadline",
    )
    .await;
    let _ = fx;
    stream
}

/// **Data messages can never consume the slot reserved for the terminal
/// trailer.**
///
/// The response channel is built with 33 and one permit is reserved before the
/// subscription sends anything, so data has 32. A resumed reader must therefore
/// find at most 32 buffered data messages and then the trailer — never 33 data
/// messages and a clean EOS, which is what a 32-capacity channel plus a
/// best-effort trailer produces, and which the Kit client reads as gRPC `OK` and
/// does not retry.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_reserved_trailer_slot_is_never_consumed_by_data_messages() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-trailer-slot", "layer-trailer").await;

    let mut stream = stall_until_slow_reader_teardown(
        &fx,
        &svc,
        &metrics,
        &session_id,
        "layer-trailer",
        object_id,
    )
    .await;

    // Now drain everything the channel is holding.
    let drained = drain_to_end(&mut stream, "a torn-down subscription").await;
    let data = drained.messages.len();

    assert_eq!(
        data, OUTER_CHANNEL_DELIVERY_SLOTS,
        "the subscription buffered {data} data messages, more than the \
         {OUTER_CHANNEL_DELIVERY_SLOTS} delivery slots the response channel reserves for them. \
         The 33rd slot belongs to the terminal trailer and is taken with an `OwnedPermit` before \
         any data is sent; a data message in it means the reservation is gone and a server-driven \
         tear-down can end as a bare EOS"
    );
    assert!(
        data > 0,
        "nothing was buffered, so the channel never filled and this test asserts nothing about \
         the reserved slot"
    );
    let status = drained.terminal.expect(
        "the stream ended without a terminal status. A slow-reader tear-down that ends in EOS is \
         gRPC OK, which the Kit client does not retry — the subscription would be dead for good",
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "the slow-reader tear-down must surface as UNAVAILABLE so the client re-subscribes; \
         got {status:?}"
    );
}

/// The tear-down releases the slot, the admission permit and the tasks **while
/// the client is still not reading**.
///
/// The whole point of the deadline is that a client which has gone away cannot
/// pin a pod's resources. If the release waited on the client draining its
/// backlog — which is what enqueueing the trailer with `tx.send().await` would
/// do — the slot would stay held for exactly as long as the misbehaving client
/// kept ignoring it.
///
/// Observed through state, not wall clock: the `subscriber_pool` gauge returns
/// to zero and a fresh subscription is admitted onto the cap-of-one pod.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_reader_teardown_releases_slot_and_permit_while_the_client_is_still_away() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-release", "layer-release").await;

    // Deliberately kept alive and un-drained for the whole test: the receiver
    // still exists, so this is a stalled reader, not a disconnect.
    let _stalled = stall_until_slow_reader_teardown(
        &fx,
        &svc,
        &metrics,
        &session_id,
        "layer-release",
        object_id,
    )
    .await;

    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the torn-down subscription's stream reader goes idle",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the torn-down subscription's admission permit is released",
    )
    .await;

    // The load-bearing end-to-end check: a fresh subscription is admitted onto
    // the cap-of-one pod, with the previous client still holding its undrained
    // stream.
    let (next_session, next_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-release-next", "layer-next").await;
    let mut next = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&next_session, "layer-next", next_object, 0, "client-next"),
    )
    .await
    .expect(
        "a new subscription was refused even though the stalled one had already been torn down. \
         The slot, the permit or the spawned task is still held by a subscription whose client \
         stopped reading — which is the resource pin the outer send deadline exists to break",
    )
    .into_inner();
    let _ = drain_catch_up(&mut next).await;

    drop(next);
}

/// `live_subscriber_updates_total` counts a commit only once every chunk of it
/// was enqueued to the subscriber's stream (b.21r). A stalled reader's channel
/// fills after [`COMMITS_TO_FILL_THE_CHANNEL`] relayed commits; the next
/// commit's send parks, the outer deadline tears the subscription down, and
/// that commit — read off the layer stream, but never enqueued — must not
/// count. Nor may any of the later commits the subscription never reached.
///
/// The expected count is taken from what the channel actually buffered, not
/// derived from the write count: every complete relayed commit (a content
/// header plus its whole body) the resumed reader drains is exactly one
/// delivery. The teardown is observed first (`slow_reader_deadline`), and the
/// exit is recorded after the subscription's delivery tally has dropped, so the
/// count read afterwards is final.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_commit_whose_send_never_completes_is_not_counted_as_a_subscriber_update() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "b21r-unsent", "layer-unsent").await;

    let mut stream = stall_until_slow_reader_teardown(
        &fx,
        &svc,
        &metrics,
        &session_id,
        "layer-unsent",
        object_id,
    )
    .await;
    let drained = drain_to_end(&mut stream, "a torn-down subscription").await;

    // Walk the backlog: a content header opens a delivery, its body chunks
    // complete it.
    let mut complete = 0u64;
    let mut owed: Option<u64> = None;
    for msg in &drained.messages {
        match &msg.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
                assert!(
                    owed.is_none(),
                    "a header arrived before the previous body completed"
                );
                if h.total_size == 0 {
                    complete += 1;
                } else {
                    owed = Some(h.total_size);
                }
            }
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                let left = owed.expect("a chunk arrived with no header open") - c.data.len() as u64;
                if left == 0 {
                    complete += 1;
                    owed = None;
                } else {
                    owed = Some(left);
                }
            }
            None => panic!("a response carried no payload"),
        }
    }
    assert_eq!(
        complete, COMMITS_TO_FILL_THE_CHANNEL as u64,
        "precondition: the stalled channel buffered exactly the commits that fit, so the next \
         commit's send is the one that parked"
    );

    let counted = sample_value(
        &render(&metrics),
        "live_subscriber_updates_total",
        &[
            ("endpoint", "SubscribeReadLayer"),
            ("service", TEST_SERVICE),
        ],
    )
    .expect("live_subscriber_updates_total{endpoint=\"SubscribeReadLayer\"} is seeded")
        as u64;
    assert_eq!(
        counted, complete,
        "exactly the {complete} commits whose every chunk reached the stream count. The commit \
         whose send parked until the deadline was never enqueued and must not count, and neither \
         may the commits behind it"
    );
}

/// `live_edit_join_latency_seconds` observes once for an admitted subscription
/// that reaches its caught-up ack, and never for a refused one: a refusal is
/// decided before any task exists, and join latency is success-only (b.21r).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_latency_is_observed_for_an_admitted_subscription_and_not_for_a_refused_one() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let join_count = || {
        sample_value(
            &render(&metrics),
            "live_edit_join_latency_seconds_count",
            &[
                ("endpoint", "SubscribeReadLayer"),
                ("service", TEST_SERVICE),
            ],
        )
        .unwrap_or(0.0) as u64
    };

    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "b21r-join-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;
    await_condition(
        || join_count() >= 1,
        "the admitted subscription observes a join latency once caught up",
    )
    .await;
    assert_eq!(
        join_count(),
        1,
        "one admitted subscription, one observation"
    );

    let (other_session, other_object) =
        create_session_and_layer_for(&fx, &svc, "b21r-join-refused", "layer-refused").await;
    let refused = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &other_session,
            "layer-refused",
            other_object,
            0,
            "client-refused",
        ),
    )
    .await;
    let status = expect_err_status(refused, "a second subscription onto a full cap-of-one pod");
    assert_eq!(status.code(), Code::Unavailable);
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        1,
        "precondition: the second subscription was refused at admission"
    );

    assert_eq!(
        join_count(),
        1,
        "a refused subscription reached no state and must not observe a join latency"
    );
    drop(held);
}

/// A client that resumes reading gets its backlog **in order**, and then the
/// `UNAVAILABLE` trailer as the last item — never a clean EOS.
///
/// This is the half of the contract the release test above cannot see: the
/// tear-down happened while the client was away, and the status was nevertheless
/// waiting for it when it came back.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_resumed_reader_drains_its_backlog_in_order_then_the_unavailable_trailer() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-resume", "layer-resume").await;

    let mut stream = stall_until_slow_reader_teardown(
        &fx,
        &svc,
        &metrics,
        &session_id,
        "layer-resume",
        object_id,
    )
    .await;

    // Let everything settle, so a trailer that only arrives because the drain
    // raced it cannot pass for one that was enqueued at tear-down time.
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the subscription is fully torn down before the client resumes reading",
    )
    .await;

    let drained = drain_to_end(&mut stream, "a resumed reader's backlog").await;
    let sequences = drained.header_sequences();

    assert!(
        sequences.len() >= 2,
        "expected at least two buffered update headers to compare; got {sequences:?}"
    );
    let mut sorted = sequences.clone();
    sorted.sort_unstable();
    assert_eq!(
        sequences, sorted,
        "the buffered backlog was delivered out of order: {sequences:?}. The response channel is \
         FIFO and the reserved trailer slot must not reorder what is already queued"
    );

    let status = drained.terminal.expect(
        "the resumed reader drained its backlog and then got a clean end-of-stream. gRPC reads \
         that as OK; the Kit client's retry policy does not retry OK, so a \
         server-driven tear-down would end the subscription permanently. The terminal status must \
         be enqueued into the reserved slot at tear-down time, not attempted afterwards",
    );
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_eq!(
        drained.after_terminal, 0,
        "data was delivered AFTER the terminal status; the trailer must be the last item"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Which endings are clean, and which are not
// ═══════════════════════════════════════════════════════════════════════════

/// An **idle** steady-state subscriber whose session token fires ends
/// `NOT_FOUND "session was deleted"`, counted under `session_cancel`.
///
/// The session is gone, there is nothing to reconnect to, and the client must
/// not retry: `NOT_FOUND`, which it does not retry, says exactly that — where a
/// clean `OK` would be indistinguishable from a shutdown. The token is
/// cancelled directly, the way `DeleteSession` cancels it, which is the
/// deletion path whose message and reason are deterministic: no layer was
/// deleted, so no `deleted` sentinel can win the race to the subscriber (see
/// `an_idle_subscriber_whose_session_is_deleted_ends_not_found` for the real
/// `DeleteSession`, where it can).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_idle_subscriber_whose_session_token_fires_ends_not_found_session_deleted() {
    an_idle_subscriber_whose_session_token_fires_ends_with_the_causes_status(Cause::Deletion).await;
}

/// An **idle** steady-state subscriber whose session token fires because the
/// session was found missing from the live store ends `UNAVAILABLE "session is
/// missing from the live store; …"`, counted under `session_absent` — never the
/// deletion's `NOT_FOUND`, which the client would not retry, losing a session a
/// checkpoint could restore at the retry's setup.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_idle_subscriber_whose_session_goes_missing_from_the_store_ends_unavailable() {
    an_idle_subscriber_whose_session_token_fires_ends_with_the_causes_status(
        Cause::AbsentFromStore,
    )
    .await;
}

/// The body of the two idle-subscriber tests above: `cause` is one of the two
/// session-token causes.
#[cfg(feature = "test-fault-injection")]
async fn an_idle_subscriber_whose_session_token_fires_ends_with_the_causes_status(cause: Cause) {
    assert_ne!(cause, Cause::Shutdown, "a session-token cause only");
    let fx = AdmissionFixture::setup(2, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-idle-delete", "layer-idle").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-idle", object_id, 0, "client-idle"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the idle subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // The client is READING — it is parked on `next()` below — and the layer is
    // quiet, so no send is in flight and the outer deadline is not even armed.
    // A regression that armed it on the idle `recv()` would disconnect this
    // healthy subscriber after TEST_OUTER_SEND_DEADLINE.
    tokio::time::sleep(TEST_OUTER_SEND_DEADLINE * 3).await;

    cause.fire(&svc, &session_id);

    let ending = tokio::time::timeout(SETTLE_DEADLINE, stream.next())
        .await
        .expect("a cancelled idle subscription must end promptly");
    let status = match ending {
        Some(Err(status)) => status,
        other => panic!(
            "an idle subscriber whose session token fired ({cause:?}) must end with a {:?} \
             status. A clean end (gRPC OK) is indistinguishable from the shutdown ending the \
             client must retry, and a data message means the stream did not end; got {other:?}",
            cause.code()
        ),
    };
    assert_eq!(status.code(), cause.code(), "got {status:?}");
    assert_eq!(status.message(), cause.message());

    await_condition(
        || read_stream_exit(&metrics, cause.exit_reason()) == 1,
        "the idle cancelled subscription records exactly one exit under the cause's reason",
    )
    .await;
    for other in cause.other_reasons() {
        assert_eq!(
            read_stream_exit(&metrics, other),
            0,
            "{cause:?} must not be counted under `{other}`"
        );
    }
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "a healthy idle subscriber on a quiet layer must never hit the outer send deadline: no \
         send is in flight, so the deadline is not even armed. Arming it on the idle \
         `consumer.recv()` would disconnect every quiet subscriber every \
         {TEST_OUTER_SEND_DEADLINE:?}"
    );
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the ended subscription idles its stream reader",
    )
    .await;
}

/// An idle subscriber whose session is deleted through the real
/// `DeleteSession` ends `NOT_FOUND`, and nothing else.
///
/// `DeleteSession` deletes the session's layers — appending each layer's
/// `deleted` sentinel — and cancels the session token, and the subscriber ends
/// on whichever it sees first: `NOT_FOUND "session was deleted"` with
/// `reason="session_cancel"` for the token, `NOT_FOUND "layer '<name>' was
/// deleted"` with `reason="error"` for the sentinel. So only the code is
/// asserted, and the exit is required to be exactly one of those two reasons.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_idle_subscriber_whose_session_is_deleted_ends_not_found() {
    use live_edit::grpc::proto::DeleteSessionRequest;
    use live_edit::grpc::session::handle_delete_session;

    let fx = AdmissionFixture::setup(2, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "e1h-idle-delete", "layer-idle").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-idle", object_id, 0, "client-idle"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;

    let mut delete = Request::new(DeleteSessionRequest {
        session_id: session_id.clone(),
    });
    delete
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    handle_delete_session(&svc, delete)
        .await
        .expect("DeleteSession must succeed");

    let ending = tokio::time::timeout(SETTLE_DEADLINE, async {
        loop {
            match stream.next().await {
                Some(Ok(_)) => continue,
                other => return other,
            }
        }
    })
    .await
    .expect("a deleted session's idle subscription must end promptly");
    let status = match ending {
        Some(Err(status)) => status,
        None => panic!(
            "a deleted session's subscription ended with a clean end of stream (gRPC OK), which \
             is indistinguishable from the shutdown ending the client must retry"
        ),
        Some(Ok(_)) => unreachable!("data messages are skipped above"),
    };
    assert_eq!(
        status.code(),
        Code::NotFound,
        "a deletion ends the subscription NOT_FOUND whichever signal wins; got {status:?}"
    );

    await_condition(
        || {
            read_stream_exit(&metrics, STREAM_EXIT_SESSION_CANCEL)
                + read_stream_exit(&metrics, STREAM_EXIT_ERROR)
                == 1
        },
        "the deleted subscription records exactly one exit, under session_cancel or error",
    )
    .await;
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SHUTDOWN),
        0,
        "a deletion is not a shutdown"
    );
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the ended subscription idles its stream reader",
    )
    .await;
}

/// A slow-reader exit is **not** the clean ending, and the clean-end exception
/// cannot absorb it.
///
/// The exhaustive clean-end set is `{client_disconnect}`. This test forces the
/// neighbouring reason and requires a delivered `UNAVAILABLE`, so a future edit
/// that widens the exception (say, to "any exit where the client is not
/// reading") fails here rather than shipping a silently unretryable tear-down.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_reader_exit_delivers_unavailable_and_is_not_a_clean_end() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-slow-exit", "layer-slow").await;

    let mut stream =
        stall_until_slow_reader_teardown(&fx, &svc, &metrics, &session_id, "layer-slow", object_id)
            .await;

    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        1,
        "the exit must be classified `slow_reader_deadline`, not folded into `error` — the two \
         call for different operator responses (a misbehaving client vs. a backend fault)"
    );
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT),
        0,
        "a client that is present but not reading is NOT a disconnect. Classifying it as one \
         would put it in the clean-end set and suppress its trailer"
    );

    let drained = drain_to_end(&mut stream, "a slow-reader tear-down").await;
    let status = drained
        .terminal
        .expect("the slow-reader exit must deliver a status, never a clean EOS");
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
}

/// A client that genuinely disconnects is the **only** clean ending, and it
/// releases everything.
///
/// Paired with the test above so the two halves of the exception are pinned
/// together: present-but-not-reading delivers a status, actually-gone does not
/// (there is nobody to deliver to) — and either way the slot comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_disconnected_client_is_the_clean_ending_and_still_releases_its_slot() {
    let fx = AdmissionFixture::setup(1, None).await;
    // Production deadline: a disconnect is observed through `tx.closed()`, not
    // through the send deadline, so shrinking it would only blur which arm won.
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-disconnect", "layer-disconnect").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-disconnect", object_id, 0, "client-gone"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the subscription's reader shows in valkey_pool_in_use",
    )
    .await;
    assert_eq!(
        read_admitted(&metrics),
        1,
        "an admitted subscription counts on live_edit_subscriber_admitted"
    );
    assert_eq!(
        svc.shutdown.enrolled(),
        1,
        "precondition: the live subscription is enrolled for graceful shutdown's paced ending"
    );

    drop(stream);

    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "a dropped receiver is classified `client_disconnect`",
    )
    .await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "a disconnected subscription idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "a disconnected subscription releases its admission permit",
    )
    .await;
    await_condition(
        || read_admitted(&metrics) == 0,
        "a disconnected subscription leaves live_edit_subscriber_admitted",
    )
    .await;
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "a disconnect must not also be scored as a slow reader: exactly one exit observation is \
         emitted per established subscription, which is what makes `sum by (reason)` the total"
    );
    // The task's exit releases its shutdown enrollment: a leaked one would
    // leave the registry growing by one per stream ever opened, and every
    // shutdown would pace endings for streams long gone.
    await_condition(
        || svc.subscription_tracker.is_empty(),
        "the disconnected subscription's task finishes",
    )
    .await;
    assert_eq!(
        svc.shutdown.enrolled(),
        0,
        "an ended subscription must leave graceful shutdown's enrollment registry"
    );
}

/// A failure raised from inside the subscription task delivers its status too.
///
/// This is the third arm of the same contract: the clean-end set is exactly
/// `{client_disconnect}`, and the `error` bucket — where any backend failure, a
/// `NOT_FOUND`, or an unclassified `?` return lands — is not in it. The exit is
/// driven by deleting the layer under an established subscription, which is the
/// backend-class failure reachable from the gRPC boundary without a new fault
/// seam.
///
/// Discriminating because the tail's match is written as "clean end **iff**
/// `client_disconnect`" rather than as a list of statuses to suppress: a future
/// edit that inverts it into a suppression list would let a new failure mode
/// join the clean set silently, and this test is what notices.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_mid_stream_failure_delivers_its_status_and_is_not_a_clean_end() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-backend-fail", "layer-fail").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-fail", object_id, 0, "client-fail"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    let mut delete = Request::new(live_edit::grpc::proto::DeleteLayerRequest {
        session_id: session_id.clone(),
        layer_name: "layer-fail".to_owned(),
    });
    delete
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    live_edit::grpc::layer::handle_delete_layer(&svc, delete)
        .await
        .expect("DeleteLayer must succeed");

    let item = tokio::time::timeout(SETTLE_DEADLINE, stream.next())
        .await
        .expect("the subscription must end promptly once its layer is gone")
        .expect(
            "the stream closed silently. A subscription whose layer disappeared ended as gRPC \
             OK, which tells the client the subscription is simply over — it never retries and \
             never learns the layer was removed",
        );
    let status = match item {
        Err(status) => status,
        Ok(msg) => panic!("expected a terminal status, got a data message: {msg:?}"),
    };
    assert_eq!(
        status.code(),
        Code::NotFound,
        "a subscription whose layer was deleted must end NOT_FOUND; got {status:?}"
    );

    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_ERROR) == 1,
        "the failure is classified into the `error` bucket",
    )
    .await;
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT),
        0,
        "a backend-class failure must never be scored as a client disconnect — that is the one \
         reason whose trailer is suppressed"
    );
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the failed subscription idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the failed subscription releases its admission permit",
    )
    .await;
    await_condition(
        || read_admitted(&metrics) == 0,
        "a subscription that ended with its status leaves live_edit_subscriber_admitted",
    )
    .await;
}

/// What ends a subscription mid-stream in the cancellation tests below.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cause {
    /// The service shutdown token alone, as graceful shutdown fires it.
    Shutdown,
    /// The session token alone, cancelled as `DeleteSession` cancels it
    /// (`cancel_deleted`). Cancelled directly, so no layer `deleted` sentinel
    /// can race it and the ending's message and reason are deterministic.
    Deletion,
    /// The session token alone, cancelled with the cause the `session_tokens`
    /// sweeper and `acquire_session_cancel` record for a session missing from
    /// the live store with no deletion marker (`cancel_absent_from_store`).
    /// The session itself is left in the store, so a retry converges.
    /// `session_cancel_cause_test.rs` drives the real cancellers.
    AbsentFromStore,
}

#[cfg(feature = "test-fault-injection")]
impl Cause {
    /// Every cause, for the "the others stay at zero" checks.
    const ALL: [Cause; 3] = [Cause::Shutdown, Cause::Deletion, Cause::AbsentFromStore];

    fn fire(self, svc: &LiveEditService, session_id: &str) {
        let session_token = || {
            svc.session_tokens
                .get(session_id)
                .expect("the subscribe handler must have registered a session cancellation token")
                .clone()
        };
        match self {
            Cause::Shutdown => svc.shutdown.cancel(),
            Cause::Deletion => session_token().cancel_deleted(),
            Cause::AbsentFromStore => session_token().cancel_absent_from_store(),
        }
    }

    /// The status the client must receive: `UNAVAILABLE` for a shutdown or a
    /// session missing from the live store, which it retries (where the retry
    /// lands is the ingress's choice; a restorable session is recovered at
    /// its setup); `NOT_FOUND` for a deletion, which it does not.
    fn code(self) -> Code {
        match self {
            Cause::Shutdown | Cause::AbsentFromStore => Code::Unavailable,
            Cause::Deletion => Code::NotFound,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Cause::Shutdown => "server shutting down",
            Cause::Deletion => "session was deleted",
            Cause::AbsentFromStore => {
                "session is missing from the live store; retry so it can be recovered"
            }
        }
    }

    /// The `stream_exits_total{reason}` this cause must be counted under.
    fn exit_reason(self) -> &'static str {
        match self {
            Cause::Shutdown => STREAM_EXIT_SHUTDOWN,
            Cause::Deletion => STREAM_EXIT_SESSION_CANCEL,
            Cause::AbsentFromStore => STREAM_EXIT_SESSION_ABSENT,
        }
    }

    /// The other causes' reasons, which must stay at zero.
    fn other_reasons(self) -> Vec<&'static str> {
        Self::ALL
            .into_iter()
            .filter(|c| *c != self)
            .map(Cause::exit_reason)
            .collect()
    }

    /// Whether the client retries this ending, so a retry is owed a
    /// convergent catch-up.
    fn is_retried(self) -> bool {
        self.code() == Code::Unavailable
    }
}

/// A shutdown, a deletion or a session found missing from the live store that
/// lands **mid-delivery** ends the stream with its own status — `UNAVAILABLE`,
/// `NOT_FOUND` or `UNAVAILABLE` — never cleanly.
///
/// An aborted delivery leaves the client holding a truncated OMPF sequence;
/// presenting that as `OK` leaves the Kit client stuck in initial sync, never
/// marked caught up. A shutdown's or a missing session's `UNAVAILABLE`
/// makes the client retry — wherever the ingress routes it — and catch up in
/// window; a deletion's `NOT_FOUND` tells it there is nothing to catch up to.
///
/// # Establishing mid-delivery, rather than hoping for it
///
/// "Mid-delivery" is a state, and the test establishes it positively instead of
/// racing a timer to it. Three things do that, in order of how much they carry:
///
/// 1. **A backlog several times the channel.** `send_or_disconnect`'s race is
///    `biased` with the shutdown and session tokens as its first two arms, so
///    as long as the subscription still owes messages when the token fires, the
///    cause is observed at the next send whether or not one is parked at that
///    instant. The only way this stream can end on the slow-reader timer
///    instead is if a send was ALREADY parked and aged out before the token
///    fired — which is what the other two items bound. That `biased` ordering
///    is a premise this test rests on rather than checks, so it is pinned where
///    it lives: `live_edit::grpc::subscribe`'s unit tests of
///    `send_racing_endings` fail if the `biased;` is removed or either token
///    arm stops preceding the send.
/// 2. **A positive observation that delivery is live.** The test receives one
///    message after writing a backlog several times the channel's 32 delivery
///    slots. What that receive establishes is that the subscription is actively
///    relaying and still owes messages. It does **not** establish that a send
///    was parked at that instant: what the test observes is messages *owed*, not
///    the channel's depth, so a relay lagging behind the writes could hand back
///    a buffered message with nothing parked behind it. In practice the writes
///    complete long before the relay drains them and the channel is full, which
///    is why this is the establishment step — but that is a practical argument,
///    not a provable one, and the test does not rest on it: item 1's biased
///    token arms reach the exit at the next send, parked or not.
/// 3. **A test-only [`OUTER_SEND_DEADLINE`] override raised to
///    `MID_DELIVERY_SEND_DEADLINE`**, so a send that *is* already parked when
///    the token fires has four times production's grace before the timer could
///    beat the cause to the exit. This is the fallback behind items 1 and 2,
///    not the primary defence — see the constant.
///
/// [`OUTER_SEND_DEADLINE`]: live_edit::storage::notifications::OUTER_SEND_DEADLINE
///
/// **Residual, stated precisely.** None of this makes the ordering a proof: a
/// machine that stalls this task for longer than `MID_DELIVERY_SEND_DEADLINE`
/// between the establishing receive and the token firing two statements later
/// would still exit on the timer. What the test does guarantee is that such a
/// run **fails loudly and unambiguously** rather than flaking: the
/// `slow_reader_deadline == 0` assertion below names that exact scenario, so a
/// red run says "the token lost the race", not "the ending is broken".
#[cfg(feature = "test-fault-injection")]
async fn cancellation_mid_delivery_ends_the_stream_with_its_status(cause: Cause) {
    /// Test-only `OUTER_SEND_DEADLINE`. Raised ABOVE production's 5 s rather
    /// than shrunk: this test needs the slow-reader timer to be the thing that
    /// does NOT fire, which is the opposite of what every other seam user here
    /// wants.
    ///
    /// **It is the fallback, not the defence.** What keeps the cause ahead of
    /// the timer is the establishment step below and `send_or_disconnect`'s
    /// `biased` token arms, which are checked before the send at every send
    /// the subscription still owes; and if both of those lose
    /// anyway, the `slow_reader_deadline == 0` assertion says so out loud
    /// instead of letting the run flake. Raising this constant further buys none
    /// of that, and it is charged in full against nextest's 120 s hang-killer
    /// (module header, § Wait budget) — where a SIGKILL would take every
    /// diagnostic panic in this test with it. So: 20 s, four times production
    /// and still orders of magnitude past the two statements the residual spans.
    const MID_DELIVERY_SEND_DEADLINE: Duration = Duration::from_secs(20);
    /// Commits to write before the token fires. Three channels' worth, so the
    /// subscription still owes messages long after the establishing receive and
    /// the biased token arm is reached at the next send.
    const BACKLOG_COMMITS: u64 = COMMITS_TO_FILL_THE_CHANNEL as u64 * 3;

    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(MID_DELIVERY_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "e1h-cancel-mid", "layer-cancel").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-cancel", object_id, 0, "client-cancel"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // Write a backlog several times the channel's 32 delivery slots, so the
    // subscription still owes several channels' worth of messages when the
    // token fires — the mid-delivery state, as opposed to the idle wait the
    // idle tests cover. Owed messages are what this establishes and what the
    // biased token arms need; an outer send being parked at any particular
    // instant is not observed here (doc comment, item 2).
    for section_id in 1..=BACKLOG_COMMITS {
        apply_unique_update(&svc, &session_id, "layer-cancel", object_id, section_id).await;
    }

    // Establish it. Arrival says the subscription is relaying live and still
    // owes the rest of a backlog several times the 32 delivery slots — which is
    // what the biased token arms need, since they are reached at the next send
    // whether or not one is parked. It does NOT prove a send was parked at this
    // instant; see the doc comment's item 2 for why that is a practical argument
    // and why the test does not rest on it.
    let resumed = next_ok(
        &mut stream,
        "one message from the backlog, which establishes that the subscription is relaying and \
         still owes messages rather than sitting idle",
    )
    .await;

    cause.fire(&svc, &session_id);

    let drained = drain_to_end(&mut stream, "a subscription ended mid-delivery").await;
    let status = drained.terminal.unwrap_or_else(|| {
        panic!(
            "{cause:?} landing mid-delivery ended the stream CLEANLY. The client is holding a \
             truncated OMPF sequence and reads a clean end as `this subscription is over`, so it \
             never retries"
        )
    });
    assert_eq!(
        drained.after_terminal, 0,
        "data was delivered after the terminal status; it must be the last item"
    );
    // The residual named in the doc comment, converted from a silent flake into
    // a loud one — and asserted before the reason: in the residual run the exit
    // is `slow_reader_deadline`, and the reason assertion below would panic
    // with "the ending is broken", so this diagnostic — the reason it exists —
    // would never run.
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "the stream exited on the slow-reader timer, not on the {cause:?}. It means an \
         already-parked send aged past the {MID_DELIVERY_SEND_DEADLINE:?} test deadline in the \
         two statements between the establishing receive and the token firing — a scheduling \
         stall, not an ending defect. Re-run; if it recurs, the establishment needs a real seam"
    );
    assert_eq!(
        status.code(),
        cause.code(),
        "{cause:?} mid-delivery must end {:?}; got {status:?}",
        cause.code()
    );
    assert_eq!(status.message(), cause.message());
    await_condition(
        || read_stream_exit(&metrics, cause.exit_reason()) == 1,
        "the exit is classified under the cause's reason",
    )
    .await;
    for other in cause.other_reasons() {
        assert_eq!(read_stream_exit(&metrics, other), 0, "{other}");
    }
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "an ended subscription idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "an ended subscription releases its admission permit",
    )
    .await;
    await_condition(
        || read_admitted(&metrics) == 0,
        "a subscription ended by its cause leaves live_edit_subscriber_admitted",
    )
    .await;

    if cause.is_retried() {
        // The client retries the `UNAVAILABLE` — on a service that is not
        // shutting down, with a fresh token map — from the last commit it
        // received whole, and the in-window catch-up delivers every commit
        // after it: nothing that was in flight when the cause landed is lost.
        let mut last_whole = None;
        let mut owed: Option<(u64, u64)> = None;
        for msg in std::iter::once(&resumed).chain(drained.messages.iter()) {
            match &msg.response_type {
                Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
                    owed = Some((h.sequence_num, h.total_size));
                }
                Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                    let (sequence, left) = owed.expect("a chunk arrived with no header open");
                    let left = left - c.data.len() as u64;
                    if left == 0 {
                        last_whole = Some(sequence);
                        owed = None;
                    } else {
                        owed = Some((sequence, left));
                    }
                }
                None => panic!("a response carried no payload"),
            }
        }
        let last_whole = last_whole.expect("fixture: at least one commit arrived whole");
        let server_sequence = 1 + BACKLOG_COMMITS;
        assert!(
            last_whole < server_sequence,
            "fixture self-check: the {cause:?} must land with commits still owed"
        );

        let (retry_svc, _) =
            build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
        let mut retried = handle_subscribe_read_layer(
            &retry_svc,
            subscribe_request(
                &session_id,
                "layer-cancel",
                object_id,
                last_whole,
                "client-cancel",
            ),
        )
        .await
        .expect("the retry must be admitted — the ended subscription released its permit")
        .into_inner();
        // One deadline over the whole replay, which is up to a backlog's worth
        // of messages (module header, § Wait budget).
        let (recovered, ack) = tokio::time::timeout(DRAIN_DEADLINE, async {
            let mut recovered = Vec::new();
            loop {
                let item = retried
                    .next()
                    .await
                    .expect("the retry's stream ended during its catch-up")
                    .unwrap_or_else(|s| panic!("the retry's catch-up failed: {s:?}"));
                let header = match item.response_type {
                    Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
                    other => panic!("expected a catch-up header, got {other:?}"),
                };
                if header.total_size == 0 {
                    return (recovered, header);
                }
                recovered.push(header.sequence_num);
                let mut seen = 0;
                while seen < header.total_size {
                    let item = retried
                        .next()
                        .await
                        .expect("the retry's stream ended mid-body")
                        .unwrap_or_else(|s| panic!("the retry's catch-up failed: {s:?}"));
                    match item.response_type {
                        Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                            seen += c.data.len() as u64
                        }
                        other => panic!("expected a body chunk, got {other:?}"),
                    }
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("the retry's catch-up did not finish within {DRAIN_DEADLINE:?}")
        });
        assert_eq!(
            recovered,
            ((last_whole + 1)..=server_sequence).collect::<Vec<_>>(),
            "the in-window catch-up must deliver every commit after the last one the client \
             received whole, in order"
        );
        assert_eq!(ack.sequence_num, server_sequence);
        drop(retried);
    }
}

/// A shutdown mid-delivery ends the stream `UNAVAILABLE "server shutting
/// down"`, `reason=shutdown` — retried, wherever the ingress routes it, and
/// the retry's in-window catch-up delivers what was in flight.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_mid_delivery_ends_the_stream_unavailable() {
    cancellation_mid_delivery_ends_the_stream_with_its_status(Cause::Shutdown).await;
}

/// A deletion mid-delivery ends the stream `NOT_FOUND "session was deleted"`,
/// `reason=session_cancel` — not retried.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deletion_mid_delivery_ends_the_stream_not_found() {
    cancellation_mid_delivery_ends_the_stream_with_its_status(Cause::Deletion).await;
}

/// A session found missing from the live store mid-delivery ends the stream
/// `UNAVAILABLE "session is missing from the live store; …"`,
/// `reason=session_absent` — retried, and the retry's in-window catch-up
/// delivers what was in flight.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_absence_mid_delivery_ends_the_stream_unavailable() {
    cancellation_mid_delivery_ends_the_stream_with_its_status(Cause::AbsentFromStore).await;
}

/// A shutdown, a deletion or a session found missing from the live store that
/// lands while a **reconnecting** subscriber is being sent its in-window
/// backlog ends the stream with the cause's own status — code, message and
/// exit reason — before the caught-up ack, and never through the re-bootstrap
/// path.
///
/// This is the path every Kit client takes when it reconnects in window after
/// a rolling restart: a resume cursor `(object_id, have_version)` whose next
/// commit is still retained, served by the legacy per-commit replay
/// (`catchup_snapshot_enabled = false`, as this file's services are built).
/// That replay computes every commit's diff first, under its one catch-up
/// slot, then releases the slot and sends the computed diffs in order. Each
/// send races the tokens, and a failed send's status is the subscription's
/// ending, returned as-is. The replay must not read it as a failed state read:
/// a deletion's `NOT_FOUND` taken for "the old object's state is gone" would go
/// to `handle_deleted_and_rebootstrap`, which — the mapping still naming this
/// object — ends `NOT_FOUND "layer '…' state was lost and could not be
/// recovered"` under `reason=error`. Asserting the exact message and reason is
/// what rules that out.
///
/// Established like the mid-delivery tests: a backlog three channels deep, and
/// the token fired just after the first replayed message arrives, which is the
/// per-commit header for `have_version + 1` — the positive evidence that the
/// in-window replay, and not a coalesced or full-state catch-up, is running.
/// The same raised `OUTER_SEND_DEADLINE` keeps the slow-reader timer out of the
/// race.
#[cfg(feature = "test-fault-injection")]
async fn cancellation_mid_in_window_replay_ends_with_its_status(cause: Cause) {
    /// As `cancellation_mid_delivery_ends_the_stream_with_its_status`'s.
    const MID_DELIVERY_SEND_DEADLINE: Duration = Duration::from_secs(20);
    /// Commits the reconnecting subscriber has missed: three channels' worth,
    /// two messages each, so the replay still owes most of them when the token
    /// fires.
    const BACKLOG_COMMITS: u64 = COMMITS_TO_FILL_THE_CHANNEL as u64 * 3;

    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(MID_DELIVERY_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "e1h-in-window", "layer-in-window").await;
    let have_version = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the new layer's header must read")
        .sequence_num;
    for section_id in 1..=BACKLOG_COMMITS {
        apply_unique_update(&svc, &session_id, "layer-in-window", object_id, section_id).await;
    }

    // The reconnect, holding the position it had before the backlog.
    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-in-window",
            object_id,
            have_version,
            "client-in-window",
        ),
    )
    .await
    .expect("the reconnect must be admitted")
    .into_inner();
    let first = next_ok(&mut stream, "the in-window replay's first commit").await;
    match &first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
            assert_eq!(
                h.sequence_num,
                have_version + 1,
                "fixture self-check: the reconnect must be sent its per-commit replay from \
                 have_version + 1 (the in-window regime), computed and then sent; got {h:?}"
            );
            assert!(h.total_size > 0, "a replayed commit carries content");
        }
        other => panic!("expected the first replayed commit's header, got {other:?}"),
    }

    cause.fire(&svc, &session_id);

    let drained = drain_to_end(&mut stream, "an in-window replay ended mid-way").await;
    let status = drained.terminal.clone().unwrap_or_else(|| {
        panic!(
            "{cause:?} landing mid in-window replay ended the stream CLEANLY; the client holds a \
             partial replay and no caught-up ack"
        )
    });
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "the stream exited on the slow-reader timer, not on the {cause:?}: an already-parked \
         send aged past {MID_DELIVERY_SEND_DEADLINE:?} between the establishing receive and the \
         token firing — a scheduling stall, not an ending defect"
    );
    assert_eq!(
        status.code(),
        cause.code(),
        "{cause:?} mid in-window replay must end {:?}; got {status:?}",
        cause.code()
    );
    assert_eq!(
        status.message(),
        cause.message(),
        "the send's status must be the ending verbatim; a `state was lost` message means the \
         replay took it for a failed state read and went through the re-bootstrap path"
    );
    await_condition(
        || read_stream_exit(&metrics, cause.exit_reason()) == 1,
        "the exit is classified under the cause's reason",
    )
    .await;
    for other in cause.other_reasons() {
        assert_eq!(read_stream_exit(&metrics, other), 0, "{other}");
    }
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_ERROR),
        0,
        "a re-bootstrap's terminal NOT_FOUND is counted `error`; this exit must not be"
    );

    // Nothing but per-commit replay reached the client, in order, and it
    // stopped short: no caught-up ack, and no full-state re-bootstrap header
    // (which would carry the layer's current sequence).
    let mut replayed = vec![have_version + 1];
    for msg in &drained.messages {
        if let Some(subscribe_read_layer_response::ResponseType::Header(h)) = &msg.response_type {
            assert!(
                h.total_size > 0,
                "a caught-up ack was delivered after the {cause:?}; the cursor must not advance \
                 over a truncated replay"
            );
            replayed.push(h.sequence_num);
        }
    }
    let last = *replayed.last().expect("at least the first commit");
    assert_eq!(
        replayed,
        ((have_version + 1)..=last).collect::<Vec<_>>(),
        "the client must have received consecutive per-commit replay only"
    );
    assert!(
        last < have_version + BACKLOG_COMMITS,
        "fixture self-check: the {cause:?} must land with replayed commits still owed"
    );

    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the ended replay idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the ended replay releases its admission permit",
    )
    .await;
}

/// A shutdown mid in-window replay ends `UNAVAILABLE "server shutting down"`,
/// `reason=shutdown`, before the caught-up ack.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_mid_in_window_replay_ends_the_stream_unavailable() {
    cancellation_mid_in_window_replay_ends_with_its_status(Cause::Shutdown).await;
}

/// A deletion mid in-window replay ends `NOT_FOUND "session was deleted"`,
/// `reason=session_cancel`, without re-bootstrapping.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deletion_mid_in_window_replay_ends_not_found_without_rebootstrap() {
    cancellation_mid_in_window_replay_ends_with_its_status(Cause::Deletion).await;
}

/// A session found missing from the live store mid in-window replay ends
/// `UNAVAILABLE`, `reason=session_absent`, before the caught-up ack and
/// without re-bootstrapping.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_absence_mid_in_window_replay_ends_unavailable_without_rebootstrap() {
    cancellation_mid_in_window_replay_ends_with_its_status(Cause::AbsentFromStore).await;
}

// ═══════════════════════════════════════════════════════════════════════════
// Policy resolution and re-bootstrap under saturation
// ═══════════════════════════════════════════════════════════════════════════

/// An unset `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS` resolves to the
/// subscription cap the gate was built with, and an explicit value is taken as
/// given.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unset_waiter_cap_resolves_to_the_subscription_cap() {
    let fx = AdmissionFixture::setup(3, None).await;
    let gate = &fx.client().subscriber_admission;
    assert_eq!(
        gate.capacity(),
        Some(3),
        "fixture self-check: the configured cap"
    );
    assert_eq!(
        gate.policy().max_waiters,
        3,
        "an unset waiter cap must resolve to the subscription cap"
    );
    assert_eq!(
        gate.policy().wait,
        Duration::from_millis(TEST_ADMISSION_WAIT_MS),
        "the configured admission wait must reach the gate"
    );

    let explicit = AdmissionFixture::setup(3, Some(7)).await;
    assert_eq!(
        explicit.client().subscriber_admission.policy().max_waiters,
        7,
        "an explicit cap must be taken as given, not clamped to the subscription cap"
    );
}

/// A generation transition on a **saturated** pod completes.
///
/// This is why the admission permit lives in the RPC task and not in the
/// consumer: a re-bootstrap unregisters its consumer and registers a new one,
/// and if that required re-entering admission it would be queued behind — and
/// on this fixture refused by — competing arrivals. An established subscriber
/// would then be killed mid-stream by unrelated load, at the exact moment its
/// layer was overwritten.
///
/// The fixture is deliberately the tightest one that can express it: a cap of
/// **one** and a **full** waiter queue, so competing arrivals are actively
/// contending while the transition runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_generation_transition_completes_on_a_saturated_pod() {
    let fx = AdmissionFixture::setup(1, Some(1)).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, old_object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-rebootstrap", "layer-rb").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-rb", old_object_id, 0, "client-rb"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // Keep arrivals hammering admission for the whole transition. They will be
    // refused (`refused_saturated` / `refused_waiter_cap`); the point is that
    // they are contending for the slot the re-bootstrap is about to release.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let contenders = {
        let svc = svc.clone();
        let session_id = session_id.clone();
        let stop = Arc::clone(&stop);
        tokio::spawn(async move {
            let mut attempts = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = handle_subscribe_read_layer(
                    &svc,
                    subscribe_request(&session_id, "layer-rb", 0, 0, "client-contender"),
                )
                .await;
                attempts += 1;
            }
            attempts
        })
    };
    await_condition(
        || {
            read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED)
                + read_admission(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP)
                >= 1
        },
        "at least one arrival is actively being refused while the transition runs",
    )
    .await;

    // Overwrite the layer: the established subscriber reads the `deleted`
    // sentinel off its stream and re-bootstraps onto the new generation.
    let ompf_bytes = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "layer-rb"),
        make_create_data(ompf_bytes),
    ]);
    let new_object_id = handle_create_layer(&svc, make_request(create_stream))
        .await
        .expect("CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "fixture self-check: an overwrite must allocate a new object_id, or there is no \
         generation transition to survive"
    );

    let rebootstrap = next_ok(
        &mut stream,
        "the re-bootstrap's full-state header for the new generation (an established \
         subscription must not be refused its own slot back by competing arrivals)",
    )
    .await;
    match rebootstrap.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => assert_eq!(
            h.object_id, new_object_id,
            "the re-bootstrap must land on the new object"
        ),
        other => panic!("expected the re-bootstrap header, got: {other:?}"),
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let attempts = tokio::time::timeout(SETTLE_DEADLINE, contenders)
        .await
        .expect("the contending task must stop")
        .expect("the contending task must not panic");
    assert!(
        attempts >= 1,
        "no arrival ever contended, so the transition was not run against a saturated pod"
    );

    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_ADMITTED),
        1,
        "the subscription must have been admitted exactly ONCE across the whole generation \
         transition. A second admission means the re-bootstrap re-entered the queue, which is \
         what would have let a competing arrival take the slot and refuse an established stream"
    );
    assert_eq!(
        fx.client().stream_mux.snapshot().subscriptions(),
        1,
        "after the transition the subscription must hold exactly one registration: the old \
         generation's consumer unregistered and the new one registered"
    );

    drop(stream);
}

// ═══════════════════════════════════════════════════════════════════════════
// Convergence after a tear-down
// ═══════════════════════════════════════════════════════════════════════════

/// A torn-down subscription never left the client believing it was further
/// ahead than the content it actually received, and a reconnect converges.
///
/// This is the half of the slow-reader deadline that could silently corrupt a
/// session rather than merely ending it. The tear-down discards commits the
/// server had read off the stream but could not hand over; if the client's
/// notion of its own version had been advanced past them — by an ack claiming a
/// sequence whose content never landed — the reconnect would resume above the
/// gap and those edits would be lost permanently, with no error anywhere.
///
/// Driven the way a real client does it: reconnect carrying the highest
/// `sequence_num` it actually *received*, and require the server to serve
/// everything after it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnect_after_a_slow_reader_teardown_converges_with_no_lost_commits() {
    /// Commits written while the client is not reading. More than the 32
    /// delivery slots, so some provably cannot be handed over before the
    /// tear-down and the reconnect has real work to do.
    const COMMITS: u64 = OUTER_CHANNEL_DELIVERY_SLOTS as u64 + 8;
    /// The layer is created at sequence 1, and each commit bumps it by one.
    const SERVER_SEQUENCE_AFTER_COMMITS: u64 = 1 + COMMITS;

    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-converge", "layer-converge").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-converge",
            object_id,
            0,
            "client-converge",
        ),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let ack = drain_catch_up(&mut stream).await;
    assert_eq!(
        ack.sequence_num, 1,
        "fixture self-check: a freshly created layer is at sequence 1"
    );

    for section_id in 1..=COMMITS {
        apply_unique_update(&svc, &session_id, "layer-converge", object_id, section_id).await;
    }
    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE) == 1,
        "the stalled subscriber is torn down on the outer send deadline",
    )
    .await;

    // Drain what the client actually received. `highest_received` is the only
    // version it is entitled to claim on reconnect.
    let drained = drain_to_end(&mut stream, "a torn-down subscription").await;
    let highest_received = drained
        .header_sequences()
        .into_iter()
        .fold(ack.sequence_num, u64::max);
    assert_eq!(
        drained.terminal.map(|s| s.code()),
        Some(Code::Unavailable),
        "the tear-down must be delivered, or the client would not reconnect at all"
    );
    assert!(
        highest_received < SERVER_SEQUENCE_AFTER_COMMITS,
        "the client received every commit ({highest_received} of \
         {SERVER_SEQUENCE_AFTER_COMMITS}), so nothing was discarded and this test asserts \
         nothing about recovering discarded content"
    );

    // Reconnect exactly where the client actually is.
    let mut reconnected = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-converge",
            object_id,
            highest_received,
            "client-converge",
        ),
    )
    .await
    .expect("the reconnect must be admitted — the torn-down subscription released its permit")
    .into_inner();
    let recovery_ack = drain_catch_up(&mut reconnected).await;
    assert_eq!(
        recovery_ack.sequence_num, SERVER_SEQUENCE_AFTER_COMMITS,
        "the reconnect's catch-up acked sequence {} but the layer is at \
         {SERVER_SEQUENCE_AFTER_COMMITS}. Resuming above the gap is how a slow-reader tear-down \
         turns into permanent edit loss: the discarded commits are never re-sent and nothing \
         reports an error",
        recovery_ack.sequence_num
    );

    // …and the recovered subscription is genuinely live, not merely acked up to
    // date.
    apply_unique_update(&svc, &session_id, "layer-converge", object_id, COMMITS + 1).await;
    let relayed = next_ok(&mut reconnected, "the first commit after reconnecting").await;
    match relayed.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => assert_eq!(
            h.sequence_num,
            SERVER_SEQUENCE_AFTER_COMMITS + 1,
            "the recovered subscription must resume steady state at the next sequence"
        ),
        other => panic!("expected the relayed commit's header, got: {other:?}"),
    }

    drop(reconnected);
}

// ═══════════════════════════════════════════════════════════════════════════
// A genuine backend failure, raised from inside the subscription task
// ═══════════════════════════════════════════════════════════════════════════

/// A **permanent backend failure** on the subscription's own XREAD ends the
/// stream `UNAVAILABLE` and releases everything.
///
/// This is operator arm 4b in its intended form. The sibling
/// `a_mid_stream_failure_delivers_its_status_and_is_not_a_clean_end` drives a
/// `DeleteLayer`, which lands in the same `error` bucket but is an *intentional*
/// removal — it cannot show what happens when the backing store itself fails
/// under a live subscription, because nothing about that path is a failure. This
/// one injects a fred error the reader's taxonomy classifies as permanent
/// (`Auth` is outside the transient set), so the reader ends its subscriptions
/// with `ServiceError::Unavailable` per the b.c7a contract, and the handler
/// turns that into a delivered trailer.
///
/// Discriminating in two directions:
///
/// * against a **clean end** — the b.c7a contract exists because this arm used
///   to `return` bare, which `run_steady_state` read as "the consumer just
///   finished" and closed the RPC with gRPC `OK`. Kit does not retry `OK`, so the
///   subscription died permanently on a transient-looking backend fault. The
///   assertion is therefore on the delivered status, not merely on the metric.
/// * against a **leak** — a subscription that ends through an error path still
///   holds an admission permit and a mux registration, and the failure mode worth
///   catching is an exit that forgets one of them. Under a cap of one a leak is
///   not subtle: the next subscription cannot be admitted at all.
///
/// **One subscription, deliberately.** The fault queue is shared by every
/// reader of the client; the fixture keeps exactly one subscription alive while
/// the queue is non-empty.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanent_backend_failure_mid_stream_delivers_unavailable_and_releases_everything() {
    use crate::storage::mux_test_helpers::{push_faults, read_permanent_exits};
    use live_edit::observability::metrics::CONSUMER_PHASE_XREAD;

    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-backend", "layer-backend").await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-backend", object_id, 0, "client-backend"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut stream).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the established subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // Queued only now, so it lands on a steady-state read: the reader is parked
    // in `XREAD BLOCK`, this commit returns that read, and the reader's NEXT
    // XREAD takes the fault instead of reaching Valkey.
    push_faults(
        fx.client(),
        1,
        fred::error::ErrorKind::Auth,
        "synthetic permanent XREAD failure",
    );
    apply_unique_update(&svc, &session_id, "layer-backend", object_id, 1).await;

    let drained = drain_to_end(&mut stream, "a subscription whose backend failed").await;
    let status = drained.terminal.expect(
        "the stream ended with a clean EOS after a permanent backend failure. gRPC reads that as \
         OK and the Kit client's retry policy does not retry OK, so a backend \
         fault the client should have ridden out by reconnecting would have killed the \
         subscription for good — the exact regression the b.c7a terminal-error contract exists \
         to prevent",
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a permanent backend failure must surface as retryable UNAVAILABLE; got {status:?}"
    );
    assert_eq!(
        drained.after_terminal, 0,
        "data was delivered after the terminal status; the trailer must be last"
    );

    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_ERROR),
        1,
        "the exit must land in the `error` bucket — the default, which is where any backend \
         failure, NOT_FOUND or unclassified `?` return goes. Exactly one observation per \
         established subscription is what makes `sum by (reason)` the total"
    );
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT),
        0,
        "a backend failure must never be scored as a client disconnect — that is the one reason \
         whose trailer is suppressed, so misclassifying here would silently turn this into the \
         clean end the test above rules out"
    );
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "the client drained promptly; no send ever hit the outer deadline"
    );

    // Non-vacuity: the fault really was popped, and it is what ended the
    // subscription (a permanent XREAD exit was counted).
    assert_eq!(
        fx.client()
            .stream_mux
            .pending_faults_for_test(live_edit::storage::stream_mux::MuxFaultSite::Xread),
        0,
        "the injected fault was never popped, so whatever ended this subscription, it was not \
         the backend failure this test means to drive"
    );
    assert_eq!(
        read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD),
        1,
        "the permanent exit must be attributed to the XREAD phase"
    );

    // Everything released. On a cap-of-one pod the end-to-end check is the
    // load-bearing one: a leaked slot or permit makes the next subscription
    // impossible, not merely slow.
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the failed subscription idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the failed subscription releases its admission permit",
    )
    .await;

    let (next_session, next_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-backend-next", "layer-next").await;
    let mut next = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&next_session, "layer-next", next_object, 0, "client-next"),
    )
    .await
    .expect(
        "a new subscription was refused after the previous one failed on a backend error. The \
         slot, the permit or the spawned task did not come back — an error path that leaks is \
         how one bad backend moment permanently shrinks a pod's subscription capacity",
    )
    .into_inner();
    let _ = drain_catch_up(&mut next).await;

    drop(next);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cancellation landing inside the full-state catch-up
// ═══════════════════════════════════════════════════════════════════════════

/// A shutdown, a deletion or a session found missing from the live store that
/// lands **mid full-state catch-up** ends the stream with its own status —
/// `UNAVAILABLE`, `NOT_FOUND` or `UNAVAILABLE` — leaves the client's cursor
/// where it was, and releases everything; after a retried ending, a retry then
/// converges.
///
/// The window this opens is the one the round-1 suite could not reach. Its
/// sibling `cancellation_mid_delivery_ends_the_stream_with_its_status` ends a
/// *steady-state* subscriber that still owes a backlog several times its
/// delivery channel; this one ends it during `deliver_catch_up_content`, which
/// is a different claim about a different phase. An aborted catch-up is a
/// **truncated OMPF sequence**, and presenting that as gRPC `OK` leaves the Kit
/// client stuck in initial sync, never marked caught up, holding a header
/// that promised `total_size` bytes it never received.
///
/// The interleaving is a rendezvous, not wall-clock luck: `CatchUpChunkSeam`
/// parks the chunk loop after the first `send_or_disconnect` returns `Ok`. The
/// first message `chunk_ompf_for_stream` yields is the content **header**
/// (`src/grpc/streaming.rs:65`), so at the park the client has been told a size
/// and a sequence and has received none of the body — the sharpest form of the
/// window, and exactly the state a clean end would misreport.
///
/// **One content-delivering subscription while the seam is installed.** Each of
/// the seam's two `Barrier`s is 2-party, so a second arrival parks forever
/// (bounded only by nextest's hang-killer). The convergence half of this test
/// therefore reconnects through a **second, seam-free service** over the same
/// fixture rather than through `svc`.
#[cfg(feature = "test-fault-injection")]
async fn cancellation_mid_catch_up_truncates_with_its_status(cause: Cause) {
    use live_edit::grpc::subscribe::CatchUpChunkSeam;

    let fx = AdmissionFixture::setup(1, None).await;
    let seam = Arc::new(CatchUpChunkSeam::new());
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            catch_up_chunk: Some(Arc::clone(&seam)),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-cancel-catchup", "layer-catchup").await;

    // A couple of commits so the full-state payload is real content rather than
    // an edge case, and so the convergence check below has something to recover.
    for section_id in 1..=3u64 {
        apply_unique_update(&svc, &session_id, "layer-catchup", object_id, section_id).await;
    }
    const SERVER_SEQUENCE: u64 = 1 + 3;

    // `object_id = 0` is a BRAND-NEW subscribe, which is what routes catch-up
    // through `send_full_state` → `deliver_catch_up_content` — the loop the seam
    // is installed on. Passing the current `object_id` with `sequence_num = 0`
    // would select the legacy in-window replay instead (this file's services run
    // with `catchup_snapshot_enabled = false`), whose sends go through
    // `send_per_commit_payload`, which deliberately carries no seam, and the
    // rendezvous below would time out rather than fail on anything this test is
    // about.
    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-catchup", 0, 0, "client-catchup"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();

    // Rendezvous 1. Bounded, because a catch-up that delivered no content at all
    // would never reach the seam and the bare barrier would hang until the
    // hang-killer — a failure mode worth naming rather than waiting out.
    tokio::time::timeout(SETTLE_DEADLINE, seam.wait_until_parked())
        .await
        .expect(
            "the subscriber never parked at the catch-up chunk seam. Its catch-up delivered no \
             content, so there is no 'content sent, ack not sent' window to cancel into and this \
             test would be asserting nothing. The seam lives in `deliver_catch_up_content`, so \
             the subscribe above must select a regime that routes through it",
        );
    // A catch-up parked in delivery holds no catch-up slot: its slot covered
    // its reads and its build and was released before its first send. What it
    // still holds is its admission.
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
        0,
        "fixture self-check: a catch-up parked in delivery must hold no catch-up slot — it \
         releases its slot before its first send, so a slow client cannot hold one"
    );
    assert_eq!(
        fx.client().catchup_gate.available(),
        fx.client()
            .catchup_gate
            .limit()
            .expect("the fixture caps concurrent catch-ups"),
        "fixture self-check: every catch-up slot is free while the catch-up is parked in delivery"
    );
    assert_eq!(read_admitted(&metrics), 1);

    // The cause lands in the window. No race against the send: the subscriber
    // is parked and cannot progress until `release` below.
    cause.fire(&svc, &session_id);
    // Rendezvous 2, bounded for the same reason as rendezvous 1: the subscriber
    // is parked between the two barriers so this cannot legitimately block, and
    // a bare `wait()` that somehow did would be bounded only by the hang-killer,
    // which prints nothing.
    tokio::time::timeout(SETTLE_DEADLINE, seam.release())
        .await
        .expect("the parked subscriber must accept its release");

    let drained = drain_to_end(&mut stream, "a catch-up ended mid-delivery").await;
    let status = drained.terminal.unwrap_or_else(|| {
        panic!(
            "the catch-up truncated by {cause:?} ended with a clean EOS. The client is holding a \
             content header promising bytes it never received; gRPC reads a clean end as `this \
             subscription is over`, so the Kit client never retries and stays in initial sync \
             forever, never marked caught up"
        )
    });
    assert_eq!(
        status.code(),
        cause.code(),
        "{cause:?} caught mid-catch-up must end {:?}; got {status:?}",
        cause.code()
    );
    assert_eq!(status.message(), cause.message());
    await_condition(
        || read_stream_exit(&metrics, cause.exit_reason()) == 1,
        "the exit is classified under the cause's reason",
    )
    .await;
    for other in cause.other_reasons() {
        assert_eq!(read_stream_exit(&metrics, other), 0, "{other}");
    }

    // The cursor did not advance: the caught-up ack is the ONLY thing that tells
    // a client it is up to date, and it must not have been sent. (`send_ack_header`
    // is the only producer of a `total_size == 0` header.)
    assert!(
        !drained.messages.iter().any(|msg| matches!(
            &msg.response_type,
            Some(subscribe_read_layer_response::ResponseType::Header(h)) if h.total_size == 0
        )),
        "the subscriber received a caught-up ack for a catch-up that was aborted half-delivered. \
         The ack IS the cursor advance — a client that takes it will resume above content it \
         never got, and those edits are then lost with no error anywhere"
    );
    assert!(
        !drained.messages.is_empty(),
        "fixture self-check: the seam parks only after a message is away, so the client must \
         have received at least the content header"
    );

    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the ended catch-up idles its stream reader",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the ended catch-up releases its admission permit",
    )
    .await;
    // …and it took no catch-up slot back on its way out: a leaked one would
    // starve every later catch-up once the limit's worth had leaked.
    await_condition(
        || read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_inflight") == 0,
        "the catch-up ended mid-delivery holds no catch-up slot",
    )
    .await;
    let catch_up_limit = fx
        .client()
        .catchup_gate
        .limit()
        .expect("the fixture caps concurrent catch-ups");
    assert_eq!(
        fx.client().catchup_gate.available(),
        catch_up_limit,
        "every catch-up slot is back"
    );
    await_condition(
        || read_admitted(&metrics) == 0,
        "the ended catch-up leaves live_edit_subscriber_admitted",
    )
    .await;

    if !cause.is_retried() {
        // A deleted session has nothing to converge to; the client does not
        // retry `NOT_FOUND`.
        return;
    }

    // Convergence after a retried ending (a shutdown, or a session the store
    // still holds by the time the retry arrives). A second service over the
    // same fixture, WITHOUT the seam — the retry's own catch-up delivers
    // content and would otherwise park on a barrier whose other party is long
    // gone.
    let (retry_svc, _retry_metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let mut retried = handle_subscribe_read_layer(
        &retry_svc,
        // The client retries from where it actually was, which is nowhere: it
        // never received an ack, so it still has no generation and no version.
        subscribe_request(&session_id, "layer-catchup", 0, 0, "client-catchup"),
    )
    .await
    .expect("the retry must be admitted — the truncated attempt released its permit")
    .into_inner();
    let ack = drain_catch_up(&mut retried).await;
    assert_eq!(
        ack.sequence_num, SERVER_SEQUENCE,
        "the retry's catch-up acked sequence {} but the layer is at {SERVER_SEQUENCE}. A \
         truncated attempt must leave no server-side state that a retry cannot recover from",
        ack.sequence_num
    );

    drop(retried);
}

/// A shutdown mid-catch-up ends `UNAVAILABLE "server shutting down"`,
/// `reason=shutdown`, with no caught-up ack, and the retry converges.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_mid_catch_up_truncates_with_unavailable_and_a_retry_converges() {
    cancellation_mid_catch_up_truncates_with_its_status(Cause::Shutdown).await;
}

/// A deletion mid-catch-up ends `NOT_FOUND "session was deleted"`,
/// `reason=session_cancel`, with no caught-up ack.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deletion_mid_catch_up_truncates_with_not_found() {
    cancellation_mid_catch_up_truncates_with_its_status(Cause::Deletion).await;
}

/// A session found missing from the live store mid-catch-up ends
/// `UNAVAILABLE`, `reason=session_absent`, with no caught-up ack, and the
/// retry converges.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_absence_mid_catch_up_truncates_with_unavailable_and_a_retry_converges() {
    cancellation_mid_catch_up_truncates_with_its_status(Cause::AbsentFromStore).await;
}

/// **The shutdown sequence's bounded subscription drain covers a subscription
/// parked in its catch-up**, the subscription's stream reader and admission
/// permit are back by the time the drain finishes, and the pool quit that
/// follows is not held up by the reader's server-side read.
///
/// The subscription is parked at the catch-up chunk seam, registered on a
/// reader;
/// the park observes no token. The test then does what `main.rs`'s post-server
/// sequence does: cancel the service shutdown token, then
/// `drain_stream_tasks`, which closes `subscription_tracker` and waits on it
/// under a bound. The drain must not finish while the subscription is parked —
/// its task is tracked — and must finish once the park is released and the
/// subscription observes the shutdown at its next send and ends `UNAVAILABLE`.
///
/// Then `ValkeyClient::quit`, as the sequence's last step. The stream was
/// registered before the catch-up's content, and the test waits for its
/// reader to have issued its `XREAD BLOCK` (up to 30 s) before it releases the
/// park, so that read is still blocked on the server after the drained
/// consumer unregistered (removals do not wake a reader). A `QUIT` queued
/// behind it would overrun the 2 s `QUIT_TIMEOUT` and log the reader pool's
/// quit-timeout WARN. So the quit must log none of the reader pool's WARNs,
/// and must log the stream multiplexer's shutdown DEBUG line counting exactly
/// the one read it woke: that line is what proves a read really was blocked,
/// so a clean quit cannot pass because there was nothing to wait for.
///
/// The fixture's keys are purged before the quit: its `Drop` cleanup would
/// otherwise `SCAN` through the quit client and never complete.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_bounded_subscription_drain_covers_a_subscription_parked_in_its_catch_up() {
    use live_edit::grpc::subscribe::CatchUpChunkSeam;
    use live_edit::grpc::transport::drain_stream_tasks;

    /// The drain's bound here — `main.rs` uses the per-task shutdown budget.
    /// Unreachable on the healthy path.
    const SUBSCRIPTION_DRAIN_BUDGET: Duration = Duration::from_secs(5);

    let mut fx = AdmissionFixture::setup(1, None).await;
    let seam = Arc::new(CatchUpChunkSeam::new());
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            catch_up_chunk: Some(Arc::clone(&seam)),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "e1h-drain-parked", "layer-drain").await;
    apply_unique_update(&svc, &session_id, "layer-drain", object_id, 1).await;

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-drain", 0, 0, "client-drain"),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    tokio::time::timeout(SETTLE_DEADLINE, seam.wait_until_parked())
        .await
        .expect("the subscriber never parked at the catch-up chunk seam");
    assert_eq!(
        svc.subscription_tracker.len(),
        1,
        "precondition: the parked subscription's task is on the tracker"
    );
    // Polled, not read once: the stream reader's own task raises the gauge
    // when it handles the registration (`Gauges::sync`), independently of the
    // subscriber reaching the seam, so the park does not imply the gauge is up.
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "precondition: the parked subscription keeps a stream reader busy",
    )
    .await;

    // The post-server sequence, as `main.rs` runs it.
    svc.shutdown.cancel();
    let drain = {
        let presence_cleanup = svc.presence_cleanup_tracker.clone();
        let subscriptions = svc.subscription_tracker.clone();
        tokio::spawn(async move {
            drain_stream_tasks(&presence_cleanup, &subscriptions, SUBSCRIPTION_DRAIN_BUDGET).await
        })
    };
    tokio::task::yield_now().await;
    assert!(
        !drain.is_finished(),
        "the drain finished while a tracked subscription was still parked in its catch-up"
    );

    // The subscription's `Remove` can only follow the release, and a removal
    // does not wake a read, so the reader's read must already be in flight
    // for the shutdown to have a blocked read to wake. Registration alone does
    // not mean that: the reader connects and probes its connection before its
    // first XREAD. (`shutdown_ending_test.rs`'s cluster quit test waits on the
    // same premise.)
    await_condition(
        || fx.client().stream_mux.snapshot().xreads() >= 1,
        "the parked subscription's stream reader issues its read",
    )
    .await;
    tokio::time::timeout(SETTLE_DEADLINE, seam.release())
        .await
        .expect("the parked subscriber must accept its release");
    tokio::time::timeout(SETTLE_DEADLINE, drain)
        .await
        .expect("the drain task must finish")
        .expect("the drain task must not panic");
    assert!(
        svc.subscription_tracker.is_empty(),
        "the subscription drain must finish inside its bound once the park is released; {} \
         task(s) left",
        svc.subscription_tracker.len()
    );

    let drained = drain_to_end(&mut stream, "the parked subscription after shutdown").await;
    let status = drained
        .terminal
        .expect("the parked subscription ended cleanly on shutdown");
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_eq!(read_stream_exit(&metrics, STREAM_EXIT_SHUTDOWN), 1);

    // The drained subscription's task has dropped its consumer, which
    // unregisters it from its reader and returns its admission permit.
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 0,
        "the drained subscription's stream reader goes idle",
    )
    .await;
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the drained subscription's admission permit is released",
    )
    .await;

    // The sequence's last step: the pools quit.
    drop(stream);
    fx.purge_keys_now().await;
    let ((), captured) = super::log_capture::capture_debug(fx.client().quit()).await;
    super::log_capture::assert_quit_was_not_held_by_a_blocked_read(&captured, 1);
}

/// A **full** response channel is not itself a tear-down condition: the deadline
/// arms only around a send that cannot complete.
///
/// Every other slow-reader test in this file goes through
/// [`stall_until_slow_reader_teardown`], which writes `32 + 8` commits in a tight
/// loop — so "the channel became full" and "a send parked past the deadline"
/// happen within milliseconds of each other and nothing distinguishes them. A
/// capacity-based policy (tear down once the channel has no free delivery slot)
/// would pass all of them identically while behaving very differently in
/// production: a client that reads in bursts legitimately fills the channel and
/// drains it again, and killing it for that would disconnect healthy subscribers
/// under exactly the load they are most useful in.
///
/// This test separates the two **in time**. It writes exactly enough commits to
/// fill every delivery slot, so the last send takes the final slot and completes
/// — the channel is full and no send is parked — then holds that state for three
/// times the deadline and requires nothing to happen. Only the next commit, whose
/// send has nowhere to go, may trigger the tear-down.
///
/// Both halves are also the fixture's own self-check on
/// [`COMMITS_TO_FILL_THE_CHANNEL`]: if the messages-per-commit ratio ever
/// changes, one of them fails loudly rather than the test quietly losing its
/// meaning (too few commits and the "full" window is not full; too many and the
/// tear-down happens during it).
///
/// The absence half uses the same shape as
/// `an_idle_subscriber_whose_session_token_fires_ends_not_found_session_deleted`: a bounded window
/// in which a counter must stay at zero, paired with positive evidence that the
/// subscription is still alive (its slot is still held), so "nothing happened"
/// cannot be satisfied by a subscription that had already quietly died.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_response_channel_alone_does_not_tear_the_subscription_down() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-full-not-stalled", "layer-full").await;

    // Deliberately kept alive and never polled from here on: this is a client
    // that has stopped reading, not one that has gone away.
    let _stalled = {
        let mut stream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(&session_id, "layer-full", object_id, 0, "client-full"),
        )
        .await
        .expect("the subscription must be admitted")
        .into_inner();
        // Draining catch-up (including its ack) leaves the channel empty, which
        // is what makes the commit count below equal the slot count.
        let _ = drain_catch_up(&mut stream).await;
        stream
    };
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the subscription's reader shows in valkey_pool_in_use",
    )
    .await;

    // Exactly enough to fill every delivery slot, and not one more.
    for section_id in 1..=(COMMITS_TO_FILL_THE_CHANNEL as u64) {
        apply_unique_update(&svc, &session_id, "layer-full", object_id, section_id).await;
    }

    // The channel is now full and no send is in flight. Nothing may happen.
    tokio::time::sleep(TEST_OUTER_SEND_DEADLINE * 3).await;
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "the subscription was torn down with its channel merely FULL and no send parked. The \
         outer deadline must arm around a send that cannot complete, not around the queue depth \
         — a client that reads in bursts fills this channel routinely and is not a slow reader"
    );
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_ERROR),
        0,
        "nor may a full channel be scored as a backend failure"
    );
    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        1,
        "positive evidence the subscription is still alive: a counter that stayed at zero \
         because the stream had already ended some other way would prove nothing"
    );

    // …and the very next commit, whose send has nowhere to go, does trigger it.
    apply_unique_update(
        &svc,
        &session_id,
        "layer-full",
        object_id,
        COMMITS_TO_FILL_THE_CHANNEL as u64 + 1,
    )
    .await;
    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE) == 1,
        "the first send with nowhere to go tears the subscription down on the outer deadline",
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════
// Send stalls: live_edit_subscriber_send_stalls_total and _send_wait_seconds
// ═══════════════════════════════════════════════════════════════════════════

/// `live_edit_subscriber_send_stalls_total{phase}`. See [`read_admission`] on
/// why this fails loudly on an absent series: both `phase` cells are seeded at
/// zero in `Metrics::new`.
#[cfg(feature = "test-fault-injection")]
fn read_send_stalls(metrics: &Metrics, phase: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_send_stalls_total",
        &[("phase", phase)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_subscriber_send_stalls_total{{phase=\"{phase}\"}} is absent from the \
             scrape; both phase cells are pre-initialised at zero in `Metrics::new`"
        )
    }) as u64
}

/// `live_edit_subscriber_send_wait_seconds_count{phase}`: stalls whose wait has
/// ended. A histogram, so it is not seeded and an absent series reads as 0;
/// every test reading it also reads [`read_send_stalls`] on the same phase,
/// and equality between the two is the property under test.
#[cfg(feature = "test-fault-injection")]
fn read_send_wait_count(metrics: &Metrics, phase: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_send_wait_seconds_count",
        &[("phase", phase)],
    )
    .unwrap_or(0.0) as u64
}

/// Cumulative count of bucket `le` of histogram `family` (absent reads as 0).
#[cfg(feature = "test-fault-injection")]
fn read_bucket(metrics: &Metrics, family: &str, le: &str, labels: &[(&str, &str)]) -> u64 {
    live_edit::observability::metrics::test_util::histogram_bucket(
        &render(metrics),
        family,
        le,
        labels,
    )
    .unwrap_or(0.0) as u64
}

/// Read whole deliveries — a content header and its body chunks — off
/// `stream` until the one at `last_sequence` is complete, under ONE
/// [`DRAIN_DEADLINE`] (see [`drain_to_end`] on why not one per message), and
/// return their sequences in arrival order.
#[cfg(feature = "test-fault-injection")]
async fn read_deliveries_through<S>(stream: &mut S, last_sequence: u64, what: &str) -> Vec<u64>
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(DRAIN_DEADLINE, async {
        let mut sequences = Vec::new();
        loop {
            let header = match stream.next().await {
                Some(Ok(SubscribeReadLayerResponse {
                    response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
                })) => h,
                other => panic!("{what}: expected a delivery header, got {other:?}"),
            };
            let mut seen = 0;
            while seen < header.total_size {
                match stream.next().await {
                    Some(Ok(SubscribeReadLayerResponse {
                        response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
                    })) => seen += c.data.len() as u64,
                    other => panic!("{what}: expected a body chunk, got {other:?}"),
                }
            }
            sequences.push(header.sequence_num);
            if header.sequence_num == last_sequence {
                return sequences;
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!("{what}: delivery {last_sequence} was not read whole within {DRAIN_DEADLINE:?}")
    })
}

/// **A live send that finds the response channel full is a stall: counted when
/// it begins, its wait recorded once it ends, and only under `phase="live"`.**
///
/// The reader stops after its catch-up and the subscription is sent commits
/// until one finds all 32 data slots full, which must count at once. With the
/// outer send deadline raised to 30 s that send stays parked, so its wait is
/// not yet recorded. A second subscription — the sentinel, caught up at the
/// layer's head so its own catch-up sends only an ack and cannot stall — then
/// receives `QUEUED_BEHIND_THE_STALL` more commits, which proves the stream
/// multiplexer has already offered them to the parked subscription's backlog.
/// The send is then held parked for `STALL_HELD` (1.5 s, a sleep used as a
/// lower bound) before the reader resumes and drains everything.
///
/// Once both subscriptions have ended, every counted stall has ended too, so
/// the wait histogram's `_count` must equal the stall counter. The backlog
/// sampled by the first delivery after the resume held the queued commits, so
/// some window lies above `le="0"`. Nothing was a catch-up stall, and the
/// raised deadline never fired. The parked commit was enqueued at least
/// `STALL_HELD` after its commit, so some `enqueued` lag window lies above
/// `le="1"`. Only that lower bound is asserted: an upper bound on a caught-up
/// subscriber's lag would be a wall-clock speed assertion.
///
/// Worst case at full budget: two catch-up drains (≤ 4 messages each at
/// `SETTLE_DEADLINE`, 40 s) + two condition polls (10 s) + two
/// `DRAIN_DEADLINE` reads (40 s) + `STALL_HELD` (1.5 s) ≈ 92 s, inside the
/// 120 s hang-killer. The raised deadline is not payable: nothing awaits it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stalled_live_send_is_counted_at_once_and_its_wait_recorded_when_the_reader_resumes() {
    /// Raised above production's 5 s so the parked send outlasts the sentinel
    /// steps below.
    const STALL_SEND_DEADLINE: Duration = Duration::from_secs(30);
    /// Commits the multiplexer queues for the parked subscription behind its
    /// stalled send.
    const QUEUED_BEHIND_THE_STALL: u64 = 4;
    /// How long the send stays parked before the reader resumes, at least:
    /// past the 1 s lagging-log threshold with margin, so the parked commit's
    /// `enqueued` lag provably exceeds it.
    const STALL_HELD: Duration = Duration::from_millis(1_500);
    assert!(STALL_HELD > LAGGING_LOG_THRESHOLD);

    let fx = AdmissionFixture::setup(2, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(STALL_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "s7j-live-stall", "layer-live-stall").await;

    let mut stalled = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-live-stall",
            object_id,
            0,
            "client-stalled",
        ),
    )
    .await
    .expect("the subscription must be admitted")
    .into_inner();
    let ack = drain_catch_up(&mut stalled).await;

    // From here the client reads nothing until it resumes. Sixteen commits
    // fill the 32 data slots (`COMMITS_TO_FILL_THE_CHANNEL`); the
    // seventeenth's first send finds none free.
    let stalling_commits = COMMITS_TO_FILL_THE_CHANNEL as u64 + 1;
    for section_id in 1..=stalling_commits {
        apply_unique_update(&svc, &session_id, "layer-live-stall", object_id, section_id).await;
    }
    await_condition(
        || read_send_stalls(&metrics, SEND_PHASE_LIVE) >= 1,
        "the send that finds every data slot full is counted as a live stall when it begins",
    )
    .await;
    assert_eq!(
        read_send_wait_count(&metrics, SEND_PHASE_LIVE),
        0,
        "precondition: the stalled send is still parked, and its wait is recorded only when it \
         ends"
    );

    let head = ack.sequence_num + stalling_commits;
    let mut sentinel = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-live-stall",
            object_id,
            head,
            "client-sentinel",
        ),
    )
    .await
    .expect("the sentinel must be admitted")
    .into_inner();
    let sentinel_ack = drain_catch_up(&mut sentinel).await;
    assert_eq!(
        sentinel_ack.sequence_num, head,
        "precondition: the sentinel is caught up at the head with an ack alone"
    );
    for section_id in (stalling_commits + 1)..=(stalling_commits + QUEUED_BEHIND_THE_STALL) {
        apply_unique_update(&svc, &session_id, "layer-live-stall", object_id, section_id).await;
    }
    let last = head + QUEUED_BEHIND_THE_STALL;
    assert_eq!(
        read_deliveries_through(&mut sentinel, last, "the sentinel's live deliveries").await,
        ((head + 1)..=last).collect::<Vec<_>>(),
        "the sentinel received every queued commit, so each was offered to the parked \
         subscription's backlog too"
    );

    // Keep the send parked for a guaranteed minimum before resuming. A sleep
    // is a lower bound on elapsed time, not coordination: whatever the
    // machine's speed, the parked commit is enqueued at least this long after
    // it was committed.
    tokio::time::sleep(STALL_HELD).await;

    // Resume reading: the parked send completes and the backlog drains.
    assert_eq!(
        read_deliveries_through(&mut stalled, last, "the resumed reader's backlog").await,
        ((ack.sequence_num + 1)..=last).collect::<Vec<_>>(),
        "the resumed reader receives every commit in order"
    );
    drop(stalled);
    drop(sentinel);
    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 2,
        "both subscriptions record their client_disconnect exits",
    )
    .await;

    let stalls = read_send_stalls(&metrics, SEND_PHASE_LIVE);
    assert!(stalls >= 1, "the stall counted above is still counted");
    assert_eq!(
        read_send_wait_count(&metrics, SEND_PHASE_LIVE),
        stalls,
        "every counted live stall has ended, so each recorded its wait exactly once"
    );
    let backlog_windows = sample_value(
        &render(&metrics),
        "live_edit_subscriber_backlog_entries_count",
        &[],
    )
    .unwrap_or(0.0) as u64;
    assert!(
        backlog_windows >= 1,
        "the deliveries published a backlog window"
    );
    assert!(
        read_bucket(&metrics, "live_edit_subscriber_backlog_entries", "0", &[]) < backlog_windows,
        "the first delivery after the resume sampled the {QUEUED_BEHIND_THE_STALL} commits \
         queued behind the stall, so some window lies above le=\"0\""
    );
    assert_eq!(
        (
            read_send_stalls(&metrics, SEND_PHASE_CATCHUP),
            read_send_wait_count(&metrics, SEND_PHASE_CATCHUP),
        ),
        (0, 0),
        "no catch-up send stalled: both catch-ups were read as they were sent"
    );
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "the raised deadline never fired"
    );
    let lag_windows = sample_value(
        &render(&metrics),
        "live_edit_subscriber_delivery_lag_seconds_count",
        &[("stage", DELIVERY_LAG_STAGE_ENQUEUED)],
    )
    .unwrap_or(0.0) as u64;
    assert!(lag_windows >= 1, "the deliveries published enqueued lag");
    assert!(
        read_bucket(
            &metrics,
            "live_edit_subscriber_delivery_lag_seconds",
            "1",
            &[("stage", DELIVERY_LAG_STAGE_ENQUEUED)],
        ) < lag_windows,
        "the parked commit was enqueued at least {STALL_HELD:?} after it was committed, so \
         some enqueued window lies above le=\"1\""
    );
}

/// **A stall that runs out the outer send deadline is counted and its wait
/// recorded before the subscription is torn down `slow_reader_deadline`.**
///
/// Sixteen commits fill the channel and the seventeenth's first send is the
/// one stall; it parks until the shrunk 300 ms deadline. The wait recorded is
/// the deadline's: at least 300 ms, so above `le="0.25"` — a lower bound the
/// deadline guarantees, not a measure of speed.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_send_that_runs_out_the_slow_reader_deadline_is_a_counted_stall_with_its_wait_recorded() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "s7j-deadline-stall", "layer-deadline-stall").await;

    let _stalled = stall_until_slow_reader_teardown(
        &fx,
        &svc,
        &metrics,
        &session_id,
        "layer-deadline-stall",
        object_id,
    )
    .await;

    assert_eq!(
        read_send_stalls(&metrics, SEND_PHASE_LIVE),
        1,
        "exactly one send found the channel full: the seventeenth commit's, which the deadline \
         then ended"
    );
    assert_eq!(
        read_send_wait_count(&metrics, SEND_PHASE_LIVE),
        1,
        "the stall the deadline ended recorded its wait"
    );
    assert_eq!(
        read_bucket(
            &metrics,
            "live_edit_subscriber_send_wait_seconds",
            "0.25",
            &[("phase", SEND_PHASE_LIVE)],
        ),
        0,
        "the wait ran out the {TEST_OUTER_SEND_DEADLINE:?} deadline, so it is above le=\"0.25\""
    );
    assert_eq!(read_send_stalls(&metrics, SEND_PHASE_CATCHUP), 0);
}

/// **A rollback-mode in-window replay to a reader that is not reading stalls
/// under `phase="catchup"`, never `live`.**
///
/// The replay (`catchup_snapshot_enabled = false`, as this file's services are
/// built) sends each missed commit as a header and a body, so 20 missed
/// commits are 40 messages for 32 data slots: the seventeenth commit's header
/// stalls and the shrunk deadline ends the subscription. The drained backlog's
/// first header — the commit after the reconnect's position, with content — is
/// the fixture's evidence that this was the per-commit replay.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_mode_in_window_replay_to_a_reader_that_is_not_reading_stalls_as_catchup() {
    const MISSED_COMMITS: u64 = COMMITS_TO_FILL_THE_CHANNEL as u64 + 4;

    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            outer_send_deadline: Some(TEST_OUTER_SEND_DEADLINE),
            ..Default::default()
        },
    );
    assert!(
        !svc.catchup_snapshot_enabled,
        "fixture: this file's services run the rollback-mode catch-up"
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "s7j-replay-stall", "layer-replay-stall").await;
    let have_version = svc
        .layer_store
        .read_object_header(object_id)
        .await
        .expect("the new layer's header must read")
        .sequence_num;
    for section_id in 1..=MISSED_COMMITS {
        apply_unique_update(
            &svc,
            &session_id,
            "layer-replay-stall",
            object_id,
            section_id,
        )
        .await;
    }

    // The reconnect, which then reads nothing.
    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &session_id,
            "layer-replay-stall",
            object_id,
            have_version,
            "client-replay-stall",
        ),
    )
    .await
    .expect("the reconnect must be admitted")
    .into_inner();
    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_SLOW_READER_DEADLINE) == 1,
        "the replay's stalled send runs out the deadline",
    )
    .await;

    let drained = drain_to_end(&mut stream, "the stalled replay").await;
    match drained
        .messages
        .first()
        .and_then(|m| m.response_type.as_ref())
    {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => assert!(
            h.sequence_num == have_version + 1 && h.total_size > 0,
            "fixture self-check: the reconnect was sent the per-commit replay from \
             have_version + 1; got {h:?}"
        ),
        other => panic!("expected the replay's first header, got {other:?}"),
    }

    let stalls = read_send_stalls(&metrics, SEND_PHASE_CATCHUP);
    assert!(
        stalls >= 1,
        "the replay's send that found the channel full is a catchup stall"
    );
    assert_eq!(
        read_send_wait_count(&metrics, SEND_PHASE_CATCHUP),
        stalls,
        "each catchup stall recorded its wait when the deadline ended it"
    );
    assert_eq!(
        (
            read_send_stalls(&metrics, SEND_PHASE_LIVE),
            read_send_wait_count(&metrics, SEND_PHASE_LIVE),
        ),
        (0, 0),
        "a replay is catch-up content, never a live delivery"
    );
}

/// **A client that disconnects while queued for admission removes its waiting
/// work**, and the pool is no worse off for having queued it.
///
/// The ticket's acceptance clause is "cancellation removes waiting work, and
/// retrying disconnected clients does not accumulate tasks". Until now that was
/// pinned only at the checkout unit level, by dropping an `acquire` future
/// directly (`a_cancelled_waiter_returns_its_permit_and_its_queue_position`).
/// This drives it through the real RPC, because the unit test cannot show the
/// thing that actually goes wrong in production: the waiter is inside
/// `handle_subscribe_read_layer`, and what removes it is hyper's `H2Stream`
/// dropping that whole handler future on RST_STREAM. `abort()` on the spawned
/// handler is that same drop.
///
/// Three things are asserted, and the third is the one that bites:
///
///   * the queue empties (`waiters()` back to 0) — a leaked `WaiterSlot` would
///     ratchet the gauge AND permanently consume one of the
///     `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS` slots for the life of the
///     pod, so a pod that saw N disconnects mid-queue would stop queueing at all;
///   * no permit was consumed — an abort that leaked a permit would shrink the
///     pool itself;
///   * **the aborted request records nothing** in
///     `live_edit_subscriber_admission_total`. It never reached a decision, so
///     scoring it as `refused_saturated` would inflate the pod-capacity signal
///     with ordinary client churn, and scoring it as `cancelled` would inflate
///     the shutdown signal. Neither is a refusal the pod made.
///
/// The long admission wait is load-bearing: with the file's usual shrunk wait
/// the waiter would refuse itself as `refused_saturated` before the abort could
/// land, and the abort — the thing under test — would never be what removed it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_disconnects_while_queued_for_admission_leaves_no_waiting_work() {
    /// Long enough that the queued waiter cannot time out during the test.
    const PARKED_WAIT_MS: u64 = 60_000;

    let fx = AdmissionFixture::setup_with_wait(1, None, PARKED_WAIT_MS).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "f4f-queued-disconnect", "layer-queued").await;

    // The holder takes the pool's only permit.
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-queued", object_id, 0, "client-holder"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;
    await_condition(
        || read_pool_in_use(&metrics, "subscriber_pool") == 1,
        "the holder's reader shows in valkey_pool_in_use",
    )
    .await;
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        0,
        "fixture self-check: the holder has the pool's only permit, so the next request queues"
    );
    let admitted_before = read_admission(&metrics, ADMISSION_RESULT_ADMITTED);
    let saturated_before = read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED);
    let cancelled_before = read_admission(&metrics, ADMISSION_RESULT_CANCELLED);
    let waiter_cap_before = read_admission(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP);

    // A second subscribe, spawned so the test can drop it out from under itself
    // the way hyper drops a handler whose client went away.
    let queued = {
        let svc = svc.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            handle_subscribe_read_layer(
                &svc,
                subscribe_request(&session_id, "layer-queued", object_id, 0, "client-queued"),
            )
            .await
            .map(|_| ())
        })
    };
    await_condition(
        || fx.client().subscriber_admission.waiters() >= 1,
        "the second subscribe joins the admission queue",
    )
    .await;

    // The disconnect.
    queued.abort();
    let join = tokio::time::timeout(SETTLE_DEADLINE, queued)
        .await
        .expect("the aborted handler must resolve promptly");
    assert!(
        join.is_err_and(|e| e.is_cancelled()),
        "fixture self-check: the handler must have been ABORTED mid-queue, not have completed \
         on its own — a handler that returned normally would be exercising the ordinary refusal \
         path instead of the disconnect path"
    );

    await_condition(
        || fx.client().subscriber_admission.waiters() == 0,
        "the aborted request's queue position is released",
    )
    .await;
    assert_eq!(
        read_pool_waiters_gauge(&metrics, "subscriber_pool"),
        0,
        "the EXPORTED waiter gauge must come back down too. `WaiterSlot`'s decrement is a `Drop`, \
         and this is the path that tests it: a leak here ratchets the gauge and permanently \
         consumes one LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS slot, so a pod that saw enough \
         mid-queue disconnects would stop queueing altogether"
    );
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        0,
        "the aborted request must not have consumed a permit — the holder still has the only one"
    );

    // It never reached a decision, so it belongs in no `result` cell.
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_ADMITTED),
        admitted_before,
        "an aborted waiter was never admitted"
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        saturated_before,
        "an aborted waiter must NOT be scored as saturation. It is client churn, not a pod that \
         ran out of capacity, and `refused_saturated` is the series an operator sizes the pool on"
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_CANCELLED),
        cancelled_before,
        "nor as `cancelled` — that series means the SERVER withdrew the subscription (shutdown \
         or a DeleteSession), not that the client went away"
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP),
        waiter_cap_before,
        "nor as a waiter-cap refusal: it joined the queue successfully"
    );

    // …and the pool is genuinely usable again, not merely arithmetically tidy:
    // the freed queue slot and the holder's permit both come back.
    drop(held);
    await_condition(
        || fx.client().subscriber_admission.available_permits() == 1,
        "the holder releases its permit once its stream is dropped",
    )
    .await;
    let (next_session, next_object) =
        create_session_and_layer_for(&fx, &svc, "f4f-queued-next", "layer-next").await;
    let mut next = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&next_session, "layer-next", next_object, 0, "client-next"),
    )
    .await
    .expect(
        "a fresh subscribe must be admitted after a mid-queue disconnect and the holder's exit. \
         A refusal here means the abort leaked either the permit or the queue position",
    )
    .into_inner();
    let _ = drain_catch_up(&mut next).await;
    drop(next);
}

// ═══════════════════════════════════════════════════════════════════════════
// Refusal log lines (SDD §17.2.1)
// ═══════════════════════════════════════════════════════════════════════════
//
// A refusal used to record `admission_total{result}` and nothing else: a
// saturated-replica incident left no server-log evidence of which sessions,
// layers and clients were turned away. Each refusal now logs one `RPC failed`
// at `stage=admission`, from the handler wrapper, carrying the request's
// identity and the cause in `status_message`: WARN `UNAVAILABLE` for the two
// capacity refusals, INFO `UNAVAILABLE` for a shutdown (an expected event on
// every rolling restart), INFO `NOT_FOUND` for a deleted session. Captured
// with `WithSubscriber` around the refused handler call itself — the line is
// emitted inside that future, before it returns.

/// Assert the one refusal line a refused subscribe logged — at `level`, with
/// `grpc_code` — naming `session`, `layer` and `client` and carrying `status`
/// as its message.
fn assert_one_refusal_line(
    captured: &super::log_capture::Captured,
    level: &str,
    grpc_code: &str,
    status: &Status,
    session: &str,
    layer: &str,
    client: &str,
) {
    use super::log_capture::{assert_common_fields, assert_field, RPC_FAILURE_MSG};
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, level, "SubscribeReadLayer", "admission", grpc_code);
    assert_field(event, "session_id", session.into());
    assert_field(event, "layer_name", layer.into());
    assert_field(event, "client_id", client.into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "status_message", status.message().into());
    assert!(
        event["fields"].get("regime").is_none() && event["fields"].get("reason").is_none(),
        "a refusal never opened a stream, so it has no teardown fields; got {event}"
    );
}

/// A saturated-pool refusal logs one WARN at `stage=admission` naming the
/// refused session, layer and client, with the saturation cause as its message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_saturated_refusal_logs_one_warn_at_the_admission_stage() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "log-sat-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;

    let (session, object) =
        create_session_and_layer_for(&fx, &svc, "log-sat-refused", "layer-refused").await;
    let (result, captured) = super::log_capture::capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session, "layer-refused", object, 0, "client-refused"),
    ))
    .await;
    let status = expect_err_status(
        result,
        "a subscribe onto a full cap-of-one pod was admitted",
    );
    assert_eq!(status.code(), Code::Unavailable);
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        1,
        "precondition: the refusal was a saturation"
    );

    assert_one_refusal_line(
        &captured,
        "WARN",
        "UNAVAILABLE",
        &status,
        &session,
        "layer-refused",
        "client-refused",
    );
    assert!(
        status.message().contains("concurrent-subscription ceiling"),
        "the saturation cause is what tells this refusal from the others on the line"
    );
    drop(held);
}

/// A waiter-cap refusal logs one WARN at `stage=admission` with the queue-full
/// cause — distinguishable on the line from a saturation, as it is on the
/// metric.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiter_cap_refusal_logs_one_warn_at_the_admission_stage() {
    let fx = AdmissionFixture::setup(1, Some(1)).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "log-cap-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;

    let queued = {
        let svc = svc.clone();
        let session = held_session.clone();
        tokio::spawn(async move {
            handle_subscribe_read_layer(
                &svc,
                subscribe_request(&session, "layer-held", held_object, 0, "client-queued"),
            )
            .await
            .map(|_| ())
        })
    };
    await_condition(
        || fx.client().subscriber_admission.waiters() >= 1,
        "the first waiter joins the admission queue",
    )
    .await;

    let (result, captured) = super::log_capture::capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(
            &held_session,
            "layer-held",
            held_object,
            0,
            "client-overflow",
        ),
    ))
    .await;
    let status = expect_err_status(result, "a request past the waiter cap was admitted");
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP),
        1,
        "precondition: the refusal was the waiter cap"
    );
    assert_one_refusal_line(
        &captured,
        "WARN",
        "UNAVAILABLE",
        &status,
        &held_session,
        "layer-held",
        "client-overflow",
    );
    assert!(
        status
            .message()
            .contains("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS"),
        "the waiter-cap cause must be on the line"
    );

    let _ = tokio::time::timeout(SETTLE_DEADLINE, queued)
        .await
        .expect("the queued waiter must resolve once its admission wait elapses");
    drop(held);
}

/// A queued waiter refused by `fire` — which cancels one token — gets
/// `code`/`message`, records `result="cancelled"`, and logs one line at
/// `stage=admission` at `level`.
///
/// The waiter runs in its own task (it has to be parked while the test fires
/// the token), so the capture wraps that task's future: the line is emitted
/// inside it, by the handler wrapper, before it returns.
async fn queued_refusal_logs_one_line_at_the_admission_stage(
    fire: impl FnOnce(&LiveEditService, &str),
    level: &str,
    code: Code,
    message: &str,
) {
    /// Long enough that the waiter cannot refuse itself as saturated first.
    const PARKED_WAIT_MS: u64 = 60_000;

    let fx = AdmissionFixture::setup_with_wait(1, None, PARKED_WAIT_MS).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (held_session, held_object) =
        create_session_and_layer_for(&fx, &svc, "log-cancel-held", "layer-held").await;
    let mut held = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&held_session, "layer-held", held_object, 0, "client-held"),
    )
    .await
    .expect("the first subscription must be admitted")
    .into_inner();
    let _ = drain_catch_up(&mut held).await;

    // A different session, so cancelling it leaves the holder alone.
    let (session, object) =
        create_session_and_layer_for(&fx, &svc, "log-cancel-queued", "layer-queued").await;
    let queued = {
        let svc = svc.clone();
        let session = session.clone();
        tokio::spawn(async move {
            let (result, captured) = super::log_capture::capture(handle_subscribe_read_layer(
                &svc,
                subscribe_request(&session, "layer-queued", object, 0, "client-queued"),
            ))
            .await;
            (result.map(|_| ()), captured)
        })
    };
    await_condition(
        || fx.client().subscriber_admission.waiters() >= 1,
        "the second subscribe joins the admission queue",
    )
    .await;

    fire(&svc, &session);
    let (result, captured) = tokio::time::timeout(SETTLE_DEADLINE, queued)
        .await
        .expect("a refused waiter must resolve promptly")
        .expect("the queued handler must not panic");
    let status = expect_err_status(result, "a waiter whose token fired was admitted");
    assert_eq!(status.code(), code, "got {status:?}");
    assert_eq!(status.message(), message);
    // A shutdown refusal carries the shutdown retry hint; a session's
    // deletion or absence is not a load-shedding refusal and carries none.
    if message == "server shutting down" {
        assert_shutdown_status(&status);
    } else {
        assert_no_refusal_detail(&status);
    }
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_CANCELLED),
        1,
        "the refusal records `result=\"cancelled\"` — the request went away, not the capacity"
    );
    assert_eq!(
        read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
        0
    );
    assert_one_refusal_line(
        &captured,
        level,
        grpc_status_label(code),
        &status,
        &session,
        "layer-queued",
        "client-queued",
    );
    drop(held);
}

/// The `grpc_code` a completion line carries for `code`.
fn grpc_status_label(code: Code) -> &'static str {
    match code {
        Code::Unavailable => "UNAVAILABLE",
        Code::NotFound => "NOT_FOUND",
        other => panic!("no refusal in this file ends {other:?}"),
    }
}

/// A queued waiter whose **session is deleted** is refused `NOT_FOUND
/// "session was deleted"` — not retried — and logs one INFO line at
/// `stage=admission`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_subscribe_whose_session_is_deleted_is_refused_not_found_and_logs_one_info() {
    queued_refusal_logs_one_line_at_the_admission_stage(
        |svc, session| {
            svc.session_tokens
                .get(session)
                .expect("the queued handler registered its session token before queueing")
                .cancel_deleted()
        },
        "INFO",
        Code::NotFound,
        "session was deleted",
    )
    .await;
}

/// A queued waiter whose session is **found missing from the live store** is
/// refused `UNAVAILABLE "session is missing from the live store; …"` — retried,
/// so the retry's setup can recover it — records `result="cancelled"` like the
/// deletion, and logs one WARN line at `stage=admission`: a session vanishing
/// from under a live request without a deletion is keyspace loss.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_subscribe_whose_session_goes_missing_is_refused_unavailable_and_logs_one_warn() {
    queued_refusal_logs_one_line_at_the_admission_stage(
        |svc, session| {
            svc.session_tokens
                .get(session)
                .expect("the queued handler registered its session token before queueing")
                .cancel_absent_from_store()
        },
        "WARN",
        Code::Unavailable,
        "session is missing from the live store; retry so it can be recovered",
    )
    .await;
}

/// A queued waiter on a replica that **begins shutting down** is refused
/// `UNAVAILABLE "server shutting down"` — retried, wherever the ingress routes
/// it — and logs one INFO
/// line at `stage=admission`: a shutdown is expected on every rolling restart,
/// and WARN would bury real capacity refusals under routine deploys.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_subscribe_on_a_shutting_down_replica_is_refused_unavailable_and_logs_one_info() {
    queued_refusal_logs_one_line_at_the_admission_stage(
        |svc, _session| svc.shutdown.cancel(),
        "INFO",
        Code::Unavailable,
        "server shutting down",
    )
    .await;
}

/// A subscribe arriving **after shutdown began**, on a pod with a free slot, is
/// refused before any task is spawned: Trailers-Only `UNAVAILABLE "server
/// shutting down"`, `result="cancelled"`, no slot taken, nothing on the
/// subscription tracker, and one INFO line at `stage=admission`.
///
/// The free slot is what makes this the entry pre-check: the fast path takes a
/// permit without consulting any token, so without the check the request
/// would be admitted, spawned, and pay for a catch-up on a replica that is
/// going away.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscribe_after_shutdown_began_is_refused_before_any_task_is_spawned() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session, object) =
        create_session_and_layer_for(&fx, &svc, "e1h-refused-shutdown", "layer-refused").await;
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        1,
        "fixture self-check: the slot is free, so only the shutdown can refuse"
    );

    svc.shutdown.cancel();
    let (result, captured) = super::log_capture::capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session, "layer-refused", object, 0, "client-refused"),
    ))
    .await;

    let status = expect_err_status(
        result,
        "a subscribe after shutdown began was admitted onto a free slot",
    );
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_shutdown_status(&status);
    assert_eq!(read_admission(&metrics, ADMISSION_RESULT_CANCELLED), 1);
    assert_eq!(read_admission(&metrics, ADMISSION_RESULT_ADMITTED), 0);
    assert_eq!(
        svc.subscription_tracker.len(),
        0,
        "the refusal must come before the subscription task is spawned"
    );
    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        0,
        "a refused subscribe must not take a slot"
    );
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        1,
        "a refused subscribe must not take a permit"
    );
    assert_one_refusal_line(
        &captured,
        "INFO",
        "UNAVAILABLE",
        &status,
        &session,
        "layer-refused",
        "client-refused",
    );
}

/// A subscribe arriving **once the paced stream ending has started** — `begun`
/// cancelled, while the streams enrolled before it are still being ended in
/// their batches and the service shutdown token itself is not yet cancelled —
/// is refused at admission, Trailers-Only, with the shutdown status and its
/// retry hint, before any task is spawned or any slot or permit taken: its
/// enrollment is born cancelled. Today's single cancel had no such window; a
/// refusal that waited for the service token would admit this request onto
/// the free slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscribe_after_the_paced_ending_began_is_refused_trailers_only_with_the_retry_hint() {
    let fx = AdmissionFixture::setup(1, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session, object) =
        create_session_and_layer_for(&fx, &svc, "kgb-refused-paced", "layer-refused").await;
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        1,
        "fixture self-check: the slot is free, so only the shutdown can refuse"
    );

    let paced = begin_paced_ending_held_open(&svc.shutdown).await;
    let result = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session, "layer-refused", object, 0, "client-refused"),
    )
    .await;

    let status = expect_err_status(
        result,
        "a subscribe after the paced ending began was admitted onto a free slot",
    );
    assert_shutdown_status(&status);
    assert!(
        !svc.shutdown.is_cancelled(),
        "precondition: the refusal came while the service shutdown token was still uncancelled"
    );
    assert_eq!(read_admission(&metrics, ADMISSION_RESULT_CANCELLED), 1);
    assert_eq!(read_admission(&metrics, ADMISSION_RESULT_ADMITTED), 0);
    assert_eq!(
        svc.subscription_tracker.len(),
        0,
        "the refusal must come before the subscription task is spawned"
    );
    assert_eq!(
        fx.client().subscriber_admission.available_permits(),
        1,
        "a refused subscribe must not take a permit"
    );
    drop(paced);
}

// ── Concurrent catch-up limit (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS) ──

fn read_catchup_gauge(metrics: &Metrics, name: &str) -> i64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(&render(metrics), name, &[]).expect("pre-initialised in Metrics::new") as i64
}

/// With one catch-up slot, a second subscription waits — delivering nothing —
/// until the first catch-up has done its reads and its build, then catches up
/// normally.
///
/// The slot holder is parked by `CatchUpStateReadSeam` at its pre-full-state
/// point, after its slot wait and before its read permit and full-state read,
/// which is inside the slot. (A catch-up parked in delivery by `CatchUpChunkSeam`
/// holds no slot at all; `cancellation_mid_catch_up_truncates_with_its_status`
/// pins that.) The seam is shared by every catch-up on the service, so the
/// second one parks at the same point once it holds the slot, and the test
/// releases it the same way.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_past_the_limit_waits_for_a_running_one_to_finish() {
    use live_edit::grpc::subscribe::CatchUpStateReadSeam;

    let fx = AdmissionFixture::setup_with_catchups(1, PARKED_CATCH_UP_BUDGET_MS).await;
    let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            catch_up_state_read: Some(Arc::clone(&seam)),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "catchup-limit", "layer-limit").await;
    for section_id in 1..=3u64 {
        apply_unique_update(&svc, &session_id, "layer-limit", object_id, section_id).await;
    }

    let mut first = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-limit", 0, 0, "client-first"),
    )
    .await
    .expect("admitted")
    .into_inner();
    tokio::time::timeout(SETTLE_DEADLINE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("the first catch-up must park before its state read, holding the only slot");
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
        1,
        "fixture self-check: the catch-up parked before its state read holds the only slot"
    );

    let mut second = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-limit", 0, 0, "client-second"),
    )
    .await
    .expect("admitted: subscriptions are unlimited")
    .into_inner();
    await_condition(
        || read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_waiting") == 1,
        "the second catch-up queues for the slot",
    )
    .await;
    assert!(
        futures::poll!(second.next()).is_pending(),
        "a queued catch-up must deliver nothing until it holds a slot"
    );

    tokio::time::timeout(SETTLE_DEADLINE, seam.arrive_at_pre_full_state())
        .await
        .expect("release the first catch-up");
    let ack = drain_catch_up(&mut first).await;
    assert_eq!(ack.sequence_num, 4);
    // The seam is shared: the second catch-up, now holding the slot, parks at
    // the same point, and the test's arrival is what releases it. (The
    // `parked` latch stays raised from the first catch-up, so it cannot tell
    // the second one's park apart; the 2-party arrival can.)
    tokio::time::timeout(SETTLE_DEADLINE, seam.arrive_at_pre_full_state())
        .await
        .expect("the queued catch-up starts once the slot frees");
    let ack = drain_catch_up(&mut second).await;
    assert_eq!(
        ack.sequence_num, 4,
        "the queued catch-up completes once the slot frees"
    );
    await_condition(
        || read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_inflight") == 0,
        "both slots return once caught up",
    )
    .await;
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
        0
    );
}

/// A caught-up subscription holds no catch-up slot: with a limit of one, a
/// second subscription catches up at once and both then receive live updates.
///
/// The shrunk budget makes a held slot fail loudly: the second catch-up would
/// be refused `catchup_capacity` within `TEST_ADMISSION_WAIT_MS` rather than
/// merely wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_caught_up_subscription_holds_no_catch_up_slot() {
    let fx = AdmissionFixture::setup_with_catchups(1, TEST_ADMISSION_WAIT_MS).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "catchup-steady", "layer-steady").await;

    let mut first = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-steady", 0, 0, "client-a"),
    )
    .await
    .expect("admitted")
    .into_inner();
    drain_catch_up(&mut first).await;
    let mut second = tokio::time::timeout(
        SETTLE_DEADLINE,
        handle_subscribe_read_layer(
            &svc,
            subscribe_request(&session_id, "layer-steady", 0, 0, "client-b"),
        ),
    )
    .await
    .expect("no wait")
    .expect("admitted")
    .into_inner();
    tokio::time::timeout(SETTLE_DEADLINE, drain_catch_up(&mut second))
        .await
        .expect("the second catch-up must not wait behind a caught-up subscription");
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
        0
    );

    apply_unique_update(&svc, &session_id, "layer-steady", object_id, 1).await;
    for stream in [&mut first, &mut second] {
        let msg = next_ok(stream, "a live update").await;
        match msg.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
                assert_eq!(h.sequence_num, 2)
            }
            other => panic!("expected a live Header, got {other:?}"),
        }
    }
}

/// What ends a catch-up queued behind the limit, in
/// [`a_queued_catch_up_ends_on`].
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, Debug)]
enum QueuedEnding {
    /// One token, as [`Cause::fire`] fires it.
    Token(Cause),
    /// The client drops its stream while its catch-up is queued.
    ClientDisconnect,
}

/// `live_edit_subscribe_catchup_wait_seconds_count{result}`; `0` while the
/// series is absent (a histogram series appears with its first observation).
#[cfg(feature = "test-fault-injection")]
fn read_catchup_waits(metrics: &Metrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscribe_catchup_wait_seconds_count",
        &[("result", result)],
    )
    .unwrap_or(0.0) as u64
}

/// A catch-up queued behind the limit ends by its cause, exactly as every
/// other wait does — a session deletion (`NOT_FOUND`, `session_cancel`), a
/// session found missing from the live store (`UNAVAILABLE`,
/// `session_absent`), the replica shutting down (`UNAVAILABLE`, `shutdown`)
/// or the client going away (`client_disconnect`) — without ever taking the
/// slot, delivering nothing, recording its slot wait as `cancelled`, and
/// returning its admission at once.
///
/// The slot holder is a catch-up parked by `CatchUpStateReadSeam` before its
/// full-state read, inside its slot: a catch-up parked in delivery holds no
/// slot, so it could not keep another one queued. The capacity budget is long
/// ([`PARKED_CATCH_UP_BUDGET_MS`]) so the queued wait can only end by the
/// cause the test fires.
#[cfg(feature = "test-fault-injection")]
async fn a_queued_catch_up_ends_on(ending: QueuedEnding) {
    use live_edit::grpc::subscribe::CatchUpStateReadSeam;
    use live_edit::observability::metrics::{
        CATCHUP_WAIT_RESULT_ACQUIRED, CATCHUP_WAIT_RESULT_CANCELLED, CATCHUP_WAIT_RESULT_TIMED_OUT,
        STREAM_EXIT_CATCHUP_CAPACITY,
    };

    let fx = AdmissionFixture::setup_with_catchups(1, PARKED_CATCH_UP_BUDGET_MS).await;
    let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            catch_up_state_read: Some(Arc::clone(&seam)),
            ..Default::default()
        },
    );
    let (holder_session, holder_oid) =
        create_session_and_layer_for(&fx, &svc, "catchup-holder", "layer-holder").await;
    apply_unique_update(&svc, &holder_session, "layer-holder", holder_oid, 1).await;
    let (queued_session, _) =
        create_session_and_layer_for(&fx, &svc, "catchup-queued", "layer-queued").await;

    let mut holder = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&holder_session, "layer-holder", 0, 0, "client-holder"),
    )
    .await
    .expect("admitted")
    .into_inner();
    tokio::time::timeout(SETTLE_DEADLINE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("the holder parks with the only slot");

    let mut queued = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&queued_session, "layer-queued", 0, 0, "client-queued"),
    )
    .await
    .expect("admitted")
    .into_inner();
    await_condition(
        || read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_waiting") == 1,
        "the second catch-up queues",
    )
    .await;
    assert_eq!(
        read_admitted(&metrics),
        2,
        "fixture self-check: both admitted"
    );

    let reason = match ending {
        QueuedEnding::Token(cause) => {
            cause.fire(&svc, &queued_session);
            let drained = drain_to_end(&mut queued, "a queued catch-up that was ended").await;
            let status = drained
                .terminal
                .expect("must end with a status, not a clean close");
            assert_eq!(status.code(), cause.code(), "{ending:?}: {status:?}");
            assert_eq!(status.message(), cause.message(), "{ending:?}");
            assert!(
                drained.messages.is_empty(),
                "a queued catch-up delivers nothing"
            );
            cause.exit_reason()
        }
        QueuedEnding::ClientDisconnect => {
            drop(queued);
            STREAM_EXIT_CLIENT_DISCONNECT
        }
    };
    await_condition(
        || read_stream_exit(&metrics, reason) == 1,
        "the queued catch-up's exit carries the ending's reason",
    )
    .await;
    assert_eq!(
        read_stream_exit(&metrics, STREAM_EXIT_CATCHUP_CAPACITY),
        0,
        "{ending:?}: the wait ended by its cause, not by its budget running out"
    );
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
        0
    );
    assert_eq!(
        read_catchup_waits(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
        1,
        "{ending:?}: the abandoned slot wait records one `cancelled` observation"
    );
    assert_eq!(
        read_catchup_waits(&metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
        0
    );
    assert_eq!(
        read_catchup_waits(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
        1,
        "only the holder took a slot"
    );
    await_condition(
        || read_admitted(&metrics) == 1,
        "the ended catch-up returns its admission while the holder keeps its own",
    )
    .await;

    tokio::time::timeout(SETTLE_DEADLINE, seam.arrive_at_pre_full_state())
        .await
        .expect("release the holder");
    match ending {
        QueuedEnding::Token(Cause::Shutdown) => {
            let drained = drain_to_end(&mut holder, "the holder after shutdown").await;
            assert_eq!(drained.terminal.map(|s| s.code()), Some(Code::Unavailable));
        }
        _ => {
            drain_catch_up(&mut holder).await;
        }
    }
    drop(holder);
    await_condition(
        || read_admitted(&metrics) == 0,
        "every admission is returned once both subscriptions ended",
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_catch_up_ends_not_found_when_its_session_is_deleted() {
    a_queued_catch_up_ends_on(QueuedEnding::Token(Cause::Deletion)).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_catch_up_ends_unavailable_when_its_session_goes_missing_from_the_store() {
    a_queued_catch_up_ends_on(QueuedEnding::Token(Cause::AbsentFromStore)).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_catch_up_ends_unavailable_when_the_replica_shuts_down() {
    a_queued_catch_up_ends_on(QueuedEnding::Token(Cause::Shutdown)).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_catch_up_ends_client_disconnect_when_its_client_goes_away() {
    a_queued_catch_up_ends_on(QueuedEnding::ClientDisconnect).await;
}

// ── /scaling/v1 counters (b.wnb) ────────────────────────────────────────────
//
// The autoscaler sizes the fleet from each replica's pending and completed
// first catch-ups (`Metrics::scaling_counters`, served on `/scaling/v1`, with
// `live_edit_subscribe_catchups_pending` the gauge beside them). A
// subscription counts in when its first catch-up that reads layer state is
// selected, and out exactly once: completed at its first caught-up ack, or
// not completed when it ends before that ack.

/// `(admitted, pending, completed)` as `/scaling/v1` would serve them.
fn scaling_counts(metrics: &Metrics) -> (u64, u64, u64) {
    let c = metrics.scaling_counters();
    (
        c.subscriptions_admitted(),
        c.catchups_pending(),
        c.catchups_completed(),
    )
}

/// Each subscription's first state-reading catch-up completes exactly once:
/// two brand-new subscriptions to a populated layer and one in-window
/// reconnect each add one to the completed count and leave nothing pending;
/// the two first subscriptions show as two admitted; and live deliveries
/// afterwards add nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_first_catch_up_that_reads_state_completes_once_and_leaves_nothing_pending() {
    const LAYER: &str = "layer-scaling-counts";
    let fx = AdmissionFixture::setup(4, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "scaling-counts", LAYER).await;
    for section_id in 1..=2u64 {
        apply_unique_update(&svc, &session_id, LAYER, object_id, section_id).await;
    }
    assert_eq!(scaling_counts(&metrics), (0, 0, 0), "fixture self-check");

    let mut streams = Vec::new();
    for (n, client) in ["client-a", "client-b"].into_iter().enumerate() {
        let mut stream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0, client))
                .await
                .expect("admitted")
                .into_inner();
        assert_eq!(drain_catch_up(&mut stream).await.sequence_num, 3);
        let completed = n as u64 + 1;
        await_condition(
            || scaling_counts(&metrics).2 == completed,
            "a brand-new catch-up of a populated layer completes once",
        )
        .await;
        streams.push(stream);
    }
    assert_eq!(
        scaling_counts(&metrics),
        (2, 0, 2),
        "two subscriptions admitted, none pending, two completed"
    );

    // An in-window reconnect (have version 1 of 3) reads state too.
    let mut reconnect = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, LAYER, object_id, 1, "client-c"),
    )
    .await
    .expect("admitted")
    .into_inner();
    assert_eq!(drain_catch_up(&mut reconnect).await.sequence_num, 3);
    await_condition(
        || scaling_counts(&metrics).2 == 3,
        "an in-window catch-up completes once",
    )
    .await;
    streams.push(reconnect);

    apply_unique_update(&svc, &session_id, LAYER, object_id, 3).await;
    for stream in &mut streams {
        let msg = next_ok(stream, "a live update").await;
        assert!(
            matches!(
                msg.response_type,
                Some(subscribe_read_layer_response::ResponseType::Header(ref h)) if h.sequence_num == 4
            ),
            "{msg:?}"
        );
    }
    assert_eq!(
        scaling_counts(&metrics),
        (3, 0, 3),
        "steady-state deliveries are not catch-ups"
    );
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_pending"),
        0
    );

    drop(streams);
    await_condition(
        || scaling_counts(&metrics).0 == 0,
        "ended subscriptions leave the admitted count",
    )
    .await;
    assert_eq!(
        scaling_counts(&metrics),
        (0, 0, 3),
        "ending an established stream changes neither pending nor completed"
    );
}

/// An up-to-date subscriber reads no layer state, so its catch-up is never
/// counted: nothing pending, nothing completed, even once it is past its ack
/// (its live delivery is sent after the ack). A brand-new subscription on the
/// same service then completes one, so the counter is live in this fixture.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_up_to_date_subscribe_adds_nothing_to_the_completed_count() {
    const LAYER: &str = "layer-up-to-date";
    let fx = AdmissionFixture::setup(4, None).await;
    let (svc, metrics) =
        build_admission_service(&fx, Arc::new(AllowAllPermissions), Seams::default());
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "scaling-up-to-date", LAYER).await;

    let mut up_to_date = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, LAYER, object_id, 1, "client-current"),
    )
    .await
    .expect("admitted")
    .into_inner();
    let ack = drain_catch_up(&mut up_to_date).await;
    assert_eq!(ack.sequence_num, 1, "the up-to-date ack");
    apply_unique_update(&svc, &session_id, LAYER, object_id, 1).await;
    let live = next_ok(&mut up_to_date, "a live update after the ack").await;
    assert!(
        matches!(
            live.response_type,
            Some(subscribe_read_layer_response::ResponseType::Header(ref h)) if h.sequence_num == 2
        ),
        "{live:?}"
    );
    assert_eq!(
        scaling_counts(&metrics),
        (1, 0, 0),
        "an up-to-date subscribe is admitted but never pending or completed"
    );

    let mut brand_new = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, LAYER, 0, 0, "client-new"),
    )
    .await
    .expect("admitted")
    .into_inner();
    drain_catch_up(&mut brand_new).await;
    await_condition(
        || scaling_counts(&metrics).2 == 1,
        "non-vacuity: a brand-new catch-up on the same service completes one",
    )
    .await;
    assert_eq!(scaling_counts(&metrics), (2, 0, 1));
}

/// A client that goes away while its first catch-up is under way leaves the
/// pending count without completing. The catch-up is parked by
/// `CatchUpStateReadSeam` before its full-state read, where it is counted
/// pending (and the gauge shows it); the client drops its stream and the
/// test releases the park; the subscription then ends without an ack.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_gone_before_its_first_ack_leaves_pending_without_completing() {
    use live_edit::grpc::subscribe::CatchUpStateReadSeam;
    const LAYER: &str = "layer-gone-before-ack";

    let fx = AdmissionFixture::setup_with_catchups(1, PARKED_CATCH_UP_BUDGET_MS).await;
    let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
    let (svc, metrics) = build_admission_service(
        &fx,
        Arc::new(AllowAllPermissions),
        Seams {
            catch_up_state_read: Some(Arc::clone(&seam)),
            ..Default::default()
        },
    );
    let (session_id, object_id) =
        create_session_and_layer_for(&fx, &svc, "scaling-gone", LAYER).await;
    apply_unique_update(&svc, &session_id, LAYER, object_id, 1).await;

    let stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, LAYER, 0, 0, "client-gone"),
    )
    .await
    .expect("admitted")
    .into_inner();
    tokio::time::timeout(SETTLE_DEADLINE, seam.wait_until_parked_pre_full_state())
        .await
        .expect("the catch-up parks before its state read");
    assert_eq!(
        scaling_counts(&metrics),
        (1, 1, 0),
        "a selected state-reading catch-up is pending until its ack"
    );
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_pending"),
        1
    );

    drop(stream);
    tokio::time::timeout(SETTLE_DEADLINE, seam.arrive_at_pre_full_state())
        .await
        .expect("release the parked catch-up");
    await_condition(
        || read_stream_exit(&metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "the subscription ends as a client disconnect",
    )
    .await;
    await_condition(
        || scaling_counts(&metrics) == (0, 0, 0),
        "the ended subscription leaves pending and admitted, and completes nothing",
    )
    .await;
    assert_eq!(
        read_catchup_gauge(&metrics, "live_edit_subscribe_catchups_pending"),
        0
    );
}
