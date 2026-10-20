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

//! Integration tests for the b.b2j object-id-counter repair path — the
//! runtime half of the counter-seed contract.
//!
//! ## The failure this file pins
//!
//! The startup counter-seed routine is one-shot, but the failure it guards
//! against is not. A Valkey restart that loses the keyspace does **not**
//! restart the pods: liveness targets the bare gRPC health service, which is
//! `SERVING` from startup to shutdown, and readiness recovers the moment
//! Valkey answers pings again — against an empty keyspace as happily as a
//! full one. So a surviving process gets asked to allocate an `object_id`
//! against a counter that no longer exists. Pre-b.b2j that allocation was a
//! bare `INCR`, which returns `1` on an absent key, so the next `CreateLayer`
//! after a wipe re-minted ids that surviving checkpoint blobs still claim.
//!
//! The tests below cover every arm the fix introduces. Grouped by which
//! decision they pin:
//!
//! **Allocation outcome.**
//!
//! 1. [`keyspace_loss_repairs_counter_and_mints_id_above_every_pre_wipe_id`]
//!    — the regression guard. With a repair hook wired, the first
//!    post-wipe `CreateLayer` re-seeds from the counters sidecar and mints an
//!    id strictly above every pre-wipe id. **This is the test that fails
//!    without the fix**: the old `INCR` path mints `1` here.
//! 2. [`absent_counter_without_repair_hook_initializes_and_mints_from_one`]
//!    — the `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled` posture (and the posture
//!    of every test / tool that builds a bare `LayerStore`): no hook, so the
//!    counter is established at zero, loudly.
//!
//! **Singleflight.**
//!
//! 3. [`repair_singleflights_concurrent_create_layer_burst`] — one keyspace
//!    loss costs one counters-sidecar read, not one per in-flight
//!    `CreateLayer`, and every caller still gets a distinct id.
//! 4. [`repair_singleflight_issues_exactly_one_counters_sidecar_fetch`] — the
//!    same claim witnessed at the adapter rather than derived from the
//!    repair's own metric vocabulary.
//!
//! **The repair attempt's own outcome vector** — every arm of
//! `run_repair_attempt` now has a driver:
//!
//! 5. `repaired_seeded` — test 1 above.
//! 6. [`hooked_repair_without_a_sidecar_initializes_counter_and_mints_one`] —
//!    `SeedOutcome::Absent` → `repaired_initialized`, the fresh-store shape
//!    (adapter configured, counters-sync loop has never run).
//! 7. [`peer_seed_landing_during_the_sidecar_read_reports_repaired_noop`] —
//!    `SeedOutcome::NoOp` → `repaired_noop`, reached deterministically by a
//!    checkpoint store that advances the counter *inside* its
//!    `get_counters_sidecar`, which is exactly the ordering a peer replica
//!    winning the re-seed race produces.
//! 8. [`repair_failure_surfaces_unavailable_without_minting_an_id`] and
//!    [`fetch_budget_elapse_fails_the_repair_without_minting_an_id`] —
//!    `failed`, from an unreachable adapter and from an adapter slow enough
//!    to blow the repair's fetch budget.
//! 9. [`wait_budget_elapse_returns_unavailable_while_the_attempt_completes`]
//!    — `wait_timeout`, the caller-side give-up. Pins the whole point of the
//!    detached-task design: the caller returns `UNAVAILABLE` naming the
//!    still-running repair, and that repair goes on to record its *own*
//!    outcome and leave a usable counter behind.
//! 10. [`counter_absent_again_after_successful_repair_fails_closed_without_minting`]
//!     and [`already_repaired_verdict_then_absent_counter_fails_closed`] —
//!     `LayerStore::allocate_object_id`'s post-repair second-`None` arm
//!     (`failed_after_repair`), reached through a decorator
//!     `ObjectIdCounterRepair` that delegates to the real repair and then
//!     deletes the counter. Both upstream verdicts are covered: the
//!     sidecar-reading repair and the cheap `already_repaired` short-circuit.
//!
//! **The singleflight slot's two non-joinable shapes.**
//!
//! 11. [`repair_attempt_that_dies_without_reporting_self_heals_on_the_next_create_layer`]
//!     — an attempt that ends without sending (a panic, which `tokio::spawn`
//!     swallows). The caller gets the "ended without reporting" arm
//!     (`failed` + `UNAVAILABLE`), and — the half that fails against a
//!     `None`-only join predicate — the **next** `CreateLayer` spawns a fresh
//!     attempt instead of re-joining the dead one. Without that, one panicking
//!     attempt wedges `CreateLayer` for the process lifetime, with nothing to
//!     restart the pod.
//! 12. [`caller_future_dropped_mid_wait_still_lands_the_repair_without_ticking_wait_timeout`]
//!     — the caller's future **dropped** mid-wait, which is what the tower
//!     deadline layer actually does to a non-exempt handler. No arm of
//!     `repair_object_id_counter` runs, so no cell ticks on the caller's
//!     behalf; the attempt nevertheless completes, books `repaired_seeded`, and
//!     leaves the counter at the sidecar-derived seed. Distinct from test 9,
//!     which covers the caller *returning* on its own budget.
//!
//! **The caller's wait sized against the REMAINING deadline (b.hka (f)).**
//!
//! 13. [`create_layer_on_a_short_client_deadline_returns_unavailable_not_deadline_exceeded`]
//!     — the same caller-side give-up as test 9, but with the budget derived
//!     from what is actually left of the request rather than from
//!     `LIVE_EDIT_RPC_DEADLINE_MS`. **This is the test that fails without (f)**:
//!     with the fixed budget the client's shorter `grpc-timeout` elapses first,
//!     the handler future is dropped, and the client sees a bare
//!     `DEADLINE_EXCEEDED` with no `wait_timeout` tick and nothing said about
//!     the repair still running.
//!
//! ## Feature gating
//!
//! Tests 4 and 8a sit behind `test-checkpoint-fault-injection` because they
//! drive `InMemoryCheckpointStore::with_fault_injector`. Per
//! `docs-internal/test-writing-guide.md`, a missing feature is the *silent*
//! activation condition — the test compiles out entirely and the run reports
//! green having covered nothing. The workspace **Full test** and **Lint**
//! commands both pass it; a hand-rolled narrow invocation must too.
//!
//! Everything else in this file is deliberately **ungated**. The
//! decorator-`ObjectIdCounterRepair` and the scripted [`ScriptedCountersStore`]
//! reach their arms through the production trait surfaces
//! (`ObjectIdCounterRepair`, `CheckpointStore`), which need no test-only
//! feature — so the coverage that matters most here survives a narrow run that
//! forgets the flags.
//!
//! ## Shared-Valkey discipline
//!
//! The "keyspace loss" these tests simulate is scoped to the fixture's own
//! `TestValkey` prefix via [`full_valkey_wipe`] (SCAN + UNLINK over
//! `tv.prefix()*`). **No test here issues `FLUSHALL` / `FLUSHDB`** — the
//! backing instance is shared with concurrently-running agents and CI jobs,
//! and a real flush would destroy their fixtures. The per-test prefix makes
//! the namespaced wipe an exact model of the production shape anyway: from
//! this `LayerStore`'s point of view every key it can name is gone.

use std::collections::HashSet;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tokio::sync::{Barrier, Notify};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{
    seed_object_id_counter, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta,
    CheckpointObjectIdCounterRepair, CheckpointStore, SeedOutcome,
};
use live_edit::config::Config;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::counters::{CounterStore, ObjectIdCounterRepair};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use tower::{Layer, Service};

use super::helpers::full_valkey_wipe;

const SERVICE_LABEL: &str = "counter-repair-test-replica";

/// `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN` used throughout. Below the
/// shipped default (`CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT`, 1,000,000) so
/// the pinned ids stay small; the repair's arithmetic (sidecar + margin) is
/// the same at any accepted margin.
const SAFETY_MARGIN: u64 = 1000;

/// Bound on the repair's counters-sidecar fetch. Production derives this from
/// `Config::checkpoint_recovery_fetch_budget()`; against the in-memory
/// adapter the fetch is effectively instant, so any generous value works and
/// a tight one would only add CI flake surface.
const FETCH_BUDGET: Duration = Duration::from_secs(10);

/// Bound on how long a caller waits for the in-flight repair. Production
/// derives this from `Config::object_id_counter_repair_wait_budget()` (⅔ of the
/// RPC deadline); against the in-memory adapter the repair completes in
/// microseconds, so a generous value keeps the wait-timeout arm out of these
/// fixtures entirely — it is a separate contract with its own coverage need.
const WAIT_BUDGET: Duration = Duration::from_secs(10);

// ─────────────────────────────────────────────────────────────────────────
// Shared helpers.
// ─────────────────────────────────────────────────────────────────────────

fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL))
}

/// Read one `live_edit_object_id_counter_repair_total{outcome}` cell off the
/// rendered scrape exposition. Absent cells read as `0` — an OTel counter
/// emits no series until its first observation.
///
/// The metric is deliberately unlabeled by `service` (the `LayerStore`
/// fallback path holds no service label), so `outcome` is the whole label
/// set.
fn repair_metric(metrics: &SharedMetrics, outcome: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_object_id_counter_repair_total",
        &[("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// Persist a counters sidecar carrying `max_object_id_observed`, through the
/// real `CountersSidecar` struct rather than hand-rolled JSON — the repair
/// deserializes with the same type, so the round-trip pins the actual
/// producer/consumer contract.
async fn put_sidecar(store: &dyn CheckpointStore, max_object_id_observed: u64) {
    let sidecar = CountersSidecar {
        max_object_id_observed,
        updated_at: "2026-08-27T12:00:00Z".to_string(),
    };
    let bytes = serde_json::to_vec(&sidecar).expect("CountersSidecar serialization must succeed");
    store
        .put_counters_sidecar(Bytes::from(bytes))
        .await
        .expect("put_counters_sidecar must succeed on the InMemory adapter");
}

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Minimal `ObjectState` — these tests assert on allocated ids, not on
/// materialized layer content, so the smallest state that `write_object`
/// accepts keeps the fixtures honest about what is under test.
fn minimal_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    }
}

/// Build a `CounterStore` over the fixture's namespace.
fn counter_store(tv: &TestValkey) -> Arc<CounterStore> {
    Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ))
}

/// Build a `LayerStore` over the fixture's namespace, with metrics attached
/// (required for every repair-outcome assertion) and the repair hook
/// optionally installed.
fn layer_store(
    tv: &TestValkey,
    metrics: &SharedMetrics,
    repair: Option<Arc<dyn ObjectIdCounterRepair>>,
) -> LayerStore {
    let mut ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(Arc::clone(metrics));
    if let Some(repair) = repair {
        ls = ls.with_object_id_counter_repair(repair);
    }
    ls
}

/// Whether an arbitrary key currently exists in the fixture namespace.
async fn key_exists(tv: &TestValkey, key: String) -> bool {
    use fred::interfaces::KeysInterface;
    tv.client()
        .pool
        .exists(key.clone())
        .await
        .unwrap_or_else(|e| panic!("EXISTS {key} must succeed: {e}"))
}

/// Whether `le:counter:object_id` currently exists in the fixture namespace.
async fn counter_key_exists(tv: &TestValkey) -> bool {
    key_exists(tv, tv.key_schema().counter_object_id()).await
}

/// Raw value of `le:counter:object_id`, or `None` when the key is absent.
///
/// Deliberately NOT `CounterStore::current_object_id`, which collapses absent
/// to `0`: several assertions below have to tell "gone" apart from "present and
/// zero", and that conflation is precisely what the `SET NX 0` establishment
/// step exists to remove.
async fn raw_counter(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET le:counter:object_id must succeed")
}

/// Poll `repair_metric(metrics, outcome)` until it reaches `target`, or panic
/// on deadline expiry.
///
/// Needed by the wait-budget test, whose whole subject is an attempt that
/// outlives the caller that started it: the `CreateLayer` returns while the
/// detached task is still running, so its outcome cell lands at an instant no
/// RPC return is synchronized with. A bounded poll is the project's documented
/// answer to that shape (`docs-internal/test-writing-guide.md`
/// `## Synchronization Patterns`) — a bare sleep would either flake or hide a
/// regression that stopped recording the outcome at all.
async fn wait_for_repair_metric(metrics: &SharedMetrics, outcome: &str, target: u64) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let observed = repair_metric(metrics, outcome);
        if observed >= target {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "live_edit_object_id_counter_repair_total{{outcome=\"{outcome}\"}} never reached \
                 {target} (stuck at {observed}) within 30s. The detached repair attempt is \
                 supposed to run to completion and record its own outcome even after every \
                 caller has given up on it — that independence is the entire reason the attempt \
                 is spawned rather than awaited inline"
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Await a [`Notify`] under a generous deadline, panicking with `what` on
/// expiry.
///
/// The bare `notified().await` would hang the whole binary if the awaited event
/// never happened — nextest's 120 s hard-kill would eventually end it, but with
/// a "test timed out" verdict that says nothing about which side never arrived.
/// The deadline is deliberately far above any plausible scheduling latency: it
/// is a diagnostic, never a synchronization primitive, and nothing here is
/// asserting on time.
async fn await_notified(notify: &Notify, what: &str) {
    tokio::time::timeout(Duration::from_secs(30), notify.notified())
        .await
        .unwrap_or_else(|_| panic!("timed out after 30s waiting for: {what}"));
}

/// Drive `WRITERS` concurrent `create_layer` calls that all release from one
/// [`Barrier`], and return the ids they were handed in completion order.
///
/// Extracted because both burst tests need the identical driver and the barrier
/// discipline is the load-bearing part: without it the writers serialize on
/// task-spawn latency and never actually race into the repair, which is the
/// thing under test. Pool sizing is the caller's job (`TestValkey::
/// setup_with_pool_size(WRITERS)`) for the same reason — a pool smaller than
/// the burst re-serializes the race at the connection layer.
async fn create_layer_burst(ls: Arc<LayerStore>, name_prefix: &str, writers: usize) -> Vec<u64> {
    let barrier = Arc::new(Barrier::new(writers));
    let mut handles = Vec::with_capacity(writers);
    for i in 0..writers {
        let ls = Arc::clone(&ls);
        let barrier = Arc::clone(&barrier);
        let session = format!("{name_prefix}-session");
        let layer = format!("{name_prefix}-layer-{i}");
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            ls.create_layer(
                &session,
                &layer,
                "b2j-test-user",
                &minimal_state(),
                mock_ext_store(),
            )
            .await
        }));
    }

    let mut ids = Vec::with_capacity(writers);
    for handle in handles {
        ids.push(
            handle
                .await
                .expect("burst writer task must not panic")
                .expect("every concurrent create_layer must succeed through one shared repair"),
        );
    }
    ids
}

/// Sorted copy of `ids` — the burst assertions pin the exact id *set*, so they
/// need a deterministic ordering to compare against. Completion order is
/// scheduler-determined and asserting it would be asserting the scheduler.
fn sorted(ids: &[u64]) -> Vec<u64> {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v
}

// ─────────────────────────────────────────────────────────────────────────
// Scripted checkpoint store — drives the repair arms that need something to
// happen *during* the counters-sidecar read.
//
// Deliberately NOT the `test-checkpoint-fault-injection` injector: that seam's
// `maybe_inject` is synchronous and can only pass or fail, so it can neither
// sleep (the fetch-budget arm), block on a `Notify` until the test body releases
// it (the wait-budget and dropped-caller arms), perform an ordered Valkey write
// (the `repaired_noop` arm), nor panic on a chosen call (the dead-attempt arm).
// A plain decorator over the production `CheckpointStore` trait does all of them
// and needs no cargo feature.
// ─────────────────────────────────────────────────────────────────────────

/// What [`ScriptedCountersStore`] does *before* delegating
/// `get_counters_sidecar` to the inner adapter.
enum FetchBehavior {
    /// Delegate immediately. Used where only the call count matters.
    Passthrough,
    /// Sleep first. Drives the fetch-budget elapse (`delay > fetch_budget`),
    /// where the *attempt* is the thing that must give up.
    Delay(Duration),
    /// Block until the test explicitly releases the fetch, signalling `entered`
    /// on the way in.
    ///
    /// The structural form of "hold the attempt inside its sidecar read": the
    /// two tests that need an attempt to still be in flight while they assert
    /// get a guarantee rather than a timing margin. A `Delay` long enough to
    /// look safe is still a race — this suite is release-gating and runs on a
    /// shared 10-CPU VM, where a 750 ms margin is a bet on scheduler latency,
    /// and losing the bet does not fail loudly: the attempt simply finishes
    /// early and the "no verdict cell has ticked yet" assertions pass for the
    /// wrong reason, or trip and report a nonexistent regression.
    ///
    /// `entered` fires as the fetch is entered, so a test can positively
    /// confirm the attempt reached the block before acting. `release` lets it
    /// through. Both use `notify_one`, which *stores* a permit — so a test that
    /// signals before the other side waits is not a lost wakeup, and neither
    /// ordering can hang.
    BlockUntil {
        /// Signalled by the fetch as it enters the block.
        entered: Arc<Notify>,
        /// Awaited by the fetch; signal it to let the attempt proceed.
        release: Arc<Notify>,
    },
    /// Panic on the FIRST `get_counters_sidecar` call; behave as
    /// [`FetchBehavior::Passthrough`] on every later one.
    ///
    /// Models a repair attempt that dies without reporting. `tokio::spawn`
    /// swallows a panic in the spawned future, so the attempt's `watch::Sender`
    /// is dropped with no send and the singleflight slot is left holding a
    /// receiver whose value is `None` forever. Under a `None`-only join
    /// predicate that dead receiver is handed to every subsequent caller and no
    /// further attempt is ever spawned — `CreateLayer` stays wedged for the
    /// process lifetime, with nothing to restart the pod (liveness targets the
    /// bare gRPC health service, which is the same reason the runtime repair
    /// exists at all).
    ///
    /// A panic is the honest driver here rather than a convenient one: it is
    /// the only way to produce the shape from outside the module, and it is
    /// precisely what the production hazard is.
    PanicOnce,
    /// Advance `le:counter:object_id` to the given value first.
    ///
    /// A deterministic stand-in for a peer replica landing its own re-seed
    /// while this attempt's sidecar read is in flight: the attempt's opening
    /// `current_object_id()` re-check has already observed the counter absent,
    /// so it proceeds, and the conditional Lua then finds the counter at-or-
    /// above the sidecar-derived seed and no-ops. That is `SeedOutcome::NoOp`
    /// → `repaired_noop`, an arm no arrangement of pre-test state alone can
    /// reach (pre-seeding before the call instead short-circuits the re-check
    /// into `already_repaired`).
    PeerSeedsTo(Arc<CounterStore>, u64),
}

/// Wraps an [`InMemoryCheckpointStore`], counts `get_counters_sidecar` calls,
/// and applies a [`FetchBehavior`] before each one. Every other trait method
/// delegates verbatim.
struct ScriptedCountersStore {
    inner: InMemoryCheckpointStore,
    behavior: FetchBehavior,
    fetch_calls: AtomicU32,
}

impl ScriptedCountersStore {
    fn new(behavior: FetchBehavior) -> Self {
        Self {
            inner: InMemoryCheckpointStore::new(),
            behavior,
            fetch_calls: AtomicU32::new(0),
        }
    }

    fn fetch_calls(&self) -> u32 {
        self.fetch_calls.load(AtomicOrdering::SeqCst)
    }
}

#[async_trait]
impl CheckpointStore for ScriptedCountersStore {
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        let call = self.fetch_calls.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        match &self.behavior {
            FetchBehavior::Passthrough => {}
            FetchBehavior::Delay(d) => tokio::time::sleep(*d).await,
            FetchBehavior::BlockUntil { entered, release } => {
                entered.notify_one();
                release.notified().await;
            }
            FetchBehavior::PanicOnce => {
                if call == 1 {
                    // Deliberate. The panic unwinds the detached repair task,
                    // which `tokio::spawn` swallows — so this does NOT fail the
                    // test process, and the default panic hook is left alone
                    // (it is process-global, and this binary runs many tests in
                    // one process under plain `cargo test`). The message says so
                    // out loud because it WILL appear on stderr and a reader
                    // scanning a green run's output should not have to guess.
                    panic!(
                        "EXPECTED PANIC (b.b2j test fixture): FetchBehavior::PanicOnce is \
                         simulating a repair attempt that dies without reporting; this unwinds \
                         the detached repair task only and the test asserts the recovery"
                    );
                }
            }
            FetchBehavior::PeerSeedsTo(counters, value) => {
                counters
                    .seed_object_id_if_below(*value)
                    .await
                    .expect("scripted peer re-seed must succeed");
            }
        }
        self.inner.get_counters_sidecar().await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would mask a generation the
    // inner store holds rather than merely leaving the call un-scripted.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Decorator repair — drives `allocate_object_id`'s post-repair second-`None`
// arm (`failed_after_repair`).
//
// The arm fires when the counter is absent AGAIN one round trip after a repair
// that reported success: a backing store dropping confirmed writes. Reaching it
// through a decorator over the production `ObjectIdCounterRepair` trait (rather
// than a fault-injection seam inside `LayerStore`, which is what the earlier
// revision used and which has since been deleted) has three properties that
// matter:
//
// - The deletion is strictly ordered AFTER the real repair returns, so there is
//   no race to lose.
// - The allocation path itself is completely unmodified — the arm is reached by
//   the same code production runs.
// - It reproduces the production shape exactly rather than simulating its
//   symptom: a repair that genuinely succeeded against a counter that is
//   genuinely gone by the time the allocator looks again.
// ─────────────────────────────────────────────────────────────────────────

/// Delegates to the real [`CheckpointObjectIdCounterRepair`], then deletes
/// `le:counter:object_id` before returning `Ok(())`.
///
/// `pre_set` (when `Some`) writes the counter *before* delegating, which drives
/// the real attempt's opening `current_object_id() > 0` re-check into its
/// `already_repaired` short-circuit — the second upstream verdict that can
/// precede the post-repair `None`, and the one that never touches the
/// checkpoint store at all.
///
/// `observed_before_delete` records what the counter held at the instant before
/// the delete. That is the file's only window onto the exact post-repair
/// counter value (the delete destroys it a microsecond later), and it is what
/// makes the "no re-seed stomp, no burned id" claims exact rather than
/// directional.
struct RepairThenDeleteCounter {
    inner: Arc<CheckpointObjectIdCounterRepair>,
    tv_client: live_edit::storage::client::ValkeyClient,
    counter_key: String,
    pre_set: Option<u64>,
    observed_before_delete: StdMutex<Option<String>>,
    calls: AtomicU32,
}

impl RepairThenDeleteCounter {
    fn new(
        tv: &TestValkey,
        inner: Arc<CheckpointObjectIdCounterRepair>,
        pre_set: Option<u64>,
    ) -> Self {
        Self {
            inner,
            tv_client: tv.client().clone(),
            counter_key: tv.key_schema().counter_object_id(),
            pre_set,
            observed_before_delete: StdMutex::new(None),
            calls: AtomicU32::new(0),
        }
    }

    fn observed_before_delete(&self) -> Option<String> {
        self.observed_before_delete
            .lock()
            .expect("observation mutex must not be poisoned")
            .clone()
    }

    fn calls(&self) -> u32 {
        self.calls.load(AtomicOrdering::SeqCst)
    }
}

#[async_trait]
impl ObjectIdCounterRepair for RepairThenDeleteCounter {
    async fn repair_object_id_counter(&self) -> Result<(), ServiceError> {
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);

        if let Some(value) = self.pre_set {
            let _: () = self
                .tv_client
                .pool
                .set(
                    self.counter_key.clone(),
                    value.to_string(),
                    None,
                    None,
                    false,
                )
                .await
                .expect("decorator pre-set of the counter must succeed");
        }

        // Delegate to the REAL repair. Its verdict, its metric cell, its log
        // line — the decorator adds nothing to them.
        self.inner.repair_object_id_counter().await?;

        let observed: Option<String> = self
            .tv_client
            .pool
            .get(self.counter_key.clone())
            .await
            .expect("decorator read-back of the counter must succeed");
        *self
            .observed_before_delete
            .lock()
            .expect("observation mutex must not be poisoned") = observed;

        let _: () = self
            .tv_client
            .pool
            .del(self.counter_key.clone())
            .await
            .expect("decorator delete of the counter must succeed");

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 1 — the regression guard.
// ─────────────────────────────────────────────────────────────────────────

/// Full keyspace-loss cycle under a surviving process, with a real checkpoint
/// adapter wired:
///
/// 1. A counters sidecar reports `max_object_id_observed = 5000`; the startup
///    seed lifts `le:counter:object_id` to `6000`.
/// 2. `CreateLayer` mints `6001`.
/// 3. The periodic counters-sync tick is simulated by re-writing the sidecar
///    at `6001` — the durable record of "no live object id exceeds this".
/// 4. The Valkey keyspace is lost (namespaced wipe: counter key, object keys,
///    layer mapping, session — everything this store can name).
/// 5. `CreateLayer` runs again. It MUST NOT mint a colliding id.
///
/// The post-wipe id is pinned exactly (`7002 = 6001 + 1000 + 1`) rather than
/// merely bounded, because the arithmetic is the contract: the repair reads
/// the sidecar, adds `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`, and
/// allocates above that. A weaker `> oid_pre` assertion would pass on a
/// repair that seeded from, say, a hard-coded constant.
///
/// **Fails without the fix.** Pre-b.b2j, `next_object_id` was a bare `INCR`
/// with no hook and no fail-closed arm, so step 5 returns `1` — colliding
/// with the pre-wipe `6001` line that the surviving checkpoint blobs claim,
/// which is the cross-layer restore corruption this ticket closes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keyspace_loss_repairs_counter_and_mints_id_above_every_pre_wipe_id() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    // ── Step 1: sidecar + startup seed.
    put_sidecar(store.as_ref(), 5000).await;
    let outcome = seed_object_id_counter(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
    )
    .await
    .expect("startup counter seed must succeed against a healthy sidecar");
    assert!(
        matches!(
            outcome,
            SeedOutcome::Seeded {
                seed: 6000,
                new_counter: 6000
            }
        ),
        "fixture precondition: the startup seed must lift the counter to \
         max_object_id_observed + safety_margin; got {outcome:?}"
    );

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    // ── Step 2: pre-wipe CreateLayer.
    let oid_pre = ls
        .create_layer(
            "pre-wipe-session",
            "pre-wipe-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("pre-wipe create_layer must succeed");
    assert_eq!(
        oid_pre, 6001,
        "pre-wipe allocation must continue from the seeded counter"
    );

    // ── Step 3: simulate the periodic counters-sync tick landing the live
    // counter into the durable sidecar. Without this the sidecar would still
    // read 5000 and the repair would legitimately re-seed to 6000 — a
    // *stale-sidecar* scenario, which is the counters-sync loop's problem,
    // not this ticket's. Modelling the tick keeps the test pointed at the
    // reuse window rather than at an RPO gap.
    put_sidecar(store.as_ref(), oid_pre).await;

    // ── Step 4: the keyspace loss. Namespaced wipe — never FLUSHALL; see the
    // module doc-comment.
    full_valkey_wipe(&tv).await;
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: the wipe must have removed le:counter:object_id — \
         that absence is the whole signal the repair path keys off"
    );

    let repaired_pre = repair_metric(&metrics, "repaired_seeded");

    // ── Step 5: the first post-wipe CreateLayer.
    let oid_post = ls
        .create_layer(
            "post-wipe-session",
            "post-wipe-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect(
            "post-wipe create_layer must succeed — the counter is a repairable dependency \
             state, not a fatal one",
        );

    assert_eq!(
        oid_post, 7002,
        "the post-wipe allocation must resume above the sidecar-derived floor \
         (max_object_id_observed={oid_pre} + safety_margin={SAFETY_MARGIN} = 7001, \
         first allocation = 7002); got {oid_post}. A value of 1 here is the b.b2j \
         bug verbatim — a bare INCR against an absent counter, re-minting ids that \
         surviving checkpoint blobs still claim"
    );
    assert!(
        oid_post > oid_pre,
        "the post-wipe id ({oid_post}) MUST be strictly above every pre-wipe id \
         ({oid_pre}); a collision lets get_latest_layer_checkpoint hand one layer's \
         blob to another layer's identity on the DR path"
    );

    assert_eq!(
        repair_metric(&metrics, "repaired_seeded") - repaired_pre,
        1,
        "exactly one repair must have re-seeded from the counters sidecar. Every \
         increment of this metric is evidence the Valkey keyspace was lost under a \
         running pod — it is the operator's primary signal for this failure shape"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        0,
        "no repair may have reported `failed` on a healthy adapter"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "the repair stuck: the post-repair re-observation found the counter and minted \
         off it, so the second-`None` arm must not have been reached"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook IS installed, so the LayerStore fallback arm must never run — an \
         increment here means `with_object_id_counter_repair` did not take effect and \
         the counter was established at 0 without consulting any checkpoint state"
    );

    // A subsequent allocation continues the repaired line rather than
    // re-repairing: the counter now exists, so `next_object_id` takes the
    // ordinary arm.
    let oid_next = ls
        .create_layer(
            "post-wipe-session",
            "post-wipe-layer-2",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("second post-wipe create_layer must succeed");
    assert_eq!(
        oid_next, 7003,
        "once repaired, allocation must proceed normally without a second repair"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded") - repaired_pre,
        1,
        "a healthy counter must not trigger another repair — the repair is driven by \
         the allocator's absent-counter verdict, not by a schedule"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 2 — no repair hook (the `adapter=Disabled` posture).
// ─────────────────────────────────────────────────────────────────────────

/// A `LayerStore` with no repair hook, against an absent counter, must
/// establish the counter at zero and mint from `1`, ticking
/// `unhooked_initialized`.
///
/// This is the honest posture under `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled`:
/// there is no checkpoint state to consult and none to collide with, so
/// starting the id line at `1` cannot resurrect anyone's content. It is also
/// the arm every test and tool that builds a bare `LayerStore` takes, which
/// is why it repairs rather than failing — but it is still evidence of
/// keyspace loss on a configured deployment, hence the dedicated metric
/// outcome.
///
/// The second allocation pins that establishment is one-shot: the counter now
/// exists, so the fallback arm must not run again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn absent_counter_without_repair_hook_initializes_and_mints_from_one() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let ls = layer_store(&tv, &metrics, None);

    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: a fresh test prefix must have no counter key"
    );

    let first = ls
        .create_layer(
            "no-hook-session",
            "no-hook-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect(
            "create_layer with no repair hook must still succeed — the no-adapter arm \
             establishes the counter rather than failing closed",
        );
    assert_eq!(
        first, 1,
        "with no checkpoint adapter there is no durable id to avoid, so the line \
         starts at 1"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        1,
        "the unhooked fallback must be counted — on a configured deployment this cell \
         is how an operator learns the counter vanished without any adapter to re-seed \
         it from"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        0,
        "no hook is installed, so no sidecar-derived repair can have run"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "the establishment stuck: the re-observation after it found the counter, so the \
         unhooked arm must not have fallen through to the fail-closed second-`None` arm"
    );

    let second = ls
        .create_layer(
            "no-hook-session",
            "no-hook-layer-2",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("second create_layer must succeed");
    assert_eq!(
        second, 2,
        "allocation must continue from the established counter"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        1,
        "establishment is one-shot: the second allocation finds the counter present and \
         must not re-enter the fallback arm"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 3 — singleflight, witnessed through the repair's own outcome vector.
// ─────────────────────────────────────────────────────────────────────────

/// A burst of concurrent `CreateLayer`s arriving right after the wipe must
/// collapse to a single counters-sidecar read, and every caller must still
/// receive a distinct id.
///
/// The exactly-one-fetch claim is read off the outcome vector: each of
/// `repaired_seeded` / `repaired_noop` / `repaired_initialized` is emitted
/// once per completed sidecar read, while `already_repaired` is emitted by a
/// waiter that observed the winner's work and returned *without* a read. So
/// `seeded + noop + initialized == 1` is an exact fetch count.
/// [`repair_singleflight_issues_exactly_one_counters_sidecar_fetch`] pins the
/// same claim at the adapter, independent of this vocabulary.
///
/// `already_repaired` is deliberately NOT pinned to `WRITERS - 1`: a task
/// scheduled after the winner released the lock finds the counter healthy at
/// `next_object_id` and never enters the repair at all. Both shapes are
/// correct; asserting the split would be asserting the scheduler.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repair_singleflights_concurrent_create_layer_burst() {
    const WRITERS: usize = 8;

    // Pool sized to the burst so connection-pool serialization does not mask
    // the race the barrier sets up (test-writing guide, § Synchronization
    // Patterns).
    let tv = TestValkey::setup_with_pool_size(WRITERS as u32).await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    // A sidecar exists but the counter does not — exactly the post-wipe
    // state, reached here without a pre-roll so the burst is the very first
    // traffic the store sees.
    put_sidecar(store.as_ref(), 5000).await;
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: the counter must be absent so every writer races \
         into the repair"
    );

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = Arc::new(layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    ));

    let ids = create_layer_burst(ls, "burst", WRITERS).await;

    // Exact set, not a bound. The sidecar says `max_object_id_observed = 5000`
    // and the margin is 1000, so the one repair seeds the counter to exactly
    // 6001 and eight allocations off it are exactly 6001..=6008 — no gaps, no
    // duplicates, nothing above. A `> 6000` bound accepts a repair that seeded
    // from an arbitrarily larger floor and a distinctness-only check accepts a
    // sparse line; both would pass while the counter had silently jumped, which
    // is the failure mode a keyspace-loss repair is most likely to produce.
    assert_eq!(
        sorted(&ids),
        (6001..=6008).collect::<Vec<u64>>(),
        "the burst must consume exactly the 8 ids immediately above the sidecar-derived \
         floor (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN} = 6000); \
         got {ids:?}"
    );

    let seeded = repair_metric(&metrics, "repaired_seeded");
    let noop = repair_metric(&metrics, "repaired_noop");
    let initialized = repair_metric(&metrics, "repaired_initialized");
    assert_eq!(
        seeded + noop + initialized,
        1,
        "exactly ONE repair may have read the counters sidecar across the whole burst \
         (seeded={seeded}, noop={noop}, initialized={initialized}); every additional \
         read is a per-request adapter round trip the singleflight lock exists to \
         collapse"
    );
    assert_eq!(
        seeded, 1,
        "the single read must have taken the Seeded arm — a sidecar is present and the \
         counter was absent"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        0,
        "no writer may have observed a failed repair on a healthy adapter"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "every writer got an id, so no writer's post-repair re-observation may have come \
         back absent — the shared repair stuck for all of them, not just the leader"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook is installed on the shared LayerStore, so no writer may fall back to \
         establishing the counter at 0"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Tests 4 & 5 — adapter-level witnesses. Gated on
// `test-checkpoint-fault-injection`: `InMemoryCheckpointStore::
// with_fault_injector` and the `CheckpointFaultInjector` trait are compiled
// out of production library builds. A missing feature removes these tests
// SILENTLY (unlike a missing env var, which prints a SKIP line), so any
// hand-rolled narrow run must pass it explicitly.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::CheckpointFaultInjector;

/// Counts `get_counters_sidecar` invocations at the adapter, and optionally
/// fails every one of them. Non-`get_counters_sidecar` ops always pass, so a
/// `put_counters_sidecar` used to stage the fixture is never short-circuited.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct CountingCountersFetch {
    fail: bool,
    calls: AtomicU32,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl CheckpointFaultInjector for CountingCountersFetch {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == "get_counters_sidecar" {
            self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            if self.fail {
                return Err(CheckpointError::Transient {
                    source: "injected: counters sidecar unreachable".into(),
                });
            }
        }
        Ok(())
    }
}

/// The singleflight claim, witnessed at the adapter rather than derived from
/// the repair's metric vocabulary: a burst of concurrent allocations against
/// an absent counter must produce exactly ONE `get_counters_sidecar` call.
///
/// Worth having alongside
/// [`repair_singleflights_concurrent_create_layer_burst`] because that test's
/// fetch count is inferred from which outcome labels the repair chooses. If a
/// regression both dropped the lock and mislabelled the extra passes, the
/// metric-derived assertion could still read `1`; this one cannot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn repair_singleflight_issues_exactly_one_counters_sidecar_fetch() {
    const WRITERS: usize = 8;

    let tv = TestValkey::setup_with_pool_size(WRITERS as u32).await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let injector = Arc::new(CountingCountersFetch {
        fail: false,
        calls: AtomicU32::new(0),
    });
    let store: Arc<dyn CheckpointStore> = Arc::new(
        InMemoryCheckpointStore::new()
            .with_fault_injector(Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>),
    );
    put_sidecar(store.as_ref(), 5000).await;

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = Arc::new(layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    ));

    let ids = create_layer_burst(ls, "fetch-count", WRITERS).await;

    assert_eq!(
        injector.calls.load(AtomicOrdering::SeqCst),
        1,
        "the whole burst must cost exactly ONE adapter counters-sidecar fetch; got {}. \
         More than one means the singleflight lock is not collapsing waiters (each \
         post-wipe CreateLayer would pay its own checkpoint-store round trip); zero \
         means the repair never ran and the ids came from somewhere else",
        injector.calls.load(AtomicOrdering::SeqCst)
    );
    // Exact set, matching the sibling burst test: sidecar floor 6000, so the
    // eight allocations are exactly 6001..=6008. See that test for why the
    // exact set rather than a distinctness check.
    assert_eq!(
        sorted(&ids),
        (6001..=6008).collect::<Vec<u64>>(),
        "the burst must consume exactly the 8 ids immediately above the sidecar-derived \
         floor (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN} = 6000); \
         got {ids:?}"
    );
}

/// A repair that cannot reach the counters sidecar must fail the
/// `CreateLayer` with `UNAVAILABLE` and mint nothing.
///
/// This is the fail-closed end of the contract: allocating a possibly-
/// colliding id is worse than refusing the request, and `UNAVAILABLE` (rather
/// than `INTERNAL`) tells the client the state is repairable and the call is
/// worth retrying.
///
/// The counter MUST remain absent afterwards. A repair that established it at
/// zero on the way out would leave the next `CreateLayer` allocating `1`
/// against surviving checkpoint blobs — the original bug, reached through the
/// error path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn repair_failure_surfaces_unavailable_without_minting_an_id() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let injector = Arc::new(CountingCountersFetch {
        fail: true,
        calls: AtomicU32::new(0),
    });
    let store: Arc<dyn CheckpointStore> = Arc::new(
        InMemoryCheckpointStore::new()
            .with_fault_injector(Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>),
    );
    // Staged through the injector's pass-through arm — only the GET fails.
    put_sidecar(store.as_ref(), 5000).await;

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let err = ls
        .create_layer(
            "repair-fail-session",
            "repair-fail-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail when the counter is absent and the repair cannot \
             prove a floor — minting an id here is the corruption the ticket closes",
        );
    // The variant AND the message. `matches!(err, Unavailable(_))` alone is
    // satisfied by every `Unavailable` this path can produce — including the
    // wait-budget give-up and the post-repair `failed_after_repair` arm, which
    // are different operator situations with different next actions. The
    // message is the only thing that tells them apart at the client, so it is
    // part of the contract, not decoration.
    let ServiceError::Unavailable(msg) = &err else {
        panic!(
            "an unreachable counters sidecar is a transient, retriable condition and must \
             surface as Unavailable (→ gRPC UNAVAILABLE), not Internal; got {err:?}"
        );
    };
    assert!(
        msg.starts_with("object-id counter repair failed:"),
        "the caller must carry out the ATTEMPT's verdict verbatim, prefixed as a repair \
         failure — that prefix is what distinguishes this from the wait-budget give-up and \
         from the post-repair `failed_after_repair` arm, which need different operator \
         responses; got {msg:?}"
    );
    assert!(
        msg.contains("injected: counters sidecar unreachable"),
        "and it must carry the underlying checkpoint-store error through, so an operator \
         reading the client-side status learns WHICH dependency failed rather than only \
         that something did; got {msg:?}"
    );

    assert!(
        !counter_key_exists(&tv).await,
        "a failed repair must leave le:counter:object_id absent. Establishing it at 0 \
         on the error path would let the client's retry allocate 1 against surviving \
         checkpoint blobs — the b.b2j bug via the failure branch"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        1,
        "the failed repair must be counted so operators can distinguish a wedged \
         backing store from a healthy one"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        0,
        "no seed can have landed — the sidecar read never returned"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "the two alertable failure cells must stay distinct: the repair itself failed, so \
         `allocate_object_id` returned on its error and never reached the post-repair \
         re-observation. An increment here would send an operator after a backing store \
         losing writes when the actual fault is an unreachable checkpoint store"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 6 — the hooked `SeedOutcome::Absent` arm (`repaired_initialized`).
// ─────────────────────────────────────────────────────────────────────────

/// A repair hook against a checkpoint store that holds **no** counters sidecar
/// must establish `le:counter:object_id` at zero and let the `CreateLayer`
/// through, minting `1`.
///
/// This is a genuinely different arm from
/// [`absent_counter_without_repair_hook_initializes_and_mints_from_one`], and
/// the distinction is the point of the two metric cells. That test covers "no
/// adapter is configured, so there is nothing to consult". This one covers "an
/// adapter IS configured and was consulted, and it answered that nothing
/// durable has ever recorded an object id" — the once-per-deployment shape of a
/// store whose periodic counters-sync loop has not run yet (the
/// `LIVE_EDIT_CHECKPOINT_ENABLED=false` + adapter-configured combination). The
/// conclusion is the same, `1`, but only in the second case has anything
/// actually proven it safe, which is why the arm logs at WARN and books
/// `repaired_initialized` rather than the unhooked cell.
///
/// The `unhooked_initialized == 0` assertion is what keeps the two apart: it
/// fails if the hook silently did not take effect and the fallback produced the
/// same `1` for the wrong reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hooked_repair_without_a_sidecar_initializes_counter_and_mints_one() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    // Deliberately NO `put_sidecar` — an adapter that exists and is reachable
    // but has never had a counters sidecar written to it.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    assert!(
        store
            .get_counters_sidecar()
            .await
            .expect("the in-memory adapter must answer a sidecar read")
            .is_none(),
        "fixture precondition: the adapter must hold no counters sidecar, which is what \
         drives the repair to its Absent arm"
    );
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: a fresh test prefix must have no counter key"
    );

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let oid = ls
        .create_layer(
            "no-sidecar-session",
            "no-sidecar-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect(
            "create_layer must succeed: the adapter answered, and its answer was that no \
             durable checkpoint claims any object id",
        );
    assert_eq!(
        oid, 1,
        "with no counters sidecar there is no durable id to avoid, so the line starts at 1"
    );

    assert_eq!(
        repair_metric(&metrics, "repaired_initialized"),
        1,
        "the repair consulted the adapter and concluded `establish at 0` — that verdict has \
         its own cell precisely because it is NOT evidence of keyspace loss and must not be \
         alerted on alongside the cells that are"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook IS installed, so the LayerStore fallback arm must never run. An increment \
         here means the hook did not take effect and this test would be silently asserting \
         the unhooked path's identical `1` instead of the repaired one"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        0,
        "no sidecar exists, so nothing can have been seeded from one"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        0,
        "the adapter answered cleanly; a `failed` tick would mean the Absent arm's own \
         `SET NX 0` did not land"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "the establishment stuck: the post-repair re-observation found the counter and minted \
         off it"
    );

    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("1"),
        "the counter key must EXIST afterwards holding the last allocated id. Its existence \
         is what makes a future absence mean `the keyspace was lost` rather than `nobody has \
         created a layer yet` — the whole premise of the fail-closed allocator"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 7 — the hooked `SeedOutcome::NoOp` arm (`repaired_noop`).
// ─────────────────────────────────────────────────────────────────────────

/// A peer replica that lands its own re-seed **while this attempt's sidecar
/// read is in flight** drives the repair to `SeedOutcome::NoOp` →
/// `repaired_noop`, and the `CreateLayer` continues off the peer's value.
///
/// Reaching this arm needs an event ordered *between* two steps of the same
/// attempt, which is why it uses [`ScriptedCountersStore`] rather than
/// arranged-in-advance state:
///
/// - The attempt's opening `current_object_id() > 0` re-check must see the
///   counter ABSENT (otherwise it short-circuits to `already_repaired` and
///   never reads the sidecar at all — that is test 10b's arm, not this one).
/// - The conditional Lua seed that follows must see the counter at-or-above
///   the sidecar-derived seed (otherwise it advances and reports `Seeded`).
///
/// Setting the counter inside `get_counters_sidecar` is exactly that window,
/// deterministically. The peer value (9000) is chosen above the sidecar-derived
/// seed (5000 + 1000) so the no-op is unambiguous, and the resulting id pins
/// that allocation continued off the PEER's line rather than re-seeding on top
/// of it — a re-seed there would hand out 6001, an id the peer replica may
/// already have issued.
///
/// The ERROR log level on this arm is not incidental: a sidecar exists, so the
/// absence this attempt observed was real data loss and a peer simply won the
/// race to repair it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peer_seed_landing_during_the_sidecar_read_reports_repaired_noop() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::PeerSeedsTo(
        Arc::clone(&counters),
        9000,
    )));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    put_sidecar(store.as_ref(), 5000).await;
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: the counter must be absent so the attempt's opening re-check \
         proceeds to the sidecar read instead of short-circuiting"
    );

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let oid = ls
        .create_layer(
            "peer-seed-session",
            "peer-seed-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed — a peer's seed is as good as our own");

    assert_eq!(
        oid, 9001,
        "allocation must continue off the PEER's value (9000), not off a seed this attempt \
         wrote on top of it. Re-seeding to the sidecar-derived 6000 here would hand out \
         6001 — an id the peer replica may already have issued"
    );
    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("9001"),
        "and the counter itself must carry the peer's line forward, not be rewound to the \
         sidecar-derived seed"
    );

    assert_eq!(
        repair_metric(&metrics, "repaired_noop"),
        1,
        "the attempt read the sidecar and found the counter already at-or-above the derived \
         seed — that is the NoOp arm, and it is distinct from `already_repaired` (which \
         returns BEFORE any adapter read) because only this one proves a sidecar exists and \
         therefore that the absence was real data loss"
    );
    assert_eq!(
        repair_metric(&metrics, "already_repaired"),
        0,
        "the counter was absent when the attempt's opening re-check ran, so the cheap \
         short-circuit must NOT have fired — if it had, the sidecar would never have been \
         read and this test would be covering test 10b's arm by accident"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        0,
        "the conditional Lua must have left the counter alone; a `Seeded` tick means it \
         rewound the peer's value"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_initialized"),
        0,
        "a sidecar IS present, so the Absent arm is unreachable here"
    );
    assert_eq!(
        repair_metric(&metrics, "failed") + repair_metric(&metrics, "failed_after_repair"),
        0,
        "nothing failed: the adapter answered and the counter was present at the retry"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook is installed, so the LayerStore fallback arm must never run"
    );
    assert_eq!(
        scripted.fetch_calls(),
        1,
        "exactly one sidecar read — the NoOp verdict is a conclusion the attempt reaches \
         AFTER reading, not a reason to read again"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 8b — the fetch budget elapsing inside the detached attempt.
// ─────────────────────────────────────────────────────────────────────────

/// An adapter slow enough to blow the repair's **fetch budget** must fail the
/// attempt (`failed`), fail the `CreateLayer` with `UNAVAILABLE`, and mint
/// nothing.
///
/// Distinct from [`repair_failure_surfaces_unavailable_without_minting_an_id`],
/// which covers an adapter that answers with an error. Here the adapter never
/// answers at all, which is the browning-out shape the budget exists for: the
/// StorageApi adapter's own per-call ceiling is 180 s at the defaults, far
/// longer than any operator wants a repair to sit on, and abandoning a pure
/// read costs nothing because it leaves no partial durable state behind.
///
/// The message assertion is load-bearing. An elapse deliberately lands on the
/// same `SeedError::Store` arm a genuinely unreachable adapter takes — no third
/// error shape — so the *text* is the only thing that tells an operator whether
/// the checkpoint store refused or merely dawdled. Those have different fixes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_budget_elapse_fails_the_repair_without_minting_an_id() {
    // Well above the budget, well below the wait budget: the attempt must be
    // the thing that gives up, and it must do so with the caller still
    // listening so the caller reports the ATTEMPT's verdict rather than its own
    // wait-timeout.
    const SIDECAR_DELAY: Duration = Duration::from_secs(30);
    const TIGHT_FETCH_BUDGET: Duration = Duration::from_millis(100);

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::Delay(
        SIDECAR_DELAY,
    )));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    // Staged before the repair runs; only the GET is slowed.
    put_sidecar(store.as_ref(), 5000).await;

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        TIGHT_FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let err = ls
        .create_layer(
            "fetch-budget-session",
            "fetch-budget-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail closed: the repair could not prove a floor, so any id it \
             minted could collide with one a surviving checkpoint claims",
        );

    let ServiceError::Unavailable(msg) = &err else {
        panic!(
            "a checkpoint store that is merely slow is a transient, retriable condition and \
             must surface as Unavailable, not Internal; got {err:?}"
        );
    };
    assert!(
        msg.starts_with("object-id counter repair failed:"),
        "the ATTEMPT gave up, not the caller, so the caller must report the attempt's \
         verdict. A wait-budget message here would mean the two budgets are mis-ordered — \
         the caller must outlast the fetch bound; got {msg:?}"
    );
    assert!(
        msg.contains("counter-seed fetch budget"),
        "and the message must say the fetch budget elapsed rather than only that the store \
         errored — an elapse and a refusal share the `SeedError::Store` arm by design, so \
         the text is the only thing distinguishing `raise the budget / fix the latency` from \
         `the store is down`; got {msg:?}"
    );
    assert!(
        msg.contains("100ms"),
        "and it must name the budget that was actually applied, so the operator can compare \
         it against the observed adapter latency without reading the code; got {msg:?}"
    );

    assert_eq!(
        repair_metric(&metrics, "failed"),
        1,
        "an abandoned fetch is a failed repair and must be booked as one — an attempt that \
         returns without repairing must never be invisible in the outcome vector"
    );
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        0,
        "the CALLER did not time out (its budget is 100x the fetch bound); a tick here would \
         mean the wait budget is not strictly above the fetch budget and the caller is \
         racing the attempt it is supposed to outlast"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded") + repair_metric(&metrics, "repaired_noop"),
        0,
        "the sidecar bytes never arrived, so nothing can have been derived from them"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_initialized"),
        0,
        "an abandoned read is NOT the same verdict as `the store holds no sidecar` — \
         establishing the counter at 0 on an unread store is exactly the collision this \
         ticket closes"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "the repair itself failed, so `allocate_object_id` returned on its error and never \
         reached the post-repair re-observation"
    );
    assert!(
        !counter_key_exists(&tv).await,
        "a failed repair must leave le:counter:object_id absent — establishing it at 0 here \
         would let the client's retry allocate 1 against surviving checkpoint blobs"
    );
    assert_eq!(
        scripted.fetch_calls(),
        1,
        "one attempt, one fetch: abandoning the read must not retry it under the same budget"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 9 — the caller's wait budget elapsing while the attempt runs on.
// ─────────────────────────────────────────────────────────────────────────

/// The caller's **wait budget** elapsing must return `UNAVAILABLE` naming the
/// still-running repair, tick `wait_timeout`, and — the part that matters —
/// leave the attempt running to completion so it records its own outcome and
/// leaves a usable counter behind.
///
/// This is the entire justification for running the attempt on a detached task.
/// `CreateLayer` is not in `EXEMPT_FROM_DEADLINE`, so an inline repair would be
/// bounded by the tower deadline layer, which **drops** the handler future on
/// elapse — cancelling the fetch mid-flight, recording no outcome, logging no
/// diagnosis, and leaving the next request to restart the same doomed fetch
/// from zero. The failure mode the repair exists to make legible would be the
/// one it reported nothing about.
///
/// So the test asserts both halves of the split, in order:
///
/// 1. At the moment `create_layer` returns, `wait_timeout` has ticked and no
///    verdict cell has — the attempt is genuinely still in flight, not
///    finished-and-mislabelled.
/// 2. Bounded-poll until `repaired_seeded` ticks, then read the counter: the
///    abandoned attempt ran to completion, recorded its verdict, and left
///    `le:counter:object_id` at the sidecar-derived seed.
/// 3. A retry then allocates off that seed with no further repair — which is
///    what makes the `UNAVAILABLE` an honest "retry me" rather than a wall.
///
/// **Why `BlockUntil` and not a delay.** Step 1's assertions are only meaningful
/// while the attempt is genuinely mid-fetch, and a sleep sized to "comfortably
/// outlast the wait budget" makes that a timing bet. This suite is
/// release-gating and runs on a shared 10-CPU VM; losing the bet is not a loud
/// failure but a quiet inversion — the attempt finishes early, `repaired_seeded`
/// ticks before the assertion reads it, and the test either reports a regression
/// that does not exist or (worse, with a `>= 0`-shaped assertion) passes without
/// covering the arm. Holding the attempt inside `get_counters_sidecar` until the
/// test body explicitly releases it turns the margin into a guarantee: the
/// attempt cannot possibly have reported, whatever the scheduler does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wait_budget_elapse_returns_unavailable_while_the_attempt_completes() {
    const TIGHT_WAIT_BUDGET: Duration = Duration::from_millis(50);

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::BlockUntil {
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    }));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    put_sidecar(store.as_ref(), 5000).await;

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        TIGHT_WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let err = ls
        .create_layer(
            "wait-budget-session",
            "wait-budget-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail rather than mint an id while the repair's verdict is \
             still unknown",
        );

    // Positive confirmation that the attempt really is parked inside the sidecar
    // read — not merely "probably not finished yet". Everything below asserts
    // about an in-flight attempt, so this is the fact those assertions rest on.
    await_notified(
        &entered,
        "the detached repair attempt must have entered get_counters_sidecar",
    )
    .await;

    let ServiceError::Unavailable(msg) = &err else {
        panic!(
            "a repair that has not finished yet is the most retriable condition there is and \
             must surface as Unavailable, not Internal; got {err:?}"
        );
    };
    assert!(
        msg.contains("50ms wait budget"),
        "the message must name the budget the caller applied — an operator seeing repeated \
         UNAVAILABLEs needs to know whether to raise it or to fix the checkpoint store; \
         got {msg:?}"
    );
    assert!(
        msg.contains("still running"),
        "and it must say the repair is still running, because that is what makes the retry \
         advice honest: the retry joins the in-flight attempt rather than starting a second \
         one; got {msg:?}"
    );

    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        1,
        "the caller-side give-up has its own cell precisely because it is NOT a verdict on \
         the counter — booking it as `failed` would tell an operator the repair failed when \
         it may be about to succeed"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded")
            + repair_metric(&metrics, "repaired_noop")
            + repair_metric(&metrics, "repaired_initialized")
            + repair_metric(&metrics, "already_repaired")
            + repair_metric(&metrics, "failed"),
        0,
        "no verdict cell may have ticked yet — the attempt is still inside its sidecar read. \
         A tick here means the wait budget did not actually end this wait and the test is \
         not exercising the arm it claims to"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "`allocate_object_id` returned on the repair's Err and never reached the post-repair \
         re-observation"
    );

    // ── Every "still in flight" claim above is now banked, so let the attempt
    // finish. The release is explicit rather than a timer, so the boundary
    // between the two halves of this test is a program order, not a hope.
    release.notify_one();

    // ── The half a dropped-future design would lose: the attempt runs on.
    wait_for_repair_metric(&metrics, "repaired_seeded", 1).await;
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        1,
        "the abandoned attempt must not re-tick the caller-side cell on its way out"
    );
    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("6000"),
        "and it must have left the counter at the sidecar-derived seed \
         (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN}). This is the payoff of \
         the detached task: the work the caller gave up on still landed"
    );

    // ── And the retry the UNAVAILABLE invited actually works.
    let oid = ls
        .create_layer(
            "wait-budget-session",
            "wait-budget-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("the retry must succeed against the counter the abandoned attempt repaired");
    assert_eq!(
        oid, 6001,
        "the retry must allocate off the repaired seed, not off a fresh counter"
    );
    assert_eq!(
        scripted.fetch_calls(),
        1,
        "and it must not have paid for a second sidecar read — the counter now exists, so \
         `next_object_id` takes the ordinary arm and the repair is never entered"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook is installed throughout; the LayerStore fallback arm must never run"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 10 — `allocate_object_id`'s post-repair second-`None` arm
// (`failed_after_repair`).
//
// `allocate_object_id` observes the counter twice: once before the repair and
// once after it. A second `None` means the counter vanished AGAIN between a
// repair that reported success and the retry — a backing store actively
// dropping confirmed writes — and must surface `UNAVAILABLE` without minting an
// id.
//
// Both tests drive the arm through [`RepairThenDeleteCounter`], which delegates
// to the real `CheckpointObjectIdCounterRepair` and deletes the counter
// afterwards. The two differ only in which upstream verdict the real repair
// reaches first, and both are reachable in production: the sidecar-reading
// repair (`repaired_seeded`) and the cheap opening short-circuit
// (`already_repaired`), which returns without touching the checkpoint store at
// all.
// ─────────────────────────────────────────────────────────────────────────

/// A sidecar-derived repair that reports success against a counter that is gone
/// again one round trip later: `UNAVAILABLE`, `failed_after_repair`, no id, no
/// object written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_absent_again_after_successful_repair_fails_closed_without_minting() {
    const SESSION: &str = "post-repair-absent-session";
    const LAYER: &str = "post-repair-absent-layer";

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    put_sidecar(store.as_ref(), 5000).await;

    let real = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let decorated = Arc::new(RepairThenDeleteCounter::new(&tv, real, None));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(Arc::clone(&decorated) as Arc<dyn ObjectIdCounterRepair>),
    );

    let err = ls
        .create_layer(
            SESSION,
            LAYER,
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail closed. This is the most severe verdict the subsystem \
             reaches — the repair reported success and the counter is gone anyway — and \
             minting an id here would come off a counter the backing store has already \
             proven it cannot hold",
        );

    let ServiceError::Unavailable(msg) = &err else {
        panic!(
            "a backing store losing writes is still a repairable dependency state, so the \
             caller must see Unavailable (retriable), not Internal; got {err:?}"
        );
    };
    assert_eq!(
        msg,
        "object-id counter is still absent immediately after repair; the backing store is \
         losing writes",
        "this arm's message must be its OWN — it is the only one of the three Unavailable \
         shapes that indicts the backing store rather than the checkpoint store or the wait \
         budget, and it is the only one for which `retry harder` is the wrong advice"
    );

    assert_eq!(
        decorated.calls(),
        1,
        "exactly ONE repair may have been attempted. `allocate_object_id` must not loop here: \
         a retry loop only widens the window in which an allocation could come off a torn \
         counter"
    );
    assert_eq!(
        decorated.observed_before_delete().as_deref(),
        Some("6000"),
        "the counter must have held EXACTLY the sidecar-derived seed \
         (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN}) at the instant before \
         it vanished. 6001 would mean the failed allocation burned an id on its way out; any \
         other value would mean the repair seeded from something other than the sidecar"
    );

    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        1,
        "the underlying repair genuinely succeeded — that is what makes this arm distinct \
         from a repair that simply failed"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        1,
        "and the post-repair re-observation must book its own cell. The two alertable failure \
         cells point at different subsystems: `failed` at the checkpoint store, this one at \
         the backing store"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        0,
        "the repair itself did not fail; conflating the two would send an operator after the \
         wrong dependency"
    );
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        0,
        "the caller never timed out — it got a verdict, and the verdict was `Ok`"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "a hook IS installed (the decorator), so the LayerStore fallback arm must never run"
    );

    assert!(
        !counter_key_exists(&tv).await,
        "the arm must NOT re-establish the counter at 0 on its way out. Doing so would hand \
         the client's retry id 1 against surviving checkpoint blobs — the b.b2j bug, reached \
         through the one path that has already proven the store cannot be trusted"
    );

    // Nothing downstream of the allocation may have run: the allocation is
    // step 1 of `create_layer`, and every later step writes.
    assert!(
        !key_exists(&tv, tv.key_schema().obj_header("6001")).await,
        "no object header may exist for the id this call would have minted (6001) — a header \
         written against an unallocatable id is exactly the live-state clobber the \
         fail-closed allocator exists to prevent"
    );
    assert!(
        !key_exists(&tv, tv.key_schema().layer_mapping(SESSION, LAYER)).await,
        "and no layer mapping may have been written — a mapping pointing at an object that \
         was never created would surface to the next reader as a NotFound it cannot recover \
         from"
    );
}

/// The same post-repair second-`None` arm, reached through the repair's cheap
/// `already_repaired` short-circuit instead of a sidecar read.
///
/// Worth a separate test because the two upstream verdicts exercise different
/// code: `already_repaired` returns from the attempt's opening
/// `current_object_id() > 0` re-check having touched no checkpoint store at all.
/// A regression that made `allocate_object_id` trust that cheap arm — skipping
/// the post-repair re-observation because "a peer already handled it" — would
/// leave the sibling test above passing and mint a colliding id here.
///
/// The decorator's `pre_set` (4242) is deliberately NOT the sidecar-derived
/// seed (6000), so the observed-before-delete assertion doubles as proof the
/// short-circuit left the peer's value alone rather than re-seeding over it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn already_repaired_verdict_then_absent_counter_fails_closed() {
    const SESSION: &str = "already-repaired-absent-session";
    const LAYER: &str = "already-repaired-absent-layer";
    /// A peer replica's value, deliberately distinct from the sidecar-derived
    /// seed (6000) so "unchanged" and "re-seeded" are distinguishable.
    const PEER_VALUE: u64 = 4242;

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::Passthrough));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    // A sidecar EXISTS — so an assertion that it was never read is meaningful.
    put_sidecar(store.as_ref(), 5000).await;

    let real = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let decorated = Arc::new(RepairThenDeleteCounter::new(&tv, real, Some(PEER_VALUE)));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(Arc::clone(&decorated) as Arc<dyn ObjectIdCounterRepair>),
    );

    let err = ls
        .create_layer(
            SESSION,
            LAYER,
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail closed even when the repair's verdict came from the cheap \
             short-circuit — `a peer already repaired it` is a claim about the past, and the \
             re-observation is what checks it is still true",
        );

    let ServiceError::Unavailable(msg) = &err else {
        panic!("must surface as Unavailable (retriable), not Internal; got {err:?}");
    };
    assert_eq!(
        msg,
        "object-id counter is still absent immediately after repair; the backing store is \
         losing writes",
        "the arm's message must not vary with which upstream verdict preceded it — the \
         diagnosis is about the backing store either way"
    );

    assert_eq!(
        repair_metric(&metrics, "already_repaired"),
        1,
        "the attempt must have taken its opening short-circuit: the counter was above zero \
         when it looked"
    );
    assert_eq!(
        scripted.fetch_calls(),
        0,
        "and it must have done so WITHOUT reading the checkpoint store — that cheapness is \
         the whole point of the short-circuit, and a read here would mean a post-wipe burst \
         pays one adapter round trip per waiter"
    );
    assert_eq!(
        decorated.observed_before_delete().as_deref(),
        Some("4242"),
        "the short-circuit must have left the peer's value EXACTLY as it found it. 6000 here \
         would mean it re-seeded from the sidecar on top of a live peer counter, potentially \
         rewinding an id line already in use"
    );

    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        1,
        "the post-repair re-observation must fire on this verdict too — trusting \
         `already_repaired` without re-checking is exactly how a colliding id gets minted"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded")
            + repair_metric(&metrics, "repaired_noop")
            + repair_metric(&metrics, "repaired_initialized"),
        0,
        "no sidecar-derived arm may have run — the attempt returned before reaching one"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        0,
        "the repair itself succeeded; the backing store is what failed afterwards"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "a hook IS installed (the decorator), so the LayerStore fallback arm must never run"
    );

    assert!(
        !counter_key_exists(&tv).await,
        "the arm must not re-establish the counter at 0 on its way out"
    );
    assert!(
        !key_exists(&tv, tv.key_schema().obj_header("4243")).await,
        "no object header may exist for the id this call would have minted (4243)"
    );
    assert!(
        !key_exists(&tv, tv.key_schema().layer_mapping(SESSION, LAYER)).await,
        "and no layer mapping may have been written"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 11 — an attempt that dies without reporting, and the self-heal.
// ─────────────────────────────────────────────────────────────────────────

/// A repair attempt that dies without reporting must fail its caller honestly
/// — and must NOT poison the singleflight slot: the next `CreateLayer` spawns a
/// **fresh** attempt, reads the sidecar, and repairs.
///
/// `tokio::spawn` swallows a panic in the spawned future, so a panicking
/// repair attempt drops its `watch::Sender` with no send and the slot is left
/// holding a receiver whose value is `None` forever. That is why
/// `join_or_spawn_attempt`'s join predicate is two conditions rather than one
/// (`rx.borrow().is_none() && rx.has_changed().is_ok()`): `has_changed()` is
/// `Err` exactly when every sender has been dropped, which for this channel
/// means the attempt ended without sending.
///
/// **The second half is the assertion that fails against the old predicate.**
/// Under a `None`-only join, the dead receiver is handed to every subsequent
/// caller: `rx.changed()` resolves `Err` immediately, so each one takes the
/// "ended without reporting" arm, no further attempt is ever spawned
/// (`fetch_calls` stays at `1`), and `CreateLayer` stays wedged behind a repair
/// that can no longer run — for the **process lifetime**, with nothing to
/// restart the pod, since liveness targets the bare gRPC health service. The
/// first half of this test (the `UNAVAILABLE`, the `failed` tick) passes either
/// way; only the recovery distinguishes them.
///
/// Panic hygiene: the panic happens on the detached task, so it cannot fail the
/// test process, and this test deliberately does not touch the panic hook —
/// that hook is process-global and this binary runs many tests in one process
/// under plain `cargo test`, so swapping it would suppress genuine panics
/// elsewhere. The fixture's panic message announces itself instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_attempt_that_dies_without_reporting_self_heals_on_the_next_create_layer() {
    const SESSION: &str = "dead-attempt-session";

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::PanicOnce));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    put_sidecar(store.as_ref(), 5000).await;
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: the counter must be absent so the first CreateLayer enters the \
         repair"
    );

    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    // ── First CreateLayer: the attempt dies mid-fetch.
    let err = ls
        .create_layer(
            SESSION,
            "dead-attempt-layer-1",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "create_layer must fail closed: the repair produced no verdict, so no floor was \
             proven and any id minted here could collide with one a surviving checkpoint claims",
        );

    let ServiceError::Unavailable(msg) = &err else {
        panic!(
            "an attempt that died is a transient, retriable condition — the very next call \
             starts a new one — so it must surface as Unavailable, not Internal; got {err:?}"
        );
    };
    assert_eq!(
        msg, "object-id counter repair task ended without reporting an outcome",
        "this arm has its OWN message: it is the one Unavailable shape where no attempt-side \
         diagnosis exists to carry out (there is no verdict — that is the whole point), so the \
         text is all the operator gets; got {msg:?}"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        1,
        "an attempt that ends without reporting is still a failed repair and must be booked as \
         one by the waiter — nothing else can book it, because the attempt itself is gone"
    );
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        0,
        "the caller did NOT hit its wait budget (10s against an attempt that died in \
         microseconds); a tick here would mean the dead channel failed to wake the waiter and \
         it sat until its budget instead — a 10s stall on every request rather than a fast \
         retriable error"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded") + repair_metric(&metrics, "repaired_noop"),
        0,
        "the attempt never returned from the sidecar read, so nothing can have been derived \
         from it"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair"),
        0,
        "`allocate_object_id` returned on the repair's Err and never reached the post-repair \
         re-observation"
    );
    assert_eq!(
        scripted.fetch_calls(),
        1,
        "exactly one adapter read so far — the one that died"
    );
    assert!(
        !counter_key_exists(&tv).await,
        "a repair that produced no verdict must leave le:counter:object_id absent; \
         establishing it at 0 here would let the retry allocate 1 against surviving checkpoint \
         blobs"
    );

    // ── Second CreateLayer: the self-heal. THIS is what the fixed predicate
    // buys, and what a `None`-only join predicate makes impossible.
    let oid = ls
        .create_layer(
            SESSION,
            "dead-attempt-layer-2",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect(
            "the next CreateLayer must spawn a FRESH repair attempt and succeed. Failing here \
             is the wedge this fix closes: the singleflight slot still holds the dead attempt's \
             receiver, and a join predicate that only checks `value is None` hands it to every \
             later caller forever",
        );

    assert_eq!(
        scripted.fetch_calls(),
        2,
        "the second CreateLayer must have driven a SECOND adapter read. A count of 1 means no \
         new attempt was spawned — the caller joined the dead attempt's receiver instead, which \
         is the poisoned-slot wedge verbatim"
    );
    assert_eq!(
        oid, 6001,
        "and that fresh attempt must have seeded from the counters sidecar \
         (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN} = 6000), so the first id \
         off it is 6001; got {oid}"
    );
    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("6001"),
        "and the counter must carry that line forward"
    );
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded"),
        1,
        "the fresh attempt must report its own verdict — a self-heal that repaired silently \
         would leave an operator with a `failed` tick and no evidence the keyspace loss was \
         ever resolved"
    );
    assert_eq!(
        repair_metric(&metrics, "failed"),
        1,
        "and it must not have added a second failure: the only `failed` here is the dead \
         attempt's, booked by its waiter"
    );
    assert_eq!(
        repair_metric(&metrics, "failed_after_repair") + repair_metric(&metrics, "wait_timeout"),
        0,
        "the fresh repair stuck and the caller got its verdict inside the budget"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook is installed throughout; the LayerStore fallback arm must never run"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 12 — the caller's future DROPPED mid-wait (what the deadline layer
// actually does).
// ─────────────────────────────────────────────────────────────────────────

/// A `CreateLayer` future **dropped** while it waits on the repair must not take
/// the repair with it: the detached attempt runs to completion, records its own
/// outcome, and leaves the counter at the sidecar-derived seed — with
/// `wait_timeout` untouched, because nobody was left to report one.
///
/// Distinct from
/// [`wait_budget_elapse_returns_unavailable_while_the_attempt_completes`], and
/// the distinction is the production shape. That test covers the caller ending
/// its own wait and returning `UNAVAILABLE`. This one covers the caller being
/// *cancelled*: `CreateLayer` is not in `EXEMPT_FROM_DEADLINE`
/// (`src/grpc/metrics_layer.rs` exempts exactly `JoinSession` and
/// `SubscribeReadLayer`), so the tower deadline layer bounds the handler future
/// and **drops** it on elapse. Dropping is not returning — no arm of
/// `repair_object_id_counter` runs, no cell ticks, nothing is logged by the
/// caller. `tokio::time::timeout(..).await` around the call reproduces exactly
/// that: the future is dropped mid-poll.
///
/// Everything asserted below therefore belongs to the *attempt*, which is the
/// entire argument for spawning it. An earlier revision ran the fetch inline
/// under a budget that was `>= rpc_deadline_ms` by construction, so on a
/// browning-out checkpoint store the handler future was always dropped first:
/// the fetch was cancelled mid-flight, no outcome was recorded, no diagnosis was
/// logged, the client saw a bare `DEADLINE_EXCEEDED`, and the next request
/// restarted the same doomed fetch from zero. The failure mode the repair exists
/// to make legible was the one it reported nothing about — and the only way to
/// tell that design from this one is to drop the caller, which is what this test
/// does.
///
/// The attempt is held inside `get_counters_sidecar` until the test releases it,
/// so "the drop preceded the fetch completing" is structural rather than a
/// timing bet: the fetch *cannot* complete before the release, which happens
/// after the drop in program order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn caller_future_dropped_mid_wait_still_lands_the_repair_without_ticking_wait_timeout() {
    /// How long the caller is allowed to run before its future is dropped. Not
    /// a synchronization primitive — the attempt is blocked until released
    /// regardless — just a short leash so the test does not idle.
    const CALLER_LIFETIME: Duration = Duration::from_millis(20);

    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::BlockUntil {
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    }));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    put_sidecar(store.as_ref(), 5000).await;

    // WAIT_BUDGET is generous (10s) on purpose: the caller must be ended by the
    // DROP, not by its own budget. If the budget were the shorter of the two
    // this test would silently become a duplicate of the wait-budget test, and
    // the `wait_timeout == 0` assertion below is what pins the difference.
    let repair = Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        FETCH_BUDGET,
        WAIT_BUDGET,
    ));
    let ls = layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    );

    let dropped = tokio::time::timeout(
        CALLER_LIFETIME,
        ls.create_layer(
            "dropped-caller-session",
            "dropped-caller-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        ),
    )
    .await;
    assert!(
        dropped.is_err(),
        "fixture precondition: the caller future must have been DROPPED mid-wait, not \
         completed. The attempt is blocked inside its sidecar read, so a completion here means \
         the allocation never entered the repair at all"
    );

    // The attempt is parked in the fetch — and it got there without any caller
    // holding it, which is the claim.
    await_notified(
        &entered,
        "the detached repair attempt must have entered get_counters_sidecar",
    )
    .await;
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        0,
        "a dropped future returns through NO arm of `repair_object_id_counter`, so the \
         caller-side give-up cell must stay zero. A tick here would mean the test is measuring \
         the wait budget instead of the drop"
    );

    release.notify_one();

    // ── The payoff: the abandoned attempt finishes and records its verdict.
    wait_for_repair_metric(&metrics, "repaired_seeded", 1).await;
    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("6000"),
        "the counter must sit at the sidecar-derived seed \
         (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN}). Under the inline design \
         the drop cancelled this fetch mid-flight and the counter stayed absent — the next \
         request restarting the same doomed read from zero"
    );
    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        0,
        "and still no caller-side give-up on the way out: there was no caller left to give up"
    );
    assert_eq!(
        repair_metric(&metrics, "failed") + repair_metric(&metrics, "failed_after_repair"),
        0,
        "nothing failed — the attempt completed against a healthy adapter with its caller long \
         gone, which is precisely the independence the detached task buys"
    );
    assert_eq!(
        repair_metric(&metrics, "unhooked_initialized"),
        0,
        "the hook is installed; the LayerStore fallback arm must never run"
    );

    // ── And the client's retry lands on the repaired line without paying for a
    // second adapter read.
    let oid = ls
        .create_layer(
            "dropped-caller-session",
            "dropped-caller-layer",
            "b2j-test-user",
            &minimal_state(),
            mock_ext_store(),
        )
        .await
        .expect("the retry must succeed against the counter the abandoned attempt repaired");
    assert_eq!(
        oid, 6001,
        "the retry must allocate off the repaired seed, not off a fresh counter"
    );
    assert_eq!(
        scripted.fetch_calls(),
        1,
        "one keyspace-loss event, one sidecar read: the counter now exists, so `next_object_id` \
         takes the ordinary arm and the repair is never entered again"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 13 — the wait sized against the REMAINING deadline (b.hka (f)).
// ─────────────────────────────────────────────────────────────────────────

/// The server-side RPC deadline these fixtures configure, on both `Config` and
/// the `MetricsLayer` — the shipped default.
const SERVER_DEADLINE_MS: u64 = 30_000;

/// The fixed budget `SERVER_DEADLINE_MS` implies (⅔ × 30 s). Named so the
/// assertions can say "not this" about the pre-(f) value.
const FIXED_BUDGET_MS: u64 = 20_000;

/// The client's `grpc-timeout`, on the wire and as the enforced bound.
///
/// Deliberately far below the server deadline: that gap is the whole hazard.
///
/// # Sizing
///
/// The give-up is ⅔ of what is *left* when allocation is reached, so with `R`
/// remaining at that point the wait returns at `5 s − R/3` and the headroom
/// before the client bound is `R/3`. At the shipped `R ≈ 5 s` that is a give-up
/// at ≈3.3 s with ≈1.7 s of headroom.
///
/// This was 3 s, which left under a second of headroom — too thin on the shared
/// Valkey this suite runs against, where the pre-allocation work (session
/// setup, auth, upload admission, the content read, OMPF parse) is what eats
/// `R` and is exactly
/// what a loaded box slows down. Widening the client bound widens the headroom
/// proportionally without weakening anything: the hazard is the *ratio* between
/// the client bound and the 30 s server deadline (still an order of magnitude,
/// and the [`FIXED_BUDGET_MS`] fixture precondition below asserts the fixed
/// budget would still outlive it), not the absolute number.
///
/// [`CLIENT_TIMEOUT`] must stay equal to what [`CLIENT_TIMEOUT_HEADER`]
/// declares: it is the stand-in for tonic's `GrpcTimeout` (see the test's doc),
/// so the "the handler answered within the client's deadline" assertion is only
/// meaningful while the outer bound IS the deadline the client asked for.
const CLIENT_TIMEOUT_HEADER: &str = "5000m";
const CLIENT_TIMEOUT: Duration = Duration::from_millis(5000);

/// Runs the real `CreateLayer` handler as a tower service, so it can be wrapped
/// in the production [`MetricsLayer`] and driven by an ordinary `http::Request`
/// carrying a `grpc-timeout` header.
///
/// The gRPC codec and tonic's router are the only things stubbed out — every
/// other production element is real: `MetricsLayer` parses the header, computes
/// `min(server, client)`, installs the deadline scope and applies its own
/// timeout; `handle_create_layer` runs the full CreateLayer path (header
/// validation, auth, upload admission, the content read, OMPF parse, ext-value
/// storage, and
/// finally allocation); `LayerStore` reaches the deadline-aware repair. The
/// handler's `Result<_, Status>` is captured verbatim rather than re-encoded
/// into trailers, so the assertions read the real `tonic::Status` the client
/// would receive.
#[derive(Clone)]
struct CreateLayerThroughHandler {
    svc: Arc<LiveEditService>,
    session_id: String,
    layer_name: String,
    /// The handler's own verdict, captured for the test body.
    outcome: Arc<StdMutex<Option<Result<u64, tonic::Status>>>>,
}

impl<ReqBody> Service<http::Request<ReqBody>> for CreateLayerThroughHandler
where
    ReqBody: Send + 'static,
{
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _request: http::Request<ReqBody>) -> Self::Future {
        let svc = Arc::clone(&self.svc);
        let outcome = Arc::clone(&self.outcome);
        let session_id = self.session_id.clone();
        let layer_name = self.layer_name.clone();
        Box::pin(async move {
            let ompf = crate::common::owned_delta::delta_to_ompf_bytes(
                &crate::common::owned_delta::empty_delta(),
                &[],
            );
            let stream = crate::grpc::helpers::streaming_create(vec![
                crate::grpc::helpers::make_create_header(&session_id, &layer_name),
                crate::grpc::helpers::make_create_data(ompf),
            ]);
            let request = crate::grpc::helpers::make_request(stream);

            let result = handle_create_layer(&svc, request)
                .await
                .map(|response| response.into_inner().object_id);
            *outcome.lock().expect("outcome mutex poisoned") = Some(result);

            Ok(http::Response::builder()
                .status(200)
                .body(tonic::body::Body::empty())
                .expect("response construction must succeed"))
        })
    }
}

/// **A `CreateLayer` carrying a client `grpc-timeout` far shorter than the
/// server deadline still ends in a deliberate `UNAVAILABLE` naming the pending
/// repair — not a bare `DEADLINE_EXCEEDED`.**
///
/// The whole point of the caller-side wait budget is that *it*, and not the
/// deadline layer, ends the wait: a budget that returns leaves the detached
/// attempt running, ticks `wait_timeout`, and hands the client a retriable
/// error that says what it is waiting on. A budget that gets outlived instead
/// has its handler future DROPPED — no arm of `repair_object_id_counter` runs,
/// so nothing ticks and nothing is logged, and the client gets a bare
/// `DEADLINE_EXCEEDED` (that is the shape test 12 pins, deliberately).
///
/// Before b.hka (f) that ordering only held when the wait started at RPC t=0
/// against the server's own deadline. It does not: allocation is the terminal
/// step of `handle_create_layer`, and a client may set a shorter
/// `grpc-timeout` — as this test's does, [`CLIENT_TIMEOUT_HEADER`], an order of
/// magnitude below the 30 s server deadline. The fixed budget would be
/// [`FIXED_BUDGET_MS`] ms, the client bound fires first, and the deliberate
/// shape is lost exactly when a client is most explicitly asking for a fast
/// answer.
///
/// **This is what fails without the fix.** With `WaitBudget::Fixed`, the
/// `tokio::time::timeout` standing in for tonic's `GrpcTimeout` below elapses
/// first: the assertion that the call *returned* fails, and `wait_timeout` reads
/// 0.
///
/// **What is stubbed, and why that is honest.** `GrpcTimeout` — the tonic layer
/// that actually enforces `grpc-timeout` — is `pub(crate)` in tonic and sits
/// *outside* our stack, so it is modelled by the outer
/// `tokio::time::timeout(CLIENT_TIMEOUT, ..)`. Everything our code owns is
/// real: the same `MetricsLayer` `main.rs` installs parses the header, takes
/// `min(server, client)`, and publishes it; `handle_create_layer` is the
/// production handler; the `Status` asserted on is the one it produced. That
/// `MetricsLayer` publishes exactly this pair is pinned separately in
/// `src/grpc/metrics_layer.rs`'s `deadline_scope_*` tests, so the stand-in
/// carries no load-bearing claim of its own.
///
/// **What this test does NOT re-pin.** The *arithmetic* — that the effective
/// budget is ⅔ of what remains rather than ⅔ of the configured deadline, and
/// that it is re-derived on every call rather than captured once — is pinned
/// exactly, as numbers, by
/// `config::tests::config_object_id_counter_repair_wait_budget_within_narrows_to_the_remaining_deadline`
/// and
/// `checkpoint::counter_seed::tests::wait_budget_now_re_reads_the_injected_remaining_deadline_on_every_call`.
/// Here the budget cannot be predicted anyway (it depends on how long auth,
/// upload admission, the content read and the OMPF parse took), and the
/// observable consequence
/// is what matters: a 20 s fixed budget cannot possibly return inside the
/// [`CLIENT_TIMEOUT`] bound, so the call returning at all is itself proof that
/// the wait was narrowed. That, plus the `UNAVAILABLE` code, the retry claim
/// and the ticked
/// `wait_timeout`, is the whole contract this test owns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_on_a_short_client_deadline_returns_unavailable_not_deadline_exceeded() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = counter_store(&tv);

    // The repair is held inside its sidecar read until this test releases it,
    // so "the caller gave up while the attempt was still running" is structural
    // rather than a timing bet — same reasoning as test 9.
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let scripted = Arc::new(ScriptedCountersStore::new(FetchBehavior::BlockUntil {
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    }));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&scripted) as Arc<dyn CheckpointStore>;
    put_sidecar(store.as_ref(), 5000).await;

    let config = Arc::new(Config {
        rpc_deadline_ms: SERVER_DEADLINE_MS,
        ..crate::common::test_config()
    });
    assert_eq!(
        config.object_id_counter_repair_wait_budget(),
        Duration::from_millis(FIXED_BUDGET_MS),
        "fixture precondition: the FIXED budget must be the one that would outlive the client's \
         {CLIENT_TIMEOUT_HEADER} timeout, or this test is not exercising the hazard"
    );

    let repair = Arc::new(
        CheckpointObjectIdCounterRepair::new(
            Arc::clone(&store),
            Arc::clone(&counters),
            SAFETY_MARGIN,
            Arc::clone(&metrics),
            FETCH_BUDGET,
            config.object_id_counter_repair_wait_budget(),
        )
        .with_deadline_aware_wait_budget(
            Arc::clone(&config),
            Box::new(live_edit::grpc::deadline::remaining),
        ),
    );

    // The production service, with the repair-hooked LayerStore swapped in.
    let mut service = crate::grpc::helpers::build_service(
        &tv,
        Arc::new(crate::grpc::helpers::AllowAllPermissions),
    );
    service.layer_store = Arc::new(layer_store(
        &tv,
        &metrics,
        Some(repair as Arc<dyn ObjectIdCounterRepair>),
    ));

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("hka-f-session", "/stage", "b2j-test-user")
        .await
        .expect("session creation must succeed");
    assert!(
        !counter_key_exists(&tv).await,
        "fixture precondition: `le:counter:object_id` must be absent so allocation enters the \
         repair at all"
    );

    let outcome: Arc<StdMutex<Option<Result<u64, tonic::Status>>>> = Arc::new(StdMutex::new(None));
    let mut stack = MetricsLayer::new(
        Arc::clone(&metrics),
        SERVICE_LABEL,
        SERVER_DEADLINE_MS,
        2_000,
    )
    .layer(CreateLayerThroughHandler {
        svc: Arc::new(service),
        session_id,
        layer_name: "hka-f-layer".to_string(),
        outcome: Arc::clone(&outcome),
    });

    let request = http::Request::builder()
        .uri("/nvidia.omniverse.live.v1alpha.LiveService/CreateLayer")
        .header("grpc-timeout", CLIENT_TIMEOUT_HEADER)
        .body(())
        .expect("request construction must succeed");

    // The stand-in for tonic's `GrpcTimeout`. Its elapsing is the failure.
    let served = tokio::time::timeout(CLIENT_TIMEOUT, stack.call(request)).await;

    assert!(
        served.is_ok(),
        "the handler MUST answer within the client's {CLIENT_TIMEOUT_HEADER} deadline. Elapsing \
         here is the pre-b.hka(f) behaviour: the wait was sized as ⅔ of the 30s SERVER deadline \
         ({FIXED_BUDGET_MS}ms), so the client bound fired first, the handler future was DROPPED, \
         and the client got a bare DEADLINE_EXCEEDED with nothing recorded about the repair still \
         running"
    );
    served
        .expect("checked above")
        .expect("the metrics layer must not error");

    // ── The verdict the client would actually receive.
    let verdict = outcome
        .lock()
        .expect("outcome mutex poisoned")
        .take()
        .expect("the handler must have run");
    let status = verdict.expect_err(
        "allocation cannot succeed while the repair is still blocked inside its sidecar read; a \
         minted id here would mean the counter was reachable and the test covered nothing",
    );
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "a repair that has not finished yet is the most retriable condition there is, so the \
         handler MUST answer UNAVAILABLE. DEADLINE_EXCEEDED here means the deadline layer ended \
         the wait instead of the budget. Message: {:?}",
        status.message()
    );
    assert!(
        status.message().contains("still running"),
        "and it must say the repair is still running — that is what makes the retry advice \
         honest, since the retry joins the in-flight attempt rather than starting a second one; \
         got {:?}",
        status.message()
    );

    assert_eq!(
        repair_metric(&metrics, "wait_timeout"),
        1,
        "the caller-side give-up MUST be booked. A zero here with a DEADLINE_EXCEEDED is exactly \
         the b.hka (f) symptom: the handler future was dropped mid-wait, so no arm of \
         `repair_object_id_counter` ran and the give-up went unrecorded"
    );

    // Positive confirmation that the attempt really was still in flight, which
    // is what every assertion above rests on.
    await_notified(
        &entered,
        "the detached repair attempt must have entered get_counters_sidecar",
    )
    .await;
    assert_eq!(
        repair_metric(&metrics, "repaired_seeded")
            + repair_metric(&metrics, "repaired_noop")
            + repair_metric(&metrics, "repaired_initialized")
            + repair_metric(&metrics, "already_repaired")
            + repair_metric(&metrics, "failed"),
        0,
        "no verdict cell may have ticked yet — the attempt is still inside its sidecar read, so \
         the wait really was ended by the caller's budget"
    );

    // ── And the detached attempt still lands, exactly as it does under the
    // fixed budget: shrinking the CALLER's wait must not shorten the ATTEMPT's.
    release.notify_one();
    wait_for_repair_metric(&metrics, "repaired_seeded", 1).await;
    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("6000"),
        "the abandoned attempt must have left the counter at the sidecar-derived seed \
         (max_object_id_observed=5000 + safety_margin={SAFETY_MARGIN}); the deadline-aware budget \
         narrows the caller's wait only, never the attempt's own fetch budget"
    );
}
