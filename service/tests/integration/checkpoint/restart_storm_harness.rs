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

//! Restart-storm test harness — Task `t2.fgj.mh.zp`, Subtask
//! `t3.fgj.mh.zp.2w`. This module lands ONLY the scaffolding; sibling
//! Subtasks `t3.fgj.mh.zp.vy` (positive 100-restart invariant) and
//! `t3.fgj.mh.zp.d8` (negative-control `safety_margin=0` collision
//! variant) consume it.
//!
//! ## Restart-simulation fidelity — prefix-scoped wipe, NOT FLUSHALL
//!
//! Parent Task `t2.fgj.mh.zp`'s body text uses `CONFIG SET appendonly
//! no` + `FLUSHALL` as the restart-simulation primitive. This harness
//! deliberately uses a **prefix-scoped SCAN + UNLINK across every
//! `tv.key_schema().prefix()*` key plus an explicit DEL of
//! `{prefix}counter:object_id`** instead, because:
//!
//! 1. `CONFIG SET appendonly no` on the shared Valkey would affect
//!    every concurrent integration test using the same instance.
//! 2. `FLUSHALL` is a superset of the prefix wipe but also flushes
//!    other tests' state — same cross-test pollution problem.
//! 3. The simulation's TRUE intent is **"lose Valkey state across the
//!    restart while keeping checkpoint-adapter state intact"**, which
//!    the prefix wipe + counter-key delete achieves with strict
//!    test-isolation against any sibling integration test running on
//!    the same Valkey.
//!
//! The explicit `DEL {prefix}counter:object_id` is load-bearing: the
//! SCAN above is best-effort across slots and a missed `counter:*` key
//! would silently masquerade as a working AOF-on path, defeating the
//! whole point of this test. The DEL is the simulation's ground truth.
//!
//! This deviates from the Task body's literal text but preserves its
//! intent. A future maintainer reading this comment will see the
//! deviation explicitly called out rather than treating the harness as
//! drifted from the canonical Task body.
//!
//! ## Harness contract surface
//!
//! - [`RestartStormHarness`] — bundle of per-iteration state. The
//!   `TestValkey` and the LocalDisk `TempDir` survive across every
//!   `restart()` call; the `LiveEditService`, checkpoint loop, metrics
//!   registry, and counter-seed readiness signal are re-spun fresh on
//!   each restart (matching production replica-restart semantics).
//! - [`RestartStormHarness::setup_initial`] — initial spin-up. Takes
//!   `safety_margin` and `counters_sync_secs` so the negative-control
//!   Subtask can pin `safety_margin=0` while keeping the rest of the
//!   wiring identical to the positive test.
//! - [`RestartStormHarness::restart`] — the load-bearing restart
//!   method. Drains the checkpoint loop, drops the current service,
//!   prefix-wipes Valkey + DELs the counter, re-spins fresh state,
//!   runs Task 4's `seed_object_id_counter` against the same
//!   checkpoint-store root, and blocks until the readiness watch
//!   signal flips `false → true` before returning.
//! - [`RestartStormHarness::create_layer_once`] — thin CreateLayer
//!   helper that returns the minted `object_id`. Used by both
//!   per-iteration loops to drive the ~50-layers-per-iteration burst.
//! - [`RestartStormHarness::wait_for_counters_tick`] — blocks until
//!   the counters-sidecar blob's `updated_at` is strictly after a
//!   caller-supplied `Instant`. Mirrors `wait_for_checkpoint` shape
//!   but watches the [`CountersSidecar`] producer-tick surface (Task
//!   3) instead of the per-layer assembler.
//! - [`RestartStormHarness::current_counter_value`] — reads
//!   `{prefix}counter:object_id` directly via the `TestValkey` client;
//!   used by the negative-control Subtask to assert the counter
//!   advance pattern.
//!
//! ## Why Prometheus registry must be fresh per restart
//!
//! `prometheus::Registry::register_*` enforces per-name idempotency: a
//! second registration of the same family panics. Re-using the
//! registry across restart() would force every metric to be
//! constructed-once and held outside the per-restart bundle, which
//! diverges from production replica-restart semantics (a fresh replica
//! observes a fresh registry on its own startup). The harness
//! allocates a fresh `prometheus::Registry` and `SharedMetrics` on
//! each restart so the simulation matches production exactly.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use fred::interfaces::{ClientLike, KeysInterface};
use fred::types::scan::ScanType;
use futures::StreamExt;
use tempfile::TempDir;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{
    seed_object_id_counter, spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore,
    RecoveryHook, SeedError, SessionRecoveryHook, TailCaptureConfig,
};
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::counters::CounterStore;
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

use super::helpers::{
    build_recovery_hook_for_e2e, make_create_data, make_create_header,
    make_streaming_token_request, streaming_create, E2eRecoveryHookWiring, SERVICE_LABEL,
};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::helpers::{AllowAllPermissions, StubUserinfo};

/// Constant the harness drives `LIVE_EDIT_CHECKPOINT_TICK_SECS` to.
/// Mirrors the LocalDisk [`super::helpers::setup_checkpoint_harness`]
/// `tick_secs=1` convention so per-layer checkpoint passes converge
/// quickly under CI jitter. The per-iteration positive test does not
/// need per-layer checkpoint timing — only the counters-sidecar tick is
/// load-bearing — but pinning the value to 1 keeps the loop's wakeup
/// cadence cheap and matches sibling harnesses.
const CHECKPOINT_TICK_SECS: u64 = 1;

/// Bundle returned by [`RestartStormHarness::setup_initial`] and
/// re-bound on every [`RestartStormHarness::restart`] call.
///
/// The bundle is split into two field families:
///
/// **Survive-across-restart fields** — held by the harness itself, not
/// re-allocated on `restart()`:
/// - `tv` (the `TestValkey` keeping the per-test prefix stable so the
///   wipe is scoped correctly and so the restart simulates losing
///   Valkey *state* not Valkey *connection*).
/// - `tempdir` (the LocalDisk checkpoint-store root; the whole point of
///   the simulation is that the adapter blobs survive the restart).
/// - `safety_margin` and `counters_sync_secs` (the harness re-applies
///   these to Task 4's seed routine and Task 3's sidecar tick after
///   each restart).
///
/// **Per-restart-bound fields** — torn down and re-spun on each
/// `restart()`:
/// - `store` (a fresh `Arc<dyn CheckpointStore>` over the same
///   `tempdir` path).
/// - `svc` (a fresh `LiveEditService` with fresh `LayerStore` +
///   `SessionStore` + `RecoveryHook` wired against the new
///   `metrics`).
/// - `metrics` (a fresh `prometheus::Registry` + `SharedMetrics`; see
///   module-level doc for why this must be fresh per restart).
/// - `loop_handle` + `cancel` (the leader-elected checkpoint loop
///   task; cancelled and joined on restart, re-spawned after the
///   counter-seed routine has flipped the readiness signal).
/// - `counter_seed_ready_rx` (the readiness watch receiver Task 4 /
///   Task 5 use to gate; held so callers can re-check after a restart
///   if they ever need to, though `restart()` itself blocks until the
///   value is `true`).
pub struct RestartStormHarness {
    /// `TestValkey` survives across restarts so the per-test prefix
    /// stays stable. A fresh `TestValkey` would allocate a new UUID
    /// prefix and the wipe would no longer hit the keys the previous
    /// iteration wrote.
    pub tv: TestValkey,
    /// LocalDisk checkpoint-store root. The whole point of the
    /// restart-storm simulation is that the adapter blobs (the
    /// counters-sidecar in particular) survive the simulated Valkey
    /// wipe. Held in a `TempDir` so the directory survives across
    /// `restart()` and gets RAII-cleaned on harness drop.
    ///
    /// `#[allow(dead_code)]`: the consumer Subtasks read the path
    /// indirectly through `store` (which points into this directory);
    /// the field exists to keep the `TempDir` alive across restarts.
    #[allow(dead_code)]
    pub tempdir: TempDir,
    /// Safety margin Task 4's `seed_object_id_counter` passes to the
    /// conditional Lua. Re-applied on every `restart()`. The negative-
    /// control Subtask `t3.fgj.mh.zp.d8` constructs the harness with
    /// `safety_margin=0` to surface collisions.
    pub safety_margin: u64,
    /// Counters-sidecar tick cadence Task 3's loop is wired with. Held
    /// so `restart()` can re-apply the same value to the re-spun loop.
    /// The positive Subtask sets this to 1s so the producer-side blob
    /// converges quickly between iterations.
    ///
    /// `#[allow(dead_code)]`: indirect — read once on every
    /// [`Self::respin`] and stored back into the new bundle; the
    /// consumer Subtasks do not name the field directly. Field-scoped
    /// allow keeps lint clean without a crate-wide allow.
    #[allow(dead_code)]
    pub counters_sync_secs: u64,

    /// Per-restart `Arc<dyn CheckpointStore>` rooted at `tempdir`.
    /// Re-allocated on each `restart()` so the per-restart `svc`
    /// reaches the same blobs the loop wrote on the previous
    /// iteration.
    pub store: Arc<dyn CheckpointStore>,
    /// Per-restart `LiveEditService`. Tests pass this directly to
    /// `handle_create_layer` / `handle_create_session` / etc.
    pub svc: LiveEditService,
    /// Per-restart `SharedMetrics`. Re-allocated against a fresh
    /// `prometheus::Registry` on each restart — see module-level doc
    /// for the `register_*` idempotency rationale.
    ///
    /// `#[allow(dead_code)]`: the positive / negative test bodies
    /// inspect counter / histogram cells through `harness.svc.metrics`
    /// directly when needed; this top-level handle is held so the
    /// metrics outlive the loop. Field-scoped allow.
    #[allow(dead_code)]
    pub metrics: SharedMetrics,
    /// Per-restart checkpoint-loop `JoinHandle`. Cancelled +
    /// awaited under a 5s bounded timeout on every `restart()`.
    pub loop_handle: JoinHandle<()>,
    /// Per-restart `CancellationToken` paired with `loop_handle`.
    /// Fired in `restart()` to drain the loop before re-spinning.
    pub cancel: CancellationToken,
    /// Per-restart counter-seed readiness watch receiver. Mirrors the
    /// shape `main.rs` Step 3b binds (`tokio::sync::watch::Receiver<bool>`).
    /// `restart()` blocks until `*rx.borrow() == true` before returning
    /// control to the test body — matches the production startup
    /// invariant that no RPC handler accepts traffic until the gate
    /// flips.
    ///
    /// `#[allow(dead_code)]`: `restart()` consumes this internally;
    /// the test bodies do not need direct access. Holding the receiver
    /// on the bundle keeps the watch channel alive across the await
    /// boundary in `restart()` (the sender is bound to a stack local
    /// inside `restart()`; the receiver must outlive it to read the
    /// final value).
    #[allow(dead_code)]
    pub counter_seed_ready_rx: watch::Receiver<bool>,
}

impl RestartStormHarness {
    /// Spin up the initial harness state — analogous to a fresh replica
    /// boot before any restart has happened.
    ///
    /// `safety_margin`: passed verbatim into
    /// [`seed_object_id_counter`] on each restart. The positive Subtask
    /// uses `1000`, deliberately below the production default
    /// (`CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT`, 1,000,000); the
    /// negative-control Subtask uses `0` to surface collisions.
    ///
    /// `counters_sync_secs`: the cadence of Task 3's counters-sidecar
    /// tick on the leader-elected checkpoint loop. The positive Subtask
    /// uses `1s` so the producer-side blob converges quickly between
    /// iterations.
    ///
    /// `#[allow(dead_code)]`: this Subtask lands the harness;
    /// `t3.fgj.mh.zp.vy` (positive) and `t3.fgj.mh.zp.d8` (negative
    /// control) consume it.
    #[allow(dead_code)]
    pub async fn setup_initial(safety_margin: u64, counters_sync_secs: u64) -> Self {
        let tv = TestValkey::setup().await;
        let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");

        let (store, svc, metrics, loop_handle, cancel, counter_seed_ready_rx) = Self::respin(
            &tv,
            tempdir.path().to_path_buf(),
            safety_margin,
            counters_sync_secs,
        )
        .await;

        Self {
            tv,
            tempdir,
            safety_margin,
            counters_sync_secs,
            store,
            svc,
            metrics,
            loop_handle,
            cancel,
            counter_seed_ready_rx,
        }
    }

    /// The load-bearing restart method.
    ///
    /// Sequence:
    ///   1. Cancel the per-restart checkpoint loop and await its
    ///      `JoinHandle` under a 5s bounded `tokio::time::timeout`.
    ///   2. Explicitly drop the current `LiveEditService` (the loop
    ///      held the only outstanding clone of the per-restart
    ///      metrics + recovery hooks; this surfaces leaks).
    ///   3. SCAN + UNLINK every `{prefix}*` key matching the
    ///      `TestValkey` prefix. Cluster-aware via the same
    ///      `is_clustered()` branch sibling wipe helpers use; the
    ///      Subtask body specifies single-node-only (AC #4) but the
    ///      defensive branching costs nothing and matches the codebase
    ///      pattern.
    ///   4. Explicitly `DEL {prefix}counter:object_id` — load-bearing
    ///      AOF-off-restart simulation, because the sidecar at the
    ///      checkpoint-adapter root SURVIVES this wipe and a missed
    ///      counter-key delete would silently masquerade as AOF-on.
    ///   5. Call Task 4's [`seed_object_id_counter`] against the same
    ///      checkpoint-store root (the adapter still holds the
    ///      pre-restart sidecar blob) and a fresh `CounterStore`.
    ///   6. Flip the new watch sender to `true` on `Ok(_)`, leave it
    ///      `false` on `Err(_)` — same orchestrator-side decision as
    ///      `main.rs` Step 3b.
    ///   7. Block until the receiver observes `true`. Bounded under a
    ///      `timeout(Duration::from_secs(10), rx.wait_for(|v| *v))`
    ///      so a wiring regression fails the test loudly rather than
    ///      hanging the run.
    ///   8. Re-spin the leader-elected checkpoint loop against the
    ///      same `tempdir` root (same `interval=0` + `quiesce=0`
    ///      tight-cadence wiring sibling harnesses use).
    ///   9. Re-build the `LiveEditService` shell. Return the updated
    ///      `Self`.
    ///
    /// `#[allow(dead_code)]`: consumed by the two sibling Subtasks.
    /// See the module docstring for the consumption contract.
    #[allow(dead_code)]
    pub async fn restart(self) -> Self {
        // ── 1-2. Drain the old loop, drop the old service. ─────────
        self.cancel.cancel();
        let _ = timeout(Duration::from_secs(5), self.loop_handle)
            .await
            .expect("checkpoint loop must drain within the 5s shutdown budget");
        drop(self.svc);

        // ── 3. Prefix-scoped SCAN + UNLINK. ────────────────────────
        wipe_prefix_keys(&self.tv).await;

        // ── 4. Explicit DEL of the counter key. The SCAN above is
        //     best-effort and a missed counter:object_id key would
        //     defeat the simulation. The harness's contract is
        //     "lose Valkey state across the restart"; this DEL is
        //     the ground truth of that contract. ────────────────────
        let counter_key = self.tv.key_schema().counter_object_id();
        let _: () = self
            .tv
            .client()
            .pool
            .del(counter_key)
            .await
            .expect("DEL {prefix}counter:object_id must succeed");

        // ── 5-8. Re-spin against the same tempdir. ─────────────────
        let (store, svc, metrics, loop_handle, cancel, counter_seed_ready_rx) = Self::respin(
            &self.tv,
            self.tempdir.path().to_path_buf(),
            self.safety_margin,
            self.counters_sync_secs,
        )
        .await;

        Self {
            tv: self.tv,
            tempdir: self.tempdir,
            safety_margin: self.safety_margin,
            counters_sync_secs: self.counters_sync_secs,
            store,
            svc,
            metrics,
            loop_handle,
            cancel,
            counter_seed_ready_rx,
        }
    }

    /// Shared spin-up routine, called once from `setup_initial` and
    /// again from every `restart()`. Returns the per-restart bundle
    /// fields; the caller stitches them onto the survive-across-restart
    /// state (`tv`, `tempdir`, `safety_margin`, `counters_sync_secs`).
    ///
    /// Routine: build a fresh `Arc<dyn CheckpointStore>` rooted at the
    /// supplied `local_disk_path`, run Task 4's
    /// `seed_object_id_counter` against it, flip the readiness watch
    /// sender on `Ok(_)`, wait for the receiver to observe `true`
    /// under a bounded timeout, then spawn the leader-elected
    /// checkpoint loop and build the `LiveEditService` shell. The
    /// counter-seed routine MUST complete before the service is built
    /// so the order matches production (`main.rs` Step 3b ≪ Step
    /// `serve_with_shutdown`).
    async fn respin(
        tv: &TestValkey,
        local_disk_path: PathBuf,
        safety_margin: u64,
        counters_sync_secs: u64,
    ) -> (
        Arc<dyn CheckpointStore>,
        LiveEditService,
        SharedMetrics,
        JoinHandle<()>,
        CancellationToken,
        watch::Receiver<bool>,
    ) {
        let local_disk_path_str = local_disk_path.to_string_lossy().into_owned();
        let store: Arc<dyn CheckpointStore> = Arc::new(
            LocalDiskCheckpointStore::new(local_disk_path)
                .expect("LocalDiskCheckpointStore::new must succeed on a re-spin root"),
        );

        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

        // ── Counter-seed startup recovery + readiness watch ────────
        //
        // Mirrors the production Step 3b in main.rs: build a
        // CounterStore, allocate a watch::channel(false), call the
        // routine, flip on Ok, leave at false on Err. The wait below
        // is bounded so a wiring regression that never flips the gate
        // fails the test loudly rather than hanging.
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let (counter_seed_ready_tx, counter_seed_ready_rx) = watch::channel(false);

        let seed_result = seed_object_id_counter(
            Arc::clone(&store),
            Arc::clone(&counters),
            safety_margin,
            Arc::clone(&metrics),
        )
        .await;

        match seed_result {
            Ok(outcome) => {
                tracing::debug!(?outcome, "restart-storm respin: counter-seed succeeded");
                counter_seed_ready_tx
                    .send(true)
                    .expect("watch send on Ok path must succeed");
            }
            Err(SeedError::ParseFailed(e)) => {
                // The negative-control test does not exercise this
                // arm — surface as a hard assertion so a future
                // regression that accidentally corrupts the sidecar
                // bytes is loud instead of silently leaving the gate
                // at false.
                panic!(
                    "respin: counter-seed routine returned ParseFailed unexpectedly: {e}. \
                     The restart-storm harness only writes well-formed CountersSidecar \
                     blobs through the Task-3 producer tick; a ParseFailed here means \
                     either the producer wrote corrupted bytes or the harness's bundle \
                     is observing a foreign sidecar."
                );
            }
            Err(SeedError::Store(e)) => {
                panic!(
                    "respin: counter-seed routine returned CheckpointStore error: {e}. \
                     The harness re-uses the same LocalDisk tempdir across restarts; \
                     a Store-error here means the adapter wiring broke between \
                     setup_initial and the first restart."
                );
            }
            Err(SeedError::Seed(e)) => {
                panic!(
                    "respin: counter-seed routine returned CounterStore seed error: {e}. \
                     The harness wipes le:counter:object_id between restarts; a Seed \
                     error here means the EVAL itself failed (Valkey transport / Lua \
                     reply parse)."
                );
            }
        }

        // Block until the receiver observes true. Bounded — a wiring
        // regression that never flips the gate must fail loudly.
        let mut rx_clone = counter_seed_ready_rx.clone();
        timeout(Duration::from_secs(10), async {
            if *rx_clone.borrow() {
                return;
            }
            rx_clone
                .changed()
                .await
                .expect("watch sender must outlive the await");
        })
        .await
        .expect(
            "counter-seed readiness watch must flip false → true within 10s of the routine \
             returning Ok(_). A timeout here means the routine returned Err and the \
             orchestrator-side flip never fired — see the inline panic arms above.",
        );

        // ── Stores + recovery hook ─────────────────────────────────
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let session_store = Arc::new(SessionStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));

        // RecoveryHook + SessionRecoveryHook: built against a Config
        // shaped to match the harness wiring. We re-derive `Config`
        // from `test_config()` + the checkpoint-specific knobs to keep
        // the field-rename-surfaces-as-compile-error invariant.
        let cfg = harness_config(local_disk_path_str);

        // Hoisted above the hook build so the SAME `Arc`s reach the hook and
        // the service below — `main.rs` clones one object into each, and a
        // hook/service split would leave writes landing in one ext store while
        // reads come out of the other. The bootstrap is the disabled stub this
        // file already used: this harness runs with the tail flag off and
        // asserts nothing about a re-base.
        let ext_value_store: Arc<dyn ExternalValueStore> = Arc::new(MockExternalValueStore::new());
        let base_bootstrap = crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        );

        let recovery: Arc<RecoveryHook> = build_recovery_hook_for_e2e(
            Arc::clone(&layer_store),
            Arc::clone(&store),
            cfg.clone(),
            Arc::clone(&metrics),
            E2eRecoveryHookWiring {
                ext_value_store: Arc::clone(&ext_value_store),
                base_bootstrap: Some(Arc::clone(&base_bootstrap)),
            },
        );
        let session_recovery: Arc<SessionRecoveryHook> = Arc::new(SessionRecoveryHook::new(
            Arc::clone(&store),
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(cfg),
            Arc::clone(&metrics),
        ));

        // ── LiveEditService shell ──────────────────────────────────
        let svc = LiveEditService {
            catchup_snapshot_enabled: false,
            session_store,
            layer_store: Arc::clone(&layer_store),
            ext_value_store,
            session_tokens: Arc::new(DashMap::new()),
            userinfo_client: Arc::new(StubUserinfo),
            permissions_client: Arc::new(AllowAllPermissions),
            max_layer_size: 10 * 1024 * 1024,
            upload_admission: std::sync::Arc::new(
                live_edit::grpc::upload_admission::UploadAdmissionGate::new(
                    Default::default(),
                    None,
                ),
            ),
            snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
                tv.client().clone(),
                tv.key_schema().clone(),
            ),
            valkey: tv.client().clone(),
            key_schema: tv.key_schema().clone(),
            presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
            presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs:
                live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms:
                live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
            subscription_tracker: tokio_util::task::TaskTracker::new(),
            shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
            metrics: Arc::clone(&metrics),
            service_label: SERVICE_LABEL.to_string(),
            session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new()
                .with_recovery_hook(Arc::clone(&session_recovery)),
            recovery,
            session_recovery,
            base_bootstrap,
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
        };

        // ── Leader-elected checkpoint loop. Same tight-cadence
        //     `interval=0` + `quiesce=0` wiring sibling harnesses use,
        //     but with the caller-supplied `counters_sync` cadence so
        //     the producer-side sidecar tick is fast enough for the
        //     restart-storm budget. ────────────────────────────────
        let loop_config = CheckpointLoopConfig {
            tick: Duration::from_secs(CHECKPOINT_TICK_SECS),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(counters_sync_secs.max(1)),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(SERVICE_LABEL),
            // Tails pinned explicitly OFF. b.3n1 promoted the binary default
            // to `true`; this suite predates the tail feature and was written
            // against an inert tail path, so the flag is part of the fixture
            // rather than something to inherit.
            tail: TailCaptureConfig {
                enabled: false,
                ..TailCaptureConfig::default()
            },
        };
        let cancel = CancellationToken::new();
        let loop_handle = spawn_checkpoint_loop(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&store),
            loop_config,
            Some(Arc::clone(&metrics)),
            cancel.clone(),
        );

        (
            store,
            svc,
            metrics,
            loop_handle,
            cancel,
            counter_seed_ready_rx,
        )
    }
}

/// Thin helper that issues one CreateLayer (empty delta) and returns
/// the freshly-minted `object_id`. Used by both sibling Subtasks to
/// drive the per-iteration ~50-layers burst.
///
/// The delta is `OwnedDelta::default()` — Task `t2.fgj.mh.zp` only
/// cares about `object_id` allocation, not delta content; an empty
/// delta keeps the OMPF payload minimal and the loop's pre-checkpoint
/// work negligible.
///
/// `client_id` on `CreateLayer` is not part of the header (it lives on
/// `UpdateLayer` only), so the caller passes just `session_id` and
/// `layer_name`.
///
/// `#[allow(dead_code)]`: consumed by the two sibling Subtasks.
#[allow(dead_code)]
pub async fn create_layer_once(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
) -> Result<u64, tonic::Status> {
    let ompf = delta_to_ompf_bytes(&OwnedDelta::default(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(ompf),
    ]);
    let request = make_streaming_token_request(stream);
    let response = handle_create_layer(svc, request).await?;
    Ok(response.into_inner().object_id)
}

/// Block until the counters-sidecar blob's `updated_at` is strictly
/// after `after`, or panic with a diagnostic on deadline expiry.
///
/// Mirrors [`super::helpers::wait_for_checkpoint`]'s shape but watches
/// the [`CountersSidecar`] surface from Task 3 (`src/checkpoint/
/// counters.rs`) instead of the per-layer assembler. Compares
/// `sidecar.updated_at` (RFC3339) against `after` using `chrono`
/// parsing so a clock-skew or formatting drift on the producer side
/// surfaces cleanly.
///
/// `deadline` is taken as a `Duration` (not seconds) so the positive
/// Subtask can pass `Duration::from_secs(5)` directly without a
/// per-call `.as_secs()` shuffle.
///
/// `#[allow(dead_code)]`: consumed by the positive Subtask only — the
/// negative-control Subtask explicitly skips this wait per its body's
/// "the collision the negative control demonstrates depends on the
/// counter advancing FASTER than the sidecar tick" rationale.
#[allow(dead_code)]
pub async fn wait_for_counters_tick(
    harness: &RestartStormHarness,
    after: Instant,
    deadline: Duration,
) {
    // `Instant` is monotonic and incomparable across process / wall-
    // clock contexts, so we record a parallel `chrono::DateTime` at the
    // same moment as the caller's `after` Instant. Practical accuracy:
    // the gap between this `Utc::now()` and the caller's `after =
    // Instant::now()` is the few hundred microseconds it takes to call
    // this helper, which is well below the Task-3 producer tick's
    // second-granularity cadence.
    //
    // We map `after` onto wall time by computing how long ago it
    // happened (`now_inst - after`) and subtracting from current wall
    // time. Negative gaps (caller passed a future Instant) clamp to
    // "now" — the producer must produce a sidecar with `updated_at >=
    // chrono_now` for the loop below to return.
    let now_inst = Instant::now();
    let now_wall = chrono::Utc::now();
    let after_wall = if now_inst >= after {
        now_wall - chrono::Duration::from_std(now_inst - after).unwrap_or_default()
    } else {
        now_wall
    };

    let started = Instant::now();
    let poll_interval = Duration::from_millis(50);
    loop {
        match harness.store.get_counters_sidecar().await {
            Ok(Some(bytes)) => {
                if let Ok(sidecar) = serde_json::from_slice::<CountersSidecar>(&bytes) {
                    if let Ok(updated_at) =
                        chrono::DateTime::parse_from_rfc3339(&sidecar.updated_at)
                    {
                        let updated_utc = updated_at.with_timezone(&chrono::Utc);
                        if updated_utc >= after_wall {
                            return;
                        }
                    }
                }
            }
            Ok(None) => {
                // Sidecar not yet written by the leader-elected loop
                // — keep polling. This is the dominant pre-tick state
                // on the very first iteration before the producer has
                // had a chance to fire.
            }
            Err(e) => {
                panic!(
                    "get_counters_sidecar failed during wait_for_counters_tick: {e}. \
                     The restart-storm harness assumes the LocalDisk adapter is \
                     reachable; a transport error here means the tempdir was \
                     deleted out from under us or the adapter wiring broke."
                );
            }
        }
        if started.elapsed() >= deadline {
            let observed = match harness.store.get_counters_sidecar().await {
                Ok(Some(bytes)) => format!(
                    "sidecar bytes present ({} bytes) but updated_at not yet > after_wall",
                    bytes.len()
                ),
                Ok(None) => "no sidecar yet — producer tick has never fired".to_string(),
                Err(e) => format!("get_counters_sidecar errored: {e}"),
            };
            panic!(
                "wait_for_counters_tick timed out after {:?} (after_wall={after_wall:?}). \
                 Observed state: {observed}. Common causes: the Task-3 producer tick is \
                 not firing (counters_sync_secs too long), the harness's harness restart \
                 did not respawn the loop, or the leader-lock acquisition is hanging.",
                deadline
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Read `{prefix}counter:object_id` directly via the `TestValkey`
/// client. Returns `0` for the absent / pre-first-INCR state — matches
/// the `CounterStore::current_object_id` contract.
///
/// Used by the negative-control Subtask `t3.fgj.mh.zp.d8` to assert
/// the counter advance pattern across iterations without going through
/// the `LiveEditService` shell. The positive Subtask `t3.fgj.mh.zp.vy`
/// also uses this post-restart to assert the seed routine landed the
/// counter at `>= per_iter_max + safety_margin`.
///
/// `#[allow(dead_code)]`: consumed by both sibling Subtasks.
#[allow(dead_code)]
pub async fn current_counter_value(tv: &TestValkey) -> u64 {
    let key = tv.key_schema().counter_object_id();
    let raw: Option<String> = tv
        .client()
        .pool
        .get(key)
        .await
        .expect("GET {prefix}counter:object_id must succeed");
    match raw {
        None => 0,
        Some(s) => s
            .parse::<u64>()
            .expect("counter value must parse as u64; non-numeric reply is a contract violation"),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Internal helpers — not exported. Used by setup_initial / restart and
// by the public `create_layer_once` helper above.
// ──────────────────────────────────────────────────────────────────────

/// Build the `Config` the restart-storm harness threads into the
/// recovery hook. Mirrors `super::helpers::harness_config` (which is
/// private to that module) but pinned to the LocalDisk adapter with
/// the path threaded in from the harness's `tempdir`.
fn harness_config(local_disk_path: String) -> live_edit::config::Config {
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::LocalDisk;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = CHECKPOINT_TICK_SECS;
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_local_disk_path = Some(local_disk_path);
    cfg
}

/// SCAN + UNLINK every key matching `{prefix}*`. Cluster-aware via the
/// same `is_clustered()` branch [`super::helpers::wipe_layer_state`]
/// uses. Mirrors `src/storage/test_support.rs::cleanup_keys` shape but
/// scoped to the test prefix.
async fn wipe_prefix_keys(tv: &TestValkey) {
    let pattern = format!("{}*", tv.key_schema().prefix());
    let client = tv.client().pool.next();
    let is_clustered = client.is_clustered();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if is_clustered {
        Box::pin(client.scan_cluster_buffered(pattern.clone(), Some(100), None::<ScanType>))
    } else {
        Box::pin(client.scan_buffered(pattern.clone(), Some(100), None::<ScanType>))
    };
    let mut to_delete = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(key) = item {
            to_delete.push(key);
        }
    }
    if to_delete.is_empty() {
        return;
    }
    if is_clustered {
        // Cluster mode: a multi-key UNLINK across hash slots fails with
        // CROSSSLOT. Group by slot and issue one UNLINK per group —
        // same discipline as `super::helpers::full_valkey_wipe`. AC #4
        // is single-node-only but the branching costs nothing.
        let mut by_slot: std::collections::HashMap<u16, Vec<fred::types::Key>> =
            std::collections::HashMap::new();
        for key in to_delete {
            let slot = fred::util::redis_keyslot(key.as_bytes());
            by_slot.entry(slot).or_default().push(key);
        }
        for (_slot, group) in by_slot {
            let _: () = client
                .unlink(group)
                .await
                .expect("UNLINK of per-slot prefix keys must succeed");
        }
    } else {
        let _: () = tv
            .client()
            .pool
            .unlink(to_delete)
            .await
            .expect("UNLINK of prefix keys must succeed");
    }
}

// ──────────────────────────────────────────────────────────────────────
// Streaming-framing helpers are imported from
// `tests/integration/checkpoint/helpers.rs` via the existing
// `use super::helpers::...` line above (b.dzd Phase 1 deduplication).
// ──────────────────────────────────────────────────────────────────────
