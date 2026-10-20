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

//! Bounded bootstrap concurrency — acquire-or-shed, one bound across BOTH call
//! sites. Task `t2.dfh.yj.4f`, Subtask `t3.dfh.yj.4f.yd` (Epic-3 interim hazard
//! #4, recorded on `t1.dfh.57`).
//!
//! ## What is under test
//!
//! `BaseBootstrap` hands its spawned task a full `ObjectState` and the task then
//! builds an OMPF from it, so N concurrent bootstraps hold N × (state + OMPF)
//! resident until each PUT resolves — and a degraded adapter's PUT can sit for
//! the 180 s per-call ceiling. The landed answer is an admission semaphore at the
//! top of `BaseBootstrap::run_once` that **sheds rather than queues**
//! (`try_acquire_owned`, never `acquire().await`): a queued task would still be
//! holding the state the bound exists to cap, so queueing would convert a memory
//! bound into a latency bound and close nothing.
//!
//! Three properties, and this file is the integration-level evidence for all
//! three:
//!
//! 1. **The bound is honoured.** No more than the bound may be in the PUT at
//!    once, however many triggers fire together.
//! 2. **A shed is a shed, not a deferral.** After a full drain the total PUT
//!    count is the bound, not the trigger count. This is the timing-free
//!    discriminator against a queueing implementation, which eventually issues
//!    all N.
//! 3. **ONE bound spans both call sites.** Production constructs exactly one
//!    `Arc<BaseBootstrap>` (`main.rs`) and hands the same `Arc` to the gRPC
//!    service (`LiveEditService::base_bootstrap`, creation-time) and to
//!    `RecoveryHook::with_base_bootstrap` (restore-time). Case 4 mirrors that
//!    single-instance wiring exactly — two `Arc::clone`s of ONE unit — because a
//!    two-instance fixture would be testing a wiring shape production does not
//!    have and would fail against a correct implementation.
//!
//! ## The bound is READ, never spelled
//!
//! [`BOOTSTRAP_MAX_INFLIGHT`] is a `pub const` re-exported through
//! `live_edit::checkpoint`, and it is the only source of truth this file uses.
//! No case writes a literal `2`. A test with a hardcoded bound keeps passing
//! after the bound changes, which is exactly the regression this file exists to
//! catch. `BaseBootstrap::with_max_inflight` (an ungated test affordance;
//! production never calls it) shrinks the bound for the cases that need the shed
//! arm reachable with fewer parked PUTs — and even there the *smallest accepted*
//! value is derived ([`SMALLEST_ACCEPTED_BOUND`]), not asserted to be 2's
//! neighbour.
//!
//! ## The landed metric posture, PINNED (case 6)
//!
//! A shed ticks exactly ONE cell, and it is on the family Epic 5's SR-7
//! adjudication (`t2.dfh.be.s8`) minted for this unit:
//! `live_edit_checkpoint_base_bootstrap_total{outcome="shed",
//! site="creation_base"}`. It moves **no** `checkpoint_writes_total` cell and
//! **no** `checkpoint_tail_total` cell. The WARN is no longer the shed's only
//! evidence — which is what changed here, and the ONE fence in this file that the
//! adjudication was designed to flip.
//!
//! `checkpoint_writes_total` deliberately did NOT change, and that is a verdict
//! rather than an omission — a future reader should not "fix" it. That family
//! counts GENERATION-COMPLETION outcomes fleet-wide: `checkpointer.rs` ticks
//! `{success, layer}` only after a cursor stamp actually lands, `{error, layer}`
//! on the blob-landed / stamp-failed shape (`rebase_on_chain_full`'s pre-stamp
//! `HMGET` arm), and `{skipped, layer}` to record that a snapshot FORM was
//! skipped by a tick that did run (`run_one_tick`'s not-quiesced arm) — that is
//! the cross-producer precedent. A shed completes no generation and never
//! reaches the adapter at all, so folding it onto `skipped` there would file an
//! ADMISSION decision in a completion counter and contaminate every
//! `success/(success+skipped+error)` ratio an operator builds over
//! `kind="layer"`. The two families are additive, never substitutes;
//! `base_bootstrap.rs`'s module header carries the same statement from the
//! producer's side.
//!
//! [`shed_ticks_only_the_bootstrap_outcome_cell`] asserts all of it in both
//! directions, on both families: `checkpoint_writes_total{success, layer}` moves
//! by exactly the number of ADMITTED bootstraps and every other cell of that
//! family is byte-identical, its rendered `result` × `kind` vocabulary is
//! unchanged, no `live_edit_checkpoint_tail_total{outcome}` cell moves, and on
//! the bootstrap family `{shed, creation_base}` moves by exactly the shed count
//! while `{persisted, creation_base}` moves by exactly the admitted count and
//! every other cell of THAT family — the whole `restore_rebase` column above all
//! — is byte-identical too. Pinning both directions is the point: the shed's one
//! signal and the other two families' silence are each *what landed*, not
//! incidental.
//!
//! ## Why this module is UNGATED
//!
//! No `#![cfg(feature = "test-checkpoint-fault-injection")]`, deliberately, for
//! the same reason its two siblings are ungated: a gated module compiles to
//! nothing and reports green having covered nothing
//! (`docs-internal/test-writing-guide.md`), and this file is the only evidence
//! that hazard #4 was actually closed. The gating/recording double is a
//! hand-rolled [`DelegatingCheckpointStore`] newtype
//! ([`ConcurrencyProbeStore`]) — the repo's documented preference over the
//! `CheckpointFaultInjector` seam, and the only shape that can carry an in-flight
//! high-water instrument at all.
//!
//! ## Why the instrument is a peak counter and never a clock
//!
//! Every claim here is proven by two structural instruments —
//! `ConcurrencyProbeStore`'s `max_observed_inflight` high-water mark and its
//! total PUT count — and by `BootstrapOutcome` arms. There is **no wall-clock
//! assertion anywhere in this file**: timing assertions never belong in a gating
//! suite (`docs-internal/test-writing-guide.md` § Performance-Sensitive Tests),
//! and "it finished fast" is not evidence that a bound was enforced. The gate is
//! stimulus, not measurement — it holds admitted PUTs open so that a *structural*
//! overlap can be observed, and the peak counter is what observes it.
//!
//! The peak counter is demonstrably non-vacuous rather than trivially satisfied:
//! [`concurrent_creation_bootstraps_never_exceed_the_inflight_bound`] and
//! [`one_unit_shares_its_bound_across_both_call_sites_at_the_production_bound`]
//! both assert the peak REACHED the bound (`== bound`, not merely `<= bound`), so
//! a probe that could never see overlap would fail them.
//!
//! ## Why a separate file from the SR-3.6 lane
//!
//! The bound lives in shared `run_once` and is reachable from both call sites,
//! and its fixture is a concurrency probe rather than a restore fixture:
//! `restore_rebase_test.rs`'s fixture plants a lineage, wipes layer state and
//! drives a real `RecoveryHook::try_restore_layer` per case, none of which a
//! peak-inflight probe needs or wants. The restore-shaped triggers here are
//! header-stamped rather than restored (see [`ProbeFixture::stamp_restore_shaped_header`])
//! precisely because this file makes no claim about restore mechanics — the
//! sibling owns those — only about admission.
//!
//! **No existing test file's assertions were modified for this Subtask.** If one
//! had had to be, that is an escalation signal, not something to absorb: a bound
//! that alters single-threaded creation-time behaviour is a defect.
//!
//! ## Bounded waits — no `sleep` is used for synchronization
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | [`ConcurrencyProbeStore`] gate park | [`GATE_WAIT`] (10 s) | the test never released the gate; the PUT then proceeds so the case fails on an assertion rather than hanging |
//! | [`poll_until`] — "the admitted PUTs are parked and every shed trigger has returned" | [`ADMISSION_POLL_BUDGET`] (10 s) | fewer bootstraps were admitted than the bound allows (a spurious shed), or a trigger never reached `try_acquire_owned` |
//! | [`ProbeFixture::drain`] (terminal `close()` + `wait()`) | [`DRAIN_TIMEOUT`] (10 s) | a bootstrap task never completed, or was spawned onto a different tracker |
//!
//! Every wait is a deadline, not a sleep: the healthy path returns in
//! milliseconds and pays none of it. Worst case for the heaviest test (case 5,
//! the only one holding all three) is 10 + 10 + 10 = 30 s, comfortably inside
//! `.config/nextest.toml`'s 120 s hard kill — a SIGKILLed test prints none of its
//! diagnostics, which is exactly when they are needed.
//!
//! Sandbox env note: `TestValkey::setup*` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).
//!
//! [`DelegatingCheckpointStore`]: super::helpers::DelegatingCheckpointStore

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tokio::sync::{watch, Barrier};
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    BaseBootstrap, BootstrapOutcome, BootstrapTarget, CheckpointAdapter, CheckpointBlob,
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore,
    BOOTSTRAP_MAX_INFLIGHT,
};
use live_edit::config::{Config, PodMemorySource};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::{
    Metrics, SharedMetrics, CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES, CHECKPOINT_BASE_BOOTSTRAP_SITES,
    CHECKPOINT_WRITE_KINDS,
};
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    gather_base_bootstrap_cells, gather_tail_outcome_cells, DelegatingCheckpointStore,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Replica id stamped into every persisted `CheckpointMeta` and carried as this
/// file's registry label. Note `checkpoint_writes_total` renders NO `service`
/// label, so it is not a metric selector here — cross-test isolation comes from
/// the per-fixture registry (`crate::common::metrics_scrape_only`).
const SERVICE_LABEL: &str = "bootstrap-concurrency-bound-test-replica";

/// The smallest bound `BaseBootstrap::with_max_inflight` accepts.
///
/// Derived from the landed clamp rather than asserted: `with_max_inflight` does
/// `max_inflight.max(1)`, because `0` would shed unconditionally — not a state
/// any caller wants and not one production can reach.
/// [`with_max_inflight_zero_clamps_to_the_smallest_accepted_bound`] pins the
/// clamp behaviourally, so this constant cannot drift silently.
const SMALLEST_ACCEPTED_BOUND: usize = 1;

/// The shrunk bound must be a value the production bound could itself take, or
/// the three cases built on it would be probing a configuration
/// `with_max_inflight` does not model. Compile-time, so shrinking
/// [`BOOTSTRAP_MAX_INFLIGHT`] below the shrunk bound is a build failure rather
/// than a silently-degenerate test run.
const _: () = assert!(SMALLEST_ACCEPTED_BOUND <= BOOTSTRAP_MAX_INFLIGHT);

/// `snap_seq` every creation-time bootstrap generation lands at
/// (`BootstrapTarget::CreationBase::snap_seq`). Named so the assertions read as
/// the contract rather than as a magic `1`.
const CREATION_SNAP_SEQ: u64 = 1;

/// The `last_checkpoint_seq` a restore's own audit stamp leaves behind — i.e.
/// the generation the restore restored. Any value below
/// [`REBASE_SEEDED_SEQ`] works; the pairing with the band below is what matters.
const REBASE_SNAP_SEQ: u64 = 40;

/// The guard band a restore adds on top of `restore_floor`, stamped into
/// `obj:hdr restore_seq_bump`. Non-zero because `band >= 1` unconditionally
/// post-b.8y5 — a zero band would make the restore-shaped header a shape no
/// restore can produce.
const REBASE_BAND: u64 = 50;

/// `seeded_seq` for the restore-shaped triggers — `restore_floor + band`, and
/// the live header's `seq` at the moment such a restore committed. Above
/// [`REBASE_SNAP_SEQ`] so guard (a) (`observed >= target.snap_seq()` → skip)
/// admits the re-base rather than aborting it.
const REBASE_SEEDED_SEQ: u64 = REBASE_SNAP_SEQ + REBASE_BAND;

/// Stream retention for the fixture's `LayerStore` — the production default.
const STREAM_MAXLEN: u64 = 4096;

const CREATED_BY: &str = "bootstrap-concurrency-test-user";
const SESSION_BASE_PATH: &str = "/bootstrap-concurrency-bound";

/// Ceiling on how long a gated PUT parks before giving up and proceeding. Only
/// reached when a test forgot to release it — the PUT then completes and the
/// case fails on an assertion instead of hanging until nextest's hard kill.
const GATE_WAIT: Duration = Duration::from_secs(10);

/// Budget for [`poll_until`]'s "the admission steady state has been reached"
/// probes. A timeout here is a real finding, not a flake: it means fewer
/// bootstraps were admitted than the bound allows, or a trigger never reached
/// `try_acquire_owned`.
const ADMISSION_POLL_BUDGET: Duration = Duration::from_secs(10);

/// Bound on a terminal tracker drain. A hang here is a `BaseBootstrap` bug (a
/// task that never completes, or one spawned onto a different tracker), not a
/// test-only flake.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Probe interval for [`poll_until`]. A bounded probe loop is the sanctioned
/// "wait for some background condition" primitive
/// (`docs-internal/test-writing-guide.md` § Synchronization Patterns); an
/// unconditional `sleep` for synchronization is not, and this file uses none.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

// ─────────────────────────────────────────────────────────────────────────────
// Gate derivation — derive from `Config`, never hardcode
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `Config` with the three SR-7.1 gate inputs on, through the REAL
/// `Config::from_env` under `temp_env::with_vars`.
///
/// Same shape as `first_base_bootstrap_test.rs::bootstrap_config` and
/// `restore_rebase_test.rs::rebase_config`: the parsed values are the ones a
/// deployment with these env vars would actually get, rather than a
/// hand-assembled struct that could drift from the parser. The two endpoint vars
/// are required unconditionally by `Config::from_env` and are irrelevant here.
fn all_gates_on_config() -> Config {
    temp_env::with_vars(
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_SERVICE_LABEL", Some(SERVICE_LABEL)),
            ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("true")),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("InMemory")),
            ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("true")),
        ],
        || {
            Config::from_env_with(PodMemorySource::Reference)
                .expect("Config::from_env must succeed for the bootstrap admission gate")
        },
    )
}

/// The SR-7.1 three-way gate, spelled exactly as `src/main.rs` spells it at the
/// `BaseBootstrap::new` call site.
///
/// Derived, never hardcoded: a test that passed a literal `true` would keep
/// passing if production stopped gating. Load-bearing for case 5, which drives
/// the `spawn_creation_base` wrapper — `spawn` short-circuits on `!enabled`
/// before touching the semaphore, so a wrongly-resolved gate would make that
/// case observe zero PUTs and read as a (false) pass on "the bound shed
/// everything".
fn derive_bootstrap_enabled(cfg: &Config) -> bool {
    cfg.checkpoint_enabled
        && !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled)
        && cfg.checkpoint_tail_enabled
}

// ─────────────────────────────────────────────────────────────────────────────
// The probe store
// ─────────────────────────────────────────────────────────────────────────────

/// One `put_layer_checkpoint` attempt, as the probe saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
}

/// `CheckpointStore` newtype that instruments concurrent occupancy of
/// `put_layer_checkpoint` and can park every PUT on a shared release gate.
///
/// # The instrument
///
/// `max_observed_inflight` is a high-water mark, and it is the primary
/// instrument of this whole file: it is a **structural** invariant (how many
/// bodies held a permit simultaneously) rather than a timing observation. The
/// counter is incremented on entry, BEFORE the gate park, and decremented after
/// the delegate returns, so a parked PUT is counted as in flight for exactly as
/// long as it is holding its `ObjectState` — which is the retention the bound
/// exists to cap.
///
/// # The gate
///
/// A `watch::Receiver<bool>` rather than a `oneshot`, deliberately: `oneshot` is
/// consumed by one waiter, and every case here needs to hold *several* PUTs open
/// at once. `wait_for(|v| *v)` releases all parked PUTs together when the test
/// flips the sender. Bounded by [`GATE_WAIT`] so a test that forgot to release
/// fails on an assertion rather than hanging until the hang-killer, which prints
/// nothing.
///
/// # Delegation
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would not
/// leave a call un-instrumented, it would SUBSTITUTE behaviour behind the wrapper
/// — `read_ext_refs_sidecar`'s default returns an EMPTY set — which is the trap
/// that helper's doc records.
struct ConcurrencyProbeStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    attempts: Mutex<Vec<PutAttempt>>,
    inflight: AtomicUsize,
    max_observed_inflight: AtomicUsize,
    /// Parked PUTs wait for this to carry `true`. Constructed already-`true`
    /// when the fixture is ungated, so a non-gating case pays nothing.
    release: watch::Receiver<bool>,
}

impl ConcurrencyProbeStore {
    fn new(release: watch::Receiver<bool>) -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            attempts: Mutex::new(Vec::new()),
            inflight: AtomicUsize::new(0),
            max_observed_inflight: AtomicUsize::new(0),
            release,
        }
    }

    fn attempts(&self) -> Vec<PutAttempt> {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Total `put_layer_checkpoint` calls ISSUED — whatever became of them.
    /// The shed-vs-queue discriminator: a queueing implementation eventually
    /// issues one per trigger.
    fn put_count(&self) -> usize {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Object ids that reached the PUT, in issue order.
    fn put_object_ids(&self) -> Vec<u64> {
        self.attempts().into_iter().map(|a| a.object_id).collect()
    }

    fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    fn max_observed_inflight(&self) -> usize {
        self.max_observed_inflight.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for ConcurrencyProbeStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE the gate: the test wants to know the call was ISSUED,
        // whatever happens to it afterwards.
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                object_id,
                snap_seq,
                kind: meta.kind,
            });

        // Occupancy is raised before the park and lowered after the delegate
        // returns, so the high-water mark measures bodies holding a permit and
        // a state, not merely bodies that touched the adapter.
        let now = self.inflight.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_observed_inflight.fetch_max(now, Ordering::AcqRel);

        // Bounded: a test that forgot to release must fail on an assertion, not
        // hang until nextest's hard kill.
        let mut release = self.release.clone();
        let _ = tokio::time::timeout(GATE_WAIT, release.wait_for(|v| *v)).await;

        let result = self
            .delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await;

        self.inflight.fetch_sub(1, Ordering::AcqRel);
        result
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // ── The four non-trivial trait defaults ──────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
        self.delegate.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Everything one admission-bound case needs, owned together so `TestValkey`'s
/// prefix-scoped Drop cleanup outlives the assertions.
struct ProbeFixture {
    tv: TestValkey,
    session_id: String,
    layers: LayerStore,
    ext_values: Arc<ValkeyExternalValueStore>,
    tracker: TaskTracker,
    metrics: SharedMetrics,
    probe: Arc<ConcurrencyProbeStore>,
    /// The ONE unit, exactly as `main.rs` constructs it. Both call-site handles
    /// in case 4 are `Arc::clone`s of this — the production wiring shape.
    unit: Arc<BaseBootstrap>,
    /// Flip to `true` to release every parked PUT together.
    release: watch::Sender<bool>,
    /// The bound this fixture's unit was built with — [`BOOTSTRAP_MAX_INFLIGHT`]
    /// unless the case shrank it via `with_max_inflight`.
    bound: usize,
}

/// Wire the fixture.
///
/// - `pool_size` — hand `TestValkey::setup_with_pool_size` at least the trigger
///   count, so connection-pool serialization cannot mask the race the case is
///   trying to observe (`docs-internal/test-writing-guide.md`
///   § Synchronization Patterns).
/// - `gated` — when `true`, every PUT parks until [`ProbeFixture::release_gate`].
/// - `max_inflight` — `None` uses the production bound
///   ([`BOOTSTRAP_MAX_INFLIGHT`], via `BaseBootstrap::new`); `Some(n)` applies the
///   ungated `with_max_inflight` test affordance.
async fn probe_fixture(pool_size: u32, gated: bool, max_inflight: Option<usize>) -> ProbeFixture {
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the all-gates-on Config must resolve the SR-7.1 three-way gate to TRUE, otherwise every \
         case in this file would be asserting against a unit that cannot write"
    );

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let tracker = TaskTracker::new();

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "bootstrap-concurrency-session",
            SESSION_BASE_PATH,
            CREATED_BY,
        )
        .await
        .expect("session creation must succeed against a healthy Valkey");

    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN);
    let ext_values = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    // Already-released when the case does not gate, so an ungated PUT never
    // parks and never pays `GATE_WAIT`.
    let (release, release_rx) = watch::channel(!gated);
    let probe = Arc::new(ConcurrencyProbeStore::new(release_rx));
    let store_dyn: Arc<dyn CheckpointStore> = Arc::clone(&probe) as Arc<dyn CheckpointStore>;

    let mut built = BaseBootstrap::new(
        store_dyn,
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        cfg.service_label.clone(),
        enabled,
    );
    // Read the effective bound off the same expression that produced it, so no
    // case can assert against a number the unit is not actually using.
    let bound = match max_inflight {
        None => BOOTSTRAP_MAX_INFLIGHT,
        Some(n) => {
            built = built.with_max_inflight(n);
            n.max(SMALLEST_ACCEPTED_BOUND)
        }
    };

    ProbeFixture {
        tv,
        session_id,
        layers,
        ext_values,
        tracker,
        metrics,
        probe,
        unit: Arc::new(built),
        release,
        bound,
    }
}

impl ProbeFixture {
    /// Materialize a real layer, returning its freshly-allocated `object_id` —
    /// the same value `handle_create_layer` passes to `spawn_creation_base`.
    async fn create_layer(&self, layer_name: &str, seed: u8) -> u64 {
        self.layers
            .create_layer(
                &self.session_id,
                layer_name,
                CREATED_BY,
                &seed_state(seed),
                Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>,
            )
            .await
            .expect("create_layer must succeed against a healthy Valkey")
    }

    /// Give a freshly-created layer the header shape a DR restore leaves behind,
    /// so a `BootstrapTarget::RestoreRebase` trigger against it reaches the PUT.
    ///
    /// Hand-stamped rather than restored, deliberately: this file makes no claim
    /// about restore mechanics — `restore_rebase_test.rs` owns those, driving the
    /// real `RecoveryHook::try_restore_layer` — and a peak-inflight probe needs
    /// only a header that passes the two restore-site guards. The three columns
    /// are exactly what `LayerStore::write_object`'s `CheckpointCursorStamp`
    /// writes inside a restore's own MULTI/EXEC:
    ///
    /// - `last_checkpoint_seq = REBASE_SNAP_SEQ` — the restore's audit stamp,
    ///   strictly BELOW the target, so guard (a)
    ///   (`observed >= target.snap_seq()` → skip) admits the re-base.
    /// - `restore_seq_bump = REBASE_BAND` — non-zero, matching the
    ///   `absorbed_band` the trigger carries, so guard (b)'s
    ///   `RestoreRebase`-only bump comparison permits the stamp.
    /// - `seq = REBASE_SEEDED_SEQ` — the live writer cursor. `run_once` never
    ///   reads it; written so the header is not a shape production cannot
    ///   produce.
    ///
    /// Both columns parse as decimal `u64`, so `cursor_columns_trusted` is
    /// `true` and the `SkippedUntrustedCursor` arm stays out of the way.
    async fn stamp_restore_shaped_header(&self, object_id: u64) {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        let _: u64 = self
            .tv
            .client()
            .pool
            .hset(
                &key,
                vec![
                    ("seq", REBASE_SEEDED_SEQ.to_string()),
                    ("last_checkpoint_seq", REBASE_SNAP_SEQ.to_string()),
                    ("restore_seq_bump", REBASE_BAND.to_string()),
                ],
            )
            .await
            .expect("HSET obj:hdr must succeed");
    }

    /// The three cursor columns, read in ONE `HMGET` so they cannot disagree
    /// about which write they describe. `None` distinguishes a genuinely absent
    /// column (the creation-time never-bootstrapped shape) from a present one.
    async fn cursor_columns(&self, object_id: u64) -> CursorColumns {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        let raw: Vec<Option<String>> = self
            .tv
            .client()
            .pool
            .hmget(
                &key,
                vec![
                    "last_checkpoint_seq",
                    "last_checkpoint_ms",
                    "restore_seq_bump",
                ],
            )
            .await
            .expect("HMGET obj:hdr cursor columns must succeed");
        CursorColumns {
            last_checkpoint_seq: raw.first().cloned().flatten(),
            last_checkpoint_ms: raw.get(1).cloned().flatten(),
            restore_seq_bump: raw.get(2).cloned().flatten(),
        }
    }

    /// Release every parked PUT together.
    fn release_gate(&self) {
        self.release
            .send(true)
            .expect("the probe store holds a receiver, so the release channel is never closed");
    }

    /// Terminal drain: close the tracker and wait, bounded, for every spawned
    /// bootstrap task to finish. `TaskTracker::close` is terminal, so this runs
    /// at most once per fixture.
    async fn drain(&self) {
        self.tracker.close();
        tokio::time::timeout(DRAIN_TIMEOUT, self.tracker.wait())
            .await
            .expect(
                "the bootstrap TaskTracker must drain inside DRAIN_TIMEOUT — a hang here points at \
                 a BaseBootstrap task that never completes, not at a test-only flake",
            );
    }

    /// `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}` off
    /// this fixture's own registry.
    ///
    /// Matched on `(result, kind)` only: this family renders NO `service` label,
    /// and a `service` term would match zero lines and fold to a silent `0` —
    /// making every tick assertion read as "counter never moved".
    fn layer_writes(&self, result: &str) -> u64 {
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&self.metrics),
            "live_edit_checkpoint_writes_total",
            &[("result", result), ("kind", "layer")],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_checkpoint_writes_total{{result={result:?}, kind=\"layer\"}} is ABSENT. \
                 The whole result × kind grid is zero-seeded at Metrics::new, so an absent series \
                 means the grid or the label set changed"
            )
        }) as u64
    }
}

/// The three columns a bootstrap stamps (two at the creation site, three at the
/// restore site). `Option` rather than `u64`: an ABSENT column and a `0` are
/// different claims, and several assertions here turn on the difference.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CursorColumns {
    last_checkpoint_seq: Option<String>,
    last_checkpoint_ms: Option<String>,
    restore_seq_bump: Option<String>,
}

impl CursorColumns {
    /// The creation-time never-bootstrapped shape: `create_layer` passes
    /// `checkpoint_cursor = None`, so a fresh layer carries no cursor columns at
    /// all.
    fn is_unstamped_creation_shape(&self) -> bool {
        self.last_checkpoint_seq.is_none()
            && self.last_checkpoint_ms.is_none()
            && self.restore_seq_bump.is_none()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Creation state
// ─────────────────────────────────────────────────────────────────────────────

/// A NON-TRIVIAL creation state: two root sections, one inline field each.
///
/// Deliberately not `ObjectState::default()` — an empty state would let a broken
/// build path (one that emitted nothing at all) pass every assertion. `seed`
/// varies the section ids and payload bytes so two layers in one test cannot be
/// confused for each other. Shape copied from
/// `restore_rebase_test.rs::seed_state`.
fn seed_state(seed: u8) -> ObjectState {
    let mut state = ObjectState {
        sequence_num: CREATION_SNAP_SEQ,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };

    let sec_a = u64::from(seed) * 10 + 1;
    let sec_b = u64::from(seed) * 10 + 2;
    for (order, sid) in [sec_a, sec_b].into_iter().enumerate() {
        state.sections.insert(
            sid,
            Section {
                section_id: sid,
                parent_id: 0,
                name: format!("bootstrap-concurrency-{seed}-sec-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        let key = format!("k{sid}");
        state.fields.insert(
            (sid, key.clone()),
            Field {
                section_id: sid,
                key_name: key,
                value: FieldValue::Inline(vec![seed, sid as u8, 0xAA]),
                set_order: u64::from(seed),
                authoring_set_order: u64::from(seed),
                version: 1,
            },
        );
    }
    // Root children list — `DiffEngine::sections_in_dfs_order` walks from root,
    // so without it the reset delta emits no sections at all.
    state.children.insert(0, vec![sec_a, sec_b]);

    state
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent driver
// ─────────────────────────────────────────────────────────────────────────────

/// One `run_once` invocation to fire, with the call-site handle it goes through.
///
/// `unit` is per-trigger so case 4 can hand different `Arc::clone`s of the SAME
/// unit to the creation-time and restore-time triggers, mirroring how `main.rs`
/// hands one `Arc<BaseBootstrap>` to the gRPC service and to the recovery hook.
struct Trigger {
    unit: Arc<BaseBootstrap>,
    object_id: u64,
    state: ObjectState,
    target: BootstrapTarget,
}

/// Fire every trigger at the same instant and return their in-flight handles.
///
/// `Arc<tokio::sync::Barrier>` is the guide's "race N writers at the same
/// instant" primitive: every task completes its setup, waits, and all N unblock
/// together — so the admission race is real rather than an artefact of spawn
/// order.
///
/// Returns handles rather than awaiting, because every gated case must observe
/// the admission steady state (`poll_until`) BEFORE releasing the gate. `returned`
/// counts triggers whose `run_once` has resolved, which is how a case knows every
/// shed has already happened.
///
/// `run_once` is driven directly rather than through the `spawn_*` wrappers
/// because it returns the [`BootstrapOutcome`] arm — the only surface on which
/// `Shed` is distinguishable from `Persisted` — and because the admission
/// semaphore is taken INSIDE `run_once` (step 0), so the seam under test is
/// identical either way. Case 5 drives `spawn_creation_base` instead, precisely
/// to pin the wrapper path.
fn fire_concurrently(
    triggers: Vec<Trigger>,
    returned: Arc<AtomicUsize>,
) -> Vec<JoinHandle<BootstrapOutcome>> {
    let barrier = Arc::new(Barrier::new(triggers.len()));
    triggers
        .into_iter()
        .map(|t| {
            let barrier = Arc::clone(&barrier);
            let returned = Arc::clone(&returned);
            tokio::spawn(async move {
                barrier.wait().await;
                let outcome = t.unit.run_once(t.object_id, t.state, t.target).await;
                returned.fetch_add(1, Ordering::AcqRel);
                outcome
            })
        })
        .collect()
}

async fn join_all(handles: Vec<JoinHandle<BootstrapOutcome>>) -> Vec<BootstrapOutcome> {
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(h.await.expect("no bootstrap trigger task may panic"));
    }
    out
}

/// `(persisted, shed)`, with a guard that no OTHER arm was taken.
///
/// The guard matters: a `SkippedHeaderAbsent` or `Failed` would otherwise be
/// silently folded into "not persisted" and make a shed assertion pass for the
/// wrong reason.
fn tally(outcomes: &[BootstrapOutcome], context: &str) -> (usize, usize) {
    let persisted = outcomes
        .iter()
        .filter(|o| matches!(o, BootstrapOutcome::Persisted { .. }))
        .count();
    let shed = outcomes
        .iter()
        .filter(|o| **o == BootstrapOutcome::Shed)
        .count();
    assert_eq!(
        persisted + shed,
        outcomes.len(),
        "{context}: every trigger must end on Persisted or Shed. Any other arm (a guard skip, an \
         absent header, a failure) means the fixture is not exercising admission at all; got \
         {outcomes:?}"
    );
    (persisted, shed)
}

/// Bounded probe loop for a background condition. Panics with `label` on
/// deadline expiry rather than hanging — a stuck test must fail, not hang.
async fn poll_until(label: &str, budget: Duration, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + budget;
    loop {
        if cond() {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "poll_until({label}) timed out after {budget:?}. This is a finding, not a flake: \
                 the admission steady state this case is built on was never reached"
            );
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Wait for the gated steady state: `bound` PUTs parked in the probe AND every
/// non-admitted trigger already returned.
///
/// Both halves are needed. Without the first, a case could release the gate
/// before the second admitted body reached the adapter and under-count the peak.
/// Without the second, a case could release before some trigger had even called
/// `try_acquire_owned`, which would let it be admitted afterwards and turn a
/// deterministic shed count into a race.
async fn await_admission_steady_state(
    fx: &ProbeFixture,
    trigger_count: usize,
    returned: &Arc<AtomicUsize>,
) {
    let expected_sheds = trigger_count - fx.bound;
    poll_until(
        "bound PUTs parked and every shed trigger returned",
        ADMISSION_POLL_BUDGET,
        || fx.probe.inflight() == fx.bound && returned.load(Ordering::Acquire) == expected_sheds,
    )
    .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric readers (case 6)
// ─────────────────────────────────────────────────────────────────────────────

fn label_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let i = line.find(&needle)?;
    let after = &line[i + needle.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_owned())
}

/// Every rendered cell of `live_edit_checkpoint_writes_total`, as
/// `(result, kind) → value`.
///
/// Parsing the family WHOLE is what makes an out-of-vocabulary cell visible: a
/// new `result` value materialised on its first occurrence would appear as an
/// EXTRA key, where a per-cell [`ProbeFixture::layer_writes`] probe simply would
/// not look for it. Mirrors `helpers::gather_tail_outcome_cells`'s reasoning for
/// the tail-outcome family.
fn gather_write_cells(metrics: &Metrics) -> HashMap<(String, String), f64> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashMap<(String, String), f64> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_writes_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let (Some(result), Some(kind)) = (label_value(line, "result"), label_value(line, "kind"))
        else {
            continue;
        };
        let value = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
            .expect("every rendered sample line must end in a parseable value");
        out.insert((result, kind), value);
    }
    out
}

// The `live_edit_checkpoint_base_bootstrap_total` reader case 6 uses — the
// shed's one signal AND the silence of every other cell of the same family — is
// `helpers::gather_base_bootstrap_cells`, imported above. It parses the family
// WHOLE for the same reason `gather_write_cells` does: an out-of-vocabulary
// `outcome` or `site` materialised on first occurrence shows up as an EXTRA key
// that a per-cell probe would never look for.

// =============================================================================
// Case 1 — the bound is honoured.
// =============================================================================

/// `bound × 4` creation-time triggers released at the same instant never put
/// more than `bound` bootstrap bodies in flight simultaneously.
///
/// The instrument is `max_observed_inflight`, a structural high-water mark, and
/// the assertion is `== bound` rather than merely `<= bound` in both directions
/// at once: `<=` is the contract, and `>=` is what makes the case non-vacuous —
/// a probe that could never observe overlap (because the bound had collapsed to
/// 1, or because pool serialization had quietly serialized the race) would fail
/// it.
///
/// `TestValkey::setup_with_pool_size(TRIGGERS)` is load-bearing for that second
/// half: with a smaller pool the Valkey reads in step 1 would serialize and the
/// race the case is built to observe could never form.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_creation_bootstraps_never_exceed_the_inflight_bound() {
    let triggers = BOOTSTRAP_MAX_INFLIGHT * 4;
    let fx = probe_fixture(triggers as u32, true, None).await;
    assert_eq!(
        fx.bound, BOOTSTRAP_MAX_INFLIGHT,
        "this case must run at the PRODUCTION bound, read from the pub const — a literal here \
         would keep passing after the bound changed"
    );

    let mut object_ids = Vec::new();
    let mut fire = Vec::new();
    for i in 0..triggers {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("bound-honoured-layer-{i}"), seed)
            .await;
        object_ids.push(object_id);
        fire.push(Trigger {
            unit: Arc::clone(&fx.unit),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::CreationBase,
        });
    }

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));

    await_admission_steady_state(&fx, triggers, &returned).await;
    fx.release_gate();
    let outcomes = join_all(handles).await;

    let (persisted, shed) = tally(&outcomes, "bound honoured");
    assert_eq!(
        fx.probe.max_observed_inflight(),
        fx.bound,
        "peak concurrent bootstrap bodies must EQUAL the bound: `<= bound` is the contract \
         (`try_acquire_owned` on a {}-permit semaphore), and `>= bound` is what proves the probe \
         actually observed overlap rather than passing vacuously on a serialized race",
        fx.bound
    );
    assert!(
        fx.probe.max_observed_inflight() <= fx.bound,
        "the admission bound must never be exceeded"
    );
    assert_eq!(
        persisted, fx.bound,
        "exactly `bound` triggers hold a permit while the gate is closed, so exactly `bound` land"
    );
    assert_eq!(
        shed,
        triggers - fx.bound,
        "every trigger that could not acquire must SHED — the permit holders never release before \
         the gate opens, so no later trigger can be admitted"
    );
    assert_eq!(
        fx.probe.put_count(),
        fx.bound,
        "the total PUT count over the whole run is the bound, not the trigger count"
    );

    // The shed layers carry no cursor columns: a shed reads nothing, builds
    // nothing, writes nothing, stamps nothing.
    let admitted: HashSet<u64> = fx.probe.put_object_ids().into_iter().collect();
    for object_id in object_ids {
        if admitted.contains(&object_id) {
            continue;
        }
        assert!(
            fx.cursor_columns(object_id)
                .await
                .is_unstamped_creation_shape(),
            "shed object {object_id} must carry NO cursor column — a shed returns at step 0, \
             before the identity read"
        );
    }

    fx.drain().await;
}

// =============================================================================
// Case 2 — shed, NOT queue. The load-bearing case.
// =============================================================================

/// With the bound at its smallest accepted value, N concurrent triggers issue
/// exactly ONE PUT in total — measured after the gate has opened and everything
/// has fully drained.
///
/// This is the timing-free discriminator. A queueing implementation
/// (`acquire().await` instead of `try_acquire_owned`) is indistinguishable from a
/// shedding one *while the gate is closed* — both show one PUT — and diverges
/// only afterwards: the queue drains and eventually issues all N, the shed does
/// not. So the count is read AFTER the release, and it is a count, never an
/// elapsed time.
///
/// The N−1 losers are asserted to carry no cursor columns, which is the shed's
/// other half: SR-7.2's observable posture, identical to a failed bootstrap —
/// nothing stamped, capture refuses observably.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_saturated_bound_sheds_rather_than_queueing_the_losers() {
    const TRIGGERS: usize = 6;
    let fx = probe_fixture(TRIGGERS as u32, true, Some(SMALLEST_ACCEPTED_BOUND)).await;
    assert_eq!(
        fx.bound, SMALLEST_ACCEPTED_BOUND,
        "this case needs the smallest accepted bound so a single parked PUT saturates admission"
    );

    let mut object_ids = Vec::new();
    let mut fire = Vec::new();
    for i in 0..TRIGGERS {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("shed-not-queue-layer-{i}"), seed)
            .await;
        object_ids.push(object_id);
        fire.push(Trigger {
            unit: Arc::clone(&fx.unit),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::CreationBase,
        });
    }

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));

    await_admission_steady_state(&fx, TRIGGERS, &returned).await;
    // Every loser has ALREADY returned at this point, which is what makes the
    // post-release count a claim about shedding rather than about scheduling.
    assert_eq!(
        fx.probe.put_count(),
        fx.bound,
        "while the gate is closed only the admitted bodies may have reached the adapter"
    );

    fx.release_gate();
    let outcomes = join_all(handles).await;
    fx.drain().await;

    let (persisted, shed) = tally(&outcomes, "shed not queue");
    assert_eq!(
        fx.probe.put_count(),
        1,
        "EXACTLY ONE put_layer_checkpoint may ever be issued. A queueing implementation would have \
         drained its waiters after the gate opened and issued {TRIGGERS}; a shedding one issues \
         the bound and nothing more, because a queued task still holds the ObjectState the bound \
         exists to cap"
    );
    assert_eq!(persisted, 1, "exactly one trigger may land a base");
    assert_eq!(
        shed,
        TRIGGERS - 1,
        "every other trigger must report BootstrapOutcome::Shed"
    );

    let admitted: HashSet<u64> = fx.probe.put_object_ids().into_iter().collect();
    assert_eq!(admitted.len(), 1, "one PUT means one admitted object");
    for object_id in object_ids {
        if admitted.contains(&object_id) {
            let columns = fx.cursor_columns(object_id).await;
            assert_eq!(
                columns.last_checkpoint_seq.as_deref(),
                Some(CREATION_SNAP_SEQ.to_string().as_str()),
                "the admitted layer must carry the creation-time cursor stamp"
            );
            continue;
        }
        assert!(
            fx.cursor_columns(object_id)
                .await
                .is_unstamped_creation_shape(),
            "shed object {object_id} must carry NO cursor column: a shed has the SAME observable \
             posture as a failed bootstrap — nothing read, built, written or stamped, so tail \
             capture refuses observably until its next eligible periodic checkpoint tick lands a \
             base"
        );
    }
}

// =============================================================================
// Case 3 — no spurious shed.
// =============================================================================

/// Exactly `bound` concurrent triggers ALL land a base and stamp their cursors —
/// nothing is shed.
///
/// Without this, case 2 would pass just as well against an implementation that
/// shed everything (or against a bound that had collapsed to zero permits), which
/// would be a strictly worse regression than the unbounded original: every hot
/// layer would lose its lineage root.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exactly_the_bound_many_concurrent_bootstraps_all_land() {
    let triggers = BOOTSTRAP_MAX_INFLIGHT;
    let fx = probe_fixture(triggers as u32, true, None).await;

    let mut object_ids = Vec::new();
    let mut fire = Vec::new();
    for i in 0..triggers {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("no-spurious-shed-layer-{i}"), seed)
            .await;
        object_ids.push(object_id);
        fire.push(Trigger {
            unit: Arc::clone(&fx.unit),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::CreationBase,
        });
    }

    let writes_success_before = fx.layer_writes("success");
    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));

    // `expected_sheds` is zero here, so this reduces to "all `bound` bodies are
    // parked in the adapter at once" — the strongest statement of no-spurious-shed
    // available before the gate opens.
    await_admission_steady_state(&fx, triggers, &returned).await;
    fx.release_gate();
    let outcomes = join_all(handles).await;
    fx.drain().await;

    let (persisted, shed) = tally(&outcomes, "no spurious shed");
    assert_eq!(
        shed, 0,
        "at exactly `bound` concurrent triggers NOTHING may be shed — acquire-or-shed must shed \
         only on genuine saturation"
    );
    assert_eq!(persisted, triggers, "every trigger must land a base");
    assert_eq!(
        fx.probe.put_count(),
        triggers,
        "one PUT per trigger, all of them issued"
    );

    for object_id in object_ids {
        let columns = fx.cursor_columns(object_id).await;
        assert_eq!(
            columns.last_checkpoint_seq.as_deref(),
            Some(CREATION_SNAP_SEQ.to_string().as_str()),
            "object {object_id} must carry last_checkpoint_seq = {CREATION_SNAP_SEQ}"
        );
        assert!(
            columns.last_checkpoint_ms.is_some(),
            "object {object_id} must carry last_checkpoint_ms — the pair rides ONE HSET, so one \
             present and the other absent is impossible unless the stamp changed shape"
        );
        assert!(
            columns.restore_seq_bump.is_none(),
            "the creation site deliberately does NOT write restore_seq_bump: a fresh layer never \
             had a restore, and writing the column would assert that one happened"
        );
    }

    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before + triggers as u64,
        "every landed base ticks {{success, layer}} exactly once"
    );
}

// =============================================================================
// Case 4 — ONE bound spans BOTH call sites.
// =============================================================================

/// Case 4a — the sharp discriminator. ONE unit, one creation-time trigger and
/// one restore-time trigger fired together at a bound of one: exactly ONE PUT.
///
/// The fixture threads a single `Arc<BaseBootstrap>` to both call-site handles by
/// `Arc::clone`, which is exactly the production wiring: `main.rs` builds one
/// unit, hands `Arc::clone(&base_bootstrap)` to `LiveEditService::base_bootstrap`
/// (creation-time) and `Arc::clone(&base_bootstrap)` to
/// `RecoveryHook::with_base_bootstrap` (restore-time). Building TWO units here
/// would fail against a *correct* implementation — the bound is a per-instance
/// field — so this case would be testing a shape production does not have.
///
/// With one shared semaphore the loser sheds and the total is one PUT. With two
/// independent semaphores — one per call site, the defect this Task forbids —
/// each site admits its own trigger and the total is two.
///
/// Which of the two wins is genuinely arbitrary, so the assertions are
/// site-agnostic: counts, the peak, and "whichever lost had none of its columns
/// touched".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_unit_shares_its_bound_across_both_call_sites() {
    let fx = probe_fixture(4, true, Some(SMALLEST_ACCEPTED_BOUND)).await;

    // The two call-site handles. Same `Arc`, two holders — the production shape.
    let grpc_side = Arc::clone(&fx.unit);
    let hook_side = Arc::clone(&fx.unit);

    let creation_oid = fx.create_layer("both-sites-creation-layer", 1).await;
    let restore_oid = fx.create_layer("both-sites-restore-layer", 2).await;
    fx.stamp_restore_shaped_header(restore_oid).await;

    let creation_before = fx.cursor_columns(creation_oid).await;
    let restore_before = fx.cursor_columns(restore_oid).await;

    let fire = vec![
        Trigger {
            unit: grpc_side,
            object_id: creation_oid,
            state: seed_state(1),
            target: BootstrapTarget::CreationBase,
        },
        Trigger {
            unit: hook_side,
            object_id: restore_oid,
            state: seed_state(2),
            target: BootstrapTarget::RestoreRebase {
                seeded_seq: REBASE_SEEDED_SEQ,
                // A hand-put lineage records no span measurement; the value is
                // carried onto the re-base's meta and is irrelevant to admission.
                observed_commit_span: 0,
                absorbed_band: REBASE_BAND,
            },
        },
    ];

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));

    await_admission_steady_state(&fx, 2, &returned).await;
    fx.release_gate();
    let outcomes = join_all(handles).await;
    fx.drain().await;

    let (persisted, shed) = tally(&outcomes, "one bound, both call sites");
    assert_eq!(
        fx.probe.put_count(),
        1,
        "ONE bound must span BOTH call sites: a creation-time first base and a restore-time \
         re-base contending for the same single permit means exactly one PUT. Two — one per site \
         — is the two-independent-semaphores defect this case exists to catch"
    );
    assert_eq!(
        fx.probe.max_observed_inflight(),
        1,
        "peak occupancy over the UNION of both call sites must respect the one shared bound"
    );
    assert_eq!(persisted, 1, "exactly one site may be admitted");
    assert_eq!(shed, 1, "the other must shed, whichever it turns out to be");

    // Site-agnostic: whichever lost had NOTHING touched.
    let admitted = fx.probe.put_object_ids();
    let admitted_oid = admitted
        .first()
        .copied()
        .expect("the single PUT records its object id");
    let (shed_oid, shed_before) = if admitted_oid == creation_oid {
        (restore_oid, restore_before)
    } else {
        assert_eq!(
            admitted_oid, restore_oid,
            "the admitted PUT must belong to one of the two triggers"
        );
        (creation_oid, creation_before)
    };
    assert_eq!(
        fx.cursor_columns(shed_oid).await,
        shed_before,
        "the shed site must leave EVERY cursor column exactly as it found it — unstamped at the \
         creation site, still carrying the restore's own band at the restore site. Never merely \
         `absent`: at the restore site the columns are present before the re-base runs"
    );
}

/// Case 4b — the same claim at the PRODUCTION bound: `bound` creation-time plus
/// `bound` restore-time triggers, all fired together, still peak at `bound` and
/// still issue `bound` PUTs in total.
///
/// 4a's `with_max_inflight(1)` proves the sharing; this proves it at the number
/// that actually ships. A two-semaphore implementation peaks at `2 × bound` and
/// issues `2 × bound` PUTs here, so it fails on both instruments.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_unit_shares_its_bound_across_both_call_sites_at_the_production_bound() {
    let per_site = BOOTSTRAP_MAX_INFLIGHT;
    let triggers = per_site * 2;
    let fx = probe_fixture(triggers as u32, true, None).await;
    assert_eq!(fx.bound, BOOTSTRAP_MAX_INFLIGHT);

    let grpc_side = Arc::clone(&fx.unit);
    let hook_side = Arc::clone(&fx.unit);

    let mut before: HashMap<u64, CursorColumns> = HashMap::new();
    let mut fire = Vec::new();
    for i in 0..per_site {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("union-creation-layer-{i}"), seed)
            .await;
        before.insert(object_id, fx.cursor_columns(object_id).await);
        fire.push(Trigger {
            unit: Arc::clone(&grpc_side),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::CreationBase,
        });
    }
    for i in 0..per_site {
        let seed = (per_site + i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("union-restore-layer-{i}"), seed)
            .await;
        fx.stamp_restore_shaped_header(object_id).await;
        before.insert(object_id, fx.cursor_columns(object_id).await);
        fire.push(Trigger {
            unit: Arc::clone(&hook_side),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::RestoreRebase {
                seeded_seq: REBASE_SEEDED_SEQ,
                observed_commit_span: 0,
                absorbed_band: REBASE_BAND,
            },
        });
    }

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));

    await_admission_steady_state(&fx, triggers, &returned).await;
    fx.release_gate();
    let outcomes = join_all(handles).await;
    fx.drain().await;

    let (persisted, shed) = tally(&outcomes, "union at the production bound");
    assert_eq!(
        fx.probe.max_observed_inflight(),
        fx.bound,
        "peak over the UNION of {per_site} creation-time and {per_site} restore-time triggers must \
         equal the ONE shared bound. Two independent semaphores would peak at {}",
        fx.bound * 2
    );
    assert_eq!(
        fx.probe.put_count(),
        fx.bound,
        "the union issues `bound` PUTs in total, not `bound` per call site"
    );
    assert_eq!(persisted, fx.bound);
    assert_eq!(shed, triggers - fx.bound);

    let admitted: HashSet<u64> = fx.probe.put_object_ids().into_iter().collect();
    for (object_id, columns_before) in before {
        if admitted.contains(&object_id) {
            continue;
        }
        assert_eq!(
            fx.cursor_columns(object_id).await,
            columns_before,
            "shed object {object_id} must leave every cursor column exactly as it found it"
        );
    }
}

// =============================================================================
// Case 5 — a shed is terminal.
// =============================================================================

/// A shed is terminal: no retry, no re-spawn. After the gate opens and the
/// tracker fully drains, the PUT count has not grown past the bound.
///
/// Driven through `spawn_creation_base` — the production wrapper — rather than
/// `run_once`, precisely so "the tracker drains" is a real statement about the
/// shared `TaskTracker` graceful shutdown owns. SR-7 defines ONE attempt for the
/// whole unit; the shed arm inherits that, and nothing in `run_once` re-spawns.
///
/// "Does not grow" is proven structurally rather than by waiting and looking
/// again: after the terminal `close()` + `wait()` there is no task left that
/// could issue a retry (`tracker.is_empty()`), and no metric moved that a retry
/// would have moved — a retried bootstrap ends on either `{success, layer}` or
/// `{error, layer}`, and both are pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shed_is_terminal_and_is_never_retried_or_respawned() {
    const TRIGGERS: usize = 5;
    let fx = probe_fixture(TRIGGERS as u32, true, Some(SMALLEST_ACCEPTED_BOUND)).await;

    let mut object_ids = Vec::new();
    for i in 0..TRIGGERS {
        object_ids.push(
            fx.create_layer(&format!("shed-terminal-layer-{i}"), (i + 1) as u8)
                .await,
        );
    }

    let writes_success_before = fx.layer_writes("success");
    let writes_error_before = fx.layer_writes("error");

    // The production entry point: synchronous, fire-and-forget onto the tracker.
    // Fired in a tight loop rather than behind a barrier because the barrier
    // primitive coordinates the spawned BODIES, and here the bodies are spawned
    // by the unit itself.
    for (i, object_id) in object_ids.iter().enumerate() {
        fx.unit
            .spawn_creation_base(*object_id, seed_state((i + 1) as u8));
    }

    poll_until(
        "the single admitted bootstrap parked in the adapter",
        ADMISSION_POLL_BUDGET,
        || fx.probe.inflight() == fx.bound,
    )
    .await;

    fx.release_gate();
    fx.drain().await;

    assert!(
        fx.tracker.is_empty(),
        "after the terminal drain no bootstrap task may remain — a re-spawned shed would still be \
         tracked here, since `spawn` uses the same TaskTracker"
    );
    assert_eq!(
        fx.probe.put_count(),
        fx.bound,
        "the PUT count must not grow past the bound after the drain: SR-7 defines ONE attempt and \
         the shed inherits it — there is no retry loop and no re-spawn"
    );
    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before + fx.bound as u64,
        "only the admitted bootstrap may tick {{success, layer}}"
    );
    assert_eq!(
        fx.layer_writes("error"),
        writes_error_before,
        "a shed is not an error, and no retry of one produced an error either"
    );

    let admitted: HashSet<u64> = fx.probe.put_object_ids().into_iter().collect();
    for object_id in object_ids {
        if admitted.contains(&object_id) {
            continue;
        }
        assert!(
            fx.cursor_columns(object_id)
                .await
                .is_unstamped_creation_shape(),
            "shed object {object_id} must still carry NO cursor column after the full drain — a \
             deferred retry would have stamped it"
        );
    }
}

// =============================================================================
// Case 6 — metric fence.
// =============================================================================

/// The landed metric posture of a shed, PINNED: it ticks EXACTLY ONE cell, and
/// that cell is on the bootstrap-outcome family.
///
/// Epic 5's SR-7 adjudication (`t2.dfh.be.s8`) gave this unit its own family,
/// `live_edit_checkpoint_base_bootstrap_total{outcome,site}`, and a shed ticks
/// `{shed, creation_base}` there. What it deliberately still does NOT do is move
/// a `checkpoint_writes_total` cell — whose `result` vocabulary is
/// `success | error | skipped`, the periodic loop using `skipped` for a snapshot
/// form it declined on a tick that DID run. A shed completes no generation and
/// never reaches the adapter, so it belongs in neither. The module header argues
/// why that silence is a verdict to keep rather than a gap to close.
///
/// Asserted in both directions on BOTH families, so each posture is pinned rather
/// than incidental. Each family is gathered whole before and after:
/// `checkpoint_writes_total{success, layer}` must move by exactly the admitted
/// count and EVERY other cell of it must be byte-identical (which covers
/// `skipped` specifically), its rendered `result × kind` vocabulary must be
/// unchanged and must match the pinned `CHECKPOINT_WRITE_KINDS`; no
/// `live_edit_checkpoint_tail_total{outcome}` cell may move; and on the bootstrap
/// family `{shed, creation_base}` must move by exactly the shed count,
/// `{persisted, creation_base}` by exactly the admitted count, and every other
/// cell of it — the whole `restore_rebase` column above all — must be
/// byte-identical.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shed_ticks_only_the_bootstrap_outcome_cell() {
    const TRIGGERS: usize = 6;
    let fx = probe_fixture(TRIGGERS as u32, true, Some(SMALLEST_ACCEPTED_BOUND)).await;

    let mut fire = Vec::new();
    for i in 0..TRIGGERS {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("metric-fence-layer-{i}"), seed)
            .await;
        fire.push(Trigger {
            unit: Arc::clone(&fx.unit),
            object_id,
            state: seed_state(seed),
            target: BootstrapTarget::CreationBase,
        });
    }

    let writes_before = gather_write_cells(&fx.metrics);
    let tail_before = gather_tail_outcome_cells(&fx.metrics);
    let bootstrap_before = gather_base_bootstrap_cells(&fx.metrics);
    assert!(
        !writes_before.is_empty(),
        "the whole result × kind grid is zero-seeded at Metrics::new, so an empty gather means the \
         parser or the family name drifted and every assertion below would be vacuous"
    );
    assert!(
        !bootstrap_before.is_empty(),
        "the whole outcome × site grid is zero-seeded at Metrics::new too, so an empty gather here \
         would make assertion (e) vacuous in the one direction that carries the signal"
    );

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));
    await_admission_steady_state(&fx, TRIGGERS, &returned).await;
    fx.release_gate();
    let outcomes = join_all(handles).await;
    fx.drain().await;

    let (persisted, shed) = tally(&outcomes, "metric fence");
    assert_eq!(persisted, fx.bound);
    assert_eq!(
        shed,
        TRIGGERS - fx.bound,
        "the case is only about anything if sheds actually happened"
    );

    let writes_after = gather_write_cells(&fx.metrics);
    let tail_after = gather_tail_outcome_cells(&fx.metrics);
    let bootstrap_after = gather_base_bootstrap_cells(&fx.metrics);

    // (a) The vocabulary did not grow. An out-of-band cell — a new `result`
    // value materialised on first occurrence — shows up as an extra key here.
    let before_keys: HashSet<&(String, String)> = writes_before.keys().collect();
    let after_keys: HashSet<&(String, String)> = writes_after.keys().collect();
    assert_eq!(
        before_keys, after_keys,
        "a shed must introduce NO new checkpoint_writes_total cell. Epic 5's SR-7 adjudication \
         extended the checkpoint vocabulary on its OWN family (assertion (e)) and left this one \
         alone; a new `result` or `kind` value here would mean an arm folded an admission \
         decision into the generation-completion counter after all"
    );

    // (b) The rendered label vocabulary is the pinned one, both axes.
    let rendered_results: HashSet<&str> = writes_after.keys().map(|(r, _)| r.as_str()).collect();
    assert_eq!(
        rendered_results,
        HashSet::from(["success", "error", "skipped"]),
        "the `result` vocabulary is success|error|skipped and this Task did not extend it"
    );
    let rendered_kinds: HashSet<&str> = writes_after.keys().map(|(_, k)| k.as_str()).collect();
    assert_eq!(
        rendered_kinds,
        CHECKPOINT_WRITE_KINDS
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        "the `kind` vocabulary is pinned by CHECKPOINT_WRITE_KINDS, read from the production const"
    );

    // (c) Exactly one cell moved, and by exactly the admitted count. Every other
    // cell — `skipped × layer` above all, the one a shed could most plausibly
    // have been folded onto — is byte-identical.
    let landed_cell = ("success".to_string(), "layer".to_string());
    for (cell, after) in &writes_after {
        let before = writes_before
            .get(cell)
            .copied()
            .expect("key sets were just asserted equal");
        if *cell == landed_cell {
            assert_eq!(
                *after - before,
                fx.bound as f64,
                "{{success, layer}} must move by exactly the number of ADMITTED bootstraps"
            );
        } else {
            assert_eq!(
                *after, before,
                "cell {cell:?} must not move. The landed posture is that a shed moves NOTHING on \
                 this family, deliberately: it counts generation-completion outcomes, and a shed \
                 completes no generation and never reaches the adapter. Its signal lives on \
                 live_edit_checkpoint_base_bootstrap_total instead (assertion (e)) — the two \
                 families are additive, never substitutes"
            );
        }
    }

    // (d) The tail-outcome family is untouched: this unit writes bases, and a
    // shed writes nothing at all.
    assert_eq!(
        tail_after, tail_before,
        "no live_edit_checkpoint_tail_total{{outcome}} cell may move — the bootstrap unit is not a \
         tail producer (SR-4.1's chain-collapse re-base in `rebase.rs` is the sole `{{rebased}}` \
         producer), and a shed is not a capture outcome. Epic 5's bootstrap-outcome family is a \
         THIRD, additive family, not a re-home of this one"
    );

    // (e) The family that DOES carry the signal moved, and only where the
    // adjudication says it may. Same both-directions discipline as (c), now on
    // live_edit_checkpoint_base_bootstrap_total: `{shed, creation_base}` by
    // exactly the shed count, `{persisted, creation_base}` by exactly the
    // admitted count, every other cell — the whole `restore_rebase` column
    // included, since this case fires no restore-shaped trigger — byte-identical.
    let bootstrap_before_keys: HashSet<&(String, String)> = bootstrap_before.keys().collect();
    let bootstrap_after_keys: HashSet<&(String, String)> = bootstrap_after.keys().collect();
    assert_eq!(
        bootstrap_before_keys, bootstrap_after_keys,
        "the outcome × site grid is fully zero-seeded at Metrics::new, so a shed must materialise \
         NO new cell on the bootstrap family either"
    );
    let shed_cell = ("shed".to_string(), "creation_base".to_string());
    let persisted_cell = ("persisted".to_string(), "creation_base".to_string());
    // The spellings this case names are read back against the production
    // vocabulary rather than trusted: a rename there must fail here loudly
    // instead of silently making the deltas below unobservable.
    assert!(
        CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.contains(&shed_cell.0.as_str())
            && CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.contains(&persisted_cell.0.as_str())
            && CHECKPOINT_BASE_BOOTSTRAP_SITES.contains(&shed_cell.1.as_str()),
        "the outcome/site spellings asserted below must be members of the pinned production \
         vocabularies"
    );
    assert!(
        bootstrap_before.contains_key(&shed_cell) && bootstrap_before.contains_key(&persisted_cell),
        "both cells must already exist as zero-seeded series before the triggers fire, or the \
         deltas below would be measured against a cell that materialised on first tick"
    );
    for (cell, after) in &bootstrap_after {
        let before = bootstrap_before
            .get(cell)
            .copied()
            .expect("key sets were just asserted equal");
        let expected = if *cell == shed_cell {
            (TRIGGERS - fx.bound) as f64
        } else if *cell == persisted_cell {
            fx.bound as f64
        } else {
            0.0
        };
        assert_eq!(
            *after - before,
            expected,
            "live_edit_checkpoint_base_bootstrap_total{cell:?} must move by exactly {expected}: a \
             shed ticks {{shed, creation_base}} and an admitted bootstrap ticks \
             {{persisted, creation_base}}, one increment per run_once invocation and nothing \
             else — an arm that ticked twice, ticked the wrong outcome, or leaked onto the \
             restore_rebase site would show up here"
        );
    }
}

// =============================================================================
// Robustness — the affordance the shrunk-bound cases stand on.
// =============================================================================

/// `with_max_inflight(0)` clamps to the smallest accepted bound rather than
/// silently disabling the unit.
///
/// `0` would shed unconditionally — not a state any caller wants and not one
/// production can reach — so the landed affordance does `max_inflight.max(1)`.
/// Pinned behaviourally (a bootstrap still lands) rather than by reading the
/// private field, and it is what keeps [`SMALLEST_ACCEPTED_BOUND`] honest: were
/// the clamp removed, this case would observe a shed instead of a landing and the
/// three cases built on that constant would be probing a bound that admits
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_max_inflight_zero_clamps_to_the_smallest_accepted_bound() {
    let fx = probe_fixture(2, false, Some(0)).await;
    assert_eq!(
        fx.bound, SMALLEST_ACCEPTED_BOUND,
        "the fixture's derived bound must agree with the production clamp"
    );

    let object_id = fx.create_layer("clamped-bound-layer", 1).await;
    let outcome = fx
        .unit
        .run_once(object_id, seed_state(1), BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    assert_eq!(
        outcome,
        BootstrapOutcome::Persisted {
            snap_seq: CREATION_SNAP_SEQ
        },
        "a bound of 0 must clamp to {SMALLEST_ACCEPTED_BOUND} and still admit one bootstrap; a \
         literal 0-permit semaphore would shed unconditionally and disable the unit"
    );
    assert_eq!(fx.probe.put_count(), 1);
    assert_eq!(
        fx.cursor_columns(object_id)
            .await
            .last_checkpoint_seq
            .as_deref(),
        Some(CREATION_SNAP_SEQ.to_string().as_str()),
        "the admitted bootstrap must stamp its cursor"
    );
}
