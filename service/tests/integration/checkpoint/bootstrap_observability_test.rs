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

// Plan b.dfh, Epic 5 (t1.dfh.be), Task 5 (t2.dfh.be.s8), Subtask
// t3.dfh.be.s8.fa — the per-arm producer fence for the SR-7 observability
// adjudication.
//
//! Every `BaseBootstrap::run_once` arm that the SR-7 adjudication gave a metric
//! cell, driven to that arm and pinned against the WHOLE family — plus the
//! `refused_lineage_unrootable` capture verdict and the sum identity the whole
//! adjudication rests on.
//!
//! ## The verdicts this module pins
//!
//! The adjudication landed (dated block on Epic ticket `t1.dfh.be`) as: ONE new
//! family `live_edit_checkpoint_base_bootstrap_total{outcome,site}` — 9 outcomes
//! × 2 sites (b.vn6 later added a tenth, `unstamped_header_absent`; every shape
//! assertion here derives the grid from
//! [`expected_bootstrap_grid_size`] rather than spelling it, so the vocabulary
//! moved without touching them), the full grid zero-seeded at `Metrics::new`,
//! no `service` label —
//! with **exactly one increment per `run_once` invocation**, minted in the
//! `run_once` wrapper from `BootstrapOutcome::outcome_label()` and
//! `BootstrapTarget::label()`; ONE additive tail member
//! (`refused_lineage_unrootable`, the tenth `CHECKPOINT_TAIL_OUTCOMES` value);
//! and NO change to `checkpoint_writes_total`, whose `result` axis keeps counting
//! generation-completion outcomes.
//!
//! The per-arm cell assignment, quoted verbatim from the engineer lane's
//! completion report (`.quorum-runstate/task5-lane-reports/te-split-outcome-engineer.md`,
//! "Variants → labels"):
//!
//! | Variant | Label \[idx\] | `run_once_inner` arm |
//! |---|---|---|
//! | `Persisted { snap_seq }` | `persisted` \[0\] | success; also `writes_total{success,layer}` |
//! | `Shed` | `shed` \[1\] | step-0 admission shed (drops `state`, returns first) |
//! | `SkippedHeaderAbsent` | `skipped_header_absent` \[2\] | **two** producers, both pre-PUT: step-1 `Ok(None)`, and (b.vn6) the step-4b pre-PUT `HEXISTS obj:hdr seq` liveness re-check returning `Ok(false)` — the header survived step 1 but is not live by the time the OMPF and sidecar are built. One cell for both because the cell's meaning is "refused before the PUT, adapter untouched"; the post-PUT refusal is `unstamped_header_absent` \[5\] for exactly that reason. The second producer is driven by `first_base_bootstrap_test.rs` case 16 |
//! | `SkippedGenerationPresent` | `skipped_generation_present` \[3\] | guard (a), PRE-PUT only |
//! | `SkippedUntrustedCursor` | `skipped_untrusted_cursor` \[4\] | `!cursor_columns_trusted`, RestoreRebase only, pre-PUT |
//! | `UnstampedHeaderAbsent` (b.vn6) | `unstamped_header_absent` \[5\] | the conditional cursor stamp REFUSED, post-PUT: the layer was deleted while the PUT was in flight. Reaps the just-written base; moves NO `writes_total` cell. Owned by `first_base_bootstrap_test.rs` cases 14, 15, 17 and 18 |
//! | `UnstampedGenerationPresent` (new) | `unstamped_generation_present` \[6\] | guard (b) seq half, post-PUT |
//! | `UnstampedForeignRestore` (new) | `unstamped_foreign_restore` \[7\] | guard (b) bump half, post-PUT, RestoreRebase only |
//! | `UnstampedWriteError` (new) | `unstamped_write_error` \[8\] | pre-stamp HMGET failure + conditional-stamp ERROR (as distinct from its refusal above); both still tick `writes_total{error,layer}` |
//! | `Failed` | `failed` \[9\] | identity-read / OMPF-build / sidecar / PUT errors, `NotImplemented` (no `writes_total` tick) |
//!
//! ## What this module does NOT duplicate
//!
//! Four sibling files already own claims that touch the same unit. Nothing here
//! restates them, and no assertion in any of them was edited for this Subtask:
//!
//! - `bootstrap_concurrency_bound_test.rs` — the admission bound itself, and
//!   (case 6, `shed_ticks_only_the_bootstrap_outcome_cell`) the **creation-site**
//!   `{shed, creation_base}` / `{persisted, creation_base}` deltas. The shed case
//!   here is therefore the RESTORE-site one, and it is the only shed case in this
//!   file.
//! - `restore_rebase_test.rs` — restore mechanics, the landed cursor-column
//!   shapes, and the PUT-failure classification matrix. Its case 4b names the
//!   `{unstamped_generation_present, restore_rebase}` cell in prose but asserts
//!   no metric for it; the positive fence lives here.
//! - `first_base_bootstrap_test.rs` — the creation-site landing shape, the
//!   `writes_total` postures, and the vanished-header / guard-(a) stimuli this
//!   file re-uses. Its assertions are `writes_total`-scoped; the bootstrap-family
//!   cells are this file's.
//! - `e2e_tail_bootstrap_test.rs` — the same unit through the real `CreateLayer`
//!   commit path and the real periodic loop. Loop-level, end-to-end; this file is
//!   arm-level and drives the unit directly.
//!
//! The `metrics.rs` `#[cfg(test)]` pins for the new family (zero-seed, label set,
//! literal vocabulary + `.len()`, typo guard) and for the 9→10 tail vocabulary
//! belong to the vocabulary lane (`t3.dfh.be.s8.mf`) and are deliberately absent
//! here.
//!
//! ## Why `run_once` directly, and why the header is hand-stamped at the restore site
//!
//! Every case awaits `run_once` and asserts the returned [`BootstrapOutcome`]
//! variant BEFORE it looks at a metric. That non-vacuity gate is what keeps a
//! fixture that silently took a different arm from asserting flatness for the
//! wrong reason — and it rules out the `spawn_*` wrappers, which discard the
//! outcome, and rules out driving the restore site through
//! `RecoveryHook::try_restore_layer`, which spawns fire-and-forget.
//!
//! So the restore-site cases stamp the header shape a DR restore leaves behind
//! (`OutcomeFixture::stamp_restore_shaped_header`) and hand `run_once` a
//! `BootstrapTarget::RestoreRebase` assembled from those same values, exactly as
//! `bootstrap_concurrency_bound_test.rs` does and for the same reason: this file
//! makes no claim about restore mechanics — `restore_rebase_test.rs` owns those,
//! driving the real hook — only about which cell each arm ticks. The stimulus
//! recipes themselves are ported, not invented: guard (a) is
//! `first_base_bootstrap_test.rs`'s pre-stamped cursor, the vanished header is
//! its `UNLINK obj:hdr`, the untrusted column is `restore_rebase_test.rs`'s case
//! 4c, and the two guard-(b) halves are its case 4b's park-the-PUT-then-stamp
//! shape.
//!
//! ## The all-cells discipline
//!
//! Every case snapshots BOTH families whole ([`helpers::gather_base_bootstrap_cells`]
//! and [`helpers::gather_tail_outcome_cells`]) before and after, then asserts:
//! the key SETS are unchanged (an out-of-band `outcome` or `site` materialising
//! on first occurrence shows up as an extra key rather than being invisible), the
//! grid is the full 9 × 2, the adjudicated cell moved by exactly the expected
//! amount, EVERY sibling cell of the family stayed flat with a message saying
//! what a tick there would have meant, and no
//! `live_edit_checkpoint_tail_total{outcome}` cell moved at all — unconditionally
//! and under every verdict, because the bootstrap unit is not a tail producer.
//! `checkpoint_writes_total{result, kind="layer"}` is asserted in both directions
//! per case as well, since the two families are additive and never substitutes.
//!
//! Shape copied — not code — from `composed_restore_ext_matrix_test.rs`'s
//! `ExtRecoveryCells` / `assert_ext_recovery_delta` pair.
//!
//! ## Bounded waits — no `sleep` is used for synchronization
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | [`ArmedProbeStore`] gate park | [`GATE_WAIT`] (10 s) | the case never released the gate; the PUT then proceeds so the case fails on an assertion rather than hanging |
//! | [`poll_until`] — "the PUT is parked" / "the admission steady state is reached" | [`PARK_POLL_BUDGET`] (10 s) | the body never reached the adapter, or fewer bootstraps were admitted than the bound allows |
//!
//! Worst case for the heaviest case (the restore-site shed, which holds both) is
//! 10 + 10 = 20 s, comfortably inside `.config/nextest.toml`'s 120 s hard kill —
//! a SIGKILLed test prints none of its diagnostics, which is exactly when they
//! are needed. Nothing here asserts on elapsed time.
//!
//! Deliberately NOT feature-gated: a gated module compiles to nothing and reports
//! green having covered nothing (`docs-internal/test-writing-guide.md`), and this
//! file is the only evidence that the SR-7 adjudication actually landed on the
//! producer.
//!
//! Sandbox env note: `TestValkey::setup*` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).
//!
//! [`helpers::gather_base_bootstrap_cells`]: super::helpers::gather_base_bootstrap_cells
//! [`helpers::gather_tail_outcome_cells`]: super::helpers::gather_tail_outcome_cells

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::sync::{watch, Barrier};
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::tail_capture::{
    capture_layer_tail, TailCaptureDecision, TailCaptureRefusal,
};
use live_edit::checkpoint::{
    BaseBootstrap, BootstrapOutcome, BootstrapTarget, CheckpointAdapter, CheckpointBlob,
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore,
    TailCaptureConfig,
};
use live_edit::config::{Config, PodMemorySource};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::{
    Metrics, SharedMetrics, CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES, CHECKPOINT_BASE_BOOTSTRAP_SITES,
    CHECKPOINT_TAIL_OUTCOMES,
};
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    gather_base_bootstrap_cells, gather_tail_outcome_cells, put_tail_generation,
    read_layer_writes_counter, DelegatingCheckpointStore,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Replica id stamped into every persisted `CheckpointMeta` and carried as this
/// file's registry label. Neither family under test renders a `service` label, so
/// it is not a metric selector here — cross-test isolation comes from the
/// per-fixture registry (`crate::common::metrics_scrape_only`) plus nextest's
/// process-per-test isolation. No shared static state anywhere in this file.
const SERVICE_LABEL: &str = "bootstrap-observability-test-replica";

const CREATED_BY: &str = "bootstrap-observability-test-user";
const SESSION_BASE_PATH: &str = "/bootstrap-observability";

/// Layer-name prefix, distinct from every bootstrap sibling's, so a stray key in
/// a shared Valkey is attributable to this file at a glance.
const LAYER_PREFIX: &str = "bootstrap-observability";

/// `snap_seq` every creation-time bootstrap generation lands at
/// (`BootstrapTarget::CreationBase::snap_seq`). Named so the assertions read as
/// the contract rather than as a magic `1`.
const CREATION_SNAP_SEQ: u64 = 1;

/// The `last_checkpoint_seq` a restore's own audit stamp leaves behind — i.e. the
/// generation the restore restored. Strictly below [`REBASE_SEEDED_SEQ`], which
/// is what makes guard (a) ADMIT the re-base instead of aborting it.
const REBASE_SNAP_SEQ: u64 = 40;

/// The guard band a restore adds on top of `restore_floor`, stamped into
/// `obj:hdr restore_seq_bump`. Non-zero because `band >= 1` unconditionally
/// post-b.8y5 — a zero band would make the restore-shaped header a shape no
/// restore can produce.
const REBASE_BAND: u64 = 50;

/// `seeded_seq` for the restore-shaped triggers — `restore_floor + band`, and the
/// live header's `seq` at the moment such a restore committed.
const REBASE_SEEDED_SEQ: u64 = REBASE_SNAP_SEQ + REBASE_BAND;

/// A `restore_seq_bump` value that is NOT [`REBASE_BAND`] — i.e. a band this
/// re-base did not absorb, which is exactly what `raw_bump_blocks_stamp` refuses
/// to clear. Any numeric value other than the absorbed band works; it is spelled
/// as an offset so the inequality is visible rather than asserted separately.
const FOREIGN_BAND: u64 = REBASE_BAND + 7;

/// The two inequalities the restore-shaped fixture rests on, checked at COMPILE
/// time so re-tuning a constant is a build failure rather than a case that
/// silently pins the neighbouring arm.
///
/// - `REBASE_SNAP_SEQ < REBASE_SEEDED_SEQ` — guard (a) admits the re-base at step
///   1, and guard (b)'s SEQ half does not fire at the pre-stamp re-read, so the
///   BUMP half is reachable at all.
/// - `FOREIGN_BAND != REBASE_BAND` — the interloper band is one this re-base did
///   NOT absorb. Were they equal, `raw_bump_blocks_stamp` would permit the stamp
///   and the bump-half case would pin `persisted` instead.
const _: () = {
    assert!(REBASE_SNAP_SEQ < REBASE_SEEDED_SEQ);
    assert!(FOREIGN_BAND != REBASE_BAND);
};

/// The smallest bound `BaseBootstrap::with_max_inflight` accepts (it clamps
/// `max_inflight.max(1)`). `bootstrap_concurrency_bound_test.rs` owns the
/// behavioural pin on that clamp.
const SMALLEST_ACCEPTED_BOUND: usize = 1;

/// Stream retention for the fixture's `LayerStore` — the production default.
const STREAM_MAXLEN: u64 = 4096;

/// Ceiling on how long a gated PUT parks before giving up and proceeding. Only
/// reached when a case forgot to release it — the PUT then completes and the case
/// fails on an assertion instead of hanging until nextest's hard kill.
const GATE_WAIT: Duration = Duration::from_secs(10);

/// Budget for [`poll_until`]'s "the PUT has parked" / "the admission steady state
/// has been reached" probes. A timeout here is a real finding, not a flake: the
/// mid-flight state every guard-(b) case is built on was never reached.
const PARK_POLL_BUDGET: Duration = Duration::from_secs(10);

/// Probe interval for [`poll_until`]. A bounded probe loop is the sanctioned
/// "wait for some background condition" primitive
/// (`docs-internal/test-writing-guide.md` § Synchronization Patterns); an
/// unconditional `sleep` for synchronization is not, and this file uses none.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// The full grid size of `live_edit_checkpoint_base_bootstrap_total`, derived
/// from the two production vocabularies rather than spelled as `18`: a
/// vocabulary that grows must fail the pin in `metrics.rs` (lane `mf`), not
/// silently widen this file's shape check.
fn expected_bootstrap_grid_size() -> usize {
    CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len() * CHECKPOINT_BASE_BOOTSTRAP_SITES.len()
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate derivation — derive from `Config`, never hardcode
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `Config` with the three SR-7.1 gate inputs on, through the REAL
/// `Config::from_env` under `temp_env::with_vars`.
///
/// Same shape as `first_base_bootstrap_test.rs::bootstrap_config` and
/// `restore_rebase_test.rs::rebase_config`: the parsed values are the ones a
/// deployment with these env vars would actually get, rather than a
/// hand-assembled struct that could drift from the parser.
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
                .expect("Config::from_env must succeed for the SR-7 observability fixture")
        },
    )
}

/// The SR-7.1 three-way gate, spelled exactly as `src/main.rs` spells it at the
/// `BaseBootstrap::new` call site.
///
/// Derived, never hardcoded: a test that passed a literal `true` would keep
/// passing if production stopped gating.
fn derive_bootstrap_enabled(cfg: &Config) -> bool {
    cfg.checkpoint_enabled
        && !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled)
        && cfg.checkpoint_tail_enabled
}

// ─────────────────────────────────────────────────────────────────────────────
// The store every case shares
// ─────────────────────────────────────────────────────────────────────────────

/// `CheckpointStore` newtype that can PARK the base PUT (so a case can move the
/// header mid-flight) and can FAIL it (so a case can reach the `failed` arm).
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would not
/// leave a call un-instrumented, it would SUBSTITUTE behaviour behind the wrapper
/// and make the listing / sidecar assertions vacuously green — the trap that
/// helper's doc records.
struct ArmedProbeStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    /// Base PUTs ISSUED, counted before the park and before the failure arm — a
    /// case wants to know the call was made, whatever happened to it after.
    base_attempts: AtomicUsize,
    /// Bodies currently parked inside the gate. The instrument the guard-(b)
    /// cases poll on before moving the header.
    parked: AtomicUsize,
    /// Base PUTs that actually reached the delegate, so "the blob is durable even
    /// though the stamp was declined" is directly assertable.
    base_puts_delegated: AtomicUsize,
    release: watch::Receiver<bool>,
    fail_base_puts: AtomicBool,
}

impl ArmedProbeStore {
    fn new(release: watch::Receiver<bool>) -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            base_attempts: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            base_puts_delegated: AtomicUsize::new(0),
            release,
            fail_base_puts: AtomicBool::new(false),
        }
    }

    fn base_attempts(&self) -> usize {
        self.base_attempts.load(Ordering::Acquire)
    }

    fn parked(&self) -> usize {
        self.parked.load(Ordering::Acquire)
    }

    fn base_puts_delegated(&self) -> usize {
        self.base_puts_delegated.load(Ordering::Acquire)
    }

    /// Fail every subsequent base PUT.
    ///
    /// One classification only. `run_once` takes the identical terminal arm for
    /// `Transient` and `Permanent` — the classification rides the log line, not
    /// the control flow — and `restore_rebase_test.rs` /
    /// `first_base_bootstrap_test.rs` already run the two-classification matrix
    /// on that contract. What is being pinned here is the CELL, which is
    /// classification-independent by construction.
    fn fail_base_puts(&self) {
        self.fail_base_puts.store(true, Ordering::Release);
    }
}

#[async_trait]
impl CheckpointStore for ArmedProbeStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        if meta.kind == CheckpointKind::Base {
            self.base_attempts.fetch_add(1, Ordering::AcqRel);
            self.parked.fetch_add(1, Ordering::AcqRel);
            // Bounded: a case that forgot to release must fail on an assertion,
            // not hang until nextest's hard kill.
            let mut release = self.release.clone();
            let _ = tokio::time::timeout(GATE_WAIT, release.wait_for(|v| *v)).await;
            self.parked.fetch_sub(1, Ordering::AcqRel);

            if self.fail_base_puts.load(Ordering::Acquire) {
                return Err(CheckpointError::Transient {
                    source: "synthetic base PUT failure (SR-7 observability test)".into(),
                });
            }
            self.base_puts_delegated.fetch_add(1, Ordering::AcqRel);
        }

        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
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
    // skipping instrumentation: `read_ext_refs_sidecar`'s default returns an
    // EMPTY set, which would make every sidecar-shaped assertion vacuously green.
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

/// Everything one arm case needs, owned together so `TestValkey`'s prefix-scoped
/// Drop cleanup outlives the assertions.
struct OutcomeFixture {
    tv: TestValkey,
    session_id: String,
    layers: LayerStore,
    ext_values: Arc<ValkeyExternalValueStore>,
    metrics: SharedMetrics,
    store: Arc<ArmedProbeStore>,
    store_dyn: Arc<dyn CheckpointStore>,
    unit: Arc<BaseBootstrap>,
    release: watch::Sender<bool>,
    /// The bound this fixture's unit was built with, read off the same expression
    /// that produced it so no case can assert against a number the unit is not
    /// using.
    bound: usize,
    config: Config,
}

/// Wire the fixture.
///
/// - `pool_size` — hand `TestValkey::setup_with_pool_size` at least the
///   concurrent-trigger count, so connection-pool serialization cannot mask the
///   admission race (`docs-internal/test-writing-guide.md`
///   § Synchronization Patterns).
/// - `gated` — when `true`, every base PUT parks until [`OutcomeFixture::release_gate`].
/// - `max_inflight` — `None` uses the production bound (via `BaseBootstrap::new`);
///   `Some(n)` applies the ungated `with_max_inflight` test affordance.
///
/// No `TaskTracker` is retained: every case awaits `run_once` directly rather than
/// going through a `spawn_*` wrapper (the wrappers discard the outcome, which the
/// non-vacuity gate needs), so nothing is ever spawned onto the tracker and there
/// is no drain to bound.
async fn outcome_fixture(
    pool_size: u32,
    gated: bool,
    max_inflight: Option<usize>,
) -> OutcomeFixture {
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the all-gates-on Config must resolve the SR-7.1 three-way gate to TRUE, otherwise every \
         case in this file would be asserting cells against a unit that cannot write"
    );

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "bootstrap-observability-session",
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

    // Already-released when the case does not gate, so an ungated PUT never parks
    // and never pays `GATE_WAIT`.
    let (release, release_rx) = watch::channel(!gated);
    let store = Arc::new(ArmedProbeStore::new(release_rx));
    let store_dyn: Arc<dyn CheckpointStore> = Arc::clone(&store) as Arc<dyn CheckpointStore>;

    let mut built = BaseBootstrap::new(
        Arc::clone(&store_dyn),
        tv.client().clone(),
        tv.key_schema().clone(),
        TaskTracker::new(),
        Arc::clone(&metrics),
        cfg.service_label.clone(),
        enabled,
    );
    let bound = match max_inflight {
        None => live_edit::checkpoint::BOOTSTRAP_MAX_INFLIGHT,
        Some(n) => {
            built = built.with_max_inflight(n);
            n.max(SMALLEST_ACCEPTED_BOUND)
        }
    };

    OutcomeFixture {
        tv,
        session_id,
        layers,
        ext_values,
        metrics,
        store,
        store_dyn,
        unit: Arc::new(built),
        release,
        bound,
        config: cfg,
    }
}

impl OutcomeFixture {
    /// Materialize a real layer at [`CREATION_SNAP_SEQ`], returning its
    /// freshly-allocated `object_id` — the same value `handle_create_layer` passes
    /// to `spawn_creation_base`.
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
    /// so a [`BootstrapTarget::RestoreRebase`] trigger against it reaches the PUT.
    ///
    /// The three columns are exactly what `LayerStore::write_object`'s
    /// `CheckpointCursorStamp` writes inside a restore's own MULTI/EXEC:
    ///
    /// - `last_checkpoint_seq = REBASE_SNAP_SEQ` — the restore's audit stamp,
    ///   strictly BELOW the target, so guard (a) admits the re-base.
    /// - `restore_seq_bump = REBASE_BAND` — non-zero, matching the `absorbed_band`
    ///   the trigger carries, so guard (b)'s bump comparison permits the stamp.
    /// - `seq = REBASE_SEEDED_SEQ` — the live writer cursor. `run_once` never
    ///   reads it; written so the header is not a shape production cannot produce.
    ///
    /// Both cursor columns parse as decimal `u64`, so `cursor_columns_trusted` is
    /// `true` and the `SkippedUntrustedCursor` arm stays out of the way unless a
    /// case puts it there on purpose.
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

    /// The target a re-base of a [`Self::stamp_restore_shaped_header`] layer
    /// carries. `observed_commit_span = 0` is the hand-put no-measurement
    /// sentinel; no arm under test reads it.
    fn restore_target(&self) -> BootstrapTarget {
        BootstrapTarget::RestoreRebase {
            seeded_seq: REBASE_SEEDED_SEQ,
            observed_commit_span: 0,
            absorbed_band: REBASE_BAND,
        }
    }

    /// `HSET le:{oid}:obj:hdr <field> <value>` — the "another writer moved the
    /// header while our PUT was in flight" stimulus.
    async fn set_header_field(&self, object_id: u64, field: &str, value: &str) {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        let _: u64 = self
            .tv
            .client()
            .pool
            .hset(&key, vec![(field, value)])
            .await
            .expect("HSET obj:hdr must succeed");
    }

    /// `HGET le:{oid}:obj:hdr <field>`.
    async fn header_field(&self, object_id: u64, field: &str) -> Option<String> {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        self.tv
            .client()
            .pool
            .hget(&key, field)
            .await
            .expect("HGET obj:hdr must succeed")
    }

    /// Delete `obj:hdr` — the delete/overwrite race that lands inside the
    /// bootstrap window.
    async fn unlink_header(&self, object_id: u64) {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        let _: u64 = self
            .tv
            .client()
            .pool
            .unlink(vec![key.clone()])
            .await
            .expect("UNLINK obj:hdr must succeed");
        assert!(
            !self
                .tv
                .client()
                .pool
                .exists::<bool, _>(&key)
                .await
                .expect("EXISTS obj:hdr must succeed"),
            "precondition: obj:hdr for object {object_id} must actually be gone"
        );
    }

    /// Replace `obj:hdr` with a STRING key of the same name.
    ///
    /// The only stimulus that fails a Valkey hash command against a header the
    /// bootstrap already read successfully: every later `HMGET` / `HSET` on it
    /// returns `WRONGTYPE`.
    async fn wrongtype_header(&self, object_id: u64) {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        let _: u64 = self
            .tv
            .client()
            .pool
            .unlink(vec![key.clone()])
            .await
            .expect("UNLINK obj:hdr must succeed");
        let _: () = self
            .tv
            .client()
            .pool
            .set(&key, "not-a-hash", None, None, false)
            .await
            .expect("SET obj:hdr (as a string) must succeed");
    }

    async fn listing(&self, object_id: u64) -> Vec<CheckpointHandle> {
        self.store_dyn
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed")
    }

    /// Invoke the tail-capture unit DIRECTLY (it is `pub`; the precedents are
    /// `checkpointer_tail_outcomes_test.rs` and `restore_rebase_test.rs`).
    async fn capture(&self, object_id: u64) -> TailCaptureDecision {
        capture_layer_tail(
            self.tv.client(),
            self.tv.key_schema(),
            object_id,
            self.store_dyn.as_ref(),
            SERVICE_LABEL,
            &TailCaptureConfig::from_config(&self.config),
        )
        .await
        .expect("capture must not error; the arm under test is a decision, not a failure")
    }

    /// Release every parked base PUT together.
    fn release_gate(&self) {
        self.release
            .send(true)
            .expect("the probe store holds a receiver, so the release channel is never closed");
    }

    fn cells(&self) -> FamilyCells {
        FamilyCells::read(&self.metrics)
    }
}

/// A NON-TRIVIAL creation state: two root sections, one inline field each.
///
/// Deliberately not `ObjectState::default()` — an empty state would let a broken
/// build path (one that emitted nothing at all) pass every assertion. `seed`
/// varies the section ids and payload bytes so two layers in one test cannot be
/// confused for each other. Shape copied from
/// `bootstrap_concurrency_bound_test.rs::seed_state`.
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
                name: format!("{LAYER_PREFIX}-{seed}-sec-{sid}"),
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
    // Root children list — `DiffEngine::sections_in_dfs_order` walks from root, so
    // without it the reset delta emits no sections at all.
    state.children.insert(0, vec![sec_a, sec_b]);

    state
}

// ─────────────────────────────────────────────────────────────────────────────
// The all-cells reader + its delta assertion
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of EVERY cell of both families at one instant, plus the two
/// `checkpoint_writes_total` layer cells.
///
/// The whole grid, never one cell: proving the arm under test moved is only half
/// the claim — the other half is that every sibling stayed flat, which is what
/// separates "this arm ticks its cell" from "this arm ticks its cell and also
/// leaks onto a neighbour". Use as a BEFORE/AFTER pair around the drive.
///
/// Both families are parsed WHOLE (`(outcome, site)` / `outcome` keyed), so an
/// out-of-band label value materialised on its first occurrence surfaces as an
/// EXTRA key rather than being invisible to a named-cell probe.
#[derive(Debug, Clone)]
struct FamilyCells {
    bootstrap: HashMap<(String, String), f64>,
    tail: HashMap<String, f64>,
    writes_success: u64,
    writes_error: u64,
}

impl FamilyCells {
    fn read(metrics: &Metrics) -> Self {
        Self {
            bootstrap: gather_base_bootstrap_cells(metrics),
            tail: gather_tail_outcome_cells(metrics),
            writes_success: read_layer_writes_counter(metrics, "success"),
            writes_error: read_layer_writes_counter(metrics, "error"),
        }
    }
}

/// What a tick on a given `outcome` cell would have MEANT — the per-cell message
/// half of the flatness assertion.
///
/// A flat-sibling failure that says only "cell moved" tells a reader nothing; one
/// that says which arm of `run_once_inner` must therefore have run tells them
/// where to look.
fn outcome_means(outcome: &str) -> &'static str {
    match outcome {
        "persisted" => "a base landed on the adapter AND the cursor stamp committed",
        "shed" => {
            "the admission bound was saturated, so nothing was read, built, written or stamped"
        }
        "skipped_header_absent" => {
            "obj:hdr was not live at one of the TWO pre-PUT reads — the step-1 identity read \
             returned None, or (b.vn6) the step-4b `HEXISTS obj:hdr seq` re-check returned false \
             after the OMPF and sidecar were built. Either way nothing reached the adapter, which \
             is what this cell means; the post-PUT refusal is `unstamped_header_absent`"
        }
        "skipped_generation_present" => {
            "guard (a) fired PRE-PUT — a generation at or above the target already existed"
        }
        "skipped_untrusted_cursor" => {
            "a cursor column was present but unparseable, so the restore-site re-base declined \
             before the PUT"
        }
        "unstamped_header_absent" => {
            "the conditional cursor stamp REFUSED POST-PUT (b.vn6) — obj:hdr had no `seq` by the \
             time the base landed, i.e. the layer was deleted or overwritten while the PUT was in \
             flight, and the just-written base was reaped"
        }
        "unstamped_generation_present" => {
            "guard (b)'s seq half fired POST-PUT — a generation landed while the PUT was in flight"
        }
        "unstamped_foreign_restore" => {
            "guard (b)'s bump half fired POST-PUT — obj:hdr carried a restore_seq_bump this \
             re-base did not absorb"
        }
        "unstamped_write_error" => {
            "the base landed but a Valkey command ERRORED (the pre-stamp HMGET, or the conditional \
             stamp itself — as distinct from that stamp cleanly refusing, which is \
             `unstamped_header_absent`)"
        }
        "failed" => {
            "no base landed at all — the identity read, the OMPF build, the sidecar, or the PUT \
             failed"
        }
        _ => "an outcome outside the pinned CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES vocabulary",
    }
}

/// Assert the bootstrap family moved by exactly `expected` and by ZERO on every
/// other cell, that the whole tail-outcome family is untouched, and that
/// `checkpoint_writes_total{result, kind="layer"}` moved by exactly the two
/// deltas named.
///
/// `expected` is `(outcome, site, delta)`. Every spelling in it is read back
/// against the production vocabularies rather than trusted, so a rename there
/// fails here loudly instead of silently making the delta unobservable.
fn assert_cells_delta(
    pre: &FamilyCells,
    post: &FamilyCells,
    expected: &[(&str, &str, u64)],
    writes_success_delta: u64,
    writes_error_delta: u64,
    context: &str,
) {
    // ── Non-vacuity of the reader itself ─────────────────────────────────
    let grid = expected_bootstrap_grid_size();
    assert_eq!(
        pre.bootstrap.len(),
        grid,
        "{context}: the whole outcome × site grid is zero-seeded at Metrics::new, so the BEFORE \
         snapshot must render all {grid} series. A short read means the parser or the family name \
         drifted and every assertion below would be vacuous"
    );
    assert_eq!(
        pre.tail.len(),
        CHECKPOINT_TAIL_OUTCOMES.len(),
        "{context}: the tail-outcome family is zero-seeded from CHECKPOINT_TAIL_OUTCOMES, so the \
         BEFORE snapshot must render one series per member ({} of them since the SR-7 \
         adjudication appended refused_lineage_unrootable)",
        CHECKPOINT_TAIL_OUTCOMES.len()
    );

    // ── The vocabulary did not grow ──────────────────────────────────────
    let before_keys: HashSet<&(String, String)> = pre.bootstrap.keys().collect();
    let after_keys: HashSet<&(String, String)> = post.bootstrap.keys().collect();
    assert_eq!(
        before_keys, after_keys,
        "{context}: no run_once arm may MATERIALISE a bootstrap-family cell. The full grid is \
         zero-seeded, so an extra key here is an out-of-band `outcome` or `site` value — a \
         label spelling that is not a member of the pinned production vocabulary"
    );

    for (outcome, site, _) in expected {
        assert!(
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.contains(outcome),
            "{context}: {outcome:?} must be a member of CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES"
        );
        assert!(
            CHECKPOINT_BASE_BOOTSTRAP_SITES.contains(site),
            "{context}: {site:?} must be a member of CHECKPOINT_BASE_BOOTSTRAP_SITES"
        );
        assert!(
            pre.bootstrap
                .contains_key(&((*outcome).to_string(), (*site).to_string())),
            "{context}: cell {{{outcome}, {site}}} must already exist as a zero-seeded series \
             BEFORE the drive, or its delta would be measured against a cell that materialised on \
             first tick"
        );
    }

    // The whole grid in vocabulary order, so a failure names a deterministic cell
    // rather than whichever one `HashMap` iteration reached first.
    let delta_of = |cell: &(String, String)| -> f64 {
        let before = pre
            .bootstrap
            .get(cell)
            .copied()
            .expect("key sets were just asserted equal");
        let after = post
            .bootstrap
            .get(cell)
            .copied()
            .expect("key sets were just asserted equal");
        after - before
    };
    let ordered_cells: Vec<(String, String)> = CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES
        .iter()
        .flat_map(|outcome| {
            CHECKPOINT_BASE_BOOTSTRAP_SITES
                .iter()
                .map(move |site| ((*outcome).to_string(), (*site).to_string()))
        })
        .collect();

    // ── (a) The adjudicated cells moved, by exactly the expected amount ──
    for (outcome, site, want) in expected {
        let cell = ((*outcome).to_string(), (*site).to_string());
        assert_eq!(
            delta_of(&cell),
            *want as f64,
            "{context}: live_edit_checkpoint_base_bootstrap_total{{outcome={outcome:?}, \
             site={site:?}}} must move by exactly {want} — this is the cell the SR-7 adjudication \
             assigned to the arm the fixture just drove, and the returned BootstrapOutcome above \
             already proved that arm was reached. A zero here means the producer stopped ticking; \
             a wrong count means it ticked more than once per invocation",
        );
    }

    // ── (b) EVERY other cell of the family stayed flat ───────────────────
    for cell in &ordered_cells {
        if expected
            .iter()
            .any(|(outcome, site, _)| *outcome == cell.0 && *site == cell.1)
        {
            continue;
        }
        assert_eq!(
            delta_of(cell),
            0.0,
            "{context}: live_edit_checkpoint_base_bootstrap_total{{outcome={:?}, site={:?}}} must \
             stay FLAT. A tick here would mean {} at the {} call site — which is not what this \
             fixture drove. The wrapper ticks exactly one cell per run_once invocation, so a \
             stray tick here is a mis-mapped outcome_label(), a leaked site label, or a second \
             increment",
            cell.0,
            cell.1,
            outcome_means(&cell.0),
            cell.1
        );
    }

    // ── (c) The sum identity over this drive ─────────────────────────────
    let summed_delta: f64 = ordered_cells.iter().map(delta_of).sum();
    let want_sum: f64 = expected.iter().map(|(_, _, d)| *d as f64).sum();
    assert_eq!(
        summed_delta, want_sum,
        "{context}: the family's SUMMED delta must equal the number of run_once invocations \
         ({want_sum}). This is the identity every ratio in the SR-7 adjudication depends on — a \
         mismatch means an invocation ticked twice or not at all"
    );

    // ── The tail-outcome family is untouched, unconditionally ────────────
    assert_eq!(
        post.tail, pre.tail,
        "{context}: NO live_edit_checkpoint_tail_total{{outcome}} cell may move. The bootstrap \
         unit writes bases and is not a tail producer at all — `rebase.rs` is the sole `rebased` \
         producer and the capture path owns the refusals. The bootstrap family is a THIRD, \
         additive family, never a re-home of this one"
    );

    // ── The generation-completion family, both directions ────────────────
    assert_eq!(
        post.writes_success - pre.writes_success,
        writes_success_delta,
        "{context}: checkpoint_writes_total{{result=\"success\",kind=\"layer\"}} must move by \
         exactly {writes_success_delta}. That family counts COMPLETED generations (blob durable \
         AND cursor stamped); the bootstrap family counts invocations. They are additive, never \
         substitutes"
    );
    assert_eq!(
        post.writes_error - pre.writes_error,
        writes_error_delta,
        "{context}: checkpoint_writes_total{{result=\"error\",kind=\"layer\"}} must move by \
         exactly {writes_error_delta}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent driver (the shed case only)
// ─────────────────────────────────────────────────────────────────────────────

/// One `run_once` invocation to fire.
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
/// order. `returned` counts triggers whose `run_once` has resolved, which is how a
/// case knows every shed has already happened.
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

/// Bounded probe loop for a background condition. Panics with `label` on deadline
/// expiry rather than hanging — a stuck test must fail, not hang.
async fn poll_until(label: &str, budget: Duration, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + budget;
    loop {
        if cond() {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "poll_until({label}) timed out after {budget:?}. This is a finding, not a flake: \
                 the mid-flight state this case is built on was never reached"
            );
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Drive one `run_once` to the point where its base PUT is PARKED in the store,
/// returning the in-flight handle. The caller then moves the header and releases.
///
/// This is `restore_rebase_test.rs` case 4b's shape, ported: park the PUT, stamp
/// an interloper column, release. The recipe is what makes the two guard-(b) arms
/// — both of which fire only AFTER a successful PUT — reachable at all.
async fn park_a_base_put(
    fx: &OutcomeFixture,
    object_id: u64,
    seed: u8,
    target: BootstrapTarget,
) -> JoinHandle<BootstrapOutcome> {
    let unit = Arc::clone(&fx.unit);
    let state = seed_state(seed);
    let handle = tokio::spawn(async move { unit.run_once(object_id, state, target).await });

    let store = Arc::clone(&fx.store);
    poll_until(
        "the base PUT has parked in the store",
        PARK_POLL_BUDGET,
        || store.parked() == 1,
    )
    .await;

    handle
}

// =============================================================================
// Case 1 — the RESTORE-site admission shed.
// =============================================================================

/// A shed at the RESTORE site ticks `{shed, restore_rebase}` and nothing else.
///
/// The restore-site shed is the operationally sharp one, and it had no coverage
/// before this Subtask: `bootstrap_concurrency_bound_test.rs`'s case 6 pins the
/// CREATION-site column (`{shed, creation_base}` / `{persisted, creation_base}`),
/// which this case deliberately does not restate. A restored layer that sheds
/// stays at the generation it was restored from until its next eligible
/// periodic checkpoint tick lands a base (or it is restored again) — so
/// `{shed, restore_rebase}` is the cell an operator watches, and it must be
/// attributable to the restore site rather than folded in with creation-time
/// sheds.
///
/// The shed count is deterministic rather than raced: the case waits for the
/// admission steady state (`bound` PUTs parked AND every non-admitted trigger
/// already returned) before releasing the gate, so no trigger can be admitted
/// late.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_site_shed_ticks_only_the_shed_cell_of_its_own_site() {
    const TRIGGERS: usize = 4;
    let fx = outcome_fixture(TRIGGERS as u32, true, Some(SMALLEST_ACCEPTED_BOUND)).await;
    assert_eq!(
        fx.bound, SMALLEST_ACCEPTED_BOUND,
        "the shrunk bound must be the one the unit is actually using"
    );

    let mut fire = Vec::new();
    for i in 0..TRIGGERS {
        let seed = (i + 1) as u8;
        let object_id = fx
            .create_layer(&format!("{LAYER_PREFIX}-restore-shed-{i}"), seed)
            .await;
        fx.stamp_restore_shaped_header(object_id).await;
        fire.push(Trigger {
            unit: Arc::clone(&fx.unit),
            object_id,
            state: seed_state(seed),
            target: fx.restore_target(),
        });
    }

    let pre = fx.cells();

    let returned = Arc::new(AtomicUsize::new(0));
    let handles = fire_concurrently(fire, Arc::clone(&returned));
    let expected_sheds = TRIGGERS - fx.bound;
    {
        let store = Arc::clone(&fx.store);
        let returned = Arc::clone(&returned);
        poll_until(
            "the bound PUTs are parked and every shed trigger has returned",
            PARK_POLL_BUDGET,
            || store.parked() == fx.bound && returned.load(Ordering::Acquire) == expected_sheds,
        )
        .await;
    }
    fx.release_gate();
    let outcomes = join_all(handles).await;

    // ── Non-vacuity: the arms actually taken ─────────────────────────────
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
        "every trigger must end on Persisted or Shed. Any other arm (a guard skip, an absent \
         header, a failure) means the fixture is not exercising admission at all; got {outcomes:?}"
    );
    assert_eq!(persisted, fx.bound, "exactly `bound` triggers are admitted");
    assert_eq!(
        shed, expected_sheds,
        "the case is only about anything if sheds actually happened"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[
            ("shed", "restore_rebase", expected_sheds as u64),
            ("persisted", "restore_rebase", fx.bound as u64),
        ],
        // The one admitted re-base completed a generation.
        fx.bound as u64,
        0,
        "restore-site shed",
    );
}

// =============================================================================
// Case 2 — guard (a), PRE-PUT.
// =============================================================================

/// A generation already at or above the target makes guard (a) decline BEFORE the
/// PUT, and that arm ticks `{skipped_generation_present, creation_base}`.
///
/// Stimulus ported verbatim from `first_base_bootstrap_test.rs`'s
/// `bootstrap_does_not_regress_a_cursor_a_later_generation_already_advanced`:
/// pre-stamp the header as a later generation would have left it. The PUT-count
/// assertion is what separates this cell from the post-PUT guard-(b) siblings —
/// they share the predicate and differ entirely in the residual they leave, which
/// is why the adjudication gave them different cells.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn guard_a_ticks_skipped_generation_present_before_any_adapter_work() {
    let fx = outcome_fixture(2, false, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-guard-a"), 11)
        .await;

    // A later generation's cursor, obviously synthetic so an overwrite by the
    // bootstrap's own stamp would be unmistakable rather than a near-miss.
    const ADVANCED_SEQ: &str = "500";
    fx.set_header_field(object_id, "last_checkpoint_seq", ADVANCED_SEQ)
        .await;

    let attempts_before = fx.store.base_attempts();
    let pre = fx.cells();

    let outcome = fx
        .unit
        .run_once(object_id, seed_state(11), BootstrapTarget::CreationBase)
        .await;

    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedGenerationPresent,
        "non-vacuity: the fixture must reach guard (a)'s PRE-PUT arm. Any other variant means the \
         pre-stamp did not take and the cell assertions below would be about a different arm"
    );
    assert_eq!(
        fx.store.base_attempts(),
        attempts_before,
        "guard (a) fires on the step-1 identity read, before any adapter work — an attempt here \
         means the guard moved to the pre-stamp re-read, which is the post-PUT sibling arm and a \
         DIFFERENT cell"
    );
    assert_eq!(
        fx.header_field(object_id, "last_checkpoint_seq")
            .await
            .as_deref(),
        Some(ADVANCED_SEQ),
        "the declined bootstrap must leave the advanced cursor exactly as it found it"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("skipped_generation_present", "creation_base", 1)],
        0,
        0,
        "guard (a) skip",
    );
}

// =============================================================================
// Case 3 — the vanished header.
// =============================================================================

/// `obj:hdr` deleted between the commit and the identity read ticks
/// `{skipped_header_absent, creation_base}` and moves no `writes_total` cell.
///
/// Stimulus ported from `first_base_bootstrap_test.rs`'s vanished-header case.
/// The posture the SR-7 adjudication landed for this arm is precise and worth
/// stating: it is a routine delete/overwrite race, so it must NOT read as an
/// error on the generation-completion family — ticking `error` there would put a
/// permanent alert floor under every normal create-then-delete workload — but it
/// IS an invocation, so it must be counted on the bootstrap family, or the
/// denominator that makes every other cell a ratio would under-count.
///
/// This case drives the **step-1** producer of `{skipped_header_absent, …}`. The
/// cell has a second producer since b.vn6 — the step-4b pre-PUT `HEXISTS obj:hdr
/// seq` re-check — which is `first_base_bootstrap_test.rs` case 16's, not this
/// file's: both are pre-PUT and leave the adapter untouched, which is precisely
/// what the cell means, so they share it by design rather than by omission.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vanished_header_ticks_skipped_header_absent_and_no_writes_cell() {
    let fx = outcome_fixture(2, false, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-vanished-header"), 12)
        .await;

    fx.unlink_header(object_id).await;

    let attempts_before = fx.store.base_attempts();
    let pre = fx.cells();

    let outcome = fx
        .unit
        .run_once(object_id, seed_state(12), BootstrapTarget::CreationBase)
        .await;

    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedHeaderAbsent,
        "non-vacuity: the fixture must reach the step-1 `Ok(None)` arm"
    );
    assert_eq!(
        fx.store.base_attempts(),
        attempts_before,
        "a vanished header must abort before any adapter work — a base for a layer that no longer \
         exists is an orphan created on purpose"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("skipped_header_absent", "creation_base", 1)],
        0,
        0,
        "vanished header",
    );
}

// =============================================================================
// Case 4 — the untrusted cursor column, PRE-PUT, restore site only.
// =============================================================================

/// An unparseable cursor column makes the restore-site re-base decline entirely,
/// ticking `{skipped_untrusted_cursor, restore_rebase}`.
///
/// Stimulus ported from `restore_rebase_test.rs`'s case 4c: another writer puts a
/// non-numeric value in `restore_seq_bump`, `read_header_identity` folds it to 0
/// and reports `cursor_columns_trusted = false`, and the arm returns before the
/// OMPF build and before the PUT.
///
/// The fold is what makes this arm reachable at all, and it survives b.hka review
/// round 2 unchanged: that change narrowed the loader's refusal to `session_id` /
/// `layer_name` — the two columns the restore path guards on — while the cursor
/// and descriptive columns keep folding, precisely so a bookkeeping column cannot
/// drop a restorable layer out of DR. Had it widened to every column, this header
/// would read as ABSENT and the outcome below would be `SkippedHeaderAbsent`
/// instead. The arm is `RestoreRebase`-only by design, so
/// the `creation_base` half of this cell is deliberately unreachable and must
/// stay flat — which the sibling-flatness half of the assertion covers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_untrusted_cursor_column_ticks_skipped_untrusted_cursor_at_the_restore_site() {
    let fx = outcome_fixture(2, false, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-untrusted-cursor"), 13)
        .await;
    fx.stamp_restore_shaped_header(object_id).await;
    fx.set_header_field(object_id, "restore_seq_bump", "not-a-number")
        .await;

    let attempts_before = fx.store.base_attempts();
    let pre = fx.cells();

    let outcome = fx
        .unit
        .run_once(object_id, seed_state(13), fx.restore_target())
        .await;

    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedUntrustedCursor,
        "non-vacuity: an unparseable cursor column must make the re-base decline ENTIRELY"
    );
    assert_eq!(
        fx.store.base_attempts(),
        attempts_before,
        "the untrusted-cursor arm returns at step 1, before the OMPF build and before the PUT"
    );
    assert_eq!(
        fx.header_field(object_id, "restore_seq_bump")
            .await
            .as_deref(),
        Some("not-a-number"),
        "the declined re-base must leave the untrustworthy column exactly as it found it — a blind \
         CLEARING of an unreadable bump is precisely the write this guard exists to prevent"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("skipped_untrusted_cursor", "restore_rebase", 1)],
        0,
        0,
        "untrusted cursor column",
    );
}

// =============================================================================
// Case 5 — guard (b), seq half, POST-PUT.
// =============================================================================

/// A generation landing mid-flight leaves the blob durable and every cursor
/// column untouched, ticking `{unstamped_generation_present, restore_rebase}` and
/// NO `writes_total` cell in either direction.
///
/// `restore_rebase_test.rs`'s case 4b drives the same interleaving and pins the
/// landed cursor/blob posture plus the `writes_total` silence — in prose it names
/// this cell, but it asserts no metric for it. This is the positive fence.
///
/// The `writes_total` posture is the sharp part and is asserted in BOTH
/// directions: the blob IS durable, so a reader might expect `success`; no
/// generation completed (no cursor stamp landed), so the family that counts
/// generation completions must not move, and the arm's whole signal lives on the
/// bootstrap family instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_generation_landing_mid_flight_ticks_unstamped_generation_present() {
    let fx = outcome_fixture(2, true, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-guard-b-seq"), 14)
        .await;
    fx.stamp_restore_shaped_header(object_id).await;

    let delegated_before = fx.store.base_puts_delegated();
    let pre = fx.cells();

    let handle = park_a_base_put(&fx, object_id, 14, fx.restore_target()).await;

    // A periodic generation lands while the PUT is parked, stamping a cursor
    // ABOVE the re-base's target.
    let interloper_seq = REBASE_SEEDED_SEQ + 1;
    fx.set_header_field(
        object_id,
        "last_checkpoint_seq",
        &interloper_seq.to_string(),
    )
    .await;

    fx.release_gate();
    let outcome = handle.await.expect("the run_once task must not panic");

    assert_eq!(
        outcome,
        BootstrapOutcome::UnstampedGenerationPresent,
        "non-vacuity: the fixture must reach guard (b)'s SEQ half, POST-PUT. A \
         SkippedGenerationPresent here would mean the interloper landed before step 1 and the \
         case is pinning the wrong arm's cell"
    );
    assert_eq!(
        fx.store.base_puts_delegated(),
        delegated_before + 1,
        "the PUT completed before guard (b)'s pre-stamp re-read — the blob is durable even though \
         the stamp was declined"
    );
    assert!(
        fx.listing(object_id)
            .await
            .iter()
            .any(|h| h.snap_seq == REBASE_SEEDED_SEQ && h.kind == CheckpointKind::Base),
        "landed posture is BLOB PRESENT + CURSOR UNTOUCHED: the Base at seeded_seq must exist, \
         because guard (b) fires AFTER the PUT"
    );
    assert_eq!(
        fx.header_field(object_id, "last_checkpoint_seq")
            .await
            .as_deref(),
        Some(interloper_seq.to_string().as_str()),
        "the skip must leave last_checkpoint_seq at the HIGHER value the mid-flight generation \
         stamped, never regress it to the re-base's own target"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("unstamped_generation_present", "restore_rebase", 1)],
        // The blob is durable but the GENERATION did not complete, so the family
        // that counts generation-completion outcomes stays flat in BOTH
        // directions. This is the posture the SR-7 adjudication kept deliberately
        // (hazard #5 discharged with no writes_total change).
        0,
        0,
        "guard (b) seq half",
    );
}

// =============================================================================
// Case 6 — guard (b), bump half, POST-PUT, restore site only.
// =============================================================================

/// A `restore_seq_bump` this re-base did not absorb makes guard (b)'s bump half
/// leave every cursor column standing, ticking
/// `{unstamped_foreign_restore, restore_rebase}`.
///
/// This arm had NO coverage above the `raw_bump_blocks_stamp` predicate unit test
/// before this Subtask, and it is the ONLY evidence for the same-band residual's
/// log-only discharge: the residual is that guard's false negative, undetectable
/// at the decision point, and this cell is the enclosing population — a RISK
/// INDICATOR that same-band double restores are live at all, not a harm count.
///
/// The stimulus is PM correction B3's, and the distinction from case 4c matters:
/// an *unparseable* `restore_seq_bump` trips `cursor_columns_trusted` at step 1
/// and returns `SkippedUntrustedCursor` before the PUT, so that recipe cannot
/// reach this arm. What reaches it is a NUMERIC bump different from the band this
/// re-base absorbed, stamped while the PUT is parked, with `last_checkpoint_seq`
/// left strictly below the target so the seq half above does not fire first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_foreign_restore_bump_ticks_unstamped_foreign_restore() {
    let fx = outcome_fixture(2, true, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-guard-b-bump"), 15)
        .await;
    fx.stamp_restore_shaped_header(object_id).await;

    let delegated_before = fx.store.base_puts_delegated();
    let pre = fx.cells();

    let handle = park_a_base_put(&fx, object_id, 15, fx.restore_target()).await;

    // A SECOND restore lands while the PUT is parked: it stamps its own band,
    // which this re-base never absorbed. The two inequalities this depends on —
    // `FOREIGN_BAND != REBASE_BAND` and `REBASE_SNAP_SEQ < REBASE_SEEDED_SEQ` —
    // are pinned at compile time beside the constants.
    fx.set_header_field(object_id, "restore_seq_bump", &FOREIGN_BAND.to_string())
        .await;

    // …and the seq column is left alone, READ rather than assumed: it must still
    // sit strictly below the target at the moment the PUT is parked, or guard
    // (b)'s SEQ half would fire first at the pre-stamp re-read and claim the arm.
    let mid_flight_seq = fx
        .header_field(object_id, "last_checkpoint_seq")
        .await
        .and_then(|v| v.parse::<u64>().ok())
        .expect("the restore-shaped header's last_checkpoint_seq must be present and parseable");
    assert!(
        mid_flight_seq < REBASE_SEEDED_SEQ,
        "precondition: last_checkpoint_seq ({mid_flight_seq}) must still be strictly below the \
         target ({REBASE_SEEDED_SEQ}) mid-flight, so the BUMP half is the arm reached"
    );

    fx.release_gate();
    let outcome = handle.await.expect("the run_once task must not panic");

    assert_eq!(
        outcome,
        BootstrapOutcome::UnstampedForeignRestore,
        "non-vacuity: the fixture must reach guard (b)'s BUMP half. An UnstampedGenerationPresent \
         here would mean the seq half fired first; a SkippedUntrustedCursor would mean the \
         interloper value did not parse and the arm returned at step 1"
    );
    assert_eq!(
        fx.store.base_puts_delegated(),
        delegated_before + 1,
        "the blob IS durable — this arm fires after the PUT, which is exactly what makes it a \
         different cell from the pre-PUT skips"
    );
    assert_eq!(
        fx.header_field(object_id, "restore_seq_bump")
            .await
            .as_deref(),
        Some(FOREIGN_BAND.to_string().as_str()),
        "the newer restore's bump must be left STANDING. Clearing a bump this re-base did not \
         absorb would unblock tail capture over that restore's fossils — the harm the guard exists \
         to prevent"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("unstamped_foreign_restore", "restore_rebase", 1)],
        // Unmoved in BOTH directions: the blob is durable, and the generation did
        // not complete.
        0,
        0,
        "guard (b) bump half",
    );
}

// =============================================================================
// Case 7 — the unstamped write error (the reachable, HMGET half).
// =============================================================================

/// A `WRONGTYPE` on `obj:hdr` fails the pre-stamp HMGET one statement after a
/// successful PUT, ticking `{unstamped_write_error, restore_rebase}` AND
/// `writes_total{error, layer}`.
///
/// **Which half of the arm this is, and why the other is not tested.** The
/// `UnstampedWriteError` variant covers two producers: the pre-stamp `HMGET`
/// failure and the stamp `HSET` failure. Only the HMGET half is reachable
/// — `restore_rebase_test.rs` (~:1443-1452) already records why: the HSET is an
/// unconditional `hset` on the data pool with no fault seam, and any stimulus
/// crude enough to fail it fails the HMGET one statement earlier. The HSET half
/// is therefore a DOCUMENTED-UNREACHABLE TWIN, not a gap: both producers return
/// the same variant, tick the same cell, and tick the same
/// `writes_total{error, layer}` cell, so the cell mapping this case pins is the
/// whole of what the adjudication assigned to that arm.
///
/// The stimulus needs the gate for a reason worth stating: step 1 reads `obj:hdr`
/// too, so a WRONGTYPE planted before `run_once` would fail the identity read and
/// take the `failed` arm instead. Planting it while the PUT is PARKED is what puts
/// the failure strictly between a successful identity read and the pre-stamp
/// re-read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrongtype_header_ticks_unstamped_write_error_on_the_pre_stamp_hmget() {
    let fx = outcome_fixture(2, true, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-unstamped-write-error"), 16)
        .await;
    fx.stamp_restore_shaped_header(object_id).await;

    let delegated_before = fx.store.base_puts_delegated();
    let pre = fx.cells();

    let handle = park_a_base_put(&fx, object_id, 16, fx.restore_target()).await;
    fx.wrongtype_header(object_id).await;

    fx.release_gate();
    let outcome = handle.await.expect("the run_once task must not panic");

    assert_eq!(
        outcome,
        BootstrapOutcome::UnstampedWriteError,
        "non-vacuity: the fixture must reach the pre-stamp HMGET failure. A `Failed` here would \
         mean the WRONGTYPE landed before the step-1 identity read, which is a different arm and a \
         different cell"
    );
    assert_eq!(
        fx.store.base_puts_delegated(),
        delegated_before + 1,
        "the base IS durable at the target and only the header disagrees — that is the whole \
         reason this arm is a different cell from `failed`, which means no base landed at all"
    );
    assert!(
        fx.listing(object_id)
            .await
            .iter()
            .any(|h| h.snap_seq == REBASE_SEEDED_SEQ && h.kind == CheckpointKind::Base),
        "the durable base must be visible in the listing: the next eligible tick re-snapshots this \
         layer rather than the layer having no lineage root at all"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("unstamped_write_error", "restore_rebase", 1)],
        0,
        // Additive, not a substitute: this arm is the one `unstamped_*` member
        // that ALSO ticks the generation-completion family's error arm, because a
        // Valkey command genuinely failed.
        1,
        "unstamped write error (HMGET half)",
    );
}

// =============================================================================
// Case 8 — the failure arm.
// =============================================================================

/// A failed base PUT ticks `{failed, creation_base}` AND
/// `writes_total{error, layer}`, and leaves no generation behind.
///
/// One classification only, deliberately: `run_once` takes the identical terminal
/// arm for `Transient` and `Permanent` (the classification rides the log line,
/// not the control flow), and the two-classification matrix is already pinned by
/// `first_base_bootstrap_test.rs::assert_put_failure_stamps_nothing` and
/// `restore_rebase_test.rs`'s cases 5a/5b. What is pinned HERE is the cell, which
/// is classification-independent by construction.
///
/// The one `failed` producer that does NOT tick `writes_total{error, layer}` is
/// the `Disabled` adapter's `NotImplemented` return, which the three-way gate
/// makes unreachable — ticking an adapter error for "checkpointing is off" would
/// manufacture a noise floor. That asymmetry is a production comment, not a
/// reachable fixture.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_base_put_ticks_failed_and_the_writes_error_cell() {
    let fx = outcome_fixture(2, false, None).await;
    let object_id = fx
        .create_layer(&format!("{LAYER_PREFIX}-failed-put"), 17)
        .await;

    fx.store.fail_base_puts();
    let attempts_before = fx.store.base_attempts();
    let pre = fx.cells();

    let outcome = fx
        .unit
        .run_once(object_id, seed_state(17), BootstrapTarget::CreationBase)
        .await;

    assert_eq!(
        outcome,
        BootstrapOutcome::Failed,
        "non-vacuity: the fixture must reach the PUT-failure arm"
    );
    assert_eq!(
        fx.store.base_attempts(),
        attempts_before + 1,
        "the bootstrap makes exactly ONE best-effort put_layer_checkpoint attempt — a count above \
         one means a retry loop appeared"
    );
    assert!(
        fx.listing(object_id).await.is_empty(),
        "a failed PUT must leave NO generation behind"
    );
    assert!(
        fx.header_field(object_id, "last_checkpoint_seq")
            .await
            .is_none(),
        "SR-7.2 stamps-nothing at the creation site: the columns were never written, so they must \
         still be ABSENT (the restore site's correct claim is `unchanged`, and \
         restore_rebase_test.rs owns it)"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        &[("failed", "creation_base", 1)],
        0,
        1,
        "failed base PUT",
    );
}

// =============================================================================
// Case 9 — the tenth tail outcome, through the capture path.
// =============================================================================

/// A tails-only lineage is refused `refused_lineage_unrootable`, NOT
/// `refused_no_base` — the split the SR-7 adjudication landed as the tenth
/// `CHECKPOINT_TAIL_OUTCOMES` member.
///
/// The two conditions had shared one cell and have opposite remediations: a
/// `refused_no_base` layer needs nothing done to it (the next eligible tick roots
/// a lineage and the condition drains on its own), while a `LineageUnrootable`
/// layer needs a HUMAN to repair the store — no shipped write path can produce a
/// tails-only lineage, since a tail is only ever written after the base it names,
/// so it means manual pruning of the base or partial checkpoint-store loss.
/// Folding them left an operator with an alertable condition hidden inside a
/// benign one.
///
/// **The capture unit is metrics-free BY DESIGN** (its module header): it decides
/// and persists, and the tick wiring maps the decision onto
/// `live_edit_checkpoint_tail_total`. So the assertion surface here is the
/// decision variant, the outcome LABEL, and the fact that the label is a member of
/// the pinned ten-value vocabulary with a zero-seeded series waiting for it — plus
/// the two silences this file owns: no tail cell moves (the unit does not tick),
/// and no bootstrap cell moves either, because no `run_once` ran. That second
/// silence is what keeps the sum identity honest: no invocation, no tick.
/// The label→cell mapping itself is pinned loop-side by
/// `checkpointer_tail_cadence_test.rs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tails_only_lineage_is_refused_lineage_unrootable_not_no_base() {
    let fx = outcome_fixture(2, false, None).await;
    let layer_name = format!("{LAYER_PREFIX}-tails-only");
    let object_id = fx.create_layer(&layer_name, 18).await;

    // A lineage of tails and nothing else — the shape no shipped write path can
    // produce. `base_snap_seq` names a base that is NOT in the store, which is
    // precisely the "manual pruning / partial store loss" state.
    const ABSENT_BASE_SNAP_SEQ: u64 = 40;
    put_tail_generation(
        &fx.store_dyn,
        object_id,
        &fx.session_id,
        &layer_name,
        ABSENT_BASE_SNAP_SEQ,
        ABSENT_BASE_SNAP_SEQ,
        ABSENT_BASE_SNAP_SEQ + 5,
    )
    .await;
    let listing_before = fx.listing(object_id).await;
    assert!(
        !listing_before.is_empty()
            && listing_before
                .iter()
                .all(|h| h.kind == CheckpointKind::Tail),
        "precondition: the listing must be NON-EMPTY and contain no Base at all. An empty listing \
         takes the `refused_no_base` arm one branch earlier and this case would pin the wrong \
         verdict; got {listing_before:?}"
    );

    let pre = fx.cells();
    let decision = fx.capture(object_id).await;

    assert!(
        matches!(
            decision,
            TailCaptureDecision::Refused(TailCaptureRefusal::LineageUnrootable)
        ),
        "non-vacuity: a tails-only listing must take the LineageUnrootable arm; got {decision:?}"
    );
    assert_eq!(
        decision.outcome_label(),
        "refused_lineage_unrootable",
        "the tails-only arm must NOT report refused_no_base — the two conditions have opposite \
         remediations and the SR-7 adjudication paid a series to separate them"
    );
    assert!(
        CHECKPOINT_TAIL_OUTCOMES.contains(&decision.outcome_label()),
        "the label must be a member of the pinned tail vocabulary, or it would open an unseeded \
         series"
    );
    assert!(
        pre.tail.contains_key("refused_lineage_unrootable"),
        "the tenth member must already exist as a ZERO-SEEDED series before anything fires — an \
         additive vocabulary member that is not seeded reads as an absent series until its first \
         tick, which is exactly the alerting gap the seeding loop exists to close"
    );
    assert_eq!(
        fx.listing(object_id).await,
        listing_before,
        "a refusal persists nothing: the listing must be byte-identical afterwards"
    );

    let post = fx.cells();
    assert_cells_delta(
        &pre,
        &post,
        // No `run_once` ran, so the bootstrap family must be entirely flat. That
        // "no invocation, no tick" property is what makes the family's sum a
        // usable denominator.
        &[],
        0,
        0,
        "tails-only capture refusal",
    );
}

// =============================================================================
// Case 10 — the sum identity.
// =============================================================================

/// `sum(live_edit_checkpoint_base_bootstrap_total)` moves by exactly the number of
/// `run_once` invocations, across a MIX of arms, with no cell outside the pinned
/// vocabulary.
///
/// This is the only direct fence on the property every ratio in the SR-7
/// adjudication depends on. The family's benign cells are usable as a denominator
/// only if each invocation contributes exactly one tick — no more (a second
/// increment on some path would inflate one arm's share) and no fewer (a silent
/// exit would deflate the denominator and make every alert threshold read high).
///
/// The mix is deliberately five DIFFERENT arms across BOTH sites, including one
/// that completes a generation, one that fails, and three that decline — so the
/// identity is asserted over the whole spread rather than over a single repeated
/// arm, where a per-arm double tick would cancel against the count.
///
/// The creation-site `shed` / `persisted` pair is deliberately absent from the
/// mix: those two cells are fenced end-to-end by
/// `bootstrap_concurrency_bound_test.rs`'s case 6, and restating them here would
/// duplicate rather than extend.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_family_sums_to_exactly_one_tick_per_run_once_invocation() {
    let fx = outcome_fixture(2, false, None).await;
    let pre = fx.cells();

    let mut outcomes: Vec<BootstrapOutcome> = Vec::new();

    // 1. `{persisted, restore_rebase}` — a re-base that completes.
    let persisted_oid = fx
        .create_layer(&format!("{LAYER_PREFIX}-identity-persisted"), 21)
        .await;
    fx.stamp_restore_shaped_header(persisted_oid).await;
    outcomes.push(
        fx.unit
            .run_once(persisted_oid, seed_state(21), fx.restore_target())
            .await,
    );

    // 2. `{skipped_generation_present, creation_base}` — guard (a).
    let guard_a_oid = fx
        .create_layer(&format!("{LAYER_PREFIX}-identity-guard-a"), 22)
        .await;
    fx.set_header_field(guard_a_oid, "last_checkpoint_seq", "500")
        .await;
    outcomes.push(
        fx.unit
            .run_once(guard_a_oid, seed_state(22), BootstrapTarget::CreationBase)
            .await,
    );

    // 3. `{skipped_header_absent, creation_base}` — the delete/overwrite race.
    let absent_oid = fx
        .create_layer(&format!("{LAYER_PREFIX}-identity-absent"), 23)
        .await;
    fx.unlink_header(absent_oid).await;
    outcomes.push(
        fx.unit
            .run_once(absent_oid, seed_state(23), BootstrapTarget::CreationBase)
            .await,
    );

    // 4. `{skipped_untrusted_cursor, restore_rebase}` — the unparseable column.
    let untrusted_oid = fx
        .create_layer(&format!("{LAYER_PREFIX}-identity-untrusted"), 24)
        .await;
    fx.stamp_restore_shaped_header(untrusted_oid).await;
    fx.set_header_field(untrusted_oid, "restore_seq_bump", "not-a-number")
        .await;
    outcomes.push(
        fx.unit
            .run_once(untrusted_oid, seed_state(24), fx.restore_target())
            .await,
    );

    // 5. `{failed, creation_base}` — armed last, so it cannot disturb the four
    //    invocations above.
    let failed_oid = fx
        .create_layer(&format!("{LAYER_PREFIX}-identity-failed"), 25)
        .await;
    fx.store.fail_base_puts();
    outcomes.push(
        fx.unit
            .run_once(failed_oid, seed_state(25), BootstrapTarget::CreationBase)
            .await,
    );

    // ── Non-vacuity: the five arms the identity is asserted over ─────────
    assert!(
        matches!(outcomes[0], BootstrapOutcome::Persisted { .. }),
        "invocation 1 must complete a re-base; got {:?}",
        outcomes[0]
    );
    assert_eq!(outcomes[1], BootstrapOutcome::SkippedGenerationPresent);
    assert_eq!(outcomes[2], BootstrapOutcome::SkippedHeaderAbsent);
    assert_eq!(outcomes[3], BootstrapOutcome::SkippedUntrustedCursor);
    assert_eq!(outcomes[4], BootstrapOutcome::Failed);
    let invocations = outcomes.len() as u64;
    assert_eq!(
        invocations, 5,
        "the mix must stay five invocations across five distinct arms, or the identity below is \
         asserting a different number than the case documents"
    );

    let post = fx.cells();
    // `assert_cells_delta` asserts the per-cell deltas AND their sum against the
    // invocation count, and it asserts the key sets are unchanged — which is the
    // "no cell outside the pinned vocabulary" half.
    assert_cells_delta(
        &pre,
        &post,
        &[
            ("persisted", "restore_rebase", 1),
            ("skipped_generation_present", "creation_base", 1),
            ("skipped_header_absent", "creation_base", 1),
            ("skipped_untrusted_cursor", "restore_rebase", 1),
            ("failed", "creation_base", 1),
        ],
        1,
        1,
        "sum identity over a five-arm mix",
    );

    let summed: f64 = post
        .bootstrap
        .iter()
        .map(|(cell, after)| after - pre.bootstrap.get(cell).copied().unwrap_or(0.0))
        .sum();
    assert_eq!(
        summed, invocations as f64,
        "sum(live_edit_checkpoint_base_bootstrap_total) must move by exactly the number of \
         run_once invocations ({invocations}). Stated separately from the per-cell assertion on \
         purpose: this is the identity the family's consumers read `sum(...)` against, and it is \
         what makes the benign cells a usable denominator"
    );
}
