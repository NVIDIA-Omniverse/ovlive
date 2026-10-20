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

//! SR-3.6 post-restore gap re-base — landing evidence, restore-site cursor
//! semantics, the failure arm, and the fossil-free follow-up. Task
//! `t2.dfh.yj.4f`, Subtask `t3.dfh.yj.4f.gw`.
//!
//! ## What is under test
//!
//! Every DR restore seeds the live writer cursor at `seeded_seq = restore_floor
//! + band` with `band >= 1` unconditionally (post-b.8y5, `apply.rs` step 4), so
//! `seeded_seq > snap_seq` on EVERY restore — base-only and composed alike — and
//! SR-3.6's old conditional no-gap arm is dead. The periodic loop would close
//! the gap only on the layer's next eligible tick, whose proven read lands a
//! base and clears `restore_seq_bump`; to cover the layer from the restore on,
//! [`RecoveryHook::try_restore_layer`]'s applied arm hands the already-in-memory
//! restored state straight to `BaseBootstrap::spawn_restore_rebase`, which
//! persists it as a fresh [`CheckpointKind::Base`] at `seeded_seq` and stamps
//! THREE header cursor columns (`last_checkpoint_seq`, `last_checkpoint_ms`, and
//! `restore_seq_bump = 0`).
//!
//! Two landed pieces make that observable, and both are pinned here:
//!
//! - `LayerStore::write_object`'s `CheckpointCursorStamp` stamps
//!   `last_checkpoint_seq = snap_seq`, `last_checkpoint_ms` (the restored blob's
//!   own `written_at_unix_ms`) and `restore_seq_bump = band` inside the
//!   restore's OWN MULTI/EXEC — **before** the async re-base runs. The
//!   restore-site guard semantics are therefore NOT the creation-time ones: the
//!   creation-time guard's `last_checkpoint_seq != 0` early-abort is non-zero on
//!   every single restore and would kill every re-base. The landed guard is
//!   stated relative to the target (`observed >= target.snap_seq()`), which
//!   admits the restore's own audit stamp because it sits a whole band below
//!   `seeded_seq`. Test 4a is the assertion that catches a regression back to
//!   the `!= 0` form.
//! - `tail_capture`'s post-restore fossil guard refuses
//!   `TailCaptureRefusal::PostRestoreUnrebased` (outcome label
//!   `refused_seq_gap`) whenever `restore_seq_bump != 0` or
//!   `cursor_columns_trusted == false`, **before** the stream read. A successful
//!   re-base is fossil-free only if it BOTH lands a Base at `seeded_seq` (moving
//!   S0 above every fossil) AND clears that stamp. Test 6 proves both halves.
//!
//! ## The other re-base, and the live invariant that separates them
//!
//! Every "re-base" in this file is SR-3.6's post-restore one, described above.
//! The service has a second: the **SR-4.1 chain-collapse re-base**
//! (`src/checkpoint/rebase.rs`), which the checkpoint tick runs when a hot
//! layer's tail chain reaches K and which ticks
//! `live_edit_checkpoint_tail_total{outcome="rebased"}`.
//!
//! **SR-3.6's re-base must never tick `rebased`.** That is a live, falsifiable
//! invariant now that the cell has a real producer, and this file is where it is
//! pinned — Case 1's vocabulary fence asserts the whole tail-outcome family is
//! byte-for-byte unmoved across a completed post-restore re-base. It matters
//! because the cell's meaning is "a hot layer's chain was collapsed": folding a
//! recovery write into it would make a DR restore of N layers read as N chain
//! collapses, and bury the chain-health signal under a recovery that has nothing
//! to do with it. The two producers are also reached from different places —
//! this one from the recovery hook, that one from the tick's capture arm — so a
//! shared cell would not even be recoverable by label.
//!
//! ## Why a NEW file rather than extending `first_base_bootstrap_test.rs`
//!
//! That file's header declares hard scope fences this lane would have to break
//! ("No checkpoint loop", "No `refused_no_base`, no tail capture"), and its
//! `bootstrap_fixture` creates a layer and drives `run_once` with no
//! store-planted generation, no state wipe, and no `RecoveryHook`. Every case
//! here needs a planted lineage + a wipe + a hook-level restore. The fixtures
//! genuinely diverge.
//!
//! **No existing test file's assertions were modified for this Subtask.** If one
//! had had to be, that is an escalation signal — the shared `run_once` would
//! have altered creation-time behaviour, which Task 4 forbids.
//!
//! ## Why this module is UNGATED
//!
//! No `#![cfg(feature = "test-checkpoint-fault-injection")]`, deliberately. A
//! gated module compiles to nothing and reports green
//! (`docs-internal/test-writing-guide.md`), and this file is the only
//! integration-level evidence for SR-3.6 — it would silently cover nothing under
//! a feature-free narrow run. Failure injection uses the documented preference,
//! a hand-rolled [`DelegatingCheckpointStore`] newtype
//! ([`RecordingRebaseStore`]), which unlike the `CheckpointFaultInjector` seam
//! can target `(snap_seq, kind)` and so separates the re-base PUT from the
//! loop's own.
//!
//! [`DelegatingCheckpointStore`]: super::helpers::DelegatingCheckpointStore
//! [`RecoveryHook::try_restore_layer`]: live_edit::checkpoint::RecoveryHook::try_restore_layer
//!
//! ## The landed `restore_seq_bump` value, pinned
//!
//! Every case asserts the column explicitly rather than ignoring it.
//! [`LANDED_RETIRED_BUMP`] is `0` — the value
//! `BootstrapTarget::cursor_pairs` writes on the `RestoreRebase` arm, i.e. the
//! bump is RETIRED. The failure and skip arms assert the column is UNCHANGED
//! (still carrying the restore's own `band`), never merely "absent".
//!
//! ## The PUT-ok / HSET-fails arm — documented posture, not a test
//!
//! `run_once` step 6's cursor stamp is an unconditional `hset` on the data pool
//! with no injection seam, and the only stimulus crude enough to fail it (making
//! `obj:hdr` the wrong Valkey type) fails the pre-stamp `HMGET` one statement
//! earlier instead. Reaching that arm literally would require NEW production
//! fault machinery, which this Task does not add. Its **observable consequence**
//! is nevertheless covered, at the end of Test 4b: guard (b)'s skip produces the
//! identical residual — a durable Base at `seeded_seq` whose `restore_seq_bump`
//! was never retired — and the test spends it by showing the layer's next
//! capture still refuses `refused_seq_gap`. So the `refused_seq_gap` residual
//! includes "a re-base whose blob landed but whose cursor stamp did not": such a
//! layer stays refusing until its next eligible periodic checkpoint tick lands a
//! base (its proven read retires the bump with its cursor stamp), or it is
//! re-restored.
//!
//! ## Bounded waits — no `sleep` is used for synchronization
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | [`RestoreRebaseFixture::drain`] (terminal `close()` + `wait()`) | [`DRAIN_TIMEOUT`] (10 s) | a re-base task never completed, or was spawned onto a different tracker |
//! | `helpers::wait_for_writer_idle` (non-terminal drain) | [`TRACKER_IDLE_DEADLINE_SECS`] (10 s, clamped up to the helpers' load-tolerant floor) | same, for a fixture whose tracker must stay open |
//! | [`RecordingRebaseStore`] gate park | [`GATE_WAIT`] (10 s) | the test never released the gate; the PUT then proceeds so the case fails on an assertion rather than hanging |
//! | [`poll_until`] — "the gated PUT reached the wrapper" | [`GATE_WAIT`] (10 s) | the re-base never issued its PUT, which is exactly the symptom a regression to the creation-time guard would produce (Test 4a names it) |
//! | `helpers::drive_captured_tail` (Test 3 only; `test-checkpoint-fault-injection`) | its own 30 s internal deadline | the composed-lineage fixture never produced a real tail |
//!
//! Worst case per test is Test 3 at roughly 30 s + 10 s, comfortably inside
//! `.config/nextest.toml`'s 120 s hard kill — a SIGKILLed test prints none of
//! its diagnostics.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, StreamsInterface};
use tokio::sync::oneshot;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::tail_capture::{
    capture_layer_tail, TailCaptureDecision, TailCaptureRefusal,
};
use live_edit::checkpoint::{
    decode_tail_envelope, BaseBootstrap, BootstrapOutcome, BootstrapTarget, CheckpointAdapter,
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta,
    CheckpointStore, RecoveryHook, RecoveryOutcome, TailCaptureConfig,
};
use live_edit::config::{Config, PodMemorySource};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::helpers::drive_captured_tail;
use super::helpers::{
    gather_tail_outcome_cells, put_base_generation, wait_for_writer_idle, wipe_layer_state,
    wipe_layer_state_preserving_stream, DelegatingCheckpointStore, HAND_PUT_WRITTEN_AT_UNIX_MS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Replica id stamped into every persisted `CheckpointMeta` and carried as this
/// file's registry label. Note `checkpoint_writes_total` renders NO `service`
/// label, so it is not a metric selector here — cross-test isolation comes from
/// the per-fixture registry (`crate::common::metrics_scrape_only`).
const SERVICE_LABEL: &str = "restore-rebase-test-replica";

/// `sequence_num` every fixture layer is created at, and the `snap_seq` of the
/// base planted as its lineage root.
///
/// The pairing is deliberate and mirrors `helpers::CAPTURED_TAIL_SEED_SEQ`'s:
/// a lineage root at the layer's own header sequence means the restore's
/// `restore_floor` is exactly `snap_seq` on the wiped-stream shape, so
/// `seeded_seq` is one clean band above it.
const SEED_SEQ: u64 = 40;

/// Stream retention for the fixture's `LayerStore` — the production default.
const STREAM_MAXLEN: u64 = 4096;

const CREATED_BY: &str = "restore-rebase-test-user";
const WRITER_CLIENT_ID: &str = "restore-rebase-test-writer";
const SESSION_BASE_PATH: &str = "/restore-rebase";

/// The landed value of `obj:hdr restore_seq_bump` after a SUCCESSFUL re-base.
///
/// `0` — RETIRED, written as the third pair of the re-base's single HSET
/// (`BootstrapTarget::cursor_pairs`). Retiring it is what actually unblocks tail
/// capture: `tail_capture` refuses `PostRestoreUnrebased` while it is non-zero,
/// *before* the stream read. Named rather than spelled inline so every case
/// asserts against the contract, and so a change to the landed posture is one
/// edit here plus a wave of failures rather than a silent drift.
const LANDED_RETIRED_BUMP: u64 = 0;

/// Bound on a terminal tracker drain. A hang here is a `BaseBootstrap` bug (a
/// task that never completes, or one spawned onto a different tracker), not a
/// test-only flake.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Deadline handed to `helpers::wait_for_writer_idle` for the non-terminal
/// drains. The helper clamps it up to its own load-tolerant floor.
const TRACKER_IDLE_DEADLINE_SECS: u64 = 10;

/// Ceiling on how long a gated re-base PUT parks before giving up and
/// proceeding. Only reached when a test forgot to release it — the PUT then
/// completes and the case fails on an assertion instead of hanging the binary.
const GATE_WAIT: Duration = Duration::from_secs(10);

/// Post-restore commits every fossil-free case lands before capturing.
///
/// Two, not one: the capture's claimed range must EQUAL its carried content
/// (b.pw6), and a single-record envelope makes an off-by-one in that
/// relationship unobservable.
const POST_RESTORE_COMMITS: u64 = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Gate derivation — derive from `Config`, never hardcode
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `Config` with the three SR-7.1 gate inputs set explicitly, through
/// the REAL `Config::from_env` under `temp_env::with_vars`.
///
/// Same shape as `first_base_bootstrap_test.rs::bootstrap_config`: the parsed
/// values are the ones a deployment with these env vars would actually get,
/// rather than a hand-assembled struct that could drift from the parser. The two
/// endpoint vars are required unconditionally by `Config::from_env` and are
/// irrelevant to this file; `LIVE_EDIT_SERVICE_LABEL` is pinned so the hook's
/// own `service_label` and the bootstrap's `creator_replica_id` agree, exactly
/// as `main.rs` makes them agree.
fn rebase_config(checkpoint_enabled: bool, adapter: &str, tail_enabled: bool) -> Config {
    let enabled_str = if checkpoint_enabled { "true" } else { "false" };
    let tail_str = if tail_enabled { "true" } else { "false" };
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
            ("LIVE_EDIT_CHECKPOINT_ENABLED", Some(enabled_str)),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some(adapter)),
            ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some(tail_str)),
        ],
        || {
            Config::from_env_with(PodMemorySource::Reference)
                .expect("Config::from_env must succeed for the SR-3.6 re-base gate")
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

/// All three gate inputs on — the shape every case in this file runs under.
fn all_gates_on_config() -> Config {
    rebase_config(true, "InMemory", true)
}

// ─────────────────────────────────────────────────────────────────────────────
// The store wrapper every case shares
// ─────────────────────────────────────────────────────────────────────────────

/// One `put_layer_checkpoint` attempt, as the wrapper saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
}

/// Classification of the synthetic adapter failure Test 5 injects.
///
/// SR-7.2's stamps-nothing contract is classification-independent — `run_once`
/// logs `is_transient` / `is_permanent` for the operator but takes the identical
/// terminal arm either way — so both arms run, matching
/// `first_base_bootstrap_test.rs`'s `assert_put_failure_stamps_nothing` matrix.
#[derive(Debug, Clone, Copy)]
enum FailureMode {
    Transient,
    Permanent,
}

/// `CheckpointStore` newtype that records every layer PUT and can GATE or FAIL
/// the re-base PUT specifically.
///
/// # Targeting
///
/// The re-base writes `(object_id, seeded_seq, CheckpointKind::Base)`, and
/// `seeded_seq` is not knowable before the restore commits — so the gate and the
/// fail arm key on `kind == Base` and are **armed** by the test only after its
/// lineage has been planted. Every fixture plants its base(s) through
/// `helpers::put_base_generation` (or, in Test 3, through
/// `helpers::drive_captured_tail`) BEFORE arming, so the first `Base` PUT the
/// wrapper sees while armed is necessarily the re-base. Tests assert the
/// recorded `snap_seq` against the header they read back, so a mis-targeted arm
/// surfaces as a failed assertion rather than as a silent pass.
///
/// # Delegation
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would
/// not leave a call un-instrumented, it would SUBSTITUTE behaviour behind the
/// wrapper and make the sidecar / listing assertions vacuously green — the trap
/// that helper's doc records.
struct RecordingRebaseStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    attempts: Mutex<Vec<PutAttempt>>,
    /// Non-`None` once armed: the re-base PUT returns this classification
    /// instead of delegating.
    fail_base_put: Mutex<Option<FailureMode>>,
    /// Taken by the first armed `Base` PUT, which then parks on it (bounded by
    /// [`GATE_WAIT`]) before delegating.
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    /// Counts `Base` PUTs that actually reached the delegate (i.e. were not
    /// failed), so "the blob landed even though the stamp did not" is directly
    /// assertable in Test 4b.
    base_puts_delegated: AtomicUsize,
}

impl RecordingRebaseStore {
    fn new() -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            attempts: Mutex::new(Vec::new()),
            fail_base_put: Mutex::new(None),
            gate: Mutex::new(None),
            base_puts_delegated: AtomicUsize::new(0),
        }
    }

    /// Park the next `Base` PUT until the returned sender fires.
    fn arm_gate(&self) -> oneshot::Sender<()> {
        let (tx, rx) = oneshot::channel();
        *self.gate.lock().unwrap_or_else(|p| p.into_inner()) = Some(rx);
        tx
    }

    /// Fail every subsequent `Base` PUT with `mode`.
    fn fail_base_puts(&self, mode: FailureMode) {
        *self.fail_base_put.lock().unwrap_or_else(|p| p.into_inner()) = Some(mode);
    }

    fn attempts(&self) -> Vec<PutAttempt> {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Every `Base` PUT recorded after `from_index` (a snapshot of
    /// `attempts().len()` taken before the act step).
    fn base_puts_since(&self, from_index: usize) -> Vec<PutAttempt> {
        self.attempts()
            .into_iter()
            .skip(from_index)
            .filter(|a| a.kind == CheckpointKind::Base)
            .collect()
    }

    fn base_puts_delegated(&self) -> usize {
        self.base_puts_delegated.load(Ordering::Acquire)
    }

    fn synthesise_error(mode: FailureMode) -> CheckpointError {
        match mode {
            FailureMode::Transient => CheckpointError::Transient {
                source: "synthetic transient re-base PUT failure (SR-3.6 test)".into(),
            },
            FailureMode::Permanent => CheckpointError::Permanent {
                source: "synthetic permanent re-base PUT failure (SR-3.6 test)".into(),
            },
        }
    }
}

#[async_trait]
impl CheckpointStore for RecordingRebaseStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE any gate or failure arm: the test wants to know the
        // call was ISSUED, whatever happens to it afterwards.
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                object_id,
                snap_seq,
                kind: meta.kind,
            });

        if meta.kind == CheckpointKind::Base {
            let gated = self.gate.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(rx) = gated {
                // Bounded: a test that forgot to release must fail on an
                // assertion, not hang until nextest's hard kill.
                let _ = tokio::time::timeout(GATE_WAIT, rx).await;
            }
            let failure = *self.fail_base_put.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(mode) = failure {
                return Err(Self::synthesise_error(mode));
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

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the restore-source selection whose
    // whole job is deciding what exists.
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
    // EMPTY set, which would make every sidecar-shaped assertion vacuously
    // green.
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

/// Everything one restore-re-base case needs, owned together so `TestValkey`'s
/// prefix-scoped Drop cleanup outlives the assertions.
///
/// Modelled on `first_base_bootstrap_test.rs::bootstrap_fixture` but
/// restore-shaped: a real `SessionStore`-created session, a real `LayerStore`, a
/// real `ValkeyExternalValueStore`, the shared `TaskTracker` graceful shutdown
/// owns, a `SharedMetrics` over this file's own registry, a
/// [`RecordingRebaseStore`], and a real (enabled) `RecoveryHook` wired the way
/// `main.rs` wires it.
struct RestoreRebaseFixture {
    tv: TestValkey,
    session_id: String,
    layers: Arc<LayerStore>,
    ext_values: Arc<ValkeyExternalValueStore>,
    tracker: TaskTracker,
    metrics: SharedMetrics,
    store: Arc<RecordingRebaseStore>,
    store_dyn: Arc<dyn CheckpointStore>,
    /// The SAME unit the hook holds when `wire_rebase` was `true` — retained so
    /// the direct-drive cases (Test 4c) can call `run_once` with a target
    /// assembled from values READ off the header.
    bootstrap: Arc<BaseBootstrap>,
    hook: RecoveryHook,
    config: Config,
}

/// Wire the fixture. `wire_rebase == false` builds the hook WITHOUT the
/// bootstrap — the documented `None` posture, i.e. no re-base is spawned — which
/// is what lets Test 4c corrupt the header between the restore and a directly
/// awaited `run_once` instead of racing a fire-and-forget task.
async fn restore_rebase_fixture(wire_rebase: bool) -> RestoreRebaseFixture {
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the all-gates-on Config must resolve the SR-7.1 three-way gate to TRUE, otherwise every \
         case in this file would be asserting against a unit that cannot write"
    );

    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let tracker = TaskTracker::new();

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("restore-rebase-session", SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed against a healthy Valkey");

    let layers = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        STREAM_MAXLEN,
    ));
    let ext_values = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let store = Arc::new(RecordingRebaseStore::new());
    let store_dyn: Arc<dyn CheckpointStore> = Arc::clone(&store) as Arc<dyn CheckpointStore>;

    let bootstrap = Arc::new(BaseBootstrap::new(
        Arc::clone(&store_dyn),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        cfg.service_label.clone(),
        enabled,
    ));

    // Hand-built on purpose, NOT a surviving `CheckpointHarness` workaround: this
    // file's subject IS the wiring matrix — every case varies the `rebase_config`
    // gate inputs and whether `.with_base_bootstrap(..)` is called at all, both of
    // which the harness fixes at construction time.
    let mut hook = RecoveryHook::new(
        Arc::clone(&store_dyn),
        Arc::clone(&layers),
        Arc::new(cfg.clone()),
        Arc::clone(&metrics),
    )
    .with_ext_value_store(Arc::clone(&ext_values) as Arc<dyn ExternalValueStore>);
    if wire_rebase {
        hook = hook.with_base_bootstrap(Arc::clone(&bootstrap));
    }

    RestoreRebaseFixture {
        tv,
        session_id,
        layers,
        ext_values,
        tracker,
        metrics,
        store,
        store_dyn,
        bootstrap,
        hook,
        config: cfg,
    }
}

impl RestoreRebaseFixture {
    /// Materialize a real layer at [`SEED_SEQ`] and plant its lineage root base
    /// at the same sequence, returning the freshly-allocated `object_id`.
    ///
    /// The two steps are one method because the pairing is load-bearing: a base
    /// at the layer's own header sequence is what makes `restore_floor ==
    /// snap_seq` on the wiped-stream shape, so `seeded_seq` is one clean band
    /// above the generation restored.
    async fn create_layer_with_base(&self, layer_name: &str, seed: u8) -> u64 {
        let state = seed_state(seed, SEED_SEQ);
        let object_id = self
            .layers
            .create_layer(
                &self.session_id,
                layer_name,
                CREATED_BY,
                &state,
                Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>,
            )
            .await
            .expect("create_layer must succeed against a healthy Valkey");

        put_base_generation(
            &self.store_dyn,
            object_id,
            &self.session_id,
            layer_name,
            SEED_SEQ,
        )
        .await;

        object_id
    }

    /// One real `UpdateLayer` commit carrying a REAL `relay_ompf`.
    ///
    /// Both halves matter: a field-only delta rides the Lua fast path, and an
    /// `ompf`-less update entry is classified as an EMPTY commit by invariant —
    /// a fixture built that way records real commits as no-ops and tests a lie
    /// (b.pw6). Returns the committed `sequence_num`.
    async fn commit(&self, layer_name: &str, object_id: u64, nonce: u64) -> u64 {
        let delta = oodf_ffi::OwnedDelta {
            is_diff: true,
            base_version: 1,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "restore-rebase-field".to_string(),
                // Distinct per commit, so a silently-dropped delta cannot be
                // mistaken for one that was applied.
                value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: nonce,
            }],
            ..Default::default()
        };
        let ompf = Bytes::from(crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
        let (sequence_num, _) = self
            .layers
            .update_layer(
                &self.session_id,
                layer_name,
                object_id,
                Some(delta),
                Some(ompf),
                WRITER_CLIENT_ID,
                0,
            )
            .await
            .expect("update_layer must succeed against a healthy Valkey");
        sequence_num
    }

    /// Land [`POST_RESTORE_COMMITS`] commits on a just-restored layer.
    async fn land_post_restore_commits(&self, layer_name: &str, object_id: u64) -> Vec<u64> {
        let mut seqs = Vec::new();
        for i in 0..POST_RESTORE_COMMITS {
            seqs.push(self.commit(layer_name, object_id, 90_000 + i).await);
        }
        seqs
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

    /// `HGET`, parsed as `u64`. Panics on absence or on an unparseable value —
    /// an absent cursor column read as `0` would make several assertions in this
    /// file pass for the wrong reason.
    async fn header_u64(&self, object_id: u64, field: &str) -> u64 {
        let raw = self
            .header_field(object_id, field)
            .await
            .unwrap_or_else(|| panic!("obj:hdr `{field}` must be PRESENT for object {object_id}"));
        raw.parse::<u64>()
            .unwrap_or_else(|_| panic!("obj:hdr `{field}` must parse as u64; got {raw:?}"))
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

    /// The live writer cursor `seq` — i.e. the restore's `seeded_seq`. READ,
    /// never recomputed from the b.8y5 seeding formula (duplicating it in a test
    /// guarantees drift against it).
    async fn seeded_seq(&self, object_id: u64) -> u64 {
        self.header_u64(object_id, "seq").await
    }

    async fn listing(&self, object_id: u64) -> Vec<CheckpointHandle> {
        self.store_dyn
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed")
    }

    /// `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}` off
    /// this fixture's own registry.
    ///
    /// Matched on `(result, kind)` only: this family renders NO `service` label,
    /// and a `service` term would match zero lines and fold to a silent `0`.
    fn layer_writes(&self, result: &str) -> u64 {
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&self.metrics),
            "live_edit_checkpoint_writes_total",
            &[("result", result), ("kind", "layer")],
        )
        .unwrap_or(0.0) as u64
    }

    /// The loop's own tail knobs, re-derived from this fixture's `Config` — the
    /// same single source of truth `main.rs` and the checkpoint loop read.
    fn tail_config(&self) -> TailCaptureConfig {
        TailCaptureConfig::from_config(&self.config)
    }

    /// Invoke the tail-capture unit DIRECTLY (it is `pub`; the precedents are
    /// `checkpointer_tail_duplicate_leader_test.rs` and
    /// `checkpointer_tail_outcomes_test.rs`).
    ///
    /// Deliberately NOT the periodic loop: `capture_layer_tail` is the narrow,
    /// deterministic instrument, and the loop-level E2E belongs to Task 6
    /// (`t2.dfh.yj.u3`).
    async fn capture(&self, object_id: u64) -> TailCaptureDecision {
        capture_layer_tail(
            self.tv.client(),
            self.tv.key_schema(),
            object_id,
            self.store_dyn.as_ref(),
            SERVICE_LABEL,
            &self.tail_config(),
        )
        .await
        .expect("capture must not error; every arm under test is a decision, not a failure")
    }

    /// `XINFO STREAM`-free high-water read: the last entry id in the per-object
    /// stream, as a sequence number.
    async fn stream_top_seq(&self, object_id: u64) -> u64 {
        let key = self.tv.key_schema().layer_stream(object_id);
        let entries: Vec<(String, Vec<(String, Bytes)>)> = self
            .tv
            .client()
            .pool
            .xrevrange(&key, "+", "-", Some(1))
            .await
            .expect("XREVRANGE over the layer stream must succeed");
        let (id, _) = entries
            .first()
            .expect("the surviving stream must carry at least one entry");
        id.split('-')
            .next()
            .and_then(|ms| ms.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("unparseable stream entry id {id:?}"))
    }

    /// Terminal drain: close the tracker and wait, bounded, for every spawned
    /// re-base task to finish. `TaskTracker::close` is terminal, so this runs at
    /// most once per fixture — a case that needs the tracker to stay open uses
    /// [`Self::wait_idle`] instead.
    async fn drain(&self) {
        self.tracker.close();
        tokio::time::timeout(DRAIN_TIMEOUT, self.tracker.wait())
            .await
            .expect(
                "the bootstrap TaskTracker must drain inside DRAIN_TIMEOUT — a hang here points at \
                 a BaseBootstrap task that never completes, not at a test-only flake",
            );
    }

    /// Non-terminal drain, for cases that spawn again afterwards.
    async fn wait_idle(&self) {
        wait_for_writer_idle(&self.tracker, TRACKER_IDLE_DEADLINE_SECS).await;
    }
}

/// A NON-TRIVIAL creation state at `sequence_num`.
///
/// Deliberately not `ObjectState::default()`: an empty state would let a broken
/// path (one that emitted nothing at all) pass. Section `1` always exists so
/// [`RestoreRebaseFixture::commit`]'s field writes land somewhere real.
fn seed_state(seed: u8, sequence_num: u64) -> ObjectState {
    let mut state = ObjectState {
        sequence_num,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };

    let sec_a = 1u64;
    let sec_b = u64::from(seed) * 10 + 2;
    for (order, sid) in [sec_a, sec_b].into_iter().enumerate() {
        state.sections.insert(
            sid,
            Section {
                section_id: sid,
                parent_id: 0,
                name: format!("restore-rebase-{seed}-sec-{sid}"),
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
    state.children.insert(0, vec![sec_a, sec_b]);

    state
}

/// The four columns every landing assertion reads, in one HMGET so they cannot
/// disagree about which write they describe.
struct CursorColumns {
    seq: u64,
    last_checkpoint_seq: Option<String>,
    last_checkpoint_ms: Option<String>,
    restore_seq_bump: Option<String>,
}

async fn read_cursor_columns(fx: &RestoreRebaseFixture, object_id: u64) -> CursorColumns {
    let key = fx.tv.key_schema().obj_header(&object_id.to_string());
    let raw: Vec<Option<String>> = fx
        .tv
        .client()
        .pool
        .hmget(
            &key,
            vec![
                "seq",
                "last_checkpoint_seq",
                "last_checkpoint_ms",
                "restore_seq_bump",
            ],
        )
        .await
        .expect("HMGET obj:hdr cursor columns must succeed");
    CursorColumns {
        seq: raw
            .first()
            .and_then(|v| v.as_deref())
            .and_then(|v| v.parse::<u64>().ok())
            .expect("obj:hdr `seq` must be present and parseable after a restore"),
        last_checkpoint_seq: raw.get(1).cloned().flatten(),
        last_checkpoint_ms: raw.get(2).cloned().flatten(),
        restore_seq_bump: raw.get(3).cloned().flatten(),
    }
}

/// The shared landing assertion for a SUCCESSFUL re-base — Tests 1, 2 and 3 all
/// make exactly these claims, differing only in the lineage they restore from.
///
/// `restored_generation_seq` is the `snap_seq` of the generation the restore was
/// seeded from (the base's own sequence on a base-only lineage, the composed
/// head's `to_seq` on a composed one). It is asserted STRICTLY BELOW the seeded
/// cursor so no case can go vacuous — that inequality is the whole reason SR-3.6
/// exists, and the reason its old conditional no-gap arm is dead.
async fn assert_rebase_landed(
    fx: &RestoreRebaseFixture,
    object_id: u64,
    restored_generation_seq: u64,
    attempts_before: usize,
    writes_success_before: u64,
    context: &str,
) -> u64 {
    let columns = read_cursor_columns(fx, object_id).await;
    let seeded_seq = columns.seq;

    assert!(
        seeded_seq > restored_generation_seq,
        "{context}: every restore seeds `seq` strictly above the generation it restored (band >= 1 \
         unconditionally, post-b.8y5). seeded_seq = {seeded_seq}, restored generation = \
         {restored_generation_seq}. An equality here would make every claim below vacuous"
    );

    // ── Exactly ONE new base PUT, at the seeded cursor ───────────────────
    let base_puts = fx.store.base_puts_since(attempts_before);
    assert_eq!(
        base_puts.len(),
        1,
        "{context}: the restore must issue exactly ONE re-base PUT (one attempt, no retry, no \
         re-spawn); got {base_puts:?}"
    );
    assert_eq!(
        base_puts[0].object_id, object_id,
        "{context}: the re-base must be keyed on the restored object_id"
    );
    assert_eq!(
        base_puts[0].kind,
        CheckpointKind::Base,
        "{context}: SR-3.6 persists a fresh BASE generation, not a tail"
    );
    assert_eq!(
        base_puts[0].snap_seq, seeded_seq,
        "{context}: the re-base must land AT the restore's seeded writer cursor (read off \
         obj:hdr, never recomputed) — that is what closes the gap the restore opened"
    );

    // ── The same generation is visible in the listing ────────────────────
    let handles = fx.listing(object_id).await;
    assert!(
        handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "{context}: a Base generation MUST exist at seeded_seq = {seeded_seq}; got {handles:?}"
    );

    // ── All THREE cursor columns ─────────────────────────────────────────
    assert_eq!(
        columns.last_checkpoint_seq.as_deref(),
        Some(seeded_seq.to_string().as_str()),
        "{context}: the re-base must advance last_checkpoint_seq to the seeded cursor"
    );
    let stamped_ms = columns
        .last_checkpoint_ms
        .as_deref()
        .and_then(|ms| ms.parse::<u64>().ok())
        .unwrap_or_else(|| {
            panic!(
                "{context}: last_checkpoint_ms must be present and parseable; got {:?}",
                columns.last_checkpoint_ms
            )
        });
    assert!(
        stamped_ms > HAND_PUT_WRITTEN_AT_UNIX_MS,
        "{context}: the re-base stamps a FRESH last_checkpoint_ms (now_unix_ms), which must sit \
         above the restore's own stamp — the restored blob's recorded write time \
         ({HAND_PUT_WRITTEN_AT_UNIX_MS}). Got {stamped_ms}"
    );
    assert_eq!(
        columns
            .restore_seq_bump
            .as_deref()
            .and_then(|v| v.parse::<u64>().ok()),
        Some(LANDED_RETIRED_BUMP),
        "{context}: the re-base must RETIRE restore_seq_bump to {LANDED_RETIRED_BUMP} — that is \
         what unblocks tail capture, which refuses PostRestoreUnrebased before the stream read \
         while the column is non-zero. Got {:?}",
        columns.restore_seq_bump
    );

    // ── The operator-visible signal: the THIRD producer of this cell ─────
    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before + 1,
        "{context}: a completed re-base must tick \
         checkpoint_writes_total{{result=\"success\",kind=\"layer\"}} exactly once (the \
         restore-time producer, alongside the periodic loop's and the creation-time bootstrap's)"
    );

    seeded_seq
}

// =============================================================================
// Case 1 — a BASE-ONLY restore re-bases at seeded_seq with all three columns.
// =============================================================================

/// The Epic's acceptance behaviour, and the proof that SR-3.6's conditional
/// no-gap arm is dead: a **base-only** restore — the shape the old spec text
/// said would need no re-base — leaves a `Base` at `seeded_seq` and stamps all
/// three cursor columns.
///
/// Also carries the file's vocabulary fence (item 10), and it is now a LIVE
/// separation rather than a reservation: SR-3.6's post-restore re-base is a
/// WRITE, not a capture, so it must move the existing
/// `checkpoint_writes_total{result, kind="layer"}` family and move NO
/// `live_edit_checkpoint_tail_total{outcome}` cell — **including `rebased`**,
/// which the SR-4.1 chain-collapse re-base
/// (`src/checkpoint/rebase.rs`) now mints and which this producer must never
/// touch. Before that landed, the assertion below was true of a permanently-zero
/// cell and could not fail for the right reason; it can now, and this file is
/// the natural home for it. This Task introduced no new metric name, label, or
/// family; Epic 5's Task 5 (`t2.dfh.be.s8`) landed the SR-7 observability
/// adjudication (2026-09-01), which added the separate
/// `live_edit_checkpoint_base_bootstrap_total` family — asserted in
/// `bootstrap_observability_test.rs`, and orthogonal to the fence above.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn base_only_restore_rebases_at_seeded_seq_and_retires_the_bump() {
    let fx = restore_rebase_fixture(true).await;
    let layer_name = "restore-rebase-base-only-layer";
    let object_id = fx.create_layer_with_base(layer_name, 1).await;

    wipe_layer_state(&fx.tv, object_id).await;

    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");
    let writes_error_before = fx.layer_writes("error");
    let tail_cells_before = gather_tail_outcome_cells(&fx.metrics);

    let outcome = fx.hook.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "a planted base-only lineage must restore; got {outcome:?}"
    );

    // The re-base is a best-effort tracker task the hook never awaits; draining
    // the tracker is what makes it deterministically observable.
    fx.drain().await;

    assert_rebase_landed(
        &fx,
        object_id,
        SEED_SEQ,
        attempts_before,
        writes_success_before,
        "base-only restore",
    )
    .await;

    assert_eq!(
        fx.layer_writes("error"),
        writes_error_before,
        "a successful re-base must not tick the error arm"
    );

    // ── Vocabulary fence ─────────────────────────────────────────────────
    let tail_cells_after = gather_tail_outcome_cells(&fx.metrics);
    assert_eq!(
        tail_cells_after, tail_cells_before,
        "SR-3.6's post-restore re-base is a WRITE, not a capture: it must move NO \
         live_edit_checkpoint_tail_total{{outcome}} cell — not `rebased`, which belongs to the \
         SR-4.1 chain-collapse re-base and would read as N chain collapses after a DR restore of \
         N layers; not any other existing cell; and not a new out-of-band key. Cells before: \
         {tail_cells_before:?}"
    );
}

// =============================================================================
// Case 2 — the SURVIVING-STREAM shape: S0 lands above every fossil.
// =============================================================================

/// A partial state loss (the materialized `obj:*` hashes gone, the per-object
/// stream intact) is the shape the post-restore fossil boundary is actually
/// about: the surviving stream still holds the abandoned timeline's entries.
///
/// Same landing claims as Case 1, plus the one that only this shape can make —
/// the re-base's `snap_seq` sits STRICTLY ABOVE the surviving stream's
/// last-generated id, so the capture window `{seeded_seq+1}-0` opens above every
/// fossil. That is the whole point of SR-3.6 on this shape, and Case 6 spends it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn surviving_stream_restore_rebases_above_every_fossil() {
    let fx = restore_rebase_fixture(true).await;
    let layer_name = "restore-rebase-surviving-stream-layer";
    let object_id = fx.create_layer_with_base(layer_name, 2).await;

    // Real pre-loss commits, so the surviving stream carries genuine
    // abandoned-timeline entries above the planted base rather than only the
    // creation seed.
    for i in 0..3u64 {
        fx.commit(layer_name, object_id, 5_000 + i).await;
    }
    let fossil_top = fx.stream_top_seq(object_id).await;
    assert!(
        fossil_top > SEED_SEQ,
        "the pre-loss commits must have advanced the stream above the planted base \
         ({fossil_top} vs {SEED_SEQ}); otherwise this shape is indistinguishable from Case 1's"
    );

    wipe_layer_state_preserving_stream(&fx.tv, object_id).await;

    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");

    let outcome = fx.hook.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "a planted base-only lineage with a surviving stream must restore; got {outcome:?}"
    );
    fx.drain().await;

    let seeded_seq = assert_rebase_landed(
        &fx,
        object_id,
        SEED_SEQ,
        attempts_before,
        writes_success_before,
        "surviving-stream restore",
    )
    .await;

    assert!(
        seeded_seq > fossil_top,
        "the re-base's Base must sit STRICTLY ABOVE the surviving stream's last-generated id \
         ({seeded_seq} vs {fossil_top}). S0 above every fossil is what makes the next capture's \
         read window `{{seeded_seq+1}}-0` start above the abandoned timeline — the whole point of \
         SR-3.6 on this shape"
    );

    // Non-degeneracy: with a band of `checkpoint_seq_safety_margin` the seeded
    // cursor would be `SEED_SEQ + margin` if the high-water probe had folded to
    // `None`. A strictly larger value proves the SURVIVING STREAM raised the
    // restore floor, i.e. that this case is really about the shape it names and
    // not a second copy of Case 1.
    let margin = fx.config.checkpoint_seq_safety_margin;
    assert!(
        seeded_seq > SEED_SEQ + margin,
        "the surviving stream must have raised the restore floor above the base's own snap_seq: \
         seeded_seq = {seeded_seq} must exceed SEED_SEQ + margin = {}. An equality would mean the \
         high-water probe folded to None and this case degenerated into Case 1",
        SEED_SEQ + margin
    );
}

// =============================================================================
// Case 3 — a COMPOSED restore re-bases identically.
// =============================================================================

/// The composed path fires the same re-base at the composed head's seeded
/// cursor.
///
/// Built on `helpers::drive_captured_tail` — the sanctioned REAL captured-tail
/// producer. A hand-put `put_tail_generation` tail carries SYNTHETIC records
/// (reset OMPFs that wipe state on replay) and must not be the only composed
/// evidence; that is the b.gze / b.zju lesson. Task 6 (`t2.dfh.yj.u3`) owns
/// real-traffic golden replay — this case only needs the re-base to FIRE on the
/// composed path and land the same shape.
///
/// Compiled only under `test-checkpoint-fault-injection`: the fixture reaches
/// the loop's tail capture through the unprovable-cut seam. Without the
/// feature it is compiled out, not skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn composed_restore_rebases_at_the_composed_seeded_seq() {
    let fx = restore_rebase_fixture(true).await;
    let layer_name = "restore-rebase-composed-layer";

    // A real base + a real captured tail, produced by the real periodic loop.
    // The fixture cancels and drains its own loop before returning, so nothing
    // but the re-base can write afterwards.
    let tail = drive_captured_tail(
        &fx.tv,
        &fx.store_dyn,
        &fx.metrics,
        &fx.session_id,
        layer_name,
        fx.tail_config(),
    )
    .await;
    assert_eq!(
        tail.kind,
        CheckpointKind::Tail,
        "drive_captured_tail must return a Tail handle; got {tail:?}"
    );
    let object_id = tail.object_id;
    // A tail's handle sequence IS its envelope `to_seq` — the composed head.
    let composed_head_seq = tail.snap_seq;

    wipe_layer_state(&fx.tv, object_id).await;

    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");

    let outcome = fx.hook.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "a real base+tail lineage must restore; got {outcome:?}"
    );
    fx.drain().await;

    assert_rebase_landed(
        &fx,
        object_id,
        composed_head_seq,
        attempts_before,
        writes_success_before,
        "composed restore",
    )
    .await;
}

// =============================================================================
// Case 4 — restore-site cursor-regression guard, three probes.
// =============================================================================

/// **4a.** A non-zero `last_checkpoint_seq` at entry — true of EVERY restore,
/// because `write_object` stamps `last_checkpoint_seq = snap_seq` inside the
/// restore's own MULTI/EXEC — must NOT abort the re-base.
///
/// This is the assertion that catches a verbatim copy of the creation-time guard
/// (a). That guard aborted on any non-zero `last_checkpoint_seq`; at a restore
/// call site the column is non-zero every single time, so the un-generalized
/// form would kill every re-base and leave every restored hot layer permanently
/// without DR coverage. The landed guard is `observed >= target.snap_seq()`,
/// which admits the restore's own audit stamp because that stamp sits a whole
/// band BELOW `seeded_seq`.
///
/// The gate is what makes the pre-re-base header observable: the re-base PUT
/// parks, so the header can be read at a moment when the bootstrap is provably
/// mid-flight and the restore's own stamp is provably still the one in place.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_site_guard_admits_the_restores_own_nonzero_cursor() {
    let fx = restore_rebase_fixture(true).await;
    let layer_name = "restore-rebase-guard-a-layer";
    let object_id = fx.create_layer_with_base(layer_name, 3).await;

    wipe_layer_state(&fx.tv, object_id).await;

    // Armed only now, after the lineage root is planted — see
    // `RecordingRebaseStore`'s targeting note.
    let release = fx.store.arm_gate();
    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));

    // The gated PUT proves the bootstrap got PAST guard (a): the guard sits at
    // step 1, strictly before the PUT, so a PUT that reached the wrapper is a
    // guard that admitted.
    let parked = poll_until(
        || {
            let store = Arc::clone(&fx.store);
            async move { !store.base_puts_since(attempts_before).is_empty() }
        },
        GATE_WAIT,
    )
    .await;
    assert!(
        parked,
        "the re-base PUT must reach the store within {GATE_WAIT:?}. It did not, which is exactly \
         what a verbatim copy of the creation-time `last_checkpoint_seq != 0` guard would produce: \
         a restore site whose cursor is non-zero on every single call, so every re-base aborts at \
         step 1"
    );

    // …and while it is parked, the header still carries the RESTORE's own stamp.
    let mid_flight = read_cursor_columns(&fx, object_id).await;
    let restored_from = mid_flight
        .last_checkpoint_seq
        .as_deref()
        .and_then(|v| v.parse::<u64>().ok())
        .expect("the restore's own last_checkpoint_seq stamp must be present mid-flight");
    assert_eq!(
        restored_from, SEED_SEQ,
        "the restore stamps last_checkpoint_seq = the restored generation's snap_seq, inside its \
         own MULTI/EXEC and before the async re-base runs"
    );
    assert_ne!(
        restored_from, 0,
        "the whole point of this case: the value guard (a) sees at a restore call site is NON-ZERO"
    );
    assert!(
        restored_from < mid_flight.seq,
        "the restore's audit stamp sits a whole band BELOW the seeded cursor ({restored_from} < \
         {}), which is why the target-relative guard admits it",
        mid_flight.seq
    );

    let _ = release.send(());
    fx.drain().await;

    assert_rebase_landed(
        &fx,
        object_id,
        SEED_SEQ,
        attempts_before,
        writes_success_before,
        "guard (a) admits the restore's own stamp",
    )
    .await;
}

/// **4b.** A HIGHER `last_checkpoint_seq` landing mid-flight (a periodic
/// generation) must not be stamped BACKWARDS by the in-flight re-base.
///
/// Landed posture, pinned precisely: guard (b) re-reads both cursor columns in
/// ONE `HMGET` immediately before the stamp and returns
/// `BootstrapOutcome::UnstampedGenerationPresent` when the observed
/// `last_checkpoint_seq` is at or above the target. The PUT has already
/// completed by then, so the blob IS present at `seeded_seq` while EVERY cursor
/// column is left untouched — including `restore_seq_bump`, which therefore
/// still carries the restore's band and keeps tail capture refusing observably.
/// Epic 5's SR-7 adjudication gave the arm its own cell,
/// `live_edit_checkpoint_base_bootstrap_total{outcome="unstamped_generation_present",
/// site="restore_rebase"}`, and deliberately left it moving no
/// `checkpoint_writes_total` cell at all — which is what the two assertions
/// below still pin, unedited.
///
/// The case then spends that last clause, which doubles as the file's cover for
/// the PUT-ok / cursor-stamp-absent residual (PM condition C10 — see the module
/// header for why the literal HSET-failure arm is a documented posture rather
/// than a test).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_generation_landing_mid_flight_blocks_a_backwards_cursor_stamp() {
    let fx = restore_rebase_fixture(true).await;
    let layer_name = "restore-rebase-guard-b-layer";
    let object_id = fx.create_layer_with_base(layer_name, 4).await;

    wipe_layer_state(&fx.tv, object_id).await;

    let release = fx.store.arm_gate();
    let attempts_before = fx.store.attempts().len();
    // The planted lineage root was itself a delegated `Base` PUT, so the claim
    // below is a DELTA, never an absolute count.
    let delegated_before = fx.store.base_puts_delegated();
    let writes_success_before = fx.layer_writes("success");
    let writes_error_before = fx.layer_writes("error");

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));

    let parked = poll_until(
        || {
            let store = Arc::clone(&fx.store);
            async move { !store.base_puts_since(attempts_before).is_empty() }
        },
        GATE_WAIT,
    )
    .await;
    assert!(
        parked,
        "the re-base PUT must reach the store within {GATE_WAIT:?}"
    );

    // A periodic generation lands while the PUT is parked, stamping a cursor
    // ABOVE the re-base's target.
    let seeded_seq = fx.seeded_seq(object_id).await;
    let interloper_seq = seeded_seq + 1;
    let restore_bump = fx.header_u64(object_id, "restore_seq_bump").await;
    assert_ne!(
        restore_bump, 0,
        "the restore's own band must still be outstanding before the interloper lands"
    );
    fx.set_header_field(
        object_id,
        "last_checkpoint_seq",
        &interloper_seq.to_string(),
    )
    .await;

    let _ = release.send(());
    fx.drain().await;

    // ── The landed posture: blob present, EVERY cursor column untouched ──
    assert_eq!(
        fx.store.base_puts_delegated(),
        delegated_before + 1,
        "the re-base PUT completed before guard (b)'s pre-stamp re-read — the blob is durable even \
         though the stamp was declined"
    );
    let handles = fx.listing(object_id).await;
    assert!(
        handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "landed posture is BLOB PRESENT + CURSOR UNTOUCHED: the Base at seeded_seq = {seeded_seq} \
         must exist, because guard (b) fires AFTER the PUT. Got {handles:?}"
    );

    let columns = read_cursor_columns(&fx, object_id).await;
    assert_eq!(
        columns.last_checkpoint_seq.as_deref(),
        Some(interloper_seq.to_string().as_str()),
        "landed posture: guard (b) must leave last_checkpoint_seq at the HIGHER value the \
         mid-flight generation stamped ({interloper_seq}), never regress it back down to the \
         re-base's own target ({seeded_seq}). A regression there would un-suppress the loop's \
         skip-if-no-commits gate and inflate the next assembly's observed_commit_span"
    );
    assert_eq!(
        columns
            .restore_seq_bump
            .as_deref()
            .and_then(|v| v.parse::<u64>().ok()),
        Some(restore_bump),
        "landed posture: the skip leaves EVERY cursor column untouched, so restore_seq_bump still \
         carries the restore's own band ({restore_bump}) and tail capture keeps refusing \
         observably. Got {:?}",
        columns.restore_seq_bump
    );

    // ── The arm's cell is on the bootstrap family, never on writes_total ──
    //
    // Epic 5's SR-7 adjudication (Task 5 `t2.dfh.be.s8`) landed this arm's
    // signal as
    // `live_edit_checkpoint_base_bootstrap_total{outcome="unstamped_generation_present",
    // site="restore_rebase"}`, so it is no longer a silent arm. It still moves
    // NO `checkpoint_writes_total` cell, deliberately: the blob is durable but
    // the GENERATION did not complete — no cursor stamp landed — and that family
    // counts generation-completion outcomes. The periodic loop takes the
    // identical posture on the identical shape (`checkpointer.rs`,
    // `rebase_on_chain_full`'s pre-stamp regression guard: every cursor column
    // untouched, no writes_total cell, prune skipped). The two assertions below
    // are UNEDITED across that adjudication, and that is the point.
    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before,
        "UnstampedGenerationPresent must not read as a success on \
         checkpoint_writes_total: the blob is durable but no generation completed, and the arm's \
         own signal lives on live_edit_checkpoint_base_bootstrap_total instead"
    );
    assert_eq!(
        fx.layer_writes("error"),
        writes_error_before,
        "a guard skip is not an error either"
    );

    // ── The `refused_seq_gap` residual, spent (PM condition C10) ─────────
    //
    // "A re-base whose blob landed but whose cursor stamp did not" is a real
    // residual of this feature, and this state IS it: the Base is durable at
    // `seeded_seq`, yet `restore_seq_bump` was never retired. The literal
    // PUT-ok / HSET-fails arm is not reachable without new production fault
    // machinery — the HSET is an unconditional `hset` on the data pool with no
    // seam, and any stimulus crude enough to fail it (a WRONGTYPE on
    // `obj:hdr`) fails the pre-stamp HMGET one statement earlier — but its
    // OBSERVABLE consequence is identical to guard (b)'s skip, and that is what
    // is pinned here: the layer keeps refusing until its next eligible
    // periodic checkpoint tick lands a base, or it is re-restored.
    //
    // Since Epic 5's SR-7 adjudication the residual also has a METRIC selector
    // of its own — `live_edit_checkpoint_base_bootstrap_total{outcome=~"unstamped_.*"}`,
    // the grouping that means "the blob is durable but its cursor is not" — so
    // this prose is no longer the operator's only handle on it: a rising
    // `refused_seq_gap` can now be attributed to (or cleared of) an unstamped
    // re-base without reading logs.
    let post = fx.land_post_restore_commits(layer_name, object_id).await;
    assert_eq!(post.first().copied(), Some(seeded_seq + 1));
    let decision = fx.capture(object_id).await;
    assert!(
        matches!(
            decision,
            TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased { .. })
        ),
        "the blob-landed / stamp-declined residual must keep tail capture refusing: the guard keys \
         on the un-retired restore_seq_bump ({restore_bump}) BEFORE the stream read, so a durable \
         Base at seeded_seq does not on its own restore coverage. Got {decision:?}"
    );
    assert_eq!(
        decision.outcome_label(),
        "refused_seq_gap",
        "the residual surfaces to an operator as refused_seq_gap — the same cell a never-re-based \
         restore produces, with the same remediation"
    );
}

/// **4c.** Untrusted cursor columns — a `restore_seq_bump` that is PRESENT but
/// unparseable — make the re-base decline to stamp at all.
///
/// The conservative direction, and the landed one: `run_once`'s step-1 arm
/// returns `BootstrapOutcome::SkippedUntrustedCursor` on `RestoreRebase` when
/// `identity.cursor_columns_trusted == false`, before the OMPF build and before
/// the PUT. Nothing is lost by skipping — `tail_capture` independently refuses
/// on `!cursor_columns_trusted` (`tail_capture.rs`, step 3), so the layer's
/// capture is refusing either way, and a blind CLEARING of an unreadable bump is
/// exactly the write that would unblock capture over fossils it cannot account
/// for.
///
/// Driven through `run_once` **directly** rather than through the hook's
/// fire-and-forget spawn: the guard reads the header at step 1, which a spawned
/// task reaches before a test could corrupt the column. The fixture is therefore
/// built with `wire_rebase = false` (the documented `None` posture — no re-base
/// is spawned) and the target is assembled from values READ off the header, not
/// recomputed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_untrusted_cursor_column_declines_the_rebase_stamp() {
    let fx = restore_rebase_fixture(false).await;
    let layer_name = "restore-rebase-guard-c-layer";
    let object_id = fx.create_layer_with_base(layer_name, 5).await;

    wipe_layer_state(&fx.tv, object_id).await;

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));
    // No re-base was wired, so nothing was spawned; the header carries exactly
    // what `write_object` wrote.
    fx.wait_idle().await;

    let seeded_seq = fx.seeded_seq(object_id).await;
    let absorbed_band = fx.header_u64(object_id, "restore_seq_bump").await;
    assert_ne!(
        absorbed_band, 0,
        "the restore must have left a non-zero band for this case to be about anything"
    );

    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");
    let writes_error_before = fx.layer_writes("error");

    // Another writer puts garbage in the header. `read_header_identity` folds an
    // unparseable column to 0 and reports `cursor_columns_trusted = false`.
    fx.set_header_field(object_id, "restore_seq_bump", "not-a-number")
        .await;

    let outcome = fx
        .bootstrap
        .run_once(
            object_id,
            seed_state(5, seeded_seq),
            BootstrapTarget::RestoreRebase {
                seeded_seq,
                // A hand-put base carries the no-measurement sentinel; the value
                // is irrelevant to this arm, which returns before the meta is
                // built.
                observed_commit_span: 0,
                absorbed_band,
            },
        )
        .await;

    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedUntrustedCursor,
        "landed posture: an unparseable cursor column makes the post-restore re-base decline \
         ENTIRELY — no PUT, no stamp, no checkpoint_writes_total cell. (Since the SR-7 \
         adjudication, 2026-09-01, the arm DOES tick the bootstrap-outcome family at \
         {{skipped_untrusted_cursor, restore_rebase}}; that is an invocation-outcome record, not \
         a generation-completion one, and it is asserted in bootstrap_observability_test.rs.) \
         Declining is conservative and costs nothing, because tail_capture independently refuses \
         on !cursor_columns_trusted"
    );

    assert!(
        fx.store.base_puts_since(attempts_before).is_empty(),
        "the untrusted-cursor arm returns at step 1, before the OMPF build and before the PUT"
    );
    assert_eq!(
        fx.header_field(object_id, "restore_seq_bump").await.as_deref(),
        Some("not-a-number"),
        "the declined re-base must leave the untrustworthy column exactly as it found it — a blind \
         CLEARING of an unreadable bump is precisely the write this guard exists to prevent"
    );
    assert_eq!(
        fx.header_u64(object_id, "last_checkpoint_seq").await,
        SEED_SEQ,
        "the declined re-base must leave last_checkpoint_seq at the restore's own audit stamp"
    );
    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before,
        "SkippedUntrustedCursor completes no generation, so it deliberately moves no \
         checkpoint_writes_total cell — the bootstrap-outcome cell it does tick is a different \
         family with a different meaning"
    );
    assert_eq!(
        fx.layer_writes("error"),
        writes_error_before,
        "a guard skip is not an error"
    );
}

// =============================================================================
// Case 5 — a failed re-base PUT stamps NOTHING, leaving the restore's columns
//          UNCHANGED (not absent).
// =============================================================================

/// SR-7.2 stamps-nothing, at the RESTORE site — where the correct claim is
/// subtly different from the creation site's.
///
/// At the creation site a failed PUT leaves the cursor columns ABSENT, because
/// nothing ever wrote them. At the restore site the restore ALREADY stamped all
/// three inside its own MULTI/EXEC, so the correct claim is that they are
/// **unchanged**: `last_checkpoint_seq` still the restored generation's
/// `snap_seq`, `last_checkpoint_ms` still the restored blob's own recorded write
/// time, and `restore_seq_bump` still the band — non-zero, which is exactly what
/// keeps `tail_capture` refusing observably (Case 6's failure arm spends that).
/// Asserting merely "absent" here would be the mis-assertion this case exists to
/// foreclose.
///
/// Runs for both classifications: the contract is classification-independent,
/// and a future "retry the transient case" change must break this rather than
/// pass it.
async fn assert_failed_rebase_leaves_the_restores_columns_unchanged(
    mode: FailureMode,
    layer_name: &str,
    seed: u8,
) {
    let fx = restore_rebase_fixture(true).await;
    let object_id = fx.create_layer_with_base(layer_name, seed).await;

    wipe_layer_state(&fx.tv, object_id).await;

    // Armed after the lineage root is planted, so the only Base PUT it can
    // catch is the re-base.
    fx.store.fail_base_puts(mode);
    let attempts_before = fx.store.attempts().len();
    let writes_success_before = fx.layer_writes("success");
    let writes_error_before = fx.layer_writes("error");

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));
    fx.drain().await;

    // ── Exactly ONE attempt, no retry ────────────────────────────────────
    let base_puts = fx.store.base_puts_since(attempts_before);
    assert_eq!(
        base_puts.len(),
        1,
        "{mode:?}: the re-base makes exactly ONE best-effort put_layer_checkpoint attempt. A count \
         above 1 means a retry loop appeared — a design regression: the task is joined only by \
         graceful shutdown, so retries put unbounded adapter traffic inside the shutdown budget"
    );

    // ── No generation at the seeded cursor ───────────────────────────────
    let seeded_seq = fx.seeded_seq(object_id).await;
    let handles = fx.listing(object_id).await;
    assert!(
        !handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "{mode:?}: a failed PUT must leave NO generation at seeded_seq = {seeded_seq}; got \
         {handles:?}"
    );
    assert!(
        handles
            .iter()
            .any(|h| h.snap_seq == SEED_SEQ && h.kind == CheckpointKind::Base),
        "{mode:?}: the planted lineage root must survive a failed re-base; got {handles:?}"
    );

    // ── The restore's own columns, UNCHANGED — not absent ────────────────
    let columns = read_cursor_columns(&fx, object_id).await;
    assert_eq!(
        columns.last_checkpoint_seq.as_deref(),
        Some(SEED_SEQ.to_string().as_str()),
        "{mode:?}: last_checkpoint_seq must still carry the RESTORE's own audit stamp \
         ({SEED_SEQ}). The restore stamped it before the async re-base ran, so 'absent' is the \
         WRONG claim at this call site"
    );
    assert_eq!(
        columns.last_checkpoint_ms.as_deref(),
        Some(HAND_PUT_WRITTEN_AT_UNIX_MS.to_string().as_str()),
        "{mode:?}: last_checkpoint_ms must be UNMOVED — the restored blob's own recorded write \
         time, which write_object stamped. A fresh now_unix_ms() here would mean the failure arm \
         stamped anyway"
    );
    let bump = columns
        .restore_seq_bump
        .as_deref()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| {
            panic!(
                "{mode:?}: restore_seq_bump must be PRESENT and parseable; got {:?}",
                columns.restore_seq_bump
            )
        });
    assert_eq!(
        bump, fx.config.checkpoint_seq_safety_margin,
        "{mode:?}: restore_seq_bump must still carry the restore's own band. A hand-put base \
         records observed_commit_span = 0, so band = max(margin, 0) = the configured margin. It \
         must NOT be {LANDED_RETIRED_BUMP} — retiring it on a failed re-base would unblock tail \
         capture over a gap no generation covers"
    );
    assert_ne!(
        bump, LANDED_RETIRED_BUMP,
        "{mode:?}: the bump must remain OUTSTANDING so capture refuses observably"
    );

    // ── The operator-visible signal ──────────────────────────────────────
    assert_eq!(
        fx.layer_writes("error"),
        writes_error_before + 1,
        "{mode:?}: a failed re-base PUT must tick \
         checkpoint_writes_total{{result=\"error\",kind=\"layer\"}} exactly once — it is what \
         makes a fleet-wide re-base failure visible immediately rather than only later, per-layer, \
         as a refused capture"
    );
    assert_eq!(
        fx.layer_writes("success"),
        writes_success_before,
        "{mode:?}: a failed re-base PUT must not tick the success arm"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transient_rebase_put_failure_leaves_the_restores_columns_unchanged() {
    assert_failed_rebase_leaves_the_restores_columns_unchanged(
        FailureMode::Transient,
        "restore-rebase-put-failure-transient-layer",
        6,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn permanent_rebase_put_failure_leaves_the_restores_columns_unchanged() {
    assert_failed_rebase_leaves_the_restores_columns_unchanged(
        FailureMode::Permanent,
        "restore-rebase-put-failure-permanent-layer",
        7,
    )
    .await;
}

// =============================================================================
// Case 6 — the fossil-free property, at harness level.
// =============================================================================

/// After a successful re-base, a hot layer's very next tail capture SUCCEEDS and
/// roots at `seeded_seq` — the Epic's acceptance criterion, "a restored
/// never-quiescent layer's next tail capture succeeds rather than refusing".
///
/// Both halves of the fossil-free property are load-bearing and both are pinned:
/// the Base at `seeded_seq` moves S0 above every fossil, and the retired
/// `restore_seq_bump` is what gets the capture past the post-restore guard at
/// all (that guard runs BEFORE the stream read, so the S0 move alone would not
/// suffice).
///
/// Driven over BOTH stream shapes, because they fail differently if the re-base
/// is wrong: a wiped stream would refuse `trimmed`, a surviving one would carry
/// fossils into the envelope.
async fn assert_post_rebase_capture_succeeds(preserve_stream: bool, layer_name: &str, seed: u8) {
    let shape = if preserve_stream {
        "surviving-stream"
    } else {
        "wiped-stream"
    };

    let fx = restore_rebase_fixture(true).await;
    let object_id = fx.create_layer_with_base(layer_name, seed).await;

    for i in 0..3u64 {
        fx.commit(layer_name, object_id, 6_000 + i).await;
    }

    if preserve_stream {
        wipe_layer_state_preserving_stream(&fx.tv, object_id).await;
    } else {
        wipe_layer_state(&fx.tv, object_id).await;
    }

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));
    fx.drain().await;

    let seeded_seq = fx.seeded_seq(object_id).await;
    assert_eq!(
        fx.header_u64(object_id, "restore_seq_bump").await,
        LANDED_RETIRED_BUMP,
        "{shape}: the re-base must have retired the bump before the capture can get past the \
         post-restore guard"
    );

    // A hot layer: real post-restore commits, each carrying a real relay_ompf.
    let post = fx.land_post_restore_commits(layer_name, object_id).await;
    assert_eq!(
        post.first().copied(),
        Some(seeded_seq + 1),
        "{shape}: the first post-restore commit must land at seeded_seq + 1 — that is the floor of \
         the capture window `{{seeded_seq+1}}-0`"
    );

    let decision = fx.capture(object_id).await;
    let TailCaptureDecision::Captured {
        to_seq,
        record_count,
        ..
    } = decision
    else {
        panic!(
            "{shape}: a re-based layer's next capture must SUCCEED — that is the Epic's acceptance \
             criterion. Got {decision:?} (outcome label {:?})",
            decision.outcome_label()
        );
    };
    assert_eq!(
        record_count as u64, POST_RESTORE_COMMITS,
        "{shape}: the capture must carry exactly the post-restore commits, and no fossil"
    );
    assert_eq!(
        to_seq,
        *post.last().expect("post-restore commits must be non-empty"),
        "{shape}: the capture must end at the newest post-restore commit"
    );

    // ── The persisted envelope roots at the re-base ──────────────────────
    let handles = fx.listing(object_id).await;
    let tail = handles
        .iter()
        .find(|h| h.kind == CheckpointKind::Tail)
        .unwrap_or_else(|| {
            panic!("{shape}: a Captured decision must persist a Tail; got {handles:?}")
        });
    let blob = fx
        .store_dyn
        .get_layer_checkpoint(tail)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the capture's handle must resolve to a stored artifact");
    let envelope = decode_tail_envelope(&blob.bytes).expect("the persisted envelope must decode");
    assert_eq!(
        envelope.header().from_seq,
        seeded_seq,
        "{shape}: the capture window must open at the RE-BASE's own sequence — `from_seq == \
         seeded_seq` is the fossil-free property, because every abandoned-timeline entry sits \
         strictly below it"
    );
    assert_eq!(
        envelope.records().len() as u64,
        envelope.header().to_seq - envelope.header().from_seq,
        "{shape}: claimed range must EQUAL carried content (b.pw6)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_rebase_capture_succeeds_on_a_wiped_stream() {
    assert_post_rebase_capture_succeeds(false, "restore-rebase-fossilfree-wiped-layer", 8).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_rebase_capture_succeeds_on_a_surviving_stream() {
    assert_post_rebase_capture_succeeds(true, "restore-rebase-fossilfree-surviving-layer", 9).await;
}

/// The counterpart: on a FAILED re-base the same capture call refuses — with
/// `TailCaptureRefusal::PostRestoreUnrebased` and outcome label
/// `refused_seq_gap`, on **BOTH** stream shapes.
///
/// The shape-independence is the pinned correction (PM condition C9, and the
/// spec correction now recorded on the spec child): the post-restore guard keys
/// on the restore's own cursor stamp and runs BEFORE the stream read, so
/// `TailRefusal::Trimmed` is UNREACHABLE while `restore_seq_bump != 0`. The
/// earlier "trimmed on a fresh stream / seq_gap on a surviving one" phrasing is
/// struck.
///
/// The `outcome_label()` is asserted alongside the enum variant so the
/// operator-facing signal is pinned too — the guard deliberately folds onto
/// `SeqGap`'s existing label rather than minting a tenth one, and a future edit
/// that forked the dashboard vocabulary would break here.
async fn assert_failed_rebase_capture_refuses_seq_gap(
    preserve_stream: bool,
    layer_name: &str,
    seed: u8,
) {
    let shape = if preserve_stream {
        "surviving-stream"
    } else {
        "wiped-stream"
    };

    let fx = restore_rebase_fixture(true).await;
    let object_id = fx.create_layer_with_base(layer_name, seed).await;

    for i in 0..3u64 {
        fx.commit(layer_name, object_id, 7_000 + i).await;
    }

    if preserve_stream {
        wipe_layer_state_preserving_stream(&fx.tv, object_id).await;
    } else {
        wipe_layer_state(&fx.tv, object_id).await;
    }

    fx.store.fail_base_puts(FailureMode::Transient);

    assert!(matches!(
        fx.hook.try_restore_layer(object_id).await,
        RecoveryOutcome::Restored
    ));
    fx.drain().await;

    let seeded_seq = fx.seeded_seq(object_id).await;
    let bump = fx.header_u64(object_id, "restore_seq_bump").await;
    assert_ne!(
        bump, LANDED_RETIRED_BUMP,
        "{shape}: the failed re-base must leave the bump outstanding — that is the input the guard \
         keys on"
    );

    // Land real commits so the refusal cannot be confused with "nothing to
    // capture": the guard must fire even though a sound prefix exists.
    let post = fx.land_post_restore_commits(layer_name, object_id).await;
    assert_eq!(post.first().copied(), Some(seeded_seq + 1));

    let decision = fx.capture(object_id).await;
    assert!(
        matches!(
            decision,
            TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased { .. })
        ),
        "{shape}: a failed re-base leaves restore_seq_bump = {bump}, and the post-restore fossil \
         guard runs BEFORE the stream read — so the refusal is PostRestoreUnrebased on BOTH stream \
         shapes and TailRefusal::Trimmed is unreachable here. Got {decision:?}"
    );
    let TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased { restore_seq_bump }) =
        decision
    else {
        unreachable!("guarded by the assert above");
    };
    assert_eq!(
        restore_seq_bump, bump,
        "{shape}: the refusal must report the bump it actually read off obj:hdr"
    );
    assert_eq!(
        decision.outcome_label(),
        "refused_seq_gap",
        "{shape}: PostRestoreUnrebased deliberately folds onto SeqGap's existing operator-facing \
         label rather than minting a tenth one — the condition and its remediation are the same, \
         caught one read earlier. A new label here would fork the dashboard vocabulary Epic 3 \
         pinned"
    );

    // Nothing was persisted, and the layer's lineage still ends at its root.
    let handles = fx.listing(object_id).await;
    assert!(
        handles.iter().all(|h| h.kind == CheckpointKind::Base),
        "{shape}: a refusal persists nothing; got {handles:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_rebase_capture_refuses_seq_gap_on_a_wiped_stream() {
    assert_failed_rebase_capture_refuses_seq_gap(false, "restore-rebase-refusal-wiped-layer", 10)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_rebase_capture_refuses_seq_gap_on_a_surviving_stream() {
    assert_failed_rebase_capture_refuses_seq_gap(
        true,
        "restore-rebase-refusal-surviving-layer",
        11,
    )
    .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounded polling
// ─────────────────────────────────────────────────────────────────────────────

/// Poll `predicate` until it is true or `budget` elapses; returns whether it
/// became true.
///
/// The one place this file polls at all, and it is a STATE precondition (a PUT
/// arrived at the wrapper), never a timing hack: the caller asserts on the
/// returned bool with a diagnostic naming what a `false` means. The budget is
/// always a named constant.
async fn poll_until<F, Fut>(mut predicate: F, budget: Duration) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        if predicate().await {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
