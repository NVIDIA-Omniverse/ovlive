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

//! Shared scaffolding for the Task `t2.fgj.ya.wg` end-to-end recovery
//! integration tests under `tests/integration/checkpoint/e2e_*_test.rs`.
//!
//! Subtask `t3.fgj.ya.wg.9n` lands the LocalDisk-backed scaffolding. The
//! seven sibling Subtasks (`.rt`, `.cm`, `.nx`, `.w4`, `.4q`, `.rh`, `.r1`)
//! populate the per-AC test files and `use super::helpers::*;` to pull in
//! the shared harness.
//!
//! ## LocalDisk-backed harness
//!
//! The original contract is small and focused:
//!
//! - [`setup_checkpoint_harness`] bundles a `TestValkey`, a tempdir, a
//!   `LocalDiskCheckpointStore` wrapped behind `Arc<dyn CheckpointStore>`,
//!   a `LiveEditService` wired with a real (not disabled) `RecoveryHook`,
//!   and the leader-elected checkpoint-loop `JoinHandle` +
//!   `CancellationToken` so the test can drive a graceful shutdown at the
//!   end. The loop's `interval` and `quiesce` are set to zero so every
//!   commit is immediately due — `wait_for_checkpoint` deadlines stay
//!   tight under CI jitter.
//! - [`wipe_layer_state`] `UNLINK`s every `le:{oid}:obj:*` key for the
//!   target `object_id`, mirroring the data-loss surface the recovery
//!   path observes when Valkey state has been lost (failover to a fresh
//!   replica, persistence wipe, etc.).
//! - [`wait_for_checkpoint`] polls `list_layer_checkpoints` against the
//!   target `object_id` with a per-test deadline so the harness can
//!   block until the leader-elected loop has landed at least one
//!   generation. The default deadline (10 s) covers the worst-case CI
//!   path through the `tick=1s` schedule with a comfortable safety
//!   margin.
//! - [`build_recovery_hook_for_e2e`] constructs a real (enabled)
//!   `RecoveryHook` pointed at the same `LocalDiskCheckpointStore` as
//!   the loop, so the recovery branch in `handle_update_layer` /
//!   `handle_stat_layer` / `handle_subscribe_read_layer` can actually
//!   restore from the just-landed checkpoint instead of hitting the
//!   disabled short-circuit that the production `build_service` plumbs
//!   in by default. It also installs the two collaborators `src/main.rs`
//!   installs beyond `RecoveryHook::new` — see [`E2eRecoveryHookWiring`],
//!   and [`recovery_hook_for_e2e`] for the unwrapped form that lets a
//!   caller chain test-only builders without re-deciding the wiring.
//!
//! Sibling Subtasks `.rt`, `.cm`, `.nx`, `.w4`, `.4q`, `.rh`, `.r1`
//! consume every item in this module.
//!
//! ## StorageApi-backed harness (Task `t2.fgj.jd.ra`)
//!
//! Subtask `t3.fgj.jd.ra.13` extends this module with an analogous
//! storage-api-adapter harness consumed by sibling Subtask
//! `t3.fgj.jd.ra.z1`. The contract mirrors the LocalDisk surface so the
//! e2e DR test body for the storage-api adapter can re-use the same
//! [`wipe_layer_state`] / [`wait_for_checkpoint`] / metric-read shape:
//!
//! - [`setup_storage_api_checkpoint_harness`] bundles a `TestValkey`, a
//!   per-harness `wiremock::MockServer` rooted under a per-test base path
//!   (`BASE_PATH_STORAGE_API`), an in-process stateful PUT/GET/LIST/DELETE
//!   fake (`Arc<Mutex<HashMap<String, Bytes>>>`) mounted as default
//!   wiremock routes, a tempfile-backed [`ServiceTokenProvider`] primed
//!   with `Bearer test-token`, a `StorageApiCheckpointStore` wrapped
//!   behind `Arc<dyn CheckpointStore>`, a `LiveEditService` wired with a
//!   real (not disabled) `RecoveryHook` AND with
//!   `service_token_provider = Some(provider)` (the outbound-auth path is
//!   exercised end-to-end), and the leader-elected checkpoint-loop
//!   `JoinHandle` + `CancellationToken`.
//! - The stateful PUT/GET/LIST/DELETE handlers are lifted from the
//!   conformance harness in `adapter_storage_api_test.rs` so the fake's
//!   correctness target is identical: layer-PUT captures body bytes,
//!   subsequent GET on the same OMPF path returns the previously-PUT
//!   bytes, LIST enumerates the OMPF objects already PUT under a prefix.
//!   The recovery hook's `get_latest_layer_checkpoint` round-trip
//!   depends on this stateful behaviour.
//! - The returned [`StorageApiCheckpointHarness`] exposes the
//!   `MockServer` handle (so tests can inspect `received_requests()`),
//!   the `TempDir` backing the token file (RAII), the
//!   `Arc<ServiceTokenProvider>`, and the same `tv` / `svc` / `metrics` /
//!   `loop_handle` / `cancel` / `checkpoint_adapter` fields as
//!   [`CheckpointHarness`]. Its `LayerStore` carries the same
//!   allocation-path wiring — `.with_metrics(...)` plus the b.b2j
//!   `.with_object_id_counter_repair(...)` hook — which
//!   [`assert_harness_carries_object_id_counter_repair`] pins structurally
//!   for both harnesses.
//!
//! Per-harness-call isolation: each [`setup_storage_api_checkpoint_harness`]
//! invocation allocates a fresh `MockServer`, a fresh
//! `FakeStorageApiState` map, and a fresh `TempDir` — cross-harness state
//! leakage is impossible by construction. The LocalDisk-equivalent
//! [`CheckpointHarness`] is left intact; the two harnesses coexist.
//!
//! The `Config` fields the storage-api harness writes
//! (`checkpoint_storage_api_endpoint`, `checkpoint_base_path`,
//! `checkpoint_connect_timeout_ms`, `checkpoint_request_timeout_ms`) are
//! the contract Task `t2.fgj.jd.dz` (Subtask 1) added to `Config`. A
//! downstream rename of any of those fields surfaces here as a compile
//! error rather than a silent drift.
//!
//! ## Task `t2.fgj.7v.o7` — End-to-end session-recovery integration tests
//!
//! Subtask `t3.fgj.7v.o7.z7` (this foundation Subtask) extends the helper
//! surface with three additional primitives the eight sibling Subtasks
//! consume, plus eight stub modules in `mod.rs` that the siblings then
//! replace with per-AC test bodies. The new helpers below complement the
//! session-checkpoint helpers landed by `t3.fgj.7v.fp.xx`
//! ([`wipe_session_state`], [`wait_for_session_checkpoint`]) and
//! `t3.fgj.7v.fp.rd` ([`build_session_recovery_hook_for_e2e`]) — those are
//! reused unchanged by the new modules.
//!
//! - [`session_recovery_metric`] — reads the
//!   `live_edit_checkpoint_recovery_attempts_total{scope="session", result, service=SERVICE_LABEL}`
//!   cell. Task 6 extended this metric with the `scope` label and
//!   harmonized the result vocabulary to
//!   `restored | not_recoverable | already_present | error`; this helper
//!   centralises the read shape that previously lived in three
//!   per-file definitions (`e2e_session_recovery_per_handler_test`,
//!   `e2e_session_recovery_concurrent_handlers_test`,
//!   `e2e_session_recovery_no_checkpoint_test`). Those existing local
//!   definitions are deliberately left in place — this helper is the
//!   contract for the new Task 7 e2e modules, not a retroactive refactor
//!   of Tasks 4/6's per-handler tests.
//! - [`wait_for_recovery_metric`] — bounded-poll variant of
//!   `session_recovery_metric` for cases where recovery may land between
//!   RPC completion and metric assertion (e.g. the `not_recoverable`
//!   sweep path in `e2e_session_list_sessions_recovery_test` — the
//!   path-index walk fires recovery for ghost members on its own
//!   schedule, decoupled from the RPC return). See `docs-internal/test-writing-guide.md`
//!   `## Synchronization Patterns`.
//! - [`object_id_counter_repair_metric`] and
//!   [`assert_harness_carries_object_id_counter_repair`] — the b.b2j
//!   counter-repair outcome-cell reader and the structural parity guard that
//!   pins the harness's `LayerStore` carries the repair hook. See the latter's
//!   docstring: this file assembles the `LayerStore` by hand, so a dropped
//!   builder call here is invisible to every test that consumes the harness.
//!   Both are generic over [`CounterRepairParityHarness`], which
//!   [`CheckpointHarness`] and [`StorageApiCheckpointHarness`] both implement —
//!   the guard is deliberately not per-harness, because a per-harness copy is
//!   exactly how the StorageApi harness came to be missing the wiring while its
//!   own comment claimed parity with the LocalDisk one.
//! - [`full_valkey_wipe`] — SCAN + UNLINK across every `tv.prefix()*` key.
//!   Cluster-aware via the same `is_clustered()` branch [`wipe_layer_state`]
//!   and [`wipe_session_state`] use. The parent Task body refers to a
//!   `flush_namespace` method on `TestValkey` for the AC #2 wipe step —
//!   no such method exists on `TestValkey`, so this helper supplies the
//!   equivalent.
//!
//! Eight stub modules land alongside the helper additions:
//!
//! - `e2e_session_checkpoint_lifecycle_test` — AC #1
//!   (`t3.fgj.7v.o7.hr`).
//! - `e2e_session_full_wipe_recovery_test` — AC #2
//!   (`t3.fgj.7v.o7.c6`).
//! - `e2e_session_recovery_atomicity_test` — AC #3 / Step B
//!   (`t3.fgj.7v.o7.6f`).
//! - `e2e_session_recovery_path_index_atomicity_test` — AC #4 / Step A
//!   (`t3.fgj.7v.o7.xt`).
//! - `e2e_session_recovery_orphan_sweep_tolerance_test` — AC #5
//!   (`t3.fgj.7v.o7.c4`).
//! - `e2e_session_recovery_admission_test` — AC #6
//!   (`t3.fgj.7v.o7.u6`).
//! - `e2e_session_list_sessions_recovery_test` — AC #7
//!   (`t3.fgj.7v.o7.wz`).
//! - `e2e_session_cluster_recovery_test` — Cluster gate
//!   (`t3.fgj.7v.o7.v9`).

// The tail-capture fixtures (`drive_captured_tail` and the metric readers its
// callers use) are compiled only under `test-checkpoint-fault-injection`, or
// are used only by tests that are; without the feature they read as dead.
#![cfg_attr(
    not(feature = "test-checkpoint-fault-injection"),
    allow(dead_code, unused_imports)
)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use chrono::Utc;
use dashmap::DashMap;
use fred::interfaces::{ClientLike, KeysInterface};
use fred::types::scan::ScanType;
use futures::StreamExt;
use ordered_float::OrderedFloat;
use percent_encoding::percent_decode_str;
use serde_json::json;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use live_edit::auth::service_token::ServiceTokenProvider;
use live_edit::checkpoint::adapters::{LocalDiskCheckpointStore, StorageApiCheckpointStore};
use live_edit::checkpoint::{
    encode_tail_envelope, spawn_checkpoint_loop, BaseBootstrap, CheckpointAdapter,
    CheckpointHandle, CheckpointKind, CheckpointLoopConfig, CheckpointMeta,
    CheckpointObjectIdCounterRepair, CheckpointStore, RecoveryHook, SessionCheckpointWriter,
    SessionRecoveryHook, TailCaptureConfig, TailHeader, TailRecord,
};
use live_edit::config::Config;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::counters::CounterStore;
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::external_values::{
    ExternalValueStore, MockExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
// b.h4m. Gated exactly as the seam it carries is: `WatchWindowHook` itself is
// `#[cfg(any(test, feature = "test-fault-injection"))]` in `src`, and the
// integration binary is also compiled WITHOUT that feature (the release-shape
// clippy pass in CLAUDE.md § Build Commands), where naming the type would not
// resolve.
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::watched_txn::WatchWindowHook;
use tokio_util::task::TaskTracker;

use crate::grpc::helpers::{AllowAllPermissions, StubUserinfo};

/// Service label stamped on the `Metrics` registry and the
/// `LiveEditService::service_label` field. Stable across every E2E
/// recovery test so the `recovery_metric` helpers in sibling test files
/// can read the same label-cell.
///
/// A **re-export**, not a second `"test"` literal. This name is the *write*
/// side (helpers here stamp it onto `Config::service_label` and the metrics
/// registry) and [`crate::common::RECOVERY_SERVICE_LABEL`] is the *read* side
/// (`common::recovery_metric` pins it in the `{service=...}` cell it samples).
/// Two literals spelled the same is not the same guarantee as one address:
/// re-typing either copy would make every cross-file label-cell read return 0
/// forever and every assertion built on it vacuous, and nothing would fail
/// loudly. Aliasing makes that drift unrepresentable rather than merely
/// unlikely, which is what `RECOVERY_SERVICE_LABEL`'s own doc-comment claims
/// this arrangement does.
pub use crate::common::RECOVERY_SERVICE_LABEL as SERVICE_LABEL;

/// Lower bound on the effective deadline used by the `wait_for_*` checkpoint
/// helpers below. Callers historically pass 10 s, which is comfortable on an
/// unloaded box but flakes under heavy parallel load on a contended Valkey
/// (CI's `test-cluster` job on a 3-shard Valkey Cluster, multiple test binaries
/// hammering a shared dev/sandbox Valkey, etc.). The helpers clamp the caller's
/// `deadline_secs` to at least this value so a load-induced stall is absorbed
/// rather than reported as a spurious panic — a `tokio::time::sleep`-based
/// poll loop only finishes early when the awaited condition holds, so a larger
/// floor costs nothing on the happy path.
///
/// Applies uniformly to `wait_for_checkpoint{,_at_least}`, `wait_for_session_
/// checkpoint{,_absent,_quiesced}`, and `wait_for_writer_idle`. Each helper
/// names this constant in its panic message so a true logic failure (writer
/// never enqueues, lifecycle hook short-circuits, etc.) still surfaces with a
/// clear "even after the load-tolerant floor expired, the awaited state never
/// converged" report.
const LOAD_TOLERANT_CHECKPOINT_FLOOR: Duration = Duration::from_secs(30);

/// Bundle returned by [`setup_checkpoint_harness`]. Owns every piece of
/// state a recovery E2E test needs to drive a write → checkpoint →
/// wipe → recover cycle end-to-end.
///
/// The `TempDir` is kept alive through this struct so the LocalDisk
/// adapter root survives until the harness is dropped. The loop
/// `JoinHandle` and its `CancellationToken` let the test fire a
/// graceful shutdown at the end (cancel + `await` the handle); for
/// simple tests that don't need a clean shutdown, dropping the harness
/// at end-of-scope is sufficient — the `tokio` runtime tears the spawned
/// task down with the test runtime.
pub struct CheckpointHarness {
    /// Live Valkey connection + key schema shared across the service,
    /// the recovery hook, and the periodic checkpoint loop.
    pub tv: TestValkey,
    /// `Arc<dyn CheckpointStore>` rooted at [`CheckpointHarness::tempdir`].
    /// Shared with the `RecoveryHook` so the hook reads the same
    /// generations the loop writes.
    pub store: Arc<dyn CheckpointStore>,
    /// Live `LiveEditService` wired against the real `LocalDisk`-backed
    /// `RecoveryHook`. Tests pass this directly to
    /// `handle_create_layer`, `handle_update_layer`,
    /// `handle_stat_layer`, and `handle_subscribe_read_layer`.
    pub svc: LiveEditService,
    /// Live `SharedMetrics` shared between the service and the loop —
    /// recovery-attempt counter assertions read this Arc.
    pub metrics: SharedMetrics,
    /// `JoinHandle` for the periodic checkpoint loop spawned by
    /// [`spawn_checkpoint_loop`]. Tests that need a clean shutdown
    /// should `cancel.cancel()` and `handle.await` at the end.
    pub loop_handle: JoinHandle<()>,
    /// `CancellationToken` paired with [`CheckpointHarness::loop_handle`].
    /// Fire it to drain the loop.
    pub cancel: CancellationToken,
    /// `TaskTracker` shared with the harness's [`SessionCheckpointWriter`].
    /// Every CreateSession / CreateLayer / DeleteLayer / DeleteSession
    /// fires a fire-and-forget put / delete task that is registered on
    /// this tracker. [`wait_for_session_checkpoint_quiesced`] polls
    /// `tracker.is_empty()` so callers can wait for end-state quiescence
    /// — observing one blob on disk does NOT prove the writer is idle
    /// when multiple lifecycle events were dispatched in sequence (e.g.,
    /// CreateSession + CreateLayer enqueue two writes; the first to land
    /// satisfies `wait_for_session_checkpoint`, but the second may still
    /// be in flight and will silently overwrite a subsequent
    /// `delete_session_checkpoint`).
    pub session_checkpoint_tracker: TaskTracker,
    /// LocalDisk adapter root — kept alive for the duration of the
    /// test. RAII-cleaned on drop. Never read by tests directly; the
    /// field exists purely so the `TempDir` survives until the harness
    /// itself is dropped (the adapter holds a `PathBuf` rooted under
    /// this directory). A field-scoped `allow(dead_code)` is correct
    /// here — the crate-level allow that previously covered the whole
    /// module was removed so unused helpers surface as lint errors.
    #[allow(dead_code)]
    pub tempdir: TempDir,
    /// The `Config` this harness's recovery hooks, counter-repair hook,
    /// first-base bootstrap gate and loop tail knobs were all built from —
    /// [`harness_config_with_seq_margin`]'s output, cloned.
    ///
    /// Exposed so a test can build a SECOND collaborator against the same
    /// wiring (e.g. `build_recovery_hook_for_e2e` with a fault-injecting store,
    /// or a hook driven directly rather than through a handler) without
    /// re-deriving the config and silently drifting from the one the harness's
    /// own hook runs under. Re-deriving it is how a test ends up asserting
    /// against a `checkpoint_enabled = false` hook that short-circuits every
    /// restore to `NotRecoverable`.
    pub config: Config,
    /// The adapter this harness was built against. Recorded so
    /// [`assert_harness_carries_object_id_counter_repair`] can state the
    /// production wiring condition structurally rather than by convention —
    /// `main.rs` installs the b.b2j repair hook under exactly
    /// `!matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled)`, and
    /// every harness in this module is built with a real adapter.
    pub checkpoint_adapter: CheckpointAdapter,
    /// The ONE [`BaseBootstrap`] this harness constructed. `svc.base_bootstrap`
    /// holds a clone of it, and — unless
    /// [`CheckpointHarnessOptions::wire_restore_rebase`] was cleared — so does
    /// `svc.recovery`.
    ///
    /// Exposed so a parity test can assert `Arc::ptr_eq(&harness.base_bootstrap,
    /// &harness.svc.base_bootstrap)`: the single-unit property is a wiring
    /// **invariant**, not a saving, because
    /// [`BOOTSTRAP_MAX_INFLIGHT`](live_edit::checkpoint::BOOTSTRAP_MAX_INFLIGHT)
    /// is a field on the unit (`RecoveryHook::with_base_bootstrap`'s rustdoc,
    /// `src/checkpoint/recovery.rs`). The hook stores its copy in a private
    /// field with no accessor, so this is the structural half of the claim; the
    /// hook's half is carried behaviourally (a restore that re-bases).
    pub base_bootstrap: Arc<BaseBootstrap>,
    /// Whether [`Self::base_bootstrap`] was threaded into the recovery hook —
    /// i.e. [`CheckpointHarnessOptions::wire_restore_rebase`] as this harness was
    /// built. Read by [`Self::install_recovery_hook`] so a rebuilt hook keeps the
    /// harness's chosen posture instead of silently re-wiring an opt-out away.
    pub wire_restore_rebase: bool,
}

impl CheckpointHarness {
    /// Rebuild this harness's `RecoveryHook` from the harness's OWN pieces,
    /// let the caller chain extra builders onto it, and install the result on
    /// `self.svc.recovery`.
    ///
    /// **Extend this rather than rebuilding a hook by hand.** A hand-rolled
    /// `RecoveryHook::new(..)` in a test file is how the harness/production
    /// wiring gap (Issue b.kfg) reached six separate copies: each one re-derived
    /// the collaborators, and each one could — and did — drop a builder call
    /// that `src/main.rs` installs. Everything this method threads (`store`,
    /// `svc.layer_store`, `config`, `metrics`, the ONE `ext_value_store`, the
    /// ONE `base_bootstrap` under the harness's own
    /// [`CheckpointHarnessOptions::wire_restore_rebase`] posture) is the harness's
    /// single instance, so a rebuilt hook cannot fork a second bound or observe a
    /// different Valkey keyspace than the service does.
    ///
    /// `f` receives the hook UNWRAPPED, so test-only builders chain naturally:
    ///
    /// ```ignore
    /// harness.install_recovery_hook(|hook| {
    ///     hook.with_restore_fault_injector_for_test(Arc::clone(&injector))
    /// });
    /// ```
    pub fn install_recovery_hook(&mut self, f: impl FnOnce(RecoveryHook) -> RecoveryHook) {
        self.install_recovery_hook_with_config(None, f)
    }

    /// [`Self::install_recovery_hook`] with an optional `Config` override.
    ///
    /// `None` uses `self.config` — the config the harness's own hook already runs
    /// under, which is the safe default: a re-derived `Config` silently drifts,
    /// and a `checkpoint_enabled = false` one short-circuits every restore to
    /// `NotRecoverable` so every downstream assertion reads a no-op.
    ///
    /// `Some(cfg)` is for the arms that must dial a knob the harness cannot vary
    /// at construction time (a recovery fetch/wait budget, say). Derive it from
    /// `self.config.clone()` and mutate the one field rather than starting from
    /// `test_config()`.
    pub fn install_recovery_hook_with_config(
        &mut self,
        config: Option<Config>,
        f: impl FnOnce(RecoveryHook) -> RecoveryHook,
    ) {
        let hook = recovery_hook_for_e2e(
            Arc::clone(&self.svc.layer_store),
            Arc::clone(&self.store),
            config.unwrap_or_else(|| self.config.clone()),
            Arc::clone(&self.metrics),
            E2eRecoveryHookWiring {
                ext_value_store: Arc::clone(&self.svc.ext_value_store),
                base_bootstrap: self
                    .wire_restore_rebase
                    .then(|| Arc::clone(&self.base_bootstrap)),
            },
        );
        self.svc.recovery = Arc::new(f(hook));
    }
}

/// Build the b.b2j object-id counter-repair hook exactly the way `main.rs`
/// Step 1b-iii.6 does: same `CounterStore` over the live client + key schema,
/// same `checkpoint_counter_safety_margin`, same
/// `Config::checkpoint_recovery_fetch_budget()` /
/// `Config::object_id_counter_repair_wait_budget()` pair.
///
/// Deriving both budgets from the harness `Config` rather than hard-coding them
/// is what keeps this wiring honest: a change to either derivation reaches the
/// harness automatically, so the e2e keyspace-loss test keeps exercising the
/// budgets production actually runs under.
fn object_id_counter_repair_for_harness(
    tv: &TestValkey,
    store: &Arc<dyn CheckpointStore>,
    cfg: &Config,
    metrics: &SharedMetrics,
) -> Arc<CheckpointObjectIdCounterRepair> {
    Arc::new(CheckpointObjectIdCounterRepair::new(
        Arc::clone(store),
        Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        )),
        cfg.checkpoint_counter_safety_margin,
        Arc::clone(metrics),
        cfg.checkpoint_recovery_fetch_budget(),
        cfg.object_id_counter_repair_wait_budget(),
    ))
}

/// The two structural facts the b.b2j counter-repair parity guard needs from a
/// harness, so [`assert_harness_carries_object_id_counter_repair`] covers every
/// harness in this module rather than only the LocalDisk one.
///
/// A trait rather than a second copy of the guard: the harness structs are
/// unrelated types with no shared supertype, but the claim being checked — "this
/// harness's `LayerStore` carries the repair hook production installs for its
/// adapter" — is word-for-word the same for both. A per-harness copy is how the
/// gap arose in the first place: [`StorageApiCheckpointHarness`] built its
/// `LayerStore` under a comment claiming "same wiring as the LocalDisk harness"
/// while omitting both `.with_metrics(...)` and
/// `.with_object_id_counter_repair(...)`, and no guard existed that could see it.
pub trait CounterRepairParityHarness {
    /// The checkpoint adapter this harness was built against. `main.rs` Step
    /// 1b-iii.6 installs the repair hook under exactly
    /// `!matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled)`, so
    /// this is the harness's half of that condition.
    fn repair_guard_adapter(&self) -> CheckpointAdapter;
    /// The registry the harness's `LayerStore` was handed via
    /// `.with_metrics(...)`. Reading the outcome cell off it is what makes the
    /// guard a behavioural witness rather than a restatement of the builder
    /// call — and it is why a harness missing `.with_metrics(...)` is itself a
    /// defect: an unhooked `LayerStore` with no metrics cannot tick the cell
    /// that would expose it.
    fn repair_guard_metrics(&self) -> &SharedMetrics;
}

impl CounterRepairParityHarness for CheckpointHarness {
    fn repair_guard_adapter(&self) -> CheckpointAdapter {
        self.checkpoint_adapter
    }
    fn repair_guard_metrics(&self) -> &SharedMetrics {
        &self.metrics
    }
}

impl CounterRepairParityHarness for StorageApiCheckpointHarness {
    fn repair_guard_adapter(&self) -> CheckpointAdapter {
        self.checkpoint_adapter
    }
    fn repair_guard_metrics(&self) -> &SharedMetrics {
        &self.metrics
    }
}

/// Read one `live_edit_object_id_counter_repair_total{outcome}` cell off the
/// harness's rendered scrape exposition. Absent cells read as `0` — an OTel
/// counter emits no series until its first observation.
///
/// The metric carries no `service` label (the `LayerStore` fallback arm holds
/// no service label), so `outcome` is the whole label set.
pub fn object_id_counter_repair_metric<H: CounterRepairParityHarness>(
    harness: &H,
    outcome: &str,
) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(harness.repair_guard_metrics()),
        "live_edit_object_id_counter_repair_total",
        &[("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// Structural parity guard: a harness built against a **non-`Disabled`**
/// checkpoint adapter MUST carry the b.b2j object-id counter-repair hook on its
/// `LayerStore`.
///
/// This is the counter-repair analogue of the `with_checkpoint_writer` note in
/// [`wait_for_session_checkpoint`]'s panic message, and it exists for the same
/// reason: the harness's `LayerStore` is assembled by hand in this file rather
/// than by `main.rs`, so a builder call dropped *here* is invisible to every
/// test that consumes the harness. The failure mode is quiet rather than loud —
/// an unhooked `LayerStore` still creates layers, it just re-establishes
/// `le:counter:object_id` at `0` after a keyspace loss and re-mints object ids
/// that surviving checkpoint blobs still claim, which is b.b2j verbatim.
///
/// Two assertions, in the order a reader needs them:
///
/// 1. The harness's adapter is non-`Disabled` — the exact condition `main.rs`
///    gates the hook on. If a future harness variant ever built on `Disabled`
///    the unhooked arm would be *correct* there, and this guard would be
///    asserting the wrong thing; failing loudly beats silently weakening.
/// 2. `unhooked_initialized == 0`. That cell increments only on
///    `LayerStore::allocate_object_id`'s no-hook arm, so a non-zero value is
///    positive proof the hook is absent. Safe to call at any point in a test —
///    the cell must never tick on this harness, whether or not the counter has
///    gone missing yet.
///
/// Generic over [`CounterRepairParityHarness`] so it applies to
/// [`CheckpointHarness`] (LocalDisk) and [`StorageApiCheckpointHarness`] alike.
/// The StorageApi harness is the reason the trait exists: it carried a "same
/// wiring as the LocalDisk harness" comment over a `LayerStore` that had neither
/// the hook nor `.with_metrics(...)`, and a guard bound to one concrete harness
/// type could not have caught that.
pub fn assert_harness_carries_object_id_counter_repair<H: CounterRepairParityHarness>(harness: &H) {
    assert!(
        !matches!(harness.repair_guard_adapter(), CheckpointAdapter::Disabled),
        "harness parity precondition: this guard encodes `main.rs`'s wiring condition \
         (`!matches!(config.checkpoint_adapter, Disabled)`), so it is only meaningful on a \
         harness built with a real adapter; got {:?}",
        harness.repair_guard_adapter()
    );
    assert_eq!(
        object_id_counter_repair_metric(harness, "unhooked_initialized"),
        0,
        "the harness LayerStore must carry `.with_object_id_counter_repair(...)` — the \
         `unhooked_initialized` cell increments ONLY on `allocate_object_id`'s no-hook arm, \
         so a tick here means this file (not production) dropped the builder call, and every \
         keyspace-loss test in this module is silently exercising the unhooked fallback \
         instead of the sidecar-derived repair"
    );
}

/// Every knob [`setup_checkpoint_harness_with_options`] varies, in one place.
///
/// The four named constructors below are thin wrappers over it. They used to be
/// three ~170-line near-copies of the same wiring, which is precisely how a
/// builder call goes missing from one of them and nothing notices — the failure
/// mode [`assert_harness_carries_object_id_counter_repair`]'s doc records for
/// `.with_object_id_counter_repair(...)` and the one the first-base bootstrap
/// (item 8 of this module's kind-aware-helpers work) would have inherited next.
/// One body means a new collaborator is wired once for every LocalDisk harness
/// in the module.
///
/// Construct with [`CheckpointHarnessOptions::new`] and mutate the fields you
/// care about; the defaults reproduce [`setup_checkpoint_harness`] exactly.
pub struct CheckpointHarnessOptions {
    /// `LIVE_EDIT_CHECKPOINT_TICK_SECS`. Tests typically pass `1`.
    pub tick_secs: u64,
    /// `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` on the recovery hook's `Config`.
    pub seq_safety_margin: u64,
    /// `None` (the default) roots a fresh [`LocalDiskCheckpointStore`] under the
    /// harness `TempDir`. `Some(store)` installs a caller-supplied store — a
    /// counting / failing / recording [`DelegatingCheckpointStore`] wrapper, say
    /// — everywhere the default one would have gone: the loop, the recovery
    /// hooks, the session-checkpoint writer, the counter-repair hook, the
    /// first-base bootstrap, and [`CheckpointHarness::store`].
    ///
    /// The `TempDir` is still created and still held, so a wrapper *around* a
    /// LocalDisk store built on `tempdir.path()` is not an option here — wrap
    /// whatever store you like; the harness's own tempdir simply goes unused.
    pub store: Option<Arc<dyn CheckpointStore>>,
    /// A PRE-BUILT external-value store to install everywhere the default one
    /// would go — the service AND the recovery hook, one `Arc` shared by both.
    ///
    /// `None` (the default) resolves through [`Self::ext_value_store_factory`],
    /// and with that also `None` lands
    /// [`default_ext_value_store_for_harness`] — a REAL
    /// `ValkeyExternalValueStore` over the harness's own `TestValkey`.
    ///
    /// Use this field only for a store that can be built with no knowledge of the
    /// harness (a `MockExternalValueStore`, a recording decorator over one).
    /// Anything that must be built against the harness's live Valkey, metrics, or
    /// checkpoint store belongs in [`Self::ext_value_store_factory`], which runs
    /// after those exist. Setting both is a wiring mistake and panics.
    pub ext_value_store: Option<Arc<dyn ExternalValueStore>>,
    /// Build the external-value store from the harness's own pieces, after `tv` /
    /// `store` / `metrics` exist and before the recovery hook is built.
    ///
    /// This is the resolution of the ordering constraint that produced four
    /// near-identical field-swap workarounds across this module: a
    /// `ValkeyExternalValueStore` has to be built against the harness's own
    /// `TestValkey` (key prefix + `Drop`-time cleanup), which does not exist until
    /// the harness has been constructed — so tests built the harness, then
    /// overwrote `svc.ext_value_store`, leaving the recovery hook pointing at the
    /// store the harness had already wired. The closure closes that window: the
    /// store it returns is the one BOTH the service and the hook hold.
    ///
    /// [`ExtStoreCtx`] carries the trio every landed variant needs — `tv`,
    /// `metrics`, and the `Arc<dyn CheckpointStore>` — so
    /// `.with_metrics(..).with_checkpoint_store(..)` chains build here rather
    /// than after the fact.
    ///
    /// A caller that also needs a TYPED handle back (an
    /// `Arc<ValkeyExternalValueStore>` to call `process_pending()` on, say)
    /// stashes it from inside the closure — build the concrete type, clone the
    /// `Arc` into an outer `Option`/`OnceCell`, return the erased one.
    ///
    /// Mutually exclusive with [`Self::ext_value_store`]; setting both panics.
    pub ext_value_store_factory: Option<ExtValueStoreFactory>,
    /// Whether the recovery hook is handed the harness's [`BaseBootstrap`], i.e.
    /// whether a restore may spawn an SR-3.6 post-restore gap re-base.
    ///
    /// Defaults to `true` — the production posture. `main.rs` calls
    /// `.with_base_bootstrap(..)` unconditionally and lets the unit's own
    /// three-way gate (`checkpoint_enabled && !Disabled && checkpoint_tail_enabled`)
    /// decide whether the spawn does anything, so `true` is what a harness must
    /// default to for its hook to be the shipped one.
    ///
    /// `false` threads `None` to the hook, which is the documented opt-out
    /// (see [`E2eRecoveryHookWiring::base_bootstrap`]): no re-base is spawned, so
    /// the restore's OWN cursor columns survive into the assertion phase. The one
    /// landed consumer that needs it is `e2e_full_wipe_composed_restore_test`
    /// Test 1, whose claim is `last_checkpoint_seq == S` — what the RESTORE
    /// stamped — and a successful re-base overwrites that with `seeded_seq`.
    ///
    /// It is NOT how a tail-flag-off deployment is expressed: leave this `true`
    /// and leave [`Self::tail`] disabled, exactly as production does.
    pub wire_restore_rebase: bool,
    /// The four `LIVE_EDIT_CHECKPOINT_TAIL_*` knobs. Written into the harness
    /// `Config`, from which the loop's own [`TailCaptureConfig`] is re-derived
    /// via [`TailCaptureConfig::from_config`] — the same round-trip `main.rs`
    /// performs, so a harness cannot drift from production's mapping.
    ///
    /// Defaults to [`TailCaptureConfig::default`] — whatever the BINARY default
    /// is, never a harness-local literal. Since b.3n1 that is `enabled: true`,
    /// so a stock harness runs the tail path exactly as a stock deployment does:
    /// the SR-7.1 three-way gate resolves TRUE, a `CreateLayer` lands the
    /// first-base bootstrap, and a restore spawns the SR-3.6 re-base.
    ///
    /// A test that needs the tail path INERT — because its expected numbers were
    /// derived under a flag-off deployment, or because flag-off *is* its subject
    /// — pins it explicitly:
    ///
    /// ```ignore
    /// opts.tail = TailCaptureConfig { enabled: false, ..TailCaptureConfig::default() };
    /// ```
    ///
    /// or reaches for [`setup_checkpoint_harness_tails_off`]. Pinning is
    /// deliberately the caller's job: hard-coding `false` here would make the
    /// harness silently contradict the shipped default, which is the one thing
    /// [`harness_at_the_default_tail_flag_writes_the_first_base_bootstrap_generation`]
    /// exists to forbid.
    pub tail: TailCaptureConfig,
    /// A hook awaited once inside `LayerStore::write_restored_object`'s `WATCH`
    /// window — after the `WATCH` on `obj:hdr` is armed, before the header read
    /// the b.h4m fence decides on (`with_restore_write_watch_window_hook_for_test`).
    ///
    /// `None` (the default) installs nothing, so every pre-existing harness is
    /// byte-for-byte unaffected.
    ///
    /// It belongs on the OPTIONS rather than being field-swapped afterwards for
    /// the same reason [`Self::ext_value_store_factory`] does: the `LayerStore`
    /// is built inside [`setup_checkpoint_harness_with_options`] and handed by
    /// `Arc` to the service, the recovery hook, the counter-repair hook and the
    /// session-checkpoint writer, so a store rebuilt after the fact would be a
    /// SECOND store — the hook would ride an object no restore ever runs
    /// through, and every assertion downstream of it would read a fence that
    /// never fired.
    ///
    /// The ordering problem this creates for a caller — the hook usually needs
    /// the `object_id`, which does not exist until a layer has been created on
    /// the harness — is the caller's to solve with an interior-mutability
    /// latch it fills in after setup (see
    /// `restore_write_fence_test.rs`'s `PeerLander`), not the harness's:
    /// deferring the whole `LayerStore` construction would defeat the shared-`Arc`
    /// invariant above.
    #[cfg(feature = "test-fault-injection")]
    pub restore_write_watch_window_hook: Option<WatchWindowHook>,
    /// A hook awaited once inside `LayerStore::write_restored_object`'s `WATCH`
    /// window at its LAST moment — after the header read found `obj:hdr` absent,
    /// after every mutation is queued, after the L2 pre-dispatch check said
    /// dispatch, and immediately before `EXEC`
    /// (`with_restore_write_before_exec_hook_for_test`).
    ///
    /// The SIBLING of [`Self::restore_write_watch_window_hook`], and the two are
    /// not interchangeable: that one fires BEFORE the fence's header read, so a
    /// peer header landed from it is caught by that read and the attempt exits at
    /// the pre-`MULTI` `PeerLandedHeader` arm without ever opening a `MULTI` —
    /// the `WATCH` is never load-bearing. A peer header landed from THIS hook
    /// lands in the read→`EXEC` window instead, so the `EXEC` trips and the
    /// attempt takes the `WatchedExec::Conflict` arm: the post-conflict
    /// positive-evidence re-read, its `PeerLandedHeader` / `Unavailable` split,
    /// and that read's own `classify_restore_header_read_error` call site. It is
    /// the only deterministic way into any of those.
    ///
    /// `None` (the default) installs nothing. Both may be set at once; they are
    /// separate fields on `LayerStore` for the same reason they are separate
    /// here.
    ///
    /// Same OPTIONS-not-field-swap and same latch-fill-after-setup constraints as
    /// the sibling above — see its doc; `restore_write_fence_test.rs`'s
    /// `PeerLander` is shared by both seams.
    #[cfg(feature = "test-fault-injection")]
    pub restore_write_before_exec_hook: Option<WatchWindowHook>,
}

impl CheckpointHarnessOptions {
    /// Defaults matching [`setup_checkpoint_harness`].
    pub fn new(tick_secs: u64) -> Self {
        Self {
            tick_secs,
            seq_safety_margin: crate::common::test_config().checkpoint_seq_safety_margin,
            store: None,
            ext_value_store: None,
            ext_value_store_factory: None,
            wire_restore_rebase: true,
            tail: TailCaptureConfig::default(),
            #[cfg(feature = "test-fault-injection")]
            restore_write_watch_window_hook: None,
            #[cfg(feature = "test-fault-injection")]
            restore_write_before_exec_hook: None,
        }
    }
}

/// Boxed builder for [`CheckpointHarnessOptions::ext_value_store_factory`].
///
/// A named alias rather than the inline `Box<dyn FnOnce(..) -> ..>` so the
/// options struct stays readable and so a future signature change lands in one
/// place.
pub type ExtValueStoreFactory = Box<dyn FnOnce(ExtStoreCtx<'_>) -> Arc<dyn ExternalValueStore>>;

/// What [`CheckpointHarnessOptions::ext_value_store_factory`] is handed: the
/// harness pieces an external-value store can be built against, borrowed at the
/// point in [`setup_checkpoint_harness_with_options`] where all three exist and
/// nothing has been wired yet.
///
/// The trio is exactly what the production construction chain in `src/main.rs`
/// consumes — the Valkey client + key schema (via `tv`), the metrics registry
/// (`.with_metrics(..)`, without which `gc_underflow_total` is silently dropped),
/// and the checkpoint store (`.with_checkpoint_store(..)`, the SR-3.2
/// fetch-on-miss handle). A closure that needs none of them can ignore the
/// argument entirely.
pub struct ExtStoreCtx<'a> {
    /// The harness's live Valkey. `tv.client()` / `tv.key_schema()` are the two
    /// arguments `ValkeyExternalValueStore::new` takes; using THIS `TestValkey`
    /// is the whole point — it carries the per-harness key prefix and the
    /// `Drop`-time cleanup.
    pub tv: &'a TestValkey,
    /// The registry the harness's service, loop, and hooks all share.
    pub metrics: &'a SharedMetrics,
    /// The harness's checkpoint store — the same `Arc` the loop, the recovery
    /// hooks, and `CheckpointHarness::store` hold. Pass it to
    /// `.with_checkpoint_store(..)` to arm the SR-3.2 fetch-on-miss fallback.
    pub checkpoint_store: &'a Arc<dyn CheckpointStore>,
}

/// The harness's DEFAULT external-value store: a real `ValkeyExternalValueStore`
/// over the harness's own `TestValkey`, with `.with_metrics(..)` attached.
///
/// **Why real and not a `MockExternalValueStore`.** The mock is not a stub — it
/// "behaves identically to `ValkeyExternalValueStore` in terms of dedup,
/// refcounting, and pending GC logic" (`src/storage/external_values.rs`) against
/// an in-process map. Once the recovery hook is wired with an ext store (which is
/// the point of [`E2eRecoveryHookWiring`]), a stock-harness composed restore
/// carrying external values would pass GREEN with every blob in process memory
/// and NOTHING in Valkey — a false green that no assertion in this module could
/// see. Building the production unit and letting reality decide is the same
/// resolution [`base_bootstrap_for_harness`] already applies to the bootstrap.
///
/// **Deliberately NOT wired here**, both available via
/// [`CheckpointHarnessOptions::ext_value_store_factory`]:
///
/// - `.with_checkpoint_store(..)` — the SR-3.2 fetch-on-miss fallback. Arming it
///   by default would let a stock-harness ext-value MISS silently self-heal out
///   of the checkpoint adapter, which is the same class of invisible green this
///   default exists to close. The tests whose subject IS that fallback wire it
///   explicitly (`composed_restore_ext_matrix_test`), and the ones whose subject
///   is the SR-3.4 pre-store ordering deliberately build without it.
/// - `.with_ext_mirror_sender(..)` — this harness runs no ext-mirror pipeline, so
///   there is no receiver to send to.
fn default_ext_value_store_for_harness(
    tv: &TestValkey,
    metrics: &SharedMetrics,
) -> Arc<dyn ExternalValueStore> {
    Arc::new(
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(metrics)),
    )
}

/// Construct a fully-wired recovery-E2E harness.
///
/// Defaults — every value below is set explicitly so a downstream config
/// drift surfaces as a compile error rather than a silent default flip:
///
/// - `LIVE_EDIT_CHECKPOINT_ENABLED=true`
/// - `LIVE_EDIT_CHECKPOINT_ADAPTER=LocalDisk` rooted at a fresh `TempDir`
/// - `LIVE_EDIT_CHECKPOINT_TICK_SECS=tick_secs` (caller-chosen; tests
///   typically pass `1` for fast convergence)
/// - `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS=0` and
///   `LIVE_EDIT_CHECKPOINT_QUIESCE_SECS=0` — the loop's skip-if-idle gate
///   would otherwise hold every just-written layer in skip-state for the
///   default 60 s interval and stall every `wait_for_checkpoint` deadline.
/// - `LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS=2`
/// - `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` at the binary default (`true` since
///   b.3n1) — use [`setup_checkpoint_harness_tails_off`] for a flag-off harness
///
/// The harness wires a real `RecoveryHook` (NOT the disabled stub from
/// `build_service`) against the same `LocalDiskCheckpointStore` the loop
/// is using, so the handler-level recovery branch can actually restore
/// from the generations the loop produces. That hook carries BOTH builders
/// `src/main.rs` installs — see [`E2eRecoveryHookWiring`] — over the harness's
/// single [`default_ext_value_store_for_harness`] and its single
/// [`base_bootstrap_for_harness`] unit.
pub async fn setup_checkpoint_harness(tick_secs: u64) -> CheckpointHarness {
    setup_checkpoint_harness_with_options(CheckpointHarnessOptions::new(tick_secs)).await
}

/// [`setup_checkpoint_harness`] with `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` pinned
/// explicitly OFF, independently of what the shipped binary default happens to
/// be (it flipped to `true` on 2026-09-11, b.3n1).
///
/// This is the entry point for a suite whose subject is NOT the tail path and
/// whose expected numbers were derived against an inert one — restore floors,
/// wipe windows, lineage selection, generation listings, ext `ref_count` deltas.
/// With the flag on, a restore additionally spawns the SR-3.6 post-restore
/// re-base, which seeds one more sequence number, adds a `Base` generation to
/// the listing and re-stamps the restore's own cursor columns; pinning the flag
/// keeps such a suite measuring what it was written to measure instead of
/// absorbing the re-base into re-derived constants.
///
/// It is NOT a way to express "the harness should follow a flag-off default" —
/// there is no such default. [`CheckpointHarnessOptions::tail`] tracks the
/// binary, and that contract is pinned by
/// [`harness_at_the_default_tail_flag_writes_the_first_base_bootstrap_generation`].
#[allow(dead_code)]
pub async fn setup_checkpoint_harness_tails_off(tick_secs: u64) -> CheckpointHarness {
    setup_checkpoint_harness_with_options(tails_off_options(tick_secs)).await
}

/// [`setup_checkpoint_harness_tails_off`] with the recovery hook's
/// `checkpoint_seq_safety_margin` overridden — the tails-off twin of
/// [`setup_checkpoint_harness_with_seq_margin`], for the b.8y5 seq-floor suites
/// whose floors are asserted to the unit.
#[allow(dead_code)]
pub async fn setup_checkpoint_harness_tails_off_with_seq_margin(
    tick_secs: u64,
    seq_safety_margin: u64,
) -> CheckpointHarness {
    let mut opts = tails_off_options(tick_secs);
    opts.seq_safety_margin = seq_safety_margin;
    setup_checkpoint_harness_with_options(opts).await
}

/// [`CheckpointHarnessOptions::new`] with `tail.enabled` pinned explicitly
/// `false`, in the same `..TailCaptureConfig::default()` shape the production
/// fixtures use — every other tail knob keeps its shipped value, so only the
/// one variable moves.
#[allow(dead_code)]
pub fn tails_off_options(tick_secs: u64) -> CheckpointHarnessOptions {
    let mut opts = CheckpointHarnessOptions::new(tick_secs);
    opts.tail = TailCaptureConfig {
        enabled: false,
        ..TailCaptureConfig::default()
    };
    opts
}

/// [`setup_checkpoint_harness`] with the recovery hook's
/// `checkpoint_seq_safety_margin` overridden.
///
/// Every other knob matches [`setup_checkpoint_harness`], so callers reuse
/// the same helpers (`wait_for_checkpoint`, `wipe_layer_state`, …) without
/// divergence. Consumed by the b.8y5 seq-advance E2E tests, which need
/// `snap_seq > margin` — unreachable at the default margin of `1000`
/// without issuing a thousand UpdateLayers.
#[allow(dead_code)]
pub async fn setup_checkpoint_harness_with_seq_margin(
    tick_secs: u64,
    seq_safety_margin: u64,
) -> CheckpointHarness {
    let mut opts = CheckpointHarnessOptions::new(tick_secs);
    opts.seq_safety_margin = seq_safety_margin;
    setup_checkpoint_harness_with_options(opts).await
}

/// Construct a [`setup_checkpoint_harness`]-equivalent harness against a
/// caller-supplied, ALREADY-BUILT `Arc<dyn ExternalValueStore>` — installed on
/// the service AND the recovery hook, one `Arc` shared by both.
///
/// Prefer [`CheckpointHarnessOptions::ext_value_store_factory`] whenever the
/// store has to be built against the harness itself (anything Valkey-backed:
/// the `TestValkey` does not exist until the harness does). This entry point is
/// for stores that can be constructed standalone — a `MockExternalValueStore`
/// for a test that deliberately wants the in-process map, or a decorator over
/// one.
///
/// Note that the default is no longer a mock: see
/// [`default_ext_value_store_for_harness`].
#[allow(dead_code)]
pub async fn setup_checkpoint_harness_with_ext_store(
    tick_secs: u64,
    ext_value_store: Arc<dyn ExternalValueStore>,
) -> CheckpointHarness {
    let mut opts = CheckpointHarnessOptions::new(tick_secs);
    opts.ext_value_store = Some(ext_value_store);
    setup_checkpoint_harness_with_options(opts).await
}

/// Construct a [`setup_checkpoint_harness`]-equivalent harness against a
/// caller-supplied `Arc<dyn CheckpointStore>`.
///
/// The supplied store replaces the internally-rooted `LocalDiskCheckpointStore`
/// **everywhere** the default one goes — loop, recovery hook, session-recovery
/// hook, session-checkpoint writer, object-id counter repair, first-base
/// bootstrap, and the `store` field the test reads generations back through.
/// A wrapper that intercepted only some of those would let the intercepted call
/// and the observed state disagree, which is the failure this single wiring
/// point exists to prevent.
///
/// The intended shape is a [`DelegatingCheckpointStore`]-based newtype: a
/// counting wrapper, or one that fails a selected `put_layer_checkpoint`
/// (`snap_seq == 1 && kind == Base` targets the bootstrap PUT alone, which the
/// `CheckpointFaultInjector` seam cannot do — it dispatches on the method-name
/// label, and the bootstrap PUT and the loop PUT share one).
///
/// `Config::checkpoint_adapter` stays `LocalDisk`: it is the harness's claim
/// about which production wiring arm is being modelled (it is what
/// [`assert_harness_carries_object_id_counter_repair`] checks and what the
/// recovery hook's `Config` reports), not a statement about the concrete type
/// behind the trait object.
#[allow(dead_code)]
pub async fn setup_checkpoint_harness_with_store(
    tick_secs: u64,
    store: Arc<dyn CheckpointStore>,
) -> CheckpointHarness {
    let mut opts = CheckpointHarnessOptions::new(tick_secs);
    opts.store = Some(store);
    setup_checkpoint_harness_with_options(opts).await
}

/// The one body every LocalDisk-shaped [`CheckpointHarness`] is built from.
/// See [`CheckpointHarnessOptions`] for what varies and [`setup_checkpoint_harness`]
/// for the defaults.
pub async fn setup_checkpoint_harness_with_options(
    opts: CheckpointHarnessOptions,
) -> CheckpointHarness {
    let CheckpointHarnessOptions {
        tick_secs,
        seq_safety_margin,
        store,
        ext_value_store,
        ext_value_store_factory,
        wire_restore_rebase,
        tail,
        #[cfg(feature = "test-fault-injection")]
        restore_write_watch_window_hook,
        #[cfg(feature = "test-fault-injection")]
        restore_write_before_exec_hook,
    } = opts;

    assert!(
        !(ext_value_store.is_some() && ext_value_store_factory.is_some()),
        "CheckpointHarnessOptions: `ext_value_store` and `ext_value_store_factory` are two \
         spellings of the same slot and only one store is installed — setting both means one \
         of them is silently discarded, which is exactly the kind of half-applied wiring this \
         options struct exists to prevent. Pass the pre-built store OR the factory."
    );

    let tv = TestValkey::setup().await;

    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = store.unwrap_or_else(|| {
        Arc::new(
            LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
                .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
        )
    });

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    // The ONE external-value store. Resolved HERE — after `tv` / `store` /
    // `metrics` exist and strictly before the recovery hook is built — so that
    // the same `Arc` reaches `svc.ext_value_store` AND `svc.recovery`, the way
    // `main.rs` clones one object into both. Resolving it later (as the
    // `LiveEditService` literal used to) is what forced four consumer files to
    // field-swap a Valkey-backed store in afterwards, leaving the hook pointing
    // at a different store than the service.
    let ext_value_store: Arc<dyn ExternalValueStore> =
        match (ext_value_store, ext_value_store_factory) {
            (Some(prebuilt), _) => prebuilt,
            (None, Some(factory)) => factory(ExtStoreCtx {
                tv: &tv,
                metrics: &metrics,
                checkpoint_store: &store,
            }),
            (None, None) => default_ext_value_store_for_harness(&tv, &metrics),
        };

    // ONE `Config` for the whole harness: the recovery hooks, the counter-repair
    // hook, the first-base bootstrap gate, and (via
    // `TailCaptureConfig::from_config`) the loop's tail knobs all read it. The
    // three constructors this body replaced each built two or three separate
    // `Config`s from the same arguments; one value is what makes "the loop and
    // the bootstrap agree about whether the tail path is on" structural rather
    // than a convention.
    let harness_cfg = harness_config_with_seq_margin(
        tick_secs,
        tempdir.path().to_string_lossy().into_owned(),
        seq_safety_margin,
        &tail,
    );

    // Session-checkpoint writer wiring — required for the per-handler
    // session-recovery E2E tests (Subtask `t3.fgj.7v.fp.xx`). The writer is
    // installed on BOTH `SessionStore` and `LayerStore` so that
    // CreateSession / CreateLayer / DeleteLayer / DeleteSession lifecycle
    // events enqueue fire-and-forget session-checkpoint writes (and
    // DeleteSession enqueues a delete) via the shared dispatcher. Without
    // the writer wired, `wait_for_session_checkpoint` would block forever:
    // no lifecycle event produces a session-checkpoint blob on disk for the
    // recovery hook to read back.
    //
    // The `TaskTracker` is cloned onto `CheckpointHarness` so tests that
    // need quiescence (e.g., wipe + `delete_session_checkpoint` + assert
    // blob absent) can wait for in-flight writer tasks to drain via
    // [`wait_for_session_checkpoint_quiesced`]. Observing one blob on
    // disk via [`wait_for_session_checkpoint`] does NOT prove the writer
    // is idle: CreateSession + CreateLayer enqueue two writes; if a
    // subsequent `delete_session_checkpoint` races the second write, the
    // delete is silently overwritten.
    let session_checkpoint_tracker = TaskTracker::new();
    let session_checkpoint_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        session_checkpoint_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));

    // b.b2j: the harness holds a real (LocalDisk) checkpoint store, which is
    // exactly the condition `main.rs` Step 1b-iii.6 installs the object-id
    // counter-repair hook under. Wiring it here — with `.with_metrics(...)`,
    // without which every repair-outcome cell is silently dropped — is what
    // makes a keyspace-loss e2e test through `handle_create_layer` exercise the
    // production allocation path rather than the unhooked fallback that
    // re-establishes the counter at zero.
    let layer_store = {
        let built = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer))
            .with_metrics(Arc::clone(&metrics))
            .with_object_id_counter_repair(object_id_counter_repair_for_harness(
                &tv,
                &store,
                &harness_cfg,
                &metrics,
            ));
        // b.h4m: chained HERE, on the ONE store every consumer below shares, so
        // the hook fires inside the restore the harness's own recovery hook
        // drives. `None` leaves the builder uncalled, so the default harness is
        // byte-identical to its pre-b.h4m self.
        #[cfg(feature = "test-fault-injection")]
        let built = match restore_write_watch_window_hook {
            Some(hook) => built.with_restore_write_watch_window_hook_for_test(hook),
            None => built,
        };
        // b.h4m, second seam: the read→`EXEC` window, which is the ONLY way into
        // the tripped-`EXEC` (`WatchedExec::Conflict`) arm of the same fence.
        // Chained onto the same store for the same reason.
        #[cfg(feature = "test-fault-injection")]
        let built = match restore_write_before_exec_hook {
            Some(hook) => built.with_restore_write_before_exec_hook_for_test(hook),
            None => built,
        };
        Arc::new(built)
    };
    let session_store = Arc::new(
        SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_checkpoint_writer(Arc::clone(&session_checkpoint_writer)),
    );

    // The ONE `BaseBootstrap`, hoisted above the hook build for the same reason
    // the ext store was: `main.rs` constructs the unit once and hands the SAME
    // `Arc` to `LiveEditService` and to `RecoveryHook::with_base_bootstrap`.
    // Sharing one instance is a wiring INVARIANT, not a saving — the unit's
    // in-flight admission bound `BOOTSTRAP_MAX_INFLIGHT` is a field ON the unit
    // (see `RecoveryHook::with_base_bootstrap`'s rustdoc in
    // `src/checkpoint/recovery.rs`), so a second instance built for the hook
    // would be a second independent bound and "one bound across both sites"
    // would be silently untrue.
    let base_bootstrap = base_bootstrap_for_harness(
        &tv,
        &store,
        &harness_cfg,
        &session_checkpoint_tracker,
        &metrics,
    );

    // Build an ENABLED recovery hook against the real LocalDisk store.
    // This is the load-bearing difference vs. `build_service`'s disabled
    // stub — the handler-level recovery branch must actually call into
    // the store under E2E test, not short-circuit.
    let recovery = build_recovery_hook_for_e2e(
        Arc::clone(&layer_store),
        Arc::clone(&store),
        harness_cfg.clone(),
        Arc::clone(&metrics),
        E2eRecoveryHookWiring {
            ext_value_store: Arc::clone(&ext_value_store),
            base_bootstrap: wire_restore_rebase.then(|| Arc::clone(&base_bootstrap)),
        },
    );
    let session_recovery = build_session_recovery_hook_for_e2e(
        &tv,
        Arc::clone(&store),
        harness_cfg.clone(),
        Arc::clone(&metrics),
    );

    // Wire the session-recovery hook into the cache so the seven session-
    // resolving gRPC handlers (CreateLayer / UpdateLayer / DeleteLayer /
    // StatLayer / DeleteSession / JoinSession / SubscribeReadLayer) funnel
    // session-recovery through one choke point — matches the production
    // wiring in `main.rs` and is load-bearing for the per-handler recovery
    // E2E tests (Subtask `t3.fgj.7v.fp.xx`). Without this `.with_recovery_hook`
    // call, `SessionBasePathCache::get_base_path` would surface the raw
    // `NotFound` from `session_store.get_base_path` and recovery would
    // never fire — every per-handler E2E test would observe `NotFound`
    // instead of the expected `restored=1` outcome.
    let session_base_path_cache = live_edit::grpc::session_cache::SessionBasePathCache::new()
        .with_recovery_hook(Arc::clone(&session_recovery));

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store: Arc::clone(&layer_store),
        ext_value_store: Arc::clone(&ext_value_store),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics: Arc::clone(&metrics),
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache,
        recovery,
        session_recovery,
        base_bootstrap: Arc::clone(&base_bootstrap),
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

    // Spawn the leader-elected periodic checkpoint loop. `interval=0` +
    // `quiesce=0` disables the skip-if-idle gates so a just-written layer
    // is immediately due on the next tick.
    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_secs(tick_secs.max(1)),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this harness.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(SERVICE_LABEL),
        // Re-derived from the harness `Config` rather than moved in from the
        // options, exactly as `main.rs` derives it — so the loop and the
        // bootstrap gate can only ever read one source of truth.
        tail: TailCaptureConfig::from_config(&harness_cfg),
    };
    // A clone of the harness client, so a test that needs the loop's
    // tail-capture fallback for a layer arms the unprovable-cut seam on
    // `harness.tv.client()` once the layer exists — the seam is shared by every
    // clone (`ValkeyClient::arm_unprovable_checkpoint_cut_for_test`).
    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    CheckpointHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        session_checkpoint_tracker,
        tempdir,
        config: harness_cfg,
        checkpoint_adapter: CheckpointAdapter::LocalDisk,
        base_bootstrap,
        wire_restore_rebase,
    }
}

/// Build the SR-7.1 creation-time first-base bootstrap the way `main.rs` Step
/// 1b-iv.0 does, for a harness holding `store`.
///
/// **A real unit, not `common::build_disabled_base_bootstrap`.** The disabled stub
/// hardcodes `enabled: false` over a throwaway `InMemoryCheckpointStore`, so a
/// harness carrying it can never produce the AC-5 bootstrap base no matter how
/// its `Config` is set — the same class of gap
/// [`assert_harness_carries_object_id_counter_repair`]'s doc records for the
/// counter-repair hook, and the reason the stub is now confined to
/// `LiveEditService` literals in tests that have no checkpoint store at all.
///
/// Two things make installing it here safe for the ~40 pre-existing recovery
/// tests in this module:
///
/// 1. The gate is **derived**, never hardcoded — the same three-way conjunction
///    `main.rs` spells, read off the harness `Config`. With
///    `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` pinned off
///    ([`setup_checkpoint_harness_tails_off`], or `opts.tail.enabled = false`),
///    the conjunction is `false` and `spawn_creation_base` returns before
///    allocating, spawning, or issuing a single Valkey command. Byte-identical
///    to the stub's behaviour. Since b.3n1 that is no longer the DEFAULT
///    posture — a stock harness resolves the gate TRUE, exactly as a stock
///    deployment does — so a test that needs the pre-tail shape says so.
/// 2. It shares `session_checkpoint_tracker`, as `main.rs` does, so
///    [`wait_for_session_checkpoint_quiesced`] already drains bootstrap tasks
///    and no new shutdown wiring appears.
///
/// At the default (`opts.tail.enabled == true`) a harness CreateLayer therefore
/// performs a real AC-5 bootstrap against the harness's own store.
fn base_bootstrap_for_harness(
    tv: &TestValkey,
    store: &Arc<dyn CheckpointStore>,
    cfg: &Config,
    tracker: &TaskTracker,
    metrics: &SharedMetrics,
) -> Arc<BaseBootstrap> {
    Arc::new(BaseBootstrap::new(
        Arc::clone(store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(metrics),
        cfg.service_label.clone(),
        cfg.checkpoint_enabled
            && !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled)
            && cfg.checkpoint_tail_enabled,
    ))
}

/// The two collaborators `src/main.rs` installs on the production `RecoveryHook`
/// beyond `RecoveryHook::new`'s four arguments, made EXPLICIT at the call site
/// rather than left implicit.
///
/// Both were previously absent from [`build_recovery_hook_for_e2e`], so every
/// stock-harness E2E exercised a hook production does not ship (Issue b.kfg).
/// Making them a required argument — rather than an optional builder a caller
/// may forget — is what stops the defect recurring: a new consumer must state
/// its posture.
pub struct E2eRecoveryHookWiring {
    /// The external-value store the composed-restore path pre-stores a lineage's
    /// tail-carried values through, strictly before the single `write_object`
    /// MULTI/EXEC (SR-3.4).
    ///
    /// **Not `Option`, deliberately.** `main.rs` calls
    /// `.with_ext_value_store(..)` unconditionally, and a hook without one does
    /// not skip the pre-store quietly — it fails closed at the apply core's
    /// `Permanent` arm ("This is a wiring defect"). There is no legitimate
    /// harness posture for the `None` case, so the type does not offer it.
    ///
    /// Pass the SAME `Arc` the `LiveEditService` holds. A hook/service split is
    /// invisible to every assertion in this module: writes land in one store and
    /// reads come out of the other.
    pub ext_value_store: Arc<dyn ExternalValueStore>,
    /// The SR-3.6 post-restore gap re-base vehicle — the same [`BaseBootstrap`]
    /// instance `CreateLayer` uses (see
    /// [`CheckpointHarness::base_bootstrap`] for why one instance is an
    /// invariant).
    ///
    /// `None` is the DOCUMENTED OPT-OUT, restating
    /// `RecoveryHook::with_base_bootstrap`'s rustdoc in this harness's terms: no
    /// re-base is spawned on a restore — the pre-SR-3.6 behaviour exactly — so
    /// the restore's own cursor columns (`last_checkpoint_seq`,
    /// `restore_seq_bump`) survive into the assertion phase rather than being
    /// restamped by a successful re-base. Reach for it only when the test's claim
    /// IS about what the restore itself stamped.
    ///
    /// `None` is NOT how a tail-flag-off deployment is expressed. Production
    /// wires `Some` unconditionally and the unit's own three-way gate
    /// (`checkpoint_enabled && !Disabled && checkpoint_tail_enabled`) decides
    /// whether the spawn does anything — so a harness modelling "the tail feature
    /// is off" passes `Some(..)` and leaves
    /// [`CheckpointHarnessOptions::tail`] disabled, exactly as the binary does.
    pub base_bootstrap: Option<Arc<BaseBootstrap>>,
}

/// Build a real (enabled) `RecoveryHook` configured against the supplied
/// `CheckpointStore`, the harness `Config`, and the [`E2eRecoveryHookWiring`]
/// collaborators. Mirrors `build_disabled_recovery_hook` in
/// `tests/integration/grpc/helpers.rs` but flips `checkpoint_enabled=true` so the
/// handler-level recovery branch actually exercises the apply pipeline.
///
/// Installs BOTH builders `src/main.rs` installs, so a harness built on this is
/// running the shipped hook.
///
/// Exposed as a `pub fn` so individual sibling tests can swap in a
/// counting / fault-injecting `CheckpointStore` and still pin the same
/// config shape the harness uses by default. Callers that need to chain a
/// test-only builder (`with_restore_fault_injector_for_test`,
/// `with_force_post_restore_notfound_for_test`) want the UNWRAPPED
/// [`recovery_hook_for_e2e`] instead — or, from inside a harness,
/// [`CheckpointHarness::install_recovery_hook`], which supplies the wiring too.
pub fn build_recovery_hook_for_e2e(
    layer_store: Arc<LayerStore>,
    store: Arc<dyn CheckpointStore>,
    config: Config,
    metrics: SharedMetrics,
    wiring: E2eRecoveryHookWiring,
) -> Arc<RecoveryHook> {
    Arc::new(recovery_hook_for_e2e(
        layer_store,
        store,
        config,
        metrics,
        wiring,
    ))
}

/// [`build_recovery_hook_for_e2e`], UNWRAPPED.
///
/// Same body, same wiring; returns the `RecoveryHook` by value so a caller can
/// chain the test-only builders (`with_restore_fault_injector_for_test`,
/// `with_force_post_restore_notfound_for_test`) before `Arc::new`-ing it into
/// `svc.recovery`. That need is why several test files forked their own
/// `RecoveryHook::new(..)` — and each fork then had to re-decide the wiring,
/// which is where the missing builders came from.
pub fn recovery_hook_for_e2e(
    layer_store: Arc<LayerStore>,
    store: Arc<dyn CheckpointStore>,
    config: Config,
    metrics: SharedMetrics,
    wiring: E2eRecoveryHookWiring,
) -> RecoveryHook {
    let E2eRecoveryHookWiring {
        ext_value_store,
        base_bootstrap,
    } = wiring;
    let hook = RecoveryHook::new(store, layer_store, Arc::new(config), metrics)
        .with_ext_value_store(ext_value_store);
    match base_bootstrap {
        Some(bootstrap) => hook.with_base_bootstrap(bootstrap),
        None => hook,
    }
}

/// Build a real (enabled) `SessionRecoveryHook` against the supplied
/// `CheckpointStore`, Valkey wiring, and harness `Config`. Sibling to
/// [`build_recovery_hook_for_e2e`] for the session-state recovery surface
/// added in Subtask `t3.fgj.7v.fp.rd`.
pub fn build_session_recovery_hook_for_e2e(
    tv: &TestValkey,
    store: Arc<dyn CheckpointStore>,
    config: Config,
    metrics: SharedMetrics,
) -> Arc<SessionRecoveryHook> {
    Arc::new(SessionRecoveryHook::new(
        store,
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(config),
        metrics,
    ))
}

/// Build the `Config` the harness threads into the recovery hook, the
/// counter-repair hook, the first-base bootstrap gate, and (via
/// [`TailCaptureConfig::from_config`]) the loop's own tail knobs.
///
/// Fields are derived from `tests/common/mod.rs::test_config()` plus the
/// checkpoint-specific overrides documented on [`setup_checkpoint_harness`].
/// The local-disk path is the same tempdir the loop's
/// `LocalDiskCheckpointStore` is rooted at, so the recovery hook's config
/// faithfully reflects production wiring (env var → `Config::from_env` →
/// `LocalDiskCheckpointStore::new`).
///
/// The default margin (`1000`, from `common::test_config()`) towers over
/// the handful of commits an E2E test drives, which makes it impossible to
/// build a `snap_seq > margin` restore — the b.8y5 shape — without issuing
/// a thousand UpdateLayers. Tests that need that shape dial the margin down
/// via [`setup_checkpoint_harness_with_seq_margin`].
///
/// The tail knobs land on the `Config` rather than only on the loop's
/// [`TailCaptureConfig`] because TWO consumers must agree about them: the loop
/// (which re-derives its copy via [`TailCaptureConfig::from_config`]) and the
/// SR-7.1 first-base bootstrap gate, whose third conjunct is
/// `checkpoint_tail_enabled`. A harness that enabled the loop's tail path while
/// leaving the bootstrap's `Config` at `false` would run the AC-5 feature with
/// its lineage-root producer silently off.
fn harness_config_with_seq_margin(
    tick_secs: u64,
    local_disk_path: String,
    seq_safety_margin: u64,
    tail: &TailCaptureConfig,
) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::LocalDisk;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = tick_secs.max(1);
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_local_disk_path = Some(local_disk_path);
    cfg.checkpoint_seq_safety_margin = seq_safety_margin;
    cfg.checkpoint_tail_enabled = tail.enabled;
    cfg.checkpoint_tail_max_chain = tail.max_chain as u32;
    cfg.checkpoint_tail_max_entries = tail.max_entries;
    cfg.checkpoint_tail_max_bytes = tail.max_bytes;
    cfg
}

/// `UNLINK` every `le:{object_id}:obj:*` key, simulating the
/// data-loss surface the recovery path observes when Valkey state has
/// been lost. Leaves session metadata, layer mappings, and external-value
/// blobs intact — the recovery hook's contract is "restore the object's
/// own materialized state", not "rebuild the entire session graph".
///
/// SCAN + UNLINK; the `{object_id}` hash tag keeps every matching key on
/// one Cluster slot, so a single multi-key `UNLINK` is slot-safe. The
/// pattern is identical to the production assembler's per-object key
/// enumeration, ensuring this helper wipes exactly the surface the
/// assembler reads.
pub async fn wipe_layer_state(tv: &TestValkey, object_id: u64) {
    wipe_obj_keys(tv, object_id, None).await;
}

/// SCAN + UNLINK every `le:{object_id}:obj:*` key, optionally sparing one
/// named key. The single implementation behind [`wipe_layer_state`] and
/// [`wipe_layer_state_preserving_stream`] — the two differ only in whether
/// `skip` is supplied.
///
/// `skip` takes the FULLY-QUALIFIED key (prefix included), i.e. exactly what
/// `KeySchema::layer_stream` returns, because that is what SCAN yields.
async fn wipe_obj_keys(tv: &TestValkey, object_id: u64, skip: Option<&str>) {
    let pattern = format!("{}{{{object_id}}}:obj:*", tv.key_schema().prefix());
    let client = tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
        Box::pin(client.scan_cluster_buffered(pattern.clone(), Some(100), None::<ScanType>))
    } else {
        Box::pin(client.scan_buffered(pattern.clone(), Some(100), None::<ScanType>))
    };
    let mut to_delete = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(key) = item {
            if skip.is_some() && key.as_str() == skip {
                continue;
            }
            to_delete.push(key);
        }
    }
    if !to_delete.is_empty() {
        let _: () = tv
            .client()
            .pool
            .unlink(to_delete)
            .await
            .expect("UNLINK of obj keys must succeed");
    }
}

/// [`wipe_layer_state`], but SPARING `le:{object_id}:obj:stream`.
///
/// Models a **partial** state loss: the materialized `obj:*` hashes and
/// indices are gone (so `read_object_header` surfaces `NotFound` and the
/// recovery branch fires) while the per-object live-update stream — an
/// independent key that merely shares the `{object_id}` hash tag — survived.
///
/// This is the shape b.8y5's `probe_stream_high_water` exists for. Two
/// things follow from a surviving stream that do not follow from a wiped
/// one:
///
/// 1. Its `last-generated-id` is the EXACT pre-crash sequence number,
///    strictly better than the checkpoint's `snap_seq` (which lags by up
///    to one RPO window).
/// 2. Both commit paths `XADD` at `{new_seq}-0`, so seeding at or below
///    that high-water wedges every subsequent `UpdateLayer` on `ERR The ID
///    specified in XADD is equal or smaller than the target stream top
///    item`.
///
/// Returns the surviving stream key so callers can assert on it directly.
#[allow(dead_code)]
pub async fn wipe_layer_state_preserving_stream(tv: &TestValkey, object_id: u64) -> String {
    let stream_key = tv.key_schema().layer_stream(object_id);
    wipe_obj_keys(tv, object_id, Some(&stream_key)).await;

    // Sanity: the stream MUST have survived, otherwise the caller's probe
    // assertions would be vacuous (a wiped stream folds the restore floor
    // back to `snap_seq`, which is the arm this helper exists to avoid).
    let stream_exists: bool = tv
        .client()
        .pool
        .exists(&stream_key)
        .await
        .expect("EXISTS obj:stream must succeed");
    assert!(
        stream_exists,
        "wipe_layer_state_preserving_stream must leave {stream_key} intact"
    );

    stream_key
}

/// `UNLINK` every `le:{session_id}*` key — the data-loss surface the
/// session-recovery path is designed to defend against. Covers
/// `le:{sid}:sess` (session metadata hash), every `le:{sid}:layer:{hex(name)}`
/// mapping, and `le:{sid}:layers` (layers index). Leaves the path-index
/// (`le:path:{hex(base_path)}` + `:names`) intact: SR-2.7's asymmetric
/// self-healing contract says path-index members survive the wipe and the
/// recovery hook re-materializes the session-scoped keys from the
/// checkpoint blob. Wiping the path-index would simulate a different
/// failure mode (orphan-ghost) that has its own reconciliation path
/// (`reconcile_ghost_members`, `ListSessions`).
///
/// The `{session_id}` hash tag keeps every matching key on one Cluster
/// slot, so a single multi-key `UNLINK` is slot-safe. Consumed by the
/// per-handler session-recovery E2E tests (Subtask `t3.fgj.7v.fp.xx` /
/// siblings `.kv` / `.8n`).
pub async fn wipe_session_state(tv: &TestValkey, session_id: &str) {
    let pattern = format!("{}{{{session_id}}}*", tv.key_schema().prefix());
    let client = tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
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
    if !to_delete.is_empty() {
        let _: () = tv
            .client()
            .pool
            .unlink(to_delete)
            .await
            .expect("UNLINK of session keys must succeed");
    }
}

/// Poll [`CheckpointStore::get_session_checkpoint`] until a blob exists
/// for `session_id`, or panic with a diagnostic on deadline expiry.
///
/// Mirrors [`wait_for_checkpoint`] but for the session-checkpoint store
/// (which exposes `get_session_checkpoint` rather than a `list_*`
/// equivalent — session checkpoints carry no generation history). The
/// session-checkpoint writer's enqueue path is fire-and-forget through a
/// [`tokio_util::task::TaskTracker`], so a successful gRPC CreateSession
/// returns to the caller well before the spawned put-task lands the blob;
/// callers MUST poll for the blob's presence before they can assume the
/// checkpoint is durable.
///
/// Caller passes the deadline in seconds — typical use is `10`, generous
/// enough to absorb CI jitter on the writer's spawn + put latency while
/// short enough that a genuinely broken wiring (writer not installed on
/// the store) fails the test in bounded time rather than hanging
/// indefinitely.
///
/// Consumed by the per-handler session-recovery E2E tests
/// (`t3.fgj.7v.fp.xx` and siblings).
pub async fn wait_for_session_checkpoint(
    store: &Arc<dyn CheckpointStore>,
    session_id: &str,
    deadline_secs: u64,
) {
    let effective = Duration::from_secs(deadline_secs).max(LOAD_TOLERANT_CHECKPOINT_FLOOR);
    let deadline = Instant::now() + effective;
    let poll_interval = Duration::from_millis(50);
    loop {
        match store.get_session_checkpoint(session_id).await {
            Ok(Some(_)) => return,
            Ok(None) => {
                // Writer's put task has not landed yet — keep polling.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during wait_for_session_checkpoint \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_session_checkpoint timed out after {effective:?} (caller asked \
                 {deadline_secs}s, clamped to load-tolerant floor) waiting for \
                 session_id={session_id} (no blob landed). Common causes: the harness's \
                 SessionStore / LayerStore were constructed without `with_checkpoint_writer`, \
                 or the lifecycle hook in `SessionStore::create` / `LayerStore::create_layer` \
                 short-circuited before reaching the enqueue call."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Like [`wait_for_session_checkpoint`], but additionally waits for the
/// harness's [`SessionCheckpointWriter`] task tracker to be empty.
///
/// Use this when a subsequent test step calls `delete_session_checkpoint`
/// (or otherwise depends on the writer being IDLE, not just on at least
/// one blob having landed). [`wait_for_session_checkpoint`] returns as
/// soon as it observes one blob on disk; under fast Valkey it can return
/// before a sibling fire-and-forget write task that was dispatched
/// earlier has completed. Subsequent `delete_session_checkpoint` then
/// races the late write — the delete lands first, the write resurrects
/// the blob, and any "blob is gone" assertion fires spuriously.
///
/// Test paths that exercise `assert_session_fully_absent` after a wipe +
/// delete MUST use this helper. Paths that only need to observe one blob
/// (e.g., read-back-and-assert-content) can keep using
/// [`wait_for_session_checkpoint`].
pub async fn wait_for_session_checkpoint_quiesced(
    harness: &CheckpointHarness,
    session_id: &str,
    deadline_secs: u64,
) {
    wait_for_session_checkpoint(&harness.store, session_id, deadline_secs).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, deadline_secs).await;
}

/// Poll the supplied session-checkpoint-writer
/// [`TaskTracker`](tokio_util::task::TaskTracker) until it reports
/// `is_empty()`, or panic on `deadline_secs` expiry.
///
/// Useful when the caller drove the blob wait through a custom
/// `CheckpointStore` wrapper (e.g., a counting adapter) and cannot call
/// [`wait_for_session_checkpoint_quiesced`] directly — call this AFTER
/// the custom `wait_for_session_checkpoint` to drain the writer's
/// in-flight tasks before any subsequent `delete_session_checkpoint` /
/// wipe step. Takes `&TaskTracker` directly so file-local harnesses
/// (e.g., `CountingHarness` in `e2e_session_recovery_admission_test`)
/// that mirror [`CheckpointHarness`]'s writer wiring can reuse it
/// without conforming to the full [`CheckpointHarness`] shape.
pub async fn wait_for_writer_idle(tracker: &TaskTracker, deadline_secs: u64) {
    let effective = Duration::from_secs(deadline_secs).max(LOAD_TOLERANT_CHECKPOINT_FLOOR);
    let deadline = Instant::now() + effective;
    let poll_interval = Duration::from_millis(10);
    loop {
        if tracker.is_empty() {
            return;
        }
        if Instant::now() >= deadline {
            let in_flight = tracker.len();
            panic!(
                "wait_for_writer_idle timed out after {effective:?} (caller asked \
                 {deadline_secs}s, clamped to load-tolerant floor) waiting for \
                 the session-checkpoint writer's TaskTracker to drain. A \
                 fire-and-forget put / delete task has not completed; a downstream \
                 step that depends on writer quiescence (delete + assert blob \
                 absent, wipe + assert restored == 0, etc.) would race it. \
                 ({in_flight} tasks still in flight at deadline)"
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Poll [`CheckpointStore::get_session_checkpoint`] until the blob is
/// ABSENT for `session_id`, or panic with a diagnostic on deadline
/// expiry. Inverse of [`wait_for_session_checkpoint`].
///
/// Consumed by the `recovery_fires_on_delete_session_then_deletes` test
/// to assert the post-call invariant from Task 4 of `t1.fgj.7v`:
/// `SessionStore::delete` enqueues `enqueue_session_checkpoint_delete`,
/// which calls `delete_session_checkpoint` on the adapter — the blob is
/// gone after the DeleteSession completes.
pub async fn wait_for_session_checkpoint_absent(
    store: &Arc<dyn CheckpointStore>,
    session_id: &str,
    deadline_secs: u64,
) {
    let effective = Duration::from_secs(deadline_secs).max(LOAD_TOLERANT_CHECKPOINT_FLOOR);
    let deadline = Instant::now() + effective;
    let poll_interval = Duration::from_millis(50);
    loop {
        match store.get_session_checkpoint(session_id).await {
            Ok(None) => return,
            Ok(Some(_)) => {
                // Writer's delete task has not landed yet — keep polling.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during wait_for_session_checkpoint_absent \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_session_checkpoint_absent timed out after {effective:?} (caller \
                 asked {deadline_secs}s, clamped to load-tolerant floor) waiting for \
                 session_id={session_id}'s blob to be deleted. \
                 The DeleteSession enqueue_session_checkpoint_delete dispatch may have \
                 been short-circuited, or the writer's TaskTracker is starved."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

// ===========================================================================
// StorageApi-backed harness (Task `t2.fgj.jd.ra`, Subtask `t3.fgj.jd.ra.13`)
//
// Everything below this banner is the storage-api analog of
// `CheckpointHarness` / `setup_checkpoint_harness` above. The two harnesses
// coexist — the LocalDisk path is the original Epic 1 contract and is
// unchanged; this section adds a wiremock-backed `StorageApiCheckpointStore`
// path consumed by sibling Subtask `t3.fgj.jd.ra.z1`.
// ===========================================================================

/// Connect-phase timeout threaded into the storage-api adapter's
/// `reqwest::Client`. Mirrors the realistic-timeout pair used by
/// `tests/integration/checkpoint/adapter_storage_api_test.rs` so the
/// harness's adapter wiring matches the conformance harness exactly.
const STORAGE_API_CONNECT_TIMEOUT_MS: u64 = 5_000;

/// Per-request timeout threaded into the storage-api adapter's
/// `reqwest::Client`. Mirrors the realistic-timeout pair noted above.
const STORAGE_API_REQUEST_TIMEOUT_MS: u64 = 30_000;

/// Base path the storage-api harness composes every `<base>/.live-edit/...`
/// address under. Pinned per-harness-call (one mock-server per harness, one
/// state map per harness), so a single fixed value is correct — there is no
/// cross-harness collision risk.
const BASE_PATH_STORAGE_API: &str = "/test/live-edit-e2e";

/// Poll interval for the `ServiceTokenProvider`'s background polling task.
/// Short enough that an operator-style file rotation would converge quickly
/// without observably gating the test; the actual cache-population happens
/// through an explicit `provider.force_refresh().await` during harness
/// construction, so the polling task's tick cadence is not on the critical
/// path. Tests that rewrite [`StorageApiCheckpointHarness::token_path`]
/// mid-run should likewise drive `force_refresh()` themselves rather than
/// sleeping for a tick — the cadence here is a convergence backstop, not a
/// synchronisation point.
///
/// `#[allow(dead_code)]`: this Subtask lands the harness; the sibling
/// Subtask `t3.fgj.jd.ra.z1` lands the e2e test body that consumes it.
/// Until the sibling lands, every helper below is `dead` to the compiler.
#[allow(dead_code)]
const STORAGE_API_TOKEN_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Stable bearer-token value the harness primes the `ServiceTokenProvider`
/// with. Sibling test `t3.fgj.jd.ra.z1` asserts every outbound GET carries
/// `Authorization: Bearer test-token`, so this constant is the contract.
#[allow(dead_code)]
const STORAGE_API_TEST_BEARER: &str = "test-token";

/// In-process emulation of the storage-api `by-address` + `data-objects/`
/// routes. Maps **percent-decoded** addresses to their recorded bytes; PUT
/// inserts, GET reads, HEAD stats, DELETE removes, LIST enumerates keys by
/// prefix.
///
/// Wrapped in an `Arc<Mutex<...>>` so the default-route closures share state
/// across the five method-handlers (PUT, GET, HEAD, LIST, DELETE) for one
/// `MockServer`. Per-harness isolation is preserved at the
/// [`setup_storage_api_checkpoint_harness`] level — every call allocates a
/// fresh `FakeStorageApiState` instance.
///
/// This is a faithful re-implementation of the same stateful fake that lives
/// in `tests/integration/checkpoint/adapter_storage_api_test.rs`. We do NOT
/// share the implementation across modules because the conformance harness
/// has narrower wiring requirements (no `LiveEditService`, no
/// `RecoveryHook`, no `JoinHandle`) and lifting a shared submodule out would
/// introduce cross-file coupling for marginal savings.
#[allow(dead_code)]
type FakeStorageApiState = Arc<Mutex<HashMap<String, Bytes>>>;

/// `Respond` impl backing the PUT route. Records the request body in the
/// shared map under the percent-decoded `by-address` tail, then returns 201
/// with an opaque-handle JSON payload.
#[allow(dead_code)]
struct StorageApiPutHandler {
    state: FakeStorageApiState,
}

impl Respond for StorageApiPutHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let body = Bytes::copy_from_slice(&request.body);
        self.state
            .lock()
            .expect("fake storage state mutex must not be poisoned")
            .insert(address.clone(), body);
        ResponseTemplate::new(201).set_body_json(json!({
            "resource_address": address,
        }))
    }
}

/// `Respond` impl backing the GET route. Looks up the percent-decoded
/// `by-address` tail; on hit returns 200 + bytes, on miss returns 404.
#[allow(dead_code)]
struct StorageApiGetHandler {
    state: FakeStorageApiState,
}

impl Respond for StorageApiGetHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        match map.get(&address) {
            Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()),
            None => ResponseTemplate::new(404),
        }
    }
}

/// `Respond` impl backing the HEAD route — storage-api's `stat_by_address`.
/// On hit returns 204 with no body, on miss 404. Without it an unmatched HEAD
/// gets wiremock's default 404, so `ext_blob_exists` and the put-rollback
/// existence probes would read "absent" whatever the map holds.
#[allow(dead_code)]
struct StorageApiHeadHandler {
    state: FakeStorageApiState,
}

impl Respond for StorageApiHeadHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        if map.contains_key(&address) {
            ResponseTemplate::new(204)
        } else {
            ResponseTemplate::new(404)
        }
    }
}

/// `Respond` impl backing the DELETE route. Removes the entry under the
/// percent-decoded `by-address` tail; on hit returns 204, on miss returns
/// 404 (the adapter swallows that 404 per the idempotent-on-missing
/// contract).
#[allow(dead_code)]
struct StorageApiDeleteHandler {
    state: FakeStorageApiState,
}

impl Respond for StorageApiDeleteHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let mut map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        if map.remove(&address).is_some() {
            ResponseTemplate::new(204)
        } else {
            ResponseTemplate::new(404)
        }
    }
}

/// `Respond` impl backing the LIST (Enumerate) route. Percent-decodes the
/// prefix from the URL tail, scans the shared map for matching keys, and
/// returns an `EnumerateResult` JSON page with the matching keys as
/// `AddressInfo` entries. Always single-page (`next_continuation_handle:
/// null`).
///
/// A prefix that matches no key answers **404 with no body**, as the real
/// storage-api does for an Enumerate of a prefix holding no objects
/// (storage-core `enumerate_nonexistent_prefix`), so the adapter's
/// first-page-404 → empty mapping runs here rather than a 200 with no items
/// the real service never sends. A `continuation_handle` query parameter
/// answers 404 too, as the real service does once a later page holds
/// nothing; it is unreachable, because this fake never hands out a
/// continuation handle.
#[allow(dead_code)]
struct StorageApiListHandler {
    state: FakeStorageApiState,
}

impl Respond for StorageApiListHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if request
            .url
            .query_pairs()
            .any(|(k, _)| k == "continuation_handle")
        {
            return ResponseTemplate::new(404);
        }
        let Some(prefix) = decode_data_objects_prefix_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        let mut items: Vec<serde_json::Value> = map
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| json!({ "resource_address": k }))
            .collect();
        items.sort_by(|a, b| {
            a.get("resource_address")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .cmp(
                    b.get("resource_address")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )
        });
        if items.is_empty() {
            return ResponseTemplate::new(404);
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "next_continuation_handle": null,
            "items": items,
        }))
    }
}

/// Percent-decode the `by-address` tail of a request path. Returns
/// `Some(decoded_address)` for paths shaped like
/// `/v1alpha/fileobject/by-address/<encoded>`; `None` otherwise.
///
/// The adapter percent-encodes the entire composed address (including the
/// leading `/`) through `encode_resource_address`, so the round-trip here
/// recovers the original decoded address verbatim.
#[allow(dead_code)]
fn decode_by_address_from_path(path: &str) -> Option<String> {
    const PREFIX: &str = "/v1alpha/fileobject/by-address/";
    let encoded = path.strip_prefix(PREFIX)?;
    let decoded = percent_decode_str(encoded).decode_utf8().ok()?;
    Some(decoded.into_owned())
}

/// Percent-decode the `data-objects/` tail of a request path. Returns
/// `Some(decoded_prefix)` for paths shaped like
/// `/v1alpha/fileobject/data-objects/<encoded>`; `None` otherwise.
#[allow(dead_code)]
fn decode_data_objects_prefix_from_path(path: &str) -> Option<String> {
    const PREFIX: &str = "/v1alpha/fileobject/data-objects/";
    let encoded = path.strip_prefix(PREFIX)?;
    let decoded = percent_decode_str(encoded).decode_utf8().ok()?;
    Some(decoded.into_owned())
}

/// Bundle returned by [`setup_storage_api_checkpoint_harness`]. Owns every
/// piece of state a storage-api-backed recovery E2E test needs to drive a
/// write → checkpoint → wipe → recover cycle end-to-end through the
/// `StorageApiCheckpointStore` adapter.
///
/// The `MockServer` handle is exposed so tests can inspect
/// `received_requests()` post-recovery to assert the over-the-wire surface
/// (e.g., "exactly one GET on the OMPF path"). The `TempDir` backing the
/// `ServiceTokenProvider`'s token file is kept alive through this struct so
/// the file survives until the harness is dropped (the provider's polling
/// task tolerates a missing file silently per SR-6.2, but tearing the dir
/// down while the test future is still running is racy and pointless). The
/// `Arc<ServiceTokenProvider>` handle and the [`Self::token_path`] it reads
/// are exposed together so a test can rewrite the credential mid-test and
/// load the new document synchronously via `force_refresh()`, without
/// waiting on the polling task — see
/// `adapter_storage_api_401_refresh_test.rs`, which uses exactly that to age
/// the cached token out and drive the adapter's unauthenticated-retry arm.
#[allow(dead_code)]
pub struct StorageApiCheckpointHarness {
    /// Live Valkey connection + key schema shared across the service,
    /// the recovery hook, and the periodic checkpoint loop.
    pub tv: TestValkey,
    /// `Arc<dyn CheckpointStore>` wrapping a `StorageApiCheckpointStore`
    /// pointed at [`Self::mock_server`]. Shared with the `RecoveryHook` so
    /// the hook reads the same generations the loop writes.
    pub store: Arc<dyn CheckpointStore>,
    /// Live `LiveEditService` wired against the real (enabled)
    /// `RecoveryHook` AND with `service_token_provider = Some(...)` so the
    /// outbound storage-api auth path is exercised end-to-end. Tests pass
    /// this directly to `handle_create_layer`, `handle_update_layer`,
    /// `handle_stat_layer`, and `handle_subscribe_read_layer`.
    pub svc: LiveEditService,
    /// Live `SharedMetrics` shared between the service and the loop —
    /// recovery-attempt counter assertions read this Arc.
    pub metrics: SharedMetrics,
    /// `JoinHandle` for the periodic checkpoint loop spawned by
    /// [`spawn_checkpoint_loop`]. Tests that need a clean shutdown should
    /// `cancel.cancel()` and `handle.await` at the end.
    pub loop_handle: JoinHandle<()>,
    /// `CancellationToken` paired with [`Self::loop_handle`]. Fire it to
    /// drain the loop.
    pub cancel: CancellationToken,
    /// Live `wiremock::MockServer`. Exposed so tests can inspect
    /// `received_requests().await` to assert the over-the-wire surface
    /// (PUT path families, GET path family, bearer header, etc.).
    pub mock_server: MockServer,
    /// `Arc<ServiceTokenProvider>` mirroring the one threaded into the
    /// adapter and the `LiveEditService`. Exposed so tests can read the cache
    /// through the same accessor each send site reads when it decides whether
    /// to attach a bearer (`cached_bearer_if_unexpired()`, plus
    /// `cached_token_expires_on()` to state *why*), and can drive
    /// `force_refresh()` directly after rewriting [`Self::token_path`] — the
    /// deterministic way to move the cached credential mid-test without racing
    /// the polling task.
    ///
    /// A cache read is a fixture statement, not an on-wire one: what the
    /// adapter classifies a 401 on is whether the rejected request actually
    /// carried an `Authorization` header. Where the request log is
    /// attributable, assert that header off [`Self::mock_server`]'s
    /// `received_requests()`; a harness test that mounts a catch-all cannot
    /// (the checkpoint loop's own traffic shares the log) and should instead
    /// bracket the call with cache reads either side, as
    /// `adapter_storage_api_401_refresh_test.rs`'s composed cases do.
    pub service_token_provider: Arc<ServiceTokenProvider>,
    /// Stable base path the adapter composes every `<base>/.live-edit/...`
    /// address under. Exposed so test assertions can construct the
    /// expected percent-encoded OMPF path tail without re-deriving the
    /// constant from `BASE_PATH_STORAGE_API`.
    pub base_path: String,
    /// `TempDir` backing the `ServiceTokenProvider`'s polling-task target
    /// file. Kept alive for the duration of the test. RAII-cleaned on
    /// drop. Tests address the file through [`Self::token_path`] rather than
    /// re-deriving it from this handle; the field exists so the `TempDir`
    /// survives until the harness itself is dropped. Field-scoped
    /// `allow(dead_code)` is correct here — the crate-level allow that
    /// previously covered the whole module was removed so unused helpers
    /// surface as lint errors.
    #[allow(dead_code)]
    pub token_dir: TempDir,
    /// Path of the sidecar-style token document inside [`Self::token_dir`],
    /// i.e. the file the harness's `ServiceTokenProvider` polls.
    ///
    /// Exposed so a test can rewrite the credential mid-run — see
    /// `adapter_storage_api_401_refresh_test.rs`, which writes an
    /// already-expired document here and then calls
    /// `service_token_provider.force_refresh()` to age the cache out
    /// deterministically. Carrying the path on the harness keeps the
    /// `access_token` basename a single fact owned by
    /// [`setup_storage_api_checkpoint_harness`] instead of a literal
    /// duplicated into every test that needs it.
    pub token_path: PathBuf,
    /// The adapter this harness was built against — always
    /// [`CheckpointAdapter::StorageApi`]. Recorded for the same reason
    /// [`CheckpointHarness::checkpoint_adapter`] is: it lets
    /// [`assert_harness_carries_object_id_counter_repair`] state the production
    /// wiring condition structurally (`main.rs` installs the b.b2j repair hook
    /// under exactly `!matches!(config.checkpoint_adapter, Disabled)`) rather
    /// than by convention.
    pub checkpoint_adapter: CheckpointAdapter,
}

/// Construct a fully-wired storage-api-backed recovery-E2E harness.
///
/// Defaults — every value below is set explicitly so a downstream config
/// drift surfaces as a compile error rather than a silent default flip:
///
/// - `LIVE_EDIT_CHECKPOINT_ENABLED=true`
/// - `LIVE_EDIT_CHECKPOINT_ADAPTER=StorageApi`
/// - `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT=<mock_server.uri()>`
/// - `LIVE_EDIT_CHECKPOINT_BASE_PATH=BASE_PATH_STORAGE_API`
/// - `LIVE_EDIT_CHECKPOINT_TICK_SECS=tick_secs` (caller-chosen; tests
///   typically pass `1` for fast convergence)
/// - `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS=0` and
///   `LIVE_EDIT_CHECKPOINT_QUIESCE_SECS=0` — the loop's skip-if-idle gate
///   would otherwise hold every just-written layer in skip-state for the
///   default 60 s interval and stall every `wait_for_checkpoint` deadline.
/// - `LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS=2`
/// - `LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS` /
///   `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS` at the production defaults
///   (5 s / 30 s) — long enough that the test's quick happy-path requests
///   never observe a transport timeout.
///
/// The harness wires a real `RecoveryHook` (NOT the disabled stub from
/// `build_service`) against the same `StorageApiCheckpointStore` the loop
/// is using, so the handler-level recovery branch can actually restore
/// from the generations the loop produces — and the outbound auth path is
/// exercised because `service_token_provider` is set to `Some(provider)`
/// on the `LiveEditService` rather than left as `None` (which is what the
/// LocalDisk [`setup_checkpoint_harness`] does — LocalDisk has no
/// outbound-auth surface).
///
/// Per-harness-call isolation: each invocation allocates a fresh
/// `MockServer`, a fresh `FakeStorageApiState` map, and a fresh `TempDir`.
/// Cross-harness state leakage is impossible by construction.
#[allow(dead_code)]
pub async fn setup_storage_api_checkpoint_harness(tick_secs: u64) -> StorageApiCheckpointHarness {
    let tv = TestValkey::setup().await;

    // ── 1. Wiremock + stateful fake-storage state ──────────────────────────
    let mock_server = MockServer::start().await;
    let state: FakeStorageApiState = Arc::new(Mutex::new(HashMap::new()));

    // ── 2. Default routes: PUT/GET/HEAD/DELETE on by-address, GET (LIST)
    //     on data-objects/. Mirrors the conformance harness's
    //     `mount_fake_storage_routes` in adapter_storage_api_test.rs. ────
    Mock::given(method("PUT"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(StorageApiPutHandler {
            state: Arc::clone(&state),
        })
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(StorageApiGetHandler {
            state: Arc::clone(&state),
        })
        .mount(&mock_server)
        .await;

    Mock::given(method("HEAD"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(StorageApiHeadHandler {
            state: Arc::clone(&state),
        })
        .mount(&mock_server)
        .await;

    Mock::given(method("DELETE"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(StorageApiDeleteHandler {
            state: Arc::clone(&state),
        })
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/data-objects/"))
        .respond_with(StorageApiListHandler {
            state: Arc::clone(&state),
        })
        .mount(&mock_server)
        .await;

    // ── 3. Tempfile-backed ServiceTokenProvider primed with a fresh token.
    //     The token file is atomically written via NamedTempFile::persist so
    //     a partial reader on the polling task never observes a torn file —
    //     same atomic-rename pattern as the production credential-
    //     provisioner sidecar. `force_refresh()` is awaited so the
    //     adapter's first outbound call sees a populated bearer cache. ──
    let token_dir = tempfile::tempdir().expect("tempdir for storage-api service-token");
    let token_path = token_dir.path().join("access_token");
    write_token_file_atomic(&token_path, STORAGE_API_TEST_BEARER);

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        STORAGE_API_TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        cancel.clone(),
    ));
    provider
        .force_refresh()
        .await
        .expect("initial force_refresh on a present, valid token file must succeed");

    // ── 4. StorageApiCheckpointStore over the mock server, the fixed
    //     base path, the primed provider, and realistic timeouts. ────────
    let endpoint_url = mock_server.uri();
    let adapter = StorageApiCheckpointStore::new(
        endpoint_url.clone(),
        PathBuf::from(BASE_PATH_STORAGE_API),
        Arc::clone(&provider),
        Duration::from_millis(STORAGE_API_CONNECT_TIMEOUT_MS),
        Duration::from_millis(STORAGE_API_REQUEST_TIMEOUT_MS),
    )
    .expect("StorageApiCheckpointStore::new must succeed on realistic inputs");
    let store: Arc<dyn CheckpointStore> = Arc::new(adapter);

    // ── 5. LayerStore + SessionStore. The storage-api adapter swap is
    //     orthogonal to the Valkey-backed layer/session state, so the
    //     allocation-path wiring must match the LocalDisk harness exactly:
    //     `.with_metrics(...)` (without which every repair-outcome cell is
    //     silently dropped and the parity guard below cannot see anything) and
    //     the b.b2j `.with_object_id_counter_repair(...)` hook, which `main.rs`
    //     Step 1b-iii.6 installs under precisely this harness's condition — a
    //     non-`Disabled` adapter. Both were missing here while this comment
    //     claimed parity, which meant every CreateLayer in the storage-api e2e
    //     took `allocate_object_id`'s UNHOOKED fallback: after a keyspace loss
    //     it re-establishes `le:counter:object_id` at 0 and re-mints ids the
    //     surviving checkpoint blobs still claim, i.e. b.b2j verbatim, on the
    //     one adapter where those blobs live outside the pod.
    //
    //     Deliberately still absent: `.with_checkpoint_writer(...)`. The
    //     storage-api harness has no session-checkpoint surface under test and
    //     no `wait_for_session_checkpoint_quiesced` consumer, so wiring a writer
    //     would add background PUT traffic to the very `received_requests()`
    //     stream `e2e_storage_api_recovery_test` asserts over. "Same wiring" is
    //     scoped to the allocation path.
    let storage_api_cfg = harness_config_storage_api(
        tick_secs,
        endpoint_url.clone(),
        BASE_PATH_STORAGE_API.to_string(),
    );
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_metrics(Arc::clone(&metrics))
            .with_object_id_counter_repair(object_id_counter_repair_for_harness(
                &tv,
                &store,
                &storage_api_cfg,
                &metrics,
            )),
    );
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    // ── 6. Real (enabled) RecoveryHook pointed at the storage-api store —
    //     load-bearing difference vs. `build_service`'s disabled stub.
    //
    //     The two collaborators are hoisted above the hook build and cloned
    //     into BOTH the hook and the service, the way `main.rs` clones one
    //     object into both. For the bootstrap that sharing is a wiring
    //     INVARIANT rather than a saving — `BOOTSTRAP_MAX_INFLIGHT` is a field
    //     ON the unit (`RecoveryHook::with_base_bootstrap`'s rustdoc in
    //     `src/checkpoint/recovery.rs`), so a second instance would be a second
    //     independent bound.
    //
    //     `TaskTracker`: this harness wires no `SessionCheckpointWriter`, so
    //     there is no shared tracker to join. The local one is held in a
    //     variable rather than constructed inline so a future consumer has
    //     something to drain; today nothing is ever spawned on it, because the
    //     SR-7.1 gate resolves `false` (this harness's `Config`, built from
    //     `harness_config_storage_api` over `common::test_config()`, pins
    //     `checkpoint_tail_enabled` off — NOT the binary default, which is
    //     `true` since b.3n1), so the untracked-spawn hazard cannot arise here.
    //     The loop config at step 8 pins the same value, for the reason spelled
    //     on `apply_tail_to_config`: the two consumers must agree.
    //
    //     Ext store: a REAL `ValkeyExternalValueStore` over this harness's own
    //     `TestValkey`, matching the LocalDisk harness's default and for the
    //     same reason (`default_ext_value_store_for_harness`). Deliberately NOT
    //     `.with_checkpoint_store(..)`: that arms the SR-3.2 fetch-on-miss
    //     fallback, whose adapter calls would land in the very
    //     `mock_server.received_requests()` stream `e2e_storage_api_recovery_test`
    //     asserts over — the same reason `.with_checkpoint_writer(...)` is
    //     absent from the `LayerStore` above.
    let bootstrap_tracker = TaskTracker::new();
    let base_bootstrap =
        base_bootstrap_for_harness(&tv, &store, &storage_api_cfg, &bootstrap_tracker, &metrics);
    let ext_value_store = default_ext_value_store_for_harness(&tv, &metrics);
    let recovery = build_recovery_hook_for_e2e(
        Arc::clone(&layer_store),
        Arc::clone(&store),
        harness_config_storage_api(
            tick_secs,
            endpoint_url.clone(),
            BASE_PATH_STORAGE_API.to_string(),
        ),
        Arc::clone(&metrics),
        E2eRecoveryHookWiring {
            ext_value_store: Arc::clone(&ext_value_store),
            base_bootstrap: Some(Arc::clone(&base_bootstrap)),
        },
    );
    let session_recovery = build_session_recovery_hook_for_e2e(
        &tv,
        Arc::clone(&store),
        harness_config_storage_api(tick_secs, endpoint_url, BASE_PATH_STORAGE_API.to_string()),
        Arc::clone(&metrics),
    );

    // ── 7. LiveEditService wired with the recovery hook AND the service-
    //     token provider. Flipping `service_token_provider` to `Some(...)`
    //     is the load-bearing AC for this harness — the outbound auth
    //     path on the storage-api adapter is exercised end-to-end. ──────
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
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics: Arc::clone(&metrics),
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        // The SAME real unit the recovery hook holds — see the hoist comment at
        // step 6 for why one instance is an invariant, and for the tracker note.
        base_bootstrap,
        service_token_provider: Some(Arc::clone(&provider)),
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

    // ── 8. Spawn the leader-elected periodic checkpoint loop. Same
    //     interval=0 + quiesce=0 idle-gate disablement as the LocalDisk
    //     harness — the loop must immediately schedule every just-written
    //     layer so `wait_for_checkpoint` deadlines stay tight. ───────────
    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_secs(tick_secs.max(1)),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this harness.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(SERVICE_LABEL),
        // Pinned OFF to match this harness's own `Config` (see the step-6
        // tracker note). Tracking `TailCaptureConfig::default()` would put the
        // loop's tail path ON while the bootstrap gate stayed off — the exact
        // two-consumer disagreement `apply_tail_to_config` documents.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let loop_handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    StorageApiCheckpointHarness {
        tv,
        store,
        svc,
        metrics,
        loop_handle,
        cancel,
        mock_server,
        service_token_provider: provider,
        base_path: BASE_PATH_STORAGE_API.to_string(),
        token_dir,
        token_path,
        checkpoint_adapter: CheckpointAdapter::StorageApi,
    }
}

/// Build the `Config` the storage-api harness threads into the recovery
/// hook. Mirrors [`harness_config`] but flips the adapter selector to
/// `StorageApi` and populates the storage-api-specific fields.
///
/// The field names referenced here (`checkpoint_storage_api_endpoint`,
/// `checkpoint_base_path`, `checkpoint_connect_timeout_ms`,
/// `checkpoint_request_timeout_ms`) are the contract Task `t2.fgj.jd.dz`
/// (Subtask 1) added to `Config`. A downstream rename surfaces here as a
/// compile error rather than a silent drift.
#[allow(dead_code)]
fn harness_config_storage_api(tick_secs: u64, endpoint_url: String, base_path: String) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_enabled = true;
    cfg.checkpoint_adapter = live_edit::checkpoint::CheckpointAdapter::StorageApi;
    cfg.checkpoint_interval_secs = 0;
    cfg.checkpoint_quiesce_secs = 0;
    cfg.checkpoint_tick_secs = tick_secs.max(1);
    cfg.checkpoint_retain_generations = 2;
    cfg.checkpoint_storage_api_endpoint = Some(endpoint_url);
    cfg.checkpoint_base_path = Some(PathBuf::from(base_path));
    cfg.checkpoint_connect_timeout_ms = STORAGE_API_CONNECT_TIMEOUT_MS;
    cfg.checkpoint_request_timeout_ms = STORAGE_API_REQUEST_TIMEOUT_MS;
    cfg
}

/// Atomically write a fresh sidecar-shaped token JSON document to `path`.
///
/// Mirrors the production credential-provisioner sidecar's write contract
/// (write-temp + rename) so a partial reader on the polling task never
/// observes a torn file. Same atomic-rename pattern as
/// `tests/integration/service_token_rotation.rs:57-68` and the variant in
/// `adapter_storage_api_401_refresh_test.rs::atomic_write`.
///
/// `expires_on = now + 1 hour` is comfortably live, so the adapter's
/// `cached_bearer_if_unexpired()` serves it on every send site for the whole
/// life of any harness built on this helper. It is also comfortably clear of
/// the provider's 60 s refresh lead, which post-b.e49 is only a re-read
/// cadence — no consumer withholds on it — but staying clear keeps a harness
/// from silently drifting onto the near-expiry boundary that
/// `adapter_storage_api_401_refresh_test.rs` owns deliberately.
#[allow(dead_code)]
fn write_token_file_atomic(path: &std::path::Path, access_token: &str) {
    let expires_on = Utc::now() + chrono::Duration::hours(1);
    let body = format!(
        "{{\"expires_on\":\"{}\",\"access_token\":\"{}\"}}",
        expires_on.to_rfc3339(),
        access_token
    );
    let parent = path
        .parent()
        .expect("token file path must have a parent directory");
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .expect("NamedTempFile::new_in must succeed in test tempdir");
    use std::io::Write;
    tmp.write_all(body.as_bytes())
        .expect("write to tempfile must succeed");
    tmp.persist(path)
        .expect("persist (atomic rename) must succeed");
}

// ===========================================================================
// Task `t2.fgj.7v.o7` — End-to-end session-recovery shared helpers
//
// The three helpers below are the Task-7 additions consumed by every sibling
// Subtask (`.hr` / `.c6` / `.6f` / `.xt` / `.c4` / `.u6` / `.wz` / `.v9`).
// `session_recovery_metric` is the centralised label-cell reader; the
// existing per-file definitions in the Task 4/6 e2e modules are left in
// place (no retroactive refactor in this Subtask). `wait_for_recovery_metric`
// covers the asynchronous-landing surfaces (ListSessions ghost-member sweep)
// where the recovery decision is decoupled from the RPC return. `full_valkey_wipe`
// is the AC #2 wipe primitive — the parent Task body refers to it as
// `flush_namespace`, but `TestValkey` exposes no such method, so this helper
// supplies the SCAN + UNLINK across every `tv.prefix()*` key.
// ===========================================================================

/// Read the live label-cell counter value for the supplied `result` on the
/// `live_edit_checkpoint_recovery_attempts_total{result, scope="session",
/// service=SERVICE_LABEL}` metric.
///
/// `scope="session"` (NOT `"layer"`) is the load-bearing difference vs. the
/// layer-recovery-metric reads scattered across Epic 1's e2e modules:
/// session-recovery and layer-recovery tick disjoint metric cells, so the
/// Task-7 e2e tests must inspect the session scope to assert their
/// invariants. Task 6 extended the metric with the `scope` label and
/// harmonized the result vocabulary to
/// `restored | not_recoverable | already_present | error` — passing any
/// other string yields a zero counter (the pre-init loop in
/// `Metrics::new` only seeds those four `result` values × two `scope`
/// values).
///
/// `harness` is taken as `&CheckpointHarness` rather than
/// `&SharedMetrics` so siblings can write `session_recovery_metric(&harness,
/// "restored")` without re-deriving the `metrics` accessor at every call
/// site — the three Task 4/6 per-file definitions consistently pass the
/// `SharedMetrics` directly; this new contract centralises the harness
/// pattern that the Task-7 modules will use uniformly.
///
/// `#[allow(dead_code)]`: this foundation Subtask lands the contract; the
/// eight sibling Subtasks (`t3.fgj.7v.o7.{hr,c6,6f,xt,c4,u6,wz,v9}`) land
/// the per-AC bodies that consume it.
#[allow(dead_code)]
pub fn session_recovery_metric(harness: &CheckpointHarness, result: &str) -> u64 {
    recovery_attempts_cell(&harness.metrics, result, "session")
}

/// [`session_recovery_metric`]'s `scope="layer"` sibling: read
/// `live_edit_checkpoint_recovery_attempts_total{result, scope="layer",
/// service=SERVICE_LABEL}` off a harness.
///
/// `result` is one of `restored | not_recoverable | already_present | error`
/// (the four-value grid `Metrics::new` seeds); any other string reads `0`.
/// Use as a BEFORE/AFTER delta, never as an absolute — the grid is
/// zero-seeded, so "this cell is 0" is equally satisfied by a recovery path
/// that never ran.
///
/// # What this unifies, and what it deliberately does not
///
/// `recovery_metric` exists as **nine** file-local copies across this test
/// tree, over **three** different first-parameter types: some take
/// `&CheckpointHarness`, some take `&SharedMetrics`, and the StorageApi
/// suite takes its own harness type. This wrapper unifies only the
/// **harness-typed subset** — it is a `&CheckpointHarness` reader and nothing
/// else. The `&SharedMetrics`-typed copies and the
/// [`StorageApiCheckpointHarness`]-typed one are out of its reach by
/// signature, so "nine copies" does not become "zero" by adding this.
///
/// The existing copies are also deliberately **not** retro-fitted onto it:
/// every `e2e_*_recovery_test.rs` in this module is the base-only regression
/// fence for Epic `t1.dfh.yj`, and an edit to one of them is an escalation
/// signal rather than a cleanup. New files use this; old files stay as they
/// are.
#[allow(dead_code)]
pub fn layer_recovery_metric(harness: &CheckpointHarness, result: &str) -> u64 {
    recovery_attempts_cell(&harness.metrics, result, "layer")
}

/// Read a `live_edit_checkpoint_recovery_attempts_total{result,scope,service}`
/// cell from the rendered scrape exposition. The `result × scope × service`
/// grid is pre-initialised in `Metrics::new`, so every cell renders from boot;
/// an absent series reads as zero.
fn recovery_attempts_cell(metrics: &SharedMetrics, result: &str, scope: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", scope),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Live value of one `reason` cell of
/// `live_edit_checkpoint_recovery_stream_unresolved_total` for this harness's
/// service label (b.4et).
///
/// Each harness owns its own metrics registry, so the reading is scoped to the
/// calling test. Deliberately a SEPARATE counter from
/// `live_edit_checkpoint_recovery_stream_refused_total`: `..._refused_total`
/// counts a restore that LANDED after verifiably throwing a stream away, this
/// one counts a restore that could not establish what the stream was. Reading
/// one when you mean the other is the exact confusion the two-counter split
/// exists to prevent.
///
/// Shared rather than per-file: `e2e_object_id_preservation_test.rs` (the arm
/// table) and `restore_fault_injection_matrix_test.rs` (the wrote-nothing /
/// converges-on-retry claims) both assert on this family, and the two copies
/// were byte-identical.
#[allow(dead_code)]
pub fn stream_unresolved_metric(harness: &CheckpointHarness, reason: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_checkpoint_recovery_stream_unresolved_total",
        &[("reason", reason), ("service", SERVICE_LABEL)],
    )
    .unwrap_or(0.0) as u64
}

/// Snapshot EVERY `reason` cell of the unresolved family, for a before/after
/// delta.
///
/// The whole pinned vocabulary, never one cell: an arm that ticks the WRONG
/// reason is a different defect from one that does not tick at all — they send
/// an operator at different remediations, and two of the three reasons end the
/// restore while the third does not — and only a full sweep tells them apart.
#[allow(dead_code)]
pub fn stream_unresolved_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    live_edit::observability::metrics::CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS
        .iter()
        .map(|reason| (*reason, stream_unresolved_metric(harness, reason)))
        .collect()
}

/// Assert the per-`reason` deltas since `before`. Reasons absent from
/// `expected` must be FLAT — that is what makes "this arm reported EXACTLY
/// `probe_unreadable`, and did not also report a wrong-typed key" assertable.
#[allow(dead_code)]
pub fn assert_stream_unresolved_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (reason, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == reason)
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let got = stream_unresolved_metric(harness, reason) - pre;
        assert_eq!(
            got, want,
            "{context}: live_edit_checkpoint_recovery_stream_unresolved_total \
             reason={reason:?} MUST tick {want} time(s), got {got}. Each reason names a \
             different operator situation and a different remediation, so a tick on the wrong \
             one is its own defect"
        );
    }
}

/// Live value of one `live_edit_checkpoint_restore_fenced_total{stage, service}`
/// cell (b.h4m).
///
/// Panics on an absent series rather than reading it as zero: the whole `stage`
/// vocabulary is pre-initialised in `Metrics::new` (pinned by
/// `observability_metrics_test.rs`), so absence means the cell was renamed or
/// dropped — a wiring fault, not a count.
///
/// Shared rather than per-file for the same reason as
/// [`stream_unresolved_metric`]: `restore_write_fence_test.rs` (stages
/// `presence_probe` / `object_write`), `restore_fault_injection_matrix_test.rs`
/// (the only place in the suite that can tick `residue_wipe`, the middle fence)
/// and `e2e_object_id_preservation_test.rs` (test 23's `presence_probe` skip) all
/// read this family, and the copies were byte-identical down to the panic text.
/// Between them they cover all three stages; one reader is what keeps their
/// absent-series discipline from drifting apart. Callers normally want the
/// whole-vocabulary [`restore_fenced_cells`] snapshot rather than this per-cell
/// read.
///
/// Use as a BEFORE/AFTER delta, never as an absolute — every cell is zero-seeded,
/// so "this cell is 0" is equally satisfied by a fence that never fired.
#[allow(dead_code)]
pub fn restore_fenced_metric(harness: &CheckpointHarness, stage: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_checkpoint_restore_fenced_total",
        &[("stage", stage), ("service", SERVICE_LABEL)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_restore_fenced_total{{stage=\"{stage}\"}} must be \
             pre-initialised; an absent series means the cell was renamed or dropped, not that it \
             read zero"
        )
    }) as u64
}

/// Snapshot EVERY `stage` cell of the restore-fenced family, for a before/after
/// delta. The `restore_fenced_metric` sibling of [`stream_unresolved_cells`].
///
/// The whole pinned vocabulary, never one cell: the three stages say HOW LATE the
/// peer was noticed, so an arm that ticks the WRONG stage is a different incident
/// from one that does not tick at all, and only a full sweep tells them apart.
///
/// Shared rather than per-file: `restore_write_fence_test.rs` (stages
/// `presence_probe` / `object_write`), `e2e_object_id_preservation_test.rs` (test
/// 23's `presence_probe` skip) and `restore_fault_injection_matrix_test.rs` (the
/// suite's only producer of the middle stage, `residue_wipe`) all take this
/// snapshot, and the three copies were the same four lines.
#[allow(dead_code)]
pub fn restore_fenced_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    live_edit::observability::metrics::CHECKPOINT_RESTORE_FENCED_STAGES
        .iter()
        .map(|stage| (*stage, restore_fenced_metric(harness, stage)))
        .collect()
}

/// Assert exactly the named stages moved by the given deltas, and every other
/// stage in the pinned vocabulary stayed flat. The `restore_fenced_metric`
/// sibling of [`assert_stream_unresolved_deltas`].
///
/// Whole-vocabulary rather than per-cell, so an arm cannot pass while ALSO
/// ticking a fence it should never have reached. `presence_probe` moving on an
/// arm that drives a later window would mean the peer's header landed before the
/// attempt started — i.e. the seam is not driving the race the arm claims;
/// `residue_wipe` moving on an arm that planted no residue would mean the wipe
/// took its `WATCH` path over an empty keyspace; `object_write` moving on an arm
/// that claims an earlier verdict would mean the attempt got past that verdict to
/// the materialization.
///
/// `before` MUST come from [`restore_fenced_cells`], which walks the whole
/// `CHECKPOINT_RESTORE_FENCED_STAGES` vocabulary — that is what makes the
/// `unwrap_or(0)` below a "this stage is not in the expected set" default rather
/// than an absent-series fold: presence of every cell is already guaranteed by
/// [`restore_fenced_metric`]'s panic-on-absent read inside the snapshot.
///
/// `context` identifies the arm, so the shared prose below can carry the general
/// claim while each call site keeps its own — the same split
/// [`assert_stream_unresolved_deltas`] uses.
#[allow(dead_code)]
pub fn assert_restore_fenced_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (stage, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == stage)
            .map(|(_, delta)| *delta)
            .unwrap_or(0);
        let got = restore_fenced_metric(harness, stage) - *pre;
        assert_eq!(
            got, want,
            "{context}: live_edit_checkpoint_restore_fenced_total{{stage=\"{stage}\"}} must move \
             by {want}, got {got}. The three stages say HOW LATE the peer was noticed, and that \
             distance is the operator's whole read on how tight the race was — a tick on the wrong \
             one reports a different incident"
        );
    }
}

/// Point the harness's `ValkeyClient` observability hooks at the harness's own
/// metrics registry.
///
/// [`setup_checkpoint_harness_with_options`] wires `SharedMetrics` into the
/// service, the loop and the hooks but does **not** call
/// `install_observability_hooks` on the client — and both
/// `live_edit_checkpoint_restore_fenced_total` and
/// `live_edit_checkpoint_restore_residue_wiped_total` are ticked through
/// `ValkeyClient::metrics_snapshot()`, which is `None` until it has run. Without
/// this call every assertion on those families (and on
/// `live_edit_storage_watch_voided_total{path="restore_residue_wipe"}`) would
/// read a permanent zero and pass whatever the restore did.
///
/// That the wiring is LIVE is proven inside the arms rather than assumed: every
/// caller ends on an arm asserting a `+1` on one of those counters, and a dead
/// handle fails that assertion — which is what keeps the "unchanged" assertions
/// on the sibling arms from being vacuous.
///
/// Shared by `restore_write_fence_test.rs`,
/// `restore_fault_injection_matrix_test.rs` and
/// `e2e_object_id_preservation_test.rs`; the copies were byte-identical modulo
/// the `Arc` import path.
#[allow(dead_code)]
pub async fn wire_client_metrics(harness: &CheckpointHarness) {
    harness
        .tv
        .client()
        .install_observability_hooks(Arc::clone(&harness.metrics))
        .await;
}

/// Live value of `live_edit_checkpoint_restore_residue_wiped_total{service}` —
/// the counter the step-3b residue wipe ticks only on its `Wiped` (committed
/// `EXEC`) arm — read off a bare registry.
///
/// Registry-typed rather than harness-typed so the ONE implementation reaches
/// both shapes of caller: the [`CheckpointHarness`] arms go through
/// [`residue_wiped_metric`] below, and `e2e_restore_over_surviving_residue_test.rs`
/// — which drives `write_object` against a bare `TestValkey` + registry, with no
/// harness anywhere — calls this directly. That file previously carried a fourth
/// copy purely because the shared helper's signature put it out of reach; the
/// split is what retires that exemption. Mirrors the
/// [`recovery_attempts_cell`] / [`layer_recovery_metric`] arrangement already in
/// this file.
///
/// Panic-on-absent, matching [`restore_fenced_metric`] and
/// [`read_restore_lineage_counter`]: the counter is zero-seeded at
/// `Metrics::new` (pinned by
/// `observability_metrics_test::checkpoint_restore_residue_wiped_counter_is_zero_seeded`),
/// so a `.unwrap_or(0)` here could only ever mask a renamed or dropped cell — and
/// it would mask it as the *unchanged* reading that several arms assert, which is
/// the quiet green this discipline exists to prevent.
#[allow(dead_code)]
pub fn residue_wiped_cell(metrics: &SharedMetrics) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_restore_residue_wiped_total",
        &[("service", SERVICE_LABEL)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_restore_residue_wiped_total{{service={SERVICE_LABEL:?}}} is \
             ABSENT. The counter is zero-seeded at Metrics::new, so an absent series means it was \
             renamed or its label set changed. Deliberately not folded to 0 — that would render a \
             renamed metric as the `unchanged` reading the fenced arms assert"
        )
    }) as u64
}

/// [`residue_wiped_cell`] off a harness — the form
/// `restore_write_fence_test.rs` (FLAT on the total-loss arms, `+1` on the
/// residue arm) and `restore_fault_injection_matrix_test.rs` (whose converging
/// retries assert the same `+1`) use.
#[allow(dead_code)]
pub fn residue_wiped_metric(harness: &CheckpointHarness) -> u64 {
    residue_wiped_cell(&harness.metrics)
}

/// How many of `keys` currently exist.
///
/// A single `EXISTS` over the whole list rather than one round trip per key: the
/// callers use it to state "both planted residue keys are still there" / "the
/// loser's per-record hash is gone", and a count is the sharper form of that
/// claim than a fold of booleans. Shared by `restore_write_fence_test.rs` and
/// `restore_fault_injection_matrix_test.rs`, where the two copies were
/// byte-identical.
#[allow(dead_code)]
pub async fn count_existing(harness: &CheckpointHarness, keys: Vec<String>) -> u64 {
    harness
        .tv
        .client()
        .pool
        .exists(keys)
        .await
        .expect("EXISTS must succeed")
}

/// Attempts the b.4et stream adjudication makes at each Valkey round-trip —
/// the `max_attempts` of `RECOVERY_STREAM_PROBE_RETRY` in
/// `src/checkpoint/recovery.rs:2743`.
///
/// # Why a mirror and not an import
///
/// The production constant is private to that module, so no test can name it.
/// That is deliberate on both sides: a test that re-derived the number *from
/// the module under test* could not detect a change to it, which is the whole
/// point of asserting an attempt count. The mirror is what makes a production
/// edit land as a red test rather than as a silently-adjusted expectation.
///
/// # Why ONE mirror
///
/// It was three, under three different names
/// (`RECOVERY_STREAM_PROBE_ATTEMPTS` and `RECOVERY_STREAM_ADJUDICATION_BUDGET`
/// in `e2e_object_id_preservation_test.rs`, `B4ET_PROBE_ATTEMPTS` in
/// `restore_fault_injection_matrix_test.rs`). Three copies of one production
/// fact is three places to update and two places to forget, and the differing
/// names hid that they were the same fact. One mirror per production constant,
/// here, is the invariant.
///
/// # What pins the relationship
///
/// `recovery.rs`'s own
/// `the_stream_adjudication_budgets_stay_within_their_documented_bounds` unit
/// test — it derives the retry's whole backoff budget from the real constants
/// and asserts it stays negligible against the deadline, and asserts the
/// deadline against `2 ×` the default `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`
/// rather than against a literal. This mirror carries the digit; that test
/// carries the reasoning.
///
/// Not feature-gated (only its consumers are), so `#[allow(dead_code)]` covers
/// the feature shapes in which no test reads it.
#[allow(dead_code)]
pub const RECOVERY_STREAM_PROBE_ATTEMPTS: usize = 3;

/// The overall deadline the b.4et stream adjudication runs under — mirror of
/// the private `RECOVERY_STREAM_ADJUDICATION_BUDGET` in
/// `src/checkpoint/recovery.rs`.
///
/// Same mirror-not-import reasoning as [`RECOVERY_STREAM_PROBE_ATTEMPTS`]
/// above, and the same pinning unit test — which additionally holds
/// `budget + grace` under the 11 s ceiling SDD §22.5 step 6 records.
#[allow(dead_code)]
pub const RECOVERY_STREAM_ADJUDICATION_BUDGET: Duration = Duration::from_secs(10);

/// Bounded-poll [`session_recovery_metric`] (or its `scope="layer"`
/// equivalent, controlled by `scope`) until the cell equals `expected`,
/// panicking with a diagnostic on deadline expiry.
///
/// Recovery may land between RPC completion and metric assertion on the
/// asynchronous-landing surfaces — for instance, the AC #7 ghost-member
/// SREM in `ListSessions`'s opportunistic-cleanup branch fires through the
/// session-recovery hook on its own schedule, decoupled from the RPC
/// return. Per `docs-internal/test-writing-guide.md` `## Synchronization Patterns`,
/// the test pattern is "bounded poll on the observable, not a fixed
/// sleep" — this helper is the canonical implementation for the metric
/// observation surface.
///
/// `scope` should be `"session"` for session-recovery assertions and
/// `"layer"` for the layer-recovery half of AC #2's two-hop chain (which
/// also lands asynchronously through the per-handler recovery branch).
/// Any other value yields a zero cell (see pre-init contract above) and
/// the helper will always panic at the deadline.
///
/// `#[allow(dead_code)]`: this foundation Subtask lands the contract; the
/// eight sibling Subtasks land the per-AC bodies that consume it.
#[allow(dead_code)]
pub async fn wait_for_recovery_metric(
    harness: &CheckpointHarness,
    scope: &str,
    result: &str,
    expected: u64,
    deadline_secs: u64,
) {
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(50);
    loop {
        let observed = recovery_attempts_cell(&harness.metrics, result, scope);
        if observed == expected {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_recovery_metric timed out after {deadline_secs}s waiting for \
                 cell scope={scope} result={result} to equal {expected} (last observed \
                 {observed}). Common causes: the recovery branch never fired (wiring \
                 regression), the latch already-fired in an earlier RPC and is not \
                 re-incrementing this attempt, or the metric was read under the wrong \
                 scope label."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// SCAN + UNLINK every key under `tv.prefix()*`, simulating the full
/// Valkey-wipe data-loss surface AC #2 / AC #5 / the cluster gate are
/// designed to defend against.
///
/// `wipe_session_state` and `wipe_layer_state` cover the per-session and
/// per-object slices respectively; this helper is the union — every key
/// under the per-test prefix is removed, including the path-index
/// (`le:path:{hex(base_path)}` + `:names`), the per-object live-update
/// stream, the external-value store, GC bookkeeping, and any leader-lock
/// or counter keys. Cluster-aware via the same `is_clustered()` branch the
/// per-slice wipes use.
///
/// Replaces the parent Task body's `flush_namespace` reference — no such
/// method exists on `TestValkey`. Mirrors `scan_and_unlink` in
/// `src/storage/test_support.rs` shape-wise but is scoped to the test
/// prefix (not the global Valkey keyspace), so concurrent integration
/// tests against the same Valkey instance remain isolated.
///
/// `#[allow(dead_code)]`: this foundation Subtask lands the contract; the
/// eight sibling Subtasks land the per-AC bodies that consume it.
#[allow(dead_code)]
pub async fn full_valkey_wipe(tv: &TestValkey) {
    let pattern = format!("{}*", tv.key_schema().prefix());
    let client = tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
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
    // In cluster mode, group keys by slot — a multi-key UNLINK across slots
    // surfaces as CROSSSLOT. The per-slice wipe helpers above don't need
    // this discipline because their `{hash_tag}` constrains every match to
    // one slot; full_valkey_wipe crosses every slot the prefix touches.
    if client.is_clustered() {
        let mut by_slot: HashMap<u16, Vec<fred::types::Key>> = HashMap::new();
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

// ===========================================================================
// Cross-file fault-injector serialization
//
// `APPLY_STEP_FAULT_INJECTORS` (defined in `src/checkpoint/session.rs`) is a
// `OnceLock<Mutex<HashMap<ApplyStep, VecDeque<CheckpointError>>>>` shared
// across every test in the `tests/integration` binary. Multiple e2e test
// files install / consume entries on the same map; without cross-file
// serialization, one file's `set_apply_step_fault_injector_for_test` install
// can be popped by another file's `apply_session_checkpoint` call —
// observable as the OTHER test seeing an unexpected `Err` AND this test
// seeing an unexpected `Ok` (the fault never fired).
//
// `cargo test` parallelises tests within one binary by default, so a
// file-local `Mutex` is insufficient: tests in files A and B will still
// race even if each file serializes its own scenarios. This single global
// `Mutex` is the cross-file gate — every test body that touches the
// injector map MUST acquire it for the duration of the body.
//
// Files currently consuming this gate:
// - `e2e_session_recovery_atomicity_test.rs` (AC #3 / Step B) — producer:
//   actively installs fault-injection scenarios that mutate the global
//   injector map.
// - `e2e_session_recovery_path_index_atomicity_test.rs` (AC #4 / Step A) —
//   producer: same role as the atomicity_test producer above.
// - `e2e_session_recovery_per_handler_test.rs` — DEFENSIVE consumer: this
//   is a non-fault-injection test that nonetheless acquires the gate to
//   serialize itself against the two producers above. The per-handler test
//   shares the global per-process fault-injector / handler-singleton queue
//   state, so without this defensive acquire the producers' map mutations
//   could interleave with this test's per-handler assertions even though
//   the per-handler test itself never installs a fault. Role-wise this is
//   a serialization barrier, NOT a Step-A / Step-B producer.
//
// `std::sync::Mutex` (NOT `tokio::sync::Mutex`) is intentional — guards
// are held across `.await` points, but a single test holds the guard
// throughout its body (no contended-async-await pattern) and the runtime
// cost of acquiring an uncontended `std::sync::Mutex` is negligible.
// Poisoning would only happen if a previous test panicked while holding
// the lock; that test's failure would already cause the run to fail, so
// [`lock_fault_injector_serial`] recovers the inner guard so observability
// surfaces the test that actually broke.
//
// Callers should use [`lock_fault_injector_serial`] (NOT the raw
// `APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL.lock()` call) so the
// poison-recovery contract is uniform across consumers.
#[allow(dead_code)]
pub static APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL: Mutex<()> = Mutex::new(());

/// Acquire the cross-file fault-injector serialization guard for the
/// duration of the caller's stack frame. Drops at end-of-scope. Recovers
/// from a poisoned inner guard so a previous panic doesn't cascade and
/// obscure the actual failure on the current test.
///
/// `#[allow(dead_code)]` because not every test in this binary consumes
/// the gate — see the "Files currently consuming this gate" bullet list
/// on [`APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL`] above for the full
/// producer / defensive-consumer breakdown.
#[allow(dead_code)]
pub fn lock_fault_injector_serial() -> std::sync::MutexGuard<'static, ()> {
    APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Cross-file serialization for the PUBLISH fault-injector seam (b.443).
//
// `PUBLISH_FAULT_INJECTOR` (defined in `src/checkpoint/session_writer.rs`) is
// a process-global `VecDeque<PublishFault>` popped by EVERY `publish_stamp`
// call in the process — the publish path is a free function, so the seam
// cannot be instance-bound. Any test that installs a fault therefore has to
// serialize itself against peers that also dispatch session checkpoints, or
// its fault is popped by the wrong operation.
//
// Deliberately a SEPARATE mutex from [`APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL`]
// rather than a reuse of it: the two seams are on unrelated code paths (the
// apply/restore side vs. the write/publish side) and folding them onto one
// gate would serialize two disjoint sets of slow Valkey-hitting tests against
// each other for no correctness gain.
//
// **This gate is necessary but not sufficient under `cargo test`.** It
// serializes opt-in consumers against each other; it cannot stop an unrelated
// concurrent test in the same binary from dispatching a session checkpoint
// and popping someone's fault. `cargo nextest` is canonical here (CLAUDE.md
// `## Build Commands`) precisely because its process-per-test isolation makes
// the whole class moot.
#[allow(dead_code)]
pub static PUBLISH_FAULT_INJECTOR_TEST_SERIAL: Mutex<()> = Mutex::new(());

/// Acquire the cross-file publish-fault-injector serialization guard for the
/// duration of the caller's stack frame. Recovers from a poisoned inner guard
/// so a previous panic doesn't cascade and obscure the actual failure.
///
/// Consumed by `lifecycle_session_publish_watermark_test.rs`, the only file
/// that installs publish faults today.
#[allow(dead_code)]
pub fn lock_publish_fault_injector_serial() -> std::sync::MutexGuard<'static, ()> {
    PUBLISH_FAULT_INJECTOR_TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Poll [`CheckpointStore::list_layer_checkpoints`] until a **base**
/// generation lands for `object_id`, returning the newest such handle on
/// success or panicking with a diagnostic on deadline expiry.
///
/// Caller passes the deadline in seconds — sibling tests typically use
/// `10`, which is comfortably above the worst-case CI `tick=1s` schedule.
///
/// The poll cadence (50 ms) is intentionally tighter than the loop's
/// tick so we don't add observation jitter on top of the loop's own
/// scheduling.
pub async fn wait_for_checkpoint(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    deadline_secs: u64,
) -> CheckpointHandle {
    // `min_snap_seq = 0` accepts the first base that lands — correct for
    // callers that create a layer and immediately wait (the first generation IS
    // the full state). Callers that apply updates AFTER create and then wipe must
    // use `wait_for_checkpoint_at_least` instead — see its doc-comment for the
    // race it closes.
    wait_for_checkpoint_at_least(store, object_id, 0, deadline_secs).await
}

/// Like [`wait_for_checkpoint`], but blocks until a **base** generation whose
/// `snap_seq` is **at least `min_snap_seq`** has landed (returning the newest
/// such handle).
///
/// Closes a latent flake in recovery tests that apply updates *after*
/// `CreateLayer`: the periodic checkpoint loop (tick cadence ~1 s) can fire
/// between the create and the updates, landing an early generation that captures
/// only the empty post-create state (`snap_seq == 1`, zero sections). Plain
/// `wait_for_checkpoint` returns on that early generation, so a subsequent
/// wipe + recovery restores the empty checkpoint and the post-recovery
/// assertions see no sections. Under light load the updates usually beat the
/// first tick and the bug stays hidden; under heavy parallel load it surfaces as
/// an intermittent "expected N sections, got {}" failure.
///
/// Passing `min_snap_seq = <layer seq after all updates>` makes the wait
/// deterministic: it only returns once the loop has checkpointed a generation
/// that includes every applied update.
///
/// # Base-only selection
///
/// The predicate is `kind == CheckpointKind::Base && snap_seq >= min_snap_seq`
/// over the store's newest-first listing. Base-only is the DEFAULT because it is
/// what an un-annotated caller means: every caller of this helper and of
/// [`wait_for_checkpoint`] goes on to wipe Valkey and expect the returned handle
/// to restore a full layer state, and only a base carries one. A tail is a
/// stream-replay envelope (`LETL` magic) that the reset-only decoder rejects
/// `Permanent`, so handing one back would surface as a decode/restore error in
/// whichever file did the restoring rather than as a selection bug here.
///
/// With tail capture landed a hot lineage's NEWEST generation is routinely a
/// tail, so this is load-bearing rather than theoretical. Tests that genuinely
/// want the newest generation whatever its kind call
/// [`wait_for_any_kind_checkpoint_at_least`] and say so at the call site.
pub async fn wait_for_checkpoint_at_least(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    min_snap_seq: u64,
    deadline_secs: u64,
) -> CheckpointHandle {
    wait_for_generation(
        store,
        object_id,
        min_snap_seq,
        deadline_secs,
        Some(CheckpointKind::Base),
    )
    .await
}

/// [`wait_for_checkpoint_at_least`] with **no kind predicate**: returns the
/// newest generation at or above `min_snap_seq`, base or tail.
///
/// For tests whose subject IS the lineage listing — "a tail landed", "the newest
/// generation advanced" — rather than tests that restore from what they get
/// back. If the returned handle is going to be fed to a restore, this is the
/// wrong helper: use the base-only default.
#[allow(dead_code)]
pub async fn wait_for_any_kind_checkpoint_at_least(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    min_snap_seq: u64,
    deadline_secs: u64,
) -> CheckpointHandle {
    wait_for_generation(store, object_id, min_snap_seq, deadline_secs, None).await
}

/// The shared poll body behind the three public waiters. `want_kind = None`
/// accepts any kind.
async fn wait_for_generation(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    min_snap_seq: u64,
    deadline_secs: u64,
    want_kind: Option<CheckpointKind>,
) -> CheckpointHandle {
    let effective = Duration::from_secs(deadline_secs).max(LOAD_TOLERANT_CHECKPOINT_FLOOR);
    let deadline = Instant::now() + effective;
    let poll_interval = Duration::from_millis(50);
    let wanted = match want_kind {
        Some(kind) => format!("a {kind:?} generation"),
        None => "a generation of any kind".to_string(),
    };
    // Assigned on every non-diverging path through the match below (the `Err`
    // arm panics), so the deadline diagnostic always reads a listing the poll
    // actually observed rather than a placeholder.
    let mut last_listing: Vec<CheckpointHandle>;
    loop {
        match store.list_layer_checkpoints(object_id).await {
            Ok(handles) => {
                // `list_layer_checkpoints` returns newest-first; the first handle
                // matching both the kind predicate and the floor is the newest
                // qualifying generation.
                if let Some(handle) = handles
                    .iter()
                    .find(|h| want_kind.is_none_or(|k| h.kind == k) && h.snap_seq >= min_snap_seq)
                {
                    return handle.clone();
                }
                last_listing = handles;
            }
            Err(err) => {
                panic!(
                    "list_layer_checkpoints failed during wait_for_generation \
                     (object_id={object_id}, want_kind={want_kind:?}, \
                     min_snap_seq={min_snap_seq}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "timed out after {effective:?} (caller asked {deadline_secs}s, clamped to the \
                 load-tolerant floor of {LOAD_TOLERANT_CHECKPOINT_FLOOR:?}) waiting for \
                 object_id={object_id} to reach {wanted} at snap_seq>={min_snap_seq}.\n\
                 {}\n\
                 Last listing (newest-first): {last_listing:?}",
                deadline_diagnosis(&last_listing, min_snap_seq, want_kind)
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Turn the final listing into the sentence a reader of the panic needs.
///
/// The three arms are deliberately distinct because they have three different
/// causes and only the message tells them apart. In particular an EMPTY listing
/// and a TAILS-ONLY listing are indistinguishable downstream: a tails-only store
/// correctly reports `not_recoverable` on restore, which is exactly what a
/// store holding nothing produces AND exactly what a broken kind predicate would
/// produce (the distinction `recovery_mixed_kind_selection_test.rs`'s prune
/// fence documents at length).
fn deadline_diagnosis(
    listing: &[CheckpointHandle],
    min_snap_seq: u64,
    want_kind: Option<CheckpointKind>,
) -> String {
    if listing.is_empty() {
        return "DIAGNOSIS: the store holds NO generation at all for this object. The periodic \
                loop never landed one — check the leader lock, the interval/quiesce gates, and \
                that the writer actually committed."
            .to_string();
    }
    let highest_of_any_kind = listing.iter().map(|h| h.snap_seq).max().unwrap_or(0);
    let matching_kind: Vec<&CheckpointHandle> = listing
        .iter()
        .filter(|h| want_kind.is_none_or(|k| h.kind == k))
        .collect();
    if matching_kind.is_empty() {
        // Only reachable on the base-only waiters, and it is the arm this
        // diagnosis exists for.
        return format!(
            "DIAGNOSIS: generations EXIST but every one of the {} is a TAIL (highest to_seq: \
             {highest_of_any_kind}) — no base at all. A tail here means the capture path is \
             writing through the loop and this caller's premise (\"the generation I am about to \
             restore from is a full-state base\") no longer holds. This is reported as a wait \
             failure on purpose: a tails-only store restores as `not_recoverable`, which is also \
             what an empty store and a broken kind predicate produce, so letting it through would \
             surface as an unrelated decode/restore error somewhere downstream.",
            listing.len()
        );
    }
    let highest_matching = matching_kind.iter().map(|h| h.snap_seq).max().unwrap_or(0);
    format!(
        "DIAGNOSIS: {} generation(s) of the wanted kind exist but none reached the floor \
         (highest matching snap_seq: {highest_matching}; highest of any kind: \
         {highest_of_any_kind}; floor: {min_snap_seq}). The loop is running but has not yet \
         checkpointed a generation covering every commit the caller drove.",
        matching_kind.len()
    )
}

// ===========================================================================
// Streaming-framing helpers (b.dzd Phase 1)
//
// Lifted from the ~14 byte-identical per-file copies scattered across the
// `e2e_*_test.rs` / `recovery_hook_test.rs` / `restart_storm_harness.rs`
// modules in this directory. Same gRPC-frame shape as the existing
// `crate::grpc::helpers::streaming_create` / `make_create_*` / `make_request`
// helpers; the CreateLayer-side helpers are re-exported below so callers can
// `use super::helpers::*;` and pick up the whole set.
//
// `make_update_header` is pinned to the canonical 5-arg shape:
// `(session_id, layer_name, object_id, timestamp, client_id)`. Variants in
// the pre-refactor files used 3-arg / 4-arg / reordered shapes; per-file
// migration shimmed each call site to the canonical order.
// ===========================================================================

pub use crate::grpc::helpers::{make_create_data, make_create_header, streaming_create};

/// Wrap a request stream in a `tonic::Request` with a `BearerToken("test-token")`
/// extension (and, for an upload, its end-of-stream record), mirroring
/// [`crate::grpc::helpers::make_request`]. Renamed here so the streaming vs.
/// unary distinction is obvious at call sites (the original per-file
/// `make_token_request` collapsed both shapes onto one name).
#[allow(dead_code)]
pub fn make_streaming_token_request<T>(
    stream: impl crate::grpc::helpers::IntoHandlerRequest<T>,
) -> tonic::Request<tonic::codec::Streaming<T>> {
    crate::grpc::helpers::make_request(stream)
}

/// Wrap a unary protobuf message in a `tonic::Request` with a
/// `BearerToken("test-token")` extension. Counterpart to
/// [`make_streaming_token_request`] for the unary-RPC surface
/// (StatLayer / DeleteLayer / CreateSession / etc.).
#[allow(dead_code)]
pub fn make_unary_token_request<T>(message: T) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request
        .extensions_mut()
        .insert(live_edit::grpc::BearerToken::new("test-token"));
    request
}

/// Encode `UpdateLayerRequest` messages into a gRPC-framed body that ends with
/// END_STREAM, paired with its end-of-stream record. Sibling to
/// [`crate::grpc::helpers::streaming_create`] for the UpdateLayer-side of the
/// e2e recovery harness.
#[allow(dead_code)]
pub fn streaming_update(
    messages: Vec<live_edit::grpc::proto::UpdateLayerRequest>,
) -> crate::common::upload_stream::AttestedStreaming<live_edit::grpc::proto::UpdateLayerRequest> {
    crate::common::upload_stream::AttestedStreaming::from_messages(&messages)
}

/// Build an `UpdateLayerRequest` header message. Canonical 5-arg shape:
/// `(session_id, layer_name, object_id, timestamp, client_id)`.
#[allow(dead_code)]
pub fn make_update_header(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp: u64,
    client_id: &str,
) -> live_edit::grpc::proto::UpdateLayerRequest {
    use live_edit::grpc::proto::{update_layer_request, UpdateLayerHeader, UpdateLayerRequest};
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: timestamp,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
            },
        )),
    }
}

/// Build an `UpdateLayerRequest` data chunk message.
#[allow(dead_code)]
pub fn make_update_data(payload: Vec<u8>) -> live_edit::grpc::proto::UpdateLayerRequest {
    use live_edit::grpc::proto::{update_layer_request, UpdateLayerChunk, UpdateLayerRequest};
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(UpdateLayerChunk {
            data: payload,
        })),
    }
}

// ===========================================================================
// CountingGetSessionCheckpointStore (b.dzd Phase 2)
//
// Consolidated from the byte-identical per-file copies in
// `e2e_session_recovery_admission_test.rs` and
// `e2e_session_recovery_concurrent_handlers_test.rs`. The two files used
// slightly different field / accessor names (`get_session_checkpoint_call_count`
// vs `get_session_call_count`); the canonical name landing here is
// `get_session_calls`.
//
// `DelegatingCheckpointStore<T>` below is the generic delegation harness the
// other three counting-wrapper files (`recovery_admission_test.rs`,
// `checkpointer_leader_race_test.rs`, `checkpointer_counters_sync_test.rs`)
// compose with a single-method override to express their narrower instrumentation.
// ===========================================================================

/// `Arc<dyn CheckpointStore>` wrapper that counts `get_session_checkpoint`
/// invocations on the underlying adapter via an `AtomicUsize`. Every other
/// trait method delegates verbatim to the inner store.
///
/// The counter is the load-bearing instrumentation for the
/// "exactly one adapter fetch under N concurrent recoveries on the same
/// session_id" assertion in
/// `e2e_session_recovery_admission_test.rs` and
/// `e2e_session_recovery_concurrent_handlers_test.rs`. The `reset_count`
/// method lets a test zero the counter after pre-wipe setup (which itself
/// does `get_session_checkpoint` polls via `wait_for_session_checkpoint`)
/// so the post-wipe assertion measures only the recovery-driven fetches.
#[allow(dead_code)]
pub struct CountingGetSessionCheckpointStore {
    pub inner: Arc<dyn CheckpointStore>,
    get_session_calls: std::sync::atomic::AtomicUsize,
}

#[allow(dead_code)]
impl CountingGetSessionCheckpointStore {
    pub fn new(inner: Arc<dyn CheckpointStore>) -> Self {
        Self {
            inner,
            get_session_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn get_session_calls(&self) -> usize {
        self.get_session_calls
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn reset_count(&self) {
        self.get_session_calls
            .store(0, std::sync::atomic::Ordering::Release);
    }
}

#[async_trait::async_trait]
impl CheckpointStore for CountingGetSessionCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: live_edit::checkpoint::CheckpointMeta,
    ) -> Result<CheckpointHandle, live_edit::checkpoint::CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST: the trait
    // default is `NotImplemented`, so an omission here would report "no such
    // generation" for a generation the inner store holds.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: live_edit::checkpoint::CheckpointMeta,
    ) -> Result<CheckpointHandle, live_edit::checkpoint::CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        // Increment BEFORE the inner call so a concurrent caller racing
        // through here cannot mask the counter from observers. Pre- vs
        // post-inner-call ordering does not matter for correctness — the
        // counter only needs to capture whether the call was issued at
        // all, not its outcome.
        self.get_session_calls
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    // Forwarded per the trait's MUST: the default is `NotImplemented`, so an
    // omission here fails the DR restore Job's session listing loudly rather
    // than hiding the inner store's session backups.
    async fn list_session_checkpoint_ids(
        &self,
    ) -> Result<Vec<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(
        &self,
        hash_hex: &str,
        bytes: Bytes,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(
        &self,
        hash_hex: &str,
    ) -> Result<Option<Bytes>, live_edit::checkpoint::CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    // Forwarded per the trait's MUST, for the same reason as
    // `list_session_checkpoint_ids` above.
    async fn ext_blob_exists(
        &self,
        hash_hex: &str,
    ) -> Result<bool, live_edit::checkpoint::CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(
        &self,
        hash_hex: &str,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(
        &self,
        body: Bytes,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(
        &self,
    ) -> Result<Option<Bytes>, live_edit::checkpoint::CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(
        &self,
    ) -> Result<Vec<u64>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, live_edit::checkpoint::CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four methods below carry non-trivial trait defaults, so an
    // "every method delegates verbatim" wrapper that omits them does not
    // merely forget to instrument something — it silently SUBSTITUTES
    // behaviour: `Ok(empty)` for the per-handle sidecar read, `Ok(vec![])`
    // for the raw artifact enumeration, `Ok(0)` for temp reclamation, and
    // the local publish bound for the reconciliation grace. Nothing fails;
    // the b.9qd reconciliation pass just goes inert behind the wrapper and
    // every assertion about it passes vacuously.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, live_edit::checkpoint::CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> std::time::Duration {
        self.inner.publish_worst_case()
    }
}

/// Generic `CheckpointStore` delegation harness — every trait method forwards
/// verbatim to `self.inner.<method>(...).await`. File-local counting / recording
/// newtypes can wrap a `DelegatingCheckpointStore<T>` and override a single
/// method to express narrower instrumentation without re-typing every verbatim
/// trait-method body.
///
/// "Every method" is load-bearing and includes the ones with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). An omission there is
/// not an un-instrumented pass-through, it is a behaviour substitution that no
/// test fails on — see the comment at those methods. It also includes
/// `list_session_checkpoint_ids` and `ext_blob_exists`, whose trait defaults
/// answer `NotImplemented`: omitting those fails loudly rather than silently,
/// but a wrapper that drops them still breaks the DR restore Job behind it.
/// When the trait grows a method, it must be added here too.
///
/// Consumed by `recovery_admission_test.rs` (counts `list_layer_checkpoints`
/// — the call the admitted restorer issues unconditionally after the
/// under-mutex `obj:hdr` re-check, which is what preserves that fixture's 1:1
/// admitted-restorer-to-store-round-trip property),
/// `checkpointer_leader_race_test.rs` (records `creator_replica_id` strings),
/// and `checkpointer_counters_sync_test.rs` (records each counters-sidecar
/// PUT's body, result and preceding GET count). The per-file counting types
/// stay file-local — only the
/// verbatim-delegation skeleton is shared.
///
/// This type is a held FIELD, not a blanket impl: every consumer is its own
/// newtype with a full method-by-method `impl CheckpointStore` that forwards
/// into the field. Adding a method here therefore fixes nothing on its own —
/// each consumer newtype needs the same arm.
#[allow(dead_code)]
pub struct DelegatingCheckpointStore<T: CheckpointStore + ?Sized> {
    pub inner: Arc<T>,
}

#[allow(dead_code)]
impl<T: CheckpointStore + ?Sized> DelegatingCheckpointStore<T> {
    pub fn new(inner: Arc<T>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<T: CheckpointStore + ?Sized + Send + Sync> CheckpointStore for DelegatingCheckpointStore<T> {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: live_edit::checkpoint::CheckpointMeta,
    ) -> Result<CheckpointHandle, live_edit::checkpoint::CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST. Its trait
    // default returns `NotImplemented`, so omitting it here is not a missing
    // pass-through — it is a behaviour substitution that reports "no such
    // generation" for a generation the inner store holds, to the one caller
    // (restore selection / lineage gather) whose whole job is deciding what
    // exists. Note this skeleton is a held FIELD, not a blanket impl: each
    // consumer newtype re-types the trait method-by-method and must carry its
    // own forwarding arm too.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: live_edit::checkpoint::CheckpointMeta,
    ) -> Result<CheckpointHandle, live_edit::checkpoint::CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<live_edit::checkpoint::CheckpointBlob>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    // Forwarded per the trait's MUST: the default is `NotImplemented`, so an
    // omission here fails the DR restore Job's session listing loudly rather
    // than hiding the inner store's session backups.
    async fn list_session_checkpoint_ids(
        &self,
    ) -> Result<Vec<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(
        &self,
        hash_hex: &str,
        bytes: Bytes,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(
        &self,
        hash_hex: &str,
    ) -> Result<Option<Bytes>, live_edit::checkpoint::CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    // Forwarded per the trait's MUST, for the same reason as
    // `list_session_checkpoint_ids` above.
    async fn ext_blob_exists(
        &self,
        hash_hex: &str,
    ) -> Result<bool, live_edit::checkpoint::CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(
        &self,
        hash_hex: &str,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(
        &self,
        body: Bytes,
    ) -> Result<(), live_edit::checkpoint::CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(
        &self,
    ) -> Result<Option<Bytes>, live_edit::checkpoint::CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(
        &self,
    ) -> Result<Vec<u64>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, live_edit::checkpoint::CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four methods below carry non-trivial trait defaults, so an
    // "every method delegates verbatim" wrapper that omits them does not
    // merely forget to instrument something — it silently SUBSTITUTES
    // behaviour: `Ok(empty)` for the per-handle sidecar read, `Ok(vec![])`
    // for the raw artifact enumeration, `Ok(0)` for temp reclamation, and
    // the local publish bound for the reconciliation grace. Nothing fails;
    // the b.9qd reconciliation pass just goes inert behind the wrapper and
    // every assertion about it passes vacuously.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, live_edit::checkpoint::CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, live_edit::checkpoint::CheckpointError>
    {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, live_edit::checkpoint::CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> std::time::Duration {
        self.inner.publish_worst_case()
    }
}

// ===========================================================================
// Hand-put generation builders (Epic `t1.dfh.57`, Task 7)
//
// Lifted verbatim from `recovery_mixed_kind_selection_test.rs`, where they were
// file-local and bound to `&CheckpointHarness`. Parameterised on
// `&Arc<dyn CheckpointStore>` so the storage-side `checkpointer_*` suites — which
// hold a store but no harness — build the same fixtures rather than growing a
// second, subtly different copy. Epic 4 (`t1.dfh.yj`) builds its restore tests
// on these.
// ===========================================================================

/// `created_by` written into every hand-put [`CheckpointMeta`]. Non-empty is
/// load-bearing: `StatLayerResponse::created_by_user_name` is restored verbatim
/// from this field, and tests assert it survived.
pub const HAND_PUT_CREATED_BY: &str = "hand-put-generation-user";

/// `server_ts` written into every hand-put [`CheckpointMeta`]. Non-zero is
/// load-bearing for the same reason as [`HAND_PUT_CREATED_BY`] — apply collapses
/// `created_at` / `modified_at` / `last_server_ts` onto this one value.
pub const HAND_PUT_SERVER_TS: u64 = 1_700_000_000_000_000;

/// `written_at_unix_ms` for every hand-put [`CheckpointMeta`]. Provenance only;
/// nothing asserts on it.
pub const HAND_PUT_WRITTEN_AT_UNIX_MS: u64 = 1_700_000_000_000;

/// Hand-put a **base** generation at `snap_seq` into `store`.
///
/// The meta's `session_id` / `layer_name` MUST be the live layer mapping.
/// `apply_checkpoint`'s SR-2.4 identity guard resolves
/// `le:{sid}:layer:{hex(name)}` and refuses any blob whose identity does not
/// map back to `object_id`; a mismatched (or empty) identity collapses every
/// recovery outcome to `NotRecoverable`, which would silently mask whatever
/// selection assertion the caller is making.
///
/// Fields are enumerated rather than `..CheckpointMeta::default()`-ed so the
/// base's `kind` / `base_snap_seq` / `from_seq` triple sits literally beside
/// the tail's in [`put_tail_generation`], and so a new meta field surfaces
/// here as a compile error rather than a silent default.
#[allow(dead_code)]
pub async fn put_base_generation(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    session_id: &str,
    layer_name: &str,
    snap_seq: u64,
) -> CheckpointHandle {
    let meta = CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms: HAND_PUT_WRITTEN_AT_UNIX_MS,
        snap_seq: Some(snap_seq),
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: HAND_PUT_CREATED_BY.to_string(),
        server_ts: HAND_PUT_SERVER_TS,
        // Zero is the no-measurement sentinel, which folds `apply`'s guard
        // band back to `config.checkpoint_seq_safety_margin` alone. A caller
        // asserting an exact restored `seq` depends on that fold.
        observed_commit_span: 0,
        kind: CheckpointKind::Base,
        // A base carries no lineage: `validate_tail_put` only constrains the
        // tail branch, but SR-2.3's invariant is `Base ⇒ both None`.
        base_snap_seq: None,
        from_seq: None,
        // A hand-put base references no external values, so there is nothing
        // for an ext-refs sidecar to pin.
        sidecar_bytes: None,
    };
    store
        .put_layer_checkpoint(object_id, snap_seq, crate::common::empty_reset_ompf(), meta)
        .await
        .unwrap_or_else(|e| panic!("hand-put of base@{snap_seq} must succeed: {e}"))
}

/// Hand-put a **tail** generation covering `(from_seq, to_seq]` into `store`.
///
/// The payload is a real tail envelope from `encode_tail_envelope` — magic
/// `LETL`, one `Verbatim` record per covered sequence number. The encoder
/// cross-checks the header's claimed span against the records supplied, so a
/// fixture that claimed coverage it did not carry could not be built here at
/// all.
///
/// The record payloads are reset-OMPF bytes. They are opaque to the codec
/// (`tail_codec` never parses a record payload), which is fine for selection /
/// listing / chain-bound fixtures — but they are SYNTHETIC: nothing in them
/// corresponds to a commit that was ever applied to the layer. A test that
/// REPLAYS a tail wants [`drive_captured_tail`] instead.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub async fn put_tail_generation(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    session_id: &str,
    layer_name: &str,
    base_snap_seq: u64,
    from_seq: u64,
    to_seq: u64,
) -> CheckpointHandle {
    let header = TailHeader {
        object_id,
        base_snap_seq,
        from_seq,
        to_seq,
    };
    let records: Vec<TailRecord> = ((from_seq + 1)..=to_seq)
        .map(|sequence_num| TailRecord::Verbatim {
            sequence_num,
            server_ts: HAND_PUT_SERVER_TS as i64,
            ompf: crate::common::empty_reset_ompf(),
        })
        .collect();
    let envelope = encode_tail_envelope(&header, records)
        .unwrap_or_else(|e| panic!("tail envelope ({from_seq}, {to_seq}] must encode: {e}"));

    let meta = CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms: HAND_PUT_WRITTEN_AT_UNIX_MS,
        // A tail's ordering sequence is its `to_seq` — one number, one home,
        // shared with `CheckpointHandle::snap_seq` so both kinds order and
        // address identically.
        snap_seq: Some(to_seq),
        // Identity is carried on a tail's meta too. `CheckpointMeta`'s
        // lockstep block forbids an identity-empty tail meta precisely so a
        // future lineage restore never feeds one into the SR-2.4 guard.
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: HAND_PUT_CREATED_BY.to_string(),
        server_ts: HAND_PUT_SERVER_TS,
        observed_commit_span: 0,
        kind: CheckpointKind::Tail,
        // Both `Some` — `validate_tail_put` rejects a tail put with either
        // lineage field `None` as `Permanent`, and adapters MUST NOT infer.
        base_snap_seq: Some(base_snap_seq),
        from_seq: Some(from_seq),
        // MUST stay `None`: `validate_tail_put` rejects a tail carrying an
        // ext-refs sidecar (SR-2.4) — one written under a tail stem is never
        // reaped and would pin its hashes in the live set forever.
        sidecar_bytes: None,
    };
    store
        .put_layer_checkpoint(object_id, to_seq, envelope, meta)
        .await
        .unwrap_or_else(|e| panic!("hand-put of tail@{to_seq} must succeed: {e}"))
}

/// Hand-put a **tail** generation whose records are the caller's own
/// per-commit OMPF bytes — the builder for tests that REPLAY a lineage.
///
/// `records` is `(sequence_num, the OMPF that commit sent)`, in commit order.
/// The span is `(from_seq, records.last().0]`, so `to_seq` is derived rather
/// than passed: a header claiming coverage the records do not carry cannot be
/// built here at all.
///
/// # Why this exists — neither sibling fixture can serve a round trip
///
/// Epic `t1.dfh.yj` is the first consumer that actually replays a tail onto
/// restored state and compares the result with the live layer. Both existing
/// builders are structurally unable to support that comparison:
///
/// - [`put_tail_generation`]'s record payloads are **reset OMPFs**
///   (`common::empty_reset_ompf`: `is_diff = true, base_version = 0`). Replaying
///   one takes `apply_delta`'s reset branch and **WIPES** the state on every
///   record, so composing a base with such a tail yields an empty layer at the
///   head sequence — a fixture that goes green on a round-trip assertion while
///   proving nothing. Its own rustdoc says as much.
/// - [`drive_captured_tail`] produces a genuinely captured tail, but plants its
///   lineage root with [`put_base_generation`] — an **empty** base at
///   [`CAPTURED_TAIL_SEED_SEQ`] — while the live layer carries 120 sections. Its
///   composed result therefore cannot equal the live state either.
///
/// The shape that does work is a **real-commit lineage**: let the periodic loop
/// write the base (a real assembler-produced snapshot at a real `snap_seq`),
/// fence the loop, drive further `UpdateLayer`s, and hand the exact OMPF bytes
/// those commits sent to this builder at their returned `sequence_num`s. That is
/// byte-for-byte what the capture path persists — `le:{oid}:obj:stream`'s `ompf`
/// field is the writer's raw inbound OMPF — so the artifact is real, and it is
/// fully deterministic.
///
/// # Contiguity is asserted, not assumed
///
/// `from_seq` is the **EXCLUSIVE** lower bound of the span (b.rrx), so the first
/// record must be `from_seq + 1` and each subsequent record its predecessor's
/// successor. The composition engine rejects a gap with `RecordSeqGap` /
/// `TailSeqGap`, which would surface as a step-down in whatever test built the
/// fixture wrong; asserting here fails at the fixture instead.
///
/// # `observed_commit_span`
///
/// Written onto the meta verbatim. Pass `0` for the placeholder shape both
/// adapters' meta-absent fallbacks produce — that is the value a test uses to
/// discriminate whether the restore path sources the guard band's span input
/// from the meta or from the envelope **header** (`to_seq - from_seq`, which no
/// placeholder can zero).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub async fn put_tail_generation_from_records(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    session_id: &str,
    layer_name: &str,
    base_snap_seq: u64,
    from_seq: u64,
    records: &[(u64, Bytes)],
    written_at_unix_ms: u64,
    observed_commit_span: u64,
) -> CheckpointHandle {
    assert!(
        !records.is_empty(),
        "put_tail_generation_from_records: a tail must carry at least one record; an empty \
         slice would encode a span the header cannot describe"
    );
    let mut want = from_seq + 1;
    for (index, (sequence_num, _)) in records.iter().enumerate() {
        assert_eq!(
            *sequence_num,
            want,
            "put_tail_generation_from_records: records must be contiguous starting at \
             `from_seq + 1` ({}) — `from_seq` is the EXCLUSIVE lower bound of the \
             ({from_seq}, to_seq] span (b.rrx). Record {index} carries sequence \
             {sequence_num}, expected {want}. Sequences supplied: {:?}",
            from_seq + 1,
            records.iter().map(|(s, _)| *s).collect::<Vec<_>>()
        );
        want += 1;
    }
    let to_seq = records.last().expect("non-empty records checked above").0;

    let header = TailHeader {
        object_id,
        base_snap_seq,
        from_seq,
        to_seq,
    };
    let envelope_records: Vec<TailRecord> = records
        .iter()
        .map(|(sequence_num, ompf)| TailRecord::Verbatim {
            sequence_num: *sequence_num,
            server_ts: HAND_PUT_SERVER_TS as i64,
            ompf: ompf.clone(),
        })
        .collect();
    let envelope = encode_tail_envelope(&header, envelope_records)
        .unwrap_or_else(|e| panic!("tail envelope ({from_seq}, {to_seq}] must encode: {e}"));

    // The same identity-complete meta shape `put_tail_generation` writes:
    // `validate_tail_put` rejects a tail whose lineage fields are not both
    // `Some`, or that carries an ext-refs sidecar.
    let meta = CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms,
        snap_seq: Some(to_seq),
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: HAND_PUT_CREATED_BY.to_string(),
        server_ts: HAND_PUT_SERVER_TS,
        observed_commit_span,
        kind: CheckpointKind::Tail,
        base_snap_seq: Some(base_snap_seq),
        from_seq: Some(from_seq),
        sidecar_bytes: None,
    };
    store
        .put_layer_checkpoint(object_id, to_seq, envelope, meta)
        .await
        .unwrap_or_else(|e| panic!("hand-put of record-backed tail@{to_seq} must succeed: {e}"))
}

// ===========================================================================
// Tail metric readers (Epic `t1.dfh.57`, Task 7)
//
// LABEL DISCIPLINE — the whole reason these live here rather than being
// re-derived per file. Two opposite traps sit on these families, and BOTH
// present as a permanently-green assertion:
//
//  1. **Absent-label subset match.** `metrics::test_util::sample_value`
//     (src/observability/metrics.rs) matches labels as a SUBSET, first match
//     wins. `live_edit_checkpoint_tail_total` carries `outcome` ONLY — Task 1
//     pinned it service-free so it pairs with `checkpoint_writes_total` without
//     a `sum without(service)` rewrite (src/observability/metrics.rs, the
//     `CHECKPOINT_TAIL_OUTCOMES` seeding loop and its "Deliberately unlabeled by
//     `service`" comment). A matcher naming a `service` label therefore matches
//     NOTHING on this family and every assertion silently reads 0. That is why
//     [`read_tail_counter`] takes `outcome` alone and has no `service` overload.
//  2. **Pre-init zero.** `Metrics::new` zero-seeds every cell of both families
//     at construction. So an absolute-value assertion ("this cell is 0") is
//     satisfied by a feature that never ran, and is not evidence of anything.
//     The correct posture — the `read_writes_counter` posture from
//     `checkpointer_metrics_test.rs` — is BEFORE/AFTER DELTAS around the drive
//     under test.
//
// Consequence of (2): every reader below PANICS when its named series is
// absent, and never folds to 0. An absent series on a zero-seeded family means
// the seeding loop or the label set changed underneath the test, which is
// exactly the regression a `.unwrap_or(0)` would convert into a green run.
// ===========================================================================

/// Read `live_edit_checkpoint_tail_total{outcome}`.
///
/// `outcome` must be a member of
/// `live_edit::observability::metrics::CHECKPOINT_TAIL_OUTCOMES`; every member
/// is zero-seeded at `Metrics::new`, so a panic here means the vocabulary or the
/// label set moved, not that the tail path is idle.
///
/// Use as a BEFORE/AFTER delta — see the label-discipline banner above.
#[allow(dead_code)]
pub fn read_tail_counter(metrics: &Metrics, outcome: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_tail_total",
        &[("outcome", outcome)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} is ABSENT. Every member of \
             CHECKPOINT_TAIL_OUTCOMES is zero-seeded at Metrics::new, and this family carries NO \
             `service` label, so an absent series means either {outcome:?} is not in the pinned \
             vocabulary or the family's label set changed. Deliberately not folded to 0 — that \
             would make a renamed metric read as an idle one"
        )
    }) as u64
}

/// The two `live_edit_checkpoint_tail_total{outcome}` cells that both mean
/// **a capture landed and persisted a Tail artifact**.
///
/// # Why these two are one claim
///
/// SR-1.4 splits a landed capture by whether the per-capture caps
/// (`LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES`, default 4096;
/// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`) cut the sound prefix short:
/// `captured` when the whole prefix fit, `captured_capped` when it did not.
/// Both persist exactly one Tail, both advance the recovery point, and both
/// keep the invariant that a tail never claims coverage beyond its content —
/// so `captured_capped` is explicitly **not** an error signal (see the
/// `checkpoint_tail_total` doc on `Metrics`). Which of the two fires is the capture
/// unit's cap decision, never the claim of a test that only needs "a capture
/// landed".
///
/// # Why reading `captured` alone is a runner-speed flake
///
/// The split is decided by how many stream entries accumulated in the window,
/// which on a hot-writer fixture is a function of **how fast the runner
/// commits** — not of the behaviour under test. An unpaced `HotWriter` that
/// outruns the entry cap between ticks produces `captured_capped` on every
/// tick forever, so a poll waiting on `captured` never completes while the
/// system is doing exactly the right thing. That is a real observed CI failure
/// (pipeline 65897884): the loop landed 16 captures and 2 chain collapses, and
/// the test failed because all 16 were the capped spelling.
///
/// Raising a fixture's entry cap "out of reach" is not a fix — the same
/// pipeline showed ~196k commits inside one 25 s window, above any headroom a
/// fixture can name. Summing the two cells is speed-independent, and it is what
/// the assertions actually mean.
pub const TAIL_CAPTURE_LANDED_OUTCOMES: [&str; 2] = ["captured", "captured_capped"];

/// `captured + captured_capped` off a live `Metrics` — the counter-style
/// reader for [`TAIL_CAPTURE_LANDED_OUTCOMES`].
///
/// Delegates to [`read_tail_counter`] per member, so an absent or renamed cell
/// still panics with that reader's diagnostic rather than silently reading 0.
pub fn read_tail_captures_landed(metrics: &Metrics) -> u64 {
    TAIL_CAPTURE_LANDED_OUTCOMES
        .iter()
        .map(|outcome| read_tail_counter(metrics, outcome))
        .sum()
}

/// `captured + captured_capped` off a [`gather_tail_outcome_cells`] snapshot —
/// the snapshot-style reader for [`TAIL_CAPTURE_LANDED_OUTCOMES`].
///
/// Panics on an absent member for the same reason [`read_tail_counter`] does:
/// every member of the vocabulary is zero-seeded at `Metrics::new`, so absence
/// means the family's label set moved, and folding it to 0 would make a renamed
/// metric read as an idle one.
pub fn tail_captures_landed(cells: &HashMap<String, f64>) -> f64 {
    TAIL_CAPTURE_LANDED_OUTCOMES
        .iter()
        .map(|outcome| {
            *cells.get(*outcome).unwrap_or_else(|| {
                panic!(
                    "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} was ABSENT from a \
                     tail-outcome snapshot. Every member of CHECKPOINT_TAIL_OUTCOMES is \
                     zero-seeded at Metrics::new, so an absent cell means the vocabulary or the \
                     label set moved. Cells: {cells:?}"
                )
            })
        })
        .sum()
}

/// Read `live_edit_checkpoint_restore_lineage_total{outcome, service}` — the
/// restore-side sibling of [`read_tail_counter`]'s capture-side cell.
///
/// `outcome` must be a member of
/// `live_edit::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`
/// (`gathered_base`, `gathered_composed`, the six `stepped_down_*` reasons, and
/// the two exhaustion cells `exhausted` / `exhausted_budget`). Every member is
/// zero-seeded at `Metrics::new`.
///
/// The two exhaustion cells are not interchangeable, and a test asserting on
/// one should say which it means: `exhausted` is the CONCLUSIVE arm — the
/// selection adjudicated every full generation the listing held and none is
/// usable, so the client gets a terminal `NOT_FOUND` — while
/// `exhausted_budget` is `Exhaustion::BudgetInconclusive`, where a full
/// generation went unread and the client gets a retriable `UNAVAILABLE`
/// (b.hxk). Only the first is evidence of checkpoint loss.
///
/// # Label discipline — this family DOES carry `service`
///
/// Read off the increment site, not off a ticket: `Metrics::inc_checkpoint_
/// restore_lineage(outcome, service)` emits `{outcome, service}`, unlike
/// `live_edit_checkpoint_tail_total`, which is deliberately service-free. Both
/// traps in the banner above therefore bite in the opposite direction here: a
/// matcher naming `outcome` alone would subset-match the FIRST rendered cell
/// carrying that outcome under ANY service label, so this reader pins
/// `service = SERVICE_LABEL` (the harness's registry label) to address exactly
/// the cell the harness's own hook increments.
///
/// Panics (never `.unwrap_or(0)`) on an absent series, matching
/// [`read_tail_counter`]: on a zero-seeded family an absent cell means the
/// vocabulary or the label set moved, which is precisely the regression a fold
/// to `0` would render as a quiet green.
///
/// Use as a BEFORE/AFTER delta. This counter is explicitly **not**
/// one-tick-per-attempt: one recovery attempt that steps down twice before
/// resolving increments three cells.
#[allow(dead_code)]
pub fn read_restore_lineage_counter(metrics: &Metrics, outcome: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_restore_lineage_total",
        &[("outcome", outcome), ("service", SERVICE_LABEL)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_restore_lineage_total{{outcome={outcome:?}, \
             service={SERVICE_LABEL:?}}} is ABSENT. Every member of \
             CHECKPOINT_RESTORE_LINEAGE_OUTCOMES is zero-seeded at Metrics::new, so an absent \
             series means either {outcome:?} is not in the pinned vocabulary or the family's \
             label set changed (it carries {{outcome, service}} — check \
             `Metrics::inc_checkpoint_restore_lineage`). Deliberately not folded to 0 — that \
             would make a renamed metric read as a restore path that never stepped down"
        )
    }) as u64
}

/// Read `live_edit_checkpoint_writes_total{result, kind="layer_tail"}` — the
/// artifact-level counterpart of [`read_tail_counter`]'s outcome-level cell.
///
/// The two are a PAIR and are read together on purpose: a capture that succeeds
/// ticks `tail_total{captured}` AND `writes_total{success, layer_tail}`, while a
/// refusal ticks only the former (nothing was written) and a store error ticks
/// only the latter (the outcome family has no error cell by design).
///
/// `checkpoint_writes_total` is likewise service-free, and its whole
/// `result × kind` grid is zero-seeded, so the same panic-on-absent discipline
/// applies.
#[allow(dead_code)]
pub fn read_tail_writes_counter(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer_tail")],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_writes_total{{result={result:?}, kind=\"layer_tail\"}} is \
             ABSENT. The whole result × kind grid is zero-seeded at Metrics::new from \
             CHECKPOINT_WRITE_KINDS, so an absent series means the grid or the label set changed"
        )
    }) as u64
}

/// Read `live_edit_checkpoint_writes_total{result, kind="layer"}` — the exact
/// sibling of [`read_tail_writes_counter`], addressing the artifact-level cell a
/// BASE (or re-base) write ticks rather than a tail's.
///
/// Takes `&Metrics` rather than a harness type deliberately: a
/// `SharedMetrics` (`Arc<Metrics>`) argument deref-coerces at the call site, so
/// this one signature serves the `CheckpointHarness` shape, the
/// `StorageApiCheckpointHarness` shape, and a bare registry alike. That is the
/// property the three file-local copies this replaces did NOT have — one was
/// typed on `&CheckpointHarness` and was therefore out of reach by signature for
/// every other caller, which is how the third copy came to exist.
///
/// Panic-on-absent for the same reason as the tail sibling: the whole
/// `result × kind` grid is zero-seeded at `Metrics::new`, so an absent series
/// means the vocabulary or the label set moved, and a fold to `0` would render
/// that as a quiet green.
pub fn read_layer_writes_counter(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer")],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_writes_total{{result={result:?}, kind=\"layer\"}} is ABSENT. \
             The whole result × kind grid is zero-seeded at Metrics::new from \
             CHECKPOINT_WRITE_KINDS, so an absent series means the grid or the label set changed. \
             Deliberately not folded to 0"
        )
    }) as u64
}

/// The pinned `result` vocabulary of `live_edit_checkpoint_ext_recovery_total`,
/// verbatim from the seeding loop in `src/observability/metrics.rs`
/// (`for result in ["recovered", "adapter_miss", "adapter_transient",
/// "adapter_permanent", "adapter_other"]`).
///
/// Named here as the WRITTEN-DOWN vocabulary, not as an iteration source:
/// [`ExtRecoveryCells::read`] hand-lists all five labels (one per named struct
/// field, so an array walk buys it nothing), and [`ext_recovery_count`]'s
/// panic-on-absent message names this const as the authority a reader checks a
/// suspect `result` against. Hence the `#[allow(dead_code)]` — genuinely
/// unreferenced by code, and kept because a vocabulary the panic text cites has
/// to exist somewhere the reader can find it. A new arm in
/// `src/observability/metrics.rs`'s seeding loop belongs here AND in
/// [`ExtRecoveryCells`].
#[allow(dead_code)]
pub const EXT_RECOVERY_RESULTS: [&str; 5] = [
    "recovered",
    "adapter_miss",
    "adapter_transient",
    "adapter_permanent",
    "adapter_other",
];

/// Read one `live_edit_checkpoint_ext_recovery_total{result}` cell. Returns the
/// LIVE value, not a delta — callers snapshot pre/post (see
/// [`ExtRecoveryCells`]).
///
/// `result` is the whole label set: the family carries no `service` label by
/// design of SR-3.2's metric (operator alerts care about the aggregate rate, not
/// the per-replica one) — read off the increment site,
/// `Metrics::inc_ext_recovery`.
///
/// **Panic-on-absent, deliberately** — this is the one behavioural change from
/// the two `.unwrap_or(0.0)` copies it replaces. The five-value `result` grid is
/// zero-seeded at `Metrics::new`, so an absent series cannot mean "the recovery
/// path is idle"; it can only mean the vocabulary or the label set moved, which
/// is exactly the regression a fold to `0` converts into a green run. Same
/// posture as [`read_tail_counter`] / [`read_tail_writes_counter`].
///
/// Takes `&Metrics` for the same reach reason as
/// [`read_layer_writes_counter`]: `&SharedMetrics` and `&harness.metrics` both
/// deref-coerce, so one signature replaces both the `&SharedMetrics` copy and
/// the `&ExtValueRecoveryHarness` copy.
pub fn ext_recovery_count(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_ext_recovery_total",
        &[("result", result)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_ext_recovery_total{{result={result:?}}} is ABSENT. All five \
             members of EXT_RECOVERY_RESULTS are zero-seeded at Metrics::new and the family \
             carries NO `service` label, so an absent series means either {result:?} is not in \
             the pinned vocabulary or the label set changed (check \
             `Metrics::inc_ext_recovery`). Deliberately not folded to 0 — that would make a \
             renamed metric read as a recovery path that never fired"
        )
    }) as u64
}

/// A snapshot of ALL FIVE `ext_recovery` outcome cells at one instant.
///
/// The whole grid, never one cell: proving the arm under test moved is only half
/// the claim — the other half is that the OTHER four stayed flat, which is what
/// separates "recovered once" from "recovered once after two adapter errors it
/// swallowed". Use as a BEFORE/AFTER pair around the drive.
#[derive(Debug, Clone, Copy)]
pub struct ExtRecoveryCells {
    pub recovered: u64,
    pub adapter_miss: u64,
    pub adapter_transient: u64,
    pub adapter_permanent: u64,
    pub adapter_other: u64,
}

impl ExtRecoveryCells {
    pub fn read(metrics: &Metrics) -> Self {
        Self {
            recovered: ext_recovery_count(metrics, "recovered"),
            adapter_miss: ext_recovery_count(metrics, "adapter_miss"),
            adapter_transient: ext_recovery_count(metrics, "adapter_transient"),
            adapter_permanent: ext_recovery_count(metrics, "adapter_permanent"),
            adapter_other: ext_recovery_count(metrics, "adapter_other"),
        }
    }
}

/// Assert the `ext_recovery` family moved by exactly `(recovered,
/// adapter_miss)` and by ZERO on the three adapter-error arms.
///
/// Lifted alongside [`ExtRecoveryCells`] because the struct on its own is only
/// the reading half — this is the claim it exists to make, and leaving it
/// file-local is what would let a consumer re-fork a subset comparison that
/// silently stops covering a new arm. Callers that DO inject an adapter fault
/// want a bespoke assertion rather than this one; its whole contract is "the
/// error arms stayed flat".
pub fn assert_ext_recovery_delta(
    pre: ExtRecoveryCells,
    post: ExtRecoveryCells,
    recovered: u64,
    adapter_miss: u64,
    context: &str,
) {
    assert_eq!(
        post.recovered - pre.recovered,
        recovered,
        "{context}: ext_recovery{{result=\"recovered\"}} must move by exactly {recovered} \
         ({} → {})",
        pre.recovered,
        post.recovered
    );
    assert_eq!(
        post.adapter_miss - pre.adapter_miss,
        adapter_miss,
        "{context}: ext_recovery{{result=\"adapter_miss\"}} must move by exactly {adapter_miss} \
         ({} → {})",
        pre.adapter_miss,
        post.adapter_miss
    );
    assert_eq!(
        post.adapter_transient - pre.adapter_transient,
        0,
        "{context}: ext_recovery{{result=\"adapter_transient\"}} MUST stay flat — a tick here \
         means the adapter failed transiently for a reason the caller does not model. A test \
         that deliberately injects a transient fault must assert its own delta rather than \
         calling this helper"
    );
    assert_eq!(
        post.adapter_permanent - pre.adapter_permanent,
        0,
        "{context}: ext_recovery{{result=\"adapter_permanent\"}} MUST stay flat"
    );
    assert_eq!(
        post.adapter_other - pre.adapter_other,
        0,
        "{context}: ext_recovery{{result=\"adapter_other\"}} MUST stay flat — a tick here means \
         `get_ext_blob` returned a variant its trait contract reserves to other methods"
    );
}

/// Every rendered cell of `live_edit_checkpoint_tail_total`, as
/// `outcome → value`.
///
/// The cardinality guard reads `.keys()`: parsing the family whole is what makes
/// an OUT-OF-BAND outcome (one the pre-init loop never seeded, materialised on
/// its first occurrence) visible as an EXTRA key, where a per-name
/// [`read_tail_counter`] probe simply would not look for it.
pub fn gather_tail_outcome_cells(metrics: &Metrics) -> HashMap<String, f64> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_tail_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let needle = "outcome=\"";
        let Some(i) = line.find(needle) else {
            continue;
        };
        let after = &line[i + needle.len()..];
        let Some(end) = after.find('"') else {
            continue;
        };
        let value = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
            .expect("every rendered sample line must end in a parseable value");
        out.insert(after[..end].to_owned(), value);
    }
    out
}

/// Every rendered cell of `live_edit_checkpoint_base_bootstrap_total`, as
/// `(outcome, site) → value`.
///
/// The family Epic 5's SR-7 adjudication minted for `BaseBootstrap::run_once`,
/// parsed WHOLE for the same reason as [`gather_tail_outcome_cells`]: an
/// out-of-vocabulary `outcome` or `site` materialised on first occurrence appears
/// here as an EXTRA key, where a per-cell probe would not look for it. Callers
/// snapshot before/after and compare key SETS as well as values.
///
/// The single reader for this family across the checkpoint suite: the per-arm
/// producer module (`bootstrap_observability_test.rs`), the concurrency-bound
/// file's creation-site shed fence (`bootstrap_concurrency_bound_test.rs`, case
/// 6) and `first_base_bootstrap_test.rs`'s gate-off arm (via
/// `BootstrapFixture::base_bootstrap_cells`, a one-line delegation) all call
/// this one body. They parse the same family for opposite claims — one names a
/// single moving cell, another names none — which is a difference in the
/// ASSERTIONS, not in the reader, so one parse serves all three.
pub fn gather_base_bootstrap_cells(metrics: &Metrics) -> HashMap<(String, String), f64> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashMap<(String, String), f64> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_base_bootstrap_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let (Some(outcome), Some(site)) = (
            rendered_label_value(line, "outcome"),
            rendered_label_value(line, "site"),
        ) else {
            continue;
        };
        let value = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
            .expect("every rendered sample line must end in a parseable value");
        out.insert((outcome, site), value);
    }
    out
}

/// One label value off a rendered Prometheus sample line, by label name.
///
/// Deliberately string-scanning rather than a full exposition parser: the
/// gatherers above want exactly one thing from a line they have already matched
/// by family name, and a `None` return simply skips that line.
fn rendered_label_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let i = line.find(&needle)?;
    let after = &line[i + needle.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_owned())
}

// ===========================================================================
// The faithful-column oracle comparison (Epic `t1.dfh.be`, Task 2 / Issue
// b.kfg sub-task 4)
//
// ONE `assert_faithful_columns` and ONE `assert_restored_entry_versions_flattened`
// for the whole integration side. They replace three diverged file-local copies
// (`e2e_composed_restore_test.rs`, `restore_fault_injection_matrix_test.rs`,
// `e2e_full_wipe_composed_restore_test.rs`), whose divergence was live and
// directional rather than cosmetic:
//
//   * only the full-wipe copy normalized external values through
//     `ComparableValue` — the other two compared `FieldValue` whole, which was
//     indistinguishable only for the external-value-free
//     `e2e_composed_restore_test.rs` fixture; `ComparableValue`'s own doc
//     records why the third file's `ext_value_index` comparison is dropped by
//     design;
//   * only the full-wipe copy took `ignored_field_keys`;
//   * NONE of the three compared section `last_version` above the base, while
//     the two co-equal lanes on the same basis — `src/checkpoint/compose.rs`'s
//     `versions_above` and `tests/proptests.rs`'s `ac4_versions_above` — DO.
//
// The consolidated form below is the SUPERSET: the union of the three copies'
// compared columns, plus the section arm the other two lanes already carry.
// Extending is the contract for anything a re-base lane needs next; forking a
// fourth copy is what this block exists to prevent.
// ===========================================================================

/// A field / time-sample value projected onto the columns a checkpoint round
/// trip carries faithfully.
///
/// `ext_value_index` is deliberately ABSENT from the `External` variant — it is
/// an OMPF part index, re-minted per serialization, so live and composed can
/// never agree on it by construction. Everything else about an external
/// reference (the content hash and the declared size) IS compared.
///
/// Of the two deleted copies that compared `FieldValue` whole, only
/// `e2e_composed_restore_test.rs`'s fixture is external-value-free, so there the
/// two forms are indistinguishable and nothing is lost.
/// `restore_fault_injection_matrix_test.rs`'s is NOT: it carries an external
/// value inside the replayed range (`tail_ext_value_delta`, written at
/// `ext_value_index: EXT_VALUE_INDEX`), so routing it through here genuinely
/// DROPS that file's `ext_value_index` comparison. That drop is BY DESIGN, not
/// an oversight of the sweep: the adjudicated basis excludes the index, because
/// it addresses a position inside an OMPF container rather than state. Both
/// co-equal lanes on that basis already exclude it —
/// `src/checkpoint/compose.rs`'s `assert_field_value_faithful` and the AC-4
/// proptest table's `FieldValue::External::ext_value_index` row — so the
/// consolidated form brings the third lane onto the same column set rather than
/// weakening it below them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparableValue {
    Inline(Vec<u8>),
    External { hash: [u8; 32], size: u64 },
}

impl ComparableValue {
    pub fn of(value: &FieldValue) -> Self {
        match value {
            FieldValue::Inline(bytes) => ComparableValue::Inline(bytes.clone()),
            FieldValue::External {
                hash,
                size,
                ext_value_index: _,
            } => ComparableValue::External {
                hash: *hash,
                size: *size,
            },
        }
    }
}

/// Per-call knobs for [`assert_faithful_columns`].
///
/// `Default` is the empty slice — compare every field on both sides. Construct
/// with `FaithfulColumnsOptions::default()` unless the test has a field that
/// legitimately exists on ONE side only.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaithfulColumnsOptions<'a> {
    /// Field key names to drop from BOTH sides before comparing.
    ///
    /// The one live use is a post-recovery marker: a commit the test re-issues
    /// through the recovery branch AFTER the oracle was captured, which is
    /// therefore present on the restored side and absent from the oracle by
    /// design. Anything else named here is a claim that a column is
    /// uncomparable, and belongs in the excluded-column block on
    /// [`assert_faithful_columns`] with its reason.
    pub ignored_field_keys: &'a [&'a str],
}

impl<'a> FaithfulColumnsOptions<'a> {
    /// Ignore the named field keys on both sides.
    pub fn ignoring(ignored_field_keys: &'a [&'a str]) -> Self {
        Self { ignored_field_keys }
    }
}

/// Compare a restored snapshot against a live oracle on the **faithful-column
/// basis**: the columns a checkpoint round trip CARRIES, rather than the ones it
/// re-derives.
///
/// # Compared
///
/// - **sections** — `parent_id`, `name`, `section_type`, `section_order`
/// - **children lists**, IN ORDER (the order is the authoring order, which is
///   persisted payload, not an implementation detail, so an order-insensitive
///   comparison would miss a reordering)
/// - **fields** — `(section_id, key_name)` → [`ComparableValue`] +
///   `authoring_set_order`
/// - **time samples** — `(section_id, time)` → `value`
///
/// Per-entry `version` is the companion
/// [`assert_restored_entry_versions_flattened`]'s half.
///
/// # Deliberately NOT compared — the authoritative excluded-column list
///
/// One copy, here. `src/checkpoint/compose.rs` (`assert_faithful_columns` /
/// `versions_above`, in-crate) and `tests/proptests.rs` (the AC-4 block's
/// `ac4_*` helpers) are the CO-EQUAL lanes on this same basis; their block
/// headers carry the same table and the three must be changed together.
///
/// Those two lanes compare **in-memory** composed state against live, which is
/// where the per-entry version columns are still a witness. This lane compares
/// what a restore PERSISTED, and since b.2ti `apply_restore` re-stamps the three
/// versions a diff builder FILTERS on — `Section::last_version`,
/// `Field::version`, `TimeSample::version` — onto `seeded_seq` before
/// `write_object`, so those columns carry no per-commit information at all.
/// `Section::first_version` is deliberately exempt from the re-stamp and
/// survives at the value the base decode / composition produced, which is not
/// the live value either. That whole shape is what
/// [`assert_restored_entry_versions_flattened`] asserts here, and it is why the
/// version rows below read "stamped at `seeded_seq`" (three columns) and
/// "re-derived by the decode" (the fourth) rather than "flattened to 1".
///
/// Every column below is re-derived identically by a base-only restore at HEAD,
/// so comparing it would test the reset round trip rather than the restore:
///
/// - **cumulative `Field::set_order` and `ObjectState::next_set_order`** — the
///   reset OMPF carries the AUTHORING rank in OODF's single `set_order` slot,
///   and `apply_set_field` re-accumulates the cursor from `0` in the base's
///   emission order, over the surviving fields only — a different walk from the
///   authoring interleaving live accumulated across many commits. What a
///   restored cursor CAN be held to has its own pinning assertion in
///   `e2e_composed_restore_test.rs` (`assert_cumulative_set_order_columns`).
/// - **the three diff-FILTERED entry versions** (`Field::version`,
///   `TimeSample::version`, `Section::last_version`) — the decode leaves
///   base-era entries at `version = 1` and a composed lineage leaves them below
///   its tail-era ones, but neither shape survives to Valkey: `apply_restore`
///   re-stamps these three onto `seeded_seq` (b.2ti), because a restore is one
///   write of the whole object and `src/domain/diff.rs`'s
///   `version > have_version` filter would otherwise withhold every restored
///   entry from every reconnecting subscriber. What the persisted columns CAN be
///   held to is therefore a single value, and that is
///   [`assert_restored_entry_versions_flattened`]'s claim.
/// - **`Section::first_version`** — excluded for a DIFFERENT reason from the
///   three above, and the difference matters. It is NOT re-stamped: the
///   abandoned-prim retraction arm of `plan_faithful_coalesced` fires on
///   `first_version <= have_version && !live` (`src/domain/diff.rs`), so
///   raising it would strand the orphan sections a restored base carries as
///   permanently unretractable. It is uncomparable against a LIVE oracle
///   anyway, because the base decode re-derives it: one reset delta applied to
///   a fresh state stamps every section the base carried at `1`, whatever
///   sequence created it live. What the restored column CAN be held to is the
///   value its own producer (decode or replay) minted, which is
///   [`RestoredFirstVersions`]'s rule and
///   [`assert_restored_entry_versions_flattened`]'s arm 3.
/// - **`ObjectState::first_version`** — the decoder zeroes the object-level
///   column and the restore leaves it there (nothing reads it; the first
///   post-recovery write populates it).
/// - **tombstones** — a reset delta describes current state and carries no
///   tombstone list by construction (`build_reset_ompf_from_state`).
/// - **`FieldValue::External::ext_value_index`** — an OMPF part index, re-minted
///   per serialization; it addresses a position inside a container, not state.
///   This is why the field / time-sample comparison goes through
///   [`ComparableValue`] instead of comparing `FieldValue` whole.
///
/// **Adding a column to this list is a claim, and the claim must be written
/// down here.** The list is the only place a reader can tell "this column is
/// re-derived, so a difference is not evidence" apart from "this column
/// diverged and someone silenced it". A new entry with no stated reason is the
/// failure mode the three diverged copies produced.
pub fn assert_faithful_columns(
    restored: &ObjectSnapshot,
    oracle: &ObjectSnapshot,
    opts: FaithfulColumnsOptions<'_>,
    context: &str,
) {
    let section_columns = |snap: &ObjectSnapshot| -> BTreeMap<u64, (u64, String, u8, u32)> {
        snap.sections
            .values()
            .map(|s| {
                (
                    s.section_id,
                    (s.parent_id, s.name.clone(), s.section_type, s.section_order),
                )
            })
            .collect()
    };
    assert_eq!(
        section_columns(restored),
        section_columns(oracle),
        "{context}: restored sections must equal the oracle on \
         (parent_id, name, section_type, section_order)"
    );

    let children = |snap: &ObjectSnapshot| -> BTreeMap<u64, Vec<u64>> {
        snap.children
            .iter()
            .map(|(parent, kids)| (*parent, kids.clone()))
            .collect()
    };
    assert_eq!(
        children(restored),
        children(oracle),
        "{context}: restored children lists must equal the oracle IN ORDER — the list order is \
         the authoring order, not an implementation detail"
    );

    let field_columns = |snap: &ObjectSnapshot| -> BTreeMap<(u64, String), (ComparableValue, u64)> {
        snap.fields
            .iter()
            .filter(|f| !opts.ignored_field_keys.contains(&f.key_name.as_str()))
            .map(|f| {
                (
                    (f.section_id, f.key_name.clone()),
                    (ComparableValue::of(&f.value), f.authoring_set_order),
                )
            })
            .collect()
    };
    assert_eq!(
        field_columns(restored),
        field_columns(oracle),
        "{context}: restored fields must equal the oracle on (value, authoring_set_order), with \
         external references compared by (hash, size)"
    );

    let time_sample_columns = |snap: &ObjectSnapshot| {
        let mut out: Vec<_> = snap
            .time_samples
            .iter()
            .map(|t| (t.section_id, t.time, ComparableValue::of(&t.value)))
            .collect();
        out.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        out
    };
    assert_eq!(
        time_sample_columns(restored),
        time_sample_columns(oracle),
        "{context}: restored time samples must equal the oracle on (section_id, time, value)"
    );
}

/// A single addressable entry in an [`ObjectSnapshot`], for the per-entry
/// version comparison.
///
/// Mirrors `src/checkpoint/compose.rs`'s `EntryKey` and `tests/proptests.rs`'s
/// `Ac4EntryKey` variant-for-variant, so the three lanes read one column set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntryKey {
    Section(u64),
    Field(u64, String),
    TimeSample(u64, OrderedFloat<f64>),
}

/// Every entry stamped at a version strictly above `base`, grouped by that
/// version.
///
/// **Sections are keyed on `last_version`** — the version of the write that last
/// touched them. That is the column the two co-equal lanes
/// (`compose.rs::versions_above`, `proptests.rs::ac4_versions_above`) key on,
/// and the one all three integration copies omitted. `Section::first_version` is
/// not read here at all, so b.2ti's exemption of that one column does not reach
/// this map.
///
/// Exposed as its own function so a lane can read the shape directly — on an
/// ORACLE, to state what the fixture's replayed range covers, or on a restored
/// snapshot, to name what came back. Since b.2ti a restored snapshot's map is
/// always the single key `seeded_seq` — every column it reads is one of the
/// three the restore re-stamps; the per-commit spread survives only on the
/// in-memory lanes named above, so a whole-map equality between a restored side
/// and an oracle is no longer a comparison anything can pass. Use
/// [`assert_restored_entry_versions_flattened`] for the restored side.
pub fn versions_above(snap: &ObjectSnapshot, base: u64) -> BTreeMap<u64, BTreeSet<EntryKey>> {
    let mut out: BTreeMap<u64, BTreeSet<EntryKey>> = BTreeMap::new();
    for section in snap.sections.values() {
        if section.last_version > base {
            out.entry(section.last_version)
                .or_default()
                .insert(EntryKey::Section(section.section_id));
        }
    }
    for field in &snap.fields {
        if field.version > base {
            out.entry(field.version)
                .or_default()
                .insert(EntryKey::Field(field.section_id, field.key_name.clone()));
        }
    }
    for sample in &snap.time_samples {
        if sample.version > base {
            out.entry(sample.version)
                .or_default()
                .insert(EntryKey::TimeSample(sample.section_id, sample.time));
        }
    }
    out
}

/// The [`versions_above`] shape restricted to its `Section` entries, with
/// versions that carry no section dropped so the two sides' maps compare
/// key-for-key.
///
/// Public alongside [`versions_above`] so a re-base lane can assert on the
/// section projection directly — typically as a fixture precondition on the
/// ORACLE ("the replayed range really did touch a section"), which is the use
/// that survives b.2ti's restore-side flattening.
pub fn sections_above(snap: &ObjectSnapshot, base: u64) -> BTreeMap<u64, BTreeSet<EntryKey>> {
    versions_above(snap, base)
        .into_iter()
        .filter_map(|(version, entries)| {
            let sections: BTreeSet<EntryKey> = entries
                .into_iter()
                .filter(|entry| matches!(entry, EntryKey::Section(_)))
                .collect();
            (!sections.is_empty()).then_some((version, sections))
        })
        .collect()
}

/// The `first_version` every section a base decode produced carries.
///
/// `decode_checkpoint_ompf` applies ONE reset delta to a fresh `ObjectState`, so
/// `apply_delta` computes `new_version = 1` and
/// `apply_diff_change_section`'s new-section branch stamps every section the
/// base OMPF carried at that value — whatever sequence created it live. Named
/// here rather than spelled `1` inline for the same reason
/// `src/checkpoint/compose.rs` names its `DECODER_STAMP`: the literal is a claim
/// about the decode step, not a magic number.
const DECODER_SECTION_STAMP: u64 = 1;

/// The two pre-restore sequences [`assert_restored_entry_versions_flattened`]
/// reasons about, kept as one named struct rather than two adjacent `u64`
/// parameters.
///
/// They are numerically equal at every call site today, which is exactly why
/// they must not be positional: two same-typed arguments that happen to agree
/// can be swapped for years without a single test going red, and the day a
/// fixture separates them the swap becomes a silent wrong answer. Naming both
/// at the call site also forces the author to decide which job each value is
/// doing. See the function's rustdoc for what each one claims and the guards
/// relating them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoredVersionRanges {
    /// Floor of the range whose SURVIVAL the presence half checks — arm 1's
    /// non-vacuity guard demands an oracle field stamped above it. A statement
    /// about the FIXTURE.
    pub replayed_range_floor: u64,
    /// The lineage root the base-era/tail-era `first_version` split keys on
    /// under [`RestoredFirstVersions::ComposedLineage`]. A statement about the
    /// restore PAYLOAD's provenance; inert under
    /// [`RestoredFirstVersions::DecodedBase`], but still passed honestly.
    pub lineage_root: u64,
}

/// What [`assert_restored_entry_versions_flattened`]'s section arm must expect
/// on `Section::first_version` — the ONE version column `apply_restore`
/// deliberately does NOT re-stamp (b.2ti step 4a).
///
/// The surviving value is whatever produced the restore payload, and WHICH
/// producer that was is a property of the call site's lineage, not something the
/// helper can read off a restored snapshot. So the caller names it. Both
/// variants are the same rule the co-equal in-crate lane states as
/// `expected(live_version)` in
/// `src/checkpoint/compose.rs::compose_leaves_base_only_entries_at_the_base_decoders_stamp`:
/// a base decode stamps [`DECODER_SECTION_STAMP`], a replayed tail record stamps
/// its own sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoredFirstVersions {
    /// The restore resolved a **base only** — a HEAD base, or the self-contained
    /// artifact an SR-4.1 chain-collapse re-base produced. There is no replay,
    /// so EVERY section comes back at [`DECODER_SECTION_STAMP`], including ones
    /// the oracle stamps far above it.
    ///
    /// Read the lineage outcome the call site already asserts
    /// (`gathered_base` vs `gathered_composed`) rather than guessing: a
    /// collapsed-chain restore is `gathered_base` and belongs here even though
    /// the fixture that built it had a tail.
    DecodedBase,
    /// The restore resolved a **composed lineage** — the base at the lineage
    /// root plus tail records replayed onto it (`gathered_composed`).
    ///
    /// Base-era sections carry [`DECODER_SECTION_STAMP`]; a section a replayed
    /// record CREATED carries that record's own sequence, which is the sequence
    /// the same commit stamped live, so the ORACLE's `first_version` is the
    /// expected value for it. The two split at
    /// [`assert_restored_entry_versions_flattened`]'s `lineage_root` parameter:
    /// at or below it the section was inside the base blob and went through the
    /// decode; above it the section was created inside the replayed range and
    /// composition replayed its real sequence.
    ///
    /// This variant carries its OWN non-vacuity guard, symmetric with arm 1's:
    /// the oracle must hold at least one section stamped above `lineage_root`.
    /// Without one, every branch of the split resolves to
    /// [`DECODER_SECTION_STAMP`] and the arm degenerates into a strictly weaker
    /// copy of [`Self::DecodedBase`] — it would still pass against a restore
    /// that dropped the tail entirely, which is the exact defect the variant
    /// exists to catch.
    ComposedLineage,
}

/// The per-entry `version` half of the composed comparison, split out from
/// [`assert_faithful_columns`] because these are the columns a restore
/// deliberately does NOT carry through.
///
/// # The claim
///
/// A restore is ONE write of the WHOLE object at `seeded_seq`, so every entry it
/// materializes must be VISIBLE as of that sequence: the three versions a diff
/// builder filters on — `Section::last_version`, `Field::version`,
/// `TimeSample::version` — read exactly `seeded_seq`. `apply_restore`'s step 4a
/// re-stamps those three after composition and before `write_object` (b.2ti). It
/// preserves `object_id`, so a subscriber that survives the restore does not
/// re-bootstrap: it reconnects onto the incremental path, where
/// `src/domain/diff.rs` emits only entries with `version > have_version`
/// (`plan_faithful_coalesced` also restates the other children of a parent
/// with such a child, and nothing else).
/// Carrying the pre-restore stamps would withhold the restored content from
/// exactly the subscribers the restore exists to correct — the base decode
/// leaves everything at `version = 1`, and a composed lineage leaves base-era
/// entries below its tail-era ones.
///
/// **`Section::first_version` is the exception, and asserting it is half of this
/// function's job.** It is NOT re-stamped: `plan_faithful_coalesced`'s
/// abandoned-prim retraction arm fires on `first_version <= have_version &&
/// !live`, so raising it above every reconnector's `have_version` would strand
/// the orphan sections a restored base carries (`plan_full_content` emits
/// unreachable sections into the base OMPF) as permanently unretractable. It
/// must therefore come back at the value its producer minted — the decoder's
/// `1`, or a replayed tail record's own sequence — which is what
/// [`RestoredFirstVersions`] names per call site. Delivery is unaffected either
/// way: a section that lands in the "new" bucket (`first_version >
/// have_version`) and one that lands in the "modified" bucket
/// (`first_version <= have_version && last_version > have_version`) both emit
/// the same `DiffChangeSection` upsert, and the flattened `last_version` is what
/// puts every restored section in one bucket or the other.
///
/// **This is therefore a flattening assertion, not an in-order-replay witness.**
/// The witness that the tail records were replayed IN ORDER rather than merely
/// all applied lives on the two lanes that read composition's output BEFORE the
/// restore persists it: `src/checkpoint/compose.rs`'s in-crate `versions_above`
/// tests and `tests/proptests.rs`'s `ac4_versions_above` property, both of which
/// compare in-memory composed state against live and are untouched by the
/// re-stamp. What survives here is the other half of the old assertion's value:
/// every entry the oracle held above `replayed_range_floor` — i.e. everything
/// the replayed range wrote — must still EXIST after the restore, which is the
/// arm that catches a lineage restored base-only.
///
/// # `replayed_range_floor` vs `lineage_root` — two jobs, two fields
///
/// Both are sequences below `seeded_seq` and at every call site today they are
/// the SAME value (the lineage root, e.g. `fixture.s0`), which is why
/// [`RestoredVersionRanges`] makes each caller name both rather than defaulting
/// one from the other. They answer different questions and a future fixture can
/// legitimately separate them:
///
/// - **`replayed_range_floor`** bounds arm 1's non-vacuity guard: "the oracle
///   must hold at least one FIELD stamped above this". It is a statement about
///   the fixture — did the test write anything inside the range whose survival
///   this function is checking. A caller whose oracle spans several lineage
///   generations but only wants the presence guard to bite on the newest one
///   would raise this above the root.
/// - **`lineage_root`** is a claim about the RESTORE PAYLOAD's provenance,
///   consumed only by [`RestoredFirstVersions::ComposedLineage`]: sections at or
///   below it came out of the base blob's decode and must read
///   [`DECODER_SECTION_STAMP`]; sections above it were created by a replayed
///   tail record and must read the oracle's own value. Moving it is not a
///   tightening knob — a wrong value here mis-predicts `first_version` and goes
///   red on a correct restore.
///
/// Guarded rather than assumed: `lineage_root <= replayed_range_floor` (the
/// replay range cannot start below the lineage it replays onto) and both
/// strictly below `seeded_seq` (a floor at or above the restore's own cursor
/// would make the guards vacuous and the split meaningless).
///
/// # Four arms
///
/// 1. **Fields**, walking the ORACLE and looking each entry up on the restored
///    side. That direction is deliberate: a restored-side field the oracle lacks
///    is legitimate (a post-recovery marker commit re-issued after the oracle
///    was captured — the same asymmetry
///    [`FaithfulColumnsOptions::ignored_field_keys`] exists for), so a whole-map
///    equality over [`versions_above`] would go red on a landed test for a
///    fixture asymmetry rather than a restore defect. Carries the non-vacuity
///    guard: the oracle MUST hold at least one field above
///    `replayed_range_floor`, or the fixture wrote nothing inside the replayed
///    range and arm 1 says nothing about a composed restore.
/// 2. **Time samples**, same shape, no non-vacuity guard (a replayed range that
///    writes no time sample is legitimate).
/// 3. **Sections**, on both version columns and in opposite directions, walking
///    the oracle's section ids: `last_version` re-stamped at `seeded_seq`,
///    `first_version` PRESERVED at the value [`RestoredFirstVersions`] derives
///    for this call site, and `first_version < last_version` STRICTLY. The
///    strict `<` is not a tidiness check — it is the standing integration-level
///    guard that no reachable configuration collapses the two. `seeded_seq =
///    restore_floor + band`, `restore_floor` dominates every sequence the object
///    can be proven to have issued (the checkpoint's `snap_seq`, or the
///    surviving stream top), and `band >= 1` because
///    `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` rejects zero. Every surviving
///    `first_version` is `1` or a pre-restore sequence, so it is strictly below.
///    An equality here means either the band went to zero or the re-stamp
///    reached `first_version`, and both are regressions. Under
///    [`RestoredFirstVersions::ComposedLineage`] this arm carries its own
///    non-vacuity guard, symmetric with arm 1's: at least one oracle section
///    must be stamped above `lineage_root`, or the base-era/tail-era split never
///    takes its tail-era branch and the arm silently degenerates to
///    [`RestoredFirstVersions::DecodedBase`].
/// 4. **The restored side as a whole**: no entry may carry a diff-FILTERED
///    version strictly BELOW `seeded_seq`. This is the direction arms 1–3 cannot
///    see — a restored-side entry the oracle never had, brought back at a
///    pre-restore stamp. Entries ABOVE `seeded_seq` are legitimate and
///    deliberately not constrained: they are post-restore commits (the re-issued
///    marker). Sections are checked on `last_version` only; their preserved
///    `first_version` gets the well-formedness pair instead (`>= 1`, and `<=
///    last_version`), because a restored-only section's pre-restore stamp is by
///    definition not on the oracle to compare against.
///
/// # `opts` is honored on the FIELD arms only
///
/// [`FaithfulColumnsOptions::ignored_field_keys`] is consulted in arm 1 and in
/// arm 4's field loop. Arm 2 (time samples) and arm 3 (sections) ignore `opts`
/// entirely: arm 3 in particular asserts a STRICT `seeded_seq` on `last_version`
/// and a STRICT per-section equality on `first_version` for every oracle
/// section, with no escape hatch. That is not an oversight to be relied on
/// quietly — it is safe only under a property of today's single excluding
/// caller.
///
/// The exclusion exists for a post-recovery marker commit: a write the fixture
/// re-issues AFTER the restore, so its entries legitimately sit ABOVE
/// `seeded_seq` and would fail arms 1/3/4 if compared. The one caller that
/// passes `ignoring(..)` is `e2e_full_wipe_composed_restore_test.rs`, whose
/// marker (`post_recovery_marker_delta`, ~:484) is a **`set_fields`-only**
/// delta: no `struct_commands`, no time samples. A field write moves the
/// field's own `version` and nothing else — a section's `last_version` advances
/// only on a structural command touching that section, and its `first_version`
/// only on the command that created it — so the marker leaves every section on
/// both columns exactly where the restore left them and arm 3 passes without
/// needing the option. Arm 2 is likewise unreachable because the marker writes
/// no time sample.
///
/// **A future caller whose marker carries a structural command or a time
/// sample will need `opts` threaded through arms 2 and 3** (and arm 4's section
/// loop) before it can be excluded. Do not instead widen arm 3's `last_version`
/// check to a `>=`: the strict equality is what catches a restore that carried a
/// pre-restore section stamp through, which is the b.2ti defect itself. And do
/// not "fix" a red `first_version` by widening it to `<= seeded_seq` either —
/// the exact value is the claim that `apply_restore` left that column alone.
///
/// # Triage when an arm goes red
///
/// A restored-side version of `0` on a section is the shape to look for first:
/// `read_object_state_for_test` reads the column as
/// `parse_u64(&data, "last_ver").unwrap_or(0)`, so an ABSENT `last_ver` compares
/// as `0` rather than failing to parse. That is "the restore path is not writing
/// `last_ver` for this section class" — a genuine restore gap, not an
/// uncomparable column. A uniform value that is neither `seeded_seq` nor a
/// pre-restore stamp usually means the caller derived the wrong `seeded_seq`
/// (`restore_floor + band`, where the floor folds to the checkpoint head only
/// when the stream was wiped along with the rest of `obj:*`). Report either;
/// do not move the column onto [`assert_faithful_columns`]'s excluded list
/// without a stated reason.
///
/// A red `first_version` splits the same way. `seeded_seq` there is the b.2ti
/// regression — the re-stamp reached the exempt column. `1` where the oracle's
/// value was expected means the lineage was restored BASE-ONLY (the tail was
/// dropped, or the fixture collapsed the chain), which arm 1's presence half
/// normally catches first. The oracle's live value where `1` was expected means
/// the payload never went through a base decode at all. And a caller whose
/// restore resolved a collapsed re-base artifact must pass
/// [`RestoredFirstVersions::DecodedBase`], not `ComposedLineage` — that is a
/// call-site error, not a production defect.
///
/// A red NON-VACUITY guard (either arm 1's field count or arm 3's
/// `ComposedLineage` section count) is a FIXTURE verdict, never a restore one:
/// the restore may well be correct, but the comparison that just ran proved
/// less than the call site claims. Fix the fixture so the replayed range
/// actually writes inside itself — do NOT lower `replayed_range_floor` /
/// `lineage_root` to make the count non-zero, which buys the guard back by
/// widening the range it was measuring.
pub fn assert_restored_entry_versions_flattened(
    restored: &ObjectSnapshot,
    oracle: &ObjectSnapshot,
    ranges: RestoredVersionRanges,
    seeded_seq: u64,
    first_versions: RestoredFirstVersions,
    opts: FaithfulColumnsOptions<'_>,
    context: &str,
) {
    let RestoredVersionRanges {
        replayed_range_floor,
        lineage_root,
    } = ranges;
    assert!(
        seeded_seq > 0,
        "{context}: `seeded_seq` must be the restore's own writer cursor (`restore_floor + \
         band`), never 0 — a zero here makes every arm below vacuous"
    );
    assert!(
        lineage_root <= replayed_range_floor,
        "{context}: `lineage_root` ({lineage_root}) must be at or below `replayed_range_floor` \
         ({replayed_range_floor}) — the replayed range sits ON the lineage it replays onto, so a \
         root ABOVE the floor describes no reachable fixture and would put base-era sections \
         inside the range whose survival arm 1 checks"
    );
    assert!(
        replayed_range_floor < seeded_seq,
        "{context}: `replayed_range_floor` ({replayed_range_floor}) must be strictly below the \
         seeded cursor ({seeded_seq}) — at or above it, arm 1's non-vacuity guard can only be \
         satisfied by post-restore entries and stops witnessing the replayed range"
    );
    assert!(
        lineage_root < seeded_seq,
        "{context}: `lineage_root` ({lineage_root}) must be strictly below the seeded cursor \
         ({seeded_seq}) — every section the restore materialized was minted before it, so a root \
         at or above the cursor collapses arm 3's base-era/tail-era split onto one branch"
    );

    // ── Arm 1: fields ────────────────────────────────────────────────────
    let restored_fields: BTreeMap<(u64, String), u64> = restored
        .fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.version))
        .collect();
    let mut replayed = 0usize;
    for field in &oracle.fields {
        if opts.ignored_field_keys.contains(&field.key_name.as_str()) {
            continue;
        }
        let restored_version = restored_fields
            .get(&(field.section_id, field.key_name.clone()))
            .unwrap_or_else(|| {
                panic!(
                    "{context}: field ({}, {:?}) is on the oracle at version {} and MUST exist \
                     after the restore",
                    field.section_id, field.key_name, field.version
                )
            });
        assert_eq!(
            *restored_version, seeded_seq,
            "{context}: field ({}, {:?}) MUST come back stamped at the restore's own seeded \
             cursor. It read the pre-restore value ({}) instead, which is below every \
             reconnecting subscriber's `have_version` and therefore invisible to the \
             `version > have_version` diff filter (b.2ti)",
            field.section_id, field.key_name, field.version
        );
        if field.version > replayed_range_floor {
            replayed += 1;
        }
    }
    assert!(
        replayed > 0,
        "{context}: the oracle carries NO field stamped above the replayed-range floor \
         ({replayed_range_floor}), so the presence half of this comparison asserted nothing about \
         the replayed range. The fixture must write at least one field inside it"
    );

    // ── Arm 2: time samples ──────────────────────────────────────────────
    let restored_samples: BTreeMap<(u64, u64), u64> = restored
        .time_samples
        .iter()
        .map(|t| ((t.section_id, t.time.to_bits()), t.version))
        .collect();
    for sample in &oracle.time_samples {
        let restored_version = restored_samples
            .get(&(sample.section_id, sample.time.to_bits()))
            .unwrap_or_else(|| {
                panic!(
                    "{context}: time sample ({}, {}) is on the oracle and MUST exist after the \
                     restore",
                    sample.section_id, sample.time
                )
            });
        assert_eq!(
            *restored_version, seeded_seq,
            "{context}: time sample ({}, {}) MUST come back stamped at the seeded cursor, not at \
             its pre-restore version ({})",
            sample.section_id, sample.time, sample.version
        );
    }

    // ── Arm 3: sections — both columns, in OPPOSITE directions ───────────
    //
    // `tail_era` counts the oracle sections that land on the split's ABOVE
    // branch. Under `ComposedLineage` a zero means every branch below resolved
    // to `DECODER_SECTION_STAMP`, i.e. the arm ran as `DecodedBase` — see the
    // guard after the loop.
    let mut tail_era = 0usize;
    for section in oracle.sections.values() {
        let restored_section = restored
            .sections
            .get(&section.section_id)
            .unwrap_or_else(|| {
                panic!(
                    "{context}: section {} is on the oracle and MUST exist after the restore",
                    section.section_id
                )
            });
        assert_eq!(
            restored_section.last_version, seeded_seq,
            "{context}: section {} MUST come back with `last_version` at the seeded cursor \
             ({seeded_seq}); it read {}. That is the column a diff builder filters a MODIFIED \
             section on, so a pre-restore value there withholds the section from every \
             reconnector. A `0` means the restore never wrote it (`read_object_state_for_test` folds an \
             absent column to 0) — see this function's triage note",
            section.section_id, restored_section.last_version
        );

        let expected_first = match first_versions {
            RestoredFirstVersions::DecodedBase => DECODER_SECTION_STAMP,
            RestoredFirstVersions::ComposedLineage => {
                if section.first_version > lineage_root {
                    tail_era += 1;
                    section.first_version
                } else {
                    DECODER_SECTION_STAMP
                }
            }
        };
        assert_eq!(
            restored_section.first_version, expected_first,
            "{context}: section {} MUST come back with `first_version` PRESERVED at {expected_first} \
             ({first_versions:?}, lineage root {lineage_root}, oracle {}); it read {}. \
             `apply_restore` deliberately exempts this column from the b.2ti re-stamp — \
             `plan_faithful_coalesced`'s abandoned-prim arm fires on \
             `first_version <= have_version`, so a value at the seeded cursor ({seeded_seq}) \
             strands every orphan section a restored base carries as unretractable",
            section.section_id,
            section.first_version,
            restored_section.first_version
        );
        assert!(
            restored_section.first_version < restored_section.last_version,
            "{context}: restored section {} must keep `first_version` STRICTLY below \
             `last_version`, got (first {}, last {}). `seeded_seq` is `restore_floor + band`, the \
             floor dominates every sequence the object can be proven to have issued and the band \
             is at least 1 (a zero `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` is rejected at \
             startup), so a preserved `first_version` is always strictly lower. An equality here \
             is either a zero band or the re-stamp reaching the exempt column",
            section.section_id,
            restored_section.first_version,
            restored_section.last_version
        );
    }
    if first_versions == RestoredFirstVersions::ComposedLineage {
        assert!(
            tail_era > 0,
            "{context}: the oracle carries NO section stamped above the lineage root \
             ({lineage_root}), so every `first_version` this arm predicted came from the \
             base-era branch and `ComposedLineage` silently degenerated to `DecodedBase`. That \
             is strictly WEAKER than the claim the call site is making: the same assertions \
             would pass against a restore that dropped the replayed tail entirely and decoded \
             the base alone. Either the fixture's replayed range creates no section (make it \
             create one — arm 1's field guard does not cover this, a field write moves no \
             section column) or the call site should be passing `DecodedBase`"
        );
    }

    // ── Arm 4: nothing on the restored side sits BELOW the seed ──────────
    for section in restored.sections.values() {
        assert!(
            section.last_version >= seeded_seq,
            "{context}: restored section {} carries `last_version` {} below the seeded cursor \
             ({seeded_seq}). Every entry a restore materializes is written by that one restore; \
             anything lower is a pre-restore stamp that survived",
            section.section_id,
            section.last_version
        );
        // `first_version` is EXEMPT from the re-stamp, so it legitimately sits
        // below the seed and there is no oracle value to compare a
        // restored-ONLY section against. What still has to hold is
        // well-formedness: a `0` is the absent-column fold (a genuine restore
        // gap, per the triage note), and an inversion would break the
        // new-vs-modified classification both diff builders key on.
        assert!(
            section.first_version >= 1 && section.first_version <= section.last_version,
            "{context}: restored section {} carries a malformed version pair (first {}, last {}) \
             — `first_version` must be a real stamp (>= 1, never the absent-column fold) and \
             never above `last_version`",
            section.section_id,
            section.first_version,
            section.last_version
        );
    }
    for field in &restored.fields {
        if opts.ignored_field_keys.contains(&field.key_name.as_str()) {
            continue;
        }
        assert!(
            field.version >= seeded_seq,
            "{context}: restored field ({}, {:?}) carries version {}, below the seeded cursor \
             ({seeded_seq}) — a pre-restore stamp that survived",
            field.section_id,
            field.key_name,
            field.version
        );
    }
    for sample in &restored.time_samples {
        assert!(
            sample.version >= seeded_seq,
            "{context}: restored time sample ({}, {}) carries version {}, below the seeded \
             cursor ({seeded_seq})",
            sample.section_id,
            sample.time,
            sample.version
        );
    }
}

// ===========================================================================
// Driven-tail fixture (Epic `t1.dfh.57`, Task 7)
//
// The ONE composable step that produces a REAL captured tail — a lineage root,
// a layer whose every assembly is skipped (the unprovable-cut seam, via
// `unprovable_cut_client`), ompf-bearing commits, and the periodic loop's own
// capture fallback writing the artifact.
// Everything a `put_tail_generation` tail is not: the records correspond to
// commits that were actually applied, which is what Epic 4 (`t1.dfh.yj`) needs
// to replay one onto restored state.
//
// The per-file loop-driving plumbing (`poll_until` / `cancel_and_join` /
// `set_gate_timestamps`) deliberately stays file-local in the consuming suites;
// only the tail-PRODUCING step is shared.
// ===========================================================================

/// The atomic-read item limit [`fenced_path_client`] connects with. One item
/// is below the item bound of any layer with a section in it, so the atomic
/// read answers `TOO_LARGE` at its early-out for every such layer.
pub const FENCED_PATH_MAX_ITEMS: u64 = 1;

/// The byte half of [`FENCED_PATH_MAX_ITEMS`]'s forced-low limit.
pub const FENCED_PATH_MAX_BYTES: u64 = 1;

/// A client on the test store whose atomic-read limits are forced low
/// ([`FENCED_PATH_MAX_ITEMS`] / [`FENCED_PATH_MAX_BYTES`]), so every layer with
/// a section in it is above them and the checkpoint assembler reads it through
/// its fenced read (Spec SDD SR-11.5) instead of the atomic read.
///
/// For a fixture whose subject is the above-ceiling path. The limits are
/// copied from `Config` at connect and are not settable on an existing client,
/// hence a client of its own; it reads and writes the same store under the
/// caller's key schema, so the caller's `TestValkey` still owns the cleanup.
/// Being its own connection, it does NOT share `tv`'s unprovable-cut seam.
#[allow(dead_code)]
pub async fn fenced_path_client() -> ValkeyClient {
    crate::storage::snapshot_read_support::client_with_limits(
        FENCED_PATH_MAX_ITEMS,
        FENCED_PATH_MAX_BYTES,
    )
    .await
}

/// A clone of `tv`'s client with the test-only unprovable-cut seam armed for
/// `object_id` (`ValkeyClient::arm_unprovable_checkpoint_cut_for_test`,
/// `test-checkpoint-fault-injection`): every checkpoint assembly of that layer
/// pins its header sequence above every cut, so its proven read — run for
/// real — is judged regressed, the assembly persists nothing, and a loop with
/// the tail flag on falls back to tail capture.
///
/// The route to `AssemblyDecision::SkippedNotQuiesced`, and with it the
/// capture fallback, since every read proves its cut (Spec SDD SR-4.1,
/// SR-11.5): a hot writer alone no longer keeps the assembler from writing at
/// any layer size. The seam is shared by every clone of `tv`'s client, the
/// harness loop's included, and undone by
/// `disarm_unprovable_checkpoint_cut_for_test`.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[allow(dead_code)]
pub fn unprovable_cut_client(tv: &TestValkey, object_id: u64) -> ValkeyClient {
    let client = tv.client().clone();
    client.arm_unprovable_checkpoint_cut_for_test(object_id);
    client
}

/// The sequence a [`drive_captured_tail`] layer is created at, and the
/// `snap_seq` of the base planted as its lineage root.
///
/// **The coupling is load-bearing, not cosmetic** — the hardest-won lesson from
/// the in-crate precedent (`src/checkpoint/checkpointer.rs`'s `tail_tick_tests`,
/// its `S0` constant). The capture unit opens its read window at `{S0 + 1}-0`
/// where S0 is the newest generation in the STORE LISTING. Pairing a base at S0
/// with a header sequence of S0 means the window opens on exactly the writer's
/// first real commit. Any other pairing leaves a hole below the window's floor,
/// the classifier returns `Trimmed`, and the fixture silently exercises
/// `refused_trimmed` while its caller believes it covers success.
pub const CAPTURED_TAIL_SEED_SEQ: u64 = 92;

/// Sections seeded into a [`drive_captured_tail`] layer.
///
/// 120 is `case_d_unprovable_cut_ticks_unverified_and_persists_nothing`'s
/// number (`checkpointer_metrics_test.rs`): content every tail record can
/// address. Which arm a tick takes is the unprovable-cut seam's
/// ([`unprovable_cut_client`]), not the width's.
#[cfg(feature = "test-checkpoint-fault-injection")]
const CAPTURED_TAIL_SECTIONS: u64 = 120;

/// Stream retention for the driven layer's `LayerStore`.
///
/// Deliberately far above the harness's 4096: the writer commits continuously
/// for as long as the drive runs, and if `MAXLEN` trimming ever reached past the
/// capture window's floor the unit would refuse `refused_trimmed` — a fixture
/// artifact indistinguishable, at the assertion, from the wiring being broken.
#[cfg(feature = "test-checkpoint-fault-injection")]
const CAPTURED_TAIL_STREAM_MAXLEN: u64 = 1_000_000;

/// Commits the writer must land before the loop is allowed to start.
///
/// A state precondition polled off the writer's own counter, not a timing hack.
/// Without it the first tick races the writer's start-up and may find no
/// commit above the planted base for a tail to cover.
#[cfg(feature = "test-checkpoint-fault-injection")]
const CAPTURED_TAIL_WARM_UP_COMMITS: u64 = 20;

/// Wall-clock ceiling on the whole drive.
///
/// Comfortably inside `.config/nextest.toml`'s 120 s hard kill (a SIGKILLed test
/// prints nothing at all, losing exactly the diagnostics below). The in-crate
/// precedent observes the arm within one tick once the writer has committed;
/// 30 s is roughly an order of magnitude of headroom over that.
#[cfg(feature = "test-checkpoint-fault-injection")]
const CAPTURED_TAIL_DEADLINE: Duration = Duration::from_secs(30);

/// Drive a real captured tail for a fresh layer under `session_id`, returning
/// the `Tail` handle the periodic loop wrote.
///
/// The handle carries the layer's `object_id` and, in `snap_seq`, the envelope's
/// `to_seq`; the lineage root is a base at [`CAPTURED_TAIL_SEED_SEQ`], so a
/// caller can assert `from_seq == CAPTURED_TAIL_SEED_SEQ` on the decoded
/// envelope without threading extra return values.
///
/// What the fixture does, and why each step is mandatory:
///
/// 1. Creates a layer at `sequence_num = CAPTURED_TAIL_SEED_SEQ` with
///    [`CAPTURED_TAIL_SECTIONS`] sections over a
///    [`CAPTURED_TAIL_STREAM_MAXLEN`] stream — retention is what keeps the
///    window intact (see both constants).
/// 2. Hand-puts the lineage root base at the same sequence via
///    [`put_base_generation`] — without a base the capture refuses `no_base`
///    forever.
/// 3. Spawns a writer committing **field-set** deltas as fast as the store will
///    take them, each carrying a REAL `relay_ompf`. Both halves are
///    load-bearing: a field-only delta rides `update_layer`'s Lua fast path (a
///    structural command takes a slower path — the structural route on `lua`,
///    or the WATCH/MULTI/EXEC slow path on the rollback backends, which
///    commits roughly an order of magnitude less often), and an `ompf`-less
///    update entry is read by the classifier as an EMPTY commit by invariant —
///    a fixture built that way records real commits as no-ops and tests a lie.
/// 4. Waits for [`CAPTURED_TAIL_WARM_UP_COMMITS`], then spawns the real
///    periodic loop with `tail` on [`unprovable_cut_client`] armed for the
///    layer — without which every read would prove a cut and the first tick
///    would write a base — and polls the listing for a `Tail`. The seam is
///    disarmed again before returning, so a caller's later loop or assembly
///    of the layer is not skipped.
/// 5. Stops the writer and drains the loop before returning, so the artifact the
///    caller inspects is stable.
///
/// `tail.enabled` MUST be `true` — asserted, because a disabled config makes the
/// wait unsatisfiable and the 30 s deadline would report it as a race.
///
/// Compiled only under `test-checkpoint-fault-injection`, the seam's feature.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[allow(dead_code)]
pub async fn drive_captured_tail(
    tv: &TestValkey,
    store: &Arc<dyn CheckpointStore>,
    metrics: &SharedMetrics,
    session_id: &str,
    layer_name: &str,
    tail: TailCaptureConfig,
) -> CheckpointHandle {
    assert!(
        tail.enabled,
        "drive_captured_tail requires tail.enabled = true; with the flag off the loop's \
         SkippedNotQuiesced arm never reaches the capture path and this call could only ever \
         time out"
    );

    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        CAPTURED_TAIL_STREAM_MAXLEN,
    ));

    // ── 1. A wide layer at the seed sequence ────────────────────────────
    let mut state = ObjectState {
        sequence_num: CAPTURED_TAIL_SEED_SEQ,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let ids: Vec<u64> = (1..=CAPTURED_TAIL_SECTIONS).collect();
    for (order, sid) in ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("driven-tail-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (*sid, "seed-field".to_string()),
            Field {
                section_id: *sid,
                key_name: "seed-field".to_string(),
                value: FieldValue::Inline(vec![0x5E, 0xED]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
    }
    state.children.insert(0, ids);

    let object_id = layer_store
        .create_layer(
            session_id,
            layer_name,
            HAND_PUT_CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed for the driven-tail fixture");

    // ── 2. The lineage root, at the layer's own sequence ────────────────
    let base = put_base_generation(
        store,
        object_id,
        session_id,
        layer_name,
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;

    // ── 3. The hot writer ───────────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicU64::new(0));
    let writer_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let writer = {
        let stop = Arc::clone(&stop);
        let commits = Arc::clone(&commits);
        let writer_error = Arc::clone(&writer_error);
        let layer_store = Arc::clone(&layer_store);
        let session_id = session_id.to_string();
        let layer_name = layer_name.to_string();
        tokio::spawn(async move {
            let mut nonce = 10_000u64;
            while !stop.load(Ordering::Relaxed) {
                let delta = oodf_ffi::OwnedDelta {
                    set_fields: vec![oodf_ffi::SetField {
                        // Section 1 exists in every layer this fixture creates.
                        section_id: 1,
                        key_name: "driven-tail-field".to_string(),
                        // Distinct per commit, so a silently-dropped delta
                        // cannot be mistaken for one that was applied.
                        value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: nonce,
                    }],
                    ..Default::default()
                };
                let ompf =
                    Bytes::from(crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
                // No pacing sleep: the writer commits as fast as the backing
                // store will take it, so every tick finds commits to capture.
                if let Err(e) = layer_store
                    .update_layer(
                        &session_id,
                        &layer_name,
                        object_id,
                        Some(delta),
                        Some(ompf),
                        "driven-tail-writer",
                        0,
                    )
                    .await
                {
                    // Normal at the END of a fixture (the pool is torn down at
                    // test exit), but recorded either way: a writer that died on
                    // its second commit and one whose arm was simply not reached
                    // produce the same "no tail landed" symptom, and
                    // only this string tells them apart.
                    *writer_error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                    return;
                }
                commits.fetch_add(1, Ordering::Relaxed);
                nonce += 1;
            }
        })
    };

    let deadline = Instant::now() + CAPTURED_TAIL_DEADLINE;
    let writer_report = |commits: &Arc<AtomicU64>, err: &Arc<Mutex<Option<String>>>| -> String {
        let n = commits.load(Ordering::Relaxed);
        match err.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            Some(e) => format!("{n} writer commits, then died with: {e}"),
            None => format!("{n} writer commits, no writer error"),
        }
    };

    // Meet the loop with an already-hot layer.
    while commits.load(Ordering::Relaxed) < CAPTURED_TAIL_WARM_UP_COMMITS {
        assert!(
            Instant::now() < deadline,
            "driven-tail fixture never reached {CAPTURED_TAIL_WARM_UP_COMMITS} warm-up commits \
             within {CAPTURED_TAIL_DEADLINE:?} ({})",
            writer_report(&commits, &writer_error)
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // ── 4. The real periodic loop, tail path on ─────────────────────────
    // With the unprovable-cut seam armed for the layer, so every assembly is
    // skipped and the tick falls back to the capture.
    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(store),
        CheckpointLoopConfig {
            tick: Duration::from_secs(1),
            // Both gates open so every tick reaches the assembler; the capture
            // fallback is downstream of the assembly being SKIPPED, not of a
            // gate.
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            // Well above anything this fixture plants: the landed prune counts
            // LINEAGE UNITS (one per base, with the tails grouped above it),
            // and a low quota would be one more variable in a fixture whose
            // failures are already hard to read — the more so now that an
            // over-run quota can delete a tail as well as a base.
            retain_generations: 8,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(SERVICE_LABEL),
            tail,
        },
        Some(Arc::clone(metrics)),
        cancel.clone(),
    );

    let mut captured: Option<CheckpointHandle> = None;
    let mut last_listing: Vec<CheckpointHandle> = Vec::new();
    while Instant::now() < deadline {
        last_listing = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed during the driven-tail drive");
        if let Some(handle) = last_listing
            .iter()
            .find(|h| h.kind == CheckpointKind::Tail)
            .cloned()
        {
            captured = Some(handle);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // ── 5. Quiesce the fixture before anyone reads it ───────────────────
    stop.store(true, Ordering::Relaxed);
    let _ = writer.await;
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_handle).await;
    tv.client()
        .disarm_unprovable_checkpoint_cut_for_test(object_id);

    captured.unwrap_or_else(|| {
        panic!(
            "driven-tail fixture produced no Tail generation for object_id={object_id} within \
             {CAPTURED_TAIL_DEADLINE:?} ({}). Base planted at {base:?}. Last listing: \
             {last_listing:?}. The two usual causes: the seam did not fire (the assembler \
             wrote an ordinary base — look for a second Base in the listing), or \
             the capture refused (a refusal writes nothing, so read \
             live_edit_checkpoint_tail_total's refused_* cells via read_tail_counter to see \
             which arm)",
            writer_report(&commits, &writer_error)
        )
    })
}

// ===========================================================================
// Harness self-guards (Epic `t1.dfh.57`, Task 7 — item 8)
//
// The counter-repair parity guard above is a `fn` sibling tests call. These are
// real `#[tokio::test]`s because the claim is about the harness ITSELF and no
// consumer test would fail if it broke: a harness carrying a disabled first-base
// bootstrap still creates layers, still checkpoints, still recovers — it simply
// never produces the SR-7.1 lineage root, and every AC-5 test built on it would
// report "the bootstrap did not fire" indistinguishably from a real defect in
// the unit. `CounterRepairParityHarness`'s doc records how that class of gap
// arose the last time it was left to convention.
// ===========================================================================

/// Tick cadence for the two guards below — the BACKSTOP fence, not the primary
/// one.
///
/// An hour, so a second tick cannot land inside a test. The primary fence is
/// [`drain_harness_loop`]: `tokio::time::interval` fires immediately, but that
/// first pass runs on the spawned task's first poll, which the multi-thread
/// scheduler is free to place AFTER the test's `CreateLayer` — so "the immediate
/// tick happens before any layer exists" is not a fact, it is a race, and
/// observed as one (the negative arm below saw the loop's own `snap_seq = 1`
/// base). Draining the loop outright is what makes "the bootstrap is the only
/// possible writer" true rather than probable, for BOTH arms.
const BOOTSTRAP_GUARD_TICK_SECS: u64 = 3600;

/// Cancel the harness's periodic checkpoint loop and await its exit, so nothing
/// but the code under test can write a generation afterwards.
///
/// Swaps a no-op handle into `loop_handle` so the harness's drop stays safe —
/// the same `mem::replace` shape `recovery_mixed_kind_selection_test.rs` uses
/// for its prune fence. Bounded so a wedged loop fails the test promptly rather
/// than hanging the binary.
async fn drain_harness_loop(harness: &mut CheckpointHarness) {
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_task).await;
}

/// At the STOCK harness default — which is the binary default, `true` since
/// b.3n1 — a `CreateLayer` through the harness must land the bootstrap base at
/// `snap_seq = 1` in the harness's own store.
///
/// Two claims in one, and both are load-bearing. The first is the harness
/// contract: [`CheckpointHarnessOptions::tail`] must track the binary's flag
/// rather than carry a harness-local literal, so a stock harness exercises the
/// shipped posture. The second is behavioural — with the gate TRUE the
/// bootstrap actually writes.
///
/// Behavioural, deliberately — reading the `base_bootstrap:` field at the
/// construction site would prove only that *something* was assigned there, which
/// is exactly what the disabled stub also satisfies.
///
/// Synchronised on the shared `TaskTracker` rather than on a poll: `main.rs`
/// hangs the bootstrap task off the session-checkpoint tracker and
/// [`base_bootstrap_for_harness`] does the same, so a tracker that reports idle has
/// necessarily joined the bootstrap task if one was ever spawned. Combined with
/// [`drain_harness_loop`] that makes BOTH arms deterministic — no sleep, no
/// polling, and no "maybe it just had not landed yet" reading of the negative
/// arm below.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn harness_at_the_default_tail_flag_writes_the_first_base_bootstrap_generation() {
    let opts = CheckpointHarnessOptions::new(BOOTSTRAP_GUARD_TICK_SECS);
    assert!(
        opts.tail.enabled,
        "CheckpointHarnessOptions must default to the BINARY's tail flag, which is `true` since \
         b.3n1 — it must never carry a harness-local literal. A `false` here means the harness \
         stopped modelling the shipped deployment and every stock-harness test below silently \
         changed subject"
    );
    let mut harness = setup_checkpoint_harness_with_options(opts).await;
    drain_harness_loop(&mut harness).await;

    let (session_id, object_id) = crate::grpc::helpers::create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "first-base-harness-guard-session",
        "first-base-harness-guard-layer",
    )
    .await;
    wait_for_session_checkpoint_quiesced(&harness, &session_id, 10).await;

    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    let bootstrap = generations
        .iter()
        .find(|h| h.kind == CheckpointKind::Base && h.snap_seq == 1)
        .unwrap_or_else(|| {
            panic!(
                "with checkpoint_enabled && adapter != Disabled && checkpoint_tail_enabled, a \
                 CreateLayer through the harness MUST leave the SR-7.1 bootstrap base at \
                 snap_seq = 1 in the harness's store. Nothing at snap_seq = 1 means the harness \
                 installed a bootstrap that cannot write — a disabled gate, or a unit pointed at \
                 some store other than `harness.store`. The periodic loop was drained before the \
                 CreateLayer and cannot be the writer either way. Listing: {generations:?}"
            )
        });
    assert_eq!(
        bootstrap.object_id, object_id,
        "the bootstrap generation must be keyed on the object_id CreateLayer returned"
    );
}

/// With the tail flag PINNED OFF — the operator opt-out, and the posture every
/// suite in this module that predates the tail feature pins for itself — the
/// same CreateLayer must write NOTHING.
///
/// The other half of the guard, and the one that keeps the flag meaningful: the
/// SR-7.1 gate is a three-way conjunction whose third conjunct is
/// `checkpoint_tail_enabled`, so with it off the unit short-circuits before
/// allocating, spawning, or issuing a single Valkey command. If this arm ever
/// fails, every `wait_for_checkpoint` in a tails-off suite is now racing a
/// generation nobody asked for.
///
/// The flag is pinned HERE rather than inherited: since b.3n1 the default is
/// `true` (the sibling arm above is what pins that), so an inherited value would
/// make this arm assert the exact opposite of its own name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn harness_with_tails_pinned_off_writes_no_bootstrap_generation() {
    let opts = tails_off_options(BOOTSTRAP_GUARD_TICK_SECS);
    assert!(
        !opts.tail.enabled,
        "fixture precondition: this arm pins the flag off explicitly; it is vacuous otherwise"
    );
    let mut harness = setup_checkpoint_harness_with_options(opts).await;
    // Without this the loop's own immediate tick can land a `snap_seq = 1` base
    // and the assertion below fails on a generation the bootstrap did not write.
    drain_harness_loop(&mut harness).await;

    let (session_id, object_id) = crate::grpc::helpers::create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "first-base-harness-guard-off-session",
        "first-base-harness-guard-off-layer",
    )
    .await;
    wait_for_session_checkpoint_quiesced(&harness, &session_id, 10).await;

    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations.is_empty(),
        "with the tail flag pinned off the first-base bootstrap gate is false, so a CreateLayer \
         must leave the store empty for this object (the loop was drained first, so nothing else \
         could have written). Found {generations:?}"
    );
}

/// [`drive_captured_tail`] must actually produce a REAL captured tail, rooted
/// at the base it plants and covering the commits its writer landed.
///
/// A self-guard on shared infrastructure, and the reason it is worth its own
/// runtime: three sibling suites and Epic `t1.dfh.yj` build on this fixture, and
/// every failure mode it has is silent at the consumer. A fixture that drifted
/// into `refused_trimmed` (retention reaching past the window floor), or into an
/// ordinary base (the unprovable-cut seam stopped firing), or into
/// `refused_no_base` (the S0 coupling broken) would leave each consumer
/// reporting "the arm was never reached" — the exact symptom of a real defect in
/// the code under test. Pinning it here means the fixture fails in ONE place,
/// with a fixture-shaped message.
///
/// The claimed-range-equals-content assertion is the b.pw6 guard restated at
/// fixture level: an `ompf`-less update entry is read by the classifier as an
/// EMPTY commit by invariant, so a fixture whose writer forgot `relay_ompf`
/// would still yield a decodable envelope — with no-op records over real
/// content. Record count == span is what catches that.
///
/// Compiled only under `test-checkpoint-fault-injection` (the fixture's seam):
/// without the feature it is compiled out, not skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drive_captured_tail_produces_a_real_tail_rooted_at_the_seed_base() {
    let tv = TestValkey::setup().await;
    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("driven-tail-guard-session", "/stage", HAND_PUT_CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = "driven-tail-guard-layer";

    let tail_config = TailCaptureConfig {
        enabled: true,
        ..TailCaptureConfig::default()
    };
    let handle =
        drive_captured_tail(&tv, &store, &metrics, &session_id, layer_name, tail_config).await;

    assert_eq!(
        handle.kind,
        CheckpointKind::Tail,
        "drive_captured_tail must return a Tail handle; got {handle:?}"
    );
    assert!(
        handle.snap_seq > CAPTURED_TAIL_SEED_SEQ,
        "the captured tail's to_seq ({}) must sit strictly above the seed base at \
         {CAPTURED_TAIL_SEED_SEQ} — an equal value would mean the envelope covers nothing",
        handle.snap_seq
    );

    let blob = store
        .get_layer_checkpoint(&handle)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the handle the fixture returned must address a live artifact");
    let envelope = live_edit::checkpoint::decode_tail_envelope(&blob.bytes)
        .expect("the fixture's artifact must be a real tail envelope");
    assert_eq!(
        envelope.header().from_seq,
        CAPTURED_TAIL_SEED_SEQ,
        "the capture window must open at the seed base's sequence — a different from_seq means \
         the S0 / header-sequence coupling documented on CAPTURED_TAIL_SEED_SEQ has broken and \
         the fixture is silently exercising a trimmed prefix"
    );
    assert_eq!(
        envelope.header().to_seq,
        handle.snap_seq,
        "a tail's handle sequence IS its envelope to_seq"
    );
    assert_eq!(
        envelope.records().len() as u64,
        envelope.header().to_seq - envelope.header().from_seq,
        "claimed range must EQUAL carried content (b.pw6): {} record(s) for the span \
         ({}, {}]",
        envelope.records().len(),
        envelope.header().from_seq,
        envelope.header().to_seq
    );

    // The lineage root the fixture plants must still be there for a consumer to
    // restore from — a tail without its base is unrestorable by construction.
    let generations = store
        .list_layer_checkpoints(handle.object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == CAPTURED_TAIL_SEED_SEQ),
        "the seed base at {CAPTURED_TAIL_SEED_SEQ} must survive the drive. Listing: \
         {generations:?}"
    );
}
