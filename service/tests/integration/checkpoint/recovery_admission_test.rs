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

//! Concurrency tests for the per-`object_id` recovery admission discipline
//! (SDD SR-2.6). Subtask `t3.fgj.ya.qm.h7` under Task `t2.fgj.ya.qm`.
//!
//! `RecoveryHook` holds a K=1 `DashMap<u64, Arc<AdmissionEntry>>` keyed on
//! `object_id`. The map collapses N concurrent
//! [`RecoveryHook::try_restore_layer`] calls on the same `object_id` into
//! ONE [`apply_checkpoint`](live_edit::checkpoint::apply::apply_checkpoint)
//! invocation — the winner runs the full pipeline (decode plus
//! `write_object` plus `HSET last_checkpoint_seq`) while every loser
//! queues behind the K=1 mutex, observes the restored `obj:hdr` via the
//! post-acquire EXISTS re-check (step 4 of `try_restore_layer`), and
//! returns `Restored` without re-issuing the apply.
//!
//! Cross-object_id calls MUST NOT serialize: a separate `AdmissionEntry`
//! lives in the map under each `object_id`, so two restorers targeting
//! distinct objects must each run a full apply concurrently.
//!
//! The entry is reference-counted: the `AdmissionMapGuard` retires the
//! map entry only when the LAST in-flight restorer drops out. The map
//! must therefore drain back to zero after every concurrent burst — any
//! residual entry signals a refcount-decrement bug that would
//! monotonically leak memory under sustained recovery storms.
//!
//! ## Apply-count instrumentation
//!
//! We assert "ONE apply per object_id" by wrapping the in-memory
//! `CheckpointStore` with a [`CountingCheckpointStore`] that counts the
//! store calls the restore-source selection issues:
//! **`list_layer_checkpoints`** (the primary counter) and
//! `get_layer_checkpoint` (a corroborating second counter).
//!
//! Why the list call is the correct admission proxy: `try_restore_layer`
//! re-checks `obj:hdr` under the per-`object_id` admission mutex (step 4)
//! BEFORE it enters `RecoveryHook::select_restore_source` (step 5). The
//! losers that queue behind the K=1 mutex observe the restored header at
//! that re-check and return `Restored` without ever touching the store.
//! Selection then issues `list_layer_checkpoints` **unconditionally** as
//! its very first statement — no early return, no branch, precedes it. So
//! the list count is exactly the number of restorers admitted past the
//! re-check: a 1:1 proxy for "how many apply-checkpoint pipelines actually
//! ran", which is the load-bearing signal for the K=1 collapse.
//!
//! `get_layer_checkpoint` is counted as well, but it is corroboration
//! rather than the primary signal: selection skips it entirely when the
//! listing comes back empty or holds no `Base` generation, so it is NOT
//! 1:1 with admission in general. These fixtures seed exactly one base
//! generation per `object_id`, so here every admitted restorer does reach
//! it and both counters must read the same number. Asserting both is what
//! stops an implementation that lists twice per admission from hiding
//! behind a single get.
//!
//! Caveat on that equality: it also assumes nothing cancels selection
//! between the two calls. The whole `select_restore_source` list-then-get
//! pair sits inside one `Config::checkpoint_recovery_fetch_budget`
//! `tokio::time::timeout`, so a budget elapse can drop the future after
//! the list and before the get. A future latency-injecting adapter added
//! to this file would therefore break the `list_call_count ==
//! get_by_handle_call_count` correspondence without any regression in the
//! admission discipline itself; the counters would have to be asserted
//! independently against their expected values rather than against each
//! other.
//!
//! Counting `get_latest_layer_checkpoint` — this fixture's pre-SR-2.5
//! shape — no longer measures anything: the hook does not call it at all
//! (SDD SR-2.5 replaced the single `get_latest` with the
//! list-then-get-by-handle pair so a newest-generation `Tail` is never fed
//! to the reset-only decoder), so that counter would read zero on a
//! perfectly healthy K=1 collapse.
//!
//! A blunter alternative — counting `obj:hdr` `HSET`s via a Valkey
//! `MONITOR` stream — would also work, but the wrapper is local,
//! deterministic, and survives cluster-mode dispatching where MONITOR
//! does not. We deliberately do NOT wrap `LayerStore`: it is not behind
//! a trait, so wrapping would force a duplicated mock surface ~2k lines
//! wide. The store-side counters pin the exact invariant the K=1
//! discipline protects (no duplicate restore-source selection + decode +
//! write pipeline runs) without touching production code.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tokio::sync::Barrier;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    RecoveryHook, RecoveryOutcome,
};
use live_edit::config::Config;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

/// `UNLINK` every `le:{oid}:obj:*` key while leaving session metadata and the
/// layer mapping intact — promoted to `tests/common` by b.f4t sub-task 5 and
/// shared with the other checkpoint/gRPC recovery suites. This file used to
/// carry a byte-identical private copy whose doc-comment claimed a shared home
/// "would require a third helper module and a re-export dance"; sub-task 5
/// built exactly that home, so the claim (and the copy) are gone.
use crate::common::wipe_object_keys;

const SERVICE_LABEL: &str = "recovery-admission-test-replica";

// =============================================================================
// Counting checkpoint store wrapper.
//
// Composes `super::helpers::DelegatingCheckpointStore` for the verbatim
// trait delegation and overrides only the two calls the restore-source
// selection makes — `list_layer_checkpoints` (primary) and
// `get_layer_checkpoint` (corroborating) — with `AtomicUsize` counters. The
// post-b.dzd shape: file-local newtype, not a 200-line inline impl.
// =============================================================================

/// Wraps an [`InMemoryCheckpointStore`] and counts the two store calls
/// `RecoveryHook::select_restore_source` issues: every
/// `list_layer_checkpoints` and every `get_layer_checkpoint`. The list
/// counter is the load-bearing instrumentation for the "exactly N apply
/// pipelines ran" assertion below (Case A: N=1, Case B: N=2); the get
/// counter corroborates it.
///
/// Why `list_layer_checkpoints` is the right thing to count:
/// `try_restore_layer` performs the post-acquire `obj:hdr` EXISTS re-check
/// (step 4) BEFORE entering `select_restore_source` (step 5). The losers
/// that queue behind the K=1 mutex observe the restored header via that
/// re-check and return `Restored` without ever touching the store. The
/// admitted restorer then issues `list_layer_checkpoints` unconditionally
/// as selection's first statement — nothing between the re-check and the
/// list can skip it. So the count of `list_layer_checkpoints` calls is
/// exactly the count of recoveries that ran the apply pipeline through to
/// `write_object`.
///
/// Why `get_layer_checkpoint` is only corroboration: selection returns
/// `Ok(None)` before the get when the listing is empty or holds no `Base`
/// generation, so the get count is NOT 1:1 with admission in general. It
/// is 1:1 *here* because `create_layer_with_checkpoint` seeds exactly one
/// base generation per `object_id`, so both counters must read the same
/// number — which is what denies cover to an implementation that lists
/// twice per admitted restorer and gets once.
struct CountingCheckpointStore {
    delegate: super::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    list_calls: AtomicUsize,
    get_by_handle_calls: AtomicUsize,
}

impl CountingCheckpointStore {
    fn new(inner: Arc<InMemoryCheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            list_calls: AtomicUsize::new(0),
            get_by_handle_calls: AtomicUsize::new(0),
        }
    }

    fn list_call_count(&self) -> usize {
        self.list_calls.load(Ordering::Acquire)
    }

    fn get_by_handle_call_count(&self) -> usize {
        self.get_by_handle_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for CountingCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    // Deliberately UNCOUNTED: since SR-2.5 the recovery hook does not call
    // `get_latest_layer_checkpoint` at all (selection is
    // `list_layer_checkpoints` -> newest `Base` handle ->
    // `get_layer_checkpoint`). Forwarded verbatim so the wrapper stays a
    // faithful pass-through for any other caller.
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its
    // trait default returns `NotImplemented`, so omitting the arm would not
    // leave an un-instrumented pass-through, it would break selection's
    // second call outright and fail these tests on OUTCOME rather than on
    // the counter (reading like an unrelated hook bug).
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.get_by_handle_calls.fetch_add(1, Ordering::AcqRel);
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        // Increment BEFORE the inner call so a concurrent call that races
        // here cannot mask the counter from observers. The exact ordering
        // (pre- vs post-inner-call) does not matter for correctness: the
        // counter only needs to capture whether the call was issued at
        // all, not its outcome.
        self.list_calls.fetch_add(1, Ordering::AcqRel);
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

    // The four methods below carry non-trivial trait defaults. This newtype
    // is a method-by-method `impl` over a held `DelegatingCheckpointStore`
    // FIELD, not a blanket impl, so an omitted arm here inherits the trait
    // default rather than the skeleton's forwarding — a silent behaviour
    // SUBSTITUTION (`Ok(empty)`, `Ok(vec![])`, `Ok(0)`, the local publish
    // bound) that no assertion in this file would fail on. See the matching
    // note on `helpers::DelegatingCheckpointStore`.
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

    fn publish_worst_case(&self) -> std::time::Duration {
        self.delegate.publish_worst_case()
    }
}

// =============================================================================
// Test-config + helpers.
// =============================================================================

/// Build a `Config` with `checkpoint_enabled=true` and the small set of fields
/// the recovery + apply path actually reads. Mirrors the local helper in
/// `tests/integration/storage/checkpoint_apply_test.rs` — duplicated here
/// instead of lifted to a shared module because adding a fresh field to
/// `Config` should produce two compile errors (one per consumer) so both
/// helpers stay aligned with the production struct.
fn test_config() -> Config {
    Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url: String::new(),
        valkey_pool_size: 1,
        subscriber_pool_size: 1,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
        subscriber_max_concurrent_catchups: 64,
        subscriber_admission_max_waiters: None,
        maintenance_pool_size: 1,
        catchup_snapshot_read_concurrency:
            live_edit::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
        catchup_fenced_read_concurrency: live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
        catchup_build_memory_budget_mib: live_edit::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
        snapshot_pool_size: 1,
        snapshot_atomic_read_max_items: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        snapshot_atomic_read_max_bytes: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
        structural_cache_max_mib: live_edit::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
        structural_cache_audit_rate: live_edit::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
        structural_cache_audit_above_ceiling: false,
        valkey_connect_timeout_secs: 5,
        valkey_command_timeout_ms: 5000,
        valkey_reconnect_max_delay_ms: 30_000,
        valkey_auth_mode: live_edit::config::ValkeyAuthMode::Static,
        valkey_token_file: None,
        valkey_token_username: None,
        valkey_token_refresh_lead_secs: 180,
        valkey_iam_username: None,
        valkey_iam_region: None,
        valkey_iam_cluster_host: None,
        valkey_iam_service: live_edit::config::ValkeyIamService::MemoryDb,
        storage_service_endpoint: None,
        auth_timeout_ms: 1000,
        auth_cache_ttl_secs: 60,
        auth_cache_negative_ttl_secs: 30,
        auth_userinfo_negative_ttl_secs: 5,
        auth_cache_max_entries: 1_000,
        userinfo_endpoint: None,
        max_layer_size: 1_048_576,
        upload_memory_budget_mib:
            live_edit::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
        upload_admission_wait_ms:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
        upload_admission_max_waiters:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
        layer_stream_maxlen: 4096,
        layer_stream_max_bytes: live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
        presence_broadcast_capacity: 4096,
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        shutdown_timeout_secs: 1,
        shutdown_overall_budget_secs: 5,
        shutdown_stream_end_rate: 25,
        shutdown_stream_end_max_window_secs: 20,
        log_level: "warn".to_string(),
        log_format: live_edit::config::LogFormat::Json,
        trace_filter: live_edit::config::DEFAULT_TRACE_FILTER.to_string(),
        sweeper_interval_secs: 15,
        session_list_timeout_ms: 1000,
        rpc_deadline_ms: 5000,
        service_label: SERVICE_LABEL.to_string(),
        valkey_memory_bytes: None,
        ext_release_max_bytes: None,
        ext_reconcile_interval_secs: 3600,
        auth_max_attempts: 3,
        auth_retry_initial_delay_ms: 100,
        authn_enabled: true,
        authz_enabled: false,
        userinfo_service_enabled: true,
        test_only_dangerously_allow_unvalidated_tokens: false,
        health_probe_interval_ms: 5_000,
        health_probe_timeout_ms: 2_000,
        auth_protocol: live_edit::config::AuthProtocol::Rest,
        update_layer_retry_max_attempts: 128,
        update_layer_per_object_concurrency: 16,
        update_layer_backend: live_edit::config::UpdateLayerBackend::Lua,
        ext_value_verify: live_edit::config::ExtValueVerifyMode::Enforce,
        session_retry_max_attempts: 32,
        session_retry_initial_delay_ms: 1,
        pod_memory: live_edit::config::PodMemoryPlan::reference(),
        session_tokens_sweep_interval_secs: 60,
        checkpoint_enabled: true,
        checkpoint_adapter: live_edit::checkpoint::CheckpointAdapter::InMemory,
        checkpoint_interval_secs: 60,
        checkpoint_quiesce_secs: 30,
        checkpoint_tick_secs: 10,
        checkpoint_retain_generations: 2,
        checkpoint_seq_safety_margin: 1000,
        checkpoint_counters_sync_secs: 60,
        checkpoint_reconcile_interval_secs: 3600,
        checkpoint_counter_safety_margin: 1000,
        checkpoint_tail_enabled: false,
        checkpoint_tail_max_chain: 8,
        checkpoint_tail_max_entries: 4096,
        checkpoint_tail_max_bytes: 67_108_864,
        checkpoint_local_disk_path: None,
        checkpoint_storage_api_endpoint: None,
        checkpoint_base_path: None,
        checkpoint_connect_timeout_ms: 5_000,
        checkpoint_request_timeout_ms: 30_000,
        checkpoint_ext_mirror_workers: 2,
        checkpoint_ext_mirror_queue_capacity: 1_000,
        service_token_file: None,
        service_token_poll_interval_secs: 15,
        service_token_client_credentials_enabled: false,
    }
}

fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL))
}

/// Probe the live label-cell counter value for a given outcome label on the
/// `live_edit_checkpoint_recovery_attempts_total{result, service}` metric.
fn metric_value(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Build the minimal valid reset-delta OMPF the apply path accepts.
///
/// The apply path requires `is_diff=true && base_version=0` (the
/// assembler's `DiffPlan::FullContent` reset shape; see
/// `src/checkpoint/apply.rs::decode_checkpoint_ompf`). An OMPF carrying
/// an empty reset delta is sufficient — the recovery test only needs the
/// apply pipeline to land an `obj:hdr` so the next restorer's
/// post-acquire EXISTS re-check observes it. We deliberately do NOT
/// populate sections/fields here: the test surface is the K=1 admission
/// collapse, not the materialized state contents.
fn build_minimal_reset_ompf() -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        ..OwnedDelta::default()
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta must succeed");
    let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize must succeed");
    Bytes::from(ompf)
}

/// Helper: create a fresh session and return its id.
async fn create_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "recovery-admission-session",
            "/stage",
            "recovery-admission-user",
        )
        .await
        .expect("session creation must succeed")
}

/// Helper: create a populated layer, persist its OMPF checkpoint under the
/// `CountingCheckpointStore`'s inner [`InMemoryCheckpointStore`], then wipe
/// the live `obj:*` keys so a subsequent `try_restore_layer` actually runs
/// the apply pipeline. Returns the `object_id` of the layer.
async fn create_layer_with_checkpoint(
    tv: &TestValkey,
    layer_store: &LayerStore,
    inner_store: &InMemoryCheckpointStore,
    session_id: &str,
    layer_name: &str,
) -> u64 {
    let initial = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = layer_store
        .create_layer(
            session_id,
            layer_name,
            "recovery-admission-user",
            &initial,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");

    // Seed a checkpoint for the layer. The session/layer identity must
    // match the live mapping the apply path resolves against, otherwise
    // the SR-2.4 live-mapping guard would short-circuit with NotFound and
    // we would never observe the K=1 collapse we are testing.
    let meta = CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms: 1_700_000_000_000,
        snap_seq: Some(5),
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: "recovery-admission-user".to_string(),
        server_ts: 1_700_000_000_000,
        ..CheckpointMeta::default()
    };
    inner_store
        .put_layer_checkpoint(oid, 5, build_minimal_reset_ompf(), meta)
        .await
        .expect("seed checkpoint must succeed");

    // Wipe the live obj:* keys so the recovery hook actually runs the
    // apply pipeline (the post-acquire EXISTS re-check would otherwise
    // observe an existing header and short-circuit every restorer to
    // Restored without exercising the apply path — robbing the test of
    // any signal about the K=1 collapse).
    wipe_object_keys(tv, oid).await;

    oid
}

// =============================================================================
// Case A — same-object_id race collapses to ONE apply.
// =============================================================================

/// Fire N=10 concurrent `try_restore_layer` calls against the same
/// `object_id`. Per SR-2.6 the K=1 admission discipline must collapse the
/// burst to exactly ONE trip through the restore-source selection
/// (`list_layer_checkpoints` -> newest `Base` handle ->
/// `get_layer_checkpoint`) plus `write_object`; the other 9 must observe
/// the restored `obj:hdr` via the post-acquire EXISTS re-check (step 4 of
/// `try_restore_layer`) and report `Restored` without touching the
/// underlying store.
///
/// Pre-fix regression mode (K removed or weakened): all 10 restorers would
/// race through `select_restore_source` + `apply_checkpoint`'s own EXISTS
/// short-circuit, producing somewhere between 1 and 10 store listings
/// depending on scheduling. The exact-equals-1 assertion makes the K=1
/// invariant a bright-line falsifier rather than a schedule-dependent
/// maybe.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn same_object_id_race_collapses_to_one_apply() {
    const RESTORERS: usize = 10;

    let tv = TestValkey::setup().await;
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let session_id = create_session(&tv).await;
    let inner = Arc::new(InMemoryCheckpointStore::new());
    let oid = create_layer_with_checkpoint(
        &tv,
        &layer_store,
        &inner,
        &session_id,
        "recovery-admission-A-layer",
    )
    .await;

    // Wrap the in-memory store with the counting adapter. `Arc::clone`s
    // share the same atomic counter so the test thread can observe the
    // count after the join.
    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    let metrics = make_metrics();
    let hook = Arc::new(RecoveryHook::new(
        store,
        Arc::clone(&layer_store),
        Arc::new(test_config()),
        metrics.clone(),
    ));

    // `Barrier::new(RESTORERS)` releases every spawned task at the same
    // instant so the contention window the K=1 mutex is supposed to
    // collapse is actually exercised. Pre-`barrier.wait().await`, the
    // first spawned task could finish its entire admission-acquire +
    // apply pipeline before any later task even entered
    // `try_restore_layer` — that would let the post-acquire EXISTS
    // re-check on the followers fire on a no-contention path, making
    // the test vacuously pass. The barrier forces the path through the
    // contended mutex.
    let barrier = Arc::new(Barrier::new(RESTORERS));

    let mut handles = Vec::with_capacity(RESTORERS);
    for _ in 0..RESTORERS {
        let hook = Arc::clone(&hook);
        let barrier = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            hook.try_restore_layer(oid).await
        }));
    }

    let mut restored_count = 0_usize;
    let mut non_restored: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("restorer task must not panic") {
            RecoveryOutcome::Restored => restored_count += 1,
            other => non_restored.push(format!("{other:?}")),
        }
    }

    assert!(
        non_restored.is_empty(),
        "every concurrent restorer MUST observe Restored (winner via apply, \
         losers via post-acquire EXISTS re-check); got {} non-Restored outcomes: \
         {non_restored:?}",
        non_restored.len()
    );
    assert_eq!(
        restored_count, RESTORERS,
        "expected all {RESTORERS} restorers to report Restored; got {restored_count}"
    );

    // Apply-count assertion: exactly ONE `list_layer_checkpoints` call
    // must have reached the store. The K=1 mutex serializes restorers,
    // and the post-acquire `obj:hdr` EXISTS re-check in
    // `try_restore_layer` (step 4) returns `Restored` for the losers
    // BEFORE step 5 enters `select_restore_source` — whose first and
    // unconditional statement is that list. So the list count is exactly
    // the number of admitted restorers. Pre-fix regression (semaphore not
    // wired) would produce >1; a re-check ordering bug
    // (select-before-check) would also produce >1.
    let apply_count = counting.list_call_count();
    assert_eq!(
        apply_count, 1,
        "Case A: exactly ONE list_layer_checkpoints call must reach the store \
         under N={RESTORERS} concurrent restorers on the same object_id — the K=1 \
         admission discipline (SR-2.6) collapses the burst to one apply, and the \
         under-mutex obj:hdr re-check short-circuits the losers BEFORE selection \
         issues its unconditional list. Got apply_count={apply_count}; if >1 the \
         per-object_id admission mutex is not serializing restorers and the \
         recovery pipeline is duplicating work."
    );

    // Corroborating counter: the one admitted restorer proceeds from the
    // list to exactly one `get_layer_checkpoint` on the selected base
    // handle (this fixture seeds exactly one base generation, so the get
    // is reached). Asserting both counters denies cover to an
    // implementation that lists twice per admitted restorer and gets once.
    let get_count = counting.get_by_handle_call_count();
    assert_eq!(
        get_count, 1,
        "Case A: exactly ONE get_layer_checkpoint call must reach the store — the \
         single admitted restorer selects the newest Base handle from its one \
         listing and fetches it once. Got get_count={get_count} against \
         apply_count={apply_count}; a mismatch between the two means selection is \
         issuing an unbalanced number of round trips per admitted restorer."
    );

    // Final-state assertion: the apply pipeline landed the restored
    // `obj:hdr`. This pairs with the apply-count check above — proving
    // that the ONE apply we counted actually wrote state (not just
    // fetched the blob and bailed out). If the apply landed but the
    // counter recorded zero, this assertion would still pass while the
    // earlier one would fail; if the counter recorded 1 but the apply
    // silently dropped the write, this assertion would fail. Both
    // together pin "exactly one full restore happened".
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header_exists: bool = tv
        .client()
        .pool
        .exists(&header_key)
        .await
        .expect("EXISTS obj:hdr must succeed");
    assert!(
        header_exists,
        "obj:hdr must be present after concurrent restore on object_id={oid}; \
         the winner's apply pipeline did not materialize the header"
    );

    // Metric assertion: every restorer (winner + losers) increments the
    // `restored` cell. The losers' post-acquire EXISTS re-check
    // hits the `Ok(true)` arm and bumps the metric (recovery.rs step 4).
    let restored_metric = metric_value(&metrics, "restored");
    assert_eq!(
        restored_metric, RESTORERS as u64,
        "live_edit_checkpoint_recovery_attempts_total{{result=\"restored\"}} must equal \
         the restorer count ({RESTORERS}); got {restored_metric}. Each restorer (the \
         winner running the full apply AND the losers short-circuiting at the \
         post-acquire EXISTS re-check) records its outcome separately."
    );
    // Every restorer should land in the `restored` arm; the other arms
    // must stay at zero.
    assert_eq!(
        metric_value(&metrics, "not_recoverable"),
        0,
        "no restorer should land in not_recoverable on the happy-path race"
    );
    assert_eq!(
        metric_value(&metrics, "error"),
        0,
        "no restorer should land in error on the happy-path race"
    );

    // Case C precondition (asserted again under Case B below): the
    // admission map must drain back to zero after every concurrent
    // restorer drops out. A residual entry signals an
    // `AdmissionMapGuard` refcount-decrement bug that would
    // monotonically leak entries under sustained recovery storms.
    assert_eq!(
        hook.admission_len(),
        0,
        "Case C (post-Case-A): admission DashMap must drain to zero after every \
         concurrent restorer for the shared object_id drops out; got len={}",
        hook.admission_len()
    );
}

// =============================================================================
// Case B — different object_ids run concurrently, no false serialization.
// =============================================================================

/// Two distinct populated+checkpointed layers, each wiped, with 5
/// concurrent restorers per layer (10 total). The per-`object_id`
/// admission map keys on `object_id`, so the two oids must run their
/// apply pipelines IN PARALLEL — neither should serialize on the other's
/// K=1 mutex.
///
/// Apply-count assertion: exactly TWO `list_layer_checkpoints` calls
/// (one per oid). Both must produce `Restored` outcomes across all 10
/// callers (5 winners-by-collapse + 5 losers per oid, but the count is
/// counted per oid: one of the 5 restorers per oid is the winner, the
/// other 4 short-circuit on the re-check).
///
/// Pre-fix regression mode where the admission map keyed on a constant
/// (or a singleton mutex covered all oids) would serialize all 10
/// restorers behind one mutex AND the apply-count would still be 2 — so
/// the apply-count alone cannot distinguish that mode. We assert the map
/// drains to zero AND every restorer succeeds; the former rejects an
/// entry-leak regression and the latter rejects a "different-oid restorers
/// deadlock or starve" regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn distinct_object_ids_run_concurrently_no_false_serialization() {
    const RESTORERS_PER_OID: usize = 5;
    const OID_COUNT: usize = 2;
    const TOTAL_RESTORERS: usize = RESTORERS_PER_OID * OID_COUNT;

    let tv = TestValkey::setup().await;
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let session_id = create_session(&tv).await;
    let inner = Arc::new(InMemoryCheckpointStore::new());

    let oid_a = create_layer_with_checkpoint(
        &tv,
        &layer_store,
        &inner,
        &session_id,
        "recovery-admission-B-layer-A",
    )
    .await;
    let oid_b = create_layer_with_checkpoint(
        &tv,
        &layer_store,
        &inner,
        &session_id,
        "recovery-admission-B-layer-B",
    )
    .await;
    assert_ne!(
        oid_a, oid_b,
        "monotonic object_id counter must allocate distinct ids for two layers"
    );

    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    let metrics = make_metrics();
    let hook = Arc::new(RecoveryHook::new(
        store,
        Arc::clone(&layer_store),
        Arc::new(test_config()),
        metrics.clone(),
    ));

    // One barrier across all 10 tasks so cross-oid contention is real:
    // the two object_ids' restorer bursts hit the recovery hook
    // simultaneously and the per-oid K=1 mutexes must not block each
    // other. A separate barrier per oid would let oid A's burst finish
    // before oid B's burst started, vacuously passing.
    let barrier = Arc::new(Barrier::new(TOTAL_RESTORERS));

    let mut handles = Vec::with_capacity(TOTAL_RESTORERS);
    for oid in [oid_a, oid_b] {
        for _ in 0..RESTORERS_PER_OID {
            let hook = Arc::clone(&hook);
            let barrier = Arc::clone(&barrier);
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                (oid, hook.try_restore_layer(oid).await)
            }));
        }
    }

    let mut restored_per_oid: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::new();
    let mut non_restored: Vec<(u64, String)> = Vec::new();
    for h in handles {
        let (oid, outcome) = h.await.expect("restorer task must not panic");
        match outcome {
            RecoveryOutcome::Restored => {
                *restored_per_oid.entry(oid).or_insert(0) += 1;
            }
            other => non_restored.push((oid, format!("{other:?}"))),
        }
    }

    assert!(
        non_restored.is_empty(),
        "every restorer across both object_ids MUST observe Restored; got \
         {} non-Restored outcomes: {non_restored:?}",
        non_restored.len()
    );
    assert_eq!(
        restored_per_oid.get(&oid_a).copied().unwrap_or(0),
        RESTORERS_PER_OID,
        "object_id A ({oid_a}) must see all {RESTORERS_PER_OID} restorers report Restored"
    );
    assert_eq!(
        restored_per_oid.get(&oid_b).copied().unwrap_or(0),
        RESTORERS_PER_OID,
        "object_id B ({oid_b}) must see all {RESTORERS_PER_OID} restorers report Restored"
    );

    // Apply-count assertion: exactly TWO `list_layer_checkpoints` calls
    // reach the store — one per oid. Each oid's K=1 mutex collapses its
    // 5-restorer burst to one admitted restorer, and each admitted
    // restorer issues the list exactly once (unconditionally, immediately
    // after the under-mutex `obj:hdr` re-check its 4 losers short-circuit
    // on).
    let apply_count = counting.list_call_count();
    assert_eq!(
        apply_count, OID_COUNT,
        "Case B: exactly TWO list_layer_checkpoints calls must reach the store \
         (one per oid); got apply_count={apply_count}. The per-object_id admission \
         discipline must collapse each oid's burst to ONE apply independently — \
         neither shared serialization (all 10 behind one mutex, still apply_count=2 \
         but indistinguishable from the correct path here) nor missing collapse \
         (apply_count > 2 because the K=1 mutex per oid is broken)."
    );

    // Corroborating counter: each oid's admitted restorer proceeds from
    // its listing to exactly one `get_layer_checkpoint` on the selected
    // base handle (both fixtures seed exactly one base generation), so the
    // two counters must agree at TWO.
    let get_count = counting.get_by_handle_call_count();
    assert_eq!(
        get_count, OID_COUNT,
        "Case B: exactly TWO get_layer_checkpoint calls must reach the store (one \
         per oid); got get_count={get_count} against apply_count={apply_count}. A \
         mismatch between the two means selection is issuing an unbalanced number \
         of round trips per admitted restorer."
    );

    // Final-state assertion: BOTH layers' `obj:hdr` must be present
    // after the concurrent restore. Pairs with the apply-count check
    // above — confirms each per-oid winner actually landed its
    // write_object call, not just fetched the blob.
    for oid in [oid_a, oid_b] {
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let header_exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr must succeed");
        assert!(
            header_exists,
            "obj:hdr must be present after concurrent restore on object_id={oid}; \
             the winner's apply pipeline for this oid did not materialize the header"
        );
    }

    // Metric assertion: every restorer (10 total across both oids) lands
    // in the `restored` cell. Operator dashboards depend on this being
    // an exact-per-attempt counter, not deduplicated by oid.
    let restored_metric = metric_value(&metrics, "restored");
    assert_eq!(
        restored_metric, TOTAL_RESTORERS as u64,
        "live_edit_checkpoint_recovery_attempts_total{{result=\"restored\"}} must equal \
         the total restorer count ({TOTAL_RESTORERS}); got {restored_metric}. The \
         per-attempt accounting is the load-bearing operator-signal contract for \
         the recovery rate alert."
    );
    assert_eq!(
        metric_value(&metrics, "not_recoverable"),
        0,
        "no restorer should land in not_recoverable on the happy-path race"
    );
    assert_eq!(
        metric_value(&metrics, "error"),
        0,
        "no restorer should land in error on the happy-path race"
    );

    // Case C: the admission map must drain to zero after every restorer
    // for BOTH oids has dropped out. The map carries entries for both
    // oid_a and oid_b during the burst (peak DashMap size = 2 + per-oid
    // refcounts); after every guard drops, len() must return to 0.
    assert_eq!(
        hook.admission_len(),
        0,
        "Case C (post-Case-B): admission DashMap must drain to zero after every \
         concurrent restorer across BOTH object_ids drops out; got len={}. A \
         non-zero residual means the AdmissionMapGuard's refcount-decrement \
         logic dropped an entry without removing the map slot, which would \
         monotonically leak entries under recovery storms.",
        hook.admission_len()
    );
}
