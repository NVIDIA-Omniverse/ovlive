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

//! Integration tests for `apply_checkpoint` (Task `t2.fgj.ya.n9`).
//!
//! `apply_checkpoint` is the load-bearing recovery primitive for SR-2.4 /
//! SR-2.5 / SR-2.6. It composes:
//!
//! 1. `EXISTS le:{oid}:obj:hdr` — short-circuit `Ok(())` when a sibling replica
//!    already landed the recovery.
//! 2. The SR-2.4 identity guard — the live
//!    `le:{meta.session_id}:layer:{hex(meta.layer_name)}` mapping, named by the
//!    blob's OWN identity, must resolve to `object_id` (b.b2j; previously
//!    "some mapping anywhere resolves to `object_id`", which could not tell a
//!    reused id apart from a live one).
//! 3. `decode_checkpoint_ompf` — OMPF / OODF wire-shape validation.
//! 4. SR-2.5 sequence-number advance: stamp `state.sequence_num` =
//!    `max(snap_seq, max_seq_observed.unwrap_or(0)) + band` — the band is
//!    an ADDITIVE guard above the restore floor, not an alternative floor
//!    (b.8y5), and is sized `max(safety_margin,
//!    meta.observed_commit_span)` so the configured knob is a floor rather
//!    than the whole answer (b.vnw).
//! 5. `LayerStore::write_object` — MULTI/EXEC materialization on the txn pool.
//! 6. Post-write `HSET le:{oid}:obj:hdr last_checkpoint_seq = snap_seq` so the
//!    periodic checkpoint loop's skip-if-no-commits gate can fire on the
//!    restored layer.
//!
//! Each test below pins one of the six invariants above (or a combination)
//! against a live Valkey. The fixture shape across tests is uniform:
//!
//! - `TestValkey::setup().await` for an isolated key namespace.
//! - A `LayerStore` over that namespace.
//! - A `SessionStore` for `create_test_session`.
//! - A `rich_object_state()` so the OMPF round-trip exercises sections,
//!   children, fields, and time samples (the load-bearing primitive surfaces
//!   of `write_object`).
//! - The OMPF blob built via the production assembler
//!   `assemble_layer_checkpoint` against an in-memory `CheckpointStore` sink —
//!   keeps the round-trip authentic (the apply path consumes the exact byte
//!   shape the assembler writes) without forcing tests to hand-construct
//!   `oodf_ffi::OwnedDelta` for every variant.
//! - `crate::common::wipe_object_keys` UNLINKs every `le:{oid}:obj:*` key while
//!   leaving the session and layer-mapping keys intact — emulating the post-loss
//!   / pre-recovery keyspace shape the apply path runs against, and asserting the
//!   wipe actually landed.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use ordered_float::OrderedFloat;
use tokio::sync::Barrier;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::checkpoint::assembler::{assemble_layer_checkpoint, AssemblyDecision};
use live_edit::checkpoint::{CheckpointError, CheckpointMeta, CheckpointStore};
use live_edit::config::Config;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{LayerStore, HEADER_FIELD_CHS_COMPLETE, HEADER_FIELD_COMMIT_CHAIN};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::common::read_header_u64;

const SERVICE_LABEL: &str = "checkpoint-apply-test-replica";

/// Helper: build a `MockExternalValueStore` for `create_layer` setup calls.
/// The `rich_object_state` payloads use `FieldValue::Inline`, so the mock store
/// receives no `store_values` calls in practice — but `create_layer`'s
/// signature requires the trait object, so the mock keeps the construction
/// path uniform.
fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Helper: build a `LayerStore` from a `TestValkey` fixture.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

/// Helper: build a `SessionStore` from a `TestValkey` fixture.
fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: create a session and return its id.
async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("apply-test-session", "/stage", "apply-test-user")
        .await
        .expect("session creation must succeed")
}

/// A populated `ObjectState` exercising sections, children, fields, and time
/// samples — the four leaf shapes `write_object` MULTI/EXECs and the four
/// shapes `read_object_state_for_test` pipelines back out.
///
/// Mirrors the `rich_object_state` helper in `diff_read_test.rs` so the apply
/// round-trip exercises the same data surface as the existing snapshot-load
/// regression coverage. Two top-level sections (sec 1 root child, sec 2 child
/// of 1), two fields, two time samples.
fn rich_object_state() -> ObjectState {
    let mut state = ObjectState {
        object_id: 0, // will be assigned by LayerStore::create_layer
        sequence_num: 7,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };

    let sec1 = Section {
        section_id: 1,
        parent_id: 0,
        name: "World".to_string(),
        section_type: 1,
        section_order: 0,
        first_version: 1,
        last_version: 5,
    };
    let sec2 = Section {
        section_id: 2,
        parent_id: 1,
        name: "Cube".to_string(),
        section_type: 2,
        section_order: 1,
        first_version: 2,
        last_version: 7,
    };
    state.sections.insert(1, sec1);
    state.sections.insert(2, sec2);

    // Children: root has [1], section 1 has [2].
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);

    let field1 = Field {
        section_id: 1,
        key_name: "color".to_string(),
        value: FieldValue::Inline(vec![0xFF, 0x00, 0x80]),
        set_order: 1,
        authoring_set_order: 1,
        version: 3,
    };
    let field2 = Field {
        section_id: 2,
        key_name: "size".to_string(),
        value: FieldValue::Inline(vec![0x42]),
        set_order: 2,
        authoring_set_order: 2,
        version: 5,
    };
    state
        .fields
        .insert((field1.section_id, field1.key_name.clone()), field1);
    state
        .fields
        .insert((field2.section_id, field2.key_name.clone()), field2);

    let ts1 = TimeSample {
        section_id: 1,
        time: OrderedFloat(0.0),
        value: FieldValue::Inline(vec![0xAA, 0xBB]),
        version: 2,
    };
    let ts2 = TimeSample {
        section_id: 2,
        time: OrderedFloat(1.5),
        value: FieldValue::Inline(vec![0xCC]),
        version: 6,
    };
    state.time_samples.insert((ts1.section_id, ts1.time), ts1);
    state.time_samples.insert((ts2.section_id, ts2.time), ts2);

    state
}

/// Structural seed for the authoring-rank round-trip test: the same section /
/// children / time-sample shape as [`rich_object_state`], but with **no fields**
/// and `next_set_order == 0`.
///
/// Fields are deliberately omitted here and materialized instead via a burst of
/// `LayerStore::update_layer` commits in
/// `apply_checkpoint_preserves_per_field_authoring_rank`, so that each field's
/// stored CUMULATIVE `set_order` (which folds in the ever-advancing
/// `next_set_order` cursor) diverges from its raw per-delta `authoring_set_order`.
/// `rich_object_state` cannot exercise the DR authoring-rank regression because
/// it hand-sets `set_order == authoring_set_order` on every field.
fn divergent_seed_object_state() -> ObjectState {
    let mut state = ObjectState {
        object_id: 0, // assigned by LayerStore::create_layer
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };

    let sec1 = Section {
        section_id: 1,
        parent_id: 0,
        name: "World".to_string(),
        section_type: 1,
        section_order: 0,
        first_version: 1,
        last_version: 1,
    };
    let sec2 = Section {
        section_id: 2,
        parent_id: 1,
        name: "Cube".to_string(),
        section_type: 2,
        section_order: 1,
        first_version: 1,
        last_version: 1,
    };
    state.sections.insert(1, sec1);
    state.sections.insert(2, sec2);

    // Children: root has [1], section 1 has [2].
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);

    let ts1 = TimeSample {
        section_id: 1,
        time: OrderedFloat(0.0),
        value: FieldValue::Inline(vec![0xAA, 0xBB]),
        version: 1,
    };
    let ts2 = TimeSample {
        section_id: 2,
        time: OrderedFloat(1.5),
        value: FieldValue::Inline(vec![0xCC]),
        version: 1,
    };
    state.time_samples.insert((ts1.section_id, ts1.time), ts1);
    state.time_samples.insert((ts2.section_id, ts2.time), ts2);

    state
}

/// Build a `Config` carrying every field at its `LIVE_EDIT_*` default, with
/// `checkpoint_seq_safety_margin` overridden to the caller-supplied value.
///
/// The apply path only consumes `checkpoint_seq_safety_margin`; every other
/// field is filler. Building the `Config` literally (rather than reaching for
/// `Config::from_env`) avoids the process-global env-var mutation that
/// `temp_env::with_vars` would otherwise require and keeps the tests free of
/// runtime env coupling.
fn test_config(safety_margin: u64) -> Config {
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
        checkpoint_enabled: false,
        checkpoint_adapter: live_edit::checkpoint::CheckpointAdapter::Disabled,
        checkpoint_interval_secs: 60,
        checkpoint_quiesce_secs: 30,
        checkpoint_tick_secs: 10,
        checkpoint_retain_generations: 2,
        checkpoint_seq_safety_margin: safety_margin,
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

// The local `wipe_object_keys` that used to sit here is gone (b.6de sub-task 4);
// every call site below names `crate::common::wipe_object_keys` instead.
//
// It had converged, line for line, on the shared helper — same cluster-aware
// `crate::common::scan_object_keys` enumeration, same multi-key UNLINK (slot-safe
// because every key carries the `{object_id}` hash tag), same survivors-empty
// postcondition. A helper duplicated per test file is how one copy quietly stops
// matching the others, and this one had three copies. The shared helper's
// doc-comment carries the `obj:stream` rationale (the restore materializer queues
// no synthetic `1-0` XADD, so a surviving stream would leak across the wipe) and
// the argument for asserting the postcondition.

/// True iff `le:{oid}:obj:hdr` exists. `write_object` is the only function
/// that writes the obj header; using its absence as the negative-write
/// signal in tests #2 / #3 / #10 lets us assert "the apply did not call
/// `write_object`" without depending on the full obj:* keyspace being clean
/// (the per-object stream key persists with PEXPIRE TTL after DeleteLayer, so
/// a raw "no `obj:*` keys" assertion is too brittle).
async fn obj_header_exists(tv: &TestValkey, object_id: u64) -> bool {
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    tv.client()
        .pool
        .exists(&header_key)
        .await
        .expect("EXISTS obj:hdr must succeed")
}

/// True iff ANY non-header state key written by `write_object` is currently
/// present for `object_id`: sections-index (`obj:secs`), fields-index
/// (`obj:flds`), time-samples-index (`obj:tss`), deleted-sections-index
/// (`obj:dels`), the children-parent-id index (`obj:chs`), or the root
/// children-ordering list (`obj:ch:0`). Combined with [`obj_header_exists`],
/// this is the exact write surface `write_object` would have landed on the
/// apply path — pinning these to absent proves the failed apply did not
/// execute steps 5-6.
///
/// We intentionally do NOT scan `le:{oid}:obj:*` blanket-style: `obj:stream`
/// is PEXPIRE'd with grace so lagging subscribers can observe the `deleted`
/// sentinel before the key vanishes (`STREAM_DELETE_GRACE_MS`), and the apply
/// path deliberately does NOT XADD to it (per the apply.rs Step-5
/// doc-comment). Its presence after a failed apply is a survivor of the prior
/// cleanup, not a sign that the apply touched the keyspace, so a raw `obj:*`
/// count would produce false-positive failures here. That exemption is
/// permanent and specific to one key.
///
/// **`obj:ch:0` is deliberately on the checked list (b.9fn).** It used to be a
/// second exemption for a much worse reason: `cleanup_old_object` derived
/// `obj:ch:{parent}` keys by walking `obj:secs` / `obj:dels`, and parent id 0
/// — the synthetic root — is in neither, so the root children list survived
/// every retirement and leaked forever. Now that both retirement paths build
/// their inventory from `KeySchema::object_cleanup_keys`, the old carve-out
/// inverts into a regression guard: tests #2 / #3 retire a layer that had
/// top-level sections, so a `obj:ch:0` sighting here means the leak is back.
///
/// **`obj:chs` is on the list for the same reason (b.jdr).** It is the
/// children-list parent-id index, SADDed inside the same MULTI/EXEC as the
/// lists themselves, so it is part of `write_object`'s write surface and its
/// presence after a refused apply would mean step 5 ran. It is also reaped by
/// the same shared inventory, so a sighting after a retirement is the b.jdr
/// leak returning — this time in the index rather than what it indexes.
async fn any_write_object_state_key_exists(tv: &TestValkey, object_id: u64) -> bool {
    let oid_str = object_id.to_string();
    let keys = tv.key_schema();
    for k in [
        keys.obj_sections_index(&oid_str),
        keys.obj_fields_index(&oid_str),
        keys.obj_time_samples_index(&oid_str),
        keys.obj_deleted_sections_index(&oid_str),
        keys.obj_children_index(&oid_str),
        keys.obj_children(&oid_str, "0"),
    ] {
        let exists: bool = tv
            .client()
            .pool
            .exists(&k)
            .await
            .expect("EXISTS index key must succeed");
        if exists {
            return true;
        }
    }
    false
}

/// Current wall-clock milliseconds since the UNIX epoch, matching
/// `checkpoint::now_unix_ms` in production. Used to bracket the value
/// `apply_checkpoint` stamps into `obj:hdr last_checkpoint_ms`.
fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Run the production assembler against a populated layer and return the
/// produced OMPF bytes + `CheckpointMeta`. The assembler reads the live
/// `obj:hdr` so the meta it returns carries the session_id / layer_name /
/// created_by / server_ts that the apply path will use to reconstruct the
/// header. Single source of truth for the round-trip — no hand-rolled OMPF
/// per test. `pub(crate)` so `chs_complete_tests` builds its restore fixture
/// the same way rather than through a copy.
pub(crate) async fn build_checkpoint_ompf(
    tv: &TestValkey,
    object_id: u64,
) -> (Bytes, u64, CheckpointMeta) {
    let store = Arc::new(InMemoryCheckpointStore::new());
    // The fixture layer has no concurrent writer and no regression, so the
    // assembler's proven read verifies and always writes; a
    // `SkippedNotQuiesced` here would itself be the regression.
    let outcome = match assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        object_id,
        store.as_ref(),
        SERVICE_LABEL,
        &mut live_edit::checkpoint::assembler::NoLeaseToKeep,
    )
    .await
    .expect("assembler must succeed against a populated live layer")
    {
        AssemblyDecision::Written(outcome) => outcome,
        AssemblyDecision::SkippedNotQuiesced => panic!(
            "assembler must prove a cut at or above its pin against a quiescent layer; \
             got SkippedNotQuiesced"
        ),
        AssemblyDecision::LeaseLost => {
            panic!("the assembler was handed a keeper that never loses its lease; got LeaseLost")
        }
        AssemblyDecision::ShuttingDown => {
            panic!("the assembler was handed a keeper that never shuts down; got ShuttingDown")
        }
    };

    let blob = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("checkpoint get must succeed")
        .expect("checkpoint must exist for the assembled object");

    (blob.bytes.clone(), outcome.snap_seq, blob.meta.clone())
}

// =============================================================================
// Test 1 — Happy path: apply round-trips full state via read_object_state_for_test.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_restores_full_state_via_read_object_state_for_test() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Step 1: create a layer with a populated state.
    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "round-trip-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Step 2: assemble a checkpoint OMPF from the live state.
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;

    // Step 3: wipe the obj-state keys (emulating post-loss keyspace).
    crate::common::wipe_object_keys(&tv, oid).await;

    // Step 4: apply the checkpoint. With `max_seq_observed=None` and the
    // default safety margin, `seeded_seq = max(snap_seq, 0) + 1000`, but the
    // focus of this test is structural round-trip; the seq pin is covered by
    // tests #4 / #5 / #6.
    let config = test_config(1000);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply_checkpoint must succeed on a wiped obj keyspace");

    // Step 5: read_object_state_for_test back out and assert structural fidelity.
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed against the restored object");

    // Sections — same set of section_ids, names, types preserved.
    assert_eq!(snapshot.sections.len(), 2, "two sections must round-trip");
    let sec1 = &snapshot.sections[&1];
    assert_eq!(sec1.name, "World");
    assert_eq!(sec1.parent_id, 0);
    let sec2 = &snapshot.sections[&2];
    assert_eq!(sec2.name, "Cube");
    assert_eq!(sec2.parent_id, 1);

    // Children list — parent → child relationship preserved.
    assert_eq!(
        snapshot.children.get(&0).map(|v| v.as_slice()),
        Some([1u64].as_slice()),
        "root children must round-trip"
    );
    assert_eq!(
        snapshot.children.get(&1).map(|v| v.as_slice()),
        Some([2u64].as_slice()),
        "sec1 children must round-trip"
    );

    // Fields — set + payloads preserved.
    let field_map: HashMap<(u64, String), FieldValue> = snapshot
        .fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.value.clone()))
        .collect();
    assert_eq!(field_map.len(), 2);
    assert_eq!(
        field_map[&(1, "color".to_string())],
        FieldValue::Inline(vec![0xFF, 0x00, 0x80])
    );
    assert_eq!(
        field_map[&(2, "size".to_string())],
        FieldValue::Inline(vec![0x42])
    );

    // Time samples — keys + payloads preserved.
    let ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = snapshot
        .time_samples
        .iter()
        .map(|ts| ((ts.section_id, ts.time), ts.value.clone()))
        .collect();
    assert_eq!(ts_map.len(), 2);
    assert_eq!(
        ts_map[&(1, OrderedFloat(0.0))],
        FieldValue::Inline(vec![0xAA, 0xBB])
    );
    assert_eq!(
        ts_map[&(2, OrderedFloat(1.5))],
        FieldValue::Inline(vec![0xCC])
    );

    // Header session-identity fields restored.
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL header must succeed");
    assert_eq!(
        header.get("session_id").map(String::as_str),
        Some(session_id.as_str()),
        "session_id must be reconstructed from CheckpointMeta"
    );
    assert_eq!(
        header.get("layer_name").map(String::as_str),
        Some("round-trip-layer")
    );
    assert_eq!(
        header.get("created_by").map(String::as_str),
        Some("apply-test-user")
    );
    assert_eq!(
        header.get("src_fmt").map(String::as_str),
        Some("usda"),
        "source_format must round-trip via the reset delta"
    );

    // b.jdr: `write_object` is the apply path's materializer, so the
    // children-list parent-id index is part of what a restore must rebuild —
    // otherwise a DR-restored object retires with the pre-b.jdr inventory and
    // leaks its children lists. Exactly the parents holding a non-empty list:
    // root 0 and section 1. Section 2 is a leaf, so it owns no `obj:ch:2` key
    // and must not appear (membership ⟺ key existence).
    let mut indexed: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    indexed.sort();
    assert_eq!(
        indexed,
        vec!["0".to_string(), "1".to_string()],
        "apply_checkpoint must rebuild `obj:chs` from the restored \
         `state.children` — the checkpoint blob carries no index of its own \
         (which is why b.jdr needed no blob-format change)"
    );
}

// =============================================================================
// Test 1b — an orphan restores as a row with no list under its missing parent.
// =============================================================================

/// [`rich_object_state`] plus an orphan section whose `parent_id` names no
/// section — the tolerated state of SDD §10.4 that `apply_diff_change_section`
/// produces — and the children list `children[77] = [3]` an earlier build
/// stored for it. `create_layer` materializes the state as given, so the
/// created object holds `obj:ch:77` and indexes 77, as such an object does.
fn state_with_dangling_parent() -> ObjectState {
    let mut state = rich_object_state();
    state.sections.insert(
        3,
        Section {
            section_id: 3,
            parent_id: 77,
            name: "Orphan".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 7,
        },
    );
    state.children.insert(77, vec![3]);
    state
}

/// The restore path reaches an orphan by a route nothing else tests: the
/// checkpoint blob stores no children map and no index, only per-section
/// `parent_id`s. The assembler emits the orphan section after the reachable
/// DFS walk carrying its stored `parent_id`, and apply's reset delta
/// re-creates it through `apply_diff_change_section`. The engine keeps no
/// list under a parent with no row (SDD §10.4), so the restored object holds
/// the orphan's row — `parent_id` verbatim, neither reparented nor dropped —
/// and no `obj:ch:77`, and `write_object` indexes exactly the lists it wrote.
///
/// What this pins is the restore's own output over a wiped keyspace: orphan
/// 3 comes back under 77, no `obj:ch:77` is written, and no `77` is added to
/// `obj:chs`. The fixture's earlier-build list only shapes the created
/// object; the wipe removes it before the restore runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_restores_an_orphan_with_no_list_under_its_missing_parent() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "dangling-parent-layer",
            "apply-test-user",
            &state_with_dangling_parent(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let dangling_key = tv.key_schema().obj_children(&oid_str, "77");

    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    let config = test_config(1000);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply_checkpoint must succeed on a wiped obj keyspace");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid_str)
        .await
        .expect("read_object_state_for_test must succeed after apply");
    assert_eq!(
        snapshot.sections.get(&3).map(|s| s.parent_id),
        Some(77),
        "the orphan is restored with its stored `parent_id` (SDD §10.4 — the \
         engine neither reparents nor drops it)"
    );

    let mut indexed: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid_str))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    indexed.sort();
    assert_eq!(
        indexed,
        vec!["0".to_string(), "1".to_string()],
        "the restored `obj:chs` names exactly the lists under the root and the \
         live sections — 77 has no row, so no list and no index member"
    );
    let restored_with_list: bool = tv
        .client()
        .pool
        .exists(&dangling_key)
        .await
        .expect("EXISTS obj:ch:77 must succeed");
    assert!(
        !restored_with_list,
        "the restore writes no list under the missing parent 77"
    );
}

// =============================================================================
// Test 2 — No live mapping returns NotFound and writes nothing
// =============================================================================
//
// SR-2.4 guard: if no `le:{sid}:layer:{hex(name)}` mapping resolves to the
// candidate `object_id`, the apply MUST surface `CheckpointError::NotFound`
// AND MUST NOT call `write_object` (no `le:{oid}:obj:*` keys created).
//
// We emulate the "layer deleted between checkpoint and apply" race by
// creating + assembling + then DeleteLayer'ing the layer — the OMPF blob is
// preserved in the in-memory checkpoint store but the live mapping is gone.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_no_live_mapping_returns_not_found_and_does_not_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Step 1: create the layer + assemble the checkpoint.
    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "vanishing-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;

    // Step 2: DeleteLayer the layer — wipes the mapping + the obj state keys.
    // The per-object stream key is PEXPIRE'd with grace (see
    // `STREAM_DELETE_GRACE_MS`), so it survives for ~30s; that's fine — the
    // apply path's write surface is obj:hdr + the index/data hashes, NOT the
    // stream key (the apply path deliberately does not XADD a synthetic
    // entry, per the apply.rs Step-5 doc-comment).
    ls.delete_layer(&session_id, "vanishing-layer", mock_ext_store())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(Duration::from_secs(5)).await;

    // Sanity: post-DeleteLayer the obj:hdr must be gone — that's the write
    // surface we're checking for in step 4.
    assert!(
        !obj_header_exists(&tv, oid).await,
        "DeleteLayer must remove obj:hdr before the apply runs"
    );

    // Step 3: apply must return NotFound without writing anything.
    let config = test_config(1000);
    let err = apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect_err("apply must fail when no live mapping resolves to object_id");
    assert!(
        matches!(err, CheckpointError::NotFound),
        "guard-false must surface NotFound; got {err:?}"
    );

    // Step 4: no state-bearing obj keys (header + sections/fields/ts/dels
    // indices + root children list) were created by the failed apply. Only
    // the pre-existing PEXPIRE'd stream key is excluded — see
    // `any_write_object_state_key_exists` for why. The `obj:ch:0` arm doubles
    // as the b.9fn leak guard: DeleteLayer above retired a layer with a
    // top-level section, so that key must already be gone.
    assert!(
        !obj_header_exists(&tv, oid).await,
        "guard-false apply must NOT write obj:hdr"
    );
    assert!(
        !any_write_object_state_key_exists(&tv, oid).await,
        "guard-false apply must NOT write any obj state key \
         (secs/flds/tss/dels/ch:0)"
    );
}

// =============================================================================
// Test 3 — Stale object_id after CreateLayer overwrite returns NotFound
// =============================================================================
//
// CreateLayer-overwrite-during-loss race: a checkpoint was assembled against
// `object_id = N`; the keyspace was lost; before recovery ran, a new
// `CreateLayer` on the same `(session, layer_name)` minted `object_id = N+1`
// and the live mapping now points there. Applying the stale `N` OMPF would
// resurrect an orphaned object shadowing the post-recovery state.
//
// The SR-2.4 identity guard GETs the mapping key named by the blob's own
// `(session_id, layer_name)` and requires it to equal `N`, so it must reject
// `N` once that mapping points at `N+1`.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_stale_object_id_after_create_layer_overwrite_returns_not_found() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Step 1: create layer N, assemble its checkpoint, hold the OMPF.
    let initial = rich_object_state();
    let old_oid = ls
        .create_layer(
            &session_id,
            "overwritten-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, old_oid).await;

    // Step 2: overwrite via a second create_layer — mints N+1 and points the
    // mapping there. The old `N` obj keys are cleaned up asynchronously.
    let new_oid = ls
        .create_layer(
            &session_id,
            "overwritten-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer (overwrite) must succeed");
    assert_ne!(
        old_oid, new_oid,
        "overwrite must allocate a fresh object_id"
    );
    ls.wait_for_cleanup(Duration::from_secs(5)).await;

    // Sanity: live mapping points at new_oid, old obj keys are gone.
    let mapping_value: Option<String> = tv
        .client()
        .pool
        .get(
            tv.key_schema()
                .layer_mapping(&session_id, "overwritten-layer"),
        )
        .await
        .expect("GET layer mapping must succeed");
    assert_eq!(
        mapping_value.as_deref(),
        Some(new_oid.to_string().as_str()),
        "mapping must point at the new oid after overwrite"
    );
    assert!(
        !obj_header_exists(&tv, old_oid).await,
        "old obj:hdr must be cleaned up by overwrite cleanup"
    );

    // Step 3: applying with the OLD oid must fail — no mapping resolves to N.
    let config = test_config(1000);
    let err = apply_checkpoint(&ls, old_oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect_err("apply with stale oid must fail (mapping now points elsewhere)");
    assert!(
        matches!(err, CheckpointError::NotFound),
        "stale-oid apply must surface NotFound; got {err:?}"
    );

    // Step 4: no obj-state keys created under the stale oid. Only the
    // PEXPIRE'd stream key from the overwrite cleanup is excluded — see
    // `any_write_object_state_key_exists` for why. The `obj:ch:0` arm doubles
    // as the b.9fn leak guard: the overwrite above retired a layer with a
    // top-level section, so that key must already be gone.
    assert!(
        !obj_header_exists(&tv, old_oid).await,
        "stale-oid apply must NOT write obj:hdr"
    );
    assert!(
        !any_write_object_state_key_exists(&tv, old_oid).await,
        "stale-oid apply must NOT write any obj state key \
         (secs/flds/tss/dels/ch:0)"
    );
}

// =============================================================================
// Test 4 — SR-2.5 fallback: max_seq_observed=None seeds snap_seq + margin
// =============================================================================
//
// b.8y5 regression surface. When `max_seq_observed=None` the restore floor
// is `snap_seq` alone, and the safety margin is added ON TOP:
// `seeded_seq = max(snap_seq, 0) + safety_margin`.
//
// The inputs are chosen so `snap_seq > safety_margin` (7 > 1) — the exact
// shape the pre-fix `max(snap_seq, 0 + safety_margin)` formula got wrong.
// Under that formula the margin was inert here and the restore resumed at
// exactly `snap_seq = 7`, handing the next writer a sequence number
// subscribers may already hold attached to now-lost content. The additive
// formula seeds `8`, one clear guard-band step above the floor.
//
// `snap_seq > margin` is not a corner case: it is the steady state for any
// layer past its first `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` commits.
//
// Pin both `obj_header.seq` and the post-write
// `obj_header.last_checkpoint_seq` — the latter still stamps `snap_seq`
// verbatim, so the two now differ and the test distinguishes them.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_seeds_snap_seq_plus_margin_when_max_seq_observed_is_none() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "seq-fallback-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    // The assembler stamps `snap_seq` from the post-fan-out header read;
    // pin it explicitly so a future assembler refactor that changes the
    // value source is caught here too.
    assert_eq!(
        snap_seq, 7,
        "assembler must capture snap_seq from obj_header.seq"
    );
    crate::common::wipe_object_keys(&tv, oid).await;

    // b.vnw: hold the blob's measured commit span at its "no measurement"
    // value so `band = max(configured margin, observed_commit_span)` reduces
    // to the configured margin, and this case keeps discriminating the one
    // thing it was written for — that the margin is added ABOVE the restore
    // floor rather than folded into a `max()` against it. The assembler
    // naturally stamps `span = 7` on this fixture (a first generation
    // measured against `last_checkpoint_seq = 0`, i.e. commits-since-
    // creation), which would dominate `margin = 1` and make the b.8y5
    // arithmetic unobservable here. The span-dominates dimension gets its
    // own case (`..._seeds_from_observed_commit_span_...`), as does the
    // legacy-blob fallback this line happens to also model.
    meta.observed_commit_span = 0;

    // safety_margin = 1, deliberately BELOW snap_seq=7, so the additive
    // guard band is the only thing that can lift the seeded seq above 7:
    // `max(7, 0) + 1 = 8`.
    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(
        seq.as_deref(),
        Some("8"),
        "b.8y5: seeded_seq must be restore_floor + safety_margin = max(snap_seq=7, 0) + 1 = 8. \
         A value of 7 means the margin was folded into a max() and went inert because \
         snap_seq exceeded it — the restored layer would resume at exactly the last \
         provable sequence number, with no guard band over the commits lost inside the \
         RPO window"
    );

    let last_cp_seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert_eq!(
        last_cp_seq.as_deref(),
        Some("7"),
        "post-write HSET must stamp last_checkpoint_seq = snap_seq (NOT seeded_seq) \
         (load-bearing for the skip-if-no-commits gate on the next checkpoint tick)"
    );
}

// =============================================================================
// Test 5 — SR-2.5 main path: max_seq_observed raises the restore floor
// =============================================================================
//
// `seeded_seq = max(snap_seq, max_seq_observed) + safety_margin`. With
// `snap_seq = 7`, `max_seq_observed = Some(100)`, `safety_margin = 50`, the
// restore floor is `max(7, 100) = 100` and seeded_seq = `100 + 50 = 150`.
//
// Deliberately kept as a CROSS-FORMULA ANCHOR: `max(7, 100 + 50)` and
// `max(7, 100) + 50` both evaluate to 150, so this case is invariant under
// the b.8y5 change and stays green either way. Its job is to prove the
// caller-supplied high-water hint still dominates a lower `snap_seq` —
// tests #4 and #6 are the cases that discriminate between the two formulas.
//
// b.vnw gave it a second, orthogonal role: it is the designated cover for
// "the configured margin is a FLOOR under the measured band", the complement
// of test #14's "a measured span above the margin sizes the band". That role
// only holds if the blob carries a real, non-zero span strictly BELOW the
// margin, which is why the fixture pins the assembler's natural sentinel and
// then overrides it — see the two blocks in the body. A span left at the
// sentinel 0 would exercise "no measurement at all", which is case #15's
// subject, and would make `max(50, span)` win by default rather than by
// comparison.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_advances_seq_to_max_seq_observed_plus_safety_margin() {
    // Strictly below `test_config(50)`'s margin, non-zero, and distinct from
    // every other number in the fixture (snap_seq 7, max_seq_observed 100,
    // margin 50, seeded 150) so no assertion below can be satisfied by
    // coincidence.
    const SUB_MARGIN_SPAN: u64 = 9;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "seq-observed-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");
    // The assembler's own measurement on this fixture, before any override:
    // a first generation (`last_checkpoint_seq` absent → 0) declines to
    // measure and stamps the no-measurement sentinel (b.vnw round-2). Pinned
    // explicitly so the override on the next line is demonstrably what puts a
    // real sub-margin span into the arithmetic, rather than the case silently
    // testing the sentinel path if the fixture's natural value ever changes.
    assert_eq!(
        meta.observed_commit_span, 0,
        "test precondition: the assembler must stamp the no-measurement \
         sentinel on a layer's first generation"
    );
    crate::common::wipe_object_keys(&tv, oid).await;

    // A COLD layer: it really did commit, but fewer times over the interval
    // than the operator's configured margin covers. The band must be the
    // margin — this is the direction that guarantees b.vnw can only ever
    // widen a guard band, never narrow one.
    meta.observed_commit_span = SUB_MARGIN_SPAN;

    let config = test_config(50);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, Some(100), &config)
        .await
        .expect("apply must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(
        seq.as_deref(),
        Some("150"),
        "seeded_seq must equal max(snap_seq, max_seq_observed) + \
         max(safety_margin, observed_commit_span) = max(7, 100) + \
         max(50, {SUB_MARGIN_SPAN}) = 150 when the observed high-water \
         dominates snap_seq and the configured margin dominates the measured \
         span. A value of 109 means the span replaced the margin instead of \
         being floored by it, i.e. b.vnw can NARROW a guard band — the \
         b.8y5 sequence-reuse direction"
    );

    // The band is the margin, recorded as such. Reading the stamp (rather
    // than only `seq`) separates the two ways the assertion above could pass:
    // the right band, or a wrong band cancelled out by a wrong floor.
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(50),
        "restore_seq_bump must record the band that was applied — the \
         configured margin (50), which floored the measured span \
         ({SUB_MARGIN_SPAN})"
    );

    let last_cp_seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert_eq!(
        last_cp_seq.as_deref(),
        Some("7"),
        "last_checkpoint_seq must equal snap_seq (NOT seeded_seq) so operators \
         can tell which checkpoint generation produced the state"
    );
}

// =============================================================================
// Test 6 — SR-2.5 edge: snap_seq dominates the observed high-water
// =============================================================================
//
// When `snap_seq > max_seq_observed`, the `max(...)` selects `snap_seq` as
// the restore floor — and the margin is STILL added on top. With
// `snap_seq = 7`, `max_seq_observed = Some(2)`, `safety_margin = 1`:
// `max(7, 2) + 1 = 8`.
//
// Second b.8y5 discriminator alongside test #4: the pre-fix
// `max(snap_seq, max_seq_observed + safety_margin)` form evaluated
// `max(7, 3) = 7` here, dropping the guard band entirely. This case adds
// the "a stale observed hint must not suppress the margin either" dimension
// that test #4 (`max_seq_observed = None`) does not cover.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_seeds_snap_seq_plus_margin_when_snap_seq_dominates_observed() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "seq-snap-dominates-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");
    crate::common::wipe_object_keys(&tv, oid).await;

    // b.vnw: same reason as the `max_seq_observed = None` case above — the
    // measured span is held at "no measurement" so `band` reduces to the
    // configured margin and the b.8y5 additive-vs-max discrimination stays
    // the only variable this case moves.
    meta.observed_commit_span = 0;

    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, Some(2), &config)
        .await
        .expect("apply must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(
        seq.as_deref(),
        Some("8"),
        "b.8y5: seeded_seq must be max(snap_seq=7, max_seq_observed=2) + safety_margin=1 = 8. \
         A value of 7 means a stale (lower) observed high-water suppressed the guard band \
         because the margin was folded into the max() instead of added above the floor"
    );

    let last_cp_seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert_eq!(
        last_cp_seq.as_deref(),
        Some("7"),
        "last_checkpoint_seq must equal snap_seq irrespective of which branch \
         of the max() produced the restore floor"
    );
}

// =============================================================================
// Test 7 — EXISTS-short-circuit consulted BEFORE write
// =============================================================================
//
// Step 1 of the apply algorithm short-circuits with `Ok(())` when
// `EXISTS le:{oid}:obj:hdr` returns true. We prove the short-circuit fires
// BEFORE `write_object` by:
//
// (a) Running a successful apply once so the header lands in Valkey.
// (b) Re-running apply with deliberately invalid OMPF bytes (`Bytes::new()`).
//     If step 1 ran before step 3, the call succeeds and the pre-existing
//     header is preserved verbatim. If anything beyond EXISTS ran, the empty
//     OMPF would surface `CheckpointError::Permanent` from the decoder.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_exists_short_circuit_returns_ok_without_rewriting() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "exists-shortcircuit-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    // First apply: lands the header, stamps last_checkpoint_seq.
    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("first apply must succeed");

    // Capture the post-apply header state for diff comparison.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header_before: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL header before short-circuit apply");

    // Second apply: deliberately invalid OMPF bytes. If anything past EXISTS
    // ran, decode_checkpoint_ompf would surface Permanent.
    apply_checkpoint(&ls, oid, Bytes::new(), 999_999, &meta, None, &config)
        .await
        .expect(
            "EXISTS short-circuit must return Ok(()) on a pre-existing obj_header, \
             regardless of the OMPF / snap_seq supplied",
        );

    // Header must be unchanged — short-circuit skipped step 5 (write_object)
    // AND step 6 (HSET last_checkpoint_seq). A regression that ran step 6 on
    // the short-circuit path would clobber the `7` we stamped above with the
    // supplied `999_999`.
    let header_after: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL header after short-circuit apply");
    assert_eq!(
        header_before, header_after,
        "EXISTS short-circuit must leave the existing header byte-identical \
         (no rewrite of seq, last_checkpoint_seq, or any other field)"
    );
}

// =============================================================================
// Test 8 — Concurrent apply, no duplicated keys
// =============================================================================
//
// Two concurrent `apply_checkpoint` calls against the same wiped object must
// both return `Ok(())` and produce a final state matching `rich_object_state()`
// — one call wins via `write_object`'s MULTI/EXEC and the other short-circuits
// on the EXISTS check.
//
// Uses `Arc<Barrier::new(2)>` for lockstep entry; without the barrier the
// spawn ramp-up could let one apply complete before the other even begins,
// collapsing the test to two sequential applies.
//
// Requires `setup_with_pool_size(4)` so the txn pool acquire never blocks.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn apply_checkpoint_concurrent_apply_no_duplicated_keys() {
    let tv = TestValkey::setup_with_pool_size(4).await;
    let ls = Arc::new(layer_store(&tv));
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "concurrent-apply-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    let config = Arc::new(test_config(1));
    let meta_arc = Arc::new(meta);
    let ompf_arc = Arc::new(ompf);
    let barrier = Arc::new(Barrier::new(2));

    let mut handles = Vec::with_capacity(2);
    for _ in 0..2 {
        let ls = Arc::clone(&ls);
        let config = Arc::clone(&config);
        let meta_arc = Arc::clone(&meta_arc);
        let ompf_arc = Arc::clone(&ompf_arc);
        let barrier = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            apply_checkpoint(
                ls.as_ref(),
                oid,
                ompf_arc.as_ref().clone(),
                snap_seq,
                meta_arc.as_ref(),
                None,
                config.as_ref(),
            )
            .await
        }));
    }

    for h in handles {
        let result = h.await.expect("apply task must not panic");
        result.expect("both concurrent applies must return Ok(())");
    }

    // The resulting state must be exactly one materialization of
    // rich_object_state(). We verify by loading the snapshot and asserting
    // section / field / time-sample counts match the input — a duplicated
    // write would show up as duplicate index entries (e.g. ZADD adding the
    // same member twice with different scores would still leave one member,
    // but extra HSETs on the same key are idempotent — so the strongest
    // assertion is on collection sizes + payload equality, which would catch
    // any per-key drift).
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed after concurrent applies");
    assert_eq!(snapshot.sections.len(), 2, "no duplicate section entries");
    assert_eq!(snapshot.fields.len(), 2, "no duplicate field entries");
    assert_eq!(
        snapshot.time_samples.len(),
        2,
        "no duplicate time-sample entries"
    );

    // last_checkpoint_seq must be present exactly once — a stray double-HSET
    // would still produce a single field but with the same value, so we pin
    // the value to ensure the HSET executed cleanly under contention.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let last_cp_seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert_eq!(
        last_cp_seq.as_deref(),
        Some("7"),
        "last_checkpoint_seq must equal snap_seq after concurrent applies; \
         a stray write from the EXISTS-short-circuit branch would clobber this"
    );
}

// =============================================================================
// Test 9 — OMPF + apply round trip is materialized-state-isomorphic
// =============================================================================
//
// Stronger variant of test #1: build a checkpoint, wipe the keyspace, apply,
// then compare the post-apply `ObjectSnapshot` to a snapshot loaded BEFORE
// the wipe. The two snapshots must match on every structural surface — same
// sections, same children map, same fields, same time samples, same
// source_format. Catches any drift introduced by the OMPF wire shape (e.g.
// a fielded value with `value_or_ext_hash` truncation, a section's
// section_type stamped incorrectly).

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_ompf_round_trip_state_isomorphism() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "isomorphism-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Snapshot of the live state BEFORE the wipe.
    let pre_wipe = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test pre-wipe must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply must succeed");

    let post_apply = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test post-apply must succeed");

    // Sections set + per-section structure isomorphic.
    let pre_sec_ids: std::collections::HashSet<u64> = pre_wipe.sections.keys().copied().collect();
    let post_sec_ids: std::collections::HashSet<u64> =
        post_apply.sections.keys().copied().collect();
    assert_eq!(
        pre_sec_ids, post_sec_ids,
        "section_id set must be invariant across apply round-trip"
    );
    for sid in &pre_sec_ids {
        let pre = &pre_wipe.sections[sid];
        let post = &post_apply.sections[sid];
        assert_eq!(post.name, pre.name, "section {sid} name");
        assert_eq!(post.parent_id, pre.parent_id, "section {sid} parent_id");
        assert_eq!(
            post.section_type, pre.section_type,
            "section {sid} section_type"
        );
    }

    // Children map invariant.
    assert_eq!(
        post_apply.children, pre_wipe.children,
        "children map must round-trip exactly"
    );

    // Fields set + payloads invariant. Compared as a key→payload map because
    // `Vec<Field>` ordering depends on the underlying ZRANGE order and we
    // only care about structural identity.
    let pre_field_map: HashMap<(u64, String), FieldValue> = pre_wipe
        .fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.value.clone()))
        .collect();
    let post_field_map: HashMap<(u64, String), FieldValue> = post_apply
        .fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.value.clone()))
        .collect();
    assert_eq!(
        pre_field_map, post_field_map,
        "fields (keys + payloads) must round-trip exactly"
    );

    // Time samples set + payloads invariant. Same reasoning as fields.
    let pre_ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = pre_wipe
        .time_samples
        .iter()
        .map(|ts| ((ts.section_id, ts.time), ts.value.clone()))
        .collect();
    let post_ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = post_apply
        .time_samples
        .iter()
        .map(|ts| ((ts.section_id, ts.time), ts.value.clone()))
        .collect();
    assert_eq!(
        pre_ts_map, post_ts_map,
        "time samples (keys + payloads) must round-trip exactly"
    );

    // source_format invariant (carried on the reset delta via build_delta).
    assert_eq!(
        post_apply.source_format, pre_wipe.source_format,
        "source_format must round-trip via the reset delta"
    );
}

// =============================================================================
// Test 10 — Invalid OMPF surfaces Permanent; no Valkey writes
// =============================================================================
//
// Garbage / truncated OMPF bytes must surface `CheckpointError::Permanent`
// (retry would loop on the same input) and MUST NOT have written any obj
// keys before failing — the live-mapping guard runs before the decoder, so
// the apply path could in principle have run step 2 successfully and then
// failed at step 3 without touching the obj keyspace. Pin both.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_invalid_ompf_returns_permanent_error() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Create a layer so the live-mapping guard passes — otherwise the apply
    // would fail with NotFound at step 2 and never reach step 3 (the decoder),
    // which is the layer we are trying to exercise here.
    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "invalid-ompf-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    crate::common::wipe_object_keys(&tv, oid).await;

    // Construct a CheckpointMeta directly — we don't need the assembler to
    // produce it, because the apply will fail at the decoder before it
    // consumes any meta field.
    let meta = CheckpointMeta {
        session_id: session_id.clone(),
        layer_name: "invalid-ompf-layer".to_string(),
        created_by: "apply-test-user".to_string(),
        server_ts: 12345,
        ..CheckpointMeta::default()
    };

    // 32 bytes of garbage — not a valid OMPF magic (`0x46504d4f` = "OMPF").
    let garbage: Vec<u8> = (0..32).map(|i| 0x55u8 ^ i as u8).collect();

    let config = test_config(1);
    let err = apply_checkpoint(&ls, oid, Bytes::from(garbage), 7, &meta, None, &config)
        .await
        .expect_err("garbage OMPF must Err");
    assert!(
        matches!(err, CheckpointError::Permanent { .. }),
        "garbage OMPF must classify Permanent; got {err:?}"
    );

    // Failed apply must NOT have written any obj-state keys — the decoder
    // runs at step 3, AFTER the EXISTS check (step 1) and the layer-mapping
    // guard (step 2) but BEFORE write_object (step 5). A regression that
    // reordered these would surface here as a stray obj:hdr or any
    // write_object state key.
    assert!(
        !obj_header_exists(&tv, oid).await,
        "Permanent-erroring apply must NOT write obj:hdr"
    );
    assert!(
        !any_write_object_state_key_exists(&tv, oid).await,
        "Permanent-erroring apply must NOT write any obj state key \
         (secs/flds/tss/dels/ch:0)"
    );
}

// =============================================================================
// Test 11 — Atomic header (b.4fh regression): apply_checkpoint never lands
//           a skeletal 1-field header
// =============================================================================
//
// b.4fh root cause: `apply_checkpoint` previously stamped the header in two
// Valkey round-trips — a `write_object` MULTI/EXEC for the 12 canonical fields,
// then a trailing `HSET obj:hdr last_checkpoint_seq=<snap_seq>`. A wipe (orphan
// sweep, manual operator UNLINK, or a sibling-replica race) landing BETWEEN
// the EXEC and the trailing HSET would leave the header with a single
// `last_checkpoint_seq` field set — a skeletal shape that downstream
// `EXISTS`-style guards (apply step 1, orphan-sweep classification) treat as
// "healthy" but that omits every structural field.
//
// The fix folds `last_checkpoint_seq` into the same single `HSET` issued for
// the 12 canonical header fields, all inside one MULTI/EXEC. b.yut then added
// the second half of the checkpoint cursor, `last_checkpoint_ms`, to the same
// bundle (they travel together in `CheckpointCursorStamp` precisely so they
// cannot disagree about whether a generation landed). Post-fix the header is
// either:
//
// - **Empty** (key absent) when no write has landed, or
// - **Complete** (all 17 fields present — b.vnw added `restore_seq_bump` as
//   the third cursor column, and the children-index completeness marker
//   `chs_complete` and the `commit_chain` fence identity ride the same HSET)
//   when `apply_checkpoint` succeeded.
//
// It can NEVER be in a 1-field partial state. This test asserts the complete
// shape directly after a successful apply, pinning every field name so a
// future regression that re-introduces a separate HSET (or drops a field
// from the bundled HSET) surfaces here.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_lands_complete_header_atomically_with_the_checkpoint_cursor() {
    // b.vnw — the band this restore will apply, chosen so the stamped
    // `restore_seq_bump` can ONLY be the band. It dominates the configured
    // margin (so the margin cannot produce it), it is nowhere near `snap_seq`
    // (7) or any commit count in the fixture, and it stays under the
    // outsized-band WARN threshold (8 × margin = 8_000) so this case does not
    // incidentally exercise a different branch of apply.
    const MEASURED_SPAN: u64 = 4_242;
    const SAFETY_MARGIN: u64 = 1_000;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "atomic-header-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    // The assembler declines to measure on a layer's first generation and
    // stamps the no-measurement sentinel (b.vnw round-2). Pin that, then
    // override: with the sentinel in place the band would fall back to the
    // margin, and a stamp equal to the margin is a value the column could
    // also carry by accident.
    assert_eq!(
        meta.observed_commit_span, 0,
        "test precondition: the assembler must stamp the no-measurement \
         sentinel on a layer's first generation (last_checkpoint_seq == 0)"
    );
    meta.observed_commit_span = MEASURED_SPAN;
    crate::common::wipe_object_keys(&tv, oid).await;

    let config = test_config(SAFETY_MARGIN);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply_checkpoint must succeed on a wiped obj keyspace");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr after apply_checkpoint");

    // The header MUST contain exactly the 17-field complete set — 12
    // canonical fields, the children-index completeness marker
    // (`chs_complete`, which `queue_object_state` stamps on every
    // materialization so the restored `obj:chs` is trusted by readers), the
    // fence identity (`commit_chain`, a fresh random value every
    // materialization writes, a restore included), plus
    // ALL THREE checkpoint-cursor columns (`last_checkpoint_seq`, the audit
    // stamp; `last_checkpoint_ms`, the cadence stamp the periodic loop's
    // interval arm measures against; and `restore_seq_bump` (b.vnw), the guard
    // band this restore applied, which the next assembly subtracts so the
    // artificial sequence jump is not measured as commit traffic).
    // Listing every expected name pins the canonical surface so a future
    // regression that drops a field from the bundled HSET surfaces here rather
    // than silently propagating to downstream consumers (snapshot loaders, the
    // assembler, the orphan-sweep classifier, the checkpoint gate).
    let expected: std::collections::HashSet<&'static str> = [
        "seq",
        "first_ver",
        "src_fmt",
        "next_set_order",
        "session_id",
        "layer_name",
        "created_by",
        "created_at",
        "modified_at",
        "last_client_id",
        "last_client_ts",
        "last_server_ts",
        HEADER_FIELD_CHS_COMPLETE,
        HEADER_FIELD_COMMIT_CHAIN,
        "last_checkpoint_seq",
        "last_checkpoint_ms",
        "restore_seq_bump",
    ]
    .into_iter()
    .collect();
    let observed: std::collections::HashSet<&str> = header.keys().map(String::as_str).collect();
    assert_eq!(
        observed,
        expected,
        "apply_checkpoint MUST land a complete 17-field header (12 canonical + \
         the children-index marker + the commit_chain fence identity + the \
         three-column checkpoint cursor) \
         atomically; got {observed:?} \
         (missing: {:?}, extra: {:?})",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(&expected).collect::<Vec<_>>()
    );

    // The skeletal 1-field shape the pre-fix race produced — a header
    // containing ONLY `last_checkpoint_seq` — MUST never appear after a
    // successful apply. Pin this directly so a regression that re-splits
    // the header writes (and races the structural HSET losing to a wipe)
    // surfaces as a clear failure even if the field count happens to
    // match.
    assert!(
        header.contains_key("seq")
            && header.contains_key("session_id")
            && header.contains_key("layer_name"),
        "post-apply header MUST contain the structural fields (seq, session_id, \
         layer_name); a skeletal `last_checkpoint_seq`-only header indicates \
         the b.4fh partial-write window has regressed; got: {header:?}"
    );

    // Pin the load-bearing values too: `last_checkpoint_seq=snap_seq` (the
    // audit-trail invariant in apply.rs Step 5's doc-comment) and the
    // session-identity fields (so a reordering of the HSET tuple cannot
    // silently swap field/value pairings — `Vec<(&str, String)>` ordering
    // is not what HSET reads; the pair structure is — but a future
    // refactor that switches to a flat `[k, v, k, v]` vec could go wrong).
    assert_eq!(
        header.get("last_checkpoint_seq").map(String::as_str),
        Some(snap_seq.to_string().as_str()),
        "last_checkpoint_seq must equal snap_seq inside the atomic HSET"
    );

    // b.yut — `last_checkpoint_ms` must carry the restored blob's OWN recorded
    // write time, not `now` and not a sentinel. That column is the periodic
    // loop's cadence reference ("when did the newest durable copy of this
    // layer land"), and post-restore the newest durable copy IS this blob,
    // written when it says it was. Stamping `now` instead would silently reset
    // the cadence clock on every recovery, deferring the restored layer's next
    // interval-arm eligibility by a full LIVE_EDIT_CHECKPOINT_INTERVAL_SECS.
    //
    // The assembler stamps a real wall clock, so the fixture's meta is
    // non-zero here; that is asserted as an explicit precondition because the
    // `== 0` case takes a DIFFERENT branch (the now_unix_ms fallback), covered
    // by `apply_checkpoint_falls_back_to_now_when_the_blob_records_no_write_time`.
    assert_ne!(
        meta.written_at_unix_ms, 0,
        "test precondition: the assembler must stamp a real write time, \
         otherwise this case is silently exercising the fallback branch instead \
         of the pass-through one"
    );
    assert_eq!(
        header.get("last_checkpoint_ms").map(String::as_str),
        Some(meta.written_at_unix_ms.to_string().as_str()),
        "last_checkpoint_ms must equal the restored blob's \
         CheckpointMeta::written_at_unix_ms ({}) inside the same atomic HSET",
        meta.written_at_unix_ms
    );

    // b.vnw — `restore_seq_bump` must land in the SAME HSET carrying the
    // seeded `seq` it explains. The two are a matched pair: `seq` jumped
    // `band` above the restore floor, and this column is the record of by how
    // much, so the next assembly can subtract it instead of measuring the
    // jump as commit traffic. A crash between two separate writes would leave
    // a header whose sequence jump has no explanation, and the next
    // `observed_commit_span` would over-measure by a full band — inflating
    // the band on the NEXT restore, compounding on every cycle.
    //
    // The value is `band = max(safety_margin, meta.observed_commit_span)` =
    // `max(1_000, 4_242)` = 4_242 — a number that cannot be produced by the
    // margin, by `snap_seq`, or by any commit count this fixture generates.
    // Pinning a DISTINCTIVE value is what makes the same-HSET claim mean
    // something: if the band ever fell back to the margin here, or the column
    // were stamped from `seeded_seq − snap_seq`, or written by a second HSET
    // that a wipe could lose, the assertion moves.
    assert_eq!(
        header.get("restore_seq_bump").map(String::as_str),
        Some(MEASURED_SPAN.to_string().as_str()),
        "restore_seq_bump must equal the band the restore applied \
         (max(safety_margin={SAFETY_MARGIN}, observed_commit_span={MEASURED_SPAN}) \
         = {MEASURED_SPAN}) inside the same atomic HSET as the seeded seq. The \
         band-vs-margin arithmetic itself is Test 14's subject; what this case \
         pins is that the column lands in the SAME HSET as the seq it explains"
    );
    // …and the `seq` it explains is in that same hash, exactly a band above
    // the restore floor (`max_seq_observed = None`, so the floor is
    // `snap_seq`). The pair is the whole point of bundling them: a header
    // carrying the jump without the column that accounts for it would make
    // the next assembly measure a full band of artificial jump as commit
    // traffic.
    assert_eq!(
        header.get("seq").map(String::as_str),
        Some((snap_seq + MEASURED_SPAN).to_string().as_str()),
        "the seeded seq and the bump that explains it must land together: \
         snap_seq({snap_seq}) + band({MEASURED_SPAN})"
    );

    assert_eq!(
        header.get("session_id").map(String::as_str),
        Some(session_id.as_str())
    );
    assert_eq!(
        header.get("layer_name").map(String::as_str),
        Some("atomic-header-layer")
    );
}

// =============================================================================
// Test 11b — b.yut: a blob that records NO write time falls back to `now`,
//            never to a literal 0
// =============================================================================
//
// `CheckpointMeta::written_at_unix_ms == 0` means the adapter did not record a
// write time. It is reachable two ways: `CheckpointMeta::default()`, and the
// LocalDisk adapter's meta-sidecar-missing path (a `<oid>.<snap_seq>.ompf`
// whose `.meta.json` sidecar was lost or never written).
//
// Writing that 0 through to `obj:hdr last_checkpoint_ms` would be read by the
// periodic loop's interval arm as "never checkpointed", making the restored
// layer due on the first eligible tick forever. `now_unix_ms()` is the honest
// substitute: we do not know when the blob was written, but we DO know a
// durable copy of this layer exists as of right now, so the cadence clock
// starts here.
//
// The fixture takes the cheapest deterministic route to the branch — a real
// assembled blob whose meta is copied with `written_at_unix_ms` zeroed — so no
// adapter surgery or sidecar deletion is needed and the rest of the apply path
// is exercised exactly as in Test 11.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_falls_back_to_now_when_the_blob_records_no_write_time() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "no-write-time-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    // The branch under test: an adapter that recorded no write time.
    let meta_without_write_time = CheckpointMeta {
        written_at_unix_ms: 0,
        ..meta
    };

    let before_apply_ms = now_unix_ms();
    let config = test_config(1);
    apply_checkpoint(
        &ls,
        oid,
        ompf,
        snap_seq,
        &meta_without_write_time,
        None,
        &config,
    )
    .await
    .expect("apply_checkpoint must succeed on a wiped obj keyspace");
    let after_apply_ms = now_unix_ms();

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let stamped: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_ms")
        .await
        .expect("HGET last_checkpoint_ms must succeed");
    let stamped_ms: u64 = stamped
        .as_deref()
        .unwrap_or_else(|| {
            panic!(
                "apply_checkpoint MUST stamp last_checkpoint_ms even when the \
                 blob records no write time — leaving it absent reads as \
                 'never checkpointed' to the periodic loop's interval arm"
            )
        })
        .parse()
        .expect("last_checkpoint_ms must be numeric");

    assert_ne!(
        stamped_ms, 0,
        "a zero `written_at_unix_ms` must NOT be written through verbatim: the \
         gate reads 0 as 'never checkpointed' and would re-checkpoint the \
         restored layer on every eligible tick forever (b.yut)"
    );
    // Bracketed by clocks read either side of the call, so this is exact
    // rather than a tolerance window: the stamped value can only be a
    // `now_unix_ms()` read taken inside `apply_checkpoint`.
    assert!(
        (before_apply_ms..=after_apply_ms).contains(&stamped_ms),
        "last_checkpoint_ms ({stamped_ms}) must be a wall clock read DURING the \
         apply — it must fall in [{before_apply_ms}, {after_apply_ms}]"
    );
}

// =============================================================================
// Test 12 — create_layer (write_object with checkpoint_cursor=None) MUST NOT
//           populate ANY checkpoint-cursor column
// =============================================================================
//
// b.4fh fix added a `last_checkpoint_seq: Option<u64>` parameter to
// `write_object`; b.yut widened it to a `checkpoint_cursor:
// Option<CheckpointCursorStamp>` carrying both cursor columns, and b.vnw added
// a third (`restore_seq_bump`) to the same struct.
// `create_layer` passes `None` because a freshly created layer has never been
// checkpointed and has never been restored — no cursor field may appear on the
// header until the first `apply_checkpoint` recovery or the first periodic
// checkpoint landing stamps them.
//
// Encoding choice the test pins: `None` means the HSET tuple omits every
// cursor pair entirely (i.e. the field keys are ABSENT from the hash), NOT
// "present with empty string" or "present with sentinel 0". Downstream
// consumers distinguish "never checkpointed" from a real cursor by exactly
// this absence — the skip-if-no-commits gate for `last_checkpoint_seq`, and
// the interval arm for `last_checkpoint_ms` (absent → 0 → due on the first
// eligible tick, which is the intended behaviour for a brand-new layer).
// `restore_seq_bump` reads absent → 0 in the assembler, which is the truthful
// "no restore bump is outstanding against this header" (b.vnw); a fabricated
// non-zero there would deflate the layer's first measured commit span.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_omits_every_checkpoint_cursor_field_on_fresh_creation() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "fresh-create-no-cp-seq-layer",
            "create-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr after create_layer");

    // The 12 canonical fields MUST be present so the layer is structurally
    // complete and the snapshot loader / assembler can run against it.
    for field in [
        "seq",
        "first_ver",
        "src_fmt",
        "next_set_order",
        "session_id",
        "layer_name",
        "created_by",
        "created_at",
        "modified_at",
        "last_client_id",
        "last_client_ts",
        "last_server_ts",
    ] {
        assert!(
            header.contains_key(field),
            "create_layer must stamp canonical field `{field}`; header: {header:?}"
        );
    }

    // EVERY cursor column MUST be ABSENT — None encoding is "field omitted",
    // not "field=0" or "field=''". The periodic-checkpoint skip-if-no-commits
    // gate depends on this absence to distinguish "never checkpointed" from
    // "checkpointed at seq=0", and the interval arm reads an absent
    // `last_checkpoint_ms` as 0 = "never checkpointed".
    for field in [
        "last_checkpoint_seq",
        "last_checkpoint_ms",
        "restore_seq_bump",
    ] {
        assert!(
            !header.contains_key(field),
            "create_layer (write_object with checkpoint_cursor=None) MUST NOT \
             stamp `{field}` on obj:hdr; got value: {:?}",
            header.get(field)
        );

        // Belt-and-suspenders: HEXISTS-equivalent via direct HGET → None.
        let raw: Option<String> = tv
            .client()
            .pool
            .hget(&header_key, field)
            .await
            .expect("HGET checkpoint-cursor column after create_layer");
        assert!(
            raw.is_none(),
            "HGET {field} must return None on a freshly-created layer; got: {raw:?}"
        );
    }
}

// =============================================================================
// Test 13 — DR round-trip preserves each field's TRUE per-delta authoring rank
//           (Task t3.msb.js.6j — build_reset_ompf authoring-rank fix)
// =============================================================================
//
// The DR checkpoint serialize→restore round-trip must carry each field's raw
// per-delta `authoring_set_order` through unchanged — it must NOT collapse to
// the cumulative `set_order` cursor. Pre-fix, `build_reset_ompf` routed the
// snapshot through the legacy cumulative planner, so on restore
// `decode_checkpoint_ompf` → `apply_delta` recomputed `authoring_set_order`
// from the cumulative `set_order` the reset delta carried, erasing the real
// authoring order. The Engineer's `build_reset_ompf` change remaps each
// emitted field's `set_order = authoring_set_order` before `build_delta`, so
// the raw rank survives re-apply.
//
// Fixture (the load-bearing part): a scene where the stored CUMULATIVE
// `set_order` diverges from the per-delta `authoring_set_order`, built the way
// the real writer produces divergence — a burst of `update_layer` commits, each
// carrying its own small per-delta `set_order`, across an ever-advancing
// `next_set_order` cursor (mirrors
// `subscribe_regimes_test::snapshot_late_join_catch_up_establishes_correct_scene`).
//
//   commit A: sec1 "name"  authoring rank 1  → stored cumulative 1  (no divergence)
//   commit B: sec2 "posX"  authoring rank 5  → stored cumulative advances
//   commit C: sec2 "posX"  authoring rank 2  → last-write-wins; stored cumulative
//             now folds in the advanced cursor (e.g. 10) while the authoring
//             rank stays 2 — the divergent, load-bearing field.
//
// Anti-regression assertion: after restore, `posX.authoring_set_order` must
// equal its ORIGINAL rank (2) and must NOT equal the pre-checkpoint cumulative
// value it collapsed to pre-fix. Pre-checkpoint authoring/cumulative are read
// out of `read_object_state_for_test` rather than hardcoded, so the test is robust to
// the exact `next_set_order` advance mechanics — it only requires (and asserts)
// that the fixture actually diverged.
//
// This also regression-pins the structure / final-value surface (sections,
// children, field payloads, time samples, source_format, header identity) so
// the authoring-rank fix leaves restore correctness otherwise unchanged.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_preserves_per_field_authoring_rank() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Step 1: create the structural seed (sections + children + time samples,
    // NO fields, next_set_order=0).
    let seed = divergent_seed_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "authoring-rank-layer",
            "apply-test-user",
            &seed,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Step 2: burst-write fields so the cumulative set_order diverges from the
    // per-delta authoring rank. Each commit is its own single-field OwnedDelta
    // carrying its own per-delta set_order, advancing next_set_order between
    // writes — exactly how the real writer produces the divergence.
    ls.update_layer_relayed(
        &session_id,
        "authoring-rank-layer",
        oid,
        Some(delta_with_set_field_inline(1, "name", b"scene".to_vec(), 1)),
        "writer-a",
        100,
    )
    .await
    .expect("commit A (name @ rank 1) must succeed");
    ls.update_layer_relayed(
        &session_id,
        "authoring-rank-layer",
        oid,
        Some(delta_with_set_field_inline(2, "posX", b"first".to_vec(), 5)),
        "writer-b",
        200,
    )
    .await
    .expect("commit B (posX @ rank 5, initial value) must succeed");
    ls.update_layer_relayed(
        &session_id,
        "authoring-rank-layer",
        oid,
        Some(delta_with_set_field_inline(2, "posX", b"final".to_vec(), 2)),
        "writer-c",
        300,
    )
    .await
    .expect("commit C (posX @ rank 2, final value — last write wins) must succeed");

    // Step 3: snapshot the live pre-checkpoint state and capture the authoring /
    // cumulative ranks. The fixture MUST actually diverge for this test to mean
    // anything — assert that up front.
    let pre = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("pre-checkpoint read_object_state_for_test must succeed");
    let pre_field = |sid: u64, key: &str| -> Field {
        pre.fields
            .iter()
            .find(|f| f.section_id == sid && f.key_name == key)
            .unwrap_or_else(|| panic!("pre-checkpoint field ({sid},{key}) must exist"))
            .clone()
    };
    let pre_name = pre_field(1, "name");
    let pre_pos_x = pre_field(2, "posX");
    assert_eq!(
        pre_pos_x.authoring_set_order, 2,
        "posX authoring rank is the raw last-write per-delta set_order"
    );
    assert!(
        pre_pos_x.set_order > pre_pos_x.authoring_set_order,
        "fixture invariant: posX stored cumulative set_order ({}) must exceed its \
         authoring rank ({}) — otherwise the test cannot detect the collapse-on-restore bug",
        pre_pos_x.set_order,
        pre_pos_x.authoring_set_order
    );
    assert_eq!(
        pre_name.authoring_set_order, 1,
        "name authoring rank is its per-delta set_order"
    );
    assert!(
        pre_name.authoring_set_order < pre_pos_x.authoring_set_order,
        "fixture invariant: name authoring rank must sort before posX authoring rank"
    );

    // Step 4: assemble a checkpoint, wipe the obj keyspace, apply it back.
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;
    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply_checkpoint must succeed on a wiped obj keyspace");

    // Step 5: load the restored state back out.
    let post = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("post-apply read_object_state_for_test must succeed");
    let post_field = |sid: u64, key: &str| -> Field {
        post.fields
            .iter()
            .find(|f| f.section_id == sid && f.key_name == key)
            .unwrap_or_else(|| panic!("restored field ({sid},{key}) must exist"))
            .clone()
    };
    let post_name = post_field(1, "name");
    let post_pos_x = post_field(2, "posX");

    // ── (a) The load-bearing authoring-rank assertions. ──
    assert_eq!(
        post_pos_x.authoring_set_order, pre_pos_x.authoring_set_order,
        "restored posX authoring rank must equal the original per-delta rank ({}), \
         NOT be recomputed from the cumulative cursor on re-apply",
        pre_pos_x.authoring_set_order
    );
    // The exact anti-regression pin: pre-fix, restore collapsed the authoring
    // rank to the cumulative set_order the reset delta carried. Assert the
    // restored rank is NOT that cumulative value — this is what fails today.
    assert_ne!(
        post_pos_x.authoring_set_order, pre_pos_x.set_order,
        "restored posX authoring rank must NOT collapse to the pre-checkpoint \
         cumulative set_order ({}) — that collapse is exactly the DR bug this fix closes",
        pre_pos_x.set_order
    );
    assert_eq!(
        post_name.authoring_set_order, pre_name.authoring_set_order,
        "restored name authoring rank must equal its original per-delta rank"
    );
    // Relative authoring order across fields preserved after restore.
    assert!(
        post_name.authoring_set_order < post_pos_x.authoring_set_order,
        "restored authoring order must preserve name-rank ({}) < posX-rank ({})",
        post_name.authoring_set_order,
        post_pos_x.authoring_set_order
    );

    // ── (c) Structure / final-value regression pins — the authoring-rank fix
    //        must leave everything else about restore correctness unchanged. ──

    // Sections — same set, names, parents preserved.
    assert_eq!(post.sections.len(), 2, "two sections must round-trip");
    assert_eq!(post.sections[&1].name, "World");
    assert_eq!(post.sections[&1].parent_id, 0);
    assert_eq!(post.sections[&2].name, "Cube");
    assert_eq!(post.sections[&2].parent_id, 1);

    // Children — parent → child relationships preserved.
    assert_eq!(
        post.children.get(&0).map(|v| v.as_slice()),
        Some([1u64].as_slice()),
        "root children must round-trip"
    );
    assert_eq!(
        post.children.get(&1).map(|v| v.as_slice()),
        Some([2u64].as_slice()),
        "sec1 children must round-trip"
    );

    // Field final values — last-write-wins value on posX, name value preserved.
    assert_eq!(
        post_name.value,
        FieldValue::Inline(b"scene".to_vec()),
        "name field payload must round-trip"
    );
    assert_eq!(
        post_pos_x.value,
        FieldValue::Inline(b"final".to_vec()),
        "posX field payload must be the last-write-wins (commit C) value"
    );

    // Time samples — keys + payloads preserved.
    let ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = post
        .time_samples
        .iter()
        .map(|ts| ((ts.section_id, ts.time), ts.value.clone()))
        .collect();
    assert_eq!(ts_map.len(), 2, "two time samples must round-trip");
    assert_eq!(
        ts_map[&(1, OrderedFloat(0.0))],
        FieldValue::Inline(vec![0xAA, 0xBB])
    );
    assert_eq!(
        ts_map[&(2, OrderedFloat(1.5))],
        FieldValue::Inline(vec![0xCC])
    );

    // source_format round-trips via the reset delta.
    assert_eq!(
        post.source_format, "usda",
        "source_format must round-trip via the reset delta"
    );

    // Header session-identity fields restored from CheckpointMeta.
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL header must succeed");
    assert_eq!(
        header.get("session_id").map(String::as_str),
        Some(session_id.as_str()),
        "session_id must be reconstructed from CheckpointMeta"
    );
    assert_eq!(
        header.get("layer_name").map(String::as_str),
        Some("authoring-rank-layer")
    );
    assert_eq!(
        header.get("created_by").map(String::as_str),
        Some("apply-test-user")
    );
    assert_eq!(
        header.get("src_fmt").map(String::as_str),
        Some("usda"),
        "source_format must round-trip on the header via the reset delta"
    );
}

// =============================================================================
// b.vnw — the DR-restore guard band is measured per object, not configured
// =============================================================================
//
// Pre-b.vnw the band above the restore floor was
// `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` and nothing else. Its adequacy
// rested entirely on the operator having sized that one global knob against a
// peak they had to predict: the requirement is `margin >= peak_commits_per_sec
// × LIVE_EDIT_CHECKPOINT_INTERVAL_SECS`, and a layer that commits faster than
// the guess restores INTO sequence numbers subscribers already hold attached
// to now-lost content. That failure is silent — the restore succeeds, and the
// reuse only surfaces later as a subscriber dropping live updates whose
// `sequence_num <= tracked_version`.
//
// The fix carries the measurement in the blob: the assembler stamps
// `CheckpointMeta::observed_commit_span` = commits this generation covers that
// the previous one did not, and apply sizes the band as
// `max(configured margin, observed_commit_span)`.
//
// The five cases below pin the five distinct behaviours that makes:
//
//   14  span > margin           → the measurement sizes the band
//   15  span absent (legacy)    → byte-identical to pre-b.vnw behaviour
//   16  surviving stream probe  → the STAMP is the band, not the whole jump
//   17  post-restore checkpoint → the band is not re-measured as traffic
//   18  absurd span             → the f64 ceiling still refuses, and writes
//                                 nothing
//
// The complementary "the configured margin is a floor" direction (margin >
// span, both non-zero) is covered by
// `apply_checkpoint_advances_seq_to_max_seq_observed_plus_safety_margin`
// above, whose `test_config(50)` dominates the sub-margin span that case pins
// explicitly. The sentinel-0 direction (no measurement at all → the margin is
// the whole band) is covered by case 15 below.
//
// The `obj:hdr` column reads below go through `crate::common::read_header_u64`
// — shared with `checkpointer_interval_cadence_test.rs`, which reads the same
// cursor columns from the periodic loop's side of the same contract.

// =============================================================================
// Test 14 — b.vnw: a measured span ABOVE the configured margin sizes the band
// =============================================================================
//
// The headline case. `margin = 10` is the operator's guess; the blob measured
// 500 commits over the interval it covers. The band must be 500, so the
// restore seeds 500 above its floor rather than 10 — the difference between
// clearing the lost-commit window and landing inside it.
//
// `max_seq_observed = None` deliberately: that is the full-keyspace-loss
// shape, where the stream is gone and the probe contributes nothing, so the
// band is the WHOLE defense. It is also the shape where the pre-b.vnw
// behaviour was weakest.
//
// The fixture first pins the assembler's natural measurement — the
// no-measurement sentinel 0, because this is the layer's first generation
// (b.vnw round-2) — before overriding it, so the override is demonstrably what
// produces the band below rather than an accident of the fixture. The
// end-to-end measurement path (a real non-zero span between two consecutive
// generations) is `assembler_measures_the_span_between_consecutive_generations`
// and `post_restore_recheckpoint_measures_commits_not_the_restore_band`.
//
// The 10 / 500 pairing also puts this case 50× above the configured margin,
// i.e. well past `apply::OUTSIZED_BAND_MARGIN_MULTIPLE` (8×) — so it doubles
// as the pin on the round-2 decision that the outsized band is REPORTED and
// not CLAMPED. That distinction is the safety-relevant half of the WARN: an
// oversized span is usually an honest measurement of a post-outage generation,
// approximating the very commit range a restore lost, and clamping it to 8×
// margin (80 → seeded 87 here) would re-open the sequence-reuse window the
// band exists to close. The WARN line itself is deliberately not asserted —
// it is operator signal, not contract, and log-text assertions are what the
// test-writing guide's "What NOT to Test" rules out.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_seeds_from_observed_commit_span_when_it_exceeds_the_margin() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "span-dominates-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");

    // The assembler's own measurement, before any override: this is a first
    // generation (`last_checkpoint_seq` absent → 0), so the assembler DECLINES
    // to measure and stamps the no-measurement sentinel (b.vnw round-2) rather
    // than reporting commits-since-creation as if it were the per-interval rate.
    assert_eq!(
        meta.observed_commit_span, 0,
        "the assembler must stamp the no-measurement sentinel 0 on a layer's \
         first generation (last_checkpoint_seq == 0)"
    );

    crate::common::wipe_object_keys(&tv, oid).await;

    // Override to a span far above any plausible margin so the arithmetic is
    // unambiguous: 500 could not be produced by `max()` picking the margin,
    // by the restore floor, or by any sum of the two.
    meta.observed_commit_span = 500;
    let config = test_config(10);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply must succeed");

    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(507),
        "b.vnw: seeded_seq must be restore_floor + max(margin=10, span=500) \
         = max(snap_seq=7, 0) + 500 = 507. A value of 17 means the measured \
         span was ignored and the band fell back to the configured margin, \
         leaving the restore inside the window of commits lost with the \
         crashed state; a value of 87 means the outsized band was CLAMPED to \
         OUTSIZED_BAND_MARGIN_MULTIPLE (8) × margin instead of merely WARNed \
         about, which re-opens the same window more quietly"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(500),
        "the band applied must be recorded on the header so the next assembly \
         can subtract it"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "last_checkpoint_seq").await,
        Some(7),
        "last_checkpoint_seq stays snap_seq regardless of which input sized \
         the band — it is the audit trail for WHICH generation produced the \
         state, not for where the writer resumed"
    );
}

// =============================================================================
// Test 15 — b.vnw back-compat: a legacy blob restores exactly as it did before
// =============================================================================
//
// Every checkpoint written before b.vnw has no `observed_commit_span` key in
// its meta JSON. `#[serde(default)]` deserializes those to 0, and 0 folds the
// band back to the configured margin alone — so a restore from a pre-b.vnw
// blob must be identical to pre-b.vnw behaviour. Without that, the fix would
// be a silent behaviour change on exactly the blobs an operator reaches for
// during a real incident.
//
// The legacy meta is produced the honest way — serialize a current
// `CheckpointMeta`, REMOVE the key from the JSON object, deserialize — so the
// test exercises the actual serde contract rather than asserting against a
// hand-set struct field. That also makes this the serde back-compat case: the
// `assert_eq!(legacy.observed_commit_span, 0)` below is the whole claim of
// `#[serde(default)]`, checked at the seam that consumes it.
//
// The apply inputs are those of
// `apply_checkpoint_seeds_snap_seq_plus_margin_when_max_seq_observed_is_none`
// (margin=1, snap_seq=7, no probe), and the expected seeded seq is the same 8,
// deliberately: "unchanged from pre-b.vnw" is most convincingly stated as
// reproducing that case's pinned number through a blob that genuinely lacks
// the field.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_falls_back_to_the_configured_margin_for_a_legacy_blob() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "legacy-blob-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");
    // This fixture is the layer's first generation, so the assembler stamps the
    // no-measurement sentinel 0 (b.vnw round-2). Inject a non-zero span before
    // serializing: stripping the key below has to change the deserialized value
    // for this case to prove anything about the legacy shape, and a `0 → 0`
    // strip would pass whether `#[serde(default)]` were present or not.
    meta.observed_commit_span = 500;
    crate::common::wipe_object_keys(&tv, oid).await;

    // Forge a pre-b.vnw meta JSON: serialize the current shape, drop the key
    // that did not exist then, parse it back through the production
    // `Deserialize` impl the LocalDisk / StorageApi adapters use.
    let mut json: serde_json::Value =
        serde_json::to_value(&meta).expect("CheckpointMeta must serialize");
    let removed = json
        .as_object_mut()
        .expect("CheckpointMeta serializes as a JSON object")
        .remove("observed_commit_span");
    assert!(
        removed.is_some(),
        "the field must be present before removal — if serde ever renamed or \
         skipped it, this case would silently stop testing the legacy shape"
    );
    let legacy: CheckpointMeta =
        serde_json::from_value(json).expect("a pre-b.vnw meta JSON must still deserialize");
    assert_eq!(
        legacy.observed_commit_span, 0,
        "b.vnw serde back-compat: a meta JSON with no observed_commit_span key \
         MUST deserialize to 0 (`#[serde(default)]`), the 'no measurement \
         available' sentinel apply falls back on. Any other value — or a \
         deserialize error — breaks every checkpoint written before b.vnw, \
         i.e. exactly the blobs a real incident reaches for"
    );

    let config = test_config(1);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &legacy, None, &config)
        .await
        .expect("a legacy blob must still apply");

    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(8),
        "a legacy blob must restore EXACTLY as it did pre-b.vnw: \
         restore_floor(7) + margin(1) = 8, identical to \
         apply_checkpoint_seeds_snap_seq_plus_margin_when_max_seq_observed_is_none"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(1),
        "the stamped bump is the band that was actually applied — the \
         configured margin, on this path — not a sentinel 0. Stamping 0 here \
         would make the next assembly measure the margin-sized jump as commit \
         traffic and inflate the band on the next restore"
    );
}

// =============================================================================
// Test 16 — b.vnw: the stamp is the BAND, not the whole seq jump
// =============================================================================
//
// When the stream probe survives, `restore_floor` sits above `snap_seq` and
// the total jump is `(restore_floor − snap_seq) + band`. Only the second term
// is artificial. The first is real commit traffic the probe PROVED had landed
// past the checkpoint, so it belongs in the next generation's measured span;
// subtracting it out would under-count the layer's real rate and shrink the
// band on the next restore.
//
// So the header column must record `band`, not `seeded_seq − snap_seq`. With
// `snap_seq = 7`, `max_seq_observed = Some(1000)`, `margin = 10`,
// `span = 500`: floor = 1000, band = 500, seeded = 1500, and the two candidate
// stamps (500 vs 1493) are far enough apart that neither can be mistaken for
// the other or produced by an off-by-one.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_stamps_the_band_not_the_full_seq_jump_when_the_probe_survives() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "probe-survives-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");
    crate::common::wipe_object_keys(&tv, oid).await;

    meta.observed_commit_span = 500;
    let config = test_config(10);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, Some(1000), &config)
        .await
        .expect("apply must succeed");

    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(1500),
        "seeded_seq = max(snap_seq=7, max_seq_observed=1000) + \
         max(margin=10, span=500) = 1000 + 500"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(500),
        "b.vnw: the stamp is the BAND (500), NOT the full jump \
         seeded_seq − snap_seq (1493). The 993 commits the stream probe proved \
         had landed past the checkpoint are real traffic and must stay inside \
         the next generation's measured span; recording them here would \
         subtract them out and under-size the band on the next restore"
    );
}

// =============================================================================
// Test 17 — b.vnw: a post-restore re-checkpoint measures commits, not the band
// =============================================================================
//
// The end-to-end proof that the `CheckpointCursorStamp` → `HeaderIdentity` →
// `observed_commit_span` subtraction actually closes. Without it the fix eats
// itself: a restore stamps `last_checkpoint_seq = snap_seq` while seeding
// `seq = floor + band`, so the very next assembly would read a raw difference
// of `band + real commits` and stamp a span inflated by a full band. The next
// restore would apply THAT as its band, and the number would compound on every
// restore/checkpoint cycle until the `F64_SAFE_INT_MAX` guard caught it.
//
// Sequence: restore with a known band of 200 → three real `update_layer`
// commits → re-assemble → the new blob's span must be 3.
//
// `k = 3` and `band = 200` are chosen far apart so the two candidate answers
// (3 vs 203) cannot be confused, and both differ from every other number in
// play (snap_seq=7, seeded=207).

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn post_restore_recheckpoint_measures_commits_not_the_restore_band() {
    const BAND: u64 = 200;
    const COMMITS_AFTER_RESTORE: u64 = 3;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer_name = "post-restore-span-layer";

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(snap_seq, 7, "snap_seq pin");
    crate::common::wipe_object_keys(&tv, oid).await;

    // Restore with a band of exactly BAND. The span is held at 0 so the band
    // comes from the configured margin alone — a deterministic input, rather
    // than one that moves if `rich_object_state`'s commit count ever changes.
    meta.observed_commit_span = 0;
    let config = test_config(BAND);
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("apply must succeed");

    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(snap_seq + BAND),
        "restore precondition: seq seeded a full band above the floor"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(BAND),
        "restore precondition: the band is recorded for the next assembly"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "last_checkpoint_seq").await,
        Some(snap_seq),
        "restore precondition: the cursor points at the restored generation, \
         NOT at the seeded seq — which is precisely why the raw difference on \
         the next assembly contains the band"
    );

    // Real commits through the production write path, so the seq advance is
    // produced the way it is in a live fleet rather than by an HSET.
    for i in 0..COMMITS_AFTER_RESTORE {
        ls.update_layer_relayed(
            &session_id,
            layer_name,
            oid,
            Some(delta_with_set_field_inline(
                1,
                "name",
                format!("post-restore-{i}").into_bytes(),
                1,
            )),
            "post-restore-writer",
            1_000 + i,
        )
        .await
        .unwrap_or_else(|e| panic!("post-restore commit {i} must succeed: {e:?}"));
    }

    let expected_seq = snap_seq + BAND + COMMITS_AFTER_RESTORE;
    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(expected_seq),
        "fixture invariant: {COMMITS_AFTER_RESTORE} commits must advance seq \
         by exactly {COMMITS_AFTER_RESTORE}"
    );

    // Re-assemble. The assembler reads `last_checkpoint_seq` (7) and
    // `restore_seq_bump` (200) off the same header read it measures `snap_seq`
    // against, and subtracts both.
    let (_, next_snap_seq, next_meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(
        next_snap_seq, expected_seq,
        "the re-assembly must cut at the post-commit seq"
    );
    assert_eq!(
        next_meta.observed_commit_span,
        COMMITS_AFTER_RESTORE,
        "b.vnw: the post-restore generation's span must be the \
         {COMMITS_AFTER_RESTORE} real commits — snap_seq({expected_seq}) − \
         last_checkpoint_seq({snap_seq}) − restore_seq_bump({BAND}). A value of \
         {} means the restore's artificial jump was measured as commit \
         traffic, which would size the NEXT restore's band from it and \
         compound on every restore/checkpoint cycle",
        BAND + COMMITS_AFTER_RESTORE
    );
}

// =============================================================================
// Test 18 — b.vnw: an absurd span still hits the f64 ceiling refusal, and the
//           refusal writes NOTHING
// =============================================================================
//
// b.vnw is what makes this path reachable from DATA. Before it, `band` came
// only from `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` — an operator-set knob
// validated at startup — so reaching `2^53 - 1` required either a runaway
// writer pushing `max_seq_observed` there or a deliberately absurd config.
// Now the band is sourced from a field inside the blob, and a corrupted or
// hand-edited `.meta.json` can carry any `u64` straight into the arithmetic.
//
// Two things must hold, and the second is the one that matters:
//
//   1. The refusal is `Permanent`, not `Transient`. Retrying the apply
//      against the same bytes hits the same ceiling forever; classifying it
//      retriable would spin the recovery path indefinitely instead of
//      surfacing a corrupt blob to the operator.
//   2. NOTHING is materialized. `saturating_add` means an absurd span
//      produces `u64::MAX` rather than wrapping to a small number, so the
//      check fires — but if the refusal ran AFTER `write_object`, the layer
//      would come back with a sequence number both commit paths reject, and
//      every subsequent UpdateLayer against it would fail. That is strictly
//      worse than no restore at all: the layer looks recovered and is
//      permanently unwritable.
//
// `u64::MAX` is used rather than a value just above the ceiling precisely to
// exercise the saturating arm — the one the "so `seeded_seq == restore_floor
// + band` exactly on this path" comment in apply.rs step 5 depends on being
// unreachable past this check.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_rejects_an_absurd_observed_commit_span_without_writing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "absurd-span-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, mut meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    meta.observed_commit_span = u64::MAX;
    let config = test_config(1);
    let err = apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect_err("a span that drives seeded_seq past 2^53-1 must be refused");
    assert!(
        matches!(err, CheckpointError::Permanent { .. }),
        "the ceiling refusal must classify Permanent — the same bytes will \
         reach the same ceiling on every retry, so a Transient here spins the \
         recovery path instead of surfacing the corrupt blob; got {err:?}"
    );

    assert!(
        !obj_header_exists(&tv, oid).await,
        "the ceiling refusal must run BEFORE write_object: a materialized \
         header here means the layer was restored at a sequence number both \
         commit paths reject, so it would look recovered while every \
         subsequent UpdateLayer against it failed"
    );
    assert!(
        !any_write_object_state_key_exists(&tv, oid).await,
        "and no obj state key (secs/flds/tss/dels/ch:0) may be written either"
    );
}

// =============================================================================
// Test 19 — b.b2j: the guard is anchored to the BLOB's identity, not to
//           "some live mapping somewhere"
// =============================================================================
//
// The object-id reuse shape. A checkpoint blob was written for `object_id = N`
// under `(session_A, "foo")`. The Valkey keyspace was then lost under a
// surviving process, the counter restarted, and a DIFFERENT layer —
// `(session_B, "bar")` — was minted at the same `N`. On the DR path
// `get_latest_layer_checkpoint(N)` still returns session_A's blob.
//
// The pre-b.b2j guard asked "does SOME live `le:{*}:layer:*` mapping resolve
// to N?" — and here one does, `(session_B, "bar")`. So it passed, and
// `apply_checkpoint` materialized session_A's layer content under session_B's
// layer identity: a silent cross-layer data leak, the highest-severity
// outcome in the ticket.
//
// The corrected guard reads exactly one key, `le:{meta.session_id}:layer:
// {hex(meta.layer_name)}`, and requires it to equal N. There is no
// RenameLayer RPC, so `(session_id, layer_name)` is immutable for a given
// `object_id`: the targeted read is equivalent to the old scan in the honest
// case and strictly stronger under reuse.
//
// This test is a regression guard in its own right — it fails against the
// scan-based guard, independently of whether the counter fix is present.
// The positive control at the end proves the rejection is discriminating
// rather than blanket: the SAME blob applies cleanly once the mapping that
// resolves to N is the blob's own.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_rejects_blob_whose_identity_maps_to_a_different_layer() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);

    // Two sessions. `create_test_session`'s fixed name/base-path pair would
    // collide on the composite-name uniqueness index, so session B is created
    // directly with its own name and path.
    let session_a = create_test_session(&tv).await;
    let session_b = session_store(&tv)
        .create("apply-test-session-b", "/stage-b", "apply-test-user")
        .await
        .expect("second session creation must succeed");

    // Layer (session_a, "foo") → N, checkpointed.
    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_a,
            "foo",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    assert_eq!(
        (meta.session_id.as_str(), meta.layer_name.as_str()),
        (session_a.as_str(), "foo"),
        "fixture precondition: the assembler must stamp the blob with the layer's own \
         identity — the guard has nothing to anchor on otherwise"
    );

    crate::common::wipe_object_keys(&tv, oid).await;

    // Simulate the reuse: session_A's mapping is gone with the keyspace, and
    // a DIFFERENT layer now holds the same object_id.
    let mapping_a = tv.key_schema().layer_mapping(&session_a, "foo");
    let mapping_b = tv.key_schema().layer_mapping(&session_b, "bar");
    let _: () = tv
        .client()
        .pool
        .del(&mapping_a)
        .await
        .expect("DEL of session_a's mapping must succeed");
    let _: () = tv
        .client()
        .pool
        .set(&mapping_b, oid.to_string(), None, None, false)
        .await
        .expect("SET of session_b's mapping must succeed");

    // Sanity: a scan of live mappings WOULD find one resolving to `oid`.
    // Without this the test could pass for the trivial reason that no mapping
    // exists at all — i.e. it would degenerate into Test 2.
    let mapped_b: Option<String> = tv
        .client()
        .pool
        .get(&mapping_b)
        .await
        .expect("GET session_b mapping must succeed");
    assert_eq!(
        mapped_b.as_deref(),
        Some(oid.to_string().as_str()),
        "fixture precondition: a live mapping MUST resolve to the reused object_id, \
         otherwise this test does not discriminate the identity guard from the old \
         any-mapping scan"
    );

    let config = test_config(1000);
    let err = apply_checkpoint(&ls, oid, ompf.clone(), snap_seq, &meta, None, &config)
        .await
        .expect_err(
            "applying session_a's blob while object_id belongs to session_b's layer must \
             fail — materializing it would surface one layer's content under another \
             layer's identity",
        );
    assert!(
        matches!(err, CheckpointError::NotFound),
        "the identity-guard rejection must surface NotFound (→ \
         RecoveryOutcome::NotRecoverable, leaving the recovery-attempt metric \
         vocabulary unchanged); got {err:?}"
    );

    assert!(
        !obj_header_exists(&tv, oid).await,
        "the rejected apply must NOT write obj:hdr. A header here is the cross-layer \
         corruption itself: session_b's layer name now resolves to a header carrying \
         session_a's content"
    );
    assert!(
        !any_write_object_state_key_exists(&tv, oid).await,
        "and no obj state key (secs/flds/tss/dels/ch:0) may be written either"
    );

    // ── Positive control: restore the blob's OWN mapping and the same bytes
    // apply cleanly. Without this the test could be satisfied by a guard that
    // rejects everything.
    let _: () = tv
        .client()
        .pool
        .del(&mapping_b)
        .await
        .expect("DEL of session_b's mapping must succeed");
    let _: () = tv
        .client()
        .pool
        .set(&mapping_a, oid.to_string(), None, None, false)
        .await
        .expect("re-SET of session_a's mapping must succeed");

    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect(
            "the SAME blob must apply once the mapping that resolves to object_id is the \
             blob's own — the guard rejects mismatched identity, not every restore",
        );
    assert!(
        obj_header_exists(&tv, oid).await,
        "the identity-matched apply must materialize obj:hdr"
    );
}

// =============================================================================
// Test 20 — b.b2j: a blob carrying no identity is refused
// =============================================================================
//
// Two production paths can produce a blob with an empty `session_id` or
// `layer_name`: the LocalDisk adapter's pre-b1 no-`meta.json` fallback, and
// the assembler's fabricated-baseline arm (taken when `obj:hdr` vanished
// mid-assembly). Such a blob carries nothing to verify against, so the guard
// has no way to prove the `object_id` still belongs to it.
//
// Both sub-cases leave a live, healthy mapping resolving to `object_id`, so
// the old any-mapping scan would have waved them through and written an
// identity-less header. The verdict must be `NotFound` (not `Permanent`) so
// the outcome stays `RecoveryOutcome::NotRecoverable` and the
// recovery-attempt metric label vocabulary is unchanged.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_rejects_blob_with_empty_identity() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let initial = rich_object_state();
    let oid = ls
        .create_layer(
            &session_id,
            "identityless-layer",
            "apply-test-user",
            &initial,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;

    // `wipe_object_keys` deliberately spares the layer mapping, so the live
    // mapping still resolves to `oid` throughout — the guard must refuse on
    // the blob's missing identity alone, not because the mapping is gone.
    crate::common::wipe_object_keys(&tv, oid).await;
    let mapped: Option<String> = tv
        .client()
        .pool
        .get(
            tv.key_schema()
                .layer_mapping(&session_id, "identityless-layer"),
        )
        .await
        .expect("GET layer mapping must succeed");
    assert_eq!(
        mapped.as_deref(),
        Some(oid.to_string().as_str()),
        "fixture precondition: the live mapping must still resolve to object_id, so the \
         only thing that can reject the apply is the blob's empty identity"
    );

    let config = test_config(1000);

    // Sub-case A: empty session_id.
    {
        let mut meta_a = meta.clone();
        meta_a.session_id = String::new();
        let err = apply_checkpoint(&ls, oid, ompf.clone(), snap_seq, &meta_a, None, &config)
            .await
            .expect_err("a blob with an empty session_id must be refused");
        assert!(
            matches!(err, CheckpointError::NotFound),
            "an unverifiable identity must surface NotFound so the hook reports \
             NotRecoverable rather than a new error class; got {err:?}"
        );
        assert!(
            !obj_header_exists(&tv, oid).await,
            "sub-case A: writing an identity-less obj:hdr is never the right outcome"
        );
    }

    // Sub-case B: empty layer_name.
    {
        let mut meta_b = meta.clone();
        meta_b.layer_name = String::new();
        let err = apply_checkpoint(&ls, oid, ompf.clone(), snap_seq, &meta_b, None, &config)
            .await
            .expect_err("a blob with an empty layer_name must be refused");
        assert!(
            matches!(err, CheckpointError::NotFound),
            "sub-case B must reject on the same NotFound arm as sub-case A; got {err:?}"
        );
        assert!(
            !obj_header_exists(&tv, oid).await,
            "sub-case B: no obj:hdr may be written"
        );
        assert!(
            !any_write_object_state_key_exists(&tv, oid).await,
            "sub-case B: no obj state key (secs/flds/tss/dels/ch:0) may be written"
        );
    }

    // Control: the same blob with its identity intact still applies, proving
    // the two rejections above are attributable to the emptied field.
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &config)
        .await
        .expect("the unmodified blob must apply against its own live mapping");
    assert!(
        obj_header_exists(&tv, oid).await,
        "the control apply must materialize obj:hdr"
    );
}

// =============================================================================
// Test 21 — b.6de sub-task 6: `HEADER_STRUCTURAL_FIELDS` is a drift-proof
// mirror of the read-path guards, not an independent list
// =============================================================================

/// [`HEADER_STRUCTURAL_FIELDS`] is exactly the set of header fields whose
/// absence a storage read path answers `NotFound` for — both directions:
///
/// * every LISTED field must be `NotFound`-guarded by at least one read path;
/// * every field production writes that is NOT listed must be guarded by
///   NEITHER.
///
/// # What this is defending
///
/// The list is the *checkpoint* lane's presence predicate: `header_is_structurally
/// _complete` in `src/checkpoint/apply.rs` folds it into an `HGETALL obj:hdr` and
/// a header failing it reads as ABSENT — restorable, and restored over. The
/// *storage* lane has no such list; it has two hand-written guards, one per field,
/// in two different functions (`read_object_header` requires `seq`,
/// `stat_layer_for_object_id` requires `created_by`). The list is only meaningful
/// as a mirror of those guards — that is what b.f4t established and what the
/// `HEADER_FIELD_SEQ` / `HEADER_FIELD_CREATED_BY` consts exist to keep honest at
/// the spelling level.
///
/// A const cannot make the two agree on *membership*, though, and that is the
/// drift this test owns. It runs in both directions because the two lanes can
/// disagree about what "present" means in both:
///
/// * **A listed field with no guard.** The checkpoint lane calls such a header
///   absent and restores over a live object, while every storage read still
///   serves it.
/// * **A guarded field that is not listed** — b.f4t itself. The checkpoint lane
///   calls the header present and declines to restore, while the read path
///   404s. Nothing ever repairs it, because the one thing that could has been
///   told the layer is fine.
///
/// Nothing else in the suite fails on either — the checkpoint-side tests seed
/// complete headers, and the storage-side guards are tested one literal field at
/// a time.
///
/// # Why it is written over the list rather than field by field
///
/// Both read paths are called for EVERY field in the list and the assertion is
/// that at least one answers `NotFound`. Spelling it per-field (`seq` →
/// `read_object_header`, `created_by` → `stat_layer_for_object_id`) would pass
/// vacuously for a third field, because a per-field table has to be extended by
/// hand too — the same edit that introduces the drift. Iterating the production
/// const and demanding *some* guard fires is what makes the unguarded third field
/// a red test on the diff that adds it.
///
/// The `header_fields_are_structurally_complete` assertions ride along because
/// the predicate and the guards are the two halves of the same claim: the
/// predicate must reject exactly the headers the guards reject.
///
/// # Why Valkey-backed
///
/// The guards live behind `HGETALL`; there is no seam that lets a unit test hand
/// them a `HashMap`. Planting the partial header in the keyspace and calling the
/// real read paths is the only way to assert on the guard rather than on a
/// re-implementation of it. The full header is captured from a real
/// `create_layer` rather than hand-built, so the non-structural fields are
/// exactly what production writes and cannot drift out from under the fixture.
#[tokio::test]
async fn header_structural_fields_is_the_union_of_the_read_path_guards() {
    use live_edit::error::ServiceError;
    use live_edit::storage::layer::{
        header_fields_are_structurally_complete, HEADER_STRUCTURAL_FIELDS,
    };

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "header-drift-layer",
            "apply-test-user",
            &rich_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let header_key = tv.key_schema().obj_header(&oid.to_string());

    // The authentic full header: whatever `queue_object_state` actually wrote.
    let full: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");
    assert!(
        !full.is_empty(),
        "fixture precondition: a materialized layer must have a header to take fields away from"
    );
    for field in HEADER_STRUCTURAL_FIELDS {
        assert!(
            full.contains_key(field),
            "`{field}` is in HEADER_STRUCTURAL_FIELDS but the materializer does not write it — \
             the list would then declare EVERY production header incomplete, so the checkpoint \
             lane would treat every live object as absent and restore over it. Present header \
             fields: {:?}",
            {
                let mut ks: Vec<&String> = full.keys().collect();
                ks.sort();
                ks
            }
        );
    }
    assert!(
        header_fields_are_structurally_complete(&full),
        "a header straight out of `create_layer` must satisfy the predicate; if it does not, the \
         checkpoint lane reads every healthy object as absent"
    );
    ls.read_object_header(oid)
        .await
        .expect("the complete header must satisfy `read_object_header`'s guard");
    ls.stat_layer_for_object_id(oid)
        .await
        .expect("the complete header must satisfy `stat_layer_for_object_id`'s guard");

    for missing in HEADER_STRUCTURAL_FIELDS {
        let mut partial = full.clone();
        partial.remove(missing);
        assert!(
            !header_fields_are_structurally_complete(&partial),
            "dropping `{missing}` must make the predicate answer false — it is in \
             HEADER_STRUCTURAL_FIELDS, so a header without it is by definition structurally \
             incomplete"
        );

        replant_header_exactly(&tv, &header_key, &partial, missing).await;

        // Both read paths, every time. Asserting the OR — rather than the one
        // path this field happens to belong to — is what makes an unguarded
        // third field fail instead of silently passing on a hand-maintained
        // field→path table.
        let header_verdict = ls.read_object_header(oid).await;
        let stat_verdict = ls.stat_layer_for_object_id(oid).await;
        let guarded = matches!(header_verdict, Err(ServiceError::NotFound(_)))
            || matches!(stat_verdict, Err(ServiceError::NotFound(_)));
        assert!(
            guarded,
            "a header missing `{missing}` must be refused as `NotFound` by at least one storage \
             read path. Neither did, so `{missing}` sits in HEADER_STRUCTURAL_FIELDS with no \
             read-path guard behind it: the checkpoint lane's presence predicate calls such a \
             header ABSENT and restores over it, while `read_object_header` / \
             `stat_layer_for_object_id` keep serving the live object — the two lanes disagree \
             about whether the layer exists. Fix by adding the guard in `src/storage/layer.rs`, \
             not by removing the field from the list. Got read_object_header={header_verdict:?}, \
             stat_layer_for_object_id={stat_verdict:?}"
        );
    }

    // ── The reverse direction: union means NOTHING is missing from the list ──
    //
    // The loop above proves every LISTED field is guarded. On its own that is
    // only half of "is the union of the read-path guards", and the half it
    // leaves open is the one b.f4t was: a read path guarding a field the list
    // does NOT name. Then the checkpoint lane calls the header structurally
    // complete — present, do not restore — while the storage read path answers
    // `NotFound`, and the layer is a 404 that nothing will ever repair.
    //
    // So: for every field production actually writes that is NOT in the list,
    // re-plant full-minus-that-field and require that NEITHER read path answers
    // `NotFound`. They are free to `Ok` or to fail some other way (a parse
    // failure on a dropped numeric field is an `Internal`, not a presence
    // verdict) — `NotFound` specifically is the answer that means "this field
    // decides existence", and any field that decides existence belongs in
    // HEADER_STRUCTURAL_FIELDS.
    let non_structural: Vec<String> = {
        let mut v: Vec<String> = full
            .keys()
            .filter(|k| !HEADER_STRUCTURAL_FIELDS.contains(&k.as_str()))
            .cloned()
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        non_structural.len(),
        full.len() - HEADER_STRUCTURAL_FIELDS.len(),
        "fixture arithmetic: every structural field is in the header (asserted above), so the \
         non-structural remainder is exactly the difference. Got {non_structural:?} out of {:?}",
        {
            let mut ks: Vec<&String> = full.keys().collect();
            ks.sort();
            ks
        }
    );
    assert!(
        !non_structural.is_empty(),
        "fixture precondition: a production header carries fields beyond the structural ones, so \
         this direction must have something to check. With an empty list it passes vacuously and \
         the b.f4t direction is unguarded again"
    );

    for (checked, extra) in non_structural.iter().enumerate() {
        let mut partial = full.clone();
        partial.remove(extra);
        assert!(
            header_fields_are_structurally_complete(&partial),
            "dropping the non-structural `{extra}` must leave the predicate answering true — it \
             is not in HEADER_STRUCTURAL_FIELDS, so by definition it does not decide structural \
             completeness"
        );

        replant_header_exactly(&tv, &header_key, &partial, extra).await;

        let header_verdict = ls.read_object_header(oid).await;
        let stat_verdict = ls.stat_layer_for_object_id(oid).await;
        assert!(
            !matches!(header_verdict, Err(ServiceError::NotFound(_))),
            "`read_object_header` answered `NotFound` for a header missing `{extra}`, which is \
             NOT in HEADER_STRUCTURAL_FIELDS — so this read path guards a field the checkpoint \
             lane's presence predicate ignores. That is b.f4t exactly: the checkpoint lane reads \
             such a header as PRESENT and declines to restore, while every `read_object_header` \
             caller gets a 404 on a layer that is never repaired. Fix by adding `{extra}` to \
             HEADER_STRUCTURAL_FIELDS in `src/storage/layer.rs`, not by weakening the guard. \
             (non-structural field {} of {}: {non_structural:?}). Got {header_verdict:?}",
            checked + 1,
            non_structural.len()
        );
        assert!(
            !matches!(stat_verdict, Err(ServiceError::NotFound(_))),
            "`stat_layer_for_object_id` answered `NotFound` for a header missing `{extra}`, \
             which is NOT in HEADER_STRUCTURAL_FIELDS — same b.f4t split as above, reached \
             through the `StatLayer` surface instead: checkpoint says present, the read path \
             404s, nothing repairs it. Fix by adding `{extra}` to HEADER_STRUCTURAL_FIELDS in \
             `src/storage/layer.rs`. (non-structural field {} of {}: {non_structural:?}). Got \
             {stat_verdict:?}",
            checked + 1,
            non_structural.len()
        );
    }
}

/// Re-plant `header_key` as exactly `partial` and assert it landed byte for
/// byte.
///
/// `DEL` first: `HSET` merges, so without it the field the caller took away
/// would survive from the previous iteration and the loop would test nothing.
/// `dropped` names that field, for the failure message only.
async fn replant_header_exactly(
    tv: &TestValkey,
    header_key: &str,
    partial: &HashMap<String, String>,
    dropped: &str,
) {
    let _: () = tv
        .client()
        .pool
        .del(header_key)
        .await
        .expect("DEL obj:hdr must succeed");
    let pairs: Vec<(String, String)> = partial
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let _: () = tv
        .client()
        .pool
        .hset(header_key, pairs)
        .await
        .expect("HSET of the partial header must succeed");
    let replanted: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(header_key)
        .await
        .expect("HGETALL of the re-planted header must succeed");
    assert_eq!(
        &replanted, partial,
        "fixture precondition: the planted header must be exactly `full` minus `{dropped}` — a \
         stale field surviving here would let a guard fire for the wrong reason"
    );
}

// =============================================================================
// b.yvh — a restore starts the relay byte accounting fresh
// =============================================================================

/// The live object carries the relay byte budget's `relay_gens` column (every
/// commit writes it); a DR restore materializes the header whole, so the
/// restored header carries none, and the first commit after it folds a fresh
/// column at its own — bumped — sequence. A stale column surviving the restore
/// would account the new history against the old stream's bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_restores_no_relay_gens_and_the_next_commit_folds_a_fresh_column() {
    use live_edit::storage::relay_retention::RelayRetention;
    use live_edit::storage::test_support::relay::relay_ompf;

    const LAYER: &str = "relay-gens-restore-layer";
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = ls
        .create_layer(
            &session_id,
            LAYER,
            "apply-test-user",
            &rich_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    async fn column(tv: &TestValkey, oid: u64) -> Option<String> {
        tv.client()
            .pool
            .hget(
                tv.key_schema().obj_header(&oid.to_string()),
                live_edit::storage::relay_retention::HEADER_FIELD_RELAY_GENS,
            )
            .await
            .expect("HGET relay_gens must succeed")
    }
    for i in 0..3u64 {
        ls.update_layer_relayed(
            &session_id,
            LAYER,
            oid,
            Some(delta_with_set_field_inline(
                1,
                "name",
                format!("pre-restore-{i}").into_bytes(),
                1,
            )),
            "pre-restore-writer",
            1_000 + i,
        )
        .await
        .unwrap_or_else(|e| panic!("pre-restore commit {i} must succeed: {e:?}"));
    }
    assert!(
        column(&tv, oid).await.is_some(),
        "fixture: the live object's commits wrote relay_gens"
    );

    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;
    apply_checkpoint(&ls, oid, ompf, snap_seq, &meta, None, &test_config(1_000))
        .await
        .expect("apply_checkpoint must succeed on a wiped obj keyspace");
    assert_eq!(
        column(&tv, oid).await,
        None,
        "the restored header carries no relay_gens"
    );

    let restored_seq = read_header_u64(&tv, oid, "seq")
        .await
        .expect("the restored header carries seq");
    let delta = delta_with_set_field_inline(1, "name", b"post-restore".to_vec(), 1);
    let relay_len = relay_ompf(&delta).len() as u64;
    let (seq, _) = ls
        .update_layer_relayed(
            &session_id,
            LAYER,
            oid,
            Some(delta),
            "post-restore-writer",
            2_000,
        )
        .await
        .expect("the first post-restore commit must succeed");
    assert_eq!(seq, restored_seq + 1);
    assert_eq!(
        column(&tv, oid).await,
        RelayRetention::from_budget(live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT)
            .fold(None, seq, relay_len)
            .map(|folded| folded.value),
        "the first post-restore commit folds a fresh column at its own sequence"
    );
}
