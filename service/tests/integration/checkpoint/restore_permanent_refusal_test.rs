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

//! b.f4t sub-task 2 — **a wrong-typed key refuses the restore PERMANENTLY.**
//!
//! # The defect this file fences
//!
//! Every failure on the DR restore path used to fold to
//! `CheckpointError::Transient` (bar a `NotFound`), and the recovery hook maps
//! `Transient` to a retriable `UNAVAILABLE`. For a wrong-typed key that is
//! exactly the wrong answer: no retry re-types a key, so the handler's
//! retry-once re-drives a restore that cannot succeed, the keyspace-type-error
//! ERROR line re-emits, `live_edit_keyspace_type_error_total` re-ticks, and the
//! client is told to come back and do it all again — forever. An operator has
//! to delete the key; the honest status is non-retriable `INTERNAL`.
//!
//! # The four gates a wrong-typed key can hit, and why each gets a test
//!
//! The restore path touches keys at four distinct places, in this order, and
//! each was independently capable of misclassifying:
//!
//! | # | Gate | Key | Test(s) |
//! |---|------|-----|---------|
//! | 1 | SR-2.4 identity guard (`GET`) | `le:{sid}:layer:{hex(name)}` | [`wrong_typed_layer_mapping_refuses_permanently`] |
//! | 2 | step-1 structural header (`HGETALL`) / `is_layer_present` | `obj:hdr` | [`wrong_typed_object_header_refuses_the_recovery_hook_permanently`], [`wrong_typed_object_header_surfaces_internal_from_stat_layer_without_consulting_recovery`] |
//! | 3 | residue-wipe enumeration (`SMEMBERS` / `ZRANGE`) | `obj:secs` &c. | [`wrong_typed_index_key_refuses_permanently_and_wipes_nothing`] |
//! | 4 | `write_restored_object`'s pre-`MULTI` keyspace type probe | a per-record key | [`wrong_typed_record_key_refuses_permanently`], [`wrong_typed_record_key_surfaces_internal_at_the_handler`] |
//!
//! Gates 2 and 4 carry two tests each because each has two observation
//! points: the `CheckpointError` the restore path itself produces, and the
//! gRPC status the same fixture surfaces at a handler (which for gate 2 is
//! `INTERNAL` from the read path *without* the hook being consulted at all
//! — see that test's own rustdoc). Both are named in full above; an
//! abbreviated or stale name in this table is how a reader ends up hunting
//! for a test that does not exist under that spelling.
//!
//! Gates 1–3 route through the one shared classifier
//! (`checkpoint::apply::classify_object_key_read_error`); gate 4 is
//! `write_restored_object`'s own refusal, re-classified by
//! `write_object_err_to_checkpoint_err` on the
//! `ServiceError::PermanentKeyspaceRefusal` VARIANT (b.6de sub-task 2 — it used
//! to grep the message for `KEYSPACE_WRONGTYPE_TAG`, so any rewording disarmed
//! it). Four producers, one contract — which is why they are asserted separately
//! rather than assumed to share an implementation.
//!
//! # Reaching gate 4 at all requires a key the wipe cannot clear
//!
//! Since b.f4t the restore wipes the surviving keyspace before writing, so most
//! wrong-typed `obj:*` keys are simply `UNLINK`ed (an `UNLINK` is
//! type-agnostic) and never reach the probe. The one shape that survives is a
//! key **no index names**: with `obj:secs` absent, the wipe's enumeration
//! cannot discover `obj:sec:{id}`, the eleven-key probe short-circuits `Clean`,
//! and the wrong-typed record is still there when `write_restored_object` probes
//! the keys it is about to write. That is the fixture gates 4's tests build, and
//! the reason they seed a checkpoint carrying an actual section rather than the
//! empty reset delta the other suites use.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::KeysInterface;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    apply_checkpoint, CheckpointError, CheckpointMeta, CheckpointStore, RecoveryHook,
    RecoveryOutcome,
};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::handle_stat_layer;
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

use crate::common::{recovery_metric, replace_metrics, wipe_object_keys};
use crate::grpc::helpers::{build_service, create_session_and_layer, AllowAllPermissions};

const SERVICE_LABEL: &str = crate::common::RECOVERY_SERVICE_LABEL;

/// The section the seeded checkpoint restores. Its per-record key
/// `obj:sec:{RESTORED_SECTION_ID}` is what the gate-4 fixtures plant a string
/// at — a key `write_restored_object` probes and no index names.
const RESTORED_SECTION_ID: u64 = 77;

const SNAP_SEQ: u64 = 30;

// =============================================================================
// Fixtures
// =============================================================================

/// A reset-delta OMPF carrying one root-level section, so the restored state
/// makes `write_restored_object` write (and therefore probe) `obj:sec:{id}`.
///
/// `common::empty_reset_ompf` is deliberately not used here: an empty state
/// probes only `obj:hdr`, which would collapse gate 4 into gate 2 and make the
/// per-record fixtures untestable.
fn reset_ompf_with_one_section() -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: RESTORED_SECTION_ID,
            section_name: "restored".to_string(),
            section_type: 1,
        })],
        set_fields: vec![],
        time_samples: vec![],
        source_format: "usda".to_string(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta must succeed");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize must succeed"))
}

fn meta_for(session_id: &str, layer_name: &str) -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms: 1_700_000_000_000,
        snap_seq: Some(SNAP_SEQ),
        session_id: session_id.to_string(),
        layer_name: layer_name.to_string(),
        created_by: "refusal-user".to_string(),
        server_ts: 1_700_000_000_000,
        ..CheckpointMeta::default()
    }
}

/// Seed a checkpoint whose restored state carries [`RESTORED_SECTION_ID`].
async fn seed_checkpoint_with_section(
    store: &InMemoryCheckpointStore,
    object_id: u64,
    session_id: &str,
    layer_name: &str,
) {
    store
        .put_layer_checkpoint(
            object_id,
            SNAP_SEQ,
            reset_ompf_with_one_section(),
            meta_for(session_id, layer_name),
        )
        .await
        .expect("seeding the layer checkpoint must succeed");
}

/// Plant a plain string at `key`. The service retypes no key it owns, so this
/// stands in for whatever foreign writer an operator will have to hunt down.
async fn plant_string_at(tv: &TestValkey, key: &str) {
    let _: () = tv
        .client()
        .pool
        .set(key, "planted-by-a-foreign-writer", None, None, false)
        .await
        .unwrap_or_else(|e| panic!("SET {key} must succeed: {e}"));
}

/// Swap in an ENABLED `RecoveryHook` sharing the service's current
/// `layer_store` and `metrics`. `build_service` wires a *disabled* hook, which
/// short-circuits to `NotRecoverable` without touching the store — that would
/// make every assertion here vacuous.
fn replace_recovery(svc: &mut LiveEditService, store: Arc<dyn CheckpointStore>) {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = true;
    cfg.service_label = SERVICE_LABEL.to_string();
    svc.recovery = Arc::new(RecoveryHook::new(
        store,
        Arc::clone(&svc.layer_store),
        Arc::new(cfg),
        Arc::clone(&svc.metrics),
    ));
}

fn stat_request(session_id: &str, layer_name: &str) -> tonic::Request<StatLayerRequest> {
    let mut request = tonic::Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// `live_edit_keyspace_type_error_total{backend, site}`.
///
/// Panics on an absent series rather than folding to zero: both cells are
/// zero-seeded in `Metrics::new`, so absence means the store never received
/// this metrics handle and every count read from it would be vacuous.
fn keyspace_type_errors(metrics: &SharedMetrics, backend: &str, site: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent from the exposition — the LayerStore never received this metrics handle, so \
             any count read from it would be vacuous"
        )
    }) as u64
}

/// Assert a `CheckpointError` is the permanent, non-retriable kind AND that
/// the shared classifier agrees — the two are separate claims, and it is the
/// second that decides the client's status.
fn assert_permanent_and_non_retriable(err: &CheckpointError, ctx: &str) {
    assert!(
        matches!(err, CheckpointError::Permanent { .. }),
        "{ctx}: a wrong-typed key must classify Permanent. Transient makes the recovery hook \
         re-drive a restore that cannot succeed, on every attempt, forever; got {err:?}"
    );
    assert!(
        !ServiceError::recovery_error_is_retriable(err),
        "{ctx}: `recovery_error_is_retriable` must answer false — that predicate is what maps \
         the verdict to UNAVAILABLE (retriable) vs INTERNAL, so a Permanent it calls retriable \
         reproduces the whole defect one layer down"
    );
}

// =============================================================================
// Gate 1 — the SR-2.4 identity guard's layer-mapping GET.
// =============================================================================

/// A wrong-typed `le:{sid}:layer:{hex(name)}` refuses the restore permanently.
///
/// This key is the odd one out on the restore path: it is under the SESSION
/// hash tag, not the object's, and `write_restored_object` never probes it — so if the
/// identity guard folded its `WRONGTYPE` to `Transient` nothing downstream
/// would ever correct the verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_layer_mapping_refuses_permanently() {
    let tv = TestValkey::setup().await;
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let oid: u64 = 9_501;

    // A HASH where a string is expected. `GET` against it is a WRONGTYPE.
    let mapping_key = tv
        .key_schema()
        .layer_mapping("refusal-session", "refusal-layer");
    let _: () = {
        use fred::interfaces::HashesInterface;
        tv.client()
            .pool
            .hset(&mapping_key, [("not", "a-string")])
            .await
            .expect("HSET to mistype the layer mapping")
    };

    let err = apply_checkpoint(
        &layer_store,
        oid,
        reset_ompf_with_one_section(),
        SNAP_SEQ,
        &meta_for("refusal-session", "refusal-layer"),
        None,
        &crate::common::test_config(),
    )
    .await
    .expect_err("a wrong-typed layer mapping must refuse the restore");

    assert_permanent_and_non_retriable(&err, "layer-mapping GET");
    assert!(
        err.to_string().contains(&mapping_key),
        "the refusal MUST name the offending key — Valkey's WRONGTYPE reply names none, and \
         this key is not even under the object's hash tag, so an operator has nothing else to \
         go on. Got: {err}"
    );
}

// =============================================================================
// Gate 2 — the structural header read, all the way to the wire.
// =============================================================================

/// A wrong-typed `obj:hdr` refuses the recovery hook PERMANENTLY.
///
/// This is the sharpest case of sub-task 2 and the direct fence on b.f4t's
/// change to `RecoveryHook::is_layer_present`: the presence probe is the FIRST
/// thing the hook does, and pre-fix it was a bare `EXISTS` whose fred error
/// folded to `Transient`. `Transient` → `UNAVAILABLE` → "retry me", and no
/// retry re-types a key, so a conforming gRPC client would hammer a dead layer
/// indefinitely while `checkpoint_recovery_attempts_total{result="error"}`
/// ticked on every attempt.
///
/// Driven against the hook directly rather than through a handler, and that is
/// forced rather than stylistic — see the sibling
/// [`wrong_typed_object_header_surfaces_internal_from_stat_layer_without_consulting_recovery`]
/// for why no handler can reach the hook in this state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_object_header_refuses_the_recovery_hook_permanently() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_metrics(Arc::clone(&metrics)),
    );
    let oid: u64 = 9_504;

    let mapping_key = tv
        .key_schema()
        .layer_mapping("refusal-hook-session", "refusal-hook-layer");
    let _: () = tv
        .client()
        .pool
        .set(&mapping_key, oid.to_string(), None, None, false)
        .await
        .expect("SET layer mapping so the SR-2.4 guard passes");

    let store = InMemoryCheckpointStore::new();
    seed_checkpoint_with_section(&store, oid, "refusal-hook-session", "refusal-hook-layer").await;

    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = true;
    cfg.service_label = SERVICE_LABEL.to_string();
    let hook = RecoveryHook::new(
        Arc::new(store) as Arc<dyn CheckpointStore>,
        layer_store,
        Arc::new(cfg),
        Arc::clone(&metrics),
    );

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    plant_string_at(&tv, &header_key).await;

    let outcome = hook.try_restore_layer(oid).await;

    let RecoveryOutcome::Error(err) = outcome else {
        panic!(
            "a wrong-typed obj:hdr must end the attempt as Error; got {outcome:?}. `Restored` \
             would be worse than wrong — it claims a repair that never happened"
        );
    };
    assert_permanent_and_non_retriable(&err, "is_layer_present HGETALL obj:hdr");
    assert!(
        err.to_string().contains(&header_key),
        "the refusal MUST name the offending key so an operator knows which one to remove. \
         Got: {err}"
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "the attempt must be recorded as `error`, exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// At the wire, a wrong-typed `obj:hdr` is `INTERNAL` — and recovery is never
/// consulted at all, because the read path refuses first.
///
/// `stat_layer_for_object_id`'s own `HGETALL obj:hdr` hits the `WRONGTYPE`
/// before it can answer `NotFound`, and `map_fred_err_for_grpc` classifies a
/// non-transient fred error as `ServiceError::Internal`. The handler's recovery
/// branch fires on `NotFound` and nothing else, so it never runs.
///
/// Both halves are asserted because either alone would mislead. The status
/// alone would let a reader conclude the hook produced it (it did not); the
/// zero metric cells alone would look like a regression that disarmed
/// recovery. Together they state the actual contract: on this key the storage
/// read is the refusing layer, it refuses non-retriably, and the hook's own
/// `Permanent` (pinned by the test above) is the belt to that braces — live
/// whenever a *different* `NotFound` origin routes a wrong-typed header into
/// the hook, which is exactly what `handle_update_layer`'s mapping-resolve arm
/// and `subscribe`'s snapshot read can do.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_object_header_surfaces_internal_from_stat_layer_without_consulting_recovery() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let layer_name = "refusal-hdr-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "refusal-hdr-session", layer_name).await;

    let store = InMemoryCheckpointStore::new();
    seed_checkpoint_with_section(&store, object_id, &session_id, layer_name).await;
    replace_recovery(&mut svc, Arc::new(store) as Arc<dyn CheckpointStore>);

    wipe_object_keys(&tv, object_id).await;
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    plant_string_at(&tv, &header_key).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, layer_name))
        .await
        .expect_err("a wrong-typed obj:hdr must fail the RPC");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a wrong-typed obj:hdr MUST surface as INTERNAL. UNAVAILABLE means 'retry me', and no \
         retry re-types a key, so a conforming client would hammer a dead layer forever. \
         Got: {}",
        status.code()
    );
    for result in ["restored", "not_recoverable", "error", "already_present"] {
        assert_eq!(
            recovery_metric(&metrics, result),
            0,
            "the recovery hook must NOT have been consulted — `stat_layer_for_object_id`'s own \
             HGETALL refuses the wrong-typed key before it can answer NotFound, and the \
             handler's branch fires on NotFound alone. A tick on `{result}` means the read path \
             started laundering a WRONGTYPE into a not-found"
        );
    }
}

// =============================================================================
// Gate 3 — the residue wipe's index enumeration.
// =============================================================================

/// A wrong-typed INDEX key refuses at the wipe's enumeration, before anything
/// is deleted and before anything is written.
///
/// Two independent claims, and the second is what makes the first safe: a
/// `SMEMBERS` against a string cannot be made to succeed by retrying, so the
/// verdict must be `Permanent`; and because the enumeration is what BUILDS the
/// `UNLINK` inventory, refusing there must leave the keyspace exactly as it
/// was. A wipe that proceeded on a partial inventory would strand the records
/// it failed to enumerate with no header left for the orphan sweep to
/// rediscover them from — the b.u96 hazard.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_index_key_refuses_permanently_and_wipes_nothing() {
    let tv = TestValkey::setup().await;
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let oid: u64 = 9_502;
    let oid_str = oid.to_string();

    let mapping_key = tv
        .key_schema()
        .layer_mapping("refusal-idx-session", "refusal-idx-layer");
    let _: () = tv
        .client()
        .pool
        .set(&mapping_key, oid.to_string(), None, None, false)
        .await
        .expect("SET layer mapping so the SR-2.4 guard passes");

    // A string where the sections index Set belongs. Other, well-typed residue
    // sits alongside it and is the witness that nothing was wiped.
    let sections_index = tv.key_schema().obj_sections_index(&oid_str);
    plant_string_at(&tv, &sections_index).await;

    let tombstones_index = tv.key_schema().obj_deleted_sections_index(&oid_str);
    let _: () = {
        use fred::interfaces::SetsInterface;
        tv.client()
            .pool
            .sadd(&tombstones_index, "3")
            .await
            .expect("SADD a well-typed tombstone alongside the poisoned index")
    };

    let err = apply_checkpoint(
        &layer_store,
        oid,
        reset_ompf_with_one_section(),
        SNAP_SEQ,
        &meta_for("refusal-idx-session", "refusal-idx-layer"),
        None,
        &crate::common::test_config(),
    )
    .await
    .expect_err("a wrong-typed index key must refuse the restore");

    assert_permanent_and_non_retriable(&err, "obj:secs enumeration");
    assert!(
        err.to_string().contains(&sections_index),
        "the refusal MUST name the offending key so an operator knows which one to remove. \
         Got: {err}"
    );

    let survivors: u64 = tv
        .client()
        .pool
        .exists(vec![tombstones_index.clone(), sections_index.clone()])
        .await
        .expect("EXISTS after the refusal");
    assert_eq!(
        survivors, 2,
        "the refusal must leave the keyspace untouched: the enumeration BUILDS the UNLINK \
         inventory, so a wipe that proceeded past a failed read would delete a partial \
         inventory and strand whatever it could not enumerate"
    );
    let header_exists: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_header(&oid_str))
        .await
        .expect("EXISTS obj:hdr after the refusal");
    assert_eq!(
        header_exists, 0,
        "the refusal must also fire BEFORE write_restored_object — nothing may be materialized \
         over a keyspace the restore could not first establish the shape of"
    );
}

// =============================================================================
// Gate 4 — write_restored_object's own pre-MULTI keyspace type probe.
// =============================================================================

/// A wrong-typed per-record key that no index names survives the wipe and is
/// caught by `write_restored_object`'s probe:
/// `ServiceError::PermanentKeyspaceRefusal` → `CheckpointError::Permanent`. The
/// join is the VARIANT since b.6de sub-task 2 — the classifier no longer greps
/// the message for `KEYSPACE_WRONGTYPE_TAG`, so the verdict survives a
/// rewording of the operator-facing string.
///
/// The index-unreachability is the whole fixture. With `obj:secs` absent the
/// eleven-key residue probe short-circuits `Clean`, so nothing is enumerated and
/// nothing is `UNLINK`ed — which is what lets the poisoned record still be
/// there when `write_restored_object` probes the keys it is about to write. (An
/// index-REACHABLE wrong-typed record would simply be `UNLINK`ed by the wipe:
/// `UNLINK` is type-agnostic. That is a different, benign path.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_record_key_refuses_permanently() {
    let tv = TestValkey::setup().await;
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let oid: u64 = 9_503;

    let mapping_key = tv
        .key_schema()
        .layer_mapping("refusal-rec-session", "refusal-rec-layer");
    let _: () = tv
        .client()
        .pool
        .set(&mapping_key, oid.to_string(), None, None, false)
        .await
        .expect("SET layer mapping so the SR-2.4 guard passes");

    // Index-unreachable on purpose: `obj:secs` stays absent.
    let record_key = tv
        .key_schema()
        .obj_section(&oid.to_string(), &RESTORED_SECTION_ID.to_string());
    plant_string_at(&tv, &record_key).await;

    let err = apply_checkpoint(
        &layer_store,
        oid,
        reset_ompf_with_one_section(),
        SNAP_SEQ,
        &meta_for("refusal-rec-session", "refusal-rec-layer"),
        None,
        &crate::common::test_config(),
    )
    .await
    .expect_err("a wrong-typed per-record key must refuse the restore");

    assert_permanent_and_non_retriable(&err, "write_restored_object keyspace type probe");

    let header_exists: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("EXISTS obj:hdr after the refusal");
    assert_eq!(
        header_exists, 0,
        "the probe is pre-MULTI, so a refusal writes nothing at all"
    );
}

/// The same refusal at the wire, with the metric an operator pages on.
///
/// `keyspace_type_error_total{backend="txn", site="create"} == 1` across the
/// handler's retry-once is the load-bearing count. Pre-fix this refusal was
/// `Transient`, so the handler re-drove the restore and the cell ticked again —
/// and would have kept ticking for every retry of every client, turning one
/// operator-actionable condition into an unbounded alert stream that says
/// nothing about how many keys are actually broken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_record_key_surfaces_internal_at_the_handler() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let layer_name = "refusal-rec-wire-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "refusal-rec-wire-session", layer_name).await;

    // The LayerStore the handler and the hook share must carry the metrics
    // handle, or `keyspace_type_errors` reads an absent series and panics.
    svc.layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_metrics(Arc::clone(&metrics)),
    );

    let store = InMemoryCheckpointStore::new();
    seed_checkpoint_with_section(&store, object_id, &session_id, layer_name).await;
    // AFTER the layer_store swap: the hook must share the store the handler uses.
    replace_recovery(&mut svc, Arc::new(store) as Arc<dyn CheckpointStore>);

    wipe_object_keys(&tv, object_id).await;
    let record_key = tv
        .key_schema()
        .obj_section(&object_id.to_string(), &RESTORED_SECTION_ID.to_string());
    plant_string_at(&tv, &record_key).await;

    let before = keyspace_type_errors(&metrics, "txn", "create");

    let status = handle_stat_layer(&svc, stat_request(&session_id, layer_name))
        .await
        .expect_err("a wrong-typed per-record key must fail the RPC");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a keyspace type refusal MUST surface as INTERNAL — an operator has to delete the key, \
         which no client retry can do. Got: {}",
        status.code()
    );
    assert_eq!(
        keyspace_type_errors(&metrics, "txn", "create") - before,
        1,
        "the keyspace type probe MUST have refused exactly ONCE across the handler's \
         retry-once. Two is the pre-fix shape: `Transient` told the handler to try again, so \
         the same unfixable condition re-emitted its ERROR line and re-ticked this cell — the \
         alert then measures client retries rather than broken keys"
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "exactly one recovery attempt, ending `error`"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
}
