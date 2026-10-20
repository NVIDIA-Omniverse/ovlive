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

//! b.f1v — `handle_update_layer`'s SINGLE checkpoint-recovery arm, driven from
//! the gRPC boundary.
//!
//! Before b.f1v the handler carried two recovery arms: one matching
//! `ServiceError::NotFound` (the layer mapping vanished) and a second matching
//! `ServiceError::FailedPrecondition` *by message equality* against the storage
//! layer's empty-header sentinel (the mapping resolves but `obj:hdr` is gone —
//! the natural symptom of a partial Valkey keyspace loss). b.f1v flipped the
//! storage surface to `NotFound` and deleted the second arm, so both origins now
//! share one arm keyed on the VARIANT.
//!
//! That collapse is only safe if two things hold at the wire, and neither is
//! observable from a storage-level test:
//!
//!   1. an empty header still reaches recovery, and its outcome is mapped to the
//!      right gRPC status (`NOT_FOUND`, not the pre-fix `FAILED_PRECONDITION`);
//!   2. a *partial* header — present but with no usable `seq` — still does NOT
//!      reach recovery **from the commit path**. It stays
//!      `FAILED_PRECONDITION`, which is what keeps a *write* RPC from
//!      triggering a restore over an object that is corrupt-but-present rather
//!      than gone: the client is mid-commit, and a restore there would replace
//!      whatever state is actually live with the checkpoint's, discarding the
//!      difference on a request that never asked for one.
//!
//! Point 2's *reason* changed under b.f4t and this file records the current
//! one. It used to read "…because the hook's presence probe is a bare `EXISTS`
//! and would report the layer `Restored` while the corrupt header sat
//! untouched", which is no longer true: the probe is now structural
//! (`header_is_structurally_complete` — `seq` AND `created_by`), so a partial
//! header reaching the hook drives a REAL restore that repairs it. That is
//! exactly what `StatLayer` / `SubscribeReadLayer` now do with one (see
//! `checkpoint/e2e_stat_layer_recovery_test.rs`). The commit path's refusal is
//! therefore a deliberate scoping decision, not a workaround for a weak probe —
//! and this file is what pins it.
//!
//! # Why this file is not in `tests/integration/checkpoint/`
//!
//! `recovery_hook_test.rs` is the natural home and owns the arm-by-arm b.33e
//! error-classification matrix, but it is gated on
//! `test-checkpoint-fault-injection` and was being edited in a parallel lane
//! during this change. These tests need no fault injection — a genuinely empty
//! `InMemoryCheckpointStore` is enough to reach `NotRecoverable`, and a seeded
//! one is enough to reach `Restored` — so they live here, ungated, next to the
//! other handler-level UpdateLayer coverage.
//!
//! # Relationship to the checkpoint-lane e2e test
//!
//! `checkpoint/e2e_update_layer_recovery_test.rs::
//! update_layer_recovers_layer_state_after_valkey_wipe` drives the same merged
//! arm through a real LocalDisk checkpoint loop and asserts the restored layer's
//! CONTENT. This file is the cheap complement: it pins the status codes and the
//! `live_edit_checkpoint_recovery_attempts_total` outcome cells for all three
//! shapes (not_recoverable / restored / never-attempted) without a checkpoint
//! loop.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface};
use tonic::Code;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{CheckpointStore, RecoveryHook};
use live_edit::config::UpdateLayerBackend;
use live_edit::grpc::layer::handle_update_layer;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{build_service, create_session_and_layer, AllowAllPermissions};
/// `EMPTY_HEADER_SENTINEL_MSG` / `PARTIAL_HEADER_SENTINEL_MSG` mirror the
/// module-private `storage::layer` constants; the rest were promoted out of
/// this file by b.f4t sub-task 5. Both live in `tests/common` so every module
/// asserting them holds one copy.
use crate::common::{
    recovery_metric, replace_metrics, seed_layer_checkpoint, update_request, wipe_object_keys,
    EMPTY_HEADER_SENTINEL_MSG, PARTIAL_HEADER_SENTINEL_MSG,
};

/// Must match the `service_label` the replacement `Config` carries, or
/// [`recovery_metric`] reads a label cell nothing ever writes and every
/// assertion below passes vacuously at zero. Pinned to
/// `common::RECOVERY_SERVICE_LABEL` by construction — the shared helper reads
/// that label, so a divergence here would silently zero every metric read.
const SERVICE_LABEL: &str = crate::common::RECOVERY_SERVICE_LABEL;

/// The `(timestamp, client_id)` this file's `UpdateLayer` headers carry.
/// Unchanged from the pre-promotion local `update_request`; named so the
/// five-argument shared helper reads as one call rather than five literals.
const CLIENT_TS: u64 = 1000;
const CLIENT_ID: &str = "test-client";

// =============================================================================
// Local helpers.
//
// The six that used to live here — `replace_metrics`, `wipe_object_keys`,
// `recovery_metric`, `update_request`, `seed_layer_checkpoint`,
// `empty_reset_ompf` — were promoted to `tests/common` by b.f4t sub-task 5 and
// are imported above. `replace_recovery` stays local: the sibling copies take
// different arguments and one chains a test-only latch, so unifying them would
// need a builder rather than a shared function.
//
// What remains below is this file's own fixture vocabulary.
// =============================================================================

/// Assert every outcome cell of the recovery counter is zero — i.e. the hook was
/// never consulted at all.
///
/// The full four-value vocabulary is enumerated on purpose: `sum by (result)`
/// equals the attempt count, so checking a subset would let a regression that
/// routed the attempt to a *different* outcome slip through as "no recovery".
fn assert_no_recovery_attempted(metrics: &SharedMetrics, ctx: &str) {
    for result in ["restored", "not_recoverable", "error", "already_present"] {
        assert_eq!(
            recovery_metric(metrics, result),
            0,
            "{ctx}: the recovery hook must NOT have been consulted, but \
             live_edit_checkpoint_recovery_attempts_total{{result={result}}} ticked"
        );
    }
}

/// Swap in an ENABLED `RecoveryHook` backed by `store`, sharing the service's
/// current `layer_store` and `metrics`.
///
/// `build_service` wires a *disabled* hook, which short-circuits to
/// `NotRecoverable` without touching the store — that would make the
/// `Restored` test impossible and the `not_recoverable` test vacuous.
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

/// Assert the wipe / damage actually produced the header shape the test names.
/// Without this a failed setup would let the UpdateLayer commit succeed and the
/// rest of the test would pass having exercised nothing.
async fn assert_header_field_count(tv: &TestValkey, object_id: u64, want_empty: bool, ctx: &str) {
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let header: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");
    if want_empty {
        assert!(
            header.is_empty(),
            "{ctx}: obj:hdr must be ABSENT after the wipe; got {header:?}"
        );
    } else {
        assert!(
            !header.is_empty() && !header.contains_key("seq"),
            "{ctx}: obj:hdr must be PRESENT and missing only `seq`; got {header:?}"
        );
    }
}

/// Assert the layer mapping survived — the discriminator between the merged
/// arm's two origins. If this key were gone the handler would short-circuit at
/// step 4 and never reach the commit, so every test here would be testing the
/// mapping-absent origin under a header-absent name.
async fn assert_mapping_alive(tv: &TestValkey, session_id: &str, layer_name: &str) {
    let mapping_key = tv.key_schema().layer_mapping(session_id, layer_name);
    let alive: u64 = tv
        .client()
        .pool
        .exists(&mapping_key)
        .await
        .expect("EXISTS on the layer mapping must succeed");
    assert_eq!(
        alive, 1,
        "the layer mapping `{mapping_key}` must survive — 'mapping present, header gone' is \
         the condition under test"
    );
}

// =============================================================================
// 1. Empty header → the merged NotFound arm.
// =============================================================================

/// Empty header + nothing to restore → `NOT_FOUND` at the wire, with the
/// recovery attempt recorded as `not_recoverable`.
///
/// Pre-b.f1v this returned `FAILED_PRECONDITION` (the deleted sentinel arm
/// surfaced the original `ServiceError::FailedPrecondition` verbatim on its
/// `NotRecoverable` branch), so the status assertion is the regression.
///
/// `not_recoverable == 1` is what keeps it honest: if a refactor moved the
/// empty-header surface off the recovery branch entirely, the RPC would still
/// answer `NOT_FOUND` — from the storage error's plain mapping — with every
/// metric cell at zero, and the status assertion alone could not tell the two
/// apart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_header_not_recoverable_surfaces_not_found() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let layer_name = "f1v-not-recoverable-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "f1v-not-recoverable-session", layer_name).await;

    // An EMPTY store is the whole fixture: selection finds no candidate and the
    // hook reports `NotRecoverable` without any fault injection.
    replace_recovery(
        &mut svc,
        Arc::new(InMemoryCheckpointStore::new()) as Arc<dyn CheckpointStore>,
    );

    wipe_object_keys(&tv, object_id).await;
    assert_header_field_count(&tv, object_id, true, "not-recoverable fixture").await;
    assert_mapping_alive(&tv, &session_id, layer_name).await;

    let status = handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id, CLIENT_TS, CLIENT_ID),
    )
    .await
    .expect_err("an unrecoverable empty header must fail the RPC");

    assert_eq!(
        status.code(),
        Code::NotFound,
        "b.f1v: an empty `obj:hdr` must surface as NOT_FOUND. FAILED_PRECONDITION is the \
         pre-fix shape and means the storage surface (or the handler arm keyed on it) \
         regressed. Got: {}",
        status.code()
    );
    // `ServiceError::NotFound` maps to `Status::not_found(msg)` verbatim
    // (`src/error.rs`), and post-merge the message is the ONLY wire-observable
    // discriminator between the arm's two origins — "the mapping vanished"
    // (`layer '<name>' not found in session`) and "the mapping resolves but the
    // header is gone" (this one). Without this the test could not tell a
    // genuine empty-header recovery attempt from a mapping-resolve failure that
    // happened to tick the same metric cell.
    assert_eq!(
        status.message(),
        EMPTY_HEADER_SENTINEL_MSG,
        "the NOT_FOUND must be the EMPTY-HEADER origin of the merged arm, not the \
         mapping-deleted one"
    );
    assert_eq!(
        recovery_metric(&metrics, "not_recoverable"),
        1,
        "the merged arm MUST have actually run one recovery attempt, ending `not_recoverable` \
         — a zero here makes the status assertion above vacuous"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
}

/// Empty header + a usable checkpoint → the merged arm restores and the SINGLE
/// re-issue commits, so the RPC returns OK.
///
/// This is the "still recovers through one arm" proof: b.f1v deleted the
/// dedicated empty-header arm, and if the surviving arm failed to re-issue the
/// commit, a post-wipe UpdateLayer would answer NOT_FOUND to a client whose
/// layer was recoverable — silent, permanent write loss on a healthy checkpoint.
///
/// `restored == 1` (not `>= 1`) pins that exactly ONE restore attempt was made
/// for this RPC and that it succeeded — so the OK above is attributable to a
/// real restore rather than to a commit that never needed one.
///
/// It does **not** pin the `recovered_once` at-most-once latch, and should not
/// be read as doing so: the merged arm's retry is straight-line (match →
/// `Restored` → one re-issue → done) and `recovered_once` is set only by the
/// step-4 mapping-resolve loop, so no path through this fixture could yield 2.
/// The latch is pinned via the mapping-deleted origin, by
/// `tests/integration/checkpoint/recovery_hook_test.rs::
/// update_layer_retry_once_after_restore_second_not_found_surfaces_not_found`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_header_restored_retries_once_and_succeeds() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let layer_name = "f1v-restored-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "f1v-restored-session", layer_name).await;

    let store = InMemoryCheckpointStore::new();
    seed_layer_checkpoint(&store, object_id, &session_id, layer_name, 3).await;
    replace_recovery(&mut svc, Arc::new(store) as Arc<dyn CheckpointStore>);

    wipe_object_keys(&tv, object_id).await;
    assert_header_field_count(&tv, object_id, true, "restored fixture").await;
    assert_mapping_alive(&tv, &session_id, layer_name).await;

    let response = handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id, CLIENT_TS, CLIENT_ID),
    )
    .await
    .expect(
        "the merged NotFound arm must restore the layer and re-issue the commit — an error \
             here means a recoverable post-wipe UpdateLayer now fails at the wire",
    )
    .into_inner();

    assert!(
        response.sequence_num > 0,
        "the re-issued commit must acknowledge a real sequence number; got {}",
        response.sequence_num
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        1,
        "exactly one restore attempt must have run, and it must have SUCCEEDED — a zero here \
         means the OK above came from a commit that never needed recovery, making the whole \
         fixture vacuous"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
}

// =============================================================================
// 2. Partial header → NOT the recovery arm, on both backends.
// =============================================================================

/// Shared body. Both commit backends implement the partial-header guard
/// independently (`scripts/update_layer_fast.lua`'s `PARTIAL_HEADER:` reply vs.
/// `update_layer_once_watch`'s `seq_ok` check), so the handler-level contract has to
/// be asserted against each.
async fn partial_header_does_not_invoke_recovery(backend: UpdateLayerBackend, layer_name: &str) {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "f1v-partial-session", layer_name).await;

    // Pin the backend. Mirrors `build_service`'s LayerStore wiring verbatim
    // (same client, same key schema, same 4096 stream maxlen) with only the
    // backend knob changed.
    svc.layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend),
    );
    // AFTER the swap: the hook must share the store the handler uses.
    replace_recovery(
        &mut svc,
        Arc::new(InMemoryCheckpointStore::new()) as Arc<dyn CheckpointStore>,
    );

    // Damage rather than delete: `seq` alone goes, so the header is PRESENT.
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let _: i64 = tv
        .client()
        .pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL seq must succeed");
    assert_header_field_count(&tv, object_id, false, "partial-header fixture").await;

    let status = handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id, CLIENT_TS, CLIENT_ID),
    )
    .await
    .expect_err("a commit against a partial header must fail the RPC");

    assert_eq!(
        status.code(),
        Code::FailedPrecondition,
        "a corrupt-but-PRESENT header must stay FAILED_PRECONDITION on the COMMIT path. \
         NOT_FOUND would route it into the merged recovery arm, and since b.f4t that arm no \
         longer no-ops on a partial header — it drives a real restore that replaces the \
         object's live state with the checkpoint's. Doing that from inside a write RPC would \
         silently discard whatever the surviving keyspace still held, on a request that asked \
         to append to it, not to roll it back. Got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        PARTIAL_HEADER_SENTINEL_MSG,
        "both backends must surface the partial-header sentinel verbatim at the wire"
    );
    assert_no_recovery_attempted(&metrics, "partial header");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_partial_header_does_not_invoke_recovery_lua() {
    partial_header_does_not_invoke_recovery(UpdateLayerBackend::Lua, "f1v-partial-lua-layer").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_partial_header_does_not_invoke_recovery_txn() {
    partial_header_does_not_invoke_recovery(UpdateLayerBackend::Txn, "f1v-partial-txn-layer").await;
}
