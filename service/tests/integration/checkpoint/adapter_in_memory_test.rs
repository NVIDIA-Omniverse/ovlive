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

//! SR-1.3 conformance wrapper for the [`InMemoryCheckpointStore`] adapter.
//!
//! Task `t2.fgj.ya.74` Subtask `t3.fgj.ya.74.77`. This file is the
//! per-adapter test-only entry point that drives every enumerated SR-1.3
//! conformance case (`case_01_*` … `case_42_*`, defined in
//! [`super::conformance`]) against a fresh [`InMemoryCheckpointStore`]. The
//! per-case body lives in the harness; each `#[tokio::test]` here is a
//! single line that hands the harness a `setup` closure.
//!
//! ## Why the InMemory adapter is the canonical wrapper
//!
//! The InMemory adapter is the simplest reference implementation — no I/O,
//! no tempdir, no on-disk atomicity story — so a green run of every
//! conformance case against it is the strongest signal that the harness
//! itself is internally consistent. Behaviour drift in the LocalDisk
//! adapter that the InMemory adapter does not also exhibit is the harness
//! failing to pin a trait-level invariant; behaviour drift in the InMemory
//! adapter is the adapter failing the trait. Keeping this file
//! single-line-per-case ensures the gap surfaces in one place.
//!
//! ## Cases #19, #20, #22 (and Group J's #35, #36, Group M's #42) — fault-injection cfg gating
//!
//! These cases drive the `Transient` / `Permanent` / `Conflict`
//! classification surface through the [`CheckpointFaultInjector`] seam
//! exposed by [`InMemoryCheckpointStore::with_fault_injector`] — #35 and #36
//! extending it to `get_layer_checkpoint` and to a **tail** put, #42 to the
//! DR restore Job's two read methods. The seam
//! only compiles under the `test-checkpoint-fault-injection` cargo
//! feature, and the harness `case_19/20/22/35/36/42_*` functions themselves are
//! `#[cfg(feature = "test-checkpoint-fault-injection")]`, so these
//! wrappers are gated on the same feature — running
//! `cargo test --features oodf-ffi/internal-bridge` without the
//! injection feature simply omits them; running with the feature
//! includes them in the `cargo test` selection.
//!
//! ## Group J — tail generations (#28–#37)
//!
//! The Epic-2 kind-aware surface: a tail put through the shared
//! `put_layer_checkpoint`, get-by-handle addressing off
//! `(object_id, snap_seq, kind)`, and a listing that interleaves both kinds
//! newest-first with the pinned Base-before-Tail tie-break at an equal
//! sequence. Plain `#[tokio::test]`, no cfg gate and no skip gate — this
//! adapter stores tails, so a `NotImplemented` is a defect to fail on.

use std::sync::Arc;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::CheckpointStore;

use super::conformance::{CheckpointStorePair, NoopGuard};

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::conformance::CheckpointStoreInjectedPair;
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::CheckpointFaultInjector;

/// Construct a fresh, empty [`InMemoryCheckpointStore`] paired with a
/// [`NoopGuard`] — no per-test cleanup is needed because the adapter is
/// process-local and dropped with the test future.
///
/// This is the trait-surface setup closure handed to every non-injection
/// conformance case. Each invocation MUST yield an independent adapter so
/// the harness never observes cross-case state (the `factory` closure
/// inside the harness calls `setup()` itself at the top of every case).
async fn setup() -> CheckpointStorePair {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let guard: Box<dyn super::conformance::CheckpointStoreGuard> = Box::new(NoopGuard);
    (store, guard)
}

/// Fault-injection setup closure — installs the supplied injector on a
/// fresh [`InMemoryCheckpointStore`] via the `with_fault_injector` builder
/// and returns it pre-upcast as `Arc<dyn CheckpointStore>`. Used by cases
/// #19, #20, #22.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn setup_with_injector(
    injector: Arc<dyn CheckpointFaultInjector>,
) -> CheckpointStoreInjectedPair {
    let store: Arc<dyn CheckpointStore> =
        Arc::new(InMemoryCheckpointStore::new().with_fault_injector(injector));
    let guard: Box<dyn super::conformance::CheckpointStoreGuard> = Box::new(NoopGuard);
    (store, guard)
}

// ---------------------------------------------------------------------------
// Group A — round-trip puts and gets (#1–#4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_01_put_layer_then_get_latest_round_trips_bytes() {
    super::conformance::case_01_put_layer_then_get_latest_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_02_put_session_then_get_session_round_trips_bytes() {
    super::conformance::case_02_put_session_then_get_session_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes() {
    super::conformance::case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_04_put_counters_sidecar_then_get_round_trips_bytes() {
    super::conformance::case_04_put_counters_sidecar_then_get_round_trips_bytes(setup).await;
}

// ---------------------------------------------------------------------------
// Group B — absent-key behaviour (#5–#8)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_05_get_latest_layer_checkpoint_unknown_object_returns_none() {
    super::conformance::case_05_get_latest_layer_checkpoint_unknown_object_returns_none(setup)
        .await;
}

#[tokio::test]
async fn case_06_get_session_checkpoint_unknown_session_returns_none() {
    super::conformance::case_06_get_session_checkpoint_unknown_session_returns_none(setup).await;
}

#[tokio::test]
async fn case_07_get_ext_blob_unknown_hash_returns_none() {
    super::conformance::case_07_get_ext_blob_unknown_hash_returns_none(setup).await;
}

#[tokio::test]
async fn case_08_get_counters_sidecar_fresh_adapter_returns_none() {
    super::conformance::case_08_get_counters_sidecar_fresh_adapter_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group C — idempotent puts (#9–#10)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_09_put_layer_idempotent_single_generation() {
    super::conformance::case_09_put_layer_idempotent_single_generation(setup).await;
}

#[tokio::test]
async fn case_10_put_ext_blob_idempotent_no_duplicate_storage() {
    super::conformance::case_10_put_ext_blob_idempotent_no_duplicate_storage(setup).await;
}

// ---------------------------------------------------------------------------
// Group D — list ordering (#11–#12)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_11_list_layer_checkpoints_newest_first_ordering() {
    super::conformance::case_11_list_layer_checkpoints_newest_first_ordering(setup).await;
}

#[tokio::test]
async fn case_12_list_layer_checkpoints_unknown_object_returns_empty() {
    super::conformance::case_12_list_layer_checkpoints_unknown_object_returns_empty(setup).await;
}

// ---------------------------------------------------------------------------
// Group E — delete semantics (#13–#16)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_13_delete_layer_then_get_returns_next_newest_or_none() {
    super::conformance::case_13_delete_layer_then_get_returns_next_newest_or_none(setup).await;
}

#[tokio::test]
async fn case_14_delete_layer_nonexistent_handle_is_idempotent() {
    super::conformance::case_14_delete_layer_nonexistent_handle_is_idempotent(setup).await;
}

#[tokio::test]
async fn case_15_delete_session_then_get_returns_none() {
    super::conformance::case_15_delete_session_then_get_returns_none(setup).await;
}

#[tokio::test]
async fn case_16_delete_ext_blob_then_get_returns_none() {
    super::conformance::case_16_delete_ext_blob_then_get_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group F — concurrency safety (#17–#18)
//
// Multi-threaded runtime: the cases spawn fan-out writers + readers that
// race through a `tokio::sync::Barrier`. A single-threaded runtime would
// serialize them at the executor and defeat the test.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn case_17_concurrent_put_delete_unrelated_keys_no_interference() {
    super::conformance::case_17_concurrent_put_delete_unrelated_keys_no_interference(setup).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn case_18_concurrent_put_get_same_key_no_torn_read() {
    super::conformance::case_18_concurrent_put_get_same_key_no_torn_read(setup).await;
}

// ---------------------------------------------------------------------------
// Group G — error classification (#19–#22)
//
// Cases #19, #20, #22 drive the fault-injection seam and are cfg-gated on
// `test-checkpoint-fault-injection`. Case #21 needs no injector — it
// asserts the natural `NotFound`-never-surfaced contract across every
// `get_*` / `list_*` / `delete_*` path.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_19_transient_error_classification() {
    super::conformance::case_19_transient_error_classification(setup_with_injector).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_20_permanent_error_classification() {
    super::conformance::case_20_permanent_error_classification(setup_with_injector).await;
}

#[tokio::test]
async fn case_21_not_found_classification() {
    super::conformance::case_21_not_found_classification(setup).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_22_conflict_error_classification() {
    super::conformance::case_22_conflict_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group H — cross-cutting enumeration + size / retention (#23–#25)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_23_list_referenced_ext_hashes_union_across_layers() {
    super::conformance::case_23_list_referenced_ext_hashes_union_across_layers(setup).await;
}

#[tokio::test]
async fn case_24_size_bound_sanity_accepts_large_blob() {
    super::conformance::case_24_size_bound_sanity_accepts_large_blob(setup).await;
}

#[tokio::test]
async fn case_25_retention_bound_sanity_after_prune() {
    super::conformance::case_25_retention_bound_sanity_after_prune(setup).await;
}

#[tokio::test]
async fn case_26_read_ext_refs_sidecar_per_handle() {
    super::conformance::case_26_read_ext_refs_sidecar_per_handle(setup).await;
}

// ---------------------------------------------------------------------------
// Group I — raw artifact enumeration (#27)
//
// Both planting closures are `None`, which is a positive claim rather than a
// skip: `InMemoryCheckpointStore` inserts a `LayerEntry` under one mutex with
// its OMPF (and, when the caller paired one, its sidecar) already in hand, and
// removes it as a unit. There is no intermediate state in which companions
// exist without an OMPF, and there is no temp-then-rename step. The harness
// asserts the matching honest shape — every enumerated stem carries an `Ompf`,
// no `Temp` is ever emitted, and `delete_layer_temp_artifacts` honours the
// default `Ok(0)` — so the InMemory run cannot pass vacuously.
// ---------------------------------------------------------------------------

/// Artifact-enumeration fixture: a fresh adapter and no planting seams.
async fn setup_artifacts() -> super::conformance::ArtifactFixture {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::conformance::ArtifactFixture {
        store,
        guard: Box::new(NoopGuard),
        plant_orphan: None,
        plant_temp: None,
    }
}

#[tokio::test]
async fn case_27_list_layer_artifacts_enumerates_orphans_and_temps() {
    super::conformance::case_27_list_layer_artifacts_enumerates_orphans_and_temps(setup_artifacts)
        .await;
}

// ---------------------------------------------------------------------------
// Group J — tail generations (#28–#37)
//
// Tail generations ride the regular `setup` closure: `InMemoryCheckpointStore`
// holds both kinds in one per-sequence slot, so nothing about a tail put needs
// per-adapter wiring here. Deliberately NOT skip-gated — this adapter stores
// tails as of Epic 2 Task 3, so a `NotImplemented` out of a tail put would be
// an adapter defect, and gating would turn it into a silent pass.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_28_tail_put_then_get_by_handle_and_list_round_trips() {
    super::conformance::case_28_tail_put_then_get_by_handle_and_list_round_trips(setup).await;
}

#[tokio::test]
async fn case_29_get_layer_checkpoint_by_handle_contract_both_kinds() {
    super::conformance::case_29_get_layer_checkpoint_by_handle_contract_both_kinds(setup).await;
}

#[tokio::test]
async fn case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail() {
    super::conformance::case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail(
        setup,
    )
    .await;
}

#[tokio::test]
async fn case_31_delete_tail_is_idempotent_and_kind_isolated() {
    super::conformance::case_31_delete_tail_is_idempotent_and_kind_isolated(setup).await;
}

#[tokio::test]
async fn case_32_tail_put_validation_rejections_are_permanent_and_write_nothing() {
    super::conformance::case_32_tail_put_validation_rejections_are_permanent_and_write_nothing(
        setup,
    )
    .await;
}

#[tokio::test]
async fn case_33_tails_are_invisible_to_the_ext_refs_surface() {
    super::conformance::case_33_tails_are_invisible_to_the_ext_refs_surface(setup).await;
}

#[tokio::test]
async fn case_34_list_all_object_ids_includes_tail_only_objects() {
    super::conformance::case_34_list_all_object_ids_includes_tail_only_objects(setup).await;
}

#[tokio::test]
async fn case_37_list_layer_artifacts_reports_both_generation_kinds() {
    super::conformance::case_37_list_layer_artifacts_reports_both_generation_kinds(setup).await;
}

// ---------------------------------------------------------------------------
// Group K — lineage-aware retention (#38)
//
// The Epic-5 retention surface: a whole lineage's delete run, issued
// tails-newest-first then base-last over handles captured from an earlier
// listing, must be idempotent at every position and must not perturb the
// sibling lineage under the same `object_id`. Rides the regular `setup`
// closure — `InMemoryCheckpointStore` branches its delete on `handle.kind`
// under one mutex, so nothing about the run needs per-adapter wiring here.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_38_lineage_prune_is_idempotent_and_lineage_isolated() {
    super::conformance::case_38_lineage_prune_is_idempotent_and_lineage_isolated(setup).await;
}

// ---------------------------------------------------------------------------
// Case #39 — `validate_generation_identity` (b.hka (b)). Rides the regular
// `setup` closure: the gate is the shared helper in `adapters/mod.rs`, called
// as this adapter's first statement in `put_layer_checkpoint`, so nothing about
// the run needs per-adapter wiring.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_39_identity_empty_puts_are_permanent_and_write_nothing() {
    super::conformance::case_39_identity_empty_puts_are_permanent_and_write_nothing(setup).await;
}

// Cases #35 / #36 extend Group G's classification surface to the tail methods
// and so ride `setup_with_injector`, cfg-gated on the same feature as #19 /
// #20 / #22.

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_35_get_layer_checkpoint_error_classification() {
    super::conformance::case_35_get_layer_checkpoint_error_classification(setup_with_injector)
        .await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_36_tail_put_error_classification() {
    super::conformance::case_36_tail_put_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group M — the DR restore Job's read surface (#40–#42)
//
// #40's planting seam is `None`, a positive claim rather than a skip:
// `InMemoryCheckpointStore` keeps session backups in a map of committed
// entries, with no temp-then-rename step and no file namespace, so a listing
// has nothing to filter out. The case still stores every other artifact kind
// beside the backups. #41 rides `setup`; #42 rides `setup_with_injector`, the
// adapter consulting its seam under the `list_session_checkpoint_ids` and
// `ext_blob_exists` labels.
// ---------------------------------------------------------------------------

/// Session-listing fixture: a fresh adapter and no planting seam.
async fn setup_session_listing() -> super::conformance::SessionListingFixture {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    super::conformance::SessionListingFixture {
        store,
        guard: Box::new(NoopGuard),
        plant_in_sessions: None,
    }
}

#[tokio::test]
async fn case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids() {
    super::conformance::case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids(
        setup_session_listing,
    )
    .await;
}

#[tokio::test]
async fn case_41_ext_blob_exists_is_true_only_for_stored_bytes() {
    super::conformance::case_41_ext_blob_exists_is_true_only_for_stored_bytes(setup).await;
}

// Case #43 (Group N) — the live-set walk's own artifact count. Rides the
// regular `setup`: the count comes from the same locked state the hashes do.
#[tokio::test]
async fn case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set() {
    super::conformance::case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set(
        setup,
    )
    .await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_42_dr_restore_read_methods_surface_backend_failures_as_errors() {
    super::conformance::case_42_dr_restore_read_methods_surface_backend_failures_as_errors(
        setup_with_injector,
    )
    .await;
}
