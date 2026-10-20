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

//! LocalDisk wrapper for the SR-1.3 [`CheckpointStore`] conformance harness.
//!
//! Subtask `t3.fgj.ya.74.ps` under Task `t2.fgj.ya.74`. Drives every
//! `case_XX_*` body from
//! [`super::conformance`](super::conformance) against
//! [`LocalDiskCheckpointStore`] — the Epic 1 default-on adapter. Each test
//! allocates its own `tempfile::TempDir` so no two cases share an on-disk
//! root; the `TempDir` guard is held to end-of-test scope and Drop's
//! recursive delete cleans up the namespace under
//! `<tmp>/.live-edit/{layers,sessions,ext}` automatically.
//!
//! The injection-seam cases (#19, #20, #22, Group J's #35, #36, and Group M's
//! #42) are
//! `#[cfg]`-gated on the `test-checkpoint-fault-injection` cargo feature —
//! `LocalDiskCheckpointStore` only exposes
//! [`LocalDiskCheckpointStore::with_fault_injector`] under that feature, so the
//! test bodies and the case factory closures are compiled out together when the
//! feature is off.
//!
//! Concurrency cases (#17, #18) opt into `#[tokio::test(flavor = "multi_thread",
//! worker_threads = 4)]` so the spawned writer / reader tasks actually run in
//! parallel rather than cooperatively scheduling on a single thread. Every
//! other case uses the default single-threaded runtime — these case bodies
//! are sequential and the multi-thread runtime would only add scheduler
//! overhead without any signal.
//!
//! The wrapper drives cases `case_01_*` … `case_42_*`; Group J (#28–#37) is the
//! Epic-2 tail-generation block — a tail put through the shared
//! `put_layer_checkpoint`, get-by-handle addressing off
//! `(object_id, snap_seq, kind)`, and a listing that interleaves both kinds
//! newest-first with the pinned Base-before-Tail tie-break. On this adapter a
//! tail is the `<oid>.<to_seq>.tail` stem published meta-first, with no
//! ext-refs artifact of any kind. No cfg gate and no skip gate.

use std::sync::Arc;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{CheckpointError, CheckpointStore};
use tempfile::TempDir;

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::CheckpointFaultInjector;

use super::conformance::{self, CheckpointStoreGuard, CheckpointStorePair};

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::conformance::CheckpointStoreInjectedPair;

// ---------------------------------------------------------------------------
// Per-test guard
// ---------------------------------------------------------------------------

/// `CheckpointStoreGuard` impl that owns the `tempfile::TempDir` for the
/// duration of a single conformance case. Dropping the guard at end-of-scope
/// triggers `TempDir`'s recursive delete of the per-test root, so no state
/// leaks between cases even when the suite runs in parallel.
struct TempDirGuard {
    // `TempDir` itself is `Send + Sync + 'static` and its Drop deletes the
    // directory tree. The field is never read after construction — holding
    // it is the entire contract.
    _dir: TempDir,
}

impl CheckpointStoreGuard for TempDirGuard {}

// ---------------------------------------------------------------------------
// Factory closures
// ---------------------------------------------------------------------------

/// Build a fresh `LocalDiskCheckpointStore` rooted at a freshly-allocated
/// `tempfile::tempdir()`. Each call produces an independent on-disk root —
/// no two factory invocations ever share state.
async fn setup() -> CheckpointStorePair {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    let store: Arc<dyn CheckpointStore> = Arc::new(store);
    let guard: Box<dyn CheckpointStoreGuard> = Box::new(TempDirGuard { _dir: dir });
    (store, guard)
}

/// Injection-seam variant of [`setup`] — accepts a fault injector and threads
/// it through [`LocalDiskCheckpointStore::with_fault_injector`]. Used by
/// cases #19, #20, #22 (error-classification cases).
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn setup_with_injector(
    injector: Arc<dyn CheckpointFaultInjector>,
) -> CheckpointStoreInjectedPair {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir")
        .with_fault_injector(injector);
    let store: Arc<dyn CheckpointStore> = Arc::new(store);
    let guard: Box<dyn CheckpointStoreGuard> = Box::new(TempDirGuard { _dir: dir });
    (store, guard)
}

// ---------------------------------------------------------------------------
// Group A: round-trip puts and gets (#1–#4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_01_put_layer_then_get_latest_round_trips_bytes() {
    conformance::case_01_put_layer_then_get_latest_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_02_put_session_then_get_session_round_trips_bytes() {
    conformance::case_02_put_session_then_get_session_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes() {
    conformance::case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_04_put_counters_sidecar_then_get_round_trips_bytes() {
    conformance::case_04_put_counters_sidecar_then_get_round_trips_bytes(setup).await;
}

// ---------------------------------------------------------------------------
// Group B: absent-key behaviour (#5–#8)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_05_get_latest_layer_checkpoint_unknown_object_returns_none() {
    conformance::case_05_get_latest_layer_checkpoint_unknown_object_returns_none(setup).await;
}

#[tokio::test]
async fn case_06_get_session_checkpoint_unknown_session_returns_none() {
    conformance::case_06_get_session_checkpoint_unknown_session_returns_none(setup).await;
}

#[tokio::test]
async fn case_07_get_ext_blob_unknown_hash_returns_none() {
    conformance::case_07_get_ext_blob_unknown_hash_returns_none(setup).await;
}

#[tokio::test]
async fn case_08_get_counters_sidecar_fresh_adapter_returns_none() {
    conformance::case_08_get_counters_sidecar_fresh_adapter_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group C: idempotent puts (#9–#10)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_09_put_layer_idempotent_single_generation() {
    conformance::case_09_put_layer_idempotent_single_generation(setup).await;
}

#[tokio::test]
async fn case_10_put_ext_blob_idempotent_no_duplicate_storage() {
    conformance::case_10_put_ext_blob_idempotent_no_duplicate_storage(setup).await;
}

// ---------------------------------------------------------------------------
// Group D: list ordering (#11–#12)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_11_list_layer_checkpoints_newest_first_ordering() {
    conformance::case_11_list_layer_checkpoints_newest_first_ordering(setup).await;
}

#[tokio::test]
async fn case_12_list_layer_checkpoints_unknown_object_returns_empty() {
    conformance::case_12_list_layer_checkpoints_unknown_object_returns_empty(setup).await;
}

// ---------------------------------------------------------------------------
// Group E: delete semantics (#13–#16)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_13_delete_layer_then_get_returns_next_newest_or_none() {
    conformance::case_13_delete_layer_then_get_returns_next_newest_or_none(setup).await;
}

#[tokio::test]
async fn case_14_delete_layer_nonexistent_handle_is_idempotent() {
    conformance::case_14_delete_layer_nonexistent_handle_is_idempotent(setup).await;
}

#[tokio::test]
async fn case_15_delete_session_then_get_returns_none() {
    conformance::case_15_delete_session_then_get_returns_none(setup).await;
}

#[tokio::test]
async fn case_16_delete_ext_blob_then_get_returns_none() {
    conformance::case_16_delete_ext_blob_then_get_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group F: concurrency safety (#17–#18)
//
// Multi-thread runtime so the spawned writer / reader / delete tasks make
// real progress in parallel and the race window the case bodies depend on
// actually opens up.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_17_concurrent_put_delete_unrelated_keys_no_interference() {
    conformance::case_17_concurrent_put_delete_unrelated_keys_no_interference(setup).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_18_concurrent_put_get_same_key_no_torn_read() {
    conformance::case_18_concurrent_put_get_same_key_no_torn_read(setup).await;
}

// ---------------------------------------------------------------------------
// Group G: error classification (#19–#22)
//
// #19, #20, #22 use the fault-injection seam and are gated on
// `test-checkpoint-fault-injection`. #21 needs no injector — it asserts the
// natural `NotFound` classification surface and runs unconditionally.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_19_transient_error_classification() {
    conformance::case_19_transient_error_classification(setup_with_injector).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_20_permanent_error_classification() {
    conformance::case_20_permanent_error_classification(setup_with_injector).await;
}

#[tokio::test]
async fn case_21_not_found_classification() {
    conformance::case_21_not_found_classification(setup).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_22_conflict_error_classification() {
    conformance::case_22_conflict_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group H: cross-cutting enumeration + size/retention (#23–#25)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_23_list_referenced_ext_hashes_union_across_layers() {
    conformance::case_23_list_referenced_ext_hashes_union_across_layers(setup).await;
}

#[tokio::test]
async fn case_24_size_bound_sanity_accepts_large_blob() {
    conformance::case_24_size_bound_sanity_accepts_large_blob(setup).await;
}

#[tokio::test]
async fn case_25_retention_bound_sanity_after_prune() {
    conformance::case_25_retention_bound_sanity_after_prune(setup).await;
}

#[tokio::test]
async fn case_26_read_ext_refs_sidecar_per_handle() {
    conformance::case_26_read_ext_refs_sidecar_per_handle(setup).await;
}

// ---------------------------------------------------------------------------
// Group I — raw artifact enumeration (#27)
//
// LocalDisk supplies BOTH planting seams: the `layers/` directory is the
// adapter's whole namespace, so a partial publish (companions with no OMPF)
// and a leaked `<final-name>.<uuid>.tmp` are both a `tokio::fs::write` away.
// This is the adapter where the b.9qd residuals actually occur — `atomic_write`
// removes its own temp on every error path it can observe, but a hard kill
// between the write and the rename leaves one behind.
//
// The namespace layout (`<root>/.live-edit/layers/`) is spelled out here
// rather than read off the adapter: `namespace_root()` is `pub(crate)`, and
// the layout is an SR-4.6 contract, so pinning it literally in the test is the
// stronger assertion anyway.
// ---------------------------------------------------------------------------

/// `<root>/.live-edit/layers` — the flat artifact directory per SR-4.6.
fn layers_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".live-edit").join("layers")
}

/// Artifact-enumeration fixture with both planting seams wired to the
/// per-test tempdir.
async fn setup_artifacts() -> conformance::ArtifactFixture {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let layers = layers_dir(dir.path());
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    let store: Arc<dyn CheckpointStore> = Arc::new(store);

    let orphan_layers = layers.clone();
    let temp_layers = layers;
    conformance::ArtifactFixture {
        store,
        guard: Box::new(TempDirGuard { _dir: dir }),
        plant_orphan: Some(Box::new(move |object_id, snap_seq| {
            let layers = orphan_layers.clone();
            Box::pin(async move {
                tokio::fs::write(
                    layers.join(format!("{object_id}.{snap_seq}.ext-refs.json")),
                    conformance::ext_refs_sidecar_for(&[conformance::CASE_27_ORPHAN_HASH]),
                )
                .await
                .expect("planting the orphan sidecar must succeed");
                tokio::fs::write(
                    layers.join(format!("{object_id}.{snap_seq}.meta.json")),
                    b"{}".as_slice(),
                )
                .await
                .expect("planting the orphan meta must succeed");
            })
        })),
        plant_temp: Some(Box::new(move |object_id, snap_seq| {
            let layers = temp_layers.clone();
            Box::pin(async move {
                // The exact shape `tmp_path_for` produces: the complete final
                // filename, a UUID disambiguator, then `.tmp`.
                tokio::fs::write(
                    layers.join(format!(
                        "{object_id}.{snap_seq}.ompf.0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9.tmp"
                    )),
                    b"half-written".as_slice(),
                )
                .await
                .expect("planting the leaked temp must succeed");
            })
        })),
    }
}

#[tokio::test]
async fn case_27_list_layer_artifacts_enumerates_orphans_and_temps() {
    conformance::case_27_list_layer_artifacts_enumerates_orphans_and_temps(setup_artifacts).await;
}

// ---------------------------------------------------------------------------
// Group J — tail generations (#28–#37)
//
// Tail generations ride the regular `setup` closure — a tail is two more files
// in the same per-test `layers/` directory (`<oid>.<to_seq>.tail` and its
// `.tail.meta.json` companion), so no extra wiring is needed. Deliberately NOT
// skip-gated: this adapter stores tails as of Epic 2 Task 3, so a
// `NotImplemented` out of a tail put would be an adapter defect, and gating
// would turn it into a silent pass.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_28_tail_put_then_get_by_handle_and_list_round_trips() {
    conformance::case_28_tail_put_then_get_by_handle_and_list_round_trips(setup).await;
}

#[tokio::test]
async fn case_29_get_layer_checkpoint_by_handle_contract_both_kinds() {
    conformance::case_29_get_layer_checkpoint_by_handle_contract_both_kinds(setup).await;
}

#[tokio::test]
async fn case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail() {
    conformance::case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail(setup)
        .await;
}

#[tokio::test]
async fn case_31_delete_tail_is_idempotent_and_kind_isolated() {
    conformance::case_31_delete_tail_is_idempotent_and_kind_isolated(setup).await;
}

#[tokio::test]
async fn case_32_tail_put_validation_rejections_are_permanent_and_write_nothing() {
    conformance::case_32_tail_put_validation_rejections_are_permanent_and_write_nothing(setup)
        .await;
}

#[tokio::test]
async fn case_33_tails_are_invisible_to_the_ext_refs_surface() {
    conformance::case_33_tails_are_invisible_to_the_ext_refs_surface(setup).await;
}

#[tokio::test]
async fn case_34_list_all_object_ids_includes_tail_only_objects() {
    conformance::case_34_list_all_object_ids_includes_tail_only_objects(setup).await;
}

#[tokio::test]
async fn case_37_list_layer_artifacts_reports_both_generation_kinds() {
    conformance::case_37_list_layer_artifacts_reports_both_generation_kinds(setup).await;
}

// ---------------------------------------------------------------------------
// Group K — lineage-aware retention (#38)
//
// The Epic-5 retention surface: a whole lineage's delete run, issued
// tails-newest-first then base-last over handles captured from an earlier
// listing, must be idempotent at every position and must not perturb the
// sibling lineage under the same `object_id`. This adapter is where the claim
// has teeth on the artifact side — the base triplet's `ext-refs` / `ompf` /
// `meta` paths and the tail pair's `.tail` / `.tail.meta.json` paths are all
// derived from `(oid, snap_seq)`, so only the kind branch keeps the run off the
// same-sequence sibling base. Rides the regular `setup` closure.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_38_lineage_prune_is_idempotent_and_lineage_isolated() {
    conformance::case_38_lineage_prune_is_idempotent_and_lineage_isolated(setup).await;
}

// ---------------------------------------------------------------------------
// Case #39 — `validate_generation_identity` (b.hka (b)). Rides the regular
// `setup` closure: the gate is the shared helper in `adapters/mod.rs`, called
// as this adapter's first statement in `put_layer_checkpoint` and therefore
// before any path derivation, so a rejected put writes no file of any kind.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_39_identity_empty_puts_are_permanent_and_write_nothing() {
    conformance::case_39_identity_empty_puts_are_permanent_and_write_nothing(setup).await;
}

// Cases #35 / #36 extend Group G's classification surface to the tail methods
// and so ride `setup_with_injector`, cfg-gated on the same feature as #19 /
// #20 / #22.

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_35_get_layer_checkpoint_error_classification() {
    conformance::case_35_get_layer_checkpoint_error_classification(setup_with_injector).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_36_tail_put_error_classification() {
    conformance::case_36_tail_put_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group M — the DR restore Job's read surface (#40–#42)
//
// #40 plants into `<root>/.live-edit/sessions/`, the directory the listing
// reads: a `<sid>.json.<uuid>.tmp` (the exact shape a session write leaves if
// it dies between its write and its rename) and a stray file. #41 rides
// `setup`; #42 rides `setup_with_injector`, the adapter consulting its seam
// under the `list_session_checkpoint_ids` and `ext_blob_exists` labels.
//
// The tests after the registrations pin what only this adapter can meet: a
// missing `sessions/` directory, a directory or a non-UTF-8 name inside it,
// and a filesystem error on the existence probe.
// ---------------------------------------------------------------------------

/// `<root>/.live-edit/sessions` — the session-backup directory per SR-4.6.
fn sessions_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".live-edit").join("sessions")
}

/// `<root>/.live-edit/ext` — the ext-blob directory per SR-4.6.
fn ext_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".live-edit").join("ext")
}

/// Session-listing fixture with the planting seam wired to the per-test
/// tempdir's `sessions/` directory.
async fn setup_session_listing() -> conformance::SessionListingFixture {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let sessions = sessions_dir(dir.path());
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    conformance::SessionListingFixture {
        store: Arc::new(store),
        guard: Box::new(TempDirGuard { _dir: dir }),
        plant_in_sessions: Some(Box::new(move |filename| {
            let path = sessions.join(filename);
            Box::pin(async move {
                tokio::fs::write(&path, b"not a session backup".as_slice())
                    .await
                    .expect("planting into sessions/ must succeed");
            })
        })),
    }
}

#[tokio::test]
async fn case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids() {
    conformance::case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids(
        setup_session_listing,
    )
    .await;
}

#[tokio::test]
async fn case_41_ext_blob_exists_is_true_only_for_stored_bytes() {
    conformance::case_41_ext_blob_exists_is_true_only_for_stored_bytes(setup).await;
}

// Case #43 (Group N) — the live-set walk's own artifact count. Rides the
// regular `setup`: the count comes from the same `read_dir` pass the hashes do.
#[tokio::test]
async fn case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set() {
    conformance::case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set(
        setup,
    )
    .await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_42_dr_restore_read_methods_surface_backend_failures_as_errors() {
    conformance::case_42_dr_restore_read_methods_surface_backend_failures_as_errors(
        setup_with_injector,
    )
    .await;
}

/// A store whose `sessions/` directory does not exist lists no sessions, as
/// `Ok(vec![])`: the directory is gone only if nothing was ever written there
/// or an operator removed it, and either way there is no backup to restore.
#[tokio::test]
async fn local_disk_list_session_checkpoint_ids_on_a_missing_sessions_dir_returns_empty() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    // The constructor creates `sessions/`; remove it (it is empty) to reach
    // the missing-directory state.
    std::fs::remove_dir(sessions_dir(dir.path())).expect("removing the empty sessions/ dir");

    let ids = store
        .list_session_checkpoint_ids()
        .await
        .expect("a missing sessions/ directory is an empty listing, not an error");
    assert!(ids.is_empty(), "got {ids:?}");
}

/// A directory named like a session backup (`<name>.json/`) is not one: no
/// session put creates a directory, and `get_session_checkpoint` could not
/// read it. Listed beside a real backup so the listing is not vacuously empty.
#[tokio::test]
async fn local_disk_list_session_checkpoint_ids_skips_a_directory_named_like_a_backup() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    store
        .put_session_checkpoint(
            "sess-real",
            conformance::sample_session_body(1),
            conformance::sample_meta(0),
        )
        .await
        .expect("put_session_checkpoint must succeed");
    let decoy = sessions_dir(dir.path()).join("sess-dir.json");
    std::fs::create_dir(&decoy).expect("creating the decoy directory");
    assert!(decoy.is_dir(), "precondition: the decoy is a directory");

    let ids = store
        .list_session_checkpoint_ids()
        .await
        .expect("list_session_checkpoint_ids must succeed");
    assert_eq!(
        ids,
        vec!["sess-real".to_owned()],
        "a directory named `sess-dir.json` is not a session backup and must not be listed"
    );
}

/// A file whose name is not UTF-8 is skipped, not listed and not an error: a
/// session put never writes one, and an id the listing cannot represent as a
/// `String` is not one the DR restore Job could restore. Listed beside a real
/// backup so the listing is not vacuously empty.
///
/// Linux-only: the fixture needs a filesystem that accepts arbitrary name
/// bytes.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn local_disk_list_session_checkpoint_ids_skips_a_non_utf8_name() {
    use std::os::unix::ffi::OsStrExt;

    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    store
        .put_session_checkpoint(
            "sess-real",
            conformance::sample_session_body(1),
            conformance::sample_meta(0),
        )
        .await
        .expect("put_session_checkpoint must succeed");
    let name = std::ffi::OsStr::from_bytes(b"sess-\xff.json");
    assert!(
        name.to_str().is_none(),
        "precondition: the name is not UTF-8"
    );
    std::fs::write(sessions_dir(dir.path()).join(name), b"opaque")
        .expect("writing the non-UTF-8 file");

    let ids = store
        .list_session_checkpoint_ids()
        .await
        .expect("a non-UTF-8 name is skipped, not an error");
    assert_eq!(ids, vec!["sess-real".to_owned()]);
}

/// A filesystem error other than "not found" on the existence probe is an
/// error, never `Ok(false)`: the DR restore Job strips a value it is told is
/// absent, and an unreadable path says nothing about whether the bytes exist.
///
/// The fault: `ext/` replaced by a regular file, so the probe of
/// `ext/<hash>.blob` fails with "not a directory" rather than "not found".
#[tokio::test]
async fn local_disk_ext_blob_exists_on_an_io_error_other_than_not_found_is_permanent_never_false() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    let ext = ext_dir(dir.path());
    std::fs::remove_dir(&ext).expect("removing the empty ext/ dir");
    std::fs::write(&ext, b"not a directory").expect("replacing ext/ with a file");

    let result = store
        .ext_blob_exists(&conformance::sample_hash_hex(0x4401))
        .await;
    assert!(
        matches!(result, Err(CheckpointError::Permanent { .. })),
        "an I/O error other than NotFound must surface as Permanent — Ok(false) would read as \
         'bytes absent'; got {result:?}"
    );
}

/// A filesystem error other than "not found" on the session listing is an
/// error, never `Ok(vec![])`: only a missing `sessions/` directory means "no
/// session backups", and the DR restore Job restores nothing it is told is
/// not there.
///
/// The fault: `sessions/` replaced by a regular file, so reading the
/// directory fails with "not a directory" rather than "not found".
#[tokio::test]
async fn local_disk_list_session_checkpoint_ids_on_an_io_error_other_than_not_found_is_permanent_never_empty(
) {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    let sessions = sessions_dir(dir.path());
    std::fs::remove_dir(&sessions).expect("removing the empty sessions/ dir");
    std::fs::write(&sessions, b"not a directory").expect("replacing sessions/ with a file");

    let result = store.list_session_checkpoint_ids().await;
    assert!(
        matches!(result, Err(CheckpointError::Permanent { .. })),
        "an I/O error other than NotFound must surface as Permanent — Ok(vec![]) would read as \
         'no session backups'; got {result:?}"
    );
}
