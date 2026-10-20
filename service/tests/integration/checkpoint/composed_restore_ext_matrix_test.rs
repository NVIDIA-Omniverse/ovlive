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

//! PRD **AC-3** — the external-value *presence/absence matrix* on a composed
//! restore. Task `t2.dfh.yj.u3` (Task 6, the proof burden), Subtask
//! `t3.dfh.yj.u3.ad`.
//!
//! ## The two ext paths, and why one file exercises both
//!
//! A composed restore reaches external values by **two structurally different
//! routes**, and this file is the only place both run against one restored
//! layer:
//!
//! 1. **Tail-carried (inline) values — SR-3.4, eager.**
//!    `ComposedLineage::ext_values` (`src/checkpoint/compose.rs`) collects the
//!    `(hash, Bytes)` pairs off every replayed record's OMPF Parts 1..N, and
//!    `apply_checkpoint`'s `ExtPrestore` step (`src/checkpoint/apply.rs`) writes
//!    them through `ExternalValueStore::store_values`
//!    (`src/storage/external_values.rs:246`, impl `:676`) **strictly before**
//!    the single `write_object`. Nothing lazy: the bytes must exist in
//!    `le:ext:meta:{hash}` / `le:ext:data:{hash}` the moment the restore
//!    returns.
//! 2. **Base-referenced values — SR-3.2, lazy.** The checkpoint assembler
//!    serializes a base OMPF with **zero** external parts
//!    (`src/checkpoint/assembler.rs::build_reset_ompf_from_state` — the field
//!    keeps its hash and `ext_value_index`, the bytes do not travel), so a hash
//!    that only the base references never enters `ext_values` and is never
//!    pre-stored. It stays on the pre-existing fetch-on-miss path:
//!    `read_batch` → `try_recover_via_checkpoint`
//!    (`src/storage/external_values.rs:505-620`) → `CheckpointStore::get_ext_blob`
//!    → rewrite into Valkey via `store_recovered_values` (`store_values`'
//!    write, sending no ext-mirror copy, unlike the pre-store above), ticking
//!    `live_edit_checkpoint_ext_recovery_total{result=…}`.
//!
//! **That second path is PRE-EXISTING and UNCHANGED by Epic 4** (the
//! `t3.dfh.yj.b7.qv` audit says so explicitly). Everything this file asserts
//! about a base-referenced miss is therefore *today's behaviour preserved*, not
//! new behaviour: an unrecoverable base-referenced value must surface as
//! `adapter_miss` plus the original error **at read time**
//! (`external_values.rs:564-574`), and must never become a restore failure or —
//! worse — a silent zero-length value.
//!
//! ## Scope fence — what belongs to `t3.dfh.yj.b7.74`, not here
//!
//! `e2e_composed_restore_ext_order_test.rs` (Subtask `t3.dfh.yj.b7.74`) owns
//! SR-3.4 **ordering** (a recording `store_values` decorator proving the
//! pre-store observed `le:{oid}:obj:hdr` ABSENT, plus the one-shot
//! `RESTORE_SEAM_AFTER_EXT_PRESTORE` firing) and the single retried-restore
//! `ref_count` characterization. This file re-asserts **neither**: no ordering
//! probe, no `ref_count` delta.
//!
//! What this file owns is the **presence/absence matrix** and the
//! **completeness claim**: after one restore, every hash the composed state
//! references is either reachable, or unreachable in exactly the pre-existing
//! observable way.
//!
//! Case (c) is the case whose overlap with 74 needs stating precisely, because
//! 74's Test 1 also reads a tail-carried value back byte-identically. The
//! differentiator here is the **NEGATIVE** claim: `get_ext_blob` was never
//! called for a tail-carried hash. Byte-identical readback alone cannot tell a
//! working SR-3.4 pre-store apart from an accidental SR-3.2 lazy recovery that
//! happened to find the blob in the adapter — the un-called adapter is what
//! distinguishes them, and it is why case (c) also asserts the adapter does not
//! hold those blobs at all.
//!
//! ## The matrix
//!
//! One fixture lineage carries all four reference classes at once:
//!
//! | Hash | Referenced by | Class |
//! |---|---|---|
//! | `base_present` | base only | (a) base-referenced, blob in the adapter mirror |
//! | `base_missing` | base only | (b) base-referenced, blob in NEITHER mirror nor Valkey |
//! | `shared` | base **and** one tail commit | (c)/(d) tail-carried ⇒ pre-stored |
//! | `tail_only` | **two** tail commits | (c)/(d) tail-carried, N>1 references ⇒ pre-stored once |
//!
//! Each test seeds only the adapter / Valkey state its own case names, so a
//! failure names one variable.
//!
//! ## Wiring: production-shaped, entirely through the shared harness
//!
//! Nothing is field-swapped onto the service after construction, and no hook is
//! built by hand. Both collaborators this file needs arrive from
//! `setup_checkpoint_harness_with_options`:
//!
//! - **The recovery hook is the harness's own.** `helpers::recovery_hook_for_e2e`
//!   calls `RecoveryHook::with_ext_value_store` unconditionally, exactly as
//!   `main.rs` does, over the same `Arc` the service holds. That builder is not
//!   optional wiring: a hook without an ext store makes `apply_checkpoint` fail
//!   closed on any lineage with a non-empty ext set (`UNAVAILABLE … no
//!   ExternalValueStore was wired`), which is why the hand-rolled hook this file
//!   used to carry is gone rather than merely tidied.
//! - **The external-value store is a real `ValkeyExternalValueStore`**, built
//!   inside `CheckpointHarnessOptions::ext_value_store_factory`. The factory is
//!   what resolves the ordering this file could not otherwise satisfy: it runs
//!   once the harness's `TestValkey` (key prefixing + `Drop`-time cleanup), its
//!   `SharedMetrics`, and its `CheckpointStore` all exist, and strictly before
//!   the recovery hook is built — so the store it returns is the ONE object both
//!   the service and the hook hold. This file's factory attaches
//!   `.with_metrics(..)` (without which every `ext_recovery` cell read below
//!   stays flat at zero) and `.with_checkpoint_store(..)` — the SR-3.2
//!   fetch-on-miss handle the harness default deliberately leaves unarmed, and
//!   the very thing cases (a) and (b) are about.
//!
//! The `CheckpointStore` itself is supplied through
//! `CheckpointHarnessOptions::store`, so the recording decorator sits in
//! **every** position the default store would occupy (loop, recovery hook,
//! session hooks, and the `with_checkpoint_store` handle the SR-3.2 fallback
//! consults) — a decorator wired into only some of those would let the counted
//! call and the observed state disagree.
//!
//! ## Synchronization budget
//!
//! No `tokio::time::sleep` is used for coordination anywhere in this file. Every
//! wait is a named constant; the per-test worst case is the sum of the two rows
//! below, i.e. **15 s**, comfortably under nextest's 120 s hard kill.
//!
//! | Wait | Bound | Why that number |
//! |---|---|---|
//! | first base lands | [`LOOP_BASE_DEADLINE_SECS`] (bounded poll on the store listing) | the loop ticks at [`TICK_SECS`]; `wait_for_checkpoint_at_least` additionally clamps to its own load-tolerant floor |
//! | loop task drains after `cancel()` | [`LOOP_DRAIN_BUDGET`] | a cancelled loop returns at its next `select!`; the budget only stops a wedged one from hanging the binary |
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tempfile::TempDir;
use tonic::Request;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    LayerArtifact,
};
use live_edit::domain::types::FieldValue;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::external_values::{
    collect_external_hashes, ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
// Reused rather than re-implemented, deliberately: the surgical ext-namespace
// wipe is a correctness contract (explicit hashes only — a wildcard would also
// catch the `le:ext:{gc}:*` bookkeeping sets and change what these tests mean),
// and one home for it is what keeps the sibling files from drifting apart on
// it. It is `pub(super)` precisely so siblings can consume it; this is a
// read-only dependency and does not modify that file.
use super::e2e_ext_value_recovery_test::wipe_ext_namespace_for_hashes;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to `wait_for_checkpoint_at_least`. The helper clamps it up
/// to its own load-tolerant floor, so this is the *lower* bound on how long a
/// contended Valkey may take before the wait is called a failure.
const LOOP_BASE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of the checkpoint loop's `JoinHandle`.
/// The loop returns at its next `select!` on the cancellation token, so this is
/// never the wait actually observed — it exists so a wedged loop fails the test
/// promptly instead of hanging the binary.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// `client_id` on every UpdateLayer header. Only needs to be stable and
/// non-empty; self-echo suppression is not under test.
const CLIENT_ID: &str = "ext-matrix-client";

const SESSION_BASE_PATH: &str = "/stage";
const CREATED_BY: &str = "ext-matrix-user";

/// The section every pre-base commit writes into.
const ROOT_SECTION_ID: u64 = 42;

/// Section minted by the first tail-range commit, so the replayed range carries
/// structural work and not only field writes.
const TAIL_SECTION_ID: u64 = 1_000;

/// Authoring rank stamped on every `SetField`. Every field in this file is
/// written exactly once, so the value only has to be stable.
const AUTHORING_RANK: u64 = 0;

/// Byte length of every external value. Large enough that the blob is
/// unambiguously an out-of-band OMPF part rather than something that could be
/// mistaken for an inline value, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index every ext-carrying `SetField` points at. Part
/// 0 is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

// Field names. Named constants rather than inline literals because case (b)
// looks its field up by key in the restored snapshot — a typo there would make
// the "external reference survived" assertion silently unreachable.
const BASE_PRESENT_FIELD: &str = "base-present-field";
const BASE_MISSING_FIELD: &str = "base-missing-field";
const BASE_SHARED_FIELD: &str = "base-shared-field";
const TAIL_SHARED_FIELD: &str = "tail-shared-field";
const TAIL_ONLY_FIELD_A: &str = "tail-only-field-a";
const TAIL_ONLY_FIELD_B: &str = "tail-only-field-b";

/// Per-blob discriminator bytes, so the four fixture blobs hash differently and
/// a byte-identity assertion cannot pass by comparing the wrong pair.
const BLOB_TAG_BASE_PRESENT: u8 = 0x11;
const BLOB_TAG_BASE_MISSING: u8 = 0x22;
const BLOB_TAG_SHARED: u8 = 0x33;
const BLOB_TAG_TAIL_ONLY: u8 = 0x44;

/// Substring of `ExternalValueError::MetaNotFound`'s `Display`
/// (`src/error.rs:15`), which `ServiceError::from` folds into
/// `ServiceError::Internal`. Pinned as a constant so the case (b) / case (d)
/// `matches!` guard cannot silently match some other `Internal` the read path
/// might produce.
const META_NOT_FOUND_MESSAGE: &str = "external value meta not found for hash";

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: one section plus one inline field, so the loop's base
/// carries real state independent of any external value.
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "ext-matrix-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// One external-value `SetField` on `section_id`.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn ext_value_delta(
    section_id: u64,
    key_name: &str,
    hash: &[u8; 32],
    blob_len: usize,
) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id,
            key_name: key_name.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// The first tail-range commit: a new section plus an inline field inside it.
/// Structural work in the replayed range carrying **no** external value — so
/// the ext set the pre-store receives is provably sourced from the other tail
/// commits rather than from "whatever the range happened to contain".
fn tail_structural_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id: TAIL_SECTION_ID,
            section_name: "ext-matrix-tail-section".to_string(),
            section_type: 2,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-inline-field".to_string(),
            value_or_ext_hash: vec![0xC0, 0xDE, 0x01],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload, discriminated by `tag`. An RNG would
/// make a failing run unreproducible for no benefit — the bytes only have to be
/// non-trivial, distinct per class, and stable so `ext_content_hash` of them is
/// the fixture's own contract.
fn ext_blob(tag: u8) -> Vec<u8> {
    (0..EXT_BLOB_LEN).map(|i| ((i % 251) as u8) ^ tag).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The recording CheckpointStore
// ─────────────────────────────────────────────────────────────────────────────

/// A `CheckpointStore` newtype that records every `get_ext_blob` call by
/// `hash_hex` and forwards everything else verbatim.
///
/// `get_ext_blob` is the SR-3.2 fetch-on-miss adapter consult and the **only**
/// way a base-referenced blob can reach Valkey after a restore. Counting it is
/// what turns case (c)'s claim from "the bytes are readable" (which a lazy
/// recovery would also satisfy) into "the SR-3.4 pre-store is what made them
/// readable".
///
/// **Every trait method is forwarded**, including the four carrying non-trivial
/// trait defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). The forwarding is
/// delegated to [`DelegatingCheckpointStore`], whose own doc-comment spells out
/// why an omission there is not a missing pass-through but a silent behaviour
/// substitution — but note that skeleton is a held FIELD, not a blanket impl,
/// so each arm below still has to exist.
struct RecordingExtFetchStore {
    inner: DelegatingCheckpointStore<dyn CheckpointStore>,
    /// `hash_hex` of every `get_ext_blob` call, in call order.
    get_ext_blob_calls: Mutex<Vec<String>>,
}

impl RecordingExtFetchStore {
    fn new(inner: Arc<dyn CheckpointStore>) -> Self {
        Self {
            inner: DelegatingCheckpointStore::new(inner),
            get_ext_blob_calls: Mutex::new(Vec::new()),
        }
    }

    /// How many `get_ext_blob` calls have named `hash_hex` so far. Callers take
    /// a before/after difference around the operation under test — the harness
    /// itself is free to consult the adapter for reasons this file does not
    /// model, so an absolute count would be a hostage to unrelated wiring.
    fn get_ext_blob_calls_for(&self, hash_hex: &str) -> usize {
        self.get_ext_blob_calls
            .lock()
            .expect("get_ext_blob call log mutex is never poisoned — no panic runs while held")
            .iter()
            .filter(|recorded| recorded.as_str() == hash_hex)
            .count()
    }
}

#[async_trait::async_trait]
impl CheckpointStore for RecordingExtFetchStore {
    // ── The one instrumented method ──────────────────────────────────────
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        // Scoped so the std mutex is never held across the `.await` below.
        {
            self.get_ext_blob_calls
                .lock()
                .expect("get_ext_blob call log mutex is never poisoned")
                .push(hash_hex.to_owned());
        }
        self.inner.get_ext_blob(hash_hex).await
    }

    // ── Verbatim forwarding, method by method ────────────────────────────
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four below carry non-trivial trait defaults: omitting any of them
    // substitutes `Ok(empty)` / `Ok(vec![])` / `Ok(0)` / the local publish bound
    // for the inner store's real answer, and nothing fails — the reconciliation
    // pass simply goes inert behind the wrapper.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wiring + drive helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Everything one test needs, wired the way `main.rs` wires production:
/// ONE `Arc<dyn ExternalValueStore>` shared by the service and the recovery
/// hook, and ONE `Arc<dyn CheckpointStore>` — the recording decorator — in
/// every position the harness's default store would occupy.
struct MatrixHarness {
    harness: CheckpointHarness,
    /// The recording decorator, also reachable as `harness.store`. Held
    /// concretely so the call log is readable.
    store: Arc<RecordingExtFetchStore>,
    /// The real `ValkeyExternalValueStore`, `with_metrics` + `with_checkpoint_store`,
    /// read straight back off `harness.svc.ext_value_store`. Tests read blobs
    /// through this handle, which is by construction the same object the service
    /// and the recovery hook hold.
    ext: Arc<dyn ExternalValueStore>,
    /// LocalDisk root behind the recording decorator. Never read; held so the
    /// `TempDir` outlives the adapter.
    #[allow(dead_code)]
    tempdir: TempDir,
}

/// Build the harness described by [`MatrixHarness`].
///
/// Construction order is forced and worth reading, because it is what the
/// `ext_value_store_factory` seam exists to resolve. The recording
/// `CheckpointStore` can be built BEFORE the harness and is handed in through
/// `CheckpointHarnessOptions::store`. The `ValkeyExternalValueStore` cannot: it
/// needs the harness's own `TestValkey` (key prefixing + `Drop`-time cleanup),
/// its `SharedMetrics` (so the `ext_recovery` cells this file reads actually
/// tick), and that same recording store (the SR-3.2 fetch-on-miss handle) —
/// none of which exist until the harness is being constructed. The factory
/// closure runs at exactly the point all three do exist and strictly before the
/// recovery hook is built, so the store it returns is the ONE object the
/// service AND the hook hold, with no post-construction field swap and no
/// hand-rolled hook to keep in sync.
async fn setup_matrix_harness() -> MatrixHarness {
    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let backing: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );
    let store = Arc::new(RecordingExtFetchStore::new(backing));

    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.store = Some(Arc::clone(&store) as Arc<dyn CheckpointStore>);
    // Real `ValkeyExternalValueStore` — the production construction chain from
    // `main.rs`, minus only the ext-mirror sender. Leaving the mirror out is
    // deliberate: it would asynchronously push blobs into the adapter, and the
    // whole matrix rests on the adapter holding exactly what each test put
    // there. Every seeding in this file is an explicit `put_ext_blob`.
    //
    // Both builders are load-bearing and neither is on the harness default:
    // `.with_metrics(..)` is what makes every `ext_recovery` cell this file
    // asserts on observable at all (without it the store holds no registry and
    // the whole family stays flat at zero — case (a)'s `recovered` +1 is the
    // behavioural proof it is attached), and `.with_checkpoint_store(..)` arms
    // the SR-3.2 fetch-on-miss fallback that IS this file's subject.
    opts.ext_value_store_factory = Some(Box::new(|ctx: ExtStoreCtx<'_>| {
        Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics))
                .with_checkpoint_store(Arc::clone(ctx.checkpoint_store)),
        )
    }));
    let harness = setup_checkpoint_harness_with_options(opts).await;

    // Read the store back off the SERVICE rather than stashing a clone out of
    // the closure: this is literally the `Arc` the service holds, and the
    // harness handed that same `Arc` to `RecoveryHook::with_ext_value_store`
    // before returning. "One store everywhere" is therefore structural here,
    // not a property this file has to maintain by convention.
    let ext = Arc::clone(&harness.svc.ext_value_store);

    MatrixHarness {
        harness,
        store,
        ext,
        tempdir,
    }
}

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Issue one `UpdateLayer` and return `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the point: they are what
/// `put_tail_generation_from_records` encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including, for the
/// ext-value commits, the blob as OMPF Part 1.
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// `EXISTS` on a fully-qualified key.
async fn key_exists(harness: &CheckpointHarness, key: &str) -> bool {
    harness
        .tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed against the live test Valkey")
}

/// Assert both content-addressed keys for `hash_hex` exist.
async fn assert_ext_keys_present(harness: &CheckpointHarness, hash_hex: &str, context: &str) {
    let meta_key = harness.tv.key_schema().ext_meta(hash_hex);
    let data_key = harness.tv.key_schema().ext_data(hash_hex);
    assert!(
        key_exists(harness, &meta_key).await,
        "{context}: `{meta_key}` MUST exist"
    );
    assert!(
        key_exists(harness, &data_key).await,
        "{context}: `{data_key}` MUST exist"
    );
}

/// Assert both content-addressed keys for `hash_hex` are absent.
async fn assert_ext_keys_absent(harness: &CheckpointHarness, hash_hex: &str, context: &str) {
    let meta_key = harness.tv.key_schema().ext_meta(hash_hex);
    let data_key = harness.tv.key_schema().ext_data(hash_hex);
    assert!(
        !key_exists(harness, &meta_key).await,
        "{context}: `{meta_key}` MUST be absent"
    );
    assert!(
        !key_exists(harness, &data_key).await,
        "{context}: `{data_key}` MUST be absent"
    );
}

/// `read_batch` for a single hash, flattened to "the bytes" or "the error".
///
/// This is the production read surface the SR-3.2 fallback hangs off:
/// `read_batch` is what `compute_diff_at_cut` calls, and `try_recover_via_checkpoint`
/// runs inside its two miss arms (`external_values.rs:918-943`). Driving it
/// per-hash rather than through `SubscribeReadLayer` is deliberate — a
/// whole-layer diff batches every referenced hash into one call and fails on the
/// first unrecoverable one, which would collapse this file's four classes into a
/// single indistinguishable failure. The composed handler-level path is already
/// pinned by `e2e_ext_value_recovery_test.rs`.
async fn read_one(
    ext: &Arc<dyn ExternalValueStore>,
    hash: &[u8; 32],
) -> Result<Vec<u8>, ServiceError> {
    let map = ext.read_batch(std::slice::from_ref(hash)).await?;
    Ok(map
        .get(hash)
        .unwrap_or_else(|| {
            panic!(
                "read_batch returned Ok for {} but no entry for the requested hash — \
                 `read_batch`'s contract is that a missing hash is an Err, never a silent omission",
                sha256_to_hex(hash)
            )
        })
        .clone())
}

/// Assert an unreachable base-referenced value surfaces exactly as it does
/// today: `ServiceError::Internal` carrying `ExternalValueError::MetaNotFound`'s
/// message, never `Ok`, never an empty value.
///
/// **This is pre-existing behaviour preserved, not new behaviour.** The arm is
/// `external_values.rs:564-574` — the adapter returned `Ok(None)`, so
/// `try_recover_via_checkpoint` ticks `adapter_miss` and returns
/// `Err(original_error)` verbatim, which is the `MetaNotFound` the read arm
/// minted at `:923`. Epic 4 does not touch that path
/// (`t3.dfh.yj.b7.qv`'s audit), and a composed restore must not convert it into
/// a restore failure.
fn assert_pre_existing_miss_shape(result: Result<Vec<u8>, ServiceError>, hash_hex: &str) {
    let err = match result {
        Ok(bytes) => panic!(
            "reading unrecoverable base-referenced hash {hash_hex} MUST fail. It returned Ok with \
             {} bytes — a silent value (empty or otherwise) substituted for a missing external \
             value is strictly worse than the error, because nothing downstream can tell it apart \
             from real content",
            bytes.len()
        ),
        Err(err) => err,
    };
    assert!(
        matches!(&err, ServiceError::Internal(message) if message.contains(META_NOT_FOUND_MESSAGE)),
        "the miss MUST surface as the ORIGINAL `ExternalValueError::MetaNotFound` folded into \
         `ServiceError::Internal` (src/error.rs:15,:22) — that is the unchanged pre-existing wire \
         contract `try_recover_via_checkpoint`'s `Ok(None)` arm preserves \
         (external_values.rs:564-574). Got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture: one lineage carrying all four ext reference classes
// ─────────────────────────────────────────────────────────────────────────────

/// One external value, in the three spellings the assertions need.
struct ExtValueFixture {
    hash: [u8; 32],
    hash_hex: String,
    blob: Vec<u8>,
}

impl ExtValueFixture {
    fn new(tag: u8) -> Self {
        let blob = ext_blob(tag);
        let hash = ext_content_hash(&blob);
        Self {
            hash_hex: sha256_to_hex(&hash),
            hash,
            blob,
        }
    }
}

/// The lineage every test in this file restores.
struct ExtMatrixFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// Referenced by the BASE only; blob is what case (a) seeds into the adapter.
    base_present: ExtValueFixture,
    /// Referenced by the BASE only; blob is deliberately in no adapter at all.
    base_missing: ExtValueFixture,
    /// Referenced by the BASE **and** by one tail commit — tail-carried, so
    /// pre-stored.
    shared: ExtValueFixture,
    /// Referenced by TWO tail commits and nothing else — tail-carried, so
    /// pre-stored once (lineage-wide dedup).
    tail_only: ExtValueFixture,
}

impl ExtMatrixFixture {
    /// Every hash the composed state references, in a stable order.
    fn all(&self) -> [&ExtValueFixture; 4] {
        [
            &self.base_present,
            &self.base_missing,
            &self.shared,
            &self.tail_only,
        ]
    }

    /// The two hashes that ride the SR-3.4 pre-store (they appear inline in a
    /// tail record's OMPF parts).
    fn tail_carried(&self) -> [&ExtValueFixture; 2] {
        [&self.shared, &self.tail_only]
    }
}

/// CreateSession → CreateLayer → three ext-carrying pre-base commits → wait for
/// the loop's base → **fence the loop** → four tail-range commits (one purely
/// structural, three carrying external values) → hand-put the tail from the
/// exact OMPF bytes those commits sent.
///
/// The fence (`cancel()` + a bounded join) is the shape `e2e_composed_restore_test`
/// uses: after it, no tick exists to land a generation, so retention cannot
/// prune the fixture and no fresh base can appear at a sequence the hand-put
/// tail does not link onto.
///
/// **The base is written between the two groups of ext commits on purpose.**
/// That split is what creates the matrix at all: the assembler serializes base
/// OMPFs with zero external parts, so hashes whose only reference is a pre-base
/// commit end up referenced-but-not-carried (the lazy SR-3.2 class), while
/// hashes a post-fence commit writes travel inline inside the tail records (the
/// eager SR-3.4 class). Collapse the split either way and the file tests one
/// path twice.
async fn setup_ext_matrix_fixture(harness: &mut CheckpointHarness, name: &str) -> ExtMatrixFixture {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    let base_present = ExtValueFixture::new(BLOB_TAG_BASE_PRESENT);
    let base_missing = ExtValueFixture::new(BLOB_TAG_BASE_MISSING);
    let shared = ExtValueFixture::new(BLOB_TAG_SHARED);
    let tail_only = ExtValueFixture::new(BLOB_TAG_TAIL_ONLY);

    // ── Pre-base range: three external values that will be BASE-referenced ──
    let pre_base: [(&str, &ExtValueFixture); 3] = [
        (BASE_PRESENT_FIELD, &base_present),
        (BASE_MISSING_FIELD, &base_missing),
        (BASE_SHARED_FIELD, &shared),
    ];
    let mut committed_seq = 0;
    for (index, (field_name, value)) in pre_base.iter().enumerate() {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &ext_value_delta(ROOT_SECTION_ID, field_name, &value.hash, value.blob.len()),
            &[&value.blob],
            1_000 + index as u64,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and — worse here — leave one of the three "base-
    // referenced" hashes out of the base, silently reclassifying it.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        LOOP_BASE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture does \
         not know about, and the hand-put tail's `from_seq = S0` link would name a predecessor \
         covering commits nothing replays"
    );

    // ── The replayed range ───────────────────────────────────────────────
    // Structural first (so the range carries structural work and the ext set is
    // provably sourced from the three commits that follow), then three
    // ext-carrying commits: `shared` (also base-referenced) and `tail_only`
    // twice (N>1 tail references to one hash).
    let mut records: Vec<(u64, Bytes)> = Vec::with_capacity(4);
    let structural = drive_commit(
        harness,
        &session_id,
        &layer_name,
        object_id,
        &tail_structural_delta(),
        &[],
        2_000,
    )
    .await;
    assert_eq!(
        structural.0,
        s0 + 1,
        "fixture precondition: the first tail-range commit must land at S0 + 1; a gap would be \
         rejected downstream as a RecordSeqGap and the test would assert the step-down path"
    );
    records.push(structural);

    let tail_ext: [(&str, &ExtValueFixture); 3] = [
        (TAIL_SHARED_FIELD, &shared),
        (TAIL_ONLY_FIELD_A, &tail_only),
        (TAIL_ONLY_FIELD_B, &tail_only),
    ];
    for (index, (field_name, value)) in tail_ext.iter().enumerate() {
        let record = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &ext_value_delta(TAIL_SECTION_ID, field_name, &value.hash, value.blob.len()),
            &[&value.blob],
            2_001 + index as u64,
        )
        .await;
        assert_eq!(
            record.0,
            s0 + 2 + index as u64,
            "fixture precondition: commit sequences must be contiguous"
        );
        records.push(record);
    }

    // The live UpdateLayer path pre-flighted every value through
    // `store_external_values`, so all four ext key pairs exist right now. Each
    // test wipes the ones its case names — otherwise a post-restore presence
    // assertion would pass on this residue and prove nothing.
    for value in [&base_present, &base_missing, &shared, &tail_only] {
        assert_ext_keys_present(harness, &value.hash_hex, "fixture precondition").await;
    }

    put_tail_generation_from_records(
        &harness.store,
        object_id,
        &session_id,
        &layer_name,
        s0,
        s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    ExtMatrixFixture {
        session_id,
        layer_name,
        object_id,
        base_present,
        base_missing,
        shared,
        tail_only,
    }
}

/// Drive the restore through a production handler and assert it resolved to a
/// COMPOSED lineage.
///
/// `handle_stat_layer` is the precedent surface in this directory. The
/// `gathered_composed` gate is non-vacuity insurance: a base-only restore
/// carries no external values at all (the assembler emits zero ext parts), so
/// on one, every ext assertion downstream would be reading an empty set.
async fn restore_and_assert_composed(harness: &CheckpointHarness, fixture: &ExtMatrixFixture) {
    let restored_pre = layer_recovery_metric(harness, "restored");
    let error_pre = layer_recovery_metric(harness, "error");
    let composed_pre = read_restore_lineage_counter(&harness.metrics, "gathered_composed");

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the wipe MUST restore the composed lineage and return Ok, got {:?}: \
             {}. An UNAVAILABLE carrying `no ExternalValueStore was wired` means the hook swap in \
             `setup_matrix_harness` did not take",
            status.code(),
            status.message()
        )
    });

    assert_eq!(
        layer_recovery_metric(harness, "restored") - restored_pre,
        1,
        "exactly ONE `restored` tick"
    );
    assert_eq!(
        layer_recovery_metric(harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — a tick here means the restore failed rather than completing"
    );
    assert_eq!(
        read_restore_lineage_counter(&harness.metrics, "gathered_composed") - composed_pre,
        1,
        "the lineage MUST resolve COMPOSED. A base-only restore carries no external values at \
         all, so every ext assertion below would be vacuous on one"
    );
}

/// The set of external-value hashes the RESTORED keyspace references, read back
/// through the test-support state reader and the production helpers
/// (`read_object_state_for_test` → `snapshot_to_object_state` →
/// `collect_external_hashes`) rather than
/// re-derived from the fixture. Re-deriving would make the completeness claim a
/// restatement of the fixture; reading it back is what makes it a claim about
/// what the restore actually wrote.
async fn restored_reference_set(harness: &CheckpointHarness, object_id: u64) -> BTreeSet<String> {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("post-restore read_object_state_for_test must succeed");
    collect_external_hashes(&snapshot_to_object_state(&snapshot))
        .iter()
        .map(sha256_to_hex)
        .collect()
}

/// `(hash, size)` of one restored field's external reference, or a panic naming
/// what it carried instead.
async fn restored_external_reference(
    harness: &CheckpointHarness,
    object_id: u64,
    section_id: u64,
    key_name: &str,
) -> ([u8; 32], u64) {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("post-restore read_object_state_for_test must succeed");
    let state = snapshot_to_object_state(&snapshot);
    let field = state
        .fields
        .get(&(section_id, key_name.to_string()))
        .unwrap_or_else(|| {
            panic!(
                "the restored object MUST still carry field ({section_id}, {key_name}); present \
                 field keys: {:?}",
                state.fields.keys().collect::<Vec<_>>()
            )
        });
    match &field.value {
        FieldValue::External { hash, size, .. } => (*hash, *size),
        FieldValue::Inline(bytes) => panic!(
            "field ({section_id}, {key_name}) MUST still be an EXTERNAL reference after the \
             restore. It came back Inline with {} bytes — silently substituting an inline value \
             for an unrecoverable external one is the single worst outcome available here, \
             because it is indistinguishable downstream from real content",
            bytes.len()
        ),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Case (a) — base-referenced hash PRESENT in the adapter mirror
// ═════════════════════════════════════════════════════════════════════════════

/// AC-3(a): a base-referenced external value whose blob lives in the adapter
/// mirror is recoverable byte-identically after a composed restore, through the
/// unchanged SR-3.2 fetch-on-miss path.
///
/// Four assertions, each excluding a different regression:
///
/// - **The restore succeeds and resolves COMPOSED.**
/// - **The value is NOT pre-stored by the restore** — its `le:ext:*` keys are
///   still absent the instant `handle_stat_layer` returns, and the adapter was
///   never consulted during the restore. This is the load-bearing structural
///   claim of the whole case: base-referenced hashes are on the *lazy* path, and
///   without this line the recovery below could be an eager pre-store in
///   disguise.
/// - **The read recovers it byte-identically**, and refills both
///   content-addressed keys so a second read is a plain Valkey hit.
/// - **`ext_recovery{result="recovered"}` moves by exactly 1**, with the other
///   four cells flat. This assertion doubles as the wiring proof for the
///   `.with_metrics(..)` chained in [`setup_matrix_harness`]'s
///   `ext_value_store_factory`: a store built without it holds no registry, so
///   `Metrics::inc_ext_recovery` is never reached and this cell would stay at
///   its zero-seeded value — the delta below is the only observable that can
///   tell an attached registry from a dropped one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn base_referenced_ext_value_present_in_the_mirror_recovers_byte_identically_after_restore() {
    let MatrixHarness {
        mut harness,
        store,
        ext,
        tempdir: _tempdir,
    } = setup_matrix_harness().await;
    let fixture = setup_ext_matrix_fixture(&mut harness, "ext-matrix-present").await;
    let target = &fixture.base_present;

    // Seed the adapter with exactly the hash this case names. The ext-mirror is
    // not wired (see `setup_matrix_harness`), so this `put_ext_blob` is the only
    // way the blob can be in the adapter and the case cannot pass by accident.
    harness
        .store
        .put_ext_blob(&target.hash_hex, Bytes::from(target.blob.clone()))
        .await
        .expect("seeding the adapter mirror must succeed");

    // Wipe the object keys (so recovery fires at all) and — surgically, by
    // explicit hash — the ext namespace for the hash under test.
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&target.hash_hex)).await;
    assert_ext_keys_absent(&harness, &target.hash_hex, "wipe precondition").await;

    let metrics_pre = ExtRecoveryCells::read(&harness.metrics);
    let fetches_pre = store.get_ext_blob_calls_for(&target.hash_hex);

    restore_and_assert_composed(&harness, &fixture).await;

    // ── The lazy-path structural claim ───────────────────────────────────
    assert_ext_keys_absent(
        &harness,
        &target.hash_hex,
        "immediately after the composed restore, a BASE-referenced hash MUST still be absent from \
         Valkey — the assembler emits base OMPFs with zero external parts, so this hash never \
         enters `ComposedLineage::ext_values` and the SR-3.4 pre-store cannot have touched it",
    )
    .await;
    assert_eq!(
        store.get_ext_blob_calls_for(&target.hash_hex) - fetches_pre,
        0,
        "the restore itself MUST NOT consult the adapter for a base-referenced blob. A call here \
         means the restore path grew an eager base-ext fetch, which would change the failure mode \
         of case (b) from a read-time miss into a restore-time one"
    );
    assert_ext_recovery_delta(
        metrics_pre,
        ExtRecoveryCells::read(&harness.metrics),
        0,
        0,
        "across the restore alone",
    );

    // ── The read-time recovery ───────────────────────────────────────────
    let metrics_mid = ExtRecoveryCells::read(&harness.metrics);
    let bytes = read_one(&ext, &target.hash)
        .await
        .expect("reading a base-referenced hash whose blob is in the adapter mirror must succeed");
    assert_eq!(
        bytes.as_slice(),
        target.blob.as_slice(),
        "the recovered value MUST be byte-identical to what the live commit uploaded — recovering \
         bytes that differ loses the data just as thoroughly as not recovering them"
    );
    assert_ext_recovery_delta(
        metrics_mid,
        ExtRecoveryCells::read(&harness.metrics),
        1,
        0,
        "across the read that recovered the value",
    );
    assert_eq!(
        store.get_ext_blob_calls_for(&target.hash_hex) - fetches_pre,
        1,
        "exactly ONE adapter fetch — `try_recover_via_checkpoint` single-flights per hash_hex"
    );

    // ── The refill ───────────────────────────────────────────────────────
    assert_ext_keys_present(
        &harness,
        &target.hash_hex,
        "after the recovery the blob MUST be rewritten into Valkey (external_values.rs:533-542) so \
         the next read is a plain hot-path hit",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Case (b) — base-referenced hash MISSING from the mirror AND from Valkey
// ═════════════════════════════════════════════════════════════════════════════

/// AC-3(b): an unrecoverable base-referenced external value does NOT fail the
/// composed restore. It surfaces at read time exactly as it does today.
///
/// The three things this case pins, in the order they could go wrong:
///
/// - **The restore succeeds.** A composed restore that failed whole because one
///   base-era blob is unreachable would take an entire layer offline over a
///   single lost value — and would do so for a class of loss (a base-referenced
///   blob absent from both stores) that the pre-Epic-4 base-only restore
///   tolerated fine.
/// - **The external reference survives intact.** Hash and size unchanged, no
///   inline substitution, no zero-length value.
/// - **The miss surfaces at read time, unchanged**: `adapter_miss` +1 and the
///   original `MetaNotFound`-derived error.
///
/// This is TODAY'S BEHAVIOUR PRESERVED, not new behaviour —
/// `external_values.rs:564-574` is the arm, and `t3.dfh.yj.b7.qv`'s audit
/// records that Epic 4 does not touch it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unrecoverable_base_referenced_ext_value_leaves_the_restore_intact_and_misses_at_read_time()
{
    let MatrixHarness {
        mut harness,
        store,
        ext,
        tempdir: _tempdir,
    } = setup_matrix_harness().await;
    let fixture = setup_ext_matrix_fixture(&mut harness, "ext-matrix-missing").await;
    let target = &fixture.base_missing;

    // Nothing is seeded into the adapter for this hash — and nothing else can
    // put it there, since the ext-mirror is deliberately unwired. Probe rather
    // than assume: a blob that had somehow landed would turn this case silently
    // into case (a).
    assert!(
        harness
            .store
            .get_ext_blob(&target.hash_hex)
            .await
            .expect("probing the adapter must succeed")
            .is_none(),
        "case precondition: the adapter MUST NOT hold {}",
        target.hash_hex
    );

    wipe_layer_state(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&target.hash_hex)).await;
    assert_ext_keys_absent(&harness, &target.hash_hex, "wipe precondition").await;

    let metrics_pre = ExtRecoveryCells::read(&harness.metrics);
    let fetches_pre = store.get_ext_blob_calls_for(&target.hash_hex);

    // ── The restore still succeeds ───────────────────────────────────────
    restore_and_assert_composed(&harness, &fixture).await;
    assert_ext_recovery_delta(
        metrics_pre,
        ExtRecoveryCells::read(&harness.metrics),
        0,
        0,
        "across the restore alone — the restore never reads external values",
    );

    // ── The external reference survived intact ───────────────────────────
    let (restored_hash, restored_size) = restored_external_reference(
        &harness,
        fixture.object_id,
        ROOT_SECTION_ID,
        BASE_MISSING_FIELD,
    )
    .await;
    assert_eq!(
        restored_hash, target.hash,
        "the restored field MUST still name the same external hash"
    );
    assert_eq!(
        restored_size,
        target.blob.len() as u64,
        "the restored field MUST still carry the original external value size — a zeroed size is \
         how a lost value masquerades as an empty one"
    );

    // ── The read-time miss, unchanged ────────────────────────────────────
    let metrics_mid = ExtRecoveryCells::read(&harness.metrics);
    assert_pre_existing_miss_shape(read_one(&ext, &target.hash).await, &target.hash_hex);
    assert_ext_recovery_delta(
        metrics_mid,
        ExtRecoveryCells::read(&harness.metrics),
        0,
        1,
        "across the read that missed",
    );
    assert_eq!(
        store.get_ext_blob_calls_for(&target.hash_hex) - fetches_pre,
        1,
        "the adapter MUST have been consulted exactly once. Zero would mean the fallback was \
         short-circuited and `adapter_miss` came from somewhere else"
    );
    assert_ext_keys_absent(
        &harness,
        &target.hash_hex,
        "after the miss, nothing MUST have been written — a placeholder / zero-length value here \
         would convert a loud loss into a silent corruption",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Case (c) — values that arrive ONLY inline in tail entries
// ═════════════════════════════════════════════════════════════════════════════

/// AC-3(c): tail-carried external values are reachable after a composed restore
/// **because the SR-3.4 pre-store put them there** — not because a lazy
/// recovery happened to find them.
///
/// The negative claim is the whole point of this case, and is the line that
/// separates it from `t3.dfh.yj.b7.74`'s Test 1 (PM condition C6): that test
/// also reads a tail-carried value back byte-identically, and a byte-identical
/// readback on its own cannot distinguish a working pre-store from an
/// accidental SR-3.2 recovery. Here the adapter is asserted not to hold the
/// blobs at all AND `get_ext_blob` is asserted never to have been called for
/// them, so the pre-store is the only remaining explanation.
///
/// Both tail-carried classes are covered in one restore: `shared` (also
/// base-referenced) and `tail_only` (referenced by two separate tail commits).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tail_carried_ext_values_are_pre_stored_and_never_fetched_from_the_adapter() {
    let MatrixHarness {
        mut harness,
        store,
        ext,
        tempdir: _tempdir,
    } = setup_matrix_harness().await;
    let fixture = setup_ext_matrix_fixture(&mut harness, "ext-matrix-inline").await;

    // The only possible source for these bytes must be the tail records' own
    // OMPF parts: absent from Valkey, absent from the adapter.
    for value in fixture.tail_carried() {
        assert!(
            harness
                .store
                .get_ext_blob(&value.hash_hex)
                .await
                .expect("probing the adapter must succeed")
                .is_none(),
            "case precondition: the adapter MUST NOT hold tail-carried hash {} — if it did, a \
             lazy recovery could satisfy the readback and the negative claim below would be the \
             only thing distinguishing the two paths",
            value.hash_hex
        );
    }

    wipe_layer_state(&harness.tv, fixture.object_id).await;
    let tail_hexes: Vec<String> = fixture
        .tail_carried()
        .iter()
        .map(|value| value.hash_hex.clone())
        .collect();
    wipe_ext_namespace_for_hashes(&harness.tv, &tail_hexes).await;
    for value in fixture.tail_carried() {
        assert_ext_keys_absent(&harness, &value.hash_hex, "wipe precondition").await;
    }

    // Snapshot the fetch counts AFTER the absence probes above, so the probes
    // themselves are outside the measured window.
    let fetches_pre: Vec<usize> = fixture
        .tail_carried()
        .iter()
        .map(|value| store.get_ext_blob_calls_for(&value.hash_hex))
        .collect();
    let metrics_pre = ExtRecoveryCells::read(&harness.metrics);

    restore_and_assert_composed(&harness, &fixture).await;

    // ── Presence + byte-identical readback ───────────────────────────────
    for value in fixture.tail_carried() {
        assert_ext_keys_present(
            &harness,
            &value.hash_hex,
            "after the composed restore, every tail-carried hash MUST be materialized by the \
             SR-3.4 pre-store",
        )
        .await;
        let bytes = read_one(&ext, &value.hash).await.unwrap_or_else(|e| {
            panic!(
                "reading pre-stored hash {} must succeed: {e:?}",
                value.hash_hex
            )
        });
        assert_eq!(
            bytes.as_slice(),
            value.blob.as_slice(),
            "the pre-stored value for {} MUST come back byte-identical to what the live commit \
             uploaded",
            value.hash_hex
        );
    }

    // ── The negative claim ───────────────────────────────────────────────
    for (value, before) in fixture.tail_carried().iter().zip(fetches_pre) {
        assert_eq!(
            store.get_ext_blob_calls_for(&value.hash_hex) - before,
            0,
            "`get_ext_blob` MUST NEVER have been called for tail-carried hash {} — across the \
             restore AND the readback. A call here means the bytes became reachable through the \
             lazy SR-3.2 path rather than the eager SR-3.4 pre-store, which is a different \
             mechanism with a different failure mode (it would go silently unreachable the moment \
             the adapter did not hold the blob — which, in this test, it does not)",
            value.hash_hex
        );
    }
    assert_ext_recovery_delta(
        metrics_pre,
        ExtRecoveryCells::read(&harness.metrics),
        0,
        0,
        "across the restore and the readback — the `ext_recovery` family is scoped to adapter \
         consults, and there were none",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Case (d) — mixed-lineage completeness
// ═════════════════════════════════════════════════════════════════════════════

/// AC-3(d): over ONE restore of a lineage in which all three classes coexist,
/// every hash the composed state references is either reachable, or unreachable
/// in exactly the pre-existing observable way — and nothing referenced went
/// missing from the state itself.
///
/// The completeness claim has two halves, and the second is the one a
/// per-hash test cannot make:
///
/// 1. **The reference set is complete.** Read back out of the restored keyspace
///    (`read_object_state_for_test` → `collect_external_hashes`), it equals the four
///    hashes the fixture wrote — no reference dropped, none invented.
/// 2. **The set partitions exhaustively.** Every member falls into the
///    reachable bucket or the unreachable-exactly-as-before bucket, and the
///    unreachable bucket is precisely `{base_missing}` — the one hash this test
///    deliberately made unrecoverable. A hash that vanished into neither bucket
///    (an `Ok` with wrong bytes, or an error of some other shape) fails here.
///
/// The lineage also carries both dedup shapes at once: `shared` is referenced by
/// the base AND a tail commit, `tail_only` by two tail commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_referenced_ext_hash_is_reachable_or_unreachable_exactly_as_before_after_restore() {
    let MatrixHarness {
        mut harness,
        store,
        ext,
        tempdir: _tempdir,
    } = setup_matrix_harness().await;
    let fixture = setup_ext_matrix_fixture(&mut harness, "ext-matrix-mixed").await;

    // Adapter holds exactly one of the two base-referenced blobs, so the mixed
    // lineage contains a recoverable base reference AND an unrecoverable one.
    harness
        .store
        .put_ext_blob(
            &fixture.base_present.hash_hex,
            Bytes::from(fixture.base_present.blob.clone()),
        )
        .await
        .expect("seeding the adapter mirror must succeed");

    // Wipe ALL four hashes: no assertion below may pass on the live upload's
    // residue.
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    let all_hexes: Vec<String> = fixture
        .all()
        .iter()
        .map(|value| value.hash_hex.clone())
        .collect();
    wipe_ext_namespace_for_hashes(&harness.tv, &all_hexes).await;
    for value in fixture.all() {
        assert_ext_keys_absent(&harness, &value.hash_hex, "wipe precondition").await;
    }

    let fetches_pre: HashMap<String, usize> = fixture
        .all()
        .iter()
        .map(|value| {
            (
                value.hash_hex.clone(),
                store.get_ext_blob_calls_for(&value.hash_hex),
            )
        })
        .collect();
    let metrics_pre = ExtRecoveryCells::read(&harness.metrics);

    restore_and_assert_composed(&harness, &fixture).await;

    // ── Half 1: the reference set is complete ────────────────────────────
    let expected: BTreeSet<String> = all_hexes.iter().cloned().collect();
    assert_eq!(
        restored_reference_set(&harness, fixture.object_id).await,
        expected,
        "the restored object's external-reference set MUST be exactly the four hashes the lineage \
         wrote. A missing member means the restore dropped a reference (the value is then \
         unrecoverable AND invisible, the worst combination); an extra one means it invented a \
         reference no commit made"
    );

    // ── The two dedup shapes are actually present ────────────────────────
    // Without these four reads, "referenced by BOTH the base and a tail commit"
    // and "referenced by N>1 tail commits" would be claims about the fixture's
    // intent rather than about the restored state — and a base that silently
    // dropped `base-shared-field` would leave the completeness assertion above
    // still green, because `shared` also arrives via the tail field.
    for (section_id, field_name, expected) in [
        (ROOT_SECTION_ID, BASE_SHARED_FIELD, &fixture.shared),
        (TAIL_SECTION_ID, TAIL_SHARED_FIELD, &fixture.shared),
        (TAIL_SECTION_ID, TAIL_ONLY_FIELD_A, &fixture.tail_only),
        (TAIL_SECTION_ID, TAIL_ONLY_FIELD_B, &fixture.tail_only),
    ] {
        let (hash, size) =
            restored_external_reference(&harness, fixture.object_id, section_id, field_name).await;
        assert_eq!(
            hash, expected.hash,
            "field ({section_id}, {field_name}) MUST reference {} after the restore — this is what \
             makes the multi-reference shapes real rather than assumed",
            expected.hash_hex
        );
        assert_eq!(
            size,
            expected.blob.len() as u64,
            "field ({section_id}, {field_name}) MUST keep its external value size"
        );
    }

    // ── Half 2: the set partitions exhaustively ──────────────────────────
    // Note on ref counts — deliberately NOT asserted here, that is
    // `t3.dfh.yj.b7.74`'s characterization. Worth recording, though:
    // `ComposedLineage::ext_values` dedups LINEAGE-WIDE, so `tail_only` —
    // referenced by two separate tail commits, which live would have counted
    // twice — restores at `ref_count 1`. That is the ACCEPTED leak-not-loss
    // posture documented on `ExtPrestore` (`src/checkpoint/apply.rs`).
    let mut unreachable: BTreeSet<String> = BTreeSet::new();
    for value in fixture.all() {
        match read_one(&ext, &value.hash).await {
            Ok(bytes) => assert_eq!(
                bytes.as_slice(),
                value.blob.as_slice(),
                "hash {} came back reachable but with the WRONG bytes — a reachable-but-corrupt \
                 value is in neither bucket and is worse than either",
                value.hash_hex
            ),
            Err(err) => {
                assert_pre_existing_miss_shape(Err(err), &value.hash_hex);
                unreachable.insert(value.hash_hex.clone());
            }
        }
    }
    assert_eq!(
        unreachable,
        BTreeSet::from([fixture.base_missing.hash_hex.clone()]),
        "exactly ONE referenced hash may be unreachable — the base-referenced one this test kept \
         out of both stores. A larger set means the restore lost values it should have carried \
         (most likely the SR-3.4 pre-store not running); an empty set means the unrecoverable \
         hash was satisfied from somewhere the fixture does not model, and case (b)'s miss \
         assertion is passing for the wrong reason"
    );

    // ── The per-class mechanism, cross-checked at the adapter ────────────
    let fetch_delta = |value: &ExtValueFixture| -> usize {
        store.get_ext_blob_calls_for(&value.hash_hex) - fetches_pre[&value.hash_hex]
    };
    assert_eq!(
        fetch_delta(&fixture.base_present),
        1,
        "the recoverable base-referenced hash MUST have been served by exactly one adapter fetch"
    );
    assert_eq!(
        fetch_delta(&fixture.base_missing),
        1,
        "the unrecoverable base-referenced hash MUST have been looked for exactly once"
    );
    for value in fixture.tail_carried() {
        assert_eq!(
            fetch_delta(value),
            0,
            "tail-carried hash {} MUST have been reachable without any adapter consult — it rode \
             the SR-3.4 pre-store",
            value.hash_hex
        );
    }
    assert_ext_recovery_delta(
        metrics_pre,
        ExtRecoveryCells::read(&harness.metrics),
        1,
        1,
        "across the whole mixed-lineage restore and readback: one `recovered` (base_present) and \
         one `adapter_miss` (base_missing), and nothing from the two pre-stored hashes",
    );
}
