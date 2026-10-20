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

//! Pod-local filesystem [`CheckpointStore`](super::super::CheckpointStore)
//! adapter — the Epic 1 default-on backend.
//!
//! Per SDD SR-4.6 the on-disk layout under a single operator-configured root
//! (`LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH`) is:
//!
//! ```text
//! <root>/.live-edit/
//!     layers/<oid>.<snap_seq>.ompf
//!     layers/<oid>.<snap_seq>.ext-refs.json
//!     layers/<oid>.<snap_seq>.meta.json
//!     layers/<oid>.<to_seq>.tail
//!     layers/<oid>.<to_seq>.tail.meta.json
//!     sessions/<session_id>.json
//!     ext/<hash_hex>.blob
//!     counters.json
//! ```
//!
//! The `meta.json` companion persists `CheckpointMeta` — `creator_replica_id`,
//! `written_at_unix_ms`, `snap_seq`, and the session-level identity fields
//! (`session_id` / `layer_name` / `created_by` / `server_ts`) the apply path
//! needs to rebuild `obj:hdr` in `checkpoint::apply`. An adapter that
//! did not persist the meta would leave apply downstream of
//! `get_latest_layer_checkpoint` without session identity, forcing it to invent
//! `session_id` / `layer_name` and produce a partially-recovered layer. This
//! adapter synthesized exactly such a meta for an absent companion until
//! b.hka (d); it now refuses the generation instead
//! ([`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion)).
//!
//! Every file write is `write(temp) -> rename(temp -> final)` with a fresh
//! `Uuid::new_v4()` suffix on the temp filename. The rename step is what makes
//! the published bytes atomic from the reader's point of view — readers either
//! see the prior file, or the new file, never a partial write. On any error the
//! temp file is best-effort removed; a leaked temp file is a low-severity
//! garbage condition that the operator can scrub manually.
//!
//! All file writes but one go through [`LocalDiskCheckpointStore::atomic_write`].
//! The exception is the external-value blob write
//! ([`CheckpointStore::put_ext_blob`]), which runs its existence check, write
//! and rename in one blocking task that owns the caller's `Bytes`: it writes
//! from those bytes rather than a copy, so the ext-mirror's byte charge riding
//! in them covers the write, and it ends in a rename or a removed temp even
//! when its caller stops waiting. Only a hard kill or crash mid-write leaves an
//! `ext/<hash_hex>.blob.<uuid>.tmp` behind, and nothing deletes those.
//!
//! ## Sidecar handling and publish order
//!
//! `put_layer_checkpoint` publishes the triplet in the order **ext-refs →
//! meta → OMPF** (b.s5y, b.9qd — both adapters share this order). Two
//! invariants ride on it. Ext-refs first is load-bearing for the
//! retention-prune cascade, not an implementation detail: that cascade's
//! live set is built by enumerating `*.ext-refs.json`, so a generation whose
//! OMPF were visible before its sidecar could have its blobs classified
//! prunable while the generation itself is durable. Meta before the OMPF is
//! load-bearing for DR: a durable OMPF with no meta beside it is listed as a
//! generation but refused on read
//! ([`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion)),
//! so the restore walk has to step down past it. See the ordering comment
//! inside `put_layer_checkpoint` for the full argument. The legacy
//! put-then-attach mutator [`LocalDiskCheckpointStore::set_layer_sidecar`]
//! stays available for callers (tests, future writers) that prefer it; both
//! paths land at the same SR-4.6 filesystem layout.
//!
//! ## Tail generations (SR-2.3 / SR-2.4)
//!
//! A tail is a second generation *kind* living in the same `layers` directory
//! under its own stem `<oid>.<to_seq>.tail`, with a kind-qualified meta
//! companion `<oid>.<to_seq>.tail.meta.json` and — by MUST-level design — **no**
//! ext-refs sidecar. The two stems are disjoint by suffix (see the partition
//! argument above [`parse_tail_filename`]), so `base@S` and `tail@to_seq == S`
//! coexist as four distinct filenames with independent metas.
//!
//! The tail pair publishes **meta → artifact**, the mirror of its delete order
//! (**artifact → meta**), so "a listed tail has its meta" holds at every
//! instant. The full argument, including why this is not a divergence from the
//! base path's meta-before-artifact rule, is in `put_layer_checkpoint`'s tail
//! branch.
//!
//! ## Error classification (SR-1.3)
//!
//! `std::io::ErrorKind::NotFound` maps to [`CheckpointError::NotFound`] on
//! `delete_*` paths and to `Ok(None)` on the `get_*` paths; every other I/O
//! error maps to [`CheckpointError::Permanent`]. [`CheckpointError::Transient`]
//! and [`CheckpointError::Conflict`] only originate from the fault-injection
//! seam in test builds — the production filesystem code path never emits
//! them on its own.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use uuid::Uuid;

use super::super::store::{
    parse_layer_artifact_filename, parse_layer_artifact_stem, CheckpointBlob, CheckpointError,
    CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact,
    LayerArtifactKind, ListedSidecarAbsent, LiveExtSet, MissingMetaCompanion, EXT_REFS_SUFFIX,
    META_SUFFIX, OMPF_SUFFIX, SESSION_CHECKPOINT_SUFFIX, TAIL_META_SUFFIX, TAIL_SUFFIX,
    TEMP_SUFFIX,
};

use super::{generation_sort_key, validate_generation_identity, validate_tail_put};

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::CheckpointFaultInjector;
#[cfg(feature = "test-checkpoint-fault-injection")]
use std::sync::Arc;

/// Fixed namespace beneath the operator-configured root per SR-4.6.
const NAMESPACE_DIR: &str = ".live-edit";
const LAYERS_SUBDIR: &str = "layers";
const SESSIONS_SUBDIR: &str = "sessions";
const EXT_SUBDIR: &str = "ext";
const COUNTERS_FILE: &str = "counters.json";

/// Distinguishing phrase of the WARN
/// [`LocalDiskCheckpointStore::rollback_partial_tail_publish`] emits when the
/// durable-envelope probe cannot answer, so the rollback DECLINES the decision.
/// Shared with the tests so they can discriminate this arm from
/// [`ROLLBACK_REMOVE_FAILED_PHRASE`] without duplicating the prose — a reword
/// then breaks compilation instead of silently voiding a negative assertion.
pub(super) const TAIL_ROLLBACK_PROBE_DECLINED_PHRASE: &str =
    "could not determine whether a tail envelope is already durable";

/// Distinguishing phrase of the WARN
/// [`LocalDiskCheckpointStore::best_effort_remove`] emits when the decision was
/// TAKEN and the removal itself failed. Sibling of
/// [`TAIL_ROLLBACK_PROBE_DECLINED_PHRASE`]; shared with the tests for the same
/// reason.
pub(super) const ROLLBACK_REMOVE_FAILED_PHRASE: &str =
    "could not remove a partially-published artifact";

/// Pod-local filesystem checkpoint store.
///
/// Construct via [`LocalDiskCheckpointStore::new`] with the operator-configured
/// root path. The constructor `create_dir_all`s the `.live-edit/{layers,
/// sessions,ext}` subdirectories so subsequent writes never hit a "no such
/// directory" path on a fresh deploy.
pub struct LocalDiskCheckpointStore {
    /// `<root>/.live-edit/` — the namespace root computed once at construction.
    namespace_root: PathBuf,
    layers_dir: PathBuf,
    sessions_dir: PathBuf,
    ext_dir: PathBuf,
    counters_path: PathBuf,
    #[cfg(feature = "test-checkpoint-fault-injection")]
    fault_injector: Option<Arc<dyn CheckpointFaultInjector>>,
}

impl LocalDiskCheckpointStore {
    /// Construct a `LocalDiskCheckpointStore` rooted at `root`.
    ///
    /// Does NOT read environment variables — env-var parsing is the caller's
    /// job (`Config::from_env` validates `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH`
    /// non-empty when the adapter is `LocalDisk` and passes the resolved path
    /// here). Creates the `.live-edit/{layers,sessions,ext}` directory tree
    /// eagerly so subsequent writes do not race on the parent-directory
    /// creation.
    ///
    /// Errors:
    /// - [`CheckpointError::Permanent`] — `create_dir_all` failed (typically a
    ///   permissions or read-only-filesystem condition that retrying will not
    ///   resolve).
    pub fn new(root: PathBuf) -> Result<Self, CheckpointError> {
        let namespace_root = root.join(NAMESPACE_DIR);
        let layers_dir = namespace_root.join(LAYERS_SUBDIR);
        let sessions_dir = namespace_root.join(SESSIONS_SUBDIR);
        let ext_dir = namespace_root.join(EXT_SUBDIR);
        let counters_path = namespace_root.join(COUNTERS_FILE);

        for dir in [&layers_dir, &sessions_dir, &ext_dir] {
            std::fs::create_dir_all(dir).map_err(io_to_permanent)?;
        }

        Ok(Self {
            namespace_root,
            layers_dir,
            sessions_dir,
            ext_dir,
            counters_path,
            #[cfg(feature = "test-checkpoint-fault-injection")]
            fault_injector: None,
        })
    }

    /// Install a fault injector so the SR-1.3 conformance harness can drive
    /// `Transient` / `Conflict` paths through every trait method without
    /// touching production code paths. Returns `self` for builder-style
    /// composition.
    ///
    /// Gated on the `test-checkpoint-fault-injection` cargo feature; release
    /// builds and plain `cargo build` / `cargo test` never see this method.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    pub fn with_fault_injector(mut self, injector: Arc<dyn CheckpointFaultInjector>) -> Self {
        self.fault_injector = Some(injector);
        self
    }

    /// Adapter-specific (non-trait) sidecar setter — mirrors the InMemory
    /// adapter's `set_layer_sidecar` so the assembler Task can call either
    /// adapter through the same surface.
    ///
    /// Writes `<layers>/<oid>.<snap_seq>.ext-refs.json` atomically. Callers
    /// are expected to invoke this only after a successful
    /// `put_layer_checkpoint`; calling it before does not error but leaves
    /// the sidecar dangling without an OMPF if the OMPF write never lands.
    pub async fn set_layer_sidecar(
        &self,
        object_id: u64,
        snap_seq: u64,
        sidecar: Bytes,
    ) -> Result<(), CheckpointError> {
        self.maybe_inject("set_layer_sidecar")?;
        let path = self.sidecar_path(object_id, snap_seq);
        Self::atomic_write(&path, &sidecar).await
    }

    /// Inline fault-injection check. With the feature off this is a
    /// zero-cost shim that always returns `Ok(())`; with the feature on it
    /// consults the installed injector (if any) and lets it short-circuit
    /// the calling trait method with a classified error.
    #[inline]
    fn maybe_inject(&self, _op: &str) -> Result<(), CheckpointError> {
        #[cfg(feature = "test-checkpoint-fault-injection")]
        if let Some(injector) = self.fault_injector.as_ref() {
            return injector.maybe_inject(_op);
        }
        Ok(())
    }

    fn ompf_path(&self, object_id: u64, snap_seq: u64) -> PathBuf {
        self.layers_dir
            .join(format!("{}.{}{}", object_id, snap_seq, OMPF_SUFFIX))
    }

    /// `<layers>/<oid>.<snap_seq>.ext-refs.json` — the ext-refs sidecar of a
    /// **base** generation.
    ///
    /// There is deliberately no `tail_sidecar_path` companion to this helper.
    /// Tails never carry an ext-refs artifact (SR-2.4, b.s5y MUST-level): an
    /// ext-refs sidecar written under a tail stem is never reaped and would pin
    /// its hashes in the retention cascade's live set forever. The *absence* of
    /// the helper is the structural guarantee — there is no path to write such a
    /// file to. [`Self::set_layer_sidecar`], the only sidecar writer, is
    /// base-only by construction for the same reason: it builds the base stem
    /// through this helper and takes a `snap_seq`, with no kind to vary.
    fn sidecar_path(&self, object_id: u64, snap_seq: u64) -> PathBuf {
        self.layers_dir
            .join(format!("{}.{}{}", object_id, snap_seq, EXT_REFS_SUFFIX))
    }

    /// `<layers>/<oid>.<snap_seq>.meta.json` — the JSON-encoded
    /// [`CheckpointMeta`] companion. Persisted by `put_layer_checkpoint`,
    /// read back by `get_latest_layer_checkpoint`, removed by
    /// `delete_layer_checkpoint`.
    fn meta_path(&self, object_id: u64, snap_seq: u64) -> PathBuf {
        self.layers_dir
            .join(format!("{}.{}{}", object_id, snap_seq, META_SUFFIX))
    }

    /// `<layers>/<oid>.<to_seq>.tail` — the tail envelope of a tail generation,
    /// alongside the base family in the same `layers` directory.
    ///
    /// The filename stem `{oid}.{to_seq}.tail` is the shared convention SDD
    /// SR-2.3 pins, and it is a **cross-Epic contract**, not a LocalDisk detail:
    /// Epic 4's lineage gather and Epic 5's retention cascade key on
    /// `(object_id, sequence, kind)` derived from exactly these stems. Do NOT
    /// deviate from it — a differently-shaped stem is invisible to
    /// [`parse_generation_filename`] and therefore invisible to lineage and
    /// retention, which is a silent-data-loss shape, not a cosmetic one.
    fn tail_path(&self, object_id: u64, to_seq: u64) -> PathBuf {
        self.layers_dir
            .join(format!("{}.{}{}", object_id, to_seq, TAIL_SUFFIX))
    }

    /// `<layers>/<oid>.<to_seq>.tail.meta.json` — the JSON-encoded
    /// [`CheckpointMeta`] companion of a tail generation.
    ///
    /// # Why the name is kind-qualified
    ///
    /// The naive `<oid>.<to_seq>.meta.json` **collides** with the base meta at
    /// the same sequence, and that collision is live, not hypothetical: SR-4.1
    /// re-base mints a fresh **base at `S`**, where `S` is the newest consumed
    /// tail's `to_seq` (SR-9.2 pins re-base idempotency at `(object_id, S)`), so
    /// `base@S` and `tail@to_seq == S` genuinely coexist on disk until retention
    /// removes one. Under the naive name one of the two metas would silently
    /// clobber the other's, and the survivor's `kind` field would mislabel the
    /// generation it was left standing for. Qualifying the tail meta with
    /// `.tail` keeps the two in separate files.
    fn tail_meta_path(&self, object_id: u64, to_seq: u64) -> PathBuf {
        self.layers_dir
            .join(format!("{}.{}{}", object_id, to_seq, TAIL_META_SUFFIX))
    }

    fn ext_path(&self, hash_hex: &str) -> PathBuf {
        self.ext_dir.join(format!("{}.blob", hash_hex))
    }

    /// Best-effort `unlink` used by `put_layer_checkpoint`'s rollback paths
    /// (b.s5y R2). Never returns an error and never masks one: the caller is
    /// already on its way to returning the *original* failure, which is the
    /// one worth surfacing.
    ///
    /// A `NotFound` is the ordinary case (the write we are undoing may never
    /// have reached its rename) and is silent. Any other error is warned, not
    /// propagated — the residual is an orphaned artifact, which is exactly the
    /// condition the reconciliation sweep exists to mop up, and failing the put
    /// twice over would tell the operator nothing new.
    async fn best_effort_remove(
        &self,
        path: &Path,
        object_id: u64,
        snap_seq: u64,
        what: &str,
    ) -> bool {
        match tokio::fs::remove_file(path).await {
            Ok(()) => true,
            Err(err) if err.kind() == io::ErrorKind::NotFound => true,
            Err(err) => {
                tracing::warn!(
                    object_id,
                    snap_seq,
                    path = %path.display(),
                    artifact = what,
                    error = %err,
                    "LocalDisk checkpoint put rollback {ROLLBACK_REMOVE_FAILED_PHRASE}; it is left \
                     behind as an orphan until the reconciliation sweep reaps it (harmless to \
                     reads; an orphaned ext-refs sidecar additionally pins this snapshot's ext \
                     hashes in the retention cascade's live set until then, while an orphaned meta \
                     companion — base or tail — is inert)"
                );
                false
            }
        }
    }

    /// Undo the companion artifacts of a `put_layer_checkpoint` that failed
    /// before its OMPF landed — **but only if no OMPF is already on disk for
    /// this `(object_id, snap_seq)`** (b.9qd).
    ///
    /// The guard is the whole point, and it protects the re-put path SDD §22.7
    /// step 8 describes: when the cursor HSET fails, the next tick re-assembles
    /// the *same* layer at the *same* `snap_seq`. If that re-put then fails at
    /// its meta or OMPF write, an unguarded rollback would delete the ext-refs
    /// and meta of a generation whose OMPF is durable from the previous
    /// attempt — leaving a live, enumerable generation stripped of the sidecar
    /// that protects its ext blobs from the retention cascade. That is the
    /// over-delete direction, i.e. exactly the failure class this whole area
    /// exists to prevent, introduced by the cleanup meant to prevent it.
    ///
    /// An errored existence probe resolves to "an OMPF may exist" and skips the
    /// rollback: the residual is an orphan, which the reconciliation pass
    /// reclaims, whereas a wrong deletion is unrecoverable. Fail toward leaving
    /// bytes behind.
    ///
    /// **Base-only.** The durable-OMPF probe is the whole guard, and it is
    /// meaningless for a tail stem — `ompf_path` names a *different* generation
    /// than `tail_path` at the same sequence, so probing it would answer a
    /// question about the base while deciding a tail's cleanup. The tail branch
    /// of `put_layer_checkpoint` therefore calls
    /// [`Self::rollback_partial_tail_publish`], which is this function's
    /// tail-addressed sibling rather than an unguarded removal.
    async fn rollback_partial_publish(&self, object_id: u64, snap_seq: u64, sidecar_written: bool) {
        let ompf_path = self.ompf_path(object_id, snap_seq);
        match tokio::fs::metadata(&ompf_path).await {
            // `is_file()`, not mere existence: something at the OMPF path that
            // is NOT a regular file (a directory, most plausibly an operator's
            // mistake or a hostile pre-creation) is not a durable generation,
            // and treating it as one would suppress the rollback and strand the
            // companions permanently.
            Ok(meta) if meta.is_file() => {
                tracing::debug!(
                    object_id,
                    snap_seq,
                    "LocalDisk checkpoint put rollback skipped: an OMPF is already durable at \
                     this (object_id, snap_seq), so the companion artifacts belong to it"
                );
                return;
            }
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => {
                tracing::warn!(
                    object_id,
                    snap_seq,
                    error = %err,
                    "LocalDisk checkpoint put rollback could not determine whether an OMPF is \
                     already durable; skipping the rollback and leaving the companion artifacts \
                     for the reconciliation sweep (an unnecessary orphan is recoverable; \
                     deleting a live generation's sidecar is not)"
                );
                return;
            }
        }

        // Ext-refs FIRST, mirroring `delete_layer_checkpoint`'s order and for
        // the same reason: of the two companions it is the harmful one (it pins
        // this snapshot's whole ext hash set in the retention cascade's live
        // set), so if only one removal lands it should be that one. An orphaned
        // meta is inert.
        if sidecar_written {
            self.best_effort_remove(
                &self.sidecar_path(object_id, snap_seq),
                object_id,
                snap_seq,
                "ext-refs sidecar (orphaned by a failed publish)",
            )
            .await;
        }
        // The meta is scrubbed unconditionally. Both call sites are downstream
        // of the meta write — one *is* its failure arm — so "was the meta
        // attempted" is a constant `true`, and `best_effort_remove` already
        // treats `NotFound` (attempted, never landed) as the ordinary case.
        self.best_effort_remove(
            &self.meta_path(object_id, snap_seq),
            object_id,
            snap_seq,
            "meta companion (orphaned by a failed publish)",
        )
        .await;
    }

    /// Tail analogue of [`Self::rollback_partial_publish`]: remove the
    /// `.tail.meta.json` a failed tail publish left behind — **but only if no
    /// tail envelope is already durable at this `(object_id, to_seq)`** (b.hka
    /// review round 2).
    ///
    /// Same guard, same fail-toward-leaving-bytes posture, asked at `tail_path`
    /// instead of `ompf_path`. It closes the same over-delete the base guard
    /// closes, in the tail's terms, and it is reachable on a single node: the
    /// SDD §22.7 step 8 re-put path re-writes the same layer at the same
    /// sequence after a failed cursor stamp, so attempt 2's failed envelope
    /// write would otherwise remove the companion of attempt 1's durable
    /// envelope. (Under a shared filesystem it is additionally SR-9.2's
    /// duplicate-leader race.)
    ///
    /// Why that matters more than it did: since the meta placeholder's removal,
    /// an envelope with no companion is not a degraded read but a generation
    /// `checkpoint::lineage` steps PAST — a lost recovery point. See
    /// [`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion).
    async fn rollback_partial_tail_publish(&self, object_id: u64, to_seq: u64) {
        let tail_path = self.tail_path(object_id, to_seq);
        match tokio::fs::metadata(&tail_path).await {
            // `is_file()` for the same reason the base probe uses it: something
            // at the envelope path that is not a regular file is not a durable
            // generation, and treating it as one would strand the companion.
            Ok(meta) if meta.is_file() => {
                tracing::info!(
                    object_id,
                    to_seq,
                    "LocalDisk tail checkpoint put rollback skipped: a tail envelope is already \
                     durable at this (object_id, to_seq), so the meta companion belongs to it \
                     and removing it would strip a live tail of its identity"
                );
                return;
            }
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => {
                tracing::warn!(
                    object_id,
                    to_seq,
                    error = %err,
                    "LocalDisk tail checkpoint put rollback {TAIL_ROLLBACK_PROBE_DECLINED_PHRASE}; \
                     skipping the rollback and leaving the meta companion for the reconciliation \
                     sweep (an unnecessary orphan is recoverable; deleting a live tail's meta is \
                     not)"
                );
                return;
            }
        }

        self.best_effort_remove(
            &self.tail_meta_path(object_id, to_seq),
            object_id,
            to_seq,
            "tail meta companion (orphaned by a failed tail publish)",
        )
        .await;
    }

    /// `<root>/.live-edit/sessions/<session_id>.json` — the session-checkpoint
    /// file path per SDD SR-4.6. `session_id` flows through directly; the
    /// production caller (Task `t2.fgj.7v.p9`) is the right enforcement point
    /// for path-traversal sanitization because `validate_session_name` at
    /// `CreateSession` ingress already rejects NUL and other unsafe bytes.
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir
            .join(format!("{}{}", session_id, SESSION_CHECKPOINT_SUFFIX))
    }

    /// Write-temp-then-rename atomic write. Temp filename is
    /// `<final>.<random-uuid>.tmp` so concurrent writers to the same final
    /// path never collide on the temp path.
    ///
    /// On error from either the write or the rename, the temp file is
    /// best-effort removed; a leaked temp file is a low-severity garbage
    /// condition that operators can scrub manually if it ever happens.
    async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
        let tmp_path = tmp_path_for(path);
        match tokio::fs::write(&tmp_path, bytes).await {
            Ok(()) => {}
            Err(err) => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(io_to_permanent(err));
            }
        }
        match tokio::fs::rename(&tmp_path, path).await {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                Err(io_to_permanent(err))
            }
        }
    }

    /// Walk the `layers` directory and collect every generation belonging to
    /// `object_id` as a `(sequence, kind)` pair — a base's `snap_seq` or a
    /// tail's `to_seq`, tagged with which it is.
    ///
    /// Admission is [`parse_generation_filename`], so companions (`.meta.json`,
    /// `.ext-refs.json`, `.tail.meta.json`) and in-flight `.tmp` files are
    /// excluded by the suffix partition documented at that parser.
    ///
    /// Ordering comes from the shared [`generation_sort_key`] — newest sequence
    /// first, Base before Tail at an equal sequence — and deliberately NOT from
    /// a comparator re-derived here, so LocalDisk, InMemory and StorageApi
    /// cannot drift on the tie-break.
    ///
    /// A `NotFound` on the directory itself resolves to an empty list rather
    /// than an error: an adapter whose `layers` dir was never created (or was
    /// removed underneath us) has no generations, which is the caller's
    /// "nothing checkpointed yet" path.
    async fn list_generations_for_object(
        &self,
        object_id: u64,
    ) -> Result<Vec<(u64, CheckpointKind)>, CheckpointError> {
        let mut reader = match tokio::fs::read_dir(&self.layers_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut generations: Vec<(u64, CheckpointKind)> = Vec::new();
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if let Some(generation) = parse_generation_filename(name, object_id) {
                generations.push(generation);
            }
        }
        // Newest-first per the `list_layer_checkpoints` trait contract, with the
        // cross-adapter Base-before-Tail tie-break.
        generations.sort_unstable_by_key(|&(sequence, kind)| generation_sort_key(sequence, kind));
        Ok(generations)
    }

    /// Read one generation's artifact + meta at an exact
    /// `(object_id, sequence, kind)`.
    ///
    /// The shared body behind [`CheckpointStore::get_latest_layer_checkpoint`]
    /// (which supplies the newest `(sequence, kind)` from
    /// [`Self::list_generations_for_object`]) and
    /// [`CheckpointStore::get_layer_checkpoint`] (which supplies the addressed
    /// handle's). Sharing it — rather than duplicating the read — is what keeps
    /// the trait's "`get_latest` == `list().first()`" conformance assertion true
    /// by construction: the two entry points cannot drift on read discipline,
    /// meta fallback, or error classification, because there is only one of each.
    ///
    /// `kind` selects the stem pair — `<oid>.<seq>.ompf` + `<oid>.<seq>.meta.json`
    /// for a [`CheckpointKind::Base`], `<oid>.<seq>.tail` +
    /// `<oid>.<seq>.tail.meta.json` for a [`CheckpointKind::Tail`] — and is
    /// therefore part of the address, not a hint: at an equal sequence the two
    /// kinds are two different artifacts.
    ///
    /// Read discipline, in order:
    /// - artifact `NotFound` ⇒ `Ok(None)`. For `get_latest` that is the
    ///   concurrent-prune race (the directory walk just listed the name); for
    ///   the by-handle get it is the same race plus the ordinary
    ///   never-existed-here case. Absence is never `NotFound` on a get path.
    /// - meta companion absent ⇒ the typed
    ///   [`MissingMetaCompanion`] `Permanent`, for BOTH kinds, with a WARN
    ///   (b.hka (d); see the `NotFound` arm below). No placeholder meta is
    ///   synthesized on any path.
    /// - meta companion malformed ⇒ [`CheckpointError::Permanent`]: the bytes
    ///   are durable and were last written by us, so a retry cannot fix them.
    ///   Identical for both kinds.
    /// - `sidecar_bytes` on the returned meta is always `None` — the ext-refs
    ///   sidecar is a separate artifact, fetched on demand, and a tail has none
    ///   at all (SR-2.4).
    /// - `blob.handle` carries **this adapter's** canonical `adapter_key` for the
    ///   generation (the artifact path) and the addressed `kind`, never an echo
    ///   of a caller-supplied key.
    async fn read_generation_at(
        &self,
        object_id: u64,
        sequence: u64,
        kind: CheckpointKind,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        let (artifact_path, meta_path) = match kind {
            CheckpointKind::Base => (
                self.ompf_path(object_id, sequence),
                self.meta_path(object_id, sequence),
            ),
            CheckpointKind::Tail => (
                self.tail_path(object_id, sequence),
                self.tail_meta_path(object_id, sequence),
            ),
        };
        let bytes = match tokio::fs::read(&artifact_path).await {
            Ok(b) => Bytes::from(b),
            // Return `Ok(None)` rather than surfacing a synthetic error — the
            // caller's "no checkpoint yet" path is the right next step.
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(io_to_permanent(err)),
        };
        // Read the kind's meta companion and deserialize into `CheckpointMeta`.
        // An absent one refuses the generation — same typed `Permanent` for both
        // kinds; see the `NotFound` arm below.
        let meta = match tokio::fs::read(&meta_path).await {
            Ok(meta_bytes) => match serde_json::from_slice::<CheckpointMeta>(&meta_bytes) {
                Ok(parsed) => CheckpointMeta {
                    // `sidecar_bytes` is `#[serde(skip)]` and so the
                    // deserialized value carries `None`; the ext-refs
                    // sidecar is fetched separately by the orphan sweep
                    // via `list_referenced_ext_hashes`, not threaded back
                    // through `CheckpointBlob`. Leaving it `None` here
                    // matches the trait's get-path contract.
                    sidecar_bytes: None,
                    ..parsed
                },
                Err(err) => {
                    // A malformed meta sidecar is a permanent corruption
                    // signal — the bytes were last written by us, so
                    // re-read won't fix it. Surface so the operator can
                    // investigate rather than silently downgrading to
                    // empty identity.
                    return Err(CheckpointError::Permanent {
                        source: format!(
                            "CheckpointMeta JSON parse failed for {}: {}",
                            meta_path.display(),
                            err
                        )
                        .into(),
                    });
                }
            },
            // No meta companion beside a durable artifact. Refused as a typed
            // `Permanent` for BOTH kinds (b.hka (d)), where both arms used to
            // synthesize `CheckpointMeta::default()` and return `Ok(Some(..))`.
            //
            // The base arm's placeholder existed for a "legacy pre-meta
            // generation" population that is empty — no release tag predates
            // the meta companion (all tags are pre-release), so nothing
            // deployed predates it —
            // and since b.b2j its empty identity is refused by
            // `apply::live_mapping_resolves_blob_identity` anyway, three layers
            // up and as a generic `NotFound` that names neither the artifact
            // nor the missing file. The tail arm's placeholder was Epic 2's
            // deliberate mirror of the base arm, pinned (`t2.dfh.gy.uv` item 4)
            // to be revisited rather than preserved once this landed; under the
            // shipped meta-before-artifact publish and artifact-before-meta
            // delete order the state is unreachable in both windows for both
            // kinds, so corruption / torn rolled-back put / tampering is the
            // whole reachable population and a fabricated meta serves none of
            // it.
            //
            // Deleting the tail placeholder also removes the ONE place a `Tail`
            // meta could be observed violating `Tail ⇒ base_snap_seq.is_some()
            // && from_seq.is_some()`, and removes this adapter from the
            // empty-identity producer list `crate::checkpoint::apply` maintains.
            //
            // WARN, not silent: nothing legitimately reaches here any more.
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                tracing::warn!(
                    object_id,
                    sequence,
                    kind = kind.as_str(),
                    artifact_path = %artifact_path.display(),
                    meta_path = %meta_path.display(),
                    "LocalDisk checkpoint artifact is durable but its meta companion is absent; \
                     refusing the generation as unrestorable rather than serving an \
                     identity-empty placeholder meta"
                );
                return Err(CheckpointError::missing_meta_companion(
                    MissingMetaCompanion {
                        artifact: artifact_path.to_string_lossy().into_owned(),
                        companion: meta_path.to_string_lossy().into_owned(),
                        object_id,
                        sequence,
                        kind,
                    },
                ));
            }
            Err(err) => return Err(io_to_permanent(err)),
        };
        let adapter_key = artifact_path.to_string_lossy().into_owned();
        let handle = match kind {
            CheckpointKind::Base => CheckpointHandle::base(object_id, sequence, adapter_key),
            CheckpointKind::Tail => CheckpointHandle::tail(object_id, sequence, adapter_key),
        };
        Ok(Some(CheckpointBlob {
            handle,
            bytes,
            meta,
        }))
    }
}

/// Shape of the ext-refs sidecar JSON the assembler emits. Parsed on demand
/// inside `list_referenced_ext_hashes` because storage is opaque bytes
/// (SR-1.5) — the adapter does not interpret the sidecar at put time.
#[derive(Debug, Deserialize)]
struct ExtRefsSidecar {
    ext_hashes: Vec<String>,
}

/// Map a `std::io::Error` to a [`CheckpointError::Permanent`]. The
/// `NotFound` path is handled at the call sites (it maps to `Ok(None)` for
/// `get_*`, `Ok(())` for `delete_*`, `Ok(false)` for `ext_blob_exists`, and an
/// empty result for a directory walk); everything else is permanent in
/// the LocalDisk adapter — transient I/O recoveries are the operator's
/// problem at the filesystem layer, not retry surface here.
fn io_to_permanent(err: io::Error) -> CheckpointError {
    CheckpointError::Permanent {
        source: Box::new(err),
    }
}

/// Build the temp-file path for an atomic write: `<final>.<uuid>.tmp`. Using
/// `uuid::Uuid::new_v4()` for the disambiguator is overkill for the single-
/// writer-per-object steady-state but cheap and keeps concurrent writers
/// from ever colliding on a shared temp path (e.g. two replicas racing on
/// the same `(object_id, snap_seq)` slot — the rename step is what serializes
/// them, and unique temp paths keep them from clobbering each other before
/// the rename).
fn tmp_path_for(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let mut new_name = path
        .file_name()
        .map(OsStr::to_os_string)
        .unwrap_or_default();
    new_name.push(format!(".{}{}", Uuid::new_v4(), TEMP_SUFFIX));
    tmp.set_file_name(new_name);
    tmp
}

/// Parse `<object_id>.<snap_seq>.ompf`. Returns the `snap_seq` iff the
/// filename matches and the leading number equals the requested `object_id`.
///
/// Built on the shared [`parse_layer_artifact_stem`], exactly as the StorageApi
/// siblings are, so this walk and the reconciliation artifact surface admit the
/// same names: strict two components, each the canonical rendering of its `u64`
/// (`12.3.4.ompf`, `007.3.ompf` and `+12.3.ompf` are all rejected rather than
/// silently aliased onto `12.3.ompf`'s stem).
fn parse_ompf_filename(name: &str, expected_object_id: u64) -> Option<u64> {
    let (object_id, snap_seq) = parse_layer_artifact_stem(name.strip_suffix(OMPF_SUFFIX)?)?;
    if object_id != expected_object_id {
        return None;
    }
    Some(snap_seq)
}

/// Parse the `<object_id>` half of an `<oid>.<snap_seq>.ompf` filename.
fn parse_ompf_object_id(name: &str) -> Option<u64> {
    let (object_id, _snap_seq) = parse_layer_artifact_stem(name.strip_suffix(OMPF_SUFFIX)?)?;
    Some(object_id)
}

// # Why the OMPF and tail parser families cannot alias
//
// This suffix partition is the whole safety case for walking one flat `layers`
// directory that holds both generation kinds and their companions:
//
// - `<oid>.<seq>.meta.json`, `<oid>.<seq>.ext-refs.json` and
//   `<oid>.<to_seq>.tail.meta.json` all end in `.json`, so neither
//   `strip_suffix(OMPF_SUFFIX)` nor `strip_suffix(TAIL_SUFFIX)` matches them. In
//   particular the tail meta is NOT admitted as a tail envelope.
// - `<oid>.<to_seq>.tail` does not end in `.ompf`, and `<oid>.<seq>.ompf` does
//   not end in `.tail`, so the two generation families are disjoint in both
//   directions.
// - Every temp file `tmp_path_for` mints ends in `.tmp`, so no parser admits
//   one. That exclusion is precisely what makes an in-flight `atomic_write`
//   invisible to a listing — i.e. it is the publish-order guarantee, not a
//   tidiness rule: a generation becomes enumerable only at its rename.
//
// The four parsers around this block share one basis, which is what makes the
// non-overlap provable by inspection rather than by test enumeration: each is
// (a) a `strip_suffix` of one shared suffix constant, partitioning the
// namespace as above, composed with (b) the single shared
// `parse_layer_artifact_stem`, which is where the strict two-component,
// canonical-`u64` rejection lives. Neither half is re-derived per parser, so
// the four cannot drift apart from each other or from the reconciliation
// surface's `parse_layer_artifact_filename`, which is built on the same two
// pieces.

/// Parse `<object_id>.<to_seq>.tail`. Returns the `to_seq` iff the filename
/// matches and the leading number equals the requested `object_id`.
fn parse_tail_filename(name: &str, expected_object_id: u64) -> Option<u64> {
    let (object_id, to_seq) = parse_layer_artifact_stem(name.strip_suffix(TAIL_SUFFIX)?)?;
    if object_id != expected_object_id {
        return None;
    }
    Some(to_seq)
}

/// Parse the `<object_id>` half of an `<oid>.<to_seq>.tail` filename.
fn parse_tail_object_id(name: &str) -> Option<u64> {
    let (object_id, _to_seq) = parse_layer_artifact_stem(name.strip_suffix(TAIL_SUFFIX)?)?;
    Some(object_id)
}

/// Kind-dispatching parse of one `layers` directory entry belonging to
/// `expected_object_id`: `<oid>.<snap_seq>.ompf` yields
/// `(snap_seq, CheckpointKind::Base)`, `<oid>.<to_seq>.tail` yields
/// `(to_seq, CheckpointKind::Tail)`, everything else yields `None`.
///
/// This and [`parse_generation_object_id`] are the ONLY parsers the
/// directory walks call, so adding a third generation kind is one edit here
/// rather than a hunt through every walk.
fn parse_generation_filename(name: &str, expected_object_id: u64) -> Option<(u64, CheckpointKind)> {
    if let Some(snap_seq) = parse_ompf_filename(name, expected_object_id) {
        return Some((snap_seq, CheckpointKind::Base));
    }
    parse_tail_filename(name, expected_object_id).map(|to_seq| (to_seq, CheckpointKind::Tail))
}

/// Kind-dispatching parse of the `<object_id>` half of any generation
/// filename — the object-enumeration counterpart to
/// [`parse_generation_filename`].
fn parse_generation_object_id(name: &str) -> Option<u64> {
    parse_ompf_object_id(name).or_else(|| parse_tail_object_id(name))
}

#[async_trait]
impl CheckpointStore for LocalDiskCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.maybe_inject("put_layer_checkpoint")?;
        // The shared put-time gate (lineage complete, no ext-refs sidecar on a
        // tail). Lives in `adapters/mod.rs` so all three adapters enforce one
        // copy of the rule; see its doc for why each rejection is `Permanent`.
        // It runs *after* `maybe_inject` — the injector is the outermost
        // short-circuit on every method — and before any path derivation, so a
        // rejected tail put writes no file of any kind.
        validate_tail_put(&meta)?;
        // Its sibling gate, same site and same posture: no generation of either
        // kind is persisted with an empty `session_id`/`layer_name`, because
        // the restore path refuses one. Store-side backstop for the assembler's
        // own refusal on the vanished-`obj:hdr` arm; no file is written.
        validate_generation_identity(&meta)?;

        if meta.kind == CheckpointKind::Tail {
            // ── Tail publish order: meta → envelope (SR-2.3) ──────────────
            //
            // The trait's tail publish-order MUST is "the tail artifact MUST
            // NOT become visible to `list_layer_checkpoints` before its bytes
            // are fully written". Listing here is `.tail`-**stem**-driven
            // (`parse_generation_filename`), so writing the meta first makes a
            // listed tail one that *always* has its meta: there is no window in
            // which a concurrent lister sees a tail whose meta is absent.
            // Envelope-first would open exactly that window and force the read
            // path to tolerate it as an ordinary state rather than the torn
            // residual it is.
            //
            // This is NOT a divergence from the base path. Since b.9qd the base
            // path already publishes its meta *before* its artifact, for the
            // same reason in the base's own terms: a durable OMPF whose meta
            // never landed shadows the last good generation and restores it with
            // empty session identity. The tail branch simply inherits that
            // order — and, unlike the base, has no legacy meta-absent population
            // to tolerate (tails only exist from this Task forward), so a
            // meta-absent tail can only mean a torn publish. Since b.hka (d)
            // that is what `read_generation_at` reports for BOTH kinds: a typed
            // `MissingMetaCompanion` refusal with a WARN, no placeholder meta.
            //
            // The base path's *sidecar*-first argument likewise does not put a
            // ceiling on this order: an orphaned `.ext-refs.json` is unreapable
            // and pins its hashes in the retention cascade's live set forever,
            // whereas an orphaned `.tail.meta.json` references no ext hashes at
            // all — its whole cost is one stranded file. Meta-first therefore
            // buys the stronger visibility invariant at a strictly smaller
            // residual. No `.ext-refs.json` is written on this branch under any
            // circumstance: it never calls `sidecar_path`, and `validate_tail_put`
            // has already rejected a `Some(sidecar_bytes)`.
            //
            // Serialize up front so an encoding failure aborts before any bytes
            // land, mirroring the base path.
            let meta_bytes = serde_json::to_vec(&meta).map_err(|e| CheckpointError::Permanent {
                source: format!("CheckpointMeta JSON serialization failed: {e}").into(),
            })?;

            let tail_meta_path = self.tail_meta_path(object_id, snap_seq);
            Self::atomic_write(&tail_meta_path, &meta_bytes).await?;

            let tail_path = self.tail_path(object_id, snap_seq);
            if let Err(e) = Self::atomic_write(&tail_path, &ompf).await {
                // Deliberately NOT `rollback_partial_publish`: that helper's
                // guard is a durable-OMPF probe, which is meaningless for a tail
                // stem — `ompf_path` names a different generation than
                // `tail_path` at the same sequence, so the probe would answer a
                // question about the base while deciding the tail's cleanup.
                // `rollback_partial_tail_publish` is the tail-addressed sibling:
                // the same guard asked at `tail_path`, then the best-effort
                // removal of the meta we just wrote (there is no third
                // artifact). Removing it unguarded — the round-1 shape — would
                // strip the identity from an envelope that is durable from an
                // earlier attempt at the same sequence, which since the meta
                // placeholder's removal costs a recovery point rather than a
                // WARN (b.hka review round 2).
                self.rollback_partial_tail_publish(object_id, snap_seq)
                    .await;
                return Err(e);
            }

            return Ok(CheckpointHandle::tail(
                object_id,
                snap_seq,
                tail_path.to_string_lossy().into_owned(),
            ));
        }

        // ── Base branch ───────────────────────────────────────────────────
        // Serialize meta up front so an encoding failure aborts the triplet
        // before any bytes land on disk.
        //
        // `sidecar_bytes` is `#[serde(skip)]` on `CheckpointMeta` — the
        // ext-refs sidecar is a distinct `.ext-refs.json` file written
        // below; duplicating it inside the meta JSON would store the bytes
        // twice.
        let meta_bytes = serde_json::to_vec(&meta).map_err(|e| CheckpointError::Permanent {
            source: format!("CheckpointMeta JSON serialization failed: {e}").into(),
        })?;

        // ── Publish order: ext-refs → meta → OMPF (b.s5y, b.9qd) ──────────
        //
        // Load-bearing, and the reverse of what this adapter did before
        // b.s5y. The rule is: **the ext-refs sidecar must be visible no
        // later than the OMPF whose blobs it protects.**
        //
        // The retention-prune cascade in `checkpointer::prune_old_generations`
        // computes `prunable = pruned_hashes − live_hashes`, where the live
        // set comes from `list_referenced_ext_hashes` — which enumerates
        // `*.ext-refs.json`. Under duplicate leadership (two replicas holding
        // the lock across a lapse, which no TTL can make impossible — one
        // adapter call can outlive the lease) one replica can be mid-`put` for
        // a new generation while the other runs the cascade. If the OMPF
        // landed first, the in-flight generation's ext-refs would still be
        // invisible to the live set, so hashes only that generation
        // references classify as prunable and the cascade deletes blobs a
        // *durable* checkpoint points at — a checkpoint that restores with
        // dangling ext-refs, i.e. silent unrecoverable data loss on a real DR
        // restore.
        //
        // Sidecar-first **shrinks** that window rather than closing it: the
        // generation is invisible to the live set only until the sidecar
        // lands, instead of until the OMPF lands. On this adapter the sidecar
        // is the first write, so the residual is a single `atomic_write`
        // wide — but it is not zero, and the lease gate at the cascade's call
        // site is what keeps a peer from entering it. See
        // `checkpointer::prune_old_generations` for the precise residual.
        //
        // The prior comment justified OMPF-first on the grounds that "a
        // dangling sidecar pointing at a missing OMPF is the more hazardous
        // failure mode". That is wrong for this adapter's actual sidecar
        // readers, of which there are exactly two:
        //   - the cascade's live-set union (`list_referenced_ext_hashes`),
        //     which *wants* the extras — an unreferenced hash in the live set
        //     costs one deferred blob delete, nothing more; and
        //   - `read_ext_refs_sidecar`, which is addressed per-handle from
        //     handles produced by `list_layer_checkpoints` — a
        //     generation-stem-driven enumeration (`*.ompf` for a base,
        //     `*.tail` for a tail) — so on the base handles this branch
        //     publishes it structurally never sees a sidecar whose OMPF is
        //     absent. (A `Tail` handle short-circuits to the empty set before
        //     any sidecar path is derived; a tail owns no sidecar.)
        // Neither is harmed by an orphan sidecar; both are harmed by a
        // missing one.
        //
        // ── Why meta moved ahead of the OMPF (b.9qd) ─────────────────────
        //
        // Until b.9qd the order was ext-refs → OMPF → meta, on the grounds
        // that the read path tolerated an absent meta. A durable OMPF whose
        // meta never landed is now listed as a generation but refused on read
        // (`MissingMetaCompanion`), so the restore walk must step down past it.
        // Before that refusal existed it was served with a synthesized,
        // identity-empty meta that **shadowed the last good generation** —
        // `session_id`, `layer_name`, `created_by`, `server_ts` all blank —
        // a worse DR outcome than the generation simply not existing.
        //
        // Publishing meta before the OMPF closes it: the OMPF is still the
        // last artifact, so "visible as a generation" continues to imply
        // "fully durable", and now it also implies "has its identity". Both
        // adapters share this order as of b.9qd.
        //
        // ── Rollback on failure ──────────────────────────────────────────
        //
        // An orphaned `.ext-refs.json` is NOT self-healing, and an earlier
        // version of this comment wrongly said it was ("the next put for the
        // same (object_id, snap_seq) overwrites it"). That is true only for a
        // repeat at the *identical* snap_seq, which does not recur once the
        // layer takes another commit — the next attempt uses a new snap_seq
        // and writes a different filename. Meanwhile every deletion path in
        // this adapter is generation-stem-driven (`list_layer_checkpoints`
        // enumerates `*.ompf` and `*.tail` stems; a companion is only ever
        // deleted alongside the generation that owns it), so a sidecar whose
        // OMPF never landed is unreapable by the
        // retention prune — and it keeps pinning that snapshot's entire ext
        // set in the live set, which degrades the cascade toward deleting
        // nothing at all. (Since b.9qd the reconciliation pass in
        // `checkpointer::run_reconcile_tick` does eventually reap it; that is
        // a backstop measured in the reconcile interval, not a licence to
        // leave orphans behind here.)
        //
        // So a generation that will not become durable must not leave one
        // behind: if a later step fails, best-effort remove the artifacts we
        // wrote. Failures of the cleanup itself are warn-and-swallowed — the
        // original error is the one worth surfacing.
        let sidecar_written = if let Some(sidecar) = meta.sidecar_bytes.as_ref() {
            let sidecar_path = self.sidecar_path(object_id, snap_seq);
            if let Err(e) = Self::atomic_write(&sidecar_path, sidecar).await {
                // The sidecar write itself failed. `atomic_write` cleans up
                // its own temp file, but the rename may still have landed
                // before a later step of the helper failed, so scrub
                // defensively rather than reason about which.
                self.best_effort_remove(&sidecar_path, object_id, snap_seq, "ext-refs sidecar")
                    .await;
                return Err(e);
            }
            true
        } else {
            false
        };

        let meta_path = self.meta_path(object_id, snap_seq);
        if let Err(e) = Self::atomic_write(&meta_path, &meta_bytes).await {
            self.rollback_partial_publish(object_id, snap_seq, sidecar_written)
                .await;
            return Err(e);
        }

        let ompf_path = self.ompf_path(object_id, snap_seq);
        if let Err(e) = Self::atomic_write(&ompf_path, &ompf).await {
            self.rollback_partial_publish(object_id, snap_seq, sidecar_written)
                .await;
            return Err(e);
        }

        Ok(CheckpointHandle::base(
            object_id,
            snap_seq,
            ompf_path.to_string_lossy().into_owned(),
        ))
    }

    /// The newest generation of **either kind** — deliberately not "the newest
    /// base".
    ///
    /// The trait pins `get_latest == list_layer_checkpoints(oid).first()`
    /// verbatim in the tail era, so when the newest generation is a tail this
    /// MUST return that tail. Filtering to bases here is the tempting and wrong
    /// change: SR-2.5's "newest base wins" is **restore-path policy**,
    /// implemented once by kind-aware selection at the recovery hook (Task 5,
    /// `t2.dfh.gy.4o`: list → newest `Base` handle → `get_layer_checkpoint`),
    /// not three times adapter-side where it can only be tested three times —
    /// and re-defining it here would break the SR-1.3 conformance assertion
    /// outright.
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_latest_layer_checkpoint")?;
        // `list_generations_for_object` is the same directory walk
        // `list_layer_checkpoints` uses and is sorted newest-first through the
        // shared `generation_sort_key`, so `.first()` here IS `list().first()`
        // — kind included; the read that follows is the one
        // `get_layer_checkpoint` performs. The trait's
        // "`get_latest` == `list().first()`" equivalence is therefore
        // structural, not an assertion this method has to remember to honour.
        let generations = self.list_generations_for_object(object_id).await?;
        let Some(&(sequence, kind)) = generations.first() else {
            return Ok(None);
        };
        self.read_generation_at(object_id, sequence, kind).await
    }

    /// Fetch one specific generation, addressed by handle (SR-2.3).
    ///
    /// Addressed off `(object_id, snap_seq, kind)` through the adapter-internal
    /// path helpers. `kind` is part of the address, not a hint: at an equal
    /// sequence `base@S` and `tail@to_seq == S` are two different files, and each
    /// resolves only to its own stem.
    ///
    /// `adapter_key` is opaque carrier state on the trait surface and is
    /// deliberately ignored, the same stance [`Self::read_ext_refs_sidecar`]
    /// takes — a caller that synthesized the handle from
    /// `list_layer_checkpoints`, or fabricated one with a junk key, must still
    /// resolve, and the returned handle carries this adapter's own canonical key
    /// (the artifact path for the addressed kind).
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_layer_checkpoint")?;
        self.read_generation_at(handle.object_id, handle.snap_seq, handle.kind)
            .await
    }

    /// Every generation of the object, **both kinds**, newest-first.
    ///
    /// Order and tie-break come from the shared `generation_sort_key` via
    /// [`Self::list_generations_for_object`] — never re-derived here — so this
    /// adapter cannot drift from InMemory / StorageApi on the pinned
    /// Base-before-Tail ordering at an equal sequence.
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.maybe_inject("list_layer_checkpoints")?;
        let handles = self
            .list_generations_for_object(object_id)
            .await?
            .into_iter()
            .map(|(sequence, kind)| match kind {
                // The `adapter_key` is per-kind: the OMPF path for a base, the
                // `.tail` path for a tail. Minting both as an OMPF path would
                // hand a tail handle a key naming a *different* generation.
                CheckpointKind::Base => CheckpointHandle::base(
                    object_id,
                    sequence,
                    self.ompf_path(object_id, sequence)
                        .to_string_lossy()
                        .into_owned(),
                ),
                CheckpointKind::Tail => CheckpointHandle::tail(
                    object_id,
                    sequence,
                    self.tail_path(object_id, sequence)
                        .to_string_lossy()
                        .into_owned(),
                ),
            })
            .collect();
        Ok(handles)
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_layer_checkpoint")?;
        if handle.kind == CheckpointKind::Tail {
            // ── Tail delete order: envelope → meta (SR-2.4) ───────────────
            //
            // The mirror image of the meta-first publish order: publish
            // establishes the meta before the envelope, delete removes the
            // envelope before the meta, so "envelope visible ⇒ meta present"
            // holds at every instant, in both directions of the artifact's
            // lifetime.
            //
            // The residual on a partial failure is an orphaned
            // `.tail.meta.json` — stranded but harmless: it references no ext
            // hashes, so unlike an orphaned `.ext-refs.json` it pins nothing in
            // the retention cascade's live set. It is not permanent either: with
            // no `.tail` envelope at its stem it is a Tail-family orphan
            // candidate to the reconciliation sweep, registered on first
            // sighting and reaped a grace later. That is the same trade the base
            // path already accepts for its own meta companion.
            //
            // Only the two tail paths are touched. `ompf_path` / `sidecar_path`
            // / `meta_path` are kind-blind, so reaching for them here would
            // unlink the BASE triplet at `snap_seq == to_seq` — the
            // stranded-generation shape SR-2.4 / SR-2.5 forbid.
            //
            // Both removals are `NotFound`-tolerant, so deleting an
            // already-absent tail returns `Ok(())` by construction, per the
            // trait's idempotent-on-missing contract.
            for path in [
                self.tail_path(handle.object_id, handle.snap_seq),
                self.tail_meta_path(handle.object_id, handle.snap_seq),
            ] {
                match tokio::fs::remove_file(&path).await {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => return Err(io_to_permanent(err)),
                }
            }
            return Ok(());
        }
        let ompf_path = self.ompf_path(handle.object_id, handle.snap_seq);
        let sidecar_path = self.sidecar_path(handle.object_id, handle.snap_seq);
        let meta_path = self.meta_path(handle.object_id, handle.snap_seq);
        // All three deletes are idempotent on missing per the trait
        // contract. The meta.json companion joins this set as of
        // t3.fgj.ya.n9.b1 — without it a delete would leave a dangling
        // `<oid>.<snap_seq>.meta.json` behind that the next
        // `get_latest_layer_checkpoint` for a re-created (oid, snap_seq)
        // tuple would falsely round-trip into the new blob.
        //
        // ── Delete order: ext-refs → OMPF → meta (b.s5y R2) ──────────────
        //
        // The mirror image of `put_layer_checkpoint`'s publish order, and
        // load-bearing for the same reason. Deleting the OMPF first — what
        // this adapter did before — meant a partial failure left a sidecar
        // whose OMPF was gone. A base's companions are only ever offered for
        // deletion alongside the OMPF stem that owns them
        // (`list_layer_checkpoints` enumerates the `*.ompf` stem for a base),
        // so that sidecar would never be offered for deletion again: a
        // permanent orphan pinning this snapshot's whole ext set in the
        // retention cascade's live set.
        //
        // Ext-refs first inverts the residual into a self-correcting one. A
        // failure after the sidecar delete leaves the OMPF present, so the
        // generation stays enumerable and the next tick's prune retries the
        // whole delete. In the meantime `read_ext_refs_sidecar` returns the
        // empty set for the missing sidecar, which under-counts the pruned
        // hashes — the documented-safe direction (under-delete, never
        // over-delete).
        for path in [&sidecar_path, &ompf_path, &meta_path] {
            match tokio::fs::remove_file(path).await {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(io_to_permanent(err)),
            }
        }
        Ok(())
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.maybe_inject("put_session_checkpoint")?;
        let path = self.session_path(session_id);
        // `meta` is intentionally discarded: per SR-2.7 the session-checkpoint
        // Task-1 payload (`SessionCheckpoint` — `name`, `base_path`,
        // `created_by`, `created_at`, plus the layer-mapping tables) carries
        // provenance inside the body envelope. The layer-checkpoint adapter
        // persists meta because `LayerStore::apply_checkpoint` needs identity
        // to rebuild `obj:hdr`; the session path has the identity inside the
        // body and a `.meta.json` companion would double-store the same fields.
        let _ = meta;
        // Write-temp + rename — readers never observe a torn file, matches the
        // layer-checkpoint atomicity discipline.
        Self::atomic_write(&path, &body).await?;
        // Handle mirrors the InMemory sibling: session checkpoints carry no
        // `(object_id, snap_seq)` identity, so we surface zeros and use the
        // on-disk path as the opaque `adapter_key` carrier.
        Ok(CheckpointHandle::base(
            0,
            0,
            path.to_string_lossy().into_owned(),
        ))
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_session_checkpoint")?;
        let path = self.session_path(session_id);
        // Per SR-1.5 the adapter treats the body as opaque bytes: corrupt-bytes
        // classification (`Permanent`) is the job of
        // `SessionCheckpoint::from_bytes` (Task 1), not the adapter. We just
        // hand the bytes back; the caller parses them.
        match tokio::fs::read(&path).await {
            Ok(b) => Ok(Some(CheckpointBlob {
                handle: CheckpointHandle::base(0, 0, path.to_string_lossy().into_owned()),
                bytes: Bytes::from(b),
                // Defaulted because `put_session_checkpoint` does not persist
                // meta — provenance lives inside the `SessionCheckpoint` JSON
                // body (Task 1 payload). Callers needing it parse the body.
                meta: CheckpointMeta::default(),
            })),
            // Trait contract: `get_*` returns `Ok(None)` on absent, never
            // `NotFound` (see `store.rs:270-273`).
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_session_checkpoint")?;
        let path = self.session_path(session_id);
        // The delete only removes the JSON file; there is no per-session meta
        // or sidecar companion to scrub (put_session_checkpoint never wrote
        // one). Idempotent on missing per the trait contract.
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    /// The stem of every `<session_id>.json` file under `sessions/`.
    ///
    /// Write temps are `<session_id>.json.<uuid>.tmp` ([`tmp_path_for`]), so
    /// they end in `.tmp`, not `.json`, and fall out of the suffix match; so
    /// does any other stray file. Directories are skipped: no session put
    /// creates one, and [`Self::get_session_checkpoint`] could not read one.
    /// A name that is not UTF-8 is skipped, as the `layers/` walks skip it — a
    /// session put never writes one.
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.maybe_inject("list_session_checkpoint_ids")?;
        let mut reader = match tokio::fs::read_dir(&self.sessions_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut session_ids = Vec::new();
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            let Some(session_id) = name.strip_suffix(SESSION_CHECKPOINT_SUFFIX) else {
                continue;
            };
            let file_type = entry.file_type().await.map_err(io_to_permanent)?;
            if file_type.is_dir() {
                continue;
            }
            session_ids.push(session_id.to_string());
        }
        Ok(session_ids)
    }

    /// Write `ext/<hash_hex>.blob` from `bytes`, or skip it when the file
    /// already exists.
    ///
    /// The existence check, the write to a temp file and the rename run in
    /// **one blocking task that owns `bytes`**, not through
    /// [`Self::atomic_write`]:
    ///
    /// - **No copy.** The file is written from the caller's bytes. A write
    ///   through `tokio::fs::write` of a borrowed slice copies it into its
    ///   blocking task first, an uncounted second copy of a value up to the
    ///   upload size cap beside the ext-mirror's charged one.
    /// - **The charge lasts as long as the write.** The ext-mirror's byte
    ///   charge rides inside `bytes` (`checkpoint::ext_mirror`), and the task
    ///   holds them until the write ends, so a write in progress is always
    ///   counted against `M_ext`.
    /// - **It finishes when its caller stops waiting.** A caller dropped
    ///   mid-write (a timeout, the repair's pass deadline, shutdown) detaches
    ///   from the task, which still ends in the rename or in removing its
    ///   temp: no partial file is left, and a write that lands late is
    ///   confirmed by the backup repair's next existence check. Only a hard
    ///   kill or crash mid-write leaves a temp behind.
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.maybe_inject("put_ext_blob")?;
        let path = self.ext_path(hash_hex);
        tokio::task::spawn_blocking(move || {
            // Content-addressed idempotency — if a file with this hash already
            // exists its bytes are by definition identical to the new bytes
            // (the hash is the contract). Skip the write entirely so re-puts
            // are cheap, matching the trait contract and the InMemory
            // adapter's no-op behavior.
            match std::fs::metadata(&path) {
                Ok(_) => return Ok(()),
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(io_to_permanent(err)),
            }
            let tmp_path = tmp_path_for(&path);
            let written = std::fs::write(&tmp_path, &bytes[..])
                .and_then(|()| std::fs::rename(&tmp_path, &path));
            if let Err(err) = written {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(io_to_permanent(err));
            }
            Ok(())
        })
        .await
        .map_err(|join_err| CheckpointError::Permanent {
            source: Box::new(join_err),
        })?
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.maybe_inject("get_ext_blob")?;
        let path = self.ext_path(hash_hex);
        match tokio::fs::read(&path).await {
            Ok(b) => Ok(Some(Bytes::from(b))),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    /// A metadata probe on `ext/<hash_hex>.blob`, the same presence test
    /// [`Self::put_ext_blob`]'s idempotency skip uses. Only `NotFound` is a
    /// positive "absent"; any other I/O error (a permission or read-only
    /// mount fault) is an indeterminate answer and stays an error.
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.maybe_inject("ext_blob_exists")?;
        let path = self.ext_path(hash_hex);
        match tokio::fs::metadata(&path).await {
            Ok(_) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_ext_blob")?;
        let path = self.ext_path(hash_hex);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            // Idempotent on missing hash per trait contract.
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.maybe_inject("put_counters_sidecar")?;
        Self::atomic_write(&self.counters_path, &body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.maybe_inject("get_counters_sidecar")?;
        match tokio::fs::read(&self.counters_path).await {
            Ok(b) => Ok(Some(Bytes::from(b))),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(io_to_permanent(err)),
        }
    }

    /// Every `object_id` with at least one **generation** artifact under
    /// `layers/` — `*.ompf` or `*.tail`.
    ///
    /// The walk is kind-agnostic via [`parse_generation_object_id`]. That is the
    /// resolution of the obligation Task 2 left here: an object whose only
    /// surviving artifact is a tail MUST still be enumerated. It is not
    /// restorable on its own, but this method's callers need to *see* it in
    /// order to reach that conclusion — a tail-only object that no enumeration
    /// reports is one no retention, reconciliation or recovery pass can act on,
    /// which is a silent-leak shape rather than a conservative one.
    ///
    /// Companions stay excluded for free: `.meta.json`, `.ext-refs.json`,
    /// `.tail.meta.json` and every `.tmp` fall outside both parsers' suffixes
    /// (the partition argument above [`parse_tail_filename`]), so no companion —
    /// and no in-flight `atomic_write` — can conjure an object id.
    ///
    /// Dedup is over `object_id` alone, so an object with several generations of
    /// either kind still reports once.
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.maybe_inject("list_all_object_ids")?;
        let mut reader = match tokio::fs::read_dir(&self.layers_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut object_ids: HashSet<u64> = HashSet::new();
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if let Some(oid) = parse_generation_object_id(name) {
                object_ids.insert(oid);
            }
        }
        Ok(object_ids.into_iter().collect())
    }

    /// Union of every `*.ext-refs.json` sidecar under `layers/` — the retention
    /// cascade's live set.
    ///
    /// Walking the **sidecars themselves** rather than the generations that own
    /// them is load-bearing, not incidental (SR-2.4/SR-5.3), and now that tails
    /// genuinely exist on disk it is what keeps this method correct without a
    /// kind branch: a tail carries no ext-refs sidecar by design, so no `*.tail`
    /// stem can contribute to the live set no matter how many of them the
    /// directory holds. Keep it that way — re-deriving the live set from an
    /// enumeration of generations would put tails back in scope and require the
    /// branch this shape avoids. The structural guarantee behind it is the
    /// absence of any tail-sidecar path helper (see [`Self::sidecar_path`]) plus
    /// `validate_tail_put`'s rejection of a tail carrying `sidecar_bytes`.
    ///
    /// The walk itself lives in [`Self::list_live_ext_set`] (b.oya); this
    /// method is its projection onto the hash set.
    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        Ok(self.list_live_ext_set().await?.hashes)
    }

    /// The live-set walk [`Self::list_referenced_ext_hashes`] documents, plus
    /// `listed_layer_artifacts`: how many names in the **same** `read_dir`
    /// pass [`parse_layer_artifact_filename`] accepts, of any kind — counted
    /// before the `.ext-refs.json` filter, so it equals what
    /// `list_layer_artifacts` would report from that pass (b.oya; see the
    /// trait doc). A missing `layers/` directory yields `(∅, 0)`.
    ///
    /// Injects under the `list_referenced_ext_hashes` label: it is that
    /// method's walk, and the fault-injection tests target that label.
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.maybe_inject("list_referenced_ext_hashes")?;
        let mut reader = match tokio::fs::read_dir(&self.layers_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Ok(LiveExtSet::default());
            }
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut hashes: HashSet<String> = HashSet::new();
        let mut listed_layer_artifacts: usize = 0;
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            // The same-request evidence: counted in this very `read_dir`
            // pass, before the sidecar filter.
            if parse_layer_artifact_filename(name).is_some() {
                listed_layer_artifacts += 1;
            }
            if !name.ends_with(".ext-refs.json") {
                continue;
            }
            let bytes = match tokio::fs::read(entry.path()).await {
                Ok(b) => b,
                // A sidecar the listdir just named that now reads absent is an
                // inconsistency, never an empty sidecar (b.oya). Skipping it
                // would drop its hashes from the live set, and the collect
                // phase deletes every tombstoned blob that set does not name —
                // an over-delete. The typed `ListedSidecarAbsent` `Transient`,
                // naming the sidecar, never a partial set: the collect phase
                // deletes nothing that pass, and decides by who holds the
                // checkpoint lease whether the abort counts as a live-set
                // failure — see `checkpointer::run_reconcile_collect`.
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    return Err(CheckpointError::listed_sidecar_absent(
                        ListedSidecarAbsent {
                            artifact: entry.path().display().to_string(),
                        },
                    ));
                }
                Err(err) => return Err(io_to_permanent(err)),
            };
            // Per SR-1.5 the sidecar is opaque at put time; parsing happens
            // here on demand. A malformed sidecar surfaces as `Permanent`
            // (it cannot be retried into success), matching the InMemory
            // adapter so the conformance harness sees consistent
            // classification.
            let parsed: ExtRefsSidecar =
                serde_json::from_slice(&bytes).map_err(|err| CheckpointError::Permanent {
                    source: Box::new(err),
                })?;
            hashes.extend(parsed.ext_hashes);
        }
        Ok(LiveExtSet {
            hashes,
            listed_layer_artifacts,
        })
    }

    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.maybe_inject("read_ext_refs_sidecar")?;
        // PERMANENT, not interim (SR-2.4, b.s5y MUST-level): a tail carries no
        // ext-refs sidecar by design, so the empty set is the final answer and
        // Tasks 3/4 must not "complete" this arm. Short-circuited before
        // `sidecar_path`, which is kind-blind — a tail handle falling through
        // would return the BASE generation's hashes as if they were the tail's.
        if handle.kind == CheckpointKind::Tail {
            return Ok(HashSet::new());
        }
        // Address the sidecar through `(object_id, snap_seq)` — the
        // adapter-internal `sidecar_path` helper builds the SR-4.6
        // `<layers>/<oid>.<snap_seq>.ext-refs.json` path. `adapter_key` is
        // an opaque carrier on the trait surface and we deliberately ignore
        // it here so the cascade caller can synthesize handles from
        // `list_layer_checkpoints` without round-tripping the adapter_key.
        let path = self.sidecar_path(handle.object_id, handle.snap_seq);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            // Absent sidecar is `Ok(empty)` per the trait contract: a legacy
            // generation that pre-dates sidecar emission, or one put with no
            // sidecar, legitimately has none, and this keeps the
            // retention-prune cascade from over-classifying absence as
            // failure. `list_referenced_ext_hashes` differs because it reads
            // only sidecars its own listdir named, so absence there is an
            // inconsistency and an error.
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(HashSet::new()),
            Err(err) => return Err(io_to_permanent(err)),
        };
        // Malformed sidecar JSON classifies as `Permanent` per the trait
        // contract — durable bytes will not parse on retry, so surface the
        // error instead of silently dropping references for that handle.
        let parsed: ExtRefsSidecar =
            serde_json::from_slice(&bytes).map_err(|err| CheckpointError::Permanent {
                source: Box::new(err),
            })?;
        Ok(parsed.ext_hashes.into_iter().collect())
    }

    /// Raw `layers/` directory walk, unfiltered (b.9qd).
    ///
    /// The same `read_dir` the generation listings use, but widened past the
    /// generation artifacts to every artifact shape in the namespace, **both
    /// families**: the base stem's `.ompf` / `.meta.json` / `.ext-refs.json`,
    /// the tail stem's `.tail` / `.tail.meta.json`, and the
    /// `<final-name>.<uuid>.tmp` form [`tmp_path_for`] produces over either.
    /// Names that parse as none of those are skipped silently: an operator's
    /// stray file in the namespace is not this pass's business, and
    /// mis-classifying one would put it on a deletion path.
    ///
    /// The walk needs no kind branch of its own — [`parse_layer_artifact_filename`]
    /// carries the whole vocabulary, and the caller separates the families by
    /// [`LayerArtifactKind::generation_kind`], which is what keeps a base's
    /// `.ompf` from vouching for the tail stem sharing its sequence. A torn tail
    /// publish's residual `.tail.meta.json` is therefore a Tail-family orphan
    /// candidate here, visible to the reconciliation pass on the same terms as a
    /// base stem's orphaned companion.
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.maybe_inject("list_layer_artifacts")?;
        let mut reader = match tokio::fs::read_dir(&self.layers_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut artifacts: Vec<LayerArtifact> = Vec::new();
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if let Some(artifact) = parse_layer_artifact_filename(name) {
                artifacts.push(artifact);
            }
        }
        Ok(artifacts)
    }

    /// Remove every `<oid>.<snap_seq>.*.<uuid>.tmp` leftover for one stem
    /// (b.9qd).
    ///
    /// **Kind-agnostic, by contract.** The `(object_id, snap_seq)` argument pair
    /// names a numeric stem, not a generation family: one call reclaims the
    /// leaked temps of *both* families at that stem — `<oid>.<seq>.ompf.*.tmp`,
    /// `<oid>.<seq>.tail.*.tmp`, `<oid>.<seq>.tail.meta.json.*.tmp` and the rest
    /// — because [`LayerArtifactKind::Temp`] is deliberately a single
    /// family-less kind and the shared parser folds every temp shape onto it. A
    /// temp is never a live artifact of either family, so reclaiming across both
    /// can never take something a generation still needs.
    ///
    /// A leaked temp survives a hard kill, or a caller dropped mid-write,
    /// between `tokio::fs::write` and the rename — [`Self::atomic_write`]
    /// removes its own temp on every error path it can observe, but the rename
    /// and the removal run in its caller's future, which a drop ends. It is inert (no reader ever names it) but it is
    /// bytes, and it is the one artifact class that cannot be addressed from
    /// `(object_id, snap_seq)` alone, which is why this method walks the
    /// directory instead of taking a path.
    ///
    /// Deliberately NOT folded into `delete_layer_checkpoint`: a temp can be
    /// garbage while its stem's generation is perfectly live, and the caller
    /// that reclaims it must not be able to reach for a generation delete by
    /// accident.
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.maybe_inject("delete_layer_temp_artifacts")?;
        let mut reader = match tokio::fs::read_dir(&self.layers_dir).await {
            Ok(r) => r,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(err) => return Err(io_to_permanent(err)),
        };
        let mut removed = 0usize;
        loop {
            let entry = reader.next_entry().await.map_err(io_to_permanent)?;
            let Some(entry) = entry else { break };
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            let Some(artifact) = parse_layer_artifact_filename(name) else {
                continue;
            };
            if artifact.kind != LayerArtifactKind::Temp
                || artifact.object_id != object_id
                || artifact.snap_seq != snap_seq
            {
                continue;
            }
            match tokio::fs::remove_file(entry.path()).await {
                Ok(()) => removed += 1,
                // Raced with another sweeper or with `atomic_write`'s own
                // cleanup — idempotent on missing per the trait contract.
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(io_to_permanent(err)),
            }
        }
        Ok(removed)
    }
}

// Silence unused-field warnings on `namespace_root` — it's retained on the
// struct as a debugging aid (so `Debug` impls and operator log spew can refer
// to the configured root without re-derivation) and is documented as part of
// the type's public-facing shape even though no method reads it directly.
impl LocalDiskCheckpointStore {
    /// Returns the `<root>/.live-edit/` namespace root. Intended for operator
    /// diagnostics and tests; not part of the trait surface.
    #[allow(dead_code)]
    pub(crate) fn namespace_root(&self) -> &Path {
        &self.namespace_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_meta() -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(1),
            // `identity_default`, not `CheckpointMeta::default`: a stored
            // generation MUST carry identity or `validate_generation_identity`
            // refuses the put.
            ..crate::checkpoint::adapters::identity_default()
        }
    }

    /// Build a sidecar JSON body that `list_referenced_ext_hashes` will parse
    /// as the supplied set of ext-hashes.
    fn sidecar_for(hashes: &[&str]) -> Bytes {
        let json = serde_json::json!({ "ext_hashes": hashes });
        Bytes::from(serde_json::to_vec(&json).expect("sidecar JSON encodes"))
    }

    #[tokio::test]
    async fn new_creates_namespace_subtree_under_fresh_root() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf())
            .expect("constructor must succeed on a fresh empty root");

        // Constructor MUST have eagerly created the `.live-edit/{layers,
        // sessions,ext}` subtree so subsequent writes never race the parent-
        // directory creation.
        let ns = root.path().join(NAMESPACE_DIR);
        assert!(ns.is_dir(), "namespace root must exist");
        assert!(ns.join(LAYERS_SUBDIR).is_dir(), "layers/ must exist");
        assert!(ns.join(SESSIONS_SUBDIR).is_dir(), "sessions/ must exist");
        assert!(ns.join(EXT_SUBDIR).is_dir(), "ext/ must exist");

        // Counters sidecar is created lazily on first put — verify it does
        // NOT exist on a fresh adapter.
        assert!(
            !ns.join(COUNTERS_FILE).exists(),
            "counters.json must not exist until first put_counters_sidecar"
        );

        // namespace_root() reports the same path used internally.
        assert_eq!(store.namespace_root(), ns.as_path());
    }

    #[tokio::test]
    async fn put_layer_checkpoint_writes_ompf_to_sr_4_6_path() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 42;
        let snap_seq: u64 = 7;
        let payload = Bytes::from_static(b"opaque-ompf-bytes");
        let handle = store
            .put_layer_checkpoint(oid, snap_seq, payload.clone(), test_meta())
            .await
            .expect("put_layer_checkpoint succeeds");

        // SDD SR-4.6 — `<root>/.live-edit/layers/<oid>.<snap_seq>.ompf`.
        let expected = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("42.7.ompf");
        assert!(expected.is_file(), "OMPF must land at SR-4.6 path");
        let written = tokio::fs::read(&expected).await.expect("ompf readable");
        assert_eq!(written, payload.as_ref());
        // Handle must reflect identity arguments + path-shaped adapter_key.
        assert_eq!(handle.object_id, oid);
        assert_eq!(handle.snap_seq, snap_seq);
        assert_eq!(handle.adapter_key, expected.to_string_lossy());
    }

    /// b.s5y — `put_layer_checkpoint` must publish the whole triplet, and the
    /// ext-refs sidecar must be one of them (it is what protects the
    /// generation's ext blobs from the retention cascade). The ordering itself
    /// is pinned by the failure-window test below; this one pins that the
    /// happy path leaves all three artifacts behind.
    #[tokio::test]
    async fn put_layer_checkpoint_publishes_the_whole_triplet() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let mut meta = test_meta();
        meta.sidecar_bytes = Some(sidecar_for(&["aa", "bb"]));
        store
            .put_layer_checkpoint(11, 4, Bytes::from_static(b"ompf-bytes"), meta)
            .await
            .expect("put_layer_checkpoint succeeds");

        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        for name in ["11.4.ext-refs.json", "11.4.ompf", "11.4.meta.json"] {
            assert!(
                layers.join(name).is_file(),
                "SR-4.6 artifact `{name}` must exist after a successful put"
            );
        }
    }

    /// b.s5y R2 — a generation that will NOT become durable must leave no
    /// ext-refs sidecar behind: when the OMPF write fails, the sidecar this
    /// put already published is rolled back.
    ///
    /// # Why the residual is not benign
    ///
    /// An orphaned `<oid>.<snap>.ext-refs.json` is **unreapable**. Every
    /// deletion path in this adapter is generation-stem-driven
    /// ([`LocalDiskCheckpointStore::list_layer_checkpoints`] enumerates the
    /// generation stems — `*.ompf` for a base, `*.tail` for a tail — never a
    /// companion), so a sidecar whose OMPF never landed is never offered to
    /// `delete_layer_checkpoint` — not by the retention prune, not by
    /// anything. Widening that listing to tails does not widen the reaper
    /// here: a tail owns no sidecar to begin with. And it is not inert: `list_referenced_ext_hashes` enumerates
    /// sidecars *directly*, so the orphan pins that snapshot's entire ext hash
    /// set in the retention cascade's live set forever, and `pruned − live`
    /// degrades toward the empty set. The cascade quietly stops reclaiming
    /// bytes, fleet-wide, with no failing operation to point at.
    ///
    /// Note this is NOT self-healing by re-put: a retry only overwrites the
    /// orphan if it uses the *identical* `snap_seq`, which stops being true as
    /// soon as the layer takes another commit.
    ///
    /// # How the window is opened without a fault-injection seam
    ///
    /// [`CheckpointFaultInjector`] only fires at the TOP of
    /// `put_layer_checkpoint`, so there is no seam *between* the three writes.
    /// Instead the OMPF's final path is pre-created as a **directory**:
    /// `atomic_write` writes its temp file fine and then `rename(2)` fails with
    /// `EISDIR`, aborting the put exactly at the meta → OMPF boundary. That
    /// is a genuine mid-put failure, not a simulated one.
    ///
    /// # The probe's `is_file()` check (b.9qd)
    ///
    /// Since b.9qd the rollback is *guarded*: it is skipped when an OMPF is
    /// already durable at this `(object_id, snap_seq)`. This fixture is also
    /// the guard's negative case, and a pointed one — the poisoned path is a
    /// **directory** at the OMPF path, which `metadata()` reports as existing.
    /// Only the `is_file()` half of the probe tells that apart from a real
    /// generation. If the guard tested mere existence, the rollback would be
    /// suppressed here and the companions would be stranded permanently, so
    /// the two "must be rolled back" assertions below are simultaneously the
    /// b.s5y R2 regression and the b.9qd probe-precision regression.
    ///
    /// # What this does NOT claim
    ///
    /// It does not pin the publish *order*; a rolled-back companion is
    /// invisible either way. The two sibling tests below pin the order by
    /// aiming the same `EISDIR` technique one artifact earlier each time.
    /// The ordering *rationale* — the sidecar must be visible no later than
    /// the OMPF whose blobs it protects, so a concurrent cascade's live set
    /// can see an in-flight generation — is pinned end-to-end by
    /// `checkpointer::cascade_inflight_generation_tests`.
    #[tokio::test]
    async fn put_layer_checkpoint_rolls_back_the_sidecar_when_the_ompf_write_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        // Poison the OMPF path so the middle write of the triplet fails.
        tokio::fs::create_dir(layers.join("77.2.ompf"))
            .await
            .expect("pre-creating a directory at the OMPF path must succeed");

        let mut meta = test_meta();
        meta.snap_seq = Some(2);
        meta.sidecar_bytes = Some(sidecar_for(&["deadbeef"]));
        let err = store
            .put_layer_checkpoint(77, 2, Bytes::from_static(b"ompf-bytes"), meta)
            .await
            .expect_err("the OMPF write must fail when its path is an existing directory");
        assert!(
            err.is_permanent(),
            "a rename onto a directory is not retryable into success; expected Permanent, \
             got {err:?}"
        );

        assert!(
            !layers.join("77.2.ext-refs.json").exists(),
            "b.s5y R2: the sidecar published earlier in this put must be rolled back once the \
             OMPF write fails. Left behind it is UNREAPABLE (every delete path here is \
             generation-stem-driven — a companion is only ever deleted alongside the \
             generation that owns it) and it pins this snapshot's whole ext set in the \
             cascade's live set forever"
        );
        assert!(
            !layers.join("77.2.meta.json").exists(),
            "b.9qd: meta is published BEFORE the OMPF (so a durable OMPF always carries its \
             identity), which means an OMPF failure strands it too and the rollback must take \
             it as well. A meta here means either the rollback skipped it or the probe let a \
             DIRECTORY at the OMPF path pass for a durable generation"
        );
        assert!(
            !layers.join("77.2.ompf").is_file(),
            "the OMPF write is the one that failed; the poisoned path must still be a \
             directory, not a file"
        );
        // The observable consequence, and the reason the rollback exists: the
        // failed generation contributes nothing to the retention cascade's
        // live set, so it cannot pin `deadbeef` against collection.
        let live = store
            .list_referenced_ext_hashes()
            .await
            .expect("list_referenced_ext_hashes must succeed");
        assert!(
            !live.contains("deadbeef"),
            "a generation that failed to publish must leave NO trace in the live set; got \
             {live:?}"
        );

        // No stray temp files either — `atomic_write` cleans up after itself,
        // and a leaked `.tmp` would be a second unreapable residual.
        let mut entries = tokio::fs::read_dir(&layers)
            .await
            .expect("layers dir readable");
        let mut names: Vec<String> = Vec::new();
        while let Some(entry) = entries.next_entry().await.expect("dir entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(
            names,
            vec!["77.2.ompf".to_string()],
            "the only thing left in the layers dir must be the poisoned path the test itself \
             created"
        );
    }

    /// b.s5y R2 — the ext-refs sidecar is the FIRST artifact
    /// `put_layer_checkpoint` attempts, so a failure there publishes nothing
    /// at all.
    ///
    /// This is what remains assertable about publish *order* now that a failed
    /// OMPF write rolls the sidecar back: the two failure points are told apart
    /// by which artifacts exist afterwards, and only a sidecar-first
    /// implementation leaves the OMPF unwritten when the sidecar is the thing
    /// that fails. Under the pre-b.s5y order (OMPF → ext-refs → meta) the OMPF
    /// would be on disk here.
    ///
    /// Same `EISDIR` technique as above, aimed one artifact earlier.
    #[tokio::test]
    async fn put_layer_checkpoint_aborts_before_any_artifact_when_the_sidecar_write_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        tokio::fs::create_dir(layers.join("77.2.ext-refs.json"))
            .await
            .expect("pre-creating a directory at the sidecar path must succeed");

        let mut meta = test_meta();
        meta.snap_seq = Some(2);
        meta.sidecar_bytes = Some(sidecar_for(&["deadbeef"]));
        let err = store
            .put_layer_checkpoint(77, 2, Bytes::from_static(b"ompf-bytes"), meta)
            .await
            .expect_err("the sidecar write must fail when its path is an existing directory");
        assert!(err.is_permanent(), "expected Permanent, got {err:?}");

        assert!(
            !layers.join("77.2.ompf").exists(),
            "the sidecar is attempted FIRST, so its failure must abort the put before the \
             OMPF is written. An OMPF here means the adapter reverted to OMPF-first publish, \
             which reopens the window where a concurrent retention cascade cannot see an \
             in-flight generation's ext-refs"
        );
        assert!(
            !layers.join("77.2.meta.json").exists(),
            "and certainly before the meta companion"
        );
    }

    /// b.9qd — the meta companion is published BEFORE the OMPF, so a failure
    /// at the meta write must abort the put with no OMPF on disk.
    ///
    /// This is the second of the two order assertions. Together with
    /// `put_layer_checkpoint_aborts_before_any_artifact_when_the_sidecar_write_fails`
    /// above they pin the full publish order — ext-refs → meta → OMPF — by
    /// aiming the same `EISDIR` technique one artifact earlier each time:
    /// a sidecar failure leaves neither companion nor OMPF, and a meta failure
    /// leaves no OMPF.
    ///
    /// Why meta must precede the OMPF: a durable OMPF whose meta never landed
    /// is listed as a generation but refused on read (`MissingMetaCompanion`),
    /// so the restore walk must step down past it; before that refusal existed
    /// it was served with a synthesized, identity-empty meta that shadowed the
    /// last good generation. Publishing meta first keeps every listed
    /// generation carrying its identity.
    #[tokio::test]
    async fn put_layer_checkpoint_aborts_before_the_ompf_when_the_meta_write_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        tokio::fs::create_dir(layers.join("77.2.meta.json"))
            .await
            .expect("pre-creating a directory at the meta path must succeed");

        let mut meta = test_meta();
        meta.snap_seq = Some(2);
        meta.sidecar_bytes = Some(sidecar_for(&["deadbeef"]));
        let err = store
            .put_layer_checkpoint(77, 2, Bytes::from_static(b"ompf-bytes"), meta)
            .await
            .expect_err("the meta write must fail when its path is an existing directory");
        assert!(err.is_permanent(), "expected Permanent, got {err:?}");

        assert!(
            !layers.join("77.2.ompf").exists(),
            "the meta companion is published BEFORE the OMPF, so its failure must abort the \
             put before the OMPF is written. An OMPF here means the adapter reverted to \
             publishing meta last, which lets a durable OMPF shadow the last good generation \
             and restore it with empty session identity"
        );
        assert!(
            !layers.join("77.2.ext-refs.json").exists(),
            "and the sidecar written earlier in this put must be rolled back — no OMPF is \
             durable at this stem, so the guard does not fire"
        );
    }

    /// b.9qd — **the over-delete guard**. When an OMPF is already durable at
    /// this `(object_id, snap_seq)`, a failed re-put must NOT roll back the
    /// companions, because they belong to the generation that OMPF makes live.
    ///
    /// The path this protects is the SDD §22.7 step-8 re-put: when the cursor
    /// HSET fails, the next tick re-assembles the *same* layer at the *same*
    /// `snap_seq`. An unguarded rollback would then delete the ext-refs and
    /// meta of a generation whose OMPF is durable from the previous attempt,
    /// leaving a live, enumerable generation stripped of the sidecar that
    /// protects its ext blobs from the retention cascade — the over-delete
    /// direction, produced by the very cleanup meant to prevent it.
    ///
    /// The failure is opened at the meta write (a directory at the meta path),
    /// which is the only step that can fail *after* the sidecar is written
    /// while a durable OMPF is already present.
    #[tokio::test]
    async fn put_layer_checkpoint_rollback_is_skipped_when_a_durable_ompf_already_exists() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        // Attempt 1: a complete, durable generation.
        let mut meta = test_meta();
        meta.snap_seq = Some(5);
        meta.sidecar_bytes = Some(sidecar_for(&["cafebabe"]));
        store
            .put_layer_checkpoint(88, 5, Bytes::from_static(b"ompf-bytes"), meta)
            .await
            .expect("the first put must succeed");
        for name in ["88.5.ompf", "88.5.meta.json", "88.5.ext-refs.json"] {
            assert!(layers.join(name).is_file(), "setup: `{name}` must exist");
        }

        // Poison the meta path so attempt 2 fails after re-writing the sidecar.
        tokio::fs::remove_file(layers.join("88.5.meta.json"))
            .await
            .expect("removing the meta file must succeed");
        tokio::fs::create_dir(layers.join("88.5.meta.json"))
            .await
            .expect("pre-creating a directory at the meta path must succeed");

        // Attempt 2: the SAME snap_seq — the re-put shape.
        let mut meta2 = test_meta();
        meta2.snap_seq = Some(5);
        meta2.sidecar_bytes = Some(sidecar_for(&["cafebabe"]));
        let err = store
            .put_layer_checkpoint(88, 5, Bytes::from_static(b"ompf-bytes"), meta2)
            .await
            .expect_err("the re-put's meta write must fail");
        assert!(err.is_permanent(), "expected Permanent, got {err:?}");

        assert!(
            layers.join("88.5.ext-refs.json").is_file(),
            "THE guard: an OMPF is already durable at (88, 5), so the companions belong to a \
             LIVE generation and must survive the failed re-put. Rolling the sidecar back here \
             would strip an enumerable generation of the only thing protecting its ext blobs \
             from the retention cascade — silent, unrecoverable loss on a DR restore"
        );
        assert!(
            layers.join("88.5.ompf").is_file(),
            "and the durable OMPF itself is never a rollback target"
        );
        let live = store
            .list_referenced_ext_hashes()
            .await
            .expect("list_referenced_ext_hashes must succeed");
        assert!(
            live.contains("cafebabe"),
            "the observable consequence: the live generation still pins its ext hash. Live \
             set: {live:?}"
        );
        assert_eq!(
            store
                .list_layer_checkpoints(88)
                .await
                .expect("list must succeed")
                .len(),
            1,
            "sanity: the generation is still enumerable, which is what makes stripping its \
             sidecar dangerous rather than merely untidy"
        );
    }

    /// b.9qd — `list_layer_artifacts` is the one enumeration on this adapter
    /// that is deliberately unfiltered.
    ///
    /// It must surface exactly the artifacts a generation-stem-driven listing
    /// cannot: the orphan stem (companions with no generation artifact) and the
    /// write-temp leftover.
    /// It must also skip anything that is not a layer artifact — an operator's
    /// stray file in the namespace is not this pass's business, and
    /// mis-classifying one would put it on a deletion path.
    ///
    /// Both generation families are exercised in one directory, because on this
    /// adapter the whole enumeration is filename parsing over one flat walk and
    /// the families are separated by suffix alone. The tail half adds four
    /// shapes the base half cannot reach:
    ///
    /// - a **complete tail** (`5.4.tail` + `5.4.tail.meta.json`), put through the
    ///   trait so the names come from the adapter's own builders;
    /// - a **torn tail meta** at a stem with no envelope (`6.3.tail.meta.json`)
    ///   — the ONLY durable residual a tail publish can strand, since the
    ///   publish is meta-first and a tail carries no ext-refs sidecar (SR-2.4).
    ///   It must enumerate as `TailMeta`, or the reconciliation pass never
    ///   learns it exists and it leaks forever;
    /// - **tail temps of both durable tail shapes** — a `.tail.<uuid>.tmp` and a
    ///   `.tail.meta.json.<uuid>.tmp` — which must fold onto the single
    ///   kind-agnostic `Temp` rather than onto a phantom tail generation;
    /// - **tail-shaped strays** (`5.1.tail.json`, `5.1.tail.meta`, `5.1.2.tail`)
    ///   that resemble the family closely enough to catch a parser matching on
    ///   `.tail` loosely, but that no writer here produces. Admitting one would
    ///   put an operator's file on a deletion path.
    #[tokio::test]
    async fn list_layer_artifacts_surfaces_orphans_and_temps_and_skips_strays() {
        use std::collections::HashSet;

        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        // A complete generation.
        let mut meta = test_meta();
        meta.sidecar_bytes = Some(sidecar_for(&["aa"]));
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"ompf"), meta)
            .await
            .expect("put must succeed");
        // A complete TAIL generation, published through the trait so the two
        // artifact names are the adapter's own builders' output rather than
        // literals this test agreed with itself about.
        store
            .put_layer_checkpoint(5, 4, Bytes::from_static(b"tail"), tail_meta(1, 2, 4))
            .await
            .expect("the tail put must succeed");
        for name in ["5.4.tail", "5.4.tail.meta.json"] {
            assert!(
                layers.join(name).is_file(),
                "setup: the tail put must have published `{name}`"
            );
        }
        // The torn-tail residual: a tail meta whose envelope never landed. The
        // publish is meta-first and a tail has no sidecar, so this is the only
        // durable shape a dead tail publish can leave.
        tokio::fs::write(layers.join("6.3.tail.meta.json"), b"{}".as_slice())
            .await
            .expect("planting the orphaned tail meta must succeed");
        // An orphan: companions with no OMPF.
        tokio::fs::write(layers.join("2.9.ext-refs.json"), sidecar_for(&["bb"]))
            .await
            .expect("planting the orphan sidecar must succeed");
        tokio::fs::write(layers.join("2.9.meta.json"), b"{}".as_slice())
            .await
            .expect("planting the orphan meta must succeed");
        // The two ONE-companion orphan shapes. Under the b.9qd publish order
        // (`ext-refs → meta → OMPF`) the both-companions stem above is the
        // narrowest of the three crash windows; a publish that died after step
        // 1 leaves a bare sidecar, and an OMPF-only publish
        // (`sidecar_bytes: None`) that died before its OMPF leaves a bare meta.
        // Both are just as invisible to every generation-stem-driven
        // enumeration (a companion is never a generation stem), so both
        // have to surface here or the reconciliation pass never learns they
        // exist.
        tokio::fs::write(layers.join("3.7.ext-refs.json"), sidecar_for(&["cc"]))
            .await
            .expect("planting the sidecar-only orphan must succeed");
        tokio::fs::write(layers.join("4.2.meta.json"), b"{}".as_slice())
            .await
            .expect("planting the meta-only orphan must succeed");
        // A leaked write-temp beside the live generation.
        let temp_name = "1.1.ompf.0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9.tmp";
        tokio::fs::write(layers.join(temp_name), b"half".as_slice())
            .await
            .expect("planting the temp must succeed");
        // Leaked temps of BOTH durable tail shapes, beside the live tail. Both
        // must fold onto the single kind-agnostic `Temp` at (5, 4) — a parser
        // that stripped `.tmp` and then admitted the residual as a generation
        // would fabricate a second tail at a stem that already has one.
        for tail_temp in [
            "5.4.tail.4e5f6071-8293-a4b5-c6d7-e8f9a0b1c2d3.tmp",
            "5.4.tail.meta.json.5f607182-93a4-b5c6-d7e8-f9a0b1c2d3e4.tmp",
        ] {
            tokio::fs::write(layers.join(tail_temp), b"half".as_slice())
                .await
                .expect("planting a tail temp must succeed");
        }
        // Strays that must be ignored. The last three are tail-SHAPED: they end
        // in something close to a tail suffix without being one, so a parser
        // that matched `.tail` loosely (or that rebuilt the tail-meta suffix by
        // concatenation instead of testing it whole) would admit them.
        for stray in [
            "readme.txt",
            "not-a-number.9.ompf",
            "1.2.3.ompf",
            // `.tail.json` is not a suffix any writer here produces; only
            // `.tail.meta.json` is.
            "5.1.tail.json",
            // `.tail.meta` without the `.json` likewise.
            "5.1.tail.meta",
            // A three-component stem under the tail suffix — the strict
            // two-component canonical check must reject it exactly as it does
            // `1.2.3.ompf` for the base family.
            "5.1.2.tail",
        ] {
            tokio::fs::write(layers.join(stray), b"junk".as_slice())
                .await
                .expect("planting a stray must succeed");
        }

        let raw = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        let artifacts: HashSet<LayerArtifact> = raw.iter().copied().collect();
        let expected: HashSet<LayerArtifact> = [
            LayerArtifact {
                object_id: 1,
                snap_seq: 1,
                kind: LayerArtifactKind::Ompf,
            },
            LayerArtifact {
                object_id: 1,
                snap_seq: 1,
                kind: LayerArtifactKind::Meta,
            },
            LayerArtifact {
                object_id: 1,
                snap_seq: 1,
                kind: LayerArtifactKind::ExtRefs,
            },
            LayerArtifact {
                object_id: 1,
                snap_seq: 1,
                kind: LayerArtifactKind::Temp,
            },
            LayerArtifact {
                object_id: 2,
                snap_seq: 9,
                kind: LayerArtifactKind::ExtRefs,
            },
            LayerArtifact {
                object_id: 2,
                snap_seq: 9,
                kind: LayerArtifactKind::Meta,
            },
            LayerArtifact {
                object_id: 3,
                snap_seq: 7,
                kind: LayerArtifactKind::ExtRefs,
            },
            LayerArtifact {
                object_id: 4,
                snap_seq: 2,
                kind: LayerArtifactKind::Meta,
            },
            // The complete tail's two members.
            LayerArtifact {
                object_id: 5,
                snap_seq: 4,
                kind: LayerArtifactKind::TailEnvelope,
            },
            LayerArtifact {
                object_id: 5,
                snap_seq: 4,
                kind: LayerArtifactKind::TailMeta,
            },
            // Its two leaked temps fold onto ONE set member — `Temp` carries no
            // family axis, and this is a set, so both `.tmp` files collapse
            // here. The raw-vector assertion below is what keeps that collapse
            // from hiding a walk that only recognised one of the two shapes.
            LayerArtifact {
                object_id: 5,
                snap_seq: 4,
                kind: LayerArtifactKind::Temp,
            },
            // The torn-tail residual, with no `TailEnvelope` beside it: a
            // Tail-family orphan candidate.
            LayerArtifact {
                object_id: 6,
                snap_seq: 3,
                kind: LayerArtifactKind::TailMeta,
            },
        ]
        .into_iter()
        .collect();
        assert_eq!(
            artifacts, expected,
            "the raw enumeration must include every orphan stem of BOTH families — the base \
             stem's both-companions / sidecar-only / meta-only shapes and the tail stem's \
             lone `.tail.meta.json` — plus the temps of both families (none of them is \
             reachable from any generation-stem-driven listing), and exclude every stray, \
             including the tail-shaped ones"
        );

        // Both tail temp SHAPES, counted over the raw vector. The set above
        // cannot make this claim: `Temp` is family-less, so `5.4.tail.*.tmp` and
        // `5.4.tail.meta.json.*.tmp` collapse onto one member and a walk that
        // recognised only one of them would still satisfy it — while leaving the
        // other stranded, since a temp is reachable from no other enumeration.
        assert_eq!(
            raw.iter()
                .filter(|a| a.object_id == 5
                    && a.snap_seq == 4
                    && a.kind == LayerArtifactKind::Temp)
                .count(),
            2,
            "both durable tail shapes' temps must enumerate individually; got {raw:?}"
        );

        // The contrast that makes the point: the generation-stem-driven
        // listings see neither the orphan nor the temp, in either family.
        assert!(
            store
                .list_layer_checkpoints(2)
                .await
                .expect("list must succeed")
                .is_empty(),
            "the orphan stem is invisible to list_layer_checkpoints — that invisibility is \
             exactly why it can never be handed to delete_layer_checkpoint"
        );
        assert!(
            store
                .list_layer_checkpoints(6)
                .await
                .expect("list must succeed")
                .is_empty(),
            "and so is the torn TAIL stem: `6.3.tail.meta.json` is a companion, and a companion \
             never satisfies its stem, so no handle-producing path can reach it. That is the \
             whole reason `list_layer_artifacts` has to see it"
        );
        assert_eq!(
            store
                .list_layer_checkpoints(1)
                .await
                .expect("list must succeed")
                .len(),
            1,
            "and the temp must not fabricate a second generation"
        );
        assert_eq!(
            store
                .list_layer_checkpoints(5)
                .await
                .expect("list must succeed")
                .len(),
            1,
            "the live tail is ONE generation: neither its own `.tail.meta.json` companion nor \
             either of its two leaked `.tmp` files may fabricate a second"
        );
    }

    /// b.9qd — `delete_layer_temp_artifacts` removes every temp for ONE stem
    /// and nothing else, and is idempotent on a stem with no temps.
    ///
    /// The scoping is the contract: a temp is not addressable from
    /// `(object_id, snap_seq)` alone (its name carries a random
    /// disambiguator), so this method walks the directory — and a walk that
    /// matched too loosely would delete another generation's in-flight temp
    /// out from under a concurrent writer.
    ///
    /// The scoping claim spans **both generation families**, because the
    /// `(object_id, snap_seq)` pair names a numeric *stem*, not a family: one
    /// call reclaims `3.1.ompf.*.tmp`, `3.1.tail.*.tmp` and
    /// `3.1.tail.meta.json.*.tmp` alike, since `LayerArtifactKind::Temp` is
    /// deliberately family-less. The fixture therefore plants tail temps both at
    /// the target stem (must go) and at another stem (must stay), so neither
    /// half of the claim is inferred from the base family alone. A base at 1 and
    /// a tail whose `to_seq` is 1 legitimately coexist on this object, which is
    /// exactly why a caller reclaiming one family's temps must not be able to
    /// spare — or over-reach into — the other's.
    #[tokio::test]
    async fn delete_layer_temp_artifacts_removes_only_this_stems_temps() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        store
            .put_layer_checkpoint(3, 1, Bytes::from_static(b"ompf"), test_meta())
            .await
            .expect("put must succeed");
        // A live TAIL sharing the target stem's sequence — the SR-4.1 re-base
        // shape. Its durable artifacts must survive the temp reclaim just as the
        // base's do.
        store
            .put_layer_checkpoint(3, 1, Bytes::from_static(b"tail"), tail_meta(1, 1, 1))
            .await
            .expect("the same-sequence tail put must succeed");
        let mine = [
            "3.1.ompf.0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9.tmp",
            "3.1.meta.json.1b2c3d4e-5f60-7182-93a4-b5c6d7e8f9a0.tmp",
            // Both durable tail shapes at the same stem.
            "3.1.tail.60718293-a4b5-c6d7-e8f9-a0b1c2d3e4f5.tmp",
            "3.1.tail.meta.json.71829304-a4b5-c6d7-e8f9-a0b1c2d3e4f5.tmp",
        ];
        let others = [
            "4.1.ompf.2c3d4e5f-6071-8293-a4b5-c6d7e8f9a0b1.tmp",
            "3.2.ompf.3d4e5f60-7182-93a4-b5c6-d7e8f9a0b1c2.tmp",
            // A tail temp at a DIFFERENT stem: the family-spanning reclaim must
            // still be stem-scoped, or it corrupts another writer's tail
            // publish.
            "3.2.tail.8293a4b5-c6d7-e8f9-a0b1-c2d3e4f5a6b7.tmp",
        ];
        for name in mine.iter().chain(others.iter()) {
            tokio::fs::write(layers.join(name), b"half".as_slice())
                .await
                .expect("planting a temp must succeed");
        }

        let removed = store
            .delete_layer_temp_artifacts(3, 1)
            .await
            .expect("delete_layer_temp_artifacts must succeed");
        assert_eq!(
            removed, 4,
            "all four of this stem's temps must be removed — both families'. The count is the \
             caller's evidence that bytes moved, so a reclaim that silently skipped the tail \
             shapes would report a partial success and re-register the same candidates forever"
        );
        for name in mine {
            assert!(!layers.join(name).exists(), "`{name}` must be gone");
        }
        for name in others {
            assert!(
                layers.join(name).exists(),
                "`{name}` belongs to a different stem and may be a concurrent writer's \
                 in-flight temp — deleting it would corrupt that publish"
            );
        }
        assert!(
            layers.join("3.1.ompf").is_file(),
            "the stem's durable generation must be untouched: a temp can be garbage while its \
             generation is perfectly live"
        );
        for durable in ["3.1.tail", "3.1.tail.meta.json"] {
            assert!(
                layers.join(durable).is_file(),
                "`{durable}` is a durable artifact of the LIVE TAIL sharing this stem. Reaching \
                 it while sweeping temps is the catastrophic mis-match the scoping exists to \
                 prevent, and it is the one a `.tail`-prefix match would hit: every temp name \
                 here starts with a durable name"
            );
        }

        assert_eq!(
            store
                .delete_layer_temp_artifacts(3, 1)
                .await
                .expect("a second call must be Ok, not NotFound"),
            0,
            "idempotent on a stem with no temps, per the trait contract"
        );
    }

    /// A non-canonically-spelled stem (`007.3.ompf`, `+12.3.ompf` and their tail
    /// analogues) enumerates as **nothing at all** — not as a generation, not as
    /// an object, not as an artifact.
    ///
    /// This pins a deliberate behavioural tightening, and the shape of the bug it
    /// closes is why the negative claim is worth a test of its own. This
    /// adapter's four generation parsers used to call `u64::from_str` directly,
    /// which accepts `007` and `+12` and aliases both onto the canonical `12`;
    /// the shared `parse_layer_artifact_stem` behind `list_layer_artifacts` never
    /// did. The two surfaces therefore disagreed, and the disagreement pointed
    /// the wrong way: such a file enumerated as a **live generation** whose
    /// canonical delete path (`<oid>.<seq>.ompf`, spelled canonically) does not
    /// exist, so nothing could ever remove it — while the one enumeration the
    /// reconciliation pass consumes could not see it to classify it. Unreapable,
    /// re-enumerating on every pass, forever.
    ///
    /// Routing all four parsers through the shared strict stem parser makes both
    /// surfaces agree on *exclusion*, which is the resolvable direction: the file
    /// is an operator's stray, and a stray in the namespace is not this pass's
    /// business. Admitting it on either surface would put a name no writer here
    /// produces onto a deletion path.
    ///
    /// The canonical control at object 8 is what keeps the test from passing
    /// because the walk broke outright.
    #[tokio::test]
    async fn non_canonically_spelled_stems_enumerate_as_nothing_on_either_surface() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        // The control: a genuine generation, spelled canonically.
        store
            .put_layer_checkpoint(8, 3, Bytes::from_static(b"ompf"), test_meta())
            .await
            .expect("the canonical put must succeed");

        // Zero-padded and `+`-prefixed spellings of both halves, across both
        // generation families and one companion. `u64::from_str` accepts every
        // one of these; the canonical parser accepts none.
        let aliased = [
            "007.3.ompf",
            "+12.3.ompf",
            "007.5.tail",
            "+12.5.tail",
            "007.3.meta.json",
            "12.+3.ompf",
            "12.007.tail",
        ];
        for name in aliased {
            tokio::fs::write(layers.join(name), b"junk".as_slice())
                .await
                .expect("planting an aliased-spelling file must succeed");
        }

        // 1. No phantom generation, in either family.
        for object_id in [7_u64, 12] {
            assert!(
                store
                    .list_layer_checkpoints(object_id)
                    .await
                    .expect("list_layer_checkpoints must succeed")
                    .is_empty(),
                "object {object_id} owns no canonically-named artifact, so it must report no \
                 generation. A handle minted here would name a file whose canonical path does \
                 not exist — every read and the delete would miss it, and the listing would \
                 keep re-offering it"
            );
        }
        assert!(
            !store
                .list_layer_checkpoints(8)
                .await
                .expect("list_layer_checkpoints must succeed")
                .is_empty(),
            "control: the canonically-spelled generation at object 8 must still enumerate — \
             otherwise the assertions above pass because the walk is broken outright"
        );

        // 2. No phantom object, which is what fans the recovery hook out.
        let object_ids = store
            .list_all_object_ids()
            .await
            .expect("list_all_object_ids must succeed");
        assert_eq!(
            object_ids,
            vec![8_u64],
            "only the canonical object may enumerate: a phantom object id sends the recovery \
             hook at a layer that does not exist. Got {object_ids:?}"
        );

        // 3. And the artifact surface agrees — the actual invariant, since the
        //    two surfaces disagreeing is what made the residual unreapable.
        let artifacts = store
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed");
        for artifact in &artifacts {
            assert_eq!(
                artifact.object_id, 8,
                "the raw artifact enumeration must agree with the generation walk about which \
                 names are ours. {artifact:?} came from an aliased spelling, and a stem the two \
                 surfaces disagree about is exactly the one nothing can resolve. \
                 Got {artifacts:?}"
            );
        }
    }

    #[tokio::test]
    async fn set_layer_sidecar_writes_ext_refs_to_sr_4_6_path() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 5;
        let snap_seq: u64 = 3;
        store
            .put_layer_checkpoint(oid, snap_seq, Bytes::from_static(b"o"), test_meta())
            .await
            .expect("put_layer_checkpoint");
        let sidecar = sidecar_for(&["aa", "bb"]);
        store
            .set_layer_sidecar(oid, snap_seq, sidecar.clone())
            .await
            .expect("set_layer_sidecar");

        let expected = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("5.3.ext-refs.json");
        assert!(expected.is_file(), "sidecar must land at SR-4.6 path");
        let written = tokio::fs::read(&expected).await.expect("sidecar readable");
        assert_eq!(written, sidecar.as_ref());
    }

    #[tokio::test]
    async fn put_ext_blob_writes_to_sr_4_6_path() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let hash = "deadbeefcafef00d";
        let bytes = Bytes::from_static(b"ext-payload");
        store
            .put_ext_blob(hash, bytes.clone())
            .await
            .expect("put_ext_blob");

        let expected = root
            .path()
            .join(NAMESPACE_DIR)
            .join(EXT_SUBDIR)
            .join("deadbeefcafef00d.blob");
        assert!(expected.is_file(), "ext blob must land at SR-4.6 path");
        let written = tokio::fs::read(&expected).await.expect("blob readable");
        assert_eq!(written, bytes.as_ref());
    }

    #[tokio::test]
    async fn put_counters_sidecar_writes_to_sr_4_6_path() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let body = Bytes::from_static(b"{\"object_id\":42}");
        store
            .put_counters_sidecar(body.clone())
            .await
            .expect("put_counters_sidecar");

        let expected = root.path().join(NAMESPACE_DIR).join(COUNTERS_FILE);
        assert!(expected.is_file(), "counters.json must land at SR-4.6 path");
        let written = tokio::fs::read(&expected).await.expect("counters readable");
        assert_eq!(written, body.as_ref());
    }

    #[tokio::test]
    async fn put_layer_checkpoint_same_key_twice_retains_one_file() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 11;
        let seq: u64 = 4;
        store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"first"), test_meta())
            .await
            .expect("first put");
        store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"second"), test_meta())
            .await
            .expect("second put");

        // Filesystem introspection: exactly one OMPF file with the
        // `<oid>.<seq>.ompf` name remains in the layers dir.
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        let mut ompf_files: Vec<String> = Vec::new();
        let mut reader = tokio::fs::read_dir(&layers_dir).await.expect("read_dir");
        while let Some(entry) = reader.next_entry().await.expect("next_entry") {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".ompf") {
                ompf_files.push(name);
            }
        }
        assert_eq!(
            ompf_files,
            vec!["11.4.ompf"],
            "duplicate (object_id, snap_seq) puts must collapse to one OMPF file"
        );

        // Last-writer-wins on bytes — the rename-into-place is atomic.
        let written = tokio::fs::read(layers_dir.join("11.4.ompf"))
            .await
            .expect("readable");
        assert_eq!(written, b"second");
    }

    #[tokio::test]
    async fn put_ext_blob_same_hash_twice_does_not_rewrite_file() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let hash = "abc123";
        store
            .put_ext_blob(hash, Bytes::from_static(b"payload"))
            .await
            .expect("first put");

        let path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(EXT_SUBDIR)
            .join("abc123.blob");
        let mtime_before = tokio::fs::metadata(&path)
            .await
            .expect("metadata")
            .modified()
            .expect("modified time");

        // Sleep briefly so a hypothetical rewrite would produce a distinct
        // mtime — most filesystems have at least ms granularity. We do not
        // depend on this sleep for synchronization; it's just a contrast
        // window for the mtime check below. Bounded loop with millisecond
        // polling would be overkill for what is effectively a "rule out
        // sub-ms coincidence" guard.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Content-addressed idempotency contract — second put MUST short-circuit
        // via metadata.is_ok() and skip the rename. We pass distinct bytes to
        // prove the adapter does not even open the file to write.
        store
            .put_ext_blob(hash, Bytes::from_static(b"different-bytes-same-hash"))
            .await
            .expect("second put short-circuits");

        let mtime_after = tokio::fs::metadata(&path)
            .await
            .expect("metadata")
            .modified()
            .expect("modified time");
        assert_eq!(
            mtime_before, mtime_after,
            "second put_ext_blob with same hash must not rewrite the file"
        );
        // And the bytes must be the ORIGINAL — proving the second call was
        // a true no-op, not an overwrite-with-different-content.
        let stored = tokio::fs::read(&path).await.expect("readable");
        assert_eq!(stored, b"payload");
    }

    /// An ext blob write whose caller stops waiting mid-write (b.wsx): the
    /// blocking task that owns the bytes still ends in the rename, and the
    /// ext-mirror charge riding in those bytes lasts exactly as long as the
    /// write.
    ///
    /// **How a write is stopped mid-write.** The test polls `put_ext_blob` by
    /// hand, looking under `ext/` after each poll, and drops the call at the
    /// first poll after which the write's temp is there. The value is large
    /// enough that the write is nearly always still running when its temp is
    /// seen; an attempt whose call finishes before its temp is seen stopped
    /// nothing, so the test tries again in a fresh directory, up to
    /// [`MID_WRITE_ATTEMPTS`] times.
    mod put_ext_blob_mid_write {
        use std::time::{Duration, Instant};

        use super::*;
        use crate::checkpoint::ext_mirror::{ext_mirror_channel, item_charge_kib};

        /// The value written: tens of MiB, so the blocking write is still
        /// running when the test sees its temp.
        const MID_WRITE_LEN: usize = 32 << 20;

        /// The writes a test starts before it gives up on stopping one
        /// mid-write.
        const MID_WRITE_ATTEMPTS: usize = 20;

        /// How long any wait for the write to land, or its charge to return,
        /// may take before the test fails.
        const SETTLE_BUDGET: Duration = Duration::from_secs(10);

        const HASH: &str = "5eed0000000000000000000000000000000000000000000000000000000000ab";

        /// The value, every byte position told apart from its neighbours so
        /// a short or shifted file does not compare equal.
        fn large_value() -> Vec<u8> {
            (0..MID_WRITE_LEN).map(|i| (i % 251) as u8).collect()
        }

        /// One LocalDisk store in its own directory.
        struct Disk {
            _root: tempfile::TempDir,
            store: LocalDiskCheckpointStore,
            ext_dir: PathBuf,
            blob: PathBuf,
        }

        impl Disk {
            fn new() -> Self {
                let root = tempdir().expect("tempdir");
                let store =
                    LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
                let ext_dir = root.path().join(NAMESPACE_DIR).join(EXT_SUBDIR);
                let blob = ext_dir.join(format!("{HASH}.blob"));
                Self {
                    _root: root,
                    store,
                    ext_dir,
                    blob,
                }
            }

            /// The names of the `*.tmp` files under `ext/`.
            fn temps(&self) -> Vec<String> {
                std::fs::read_dir(&self.ext_dir)
                    .expect("read_dir ext/")
                    .map(|entry| {
                        entry
                            .expect("an ext/ entry")
                            .file_name()
                            .to_string_lossy()
                            .into_owned()
                    })
                    .filter(|name| name.ends_with(".tmp"))
                    .collect()
            }

            /// Start `put_ext_blob(HASH, bytes)` and drop the call once its
            /// temp is seen under `ext/`. `false` when the call finished
            /// before its temp was seen: nothing was stopped mid-write.
            async fn drop_caller_mid_write(&self, bytes: Bytes) -> bool {
                let mut put = Box::pin(self.store.put_ext_blob(HASH, bytes));
                let deadline = Instant::now() + SETTLE_BUDGET;
                loop {
                    // Polled by hand, so the call runs exactly as far as a
                    // caller that waits on it, and is dropped at a poll's end.
                    if let std::task::Poll::Ready(result) = futures::poll!(&mut put) {
                        result.expect("the write succeeds");
                        return false;
                    }
                    if !self.temps().is_empty() {
                        break;
                    }
                    assert!(Instant::now() < deadline, "the write never started");
                    // Back to the runtime, which refills the task's
                    // cooperative budget; a poll on an exhausted budget
                    // answers `Pending` without making progress.
                    tokio::task::yield_now().await;
                }
                drop(put);
                true
            }

            /// Wait until `ext/<HASH>.blob` exists, within [`SETTLE_BUDGET`].
            async fn wait_landed(&self) {
                let deadline = Instant::now() + SETTLE_BUDGET;
                while !self.blob.exists() {
                    assert!(
                        Instant::now() < deadline,
                        "the write its caller stopped waiting for never landed; ext/ temps: {:?}",
                        self.temps()
                    );
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }

        /// T24 (b.wsx): a write whose caller is dropped mid-write still lands:
        /// `ext/<hash>.blob` holds every byte, and no `*.tmp` is left under
        /// `ext/`. A write whose rename and temp removal ran in the dropped
        /// caller's future would leave the temp and no blob.
        #[tokio::test]
        async fn a_write_whose_caller_is_dropped_mid_write_lands_whole_and_leaves_no_temp() {
            let value = large_value();
            for _ in 0..MID_WRITE_ATTEMPTS {
                let disk = Disk::new();
                if !disk.drop_caller_mid_write(Bytes::from(value.clone())).await {
                    continue;
                }

                disk.wait_landed().await;
                let landed = std::fs::read(&disk.blob).expect("the blob is readable");
                assert!(
                    landed == value,
                    "the blob holds the whole value: {} of {} bytes, equal = false",
                    landed.len(),
                    value.len()
                );
                assert_eq!(disk.temps(), Vec::<String>::new(), "no temp is left");
                return;
            }
            panic!("no write of {MID_WRITE_ATTEMPTS} was still running when its temp was seen");
        }

        /// T25 (b.wsx): the ext-mirror charge riding in the bytes is held
        /// while a write whose caller was dropped still runs, and comes back
        /// whole once the write has landed. The budget is read first and the
        /// temp checked after: a temp still there proves the read came before
        /// the rename, so the write was still running when the charge was
        /// read. A store that wrote from a copy would give the charge back
        /// with the caller's bytes while the copy was still being written.
        #[tokio::test]
        async fn a_write_whose_caller_is_dropped_holds_the_mirror_charge_until_it_lands() {
            let value = large_value();
            let len = value.len() as u64;
            let charge = item_charge_kib(len) * 1024;
            for _ in 0..MID_WRITE_ATTEMPTS {
                let disk = Disk::new();
                let metrics = std::sync::Arc::new(
                    crate::observability::metrics::metrics_scrape_only("local-disk-mid-write"),
                );
                let budget = std::num::NonZeroU64::new(2 * charge).expect("non-zero");
                let (sender, _rx) = ext_mirror_channel(4, budget, metrics);
                let reservation = sender.try_reserve(len).expect("the budget has room");
                let bytes = reservation.wrap(Bytes::from(value.clone()));
                assert_eq!(
                    sender.available_budget_bytes(),
                    sender.budget_bytes() - charge,
                    "fixture: the bytes carry the charge"
                );

                if !disk.drop_caller_mid_write(bytes).await {
                    continue;
                }
                let available = sender.available_budget_bytes();
                if disk.temps().is_empty() {
                    // The write landed between the drop and the read: the
                    // read may have come after the rename.
                    continue;
                }
                assert_eq!(
                    available,
                    sender.budget_bytes() - charge,
                    "the charge is held while the write its caller stopped waiting for runs"
                );

                disk.wait_landed().await;
                let deadline = Instant::now() + SETTLE_BUDGET;
                while sender.available_budget_bytes() != sender.budget_bytes() {
                    assert!(
                        Instant::now() < deadline,
                        "the charge never came back after the write landed: {} of {} bytes \
                         available",
                        sender.available_budget_bytes(),
                        sender.budget_bytes()
                    );
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                return;
            }
            panic!("no write of {MID_WRITE_ATTEMPTS} was still running when its charge was read");
        }
    }

    #[tokio::test]
    async fn list_layer_checkpoints_returns_snap_seqs_newest_first() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 200;

        // Insert in deliberately scrambled order — proves ordering does not
        // depend on insertion sequence or filesystem `read_dir` order.
        for seq in [2_u64, 4, 1, 3] {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from_static(b"x"), test_meta())
                .await
                .expect("put");
        }
        // Drop a sibling file from a DIFFERENT object_id into the same
        // directory — list_layer_checkpoints must filter to the
        // `<oid>.*.ompf` family only.
        store
            .put_layer_checkpoint(999, 99, Bytes::from_static(b"other"), test_meta())
            .await
            .expect("put other oid");
        // And a sidecar file for our oid — the listing must skip it
        // (only `.ompf` family rows count).
        store
            .set_layer_sidecar(oid, 1, sidecar_for(&["aa"]))
            .await
            .expect("sidecar");

        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        let seqs: Vec<u64> = handles.iter().map(|h| h.snap_seq).collect();
        assert_eq!(
            seqs,
            vec![4, 3, 2, 1],
            "list_layer_checkpoints must return generations newest-first by snap_seq, filtered to the requested oid"
        );
        for h in &handles {
            assert_eq!(
                h.object_id, oid,
                "every handle must carry the requested oid"
            );
        }
    }

    #[tokio::test]
    async fn list_referenced_ext_hashes_unions_across_objects() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Object 1 references {"aa", "bb"}.
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o1"), test_meta())
            .await
            .expect("put 1");
        store
            .set_layer_sidecar(1, 1, sidecar_for(&["aa", "bb"]))
            .await
            .expect("sidecar 1");

        // Object 2 references {"bb", "cc"} — overlap on "bb".
        store
            .put_layer_checkpoint(2, 1, Bytes::from_static(b"o2"), test_meta())
            .await
            .expect("put 2");
        store
            .set_layer_sidecar(2, 1, sidecar_for(&["bb", "cc"]))
            .await
            .expect("sidecar 2");

        let refs = store
            .list_referenced_ext_hashes()
            .await
            .expect("list_referenced_ext_hashes");
        let expected: HashSet<String> = ["aa", "bb", "cc"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "list_referenced_ext_hashes must equal the union with no duplicates"
        );
    }

    #[tokio::test]
    async fn list_referenced_ext_hashes_malformed_sidecar_returns_permanent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Drop a malformed (non-JSON) sidecar straight onto the filesystem —
        // bypassing the adapter so we control the bytes exactly. The filename
        // must match the `*.ext-refs.json` suffix so the listing picks it up.
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        let bad = layers_dir.join("7.1.ext-refs.json");
        tokio::fs::write(&bad, b"not-json-at-all{{{")
            .await
            .expect("write malformed sidecar");

        let err = store
            .list_referenced_ext_hashes()
            .await
            .expect_err("malformed sidecar must surface an error, not a panic or silent skip");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "malformed sidecar must classify as Permanent (cannot retry into success); got {:?}",
            err
        );
        assert!(
            !err.is_listed_sidecar_absent(),
            "b.oya: a malformed sidecar is not the listed-sidecar-absent marker, so the collect \
             phase counts it whoever holds the checkpoint lease; got {err:?}"
        );
    }

    /// b.oya (decider S1) — a sidecar the listdir named that then reads
    /// `NotFound` is `Transient`, naming the sidecar, never skipped.
    ///
    /// A dangling symlink makes the listed-then-absent state deterministic:
    /// `read_dir` names `7.1.ext-refs.json` and the read answers `NotFound`.
    /// Skipping it would return a live set without that sidecar's hashes, and
    /// the reconciliation collect deletes every tombstoned blob the live set
    /// does not name. Pre-fix the walk `continue`d and answered `{"aa"}`.
    #[cfg(unix)]
    #[tokio::test]
    async fn list_referenced_ext_hashes_listed_sidecar_that_reads_absent_is_transient_naming_it() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // A readable sidecar beside the unreadable one, so a skip would return
        // a non-empty, plausible-looking set rather than nothing.
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o1"), test_meta())
            .await
            .expect("put 1");
        store
            .set_layer_sidecar(1, 1, sidecar_for(&["aa"]))
            .await
            .expect("sidecar 1");
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        crate::checkpoint::adapters::plant_dangling_sidecar(&layers_dir.join("7.1.ext-refs.json"));

        let err = match store.list_referenced_ext_hashes().await {
            Err(err) => err,
            Ok(hashes) => panic!(
                "a listed sidecar that reads absent must fail the walk, never yield a live set \
                 without its hashes; got Ok({hashes:?})"
            ),
        };
        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "Transient, so the collect phase defers to the next pass; got {err:?}"
        );
        assert!(
            err.is_listed_sidecar_absent(),
            "and typed as the listed-sidecar-absent marker, the one live-set failure the collect \
             phase qualifies by who holds the checkpoint lease; got {err:?}"
        );
        assert!(
            err.to_string().contains("7.1.ext-refs.json"),
            "the error must name the sidecar that read absent; got: {err}"
        );
        // The reconciliation collect reads the walk through
        // `list_live_ext_set`, so the marker has to survive there too — never
        // a set, and never a count without its hashes.
        match store.list_live_ext_set().await {
            Err(err) => assert!(
                err.is_listed_sidecar_absent(),
                "list_live_ext_set must carry the same marker; got {err:?}"
            ),
            Ok(live) => panic!("list_live_ext_set must fail the walk too, never yield {live:?}"),
        }
    }

    /// b.oya — a missing `layers/` directory is an empty store to the
    /// live-set walk: `(∅, 0)`, an empty set that vouches for no listing.
    ///
    /// A non-zero count here would license the reconciliation collect to
    /// delete every aged tombstoned blob against an empty set it read from
    /// nothing.
    #[tokio::test]
    async fn list_live_ext_set_on_a_missing_layers_dir_vouches_for_no_listing() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        tokio::fs::remove_dir(&layers_dir)
            .await
            .expect("removing the constructor's empty layers/ directory must succeed");

        let live = store
            .list_live_ext_set()
            .await
            .expect("a missing layers/ directory is an empty store, not an error");

        assert_eq!(live, LiveExtSet::default(), "(∅, 0)");
    }

    /// b.oya — the live-set walk counts every layer artifact its own
    /// `read_dir` pass named, of any kind, before the sidecar filter: a base
    /// with no sidecar gives `(∅, 2)` (its OMPF and meta), and a stray file
    /// beside it is not counted.
    ///
    /// The base-without-sidecar shape is the one that matters: an empty set
    /// with a non-zero count is the license to delete against it, and a
    /// count taken after the `.ext-refs.json` filter would read 0 here and
    /// leak every unreferenced blob on a store whose backups name none.
    #[tokio::test]
    async fn list_live_ext_set_counts_a_base_without_a_sidecar_and_no_stray_file() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o1"), test_meta())
            .await
            .expect("put 1");

        let live = store.list_live_ext_set().await.expect("list_live_ext_set");
        assert!(live.hashes.is_empty(), "no sidecar, no hash");
        assert_eq!(
            live.listed_layer_artifacts, 2,
            "the base's OMPF and meta, counted before the sidecar filter"
        );

        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        tokio::fs::write(
            layers_dir.join("NOTES.txt"),
            b"not a layer artifact".as_slice(),
        )
        .await
        .expect("planting the stray file must succeed");
        let live = store.list_live_ext_set().await.expect("list_live_ext_set");
        assert_eq!(
            live.listed_layer_artifacts, 2,
            "a name the artifact parser rejects is not counted"
        );
        assert_eq!(
            live.listed_layer_artifacts,
            store
                .list_layer_artifacts()
                .await
                .expect("list_layer_artifacts")
                .len(),
            "and the count equals what `list_layer_artifacts` reports from the same directory"
        );
    }

    // ----- t3.fgj.7v.cq.ih: positive session-checkpoint replacement tests --
    //
    // These replace the three Epic-1 `NotImplemented` negatives now that the
    // session-checkpoint methods are real filesystem operations. The
    // conformance harness (`tests/integration/checkpoint/conformance.rs`
    // cases #2 / #6 / #15) exercises the same surface through the trait;
    // these inline tests target adapter-internal behaviors (on-disk path
    // shape, atomic-write hygiene, idempotence) below the trait surface.

    #[tokio::test]
    async fn local_disk_put_then_get_session_checkpoint_round_trips_bytes() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let session_id = "sess-roundtrip";
        let body = Bytes::from_static(b"opaque-session-body");
        let handle = store
            .put_session_checkpoint(session_id, body.clone(), test_meta())
            .await
            .expect("put_session_checkpoint");

        let blob = store
            .get_session_checkpoint(session_id)
            .await
            .expect("get_session_checkpoint")
            .expect("Some(blob) after a put");
        assert_eq!(blob.bytes, body, "session-checkpoint bytes must round-trip");
        assert_eq!(
            blob.handle, handle,
            "get must return the same handle shape as put"
        );

        // The adapter_key MUST be the on-disk path so the conformance harness
        // (and the orphan sweep) can correlate handles with filesystem state.
        let expected_path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(SESSIONS_SUBDIR)
            .join("sess-roundtrip.json");
        assert_eq!(handle.adapter_key, expected_path.to_string_lossy());
    }

    #[tokio::test]
    async fn local_disk_get_session_checkpoint_absent_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Trait contract (store.rs:270-273): `get_*` returns `Ok(None)` on
        // absent, NEVER `Err(NotFound)`.
        let result = store
            .get_session_checkpoint("never-put")
            .await
            .expect("get on absent session must be Ok, not Err");
        assert!(
            result.is_none(),
            "fresh adapter must report no session for any session_id"
        );
    }

    #[tokio::test]
    async fn local_disk_put_then_delete_then_get_session_checkpoint_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let session_id = "sess-delete-roundtrip";
        store
            .put_session_checkpoint(session_id, Bytes::from_static(b"body"), test_meta())
            .await
            .expect("put");

        let path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(SESSIONS_SUBDIR)
            .join("sess-delete-roundtrip.json");
        assert!(
            tokio::fs::try_exists(&path).await.expect("try_exists"),
            "precondition: session JSON file exists after put"
        );

        store
            .delete_session_checkpoint(session_id)
            .await
            .expect("delete must succeed");

        assert!(
            !tokio::fs::try_exists(&path).await.expect("try_exists"),
            "session JSON file must be gone after delete"
        );

        let result = store
            .get_session_checkpoint(session_id)
            .await
            .expect("get after delete must be Ok, not Err");
        assert!(
            result.is_none(),
            "get after delete must return Ok(None), not Err"
        );
    }

    #[tokio::test]
    async fn local_disk_delete_session_checkpoint_on_absent_returns_ok() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Idempotent-delete invariant per the trait contract. The recovery
        // path re-issues deletes without first checking existence; a spurious
        // `NotFound` here would break it.
        store
            .delete_session_checkpoint("never-existed")
            .await
            .expect("delete on absent session must be Ok, not Err");
    }

    #[tokio::test]
    async fn local_disk_put_session_checkpoint_writes_at_expected_path() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let session_id = "sess-path-shape";
        let body = Bytes::from_static(b"opaque-body");
        store
            .put_session_checkpoint(session_id, body.clone(), test_meta())
            .await
            .expect("put");

        // SDD SR-4.6 — `<root>/.live-edit/sessions/<session_id>.json`.
        let expected = root
            .path()
            .join(NAMESPACE_DIR)
            .join(SESSIONS_SUBDIR)
            .join("sess-path-shape.json");
        assert!(
            tokio::fs::try_exists(&expected).await.expect("try_exists"),
            "session JSON must land at SR-4.6 path: {}",
            expected.display()
        );
        let written = tokio::fs::read(&expected).await.expect("readable");
        assert_eq!(written, body.as_ref(), "bytes on disk must equal body");
    }

    #[tokio::test]
    async fn local_disk_put_session_checkpoint_uses_atomic_write_no_visible_tmp_after_success() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        store
            .put_session_checkpoint("sess-tmp-hygiene", Bytes::from_static(b"body"), test_meta())
            .await
            .expect("put");

        // After a successful put, sessions/ MUST contain exactly one entry —
        // the final `.json` file — and no `.tmp.*` leftover. The write-temp +
        // rename discipline cleans up on success via the kernel rename; a
        // visible `.tmp` would indicate a path that bypassed `atomic_write`.
        // (The high-fidelity SIGKILL atomicity test lives in a separate
        // integration-test Subtask under this Task because it requires a
        // child-process harness.)
        let sessions_dir = root.path().join(NAMESPACE_DIR).join(SESSIONS_SUBDIR);
        let mut names: Vec<String> = Vec::new();
        let mut reader = tokio::fs::read_dir(&sessions_dir).await.expect("read_dir");
        while let Some(entry) = reader.next_entry().await.expect("next_entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(
            names,
            vec!["sess-tmp-hygiene.json".to_string()],
            "sessions/ must contain exactly the final JSON file, no .tmp leftover"
        );
    }

    #[tokio::test]
    async fn local_disk_put_session_checkpoint_last_writer_wins() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let session_id = "sess-overwrite";
        let b1 = Bytes::from_static(b"first-body");
        let b2 = Bytes::from_static(b"second-body-overwrites");
        store
            .put_session_checkpoint(session_id, b1, test_meta())
            .await
            .expect("first put");
        store
            .put_session_checkpoint(session_id, b2.clone(), test_meta())
            .await
            .expect("second put");

        let blob = store
            .get_session_checkpoint(session_id)
            .await
            .expect("get")
            .expect("Some(blob)");
        assert_eq!(
            blob.bytes, b2,
            "last-writer-wins: second put bytes must replace first"
        );

        // Filesystem introspection — exactly one session JSON file remains.
        let sessions_dir = root.path().join(NAMESPACE_DIR).join(SESSIONS_SUBDIR);
        let mut names: Vec<String> = Vec::new();
        let mut reader = tokio::fs::read_dir(&sessions_dir).await.expect("read_dir");
        while let Some(entry) = reader.next_entry().await.expect("next_entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(
            names,
            vec!["sess-overwrite.json".to_string()],
            "duplicate put on the same session_id must collapse to one file"
        );
    }

    #[tokio::test]
    async fn get_latest_layer_checkpoint_empty_adapter_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Trait contract: get_* paths return Ok(None) on absent, never NotFound.
        let result = store
            .get_latest_layer_checkpoint(999)
            .await
            .expect("get on empty adapter must be Ok, not Err");
        assert!(
            result.is_none(),
            "empty adapter must report no checkpoint for any object_id"
        );
    }

    #[tokio::test]
    async fn get_ext_blob_unknown_hash_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let result = store
            .get_ext_blob("never-stored")
            .await
            .expect("get on unknown hash must be Ok, not Err");
        assert!(
            result.is_none(),
            "unknown hash must report Ok(None), never NotFound"
        );
    }

    // ----- t3.fgj.ya.r7.3a: gap-coverage tests for engineer-added paths -----
    //
    // The cross-adapter SR-1.3 behavioral contracts (idempotent puts,
    // delete-then-get, conflict / transient classification, etc.) are
    // covered by the conformance harness in sibling Task `t2.fgj.ya.74`.
    // These tests target adapter-internal helpers and error-handling
    // branches that live below the trait surface and are otherwise
    // invisible to the conformance harness.

    /// `parse_ompf_filename` accepts a well-formed `<oid>.<snap>.ompf` for the
    /// requested object_id and returns the parsed `snap_seq`.
    #[test]
    fn parse_ompf_filename_valid_returns_snap_seq() {
        assert_eq!(parse_ompf_filename("42.7.ompf", 42), Some(7));
        assert_eq!(parse_ompf_filename("0.0.ompf", 0), Some(0));
        // u64 max round-trips both halves.
        let name = format!("{0}.{0}.ompf", u64::MAX);
        assert_eq!(parse_ompf_filename(&name, u64::MAX), Some(u64::MAX));
    }

    /// `parse_ompf_filename` rejects names whose object_id does not match,
    /// names without the `.ompf` extension, non-numeric components, and
    /// stems carrying more than two dot-separated parts. This is the
    /// directory-walk filter used by `list_generations_for_object` — any
    /// false positive would surface a phantom checkpoint to callers.
    #[test]
    fn parse_ompf_filename_rejects_malformed_inputs() {
        // Wrong object_id — match arm in callee.
        assert_eq!(parse_ompf_filename("42.7.ompf", 99), None);

        // Missing `.ompf` extension.
        assert_eq!(parse_ompf_filename("42.7", 42), None);
        assert_eq!(parse_ompf_filename("42.7.bin", 42), None);

        // Wrong companion suffix — sidecar files must be filtered out.
        assert_eq!(parse_ompf_filename("42.7.ext-refs.json", 42), None);

        // Non-numeric snap_seq.
        assert_eq!(parse_ompf_filename("42.abc.ompf", 42), None);
        // Non-numeric object_id.
        assert_eq!(parse_ompf_filename("abc.7.ompf", 42), None);

        // Negative snap_seq is non-numeric for u64.
        assert_eq!(parse_ompf_filename("42.-1.ompf", 42), None);

        // Extra dot components past the two-component split — protects
        // against names like `42.7.extra.ompf` masquerading as the file
        // family.
        assert_eq!(parse_ompf_filename("42.7.extra.ompf", 42), None);

        // Empty halves.
        assert_eq!(parse_ompf_filename(".7.ompf", 0), None);
        assert_eq!(parse_ompf_filename("42..ompf", 42), None);
        assert_eq!(parse_ompf_filename("", 0), None);
    }

    /// `parse_ompf_object_id` mirrors `parse_ompf_filename` but is the lookup
    /// helper for `list_all_object_ids`. The non-obvious branch is that it
    /// ALSO validates the snap_seq parses to u64 — a name like `5.abc.ompf`
    /// must be rejected so callers don't see a phantom object_id from a
    /// malformed sidecar of a different family.
    #[test]
    fn parse_ompf_object_id_validates_snap_seq_half() {
        // Happy path returns the object_id.
        assert_eq!(parse_ompf_object_id("42.7.ompf"), Some(42));

        // Both halves must parse, even though we only return the oid.
        assert_eq!(
            parse_ompf_object_id("42.abc.ompf"),
            None,
            "non-numeric snap_seq must filter out the object_id too"
        );

        // Missing extension and extra dots reject just like the snap_seq parser.
        assert_eq!(parse_ompf_object_id("42.7"), None);
        assert_eq!(parse_ompf_object_id("42.7.bin"), None);
        assert_eq!(parse_ompf_object_id("42.7.extra.ompf"), None);
        assert_eq!(parse_ompf_object_id("42.7.ext-refs.json"), None);
    }

    /// `tmp_path_for` builds a UUID-disambiguated `.tmp` companion that
    /// (a) sits alongside the final file in the same directory, (b) keeps
    /// the final filename as the prefix so an orphan recovery script can
    /// match them, and (c) is unique across calls so two concurrent writers
    /// to the same final path never collide on the temp path before the
    /// rename step.
    #[test]
    fn tmp_path_for_produces_unique_sibling_path() {
        let final_path = PathBuf::from("/var/checkpoints/layers/42.7.ompf");

        let tmp_a = tmp_path_for(&final_path);
        let tmp_b = tmp_path_for(&final_path);

        // Same parent directory — write-temp-then-rename relies on the
        // rename being intra-filesystem to be atomic.
        assert_eq!(
            tmp_a.parent(),
            final_path.parent(),
            "tmp file must live next to the final file for atomic rename"
        );
        // Distinct from the final path so the rename has somewhere to come
        // FROM.
        assert_ne!(tmp_a, final_path);
        // The temp filename starts with the final filename so an operator
        // grepping for orphaned `<oid>.<snap>.*.tmp` files finds them.
        let final_name = final_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let tmp_a_name = tmp_a.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            tmp_a_name.starts_with(&final_name),
            "temp filename must start with the final filename, got: {tmp_a_name}"
        );
        assert!(
            tmp_a_name.ends_with(".tmp"),
            "temp filename must end with .tmp, got: {tmp_a_name}"
        );
        // Two calls produce DIFFERENT temp paths — UUID uniqueness is what
        // prevents two replicas racing on the same `(oid, snap)` slot from
        // clobbering each other's temp file before the rename.
        assert_ne!(
            tmp_a, tmp_b,
            "two tmp_path_for calls must produce distinct paths (UUID uniqueness)"
        );
    }

    /// `tmp_path_for` on a path with no filename component (e.g. `/`) must
    /// not panic — it falls back to an empty filename and appends the
    /// `.<uuid>.tmp` suffix. The production code path never reaches this
    /// case (the OMPF / sidecar paths always have filenames), but the helper
    /// must be panic-free for the rust-analyzer-grade defensive coverage
    /// that the test review guide calls out.
    #[test]
    fn tmp_path_for_empty_filename_does_not_panic() {
        let weird = PathBuf::from("/");
        let tmp = tmp_path_for(&weird);
        // We do not pin the exact result — only that it didn't panic and is
        // still under the root directory.
        assert!(
            tmp.to_string_lossy().contains(".tmp"),
            "fallback path must still carry the .tmp suffix, got: {}",
            tmp.display()
        );
    }

    /// `io_to_permanent` maps a `std::io::Error` to a
    /// `CheckpointError::Permanent` carrying the original error as its
    /// source. This is what makes filesystem write/read failures
    /// non-retryable in the LocalDisk adapter — the orphan-sweep and
    /// recovery hook depend on the classification being `Permanent` so they
    /// fail fast instead of looping.
    #[test]
    fn io_to_permanent_classifies_as_permanent_and_preserves_source() {
        // Use `PermissionDenied` as a representative non-NotFound kind. The
        // `NotFound` mapping is handled at the call sites (Ok(None) for
        // gets, Ok(()) for deletes), so it never reaches `io_to_permanent`.
        let err = io::Error::new(io::ErrorKind::PermissionDenied, "test perms");
        let ckp_err = io_to_permanent(err);
        assert!(
            matches!(ckp_err, CheckpointError::Permanent { .. }),
            "io_to_permanent must classify as Permanent, got {:?}",
            ckp_err
        );
        // The error message must surface the source so operator logs can
        // see the underlying io::Error.
        assert!(
            ckp_err.to_string().contains("test perms"),
            "Permanent error must surface the io::Error message, got: {}",
            ckp_err
        );
    }

    /// `delete_layer_checkpoint` removes BOTH the `.ompf` file and the
    /// companion `.ext-refs.json` sidecar in one call. The orphan-sweep
    /// depends on this — a leaked sidecar would over-count referenced ext
    /// blobs and silently pin them past their retention window.
    #[tokio::test]
    async fn delete_layer_checkpoint_removes_ompf_and_sidecar() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 8;
        let seq: u64 = 3;
        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"o"), test_meta())
            .await
            .expect("put");
        store
            .set_layer_sidecar(oid, seq, sidecar_for(&["aa"]))
            .await
            .expect("sidecar");

        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        // Sanity preconditions.
        assert!(layers_dir.join("8.3.ompf").is_file());
        assert!(layers_dir.join("8.3.ext-refs.json").is_file());

        store
            .delete_layer_checkpoint(handle)
            .await
            .expect("delete must succeed");

        // Both files MUST be gone — orphan sweep depends on the pair being
        // dropped together so the sidecar's referenced hashes stop pinning
        // their ext blobs.
        assert!(
            !layers_dir.join("8.3.ompf").exists(),
            "ompf file must be removed by delete_layer_checkpoint"
        );
        assert!(
            !layers_dir.join("8.3.ext-refs.json").exists(),
            "sidecar file must be removed by delete_layer_checkpoint"
        );
    }

    /// `delete_layer_checkpoint` must be idempotent on missing files — the
    /// retention pruner re-issues deletes on the prune set without first
    /// checking existence (SDD SR-1.3 test #14 contract). A spurious
    /// `NotFound` here would break the pruner.
    #[tokio::test]
    async fn delete_layer_checkpoint_idempotent_on_missing() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // No put was ever done — the handle references nothing on disk.
        let handle = CheckpointHandle::base(99, 1, "phantom".into());
        store
            .delete_layer_checkpoint(handle)
            .await
            .expect("delete on a missing handle must be Ok, not NotFound");
    }

    /// `delete_ext_blob` removes the blob from the `ext/` directory and is
    /// idempotent on missing — the orphan sweep re-deletes any hash it
    /// already pruned in a prior tick and must not error on the repeat.
    #[tokio::test]
    async fn delete_ext_blob_removes_file_then_idempotent_on_missing() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let hash = "feedface";
        store
            .put_ext_blob(hash, Bytes::from_static(b"payload"))
            .await
            .expect("put");
        let path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(EXT_SUBDIR)
            .join("feedface.blob");
        assert!(path.is_file(), "precondition: blob exists after put");

        store
            .delete_ext_blob(hash)
            .await
            .expect("first delete must succeed");
        assert!(!path.exists(), "blob file must be removed after delete");

        // Second delete on the now-missing hash is the idempotent case.
        store
            .delete_ext_blob(hash)
            .await
            .expect("second delete on a missing hash must be Ok, not NotFound");
    }

    /// `list_all_object_ids` walks the layers dir and returns the deduplicated
    /// set of object_ids over **every generation kind**, not over OMPF files.
    /// The recovery hook drives this method to enumerate every layer the
    /// adapter has — any miss or duplicate would surface as a wrong fanout
    /// there.
    ///
    /// # Why the OMPF-driven wording is gone
    ///
    /// This test previously asserted the enumeration was "every well-formed
    /// `<oid>.<snap>.ompf` file's object_id", which was true only while a base
    /// was the sole generation kind on disk. Since `t3.dfh.gy.ib.ck` the walk
    /// runs through [`parse_generation_object_id`], which admits `<oid>.<to_seq>.tail`
    /// as well. That is load-bearing rather than cosmetic: a **tail-only**
    /// object — one whose base has already been retention-pruned, or that has
    /// only ever been tail-checkpointed — has no `.ompf` file at all, so an
    /// OMPF-driven enumeration reports it as *not existing*. Nothing downstream
    /// re-derives the working set, so such an object would be invisible to
    /// retention, reconciliation and recovery alike: a silent leak, and a silent
    /// failure to restore. The fixture below therefore covers a base-only
    /// object, a tail-only object, and an object holding both kinds (which must
    /// still appear exactly **once** — the dedup is over `object_id`, not over
    /// generations, so being seen twice must not double it).
    ///
    /// The companion / temp / foreign-name strays assert the other half: the
    /// widened parser must not have widened far enough to admit a `.meta.json`,
    /// a `.tail.meta.json`, a leaked `.tmp`, or a non-numeric object id — any of
    /// which would conjure a phantom object for the recovery hook to fan out to.
    #[tokio::test]
    async fn list_all_object_ids_returns_dedup_set_across_generations() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // (a) oid 5 — base-only, two generations (dedup within one kind).
        // (c) oid 12 — base half of a both-kinds object.
        for (oid, seq) in [(5_u64, 1_u64), (5, 2), (12, 1)] {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from_static(b"x"), test_meta())
                .await
                .expect("put base");
        }
        // (b) oid 7 — TAIL-ONLY: no `.ompf` file exists for it anywhere. An
        // OMPF-driven enumeration omits this object entirely.
        store
            .put_layer_checkpoint(
                7,
                9,
                Bytes::from_static(b"tail-envelope"),
                tail_meta(1, 2, 9),
            )
            .await
            .expect("put tail-only object");
        // (c) oid 12 — tail half, at a sequence that also collides with nothing
        // in particular; the object must still report exactly once.
        store
            .put_layer_checkpoint(
                12,
                4,
                Bytes::from_static(b"tail-envelope"),
                tail_meta(1, 2, 4),
            )
            .await
            .expect("put tail on a both-kinds object");

        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        // Strays that must all be skipped: a plain file, a malformed-name OMPF,
        // a malformed-name tail, and both companion families plus a leaked temp
        // under a tail stem (the ones a widened parser is most likely to admit).
        for (name, body) in [
            ("readme.txt", &b"not a checkpoint"[..]),
            ("not-a-number.99.ompf", b"junk"),
            ("not-a-number.99.tail", b"junk"),
            ("31.4.meta.json", b"{}"),
            ("31.4.ext-refs.json", b"{}"),
            ("31.4.tail.meta.json", b"{}"),
            (
                "31.4.tail.11111111-2222-3333-4444-555555555555.tmp",
                b"junk",
            ),
        ] {
            tokio::fs::write(layers_dir.join(name), body)
                .await
                .unwrap_or_else(|e| panic!("write stray {name}: {e}"));
        }

        let mut ids = store.list_all_object_ids().await.expect("list");
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![5, 7, 12],
            "list_all_object_ids must return the deduplicated set of valid object_ids across BOTH \
             generation kinds — 7 is tail-only and 12 holds a base and a tail"
        );
    }

    /// `list_all_object_ids` on an adapter with NO layer checkpoints must
    /// return an empty vec (not an error). This is the recovery hook's
    /// fast-path on a fresh deploy.
    #[tokio::test]
    async fn list_all_object_ids_empty_returns_empty_vec() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let ids = store.list_all_object_ids().await.expect("list");
        assert!(ids.is_empty(), "fresh adapter must report no object_ids");
    }

    /// `list_layer_checkpoints` on an unknown object_id MUST return an empty
    /// vec, not an error. The retention pruner loops over the result and
    /// surfaces this case when the object was just GC'd.
    #[tokio::test]
    async fn list_layer_checkpoints_unknown_object_returns_empty() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let handles = store
            .list_layer_checkpoints(12345)
            .await
            .expect("list on unknown object must be Ok, not Err");
        assert!(handles.is_empty());
    }

    /// `get_latest_layer_checkpoint` happy-path round-trips both the OMPF
    /// bytes AND the `CheckpointMeta` the caller wrote — including the
    /// session-level identity fields the apply path needs to rebuild
    /// `obj:hdr`. Pre-b1 this adapter returned a synthetic placeholder
    /// meta on every get; that forced the apply path to invent identity
    /// and made round-tripping impossible. The persisted `.meta.json`
    /// sidecar is what fixes it.
    #[tokio::test]
    async fn get_latest_layer_checkpoint_round_trips_bytes_and_persisted_meta() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 50;
        let seq: u64 = 17;
        let payload = Bytes::from_static(b"opaque-ompf-payload");
        let meta_in = CheckpointMeta {
            creator_replica_id: "writer-pod-7".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(seq),
            session_id: "sess-42".to_string(),
            layer_name: "layer-A".to_string(),
            created_by: "alice".to_string(),
            server_ts: 1_700_000_000_500,
            observed_commit_span: 0,
            sidecar_bytes: None,
            ..CheckpointMeta::default()
        };
        store
            .put_layer_checkpoint(oid, seq, payload.clone(), meta_in.clone())
            .await
            .expect("put");

        let blob = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect("get")
            .expect("Some(blob)");

        assert_eq!(blob.handle.object_id, oid);
        assert_eq!(blob.handle.snap_seq, seq);
        assert_eq!(blob.bytes, payload, "bytes must round-trip unchanged");

        // Every persisted field MUST round-trip. The `.meta.json` sidecar
        // is the load-bearing primitive that makes `LayerStore::apply_checkpoint`
        // possible: without persisted `session_id` / `layer_name` /
        // `created_by` / `server_ts` the apply path has no way to rebuild
        // `obj:hdr` correctly.
        assert_eq!(blob.meta.creator_replica_id, meta_in.creator_replica_id);
        assert_eq!(blob.meta.written_at_unix_ms, meta_in.written_at_unix_ms);
        assert_eq!(blob.meta.snap_seq, meta_in.snap_seq);
        assert_eq!(blob.meta.session_id, meta_in.session_id);
        assert_eq!(blob.meta.layer_name, meta_in.layer_name);
        assert_eq!(blob.meta.created_by, meta_in.created_by);
        assert_eq!(blob.meta.server_ts, meta_in.server_ts);
        // `sidecar_bytes` is `#[serde(skip)]` — the ext-refs sidecar is a
        // distinct file that this get path does NOT thread back through
        // `CheckpointBlob` (the orphan sweep reads it separately). Round
        // trip MUST surface `None` here regardless of what the caller put.
        assert!(
            blob.meta.sidecar_bytes.is_none(),
            "sidecar_bytes is intentionally not surfaced on the read path"
        );
    }

    /// An `.ompf` on disk with no `.meta.json` companion is REFUSED, as the
    /// typed artifact-present / meta-absent `Permanent` (b.hka (d)).
    ///
    /// This case previously asserted the opposite — that the adapter
    /// synthesized `CheckpointMeta::default()` and served the blob — on the
    /// theory that a pre-meta-companion generation might be left on disk by an
    /// in-place upgrade. That population is empty (no release tag predates the
    /// meta companion — all tags are pre-release), and since b.b2j the apply
    /// path's identity guard refuses
    /// an identity-empty blob anyway, three layers above the adapter that
    /// already knew exactly what was wrong. So the placeholder bought nothing
    /// but a `NotFound` naming neither the artifact nor the missing file.
    ///
    /// The assertion is on the *typed* marker
    /// ([`CheckpointError::is_missing_meta_companion`]) rather than on the
    /// message text, because that predicate is what
    /// `checkpoint::lineage`'s restore-lineage walk keys its step-down on — a
    /// `Permanent` built any other way would hard-fail the restore instead.
    /// The message check is deliberately narrow: it names the two addresses an
    /// operator needs to go look at, which is the whole point of classifying
    /// here rather than upstream.
    #[tokio::test]
    async fn get_latest_layer_checkpoint_missing_meta_json_is_a_typed_permanent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 5;

        // A raw .ompf with NO .meta.json companion: a torn rolled-back put, or
        // out-of-band removal. Unreachable through the adapter's own publish
        // (meta first) and delete (artifact first) ordering.
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        tokio::fs::write(layers_dir.join("5.9.ompf"), b"legacy-payload")
            .await
            .expect("write ompf without its companion");

        let err = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect_err("an artifact whose identity cannot be read must be refused");

        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "the companion's absence is a durable fact — a retry re-reads the same absence; \
             got {err:?}"
        );
        assert!(
            err.is_missing_meta_companion(),
            "must carry the TYPED marker, not just any Permanent: the restore-lineage walk \
             steps down on this predicate and on nothing else; got {err:?}"
        );
        let message = err.to_string();
        assert!(
            message.contains("5.9.ompf"),
            "the message must name the artifact that IS present; got {message}"
        );
        assert!(
            message.contains("5.9.meta.json"),
            "the message must name the companion that is NOT present; got {message}"
        );
    }

    /// The same refusal for a **tail** envelope whose `.tail.meta.json` never
    /// landed. Both kinds, one rule — `MissingMetaCompanion`'s "both kinds"
    /// section — so a tail is not silently admitted through a base-only guard.
    #[tokio::test]
    async fn get_layer_checkpoint_on_a_tail_missing_its_meta_is_a_typed_permanent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 5;
        let to_seq: u64 = 9;

        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        tokio::fs::write(layers_dir.join("5.9.tail"), b"tail-envelope")
            .await
            .expect("write tail envelope without its companion");

        let err = store
            .get_layer_checkpoint(&CheckpointHandle::tail(oid, to_seq, "5.9.tail".to_string()))
            .await
            .expect_err("a tail whose identity cannot be read must be refused too");

        assert!(
            err.is_missing_meta_companion(),
            "tails are NOT exempt: an identity-less tail is as unusable as an identity-less \
             base; got {err:?}"
        );
        let message = err.to_string();
        assert!(
            message.contains("5.9.tail.meta.json"),
            "the message must name the tail's own companion; got {message}"
        );
    }

    /// A `.meta.json` whose bytes are not valid JSON (corruption /
    /// truncation) MUST classify as `Permanent` so the recovery hook surfaces
    /// the failure to the operator instead of silently swallowing identity
    /// fields. Re-reading the same corrupt bytes will never succeed; treating
    /// it as transient would loop the recovery path indefinitely.
    #[tokio::test]
    async fn get_latest_layer_checkpoint_malformed_meta_json_returns_permanent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 6;
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        // Drop a valid .ompf alongside a deliberately-malformed .meta.json.
        tokio::fs::write(layers_dir.join("6.2.ompf"), b"ompf-bytes")
            .await
            .expect("write ompf");
        tokio::fs::write(layers_dir.join("6.2.meta.json"), b"not-json-at-all{{{")
            .await
            .expect("write malformed meta");

        let err = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect_err("malformed meta.json must Err, not silently fall back");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "malformed CheckpointMeta JSON must classify Permanent; got {:?}",
            err
        );
    }

    /// `put_layer_checkpoint` MUST land both the `.ompf` and the
    /// `.meta.json` on disk — the absence of the latter is what reduced
    /// previous LocalDisk checkpoints to synthetic-placeholder reads and
    /// blocked `LayerStore::apply_checkpoint` from recovering identity.
    #[tokio::test]
    async fn put_layer_checkpoint_writes_meta_json_alongside_ompf() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 11;
        let seq: u64 = 22;
        let meta_in = CheckpointMeta {
            creator_replica_id: "pod-3".to_string(),
            written_at_unix_ms: 42,
            snap_seq: Some(seq),
            session_id: "sess-x".to_string(),
            layer_name: "layer-y".to_string(),
            created_by: "bob".to_string(),
            server_ts: 99,
            // b.vnw — deliberately non-zero, and distinct from every other
            // number in this fixture. `0` is simultaneously the field's
            // "no measurement available" sentinel and the `#[serde(default)]`
            // value a sidecar that never carried the key deserializes to, so a
            // serializer that dropped the field entirely would still satisfy
            // the round-trip assertion below. Only a non-zero value
            // discriminates.
            observed_commit_span: 4_242,
            sidecar_bytes: None,
            ..CheckpointMeta::default()
        };

        store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"o"), meta_in.clone())
            .await
            .expect("put");

        let meta_path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("11.22.meta.json");
        assert!(
            meta_path.is_file(),
            "put_layer_checkpoint must persist <oid>.<snap_seq>.meta.json"
        );

        // Verify the JSON round-trips byte-for-byte through
        // `serde_json::from_slice`. (We do not pin exact JSON spelling —
        // that's serde's internal contract — but every field value must
        // survive.)
        let bytes = tokio::fs::read(&meta_path).await.expect("read meta");
        let parsed: CheckpointMeta = serde_json::from_slice(&bytes).expect("meta JSON parses");
        assert_eq!(parsed.creator_replica_id, meta_in.creator_replica_id);
        assert_eq!(parsed.session_id, meta_in.session_id);
        assert_eq!(parsed.layer_name, meta_in.layer_name);
        assert_eq!(parsed.created_by, meta_in.created_by);
        assert_eq!(parsed.server_ts, meta_in.server_ts);
        assert_eq!(parsed.observed_commit_span, meta_in.observed_commit_span);
    }

    /// b.vnw — a `.meta.json` written by a PRE-b.vnw build has no
    /// `observed_commit_span` key at all. It must still deserialize, with the
    /// field defaulting to 0.
    ///
    /// This is the field's whole back-compat contract, and the blobs it
    /// protects are exactly the ones a real incident reaches for: the sidecars
    /// already sitting on an operator's disk when they upgrade. A
    /// deserialize error here would classify `Permanent` (see
    /// `malformed_meta_json_*`) and make every pre-upgrade generation
    /// unreadable — turning a guard-band improvement into a total loss of the
    /// existing DR corpus.
    ///
    /// 0 is not merely a filler value: it is the "no measurement available"
    /// sentinel `apply_checkpoint` folds back to the configured
    /// `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN`, so a legacy blob restores
    /// with exactly its pre-b.vnw band.
    ///
    /// The JSON is hand-written rather than round-tripped through a struct so
    /// the test pins the on-disk shape a previous release actually produced —
    /// a serialize-then-strip fixture would silently track any future rename
    /// of the field instead of catching it.
    #[tokio::test]
    async fn legacy_meta_json_without_observed_commit_span_reads_back_as_zero() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 77;
        let seq: u64 = 5;
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        tokio::fs::write(layers_dir.join("77.5.ompf"), b"legacy-ompf-bytes")
            .await
            .expect("write ompf");
        // Exactly the pre-b.vnw key set — no `observed_commit_span`.
        tokio::fs::write(
            layers_dir.join("77.5.meta.json"),
            br#"{
              "creator_replica_id": "pod-legacy",
              "written_at_unix_ms": 1700000000000,
              "snap_seq": 5,
              "session_id": "sess-legacy",
              "layer_name": "layer-legacy",
              "created_by": "alice",
              "server_ts": 1699999999999
            }"#,
        )
        .await
        .expect("write legacy meta");

        let blob = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect("a pre-b.vnw meta sidecar must still read back, not Err")
            .expect("the generation must be visible");

        assert_eq!(
            blob.meta.observed_commit_span, 0,
            "b.vnw: a meta.json with no observed_commit_span key must default \
             to 0 — the 'no measurement' sentinel that restores a legacy blob \
             with the configured safety margin, unchanged from pre-b.vnw"
        );
        // The rest of the identity must survive untouched: the new field must
        // not have disturbed how the existing ones deserialize.
        assert_eq!(blob.meta.session_id, "sess-legacy");
        assert_eq!(blob.meta.layer_name, "layer-legacy");
        assert_eq!(blob.meta.created_by, "alice");
        assert_eq!(blob.meta.server_ts, 1_699_999_999_999);
        assert_eq!(blob.meta.snap_seq, Some(seq));
        assert_eq!(blob.bytes.as_ref(), b"legacy-ompf-bytes");
    }

    /// `delete_layer_checkpoint` MUST scrub the `.meta.json` companion the
    /// same way it scrubs the `.ext-refs.json` companion. Without this,
    /// re-creating a checkpoint at the same `(oid, snap_seq)` after a
    /// delete would silently revive stale identity from the old run.
    #[tokio::test]
    async fn delete_layer_checkpoint_also_removes_meta_json() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 13;
        let seq: u64 = 4;
        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"o"), test_meta())
            .await
            .expect("put");
        store
            .set_layer_sidecar(oid, seq, sidecar_for(&["aa"]))
            .await
            .expect("ext-refs sidecar");

        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        // Sanity preconditions — all three companions exist.
        assert!(layers_dir.join("13.4.ompf").is_file());
        assert!(layers_dir.join("13.4.ext-refs.json").is_file());
        assert!(layers_dir.join("13.4.meta.json").is_file());

        store
            .delete_layer_checkpoint(handle)
            .await
            .expect("delete must succeed");

        assert!(
            !layers_dir.join("13.4.ompf").exists(),
            "ompf must be removed"
        );
        assert!(
            !layers_dir.join("13.4.ext-refs.json").exists(),
            "ext-refs sidecar must be removed"
        );
        assert!(
            !layers_dir.join("13.4.meta.json").exists(),
            "meta.json companion must be removed by delete_layer_checkpoint \
             so re-creating (oid, snap_seq) doesn't revive stale identity"
        );
    }

    /// b.s5y R2 — `delete_layer_checkpoint` removes the ext-refs sidecar
    /// FIRST, so a partial delete leaves the generation still enumerable and
    /// the whole delete retryable.
    ///
    /// # The two possible residuals, and why only one is recoverable
    ///
    /// Deleting the OMPF first — what this adapter did before R2 — means a
    /// failure part-way through leaves a sidecar whose OMPF is gone. Every
    /// enumeration here is generation-stem-driven
    /// ([`LocalDiskCheckpointStore::list_generations_for_object`] admits only
    /// generation stems — `*.ompf` and `*.tail` — never a `*.ext-refs.json`),
    /// so nothing will ever offer that sidecar for deletion again: a permanent
    /// orphan pinning this snapshot's entire ext hash set in the retention
    /// cascade's live set, exactly like the failed-put orphan above.
    ///
    /// Ext-refs first inverts it into a self-correcting residual: the OMPF
    /// survives, `list_layer_checkpoints` still reports the generation, and the
    /// next tick's prune retries the whole delete. In between,
    /// `read_ext_refs_sidecar` returns the empty set for the missing sidecar,
    /// which UNDER-counts the pruned hashes — the documented-safe direction.
    ///
    /// The mid-delete failure is real, not simulated: the OMPF path is replaced
    /// by a directory, so `remove_file(2)` on it fails with `EISDIR` while the
    /// filename still parses as an enumerable generation.
    #[tokio::test]
    async fn delete_layer_checkpoint_leaves_the_generation_enumerable_when_a_later_step_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        let oid: u64 = 31;
        let seq: u64 = 6;

        // Plant the triplet directly, with the OMPF as a DIRECTORY so its
        // `remove_file` fails while its filename still enumerates.
        tokio::fs::write(layers_dir.join("31.6.ext-refs.json"), sidecar_for(&["aa"]))
            .await
            .expect("planting the sidecar must succeed");
        tokio::fs::write(layers_dir.join("31.6.meta.json"), b"{}")
            .await
            .expect("planting the meta must succeed");
        tokio::fs::create_dir(layers_dir.join("31.6.ompf"))
            .await
            .expect("planting the un-deletable OMPF must succeed");

        let handle = CheckpointHandle::base(
            oid,
            seq,
            layers_dir.join("31.6.ompf").to_string_lossy().into_owned(),
        );
        let err = store
            .delete_layer_checkpoint(handle)
            .await
            .expect_err("removing a directory with remove_file must fail");
        assert!(err.is_permanent(), "expected Permanent, got {err:?}");

        assert!(
            !layers_dir.join("31.6.ext-refs.json").exists(),
            "the sidecar is deleted FIRST, so it must be gone by the time the OMPF step fails"
        );
        assert!(
            layers_dir.join("31.6.meta.json").exists(),
            "the delete halts at the first hard error, so the meta step never ran"
        );

        let handles = store
            .list_layer_checkpoints(oid)
            .await
            .expect("list_layer_checkpoints must succeed");
        assert_eq!(
            handles.iter().map(|h| h.snap_seq).collect::<Vec<_>>(),
            vec![seq],
            "THE property: a partially-deleted generation must stay enumerable, so the next \
             retention prune retries the delete. Under the pre-R2 OMPF-first order this list \
             is empty and the surviving sidecar is unreachable forever"
        );
        assert_eq!(
            store
                .read_ext_refs_sidecar(&handles[0])
                .await
                .expect("read_ext_refs_sidecar must tolerate the missing sidecar"),
            std::collections::HashSet::new(),
            "and the intermediate state under-counts rather than over-counts: a missing \
             sidecar reads as the empty set, so the cascade defers blob deletes instead of \
             performing unsafe ones"
        );
    }

    /// `get_latest_layer_checkpoint` returns the HIGHEST snap_seq when
    /// multiple generations exist — duplicates the `BTreeMap::last_key_value`
    /// equivalent in the InMemory adapter but verified end-to-end through
    /// the filesystem walk.
    #[tokio::test]
    async fn get_latest_layer_checkpoint_with_multiple_generations_returns_highest() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let oid: u64 = 22;
        for (seq, body) in [(1_u64, b"v1".as_slice()), (3, b"v3"), (2, b"v2")] {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from(body.to_vec()), test_meta())
                .await
                .expect("put");
        }

        let blob = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect("get")
            .expect("Some(blob)");
        assert_eq!(
            blob.handle.snap_seq, 3,
            "get_latest must return the highest snap_seq across all generations"
        );
        assert_eq!(
            blob.bytes,
            Bytes::from_static(b"v3"),
            "bytes must match the highest-snap_seq generation"
        );
    }

    /// `get_ext_blob` happy-path round-trip — proves bytes survive the
    /// file-system journey unchanged. The Bytes equality is content equality;
    /// the LocalDisk adapter does not preserve allocation identity (it can't,
    /// the bytes go through a kernel buffer).
    #[tokio::test]
    async fn get_ext_blob_round_trips_payload_after_put() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let payload = Bytes::from_static(b"\x00\x01\x02binary-payload\xff");
        store
            .put_ext_blob("ff00", payload.clone())
            .await
            .expect("put");

        let got = store
            .get_ext_blob("ff00")
            .await
            .expect("get")
            .expect("Some(bytes)");
        assert_eq!(got, payload, "bytes must round-trip unchanged");
    }

    /// `get_counters_sidecar` round-trips bytes after a put and follows the
    /// last-writer-wins contract — a second put replaces the first.
    #[tokio::test]
    async fn put_then_get_counters_sidecar_round_trips_last_write() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Fresh adapter — get must report Ok(None).
        assert!(
            store.get_counters_sidecar().await.expect("get").is_none(),
            "fresh adapter must report no counters sidecar"
        );

        let v1 = Bytes::from_static(b"{\"object_id\":1}");
        store
            .put_counters_sidecar(v1.clone())
            .await
            .expect("put v1");
        let got_v1 = store
            .get_counters_sidecar()
            .await
            .expect("get v1")
            .expect("Some(bytes)");
        assert_eq!(got_v1, v1);

        // Last-writer-wins per the trait contract — atomic_write replaces
        // the file in place.
        let v2 = Bytes::from_static(b"{\"object_id\":2}");
        store
            .put_counters_sidecar(v2.clone())
            .await
            .expect("put v2");
        let got_v2 = store
            .get_counters_sidecar()
            .await
            .expect("get v2")
            .expect("Some(bytes)");
        assert_eq!(
            got_v2, v2,
            "second put must replace the first (last-writer-wins)"
        );
    }

    // ----- t3.fgj.zc.56.uq: per-handle `read_ext_refs_sidecar` tests --------
    //
    // The retention-prune cascade (parent Task t2.fgj.zc.56) needs the
    // per-generation referenced-hash set to compute "what hashes are being
    // pruned by this delete?" — independently of `list_referenced_ext_hashes`
    // which only answers the union across every live generation. These four
    // tests cover the trait-contract branches (happy path / absent sidecar /
    // malformed sidecar / handle-for-absent-generation) end-to-end through
    // the filesystem layer.

    /// Happy path: a layer checkpoint put with `meta.sidecar_bytes = Some(_)`
    /// lands an ext-refs sidecar file on disk; `read_ext_refs_sidecar` for
    /// the corresponding handle returns the exact `HashSet` parsed from
    /// that sidecar. The cascade caller derives the handle from
    /// `list_layer_checkpoints`, so we exercise the same `(object_id,
    /// snap_seq)` addressing path here.
    #[tokio::test]
    async fn read_ext_refs_sidecar_returns_parsed_hashes_for_happy_path_handle() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 30;
        let seq: u64 = 4;

        // Put with `sidecar_bytes` threaded through the meta — the canonical
        // assembler path. The adapter writes <oid>.<seq>.ext-refs.json
        // alongside the OMPF and the meta.json companion.
        let mut meta = test_meta();
        meta.snap_seq = Some(seq);
        meta.sidecar_bytes = Some(sidecar_for(&["aa", "bb", "cc"]));
        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"opaque-ompf"), meta)
            .await
            .expect("put");

        let refs = store
            .read_ext_refs_sidecar(&handle)
            .await
            .expect("read_ext_refs_sidecar must succeed on happy-path handle");
        let expected: HashSet<String> = ["aa", "bb", "cc"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "per-handle sidecar reader must return the exact parsed hash set"
        );
    }

    /// Absent-sidecar path: a layer checkpoint put with `meta.sidecar_bytes =
    /// None` writes the OMPF + meta.json but NO `.ext-refs.json`.
    /// `read_ext_refs_sidecar` MUST return `Ok(HashSet::new())` — absence is
    /// the safe degradation per the trait contract; under-counting refs is
    /// acceptable, over-counting would falsely pin blobs.
    #[tokio::test]
    async fn read_ext_refs_sidecar_returns_empty_when_sidecar_file_absent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 31;
        let seq: u64 = 5;

        // `meta.sidecar_bytes = None` (default) — no sidecar file is written.
        // Identity is still populated: `validate_generation_identity` refuses a
        // put without it, and this case is about the sidecar, not identity.
        let handle = store
            .put_layer_checkpoint(
                oid,
                seq,
                Bytes::from_static(b"opaque-ompf"),
                CheckpointMeta {
                    snap_seq: Some(seq),
                    ..crate::checkpoint::adapters::identity_default()
                },
            )
            .await
            .expect("put");
        // Sanity: the sidecar file is genuinely absent on disk.
        let sidecar_path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("31.5.ext-refs.json");
        assert!(
            !sidecar_path.exists(),
            "precondition: no sidecar file should have been written when meta.sidecar_bytes is None"
        );

        let refs = store
            .read_ext_refs_sidecar(&handle)
            .await
            .expect("absent sidecar must be Ok(empty), not Err");
        assert!(
            refs.is_empty(),
            "missing sidecar file must yield Ok(empty) per trait contract; got {refs:?}"
        );
    }

    /// Malformed sidecar: drop a non-JSON byte sequence at the canonical
    /// sidecar path and verify the per-handle reader classifies the parse
    /// failure as `Permanent`. Durable bytes will not parse on retry, so
    /// surfacing `Permanent` is what lets the cascade halt and surface the
    /// corruption to the operator instead of looping on a transient retry.
    #[tokio::test]
    async fn read_ext_refs_sidecar_malformed_bytes_classify_as_permanent() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 32;
        let seq: u64 = 6;

        // Put the OMPF + meta first so the handle is well-formed; then drop
        // a deliberately-malformed sidecar straight onto the filesystem at
        // the SR-4.6 path so we control the bytes exactly.
        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"opaque"), test_meta())
            .await
            .expect("put");
        let sidecar_path = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("32.6.ext-refs.json");
        tokio::fs::write(&sidecar_path, b"not-json-at-all{{{")
            .await
            .expect("write malformed sidecar");

        let err = store
            .read_ext_refs_sidecar(&handle)
            .await
            .expect_err("malformed sidecar must Err, not silently downgrade to empty");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "malformed sidecar must classify as Permanent (durable bytes, no retry possible); got {:?}",
            err
        );
    }

    /// Handle-for-absent-generation: synthesize a handle for an `(object_id,
    /// snap_seq)` tuple that was never `put_layer_checkpoint`'d. The reader
    /// MUST return `Ok(HashSet::new())` — the cascade caller passes
    /// already-pruned handles through this method when the prune set races
    /// with a sibling deleter, and a synthetic `Permanent` here would break
    /// the cascade.
    #[tokio::test]
    async fn read_ext_refs_sidecar_for_absent_generation_returns_empty() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // A completely-synthetic handle: nothing has ever been put, so no
        // OMPF / sidecar / meta.json exists at (99, 1).
        let phantom = CheckpointHandle::base(99, 1, "phantom".into());
        let refs = store
            .read_ext_refs_sidecar(&phantom)
            .await
            .expect("absent generation must be Ok(empty), not Err");
        assert!(
            refs.is_empty(),
            "handle for never-put (oid, snap_seq) must yield Ok(empty); got {refs:?}"
        );
    }

    /// `list_referenced_ext_hashes` on an adapter with NO sidecars returns an
    /// empty set — the orphan sweep relies on this so a freshly-deployed
    /// replica does not falsely delete blobs the assembler hasn't yet
    /// linked. (Verified for both the empty layers-dir case and the
    /// "OMPF-only, no sidecar" case.)
    #[tokio::test]
    async fn list_referenced_ext_hashes_no_sidecars_returns_empty() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Case 1: fresh adapter, layers dir is empty.
        let refs = store.list_referenced_ext_hashes().await.expect("list");
        assert!(
            refs.is_empty(),
            "fresh adapter must report no referenced hashes"
        );

        // Case 2: OMPF present but no sidecar — orphan sweep must NOT see
        // any referenced hashes yet (the assembler hasn't run).
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o1"), test_meta())
            .await
            .expect("put");
        let refs = store.list_referenced_ext_hashes().await.expect("list");
        assert!(
            refs.is_empty(),
            "OMPF without sidecar must not contribute any referenced hashes"
        );
    }

    // ----- t3.dfh.gy.bd.11: `get_layer_checkpoint` (SR-2.3) ----------------
    //
    // Adapter-local mechanics only. The cross-adapter conformance cases for
    // this method belong to Task `t2.dfh.gy.qb`; these pin what is specific to
    // this adapter's on-disk `<oid>.<snap_seq>.*` layout — including a
    // hand-written pre-kind `.meta.json` fixture — plus the tail arms this
    // adapter grew in `t3.dfh.gy.bd.1r` and completed in `t3.dfh.gy.ib.ck`.

    /// Build a `CheckpointMeta` whose identity fields are all distinct per
    /// generation, so a read that returns the *wrong* generation's meta is
    /// distinguishable from one that returns the right one.
    fn meta_for_generation(seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: format!("replica-{seq}"),
            written_at_unix_ms: 1_700_000_000_000 + seq,
            snap_seq: Some(seq),
            session_id: format!("sess-{seq}"),
            layer_name: format!("layer-{seq}"),
            created_by: format!("user-{seq}"),
            server_ts: 1_700_000_500_000 + seq,
            ..CheckpointMeta::default()
        }
    }

    /// The capability `get_latest_layer_checkpoint` structurally cannot
    /// provide: fetch a generation that is **not** the newest. Three
    /// generations are written and the *middle* handle is fetched, so an
    /// implementation that lazily delegated to `get_latest` would read the
    /// wrong triplet and fail here.
    #[tokio::test]
    async fn get_layer_checkpoint_returns_the_addressed_middle_generation_not_the_newest() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 90;
        for seq in [3u64, 5, 9] {
            store
                .put_layer_checkpoint(
                    oid,
                    seq,
                    Bytes::from(format!("ompf-{seq}")),
                    meta_for_generation(seq),
                )
                .await
                .expect("put");
        }

        // Newest-first ordering, so index 1 is the middle generation (seq 5).
        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        assert_eq!(handles.len(), 3, "three generations were written");
        let middle = handles[1].clone();
        assert_eq!(
            middle.snap_seq, 5,
            "index 1 of a newest-first list is seq 5"
        );

        let blob = store
            .get_layer_checkpoint(&middle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("the addressed generation exists");

        assert_eq!(blob.bytes, Bytes::from_static(b"ompf-5"));
        assert_eq!(blob.handle, middle, "handle must round-trip identically");
        assert_eq!(blob.handle.kind, CheckpointKind::Base);
        // Identity fields the apply path rebuilds `obj:hdr` from — all four
        // must belong to the *addressed* generation, not the newest one.
        assert_eq!(blob.meta.session_id, "sess-5");
        assert_eq!(blob.meta.layer_name, "layer-5");
        assert_eq!(blob.meta.created_by, "user-5");
        assert_eq!(blob.meta.server_ts, 1_700_000_500_005);
        assert_eq!(blob.meta.snap_seq, Some(5));
        assert_eq!(blob.meta.kind, CheckpointKind::Base);
    }

    /// `adapter_key` is opaque carrier state: a handle a caller fabricated
    /// without ever seeing this adapter's key MUST still resolve, and the
    /// returned handle MUST carry the adapter's own canonical key (the OMPF
    /// path) rather than echoing the caller's.
    #[tokio::test]
    async fn get_layer_checkpoint_ignores_adapter_key_and_returns_the_canonical_one() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        store
            .put_layer_checkpoint(7, 2, Bytes::from_static(b"payload"), meta_for_generation(2))
            .await
            .expect("put");

        let fabricated = CheckpointHandle::base(7, 2, "junk-key-the-caller-invented".to_string());
        let blob = store
            .get_layer_checkpoint(&fabricated)
            .await
            .expect("get")
            .expect("a fabricated adapter_key must still resolve");

        assert_eq!(blob.bytes, Bytes::from_static(b"payload"));
        let expected_key = root
            .path()
            .join(NAMESPACE_DIR)
            .join(LAYERS_SUBDIR)
            .join("7.2.ompf");
        assert_eq!(
            blob.handle.adapter_key,
            expected_key.to_string_lossy(),
            "returned handle must carry this adapter's canonical OMPF path, not the caller's key"
        );
    }

    /// Absence flavour 1 — an `object_id` that was never written.
    #[tokio::test]
    async fn get_layer_checkpoint_unknown_object_id_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .expect("put");

        let handle = CheckpointHandle::base(999, 1, "unused".to_string());
        match store.get_layer_checkpoint(&handle).await {
            Ok(None) => {}
            other => panic!("absent object_id must be Ok(None), never NotFound; got {other:?}"),
        }
    }

    /// Absence flavour 2 — a real `object_id` at a `snap_seq` that does not
    /// exist. This is also the concurrent-prune race shape: a listing named
    /// the handle and retention removed the triplet before the get landed.
    #[tokio::test]
    async fn get_layer_checkpoint_unknown_snap_seq_on_known_object_returns_ok_none() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .expect("put");

        let handle = CheckpointHandle::base(1, 4_242, "unused".to_string());
        match store.get_layer_checkpoint(&handle).await {
            Ok(None) => {}
            other => panic!("absent snap_seq must be Ok(None), never NotFound; got {other:?}"),
        }
    }

    /// The invariant Task 5's kind-aware selection is built on: list → pick a
    /// handle → fetch it agrees with `get_latest_layer_checkpoint` for the
    /// newest generation.
    ///
    /// Driven through `Arc<dyn CheckpointStore>` so object safety is proven at
    /// a real call site rather than only by the
    /// `_assert_checkpoint_store_object_safe` compile guard.
    #[tokio::test]
    async fn get_layer_checkpoint_for_the_newest_handle_matches_get_latest_layer_checkpoint() {
        let root = tempdir().expect("tempdir");
        let store: std::sync::Arc<dyn CheckpointStore> = std::sync::Arc::new(
            LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor"),
        );
        let oid: u64 = 91;
        for seq in [2u64, 6] {
            store
                .put_layer_checkpoint(
                    oid,
                    seq,
                    Bytes::from(format!("bytes-{seq}")),
                    meta_for_generation(seq),
                )
                .await
                .expect("put");
        }

        let newest = store.list_layer_checkpoints(oid).await.expect("list")[0].clone();
        let by_handle = store
            .get_layer_checkpoint(&newest)
            .await
            .expect("get by handle")
            .expect("newest generation exists");
        let latest = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect("get latest")
            .expect("newest generation exists");

        assert_eq!(by_handle.handle, latest.handle);
        assert_eq!(by_handle.bytes, latest.bytes);
        assert_eq!(by_handle.meta.snap_seq, latest.meta.snap_seq);
    }

    /// An existing dev/rc store's `.meta.json` predates the `kind` field
    /// entirely (PRD A-6 / SR-2.5). Reading it through the *new* method must
    /// succeed and report [`CheckpointKind::Base`] with no lineage — the
    /// end-to-end complement to the type-level `serde(default)` test.
    ///
    /// The fixture is hand-written rather than produced by
    /// `put_layer_checkpoint`, because a put by this build would stamp
    /// `"kind":"base"` explicitly and could not distinguish "defaulted" from
    /// "persisted".
    #[tokio::test]
    async fn get_layer_checkpoint_reads_a_pre_kind_meta_json_on_disk_as_base() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);

        tokio::fs::write(layers_dir.join("77.4.ompf"), b"pre-kind-payload")
            .await
            .expect("write legacy ompf");
        // Exactly the shape a pre-feature build persisted: identity fields
        // present, NO `kind` / `base_snap_seq` / `from_seq` keys.
        let legacy_meta = br#"{
            "creator_replica_id": "old-pod-1",
            "written_at_unix_ms": 1700000000000,
            "snap_seq": 4,
            "session_id": "sess-legacy",
            "layer_name": "layer-legacy",
            "created_by": "carol",
            "server_ts": 1700000000500
        }"#;
        tokio::fs::write(layers_dir.join("77.4.meta.json"), legacy_meta)
            .await
            .expect("write legacy meta");

        let blob = store
            .get_layer_checkpoint(&CheckpointHandle::base(77, 4, "unused".to_string()))
            .await
            .expect("a pre-kind meta.json must not error")
            .expect("the legacy generation must be readable");

        assert_eq!(blob.bytes, Bytes::from_static(b"pre-kind-payload"));
        assert_eq!(blob.meta.session_id, "sess-legacy");
        assert_eq!(blob.meta.created_by, "carol");
        assert_eq!(
            blob.meta.kind,
            CheckpointKind::Base,
            "a meta.json with no `kind` key must deserialize as Base"
        );
        assert_eq!(blob.meta.base_snap_seq, None);
        assert_eq!(blob.meta.from_seq, None);
        assert_eq!(blob.handle.kind, CheckpointKind::Base);
    }

    // ----- Tail arms (t3.dfh.gy.bd.1r) -------------------------------------

    /// A tail carrying an ext-refs sidecar is rejected `Permanent` by the shared
    /// `validate_tail_put` gate **before any artifact is written** — of either
    /// stem. Supersedes the Epic-2 interim assertion that this same call
    /// classified `NotImplemented`; the rejection survives, its reason and
    /// classification are now the SR-2.4 sidecar MUST rather than "tail storage
    /// is unimplemented".
    #[tokio::test]
    async fn put_layer_checkpoint_with_a_tail_sidecar_is_rejected_and_writes_no_artifact() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let tail_meta = CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(4),
            from_seq: Some(4),
            snap_seq: Some(8),
            sidecar_bytes: Some(sidecar_for(&["aa"])),
            ..CheckpointMeta::default()
        };

        let err = store
            .put_layer_checkpoint(12, 8, Bytes::from_static(b"tail-envelope"), tail_meta)
            .await
            .expect_err("a tail put carrying an ext-refs sidecar must be rejected");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a tail sidecar is a caller bug — Permanent, got {err:?}"
        );

        // Nothing partial, on either stem: not even the ext-refs sidecar, which
        // is ordinarily the FIRST artifact this adapter publishes on a base.
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        for name in [
            "12.8.ext-refs.json",
            "12.8.meta.json",
            "12.8.ompf",
            "12.8.tail",
            "12.8.tail.meta.json",
        ] {
            assert!(
                !layers_dir.join(name).exists(),
                "a rejected tail put must leave no `{name}` behind"
            );
        }
        // Stronger than the name probes above, which can only rule out names
        // the test thought to enumerate: the directory is *empty*, so no
        // differently-stemmed sidecar (`12.8.tail.ext-refs.json`, say) and no
        // leaked `.tmp` survives either.
        assert!(
            sorted_layer_names(&store).await.is_empty(),
            "a rejected tail put must leave the layers directory untouched; found {:?}",
            sorted_layer_names(&store).await
        );
        // And the observable consequence the SR-2.4 rejection exists for: the
        // rejected tail contributed nothing to the retention cascade's live
        // set, so it pins no blob against collection.
        assert!(
            store
                .list_referenced_ext_hashes()
                .await
                .expect("list_referenced_ext_hashes must succeed")
                .is_empty(),
            "a rejected tail put must pin no ext hash in the cascade's live set"
        );
    }

    /// A tail handle whose `to_seq` collides with an existing base's
    /// `snap_seq` MUST NOT read that base's triplet — otherwise a caller
    /// would be handed a full snapshot labelled as a tail.
    #[tokio::test]
    async fn get_layer_checkpoint_on_a_tail_handle_returns_ok_none_even_when_a_base_shares_the_seq()
    {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        store
            .put_layer_checkpoint(20, 5, Bytes::from_static(b"base-bytes"), test_meta())
            .await
            .expect("put base");

        let tail_handle = CheckpointHandle::tail(20, 5, "unused".to_string());
        match store.get_layer_checkpoint(&tail_handle).await {
            Ok(None) => {}
            other => panic!("a tail handle must resolve to Ok(None), got {other:?}"),
        }
    }

    /// The load-bearing tail-delete assertion: a tail handle at `to_seq == N`
    /// must never delete the base triplet at `snap_seq == N`. `ompf_path` /
    /// `sidecar_path` / `meta_path` are kind-blind, so a tail delete that
    /// reached for them would unlink all three files and strand the generation.
    #[tokio::test]
    async fn delete_layer_checkpoint_on_a_tail_handle_leaves_the_same_sequence_base_intact() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let meta = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["aa"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(21, 5, Bytes::from_static(b"base-bytes"), meta)
            .await
            .expect("put base");

        store
            .delete_layer_checkpoint(CheckpointHandle::tail(21, 5, "unused".to_string()))
            .await
            .expect("deleting an absent tail is Ok(()) — idempotent on missing");

        // The WHOLE triplet must survive, not just the OMPF: deleting only
        // the sidecar would silently unpin this generation's ext blobs.
        let layers_dir = root.path().join(NAMESPACE_DIR).join(LAYERS_SUBDIR);
        for name in ["21.5.ext-refs.json", "21.5.meta.json", "21.5.ompf"] {
            assert!(
                layers_dir.join(name).is_file(),
                "a tail delete must leave the same-sequence base's `{name}` intact"
            );
        }
    }

    /// **FINAL behaviour, not a placeholder** (SR-2.4): a tail carries no
    /// ext-refs sidecar by design, so the empty set is the answer Tasks 3/4
    /// must preserve rather than "complete". Falling through to the base's
    /// sidecar would report the BASE's hashes as the tail's and inflate the
    /// retention cascade's per-handle pruned set.
    #[tokio::test]
    async fn read_ext_refs_sidecar_on_a_tail_handle_returns_empty_and_not_the_bases_hashes() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let meta = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["aa", "bb"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(22, 5, Bytes::from_static(b"base-bytes"), meta)
            .await
            .expect("put base with sidecar");
        // Sanity: the base handle DOES report the hashes, so an empty result
        // below cannot pass vacuously.
        let base_refs = store
            .read_ext_refs_sidecar(&CheckpointHandle::base(22, 5, "unused".to_string()))
            .await
            .expect("base read");
        assert_eq!(base_refs.len(), 2, "test precondition: base has 2 hashes");

        let tail_refs = store
            .read_ext_refs_sidecar(&CheckpointHandle::tail(22, 5, "unused".to_string()))
            .await
            .expect("tail read must succeed");
        assert!(
            tail_refs.is_empty(),
            "a tail handle's ext-refs set is empty by design; got {tail_refs:?}"
        );
    }

    // ----- Tail stems & mixed-kind enumeration (t3.dfh.gy.ib.b8) -----------
    //
    // This adapter's whole enumeration surface is *filename parsing over a
    // directory walk*, so its mixed-kind correctness lives below the trait
    // surface and is invisible to a cross-adapter conformance case (Task
    // `t2.dfh.gy.qb` owns those; nothing here belongs in `conformance.rs`).
    // Three hazards are specific to the mechanism and are what this block
    // pins:
    //
    //   1. a `read_dir`-order-dependent listing once base and tail sequences
    //      can tie,
    //   2. a meta-sidecar stem collision silently clobbering one kind's
    //      lineage, and
    //   3. leaked `.tmp` companions enumerating as phantom generations.
    //
    // Direct unit tests of the private stem helpers are the established local
    // precedent (`parse_ompf_filename_valid_returns_snap_seq` and its
    // siblings above), so the parser tests below follow it rather than the
    // test-writing guide's general "don't test private helpers" rule: these
    // helpers ARE the enumeration contract.

    /// A lineage-complete tail meta — the shape `validate_tail_put` admits,
    /// carrying identity so `validate_generation_identity` admits it too
    /// (tails are not exempt).
    fn tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(base_snap_seq),
            from_seq: Some(from_seq),
            snap_seq: Some(to_seq),
            ..crate::checkpoint::adapters::identity_default()
        }
    }

    /// The final path component of a path the adapter's own builders produced.
    fn file_name_of(path: &Path) -> String {
        path.file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("path {} has a UTF-8 file name", path.display()))
            .to_string()
    }

    /// Round-trip the **path builders' own output** back through the parsers,
    /// and prove the two generation families never alias in either direction.
    ///
    /// Feeding `tail_path` / `ompf_path` output in — rather than a hand-written
    /// `"42.7.tail"` literal — is what makes this a contract test instead of a
    /// restatement of the `format!` inside the builder: a builder that changed
    /// its stem shape would still satisfy a literal-driven test while going
    /// silently invisible to `list_generations_for_object`, which is the
    /// silent-data-loss shape SR-2.3 pins the stem to prevent.
    ///
    /// The companion names get the same treatment. `<oid>.<to_seq>.tail.meta.json`
    /// must be rejected by BOTH artifact parsers, because a meta admitted as a
    /// generation is a phantom tail; and `tmp_path_for`'s `<final>.<uuid>.tmp`
    /// must be rejected for the same reason — that exclusion is precisely what
    /// makes an in-flight `atomic_write` invisible to a listing, i.e. the
    /// publish-order guarantee rather than a tidiness rule.
    #[test]
    fn tail_stem_round_trips_through_its_parsers_and_never_aliases_the_ompf_family() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        // Boundary values on both halves: zero, one, and the u64 ceiling — the
        // ceiling is the one a narrower integer type in a future refactor
        // would silently truncate.
        for (oid, seq) in [
            (0_u64, 0_u64),
            (0, 1),
            (1, 0),
            (42, 7),
            (0, u64::MAX),
            (u64::MAX, 0),
            (u64::MAX, u64::MAX),
        ] {
            let tail_name = file_name_of(&store.tail_path(oid, seq));
            let ompf_name = file_name_of(&store.ompf_path(oid, seq));
            let tail_meta_name = file_name_of(&store.tail_meta_path(oid, seq));
            let base_meta_name = file_name_of(&store.meta_path(oid, seq));
            let tail_tmp_name = file_name_of(&tmp_path_for(&store.tail_path(oid, seq)));

            // Hazard 2, at the naming layer: the two kinds' meta companions must
            // be DIFFERENT files, or a base and a tail at one sequence overwrite
            // each other's lineage (`base_snap_seq` / `from_seq` are exactly what
            // Epic 4's gather keys on).
            assert_ne!(
                tail_meta_name, base_meta_name,
                "({oid}, {seq}): the tail meta stem must be kind-qualified, not the base's"
            );

            // ── The tail stem round-trips ─────────────────────────────────
            assert_eq!(
                parse_tail_filename(&tail_name, oid),
                Some(seq),
                "`{tail_name}` is what tail_path built for ({oid}, {seq}); the tail parser must \
                 return that sequence"
            );
            assert_eq!(
                parse_tail_object_id(&tail_name),
                Some(oid),
                "`{tail_name}` must yield its object_id for list_all_object_ids"
            );
            assert_eq!(
                parse_generation_filename(&tail_name, oid),
                Some((seq, CheckpointKind::Tail)),
                "the kind-dispatching parser must label `{tail_name}` a Tail"
            );
            assert_eq!(parse_generation_object_id(&tail_name), Some(oid));
            // A tail belongs to exactly one object.
            assert_eq!(
                parse_tail_filename(&tail_name, oid.wrapping_add(1)),
                None,
                "`{tail_name}` must not answer for a different object_id"
            );

            // ── The base stem round-trips, unchanged ──────────────────────
            assert_eq!(parse_ompf_filename(&ompf_name, oid), Some(seq));
            assert_eq!(
                parse_generation_filename(&ompf_name, oid),
                Some((seq, CheckpointKind::Base)),
                "the kind-dispatching parser must label `{ompf_name}` a Base"
            );
            assert_eq!(parse_generation_object_id(&ompf_name), Some(oid));

            // ── The two families are disjoint, both directions ────────────
            assert_eq!(
                parse_ompf_filename(&tail_name, oid),
                None,
                "`{tail_name}` must not parse as an OMPF generation"
            );
            assert_eq!(parse_ompf_object_id(&tail_name), None);
            assert_eq!(
                parse_tail_filename(&ompf_name, oid),
                None,
                "`{ompf_name}` must not parse as a tail envelope"
            );
            assert_eq!(parse_tail_object_id(&ompf_name), None);

            // ── Companions and temps are not generations ──────────────────
            for (label, name) in [
                ("tail meta companion", &tail_meta_name),
                ("base meta companion", &base_meta_name),
                ("leaked tail temp", &tail_tmp_name),
            ] {
                assert_eq!(
                    parse_generation_filename(name, oid),
                    None,
                    "the {label} `{name}` must never enumerate as a generation"
                );
                assert_eq!(
                    parse_generation_object_id(name),
                    None,
                    "the {label} `{name}` must never conjure an object_id"
                );
            }
        }
    }

    /// `parse_tail_filename`'s rejection table, mirroring
    /// `parse_ompf_filename_rejects_malformed_inputs` component for component.
    /// This is the directory-walk filter behind `list_generations_for_object`,
    /// so any false positive here is a phantom tail generation offered to
    /// retention, reconciliation and DR replay alike.
    #[test]
    fn parse_tail_filename_rejects_malformed_inputs() {
        // Wrong object_id — the match arm in the callee.
        assert_eq!(parse_tail_filename("42.7.tail", 99), None);

        // Companion families: the kind-qualified tail meta is the one that
        // MUST NOT be readmitted as the envelope it names.
        assert_eq!(parse_tail_filename("42.7.tail.meta.json", 42), None);
        assert_eq!(parse_tail_filename("42.7.meta.json", 42), None);
        assert_eq!(parse_tail_filename("42.7.ext-refs.json", 42), None);

        // A leaked temp companion. `atomic_write` removes its own temp on every
        // error path it can observe, but a hard kill between write and rename
        // strands one; admitting it would publish a half-written envelope.
        assert_eq!(
            parse_tail_filename("42.7.tail.11111111-2222-3333-4444-555555555555.tmp", 42),
            None
        );

        // The other generation family.
        assert_eq!(parse_tail_filename("42.7.ompf", 42), None);

        // Missing the second stem half entirely.
        assert_eq!(parse_tail_filename("42.tail", 42), None);
        // Non-numeric to_seq / object_id.
        assert_eq!(parse_tail_filename("42.abc.tail", 42), None);
        assert_eq!(parse_tail_filename("not-a-number.7.tail", 42), None);
        // Negative to_seq is non-numeric for u64.
        assert_eq!(parse_tail_filename("42.-1.tail", 42), None);
        // Extra dot components past the strict two-component split.
        assert_eq!(parse_tail_filename("42.7.extra.tail", 42), None);
        // Empty halves.
        assert_eq!(parse_tail_filename(".7.tail", 0), None);
        assert_eq!(parse_tail_filename("42..tail", 42), None);
        assert_eq!(parse_tail_filename("", 0), None);
        // The bare suffix with no stem at all.
        assert_eq!(parse_tail_filename(".tail", 0), None);
    }

    /// `parse_tail_object_id` is to `parse_tail_filename` what
    /// `parse_ompf_object_id` is to `parse_ompf_filename`, and it carries the
    /// same non-obvious branch: **both** halves must parse before an object_id
    /// is admitted. `list_all_object_ids` is its only caller, and a phantom
    /// object there fans the recovery hook out to a layer that does not exist.
    #[test]
    fn parse_tail_object_id_validates_to_seq_half() {
        // Happy path returns the object_id.
        assert_eq!(parse_tail_object_id("42.7.tail"), Some(42));

        // Both halves must parse, even though only the oid is returned.
        assert_eq!(
            parse_tail_object_id("42.abc.tail"),
            None,
            "non-numeric to_seq must filter out the object_id too"
        );
        assert_eq!(parse_tail_object_id("not-a-number.7.tail"), None);

        // Missing suffix, missing half, and extra dots reject just like the
        // to_seq parser.
        assert_eq!(parse_tail_object_id("42.7"), None);
        assert_eq!(parse_tail_object_id("42.tail"), None);
        assert_eq!(parse_tail_object_id("42.7.extra.tail"), None);
        assert_eq!(parse_tail_object_id("42.7.ompf"), None);
        assert_eq!(parse_tail_object_id("42.7.tail.meta.json"), None);
        assert_eq!(
            parse_tail_object_id("42.7.tail.11111111-2222-3333-4444-555555555555.tmp"),
            None
        );
        assert_eq!(parse_tail_object_id(".7.tail"), None);
        assert_eq!(parse_tail_object_id("42..tail"), None);
        assert_eq!(parse_tail_object_id(""), None);
    }

    /// A single object holding both generation kinds at interleaved sequences
    /// lists as ONE newest-first run, not as two concatenated per-kind runs.
    ///
    /// Bases at 1 and 4, tails at 2 and 7 ⇒ `[7 tail, 4 base, 2 tail, 1 base]`.
    /// The kinds alternate, so an implementation that walked `*.ompf` and
    /// `*.tail` separately and appended one list to the other reproduces the
    /// sequence *set* but not this order — hence the per-handle kind assertion
    /// rather than a sequence-vector-only check. The `adapter_key` is asserted
    /// per handle for the same reason: minting every key as an OMPF path would
    /// hand a tail handle a key naming a *different* generation, and the
    /// sequence vector alone would not notice.
    #[tokio::test]
    async fn list_layer_checkpoints_interleaves_bases_and_tails_newest_first() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 300;

        for seq in [4_u64, 1] {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from(format!("base-{seq}")), test_meta())
                .await
                .expect("put base");
        }
        for (base_snap_seq, from_seq, to_seq) in [(4_u64, 5_u64, 7_u64), (1, 2, 2)] {
            store
                .put_layer_checkpoint(
                    oid,
                    to_seq,
                    Bytes::from(format!("tail-{to_seq}")),
                    tail_meta(base_snap_seq, from_seq, to_seq),
                )
                .await
                .expect("put tail");
        }

        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        let observed: Vec<(u64, CheckpointKind)> =
            handles.iter().map(|h| (h.seq(), h.kind)).collect();
        assert_eq!(
            observed,
            vec![
                (7, CheckpointKind::Tail),
                (4, CheckpointKind::Base),
                (2, CheckpointKind::Tail),
                (1, CheckpointKind::Base),
            ],
            "a mixed-kind listing must be one newest-first run over both kinds"
        );

        for handle in &handles {
            assert_eq!(handle.object_id, oid);
            let expected_key = match handle.kind {
                CheckpointKind::Base => store.ompf_path(oid, handle.seq()),
                CheckpointKind::Tail => store.tail_path(oid, handle.seq()),
            };
            assert_eq!(
                handle.adapter_key,
                expected_key.to_string_lossy(),
                "the {:?} handle at {} must carry its OWN artifact path as adapter_key",
                handle.kind,
                handle.seq()
            );
        }
    }

    /// Hazard 1 and hazard 2 together: a base at `snap_seq = 5` and a tail at
    /// `to_seq = 5` for one object must **coexist** on disk, list as two
    /// distinct handles, each fetch its own bytes, and do so in a *stable total
    /// order* that does not depend on `read_dir` order or on write order.
    ///
    /// This is a live case, not a defensive hypothetical: SR-4.1 re-base mints a
    /// fresh base at `S`, where `S` is the newest consumed tail's `to_seq`, so
    /// `base@S` and `tail@S` genuinely coexist until retention removes one.
    ///
    /// Two independent failure modes are covered.
    ///
    /// - **Collapse to one handle** would mean the stems collided on disk — the
    ///   naive `<oid>.<seq>.meta.json` for a tail overwrites the base's meta and
    ///   the survivor's `kind` mislabels the generation it was left standing
    ///   for. Asserting `len() == 2` plus a per-handle byte fetch is what
    ///   distinguishes "two files" from "one file listed twice".
    /// - **Unstable order.** `list_generations_for_object` sorts with
    ///   `sort_unstable_by_key`, and its input is `read_dir` order. Before ties
    ///   existed the comparator was a total order by accident; the moment a base
    ///   and a tail can share a sequence, only the kind rank inside the shared
    ///   `generation_sort_key` keeps the listing a total order at all. Five
    ///   repeat calls plus both write orders are what make an order that leaned
    ///   on the filesystem fail here rather than intermittently in production.
    ///
    /// The asserted tie-break is the pinned cross-adapter one — **Base before
    /// Tail** at an equal sequence, produced by `generation_sort_key` and shared
    /// with InMemory and StorageApi, not re-derived here. At a tie the base is
    /// the artifact that restores on its own, so `get_latest_layer_checkpoint`'s
    /// retained `list().first()` contract hands a kind-blind caller the safe one.
    ///
    /// # Why several colliding stems rather than one
    ///
    /// Directory-entry order on the filesystems this runs on is a hash of the
    /// **name**, not of creation order — so for any one stem it is fixed, and a
    /// single `(oid, seq)` whose two names happen to hash Base-first would let a
    /// comparator with NO kind tie-break pass this test on every run and every
    /// write order. That is the exact failure this case exists to catch, so one
    /// stem is not enough. The stems below were chosen by observing that their
    /// entry order splits both ways in practice: at least one lists `.tail`
    /// ahead of `.ompf`, which is where the missing tie-break shows.
    ///
    /// Handle vectors are compared **whole** (`adapter_key` included), not
    /// projected to `(seq, kind)`: on this adapter the key is derived from the
    /// same per-kind path builders the listing walks, so a kind/key mismatch is
    /// exactly the drift a projection would hide. Across the two write orders
    /// the comparison strips the tempdir prefix, since that differs by
    /// construction between the two stores and is not part of the contract.
    #[tokio::test]
    async fn base_and_tail_at_the_same_sequence_list_as_two_handles_in_a_stable_total_order() {
        async fn put_base(store: &LocalDiskCheckpointStore, oid: u64, seq: u64) {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from_static(b"base-bytes"), test_meta())
                .await
                .expect("put base");
        }
        async fn put_tail(store: &LocalDiskCheckpointStore, oid: u64, seq: u64) {
            store
                .put_layer_checkpoint(
                    oid,
                    seq,
                    Bytes::from_static(b"tail-bytes"),
                    tail_meta(1, 2, seq),
                )
                .await
                .expect("put tail");
        }
        async fn seed(
            root: &Path,
            oid: u64,
            seq: u64,
            tail_first: bool,
        ) -> LocalDiskCheckpointStore {
            let store = LocalDiskCheckpointStore::new(root.to_path_buf()).expect("constructor");
            if tail_first {
                put_tail(&store, oid, seq).await;
                put_base(&store, oid, seq).await;
            } else {
                put_base(&store, oid, seq).await;
                put_tail(&store, oid, seq).await;
            }
            store
        }

        // `adapter_key` minus the per-store tempdir prefix, so two stores rooted
        // at different temp paths are comparable without weakening the handle.
        fn portable(store: &LocalDiskCheckpointStore, handles: &[CheckpointHandle]) -> Vec<String> {
            handles
                .iter()
                .map(|h| {
                    let key = Path::new(&h.adapter_key);
                    let relative = key
                        .strip_prefix(&store.layers_dir)
                        .expect("every adapter_key must name a file under this store's layers dir");
                    format!(
                        "{}/{}/{:?}/{}",
                        h.object_id,
                        h.seq(),
                        h.kind,
                        relative.display()
                    )
                })
                .collect()
        }

        for (oid, seq) in [(301_u64, 5_u64), (37, 5), (4, 11), (1234, 900), (9, 9)] {
            let mut baseline: Option<Vec<String>> = None;

            for tail_first in [false, true] {
                let root = tempdir().expect("tempdir");
                let store = seed(root.path(), oid, seq, tail_first).await;

                // Both artifact families survived, under distinct stems — the
                // on-disk half of the no-collision claim. The two metas are the
                // load-bearing pair: under the naive `<oid>.<seq>.meta.json` for
                // a tail, one of them would not be here.
                for name in [
                    format!("{oid}.{seq}.ompf"),
                    format!("{oid}.{seq}.meta.json"),
                    format!("{oid}.{seq}.tail"),
                    format!("{oid}.{seq}.tail.meta.json"),
                ] {
                    assert!(
                        store.layers_dir.join(&name).is_file(),
                        "({oid}, {seq}) tail_first={tail_first}: `{name}` must survive alongside \
                         the other kind"
                    );
                }

                let first = store.list_layer_checkpoints(oid).await.expect("list");
                assert_eq!(
                    first.len(),
                    2,
                    "({oid}, {seq}) tail_first={tail_first}: a base and a tail at the same \
                     sequence are TWO generations; a collapse to {} means their stems collided \
                     on disk",
                    first.len()
                );
                assert_eq!(
                    first.iter().map(|h| h.kind).collect::<Vec<_>>(),
                    vec![CheckpointKind::Base, CheckpointKind::Tail],
                    "({oid}, {seq}) tail_first={tail_first}: the pinned tie-break is Base before \
                     Tail at an equal sequence"
                );

                // Each handle resolves independently to ITS OWN artifact.
                // Fetching through the listed handle (not a fabricated one) is
                // what proves the listing handed out addressable, non-aliased
                // generations.
                for handle in &first {
                    let blob = store
                        .get_layer_checkpoint(handle)
                        .await
                        .expect("get must succeed")
                        .unwrap_or_else(|| {
                            panic!(
                                "({oid}, {seq}) tail_first={tail_first}: the {:?} handle must \
                                 resolve",
                                handle.kind
                            )
                        });
                    let expected: &[u8] = match handle.kind {
                        CheckpointKind::Base => b"base-bytes",
                        CheckpointKind::Tail => b"tail-bytes",
                    };
                    assert_eq!(
                        blob.bytes,
                        Bytes::from_static(expected),
                        "({oid}, {seq}) tail_first={tail_first}: the {:?} handle must return its \
                         own bytes, not the other kind's",
                        handle.kind
                    );
                    assert_eq!(blob.handle.kind, handle.kind);
                    assert_eq!(blob.meta.kind, handle.kind);
                }

                // Repeatability within one store: the WHOLE handle vector, five
                // times, against a directory walk whose entry order is not ours.
                for round in 0..5 {
                    let again = store.list_layer_checkpoints(oid).await.expect("list");
                    assert_eq!(
                        again, first,
                        "({oid}, {seq}) tail_first={tail_first} round={round}: every listing of \
                         the same stored set must be identical, adapter_key included"
                    );
                }

                // Repeatability across write orders, tempdir prefix aside.
                let observed = portable(&store, &first);
                match baseline.as_ref() {
                    None => baseline = Some(observed),
                    Some(expected) => assert_eq!(
                        &observed, expected,
                        "({oid}, {seq}): the listing must be a function of the stored set, not of \
                         the order the base and the tail were written in"
                    ),
                }
            }
        }
    }

    /// The directory walk is shared with every other object and with every
    /// companion family, so `list_layer_checkpoints` for one object must be
    /// exactly its own generations — nothing the walk merely passed over.
    ///
    /// The strays are chosen as the ones a too-permissive parser actually
    /// admits: a plain file, a non-numeric-oid tail, a `.tail.meta.json` at a
    /// sequence that has **no** envelope (so admitting it adds a third handle
    /// rather than duplicating an existing one), a leaked `.tail.<uuid>.tmp`,
    /// and another object's well-formed tail.
    #[tokio::test]
    async fn list_layer_checkpoints_ignores_companions_temps_and_foreign_objects() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 302;

        store
            .put_layer_checkpoint(oid, 3, Bytes::from_static(b"base-bytes"), test_meta())
            .await
            .expect("put base");
        store
            .put_layer_checkpoint(
                oid,
                8,
                Bytes::from_static(b"tail-bytes"),
                tail_meta(3, 4, 8),
            )
            .await
            .expect("put tail");
        // Another object's tail, written through the adapter so its name is
        // genuinely well-formed rather than approximated by hand.
        store
            .put_layer_checkpoint(
                999,
                8,
                Bytes::from_static(b"other-tail"),
                tail_meta(1, 2, 8),
            )
            .await
            .expect("put foreign tail");

        for (name, body) in [
            ("readme.txt".to_string(), &b"not a checkpoint"[..]),
            ("not-a-number.99.tail".to_string(), b"junk"),
            // A tail meta at a sequence with no envelope: if the walk admitted
            // metas, this alone would list as a third generation.
            (format!("{oid}.42.tail.meta.json"), b"{}"),
            (
                format!("{oid}.42.tail.11111111-2222-3333-4444-555555555555.tmp"),
                b"half-written",
            ),
        ] {
            tokio::fs::write(store.layers_dir.join(&name), body)
                .await
                .unwrap_or_else(|e| panic!("write stray {name}: {e}"));
        }

        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        let observed: Vec<(u64, CheckpointKind)> =
            handles.iter().map(|h| (h.seq(), h.kind)).collect();
        assert_eq!(
            observed,
            vec![(8, CheckpointKind::Tail), (3, CheckpointKind::Base)],
            "the listing must be exactly this object's own two generations"
        );
        for handle in &handles {
            assert_eq!(
                handle.object_id, oid,
                "a foreign object's generation must never appear in this object's listing"
            );
        }
    }

    // ----- Tail put/get/delete mechanics (t3.dfh.gy.ib.9h) -----------------
    //
    // The sibling block above pins *enumeration* over tail stems. This one pins
    // the write/read/unlink mechanics underneath it — on-disk shape, put-time
    // lineage validation, the no-ext-refs-for-a-tail guard, same-sequence
    // delete safety, failed-write residuals, and the injection seam. All of it
    // is below the trait surface (exact filenames, publish/delete ordering,
    // partial-write residuals), so none of it belongs in the cross-adapter
    // conformance harness (Task `t2.dfh.gy.qb`) — that harness cannot even see
    // a directory, let alone reopen a store over one.

    /// Every entry in this store's `layers` directory, sorted.
    ///
    /// A whole-directory comparison rather than a list of `exists()` probes,
    /// following `put_layer_checkpoint_rolls_back_the_sidecar_when_the_ompf_write_fails`:
    /// a probe list can only rule out the names the test author thought of, and
    /// the residuals that matter here are precisely the ones nobody enumerated
    /// (a differently-stemmed sidecar, a leaked `.tmp`, an orphaned companion).
    async fn sorted_layer_names(store: &LocalDiskCheckpointStore) -> Vec<String> {
        let mut reader = tokio::fs::read_dir(&store.layers_dir)
            .await
            .expect("layers dir readable");
        let mut names: Vec<String> = Vec::new();
        while let Some(entry) = reader.next_entry().await.expect("dir entry") {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        names
    }

    /// Opaque bytes that are deliberately **neither** a valid OMPF container nor
    /// a valid `TailHeader`-prefixed envelope.
    ///
    /// SR-1.5 makes the adapter an opaque-bytes store: it must never parse what
    /// it persists. Feeding it something unparseable is what turns that from a
    /// claim into a test — an adapter that grew a "validate the envelope" step
    /// (or a "decode the header to recover the lineage" shortcut) fails loudly
    /// here instead of silently coupling the storage layer to the codec.
    fn opaque_non_envelope_bytes() -> Bytes {
        Bytes::from_static(b"\x00\xffnot-an-OMPF-and-not-a-TailHeader\x01\x02\xfe")
    }

    /// A tail put publishes **exactly two** files — the envelope and its
    /// kind-qualified meta companion — and the pair round-trips through
    /// `get_layer_checkpoint` with its bytes and its full meta intact.
    ///
    /// The directory is compared whole rather than probed name by name. The
    /// two absences that carry the contract are invisible to a probe list:
    /// **no `*.ext-refs.json`** (SR-2.4 — a tail-authored sidecar is never
    /// reaped and would pin its hashes in the retention cascade's live set
    /// forever) and **no `.tmp`** (a leaked temp is inert but is also the shape
    /// that would tell us `atomic_write`'s rename never landed).
    ///
    /// The meta assertions cover the lineage triple *and* the identity fields,
    /// because both survive the same JSON round-trip and a tail that lost its
    /// identity restores `obj:hdr` from nothing exactly as a base would.
    #[tokio::test]
    async fn put_layer_checkpoint_with_tail_kind_publishes_only_the_envelope_and_its_meta() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 400;
        let to_seq: u64 = 12;
        let payload = opaque_non_envelope_bytes();

        let meta = CheckpointMeta {
            creator_replica_id: "replica-tail".to_string(),
            written_at_unix_ms: 1_700_000_000_777,
            snap_seq: Some(to_seq),
            session_id: "sess-tail".to_string(),
            layer_name: "layer-tail".to_string(),
            created_by: "dana".to_string(),
            server_ts: 1_700_000_500_777,
            observed_commit_span: 5,
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(4),
            from_seq: Some(9),
            sidecar_bytes: None,
        };

        let handle = store
            .put_layer_checkpoint(oid, to_seq, payload.clone(), meta)
            .await
            .expect("a lineage-complete tail put must succeed");

        assert_eq!(handle.kind, CheckpointKind::Tail);
        assert_eq!(handle.object_id, oid);
        assert_eq!(
            handle.seq(),
            to_seq,
            "a tail handle's ordering number is its to_seq"
        );
        assert_eq!(
            handle.adapter_key,
            store.tail_path(oid, to_seq).to_string_lossy(),
            "the returned key must name the `.tail` envelope, not the base stem at this sequence"
        );

        assert_eq!(
            sorted_layer_names(&store).await,
            vec![
                "400.12.tail".to_string(),
                "400.12.tail.meta.json".to_string()
            ],
            "a tail publishes exactly its envelope and its kind-qualified meta — no \
             `*.ext-refs.json` (SR-2.4: one would pin this stem's hashes in the retention \
             cascade's live set forever, unreapable) and no leaked `.tmp`"
        );

        let blob = store
            .get_layer_checkpoint(&handle)
            .await
            .expect("get must succeed")
            .expect("the tail just written must resolve by its own handle");
        assert_eq!(
            blob.bytes, payload,
            "the envelope is opaque bytes (SR-1.5) and must round-trip byte-identically"
        );
        assert_eq!(
            blob.handle, handle,
            "the handle must round-trip identically"
        );

        // Lineage — what Epic 4's gather keys on. A tail that lost these is
        // unreplayable, which is silent unrecoverability rather than an error.
        assert_eq!(blob.meta.kind, CheckpointKind::Tail);
        assert_eq!(blob.meta.base_snap_seq, Some(4));
        assert_eq!(blob.meta.from_seq, Some(9));
        assert_eq!(blob.meta.snap_seq, Some(to_seq));
        // Identity — what `apply_checkpoint` rebuilds `obj:hdr` from.
        assert_eq!(blob.meta.session_id, "sess-tail");
        assert_eq!(blob.meta.layer_name, "layer-tail");
        assert_eq!(blob.meta.created_by, "dana");
        assert_eq!(blob.meta.server_ts, 1_700_000_500_777);
        assert_eq!(blob.meta.creator_replica_id, "replica-tail");
        assert_eq!(blob.meta.observed_commit_span, 5);
        assert!(
            blob.meta.sidecar_bytes.is_none(),
            "the get path never populates `sidecar_bytes`, and a tail has no sidecar at all"
        );

        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        assert_eq!(
            handles,
            vec![handle],
            "the tail is the object's one and only generation"
        );
    }

    /// A `Tail` meta missing either lineage field is rejected `Permanent` and
    /// writes **nothing** — not even the meta companion the tail branch would
    /// otherwise publish first.
    ///
    /// Admitting one is not a cosmetic laxity: since b.hka (d) `read_generation_at`
    /// synthesizes no placeholder at all — an artifact whose meta companion is
    /// absent is refused outright — so a lineage-less tail meta can now reach a
    /// reader only by having been *stored* that way. Replay would then have to
    /// infer the base, layering the tail's commits onto whatever base happened
    /// to be newest and producing a wrong restore instead of a loud failure.
    /// This gate is what keeps that shape off the disk in the first place.
    ///
    /// Classification is identical to InMemory and (Task 4) StorageApi **by
    /// construction**: all three call the one `validate_tail_put` in
    /// `adapters/mod.rs` before touching their backend, so a divergence would
    /// mean a second, open-coded copy of the rule rather than a differently
    /// tuned one.
    #[tokio::test]
    async fn put_layer_checkpoint_with_lineage_incomplete_tail_is_permanent_and_writes_no_artifact()
    {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");

        let cases: [(&str, CheckpointMeta); 3] = [
            (
                "both lineage fields absent",
                CheckpointMeta {
                    kind: CheckpointKind::Tail,
                    snap_seq: Some(8),
                    ..CheckpointMeta::default()
                },
            ),
            (
                "base_snap_seq absent, from_seq present",
                CheckpointMeta {
                    base_snap_seq: None,
                    ..tail_meta(4, 5, 8)
                },
            ),
            (
                "from_seq absent, base_snap_seq present",
                CheckpointMeta {
                    from_seq: None,
                    ..tail_meta(4, 5, 8)
                },
            ),
        ];

        for (label, incomplete) in cases {
            let err = match store
                .put_layer_checkpoint(13, 8, opaque_non_envelope_bytes(), incomplete)
                .await
            {
                Err(err) => err,
                Ok(handle) => panic!(
                    "a lineage-incomplete tail put ({label}) must be rejected; got {handle:?}"
                ),
            };
            assert!(
                matches!(err, CheckpointError::Permanent { .. }),
                "lineage-incomplete tail put ({label}) must classify as Permanent — a retry of \
                 the identical call reproduces it exactly; got {err:?}"
            );
            assert!(
                err.is_permanent(),
                "({label}) the classification accessor must agree with the variant"
            );
            assert_eq!(
                sorted_layer_names(&store).await,
                Vec::<String>::new(),
                "({label}) the gate runs before any path derivation, so a rejected tail put must \
                 leave the layers directory empty — an orphaned `.tail.meta.json` here would \
                 mean the meta was published before the lineage was checked"
            );
        }
    }

    /// A **stored** tail — lineage-complete, `sidecar_bytes: None` — coexisting
    /// with a base at the same sequence contributes nothing to the retention
    /// cascade's live set and never aliases the base's sidecar.
    ///
    /// Two distinct guarantees, both load-bearing for b.s5y / SR-5.3:
    ///
    /// - `list_referenced_ext_hashes` walks `*.ext-refs.json` **directly**, so
    ///   the guarantee reduces to "a tail never authors one". A tail-authored
    ///   sidecar would be unreapable (every delete path is generation-stem
    ///   driven) and would pin its hashes in the live set forever, degrading the
    ///   cascade toward deleting nothing.
    /// - `read_ext_refs_sidecar` derives its path from the kind-blind
    ///   `sidecar_path(oid, seq)`, so at a *shared* sequence only the kind gate
    ///   in front of it keeps a tail handle from being handed the BASE's hash
    ///   set. That is the aliasing case, and it needs a real base sidecar at the
    ///   same sequence to be non-vacuous — which is why the base's hashes are
    ///   asserted present through the base handle in the same test.
    ///
    /// The sibling `read_ext_refs_sidecar_on_a_tail_handle_returns_empty_and_not_the_bases_hashes`
    /// pins the same arm through a *fabricated* tail handle with no tail on
    /// disk; this drives it through the real tail-storage path.
    #[tokio::test]
    async fn a_stored_tail_never_reaches_the_live_set_and_never_aliases_the_bases_sidecar() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 401;

        let base_meta = CheckpointMeta {
            snap_seq: Some(9),
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        let base = store
            .put_layer_checkpoint(oid, 9, Bytes::from_static(b"base-bytes"), base_meta)
            .await
            .expect("put base");
        let tail = store
            .put_layer_checkpoint(oid, 9, opaque_non_envelope_bytes(), tail_meta(4, 6, 9))
            .await
            .expect("put tail");

        assert_eq!(
            sorted_layer_names(&store).await,
            vec![
                "401.9.ext-refs.json".to_string(),
                "401.9.meta.json".to_string(),
                "401.9.ompf".to_string(),
                "401.9.tail".to_string(),
                "401.9.tail.meta.json".to_string(),
            ],
            "the base's triplet and the tail's pair, and nothing else: the ONE ext-refs artifact \
             belongs to the base"
        );
        assert!(
            !store.layers_dir.join("401.9.tail.ext-refs.json").exists(),
            "the shape a kind-blind sidecar writer would produce must not exist"
        );

        let live = store
            .list_referenced_ext_hashes()
            .await
            .expect("list_referenced_ext_hashes must succeed");
        assert_eq!(
            live,
            HashSet::from(["feedface".to_string()]),
            "the live set is exactly the base sidecars' union; a tail contributes nothing"
        );

        let tail_refs = store
            .read_ext_refs_sidecar(&tail)
            .await
            .expect("tail read must succeed");
        assert!(
            tail_refs.is_empty(),
            "THE aliasing guard: the kind gate must fire before the kind-blind \
             `sidecar_path(oid, seq)` derivation, or a tail handle at a shared sequence reports \
             the BASE's hashes as its own; got {tail_refs:?}"
        );
        let base_refs = store
            .read_ext_refs_sidecar(&base)
            .await
            .expect("base read must succeed");
        assert_eq!(
            base_refs,
            HashSet::from(["feedface".to_string()]),
            "and the base at the SAME sequence still reports them, so the empty tail result \
             above cannot pass vacuously"
        );
    }

    /// Seed one store rooted at `root` with a base triplet at `snap_seq = 9`
    /// (carrying a sidecar) and a tail at `to_seq = 9`. Returns the two handles.
    ///
    /// `base@S` + `tail@to_seq == S` is a live coexistence, not a contrived
    /// one: SR-4.1 re-base mints a fresh base at the newest consumed tail's
    /// `to_seq`, and SR-9.2 pins re-base idempotency at `(object_id, S)`.
    async fn seed_shared_sequence(
        store: &LocalDiskCheckpointStore,
        oid: u64,
    ) -> (CheckpointHandle, CheckpointHandle) {
        let base_meta = CheckpointMeta {
            snap_seq: Some(9),
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        let base = store
            .put_layer_checkpoint(oid, 9, Bytes::from_static(b"base-bytes"), base_meta)
            .await
            .expect("seed base");
        let tail = store
            .put_layer_checkpoint(oid, 9, opaque_non_envelope_bytes(), tail_meta(4, 6, 9))
            .await
            .expect("seed tail");
        (base, tail)
    }

    /// **Same-sequence delete safety, both directions.** With a base triplet at
    /// `snap_seq = 9` and a tail at `to_seq = 9`, deleting either kind must
    /// leave the other's artifacts byte-intact and still fetchable.
    ///
    /// `ompf_path` / `sidecar_path` / `meta_path` are kind-blind, so a tail
    /// delete that reached for them would unlink the whole base triplet at the
    /// shared sequence — silent recovery-point loss, and the exact
    /// stranded-generation shape SR-2.4 / SR-2.5 forbid. The reverse direction
    /// is the same hazard mirrored: a base delete that also swept the `.tail`
    /// stem would drop a lineage segment no enumeration would ever report
    /// missing.
    ///
    /// The tail's meta companion is asserted gone alongside its envelope
    /// (`.tail.meta.json` is NOT left behind). A dangling tail meta would
    /// falsely round-trip into a later re-put at the same `(oid, to_seq)` — the
    /// hazard `delete_layer_checkpoint` documents for the base's `meta.json`,
    /// and worse here, because the stale lineage would name a base that
    /// retention may already have collected.
    ///
    /// Repeating each delete pins the trait's idempotent-on-missing contract on
    /// the tail arm too: both removals are `NotFound`-tolerant, so the second
    /// call is `Ok(())` by construction rather than by a presence check.
    #[tokio::test]
    async fn delete_layer_checkpoint_at_a_shared_sequence_removes_only_the_addressed_kind() {
        // ── Direction 1: delete the tail, the base survives whole ──────────
        {
            let root = tempdir().expect("tempdir");
            let store =
                LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
            let (base, tail) = seed_shared_sequence(&store, 402).await;

            store
                .delete_layer_checkpoint(tail.clone())
                .await
                .expect("deleting the tail must succeed");

            assert_eq!(
                sorted_layer_names(&store).await,
                vec![
                    "402.9.ext-refs.json".to_string(),
                    "402.9.meta.json".to_string(),
                    "402.9.ompf".to_string(),
                ],
                "a tail delete must remove its envelope AND its meta companion, and must touch \
                 NONE of the kind-blind base paths at the shared sequence"
            );

            let base_blob = store
                .get_layer_checkpoint(&base)
                .await
                .expect("base get must succeed")
                .expect("the base at the shared sequence must survive the tail's delete");
            assert_eq!(base_blob.bytes, Bytes::from_static(b"base-bytes"));
            assert_eq!(base_blob.meta.kind, CheckpointKind::Base);
            assert_eq!(
                store
                    .read_ext_refs_sidecar(&base)
                    .await
                    .expect("base sidecar read"),
                HashSet::from(["feedface".to_string()]),
                "the base's ext blobs must still be protected from the retention cascade"
            );

            match store.get_layer_checkpoint(&tail).await {
                Ok(None) => {}
                other => panic!("the deleted tail must resolve to Ok(None); got {other:?}"),
            }
            assert_eq!(
                store
                    .list_layer_checkpoints(402)
                    .await
                    .expect("list")
                    .iter()
                    .map(|h| (h.seq(), h.kind))
                    .collect::<Vec<_>>(),
                vec![(9, CheckpointKind::Base)],
                "only the base remains enumerable"
            );

            store
                .delete_layer_checkpoint(tail)
                .await
                .expect("a second delete of the same tail must be Ok(()) — idempotent on missing");
            assert_eq!(
                sorted_layer_names(&store).await.len(),
                3,
                "and the repeat must not have taken the base with it"
            );
        }

        // ── Direction 2: delete the base, the tail survives ────────────────
        {
            let root = tempdir().expect("tempdir");
            let store =
                LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
            let (base, tail) = seed_shared_sequence(&store, 402).await;

            store
                .delete_layer_checkpoint(base.clone())
                .await
                .expect("deleting the base must succeed");

            assert_eq!(
                sorted_layer_names(&store).await,
                vec!["402.9.tail".to_string(), "402.9.tail.meta.json".to_string(),],
                "a base delete must take its whole triplet and leave the tail's pair alone"
            );

            let tail_blob = store
                .get_layer_checkpoint(&tail)
                .await
                .expect("tail get must succeed")
                .expect("the tail at the shared sequence must survive the base's delete");
            assert_eq!(tail_blob.bytes, opaque_non_envelope_bytes());
            assert_eq!(tail_blob.meta.kind, CheckpointKind::Tail);
            assert_eq!(
                (tail_blob.meta.base_snap_seq, tail_blob.meta.from_seq),
                (Some(4), Some(6)),
                "the surviving tail's lineage must be intact, not the placeholder a missing \
                 `.tail.meta.json` would synthesize"
            );

            match store.get_layer_checkpoint(&base).await {
                Ok(None) => {}
                other => panic!("the deleted base must resolve to Ok(None); got {other:?}"),
            }
            store
                .delete_layer_checkpoint(base)
                .await
                .expect("a second delete of the same base must be Ok(())");
            assert_eq!(
                sorted_layer_names(&store).await.len(),
                2,
                "and the repeat must not have taken the tail with it"
            );
        }
    }

    /// **Residual after a failed envelope write.** The tail branch publishes
    /// meta first, so a failure at the envelope write strands that meta — and
    /// the branch rolls it back, leaving nothing behind.
    ///
    /// Note the rollback deliberately does NOT go through
    /// `rollback_partial_publish`: that helper's guard is a durable-**OMPF**
    /// probe, which at a shared sequence answers a question about the *base*
    /// while deciding the tail's cleanup. This fixture is exactly that
    /// confusion's negative case — a live base triplet sits at the same
    /// sequence, so a tail rollback routed through the guard would find a
    /// durable OMPF, skip itself, and strand the tail meta permanently.
    ///
    /// Since b.hka review round 2 it goes through
    /// `rollback_partial_tail_publish` instead: the same guard, asked at
    /// `tail_path`. That does not change what this case observes, and it adds a
    /// second claim it now covers for free — the probe tests `is_file()`, not
    /// mere existence, so the poisoned **directory** at the envelope path is NOT
    /// read as a durable envelope. Something that is not a regular file is not a
    /// generation, and treating it as one would strand the companion permanently
    /// on exactly the operator mistake most likely to produce it. The other two
    /// probe arms — a real durable envelope, and an unreadable probe — are the
    /// three `rollback_partial_tail_publish_*` cases below.
    ///
    /// The window is opened without a fault-injection seam by the established
    /// `EISDIR` technique: the `.tail` path is pre-created as a **directory**,
    /// so `atomic_write`'s `rename(2)` fails there and nowhere else.
    #[tokio::test]
    async fn put_layer_checkpoint_tail_rolls_back_its_meta_when_the_envelope_write_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 403;

        // A live base triplet at the sequence the tail will collide with.
        let base_meta = CheckpointMeta {
            snap_seq: Some(9),
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        let base = store
            .put_layer_checkpoint(oid, 9, Bytes::from_static(b"base-bytes"), base_meta)
            .await
            .expect("setup base put");

        tokio::fs::create_dir(store.tail_path(oid, 9))
            .await
            .expect("pre-creating a directory at the tail envelope path must succeed");

        let err = store
            .put_layer_checkpoint(oid, 9, opaque_non_envelope_bytes(), tail_meta(4, 6, 9))
            .await
            .expect_err("the envelope write must fail when its path is an existing directory");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a rename onto a directory is not retryable into success; got {err:?}"
        );

        assert_eq!(
            sorted_layer_names(&store).await,
            vec![
                "403.9.ext-refs.json".to_string(),
                "403.9.meta.json".to_string(),
                "403.9.ompf".to_string(),
                // The poisoned path the test itself created.
                "403.9.tail".to_string(),
            ],
            "the tail meta published earlier in this put must be rolled back — a `403.9.tail\
             .meta.json` here means either the rollback was skipped or it was routed through \
             `rollback_partial_publish`, whose durable-OMPF guard the base beside it would trip. \
             No `*.ext-refs.json` is authored on the tail branch under any circumstance, and no \
             `.tmp` survives `atomic_write`'s own cleanup"
        );
        assert!(
            !store.tail_path(oid, 9).is_file(),
            "the envelope write is the one that failed; the poisoned path must still be a \
             directory"
        );

        // The pre-existing base is byte-intact and still whole.
        let blob = store
            .get_layer_checkpoint(&base)
            .await
            .expect("base get must succeed")
            .expect("the base must survive a failed tail publish at its sequence");
        assert_eq!(blob.bytes, Bytes::from_static(b"base-bytes"));
        assert_eq!(blob.meta.kind, CheckpointKind::Base);
        assert_eq!(
            store
                .list_referenced_ext_hashes()
                .await
                .expect("live set read"),
            HashSet::from(["feedface".to_string()]),
            "the failed tail contributes nothing to the live set, and must not have disturbed \
             the base's contribution either"
        );
    }

    // ── `rollback_partial_tail_publish`'s durable-envelope guard ────────────
    //
    // b.hka review round 2. Three cases, driven against the helper DIRECTLY
    // rather than through `put_layer_checkpoint`, and the reason is a real
    // limitation rather than a preference: the only way to fail an envelope
    // write on this adapter without a fault-injection seam is the `EISDIR`
    // technique, which by construction puts something at `tail_path` that is NOT
    // a regular file. The state these cases are about — a durable envelope
    // already at that address — makes `atomic_write`'s `rename(2)` SUCCEED
    // (rename overwrites), so no end-to-end fixture can reach the failure arm
    // with a live envelope in place. Calling the private helper is the local
    // precedent for exactly this situation (see the stem-parser banner above),
    // and the call site's routing is pinned separately by
    // `put_layer_checkpoint_tail_rolls_back_its_meta_when_the_envelope_write_fails`.

    /// Publish a complete tail at `(oid, to_seq)` and return the handle.
    async fn publish_tail(
        store: &LocalDiskCheckpointStore,
        oid: u64,
        to_seq: u64,
    ) -> CheckpointHandle {
        store
            .put_layer_checkpoint(
                oid,
                to_seq,
                opaque_non_envelope_bytes(),
                tail_meta(4, 5, to_seq),
            )
            .await
            .expect("the setup tail put must succeed")
    }

    /// **THE guard.** When a tail envelope is already durable at
    /// `(object_id, to_seq)`, the rollback keeps its `.tail.meta.json` — and the
    /// tail stays fully readable.
    ///
    /// # The producer
    ///
    /// Reachable on a SINGLE node, which is why this is not merely SR-9.2's
    /// multi-leader case: SDD §22.7 step 8's re-put path re-writes the same layer
    /// at the same sequence after a failed cursor stamp, so attempt 2's failed
    /// envelope write would otherwise remove the companion of attempt 1's durable
    /// envelope. Under a shared filesystem the duplicate-leader race produces the
    /// identical state.
    ///
    /// # Why the residual is a lost recovery point, not a degraded read
    ///
    /// Since b.hka (d) an envelope with no `.tail.meta.json` is not served with a
    /// synthesized placeholder — `read_generation_at` refuses it with the typed
    /// `MissingMetaCompanion` `Permanent`, and `checkpoint::lineage`'s selection
    /// steps down PAST it. So the final assertion here is the one that matters:
    /// not merely "the file is still there" but "the generation still reads back
    /// with its lineage intact", which is what a restore actually needs.
    ///
    /// Pre-fix the rollback was an unguarded `best_effort_remove`, so the meta
    /// was deleted, `sorted_layer_names` held only `405.9.tail`, and the
    /// `get_layer_checkpoint` below returned `Err(MissingMetaCompanion)`.
    #[tokio::test]
    async fn rollback_partial_tail_publish_keeps_the_meta_of_a_durable_envelope() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 405;

        let handle = publish_tail(&store, oid, 9).await;
        assert_eq!(
            sorted_layer_names(&store).await,
            vec!["405.9.tail".to_string(), "405.9.tail.meta.json".to_string()],
            "setup: leader A's pair must be complete before the rollback runs"
        );

        // Leader B's failed publish cleaning up after itself.
        store.rollback_partial_tail_publish(oid, 9).await;

        assert_eq!(
            sorted_layer_names(&store).await,
            vec!["405.9.tail".to_string(), "405.9.tail.meta.json".to_string()],
            "THE guard: an envelope is already durable at (405, 9), so the meta companion belongs \
             to IT and must survive. Removing it strips a live tail of its identity"
        );

        let blob = store
            .get_layer_checkpoint(&handle)
            .await
            .expect("the surviving tail must still READ, not just exist")
            .expect("and it must resolve to a generation, not Ok(None)");
        assert_eq!(
            (blob.meta.base_snap_seq, blob.meta.from_seq),
            (Some(4), Some(5)),
            "the observable consequence, and the reason the guard exists: the tail is still a \
             usable recovery point with its lineage intact. Post-b.hka (d) a meta-absent tail is \
             refused with a typed `MissingMetaCompanion` and the restore walk steps PAST it, so a \
             wrong DELETE here costs a recovery point rather than a degraded read"
        );
        assert_eq!(blob.meta.kind, CheckpointKind::Tail);
    }

    /// The negative arm: with no envelope at the address, the orphaned
    /// `.tail.meta.json` IS removed.
    ///
    /// Without this the guard could be satisfied by a helper that never removes
    /// anything — which would leave every torn tail publish stranding a companion
    /// for the reconciliation sweep to reap a grace later, i.e. it would silently
    /// convert the ordinary cleanup into deferred garbage.
    #[tokio::test]
    async fn rollback_partial_tail_publish_removes_the_meta_when_no_envelope_is_durable() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 406;

        // The torn state the rollback exists for: meta written (it is the FIRST
        // write on the tail branch), envelope never landed.
        publish_tail(&store, oid, 9).await;
        tokio::fs::remove_file(store.tail_path(oid, 9))
            .await
            .expect("removing the envelope must succeed");
        assert_eq!(
            sorted_layer_names(&store).await,
            vec!["406.9.tail.meta.json".to_string()],
            "setup: exactly the torn residual — a companion with no envelope"
        );

        store.rollback_partial_tail_publish(oid, 9).await;

        assert!(
            sorted_layer_names(&store).await.is_empty(),
            "with no envelope to protect, the companion is scrubbed and the directory is left \
             clean — the ordinary cleanup, unchanged by the guard. Got {:?}",
            sorted_layer_names(&store).await
        );
    }

    /// Sink for [`capture_warns`].
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Drive `future` under a capturing subscriber and return its value alongside
    /// everything it logged at WARN or above.
    ///
    /// `WithSubscriber` rather than `set_default`: these futures await
    /// filesystem I/O and may resume on a different worker thread, and a
    /// thread-local dispatcher would silently stop capturing at the first
    /// `.await` — the failure mode that makes a log assertion read as "the WARN
    /// was never emitted".
    async fn capture_warns<F, T>(future: F) -> (T, String)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(CapturingWriter(buf.clone()))
            .with_max_level(tracing::Level::WARN)
            // The assertions match plain substrings and ANSI escapes land inside
            // them.
            .with_ansi(false)
            .finish();
        let value = future
            .with_subscriber(tracing::Dispatch::new(subscriber))
            .await;
        let text =
            String::from_utf8_lossy(&buf.lock().expect("capture buffer poisoned")).into_owned();
        (value, text)
    }

    /// The third arm: a probe that cannot answer keeps the companion.
    ///
    /// "We do not know" must read as "an envelope may exist". An implementation
    /// that treated an errored `metadata` call as absence would strip a live
    /// tail's identity the first time the filesystem answered anything other than
    /// `NotFound`. Erring the other way leaves one inert orphan the
    /// reconciliation sweep reaps after the grace.
    ///
    /// # The injection, and why the assertion is on the LOG
    ///
    /// The error is produced by making the layers directory itself a REGULAR
    /// FILE for the duration of the call: the real directory is renamed aside and
    /// a file is written at its path, so every `<layers>/…` lookup fails
    /// `ENOTDIR` (`io::ErrorKind::NotADirectory`) rather than `NotFound` — a
    /// path-resolution error, with no fault-injection seam needed.
    ///
    /// **Deliberately not a cleared `+x` bit.** That was the first shape of this
    /// case and it is not sound in CI: the pipeline's `rust:1.94-slim` container
    /// runs as root, `CAP_DAC_OVERRIDE` bypasses the missing search bit outright,
    /// and `metadata` would answer `Err(NotFound)` — collapsing this case into the
    /// preceding one, failing the precondition assert, and (because that panic
    /// precedes the mode restore) leaving a `0o600` directory the `TempDir`'s
    /// recursive cleanup cannot descend. `ENOTDIR` is not capability-bypassable:
    /// no capability makes a regular file resolvable as a path component, so root
    /// and an unprivileged user see the identical error. Neither is `EACCES`'s
    /// other sibling worth reaching for here — this one needs no `unix` gate at
    /// all, since "a file is not a directory" is how every supported platform
    /// resolves paths.
    ///
    /// It necessarily disables the REMOVAL too — `remove_file` on
    /// `<layers>/407.9.tail.meta.json` resolves the same non-directory component
    /// and fails identically — so the surviving-file assertion the two sibling
    /// cases rely on is **vacuous here**: an unguarded pre-fix rollback would
    /// reach its `remove_file`, fail it with `ENOTDIR`, and leave the companion in
    /// place looking exactly like a correct skip. That is deliberate, and it is
    /// what keeps the WARN pair below discriminating: the pre-fix body reaches
    /// `best_effort_remove` and reports a FAILED REMOVAL, which is a different
    /// log line from a declined decision even though the filesystem looks
    /// identical.
    ///
    /// What distinguishes the two is therefore which WARN was emitted:
    /// [`TAIL_ROLLBACK_PROBE_DECLINED_PHRASE`] (the decision was declined) versus
    /// [`ROLLBACK_REMOVE_FAILED_PHRASE`] (the decision was taken and the removal
    /// was attempted). Both are the production constants, so a reword breaks
    /// compilation instead of silently voiding the negative assertion. The file
    /// check stays as the statement of intent and the log pair is what has teeth.
    ///
    /// Teardown needs no privileged restore: the injected file and the renamed
    /// directory are both ordinary entries under the `TempDir`, so even a panic
    /// mid-case leaves a tree the recursive cleanup can remove.
    #[tokio::test]
    async fn rollback_partial_tail_publish_keeps_the_meta_when_the_probe_cannot_answer() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 407;

        // Same torn residual as the arm above — so a helper that read the error
        // as absence would DECIDE to remove the companion, exactly as it
        // correctly does there. The ONLY difference is that the probe cannot see
        // it.
        publish_tail(&store, oid, 9).await;
        tokio::fs::remove_file(store.tail_path(oid, 9))
            .await
            .expect("removing the envelope must succeed");

        // Swap the layers directory for a regular file. `store` keeps addressing
        // the same path, so every lookup under it now hits a non-directory
        // component.
        let layers_dir = store.layers_dir.clone();
        let stashed = layers_dir.with_extension("stashed");
        tokio::fs::rename(&layers_dir, &stashed)
            .await
            .expect("renaming the layers dir aside must succeed");
        tokio::fs::write(&layers_dir, b"not a directory")
            .await
            .expect("writing a regular file at the layers dir path must succeed");

        let probe = tokio::fs::metadata(store.tail_path(oid, 9)).await;
        assert!(
            probe
                .as_ref()
                .err()
                .is_some_and(|e| e.kind() != io::ErrorKind::NotFound),
            "precondition: the injection must produce a NON-NotFound error, or this case is the \
             preceding one wearing a different name. Got {probe:?}"
        );
        let removal_probe = tokio::fs::remove_file(store.tail_meta_path(oid, 9)).await;
        assert!(
            removal_probe
                .as_ref()
                .err()
                .is_some_and(|e| e.kind() != io::ErrorKind::NotFound),
            "precondition: the removal must fail too, and NOT as `NotFound` — that is what makes \
             the pre-fix body reach `best_effort_remove` and emit the removal-failed WARN the \
             negative assertion below rules out. A removal that succeeded (or reported absence) \
             would leave that assertion trivially true. Got {removal_probe:?}"
        );

        let ((), warns) = capture_warns(store.rollback_partial_tail_publish(oid, 9)).await;

        tokio::fs::remove_file(&layers_dir)
            .await
            .expect("removing the injected regular file must succeed");
        tokio::fs::rename(&stashed, &layers_dir)
            .await
            .expect("restoring the layers dir must succeed");

        assert!(
            warns.contains(super::TAIL_ROLLBACK_PROBE_DECLINED_PHRASE),
            "THE discriminating assertion: the rollback must report that it DECLINED the \
             decision. An implementation that read the error as absence would instead reach \
             `best_effort_remove` and report a failed removal — indistinguishable from this on \
             the filesystem, because the same non-directory component blocks both. Captured: \
             {warns}"
        );
        assert!(
            !warns.contains(super::ROLLBACK_REMOVE_FAILED_PHRASE),
            "and specifically NOT the removal-failed WARN, which is the pre-fix signature: the \
             removal was attempted and only the filesystem stopped it. Captured: {warns}"
        );
        assert_eq!(
            sorted_layer_names(&store).await,
            vec!["407.9.tail.meta.json".to_string()],
            "and the companion is of course still there — stated for completeness, though on this \
             fixture alone it cannot discriminate (see the doc-comment): an unnecessary orphan is \
             recoverable, the reconciliation sweep classifies it as a tail-family orphan candidate \
             and reaps it after the grace, while deleting a live tail's meta is not"
        );
    }

    /// **Residual after a failed meta write.** The tail branch's meta is its
    /// FIRST write, so a failure there aborts before any `.tail` can exist:
    /// the residual class for a meta-first publish is an orphaned
    /// `.tail.meta.json`, never an orphaned envelope.
    ///
    /// This is the tail's counterpart to
    /// `put_layer_checkpoint_aborts_before_any_artifact_when_the_sidecar_write_fails`,
    /// and it is what makes the publish *order* assertable: only a meta-first
    /// implementation leaves the envelope unwritten when the meta is the thing
    /// that fails. Envelope-first would open a window in which a concurrent
    /// `list_layer_checkpoints` — which is `.tail`-**stem**-driven — reports a
    /// generation whose meta is absent, forcing the read path to treat a torn
    /// residual as an ordinary state.
    ///
    /// It deliberately does NOT inherit the base path's
    /// durable-artifact-is-never-rolled-back asymmetry: on this branch there is
    /// no durable artifact to preserve at the failing step.
    #[tokio::test]
    async fn put_layer_checkpoint_tail_aborts_before_the_envelope_when_its_meta_write_fails() {
        let root = tempdir().expect("tempdir");
        let store = LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
        let oid: u64 = 404;

        let base_meta = CheckpointMeta {
            snap_seq: Some(9),
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        let base = store
            .put_layer_checkpoint(oid, 9, Bytes::from_static(b"base-bytes"), base_meta)
            .await
            .expect("setup base put");

        tokio::fs::create_dir(store.tail_meta_path(oid, 9))
            .await
            .expect("pre-creating a directory at the tail meta path must succeed");

        let err = store
            .put_layer_checkpoint(oid, 9, opaque_non_envelope_bytes(), tail_meta(4, 6, 9))
            .await
            .expect_err("the tail meta write must fail when its path is an existing directory");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "expected Permanent, got {err:?}"
        );

        assert_eq!(
            sorted_layer_names(&store).await,
            vec![
                "404.9.ext-refs.json".to_string(),
                "404.9.meta.json".to_string(),
                "404.9.ompf".to_string(),
                // The poisoned path the test itself created.
                "404.9.tail.meta.json".to_string(),
            ],
            "the meta is the tail branch's FIRST write, so its failure must abort before any \
             `404.9.tail` exists. An envelope here means the adapter reverted to envelope-first \
             publish, which lets a stem-driven listing report a tail with no meta"
        );
        assert!(
            !store.tail_path(oid, 9).exists(),
            "no envelope of any kind — not even a leftover temp renamed into place"
        );

        let blob = store
            .get_layer_checkpoint(&base)
            .await
            .expect("base get must succeed")
            .expect("the base must survive a failed tail publish at its sequence");
        assert_eq!(blob.bytes, Bytes::from_static(b"base-bytes"));
        assert_eq!(
            store
                .list_referenced_ext_hashes()
                .await
                .expect("live set read"),
            HashSet::from(["feedface".to_string()]),
            "no tail-authored `*.ext-refs.json` joined the live set, and the base's is untouched"
        );
    }

    /// **Durability across a reopen.** A second `LocalDiskCheckpointStore` over
    /// the same root must report the tail with its kind and lineage intact.
    ///
    /// This is the genuinely durable kind-round-trip, and it can only live here:
    /// the cross-adapter conformance harness's factory owns its tempdir
    /// privately, so it cannot close and reopen a store, and an in-process
    /// assertion cannot distinguish "re-read from `.tail.meta.json`" from
    /// "answered out of state this store happened to retain". The reopened
    /// store shares nothing with the first but the bytes on disk.
    ///
    /// A base at a different sequence rides along so the reopen also proves the
    /// mixed-kind listing is rebuilt from the directory rather than from a
    /// write-time cache.
    #[tokio::test]
    async fn a_tail_survives_a_reopen_of_the_store_over_the_same_root() {
        let root = tempdir().expect("tempdir");
        let oid: u64 = 405;
        let payload = opaque_non_envelope_bytes();

        {
            let store =
                LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("constructor");
            store
                .put_layer_checkpoint(oid, 4, Bytes::from_static(b"base-bytes"), test_meta())
                .await
                .expect("put base");
            store
                .put_layer_checkpoint(oid, 7, payload.clone(), tail_meta(4, 4, 7))
                .await
                .expect("put tail");
        }

        let reopened =
            LocalDiskCheckpointStore::new(root.path().to_path_buf()).expect("reopen constructor");
        let handles = reopened
            .list_layer_checkpoints(oid)
            .await
            .expect("list after reopen");
        assert_eq!(
            handles
                .iter()
                .map(|h| (h.seq(), h.kind))
                .collect::<Vec<_>>(),
            vec![(7, CheckpointKind::Tail), (4, CheckpointKind::Base)],
            "the mixed-kind listing must be rebuilt from the directory, kinds included"
        );

        let blob = reopened
            .get_layer_checkpoint(&handles[0])
            .await
            .expect("get after reopen")
            .expect("the tail must resolve through a store that never wrote it");
        assert_eq!(blob.bytes, payload, "the envelope bytes are durable");
        assert_eq!(blob.handle.kind, CheckpointKind::Tail);
        assert_eq!(
            (blob.meta.kind, blob.meta.base_snap_seq, blob.meta.from_seq),
            (CheckpointKind::Tail, Some(4), Some(4)),
            "kind and lineage must come back off `.tail.meta.json`; the lineage-less placeholder \
             `read_generation_at` synthesizes for a meta-absent tail would report \
             (Tail, None, None) here"
        );
    }

    /// Every tail-reachable method consults `maybe_inject` at the top of its
    /// body, under the **same** op label the base path uses — the label names
    /// the *method*, not the *kind* (`adapters/mod.rs`, "Fault-injection label
    /// policy").
    ///
    /// This is what lets Task 6's conformance harness drive a tail through
    /// `Transient` / `Conflict` at all: a tail branch that grew its own
    /// early-return ahead of the injection check, or that answered under a
    /// kind-qualified label, would be silently undrivable — the harness would
    /// observe a success and report a passing case.
    ///
    /// `Conflict` is the injected classification precisely because no LocalDisk
    /// path produces it natively (every real filesystem error here is
    /// `Permanent`), so it cannot be confused with a genuine verdict. The put
    /// case additionally asserts the *directory*, since the point of injecting
    /// at the top of the body is that no bytes are written at all.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test]
    async fn tail_paths_consult_the_injector_under_the_shared_op_labels() {
        /// Fails exactly one op label and lets every other call through, so a
        /// store can be seeded through the real paths before the op under test
        /// is exercised.
        struct FailOne {
            op: &'static str,
        }
        impl CheckpointFaultInjector for FailOne {
            fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
                if op == self.op {
                    return Err(CheckpointError::Conflict);
                }
                Ok(())
            }
        }

        fn store_failing(root: &Path, op: &'static str) -> LocalDiskCheckpointStore {
            LocalDiskCheckpointStore::new(root.to_path_buf())
                .expect("constructor")
                .with_fault_injector(Arc::new(FailOne { op }))
        }

        // --- put_layer_checkpoint: the tail put short-circuits, and not one
        // --- byte reached the layers directory.
        {
            let root = tempdir().expect("tempdir");
            let store = store_failing(root.path(), "put_layer_checkpoint");
            let err = store
                .put_layer_checkpoint(80, 5, opaque_non_envelope_bytes(), tail_meta(2, 3, 5))
                .await
                .expect_err("the injector must short-circuit the tail put");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the injected classification must surface verbatim; got {err:?}"
            );
            assert_eq!(
                sorted_layer_names(&store).await,
                Vec::<String>::new(),
                "injection happens at the TOP of the method, ahead of the meta write, so a \
                 short-circuited tail put must leave no `.tail.meta.json` behind"
            );
            assert!(
                store
                    .put_layer_checkpoint(80, 5, Bytes::from_static(b"base"), test_meta())
                    .await
                    .is_err(),
                "the base put must answer to the SAME op label as the tail put — a \
                 kind-qualified label would make the conformance harness's injection cases miss \
                 one of the two branches"
            );
        }

        // --- get_layer_checkpoint on a tail handle.
        {
            let root = tempdir().expect("tempdir");
            let store = store_failing(root.path(), "get_layer_checkpoint");
            let (base, tail) = seed_shared_sequence(&store, 80).await;
            let err = store
                .get_layer_checkpoint(&tail)
                .await
                .expect_err("the injector must short-circuit the tail get");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the tail get must surface the injected error rather than Ok(None); got {err:?}"
            );
            assert!(
                store.get_layer_checkpoint(&base).await.is_err(),
                "the base get must answer to the same op label"
            );
        }

        // --- delete_layer_checkpoint on a tail handle: short-circuits, and
        // --- both tail files are still on disk afterwards.
        {
            let root = tempdir().expect("tempdir");
            let store = store_failing(root.path(), "delete_layer_checkpoint");
            let (base, tail) = seed_shared_sequence(&store, 80).await;
            let err = store
                .delete_layer_checkpoint(tail)
                .await
                .expect_err("the injector must short-circuit the tail delete");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the tail delete must surface the injected error rather than Ok(()); got {err:?}"
            );
            assert_eq!(
                sorted_layer_names(&store).await,
                vec![
                    "80.9.ext-refs.json".to_string(),
                    "80.9.meta.json".to_string(),
                    "80.9.ompf".to_string(),
                    "80.9.tail".to_string(),
                    "80.9.tail.meta.json".to_string(),
                ],
                "a short-circuited delete must have unlinked nothing — the injector fires ahead \
                 of the first `remove_file`"
            );
            assert!(
                store.delete_layer_checkpoint(base).await.is_err(),
                "the base delete must answer to the same op label"
            );
        }

        // --- list_layer_checkpoints over a mixed-kind object.
        {
            let root = tempdir().expect("tempdir");
            let store = store_failing(root.path(), "list_layer_checkpoints");
            seed_shared_sequence(&store, 80).await;
            let err = store
                .list_layer_checkpoints(80)
                .await
                .expect_err("the injector must short-circuit the mixed-kind listing");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the listing must surface the injected error rather than an empty Vec; got \
                 {err:?}"
            );
        }

        // --- list_all_object_ids over a tail-only object: the enumeration a
        // --- tail-unaware implementation would have answered from an
        // --- `*.ompf`-only walk the injector never guards.
        {
            let root = tempdir().expect("tempdir");
            let store = store_failing(root.path(), "list_all_object_ids");
            store
                .put_layer_checkpoint(81, 9, opaque_non_envelope_bytes(), tail_meta(1, 2, 9))
                .await
                .expect("seed tail-only object");
            let err = store
                .list_all_object_ids()
                .await
                .expect_err("the injector must short-circuit the enumeration");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the enumeration must surface the injected error; got {err:?}"
            );
        }
    }
}
