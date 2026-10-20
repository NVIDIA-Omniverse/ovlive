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

//! Process-local in-memory [`CheckpointStore`](super::super::CheckpointStore)
//! adapter.
//!
//! Tests and single-replica dev only — state is lost on process exit. Backs
//! every integration test that needs a `CheckpointStore` but does not care
//! about durability, and serves as the simplest reference implementation
//! whose behavior pins the trait's semantic contract for the SR-1.3
//! conformance harness (Task `t2.fgj.ya.74`).
//!
//! Per SDD SR-1.5 the adapter is opaque-bytes-in / opaque-bytes-out: no OMPF
//! parsing, no version filtering, no business logic. All inner state lives
//! behind a single `tokio::sync::Mutex` so async trait methods stay
//! cancellation-safe (a `std::sync::Mutex` would hold its guard across an
//! `.await` and block the runtime under contention).
//!
//! ## Sidecar handling
//!
//! The `CheckpointStore` trait's `put_layer_checkpoint` accepts only the OMPF
//! bytes — the ext-refs sidecar is not part of the trait signature in
//! Epic 1. The cross-task contract (`t2.fgj.ya.r7` Task body, "Sidecar
//! contract") delegates write ordering of the sidecar to the assembler Task
//! and asks adapters to expose put/get primitives that make the pair
//! possible. This adapter stores the sidecar bytes alongside the OMPF in
//! [`LayerEntry`] and exposes [`InMemoryCheckpointStore::set_layer_sidecar`]
//! as the adapter-specific (non-trait) mutation point. Until the assembler
//! Task wires this in, `list_referenced_ext_hashes` returns an empty set
//! when no sidecars have been populated — which is safe (orphan sweep
//! degrades to a no-op, never deletes a referenced blob).
//!
//! ## Tail generations (SR-2.3 / SR-2.4)
//!
//! A base and a tail genuinely coexist at the same sequence — SR-4.1 re-base
//! mints a fresh base at `S` = the newest consumed tail's `to_seq`, and SR-9.2
//! pins re-base idempotency at `(object_id, S)` — so `(object_id, snap_seq)`
//! alone is not a generation's identity here. The inner map's value is a
//! [`GenerationSlot`] holding an independent `base` and `tail` entry, keeping
//! `snap_seq` the sole ordering key while making `kind` part of the address.
//! Every layer path in this file is therefore one of three shapes: kind-aware
//! (`put` / `get` / `delete` / `list_layer_checkpoints` /
//! `list_layer_artifacts`), base-only *by decision*
//! (`list_referenced_ext_hashes`, `set_layer_sidecar` — a tail carries no
//! ext-refs sidecar at all, SR-2.4), or kind-agnostic *by construction*
//! (`list_all_object_ids`). Each says which it is at its own definition.

use std::collections::{BTreeMap, HashMap, HashSet};

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use tokio::sync::Mutex;

use super::super::store::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta,
    CheckpointStore, LayerArtifact, LayerArtifactKind, LiveExtSet,
};
use super::{handle_generation_sort_key, validate_generation_identity, validate_tail_put};

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::CheckpointFaultInjector;
#[cfg(feature = "test-checkpoint-fault-injection")]
use std::sync::Arc;

/// A single layer-state checkpoint generation: the OMPF blob the writer
/// persisted, the optional ext-refs sidecar paired with it, and the
/// provenance metadata.
#[derive(Debug, Clone)]
struct LayerEntry {
    ompf: Bytes,
    /// Ext-refs sidecar JSON bytes. `None` until [`InMemoryCheckpointStore::set_layer_sidecar`]
    /// is called by the assembler (or test). Production callers in Epic 1 do
    /// not exercise this path because the assembler is still under
    /// construction; tests set it directly to exercise
    /// [`list_referenced_ext_hashes`].
    sidecar: Option<Bytes>,
    meta: CheckpointMeta,
}

/// The two generations that can share one `(object_id, snap_seq)` address: a
/// full-state base and a tail envelope whose `to_seq` equals that sequence.
///
/// # Why a slot rather than a `(u64, CheckpointKind)` composite key
///
/// A composite key needs `Ord` on [`CheckpointKind`], and a derived `Ord`
/// (`Base < Tail`) reversed for the newest-first listing yields Tail-before-Base
/// at a tie — the exact opposite of the tie-break pinned by
/// [`generation_sort_key`](super::generation_sort_key). The slot keeps
/// `snap_seq` the sole ordering key and moves the tie-break to the one place
/// that emits it ([`InMemoryCheckpointStore::list_layer_checkpoints`]), where it
/// is reviewable instead of being an emergent property of a derive.
#[derive(Debug, Clone, Default)]
struct GenerationSlot {
    base: Option<LayerEntry>,
    tail: Option<LayerEntry>,
}

impl GenerationSlot {
    /// The entry stored under `kind`, if any — the single place the
    /// kind-to-field mapping is spelled out on the read path.
    fn entry(&self, kind: CheckpointKind) -> Option<&LayerEntry> {
        match kind {
            CheckpointKind::Base => self.base.as_ref(),
            CheckpointKind::Tail => self.tail.as_ref(),
        }
    }

    /// Neither kind present. `delete_layer_checkpoint` prunes the sequence key
    /// on this, which is what keeps `list_layer_checkpoints` from emitting a
    /// sequence with nothing under it.
    fn is_vacant(&self) -> bool {
        self.base.is_none() && self.tail.is_none()
    }
}

/// This adapter's canonical `adapter_key` for one generation.
///
/// The base spelling is unchanged from Epic 1 (`"{oid}:{seq}"`) — conformance
/// asserts the exact bytes round-trip — and a tail appends `:tail` so the key is
/// unique per `(object_id, sequence, kind)`, as `get_layer_checkpoint`'s
/// contract requires. This is the in-memory analogue of the cross-adapter
/// `{oid}.{to_seq}.tail` artifact stem.
fn adapter_key_for(object_id: u64, seq: u64, kind: CheckpointKind) -> String {
    match kind {
        CheckpointKind::Base => format!("{object_id}:{seq}"),
        CheckpointKind::Tail => format!("{object_id}:{seq}:tail"),
    }
}

/// The handle this adapter mints for one generation: the right constructor for
/// `kind` paired with [`adapter_key_for`]. Every layer path returns handles
/// through here so `kind` and `adapter_key` cannot disagree.
fn handle_for(object_id: u64, seq: u64, kind: CheckpointKind) -> CheckpointHandle {
    let adapter_key = adapter_key_for(object_id, seq, kind);
    match kind {
        CheckpointKind::Base => CheckpointHandle::base(object_id, seq, adapter_key),
        CheckpointKind::Tail => CheckpointHandle::tail(object_id, seq, adapter_key),
    }
}

/// A single session-state checkpoint: the opaque `SessionCheckpoint` body
/// (per SR-1.5 the adapter does not parse it) plus the supplied provenance
/// metadata. Mirrors [`LayerEntry`] minus the ext-refs sidecar — session
/// checkpoints carry their identity inside the body envelope, not in a
/// sidecar.
#[derive(Debug, Clone)]
struct SessionEntry {
    body: Bytes,
    meta: CheckpointMeta,
}

/// Inner state guarded by the `tokio::sync::Mutex`.
#[derive(Debug, Default)]
struct InMemoryState {
    /// `object_id` -> ordered `snap_seq` -> [`GenerationSlot`]. `BTreeMap`
    /// gives us natural ascending iteration, which we reverse for newest-first
    /// listing; the slot carries the base and the tail that can share one
    /// sequence. Holds **both** kinds deliberately — a separate tail map would
    /// have silently broken `list_all_object_ids`.
    layer_checkpoints: HashMap<u64, BTreeMap<u64, GenerationSlot>>,
    /// `session_id` -> entry. Last-writer-wins per the trait contract; the
    /// shared `tokio::sync::Mutex` keeps `put` / `get` / `delete` paths
    /// cancellation-safe alongside the layer-checkpoint paths.
    session_checkpoints: HashMap<String, SessionEntry>,
    /// Content-addressed ext-blob store.
    ext_blobs: HashMap<String, Bytes>,
    /// Singleton counters sidecar; last-writer-wins per the trait contract.
    counters_sidecar: Option<Bytes>,
}

/// Process-local in-memory checkpoint store.
///
/// Construct via [`InMemoryCheckpointStore::new`]; share across tasks via
/// `Arc<dyn CheckpointStore>` (the type is `Send + Sync + 'static`).
pub struct InMemoryCheckpointStore {
    state: Mutex<InMemoryState>,
    #[cfg(feature = "test-checkpoint-fault-injection")]
    fault_injector: Option<Arc<dyn CheckpointFaultInjector>>,
}

impl InMemoryCheckpointStore {
    /// Construct a fresh, empty in-memory checkpoint store.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(InMemoryState::default()),
            #[cfg(feature = "test-checkpoint-fault-injection")]
            fault_injector: None,
        }
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

    /// Adapter-specific (non-trait) sidecar setter.
    ///
    /// The Epic-1 `CheckpointStore` trait does not carry the ext-refs
    /// sidecar through `put_layer_checkpoint`; per the Task `t2.fgj.ya.r7`
    /// Cross-Task contract the assembler owns the write order and pairs the
    /// OMPF with the sidecar. This method is the adapter primitive the
    /// assembler (and unit tests) call to attach the sidecar bytes to an
    /// already-put layer checkpoint.
    ///
    /// No-op when no layer checkpoint exists at `(object_id, snap_seq)` —
    /// the assembler is expected to call this only after a successful
    /// `put_layer_checkpoint`; tests that exercise
    /// `list_referenced_ext_hashes` rely on the same ordering.
    ///
    /// **Base slot only, and there is deliberately no tail equivalent.** The
    /// signature is kind-blind (it predates tails), so this pins the target
    /// rather than leaving it to whichever generation happens to be at the
    /// sequence: a tail carries no ext-refs sidecar by design (SR-2.4, b.s5y
    /// MUST-level), and `validate_tail_put` rejects a tail put that supplies
    /// one. A `set_layer_sidecar_tail` would be a way to smuggle past that
    /// rejection, so it does not exist. A call naming a sequence that holds only
    /// a tail is a no-op, same as one naming a sequence that holds nothing.
    pub async fn set_layer_sidecar(&self, object_id: u64, snap_seq: u64, sidecar: Bytes) {
        let mut state = self.state.lock().await;
        if let Some(per_object) = state.layer_checkpoints.get_mut(&object_id) {
            if let Some(entry) = per_object.get_mut(&snap_seq).and_then(|s| s.base.as_mut()) {
                entry.sidecar = Some(sidecar);
            }
        }
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
}

impl Default for InMemoryCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Shape of the ext-refs sidecar JSON the assembler emits, per the Task 78
/// cross-contract. Parsed on demand inside `list_referenced_ext_hashes`
/// because storage is opaque bytes (SR-1.5) — the adapter does not interpret
/// the sidecar at put time.
#[derive(Debug, Deserialize)]
struct ExtRefsSidecar {
    ext_hashes: Vec<String>,
}

/// The map's generations as artifacts, both families — the body of
/// `list_layer_artifacts`, shared with `list_live_ext_set` so the live-set
/// walk's same-request count is exactly what `list_layer_artifacts` would
/// report from the state the caller has locked (b.oya). See
/// `list_layer_artifacts` for the emission rules.
fn layer_artifacts_of(
    layer_checkpoints: &HashMap<u64, BTreeMap<u64, GenerationSlot>>,
) -> Vec<LayerArtifact> {
    let mut artifacts: Vec<LayerArtifact> = Vec::new();
    for (object_id, per_object) in layer_checkpoints.iter() {
        for (snap_seq, slot) in per_object.iter() {
            if let Some(entry) = slot.base.as_ref() {
                artifacts.push(LayerArtifact {
                    object_id: *object_id,
                    snap_seq: *snap_seq,
                    kind: LayerArtifactKind::Ompf,
                });
                // Meta is held inline on every entry, so it is always present
                // alongside the OMPF.
                artifacts.push(LayerArtifact {
                    object_id: *object_id,
                    snap_seq: *snap_seq,
                    kind: LayerArtifactKind::Meta,
                });
                if entry.sidecar.is_some() {
                    artifacts.push(LayerArtifact {
                        object_id: *object_id,
                        snap_seq: *snap_seq,
                        kind: LayerArtifactKind::ExtRefs,
                    });
                }
            }
            if slot.tail.is_some() {
                artifacts.push(LayerArtifact {
                    object_id: *object_id,
                    snap_seq: *snap_seq,
                    kind: LayerArtifactKind::TailEnvelope,
                });
                // Same inline-meta reasoning as the base block: a tail
                // entry's `CheckpointMeta` rides the entry, so the tail's
                // meta companion is present exactly when its envelope is.
                artifacts.push(LayerArtifact {
                    object_id: *object_id,
                    snap_seq: *snap_seq,
                    kind: LayerArtifactKind::TailMeta,
                });
            }
        }
    }
    artifacts
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    /// Persist one generation, kind-aware (SR-2.3).
    ///
    /// `(object_id, snap_seq, meta.kind)` is the whole key: last-writer-wins
    /// **per kind**, so a base put never clobbers the tail at the same sequence
    /// and vice versa. The returned handle's `kind` equals `meta.kind`.
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // First statement, ahead of validation and any kind branching — every
        // sibling method gives the injector the same short-circuit-before-real-
        // work position, so a harness that fails a put sees the injected error
        // rather than a validation error that ran first.
        self.maybe_inject("put_layer_checkpoint")?;
        // The shared put-time gate (lineage complete, no ext-refs sidecar on a
        // tail). Lives in `adapters/mod.rs` so all three adapters enforce one
        // copy of the rule; see its doc for why each rejection is `Permanent`.
        validate_tail_put(&meta)?;
        // Its sibling gate, same site and same posture: no generation of either
        // kind is persisted with an empty `session_id`/`layer_name`, because
        // the restore path refuses one. Store-side backstop for the assembler's
        // own refusal on the vanished-`obj:hdr` arm; no slot is created.
        validate_generation_identity(&meta)?;
        let kind = meta.kind;
        let mut state = self.state.lock().await;
        let slot = state
            .layer_checkpoints
            .entry(object_id)
            .or_default()
            .entry(snap_seq)
            .or_default();
        match kind {
            // Last-writer-wins on identical (object_id, snap_seq, kind) per
            // SR-1.3 test #9 — assigning the slot field replaces the prior
            // entry, and the previous sidecar (if any) is dropped along with it
            // so the new entry starts fresh. When the caller threads the sidecar
            // through `meta.sidecar_bytes` (the assembler path) we attach it
            // atomically; the legacy `set_layer_sidecar` path stays available
            // for callers (tests) that put-then-set.
            CheckpointKind::Base => {
                let sidecar = meta.sidecar_bytes.clone();
                slot.base = Some(LayerEntry {
                    ompf,
                    sidecar,
                    meta,
                });
            }
            // `sidecar: None` unconditionally. `validate_tail_put` has already
            // rejected a `Some`, so this is belt-and-braces rather than a silent
            // drop — spelled out so a later reader does not "fix" it into
            // `meta.sidecar_bytes.clone()` and re-open the SR-2.4 hole (a
            // tail-authored sidecar is never reaped and pins its hashes in the
            // live set forever).
            CheckpointKind::Tail => {
                slot.tail = Some(LayerEntry {
                    ompf,
                    sidecar: None,
                    meta,
                });
            }
        }
        Ok(handle_for(object_id, snap_seq, kind))
    }

    /// The newest generation of **either kind** — deliberately not "the newest
    /// base".
    ///
    /// The trait pins `get_latest == list_layer_checkpoints(oid).first()`
    /// verbatim in the tail era, so when the newest generation is a tail this
    /// MUST return that tail. Filtering to bases here is the tempting and wrong
    /// change: SR-2.5's "newest base wins" is **restore-path policy**,
    /// implemented once by kind-aware selection at the recovery hook (Task
    /// `t2.dfh.gy.4o`: list → newest `Base` handle → `get_layer_checkpoint`),
    /// not three times adapter-side where it can only be tested three times.
    ///
    /// Agreement with `list_layer_checkpoints().first()` is by construction, not
    /// by coincidence: both walk the `BTreeMap` newest-sequence-first and both
    /// prefer `base` over `tail` within a sequence, which is the pinned
    /// Base-before-Tail tie-break.
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_latest_layer_checkpoint")?;
        let state = self.state.lock().await;
        let Some(per_object) = state.layer_checkpoints.get(&object_id) else {
            return Ok(None);
        };
        // Descending by sequence, taking the first slot that holds anything.
        // `delete_layer_checkpoint` prunes a vacated slot, so in practice the
        // highest sequence always yields — but skipping rather than stopping
        // keeps the `list().first()` identity true even if a vacant slot ever
        // survived, instead of making the identity depend on that prune.
        let found = per_object.iter().rev().find_map(|(seq, slot)| {
            // Base before Tail at an equal sequence.
            slot.base
                .as_ref()
                .map(|entry| (*seq, CheckpointKind::Base, entry))
                .or_else(|| {
                    slot.tail
                        .as_ref()
                        .map(|entry| (*seq, CheckpointKind::Tail, entry))
                })
        });
        let Some((seq, kind, entry)) = found else {
            return Ok(None);
        };
        Ok(Some(CheckpointBlob {
            handle: handle_for(object_id, seq, kind),
            bytes: entry.ompf.clone(),
            meta: entry.meta.clone(),
        }))
    }

    /// Fetch one specific generation, addressed by handle (SR-2.3).
    ///
    /// `kind` is part of the match key, not a hint: at an equal sequence a base
    /// and a tail are two different artifacts, and each resolves only to its own
    /// slot field.
    ///
    /// Addressed off `(object_id, snap_seq, kind)`; `adapter_key` is opaque
    /// carrier state and is deliberately ignored, the same stance
    /// [`Self::read_ext_refs_sidecar`] takes — a caller that synthesized the
    /// handle from `list_layer_checkpoints` (or fabricated one with a junk key)
    /// must still resolve, and the returned handle carries this adapter's own
    /// canonical key.
    ///
    /// Bytes and meta are cloned verbatim, `sidecar_bytes` included: this
    /// adapter's documented behaviour is in-memory-verbatim round-tripping, so
    /// unlike the two persisting adapters there is no serialize/deserialize step
    /// that could drop the field.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_layer_checkpoint")?;
        let state = self.state.lock().await;
        let Some(per_object) = state.layer_checkpoints.get(&handle.object_id) else {
            // Absent object / generation is `Ok(None)`, never `NotFound` — the
            // trait's get-path rule, which also covers the concurrent-prune
            // race where a listing observed this handle and retention removed
            // the entry before this call landed.
            return Ok(None);
        };
        // `seq()` rather than `snap_seq` because this is kind-neutral code: the
        // number is a base's `snap_seq` or a tail envelope's `to_seq`.
        let Some(slot) = per_object.get(&handle.seq()) else {
            return Ok(None);
        };
        // Three distinct absences collapse to `Ok(None)`: unknown object,
        // unknown sequence, and a sequence that holds the *other* kind. The
        // third is what stops a tail handle from resolving to the base sharing
        // its sequence — a caller would otherwise be handed a full snapshot
        // labelled as a tail.
        let Some(entry) = slot.entry(handle.kind) else {
            return Ok(None);
        };
        Ok(Some(CheckpointBlob {
            handle: handle_for(handle.object_id, handle.seq(), handle.kind),
            bytes: entry.ompf.clone(),
            meta: entry.meta.clone(),
        }))
    }

    /// Every generation of the object, newest-first, kinds interleaved.
    ///
    /// One strict descending order over the sequence, **not** two segregated
    /// runs (SR-2.3): a sequence holding both kinds emits its base then its
    /// tail. That is
    /// [`generation_sort_key`](super::generation_sort_key)'s order realized
    /// structurally — `BTreeMap` iteration supplies the `Reverse(sequence)`
    /// component and the base-then-tail emission supplies the kind rank — so the
    /// result is identical to sorting the handles by
    /// [`handle_generation_sort_key`](super::handle_generation_sort_key)
    /// without paying for a sort. The `debug_assert` below states that
    /// equivalence executably rather than as a comment, so the structural
    /// emission cannot drift from the shared helper the other two adapters sort
    /// with.
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.maybe_inject("list_layer_checkpoints")?;
        let state = self.state.lock().await;
        let Some(per_object) = state.layer_checkpoints.get(&object_id) else {
            return Ok(Vec::new());
        };
        // `BTreeMap` iter is ascending; reverse for newest-first per the
        // trait's `list_layer_checkpoints` contract.
        let mut handles: Vec<CheckpointHandle> = Vec::new();
        for (seq, slot) in per_object.iter().rev() {
            if slot.base.is_some() {
                handles.push(handle_for(object_id, *seq, CheckpointKind::Base));
            }
            if slot.tail.is_some() {
                handles.push(handle_for(object_id, *seq, CheckpointKind::Tail));
            }
        }
        debug_assert!(
            {
                let mut sorted = handles.clone();
                sorted.sort_unstable_by_key(handle_generation_sort_key);
                sorted == handles
            },
            "structural emission must equal the shared generation_sort_key order"
        );
        Ok(handles)
    }

    /// Remove exactly the generation the handle names — `kind` included.
    ///
    /// Clearing only the slot field named by `handle.kind` is load-bearing: a
    /// tail at `to_seq == N` must never take out the base at `snap_seq == N`
    /// (nor the reverse), which is the stranded-generation shape SR-2.4 / SR-2.5
    /// forbid. The two-level prune below — vacated slot, then emptied per-object
    /// map — is what keeps `list_all_object_ids` honest.
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_layer_checkpoint")?;
        let mut state = self.state.lock().await;
        if let Some(per_object) = state.layer_checkpoints.get_mut(&handle.object_id) {
            if let Some(slot) = per_object.get_mut(&handle.seq()) {
                match handle.kind {
                    CheckpointKind::Base => slot.base = None,
                    CheckpointKind::Tail => slot.tail = None,
                }
                // Prune the sequence key once neither kind remains, so a
                // sequence never lingers with nothing under it.
                if slot.is_vacant() {
                    per_object.remove(&handle.seq());
                }
            }
            // …and prune the object once its last sequence is gone, so a
            // fully-deleted object stops surfacing from `list_all_object_ids`
            // and falsely fanning out the recovery hook.
            if per_object.is_empty() {
                state.layer_checkpoints.remove(&handle.object_id);
            }
        }
        // Idempotent on missing handle per trait contract — an already-absent
        // object, sequence, or kind falls through to `Ok(())` without an error.
        Ok(())
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.maybe_inject("put_session_checkpoint")?;
        let mut state = self.state.lock().await;
        // Last-writer-wins per the trait contract; the shared mutex keeps
        // concurrent put/get/delete on the same session_id serialized.
        state
            .session_checkpoints
            .insert(session_id.to_string(), SessionEntry { body, meta });
        // Session checkpoints carry no `(object_id, snap_seq)` identity —
        // surface zeros and use `adapter_key` as the opaque carrier. The
        // SR-1.3 conformance harness (case #2) verifies `blob.handle ==
        // put_handle`, satisfied as long as `get_session_checkpoint` mints
        // the same shape.
        Ok(CheckpointHandle::base(
            0,
            0,
            format!("session:{session_id}"),
        ))
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.maybe_inject("get_session_checkpoint")?;
        let state = self.state.lock().await;
        let Some(entry) = state.session_checkpoints.get(session_id) else {
            // Absent session is `Ok(None)`, never `NotFound` (trait
            // contract, `store.rs:270-273`).
            return Ok(None);
        };
        Ok(Some(CheckpointBlob {
            handle: CheckpointHandle::base(0, 0, format!("session:{session_id}")),
            bytes: entry.body.clone(),
            meta: entry.meta.clone(),
        }))
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_session_checkpoint")?;
        let mut state = self.state.lock().await;
        state.session_checkpoints.remove(session_id);
        // Idempotent on missing entries per trait contract — already-absent
        // sessions fall through to `Ok(())`, mirroring the layer-checkpoint
        // delete path above.
        Ok(())
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.maybe_inject("list_session_checkpoint_ids")?;
        let state = self.state.lock().await;
        // The session map holds nothing but stored session backups — no temps,
        // no layer artifacts — so its keys are exactly the contract's set.
        Ok(state.session_checkpoints.keys().cloned().collect())
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.maybe_inject("put_ext_blob")?;
        let mut state = self.state.lock().await;
        // Stores its own copy, never the caller's `Bytes`: the ext-mirror's
        // byte charge lives as long as the last handle on the bytes it sends
        // (`CheckpointStore::put_ext_blob`), so keeping that handle would keep
        // the charge for the store's lifetime. `or_insert_with` is the
        // idempotent put — re-putting the same hash leaves the stored copy
        // untouched, a true no-op the unit test checks via inner-map identity.
        state
            .ext_blobs
            .entry(hash_hex.to_string())
            .or_insert_with(|| Bytes::copy_from_slice(&bytes));
        Ok(())
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.maybe_inject("get_ext_blob")?;
        let state = self.state.lock().await;
        Ok(state.ext_blobs.get(hash_hex).cloned())
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.maybe_inject("ext_blob_exists")?;
        let state = self.state.lock().await;
        Ok(state.ext_blobs.contains_key(hash_hex))
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.maybe_inject("delete_ext_blob")?;
        let mut state = self.state.lock().await;
        state.ext_blobs.remove(hash_hex);
        // Idempotent on missing hash per trait contract.
        Ok(())
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.maybe_inject("put_counters_sidecar")?;
        let mut state = self.state.lock().await;
        // Last-writer-wins per trait contract.
        state.counters_sidecar = Some(body);
        Ok(())
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.maybe_inject("get_counters_sidecar")?;
        let state = self.state.lock().await;
        Ok(state.counters_sidecar.clone())
    }

    /// Every `object_id` with at least one stored generation, **of either
    /// kind**.
    ///
    /// # Kind-agnostic by construction
    ///
    /// This is the one layer path that needed no kind branch when tails landed,
    /// and the reason is the state shape rather than luck: bases and tails share
    /// `layer_checkpoints`, so `keys()` enumerates a tail-only object for free.
    /// A separate `tail_checkpoints` map would have compiled, passed every
    /// existing test, and silently hidden tail-only objects from the recovery
    /// hook's working set — the shared map is what forecloses that.
    ///
    /// This is the InMemory analogue of the LocalDisk directory walk and the
    /// StorageApi address enumeration, which have no such structural guarantee:
    /// each scans a flat namespace and must recognize the `.tail` stem
    /// explicitly. Correctness here is therefore *not* evidence that the other
    /// two are correct.
    ///
    /// `delete_layer_checkpoint` prunes an emptied per-object map so a
    /// fully-deleted object stops surfacing;
    /// `delete_layer_checkpoint_prunes_empty_per_object_map` in this file's
    /// tests module pins that (the analogue of `local_disk.rs`'s
    /// `list_all_object_ids_returns_dedup_set_across_generations` and
    /// `storage_api.rs`'s `list_all_object_ids_stays_ompf_driven`).
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.maybe_inject("list_all_object_ids")?;
        let state = self.state.lock().await;
        Ok(state.layer_checkpoints.keys().copied().collect())
    }

    /// Union of every stored **base** sidecar's hashes — the live set the
    /// reconciliation collect deletes tombstoned blobs against.
    ///
    /// Walking `slot.base` and skipping `slot.tail` entirely is load-bearing,
    /// not incidental (SR-2.4, b.s5y's sidecar-driven-enumeration MUST): a tail
    /// carries no ext-refs sidecar by design, `validate_tail_put` rejects a tail
    /// put that supplies one, and `put_layer_checkpoint` stores `sidecar: None`
    /// on every tail regardless. Iterating tail slots here would therefore be
    /// dead code today — and dead code on a live-set walk is exactly the kind a
    /// later edit turns into live-set pollution, which degrades the
    /// reconciliation collect toward deleting nothing. Skipping them says the invariant out
    /// loud instead.
    ///
    /// The walk itself lives in [`Self::list_live_ext_set`] (b.oya); this
    /// method is its projection onto the hash set.
    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        Ok(self.list_live_ext_set().await?.hashes)
    }

    /// The live-set walk [`Self::list_referenced_ext_hashes`] documents, plus
    /// `listed_layer_artifacts`: under the **same** lock, the count
    /// `list_layer_artifacts` would report from that state (b.oya; see the
    /// trait doc). Injects under the `list_referenced_ext_hashes` label, the
    /// walk's label.
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.maybe_inject("list_referenced_ext_hashes")?;
        let state = self.state.lock().await;
        let listed_layer_artifacts = layer_artifacts_of(&state.layer_checkpoints).len();
        let mut hashes: HashSet<String> = HashSet::new();
        for per_object in state.layer_checkpoints.values() {
            for slot in per_object.values() {
                let Some(sidecar) = slot.base.as_ref().and_then(|e| e.sidecar.as_ref()) else {
                    continue;
                };
                // Per SR-1.5 we treat the sidecar bytes as opaque at put
                // time; parsing happens here on demand. A malformed sidecar
                // surfaces as `Permanent` (it cannot be retried into
                // success), matching the LocalDisk adapter's behavior so the
                // conformance harness sees consistent classification.
                let parsed: ExtRefsSidecar =
                    serde_json::from_slice(sidecar).map_err(|err| CheckpointError::Permanent {
                        source: Box::new(err),
                    })?;
                hashes.extend(parsed.ext_hashes);
            }
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
        // Tasks 3/4 must not "complete" this arm. It is short-circuited before
        // the lookup because `(object_id, snap_seq)` addresses the base — a tail
        // handle falling through would report the BASE's hashes as the tail's,
        // inflating the retention cascade's per-handle pruned set.
        if handle.kind == CheckpointKind::Tail {
            return Ok(HashSet::new());
        }
        let state = self.state.lock().await;
        // Address by `(object_id, snap_seq)` — `adapter_key` is opaque
        // carrier state on the trait surface and we ignore it on the read
        // path so the retention-prune cascade can synthesize handles from
        // `list_layer_checkpoints` without round-tripping the adapter_key.
        let Some(per_object) = state.layer_checkpoints.get(&handle.object_id) else {
            // Absent generation entirely — `Ok(empty)` per the trait
            // contract. Matches LocalDisk's tolerance for legacy /
            // already-pruned handles and keeps the cascade from
            // over-classifying absence as failure.
            return Ok(HashSet::new());
        };
        let Some(entry) = per_object
            .get(&handle.snap_seq)
            .and_then(|slot| slot.base.as_ref())
        else {
            return Ok(HashSet::new());
        };
        let Some(sidecar) = entry.sidecar.as_ref() else {
            // OMPF present but no inline sidecar — same absent-sidecar
            // tolerance as `list_referenced_ext_hashes`.
            return Ok(HashSet::new());
        };
        // Malformed sidecar JSON classifies as `Permanent` per the trait
        // contract — re-reading the same bytes will not produce a
        // different parse result, so surface to the caller instead of
        // silently dropping references for that handle.
        let parsed: ExtRefsSidecar =
            serde_json::from_slice(sidecar).map_err(|err| CheckpointError::Permanent {
                source: Box::new(err),
            })?;
        Ok(parsed.ext_hashes.into_iter().collect())
    }

    /// Enumerate the in-memory map's generations as artifacts, both families
    /// (b.9qd).
    ///
    /// **This adapter structurally cannot orphan.** A `LayerEntry` is inserted
    /// under one mutex with its payload and (when the caller paired one) its
    /// sidecar already in hand, and it is removed as a unit — there is no
    /// intermediate state in which a sidecar or meta exists without the payload
    /// it companions. Every base entry therefore yields an `Ompf` and every tail
    /// entry a `TailEnvelope`, so the reconciliation pass finds no stem missing
    /// its generation artifact and no candidate to defer. The override exists so
    /// the SR-1.3 conformance harness sees the same surface on all three
    /// adapters, and so a test can drive the pass against InMemory and observe a
    /// truthful "nothing to do".
    ///
    /// No `Temp` kind is ever emitted: there is no temp-then-rename step here.
    ///
    /// A tail never yields an `ExtRefs`: [`validate_tail_put`] rejects a
    /// sidecar-carrying tail put (SR-2.4), so `entry.sidecar` is structurally
    /// `None` on a `slot.tail` entry and the tail block has no branch for it.
    ///
    /// Emission order is deterministic per slot — the base triple, then the tail
    /// pair — which is what lets the in-file unit test compare a `Vec`
    /// positionally within an object. Order *across* objects follows the
    /// `HashMap`'s iteration and is not specified.
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.maybe_inject("list_layer_artifacts")?;
        let state = self.state.lock().await;
        Ok(layer_artifacts_of(&state.layer_checkpoints))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        let json = serde_json::json!({
            "ext_hashes": hashes,
        });
        Bytes::from(serde_json::to_vec(&json).expect("sidecar JSON encodes"))
    }

    #[tokio::test]
    async fn new_returns_empty_adapter_without_shared_state() {
        let a = InMemoryCheckpointStore::new();
        let b = InMemoryCheckpointStore::new();

        // Both adapters start empty — nothing surfaces from any read path.
        assert!(a.list_all_object_ids().await.unwrap().is_empty());
        assert!(b.list_all_object_ids().await.unwrap().is_empty());
        assert!(a.get_counters_sidecar().await.unwrap().is_none());

        // Mutating `a` MUST NOT bleed into `b` — confirms per-instance maps,
        // not shared static state.
        a.put_layer_checkpoint(7, 1, Bytes::from_static(b"ompf"), test_meta())
            .await
            .unwrap();
        a.put_ext_blob("hash-a", Bytes::from_static(b"v"))
            .await
            .unwrap();
        a.put_counters_sidecar(Bytes::from_static(b"counters"))
            .await
            .unwrap();

        assert_eq!(a.list_all_object_ids().await.unwrap(), vec![7]);
        assert!(
            b.list_all_object_ids().await.unwrap().is_empty(),
            "second adapter must not observe writes against the first"
        );
        assert!(b.get_ext_blob("hash-a").await.unwrap().is_none());
        assert!(b.get_counters_sidecar().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn put_layer_checkpoint_same_key_twice_retains_one_generation() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 42;
        let seq: u64 = 9;

        store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"first"), test_meta())
            .await
            .unwrap();
        store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"second"), test_meta())
            .await
            .unwrap();

        // Introspect the inner map directly: same module so private fields
        // are reachable. Exactly one generation must remain at (oid, seq).
        let state = store.state.lock().await;
        let per_object = state
            .layer_checkpoints
            .get(&oid)
            .expect("object_id entry present after put");
        assert_eq!(
            per_object.len(),
            1,
            "duplicate (object_id, snap_seq) puts must collapse to one generation"
        );
        // Last-writer-wins on bytes, per the SR-1.3 #9 contract. Both puts were
        // `Base`, so the base slot holds the second write and the tail slot was
        // never populated.
        let slot = per_object.get(&seq).expect("snap_seq slot present");
        let entry = slot.base.as_ref().expect("base slot populated");
        assert_eq!(entry.ompf, Bytes::from_static(b"second"));
        assert!(
            slot.tail.is_none(),
            "a base put must not populate the tail slot"
        );
    }

    #[tokio::test]
    async fn put_ext_blob_same_hash_twice_does_not_replace_stored_bytes() {
        let store = InMemoryCheckpointStore::new();
        let hash = "deadbeef";

        // Pointer identity (not byte equality) is what we are checking. The
        // store keeps its own copy of the first put's bytes; a true no-op
        // leaves that stored copy — and its data pointer — in the map.
        store
            .put_ext_blob(hash, Bytes::from(b"payload".to_vec()))
            .await
            .unwrap();
        let first_ptr = store
            .state
            .lock()
            .await
            .ext_blobs
            .get(hash)
            .expect("hash present after the first put")
            .as_ptr();

        store
            .put_ext_blob(hash, Bytes::from(b"payload".to_vec()))
            .await
            .unwrap();

        // Introspect the inner map: the stored Bytes must still point at the
        // first stored copy — a true HashMap-no-op, not an observationally
        // equivalent overwrite.
        let state = store.state.lock().await;
        let stored = state.ext_blobs.get(hash).expect("hash present after put");
        assert_eq!(stored.as_ptr(), first_ptr, "second put must be a no-op");
        assert_eq!(state.ext_blobs.len(), 1);
    }

    /// T14 (b.mhq): the store keeps a copy of the bytes it is handed, never
    /// the caller's `Bytes`. A value the ext-mirror sends is charged to its
    /// byte budget until the last handle on its memory drops, so a store
    /// that kept the caller's handle would hold the charge for its own
    /// lifetime: here, once the mirror's item is dropped after the put, the
    /// whole budget is back, and the stored bytes are equal to the value but
    /// live in another allocation.
    #[tokio::test]
    async fn put_ext_blob_stores_its_own_copy_so_a_mirrored_values_charge_returns_after_the_put() {
        use std::num::NonZeroU64;

        use crate::checkpoint::ext_mirror::ext_mirror_channel;

        let store = InMemoryCheckpointStore::new();
        let metrics = std::sync::Arc::new(crate::observability::metrics::metrics_scrape_only(
            "in-memory-ext-copy",
        ));
        let (sender, mut rx) =
            ext_mirror_channel(4, NonZeroU64::new(1 << 20).expect("non-zero"), metrics);
        let value = b"mirrored external value".as_slice();
        let hash = "ab".repeat(32);
        sender.send(hash.clone(), sender.copy_value(value), "0".to_owned());
        let item = rx.try_recv().expect("the value is queued");
        let sent_ptr = item.bytes().as_ptr();
        assert!(
            sender.available_budget_bytes() < sender.budget_bytes(),
            "fixture self-check: the queued value is charged"
        );

        store
            .put_ext_blob(item.hash_hex(), item.bytes().clone())
            .await
            .expect("put_ext_blob succeeds");
        drop(item);

        assert_eq!(
            sender.available_budget_bytes(),
            sender.budget_bytes(),
            "the store holds no handle on the mirror's copy, so its charge is back"
        );
        let state = store.state.lock().await;
        let stored = state.ext_blobs.get(&hash).expect("the value is stored");
        assert_eq!(stored.as_ref(), value, "the stored copy equals the value");
        assert_ne!(stored.as_ptr(), sent_ptr, "in its own allocation");
    }

    #[tokio::test]
    async fn list_layer_checkpoints_returns_generations_newest_first() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 100;

        // Insert in a deliberately scrambled order to prove ordering does
        // not depend on insertion sequence.
        for seq in [3_u64, 1, 5, 2, 4] {
            store
                .put_layer_checkpoint(oid, seq, Bytes::from_static(b"x"), test_meta())
                .await
                .unwrap();
        }

        let handles = store.list_layer_checkpoints(oid).await.unwrap();
        let seqs: Vec<u64> = handles.iter().map(|h| h.snap_seq).collect();
        assert_eq!(
            seqs,
            vec![5, 4, 3, 2, 1],
            "list_layer_checkpoints must return generations newest-first by snap_seq"
        );

        // Companion: get_latest_layer_checkpoint MUST agree with the head of
        // the list (trait-level contract — kept here as a guard that the
        // local ordering is internally consistent).
        let latest = store
            .get_latest_layer_checkpoint(oid)
            .await
            .unwrap()
            .expect("at least one checkpoint exists");
        assert_eq!(latest.handle.snap_seq, 5);
    }

    #[tokio::test]
    async fn list_referenced_ext_hashes_unions_across_objects_without_duplicates() {
        let store = InMemoryCheckpointStore::new();

        // Object 1 references {"aa", "bb"}.
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o1"), test_meta())
            .await
            .unwrap();
        store
            .set_layer_sidecar(1, 1, sidecar_for(&["aa", "bb"]))
            .await;

        // Object 2 references {"bb", "cc"} — overlap on "bb".
        store
            .put_layer_checkpoint(2, 1, Bytes::from_static(b"o2"), test_meta())
            .await
            .unwrap();
        store
            .set_layer_sidecar(2, 1, sidecar_for(&["bb", "cc"]))
            .await;

        let refs = store.list_referenced_ext_hashes().await.unwrap();
        let expected: HashSet<String> = ["aa", "bb", "cc"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "list_referenced_ext_hashes must equal the union of all per-object sidecar sets with no duplicates"
        );
    }

    #[tokio::test]
    async fn in_memory_put_then_get_session_checkpoint_round_trips_bytes() {
        let store = InMemoryCheckpointStore::new();
        let sid = "sess-round-trip";
        let body = Bytes::from_static(b"opaque-session-body");

        let put_handle = store
            .put_session_checkpoint(sid, body.clone(), test_meta())
            .await
            .expect("put_session_checkpoint must succeed");

        let blob = store
            .get_session_checkpoint(sid)
            .await
            .expect("get_session_checkpoint must succeed")
            .expect("get must return Some after a put");

        assert_eq!(
            blob.bytes, body,
            "session-checkpoint bytes must round-trip verbatim per SR-1.5"
        );
        assert_eq!(
            blob.handle, put_handle,
            "get_session_checkpoint must surface the same handle put returned (conformance case #2)"
        );
        assert_eq!(
            blob.handle.adapter_key, "session:sess-round-trip",
            "adapter_key must encode the session_id since (object_id, snap_seq) are zero",
        );
    }

    #[tokio::test]
    async fn in_memory_get_session_checkpoint_unknown_session_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        let result = store
            .get_session_checkpoint("never-put")
            .await
            .expect("get on an absent session must be Ok, not Err — trait contract");
        assert!(
            result.is_none(),
            "absent session must surface Ok(None), not Ok(Some(_))"
        );
    }

    #[tokio::test]
    async fn in_memory_put_then_delete_then_get_session_checkpoint_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        let sid = "sess-put-delete-get";

        store
            .put_session_checkpoint(sid, Bytes::from_static(b"body"), test_meta())
            .await
            .expect("put must succeed");
        store
            .delete_session_checkpoint(sid)
            .await
            .expect("delete must succeed");

        let result = store
            .get_session_checkpoint(sid)
            .await
            .expect("get after delete must be Ok, not Err");
        assert!(
            result.is_none(),
            "post-delete read MUST be Ok(None), not Ok(Some(_)) or NotFound"
        );
    }

    #[tokio::test]
    async fn in_memory_delete_session_checkpoint_on_absent_returns_ok() {
        let store = InMemoryCheckpointStore::new();
        // Never put — delete must still be Ok(()) per the trait's
        // idempotent-on-missing contract.
        store
            .delete_session_checkpoint("absent-session")
            .await
            .expect("delete on a missing session must be Ok, not NotFound");
    }

    #[tokio::test]
    async fn in_memory_concurrent_put_get_unrelated_sessions_do_not_interfere() {
        // Mirrors SR-1.3 case #17 (cross-task interleave). Two unrelated
        // session_ids must not collide under the shared mutex — `get` on a
        // never-put sid must report Ok(None) even while another task is
        // concurrently putting a different sid.
        let store = Arc::new(InMemoryCheckpointStore::new());
        let body_a = Bytes::from_static(b"sid-a-body");

        let put_store = Arc::clone(&store);
        let put_body = body_a.clone();
        let put_task = tokio::spawn(async move {
            put_store
                .put_session_checkpoint("sid-a", put_body, test_meta())
                .await
        });

        let get_store = Arc::clone(&store);
        let get_task = tokio::spawn(async move { get_store.get_session_checkpoint("sid-b").await });

        let put_handle = put_task
            .await
            .expect("put task join")
            .expect("put_session_checkpoint must succeed");
        let get_result = get_task
            .await
            .expect("get task join")
            .expect("get_session_checkpoint must be Ok");

        assert_eq!(put_handle.adapter_key, "session:sid-a");
        assert!(
            get_result.is_none(),
            "concurrent get on an unrelated session_id must remain Ok(None)"
        );

        // And the put landed cleanly: a follow-up get returns the body
        // verbatim. Confirms the concurrent get did not corrupt the put.
        let blob = store
            .get_session_checkpoint("sid-a")
            .await
            .expect("get must succeed")
            .expect("the concurrent put landed; get must surface Some");
        assert_eq!(blob.bytes, body_a);
    }

    #[tokio::test]
    async fn in_memory_put_session_checkpoint_last_writer_wins() {
        let store = InMemoryCheckpointStore::new();
        let sid = "sess-lww";
        let b1 = Bytes::from_static(b"version-1");
        let b2 = Bytes::from_static(b"version-2");

        store
            .put_session_checkpoint(sid, b1, test_meta())
            .await
            .expect("first put");
        store
            .put_session_checkpoint(sid, b2.clone(), test_meta())
            .await
            .expect("second put");

        let blob = store
            .get_session_checkpoint(sid)
            .await
            .expect("get must succeed")
            .expect("after two puts the entry must be present");
        assert_eq!(
            blob.bytes, b2,
            "second put must replace the first (last-writer-wins per trait contract)"
        );
    }

    #[tokio::test]
    async fn arc_dyn_checkpoint_store_upcast_compiles_and_dispatches() {
        // Compile-time: `Arc<InMemoryCheckpointStore>` must upcast cleanly to
        // `Arc<dyn CheckpointStore>` — this is the form `main.rs` and the
        // periodic checkpoint loop plumb through the system.
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

        // And the upcast value must dispatch through the trait surface at
        // runtime — guards against accidental object-unsafety regressions.
        let ids = store.list_all_object_ids().await.unwrap();
        assert!(ids.is_empty());
    }

    // ----- t3.fgj.ya.r7.3a: gap-coverage tests for engineer-added paths -----
    //
    // The cross-adapter SR-1.3 behavioral contracts live in the conformance
    // harness Task `t2.fgj.ya.74`. These tests target adapter-internal
    // branches in this adapter's file specifically — empty-map pruning on
    // delete, malformed-sidecar classification, and the absent-key
    // get-paths the trait documents as `Ok(None)`.

    /// `delete_layer_checkpoint` removes the addressed generation AND prunes
    /// the per-object map when its last generation is dropped. The prune step
    /// keeps `list_all_object_ids` honest — without it, a fully-deleted
    /// object_id would still surface from the keys set and falsely fan-out the
    /// recovery hook.
    ///
    /// The prune is two-level now that a sequence holds a kind-slotted pair:
    /// the sequence key survives while the *other* kind is still present, and
    /// only a fully-vacated sequence releases it. `..._two_level_prune` below
    /// pins that half.
    #[tokio::test]
    async fn delete_layer_checkpoint_prunes_empty_per_object_map() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 5;

        // Two generations on oid 5; one on oid 6 (untouched).
        let h1 = store
            .put_layer_checkpoint(oid, 1, Bytes::from_static(b"v1"), test_meta())
            .await
            .unwrap();
        let h2 = store
            .put_layer_checkpoint(oid, 2, Bytes::from_static(b"v2"), test_meta())
            .await
            .unwrap();
        store
            .put_layer_checkpoint(6, 1, Bytes::from_static(b"o6"), test_meta())
            .await
            .unwrap();

        // Delete the first generation — per-object map still has snap 2.
        store.delete_layer_checkpoint(h1).await.unwrap();
        {
            let state = store.state.lock().await;
            assert!(
                state.layer_checkpoints.contains_key(&oid),
                "object_id entry must remain while another snap_seq is present"
            );
            assert_eq!(state.layer_checkpoints[&oid].len(), 1);
        }

        // Delete the second — per-object map must be pruned so the keys
        // set drops oid entirely.
        store.delete_layer_checkpoint(h2).await.unwrap();
        let state = store.state.lock().await;
        assert!(
            !state.layer_checkpoints.contains_key(&oid),
            "per-object map must be pruned when its last snap is dropped"
        );
        assert!(
            state.layer_checkpoints.contains_key(&6),
            "unrelated object_id must remain untouched"
        );
    }

    /// `delete_layer_checkpoint` on a missing handle must be `Ok(())` — the
    /// retention pruner re-issues deletes without first checking existence.
    /// A spurious `NotFound` here would break the pruner.
    #[tokio::test]
    async fn delete_layer_checkpoint_missing_handle_idempotent() {
        let store = InMemoryCheckpointStore::new();
        // No put ever happened — the handle is purely synthetic.
        store
            .delete_layer_checkpoint(CheckpointHandle::base(999, 1, "phantom".into()))
            .await
            .expect("delete on a missing handle must be Ok, not NotFound");

        // Even an object_id that exists but a snap_seq that doesn't must
        // also be a no-op (without dropping the existing object_id).
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .unwrap();
        store
            .delete_layer_checkpoint(CheckpointHandle::base(1, 99, "phantom".into()))
            .await
            .expect("delete on a missing snap_seq must be Ok");
        // The existing (1, 1) entry must NOT have been disturbed.
        let state = store.state.lock().await;
        assert!(state.layer_checkpoints[&1].contains_key(&1));
    }

    /// `delete_ext_blob` is idempotent on missing — the orphan sweep
    /// re-deletes hashes it already pruned in earlier ticks. A spurious
    /// `NotFound` would surface as a sweep failure metric.
    #[tokio::test]
    async fn delete_ext_blob_missing_hash_idempotent() {
        let store = InMemoryCheckpointStore::new();
        store
            .delete_ext_blob("never-stored")
            .await
            .expect("delete on a missing hash must be Ok, not NotFound");
        // Deleting after a successful put then a second delete must also
        // be Ok — the second delete is the idempotent re-call.
        store
            .put_ext_blob("h", Bytes::from_static(b"x"))
            .await
            .unwrap();
        store.delete_ext_blob("h").await.expect("first delete");
        store
            .delete_ext_blob("h")
            .await
            .expect("second delete must be Ok, not NotFound");
    }

    /// `get_latest_layer_checkpoint` on an unknown object_id must report
    /// `Ok(None)`, never `NotFound`. The recovery hook treats `None` as the
    /// "no checkpoint yet" path; a synthetic `NotFound` would falsely
    /// surface as an error to the operator.
    #[tokio::test]
    async fn get_latest_layer_checkpoint_unknown_object_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        let result = store
            .get_latest_layer_checkpoint(404)
            .await
            .expect("get on unknown object_id must be Ok, not Err");
        assert!(result.is_none());
    }

    /// `get_ext_blob` on an unknown hash must report `Ok(None)` — the
    /// idempotent-put short-circuit on LocalDisk relies on this so a
    /// reference InMemory adapter's behaviour MUST match.
    #[tokio::test]
    async fn get_ext_blob_unknown_hash_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        let result = store
            .get_ext_blob("never-stored")
            .await
            .expect("get on unknown hash must be Ok, not Err");
        assert!(result.is_none());
    }

    /// `put_counters_sidecar` then `get_counters_sidecar` round-trips bytes,
    /// and a second put replaces the first per the last-writer-wins contract.
    #[tokio::test]
    async fn counters_sidecar_round_trips_last_write() {
        let store = InMemoryCheckpointStore::new();
        // Fresh adapter — get must report None.
        assert!(store.get_counters_sidecar().await.unwrap().is_none());

        store
            .put_counters_sidecar(Bytes::from_static(b"v1"))
            .await
            .unwrap();
        assert_eq!(
            store.get_counters_sidecar().await.unwrap().unwrap(),
            Bytes::from_static(b"v1")
        );

        store
            .put_counters_sidecar(Bytes::from_static(b"v2"))
            .await
            .unwrap();
        assert_eq!(
            store.get_counters_sidecar().await.unwrap().unwrap(),
            Bytes::from_static(b"v2"),
            "second put must replace the first (last-writer-wins)"
        );
    }

    /// A sidecar with non-JSON bytes attached to a layer entry must classify
    /// as `Permanent` from `list_referenced_ext_hashes` — matching the
    /// LocalDisk adapter so the conformance harness sees consistent error
    /// classification across both reference adapters.
    #[tokio::test]
    async fn list_referenced_ext_hashes_malformed_sidecar_returns_permanent() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"o"), test_meta())
            .await
            .unwrap();
        // Attach a clearly-malformed (non-JSON) sidecar — bytes are opaque at
        // put time per SR-1.5; the parse failure surfaces on the next call to
        // list_referenced_ext_hashes.
        store
            .set_layer_sidecar(1, 1, Bytes::from_static(b"not-json-at-all{{{"))
            .await;

        let err = store
            .list_referenced_ext_hashes()
            .await
            .expect_err("malformed sidecar must Err, not panic or silent skip");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "malformed sidecar must classify as Permanent (cannot retry into success); got {:?}",
            err
        );
    }

    /// `set_layer_sidecar` on a `(object_id, snap_seq)` pair that has NOT
    /// been `put_layer_checkpoint`'d is a documented no-op — the assembler
    /// is expected to put-then-set, and a misordered call must not panic or
    /// corrupt other entries. After the misordered call,
    /// `list_referenced_ext_hashes` must still report an empty set.
    #[tokio::test]
    async fn set_layer_sidecar_before_put_is_noop() {
        let store = InMemoryCheckpointStore::new();

        // No put has been done — set must NOT create a phantom entry.
        store.set_layer_sidecar(7, 1, sidecar_for(&["aa"])).await;

        let state = store.state.lock().await;
        assert!(
            !state.layer_checkpoints.contains_key(&7),
            "set_layer_sidecar must not create a layer entry by itself"
        );
        drop(state);

        // And the referenced-hashes view must be empty (no sidecar was
        // attached anywhere).
        let refs = store.list_referenced_ext_hashes().await.unwrap();
        assert!(
            refs.is_empty(),
            "no sidecar was actually attached (put never happened), got {refs:?}"
        );
    }

    // ----- t3.fgj.zc.56.uq: per-handle `read_ext_refs_sidecar` tests --------
    //
    // The retention-prune cascade (parent Task t2.fgj.zc.56) needs the
    // per-generation referenced-hash set. These four tests cover the
    // trait-contract branches end-to-end through this adapter's in-memory
    // `LayerEntry.sidecar` field (happy / absent-inline-sidecar / malformed /
    // absent-generation).

    /// Happy path: a layer checkpoint put with `meta.sidecar_bytes = Some(_)`
    /// attaches the inline sidecar to `LayerEntry`; `read_ext_refs_sidecar`
    /// for the corresponding handle returns the exact parsed `HashSet`.
    #[tokio::test]
    async fn in_memory_read_ext_refs_sidecar_returns_parsed_hashes_for_happy_path_handle() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 50;
        let seq: u64 = 7;

        let mut meta = test_meta();
        meta.snap_seq = Some(seq);
        meta.sidecar_bytes = Some(sidecar_for(&["aa", "bb", "cc"]));
        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"o"), meta)
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
    /// None` leaves `LayerEntry.sidecar` as `None`. `read_ext_refs_sidecar`
    /// MUST return `Ok(HashSet::new())` — matches the LocalDisk adapter's
    /// tolerance for OMPF-only generations.
    #[tokio::test]
    async fn in_memory_read_ext_refs_sidecar_returns_empty_when_inline_sidecar_absent() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 51;
        let seq: u64 = 8;

        // `sidecar_bytes` defaults to `None` — explicitly construct that meta.
        // Identity is still populated: `validate_generation_identity` refuses a
        // put without it, and this case is about the sidecar, not identity.
        let handle = store
            .put_layer_checkpoint(
                oid,
                seq,
                Bytes::from_static(b"o"),
                CheckpointMeta {
                    snap_seq: Some(seq),
                    ..crate::checkpoint::adapters::identity_default()
                },
            )
            .await
            .expect("put");
        // Sanity: the inline sidecar truly is `None`.
        {
            let state = store.state.lock().await;
            assert!(
                state.layer_checkpoints[&oid][&seq]
                    .base
                    .as_ref()
                    .expect("base slot populated")
                    .sidecar
                    .is_none(),
                "precondition: LayerEntry.sidecar should be None when meta.sidecar_bytes is None"
            );
        }

        let refs = store
            .read_ext_refs_sidecar(&handle)
            .await
            .expect("absent inline sidecar must be Ok(empty), not Err");
        assert!(
            refs.is_empty(),
            "missing inline sidecar must yield Ok(empty) per trait contract; got {refs:?}"
        );
    }

    /// Malformed sidecar: attach a non-JSON byte sequence via
    /// `set_layer_sidecar` (bypassing the trait surface so we control the
    /// bytes). The per-handle reader MUST classify the parse failure as
    /// `Permanent` — matches the LocalDisk adapter so the cascade sees
    /// consistent classification across both reference adapters.
    #[tokio::test]
    async fn in_memory_read_ext_refs_sidecar_malformed_bytes_classify_as_permanent() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 52;
        let seq: u64 = 9;

        let handle = store
            .put_layer_checkpoint(oid, seq, Bytes::from_static(b"o"), test_meta())
            .await
            .expect("put");
        // Attach a deliberately-malformed (non-JSON) sidecar via the
        // adapter-internal setter.
        store
            .set_layer_sidecar(oid, seq, Bytes::from_static(b"not-json-at-all{{{"))
            .await;

        let err = store
            .read_ext_refs_sidecar(&handle)
            .await
            .expect_err("malformed sidecar must Err, not silently downgrade to empty");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "malformed sidecar must classify as Permanent; got {:?}",
            err
        );
    }

    /// Handle-for-absent-generation: synthesize a handle for an `(object_id,
    /// snap_seq)` tuple that was never `put_layer_checkpoint`'d.
    /// `read_ext_refs_sidecar` MUST return `Ok(HashSet::new())` — the
    /// cascade may pass already-pruned handles through this method and a
    /// synthetic `Permanent` would break it. Verified for both the
    /// completely-absent-object_id branch and the present-object-but-absent-
    /// snap_seq branch.
    #[tokio::test]
    async fn in_memory_read_ext_refs_sidecar_for_absent_generation_returns_empty() {
        let store = InMemoryCheckpointStore::new();

        // (a) Completely-synthetic handle — nothing has ever been put.
        let phantom_oid = CheckpointHandle::base(999, 1, "phantom".into());
        let refs = store
            .read_ext_refs_sidecar(&phantom_oid)
            .await
            .expect("absent object_id must be Ok(empty), not Err");
        assert!(
            refs.is_empty(),
            "handle for never-put object_id must yield Ok(empty); got {refs:?}"
        );

        // (b) Object_id exists, but the snap_seq doesn't — also Ok(empty).
        // Without this branch the cascade would fail-fast on any handle
        // whose snap_seq was pruned between list and read.
        store
            .put_layer_checkpoint(7, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .unwrap();
        let phantom_seq = CheckpointHandle::base(7, 99, "phantom".into());
        let refs = store
            .read_ext_refs_sidecar(&phantom_seq)
            .await
            .expect("absent snap_seq must be Ok(empty), not Err");
        assert!(
            refs.is_empty(),
            "handle for present-object/absent-snap_seq must yield Ok(empty); got {refs:?}"
        );
    }

    /// `put_layer_checkpoint` on an existing `(object_id, snap_seq)` slot
    /// resets the sidecar to `None` — the prior sidecar's bytes are dropped
    /// along with the prior `LayerEntry` per the last-writer-wins contract.
    /// Without this, a re-put would silently reuse the stale sidecar and
    /// the orphan sweep would over-count referenced blobs.
    #[tokio::test]
    async fn re_put_layer_checkpoint_clears_prior_sidecar() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"v1"), test_meta())
            .await
            .unwrap();
        store
            .set_layer_sidecar(1, 1, sidecar_for(&["aa", "bb"]))
            .await;
        // Sanity: the sidecar's hashes appear in the referenced set.
        assert!(!store.list_referenced_ext_hashes().await.unwrap().is_empty());

        // Re-put the same slot — the new entry has `sidecar: None`.
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"v2"), test_meta())
            .await
            .unwrap();

        let refs = store.list_referenced_ext_hashes().await.unwrap();
        assert!(
            refs.is_empty(),
            "re-put must drop the prior sidecar so its hashes stop pinning blobs; got {refs:?}"
        );
    }

    // ----- t3.dfh.gy.bd.11: `get_layer_checkpoint` (SR-2.3) ----------------
    //
    // Adapter-local mechanics only. The cross-adapter conformance cases for
    // this method belong to Task `t2.dfh.gy.qb`; these pin what is specific to
    // the in-memory `BTreeMap` addressing, plus the four interim/final tail
    // arms this adapter grew in `t3.dfh.gy.bd.1r`.

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
    /// generations are put and the *middle* handle is fetched, so an
    /// implementation that lazily delegated to `get_latest` would return the
    /// wrong bytes and fail here.
    #[tokio::test]
    async fn get_layer_checkpoint_returns_the_addressed_middle_generation_not_the_newest() {
        let store = InMemoryCheckpointStore::new();
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
        assert_eq!(handles.len(), 3, "three generations were put");
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
    /// returned handle MUST carry the adapter's own canonical key rather than
    /// echoing the caller's.
    #[tokio::test]
    async fn get_layer_checkpoint_ignores_adapter_key_and_returns_the_canonical_one() {
        let store = InMemoryCheckpointStore::new();
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
        assert_eq!(
            blob.handle.adapter_key, "7:2",
            "returned handle must carry this adapter's canonical key, not the caller's"
        );
    }

    /// Absence flavour 1 — an `object_id` that was never written.
    #[tokio::test]
    async fn get_layer_checkpoint_unknown_object_id_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .expect("put");

        let handle = CheckpointHandle::base(999, 1, "999:1".to_string());
        match store.get_layer_checkpoint(&handle).await {
            Ok(None) => {}
            other => panic!("absent object_id must be Ok(None), never NotFound; got {other:?}"),
        }
    }

    /// Absence flavour 2 — a real `object_id` at a `snap_seq` that does not
    /// exist. This is also the concurrent-prune race shape: a listing named
    /// the handle and retention removed it before the get landed.
    #[tokio::test]
    async fn get_layer_checkpoint_unknown_snap_seq_on_known_object_returns_ok_none() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(1, 1, Bytes::from_static(b"x"), test_meta())
            .await
            .expect("put");

        let handle = CheckpointHandle::base(1, 4_242, "1:4242".to_string());
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
    /// a real call site rather than only by the `_assert_checkpoint_store_object_safe`
    /// compile guard.
    #[tokio::test]
    async fn get_layer_checkpoint_for_the_newest_handle_matches_get_latest_layer_checkpoint() {
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
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

    // ----- Tail storage (t3.dfh.gy.bd.1r arms, completed in t3.dfh.gy.ib.6x) -
    //
    // The three arms below were INTERIM placeholders in Task 2 (tail put →
    // `NotImplemented`, tail get → `Ok(None)`, tail delete → no-op). Task 3
    // replaced them with real kind-slotted storage, so each now pins the real
    // behaviour. Adapter-local mechanics only — the full mixed-kind matrix is
    // Subtask `t3.dfh.gy.ib.c8`'s.

    /// A tail meta whose lineage fields are set: `CheckpointMeta` with `kind =
    /// Tail`, `base_snap_seq`, and `from_seq` populated.
    fn tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(base_snap_seq),
            from_seq: Some(from_seq),
            snap_seq: Some(to_seq),
            // Tails are NOT exempt from the identity gate — see
            // `validate_generation_identity`'s "both kinds" section.
            ..crate::checkpoint::adapters::identity_default()
        }
    }

    /// Replaces the interim "tail put returns NotImplemented" arm: a tail put
    /// with complete lineage now succeeds, mints a `Tail` handle under this
    /// adapter's `:tail` key, and round-trips its bytes and its **whole** meta
    /// verbatim.
    ///
    /// The payload is deliberately not OMPF-shaped: SR-1.5 makes the envelope
    /// opaque bytes to the adapter, so a byte sequence that could not survive a
    /// parse step is the honest fixture — an implementation that grew any
    /// interpretation of the buffer fails here rather than in Epic 4.
    ///
    /// Meta is checked field-by-field rather than by equality because
    /// `CheckpointMeta` derives no `PartialEq`; every field the struct carries
    /// is covered, including the identity fields the apply path rebuilds
    /// `obj:hdr` from. InMemory does not serialize, so *verbatim* is the whole
    /// contract here — there is no serialize/deserialize step that could drop a
    /// field, and a tail's lineage fields must not be an exception.
    #[tokio::test]
    async fn put_layer_checkpoint_with_tail_kind_stores_and_round_trips_the_envelope() {
        let store = InMemoryCheckpointStore::new();

        // Every field distinct and non-default, so a read that returns some
        // *other* generation's meta — or a zeroed one — is distinguishable from
        // a true verbatim round-trip.
        let meta = CheckpointMeta {
            creator_replica_id: "replica-tail".to_string(),
            written_at_unix_ms: 1_700_000_000_123,
            session_id: "sess-tail".to_string(),
            layer_name: "layer-tail".to_string(),
            created_by: "user-tail".to_string(),
            server_ts: 1_700_000_500_456,
            observed_commit_span: 37,
            ..tail_meta(4, 5, 8)
        };

        let handle = store
            .put_layer_checkpoint(12, 8, Bytes::from_static(b"\x00\xffnot-ompf"), meta)
            .await
            .expect("a lineage-complete tail put must succeed");
        assert_eq!(
            handle.kind,
            CheckpointKind::Tail,
            "the returned handle's kind MUST equal meta.kind"
        );
        assert_eq!(
            handle.adapter_key, "12:8:tail",
            "a tail's canonical key must be distinct from the base's `12:8`"
        );

        let blob = store
            .get_layer_checkpoint(&handle)
            .await
            .expect("get")
            .expect("the stored tail must be gettable by its handle");
        assert_eq!(
            blob.bytes,
            Bytes::from_static(b"\x00\xffnot-ompf"),
            "opaque bytes must come back byte-identical"
        );
        assert_eq!(
            blob.handle, handle,
            "the handle a get returns must equal the one the put minted"
        );
        // Lineage fields.
        assert_eq!(blob.meta.kind, CheckpointKind::Tail);
        assert_eq!(blob.meta.base_snap_seq, Some(4));
        assert_eq!(blob.meta.from_seq, Some(5));
        assert_eq!(blob.meta.snap_seq, Some(8));
        // Identity / provenance fields — same verbatim obligation.
        assert_eq!(blob.meta.creator_replica_id, "replica-tail");
        assert_eq!(blob.meta.written_at_unix_ms, 1_700_000_000_123);
        assert_eq!(blob.meta.session_id, "sess-tail");
        assert_eq!(blob.meta.layer_name, "layer-tail");
        assert_eq!(blob.meta.created_by, "user-tail");
        assert_eq!(blob.meta.server_ts, 1_700_000_500_456);
        assert_eq!(blob.meta.observed_commit_span, 37);
        assert!(
            blob.meta.sidecar_bytes.is_none(),
            "a tail never carries an ext-refs sidecar (SR-2.4)"
        );
    }

    /// The put-time lineage gate (`validate_tail_put`) rejects a tail with a
    /// missing `base_snap_seq` / `from_seq` as `Permanent` and stores nothing —
    /// replay would otherwise have to *infer* which base the envelope layers
    /// onto.
    ///
    /// All three incomplete shapes are covered, not just "both absent": a gate
    /// that only fired when *neither* field was set would pass a one-field
    /// check and still hand replay a tail it cannot place.
    ///
    /// Classification is identical to LocalDisk's **by construction** — both
    /// adapters call the one `validate_tail_put` in `adapters/mod.rs` before
    /// touching their backend, so a divergence would be a second, open-coded
    /// copy of the rule rather than a differently-tuned constant.
    #[tokio::test]
    async fn put_layer_checkpoint_with_lineage_incomplete_tail_is_permanent_and_stores_nothing() {
        let store = InMemoryCheckpointStore::new();

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
                .put_layer_checkpoint(13, 8, Bytes::from_static(b"tail"), incomplete)
                .await
            {
                Err(err) => err,
                Ok(handle) => {
                    panic!(
                        "a lineage-incomplete tail put ({label}) must be rejected; got {handle:?}"
                    )
                }
            };
            assert!(
                matches!(err, CheckpointError::Permanent { .. }),
                "lineage-incomplete tail put ({label}) must classify as Permanent, got {err:?}"
            );
            assert!(
                store.list_layer_checkpoints(13).await.unwrap().is_empty(),
                "a rejected tail put ({label}) must leave no generation behind"
            );
        }
    }

    /// A tail handle whose `to_seq` collides with an existing base's
    /// `snap_seq` MUST NOT resolve to that base — otherwise a caller would be
    /// handed a full snapshot labelled as a tail. Still `Ok(None)` (not
    /// `NotFound`) when no tail was stored at that sequence.
    #[tokio::test]
    async fn get_layer_checkpoint_on_a_tail_handle_returns_ok_none_even_when_a_base_shares_the_seq()
    {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(20, 5, Bytes::from_static(b"base-bytes"), test_meta())
            .await
            .expect("put base");

        let tail_handle = CheckpointHandle::tail(20, 5, "20:5".to_string());
        match store.get_layer_checkpoint(&tail_handle).await {
            Ok(None) => {}
            other => panic!("a tail handle must resolve to Ok(None), got {other:?}"),
        }
    }

    /// The load-bearing tail-delete assertion, now against real storage: a base
    /// and a tail coexisting at sequence `N` are deleted independently. A tail
    /// delete taking out the base sharing its sequence (or the reverse) is the
    /// stranded-generation shape SR-2.4 / SR-2.5 forbid.
    #[tokio::test]
    async fn delete_layer_checkpoint_on_a_tail_handle_leaves_the_same_sequence_base_intact() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(21, 5, Bytes::from_static(b"base-bytes"), test_meta())
            .await
            .expect("put base");
        let tail = store
            .put_layer_checkpoint(21, 5, Bytes::from_static(b"tail-bytes"), tail_meta(2, 3, 5))
            .await
            .expect("put tail at the same sequence as the base");

        store
            .delete_layer_checkpoint(tail.clone())
            .await
            .expect("tail delete");

        let base = store
            .get_layer_checkpoint(&CheckpointHandle::base(21, 5, "21:5".to_string()))
            .await
            .expect("get")
            .expect("the base at the same sequence must survive a tail delete");
        assert_eq!(base.bytes, Bytes::from_static(b"base-bytes"));
        assert!(
            store.get_layer_checkpoint(&tail).await.unwrap().is_none(),
            "the deleted tail must be gone"
        );
        let handles = store.list_layer_checkpoints(21).await.unwrap();
        assert_eq!(
            handles.len(),
            1,
            "only the base generation must remain enumerable; got {handles:?}"
        );
        assert_eq!(handles[0].kind, CheckpointKind::Base);

        // And re-deleting the already-absent tail is still `Ok(())`.
        store
            .delete_layer_checkpoint(tail)
            .await
            .expect("delete of an already-absent tail must be Ok, not NotFound");
    }

    /// The two-level prune: a sequence key survives while the other kind is
    /// still present at it, and the `object_id` key is released only once the
    /// last generation of any kind is gone. Without the first half a base
    /// delete would strand its sibling tail; without the second,
    /// `list_all_object_ids` would keep reporting a fully-deleted object.
    #[tokio::test]
    async fn delete_layer_checkpoint_two_level_prune() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 30;

        let base = store
            .put_layer_checkpoint(oid, 5, Bytes::from_static(b"base"), test_meta())
            .await
            .expect("put base");
        let tail = store
            .put_layer_checkpoint(oid, 5, Bytes::from_static(b"tail"), tail_meta(2, 3, 5))
            .await
            .expect("put tail");

        store.delete_layer_checkpoint(base).await.expect("delete");
        {
            let state = store.state.lock().await;
            let slot = state.layer_checkpoints[&oid]
                .get(&5)
                .expect("sequence key must survive while the tail remains");
            assert!(slot.base.is_none());
            assert!(slot.tail.is_some());
        }
        assert_eq!(
            store.list_all_object_ids().await.unwrap(),
            vec![oid],
            "an object holding only a tail generation must still be enumerated"
        );
        // Enumeration alone is not enough: the surviving tail must still be
        // *fetchable*, with its own bytes. A prune that dropped the entry but
        // left the key would satisfy the assertion above and still have lost
        // the generation.
        let survivor = store
            .get_layer_checkpoint(&tail)
            .await
            .expect("get")
            .expect("the tail must survive the deletion of the last base");
        assert_eq!(survivor.bytes, Bytes::from_static(b"tail"));
        assert_eq!(survivor.handle.kind, CheckpointKind::Tail);

        store.delete_layer_checkpoint(tail).await.expect("delete");
        let state = store.state.lock().await;
        assert!(
            !state.layer_checkpoints.contains_key(&oid),
            "the object_id key must be released once its last generation of any kind is gone"
        );
    }

    /// `list_layer_checkpoints` is one strict descending order over the
    /// sequence with kinds interleaved (SR-2.3), Base before Tail at a tie —
    /// and `get_latest_layer_checkpoint` agrees with its head even when that
    /// head is a **tail**. Filtering `get_latest` to bases is the tempting and
    /// wrong change this pins against.
    #[tokio::test]
    async fn mixed_kind_listing_is_interleaved_and_get_latest_can_return_a_tail() {
        let store = InMemoryCheckpointStore::new();
        let oid: u64 = 31;

        store
            .put_layer_checkpoint(oid, 5, Bytes::from_static(b"base-5"), test_meta())
            .await
            .expect("put");
        store
            .put_layer_checkpoint(oid, 5, Bytes::from_static(b"tail-5"), tail_meta(1, 2, 5))
            .await
            .expect("put");
        store
            .put_layer_checkpoint(oid, 7, Bytes::from_static(b"base-7"), test_meta())
            .await
            .expect("put");
        store
            .put_layer_checkpoint(oid, 9, Bytes::from_static(b"tail-9"), tail_meta(7, 8, 9))
            .await
            .expect("put");

        let handles = store.list_layer_checkpoints(oid).await.expect("list");
        let observed: Vec<(u64, CheckpointKind)> =
            handles.iter().map(|h| (h.seq(), h.kind)).collect();
        assert_eq!(
            observed,
            vec![
                (9, CheckpointKind::Tail),
                (7, CheckpointKind::Base),
                (5, CheckpointKind::Base),
                (5, CheckpointKind::Tail),
            ],
            "newest-first over the sequence, Base before Tail at the tie"
        );

        let latest = store
            .get_latest_layer_checkpoint(oid)
            .await
            .expect("get_latest")
            .expect("generations exist");
        let head = store
            .get_layer_checkpoint(&handles[0])
            .await
            .expect("get by handle")
            .expect("the head handle resolves");
        assert_eq!(
            latest.handle, head.handle,
            "get_latest MUST equal list().first(), including when that is a tail"
        );
        assert_eq!(latest.handle.kind, CheckpointKind::Tail);
        assert_eq!(latest.bytes, Bytes::from_static(b"tail-9"));
    }

    /// The injector's short-circuit position: `maybe_inject` is the first
    /// statement of `put_layer_checkpoint`, ahead of `validate_tail_put`, so a
    /// harness failing a put on a lineage-incomplete tail sees the *injected*
    /// error rather than the validation error.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test]
    async fn put_layer_checkpoint_injects_before_tail_validation() {
        struct AlwaysTransient;
        impl CheckpointFaultInjector for AlwaysTransient {
            fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
                if op == "put_layer_checkpoint" {
                    return Err(CheckpointError::Transient {
                        source: "injected".into(),
                    });
                }
                Ok(())
            }
        }

        let store = InMemoryCheckpointStore::new().with_fault_injector(Arc::new(AlwaysTransient));
        // Lineage-incomplete on purpose: validation would classify this
        // `Permanent`, so seeing `Transient` proves the injector ran first.
        let incomplete = CheckpointMeta {
            kind: CheckpointKind::Tail,
            ..CheckpointMeta::default()
        };
        let err = store
            .put_layer_checkpoint(40, 1, Bytes::from_static(b"tail"), incomplete)
            .await
            .expect_err("the injector must short-circuit the put");
        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "injection MUST run ahead of tail validation; got {err:?}"
        );
    }

    /// A store holding a base-with-sidecar and a tail reports exactly the
    /// base's hashes from `list_referenced_ext_hashes` — tails are
    /// structurally invisible to the retention cascade's live set (SR-2.4).
    /// Companion to the `read_ext_refs_sidecar` tail arm below, which covers
    /// the per-handle view.
    #[tokio::test]
    async fn list_referenced_ext_hashes_ignores_tail_generations() {
        let store = InMemoryCheckpointStore::new();
        let with_sidecar = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["aa", "bb"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(41, 5, Bytes::from_static(b"base"), with_sidecar)
            .await
            .expect("put base");
        store
            .put_layer_checkpoint(41, 9, Bytes::from_static(b"tail"), tail_meta(5, 6, 9))
            .await
            .expect("put tail");

        let refs = store.list_referenced_ext_hashes().await.unwrap();
        let expected: HashSet<String> = ["aa", "bb"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "only the base sidecar's hashes belong to the live set"
        );
    }

    /// Both generation families report, each with its own members: a base
    /// contributes `Ompf` + `Meta` (+ `ExtRefs` when a sidecar was paired) and a
    /// tail contributes `TailEnvelope` + `TailMeta` and never an `ExtRefs`.
    ///
    /// The base+tail pair at 42/5 is the case the kind-in-the-key grouping
    /// exists for: with a sidecar-bearing base it is five artifacts at one
    /// numeric stem, split three/two across the two families, not one doubled
    /// stem. The tail-only object 43 pins that a tail alone makes an object
    /// visible to the pass, and that it contributes no `ExtRefs` even with a
    /// sidecar-bearing base in the same store.
    ///
    /// The per-slot emission order (base members, then tail members) is pinned
    /// positionally; the outer `HashMap` over `object_id` has no defined order,
    /// so the actual vector is stably sorted by `object_id` first — which leaves
    /// each object's internal emission order untouched.
    #[tokio::test]
    async fn list_layer_artifacts_reports_both_generation_families_with_kind_correct_members() {
        let store = InMemoryCheckpointStore::new();
        let base_meta = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["aa"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(42, 5, Bytes::from_static(b"base"), base_meta)
            .await
            .expect("put base");
        store
            .put_layer_checkpoint(42, 5, Bytes::from_static(b"tail"), tail_meta(1, 2, 5))
            .await
            .expect("put tail at the same sequence");
        store
            .put_layer_checkpoint(43, 9, Bytes::from_static(b"tail-only"), tail_meta(1, 2, 9))
            .await
            .expect("put tail-only object");

        let mut artifacts = store.list_layer_artifacts().await.unwrap();
        artifacts.sort_by_key(|artifact| artifact.object_id);
        let expected = vec![
            LayerArtifact {
                object_id: 42,
                snap_seq: 5,
                kind: LayerArtifactKind::Ompf,
            },
            LayerArtifact {
                object_id: 42,
                snap_seq: 5,
                kind: LayerArtifactKind::Meta,
            },
            LayerArtifact {
                object_id: 42,
                snap_seq: 5,
                kind: LayerArtifactKind::ExtRefs,
            },
            LayerArtifact {
                object_id: 42,
                snap_seq: 5,
                kind: LayerArtifactKind::TailEnvelope,
            },
            LayerArtifact {
                object_id: 42,
                snap_seq: 5,
                kind: LayerArtifactKind::TailMeta,
            },
            LayerArtifact {
                object_id: 43,
                snap_seq: 9,
                kind: LayerArtifactKind::TailEnvelope,
            },
            LayerArtifact {
                object_id: 43,
                snap_seq: 9,
                kind: LayerArtifactKind::TailMeta,
            },
        ];
        assert_eq!(
            artifacts, expected,
            "the base+tail pair at 42/5 must report five kind-correct members and the tail-only \
             object 43 its own two"
        );
    }

    /// **FINAL behaviour, not a placeholder** (SR-2.4): a tail carries no
    /// ext-refs sidecar by design, so the empty set is the answer Tasks 3/4
    /// must preserve rather than "complete". Falling through to the base's
    /// sidecar would report the BASE's hashes as the tail's and inflate the
    /// retention cascade's per-handle pruned set.
    #[tokio::test]
    async fn read_ext_refs_sidecar_on_a_tail_handle_returns_empty_and_not_the_bases_hashes() {
        let store = InMemoryCheckpointStore::new();
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
            .read_ext_refs_sidecar(&CheckpointHandle::base(22, 5, "22:5".to_string()))
            .await
            .expect("base read");
        assert_eq!(base_refs.len(), 2, "test precondition: base has 2 hashes");

        let tail_refs = store
            .read_ext_refs_sidecar(&CheckpointHandle::tail(22, 5, "22:5".to_string()))
            .await
            .expect("tail read must succeed");
        assert!(
            tail_refs.is_empty(),
            "a tail handle's ext-refs set is empty by design; got {tail_refs:?}"
        );
    }

    // ----- t3.dfh.gy.ib.c8: the mixed-kind matrix over real tail storage -----
    //
    // The block above pins each individual tail arm as `t3.dfh.gy.ib.6x`
    // completed it. These pin the *combinations* — the properties that only
    // appear when both kinds are resident at once, and the ones that are
    // statements about this adapter's chosen data structure rather than about
    // the trait (the cross-adapter contract is Task 6's
    // `tests/integration/checkpoint/conformance.rs`, which this subtask adds
    // nothing to).

    /// **The map-shape guard.** A base at sequence `N` and a tail whose
    /// `to_seq` is `N` are two different artifacts that genuinely coexist —
    /// SR-4.1 re-base mints a base at the newest consumed tail's `to_seq`, and
    /// SR-9.2 pins re-base idempotency at `(object_id, S)`. An adapter keyed on
    /// `(object_id, sequence)` alone silently drops one of them on the second
    /// put, and every *other* test in this file still passes: the survivor
    /// round-trips, lists, and deletes perfectly well. This is the test that
    /// fails, which is the whole reason it exists.
    ///
    /// Both delete directions run here against a pristine coexisting pair. The
    /// tail-delete direction overlaps
    /// `delete_layer_checkpoint_on_a_tail_handle_leaves_the_same_sequence_base_intact`
    /// deliberately — that test pins the *delete arm's* supersession, this one
    /// pins that the pair was ever separately addressable to begin with.
    #[tokio::test]
    async fn base_and_tail_at_the_same_sequence_coexist_and_survive_each_others_delete() {
        let store = InMemoryCheckpointStore::new();
        // 60 exercises the delete-the-base direction, 61 the delete-the-tail
        // direction; each starts from its own untouched pair.
        for oid in [60_u64, 61] {
            store
                .put_layer_checkpoint(oid, 5, Bytes::from_static(b"base-bytes"), test_meta())
                .await
                .expect("put base at 5");
            store
                .put_layer_checkpoint(
                    oid,
                    5,
                    Bytes::from_static(b"tail-bytes"),
                    tail_meta(2, 3, 5),
                )
                .await
                .expect("put tail at to_seq 5");
        }

        // Neither put clobbered the other: two handles, one per kind, both at
        // the same sequence, each resolving to its OWN bytes.
        for oid in [60_u64, 61] {
            let handles = store.list_layer_checkpoints(oid).await.expect("list");
            assert_eq!(
                handles.len(),
                2,
                "a base and a tail at one sequence are two generations; got {handles:?}"
            );
            assert_eq!(
                handles.iter().map(|h| h.kind).collect::<Vec<_>>(),
                vec![CheckpointKind::Base, CheckpointKind::Tail],
                "Base before Tail at an equal sequence (the pinned tie-break)"
            );
            assert!(
                handles.iter().all(|h| h.seq() == 5),
                "both generations sit at sequence 5"
            );
            assert_ne!(
                handles[0].adapter_key, handles[1].adapter_key,
                "the two kinds must be addressable under distinct adapter keys"
            );

            let base = store
                .get_layer_checkpoint(&handles[0])
                .await
                .expect("get base")
                .expect("the base is resident");
            let tail = store
                .get_layer_checkpoint(&handles[1])
                .await
                .expect("get tail")
                .expect("the tail is resident");
            assert_eq!(base.bytes, Bytes::from_static(b"base-bytes"));
            assert_eq!(tail.bytes, Bytes::from_static(b"tail-bytes"));
            assert_eq!(base.meta.kind, CheckpointKind::Base);
            assert_eq!(tail.meta.kind, CheckpointKind::Tail);
            assert_eq!(
                tail.meta.base_snap_seq,
                Some(2),
                "the tail's lineage must not have been overwritten by the base's meta"
            );
        }

        // Direction 1 — delete the base; the tail survives with its own bytes.
        let base_60 = CheckpointHandle::base(60, 5, "60:5".to_string());
        store
            .delete_layer_checkpoint(base_60.clone())
            .await
            .expect("delete base");
        assert!(
            store
                .get_layer_checkpoint(&base_60)
                .await
                .unwrap()
                .is_none(),
            "the deleted base must be gone"
        );
        let survivor = store
            .get_layer_checkpoint(&CheckpointHandle::tail(60, 5, "60:5:tail".to_string()))
            .await
            .expect("get")
            .expect("the tail must survive a base delete at the same sequence");
        assert_eq!(survivor.bytes, Bytes::from_static(b"tail-bytes"));
        let remaining = store.list_layer_checkpoints(60).await.expect("list");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].kind, CheckpointKind::Tail);

        // Direction 2 — delete the tail; the base survives with its own bytes.
        let tail_61 = CheckpointHandle::tail(61, 5, "61:5:tail".to_string());
        store
            .delete_layer_checkpoint(tail_61.clone())
            .await
            .expect("delete tail");
        assert!(
            store
                .get_layer_checkpoint(&tail_61)
                .await
                .unwrap()
                .is_none(),
            "the deleted tail must be gone"
        );
        let survivor = store
            .get_layer_checkpoint(&CheckpointHandle::base(61, 5, "61:5".to_string()))
            .await
            .expect("get")
            .expect("the base must survive a tail delete at the same sequence");
        assert_eq!(survivor.bytes, Bytes::from_static(b"base-bytes"));
        let remaining = store.list_layer_checkpoints(61).await.expect("list");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].kind, CheckpointKind::Base);
    }

    /// Mixed-kind listing is a **function of the stored set**, not of insertion
    /// order or of when it is called.
    ///
    /// `mixed_kind_listing_is_interleaved_and_get_latest_can_return_a_tail`
    /// pins the order for one fixture; this pins that the order is stable —
    /// five repeat calls agree byte-for-byte, and seeding tails-before-bases
    /// produces the identical listing to bases-before-tails. A `HashMap` keyed
    /// on the sequence, or a two-map implementation that concatenated its two
    /// runs, reproduces the expected order intermittently and fails here.
    ///
    /// The `tie` phase adds a tail at 4 alongside the base at 4 and asserts the
    /// pinned Base-before-Tail tie-break holds under both insertion orders —
    /// the tie is exactly where an order that leaned on insertion sequence
    /// would show it.
    #[tokio::test]
    async fn mixed_kind_listing_is_deterministic_and_insertion_order_independent() {
        const OID: u64 = 62;

        // Bases at 1 and 4; tails at 2 and 7, plus (when `tie`) a tail whose
        // `to_seq` is 4 — colliding with the base at 4.
        async fn seed(tails_first: bool, tie: bool) -> InMemoryCheckpointStore {
            let store = InMemoryCheckpointStore::new();
            let bases: [u64; 2] = [1, 4];
            let mut tails: Vec<(u64, u64, u64)> = vec![(1, 2, 2), (4, 5, 7)];
            if tie {
                tails.push((1, 2, 4));
            }

            async fn put_bases(store: &InMemoryCheckpointStore, bases: [u64; 2]) {
                for seq in bases {
                    store
                        .put_layer_checkpoint(
                            OID,
                            seq,
                            Bytes::from(format!("base-{seq}")),
                            test_meta(),
                        )
                        .await
                        .expect("put base");
                }
            }
            async fn put_tails(store: &InMemoryCheckpointStore, tails: &[(u64, u64, u64)]) {
                for &(base_snap_seq, from_seq, to_seq) in tails {
                    store
                        .put_layer_checkpoint(
                            OID,
                            to_seq,
                            Bytes::from(format!("tail-{to_seq}")),
                            tail_meta(base_snap_seq, from_seq, to_seq),
                        )
                        .await
                        .expect("put tail");
                }
            }

            if tails_first {
                put_tails(&store, &tails).await;
                put_bases(&store, bases).await;
            } else {
                put_bases(&store, bases).await;
                put_tails(&store, &tails).await;
            }
            store
        }

        for tie in [false, true] {
            let expected: Vec<(u64, CheckpointKind)> = if tie {
                vec![
                    (7, CheckpointKind::Tail),
                    (4, CheckpointKind::Base),
                    (4, CheckpointKind::Tail),
                    (2, CheckpointKind::Tail),
                    (1, CheckpointKind::Base),
                ]
            } else {
                vec![
                    (7, CheckpointKind::Tail),
                    (4, CheckpointKind::Base),
                    (2, CheckpointKind::Tail),
                    (1, CheckpointKind::Base),
                ]
            };

            // The full handle vector (adapter_key included) from the very first
            // listing of the very first insertion order — every other listing
            // must equal it exactly.
            let mut baseline: Option<Vec<CheckpointHandle>> = None;

            for tails_first in [false, true] {
                let store = seed(tails_first, tie).await;
                for round in 0..5 {
                    let handles = store.list_layer_checkpoints(OID).await.expect("list");
                    let observed: Vec<(u64, CheckpointKind)> =
                        handles.iter().map(|h| (h.seq(), h.kind)).collect();
                    assert_eq!(
                        observed, expected,
                        "tie={tie} tails_first={tails_first} round={round}: listing must be \
                         newest-sequence-first with Base before Tail at an equal sequence"
                    );
                    match baseline.as_ref() {
                        None => baseline = Some(handles),
                        Some(first) => assert_eq!(
                            &handles, first,
                            "tie={tie} tails_first={tails_first} round={round}: every listing of \
                             the same stored set must be byte-identical, adapter_key included"
                        ),
                    }
                }
            }
        }
    }

    /// `list_all_object_ids` is a set of `object_id`s, not of generations: an
    /// object appears exactly once regardless of how many generations it holds
    /// or of which kinds they are — **including a tail-only object**, which is
    /// the case a separate `tail_checkpoints` map would have hidden from the
    /// recovery hook's working set entirely.
    #[tokio::test]
    async fn list_all_object_ids_enumerates_each_object_once_across_kinds() {
        let store = InMemoryCheckpointStore::new();

        // (a) Tail-only object, two tail generations.
        for (base_snap_seq, from_seq, to_seq) in [(1_u64, 2_u64, 9_u64), (9, 10, 12)] {
            store
                .put_layer_checkpoint(
                    70,
                    to_seq,
                    Bytes::from_static(b"tail"),
                    tail_meta(base_snap_seq, from_seq, to_seq),
                )
                .await
                .expect("put tail");
        }

        // (b) Mixed-kind object: a base and a tail sharing sequence 5, plus a
        // second base at 8.
        store
            .put_layer_checkpoint(71, 5, Bytes::from_static(b"base"), test_meta())
            .await
            .expect("put base");
        store
            .put_layer_checkpoint(71, 5, Bytes::from_static(b"tail"), tail_meta(2, 3, 5))
            .await
            .expect("put tail");
        store
            .put_layer_checkpoint(71, 8, Bytes::from_static(b"base"), test_meta())
            .await
            .expect("put base");

        // Base-only control — the pre-tail shape, which must be unchanged.
        store
            .put_layer_checkpoint(72, 1, Bytes::from_static(b"base"), test_meta())
            .await
            .expect("put base");

        let mut ids = store.list_all_object_ids().await.expect("list_all");
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![70, 71, 72],
            "each object appears exactly once; the tail-only object 70 must appear at all"
        );
    }

    /// Deleting a tail that is not there is `Ok(())` and must disturb nothing —
    /// the retention pruner re-issues deletes without an existence check, and
    /// it addresses tails and bases through the same call.
    ///
    /// The load-bearing branch is (b): a tail delete naming a sequence that
    /// holds only a **base** must leave that base alone. An implementation that
    /// cleared the slot rather than the slot's `kind` field would pass every
    /// idempotency assertion here and silently destroy the base.
    #[tokio::test]
    async fn delete_layer_checkpoint_absent_tail_handle_is_idempotent_and_disturbs_nothing() {
        let store = InMemoryCheckpointStore::new();

        // (a) Nothing was ever put — a purely synthetic tail handle.
        store
            .delete_layer_checkpoint(CheckpointHandle::tail(999, 1, "phantom".into()))
            .await
            .expect("delete of a never-put tail must be Ok, not NotFound");

        // (b) The object and the sequence exist, but hold a base only.
        let base = store
            .put_layer_checkpoint(73, 5, Bytes::from_static(b"base-bytes"), test_meta())
            .await
            .expect("put base");
        let absent_tail = CheckpointHandle::tail(73, 5, "73:5:tail".to_string());
        store
            .delete_layer_checkpoint(absent_tail.clone())
            .await
            .expect("delete of an absent tail at a live sequence must be Ok");
        let survivor = store
            .get_layer_checkpoint(&base)
            .await
            .expect("get")
            .expect("deleting an absent tail must not take out the base at that sequence");
        assert_eq!(survivor.bytes, Bytes::from_static(b"base-bytes"));
        assert_eq!(
            store.list_all_object_ids().await.unwrap(),
            vec![73],
            "the object must stay enumerated — nothing was actually removed"
        );

        // (c) The same absent-tail delete again — still Ok, still a no-op.
        store
            .delete_layer_checkpoint(absent_tail)
            .await
            .expect("a repeated delete must be Ok, not NotFound");
        assert!(
            store.get_layer_checkpoint(&base).await.unwrap().is_some(),
            "the base must still be resident after the repeated no-op delete"
        );
    }

    /// SR-2.4 / b.s5y are enforced by **rejection, not by silent dropping**: a
    /// tail put carrying `sidecar_bytes` is `Permanent` and stores nothing.
    ///
    /// Silently dropping the field would leave a caller with a bug that never
    /// surfaces; honouring it is worse — an ext-refs sidecar written under a
    /// tail stem is never reaped by the artifact-family-driven deletes, so its
    /// hashes pin into the live set forever and the retention cascade degrades
    /// toward deleting nothing.
    ///
    /// The base sharing the sequence is present *before* the rejected put so
    /// the assertions also cover the collateral case: a rejected tail put must
    /// not perturb the base's slot or its sidecar.
    #[tokio::test]
    async fn put_layer_checkpoint_tail_carrying_a_sidecar_is_permanent_and_stores_nothing() {
        let store = InMemoryCheckpointStore::new();
        let with_sidecar = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(74, 9, Bytes::from_static(b"base-bytes"), with_sidecar)
            .await
            .expect("put base with sidecar");

        let rejected = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["deadbeef"])),
            ..tail_meta(4, 5, 9)
        };
        let err = store
            .put_layer_checkpoint(74, 9, Bytes::from_static(b"tail-bytes"), rejected)
            .await
            .expect_err("a tail put carrying an ext-refs sidecar must be rejected");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a sidecar-bearing tail put must classify as Permanent (a caller bug that retrying \
             reproduces exactly); got {err:?}"
        );

        // Nothing was stored under the tail kind…
        let handles = store.list_layer_checkpoints(74).await.expect("list");
        assert_eq!(
            handles.len(),
            1,
            "the rejected tail must leave only the pre-existing base; got {handles:?}"
        );
        assert_eq!(handles[0].kind, CheckpointKind::Base);
        assert!(
            store
                .get_layer_checkpoint(&CheckpointHandle::tail(74, 9, "74:9:tail".to_string()))
                .await
                .unwrap()
                .is_none(),
            "the rejected tail must not be fetchable"
        );
        {
            let state = store.state.lock().await;
            assert!(
                state.layer_checkpoints[&74][&9].tail.is_none(),
                "the rejected put must not have populated the tail slot"
            );
        }

        // …and `deadbeef` never entered the live set, while the base's
        // `feedface` is untouched.
        let refs = store.list_referenced_ext_hashes().await.expect("live set");
        let expected: HashSet<String> = ["feedface"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            refs, expected,
            "a rejected tail's hashes must never reach the live set, and the base's must survive"
        );
    }

    /// A **genuinely stored** tail contributes nothing to the retention
    /// cascade's live set, and its handle never aliases the base sharing its
    /// sequence.
    ///
    /// `list_referenced_ext_hashes_ignores_tail_generations` covers the
    /// fleet-wide union with the tail at a *different* sequence, and
    /// `read_ext_refs_sidecar_on_a_tail_handle_returns_empty_and_not_the_bases_hashes`
    /// covers a synthetic tail handle for which no tail was ever stored. The
    /// case neither reaches is the one that matters most: a real tail resident
    /// in the slot *next to* a sidecar-bearing base at the same sequence, read
    /// through the real tail-storage path.
    #[tokio::test]
    async fn a_stored_tail_never_reaches_the_live_set_or_aliases_the_bases_sidecar() {
        let store = InMemoryCheckpointStore::new();
        let with_sidecar = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        let base = store
            .put_layer_checkpoint(75, 9, Bytes::from_static(b"base-bytes"), with_sidecar)
            .await
            .expect("put base with sidecar");
        // Lineage-complete, `sidecar_bytes: None` — the only tail shape the
        // put-time gate accepts.
        let tail = store
            .put_layer_checkpoint(75, 9, Bytes::from_static(b"tail-bytes"), tail_meta(4, 5, 9))
            .await
            .expect("put tail at the base's sequence");

        let expected: HashSet<String> = ["feedface"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            store.list_referenced_ext_hashes().await.expect("live set"),
            expected,
            "the live set is the union of base sidecars only"
        );
        // Precondition, so the empty assertion below cannot pass vacuously.
        assert_eq!(
            store.read_ext_refs_sidecar(&base).await.expect("base read"),
            expected,
            "test precondition: the base at 9 does report its hashes"
        );
        let tail_refs = store
            .read_ext_refs_sidecar(&tail)
            .await
            .expect("tail read must succeed");
        assert!(
            tail_refs.is_empty(),
            "a resident tail's ext-refs set is empty — never the base's hashes at the same \
             sequence; got {tail_refs:?}"
        );

        // And the stored shape backs that up: the tail slot holds no sidecar at
        // all, so there is nothing for a later edit to start unioning.
        let state = store.state.lock().await;
        let slot = &state.layer_checkpoints[&75][&9];
        assert!(
            slot.tail.as_ref().expect("tail resident").sidecar.is_none(),
            "put_layer_checkpoint stores `sidecar: None` on every tail"
        );
        assert!(
            slot.base.as_ref().expect("base resident").sidecar.is_some(),
            "the base's sidecar must be untouched by the tail put"
        );
    }

    /// The kind-slotted analogue of `re_put_layer_checkpoint_clears_prior_sidecar`.
    ///
    /// Its literal shape is unconstructible for a tail — a tail slot can never
    /// hold a sidecar to resurrect — so the property re-puts must preserve is
    /// the pair of them: the re-put tail's slot stays `sidecar: None`, and the
    /// **base** slot's sidecar at the same sequence is not cleared as collateral
    /// damage. Last-writer-wins is per kind, so a re-put that reset the whole
    /// slot would silently drop the base's hashes out of the live set and let
    /// the orphan sweep delete blobs the base still references.
    #[tokio::test]
    async fn re_putting_a_tail_leaves_slot_sidecars_untouched() {
        let store = InMemoryCheckpointStore::new();
        let with_sidecar = CheckpointMeta {
            sidecar_bytes: Some(sidecar_for(&["feedface"])),
            ..test_meta()
        };
        store
            .put_layer_checkpoint(76, 9, Bytes::from_static(b"base-bytes"), with_sidecar)
            .await
            .expect("put base with sidecar");

        let tail = store
            .put_layer_checkpoint(76, 9, Bytes::from_static(b"tail-v1"), tail_meta(4, 5, 9))
            .await
            .expect("put tail");
        store
            .put_layer_checkpoint(76, 9, Bytes::from_static(b"tail-v2"), tail_meta(4, 5, 9))
            .await
            .expect("re-put the tail at the same (object_id, to_seq)");

        // Last-writer-wins on the tail slot only.
        let blob = store
            .get_layer_checkpoint(&tail)
            .await
            .expect("get")
            .expect("the re-put tail is resident");
        assert_eq!(
            blob.bytes,
            Bytes::from_static(b"tail-v2"),
            "the second tail put replaces the first"
        );
        let base = store
            .get_layer_checkpoint(&CheckpointHandle::base(76, 9, "76:9".to_string()))
            .await
            .expect("get")
            .expect("the base must survive a tail re-put");
        assert_eq!(base.bytes, Bytes::from_static(b"base-bytes"));

        let expected: HashSet<String> = ["feedface"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            store.list_referenced_ext_hashes().await.expect("live set"),
            expected,
            "a tail re-put must not clear the sibling base's sidecar out of the live set"
        );

        let state = store.state.lock().await;
        let slot = &state.layer_checkpoints[&76][&9];
        assert!(
            slot.tail.as_ref().expect("tail resident").sidecar.is_none(),
            "a re-put tail's slot must still hold no sidecar"
        );
        assert!(
            slot.base.as_ref().expect("base resident").sidecar.is_some(),
            "the base slot's sidecar must be untouched by the tail re-put"
        );
    }

    /// Every tail-reachable method consults `maybe_inject` at the top of its
    /// body, under the **same** op label the base path uses — the label names
    /// the *method*, not the *kind* (`adapters/mod.rs`, "Fault-injection label
    /// policy").
    ///
    /// This is what lets Task 6's conformance harness drive a tail through
    /// `Transient` / `Conflict` at all: a tail path that grew its own
    /// early-return before the injection check, or one that answered under a
    /// kind-qualified label, would be silently undrivable — the harness would
    /// observe a success and report a passing case.
    ///
    /// `Conflict` is the injected classification precisely because no tail path
    /// produces it natively, so it cannot be confused with a real verdict.
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

        fn store_failing(op: &'static str) -> InMemoryCheckpointStore {
            InMemoryCheckpointStore::new().with_fault_injector(Arc::new(FailOne { op }))
        }

        /// Seed a base and a tail sharing sequence 5 on object 80.
        async fn seed(store: &InMemoryCheckpointStore) -> (CheckpointHandle, CheckpointHandle) {
            let base = store
                .put_layer_checkpoint(80, 5, Bytes::from_static(b"base-bytes"), test_meta())
                .await
                .expect("seed put base");
            let tail = store
                .put_layer_checkpoint(80, 5, Bytes::from_static(b"tail-bytes"), tail_meta(2, 3, 5))
                .await
                .expect("seed put tail");
            (base, tail)
        }

        // --- put_layer_checkpoint: the tail put short-circuits, and nothing
        // --- was stored.
        {
            let store = store_failing("put_layer_checkpoint");
            let err = store
                .put_layer_checkpoint(80, 5, Bytes::from_static(b"tail"), tail_meta(2, 3, 5))
                .await
                .expect_err("the injector must short-circuit the tail put");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "the injected classification must surface verbatim; got {err:?}"
            );
            assert!(
                store.state.lock().await.layer_checkpoints.is_empty(),
                "a short-circuited put must store nothing at all"
            );
            // The same label governs the base path — one label, both kinds.
            assert!(
                store
                    .put_layer_checkpoint(80, 5, Bytes::from_static(b"base"), test_meta())
                    .await
                    .is_err(),
                "the base put must answer to the same op label as the tail put"
            );
        }

        // --- get_layer_checkpoint on a tail handle.
        {
            let store = store_failing("get_layer_checkpoint");
            let (base, tail) = seed(&store).await;
            let err = store
                .get_layer_checkpoint(&tail)
                .await
                .expect_err("the injector must short-circuit the tail get");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "tail get must surface the injected error rather than Ok(None); got {err:?}"
            );
            assert!(
                store.get_layer_checkpoint(&base).await.is_err(),
                "the base get must answer to the same op label"
            );
        }

        // --- delete_layer_checkpoint on a tail handle: short-circuits, and the
        // --- tail is still resident afterwards.
        {
            let store = store_failing("delete_layer_checkpoint");
            let (base, tail) = seed(&store).await;
            let err = store
                .delete_layer_checkpoint(tail.clone())
                .await
                .expect_err("the injector must short-circuit the tail delete");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "tail delete must surface the injected error rather than Ok(()); got {err:?}"
            );
            assert!(
                store
                    .get_layer_checkpoint(&tail)
                    .await
                    .expect("get is not injected here")
                    .is_some(),
                "a short-circuited delete must not have removed the tail"
            );
            assert!(
                store.delete_layer_checkpoint(base).await.is_err(),
                "the base delete must answer to the same op label"
            );
        }

        // --- list_layer_checkpoints over a mixed-kind object.
        {
            let store = store_failing("list_layer_checkpoints");
            seed(&store).await;
            let err = store
                .list_layer_checkpoints(80)
                .await
                .expect_err("the injector must short-circuit the mixed-kind listing");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "listing must surface the injected error rather than an empty Vec; got {err:?}"
            );
        }

        // --- list_all_object_ids over a tail-only object: the enumeration a
        // --- tail-unaware implementation would have answered from a map the
        // --- injector never guards.
        {
            let store = store_failing("list_all_object_ids");
            store
                .put_layer_checkpoint(81, 9, Bytes::from_static(b"tail"), tail_meta(1, 2, 9))
                .await
                .expect("seed tail-only object");
            let err = store
                .list_all_object_ids()
                .await
                .expect_err("the injector must short-circuit the enumeration");
            assert!(
                matches!(err, CheckpointError::Conflict),
                "enumeration must surface the injected error; got {err:?}"
            );
        }
    }
}
