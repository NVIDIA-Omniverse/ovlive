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

//! Checkpoint adapter implementations.
//!
//! Each child module owns one [`CheckpointStore`](super::CheckpointStore)
//! backend:
//!
//! - [`in_memory`] — process-local `HashMap`-backed adapter; tests and
//!   single-replica dev only. Implemented in sibling Subtask `t3.fgj.ya.r7.jm`.
//! - [`local_disk`] — pod-local filesystem adapter; the Epic 1 default-on
//!   adapter. Implemented in sibling Subtask `t3.fgj.ya.r7.mv`.
//! - [`storage_api`] — the storage-api-backed adapter: HTTP artifacts under a
//!   configured base path, service-token auth, write-redirect handling and the
//!   reconciliation enumeration surface. No longer the construction-time stub
//!   this list once described.
//!
//! The [`CheckpointFaultInjector`] trait is a test-only seam that
//! `test-checkpoint-fault-injection` gates so the SR-1.3 conformance harness
//! can drive `Transient` / `Conflict` paths through the InMemory and LocalDisk
//! adapters without touching their production code paths.
//!
//! ## Tail generations
//!
//! Two rules of the SR-2.3 / SR-2.4 tail contract are needed **identically** by
//! all three adapters, so they live here once and each adapter calls them
//! rather than re-deriving them:
//!
//! - [`validate_tail_put`] — the put-time lineage / no-sidecar gate every
//!   `put_layer_checkpoint` runs before it touches its backend.
//! - [`validate_generation_identity`] — its sibling gate, kind-independent:
//!   every persisted generation must carry a non-empty
//!   `session_id`/`layer_name`.
//! - [`generation_sort_key`] (and its handle-taking companion
//!   [`handle_generation_sort_key`]) — the canonical mixed-kind listing order,
//!   including the pinned Base-before-Tail tie-break at an equal sequence.
//!
//! These two are the single source of truth for those rules. An adapter that
//! open-codes either one is a bug: Epic 4's lineage gather and Epic 5's
//! retention cascade both key on `(object_id, sequence, kind)`, so a
//! per-adapter copy is how the three copies drift on exactly the contract the
//! cross-Epic work depends on.
//!
//! The shared artifact-naming convention is likewise cross-adapter: a tail's
//! **stem** is `{oid}.{to_seq}.tail` — a LocalDisk filename stem and a
//! StorageApi address stem — alongside the pre-existing base stem
//! `{oid}.{snap_seq}`. Epic 4's gather and Epic 5's retention cascade address
//! generations by that stem, so it is contract, not adapter-local formatting.
//! The reconciliation artifact surface
//! ([`CheckpointStore::list_layer_artifacts`](super::CheckpointStore::list_layer_artifacts))
//! parses those same stems in both families, so every name an adapter renders
//! and every name it parses is built from the one set of suffix constants in
//! `store.rs` (`OMPF_SUFFIX`, `EXT_REFS_SUFFIX`, `META_SUFFIX`, `TAIL_SUFFIX`,
//! `TAIL_META_SUFFIX`, `TEMP_SUFFIX`) — never an inline literal.
//!
//! ### Fault-injection label policy
//!
//! Tail work adds **no** kind-qualified op labels to
//! [`CheckpointFaultInjector`]. It reuses the existing method-named labels
//! (`"put_layer_checkpoint"`, `"get_layer_checkpoint"`,
//! `"list_layer_checkpoints"`, `"delete_layer_checkpoint"`,
//! `"list_all_object_ids"`): the label names the *method*, not the *kind*. A
//! conformance case that needs to fail a tail put but not a base put keys on
//! injector-side state (call ordering, or a flag the harness flips), which the
//! free-form `maybe_inject(op: &str)` seam already permits. LocalDisk
//! publish-order failures are driven by `EISDIR` path-poisoning rather than
//! injection (see `put_layer_checkpoint_rolls_back_the_sidecar_when_the_ompf_write_fails`
//! in `local_disk.rs`), so no new label is needed there either.

use std::cmp::Reverse;

use super::{CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta};

pub mod in_memory;
pub mod local_disk;
pub mod storage_api;

pub use in_memory::InMemoryCheckpointStore;
pub use local_disk::LocalDiskCheckpointStore;
pub use storage_api::StorageApiCheckpointStore;
// Re-exported so `Config::checkpoint_reconcile_grace` can validate the
// reconcile interval against the same publish arithmetic the adapter reports at
// runtime, without constructing an adapter (b.9qd).
pub(crate) use storage_api::storage_api_publish_worst_case;

/// Put-time gate every adapter's `put_layer_checkpoint` runs before it touches
/// its backend: rejects a [`CheckpointKind::Tail`] meta that is
/// lineage-incomplete or that carries an ext-refs sidecar.
///
/// A [`CheckpointKind::Base`] meta returns `Ok(())` immediately and
/// unconditionally — including a base carrying stray `Some` lineage fields.
/// Task 2 documents `Base ⇒ base_snap_seq == None && from_seq == None`, but
/// policing it here would gate the long-shipped base path on a rule with no
/// failure mode behind it, so the base path is byte-for-byte unaffected.
///
/// # What a `Tail` is rejected for
///
/// - **Missing `base_snap_seq` and/or `from_seq`.** The trait requires a `Tail`
///   put to set both. Without them the artifact names no base, and replay would
///   have to *infer* a lineage — layering the tail's commits onto whatever base
///   happened to be newest, which silently produces a wrong restore rather than
///   a loud failure. The error message names which field(s) are absent.
/// - **`sidecar_bytes.is_some()`.** SR-2.4 / b.s5y make "tails carry no ext-refs
///   sidecar" MUST-level. Honouring one fails silently and permanently — and it
///   fails *differently* on each adapter, which is why the rejection is equally
///   justified on all three. The unreapability half is universal: the deletion
///   paths in LocalDisk and StorageApi are artifact-family-driven per generation
///   stem and a tail's family has no sidecar member, so a tail-authored
///   `.ext-refs.json` is never reaped anywhere. What it then does to the live
///   set diverges, and both branches are bad. On **LocalDisk**, whose
///   `list_referenced_ext_hashes` walk is suffix-driven (every
///   `*.ext-refs.json` under `layers/`) rather than stem-driven, it WOULD be
///   unioned in — pinning that snapshot's whole hash set into the live set
///   forever and degrading the retention cascade toward deleting nothing. On
///   **InMemory** (which unions base slots' sidecars only) and **StorageApi**
///   (whose `parse_ext_refs_filename` runs a strict two-component stem parse and
///   so rejects `<oid>.<to_seq>.tail.ext-refs.json`) it would NOT be seen at
///   all — leaving a permanently invisible artifact plus a silent cross-adapter
///   divergence in what the live set even means. Rejecting loudly beats
///   silently dropping the field, because a caller that passes one has a bug
///   that would otherwise never surface.
///
///   This rejection is an **adapter-enforced strengthening** of the trait's
///   MUST (the trait states the caller obligation; this makes it enforced at
///   every backend). Task 6's conformance cases pin it cross-adapter.
///
/// All three are [`CheckpointError::Permanent`], not `Conflict` and not
/// `Transient`: each is a caller bug, and retrying the identical call
/// reproduces it exactly.
///
/// # What this deliberately does NOT check
///
/// The **range** invariants — `base_snap_seq <= from_seq <= to_seq` — are owned
/// by [`crate::checkpoint::tail_codec`] via `TailCodecError::InvalidRange` /
/// `BaseAfterRange`, which validates them against the envelope's own header at
/// encode and decode time. Adapters treat the envelope as opaque bytes (SR-1.5)
/// and cannot cross-check the meta against it, so re-deriving the rule here
/// would put a second, weaker copy in four places — the classic way the two
/// copies drift. Presence-only is the whole contract at this layer.
///
/// Not gated on any cargo feature: this is production validation, not a test
/// seam.
pub(crate) fn validate_tail_put(meta: &CheckpointMeta) -> Result<(), CheckpointError> {
    if meta.kind == CheckpointKind::Base {
        return Ok(());
    }

    let mut missing: Vec<&str> = Vec::new();
    if meta.base_snap_seq.is_none() {
        missing.push("base_snap_seq");
    }
    if meta.from_seq.is_none() {
        missing.push("from_seq");
    }
    if !missing.is_empty() {
        return Err(CheckpointError::Permanent {
            source: format!(
                "tail checkpoint put rejected: CheckpointMeta.kind is Tail but lineage \
                 field(s) {} are None; a tail put MUST set both base_snap_seq and \
                 from_seq, and adapters MUST NOT infer lineage",
                missing.join(" and ")
            )
            .into(),
        });
    }

    if meta.sidecar_bytes.is_some() {
        return Err(CheckpointError::Permanent {
            source: "tail checkpoint put rejected: CheckpointMeta.sidecar_bytes is Some but \
                     tails carry no ext-refs sidecar; an ext-refs sidecar written \
                     under a tail stem is never reaped and would pin its hashes in the live \
                     set forever"
                .into(),
        });
    }

    Ok(())
}

/// Sibling gate to [`validate_tail_put`], run at the same place and for the
/// same reason: every persisted generation MUST carry a non-empty
/// `session_id` **and** `layer_name`.
///
/// # Why this is a store-side gate rather than a caller-side one
///
/// This is the **backstop** for the producer-side refusal in
/// [`crate::checkpoint::assembler`], which declines to assemble a generation
/// whose `obj:hdr` vanished (or carries empty identity columns) mid-assembly
/// instead of fabricating identity-empty meta from
/// `HeaderIdentity::default()`. Both halves are wanted. The assembler's refusal
/// is where the *decision* belongs — it has the object id, the header read, and
/// the checkpoint loop's benign-and-terminal `NotFound` arm to route into. This
/// one is where the *invariant* is enforced: it holds for every producer, on
/// every backend, including a future call site that has not yet been written.
///
/// The invariant is worth enforcing because an identity-empty generation is
/// **permanently unrestorable, silently**. The restore path's identity guard
/// (`checkpoint::apply`) refuses a blob with an empty `session_id` /
/// `layer_name` — it cannot rebuild `obj:hdr` without inventing session
/// metadata, and inventing it is how a restore lands a layer under the wrong
/// session. So the write costs a PUT, consumes a retention slot that a
/// restorable generation could have held, and yields an artifact whose only
/// possible future is to fail a restore. Refusing at the store turns that into
/// a loud, actionable failure at the moment the bug happens rather than during
/// a DR event.
///
/// # Both kinds, not just bases
///
/// A [`CheckpointKind::Tail`] is checked too. Tails are not exempt on the
/// grounds that "the base carries the identity": `tail_capture` sources the
/// four identity fields of a tail meta from the same `obj:hdr` read the
/// assembler uses, and the reconciliation / lineage surfaces read a tail's meta
/// directly. An identity-empty tail is as unusable as an identity-empty base,
/// so the predicate is deliberately kind-independent — unlike
/// [`validate_tail_put`], which returns early for a base.
///
/// # Classification
///
/// [`CheckpointError::Permanent`], not `Conflict` and not `Transient` — the
/// same rule [`validate_tail_put`] follows: this is a caller bug, and retrying
/// the identical call reproduces it exactly. `Conflict` would be actively
/// wrong, since it means "a competing writer raced and won" and would invite a
/// retry that can only fail again.
pub(crate) fn validate_generation_identity(meta: &CheckpointMeta) -> Result<(), CheckpointError> {
    let mut empty: Vec<&str> = Vec::new();
    if meta.session_id.is_empty() {
        empty.push("session_id");
    }
    if meta.layer_name.is_empty() {
        empty.push("layer_name");
    }
    if !empty.is_empty() {
        return Err(CheckpointError::Permanent {
            source: format!(
                "checkpoint put rejected: CheckpointMeta identity field(s) {} are empty; \
                 every persisted generation MUST carry a non-empty session_id and \
                 layer_name, because the restore path refuses a generation whose identity \
                 it cannot verify against the live keyspace — writing one produces a \
                 permanently unrestorable artifact",
                empty.join(" and ")
            )
            .into(),
        });
    }

    Ok(())
}

/// The canonical mixed-kind listing order: newest sequence first, and at an
/// equal sequence [`CheckpointKind::Base`] before [`CheckpointKind::Tail`].
///
/// Usable directly as a `sort_unstable_by_key` key. `Reverse(sequence)` gives
/// the descending-by-sequence order `list_layer_checkpoints` owes its callers;
/// the `u8` (`0` = Base, `1` = Tail) is a plain ascending secondary key, so the
/// tie-break is visible in the returned tuple rather than hidden in a
/// comparator.
///
/// `sequence` is the generation's kind-neutral ordering number: a base's
/// `snap_seq`, a tail envelope's `to_seq`. Read it off a handle via
/// [`CheckpointHandle::seq`] — or use [`handle_generation_sort_key`], which
/// does exactly that.
///
/// # Why a base and a tail collide at all
///
/// This is a live case, not a defensive hypothetical. SR-4.1 re-base composes
/// `base@S0 + tails` into a fresh **base at S**, where `S` is the newest
/// consumed tail's `to_seq`; SR-9.2 states re-base idempotency at
/// `(object_id, S)`. So `base@S` and `tail@to_seq == S` genuinely coexist until
/// retention removes one, and the order between them has to be pinned rather
/// than left to the backend's enumeration.
///
/// # Why Base wins the tie
///
/// `get_latest_layer_checkpoint`'s retained contract is `list().first()`. At an
/// equal sequence the base is the artifact that restores on its own, so putting
/// it first is the strictly safer answer for every kind-blind caller.
///
/// # Why not `#[derive(Ord)]` on `CheckpointKind`
///
/// A derived `Ord` (`Base < Tail`) reversed alongside the sequence — the shape
/// `sort_by(|a, b| b.cmp(a))` naturally produces — yields Tail-before-Base at a
/// tie, the exact opposite of the pinned decision, and a later reader would
/// have no way to tell whether that was deliberate. Spelling the total order
/// over `(sequence, kind)` out here makes the tie-break reviewable.
///
/// # Cross-adapter
///
/// All three adapters MUST order mixed-kind listings through this helper —
/// including Task `t2.dfh.gy.uv`'s StorageApi mixed-address listing, whose
/// backend LIST returns addresses in arbitrary order.
pub(crate) fn generation_sort_key(sequence: u64, kind: CheckpointKind) -> (Reverse<u64>, u8) {
    let kind_rank = match kind {
        CheckpointKind::Base => 0,
        CheckpointKind::Tail => 1,
    };
    (Reverse(sequence), kind_rank)
}

/// [`generation_sort_key`] for an already-minted handle — the form adapters
/// that sort `Vec<CheckpointHandle>` pass straight to `sort_unstable_by_key`.
///
/// Reads the ordering number through [`CheckpointHandle::seq`] rather than
/// `snap_seq` directly, which is the accessor's purpose: this is kind-neutral
/// code, where a tail's number is its `to_seq`.
pub(crate) fn handle_generation_sort_key(handle: &CheckpointHandle) -> (Reverse<u64>, u8) {
    generation_sort_key(handle.seq(), handle.kind)
}

/// A [`CheckpointMeta::default()`] whose identity columns are populated, for
/// in-crate test fixtures.
///
/// Every `mod tests` in this subsystem builds its metas with a
/// `..CheckpointMeta::default()` tail, and `CheckpointMeta::default()` leaves
/// `session_id` / `layer_name` empty — which
/// [`validate_generation_identity`] now refuses at every `put_layer_checkpoint`
/// (b.hka (b)). A fixture that wants a *stored* generation must therefore carry
/// identity, so the whole subsystem's fixtures spell that one way, here, rather
/// than each module inventing its own pair of strings.
///
/// The values are deliberately constant rather than per-generation: a fixture
/// that needs to tell two generations' metas apart already varies a column it
/// cares about (`creator_replica_id`, `snap_seq`, or identity itself — see
/// `in_memory::tests::meta_for_generation`), and a varying default would make
/// every unrelated assertion depend on which generation it happened to read.
///
/// Use `..identity_default()` in place of `..CheckpointMeta::default()`. A
/// fixture that is *about* empty identity (the `validate_generation_identity`
/// rejection cases) keeps `CheckpointMeta::default()` and says so.
#[cfg(test)]
pub(crate) fn identity_default() -> CheckpointMeta {
    CheckpointMeta {
        session_id: "fixture-session".to_string(),
        layer_name: "fixture-layer".to_string(),
        ..CheckpointMeta::default()
    }
}

/// Replace whatever is at `sidecar` (a LocalDisk ext-refs sidecar path) with a
/// dangling symlink, so a directory listing still names the sidecar and a read
/// of it answers `NotFound` — the listed-then-absent state (b.oya), made
/// deterministic.
#[cfg(all(test, unix))]
pub(crate) fn plant_dangling_sidecar(sidecar: &std::path::Path) {
    match std::fs::remove_file(sidecar) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("removing {} must succeed: {err}", sidecar.display()),
    }
    let mut target = sidecar.as_os_str().to_owned();
    target.push(".gone");
    std::os::unix::fs::symlink(&target, sidecar)
        .expect("planting the dangling sidecar symlink must succeed");
}

/// Test-only fault-injection seam shared by the InMemory and LocalDisk
/// checkpoint adapters.
///
/// **Production code paths MUST NOT depend on this trait.** It is gated behind
/// the `test-checkpoint-fault-injection` cargo feature so that release builds
/// and plain `cargo build` / `cargo test` never see the injection seam — the
/// trait, its impls, and any adapter constructor that accepts an
/// `Arc<dyn CheckpointFaultInjector>` are compiled out entirely.
///
/// The SR-1.3 conformance harness (lands in Task `t2.fgj.ya.74`) constructs an
/// injector, installs it on an adapter under test, and uses `op` labels
/// (`"put_layer_checkpoint"`, `"get_latest_layer_checkpoint"`,
/// `"get_layer_checkpoint"`, etc.) to drive each `CheckpointStore` method
/// through its `Transient` and `Conflict` branches.
///
/// `Send + Sync` so the injector can be cloned into background tasks the
/// harness spawns to exercise concurrent puts.
#[cfg(feature = "test-checkpoint-fault-injection")]
pub trait CheckpointFaultInjector: Send + Sync {
    /// Inspect the operation label and either return `Ok(())` (let the real
    /// adapter code path run) or return a classified
    /// [`CheckpointError`](super::CheckpointError) to short-circuit the call.
    ///
    /// `op` is a stable string label naming the calling adapter method
    /// (e.g. `"put_layer_checkpoint"`, `"get_layer_checkpoint"`,
    /// `"delete_ext_blob"`). Adapter
    /// implementations call `maybe_inject` at the top of each method body so
    /// the injector can short-circuit before any backend I/O runs.
    fn maybe_inject(&self, op: &str) -> Result<(), super::CheckpointError>;
}
