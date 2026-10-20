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

//! The **composition engine**: base checkpoint blob + ordered tail records →
//! one [`ObjectState`] at the lineage head (SR-3.2, SR-3.5, SR-6.1).
//!
//! One implementation serves both consumers SR-4.1 names: composed restore
//! ([`crate::checkpoint::recovery`]) and the chain-collapse re-base
//! ([`crate::checkpoint::rebase`]). Correctness of composed restore
//! reduces to "replay produces exactly what live history produced", so the
//! proof burden is deliberately concentrated in this one module.
//!
//! # Purity
//!
//! Everything here is pure and synchronous: **no Valkey, no
//! [`CheckpointStore`] and no other adapter I/O, no `async`, no clock, no
//! randomness**. Inputs are already-fetched artifact bytes (or already-decoded
//! envelopes); outputs are owned values. That is what makes the module
//! reviewable by inspection and unit-testable without a backing store, matching
//! the convention [`crate::checkpoint::tail_codec`] and
//! [`crate::checkpoint::tail_classify`] already encode.
//!
//! The one consequence worth stating up front: because the engine performs no
//! I/O, **no failure it reports is retriable**. See the `From<ComposeError> for
//! CheckpointError` impl below.
//!
//! # Why tails need a decode path of their own
//!
//! [`crate::checkpoint::apply::decode_checkpoint_ompf`] rejects anything that
//! is not `is_diff == true && base_version == 0`. That guard is *correct there*
//! — a base checkpoint blob is always a reset delta, and accepting an
//! incremental diff into the base slot would silently materialize a partial
//! layer. A tail record, by contrast, carries a **live `UpdateLayer` delta**:
//! normally `is_diff == false`, or `is_diff == true` with a non-zero
//! `base_version`. Feeding one to the base decoder would trip its step 3.
//!
//! So the tail path is a **sibling** decode function in this module, and
//! `decode_checkpoint_ompf` is deliberately left unmodified. Relaxing the base
//! decoder's shape guard to accommodate tails would trade a real safety
//! property for code reuse that amounts to four lines.
//!
//! [`CheckpointStore`]: crate::checkpoint::store::CheckpointStore
//! [`ObjectState`]: crate::domain::types::ObjectState

use std::collections::HashSet;

use bytes::Bytes;
use thiserror::Error;

use crate::checkpoint::apply::decode_checkpoint_ompf;
use crate::checkpoint::store::CheckpointError;
use crate::checkpoint::tail_codec::{
    decode_tail_envelope, peek_tail_header, TailCodecError, TailEnvelope, TailRecord,
};
use crate::domain::delta::{apply_delta_into, ApplyDeltaStats, DeltaError, MutationSink};
use crate::domain::ompf::{OmpfContainer, OmpfError};
use crate::domain::types::ObjectState;
use crate::error::ServiceError;
use crate::storage::external_values::collect_external_values;
use oodf_ffi::{OodfError, OwnedDelta};

/// One external-value part surfaced by composition: the 32-byte content hash,
/// and a zero-copy view of its bytes over the artifact buffer that carried it.
///
/// Structurally identical to the live path's `ExtValueRef`
/// (`src/storage/external_values.rs`); owned [`Bytes`] rather than a borrow
/// because the pairs outlive the per-record parse.
pub type ExtValuePart = ([u8; 32], Bytes);

/// What [`decode_tail_record_ompf`] yields: the delta to replay (`None` for the
/// live empty-content branch) and the ext parts that delta references.
type DecodedTailRecord = (Option<OwnedDelta>, Vec<ExtValuePart>);

/// Which generation of a lineage a failure belongs to.
///
/// `index` is the tail's position in the caller's ordered tail slice, and is
/// always available. `to_seq` is the generation's identifying sequence number
/// and is `None` only when the artifact's header could not be read at all —
/// see [`ComposeError::TailDecode`] for the one case that produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSite {
    /// Zero-based position in the caller-supplied tail slice.
    pub index: usize,
    /// The generation's `to_seq`, when known.
    pub to_seq: Option<u64>,
}

impl TailSite {
    /// A site whose header has been read, so `to_seq` is known.
    pub fn known(index: usize, to_seq: u64) -> Self {
        TailSite {
            index,
            to_seq: Some(to_seq),
        }
    }

    /// A site whose header could not be read.
    pub fn unknown(index: usize) -> Self {
        TailSite {
            index,
            to_seq: None,
        }
    }
}

impl std::fmt::Display for TailSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.to_seq {
            Some(to_seq) => write!(f, "tail[{}] to_seq={}", self.index, to_seq),
            None => write!(f, "tail[{}] to_seq=unknown", self.index),
        }
    }
}

/// Which record of which generation a replay failure belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordSite {
    /// The generation carrying the record.
    pub tail: TailSite,
    /// Zero-based position within that generation's records.
    pub index: usize,
    /// The commit sequence number the record carries.
    pub sequence_num: u64,
}

impl std::fmt::Display for RecordSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} record[{}] seq={}",
            self.tail, self.index, self.sequence_num
        )
    }
}

/// Every way composing a lineage can fail.
///
/// Each variant names **which generation** failed (a [`TailSite`], or the
/// base's `base_snap_seq`) and, for replay failures, **which record**
/// (a [`RecordSite`], carrying the record index and its `sequence_num`), so a
/// caller can log the failure without adding context of its own.
#[derive(Debug, Error)]
pub enum ComposeError {
    /// The base checkpoint blob did not decode into an `ObjectState`.
    #[error("base@{base_snap_seq}: base decode failed: {source}")]
    BaseDecode {
        /// The base generation's `snap_seq`.
        base_snap_seq: u64,
        /// The base decoder's rejection.
        #[source]
        source: CheckpointError,
    },

    /// A tail artifact could not be decoded into a [`TailEnvelope`].
    ///
    /// [`TailEnvelope`]: crate::checkpoint::tail_codec::TailEnvelope
    #[error("{site}: tail envelope decode failed: {source}")]
    TailDecode {
        /// The offending generation. `to_seq` is `None` when the header could
        /// not be peeked either.
        site: TailSite,
        /// The codec's structural rejection.
        #[source]
        source: TailCodecError,
    },

    /// A tail's header names a different layer generation than the lineage.
    #[error("{site}: object_id mismatch: lineage is object {expected}, tail names {got}")]
    ObjectIdMismatch {
        /// The offending generation.
        site: TailSite,
        /// The `object_id` the caller is composing.
        expected: u64,
        /// The `object_id` the tail header carries.
        got: u64,
    },

    /// A tail's header names a different base than the lineage's.
    #[error("{site}: base mismatch: lineage base_snap_seq {expected}, tail names {got}")]
    BaseMismatch {
        /// The offending generation.
        site: TailSite,
        /// The `base_snap_seq` the caller is composing onto.
        expected: u64,
        /// The `base_snap_seq` the tail header carries.
        got: u64,
    },

    /// A tail does not start where the composed cursor stands — commits
    /// between the two would be silently skipped.
    #[error("{site}: lineage gap: composed cursor at {cursor}, tail from_seq {from_seq}")]
    LineageGap {
        /// The offending generation.
        site: TailSite,
        /// The last sequence number composed so far.
        cursor: u64,
        /// The tail's exclusive lower bound.
        from_seq: u64,
    },

    /// A record's `sequence_num` is not the composed cursor's successor.
    #[error("{site}: record[{record_index}] sequence mismatch: expected {expected}, got {got}")]
    RecordSequenceMismatch {
        /// The offending generation.
        site: TailSite,
        /// Zero-based record index within that generation.
        record_index: usize,
        /// The sequence number required at this position (`cursor + 1`).
        expected: u64,
        /// The sequence number the record carries.
        got: u64,
    },

    /// A verbatim record's OMPF container did not parse.
    #[error("{site}: OMPF parse failed: {source}")]
    OmpfParse {
        /// The offending record.
        site: RecordSite,
        /// The container-level rejection.
        #[source]
        source: OmpfError,
    },

    /// A verbatim record's OODF payload was rejected by the FlatBuffers
    /// verifier (or otherwise failed to parse).
    #[error("{site}: OODF parse failed: {source}")]
    OodfParse {
        /// The offending record.
        site: RecordSite,
        /// The parse/verifier rejection.
        #[source]
        source: OodfError,
    },

    /// A verbatim record's external-value references failed the live-path
    /// collector: an index that does not resolve against the record's own OMPF
    /// parts, or an external hash that is not exactly 32 bytes. A record of the
    /// latter shape can only predate the ingest-time hash-length check; it was
    /// never reproducible by `apply_delta`, so it is as unrestorable as a
    /// dangling index.
    #[error("{site}: external-value collection failed: {source}")]
    ExtValueIndex {
        /// The offending record.
        site: RecordSite,
        /// The collector's rejection, verbatim.
        #[source]
        source: ServiceError,
    },

    /// `apply_delta` rejected a delta that had committed on the live path.
    #[error("{site}: apply_delta rejected a delta that committed live: {source}")]
    ApplyDelta {
        /// The offending record.
        site: RecordSite,
        /// The domain engine's rejection.
        #[source]
        source: DeltaError,
    },

    /// After applying a record, the composed state's `sequence_num` is not the
    /// record's — an internal-consistency trip, not a data defect.
    #[error("{site}: post-apply sequence drift: expected {expected}, state at {got}")]
    PostApplySequence {
        /// The offending record.
        site: RecordSite,
        /// The record's `sequence_num`.
        expected: u64,
        /// The state's `sequence_num` after `apply_delta`.
        got: u64,
    },

    /// Replaying a generation's records did not land the cursor on the
    /// generation's claimed `to_seq`.
    #[error("{site}: terminal cursor mismatch: replay ended at {cursor}")]
    TerminalCursor {
        /// The offending generation.
        site: TailSite,
        /// Where replay actually ended.
        cursor: u64,
    },
}

/// Every compose failure is [`CheckpointError::Permanent`].
///
/// `Transient` is **structurally impossible** here: the module performs no I/O,
/// so a retry re-reads the same in-memory bytes and reaches the same verdict.
/// There is no backend to be briefly unavailable and no connection to drop.
///
/// This is the classification contract the step-down loop consumes. A compose
/// failure means *"this generation's lineage is unrestorable — step down to the
/// next older restorable generation"*, exactly as a broken lineage link does.
/// It never means "retry me against the same bytes".
///
/// One exception, SR-14.7: a [`ComposeError::BaseDecode`] whose base is a
/// plain OMPF delta over the per-delta entry limit
/// (`apply::is_over_entry_limit_base`) is refused `Permanent` by the recovery
/// hook, with no step-down (or answered not-found, when the hook's identity
/// verdict finds the layer gone). The base is intact but too large to restore on
/// access, so SR-14.7 makes the outcome `Permanent` outright. Normally a
/// step-down would re-offer the same base as a base-only lineage, which fails
/// at apply's decode. In the jumped-base shape of SDD §22.20.3 it would reach
/// a different, older base instead. The refusal gives that older generation up
/// even when it is restorable, a cost the SR-14.7 ruling accepts.
impl From<ComposeError> for CheckpointError {
    fn from(err: ComposeError) -> Self {
        CheckpointError::Permanent {
            source: Box::new(err),
        }
    }
}

/// Decode one [`TailRecord::Verbatim`] payload into the delta to replay and the
/// external-value parts that delta references.
///
/// This mirrors the live `handle_update_layer` step-5 pipeline
/// (`src/grpc/layer.rs`) branch for branch, except for the content check named
/// below — that fidelity is what SR-3.2's "byte-for-byte what the live commit
/// did" requires:
///
/// 1. [`OmpfContainer::parse`] over the record's raw bytes.
/// 2. **Empty `delta_bytes` ⇒ `None`, with no `parse_delta` call.** The live
///    path takes exactly that branch; calling `oodf_ffi::parse_delta` on the
///    empty slice would return [`OodfError::EmptyInput`] and mark a *healthy*
///    lineage unrestorable. A naive `parse_delta(record.ompf())` is the obvious
///    wrong implementation here.
/// 3. Otherwise `oodf_ffi::parse_delta(container.delta_bytes)` — the hardened
///    C++ FlatBuffers verifier runs inside it, exactly as on the live path.
/// 4. External-value collection via the live path's own
///    [`collect_external_values`], **before** the caller mutates any state, so
///    a bad `ext_value_index` or a wrong-length external hash fails prior to
///    touching the composed state — the same ordering the live path gets from
///    running `store_external_values` ahead of `apply_delta`.
///
/// An empty-`delta_bytes` record therefore yields **zero** ext pairs, even if
/// the container carries Parts 1..N. That is not an oversight: live, the
/// `parsed.as_ref() == None` argument short-circuits `store_external_values` to
/// `Ok(())` before `external_values` is read, so such a record stored nothing.
/// Collecting unconditionally would over-store relative to live history.
///
/// **The one deliberate divergence: no content check.** Live,
/// `store_external_values` runs the content check (`verify_external_values`)
/// between collection and store, refusing an upload whose parts do not match
/// their declared sizes and content hashes. This decoder does not. The check
/// never changes what a commit applies, only whether the commit happens (and
/// under `LIVE_EDIT_EXT_VALUE_VERIFY=observe`, not even that), and a tail
/// record exists only for a commit that landed — so replaying a record without
/// the check reproduces exactly what the live commit did, whether the record
/// passed the check, was admitted under `observe`, or was written before the
/// check existed (the at-rest residual in SDD §11.1). Replay stays
/// byte-for-byte faithful without re-hashing a whole lineage on every restore
/// and re-base.
///
/// No `is_diff` / `base_version` shape check is applied — see the module doc
/// for why the base decoder's guard is right there and wrong here.
///
/// # Zero-copy
///
/// Returned ext payloads are [`Bytes::slice_ref`] views over `record_ompf`
/// itself, so a multi-MB tail is never copied. `slice_ref` panics if the
/// subslice is not derived from that exact buffer, so the parse input is
/// derived from the same `Bytes` (`OmpfContainer::parse(record_ompf)` borrows
/// through `Deref`) — do not re-parse a copy and slice against the original.
/// `Bytes::copy_from_slice` would be correct but copies up to
/// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (64 MiB default) per tail.
///
/// # Errors
///
/// - [`ComposeError::OmpfParse`] — the container is malformed.
/// - [`ComposeError::OodfParse`] — the OODF payload failed to parse or the
///   verifier rejected it.
/// - [`ComposeError::ExtValueIndex`] — an `ext_value_index` is zero or out of
///   range for the record's own OMPF parts, or an external hash is not exactly
///   32 bytes.
///
/// All three convert to [`CheckpointError::Permanent`].
///
/// [`TailRecord::Verbatim`]: crate::checkpoint::tail_codec::TailRecord::Verbatim
pub(crate) fn decode_tail_record_ompf(
    site: RecordSite,
    record_ompf: &Bytes,
) -> Result<DecodedTailRecord, ComposeError> {
    // Step 1 — OMPF container parse. Borrows through `Deref<Target = [u8]>`,
    // which is what keeps the `slice_ref` invariant below satisfiable.
    let container = OmpfContainer::parse(record_ompf)
        .map_err(|source| ComposeError::OmpfParse { site, source })?;

    // Step 2 — the live path's empty-content branch. Load-bearing; see the
    // doc comment. Zero ext pairs, deliberately.
    if container.delta_bytes.is_empty() {
        return Ok((None, Vec::new()));
    }

    // Step 3 — OODF parse, verifier included.
    let delta = oodf_ffi::parse_delta(container.delta_bytes)
        .map_err(|source| ComposeError::OodfParse { site, source })?;

    // Step 4 — ext collection before any state mutation, using the live
    // path's own collector so hash-length checking, index resolution and
    // dedup cannot drift.
    let pairs = collect_external_values(&delta, &container.external_values)
        .map_err(|source| ComposeError::ExtValueIndex { site, source })?;

    // Zero-copy: every `data` here is a subslice of `record_ompf`'s buffer
    // (the parse above borrowed from it), which is `slice_ref`'s precondition.
    let ext_values = pairs
        .into_iter()
        .map(|(hash, data)| (hash, record_ompf.slice_ref(data)))
        .collect();

    Ok((Some(delta), ext_values))
}

/// The result of composing one lineage: the materialized state, the sequence
/// it stands at, and the external-value parts the replayed commits referenced.
#[derive(Debug)]
pub struct ComposedLineage {
    /// The composed layer state at [`ComposedLineage::composed_seq`].
    ///
    /// Returned **by value**, never written through a caller-supplied
    /// reference — see [`compose_lineage`]'s SR-3.5 note.
    pub state: ObjectState,

    /// The sequence number the composition landed on.
    ///
    /// Invariant: `composed_seq == state.sequence_num == the last envelope's
    /// to_seq` (and `== base_snap_seq` for a zero-tail lineage). This is the
    /// value the restore caller passes to
    /// [`crate::checkpoint::apply::apply_checkpoint`] as `snap_seq`
    /// (`src/checkpoint/apply.rs:482`) — the composed head is what the restored
    /// object must be stamped at, not the base generation's `snap_seq`.
    pub composed_seq: u64,

    /// Parts 1..N of every replayed record's OMPF — `([u8; 32], Bytes)` pairs
    /// (see [`ExtValuePart`]) — deduplicated by hash across the whole lineage,
    /// in first-appearance order (records in replay order; within a record,
    /// `set_fields` then `time_samples`). Zero-copy views over the tail
    /// artifact buffers.
    ///
    /// SR-3.4's hand-off: the restore caller pre-stores these via
    /// `ExternalValueStore::store_values` **strictly before** its single
    /// `write_restored_object` MULTI/EXEC, exactly as the live path stores ext values
    /// before its commit transaction. Surfacing them here is what keeps the
    /// caller from re-parsing multi-MB tails.
    ///
    /// # Dedup semantics — one increment per lineage, and that is the right unit
    ///
    /// `ref_count` counts **live objects** referencing the hash — one increment
    /// per object per distinct hash, not one per referencing commit or value row
    /// (b.qpf; `docs-internal/sdd.md` §11.3). Live, N separate commits on one
    /// object that all reference the same hash therefore leave `ref_count` at
    /// **1**, not N: each commit's pre-flight `+1` is recognised as surplus and
    /// handed straight back, because `le:{oid}:obj:ext` records that the object
    /// already held the hash.
    ///
    /// So the lineage-wide dedup here is not a divergence from live semantics —
    /// it produces the same unit. A hash referenced by N replayed commits appears
    /// once and restores at `ref_count = 1`, which is exactly one live object
    /// referencing it.
    ///
    /// **The early-collect hazard this comment used to warn about does not
    /// exist.** It was reasoned from the pre-b.qpf per-commit counting unit: if
    /// live had reached N and a restore only reached 1, then an overwrite of one
    /// referencing field could decrement to zero while other restored fields
    /// still pointed at the blob. Under per-object counting the premise is gone,
    /// and the `obj:ext` index makes it structurally unreachable besides — a
    /// commit emits a `rewrite` release only when the object's LAST row holding
    /// the hash gives it up, so a field overwritten while sibling fields still
    /// hold the hash releases nothing.
    ///
    /// What does remain is the leak direction, unchanged and accepted: a hash
    /// this lineage pre-stores that the FINAL composed state does not reference
    /// keeps its `+1` (the restore path suppresses the surplus rule — see
    /// `apply_restore`), and the reference-count reconciler drains it (b.yda).
    /// The engine's contract is unchanged: the list is complete, deduplicated,
    /// and ordered.
    ///
    /// # Memory
    ///
    /// These `Bytes` keep the tail artifact buffers alive by refcount, bounded
    /// by `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` × `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`
    /// (64 MiB × 8 at defaults). The caller should drop the artifacts once the
    /// ext pre-store completes.
    pub ext_values: Vec<ExtValuePart>,
}

/// Compose a base checkpoint blob plus an ordered run of already-decoded tail
/// envelopes into one [`ObjectState`] at the lineage head (SR-3.2).
///
/// Inputs are **already-fetched artifacts only** — no [`CheckpointStore`], no
/// handle types, no gather types. That keeps the engine free of restore-hook
/// coupling, so the chain-collapse re-base
/// ([`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage))
/// calls it with the same arguments the
/// restore path uses. Callers holding raw artifact bytes want
/// [`compose_from_artifacts`] instead.
///
/// # What it does
///
/// 1. **Base decode** — [`decode_checkpoint_ompf`], whose reset-shape guard is
///    correct for a base and is not modified (module doc).
/// 2. **Cursor seeding** — `state.sequence_num = base_snap_seq`. `apply_delta`
///    computes `new_version = state.sequence_num + 1` and commits it only after
///    every helper succeeded, so the first replayed record stamps
///    `base_snap_seq + 1` on every entry it touches — which is exactly
///    `TailHeader::from_seq`'s exclusive-lower-bound semantics (first carried
///    commit is `from_seq + 1`).
/// 3. **Per-envelope link validation** — `object_id`, `base_snap_seq`, and
///    `from_seq == cursor`, before any of that envelope's records replay.
/// 4. **Record replay** — verbatim records through
///    [`decode_tail_record_ompf`] then `apply_delta`; `EmptyDelta` records
///    through `apply_delta(state, None)`, byte-for-byte what the live §8.4
///    empty commit did.
/// 5. **Terminal check** — the cursor must land on each envelope's `to_seq`.
///
/// # The `from_seq == cursor` check is load-bearing
///
/// It cannot be delegated to the codec. The codec only rejects
/// `base_snap_seq > from_seq` (`TailCodecError::BaseAfterRange`), so an
/// envelope with `base_snap_seq = 10, from_seq = 15` decodes cleanly and would
/// silently skip commits 11..15 if composed onto base@10. A gather caller
/// checks the same links, but the engine is also called directly by the
/// chain-collapse re-base, so it defends itself.
///
/// # Versions and set-order — what round-trips and what does not
///
/// Tail-touched entries carry per-entry versions `base_snap_seq + 1 ..= S` and
/// their `authoring_set_order` verbatim, matching live history. Base-era
/// entries carry the base decoder's re-derived stamps, exactly as today's
/// base-only restore leaves them — the base decode is not re-stamped, only the
/// cursor is seeded. Likewise `state.next_set_order` is whatever `apply_reset`
/// re-derived from the base's serialized authoring ranks: the cumulative
/// write cursor is deliberately not carried in a reset OMPF (see
/// `build_reset_ompf_from_state`'s "Which rank the field `set_order` slot
/// carries", `src/checkpoint/assembler.rs`), so tail-replayed fields compute
/// their `stored_set_order` off that re-derived cursor. `authoring_set_order`
/// round-trips exactly; the cumulative rank does not — again exactly as on a
/// base-only restore at HEAD.
///
/// # Replay fidelity rests on the Lua/Txn parity fence
///
/// SR-3.2's "the same `apply_delta` engine the live commit paths use" is true
/// only in part for the **default** commit backend,
/// `LIVE_EDIT_UPDATE_LAYER_BACKEND=lua`. Its structural and reset commits are
/// planned by `apply_delta` itself, on the per-replica structural cache's
/// copy of the object, and land through the commit script's structural mode,
/// which writes the plan's `StorageMutation`s 1:1. Its field and time-sample
/// commits go through the script's fast mode, which never calls
/// `apply_delta`. What makes replay-through-`apply_delta` reproduce a
/// Lua-committed history is therefore two parity fences: the byte-parity suite
/// `run_parity` in `tests/integration/storage/update_layer_lua_test.rs`
/// (fast mode against the WATCH path's `apply_delta` commit), and the
/// structural-mode keyspace equivalence with the WATCH path
/// (`tests/integration/storage/structural_watch_equivalence_test.rs` and its
/// proptest), each asserting identical keyspaces. Those suites are this
/// engine's load-bearing dependency: if one is weakened, composed restore's
/// fidelity claim weakens with it.
///
/// # No partial state, structurally (SR-3.5)
///
/// The engine owns a local [`ObjectState`] and returns it by value. It takes no
/// `&mut ObjectState` out-parameter and never writes through a caller-supplied
/// reference. `apply_delta`'s partial-mutation contract (`src/domain/delta.rs`:
/// on `Err(_)` every field except `sequence_num` may be partially mutated, and
/// the caller MUST discard) is therefore satisfied by construction — on any
/// error the local state is dropped and nothing partial is observable to
/// anyone.
///
/// A zero-tail lineage is accepted and returns the decoded base state at
/// `base_snap_seq`, which makes the function total and gives the chain-collapse
/// re-base a zero-tail case for free.
///
/// # Errors
///
/// Every variant below converts to [`CheckpointError::Permanent`]; `Transient`
/// is structurally impossible (see the `From` impl). A failure means the
/// lineage is unrestorable and the caller should step down to the next older
/// restorable generation, never retry against the same bytes. The SR-14.7
/// exception is a [`ComposeError::BaseDecode`] whose base is a plain OMPF delta
/// over the per-delta entry limit (`apply::is_over_entry_limit_base`). The
/// recovery hook refuses that `Permanent`, with no step-down (or answers
/// not-found, when its identity verdict finds the layer gone).
///
/// - [`ComposeError::BaseDecode`] — the base blob failed to decode (bad OMPF,
///   verifier rejection, non-reset shape, or `apply_delta` rejection).
/// - [`ComposeError::ObjectIdMismatch`] / [`ComposeError::BaseMismatch`] /
///   [`ComposeError::LineageGap`] — an envelope does not belong to, or does not
///   link onto, this lineage.
/// - [`ComposeError::RecordSequenceMismatch`] — a record's `sequence_num` is
///   not the composed cursor's successor.
/// - [`ComposeError::OmpfParse`] / [`ComposeError::OodfParse`] /
///   [`ComposeError::ExtValueIndex`] — a verbatim record's payload failed the
///   live-path parse pipeline.
/// - [`ComposeError::ApplyDelta`] — `apply_delta` rejected a delta that had
///   committed live.
/// - [`ComposeError::PostApplySequence`] / [`ComposeError::TerminalCursor`] —
///   internal-consistency trips on the replay cursor.
///
/// [`CheckpointStore`]: crate::checkpoint::store::CheckpointStore
pub fn compose_lineage(
    object_id: u64,
    base_ompf: &[u8],
    base_snap_seq: u64,
    tails: &[TailEnvelope],
) -> Result<ComposedLineage, ComposeError> {
    // Step 1 — base decode. `decode_checkpoint_ompf` zeroes `sequence_num`
    // and `first_version`; `first_version` stays 0 here, matching today's
    // base-only restore (only a replayed reset delta would legitimately move
    // it, via `apply_reset`).
    let mut state = decode_checkpoint_ompf(object_id, base_ompf).map_err(|source| {
        ComposeError::BaseDecode {
            base_snap_seq,
            source,
        }
    })?;

    // Step 2 — cursor seeding. See the "Versions and set-order" doc section.
    state.sequence_num = base_snap_seq;
    let mut cursor = base_snap_seq;

    let mut ext_values: Vec<ExtValuePart> = Vec::new();
    let mut seen_hashes: HashSet<[u8; 32]> = HashSet::new();
    // b.ot3: replay reproduces `state`; the persistence mutations each
    // record would plan are never read. One sink and one stats out-parameter
    // for the whole lineage — neither is consulted after the loop.
    let mut sink = MutationSink::Discard;
    let mut stats = ApplyDeltaStats::default();

    for (tail_index, envelope) in tails.iter().enumerate() {
        let header = envelope.header();
        let site = TailSite::known(tail_index, header.to_seq);

        // Step 3 — per-envelope link validation, before any record replays.
        if header.object_id != object_id {
            return Err(ComposeError::ObjectIdMismatch {
                site,
                expected: object_id,
                got: header.object_id,
            });
        }
        if header.base_snap_seq != base_snap_seq {
            return Err(ComposeError::BaseMismatch {
                site,
                expected: base_snap_seq,
                got: header.base_snap_seq,
            });
        }
        if header.from_seq != cursor {
            return Err(ComposeError::LineageGap {
                site,
                cursor,
                from_seq: header.from_seq,
            });
        }

        // Step 4 — record replay.
        for (record_index, record) in envelope.records().iter().enumerate() {
            // `saturating_add` rather than `+`: `base_snap_seq` is
            // caller-supplied, and a `u64::MAX` cursor would panic on overflow
            // in a debug build. Saturating turns that into an ordinary
            // mismatch (or, past 2^53 - 1, an `apply_delta` ceiling rejection).
            let expected = cursor.saturating_add(1);
            let sequence_num = record.sequence_num();
            if sequence_num != expected {
                return Err(ComposeError::RecordSequenceMismatch {
                    site,
                    record_index,
                    expected,
                    got: sequence_num,
                });
            }

            let record_site = RecordSite {
                tail: site,
                index: record_index,
                sequence_num,
            };

            let owned = match record {
                TailRecord::Verbatim { ompf, .. } => {
                    let (delta, pairs) = decode_tail_record_ompf(record_site, ompf)?;
                    // Lineage-wide dedup, first-appearance order preserved —
                    // see `ComposedLineage::ext_values`.
                    for (hash, data) in pairs {
                        if seen_hashes.insert(hash) {
                            ext_values.push((hash, data));
                        }
                    }
                    delta
                }
                // Byte-for-byte what the live §8.4 empty commit did.
                TailRecord::EmptyDelta { .. } => None,
            };

            // b.ot3: replay exists to reproduce an `ObjectState`, not to
            // re-commit history — the mutation list was always dropped, so
            // build none of it.
            apply_delta_into(&mut state, owned, &mut sink, &mut stats).map_err(|source| {
                ComposeError::ApplyDelta {
                    site: record_site,
                    source,
                }
            })?;

            if state.sequence_num != sequence_num {
                return Err(ComposeError::PostApplySequence {
                    site: record_site,
                    expected: sequence_num,
                    got: state.sequence_num,
                });
            }
            cursor = state.sequence_num;
        }

        if cursor != header.to_seq {
            return Err(ComposeError::TerminalCursor { site, cursor });
        }
    }

    // Step 5 — terminal cursor. Equals `base_snap_seq` for a zero-tail
    // lineage, and `state.sequence_num` always.
    Ok(ComposedLineage {
        state,
        composed_seq: cursor,
        ext_values,
    })
}

/// Compose a lineage from **raw artifact bytes** — the surface a restore caller
/// or the chain-collapse re-base uses when it holds undecoded tail artifacts.
///
/// Each artifact is decoded with [`decode_tail_envelope`], which is eager and
/// total: an envelope in hand is already fully validated, which is what makes
/// SR-3.5's "no partial application in any failure mode" achievable. Decoding
/// happens for **every** artifact before any record replays, so a malformed
/// artifact at the end of the chain cannot abort a replay mid-way. Composition
/// then delegates to [`compose_lineage`], which owns all the semantics.
///
/// # Which entry point to use
///
/// This one, in almost every case. Lineage gather deliberately does **not**
/// decode envelopes — it peeks headers and carries the artifact bytes forward —
/// so both the restore path and the chain-collapse re-base arrive here holding
/// raw `Bytes`.
/// [`compose_lineage`] exists for callers that already hold decoded envelopes:
/// unit tests that build envelopes in memory, and any future caller that
/// decoded for its own reasons and should not pay for a second decode.
///
/// # Per-generation attribution
///
/// A decode failure is attributed to a generation on a best-effort ladder:
/// [`peek_tail_header`] is tried first to recover the artifact's `to_seq`, and
/// when that also fails the artifact's **index in `tail_artifacts`** is the
/// attribution ([`TailSite::to_seq`] is then `None`).
///
/// [`TailCodecError::UnknownFormatVersion`] is **inherently index-attributed
/// only**: the codec validates `format_version` immediately after the magic and
/// *before* reading `to_seq` (a deliberate ordering — a v2 artifact's header
/// may not even have the same layout, so reading `to_seq` out of it would be a
/// guess). The caller that assembled `tail_artifacts` is responsible for
/// mapping the index back to its `CheckpointHandle` when it logs the step-down;
/// this engine never guesses at a v2 payload's meaning (SR-6.1).
///
/// # Step-down contract — the boundary the caller consumes
///
/// - Every [`ComposeError`] converts to [`CheckpointError::Permanent`].
///   `Transient` is **structurally impossible**: the engine performs no I/O, so
///   a retry re-reads the same bytes and reaches the same verdict.
/// - A compose failure therefore means *"this generation's lineage is
///   unrestorable — step down to the next older restorable generation"*,
///   identically to a broken lineage link. It must never be retried against the
///   same bytes.
/// - **Exception (SR-14.7):** a [`ComposeError::BaseDecode`] whose base is a
///   plain OMPF delta over the per-delta entry limit
///   (`apply::is_over_entry_limit_base`) is not a step-down. The recovery hook
///   refuses it `Permanent`, with no step-down (or answers not-found, when its
///   identity verdict finds the layer gone). The base is intact but too
///   large to restore on access, so SR-14.7 makes the outcome `Permanent`
///   outright. Normally a step-down would re-offer the same base as a
///   base-only lineage that fails at apply's decode. In the jumped-base shape
///   of SDD §22.20.3 it would reach a different, older base instead. The
///   refusal gives that older generation up even when it is restorable, a
///   cost the SR-14.7 ruling accepts. A new caller that steps down must check
///   for this case first.
/// - The failure classes covered: unknown `format_version` (SR-6.1), any codec
///   structural rejection (bad magic, truncation, trailing bytes, non-contiguous
///   records, …), OMPF parse error, OODF verifier rejection, ext-index
///   resolution failure, `apply_delta` rejection on bytes that committed live,
///   and lineage-link / cursor mismatches (SR-3.5).
/// - **No partial state is observable in any of them**: the composed state is a
///   local returned by value, so an error drops it (see [`compose_lineage`]).
///
/// # Errors
///
/// - [`ComposeError::TailDecode`] — an artifact failed structural decode,
///   attributed per the ladder above.
/// - Every error [`compose_lineage`] can return, unchanged.
pub fn compose_from_artifacts(
    object_id: u64,
    base_ompf: &[u8],
    base_snap_seq: u64,
    tail_artifacts: &[Bytes],
) -> Result<ComposedLineage, ComposeError> {
    let mut envelopes = Vec::with_capacity(tail_artifacts.len());
    for (index, artifact) in tail_artifacts.iter().enumerate() {
        let envelope = decode_tail_envelope(artifact).map_err(|source| {
            // Best-effort attribution: recover `to_seq` when the header is
            // readable, fall back to the artifact's index when it is not.
            let site = match peek_tail_header(artifact) {
                Ok((_, header)) => TailSite::known(index, header.to_seq),
                Err(_) => TailSite::unknown(index),
            };
            ComposeError::TailDecode { site, source }
        })?;
        envelopes.push(envelope);
    }

    compose_lineage(object_id, base_ompf, base_snap_seq, &envelopes)
}

#[cfg(test)]
pub(crate) mod tests {
    //! Unit lanes A, B and C for the composition engine. **Lane A** (first)
    //! owns the shared fixture vocabulary plus the happy-path equivalence,
    //! version-stamping, empty-delta, zero-tail, mid-tail-reset,
    //! chain-flattening and ext-parts-order groups. **Lane B** (middle section,
    //! `#B1`–`#B11`) owns the SR-3.5 / SR-6.1 failure arms, the `Permanent`
    //! classification contract, the sequence-ceiling boundary, and the two
    //! structural source-scan guards. **Lane C** (last section, `#C1`–`#C3`)
    //! owns the tail half of the decoder boundary — that the base decoder's
    //! reset-shape guard is not inherited by the record-replay path — plus
    //! [`decode_tail_record_ompf`]'s own boundary arms (`#C4`–`#C6`). Both
    //! later lanes reuse lane A's fixtures verbatim.
    //!
    //! `pub(crate)` on the module and on every fixture below so the sibling
    //! lanes (failure arms; decoder boundary + collector) drive the SAME
    //! [`timeline`] rather than a second one that could drift from this one's
    //! view of the engine — the precedent being
    //! [`crate::checkpoint::tail_classify::tests`], which
    //! `tail_capture.rs` imports from. Nothing outside `#[cfg(test)]` can
    //! reach it.
    //!
    //! Everything here is pure: no Valkey, no adapter, no `async`, no runtime,
    //! no filesystem, no clock. That is the module header's determinism
    //! contract, and it is what makes this lane the engine's proof burden.
    //!
    //! # The fixture invariant
    //!
    //! [`run`] drives ONE [`OwnedDelta`] value down both sides of every
    //! comparison: [`commit_with_ext`] serializes it into the record's OMPF
    //! **and** advances the reference [`ObjectState`] through
    //! [`crate::domain::delta::apply_delta`]. A record and the reference it is
    //! checked against therefore cannot drift, and every payload / name /
    //! timestamp varies with the sequence number (`payload_for(seq)`,
    //! `server_ts_for(seq)`, the `ompf_for(seq)` discipline
    //! `tail_capture.rs` uses) so a record-to-reference mix-up cannot pass by
    //! comparing interchangeable data.
    //!
    //! # What a reset-OMPF round trip carries, and what it re-derives
    //!
    //! [`assert_faithful_columns`] enumerates the columns that survive
    //! `state → build_reset_ompf_from_state → decode_checkpoint_ompf` intact:
    //! `object_id`, `source_format`, each section's
    //! `parent_id` / `name` / `section_type` / `section_order`, the `children`
    //! lists **in order**, the field key set with its `FieldValue` **hash and
    //! size** and its `authoring_set_order`, and the time-sample key set with
    //! its `FieldValue`.
    //!
    //! Seven columns are deliberately NOT in that list. Each is re-derived by
    //! the base decode rather than carried, each has its own pinning test, and
    //! none of it is new loss introduced by composition — it is exactly what a
    //! base-only restore leaves behind at HEAD today:
    //!
    //! | Re-derived column | Why | Pinned by |
    //! |---|---|---|
    //! | `Field::version`, `TimeSample::version` (base era) | `decode_checkpoint_ompf` applies the reset delta to a fresh state, so `apply_delta` stamps `new_version = 1` on everything the base carried | #4 |
    //! | `Section::first_version` | same; and `apply_diff_change_section`'s new-section branch stamps `first_version = new_version` | #4 (b.2ti) |
    //! | `Section::last_version` (base era) | same | #4 (b.2ti) |
    //! | `ObjectState::first_version` | `apply_reset` sets it to 1, then the decoder zeroes it and composition never re-stamps it — only a *replayed* reset moves it | #4, #10 (b.2ti) |
    //! | `Field::set_order` (cumulative) | `build_reset_ompf_from_state` serializes `authoring_set_order` into OODF's single `set_order` slot; `apply_set_field` re-accumulates the cursor from scratch | #5 |
    //! | `ObjectState::next_set_order` | `apply_reset` zeroes the cursor; the base carries only the *surviving* fields, so overwritten / deleted writes the live cursor folded are missing | #5 |
    //! | `FieldValue::External::ext_value_index` | `build_reset_ompf_from_state` → `collect_external_value_hashes` assigns dense 1-based indices in first-appearance order over the WHOLE state, which has nothing to do with any single live delta's per-record index | #4b |
    //!
    //! An exclusion that is silent is an exclusion that will grow, so
    //! [`assert_faithful_columns`] names all seven at its call site and the
    //! table above is the index into their tests.
    //!
    //! **Two of those rows describe a value that does not survive to the
    //! persisted object (b.2ti).** After composition and before
    //! `write_restored_object`,
    //! `apply_restore` flattens the versions a diff builder FILTERS on —
    //! `Field::version` / `TimeSample::version` (row 1) and
    //! `Section::last_version` (row 3) — onto `seeded_seq`, because a restore
    //! is one write of the whole object and `src/domain/diff.rs`'s
    //! `version > have_version` filter would otherwise withhold every restored
    //! entry from every reconnecting subscriber.
    //!
    //! The other five rows are re-derived all the way to Valkey. In particular
    //! `Section::first_version` (row 2) is deliberately EXEMPT from that
    //! flattening: `plan_faithful_coalesced`'s abandoned-prim retraction arm
    //! fires on `first_version <= have_version`, so raising it would strand the
    //! orphan sections a restored base carries permanently unretractable. It
    //! reaches Valkey as the value this engine replayed (or, on a base-only
    //! restore, the decoder's `1` — so base-era sections read as pre-existing
    //! by construction, and new-vs-modified classification is truthful only
    //! for tail-created sections on the composed arm; harmless, since
    //! `build_delta` emits the same `DiffChangeSection` for both classes, and
    //! the retraction arm is the load-bearing reason for the exemption).
    //! `ObjectState::first_version` (row 4, which
    //! stays 0) and the three non-version rows are likewise untouched by the
    //! restore.
    //!
    //! So every row above stays accurate about what the *composition engine*
    //! hands back — which is what this lane asserts, in-memory, against live.
    //! Rows 1 and 3 additionally stop being accurate about the object Valkey
    //! ends up holding; the rest describe both.
    //!
    //! # Mutation table
    //!
    //! Deliberate mutations applied to the engine while authoring this lane,
    //! and the test that turned red. A lane that cannot name what it would
    //! catch is a lane nobody can audit later. Reproduce by re-applying the
    //! mutation and running `cargo nextest run -p live-edit --lib
    //! checkpoint::compose::`.
    //!
    //! | # | Mutation | Red | Survivors |
    //! |---|---|---|---|
    //! | M1 | step 2 seeds the cursor at `0` instead of `base_snap_seq` | 13 of 14 | the fixture self-check |
    //! | M2 | `TailRecord::EmptyDelta` records `continue` instead of running `apply_delta(state, None)` | 11 of 14 | self-check, #9 (no records), #10 (its timeline has no empty commit) |
    //! | M3a | a generation's records replayed newest-first | 12 of 14 | self-check, #9 — every red is the engine's own `RecordSequenceMismatch`, i.e. a *guard*, not content |
    //! | M3b | M3a **plus** `RecordSequenceMismatch` / `PostApplySequence` / `TerminalCursor` disabled, so the reorder reaches content | 10 of 14 | self-check, #8 (all records identical), #9, **and #1** — the cursor still lands on `S`, which is exactly why #3's version map, not #1, is the reorder detector |
    //! | M4 | each record's delta applied a second time | 12 of 14 | self-check, #11 (both sides double-apply identically) |
    //! | M5 | `ext_values` pushed without the `seen_hashes` dedup | 1 of 14 | everything except #12 — the sharpest row in the table |
    //!
    //! Lane B (failure arms + structural guards) appends its rows below. Its
    //! counts are denominated in the **25** tests this module holds once both
    //! lanes have landed; M1–M5 above were measured against lane A's 14 and are
    //! left at those numbers rather than re-measured, so each row still reads as
    //! the evidence its author actually collected.
    //!
    //! | # | Mutation | Red | Survivors |
    //! |---|---|---|---|
    //! | M6 | the replay error is swallowed and the record loop `continue`s with the cursor advanced (both `decode_tail_record_ompf`'s `?` and `apply_delta`'s) | 5 of 25 | #B4–#B7 — the ceiling pair, the decode-level arms and the base arm never reach record replay |
    //! | M7 | `From<ComposeError> for CheckpointError` yields `Transient` instead of `Permanent` | 7 of 25 | #B8 (`Display`-only), #B4, both source scans, all of lane A — this row is the *only* evidence the classification contract is asserted at all |
    //! | M8 | generation identity dropped: `TailSite::known(tail_index, header.to_seq)` in the envelope loop replaced by `TailSite::unknown(0)` | 5 of 25 | #B5/#B6, whose sites come from `compose_from_artifacts`'s own attribution ladder rather than the envelope loop, and #B7, which fails before it |
    //! | M9 | the sequence-ceiling comparison flipped `>` → `>=` in `tail_codec`'s `read_tail_header` **and** `encode_tail_envelope` | 2 of 25 | everything but the ceiling pair — the legal maximum stops composing, which is exactly the off-by-one the boundary companion exists to refuse |
    //!
    //! M8 is the row that changed this lane: measured first, it turned only 2
    //! tests red, because #B2 / #B3 / #B9's lineage-gap arm pinned the record's
    //! `sequence_num` but not the generation carrying it. Each now pins the
    //! whole [`TailSite`], and the row moved to 5. An arm that says *which
    //! record* without saying *which generation* does not carry enough for
    //! SR-3.1's step-down to name what it is stepping down from.
    //!
    //! M9 is measured on `tail_codec.rs`, not this module: the engine has no
    //! ceiling comparison of its own (see #B5's note on why an over-ceiling
    //! cursor is unreachable from here), so the boundary this lane pins is
    //! enforced one layer down and the mutation has to be applied there. The
    //! engine half of `compose.rs` was diff-verified byte-identical after every
    //! row above was reverted.
    //!
    //! M3b is the row worth reading twice: it is the only mutation that gets
    //! past every cursor guard the engine has, and the only test that sees it
    //! is the one comparing *which entries carry which version* against live
    //! history. A lane that asserted `composed_seq` and content-equality alone
    //! would have shipped green.
    //!
    //! Lane C appends its rows below. M10–M12 (its three planned rows) were
    //! measured against the **28** tests this module held at that point;
    //! M13–M15 against the **31** it holds now, and each row is left at the
    //! number its author actually collected. The two collector rows also name
    //! their reds in `storage::external_values`, whose suite is where the
    //! collector's own arms live (this lane owns those too —
    //! `collect_external_values` is [`decode_tail_record_ompf`]'s step 4).
    //!
    //! | # | Mutation | Red | Survivors |
    //! |---|---|---|---|
    //! | M10 | verbatim records routed through the **base** decoder: a `decode_checkpoint_ompf(object_id, ompf)?` validation pass added at the top of the `TailRecord::Verbatim` arm | 20 of 28 | the arms that never reach record replay (base-decode failure, unknown format version, over-ceiling cursor), the two record-less tails (zero-length, all-empty-delta), the fixture self-check and #B10 |
    //! | M11 | first-appearance order dropped: `collect_external_values` collects into a `HashMap` and returns `into_iter()` | 0 of 28 + 2–3 of 67 | **everything in this module, all three runs** — including #12 |
    //! | M12 | cross-map dedup dropped: the `seen` set re-declared before the `time_samples` loop | 0 of 28 + 1 of 67 | everything except the collector's cross-map arm |
    //! | M13 | [`decode_tail_record_ompf`] step 2 disabled, so an empty `delta_bytes` reaches `parse_delta` (`OodfError::EmptyInput`) | 1 of 31 | everything but #C4 |
    //! | M14 | step 4's ext-collection error swallowed: `collect_external_values(..).unwrap_or_default()` | 1 of 31 | everything but #C5 |
    //! | M15 | zero-copy dropped: `record_ompf.slice_ref(data)` → `Bytes::copy_from_slice(data)` | 1 of 31 | everything but #C6 |
    //!
    //! M13–M15 are `decode_tail_record_ompf`'s own boundary, added by this lane
    //! after the three planned rows: each turns exactly ONE test red, which is
    //! the measurement that says the arm was missing rather than redundant. M15
    //! in particular is invisible to every content assertion in this module —
    //! the copy is *correct*, it just costs up to 64 MiB × 8 per restore.
    //!
    //! M10 is the row this lane exists for, and #C3 is in its red set: the
    //! literal coupling is caught structurally *and* behaviourally. Note what
    //! the survivors say — a mutation that makes the tail path inherit the
    //! reset guard is invisible to any arm whose records never replay, which is
    //! why #C1 / #C2 assert the base decoder's verdict on the *same bytes*
    //! rather than trusting composition's success alone.
    //!
    //! M11 and M12 are the two rows that turned **nothing** red in this module,
    //! and they are why the collector's arms are this lane's to own. #12
    //! asserts `ext_values` equality, so it looks like the order/dedup fence —
    //! it is not. Its list is two entries long, so M11 is a coin flip it won
    //! three times out of three; and no [`timeline`] commit references one hash
    //! from a `SetField` *and* a `TimeSample`, so M12 is invisible to it by
    //! construction. Both are pinned in `storage::external_values::tests`
    //! instead
    //! (`collect_external_values_orders_by_first_appearance_across_both_maps`,
    //! `..._dedups_a_hash_shared_by_a_field_and_a_time_sample`), on lists long
    //! enough and shaped sharply enough to make the mutation deterministic.
    //!
    //! # Scope fences
    //!
    //! - Every **failure** arm of [`ComposeError`] belongs to lane B in the
    //!   middle of this module, which builds on the fixtures below.
    //! - [`decode_tail_record_ompf`]'s own boundary behaviour (empty
    //!   `delta_bytes`, ext-index resolution, the zero-copy view) is lane C's
    //!   (`#C4`–`#C6`); malformed containers and verifier rejections are lane
    //!   B's (#B1 / #B2). #12 asserts only the engine's surfacing contract:
    //!   order and dedup — and only lane C's collector arms in
    //!   `storage::external_values::tests` actually fence it (see M11 / M12).
    //! - The base decoder's own shape guard is `apply.rs`'s; that the tail path
    //!   does not inherit it is lane C's (`#C1`–`#C3`).
    //! - Lineage gather, persistence and the restore call site are later Tasks.

    use super::*;

    use crate::checkpoint::assembler::build_reset_ompf_from_state;
    // The collecting wrapper — the fixture drivers advance a reference state
    // and ignore the mutations; production replay uses `apply_delta_into`.
    use crate::checkpoint::tail_codec::{
        encode_tail_envelope, TailHeader, TAIL_ENVELOPE_MAGIC, TAIL_FORMAT_VERSION,
        TAIL_FORMAT_VERSION_OFFSET, TAIL_HEADER_SIZE, TAIL_MAX_SEQUENCE,
    };
    use crate::domain::delta::apply_delta;
    use crate::domain::limits::MAX_DELTA_SET_ORDER;
    use crate::domain::types::FieldValue;
    use oodf_ffi::{
        CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
        SetField as OodfSetField, StructCommand, TimeSample as OodfTimeSample,
    };
    use ordered_float::OrderedFloat;
    use std::collections::{BTreeMap, BTreeSet};

    // ── Constants ───────────────────────────────────────────────────────────

    /// The one layer generation every fixture composes.
    pub(crate) const OBJECT_ID: u64 = 7_701;

    /// Where [`timeline`] is cut into base era and tail era. Chosen so BOTH
    /// eras carry surviving sections, fields and a time sample: a cut with an
    /// empty base era would make the exclusion tests (#4, #5) vacuous.
    pub(crate) const BASE_SNAP_SEQ: u64 = 6;

    /// [`timeline`]'s head sequence, `S`.
    pub(crate) const HEAD_SEQ: u64 = 14;

    /// `source_format` the seq-1 reset establishes.
    pub(crate) const SOURCE_FORMAT: &str = "usda";

    /// `source_format` the mid-tail reset re-establishes — deliberately
    /// different from [`SOURCE_FORMAT`], so #10 cannot pass on a stale value.
    pub(crate) const RESET_SOURCE_FORMAT: &str = "usdc";

    /// Base of the synthetic `server_ts` series; distinct per sequence.
    const SERVER_TS_BASE: i64 = 1_700_000_000_000;

    /// Content hash referenced only in the BASE era. Never appears in
    /// `ComposedLineage::ext_values` — the base blob carries no OMPF parts
    /// (`build_reset_ompf_from_state` serializes with an empty parts list), so
    /// composition has no bytes to surface for it. #12 pins that boundary.
    pub(crate) const HASH_A: [u8; 32] = [0xa1; 32];
    /// Content hash referenced by TWO tail records — the dedup subject of #12.
    pub(crate) const HASH_B: [u8; 32] = [0xb2; 32];
    /// Content hash referenced by one tail record, second in first-appearance
    /// order.
    pub(crate) const HASH_C: [u8; 32] = [0xc3; 32];

    /// Payload bytes for [`HASH_A`]. Length differs per blob so a
    /// hash-to-bytes mix-up cannot pass a length-blind comparison.
    pub(crate) fn blob_a() -> Vec<u8> {
        b"ext-blob-A".to_vec()
    }
    /// Payload bytes for [`HASH_B`].
    pub(crate) fn blob_b() -> Vec<u8> {
        b"ext-blob-BB".to_vec()
    }
    /// Payload bytes for [`HASH_C`].
    pub(crate) fn blob_c() -> Vec<u8> {
        b"ext-blob-CCC".to_vec()
    }

    /// Distinct, self-identifying inline bytes per sequence.
    pub(crate) fn payload_for(seq: u64) -> Vec<u8> {
        format!("payload-seq-{seq}").into_bytes()
    }

    /// Distinct time-sample bytes per sequence.
    fn ts_payload_for(seq: u64) -> Vec<u8> {
        format!("ts-seq-{seq}").into_bytes()
    }

    /// Distinct server timestamp per sequence.
    pub(crate) fn server_ts_for(seq: u64) -> i64 {
        SERVER_TS_BASE + seq as i64
    }

    // ── Fixture vocabulary: commits ─────────────────────────────────────────

    /// One live commit: the delta the writer sent (`None` for the §8.4 empty
    /// commit) and the OMPF Parts 1..N its container carried.
    #[derive(Debug, Clone)]
    pub(crate) struct Commit {
        /// The delta. `None` models the empty commit, which carries no OMPF at
        /// all and is captured as [`TailRecord::EmptyDelta`].
        pub(crate) delta: Option<OwnedDelta>,
        /// External-value parts, in OMPF part order. A `SetField` referencing
        /// index `i` resolves to `ext_parts[i - 1]`.
        pub(crate) ext_parts: Vec<Vec<u8>>,
    }

    impl Commit {
        /// A commit carrying no external values.
        fn inline(delta: OwnedDelta) -> Self {
            Commit {
                delta: Some(delta),
                ext_parts: Vec::new(),
            }
        }

        /// A commit whose container carries `ext_parts` as Parts 1..N.
        fn with_ext(delta: OwnedDelta, ext_parts: Vec<Vec<u8>>) -> Self {
            Commit {
                delta: Some(delta),
                ext_parts,
            }
        }

        /// The §8.4 empty commit.
        fn empty() -> Self {
            Commit {
                delta: None,
                ext_parts: Vec::new(),
            }
        }
    }

    /// A delta shell with the non-reset defaults live `UpdateLayer` traffic
    /// carries (`is_diff = false`, `base_version = 0`).
    fn delta() -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            base_version: 0,
            ..OwnedDelta::default()
        }
    }

    /// A reset-shaped delta: exactly `is_diff = true, base_version = 0`, the
    /// shape both `CreateLayer` and an `UpdateLayer` reset commit.
    fn reset_delta(source_format: &str) -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: source_format.to_string(),
            ..OwnedDelta::default()
        }
    }

    fn create_section(
        parent_id: u64,
        section_id: u64,
        name: &str,
        section_type: u8,
    ) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type,
        })
    }

    fn inline_field(section_id: u64, key: &str, seq: u64, set_order: u64) -> OodfSetField {
        OodfSetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: payload_for(seq),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }
    }

    fn ext_field(
        section_id: u64,
        key: &str,
        hash: [u8; 32],
        size: u64,
        ext_value_index: u32,
        set_order: u64,
    ) -> OodfSetField {
        OodfSetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: size,
            ext_value_index,
            set_order,
        }
    }

    fn inline_time_sample(section_id: u64, time: f64, seq: u64) -> OodfTimeSample {
        OodfTimeSample {
            section_id,
            time,
            value_or_ext_hash: ts_payload_for(seq),
            ext_value_size: 0,
            ext_value_index: 0,
        }
    }

    /// The one fixed, deterministic history the lane rests on.
    ///
    /// `commits[k - 1]` is the commit that lands at `sequence_num == k`;
    /// `S == HEAD_SEQ == 14` and the base cut is `BASE_SNAP_SEQ == 6`.
    ///
    /// Every primitive the engine must carry appears, and — load-bearing —
    /// each appears in a position where it still *matters at S*:
    ///
    /// | seq | era | what it exercises |
    /// |---|---|---|
    /// | 1 | base | reset-shaped commit (`CreateLayer`), nested `CreateSection`, `source_format` |
    /// | 2 | base | second child under 10, inline `SetField` with authoring rank 1 |
    /// | 3 | base | `TimeSample` that survives untouched to S (base-era version pin) |
    /// | 4 | base | external-ref `SetField` at live `ext_value_index = 3` — re-derived to 1 by the base blob (#4b) |
    /// | 5 | base | new root 13 **and** an overwrite of `base_key_1`, which is what makes the live cumulative `set_order` cursor diverge from the re-derived one (#5) |
    /// | 6 | base | `DiffChangeSection` setting a non-zero `section_order`, so the `children` list at 10 is not in insertion order |
    /// | 7 | tail | `CreateSection` at root, first tail record → stamps `BASE_SNAP_SEQ + 1` (#6) |
    /// | 8 | tail | overwrite of a base-era field → tail-era version stamp |
    /// | 9 | tail | the §8.4 empty commit (#7) |
    /// | 10 | tail | external-ref `SetField` (first appearance of `HASH_B`) + `TimeSample` |
    /// | 11 | tail | nested `CreateSection` + two ext refs, one repeating `HASH_B` (#12 dedup) |
    /// | 12 | tail | `MoveSection` across parents |
    /// | 13 | tail | `DeleteSection` cascade (takes section 15 and both its ext fields with it) |
    /// | 14 | tail | trailing inline `SetField` + `TimeSample` at S |
    ///
    /// The mid-tail reset is deliberately NOT here: a reset wipes every
    /// base-era entry, which would make #3/#4/#5 vacuous. It lives in
    /// [`timeline_with_mid_tail_reset`], which #10 owns.
    pub(crate) fn timeline() -> Vec<Commit> {
        vec![
            // seq 1 — reset.
            Commit::inline(OwnedDelta {
                struct_commands: vec![
                    create_section(0, 10, "root_10", 1),
                    create_section(10, 11, "child_11", 2),
                ],
                set_fields: vec![inline_field(10, "base_key_1", 1, 0)],
                ..reset_delta(SOURCE_FORMAT)
            }),
            // seq 2.
            Commit::inline(OwnedDelta {
                struct_commands: vec![create_section(10, 12, "child_12", 2)],
                set_fields: vec![inline_field(11, "base_key_2", 2, 1)],
                ..delta()
            }),
            // seq 3.
            Commit::inline(OwnedDelta {
                time_samples: vec![inline_time_sample(11, 0.5, 3)],
                ..delta()
            }),
            // seq 4 — external ref at part index 3 of 3.
            Commit::with_ext(
                OwnedDelta {
                    set_fields: vec![ext_field(
                        12,
                        "base_ext_key",
                        HASH_A,
                        blob_a().len() as u64,
                        3,
                        2,
                    )],
                    ..delta()
                },
                vec![
                    b"unreferenced-part-1".to_vec(),
                    b"unreferenced-part-2".to_vec(),
                    blob_a(),
                ],
            ),
            // seq 5 — new root, plus an overwrite that the base blob cannot carry.
            Commit::inline(OwnedDelta {
                struct_commands: vec![create_section(0, 13, "root_13", 1)],
                set_fields: vec![
                    inline_field(13, "base_key_5", 5, 3),
                    inline_field(10, "base_key_1", 5, 0),
                ],
                ..delta()
            }),
            // seq 6 — non-zero section_order, reordering 10's children.
            Commit::inline(OwnedDelta {
                struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                    parent_id: 10,
                    section_id: 11,
                    section_name: "child_11".to_string(),
                    section_type: 2,
                    section_order: 5,
                })],
                ..delta()
            }),
            // ── base cut at BASE_SNAP_SEQ = 6 ───────────────────────────────
            // seq 7.
            Commit::inline(OwnedDelta {
                struct_commands: vec![create_section(0, 14, "root_14", 3)],
                set_fields: vec![inline_field(14, "tail_key_7", 7, 0)],
                ..delta()
            }),
            // seq 8 — overwrite of a base-era field.
            Commit::inline(OwnedDelta {
                set_fields: vec![inline_field(11, "base_key_2", 8, 1)],
                ..delta()
            }),
            // seq 9 — the empty commit.
            Commit::empty(),
            // seq 10 — first appearance of HASH_B.
            Commit::with_ext(
                OwnedDelta {
                    set_fields: vec![ext_field(
                        14,
                        "tail_ext_a",
                        HASH_B,
                        blob_b().len() as u64,
                        1,
                        2,
                    )],
                    time_samples: vec![inline_time_sample(14, 1.5, 10)],
                    ..delta()
                },
                vec![blob_b()],
            ),
            // seq 11 — HASH_B repeated, HASH_C new.
            Commit::with_ext(
                OwnedDelta {
                    struct_commands: vec![create_section(14, 15, "child_15", 2)],
                    set_fields: vec![
                        ext_field(15, "tail_ext_b", HASH_B, blob_b().len() as u64, 1, 0),
                        ext_field(15, "tail_ext_c", HASH_C, blob_c().len() as u64, 2, 1),
                    ],
                    ..delta()
                },
                vec![blob_b(), blob_c()],
            ),
            // seq 12 — move 12 from root_10 to root_14.
            Commit::inline(OwnedDelta {
                struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 10,
                    new_parent_id: 14,
                    section_id: 12,
                    new_name: "moved_12".to_string(),
                })],
                ..delta()
            }),
            // seq 13 — cascade delete of 15 (and both its ext fields).
            Commit::inline(OwnedDelta {
                struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: 14,
                    section_id: 15,
                })],
                ..delta()
            }),
            // seq 14 — trailing writes at S.
            Commit::inline(OwnedDelta {
                set_fields: vec![inline_field(10, "tail_key_14", 14, 4)],
                time_samples: vec![inline_time_sample(10, 2.25, 14)],
                ..delta()
            }),
        ]
    }

    /// [`timeline`]'s first eight commits, then a **reset-shaped** commit at
    /// seq 9 and one ordinary commit on top of it. `S == 10`, base cut still
    /// [`BASE_SNAP_SEQ`].
    ///
    /// The reset lands strictly inside the tail range, which is the point: it
    /// must replay through `apply_delta` unfiltered and wipe the base-restored
    /// state exactly as the live commit wiped the live state.
    pub(crate) fn timeline_with_mid_tail_reset() -> Vec<Commit> {
        let mut commits: Vec<Commit> = timeline().into_iter().take(8).collect();
        // seq 9 — the mid-tail reset.
        commits.push(Commit::inline(OwnedDelta {
            struct_commands: vec![
                create_section(0, 20, "reset_root_20", 4),
                create_section(20, 21, "reset_child_21", 5),
            ],
            set_fields: vec![inline_field(20, "reset_key_9", 9, 2)],
            time_samples: vec![inline_time_sample(21, 3.5, 9)],
            ..reset_delta(RESET_SOURCE_FORMAT)
        }));
        // seq 10 — an ordinary commit on top of the reset.
        commits.push(Commit::inline(OwnedDelta {
            set_fields: vec![inline_field(21, "post_reset_key_10", 10, 0)],
            ..delta()
        }));
        commits
    }

    // ── Fixture vocabulary: driving a history ───────────────────────────────

    /// Serialize `delta` into an OMPF container carrying `ext_parts` as Parts
    /// 1..N **and** advance `state` through the same value.
    ///
    /// The single helper both sides go through: the record's bytes and the
    /// reference state can never describe different deltas.
    pub(crate) fn commit_with_ext(
        state: &mut ObjectState,
        delta: &OwnedDelta,
        ext_parts: &[Vec<u8>],
    ) -> Bytes {
        let delta_bytes = oodf_ffi::build_delta(delta).expect("fixture delta must serialize");
        let parts: Vec<&[u8]> = ext_parts.iter().map(Vec::as_slice).collect();
        let ompf = OmpfContainer::serialize(&delta_bytes, &parts)
            .expect("fixture OMPF container must serialize");
        apply_delta(state, Some(delta.clone()))
            .expect("fixture delta must apply to the reference state");
        Bytes::from(ompf)
    }

    /// [`commit_with_ext`] for a commit that carries no external values.
    pub(crate) fn commit(state: &mut ObjectState, delta: &OwnedDelta) -> Bytes {
        commit_with_ext(state, delta, &[])
    }

    /// A driven history: the reference state after every commit, and the tail
    /// record each commit was captured as.
    #[derive(Debug)]
    pub(crate) struct History {
        /// `states[k]` is the reference state after commit `k`; `states[0]` is
        /// the empty pre-history state.
        pub(crate) states: Vec<ObjectState>,
        /// `records[k - 1]` carries `sequence_num == k`.
        pub(crate) records: Vec<TailRecord>,
    }

    impl History {
        /// The reference state at `seq`.
        pub(crate) fn state_at(&self, seq: u64) -> &ObjectState {
            &self.states[seq as usize]
        }

        /// The records covering the exclusive-lower-bound span
        /// `(from_seq, to_seq]`.
        pub(crate) fn records_in(&self, from_seq: u64, to_seq: u64) -> Vec<TailRecord> {
            self.records[from_seq as usize..to_seq as usize].to_vec()
        }
    }

    /// Replay `commits` as live history: one `apply_delta` per commit against a
    /// single reference state, one captured [`TailRecord`] per commit.
    pub(crate) fn run(commits: &[Commit]) -> History {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            ..ObjectState::default()
        };
        let mut states = vec![state.clone()];
        let mut records = Vec::with_capacity(commits.len());

        for (index, spec) in commits.iter().enumerate() {
            let sequence_num = index as u64 + 1;
            let server_ts = server_ts_for(sequence_num);
            let record = match &spec.delta {
                None => {
                    assert!(
                        spec.ext_parts.is_empty(),
                        "an empty commit carries no OMPF, so it cannot carry ext parts"
                    );
                    apply_delta(&mut state, None).expect("empty commit must apply");
                    TailRecord::EmptyDelta {
                        sequence_num,
                        server_ts,
                    }
                }
                Some(d) if spec.ext_parts.is_empty() => TailRecord::Verbatim {
                    sequence_num,
                    server_ts,
                    ompf: commit(&mut state, d),
                },
                Some(d) => TailRecord::Verbatim {
                    sequence_num,
                    server_ts,
                    ompf: commit_with_ext(&mut state, d, &spec.ext_parts),
                },
            };
            assert_eq!(
                state.sequence_num, sequence_num,
                "reference history must advance exactly one sequence per commit"
            );
            states.push(state.clone());
            records.push(record);
        }

        History { states, records }
    }

    /// The base checkpoint blob the real assembler emits for `state`.
    ///
    /// `.0` is the OMPF; `.1` is the ext-refs sidecar hash list, which the
    /// composition engine does not consume (the base blob carries no OMPF
    /// parts, so base-era ext bytes never reach `ComposedLineage::ext_values`
    /// — see #12).
    pub(crate) fn base_ompf(state: &ObjectState) -> Bytes {
        let (bytes, _ordered_ext_hashes) =
            build_reset_ompf_from_state(state).expect("fixture base blob must assemble");
        Bytes::from(bytes)
    }

    /// A tail artifact over `(from_seq, from_seq + records.len()]`, built by the
    /// real encoder — never hand-assembled bytes.
    pub(crate) fn tail(base_snap_seq: u64, from_seq: u64, records: &[TailRecord]) -> Bytes {
        let header = TailHeader {
            object_id: OBJECT_ID,
            base_snap_seq,
            from_seq,
            to_seq: from_seq + records.len() as u64,
        };
        encode_tail_envelope(&header, records.to_vec()).expect("fixture tail must encode")
    }

    /// [`tail`], decoded — the `compose_lineage` input shape.
    pub(crate) fn tail_envelope(
        base_snap_seq: u64,
        from_seq: u64,
        records: &[TailRecord],
    ) -> TailEnvelope {
        decode_tail_envelope(&tail(base_snap_seq, from_seq, records))
            .expect("fixture tail must decode")
    }

    /// The canonical setup: the driven [`timeline`] history plus its base blob
    /// at [`BASE_SNAP_SEQ`].
    pub(crate) fn canonical() -> (History, Bytes) {
        let history = run(&timeline());
        let base = base_ompf(history.state_at(BASE_SNAP_SEQ));
        (history, base)
    }

    /// Compose the canonical lineage over `(BASE_SNAP_SEQ, to_seq]` as one tail.
    pub(crate) fn compose_canonical(
        history: &History,
        base: &Bytes,
        to_seq: u64,
    ) -> ComposedLineage {
        let records = history.records_in(BASE_SNAP_SEQ, to_seq);
        let envelope = tail_envelope(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &records);
        compose_lineage(OBJECT_ID, base, BASE_SNAP_SEQ, &[envelope])
            .expect("canonical lineage must compose")
    }

    // ── Fixture vocabulary: assertions ──────────────────────────────────────

    /// A single addressable entry in an [`ObjectState`].
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) enum EntryKey {
        Section(u64),
        Field(u64, String),
        TimeSample(u64, OrderedFloat<f64>),
    }

    /// Compare the columns a reset-OMPF round trip carries faithfully, and
    /// ONLY those.
    ///
    /// Faithful: `object_id`, `source_format`, section id set + each section's
    /// `parent_id` / `name` / `section_type` / `section_order`, the `children`
    /// lists in order, field key set + `FieldValue` (external: **hash and size
    /// only**) + `authoring_set_order`, time-sample key set + `FieldValue`.
    ///
    /// Deliberately NOT compared, because the base decode re-derives them —
    /// each has its own pinning test, and the module header's table is the
    /// index:
    ///
    /// 1. `Field::version` / `TimeSample::version` (base era) → #4
    /// 2. `Section::first_version` → #4 (b.2ti)
    /// 3. `Section::last_version` (base era) → #4 (b.2ti)
    /// 4. `ObjectState::first_version` → #4, #10 (b.2ti)
    /// 5. `Field::set_order` (cumulative) → #5
    /// 6. `ObjectState::next_set_order` → #5
    /// 7. `FieldValue::External::ext_value_index` → #4b
    ///
    /// `sequence_num` is not compared here either — it is the subject of #1.
    pub(crate) fn assert_faithful_columns(composed: &ObjectState, reference: &ObjectState) {
        assert_eq!(composed.object_id, reference.object_id, "object_id");
        assert_eq!(
            composed.source_format, reference.source_format,
            "source_format"
        );

        let composed_sections: BTreeSet<u64> = composed.sections.keys().copied().collect();
        let reference_sections: BTreeSet<u64> = reference.sections.keys().copied().collect();
        assert_eq!(
            composed_sections, reference_sections,
            "section id set must match live history"
        );
        for id in &reference_sections {
            let c = &composed.sections[id];
            let r = &reference.sections[id];
            assert_eq!(c.parent_id, r.parent_id, "section {id} parent_id");
            assert_eq!(c.name, r.name, "section {id} name");
            assert_eq!(c.section_type, r.section_type, "section {id} section_type");
            assert_eq!(
                c.section_order, r.section_order,
                "section {id} section_order"
            );
        }

        // Children lists compared verbatim: order is load-bearing (authoring
        // order on the wire) and the reset blob's DFS emission is what
        // reproduces it.
        let composed_children: BTreeMap<u64, Vec<u64>> = composed
            .children
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let reference_children: BTreeMap<u64, Vec<u64>> = reference
            .children
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        assert_eq!(
            composed_children, reference_children,
            "children lists (ids and order)"
        );

        let composed_field_keys: BTreeSet<(u64, String)> =
            composed.fields.keys().cloned().collect();
        let reference_field_keys: BTreeSet<(u64, String)> =
            reference.fields.keys().cloned().collect();
        assert_eq!(
            composed_field_keys, reference_field_keys,
            "field key set must match live history"
        );
        for key in &reference_field_keys {
            let c = &composed.fields[key];
            let r = &reference.fields[key];
            assert_field_value_faithful(&c.value, &r.value, &format!("field {key:?} value"));
            assert_eq!(
                c.authoring_set_order, r.authoring_set_order,
                "field {key:?} authoring_set_order"
            );
        }

        let composed_ts_keys: BTreeSet<(u64, OrderedFloat<f64>)> =
            composed.time_samples.keys().copied().collect();
        let reference_ts_keys: BTreeSet<(u64, OrderedFloat<f64>)> =
            reference.time_samples.keys().copied().collect();
        assert_eq!(
            composed_ts_keys, reference_ts_keys,
            "time-sample key set must match live history"
        );
        for key in &reference_ts_keys {
            let c = &composed.time_samples[key];
            let r = &reference.time_samples[key];
            assert_field_value_faithful(&c.value, &r.value, &format!("time sample {key:?} value"));
        }
    }

    /// `FieldValue` equality restricted to the faithful columns: inline bytes,
    /// or an external value's **hash and size**. `ext_value_index` is excluded
    /// — see #4b.
    pub(crate) fn assert_field_value_faithful(
        composed: &FieldValue,
        reference: &FieldValue,
        what: &str,
    ) {
        match (composed, reference) {
            (FieldValue::Inline(a), FieldValue::Inline(b)) => {
                assert_eq!(a, b, "{what}: inline bytes");
            }
            (
                FieldValue::External {
                    hash: ch, size: cs, ..
                },
                FieldValue::External {
                    hash: rh, size: rs, ..
                },
            ) => {
                assert_eq!(ch, rh, "{what}: external hash");
                assert_eq!(cs, rs, "{what}: external size");
            }
            _ => panic!("{what}: FieldValue discriminant mismatch (inline vs external)"),
        }
    }

    /// Every entry stamped at a version strictly above `base`, grouped by that
    /// version. Sections are keyed on `last_version` — the version of the write
    /// that last touched them.
    pub(crate) fn versions_above(
        state: &ObjectState,
        base: u64,
    ) -> BTreeMap<u64, BTreeSet<EntryKey>> {
        let mut out: BTreeMap<u64, BTreeSet<EntryKey>> = BTreeMap::new();
        for (id, section) in &state.sections {
            if section.last_version > base {
                out.entry(section.last_version)
                    .or_default()
                    .insert(EntryKey::Section(*id));
            }
        }
        for ((sid, key), field) in &state.fields {
            if field.version > base {
                out.entry(field.version)
                    .or_default()
                    .insert(EntryKey::Field(*sid, key.clone()));
            }
        }
        for ((sid, time), sample) in &state.time_samples {
            if sample.version > base {
                out.entry(sample.version)
                    .or_default()
                    .insert(EntryKey::TimeSample(*sid, *time));
            }
        }
        out
    }

    /// Assert two states carry identical content, ignoring only
    /// `sequence_num` (which the caller is asserting separately).
    pub(crate) fn assert_content_identical(left: &ObjectState, right: &ObjectState, what: &str) {
        let mut a = left.clone();
        let mut b = right.clone();
        a.sequence_num = 0;
        b.sequence_num = 0;
        assert_eq!(a, b, "{what}");
    }

    /// The decoded base — what a base-only restore materializes today.
    fn decoded_base(base: &Bytes) -> ObjectState {
        decode_checkpoint_ompf(OBJECT_ID, base).expect("fixture base blob must decode")
    }

    // ── Fixture self-checks ─────────────────────────────────────────────────
    //
    // The whole lane rests on `timeline()` actually exercising what its table
    // claims. A silently-degraded fixture (a section that never survived to S,
    // an era that carries nothing) would make the exclusion tests vacuously
    // green, so the shape is asserted rather than assumed.

    #[test]
    fn timeline_fixture_covers_both_eras_and_every_primitive() {
        let (history, _base) = canonical();
        assert_eq!(
            history.records.len(),
            HEAD_SEQ as usize,
            "timeline length must equal S"
        );
        let head = history.state_at(HEAD_SEQ);

        // Base era leaves surviving entries at S — otherwise #4 / #5 are vacuous.
        let base_era_fields: Vec<_> = head
            .fields
            .iter()
            .filter(|(_, f)| f.version <= BASE_SNAP_SEQ)
            .collect();
        assert!(
            base_era_fields.len() >= 3,
            "base era must leave >= 3 surviving fields at S, got {}",
            base_era_fields.len()
        );
        assert!(
            head.time_samples
                .values()
                .any(|ts| ts.version <= BASE_SNAP_SEQ),
            "base era must leave a surviving time sample at S"
        );
        assert!(
            head.sections
                .values()
                .any(|s| s.first_version > 1 && s.first_version <= BASE_SNAP_SEQ),
            "base era must leave a section whose live first_version is neither 1 nor tail-era \
             — otherwise #4's section arm cannot distinguish the decoder's stamp"
        );

        // Tail era writes too.
        assert!(
            head.fields.values().any(|f| f.version > BASE_SNAP_SEQ),
            "tail era must leave surviving fields at S"
        );

        // Primitive coverage that must reach S.
        assert!(
            matches!(
                head.fields[&(12, "base_ext_key".to_string())].value,
                FieldValue::External { hash: HASH_A, .. }
            ),
            "base-era external ref must survive to S"
        );
        assert!(
            head.children[&10].len() == 1 && head.children[&0].len() == 3,
            "children lists must be non-trivial at S"
        );
        assert_eq!(
            head.sections[&12].parent_id, 14,
            "MoveSection must have taken effect"
        );
        assert!(
            !head.sections.contains_key(&15),
            "DeleteSection cascade must have removed section 15"
        );
        assert!(
            history
                .records
                .iter()
                .any(|r| matches!(r, TailRecord::EmptyDelta { .. })),
            "timeline must carry an empty commit"
        );
        assert_eq!(head.source_format, SOURCE_FORMAT);
    }

    // ── #1 — terminal sequence ──────────────────────────────────────────────

    #[test]
    fn compose_base_and_tail_returns_final_sequence_num_at_to_seq() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);

        assert_eq!(
            composed.composed_seq, HEAD_SEQ,
            "composed_seq must land on the tail's to_seq"
        );
        assert_eq!(
            composed.state.sequence_num, HEAD_SEQ,
            "the composed state's own cursor must agree with composed_seq"
        );
        assert_eq!(
            composed.state.sequence_num,
            history.state_at(HEAD_SEQ).sequence_num,
            "and with the sequence live history reached"
        );
    }

    // ── #2 — content equivalence ────────────────────────────────────────────

    #[test]
    fn compose_base_and_tail_content_equals_live_state_at_s() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);

        assert_faithful_columns(&composed.state, history.state_at(HEAD_SEQ));
    }

    // ── #3 — tail-era version stamping ──────────────────────────────────────

    #[test]
    fn compose_stamps_tail_written_entries_at_their_live_commit_sequence() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);
        let live = history.state_at(HEAD_SEQ);

        let composed_by_version = versions_above(&composed.state, BASE_SNAP_SEQ);
        let live_by_version = versions_above(live, BASE_SNAP_SEQ);

        assert_eq!(
            composed_by_version, live_by_version,
            "every entry the tail range wrote must carry its live commit sequence, and no other \
             entry may claim one"
        );

        // Non-vacuity: exactly the commits whose writes survive to S appear.
        // 9 is the empty commit; 11's section and both its ext fields are
        // cascade-deleted at 13; 13 itself only deletes.
        let sequences: BTreeSet<u64> = composed_by_version.keys().copied().collect();
        assert_eq!(
            sequences,
            BTreeSet::from([7, 8, 10, 12, 14]),
            "the surviving tail-era commit stamps"
        );

        // Spot pins, so a wholesale shift of both sides cannot pass.
        assert_eq!(
            composed.state.fields[&(11, "base_key_2".to_string())].version,
            8,
            "a base-era field overwritten at 8 must carry 8"
        );
        assert_eq!(composed.state.sections[&12].last_version, 12);
        assert_eq!(
            composed.state.time_samples[&(10, OrderedFloat(2.25))].version,
            14
        );
    }

    // ── #4 — base-era entries keep the decoder's flattened stamp ────────────

    #[test]
    fn compose_leaves_base_only_entries_at_the_base_decoders_stamp() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);
        let live = history.state_at(HEAD_SEQ);

        // `decode_checkpoint_ompf` applies the reset delta to a fresh state, so
        // `apply_delta` computes `new_version = 1` for everything the base
        // carried. Composition seeds only the cursor; it does not re-stamp.
        const DECODER_STAMP: u64 = 1;
        let expected = |live_version: u64| {
            if live_version > BASE_SNAP_SEQ {
                live_version
            } else {
                DECODER_STAMP
            }
        };

        let mut base_era_fields = 0;
        for (key, field) in &live.fields {
            assert_eq!(
                composed.state.fields[key].version,
                expected(field.version),
                "field {key:?}: live version {}",
                field.version
            );
            if field.version <= BASE_SNAP_SEQ {
                base_era_fields += 1;
            }
        }
        assert!(base_era_fields >= 3, "arm must not be vacuous");

        let mut base_era_samples = 0;
        for (key, sample) in &live.time_samples {
            assert_eq!(
                composed.state.time_samples[key].version,
                expected(sample.version),
                "time sample {key:?}: live version {}",
                sample.version
            );
            if sample.version <= BASE_SNAP_SEQ {
                base_era_samples += 1;
            }
        }
        assert!(base_era_samples >= 1, "arm must not be vacuous");

        // C2 / b.2ti: SECTION versions are re-derived the same way. Both
        // columns, not just `first_version`.
        let mut base_era_sections = 0;
        for (id, section) in &live.sections {
            let c = &composed.state.sections[id];
            assert_eq!(
                c.first_version,
                expected(section.first_version),
                "section {id} first_version: live {}",
                section.first_version
            );
            assert_eq!(
                c.last_version,
                expected(section.last_version),
                "section {id} last_version: live {}",
                section.last_version
            );
            if section.first_version <= BASE_SNAP_SEQ {
                base_era_sections += 1;
            }
        }
        assert!(base_era_sections >= 3, "arm must not be vacuous");

        // Non-vacuity for the exclusion itself: at least one base-era entry
        // whose live version is NOT the decoder's stamp, so the assertions
        // above could actually have failed.
        assert_eq!(
            live.fields[&(13, "base_key_5".to_string())].version,
            5,
            "live history must stamp this field at 5"
        );
        assert_eq!(
            composed.state.fields[&(13, "base_key_5".to_string())].version,
            DECODER_STAMP
        );
        assert_eq!(live.sections[&12].first_version, 2);
        assert_eq!(composed.state.sections[&12].first_version, DECODER_STAMP);

        // C2 / b.2ti: `ObjectState::first_version` is zeroed by the decoder and
        // composition never re-stamps it — only a replayed reset would (#10).
        assert_eq!(live.first_version, 1);
        assert_eq!(
            composed.state.first_version, 0,
            "ObjectState::first_version is left at the decoder's zero"
        );
    }

    // ── #4b — ext_value_index is re-derived, not carried (C1) ───────────────

    #[test]
    fn compose_carries_base_era_ext_indices_re_derived_and_tail_era_indices_verbatim() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);
        let live = history.state_at(HEAD_SEQ);

        // Base era: the writer referenced OMPF part 3 of 3 at seq 4, and live
        // state kept that index. The base blob re-derives indices densely over
        // the WHOLE state (`collect_external_value_hashes`), and the base
        // carries exactly one distinct hash, so the restored index is 1.
        let base_key = (12, "base_ext_key".to_string());
        let FieldValue::External {
            ext_value_index: live_index,
            hash: live_hash,
            size: live_size,
        } = live.fields[&base_key].value
        else {
            panic!("fixture must keep a base-era external ref at S");
        };
        assert_eq!(live_index, 3, "live per-delta index");
        let FieldValue::External {
            ext_value_index: composed_index,
            hash: composed_hash,
            size: composed_size,
        } = composed.state.fields[&base_key].value
        else {
            panic!("composed state must keep the base-era external ref");
        };
        assert_eq!(
            composed_index, 1,
            "base-era entries carry the reset blob's dense re-derived index, not the live one"
        );
        // The faithful half of the comparison still holds.
        assert_eq!(composed_hash, live_hash);
        assert_eq!(composed_size, live_size);
        assert_eq!(composed_hash, HASH_A);

        // Tail era: the record's own per-delta index survives verbatim,
        // because the tail record is replayed, not re-serialized.
        let tail_key = (14, "tail_ext_a".to_string());
        let FieldValue::External {
            ext_value_index: composed_tail_index,
            ..
        } = composed.state.fields[&tail_key].value
        else {
            panic!("composed state must keep the tail-era external ref");
        };
        let FieldValue::External {
            ext_value_index: live_tail_index,
            ..
        } = live.fields[&tail_key].value
        else {
            panic!("fixture must keep a tail-era external ref at S");
        };
        assert_eq!(composed_tail_index, live_tail_index);
        assert_eq!(composed_tail_index, 1, "the seq-10 record's own index");
    }

    // ── #5 — the cumulative set-order cursor is re-derived from the base ────

    #[test]
    fn compose_re_derives_the_cumulative_set_order_cursor_from_the_base() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);
        let live = history.state_at(HEAD_SEQ);
        let decoded = decoded_base(&base);

        // The base blob carries only SURVIVING fields, each at its authoring
        // rank; the live cursor additionally folded the writes that were later
        // overwritten (seq 5 rewrites `base_key_1`). So the two diverge, and
        // the composed state inherits the re-derived one.
        assert_ne!(
            decoded.next_set_order,
            history.state_at(BASE_SNAP_SEQ).next_set_order,
            "the fixture must actually make the cursor diverge, else this test is vacuous"
        );

        // Base-era fields untouched by the tail keep exactly the cumulative
        // rank the base decode assigned them.
        for (key, field) in &decoded.fields {
            if live.fields[key].version > BASE_SNAP_SEQ {
                continue; // rewritten in the tail era; #3 owns those
            }
            assert_eq!(
                composed.state.fields[key].set_order, field.set_order,
                "field {key:?} cumulative set_order must come from the decoded base"
            );
        }

        // The cursor at S: the decoded base's cursor advanced by each tail
        // commit's authoring ranks, in order. Computed from the timeline's own
        // commit list, not from the engine.
        let mut expected_cursor = decoded.next_set_order;
        for spec in timeline().iter().skip(BASE_SNAP_SEQ as usize) {
            let Some(d) = spec.delta.as_ref() else {
                continue;
            };
            for sf in &d.set_fields {
                // `stored = cursor + authoring`; `cursor = max(cursor, stored + 1)`.
                expected_cursor = expected_cursor.max(expected_cursor + sf.set_order + 1);
            }
        }
        assert_eq!(
            composed.state.next_set_order, expected_cursor,
            "next_set_order must be the re-derived base cursor advanced by the tail's ranks"
        );
        assert_ne!(
            composed.state.next_set_order, live.next_set_order,
            "and must NOT equal the live cursor — that divergence is the point"
        );

        // The same divergence at field granularity, for a field the tail never
        // rewrote.
        let key = (10, "base_key_1".to_string());
        assert_eq!(live.fields[&key].set_order, 10);
        assert_ne!(
            composed.state.fields[&key].set_order, 10,
            "the cumulative rank does not round-trip; only `authoring_set_order` does"
        );
        assert_eq!(
            composed.state.fields[&key].authoring_set_order, live.fields[&key].authoring_set_order,
            "authoring_set_order DOES round-trip"
        );
    }

    // ── #6 — the first record stamps base_snap_seq + 1 (b.rrx) ──────────────

    #[test]
    fn compose_first_record_stamps_base_snap_seq_plus_one() {
        let (history, base) = canonical();
        let records = history.records_in(BASE_SNAP_SEQ, HEAD_SEQ);
        let envelope = tail_envelope(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &records);

        // `from_seq` is the EXCLUSIVE lower bound: the newest commit NOT
        // carried. With `from_seq == base_snap_seq`, the first carried commit
        // is `base_snap_seq + 1`.
        assert_eq!(envelope.header().from_seq, BASE_SNAP_SEQ);
        assert_eq!(envelope.records()[0].sequence_num(), BASE_SNAP_SEQ + 1);

        let composed = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[envelope])
            .expect("canonical lineage must compose");

        assert_eq!(
            composed.state.fields[&(14, "tail_key_7".to_string())].version,
            BASE_SNAP_SEQ + 1,
            "entries written by the first record carry base_snap_seq + 1"
        );
        assert_eq!(
            composed.state.sections[&14].first_version,
            BASE_SNAP_SEQ + 1
        );

        // Off-by-one downwards: nothing may carry `base_snap_seq` itself. Base
        // entries sit at the decoder's 1, tail entries at 7 and above.
        assert!(
            composed
                .state
                .fields
                .values()
                .all(|f| f.version != BASE_SNAP_SEQ)
                && composed
                    .state
                    .time_samples
                    .values()
                    .all(|t| t.version != BASE_SNAP_SEQ)
                && composed
                    .state
                    .sections
                    .values()
                    .all(|s| s.last_version != BASE_SNAP_SEQ),
            "no composed entry may be stamped at base_snap_seq"
        );
    }

    // ── #7 — an empty-delta record advances the cursor only ─────────────────

    #[test]
    fn compose_empty_delta_record_advances_the_cursor_without_changing_content() {
        let (history, base) = canonical();
        // seq 9 is the empty commit.
        let before = compose_canonical(&history, &base, 8);
        let after = compose_canonical(&history, &base, 9);
        let next = compose_canonical(&history, &base, 10);

        assert_eq!(before.composed_seq, 8);
        assert_eq!(
            after.composed_seq, 9,
            "the empty record advances the cursor"
        );
        assert_content_identical(
            &before.state,
            &after.state,
            "an empty commit must not change content",
        );
        assert_eq!(
            after.state.sequence_num, 9,
            "the state's own cursor advanced too"
        );

        // The non-vacuous half: an engine that SKIPPED the empty record instead
        // of running `apply_delta(state, None)` would leave the cursor at 8, so
        // the next record's writes would land at 9 rather than 10.
        assert_eq!(next.composed_seq, 10);
        assert_eq!(
            next.state.fields[&(14, "tail_ext_a".to_string())].version,
            10,
            "the record following an empty commit must stamp k + 1, not reuse the skipped slot"
        );
        assert_eq!(
            next.state.time_samples[&(14, OrderedFloat(1.5))].version,
            10
        );
    }

    // ── #8 — an all-empty tail ──────────────────────────────────────────────

    #[test]
    fn compose_all_empty_delta_tail_equals_base_content_at_to_seq() {
        let (_history, base) = canonical();
        let records: Vec<TailRecord> = (BASE_SNAP_SEQ + 1..=BASE_SNAP_SEQ + 3)
            .map(|sequence_num| TailRecord::EmptyDelta {
                sequence_num,
                server_ts: server_ts_for(sequence_num),
            })
            .collect();
        let envelope = tail_envelope(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &records);

        let composed = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[envelope])
            .expect("an all-empty tail must compose");

        assert_eq!(composed.composed_seq, BASE_SNAP_SEQ + 3);
        assert_eq!(composed.state.sequence_num, BASE_SNAP_SEQ + 3);
        assert_content_identical(
            &composed.state,
            &decoded_base(&base),
            "content must be the base restore's, unchanged",
        );
        assert!(
            composed.ext_values.is_empty(),
            "empty records collect no ext parts"
        );
    }

    // ── #9 — a zero-length tail ─────────────────────────────────────────────

    #[test]
    fn compose_zero_length_tail_equals_the_base_restore_with_the_cursor_at_base_snap_seq() {
        let (_history, base) = canonical();
        let artifact = tail(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &[]);
        assert_eq!(
            artifact.len(),
            TAIL_HEADER_SIZE,
            "a zero-length tail is a bare header"
        );

        let envelope = decode_tail_envelope(&artifact).expect("zero-length tail must decode");
        assert!(envelope.is_empty());
        assert_eq!(envelope.header().from_seq, envelope.header().to_seq);

        let composed = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[envelope])
            .expect("a zero-length tail must compose");

        assert_eq!(composed.composed_seq, BASE_SNAP_SEQ);
        assert_eq!(composed.state.sequence_num, BASE_SNAP_SEQ);
        assert_content_identical(
            &composed.state,
            &decoded_base(&base),
            "a zero-tail lineage is exactly the base restore",
        );
        assert!(composed.ext_values.is_empty());
    }

    // ── #10 — a mid-tail reset replays unfiltered ───────────────────────────

    #[test]
    fn compose_mid_tail_reset_record_wipes_and_rebuilds_exactly_as_live_did() {
        let commits = timeline_with_mid_tail_reset();
        let history = run(&commits);
        let head_seq = commits.len() as u64;
        assert_eq!(head_seq, 10);
        let base = base_ompf(history.state_at(BASE_SNAP_SEQ));

        // The reset record must actually be reset-shaped, or the test proves
        // nothing.
        let reset = commits[8].delta.as_ref().expect("seq 9 carries a delta");
        assert!(
            reset.is_diff && reset.base_version == 0,
            "seq 9 must be the reset shape an UpdateLayer reset commits"
        );

        let records = history.records_in(BASE_SNAP_SEQ, head_seq);
        let envelope = tail_envelope(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &records);
        let composed = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[envelope])
            .expect("a mid-tail reset must compose");
        let live = history.state_at(head_seq);

        assert_eq!(composed.composed_seq, head_seq);
        assert_faithful_columns(&composed.state, live);

        // The wipe actually happened on both sides.
        for id in [10u64, 11, 12, 13, 14] {
            assert!(
                !composed.state.sections.contains_key(&id),
                "section {id} must not survive the reset"
            );
            assert!(!live.sections.contains_key(&id));
        }
        assert_eq!(composed.state.source_format, RESET_SOURCE_FORMAT);

        // A replayed reset re-stamps every re-derived column, so the composed
        // state converges on live history EXACTLY — including the three
        // columns a base-only restore loses. That convergence is the sharpest
        // available evidence the reset ran rather than being filtered out.
        assert_eq!(composed.state.first_version, 9);
        assert_eq!(live.first_version, 9);
        assert_eq!(composed.state.next_set_order, live.next_set_order);
        assert_eq!(
            &composed.state, live,
            "a replayed mid-tail reset leaves no base-era divergence at all"
        );
    }

    // ── #11 — a flattened two-generation chain ──────────────────────────────

    #[test]
    fn compose_of_a_flattened_two_generation_chain_equals_the_single_tail_over_the_same_span() {
        let (history, base) = canonical();
        const SPLIT: u64 = 10;

        let first = tail_envelope(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &history.records_in(BASE_SNAP_SEQ, SPLIT),
        );
        let second = tail_envelope(BASE_SNAP_SEQ, SPLIT, &history.records_in(SPLIT, HEAD_SEQ));
        assert_eq!(
            second.header().from_seq,
            first.header().to_seq,
            "generation 2 must link onto generation 1"
        );

        let chained = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[first, second])
            .expect("a two-generation chain must compose");
        let single = compose_canonical(&history, &base, HEAD_SEQ);

        assert_eq!(chained.composed_seq, single.composed_seq);
        assert_eq!(
            chained.state, single.state,
            "chaining is a pure flattening: the composed state must be identical"
        );
        assert_eq!(
            chained.ext_values, single.ext_values,
            "and so must the ext parts, in the same order"
        );
        assert_faithful_columns(&chained.state, history.state_at(HEAD_SEQ));
    }

    // ── #12 — ext parts surfaced in first-appearance order, deduplicated ────

    #[test]
    fn compose_surfaces_each_records_ext_parts_in_order() {
        let (history, base) = canonical();
        let composed = compose_canonical(&history, &base, HEAD_SEQ);

        // seq 10 references HASH_B; seq 11 references HASH_B again and HASH_C.
        // Lineage-wide dedup keeps the FIRST appearance, in replay order.
        let expected: Vec<ExtValuePart> = vec![
            (HASH_B, Bytes::from(blob_b())),
            (HASH_C, Bytes::from(blob_c())),
        ];
        assert_eq!(
            composed.ext_values, expected,
            "ext parts must be (hash, bytes) pairs, deduplicated, in first-appearance order"
        );

        // The base era's HASH_A is deliberately absent: the base blob carries
        // no OMPF parts, so composition has no bytes for it. A caller that
        // needs base-era blobs gets them from the ext-refs sidecar, not here.
        assert!(
            composed.ext_values.iter().all(|(h, _)| *h != HASH_A),
            "base-era ext hashes are not surfaced by composition"
        );
        // ...even though the field referencing it is very much in the state.
        assert!(composed
            .state
            .fields
            .contains_key(&(12, "base_ext_key".to_string())));

        // Dedup is by hash across the whole lineage, not per record: HASH_B is
        // referenced by two records and three fields, and appears once.
        assert_eq!(
            composed
                .ext_values
                .iter()
                .filter(|(h, _)| *h == HASH_B)
                .count(),
            1
        );

        // Truncating the lineage before seq 11 drops HASH_C but keeps the
        // order of what remains — so the order assertion above is not an
        // accident of the full run.
        let through_10 = compose_canonical(&history, &base, 10);
        assert_eq!(
            through_10.ext_values,
            vec![(HASH_B, Bytes::from(blob_b()))],
            "a lineage that stops before HASH_C's record must not surface it"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Lane B — SR-3.5 / SR-6.1 failure arms, `Permanent` classification, the
    // sequence-ceiling boundary, and the two structural guards.
    //
    // Every arm drives lane A's `timeline()` / `canonical()` / `tail()`
    // vocabulary above rather than a second fixture set, so a failure arm and
    // the happy path it mirrors cannot drift apart.
    //
    // Two things are asserted separately in every arm, because they are two
    // different obligations that a single `is_err()` would conflate:
    //
    //   1. the SPECIFIC `ComposeError` variant, with the scalar identity fields
    //      an operator needs (`matches!`, never `is_err()`);
    //   2. that NO state escaped — the `Result` carries no `ComposedLineage` at
    //      all, whatever the engine's scratch state had already absorbed.
    //
    // (2) is the load-bearing one. `apply_delta`'s documented contract
    // (`src/domain/delta.rs`) is that a mid-delta failure leaves
    // `state.sequence_num` untouched but MAY leave every other state map
    // partially mutated; SR-3.5 holds only because the engine owns that scratch
    // state and hands back an owned value, so the half-applied map is dropped
    // rather than returned. #B10 pins that structurally, since a refactor to an
    // in-place `&mut ObjectState` API would violate SR-3.5 with every
    // behavioural arm below still green.
    // ════════════════════════════════════════════════════════════════════════

    // ── Lane B fixture vocabulary ───────────────────────────────────────────

    /// Wrap `delta_bytes` in an OMPF container carrying no Parts 1..N.
    fn ompf_of(delta_bytes: &[u8]) -> Bytes {
        Bytes::from(
            OmpfContainer::serialize(delta_bytes, &[]).expect("fixture container must serialize"),
        )
    }

    /// A verbatim record carrying `ompf` **verbatim**, at `sequence_num`.
    ///
    /// Deliberately NOT [`commit`] / [`commit_with_ext`]: those also advance the
    /// reference [`ObjectState`] through `apply_delta`, which is precisely what
    /// a poison record must not be able to do — three of the arms below carry
    /// bytes no live commit could have produced, and two of those would panic
    /// the fixture on the way in.
    fn poison_record(sequence_num: u64, ompf: Bytes) -> TailRecord {
        TailRecord::Verbatim {
            sequence_num,
            server_ts: server_ts_for(sequence_num),
            ompf,
        }
    }

    /// The canonical base plus a one-generation tail whose sole record carries
    /// `ompf`, composed through the bytes-in entry point.
    ///
    /// `from_seq == BASE_SNAP_SEQ`, so the record sits at `BASE_SNAP_SEQ + 1`
    /// and clears every link check — the failure that follows is the replay
    /// pipeline's, not a lineage-link rejection.
    fn compose_one_poison_record(ompf: Bytes) -> Result<ComposedLineage, ComposeError> {
        let (_history, base) = canonical();
        let artifact = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &[poison_record(BASE_SNAP_SEQ + 1, ompf)],
        );
        compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[artifact])
    }

    /// The 16-byte ascending pattern `apply.rs`'s garbage-OMPF arm uses. Its
    /// first four bytes read `0x03020100`, not OMPF's `0x46504d4f`.
    fn garbage_ompf() -> Bytes {
        Bytes::from((0..16u8).collect::<Vec<u8>>())
    }

    /// SR-3.5's no-partial-state half, asserted uniformly: the call returned
    /// `Err`, so the caller holds NO [`ComposedLineage`] — no state at any
    /// sequence between `base_snap_seq` and `S` is observable to it, however
    /// much of the run the engine's scratch state had already absorbed.
    ///
    /// Returns the error so the caller can go on to pin the variant.
    fn expect_no_state(result: Result<ComposedLineage, ComposeError>, what: &str) -> ComposeError {
        match result {
            Ok(composed) => panic!(
                "{what}: composition must fail, but it returned a state at seq {} carrying {} \
                 sections / {} fields / {} time samples — a partial state escaped to the caller \
                 (SR-3.5)",
                composed.composed_seq,
                composed.state.sections.len(),
                composed.state.fields.len(),
                composed.state.time_samples.len()
            ),
            Err(err) => err,
        }
    }

    /// SR-3.5 / SR-6.1's classification half: the failure reaches the step-down
    /// loop as [`CheckpointError::Permanent`], never `Transient`.
    ///
    /// Consumes the error, so call it after the variant has been pinned.
    fn assert_permanent(err: ComposeError, what: &str) {
        let display = err.to_string();
        let classified = CheckpointError::from(err);
        assert!(
            matches!(classified, CheckpointError::Permanent { .. }),
            "{what}: must classify Permanent so the caller steps down to the next older \
             restorable generation rather than retrying identical bytes; got {classified:?} \
             for `{display}`"
        );
    }

    /// A two-generation canonical chain whose SECOND generation's SECOND record
    /// is poisoned, plus the base blob.
    ///
    /// The shape three arms share, and the sharpest one available: the failing
    /// record is neither the first record nor in the first generation, so a
    /// `TailSite` / `RecordSite` hard-coded to zero cannot pass — and the three
    /// records before it have already been applied to the engine's scratch
    /// state when it trips, which is what makes the no-partial-state assertion
    /// non-vacuous.
    fn poisoned_mid_chain() -> (History, Bytes, Vec<Bytes>) {
        const SPLIT: u64 = 10;
        const POISON_SEQ: u64 = 12;
        let (history, base) = canonical();

        let mut second = history.records_in(SPLIT, POISON_SEQ);
        assert_eq!(second.len(), 2, "generation 2 must carry seq 11 and seq 12");
        assert_eq!(second[1].sequence_num(), POISON_SEQ);
        second[1] = poison_record(POISON_SEQ, garbage_ompf());

        let artifacts = vec![
            tail(
                BASE_SNAP_SEQ,
                BASE_SNAP_SEQ,
                &history.records_in(BASE_SNAP_SEQ, SPLIT),
            ),
            tail(BASE_SNAP_SEQ, SPLIT, &second),
        ];
        (history, base, artifacts)
    }

    /// The non-test slice of this module's own source, with `//`-comments
    /// stripped — the input both structural guards (#B10, #B11) scan.
    ///
    /// Stripping is required, not cosmetic: the module doc *names* several of
    /// the banned tokens in order to forbid them, so a naive substring scan
    /// would trip on the prose it is enforcing. Precedent and shape:
    /// `tail_codec.rs`'s determinism scan.
    fn non_test_source() -> String {
        let src = include_str!("compose.rs");
        let cut = src
            .find("#[cfg(test)]")
            .expect("the module must have a #[cfg(test)] block to cut at");
        let non_test = &src[..cut];
        assert!(
            !non_test.contains("/*"),
            "these guards strip line comments only; a block comment would slip banned tokens \
             past them"
        );
        non_test
            .lines()
            .map(|line| line.split("//").next().unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ── #B1 — a record whose OMPF container does not parse ──────────────────

    #[test]
    fn compose_tail_record_with_unparseable_ompf_returns_permanent() {
        let garbage = garbage_ompf();
        assert!(
            OmpfContainer::parse(&garbage).is_err(),
            "the fixture bytes must actually fail the container parse, or the arm is vacuous"
        );

        let what = "a record whose OMPF container does not parse";
        let err = expect_no_state(compose_one_poison_record(garbage), what);
        assert!(
            matches!(
                &err,
                ComposeError::OmpfParse {
                    site,
                    source: OmpfError::BadMagic { .. },
                } if site.tail.index == 0
                    && site.tail.to_seq == Some(BASE_SNAP_SEQ + 1)
                    && site.index == 0
                    && site.sequence_num == BASE_SNAP_SEQ + 1
            ),
            "expected OmpfParse(BadMagic) attributed to tail[0] record[0] seq={}, got {err:?}",
            BASE_SNAP_SEQ + 1
        );
        assert_permanent(err, what);
    }

    // ── #B2 — a record the OODF verifier rejects ────────────────────────────

    #[test]
    fn compose_tail_record_failing_the_oodf_verifier_returns_permanent() {
        // A real OODF buffer, truncated below its internal offsets. The
        // truncation keeps the buffer above `kMinFlatBufferHeader` (8 bytes)
        // and keeps `OODF` at [4..8], so it clears oodf.cpp's two pre-verifier
        // short-circuits (`TooShort`, `BadMagic`) and reaches the FlatBuffers
        // verifier itself — which is the layer this arm is about.
        let full = oodf_ffi::build_delta(&OwnedDelta {
            struct_commands: vec![create_section(0, 90, "verifier_root", 1)],
            set_fields: vec![inline_field(90, "verifier_key", 1, 0)],
            ..delta()
        })
        .expect("fixture delta must serialize");
        let truncated = &full[..full.len() / 2];
        assert!(
            truncated.len() >= 8,
            "the truncated buffer must clear the 8-byte pre-verifier size guard, got {}",
            truncated.len()
        );
        assert_eq!(
            &truncated[4..8],
            b"OODF",
            "the truncated buffer must keep the file identifier, or it short-circuits on BadMagic"
        );

        // Pin the exact arm observed, per `oodf-ffi/tests/malformed_input.rs`'s
        // no-hedging convention: accepting a SET of arms is the hedge that lets
        // a real reordering of oodf.cpp's guards pass silently.
        assert!(
            matches!(
                oodf_ffi::parse_delta(truncated),
                Err(OodfError::VerifierFailed(_))
            ),
            "the fixture must reach the verifier, not a pre-verifier guard"
        );

        let what = "a record the OODF verifier rejects";
        let err = expect_no_state(compose_one_poison_record(ompf_of(truncated)), what);
        assert!(
            matches!(
                &err,
                ComposeError::OodfParse {
                    site,
                    source: OodfError::VerifierFailed(_),
                } if site.tail == TailSite::known(0, BASE_SNAP_SEQ + 1)
                    && site.index == 0
                    && site.sequence_num == BASE_SNAP_SEQ + 1
            ),
            "expected OodfParse(VerifierFailed) at record[0] seq={}, got {err:?}",
            BASE_SNAP_SEQ + 1
        );
        assert_permanent(err, what);
    }

    // ── #B3 — a record `apply_delta` refuses ────────────────────────────────

    #[test]
    fn compose_tail_record_rejected_by_apply_delta_returns_permanent() {
        // A real refusal, not a synthetic seam: `set_order` exactly at
        // `MAX_DELTA_SET_ORDER` clears `validate_delta_bounds` (which rejects
        // strictly ABOVE the cap) and then trips `apply_set_field`'s 2^53 - 1
        // ceiling, because the composed base already carries a non-zero
        // `next_set_order`.
        let poison = OwnedDelta {
            set_fields: vec![OodfSetField {
                section_id: 10,
                key_name: "ceiling_key".to_string(),
                value_or_ext_hash: b"v".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: MAX_DELTA_SET_ORDER,
            }],
            ..delta()
        };
        let ompf = ompf_of(&oodf_ffi::build_delta(&poison).expect("poison delta must serialize"));

        // Non-vacuity: the record's bytes parse AND verify cleanly. The refusal
        // below is the domain engine's, not the decoder's — without this the
        // arm could pass on a malformed buffer and prove nothing about SR-3.5's
        // "`apply_delta` error on bytes that committed live" clause.
        let probe_site = RecordSite {
            tail: TailSite::known(0, BASE_SNAP_SEQ + 1),
            index: 0,
            sequence_num: BASE_SNAP_SEQ + 1,
        };
        let (parsed, ext) =
            decode_tail_record_ompf(probe_site, &ompf).expect("the poison record must decode");
        assert!(parsed.is_some() && ext.is_empty());

        let what = "a record apply_delta refuses";
        let err = expect_no_state(compose_one_poison_record(ompf), what);
        assert!(
            matches!(
                &err,
                ComposeError::ApplyDelta {
                    site,
                    source: DeltaError::SetOrderOverflow { delta, .. },
                } if *delta == MAX_DELTA_SET_ORDER
                    && site.tail == TailSite::known(0, BASE_SNAP_SEQ + 1)
                    && site.index == 0
                    && site.sequence_num == BASE_SNAP_SEQ + 1
            ),
            "expected ApplyDelta(SetOrderOverflow) at seq={}, got {err:?}",
            BASE_SNAP_SEQ + 1
        );
        assert_permanent(err, what);
    }

    // ── #B4 — the legal sequence maximum composes ───────────────────────────

    #[test]
    fn compose_of_a_tail_ending_at_the_sequence_ceiling_succeeds() {
        // `apply_delta` trips on the PRE-apply cursor (`sequence_num >= 2^53 -
        // 1`), so a commit that LANDS on `TAIL_MAX_SEQUENCE` is legal and must
        // compose. This is the half of the boundary that an off-by-one would
        // silently refuse, which is why it is pinned alongside #B5 rather than
        // left implicit.
        let (history, base) = canonical();
        let TailRecord::Verbatim { ompf, .. } = history.records[6].clone() else {
            panic!("the seq-7 commit must be a verbatim record");
        };
        let record = poison_record(TAIL_MAX_SEQUENCE, ompf);
        let artifact = tail(TAIL_MAX_SEQUENCE - 1, TAIL_MAX_SEQUENCE - 1, &[record]);

        let composed = compose_from_artifacts(OBJECT_ID, &base, TAIL_MAX_SEQUENCE - 1, &[artifact])
            .expect("a tail ending exactly at the sequence ceiling must compose");

        assert_eq!(composed.composed_seq, TAIL_MAX_SEQUENCE);
        assert_eq!(composed.state.sequence_num, TAIL_MAX_SEQUENCE);
        // The record really replayed at the ceiling — an engine that refused it
        // and returned the bare base would still report the cursor above.
        assert_eq!(
            composed.state.fields[&(14, "tail_key_7".to_string())].version,
            TAIL_MAX_SEQUENCE,
            "the ceiling commit's writes carry the ceiling as their version"
        );
        assert_eq!(
            composed.state.sections[&14].first_version,
            TAIL_MAX_SEQUENCE
        );
    }

    // ── #B5 — past the ceiling is refused, and no state escapes ─────────────

    #[test]
    fn compose_with_the_cursor_at_the_sequence_ceiling_returns_permanent() {
        // Where the ceiling is actually enforced, and why. The engine has no
        // ceiling comparison of its own: `apply_delta`'s pre-apply guard is
        // UNREACHABLE from composition, because the cursor is seeded from
        // `base_snap_seq` and every record must be `cursor + 1`, while both
        // `encode_tail_envelope` and `read_tail_header` refuse any header whose
        // `to_seq` exceeds `TAIL_MAX_SEQUENCE`. So "a base at the ceiling plus
        // any record" is not a state the composition surface can be driven
        // into — it is refused one layer down, and this arm pins that refusal
        // on both the encode and the decode side.
        let (history, base) = canonical();
        let TailRecord::Verbatim { ompf, .. } = history.records[6].clone() else {
            panic!("the seq-7 commit must be a verbatim record");
        };

        // (a) The encoder will not build it: a base at the ceiling with one
        // record would have to span to `TAIL_MAX_SEQUENCE + 1`.
        let over = TailHeader {
            object_id: OBJECT_ID,
            base_snap_seq: TAIL_MAX_SEQUENCE,
            from_seq: TAIL_MAX_SEQUENCE,
            to_seq: TAIL_MAX_SEQUENCE + 1,
        };
        assert_eq!(
            encode_tail_envelope(
                &over,
                vec![poison_record(TAIL_MAX_SEQUENCE + 1, ompf.clone())]
            )
            .expect_err("a tail carrying a record past the ceiling must not encode"),
            TailCodecError::SequenceCeilingExceeded {
                sequence_num: TAIL_MAX_SEQUENCE + 1,
                max: TAIL_MAX_SEQUENCE,
            }
        );

        // (b) A forged artifact that claims it anyway is refused at the
        // composition entry point, as `Permanent`, with nothing returned.
        //
        // `to_seq`'s header offset is not exported (only the format-version and
        // record-count offsets are), so it is asserted against the real
        // artifact before patching rather than trusted.
        const TO_SEQ_OFFSET: usize = 32;
        let mut forged = tail(
            TAIL_MAX_SEQUENCE - 1,
            TAIL_MAX_SEQUENCE - 1,
            &[poison_record(TAIL_MAX_SEQUENCE, ompf)],
        )
        .to_vec();
        assert_eq!(
            u64::from_le_bytes(
                forged[TO_SEQ_OFFSET..TO_SEQ_OFFSET + 8]
                    .try_into()
                    .expect("8 bytes")
            ),
            TAIL_MAX_SEQUENCE,
            "the header's to_seq must sit at this offset — re-derive it if the layout moves"
        );
        forged[TO_SEQ_OFFSET..TO_SEQ_OFFSET + 8]
            .copy_from_slice(&(TAIL_MAX_SEQUENCE + 1).to_le_bytes());

        let what = "an artifact claiming a sequence past the ceiling";
        let err = expect_no_state(
            compose_from_artifacts(
                OBJECT_ID,
                &base,
                TAIL_MAX_SEQUENCE - 1,
                &[Bytes::from(forged)],
            ),
            what,
        );
        assert!(
            matches!(
                &err,
                ComposeError::TailDecode {
                    site,
                    source: TailCodecError::SequenceCeilingExceeded {
                        sequence_num,
                        max,
                    },
                } if site.index == 0
                    && *sequence_num == TAIL_MAX_SEQUENCE + 1
                    && *max == TAIL_MAX_SEQUENCE
            ),
            "expected TailDecode(SequenceCeilingExceeded) at tail[0], got {err:?}"
        );
        assert_permanent(err, what);
    }

    // ── #B6 — an unknown tail format_version (SR-6.1) ───────────────────────

    #[test]
    fn compose_unknown_tail_format_version_returns_permanent() {
        // WHICH SHAPE THE ENGINEER LANDED, for the record: the composition
        // module owns BOTH entry points. `compose_lineage` consumes
        // already-decoded `TailEnvelope`s, but `compose_from_artifacts` is
        // bytes-in and owns envelope decode itself — so SR-6.1's rejection is
        // asserted on THIS side of the Task 2 seam, driving the real decode
        // rather than the `From<TailCodecError>` impl in isolation.
        let (_history, base) = canonical();
        let mut forged = tail(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &[]).to_vec();
        assert_eq!(
            u32::from_le_bytes(
                forged[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
                    .try_into()
                    .expect("4 bytes")
            ),
            TAIL_FORMAT_VERSION,
            "the artifact must start at the supported version, or the patch below proves nothing"
        );
        forged[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
            .copy_from_slice(&(TAIL_FORMAT_VERSION + 1).to_le_bytes());

        let what = "an artifact at an unknown format_version";
        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[Bytes::from(forged)]),
            what,
        );
        assert!(
            matches!(
                &err,
                ComposeError::TailDecode {
                    site,
                    source: TailCodecError::UnknownFormatVersion { got, supported },
                } if site.index == 0
                    && site.to_seq.is_none()
                    && *got == TAIL_FORMAT_VERSION + 1
                    && *supported == TAIL_FORMAT_VERSION
            ),
            "expected TailDecode(UnknownFormatVersion) index-attributed at tail[0], got {err:?}"
        );
        // `to_seq` is deliberately unknown here: the codec validates the version
        // BEFORE reading `to_seq`, because a v2 header may not have this
        // layout at all. Guessing at it would be exactly the "never guessed at"
        // SR-6.1 forbids, so index attribution is the contract, not a gap.
        assert_permanent(err, what);
    }

    // ── #B7 — a base blob that does not decode ──────────────────────────────

    #[test]
    fn compose_base_decode_failure_returns_permanent_identifying_the_base_generation() {
        // The tail is healthy in the first case and poisoned in the second, so
        // the arm also pins the ORDER: the base is decoded before any record
        // replays, and a base failure is reported as the base's even when a
        // record downstream would also have failed.
        let (history, _base) = canonical();
        let healthy = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &history.records_in(BASE_SNAP_SEQ, HEAD_SEQ),
        );

        // (a) Garbage in the base slot.
        let what = "garbage in the base slot";
        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &garbage_ompf(), BASE_SNAP_SEQ, &[healthy]),
            what,
        );
        assert!(
            matches!(
                &err,
                ComposeError::BaseDecode { base_snap_seq, .. }
                    if *base_snap_seq == BASE_SNAP_SEQ
            ),
            "expected BaseDecode naming base_snap_seq={BASE_SNAP_SEQ}, got {err:?}"
        );
        assert_permanent(err, what);

        // (b) A `LETL` TAIL envelope handed to the base slot — the realistic
        // mix-up, and the one a bare `is_err()` would let through as
        // indistinguishable from (a).
        let letl = tail(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &[]);
        assert_eq!(
            &letl[..TAIL_ENVELOPE_MAGIC.len()],
            &TAIL_ENVELOPE_MAGIC,
            "the fixture must really be a tail envelope"
        );
        let poisoned = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &[poison_record(BASE_SNAP_SEQ + 1, garbage_ompf())],
        );
        let what = "a tail envelope in the base slot";
        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &letl, BASE_SNAP_SEQ, &[poisoned]),
            what,
        );
        assert!(
            matches!(
                &err,
                ComposeError::BaseDecode { base_snap_seq, .. }
                    if *base_snap_seq == BASE_SNAP_SEQ
            ),
            "a base failure must be reported as the base's, even with a poisoned record \
             downstream — got {err:?}"
        );
        assert_permanent(err, what);
    }

    // ── A chunked (LEBC) base ────────────────────────────────────────────────
    //
    // A layer over one delta's entry limit is backed up as a chunked base
    // (`checkpoint::chunked_base`), and composition reads its base only
    // through `decode_checkpoint_ompf`. The fixture base here is chunked at a
    // per-chunk limit of 2 entries, so the canonical base state spans several
    // chunks without being anywhere near the real limit.

    /// Tails replayed over a chunked base compose to exactly the state, the
    /// sequence and the external-value list they compose to over the unchunked
    /// base of the same layer — and that state is live history's at `S`.
    #[test]
    fn compose_over_a_chunked_base_equals_compose_over_the_unchunked_base() {
        use crate::checkpoint::chunked_base::is_chunked_base;
        use crate::checkpoint::chunked_base::tests::chunked_base_of;

        let (history, unchunked) = canonical();
        let chunked = Bytes::from(chunked_base_of(history.state_at(BASE_SNAP_SEQ), 2));
        assert!(
            !is_chunked_base(&unchunked),
            "the comparison base is the plain OMPF"
        );
        let artifact = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &history.records_in(BASE_SNAP_SEQ, HEAD_SEQ),
        );

        let over_unchunked = compose_from_artifacts(
            OBJECT_ID,
            &unchunked,
            BASE_SNAP_SEQ,
            std::slice::from_ref(&artifact),
        )
        .expect("the canonical lineage must compose");
        let over_chunked = compose_from_artifacts(OBJECT_ID, &chunked, BASE_SNAP_SEQ, &[artifact])
            .expect("tails over a chunked base must compose");

        assert_eq!(over_chunked.composed_seq, HEAD_SEQ);
        assert_eq!(
            over_chunked.composed_seq, over_unchunked.composed_seq,
            "composed_seq"
        );
        assert_eq!(
            over_chunked.state, over_unchunked.state,
            "the composed state, every column (version stamps, set-order cursor and children \
             lists included)"
        );
        assert_eq!(
            over_chunked.ext_values, over_unchunked.ext_values,
            "the external values the replayed tails carry"
        );
        assert_faithful_columns(&over_chunked.state, history.state_at(HEAD_SEQ));
    }

    /// A chunked base whose framing is malformed is a base decode failure,
    /// attributed to the base generation and `Permanent`, exactly like a
    /// malformed OMPF base (#B7) — its typed refusal travels inside.
    #[test]
    fn compose_over_a_malformed_chunked_base_returns_permanent_identifying_the_base_generation() {
        use crate::checkpoint::chunked_base::tests::chunked_base_of;
        use crate::checkpoint::chunked_base::ChunkedBaseError;

        let (history, _base) = canonical();
        let mut malformed = chunked_base_of(history.state_at(BASE_SNAP_SEQ), 2);
        malformed.push(0);
        let healthy = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &history.records_in(BASE_SNAP_SEQ, HEAD_SEQ),
        );

        let what = "a chunked base with a trailing byte";
        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &malformed, BASE_SNAP_SEQ, &[healthy]),
            what,
        );
        match &err {
            ComposeError::BaseDecode {
                base_snap_seq,
                source: CheckpointError::Permanent { source },
            } => {
                assert_eq!(*base_snap_seq, BASE_SNAP_SEQ);
                assert_eq!(
                    source.downcast_ref::<ChunkedBaseError>(),
                    Some(&ChunkedBaseError::TrailingBytes { extra: 1 }),
                    "the chunked decoder's typed refusal, not an OMPF parse error"
                );
            }
            other => panic!("expected BaseDecode(Permanent(TrailingBytes)), got {other:?}"),
        }
        assert_permanent(err, what);
    }

    // ── #B8 — every failure names the generation it belongs to ──────────────

    #[test]
    fn compose_failure_identifies_the_offending_generation() {
        // SR-3.1's step-down has to be able to say WHICH generation it is
        // stepping down from, and an operator has to be able to read it out of
        // one log line without re-running with verbose logging. So the
        // assertions here go through `Display` — the error's public surface —
        // exactly as `apply.rs`'s `msg.contains("is_diff=false")` precedent
        // does, and never through the variants' fields.
        let (_history, base) = canonical();

        // (a) The base generation, named by its `snap_seq`.
        let base_msg = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &garbage_ompf(), BASE_SNAP_SEQ, &[]),
            "a base decode failure",
        )
        .to_string();
        assert!(
            base_msg.contains(&format!("base@{BASE_SNAP_SEQ}")),
            "a base failure must name the base generation, got: {base_msg}"
        );
        assert!(
            base_msg.contains("OMPF") || base_msg.contains("ompf"),
            "and must name the layer that rejected it, got: {base_msg}"
        );

        // (b) A replay failure in the SECOND generation, at its SECOND record —
        // so neither index can be a hard-coded zero and still pass.
        let (_history, base2, artifacts) = poisoned_mid_chain();
        let record_msg = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base2, BASE_SNAP_SEQ, &artifacts),
            "a mid-chain replay failure",
        )
        .to_string();
        for fragment in ["tail[1]", "to_seq=12", "record[1]", "seq=12"] {
            assert!(
                record_msg.contains(fragment),
                "a replay failure must name `{fragment}` so step-down can attribute it, \
                 got: {record_msg}"
            );
        }

        // (c) A decode failure whose generation is index-attributed only. The
        // `to_seq=unknown` rendering is the contract (see #B6), so it is pinned
        // rather than left to look like a formatting accident.
        let mut forged = tail(BASE_SNAP_SEQ, BASE_SNAP_SEQ, &[]).to_vec();
        forged[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
            .copy_from_slice(&(TAIL_FORMAT_VERSION + 1).to_le_bytes());
        let decode_msg = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[Bytes::from(forged)]),
            "an unknown-format_version artifact",
        )
        .to_string();
        assert!(
            decode_msg.contains("tail[0]") && decode_msg.contains("to_seq=unknown"),
            "an index-attributed decode failure must say so explicitly, got: {decode_msg}"
        );
    }

    // ── #B9 — no partial state escapes, in any failure mode ─────────────────

    #[test]
    fn compose_failure_returns_err_and_never_a_partial_state() {
        // The sharp case first: a poison record at the END of a two-generation
        // chain. Three records have already been replayed into the engine's
        // scratch state when it trips, and `apply_delta`'s contract says the
        // fourth may have partially mutated it further. SR-3.5 says the caller
        // sees none of it.
        let (history, base, artifacts) = poisoned_mid_chain();

        // Non-vacuity: the legal prefix of that very chain composes to a real,
        // non-empty state — so the `Err` below is the poison record's doing and
        // not some setup mistake that would have failed anyway.
        let prefix = compose_canonical(&history, &base, 11);
        assert_eq!(prefix.composed_seq, 11);
        assert!(
            !prefix.state.sections.is_empty() && !prefix.state.fields.is_empty(),
            "the legal prefix must materialize real content, or this arm proves nothing"
        );

        expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &artifacts),
            "a poison record after three successful replays",
        );

        // And uniformly, across every failure class this lane covers: each one
        // returns `Err`, so no `ComposedLineage` — partial or otherwise — is
        // reachable through the entry point.
        let verifier_bytes = {
            let full = oodf_ffi::build_delta(&OwnedDelta {
                struct_commands: vec![create_section(0, 90, "verifier_root", 1)],
                ..delta()
            })
            .expect("fixture delta must serialize");
            ompf_of(&full[..full.len() / 2])
        };
        let apply_bytes = ompf_of(
            &oodf_ffi::build_delta(&OwnedDelta {
                set_fields: vec![OodfSetField {
                    section_id: 10,
                    key_name: "ceiling_key".to_string(),
                    value_or_ext_hash: b"v".to_vec(),
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: MAX_DELTA_SET_ORDER,
                }],
                ..delta()
            })
            .expect("poison delta must serialize"),
        );
        for (what, ompf) in [
            ("unparseable OMPF", garbage_ompf()),
            ("OODF verifier rejection", verifier_bytes),
            ("apply_delta refusal", apply_bytes),
        ] {
            let err = expect_no_state(compose_one_poison_record(ompf), what);
            assert_permanent(err, what);
        }

        // The lineage-link and decode classes too, so "every failure mode" is
        // the whole enum's shape and not just the replay pipeline's.
        let gapped = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ + 2,
            &history.records_in(BASE_SNAP_SEQ + 2, HEAD_SEQ),
        );
        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[gapped]),
            "a tail that skips commits",
        );
        assert!(
            matches!(
                &err,
                ComposeError::LineageGap {
                    site,
                    cursor,
                    from_seq,
                } if *site == TailSite::known(0, HEAD_SEQ)
                    && *cursor == BASE_SNAP_SEQ
                    && *from_seq == BASE_SNAP_SEQ + 2
            ),
            "expected LineageGap(tail[0] to_seq=14, cursor=6, from_seq=8), got {err:?}"
        );
        assert_permanent(err, "a tail that skips commits");

        let err = expect_no_state(
            compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[garbage_ompf()]),
            "an artifact that is not a tail envelope at all",
        );
        assert!(
            matches!(
                &err,
                ComposeError::TailDecode {
                    source: TailCodecError::BadMagic { .. },
                    ..
                }
            ),
            "expected TailDecode(BadMagic), got {err:?}"
        );
        assert_permanent(err, "an artifact that is not a tail envelope at all");
    }

    // ── #B10 — the entry points return state by value (structural) ──────────

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn compose_entry_point_returns_state_by_value_and_takes_no_out_parameter() {
        // SR-3.5's no-partial-state guarantee is a STRUCTURAL property: it holds
        // because the engine owns its scratch `ObjectState` and hands back an
        // owned value, so `apply_delta`'s partially-mutated maps are dropped
        // rather than written through a caller's reference. A refactor to an
        // in-place `&mut ObjectState` API would violate SR-3.5 with every
        // behavioural arm above still green, which is why this guard exists at
        // all.
        let code = non_test_source();

        // Anti-vacuity FIRST: a scan that stopped matching real code (a renamed
        // entry point, a comment stripper that ate too much) must fail here
        // rather than pass empty below.
        for anchor in [
            "pub fn compose_lineage(",
            "pub fn compose_from_artifacts(",
            "pub struct ComposedLineage",
            "pub state: ObjectState,",
            "let mut state = decode_checkpoint_ompf(",
        ] {
            assert!(
                code.contains(anchor),
                "the scan lost sight of `{anchor}`; it would otherwise pass vacuously"
            );
        }
        assert_eq!(
            code.matches("-> Result<ComposedLineage, ComposeError>")
                .count(),
            2,
            "both entry points must return the composed lineage BY VALUE in the Ok arm; a \
             third such signature (or a missing one) means this guard needs re-deriving"
        );

        assert!(
            !code.contains("&mut ObjectState"),
            "no composition entry point may take an `&mut ObjectState` out-parameter: SR-3.5's \
             no-partial-state guarantee rests on the engine owning its scratch state and \
             returning it by value (`src/domain/delta.rs` — on `Err(_)` every field except \
             `sequence_num` may be partially mutated and the caller MUST discard)"
        );
        assert!(
            !code.contains("pub state: &"),
            "`ComposedLineage::state` must be an owned `ObjectState`, not a borrow of one the \
             caller supplied"
        );
    }

    // ── #B11 — the module does no I/O (structural) ──────────────────────────

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn compose_module_does_no_io() {
        // The Task's cross-task contract for this module is "pure in-memory: no
        // Valkey I/O, no adapter I/O, no async obligations", and the module
        // header additionally promises no clock. All of that is reviewable by
        // inspection, so it is enforced by inspection — the same banned-token
        // shape `tail_codec.rs`'s determinism scan uses.
        const BANNED: &[(&str, &str)] = &[
            (
                "fred",
                "no Valkey client: composition consumes fetched bytes, never a connection",
            ),
            (
                "CheckpointStore",
                "no adapter I/O: the caller fetches artifacts, the engine only replays them",
            ),
            (
                ".await",
                "no async obligations: every entry point is synchronous by contract",
            ),
            ("async fn", "same — no async surface may appear here"),
            (
                "SystemTime",
                "no clock: composition must be a pure function of its input bytes",
            ),
            ("Instant", "no clock; see above"),
            ("now_unix_ms", "no clock; see above"),
        ];

        let code = non_test_source();
        for (token, reason) in BANNED {
            assert!(
                !code.contains(token),
                "compose.rs must not reference `{token}` outside its test module ({reason}). \
                 Purity is what makes this module unit-testable without a backing store, and \
                 it is why every failure it reports classifies `Permanent`."
            );
        }

        // Sanity: the comment stripper must not have eaten the code the scan is
        // meant to cover, or every assertion above passed on an empty string.
        for anchor in [
            "pub fn compose_lineage(",
            "pub(crate) fn decode_tail_record_ompf(",
            "apply_delta_into(&mut state, owned",
        ] {
            assert!(
                code.contains(anchor),
                "comment stripping removed real code (`{anchor}` is gone); the scan would be \
                 vacuous"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Lane C — the decoder boundary, seen from the tail side.
    //
    // `apply.rs`'s reset-shape guard rejects every Part-0 delta that is not
    // `is_diff == true && base_version == 0`. That is correct for a BASE blob
    // and would be catastrophic applied to a tail record: the ordinary
    // `UpdateLayer` commit is `is_diff == false` (#C1) and an incremental diff
    // is `is_diff == true` with a non-zero `base_version` (#C2) — both are the
    // normal case inside a tail. The boundary is therefore two-sided, and its
    // base half lives in `apply.rs`
    // (`decode_checkpoint_ompf_rejects_a_tail_envelope_as_permanent`, plus the
    // two reset-shape rejection arms this lane left untouched).
    //
    // #C1 and #C2 each hand the SAME bytes to BOTH decoders inside one test, so
    // neither can pass unless the two paths genuinely disagree: base decoder →
    // `Permanent`, composition → composes. #C3 is the structural companion for
    // the paths they do not reach.
    // ════════════════════════════════════════════════════════════════════════

    /// Compose a one-record tail carrying `delta` onto the canonical base —
    /// **and** the base decoder's verdict on that record's very own OMPF bytes.
    ///
    /// Returning both is the point. An arm that only showed composition
    /// succeeding would still pass in a world where the reset-shape guard never
    /// applied to those bytes in the first place, proving nothing about the
    /// boundary; pairing the two verdicts makes the disagreement the assertion.
    ///
    /// [`compose_one_poison_record`] is shape-neutral despite its name — it
    /// composes whatever single record it is handed, and the records handed to
    /// it here are well-formed.
    fn compose_one_record_and_base_verdict(
        delta: &OwnedDelta,
    ) -> (
        Result<ComposedLineage, ComposeError>,
        Result<ObjectState, CheckpointError>,
    ) {
        let ompf = ompf_of(&oodf_ffi::build_delta(delta).expect("fixture delta must serialize"));
        let base_verdict = decode_checkpoint_ompf(OBJECT_ID, &ompf);
        (compose_one_poison_record(ompf), base_verdict)
    }

    // ── #C1 — the ordinary `UpdateLayer` shape (`is_diff == false`) ─────────

    #[test]
    fn compose_tail_record_with_is_diff_false_composes() {
        let ordinary = OwnedDelta {
            struct_commands: vec![create_section(0, 80, "lane_c_root", 1)],
            set_fields: vec![inline_field(80, "lane_c_key", BASE_SNAP_SEQ + 1, 0)],
            ..delta()
        };
        assert!(
            !ordinary.is_diff && ordinary.base_version == 0,
            "the fixture must carry the ordinary UpdateLayer shape, or the arm proves nothing"
        );

        let (composed, base_verdict) = compose_one_record_and_base_verdict(&ordinary);

        // The base decoder refuses these exact bytes...
        let base_err = base_verdict.expect_err(
            "the reset-shape guard must still reject an is_diff=false delta on the BASE path — \
             if this stops failing, the guard was weakened, not the tail path fixed",
        );
        assert!(
            matches!(base_err, CheckpointError::Permanent { .. }),
            "the base-path rejection must be Permanent; got {base_err:?}"
        );
        assert!(
            base_err.to_string().contains("is_diff=false"),
            "the base path must reject on the is_diff knob specifically, got: {base_err}"
        );

        // ...and the tail path replays them unfiltered.
        let composed = composed.expect(
            "an ordinary is_diff=false commit is the NORMAL case inside a tail and must compose; \
             a failure here means the tail path inherited the base decoder's reset-shape guard",
        );
        assert_eq!(
            composed.composed_seq,
            BASE_SNAP_SEQ + 1,
            "the record advanced the cursor by exactly one"
        );
        assert_eq!(
            composed.state.sections[&80].name, "lane_c_root",
            "the record's own content must land"
        );
        assert!(composed
            .state
            .fields
            .contains_key(&(80, "lane_c_key".to_string())));
        // `is_diff = false` is not a reset: the base era survives underneath it.
        assert!(
            composed.state.sections.contains_key(&10),
            "an is_diff=false record must not wipe base-era state"
        );
    }

    // ── #C2 — the incremental-diff shape (`base_version != 0`) ──────────────

    #[test]
    fn compose_tail_record_with_a_nonzero_base_version_composes() {
        // `apply_delta`'s reset branch requires BOTH `is_diff` and
        // `base_version == 0` (`src/domain/delta.rs`, step 2), so this shape
        // replays as an ordinary command batch — exactly what the live commit
        // did with it.
        let incremental = OwnedDelta {
            is_diff: true,
            base_version: BASE_SNAP_SEQ,
            struct_commands: vec![create_section(0, 81, "lane_c_diff_root", 2)],
            set_fields: vec![inline_field(81, "lane_c_diff_key", BASE_SNAP_SEQ + 1, 0)],
            ..OwnedDelta::default()
        };
        assert!(
            incremental.is_diff && incremental.base_version != 0,
            "the fixture must carry the incremental-diff shape, or the arm proves nothing"
        );

        let (composed, base_verdict) = compose_one_record_and_base_verdict(&incremental);

        let base_err = base_verdict.expect_err(
            "the reset-shape guard must still reject a non-zero base_version on the BASE path",
        );
        assert!(
            matches!(base_err, CheckpointError::Permanent { .. }),
            "the base-path rejection must be Permanent; got {base_err:?}"
        );
        assert!(
            base_err
                .to_string()
                .contains(&format!("base_version={BASE_SNAP_SEQ}")),
            "the base path must reject on the base_version knob specifically, got: {base_err}"
        );

        let composed = composed.expect(
            "an incremental diff is the NORMAL case inside a tail and must compose; a failure \
             here means the tail path inherited the base decoder's reset-shape guard",
        );
        assert_eq!(composed.composed_seq, BASE_SNAP_SEQ + 1);
        assert_eq!(
            composed.state.sections[&81].name, "lane_c_diff_root",
            "the record's own content must land"
        );

        // The sharp half, from the other direction: had these bytes been read
        // as a RESET — the only shape the base decoder accepts — `apply_reset`
        // would have wiped the base era before replaying the commands. Every
        // base-era section is still here, so no reset ran.
        let (history, _base) = canonical();
        for id in history.state_at(BASE_SNAP_SEQ).sections.keys() {
            assert!(
                composed.state.sections.contains_key(id),
                "base-era section {id} was wiped: the record replayed as a reset"
            );
        }
        assert_eq!(
            composed.state.source_format, SOURCE_FORMAT,
            "no reset ran, so the base blob's source_format still stands"
        );
    }

    // ── #C3 — tail records never reach the base decoder (structural) ────────

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn compose_does_not_route_tail_records_through_the_base_reset_decoder() {
        // #C1 / #C2 cover the reachable case behaviourally. This catches the
        // refactor that reintroduces the coupling on a path they do not
        // exercise — a "validate every record with the checkpoint decoder
        // first" pass added ahead of, or alongside, the replay loop.
        let code = non_test_source();

        // Anti-vacuity FIRST, per #B10's convention: a scan that lost sight of
        // the code it is about must fail here rather than pass empty below.
        let base_decode = code
            .find("let mut state = decode_checkpoint_ompf(")
            .expect("the scan lost sight of the base decode; it would otherwise pass vacuously");
        let record_loop = code
            .find("for (record_index, record) in")
            .expect("the scan lost sight of the record-replay loop");
        let tail_decode = code
            .find("decode_tail_record_ompf(record_site, ompf)")
            .expect("the scan lost sight of the tail-record decode call");

        assert_eq!(
            code.matches("decode_checkpoint_ompf(").count(),
            1,
            "`decode_checkpoint_ompf` must be called EXACTLY once — the base decode. Its \
             reset-shape guard (`src/checkpoint/apply.rs`) rejects `is_diff == false` and \
             `base_version != 0`, and both are the normal case inside a tail (#C1 / #C2)"
        );
        assert!(
            base_decode < record_loop,
            "the single base-decode call must sit ahead of the record-replay loop, never inside it"
        );
        assert!(
            record_loop < tail_decode,
            "records must be decoded by `decode_tail_record_ompf`, inside the replay loop"
        );
    }

    // ── #C4 — a verbatim record whose Part 0 is empty ───────────────────────

    #[test]
    fn compose_verbatim_record_with_empty_delta_bytes_replays_as_an_empty_commit() {
        // [`decode_tail_record_ompf`] step 2. A container whose Part 0 is empty
        // is what the live path's `parsed == None` branch saw, and the obvious
        // wrong implementation — `parse_delta(container.delta_bytes)`
        // unconditionally — returns `OodfError::EmptyInput` and marks a
        // perfectly HEALTHY lineage unrestorable. Nothing pinned that: the
        // §8.4 empty commit is captured as `TailRecord::EmptyDelta`, which
        // never enters this function at all.
        //
        // Parts 1..N are present on purpose. Live, `store_external_values`
        // short-circuits on `parsed == None` before `external_values` is read,
        // so such a record stored nothing; collecting them here would
        // over-store relative to live history.
        let ompf = Bytes::from(
            OmpfContainer::serialize(&[], &[blob_b().as_slice()])
                .expect("a container with an empty Part 0 and one ext part must serialize"),
        );

        // Non-vacuity: the fixture really is "empty delta, one ext part".
        let container = OmpfContainer::parse(&ompf).expect("the fixture container must parse");
        assert!(
            container.delta_bytes.is_empty(),
            "the fixture's Part 0 must be empty, or the arm tests nothing"
        );
        assert_eq!(
            container.external_values.len(),
            1,
            "the fixture must carry a Part 1, or the over-store half of the arm is vacuous"
        );

        // ONE base blob drives both sides — the cumulative `set_order` cursor is
        // re-derived on apply, not carried in the blob (the column the module
        // header lists as re-derived), so pinning a single blob keeps the two
        // sides comparing the same cursor by construction rather than by
        // argument. (Two `canonical()` calls would in fact agree today:
        // `plan_full_content` emits fields in a total order over state content,
        // so a second assembly of the same state is byte-identical — b.gcb /
        // b.vz7. That is a property of the assembler this test does not need to
        // depend on, and one blob is the cheaper construction either way.)
        let (_history, base) = canonical();
        let artifact = tail(
            BASE_SNAP_SEQ,
            BASE_SNAP_SEQ,
            &[poison_record(BASE_SNAP_SEQ + 1, ompf)],
        );
        let composed = compose_from_artifacts(OBJECT_ID, &base, BASE_SNAP_SEQ, &[artifact])
            .expect("an empty-Part-0 record must replay as an empty commit, not fail the lineage");

        assert_eq!(
            composed.composed_seq,
            BASE_SNAP_SEQ + 1,
            "the record advanced the cursor without changing content"
        );
        assert!(
            composed.ext_values.is_empty(),
            "a record whose delta never parsed stored nothing live, so composition must surface \
             no ext parts for it — even though its container carries one"
        );

        // Content-identical to the same lineage with no records at all.
        let empty_lineage = compose_lineage(OBJECT_ID, &base, BASE_SNAP_SEQ, &[])
            .expect("the zero-tail lineage must compose");
        assert_content_identical(
            &composed.state,
            &empty_lineage.state,
            "an empty-Part-0 record must leave content untouched",
        );
    }

    // ── #C5 — a record whose `ext_value_index` does not resolve ─────────────

    #[test]
    fn compose_tail_record_with_an_unresolvable_ext_index_returns_permanent() {
        // [`decode_tail_record_ompf`] step 4, and the one `ComposeError` arm
        // lane B's failure group does not reach. It is also the arm that proves
        // the ordering claim in that function's doc: collection runs BEFORE any
        // state mutation, so a bad index cannot leave half a record applied.
        let poison = OwnedDelta {
            set_fields: vec![ext_field(10, "dangling_ext", HASH_C, 4, 9, 0)],
            ..delta()
        };
        // One part on the container, index 9 on the field.
        let ompf = Bytes::from(
            OmpfContainer::serialize(
                &oodf_ffi::build_delta(&poison).expect("poison delta must serialize"),
                &[blob_c().as_slice()],
            )
            .expect("fixture container must serialize"),
        );

        let what = "a record whose ext_value_index does not resolve";
        let err = expect_no_state(compose_one_poison_record(ompf), what);
        assert!(
            matches!(
                &err,
                ComposeError::ExtValueIndex {
                    site,
                    source: ServiceError::InvalidArgument(msg),
                } if site.tail == TailSite::known(0, BASE_SNAP_SEQ + 1)
                    && site.index == 0
                    && site.sequence_num == BASE_SNAP_SEQ + 1
                    && msg == "ext_value_index 9 out of range (OMPF has 1 external parts)"
            ),
            "expected ExtValueIndex(InvalidArgument) at record[0] seq={}, carrying the live \
             collector's message verbatim, got {err:?}",
            BASE_SNAP_SEQ + 1
        );
        assert_permanent(err, what);
    }

    // ── G3 — a stored record carrying a wrong-length external hash ──────────

    #[test]
    fn compose_tail_record_with_a_wrong_length_external_hash_returns_permanent() {
        // A fast-path commit made before the ingest-time hash-length check
        // could land a record whose external hash is not 32 bytes. Such a
        // record was never reproducible by complete-mode `apply_delta`, so it
        // was permanent before the collector stopped padding; it must stay
        // permanent now that the collector refuses it — and must not panic,
        // which a `try_into().unwrap()`-style collector would. The index
        // resolves (one part, index 1), so only the hash length can refuse it.
        let poisons = [
            (
                "a SetField with a 31-byte external hash",
                31,
                OwnedDelta {
                    set_fields: vec![OodfSetField {
                        value_or_ext_hash: vec![0xC3; 31],
                        ..ext_field(10, "short_hash", HASH_C, 4, 1, 0)
                    }],
                    ..delta()
                },
            ),
            (
                "a TimeSample with a 33-byte external hash",
                33,
                OwnedDelta {
                    time_samples: vec![OodfTimeSample {
                        section_id: 10,
                        time: 7.5,
                        value_or_ext_hash: vec![0xC3; 33],
                        ext_value_size: 4,
                        ext_value_index: 1,
                    }],
                    ..delta()
                },
            ),
        ];

        for (what, len, poison) in poisons {
            let ompf = Bytes::from(
                OmpfContainer::serialize(
                    &oodf_ffi::build_delta(&poison).expect("poison delta must serialize"),
                    &[blob_c().as_slice()],
                )
                .expect("fixture container must serialize"),
            );

            let err = expect_no_state(compose_one_poison_record(ompf), what);
            let expected_msg = format!(
                "delta error: invalid external-value hash length: got {len} bytes, expected 32"
            );
            assert!(
                matches!(
                    &err,
                    ComposeError::ExtValueIndex {
                        site,
                        source: ServiceError::InvalidArgument(msg),
                    } if site.tail == TailSite::known(0, BASE_SNAP_SEQ + 1)
                        && site.index == 0
                        && site.sequence_num == BASE_SNAP_SEQ + 1
                        && *msg == expected_msg
                ),
                "{what}: expected ExtValueIndex(InvalidArgument({expected_msg:?})) at record[0] \
                 seq={}, got {err:?}",
                BASE_SNAP_SEQ + 1
            );
            assert_permanent(err, what);
        }
    }

    // ── #C6 — ext payloads are views, not copies ────────────────────────────

    #[test]
    fn decode_tail_record_ompf_returns_ext_payloads_as_views_over_the_record() {
        // [`decode_tail_record_ompf`]'s zero-copy contract, which is a memory
        // bound rather than a nicety: `Bytes::copy_from_slice` would be
        // *correct* here and would copy up to
        // `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (64 MiB) per tail, times
        // `..._TAIL_MAX_CHAIN` generations. Nothing else observes the
        // difference — every content assertion in this module passes either
        // way — so this arm is the only thing standing between the engine and
        // a silent 512 MiB restore-time copy.
        let with_ext = OwnedDelta {
            set_fields: vec![ext_field(
                10,
                "viewed_ext",
                HASH_B,
                blob_b().len() as u64,
                1,
                0,
            )],
            ..delta()
        };
        let ompf = Bytes::from(
            OmpfContainer::serialize(
                &oodf_ffi::build_delta(&with_ext).expect("fixture delta must serialize"),
                &[blob_b().as_slice()],
            )
            .expect("fixture container must serialize"),
        );

        let site = RecordSite {
            tail: TailSite::known(0, BASE_SNAP_SEQ + 1),
            index: 0,
            sequence_num: BASE_SNAP_SEQ + 1,
        };
        let (parsed, ext_values) =
            decode_tail_record_ompf(site, &ompf).expect("the record must decode");
        assert!(parsed.is_some());
        assert_eq!(ext_values.len(), 1, "exactly one ext part is referenced");
        let (hash, data) = &ext_values[0];
        assert_eq!(*hash, HASH_B);
        assert_eq!(data.as_ref(), blob_b().as_slice(), "the right bytes");

        // The payload must live INSIDE the record's own buffer. A copy would
        // sit outside it — and `slice_ref` would have panicked outright if the
        // parse had been fed a different allocation, which is the other half of
        // the invariant this pins.
        let record_start = ompf.as_ptr() as usize;
        let payload_start = data.as_ptr() as usize;
        assert!(
            payload_start >= record_start && payload_start + data.len() <= record_start + ompf.len(),
            "ext payloads must be `slice_ref` views over the record's own buffer, not copies: \
             payload at {payload_start:#x}..{:#x} is outside the record at {record_start:#x}..{:#x}",
            payload_start + data.len(),
            record_start + ompf.len()
        );
    }

    // ── b.8jc — a tail record whose bytes carry a `-0.0` time ───────────────

    /// A verbatim tail record whose time sample physically carries `-0.0` —
    /// the writer's bytes as the stream keeps them, from ovlive through any
    /// build — composes, and to the stored state: the sample sits at `+0.0`,
    /// as the live commit of the same bytes stored it, because the record
    /// goes through `parse_delta`, which canonicalizes it, as the live commit
    /// did. The bytes are built at a sentinel time and patched, since
    /// `build_delta` cannot write `-0.0`.
    ///
    /// Without the canonicalization the parsed `-0.0` reaches
    /// `validate_delta_bounds`' backstop and the record is refused, so a DR
    /// restore of the layer would fail on bytes that committed live.
    #[test]
    fn compose_tail_record_physically_carrying_negative_zero_composes_to_the_stored_state() {
        const SENTINEL: f64 = 1_234.567_890_123;
        let record = OwnedDelta {
            time_samples: vec![inline_time_sample(11, SENTINEL, BASE_SNAP_SEQ + 1)],
            ..delta()
        };
        let bytes =
            crate::checkpoint::apply::tests::build_delta_with_negative_zero_time(&record, SENTINEL);

        let composed = compose_one_poison_record(ompf_of(&bytes))
            .expect("a record carrying -0.0 must compose");

        let (history, _base) = canonical();
        let mut stored = history.state_at(BASE_SNAP_SEQ).clone();
        apply_delta(
            &mut stored,
            Some(oodf_ffi::parse_delta(&bytes).expect("the patched record parses")),
        )
        .expect("the live commit of the record's bytes applies");
        assert_eq!(composed.composed_seq, BASE_SNAP_SEQ + 1);
        assert_faithful_columns(&composed.state, &stored);
        let (key, sample) = composed
            .state
            .time_samples
            .get_key_value(&(11, OrderedFloat(0.0)))
            .expect("the composed state holds section 11's time 0");
        assert_eq!(
            (
                key.1.into_inner().to_bits(),
                sample.time.into_inner().to_bits()
            ),
            (0, 0),
            "the composed sample sits at +0.0, as the stored row does"
        );
        assert_eq!(
            sample.value,
            FieldValue::Inline(ts_payload_for(BASE_SNAP_SEQ + 1))
        );
    }
}
