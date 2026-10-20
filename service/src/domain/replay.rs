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

//! Stream-record replay and state normalization.
//!
//! Two pure building blocks for keeping a decoded state current from the
//! object's layer stream instead of re-reading the whole object —
//! the repair step of a per-replica decoded-state cache for structural write
//! planning. The cache's stream repair (`storage::structural_cache_repair`)
//! replays and normalizes with them on every structural `UpdateLayer`
//! attempt under the `lua` backend that finds its entry behind the header.
//!
//! - [`replay_stream_record`] re-applies one committed layer-stream entry
//!   (`le:{oid}:obj:stream`, entry id `{seq}-0`) to a state at `seq - 1`. It
//!   repeats the per-record step of checkpoint compose's replay loop
//!   (`checkpoint::compose::compose_lineage`: OMPF container parse, the live
//!   path's empty-delta branch, `oodf_ffi::parse_delta`, a discarding
//!   `apply_delta`, the post-apply sequence check). Compose keeps its own
//!   copy and does not call this one. Unlike compose it collects no external
//!   values: `apply_delta` keeps each `FieldValue::External`'s
//!   `ext_value_index` verbatim and [`normalize_state`] zeroes it, so the
//!   record's OMPF parts are never resolved. [`replay_stream_record_into`]
//!   is the same step with the caller's mutation sink, for a caller that must
//!   see what the replayed commit wrote (a dangling list it deleted).
//! - [`replay_stream_record_indexed`] (and
//!   [`replay_stream_record_indexed_with_stats`], which also answers the
//!   engine's counters) is that step on the structural cache's compact entry
//!   through its persistent plan indexes (`domain::plan_index`): the indexes
//!   answer the record's lookups and depth check and follow every write, and
//!   the [`PlanTouches`] it fills name the lists the record wrote.
//! - Each of those is two steps, also public for a caller that acts between
//!   them: [`decode_stream_record`] checks the successor and parses the
//!   record into a [`DecodedRecord`] without touching any state, and
//!   [`apply_decoded_record_into`] or [`apply_decoded_record_indexed`]
//!   applies it. Every `replay_stream_record*` entry point is that decode
//!   followed by that apply: one code path. The structural cache's repair
//!   uses the split to price what a record will grow its entry by, from the
//!   decoded delta, and hold that in the call's commit charge before the
//!   apply allocates it (`storage::structural_cache_repair`, RP2).
//! - [`normalize_state`] brings a state to the shape a fresh snapshot read
//!   of the same cut decodes to (`storage::diff_read::snapshot_to_object_state`).
//!   A normalized state at a cut equals that read; that equality is what lets
//!   a replayed state stand in for a fresh read. Two exceptions, both on the
//!   compact form only and both about a section's time 0:
//!   - a commit an older build stored at a `-0.0` time replays at `+0.0`
//!     (the parser canonicalizes it), where the read keys the stored row by
//!     its `-0.0` bits;
//!   - the reconciler's signed-zero merge (`storage::time_sample_merge`)
//!     rewrites stored rows without moving the cut, so the compact
//!     projection of a fresh read at the same cut no longer equals an entry
//!     replayed before the merge.
//!
//!   The structural cache does not rely on the equality in either case: a
//!   reset or cascade deletes a section's time 0 at both stored spellings
//!   and checks each one's existence (rule E, `domain::delta`), so the
//!   first delete of such a row answers `CACHE_DIVERGED` and reloads the
//!   entry, and the sampled audit drops a diverged entry
//!   (`storage::structural_cache`'s module docs list the cases, and the one
//!   rollout residual).
//!
//! Replay and normalization are generic over the state's row form
//! ([`ObjectStateOf`](crate::domain::types::ObjectStateOf)): the structural
//! cache replays onto its compact entry (`CompactState`, whose normalized
//! form at a cut equals the compact projection of that read), and every
//! other caller onto the full `ObjectState`. Each record's OMPF is parsed
//! and applied either way; on a compact state the delta's inline values live
//! only in the parsed delta, for the length of one record.
//!
//! Replay does not normalize; the caller normalizes after replaying, and
//! keeps the parent ids [`normalize_state`] returns as the state's dangling
//! parents (lists the store holds but a snapshot read excludes). A caller
//! replaying a run of records normalizes the children lists after each one
//! ([`normalize_children`]) and the whole state after the last. Through the
//! indexed entry points the caller instead normalizes only the lists each
//! record touched (`domain::plan_index::normalize_touched`, which keeps the
//! indexes' list sum) and runs no per-round [`normalize_state`]: a compact
//! state holds no external-value index for it to zero.
//!
//! **Why this lives in `domain`.** The replay is pure, and its caller is
//! `storage`, which may not gain a new edge into `checkpoint` behaviour
//! except through a trait seam (SDD §3.1). Placing the step in `domain`,
//! below both, lets `storage` replay with no `storage → checkpoint` edge for
//! it. It is the one production `domain` call into `oodf_ffi::parse_delta`,
//! which is itself pure and returns owned types.
//!
//! Pure logic — zero I/O, zero async.

use thiserror::Error;

use oodf_ffi::OodfError;

use crate::domain::delta::{
    apply_delta_into_mode, apply_delta_into_mode_with, ApplyDeltaStats, ApplyMode, DeltaError,
    MutationSink,
};
use crate::domain::ompf::{OmpfContainer, OmpfError};
use crate::domain::plan_index::{IndexedHooks, PlanIndexes, PlanTouches};
use crate::domain::types::{CompactState, FieldRows, ObjectStateOf, SampleRows, ValueRows};

/// The parent id of top-level sections. Its children list is always kept.
const ROOT_PARENT_ID: u64 = 0;

/// Why [`replay_stream_record`] failed.
///
/// [`ReplayError::NotSuccessor`] and the two parse variants leave `state`
/// unchanged. After [`ReplayError::Apply`] or [`ReplayError::PostApplySequence`]
/// `state` may be partially mutated and must be discarded (`apply_delta`'s
/// partial-mutation contract, [`crate::domain::delta::apply_delta`]).
/// [`ReplayError::class`] sorts the variants into the two classes a caller
/// reports.
#[derive(Debug, Error)]
pub enum ReplayError {
    /// The entry's OMPF container is malformed.
    #[error("stream entry {seq}: OMPF container parse failed: {source}")]
    OmpfParse {
        seq: u64,
        #[source]
        source: OmpfError,
    },

    /// The entry's OODF payload failed to parse, or the verifier rejected it.
    #[error("stream entry {seq}: OODF delta parse failed: {source}")]
    OodfParse {
        seq: u64,
        #[source]
        source: OodfError,
    },

    /// `apply_delta` rejected a delta that had committed.
    #[error("stream entry {seq}: apply_delta rejected the delta: {source}")]
    Apply {
        seq: u64,
        #[source]
        source: DeltaError,
    },

    /// `seq` is not the successor of the state's sequence. Checked before
    /// anything is decoded or applied, so `state` is unchanged.
    #[error("stream entry {seq} does not follow the state's sequence {state_seq}")]
    NotSuccessor { seq: u64, state_seq: u64 },

    /// The state after applying the entry is not at `seq`.
    #[error("stream entry {seq}: state landed on sequence {state_seq}")]
    PostApplySequence { seq: u64, state_seq: u64 },
}

/// The two classes of [`ReplayError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayErrorClass {
    /// The entry's bytes could not be decoded ([`ReplayError::OmpfParse`],
    /// [`ReplayError::OodfParse`]).
    Parse,
    /// The entry did not replay onto the state ([`ReplayError::Apply`],
    /// [`ReplayError::NotSuccessor`], [`ReplayError::PostApplySequence`]).
    Replay,
}

impl ReplayError {
    /// Which class this error belongs to.
    pub fn class(&self) -> ReplayErrorClass {
        match self {
            ReplayError::OmpfParse { .. } | ReplayError::OodfParse { .. } => {
                ReplayErrorClass::Parse
            }
            ReplayError::Apply { .. }
            | ReplayError::NotSuccessor { .. }
            | ReplayError::PostApplySequence { .. } => ReplayErrorClass::Replay,
        }
    }
}

/// Re-apply the committed stream entry at `seq` to `state`, which must stand
/// at `seq - 1`.
///
/// `ompf` is the entry's `ompf` field (`storage::notifications::RELAY_OMPF_FIELD`).
/// An absent field, or an OMPF whose delta bytes are empty, is an empty commit:
/// it is applied as a `None` delta, which changes no content and still advances
/// `state.sequence_num` to `seq` (SDD §8.4).
///
/// Reading an absent `ompf` as an empty commit rests on SDD §12.2's
/// `ompf`-presence invariant and holds only for an `update` entry above
/// sequence 1: `CreateLayer`'s `1-0` seed entry also has no `ompf` but carries
/// the whole created content, and an entry written before the verbatim relay
/// existed (excluded by §12.2's upgrade precondition) would replay here as a
/// silent no-op. Nothing in the entry's text payload tells either apart.
///
/// Otherwise the steps are compose's per-record step: `OmpfContainer::parse`,
/// `oodf_ffi::parse_delta`, then `apply_delta` with a discarding sink in
/// [`ApplyMode::Replay`] — a delta with no reset and no structural command
/// other than the ignored `DiffDeleteSection` skips the section-tree depth
/// walk: it cannot change the tree, and the Lua fast path that commits such
/// deltas never walks it.
///
/// # Errors
///
/// - [`ReplayError::NotSuccessor`] — `seq` is not `state.sequence_num + 1`
///   (`state` is untouched).
/// - [`ReplayError::OmpfParse`] / [`ReplayError::OodfParse`] — the entry's
///   bytes failed to decode (`state` is untouched).
/// - [`ReplayError::Apply`] — `apply_delta` rejected the delta; `state` may be
///   partially mutated and must be discarded.
/// - [`ReplayError::PostApplySequence`] — the applied state did not land on
///   `seq`; `state` must be discarded.
pub fn replay_stream_record<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    ompf: Option<&[u8]>,
    seq: u64,
) -> Result<(), ReplayError> {
    // Replay reproduces the state; the mutations a commit would plan are
    // never read, so build none of them (the `Discard` caller list in
    // `domain::delta` names this site).
    replay_stream_record_into(state, ompf, seq, &mut MutationSink::Discard)
}

/// [`replay_stream_record`] with the caller's [`MutationSink`]: the same
/// steps, in the same [`ApplyMode::Replay`], with the replayed delta's planned
/// persistence mutations sent to `sink`.
///
/// For a caller that must see what the replayed commit wrote beyond the state
/// itself. The structural cache's repair (`storage::structural_cache_repair`)
/// is one: a commit's `DeleteChildrenList` — or empty `SetChildren` — of a
/// parent whose stored list the state deliberately does not hold (a dangling
/// parent, see [`normalize_state`]) deletes that stored list, and nothing in
/// the resulting state shows it. Replaying through a collecting sink
/// reproduces exactly the mutations the committing attempt planned, because
/// that attempt planned on the same normalized state; the repair replays
/// through [`MutationSink::ListDeletes`] instead, which keeps only the ids
/// of the lists those mutations delete.
///
/// # Errors
///
/// As [`replay_stream_record`]. On an error, `sink` may hold the mutations of
/// a partial apply; discard them with the state.
pub fn replay_stream_record_into<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    ompf: Option<&[u8]>,
    seq: u64,
    sink: &mut MutationSink,
) -> Result<(), ReplayError> {
    let record = decode_stream_record(state.sequence_num, ompf, seq)?;
    apply_decoded_record_into(state, record, sink)
}

/// [`replay_stream_record_into`] on the structural cache's compact entry,
/// keeping its persistent plan indexes (`domain::plan_index`) in step with
/// every write and recording what the record touched in `touches`: the
/// record's `(parent_id, name)` lookups and adoptions read `indexes` instead
/// of per-delta indexes built from every section, and its depth check is
/// incremental while `indexes` say the state passed the full walk.
///
/// The same steps and errors as [`replay_stream_record_into`]. On an error
/// `indexes` follow the partial apply and must be discarded with the state.
pub fn replay_stream_record_indexed(
    state: &mut CompactState,
    ompf: Option<&[u8]>,
    seq: u64,
    sink: &mut MutationSink,
    indexes: &mut PlanIndexes,
    touches: &mut PlanTouches,
) -> Result<(), ReplayError> {
    replay_stream_record_indexed_with_stats(state, ompf, seq, sink, indexes, touches).map(drop)
}

/// [`replay_stream_record_indexed`], also answering the engine's operation
/// counters for the replayed record ([`ApplyDeltaStats`]), which every
/// replay computes: with the plan indexes answering the record's lookups,
/// its `name_index_builds` and `adoption_index_builds` stay 0.
///
/// # Errors
///
/// As [`replay_stream_record_indexed`].
pub fn replay_stream_record_indexed_with_stats(
    state: &mut CompactState,
    ompf: Option<&[u8]>,
    seq: u64,
    sink: &mut MutationSink,
    indexes: &mut PlanIndexes,
    touches: &mut PlanTouches,
) -> Result<ApplyDeltaStats, ReplayError> {
    let record = decode_stream_record(state.sequence_num, ompf, seq)?;
    apply_decoded_record_indexed(state, record, sink, indexes, touches)
}

/// One committed stream entry, decoded and not yet applied: what
/// [`decode_stream_record`] answers and the `apply_decoded_record_*` steps
/// consume. Holding it changes no state, so a caller can look at the
/// record's delta between the two — the structural cache's repair prices
/// what the record will grow the entry by (RP2) before it applies it.
#[derive(Debug)]
pub struct DecodedRecord {
    seq: u64,
    delta: Option<oodf_ffi::OwnedDelta>,
}

impl DecodedRecord {
    /// The entry's `seq`, which the state lands on once the record applies.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The record's parsed delta; `None` for an empty commit (no `ompf`, or
    /// an OMPF with no delta bytes).
    pub fn delta(&self) -> Option<&oodf_ffi::OwnedDelta> {
        self.delta.as_ref()
    }
}

/// The decode half of [`replay_stream_record`]: check that `seq` follows
/// `state_seq`, the sequence of the state it will apply to, then parse the
/// entry's `ompf` field — `OmpfContainer::parse`, the live path's empty-delta
/// branch, `oodf_ffi::parse_delta` — into a [`DecodedRecord`]. Nothing is
/// mutated, on success or on error.
///
/// An absent `ompf`, or an OMPF whose delta bytes are empty, decodes to a
/// `None` delta: an empty commit ([`replay_stream_record`] gives the
/// invariant that reading rests on).
///
/// # Errors
///
/// - [`ReplayError::NotSuccessor`] — `seq` is not `state_seq + 1`, checked
///   before anything is parsed;
/// - [`ReplayError::OmpfParse`] / [`ReplayError::OodfParse`] — the entry's
///   bytes failed to decode.
pub fn decode_stream_record(
    state_seq: u64,
    ompf: Option<&[u8]>,
    seq: u64,
) -> Result<DecodedRecord, ReplayError> {
    // Checked before decoding, so a misplaced entry costs no parse.
    // `checked_add`: a state at `u64::MAX` has no successor.
    check_successor(state_seq, seq)?;

    let delta = match ompf {
        None => None,
        Some(bytes) => {
            let container = OmpfContainer::parse(bytes)
                .map_err(|source| ReplayError::OmpfParse { seq, source })?;
            // The live path's empty-content branch: no delta bytes is an empty
            // commit, whatever else the container carries. `parse_delta` would
            // reject empty input.
            if container.delta_bytes.is_empty() {
                None
            } else {
                Some(
                    oodf_ffi::parse_delta(container.delta_bytes)
                        .map_err(|source| ReplayError::OodfParse { seq, source })?,
                )
            }
        }
    };
    Ok(DecodedRecord { seq, delta })
}

/// The apply half of [`replay_stream_record_into`]: apply a record
/// [`decode_stream_record`] decoded to `state`, which must stand at
/// `record.seq() - 1`, in [`ApplyMode::Replay`] with the caller's `sink`,
/// then check that the state landed on `record.seq()`.
///
/// # Errors
///
/// - [`ReplayError::NotSuccessor`] — the record does not follow `state`'s
///   sequence, re-checked here since the state may have moved since the
///   decode (`state` is untouched);
/// - [`ReplayError::Apply`] — `apply_delta` rejected the delta; `state` may
///   be partially mutated and must be discarded;
/// - [`ReplayError::PostApplySequence`] — the applied state did not land on
///   the record's `seq`; `state` must be discarded.
///
/// On an error after the successor check, `sink` may hold the mutations of a
/// partial apply; discard them with the state.
pub fn apply_decoded_record_into<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    record: DecodedRecord,
    sink: &mut MutationSink,
) -> Result<(), ReplayError> {
    apply_decoded_with(state, record, |state, delta, stats| {
        apply_delta_into_mode(state, delta, sink, stats, ApplyMode::Replay)
    })
    .map(drop)
}

/// [`apply_decoded_record_into`] on the structural cache's compact entry
/// through its persistent plan indexes, as
/// [`replay_stream_record_indexed_with_stats`] applies: `indexes` follow
/// every write and answer the record's lookups and depth check, and
/// `touches` record what it touched. Answers the engine's counters
/// ([`ApplyDeltaStats`]).
///
/// # Errors
///
/// As [`apply_decoded_record_into`]. On an error `indexes` follow the partial
/// apply and must be discarded with the state.
pub fn apply_decoded_record_indexed(
    state: &mut CompactState,
    record: DecodedRecord,
    sink: &mut MutationSink,
    indexes: &mut PlanIndexes,
    touches: &mut PlanTouches,
) -> Result<ApplyDeltaStats, ReplayError> {
    apply_decoded_with(state, record, |state, delta, stats| {
        let hooks = IndexedHooks::new(indexes, touches);
        apply_delta_into_mode_with(state, delta, sink, stats, ApplyMode::Replay, hooks)
    })
}

/// [`ReplayError::NotSuccessor`] unless `seq` is `state_seq + 1`.
fn check_successor(state_seq: u64, seq: u64) -> Result<(), ReplayError> {
    if state_seq.checked_add(1) == Some(seq) {
        Ok(())
    } else {
        Err(ReplayError::NotSuccessor { seq, state_seq })
    }
}

/// The apply steps of the `apply_decoded_record_*` functions, with the
/// engine call itself supplied by the caller: `apply` runs the engine in
/// [`ApplyMode::Replay`] on the decoded delta, through the caller's sink,
/// with or without hooks. Answers the apply's [`ApplyDeltaStats`].
fn apply_decoded_with<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    record: DecodedRecord,
    apply: impl FnOnce(
        &mut ObjectStateOf<R>,
        Option<oodf_ffi::OwnedDelta>,
        &mut ApplyDeltaStats,
    ) -> Result<(), DeltaError>,
) -> Result<ApplyDeltaStats, ReplayError> {
    let DecodedRecord { seq, delta } = record;
    // Re-checked before applying, so a record decoded against another state
    // leaves this one intact.
    check_successor(state.sequence_num, seq)?;

    let mut stats = ApplyDeltaStats::default();
    apply(state, delta, &mut stats).map_err(|source| ReplayError::Apply { seq, source })?;

    // Compose's post-apply check: `apply_delta` lands on the predecessor plus
    // one on success, so this trips only if that contract breaks.
    if state.sequence_num != seq {
        return Err(ReplayError::PostApplySequence {
            seq,
            state_seq: state.sequence_num,
        });
    }
    Ok(stats)
}

/// Bring `state` to the shape a fresh snapshot read of the same cut decodes
/// to, and return the parent ids of the children lists dropped as dangling.
///
/// Three rewrites, each matching what the stored form or the snapshot readers
/// cannot represent:
///
/// 1. Every empty children list is dropped: `children[p] == []` has no stored
///    form. Its parent id is not returned — there is no stored list under it.
/// 2. Every `FieldValue::External`'s `ext_value_index`, on fields and time
///    samples alike, is set to 0: the index is in-memory only and a stored row
///    decodes to 0. A compact state holds no index, so this is a no-op there.
/// 3. Every remaining children list whose parent is neither the root (`0`) nor
///    a section in `state.sections` is dropped, and its parent id returned:
///    both snapshot readers exclude lists under dangling parents. The caller
///    keeps these ids as the state's dangling parents.
///
/// The returned ids are sorted ascending and unique. Idempotent: a second
/// call changes nothing and returns no ids.
pub fn normalize_state<R: ValueRows>(state: &mut ObjectStateOf<R>) -> Vec<u64> {
    let dangling = normalize_children(state);

    state.fields.zero_ext_indexes();
    state.time_samples.zero_ext_indexes();

    dangling
}

/// Rewrites 1 and 3 of [`normalize_state`] alone — the children lists — and
/// the same returned ids; external-value indexes are left as they are.
///
/// O(children lists), where [`normalize_state`] also walks every field and
/// time sample. For a caller that replays a run of records and must see each
/// record's dangling parents as it lands: no apply step reads an existing
/// row's `ext_value_index` (a `SetField` or time sample replaces the row
/// whole), so zeroing the indexes once after the run gives the same state as
/// zeroing them after every record, while the children lists must be
/// normalized per record — the next record was planned on a state without
/// them. A state is normalized once [`normalize_state`] has run after the last
/// record.
pub fn normalize_children<R: ValueRows>(state: &mut ObjectStateOf<R>) -> Vec<u64> {
    let sections = &state.sections;
    let mut dangling = Vec::new();
    state.children.retain(|&parent_id, kids| {
        if kids.is_empty() {
            return false;
        }
        if parent_id == ROOT_PARENT_ID || sections.contains_key(&parent_id) {
            return true;
        }
        dangling.push(parent_id);
        false
    });
    dangling.sort_unstable();
    dangling
}

#[cfg(test)]
mod tests {
    use super::*;

    use oodf_ffi::{
        CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
        CommandDiffDeleteSection, CommandMoveSection, CommandReorderChildren, OwnedDelta, SetField,
        StructCommand, TimeSample as OodfTimeSample,
    };
    use ordered_float::OrderedFloat;

    use crate::domain::delta::apply_delta;
    use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
    use crate::domain::types::{compact, Field, FieldValue, ObjectState, Section, TimeSample};

    const OBJECT_ID: u64 = 1;

    /// A 32-byte SHA-256-shaped hash for an external value; `ext_value_size > 0`
    /// is what makes `apply_delta` read the bytes as a hash.
    const EXT_HASH: [u8; 32] = [0xab; 32];

    // ── Builders ─────────────────────────────────────────────────────────────

    /// The stream entry's `ompf` field as a writer sends it: `delta`'s OODF
    /// bytes plus `ext` as the external-value parts.
    fn ompf(delta: &OwnedDelta, ext: &[&[u8]]) -> Vec<u8> {
        let delta_bytes = oodf_ffi::build_delta(delta).expect("build_delta");
        OmpfContainer::serialize(&delta_bytes, ext).expect("serialize OMPF")
    }

    fn create(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: 1,
        })
    }

    fn inline_field(section_id: u64, key: &str, byte: u8, set_order: u64) -> SetField {
        SetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: vec![byte; 3],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }
    }

    /// An external field whose `ext_value_index` names the OMPF's first
    /// external part (the index is 1-based over the external parts).
    fn external_field(section_id: u64, key: &str) -> SetField {
        SetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: EXT_HASH.to_vec(),
            ext_value_size: 5,
            ext_value_index: 1,
            set_order: 1,
        }
    }

    fn external_sample(section_id: u64, time: f64) -> OodfTimeSample {
        OodfTimeSample {
            section_id,
            time,
            value_or_ext_hash: EXT_HASH.to_vec(),
            ext_value_size: 5,
            ext_value_index: 1,
        }
    }

    /// A non-reset delta carrying only field and time-sample writes, plus an
    /// ignored `DiffDeleteSection`: the class replay mode lets skip the depth
    /// walk. Targets section 1, present in both [`small_tree`] and
    /// [`too_deep_chain`].
    fn field_only_delta() -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 4,
            struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
                section_id: 1,
            })],
            set_fields: vec![inline_field(1, "inline", 7, 2), external_field(1, "ext")],
            time_samples: vec![external_sample(1, 1.5)],
            ..Default::default()
        }
    }

    /// A non-reset delta with one `CreateSection` under the root, for an id
    /// outside both [`small_tree`] and [`too_deep_chain`].
    fn structural_delta() -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![create(0, 900_000, "Added")],
            ..Default::default()
        }
    }

    fn section(id: u64, parent: u64, name: &str) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: name.to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    /// `0 → 1 → 2` with one inline field on 2, standing at `seq`.
    fn small_tree(seq: u64) -> ObjectState {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: seq,
            ..Default::default()
        };
        state.sections.insert(1, section(1, 0, "A"));
        state.sections.insert(2, section(2, 1, "B"));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.fields.insert(
            (2, "k".to_string()),
            Field {
                section_id: 2,
                key_name: "k".to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
        state
    }

    /// A linear chain `1 → 2 → … → cap + 1` under the root at sequence 0: one
    /// level deeper than `MAX_SECTION_TREE_DEPTH`, so any apply that runs the
    /// depth walk over it fails `SectionTreeTooDeep`. Whether a replay walked
    /// is therefore observable from its result alone.
    fn too_deep_chain() -> ObjectState {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            ..Default::default()
        };
        for id in 1..=(MAX_SECTION_TREE_DEPTH as u64 + 1) {
            let parent = id - 1;
            state
                .sections
                .insert(id, section(id, parent, &format!("S{id}")));
            state.children.entry(parent).or_default().push(id);
        }
        state
    }

    fn external(ext_value_index: u32) -> FieldValue {
        FieldValue::External {
            hash: EXT_HASH,
            size: 5,
            ext_value_index,
        }
    }

    // ── replay_stream_record: empty commits ──────────────────────────────────

    /// SDD §8.4 via the stream: an absent `ompf` field, an empty one, and an
    /// OMPF whose delta part is empty (an external part alone does not make a
    /// delta) are each an empty commit — content unchanged, `seq` advanced.
    #[test]
    fn replay_stream_record_empty_commit_advances_seq_and_changes_no_content() {
        let empty_delta_part = OmpfContainer::serialize(&[], &[b"stray ext"]).unwrap();
        assert!(
            !empty_delta_part.is_empty(),
            "non-vacuity: this case must reach the parsed-container branch, not \
             OMPF's empty-input shortcut"
        );
        let cases: [(&str, Option<&[u8]>); 3] = [
            ("absent ompf field", None),
            ("empty ompf field", Some(&[])),
            ("OMPF with empty delta bytes", Some(&empty_delta_part)),
        ];

        for (label, entry) in cases {
            let mut state = small_tree(4);
            let mut expected = state.clone();
            expected.sequence_num = 5;

            replay_stream_record(&mut state, entry, 5)
                .unwrap_or_else(|e| panic!("{label}: an empty commit must replay: {e}"));

            assert_eq!(state, expected, "{label}: only the sequence number moves");
        }
    }

    // ── replay_stream_record: errors ─────────────────────────────────────────

    /// A `seq` that is not the state's successor is refused before anything is
    /// decoded or applied. Every case carries a delta that would mutate the
    /// state if applied, so "unchanged" is not vacuous; the malformed case
    /// pins that the successor check runs before decoding.
    #[test]
    fn replay_stream_record_not_successor_returns_not_successor_and_leaves_state_unchanged() {
        let mutating = ompf(&structural_delta(), &[]);
        let malformed: &[u8] = b"not an OMPF container";
        let cases: [(&str, u64, u64, &[u8]); 5] = [
            ("same seq", 5, 5, &mutating),
            ("skips one", 5, 7, &mutating),
            ("behind", 5, 3, &mutating),
            // `u64::MAX + 1` wraps to 0: the check must not wrap.
            ("state at u64::MAX", u64::MAX, 0, &mutating),
            ("not successor and malformed", 5, 9, malformed),
        ];

        for (label, state_seq, seq, entry) in cases {
            let mut state = small_tree(state_seq);
            let before = state.clone();

            let err = replay_stream_record(&mut state, Some(entry), seq).unwrap_err();

            assert!(
                matches!(
                    err,
                    ReplayError::NotSuccessor { seq: s, state_seq: ss }
                        if s == seq && ss == state_seq
                ),
                "{label}: expected NotSuccessor {{ seq: {seq}, state_seq: {state_seq} }}, \
                 got {err:?}"
            );
            assert_eq!(state, before, "{label}: the state is untouched");
        }
    }

    #[test]
    fn replay_stream_record_malformed_ompf_returns_ompf_parse_and_leaves_state_unchanged() {
        let mut state = small_tree(4);
        let before = state.clone();

        let err = replay_stream_record(&mut state, Some(b"not an OMPF container"), 5).unwrap_err();

        assert!(
            matches!(
                err,
                ReplayError::OmpfParse {
                    seq: 5,
                    source: OmpfError::BadMagic { .. }
                }
            ),
            "expected OmpfParse(BadMagic) at seq 5, got {err:?}"
        );
        assert_eq!(err.class(), ReplayErrorClass::Parse);
        assert_eq!(state, before);
    }

    #[test]
    fn replay_stream_record_malformed_oodf_returns_oodf_parse_and_leaves_state_unchanged() {
        let mut state = small_tree(4);
        let before = state.clone();
        let entry = OmpfContainer::serialize(b"not an OODF flatbuffer", &[]).unwrap();

        let err = replay_stream_record(&mut state, Some(&entry), 5).unwrap_err();

        assert!(
            matches!(err, ReplayError::OodfParse { seq: 5, .. }),
            "expected OodfParse at seq 5, got {err:?}"
        );
        assert_eq!(err.class(), ReplayErrorClass::Parse);
        assert_eq!(state, before);
    }

    /// A delta that decodes but that `apply_delta` rejects surfaces as
    /// `Apply`, carrying the engine's own error.
    #[test]
    fn replay_stream_record_delta_the_engine_rejects_returns_apply_with_its_source() {
        let mut state = small_tree(4);
        let self_parenting = OwnedDelta {
            struct_commands: vec![create(7, 7, "Loop")],
            ..Default::default()
        };

        let err =
            replay_stream_record(&mut state, Some(&ompf(&self_parenting, &[])), 5).unwrap_err();

        assert!(
            matches!(
                err,
                ReplayError::Apply {
                    seq: 5,
                    source: DeltaError::SelfParenting { .. }
                }
            ),
            "expected Apply(SelfParenting) at seq 5, got {err:?}"
        );
        assert_eq!(err.class(), ReplayErrorClass::Replay);
    }

    /// `class()` sorts every variant. `PostApplySequence` is unreachable while
    /// `apply_delta` keeps its contract, so its class is pinned on a
    /// constructed value.
    #[test]
    fn replay_error_class_sorts_decode_failures_as_parse_and_the_rest_as_replay() {
        let cases = [
            (
                ReplayError::OmpfParse {
                    seq: 1,
                    source: OmpfError::NoParts,
                },
                ReplayErrorClass::Parse,
            ),
            (
                ReplayError::OodfParse {
                    seq: 1,
                    source: OodfError::EmptyInput,
                },
                ReplayErrorClass::Parse,
            ),
            (
                ReplayError::Apply {
                    seq: 1,
                    source: DeltaError::SequenceOverflow,
                },
                ReplayErrorClass::Replay,
            ),
            (
                ReplayError::NotSuccessor {
                    seq: 3,
                    state_seq: 1,
                },
                ReplayErrorClass::Replay,
            ),
            (
                ReplayError::PostApplySequence {
                    seq: 2,
                    state_seq: 1,
                },
                ReplayErrorClass::Replay,
            ),
        ];

        for (err, class) in cases {
            assert_eq!(err.class(), class, "{err:?}");
        }
    }

    // ── replay_stream_record: replay mode ────────────────────────────────────

    /// SR-4.1's pin: a field-only replay, which skips the depth walk, lands on
    /// exactly the state `apply_delta` — which always walks — produces.
    #[test]
    fn replay_stream_record_field_only_delta_matches_apply_delta() {
        let delta = field_only_delta();
        let mut replayed = small_tree(4);
        let mut walked = small_tree(4);

        replay_stream_record(&mut replayed, Some(&ompf(&delta, &[b"hello"])), 5).expect("replay");
        apply_delta(&mut walked, Some(delta)).expect("apply_delta");

        assert_eq!(replayed, walked);
        assert_eq!(
            replayed.fields[&(1, "ext".to_string())].value,
            external(1),
            "non-vacuity: the delta's external field landed, ext_value_index verbatim"
        );
    }

    /// On a stored tree deeper than the cap, a field-only entry still replays:
    /// replay mode does not walk the tree for a delta that cannot change it.
    /// `apply_delta` over the same state shows the walk would have failed.
    #[test]
    fn replay_stream_record_field_only_delta_on_a_too_deep_tree_succeeds() {
        let entry = ompf(&field_only_delta(), &[b"hello"]);
        let mut state = too_deep_chain();

        replay_stream_record(&mut state, Some(&entry), 1).expect("replay skips the walk");
        assert_eq!(state.sequence_num, 1);
        assert_eq!(state.fields.len(), 2, "the delta's fields landed");

        let err = apply_delta(&mut too_deep_chain(), Some(field_only_delta())).unwrap_err();
        assert!(
            matches!(err, DeltaError::SectionTreeTooDeep { .. }),
            "non-vacuity: the walk fails on this tree, got {err:?}"
        );
    }

    #[test]
    fn replay_stream_record_structural_delta_on_a_too_deep_tree_returns_section_tree_too_deep() {
        let mut state = too_deep_chain();

        let err =
            replay_stream_record(&mut state, Some(&ompf(&structural_delta(), &[])), 1).unwrap_err();

        assert!(
            matches!(
                err,
                ReplayError::Apply {
                    seq: 1,
                    source: DeltaError::SectionTreeTooDeep { .. }
                }
            ),
            "a structural entry still walks the tree, got {err:?}"
        );
    }

    /// A whole lineage — the create, field commits, structural commits, empty
    /// commits of all three shapes, and a reset — replayed record by record
    /// lands, at every step, on the state `apply_delta` builds from the same
    /// deltas, and the two normalize identically.
    #[test]
    fn replay_stream_record_lineage_matches_apply_delta_at_every_step() {
        let empty_delta_part = OmpfContainer::serialize(&[], &[b"stray ext"]).unwrap();
        let reset = |cmds, set_fields| OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            struct_commands: cmds,
            set_fields,
            ..Default::default()
        };
        let plain = |cmds, set_fields, time_samples| OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: cmds,
            set_fields,
            time_samples,
            ..Default::default()
        };

        /// (label, delta for `apply_delta`, the stream entry's `ompf` field)
        type Record = (&'static str, Option<OwnedDelta>, Option<Vec<u8>>);
        let mut records: Vec<Record> = Vec::new();
        let mut push = |label, delta: Option<OwnedDelta>, ext: &[&[u8]]| {
            let entry = delta.as_ref().map(|d| ompf(d, ext));
            records.push((label, delta, entry));
        };
        push(
            "create",
            Some(reset(
                vec![create(0, 1, "A"), create(1, 2, "B"), create(1, 3, "C")],
                vec![inline_field(1, "a", 1, 1)],
            )),
            &[],
        );
        push(
            "field commit with external values",
            Some(plain(
                vec![],
                vec![inline_field(2, "b", 2, 1), external_field(3, "c")],
                vec![external_sample(3, 0.5)],
            )),
            &[b"hello"],
        );
        push("empty commit, absent ompf", None, &[]);
        push(
            "structural commit",
            Some(plain(
                vec![
                    StructCommand::MoveSection(CommandMoveSection {
                        old_parent_id: 1,
                        new_parent_id: 0,
                        section_id: 3,
                        new_name: "C2".to_string(),
                    }),
                    StructCommand::ReorderChildren(CommandReorderChildren {
                        section_id: 0,
                        children_list_id: 0,
                        children_list: vec![3, 1],
                    }),
                    StructCommand::DiffChangeSection(CommandDiffChangeSection {
                        parent_id: 2,
                        section_id: 4,
                        section_name: "D".to_string(),
                        section_type: 1,
                        section_order: 2,
                    }),
                ],
                vec![inline_field(4, "d", 4, 1)],
                vec![],
            )),
            &[],
        );
        push(
            "structural commit with a cascade",
            Some(plain(
                vec![
                    StructCommand::DeleteSection(CommandDeleteSection {
                        parent_id: 0,
                        section_id: 2,
                    }),
                    create(3, 5, "E"),
                ],
                vec![],
                vec![],
            )),
            &[],
        );
        push(
            "field commit with an ignored DiffDeleteSection",
            Some(plain(
                vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
                    section_id: 5,
                })],
                vec![inline_field(5, "e", 5, 3)],
                vec![],
            )),
            &[],
        );
        push(
            "reset",
            Some(reset(
                vec![create(0, 10, "X")],
                vec![external_field(10, "x")],
            )),
            &[b"hello"],
        );
        push(
            "field commit after the reset",
            Some(plain(vec![], vec![inline_field(10, "y", 9, 1)], vec![])),
            &[],
        );
        records.push(("empty commit, empty ompf", None, Some(Vec::new())));
        records.push((
            "empty commit, OMPF with empty delta bytes",
            None,
            Some(empty_delta_part),
        ));

        let start = ObjectState {
            object_id: OBJECT_ID,
            ..Default::default()
        };
        let mut replayed = start.clone();
        let mut reference = start;
        for (i, (label, delta, entry)) in records.into_iter().enumerate() {
            let seq = i as u64 + 1;
            replay_stream_record(&mut replayed, entry.as_deref(), seq)
                .unwrap_or_else(|e| panic!("{label} (seq {seq}): replay failed: {e}"));
            apply_delta(&mut reference, delta)
                .unwrap_or_else(|e| panic!("{label} (seq {seq}): apply_delta failed: {e}"));

            assert_eq!(replayed, reference, "{label} (seq {seq})");
            let (mut replayed_norm, mut reference_norm) = (replayed.clone(), reference.clone());
            assert_eq!(
                normalize_state(&mut replayed_norm),
                normalize_state(&mut reference_norm),
                "{label} (seq {seq}): dangling parents"
            );
            assert_eq!(
                replayed_norm, reference_norm,
                "{label} (seq {seq}): normalized"
            );
        }

        // Non-vacuity: the lineage ended past its reset, on the reset's rows.
        assert_eq!(replayed.sequence_num, 10);
        assert_eq!(replayed.first_version, 7, "the reset opened the generation");
        assert_eq!(replayed.sections.keys().collect::<Vec<_>>(), vec![&10]);
    }

    // ── normalize_state ──────────────────────────────────────────────────────

    #[test]
    fn normalize_state_empty_children_lists_dropped_without_reporting_their_parent() {
        let mut state = small_tree(1);
        // Under the root: the realistic empty list. `apply_delta` leaves
        // `children[0] == []` when a layer's last top-level section is deleted
        // or moved away. It pins that the empty-list rule runs before the
        // keep-the-root rule.
        state.children.insert(0, vec![]);
        state.children.insert(2, vec![]); // under a live section
        state.children.insert(77, vec![]); // under a missing parent

        let dangling = normalize_state(&mut state);

        assert_eq!(
            dangling,
            Vec::<u64>::new(),
            "an empty list has no stored form, so no dangling parent is recorded"
        );
        assert!(
            !state.children.contains_key(&0),
            "the root's empty list is dropped too"
        );
        assert!(!state.children.contains_key(&2));
        assert!(!state.children.contains_key(&77));
        assert_eq!(
            state.children.keys().collect::<Vec<_>>(),
            vec![&1],
            "only the non-empty list stays"
        );
    }

    #[test]
    fn normalize_state_zeroes_ext_value_index_on_fields_and_time_samples() {
        let mut state = small_tree(1);
        state.fields.insert(
            (1, "ext".to_string()),
            Field {
                section_id: 1,
                key_name: "ext".to_string(),
                value: external(3),
                set_order: 2,
                authoring_set_order: 2,
                version: 1,
            },
        );
        state.time_samples.insert(
            (1, OrderedFloat(2.0)),
            TimeSample {
                section_id: 1,
                time: OrderedFloat(2.0),
                value: external(4),
                version: 1,
            },
        );

        normalize_state(&mut state);

        assert_eq!(state.fields[&(1, "ext".to_string())].value, external(0));
        assert_eq!(
            state.time_samples[&(1, OrderedFloat(2.0))].value,
            external(0)
        );
        assert_eq!(
            state.fields[&(2, "k".to_string())].value,
            FieldValue::Inline(vec![1]),
            "an inline value is untouched"
        );
    }

    #[test]
    fn normalize_state_lists_under_missing_parents_dropped_and_their_ids_returned_sorted() {
        let mut state = small_tree(1);
        // Enough dangling parents that a HashMap's iteration order is unlikely
        // to come out sorted by accident.
        for parent in [900, 40, 7, 1_000_000, 3] {
            state.children.insert(parent, vec![parent + 1]);
        }

        let dangling = normalize_state(&mut state);

        assert_eq!(dangling, vec![3, 7, 40, 900, 1_000_000]);
        let mut kept: Vec<u64> = state.children.keys().copied().collect();
        kept.sort_unstable();
        assert_eq!(
            kept,
            vec![0, 1],
            "only the root's and live section 1's lists stay"
        );
    }

    #[test]
    fn normalize_state_lists_under_the_root_and_a_live_section_kept_verbatim() {
        let mut state = small_tree(1);
        let before = state.children.clone();

        let dangling = normalize_state(&mut state);

        assert_eq!(dangling, Vec::<u64>::new());
        assert_eq!(state.children, before);
    }

    #[test]
    fn normalize_state_second_call_changes_nothing_and_returns_no_ids() {
        let mut state = small_tree(1);
        state.children.insert(55, vec![56]);
        state.children.insert(2, vec![]);
        state.fields.get_mut(&(2, "k".to_string())).unwrap().value = external(9);
        assert_eq!(normalize_state(&mut state), vec![55]);
        let once = state.clone();

        let dangling = normalize_state(&mut state);

        assert_eq!(dangling, Vec::<u64>::new());
        assert_eq!(state, once);
    }

    // ── decode_stream_record + apply_decoded_record_* ────────────────────────

    /// A record of [`decode_then_apply_matches_every_entry_point`]: its label
    /// and its stream entry's `ompf` field.
    type Entry = (&'static str, Option<Vec<u8>>);

    /// One lineage covering each shape a replayed record takes: the create, a
    /// field commit with external values, empty commits of all three shapes,
    /// a structural commit (a move, a reorder, a birth by
    /// `DiffChangeSection`), a cascade with a create under the survivor, a
    /// field commit with an ignored `DiffDeleteSection`, and a reset.
    fn lineage() -> Vec<Entry> {
        let empty_delta_part = OmpfContainer::serialize(&[], &[b"stray ext"]).unwrap();
        let plain = |cmds, set_fields| OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: cmds,
            set_fields,
            ..Default::default()
        };
        vec![
            (
                "create",
                Some(ompf(
                    &OwnedDelta {
                        is_diff: true,
                        base_version: 0,
                        source_format: "usda".to_string(),
                        struct_commands: vec![
                            create(0, 1, "A"),
                            create(1, 2, "B"),
                            create(1, 3, "C"),
                        ],
                        set_fields: vec![inline_field(1, "a", 1, 1)],
                        ..Default::default()
                    },
                    &[],
                )),
            ),
            (
                "field commit with external values",
                Some(ompf(
                    &OwnedDelta {
                        time_samples: vec![external_sample(3, 0.5)],
                        ..plain(
                            vec![],
                            vec![inline_field(2, "b", 2, 1), external_field(3, "c")],
                        )
                    },
                    &[b"hello"],
                )),
            ),
            ("empty commit, absent ompf", None),
            ("empty commit, empty ompf", Some(Vec::new())),
            ("empty commit, empty delta part", Some(empty_delta_part)),
            (
                "structural commit",
                Some(ompf(
                    &plain(
                        vec![
                            StructCommand::MoveSection(CommandMoveSection {
                                old_parent_id: 1,
                                new_parent_id: 0,
                                section_id: 3,
                                new_name: "C2".to_string(),
                            }),
                            StructCommand::ReorderChildren(CommandReorderChildren {
                                section_id: 0,
                                children_list_id: 0,
                                children_list: vec![3, 1],
                            }),
                            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                                parent_id: 2,
                                section_id: 4,
                                section_name: "D".to_string(),
                                section_type: 1,
                                section_order: 2,
                            }),
                        ],
                        vec![inline_field(4, "d", 4, 1)],
                    ),
                    &[],
                )),
            ),
            (
                "cascade",
                Some(ompf(
                    &plain(
                        vec![
                            StructCommand::DeleteSection(CommandDeleteSection {
                                parent_id: 0,
                                section_id: 1,
                            }),
                            create(3, 5, "E"),
                        ],
                        vec![],
                    ),
                    &[],
                )),
            ),
            (
                "field commit with an ignored DiffDeleteSection",
                Some(ompf(
                    &plain(
                        vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
                            section_id: 5,
                        })],
                        vec![inline_field(5, "e", 5, 3)],
                    ),
                    &[],
                )),
            ),
            (
                "reset",
                Some(ompf(
                    &OwnedDelta {
                        is_diff: true,
                        base_version: 0,
                        source_format: "usda".to_string(),
                        struct_commands: vec![create(0, 10, "X")],
                        set_fields: vec![external_field(10, "x")],
                        ..Default::default()
                    },
                    &[b"hello"],
                )),
            ),
        ]
    }

    /// The decode then apply steps land, record by record, exactly where
    /// every `replay_stream_record*` entry point does: the same state, the
    /// same planned mutations through a collecting sink, and on the compact
    /// form with plan indexes the same engine counters, touches and indexes
    /// (their keys, depth flag and recomputed weight). Each entry point is
    /// run on its own copy of the state the split steps run on, so a step
    /// that did more or less than its entry point shows as a difference.
    #[test]
    fn decode_then_apply_matches_every_entry_point() {
        let start = ObjectState {
            object_id: OBJECT_ID,
            ..Default::default()
        };
        let (mut by_entry_point, mut by_steps) = (start.clone(), start.clone());
        let mut by_discarding_entry_point = start.clone();
        let (mut compact_entry_point, mut compact_steps) = (compact(&start), compact(&start));
        let (mut indexes_entry_point, mut indexes_steps) = (
            PlanIndexes::build(&compact_entry_point),
            PlanIndexes::build(&compact_steps),
        );
        for (i, (label, entry)) in lineage().into_iter().enumerate() {
            let seq = i as u64 + 1;
            let ompf = entry.as_deref();

            let mut sink = MutationSink::collecting();
            replay_stream_record_into(&mut by_entry_point, ompf, seq, &mut sink)
                .unwrap_or_else(|e| panic!("{label}: the entry point replays: {e}"));
            let mut steps_sink = MutationSink::collecting();
            let record = decode_stream_record(by_steps.sequence_num, ompf, seq)
                .unwrap_or_else(|e| panic!("{label}: decodes: {e}"));
            assert_eq!(record.seq(), seq, "{label}");
            apply_decoded_record_into(&mut by_steps, record, &mut steps_sink)
                .unwrap_or_else(|e| panic!("{label}: applies: {e}"));
            assert_eq!(by_steps, by_entry_point, "{label}: the state");
            let (MutationSink::Collect(steps), MutationSink::Collect(entry_point)) =
                (steps_sink, sink)
            else {
                panic!("{label}: collecting sinks");
            };
            assert_eq!(steps, entry_point, "{label}: the planned mutations");

            replay_stream_record(&mut by_discarding_entry_point, ompf, seq)
                .unwrap_or_else(|e| panic!("{label}: the discarding entry point replays: {e}"));
            assert_eq!(
                by_discarding_entry_point, by_steps,
                "{label}: the discarding entry point's state"
            );

            let (mut touches_entry_point, mut touches_steps) =
                (PlanTouches::default(), PlanTouches::default());
            let stats_entry_point = replay_stream_record_indexed_with_stats(
                &mut compact_entry_point,
                ompf,
                seq,
                &mut MutationSink::Discard,
                &mut indexes_entry_point,
                &mut touches_entry_point,
            )
            .unwrap_or_else(|e| panic!("{label}: the indexed entry point replays: {e}"));
            let record = decode_stream_record(compact_steps.sequence_num, ompf, seq)
                .unwrap_or_else(|e| panic!("{label}: decodes: {e}"));
            let stats_steps = apply_decoded_record_indexed(
                &mut compact_steps,
                record,
                &mut MutationSink::Discard,
                &mut indexes_steps,
                &mut touches_steps,
            )
            .unwrap_or_else(|e| panic!("{label}: applies indexed: {e}"));
            assert_eq!(
                compact_steps, compact_entry_point,
                "{label}: the compact state"
            );
            assert_eq!(
                stats_steps, stats_entry_point,
                "{label}: the engine's counters"
            );
            assert_eq!(
                (
                    &touches_steps.listed,
                    &touches_steps.lists,
                    touches_steps.reset
                ),
                (
                    &touches_entry_point.listed,
                    &touches_entry_point.lists,
                    touches_entry_point.reset
                ),
                "{label}: the touches"
            );
            let summary = |indexes: &PlanIndexes| {
                (
                    indexes.parent_keys(),
                    indexes.hash_keys(),
                    indexes.depth_ok(),
                    indexes.weight_bytes_recomputed(),
                )
            };
            assert_eq!(
                summary(&indexes_steps),
                summary(&indexes_entry_point),
                "{label}: the plan indexes"
            );
        }
        assert_eq!(by_steps.sequence_num, lineage().len() as u64);
        assert_eq!(
            by_steps.sections.keys().collect::<Vec<_>>(),
            vec![&10],
            "non-vacuity: the lineage ended past its reset"
        );
    }

    /// The split steps fail as the entry points do, error by error — the
    /// variant, its fields and its class — and leave the state as the entry
    /// point leaves it: a `seq` that is no successor, an OMPF or OODF that
    /// does not decode (each refused by the decode, which touches no state:
    /// it takes only the state's `seq`), and a delta the engine rejects
    /// (refused by the apply). A record decoded against one state and applied
    /// to another that has moved on is refused by the apply's own successor
    /// check, the state untouched.
    #[test]
    fn decode_then_apply_fails_as_every_entry_point_does_by_class() {
        let self_parenting = OwnedDelta {
            struct_commands: vec![create(7, 7, "Loop")],
            ..Default::default()
        };
        let malformed_oodf = OmpfContainer::serialize(b"not an OODF flatbuffer", &[]).unwrap();
        let cases: [(&str, u64, Option<Vec<u8>>); 4] = [
            ("not the successor", 9, Some(ompf(&structural_delta(), &[]))),
            ("malformed OMPF", 5, Some(b"not an OMPF container".to_vec())),
            ("malformed OODF", 5, Some(malformed_oodf)),
            (
                "rejected by the engine",
                5,
                Some(ompf(&self_parenting, &[])),
            ),
        ];
        for (label, seq, entry) in cases {
            let mut by_entry_point = small_tree(4);
            let entry_point = replay_stream_record_into(
                &mut by_entry_point,
                entry.as_deref(),
                seq,
                &mut MutationSink::Discard,
            )
            .expect_err(label);

            let mut by_steps = small_tree(4);
            let steps = decode_stream_record(by_steps.sequence_num, entry.as_deref(), seq)
                .and_then(|record| {
                    apply_decoded_record_into(&mut by_steps, record, &mut MutationSink::Discard)
                })
                .expect_err(label);

            assert_eq!(
                format!("{steps:?}"),
                format!("{entry_point:?}"),
                "{label}: the same error"
            );
            assert_eq!(
                steps.class(),
                entry_point.class(),
                "{label}: the same class"
            );
            assert_eq!(by_steps, by_entry_point, "{label}: the same state after it");
            if steps.class() == ReplayErrorClass::Parse
                || matches!(steps, ReplayError::NotSuccessor { .. })
            {
                assert_eq!(
                    by_steps,
                    small_tree(4),
                    "{label}: the decode touched nothing"
                );
            }
        }

        let record = decode_stream_record(4, Some(&ompf(&structural_delta(), &[])), 5)
            .expect("decoded against seq 4");
        let mut moved_on = small_tree(5);
        let err = apply_decoded_record_into(&mut moved_on, record, &mut MutationSink::Discard)
            .unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::NotSuccessor {
                    seq: 5,
                    state_seq: 5
                }
            ),
            "the apply re-checks the successor; got {err:?}"
        );
        assert_eq!(moved_on, small_tree(5), "and touches nothing");
    }
}
