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

//! The restore and accept-loss Jobs' lineage selection (SDD SR-9.6 steps 1–2;
//! PD-6, defer-3): one object's newest restorable lineage over the listing the
//! Job already took, decoded to learn what its state references, then judged
//! by the Job's absence policy ([`super::decide::judge_lineage`]).
//!
//! The walk is the shared one (`checkpoint::lineage`), started over the
//! Job's listing ([`LineageSelection::from_listing`]) instead of a listing of
//! its own; the on-access callers are untouched. What differs is the policy
//! around it, all here and in the pure judge:
//!
//! - a lineage at or below the survivor's `seq` is never fetched past;
//! - a decode or compose failure of bytes the adapter returned steps down
//!   ([`LineageSelection::record_compose_failure`]), except, for restore, a
//!   plain OMPF base over the per-delta entry limit (defer-3), which ends the
//!   walk with no step-down: compose.rs's "a new caller that steps down must
//!   check for this case first". Accept-loss checks it and steps down past it
//!   on purpose (PD-4, [`OverLimitPolicy`]);
//! - a backup-store error ends the walk as `Errored`, never a step-down: the
//!   walk propagates every error but the missing-meta `Permanent`
//!   (`LineageSelection::fetch_member`), so a transient fetch error is
//!   unknown (PD-4), not a reason to restore an older generation;
//! - a chunked (LEBC) base decodes and is used; the on-access hook's refusal
//!   of it does not apply to the Job.
//!
//! Bytes and the decoded state are dropped before the next object; the plan
//! keeps hash sets and the lineage's footprint (its fetched bytes, its
//! state's weight and its restore's command count) only. The decode is
//! [`decode_planned`], which the restore writer runs again on the re-fetched
//! bytes. In the planner's lineage stage each decode is admitted by the
//! Job's restore memory first ([`run_selection_within`]).

use std::collections::{BTreeSet, HashSet};
use std::time::Duration;

use bytes::Bytes;

use super::decide::judge_lineage;
use super::model::{
    ExtHash, LineageFootprint, LineageOutcome, LineageVerdict, OverLimitPolicy, PlannedLineage,
    SelectedLineage,
};
use super::report::referenced_hashes;
use crate::checkpoint::apply::{decode_checkpoint_ompf, is_over_entry_limit_base};
use crate::checkpoint::compose::{compose_from_artifacts, ComposeError, ExtValuePart};
use crate::checkpoint::lineage::{
    no_deadline, Exhaustion, GatheredLineage, LineageSelection, StepDownRecord,
};
use crate::checkpoint::recovery::adjudicate_composed_meta;
use crate::checkpoint::store::{CheckpointHandle, CheckpointKind, CheckpointStore};
use crate::domain::limits::MAX_DELTA_ENTRIES;
use crate::domain::types::{FieldValue, ObjectState};
use crate::dr::job::execute::memory::{over_estimate, plan_decode_charge, RestoreGate};
use crate::dr::job::{JobExit, JobMode};
use crate::storage::external_values::hash_from_hex;
use crate::storage::layer::dr_restore::restore_object_command_count;
use crate::storage::structural_cache::estimate_state_bytes;

/// The shortest per-fetch reserve the Job's selection runs with.
pub const SELECTION_RESERVE_MIN: Duration = Duration::from_secs(600);
/// The longest.
pub const SELECTION_RESERVE_MAX: Duration = Duration::from_secs(6 * 60 * 60);

/// The Job's per-fetch reserve: twice the adapter's call ceiling, clamped to
/// [`SELECTION_RESERVE_MIN`, `SELECTION_RESERVE_MAX`], so the adapter's own
/// ceiling, not the walk's per-fetch cap, ends a slow fetch; and the walk's
/// precondition `budget >= 2 x reserve` holds under `no_deadline()`'s 24 h.
pub fn selection_reserve(adapter_call_ceiling: Duration) -> Duration {
    adapter_call_ceiling
        .saturating_mul(2)
        .clamp(SELECTION_RESERVE_MIN, SELECTION_RESERVE_MAX)
}

/// One object's walk, before the absence policy: what it ended with and
/// every candidate it abandoned.
#[derive(Clone, Debug)]
pub struct SelectionRun {
    /// What the walk ended with.
    pub outcome: LineageOutcome,
    /// Every candidate abandoned, oldest abandonment first.
    pub step_downs: Vec<StepDownRecord>,
    /// Of those, the plain bases over the per-delta entry limit the walk
    /// stepped past (accept-loss, PD-4), `(head_seq, base_seq)`, in walk
    /// order; always empty for restore, whose walk ends on the first.
    pub over_limit: Vec<(u64, u64)>,
}

/// Select and judge one object's lineage: [`run_selection`], then
/// [`judge_lineage`] with `at_or_after_t_safe` (the store's clock, read just
/// before the selection).
pub async fn select(
    store: &dyn CheckpointStore,
    object_id: u64,
    handles: Vec<CheckpointHandle>,
    floor: Option<u64>,
    at_or_after_t_safe: bool,
    reserve: Duration,
) -> Result<LineageVerdict, JobExit> {
    select_within(
        store,
        object_id,
        handles,
        floor,
        at_or_after_t_safe,
        reserve,
        None,
        JobMode::Restore,
    )
    .await
}

/// [`select`], each decode admitted by `gate` first (the planner's lineage
/// stage; `None` admits at once), for the Job running in `mode`: its log
/// lines name the mode, and a defer-3 over-limit plain base is treated by the
/// mode's [`OverLimitPolicy`] (restore holds; accept-loss steps down, PD-4).
#[allow(clippy::too_many_arguments)]
pub async fn select_within(
    store: &dyn CheckpointStore,
    object_id: u64,
    handles: Vec<CheckpointHandle>,
    floor: Option<u64>,
    at_or_after_t_safe: bool,
    reserve: Duration,
    gate: Option<&RestoreGate>,
    mode: JobMode,
) -> Result<LineageVerdict, JobExit> {
    let run = run_selection_for_mode(store, object_id, handles, floor, reserve, gate, mode).await?;
    let mut verdict = judge_lineage(at_or_after_t_safe, floor, &run.step_downs, run.outcome);
    // An exhaustion past over-limit bases names them, so accept-loss keeps
    // their backups (`super::lost`).
    if let LineageVerdict::BackupsUnusable { over_limit, .. } = &mut verdict {
        *over_limit = run.over_limit;
    }
    Ok(verdict)
}

/// Walk one object's listed generations newest first, without the absence
/// policy:
///
/// 1. no handles: `Exhausted(EmptyListing)`, no fetch;
/// 2. `floor` set and the newest handle at or below it: `NotNewer`, no fetch;
/// 3. otherwise walk [`LineageSelection::from_listing`] under
///    `no_deadline()` and `reserve`, decoding each restorable lineage
///    (`spawn_blocking`): a lineage at or below `floor` is `NotNewer`
///    undecoded; a plain over-limit base is `OverLimitPlainBase` (no step
///    down); any other decode or compose failure steps down; a store error
///    is `Errored`; a decoded lineage is `Selected`.
///
/// `Err` only for a decode task that failed to join, a defect (logged).
pub async fn run_selection(
    store: &dyn CheckpointStore,
    object_id: u64,
    handles: Vec<CheckpointHandle>,
    floor: Option<u64>,
    reserve: Duration,
) -> Result<SelectionRun, JobExit> {
    run_selection_within(store, object_id, handles, floor, reserve, None).await
}

/// [`run_selection`], each decode admitted by `gate` first: after a lineage's
/// bytes are fetched and before they are decoded, the walk reserves
/// `execute::memory::plan_decode_charge(fetched)`, `fetched × (1 +
/// DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE) +
/// DR_PLAN_DECODE_CHARGE_FIXED_BYTES`, and holds it until the decode returns (SDD §22.22, *Restore memory*). A decoded state
/// above the charge's allowance is logged (`execute::memory::over_estimate`).
/// It runs as the restore Job's: a plain over-limit base ends the walk
/// ([`OverLimitPolicy::Hold`]).
pub async fn run_selection_within(
    store: &dyn CheckpointStore,
    object_id: u64,
    handles: Vec<CheckpointHandle>,
    floor: Option<u64>,
    reserve: Duration,
    gate: Option<&RestoreGate>,
) -> Result<SelectionRun, JobExit> {
    run_selection_for_mode(
        store,
        object_id,
        handles,
        floor,
        reserve,
        gate,
        JobMode::Restore,
    )
    .await
}

/// [`run_selection_within`] for the Job running in `mode`: its log lines
/// name the mode, and a plain over-limit base (defer-3) is treated by the
/// mode's [`OverLimitPolicy`]. [`OverLimitPolicy::Hold`] (restore) ends the
/// walk with `OverLimitPlainBase` and no step-down;
/// [`OverLimitPolicy::StepDown`] (accept-loss, PD-4) logs a WARN and steps
/// down past it, like any other failed decode of bytes the adapter returned,
/// so the walk goes on to the newest usable older generation.
pub async fn run_selection_for_mode(
    store: &dyn CheckpointStore,
    object_id: u64,
    handles: Vec<CheckpointHandle>,
    floor: Option<u64>,
    reserve: Duration,
    gate: Option<&RestoreGate>,
    mode: JobMode,
) -> Result<SelectionRun, JobExit> {
    let over_limit = OverLimitPolicy::for_mode(mode);
    let Some(newest) = handles.first() else {
        return Ok(SelectionRun {
            outcome: LineageOutcome::Exhausted(Exhaustion::EmptyListing),
            step_downs: Vec::new(),
            over_limit: Vec::new(),
        });
    };
    if floor.is_some_and(|f| newest.seq() <= f) {
        return Ok(SelectionRun {
            outcome: LineageOutcome::NotNewer,
            step_downs: Vec::new(),
            over_limit: Vec::new(),
        });
    }
    let mut selection =
        LineageSelection::from_listing(store, object_id, handles, no_deadline(), reserve);
    let mut over_limit_bases: Vec<(u64, u64)> = Vec::new();
    loop {
        let next = selection.next_restorable().await;
        let lineage = match next {
            Err(e) => {
                return Ok(SelectionRun {
                    outcome: LineageOutcome::Errored {
                        error: e.to_string(),
                    },
                    step_downs: selection.step_downs().to_vec(),
                    over_limit: over_limit_bases,
                })
            }
            Ok(None) => {
                let exhaustion = selection
                    .exhaustion()
                    .cloned()
                    .unwrap_or(Exhaustion::AllBroken { candidates: 0 });
                return Ok(SelectionRun {
                    outcome: LineageOutcome::Exhausted(exhaustion),
                    step_downs: selection.step_downs().to_vec(),
                    over_limit: over_limit_bases,
                });
            }
            Ok(Some(lineage)) => lineage,
        };
        if floor.is_some_and(|f| lineage.head_seq <= f) {
            return Ok(SelectionRun {
                outcome: LineageOutcome::NotNewer,
                step_downs: selection.step_downs().to_vec(),
                over_limit: over_limit_bases,
            });
        }
        let head_seq = lineage.head_seq;
        let head_kind = if lineage.is_base_only() {
            CheckpointKind::Base
        } else {
            CheckpointKind::Tail
        };
        let base_seq = lineage.base_snap_seq();
        let (planned, base_bytes, tail_bytes) = split_lineage(lineage);
        let fetched_bytes = fetched_len(&base_bytes, &tail_bytes);
        let admission = match gate {
            Some(gate) => match gate.admit(plan_decode_charge(fetched_bytes)).await {
                Some(admission) => Some(admission),
                None => {
                    tracing::error!(
                        object_id,
                        "the {mode} Job's memory budget refused a decode; this is an internal \
                         defect; it exits 1"
                    );
                    return Err(JobExit::Failed);
                }
            },
            None => None,
        };
        let decoded = tokio::task::spawn_blocking(move || {
            decode_lineage(object_id, base_seq, &base_bytes, &tail_bytes)
        })
        .await;
        drop(admission);
        let decoded = match decoded {
            Ok(decoded) => decoded,
            Err(join_err) => {
                tracing::error!(
                    object_id,
                    head_seq,
                    error = %join_err,
                    "the {mode} Job's decode of a backup failed to join; this is an internal \
                     defect, not a broken backup; it exits 1"
                );
                return Err(JobExit::Failed);
            }
        };
        match decoded {
            Decoded::OverLimit if over_limit == OverLimitPolicy::Hold => {
                return Ok(SelectionRun {
                    outcome: LineageOutcome::OverLimitPlainBase { head_seq, base_seq },
                    step_downs: selection.step_downs().to_vec(),
                    over_limit: over_limit_bases,
                })
            }
            // PD-4: accept-loss never discards a layer an older backup can
            // restore, so it steps down past the over-limit base (compose.rs's
            // "check for this case first": checked, and stepped down on
            // purpose).
            Decoded::OverLimit => {
                tracing::warn!(
                    object_id,
                    head_seq,
                    base_seq,
                    max_delta_entries = MAX_DELTA_ENTRIES,
                    "the accept-loss Job cannot read backup generation {head_seq} of object \
                     {object_id}: its full backup (generation {base_seq}) is over \
                     {MAX_DELTA_ENTRIES} entries and was written by an earlier release; it tries \
                     the next older generation"
                );
                over_limit_bases.push((head_seq, base_seq));
                selection.record_compose_failure(head_seq, head_kind);
            }
            Decoded::Failed(error) => {
                tracing::warn!(
                    object_id,
                    head_seq,
                    base_seq,
                    error = %error,
                    "the {mode} Job could not decode backup generation {head_seq} of object \
                     {object_id}; it tries the next older generation"
                );
                selection.record_compose_failure(head_seq, head_kind);
            }
            Decoded::Ok {
                state_hashes,
                tail_carried,
                state_bytes,
                commands,
                largest_inline_bytes,
            } => {
                over_estimate(object_id, fetched_bytes, state_bytes);
                let sidecar_extra = sidecar_extra(store, &planned.base, &state_hashes, mode).await;
                let planned = PlannedLineage {
                    tail_carried: tail_carried.into_boxed_slice(),
                    step_downs: selection.step_downs().to_vec(),
                    footprint: LineageFootprint {
                        fetched_bytes,
                        state_bytes,
                        commands,
                        largest_inline_bytes,
                    },
                    ..planned
                };
                return Ok(SelectionRun {
                    outcome: LineageOutcome::Selected(Box::new(SelectedLineage {
                        lineage: planned,
                        state_hashes,
                        sidecar_extra,
                    })),
                    step_downs: selection.step_downs().to_vec(),
                    over_limit: over_limit_bases,
                });
            }
        }
    }
}

/// A gathered lineage's handles and restore meta, and its bytes for the
/// decode. The restore meta is the base's, or for a composed lineage the
/// base's with the head tail's envelope span and write time
/// (`recovery::adjudicate_composed_meta`, as the recovery hook derives it).
fn split_lineage(lineage: GatheredLineage) -> (PlannedLineage, Bytes, Vec<Bytes>) {
    let GatheredLineage {
        base,
        tails,
        head_seq,
    } = lineage;
    let meta = match tails.last() {
        None => base.meta.clone(),
        Some(head) => adjudicate_composed_meta(
            base.meta.clone(),
            head.header.to_seq.saturating_sub(head.header.from_seq),
            head.meta.written_at_unix_ms,
        ),
    };
    let tail_handles = tails.iter().map(|tail| tail.handle.clone()).collect();
    let tail_bytes = tails.into_iter().map(|tail| tail.bytes).collect();
    (
        PlannedLineage {
            base: base.handle,
            tails: tail_handles,
            head_seq,
            meta,
            tail_carried: Box::new([]),
            step_downs: Vec::new(),
            footprint: LineageFootprint::default(),
        },
        base.bytes,
        tail_bytes,
    )
}

/// The bytes a lineage fetched: its base's and every tail's.
pub fn fetched_len(base: &[u8], tails: &[Bytes]) -> u64 {
    tails.iter().fold(base.len() as u64, |sum, tail| {
        sum.saturating_add(tail.len() as u64)
    })
}

/// What a lineage's decode learned, for the plan.
enum Decoded {
    /// It decoded: the hashes its state's rows reference, the hashes its
    /// tails carry, its state's weight, the commands its restore queues, and
    /// its largest inline value's bytes.
    Ok {
        state_hashes: BTreeSet<ExtHash>,
        tail_carried: Vec<ExtHash>,
        state_bytes: u64,
        commands: u64,
        largest_inline_bytes: u64,
    },
    /// defer-3: its base is a plain OMPF over the per-delta entry limit.
    OverLimit,
    /// Any other decode or compose failure: a bytes verdict.
    Failed(String),
}

/// A lineage's decoded state ([`decode_planned`]).
pub enum DecodedLineage {
    /// It decoded: the state at the lineage's head, the external values its
    /// tails carry (empty for a base-only lineage), and, for a composed
    /// lineage, the sequence the composition landed on.
    Ok {
        /// The state. `sequence_num` is 0 for a base-only decode, the
        /// composed head for a composed one.
        state: Box<ObjectState>,
        /// The values the tails carry, by hash, deduplicated.
        ext_values: Vec<ExtValuePart>,
        /// The composition's head, `None` for a base-only lineage.
        composed_seq: Option<u64>,
    },
    /// defer-3: its base is a plain OMPF over the per-delta entry limit.
    OverLimit,
    /// Any other decode or compose failure: a bytes verdict.
    Failed(String),
}

/// Decode a planned lineage's bytes (blocking): the base alone through
/// `decode_checkpoint_ompf` (LEBC-aware), a composed one through
/// `compose_from_artifacts`. The planner's decode and the restore writer's
/// re-decode both run it, so the writer restores exactly what the planner
/// judged.
pub fn decode_planned(
    object_id: u64,
    base_seq: u64,
    base: &[u8],
    tails: &[Bytes],
) -> DecodedLineage {
    if tails.is_empty() {
        return match decode_checkpoint_ompf(object_id, base) {
            Ok(state) => DecodedLineage::Ok {
                state: Box::new(state),
                ext_values: Vec::new(),
                composed_seq: None,
            },
            Err(e) if is_over_entry_limit_base(&e) => DecodedLineage::OverLimit,
            Err(e) => DecodedLineage::Failed(e.to_string()),
        };
    }
    match compose_from_artifacts(object_id, base, base_seq, tails) {
        Ok(composed) => DecodedLineage::Ok {
            state: Box::new(composed.state),
            ext_values: composed.ext_values,
            composed_seq: Some(composed.composed_seq),
        },
        Err(ComposeError::BaseDecode { source, .. }) if is_over_entry_limit_base(&source) => {
            DecodedLineage::OverLimit
        }
        Err(e) => DecodedLineage::Failed(e.to_string()),
    }
}

/// Decode a lineage's bytes (blocking) for the plan ([`decode_planned`]),
/// keeping what the plan needs of the state and dropping the state.
fn decode_lineage(object_id: u64, base_seq: u64, base: &[u8], tails: &[Bytes]) -> Decoded {
    match decode_planned(object_id, base_seq, base, tails) {
        DecodedLineage::Ok {
            state, ext_values, ..
        } => Decoded::Ok {
            state_hashes: referenced_hashes(&state),
            tail_carried: ext_values.iter().map(|(hash, _)| *hash).collect(),
            state_bytes: estimate_state_bytes(&state),
            commands: restore_object_command_count(object_id, &state),
            largest_inline_bytes: largest_inline_bytes(&state),
        },
        DecodedLineage::OverLimit => Decoded::OverLimit,
        DecodedLineage::Failed(error) => Decoded::Failed(error),
    }
}

/// The largest inline value of `state`, in bytes, over its fields and time
/// samples; 0 when it holds none.
fn largest_inline_bytes(state: &ObjectState) -> u64 {
    let inline = |value: &FieldValue| match value {
        FieldValue::Inline(bytes) => bytes.len() as u64,
        FieldValue::External { .. } => 0,
    };
    state
        .fields
        .values()
        .map(|field| inline(&field.value))
        .chain(
            state
                .time_samples
                .values()
                .map(|sample| inline(&sample.value)),
        )
        .max()
        .unwrap_or(0)
}

/// PD-2: how many hashes the base's ext-refs sidecar names that no row of
/// the decoded state references. A missing sidecar answers empty; a failed
/// read is logged and counts nothing, because these hashes never hold the
/// service.
async fn sidecar_extra(
    store: &dyn CheckpointStore,
    base: &CheckpointHandle,
    state_hashes: &BTreeSet<ExtHash>,
    mode: JobMode,
) -> usize {
    match store.read_ext_refs_sidecar(base).await {
        Ok(listed) => listed
            .iter()
            .filter_map(|hex| hash_from_hex(hex))
            .collect::<HashSet<ExtHash>>()
            .into_iter()
            .filter(|hash| !state_hashes.contains(hash))
            .count(),
        Err(e) => {
            tracing::warn!(
                object_id = base.object_id,
                generation = base.seq(),
                error = %e,
                "the {mode} Job could not read the external-value reference list of backup \
                 generation {} of object {}; the coverage check uses the backup's rows, which it \
                 read",
                base.seq(),
                base.object_id
            );
            0
        }
    }
}

#[cfg(test)]
mod tests {
    //! The Job's lineage selection over an in-memory backup store wrapped so
    //! a chosen generation's fetch answers absent, fails or never answers,
    //! and every fetch is counted (L1–L7, PD-6 condition 3); and the listing
    //! seam it walks (U8). A "nothing below it was fetched" claim is always a
    //! fetch count, never an inspection of the returned lineage.

    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::assembler::build_reset_ompf_from_state;
    use crate::checkpoint::chunked_base::tests::{chunked_base_of, plain_over_cap_base};
    use crate::checkpoint::lineage::{
        gather_restore_lineage, group_layer_generations, LineageBreak,
    };
    use crate::checkpoint::store::{
        CheckpointBlob, CheckpointError, CheckpointMeta, LayerArtifact, LayerArtifactKind,
        MissingMetaCompanion,
    };
    use crate::checkpoint::tail_codec::{encode_tail_envelope, TailHeader, TailRecord};
    use crate::checkpoint::LiveExtSet;
    use crate::domain::delta::{SPEC_ATTRIBUTE, SPEC_PRIM};
    use crate::domain::ompf::OmpfContainer;
    use crate::domain::types::{Field, FieldValue, ObjectState, Section};
    use crate::dr::job::plan::decide::{
        resolve, Fallback, LineageNeed, PreObject, PrePlan, PreStep,
    };
    use crate::dr::job::plan::model::{HeldReason, ObjectAction, RestoreReason};

    const OBJECT: u64 = 4_401;
    const SESSION: &str = "sess-plan";
    const LAYER: &str = "plan.usda";
    /// The external value the fixture layer's `points` attribute holds.
    const HASH: ExtHash = [0x5a; 32];
    /// A reserve no healthy fetch here comes near.
    const RESERVE: Duration = Duration::from_secs(600);

    // ── The backup store ────────────────────────────────────────────────

    /// What a scripted generation's fetch does instead of answering.
    #[derive(Clone, Copy, Debug)]
    enum Fault {
        /// `Ok(None)`: the listed generation reads as absent.
        Absent,
        /// The typed missing-meta `Permanent`.
        MissingMeta,
        /// A `Transient` after the adapter's own retries.
        Transient,
        /// A `Permanent` that is not the missing-meta one (a 403).
        Permanent,
        /// No answer at all.
        Hangs,
    }

    /// An in-memory backup store whose scripted generations fault, counting
    /// every `get_layer_checkpoint`.
    struct Scripted {
        inner: InMemoryCheckpointStore,
        faults: HashMap<(u64, CheckpointKind), Fault>,
        gets: Mutex<Vec<(u64, CheckpointKind)>>,
    }

    impl Scripted {
        fn new(inner: InMemoryCheckpointStore) -> Self {
            Self {
                inner,
                faults: HashMap::new(),
                gets: Mutex::new(Vec::new()),
            }
        }

        fn with(mut self, seq: u64, kind: CheckpointKind, fault: Fault) -> Self {
            self.faults.insert((seq, kind), fault);
            self
        }

        fn gets_of(&self, seq: u64, kind: CheckpointKind) -> usize {
            self.gets
                .lock()
                .unwrap()
                .iter()
                .filter(|g| **g == (seq, kind))
                .count()
        }

        fn total_gets(&self) -> usize {
            self.gets.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl CheckpointStore for Scripted {
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
            self.gets.lock().unwrap().push((handle.seq(), handle.kind));
            match self.faults.get(&(handle.seq(), handle.kind)) {
                None => self.inner.get_layer_checkpoint(handle).await,
                Some(Fault::Absent) => Ok(None),
                Some(Fault::MissingMeta) => Err(CheckpointError::missing_meta_companion(
                    MissingMetaCompanion {
                        artifact: format!("{}.{}.ompf", handle.object_id, handle.seq()),
                        companion: format!("{}.{}.meta.json", handle.object_id, handle.seq()),
                        object_id: handle.object_id,
                        sequence: handle.seq(),
                        kind: handle.kind,
                    },
                )),
                Some(Fault::Transient) => Err(CheckpointError::Transient {
                    source: "the backup store answered 503".into(),
                }),
                Some(Fault::Permanent) => Err(CheckpointError::Permanent {
                    source: "the backup store answered 403".into(),
                }),
                Some(Fault::Hangs) => std::future::pending().await,
            }
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
        async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            self.inner.get_ext_blob(hash_hex).await
        }
        async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
            self.inner.ext_blob_exists(hash_hex).await
        }
        async fn verify_list_route(&self) -> Result<(), CheckpointError> {
            self.inner.verify_list_route().await
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
        async fn list_referenced_ext_hashes(
            &self,
        ) -> Result<std::collections::HashSet<String>, CheckpointError> {
            self.inner.list_referenced_ext_hashes().await
        }
        async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
            self.inner.list_live_ext_set().await
        }
        async fn read_ext_refs_sidecar(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<std::collections::HashSet<String>, CheckpointError> {
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

    // ── Fixtures ────────────────────────────────────────────────────────

    /// The fixture layer: prim `/World` with an attribute `points` whose
    /// `default` field is the external value [`HASH`], and an inline field.
    fn layer_state() -> ObjectState {
        let section = |section_id, parent_id, name: &str, section_type| Section {
            section_id,
            parent_id,
            name: name.to_owned(),
            section_type,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        };
        let field = |key: &str, value| Field {
            section_id: 2,
            key_name: key.to_owned(),
            value,
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        };
        let mut state = ObjectState {
            object_id: OBJECT,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        for s in [
            section(1, 0, "World", SPEC_PRIM),
            section(2, 1, "points", SPEC_ATTRIBUTE),
        ] {
            state.sections.insert(s.section_id, s);
        }
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        for f in [
            field(
                "default",
                FieldValue::External {
                    hash: HASH,
                    size: 4_096,
                    ext_value_index: 1,
                },
            ),
            field("interpolation", FieldValue::Inline(b"vertex".to_vec())),
        ] {
            state.fields.insert((f.section_id, f.key_name.clone()), f);
        }
        state
    }

    /// The fixture layer's base, as the assembler writes it.
    fn valid_base() -> Vec<u8> {
        build_reset_ompf_from_state(&layer_state())
            .expect("the fixture base assembles")
            .0
    }

    fn base_meta(seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "plan-lineage-test".to_owned(),
            written_at_unix_ms: 1_700_000_000_000 + seq,
            snap_seq: Some(seq),
            session_id: SESSION.to_owned(),
            layer_name: LAYER.to_owned(),
            created_by: "user".to_owned(),
            server_ts: 1,
            observed_commit_span: 3,
            kind: CheckpointKind::Base,
            base_snap_seq: None,
            from_seq: None,
            ..CheckpointMeta::default()
        }
    }

    async fn put_base(store: &InMemoryCheckpointStore, object_id: u64, seq: u64, bytes: Vec<u8>) {
        store
            .put_layer_checkpoint(object_id, seq, Bytes::from(bytes), base_meta(seq))
            .await
            .expect("the base put succeeds");
    }

    /// A tail over `(from, to]` on the base at `base_seq`; each record an
    /// empty ordinary commit.
    async fn put_tail(
        store: &InMemoryCheckpointStore,
        object_id: u64,
        base_seq: u64,
        from: u64,
        to: u64,
    ) {
        let record = Bytes::from(
            OmpfContainer::serialize(
                &oodf_ffi::build_delta(&oodf_ffi::OwnedDelta::default())
                    .expect("an empty delta builds"),
                &[],
            )
            .expect("the record serializes"),
        );
        let records: Vec<TailRecord> = ((from + 1)..=to)
            .map(|sequence_num| TailRecord::Verbatim {
                sequence_num,
                server_ts: 1,
                ompf: record.clone(),
            })
            .collect();
        let envelope = encode_tail_envelope(
            &TailHeader {
                object_id,
                base_snap_seq: base_seq,
                from_seq: from,
                to_seq: to,
            },
            records,
        )
        .expect("the tail envelope encodes");
        let meta = CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(base_seq),
            from_seq: Some(from),
            ..base_meta(to)
        };
        store
            .put_layer_checkpoint(object_id, to, envelope, meta)
            .await
            .expect("the tail put succeeds");
    }

    /// The object's generations as the Job lists them: one enumeration,
    /// grouped.
    async fn job_listing(store: &dyn CheckpointStore) -> Vec<CheckpointHandle> {
        group_layer_generations(&store.list_layer_artifacts().await.expect("the listing"))
            .remove(&OBJECT)
            .unwrap_or_default()
    }

    /// An older valid base at 10 under a newest base at 20 of `newest` bytes.
    async fn two_bases(newest: Vec<u8>) -> InMemoryCheckpointStore {
        let inner = InMemoryCheckpointStore::new();
        put_base(&inner, OBJECT, 10, valid_base()).await;
        put_base(&inner, OBJECT, 20, newest).await;
        inner
    }

    fn selected_of(outcome: &LineageOutcome) -> &SelectedLineage {
        match outcome {
            LineageOutcome::Selected(selected) => selected,
            other => panic!("expected a selected lineage, got {}", outcome_word(other)),
        }
    }

    fn outcome_word(outcome: &LineageOutcome) -> String {
        match outcome {
            LineageOutcome::Selected(s) => format!("selected {}", s.lineage.head_seq),
            LineageOutcome::NotNewer => "not newer".to_owned(),
            LineageOutcome::Exhausted(e) => format!("exhausted ({e:?})"),
            LineageOutcome::OverLimitPlainBase { head_seq, base_seq } => {
                format!("over the limit (head {head_seq}, base {base_seq})")
            }
            LineageOutcome::Errored { error } => format!("errored ({error})"),
        }
    }

    // ── defer-3 (L1, L2) and its positive controls (L3, L4) ─────────────

    /// L1, defer-3 on the base-only arm: the newest base is a plain OMPF over
    /// the per-delta entry limit, which an earlier release wrote; an older
    /// base is valid. The walk ends `OverLimitPlainBase` with no step-down
    /// and never fetches the older base: the Job does not fall back to it.
    /// Building the over-limit base takes about a second in a debug build.
    #[tokio::test]
    async fn run_selection_stops_at_a_plain_over_limit_base_without_stepping_down() {
        let store = Scripted::new(two_bases(plain_over_cap_base()).await);
        let handles = job_listing(&store).await;

        let run = run_selection(&store, OBJECT, handles.clone(), None, RESERVE)
            .await
            .expect("no decode join failure");

        assert!(
            matches!(
                run.outcome,
                LineageOutcome::OverLimitPlainBase {
                    head_seq: 20,
                    base_seq: 20
                }
            ),
            "{}",
            outcome_word(&run.outcome)
        );
        assert!(run.step_downs.is_empty(), "{:?}", run.step_downs);
        assert_eq!(
            store.gets_of(10, CheckpointKind::Base),
            0,
            "nothing below the over-limit base is fetched"
        );
        let verdict = select(&store, OBJECT, handles, None, true, RESERVE)
            .await
            .expect("no decode join failure");
        assert!(matches!(
            verdict,
            LineageVerdict::OverLimitPlainBase {
                head_seq: 20,
                base_seq: 20
            }
        ));
    }

    /// L2, defer-3 on the composed arm: a tail over the over-limit base is
    /// the newest lineage. It ends `OverLimitPlainBase` naming the tail's
    /// head and the base, with no step-down and no fetch of the older base.
    #[tokio::test]
    async fn run_selection_stops_at_a_tail_over_a_plain_over_limit_base_without_stepping_down() {
        let inner = two_bases(plain_over_cap_base()).await;
        put_tail(&inner, OBJECT, 20, 20, 22).await;
        let store = Scripted::new(inner);
        let handles = job_listing(&store).await;

        let run = run_selection(&store, OBJECT, handles, None, RESERVE)
            .await
            .expect("no decode join failure");

        assert!(
            matches!(
                run.outcome,
                LineageOutcome::OverLimitPlainBase {
                    head_seq: 22,
                    base_seq: 20
                }
            ),
            "{}",
            outcome_word(&run.outcome)
        );
        assert!(run.step_downs.is_empty(), "{:?}", run.step_downs);
        assert_eq!(store.gets_of(10, CheckpointKind::Base), 0);
    }

    /// L1-AL, PD-4 (fail-on-old): accept-loss steps down past the plain
    /// over-limit newest base that L1 holds on, to the older valid base,
    /// recording the step-down, so a layer an older backup can restore is
    /// never discarded; at or after `T_safe` its verdict is to use the
    /// older base. The same listing still holds under restore (L1's own
    /// assertion, repeated here as this test's control).
    #[tokio::test]
    async fn run_selection_for_accept_loss_steps_down_past_a_plain_over_limit_base_to_the_older_one(
    ) {
        let store = Scripted::new(two_bases(plain_over_cap_base()).await);
        let handles = job_listing(&store).await;

        let run = run_selection_for_mode(
            &store,
            OBJECT,
            handles.clone(),
            None,
            RESERVE,
            None,
            JobMode::AcceptLoss,
        )
        .await
        .expect("no decode join failure");

        let selected = selected_of(&run.outcome);
        assert_eq!(selected.lineage.head_seq, 10, "the older base is restored");
        assert_eq!(
            run.step_downs
                .iter()
                .map(|r| (r.head_seq, r.reason.clone()))
                .collect::<Vec<_>>(),
            vec![(20, LineageBreak::Compose { head_seq: 20 })],
            "the over-limit base is stepped down from, recorded"
        );
        assert_eq!(selected.state_hashes, BTreeSet::from([HASH]));
        let verdict = select_within(
            &store,
            OBJECT,
            handles.clone(),
            None,
            true,
            RESERVE,
            None,
            JobMode::AcceptLoss,
        )
        .await
        .expect("no decode join failure");
        assert!(
            matches!(&verdict, LineageVerdict::Use(s) if s.lineage.head_seq == 10),
            "accept-loss uses the older base"
        );
        let restore = select_within(
            &store,
            OBJECT,
            handles,
            None,
            true,
            RESERVE,
            None,
            JobMode::Restore,
        )
        .await
        .expect("no decode join failure");
        assert!(
            matches!(
                restore,
                LineageVerdict::OverLimitPlainBase {
                    head_seq: 20,
                    base_seq: 20
                }
            ),
            "restore still holds on it"
        );
    }

    /// L1-AL on the composed arm (L2's shape): a tail over the plain
    /// over-limit base is the newest lineage; accept-loss steps down past
    /// it, and past the over-limit base alone, to the older valid base.
    #[tokio::test]
    async fn run_selection_for_accept_loss_steps_down_past_a_tail_over_a_plain_over_limit_base() {
        let inner = two_bases(plain_over_cap_base()).await;
        put_tail(&inner, OBJECT, 20, 20, 22).await;
        let store = Scripted::new(inner);
        let handles = job_listing(&store).await;

        let run = run_selection_for_mode(
            &store,
            OBJECT,
            handles,
            None,
            RESERVE,
            None,
            JobMode::AcceptLoss,
        )
        .await
        .expect("no decode join failure");

        assert_eq!(selected_of(&run.outcome).lineage.head_seq, 10);
        assert_eq!(
            run.step_downs.first().map(|r| r.head_seq),
            Some(22),
            "the composed lineage over the over-limit base is stepped down from first: {:?}",
            run.step_downs
        );
    }

    /// L3, L1's positive control (PD-6): a corrupt newest base that is not
    /// over the limit is a bytes verdict, and the walk steps down to the
    /// older base, recording the compose failure — before and at `T_safe`.
    #[tokio::test]
    async fn run_selection_steps_down_past_a_corrupt_base_to_the_older_one() {
        let store = Scripted::new(two_bases(b"not an OMPF".to_vec()).await);
        let handles = job_listing(&store).await;

        let run = run_selection(&store, OBJECT, handles.clone(), None, RESERVE)
            .await
            .expect("no decode join failure");

        let selected = selected_of(&run.outcome);
        assert_eq!(selected.lineage.head_seq, 10);
        assert_eq!(
            run.step_downs
                .iter()
                .map(|r| (r.head_seq, r.reason.clone()))
                .collect::<Vec<_>>(),
            vec![(20, LineageBreak::Compose { head_seq: 20 })]
        );
        assert_eq!(selected.lineage.step_downs, run.step_downs);
        assert_eq!(selected.state_hashes, BTreeSet::from([HASH]));
        for at in [false, true] {
            let verdict = select(&store, OBJECT, handles.clone(), None, at, RESERVE)
                .await
                .expect("no decode join failure");
            assert!(
                matches!(&verdict, LineageVerdict::Use(s) if s.lineage.head_seq == 10),
                "at_or_after_t_safe={at}"
            );
        }
    }

    /// L4: a chunked base decodes and is selected, with the values its rows
    /// reference; the on-access refusal of chunked bases does not apply to
    /// the Job.
    #[tokio::test]
    async fn run_selection_selects_a_chunked_base() {
        let store = Scripted::new(two_bases(chunked_base_of(&layer_state(), 1)).await);
        let handles = job_listing(&store).await;

        let run = run_selection(&store, OBJECT, handles, None, RESERVE)
            .await
            .expect("no decode join failure");

        let selected = selected_of(&run.outcome);
        assert_eq!(selected.lineage.head_seq, 20);
        assert!(selected.lineage.is_base_only());
        assert_eq!(selected.state_hashes, BTreeSet::from([HASH]));
        assert!(run.step_downs.is_empty());
        assert_eq!(
            selected.lineage.meta.observed_commit_span, 3,
            "a base-only lineage restores with the base's own meta"
        );
    }

    // ── The absence rule (L5) and the floor (L6) ────────────────────────

    /// L5, the unit half of the decider's pre-`T_safe` condition: the newest
    /// base reads absent (or its meta companion is missing) and an older
    /// base is valid. Before `T_safe` the object is deferred, naming the
    /// absent generation, and the older lineage is not used; at `T_safe`
    /// the step-down stands and the older base is restored.
    #[tokio::test]
    async fn select_defers_an_absent_newest_generation_before_t_safe_and_steps_down_at_it() {
        for fault in [Fault::Absent, Fault::MissingMeta] {
            let store =
                Scripted::new(two_bases(valid_base()).await).with(20, CheckpointKind::Base, fault);
            let handles = job_listing(&store).await;

            let before = select(&store, OBJECT, handles.clone(), None, false, RESERVE)
                .await
                .expect("no decode join failure");
            assert!(
                matches!(before, LineageVerdict::Defer { seq: 20 }),
                "{fault:?} before T_safe: {before:?}"
            );

            let at = select(&store, OBJECT, handles, None, true, RESERVE)
                .await
                .expect("no decode join failure");
            match at {
                LineageVerdict::Use(selected) => {
                    assert_eq!(selected.lineage.head_seq, 10, "{fault:?}");
                    let reasons: Vec<&str> = selected
                        .lineage
                        .step_downs
                        .iter()
                        .map(|r| r.reason.reason_label())
                        .collect();
                    let want = match fault {
                        Fault::Absent => "stepped_down_link_broken",
                        _ => "stepped_down_missing_meta",
                    };
                    assert_eq!(reasons, vec![want], "{fault:?}");
                }
                other => panic!("{fault:?} at T_safe: expected the older base, got {other:?}"),
            }
        }
    }

    /// L6: a listing whose newest generation is at or below the survivor's
    /// `seq` is `NotNewer` without a single fetch.
    #[tokio::test]
    async fn run_selection_fetches_nothing_when_the_newest_generation_is_not_above_the_floor() {
        let store = Scripted::new(two_bases(valid_base()).await);
        let handles = job_listing(&store).await;

        for floor in [20, 25] {
            let run = run_selection(&store, OBJECT, handles.clone(), Some(floor), RESERVE)
                .await
                .expect("no decode join failure");
            assert!(
                matches!(run.outcome, LineageOutcome::NotNewer),
                "floor {floor}: {}",
                outcome_word(&run.outcome)
            );
        }
        assert_eq!(store.total_gets(), 0, "no generation is fetched");

        let run = run_selection(&store, OBJECT, vec![], None, RESERVE)
            .await
            .expect("no decode join failure");
        assert!(matches!(
            run.outcome,
            LineageOutcome::Exhausted(crate::checkpoint::lineage::Exhaustion::EmptyListing)
        ));
        assert_eq!(store.total_gets(), 0);
    }

    // ── Budget cuts (L7) and fetch errors (PD-6 condition 3) ────────────

    /// L7: a fetch that never answers within the per-fetch reserve is a
    /// `Budget` step-down, and the judge makes the object unknown, before
    /// and at `T_safe`, whatever the walk went on to find below it.
    #[tokio::test]
    async fn select_makes_a_budget_cut_unknown_rather_than_a_step_down() {
        let store = Scripted::new(two_bases(valid_base()).await).with(
            20,
            CheckpointKind::Base,
            Fault::Hangs,
        );
        let handles = job_listing(&store).await;
        let reserve = Duration::from_millis(50);

        let run = run_selection(&store, OBJECT, handles.clone(), None, reserve)
            .await
            .expect("no decode join failure");
        assert!(
            run.step_downs
                .iter()
                .any(|r| matches!(r.reason, LineageBreak::Budget { seq: 20, .. })),
            "the wedged fetch is a budget step-down: {:?}",
            run.step_downs
        );

        for at in [false, true] {
            let verdict = select(&store, OBJECT, handles.clone(), None, at, reserve)
                .await
                .expect("no decode join failure");
            assert!(
                matches!(verdict, LineageVerdict::Unknown { .. }),
                "at_or_after_t_safe={at}: {verdict:?}"
            );
        }
    }

    /// PD-6 condition 3 (decider pin): a backup-store error fetching the
    /// newest generation, after the adapter's retries, is never a step-down.
    /// The walk ends `Errored` with no step-down recorded and no fetch of the
    /// older valid base; the judge makes it unknown; the resolution holds it
    /// as unknown and writes only the survivor's fallback — never a restore
    /// of the older base.
    #[tokio::test]
    async fn a_fetch_error_on_the_newest_generation_is_unknown_and_never_a_step_down() {
        for fault in [Fault::Transient, Fault::Permanent] {
            let store =
                Scripted::new(two_bases(valid_base()).await).with(20, CheckpointKind::Base, fault);
            let handles = job_listing(&store).await;

            let run = run_selection(&store, OBJECT, handles.clone(), None, RESERVE)
                .await
                .expect("no decode join failure");
            assert!(
                matches!(run.outcome, LineageOutcome::Errored { .. }),
                "{fault:?}: {}",
                outcome_word(&run.outcome)
            );
            assert!(run.step_downs.is_empty(), "{fault:?}: {:?}", run.step_downs);
            assert_eq!(store.gets_of(10, CheckpointKind::Base), 0, "{fault:?}");

            let verdict = select(&store, OBJECT, handles.clone(), None, true, RESERVE)
                .await
                .expect("no decode join failure");
            let LineageVerdict::Unknown { error } = &verdict else {
                panic!("{fault:?}: expected unknown, got {verdict:?}");
            };
            assert!(error.contains("answered"), "{fault:?}: {error}");

            for (floor, fallback, want) in [
                (None, Fallback::NoSurvivor, "nothing"),
                (Some(5), Fallback::Keep, "keep"),
            ] {
                let pre = PrePlan {
                    objects: vec![PreObject {
                        object_id: OBJECT,
                        session_id: SESSION.to_owned(),
                        layer_name: LAYER.to_owned(),
                        step: PreStep::NeedsLineage(LineageNeed {
                            handles: handles.clone(),
                            floor,
                            reason: RestoreReason::HeaderAbsent,
                            fallback,
                        }),
                    }],
                    ..PrePlan::default()
                };
                let resolved = resolve(&pre, &BTreeMap::from([(OBJECT, verdict.clone())]));
                let action = match &resolved.objects[0].action {
                    ObjectAction::Nothing => "nothing",
                    ObjectAction::Keep => "keep",
                    other => panic!("{fault:?}: never a restore of the older base: {other:?}"),
                };
                assert_eq!(action, want, "{fault:?}");
                assert_eq!(
                    resolved
                        .held
                        .iter()
                        .map(|h| h.reason.clone())
                        .collect::<Vec<_>>(),
                    vec![HeldReason::Unknown {
                        error: error.clone()
                    }],
                    "{fault:?}"
                );
            }
        }
    }

    // ── The budgeted decode (defer-45 (3), SR-12) ────────────────────────

    /// defer-45 (3): the walk admits a fetched lineage's decode through the
    /// Job's restore gate, charging `plan_decode_charge(fetched)` — the
    /// fetched bytes and `DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE` times them
    /// for the state about to be decoded, plus the fixed
    /// `DR_PLAN_DECODE_CHARGE_FIXED_BYTES` — rounded up to whole KiB and
    /// clamped to the whole budget, and gives the charge back when the decode
    /// returns. The selected lineage's footprint records the fetched bytes
    /// and the largest inline value of its decoded state, the writer's inline
    /// term's input: the fixture's 6-byte inline field, not its 4,096-byte
    /// external value. The lineage stage of a plan pass (`plan::run_pass`)
    /// passes its gate here; its count bound is the integration tests'
    /// (`tests/integration/dr/restore_memory_test.rs`), and so is the gate
    /// `run_pass` hands it, read from the execution report's
    /// `planner_memory` within the default budget and within one that holds
    /// a single charge.
    #[tokio::test]
    async fn run_selection_within_charges_the_plan_decode_charge_before_the_decode() {
        use std::num::NonZeroUsize;

        use crate::config::memory_plan::{dr_restore_line_at, DR_RESTORE_SHARE_TEN_THOUSANDTHS};
        use crate::config::PlanInputs;
        use crate::dr::job::execute::memory::{
            RestoreMemory, DR_PLAN_DECODE_CHARGE_FIXED_BYTES,
            DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE,
        };

        const KIB: u64 = 1024;
        let line = dr_restore_line_at(
            5_120 * KIB * KIB,
            DR_RESTORE_SHARE_TEN_THOUSANDTHS,
            &PlanInputs::reference(),
        );
        let memory = RestoreMemory::Bytes {
            budget: line.restore_budget_bytes,
            parallelism: NonZeroUsize::new(2).expect("non-zero"),
            line,
        };
        let store = Scripted::new(two_bases(valid_base()).await);
        let handles = job_listing(&store).await;
        let fetched = valid_base().len() as u64;
        let charge = plan_decode_charge(fetched).div_ceil(KIB) * KIB;
        assert_eq!(
            plan_decode_charge(fetched),
            (1 + DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE) * fetched
                + DR_PLAN_DECODE_CHARGE_FIXED_BYTES
        );
        let small = (charge / KIB / 2).max(1) * KIB;

        for (budget, held) in [(4 * charge, charge), (small, small)] {
            let gate = RestoreGate::new(&memory, Some(budget));
            let run =
                run_selection_within(&store, OBJECT, handles.clone(), None, RESERVE, Some(&gate))
                    .await
                    .expect("no decode join failure");

            let selected = selected_of(&run.outcome);
            assert_eq!(selected.lineage.head_seq, 20);
            assert_eq!(selected.lineage.footprint.fetched_bytes, fetched);
            assert_eq!(
                selected.lineage.footprint.largest_inline_bytes,
                b"vertex".len() as u64,
                "the largest inline value, the external one counting nothing"
            );
            assert_eq!(
                gate.peak_reserved_bytes(),
                held,
                "budget {budget} B: the decode held its plan_decode_charge, KiB-rounded, clamped"
            );
            assert_eq!(gate.max_in_flight(), 1, "budget {budget} B");
            assert_eq!(
                (gate.reserved_bytes(), gate.in_flight()),
                (0, 0),
                "budget {budget} B: the charge is given back when the decode returns"
            );
        }
    }

    /// The footprint's largest inline value is the longest inline value over
    /// the state's fields and its time samples alike, an external value
    /// counting nothing however large it is, and 0 for a state with no
    /// inline value: here a time sample's 9 bytes beat a field's 6, and a
    /// 1 MiB external value counts nothing.
    #[test]
    fn largest_inline_bytes_is_the_longest_inline_field_or_time_sample_value() {
        use ordered_float::OrderedFloat;

        use crate::domain::types::TimeSample;

        let external = FieldValue::External {
            hash: HASH,
            size: 1 << 20,
            ext_value_index: 1,
        };
        let sample = |time: f64, value: FieldValue| TimeSample {
            section_id: 2,
            time: OrderedFloat(time),
            value,
            version: 1,
        };
        let mut state = layer_state();
        assert_eq!(
            largest_inline_bytes(&state),
            6,
            "fixture: the 6-byte inline field, beside a 4,096-byte external one"
        );
        for (time, value) in [
            (1.0, FieldValue::Inline(vec![0; 9])),
            (2.0, external.clone()),
        ] {
            state
                .time_samples
                .insert((2, OrderedFloat(time)), sample(time, value));
        }
        assert_eq!(
            largest_inline_bytes(&state),
            9,
            "a time sample's value counts"
        );

        let mut only_external = layer_state();
        only_external
            .fields
            .retain(|_, f| matches!(f.value, FieldValue::External { .. }));
        only_external
            .time_samples
            .insert((2, OrderedFloat(1.0)), sample(1.0, external));
        assert_eq!(largest_inline_bytes(&only_external), 0, "no inline value");
        assert_eq!(
            largest_inline_bytes(&ObjectState::default()),
            0,
            "an empty state"
        );
    }

    /// The Job's per-fetch reserve: twice the adapter's call ceiling,
    /// clamped to [10 min, 6 h].
    #[test]
    fn selection_reserve_doubles_the_ceiling_within_its_clamp() {
        assert_eq!(
            selection_reserve(Duration::from_secs(180)),
            SELECTION_RESERVE_MIN
        );
        assert_eq!(
            selection_reserve(Duration::from_secs(3_600)),
            Duration::from_secs(7_200)
        );
        assert_eq!(
            selection_reserve(Duration::from_secs(24 * 3_600)),
            SELECTION_RESERVE_MAX
        );
    }

    // ── The listing seam (U8) ───────────────────────────────────────────

    fn artifact(object_id: u64, snap_seq: u64, kind: LayerArtifactKind) -> LayerArtifact {
        LayerArtifact {
            object_id,
            snap_seq,
            kind,
        }
    }

    /// U8: one raw enumeration groups into each object's generations,
    /// newest first with a base before a tail at an equal sequence. Only a
    /// family's generation artifact makes a generation: companions without
    /// it and write temps are left out, and a duplicate is one handle. Every
    /// handle carries an empty adapter key.
    #[test]
    fn group_layer_generations_orders_each_objects_generations_and_drops_non_generations() {
        use LayerArtifactKind::{ExtRefs, Meta, Ompf, TailEnvelope, TailMeta, Temp};
        let artifacts = [
            artifact(1, 3, Ompf),
            artifact(1, 5, TailMeta),
            artifact(1, 5, Ompf),
            artifact(1, 5, Meta),
            artifact(1, 5, ExtRefs),
            artifact(1, 7, TailEnvelope),
            artifact(1, 5, TailEnvelope),
            artifact(1, 3, Ompf),
            artifact(1, 9, Meta),
            artifact(1, 9, ExtRefs),
            artifact(1, 11, TailMeta),
            artifact(1, 13, Temp),
            artifact(2, 4, Temp),
            artifact(2, 4, Meta),
            artifact(3, 2, TailEnvelope),
        ];

        let grouped = group_layer_generations(&artifacts);

        let shape: BTreeMap<u64, Vec<(u64, CheckpointKind)>> = grouped
            .iter()
            .map(|(oid, handles)| (*oid, handles.iter().map(|h| (h.seq(), h.kind)).collect()))
            .collect();
        assert_eq!(
            shape,
            BTreeMap::from([
                (
                    1,
                    vec![
                        (7, CheckpointKind::Tail),
                        (5, CheckpointKind::Base),
                        (5, CheckpointKind::Tail),
                        (3, CheckpointKind::Base),
                    ]
                ),
                (3, vec![(2, CheckpointKind::Tail)]),
            ])
        );
        assert!(grouped
            .values()
            .flatten()
            .all(|h| h.adapter_key.is_empty() && grouped.contains_key(&h.object_id)));
    }

    /// One lineage, as a comparable shape: its head, its base and its tails.
    fn lineage_shape(lineage: &GatheredLineage) -> (u64, u64, Vec<u64>) {
        (
            lineage.head_seq,
            lineage.base_snap_seq(),
            lineage.tails.iter().map(|t| t.handle.seq()).collect(),
        )
    }

    /// Every lineage a selection offers, newest first, until it is drained.
    async fn drain(mut selection: LineageSelection<'_>) -> Vec<(u64, u64, Vec<u64>)> {
        let mut shapes = Vec::new();
        while let Some(lineage) = selection
            .next_restorable()
            .await
            .expect("the healthy store answers")
        {
            shapes.push(lineage_shape(&lineage));
        }
        shapes
    }

    /// U8 (pin): over the same store, the grouped listing is each object's
    /// own per-object listing by `(seq, kind)`, and a selection started over
    /// it offers exactly the lineages `gather_restore_lineage` offers.
    #[tokio::test]
    async fn from_listing_over_the_grouped_listing_selects_what_gather_restore_lineage_selects() {
        let inner = InMemoryCheckpointStore::new();
        put_base(&inner, OBJECT, 10, valid_base()).await;
        put_tail(&inner, OBJECT, 10, 10, 12).await;
        put_tail(&inner, OBJECT, 10, 12, 14).await;
        put_base(&inner, OBJECT, 20, valid_base()).await;
        put_tail(&inner, OBJECT, 20, 20, 25).await;
        put_base(&inner, OBJECT + 1, 1, valid_base()).await;

        let grouped =
            group_layer_generations(&inner.list_layer_artifacts().await.expect("the listing"));

        for object_id in [OBJECT, OBJECT + 1] {
            let per_object = inner
                .list_layer_checkpoints(object_id)
                .await
                .expect("the per-object listing");
            let from_group: Vec<(u64, CheckpointKind)> = grouped[&object_id]
                .iter()
                .map(|h| (h.seq(), h.kind))
                .collect();
            assert_eq!(
                from_group,
                per_object
                    .iter()
                    .map(|h| (h.seq(), h.kind))
                    .collect::<Vec<_>>(),
                "object {object_id}"
            );

            let gathered = drain(
                gather_restore_lineage(&inner, object_id, no_deadline(), RESERVE)
                    .await
                    .expect("the per-object listing"),
            )
            .await;
            let from_listing = drain(LineageSelection::from_listing(
                &inner,
                object_id,
                grouped[&object_id].clone(),
                no_deadline(),
                RESERVE,
            ))
            .await;
            assert_eq!(from_listing, gathered, "object {object_id}");
            assert!(
                !gathered.is_empty(),
                "object {object_id}: the store holds lineages"
            );
        }
        assert_eq!(
            drain(LineageSelection::from_listing(
                &inner,
                OBJECT,
                grouped[&OBJECT].clone(),
                no_deadline(),
                RESERVE,
            ))
            .await,
            vec![
                (25, 20, vec![25]),
                (20, 20, vec![]),
                (14, 10, vec![12, 14]),
                (12, 10, vec![12]),
                (10, 10, vec![])
            ],
            "newest first, every lineage once"
        );
    }
}
