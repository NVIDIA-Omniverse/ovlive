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

//! The restore plan's types (SDD SR-9.2 to SR-9.5): what the planner hands
//! the restore writer and what the recount and accept-loss's lost set
//! (`super::lost`) read. Pure data, with no I/O.

use std::collections::{BTreeMap, BTreeSet};

use crate::checkpoint::lineage::StepDownRecord;
use crate::checkpoint::session_payload::SessionCheckpoint;
use crate::checkpoint::store::{CheckpointHandle, CheckpointMeta};
use crate::dr::job::args::JobMode;

/// An external value's content hash, the key of the coverage check.
pub type ExtHash = [u8; 32];

/// When the plan's listing was taken, on the store's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanEpoch {
    /// The store-clock reading taken just before the listing the plan stands
    /// on; `None` with no backup store (nothing was listed).
    pub listing_taken_at_ms: Option<u64>,
    /// Whether that listing was taken at or after `T_safe`, so an absence in
    /// it is authoritative. `true` with no backup store: nothing was listed,
    /// so no absence was read from one and nothing waits for `T_safe`.
    pub at_or_after_t_safe: bool,
    /// Whether the plan stands on the listing re-taken at `T_safe` (a
    /// write-shaping absence was seen before it).
    pub relisted_at_t_safe: bool,
}

/// The restore plan: every session's and every mapped layer's action, the
/// coverage check, and what keeps the service held.
#[derive(Clone, Debug)]
pub struct RestorePlan {
    /// When the plan's listing was taken.
    pub epoch: PlanEpoch,
    /// One entry per session the plan names (SR-9.4).
    pub sessions: Vec<SessionPlan>,
    /// One entry per layer object mapped after the plan (SR-9.5), plus every
    /// remapped-from object.
    pub objects: Vec<ObjectPlan>,
    /// The coverage check (SR-9.3), finalised by
    /// [`CoveragePlan::finalize`](super::coverage).
    pub coverage: CoveragePlan,
    /// Write-shaping verdicts that keep the service held. A run that holds
    /// still writes everything writable.
    pub held: Vec<HeldItem>,
    /// Logged and summarised, never outcome-changing.
    pub advisories: Vec<Advisory>,
    /// The largest object id over every listed layer artifact, companions
    /// included (the recount's counter term).
    pub listed_max_object_id: Option<u64>,
    /// The largest object id over the surviving object headers (the
    /// recount's other counter term).
    pub surviving_max_object_id: Option<u64>,
    /// What the store held for each session the plan names, as read after
    /// the quiet period.
    pub sessions_seen: BTreeMap<String, SessionSeen>,
}

/// What a finished run's outcome is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanOutcome {
    /// Nothing keeps the service held: the run may un-hold once it has
    /// written the plan.
    Writable,
    /// Something keeps the service held: the run writes what it can and
    /// exits 3.
    StaysHeld,
}

impl PlanOutcome {
    /// The summary line's spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Writable => "writable",
            Self::StaysHeld => "stays held",
        }
    }
}

impl RestorePlan {
    /// The run's outcome after the coverage check was finalised: held when
    /// [`Self::held`] is non-empty or `report` names an uncovered or unknown
    /// value.
    pub fn outcome(&self, report: &CoverageReport) -> PlanOutcome {
        if self.held.is_empty() && report.uncovered_count() == 0 && report.unknown_count() == 0 {
            PlanOutcome::Writable
        } else {
            PlanOutcome::StaysHeld
        }
    }

    /// The summary line's counts.
    pub fn counts(&self) -> PlanCounts {
        let mut counts = PlanCounts::default();
        for object in &self.objects {
            match &object.action {
                ObjectAction::Keep => counts.keep += 1,
                ObjectAction::Restore { .. } => counts.restore += 1,
                ObjectAction::Remap { .. } => counts.remap += 1,
                ObjectAction::ClearFlags { .. } => counts.clear += 1,
                ObjectAction::Nothing => {}
            }
        }
        for session in &self.sessions {
            match &session.action {
                SessionAction::Restore { .. } => counts.sessions += 1,
                SessionAction::CompleteAbandonedDelete { .. } => counts.deletes += 1,
                SessionAction::Nothing => {}
            }
        }
        counts.held = self.held.len();
        for advisory in &self.advisories {
            match advisory {
                Advisory::RestoredFromOlderBackup { abandoned, .. } => {
                    if abandoned.first().is_some_and(|newest| newest.absent) {
                        counts.restored_from_older_absent += 1;
                    } else {
                        counts.restored_from_older += 1;
                    }
                }
                Advisory::SessionWithoutBackup { .. } => counts.sessions_without_backup += 1,
                _ => {}
            }
        }
        counts
    }
}

/// The summary line's counts ([`RestorePlan::counts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCounts {
    /// Layers to restore from a backup.
    pub restore: usize,
    /// Layers whose surviving copy is kept.
    pub keep: usize,
    /// Layers moved to their complete successor.
    pub remap: usize,
    /// Layers whose stale retirement flags are cleared.
    pub clear: usize,
    /// Items keeping the service held.
    pub held: usize,
    /// Sessions to restore from a session backup.
    pub sessions: usize,
    /// Unfinished session deletes to finish.
    pub deletes: usize,
    /// Layers restored from an older backup because the newest could not be
    /// read: its newest abandoned generation was stepped down on a verdict on
    /// its bytes (PD-6).
    pub restored_from_older: usize,
    /// Layers restored from an older backup because the newest needs a backup
    /// file absent from the backup store at `T_safe`: its newest abandoned
    /// generation was stepped down for a confirmed absence (PD-6).
    pub restored_from_older_absent: usize,
    /// Sessions named by evidence with no record, no deletion marker and no
    /// session backup, reported and not held (PD-1).
    pub sessions_without_backup: usize,
}

/// One session's plan (SR-9.4).
#[derive(Clone, Debug)]
pub struct SessionPlan {
    /// The session.
    pub session_id: String,
    /// What the writer does for it.
    pub action: SessionAction,
}

/// What the writer does for one session.
#[derive(Clone, Debug)]
pub enum SessionAction {
    /// Nothing: its record survives and needs nothing, or it is held.
    Nothing,
    /// Restore it from its session backup through the session applier:
    /// Step A (the path index) always, Step B (the record, the mappings
    /// `checkpoint.layer_mappings` names and `checkpoint.layers_index`) when
    /// `step_b`. `checkpoint` is filtered: its mappings are only those to
    /// write, and its layers index names every mapping that exists after Step
    /// B.
    Restore {
        /// The filtered session backup.
        checkpoint: SessionCheckpoint,
        /// Whether Step B runs.
        step_b: bool,
    },
    /// A surviving deletion marker: the session was deleted, so it is not
    /// restored, and the writer finishes the abandoned delete of what
    /// survives, in this order: register `objects` on the dead-object set,
    /// wipe each, unmap each of `mappings` (and remove the layers-index
    /// member of every other name in `layer_names`), remove the path-index
    /// members, delete the session backup. Each step is idempotent.
    CompleteAbandonedDelete {
        /// Surviving objects mapped under the session or whose header names
        /// it.
        objects: Vec<u64>,
        /// Every layer name the session still carries: its surviving
        /// mappings', its surviving headers' and its layers-index members.
        layer_names: Vec<String>,
        /// The session's base path, from its listed session backup, if any.
        base_path: Option<String>,
        /// The surviving `(layer, object)` mappings under the session: a run
        /// killed after the wipe lists the objects again through them.
        mappings: Vec<(String, u64)>,
        /// The session's composite path-names members, from its listed
        /// session backup; empty without one.
        path_name_members: Vec<String>,
    },
}

/// One layer object's plan (SR-9.5).
#[derive(Clone, Debug)]
pub struct ObjectPlan {
    /// The object.
    pub object_id: u64,
    /// The session whose mapping names it after the plan.
    pub session_id: String,
    /// The layer name.
    pub layer_name: String,
    /// What the writer does for it.
    pub action: ObjectAction,
}

/// What the writer does for one layer object.
#[derive(Clone, Debug)]
pub enum ObjectAction {
    /// Keep the surviving object as it is.
    Keep,
    /// Restore the object from `lineage`.
    Restore {
        /// Why.
        reason: RestoreReason,
        /// The lineage to restore, by its exact handles.
        lineage: Box<PlannedLineage>,
        /// The sequence band's inputs.
        band: BandInputs,
    },
    /// Map the layer to its complete successor `to`; `from` is retired.
    Remap {
        /// The fenced object the mapping names now.
        from: u64,
        /// Its complete successor, for the same session and layer.
        to: u64,
    },
    /// Clear the stale retirement flags of the surviving object: band its
    /// stream, remove its retirement carrier and attempts, and last clear the
    /// flags with the banded `seq` (Task 8's order).
    ClearFlags {
        /// The sequence band's inputs.
        band: BandInputs,
    },
    /// Nothing is written: the object is held with no survivor.
    Nothing,
}

/// Why an object is restored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreReason {
    /// No header survives.
    HeaderAbsent,
    /// A header survives without every structural field.
    HeaderPartial,
    /// The backup is newer than the complete survivor at `survivor_seq`.
    NewerThanSurvivor {
        /// The survivor's `seq`.
        survivor_seq: u64,
    },
    /// The backup is newer than the complete, retiring survivor at
    /// `survivor_seq`.
    NewerThanRetiringSurvivor {
        /// The survivor's `seq`.
        survivor_seq: u64,
    },
}

/// One restorable lineage, by its exact handles: the writer re-fetches it.
#[derive(Clone, Debug)]
pub struct PlannedLineage {
    /// The base.
    pub base: CheckpointHandle,
    /// The tails, in replay order (oldest first). Empty for a base-only
    /// lineage.
    pub tails: Vec<CheckpointHandle>,
    /// The sequence the lineage restores to.
    pub head_seq: u64,
    /// The restore meta: the base's, or for a composed lineage the base's
    /// with the head tail's span and write time
    /// (`recovery::adjudicate_composed_meta`).
    pub meta: CheckpointMeta,
    /// The external values the lineage's tails carry.
    pub tail_carried: Box<[ExtHash]>,
    /// Every newer candidate the selection stepped down from, oldest
    /// abandonment first.
    pub step_downs: Vec<StepDownRecord>,
    /// What the lineage costs to restore, recorded when the planner decoded
    /// it: what the writer charges its memory budget before it re-fetches.
    pub footprint: LineageFootprint,
}

/// A planned lineage's restore cost, measured by the planner's decode (SDD
/// §22.22, *Restore memory*). Not persisted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LineageFootprint {
    /// The bytes fetched: the base's plus every tail's.
    pub fetched_bytes: u64,
    /// The decoded state's weight
    /// (`storage::structural_cache::estimate_state_bytes`, the full-form
    /// weigher).
    pub state_bytes: u64,
    /// The commands the writer queues for the state over an empty pre-image,
    /// header and stream `UNLINK` included.
    pub commands: u64,
    /// The largest inline value of the state, in bytes, over its fields and
    /// time samples (0 when it has none): the unit the writer's charge
    /// prices apart, since one value over the restore batch's byte limit is
    /// queued alone, whole (`execute::memory::inline_value_charge`).
    pub largest_inline_bytes: u64,
}

impl PlannedLineage {
    /// Whether the lineage is the base alone.
    pub fn is_base_only(&self) -> bool {
        self.tails.is_empty()
    }
}

/// The inputs of a restored or flag-cleared object's sequence band. The
/// writer adds the stream's high-water and computes the band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandInputs {
    /// The band's floor: the restored head, or the survivor's `seq`.
    pub floor_seq: u64,
    /// The measured commit span (0: the configured margin applies).
    pub observed_commit_span: u64,
    /// The newest generation's write time (0: none recorded).
    pub written_at_ms: u64,
    /// The survivor's `(last_checkpoint_seq, last_checkpoint_ms)`, when both
    /// are present.
    pub cursor: Option<(u64, u64)>,
}

/// A write-shaping verdict that keeps the service held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldItem {
    /// The session, empty when unknown.
    pub session_id: String,
    /// The layer, empty for a session-level item, and for a listed object
    /// whose session and layer could not be learned.
    pub layer_name: String,
    /// The object, for a layer-level item or a listed object whose session
    /// and layer could not be learned (session and layer both empty).
    pub object_id: Option<u64>,
    /// Why.
    pub reason: HeldReason,
}

/// Why an item keeps the service held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeldReason {
    /// The object needs a backup and none is listed.
    NoBackup,
    /// The object needs a backup and there is no backup store (`Disabled` or
    /// `InMemory`).
    NoBackupStore,
    /// The newest full backup is a plain OMPF over the per-delta entry limit,
    /// which an earlier release wrote and this one cannot read (defer-3); no
    /// older backup is used.
    OverLimitPlainBase {
        /// The selected lineage's head.
        head_seq: u64,
        /// Its base.
        base_seq: u64,
    },
    /// A newer backup than the kept survivor exists but is a plain OMPF over
    /// the per-delta entry limit (defer-3).
    NewerBackupUnusable {
        /// The unusable lineage's head.
        head_seq: u64,
    },
    /// Every listed backup was abandoned and no survivor exists.
    BackupsUnusable {
        /// The selection's exhaustion.
        exhaustion: String,
    },
    /// Accept-loss only: every listed backup was abandoned and no survivor
    /// exists, and among them are plain full backups over the per-delta
    /// entry limit that an earlier release wrote and this one cannot read
    /// (PD-4's step-down found nothing older to use). Accept-loss unmaps the
    /// layer but keeps those backups (`super::lost`).
    BackupsOverLimit {
        /// The selection's exhaustion.
        exhaustion: String,
        /// The over-limit generations, `(head_seq, base_seq)`, newest first.
        bases: Vec<(u64, u64)>,
    },
    /// A surviving mapping names an object on the dead-object set.
    MappedToDeletedObject {
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
    },
    /// A listed session backup could not be read.
    SessionBackupUnreadable {
        /// The session.
        session_id: String,
        /// Why.
        error: String,
        /// How the read failed: what accept-loss judges the session by
        /// (`super::lost`). Restore holds on every cause alike.
        cause: SessionReadFailure,
    },
    /// The backup store's answer is unknown: a fetch error after the
    /// adapter's retries, or the selection's budget ran out (PD-4).
    Unknown {
        /// What failed.
        error: String,
    },
}

impl HeldReason {
    /// A short description for the log.
    pub fn describe(&self) -> String {
        match self {
            Self::NoBackup => "it needs a backup and the backup store lists none".to_owned(),
            Self::NoBackupStore => {
                "it needs a backup and this deployment has no backup store".to_owned()
            }
            Self::OverLimitPlainBase { head_seq, base_seq } => format!(
                "its newest backup (generation {head_seq}, base {base_seq}) is over the per-delta \
                 entry limit and was written by an earlier release"
            ),
            Self::NewerBackupUnusable { head_seq } => format!(
                "a newer backup (generation {head_seq}) than its surviving copy is over the \
                 per-delta entry limit and was written by an earlier release"
            ),
            Self::BackupsUnusable { exhaustion } => {
                format!("no listed backup of it could be used ({exhaustion})")
            }
            Self::BackupsOverLimit { exhaustion, bases } => format!(
                "no listed backup of it could be used ({exhaustion}); the full backups under {} \
                 are over the per-delta entry limit and were written by an earlier release",
                render_over_limit_bases(bases)
            ),
            Self::MappedToDeletedObject {
                session_id,
                layer_name,
            } => format!(
                "the surviving mapping of layer {layer_name} in session {session_id} names a \
                 deleted object"
            ),
            Self::SessionBackupUnreadable {
                session_id, error, ..
            } => {
                format!("the session backup of session {session_id} could not be read: {error}")
            }
            Self::Unknown { error } => {
                format!("the backup store's answer is unknown: {error}")
            }
        }
    }
}

/// Over-limit generations `(head_seq, base_seq)`, named for a line:
/// "generation 40 (full backup 40)", or "generations 40 (full backup 40)
/// and 30 (full backup 20)".
pub fn render_over_limit_bases(bases: &[(u64, u64)]) -> String {
    let named: Vec<String> = bases
        .iter()
        .map(|(head, base)| format!("{head} (full backup {base})"))
        .collect();
    match named.as_slice() {
        [one] => format!("generation {one}"),
        [init @ .., last] => format!("generations {} and {last}", init.join(", ")),
        [] => "no generation".to_owned(),
    }
}

/// How a listed session backup's read failed
/// ([`HeldReason::SessionBackupUnreadable`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionReadFailure {
    /// The bytes were read and do not decode: a verdict on bytes the backup
    /// store returned.
    Undecodable,
    /// The fetch failed after the adapter's retries (`Transient` or
    /// `Permanent`): an error, not an absence, so the backup's fate is
    /// unconfirmed.
    ReadError,
    /// The listed backup read as absent at or after `T_safe`: a confirmed
    /// absence.
    AbsentAtTSafe,
}

/// What the planner read of one session in the store (`StoreFacts`), kept in
/// the plan so a pure function can name a lost session's objects
/// (`super::lost`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionSeen {
    /// Its record (`le:{sid}:sess`) exists.
    pub record: bool,
    /// Its deletion marker (`le:{sid}:sess:deleted`) exists.
    pub deletion_marker: bool,
    /// Its surviving `(layer, object)` mappings, sorted.
    pub surviving_mappings: Vec<(String, u64)>,
    /// The surviving headers that name it, `(object, layer)`, sorted.
    pub header_objects: Vec<(u64, String)>,
    /// Its layers index's names, sorted.
    pub layers_index: Vec<String>,
}

/// One newer generation a lineage stepped down from (PD-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbandonedGeneration {
    /// The generation abandoned.
    pub seq: u64,
    /// Its step-down's outcome label (`LineageBreak::reason_label`).
    pub reason_label: &'static str,
    /// Whether it was stepped down for an absence (`LinkAbsent` or
    /// `MissingMetaCompanion`), which the Job's judge lets stand only at or
    /// after `T_safe`, rather than for a verdict on bytes the backup store
    /// returned.
    pub absent: bool,
}

/// Something the plan reports without holding the service.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Advisory {
    /// PD-1: a session named by evidence, with no record, no deletion marker
    /// and no listed session backup, under a non-empty session listing. Not
    /// restored; its layers' backups and objects are left unreferenced.
    SessionWithoutBackup {
        /// The session.
        session_id: String,
        /// What names it.
        evidence: String,
        /// The layers that name it and whose backups are listed.
        layers_with_backups: Vec<String>,
    },
    /// A surviving mapping names an object on the dead-object set, and the
    /// session backup's newer mapping of the same layer, to a live object,
    /// replaces it (Step B writes it): not held (defer-51).
    DeletedMappingReplaced {
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The deleted object the surviving mapping names.
        deleted_object_id: u64,
        /// The newer object the session backup maps the layer to.
        replacement_object_id: u64,
    },
    /// A session backup's mapping names an object on the dead-object set:
    /// dropped, not restored.
    CheckpointMappingToDeletedObject {
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The deleted object.
        object_id: u64,
    },
    /// PD-6: the layer is restored from an older backup because newer ones
    /// could not be used.
    RestoredFromOlderBackup {
        /// The object.
        object_id: u64,
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The generation restored.
        restored_seq: u64,
        /// The newer generations abandoned, newest first, with why.
        abandoned: Vec<AbandonedGeneration>,
    },
    /// PD-6: the surviving copy is kept because the newer backups could not
    /// be used.
    NewerBackupsUnusable {
        /// The object.
        object_id: u64,
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The survivor's `seq`.
        kept_seq: u64,
        /// The newer generations abandoned, newest first, with why.
        abandoned: Vec<AbandonedGeneration>,
    },
    /// defer-53: the surviving copy is kept because no newer listed backup of
    /// it could be used and none was stepped down from (every newer listed
    /// generation is a tail with no base). The shared selection walk's WARN
    /// calls such a layer unrecoverable; in the Job it is kept.
    KeptWithBackupsExhausted {
        /// The object.
        object_id: u64,
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The survivor's `seq`.
        kept_seq: u64,
        /// The selection's exhaustion, described.
        exhaustion: String,
    },
    /// PD-2: hashes a restored base's ext-refs sidecar names that no row of
    /// the restored state references. Never held on.
    SidecarHashesNotInState {
        /// The object.
        object_id: u64,
        /// How many.
        count: usize,
        /// Whether the lineage is base-only (where sidecar and state should
        /// agree).
        base_only: bool,
    },
    /// A listed backup no mapping names after the plan.
    UnreferencedBackup {
        /// The object.
        object_id: u64,
    },
    /// A surviving object of a session reported under PD-1.
    UnreferencedObject {
        /// The object.
        object_id: u64,
        /// The session its header names.
        session_id: String,
        /// The layer its header names.
        layer_name: String,
    },
}

/// The coverage verdict of one referenced external value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashVerdict {
    /// Valkey holds its blob.
    Valkey,
    /// A restored lineage's tails carry it.
    Tails,
    /// This run stored it in the backup store itself.
    OwnPut,
    /// The backup store holds it.
    Backup,
    /// The backup store answered "absent" before `T_safe`; checked once more
    /// at or after `T_safe` when the check is finalised.
    PendingReprobe,
    /// In no store.
    Uncovered,
    /// The backup store's answer is unknown.
    Unknown(String),
}

impl HashVerdict {
    /// Whether the value is covered.
    pub fn is_covered(&self) -> bool {
        matches!(
            self,
            Self::Valkey | Self::Tails | Self::OwnPut | Self::Backup
        )
    }
}

/// Where the coverage check names a referenced value's rows from.
#[derive(Clone, Debug)]
pub enum CoverageSource {
    /// A restored object: its planned lineage, re-fetched and decoded.
    Restored(Box<PlannedLineage>),
    /// A surviving object: read from Valkey.
    Survivor,
}

/// One object the coverage check covers.
#[derive(Clone, Debug)]
pub struct CoverageObject {
    /// The session.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// Where its rows are read from.
    pub source: CoverageSource,
}

/// The coverage check as planned (SR-9.3): a verdict per distinct referenced
/// value, finalised after the writes.
#[derive(Clone, Debug, Default)]
pub struct CoveragePlan {
    /// Each distinct referenced value's verdict.
    pub verdicts: BTreeMap<ExtHash, HashVerdict>,
    /// The objects that reference each value.
    pub references: BTreeMap<ExtHash, BTreeSet<u64>>,
    /// The objects the check covers.
    pub objects: BTreeMap<u64, CoverageObject>,
}

impl CoveragePlan {
    /// How many distinct values are judged uncovered.
    pub fn uncovered_count(&self) -> usize {
        self.verdicts
            .values()
            .filter(|v| matches!(v, HashVerdict::Uncovered))
            .count()
    }
}

/// One referenced value's place in a layer: what the coverage report names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueSite {
    /// The prim path, `/`-joined from below the layer's root.
    pub prim_path: String,
    /// The attribute (or field) name.
    pub attribute: String,
    /// The value.
    pub hash: ExtHash,
    /// How many rows at this site hold it.
    pub count: u64,
}

/// The coverage report of one layer.
#[derive(Clone, Debug, Default)]
pub struct LayerCoverageReport {
    /// The object.
    pub object_id: u64,
    /// The session.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// The distinct values in no store.
    pub uncovered: BTreeSet<ExtHash>,
    /// The distinct values whose presence is unknown.
    pub unknown: BTreeSet<ExtHash>,
    /// Where the uncovered values sit.
    pub uncovered_sites: Vec<ValueSite>,
    /// Where the unknown values sit.
    pub unknown_sites: Vec<ValueSite>,
    /// Why the sites could not be named, when they could not.
    pub naming_error: Option<String>,
}

/// The finalised coverage check.
#[derive(Clone, Debug, Default)]
pub struct CoverageReport {
    /// Each distinct referenced value's final verdict.
    pub verdicts: BTreeMap<ExtHash, HashVerdict>,
    /// One entry per layer referencing an uncovered or unknown value.
    pub layers: Vec<LayerCoverageReport>,
}

impl CoverageReport {
    /// Distinct values in no store.
    pub fn uncovered_count(&self) -> usize {
        self.verdicts
            .values()
            .filter(|v| matches!(v, HashVerdict::Uncovered))
            .count()
    }

    /// Distinct values whose presence is unknown.
    pub fn unknown_count(&self) -> usize {
        self.verdicts
            .values()
            .filter(|v| matches!(v, HashVerdict::Unknown(_)))
            .count()
    }
}

/// A restorable lineage the Job's selection found, with what the coverage
/// check needs of its decoded state.
#[derive(Clone, Debug)]
pub struct SelectedLineage {
    /// The lineage.
    pub lineage: PlannedLineage,
    /// The external values the decoded state's rows reference.
    pub state_hashes: BTreeSet<ExtHash>,
    /// Hashes the base's ext-refs sidecar names that no row references
    /// (PD-2), counted.
    pub sidecar_extra: usize,
}

/// What the Job's lineage walk ended with, before the absence policy
/// ([`super::decide::judge_lineage`]) is applied.
#[derive(Clone, Debug)]
pub enum LineageOutcome {
    /// A lineage newer than the floor decoded.
    Selected(Box<SelectedLineage>),
    /// The newest restorable lineage is not newer than the survivor.
    NotNewer,
    /// Nothing restorable is left.
    Exhausted(crate::checkpoint::lineage::Exhaustion),
    /// defer-3: the selected lineage's base is a plain OMPF over the per-delta
    /// entry limit.
    OverLimitPlainBase {
        /// The lineage's head.
        head_seq: u64,
        /// Its base.
        base_seq: u64,
    },
    /// A backup-store call failed after the adapter's own retries.
    Errored {
        /// The error.
        error: String,
    },
}

/// What the Job's lineage walk does with a selected lineage whose base is a
/// plain OMPF over the per-delta entry limit (defer-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverLimitPolicy {
    /// Restore: end the walk with no step-down
    /// ([`LineageOutcome::OverLimitPlainBase`]), so the layer holds and no
    /// older backup is used.
    Hold,
    /// Accept-loss (PD-4): step down past it to the newest usable older
    /// generation, recorded as a step-down and reported as a restore from an
    /// older backup, so a layer that can be restored is never discarded.
    StepDown,
}

impl OverLimitPolicy {
    /// The policy of the Job running in `mode`.
    pub fn for_mode(mode: JobMode) -> Self {
        match mode {
            JobMode::Restore => Self::Hold,
            JobMode::AcceptLoss => Self::StepDown,
        }
    }
}

/// The Job's verdict on one object's lineage, after the absence policy.
#[derive(Clone, Debug)]
pub enum LineageVerdict {
    /// Restore this lineage.
    Use(Box<SelectedLineage>),
    /// The survivor stays: no backup is newer, possibly after step-downs.
    Survivor {
        /// The newer candidates abandoned.
        step_downs: Vec<StepDownRecord>,
    },
    /// Every listed backup was abandoned.
    BackupsUnusable {
        /// The exhaustion.
        exhaustion: String,
        /// The candidates abandoned.
        step_downs: Vec<StepDownRecord>,
        /// Of those, the plain bases over the per-delta entry limit the walk
        /// stepped past (accept-loss, PD-4), `(head_seq, base_seq)`; empty
        /// for restore.
        over_limit: Vec<(u64, u64)>,
    },
    /// defer-3.
    OverLimitPlainBase {
        /// The lineage's head.
        head_seq: u64,
        /// Its base.
        base_seq: u64,
    },
    /// PD-4: unknown, held.
    Unknown {
        /// Why.
        error: String,
    },
    /// An absence seen before `T_safe`: re-selected at `T_safe`.
    Defer {
        /// The generation read as absent.
        seq: u64,
    },
}
