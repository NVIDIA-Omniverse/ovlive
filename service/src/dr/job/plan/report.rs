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

//! The restore plan's report (SDD SR-9.3): where each uncovered or unknown
//! external value sits in a layer — prim path, attribute, row count and hash
//! prefix — and, for tests, the rows that reference it (`rows_referencing`,
//! the same walk: the rows accept-loss's value removal must delete); the
//! plan's log lines, with the
//! writer's summary ([`log_execution`]) and the run's restore summary
//! (SR-14.6, [`log_restore_summary`]: per layer, then the totals); and
//! accept-loss's lines (SR-10, the `render_accept_loss_*` and
//! `render_discard*` texts). Rendering is pure, over either form of an
//! object's state ([`ObjectStateOf`]) and over the plan, its finalised
//! coverage report, the writer's report, the lost set and the discard
//! report; the `log_*` functions only emit.

use std::collections::{BTreeMap, BTreeSet};

use super::lost::{DiscardDefect, LostLayer, LostSet, Unconfirmed, UnconfirmedPolicy};
use super::model::{
    AbandonedGeneration, Advisory, CoverageReport, ExtHash, HashVerdict, LayerCoverageReport,
    ObjectAction, RestorePlan, ValueSite,
};
use crate::domain::delta::{SPEC_ATTRIBUTE, SPEC_PRIM};
use crate::domain::limits::{MAX_DELTA_ENTRIES, MAX_SECTION_TREE_DEPTH};
use crate::domain::types::{FieldRows, ObjectStateOf, SampleRows, ValueRows};
use crate::dr::job::args::{JobMode, CONFIRM_DISCARD_UNCONFIRMED_DATA};
use crate::dr::job::execute::discard::DiscardReport;
use crate::dr::job::execute::{ExecutionReport, ObjectWrite, WriteKind};
use crate::storage::ext_reconcile::RecountReport;
use crate::storage::keys::sha256_to_hex;
#[cfg(any(test, feature = "test-support"))]
use crate::storage::layer::ValueRowKey;

/// How many value sites one layer's coverage line names, at most; the rest
/// are counted as "and N more".
pub const DR_COVERAGE_REPORT_ENTRIES_PER_LAYER: usize = 100;

/// How many object ids an aggregated advisory line names, at most.
const ADVISORY_IDS_LOGGED: usize = 20;

/// The parent id of a layer's top-level sections (a section row never has
/// this id).
const ROOT_PARENT_ID: u64 = 0;

/// The attribute text of a time-sample row.
const TIME_SAMPLES: &str = "(time samples)";

/// Every external value `state`'s field and time-sample rows reference: what
/// a restored object contributes to the coverage check (PD-2).
pub fn referenced_hashes<R: ValueRows>(state: &ObjectStateOf<R>) -> BTreeSet<ExtHash> {
    let mut hashes = BTreeSet::new();
    state.fields.for_each(|_, _, hash| {
        if let Some(hash) = hash {
            hashes.insert(*hash);
        }
    });
    state.time_samples.for_each(|_, _, hash| {
        if let Some(hash) = hash {
            hashes.insert(*hash);
        }
    });
    hashes
}

/// A value's hash prefix as the logs name it: its first 16 hex characters,
/// as the ext-mirror's WARNs log it.
pub fn hash_prefix(hash: &ExtHash) -> String {
    let mut hex = sha256_to_hex(hash);
    hex.truncate(16);
    hex
}

/// The prim path of `section_id`: the section names from below the layer's
/// root down to it, joined with `/` and led by `/`. The root (the top-level
/// section, whose parent is the layer itself) is left out unless it is a
/// prim. A walk past the section-tree depth limit, a cycle or a missing
/// parent ends the path with `…` at its top.
pub fn section_path<R: ValueRows>(state: &ObjectStateOf<R>, section_id: u64) -> String {
    let mut names: Vec<&str> = Vec::new();
    let mut id = section_id;
    let mut complete = false;
    for _ in 0..=MAX_SECTION_TREE_DEPTH {
        if id == ROOT_PARENT_ID {
            complete = true;
            break;
        }
        let Some(section) = state.sections.get(&id) else {
            break;
        };
        if section.parent_id == ROOT_PARENT_ID {
            if section.section_type == SPEC_PRIM {
                names.push(&section.name);
            }
            complete = true;
            break;
        }
        names.push(&section.name);
        id = section.parent_id;
    }
    if !complete {
        names.push("…");
    }
    names.reverse();
    format!("/{}", names.join("/"))
}

/// One value row the reference walk ([`walk_referencing_rows`]) visits.
#[derive(Clone, Copy, Debug)]
enum RowAt<'a> {
    /// A field row: its section and key.
    Field(u64, &'a str),
    /// A time-sample row: its section and its time's `f64` bits.
    TimeSample(u64, u64),
}

/// Call `f` once per row of `state` whose external hash is in `hashes`:
/// every field row, then every time-sample row, in no particular order
/// within each. The one walk [`value_sites`] names sites by and the test
/// oracle `rows_referencing` collects rows by.
fn walk_referencing_rows<R: ValueRows>(
    state: &ObjectStateOf<R>,
    hashes: &BTreeSet<ExtHash>,
    mut f: impl FnMut(RowAt<'_>, &ExtHash),
) {
    state.fields.for_each(|section_id, key, hash| {
        if let Some(hash) = hash.filter(|h| hashes.contains(*h)) {
            f(RowAt::Field(section_id, key), hash);
        }
    });
    state.time_samples.for_each(|section_id, time_bits, hash| {
        if let Some(hash) = hash.filter(|h| hashes.contains(*h)) {
            f(RowAt::TimeSample(section_id, time_bits), hash);
        }
    });
}

/// Every row of `state` whose external hash is in `hashes`, grouped by hash
/// in hash order, each group's rows sorted: field rows by `(section, key)`,
/// time-sample rows by `(section, time bits)`, bit-exact, so a legacy `-0.0`
/// sample is its own row; a hash no row references has no group. A test
/// oracle: the rows the coverage report's sites name ([`value_sites`]),
/// which accept-loss's value removal must delete, and it alone
/// (`dr::job::execute::discard`, D5, which reads the rows from the store by
/// the same readers' rule rather than from a decoded state).
#[cfg(any(test, feature = "test-support"))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rows_referencing<R: ValueRows>(
    state: &ObjectStateOf<R>,
    hashes: &BTreeSet<ExtHash>,
) -> BTreeMap<ExtHash, Vec<ValueRowKey>> {
    let mut rows: BTreeMap<ExtHash, Vec<ValueRowKey>> = BTreeMap::new();
    walk_referencing_rows(state, hashes, |row, hash| {
        let key = match row {
            RowAt::Field(section_id, key_name) => ValueRowKey::Field {
                section_id,
                key_name: key_name.to_owned(),
            },
            RowAt::TimeSample(section_id, time_bits) => ValueRowKey::TimeSample {
                section_id,
                time_bits,
            },
        };
        rows.entry(*hash).or_default().push(key);
    });
    for group in rows.values_mut() {
        group.sort();
    }
    rows
}

/// Every site in `state` whose row references one of `hashes`: a row on an
/// attribute section is named by its prim (the attribute's parent) and the
/// attribute; a field on any other section by the section's own path and the
/// field key; a time-sample row as `(time samples)` (of its attribute, on an
/// attribute section). Sites group by `(prim path, attribute, hash)`, with
/// the row count, sorted.
pub fn value_sites<R: ValueRows>(
    state: &ObjectStateOf<R>,
    hashes: &BTreeSet<ExtHash>,
) -> Vec<ValueSite> {
    let mut counts: BTreeMap<(String, String, ExtHash), u64> = BTreeMap::new();
    let site_of = |section_id: u64, attribute: Option<&str>| -> (String, String) {
        match state.sections.get(&section_id) {
            Some(section) if section.section_type == SPEC_ATTRIBUTE => {
                let name = match attribute {
                    Some(_) => section.name.clone(),
                    None => format!("{} {TIME_SAMPLES}", section.name),
                };
                (section_path(state, section.parent_id), name)
            }
            _ => (
                section_path(state, section_id),
                attribute.unwrap_or(TIME_SAMPLES).to_owned(),
            ),
        }
    };
    let mut rows: Vec<(String, String, ExtHash)> = Vec::new();
    walk_referencing_rows(state, hashes, |row, hash| {
        let (prim, attribute) = match row {
            RowAt::Field(section_id, key) => site_of(section_id, Some(key)),
            RowAt::TimeSample(section_id, _time_bits) => site_of(section_id, None),
        };
        rows.push((prim, attribute, *hash));
    });
    for row in rows {
        *counts.entry(row).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|((prim_path, attribute, hash), count)| ValueSite {
            prim_path,
            attribute,
            hash,
            count,
        })
        .collect()
}

/// `sites` as one log field: `{prim path} {attribute} ×{count} {hash
/// prefix}` per site, `; `-separated, at most `cap` of them, then `and {k}
/// more`.
pub fn render_sites(sites: &[ValueSite], cap: usize) -> String {
    let mut parts: Vec<String> = sites
        .iter()
        .take(cap)
        .map(|site| {
            format!(
                "{} {} ×{} {}",
                site.prim_path,
                site.attribute,
                site.count,
                hash_prefix(&site.hash)
            )
        })
        .collect();
    if sites.len() > cap {
        parts.push(format!("and {} more", sites.len() - cap));
    }
    parts.join("; ")
}

/// Log one layer's coverage report: a WARN for its uncovered values and one
/// for its unknown ones, each naming up to
/// [`DR_COVERAGE_REPORT_ENTRIES_PER_LAYER`] sites.
pub fn log_layer_report(layer: &LayerCoverageReport) {
    if !layer.uncovered.is_empty() {
        tracing::warn!(
            object_id = layer.object_id,
            session_id = %layer.session_id,
            layer_name = %layer.layer_name,
            values = %render_sites(&layer.uncovered_sites, DR_COVERAGE_REPORT_ENTRIES_PER_LAYER),
            naming_error = layer.naming_error.as_deref().unwrap_or(""),
            "layer {} of session {} references {} values that are in no store",
            layer.layer_name,
            layer.session_id,
            layer.uncovered.len()
        );
    }
    if !layer.unknown.is_empty() {
        tracing::warn!(
            object_id = layer.object_id,
            session_id = %layer.session_id,
            layer_name = %layer.layer_name,
            values = %render_sites(&layer.unknown_sites, DR_COVERAGE_REPORT_ENTRIES_PER_LAYER),
            naming_error = layer.naming_error.as_deref().unwrap_or(""),
            "layer {} of session {} references {} values whose presence the backup store could \
             not confirm",
            layer.layer_name,
            layer.session_id,
            layer.unknown.len()
        );
    }
}

/// Log the finalised coverage report, per affected layer: one WARN for its
/// values in no store and another for its values whose presence the backup
/// store could not confirm, so a layer with both kinds gets two lines.
pub fn log_coverage_report(report: &CoverageReport) {
    for layer in &report.layers {
        log_layer_report(layer);
    }
}

/// Log defer-3's ERROR for one layer: its newest full backup is a plain OMPF
/// over the per-delta entry limit, which no step-down replaces.
pub fn log_over_limit(object_id: u64, session_id: &str, layer_name: &str, base_seq: u64) {
    tracing::error!(
        object_id,
        session_id,
        layer_name,
        generation = base_seq,
        max_delta_entries = MAX_DELTA_ENTRIES,
        "layer {layer_name} of session {session_id} (object {object_id}) cannot be restored: its \
         newest full backup (generation {base_seq}) is over {MAX_DELTA_ENTRIES} entries and was \
         written by an earlier release, which this release cannot read; the Job does not fall \
         back to an older backup of it, and the service stays held"
    );
}

/// A step-down line's facts: the newest abandoned generation, what happened
/// to it in words, and the whole abandoned list.
struct StepDownText {
    newest: u64,
    /// Completes "its newest backup, generation N, …": unreadable for a
    /// verdict on bytes, absent for an absence that stands at `T_safe`.
    cause: String,
    list: String,
}

fn step_down_text(abandoned: &[AbandonedGeneration]) -> StepDownText {
    let (newest, cause) = match abandoned.first() {
        Some(g) if g.absent => (
            g.seq,
            format!(
                "needs a backup file that is absent from the backup store at T_safe ({})",
                g.reason_label
            ),
        ),
        Some(g) => (g.seq, format!("was unreadable ({})", g.reason_label)),
        None => (0, "could not be used".to_owned()),
    };
    let list = abandoned
        .iter()
        .map(|g| format!("{} ({})", g.seq, g.reason_label))
        .collect::<Vec<_>>()
        .join(", ");
    StepDownText {
        newest,
        cause,
        list,
    }
}

/// Log the plan's advisories: the loud step-down and PD-1 WARNs one per
/// item, the rest aggregated.
pub fn log_plan_advisories(plan: &RestorePlan) {
    let mut unreferenced_backups: Vec<u64> = Vec::new();
    let mut unreferenced_objects: Vec<u64> = Vec::new();
    for advisory in &plan.advisories {
        match advisory {
            Advisory::RestoredFromOlderBackup {
                object_id,
                session_id,
                layer_name,
                restored_seq,
                abandoned,
            } => {
                let StepDownText {
                    newest,
                    cause,
                    list,
                } = step_down_text(abandoned);
                tracing::warn!(
                    object_id,
                    session_id = %session_id,
                    layer_name = %layer_name,
                    restored_generation = restored_seq,
                    newest_generation = newest,
                    abandoned = %list,
                    "layer {layer_name} of session {session_id} (object {object_id}) is restored \
                     from an older backup, generation {restored_seq}, because its newest backup, \
                     generation {newest}, {cause}"
                );
            }
            Advisory::NewerBackupsUnusable {
                object_id,
                session_id,
                layer_name,
                kept_seq,
                abandoned,
            } => {
                let StepDownText {
                    newest,
                    cause,
                    list,
                } = step_down_text(abandoned);
                tracing::warn!(
                    object_id,
                    session_id = %session_id,
                    layer_name = %layer_name,
                    kept_seq,
                    newest_generation = newest,
                    abandoned = %list,
                    "layer {layer_name} of session {session_id} (object {object_id}) keeps its \
                     surviving copy at seq {kept_seq}, because its newer backup, generation \
                     {newest}, {cause}"
                );
            }
            Advisory::SessionWithoutBackup {
                session_id,
                evidence,
                layers_with_backups,
            } => {
                if layers_with_backups.is_empty() {
                    tracing::warn!(
                        session_id = %session_id,
                        evidence = %evidence,
                        "session {session_id}, named by {evidence}, has no record in the store, \
                         no deletion marker and no session backup; it is not restored"
                    );
                } else {
                    let layers = layers_with_backups.join(", ");
                    tracing::warn!(
                        session_id = %session_id,
                        evidence = %evidence,
                        layers = %layers,
                        "session {session_id}, named by {evidence}, has no record in the store, \
                         no deletion marker and no session backup; it is not restored, and the \
                         backups of its layers ({layers}) are left unreferenced"
                    );
                }
            }
            Advisory::DeletedMappingReplaced {
                session_id,
                layer_name,
                deleted_object_id,
                replacement_object_id,
            } => tracing::warn!(
                session_id = %session_id,
                layer_name = %layer_name,
                deleted_object_id,
                replacement_object_id,
                "layer {layer_name} of session {session_id}: the store's mapping named object \
                 {deleted_object_id}, which was deleted; the session backup's newer mapping to \
                 object {replacement_object_id} replaces it"
            ),
            Advisory::CheckpointMappingToDeletedObject {
                session_id,
                layer_name,
                object_id,
            } => tracing::info!(
                session_id = %session_id,
                layer_name = %layer_name,
                object_id,
                "the session backup of session {session_id} maps layer {layer_name} to object \
                 {object_id}, which was deleted; that mapping is not restored"
            ),
            Advisory::KeptWithBackupsExhausted {
                object_id,
                session_id,
                layer_name,
                kept_seq,
                exhaustion,
            } => tracing::warn!(
                object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                kept_seq,
                exhaustion = %exhaustion,
                "layer {layer_name} of session {session_id} (object {object_id}) keeps its \
                 surviving copy at seq {kept_seq}, because its newer backups cannot be used \
                 ({exhaustion}); the restore selection's WARN about an unrecoverable layer does \
                 not apply here"
            ),
            Advisory::SidecarHashesNotInState {
                object_id,
                count,
                base_only,
            } => {
                if *base_only {
                    tracing::warn!(
                        object_id,
                        count,
                        "the backup restored for object {object_id} lists {count} external \
                         values in its reference list that none of its rows references; they are \
                         not checked"
                    );
                } else {
                    tracing::debug!(
                        object_id,
                        count,
                        "the composed backup restored for object {object_id} lists {count} \
                         external values in its base's reference list that none of its rows \
                         references; they are not checked"
                    );
                }
            }
            Advisory::UnreferencedBackup { object_id } => unreferenced_backups.push(*object_id),
            Advisory::UnreferencedObject { object_id, .. } => {
                unreferenced_objects.push(*object_id);
            }
        }
    }
    if !unreferenced_backups.is_empty() {
        tracing::info!(
            count = unreferenced_backups.len(),
            object_ids = ?&unreferenced_backups[..unreferenced_backups.len().min(ADVISORY_IDS_LOGGED)],
            "{} listed layer backups are named by no layer mapping after the plan and are not \
             restored",
            unreferenced_backups.len()
        );
    }
    if !unreferenced_objects.is_empty() {
        tracing::info!(
            count = unreferenced_objects.len(),
            object_ids = ?&unreferenced_objects[..unreferenced_objects.len().min(ADVISORY_IDS_LOGGED)],
            "{} surviving objects of sessions with no record and no session backup are left \
             unreferenced",
            unreferenced_objects.len()
        );
    }
}

/// Log each item that keeps the service held, one WARN each (defer-3's
/// layers also have their own ERROR).
pub fn log_held(plan: &RestorePlan) {
    for item in &plan.held {
        let reason = item.reason.describe();
        if let (true, true, Some(object_id)) = (
            item.session_id.is_empty(),
            item.layer_name.is_empty(),
            item.object_id,
        ) {
            // A listed backup whose session and layer are unknown.
            tracing::warn!(
                object_id,
                "object {object_id}, listed in the backup store, keeps the service held: {reason}"
            );
        } else if item.layer_name.is_empty() && item.object_id.is_none() {
            tracing::warn!(
                session_id = %item.session_id,
                "session {} keeps the service held: {reason}",
                item.session_id
            );
        } else {
            tracing::warn!(
                session_id = %item.session_id,
                layer_name = %item.layer_name,
                object_id = item.object_id,
                "layer {} of session {} (object {}) keeps the service held: {reason}",
                item.layer_name,
                item.session_id,
                item.object_id
                    .map_or_else(|| "unknown".to_owned(), |id| id.to_string())
            );
        }
    }
}

/// Log the `mode` Job's writer's summary line (`dr::job::execute`): what it
/// wrote.
pub fn log_execution(mode: JobMode, execution: &ExecutionReport) {
    let mut restored = 0usize;
    let mut batches = 0usize;
    let mut cleared = 0usize;
    for write in execution.objects.values() {
        match write.kind {
            WriteKind::Restored { .. } => {
                restored += 1;
                batches += write.batches.len();
            }
            WriteKind::FlagsCleared => cleared += 1,
        }
    }
    let moved = execution.remapped.len();
    let sessions = execution.sessions_restored + execution.sessions_path_index_only;
    let deletes = execution.deletes_completed.len();
    let valkey = execution.backfill.valkey_values;
    let puts = execution.backfill.backup_puts;
    tracing::info!(
        restored,
        batches,
        cleared,
        moved,
        sessions,
        deletes,
        backfill_valkey_values = valkey,
        backfill_backup_puts = puts,
        max_in_flight = execution.memory.max_in_flight,
        peak_reserved_bytes = execution.memory.peak_reserved_bytes,
        "the {mode} Job restored {restored} layers in {batches} batches, cleared retirement \
         flags on {cleared}, moved {moved} layers to their successor, restored {sessions} \
         sessions, finished {deletes} deletes, and stored {valkey} values carried only by \
         backups in Valkey and {puts} in the backup store"
    );
}

/// Log the plan's summary line, with the run's outcome after the coverage
/// check was finalised.
pub fn log_summary(plan: &RestorePlan, report: &CoverageReport) {
    let counts = plan.counts();
    let outcome = plan.outcome(report).as_str();
    tracing::info!(
        restore = counts.restore,
        keep = counts.keep,
        remap = counts.remap,
        clear = counts.clear,
        held = counts.held,
        sessions = counts.sessions,
        deletes = counts.deletes,
        restored_from_older = counts.restored_from_older,
        restored_from_older_absent = counts.restored_from_older_absent,
        sessions_without_backup = counts.sessions_without_backup,
        uncovered = report.uncovered_count(),
        unknown = report.unknown_count(),
        outcome,
        "the restore plan: {} layers to restore, {} kept, {} moved to their successor, {} \
         retirement flags to clear, {} keeping the service held, {} sessions to restore, {} \
         unfinished deletes to finish; outcome: {outcome}; {} layers restored from an older \
         backup because the newest was unreadable; {} layers restored from an older backup \
         because the newest needs a backup file absent from the backup store at T_safe; {} \
         sessions named by surviving data or a backup have no record and no session backup and \
         are not restored",
        counts.restore,
        counts.keep,
        counts.remap,
        counts.clear,
        counts.held,
        counts.sessions,
        counts.deletes,
        counts.restored_from_older,
        counts.restored_from_older_absent,
        counts.sessions_without_backup
    );
}

// ── The restore summary (SR-14.6) ───────────────────────────────────────────

/// One layer's referenced external values, by where their bytes are after
/// the run (the finalised coverage verdicts, `CoverageReport::verdicts`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueClasses {
    /// In Valkey: verdict `Valkey`, `Tails` (backfilled into Valkey by the
    /// writer) or `OwnPut`.
    pub in_store: u64,
    /// Only in the backup store (`Backup`): fetched on its first read.
    pub backup_only: u64,
    /// In no store (`Uncovered`).
    pub uncovered: u64,
    /// Unconfirmed (`Unknown`, or a re-probe the finalise left pending).
    pub unknown: u64,
}

impl ValueClasses {
    /// Every referenced value, whatever its class.
    pub fn total(&self) -> u64 {
        self.in_store + self.backup_only + self.uncovered + self.unknown
    }

    fn add(&mut self, verdict: Option<&HashVerdict>) {
        match verdict {
            Some(HashVerdict::Valkey | HashVerdict::Tails | HashVerdict::OwnPut) => {
                self.in_store += 1;
            }
            Some(HashVerdict::Backup) => self.backup_only += 1,
            Some(HashVerdict::Uncovered) => self.uncovered += 1,
            Some(HashVerdict::Unknown(_) | HashVerdict::PendingReprobe) | None => {
                self.unknown += 1;
            }
        }
    }
}

/// Each object's referenced values by class: the coverage check's
/// references (`plan.coverage.references`) inverted against `report`'s
/// final verdicts.
pub fn value_classes_by_object(
    plan: &RestorePlan,
    report: &CoverageReport,
) -> BTreeMap<u64, ValueClasses> {
    let mut classes: BTreeMap<u64, ValueClasses> = BTreeMap::new();
    for (hash, objects) in &plan.coverage.references {
        let verdict = report
            .verdicts
            .get(hash)
            .or_else(|| plan.coverage.verdicts.get(hash));
        for object_id in objects {
            classes.entry(*object_id).or_default().add(verdict);
        }
    }
    classes
}

/// One summary line's facts: a layer the writer restored or whose
/// retirement marks it cleared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSummary {
    /// The object.
    pub object_id: u64,
    /// The session the plan maps it under.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// What the writer wrote.
    pub write: ObjectWrite,
    /// Its referenced values by class.
    pub values: ValueClasses,
}

/// One [`LayerSummary`] per object the writer restored or cleared, in object
/// order.
pub fn layer_summaries(
    plan: &RestorePlan,
    report: &CoverageReport,
    execution: &ExecutionReport,
) -> Vec<LayerSummary> {
    let classes = value_classes_by_object(plan, report);
    execution
        .objects
        .iter()
        .map(|(object_id, write)| {
            let named = plan.objects.iter().find(|o| {
                o.object_id == *object_id
                    && matches!(
                        o.action,
                        ObjectAction::Restore { .. } | ObjectAction::ClearFlags { .. }
                    )
            });
            LayerSummary {
                object_id: *object_id,
                session_id: named.map(|o| o.session_id.clone()).unwrap_or_default(),
                layer_name: named.map(|o| o.layer_name.clone()).unwrap_or_default(),
                write: write.clone(),
                values: classes.get(object_id).copied().unwrap_or_default(),
            }
        })
        .collect()
}

/// A layer's summary line: what it was restored from and where it resumes,
/// with its values' classes, or that its retirement marks were cleared. The
/// values in no store and the unconfirmed ones are named only when there are
/// any (a run that stays held), so the classes always add up to the total.
pub fn render_layer_summary(layer: &LayerSummary) -> String {
    let LayerSummary {
        object_id,
        session_id,
        layer_name,
        write,
        values,
    } = layer;
    match write.kind {
        WriteKind::Restored {
            head_seq,
            base_seq,
            tails,
        } => {
            let mut line = format!(
                "layer {layer_name} of session {session_id} (object {object_id}): restored from \
                 backup generation {head_seq} (base {base_seq}, {tails} tails) in {} \
                 transactions; resumes at sequence {} (band {}); of its {} large values, {} are \
                 in the backing store and {} are fetched from the backup store on their first \
                 read",
                write.batches.len(),
                write.seeded_seq,
                write.band,
                values.total(),
                values.in_store,
                values.backup_only
            );
            if values.uncovered > 0 || values.unknown > 0 {
                line.push_str(&format!(
                    ", {} in no store and {} unconfirmed",
                    values.uncovered, values.unknown
                ));
            }
            line
        }
        WriteKind::FlagsCleared => format!(
            "layer {layer_name} of session {session_id} (object {object_id}): kept its surviving \
             copy and cleared its retirement marks; resumes at sequence {} (band {})",
            write.seeded_seq, write.band
        ),
    }
}

/// The summary's totals line's facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SummaryTotals {
    /// Layers restored from a backup.
    pub restored: usize,
    /// Layers whose surviving copy was kept as it is.
    pub kept: usize,
    /// Layers moved to their complete successor.
    pub moved: usize,
    /// Kept layers whose retirement marks were cleared.
    pub cleared: usize,
    /// Sessions restored (Step B, or Step A alone).
    pub sessions: usize,
    /// Unfinished session deletes finished.
    pub deletes: usize,
    /// Distinct large values referenced.
    pub values: usize,
    /// Of them, only in the backup store: the count of values left to fetch
    /// on their first read after the restart (final verdicts `Backup`).
    pub fetch_on_first_read: usize,
    /// In no store.
    pub uncovered: usize,
    /// Unconfirmed.
    pub unknown: usize,
}

/// The totals over a written plan.
pub fn summary_totals(
    plan: &RestorePlan,
    report: &CoverageReport,
    execution: &ExecutionReport,
) -> SummaryTotals {
    let mut totals = SummaryTotals {
        kept: plan.counts().keep,
        moved: execution.remapped.len(),
        sessions: execution.sessions_restored + execution.sessions_path_index_only,
        deletes: execution.deletes_completed.len(),
        values: report.verdicts.len(),
        fetch_on_first_read: report
            .verdicts
            .values()
            .filter(|v| matches!(v, HashVerdict::Backup))
            .count(),
        uncovered: report.uncovered_count(),
        unknown: report.unknown_count(),
        ..SummaryTotals::default()
    };
    for write in execution.objects.values() {
        match write.kind {
            WriteKind::Restored { .. } => totals.restored += 1,
            WriteKind::FlagsCleared => totals.cleared += 1,
        }
    }
    totals
}

/// The `mode` Job's totals line.
pub fn render_summary_totals(mode: JobMode, totals: &SummaryTotals) -> String {
    format!(
        "the {mode} Job's summary: {} layers restored, {} kept, {} moved to their successor, {} \
         with retirement marks cleared, {} sessions restored, {} unfinished deletes finished; {} \
         large values referenced: {} only in the backup store and fetched on their first read \
         after the restart, {} in no store, {} unconfirmed",
        totals.restored,
        totals.kept,
        totals.moved,
        totals.cleared,
        totals.sessions,
        totals.deletes,
        totals.values,
        totals.fetch_on_first_read,
        totals.uncovered,
        totals.unknown
    )
}

/// Log the `mode` Job's restore summary (SR-14.6): one INFO per restored
/// layer and per layer whose retirement marks were cleared, then the totals
/// INFO, which also carries the recount's counts as fields. Kept layers
/// appear only in the totals; held layers and uncovered or unknown values
/// are named by their own WARNs ([`log_held`], [`log_coverage_report`]).
pub fn log_restore_summary(
    mode: JobMode,
    plan: &RestorePlan,
    report: &CoverageReport,
    execution: &ExecutionReport,
    recount: &RecountReport,
) {
    for layer in layer_summaries(plan, report, execution) {
        tracing::info!(
            object_id = layer.object_id,
            session_id = %layer.session_id,
            layer_name = %layer.layer_name,
            seeded_seq = layer.write.seeded_seq,
            band = layer.write.band,
            batches = layer.write.batches.len(),
            values = layer.values.total(),
            values_in_store = layer.values.in_store,
            values_backup_only = layer.values.backup_only,
            values_uncovered = layer.values.uncovered,
            values_unknown = layer.values.unknown,
            "{}",
            render_layer_summary(&layer)
        );
    }
    let totals = summary_totals(plan, report, execution);
    tracing::info!(
        restored = totals.restored,
        kept = totals.kept,
        moved = totals.moved,
        cleared = totals.cleared,
        sessions = totals.sessions,
        deletes = totals.deletes,
        values = totals.values,
        fetch_on_first_read = totals.fetch_on_first_read,
        uncovered = totals.uncovered,
        unknown = totals.unknown,
        recount_values = recount.values,
        recount_created = recount.created,
        recount_raised = recount.raised,
        recount_lowered = recount.lowered,
        recount_unchanged = recount.unchanged,
        recount_repaired = recount.repaired,
        "{}",
        render_summary_totals(mode, &totals)
    );
}

// ── Accept-loss's lines (SR-10) ─────────────────────────────────────────────

/// AL2 (INFO): accept-loss's restore phase is done.
pub fn render_accept_loss_start(held: usize, uncovered: usize, unknown: usize) -> String {
    format!(
        "the accept-loss Job restored everything it could; {held} items, {uncovered} large \
         values in no store and {unknown} unconfirmed remain; it now discards what is proven lost"
    )
}

/// AL3 (ERROR, exit 3): `n` items could not be confirmed lost, so nothing is
/// discarded. Names the second confirm as the last-resort remedy (option E).
pub fn render_unconfirmed_refusal(n: usize) -> String {
    format!(
        "the accept-loss Job restored everything it could and recounted the references of the \
         large values, but discarded nothing: {n} items could not be confirmed lost, because the \
         backup store answered with an error rather than an absence (each named above); \
         accept-loss discards only what is proven lost; fix the backup store's access and run \
         accept-loss again, or, as a last resort, run it with a second --confirm \
         {CONFIRM_DISCARD_UNCONFIRMED_DATA} to discard these items too; the service stays held \
         and the Job exits 3"
    )
}

/// AL12 (WARN, option E): one item discarded although it could not be
/// confirmed lost.
pub fn render_unconfirmed_discard(item: &Unconfirmed) -> String {
    format!(
        "the accept-loss Job could not confirm {} lost: the backup store answered with an error \
         ({}); it runs with --confirm {CONFIRM_DISCARD_UNCONFIRMED_DATA}, so it {}",
        item.item, item.error, item.action
    )
}

/// What accept-loss is about to discard, counted for AL4.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiscardPlanCounts {
    /// Lost layers discarded (registered, wiped, unmapped).
    pub layers: usize,
    /// Lost layers only unmapped, their over-limit backups kept
    /// (`LostLayerKind::OverLimitBackups`).
    pub over_limit: usize,
    /// Lost sessions.
    pub sessions: usize,
    /// Distinct values removed.
    pub values: usize,
    /// Layers values are removed from.
    pub value_layers: usize,
    /// Session backups the Job rewrites, at most.
    pub rewrites: usize,
    /// Objects it registers on the dead-object set.
    pub registered: usize,
    /// Items it discards although unconfirmed (option E), counted among the
    /// above.
    pub unconfirmed: usize,
    /// Items it could not confirm and keeps as their WARN says (option E).
    pub unconfirmed_kept: usize,
}

impl DiscardPlanCounts {
    /// The counts of `lost`: its rewrites are its own and every lost layer's
    /// session's; its registrations, as the discards count them, every
    /// discarded lost layer's object once and each lost session's objects.
    pub fn of(lost: &LostSet) -> Self {
        let mut rewrites: BTreeSet<&str> = lost.rewrites.iter().map(String::as_str).collect();
        rewrites.extend(lost.layers.iter().map(|l| l.session_id.as_str()));
        let discarded = || lost.layers.iter().filter(|l| l.kind.discards_object());
        let layer_objects: BTreeSet<u64> = discarded().map(|l| l.object_id).collect();
        let registered =
            layer_objects.len() + lost.sessions.iter().map(|s| s.objects.len()).sum::<usize>();
        let (unconfirmed, unconfirmed_kept) = lost.unconfirmed_counts();
        Self {
            layers: discarded().count(),
            over_limit: lost.layers.len() - discarded().count(),
            sessions: lost.sessions.len(),
            values: lost.value_count(),
            value_layers: lost.values.len(),
            rewrites: rewrites.len(),
            registered,
            unconfirmed,
            unconfirmed_kept,
        }
    }
}

/// The clause AL4 and AL11 add for the layers only unmapped, their
/// over-limit backups kept; empty when there are none.
fn over_limit_clause(over_limit: usize, verb: &str) -> String {
    if over_limit == 0 {
        return String::new();
    }
    format!(
        "; {verb} {over_limit} more layers whose only backups are over the per-delta entry \
         limit, keeping those backups for a release whose limit fits them"
    )
}

/// The clause AL4 and AL11 add under option E: the unconfirmed items
/// discarded, counted among the totals, and those kept as their WARN says.
fn unconfirmed_clause(discarded: usize, kept: usize, tense: (&str, &str)) -> String {
    let (are, runs) = tense;
    let mut clause = format!(
        "; {discarded} items among these could not be confirmed lost and {are} discarded because \
         it {runs} with --confirm {CONFIRM_DISCARD_UNCONFIRMED_DATA}"
    );
    if kept > 0 {
        clause.push_str(&format!(
            "; {kept} more could not be confirmed and {are} left as their WARN says"
        ));
    }
    clause
}

/// AL4 (INFO, before the first discard): what the Job discards, and the
/// layers it only unmaps. Under option E it also counts apart the
/// unconfirmed items it discards and those it keeps.
pub fn render_discard_plan(counts: &DiscardPlanCounts, policy: UnconfirmedPolicy) -> String {
    let DiscardPlanCounts {
        layers,
        over_limit,
        sessions,
        values,
        value_layers,
        rewrites,
        registered,
        unconfirmed,
        unconfirmed_kept,
    } = *counts;
    let mut line = format!(
        "the accept-loss Job discards {layers} layers, {sessions} sessions and {values} large \
         values from {value_layers} layers, and rewrites {rewrites} session backups; \
         {registered} layers are registered as deleted"
    );
    line.push_str(&over_limit_clause(over_limit, "it unmaps"));
    if policy == UnconfirmedPolicy::Discard {
        line.push_str(&unconfirmed_clause(
            unconfirmed,
            unconfirmed_kept,
            ("are", "runs"),
        ));
    }
    line
}

/// AL13 (WARN): a lost layer whose only backups are over the per-delta
/// entry limit, unmapped with its backups kept (PD-4; reported as PD-7
/// reports an unreferenced backup).
pub fn render_over_limit_layer(layer: &LostLayer) -> String {
    format!(
        "the accept-loss Job removed the mapping of layer {} of session {} (object {}) but kept \
         its backups and did not register the object as deleted: {}; no mapping or session \
         backup names it any more, so no restore brings it back by itself: to recover it, \
         open a support case with this line (it needs a release whose limit fits those backups)",
        layer.layer_name, layer.session_id, layer.object_id, layer.why
    )
}

/// AL5 (WARN): one lost layer discarded.
pub fn render_lost_layer(layer: &LostLayer) -> String {
    format!(
        "the accept-loss Job discarded layer {} of session {} (object {}): {}; its mapping is \
         removed and the object is registered as deleted",
        layer.layer_name, layer.session_id, layer.object_id, layer.why
    )
}

/// AL6 (WARN): one lost session discarded, with `layers` layers removed.
pub fn render_lost_session(
    session_id: &str,
    evidence: &str,
    layers: usize,
    backup_deleted: bool,
) -> String {
    let mut line = format!(
        "the accept-loss Job discarded session {session_id}: {evidence} names it, but it has no \
         record in the store and no usable session backup; its {layers} layers are removed"
    );
    if backup_deleted {
        line.push_str(", and its session backup is deleted");
    }
    line
}

/// AL7 (WARN): `values` large values removed from one layer, which resumes
/// at `seq` (band `band`).
pub fn render_value_removal(
    layer: (&str, &str, u64),
    values: usize,
    sites: &[ValueSite],
    seq: u64,
    band: u64,
) -> String {
    let (session_id, layer_name, object_id) = layer;
    format!(
        "the accept-loss Job removed {values} large values that are in no store from layer \
         {layer_name} of session {session_id} (object {object_id}): {}; the layer resumes at \
         sequence {seq} (band {band})",
        render_sites(sites, DR_COVERAGE_REPORT_ENTRIES_PER_LAYER)
    )
}

/// AL8 (WARN): a lost layer's mapping found naming `found`, not the object
/// its plan saw, left as found (defer-55).
pub fn render_left_mapping(
    session_id: &str,
    layer_name: &str,
    found: u64,
    expected: u64,
) -> String {
    format!(
        "the accept-loss Job left layer {layer_name} of session {session_id} mapped to object \
         {found}: its plan saw object {expected}, which is lost; it removes a mapping only while \
         it names what its plan saw"
    )
}

/// AL9 (ERROR, exit 1): a lost layer's object found complete at its
/// header's re-read (defer-55).
pub fn render_lost_object_complete(layer: &LostLayer) -> String {
    format!(
        "the accept-loss Job found object {} (layer {} of session {}) complete in the store, \
         though its plan found it lost; another writer is active; it stops discarding and exits 1",
        layer.object_id, layer.layer_name, layer.session_id
    )
}

/// AL9's companion (WARN), after an AL9 at the header's re-read before the
/// wipes (D2): `removed` is what the Job removed from the dead-object set,
/// every object this run's registration added and did not wipe, sorted;
/// `foreign` names AL9's object when another writer had registered it before
/// this run, so it stays. `None` when there is nothing to say: nothing
/// removed and nothing kept.
pub fn render_lost_object_unregistered(removed: &[u64], foreign: Option<u64>) -> Option<String> {
    let kept = foreign.map(|object_id| {
        format!(
            "object {object_id} stays on it: another writer registered it as deleted before this \
             run did"
        )
    });
    if removed.is_empty() {
        return kept.map(|kept| {
            format!(
                "the accept-loss Job removed nothing from the dead-object set before exiting: \
                 {kept}"
            )
        });
    }
    let list = removed
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let line = format!(
        "the accept-loss Job removed objects {list} from the dead-object set before exiting: this \
         run registered them as deleted and wiped none of them, so its retry reads their headers \
         before it registers them again"
    );
    Some(match kept {
        Some(kept) => format!("{line}; {kept}"),
        None => line,
    })
}

/// AL10 (ERROR, exit 1): the discard check found `defect`.
pub fn render_discard_defect(defect: &DiscardDefect) -> String {
    format!(
        "the accept-loss Job's discards are inconsistent with its plan: {defect}; a defect in the \
         Job, not the store; it discarded nothing and exits 1"
    )
}

/// AL11 (INFO, after restore's summary, before the un-hold): accept-loss's
/// totals, and the layers it only unmapped. Under option E it also counts
/// apart the unconfirmed items it discarded and those it kept.
pub fn render_discard_summary(report: &DiscardReport, policy: UnconfirmedPolicy) -> String {
    let mut line = format!(
        "the accept-loss Job's summary: discarded {} layers and {} sessions, removed {} large \
         values from {} layers, left {} mappings as found, registered {} layers as deleted, \
         deleted {} session backups and rewrote {}",
        report.layers_discarded.len(),
        report.sessions_discarded.len(),
        report.value_count(),
        report.values.len(),
        report.left.len(),
        report.registered,
        report.session_backups_deleted,
        report.rewritten.len()
    );
    line.push_str(&over_limit_clause(
        report.over_limit_unmapped.len(),
        "unmapped",
    ));
    if policy == UnconfirmedPolicy::Discard {
        line.push_str(&unconfirmed_clause(
            report.unconfirmed,
            report.unconfirmed_kept,
            ("were", "ran"),
        ));
    }
    line
}

#[cfg(test)]
mod tests {
    //! The report's pure rendering (U6) over both forms of an object's state,
    //! the step-down line's cause (PD-6 condition 1), and the restore
    //! summary's classes, per-layer lines and totals (SR-14.6) over
    //! hand-built plans, reports and writer reports. The plan's log lines are
    //! pinned by the restore-plan integration tests, which capture them from
    //! a real plan, and the summary's by `restore_finish_test`; defer-53's
    //! advisory WARN, which no integration fixture reaches, is captured here.

    use std::collections::BTreeSet;

    use ordered_float::OrderedFloat;

    use super::super::model::{
        BandInputs, CoveragePlan, LineageFootprint, ObjectPlan, PlanEpoch, PlannedLineage,
        RestoreReason,
    };
    use super::*;
    use crate::checkpoint::store::{CheckpointHandle, CheckpointMeta};
    use crate::domain::types::{CompactState, Field, FieldValue, ObjectState, Section, TimeSample};
    use crate::dr::job::execute::BatchRecord;

    const H1: ExtHash = [0x11; 32];
    const H2: ExtHash = [0x22; 32];
    const H3: ExtHash = [0x33; 32];

    fn section(section_id: u64, parent_id: u64, name: &str, section_type: u8) -> Section {
        Section {
            section_id,
            parent_id,
            name: name.to_owned(),
            section_type,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    fn external(hash: ExtHash) -> FieldValue {
        FieldValue::External {
            hash,
            size: 4_096,
            ext_value_index: 1,
        }
    }

    fn field(section_id: u64, key_name: &str, value: FieldValue) -> Field {
        Field {
            section_id,
            key_name: key_name.to_owned(),
            value,
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        }
    }

    fn sample(section_id: u64, time: f64, value: FieldValue) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value,
            version: 1,
        }
    }

    /// `/World/Mesh` with two attributes, `points` and `uvs`; rows of three
    /// external values and one inline value, on the attributes and on the
    /// prim itself.
    fn rows() -> (Vec<Section>, Vec<Field>, Vec<TimeSample>) {
        let sections = vec![
            section(1, 0, "World", SPEC_PRIM),
            section(2, 1, "Mesh", SPEC_PRIM),
            section(3, 2, "points", SPEC_ATTRIBUTE),
            section(4, 2, "uvs", SPEC_ATTRIBUTE),
        ];
        let fields = vec![
            field(3, "default", external(H1)),
            field(2, "customData", external(H2)),
            field(4, "default", external(H3)),
            field(2, "kind", FieldValue::Inline(b"component".to_vec())),
        ];
        let samples = vec![
            sample(3, 1.0, external(H1)),
            sample(3, 2.0, external(H1)),
            sample(2, 0.5, external(H2)),
            sample(4, 1.0, FieldValue::Inline(vec![1])),
        ];
        (sections, fields, samples)
    }

    fn full_state() -> ObjectState {
        let (sections, fields, samples) = rows();
        ObjectState {
            sections: sections.into_iter().map(|s| (s.section_id, s)).collect(),
            fields: fields
                .into_iter()
                .map(|f| ((f.section_id, f.key_name.clone()), f))
                .collect(),
            time_samples: samples
                .into_iter()
                .map(|s| ((s.section_id, s.time), s))
                .collect(),
            ..ObjectState::default()
        }
    }

    fn compact_state() -> CompactState {
        let (sections, fields, samples) = rows();
        let mut state = CompactState {
            sections: sections.into_iter().map(|s| (s.section_id, s)).collect(),
            ..CompactState::default()
        };
        for f in fields {
            state.fields.insert_row(f);
        }
        for s in samples {
            state.time_samples.insert_row(s);
        }
        state
    }

    fn site(prim_path: &str, attribute: &str, hash: ExtHash, count: u64) -> ValueSite {
        ValueSite {
            prim_path: prim_path.to_owned(),
            attribute: attribute.to_owned(),
            hash,
            count,
        }
    }

    /// A row on an attribute section names its prim and the attribute; a
    /// field on a prim names the prim and the field key; time samples read
    /// "(time samples)", of the attribute when on one. Sites group by prim,
    /// attribute and value with their row count; values not asked for and
    /// inline rows are left out.
    #[test]
    fn value_sites_names_prim_and_attribute_per_row_kind_and_counts_rows() {
        let wanted = BTreeSet::from([H1, H2]);
        let expected = vec![
            site("/World/Mesh", "(time samples)", H2, 1),
            site("/World/Mesh", "customData", H2, 1),
            site("/World/Mesh", "points", H1, 1),
            site("/World/Mesh", "points (time samples)", H1, 2),
        ];

        assert_eq!(value_sites(&full_state(), &wanted), expected, "full form");
        assert_eq!(
            value_sites(&compact_state(), &wanted),
            expected,
            "compact form"
        );
    }

    /// What a restored object contributes to the coverage check: every
    /// external value of its field and time-sample rows, in both forms.
    #[test]
    fn referenced_hashes_takes_every_external_row_value() {
        let all = BTreeSet::from([H1, H2, H3]);
        assert_eq!(referenced_hashes(&full_state()), all);
        assert_eq!(referenced_hashes(&compact_state()), all);
    }

    /// The path is the section names from the layer's root down: a
    /// top-level prim is its first element, a top-level section of another
    /// type is left out.
    #[test]
    fn section_path_keeps_a_top_level_prim_and_drops_a_top_level_non_prim() {
        let mut state = ObjectState::default();
        for s in [
            section(10, 0, "root", 0),
            section(11, 10, "A", SPEC_PRIM),
            section(12, 11, "B", SPEC_PRIM),
            section(20, 0, "World", SPEC_PRIM),
        ] {
            state.sections.insert(s.section_id, s);
        }

        assert_eq!(section_path(&state, 12), "/A/B");
        assert_eq!(section_path(&state, 11), "/A");
        assert_eq!(section_path(&state, 20), "/World");
        assert_eq!(section_path(&state, 0), "/", "the layer itself");
    }

    /// A walk that cannot reach the root — a missing parent, or a cycle the
    /// depth limit stops — names the part it could walk under `…`.
    #[test]
    fn section_path_marks_a_missing_parent_or_a_cycle_with_an_ellipsis() {
        let mut state = ObjectState::default();
        for s in [
            section(30, 99, "orphan", SPEC_PRIM),
            section(40, 41, "a", SPEC_PRIM),
            section(41, 40, "b", SPEC_PRIM),
        ] {
            state.sections.insert(s.section_id, s);
        }

        assert_eq!(section_path(&state, 30), "/…/orphan");
        let cycle = section_path(&state, 40);
        assert!(
            cycle.starts_with("/…/"),
            "{}",
            &cycle[..cycle.len().min(40)]
        );
        assert!(cycle.ends_with("/b/a"), "the walk names what it reached");
    }

    /// A value's prefix is the first 16 hex characters of its hash, as the
    /// ext-mirror's WARNs log it.
    #[test]
    fn hash_prefix_is_the_first_sixteen_hex_characters() {
        assert_eq!(hash_prefix(&[0xab; 32]), "abababababababab");
        let mut hash = [0u8; 32];
        hash[0] = 0x01;
        hash[7] = 0xfe;
        hash[8] = 0xff;
        assert_eq!(hash_prefix(&hash), "01000000000000fe");
    }

    /// Each site renders as `{prim path} {attribute} ×{count} {prefix}`,
    /// `; `-separated, at most `cap` of them, then `and {k} more`.
    #[test]
    fn render_sites_names_at_most_cap_sites_then_counts_the_rest() {
        let sites = vec![
            site("/World", "points", H1, 3),
            site("/World", "uvs", H2, 1),
            site("/World/Mesh", "(time samples)", H3, 2),
        ];

        assert_eq!(
            render_sites(&sites, 2),
            "/World points ×3 1111111111111111; /World uvs ×1 2222222222222222; and 1 more"
        );
        assert_eq!(
            render_sites(&sites, 3),
            "/World points ×3 1111111111111111; /World uvs ×1 2222222222222222; \
             /World/Mesh (time samples) ×2 3333333333333333"
        );
        assert_eq!(render_sites(&[], 2), "");
    }

    /// One layer's line names at most 100 sites.
    #[test]
    fn render_sites_at_the_per_layer_cap_names_one_hundred_sites() {
        assert_eq!(DR_COVERAGE_REPORT_ENTRIES_PER_LAYER, 100);
        let sites: Vec<ValueSite> = (0..=100u8)
            .map(|i| site("/World", &format!("a{i:03}"), [i; 32], 1))
            .collect();

        let rendered = render_sites(&sites, DR_COVERAGE_REPORT_ENTRIES_PER_LAYER);

        assert_eq!(rendered.matches(" ×1 ").count(), 100);
        assert!(rendered.ends_with("; and 1 more"), "{rendered}");
    }

    fn generation(seq: u64, reason_label: &'static str, absent: bool) -> AbandonedGeneration {
        AbandonedGeneration {
            seq,
            reason_label,
            absent,
        }
    }

    /// PD-6 condition 1: the step-down line names the newest abandoned
    /// generation and its cause in words — unreadable for a bytes verdict,
    /// absent from the backup store at `T_safe` for an absence — and lists
    /// every abandoned generation with its label.
    #[test]
    fn step_down_text_words_the_newest_abandoned_generations_cause() {
        let bytes = step_down_text(&[
            generation(30, "stepped_down_compose", false),
            generation(20, "stepped_down_link_broken", true),
        ]);
        assert_eq!(bytes.newest, 30);
        assert_eq!(bytes.cause, "was unreadable (stepped_down_compose)");
        assert_eq!(
            bytes.list,
            "30 (stepped_down_compose), 20 (stepped_down_link_broken)"
        );

        let absent = step_down_text(&[
            generation(30, "stepped_down_missing_meta", true),
            generation(20, "stepped_down_decode", false),
        ]);
        assert_eq!(absent.newest, 30);
        assert_eq!(
            absent.cause,
            "needs a backup file that is absent from the backup store at T_safe \
             (stepped_down_missing_meta)"
        );
    }

    /// A value no row of the layer references names no site.
    #[test]
    fn value_sites_is_empty_for_a_value_no_row_references() {
        let sites = value_sites(&full_state(), &BTreeSet::from([[0x99; 32]]));
        assert!(sites.is_empty(), "{sites:?}");
    }

    // ── The restore summary (SR-14.6) ───────────────────────────────────

    const SESSION: &str = "s-sum";

    fn band() -> BandInputs {
        BandInputs {
            floor_seq: 10,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        }
    }

    /// `object_id`'s plan entry, layer `l{object_id}.usda`, with `action`.
    fn planned(object_id: u64, action: ObjectAction) -> ObjectPlan {
        ObjectPlan {
            object_id,
            session_id: SESSION.to_owned(),
            layer_name: format!("l{object_id}.usda"),
            action,
        }
    }

    /// A restore of `object_id` from a base at seq 20.
    fn restore(object_id: u64) -> ObjectAction {
        ObjectAction::Restore {
            reason: RestoreReason::HeaderAbsent,
            lineage: Box::new(PlannedLineage {
                base: CheckpointHandle::base(object_id, 20, String::new()),
                tails: Vec::new(),
                head_seq: 20,
                meta: CheckpointMeta::default(),
                tail_carried: Box::new([]),
                step_downs: Vec::new(),
                footprint: LineageFootprint::default(),
            }),
            band: band(),
        }
    }

    /// A plan of `objects` whose coverage check found `references` (each
    /// value and the objects naming it) with the plan's own `verdicts`.
    fn plan_of(
        objects: Vec<ObjectPlan>,
        references: &[(ExtHash, &[u64])],
        verdicts: &[(ExtHash, HashVerdict)],
    ) -> RestorePlan {
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(1),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: Vec::new(),
            objects,
            coverage: CoveragePlan {
                verdicts: verdicts.iter().cloned().collect(),
                references: references
                    .iter()
                    .map(|(hash, ids)| (*hash, ids.iter().copied().collect()))
                    .collect(),
                objects: BTreeMap::new(),
            },
            held: Vec::new(),
            advisories: Vec::new(),
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        }
    }

    /// A finalised report with `verdicts`.
    fn report_of(verdicts: &[(ExtHash, HashVerdict)]) -> CoverageReport {
        CoverageReport {
            verdicts: verdicts.iter().cloned().collect(),
            layers: Vec::new(),
        }
    }

    /// A write of `kind` seeded at `seeded_seq` with band 1000, in
    /// `batches` transactions.
    fn write(kind: WriteKind, seeded_seq: u64, batches: usize) -> ObjectWrite {
        ObjectWrite {
            kind,
            floor: seeded_seq - 1_000,
            high_water: None,
            band: 1_000,
            seeded_seq,
            batches: vec![
                BatchRecord {
                    commands: 1,
                    timeout: std::time::Duration::from_secs(1),
                };
                batches
            ],
        }
    }

    fn restored(head_seq: u64, base_seq: u64, tails: usize) -> WriteKind {
        WriteKind::Restored {
            head_seq,
            base_seq,
            tails,
        }
    }

    const HA: ExtHash = [0xA0; 32];
    const HB: ExtHash = [0xB0; 32];
    const HC: ExtHash = [0xC0; 32];
    const HD: ExtHash = [0xD0; 32];
    const HE: ExtHash = [0xE0; 32];
    const HF: ExtHash = [0xF0; 32];
    const HG: ExtHash = [0x10; 32];
    const HH: ExtHash = [0x20; 32];

    /// Each object's values are classed by their FINAL verdict, the plan's
    /// only where the finalise has none: `Valkey`, `Tails` and `OwnPut` are
    /// in the store, `Backup` only in the backup store, `Uncovered` in no
    /// store, and `Unknown`, a re-probe left pending, or no verdict at all
    /// unconfirmed. A value counts once for every object naming it.
    #[test]
    fn value_classes_by_object_takes_the_final_verdict_over_the_plans() {
        let plan = plan_of(
            Vec::new(),
            &[
                (HA, &[1, 2]),
                (HB, &[1]),
                (HC, &[1]),
                (HD, &[1]),
                (HE, &[1]),
                (HF, &[1, 2]),
                (HG, &[1]),
                (HH, &[1]),
            ],
            &[
                (HA, HashVerdict::Unknown("before T_safe".to_owned())),
                (HB, HashVerdict::PendingReprobe),
                (HH, HashVerdict::Valkey),
            ],
        );
        let report = report_of(&[
            (HA, HashVerdict::Backup),
            (HB, HashVerdict::Tails),
            (HC, HashVerdict::OwnPut),
            (HD, HashVerdict::PendingReprobe),
            (HF, HashVerdict::Uncovered),
            (HG, HashVerdict::Unknown("probe failed".to_owned())),
        ]);

        let classes = value_classes_by_object(&plan, &report);

        assert_eq!(
            classes,
            BTreeMap::from([
                (
                    1,
                    ValueClasses {
                        in_store: 3,
                        backup_only: 1,
                        uncovered: 1,
                        unknown: 3,
                    }
                ),
                (
                    2,
                    ValueClasses {
                        in_store: 0,
                        backup_only: 1,
                        uncovered: 1,
                        unknown: 0,
                    }
                ),
            ]),
            "1: HB (Tails over the plan's pending), HC (OwnPut), HH (the plan's Valkey) in the \
             store; HA (Backup over the plan's Unknown) backup-only; HF uncovered; HD (pending), \
             HE (no verdict), HG (Unknown) unconfirmed"
        );
        assert_eq!(classes[&1].total(), 8);
    }

    /// One summary per object the writer wrote, in object order, named by
    /// the plan's restore or flag-clear of it, with its values' classes: a
    /// write the plan names by no restore or flag-clear has no names, and an
    /// object naming no value has empty classes.
    #[test]
    fn layer_summaries_name_each_written_object_from_its_plan_entry() {
        let plan = plan_of(
            vec![
                planned(101, restore(101)),
                planned(102, ObjectAction::ClearFlags { band: band() }),
                planned(103, ObjectAction::Keep),
            ],
            &[(HA, &[101])],
            &[],
        );
        let report = report_of(&[(HA, HashVerdict::Backup)]);
        let execution = ExecutionReport {
            objects: BTreeMap::from([
                (101, write(restored(20, 20, 0), 1_020, 2)),
                (102, write(WriteKind::FlagsCleared, 1_050, 1)),
                (103, write(restored(30, 30, 0), 1_030, 1)),
            ]),
            ..ExecutionReport::default()
        };

        let summaries = layer_summaries(&plan, &report, &execution);

        assert_eq!(
            summaries,
            vec![
                LayerSummary {
                    object_id: 101,
                    session_id: SESSION.to_owned(),
                    layer_name: "l101.usda".to_owned(),
                    write: execution.objects[&101].clone(),
                    values: ValueClasses {
                        backup_only: 1,
                        ..ValueClasses::default()
                    },
                },
                LayerSummary {
                    object_id: 102,
                    session_id: SESSION.to_owned(),
                    layer_name: "l102.usda".to_owned(),
                    write: execution.objects[&102].clone(),
                    values: ValueClasses::default(),
                },
                LayerSummary {
                    object_id: 103,
                    session_id: String::new(),
                    layer_name: String::new(),
                    write: execution.objects[&103].clone(),
                    values: ValueClasses::default(),
                },
            ]
        );
    }

    fn summary(write: ObjectWrite, values: ValueClasses) -> LayerSummary {
        LayerSummary {
            object_id: 101,
            session_id: SESSION.to_owned(),
            layer_name: "l101.usda".to_owned(),
            write,
            values,
        }
    }

    /// A restored layer's line (text 5) names its generation, base, tails,
    /// transactions, resume point and band, and its values in the store and
    /// backup-only; with every value in one of those two classes it names
    /// nothing more.
    #[test]
    fn render_layer_summary_of_a_writable_restore_names_two_classes() {
        let line = render_layer_summary(&summary(
            write(restored(25, 20, 2), 1_025, 3),
            ValueClasses {
                in_store: 2,
                backup_only: 1,
                ..ValueClasses::default()
            },
        ));

        assert_eq!(
            line,
            "layer l101.usda of session s-sum (object 101): restored from backup generation 25 \
             (base 20, 2 tails) in 3 transactions; resumes at sequence 1025 (band 1000); of its 3 \
             large values, 2 are in the backing store and 1 are fetched from the backup store on \
             their first read"
        );
    }

    /// In a run that stays held, a restored layer's line also names its
    /// values in no store and unconfirmed, so its classes add up to the
    /// total it names.
    ///
    /// Fail-on-old (in-place perturbation of `render_layer_summary`, red
    /// then green once restored): with the suffix block removed, the line
    /// ends at "on their first read".
    #[test]
    fn render_layer_summary_of_a_held_restore_names_the_uncovered_and_unconfirmed_values() {
        let line = render_layer_summary(&summary(
            write(restored(25, 20, 0), 1_025, 1),
            ValueClasses {
                in_store: 1,
                backup_only: 1,
                uncovered: 1,
                unknown: 1,
            },
        ));

        assert!(
            line.ends_with(
                "of its 4 large values, 1 are in the backing store and 1 are fetched from the \
                 backup store on their first read, 1 in no store and 1 unconfirmed"
            ),
            "{line}"
        );
    }

    /// A flag-cleared layer's line (text 6) says it kept its surviving copy
    /// and cleared its retirement marks, and where it resumes.
    #[test]
    fn render_layer_summary_of_a_flag_clear_says_it_kept_the_copy() {
        let line = render_layer_summary(&summary(
            write(WriteKind::FlagsCleared, 1_050, 1),
            ValueClasses::default(),
        ));

        assert_eq!(
            line,
            "layer l101.usda of session s-sum (object 101): kept its surviving copy and cleared \
             its retirement marks; resumes at sequence 1050 (band 1000)"
        );
    }

    /// The totals: restored and cleared from the writer's objects, kept from
    /// the plan, moves, sessions (both steps) and deletes from the writer,
    /// and the values from the final verdicts — every distinct value, the
    /// backup-only ones, the uncovered and the unknown.
    #[test]
    fn summary_totals_count_the_writers_work_and_the_final_verdicts() {
        let plan = plan_of(
            vec![
                planned(101, restore(101)),
                planned(102, ObjectAction::ClearFlags { band: band() }),
                planned(103, ObjectAction::Keep),
                planned(104, ObjectAction::Keep),
            ],
            &[],
            &[],
        );
        let report = report_of(&[
            (HA, HashVerdict::Backup),
            (HB, HashVerdict::Backup),
            (HC, HashVerdict::Valkey),
            (HD, HashVerdict::Uncovered),
            (HE, HashVerdict::Unknown("probe failed".to_owned())),
        ]);
        let execution = ExecutionReport {
            objects: BTreeMap::from([
                (101, write(restored(20, 20, 0), 1_020, 1)),
                (102, write(WriteKind::FlagsCleared, 1_050, 1)),
            ]),
            remapped: vec![(201, 202)],
            sessions_restored: 2,
            sessions_path_index_only: 1,
            deletes_completed: vec!["s-gone".to_owned()],
            ..ExecutionReport::default()
        };

        let totals = summary_totals(&plan, &report, &execution);

        assert_eq!(
            totals,
            SummaryTotals {
                restored: 1,
                kept: 2,
                moved: 1,
                cleared: 1,
                sessions: 3,
                deletes: 1,
                values: 5,
                fetch_on_first_read: 2,
                uncovered: 1,
                unknown: 1,
            }
        );
        assert_eq!(
            render_summary_totals(JobMode::Restore, &totals),
            "the restore Job's summary: 1 layers restored, 2 kept, 1 moved to their successor, 1 \
             with retirement marks cleared, 3 sessions restored, 1 unfinished deletes finished; 5 \
             large values referenced: 2 only in the backup store and fetched on their first read \
             after the restart, 1 in no store, 1 unconfirmed"
        );
    }

    /// defer-53's advisory WARN (text 17): a kept layer whose newer backups
    /// are exhausted with no step-down is named with its kept `seq` and the
    /// exhaustion, and told apart from the selection's unrecoverable WARN.
    /// Captured with a thread-local subscriber: `log_plan_advisories` is
    /// synchronous.
    #[test]
    fn log_plan_advisories_warns_of_a_layer_kept_with_its_backups_exhausted() {
        let mut plan = plan_of(Vec::new(), &[], &[]);
        plan.advisories = vec![Advisory::KeptWithBackupsExhausted {
            object_id: 7,
            session_id: SESSION.to_owned(),
            layer_name: "l7.usda".to_owned(),
            kept_seq: 40,
            exhaustion: "every newer listed generation is a tail with no base".to_owned(),
        }];
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = {
            let buf = std::sync::Arc::clone(&buf);
            move || CaptureWriter(std::sync::Arc::clone(&buf))
        };
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .finish();

        tracing::subscriber::with_default(subscriber, || log_plan_advisories(&plan));

        let text = String::from_utf8(buf.lock().unwrap().clone()).expect("UTF-8 log output");
        let events: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("one JSON event"))
            .collect();
        assert_eq!(events.len(), 1, "one line: {text}");
        let event = &events[0];
        assert_eq!(event["level"], "WARN");
        assert_eq!(
            event["fields"]["message"],
            "layer l7.usda of session s-sum (object 7) keeps its surviving copy at seq 40, \
             because its newer backups cannot be used (every newer listed generation is a tail \
             with no base); the restore selection's WARN about an unrecoverable layer does not \
             apply here"
        );
        assert_eq!(event["fields"]["object_id"], 7);
        assert_eq!(event["fields"]["kept_seq"], 40);
    }

    // ── LS4: the rows accept-loss removes are the sites the report names ──

    /// [`compact_state`] plus, on `points` (section 3), a legacy `-0.0`
    /// sample and a `+0.0` sample of [`H1`]: two stored rows, keyed by their
    /// bits.
    fn compact_state_with_signed_zeros() -> CompactState {
        let mut state = compact_state();
        state.time_samples.insert(3, (-0.0f64).to_bits(), Some(H1));
        state.time_samples.insert(3, 0.0f64.to_bits(), Some(H1));
        state
    }

    /// LS4 (a pin; fail-on-old by perturbation). `rows_referencing` takes
    /// exactly the rows `value_sites` names, by the same walk: per value as
    /// many rows as its sites count, each row once, in key order, a legacy
    /// `-0.0` sample its own row beside the `+0.0` one; a value asked for
    /// that no row references has no group; inline rows and values not
    /// asked for are left out; and both forms of the state agree.
    #[test]
    fn ls4_rows_referencing_takes_exactly_the_rows_value_sites_names_signed_zeros_apart() {
        let unreferenced: ExtHash = [0x99; 32];
        let wanted = BTreeSet::from([H1, H2, unreferenced]);
        let state = compact_state_with_signed_zeros();

        let rows = rows_referencing(&state, &wanted);
        let sites = value_sites(&state, &wanted);

        assert_eq!(
            rows.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([H1, H2]),
            "a value no row references has no group; H3 was not asked for"
        );
        for (hash, group) in &rows {
            let counted: u64 = sites
                .iter()
                .filter(|site| site.hash == *hash)
                .map(|site| site.count)
                .sum();
            assert_eq!(
                group.len() as u64,
                counted,
                "{}: the rows are the sites' rows",
                hash_prefix(hash)
            );
            assert_eq!(
                group.iter().collect::<BTreeSet<_>>().len(),
                group.len(),
                "{}: each row once",
                hash_prefix(hash)
            );
        }
        let sample = |section_id: u64, time: f64| ValueRowKey::TimeSample {
            section_id,
            time_bits: time.to_bits(),
        };
        let field = |section_id: u64, key_name: &str| ValueRowKey::Field {
            section_id,
            key_name: key_name.to_owned(),
        };
        assert_eq!(
            rows[&H1],
            vec![
                field(3, "default"),
                sample(3, 0.0),
                sample(3, 1.0),
                sample(3, 2.0),
                sample(3, -0.0),
            ],
            "the -0.0 sample is a row of its own, by its bits"
        );
        assert_eq!(rows[&H2], vec![field(2, "customData"), sample(2, 0.5)]);
        assert_eq!(
            rows_referencing(&full_state(), &wanted),
            rows_referencing(&compact_state(), &wanted),
            "the full and compact forms agree"
        );
    }

    /// A writer into a shared buffer, for [`log_plan_advisories_warns_of_a_layer_kept_with_its_backups_exhausted`].
    struct CaptureWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
