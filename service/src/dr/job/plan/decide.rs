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

//! The restore planner's pure core (SDD SR-9.3 to SR-9.5): the session rules,
//! each object's decision before and after its lineage is selected, the
//! Job's absence policy over a lineage walk ([`judge_lineage`]), the
//! coverage merge, listing trust and the `T_safe` trigger set; and the
//! writer's consistency check of a finished plan ([`check_executable`]).
//!
//! No I/O and no waiting: everything here is a function of the facts the
//! planner read, so it is unit-tested without a store. A source scan pins
//! that this file names no store client, no backup-store trait and no
//! runtime.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::model::{
    AbandonedGeneration, Advisory, BandInputs, ExtHash, HashVerdict, HeldItem, HeldReason,
    LineageOutcome, LineageVerdict, ObjectAction, ObjectPlan, PlannedLineage, RestorePlan,
    RestoreReason, SessionAction, SessionPlan, SessionReadFailure, SessionSeen,
};
use crate::checkpoint::lineage::{Exhaustion, LineageBreak, StepDownRecord};
use crate::checkpoint::session_payload::SessionCheckpoint;
use crate::checkpoint::store::CheckpointHandle;
use crate::dr::job::absence::{find_contradictions, Contradiction};
use crate::dr::job::run::NoBackupStore;
use crate::storage::dr_job::{HeaderFacts, LayerMappingFact, SessionFacts};

/// What the run may read backups from, for the planning rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreAccess {
    /// A persistent backup store (`LocalDisk` or `StorageApi`).
    Persistent,
    /// No backup store.
    NoStore(NoBackupStore),
}

/// One listed session backup, as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionRead {
    /// Read and decoded.
    Present(SessionCheckpoint),
    /// The read answered absent.
    Absent,
    /// The bytes were read and did not decode.
    Undecodable(String),
    /// The read failed after the adapter's retries.
    ReadFailed(String),
}

/// What the backup store's one listing showed.
#[derive(Clone, Debug, Default)]
pub struct Listing {
    /// Each object's generations, newest first
    /// (`checkpoint::lineage::group_layer_generations`).
    pub generations: BTreeMap<u64, Vec<CheckpointHandle>>,
    /// Every object id any listed layer artifact carries, companions
    /// included.
    pub listed_object_ids: BTreeSet<u64>,
    /// The listed session backups.
    pub session_ids: BTreeSet<String>,
    /// Each listed session backup's read.
    pub sessions: BTreeMap<String, SessionRead>,
    /// Whether the listing was taken at or after `T_safe`.
    pub at_or_after_t_safe: bool,
    /// The listed objects this listing's identity read (PD-1, under an empty
    /// session listing) could not identify after some generation read absent
    /// (`Ok(None)` or a missing meta companion), each with the newest such
    /// generation. Before `T_safe` each is a trigger: the absence may be a
    /// stale not-found answer hiding the S7 evidence.
    pub identity_absences: BTreeMap<u64, u64>,
}

impl Listing {
    /// Whether the listing showed no layer artifact and no session backup.
    pub fn is_all_empty(&self) -> bool {
        self.listed_object_ids.is_empty() && self.session_ids.is_empty()
    }
}

/// What the store held, read once after the quiet period.
#[derive(Clone, Debug, Default)]
pub struct StoreFacts {
    /// Every surviving object header, by object.
    pub headers: BTreeMap<u64, HeaderFacts>,
    /// Every surviving layer mapping.
    pub mappings: Vec<LayerMappingFact>,
    /// The dead-object set.
    pub dead: HashSet<u64>,
    /// Each session's facts, for every session in the universe.
    pub sessions: BTreeMap<String, SessionFacts>,
}

/// A listed object whose identity read (PD-1, under an empty session
/// listing) learned nothing because a listed generation's read failed
/// `Permanent` and no generation gave an identity. Its session may be the S7
/// evidence, so it holds the service as unknown (PD-4) rather than being
/// reported as an unreferenced backup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnidentifiedObject {
    /// The object's listed generations when it was read, newest first. A
    /// re-take over the same generations does not read it again, unless
    /// [`Self::saw_absent`].
    pub listed: Vec<u64>,
    /// The newest generation whose read failed `Permanent`.
    pub seq: u64,
    /// That read's error.
    pub error: String,
    /// Whether some generation read as absent (`Ok(None)` or a missing meta
    /// companion). An absence seen before `T_safe` can be a stale not-found
    /// answer, so the re-take at `T_safe` reads the object again even over
    /// the same generations (INV-1: an absence is re-observed once at
    /// `T_safe`); a plain all-`Permanent` read is not repeated.
    pub saw_absent: bool,
}

impl UnidentifiedObject {
    /// Whether a re-take whose listing shows `listed` for the object reads it
    /// again: when its generations changed, or when the earlier read saw an
    /// absence.
    pub fn read_again(&self, listed: &[u64]) -> bool {
        self.saw_absent || self.listed != listed
    }
}

/// The planning rules' inputs.
#[derive(Clone, Copy, Debug)]
pub struct PlanInputs<'a> {
    /// The backup store, or why there is none.
    pub access: StoreAccess,
    /// The listing (empty with no backup store).
    pub listing: &'a Listing,
    /// The store's facts.
    pub store: &'a StoreFacts,
    /// Identities read from listed objects' backups, by object: `(session,
    /// layer)` (only with an empty session listing; PD-1).
    pub identities: &'a BTreeMap<u64, (String, String)>,
    /// Listed objects whose identity read failed `Permanent`, by object
    /// (only with an empty session listing; PD-1, PD-4).
    pub unidentified: &'a BTreeMap<u64, UnidentifiedObject>,
}

/// A write-shaping absence seen before `T_safe`: the planner waits for
/// `T_safe` and plans again over a fresh listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// (a) An all-empty listing on a persistent store.
    EmptyListing,
    /// (b) A mapped object needs a backup and none is listed.
    NoListedBackup {
        /// The object.
        object_id: u64,
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
    },
    /// (c) The lineage walk saw a listed generation read as absent.
    LineageDeferred {
        /// The object.
        object_id: u64,
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The generation.
        seq: u64,
    },
    /// (d) A listed session backup read as absent.
    SessionBackupAbsent {
        /// The session.
        session_id: String,
    },
    /// (e) The contradiction check found positive evidence the listing does
    /// not show.
    Contradiction(Contradiction),
    /// (f) PD-1 under an empty session listing: a session named by evidence
    /// with no record, no deletion marker and no session backup.
    SessionNamedWithoutBackup {
        /// The session.
        session_id: String,
        /// What names it.
        evidence: String,
    },
    /// (g) PD-1 under an empty session listing: a listed object's identity
    /// read saw a generation read as absent and learned no session, so the
    /// session that may be the S7 evidence is hidden behind a possibly stale
    /// not-found answer.
    IdentityAbsent {
        /// The object.
        object_id: u64,
        /// The newest generation that read as absent.
        seq: u64,
    },
}

impl Trigger {
    /// The reason the `T_safe` wait names (`StoreClock::wait_until_t_safe`).
    pub fn why(&self) -> String {
        match self {
            Self::EmptyListing => "the backup store listed no layer backups and no session \
                                   backups before T_safe; listing it again then"
                .to_owned(),
            Self::NoListedBackup {
                object_id,
                session_id,
                layer_name,
            } => format!(
                "layer {layer_name} of session {session_id} (object {object_id}) has no backup \
                 listed before T_safe, when the backup store's not-found answers can still be \
                 stale; listing again then"
            ),
            Self::LineageDeferred {
                object_id,
                session_id,
                layer_name,
                seq,
            } => format!(
                "the backup of layer {layer_name} in session {session_id} (object {object_id}, \
                 generation {seq}) read as absent before T_safe, when the backup store's \
                 not-found answers can still be stale; reading it again then"
            ),
            Self::SessionBackupAbsent { session_id } => format!(
                "the session backup of session {session_id} read as absent before T_safe, when \
                 the backup store's not-found answers can still be stale; reading it again then"
            ),
            Self::Contradiction(contradiction) => format!(
                "the backup store's listing contradicts the store before T_safe: \
                 {contradiction}; listing again then"
            ),
            Self::SessionNamedWithoutBackup {
                session_id,
                evidence,
            } => format!(
                "session {session_id}, named by {evidence}, has no session backup listed before \
                 T_safe; listing again then"
            ),
            Self::IdentityAbsent { object_id, seq } => format!(
                "the backup of object {object_id} (generation {seq}) read as absent before \
                 T_safe while the restore was learning which session it belongs to, when the \
                 backup store's not-found answers can still be stale; reading it again then"
            ),
        }
    }
}

/// A PD-1 session under an empty session listing, at or after `T_safe`: the
/// run stays held, naming it (S7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S7Hit {
    /// The session.
    pub session_id: String,
    /// What names it.
    pub evidence: String,
}

/// What a surviving object falls back to when its lineage is not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// No survivor: nothing is written, and the object is held.
    NoSurvivor,
    /// Keep the survivor.
    Keep,
    /// Clear the survivor's stale retirement flags.
    ClearFlags(BandInputs),
}

/// An object whose plan needs its lineage selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageNeed {
    /// The object's listed generations, newest first.
    pub handles: Vec<CheckpointHandle>,
    /// The survivor's `seq`, when a complete survivor exists: only a lineage
    /// newer than it is used.
    pub floor: Option<u64>,
    /// Why it would be restored.
    pub reason: RestoreReason,
    /// What it falls back to.
    pub fallback: Fallback,
}

/// One object's decision before lineage selection.
#[derive(Clone, Debug)]
pub enum PreStep {
    /// Decided without a lineage.
    Decided {
        /// The action.
        action: ObjectAction,
        /// What holds it, if anything.
        held: Option<HeldReason>,
    },
    /// Needs its lineage selected.
    NeedsLineage(LineageNeed),
}

/// One mapped object before lineage selection.
#[derive(Clone, Debug)]
pub struct PreObject {
    /// The object.
    pub object_id: u64,
    /// The session whose mapping names it after the plan.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// Its decision so far.
    pub step: PreStep,
}

/// The plan before lineage selection.
#[derive(Clone, Debug, Default)]
pub struct PrePlan {
    /// One entry per session the plan names.
    pub sessions: Vec<SessionPlan>,
    /// What keeps the service held so far.
    pub held: Vec<HeldItem>,
    /// What is reported without holding.
    pub advisories: Vec<Advisory>,
    /// Write-shaping absences seen before `T_safe`.
    pub triggers: Vec<Trigger>,
    /// Every mapped object, and every remapped-from one.
    pub objects: Vec<PreObject>,
    /// PD-1 sessions under an empty session listing, at or after `T_safe`.
    pub s7: Vec<S7Hit>,
    /// Contradictions found at or after `T_safe`.
    pub contradictions: Vec<Contradiction>,
}

/// How a session is classified, before the mapped set is built.
enum SessionClass<'a> {
    AbandonedDelete,
    Restore(&'a SessionCheckpoint),
    /// Listed, but its backup reads as absent or could not be read: held at
    /// or after `T_safe`, a trigger before it; its surviving mappings stay.
    BackupNotRead,
    /// Its record survives and it has no listed backup.
    Kept,
    /// PD-1: named by evidence only.
    NamedWithoutBackup,
}

/// Every session the plan names: the listed session backups, the sessions
/// of surviving mappings, the `session_id` of surviving headers, and the
/// sessions read off listed backups' identities.
pub fn session_universe(
    listing: &Listing,
    store: &StoreFacts,
    identities: &BTreeMap<u64, (String, String)>,
) -> BTreeSet<String> {
    let mut universe: BTreeSet<String> = listing.session_ids.clone();
    universe.extend(session_evidence(store, identities).into_keys());
    universe
}

/// For each session named by the store or by a listed backup, the first piece
/// of evidence that names it.
fn session_evidence(
    store: &StoreFacts,
    identities: &BTreeMap<u64, (String, String)>,
) -> BTreeMap<String, String> {
    let mut evidence: BTreeMap<String, String> = BTreeMap::new();
    for mapping in &store.mappings {
        evidence
            .entry(mapping.session_id.clone())
            .or_insert_with(|| {
                format!(
                    "the surviving mapping of layer {} (object {})",
                    mapping.layer_name, mapping.object_id
                )
            });
    }
    for header in store.headers.values() {
        if header.session_id().is_empty() {
            continue;
        }
        evidence
            .entry(header.session_id().to_owned())
            .or_insert_with(|| {
                format!(
                    "the surviving header of object {} (layer {})",
                    header.object_id,
                    header.layer_name()
                )
            });
    }
    for (object_id, (session_id, layer_name)) in identities {
        if session_id.is_empty() {
            continue;
        }
        evidence.entry(session_id.clone()).or_insert_with(|| {
            format!("the listed backup of object {object_id} (layer {layer_name})")
        });
    }
    evidence
}

/// The plan before lineage selection: the session rules (SR-9.4), the
/// mapped set, each mapped object's pre-lineage decision (SR-9.5), listing
/// trust and the `T_safe` triggers known so far.
pub fn plan_pre_lineage(inputs: &PlanInputs<'_>) -> PrePlan {
    let PlanInputs {
        access,
        listing,
        store,
        identities,
        unidentified,
    } = *inputs;
    let at = listing.at_or_after_t_safe;
    let persistent = access == StoreAccess::Persistent;
    let mut pre = PrePlan::default();
    let evidence = session_evidence(store, identities);
    let universe = session_universe(listing, store, identities);

    let mut mappings_by_session: BTreeMap<&str, Vec<&LayerMappingFact>> = BTreeMap::new();
    for mapping in &store.mappings {
        mappings_by_session
            .entry(mapping.session_id.as_str())
            .or_default()
            .push(mapping);
    }

    // Classify every session.
    let mut classes: BTreeMap<&str, SessionClass<'_>> = BTreeMap::new();
    for session_id in &universe {
        let facts = store.sessions.get(session_id).cloned().unwrap_or_default();
        let class = if facts.deleted_marker_exists {
            SessionClass::AbandonedDelete
        } else if listing.session_ids.contains(session_id) {
            match listing.sessions.get(session_id) {
                Some(SessionRead::Present(checkpoint)) => SessionClass::Restore(checkpoint),
                Some(read @ (SessionRead::Undecodable(error) | SessionRead::ReadFailed(error))) => {
                    let cause = if matches!(read, SessionRead::Undecodable(_)) {
                        SessionReadFailure::Undecodable
                    } else {
                        SessionReadFailure::ReadError
                    };
                    pre.held.push(HeldItem {
                        session_id: session_id.clone(),
                        layer_name: String::new(),
                        object_id: None,
                        reason: HeldReason::SessionBackupUnreadable {
                            session_id: session_id.clone(),
                            error: error.clone(),
                            cause,
                        },
                    });
                    SessionClass::BackupNotRead
                }
                Some(SessionRead::Absent) | None => {
                    if at {
                        pre.held.push(HeldItem {
                            session_id: session_id.clone(),
                            layer_name: String::new(),
                            object_id: None,
                            reason: HeldReason::SessionBackupUnreadable {
                                session_id: session_id.clone(),
                                error: "the listed session backup reads as absent at T_safe"
                                    .to_owned(),
                                cause: SessionReadFailure::AbsentAtTSafe,
                            },
                        });
                    } else {
                        pre.triggers.push(Trigger::SessionBackupAbsent {
                            session_id: session_id.clone(),
                        });
                    }
                    SessionClass::BackupNotRead
                }
            }
        } else if facts.meta_exists {
            SessionClass::Kept
        } else {
            SessionClass::NamedWithoutBackup
        };
        classes.insert(session_id.as_str(), class);
    }

    // The mapped set: surviving mappings, then the session backups' Step-B
    // mappings, by (session, layer).
    let mut mapped: BTreeMap<(String, String), u64> = BTreeMap::new();
    for (session_id, class) in &classes {
        let session_id = *session_id;
        let surviving = mappings_by_session
            .get(session_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        match class {
            SessionClass::AbandonedDelete => {
                let mut objects: BTreeSet<u64> = surviving.iter().map(|m| m.object_id).collect();
                // Every name a mapping, a header or the session's layers
                // index still carries. The index is defensive depth for a
                // hand-edited store: `DeleteSession` empties it in the same
                // transaction that sets the deletion marker, so on a store
                // only the service wrote, a marked session's index has no
                // members.
                let mut layer_names: BTreeSet<String> =
                    surviving.iter().map(|m| m.layer_name.clone()).collect();
                if let Some(facts) = store.sessions.get(session_id) {
                    layer_names.extend(facts.layers_index.iter().cloned());
                }
                for header in store.headers.values() {
                    if header.session_id() == session_id {
                        objects.insert(header.object_id);
                        if !header.layer_name().is_empty() {
                            layer_names.insert(header.layer_name().to_owned());
                        }
                    }
                }
                let (base_path, path_name_members) = match listing.sessions.get(session_id) {
                    Some(SessionRead::Present(checkpoint)) => (
                        Some(checkpoint.base_path.clone()),
                        checkpoint.path_name_members.iter().cloned().collect(),
                    ),
                    _ => (None, Vec::new()),
                };
                let mut mappings: Vec<(String, u64)> = surviving
                    .iter()
                    .map(|m| (m.layer_name.clone(), m.object_id))
                    .collect();
                mappings.sort();
                pre.sessions.push(SessionPlan {
                    session_id: session_id.to_owned(),
                    action: SessionAction::CompleteAbandonedDelete {
                        objects: objects.into_iter().collect(),
                        layer_names: layer_names.into_iter().collect(),
                        base_path,
                        mappings,
                        path_name_members,
                    },
                });
            }
            SessionClass::NamedWithoutBackup => {
                let evidence_text = evidence
                    .get(session_id)
                    .cloned()
                    .unwrap_or_else(|| "a listed backup".to_owned());
                let surviving_objects: BTreeSet<u64> = surviving
                    .iter()
                    .map(|m| m.object_id)
                    .chain(
                        store
                            .headers
                            .values()
                            .filter(|h| h.session_id() == session_id)
                            .map(|h| h.object_id),
                    )
                    .collect();
                match access {
                    StoreAccess::NoStore(_) => {
                        if surviving_objects.is_empty() {
                            pre.held.push(HeldItem {
                                session_id: session_id.to_owned(),
                                layer_name: String::new(),
                                object_id: None,
                                reason: HeldReason::NoBackupStore,
                            });
                        }
                        for object_id in &surviving_objects {
                            pre.held.push(HeldItem {
                                session_id: session_id.to_owned(),
                                layer_name: layer_name_of(store, *object_id, surviving),
                                object_id: Some(*object_id),
                                reason: HeldReason::NoBackupStore,
                            });
                        }
                    }
                    StoreAccess::Persistent if listing.session_ids.is_empty() => {
                        if at {
                            pre.s7.push(S7Hit {
                                session_id: session_id.to_owned(),
                                evidence: evidence_text,
                            });
                        } else {
                            pre.triggers.push(Trigger::SessionNamedWithoutBackup {
                                session_id: session_id.to_owned(),
                                evidence: evidence_text,
                            });
                        }
                    }
                    StoreAccess::Persistent => {
                        let mut layers_with_backups: BTreeSet<String> = BTreeSet::new();
                        for object_id in surviving_objects.iter().chain(
                            identities
                                .iter()
                                .filter_map(|(oid, (sid, _))| (sid == session_id).then_some(oid)),
                        ) {
                            if listing.generations.contains_key(object_id) {
                                let layer = identities
                                    .get(object_id)
                                    .map(|(_, layer)| layer.clone())
                                    .unwrap_or_else(|| layer_name_of(store, *object_id, surviving));
                                layers_with_backups.insert(layer);
                            }
                        }
                        pre.advisories.push(Advisory::SessionWithoutBackup {
                            session_id: session_id.to_owned(),
                            evidence: evidence_text,
                            layers_with_backups: layers_with_backups.into_iter().collect(),
                        });
                        for header in store
                            .headers
                            .values()
                            .filter(|h| h.session_id() == session_id)
                        {
                            pre.advisories.push(Advisory::UnreferencedObject {
                                object_id: header.object_id,
                                session_id: session_id.to_owned(),
                                layer_name: header.layer_name().to_owned(),
                            });
                        }
                    }
                }
                pre.sessions.push(SessionPlan {
                    session_id: session_id.to_owned(),
                    action: SessionAction::Nothing,
                });
            }
            SessionClass::Restore(checkpoint) => {
                let meta_exists = store
                    .sessions
                    .get(session_id)
                    .is_some_and(|f| f.meta_exists);
                // `survivor_of` holds every surviving mapping, dead targets
                // included: a surviving L → X keeps an older checkpoint L → Y
                // out whether or not X is deleted (§E rule 2), so a deleted
                // newer object never lets an older generation of its layer
                // back in (rule 4). `session_mapped` holds only the live
                // targets, the ones planned below.
                let mut survivor_of: BTreeMap<&str, u64> = BTreeMap::new();
                let mut session_mapped: BTreeMap<String, u64> = BTreeMap::new();
                let mut dead_survivors: Vec<&LayerMappingFact> = Vec::new();
                for mapping in surviving {
                    survivor_of.insert(mapping.layer_name.as_str(), mapping.object_id);
                    if store.dead.contains(&mapping.object_id) {
                        dead_survivors.push(mapping);
                    } else {
                        session_mapped.insert(mapping.layer_name.clone(), mapping.object_id);
                    }
                }
                let mut writes: BTreeMap<String, u64> = BTreeMap::new();
                for (layer_name, object_id) in &checkpoint.layer_mappings {
                    if store.dead.contains(object_id) {
                        pre.advisories
                            .push(Advisory::CheckpointMappingToDeletedObject {
                                session_id: session_id.to_owned(),
                                layer_name: layer_name.clone(),
                                object_id: *object_id,
                            });
                        continue;
                    }
                    match survivor_of.get(layer_name.as_str()) {
                        Some(survivor) if *survivor >= *object_id => {}
                        _ => {
                            writes.insert(layer_name.clone(), *object_id);
                        }
                    }
                }
                // defer-51: a surviving mapping to a deleted object that Step B
                // replaces — the backup maps its layer to a newer, live object
                // (`writes` holds a layer only when its object is not dead and
                // is newer than the survivor) — is not held: Step B writes the
                // newer mapping over it. One that nothing replaces is held.
                for mapping in dead_survivors {
                    match writes.get(&mapping.layer_name) {
                        Some(replacement) => {
                            pre.advisories.push(Advisory::DeletedMappingReplaced {
                                session_id: mapping.session_id.clone(),
                                layer_name: mapping.layer_name.clone(),
                                deleted_object_id: mapping.object_id,
                                replacement_object_id: *replacement,
                            });
                        }
                        None => pre.held.push(mapped_to_deleted(mapping)),
                    }
                }
                let mut layers_index: BTreeSet<String> =
                    surviving.iter().map(|m| m.layer_name.clone()).collect();
                layers_index.extend(writes.keys().cloned());
                for (layer_name, object_id) in &writes {
                    session_mapped.insert(layer_name.clone(), *object_id);
                }
                let step_b = !meta_exists || !writes.is_empty();
                let mut filtered = SessionCheckpoint::clone(checkpoint);
                filtered.layer_mappings = writes;
                filtered.layers_index = layers_index;
                pre.sessions.push(SessionPlan {
                    session_id: session_id.to_owned(),
                    action: SessionAction::Restore {
                        checkpoint: filtered,
                        step_b,
                    },
                });
                for (layer_name, object_id) in session_mapped {
                    mapped.insert((session_id.to_owned(), layer_name), object_id);
                }
            }
            SessionClass::BackupNotRead | SessionClass::Kept => {
                for mapping in surviving {
                    if store.dead.contains(&mapping.object_id) {
                        pre.held.push(mapped_to_deleted(mapping));
                    } else {
                        mapped.insert(
                            (session_id.to_owned(), mapping.layer_name.clone()),
                            mapping.object_id,
                        );
                    }
                }
                pre.sessions.push(SessionPlan {
                    session_id: session_id.to_owned(),
                    action: SessionAction::Nothing,
                });
            }
        }
    }

    // Each mapped object (SR-9.5), remap targets planned in turn.
    let mut planned: BTreeSet<u64> = BTreeSet::new();
    for ((session_id, layer_name), object_id) in &mapped {
        plan_object(
            inputs,
            session_id,
            layer_name,
            *object_id,
            &mut planned,
            &mut pre,
        );
    }

    // A listed backup no mapping names after the plan: reported as
    // unreferenced, unless its identity read failed under an empty session
    // listing. Then its session is unknown and may be the S7 evidence, so it
    // holds as unknown (PD-4) instead.
    for object_id in listing.generations.keys() {
        if planned.contains(object_id) || store.dead.contains(object_id) {
            continue;
        }
        match unidentified.get(object_id) {
            Some(failed) if persistent && listing.session_ids.is_empty() => {
                pre.held.push(HeldItem {
                    session_id: String::new(),
                    layer_name: String::new(),
                    object_id: Some(*object_id),
                    reason: HeldReason::Unknown {
                        error: format!(
                            "no listed backup of it could be read to learn its session and layer \
                             (generation {}: {})",
                            failed.seq, failed.error
                        ),
                    },
                });
            }
            _ => pre.advisories.push(Advisory::UnreferencedBackup {
                object_id: *object_id,
            }),
        }
    }

    // Listing trust (persistent store only).
    if persistent {
        if listing.is_all_empty() && !at {
            pre.triggers.push(Trigger::EmptyListing);
        }
        // (g) An identity read's absence before `T_safe` (INV-1): the re-take
        // reads the object again, by one of two routes. An object whose reads
        // were all absent is in neither `known` nor `unidentified`, so it is
        // read like a new one; an object whose reads also failed `Permanent`
        // is in `unidentified` with `saw_absent`, so
        // `UnidentifiedObject::read_again` lets it through. The
        // `generations` check is defensive only: every entry comes from this
        // listing's own generations, so it removes nothing in a run.
        if !at {
            pre.triggers.extend(
                listing
                    .identity_absences
                    .iter()
                    .filter(|(object_id, _)| listing.generations.contains_key(*object_id))
                    .map(|(object_id, seq)| Trigger::IdentityAbsent {
                        object_id: *object_id,
                        seq: *seq,
                    }),
            );
        }
        let stamps: Vec<_> = store
            .headers
            .values()
            .filter_map(HeaderFacts::stamp)
            .collect();
        let backed_up: Vec<String> = store
            .sessions
            .iter()
            .filter(|(_, f)| f.meta_exists && f.backup_kind.as_deref() == Some("write"))
            .map(|(sid, _)| sid.clone())
            .collect();
        let listed_objects: HashSet<u64> = listing.listed_object_ids.iter().copied().collect();
        let listed_sessions: HashSet<String> = listing.session_ids.iter().cloned().collect();
        let contradictions = find_contradictions(
            &stamps,
            &backed_up,
            &store.dead,
            &listed_objects,
            &listed_sessions,
        );
        if at {
            pre.contradictions = contradictions;
        } else {
            pre.triggers
                .extend(contradictions.into_iter().map(Trigger::Contradiction));
        }
    }
    pre
}

fn mapped_to_deleted(mapping: &LayerMappingFact) -> HeldItem {
    HeldItem {
        session_id: mapping.session_id.clone(),
        layer_name: mapping.layer_name.clone(),
        object_id: Some(mapping.object_id),
        reason: HeldReason::MappedToDeletedObject {
            session_id: mapping.session_id.clone(),
            layer_name: mapping.layer_name.clone(),
        },
    }
}

/// An object's layer name: its surviving mapping's, else its header's.
fn layer_name_of(store: &StoreFacts, object_id: u64, mappings: &[&LayerMappingFact]) -> String {
    mappings
        .iter()
        .find(|m| m.object_id == object_id)
        .map(|m| m.layer_name.clone())
        .or_else(|| {
            store
                .headers
                .get(&object_id)
                .map(|h| h.layer_name().to_owned())
        })
        .unwrap_or_default()
}

/// What the store held for each session in `store.sessions` (the plan's
/// session universe): its record and deletion marker, its surviving
/// mappings, the surviving headers that name it and its layers index
/// ([`RestorePlan::sessions_seen`]).
pub fn sessions_seen(store: &StoreFacts) -> BTreeMap<String, SessionSeen> {
    let mut seen: BTreeMap<String, SessionSeen> = store
        .sessions
        .iter()
        .map(|(session_id, facts)| {
            (
                session_id.clone(),
                SessionSeen {
                    record: facts.meta_exists,
                    deletion_marker: facts.deleted_marker_exists,
                    layers_index: facts.layers_index.clone(),
                    ..SessionSeen::default()
                },
            )
        })
        .collect();
    for mapping in &store.mappings {
        if let Some(entry) = seen.get_mut(&mapping.session_id) {
            entry
                .surviving_mappings
                .push((mapping.layer_name.clone(), mapping.object_id));
        }
    }
    for header in store.headers.values() {
        if let Some(entry) = seen.get_mut(header.session_id()) {
            entry
                .header_objects
                .push((header.object_id, header.layer_name().to_owned()));
        }
    }
    for entry in seen.values_mut() {
        entry.surviving_mappings.sort();
        entry.header_objects.sort();
    }
    seen
}

/// The pre-lineage decision of one mapped object (SR-9.5's table), and of its
/// remap target in turn.
fn plan_object(
    inputs: &PlanInputs<'_>,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    planned: &mut BTreeSet<u64>,
    pre: &mut PrePlan,
) {
    if !planned.insert(object_id) {
        return;
    }
    let PlanInputs {
        access,
        listing,
        store,
        ..
    } = *inputs;
    let handles = listing
        .generations
        .get(&object_id)
        .cloned()
        .unwrap_or_default();
    let head_seq = handles.first().map(CheckpointHandle::seq);
    let header = store.headers.get(&object_id);
    let complete_seq = header
        .filter(|h| h.is_structurally_complete())
        .and_then(HeaderFacts::seq);
    let push = |pre: &mut PrePlan, step: PreStep| {
        pre.objects.push(PreObject {
            object_id,
            session_id: session_id.to_owned(),
            layer_name: layer_name.to_owned(),
            step,
        });
    };

    let (Some(header), Some(seq)) = (header, complete_seq) else {
        let reason = if header.is_none() {
            RestoreReason::HeaderAbsent
        } else {
            RestoreReason::HeaderPartial
        };
        if let StoreAccess::NoStore(_) = access {
            push(
                pre,
                PreStep::Decided {
                    action: ObjectAction::Nothing,
                    held: Some(HeldReason::NoBackupStore),
                },
            );
        } else if handles.is_empty() {
            push(
                pre,
                PreStep::Decided {
                    action: ObjectAction::Nothing,
                    held: Some(HeldReason::NoBackup),
                },
            );
            if !listing.at_or_after_t_safe {
                pre.triggers.push(Trigger::NoListedBackup {
                    object_id,
                    session_id: session_id.to_owned(),
                    layer_name: layer_name.to_owned(),
                });
            }
        } else {
            push(
                pre,
                PreStep::NeedsLineage(LineageNeed {
                    handles,
                    floor: None,
                    reason,
                    fallback: Fallback::NoSurvivor,
                }),
            );
        }
        return;
    };

    let newer_backup = head_seq.is_some_and(|head| head > seq);
    if header.is_retiring() {
        let successor = header
            .superseded_by()
            .filter(|to| *to != object_id)
            .filter(|to| {
                store.headers.get(to).is_some_and(|x| {
                    x.is_structurally_complete()
                        && x.seq().is_some()
                        && !x.is_retiring()
                        && x.session_id() == header.session_id()
                        && x.layer_name() == header.layer_name()
                })
            });
        if let Some(to) = successor {
            push(
                pre,
                PreStep::Decided {
                    action: ObjectAction::Remap {
                        from: object_id,
                        to,
                    },
                    held: None,
                },
            );
            plan_object(inputs, session_id, layer_name, to, planned, pre);
            return;
        }
        let band = BandInputs {
            floor_seq: seq,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: match header.cursor() {
                (Some(cursor_seq), Some(cursor_ms)) => Some((cursor_seq, cursor_ms)),
                _ => None,
            },
        };
        if newer_backup {
            push(
                pre,
                PreStep::NeedsLineage(LineageNeed {
                    handles,
                    floor: Some(seq),
                    reason: RestoreReason::NewerThanRetiringSurvivor { survivor_seq: seq },
                    fallback: Fallback::ClearFlags(band),
                }),
            );
        } else {
            push(
                pre,
                PreStep::Decided {
                    action: ObjectAction::ClearFlags { band },
                    held: None,
                },
            );
        }
        return;
    }
    if newer_backup {
        push(
            pre,
            PreStep::NeedsLineage(LineageNeed {
                handles,
                floor: Some(seq),
                reason: RestoreReason::NewerThanSurvivor { survivor_seq: seq },
                fallback: Fallback::Keep,
            }),
        );
    } else {
        push(
            pre,
            PreStep::Decided {
                action: ObjectAction::Keep,
                held: None,
            },
        );
    }
}

/// The Job's absence policy over one object's lineage walk (PD-6, the
/// absence rule), in order:
///
/// 1. **Defer** when, before `T_safe`, any step-down above `floor` was an
///    absence (`LinkAbsent` or `MissingMetaCompanion`), whatever the
///    outcome: the object is selected again over the `T_safe` listing.
/// 2. **Unknown** when any step-down was a budget cut, or the exhaustion is
///    `BudgetInconclusive`: a budget cut is never a step-down.
/// 3. **Defer** an `EmptyListing` or `NoBase` exhaustion before `T_safe`.
/// 4. Otherwise the outcome stands: `Selected` is used, `NotNewer` keeps the
///    survivor, another exhaustion is `BackupsUnusable` (its over-limit
///    bases left for the caller, which knows them from the walk), defer-3
///    holds, and a backup-store error is `Unknown` (PD-4).
///
/// A step-down that stands is a bytes verdict (a decode, compose or link
/// verdict on bytes the adapter returned) or an absence re-observed at or
/// after `T_safe`. A transient or IO fetch error never reaches here as a
/// step-down: the walk propagates it, so it is `Errored`.
pub fn judge_lineage(
    at_or_after_t_safe: bool,
    floor: Option<u64>,
    step_downs: &[StepDownRecord],
    outcome: LineageOutcome,
) -> LineageVerdict {
    let above_floor = |record: &&StepDownRecord| floor.is_none_or(|f| record.head_seq > f);
    if !at_or_after_t_safe {
        if let Some(record) = step_downs.iter().filter(above_floor).find(|record| {
            matches!(
                record.reason,
                LineageBreak::LinkAbsent { .. } | LineageBreak::MissingMetaCompanion { .. }
            )
        }) {
            return LineageVerdict::Defer {
                seq: record.reason.link_seq(),
            };
        }
    }
    if let Some(record) = step_downs
        .iter()
        .find(|record| matches!(record.reason, LineageBreak::Budget { .. }))
    {
        return LineageVerdict::Unknown {
            error: format!(
                "the backup store did not answer for generation {} within the selection's \
                 budget",
                record.reason.link_seq()
            ),
        };
    }
    if let LineageOutcome::Exhausted(Exhaustion::BudgetInconclusive {
        unadjudicated_bases,
        ..
    }) = &outcome
    {
        return LineageVerdict::Unknown {
            error: format!(
                "{unadjudicated_bases} full backups were never read within the selection's \
                 budget"
            ),
        };
    }
    match outcome {
        LineageOutcome::Exhausted(Exhaustion::EmptyListing) if !at_or_after_t_safe => {
            LineageVerdict::Defer { seq: 0 }
        }
        LineageOutcome::Exhausted(Exhaustion::NoBase {
            newest_sequence, ..
        }) if !at_or_after_t_safe => LineageVerdict::Defer {
            seq: newest_sequence,
        },
        LineageOutcome::Selected(selected) => LineageVerdict::Use(selected),
        LineageOutcome::NotNewer => LineageVerdict::Survivor {
            step_downs: step_downs.to_vec(),
        },
        LineageOutcome::Exhausted(exhaustion) => LineageVerdict::BackupsUnusable {
            exhaustion: describe_exhaustion(&exhaustion),
            step_downs: step_downs.to_vec(),
            // Filled by the caller from its walk (`lineage::select_within`).
            over_limit: Vec::new(),
        },
        LineageOutcome::OverLimitPlainBase { head_seq, base_seq } => {
            LineageVerdict::OverLimitPlainBase { head_seq, base_seq }
        }
        LineageOutcome::Errored { error } => LineageVerdict::Unknown { error },
    }
}

/// An exhaustion, for the log.
pub fn describe_exhaustion(exhaustion: &Exhaustion) -> String {
    match exhaustion {
        Exhaustion::EmptyListing => "no generation is listed".to_owned(),
        Exhaustion::NoBase {
            generations,
            newest_sequence,
        } => format!(
            "{generations} generations are listed, all tails (newest {newest_sequence}), and no \
             full backup"
        ),
        Exhaustion::AllBroken { candidates } => {
            format!("every one of {candidates} candidate generations was unusable")
        }
        Exhaustion::BudgetInconclusive {
            candidates,
            unadjudicated_bases,
        } => format!(
            "{candidates} candidates tried, {unadjudicated_bases} full backups never read within \
             the budget"
        ),
    }
}

/// The newer generations a lineage stepped down from, newest first, each with
/// its reason's label and whether the reason is an absence rather than a
/// verdict on bytes.
pub fn abandoned_generations(step_downs: &[StepDownRecord]) -> Vec<AbandonedGeneration> {
    let mut abandoned: Vec<AbandonedGeneration> = step_downs
        .iter()
        .map(|record| AbandonedGeneration {
            seq: record.head_seq,
            reason_label: record.reason.reason_label(),
            absent: matches!(
                record.reason,
                LineageBreak::LinkAbsent { .. } | LineageBreak::MissingMetaCompanion { .. }
            ),
        })
        .collect();
    abandoned.sort_by(|a, b| b.seq.cmp(&a.seq));
    abandoned
}

/// The plan's objects after lineage selection.
#[derive(Clone, Debug, Default)]
pub struct Resolved {
    /// Every object's plan.
    pub objects: Vec<ObjectPlan>,
    /// What keeps the service held (the pre-plan's included).
    pub held: Vec<HeldItem>,
    /// What is reported (the pre-plan's included).
    pub advisories: Vec<Advisory>,
    /// The pre-plan's triggers, plus the deferred lineages.
    pub triggers: Vec<Trigger>,
    /// Objects whose newest backup is a defer-3 plain base over the limit:
    /// `(object, session, layer, base_seq)`, for the once-per-layer ERROR.
    pub over_limit: Vec<(u64, String, String, u64)>,
}

/// The post-lineage resolution (SR-9.5): each object's verdict, or its
/// fallback, and what it holds or reports. `verdicts` holds one entry for
/// every object whose pre-plan step needs its lineage; a missing entry is a
/// defect and resolves as unknown.
pub fn resolve(pre: &PrePlan, verdicts: &BTreeMap<u64, LineageVerdict>) -> Resolved {
    let mut resolved = Resolved {
        held: pre.held.clone(),
        advisories: pre.advisories.clone(),
        triggers: pre.triggers.clone(),
        ..Resolved::default()
    };
    for object in &pre.objects {
        let mut holds: Vec<HeldReason> = Vec::new();
        let mut held = |reason: HeldReason| holds.push(reason);
        let action = match &object.step {
            PreStep::Decided { action, held: h } => {
                if let Some(reason) = h {
                    held(reason.clone());
                }
                action.clone()
            }
            PreStep::NeedsLineage(need) => {
                let fallback_action = || match need.fallback {
                    Fallback::NoSurvivor => ObjectAction::Nothing,
                    Fallback::Keep => ObjectAction::Keep,
                    Fallback::ClearFlags(band) => ObjectAction::ClearFlags { band },
                };
                let survivor_seq = need.floor;
                match verdicts.get(&object.object_id) {
                    Some(LineageVerdict::Use(selected)) => {
                        let lineage = &selected.lineage;
                        if !lineage.step_downs.is_empty() {
                            resolved.advisories.push(Advisory::RestoredFromOlderBackup {
                                object_id: object.object_id,
                                session_id: object.session_id.clone(),
                                layer_name: object.layer_name.clone(),
                                restored_seq: lineage.head_seq,
                                abandoned: abandoned_generations(&lineage.step_downs),
                            });
                        }
                        if selected.sidecar_extra > 0 {
                            resolved.advisories.push(Advisory::SidecarHashesNotInState {
                                object_id: object.object_id,
                                count: selected.sidecar_extra,
                                base_only: lineage.is_base_only(),
                            });
                        }
                        ObjectAction::Restore {
                            reason: need.reason,
                            lineage: Box::new(lineage.clone()),
                            band: restore_band(lineage),
                        }
                    }
                    Some(LineageVerdict::Survivor { step_downs })
                    | Some(LineageVerdict::BackupsUnusable { step_downs, .. })
                        if survivor_seq.is_some() =>
                    {
                        if !step_downs.is_empty() {
                            resolved.advisories.push(Advisory::NewerBackupsUnusable {
                                object_id: object.object_id,
                                session_id: object.session_id.clone(),
                                layer_name: object.layer_name.clone(),
                                kept_seq: survivor_seq.unwrap_or_default(),
                                abandoned: abandoned_generations(step_downs),
                            });
                        } else if let Some(LineageVerdict::BackupsUnusable { exhaustion, .. }) =
                            verdicts.get(&object.object_id)
                        {
                            // defer-53: exhausted with no step-down (every
                            // newer listed generation a tail with no base).
                            // The shared walk's WARN called the layer
                            // unrecoverable; here it is kept, so say so.
                            resolved
                                .advisories
                                .push(Advisory::KeptWithBackupsExhausted {
                                    object_id: object.object_id,
                                    session_id: object.session_id.clone(),
                                    layer_name: object.layer_name.clone(),
                                    kept_seq: survivor_seq.unwrap_or_default(),
                                    exhaustion: exhaustion.clone(),
                                });
                        }
                        fallback_action()
                    }
                    Some(LineageVerdict::Survivor { .. }) => {
                        // Not reachable without a floor (no survivor to be
                        // newer than); held rather than guessed.
                        held(HeldReason::NoBackup);
                        fallback_action()
                    }
                    Some(LineageVerdict::BackupsUnusable {
                        exhaustion,
                        over_limit,
                        ..
                    }) => {
                        // Accept-loss stepped down past over-limit bases
                        // (PD-4) and found nothing older: those backups are
                        // readable by a release whose limit fits, so the
                        // reason names them (accept-loss keeps them).
                        if over_limit.is_empty() {
                            held(HeldReason::BackupsUnusable {
                                exhaustion: exhaustion.clone(),
                            });
                        } else {
                            held(HeldReason::BackupsOverLimit {
                                exhaustion: exhaustion.clone(),
                                bases: over_limit.clone(),
                            });
                        }
                        fallback_action()
                    }
                    Some(LineageVerdict::OverLimitPlainBase { head_seq, base_seq }) => {
                        resolved.over_limit.push((
                            object.object_id,
                            object.session_id.clone(),
                            object.layer_name.clone(),
                            *base_seq,
                        ));
                        if survivor_seq.is_some() {
                            held(HeldReason::NewerBackupUnusable {
                                head_seq: *head_seq,
                            });
                        } else {
                            held(HeldReason::OverLimitPlainBase {
                                head_seq: *head_seq,
                                base_seq: *base_seq,
                            });
                        }
                        fallback_action()
                    }
                    Some(LineageVerdict::Unknown { error }) => {
                        held(HeldReason::Unknown {
                            error: error.clone(),
                        });
                        fallback_action()
                    }
                    Some(LineageVerdict::Defer { seq }) => {
                        resolved.triggers.push(Trigger::LineageDeferred {
                            object_id: object.object_id,
                            session_id: object.session_id.clone(),
                            layer_name: object.layer_name.clone(),
                            seq: *seq,
                        });
                        fallback_action()
                    }
                    None => {
                        held(HeldReason::Unknown {
                            error: "the object's lineage was not selected".to_owned(),
                        });
                        fallback_action()
                    }
                }
            }
        };
        for reason in holds {
            resolved.held.push(HeldItem {
                session_id: object.session_id.clone(),
                layer_name: object.layer_name.clone(),
                object_id: Some(object.object_id),
                reason,
            });
        }
        resolved.objects.push(ObjectPlan {
            object_id: object.object_id,
            session_id: object.session_id.clone(),
            layer_name: object.layer_name.clone(),
            action,
        });
    }
    resolved
}

/// A restored lineage's band inputs: its head, and its restore meta's span
/// and write time.
pub fn restore_band(lineage: &PlannedLineage) -> BandInputs {
    BandInputs {
        floor_seq: lineage.head_seq,
        observed_commit_span: lineage.meta.observed_commit_span,
        written_at_ms: lineage.meta.written_at_unix_ms,
        cursor: None,
    }
}

/// The objects whose surviving `obj:ext` the coverage check reads: every
/// kept and flag-cleared object (a remap target is one of these, planned in
/// its own right).
pub fn survivors_to_cover(objects: &[ObjectPlan]) -> Vec<u64> {
    objects
        .iter()
        .filter(|o| {
            matches!(
                o.action,
                ObjectAction::Keep | ObjectAction::ClearFlags { .. }
            )
        })
        .map(|o| o.object_id)
        .collect()
}

/// The coverage universe (SR-9.3, PD-2): each distinct referenced value and
/// the objects referencing it. A restored object contributes its restored
/// state's rows (`restored`); a kept or flag-cleared object its surviving
/// `obj:ext` (`survivors`); a held object nothing. A restored base's ext-refs
/// sidecar adds nothing (PD-2: it is reported, never held on).
pub fn coverage_universe(
    objects: &[ObjectPlan],
    restored: &BTreeMap<u64, BTreeSet<ExtHash>>,
    survivors: &BTreeMap<u64, BTreeSet<ExtHash>>,
) -> BTreeMap<ExtHash, BTreeSet<u64>> {
    let mut universe: BTreeMap<ExtHash, BTreeSet<u64>> = BTreeMap::new();
    for object in objects {
        let hashes = match object.action {
            ObjectAction::Restore { .. } => restored.get(&object.object_id),
            ObjectAction::Keep | ObjectAction::ClearFlags { .. } => {
                survivors.get(&object.object_id)
            }
            ObjectAction::Remap { .. } | ObjectAction::Nothing => None,
        };
        for hash in hashes.into_iter().flatten() {
            universe.entry(*hash).or_default().insert(object.object_id);
        }
    }
    universe
}

/// A value's verdict from what needs no backup-store call, in the check's
/// order: Valkey, then the restored tails, then this run's own PUT. `None`:
/// it must be probed (or, with no backup store, it is uncovered).
pub fn settle_without_probe(in_valkey: bool, in_tails: bool, own_put: bool) -> Option<HashVerdict> {
    if in_valkey {
        Some(HashVerdict::Valkey)
    } else if in_tails {
        Some(HashVerdict::Tails)
    } else if own_put {
        Some(HashVerdict::OwnPut)
    } else {
        None
    }
}

/// A plan the writer must not execute: a defect in the planner, not in the
/// store ([`check_executable`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanDefect {
    /// Rule 1: two of the plan's object entries write the same object.
    TwoWritingActions {
        /// The object.
        object_id: u64,
    },
    /// Rule 2 (defer-51's writer half): a held mapping to a deleted object
    /// names a `(session, layer)` that a session restore's Step B writes, so
    /// the hold would outlive the mapping it names.
    HeldMappingWritten {
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
    },
    /// Rule 3: an object an abandoned delete removes is also planned as a
    /// layer object.
    DeletedObjectPlanned {
        /// The deleted session.
        session_id: String,
        /// The object.
        object_id: u64,
    },
    /// Rule 3: an object an abandoned delete removes is mapped by a session
    /// restore's Step B.
    DeletedObjectMapped {
        /// The deleted session.
        session_id: String,
        /// The object.
        object_id: u64,
    },
    /// Rule 4: a move to a successor the plan does not plan.
    RemapTargetUnplanned {
        /// The fenced object.
        from: u64,
        /// The successor.
        to: u64,
    },
}

impl std::fmt::Display for PlanDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TwoWritingActions { object_id } => {
                write!(f, "object {object_id} has two writing actions")
            }
            Self::HeldMappingWritten {
                session_id,
                layer_name,
            } => write!(
                f,
                "layer {layer_name} of session {session_id} is held as mapped to a deleted \
                 object, but the session's restore writes its mapping"
            ),
            Self::DeletedObjectPlanned {
                session_id,
                object_id,
            } => write!(
                f,
                "object {object_id} is removed with deleted session {session_id} and also \
                 planned as a layer"
            ),
            Self::DeletedObjectMapped {
                session_id,
                object_id,
            } => write!(
                f,
                "object {object_id} is removed with deleted session {session_id} and also \
                 mapped by a session's restore"
            ),
            Self::RemapTargetUnplanned { from, to } => write!(
                f,
                "object {from} is moved to object {to}, which the plan does not plan"
            ),
        }
    }
}

/// The writer's consistency check of a plan before any write (SDD §22.22, *The
/// restore writer*), pure. Its rules:
///
/// 1. no object has two writing actions (restore, flag clear, move);
/// 2. no held `MappedToDeletedObject` names a `(session, layer)` that a
///    session restore's Step B writes (defer-51's writer half);
/// 3. no object an abandoned delete removes is planned as a layer, or mapped
///    by a session restore's Step B;
/// 4. every move's successor is planned.
///
/// Accept-loss runs it too, on the plan it executes before its discards. Empty when the
/// plan is executable.
pub fn check_executable(plan: &RestorePlan) -> Vec<PlanDefect> {
    let mut defects = Vec::new();

    // Rule 1.
    let mut writers: BTreeMap<u64, usize> = BTreeMap::new();
    for object in &plan.objects {
        if matches!(
            object.action,
            ObjectAction::Restore { .. }
                | ObjectAction::ClearFlags { .. }
                | ObjectAction::Remap { .. }
        ) {
            *writers.entry(object.object_id).or_default() += 1;
        }
    }
    defects.extend(
        writers
            .into_iter()
            .filter(|(_, n)| *n > 1)
            .map(|(object_id, _)| PlanDefect::TwoWritingActions { object_id }),
    );

    // The mappings each session restore's Step B writes.
    let mut step_b_writes: BTreeMap<(&str, &str), u64> = BTreeMap::new();
    for session in &plan.sessions {
        if let SessionAction::Restore {
            checkpoint,
            step_b: true,
        } = &session.action
        {
            for (layer_name, object_id) in &checkpoint.layer_mappings {
                step_b_writes.insert(
                    (session.session_id.as_str(), layer_name.as_str()),
                    *object_id,
                );
            }
        }
    }

    // Rule 2.
    for item in &plan.held {
        if let HeldReason::MappedToDeletedObject {
            session_id,
            layer_name,
        } = &item.reason
        {
            if step_b_writes.contains_key(&(session_id.as_str(), layer_name.as_str())) {
                defects.push(PlanDefect::HeldMappingWritten {
                    session_id: session_id.clone(),
                    layer_name: layer_name.clone(),
                });
            }
        }
    }

    // Rule 3.
    let planned: BTreeSet<u64> = plan.objects.iter().map(|o| o.object_id).collect();
    let mapped: BTreeSet<u64> = step_b_writes.values().copied().collect();
    for session in &plan.sessions {
        if let SessionAction::CompleteAbandonedDelete {
            objects, mappings, ..
        } = &session.action
        {
            let removed: BTreeSet<u64> = objects
                .iter()
                .copied()
                .chain(mappings.iter().map(|(_, object_id)| *object_id))
                .collect();
            for object_id in removed {
                if planned.contains(&object_id) {
                    defects.push(PlanDefect::DeletedObjectPlanned {
                        session_id: session.session_id.clone(),
                        object_id,
                    });
                }
                if mapped.contains(&object_id) {
                    defects.push(PlanDefect::DeletedObjectMapped {
                        session_id: session.session_id.clone(),
                        object_id,
                    });
                }
            }
        }
    }

    // Rule 4.
    for object in &plan.objects {
        if let ObjectAction::Remap { from, to } = object.action {
            if !planned.contains(&to) {
                defects.push(PlanDefect::RemapTargetUnplanned { from, to });
            }
        }
    }
    defects
}

/// One probe's answer as a verdict: present is covered; absent is uncovered
/// only when observed at or after `T_safe`, and pending a re-probe before it;
/// an error is unknown.
pub fn probe_verdict(answer: &Result<bool, String>, at_or_after_t_safe: bool) -> HashVerdict {
    match answer {
        Ok(true) => HashVerdict::Backup,
        Ok(false) if at_or_after_t_safe => HashVerdict::Uncovered,
        Ok(false) => HashVerdict::PendingReprobe,
        Err(error) => HashVerdict::Unknown(error.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{
        CoveragePlan, CoverageReport, LineageFootprint, PlanCounts, PlanEpoch, PlanOutcome,
        SelectedLineage,
    };
    use super::*;
    use crate::checkpoint::store::{CheckpointKind, CheckpointMeta};
    use crate::storage::dr_job::CheckpointStamp;

    const SESSION: &str = "s1";
    const LAYER: &str = "layer.usda";
    const DELETED: u64 = 5;

    /// A session whose record survives, whose surviving mapping of `LAYER`
    /// names the deleted object `DELETED`, and whose listed session backup
    /// maps `LAYER` to `backed_up`; planned at `T_safe`.
    fn plan_with_backup_mapping(backed_up: u64) -> PrePlan {
        let checkpoint = SessionCheckpoint {
            schema_version: 1,
            session_id: SESSION.to_owned(),
            name: "session".to_owned(),
            base_path: "stage:/a.usd".to_owned(),
            created_by: "user".to_owned(),
            created_at: 1,
            layer_mappings: BTreeMap::from([(LAYER.to_owned(), backed_up)]),
            layers_index: BTreeSet::from([LAYER.to_owned()]),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: SESSION.to_owned(),
        };
        let listing = Listing {
            session_ids: BTreeSet::from([SESSION.to_owned()]),
            sessions: BTreeMap::from([(SESSION.to_owned(), SessionRead::Present(checkpoint))]),
            at_or_after_t_safe: true,
            ..Listing::default()
        };
        let store = StoreFacts {
            mappings: vec![LayerMappingFact {
                session_id: SESSION.to_owned(),
                layer_name: LAYER.to_owned(),
                object_id: DELETED,
            }],
            dead: HashSet::from([DELETED]),
            sessions: BTreeMap::from([(
                SESSION.to_owned(),
                SessionFacts {
                    meta_exists: true,
                    ..SessionFacts::default()
                },
            )]),
            ..StoreFacts::default()
        };
        plan_pre_lineage(&PlanInputs {
            access: StoreAccess::Persistent,
            listing: &listing,
            store: &store,
            identities: &BTreeMap::new(),
            unidentified: &BTreeMap::new(),
        })
    }

    fn restore_mappings(pre: &PrePlan) -> (BTreeMap<String, u64>, bool) {
        match &pre.sessions[..] {
            [SessionPlan {
                action: SessionAction::Restore { checkpoint, step_b },
                ..
            }] => (checkpoint.layer_mappings.clone(), *step_b),
            other => panic!("expected one session restore, got {other:?}"),
        }
    }

    fn holds_mapping_to_deleted(pre: &PrePlan) -> bool {
        pre.held.iter().any(|item| {
            item.object_id == Some(DELETED)
                && matches!(item.reason, HeldReason::MappedToDeletedObject { .. })
        })
    }

    /// §E rules 2 and 4: a surviving mapping to a deleted object still keeps
    /// an older backup mapping of its layer out, so restore never brings back
    /// an older generation of a layer whose newer object was deleted.
    #[test]
    fn a_surviving_mapping_to_a_deleted_object_keeps_an_older_backup_mapping_out() {
        let pre = plan_with_backup_mapping(DELETED - 2);
        let (writes, step_b) = restore_mappings(&pre);
        assert!(writes.is_empty(), "no mapping is written: {writes:?}");
        assert!(!step_b, "the record survives and no mapping needs writing");
        assert!(
            pre.objects.iter().all(|o| o.object_id != DELETED - 2),
            "the older object is not planned"
        );
        assert!(holds_mapping_to_deleted(&pre));
    }

    /// §E rule 2's other side, and defer-51: a backup mapping newer than the
    /// deleted survivor is written by Step B, so the survivor's mapping is
    /// replaced and not held; the replacement is reported instead.
    #[test]
    fn a_backup_mapping_newer_than_a_deleted_survivor_is_written() {
        let pre = plan_with_backup_mapping(DELETED + 2);
        let (writes, step_b) = restore_mappings(&pre);
        assert_eq!(writes, BTreeMap::from([(LAYER.to_owned(), DELETED + 2)]));
        assert!(step_b);
        assert!(pre.objects.iter().any(|o| o.object_id == DELETED + 2));
        assert!(!holds_mapping_to_deleted(&pre), "{:?}", pre.held);
        assert!(pre.advisories.contains(&Advisory::DeletedMappingReplaced {
            session_id: SESSION.to_owned(),
            layer_name: LAYER.to_owned(),
            deleted_object_id: DELETED,
            replacement_object_id: DELETED + 2,
        }));
    }

    /// INV-1: the `T_safe` re-take reads an unidentified object again when its
    /// earlier read saw an absence (possibly stale), or when its generations
    /// changed; an all-`Permanent` read over the same generations is not
    /// repeated.
    #[test]
    fn an_unidentified_object_is_read_again_after_an_absence_or_a_listing_change() {
        let read = |saw_absent: bool| UnidentifiedObject {
            listed: vec![14, 12],
            seq: 12,
            error: "permanent checkpoint-store error: forbidden".to_owned(),
            saw_absent,
        };
        assert!(!read(false).read_again(&[14, 12]));
        assert!(read(true).read_again(&[14, 12]));
        assert!(read(false).read_again(&[15, 14, 12]));
    }

    /// INV-1, trigger (g): under an empty session listing, an identity read
    /// that saw an absence and learned no session is a `T_safe` trigger before
    /// `T_safe`, since the absence may hide the S7 evidence; at or after
    /// `T_safe` the absence is authoritative and the object is an
    /// unreferenced backup.
    #[test]
    fn an_identity_absence_before_t_safe_is_a_trigger() {
        const OBJECT: u64 = 9;
        let plan_at = |at: bool| {
            let listing = Listing {
                generations: BTreeMap::from([(
                    OBJECT,
                    vec![CheckpointHandle::base(OBJECT, 12, String::new())],
                )]),
                listed_object_ids: BTreeSet::from([OBJECT]),
                at_or_after_t_safe: at,
                identity_absences: BTreeMap::from([(OBJECT, 12)]),
                ..Listing::default()
            };
            plan_pre_lineage(&PlanInputs {
                access: StoreAccess::Persistent,
                listing: &listing,
                store: &StoreFacts::default(),
                identities: &BTreeMap::new(),
                unidentified: &BTreeMap::new(),
            })
        };

        let before = plan_at(false);
        assert_eq!(
            before.triggers,
            vec![Trigger::IdentityAbsent {
                object_id: OBJECT,
                seq: 12
            }]
        );
        assert!(before.triggers[0]
            .why()
            .contains("object 9 (generation 12)"));

        let at = plan_at(true);
        assert!(at.triggers.is_empty(), "{:?}", at.triggers);
        assert!(at
            .advisories
            .contains(&Advisory::UnreferencedBackup { object_id: OBJECT }));
    }

    /// PD-1 with PD-4: under an empty session listing, a listed object whose
    /// identity read failed `Permanent` on every generation holds the service
    /// as unknown, naming the object, and is not reported as an unreferenced
    /// backup: its session may be the only S7 evidence.
    #[test]
    fn an_unidentified_listed_object_holds_as_unknown_under_an_empty_session_listing() {
        const OBJECT: u64 = 9;
        let listing = Listing {
            generations: BTreeMap::from([(
                OBJECT,
                vec![CheckpointHandle::base(OBJECT, 12, String::new())],
            )]),
            listed_object_ids: BTreeSet::from([OBJECT]),
            at_or_after_t_safe: true,
            ..Listing::default()
        };
        let unidentified = BTreeMap::from([(
            OBJECT,
            UnidentifiedObject {
                listed: vec![12],
                seq: 12,
                error: "permanent checkpoint-store error: forbidden".to_owned(),
                saw_absent: false,
            },
        )]);
        let pre = plan_pre_lineage(&PlanInputs {
            access: StoreAccess::Persistent,
            listing: &listing,
            store: &StoreFacts::default(),
            identities: &BTreeMap::new(),
            unidentified: &unidentified,
        });

        let held: Vec<&HeldItem> = pre
            .held
            .iter()
            .filter(|item| item.object_id == Some(OBJECT))
            .collect();
        assert_eq!(
            held.len(),
            1,
            "one held item for the object: {:?}",
            pre.held
        );
        match &held[0].reason {
            HeldReason::Unknown { error } => {
                assert!(error.contains("generation 12"), "{error}");
                assert!(error.contains("forbidden"), "{error}");
            }
            other => panic!("expected an unknown hold, got {other:?}"),
        }
        assert!(
            !pre.advisories.iter().any(
                |a| matches!(a, Advisory::UnreferencedBackup { object_id } if *object_id == OBJECT)
            ),
            "not reported as an unreferenced backup: {:?}",
            pre.advisories
        );

        let resolved = resolve(&pre, &BTreeMap::new());
        let plan = RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(0),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: pre.sessions.clone(),
            objects: resolved.objects,
            coverage: CoveragePlan::default(),
            held: resolved.held,
            advisories: resolved.advisories,
            listed_max_object_id: Some(OBJECT),
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        };
        assert_eq!(
            plan.outcome(&CoverageReport::default()),
            PlanOutcome::StaysHeld
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // The planning rules' tables: SR-9.5 (objects), SR-9.4 (sessions), the
    // Job's absence policy, listing trust, the coverage merge and the counts
    // ═════════════════════════════════════════════════════════════════════

    /// The session most cases plan in.
    const S: &str = "sess";
    /// The layer most cases plan.
    const L: &str = "layer.usda";
    /// The object most cases plan.
    const OID: u64 = 70;

    fn header(object_id: u64, fields: &[(&str, &str)]) -> HeaderFacts {
        HeaderFacts {
            object_id,
            fields: fields
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }

    /// A structurally complete header of `layer` in `session` at `seq`, not
    /// retiring.
    fn complete(object_id: u64, session: &str, layer: &str, seq: u64) -> HeaderFacts {
        let seq = seq.to_string();
        header(
            object_id,
            &[
                ("seq", &seq),
                ("created_by", "user"),
                ("session_id", session),
                ("layer_name", layer),
            ],
        )
    }

    /// `h` with `fields` added.
    fn plus(mut h: HeaderFacts, fields: &[(&str, &str)]) -> HeaderFacts {
        for (k, v) in fields {
            h.fields.insert((*k).to_owned(), (*v).to_owned());
        }
        h
    }

    fn mapping(session: &str, layer: &str, object_id: u64) -> LayerMappingFact {
        LayerMappingFact {
            session_id: session.to_owned(),
            layer_name: layer.to_owned(),
            object_id,
        }
    }

    /// A session whose record survives.
    fn live() -> SessionFacts {
        SessionFacts {
            meta_exists: true,
            ..SessionFacts::default()
        }
    }

    /// A session with a surviving deletion marker.
    fn deleted() -> SessionFacts {
        SessionFacts {
            deleted_marker_exists: true,
            ..SessionFacts::default()
        }
    }

    fn base(object_id: u64, seq: u64) -> CheckpointHandle {
        CheckpointHandle::base(object_id, seq, String::new())
    }

    fn facts(
        headers: Vec<HeaderFacts>,
        mappings: Vec<LayerMappingFact>,
        sessions: &[(&str, SessionFacts)],
    ) -> StoreFacts {
        StoreFacts {
            headers: headers.into_iter().map(|h| (h.object_id, h)).collect(),
            mappings,
            dead: HashSet::new(),
            sessions: sessions
                .iter()
                .map(|(s, f)| ((*s).to_owned(), f.clone()))
                .collect(),
        }
    }

    /// A listing taken at or after `T_safe` (`at`) or before it, of the base
    /// generations `generations` names per object, with no session backup.
    fn listed(at: bool, generations: &[(u64, &[u64])]) -> Listing {
        Listing {
            generations: generations
                .iter()
                .map(|(oid, seqs)| (*oid, seqs.iter().map(|s| base(*oid, *s)).collect()))
                .collect(),
            listed_object_ids: generations.iter().map(|(oid, _)| *oid).collect(),
            at_or_after_t_safe: at,
            ..Listing::default()
        }
    }

    /// `listing` with `session`'s backup listed and read as `read`.
    fn with_session(mut listing: Listing, session: &str, read: SessionRead) -> Listing {
        listing.session_ids.insert(session.to_owned());
        listing.sessions.insert(session.to_owned(), read);
        listing
    }

    /// A session backup of `session` mapping each `(layer, object)`.
    fn checkpoint(session: &str, mappings: &[(&str, u64)]) -> SessionCheckpoint {
        SessionCheckpoint {
            schema_version: 1,
            session_id: session.to_owned(),
            name: "session".to_owned(),
            base_path: format!("stage:/{session}.usd"),
            created_by: "user".to_owned(),
            created_at: 1,
            layer_mappings: mappings
                .iter()
                .map(|(l, o)| ((*l).to_owned(), *o))
                .collect(),
            layers_index: mappings.iter().map(|(l, _)| (*l).to_owned()).collect(),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session.to_owned(),
        }
    }

    fn pre_plan(access: StoreAccess, listing: &Listing, store: &StoreFacts) -> PrePlan {
        plan_pre_lineage(&PlanInputs {
            access,
            listing,
            store,
            identities: &BTreeMap::new(),
            unidentified: &BTreeMap::new(),
        })
    }

    fn persistent(listing: &Listing, store: &StoreFacts) -> PrePlan {
        pre_plan(StoreAccess::Persistent, listing, store)
    }

    /// Plan `OID`, mapped as `L` in the live session `S`, with `header` and
    /// the base generations `heads` listed.
    fn plan_object_with(at: bool, header: Option<HeaderFacts>, heads: &[u64]) -> PrePlan {
        let listing = if heads.is_empty() {
            listed(at, &[])
        } else {
            listed(at, &[(OID, heads)])
        };
        let store = facts(
            header.into_iter().collect(),
            vec![mapping(S, L, OID)],
            &[(S, live())],
        );
        persistent(&listing, &store)
    }

    fn step_of(pre: &PrePlan, object_id: u64) -> &PreStep {
        let found: Vec<&PreObject> = pre
            .objects
            .iter()
            .filter(|o| o.object_id == object_id)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "object {object_id} is planned once: {:?}",
            pre.objects
        );
        &found[0].step
    }

    fn need_of(pre: &PrePlan, object_id: u64) -> &LineageNeed {
        match step_of(pre, object_id) {
            PreStep::NeedsLineage(need) => need,
            other => panic!("object {object_id} must need its lineage, got {other:?}"),
        }
    }

    fn decided_of(pre: &PrePlan, object_id: u64) -> (&ObjectAction, &Option<HeldReason>) {
        match step_of(pre, object_id) {
            PreStep::Decided { action, held } => (action, held),
            other => panic!("object {object_id} must be decided without a lineage, got {other:?}"),
        }
    }

    fn planned_ids(pre: &PrePlan) -> BTreeSet<u64> {
        pre.objects.iter().map(|o| o.object_id).collect()
    }

    fn session_action<'p>(pre: &'p PrePlan, session: &str) -> &'p SessionAction {
        let found: Vec<&SessionPlan> = pre
            .sessions
            .iter()
            .filter(|p| p.session_id == session)
            .collect();
        assert_eq!(found.len(), 1, "session {session} is planned once");
        &found[0].action
    }

    // ── SR-9.5, before lineage selection ─────────────────────────────────

    /// No header, a listed head: the object needs its lineage, with no
    /// floor, and nothing to fall back to.
    #[test]
    fn plan_object_with_no_header_and_a_listed_backup_needs_its_lineage_with_no_floor() {
        let pre = plan_object_with(true, None, &[20, 10]);

        let need = need_of(&pre, OID);
        assert_eq!(need.floor, None);
        assert_eq!(need.reason, RestoreReason::HeaderAbsent);
        assert_eq!(need.fallback, Fallback::NoSurvivor);
        assert_eq!(need.handles, vec![base(OID, 20), base(OID, 10)]);
    }

    /// A header without `created_by` is partial: its `seq` is no floor, and
    /// it is restored like an absent one.
    #[test]
    fn plan_object_with_a_partial_header_needs_its_lineage_with_no_floor() {
        let partial = header(OID, &[("seq", "30"), ("session_id", S), ("layer_name", L)]);

        let pre = plan_object_with(true, Some(partial), &[20]);

        let need = need_of(&pre, OID);
        assert_eq!(need.floor, None, "a partial header's seq is no floor");
        assert_eq!(need.reason, RestoreReason::HeaderPartial);
        assert_eq!(need.fallback, Fallback::NoSurvivor);
    }

    /// No header and no listed backup: held `NoBackup` with nothing
    /// written; before `T_safe` that absence is trigger (b), at it the hold
    /// stands alone.
    #[test]
    fn plan_object_with_no_header_and_no_listed_backup_is_held_and_triggers_only_before_t_safe() {
        for at in [false, true] {
            let pre = plan_object_with(at, None, &[]);

            let (action, held) = decided_of(&pre, OID);
            assert!(matches!(action, ObjectAction::Nothing), "at={at}");
            assert_eq!(held, &Some(HeldReason::NoBackup), "at={at}");
            let triggered = pre.triggers.contains(&Trigger::NoListedBackup {
                object_id: OID,
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
            });
            assert_eq!(
                triggered, !at,
                "at={at}: trigger (b) only before T_safe: {:?}",
                pre.triggers
            );
        }
    }

    /// With no backup store a lost object holds `NoBackupStore`, and since no
    /// absence was read from a backup store, nothing triggers.
    #[test]
    fn plan_object_with_no_header_and_no_backup_store_holds_no_backup_store_without_a_trigger() {
        for lost in [None, Some(header(OID, &[("seq", "3"), ("session_id", S)]))] {
            let store = facts(
                lost.into_iter().collect(),
                vec![mapping(S, L, OID)],
                &[(S, live())],
            );
            let listing = Listing {
                at_or_after_t_safe: true,
                ..Listing::default()
            };

            let pre = pre_plan(
                StoreAccess::NoStore(NoBackupStore::Disabled),
                &listing,
                &store,
            );

            let (action, held) = decided_of(&pre, OID);
            assert!(matches!(action, ObjectAction::Nothing));
            assert_eq!(held, &Some(HeldReason::NoBackupStore));
            assert!(pre.triggers.is_empty(), "{:?}", pre.triggers);
        }
    }

    /// A retiring object whose `superseded_by` names a complete, live
    /// successor of the same session and layer is remapped to it, and the
    /// successor is planned by the same table in turn.
    #[test]
    fn plan_object_retiring_with_a_matching_successor_is_remapped_and_the_successor_planned() {
        const TO: u64 = 71;
        let from = plus(
            complete(OID, S, L, 10),
            &[("superseded", "1"), ("superseded_by", "71")],
        );
        let store = facts(
            vec![from, complete(TO, S, L, 12)],
            vec![mapping(S, L, OID)],
            &[(S, live())],
        );

        let pre = persistent(&listed(true, &[(TO, &[15])]), &store);

        let (action, held) = decided_of(&pre, OID);
        assert!(
            matches!(action, ObjectAction::Remap { from: OID, to: TO }),
            "{action:?}"
        );
        assert_eq!(held, &None);
        let need = need_of(&pre, TO);
        assert_eq!(
            need.floor,
            Some(12),
            "the successor is planned as a survivor"
        );
        assert_eq!(need.fallback, Fallback::Keep);
    }

    /// A retiring object whose successor is of another layer or session,
    /// retiring itself or partial, is not remapped: its stale flags are
    /// cleared, and the successor is not planned through it.
    #[test]
    fn plan_object_retiring_with_an_unmatched_successor_clears_its_flags_instead() {
        const TO: u64 = 71;
        let cases: [(&str, HeaderFacts); 4] = [
            ("another layer", complete(TO, S, "other.usda", 12)),
            ("another session", complete(TO, "other", L, 12)),
            (
                "a retiring successor",
                plus(complete(TO, S, L, 12), &[("ext_gc_scheduled", "1")]),
            ),
            (
                "a partial successor",
                header(TO, &[("seq", "12"), ("session_id", S), ("layer_name", L)]),
            ),
        ];
        for (case, successor) in cases {
            let from = plus(
                complete(OID, S, L, 10),
                &[("superseded", "1"), ("superseded_by", "71")],
            );
            let store = facts(
                vec![from, successor],
                vec![mapping(S, L, OID)],
                &[(S, live())],
            );

            let pre = persistent(&listed(true, &[]), &store);

            let (action, held) = decided_of(&pre, OID);
            assert!(
                matches!(action, ObjectAction::ClearFlags { band } if band.floor_seq == 10),
                "{case}: {action:?}"
            );
            assert_eq!(held, &None, "{case}");
            assert!(
                !planned_ids(&pre).contains(&TO),
                "{case}: the unmatched successor is not planned through the mapping"
            );
        }
    }

    /// A retiring survivor (`superseded` or `ext_gc_scheduled`) needs its
    /// lineage, floored at its `seq`, only when a newer backup is listed;
    /// otherwise its flags are cleared. The band carries its cursor columns.
    #[test]
    fn plan_object_retiring_needs_its_lineage_only_when_a_newer_backup_is_listed() {
        for marker in ["superseded", "ext_gc_scheduled"] {
            let retiring = plus(
                complete(OID, S, L, 10),
                &[
                    (marker, "1"),
                    ("last_checkpoint_seq", "9"),
                    ("last_checkpoint_ms", "900"),
                ],
            );
            let band = BandInputs {
                floor_seq: 10,
                observed_commit_span: 0,
                written_at_ms: 0,
                cursor: Some((9, 900)),
            };

            let pre = plan_object_with(true, Some(retiring.clone()), &[11]);
            let need = need_of(&pre, OID);
            assert_eq!(need.floor, Some(10), "{marker}");
            assert_eq!(
                need.reason,
                RestoreReason::NewerThanRetiringSurvivor { survivor_seq: 10 },
                "{marker}"
            );
            assert_eq!(need.fallback, Fallback::ClearFlags(band), "{marker}");

            for heads in [&[10u64][..], &[9], &[]] {
                let pre = plan_object_with(true, Some(retiring.clone()), heads);
                let (action, held) = decided_of(&pre, OID);
                assert!(
                    matches!(action, ObjectAction::ClearFlags { band: b } if *b == band),
                    "{marker}, heads {heads:?}: {action:?}"
                );
                assert_eq!(held, &None, "{marker}, heads {heads:?}");
            }
        }
    }

    /// A complete survivor that is not retiring needs its lineage, floored at
    /// its `seq`, only when a newer backup is listed; otherwise it is kept,
    /// and having no backup at all is no absence.
    #[test]
    fn plan_object_complete_survivor_needs_its_lineage_only_when_a_newer_backup_is_listed() {
        let pre = plan_object_with(true, Some(complete(OID, S, L, 10)), &[11, 5]);
        let need = need_of(&pre, OID);
        assert_eq!(need.floor, Some(10));
        assert_eq!(
            need.reason,
            RestoreReason::NewerThanSurvivor { survivor_seq: 10 }
        );
        assert_eq!(need.fallback, Fallback::Keep);

        for at in [false, true] {
            for heads in [&[10u64][..], &[9, 5], &[]] {
                let pre = plan_object_with(at, Some(complete(OID, S, L, 10)), heads);
                let (action, held) = decided_of(&pre, OID);
                assert!(
                    matches!(action, ObjectAction::Keep),
                    "heads {heads:?}: {action:?}"
                );
                assert_eq!(held, &None, "heads {heads:?}");
                assert!(
                    !pre.triggers
                        .iter()
                        .any(|t| matches!(t, Trigger::NoListedBackup { .. })),
                    "a survivor with no newer backup is no absence: {:?}",
                    pre.triggers
                );
            }
        }
    }

    // ── SR-9.5, after lineage selection ──────────────────────────────────

    fn down(head_seq: u64, reason: LineageBreak) -> StepDownRecord {
        StepDownRecord {
            head_seq,
            head_kind: CheckpointKind::Base,
            reason,
        }
    }

    fn link_absent(seq: u64) -> LineageBreak {
        LineageBreak::LinkAbsent {
            seq,
            kind: CheckpointKind::Base,
            at_fetch: true,
        }
    }

    fn missing_meta(seq: u64) -> LineageBreak {
        LineageBreak::MissingMetaCompanion {
            seq,
            message: "the meta companion is absent".to_owned(),
        }
    }

    fn decode(seq: u64) -> LineageBreak {
        LineageBreak::Decode {
            seq,
            message: "bad magic".to_owned(),
        }
    }

    fn mismatch(seq: u64) -> LineageBreak {
        LineageBreak::LinkMismatch {
            seq,
            field: "kind",
            expected: "base".to_owned(),
            observed: "tail".to_owned(),
        }
    }

    fn compose(seq: u64) -> LineageBreak {
        LineageBreak::Compose { head_seq: seq }
    }

    fn budget(seq: u64) -> LineageBreak {
        LineageBreak::Budget {
            seq,
            remaining_ms: 5,
            reserve_ms: 10,
        }
    }

    /// A lineage the walk selected: a base at `head_seq`, with a restore
    /// meta whose span and write time the band must carry.
    fn selected(
        head_seq: u64,
        step_downs: Vec<StepDownRecord>,
        sidecar_extra: usize,
    ) -> Box<SelectedLineage> {
        Box::new(SelectedLineage {
            lineage: PlannedLineage {
                base: base(OID, head_seq),
                tails: Vec::new(),
                head_seq,
                meta: CheckpointMeta {
                    observed_commit_span: 7,
                    written_at_unix_ms: 1_234,
                    ..CheckpointMeta::default()
                },
                tail_carried: Box::new([]),
                step_downs,
                footprint: LineageFootprint::default(),
            },
            state_hashes: BTreeSet::new(),
            sidecar_extra,
        })
    }

    /// `OID`'s resolution when its lineage need is floored at `floor` with
    /// `fallback`, and its verdict is `verdict` (none: missing).
    fn resolve_need(
        floor: Option<u64>,
        fallback: Fallback,
        verdict: Option<LineageVerdict>,
    ) -> Resolved {
        let reason = match floor {
            Some(survivor_seq) => RestoreReason::NewerThanSurvivor { survivor_seq },
            None => RestoreReason::HeaderAbsent,
        };
        let pre = PrePlan {
            objects: vec![PreObject {
                object_id: OID,
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
                step: PreStep::NeedsLineage(LineageNeed {
                    handles: vec![base(OID, 20)],
                    floor,
                    reason,
                    fallback,
                }),
            }],
            ..PrePlan::default()
        };
        let verdicts: BTreeMap<u64, LineageVerdict> =
            verdict.into_iter().map(|v| (OID, v)).collect();
        resolve(&pre, &verdicts)
    }

    fn only_action(resolved: &Resolved) -> &ObjectAction {
        assert_eq!(resolved.objects.len(), 1, "{:?}", resolved.objects);
        &resolved.objects[0].action
    }

    fn held_reasons(resolved: &Resolved) -> Vec<HeldReason> {
        resolved.held.iter().map(|h| h.reason.clone()).collect()
    }

    /// The table's two sides of the floor: a lineage selected above the
    /// survivor's `seq` restores (with the restore meta's band inputs), one
    /// not newer than it keeps the survivor.
    #[test]
    fn resolve_restores_a_lineage_selected_above_the_floor_and_keeps_a_survivor_not_older() {
        let restored = resolve_need(
            Some(10),
            Fallback::Keep,
            Some(LineageVerdict::Use(selected(20, vec![], 0))),
        );
        match only_action(&restored) {
            ObjectAction::Restore {
                reason,
                lineage,
                band,
            } => {
                assert_eq!(
                    *reason,
                    RestoreReason::NewerThanSurvivor { survivor_seq: 10 }
                );
                assert_eq!(lineage.head_seq, 20);
                assert_eq!(
                    *band,
                    BandInputs {
                        floor_seq: 20,
                        observed_commit_span: 7,
                        written_at_ms: 1_234,
                        cursor: None,
                    }
                );
            }
            other => panic!("expected a restore, got {other:?}"),
        }
        assert!(restored.held.is_empty());
        assert!(restored.advisories.is_empty(), "no step-down, no advisory");

        let kept = resolve_need(
            Some(10),
            Fallback::Keep,
            Some(LineageVerdict::Survivor { step_downs: vec![] }),
        );
        assert!(matches!(only_action(&kept), ObjectAction::Keep));
        assert!(kept.held.is_empty());
        assert!(kept.advisories.is_empty());
    }

    /// PD-6: a survivor kept because every newer backup was abandoned on a
    /// bytes verdict is reported, not held.
    #[test]
    fn resolve_keeps_a_survivor_whose_newer_backups_were_abandoned_and_reports_it() {
        for verdict in [
            LineageVerdict::Survivor {
                step_downs: vec![down(20, decode(20))],
            },
            LineageVerdict::BackupsUnusable {
                exhaustion: "every one of 1 candidate generations was unusable".to_owned(),
                step_downs: vec![down(20, decode(20))],
                over_limit: vec![],
            },
        ] {
            let resolved = resolve_need(Some(10), Fallback::Keep, Some(verdict));

            assert!(matches!(only_action(&resolved), ObjectAction::Keep));
            assert!(resolved.held.is_empty(), "{:?}", resolved.held);
            assert_eq!(
                resolved.advisories,
                vec![Advisory::NewerBackupsUnusable {
                    object_id: OID,
                    session_id: S.to_owned(),
                    layer_name: L.to_owned(),
                    kept_seq: 10,
                    abandoned: vec![AbandonedGeneration {
                        seq: 20,
                        reason_label: "stepped_down_decode",
                        absent: false,
                    }],
                }]
            );
        }
    }

    /// defer-53: a survivor kept because its newer backups are exhausted
    /// with no step-down (every newer listed generation a tail with no base)
    /// is reported as kept with its backups exhausted — its kept `seq` and
    /// the exhaustion — not held, whether it is kept as it is or has its
    /// flags cleared. With a step-down the PD-6 advisory is reported instead
    /// (`resolve_keeps_a_survivor_whose_newer_backups_were_abandoned_and_reports_it`),
    /// and a survivor newer than every backup reports nothing.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with the `else if` arm that reports it removed, no advisory is
    /// reported.
    #[test]
    fn resolve_keeps_a_survivor_whose_backups_are_exhausted_with_no_step_down_and_reports_it() {
        const EXHAUSTION: &str = "every newer listed generation is a tail with no base";
        let band = BandInputs {
            floor_seq: 10,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        for fallback in [Fallback::Keep, Fallback::ClearFlags(band)] {
            let resolved = resolve_need(
                Some(10),
                fallback,
                Some(LineageVerdict::BackupsUnusable {
                    exhaustion: EXHAUSTION.to_owned(),
                    step_downs: vec![],
                    over_limit: vec![],
                }),
            );

            assert!(resolved.held.is_empty(), "{:?}", resolved.held);
            assert_eq!(
                resolved.advisories,
                vec![Advisory::KeptWithBackupsExhausted {
                    object_id: OID,
                    session_id: S.to_owned(),
                    layer_name: L.to_owned(),
                    kept_seq: 10,
                    exhaustion: EXHAUSTION.to_owned(),
                }],
                "{:?}",
                only_action(&resolved)
            );
        }
        let newer = resolve_need(
            Some(10),
            Fallback::Keep,
            Some(LineageVerdict::Survivor { step_downs: vec![] }),
        );
        assert!(newer.advisories.is_empty(), "{:?}", newer.advisories);
    }

    /// Every listed backup unusable and no survivor: held, nothing written.
    /// A retiring survivor falls back to clearing its flags.
    #[test]
    fn resolve_holds_an_object_with_no_survivor_and_no_usable_backup() {
        let unusable = || LineageVerdict::BackupsUnusable {
            exhaustion: "every one of 2 candidate generations was unusable".to_owned(),
            step_downs: vec![],
            over_limit: vec![],
        };
        let resolved = resolve_need(None, Fallback::NoSurvivor, Some(unusable()));
        assert!(matches!(only_action(&resolved), ObjectAction::Nothing));
        assert_eq!(
            held_reasons(&resolved),
            vec![HeldReason::BackupsUnusable {
                exhaustion: "every one of 2 candidate generations was unusable".to_owned(),
            }]
        );

        let band = BandInputs {
            floor_seq: 10,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        let retiring = resolve_need(Some(10), Fallback::ClearFlags(band), Some(unusable()));
        assert!(matches!(
            only_action(&retiring),
            ObjectAction::ClearFlags { band: b } if *b == band
        ));
        assert!(retiring.held.is_empty());
    }

    /// defer-3 holds with or without a survivor: no survivor writes nothing
    /// (`OverLimitPlainBase`); a survivor is kept and the newer backup holds
    /// (`NewerBackupUnusable`). Either way the layer is listed for the ERROR,
    /// with its base's generation.
    #[test]
    fn resolve_holds_a_plain_over_limit_base_with_or_without_a_survivor() {
        let over = || LineageVerdict::OverLimitPlainBase {
            head_seq: 22,
            base_seq: 20,
        };

        let lost = resolve_need(None, Fallback::NoSurvivor, Some(over()));
        assert!(matches!(only_action(&lost), ObjectAction::Nothing));
        assert_eq!(
            held_reasons(&lost),
            vec![HeldReason::OverLimitPlainBase {
                head_seq: 22,
                base_seq: 20,
            }]
        );
        assert_eq!(lost.over_limit, vec![(OID, S.to_owned(), L.to_owned(), 20)]);

        let surviving = resolve_need(Some(10), Fallback::Keep, Some(over()));
        assert!(matches!(only_action(&surviving), ObjectAction::Keep));
        assert_eq!(
            held_reasons(&surviving),
            vec![HeldReason::NewerBackupUnusable { head_seq: 22 }]
        );
        assert_eq!(surviving.over_limit.len(), 1);
    }

    /// PD-4: an unknown verdict holds; the survivor is kept, and with none
    /// nothing is written. A verdict the planner never produced is unknown
    /// too.
    #[test]
    fn resolve_holds_an_unknown_verdict_and_writes_only_the_fallback() {
        let unknown = || LineageVerdict::Unknown {
            error: "transient checkpoint-store error: 503".to_owned(),
        };
        let lost = resolve_need(None, Fallback::NoSurvivor, Some(unknown()));
        assert!(matches!(only_action(&lost), ObjectAction::Nothing));
        assert_eq!(
            held_reasons(&lost),
            vec![HeldReason::Unknown {
                error: "transient checkpoint-store error: 503".to_owned(),
            }]
        );

        let surviving = resolve_need(Some(10), Fallback::Keep, Some(unknown()));
        assert!(matches!(only_action(&surviving), ObjectAction::Keep));
        assert_eq!(surviving.held.len(), 1);

        let missing = resolve_need(None, Fallback::NoSurvivor, None);
        assert!(matches!(only_action(&missing), ObjectAction::Nothing));
        assert_eq!(
            held_reasons(&missing),
            vec![HeldReason::Unknown {
                error: "the object's lineage was not selected".to_owned(),
            }]
        );
    }

    /// A deferred lineage is trigger (c), naming the generation read as
    /// absent; the object resolves to its fallback until the `T_safe` pass.
    #[test]
    fn resolve_turns_a_deferred_lineage_into_trigger_c() {
        let resolved = resolve_need(
            Some(10),
            Fallback::Keep,
            Some(LineageVerdict::Defer { seq: 20 }),
        );

        assert!(matches!(only_action(&resolved), ObjectAction::Keep));
        assert!(resolved.held.is_empty());
        assert_eq!(
            resolved.triggers,
            vec![Trigger::LineageDeferred {
                object_id: OID,
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
                seq: 20,
            }]
        );
    }

    /// PD-6: a restore from an older backup is reported with each abandoned
    /// generation, newest first; PD-2: sidecar hashes no row references are
    /// reported, never held on.
    #[test]
    fn resolve_reports_a_restore_from_an_older_backup_and_sidecar_extras_without_holding() {
        let resolved = resolve_need(
            None,
            Fallback::NoSurvivor,
            Some(LineageVerdict::Use(selected(
                10,
                vec![down(20, decode(20))],
                3,
            ))),
        );

        assert!(matches!(
            only_action(&resolved),
            ObjectAction::Restore { .. }
        ));
        assert!(resolved.held.is_empty());
        assert_eq!(
            resolved.advisories,
            vec![
                Advisory::RestoredFromOlderBackup {
                    object_id: OID,
                    session_id: S.to_owned(),
                    layer_name: L.to_owned(),
                    restored_seq: 10,
                    abandoned: vec![AbandonedGeneration {
                        seq: 20,
                        reason_label: "stepped_down_decode",
                        absent: false,
                    }],
                },
                Advisory::SidecarHashesNotInState {
                    object_id: OID,
                    count: 3,
                    base_only: true,
                },
            ]
        );
    }

    /// The abandoned generations are newest first, each with its label and
    /// whether it was an absence rather than a bytes verdict.
    #[test]
    fn abandoned_generations_lists_newest_first_and_marks_the_absences() {
        let abandoned = abandoned_generations(&[
            down(20, decode(20)),
            down(40, link_absent(39)),
            down(30, missing_meta(30)),
            down(25, mismatch(25)),
            down(22, compose(22)),
        ]);

        let generation = |seq, reason_label, absent| AbandonedGeneration {
            seq,
            reason_label,
            absent,
        };
        assert_eq!(
            abandoned,
            vec![
                generation(40, "stepped_down_link_broken", true),
                generation(30, "stepped_down_missing_meta", true),
                generation(25, "stepped_down_link_mismatch", false),
                generation(22, "stepped_down_compose", false),
                generation(20, "stepped_down_decode", false),
            ]
        );
    }

    /// The summary's counts: each action and session kind once, and the two
    /// restored-from-older counts split by the newest abandoned generation's
    /// cause (decider condition 1).
    #[test]
    fn plan_counts_count_each_kind_and_split_older_restores_by_the_newest_cause() {
        let object = |object_id, action| ObjectPlan {
            object_id,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            action,
        };
        let band = BandInputs {
            floor_seq: 1,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        let older = |abandoned| Advisory::RestoredFromOlderBackup {
            object_id: OID,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            restored_seq: 10,
            abandoned,
        };
        let session = |session_id: &str, action| SessionPlan {
            session_id: session_id.to_owned(),
            action,
        };
        let plan = RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(0),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: vec![
                session(
                    "a",
                    SessionAction::Restore {
                        checkpoint: checkpoint("a", &[]),
                        step_b: true,
                    },
                ),
                session(
                    "b",
                    SessionAction::CompleteAbandonedDelete {
                        objects: vec![],
                        layer_names: vec![],
                        base_path: None,
                        mappings: vec![],
                        path_name_members: vec![],
                    },
                ),
                session("c", SessionAction::Nothing),
            ],
            objects: vec![
                object(
                    1,
                    ObjectAction::Restore {
                        reason: RestoreReason::HeaderAbsent,
                        lineage: Box::new(selected(10, vec![], 0).lineage),
                        band,
                    },
                ),
                object(2, ObjectAction::Keep),
                object(3, ObjectAction::Keep),
                object(4, ObjectAction::Remap { from: 4, to: 3 }),
                object(5, ObjectAction::ClearFlags { band }),
                object(6, ObjectAction::Nothing),
            ],
            coverage: CoveragePlan::default(),
            held: vec![HeldItem {
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
                object_id: Some(6),
                reason: HeldReason::NoBackup,
            }],
            advisories: vec![
                older(abandoned_generations(&[down(20, decode(20))])),
                older(abandoned_generations(&[down(20, link_absent(20))])),
                // The newest abandoned decides: an absence at 30 above a
                // bytes verdict at 20.
                older(abandoned_generations(&[
                    down(20, compose(20)),
                    down(30, missing_meta(30)),
                ])),
                Advisory::SessionWithoutBackup {
                    session_id: "d".to_owned(),
                    evidence: "the surviving mapping of layer x (object 9)".to_owned(),
                    layers_with_backups: vec![],
                },
                Advisory::UnreferencedBackup { object_id: 9 },
            ],
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        };

        assert_eq!(
            plan.counts(),
            PlanCounts {
                restore: 1,
                keep: 2,
                remap: 1,
                clear: 1,
                held: 1,
                sessions: 1,
                deletes: 1,
                restored_from_older: 1,
                restored_from_older_absent: 2,
                sessions_without_backup: 1,
            }
        );
    }

    // ── The Job's absence policy (judge_lineage) ─────────────────────────

    /// A verdict as a comparable word, with the sequence it names.
    fn verdict_word(verdict: &LineageVerdict) -> String {
        match verdict {
            LineageVerdict::Use(selected) => format!("use {}", selected.lineage.head_seq),
            LineageVerdict::Survivor { .. } => "survivor".to_owned(),
            LineageVerdict::BackupsUnusable { .. } => "unusable".to_owned(),
            LineageVerdict::OverLimitPlainBase { .. } => "over-limit".to_owned(),
            LineageVerdict::Unknown { .. } => "unknown".to_owned(),
            LineageVerdict::Defer { seq } => format!("defer {seq}"),
        }
    }

    /// Each walk outcome, and the verdict it stands as when no absence or
    /// budget rule overrides it.
    fn outcomes() -> Vec<(&'static str, LineageOutcome, &'static str)> {
        vec![
            (
                "selected",
                LineageOutcome::Selected(selected(10, vec![], 0)),
                "use 10",
            ),
            ("not newer", LineageOutcome::NotNewer, "survivor"),
            (
                "all broken",
                LineageOutcome::Exhausted(Exhaustion::AllBroken { candidates: 2 }),
                "unusable",
            ),
            (
                "over the limit",
                LineageOutcome::OverLimitPlainBase {
                    head_seq: 10,
                    base_seq: 10,
                },
                "over-limit",
            ),
            (
                "errored",
                LineageOutcome::Errored {
                    error: "503".to_owned(),
                },
                "unknown",
            ),
        ]
    }

    /// INV-3: an absence (a link absent at fetch or from the listing, a
    /// missing meta companion) above the floor defers the object before
    /// `T_safe`, whatever the walk ended with, naming the absent link; at
    /// `T_safe` the step-down stands and so does the outcome.
    #[test]
    fn judge_lineage_defers_an_absence_before_t_safe_whatever_the_outcome() {
        // The absent link is generation 38, under a candidate headed at 40.
        for (name, reason) in [
            ("a link absent at fetch", link_absent(38)),
            (
                "a link absent from the listing",
                LineageBreak::LinkAbsent {
                    seq: 38,
                    kind: CheckpointKind::Base,
                    at_fetch: false,
                },
            ),
            ("a missing meta companion", missing_meta(38)),
        ] {
            for (outcome_name, outcome, stands) in outcomes() {
                let before =
                    judge_lineage(false, None, &[down(40, reason.clone())], outcome.clone());
                assert_eq!(
                    verdict_word(&before),
                    "defer 38",
                    "{name}, then {outcome_name}, before T_safe"
                );
                let at = judge_lineage(true, None, &[down(40, reason.clone())], outcome);
                assert_eq!(
                    verdict_word(&at),
                    stands,
                    "{name}, then {outcome_name}, at T_safe"
                );
            }
        }
    }

    /// PD-6: a bytes verdict (decode, link mismatch, compose) is a step-down
    /// before and at `T_safe` alike; the outcome stands.
    #[test]
    fn judge_lineage_lets_a_bytes_verdict_step_down_before_and_at_t_safe() {
        for (name, reason) in [
            ("a decode failure", decode(40)),
            ("a link mismatch", mismatch(40)),
            ("a compose failure", compose(40)),
        ] {
            for at in [false, true] {
                for (outcome_name, outcome, stands) in outcomes() {
                    let verdict = judge_lineage(at, None, &[down(40, reason.clone())], outcome);
                    assert_eq!(
                        verdict_word(&verdict),
                        stands,
                        "{name}, then {outcome_name}, at_or_after_t_safe={at}"
                    );
                }
            }
        }
    }

    /// A budget cut is never a step-down: any `Budget` step-down, or a
    /// `BudgetInconclusive` exhaustion, is unknown before and at `T_safe`,
    /// even when the walk selected an older lineage. The absence rule comes
    /// first.
    #[test]
    fn judge_lineage_makes_any_budget_cut_unknown() {
        for at in [false, true] {
            for (outcome_name, outcome, _) in outcomes() {
                let verdict = judge_lineage(at, Some(5), &[down(40, budget(40))], outcome);
                assert_eq!(
                    verdict_word(&verdict),
                    "unknown",
                    "{outcome_name}, at_or_after_t_safe={at}"
                );
            }
            let inconclusive = judge_lineage(
                at,
                None,
                &[],
                LineageOutcome::Exhausted(Exhaustion::BudgetInconclusive {
                    candidates: 2,
                    unadjudicated_bases: 1,
                }),
            );
            assert_eq!(verdict_word(&inconclusive), "unknown", "at={at}");
        }
        let both = judge_lineage(
            false,
            None,
            &[down(40, budget(40)), down(30, link_absent(30))],
            LineageOutcome::NotNewer,
        );
        assert_eq!(
            verdict_word(&both),
            "defer 30",
            "before T_safe an absence defers even beside a budget cut"
        );
    }

    /// Only a step-down of a candidate above the floor defers: one at the
    /// floor could not have been restored anyway.
    #[test]
    fn judge_lineage_defers_only_absences_above_the_floor() {
        let at_floor = judge_lineage(
            false,
            Some(40),
            &[down(40, link_absent(40))],
            LineageOutcome::NotNewer,
        );
        assert_eq!(verdict_word(&at_floor), "survivor");

        let above = judge_lineage(
            false,
            Some(40),
            &[down(41, link_absent(41))],
            LineageOutcome::NotNewer,
        );
        assert_eq!(verdict_word(&above), "defer 41");
    }

    /// An empty listing or one with no base is a listing absence: deferred
    /// before `T_safe`, "backups unusable" at it.
    #[test]
    fn judge_lineage_defers_an_empty_or_baseless_listing_only_before_t_safe() {
        let empty = LineageOutcome::Exhausted(Exhaustion::EmptyListing);
        let no_base = LineageOutcome::Exhausted(Exhaustion::NoBase {
            generations: 2,
            newest_sequence: 44,
        });

        assert_eq!(
            verdict_word(&judge_lineage(false, None, &[], empty.clone())),
            "defer 0"
        );
        assert_eq!(
            verdict_word(&judge_lineage(false, None, &[], no_base.clone())),
            "defer 44"
        );
        assert_eq!(
            verdict_word(&judge_lineage(true, None, &[], empty)),
            "unusable"
        );
        assert_eq!(
            verdict_word(&judge_lineage(true, None, &[], no_base)),
            "unusable"
        );
    }

    // ── SR-9.4, sessions ─────────────────────────────────────────────────

    /// Rule 1: a surviving deletion marker finishes the abandoned delete of
    /// every surviving object mapped under the session or naming it, with the
    /// listed backup's base path when there is one; nothing of it is planned.
    #[test]
    fn plan_session_with_a_deletion_marker_finishes_the_abandoned_delete() {
        let store = facts(
            vec![complete(11, S, "a.usda", 5)],
            vec![mapping(S, "b.usda", 12)],
            &[(S, deleted())],
        );
        for (backed_up, want_base_path) in [(true, Some("stage:/sess.usd")), (false, None)] {
            let mut listing = listed(true, &[(12, &[3])]);
            if backed_up {
                listing = with_session(
                    listing,
                    S,
                    SessionRead::Present(checkpoint(S, &[("b.usda", 12)])),
                );
            }

            let pre = persistent(&listing, &store);

            match session_action(&pre, S) {
                SessionAction::CompleteAbandonedDelete {
                    objects,
                    layer_names,
                    base_path,
                    ..
                } => {
                    assert_eq!(objects, &vec![11, 12]);
                    assert_eq!(layer_names, &vec!["a.usda".to_owned(), "b.usda".to_owned()]);
                    assert_eq!(base_path.as_deref(), want_base_path);
                }
                other => panic!("expected the abandoned delete, got {other:?}"),
            }
            assert!(
                planned_ids(&pre).is_empty(),
                "nothing of a deleted session is restored: {:?}",
                pre.objects
            );
            assert!(pre.s7.is_empty());
        }
    }

    /// Rule 2: a listed session backup restores the session. Its mapping of a
    /// layer is written when no survivor maps it or the survivor names an
    /// older object, and kept out when the survivor names the same or a newer
    /// one; Step B runs when a mapping is written or the record is gone; the
    /// layers index names every mapping that exists after Step B.
    #[test]
    fn plan_session_with_a_listed_backup_writes_only_mappings_newer_than_the_survivor() {
        /// (case, survivor of L, backup's object for L, record survives,
        /// mappings written, Step B, planned objects)
        type Case = (
            &'static str,
            Option<u64>,
            u64,
            bool,
            &'static [(&'static str, u64)],
            bool,
            &'static [u64],
        );
        let cases: [Case; 5] = [
            ("survivor newer", Some(8), 6, true, &[], false, &[8]),
            ("survivor the same", Some(8), 8, true, &[], false, &[8]),
            ("survivor older", Some(6), 8, true, &[(L, 8)], true, &[8]),
            ("no survivor", None, 8, true, &[(L, 8)], true, &[8]),
            ("record lost", Some(8), 6, false, &[], true, &[8]),
        ];
        for (case, survivor, backed_up, record, want_writes, want_step_b, want_planned) in cases {
            let mut headers = vec![];
            let mut mappings = vec![];
            if let Some(x) = survivor {
                headers.push(complete(x, S, L, 50));
                mappings.push(mapping(S, L, x));
            }
            let sessions: Vec<(&str, SessionFacts)> =
                if record { vec![(S, live())] } else { vec![] };
            let store = facts(headers, mappings, &sessions);
            let listing = with_session(
                listed(true, &[(backed_up, &[40])]),
                S,
                SessionRead::Present(checkpoint(S, &[(L, backed_up)])),
            );

            let pre = persistent(&listing, &store);

            match session_action(&pre, S) {
                SessionAction::Restore { checkpoint, step_b } => {
                    let want: BTreeMap<String, u64> = want_writes
                        .iter()
                        .map(|(l, o)| ((*l).to_owned(), *o))
                        .collect();
                    assert_eq!(checkpoint.layer_mappings, want, "{case}: mappings written");
                    assert_eq!(*step_b, want_step_b, "{case}: Step B");
                    assert_eq!(
                        checkpoint.layers_index,
                        BTreeSet::from([L.to_owned()]),
                        "{case}: layers index"
                    );
                    assert_eq!(checkpoint.base_path, "stage:/sess.usd", "{case}");
                }
                other => panic!("{case}: expected a session restore, got {other:?}"),
            }
            assert_eq!(
                planned_ids(&pre),
                want_planned.iter().copied().collect::<BTreeSet<u64>>(),
                "{case}: planned objects"
            );
        }
    }

    /// Rule 2: a backup's mapping to an object on the dead set is dropped
    /// with an advisory, and the other mappings are still written.
    #[test]
    fn plan_session_with_a_backup_mapping_to_a_deleted_object_drops_it_with_an_advisory() {
        let mut store = facts(vec![], vec![], &[(S, live())]);
        store.dead.insert(9);
        let listing = with_session(
            listed(true, &[(9, &[4]), (10, &[4])]),
            S,
            SessionRead::Present(checkpoint(S, &[(L, 9), ("other.usda", 10)])),
        );

        let pre = persistent(&listing, &store);

        match session_action(&pre, S) {
            SessionAction::Restore { checkpoint, step_b } => {
                assert_eq!(
                    checkpoint.layer_mappings,
                    BTreeMap::from([("other.usda".to_owned(), 10)])
                );
                assert_eq!(
                    checkpoint.layers_index,
                    BTreeSet::from(["other.usda".to_owned()])
                );
                assert!(*step_b);
            }
            other => panic!("expected a session restore, got {other:?}"),
        }
        assert!(pre
            .advisories
            .contains(&Advisory::CheckpointMappingToDeletedObject {
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
                object_id: 9,
            }));
        assert_eq!(planned_ids(&pre), BTreeSet::from([10]));
    }

    /// Rule 3: a listed session backup read as absent is trigger (d) before
    /// `T_safe` and holds the session at it; either way its surviving
    /// mappings are still planned. A listed id with no read is the same.
    #[test]
    fn plan_session_backup_read_as_absent_triggers_before_t_safe_and_holds_at_it() {
        for read in [Some(SessionRead::Absent), None] {
            for at in [false, true] {
                let store = facts(
                    vec![complete(OID, S, L, 10)],
                    vec![mapping(S, L, OID)],
                    &[(S, live())],
                );
                let mut listing = listed(at, &[(OID, &[9])]);
                listing.session_ids.insert(S.to_owned());
                if let Some(read) = read.clone() {
                    listing.sessions.insert(S.to_owned(), read);
                }

                let pre = persistent(&listing, &store);

                let held = pre.held.contains(&HeldItem {
                    session_id: S.to_owned(),
                    layer_name: String::new(),
                    object_id: None,
                    reason: HeldReason::SessionBackupUnreadable {
                        session_id: S.to_owned(),
                        error: "the listed session backup reads as absent at T_safe".to_owned(),
                        cause: SessionReadFailure::AbsentAtTSafe,
                    },
                });
                assert_eq!(held, at, "{read:?}, at={at}: {:?}", pre.held);
                assert_eq!(
                    pre.triggers.contains(&Trigger::SessionBackupAbsent {
                        session_id: S.to_owned()
                    }),
                    !at,
                    "{read:?}, at={at}: {:?}",
                    pre.triggers
                );
                assert!(
                    matches!(decided_of(&pre, OID).0, ObjectAction::Keep),
                    "{read:?}, at={at}: its surviving mappings are still planned"
                );
            }
        }
    }

    /// Rule 3: a listed session backup that could not be read or decoded
    /// holds the session at once, before `T_safe` too: it is no absence.
    #[test]
    fn plan_session_backup_unreadable_holds_at_once() {
        let store = facts(vec![], vec![], &[(S, live())]);
        let listing = with_session(
            listed(false, &[]),
            S,
            SessionRead::ReadFailed("permanent checkpoint-store error: malformed JSON".to_owned()),
        );

        let pre = persistent(&listing, &store);

        assert_eq!(
            pre.held,
            vec![HeldItem {
                session_id: S.to_owned(),
                layer_name: String::new(),
                object_id: None,
                reason: HeldReason::SessionBackupUnreadable {
                    session_id: S.to_owned(),
                    error: "permanent checkpoint-store error: malformed JSON".to_owned(),
                    cause: SessionReadFailure::ReadError,
                },
            }]
        );
        assert!(pre.triggers.is_empty(), "{:?}", pre.triggers);
    }

    /// Rule 4: a surviving mapping that names an object on the dead set is
    /// held, and the object is never restored.
    #[test]
    fn plan_session_surviving_mapping_to_a_deleted_object_is_held_and_not_restored() {
        let mut store = facts(vec![], vec![mapping(S, L, OID)], &[(S, live())]);
        store.dead.insert(OID);

        let pre = persistent(&listed(true, &[(OID, &[5])]), &store);

        assert_eq!(
            pre.held,
            vec![HeldItem {
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
                object_id: Some(OID),
                reason: HeldReason::MappedToDeletedObject {
                    session_id: S.to_owned(),
                    layer_name: L.to_owned(),
                },
            }]
        );
        assert!(planned_ids(&pre).is_empty());
        assert!(
            !pre.advisories
                .contains(&Advisory::UnreferencedBackup { object_id: OID }),
            "a deleted object's backup is not reported as unreferenced"
        );
    }

    /// Rule 5: a session whose record survives with no listed backup needs
    /// nothing; its mappings are planned.
    #[test]
    fn plan_session_with_a_surviving_record_and_no_backup_needs_nothing() {
        let store = facts(
            vec![complete(OID, S, L, 10)],
            vec![mapping(S, L, OID)],
            &[(S, live())],
        );

        let pre = persistent(&listed(true, &[(OID, &[9])]), &store);

        assert!(matches!(session_action(&pre, S), SessionAction::Nothing));
        assert!(matches!(decided_of(&pre, OID).0, ObjectAction::Keep));
        assert!(pre.held.is_empty());
        assert!(pre.s7.is_empty());
    }

    // ── PD-1 / S7 and listing trust ──────────────────────────────────────

    /// Rule 6 under an empty session listing (S7): a session a surviving
    /// header names, with no record, no marker and no backup, is trigger (f)
    /// before `T_safe` and an S7 hit at it, naming the evidence.
    #[test]
    fn plan_session_named_only_by_evidence_under_an_empty_session_listing_is_s7_at_t_safe() {
        let evidence = format!("the surviving header of object {OID} (layer {L})");
        for at in [false, true] {
            let store = facts(vec![complete(OID, S, L, 10)], vec![], &[]);

            let pre = persistent(&listed(at, &[(OID, &[5])]), &store);

            let hit = S7Hit {
                session_id: S.to_owned(),
                evidence: evidence.clone(),
            };
            let trigger = Trigger::SessionNamedWithoutBackup {
                session_id: S.to_owned(),
                evidence: evidence.clone(),
            };
            if at {
                assert_eq!(pre.s7, vec![hit]);
                assert!(!pre.triggers.contains(&trigger));
            } else {
                assert!(pre.s7.is_empty());
                assert!(pre.triggers.contains(&trigger), "{:?}", pre.triggers);
            }
        }
    }

    /// The evidence named is the first that names the session: a surviving
    /// mapping before a surviving header, and a listed backup's identity
    /// (PD-1's identity read) when nothing in the store names it.
    #[test]
    fn plan_session_s7_evidence_names_a_mapping_a_header_or_a_listed_backups_identity() {
        let store = facts(vec![complete(OID, S, L, 10)], vec![mapping(S, L, OID)], &[]);
        let pre = persistent(&listed(true, &[(OID, &[5])]), &store);
        assert_eq!(
            pre.s7,
            vec![S7Hit {
                session_id: S.to_owned(),
                evidence: format!("the surviving mapping of layer {L} (object {OID})"),
            }]
        );

        let identities = BTreeMap::from([(OID, (S.to_owned(), L.to_owned()))]);
        let listing = listed(true, &[(OID, &[5])]);
        let empty = StoreFacts::default();
        assert_eq!(
            session_universe(&listing, &empty, &identities),
            BTreeSet::from([S.to_owned()])
        );
        let pre = plan_pre_lineage(&PlanInputs {
            access: StoreAccess::Persistent,
            listing: &listing,
            store: &empty,
            identities: &identities,
            unidentified: &BTreeMap::new(),
        });
        assert_eq!(
            pre.s7,
            vec![S7Hit {
                session_id: S.to_owned(),
                evidence: format!("the listed backup of object {OID} (layer {L})"),
            }]
        );
    }

    /// A session named by evidence whose deletion marker survives is an
    /// abandoned delete, not an S7 hit.
    #[test]
    fn plan_session_named_by_evidence_with_a_deletion_marker_is_no_s7_hit() {
        let store = facts(vec![complete(OID, S, L, 10)], vec![], &[(S, deleted())]);

        let pre = persistent(&listed(true, &[(OID, &[5])]), &store);

        assert!(pre.s7.is_empty());
        assert!(matches!(
            session_action(&pre, S),
            SessionAction::CompleteAbandonedDelete { .. }
        ));
    }

    /// PD-1 under a non-empty session listing: the session is reported, with
    /// the layers whose backups are listed (none, for one named only by an
    /// unlisted mapping), and its objects and backups are left unreferenced;
    /// nothing holds and nothing triggers.
    #[test]
    fn plan_session_named_only_by_evidence_under_a_non_empty_session_listing_is_reported() {
        let store = facts(
            vec![complete(71, S, L, 10)],
            vec![mapping("s2", "m.usda", 72)],
            &[("other", live())],
        );
        let listing = with_session(
            listed(true, &[(71, &[5])]),
            "other",
            SessionRead::Present(checkpoint("other", &[])),
        );

        let pre = persistent(&listing, &store);

        assert!(pre.s7.is_empty());
        assert!(pre.held.is_empty(), "{:?}", pre.held);
        assert!(
            !pre.triggers
                .iter()
                .any(|t| matches!(t, Trigger::SessionNamedWithoutBackup { .. })),
            "{:?}",
            pre.triggers
        );
        for advisory in [
            Advisory::SessionWithoutBackup {
                session_id: S.to_owned(),
                evidence: format!("the surviving header of object 71 (layer {L})"),
                layers_with_backups: vec![L.to_owned()],
            },
            Advisory::SessionWithoutBackup {
                session_id: "s2".to_owned(),
                evidence: "the surviving mapping of layer m.usda (object 72)".to_owned(),
                layers_with_backups: vec![],
            },
            Advisory::UnreferencedObject {
                object_id: 71,
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
            },
            Advisory::UnreferencedBackup { object_id: 71 },
        ] {
            assert!(
                pre.advisories.contains(&advisory),
                "missing {advisory:?} in {:?}",
                pre.advisories
            );
        }
        assert!(planned_ids(&pre).is_empty(), "{:?}", pre.objects);
    }

    /// PD-1 with no backup store (P4): each surviving object of the session
    /// holds `NoBackupStore`, named by its mapping's or header's layer.
    #[test]
    fn plan_session_named_only_by_evidence_with_no_backup_store_holds_its_objects() {
        let store = facts(
            vec![complete(71, S, L, 10)],
            vec![mapping(S, "m.usda", 72)],
            &[],
        );
        let listing = Listing {
            at_or_after_t_safe: true,
            ..Listing::default()
        };

        let pre = pre_plan(
            StoreAccess::NoStore(NoBackupStore::Disabled),
            &listing,
            &store,
        );

        let held = |object_id, layer: &str| HeldItem {
            session_id: S.to_owned(),
            layer_name: layer.to_owned(),
            object_id: Some(object_id),
            reason: HeldReason::NoBackupStore,
        };
        assert_eq!(pre.held, vec![held(71, L), held(72, "m.usda")]);
        assert!(pre.s7.is_empty());
        assert!(pre.triggers.is_empty(), "{:?}", pre.triggers);
    }

    /// Trigger (a): a persistent store listing nothing at all is re-listed at
    /// `T_safe`; a listing taken at `T_safe`, a listed session backup alone,
    /// or no backup store, triggers nothing.
    #[test]
    fn plan_all_empty_listing_on_a_persistent_store_is_trigger_a_only_before_t_safe() {
        let store = StoreFacts::default();
        let empty = |at| listed(at, &[]);

        assert!(persistent(&empty(false), &store)
            .triggers
            .contains(&Trigger::EmptyListing));
        assert!(!persistent(&empty(true), &store)
            .triggers
            .contains(&Trigger::EmptyListing));
        assert!(pre_plan(
            StoreAccess::NoStore(NoBackupStore::Disabled),
            &empty(false),
            &store
        )
        .triggers
        .is_empty());
        let sessions_only = with_session(empty(false), S, SessionRead::Present(checkpoint(S, &[])));
        assert!(!persistent(&sessions_only, &store)
            .triggers
            .contains(&Trigger::EmptyListing));
    }

    /// Trigger (e): a stamped survivor the listing does not show, not on the
    /// dead set, contradicts the listing; before `T_safe` it triggers, at it
    /// it is the plan's contradiction. Listed, dead, or with no backup store,
    /// it contradicts nothing.
    #[test]
    fn plan_stamped_survivor_the_listing_does_not_show_contradicts_it() {
        let stamped = plus(complete(OID, S, L, 10), &[("last_checkpoint_seq", "9")]);
        let store = facts(vec![stamped], vec![mapping(S, L, OID)], &[(S, live())]);
        let contradiction = Contradiction::Layer(CheckpointStamp {
            object_id: OID,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            last_checkpoint_seq: Some(9),
            last_checkpoint_ms: None,
        });
        let elsewhere = |at| listed(at, &[(99, &[1])]);

        let before = persistent(&elsewhere(false), &store);
        assert!(before
            .triggers
            .contains(&Trigger::Contradiction(contradiction.clone())));
        assert!(before.contradictions.is_empty());
        assert_eq!(
            persistent(&elsewhere(true), &store).contradictions,
            vec![contradiction]
        );

        assert!(persistent(&listed(true, &[(OID, &[9])]), &store)
            .contradictions
            .is_empty());
        let mut dead = store.clone();
        dead.dead.insert(OID);
        assert!(persistent(&elsewhere(true), &dead)
            .contradictions
            .is_empty());
        assert!(pre_plan(
            StoreAccess::NoStore(NoBackupStore::Disabled),
            &listed(true, &[]),
            &store
        )
        .contradictions
        .is_empty());
    }

    /// Trigger (e), sessions: a session whose record survives and whose last
    /// landed backup-store operation stored a session backup must be listed;
    /// a last operation that deleted one is no evidence.
    #[test]
    fn plan_session_whose_last_backup_operation_was_a_write_must_be_listed() {
        let session = |kind: &str| SessionFacts {
            meta_exists: true,
            deleted_marker_exists: false,
            backup_kind: Some(kind.to_owned()),
            layers_index: Vec::new(),
        };
        let elsewhere = listed(true, &[(99, &[1])]);

        let written = facts(vec![], vec![], &[(S, session("write"))]);
        assert_eq!(
            persistent(&elsewhere, &written).contradictions,
            vec![Contradiction::Session {
                session_id: S.to_owned()
            }]
        );
        let shown = with_session(
            elsewhere.clone(),
            S,
            SessionRead::Present(checkpoint(S, &[])),
        );
        assert!(persistent(&shown, &written).contradictions.is_empty());
        let deleted_last = facts(vec![], vec![], &[(S, session("delete"))]);
        assert!(persistent(&elsewhere, &deleted_last)
            .contradictions
            .is_empty());
    }

    // ── The coverage merge (SR-9.3) and the outcome ──────────────────────

    /// The check's order: Valkey, then the restored tails, then this run's
    /// own PUT; only a value none of them holds is probed.
    #[test]
    fn settle_without_probe_takes_valkey_then_the_tails_then_an_own_put() {
        assert_eq!(
            settle_without_probe(true, true, true),
            Some(HashVerdict::Valkey)
        );
        assert_eq!(
            settle_without_probe(false, true, true),
            Some(HashVerdict::Tails)
        );
        assert_eq!(
            settle_without_probe(false, false, true),
            Some(HashVerdict::OwnPut)
        );
        assert_eq!(settle_without_probe(false, false, false), None);
    }

    /// A probe's answer: present covers; absent is uncovered only at or after
    /// `T_safe` and pending a re-probe before it; an error is unknown.
    #[test]
    fn probe_verdict_trusts_an_absent_answer_only_at_or_after_t_safe() {
        for at in [false, true] {
            assert_eq!(probe_verdict(&Ok(true), at), HashVerdict::Backup);
            assert_eq!(
                probe_verdict(&Err("503".to_owned()), at),
                HashVerdict::Unknown("503".to_owned())
            );
        }
        assert_eq!(
            probe_verdict(&Ok(false), false),
            HashVerdict::PendingReprobe
        );
        assert_eq!(probe_verdict(&Ok(false), true), HashVerdict::Uncovered);
    }

    #[test]
    fn hash_verdict_is_covered_only_by_a_store_that_holds_the_value() {
        for verdict in [
            HashVerdict::Valkey,
            HashVerdict::Tails,
            HashVerdict::OwnPut,
            HashVerdict::Backup,
        ] {
            assert!(verdict.is_covered(), "{verdict:?}");
        }
        for verdict in [
            HashVerdict::PendingReprobe,
            HashVerdict::Uncovered,
            HashVerdict::Unknown("503".to_owned()),
        ] {
            assert!(!verdict.is_covered(), "{verdict:?}");
        }
    }

    /// The universe: a restored object contributes its restored state's
    /// rows (never its surviving index), a kept or flag-cleared object its
    /// surviving `obj:ext`, a remapped-from or held object nothing; a value
    /// records every object that references it.
    #[test]
    fn coverage_universe_takes_restored_rows_and_surviving_indexes_and_nothing_from_held_objects() {
        let hash = |b: u8| [b; 32];
        let band = BandInputs {
            floor_seq: 1,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        let object = |object_id, action| ObjectPlan {
            object_id,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            action,
        };
        let objects = vec![
            object(
                1,
                ObjectAction::Restore {
                    reason: RestoreReason::HeaderAbsent,
                    lineage: Box::new(selected(20, vec![], 0).lineage),
                    band,
                },
            ),
            object(2, ObjectAction::Keep),
            object(3, ObjectAction::ClearFlags { band }),
            object(4, ObjectAction::Remap { from: 4, to: 2 }),
            object(5, ObjectAction::Nothing),
        ];
        let restored = BTreeMap::from([(1, BTreeSet::from([hash(1), hash(9)]))]);
        // Every object has a surviving index, to show whose the universe reads.
        let survivors: BTreeMap<u64, BTreeSet<ExtHash>> = (1..=5u8)
            .map(|o| (u64::from(o), BTreeSet::from([hash(10 + o), hash(9)])))
            .collect();

        let universe = coverage_universe(&objects, &restored, &survivors);

        assert_eq!(
            universe,
            BTreeMap::from([
                (hash(1), BTreeSet::from([1])),
                (hash(9), BTreeSet::from([1, 2, 3])),
                (hash(12), BTreeSet::from([2])),
                (hash(13), BTreeSet::from([3])),
            ])
        );
        assert_eq!(survivors_to_cover(&objects), vec![2, 3]);
    }

    /// The run's outcome: writable only with nothing held and every value
    /// covered; a held item, an uncovered or an unknown value keeps the
    /// service held.
    #[test]
    fn restore_plan_outcome_stays_held_on_a_held_item_or_an_uncovered_or_unknown_value() {
        let plan = |held: Vec<HeldItem>| RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(0),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: vec![],
            objects: vec![],
            coverage: CoveragePlan::default(),
            held,
            advisories: vec![],
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        };
        let report = |verdicts: Vec<HashVerdict>| CoverageReport {
            verdicts: verdicts
                .into_iter()
                .enumerate()
                .map(|(i, v)| ([i as u8; 32], v))
                .collect(),
            layers: vec![],
        };

        assert_eq!(
            plan(vec![]).outcome(&report(vec![
                HashVerdict::Valkey,
                HashVerdict::Tails,
                HashVerdict::OwnPut,
                HashVerdict::Backup,
            ])),
            PlanOutcome::Writable
        );
        assert_eq!(
            plan(vec![]).outcome(&report(vec![HashVerdict::Backup, HashVerdict::Uncovered])),
            PlanOutcome::StaysHeld
        );
        assert_eq!(
            plan(vec![]).outcome(&report(vec![HashVerdict::Unknown("503".to_owned())])),
            PlanOutcome::StaysHeld
        );
        let held = HeldItem {
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            object_id: Some(OID),
            reason: HeldReason::NoBackup,
        };
        assert_eq!(
            plan(vec![held]).outcome(&report(vec![])),
            PlanOutcome::StaysHeld
        );
    }

    // ── U2: the writer's consistency check (`check_executable`) ─────────────

    /// A plan with every kind of entry the rules look at, consistent:
    /// object 1 restored and mapped by session "a"'s Step B; 2 kept; 3 moved
    /// to 1; 4 flag-cleared; session "d"'s abandoned delete of 9 and 10; and
    /// a held mapping to a deleted object of a layer no Step B writes.
    fn executable_plan() -> RestorePlan {
        let object = |object_id, action| ObjectPlan {
            object_id,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            action,
        };
        let band = BandInputs {
            floor_seq: 1,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        let session = |session_id: &str, action| SessionPlan {
            session_id: session_id.to_owned(),
            action,
        };
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(0),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: vec![
                session(
                    "a",
                    SessionAction::Restore {
                        checkpoint: checkpoint("a", &[("one.usda", 1)]),
                        step_b: true,
                    },
                ),
                session(
                    "d",
                    SessionAction::CompleteAbandonedDelete {
                        objects: vec![9],
                        layer_names: vec!["nine.usda".to_owned(), "ten.usda".to_owned()],
                        base_path: None,
                        mappings: vec![("ten.usda".to_owned(), 10)],
                        path_name_members: vec![],
                    },
                ),
            ],
            objects: vec![
                object(
                    1,
                    ObjectAction::Restore {
                        reason: RestoreReason::HeaderAbsent,
                        lineage: Box::new(selected(10, vec![], 0).lineage),
                        band,
                    },
                ),
                object(2, ObjectAction::Keep),
                object(3, ObjectAction::Remap { from: 3, to: 1 }),
                object(4, ObjectAction::ClearFlags { band }),
            ],
            coverage: CoveragePlan::default(),
            held: vec![HeldItem {
                session_id: "a".to_owned(),
                layer_name: "gone.usda".to_owned(),
                object_id: Some(5),
                reason: HeldReason::MappedToDeletedObject {
                    session_id: "a".to_owned(),
                    layer_name: "gone.usda".to_owned(),
                },
            }],
            advisories: vec![],
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        }
    }

    #[test]
    fn check_executable_accepts_a_consistent_plan() {
        assert_eq!(check_executable(&executable_plan()), vec![]);
    }

    /// Rule 1: an object with two writing actions.
    #[test]
    fn check_executable_refuses_an_object_written_twice() {
        let mut plan = executable_plan();
        let mut again = plan.objects[3].clone();
        again.object_id = 1;
        plan.objects.push(again);
        assert_eq!(
            check_executable(&plan),
            vec![PlanDefect::TwoWritingActions { object_id: 1 }]
        );
    }

    /// Rule 2 (defer-51's writer half): a held mapping to a deleted object
    /// that a session restore's Step B writes; the same session restored
    /// without Step B writes no mapping, so the hold stands.
    #[test]
    fn check_executable_refuses_a_held_mapping_that_step_b_writes() {
        let mut plan = executable_plan();
        plan.held[0].layer_name = "one.usda".to_owned();
        plan.held[0].reason = HeldReason::MappedToDeletedObject {
            session_id: "a".to_owned(),
            layer_name: "one.usda".to_owned(),
        };
        assert_eq!(
            check_executable(&plan),
            vec![PlanDefect::HeldMappingWritten {
                session_id: "a".to_owned(),
                layer_name: "one.usda".to_owned(),
            }]
        );

        if let SessionAction::Restore { step_b, .. } = &mut plan.sessions[0].action {
            *step_b = false;
        }
        assert_eq!(check_executable(&plan), vec![], "no Step B, no write");
    }

    /// Rule 3: an object an abandoned delete removes, named by its objects
    /// or its mappings, that is also planned, or mapped by a Step B.
    #[test]
    fn check_executable_refuses_a_deleted_object_planned_or_mapped() {
        let mut planned = executable_plan();
        if let SessionAction::CompleteAbandonedDelete { objects, .. } =
            &mut planned.sessions[1].action
        {
            objects.push(2);
        }
        assert_eq!(
            check_executable(&planned),
            vec![PlanDefect::DeletedObjectPlanned {
                session_id: "d".to_owned(),
                object_id: 2,
            }]
        );

        let mut mapped = executable_plan();
        if let SessionAction::CompleteAbandonedDelete { mappings, .. } =
            &mut mapped.sessions[1].action
        {
            mappings.push(("one.usda".to_owned(), 1));
        }
        assert_eq!(
            check_executable(&mapped),
            vec![
                PlanDefect::DeletedObjectPlanned {
                    session_id: "d".to_owned(),
                    object_id: 1,
                },
                PlanDefect::DeletedObjectMapped {
                    session_id: "d".to_owned(),
                    object_id: 1,
                },
            ]
        );
    }

    /// Rule 4: a move to a successor the plan does not plan.
    #[test]
    fn check_executable_refuses_a_move_to_an_unplanned_successor() {
        let mut plan = executable_plan();
        plan.objects[2].action = ObjectAction::Remap { from: 3, to: 8 };
        assert_eq!(
            check_executable(&plan),
            vec![PlanDefect::RemapTargetUnplanned { from: 3, to: 8 }]
        );
    }
}
