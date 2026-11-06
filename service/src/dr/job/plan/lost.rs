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

//! Accept-loss's lost set (SDD SR-10; PD-1, PD-2 with option E, PD-6 to
//! PD-8; defer-47, defer-51, defer-55): what the accept-loss Job discards,
//! read from this run's restore plan and its finalised coverage report
//! alone, after the restore writer wrote everything restorable. Pure: no
//! I/O, no waiting, and never a scan of the store, so the Job acts only on
//! its own plan's items (defer-55).
//!
//! # The lost set ([`lost_set`])
//!
//! | Plan item | Accept-loss |
//! |---|---|
//! | held `NoBackup`, `BackupsUnusable` or `NoBackupStore` on a layer of a live session, whose object the plan writes nothing for | a lost layer: registered dead, wiped after its header is re-read, unmapped |
//! | held `BackupsOverLimit` on such a layer (PD-4's step-down past over-limit full backups found nothing older) | a lost layer that is only unmapped: never registered dead and never wiped, so its backups stay for a release whose limit fits them (recovery is manual: nothing maps them after D3 and D6), reported with a WARN (as PD-7 reports an unreferenced backup) |
//! | held `MappedToDeletedObject` | a lost layer whose object is already on the dead-object set: wiped and unmapped |
//! | a session with no record, no deletion marker and no restore: named by evidence with no listed session backup, or whose listed backup is undecodable or absent at `T_safe` | a lost session (PD-6): its surviving objects registered dead and wiped, its mappings and layers-index names removed, its listed backup deleted |
//! | held `SessionBackupUnreadable` (undecodable, or absent at `T_safe`) of a session whose record survives | its session backup is rewritten from the store (PD-8) |
//! | `CheckpointMappingToDeletedObject` advisory | its session backup is rewritten |
//! | a finalised `Uncovered` value of a written layer | removed from that layer in place |
//! | `UnreferencedBackup` advisory | reported only, never registered (PD-7) |
//! | anything whose read ended in a backup-store error: a session backup's read error, a held `Unknown` (the defer-47 object-level item included, PD-1), a finalised `Unknown` value — except a held item of a lost session or an unknown value on one of its objects, which the session's discard covers whatever its read answered | unconfirmed |
//!
//! # Unconfirmed items (PD-2)
//!
//! An error is not an absence. Under [`UnconfirmedPolicy::Refuse`] (the
//! default) one unconfirmed item makes the run discard nothing: the Job
//! recounts and exits 3. The one exception is an item of a lost session:
//! the session's discard (PD-6) removes its layers and objects whatever their
//! own reads answered, so a held item of a lost session, and an unknown
//! value on one of its objects, is not judged on its own and is never
//! unconfirmed. Under [`UnconfirmedPolicy::Discard`] (the second `--confirm
//! discard-unconfirmed-data`, option E) each unconfirmed item is treated as
//! the confirmed absence it stands in for, and is also listed, so each gets
//! its own WARN; the summaries count apart those it discards something for
//! and those it keeps ([`Unconfirmed::discards`]):
//!
//! - a session backup's read error: as undecodable (a lost session, or a
//!   rewrite when its record survives);
//! - a layer's `Unknown` lineage with no survivor: a lost layer; with a
//!   survivor: the survivor the writer kept stays, and nothing is discarded;
//! - the defer-47 object-level `Unknown` (a listed object whose session and
//!   layer could not be read): as an unreferenced backup — reported, its
//!   backups left, never registered (PD-7);
//! - an `Unknown` value: removed like an uncovered one.
//!
//! # The discard check ([`check_discardable`])
//!
//! Four rules, each a defect in the Job, never in the store; a defect stops
//! the run before any discard:
//!
//! 1. a lost layer, of every kind (the one only unmapped included), names
//!    only an object the plan writes nothing for (its action is `Nothing`,
//!    or it has no object plan);
//! 2. no `(session, layer)` the discards unmap, the over-limit layers'
//!    included, is one a session restore's Step B writes to another object
//!    (defer-51's accept-loss half);
//! 3. a value removal's object is one the writer restored, kept or cleared;
//! 4. a lost session is not restored and has no deletion marker.

use std::collections::{BTreeMap, BTreeSet};

use super::model::{
    Advisory, CoverageReport, ExtHash, HashVerdict, HeldItem, HeldReason, ObjectAction,
    RestorePlan, SessionAction, SessionReadFailure, SessionSeen, ValueSite,
};
use super::report::hash_prefix;

/// What the run does with unconfirmed items (PD-2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnconfirmedPolicy {
    /// The default: any unconfirmed item, and the run discards nothing.
    #[default]
    Refuse,
    /// The second confirm (`--confirm discard-unconfirmed-data`, option E):
    /// each is discarded as the absence it stands in for.
    Discard,
}

/// What a lost layer's discard rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LostLayerKind {
    /// The plan found no restorable backup and no survivor: registered dead,
    /// its header re-read before its wipe, unmapped.
    NoSurvivor,
    /// Its surviving mapping names an object on the dead-object set: the
    /// object is already registered; wiped and unmapped.
    MappedToDeleted,
    /// The plan found no survivor, and its listed backups include full
    /// backups over the per-delta entry limit, written by an earlier release,
    /// with nothing older usable (`HeldReason::BackupsOverLimit`): its header
    /// re-read like [`Self::NoSurvivor`]'s, then only unmapped. Never
    /// registered dead and never wiped, so its backups stay for a release
    /// whose limit fits them; reported with its own WARN.
    OverLimitBackups,
}

impl LostLayerKind {
    /// Whether the discard registers the object dead (and wipes it): every
    /// kind but [`Self::OverLimitBackups`], whose backups are kept.
    pub fn discards_object(self) -> bool {
        self != Self::OverLimitBackups
    }

    /// Whether the discard reads the object's header before it writes
    /// anything for it: every kind whose plan found no survivor. A mapping
    /// to a deleted object is not read (`execute::discard`, D0).
    pub fn rereads_header(self) -> bool {
        self != Self::MappedToDeleted
    }
}

/// One lost layer: a mapping `(session, layer) → object` the Job removes,
/// with its object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LostLayer {
    /// The session.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// The object its mapping names.
    pub object_id: u64,
    /// What the discard rests on.
    pub kind: LostLayerKind,
    /// Why it is lost (`HeldReason::describe`).
    pub why: String,
    /// The backup-store error it came from, when it is discarded as an
    /// unconfirmed item (option E).
    pub unconfirmed: Option<String>,
}

/// One lost session (PD-6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LostSession {
    /// The session.
    pub session_id: String,
    /// What names it.
    pub evidence: String,
    /// Its surviving objects: its mappings' and the headers that name it.
    pub objects: Vec<u64>,
    /// Its surviving `(layer, object)` mappings.
    pub mappings: Vec<(String, u64)>,
    /// Every layer name it carries: its mappings', its headers' and its
    /// layers index's.
    pub layer_names: Vec<String>,
    /// Whether the backup store lists a session backup of it, which the
    /// discard deletes.
    pub backup_listed: bool,
    /// The backup-store error it came from, when it is discarded as an
    /// unconfirmed item (option E).
    pub unconfirmed: Option<String>,
}

/// One layer's value removal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueRemoval {
    /// The object.
    pub object_id: u64,
    /// The session.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// The values removed from it.
    pub hashes: BTreeSet<ExtHash>,
    /// Of those, the ones discarded as unconfirmed (option E).
    pub unconfirmed: BTreeSet<ExtHash>,
    /// Where the values sit (the coverage report's sites).
    pub sites: Vec<ValueSite>,
}

/// One item the run could not confirm lost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unconfirmed {
    /// The item, named.
    pub item: String,
    /// The backup-store error it came from.
    pub error: String,
    /// What the run does with it under option E.
    pub action: String,
    /// Whether option E discards something for it — a layer, a session, a
    /// value or a session backup's rewrite — or keeps it as its WARN says
    /// (the defer-47 object, a layer whose survivor stays). The summaries
    /// count the two apart.
    pub discards: bool,
}

/// What accept-loss discards ([`lost_set`]). Not persisted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LostSet {
    /// The policy the set was built under.
    pub policy: UnconfirmedPolicy,
    /// The lost layers, by `(session, layer)`.
    pub layers: Vec<LostLayer>,
    /// The lost sessions, by session.
    pub sessions: Vec<LostSession>,
    /// The value removals, by object.
    pub values: Vec<ValueRemoval>,
    /// Sessions whose backup is rewritten whatever the unmappings do.
    pub rewrites: BTreeSet<String>,
    /// The items the run could not confirm lost. Under
    /// [`UnconfirmedPolicy::Refuse`] none of them is in the sets above, and
    /// one of them makes the run discard nothing; under
    /// [`UnconfirmedPolicy::Discard`] each one that [`Unconfirmed::discards`]
    /// is also in them as what it is discarded as.
    pub unconfirmed: Vec<Unconfirmed>,
    /// Held items accept-loss discards nothing for: none, since accept-loss
    /// steps down past an over-limit base (PD-4), but any there is keeps the
    /// run held after its discards.
    pub undiscardable: Vec<HeldItem>,
}

impl LostSet {
    /// PD-2's gate: whether the run discards nothing because an item could
    /// not be confirmed lost and the run was not given the second confirm.
    /// All or nothing: then not even the proven-lost items are discarded.
    pub fn refuses(&self) -> bool {
        self.policy == UnconfirmedPolicy::Refuse && !self.unconfirmed.is_empty()
    }

    /// Whether there is nothing to discard and nothing unconfirmed.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
            && self.sessions.is_empty()
            && self.values.is_empty()
            && self.rewrites.is_empty()
            && self.unconfirmed.is_empty()
            && self.undiscardable.is_empty()
    }

    /// The distinct values removed, over every layer.
    pub fn value_count(&self) -> usize {
        self.values
            .iter()
            .flat_map(|v| v.hashes.iter())
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Under option E, the unconfirmed items discarded and the ones kept as
    /// their WARN says ([`Unconfirmed::discards`]); `(0, 0)` without it.
    pub fn unconfirmed_counts(&self) -> (usize, usize) {
        if self.policy != UnconfirmedPolicy::Discard {
            return (0, 0);
        }
        let discarded = self.unconfirmed.iter().filter(|u| u.discards).count();
        (discarded, self.unconfirmed.len() - discarded)
    }
}

/// The lost set of `plan` after its writes, with `report` its finalised
/// coverage report (module docs). Reads nothing else.
pub fn lost_set(plan: &RestorePlan, report: &CoverageReport, policy: UnconfirmedPolicy) -> LostSet {
    let discard_unconfirmed = policy == UnconfirmedPolicy::Discard;
    let mut lost = LostSet {
        policy,
        ..LostSet::default()
    };
    let action_of = |object_id: u64| -> Option<&ObjectAction> {
        plan.objects
            .iter()
            .find(|o| o.object_id == object_id)
            .map(|o| &o.action)
    };
    let session_action = |session_id: &str| -> Option<&SessionAction> {
        plan.sessions
            .iter()
            .find(|s| s.session_id == session_id)
            .map(|s| &s.action)
    };
    let backup_failure = |session_id: &str| -> Option<(SessionReadFailure, String)> {
        plan.held.iter().find_map(|item| match &item.reason {
            HeldReason::SessionBackupUnreadable {
                session_id: s,
                error,
                cause,
            } if s == session_id => Some((*cause, error.clone())),
            _ => None,
        })
    };

    // Lost sessions: no record, no marker, not restored.
    for (session_id, seen) in &plan.sessions_seen {
        if seen.record || seen.deletion_marker {
            continue;
        }
        if matches!(
            session_action(session_id),
            Some(SessionAction::Restore { .. } | SessionAction::CompleteAbandonedDelete { .. })
        ) {
            continue;
        }
        let (backup_listed, unconfirmed) = match backup_failure(session_id) {
            None => (false, None),
            Some((SessionReadFailure::Undecodable | SessionReadFailure::AbsentAtTSafe, _)) => {
                (true, None)
            }
            Some((SessionReadFailure::ReadError, error)) => {
                lost.unconfirmed.push(Unconfirmed {
                    item: format!("session {session_id}"),
                    error: error.clone(),
                    action: "discards the session, which has no record in the store".to_owned(),
                    discards: true,
                });
                if !discard_unconfirmed {
                    continue;
                }
                (true, Some(error))
            }
        };
        lost.sessions.push(lost_session(
            plan,
            session_id,
            seen,
            backup_listed,
            unconfirmed,
        ));
    }
    let lost_sessions: BTreeSet<&str> = lost
        .sessions
        .iter()
        .map(|s| s.session_id.as_str())
        .collect();
    let lost_session_objects: BTreeSet<u64> = lost
        .sessions
        .iter()
        .flat_map(|s| s.objects.iter().copied())
        .collect();

    // Held items.
    for item in &plan.held {
        if lost_sessions.contains(item.session_id.as_str()) {
            // Its session's discard covers it.
            continue;
        }
        let layer_level = !item.session_id.is_empty() && !item.layer_name.is_empty();
        match (&item.reason, item.object_id) {
            (HeldReason::SessionBackupUnreadable { cause, error, .. }, _) => {
                let record = plan
                    .sessions_seen
                    .get(&item.session_id)
                    .is_some_and(|seen| seen.record);
                if !record {
                    // A session with no record is judged above: lost, or
                    // restored, or deleted.
                    continue;
                }
                match cause {
                    SessionReadFailure::Undecodable | SessionReadFailure::AbsentAtTSafe => {
                        lost.rewrites.insert(item.session_id.clone());
                    }
                    SessionReadFailure::ReadError => {
                        lost.unconfirmed.push(Unconfirmed {
                            item: format!("the session backup of session {}", item.session_id),
                            error: error.clone(),
                            action: "rewrites it from the store".to_owned(),
                            discards: true,
                        });
                        if discard_unconfirmed {
                            lost.rewrites.insert(item.session_id.clone());
                        }
                    }
                }
            }
            (
                HeldReason::NoBackup
                | HeldReason::BackupsUnusable { .. }
                | HeldReason::NoBackupStore,
                Some(object_id),
            ) if layer_level && writes_nothing(action_of(object_id)) => {
                lost.layers.push(LostLayer {
                    session_id: item.session_id.clone(),
                    layer_name: item.layer_name.clone(),
                    object_id,
                    kind: LostLayerKind::NoSurvivor,
                    why: item.reason.describe(),
                    unconfirmed: None,
                });
            }
            (HeldReason::BackupsOverLimit { .. }, Some(object_id))
                if layer_level && writes_nothing(action_of(object_id)) =>
            {
                // Unmapped only: its over-limit backups stay for a release
                // whose limit fits them (never registered, never wiped).
                lost.layers.push(LostLayer {
                    session_id: item.session_id.clone(),
                    layer_name: item.layer_name.clone(),
                    object_id,
                    kind: LostLayerKind::OverLimitBackups,
                    why: item.reason.describe(),
                    unconfirmed: None,
                });
            }
            (HeldReason::MappedToDeletedObject { .. }, Some(object_id)) if layer_level => {
                lost.layers.push(LostLayer {
                    session_id: item.session_id.clone(),
                    layer_name: item.layer_name.clone(),
                    object_id,
                    kind: LostLayerKind::MappedToDeleted,
                    why: item.reason.describe(),
                    unconfirmed: None,
                });
            }
            (HeldReason::Unknown { error }, Some(object_id)) if layer_level => {
                let survivor = !writes_nothing(action_of(object_id));
                let layer = format!(
                    "layer {} of session {} (object {object_id})",
                    item.layer_name, item.session_id
                );
                lost.unconfirmed.push(Unconfirmed {
                    item: if survivor {
                        format!("the newer backups of {layer}")
                    } else {
                        layer
                    },
                    error: error.clone(),
                    action: if survivor {
                        "keeps its surviving copy".to_owned()
                    } else {
                        "discards the layer".to_owned()
                    },
                    discards: !survivor,
                });
                if discard_unconfirmed && !survivor {
                    lost.layers.push(LostLayer {
                        session_id: item.session_id.clone(),
                        layer_name: item.layer_name.clone(),
                        object_id,
                        kind: LostLayerKind::NoSurvivor,
                        why: item.reason.describe(),
                        unconfirmed: Some(error.clone()),
                    });
                }
            }
            (HeldReason::Unknown { error }, Some(object_id))
                if item.session_id.is_empty() && item.layer_name.is_empty() =>
            {
                // PD-1 (defer-47): a listed object no read could attribute.
                lost.unconfirmed.push(Unconfirmed {
                    item: format!("object {object_id}, listed in the backup store"),
                    error: error.clone(),
                    action: "leaves its backups unreferenced".to_owned(),
                    discards: false,
                });
            }
            _ => lost.undiscardable.push(item.clone()),
        }
    }
    lost.layers.sort_by(|a, b| {
        (a.session_id.as_str(), a.layer_name.as_str())
            .cmp(&(b.session_id.as_str(), b.layer_name.as_str()))
    });

    // Rewrites the advisories ask for.
    for advisory in &plan.advisories {
        if let Advisory::CheckpointMappingToDeletedObject { session_id, .. } = advisory {
            if !lost_sessions.contains(session_id.as_str()) {
                lost.rewrites.insert(session_id.clone());
            }
        }
    }

    // Value removals, on every written layer the discards keep.
    let discarded: BTreeSet<u64> = lost
        .layers
        .iter()
        .map(|l| l.object_id)
        .chain(lost_session_objects.iter().copied())
        .collect();
    let mut unknown_values: BTreeMap<ExtHash, (String, usize)> = BTreeMap::new();
    for layer in &report.layers {
        if discarded.contains(&layer.object_id) {
            continue;
        }
        let mut hashes: BTreeSet<ExtHash> = layer.uncovered.clone();
        let mut sites: Vec<ValueSite> = layer.uncovered_sites.clone();
        let mut unconfirmed: BTreeSet<ExtHash> = BTreeSet::new();
        for hash in &layer.unknown {
            let error = match report.verdicts.get(hash) {
                Some(HashVerdict::Unknown(error)) => error.clone(),
                _ => "the backup store's answer is unknown".to_owned(),
            };
            unknown_values.entry(*hash).or_insert((error, 0)).1 += 1;
            if discard_unconfirmed {
                hashes.insert(*hash);
                unconfirmed.insert(*hash);
            }
        }
        if discard_unconfirmed {
            sites.extend(layer.unknown_sites.iter().cloned());
        }
        if hashes.is_empty() {
            continue;
        }
        lost.values.push(ValueRemoval {
            object_id: layer.object_id,
            session_id: layer.session_id.clone(),
            layer_name: layer.layer_name.clone(),
            hashes,
            unconfirmed,
            sites,
        });
    }
    for (hash, (error, layers)) in unknown_values {
        lost.unconfirmed.push(Unconfirmed {
            item: format!(
                "large value {} (referenced by {layers} layers)",
                hash_prefix(&hash)
            ),
            error,
            action: "removes it from those layers".to_owned(),
            discards: true,
        });
    }
    lost.values.sort_by_key(|v| v.object_id);
    lost
}

/// Whether the plan writes nothing for an object of action `action`: it is
/// held with no survivor (`Nothing`), or it has no object plan.
fn writes_nothing(action: Option<&ObjectAction>) -> bool {
    matches!(action, None | Some(ObjectAction::Nothing))
}

/// One lost session's discard, from what the plan saw of it.
fn lost_session(
    plan: &RestorePlan,
    session_id: &str,
    seen: &SessionSeen,
    backup_listed: bool,
    unconfirmed: Option<String>,
) -> LostSession {
    let evidence = plan
        .advisories
        .iter()
        .find_map(|advisory| match advisory {
            Advisory::SessionWithoutBackup {
                session_id: s,
                evidence,
                ..
            } if s == session_id => Some(evidence.clone()),
            _ => None,
        })
        .or_else(|| {
            seen.surviving_mappings.first().map(|(layer, object)| {
                format!("the surviving mapping of layer {layer} (object {object})")
            })
        })
        .or_else(|| {
            seen.header_objects.first().map(|(object, layer)| {
                format!("the surviving header of object {object} (layer {layer})")
            })
        })
        .unwrap_or_else(|| "its listed session backup".to_owned());
    let mut objects: BTreeSet<u64> = seen.surviving_mappings.iter().map(|(_, o)| *o).collect();
    objects.extend(seen.header_objects.iter().map(|(o, _)| *o));
    let mut layer_names: BTreeSet<String> = seen
        .surviving_mappings
        .iter()
        .map(|(layer, _)| layer.clone())
        .collect();
    layer_names.extend(
        seen.header_objects
            .iter()
            .map(|(_, layer)| layer.clone())
            .filter(|layer| !layer.is_empty()),
    );
    layer_names.extend(seen.layers_index.iter().cloned());
    LostSession {
        session_id: session_id.to_owned(),
        evidence,
        objects: objects.into_iter().collect(),
        mappings: seen.surviving_mappings.clone(),
        layer_names: layer_names.into_iter().collect(),
        backup_listed,
        unconfirmed,
    }
}

/// A discard set the Job must not carry out: a defect in the Job, not in
/// the store ([`check_discardable`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscardDefect {
    /// Rule 1: a lost layer names an object the plan writes.
    LostLayerWritten {
        /// The object.
        object_id: u64,
    },
    /// Rule 2: an unmapped `(session, layer)` is one a session restore's
    /// Step B writes to another object.
    UnmapWrittenByStepB {
        /// The session.
        session_id: String,
        /// The layer.
        layer_name: String,
        /// The object Step B writes.
        written: u64,
    },
    /// Rule 3: a value removal's object is not one the writer restored,
    /// kept or cleared.
    ValueRemovalNotWritten {
        /// The object.
        object_id: u64,
    },
    /// Rule 4: a lost session is restored or has a deletion marker.
    LostSessionNotLost {
        /// The session.
        session_id: String,
    },
}

impl std::fmt::Display for DiscardDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LostLayerWritten { object_id } => write!(
                f,
                "object {object_id} is discarded as lost, but the plan writes it"
            ),
            Self::UnmapWrittenByStepB {
                session_id,
                layer_name,
                written,
            } => write!(
                f,
                "layer {layer_name} of session {session_id} is unmapped, but the session's \
                 restore maps it to object {written}"
            ),
            Self::ValueRemovalNotWritten { object_id } => write!(
                f,
                "values are removed from object {object_id}, which the plan does not restore, \
                 keep or clear"
            ),
            Self::LostSessionNotLost { session_id } => write!(
                f,
                "session {session_id} is discarded as lost, but it is restored or was deleted"
            ),
        }
    }
}

/// The discard check of `lost` against `plan` (module docs, *The discard
/// check*), pure. Empty when every discard is consistent with the plan.
pub fn check_discardable(plan: &RestorePlan, lost: &LostSet) -> Vec<DiscardDefect> {
    let mut defects = Vec::new();
    let action_of = |object_id: u64| {
        plan.objects
            .iter()
            .find(|o| o.object_id == object_id)
            .map(|o| &o.action)
    };

    // Rule 1.
    for layer in &lost.layers {
        if !writes_nothing(action_of(layer.object_id)) {
            defects.push(DiscardDefect::LostLayerWritten {
                object_id: layer.object_id,
            });
        }
    }

    // Rule 2.
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
    let unmapped = lost
        .layers
        .iter()
        .map(|l| {
            (
                l.session_id.as_str(),
                l.layer_name.as_str(),
                Some(l.object_id),
            )
        })
        .chain(lost.sessions.iter().flat_map(|s| {
            s.layer_names.iter().map(move |layer| {
                let expected = s
                    .mappings
                    .iter()
                    .find(|(mapped, _)| mapped == layer)
                    .map(|(_, object)| *object);
                (s.session_id.as_str(), layer.as_str(), expected)
            })
        }));
    for (session_id, layer_name, expected) in unmapped {
        if let Some(written) = step_b_writes.get(&(session_id, layer_name)) {
            if Some(*written) != expected {
                defects.push(DiscardDefect::UnmapWrittenByStepB {
                    session_id: session_id.to_owned(),
                    layer_name: layer_name.to_owned(),
                    written: *written,
                });
            }
        }
    }

    // Rule 3.
    for removal in &lost.values {
        if !matches!(
            action_of(removal.object_id),
            Some(
                ObjectAction::Keep | ObjectAction::Restore { .. } | ObjectAction::ClearFlags { .. }
            )
        ) {
            defects.push(DiscardDefect::ValueRemovalNotWritten {
                object_id: removal.object_id,
            });
        }
    }

    // Rule 4.
    for session in &lost.sessions {
        let restored = plan.sessions.iter().any(|s| {
            s.session_id == session.session_id
                && matches!(
                    s.action,
                    SessionAction::Restore { .. } | SessionAction::CompleteAbandonedDelete { .. }
                )
        });
        let marked = plan
            .sessions_seen
            .get(&session.session_id)
            .is_some_and(|seen| seen.deletion_marker);
        if restored || marked {
            defects.push(DiscardDefect::LostSessionNotLost {
                session_id: session.session_id.clone(),
            });
        }
    }
    defects
}

#[cfg(test)]
mod tests {
    //! The lost set and the discard check over hand-built plans and finalised
    //! reports, pure as the functions are: LS1 (the module docs' table, row
    //! by row), LS2 (PD-2's all-or-nothing gate), LS3 (the four rules, rule
    //! 2 being P51-b), P55-b (nothing held, nothing lost; an unreferenced
    //! backup never registered), PD-1 (the unattributed object) and option
    //! E's rulings (an unattributed object reported, never registered; a
    //! layer whose newer backups are unknown keeps its survivor). What the
    //! discards then write against a real store is the accept-loss
    //! integration tests' (`tests/integration/dr/accept_loss_test.rs`).

    use super::super::model::{
        BandInputs, CoveragePlan, LayerCoverageReport, LineageFootprint, ObjectPlan, PlanEpoch,
        PlannedLineage, RestoreReason, SessionPlan,
    };
    use super::super::report::DiscardPlanCounts;
    use super::*;
    use crate::checkpoint::session_payload::SessionCheckpoint;
    use crate::checkpoint::store::{CheckpointHandle, CheckpointMeta};

    /// The session most items are in.
    const S: &str = "s-lost";
    const H1: ExtHash = [0x11; 32];
    const H2: ExtHash = [0x22; 32];
    const H3: ExtHash = [0x33; 32];
    /// What a backup-store read that ended in an error says.
    const ERROR: &str = "the backup store answered 503";

    /// A plan with nothing in it.
    fn plan() -> RestorePlan {
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(1),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: Vec::new(),
            objects: Vec::new(),
            coverage: CoveragePlan::default(),
            held: Vec::new(),
            advisories: Vec::new(),
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: BTreeMap::new(),
        }
    }

    fn held(
        session_id: &str,
        layer_name: &str,
        object_id: Option<u64>,
        reason: HeldReason,
    ) -> HeldItem {
        HeldItem {
            session_id: session_id.to_owned(),
            layer_name: layer_name.to_owned(),
            object_id,
            reason,
        }
    }

    /// A held item on layer `layer_name` of [`S`], object `object_id`.
    fn layer_held(layer_name: &str, object_id: u64, reason: HeldReason) -> HeldItem {
        held(S, layer_name, Some(object_id), reason)
    }

    /// `session_id`'s session backup held unreadable for `cause`.
    fn unreadable(session_id: &str, cause: SessionReadFailure) -> HeldItem {
        held(
            session_id,
            "",
            None,
            HeldReason::SessionBackupUnreadable {
                session_id: session_id.to_owned(),
                error: ERROR.to_owned(),
                cause,
            },
        )
    }

    /// Object `object_id`'s plan entry, layer `layer_name` of [`S`].
    fn object(object_id: u64, layer_name: &str, action: ObjectAction) -> ObjectPlan {
        ObjectPlan {
            object_id,
            session_id: S.to_owned(),
            layer_name: layer_name.to_owned(),
            action,
        }
    }

    fn band() -> BandInputs {
        BandInputs {
            floor_seq: 10,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
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

    /// A session backup of `session_id` mapping each `(layer, object)`.
    fn backup_of(session_id: &str, mappings: &[(&str, u64)]) -> SessionCheckpoint {
        SessionCheckpoint {
            schema_version: 1,
            session_id: session_id.to_owned(),
            name: format!("{session_id}-name"),
            base_path: format!("stage:/{session_id}.usd"),
            created_by: "user".to_owned(),
            created_at: 1,
            layer_mappings: mappings
                .iter()
                .map(|(layer, object)| ((*layer).to_owned(), *object))
                .collect(),
            layers_index: mappings
                .iter()
                .map(|(layer, _)| (*layer).to_owned())
                .collect(),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session_id.to_owned(),
        }
    }

    /// A session restore of `session_id` writing `mappings` (Step B when
    /// `step_b`).
    fn session_restore(session_id: &str, mappings: &[(&str, u64)], step_b: bool) -> SessionPlan {
        SessionPlan {
            session_id: session_id.to_owned(),
            action: SessionAction::Restore {
                checkpoint: backup_of(session_id, mappings),
                step_b,
            },
        }
    }

    /// What the store held of a session with neither record nor marker.
    fn seen_lost(
        mappings: &[(&str, u64)],
        headers: &[(u64, &str)],
        layers_index: &[&str],
    ) -> SessionSeen {
        SessionSeen {
            record: false,
            deletion_marker: false,
            surviving_mappings: mappings
                .iter()
                .map(|(layer, object)| ((*layer).to_owned(), *object))
                .collect(),
            header_objects: headers
                .iter()
                .map(|(object, layer)| (*object, (*layer).to_owned()))
                .collect(),
            layers_index: layers_index.iter().map(|l| (*l).to_owned()).collect(),
        }
    }

    /// What the store held of a session whose record survives.
    fn seen_live() -> SessionSeen {
        SessionSeen {
            record: true,
            ..SessionSeen::default()
        }
    }

    fn site(hash: ExtHash, attribute: &str) -> ValueSite {
        ValueSite {
            prim_path: "/World".to_owned(),
            attribute: attribute.to_owned(),
            hash,
            count: 1,
        }
    }

    /// Layer `l{object_id}.usda` of [`S`]'s coverage report: `uncovered` and
    /// `unknown`, one site each.
    fn report_layer(
        object_id: u64,
        uncovered: &[ExtHash],
        unknown: &[ExtHash],
    ) -> LayerCoverageReport {
        LayerCoverageReport {
            object_id,
            session_id: S.to_owned(),
            layer_name: format!("l{object_id}.usda"),
            uncovered: uncovered.iter().copied().collect(),
            unknown: unknown.iter().copied().collect(),
            uncovered_sites: uncovered.iter().map(|h| site(*h, "uncovered")).collect(),
            unknown_sites: unknown.iter().map(|h| site(*h, "unknown")).collect(),
            naming_error: None,
        }
    }

    /// A finalised report of `layers`, every value in them judged: the
    /// uncovered ones `Uncovered`, the unknown ones `Unknown(ERROR)`.
    fn report(layers: Vec<LayerCoverageReport>) -> CoverageReport {
        let mut verdicts = BTreeMap::new();
        for layer in &layers {
            for hash in &layer.uncovered {
                verdicts.insert(*hash, HashVerdict::Uncovered);
            }
            for hash in &layer.unknown {
                verdicts.insert(*hash, HashVerdict::Unknown(ERROR.to_owned()));
            }
        }
        CoverageReport { verdicts, layers }
    }

    fn lost_layer(
        layer_name: &str,
        object_id: u64,
        kind: LostLayerKind,
        why: &HeldReason,
    ) -> LostLayer {
        LostLayer {
            session_id: S.to_owned(),
            layer_name: layer_name.to_owned(),
            object_id,
            kind,
            why: why.describe(),
            unconfirmed: None,
        }
    }

    /// Whether `lost` discards, rewrites or registers anything at all.
    fn discards_nothing(lost: &LostSet) -> bool {
        lost.layers.is_empty()
            && lost.sessions.is_empty()
            && lost.values.is_empty()
            && lost.rewrites.is_empty()
    }

    // ── LS1: the table, row by row ──────────────────────────────────────

    /// LS1, row 1. A held `NoBackup`, `BackupsUnusable` or `NoBackupStore`
    /// on a layer whose object the plan writes nothing for (`Nothing`, or no
    /// object plan) is a lost layer with no survivor, its `why` the held
    /// reason's description. The same item on an object the plan keeps is
    /// not a lost layer: it stays held.
    #[test]
    fn ls1_a_held_layer_with_no_backup_and_no_survivor_is_a_lost_layer() {
        let unusable = HeldReason::BackupsUnusable {
            exhaustion: "every listed backup is broken".to_owned(),
        };
        let mut plan = plan();
        plan.objects = vec![
            object(1, "a.usda", ObjectAction::Nothing),
            object(2, "b.usda", ObjectAction::Nothing),
            object(4, "d.usda", ObjectAction::Keep),
        ];
        plan.held = vec![
            layer_held("a.usda", 1, HeldReason::NoBackup),
            layer_held("b.usda", 2, unusable.clone()),
            layer_held("c.usda", 3, HeldReason::NoBackupStore),
            layer_held("d.usda", 4, HeldReason::NoBackup),
        ];

        let lost = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);

        assert_eq!(
            lost.layers,
            vec![
                lost_layer(
                    "a.usda",
                    1,
                    LostLayerKind::NoSurvivor,
                    &HeldReason::NoBackup
                ),
                lost_layer("b.usda", 2, LostLayerKind::NoSurvivor, &unusable),
                lost_layer(
                    "c.usda",
                    3,
                    LostLayerKind::NoSurvivor,
                    &HeldReason::NoBackupStore
                ),
            ]
        );
        assert_eq!(
            lost.undiscardable,
            vec![layer_held("d.usda", 4, HeldReason::NoBackup)],
            "a layer the plan keeps is never lost"
        );
        assert!(lost.sessions.is_empty() && lost.values.is_empty() && lost.unconfirmed.is_empty());
        assert!(!lost.refuses());
        assert_eq!(check_discardable(&plan, &lost), Vec::new());
    }

    /// LS1, row 2. A held `MappedToDeletedObject` is a lost layer whose
    /// object is already on the dead-object set: wiped and unmapped, with no
    /// header re-read.
    #[test]
    fn ls1_a_held_mapping_to_a_deleted_object_is_a_lost_layer_already_registered() {
        let reason = HeldReason::MappedToDeletedObject {
            session_id: S.to_owned(),
            layer_name: "d.usda".to_owned(),
        };
        let mut plan = plan();
        plan.held = vec![layer_held("d.usda", 5, reason.clone())];

        let lost = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);

        assert_eq!(
            lost.layers,
            vec![lost_layer(
                "d.usda",
                5,
                LostLayerKind::MappedToDeleted,
                &reason
            )]
        );
        assert_eq!(check_discardable(&plan, &lost), Vec::new());
    }

    /// LS1, row 3 (PD-6). A session with no record, no deletion marker and
    /// no restore is lost: one named by evidence with no listed backup, one
    /// whose listed backup is undecodable and one whose listed backup read
    /// absent at `T_safe`. Each names its surviving objects (mappings and
    /// headers), every layer name it carries, its evidence, and whether a
    /// backup is listed (to delete). A held item of a lost session is
    /// covered by the session's discard, not a lost layer of its own. A
    /// session with a record, a deletion marker, a restore or an abandoned
    /// delete is not lost.
    #[test]
    fn ls1_a_session_with_no_record_no_marker_and_no_restore_is_a_lost_session() {
        let mut plan = plan();
        plan.sessions_seen = BTreeMap::from([
            (
                "s-gone".to_owned(),
                seen_lost(
                    &[("a.usda", 11)],
                    &[(11, "a.usda"), (12, "b.usda")],
                    &["a.usda", "c.usda"],
                ),
            ),
            (
                "s-undecodable".to_owned(),
                seen_lost(&[("u.usda", 21)], &[], &[]),
            ),
            ("s-absent".to_owned(), seen_lost(&[], &[], &[])),
            (
                "s-marked".to_owned(),
                SessionSeen {
                    deletion_marker: true,
                    ..seen_lost(&[("m.usda", 31)], &[], &[])
                },
            ),
            ("s-live".to_owned(), seen_live()),
            ("s-restored".to_owned(), seen_lost(&[], &[], &[])),
            (
                "s-deleting".to_owned(),
                seen_lost(&[("x.usda", 41)], &[], &[]),
            ),
        ]);
        plan.sessions = vec![
            session_restore("s-restored", &[("r.usda", 51)], true),
            SessionPlan {
                session_id: "s-deleting".to_owned(),
                action: SessionAction::CompleteAbandonedDelete {
                    objects: vec![41],
                    layer_names: vec!["x.usda".to_owned()],
                    base_path: None,
                    mappings: vec![("x.usda".to_owned(), 41)],
                    path_name_members: Vec::new(),
                },
            },
        ];
        plan.advisories = vec![Advisory::SessionWithoutBackup {
            session_id: "s-gone".to_owned(),
            evidence: "the header of object 12".to_owned(),
            layers_with_backups: Vec::new(),
        }];
        plan.held = vec![
            held("s-gone", "a.usda", Some(11), HeldReason::NoBackup),
            unreadable("s-undecodable", SessionReadFailure::Undecodable),
            unreadable("s-absent", SessionReadFailure::AbsentAtTSafe),
        ];

        let lost = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);

        assert_eq!(
            lost.sessions,
            vec![
                LostSession {
                    session_id: "s-absent".to_owned(),
                    evidence: "its listed session backup".to_owned(),
                    objects: Vec::new(),
                    mappings: Vec::new(),
                    layer_names: Vec::new(),
                    backup_listed: true,
                    unconfirmed: None,
                },
                LostSession {
                    session_id: "s-gone".to_owned(),
                    evidence: "the header of object 12".to_owned(),
                    objects: vec![11, 12],
                    mappings: vec![("a.usda".to_owned(), 11)],
                    layer_names: vec![
                        "a.usda".to_owned(),
                        "b.usda".to_owned(),
                        "c.usda".to_owned()
                    ],
                    backup_listed: false,
                    unconfirmed: None,
                },
                LostSession {
                    session_id: "s-undecodable".to_owned(),
                    evidence: "the surviving mapping of layer u.usda (object 21)".to_owned(),
                    objects: vec![21],
                    mappings: vec![("u.usda".to_owned(), 21)],
                    layer_names: vec!["u.usda".to_owned()],
                    backup_listed: true,
                    unconfirmed: None,
                },
            ]
        );
        assert!(
            lost.layers.is_empty(),
            "the lost session's held layer is its discard's: {:?}",
            lost.layers
        );
        assert!(lost.rewrites.is_empty() && lost.unconfirmed.is_empty());
        assert_eq!(check_discardable(&plan, &lost), Vec::new());
    }

    /// LS1, rows 4 and 5 (PD-8). A session whose record survives but whose
    /// backup is undecodable or read absent at `T_safe` has its backup
    /// rewritten; so has one whose backup mapped a layer to a deleted object
    /// (the advisory), unless that session is lost, whose backup is deleted
    /// instead.
    #[test]
    fn ls1_a_live_sessions_unusable_backup_and_one_naming_a_deleted_object_are_rewritten() {
        let mut plan = plan();
        plan.sessions_seen = BTreeMap::from([
            ("s-a".to_owned(), seen_live()),
            ("s-b".to_owned(), seen_live()),
            ("s-c".to_owned(), seen_live()),
            ("s-gone".to_owned(), seen_lost(&[], &[], &[])),
        ]);
        plan.held = vec![
            unreadable("s-a", SessionReadFailure::Undecodable),
            unreadable("s-b", SessionReadFailure::AbsentAtTSafe),
        ];
        plan.advisories = vec![
            Advisory::CheckpointMappingToDeletedObject {
                session_id: "s-c".to_owned(),
                layer_name: "c.usda".to_owned(),
                object_id: 61,
            },
            Advisory::CheckpointMappingToDeletedObject {
                session_id: "s-gone".to_owned(),
                layer_name: "g.usda".to_owned(),
                object_id: 62,
            },
        ];

        let lost = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);

        assert_eq!(
            lost.rewrites,
            BTreeSet::from(["s-a".to_owned(), "s-b".to_owned(), "s-c".to_owned()])
        );
        assert_eq!(
            lost.sessions
                .iter()
                .map(|s| s.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s-gone"]
        );
        assert!(lost.layers.is_empty() && lost.unconfirmed.is_empty());
    }

    /// LS1, row 6. A finalised `Uncovered` value is removed from every
    /// layer the writer restored, kept or cleared that references it, with
    /// the report's sites; never from a layer the discards remove whole (a
    /// lost layer's, or a lost session's object).
    #[test]
    fn ls1_uncovered_values_are_removed_from_each_layer_the_discards_keep() {
        let mut plan = plan();
        plan.objects = vec![
            object(31, "l31.usda", ObjectAction::Keep),
            object(32, "l32.usda", restore(32)),
            object(33, "l33.usda", ObjectAction::ClearFlags { band: band() }),
            object(34, "l34.usda", ObjectAction::Nothing),
        ];
        plan.held = vec![layer_held("l34.usda", 34, HeldReason::NoBackup)];
        plan.sessions_seen =
            BTreeMap::from([("s-gone".to_owned(), seen_lost(&[("g.usda", 35)], &[], &[]))]);
        let report = report(vec![
            report_layer(31, &[H1], &[]),
            report_layer(32, &[H1, H2], &[]),
            report_layer(33, &[H3], &[]),
            report_layer(34, &[H2], &[]),
            report_layer(35, &[H3], &[]),
        ]);

        let lost = lost_set(&plan, &report, UnconfirmedPolicy::Refuse);

        let removal = |object_id: u64, hashes: &[ExtHash]| ValueRemoval {
            object_id,
            session_id: S.to_owned(),
            layer_name: format!("l{object_id}.usda"),
            hashes: hashes.iter().copied().collect(),
            unconfirmed: BTreeSet::new(),
            sites: hashes.iter().map(|h| site(*h, "uncovered")).collect(),
        };
        assert_eq!(
            lost.values,
            vec![
                removal(31, &[H1]),
                removal(32, &[H1, H2]),
                removal(33, &[H3])
            ]
        );
        assert_eq!(lost.value_count(), 3, "distinct values over every layer");
        assert!(!lost.refuses());
        assert_eq!(check_discardable(&plan, &lost), Vec::new());
    }

    // ── P55-b, PD-7: nothing held, nothing lost ─────────────────────────

    /// P55-b (fail-on-old). A plan with no held item and no advisory, and a
    /// clean report, has an empty lost set under either policy. Advisories
    /// alone — an unreferenced backup (PD-7), an unreferenced object, a
    /// deleted mapping Step B replaces — discard, rewrite and register
    /// nothing: no layer, no session, and 0 registrations in the counts.
    #[test]
    fn p55b_a_plan_with_nothing_held_has_an_empty_lost_set_and_never_registers_an_unreferenced_backup(
    ) {
        for policy in [UnconfirmedPolicy::Refuse, UnconfirmedPolicy::Discard] {
            let lost = lost_set(&plan(), &CoverageReport::default(), policy);
            assert!(lost.is_empty(), "{policy:?}: {lost:?}");
            assert!(!lost.refuses(), "{policy:?}");

            let mut plan = plan();
            plan.advisories = vec![
                Advisory::UnreferencedBackup { object_id: 77 },
                Advisory::UnreferencedObject {
                    object_id: 78,
                    session_id: "s-x".to_owned(),
                    layer_name: "x.usda".to_owned(),
                },
                Advisory::DeletedMappingReplaced {
                    session_id: S.to_owned(),
                    layer_name: "y.usda".to_owned(),
                    deleted_object_id: 79,
                    replacement_object_id: 80,
                },
            ];
            let lost = lost_set(&plan, &CoverageReport::default(), policy);
            assert!(
                lost.is_empty(),
                "{policy:?}: advisories discard nothing: {lost:?}"
            );
            assert_eq!(
                DiscardPlanCounts::of(&lost).registered,
                0,
                "{policy:?}: nothing registered"
            );
        }
    }

    // ── PD-2: unconfirmed items, and option E ───────────────────────────

    /// The PD-2 fixture: a proven-lost layer (`l1.usda`, object 1), and
    /// one item of each unconfirmed kind — the backup of the live session
    /// `s-live` and of the record-less `s-err` failing to read, the layer
    /// `l2.usda` (object 2, no survivor) whose backups' read failed, and the
    /// value [`H2`] whose probe failed on the kept layer 3, beside [`H1`] in
    /// no store.
    fn unconfirmed_fixture() -> (RestorePlan, CoverageReport) {
        let mut plan = plan();
        plan.objects = vec![
            object(1, "l1.usda", ObjectAction::Nothing),
            object(2, "l2.usda", ObjectAction::Nothing),
            object(3, "l3.usda", ObjectAction::Keep),
        ];
        plan.sessions_seen = BTreeMap::from([
            ("s-live".to_owned(), seen_live()),
            ("s-err".to_owned(), seen_lost(&[("e.usda", 9)], &[], &[])),
        ]);
        plan.held = vec![
            layer_held("l1.usda", 1, HeldReason::NoBackup),
            layer_held(
                "l2.usda",
                2,
                HeldReason::Unknown {
                    error: ERROR.to_owned(),
                },
            ),
            unreadable("s-live", SessionReadFailure::ReadError),
            unreadable("s-err", SessionReadFailure::ReadError),
        ];
        (plan, report(vec![report_layer(3, &[H1], &[H2])]))
    }

    /// LS1, the unconfirmed rows, without the second confirm: a session
    /// backup's read error (record present or not), a layer's `Unknown`
    /// lineage and an `Unknown` value are each one unconfirmed item naming
    /// its error, and none of them is in the discard sets: no rewrite, no
    /// lost session, no lost layer, the unknown value not removed.
    #[test]
    fn ls1_an_error_is_unconfirmed_and_discards_nothing_of_its_own_without_the_second_confirm() {
        let (plan, report) = unconfirmed_fixture();

        let lost = lost_set(&plan, &report, UnconfirmedPolicy::Refuse);

        let items: Vec<(&str, &str)> = lost
            .unconfirmed
            .iter()
            .map(|u| (u.item.as_str(), u.error.as_str()))
            .collect();
        let value_item = format!("large value {} (referenced by 1 layers)", hash_prefix(&H2));
        assert_eq!(
            items,
            vec![
                ("session s-err", ERROR),
                ("layer l2.usda of session s-lost (object 2)", ERROR),
                ("the session backup of session s-live", ERROR),
                (value_item.as_str(), ERROR),
            ]
        );
        assert_eq!(
            lost.layers,
            vec![lost_layer(
                "l1.usda",
                1,
                LostLayerKind::NoSurvivor,
                &HeldReason::NoBackup
            )],
            "only the proven-lost layer"
        );
        assert!(lost.sessions.is_empty() && lost.rewrites.is_empty());
        assert_eq!(lost.values.len(), 1);
        assert_eq!(
            lost.values[0].hashes,
            BTreeSet::from([H1]),
            "H2 is not removed"
        );
        assert!(lost.values[0].unconfirmed.is_empty());
        assert!(lost.refuses());
    }

    /// Option E (A12b's pure half). Under the second confirm the same
    /// unconfirmed items are listed the same way, and each is also in the
    /// discard sets as the absence it stands in for, marked with its error:
    /// the live session's backup rewritten, the record-less session lost
    /// (its listed backup deleted), the layer with no survivor lost, the
    /// unknown value removed beside the uncovered one, its site named. The
    /// run is not refused, and the discard check passes.
    #[test]
    fn e_the_second_confirm_discards_each_unconfirmed_item_as_the_absence_it_stands_for() {
        let (plan, report) = unconfirmed_fixture();
        let refused = lost_set(&plan, &report, UnconfirmedPolicy::Refuse);

        let lost = lost_set(&plan, &report, UnconfirmedPolicy::Discard);

        assert_eq!(lost.unconfirmed, refused.unconfirmed, "listed the same way");
        assert!(!lost.refuses());
        assert_eq!(lost.rewrites, BTreeSet::from(["s-live".to_owned()]));
        assert_eq!(lost.sessions.len(), 1);
        assert_eq!(lost.sessions[0].session_id, "s-err");
        assert!(
            lost.sessions[0].backup_listed,
            "its listed backup is deleted"
        );
        assert_eq!(lost.sessions[0].unconfirmed.as_deref(), Some(ERROR));
        let unknown = HeldReason::Unknown {
            error: ERROR.to_owned(),
        };
        assert_eq!(
            lost.layers,
            vec![
                lost_layer(
                    "l1.usda",
                    1,
                    LostLayerKind::NoSurvivor,
                    &HeldReason::NoBackup
                ),
                LostLayer {
                    unconfirmed: Some(ERROR.to_owned()),
                    ..lost_layer("l2.usda", 2, LostLayerKind::NoSurvivor, &unknown)
                },
            ]
        );
        assert_eq!(lost.values.len(), 1);
        assert_eq!(lost.values[0].hashes, BTreeSet::from([H1, H2]));
        assert_eq!(lost.values[0].unconfirmed, BTreeSet::from([H2]));
        assert_eq!(
            lost.values[0].sites,
            vec![site(H1, "uncovered"), site(H2, "unknown")]
        );
        assert_eq!(check_discardable(&plan, &lost), Vec::new());
    }

    /// Option E, ruling 2. A layer whose surviving copy the plan keeps but
    /// whose newer backups could not be read is one unconfirmed item ("the
    /// newer backups of …", which keeps its surviving copy); under the
    /// second confirm nothing of it is discarded: the survivor stays mapped.
    #[test]
    fn e_a_layer_whose_newer_backups_are_unknown_keeps_its_survivor_under_the_second_confirm() {
        let mut plan = plan();
        plan.objects = vec![object(7, "k.usda", ObjectAction::Keep)];
        plan.held = vec![layer_held(
            "k.usda",
            7,
            HeldReason::Unknown {
                error: ERROR.to_owned(),
            },
        )];

        for policy in [UnconfirmedPolicy::Refuse, UnconfirmedPolicy::Discard] {
            let lost = lost_set(&plan, &CoverageReport::default(), policy);

            assert_eq!(
                lost.unconfirmed,
                vec![Unconfirmed {
                    item: "the newer backups of layer k.usda of session s-lost (object 7)"
                        .to_owned(),
                    error: ERROR.to_owned(),
                    action: "keeps its surviving copy".to_owned(),
                    discards: false,
                }],
                "{policy:?}"
            );
            assert!(
                discards_nothing(&lost),
                "{policy:?}: the survivor is kept: {lost:?}"
            );
            assert_eq!(check_discardable(&plan, &lost), Vec::new(), "{policy:?}");
        }
    }

    /// PD-1 (defer-47) and option E's ruling 1 (fail-on-old). A listed
    /// object no read could attribute (no session, no layer) is one
    /// unconfirmed item: without the second confirm the run refuses; with
    /// it the object is reported only — no lost layer, no lost session,
    /// nothing registered — so its backups are left, never reaped.
    #[test]
    fn pd1_the_unattributed_object_is_unconfirmed_and_never_registered_even_under_the_second_confirm(
    ) {
        let mut plan = plan();
        plan.held = vec![held(
            "",
            "",
            Some(91),
            HeldReason::Unknown {
                error: ERROR.to_owned(),
            },
        )];

        let refused = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);
        let discarded = lost_set(
            &plan,
            &CoverageReport::default(),
            UnconfirmedPolicy::Discard,
        );

        let item = Unconfirmed {
            item: "object 91, listed in the backup store".to_owned(),
            error: ERROR.to_owned(),
            action: "leaves its backups unreferenced".to_owned(),
            discards: false,
        };
        assert_eq!(refused.unconfirmed, vec![item.clone()]);
        assert!(refused.refuses());
        assert_eq!(discarded.unconfirmed, vec![item]);
        assert!(!discarded.refuses());
        for lost in [&refused, &discarded] {
            assert!(discards_nothing(lost), "{lost:?}");
            assert!(lost.undiscardable.is_empty());
            assert_eq!(
                DiscardPlanCounts::of(lost).registered,
                0,
                "never registered"
            );
        }
    }

    // ── LS2: all or nothing ─────────────────────────────────────────────

    /// LS2 (fail-on-old). One unconfirmed item refuses every discard without
    /// the second confirm, the proven-lost ones too: [`LostSet::refuses`] is
    /// true although the set holds a lost layer. With the second confirm, or
    /// with nothing unconfirmed, it is false.
    #[test]
    fn ls2_one_unconfirmed_item_refuses_every_discard_without_the_second_confirm() {
        let mut plan = plan();
        plan.held = vec![layer_held("l1.usda", 1, HeldReason::NoBackup)];
        plan.objects = vec![object(3, "l3.usda", ObjectAction::Keep)];
        let one_unknown = report(vec![report_layer(3, &[], &[H2])]);

        let refused = lost_set(&plan, &one_unknown, UnconfirmedPolicy::Refuse);
        assert_eq!(refused.unconfirmed.len(), 1);
        assert_eq!(refused.layers.len(), 1, "fixture: a proven-lost layer");
        assert!(refused.refuses(), "one unconfirmed item refuses them all");

        assert!(!lost_set(&plan, &one_unknown, UnconfirmedPolicy::Discard).refuses());
        let clean = lost_set(&plan, &CoverageReport::default(), UnconfirmedPolicy::Refuse);
        assert!(clean.unconfirmed.is_empty());
        assert!(!clean.refuses(), "nothing unconfirmed, nothing refused");
    }

    // ── LS3: the discard check's four rules ─────────────────────────────

    /// LS3, rule 1. A lost layer whose object the plan writes (keeps,
    /// restores, clears or moves) is a defect; one whose object the plan
    /// writes nothing for, or has no plan, is not.
    #[test]
    fn ls3_rule_1_a_lost_layer_whose_object_the_plan_writes_is_a_defect() {
        let mut plan = plan();
        plan.objects = vec![
            object(1, "l1.usda", ObjectAction::Keep),
            object(2, "l2.usda", restore(2)),
            object(3, "l3.usda", ObjectAction::Remap { from: 3, to: 30 }),
            object(4, "l4.usda", ObjectAction::Nothing),
        ];
        let lost = LostSet {
            layers: [1, 2, 3, 4, 5]
                .into_iter()
                .map(|oid| {
                    lost_layer(
                        &format!("l{oid}.usda"),
                        oid,
                        LostLayerKind::NoSurvivor,
                        &HeldReason::NoBackup,
                    )
                })
                .collect(),
            ..LostSet::default()
        };

        assert_eq!(
            check_discardable(&plan, &lost),
            vec![
                DiscardDefect::LostLayerWritten { object_id: 1 },
                DiscardDefect::LostLayerWritten { object_id: 2 },
                DiscardDefect::LostLayerWritten { object_id: 3 },
            ]
        );
    }

    /// P51-b, LS3 rule 2 (fail-on-old): defer-51's accept-loss half. A lost
    /// layer `(s, L)` whose session restore's Step B writes `L` to another
    /// object is a defect, naming the object Step B writes; so is a lost
    /// session's layers-index name Step B writes when the session saw no
    /// mapping of it. A lost session whose mapping is the one Step B writes,
    /// and a restore without Step B, are not.
    #[test]
    fn p51b_ls3_rule_2_an_unmap_of_a_layer_step_b_writes_to_another_object_is_a_defect() {
        let mut plan = plan();
        plan.sessions = vec![
            session_restore(S, &[("l.usda", 20)], true),
            session_restore("s-b", &[("same.usda", 40), ("index.usda", 41)], true),
            session_restore("s-a", &[("a.usda", 50)], false),
        ];
        let lost = LostSet {
            layers: vec![
                lost_layer(
                    "l.usda",
                    10,
                    LostLayerKind::NoSurvivor,
                    &HeldReason::NoBackup,
                ),
                LostLayer {
                    session_id: "s-a".to_owned(),
                    ..lost_layer(
                        "a.usda",
                        11,
                        LostLayerKind::NoSurvivor,
                        &HeldReason::NoBackup,
                    )
                },
            ],
            sessions: vec![LostSession {
                session_id: "s-b".to_owned(),
                evidence: "its listed session backup".to_owned(),
                objects: vec![40],
                mappings: vec![("same.usda".to_owned(), 40)],
                layer_names: vec!["index.usda".to_owned(), "same.usda".to_owned()],
                backup_listed: true,
                unconfirmed: None,
            }],
            ..LostSet::default()
        };

        assert_eq!(
            check_discardable(&plan, &lost),
            vec![
                DiscardDefect::UnmapWrittenByStepB {
                    session_id: S.to_owned(),
                    layer_name: "l.usda".to_owned(),
                    written: 20,
                },
                DiscardDefect::UnmapWrittenByStepB {
                    session_id: "s-b".to_owned(),
                    layer_name: "index.usda".to_owned(),
                    written: 41,
                },
                // Rule 4: a lost session the plan restores.
                DiscardDefect::LostSessionNotLost {
                    session_id: "s-b".to_owned(),
                },
            ]
        );
    }

    /// LS3, rule 3. A value removal's object must be one the writer
    /// restored, kept or cleared: one held with no survivor, one moved, and
    /// one with no plan are defects.
    #[test]
    fn ls3_rule_3_a_value_removal_from_an_object_the_plan_does_not_write_is_a_defect() {
        let mut plan = plan();
        plan.objects = vec![
            object(1, "l1.usda", ObjectAction::Keep),
            object(2, "l2.usda", restore(2)),
            object(3, "l3.usda", ObjectAction::ClearFlags { band: band() }),
            object(4, "l4.usda", ObjectAction::Nothing),
            object(5, "l5.usda", ObjectAction::Remap { from: 5, to: 50 }),
        ];
        let lost = LostSet {
            values: [1, 2, 3, 4, 5, 6]
                .into_iter()
                .map(|object_id| ValueRemoval {
                    object_id,
                    session_id: S.to_owned(),
                    layer_name: format!("l{object_id}.usda"),
                    hashes: BTreeSet::from([H1]),
                    unconfirmed: BTreeSet::new(),
                    sites: Vec::new(),
                })
                .collect(),
            ..LostSet::default()
        };

        assert_eq!(
            check_discardable(&plan, &lost),
            vec![
                DiscardDefect::ValueRemovalNotWritten { object_id: 4 },
                DiscardDefect::ValueRemovalNotWritten { object_id: 5 },
                DiscardDefect::ValueRemovalNotWritten { object_id: 6 },
            ]
        );
    }

    /// LS3, rule 4. A lost session the plan restores, finishes the delete
    /// of, or saw a deletion marker of is a defect; one with none of these
    /// is not.
    #[test]
    fn ls3_rule_4_a_lost_session_that_is_restored_or_was_deleted_is_a_defect() {
        let mut plan = plan();
        plan.sessions = vec![
            session_restore("s-restored", &[], false),
            SessionPlan {
                session_id: "s-deleting".to_owned(),
                action: SessionAction::CompleteAbandonedDelete {
                    objects: Vec::new(),
                    layer_names: Vec::new(),
                    base_path: None,
                    mappings: Vec::new(),
                    path_name_members: Vec::new(),
                },
            },
            SessionPlan {
                session_id: "s-nothing".to_owned(),
                action: SessionAction::Nothing,
            },
        ];
        plan.sessions_seen = BTreeMap::from([(
            "s-marked".to_owned(),
            SessionSeen {
                deletion_marker: true,
                ..SessionSeen::default()
            },
        )]);
        let session = |session_id: &str| LostSession {
            session_id: session_id.to_owned(),
            evidence: "its listed session backup".to_owned(),
            objects: Vec::new(),
            mappings: Vec::new(),
            layer_names: Vec::new(),
            backup_listed: false,
            unconfirmed: None,
        };
        let lost = LostSet {
            sessions: [
                "s-restored",
                "s-deleting",
                "s-marked",
                "s-nothing",
                "s-unplanned",
            ]
            .into_iter()
            .map(session)
            .collect(),
            ..LostSet::default()
        };

        assert_eq!(
            check_discardable(&plan, &lost),
            vec![
                DiscardDefect::LostSessionNotLost {
                    session_id: "s-restored".to_owned()
                },
                DiscardDefect::LostSessionNotLost {
                    session_id: "s-deleting".to_owned()
                },
                DiscardDefect::LostSessionNotLost {
                    session_id: "s-marked".to_owned()
                },
            ]
        );
    }
}
