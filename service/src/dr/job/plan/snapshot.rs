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

//! The restore planner's reads (SDD SR-9.2): the store's facts, read once
//! after the quiet period (`storage::dr_job`), and the backup store's one
//! listing, re-taken once at `T_safe` when a write-shaping absence was seen
//! before it. Every function here only reads.
//!
//! A failed read is logged and ends the plan with exit 1 (retryable), having
//! written nothing — a keyspace walk whose slot map changed or whose node pin
//! missed its primary's slots, and an agreeing walk's two walks that returned
//! different keys (the store changed while the Job read it), included
//! (`storage::scan_guard`) — except a keyspace walk whose `CLUSTER NODES`
//! gave no usable myself line, which ends it with exit 3, and two reads whose
//! failure the plan carries instead:
//!
//! - a listed session backup that fails to read is
//!   [`SessionRead::ReadFailed`], and one that fails to decode
//!   [`SessionRead::Undecodable`]; either holds its session;
//! - a listed object's identity read ([`identify_listed_objects`]) that fails
//!   `Permanent` moves on to the next older generation, and an object no
//!   generation identifies is held as unknown (PD-4).

use std::collections::{BTreeMap, BTreeSet};

use futures::{StreamExt, TryStreamExt};

use super::decide::{Listing, SessionRead, StoreFacts, UnidentifiedObject};
use super::model::ExtHash;
use super::DR_PLAN_VALKEY_READ_CONCURRENCY;
use crate::checkpoint::lineage::group_layer_generations;
use crate::checkpoint::session_payload::SessionCheckpoint;
use crate::checkpoint::store::{CheckpointHandle, CheckpointStore};
use crate::dr::job::{JobExit, JobMode, JobRun};
use crate::storage::dr_job::{
    read_dead_object_ids, read_layer_mappings, read_object_ext_hashes, read_object_headers,
    read_session_facts, SessionFacts,
};
use crate::storage::external_values::hash_from_hex;
use crate::storage::scan_guard::WalkError;

/// How many session backups the planner reads at once.
pub const DR_PLAN_SESSION_READ_CONCURRENCY: usize = 8;

/// Log a failed store read and end the plan with exit 1.
fn store_read_failed(mode: JobMode, what: &str, error: &dyn std::fmt::Display) -> JobExit {
    tracing::error!(
        what,
        error = %error,
        "the {mode} Job could not read the store to plan the restore ({what}), wrote nothing \
         and exits 1"
    );
    JobExit::Failed
}

/// Log a failed keyspace read (`storage::dr_job`'s agreeing walks) and end
/// the plan: a store fault as [`store_read_failed`] does, exit 1; a walk
/// whose slot map changed, or whose node pin missed its primary's slots
/// (the line says whether the routing table was refreshed), exit 1, so
/// Kubernetes retries the Job and the walk is taken again; two walks that
/// returned different keys, exit 1 the same way, its line saying plainly
/// that the store changed while the Job read it; a walk whose
/// `CLUSTER NODES` gave no usable myself line, exit 3, the service held,
/// since the planner does not plan from a walk it cannot check.
fn walk_read_failed(mode: JobMode, what: &str, error: WalkError) -> JobExit {
    match error {
        WalkError::Incomplete(incomplete) => {
            tracing::error!(
                what,
                error = %incomplete,
                "the {mode} Job could not plan the restore ({what}): the backing store changed \
                 while the Job read it ({incomplete}); it wrote nothing and exits 1 so Kubernetes \
                 retries it"
            );
            JobExit::Failed
        }
        WalkError::Changed(change) => {
            tracing::error!(
                what,
                error = %change,
                "the {mode} Job could not use its walk of the store to plan the restore ({what}): \
                 {change}; it wrote nothing and exits 1 so Kubernetes retries it"
            );
            JobExit::Failed
        }
        WalkError::WitnessMissing(missing) => {
            tracing::error!(
                what,
                error = %missing,
                "the {mode} Job cannot plan the restore ({what}): {missing}; it wrote nothing, \
                 the service stays held and the Job exits 3; open a support case with this line"
            );
            JobExit::StaysHeld
        }
        store_fault @ (WalkError::Step { .. } | WalkError::Read { .. }) => {
            store_read_failed(mode, what, &store_fault)
        }
    }
}

/// Log a failed backup-store read and end the plan with exit 1.
fn backup_read_failed(mode: JobMode, what: &str, error: &dyn std::fmt::Display) -> JobExit {
    tracing::error!(
        what,
        error = %error,
        "the {mode} Job could not read the backup store to plan the restore ({what}), wrote \
         nothing and exits 1"
    );
    JobExit::Failed
}

/// The store's facts, but its sessions': every surviving header and
/// mapping (one agreeing keyspace walk each) and the dead-object set.
pub async fn read_store_facts(run: &JobRun) -> Result<StoreFacts, JobExit> {
    let headers = read_object_headers(&run.valkey, &run.keys, DR_PLAN_VALKEY_READ_CONCURRENCY)
        .await
        .map_err(|e| walk_read_failed(run.mode, "the object headers", e))?;
    let mappings = read_layer_mappings(&run.valkey, &run.keys, DR_PLAN_VALKEY_READ_CONCURRENCY)
        .await
        .map_err(|e| walk_read_failed(run.mode, "the layer mappings", e))?;
    let dead = read_dead_object_ids(&run.valkey, &run.keys)
        .await
        .map_err(|e| store_read_failed(run.mode, "the dead-object set", &e))?;
    Ok(StoreFacts {
        headers: headers.into_iter().map(|h| (h.object_id, h)).collect(),
        mappings,
        dead,
        sessions: BTreeMap::new(),
    })
}

/// Read the facts of every session in `session_ids` that `facts` does not
/// already hold, at most [`DR_PLAN_VALKEY_READ_CONCURRENCY`] at once. The
/// store does not change after the quiet period but by the Job, so a session
/// read once is never read again.
pub async fn read_missing_session_facts(
    run: &JobRun,
    facts: &mut StoreFacts,
    session_ids: &BTreeSet<String>,
) -> Result<(), JobExit> {
    let missing: Vec<String> = session_ids
        .iter()
        .filter(|sid| !facts.sessions.contains_key(*sid))
        .cloned()
        .collect();
    let read: Vec<(String, SessionFacts)> = futures::stream::iter(missing)
        .map(|session_id| async move {
            read_session_facts(&run.valkey, &run.keys, &session_id)
                .await
                .map(|f| (session_id, f))
        })
        .buffer_unordered(DR_PLAN_VALKEY_READ_CONCURRENCY)
        .try_collect()
        .await
        .map_err(|e| store_read_failed(run.mode, "the session records", &e))?;
    facts.sessions.extend(read);
    Ok(())
}

/// The listing the plan stands on, and its largest object id.
pub struct TakenListing {
    /// The listing.
    pub listing: Listing,
    /// The store-clock reading taken just before it.
    pub taken_at_ms: u64,
    /// The largest object id any listed artifact carries.
    pub listed_max_object_id: Option<u64>,
}

/// List the backup store once: its layer artifacts and its session backups,
/// after one read of the store's clock, then read each listed session backup.
/// `previous`, the listing a re-take replaces: a session backup it read as
/// present or unreadable is not read again; one it read as absent is.
pub async fn take_listing(
    run: &JobRun,
    store: &dyn CheckpointStore,
    previous: Option<&Listing>,
) -> Result<TakenListing, JobExit> {
    let taken_at_ms = run
        .clock
        .now_ms()
        .await
        .map_err(|e| store_read_failed(run.mode, "the store's clock", &e))?;
    let artifacts = store
        .list_layer_artifacts()
        .await
        .map_err(|e| backup_read_failed(run.mode, "the layer backups' listing", &e))?;
    let session_ids: BTreeSet<String> = store
        .list_session_checkpoint_ids()
        .await
        .map_err(|e| backup_read_failed(run.mode, "the session backups' listing", &e))?
        .into_iter()
        .collect();
    let generations = group_layer_generations(&artifacts);
    let listed_object_ids: BTreeSet<u64> = artifacts.iter().map(|a| a.object_id).collect();
    let listed_max_object_id = listed_object_ids.iter().next_back().copied();
    tracing::info!(
        listing_taken_at_ms = taken_at_ms,
        artifacts = artifacts.len(),
        layers = generations.len(),
        sessions = session_ids.len(),
        "listed the backup store once: {} layer backup files for {} layers and {} session \
         backups",
        artifacts.len(),
        generations.len(),
        session_ids.len()
    );

    let mut sessions: BTreeMap<String, SessionRead> = BTreeMap::new();
    let mut to_read: Vec<String> = Vec::new();
    for session_id in &session_ids {
        match previous.and_then(|p| p.sessions.get(session_id)) {
            Some(
                read @ (SessionRead::Present(_)
                | SessionRead::Undecodable(_)
                | SessionRead::ReadFailed(_)),
            ) => {
                sessions.insert(session_id.clone(), read.clone());
            }
            Some(SessionRead::Absent) | None => to_read.push(session_id.clone()),
        }
    }
    let read: Vec<(String, SessionRead)> = futures::stream::iter(to_read)
        .map(|session_id| async move {
            let read = match store.get_session_checkpoint(&session_id).await {
                Ok(Some(blob)) => match SessionCheckpoint::from_bytes(&blob.bytes) {
                    Ok(checkpoint) => SessionRead::Present(checkpoint),
                    Err(e) => SessionRead::Undecodable(e.to_string()),
                },
                Ok(None) => SessionRead::Absent,
                Err(e) => SessionRead::ReadFailed(e.to_string()),
            };
            (session_id, read)
        })
        .buffer_unordered(DR_PLAN_SESSION_READ_CONCURRENCY)
        .collect()
        .await;
    sessions.extend(read);

    let at_or_after_t_safe = taken_at_ms >= run.deadlines().t_safe_ms;
    Ok(TakenListing {
        listing: Listing {
            generations,
            listed_object_ids,
            session_ids,
            sessions,
            at_or_after_t_safe,
            // Filled by the identity read (`identify_listed_objects`).
            identity_absences: BTreeMap::new(),
        },
        taken_at_ms,
        listed_max_object_id,
    })
}

/// PD-1 under an empty session listing: read the identity (session, layer)
/// of listed objects no surviving header identifies, off their newest
/// readable generation's meta, one object at a time, stopping at the first
/// whose session has no record and no deletion marker — the one an S7 check
/// needs. Reads the facts of each identified session it judges. `known`
/// holds the identities already read, and `unidentified` the objects no
/// generation identified because a read failed `Permanent`. A re-take reads
/// every listed object it has not identified, except an all-`Permanent`
/// unidentified one over unchanged generations, which is not read, or
/// logged, again. So an object whose reads were all absent (in neither map)
/// is read again, and so is an unidentified one whose listed generations
/// changed or whose earlier read saw an absence, which can be a stale
/// not-found answer before `T_safe` (INV-1: an absence is re-observed once at
/// `T_safe`).
///
/// A listed generation that reads absent, whose meta companion is missing,
/// or whose read fails `Permanent` (logged at WARN) is skipped for the next
/// older one. An object no generation identifies, with at least one
/// `Permanent` failure, goes into `unidentified`, which the plan holds as
/// unknown (PD-4). Any other read error (`Transient` among them) ends the
/// plan with exit 1.
///
/// Returns the objects it read and could not identify after some generation
/// read as absent, each with the newest such generation: the caller stores
/// them as [`Listing::identity_absences`], and before `T_safe` each is a
/// trigger (g), so the absence is re-observed once at `T_safe` (INV-1).
pub async fn identify_listed_objects(
    run: &JobRun,
    store: &dyn CheckpointStore,
    listing: &Listing,
    facts: &mut StoreFacts,
    known: &mut BTreeMap<u64, (String, String)>,
    unidentified: &mut BTreeMap<u64, UnidentifiedObject>,
) -> Result<BTreeMap<u64, u64>, JobExit> {
    let mut absences: BTreeMap<u64, u64> = BTreeMap::new();
    if !listing.session_ids.is_empty() {
        return Ok(absences);
    }
    // Stop as soon as one already-known identity names a session with no
    // record and no marker: the S7 check has its evidence.
    let unexplained = |facts: &StoreFacts, session_id: &str| {
        facts
            .sessions
            .get(session_id)
            .is_some_and(|f| !f.meta_exists && !f.deleted_marker_exists)
    };
    for (session_id, _) in known.values() {
        if unexplained(facts, session_id) {
            return Ok(absences);
        }
    }
    for (object_id, handles) in &listing.generations {
        let listed: Vec<u64> = handles.iter().map(CheckpointHandle::seq).collect();
        if known.contains_key(object_id)
            || facts.dead.contains(object_id)
            || facts
                .headers
                .get(object_id)
                .is_some_and(|h| !h.session_id().is_empty())
            || unidentified
                .get(object_id)
                .is_some_and(|u| !u.read_again(&listed))
        {
            continue;
        }
        let mut identity: Option<(String, String)> = None;
        // The newest generation whose read failed `Permanent`, and its error.
        let mut failed: Option<(u64, String)> = None;
        // The newest generation that read as absent: possibly stale before
        // `T_safe`, so it is a trigger then and the re-take reads the object
        // again.
        let mut absent: Option<u64> = None;
        for handle in handles {
            match store.get_layer_checkpoint(handle).await {
                Ok(Some(blob)) => {
                    identity = Some((blob.meta.session_id, blob.meta.layer_name));
                    break;
                }
                Ok(None) => {
                    absent.get_or_insert(handle.seq());
                    continue;
                }
                Err(e) if e.is_missing_meta_companion() => {
                    absent.get_or_insert(handle.seq());
                    continue;
                }
                // One artifact that fails `Permanent` on every read must not
                // make every retry of the Job exit 1 having written nothing;
                // an object no generation identifies holds as unknown instead
                // (PD-4), since its session may be the S7 evidence.
                Err(e) if e.is_permanent() => {
                    tracing::warn!(
                        object_id,
                        generation = handle.seq(),
                        error = %e,
                        "the {mode} Job could not read backup generation {} of object \
                         {object_id} to learn its session ({e}); it tries the next older \
                         generation, if there is one",
                        handle.seq(),
                        mode = run.mode
                    );
                    failed.get_or_insert_with(|| (handle.seq(), e.to_string()));
                    continue;
                }
                Err(e) => {
                    return Err(backup_read_failed(
                        run.mode,
                        "a listed backup's identity",
                        &e,
                    ))
                }
            }
        }
        let Some((session_id, layer_name)) = identity else {
            if let Some(seq) = absent {
                absences.insert(*object_id, seq);
            }
            match failed {
                Some((seq, error)) => {
                    unidentified.insert(
                        *object_id,
                        UnidentifiedObject {
                            listed,
                            seq,
                            error,
                            saw_absent: absent.is_some(),
                        },
                    );
                }
                None => {
                    unidentified.remove(object_id);
                }
            }
            continue;
        };
        unidentified.remove(object_id);
        if session_id.is_empty() {
            continue;
        }
        known.insert(*object_id, (session_id.clone(), layer_name));
        read_missing_session_facts(run, facts, &BTreeSet::from([session_id.clone()])).await?;
        if unexplained(facts, &session_id) {
            return Ok(absences);
        }
    }
    Ok(absences)
}

/// The external values each object in `object_ids` references, from its
/// surviving `obj:ext` index, at most [`DR_PLAN_VALKEY_READ_CONCURRENCY`] at
/// once. A field that is not 64 hex characters is skipped.
pub async fn read_survivor_ext(
    run: &JobRun,
    object_ids: &[u64],
) -> Result<BTreeMap<u64, BTreeSet<ExtHash>>, JobExit> {
    let read: Vec<(u64, BTreeSet<ExtHash>)> = futures::stream::iter(object_ids.iter().copied())
        .map(|object_id| async move {
            read_object_ext_hashes(&run.valkey, &run.keys, object_id)
                .await
                .map(|fields| {
                    (
                        object_id,
                        fields
                            .iter()
                            .filter_map(|hex| hash_from_hex(hex))
                            .collect::<BTreeSet<ExtHash>>(),
                    )
                })
        })
        .buffer_unordered(DR_PLAN_VALKEY_READ_CONCURRENCY)
        .try_collect()
        .await
        .map_err(|e| {
            store_read_failed(run.mode, "the surviving layers' external-value indexes", &e)
        })?;
    Ok(read.into_iter().collect())
}
