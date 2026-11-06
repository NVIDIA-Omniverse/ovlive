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

//! The writer's session-scoped stages (SDD SR-9.4, SR-9.5): abandoned
//! deletes, moves to a successor, and session restores.
//!
//! **Completing an abandoned delete** (a surviving deletion marker is the
//! evidence the session was deleted), and **discarding accept-loss's lost
//! session** (the operator's confirmed accept-loss, and no record, no marker
//! and no usable session backup at or after `T_safe`), are one discard
//! ([`discard_session`], told apart by its [`DiscardEvidence`]). Each step is
//! idempotent, in this order:
//!
//! 1. register the objects on the dead-object set (`recon_register`): an
//!    abandoned delete only with a backup store, a lost session on every
//!    adapter (PD-5). First, so a run killed after the unmap, whose plan then
//!    names no object, has already registered them;
//! 2. wipe each object, its stream included;
//! 3. remove each surviving mapping, and every other layer name's
//!    layers-index member, through the single unmapping function
//!    (`storage::dr_write::unmap_layer_if`, a compare-and-delete). A mapping
//!    found naming another object fails an abandoned delete (exit 1) and is
//!    left as found, with a WARN, by accept-loss (defer-55);
//! 4. remove the session's path-index members, when its listed backup names
//!    them;
//! 5. delete the session's backup (only with a backup store; for a lost
//!    session only when one is listed).
//!
//! No session record is touched: a session's record and its deletion marker
//! are written by one transaction, so a marked session has no record, and a
//! lost session has none by definition.
//!
//! **A move** maps a layer from a fenced object to its complete successor,
//! only from the fenced object (`storage::dr_write::remap_layer_if`); the
//! fenced object is left for the orphan sweep.
//!
//! **A session restore** runs the session applier
//! (`checkpoint::session::apply_session_checkpoint`) with the plan's filtered
//! backup: Step A, the path index, always; Step B, the record, the mappings
//! and the layers index, when the plan says so. Step A before Step B is
//! load-bearing. The Job's Step B is the applier's unconditional
//! `MULTI`/`EXEC` (`SessionApplySteps::PathIndexThenRecord`), not on-access
//! recovery's conditional script.

use futures::{StreamExt, TryStreamExt};

use super::memory::RestoreMemory;
use super::{write_step, At, ExecSeams, StepError};
use crate::checkpoint::session::{apply_session_checkpoint, SessionApplied, SessionApplySteps};
use crate::checkpoint::store::CheckpointError;
use crate::checkpoint::SessionCheckpoint;
use crate::dr::job::plan::model::{ObjectAction, RestorePlan, SessionAction};
use crate::dr::job::plan::report::render_left_mapping;
use crate::dr::job::{BackupAccess, JobExit, JobRun};
use crate::storage::dr_write::{
    remap_layer_if, remove_path_members, unmap_layer_if, RemapOutcome, UnmapOutcome,
};
use crate::storage::external_values::recon_register;
use crate::storage::layer::dr_restore::{wipe_object_for_restore, WipeStream};

/// One session discard's plan: an abandoned delete's, or accept-loss's lost
/// session's.
pub(crate) struct SessionDiscard<'p> {
    /// The session.
    pub session_id: &'p str,
    /// Its surviving objects (each mapping's object is added).
    pub objects: &'p [u64],
    /// Every layer name it still carries.
    pub layer_names: &'p [String],
    /// Its base path, from its listed session backup, if any: its path-index
    /// members are removed only with it.
    pub base_path: Option<&'p str>,
    /// Its surviving `(layer, object)` mappings.
    pub mappings: &'p [(String, u64)],
    /// Its composite path-names members, from its listed session backup.
    pub path_name_members: &'p [String],
    /// Whether its session backup is deleted (with a backup store).
    pub delete_backup: bool,
}

/// What a session discard rests on, which decides where its objects are
/// registered dead, what a mapping found naming another object means, and
/// its kill points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiscardEvidence {
    /// A surviving deletion marker: the session was deleted (an abandoned
    /// delete). Its objects are registered only with a backup store (there
    /// is nothing to reclaim without one), and a mapping naming another
    /// object means another writer is active: exit 1.
    DeletionMarker,
    /// The operator's confirmed accept-loss for this incident, and the Job's
    /// own finding at or after `T_safe` that the session is lost (no record,
    /// no deletion marker, no usable session backup) — or, for a session
    /// whose backup read ended in an error rather than an absence, the
    /// operator's second `--confirm discard-unconfirmed-data` (option E) in
    /// place of that finding: the deliberate exception to "absence observed
    /// at a distance is not evidence" (`storage::keys::KeySchema::
    /// checkpoint_reconcile_dead`). Its objects are registered on every
    /// adapter (PD-5), so the counter floor covers their ids; a mapping
    /// naming another object is left as found, with a WARN (defer-55).
    AcceptLoss,
}

/// What a session discard did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionDiscardReport {
    /// The objects registered on the dead-object set.
    pub registered: usize,
    /// Mappings left as found: `(layer, the object the plan saw, the object
    /// found)` (accept-loss only).
    pub left: Vec<(String, u64, u64)>,
    /// Whether its session backup was deleted.
    pub backup_deleted: bool,
}

/// Stage 1: finish every abandoned delete of `plan`, up to the configured
/// parallelism at once. Returns the sessions finished.
pub(super) async fn complete_abandoned_deletes(
    run: &JobRun,
    plan: &RestorePlan,
    memory: &RestoreMemory,
    seams: &ExecSeams,
) -> Result<Vec<String>, JobExit> {
    let deletes: Vec<SessionDiscard<'_>> = plan
        .sessions
        .iter()
        .filter_map(|session| match &session.action {
            SessionAction::CompleteAbandonedDelete {
                objects,
                layer_names,
                base_path,
                mappings,
                path_name_members,
            } => Some(SessionDiscard {
                session_id: &session.session_id,
                objects,
                layer_names,
                base_path: base_path.as_deref(),
                mappings,
                path_name_members,
                delete_backup: true,
            }),
            _ => None,
        })
        .collect();
    // By index, not by item: a stream closure whose argument borrows trips
    // the compiler's higher-ranked closure inference under the body's `Send`
    // bound.
    let deletes = &deletes;
    futures::stream::iter(0..deletes.len())
        .map(|index| {
            let delete = &deletes[index];
            async move {
                discard_session(run, delete, DiscardEvidence::DeletionMarker, seams).await?;
                Ok::<_, JobExit>(delete.session_id.to_owned())
            }
        })
        .buffer_unordered(memory.configured_parallelism())
        .try_collect()
        .await
}

/// Discard one session (module docs, *Completing an abandoned delete*; the
/// same steps discard accept-loss's lost sessions): register its objects
/// dead (where `evidence` says), wipe them, unmap, remove its path-index
/// members, delete its backup. Each step is idempotent.
pub(crate) async fn discard_session(
    run: &JobRun,
    discard: &SessionDiscard<'_>,
    evidence: DiscardEvidence,
    seams: &ExecSeams,
) -> Result<SessionDiscardReport, JobExit> {
    let mode = run.mode;
    let session_id = discard.session_id;
    let accept_loss = evidence == DiscardEvidence::AcceptLoss;
    let (subject, after_register, after_wipe, after_unmap) = if accept_loss {
        (
            "lost session",
            At::DiscardAfterRegister(Some(session_id)),
            At::DiscardAfterWipe(Some(session_id)),
            At::DiscardAfterUnmap(Some(session_id)),
        )
    } else {
        (
            "deleted session",
            At::DeleteAfterRegister(session_id),
            At::DeleteAfterWipe(session_id),
            At::DeleteAfterUnmap(session_id),
        )
    };
    let mut report = SessionDiscardReport::default();
    let mut objects: Vec<u64> = discard
        .objects
        .iter()
        .copied()
        .chain(discard.mappings.iter().map(|(_, object_id)| *object_id))
        .collect();
    objects.sort_unstable();
    objects.dedup();

    // 1. Register first.
    if accept_loss || matches!(run.backup, BackupAccess::Store(_)) {
        let members: Vec<String> = objects.iter().map(u64::to_string).collect();
        let dead_set = run.keys.checkpoint_reconcile_dead();
        write_step(
            run,
            &|| format!("register the objects of {subject} {session_id} as deleted"),
            || async {
                recon_register(&run.valkey, &dead_set, &members)
                    .await
                    .map_err(|e| {
                        if crate::storage::client::is_transient_fred_error(&e) {
                            StepError::Retry(e.to_string())
                        } else {
                            StepError::Fatal(e.to_string())
                        }
                    })
            },
        )
        .await?;
        report.registered = members.len();
    }
    seams.stop_at(mode, after_register)?;

    // 2. Wipe, the stream included.
    for object_id in &objects {
        let object_id = *object_id;
        write_step(
            run,
            &|| format!("wipe object {object_id} of {subject} {session_id}"),
            || async {
                wipe_object_for_restore(&run.valkey, &run.keys, object_id, WipeStream::Unlink)
                    .await
                    .map_err(StepError::from)
            },
        )
        .await?;
    }
    seams.stop_at(mode, after_wipe)?;

    // 3. Unmap: each surviving mapping from its object, every other name's
    // layers-index member where its mapping is absent.
    let unmaps = discard
        .mappings
        .iter()
        .map(|(layer_name, object_id)| (layer_name.as_str(), Some(*object_id)))
        .chain(
            discard
                .layer_names
                .iter()
                .filter(|name| !discard.mappings.iter().any(|(mapped, _)| mapped == *name))
                .map(|name| (name.as_str(), None)),
        );
    for (layer_name, expected) in unmaps {
        let outcome = write_step(
            run,
            &|| format!("remove the mapping of layer {layer_name} of {subject} {session_id}"),
            || async {
                unmap_layer_if(&run.valkey, &run.keys, session_id, layer_name, expected)
                    .await
                    .map_err(StepError::from)
            },
        )
        .await?;
        if let UnmapOutcome::Other(found) = outcome {
            if accept_loss {
                // defer-55: a mapping that names another object than the
                // plan saw is left as found.
                match expected {
                    Some(expected) => tracing::warn!(
                        session_id,
                        layer_name,
                        object_id = found,
                        expected,
                        "{}",
                        render_left_mapping(session_id, layer_name, found, expected)
                    ),
                    None => tracing::warn!(
                        session_id,
                        layer_name,
                        object_id = found,
                        "the {mode} Job left layer {layer_name} of session {session_id} mapped \
                         to object {found}: its plan saw no mapping of it; it removes a mapping \
                         only while it names what its plan saw"
                    ),
                }
                report
                    .left
                    .push((layer_name.to_owned(), expected_id(expected), found));
                continue;
            }
            tracing::error!(
                session_id,
                layer_name,
                object_id = found,
                "the {mode} Job found layer {layer_name} of deleted session {session_id} mapped \
                 to object {found}, which its plan did not see; another writer is active; it \
                 stops writing and exits 1"
            );
            return Err(JobExit::Failed);
        }
    }
    seams.stop_at(mode, after_unmap)?;

    // 4. The path-index members, when the listed backup names them.
    if let Some(base_path) = discard.base_path {
        write_step(
            run,
            &|| format!("remove the path-index members of {subject} {session_id}"),
            || async {
                remove_path_members(
                    &run.valkey,
                    &run.keys,
                    base_path,
                    session_id,
                    discard.path_name_members,
                )
                .await
                .map_err(StepError::from)
            },
        )
        .await?;
    }
    if !accept_loss {
        seams.stop_at(mode, At::DeletePathMembersDone(session_id))?;
    }

    // 5. The session backup. Never retried by the Job.
    if let (BackupAccess::Store(store), true) = (&run.backup, discard.delete_backup) {
        if run.permit.begin_unit().await.is_err() {
            return Err(JobExit::Failed);
        }
        if let Err(e) = store.delete_session_checkpoint(session_id).await {
            super::step_failed(
                mode,
                &|| format!("delete the backup of {subject} {session_id}"),
                &e.to_string(),
                1,
            );
            return Err(JobExit::Failed);
        }
        report.backup_deleted = true;
        if accept_loss {
            seams.stop_at(mode, At::LostSessionBackupDeleted(session_id))?;
        }
    }
    Ok(report)
}

/// An unmapping's expected object, for the report: 0 for a layers-index
/// name with no mapping (`None`).
fn expected_id(expected: Option<u64>) -> u64 {
    expected.unwrap_or(0)
}

/// Stage 3: every move of `plan`, after the objects. Returns `(from, to)`
/// per move.
pub(super) async fn remap_layers(
    run: &JobRun,
    plan: &RestorePlan,
    memory: &RestoreMemory,
    seams: &ExecSeams,
) -> Result<Vec<(u64, u64)>, JobExit> {
    let moves: Vec<(&str, &str, u64, u64)> = plan
        .objects
        .iter()
        .filter_map(|object| match object.action {
            ObjectAction::Remap { from, to } => Some((
                object.session_id.as_str(),
                object.layer_name.as_str(),
                from,
                to,
            )),
            _ => None,
        })
        .collect();
    // By index (see `complete_abandoned_deletes`).
    let moves = &moves;
    futures::stream::iter(0..moves.len())
        .map(|index| {
            let (session_id, layer_name, from, to) = moves[index];
            remap_layer(run, seams, session_id, layer_name, from, to)
        })
        .buffer_unordered(memory.configured_parallelism())
        .try_collect()
        .await
}

/// One move (module docs).
async fn remap_layer(
    run: &JobRun,
    seams: &ExecSeams,
    session_id: &str,
    layer_name: &str,
    from: u64,
    to: u64,
) -> Result<(u64, u64), JobExit> {
    seams.stop_at(run.mode, At::RemapBefore(from))?;
    let outcome = write_step(
        run,
        &|| format!("move layer {layer_name} of session {session_id} to object {to}"),
        || async {
            remap_layer_if(&run.valkey, &run.keys, session_id, layer_name, from, to)
                .await
                .map_err(StepError::from)
        },
    )
    .await?;
    match outcome {
        RemapOutcome::Remapped | RemapOutcome::AlreadyTo => Ok((from, to)),
        RemapOutcome::Other(found) => {
            let found = found.map_or_else(|| "nothing".to_owned(), |o| format!("object {o}"));
            tracing::error!(
                session_id,
                layer_name,
                from,
                to,
                "the {mode} Job found layer {layer_name} of session {session_id} mapped to \
                 {found}, not to object {from} as its plan saw; another writer is active; it \
                 stops writing and exits 1",
                mode = run.mode
            );
            Err(JobExit::Failed)
        }
    }
}

/// Stage 4: every session restore of `plan`, up to the configured
/// parallelism at once. Returns `(restored with Step B, with Step A alone)`.
pub(super) async fn restore_sessions(
    run: &JobRun,
    plan: &RestorePlan,
    memory: &RestoreMemory,
    seams: &ExecSeams,
) -> Result<(usize, usize), JobExit> {
    let restores: Vec<(&str, &SessionCheckpoint, bool)> = plan
        .sessions
        .iter()
        .filter_map(|session| match &session.action {
            SessionAction::Restore { checkpoint, step_b } => {
                Some((session.session_id.as_str(), checkpoint, *step_b))
            }
            _ => None,
        })
        .collect();
    // By index (see `complete_abandoned_deletes`).
    let restores = &restores;
    let step_bs: Vec<bool> = futures::stream::iter(0..restores.len())
        .map(|index| {
            let (session_id, checkpoint, step_b) = restores[index];
            async move {
                restore_session(run, session_id, checkpoint, step_b, seams).await?;
                Ok::<_, JobExit>(step_b)
            }
        })
        .buffer_unordered(memory.configured_parallelism())
        .try_collect()
        .await?;
    let restored = step_bs.iter().filter(|step_b| **step_b).count();
    Ok((restored, step_bs.len() - restored))
}

/// Restore one session through the applier (module docs).
async fn restore_session(
    run: &JobRun,
    session_id: &str,
    checkpoint: &SessionCheckpoint,
    step_b: bool,
    seams: &ExecSeams,
) -> Result<(), JobExit> {
    // The Job's two modes are the unconditional ones: the fleet is held, so
    // the Job is the only writer, and its record step must be able to write a
    // backup's mapping into a session whose record survives. Neither mode
    // answers anything but `Applied`; the assertion makes an edit that sends
    // the Job down the conditional mode fail loudly in tests.
    let apply = |steps: SessionApplySteps| async move {
        apply_session_checkpoint(&run.valkey, &run.keys, session_id, checkpoint, None, steps)
            .await
            .map(|applied| {
                debug_assert_eq!(
                    applied,
                    SessionApplied::Applied,
                    "the restore Job's apply modes are unconditional and always write"
                );
            })
            .map_err(|e| match e {
                CheckpointError::Transient { .. } => StepError::Retry(e.to_string()),
                other => StepError::Fatal(other.to_string()),
            })
    };
    let steps = if step_b {
        SessionApplySteps::PathIndexThenRecord
    } else {
        SessionApplySteps::PathIndexOnly
    };
    // The kill point after Step A runs Step A alone first (a test seam;
    // Step A is idempotent, so the whole apply after it is the same writes).
    if seams.would_stop_at(At::SessionAfterStepA(session_id)) {
        write_step(
            run,
            &|| format!("restore the path index of session {session_id}"),
            || apply(SessionApplySteps::PathIndexOnly),
        )
        .await?;
        return seams.stop_at(run.mode, At::SessionAfterStepA(session_id));
    }
    write_step(run, &|| format!("restore session {session_id}"), || {
        apply(steps)
    })
    .await
}
