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

//! Accept-loss's discards (SDD SR-10; PD-5, PD-6, PD-8; defer-55, defer-65):
//! [`discard`] carries out a lost set (`plan::lost`) after the restore
//! writer and the coverage check's finalise, and before the recount, so the
//! recount counts every discard and the un-hold's no-write-since check
//! covers them (ADD 9). Every step is a write unit through the permit
//! (`super::write_step`, or `begin_unit` before a backup-store call), in
//! this order:
//!
//! - **D0, the header pre-read**, which writes nothing to the store (each
//!   read is still a permit unit, as every step's is): the header of every
//!   lost layer whose plan found no survivor. A structurally complete,
//!   non-retiring header means another writer is active, so the Job never
//!   judges a mapping dangling when its target is complete (defer-55):
//!   ERROR (AL9), exit 1, with nothing written — no registration, wipe or
//!   unmap. Before the registration, because a registered object reads as
//!   mapped to a deleted object on the re-run (exit 1 is retried), and that
//!   kind is wiped without a header read.
//! - **D1, register** every discarded lost layer's object on the
//!   dead-object set (`recon_register`), on every adapter (PD-5): the
//!   evidence is the operator's confirmed accept-loss and the Job's own
//!   finding, at or after `T_safe`, that the object can never be mapped
//!   again — or, under the second `--confirm discard-unconfirmed-data`
//!   (option E), for a layer whose backup read ended in an error, that
//!   second confirm instead of the finding: the deliberate exception to
//!   "absence observed at a distance is not evidence"
//!   (`storage::keys::KeySchema::checkpoint_reconcile_dead`). Before the
//!   wipe and the unmap, so a kill after the unmap keeps the registration.
//!   Safe while a header stands: the clean-up's dead-object phase defers a
//!   member whose header stands and is unmapped, and nothing reaps while the
//!   fleet is held. A layer whose only backups are over the per-delta entry
//!   limit (`LostLayerKind::OverLimitBackups`) is never registered.
//!   Right before the `ZADD NX`, D1 reads which of its members the set
//!   lacks (`recon_absent`): those are the members this run adds, the only
//!   ones D2 may remove again. Read, not taken from the script's reply,
//!   because a register that runs twice (fred's re-send on a dropped
//!   connection, a retried step) answers "present" for a member its first
//!   run added; so a member another writer adds between the read and the
//!   `ZADD NX` counts as this run's, which is the leak direction.
//! - **D2, wipe** each discarded lost layer's object, its stream included
//!   (`storage::layer::dr_restore::wipe_object_for_restore`), after one more
//!   re-read of its header, for the window since D0: a complete,
//!   non-retiring header is AL9 again, and before exiting 1 the Job removes
//!   from the dead-object set every member this run added and D2 has not
//!   wiped, that object included (`recon_retire`), and logs AL9's companion
//!   WARN; nothing else is written. Not only that object: every lost layer
//!   D2 has not reached was registered at D1 too, and its object could be
//!   completed in the same window. So the retry plans those objects from
//!   their headers, and reads them at D0, instead of planning them as
//!   mapped to deleted objects, which it would wipe unread. A member
//!   another writer registered before D1's read stays, with that said in
//!   the companion line; an object D2 already wiped stays registered, as a
//!   kill after its wipe would leave it. A removal that fails past its
//!   retries logs the step's ERROR naming the objects and the key, and the
//!   Job still exits 1 with them registered. The remaining residual (SDD
//!   §22.22 *Accept-loss*): a kill — or a failure that exits without the
//!   removal, such as a wipe or a header read failing past its retries —
//!   after D1 and before the removal lands, while a writer that does not
//!   honour the hold completes the header of an object registered at D1. A
//!   mapping to a deleted object is wiped without a read; an over-limit
//!   layer is not wiped.
//! - **D3, unmap** each lost layer, the over-limit ones included (each with
//!   its own WARN, AL13), through the single unmapping function
//!   (`storage::dr_write::unmap_layer_if`), expecting the object the plan
//!   saw; a mapping found naming another object is left as found, with a
//!   WARN (defer-55). First, with a backup store, the session backup of
//!   every session with an over-limit layer is rewritten without those
//!   layers while the store still maps them (`checkpoint::session::
//!   rewrite_session_checkpoint`'s `leave_out`: a layer is left out only
//!   while it is mapped to the object the plan saw), one write unit each as
//!   at D6, and each such session joins D6's set. An over-limit object is
//!   never registered, so a kill after its unmap with the old backup still
//!   naming it would have the re-run find that mapping to an object neither
//!   dead nor mapped, take the session restore's Step B, write the session
//!   record back from that backup (a no-op unless the incident lost record
//!   fields: no RPC changes a record after its create) and map the layer
//!   again, so the re-run unmaps it a second time and AL13's "no mapping or
//!   session backup names it" is false until D6. Rewritten first, no kill point leaves a backup naming an
//!   unmapped over-limit layer; a kill between the rewrite and the unmap
//!   leaves the layer mapped beside a backup that matches the store but
//!   for it, and the re-run finds it lost again with no Step B.
//! - **Lost sessions** (PD-6), up to `LIVE_EDIT_DR_RESTORE_PARALLELISM` at
//!   once, through the abandoned delete's own steps
//!   (`super::session::discard_session` under
//!   `DiscardEvidence::AcceptLoss`): register, wipe, unmap, and **D4**,
//!   delete the session's backup when one is listed (with a backup store).
//! - **D5, value removal**, in place, up to the restore memory's parallelism
//!   at once, after one allocator purge (the run's `allocator_purge`, when
//!   installed) so the pages earlier phases freed are not resident in the
//!   first removal. Per layer:
//!   1. read its header's `seq` S and its stream's floor F, and band at F
//!      (the flag clear's rule);
//!   2. take ONE admission from the restore memory
//!      (`memory::value_removal_admission`: the purged figure only while a
//!      purge is installed and none has failed, else the un-purged
//!      fail-safe), held until the header lands and given back on every
//!      earlier exit; the removal never waits on the gate while it holds
//!      it;
//!   3. walk `obj:flds`, then `obj:tss`, by rank pages of
//!      `memory::DR_VALUE_REMOVAL_PAGE_ROWS` (`storage::scan_guard::RankWalk`
//!      on one snapshot-lane client, with no witness): read the index's
//!      `ZCARD` n, then page by descending absolute ranks, `ZRANGE n−256
//!      n−1`, then `ZRANGE n−512 n−257`, and so on down to rank 0; each
//!      read a write step, sent with one attempt per write-step attempt so
//!      the two retry bounds do not multiply, retried in place after a
//!      transient error, a deterministic one failing it at once. A row this
//!      removal deletes is at or above the current page's lowest rank, so no
//!      rank still to be read shifts, whenever a batch lands. A page that
//!      answers a number of members other than its ranks span ends the
//!      walk, exit 1.
//!      Each page's rows are read at once by the readers' rule
//!      (`storage::diff_read::read_value_row_external_hash`: a row holding a
//!      `value` names no hash, a malformed `hash` fails the step);
//!   4. add each row naming a removed value to the pending batch, by its
//!      member as read, and send the batch, fenced on `seq == S` and not
//!      retiring, before a row would cross either batch limit; then the
//!      rest (the kill point `ValueRowsAfterBatch` follows each row batch).
//!      Then, before the first `obj:ext` batch, read each index's `ZCARD`
//!      again: the walk is complete only when its pages returned the start
//!      `ZCARD` n and the end `ZCARD` is n less the rows this removal
//!      deleted from that index (`storage::scan_guard::check_count`);
//!      otherwise the store changed while the Job read it, and the removal
//!      logs one ERROR naming the index and the counts and exits 1;
//!   5. remove every removed hash but the last from `obj:ext`, in hash
//!      order, in batches within the limits, fenced the same way;
//!   6. stamp the banded header last, which carries the last hash's `HDEL
//!      obj:ext` (`storage::layer::dr_restore`, *The value removal*).
//!
//!   **The purge around each transaction.** Every row batch, `obj:ext` batch
//!   and the header is purged around (`RemovalPurge`): as each attempt
//!   begins, inside its write step (so a retried attempt does not rebuild
//!   its frame beside the last one's freed pages), and once it landed:
//!   before the kill point `ValueRowsAfterBatch` for a row batch, and before
//!   the admission is given back for the header. Each purge returns every
//!   jemalloc arena's freed pages to the OS, so the removal's resident memory
//!   is its heap plus one transaction's freed pages rather than whatever
//!   each worker arena keeps for its 10 s decay; it runs on the blocking
//!   pool and writes nothing. That is two purges per transaction when
//!   nothing retries. Their absolute time cost is unmeasured. One relative
//!   A/B put D5's wall time 20.7 % higher with the purge on than off (a
//!   local sandbox, 20,000 rows with 4,096 B names, the name limit, ABAB
//!   twice; record `restore-memory-envelope-tv3-20261022`, `purge_time_ab`).
//!   That is a relative figure from one host, never a product number; it
//!   is within the +25 % bound under which the removal purges around every
//!   transaction rather than less often. A purge that fails
//!   while the removal holds the purged figure tops the admission up to the
//!   un-purged one, forced, until the admission is given back; the purge's
//!   handle logs one WARN, and later removals are admitted at the un-purged
//!   figure (one that asked for the purged figure before the failure and
//!   was admitted after it tops up the same way the moment it is admitted).
//!   The Job logs once, before planning, whether a purge is
//!   installed and the admission that sets (`log_value_removal_purge`).
//!
//!   What it holds beside its admission is one page's members and the
//!   pending batch's rows, whatever the layer's size. The batch's rows are
//!   up to about 5 MiB of Rust-side strings and B-tree nodes [model]: not
//!   priced by the batch charge's argument-byte term (which prices fred's
//!   queued copies) but inside the admission, whose price covers the
//!   removal's whole resident growth with the purge, this set included:
//!   37.1 MB at worst, under the 50 MiB admission [measured to 719,922
//!   rows, with the purge: record `restore-memory-envelope-tv3-20261022`,
//!   `value_removal_admission`; measured on a walk that also held up to a
//!   batch of rows already sent, which this one does not].
//!   The walk is shown complete on the held, quiet store, whichever server
//!   process answered each read: a sorted index's rank order is a function
//!   of its content, and the end `ZCARD` check catches a member added or
//!   lost under the walk, a lost delete included. A walk not shown complete
//!   ends, exit 1, before the `obj:ext` batches and the header, so the
//!   header is never banded after such a walk, and every removed hash is
//!   still in `obj:ext`.
//! - **D6, session rewrites** (with a backup store), up to the configured
//!   parallelism at once: every session the lost set names, and every
//!   session of a lost layer whose mapping D3 removed or found absent, and
//!   every session D3 rewrote first, is rebuilt from the store, leaving
//!   nothing out, and PUT (`checkpoint::session::
//!   rewrite_session_checkpoint`), after the unmaps so the rewritten backups
//!   describe the final mappings. So D3's early rewrite never stands in for
//!   D6's: each of its sessions is built again from the whole store, with a
//!   later stamp from the same counter, and its PUT replaces the early one;
//!   a layer the unmap found mapped to another object is named again. A
//!   session with no record writes nothing.
//!
//! No error path makes a compensating write but D2's removal from the
//! dead-object set at an AL9, and a backup-store call is never retried by
//! the Job. A failure stops the discards where they are; a re-run
//! converges: a registered object whose mapping survives reads as
//! mapped to a deleted object; an over-limit layer whose mapping survives
//! is found lost again, beside a session backup that names it or, after
//! D3's early rewrite, one that does not, neither taking Step B for it; an
//! over-limit layer already unmapped has no session backup naming it (with
//! a backup store, D3 rewrote it first), so its listed backups read as
//! unreferenced; a value removal killed anywhere before its header
//! landed — after its last row batch included — finds at least its last
//! hash still in `obj:ext` (the header carries that hash's `HDEL`, and
//! every other hash's follows the last row batch), its object complete at
//! the same `seq`, walks its indices again from the start, removes the rest
//! and stamps the header — a removal whose walk was not shown complete ends
//! the same way, its deleted rows named only hashes still in `obj:ext`, so
//! the lost set plans the removal again at the same `seq`; a session
//! backup's rewrite set is recomputed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use fred::types::Value;
use futures::{StreamExt, TryStreamExt};

use super::memory::{value_removal_admission, Admission, RestoreGate, RestoreMemory};
use super::object::{band_for, command_timeout, stream_floor, warn_foreign, LayerRef};
use super::session::{discard_session, DiscardEvidence, SessionDiscard};
use super::{step_failed, write_step, At, BatchRecord, ExecSeams, StepError};
use crate::checkpoint::session::{rewrite_session_checkpoint, SessionRewrite};
use crate::dr::job::plan::lost::{LostLayer, LostLayerKind, LostSession, LostSet, ValueRemoval};
use crate::dr::job::plan::model::ExtHash;
use crate::dr::job::plan::report::{
    render_left_mapping, render_lost_layer, render_lost_object_complete,
    render_lost_object_unregistered, render_lost_session, render_over_limit_layer,
    render_value_removal,
};
use crate::dr::job::{BackupAccess, JobExit, JobMode, JobRun};
use crate::storage::allocator_purge::AllocatorPurge;
use crate::storage::client::{txn_timeout, FredErrOrigin, POOL_LABEL_SNAPSHOT};
use crate::storage::diff_read::{read_value_row_external_hash, SnapshotKeyFamily};
use crate::storage::dr_job::read_object_header;
use crate::storage::dr_write::{unmap_layer_if, UnmapOutcome};
use crate::storage::external_values::{recon_absent, recon_register, recon_retire};
use crate::storage::layer::dr_restore::{
    remove_value_rows, wipe_object_for_restore, write_value_removal_header, BandedFence,
    BandedHeaderWrite, BatchLimit, RemovedRow, SeqFencedOutcome, ValueRemovalBatch, WipeStream,
    RESTORE_BATCH_ARG_BYTES, VALUE_REMOVAL_HDEL_ARG_BYTES,
};
use crate::storage::memory_budget::MemoryTopUp;
use crate::storage::scan_guard::{check_count, RankWalk, WalkError, WalkIncomplete};

/// What the discards did, for the accept-loss summary (AL11) and a test's
/// discard observer. Not persisted: the log is the record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiscardReport {
    /// Lost layers whose mapping was removed, or found already absent:
    /// `(session, layer, object)`. Not the over-limit ones.
    pub layers_discarded: Vec<(String, String, u64)>,
    /// Lost layers whose only backups are over the per-delta entry limit,
    /// whose mapping was removed or found already absent, their object
    /// neither registered nor wiped: `(session, layer, object)`.
    pub over_limit_unmapped: Vec<(String, String, u64)>,
    /// Mappings left as found: `(session, layer, the object the plan saw,
    /// the object found)`; 0 for a layers-index name the plan saw no
    /// mapping of.
    pub left: Vec<(String, String, u64, u64)>,
    /// Lost sessions discarded.
    pub sessions_discarded: Vec<String>,
    /// Each value-removed object's removal.
    pub values: BTreeMap<u64, ValueRemovalRecord>,
    /// Objects registered on the dead-object set (re-registrations
    /// included).
    pub registered: usize,
    /// Session backups deleted.
    pub session_backups_deleted: usize,
    /// Session backups rewritten.
    pub rewritten: Vec<String>,
    /// Items discarded although they could not be confirmed lost (option
    /// E); 0 without it.
    pub unconfirmed: usize,
    /// Items that could not be confirmed lost and were kept as their WARN
    /// says (option E: the defer-47 object, a layer whose survivor stays);
    /// 0 without it.
    pub unconfirmed_kept: usize,
}

impl DiscardReport {
    /// The distinct values removed, over every layer.
    pub fn value_count(&self) -> usize {
        self.values
            .values()
            .flat_map(|v| v.hashes.iter())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// One layer's value removal, as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueRemovalRecord {
    /// The values removed.
    pub hashes: BTreeSet<ExtHash>,
    /// The distinct rows deleted: each row its batches committed, once.
    pub rows: usize,
    /// The `seq` the header carried before: S.
    pub prior_seq: u64,
    /// The band's floor: `max(S, plausible high-water)`.
    pub floor: u64,
    /// The band.
    pub band: u64,
    /// The banded `seq` the layer resumes at.
    pub seeded_seq: u64,
    /// Each transaction, in order; the header transaction last.
    pub batches: Vec<BatchRecord>,
}

/// Carry out `lost` (module docs): D0 to D6, in order. `Ok` with what was
/// done; `Err` ends the run, already logged.
pub async fn discard(
    run: &JobRun,
    lost: &LostSet,
    seams: &ExecSeams,
) -> Result<DiscardReport, JobExit> {
    let mode = run.mode;
    let memory = RestoreMemory::derive(&run.config, mode).map_err(|refusal| {
        tracing::error!(
            refusal = %refusal,
            "the {mode} Job could not derive its restore memory, discarded nothing and exits 1"
        );
        JobExit::Failed
    })?;
    let (unconfirmed, unconfirmed_kept) = lost.unconfirmed_counts();
    let mut report = DiscardReport {
        unconfirmed,
        unconfirmed_kept,
        ..DiscardReport::default()
    };
    let mut rewrites: BTreeSet<String> = lost.rewrites.clone();

    // D0: every no-survivor lost layer's header, read before anything is
    // written, so a complete object is never registered (defer-55).
    for layer in lost.layers.iter().filter(|l| l.kind.rereads_header()) {
        if header_is_complete(run, layer).await? {
            log_lost_object_complete(layer);
            return Err(JobExit::Failed);
        }
    }

    // D1: register every discarded lost layer's object, before its wipe and
    // its unmap, first reading which of them the set lacks: those are the
    // ones this run's registration adds, and the only ones D2 may remove.
    let mut members: Vec<u64> = lost
        .layers
        .iter()
        .filter(|l| l.kind.discards_object())
        .map(|l| l.object_id)
        .collect();
    members.sort_unstable();
    members.dedup();
    let dead_set = run.keys.checkpoint_reconcile_dead();
    let mut added: BTreeSet<u64> = BTreeSet::new();
    if !members.is_empty() {
        let names: Vec<String> = members.iter().map(u64::to_string).collect();
        let absent = write_step(
            run,
            &|| {
                format!(
                    "read which of {} lost layers' objects the dead-object set {dead_set} names",
                    names.len()
                )
            },
            || async {
                recon_absent(&run.valkey, &dead_set, &names)
                    .await
                    .map_err(fred_step_error)
            },
        )
        .await?;
        added = absent
            .iter()
            .filter_map(|m| m.parse::<u64>().ok())
            .collect();
        write_step(
            run,
            &|| format!("register {} lost layers' objects as deleted", names.len()),
            || async {
                recon_register(&run.valkey, &dead_set, &names)
                    .await
                    .map_err(fred_step_error)
            },
        )
        .await?;
        report.registered += members.len();
    }
    seams.stop_at(mode, At::DiscardAfterRegister(None))?;
    seams.after_discard_register(&members);

    // D2: the header's second read, then the wipe (none for an over-limit
    // layer, whose object is kept). A complete header stops the run (AL9)
    // after removing from the dead-object set what D1 added and D2 has not
    // wiped.
    let mut wiped: BTreeSet<u64> = BTreeSet::new();
    for layer in &lost.layers {
        if layer.kind.rereads_header() && header_is_complete(run, layer).await? {
            log_lost_object_complete(layer);
            let unwiped: Vec<u64> = added.difference(&wiped).copied().collect();
            let object_id = layer.object_id;
            let foreign = (members.binary_search(&object_id).is_ok()
                && !added.contains(&object_id))
            .then_some(object_id);
            unregister(run, &dead_set, &unwiped, foreign).await?;
            return Err(JobExit::Failed);
        }
        if layer.kind.discards_object() {
            wipe_lost_object(run, layer.object_id).await?;
            wiped.insert(layer.object_id);
        }
    }
    seams.stop_at(mode, At::DiscardAfterWipe(None))?;

    // D3, first: the session backups of the over-limit layers, rewritten
    // without them while they are still mapped, so no kill leaves a backup
    // naming a layer the store no longer maps and the Job never registered
    // dead. D6 rewrites each of these sessions again, whatever the unmap
    // finds.
    let mut over_limit: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for layer in lost
        .layers
        .iter()
        .filter(|l| l.kind == LostLayerKind::OverLimitBackups)
    {
        over_limit
            .entry(layer.session_id.clone())
            .or_default()
            .insert(layer.layer_name.clone(), layer.object_id);
    }
    rewrite_sessions(run, &over_limit, &memory, seams, Rewrite::AheadOfUnmap).await?;
    rewrites.extend(over_limit.into_keys());
    seams.stop_at(mode, At::DiscardAfterOverLimitRewrite)?;

    // D3, then: unmap, from the object the plan saw only.
    for layer in &lost.layers {
        let (session_id, layer_name, object_id) = (
            layer.session_id.as_str(),
            layer.layer_name.as_str(),
            layer.object_id,
        );
        let outcome = write_step(
            run,
            &|| format!("remove the mapping of layer {layer_name} of session {session_id}"),
            || async {
                unmap_layer_if(
                    &run.valkey,
                    &run.keys,
                    session_id,
                    layer_name,
                    Some(object_id),
                )
                .await
                .map_err(StepError::from)
            },
        )
        .await?;
        match outcome {
            UnmapOutcome::Unmapped | UnmapOutcome::Absent => {
                let entry = (session_id.to_owned(), layer_name.to_owned(), object_id);
                if layer.kind.discards_object() {
                    tracing::warn!(
                        session_id,
                        layer_name,
                        object_id,
                        "{}",
                        render_lost_layer(layer)
                    );
                    report.layers_discarded.push(entry);
                } else {
                    tracing::warn!(
                        session_id,
                        layer_name,
                        object_id,
                        "{}",
                        render_over_limit_layer(layer)
                    );
                    report.over_limit_unmapped.push(entry);
                }
                rewrites.insert(session_id.to_owned());
            }
            UnmapOutcome::Other(found) => {
                tracing::warn!(
                    session_id,
                    layer_name,
                    object_id = found,
                    expected = object_id,
                    "{}",
                    render_left_mapping(session_id, layer_name, found, object_id)
                );
                report.left.push((
                    session_id.to_owned(),
                    layer_name.to_owned(),
                    object_id,
                    found,
                ));
            }
        }
    }
    seams.stop_at(mode, At::DiscardAfterUnmap(None))?;

    // Lost sessions, and D4: their backups.
    discard_lost_sessions(run, &lost.sessions, &memory, seams, &mut report).await?;

    // D5: value removal.
    remove_values(run, &lost.values, &memory, seams, &mut report).await?;

    // D6: session rewrites, after the unmaps, each of the whole session.
    let rewrites: BTreeMap<String, BTreeMap<String, u64>> = rewrites
        .into_iter()
        .map(|session_id| (session_id, BTreeMap::new()))
        .collect();
    report.rewritten =
        rewrite_sessions(run, &rewrites, &memory, seams, Rewrite::AfterUnmaps).await?;
    Ok(report)
}

/// D2's wipe of one lost layer's object, its stream included.
async fn wipe_lost_object(run: &JobRun, object_id: u64) -> Result<(), JobExit> {
    write_step(run, &|| format!("wipe lost object {object_id}"), || async {
        wipe_object_for_restore(&run.valkey, &run.keys, object_id, WipeStream::Unlink)
            .await
            .map_err(StepError::from)
    })
    .await
}

/// D0 and D2's header read for one lost layer: whether its header is
/// structurally complete and not retiring, which the caller answers with
/// AL9 and exit 1 (module docs).
async fn header_is_complete(run: &JobRun, layer: &LostLayer) -> Result<bool, JobExit> {
    let object_id = layer.object_id;
    let header = write_step(
        run,
        &|| format!("read the header of lost object {object_id}"),
        || async {
            read_object_header(&run.valkey, &run.keys, object_id)
                .await
                .map_err(fred_step_error)
        },
    )
    .await?;
    Ok(header.is_some_and(|h| h.is_structurally_complete() && !h.is_retiring()))
}

/// AL9: `layer`'s object found complete.
fn log_lost_object_complete(layer: &LostLayer) {
    tracing::error!(
        object_id = layer.object_id,
        session_id = %layer.session_id,
        layer_name = %layer.layer_name,
        "{}",
        render_lost_object_complete(layer)
    );
}

/// D2's stop on a lost layer found complete, after its AL9: remove
/// `unwiped` — every object this run's registration added and D2 has not
/// wiped, AL9's object among them unless another writer registered it
/// (`foreign`) — from the dead-object set `dead_set`, then log AL9's
/// companion. The retry then plans those objects from their headers again
/// instead of as mapped to deleted objects, which it would wipe unread. A
/// removal that fails past its retries logs the write step's ERROR, naming
/// the objects and the key, and the caller still exits 1 with them
/// registered.
async fn unregister(
    run: &JobRun,
    dead_set: &str,
    unwiped: &[u64],
    foreign: Option<u64>,
) -> Result<(), JobExit> {
    if !unwiped.is_empty() {
        let names: Vec<String> = unwiped.iter().map(u64::to_string).collect();
        write_step(
            run,
            &|| {
                format!(
                    "remove objects {} that this run registered as deleted from the dead-object \
                     set {dead_set}",
                    names.join(", ")
                )
            },
            || async {
                recon_retire(&run.valkey, dead_set, &names)
                    .await
                    .map_err(fred_step_error)
            },
        )
        .await?;
    }
    if let Some(line) = render_lost_object_unregistered(unwiped, foreign) {
        tracing::warn!(removed = ?unwiped, foreign, dead_set, "{line}");
    }
    Ok(())
}

/// A store error of a Job step: retried when transient, else fatal.
fn fred_step_error(e: fred::error::Error) -> StepError {
    if crate::storage::client::is_transient_fred_error(&e) {
        StepError::Retry(e.to_string())
    } else {
        StepError::Fatal(e.to_string())
    }
}

/// Every lost session's discard (PD-6), up to the configured parallelism at
/// once, each logged (AL6) and counted into `report`.
async fn discard_lost_sessions(
    run: &JobRun,
    sessions: &[LostSession],
    memory: &RestoreMemory,
    seams: &ExecSeams,
    report: &mut DiscardReport,
) -> Result<(), JobExit> {
    // By index (see `session::complete_abandoned_deletes`).
    let done: Vec<(usize, super::session::SessionDiscardReport)> =
        futures::stream::iter(0..sessions.len())
            .map(|index| {
                let session = &sessions[index];
                async move {
                    let discard = SessionDiscard {
                        session_id: &session.session_id,
                        objects: &session.objects,
                        layer_names: &session.layer_names,
                        base_path: None,
                        mappings: &session.mappings,
                        path_name_members: &[],
                        delete_backup: session.backup_listed,
                    };
                    let done =
                        discard_session(run, &discard, DiscardEvidence::AcceptLoss, seams).await?;
                    tracing::warn!(
                        session_id = %session.session_id,
                        evidence = %session.evidence,
                        layers = session.layer_names.len(),
                        backup_deleted = done.backup_deleted,
                        "{}",
                        render_lost_session(
                            &session.session_id,
                            &session.evidence,
                            session.layer_names.len(),
                            done.backup_deleted
                        )
                    );
                    Ok::<_, JobExit>((index, done))
                }
            })
            .buffer_unordered(memory.configured_parallelism())
            .try_collect()
            .await?;
    let mut done = done;
    done.sort_by_key(|(index, _)| *index);
    for (index, outcome) in done {
        let session_id = &sessions[index].session_id;
        report.registered += outcome.registered;
        report.session_backups_deleted += usize::from(outcome.backup_deleted);
        report.left.extend(
            outcome
                .left
                .into_iter()
                .map(|(layer, expected, found)| (session_id.clone(), layer, expected, found)),
        );
        report.sessions_discarded.push(session_id.clone());
    }
    Ok(())
}

/// D5: every value removal, up to the restore memory's parallelism at once,
/// each admitted once by its gate, recorded into `report`.
async fn remove_values(
    run: &JobRun,
    removals: &[ValueRemoval],
    memory: &RestoreMemory,
    seams: &ExecSeams,
    report: &mut DiscardReport,
) -> Result<(), JobExit> {
    if removals.is_empty() {
        return Ok(());
    }
    // Once before the first admission: the pages earlier phases freed go
    // back now, not into the first removal's resident memory.
    if let Some(purge) = &run.allocator_purge {
        purge.purge().await;
    }
    let gate = RestoreGate::new(memory, seams.budget_override());
    let gate = &gate;
    // By index (see `session::complete_abandoned_deletes`).
    let done: Vec<(u64, ValueRemovalRecord)> = futures::stream::iter(0..removals.len())
        .map(|index| {
            let removal = &removals[index];
            async move {
                let record = remove_layer_values(run, removal, gate, seams).await?;
                Ok::<_, JobExit>((removal.object_id, record))
            }
        })
        .buffer_unordered(gate.parallelism())
        .try_collect()
        .await?;
    report.values.extend(done);
    Ok(())
}

/// One layer's value removal (module docs, D5).
async fn remove_layer_values(
    run: &JobRun,
    removal: &ValueRemoval,
    gate: &RestoreGate,
    seams: &ExecSeams,
) -> Result<ValueRemovalRecord, JobExit> {
    let mode = run.mode;
    let object_id = removal.object_id;
    let layer = LayerRef {
        object_id,
        session_id: &removal.session_id,
        layer_name: &removal.layer_name,
    };
    let not_live = |why: &str| {
        tracing::error!(
            object_id,
            session_id = %removal.session_id,
            layer_name = %removal.layer_name,
            why,
            "the {mode} Job could not remove values from layer {} of session {} (object \
             {object_id}): {why}; it stops writing and exits 1 so Kubernetes retries it, and the \
             service stays held",
            removal.layer_name,
            removal.session_id
        );
        JobExit::Failed
    };

    // 1. The header's `seq`, and the band at the stream's floor.
    let header = write_step(
        run,
        &|| format!("read the header of object {object_id}"),
        || async {
            read_object_header(&run.valkey, &run.keys, object_id)
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
    let prior_seq = match header {
        Some(h) if h.is_structurally_complete() && !h.is_retiring() => match h.seq() {
            Some(seq) => seq,
            None => return Err(not_live("its header has no seq")),
        },
        Some(h) if h.is_retiring() => return Err(not_live("its header is retiring")),
        Some(_) => return Err(not_live("its header is not structurally complete")),
        None => return Err(not_live("it has no header")),
    };
    let floor_reading = stream_floor(run, object_id, prior_seq).await?;
    let floor = floor_reading.floor(prior_seq);
    let banded = band_for(
        run,
        layer,
        "remove the values in no store from",
        prior_seq,
        floor,
        0,
    )?;
    warn_foreign(object_id, &floor_reading);

    // 2. The one admission, held from before the walk until the header
    // lands (step 6) and given back by its drop on every earlier exit: the
    // purged figure only while a purge is installed and none has failed,
    // else the un-purged fail-safe. The flag is read before the removal
    // waits at the gate and again once it is admitted: a purge that failed
    // during the wait tops the purged figure up to the un-purged one at
    // once, so no removal starts its walk on the purged figure after a purge
    // has failed. The removal never waits on the gate while it holds it: a
    // purge that fails while it holds the purged figure tops up, forced.
    let purge_failed = || {
        run.allocator_purge
            .as_ref()
            .is_some_and(AllocatorPurge::has_failed)
    };
    let purge_works = run.allocator_purge.is_some() && !purge_failed();
    let Some(admission) = gate.admit(value_removal_admission(purge_works)).await else {
        tracing::error!(
            object_id,
            "the {mode} Job's memory budget refused an object; this is an internal defect; \
             it exits 1"
        );
        return Err(JobExit::Failed);
    };
    let purge = RemovalPurge {
        purge: run.allocator_purge.as_ref(),
        admission: &admission,
        purged_charge: purge_works,
        top_up: std::sync::Mutex::new(None),
    };
    if purge_failed() {
        purge.take_failure_top_up();
    }

    // 3-4. The walk, `obj:flds` then `obj:tss` by descending rank pages,
    // each row naming a removed value joining the pending batch, which goes
    // before it would cross either limit; then the rest, and the end counts.
    let oid = object_id.to_string();
    // The one snapshot-lane client both index walks hold for their reads.
    let walk_client = run.valkey.acquire_snapshot().0.next();
    let mut walk = ValueRemovalWalk {
        run,
        seams,
        hashes: &removal.hashes,
        purge: &purge,
        object_id,
        oid: &oid,
        prior_seq,
        page_rows: seams.value_removal_page_rows.max(1),
        limit: BatchLimit {
            commands: seams.batch_commands,
            arg_bytes: RESTORE_BATCH_ARG_BYTES,
        },
        timeout_base: command_timeout(run),
        pages: [
            RankWalk::new(walk_client, &run.keys.obj_fields_index(&oid), true),
            RankWalk::new(walk_client, &run.keys.obj_time_samples_index(&oid), true),
        ],
        time_sample: false,
        batch: ValueRemovalBatch::new(object_id, prior_seq),
        counts: [IndexCounts::default(); 2],
        rows: 0,
        row_batches: 0,
        batches: Vec::new(),
    };
    for time_sample in [false, true] {
        walk.time_sample = time_sample;
        // A page short of its ranks ends the walk here, exit 1, before the
        // `obj:ext` batches and the header.
        walk.walk_index().await?;
    }
    walk.flush_rows().await?;
    // Both indices' end counts, before the first `obj:ext` batch: a walk
    // not shown complete ends here, exit 1, the header never banded and
    // every removed hash still in `obj:ext`.
    walk.check_counts().await?;

    // 5. Every removed hash but the last out of `obj:ext`, in hash order, in
    // batches within the limits, once no row names it.
    let mut ext_hashes: Vec<ExtHash> = removal.hashes.iter().copied().collect();
    let last_hash = ext_hashes.pop();
    for hash in ext_hashes {
        if !walk.batch.fits(1, VALUE_REMOVAL_HDEL_ARG_BYTES, walk.limit) {
            walk.send().await?;
        }
        walk.batch.push_hdel(hash);
    }
    walk.send().await?;
    let ValueRemovalWalk {
        rows,
        mut batches,
        timeout_base,
        ..
    } = walk;
    seams.stop_at(mode, At::ValueRemovalBeforeHeader(object_id))?;

    // 6. The header, last, with the last hash's `HDEL obj:ext`: until it
    // lands that hash stays in `obj:ext`, so a re-run finds the removal
    // unfinished. Purged around like every transaction: as each attempt
    // begins, and once it landed, before the admission is given back.
    let header_write = BandedHeaderWrite::new(object_id, prior_seq, banded.seeded_seq, banded.band)
        .removing_ext_hash(last_hash);
    let purge_ref = &purge;
    write_step(
        run,
        &|| format!("stamp the banded header of object {object_id}"),
        || async {
            purge_ref.run().await;
            match write_value_removal_header(&run.valkey, &run.keys, &header_write).await {
                Ok(SeqFencedOutcome::Committed | SeqFencedOutcome::AlreadyLanded) => Ok(()),
                Ok(SeqFencedOutcome::FenceMismatch { seq, flagged }) => Err(StepError::Fatal(
                    fence_mismatch(prior_seq, seq.as_deref(), flagged),
                )),
                Ok(SeqFencedOutcome::Conflict) => Err(StepError::Retry(
                    "the value removal's header lost its WATCH on the header".to_owned(),
                )),
                Ok(SeqFencedOutcome::Torn(e)) | Ok(SeqFencedOutcome::InDoubt(e)) => {
                    Err(StepError::Retry(e))
                }
                Err(e) => Err(StepError::from(e)),
            }
        },
    )
    .await?;
    purge.run().await;
    // The failure top-up, if any, with the admission it tops up.
    drop(purge);
    drop(admission);
    let header_commands = header_write.commands(BandedFence::NotRetiring);
    batches.push(BatchRecord {
        commands: header_commands,
        timeout: txn_timeout(timeout_base, header_commands),
    });
    tracing::warn!(
        object_id,
        session_id = %removal.session_id,
        layer_name = %removal.layer_name,
        values = removal.hashes.len(),
        seeded_seq = banded.seeded_seq,
        band = banded.band,
        "{}",
        render_value_removal(
            (&removal.session_id, &removal.layer_name, object_id),
            removal.hashes.len(),
            &removal.sites,
            banded.seeded_seq,
            banded.band
        )
    );
    Ok(ValueRemovalRecord {
        hashes: removal.hashes.clone(),
        rows,
        prior_seq,
        floor,
        band: banded.band,
        seeded_seq: banded.seeded_seq,
        batches,
    })
}

/// D5 on one layer, as `discard` runs it, outside a Job (test-support only;
/// the restore memory record's tool measures it per removed row): `run`'s
/// write permit opened, a gate of its own restore memory (`seams`' budget
/// override in place of `R`), `run`'s allocator purge run once (as
/// [`remove_values`] runs it before its first admission), then the removal.
/// `Err` already logged.
#[cfg(any(test, feature = "test-support"))]
pub async fn remove_layer_values_for_test(
    run: &JobRun,
    removal: &ValueRemoval,
    seams: &ExecSeams,
) -> Result<ValueRemovalRecord, JobExit> {
    run.permit.open();
    let memory = RestoreMemory::derive(&run.config, run.mode).map_err(|refusal| {
        tracing::error!(refusal = %refusal, "the value removal's restore memory did not derive");
        JobExit::Failed
    })?;
    let gate = RestoreGate::new(&memory, seams.budget_override());
    if let Some(purge) = &run.allocator_purge {
        purge.purge().await;
    }
    remove_layer_values(run, removal, &gate, seams).await
}

/// The accept-loss Job's line on its value removal's allocator purge, once
/// per run (`run::run_job`, before planning): INFO naming the purged
/// admission when a purge is installed; WARN naming the un-purged fail-safe
/// when none is, since the restore memory then holds fewer removals at once.
pub(crate) fn log_value_removal_purge(mode: JobMode, purge: Option<&AllocatorPurge>) {
    match purge {
        Some(_) => tracing::info!(
            mode = mode.name(),
            admission_bytes = value_removal_admission(true),
            "the {mode} Job's value removal purges the allocator around each of its \
             transactions and admits each removal at {} MiB",
            value_removal_admission(true) / (1024 * 1024)
        ),
        None => tracing::warn!(
            mode = mode.name(),
            admission_bytes = value_removal_admission(false),
            "the {mode} Job runs with no allocator purge installed: its value removal admits \
             each removal at {} MiB, the un-purged figure, so the restore memory holds fewer \
             removals at once",
            value_removal_admission(false) / (1024 * 1024)
        ),
    }
}

/// One removal's allocator purge (module docs, D5): the run's purge, if
/// any, around each of the removal's transactions, and the fail-safe when a
/// purge fails while the removal holds the purged admission.
struct RemovalPurge<'a> {
    /// The run's purge; `None` purges nothing.
    purge: Option<&'a AllocatorPurge>,
    /// The removal's admission, which a failure tops up.
    admission: &'a Admission<'a>,
    /// Whether the admission is the purged figure.
    purged_charge: bool,
    /// The forced top-up from the purged to the un-purged figure, taken at
    /// the removal's first failed purge, or at admission when a purge failed
    /// while the removal waited at the gate, and held until the admission is
    /// given back.
    top_up: std::sync::Mutex<Option<MemoryTopUp>>,
}

impl RemovalPurge<'_> {
    /// Purge (on the blocking pool; nothing without a purge). A purge that
    /// fails while the removal holds the purged figure tops the admission up
    /// to the un-purged one, forced, once; the purge's own handle logs the
    /// one WARN, and later removals are admitted at the un-purged figure.
    async fn run(&self) {
        let Some(purge) = self.purge else {
            return;
        };
        if purge.purge().await || !self.purged_charge {
            return;
        }
        self.take_failure_top_up();
    }

    /// The forced top-up from the purged to the un-purged figure, taken once
    /// and held until the admission is given back; nothing when the removal
    /// was admitted at the un-purged figure, and nothing more once taken.
    fn take_failure_top_up(&self) {
        if !self.purged_charge {
            return;
        }
        let mut held = self
            .top_up
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if held.is_none() {
            *held = self.admission.top_up(
                value_removal_admission(false).saturating_sub(value_removal_admission(true)),
            );
        }
    }
}

/// What D5's walk of one value index counted, for its end check
/// ([`ValueRemovalWalk::check_counts`]).
#[derive(Clone, Copy, Debug, Default)]
struct IndexCounts {
    /// The index's `ZCARD` before its first page.
    at_start: u64,
    /// The members its pages returned.
    returned: u64,
    /// The rows of it this removal deleted (pushed to a batch, each once;
    /// every one has landed by the end check).
    deleted: u64,
}

/// D5's walk of one layer's value indices and its batches (module docs,
/// steps 3–5): descending rank pages over `obj:flds`, then `obj:tss`
/// (`storage::scan_guard::RankWalk`), the pending batch they fill, and each
/// index's counts for the end check.
struct ValueRemovalWalk<'w> {
    run: &'w JobRun,
    seams: &'w ExecSeams,
    /// The values the removal removes.
    hashes: &'w BTreeSet<ExtHash>,
    /// The removal's allocator purge, run around each transaction.
    purge: &'w RemovalPurge<'w>,
    object_id: u64,
    oid: &'w str,
    /// The header's `seq` every batch is fenced on: S.
    prior_seq: u64,
    /// Each page's ranks, and so the most rows read at once.
    page_rows: u64,
    /// The batch limits.
    limit: BatchLimit,
    timeout_base: std::time::Duration,
    /// The rank walks of `obj:flds` then `obj:tss`, on the removal's one
    /// snapshot-lane client, each read with one attempt (its write step
    /// retries it).
    pages: [RankWalk<'w>; 2],
    /// Whether the index being walked is `obj:tss` (else `obj:flds`).
    time_sample: bool,
    /// The pending batch.
    batch: ValueRemovalBatch,
    /// Per index, `obj:flds` then `obj:tss`: what its walk counted.
    counts: [IndexCounts; 2],
    /// The distinct rows committed.
    rows: usize,
    /// The row batches committed.
    row_batches: usize,
    /// Each transaction sent, in order.
    batches: Vec<BatchRecord>,
}

impl ValueRemovalWalk<'_> {
    /// The index being walked, as the step lines name it.
    fn index_name(&self) -> &'static str {
        Self::name_of(self.time_sample)
    }

    /// `obj:tss` or `obj:flds`, as the step lines name it.
    fn name_of(time_sample: bool) -> &'static str {
        if time_sample {
            "obj:tss"
        } else {
            "obj:flds"
        }
    }

    /// The rank walk of the index `obj:tss` (`time_sample`) or `obj:flds`.
    fn pages_of(&self, time_sample: bool) -> &RankWalk<'_> {
        &self.pages[usize::from(time_sample)]
    }

    /// The counts of the index `obj:tss` (`time_sample`) or `obj:flds`.
    fn counts_mut(&mut self, time_sample: bool) -> &mut IndexCounts {
        &mut self.counts[usize::from(time_sample)]
    }

    /// Walk the index being walked (module docs, D5 steps 3–4): its `ZCARD`
    /// n, then pages of `page_rows` ranks by descending absolute rank,
    /// `[n − page_rows, n − 1]` and down to rank 0, each page's rows taken
    /// ([`Self::take_page`]) before the next is read. Each read is one write
    /// step. A page that answers a number of members other than its ranks
    /// span ends the walk with one ERROR, exit 1: the store changed while the
    /// Job read it.
    async fn walk_index(&mut self) -> Result<(), JobExit> {
        let time_sample = self.time_sample;
        let at_start = self.read_card(time_sample).await?;
        self.counts_mut(time_sample).at_start = at_start;
        // Ranks below `above` are still to be read.
        let mut above = at_start;
        while above > 0 {
            let start = above.saturating_sub(self.page_rows);
            let stop = above - 1;
            let members = self.read_page(time_sample, start, stop).await?;
            let returned = members.len() as u64;
            if returned != stop - start + 1 {
                self.log_incomplete(&WalkIncomplete::ShortPage {
                    target: Arc::clone(self.pages_of(time_sample).key()),
                    start,
                    stop,
                    returned,
                });
                return Err(JobExit::Failed);
            }
            self.counts_mut(self.time_sample).returned += returned;
            self.take_page(members).await?;
            above = start;
        }
        Ok(())
    }

    /// The end check of both index walks, after the last row batch landed
    /// and before the first `obj:ext` batch (module docs, D5 step 4): each
    /// index's `ZCARD` again, one write step each; complete only when its
    /// pages returned the start `ZCARD` and the end `ZCARD` is the start
    /// less the rows this removal deleted from it
    /// (`storage::scan_guard::check_count`). Otherwise one ERROR, exit 1.
    async fn check_counts(&self) -> Result<(), JobExit> {
        for time_sample in [false, true] {
            let at_end = self.read_card(time_sample).await?;
            let counts = self.counts[usize::from(time_sample)];
            if let Err(WalkError::Incomplete(incomplete)) = check_count(
                "ZRANGE",
                self.pages_of(time_sample).key(),
                "ZCARD",
                counts.at_start,
                counts.returned,
                at_end,
                counts.at_start.saturating_sub(counts.deleted),
            ) {
                self.log_incomplete(&incomplete);
                return Err(JobExit::Failed);
            }
        }
        Ok(())
    }

    /// The `ZCARD` of the index `obj:tss` (`time_sample`) or `obj:flds` on
    /// the walk's client: one write step, retried in place.
    async fn read_card(&self, time_sample: bool) -> Result<u64, JobExit> {
        let pages = self.pages_of(time_sample);
        let (run, object_id, key) = (self.run, self.object_id, pages.key());
        write_step(
            run,
            &|| format!("count the members of object {object_id}'s index {key}"),
            || async {
                let (_pool, _in_use) = run.valkey.acquire_snapshot();
                pages.card().await.map_err(|e| self.walk_step_error(e))
            },
        )
        .await
    }

    /// The members at ranks `start..=stop` of the index `obj:tss`
    /// (`time_sample`) or `obj:flds` on the walk's client: one write step,
    /// retried in place.
    async fn read_page(
        &self,
        time_sample: bool,
        start: u64,
        stop: u64,
    ) -> Result<Vec<String>, JobExit> {
        let pages = self.pages_of(time_sample);
        let (run, object_id, key) = (self.run, self.object_id, pages.key());
        write_step(
            run,
            &|| format!("read object {object_id}'s index {key} at ranks {start} to {stop}"),
            || async {
                let (_pool, _in_use) = run.valkey.acquire_snapshot();
                let members = pages
                    .page(start, stop)
                    .await
                    .map_err(|e| self.walk_step_error(e))?;
                Value::Array(members)
                    .convert::<Vec<String>>()
                    .map_err(|e| StepError::Fatal(e.to_string()))
            },
        )
        .await
    }

    /// Read `members`' rows at once, each by the readers' rule
    /// (`read_value_row_external_hash`: a row holding a `value` names no
    /// hash, whatever else it holds; a malformed `hash` is `Internal`, so the
    /// step fails, exit 1): each row's hash, in `members`' order. One write
    /// step, retried in place.
    async fn read_rows(&self, members: &[String]) -> Result<Vec<Option<ExtHash>>, JobExit> {
        let (run, oid, object_id, time_sample) =
            (self.run, self.oid, self.object_id, self.time_sample);
        let family = if time_sample {
            SnapshotKeyFamily::TimeSample
        } else {
            SnapshotKeyFamily::Field
        };
        let index = self.index_name();
        write_step(
            run,
            &|| {
                format!(
                    "read {} value rows of object {object_id}'s {index} index",
                    members.len()
                )
            },
            || async {
                let (pool, _in_use) = run.valkey.acquire_snapshot();
                let reads = members.iter().map(|member| {
                    let key = if time_sample {
                        run.keys.obj_ts_from_member(oid, member)
                    } else {
                        run.keys.obj_field_from_member(oid, member)
                    };
                    read_value_row_external_hash(&run.valkey, pool, oid, family, key)
                });
                let read = futures::future::try_join_all(reads)
                    .await
                    .map_err(StepError::from)?;
                Ok(read.into_iter().map(|(hash, _received)| hash).collect())
            },
        )
        .await
    }

    /// Send the pending batch, if it holds anything, fenced on `seq == S`
    /// and not retiring: one write step, retried in place (every command is
    /// idempotent), recorded. The batch is emptied once it landed. The
    /// allocator is purged around it ([`RemovalPurge`]): as each attempt
    /// begins, so a retry does not rebuild its frame beside the last
    /// attempt's freed one, and once it landed.
    async fn send(&mut self) -> Result<(), JobExit> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let (run, object_id, prior_seq) = (self.run, self.object_id, self.prior_seq);
        let (batch, number, purge) = (&self.batch, self.batches.len() + 1, self.purge);
        write_step(
            run,
            &|| format!("remove values from object {object_id}, batch {number}"),
            || async {
                purge.run().await;
                match remove_value_rows(&run.valkey, &run.keys, batch).await {
                    Ok(SeqFencedOutcome::Committed) => Ok(()),
                    Ok(SeqFencedOutcome::FenceMismatch { seq, flagged }) => Err(StepError::Fatal(
                        fence_mismatch(prior_seq, seq.as_deref(), flagged),
                    )),
                    Ok(SeqFencedOutcome::AlreadyLanded) => Err(StepError::Fatal(
                        "a value-removal batch answered already landed".to_owned(),
                    )),
                    Ok(SeqFencedOutcome::Conflict) => Err(StepError::Retry(
                        "the value removal lost its WATCH on the header".to_owned(),
                    )),
                    Ok(SeqFencedOutcome::Torn(e)) | Ok(SeqFencedOutcome::InDoubt(e)) => {
                        Err(StepError::Retry(e))
                    }
                    Err(e) => Err(StepError::from(e)),
                }
            },
        )
        .await?;
        let commands = self.batch.commands();
        self.batches.push(BatchRecord {
            commands,
            timeout: txn_timeout(self.timeout_base, commands),
        });
        self.batch.clear();
        // Landed: its frame and replies go back before anything follows it
        // (the next read, the kill point `ValueRowsAfterBatch`, the header).
        self.purge.run().await;
        Ok(())
    }

    /// Send the pending row batch, if any: its rows count as deleted, once
    /// each (the batch dedupes them), and the kill point
    /// [`At::ValueRowsAfterBatch`] follows it.
    async fn flush_rows(&mut self) -> Result<(), JobExit> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let rows = self.batch.rows();
        self.send().await?;
        self.rows += rows;
        let batch = self.row_batches;
        self.row_batches += 1;
        self.seams.stop_at(
            self.run.mode,
            At::ValueRowsAfterBatch(self.object_id, batch),
        )
    }
}

impl ValueRemovalWalk<'_> {
    /// A rank walk's read error as one attempt of a write step: a failed
    /// `ZRANGE` page or `ZCARD` read (a `Step` error, sent with one attempt)
    /// is retried in place when transient, so the write step's attempts are
    /// the read's bound, and fails the step at once, exit 1, when
    /// deterministic (a `WRONGTYPE`, a reply of another shape). A `-MOVED`
    /// reaches a one-attempt read as fred's attempt-budget exhaustion, which
    /// counts as transient and is retried. A fred error is counted on the
    /// snapshot lane. A rank walk's reads make no other error; one would fail
    /// the step, exit 1.
    fn walk_step_error(&self, e: WalkError) -> StepError {
        if let Some(source) = e.fred_error() {
            FredErrOrigin::client(&self.run.valkey, POOL_LABEL_SNAPSHOT).observe_error(source);
        }
        match e {
            WalkError::Step { ref source, .. } => {
                if crate::storage::client::is_transient_fred_error(source) {
                    StepError::Retry(e.to_string())
                } else {
                    StepError::Fatal(e.to_string())
                }
            }
            WalkError::Read { source, .. } => fred_step_error(source),
            WalkError::Incomplete(incomplete) => {
                self.log_incomplete(&incomplete);
                StepError::Exit(JobExit::Failed)
            }
            other @ (WalkError::Changed(_) | WalkError::WitnessMissing(_)) => {
                StepError::Fatal(other.to_string())
            }
        }
    }

    /// The ERROR of a walk its completeness check did not show complete (a
    /// short page, or an end `ZCARD` other than a complete walk leaves): it
    /// says plainly that the store changed while the Job read it, names the
    /// index and the counts, and carries no attempt count, since no step's
    /// attempts ran out. The walk then exits 1, before the `obj:ext` batches
    /// and the header.
    fn log_incomplete(&self, incomplete: &WalkIncomplete) {
        let (mode, object_id) = (self.run.mode, self.object_id);
        tracing::error!(
            object_id,
            error = %incomplete,
            "the {mode} Job could not remove values from object {object_id}: the backing store \
             changed while the Job read the layer's value indices ({incomplete}); it stops \
             writing before it touches the layer's external-value index or its header, and \
             exits 1 so Kubernetes retries it, and the service stays held"
        );
    }
}

impl ValueRemovalWalk<'_> {
    /// One page's members, its rows read at once: each row naming a removed
    /// value joins the pending batch, the batch sent first when the row
    /// would cross a limit, and counts as deleted from the index being
    /// walked. A rank page returns each member once, and no member it
    /// returns was in an earlier page, so a row joins once; a row already in
    /// the pending batch is not added or counted again all the same.
    async fn take_page(&mut self, members: Vec<String>) -> Result<(), JobExit> {
        if members.is_empty() {
            return Ok(());
        }
        let hashes = self.read_rows(&members).await?;
        for (member, hash) in members.into_iter().zip(hashes) {
            if !hash.is_some_and(|hash| self.hashes.contains(&hash)) {
                continue;
            }
            let row = RemovedRow {
                time_sample: self.time_sample,
                member,
            };
            if self.batch.contains(&row) {
                continue;
            }
            if !self.batch.fits(2, row.arg_bytes(), self.limit) {
                self.flush_rows().await?;
            }
            self.batch.push_row(row);
            self.counts_mut(self.time_sample).deleted += 1;
        }
        Ok(())
    }
}

/// A value removal's fence mismatch, for the step's ERROR.
fn fence_mismatch(expected: u64, seq: Option<&str>, flagged: bool) -> String {
    format!(
        "its header reads seq {} with the retirement flags {}, not the live object at seq \
         {expected} the Job read",
        seq.unwrap_or("(absent)"),
        if flagged { "set" } else { "clear" }
    )
}

/// Which of the discards' two session-backup rewrites a
/// [`rewrite_sessions`] call is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rewrite {
    /// The over-limit layers' sessions, each without its over-limit layers,
    /// ahead of D3's unmap (module docs, D3).
    AheadOfUnmap,
    /// D6, after the unmaps: the test kill point
    /// [`At::SessionRewriteBefore`] is checked before each session's.
    AfterUnmaps,
}

/// Rewrite every session backup in `sessions` from the store (with a
/// backup store; none without), up to the configured parallelism at once,
/// each without the layers its map names when the store maps them to the
/// object named (`rewrite_session_checkpoint`'s `leave_out`). Returns the
/// sessions rewritten; one with no record is skipped. Each rewrite is one
/// write unit (its build mints the session's stamp, then its PUT), never
/// retried by the Job.
async fn rewrite_sessions(
    run: &JobRun,
    sessions: &BTreeMap<String, BTreeMap<String, u64>>,
    memory: &RestoreMemory,
    seams: &ExecSeams,
    which: Rewrite,
) -> Result<Vec<String>, JobExit> {
    let BackupAccess::Store(store) = &run.backup else {
        return Ok(Vec::new());
    };
    let mode = run.mode;
    let sessions: Vec<(&String, &BTreeMap<String, u64>)> = sessions.iter().collect();
    let sessions = &sessions;
    let rewritten: Vec<Option<String>> = futures::stream::iter(0..sessions.len())
        .map(|index| {
            let (session_id, leave_out) = sessions[index];
            let session_id = session_id.as_str();
            async move {
                if which == Rewrite::AfterUnmaps {
                    seams.stop_at(mode, At::SessionRewriteBefore(session_id))?;
                }
                if run.permit.begin_unit().await.is_err() {
                    return Err(JobExit::Failed);
                }
                match rewrite_session_checkpoint(
                    &run.valkey,
                    &run.keys,
                    store.as_ref(),
                    session_id,
                    &run.lease_holder,
                    leave_out,
                )
                .await
                {
                    Ok(SessionRewrite::Rewritten) => Ok(Some(session_id.to_owned())),
                    Ok(SessionRewrite::SessionAbsent) => Ok(None),
                    Err(e) => {
                        step_failed(
                            mode,
                            &|| match which {
                                Rewrite::AfterUnmaps => {
                                    format!("rewrite the session backup of session {session_id}")
                                }
                                Rewrite::AheadOfUnmap => format!(
                                    "rewrite the session backup of session {session_id} without \
                                     layers {}, whose backups are over the per-delta entry limit",
                                    leave_out.keys().cloned().collect::<Vec<_>>().join(", ")
                                ),
                            },
                            &e.to_string(),
                            1,
                        );
                        Err(JobExit::Failed)
                    }
                }
            }
        })
        .buffer_unordered(memory.configured_parallelism())
        .try_collect()
        .await?;
    let mut rewritten: Vec<String> = rewritten.into_iter().flatten().collect();
    rewritten.sort();
    Ok(rewritten)
}

#[cfg(test)]
mod tests {
    //! What the discards make of a store that moved under the lost set, or
    //! of a step no integration fixture reaches, against a real store with
    //! the write permit open, on a lost set built by hand: P51-d and P55-a
    //! (no well-formed plan carries either case, since the planner holds
    //! neither a stale mapping (defer-51) nor a layer whose header is
    //! complete), and P55-a's late stop at D2 on an object another writer
    //! had already registered as deleted; D5's refusal of a header that is
    //! not live, its admission over the budget, its one admission (held
    //! until its header lands, and given back on each early exit, one test
    //! per exit), and its fence when the header's `seq` moves between its
    //! read and its batch; D5's walk over value rows on hand-built objects
    //! (the rows it deletes on table-encoded indices at several pages and
    //! batch sizes, against the report's oracle, the order of its writes,
    //! each row counted once, and the walk's completeness checks: a page
    //! that answers short of its ranks and an end `ZCARD` a lost delete on
    //! either index moved, each stopping it before `obj:ext` and the header,
    //! and a store that reports no usable run_id, which no longer stops it;
    //! a rank read failed transiently, retried in place up to its write
    //! step's attempts, and one answering `WRONGTYPE`, failed on its first
    //! attempt); D5's
    //! allocator purge (the admission it selects, its cadence,
    //! the top-up when a purge fails, at the failure and at a later
    //! admission, on a scripted purge, and the run's purge line in both
    //! arms); a lost session's unmap that finds another object; D6 for a
    //! session with no record; D3's early rewrite of an over-limit layer's
    //! session failing (no integration fixture fails a session backup's
    //! PUT); and D6 for an over-limit layer's session whose unmap found
    //! another object. The discards end to end are the accept-loss
    //! integration tests' (`tests/integration/dr/accept_loss_test.rs`).
    //!
    //! Not covered: the purge a retried D5 attempt runs. The pre-attempt
    //! purge sits inside each row batch's, `obj:ext` batch's and the
    //! header's retried write-step closure, so an attempt retried after a
    //! WATCH conflict, a torn or in-doubt `EXEC` or a transient store error
    //! purges again (`1 + 2 × transactions + retries` in all). The cadence
    //! test counts only runs in which nothing retries, so a purge moved just
    //! outside the write step would pass it. Pinning it needs a store that
    //! fails one D5 transaction on demand, and no seam does that: the
    //! `ExecSeams` kill points stop the run rather than retry a step, the
    //! purge runs before the transaction's `WATCH` so a hook cannot stage a
    //! conflict, and `test-fault-injection` gates nothing on this path.

    use std::future::Future;
    use std::pin::{pin, Pin};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};

    use super::super::memory::{
        DR_VALUE_REMOVAL_ADMISSION_BYTES, DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES,
    };
    use super::super::test_harness::{open_run, LogCapture};
    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::dr::job::plan::lost::LostLayerKind;
    use crate::dr::job::JobMode;
    use crate::storage::keys::{hex_encode, sha256_to_hex, KeySchema};
    use crate::storage::test_support::TestValkey;

    const SESSION: &str = "s-moved";
    const LAYER: &str = "l.usda";

    /// An accept-loss run on `tv`, its permit open.
    async fn accept_loss_run(tv: &TestValkey) -> JobRun {
        let mut run = open_run(tv, Arc::new(InMemoryCheckpointStore::new())).await;
        run.mode = JobMode::AcceptLoss;
        run
    }

    /// [`LAYER`] of [`SESSION`] mapped to `object_id`, with its layers-index
    /// member.
    async fn map(tv: &TestValkey, object_id: u64) {
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().layer_mapping(SESSION, LAYER),
                object_id.to_string(),
                None,
                None,
                false,
            )
            .await
            .expect("SET mapping");
        let _: i64 = tv
            .client()
            .pool
            .sadd(tv.key_schema().layers_index(SESSION), hex_encode(LAYER))
            .await
            .expect("SADD");
    }

    /// The object [`LAYER`] is mapped to, and whether the index names it.
    async fn mapped(tv: &TestValkey) -> (Option<String>, bool) {
        let object: Option<String> = tv
            .client()
            .pool
            .get(tv.key_schema().layer_mapping(SESSION, LAYER))
            .await
            .expect("GET");
        let indexed: bool = tv
            .client()
            .pool
            .sismember(tv.key_schema().layers_index(SESSION), hex_encode(LAYER))
            .await
            .expect("SISMEMBER");
        (object, indexed)
    }

    /// The members of the dead-object set.
    async fn dead(tv: &TestValkey) -> Vec<String> {
        tv.client()
            .pool
            .zrange(
                tv.key_schema().checkpoint_reconcile_dead(),
                0,
                -1,
                None,
                false,
                None,
                false,
            )
            .await
            .expect("ZRANGE")
    }

    /// A lost set of one lost layer, [`LAYER`] → `object_id`, of `kind`.
    fn one_lost_layer(object_id: u64, kind: LostLayerKind) -> LostSet {
        LostSet {
            layers: vec![LostLayer {
                session_id: SESSION.to_owned(),
                layer_name: LAYER.to_owned(),
                object_id,
                kind,
                why: "it needs a backup and the backup store lists none".to_owned(),
                unconfirmed: None,
            }],
            ..LostSet::default()
        }
    }

    /// P51-d (fail-on-old). A stale lost layer — its plan saw `L → X` (a
    /// mapping to a deleted object) while the store now maps `L → Y` — is
    /// left as found: the unmap answers `Other(Y)`, `L → Y` and its index
    /// member stay, the report counts it as left and not discarded, one AL8
    /// WARN names both objects, and the discards run on (no rewrite: it
    /// unmapped nothing). With `Other` treated as removed, the report
    /// counts it discarded and no AL8 is logged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn p51d_an_unmap_that_finds_another_object_leaves_the_mapping_and_logs_al8() {
        const STALE: u64 = 8_201;
        const NEWER: u64 = 8_202;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        map(&tv, NEWER).await;
        let lost = one_lost_layer(STALE, LostLayerKind::MappedToDeleted);
        let capture = LogCapture::default();
        let guard = capture.install();

        let report = discard(&run, &lost, &ExecSeams::production()).await;
        drop(guard);

        let report = report.expect("the discards run on");
        assert_eq!(
            report.left,
            vec![(SESSION.to_owned(), LAYER.to_owned(), STALE, NEWER)]
        );
        assert!(report.layers_discarded.is_empty(), "{report:?}");
        assert!(
            report.rewritten.is_empty(),
            "nothing unmapped, nothing rewritten"
        );
        assert_eq!(
            mapped(&tv).await,
            (Some(NEWER.to_string()), true),
            "L → Y stays"
        );
        let left: Vec<String> = capture
            .messages("WARN")
            .into_iter()
            .filter(|m| m.starts_with("the accept-loss Job left layer"))
            .collect();
        assert_eq!(
            left,
            vec![render_left_mapping(SESSION, LAYER, NEWER, STALE)],
            "one AL8"
        );
    }

    /// P55-a (B1; fail-on-old). A lost layer with no survivor whose header
    /// the store holds complete before the discards (another writer is
    /// active): one AL9 ERROR, exit 1, and nothing is written — the header
    /// and the mapping as they were, the object not on the dead-object set,
    /// and no WARN: D0 reads the header before D1 registers anything, so
    /// there is no registration for D2's stop to remove and no companion
    /// line. With D0's loop moved below D1's registration, the object is on
    /// the dead-object set; with D0's loop removed, D2's re-read stops the
    /// run instead, after registering the object and removing it again,
    /// and logs AL9's companion WARN.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn p55a_a_lost_layer_found_complete_at_its_header_reread_stops_with_al9() {
        const OBJECT: u64 = 8_301;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        let header_key = tv.key_schema().obj_header(&OBJECT.to_string());
        let _: i64 = tv
            .client()
            .pool
            .hset(
                &header_key,
                [
                    ("seq", "12"),
                    ("created_by", "user"),
                    ("session_id", SESSION),
                    ("layer_name", LAYER),
                ],
            )
            .await
            .expect("HSET header");
        map(&tv, OBJECT).await;
        let header = || async {
            let fields: std::collections::HashMap<String, String> = tv
                .client()
                .pool
                .hgetall(&header_key)
                .await
                .expect("HGETALL");
            fields
        };
        let before = header().await;
        let lost = one_lost_layer(OBJECT, LostLayerKind::NoSurvivor);
        let capture = LogCapture::default();
        let guard = capture.install();

        let result = discard(&run, &lost, &ExecSeams::production()).await;
        drop(guard);

        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(
            capture.messages("ERROR"),
            vec![render_lost_object_complete(&lost.layers[0])],
            "one AL9"
        );
        assert_eq!(header().await, before, "the header is not wiped");
        assert_eq!(
            mapped(&tv).await,
            (Some(OBJECT.to_string()), true),
            "the mapping is kept"
        );
        assert_eq!(dead(&tv).await, Vec::<String>::new(), "nothing registered");
        assert_eq!(
            capture.messages("WARN"),
            Vec::<String>::new(),
            "no WARN: D0 stopped the run, not D2"
        );
    }

    /// P55-a in the window D0 does not cover, a registration another writer
    /// made (B1; fail-on-old). The lost layer's object is already on the
    /// dead-object set when the discards start, and another writer completes
    /// its header after D1's registration (the after-register seam): D2's
    /// re-read logs AL9 and exits 1, and the object stays on the dead-object
    /// set, since D1's read found it there, so this run's registration did
    /// not add it. The companion WARN says that nothing was removed and why.
    /// With D1 taking every member as added (its read ignored), D2 removes
    /// the object from the set.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn p55a_late_al9_keeps_a_registration_another_writer_made_before_the_run() {
        const OBJECT: u64 = 8_311;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        map(&tv, OBJECT).await;
        let _: i64 = tv
            .client()
            .pool
            .zadd(
                tv.key_schema().checkpoint_reconcile_dead(),
                None,
                None,
                false,
                false,
                (0.0, OBJECT.to_string()),
            )
            .await
            .expect("the other writer's ZADD");
        let registered: Arc<std::sync::Mutex<Vec<Vec<u64>>>> = Arc::default();
        let seams = ExecSeams {
            after_discard_register: Some(super::super::AfterDiscardRegister({
                let client = tv.client().clone();
                let key = tv.key_schema().obj_header(&OBJECT.to_string());
                let registered = Arc::clone(&registered);
                Arc::new(move |objects: &[u64]| {
                    registered.lock().unwrap().push(objects.to_vec());
                    let (client, key) = (client.clone(), key.clone());
                    crate::storage::test_support::block_on_spawned(
                        &tokio::runtime::Handle::current(),
                        async move {
                            let _: i64 = client
                                .pool
                                .hset(
                                    key,
                                    [
                                        ("seq", "12"),
                                        ("created_by", "user"),
                                        ("session_id", SESSION),
                                        ("layer_name", LAYER),
                                    ],
                                )
                                .await
                                .expect("the other writer's HSET");
                        },
                    )
                    .expect("the other writer's write ran");
                })
            })),
            ..ExecSeams::production()
        };
        let lost = one_lost_layer(OBJECT, LostLayerKind::NoSurvivor);
        let capture = LogCapture::default();
        let guard = capture.install();

        let result = discard(&run, &lost, &seams).await;
        drop(guard);

        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(
            *registered.lock().unwrap(),
            vec![vec![OBJECT]],
            "the other writer completed the header after D1"
        );
        assert_eq!(
            capture.messages("ERROR"),
            vec![render_lost_object_complete(&lost.layers[0])],
            "one AL9"
        );
        assert_eq!(
            capture.messages("WARN"),
            vec![format!(
                "the accept-loss Job removed nothing from the dead-object set before exiting: \
                 object {OBJECT} stays on it: another writer registered it as deleted before \
                 this run did"
            )],
            "AL9's companion"
        );
        assert_eq!(
            dead(&tv).await,
            vec![OBJECT.to_string()],
            "the other writer's registration stays"
        );
        assert_eq!(
            mapped(&tv).await,
            (Some(OBJECT.to_string()), true),
            "the mapping is kept"
        );
    }

    // ── D5, D6 and a lost session's unmap on hand-built lost sets ───────

    /// The values the hand-built removals remove, in hash order: [`H0`]'s
    /// `HDEL obj:ext` goes in a batch after every row batch, [`H1`]'s, the
    /// last, rides the header transaction.
    const H0: ExtHash = [0xD0; 32];
    const H1: ExtHash = [0xD1; 32];

    /// A live header of [`LAYER`] in [`SESSION`] at `seq`.
    fn live_header(seq: &str) -> Vec<(&str, &str)> {
        vec![
            ("seq", seq),
            ("created_by", "user"),
            ("session_id", SESSION),
            ("layer_name", LAYER),
        ]
    }

    /// `object_id`'s header with `fields` (none: no header), and its
    /// `obj:ext` naming [`H0`], [`H1`] and `extra` more hashes no removal
    /// names.
    async fn seed_value_object(
        tv: &TestValkey,
        object_id: u64,
        fields: &[(&str, &str)],
        extra: u8,
    ) {
        let oid = object_id.to_string();
        if !fields.is_empty() {
            let _: i64 = tv
                .client()
                .pool
                .hset(tv.key_schema().obj_header(&oid), fields.to_vec())
                .await
                .expect("HSET header");
        }
        let ext: Vec<(String, String)> = [H0, H1]
            .into_iter()
            .chain((0..extra).map(|i| [i; 32]))
            .map(|h| (sha256_to_hex(&h), "1".to_owned()))
            .collect();
        let _: i64 = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_ext_index(&oid), ext)
            .await
            .expect("HSET obj:ext");
    }

    /// `object_id`'s header and whether its `obj:ext` names [`H0`] and
    /// [`H1`].
    async fn value_object(
        tv: &TestValkey,
        object_id: u64,
    ) -> (std::collections::HashMap<String, String>, bool, bool) {
        let oid = object_id.to_string();
        let header: std::collections::HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(tv.key_schema().obj_header(&oid))
            .await
            .expect("HGETALL");
        let named = |hash: ExtHash| {
            let key = tv.key_schema().obj_ext_index(&oid);
            async move {
                let named: bool = tv
                    .client()
                    .pool
                    .hexists(key, sha256_to_hex(&hash))
                    .await
                    .expect("HEXISTS");
                named
            }
        };
        (header, named(H0).await, named(H1).await)
    }

    /// A lost set removing [`H0`] and [`H1`] from each of `objects`.
    fn value_removals(objects: &[u64]) -> LostSet {
        LostSet {
            values: objects
                .iter()
                .map(|&object_id| ValueRemoval {
                    object_id,
                    session_id: SESSION.to_owned(),
                    layer_name: LAYER.to_owned(),
                    hashes: BTreeSet::from([H0, H1]),
                    unconfirmed: BTreeSet::new(),
                    sites: Vec::new(),
                })
                .collect(),
            ..LostSet::default()
        }
    }

    /// The ERROR D5 logs for a layer it cannot remove values from.
    fn not_live(object_id: u64, why: &str) -> String {
        format!(
            "the accept-loss Job could not remove values from layer {LAYER} of session {SESSION} \
             (object {object_id}): {why}; it stops writing and exits 1 so Kubernetes retries \
             it, and the service stays held"
        )
    }

    /// Test review 7 (iv) (fail-on-old). D5 removes values only from a live
    /// object: a header that is retiring, absent or not structurally
    /// complete makes it log one ERROR saying which and exit 1, writing
    /// nothing — the header as it was and both values still in `obj:ext`.
    /// With a retiring header taken as live, the retiring case's ERROR no
    /// longer says it is retiring.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_refuses_a_retiring_absent_or_incomplete_header_and_writes_nothing() {
        const OBJECT: u64 = 8_401;
        let mut retiring = live_header("12");
        retiring.push(("superseded", "1"));
        type Fields<'a> = Vec<(&'a str, &'a str)>;
        let cases: [(&str, Fields<'_>, &str); 3] = [
            ("retiring", retiring, "its header is retiring"),
            ("absent", Vec::new(), "it has no header"),
            (
                "incomplete",
                vec![("seq", "12")],
                "its header is not structurally complete",
            ),
        ];
        for (case, fields, why) in cases {
            let tv = TestValkey::setup().await;
            let run = accept_loss_run(&tv).await;
            seed_value_object(&tv, OBJECT, &fields, 0).await;
            let before = value_object(&tv, OBJECT).await;
            let capture = LogCapture::default();
            let guard = capture.install();

            let result = discard(&run, &value_removals(&[OBJECT]), &ExecSeams::production()).await;
            drop(guard);

            assert_eq!(result, Err(JobExit::Failed), "{case}");
            assert_eq!(
                capture.messages("ERROR"),
                vec![not_live(OBJECT, why)],
                "{case}"
            );
            assert_eq!(
                value_object(&tv, OBJECT).await,
                before,
                "{case}: nothing written"
            );
            assert!(before.1 && before.2, "{case}: fixture: both values named");
        }
    }

    /// An accept-loss run on `tv` whose restore memory is in bytes mode: a
    /// fixed 5 GiB pod memory limit, so a budget of 2,560 MiB.
    async fn bytes_mode_run(tv: &TestValkey) -> JobRun {
        let mut run = accept_loss_run(tv).await;
        let mut config = (*run.config).clone();
        config.pod_memory = crate::config::PodMemoryPlan::derive(
            crate::config::PodMemorySource::Fixed(5_120 * 1024 * 1024).resolve(),
            &crate::config::PlanInputs::from_config(&config),
        )
        .expect("the server's plan closes at 5 GiB");
        run.config = Arc::new(config);
        assert!(
            matches!(
                RestoreMemory::derive(&run.config, run.mode),
                Ok(RestoreMemory::Bytes { .. })
            ),
            "fixture: bytes mode, so the budget applies"
        );
        run
    }

    /// Test review 7 (iii) (fail-on-old). D5's one admission per removal
    /// (the purged figure, `value_removal_admission(true)`: the fixture's
    /// purge works) is taken from the restore
    /// memory under the budget (bytes mode, the budget seam at 1 KiB): two
    /// layers each charged more than the whole budget are each admitted
    /// rather than refused — the gate clamps a charge to the whole budget,
    /// so such a layer runs alone, which this test does not observe — and
    /// both removals complete within 30 s: every removed value gone from
    /// `obj:ext`, each header banded, no ERROR. A removal that asked the
    /// gate again while holding the whole budget would wait on itself
    /// forever, and the 30 s bound fails it. With the admission refusing,
    /// the run logs the memory budget's refusal and exits 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_admits_a_layer_charged_over_the_budget_and_completes() {
        const OBJECTS: [u64; 2] = [8_501, 8_502];
        let tv = TestValkey::setup().await;
        let run = bytes_mode_run(&tv).await;
        for object_id in OBJECTS {
            seed_value_object(&tv, object_id, &live_header("12"), 8).await;
        }
        let seams = ExecSeams {
            budget_override: Some(1_024),
            ..ExecSeams::production()
        };
        let capture = LogCapture::default();
        let guard = capture.install();

        let report = tokio::time::timeout(
            Duration::from_secs(30),
            discard(&run, &value_removals(&OBJECTS), &seams),
        )
        .await
        .expect("no removal waits on its own admission");
        drop(guard);

        let report = report.expect("both removals complete");
        assert_eq!(capture.messages("ERROR"), Vec::<String>::new());
        assert_eq!(
            report.values.keys().copied().collect::<Vec<_>>(),
            OBJECTS.to_vec()
        );
        for object_id in OBJECTS {
            let (header, h0, h1) = value_object(&tv, object_id).await;
            assert_eq!((h0, h1), (false, false), "object {object_id}: both removed");
            assert_eq!(
                header["seq"],
                report.values[&object_id].seeded_seq.to_string(),
                "object {object_id}: banded"
            );
        }
    }

    /// Test review 4, the fence between D5's read and its batches
    /// (fail-on-old). The header's `seq` moves after D5 read it and before
    /// its first batch (another writer, simulated when D5 logs its foreign
    /// stream WARN, which falls between the two): the batch's fence answers
    /// a mismatch, D5 logs the write step's ERROR naming both `seq`s and
    /// exits 1, and nothing more is written — both values still in
    /// `obj:ext`, the header as the other writer left it, the foreign
    /// stream key kept. With the batches' `seq` check dropped, the first
    /// batch (here its `obj:ext` batch: no row names a removed value)
    /// removes a value over the moved header.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_stops_when_the_header_seq_moves_between_its_read_and_its_batch() {
        const OBJECT: u64 = 8_601;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_value_object(&tv, OBJECT, &live_header("12"), 0).await;
        let stream_key = tv.key_schema().layer_stream(OBJECT);
        let _: () = tv
            .client()
            .pool
            .set(&stream_key, "not a stream", None, None, false)
            .await
            .expect("SET the foreign stream key");
        let capture = LogCapture::default();
        let writer = OnLine {
            capture: capture.clone(),
            trigger: "stream key held a",
            fired: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            action: {
                let client = tv.client().clone();
                let key = tv.key_schema().obj_header(&OBJECT.to_string());
                Arc::new(move || {
                    let (client, key) = (client.clone(), key.clone());
                    crate::storage::test_support::block_on_spawned(
                        &tokio::runtime::Handle::current(),
                        async move {
                            let _: i64 = client
                                .pool
                                .hset(key, [("seq", "13")])
                                .await
                                .expect("the other writer's HSET");
                        },
                    )
                    .expect("the other writer's write ran");
                })
            },
        };
        let guard = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .json()
                .with_writer(writer.clone())
                .with_max_level(tracing::Level::INFO)
                .finish(),
        );

        let result = discard(&run, &value_removals(&[OBJECT]), &ExecSeams::production()).await;
        drop(guard);

        assert_eq!(result, Err(JobExit::Failed));
        assert!(
            writer.fired.load(std::sync::atomic::Ordering::SeqCst),
            "the other writer ran between the read and the batch"
        );
        let errors = capture.messages("ERROR");
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].contains(
                "its header reads seq 13 with the retirement flags clear, not the live object \
                 at seq 12 the Job read"
            ),
            "{errors:?}"
        );
        let (header, h0, h1) = value_object(&tv, OBJECT).await;
        assert_eq!((h0, h1), (true, true), "no value removed");
        assert_eq!(
            header["seq"], "13",
            "the header as the other writer left it"
        );
        assert!(!header.contains_key("restore_seq_bump"), "not banded");
        let kept: Option<String> = tv.client().pool.get(&stream_key).await.expect("GET");
        assert_eq!(kept.as_deref(), Some("not a stream"), "the stream key kept");
    }

    /// A value removal's admission of `charge` bytes as the budget holds it,
    /// in whole KiB (`storage::memory_budget::kib_for_bytes`).
    fn held_admission_bytes(charge: u64) -> u64 {
        charge.div_ceil(1024) * 1024
    }

    /// Drive `removal` until `gate` holds at least `charge` bytes, on the
    /// gate's admission events (`RestoreGate::admissions_changed`, b.ucc
    /// D10): each round registers for the next event, checks, then polls the
    /// event and the removal together, the event first, so the drive stops
    /// at the first poll after the removal's admission and the removal has
    /// sent at most the first command after it. Panics if the removal ends
    /// first or takes 30 s.
    async fn drive_until_admitted<F>(removal: &mut Pin<&mut F>, gate: &RestoreGate, charge: u64)
    where
        F: Future<Output = Result<ValueRemovalRecord, JobExit>>,
    {
        let drive = async {
            loop {
                let mut changed = pin!(gate.admissions_changed());
                changed.as_mut().enable();
                if gate.reserved_bytes() >= charge {
                    return;
                }
                tokio::select! {
                    biased;
                    () = changed.as_mut() => {}
                    ended = removal.as_mut() => {
                        panic!("the removal ended without holding its admission: {ended:?}")
                    }
                }
            }
        };
        if tokio::time::timeout(Duration::from_secs(30), drive)
            .await
            .is_err()
        {
            panic!(
                "the removal never held its admission of {charge} B; reserved {} B",
                gate.reserved_bytes()
            );
        }
    }

    /// D5 is admitted in the restore memory (gate w9g5 (b); fail-on-old): a
    /// removal with `purge` as its run's allocator purge, driven by hand
    /// against its own bytes-mode gate, once it holds its admission holds
    /// one, exactly `charge` (in whole KiB), and has never held two, while
    /// its header is unwritten. An admission of the whole budget queued then
    /// is polled ahead of the removal (`biased`), so it would be granted at
    /// the first poll after the removal's admission is given back; the store,
    /// read at that point before either is polled again, shows the header
    /// already landed. Once the removal ends the whole budget is admitted at
    /// once: the removal holds nothing.
    ///
    /// With the admission skipped, the removal ends without ever holding
    /// it; given back before the header's transaction, the whole budget is
    /// granted while the header is unwritten (its WATCH, read and EXEC each
    /// wait on a poll).
    async fn holds_one_admission_until_its_header_lands(
        object_id: u64,
        purge: Option<AllocatorPurge>,
        charge: u64,
    ) {
        let tv = TestValkey::setup().await;
        let mut run = bytes_mode_run(&tv).await;
        run.allocator_purge = purge;
        seed_value_object(&tv, object_id, &live_header("12"), 0).await;
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("bytes mode");
        let gate = RestoreGate::new(&memory, None);
        let whole = gate.budget_bytes().expect("bytes mode has a budget");
        let lost = value_removals(&[object_id]);
        let seams = ExecSeams::production();
        let mut removal = pin!(remove_layer_values(&run, &lost.values[0], &gate, &seams));

        drive_until_admitted(&mut removal, &gate, charge).await;
        assert_eq!(
            (
                gate.reserved_bytes(),
                gate.in_flight(),
                gate.max_in_flight()
            ),
            (held_admission_bytes(charge), 1, 1),
            "one admission of {charge} B, and never two at once"
        );
        let (header, _, h1) = value_object(&tv, object_id).await;
        assert_eq!(
            (header["seq"].as_str(), h1),
            ("12", true),
            "the header is unwritten while the admission is held"
        );

        let mut probe = pin!(gate.admit(whole));
        assert!(
            futures::poll!(probe.as_mut()).is_pending(),
            "the whole budget waits behind the held admission"
        );
        let (granted, ended) = tokio::select! {
            biased;
            granted = probe.as_mut() => (Some(granted), None),
            ended = removal.as_mut() => (None, Some(ended)),
        };
        let (header, h0, h1) = value_object(&tv, object_id).await;
        assert!(
            header["seq"] != "12" && !h0 && !h1,
            "the header had landed before the removal's admission was given back: {header:?}, \
             H0 named {h0}, H1 named {h1}"
        );

        let record = match ended {
            Some(ended) => ended,
            None => removal.as_mut().await,
        }
        .expect("the removal completes");
        assert_eq!(header["seq"], record.seeded_seq.to_string(), "banded");
        let granted = match granted {
            Some(granted) => granted,
            None => tokio::time::timeout(Duration::from_secs(10), probe.as_mut())
                .await
                .expect("the whole budget is admitted once the removal ends"),
        }
        .expect("the budget admits");
        assert_eq!(
            gate.reserved_bytes(),
            whole,
            "the whole budget admitted: the removal holds nothing"
        );
        drop(granted);
        assert_eq!((gate.reserved_bytes(), gate.in_flight()), (0, 0));
    }

    /// D5 with a working allocator purge (the fixture's) holds one
    /// admission of the purged figure, [`DR_VALUE_REMOVAL_ADMISSION_BYTES`],
    /// from before its walk until its header lands:
    /// [`holds_one_admission_until_its_header_lands`].
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_holds_one_admission_from_before_its_walk_until_its_header_lands() {
        holds_one_admission_until_its_header_lands(
            8_511,
            Some(AllocatorPurge::new(|| true)),
            DR_VALUE_REMOVAL_ADMISSION_BYTES,
        )
        .await;
    }

    /// W-T2, the fail-safe in use (fail-on-old). D5 with no allocator purge
    /// installed holds one admission of the un-purged figure,
    /// [`DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES`] (158 MiB), never the
    /// purged one:
    /// [`holds_one_admission_until_its_header_lands`]. The code before the
    /// purge held its one 50 MiB charge whatever the run carried; so does an
    /// in-place perturbation of `remove_layer_values` asking for
    /// `value_removal_admission(true)` always: the drive never sees 158 MiB.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_holds_the_unpurged_admission_when_no_allocator_purge_is_installed() {
        holds_one_admission_until_its_header_lands(
            8_512,
            None,
            DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES,
        )
        .await;
    }

    /// How a D5 removal stops early in [`admission_given_back_after`].
    enum EarlyExit {
        /// At a kill point.
        Killed(super::super::KillPoint),
        /// By its fence: the header's `seq` moved to 13 while it held its
        /// admission.
        SeqMoved,
    }

    /// One early exit of D5's admission pins (fail-on-old): a removal of
    /// [`H0`] and [`H1`] from an object, with two field rows naming [`H0`]
    /// when `rows`, stopped by `exit`, exits 1 with its header unwritten
    /// (its `seq` at `seq_after`, [`H1`] still in `obj:ext`), having held its
    /// admission (the gate's peak), and leaves nothing reserved and nothing
    /// admitted on its gate. With the admission given back only where the
    /// header lands (in-place perturbation of `remove_layer_values`, the
    /// admission in a `ManuallyDrop` released only after the header), every
    /// case leaves the admission reserved. One test per exit, so a
    /// regression reports each exit it breaks (b.ucc D11).
    async fn admission_given_back_after(
        object_id: u64,
        exit: EarlyExit,
        rows: bool,
        seq_after: &str,
    ) -> TestValkey {
        let tv = TestValkey::setup().await;
        let run = bytes_mode_run(&tv).await;
        seed_value_object(&tv, object_id, &live_header("12"), 0).await;
        if rows {
            seed_rows(
                &tv,
                object_id,
                &[external_field("a", H0), external_field("b", H0)],
            )
            .await;
        }
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("bytes mode");
        let gate = RestoreGate::new(&memory, None);
        let lost = value_removals(&[object_id]);
        let seams = ExecSeams {
            kill_point: match &exit {
                EarlyExit::Killed(point) => Some(point.clone()),
                EarlyExit::SeqMoved => None,
            },
            ..ExecSeams::production()
        };
        let mut removal = pin!(remove_layer_values(&run, &lost.values[0], &gate, &seams));
        // The fixture's purge works, so the removal holds the purged figure.
        let charge = value_removal_admission(true);
        if matches!(exit, EarlyExit::SeqMoved) {
            drive_until_admitted(&mut removal, &gate, charge).await;
            let _: i64 = tv
                .client()
                .pool
                .hset(
                    tv.key_schema().obj_header(&object_id.to_string()),
                    [("seq", "13")],
                )
                .await
                .expect("the other writer's HSET");
        }

        let ended = removal.as_mut().await;

        assert_eq!(ended, Err(JobExit::Failed));
        let (header, _, h1) = value_object(&tv, object_id).await;
        assert_eq!(
            (header["seq"].as_str(), h1),
            (seq_after, true),
            "the header unwritten"
        );
        assert!(
            gate.peak_reserved_bytes() >= charge,
            "the admission was held: peak {} B",
            gate.peak_reserved_bytes()
        );
        assert_eq!(
            (gate.reserved_bytes(), gate.in_flight()),
            (0, 0),
            "nothing left reserved or admitted"
        );
        tv
    }

    /// D5's admission is given back when it is killed after a row batch
    /// (`ValueRowsAfterBatch`, its first), on an object whose rows name a
    /// removed value: [`admission_given_back_after`]. The kill came after a
    /// row batch that landed (both rows gone), with the hash they named
    /// still in `obj:ext`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_gives_its_admission_back_when_killed_after_a_row_batch() {
        use super::super::KillPoint;
        const OBJECT: u64 = 8_521;
        let tv = admission_given_back_after(
            OBJECT,
            EarlyExit::Killed(KillPoint::ValueRowsAfterBatch {
                object: Some(OBJECT),
                batch: 0,
            }),
            true,
            "12",
        )
        .await;
        for name in ["a", "b"] {
            assert_eq!(
                row_stands(&tv, OBJECT, &external_field(name, H0)).await,
                (false, false),
                "row {name}: its batch landed before the kill"
            );
        }
        let (_, h0, _) = value_object(&tv, OBJECT).await;
        assert!(h0, "the rows' hash is still in obj:ext");
    }

    /// D5's admission is given back when it is killed before its header
    /// (`ValueRemovalBeforeHeader`): [`admission_given_back_after`].
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_gives_its_admission_back_when_killed_before_its_header() {
        use super::super::KillPoint;
        const OBJECT: u64 = 8_522;
        admission_given_back_after(
            OBJECT,
            EarlyExit::Killed(KillPoint::ValueRemovalBeforeHeader(Some(OBJECT))),
            false,
            "12",
        )
        .await;
    }

    /// D5's admission is given back when its fence stops it (the header's
    /// `seq` moved while it held its admission):
    /// [`admission_given_back_after`].
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_gives_its_admission_back_when_its_header_seq_moves() {
        const OBJECT: u64 = 8_523;
        admission_given_back_after(OBJECT, EarlyExit::SeqMoved, false, "13").await;
    }

    // ── D5's walk over value rows, on hand-built objects ────────────────

    /// A value the row fixtures' removals keep: rows naming it stay.
    const KEPT: ExtHash = [0xDE; 32];

    /// One value row of a hand-built object, on section 1: its index member
    /// (a field row's by its key name, a time-sample row's by its time) and
    /// its columns, as the commit paths write them.
    struct RowSpec {
        time_sample: bool,
        member: String,
        columns: Vec<(&'static str, String)>,
    }

    impl RowSpec {
        /// The row's key in `object_id`.
        fn key(&self, keys: &KeySchema, object_id: u64) -> String {
            let oid = object_id.to_string();
            if self.time_sample {
                keys.obj_ts_from_member(&oid, &self.member)
            } else {
                keys.obj_field_from_member(&oid, &self.member)
            }
        }

        /// The row's index in `object_id`.
        fn index(&self, keys: &KeySchema, object_id: u64) -> String {
            let oid = object_id.to_string();
            if self.time_sample {
                keys.obj_time_samples_index(&oid)
            } else {
                keys.obj_fields_index(&oid)
            }
        }
    }

    /// The columns every field row carries beside its value or hash.
    fn field_columns() -> Vec<(&'static str, String)> {
        ["set_order", "authoring_set_order", "version"]
            .into_iter()
            .map(|column| (column, "1".to_owned()))
            .collect()
    }

    /// A field row `name` naming external value `hash`.
    fn external_field(name: &str, hash: ExtHash) -> RowSpec {
        let mut columns = vec![("hash", sha256_to_hex(&hash)), ("size", "1".to_owned())];
        columns.extend(field_columns());
        RowSpec {
            time_sample: false,
            member: KeySchema::field_member("1", name),
            columns,
        }
    }

    /// A field row `name` holding an inline value.
    fn inline_field(name: &str) -> RowSpec {
        let mut columns = vec![("value", "AA==".to_owned())];
        columns.extend(field_columns());
        RowSpec {
            time_sample: false,
            member: KeySchema::field_member("1", name),
            columns,
        }
    }

    /// A legacy dual-shape field row `name`: an inline value beside a `hash`
    /// naming `hash`. Every reader decodes it inline, so it names no value.
    fn dual_shape_field(name: &str, hash: ExtHash) -> RowSpec {
        let mut row = external_field(name, hash);
        row.columns.push(("value", "AA==".to_owned()));
        row
    }

    /// A time-sample row at `time` naming external value `hash`.
    fn external_sample(time: f64, hash: ExtHash) -> RowSpec {
        RowSpec {
            time_sample: true,
            member: KeySchema::time_sample_member("1", time),
            columns: vec![
                ("hash", sha256_to_hex(&hash)),
                ("size", "1".to_owned()),
                ("version", "1".to_owned()),
            ],
        }
    }

    /// A time-sample row at `time` holding an inline value.
    fn inline_sample(time: f64) -> RowSpec {
        RowSpec {
            time_sample: true,
            member: KeySchema::time_sample_member("1", time),
            columns: vec![("value", "AA==".to_owned()), ("version", "1".to_owned())],
        }
    }

    /// Seed `rows` into `object_id`: each row's columns and its index
    /// member, in one pipeline.
    async fn seed_rows(tv: &TestValkey, object_id: u64, rows: &[RowSpec]) {
        let keys = tv.key_schema();
        let seeding = tv.client().pool.next().pipeline();
        for row in rows {
            let _: () = seeding
                .hset(row.key(keys, object_id), row.columns.clone())
                .await
                .expect("queueing a row");
            let _: () = seeding
                .zadd(
                    row.index(keys, object_id),
                    None,
                    None,
                    false,
                    false,
                    (1.0, row.member.as_str()),
                )
                .await
                .expect("queueing its index member");
        }
        let _: Vec<fred::types::Value> = seeding.all().await.expect("seeding the rows");
    }

    /// Whether `row` of `object_id` still stands: its key, and its member in
    /// its index.
    async fn row_stands(tv: &TestValkey, object_id: u64, row: &RowSpec) -> (bool, bool) {
        let keys = tv.key_schema();
        let exists: u64 = tv
            .client()
            .pool
            .exists(row.key(keys, object_id))
            .await
            .expect("EXISTS");
        let score: Option<f64> = tv
            .client()
            .pool
            .zscore(row.index(keys, object_id), row.member.as_str())
            .await
            .expect("ZSCORE");
        (exists == 1, score.is_some())
    }

    /// A table-encoded object's value rows (both indices past the compact
    /// encoding's 128 members), each with whether D5 removing [`H0`] and
    /// [`H1`] must delete it: 150 field rows — inline, dual-shape naming
    /// [`H0`], naming [`KEPT`], and naming [`H0`], [`H1`], [`H0`] in turn —
    /// and 140 time-sample rows — inline, naming [`KEPT`], [`H1`], [`H0`] in
    /// turn — and, when `signed_zero`, a legacy `-0.0` sample naming [`H0`]
    /// beside a `+0.0` one naming [`H1`], two rows by their members.
    fn table_rows(signed_zero: bool) -> Vec<(RowSpec, bool)> {
        let fields = (0..150).map(|i| {
            let name = format!("k{i:03}");
            match i % 6 {
                0 => (inline_field(&name), false),
                1 => (dual_shape_field(&name, H0), false),
                2 => (external_field(&name, KEPT), false),
                3 | 5 => (external_field(&name, H0), true),
                _ => (external_field(&name, H1), true),
            }
        });
        let samples = (0..140).map(|j| {
            let time = f64::from(j) + 0.5;
            match j % 4 {
                0 => (inline_sample(time), false),
                1 => (external_sample(time, KEPT), false),
                2 => (external_sample(time, H1), true),
                _ => (external_sample(time, H0), true),
            }
        });
        let signed_zero = signed_zero
            .then(|| {
                [
                    (external_sample(-0.0, H0), true),
                    (external_sample(0.0, H1), true),
                ]
            })
            .into_iter()
            .flatten();
        fields.chain(samples).chain(signed_zero).collect()
    }

    /// Seed [`table_rows`] into `object_id` beside its live header and its
    /// `obj:ext` ([`H0`], [`H1`] and [`KEPT`]), and assert both indices are
    /// table-encoded (`skiplist`): past the compact encoding, the shape a
    /// large layer's indices have.
    async fn seed_table_object(
        tv: &TestValkey,
        object_id: u64,
        signed_zero: bool,
    ) -> Vec<(RowSpec, bool)> {
        seed_value_object(tv, object_id, &live_header("12"), 0).await;
        let _: i64 = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_ext_index(&object_id.to_string()),
                [(sha256_to_hex(&KEPT), "1")],
            )
            .await
            .expect("HSET obj:ext");
        let rows = table_rows(signed_zero);
        let specs: Vec<RowSpec> = rows
            .iter()
            .map(|(row, _)| RowSpec {
                time_sample: row.time_sample,
                member: row.member.clone(),
                columns: row.columns.clone(),
            })
            .collect();
        seed_rows(tv, object_id, &specs).await;
        let oid = object_id.to_string();
        for index in [
            tv.key_schema().obj_fields_index(&oid),
            tv.key_schema().obj_time_samples_index(&oid),
        ] {
            assert_eq!(
                crate::storage::test_support::object_encoding_for_test(tv.client(), &index).await,
                "skiplist",
                "fixture: {index} is past the compact encoding"
            );
        }
        rows
    }

    /// Part B's coverage (fail-on-old). D5 removing [`H0`] and [`H1`] from a
    /// table-encoded object ([`table_rows`], the signed-zero pair included)
    /// deletes every row naming either, by the readers' rule, and no other
    /// row: the inline rows, the dual-shape rows (a `value` beside a `hash`
    /// naming [`H0`], inline to every reader) and the rows naming [`KEPT`]
    /// stand, key and index member; the `-0.0` and `+0.0` samples both go,
    /// each by its own member. `rows` counts exactly the rows deleted; both
    /// values leave `obj:ext`, [`KEPT`] stays; the header is banded. At
    /// pages of 1 and 2 members and the production page, under a batch seam
    /// of 4 commands (two rows a batch, so batches flush while the walk is
    /// mid-index), and at the production page and batch.
    ///
    /// Fails with an ascending rank pager (in-place perturbation of the
    /// walk's `walk_index`, `ZRANGE i i+n-1` pages from rank 0 up in place
    /// of its descending pages, run with its short-page and end-count checks
    /// disabled too): each flush removes members below the next page's
    /// first rank, so the next page starts past rows the walk never reads,
    /// and they stand ("page 1, batch 4: rows left or lost").
    /// Passes with the production pager, which pages by descending absolute
    /// ranks from the index's start `ZCARD`: a row it deletes is at or above
    /// the current page's lowest rank, so no rank still to be read shifts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_deletes_every_row_naming_a_removed_value_and_no_other_on_table_encoded_indices() {
        use super::super::memory::DR_VALUE_REMOVAL_PAGE_ROWS;
        use crate::storage::layer::dr_restore::RESTORE_BATCH_COMMANDS;

        const OBJECT: u64 = 8_531;
        let cases = [
            (1, 4),
            (2, 4),
            (DR_VALUE_REMOVAL_PAGE_ROWS, 4),
            (DR_VALUE_REMOVAL_PAGE_ROWS, RESTORE_BATCH_COMMANDS),
        ];
        for (page, batch) in cases {
            let case = format!("page {page}, batch {batch}");
            let tv = TestValkey::setup().await;
            let run = accept_loss_run(&tv).await;
            let rows = seed_table_object(&tv, OBJECT, true).await;
            let lost = value_removals(&[OBJECT]);
            let removal = &lost.values[0];
            let seams = ExecSeams {
                batch_commands: batch,
                value_removal_page_rows: page,
                ..ExecSeams::production()
            };

            let record = remove_layer_values_for_test(&run, removal, &seams)
                .await
                .unwrap_or_else(|exit| panic!("{case}: the removal completes, got {exit:?}"));

            let mut wrong = Vec::new();
            for (row, removed) in &rows {
                let stands = row_stands(&tv, OBJECT, row).await;
                let want = if *removed {
                    (false, false)
                } else {
                    (true, true)
                };
                if stands != want {
                    wrong.push(format!(
                        "{} {}: (key, member) {stands:?}, want {want:?}",
                        if row.time_sample { "sample" } else { "field" },
                        row.member
                    ));
                }
            }
            assert!(
                wrong.is_empty(),
                "{case}: rows left or lost:\n{}",
                wrong.join("\n")
            );
            let removed = rows.iter().filter(|(_, removed)| *removed).count();
            assert_eq!(record.rows, removed, "{case}: rows counts the rows deleted");
            let (header, h0, h1) = value_object(&tv, OBJECT).await;
            assert_eq!(
                (h0, h1),
                (false, false),
                "{case}: both values leave obj:ext"
            );
            let kept: bool = tv
                .client()
                .pool
                .hexists(
                    tv.key_schema().obj_ext_index(&OBJECT.to_string()),
                    sha256_to_hex(&KEPT),
                )
                .await
                .expect("HEXISTS");
            assert!(kept, "{case}: the kept value stays in obj:ext");
            assert_eq!(
                header["seq"],
                record.seeded_seq.to_string(),
                "{case}: the header is banded"
            );
        }
    }

    /// I-8, the readers' rule against the report's oracle: the rows D5
    /// deletes are exactly the rows `plan::report::rows_referencing` names
    /// for the removed values over the object's compact state as the
    /// restore planner reads it (`read_object_snapshot_slim`), the sites
    /// the coverage report names; and `rows` is their number. On the
    /// table-encoded object without the signed-zero pair (the decoded state
    /// holds one sample for the pair, the store two rows), at the
    /// production page and batch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_deletes_exactly_the_rows_the_reports_oracle_names_for_the_removed_values() {
        use crate::dr::job::plan::report::rows_referencing;
        use crate::storage::diff_read::{
            read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotReadOf,
        };
        use crate::storage::layer::ValueRowKey;

        const OBJECT: u64 = 8_532;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        let rows = seed_table_object(&tv, OBJECT, false).await;
        let lost = value_removals(&[OBJECT]);
        let removal = &lost.values[0];
        let keys = tv.key_schema();
        let oid = OBJECT.to_string();
        let state = match read_object_snapshot_slim(
            tv.client(),
            keys,
            &oid,
            SlimSnapshotReadConsumer::DrRestore,
        )
        .await
        .expect("the slim read")
        {
            SnapshotReadOf::Snapshot(snapshot) => snapshot.state,
            SnapshotReadOf::TooLarge(_) => panic!("fixture: the object reads whole"),
        };
        let oracle: BTreeSet<String> = rows_referencing(&state, &removal.hashes)
            .into_values()
            .flatten()
            .map(|row| match row {
                ValueRowKey::Field {
                    section_id,
                    key_name,
                } => keys.obj_field(&oid, &section_id.to_string(), &key_name),
                ValueRowKey::TimeSample {
                    section_id,
                    time_bits,
                } => keys.obj_time_sample(&oid, &section_id.to_string(), f64::from_bits(time_bits)),
            })
            .collect();
        assert!(!oracle.is_empty(), "fixture: the oracle names rows");

        let record = remove_layer_values_for_test(&run, removal, &ExecSeams::production())
            .await
            .expect("the removal completes");

        let mut deleted = BTreeSet::new();
        for (row, _) in &rows {
            if row_stands(&tv, OBJECT, row).await == (false, false) {
                deleted.insert(row.key(keys, OBJECT));
            }
        }
        assert_eq!(deleted, oracle, "the rows deleted are the oracle's");
        assert_eq!(record.rows, oracle.len(), "rows counts them");
    }

    /// Seven values, in hash order: the first two named by rows
    /// ([`six_rows`]), the other five only in `obj:ext`.
    fn seven_values() -> Vec<ExtHash> {
        (0..7u8).map(|i| [0xC0 + i; 32]).collect()
    }

    /// Six rows: three field rows naming `values[0]`, then three time-sample
    /// rows naming `values[1]`.
    fn six_rows(values: &[ExtHash]) -> Vec<RowSpec> {
        vec![
            external_field("a", values[0]),
            external_field("b", values[0]),
            external_field("c", values[0]),
            external_sample(1.0, values[1]),
            external_sample(2.0, values[1]),
            external_sample(3.0, values[1]),
        ]
    }

    /// The removal of `values` from `object_id`.
    fn removal_of(object_id: u64, values: &[ExtHash]) -> ValueRemoval {
        ValueRemoval {
            object_id,
            session_id: SESSION.to_owned(),
            layer_name: LAYER.to_owned(),
            hashes: values.iter().copied().collect(),
            unconfirmed: BTreeSet::new(),
            sites: Vec::new(),
        }
    }

    /// `object_id`: its live header, `obj:ext` naming every one of `values`,
    /// and `rows`.
    async fn seed_values_and_rows(
        tv: &TestValkey,
        object_id: u64,
        values: &[ExtHash],
        rows: &[RowSpec],
    ) {
        let oid = object_id.to_string();
        let _: i64 = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_header(&oid), live_header("12"))
            .await
            .expect("HSET header");
        let ext: Vec<(String, String)> = values
            .iter()
            .map(|h| (sha256_to_hex(h), "1".to_owned()))
            .collect();
        let _: i64 = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_ext_index(&oid), ext)
            .await
            .expect("HSET obj:ext");
        seed_rows(tv, object_id, rows).await;
    }

    /// I-5, the order of D5's writes. A removal of seven values — the first
    /// two each named by three rows (field rows, then time-sample rows), the
    /// other five only in `obj:ext` — under a batch seam of 4 commands sends
    /// its six rows in three batches of two (4 commands each), then the
    /// `HDEL obj:ext` of every value but the last, in hash order, cut at the
    /// command limit into 4 and 2, then the header with the last value's
    /// `HDEL` (3 commands). Killed after its last row batch every value is
    /// still in `obj:ext` and every row is gone; killed before its header
    /// only the last value is, and the header is unbanded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_removes_rows_then_obj_ext_hashes_within_the_command_limit_then_its_header() {
        use super::super::KillPoint;

        const OBJECT: u64 = 8_533;
        let values = seven_values();
        let last = values[6];
        let rows = || six_rows(&values);
        let removal = removal_of(OBJECT, &values);
        let indexed = |tv: &TestValkey, hash: ExtHash| {
            let key = tv.key_schema().obj_ext_index(&OBJECT.to_string());
            let client = tv.client().clone();
            async move {
                let named: bool = client
                    .pool
                    .hexists(key, sha256_to_hex(&hash))
                    .await
                    .expect("HEXISTS");
                named
            }
        };
        let seams = |kill_point: Option<KillPoint>| ExecSeams {
            batch_commands: 4,
            kill_point,
            ..ExecSeams::production()
        };

        // Uninterrupted: the transactions, in order.
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_values_and_rows(&tv, OBJECT, &values, &rows()).await;
        let record = remove_layer_values_for_test(&run, &removal, &seams(None))
            .await
            .expect("the removal completes");
        assert_eq!(
            record
                .batches
                .iter()
                .map(|batch| batch.commands)
                .collect::<Vec<_>>(),
            vec![4, 4, 4, 4, 2, 3],
            "three row batches, two obj:ext batches cut at 4 commands, the header"
        );
        assert_eq!(record.rows, 6);
        for hash in &values {
            assert!(!indexed(&tv, *hash).await, "every value left obj:ext");
        }

        // Killed after the last row batch: every value still in obj:ext.
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_values_and_rows(&tv, OBJECT, &values, &rows()).await;
        let killed = remove_layer_values_for_test(
            &run,
            &removal,
            &seams(Some(KillPoint::ValueRowsAfterBatch {
                object: Some(OBJECT),
                batch: 2,
            })),
        )
        .await;
        assert_eq!(
            killed,
            Err(JobExit::Failed),
            "killed after its last row batch"
        );
        for row in rows() {
            assert_eq!(
                row_stands(&tv, OBJECT, &row).await,
                (false, false),
                "every row is gone after the last row batch: {}",
                row.member
            );
        }
        for hash in &values {
            assert!(
                indexed(&tv, *hash).await,
                "no value leaves obj:ext before the rows are all gone"
            );
        }

        // Killed before its header: only the last value is in obj:ext.
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_values_and_rows(&tv, OBJECT, &values, &rows()).await;
        let killed = remove_layer_values_for_test(
            &run,
            &removal,
            &seams(Some(KillPoint::ValueRemovalBeforeHeader(Some(OBJECT)))),
        )
        .await;
        assert_eq!(killed, Err(JobExit::Failed), "killed before its header");
        for hash in &values {
            assert_eq!(
                indexed(&tv, *hash).await,
                *hash == last,
                "before the header only the last value is in obj:ext"
            );
        }
        let (header, _, _) = value_object(&tv, OBJECT).await;
        assert_eq!(header["seq"], "12", "the header is unbanded");
    }

    // ── D5's walk: its completeness checks (b.f3t) ──────────────────────
    //
    // D5 pages each value index by descending ranks from its start `ZCARD`
    // and reads no serving-process witness (`storage::scan_guard`, *The DR
    // Jobs' walks*): a page short of its ranks, or an end `ZCARD` other than
    // a complete walk leaves, stops it before `obj:ext` and the header. The
    // skip is forged through the walk seam's `DropMember`, keyed by the
    // index key, which carries the test's prefix.

    /// Whether `object`'s `obj:ext` still names each of `values`, in order.
    async fn ext_named(tv: &TestValkey, object: u64, values: &[ExtHash]) -> Vec<bool> {
        let key = tv.key_schema().obj_ext_index(&object.to_string());
        let mut named = Vec::with_capacity(values.len());
        for hash in values {
            let present: bool = tv
                .client()
                .pool
                .hexists(&key, sha256_to_hex(hash))
                .await
                .expect("HEXISTS");
            named.push(present);
        }
        named
    }

    /// T5 (b.f3t), a skipped member. A page of D5's descending rank walk
    /// that answers one member short of its ranks — the walk seam's
    /// `DropMember` on the index's first page, standing in for a member a
    /// read continued on another process could skip — ends the removal: one
    /// ERROR saying the backing store changed while the Job read the layer's
    /// value indices, naming the page (`ZRANGE <index> 0 2`, 2 members where
    /// 3 were expected) and exit 1 and counting no attempts, and exit 1,
    /// before the `obj:ext` batches and the header: every removed value is
    /// still in `obj:ext` and the header's `seq` is still S. Short on
    /// `obj:flds`, no row batch has run, so every row stands; short on
    /// `obj:tss`, under a batch seam of 4 commands, the first `obj:flds`
    /// batch (rows `a` and `b`) has landed and every other row stands. A
    /// clean re-run then removes every row and value and bands the header.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with `walk_index`'s short-page check removed and
    /// `scan_guard::check_count` answering `Ok` whatever the counts, the
    /// removal completes instead of exiting 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_stops_before_obj_ext_and_its_header_when_a_page_answers_short_of_its_ranks() {
        use crate::storage::scan_guard::{arm_walk_fault_for_test, WalkFault};

        const OBJECT: u64 = 8_534;
        let values = seven_values();
        let rows = six_rows(&values);
        let removal = removal_of(OBJECT, &values);
        let oid = OBJECT.to_string();
        let seams = ExecSeams {
            batch_commands: 4,
            ..ExecSeams::production()
        };
        for time_sample in [false, true] {
            let tv = TestValkey::setup().await;
            let run = accept_loss_run(&tv).await;
            seed_values_and_rows(&tv, OBJECT, &values, &rows).await;
            let index = if time_sample {
                tv.key_schema().obj_time_samples_index(&oid)
            } else {
                tv.key_schema().obj_fields_index(&oid)
            };
            let case = format!("a short page of {index}");

            let armed = arm_walk_fault_for_test(
                &index,
                WalkFault::DropMember {
                    walk: 0,
                    step: Some(0),
                    member: None,
                },
            );
            let capture = LogCapture::default();
            let guard = capture.install();
            let result = remove_layer_values_for_test(&run, &removal, &seams).await;
            drop(guard);
            drop(armed);

            assert_eq!(result.err(), Some(JobExit::Failed), "{case}: exit 1");
            let errors = capture.messages("ERROR");
            assert_eq!(errors.len(), 1, "{case}: one ERROR: {errors:?}");
            for part in [
                format!(
                    "could not remove values from object {OBJECT}: the backing store changed \
                     while the Job read the layer's value indices"
                ),
                format!("ZRANGE {index} 0 2 returned 2 members, where 3 were expected"),
                "exits 1".to_owned(),
            ] {
                assert!(
                    errors[0].contains(&part),
                    "{case}: the ERROR names {part:?}: {}",
                    errors[0]
                );
            }
            assert!(
                !errors[0].contains("attempt"),
                "{case}: no step's attempts ran out: the ERROR counts none: {}",
                errors[0]
            );
            for (i, row) in rows.iter().enumerate() {
                // Under the batch seam, `obj:flds`'s first batch (rows `a`
                // and `b`) lands before the walk reaches `obj:tss`.
                let landed = time_sample && i < 2;
                assert_eq!(
                    row_stands(&tv, OBJECT, row).await,
                    if landed { (false, false) } else { (true, true) },
                    "{case}: row {}",
                    row.member
                );
            }
            assert_eq!(
                ext_named(&tv, OBJECT, &values).await,
                vec![true; values.len()],
                "{case}: every removed value is still in obj:ext"
            );
            let (header, _, _) = value_object(&tv, OBJECT).await;
            assert_eq!(header["seq"], "12", "{case}: the header is unbanded");

            let record = remove_layer_values_for_test(&run, &removal, &seams)
                .await
                .unwrap_or_else(|exit| panic!("{case}: the clean re-run completes, got {exit:?}"));
            for row in &rows {
                assert_eq!(
                    row_stands(&tv, OBJECT, row).await,
                    (false, false),
                    "{case}: the re-run removes row {}",
                    row.member
                );
            }
            assert_eq!(
                ext_named(&tv, OBJECT, &values).await,
                vec![false; values.len()],
                "{case}: the re-run removes every value"
            );
            let (header, _, _) = value_object(&tv, OBJECT).await;
            assert_eq!(
                header["seq"],
                record.seeded_seq.to_string(),
                "{case}: the re-run bands the header"
            );
        }
    }

    /// T5 (b.f3t), a lost delete, on each value index. D5 under a batch seam
    /// of 4 commands sends its six rows in three batches — `obj:flds` rows
    /// `a` and `b`; row `c` and the `obj:tss` sample at 1.0; the samples at
    /// 2.0 and 3.0 — each purged as it begins and once it committed, after
    /// the one purge before the removal. Its allocator purge is parked once a
    /// chosen batch has landed (the third purge for the first batch, the
    /// fifth for the second) while the test writes one of that batch's rows
    /// back, key and index member, as a failover that lost the batch's
    /// delete would leave it: `obj:flds`'s row `a`, or `obj:tss`'s sample at
    /// 1.0. No page of that index is read again, so the walk never meets the
    /// row; its end check, after the last row batch and before the first
    /// `obj:ext` batch, reads both indices' `ZCARD` and finds that index's 1
    /// where a complete walk leaves 0 (3 at its start, 3 deleted). D5 logs
    /// one ERROR saying the store changed while the Job read the layer's
    /// value indices, naming the index and those counts and exit 1,
    /// counting no attempts, and exits 1: every removed value is still in
    /// `obj:ext`, so the row written back names no value missing from it,
    /// and the header's `seq` is still S. The `obj:tss` case shows the end
    /// check reads its second index too, after an `obj:flds` that holds.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with `scan_guard::check_count` answering `Ok` whatever the counts,
    /// each case's removal completes instead of exiting 1, so it goes on to
    /// take the values out of `obj:ext` and band the header while the row
    /// written back still names one of them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_stops_before_obj_ext_and_its_header_when_a_lost_delete_moves_its_end_count() {
        const OBJECT: u64 = 8_535;
        let values = seven_values();
        let rows = six_rows(&values);
        let removal = removal_of(OBJECT, &values);
        let oid = OBJECT.to_string();
        let seams = ExecSeams {
            batch_commands: 4,
            ..ExecSeams::production()
        };
        let (gone, stands) = ((false, false), (true, true));
        /// Each of the six rows' (key, index member) when the purge parks.
        type AtPark = [(bool, bool); 6];
        // (the index, the purge parked at, which rows have landed then, the
        // row written back)
        let cases: [(bool, usize, AtPark, usize); 2] = [
            (false, 3, [gone, gone, stands, stands, stands, stands], 0),
            (true, 5, [gone, gone, gone, gone, stands, stands], 3),
        ];
        let mut stopped: Vec<String> = Vec::new();
        for (time_sample, park, landed, lost) in cases {
            let tv = TestValkey::setup().await;
            let mut run = accept_loss_run(&tv).await;
            let parks: &'static [usize] = if park == 3 { &[3] } else { &[5] };
            let mut script = PurgeScript::new(usize::MAX, parks);
            run.allocator_purge = Some(script.purge.clone());
            seed_values_and_rows(&tv, OBJECT, &values, &rows).await;
            let index = if time_sample {
                tv.key_schema().obj_time_samples_index(&oid)
            } else {
                tv.key_schema().obj_fields_index(&oid)
            };
            let case = format!("a lost delete on {index}");
            let capture = LogCapture::default();
            let guard = capture.install();

            let (result, at_park) = tokio::join!(
                remove_layer_values_for_test(&run, &removal, &seams),
                async {
                    let parked = script.entered().await;
                    let mut at_park = Vec::new();
                    for row in &rows {
                        at_park.push(row_stands(&tv, OBJECT, row).await);
                    }
                    // The lost delete: the row back, its key and its member.
                    seed_rows(&tv, OBJECT, &rows[lost..=lost]).await;
                    script.release();
                    (parked, at_park)
                }
            );
            drop(guard);

            assert_eq!(
                at_park,
                (park, landed.to_vec()),
                "{case}: fixture: parked once the chosen batch landed"
            );
            if result.is_ok() {
                stopped.push(format!(
                    "{case}: the removal completed instead of exiting 1"
                ));
                continue;
            }
            assert_eq!(result.err(), Some(JobExit::Failed), "{case}: exit 1");
            let errors = capture.messages("ERROR");
            assert_eq!(errors.len(), 1, "{case}: one ERROR: {errors:?}");
            for part in [
                format!(
                    "could not remove values from object {OBJECT}: the backing store changed \
                     while the Job read the layer's value indices"
                ),
                format!(
                    "the ZRANGE walk of {index} returned 3 members, and ZCARD read 3 at its \
                     start and 1 at its end, where a complete walk returns 3 and ends at 0"
                ),
                "exits 1".to_owned(),
            ] {
                assert!(
                    errors[0].contains(&part),
                    "{case}: the ERROR names {part:?}: {}",
                    errors[0]
                );
            }
            assert!(
                !errors[0].contains("attempt"),
                "{case}: no step's attempts ran out: the ERROR counts none: {}",
                errors[0]
            );
            assert_eq!(
                ext_named(&tv, OBJECT, &values).await,
                vec![true; values.len()],
                "{case}: every removed value is still in obj:ext"
            );
            assert_eq!(
                row_stands(&tv, OBJECT, &rows[lost]).await,
                stands,
                "{case}: the lost delete's row stands"
            );
            let (header, _, _) = value_object(&tv, OBJECT).await;
            assert_eq!(header["seq"], "12", "{case}: the header is unbanded");
        }
        assert!(stopped.is_empty(), "{}", stopped.join("\n"));
    }

    /// T6 (b.f3t). D5's walk reads no serving-process witness, so a store
    /// whose `INFO server` reports a placeholder run_id (`"0"`) or no
    /// `run_id` line, or whose run_id changes between two reads, no longer
    /// stops the removal: with each forged on both value indices' walks
    /// through the walk seam (a `RunId` fault on the first witness read; a
    /// change of serving process at the second), the removal completes —
    /// every row deleted, every value out of `obj:ext`, the header banded —
    /// and logs no ERROR.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with a witness read put back into each of `RankWalk`'s reads (an
    /// `INFO server` before each `ZCARD` and page, compared with the walk's
    /// first), every case stops the removal, exit 1, instead of completing
    /// (D5 now fails a missing witness as a failed step, no longer exit 3).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_runs_on_a_store_whose_run_id_is_unusable_or_changes() {
        use crate::storage::scan_guard::{
            arm_walk_fault_for_test, forge_process_change_for_test, ArmedWalkFault, WalkFault,
        };

        const OBJECT: u64 = 8_536;
        type Forge = fn(&str) -> ArmedWalkFault;
        let cases: [(&str, Forge); 3] = [
            ("run_id \"0\"", |key| {
                arm_walk_fault_for_test(
                    key,
                    WalkFault::RunId {
                        read: 0,
                        run_id: Some("0".to_owned()),
                    },
                )
            }),
            ("no run_id line", |key| {
                arm_walk_fault_for_test(
                    key,
                    WalkFault::RunId {
                        read: 0,
                        run_id: None,
                    },
                )
            }),
            ("a change of run_id", |key| {
                forge_process_change_for_test(key, 1)
            }),
        ];
        let values = seven_values();
        let rows = six_rows(&values);
        let removal = removal_of(OBJECT, &values);
        let oid = OBJECT.to_string();
        // A case whose removal did not complete is collected, so one run
        // names every case a regression reaches.
        let mut stopped: Vec<String> = Vec::new();
        for (case, forge) in cases {
            let tv = TestValkey::setup().await;
            let run = accept_loss_run(&tv).await;
            seed_values_and_rows(&tv, OBJECT, &values, &rows).await;
            let armed = [
                tv.key_schema().obj_fields_index(&oid),
                tv.key_schema().obj_time_samples_index(&oid),
            ]
            .map(|index| forge(&index));
            let capture = LogCapture::default();
            let guard = capture.install();
            let result =
                remove_layer_values_for_test(&run, &removal, &ExecSeams::production()).await;
            drop(guard);
            drop(armed);

            let record = match result {
                Ok(record) => record,
                Err(exit) => {
                    stopped.push(format!("{case}: the removal completes, got {exit:?}"));
                    continue;
                }
            };
            let errors = capture.messages("ERROR");
            assert!(errors.is_empty(), "{case}: no ERROR: {errors:?}");
            for row in &rows {
                assert_eq!(
                    row_stands(&tv, OBJECT, row).await,
                    (false, false),
                    "{case}: row {} deleted",
                    row.member
                );
            }
            assert_eq!(
                ext_named(&tv, OBJECT, &values).await,
                vec![false; values.len()],
                "{case}: every value left obj:ext"
            );
            let (header, _, _) = value_object(&tv, OBJECT).await;
            assert_eq!(
                header["seq"],
                record.seeded_seq.to_string(),
                "{case}: the header is banded"
            );
        }
        assert!(stopped.is_empty(), "{}", stopped.join("\n"));
    }

    /// The decider's R1 narrowing, on D5 (b.gns; b.f3t's rank walk). An
    /// `obj:flds` that holds a string, so the walk's first read of it, its
    /// start `ZCARD`, answers `WRONGTYPE`, a deterministic error: the read's
    /// write step is `Fatal` on its first attempt — one ERROR, "after 1
    /// attempts", naming the count of the index and `WRONGTYPE` — and D5
    /// exits 1 before any write, the header unbanded.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored,
    /// re-run for this `ZCARD` read): with `walk_step_error`'s `Step` arm
    /// back to `StepError::Retry` for every error, the write step tries
    /// three times and logs "after 3 attempts".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_fails_its_index_read_on_the_first_attempt_when_the_index_answers_wrongtype() {
        const OBJECT: u64 = 8_537;
        let values = seven_values();
        let removal = removal_of(OBJECT, &values);
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_values_and_rows(&tv, OBJECT, &values, &[]).await;
        let index = tv.key_schema().obj_fields_index(&OBJECT.to_string());
        let _: () = tv
            .client()
            .pool
            .set(&index, "not-a-sorted-set", None, None, false)
            .await
            .expect("SET obj:flds as a string");

        let capture = LogCapture::default();
        let guard = capture.install();
        let result = remove_layer_values_for_test(&run, &removal, &ExecSeams::production()).await;
        drop(guard);

        assert_eq!(result.err(), Some(JobExit::Failed), "exit 1");
        let errors = capture.messages("ERROR");
        assert_eq!(errors.len(), 1, "one ERROR: {errors:?}");
        for part in [
            format!("count the members of object {OBJECT}'s index {index}"),
            "after 1 attempts".to_owned(),
            "WRONGTYPE".to_owned(),
        ] {
            assert!(
                errors[0].contains(&part),
                "the ERROR names {part:?}: {}",
                errors[0]
            );
        }
        let (header, _, _) = value_object(&tv, OBJECT).await;
        assert_eq!(header["seq"], "12", "the header is unbanded");
    }

    /// PM 2 (b.f3t): D5's rank reads are retried in place after a transient
    /// failure, each read one write step whose attempts are its bound. The
    /// walk seam's `StepError` on `obj:flds`'s rank walk fails a chosen read
    /// with a transient (`IO`) error before it is sent: read 0 is the start
    /// `ZCARD`, read 1 the walk's one page (`ZRANGE 0 2`). Failing once, the
    /// read is sent again and lands, and the removal completes — every row
    /// deleted, every value out of `obj:ext`, the header banded — with no
    /// ERROR. Failing on every one of the write step's
    /// `DR_WRITE_STEP_ATTEMPTS` attempts, the removal exits 1 with one ERROR
    /// naming the read and "after 3 attempts", before any write: every row
    /// stands, every value is still in `obj:ext`, the header is unbanded.
    /// D5 is driven directly, so the seam, which has no walk selector,
    /// reaches no other rank walk of the key.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with `walk_step_error`'s `Step` arm failing every error as `Fatal`,
    /// every case stops at its first failed attempt: the once-failed reads
    /// stop the removal, and the always-failing ones log "after 1
    /// attempts".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_retries_a_transiently_failed_rank_read_in_place_up_to_its_write_step_attempts() {
        use crate::dr::job::execute::DR_WRITE_STEP_ATTEMPTS;
        use crate::storage::scan_guard::{arm_walk_fault_for_test, WalkFault};

        const OBJECT: u64 = 8_538;
        let values = seven_values();
        let rows = six_rows(&values);
        let removal = removal_of(OBJECT, &values);
        let oid = OBJECT.to_string();
        // (the read, how many attempts fail, what the failed write step
        // names)
        let cases = [
            (0, 1, "count the members"),
            (1, 1, "read object"),
            (0, DR_WRITE_STEP_ATTEMPTS, "count the members"),
            (1, DR_WRITE_STEP_ATTEMPTS, "read object"),
        ];
        let mut failures: Vec<String> = Vec::new();
        for (read, times, names) in cases {
            let tv = TestValkey::setup().await;
            let run = accept_loss_run(&tv).await;
            seed_values_and_rows(&tv, OBJECT, &values, &rows).await;
            let index = tv.key_schema().obj_fields_index(&oid);
            let case = format!("read {read} failing {times} time(s)");

            let armed = arm_walk_fault_for_test(&index, WalkFault::StepError { step: read, times });
            let capture = LogCapture::default();
            let guard = capture.install();
            let result =
                remove_layer_values_for_test(&run, &removal, &ExecSeams::production()).await;
            drop(guard);
            drop(armed);

            let errors = capture.messages("ERROR");
            let (header, _, _) = value_object(&tv, OBJECT).await;
            if times == 1 {
                let Ok(record) = result else {
                    failures.push(format!("{case}: the removal completes, got {errors:?}"));
                    continue;
                };
                assert!(errors.is_empty(), "{case}: no ERROR: {errors:?}");
                for row in &rows {
                    assert_eq!(
                        row_stands(&tv, OBJECT, row).await,
                        (false, false),
                        "{case}: row {} deleted",
                        row.member
                    );
                }
                assert_eq!(
                    ext_named(&tv, OBJECT, &values).await,
                    vec![false; values.len()],
                    "{case}: every value left obj:ext"
                );
                assert_eq!(
                    header["seq"],
                    record.seeded_seq.to_string(),
                    "{case}: the header is banded"
                );
            } else {
                assert_eq!(result.err(), Some(JobExit::Failed), "{case}: exit 1");
                assert_eq!(errors.len(), 1, "{case}: one ERROR: {errors:?}");
                let bound = format!("after {DR_WRITE_STEP_ATTEMPTS} attempts");
                if !errors[0].contains(&bound) {
                    failures.push(format!("{case}: the ERROR names {bound:?}: {}", errors[0]));
                    continue;
                }
                for part in [
                    names.to_owned(),
                    format!("index {index}"),
                    "test seam".to_owned(),
                ] {
                    assert!(
                        errors[0].contains(&part),
                        "{case}: the ERROR names {part:?}: {}",
                        errors[0]
                    );
                }
                for row in &rows {
                    assert_eq!(
                        row_stands(&tv, OBJECT, row).await,
                        (true, true),
                        "{case}: no row batch ran: {}",
                        row.member
                    );
                }
                assert_eq!(
                    ext_named(&tv, OBJECT, &values).await,
                    vec![true; values.len()],
                    "{case}: every value is still in obj:ext"
                );
                assert_eq!(header["seq"], "12", "{case}: the header is unbanded");
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    // ── D5's allocator purge ────────────────────────────────────────────

    /// A scripted allocator purge for D5's purge pins. It counts its calls;
    /// call `n` answers `n < fail_from` (accepted until then, refused from
    /// then on); and each call numbered in `park_at` reports its number
    /// ([`Self::entered`]) and blocks its blocking-pool thread until
    /// [`Self::release`], or 30 s, so a failed test cannot leave its
    /// runtime's shutdown waiting on it.
    struct PurgeScript {
        purge: AllocatorPurge,
        calls: Arc<AtomicUsize>,
        entered: tokio::sync::mpsc::UnboundedReceiver<usize>,
        release: std::sync::mpsc::Sender<()>,
    }

    impl PurgeScript {
        fn new(fail_from: usize, park_at: &'static [usize]) -> Self {
            let calls = Arc::new(AtomicUsize::new(0));
            let (entered_tx, entered) = tokio::sync::mpsc::unbounded_channel();
            let (release, release_rx) = std::sync::mpsc::channel::<()>();
            let release_rx = std::sync::Mutex::new(release_rx);
            let purge = AllocatorPurge::new({
                let calls = Arc::clone(&calls);
                move || {
                    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if park_at.contains(&n) {
                        let _ = entered_tx.send(n);
                        let _ = release_rx
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv_timeout(Duration::from_secs(30));
                    }
                    n < fail_from
                }
            });
            Self {
                purge,
                calls,
                entered,
                release,
            }
        }

        /// A purge that accepts every call and only counts them.
        fn counting() -> Self {
            Self::new(usize::MAX, &[])
        }

        /// The calls so far.
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        /// The number of the next call that parks, once it has.
        async fn entered(&mut self) -> usize {
            tokio::time::timeout(Duration::from_secs(30), self.entered.recv())
                .await
                .expect("a purge parks within 30 s")
                .expect("the script's purge lives")
        }

        /// Let the parked call answer.
        fn release(&self) {
            self.release.send(()).expect("the script's purge lives");
        }
    }

    /// W-T3, the purge's cadence (fail-on-old). D5 through `discard`, with a
    /// counting purge, on the order pin's object (six rows naming two of
    /// seven values, a batch seam of 4 commands: three row batches, two
    /// `obj:ext` batches and the header): the purge runs once before the
    /// first admission and twice around every transaction — as each attempt
    /// begins, and once it committed — so `1 + 2 × transactions`, 13 when
    /// nothing retries. Killed at `ValueRowsAfterBatch` after the first row
    /// batch it ran 3: the after-commit purge comes before the kill point.
    /// Killed at `ValueRemovalBeforeHeader` it ran `1 + 2 × 5`, 11: none for
    /// the header. Fails without the after-commit purge (in-place
    /// perturbation of `ValueRemovalWalk::send`): 8, 2 and 6.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_purges_once_before_its_first_admission_and_around_every_transaction() {
        use super::super::KillPoint;

        const OBJECT: u64 = 8_541;
        let values = seven_values();
        let lost = LostSet {
            values: vec![removal_of(OBJECT, &values)],
            ..LostSet::default()
        };
        let cases: [(&str, Option<KillPoint>, Option<usize>); 3] = [
            ("uninterrupted", None, None),
            (
                "killed after its first row batch",
                Some(KillPoint::ValueRowsAfterBatch {
                    object: Some(OBJECT),
                    batch: 0,
                }),
                Some(3),
            ),
            (
                "killed before its header",
                Some(KillPoint::ValueRemovalBeforeHeader(Some(OBJECT))),
                Some(1 + 2 * (3 + 2)),
            ),
        ];
        let mut failures = Vec::new();
        for (case, kill_point, killed_purges) in cases {
            let tv = TestValkey::setup().await;
            let mut run = accept_loss_run(&tv).await;
            let script = PurgeScript::counting();
            run.allocator_purge = Some(script.purge.clone());
            seed_values_and_rows(&tv, OBJECT, &values, &six_rows(&values)).await;
            let seams = ExecSeams {
                batch_commands: 4,
                kill_point,
                ..ExecSeams::production()
            };

            let result = discard(&run, &lost, &seams).await;

            let (got, want) = match (result, killed_purges) {
                (Ok(report), None) => {
                    let transactions = report.values[&OBJECT].batches.len();
                    assert_eq!(transactions, 6, "{case}: fixture: six transactions");
                    (script.calls(), 1 + 2 * transactions)
                }
                (Err(JobExit::Failed), Some(purges)) => (script.calls(), purges),
                (other, _) => panic!("{case}: unexpected end {:?}", other.map(|_| ())),
            };
            if got != want {
                failures.push(format!("{case}: {got} purges, want {want}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// W-T4, a purge that fails (fail-on-old). D5 on a bytes-mode gate,
    /// with a purge that accepts its first call and refuses every later
    /// one, on an object whose two rows name [`H0`]: its first transaction
    /// (the row batch) purges as it begins (accepted) and once it landed
    /// (refused). From then on, while the removal holds its purged
    /// admission, it holds a forced top-up of the difference to the
    /// un-purged figure (`value_removal_admission(false) −
    /// value_removal_admission(true)`, in whole KiB, read while its next
    /// purge is parked); afterwards nothing is over-committed. The handle
    /// logs exactly one WARN for its many refusals. The next removal, of
    /// another object, is admitted at the un-purged figure (the gate's peak).
    /// Fails with the purge's answer ignored (in-place perturbation of
    /// `RemovalPurge::run`, every purge taken as accepted): nothing is
    /// topped up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_tops_its_admission_up_to_the_unpurged_figure_when_a_purge_fails() {
        const OBJECTS: [u64; 2] = [8_551, 8_552];
        let tv = TestValkey::setup().await;
        let mut run = bytes_mode_run(&tv).await;
        let mut script = PurgeScript::new(2, &[3]);
        run.allocator_purge = Some(script.purge.clone());
        for object_id in OBJECTS {
            seed_value_object(&tv, object_id, &live_header("12"), 0).await;
        }
        seed_rows(
            &tv,
            OBJECTS[0],
            &[external_field("a", H0), external_field("b", H0)],
        )
        .await;
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("bytes mode");
        let gate = RestoreGate::new(&memory, None);
        let lost = value_removals(&OBJECTS);
        let seams = ExecSeams::production();
        let capture = LogCapture::default();
        let guard = capture.install();

        let (first, during) = tokio::join!(
            remove_layer_values(&run, &lost.values[0], &gate, &seams),
            async {
                let parked = script.entered().await;
                let during = (parked, gate.overcommit_bytes());
                script.release();
                during
            }
        );
        first.expect("the removal completes");
        let over = gate.overcommit_bytes();
        let next = remove_layer_values(&run, &lost.values[1], &gate, &seams).await;
        drop(guard);

        let difference =
            DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES - DR_VALUE_REMOVAL_ADMISSION_BYTES;
        assert_eq!(
            during,
            (3, held_admission_bytes(difference)),
            "at its third purge the removal holds the top-up to the un-purged figure"
        );
        assert_eq!(over, 0, "the top-up goes with the admission");
        next.expect("the next removal completes");
        assert_eq!(
            gate.peak_reserved_bytes(),
            held_admission_bytes(DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES),
            "the next removal is admitted at the un-purged figure"
        );
        let warns = capture
            .messages("WARN")
            .into_iter()
            .filter(|m| m.starts_with("the allocator purge did not run"))
            .count();
        assert_eq!(warns, 1, "one WARN for {} purges", script.calls());
        assert!(script.calls() > 3, "fixture: refused more than once");
    }

    /// W-T4, the post-admission site (fail-on-old). A removal that asked the
    /// gate for the purged figure and waited there while another removal's
    /// purge failed is topped up to the un-purged figure the moment it is
    /// admitted, before its first purge. Removal A, admitted on a 60 MiB
    /// gate, parks in its first purge; removal B logs its foreign-stream
    /// WARN, just before it asks for the purged figure (50 MiB), and then
    /// waits behind A. Released, A's second purge fails, A finishes and
    /// gives its admission back, and B is admitted: at B's first purge
    /// (parked), B alone is admitted and over-commits the top-up, clamped to
    /// the 60 MiB budget as every charge is. Fails without the check at
    /// admission (in-place perturbation of `remove_layer_values`, the
    /// `take_failure_top_up` after `admit` dropped): nothing is
    /// over-committed until B's own purge fails.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_tops_a_removal_admitted_after_a_purge_failed_up_at_its_admission() {
        const OBJECTS: [u64; 2] = [8_561, 8_562];
        const BUDGET: u64 = 60 * 1024 * 1024;
        let tv = TestValkey::setup().await;
        let mut run = bytes_mode_run(&tv).await;
        let mut script = PurgeScript::new(2, &[1, 5]);
        run.allocator_purge = Some(script.purge.clone());
        for object_id in OBJECTS {
            seed_value_object(&tv, object_id, &live_header("12"), 0).await;
        }
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().layer_stream(OBJECTS[1]),
                "not a stream",
                None,
                None,
                false,
            )
            .await
            .expect("SET B's foreign stream key");
        let (at_gate_tx, mut at_gate) = tokio::sync::mpsc::unbounded_channel::<()>();
        let writer = OnLine {
            capture: LogCapture::default(),
            trigger: "stream key held a",
            fired: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            action: Arc::new(move || {
                let _ = at_gate_tx.send(());
            }),
        };
        let guard = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .json()
                .with_writer(writer)
                .with_max_level(tracing::Level::INFO)
                .finish(),
        );
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("bytes mode");
        let gate = RestoreGate::new(&memory, Some(BUDGET));
        let lost = value_removals(&OBJECTS);
        let seams = ExecSeams::production();
        let mut a = pin!(remove_layer_values(&run, &lost.values[0], &gate, &seams));
        let mut b = pin!(remove_layer_values(&run, &lost.values[1], &gate, &seams));

        // A: admitted, then parked in its first purge.
        tokio::select! {
            biased;
            parked = script.entered() => assert_eq!(parked, 1, "A's first purge"),
            ended = a.as_mut() => panic!("A ended before its first purge: {ended:?}"),
        }
        // B: to its foreign-stream WARN, then on into its wait at the gate.
        tokio::select! {
            biased;
            _ = at_gate.recv() => {}
            ended = b.as_mut() => panic!("B ended before its wait: {ended:?}"),
        }
        assert!(
            futures::poll!(b.as_mut()).is_pending(),
            "B waits at the gate"
        );
        assert_eq!(
            (gate.in_flight(), script.calls(), gate.overcommit_bytes()),
            (1, 1, 0),
            "A alone admitted, parked in its first purge; nothing failed yet"
        );

        script.release();
        let (first, second, during) = tokio::join!(a.as_mut(), b.as_mut(), async {
            let parked = script.entered().await;
            let during = (parked, gate.in_flight(), gate.overcommit_bytes());
            script.release();
            during
        });
        drop(guard);

        first.expect("A completes");
        second.expect("B completes");
        let difference =
            DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES - DR_VALUE_REMOVAL_ADMISSION_BYTES;
        assert_eq!(
            during,
            (5, 1, held_admission_bytes(difference.min(BUDGET))),
            "at B's first purge B alone is admitted, already topped up"
        );
        assert_eq!(
            gate.overcommit_bytes(),
            0,
            "the top-up goes with B's admission"
        );
    }

    /// W-T4 in serial mode: a purge that fails does not stop the removal,
    /// which runs on, purging around every transaction: on an object whose
    /// two rows name [`H0`], its row batch, `obj:ext` batch and header make
    /// six purges, every one after the first refused. Serial mode takes no
    /// top-up by construction (its admission holds no byte grant, so
    /// `Admission::top_up` takes nothing, and its gate's over-commit always
    /// reads 0), so nothing here reads the gate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_purges_on_and_takes_no_top_up_in_serial_mode_when_a_purge_fails() {
        const OBJECT: u64 = 8_571;
        let tv = TestValkey::setup().await;
        let mut run = accept_loss_run(&tv).await;
        // No limit read: the reference size, so serial mode.
        let mut config = (*run.config).clone();
        config.pod_memory = crate::config::PodMemoryPlan::derive(
            crate::config::PodMemoryLimit::reference(Some(
                crate::config::pod_memory::ReferenceReason::Unlimited,
            )),
            &crate::config::PlanInputs::from_config(&config),
        )
        .expect("the reference size closes");
        run.config = Arc::new(config);
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("the restore memory");
        assert!(
            matches!(memory, RestoreMemory::Serial { .. }),
            "fixture: serial mode"
        );
        let script = PurgeScript::new(2, &[]);
        run.allocator_purge = Some(script.purge.clone());
        seed_value_object(&tv, OBJECT, &live_header("12"), 0).await;
        seed_rows(
            &tv,
            OBJECT,
            &[external_field("a", H0), external_field("b", H0)],
        )
        .await;
        let gate = RestoreGate::new(&memory, None);
        let lost = value_removals(&[OBJECT]);

        let record = remove_layer_values(&run, &lost.values[0], &gate, &ExecSeams::production())
            .await
            .expect("the removal completes");

        assert_eq!(record.batches.len(), 3, "fixture: three transactions");
        assert_eq!(script.calls(), 6, "two purges around each transaction");
        assert!(script.purge.has_failed(), "fixture: a purge failed");
    }

    /// The run's purge line (`log_value_removal_purge`, which `run::run_job`
    /// logs for accept-loss only). With no purge installed: one WARN naming
    /// the un-purged fail-safe, 158 MiB, and no INFO. With one installed: one
    /// INFO naming the purged admission, 50 MiB, and no WARN. The WARN is the
    /// operator's only sign that a binary installed no purge, and the binary
    /// test that reads the line (`tests/misc/dr_job_binary.rs`) reaches the
    /// `None` arm only off linux-gnu, which CI does not run. Fails with the
    /// `None` arm logged at INFO, or naming the purged figure (in-place
    /// perturbations of `log_value_removal_purge`).
    #[test]
    fn the_purge_line_warns_at_the_unpurged_figure_when_no_purge_is_installed() {
        let cases = [
            (
                None,
                "WARN",
                "INFO",
                "the accept-loss Job runs with no allocator purge installed: its value removal \
                 admits each removal at 158 MiB, the un-purged figure, so the restore memory holds \
                 fewer removals at once",
            ),
            (
                Some(AllocatorPurge::new(|| true)),
                "INFO",
                "WARN",
                "the accept-loss Job's value removal purges the allocator around each of its \
                 transactions and admits each removal at 50 MiB",
            ),
        ];
        for (purge, level, other, want) in cases {
            let installed = purge.is_some();
            let capture = LogCapture::default();
            let guard = capture.install();
            log_value_removal_purge(JobMode::AcceptLoss, purge.as_ref());
            drop(guard);

            assert_eq!(
                capture.messages(level),
                vec![want.to_owned()],
                "purge installed: {installed}: one {level} line"
            );
            assert!(
                capture.messages(other).is_empty(),
                "purge installed: {installed}: no {other} line: {:?}",
                capture.messages(other)
            );
        }
    }

    /// A hand-built D5 walk of `object_id`'s value indices under a header at
    /// `seq` 12, removing `hashes`, with batches of at most `commands`
    /// commands and pages of `page_rows` rows: the walk
    /// `remove_layer_values` builds, for a test that hands it a page's
    /// members itself; `purge` its allocator purge ([`no_purge`] for none).
    fn hand_walk<'w>(
        run: &'w JobRun,
        seams: &'w ExecSeams,
        hashes: &'w BTreeSet<ExtHash>,
        purge: &'w RemovalPurge<'w>,
        oid: &'w str,
        commands: u64,
        page_rows: u64,
    ) -> ValueRemovalWalk<'w> {
        let object_id: u64 = oid.parse().expect("an object id");
        let walk_client = run.valkey.acquire_snapshot().0.next();
        ValueRemovalWalk {
            run,
            seams,
            hashes,
            purge,
            object_id,
            oid,
            prior_seq: 12,
            page_rows,
            limit: BatchLimit {
                commands,
                arg_bytes: RESTORE_BATCH_ARG_BYTES,
            },
            timeout_base: command_timeout(run),
            pages: [
                RankWalk::new(walk_client, &run.keys.obj_fields_index(oid), true),
                RankWalk::new(walk_client, &run.keys.obj_time_samples_index(oid), true),
            ],
            time_sample: false,
            batch: ValueRemovalBatch::new(object_id, 12),
            counts: [IndexCounts::default(); 2],
            rows: 0,
            row_batches: 0,
            batches: Vec::new(),
        }
    }

    /// A removal's allocator purge with no purge installed, on `admission`
    /// (charged as the purged figure, as the hand-built walks admit it:
    /// `value_removal_admission(true)`), for [`hand_walk`].
    fn no_purge<'w>(admission: &'w Admission<'w>) -> RemovalPurge<'w> {
        RemovalPurge {
            purge: None,
            admission,
            purged_charge: true,
            top_up: std::sync::Mutex::new(None),
        }
    }

    /// I-8, `rows` exact: D5 counts each row it deletes once. A rank page
    /// names each of its members once, and no member an earlier page named,
    /// so a page never repeats a row the walk sent; a row already pending
    /// that a page names again (a hand-built page here) is not added or
    /// counted again all the same. Rows `a` and `m` name [`H0`]; a
    /// hand-built walk handed the page `[a, a, m]` adds `a` once: the two
    /// rows count 2, in one batch, and the deleted count of `obj:flds` — the
    /// figure its end check subtracts from the start `ZCARD` — is 2.
    ///
    /// Fails with `take_page`'s pending-row check dropped (in-place
    /// perturbation): `a` is counted deleted twice, 3.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d5_counts_each_row_once_however_often_a_page_names_its_member() {
        use super::super::memory::DR_VALUE_REMOVAL_PAGE_ROWS;

        const OBJECT: u64 = 8_534;
        let oid = OBJECT.to_string();
        let member = |name: &str| KeySchema::field_member("1", name);
        let hashes = BTreeSet::from([H0, H1]);

        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        seed_value_object(&tv, OBJECT, &live_header("12"), 0).await;
        seed_rows(
            &tv,
            OBJECT,
            &[external_field("a", H0), external_field("m", H0)],
        )
        .await;
        let memory = RestoreMemory::derive(&run.config, run.mode).expect("the restore memory");
        let gate = RestoreGate::new(&memory, None);
        let admission = gate
            .admit(value_removal_admission(true))
            .await
            .expect("admitted");
        let seams = ExecSeams::production();
        let purge = no_purge(&admission);
        let mut walk = hand_walk(
            &run,
            &seams,
            &hashes,
            &purge,
            &oid,
            crate::storage::layer::dr_restore::RESTORE_BATCH_COMMANDS,
            DR_VALUE_REMOVAL_PAGE_ROWS,
        );
        walk.take_page(vec![member("a"), member("a"), member("m")])
            .await
            .expect("the page");
        walk.flush_rows().await.expect("the flush");
        assert_eq!(
            (walk.rows, walk.batches.len(), walk.counts[0].deleted),
            (2, 1, 2),
            "a named twice while pending: two rows, one batch, two deleted from obj:flds"
        );
        for name in ["a", "m"] {
            assert_eq!(
                row_stands(&tv, OBJECT, &external_field(name, H0)).await,
                (false, false),
                "row {name} deleted"
            );
        }
    }

    /// Test review 7 (v) (fail-on-old). A lost session whose mapping the
    /// store now has naming another object than its plan saw, and a layer
    /// name its plan saw no mapping of that the store maps: each mapping is
    /// left as found, with its WARN (AL8, and the no-mapping form), counted
    /// as left, and the discards run on — the session discarded, its object
    /// registered. With a found mapping treated as the abandoned delete
    /// treats it, the run exits 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lost_sessions_unmap_that_finds_another_object_leaves_it_and_runs_on() {
        const LOST: &str = "s-lost-moved";
        const SEEN: u64 = 8_701;
        const FOUND: u64 = 8_702;
        const UNSEEN: u64 = 8_703;
        let tv = TestValkey::setup().await;
        let run = accept_loss_run(&tv).await;
        map_layer(&tv, LOST, "l.usda", FOUND).await;
        map_layer(&tv, LOST, "other.usda", UNSEEN).await;
        let lost = LostSet {
            sessions: vec![LostSession {
                session_id: LOST.to_owned(),
                evidence: "the surviving mapping of layer l.usda".to_owned(),
                objects: vec![SEEN],
                mappings: vec![("l.usda".to_owned(), SEEN)],
                layer_names: vec!["l.usda".to_owned(), "other.usda".to_owned()],
                backup_listed: false,
                unconfirmed: None,
            }],
            ..LostSet::default()
        };
        let capture = LogCapture::default();
        let guard = capture.install();

        let report = discard(&run, &lost, &ExecSeams::production()).await;
        drop(guard);

        let report = report.expect("the discards run on");
        assert_eq!(
            report.left,
            vec![
                (LOST.to_owned(), "l.usda".to_owned(), SEEN, FOUND),
                (LOST.to_owned(), "other.usda".to_owned(), 0, UNSEEN),
            ]
        );
        assert_eq!(report.sessions_discarded, vec![LOST.to_owned()]);
        assert_eq!(dead(&tv).await, vec![SEEN.to_string()]);
        assert_eq!(
            mapping_of(&tv, LOST, "l.usda").await,
            Some(FOUND.to_string())
        );
        assert_eq!(
            mapping_of(&tv, LOST, "other.usda").await,
            Some(UNSEEN.to_string())
        );
        let left: Vec<String> = capture
            .messages("WARN")
            .into_iter()
            .filter(|m| m.starts_with("the accept-loss Job left layer"))
            .collect();
        assert_eq!(
            left,
            vec![
                render_left_mapping(LOST, "l.usda", FOUND, SEEN),
                format!(
                    "the accept-loss Job left layer other.usda of session {LOST} mapped to \
                     object {UNSEEN}: its plan saw no mapping of it; it removes a mapping only \
                     while it names what its plan saw"
                ),
            ]
        );
        assert_eq!(capture.messages("ERROR"), Vec::<String>::new());
    }

    /// Test review 7 (vi) (fail-on-old). D6 for a session with no record
    /// in the store: the rewrite answers that the session is absent, so
    /// nothing is PUT and the session is not counted as rewritten; the
    /// discards end `Ok`. With an absent session counted as rewritten, the
    /// report names it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d6_rewrites_nothing_for_a_session_with_no_record() {
        const ABSENT: &str = "s-no-record";
        let tv = TestValkey::setup().await;
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mut run = open_run(&tv, Arc::clone(&store) as _).await;
        run.mode = JobMode::AcceptLoss;
        let lost = LostSet {
            rewrites: BTreeSet::from([ABSENT.to_owned()]),
            ..LostSet::default()
        };

        let report = discard(&run, &lost, &ExecSeams::production())
            .await
            .expect("the discards end Ok");

        assert!(
            report.rewritten.is_empty(),
            "not rewritten: {:?}",
            report.rewritten
        );
        assert!(
            crate::checkpoint::store::CheckpointStore::get_session_checkpoint(
                store.as_ref(),
                ABSENT
            )
            .await
            .expect("the in-memory store answers")
            .is_none(),
            "nothing PUT"
        );
    }

    /// [`SESSION`]'s record, as the session backup's build reads it.
    async fn seed_record(tv: &TestValkey) {
        let _: i64 = tv
            .client()
            .pool
            .hset(
                tv.key_schema().session_meta(SESSION),
                vec![
                    ("name", "moved"),
                    ("base_path", "omniverse://example.com/moved.usd"),
                    ("created_by", "user"),
                    ("created_at", "1"),
                ],
            )
            .await
            .expect("HSET record");
    }

    /// D3's early rewrite fails (fail-on-old): an over-limit lost layer
    /// whose session backup's PUT fails stops the discards before the
    /// unmap, so the layer stays mapped and indexed while the session's
    /// backup is not yet rewritten; the step's ERROR names the layer left
    /// out; nothing is registered. With the early rewrite a no-op, the
    /// unmap runs before D6's rewrite fails, so the layer is unmapped.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d3_a_failed_over_limit_rewrite_stops_before_the_unmap_and_names_the_layer() {
        use crate::checkpoint::adapters::CheckpointFaultInjector;
        use crate::checkpoint::store::CheckpointError;

        /// Fails every session backup's PUT, `Permanent`.
        struct FailSessionPuts;
        impl CheckpointFaultInjector for FailSessionPuts {
            fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
                if op == "put_session_checkpoint" {
                    return Err(CheckpointError::Permanent {
                        source: "the session put fails".into(),
                    });
                }
                Ok(())
            }
        }

        const OBJECT: u64 = 8_801;
        let tv = TestValkey::setup().await;
        let store = InMemoryCheckpointStore::new().with_fault_injector(Arc::new(FailSessionPuts));
        let mut run = open_run(&tv, Arc::new(store)).await;
        run.mode = JobMode::AcceptLoss;
        seed_record(&tv).await;
        map(&tv, OBJECT).await;
        let lost = one_lost_layer(OBJECT, LostLayerKind::OverLimitBackups);
        let capture = LogCapture::default();
        let guard = capture.install();

        let report = discard(&run, &lost, &ExecSeams::production()).await;
        drop(guard);

        assert!(
            matches!(report, Err(JobExit::Failed)),
            "the discards stop: {report:?}"
        );
        assert_eq!(
            mapped(&tv).await,
            (Some(OBJECT.to_string()), true),
            "the unmap never ran"
        );
        assert!(dead(&tv).await.is_empty(), "nothing registered");
        let errors = capture.messages("ERROR");
        assert_eq!(errors.len(), 1, "one ERROR: {errors:?}");
        assert!(
            errors[0].starts_with(&format!(
                "the accept-loss Job could not rewrite the session backup of session {SESSION} \
                 without layers {LAYER}, whose backups are over the per-delta entry limit after 1 \
                 attempts: "
            )),
            "the ERROR names the layer left out: {}",
            errors[0]
        );
    }

    /// An over-limit lost layer the store now maps to another object than
    /// its plan saw (fail-on-old): D3's early rewrite finds it mapped to
    /// that other object, so it leaves nothing out; the unmap leaves the
    /// mapping as found (AL8); and D6 still rewrites the session from the
    /// whole store, as it rewrites every session D3 rewrote first, so the
    /// backup names the layer's mapping as found. With D3's sessions not
    /// joining D6's set, D6 rewrites nothing (the unmap removed no
    /// mapping).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn d6_rewrites_an_over_limit_layers_session_whose_unmap_found_another_object() {
        const STALE: u64 = 8_901;
        const NEWER: u64 = 8_902;
        let tv = TestValkey::setup().await;
        let store = Arc::new(InMemoryCheckpointStore::new());
        let mut run = open_run(&tv, Arc::clone(&store) as _).await;
        run.mode = JobMode::AcceptLoss;
        seed_record(&tv).await;
        map(&tv, NEWER).await;
        let lost = one_lost_layer(STALE, LostLayerKind::OverLimitBackups);

        let report = discard(&run, &lost, &ExecSeams::production())
            .await
            .expect("the discards end Ok");

        assert_eq!(
            report.left,
            vec![(SESSION.to_owned(), LAYER.to_owned(), STALE, NEWER)]
        );
        assert!(report.over_limit_unmapped.is_empty(), "{report:?}");
        assert_eq!(
            report.rewritten,
            vec![SESSION.to_owned()],
            "D6 rewrites the session D3 rewrote first"
        );
        assert_eq!(mapped(&tv).await, (Some(NEWER.to_string()), true));
        let blob = crate::checkpoint::store::CheckpointStore::get_session_checkpoint(
            store.as_ref(),
            SESSION,
        )
        .await
        .expect("the in-memory store answers")
        .expect("the backup was PUT");
        let backup = crate::checkpoint::SessionCheckpoint::from_bytes(&blob.bytes)
            .expect("the backup decodes");
        assert_eq!(
            backup.layer_mappings,
            BTreeMap::from([(LAYER.to_owned(), NEWER)]),
            "the backup names the mapping as found"
        );
    }

    /// `layer` of `session` mapped to `object_id`, with its layers-index
    /// member.
    async fn map_layer(tv: &TestValkey, session: &str, layer: &str, object_id: u64) {
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().layer_mapping(session, layer),
                object_id.to_string(),
                None,
                None,
                false,
            )
            .await
            .expect("SET mapping");
        let _: i64 = tv
            .client()
            .pool
            .sadd(tv.key_schema().layers_index(session), hex_encode(layer))
            .await
            .expect("SADD");
    }

    /// The object `layer` of `session` is mapped to.
    async fn mapping_of(tv: &TestValkey, session: &str, layer: &str) -> Option<String> {
        tv.client()
            .pool
            .get(tv.key_schema().layer_mapping(session, layer))
            .await
            .expect("GET")
    }

    /// A [`LogCapture`] that runs `action` once, synchronously, when a line
    /// containing `trigger` is logged: a hook between two steps of the code
    /// under test that logs between them.
    #[derive(Clone)]
    struct OnLine {
        capture: LogCapture,
        trigger: &'static str,
        fired: Arc<std::sync::atomic::AtomicBool>,
        action: Arc<dyn Fn() + Send + Sync>,
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for OnLine {
        type Writer = OnLine;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for OnLine {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let line = String::from_utf8_lossy(buf);
            if line.contains(self.trigger)
                && !self.fired.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                (self.action)();
            }
            std::io::Write::write(&mut self.capture, buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
