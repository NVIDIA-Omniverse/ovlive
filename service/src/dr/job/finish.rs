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

//! The Job's finish (SDD SR-9.8, SR-9.9; ADD 9): the authoritative recount
//! of the large values' references, the object-id counter's floor, and the
//! un-hold. Both bodies run them in this order through one function,
//! [`finish_run`] — the restore body ([`super::restore`]) after the writer
//! and the coverage check's finalise, the accept-loss body
//! ([`super::accept_loss`]) after its discards (defer-65) — between the
//! recount and the stays-held exit or the floor, logging the restore
//! summary.
//!
//! # The un-hold is gated on a completed recount, three ways
//!
//! 1. **Type.** [`unhold`] takes a [`CompletedRecount`], which only
//!    [`recount`]'s `Ok` makes, and passes its receipt to
//!    `storage::dr_hold::unhold`, which requires one: no un-hold without a
//!    completed, error-free recount compiles.
//! 2. **No write since.** [`recount`] records how many write units the
//!    permit had let begin when it ended; [`unhold`] refuses (ERROR, exit 1,
//!    the marker untouched) when any began since, so every write the Job
//!    makes comes before the recount that counts it.
//! 3. **One deleter.** `scripts/dr_unhold.lua` is the hold marker's only
//!    deleter, called only from `storage::dr_hold::unhold`, called only from
//!    [`unhold`].
//!
//! # A kill converges
//!
//! A Job killed during the recount leaves some counts exact and makes no
//! receipt; one killed after it leaves them all exact. Either way the marker
//! stays, and a re-run writes nothing new, recounts from scratch to the same
//! values and un-holds. A recount whose walk was not shown complete — the
//! store changed while the Job read it (`storage::scan_guard`, *The DR Jobs'
//! walks*) — fails before it sets any count, since every set follows every
//! walk, and its re-run walks from the start. A Job
//! killed after the un-hold leaves the store not held, so a re-run exits 0
//! having written nothing; only its last log line is lost, since the summary
//! comes before the un-hold.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::execute::{write_step, At, ExecSeams, ExecutionReport, StepError};
use super::plan::model::{CoverageReport, PlanOutcome, RestorePlan};
use super::plan::report;
use super::{JobExit, JobRun};
use crate::storage::dr_hold::{self, UnholdOutcome};
use crate::storage::dr_job::read_dead_object_ids;
use crate::storage::ext_reconcile::{
    recount_authoritative, RecountError, RecountReceipt, RecountReport,
};

/// The largest object id the service can allocate: 2^53 − 1. Ids travel as
/// JSON numbers and Lua numbers, which are exact only up to it. The un-hold
/// refuses (exit 3) a counter floor above it.
pub const MAX_OBJECT_ID: u64 = (1 << 53) - 1;

/// What a body hands the shared finish ([`finish_run`]).
pub(crate) struct FinishInputs<'a> {
    /// The plan the run wrote.
    pub plan: &'a RestorePlan,
    /// Its finalised coverage report.
    pub report: &'a CoverageReport,
    /// What the restore writer wrote.
    pub execution: &'a ExecutionReport,
    /// When the run stays held: its ERROR, logged after the summary, before
    /// exit 3. `None`: the run un-holds.
    pub stays_held: Option<&'a (dyn Fn() + Send + Sync)>,
    /// Lines logged after the restore summary, before the stays-held ERROR
    /// or the un-hold (accept-loss's totals); restore logs none.
    pub after_summary: &'a (dyn Fn() + Send + Sync),
    /// Test only: run one permit unit between the recount and the floor, as
    /// a write would.
    pub write_after_recount: bool,
}

/// The shared finish (module docs): the recount ([`recount`]), the restore
/// summary (`plan::report::log_restore_summary`) and `inputs.after_summary`,
/// then the stays-held ERROR and exit 3, or the counter floor
/// ([`counter_floor`]) and the un-hold ([`unhold`]). The summary comes
/// before the un-hold, so a kill after the un-hold loses only its last line.
/// What it reached goes into `record`.
pub(crate) async fn finish_run(
    run: &JobRun,
    seams: &ExecSeams,
    inputs: FinishInputs<'_>,
    record: &mut FinishRecord,
) -> JobExit {
    let recount = match recount(run, seams).await {
        Ok(recount) => recount,
        Err(exit) => return exit,
    };
    record.recount = Some(recount.report().clone());
    report::log_restore_summary(
        run.mode,
        inputs.plan,
        inputs.report,
        inputs.execution,
        recount.report(),
    );
    (inputs.after_summary)();

    if let Some(stays_held) = inputs.stays_held {
        stays_held();
        return JobExit::StaysHeld;
    }

    if inputs.write_after_recount && run.permit.begin_unit().await.is_err() {
        return JobExit::Failed;
    }
    let floor = match counter_floor(
        run,
        inputs.plan.listed_max_object_id,
        inputs.plan.surviving_max_object_id,
        &recount,
    )
    .await
    {
        Ok(floor) => floor,
        Err(exit) => return exit,
    };
    record.floor = Some(floor);
    if let Err(exit) = seams.stop_at(run.mode, At::BeforeUnhold) {
        return exit;
    }
    let (outcome, exit) = unhold(run, &recount, &floor, seams).await;
    record.unhold = outcome;
    exit
}

/// A recount that completed without error ([`recount`]): what [`unhold`]
/// needs.
#[derive(Debug)]
pub struct CompletedRecount {
    receipt: RecountReceipt,
    /// The permit's `units_begun` when the recount ended.
    units_at_end: u64,
}

impl CompletedRecount {
    /// What the recount did.
    pub fn report(&self) -> &RecountReport {
        self.receipt.report()
    }
}

/// Why a recount's `before_write` refused.
enum RecountStop {
    /// The permit was revoked; the frame logs why.
    Revoked,
    /// The test kill point after the first group.
    Killed,
}

/// The authoritative recount (`storage::ext_reconcile::recount_authoritative`),
/// each group of count sets behind the write permit. Logs its result: one
/// INFO on success; one ERROR and exit 1 on a store fault, a keyspace walk
/// whose slot map changed or whose node pin missed its primary's slots
/// among them (the ERROR names the walk and the maps, or the slots expected
/// and served); one ERROR and exit 1, before any count is set, on a walk its
/// completeness check did not show complete (the ERROR says plainly that
/// the store changed while the Job read it, and names the walk and the
/// counts it compared); one ERROR and exit 3 on a record it cannot read
/// (PD-4), or on a keyspace walk whose `CLUSTER NODES` gave no usable
/// myself line (the ERROR names the command, the node and what was seen).
/// Call it after every other write of the run.
pub async fn recount(run: &JobRun, seams: &ExecSeams) -> Result<CompletedRecount, JobExit> {
    let mode = run.mode;
    let mut groups = 0usize;
    let result = recount_authoritative(&run.valkey, &run.keys, || {
        groups += 1;
        let group = groups;
        async move {
            if group > 1 && seams.stop_at(run.mode, At::RecountAfterFirstChunk).is_err() {
                return Err(RecountStop::Killed);
            }
            run.permit
                .begin_unit()
                .await
                .map_err(|_| RecountStop::Revoked)
        }
    })
    .await;
    match result {
        Ok(receipt) => {
            let units_at_end = run.permit.units_begun();
            let r = receipt.report();
            tracing::info!(
                mode = mode.name(),
                values = r.values,
                objects = r.objects,
                created = r.created,
                raised = r.raised,
                lowered = r.lowered,
                unchanged = r.unchanged,
                repaired = r.repaired,
                owed_only = r.owed_only,
                "the {mode} Job recounted the references of {} large values over {} layer \
                 objects: {} counts created, {} raised, {} lowered, {} unchanged, {} unreadable \
                 counts rewritten",
                r.values,
                r.objects,
                r.created,
                r.raised,
                r.lowered,
                r.unchanged,
                r.repaired
            );
            Ok(CompletedRecount {
                receipt,
                units_at_end,
            })
        }
        Err(RecountError::Store(error)) => {
            tracing::error!(
                mode = mode.name(),
                error = %error,
                "the {mode} Job could not recount the references of the large values: {error}; \
                 it exits 1 so Kubernetes retries it, and the service stays held"
            );
            Err(JobExit::Failed)
        }
        Err(RecountError::Incomplete(why)) => {
            tracing::error!(
                mode = mode.name(),
                error = %why,
                "the {mode} Job could not recount the references of the large values: the \
                 backing store changed while the Job read it ({why}); it set no count and exits \
                 1 so Kubernetes retries it, and the service stays held"
            );
            Err(JobExit::Failed)
        }
        Err(RecountError::WitnessMissing(why)) => {
            tracing::error!(
                mode = mode.name(),
                error = %why,
                "the {mode} Job cannot recount the references of the large values: {why}; it \
                 does not use a walk it cannot check, so the service stays held and the Job \
                 exits 3; open a support case with this line"
            );
            Err(JobExit::StaysHeld)
        }
        Err(RecountError::Corrupt { what, key }) => {
            tracing::error!(
                mode = mode.name(),
                record = %what,
                key = %key,
                "the {mode} Job cannot recount the references of the large values: the backing \
                 store holds a record it cannot read ({what}: {key}); the service stays held and \
                 the Job exits 3; open a support case with this line"
            );
            Err(JobExit::StaysHeld)
        }
        // The kill point logged its line; a revoked permit's cause is the
        // frame's to log.
        Err(RecountError::Stopped(RecountStop::Killed | RecountStop::Revoked)) => {
            Err(JobExit::Failed)
        }
    }
}

/// The object-id counter's floor at the un-hold (SR-9.9, defer-62): the
/// largest of the terms that exist, each saturating. The script also keeps
/// the counter's current value, the fifth term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CounterFloor {
    /// The floor: the largest term, 0 when none exists.
    pub floor: u64,
    /// `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`.
    pub margin: u64,
    /// The counters sidecar's `max_object_id_observed` (this run's PD2 read)
    /// + margin; absent with no backup store.
    pub sidecar: Option<u64>,
    /// The largest listed object id + margin; absent when nothing was listed.
    pub listed: Option<u64>,
    /// The largest surviving object id (the plan's, or the recount walk's
    /// when larger) + 1.
    pub surviving: Option<u64>,
    /// The largest id on the dead-object set + 1 (defer-62): the ids the
    /// Job's own abandoned deletes registered, whose headers its wipe
    /// removed.
    pub dead: Option<u64>,
}

impl CounterFloor {
    /// The terms that exist, by name, in a fixed order.
    pub fn terms(&self) -> Vec<(&'static str, u64)> {
        [
            ("sidecar", self.sidecar),
            ("listed", self.listed),
            ("surviving", self.surviving),
            ("dead", self.dead),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|v| (name, v)))
        .collect()
    }
}

/// The counter floor: the sidecar term from the run's PD2 read, the listed
/// and surviving terms from the plan (`listed_max_object_id`,
/// `surviving_max_object_id`) and the recount's own header walk, and the
/// dead-object set's largest member, read now. A read error logs one ERROR
/// and exits 1; a floor above [`MAX_OBJECT_ID`] logs one ERROR naming its
/// largest term and exits 3.
pub async fn counter_floor(
    run: &JobRun,
    listed_max_object_id: Option<u64>,
    surviving_max_object_id: Option<u64>,
    recount: &CompletedRecount,
) -> Result<CounterFloor, JobExit> {
    let mode = run.mode;
    let margin = run.config.checkpoint_counter_safety_margin;
    let dead = match read_dead_object_ids(&run.valkey, &run.keys).await {
        Ok(dead) => dead.into_iter().max(),
        Err(e) => {
            tracing::error!(
                mode = mode.name(),
                error = %e,
                "the {mode} Job could not read the dead-object set before clearing the hold: \
                 {e}; it exits 1 so Kubernetes retries it, and the service stays held"
            );
            return Err(JobExit::Failed);
        }
    };
    let surviving = match (surviving_max_object_id, recount.report().max_object_id) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let mut floor = CounterFloor {
        margin,
        sidecar: run
            .counters_sidecar_max
            .get()
            .map(|max| max.saturating_add(margin)),
        listed: listed_max_object_id.map(|max| max.saturating_add(margin)),
        surviving: surviving.map(|max| max.saturating_add(1)),
        dead: dead.map(|max| max.saturating_add(1)),
        floor: 0,
    };
    let largest = floor.terms().into_iter().max_by_key(|(_, value)| *value);
    floor.floor = largest.map_or(0, |(_, value)| value);
    if let Some((term, value)) = largest.filter(|(_, value)| *value > MAX_OBJECT_ID) {
        tracing::error!(
            mode = mode.name(),
            term,
            value,
            max = MAX_OBJECT_ID,
            "the {mode} Job cannot clear the hold: the object-id counter would pass \
             {MAX_OBJECT_ID}, the largest id the service can allocate ({term} = {value}); the \
             service stays held and the Job exits 3; open a support case with this line"
        );
        return Err(JobExit::StaysHeld);
    }
    Ok(floor)
}

/// The un-hold (SR-4.2, SR-9.9): refuse when a write unit began after the
/// recount; otherwise run `storage::dr_hold::unhold` through a write step
/// (three attempts, each behind the permit; a transient store error is
/// retried, nothing else is), and log its outcome. Exits 0 (cleared, or
/// found already cleared), 3 (the object-id counter is not a decimal
/// integer, or the lease, the counter or the marker is a key of the wrong
/// type; nothing cleared, the service stays held), 4 (another incident was
/// recorded meanwhile; nothing cleared) or 1 (the lease is no longer the
/// Job's, a write after the recount, or a store fault). Returns the script's
/// answer, when it answered, with the exit. The floor's own exit 3 (above
/// [`MAX_OBJECT_ID`]) is [`counter_floor`]'s, before this.
pub async fn unhold(
    run: &JobRun,
    recount: &CompletedRecount,
    floor: &CounterFloor,
    seams: &ExecSeams,
) -> (Option<UnholdOutcome>, JobExit) {
    let mode = run.mode;
    if run.permit.units_begun() != recount.units_at_end {
        tracing::error!(
            mode = mode.name(),
            units_at_recount = recount.units_at_end,
            units_now = run.permit.units_begun(),
            "the {mode} Job did not clear the hold: a write ran after its recount, so the \
             reference counts may not cover it; it exits 1 and the service stays held"
        );
        return (None, JobExit::Failed);
    }
    let incident = run.incident.as_str();
    // The Job's own lease value, but for the test seam's (never in a release
    // build), so a test can make the script answer `lease_lost`.
    let lease_value = seams.unhold_lease_value(&run.lease_holder);
    // The test seam that loses the first attempt's reply after its script
    // ran, so the write step's retry is the real in-doubt path.
    let reply_lost = AtomicBool::new(seams.unhold_reply_lost());
    let outcome = match write_step(
        run,
        &|| format!("clear the hold for incident {incident}"),
        || async {
            let answer = dr_hold::unhold(
                &run.valkey,
                &run.keys,
                &recount.receipt,
                incident,
                lease_value,
                floor.floor,
            )
            .await;
            if answer.is_ok() && reply_lost.swap(false, Ordering::SeqCst) {
                return Err(StepError::Retry("reply lost (test)".to_owned()));
            }
            answer.map_err(|e| {
                if crate::storage::client::is_transient_fred_error(&e) {
                    StepError::Retry(e.to_string())
                } else if crate::storage::client::is_wrongtype_reply(&e) {
                    // A lease, counter or marker of the wrong type: the script
                    // wrote nothing, and no retry can retype a key (PD-4).
                    // The marker's key is not named here: every production
                    // site naming it is listed by the single-deleter guard
                    // (`tests/misc/hold_unhold_scan.rs`).
                    let lease_key = run.keys.dr_job_lease();
                    let counter_key = run.keys.counter_object_id();
                    tracing::error!(
                        mode = mode.name(),
                        incident = %incident,
                        error = %e,
                        lease_key = %lease_key,
                        counter_key = %counter_key,
                        "the {mode} Job cannot clear the hold for incident {incident}: one of the \
                         keys it reads to clear it holds the wrong type of value ({e}): the lease \
                         {lease_key}, the object-id counter {counter_key} or the hold marker; it \
                         cleared nothing; the service stays held and the Job exits 3; open a \
                         support case with this line"
                    );
                    StepError::Exit(JobExit::StaysHeld)
                } else {
                    StepError::Fatal(e.to_string())
                }
            })
        },
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(exit) => return (None, exit),
    };
    if let Err(exit) = seams.stop_at(run.mode, At::AfterUnhold) {
        return (Some(outcome), exit);
    }
    let exit = match &outcome {
        UnholdOutcome::Unheld { old, new } => {
            tracing::info!(
                mode = mode.name(),
                incident = %incident,
                counter_was = *old,
                counter = *new,
                floor = floor.floor,
                margin = floor.margin,
                sidecar_term = floor.sidecar,
                listed_term = floor.listed,
                surviving_term = floor.surviving,
                dead_term = floor.dead,
                "the {mode} Job cleared the hold for incident {incident}: the object-id counter \
                 is {new} (it was {old}), above every id a backup, a surviving layer or a deleted \
                 layer used; every replica restarts within seconds"
            );
            JobExit::Done
        }
        UnholdOutcome::AlreadyUnheld { old, new } => {
            tracing::info!(
                mode = mode.name(),
                incident = %incident,
                counter_was = *old,
                counter = *new,
                floor = floor.floor,
                "the hold for incident {incident} was already cleared; the object-id counter is \
                 {new}"
            );
            JobExit::Done
        }
        UnholdOutcome::CounterUnreadable { value } => {
            let counter_key = run.keys.counter_object_id();
            tracing::error!(
                mode = mode.name(),
                incident = %incident,
                counter_key = %counter_key,
                value = %value,
                "the {mode} Job cannot clear the hold for incident {incident}: the object-id \
                 counter {counter_key} holds {value:?}, which is not a decimal integer; it \
                 cleared nothing; the service stays held and the Job exits 3; open a support \
                 case with this line"
            );
            JobExit::StaysHeld
        }
        UnholdOutcome::OtherIncident { held } => {
            tracing::error!(
                mode = mode.name(),
                incident = %incident,
                held_incident = %held,
                "held under a different incident {held}: a new incident was recorded while the \
                 {mode} Job ran for incident {incident}; it cleared nothing and exits 4"
            );
            JobExit::DifferentIncident
        }
        UnholdOutcome::LeaseLost { holder } => {
            tracing::error!(
                mode = mode.name(),
                incident = %incident,
                holder = %holder,
                "the {mode} Job's lease was no longer its own when it went to clear the hold; it \
                 cleared nothing and exits 1; the service stays held, and a re-run continues from \
                 what it wrote"
            );
            JobExit::Failed
        }
    };
    (Some(outcome), exit)
}

/// What the finish reached, for a test's observer
/// (`super::restore::RestoreBody::with_finish_observer`,
/// `super::accept_loss::AcceptLossBody::with_finish_observer`): reported once
/// per run whose body returned after its coverage check was finalised,
/// whatever its exit. On an accept-loss run that stopped before its recount
/// (a discard defect, a failed discard) only `outcome` and `exit` are set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinishRecord {
    /// The run's outcome after the coverage check was finalised; for
    /// accept-loss, after its discards (stays held when it refused to
    /// discard, or when a held item remains that it does not discard).
    pub outcome: PlanOutcome,
    /// The recount's report, when it completed.
    pub recount: Option<RecountReport>,
    /// The counter floor, when computed.
    pub floor: Option<CounterFloor>,
    /// The un-hold's outcome, when the script answered.
    pub unhold: Option<UnholdOutcome>,
    /// The run's exit.
    pub exit: JobExit,
}

/// A test's view of the finish ([`FinishRecord`]).
pub type FinishObserver = Arc<dyn Fn(&FinishRecord) + Send + Sync>;
