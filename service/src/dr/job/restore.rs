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

//! The restore Job's body (`live-edit dr restore`): it plans the restore
//! ([`RestorePlanner`]), writes the plan ([`execute_restore`]), then finishes
//! ([`RestoreBody::execute`], SDD SR-9.8, SR-9.9, SR-14.6):
//!
//! 1. finalise the coverage check (no write) and log the plan's summary and
//!    the writer's ([`restore_phase`], which the accept-loss body runs too);
//! 2. log the outcome: writable, or stays held (the plan's held items, or a
//!    finalised value uncovered or unknown);
//! 3. recount the large values' references ([`finish::recount`]), in both
//!    outcomes, so a run that stays held leaves its restorable objects
//!    written and their counts exact;
//! 4. log the restore summary (`plan::report::log_restore_summary`), before
//!    the un-hold, so a kill after the un-hold loses only its last line;
//! 5. stays held: one ERROR and exit 3; writable: the counter floor
//!    ([`finish::counter_floor`]) and the un-hold ([`finish::unhold`]), exit
//!    0, 3 (a floor above the largest allocatable id, an object-id counter
//!    that is not a decimal integer, or a key of the wrong type at the
//!    un-hold), 4 or 1.
//!
//! Steps 3 to 5 are the shared finish ([`finish::finish_run`]).
//!
//! A run killed anywhere before the un-hold leaves the service held, and a
//! re-run converges on what it wrote, writing nothing new (`super::finish`).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::execute::{execute_restore, ExecSeams, ExecutionReport, MemoryReport};
use super::finish::{self, FinishInputs, FinishObserver, FinishRecord};
use super::plan::model::{CoverageReport, PlanOutcome, RestorePlan};
use super::plan::{report, RestorePlanner};
use super::run::{JobBody, JobRun};
use super::JobExit;

/// A test's view of a finished plan and its finalised coverage report
/// ([`RestoreBody::with_plan_observer`]).
pub type PlanObserver = Arc<dyn Fn(&RestorePlan, &CoverageReport) + Send + Sync>;

/// A test's view of a written plan and what the writer did
/// ([`RestoreBody::with_execution_observer`]).
pub type ExecutionObserver = Arc<dyn Fn(&RestorePlan, &ExecutionReport) + Send + Sync>;

/// The test observers both bodies carry.
#[derive(Default)]
pub(crate) struct Observers {
    /// Called once the plan's coverage check is finalised.
    pub plan: Option<PlanObserver>,
    /// Called once the writer finished.
    pub execution: Option<ExecutionObserver>,
    /// Called once with what the finish reached.
    pub finish: Option<FinishObserver>,
}

/// What the restore phase ([`restore_phase`]) leaves for the rest of the
/// body.
pub(crate) struct RestorePhase {
    /// What the writer wrote, with the planner's lineage-stage memory.
    pub execution: ExecutionReport,
    /// The finalised coverage report.
    pub report: CoverageReport,
    /// The plan's outcome after the coverage check was finalised.
    pub outcome: PlanOutcome,
}

/// The restore phase both bodies run (module docs, step 1): write `plan`
/// ([`execute_restore`], which checks it first), with `planner_memory` as
/// the report's `planner_memory`; finalise the coverage check; log the
/// plan's summary and the writer's; call the execution and plan observers.
/// `Err` ends the run, already logged.
pub(crate) async fn restore_phase(
    run: &JobRun,
    plan: &RestorePlan,
    seams: &ExecSeams,
    planner_memory: Option<MemoryReport>,
    observers: &Observers,
) -> Result<RestorePhase, JobExit> {
    let mut execution = execute_restore(run, plan, seams).await?;
    if let Some(planner_memory) = planner_memory {
        execution.planner_memory = planner_memory;
    }
    let report = plan.coverage.finalize(run).await?;
    report::log_summary(plan, &report);
    report::log_execution(run.mode, &execution);
    if let Some(observer) = &observers.execution {
        observer(plan, &execution);
    }
    if let Some(observer) = &observers.plan {
        observer(plan, &report);
    }
    let outcome = plan.outcome(&report);
    Ok(RestorePhase {
        execution,
        report,
        outcome,
    })
}

/// The restore Job's body.
#[derive(Default)]
pub struct RestoreBody {
    observers: Observers,
    seams: ExecSeams,
    /// Test only: run one permit unit between the recount and the un-hold,
    /// as a write would.
    write_after_recount: bool,
    /// How the plan's lineage stage held memory, kept from `plan` for the
    /// execution report's `planner_memory`.
    planner_memory: Mutex<Option<MemoryReport>>,
}

impl RestoreBody {
    /// The body the binary runs.
    pub fn new() -> Self {
        Self::default()
    }

    /// The body with `observer` called once its plan's coverage check is
    /// finalised, after the writes and before the recount: the tests' seam
    /// onto the plan.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_plan_observer(mut self, observer: PlanObserver) -> Self {
        self.observers.plan = Some(observer);
        self
    }

    /// The body with `observer` called once the writer finished, with the
    /// plan and what it wrote (before the recount).
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_execution_observer(mut self, observer: ExecutionObserver) -> Self {
        self.observers.execution = Some(observer);
        self
    }

    /// The body with `observer` called once with what the finish reached
    /// ([`FinishRecord`]), on every exit the body returns after the coverage
    /// check was finalised (a revoked permit drops the body instead, and the
    /// observer is not called).
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_finish_observer(mut self, observer: FinishObserver) -> Self {
        self.observers.finish = Some(observer);
        self
    }

    /// The body with the run stopping at `point`, as a kill there would: it
    /// returns exit 1 with an INFO line and writes nothing more.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_kill_point_for_test(mut self, point: super::execute::KillPoint) -> Self {
        self.seams.kill_point = Some(point);
        self
    }

    /// The body running one no-op write unit through the permit after the
    /// recount and before the un-hold, as a write there would: the un-hold
    /// then refuses (exit 1, the marker left).
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_write_after_recount_for_test(mut self) -> Self {
        self.write_after_recount = true;
        self
    }

    /// The body with the writer's batches holding at most `commands`
    /// commands instead of `RESTORE_BATCH_COMMANDS`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_batch_commands_for_test(mut self, commands: u64) -> Self {
        self.seams.batch_commands = commands;
        self
    }

    /// The body with both stages that hold the restore budget `R` holding at
    /// most `bytes` instead (bytes mode only; serial mode has no budget): the
    /// planner's lineage stage, whose decodes are charged
    /// `fetched × 14 + 2 MiB`, and the writer's objects stage, whose layers
    /// are charged `fetched + 3 × state + 40 MiB` plus 4.59 × the state's
    /// largest inline value. In either stage a charge above the budget runs
    /// alone.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_restore_budget_for_test(mut self, bytes: u64) -> Self {
        self.seams.budget_override = Some(bytes);
        self
    }
}

#[async_trait]
impl JobBody for RestoreBody {
    type Plan = RestorePlan;

    /// Plan the restore, reading nothing before the quiet period's end
    /// ([`RestorePlanner::plan`]; with the test seam's budget, when set,
    /// capping the lineage stage). Keeps how the lineage stage held memory
    /// for the execution report.
    async fn plan(&self, run: &JobRun) -> Result<RestorePlan, JobExit> {
        let (plan, planner_memory) =
            RestorePlanner::plan_with(run, self.seams.budget_override()).await?;
        *self
            .planner_memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(planner_memory);
        Ok(plan)
    }

    /// The restore phase ([`restore_phase`]: write the plan, finalise the
    /// coverage check, log the plan's summary and the writer's, call the
    /// plan and execution observers), then the finish (module docs): log the
    /// outcome, recount, log the restore summary, and stay held (exit 3) or
    /// un-hold.
    async fn execute(&self, run: &JobRun, plan: RestorePlan) -> JobExit {
        let planner_memory = self
            .planner_memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let phase =
            match restore_phase(run, &plan, &self.seams, planner_memory, &self.observers).await {
                Ok(phase) => phase,
                Err(exit) => return exit,
            };
        let RestorePhase {
            execution,
            report,
            outcome,
        } = &phase;
        tracing::info!(
            mode = run.mode.name(),
            outcome = outcome.as_str(),
            "the restore Job finished writing; outcome: {}; it now recounts the references of \
             the large values",
            outcome.as_str()
        );
        let mut record = FinishRecord {
            outcome: *outcome,
            recount: None,
            floor: None,
            unhold: None,
            exit: JobExit::Failed,
        };
        let stays_held = || {
            tracing::error!(
                mode = run.mode.name(),
                held = plan.held.len(),
                uncovered = report.uncovered_count(),
                unknown = report.unknown_count(),
                "the restore Job restored everything it could and recounted the references of \
                 the large values, but {} layers or sessions cannot be restored, {} large values \
                 are in no store and {} could not be confirmed (each named above); the service \
                 stays held and the Job exits 3",
                plan.held.len(),
                report.uncovered_count(),
                report.unknown_count()
            );
        };
        let inputs = FinishInputs {
            plan: &plan,
            report,
            execution,
            stays_held: (*outcome == PlanOutcome::StaysHeld)
                .then_some(&stays_held as &(dyn Fn() + Send + Sync)),
            after_summary: &|| {},
            write_after_recount: self.write_after_recount,
        };
        record.exit = finish::finish_run(run, &self.seams, inputs, &mut record).await;
        if let Some(observer) = &self.observers.finish {
            observer(&record);
        }
        record.exit
    }
}
