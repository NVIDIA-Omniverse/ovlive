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

//! The accept-loss Job's body (`live-edit dr accept-loss <incident> --confirm
//! discard-lost-data`; SDD SR-10, PRD FR7): it restores everything
//! restorable first, so nothing a backup can restore is discarded — except
//! the layers of a lost session, which are discarded with it (PD-6) — then
//! discards what is proven lost, recounts and un-holds
//! ([`AcceptLossBody::execute`]):
//!
//! 1. **Plan** as restore does ([`RestorePlanner`]), but with a backup store
//!    only at or after `T_safe`, so every absence it discards on is
//!    authoritative (PD-3), and stepping down past an over-limit plain base
//!    instead of holding on it (PD-4); a layer with nothing older to step
//!    down to is only unmapped later, its over-limit backups kept. A
//!    contradiction or S7 still ends the run held, having written nothing.
//! 2. **Restore** ([`super::restore::restore_phase`]): the restore writer,
//!    which checks the plan first (`execute_restore`), the coverage check's
//!    finalise, and the plan's and the writer's summaries (AL2 follows).
//! 3. **The lost set** (`plan::lost::lost_set`), pure, from this run's plan
//!    and finalised report only. An item that could not be confirmed lost
//!    (a backup-store error, never an absence) makes the run discard
//!    nothing: it recounts, logs the summary and the refusal (AL3) and exits
//!    3 (PD-2) — unless it runs with the second `--confirm
//!    discard-unconfirmed-data` (option E), which discards each such item as
//!    the absence it stands in for, with a WARN naming it and its error. (A
//!    lost session's own items are not judged one by one: the session's
//!    discard covers them, whatever their reads answered.)
//! 4. **The discard check** (`plan::lost::check_discardable`), pure: a
//!    defect logs AL10 and exits 1 before any discard.
//! 5. **The discard plan** (AL4), logged before the first discard, so a kill
//!    leaves the intent on record.
//! 6. **The discards** ([`super::execute::discard`]: read every lost
//!    layer's header first, then register and wipe lost layers, rewrite
//!    the session backups of the over-limit ones without them, and unmap;
//!    discard lost sessions and their backups; remove values in place;
//!    rewrite session backups), every one a write unit before the recount
//!    (defer-65).
//! 7. **The finish** ([`finish::finish_run`], as restore's): the recount,
//!    the restore summary and accept-loss's (AL11), then the counter floor
//!    and the un-hold, exit 0, 3, 4 or 1. `finish::unhold` refuses a write
//!    since the recount, so no discard can follow it (ADD 9).
//!
//! A run killed anywhere before the un-hold leaves the service held, and a
//! re-run converges (`super::execute::discard`'s module docs).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::execute::discard::{discard, DiscardReport};
use super::execute::{ExecSeams, MemoryReport};
use super::finish::{self, FinishInputs, FinishRecord};
use super::plan::lost::{check_discardable, lost_set, LostSet, UnconfirmedPolicy};
use super::plan::model::{PlanOutcome, RestorePlan};
use super::plan::report::{
    render_accept_loss_start, render_discard_defect, render_discard_plan, render_discard_summary,
    render_unconfirmed_discard, render_unconfirmed_refusal, DiscardPlanCounts,
};
use super::plan::RestorePlanner;
use super::restore::{restore_phase, Observers, RestorePhase};
use super::run::{JobBody, JobRun};
use super::JobExit;
// The observer hooks are installed only by the test-support builders below.
#[cfg(any(test, feature = "test-support"))]
use super::{
    finish::FinishObserver,
    restore::{ExecutionObserver, PlanObserver},
};

/// A test's view of a lost set and what its discards did
/// ([`AcceptLossBody::with_discard_observer`]).
pub type DiscardObserver = Arc<dyn Fn(&LostSet, &DiscardReport) + Send + Sync>;

/// The accept-loss Job's body.
#[derive(Default)]
pub struct AcceptLossBody {
    policy: UnconfirmedPolicy,
    observers: Observers,
    discard_observer: Option<DiscardObserver>,
    seams: ExecSeams,
    /// Test only: run one permit unit between the recount and the un-hold,
    /// as a write would.
    write_after_recount: bool,
    /// How the plan's lineage stage held memory, kept from `plan` for the
    /// execution report's `planner_memory`.
    planner_memory: Mutex<Option<MemoryReport>>,
}

impl AcceptLossBody {
    /// The body the binary runs. `discard_unconfirmed`: the command line's
    /// second `--confirm discard-unconfirmed-data` (option E).
    pub fn new(discard_unconfirmed: bool) -> Self {
        Self {
            policy: if discard_unconfirmed {
                UnconfirmedPolicy::Discard
            } else {
                UnconfirmedPolicy::Refuse
            },
            ..Self::default()
        }
    }

    /// The body with `observer` called once its plan's coverage check is
    /// finalised, after the restore writer's writes and before any discard.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_plan_observer(mut self, observer: PlanObserver) -> Self {
        self.observers.plan = Some(observer);
        self
    }

    /// The body with `observer` called once the restore writer finished,
    /// with the plan and what it wrote.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_execution_observer(mut self, observer: ExecutionObserver) -> Self {
        self.observers.execution = Some(observer);
        self
    }

    /// The body with `observer` called once its discards finished, with the
    /// lost set and what the discards did (before the recount). Not called
    /// on a refusal, a defect, or a discard that failed.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_discard_observer(mut self, observer: DiscardObserver) -> Self {
        self.discard_observer = Some(observer);
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

    /// The body with `hook` run once, synchronously, after the discards'
    /// registration (D1) landed and before the header re-read that precedes
    /// the wipes (D2), with the objects D1 registered
    /// (`execute::AfterDiscardRegister`). Not run on a run that stops
    /// before D1, or at the `DiscardAfterRegister(None)` kill point.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_after_discard_register_for_test(
        mut self,
        hook: super::execute::AfterDiscardRegisterFn,
    ) -> Self {
        self.seams.after_discard_register = Some(super::execute::AfterDiscardRegister(hook));
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

    /// The body with the restore writer's batches and the value removal's
    /// holding at most `commands` commands instead of
    /// `RESTORE_BATCH_COMMANDS`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_batch_commands_for_test(mut self, commands: u64) -> Self {
        self.seams.batch_commands = commands;
        self
    }

    /// The body with the value removal's index steps asking for `rows`
    /// members, and its row reads taking at most `rows` rows at once,
    /// instead of `DR_VALUE_REMOVAL_PAGE_ROWS`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_value_removal_page_rows_for_test(mut self, rows: u64) -> Self {
        self.seams.value_removal_page_rows = rows;
        self
    }

    /// The body with every stage that holds the restore budget `R` holding
    /// at most `bytes` instead (bytes mode only): the planner's lineage
    /// stage, the writer's objects stage and the value removal's admission.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_restore_budget_for_test(mut self, bytes: u64) -> Self {
        self.seams.budget_override = Some(bytes);
        self
    }

    /// The body handing the un-hold script `value` as the lease value
    /// instead of its own, so the script answers `lease_lost`
    /// deterministically (a write to the lease key would race its refresh).
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_unhold_lease_value_for_test(mut self, value: &str) -> Self {
        self.seams.unhold_lease_value = Some(value.to_owned());
        self
    }

    /// The body whose un-hold's first attempt runs its script and then
    /// reports its reply lost, so its write step retries: the in-doubt path,
    /// whose second attempt answers `already_unheld`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_unhold_reply_lost_for_test(mut self) -> Self {
        self.seams.unhold_reply_lost = true;
        self
    }

    /// The finish (module docs, step 7) after the restore phase, staying
    /// held with `stays_held` when it is `Some`, and logging `after_summary`
    /// after the restore summary.
    async fn finish(
        &self,
        run: &JobRun,
        plan: &RestorePlan,
        phase: &RestorePhase,
        stays_held: Option<&(dyn Fn() + Send + Sync)>,
        after_summary: &(dyn Fn() + Send + Sync),
        mut record: FinishRecord,
    ) -> JobExit {
        let inputs = FinishInputs {
            plan,
            report: &phase.report,
            execution: &phase.execution,
            stays_held,
            after_summary,
            write_after_recount: self.write_after_recount,
        };
        record.exit = finish::finish_run(run, &self.seams, inputs, &mut record).await;
        self.observe_finish(&record);
        record.exit
    }

    fn observe_finish(&self, record: &FinishRecord) {
        if let Some(observer) = &self.observers.finish {
            observer(record);
        }
    }
}

#[async_trait]
impl JobBody for AcceptLossBody {
    type Plan = RestorePlan;

    /// Plan as restore does (module docs, step 1): `RestorePlanner` reads
    /// the run's mode, so it waits for `T_safe` with a backup store and steps
    /// down past an over-limit plain base.
    async fn plan(&self, run: &JobRun) -> Result<RestorePlan, JobExit> {
        let (plan, planner_memory) =
            RestorePlanner::plan_with(run, self.seams.budget_override()).await?;
        *self
            .planner_memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(planner_memory);
        Ok(plan)
    }

    /// Steps 2 to 7 of the module docs.
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
        let report = &phase.report;
        tracing::info!(
            mode = run.mode.name(),
            held = plan.held.len(),
            uncovered = report.uncovered_count(),
            unknown = report.unknown_count(),
            "{}",
            render_accept_loss_start(
                plan.held.len(),
                report.uncovered_count(),
                report.unknown_count()
            )
        );
        let mut record = FinishRecord {
            outcome: phase.outcome,
            recount: None,
            floor: None,
            unhold: None,
            exit: JobExit::Failed,
        };

        // 3. The lost set, and PD-2's all-or-nothing gate.
        let lost = lost_set(&plan, report, self.policy);
        if lost.refuses() {
            let unconfirmed = lost.unconfirmed.len();
            let refusal = move || {
                tracing::error!(unconfirmed, "{}", render_unconfirmed_refusal(unconfirmed));
            };
            record.outcome = PlanOutcome::StaysHeld;
            return self
                .finish(run, &plan, &phase, Some(&refusal), &|| {}, record)
                .await;
        }

        // 4. The discard check, before any discard.
        let defects = check_discardable(&plan, &lost);
        if let Some(defect) = defects.first() {
            tracing::error!(
                defects = defects.len(),
                defect = %defect,
                "{}",
                render_discard_defect(defect)
            );
            record.exit = JobExit::Failed;
            self.observe_finish(&record);
            return record.exit;
        }

        // 5. The discard plan, and each unconfirmed item discarded (option E).
        for item in &lost.unconfirmed {
            tracing::warn!(
                item = %item.item,
                error = %item.error,
                "{}",
                render_unconfirmed_discard(item)
            );
        }
        let counts = DiscardPlanCounts::of(&lost);
        tracing::info!(
            layers = counts.layers,
            over_limit = counts.over_limit,
            sessions = counts.sessions,
            values = counts.values,
            value_layers = counts.value_layers,
            rewrites = counts.rewrites,
            registered = counts.registered,
            unconfirmed = counts.unconfirmed,
            unconfirmed_kept = counts.unconfirmed_kept,
            "{}",
            render_discard_plan(&counts, self.policy)
        );

        // 6. The discards, every one before the recount.
        let discards = match discard(run, &lost, &self.seams).await {
            Ok(discards) => discards,
            Err(exit) => {
                record.exit = exit;
                self.observe_finish(&record);
                return exit;
            }
        };
        if let Some(observer) = &self.discard_observer {
            observer(&lost, &discards);
        }

        // 7. The finish. A held item accept-loss discards nothing for
        // (unreachable: PD-4 steps down past an over-limit base) keeps the
        // run held after its discards.
        let policy = self.policy;
        let summary = || {
            tracing::info!(
                layers = discards.layers_discarded.len(),
                over_limit = discards.over_limit_unmapped.len(),
                sessions = discards.sessions_discarded.len(),
                values = discards.value_count(),
                value_layers = discards.values.len(),
                left = discards.left.len(),
                registered = discards.registered,
                session_backups = discards.session_backups_deleted,
                rewrites = discards.rewritten.len(),
                unconfirmed = discards.unconfirmed,
                unconfirmed_kept = discards.unconfirmed_kept,
                "{}",
                render_discard_summary(&discards, policy)
            );
        };
        let undiscardable = lost.undiscardable.len();
        let stays_held = move || {
            tracing::error!(
                mode = "accept-loss",
                held = undiscardable,
                "the accept-loss Job restored everything it could and recounted the references \
                 of the large values, but {undiscardable} layers or sessions cannot be restored, \
                 0 large values are in no store and 0 could not be confirmed (each named above); \
                 the service stays held and the Job exits 3"
            );
        };
        record.outcome = if undiscardable == 0 {
            PlanOutcome::Writable
        } else {
            PlanOutcome::StaysHeld
        };
        let stays_held: Option<&(dyn Fn() + Send + Sync)> =
            (undiscardable > 0).then_some(&stays_held as &(dyn Fn() + Send + Sync));
        self.finish(run, &plan, &phase, stays_held, &summary, record)
            .await
    }
}

#[cfg(test)]
mod tests {
    //! The body on plans built by hand (no planner run produces either),
    //! against a real store with the write permit open: P55-c, a plan its
    //! writer's own check refuses; and the body's discard check (AL10), a
    //! plan whose lost set removes a value from an object the plan does not
    //! write. The body end to end is the accept-loss integration tests'
    //! (`tests/integration/dr/accept_loss_test.rs`).

    use std::collections::BTreeMap;

    use super::super::execute::test_harness::{open_run, LogCapture};
    use super::super::plan::model::{
        BandInputs, CoverageObject, CoveragePlan, CoverageSource, HashVerdict, ObjectAction,
        ObjectPlan, PlanEpoch,
    };
    use super::super::JobMode;
    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::storage::test_support::TestValkey;

    /// A plan whose one object is written twice: two flag clears of object
    /// 7, which `check_executable` refuses.
    fn inconsistent_plan() -> RestorePlan {
        let clear = || ObjectPlan {
            object_id: 7,
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            action: ObjectAction::ClearFlags {
                band: BandInputs {
                    floor_seq: 10,
                    observed_commit_span: 0,
                    written_at_ms: 0,
                    cursor: None,
                },
            },
        };
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(1),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: Vec::new(),
            objects: vec![clear(), clear()],
            coverage: CoveragePlan::default(),
            held: Vec::new(),
            advisories: Vec::new(),
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: BTreeMap::new(),
        }
    }

    /// P55-c (fail-on-old). The accept-loss body on an inconsistent plan —
    /// under the second confirm too — exits 1 before any write: one ERROR
    /// naming the defect, no write unit begun, no lost set taken (no AL2)
    /// and no discard. With the writer's check removed, write units begin.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn p55c_accept_loss_on_an_inconsistent_plan_exits_1_before_any_write() {
        let tv = TestValkey::setup().await;
        let mut run = open_run(&tv, Arc::new(InMemoryCheckpointStore::new())).await;
        run.mode = JobMode::AcceptLoss;
        let discarded = Arc::new(Mutex::new(0usize));
        let sink = Arc::clone(&discarded);
        let body = AcceptLossBody::new(true).with_discard_observer(Arc::new(move |_, _| {
            *sink.lock().unwrap() += 1;
        }));
        let capture = LogCapture::default();
        let guard = capture.install();

        let exit = body.execute(&run, inconsistent_plan()).await;
        drop(guard);

        assert_eq!(exit, JobExit::Failed);
        let errors = capture.messages("ERROR");
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].starts_with("the restore plan is inconsistent: "),
            "{errors:?}"
        );
        assert_eq!(run.permit.units_begun(), 0, "no write unit began");
        assert!(
            !capture
                .messages("INFO")
                .iter()
                .any(|m| m.starts_with("the accept-loss Job restored everything it could")),
            "no lost set was taken"
        );
        assert_eq!(*discarded.lock().unwrap(), 0, "no discard ran");
    }

    /// A plan the writer's own check accepts and that writes nothing, whose
    /// coverage check judges a value of object 9 uncovered while the plan
    /// has no action for that object: its lost set removes the value from
    /// an object the writer neither restored, kept nor cleared (rule 3).
    fn plan_removing_a_value_from_an_unwritten_object() -> RestorePlan {
        const HASH: [u8; 32] = [0x93; 32];
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(1),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: Vec::new(),
            objects: Vec::new(),
            coverage: CoveragePlan {
                verdicts: BTreeMap::from([(HASH, HashVerdict::Uncovered)]),
                references: BTreeMap::from([(HASH, std::collections::BTreeSet::from([9]))]),
                objects: BTreeMap::from([(
                    9,
                    CoverageObject {
                        session_id: "s".to_owned(),
                        layer_name: "l.usda".to_owned(),
                        source: CoverageSource::Survivor,
                    },
                )]),
            },
            held: Vec::new(),
            advisories: Vec::new(),
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: BTreeMap::new(),
        }
    }

    /// Test review 7 (i) (fail-on-old). The body's discard check: a lost set
    /// that removes a value from an object the plan does not write (rule
    /// 3) makes the body log AL10 naming the defect and exit 1 before any
    /// discard — no write unit begun, no discard run, the finish not
    /// reached. With the check skipped, the discards begin.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_discard_check_refuses_a_value_removal_from_an_unwritten_object_with_al10() {
        let tv = TestValkey::setup().await;
        let mut run = open_run(&tv, Arc::new(InMemoryCheckpointStore::new())).await;
        run.mode = JobMode::AcceptLoss;
        let discarded = Arc::new(Mutex::new(0usize));
        let sink = Arc::clone(&discarded);
        let finished: Arc<Mutex<Vec<FinishRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let finish_sink = Arc::clone(&finished);
        let body = AcceptLossBody::new(false)
            .with_discard_observer(Arc::new(move |_, _| {
                *sink.lock().unwrap() += 1;
            }))
            .with_finish_observer(Arc::new(move |record| {
                finish_sink.lock().unwrap().push(record.clone());
            }));
        let capture = LogCapture::default();
        let guard = capture.install();

        let exit = body
            .execute(&run, plan_removing_a_value_from_an_unwritten_object())
            .await;
        drop(guard);

        assert_eq!(exit, JobExit::Failed);
        let al10 = render_discard_defect(
            &crate::dr::job::plan::lost::DiscardDefect::ValueRemovalNotWritten { object_id: 9 },
        );
        let errors = capture.messages("ERROR");
        assert_eq!(
            errors.iter().filter(|m| **m == al10).count(),
            1,
            "one AL10: {errors:?}"
        );
        assert_eq!(run.permit.units_begun(), 0, "no write unit began");
        assert_eq!(*discarded.lock().unwrap(), 0, "no discard ran");
        let finished = finished.lock().unwrap();
        assert_eq!(finished.len(), 1, "the finish observer saw the exit");
        assert_eq!(finished[0].recount, None, "the finish was not reached");
    }
}
