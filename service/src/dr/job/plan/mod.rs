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

//! The restore plan (SDD SR-9.2 to SR-9.6 steps 1–2): before any write, the
//! restore and accept-loss Jobs list the backup store once, plan every
//! session and layer, and check that every external value a restored or
//! surviving layer references is somewhere. It writes nothing; the restore
//! writer (`dr::job::execute`) executes the plan, and accept-loss's lost set
//! ([`lost`]) reads it. Design and rationale: the Jobs section of SDD §22.
//!
//! # Order ([`RestorePlanner::plan`])
//!
//! 1. **Wait for the quiet period's end** on the store's clock (defer-39):
//!    nothing is read before it, so no late write the quiet period bounds is
//!    missed, and nothing needs a re-check after it. Accept-loss with a
//!    persistent backup store waits for `T_safe` instead (PD-3), so every
//!    absence it reads, and may discard on, is authoritative; with no backup
//!    store it waits for the quiet period's end, as restore does.
//! 2. **Read the store once** ([`snapshot`]): every surviving object header
//!    and layer mapping (one keyspace walk each), the dead-object set, and
//!    each named session's record, deletion marker and publish watermark.
//!    After the quiet period only the Job writes the store, so none of it is
//!    read again.
//! 3. **List the backup store once** ([`snapshot::take_listing`]): its layer
//!    artifacts, grouped per object, and its session backups, each read.
//!    With no backup store (`Disabled`, `InMemory`), nothing is listed or
//!    read from it at all.
//! 4. **Plan** ([`decide`]): the session rules, the mapped set, each mapped
//!    object's decision; each object that needs a backup has its lineage
//!    selected and decoded ([`lineage`]) and judged by the Job's absence
//!    policy, up to `LIVE_EDIT_DR_RESTORE_PARALLELISM` objects at once, each
//!    decode admitted by the Job's restore memory
//!    (`dr::job::execute::memory`); then the post-lineage resolution.
//! 5. **`T_safe`, once.** A write-shaping absence seen before `T_safe` — an
//!    all-empty listing, a mapped object with no listed backup, a listed
//!    generation read as absent, a listed session backup read as absent, a
//!    contradiction, a session named only by evidence under an empty
//!    session listing, or, under an empty session listing, a listed object
//!    whose identity read saw a generation read as absent and learned no
//!    session — makes the planner wait for `T_safe` (logging the
//!    wait), list again, and plan again over the fresh listing. Objects
//!    whose listing did not change keep their first verdict. At or after
//!    `T_safe` a contradiction, or a session named only by evidence under an
//!    empty session listing (S7), ends the run held (exit 3), having written
//!    nothing.
//! 6. **Coverage** ([`coverage`]): Valkey, the restored tails, this run's
//!    own PUTs, then one probe each against the backup store. A value
//!    absent before `T_safe` is checked again when the body finalises the
//!    coverage check ([`model::CoveragePlan::finalize`]), after the writes
//!    (`dr::job::execute`), so a value the writer stored in the backup store
//!    itself is covered by that PUT.
//!
//! # Layout
//!
//! - [`model`] — the plan's types, the contract the writer, the recount and
//!   accept-loss consume;
//! - [`decide`] — the pure core: no I/O, unit-tested without a store; also
//!   the writer's consistency check ([`decide::check_executable`]);
//! - [`lost`] — accept-loss's lost set and its discard check, pure;
//! - [`report`] — the coverage report's value sites and the plan's log
//!   lines;
//! - [`snapshot`] — the store and backup-store reads;
//! - [`lineage`] — the Job's lineage selection;
//! - [`coverage`] — the probe stage and [`model::CoveragePlan::finalize`].

pub mod coverage;
pub mod decide;
pub mod lineage;
pub mod lost;
pub mod model;
pub mod report;
pub mod snapshot;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::Duration;

use futures::{StreamExt, TryStreamExt};

use self::decide::{
    coverage_universe, plan_pre_lineage, resolve, session_universe, sessions_seen,
    survivors_to_cover, Listing, PlanInputs, PrePlan, PreStep, Resolved, StoreAccess, StoreFacts,
    UnidentifiedObject,
};
use self::model::{
    CoverageObject, CoverageSource, ExtHash, LineageVerdict, ObjectAction, PlanEpoch, RestorePlan,
};
use crate::checkpoint::store::{CheckpointHandle, CheckpointKind, CheckpointStore};
use crate::dr::job::execute::memory::{RestoreGate, RestoreMemory};
use crate::dr::job::execute::MemoryReport;
use crate::dr::job::{BackupAccess, JobExit, JobMode, JobRun};

pub use self::model::{CoverageReport, PlanOutcome};

/// How many store reads the planner keeps in flight at once (an internal
/// constant).
pub const DR_PLAN_VALKEY_READ_CONCURRENCY: usize = 64;

/// How many backup-store existence checks the coverage check keeps in flight
/// at once (an internal constant).
pub const DR_COVERAGE_PROBE_CONCURRENCY: usize = 32;

/// How many existence checks in a row may fail before the coverage check
/// stops and the run exits 1 (PD-5; an internal constant).
pub const DR_COVERAGE_PROBE_BREAKER: usize = 32;

/// The restore planner: [`RestorePlanner::plan`].
pub struct RestorePlanner;

/// The lineage verdicts kept across the `T_safe` re-plan, by object: its
/// exact listed generations and its floor, and the verdict. A generation's
/// bytes are fully determined by `(object_id, seq, kind)` (SDD §22.16), so a
/// verdict over an unchanged listing stands.
type LineageMemo = BTreeMap<u64, (Vec<(u64, CheckpointKind)>, Option<u64>, LineageVerdict)>;

/// One object whose lineage the lineage stage selects: the object, its
/// listed generations' key for the memo, its handles and its floor.
type ToSelect = (
    u64,
    Vec<(u64, CheckpointKind)>,
    Vec<CheckpointHandle>,
    Option<u64>,
);

/// One selection's result: the object, its memo key, its floor, its verdict.
type Selected = (u64, Vec<(u64, CheckpointKind)>, Option<u64>, LineageVerdict);

/// What the planner keeps across its passes.
#[derive(Default)]
struct Memo {
    identities: BTreeMap<u64, (String, String)>,
    unidentified: BTreeMap<u64, UnidentifiedObject>,
    lineages: LineageMemo,
}

/// What bounds the lineage stage's selections: each selection's reserve on
/// its fetch, and the Job's restore memory, whose gate admits each decode.
struct SelectionBounds {
    reserve: Duration,
    memory: RestoreMemory,
    /// Replaces `R` in the gate in bytes mode: the test seam
    /// `RestoreBody::with_restore_budget_for_test`; always `None` in a
    /// release build.
    budget_override: Option<u64>,
}

/// One planning pass over one listing.
struct Pass {
    listing: Listing,
    taken_at_ms: Option<u64>,
    listed_max_object_id: Option<u64>,
    pre: PrePlan,
    verdicts: BTreeMap<u64, LineageVerdict>,
    resolved: Resolved,
    /// How the pass's lineage stage held memory.
    lineage_memory: MemoryReport,
}

impl RestorePlanner {
    /// Plan the restore (module docs, *Order*). Reads only. `Err` ends the
    /// run, having logged why: `StaysHeld` (exit 3) for a contradiction, S7
    /// at or after `T_safe`, or a keyspace walk whose `CLUSTER NODES` gave
    /// no usable myself line; `Failed` (exit 1) for a store or backup-store
    /// read error, a keyspace walk whose slot map changed or whose node pin
    /// missed its primary's slots, an agreeing walk's two walks that
    /// returned different keys, a failed join, or the probe breaker.
    pub async fn plan(run: &JobRun) -> Result<RestorePlan, JobExit> {
        Self::plan_with(run, None).await.map(|(plan, _)| plan)
    }

    /// [`Self::plan`] with the lineage stage's gate holding at most
    /// `budget_override` bytes instead of the restore budget `R` (bytes
    /// mode only), and with how that stage held memory over every pass (the
    /// most at once in any pass). The restore body passes its test seam's
    /// budget (`RestoreBody::with_restore_budget_for_test`), which is `None`
    /// in a release build, and reports the memory as the execution report's
    /// `planner_memory`.
    pub(crate) async fn plan_with(
        run: &JobRun,
        budget_override: Option<u64>,
    ) -> Result<(RestorePlan, MemoryReport), JobExit> {
        let mode = run.mode;
        run.metrics.init_dr_restore_coverage();
        wait_for_quiet_end(run).await;
        // PD-3: accept-loss with a backup store reads it only at or after
        // `T_safe`, so every absence it acts on is authoritative, the
        // "session with no listed backup" under a non-empty listing included.
        if mode == JobMode::AcceptLoss && matches!(run.backup, BackupAccess::Store(_)) {
            if let Err(e) = run
                .clock
                .wait_until_t_safe(
                    "the accept-loss Job reads the backup store only once \
                     checkpoint.notFoundCacheSecs has passed after the quiet period, so every \
                     absence it discards on is authoritative",
                )
                .await
            {
                tracing::error!(
                    error = %e,
                    "the {mode} Job could not read the store's clock to plan the restore, wrote \
                     nothing and exits 1"
                );
                return Err(JobExit::Failed);
            }
        }
        tracing::info!(
            mode = mode.name(),
            "the {mode} Job is reading the store to plan the restore; it reads nothing before \
             the quiet period ends, so no late write the quiet period covers is missed"
        );

        let mut facts = snapshot::read_store_facts(run).await?;
        let access = match &run.backup {
            BackupAccess::Store(_) => StoreAccess::Persistent,
            BackupAccess::None(why) => StoreAccess::NoStore(*why),
        };
        let reserve = lineage::selection_reserve(run.config.checkpoint_adapter_call_ceiling());
        // The Job's restore memory bounds the lineage stage's decodes as it
        // bounds the writer (`execute::memory`). `run_job` derived it once
        // already and refuses a limit below the minimum, so an error here is
        // a defect.
        let memory = RestoreMemory::derive(&run.config, mode).map_err(|refusal| {
            tracing::error!(
                refusal = %refusal,
                "the {mode} Job could not derive its restore memory to plan the restore, wrote \
                 nothing and exits 1"
            );
            JobExit::Failed
        })?;
        let bounds = SelectionBounds {
            reserve,
            memory,
            budget_override,
        };
        let mut memo = Memo::default();

        let mut pass = run_pass(run, access, &mut facts, &mut memo, &bounds, None).await?;
        let mut lineage_memory = pass.lineage_memory.clone();
        let mut relisted = false;
        if !pass.resolved.triggers.is_empty() && !pass.listing.at_or_after_t_safe {
            let first = &pass.resolved.triggers[0];
            tracing::info!(
                absences = pass.resolved.triggers.len(),
                "the restore plan saw {} write-shaping absences before T_safe; it waits for \
                 T_safe, lists the backup store again and plans again",
                pass.resolved.triggers.len()
            );
            if let Err(e) = run.clock.wait_until_t_safe(&first.why()).await {
                tracing::error!(
                    error = %e,
                    "the {mode} Job could not read the store's clock to plan the restore, wrote \
                     nothing and exits 1"
                );
                return Err(JobExit::Failed);
            }
            let previous = pass.listing;
            pass = run_pass(run, access, &mut facts, &mut memo, &bounds, Some(&previous)).await?;
            lineage_memory.max_in_flight = lineage_memory
                .max_in_flight
                .max(pass.lineage_memory.max_in_flight);
            lineage_memory.peak_reserved_bytes = lineage_memory
                .peak_reserved_bytes
                .max(pass.lineage_memory.peak_reserved_bytes);
            relisted = true;
        }

        if let Some(hit) = pass.pre.s7.first() {
            tracing::error!(
                session_id = %hit.session_id,
                evidence = %hit.evidence,
                "the {mode} Job wrote nothing and exits 3; the service stays held: the backup \
                 store lists no session backups, but {} names session {}, which has no record \
                 in the store and no deletion marker; the backup store's identity may lack \
                 permission to list .live-edit/sessions/ under the backup prefix",
                hit.evidence,
                hit.session_id
            );
            return Err(JobExit::StaysHeld);
        }
        if let Some(contradiction) = pass.pre.contradictions.first() {
            tracing::error!(
                contradictions = pass.pre.contradictions.len(),
                "the {mode} Job wrote nothing and exits 3; the service stays held: the backup \
                 store's listing contradicts the store: {contradiction}; check the backup \
                 store's address, credentials and listing permissions"
            );
            return Err(JobExit::StaysHeld);
        }

        for (object_id, session_id, layer_name, base_seq) in &pass.resolved.over_limit {
            report::log_over_limit(*object_id, session_id, layer_name, *base_seq);
        }

        // Coverage (SR-9.3).
        let survivors =
            snapshot::read_survivor_ext(run, &survivors_to_cover(&pass.resolved.objects)).await?;
        let mut restored: BTreeMap<u64, BTreeSet<ExtHash>> = BTreeMap::new();
        let mut tails: HashSet<ExtHash> = HashSet::new();
        let mut covered: BTreeMap<u64, CoverageObject> = BTreeMap::new();
        for object in &pass.resolved.objects {
            match &object.action {
                ObjectAction::Restore { lineage, .. } => {
                    if let Some(LineageVerdict::Use(selected)) =
                        pass.verdicts.get(&object.object_id)
                    {
                        restored.insert(object.object_id, selected.state_hashes.clone());
                    }
                    tails.extend(lineage.tail_carried.iter().copied());
                    covered.insert(
                        object.object_id,
                        CoverageObject {
                            session_id: object.session_id.clone(),
                            layer_name: object.layer_name.clone(),
                            source: CoverageSource::Restored(lineage.clone()),
                        },
                    );
                }
                ObjectAction::Keep | ObjectAction::ClearFlags { .. } => {
                    covered.insert(
                        object.object_id,
                        CoverageObject {
                            session_id: object.session_id.clone(),
                            layer_name: object.layer_name.clone(),
                            source: CoverageSource::Survivor,
                        },
                    );
                }
                ObjectAction::Remap { .. } | ObjectAction::Nothing => {}
            }
        }
        let universe = coverage_universe(&pass.resolved.objects, &restored, &survivors);
        let coverage = coverage::check_coverage(run, universe, &tails, covered).await?;
        run.metrics
            .set_dr_restore_values_uncovered(coverage.uncovered_count() as u64);

        let plan = RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: pass.taken_at_ms,
                at_or_after_t_safe: pass.listing.at_or_after_t_safe,
                relisted_at_t_safe: relisted,
            },
            sessions: pass.pre.sessions.clone(),
            objects: pass.resolved.objects,
            coverage,
            held: pass.resolved.held,
            advisories: pass.resolved.advisories,
            listed_max_object_id: pass.listed_max_object_id,
            surviving_max_object_id: facts.headers.keys().next_back().copied(),
            sessions_seen: sessions_seen(&facts),
        };
        report::log_plan_advisories(&plan);
        report::log_held(&plan);
        Ok((plan, lineage_memory))
    }
}

/// Wait until the store's clock reaches the quiet period's end. A failed read
/// of the clock is retried at the frame's wait-log cadence. Logs nothing: the
/// frame logs the quiet-period wait itself.
async fn wait_for_quiet_end(run: &JobRun) {
    let target = run.deadlines().quiet_ends_at_ms;
    let interval = run.clock.log_interval();
    loop {
        let pause = match run.clock.now_ms().await {
            Ok(now) if now >= target => return,
            Ok(now) => Duration::from_millis(target - now).min(interval),
            Err(_) => interval,
        };
        tokio::time::sleep(pause).await;
    }
}

/// One planning pass: list the backup store (with one), read what the
/// listing names, plan, select each needed lineage (or keep its verdict from
/// an earlier pass over the same generations), and resolve.
async fn run_pass(
    run: &JobRun,
    access: StoreAccess,
    facts: &mut StoreFacts,
    memo: &mut Memo,
    bounds: &SelectionBounds,
    previous: Option<&Listing>,
) -> Result<Pass, JobExit> {
    let store: Option<&dyn CheckpointStore> = match &run.backup {
        BackupAccess::Store(store) => Some(store.as_ref()),
        BackupAccess::None(_) => None,
    };
    let (mut listing, taken_at_ms, listed_max_object_id) = match store {
        Some(store) => {
            let taken = snapshot::take_listing(run, store, previous).await?;
            (
                taken.listing,
                Some(taken.taken_at_ms),
                taken.listed_max_object_id,
            )
        }
        // Nothing is listed, and no absence is read from a backup store, so
        // nothing waits for `T_safe`.
        None => (
            Listing {
                at_or_after_t_safe: true,
                ..Listing::default()
            },
            None,
            None,
        ),
    };
    if let Some(store) = store {
        listing.identity_absences = snapshot::identify_listed_objects(
            run,
            store,
            &listing,
            facts,
            &mut memo.identities,
            &mut memo.unidentified,
        )
        .await?;
    }
    let universe = session_universe(&listing, facts, &memo.identities);
    snapshot::read_missing_session_facts(run, facts, &universe).await?;

    let pre = plan_pre_lineage(&PlanInputs {
        access,
        listing: &listing,
        store: facts,
        identities: &memo.identities,
        unidentified: &memo.unidentified,
    });

    // The lineage stage: every object that needs a lineage and has no
    // verdict kept from an earlier pass over the same generations, at most
    // `LIVE_EDIT_DR_RESTORE_PARALLELISM` at once (1 in serial mode), each
    // decode admitted by the Job's restore memory (`lineage::select_within`).
    // The memo and the verdicts are updated after the stream; an `Err` ends
    // the stream, dropping the selections in flight.
    let mut verdicts: BTreeMap<u64, LineageVerdict> = BTreeMap::new();
    let mut to_select: Vec<ToSelect> = Vec::new();
    if store.is_some() {
        for object in &pre.objects {
            let PreStep::NeedsLineage(need) = &object.step else {
                continue;
            };
            let key: Vec<(u64, CheckpointKind)> =
                need.handles.iter().map(|h| (h.seq(), h.kind)).collect();
            if let Some((listed, floor, verdict)) = memo.lineages.get(&object.object_id) {
                if *listed == key
                    && *floor == need.floor
                    && !matches!(verdict, LineageVerdict::Defer { .. })
                {
                    verdicts.insert(object.object_id, verdict.clone());
                    continue;
                }
            }
            to_select.push((object.object_id, key, need.handles.clone(), need.floor));
        }
    }
    let mut lineage_memory = MemoryReport::idle(&bounds.memory);
    if let (Some(store), false) = (store, to_select.is_empty()) {
        let gate = RestoreGate::new(&bounds.memory, bounds.budget_override);
        let selected: Vec<Selected> = futures::stream::iter(to_select)
            .map(|(object_id, key, handles, floor)| {
                let gate = &gate;
                async move {
                    let at_or_after_t_safe = run.clock.at_or_after_t_safe().await.map_err(|e| {
                        tracing::error!(
                            error = %e,
                            "the {mode} Job could not read the store's clock to plan the \
                             restore, wrote nothing and exits 1",
                            mode = run.mode
                        );
                        JobExit::Failed
                    })?;
                    let verdict = lineage::select_within(
                        store,
                        object_id,
                        handles,
                        floor,
                        at_or_after_t_safe,
                        bounds.reserve,
                        Some(gate),
                        run.mode,
                    )
                    .await?;
                    Ok::<_, JobExit>((object_id, key, floor, verdict))
                }
            })
            .buffer_unordered(gate.parallelism())
            .try_collect()
            .await?;
        lineage_memory.record(&gate);
        for (object_id, key, floor, verdict) in selected {
            memo.lineages
                .insert(object_id, (key, floor, verdict.clone()));
            verdicts.insert(object_id, verdict);
        }
    }
    let resolved = resolve(&pre, &verdicts);
    Ok(Pass {
        listing,
        taken_at_ms,
        listed_max_object_id,
        pre,
        verdicts,
        resolved,
        lineage_memory,
    })
}

#[cfg(test)]
mod tests {
    /// U7: the pure core — the planning rules, the report's rendering, the
    /// plan's types and accept-loss's lost set — names no runtime, no store client and no
    /// backup-store trait, so it stays unit-testable without a store. The
    /// scan covers the whole files, their tests and comments included, and
    /// lives here because the words it looks for would trip it anywhere in
    /// the files it scans.
    #[test]
    fn the_pure_core_names_no_runtime_store_client_or_backup_store() {
        const FORBIDDEN: [&str; 5] = [
            "async",
            ".await",
            "ValkeyClient",
            "CheckpointStore",
            "tokio::",
        ];
        for (file, source) in [
            ("decide.rs", include_str!("decide.rs")),
            ("report.rs", include_str!("report.rs")),
            ("model.rs", include_str!("model.rs")),
            ("lost.rs", include_str!("lost.rs")),
        ] {
            for word in FORBIDDEN {
                let lines: Vec<usize> = source
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains(word))
                    .map(|(i, _)| i + 1)
                    .collect();
                assert!(
                    lines.is_empty(),
                    "src/dr/job/plan/{file} names {word:?} on lines {lines:?}; the pure core \
                     must stay free of I/O and waiting"
                );
            }
        }
    }
}
