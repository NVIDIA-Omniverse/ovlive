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

//! The restore Job's memory end to end (`live_edit::dr::job::execute::memory`;
//! SDD §22.22 *Restore memory*, SR-12): how many layers its objects stage
//! holds at once, read from the writer's report, and how many lineages it
//! fetches at once as it plans and as it writes, over a real store.
//!
//! W14. Every case restores identical layers, so each charges the same. The
//! budget is half the Job's pod memory limit, fixed here at 5 GiB with
//! `PodMemorySource::Fixed`; no realistic limit gives a budget near a test
//! layer's charge, so the cases about the budget's bound replace it with
//! the Job's test seam (`with_restore_budget_for_test`, which caps the
//! planner's lineage stage as well as the objects stage):
//!
//! - a budget holding about two and a half layers holds two at once;
//! - a budget smaller than one layer's charge runs each alone;
//! - the budget itself, far above every charge, holds as many as the
//!   parallelism allows: all three at the default of 8, two at 2;
//! - with no readable limit (cgroup unlimited) the Job restores one layer at
//!   a time and logs one serial-mode WARN;
//! - at a parallelism of 2 over four layers, the planner's lineage stage
//!   fetches exactly two lineages at once, never more, and so does the
//!   objects stage, each counted on its own fetches (defer-45 (3): the
//!   planner's stage runs under the same count as the writer's);
//! - the planner's lineage stage holds its decodes within the same budget
//!   (defer-45 (3)), read from the report's `planner_memory`: within the
//!   default budget, two decodes at once at a parallelism of 2, each
//!   charged `plan_decode_charge(fetched)`; within a budget that holds one
//!   such charge but not two, one at a time;
//! - a layer with a large inline value is admitted by the writer at
//!   `write_charge` plus `inline_value_charge` of the value the planner
//!   recorded, read from the report's peak reservation;
//! - a restore never runs the allocator purge it carries, nor logs the
//!   value removal's purge line (only accept-loss's value removal purges).
//!
//! The counts are deterministic: every layer the objects stage admits waits
//! on its re-fetch, its decode on the blocking pool and its Valkey writes
//! before it can finish, so the stage polls each later layer, and admits it
//! if it fits, while the earlier ones still hold their charges. The fetch
//! counts are made deterministic by pacing the backup store's fetches
//! (`WriterStore::pace_fetches`). The planner charges a decode only once
//! its lineage is fetched, and holds the charge while it decodes; its cases
//! release the fetches two at a time (`WriterStore::pace_fetches_in_groups`),
//! so both lineages of a pair reach the charge together, the second while
//! the first still decodes.
//!
//! Every run uses a per-test key prefix (`TestValkey`). Waits: each test
//! plans once to learn the layers' charge (bounded by `RUN_BUDGET`) and runs
//! the Job once (the parallelism test does both twice), and its whole body
//! runs under `writer_support`'s one-test budget (its module docs, *Time*).

use std::num::NonZeroUsize;
use std::time::Duration;

use live_edit::config::pod_memory::{PodMemoryLimit, ReferenceReason};
use live_edit::config::{Config, PlanInputs, PodMemoryPlan, PodMemorySource};
use live_edit::dr::job::execute::memory::{plan_decode_charge, write_charge};
use live_edit::dr::job::execute::{ExecutionReport, MemoryReport};
use live_edit::dr::job::plan::model::ObjectAction;
use live_edit::dr::job::RestorePlanner;
use live_edit::storage::test_support::TestValkey;

use super::writer_support::*;

const MIB: u64 = 1024 * 1024;

/// The Job's pod memory limit in the bytes-mode cases: 5 GiB, the chart's
/// default, so its budget is 2,560 MiB.
const LIMIT_MIB: u64 = 5_120;

/// How long a paced fetch waits before it answers anyway: for a third to be
/// in flight beside it (`pace_fetches`), or for the rest of its group of two
/// (`pace_fetches_in_groups`). The two a stage at a parallelism of 2 starts
/// at once reach the store within milliseconds of each other, so this only
/// has to outlast that. It is paid in full once per pair under
/// `pace_fetches` (four pairs, 8 s a run), and under
/// `pace_fetches_in_groups` only by a fetch with no partner, such as a
/// writer restoring one layer at a time (four fetches, 8 s a run).
const FETCH_PAIRING_WAIT: Duration = Duration::from_secs(2);

/// `config` with its pod memory plan derived at `limit`.
fn with_limit(mut config: Config, limit: PodMemoryLimit) -> Config {
    config.pod_memory = PodMemoryPlan::derive(limit, &PlanInputs::from_config(&config))
        .expect("the server's plan closes at this limit");
    config
}

/// The writer configuration at a fixed 5 GiB limit with `parallelism`
/// layers at once.
fn bytes_config(parallelism: usize) -> Config {
    let mut config = with_limit(
        writer_config(),
        PodMemorySource::Fixed(LIMIT_MIB * MIB).resolve(),
    );
    config.dr_restore_parallelism = NonZeroUsize::new(parallelism).expect("non-zero");
    config
}

/// What one of [`identical_layers`]' layers charges, in bytes.
struct Charges {
    /// The writer's ([`write_charge`]): the fetched bytes, three times the
    /// decoded state and one batch's fixed term. The writer also reserves
    /// the inline-value term for the layer's largest inline value, which is
    /// a few bytes for these fixtures and is not counted here.
    write: u64,
    /// The planner's, before its decode ([`plan_decode_charge`]): the
    /// fetched bytes times one plus the decode factor, plus its fixed term.
    plan: u64,
}

/// `layers` identical layers of session "sm", backed up, after a full
/// wipe. Returns each layer's charges, read from the plan the planner makes
/// of them (within `RUN_BUDGET`).
async fn identical_layers(
    tv: &TestValkey,
    store: &std::sync::Arc<WriterStore>,
    config: &Config,
    layers: u64,
) -> Charges {
    let names: Vec<(String, u64)> = (0..layers)
        .map(|i| (format!("m{i}.usda"), 1_500 + i))
        .collect();
    let mappings: Vec<(&str, u64)> = names.iter().map(|(l, o)| (l.as_str(), *o)).collect();
    put_session(store.as_ref(), &session_backup("sm", &mappings, &[])).await;
    let bytes = base(20, 0x99, &[]);
    for (layer, oid) in &mappings {
        put_base(store.as_ref(), *oid, "sm", layer, 5, bytes.clone()).await;
    }
    put_counters(store.as_ref()).await;
    let run = planning_run(tv, config, store).await;
    let plan = tokio::time::timeout(RUN_BUDGET, RestorePlanner::plan(&run))
        .await
        .expect("the plan ends within its budget")
        .expect("the plan");
    let charges: Vec<(u64, u64)> = plan
        .objects
        .iter()
        .map(|o| match &o.action {
            ObjectAction::Restore { lineage, .. } => (
                write_charge(
                    lineage.footprint.fetched_bytes,
                    lineage.footprint.state_bytes,
                ),
                plan_decode_charge(lineage.footprint.fetched_bytes),
            ),
            other => panic!("every layer is restored, got {other:?}"),
        })
        .collect();
    assert_eq!(charges.len() as u64, layers);
    assert!(
        charges
            .iter()
            .all(|c| *c == charges[0] && c.0 > 0 && c.1 > 0),
        "fixture: identical layers charge the same: {charges:?}"
    );
    Charges {
        write: charges[0].0,
        plan: charges[0].1,
    }
}

/// The objects stage's memory report, after every layer was restored.
fn memory_of(report: &ExecutionReport, layers: usize) -> &MemoryReport {
    assert_eq!(report.objects.len(), layers, "every layer was restored");
    &report.memory
}

/// A budget of two and a half layers' charges holds two at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_holds_as_many_layers_as_fit_its_budget() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = bytes_config(8);
        let charge = identical_layers(&tv, &store, &config, 4).await.write;
        hold(&tv, &config).await;
        let budget = charge * 5 / 2;

        let (_plan, report, _captured) =
            run_to_the_end_with(RUN_BUDGET, &tv, &config, &store, |b| {
                b.with_restore_budget_for_test(budget)
            })
            .await;

        let memory = memory_of(&report, 4);
        assert_eq!((memory.mode.as_str(), memory.parallelism), ("bytes", 8));
        assert_eq!(memory.max_in_flight, 2, "{memory:?}, charge {charge} B");
        assert!(
            memory.peak_reserved_bytes <= budget.next_multiple_of(1024),
            "{memory:?}"
        );
        assert!(
            memory.peak_reserved_bytes >= 2 * charge,
            "two layers' charges held at once: {memory:?}"
        );
    })
    .await;
}

/// A budget smaller than one layer's charge clamps each layer to the whole
/// budget: each runs alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_runs_a_layer_larger_than_its_whole_budget_alone() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = bytes_config(8);
        let charge = identical_layers(&tv, &store, &config, 3).await.write;
        hold(&tv, &config).await;
        let budget = (charge / 2).next_multiple_of(1024);

        let (_plan, report, _captured) =
            run_to_the_end_with(RUN_BUDGET, &tv, &config, &store, |b| {
                b.with_restore_budget_for_test(budget)
            })
            .await;

        let memory = memory_of(&report, 3);
        assert_eq!(memory.max_in_flight, 1, "{memory:?}, charge {charge} B");
        assert_eq!(
            memory.peak_reserved_bytes, budget,
            "clamped to the whole budget"
        );
    })
    .await;
}

/// The budget at 5 GiB, 2,560 MiB, holds every test layer: the parallelism
/// is the bound, all three at the default of 8, and two at 2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_holds_up_to_its_parallelism_within_a_budget_that_fits_every_layer() {
    within_test_budget(async {
        for (parallelism, want) in [(8, 3), (2, 2)] {
            let tv = TestValkey::setup().await;
            let store = WriterStore::new();
            let config = bytes_config(parallelism);
            let charge = identical_layers(&tv, &store, &config, 3).await.write;
            assert!(
                3 * charge < LIMIT_MIB / 2 * MIB,
                "fixture: the budget fits every layer"
            );
            hold(&tv, &config).await;

            let (_plan, report, captured) = run_to_the_end(&tv, &config, &store).await;

            let memory = memory_of(&report, 3);
            assert_eq!(
                (
                    memory.mode.as_str(),
                    memory.parallelism,
                    memory.max_in_flight
                ),
                ("bytes", parallelism, want),
                "parallelism {parallelism}: {memory:?}"
            );
            assert_eq!(
                lines_starting(&captured, "the restore Job could not read a memory limit"),
                0,
                "bytes mode logs no serial WARN"
            );
            assert_eq!(
                lines_starting(
                    &captured,
                    &format!(
                        "the restore Job writes the plan: 3 layers to restore, 0 flag clears, 0 \
                         moves, 1 sessions, 0 unfinished deletes; up to {parallelism} layers at \
                         once within 2560 MiB"
                    )
                ),
                1,
                "the start line names the mode; capture:\n{}",
                captured.text
            );
        }
    })
    .await;
}

/// The length of the one large inline value the layer of
/// [`restore_job_charges_the_writer_for_a_layers_largest_inline_value`]
/// carries: 1 MiB, under the restore batch's byte limit, so its writer term
/// (about 4.6 MiB) is far from a KiB's rounding.
const INLINE_VALUE_BYTES: usize = 1 << 20;

/// The writer charges a layer's largest inline value beside its
/// `write_charge`: the planner records the value's length in the lineage's
/// footprint (`largest_inline_bytes`), and the objects stage admits the
/// layer at `write_charge(fetched, state) + inline_value_charge(that
/// length)`, read from the report's peak reservation (rounded up to whole
/// KiB) with the one layer alone in a budget far above it.
///
/// Fails without the term in the writer's admission (in-place perturbation
/// of `execute::object::write_objects`): the peak is `write_charge` alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_charges_the_writer_for_a_layers_largest_inline_value() {
    use bytes::Bytes;
    use live_edit::domain::ompf::OmpfContainer;
    use live_edit::dr::job::execute::memory::inline_value_charge;

    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = bytes_config(8);
        put_session(
            store.as_ref(),
            &session_backup("si", &[("inline.usda", 1_600)], &[]),
        )
        .await;
        let mut delta = base_delta(4, 0x77, &[]);
        delta.set_fields.push(oodf_ffi::SetField {
            section_id: 2,
            key_name: "customData".to_owned(),
            value_or_ext_hash: vec![0x5A; INLINE_VALUE_BYTES],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1_000,
        });
        let delta_bytes = oodf_ffi::build_delta(&delta).expect("the base delta builds");
        let bytes =
            Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("the base serializes"));
        put_base(store.as_ref(), 1_600, "si", "inline.usda", 5, bytes).await;
        put_counters(store.as_ref()).await;
        let run = planning_run(&tv, &config, &store).await;
        let plan = tokio::time::timeout(RUN_BUDGET, RestorePlanner::plan(&run))
            .await
            .expect("the plan ends within its budget")
            .expect("the plan");
        let footprint = match &plan.objects[..] {
            [object] => match &object.action {
                ObjectAction::Restore { lineage, .. } => lineage.footprint,
                other => panic!("the layer is restored, got {other:?}"),
            },
            other => panic!("one layer, got {other:?}"),
        };
        assert_eq!(
            footprint.largest_inline_bytes, INLINE_VALUE_BYTES as u64,
            "the planner records the layer's largest inline value"
        );
        let charge = write_charge(footprint.fetched_bytes, footprint.state_bytes)
            + inline_value_charge(footprint.largest_inline_bytes);
        assert!(
            charge < LIMIT_MIB / 2 * MIB,
            "fixture: the budget holds the charge whole"
        );
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        let memory = memory_of(&report, 1);
        assert_eq!(
            memory.peak_reserved_bytes,
            charge.next_multiple_of(1024),
            "the writer held write_charge plus the inline value's term: {memory:?}"
        );
    })
    .await;
}

/// With no readable limit (the cgroup sets none) the Job restores one layer
/// at a time with no byte bound, and logs one WARN saying so.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_with_no_readable_limit_restores_one_layer_at_a_time_and_warns_once() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = with_limit(
            writer_config(),
            PodMemoryLimit::reference(Some(ReferenceReason::Unlimited)),
        );
        identical_layers(&tv, &store, &config, 3).await;
        hold(&tv, &config).await;

        let (_plan, report, captured) = run_to_the_end(&tv, &config, &store).await;

        let memory = memory_of(&report, 3);
        assert_eq!(
            (
                memory.mode.as_str(),
                memory.parallelism,
                memory.max_in_flight
            ),
            ("serial", 1, 1),
            "{memory:?}"
        );
        assert_eq!(memory.peak_reserved_bytes, 0, "no byte bound");
        let warns = lines(
            &captured,
            "the restore Job could not read a memory limit (unlimited); it restores one layer at \
             a time instead of 8, so it never holds two layers' state at once; give the Job a \
             memory limit (dr.resources) to restore in parallel",
        );
        assert_eq!(
            warns.len(),
            1,
            "one serial WARN; capture:\n{}",
            captured.text
        );
        assert_eq!(warns[0]["level"], "WARN");
    })
    .await;
}

/// A restore never purges the allocator (SDD §22.22 *Restore memory*: only
/// accept-loss's value removal does): run to its end over two layers with
/// a counting purge installed, as `main.rs` installs one on linux-gnu, it
/// restores both, calls the purge not once, and logs neither arm of the
/// value removal's purge line (`run::run_job` logs it for accept-loss
/// only). Every other restore fixture installs a purge that only answers
/// `true`, so none of them would notice. Fails with the line logged for
/// every mode (in-place perturbation of `run::run_job`, its accept-loss
/// check dropped), and with the restore writer purging (in-place
/// perturbation of `execute::execute`, a purge before its objects stage).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_never_runs_the_allocator_purge_nor_logs_its_line() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use live_edit::checkpoint::CheckpointStore;
    use live_edit::dr::job::execute::KillPoint;
    use live_edit::dr::job::{run_job, JobExit, JobMode};
    use live_edit::storage::allocator_purge::AllocatorPurge;

    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = bytes_config(8);
        identical_layers(&tv, &store, &config, 2).await;
        hold(&tv, &config).await;
        let calls = Arc::new(AtomicUsize::new(0));
        let metrics = metrics();
        let mut ctx = context(
            &tv,
            JobMode::Restore,
            config.clone(),
            Arc::clone(&store) as Arc<dyn CheckpointStore>,
            &metrics,
        );
        ctx.allocator_purge = Some(AllocatorPurge::new({
            let calls = Arc::clone(&calls);
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                true
            }
        }));
        let (body, observed) =
            observed_body(|b| b.with_kill_point_for_test(KillPoint::BeforeUnhold));

        let (exit, captured) = captured_within(RUN_BUDGET, run_job(ctx, body)).await;

        assert_eq!(exit, JobExit::Failed, "{}", captured.text);
        assert_eq!(
            lines_starting(&captured, KILL_LINE),
            1,
            "fixture: the writer finished and the run stopped before its un-hold; capture:\n{}",
            captured.text
        );
        let (_plan, report) = the_execution(&observed);
        assert_eq!(report.objects.len(), 2, "fixture: both layers restored");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "a restore never purges");
        let purge_lines: Vec<&str> = captured
            .events
            .iter()
            .filter_map(|e| e["fields"]["message"].as_str())
            .filter(|m| {
                m.contains("value removal purges the allocator")
                    || m.contains("runs with no allocator purge installed")
            })
            .collect();
        assert!(
            purge_lines.is_empty(),
            "a restore logs no purge line: {purge_lines:?}"
        );
    })
    .await;
}

/// defer-45 (3). At `LIVE_EDIT_DR_RESTORE_PARALLELISM` = 2 over four
/// layers, the planner's lineage stage fetches exactly two lineages at
/// once, never more, and so does the writer's objects stage: the planner's
/// stage is bounded by the same count as the writer's. The run's first four
/// fetches are the planner's (the plan is made before any write) and its
/// last four the writer's, so each stage is counted on its own fetches.
/// Each backup-store fetch is held until a third is in flight beside it or
/// [`FETCH_PAIRING_WAIT`] passed, so a stage running more than two at once
/// reaches three, and one running a single fetch at a time never reaches
/// two.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_fetches_at_most_its_parallelism_of_lineages_at_once() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let config = bytes_config(2);
        identical_layers(&tv, &store, &config, 4).await;
        hold(&tv, &config).await;
        store.pace_fetches(3, FETCH_PAIRING_WAIT);

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.objects.len(), 4, "every layer was restored");
        assert_eq!(
            store.gets.lock().unwrap().len(),
            12,
            "fixture: four fetches to learn the charges, four by the run's planner, four by its \
             writer"
        );
        let arrivals = store.fetch_arrivals();
        assert_eq!(arrivals.len(), 8, "fixture: the run fetched eight times");
        let (planner, writer) = arrivals.split_at(4);
        for (stage, fetches) in [("the planner", planner), ("the writer", writer)] {
            assert_eq!(
                fetches.iter().max(),
                Some(&2),
                "{stage} fetches two lineages at once, never more (in flight at each of its \
                 fetches' arrival: {fetches:?})"
            );
        }
        assert_eq!(store.peak_fetches(), 2, "never more than two at once");
    })
    .await;
}

/// Run the Job over [`identical_layers`]' four layers at a parallelism of
/// 2, its backup-store fetches answering two at a time, with the budget
/// `budget_for` picks from the layers' charges (`None`: the default
/// budget); returns the charges and the planner's lineage-stage memory
/// report.
async fn planner_memory_of(
    budget_for: impl FnOnce(&Charges) -> Option<u64>,
) -> (Charges, Option<u64>, MemoryReport) {
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    let config = bytes_config(2);
    let charges = identical_layers(&tv, &store, &config, 4).await;
    let budget = budget_for(&charges);
    hold(&tv, &config).await;
    store.pace_fetches_in_groups(2, FETCH_PAIRING_WAIT);

    let (_plan, report, _captured) =
        run_to_the_end_with(RUN_BUDGET, &tv, &config, &store, |b| match budget {
            Some(bytes) => b.with_restore_budget_for_test(bytes),
            None => b,
        })
        .await;

    assert_eq!(report.objects.len(), 4, "every layer was restored");
    (charges, budget, report.planner_memory)
}

/// defer-45 (3). Within the default budget (2,560 MiB, far above two
/// planner charges) at a parallelism of 2, the planner's lineage stage
/// holds two decodes at once, each charged `plan_decode_charge(fetched)`
/// (rounded up to whole KiB): the stage is admitted by its gate, not merely bounded by
/// its count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_plans_two_lineages_at_once_within_a_budget_that_fits_both() {
    within_test_budget(async {
        let (charges, _, planner) = planner_memory_of(|_| None).await;

        assert_eq!(
            (
                planner.mode.as_str(),
                planner.parallelism,
                planner.max_in_flight
            ),
            ("bytes", 2, 2),
            "{planner:?}"
        );
        assert_eq!(
            planner.peak_reserved_bytes,
            2 * charges.plan.next_multiple_of(1024),
            "two decodes' charges held at once: {planner:?}"
        );
    })
    .await;
}

/// defer-45 (3). Within a budget that holds one planner charge but not two
/// (the test seam, which caps the planner's stage as well as the
/// writer's), the planner's lineage stage decodes one lineage at a time at
/// a parallelism of 2, and holds one charge at most.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_plans_one_lineage_at_a_time_within_a_budget_that_holds_one_charge() {
    within_test_budget(async {
        // One and a half charges, rounded up to whole KiB as the gate rounds.
        let (charges, budget, planner) = planner_memory_of(|charges| {
            let one = charges.plan.next_multiple_of(1024);
            Some((one + one / 2).next_multiple_of(1024))
        })
        .await;
        let one = charges.plan.next_multiple_of(1024);
        let budget = budget.expect("a budget");
        assert!(
            one <= budget && 2 * one > budget,
            "fixture: the budget, {budget} B, holds one charge of {one} B but not two"
        );

        assert_eq!(
            (planner.mode.as_str(), planner.max_in_flight),
            ("bytes", 1),
            "{planner:?}"
        );
        assert_eq!(
            planner.peak_reserved_bytes, one,
            "one decode's charge held at most: {planner:?}"
        );
    })
    .await;
}
