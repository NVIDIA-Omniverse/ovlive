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

//! b.26e f3c: what fred's connection-buffer sweep costs per command on a
//! cluster, report only.
//!
//! The sweep (`fred::live_edit::retention`, `sweep`) runs first on every
//! router pass, about once per command, and on a clustered client it folds
//! the warm election over one connection per primary. On a single-node store
//! the warm rule is a no-op, so only a cluster shows what it costs. One test,
//! [`retention_sweep_cost_on_a_cluster_is_reported_b26e`], times one
//! clustered general client (a pool of 1 from the production `connect`)
//! under three test sweep modes (`test_support::set_sweep_for_test`),
//! interleaved over [`ROUNDS`] rounds with the order rotated each round:
//!
//! - `AsBuilt`: the sweep as built, one warm connection per client;
//! - `EveryConnectionWarm`: the per-connection rule alone, with no cuts (the
//!   warm rule off);
//! - `Skipped`: no sweep work, and every drained connection cut at its reply.
//!
//! Each arm's turn in a round runs [`PIPELINES`] pipelines of
//! [`PIPELINE_GETS`] GETs, then [`SINGLE_GETS`] single GETs (one command per
//! router pass, the sweep's worst case), every workload's GETs alternating
//! across the primaries, all of [`VALUE_LEN`]-byte values. It prints
//! `SWEEP-COST:` lines:
//!
//! - per arm and workload, the median and the median absolute deviation
//!   (MAD) over the rounds of the wall time per command, and the sweeps run
//!   per command;
//! - for `AsBuilt`, the sweep's own time per pass and per command
//!   (`test_support::sweep_stats`);
//! - the inputs of the two results that need action: (a) `AsBuilt`'s median
//!   sweep time per pass above 1 µs; (b) `AsBuilt`'s single-GET wall time per
//!   command above `EveryConnectionWarm`'s, judged on the paired difference:
//!   the per-round `AsBuilt` − `EveryConnectionWarm` differences, whose median
//!   needs action above max(3 × their MAD, 2 % of `EveryConnectionWarm`'s
//!   median). Both arms' own medians and MADs are printed for context, with
//!   the verdict the unpaired reading (the arms' medians against 3 × the
//!   larger arm MAD) would give, labelled `context`; when the two readings
//!   give different verdicts a `SWEEP-COST: (b) readings disagree` line says
//!   so.
//!
//! It asserts nothing about the figures. Wall-clock figures never gate
//! (test-writing guide § Performance-Sensitive Tests); they are read from the
//! job log or its JUnit XML, which the test's `.config/nextest.toml` override
//! keeps on a pass. Its only assertions are fixture self-checks: the store is
//! a cluster of at least two primaries, every reply is the value, and every
//! arm's client ran sweeps (so it has the general class limits) and every
//! `AsBuilt` run timed them (so a broken `sweep_nanos` counter cannot print a
//! 0 ns pass as within the limit).
//!
//! **Build profile.** The `SWEEP-COST: setup` line names the build:
//! `profile=debug` or `profile=release`, from `cfg!(debug_assertions)`. A
//! debug build inflates the CPU-side figures (the sweep's own time and the
//! client's work per command, against a round trip that does not shrink with
//! it), so checks (a) and (b) are judged only from a release run. The
//! `test-cluster-disruptive:` job runs this test in a release build, in its
//! own step, only when a merge request changes the sweep, the connection
//! buffer limits or this test (`.gitlab-ci.yml`). A debug run, such as a
//! local one, shows only that the test runs.
//!
//! In the integration binary rather than the upload memory envelope's
//! (`tests/upload_memory_envelope`), whose counting global allocator would
//! add its bookkeeping to every allocation timed here. `#[ignore]`d and
//! cluster-gated: it runs only in the `test-cluster-disruptive:` job
//! (`.gitlab-ci.yml`), whose 3-shard cluster it shares one test at a time
//! with the `cluster_subscriber_disruptive_*` tests (the `disruptive-cluster`
//! test group), so no node pause or slot migration lands in a timed stretch.
//! Without `LIVE_EDIT_TEST_CLUSTER_URL` it prints a `SKIP:` line and passes.
//! The sweep modes and counters come with fred's
//! `live-edit-retention-report` feature, which live-edit's `test-support`
//! turns on and this binary needs anyway.

use std::time::{Duration, Instant};

use fred::bytes::Bytes;
use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface};
use fred::live_edit::retention::test_support::{
    set_sweep_for_test, sweep_stats, SweepForTest, SweepStats,
};
use live_edit::storage::test_support::{connect_test_client, TestValkey};

/// Rounds of the three arms, each round running every arm once.
const ROUNDS: usize = 10;

/// Pipelines per arm per round.
const PIPELINES: usize = 30;

/// GETs per pipeline.
const PIPELINE_GETS: usize = 100;

/// Single GETs per arm per round, each awaited before the next.
const SINGLE_GETS: usize = 300;

/// The bytes of each value read.
const VALUE_LEN: usize = 1024;

/// The deadline on [`measure`] alone; connect and setup run before it. A
/// command that stalls fails sooner, at the client's per-command timeout,
/// through its GET's `expect`. This bounds the other way the measurement can
/// overrun: a cluster that stays slow but keeps answering, which would
/// otherwise run on into nextest's 120 s kill, which prints nothing. The
/// healthy run takes a few seconds.
const MEASURE_BUDGET: Duration = Duration::from_secs(80);

/// The arms, in the order of the first round.
const ARMS: [SweepForTest; 3] = [
    SweepForTest::AsBuilt,
    SweepForTest::EveryConnectionWarm,
    SweepForTest::Skipped,
];

/// The two workloads, in the order each arm's turn runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    Pipelined,
    Single,
}

impl Workload {
    const ALL: [Workload; 2] = [Workload::Pipelined, Workload::Single];

    fn name(self) -> &'static str {
        match self {
            Workload::Pipelined => "pipelined",
            Workload::Single => "single",
        }
    }

    /// The commands one turn of this workload sends.
    fn commands(self) -> usize {
        match self {
            Workload::Pipelined => PIPELINES * PIPELINE_GETS,
            Workload::Single => SINGLE_GETS,
        }
    }
}

/// One timed run of one workload under one arm.
#[derive(Clone, Copy, Debug)]
struct Sample {
    arm: SweepForTest,
    workload: Workload,
    /// Wall time per command, in nanoseconds.
    wall_ns_per_command: f64,
    /// The client's sweeps over the run (`sweep_stats` differences).
    sweeps: SweepStats,
}

/// The gate: `LIVE_EDIT_TEST_CLUSTER_URL`, or `None` with a `SKIP:` line
/// printed when it is unset or not a `redis-cluster://` URL (the pattern
/// `cluster_subscriber_test.rs`'s `cluster_fixture` follows). A set but
/// unreachable cluster fails the test.
fn cluster_url(test: &str) -> Option<String> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: {test}: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster \
             configured); run `bash scripts/dev-valkey.sh cluster-up` or let CI's \
             `test-cluster-disruptive:` job run this"
        );
        return None;
    };
    if !url.starts_with("redis-cluster://") {
        eprintln!("SKIP: {test}: LIVE_EDIT_TEST_CLUSTER_URL is not a redis-cluster:// URL");
        return None;
    }
    Some(url)
}

/// One key per primary of `member`'s cluster, under `prefix`: each carries a
/// hash tag whose slot that primary owns, so it is read on that primary's
/// connection. In the routing table's primary order.
fn key_per_primary(member: &fred::clients::Client, prefix: &str) -> Vec<String> {
    let routing = member
        .cached_cluster_state()
        .expect("a connected cluster client has its routing");
    routing
        .unique_primary_nodes()
        .iter()
        .map(|primary| {
            let tag = (0..)
                .map(|i| format!("t{i}"))
                .find(|tag| {
                    routing.get_server(fred::util::redis_keyslot(tag.as_bytes())) == Some(primary)
                })
                .expect("some tag hashes to every primary");
            format!("{prefix}{{{tag}}}:sweep-cost")
        })
        .collect()
}

/// Check one reply is the value.
fn check_value(reply: &Bytes) {
    assert_eq!(
        reply.len(),
        VALUE_LEN,
        "fixture self-check: every GET answers the value"
    );
}

/// `GET key` on `member`, the reply checked.
async fn get(member: &fred::clients::Client, key: &str) {
    let reply: Bytes = member.get(key).await.expect("a GET of the value");
    check_value(&reply);
}

/// One turn of `workload` on `member`: its GETs alternate across `keys`.
async fn run_workload(member: &fred::clients::Client, keys: &[String], workload: Workload) {
    match workload {
        Workload::Pipelined => {
            for _ in 0..PIPELINES {
                let pipeline = member.pipeline();
                for i in 0..PIPELINE_GETS {
                    let _: () = pipeline
                        .get(&keys[i % keys.len()])
                        .await
                        .expect("a GET queues in the pipeline");
                }
                let replies: Vec<Result<Bytes, fred::error::Error>> = pipeline.try_all().await;
                assert_eq!(
                    replies.len(),
                    PIPELINE_GETS,
                    "fixture self-check: one reply per pipelined GET"
                );
                for reply in &replies {
                    check_value(reply.as_ref().expect("a pipelined GET of the value"));
                }
            }
        }
        Workload::Single => {
            for i in 0..SINGLE_GETS {
                get(member, &keys[i % keys.len()]).await;
            }
        }
    }
}

/// The sweeps between two readings.
fn sweeps_between(before: SweepStats, after: SweepStats) -> SweepStats {
    SweepStats {
        passes: after.passes - before.passes,
        nanos: after.nanos - before.nanos,
    }
}

/// The median of `values` (the mean of the middle two for an even count).
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The median absolute deviation of `values` from their median.
fn mad(values: &[f64]) -> f64 {
    let centre = median(values);
    let deviations: Vec<f64> = values.iter().map(|v| (v - centre).abs()).collect();
    median(&deviations)
}

/// `values` as whole numbers, for a report line.
fn rounded(values: &[f64]) -> Vec<i64> {
    values.iter().map(|v| v.round() as i64).collect()
}

/// The samples of `arm` on `workload`, in round order.
fn samples_of(samples: &[Sample], arm: SweepForTest, workload: Workload) -> Vec<Sample> {
    samples
        .iter()
        .filter(|s| s.arm == arm && s.workload == workload)
        .copied()
        .collect()
}

/// Every round's samples: each round runs every arm, in [`ARMS`]'s order
/// rotated by the round. Before an arm's timed workloads, one untimed GET per
/// primary brings the client into that arm's steady state (what the previous
/// arm left is cut, regrown or re-elected there, not in the timing).
async fn measure(member: &fred::clients::Client, keys: &[String]) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(ROUNDS * ARMS.len() * Workload::ALL.len());
    for round in 0..ROUNDS {
        for turn in 0..ARMS.len() {
            let arm = ARMS[(round + turn) % ARMS.len()];
            set_sweep_for_test(member, arm);
            for key in keys {
                get(member, key).await;
            }
            for workload in Workload::ALL {
                let before = sweep_stats(member);
                let started = Instant::now();
                run_workload(member, keys, workload).await;
                let wall = started.elapsed();
                let sweeps = sweeps_between(before, sweep_stats(member));
                samples.push(Sample {
                    arm,
                    workload,
                    wall_ns_per_command: wall.as_nanos() as f64 / workload.commands() as f64,
                    sweeps,
                });
            }
        }
    }
    set_sweep_for_test(member, SweepForTest::AsBuilt);
    samples
}

/// The build this binary is, for the setup line: `debug` with debug
/// assertions on (cargo's `dev` and `test` profiles), `release` without.
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// The `SWEEP-COST:` lines for `samples`, on a cluster of `primaries`.
fn report(samples: &[Sample], primaries: usize) {
    eprintln!(
        "SWEEP-COST: setup profile={} primaries={primaries} rounds={ROUNDS} arms interleaved \
         (order rotated per round); per arm per round: {PIPELINES} pipelines x \
         {PIPELINE_GETS} GETs, then {SINGLE_GETS} single GETs, GETs alternating across the \
         primaries, values {VALUE_LEN} B; one pool-1 clustered general client from the \
         production connect{}",
        build_profile(),
        if cfg!(debug_assertions) {
            "; a debug build inflates the CPU-side figures: judge checks (a) and (b) only from \
             a release run"
        } else {
            ""
        }
    );
    for workload in Workload::ALL {
        for arm in ARMS {
            let runs = samples_of(samples, arm, workload);
            let walls: Vec<f64> = runs.iter().map(|s| s.wall_ns_per_command).collect();
            let sweeps_per_command: Vec<f64> = runs
                .iter()
                .map(|s| s.sweeps.passes as f64 / workload.commands() as f64)
                .collect();
            eprintln!(
                "SWEEP-COST: arm={arm:?} workload={} wall_ns_per_command median={:.0} mad={:.0} \
                 sweeps_per_command median={:.3} rounds={:?}",
                workload.name(),
                median(&walls),
                mad(&walls),
                median(&sweeps_per_command),
                rounded(&walls)
            );
        }
    }
    let as_built: Vec<Sample> = samples
        .iter()
        .filter(|s| s.arm == SweepForTest::AsBuilt)
        .copied()
        .collect();
    let mut per_pass_all = Vec::new();
    for workload in Workload::ALL {
        let runs = samples_of(samples, SweepForTest::AsBuilt, workload);
        let per_pass: Vec<f64> = runs
            .iter()
            .map(|s| s.sweeps.nanos as f64 / s.sweeps.passes as f64)
            .collect();
        let per_command: Vec<f64> = runs
            .iter()
            .map(|s| s.sweeps.nanos as f64 / workload.commands() as f64)
            .collect();
        eprintln!(
            "SWEEP-COST: arm=AsBuilt workload={} sweep_ns_per_pass median={:.0} mad={:.0} \
             sweep_ns_per_command median={:.0} mad={:.0}",
            workload.name(),
            median(&per_pass),
            mad(&per_pass),
            median(&per_command),
            mad(&per_command)
        );
        per_pass_all.extend(per_pass);
    }
    let passes: u64 = as_built.iter().map(|s| s.sweeps.passes).sum();
    let nanos: u64 = as_built.iter().map(|s| s.sweeps.nanos).sum();
    let commands: usize = as_built.iter().map(|s| s.workload.commands()).sum();
    let median_per_pass = median(&per_pass_all);
    eprintln!(
        "SWEEP-COST: arm=AsBuilt total sweep_nanos/sweep_passes={:.0} ns ({nanos} ns / \
         {passes} passes) sweep_ns_per_command={:.0} over {commands} commands",
        nanos as f64 / passes as f64,
        nanos as f64 / commands as f64
    );
    eprintln!(
        "SWEEP-COST: check (a) AsBuilt median sweep ns per pass over every run = \
         {median_per_pass:.0} ns at P={primaries}; needs action above 1000 ns at P=3: {}",
        if median_per_pass > 1000.0 {
            "ABOVE"
        } else {
            "within"
        }
    );
    let single = |arm| -> Vec<f64> {
        samples_of(samples, arm, Workload::Single)
            .iter()
            .map(|s| s.wall_ns_per_command)
            .collect()
    };
    let (built, warm) = (
        single(SweepForTest::AsBuilt),
        single(SweepForTest::EveryConnectionWarm),
    );
    let (built_median, warm_median) = (median(&built), median(&warm));
    let (built_mad, warm_mad) = (mad(&built), mad(&warm));
    let floor = 0.02 * warm_median;
    let verdict = |above: bool| if above { "ABOVE" } else { "within" };
    // The verdict: the per-round paired differences (the arms ran in the same
    // round, so each difference cancels that round's load).
    let paired: Vec<f64> = built.iter().zip(&warm).map(|(b, w)| b - w).collect();
    let (paired_median, paired_mad) = (median(&paired), mad(&paired));
    let paired_threshold = (3.0 * paired_mad).max(floor);
    let paired_above = paired_median > paired_threshold;
    eprintln!(
        "SWEEP-COST: check (b) single-GET wall ns per command, paired AsBuilt - \
         EveryConnectionWarm per round: median={paired_median:.0} mad={paired_mad:.0} vs \
         threshold max(3 x paired MAD={:.0}, 2% x EveryConnectionWarm median={floor:.0}) = \
         {paired_threshold:.0}: {} rounds={:?}",
        3.0 * paired_mad,
        verdict(paired_above),
        rounded(&paired)
    );
    // Context only: the unpaired reading, the arms' medians against 3 x the
    // larger arm MAD.
    let excess = built_median - warm_median;
    let unpaired_threshold = (3.0 * built_mad.max(warm_mad)).max(floor);
    let unpaired_above = excess > unpaired_threshold;
    eprintln!(
        "SWEEP-COST: check (b) context, unpaired: AsBuilt median={built_median:.0} \
         mad={built_mad:.0}, EveryConnectionWarm median={warm_median:.0} mad={warm_mad:.0}; \
         excess={excess:.0} vs max(3 x larger arm MAD={:.0}, 2% floor={floor:.0}) = \
         {unpaired_threshold:.0}: {} (context)",
        3.0 * built_mad.max(warm_mad),
        verdict(unpaired_above)
    );
    if paired_above != unpaired_above {
        eprintln!(
            "SWEEP-COST: (b) readings disagree: paired says {}, unpaired (context) says {}; \
             the paired reading is the verdict",
            verdict(paired_above),
            verdict(unpaired_above)
        );
    }
}

/// Test 13 (b.26e f3c), report only: the sweep's cost per command on a
/// cluster, by the module docs' method. Self-checks only: the store is a
/// cluster of at least two primaries, every GET answers the value, every
/// arm's runs ran sweeps (the client carries the general class limits, and
/// `sweep_stats` counts in every mode), and every `AsBuilt` run's sweeps took
/// measurable time. It cannot fail on the code it
/// measures, by design; the figures are read from its `SWEEP-COST:` lines.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; b.26e f3c report-only sweep-cost measurement on a \
            cluster, in a release build, when the merge request changes what it measures"]
async fn retention_sweep_cost_on_a_cluster_is_reported_b26e() {
    let test = "retention_sweep_cost_on_a_cluster_is_reported_b26e";
    let Some(url) = cluster_url(test) else {
        return;
    };
    let tv = TestValkey::setup_with_url(url.clone(), 1).await;
    let mut config = crate::common::test_config();
    config.valkey_url = url;
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own cluster client connects");
    let member = client.pool.clients()[0].clone();
    assert!(
        member.is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment"
    );
    let keys = key_per_primary(&member, tv.prefix());
    assert!(
        keys.len() >= 2,
        "fixture self-check: the cluster has at least two primaries: {keys:?}"
    );
    for key in &keys {
        let _: () = tv
            .client()
            .pool
            .set(key.as_str(), vec![b'v'; VALUE_LEN], None, None, false)
            .await
            .expect("SET of the value");
    }

    let samples = tokio::time::timeout(MEASURE_BUDGET, measure(&member, &keys))
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{test}: the measurement did not finish within {MEASURE_BUDGET:?}: the cluster \
                 answered, but too slowly for the rounds to complete"
            )
        });
    report(&samples, keys.len());

    let idle_arms: Vec<String> = samples
        .iter()
        .filter(|s| s.sweeps.passes == 0)
        .map(|s| format!("{:?}/{}", s.arm, s.workload.name()))
        .collect();
    assert!(
        idle_arms.is_empty(),
        "fixture self-check: every run ran sweeps (a client without connection buffer limits \
         runs none, and then nothing here measures the sweep); these did not: {idle_arms:?}"
    );
    let untimed_runs: Vec<String> = samples
        .iter()
        .filter(|s| s.arm == SweepForTest::AsBuilt && s.sweeps.nanos == 0)
        .map(|s| format!("AsBuilt/{} ({} passes)", s.workload.name(), s.sweeps.passes))
        .collect();
    assert!(
        untimed_runs.is_empty(),
        "fixture self-check: every AsBuilt run timed its sweeps (sweep_nanos grew); these \
         counted passes but no time, so check (a)'s ns per pass above reads 0 and proves \
         nothing: {untimed_runs:?}"
    );
    drop(member);
    client.quit().await;
}
