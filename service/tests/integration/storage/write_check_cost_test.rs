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

//! b.bo2 PD2: what fred's check before a write costs per write on a cluster,
//! asserted.
//!
//! On a clustered client, before the router writes a command to the one
//! connection it routes to, `live_edit::before_socket_write` decides whether
//! that write can wait on its own socket while another connection holds fed
//! frames, and if so flushes them first (`fred::live_edit::liveness`'s
//! *Never waiting on unsent frames*). The decision runs on every such write:
//! one more `Router::route`, the write predicate's atomic loads and, when the
//! write can reach the socket, one walk of the client's connections. One
//! test, [`write_check_cost_on_a_cluster_is_within_its_budget_bbo2`], drives
//! one clustered general client (a pool of 1 from the production `connect`)
//! through three shapes, interleaved over [`ROUNDS`] rounds, every shape's
//! `GET`s alternating across the primaries, all of [`VALUE_LEN`]-byte values:
//!
//! - **pipelined:** [`PIPELINES`] pipelines of [`PIPELINE_GETS`] `GET`s;
//! - **single:** [`SINGLE_GETS`] `GET`s, each awaited before the next (one
//!   command per router pass);
//! - **burst:** [`BURSTS`] bursts of [`BURST_GETS`] `GET`s, every `GET` of a
//!   burst queued before the router takes the first.
//!
//! It reads fred's test-only write-check counters
//! (`test_support::write_check_stats`: checks, their decisions' nanoseconds,
//! and the checks that went on to flush, with those flushes' nanoseconds)
//! before and after each shape, and prints `WRITE-CHECK-COST:` lines: per
//! shape, the decision's nanoseconds per check (averaged over every check,
//! early returns included, and the per-round median and MAD for context),
//! the flushes and their nanoseconds.
//!
//! # What it asserts
//!
//! - **The budget (PD2), in a release build only:** per shape, the
//!   decision's time averaged over every check — routing, the predicate and
//!   the walk, never the flush — is at most [`BUDGET_NS_PER_CHECK`]
//!   (1,000 ns). A debug build inflates CPU-side figures several times over,
//!   so a debug run prints the figures and says the budget is not judged.
//! - **Batching unchanged, in any build:** each shape's flushes per round
//!   equal the count the design predicts, with no clock involved. The
//!   derivation is at [`Shape::expected_flushes`].
//! - **Self-checks:** the store is a cluster of at least two primaries,
//!   every reply is the value, and every write ran exactly one check, so the
//!   counters cannot read zero over work they never saw.
//!
//! # How it runs
//!
//! The client and the shapes run on a current-thread runtime
//! (`snapshot_read_support::on_current_thread`): a burst's `GET`s are each
//! polled once, in order, with nothing awaited between them, so all of them
//! are in the router's channel before the router, on the same thread, takes
//! the first. On a multi-thread runtime the router could take some of them
//! while later ones were still being sent, and the burst's flush count would
//! depend on the scheduler. The `TestValkey` that seeds the values and
//! cleans up stays on the test's multi-thread runtime, whose Drop cleanup
//! needs one.
//!
//! **Build profile.** The `WRITE-CHECK-COST: setup` line names the build,
//! `profile=debug` or `profile=release` (from `cfg!(debug_assertions)`). The
//! `test-cluster-disruptive:` job runs this test in a release build, in the
//! same step as the b.26e sweep-cost report and under the same gate: only
//! when a merge request changes what it measures (`.gitlab-ci.yml`).
//!
//! In the integration binary for the reason b.26e's report is
//! (`retention_sweep_cost_test.rs`): the upload memory envelope's counting
//! allocator would add its bookkeeping to what is timed. `#[ignore]`d and
//! cluster-gated: it runs only in the `test-cluster-disruptive:` job, one
//! test at a time with the other tests sharing that job's cluster (the
//! `disruptive-cluster` test group in `.config/nextest.toml`; an override
//! there also keeps its output on a pass), so no node pause or slot migration
//! lands in a timed stretch. Without `LIVE_EDIT_TEST_CLUSTER_URL` it prints a
//! `SKIP:` line and passes outside CI, and fails under `GITLAB_CI`, where it
//! gates a merge (test-writing guide § Valkey Integration Tests). The counters come with fred's `live-edit-retention-report`
//! feature, which live-edit's `test-support` turns on.

use std::time::Duration;

use fred::bytes::Bytes;
use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface};
use fred::live_edit::retention::test_support::{write_check_stats, WriteCheckStats};
use futures::FutureExt;
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use crate::storage::snapshot_read_support::on_current_thread;

/// Rounds of the three shapes, each round running every shape once.
const ROUNDS: usize = 10;

/// Pipelines per round.
const PIPELINES: usize = 30;

/// `GET`s per pipeline.
const PIPELINE_GETS: usize = 100;

/// Single `GET`s per round, each awaited before the next.
const SINGLE_GETS: usize = 300;

/// Bursts per round.
const BURSTS: usize = 30;

/// `GET`s per burst: several per primary, far fewer than fred's
/// `max_feed_count` (200) per connection, and their frames far below the
/// 8 KiB backpressure boundary, so only the counter decides which of them
/// reach a socket.
const BURST_GETS: usize = 30;

/// The bytes of each value read.
const VALUE_LEN: usize = 1024;

/// PD2: the most the check before a write may cost, per check, in a release
/// build: the decision only (route, predicate and walk), averaged over every
/// check.
const BUDGET_NS_PER_CHECK: f64 = 1_000.0;

/// The deadline on the measurement alone; connect and setup run before it.
/// A command that stalls fails sooner, at the client's per-command timeout,
/// through its `GET`'s `expect`. This bounds the other way the measurement
/// can overrun: a cluster that stays slow but keeps answering, which would
/// otherwise run on into nextest's 120 s kill, which prints nothing. The
/// healthy run takes a few seconds.
const MEASURE_BUDGET: Duration = Duration::from_secs(80);

/// The three shapes, in the order each round runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Pipelined,
    Single,
    Burst,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::Pipelined, Shape::Single, Shape::Burst];

    fn name(self) -> &'static str {
        match self {
            Shape::Pipelined => "pipelined",
            Shape::Single => "single",
            Shape::Burst => "burst",
        }
    }

    /// The `GET`s one round of this shape writes: one check each.
    fn commands(self) -> u64 {
        let commands = match self {
            Shape::Pipelined => PIPELINES * PIPELINE_GETS,
            Shape::Single => SINGLE_GETS,
            Shape::Burst => BURSTS * BURST_GETS,
        };
        commands as u64
    }

    /// The flushes before a write one round of this shape must trigger.
    ///
    /// A check flushes only when its write can reach the socket and another
    /// connection holds fed frames. A `GET` is written as a feed when the
    /// command-queue counter is above zero (it is not a kind fred always
    /// flushes, and each shape keeps every connection far below fred's feed
    /// count and the 8 KiB boundary), and is sent when the counter is zero.
    ///
    /// - **Pipelined:** a pipeline is one router command, and the test awaits
    ///   each before sending the next, so every `GET` in it is written with
    ///   the counter at zero: sent at once. Each check walks the other
    ///   connections and finds none holding a fed frame, since every earlier
    ///   write was sent. 0.
    /// - **Single:** the same, one `GET` at a time. 0.
    /// - **Burst:** every `GET` but the last is written with the later ones
    ///   still queued, so it is only fed, and its check returns at the
    ///   predicate. The last is written with the counter at zero, so its check
    ///   walks, and finds fed `GET`s on another connection (the burst
    ///   alternates over at least two primaries, with more `GET`s than
    ///   primaries): one flush. The router's flush after that send then finds
    ///   nothing left, and the next burst starts with nothing fed. One per
    ///   burst.
    fn expected_flushes(self) -> u64 {
        match self {
            Shape::Pipelined | Shape::Single => 0,
            Shape::Burst => BURSTS as u64,
        }
    }
}

/// One round of one shape.
#[derive(Clone, Copy, Debug)]
struct Sample {
    shape: Shape,
    round: usize,
    /// The client's write-check counters over the run.
    stats: WriteCheckStats,
}

/// The gate: `LIVE_EDIT_TEST_CLUSTER_URL`, or `None` with a `SKIP:` line
/// printed when it is unset or not a `redis-cluster://` URL, as b.26e's
/// report does — outside CI only. Under `GITLAB_CI` either case fails the
/// test instead: it gates a merge on its budget, so a job that lost the
/// variable must go red, not pass without measuring. A set but unreachable
/// cluster fails the test anywhere.
fn cluster_url(test: &str) -> Option<String> {
    let in_ci = std::env::var_os("GITLAB_CI").is_some();
    let skip = |why: &str| {
        assert!(
            !in_ci,
            "{test}: {why} under GITLAB_CI; the `test-cluster-disruptive:` job must set it to \
             its cluster, or this budget gate passes without measuring"
        );
        eprintln!(
            "SKIP: {test}: {why}; run `bash scripts/dev-valkey.sh cluster-up` or let CI's \
             `test-cluster-disruptive:` job run this"
        );
    };
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        skip("LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return None;
    };
    if !url.starts_with("redis-cluster://") {
        skip("LIVE_EDIT_TEST_CLUSTER_URL is not a redis-cluster:// URL");
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
            format!("{prefix}{{{tag}}}:write-check-cost")
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

/// One round of `shape` on `member`: its `GET`s alternate across `keys`.
async fn run_shape(member: &fred::clients::Client, keys: &[String], shape: Shape) {
    match shape {
        Shape::Pipelined => {
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
        Shape::Single => {
            for i in 0..SINGLE_GETS {
                let reply: Bytes = member
                    .get(&keys[i % keys.len()])
                    .await
                    .expect("a GET of the value");
                check_value(&reply);
            }
        }
        Shape::Burst => {
            for _ in 0..BURSTS {
                let mut gets = Vec::with_capacity(BURST_GETS);
                for i in 0..BURST_GETS {
                    // fred sends a command to its router on its first poll;
                    // nothing is awaited between these polls, so on this
                    // runtime the router takes none of them until all are
                    // queued.
                    let mut get = member
                        .get::<Bytes, _>(keys[i % keys.len()].as_str())
                        .boxed_local();
                    assert!(
                        futures::poll!(&mut get).is_pending(),
                        "fixture self-check: a burst's GET waits on the router after its first \
                         poll"
                    );
                    gets.push(get);
                }
                assert_eq!(
                    member
                        .inner()
                        .counters
                        .cmd_buffer_len
                        .load(std::sync::atomic::Ordering::SeqCst),
                    BURST_GETS,
                    "fixture self-check: the whole burst is queued before the router takes any \
                     of it"
                );
                for reply in futures::future::join_all(gets).await {
                    check_value(&reply.expect("a burst's GET of the value"));
                }
            }
        }
    }
}

/// The counters' change between two readings.
fn between(before: WriteCheckStats, after: WriteCheckStats) -> WriteCheckStats {
    WriteCheckStats {
        checks: after.checks - before.checks,
        nanos: after.nanos - before.nanos,
        flushes: after.flushes - before.flushes,
        flush_nanos: after.flush_nanos - before.flush_nanos,
    }
}

/// Every round's samples, each round running every shape in [`Shape::ALL`]'s
/// order. Before a round, one untimed `GET` per primary warms every
/// connection.
async fn measure(member: &fred::clients::Client, keys: &[String]) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(ROUNDS * Shape::ALL.len());
    for round in 0..ROUNDS {
        for key in keys {
            let reply: Bytes = member.get(key).await.expect("a warm-up GET of the value");
            check_value(&reply);
        }
        for shape in Shape::ALL {
            let before = write_check_stats(member);
            run_shape(member, keys, shape).await;
            samples.push(Sample {
                shape,
                round,
                stats: between(before, write_check_stats(member)),
            });
        }
    }
    samples
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

/// The build this binary is, for the setup line: `debug` with debug
/// assertions on (cargo's `dev` and `test` profiles), `release` without.
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// One shape's figures over every round.
struct Figures {
    shape: Shape,
    /// The decision's nanoseconds per check over every check of every round.
    ns_per_check: f64,
    /// Per round, the decision's nanoseconds per check.
    per_round: Vec<f64>,
    checks: u64,
    flushes: u64,
    flush_nanos: u64,
}

fn figures(samples: &[Sample], shape: Shape) -> Figures {
    let runs: Vec<&Sample> = samples.iter().filter(|s| s.shape == shape).collect();
    let checks: u64 = runs.iter().map(|s| s.stats.checks).sum();
    let nanos: u64 = runs.iter().map(|s| s.stats.nanos).sum();
    Figures {
        shape,
        ns_per_check: nanos as f64 / checks as f64,
        per_round: runs
            .iter()
            .map(|s| s.stats.nanos as f64 / s.stats.checks as f64)
            .collect(),
        checks,
        flushes: runs.iter().map(|s| s.stats.flushes).sum(),
        flush_nanos: runs.iter().map(|s| s.stats.flush_nanos).sum(),
    }
}

/// The `WRITE-CHECK-COST:` lines for `figures`, on a cluster of `primaries`.
fn report(figures: &[Figures], primaries: usize) {
    eprintln!(
        "WRITE-CHECK-COST: setup profile={} primaries={primaries} rounds={ROUNDS}; per round: \
         {PIPELINES} pipelines x {PIPELINE_GETS} GETs, {SINGLE_GETS} single GETs, {BURSTS} \
         bursts x {BURST_GETS} GETs, GETs alternating across the primaries, values {VALUE_LEN} \
         B; one pool-1 clustered general client from the production connect, on a \
         current-thread runtime{}",
        build_profile(),
        if cfg!(debug_assertions) {
            "; a debug build inflates the CPU-side figures: the budget is judged only in a \
             release run"
        } else {
            ""
        }
    );
    for f in figures {
        eprintln!(
            "WRITE-CHECK-COST: shape={} decision_ns_per_check={:.0} (over {} checks; per-round \
             median={:.0} mad={:.0}) flushes={} (expected {}) flush_nanos={} \
             flush_ns_per_flush={}",
            f.shape.name(),
            f.ns_per_check,
            f.checks,
            median(&f.per_round),
            mad(&f.per_round),
            f.flushes,
            f.shape.expected_flushes() * ROUNDS as u64,
            f.flush_nanos,
            if f.flushes == 0 {
                "n/a".to_owned()
            } else {
                format!("{:.0}", f.flush_nanos as f64 / f.flushes as f64)
            }
        );
    }
    let checks: u64 = figures.iter().map(|f| f.checks).sum();
    let nanos: f64 = figures
        .iter()
        .map(|f| f.ns_per_check * f.checks as f64)
        .sum();
    let all = nanos / checks as f64;
    eprintln!(
        "WRITE-CHECK-COST: PD2 decision ns per check over every check of every shape = {all:.0} \
         ns at P={primaries}; budget {BUDGET_NS_PER_CHECK:.0} ns per shape: {}",
        if cfg!(debug_assertions) {
            "not judged (debug build)"
        } else if figures
            .iter()
            .all(|f| f.ns_per_check <= BUDGET_NS_PER_CHECK)
        {
            "within"
        } else {
            "ABOVE"
        }
    );
}

/// b.bo2 PD2: the check before a write costs at most
/// [`BUDGET_NS_PER_CHECK`] per check on a cluster (release build), and
/// batching is unchanged (every build), by the module docs' method.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; b.bo2 PD2 write-check cost on a cluster, asserted in a \
            release build, when the merge request changes what it measures"]
async fn write_check_cost_on_a_cluster_is_within_its_budget_bbo2() {
    let test = "write_check_cost_on_a_cluster_is_within_its_budget_bbo2";
    let Some(url) = cluster_url(test) else {
        return;
    };
    let tv = TestValkey::setup_with_url(url.clone(), 1).await;
    let keys = key_per_primary(tv.client().pool.member(0), tv.prefix());
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

    on_current_thread(move || async move {
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
        let samples = tokio::time::timeout(MEASURE_BUDGET, measure(&member, &keys))
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{test}: the measurement did not finish within {MEASURE_BUDGET:?}: the \
                     cluster answered, but too slowly for the rounds to complete"
                )
            });
        let figures: Vec<Figures> = Shape::ALL
            .iter()
            .map(|shape| figures(&samples, *shape))
            .collect();
        report(&figures, keys.len());

        for sample in &samples {
            let (shape, round) = (sample.shape.name(), sample.round);
            assert_eq!(
                sample.stats.checks,
                sample.shape.commands(),
                "fixture self-check: every write of round {round}'s {shape} shape ran one check \
                 before it"
            );
            assert_eq!(
                sample.stats.flushes,
                sample.shape.expected_flushes(),
                "batching unchanged: round {round}'s {shape} shape must flush before a write \
                 exactly as the design predicts (`Shape::expected_flushes`); {:?}",
                sample.stats
            );
        }
        if !cfg!(debug_assertions) {
            for f in &figures {
                assert!(
                    f.ns_per_check <= BUDGET_NS_PER_CHECK,
                    "PD2: the check before a write must cost at most {BUDGET_NS_PER_CHECK:.0} ns \
                     per check in a release build; the {} shape's decisions took {:.0} ns per \
                     check over {} checks",
                    f.shape.name(),
                    f.ns_per_check,
                    f.checks
                );
            }
        }
        drop(member);
        client.quit().await;
    })
    .await;
}
