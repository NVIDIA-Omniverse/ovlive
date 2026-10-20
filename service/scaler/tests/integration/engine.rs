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

//! The decision engine (`decide::Engine`), driven poll by poll with injected
//! times: no clock, no I/O.
//!
//! Every test that asserts a hold also has a case, in the same test or the
//! settled-fleet baseline [`a_settled_idle_fleet_past_its_window_scales_down_by_one`],
//! in which the same engine does move, so a predicate that ignored its input
//! (always hold, or always move) fails one of them.
//!
//! Rates are made exact: polls are [`POLL`] apart, and a CPU or completed
//! count difference over that interval is chosen to give the rate the case
//! names.

use std::time::{Duration, Instant};

use live_edit_scaler::config::CpuTarget;
use live_edit_scaler::decide::{
    Action, BlockReason, Decision, Engine, EngineConfig, Observation, PodCounts, PodObservation,
    PodReport, ReadBudgetExceeded, Terms, UpReason, CATCHUP_RATE_FLOOR, CATCHUP_RATE_PRIOR,
};
use live_edit_scaler::snapshot::ScalingSnapshot;

use crate::common::idle_snapshot;

/// Time between polls in these tests.
const POLL: Duration = Duration::from_secs(10);
/// The scale-down window: the chart's default.
const WINDOW: Duration = Duration::from_secs(600);
/// The minimum time between two scale-downs: the chart's default.
const INTERVAL: Duration = Duration::from_secs(60);
/// The catch-up target time: the chart's default. With the prior rate of 7
/// catch-ups per second per replica, one replica drains 70 in it.
const CATCHUP_TARGET: Duration = Duration::from_secs(10);

fn config() -> EngineConfig {
    EngineConfig {
        min_replicas: 1,
        max_replicas: 10,
        cpu: None,
        catchup_target: CATCHUP_TARGET,
        occupancy_headroom_percent: 80,
        scale_down_window: WINDOW,
        scale_down_interval: INTERVAL,
        pod_deletion_cost: false,
    }
}

/// `idle_snapshot()` with `edit` applied.
fn snap(edit: impl FnOnce(&mut ScalingSnapshot)) -> ScalingSnapshot {
    let mut s = idle_snapshot();
    edit(&mut s);
    s
}

fn pod(uid: &str, ready: bool, report: PodReport) -> PodObservation {
    PodObservation {
        uid: uid.to_owned(),
        name: format!("pod-{uid}"),
        ready,
        report,
    }
}

/// A Ready pod that answered with `s`.
fn reporting(uid: &str, s: ScalingSnapshot) -> PodObservation {
    pod(uid, true, PodReport::Snapshot(s))
}

/// `n` Ready, reporting, idle pods named `p0`, `p1`, …
fn idle_fleet(n: usize) -> Vec<PodObservation> {
    (0..n)
        .map(|i| reporting(&format!("p{i}"), idle_snapshot()))
        .collect()
}

/// A settled observation: the Deployment has exactly `replicas` pods and no
/// terminating endpoint.
fn observe(now: Instant, replicas: u32, pods: Vec<PodObservation>) -> Observation {
    Observation {
        now,
        replicas,
        status_replicas: replicas,
        pods,
        terminating: 0,
    }
}

/// An engine started at some `t0` that has polled `pods` (at `replicas`)
/// every [`POLL`] for one whole window, and the time of its next poll, at
/// which a scale-down is first allowed. Every poll it made was held
/// `startup`, which the function asserts, so the window is covered by polls
/// that each asked for fewer replicas than `replicas`.
fn engine_past_window(
    cfg: EngineConfig,
    replicas: u32,
    pods: &[PodObservation],
) -> (Engine, Instant) {
    let t0 = Instant::now();
    let mut engine = Engine::new(cfg, t0);
    let mut t = t0;
    while t < t0 + WINDOW {
        let d = engine.decide(&observe(t, replicas, pods.to_vec()));
        assert_eq!(
            (&d.action, d.scale_down_blocked),
            (&Action::Hold, Some(BlockReason::Startup)),
            "fixture self-check: before one window has passed the engine holds `startup`"
        );
        t += POLL;
    }
    (engine, t)
}

fn scale_down(to: u32) -> Action {
    Action::ScaleDown {
        to,
        deletion_costs: Vec::new(),
    }
}

/// `(action, blocked reason)` of a decision, for one-line comparisons.
type Outcome = (Action, Option<BlockReason>);

fn outcome(d: &Decision) -> Outcome {
    (d.action.clone(), d.scale_down_blocked)
}

// ── CPU term ────────────────────────────────────────────────────────────────

/// A fleet whose every pod used `rate` cores over the last poll interval,
/// polled twice so each pod has a CPU rate. Returns the second decision.
fn decide_at_cpu_rates(
    cfg: EngineConfig,
    replicas: u32,
    pods: &[(&str, bool, u64, f64)],
) -> Decision {
    // (uid, ready, uptime_secs, cores used over the interval)
    let t0 = Instant::now();
    let mut engine = Engine::new(cfg, t0);
    let at = |cpu_seconds: fn(f64) -> f64| -> Vec<PodObservation> {
        pods.iter()
            .map(|&(uid, ready, uptime, rate)| {
                pod(
                    uid,
                    ready,
                    PodReport::Snapshot(snap(|s| {
                        s.uptime_secs = uptime;
                        s.cpu_seconds_total = Some(cpu_seconds(rate));
                    })),
                )
            })
            .collect()
    };
    engine.decide(&observe(t0, replicas, at(|_| 0.0)));
    engine.decide(&observe(
        t0 + POLL,
        replicas,
        at(|rate| rate * POLL.as_secs_f64()),
    ))
}

/// The CPU term reproduces the HPA's utilization rule: a fleet whose CPU
/// use is just outside the 10 % tolerance around the target moves to
/// `ceil(Σ cpu ÷ target)`, and one just inside it stays where it is. The
/// target here is 50 % of a 1-core request, 0.5 cores per pod.
#[test]
fn cpu_term_moves_just_outside_the_tolerance_and_holds_just_inside_it() {
    let cfg = EngineConfig {
        cpu: Some(CpuTarget {
            request_cores: 1.0,
            target_percent: 50,
        }),
        ..config()
    };
    // (replicas, utilization ÷ target, the CPU term)
    for (replicas, ratio, expected) in [
        (3u32, 1.11, 4u32), // ceil(3 × 1.11) = 4: outside the tolerance, up
        (3, 1.09, 3),       // inside: unchanged
        (10, 0.89, 9),      // ceil(10 × 0.89) = 9: outside, down
        (10, 0.91, 10),     // inside: unchanged
    ] {
        let pods: Vec<(String, f64)> = (0..replicas)
            .map(|i| (format!("p{i}"), 0.5 * ratio))
            .collect();
        let pods: Vec<(&str, bool, u64, f64)> = pods
            .iter()
            .map(|(uid, rate)| (uid.as_str(), true, 3_600, *rate))
            .collect();
        let d = decide_at_cpu_rates(cfg.clone(), replicas, &pods);
        assert_eq!(
            d.terms.cpu,
            Some(expected),
            "{replicas} pods at {ratio} × the target: the HPA rule gives {expected}"
        );
    }
}

/// The just-above case scales up on the CPU term, and the just-inside case
/// does not move at all.
#[test]
fn cpu_term_above_the_tolerance_scales_up_and_inside_it_holds() {
    let cfg = EngineConfig {
        cpu: Some(CpuTarget {
            request_cores: 1.0,
            target_percent: 50,
        }),
        ..config()
    };
    let up = decide_at_cpu_rates(
        cfg.clone(),
        3,
        &[
            ("a", true, 3_600, 0.555),
            ("b", true, 3_600, 0.555),
            ("c", true, 3_600, 0.555),
        ],
    );
    assert_eq!(
        up.action,
        Action::ScaleUp {
            to: 4,
            reason: UpReason::Cpu
        }
    );
    let hold = decide_at_cpu_rates(
        cfg,
        3,
        &[
            ("a", true, 3_600, 0.545),
            ("b", true, 3_600, 0.545),
            ("c", true, 3_600, 0.545),
        ],
    );
    assert_eq!(outcome(&hold), (Action::Hold, None));
}

/// A pod without a usable CPU reading — not Ready, or younger than the 30 s
/// warm-up, or not reporting — counts at the target when the fleet would
/// shrink and at zero when it would grow, so it damps either move, as the
/// HPA does.
///
/// The unready and young pods report a high CPU rate (2 cores, four times
/// the target), so a term that used their readings anyway would scale up in
/// the shrink case.
#[test]
fn cpu_term_counts_pods_without_a_reading_at_the_target_shrinking_and_at_zero_growing() {
    let cfg = EngineConfig {
        cpu: Some(CpuTarget {
            request_cores: 1.0,
            target_percent: 50,
        }),
        ..config()
    };

    // Shrinking: two measured pods at half the target. Filled at the target,
    // the two unmeasured pods make (0.5 + 2 × 0.5) ÷ (4 × 0.5) = 0.75, so
    // ceil(0.75 × 4) = 3. Counted at zero they would give 1; their own
    // readings would give 9.
    let shrink = decide_at_cpu_rates(
        cfg.clone(),
        4,
        &[
            ("a", true, 3_600, 0.25),
            ("b", true, 3_600, 0.25),
            ("unready", false, 3_600, 2.0),
            ("young", true, 10, 2.0),
        ],
    );
    assert_eq!(
        shrink.terms.cpu,
        Some(3),
        "shrinking: unmeasured pods at the target"
    );

    // Growing: one measured pod at three times the target and three without a
    // reading. Counted at zero they make 1.5 ÷ (4 × 0.5) = 0.75, which
    // reverses the direction, so the count stays at 4. Counted at the target
    // they would give 6; ignored, 3.
    let t0 = Instant::now();
    let mut engine = Engine::new(cfg, t0);
    let pods = |busy_cpu: f64, idle_cpu: f64| {
        vec![
            reporting("a", snap(|s| s.cpu_seconds_total = Some(busy_cpu))),
            pod(
                "unready",
                false,
                PodReport::Snapshot(snap(|s| s.cpu_seconds_total = Some(idle_cpu))),
            ),
            reporting(
                "young",
                snap(|s| {
                    s.uptime_secs = 10;
                    s.cpu_seconds_total = Some(idle_cpu);
                }),
            ),
            pod("unreachable", true, PodReport::Unreachable),
        ]
    };
    engine.decide(&observe(t0, 4, pods(0.0, 0.0)));
    let grow = engine.decide(&observe(t0 + POLL, 4, pods(15.0, 0.0)));
    assert_eq!(grow.terms.cpu, Some(4), "growing: unmeasured pods at zero");
}

// ── Catch-up term ──────────────────────────────────────────────────────────

/// Three pods each draining 7.7 catch-ups per second, with 892 queued in
/// all, need ceil(892 ÷ (7.7 × 10)) = 12 replicas to drain in 10 s; the
/// maximum of 10 clamps it, and one poll adds at most max(4, R) = 4 pods.
#[test]
fn catchup_term_sizes_the_fleet_to_drain_the_queue_and_is_clamped_at_the_maximum() {
    let t0 = Instant::now();
    let mut engine = Engine::new(config(), t0);
    let pods = |completed: u64| {
        [300u64, 300, 292]
            .iter()
            .enumerate()
            .map(|(i, &pending)| {
                reporting(
                    &format!("p{i}"),
                    snap(|s| {
                        s.catchups_pending = pending;
                        s.catchups_completed_total = completed;
                    }),
                )
            })
            .collect::<Vec<_>>()
    };
    engine.decide(&observe(t0, 3, pods(0)));
    // 77 completed per pod over the 10 s poll: 7.7 per second each.
    let d = engine.decide(&observe(t0 + POLL, 3, pods(77)));

    assert!(
        (d.catchup_rate - 7.7).abs() < 1e-9,
        "λ̂ is the fleet's measured drain rate, got {}",
        d.catchup_rate
    );
    assert_eq!(d.terms.catchup, 12);
    assert_eq!(d.desired, 10, "the maximum clamps the catch-up term");
    assert_eq!(
        d.action,
        Action::ScaleUp {
            to: 7,
            reason: UpReason::Catchup
        },
        "3 + max(4, 3) = 7 this poll"
    );
}

/// A queue the fleet can drain in the target time asks for fewer replicas
/// than it has, so it never scales up: 60 pending catch-ups at the prior
/// rate need one replica, against three.
#[test]
fn a_small_catch_up_queue_does_not_scale_up() {
    let t0 = Instant::now();
    let mut engine = Engine::new(config(), t0);
    let mut pods = idle_fleet(3);
    pods[0].report = PodReport::Snapshot(snap(|s| s.catchups_pending = 60));
    let d = engine.decide(&observe(t0, 3, pods));
    assert_eq!(d.terms.catchup, 1);
    assert!(
        !matches!(d.action, Action::ScaleUp { .. }),
        "a drainable queue must not scale up, got {:?}",
        d.action
    );
}

/// λ̂ is learned only over intervals in which the pod was saturated at both
/// ends: at least as many first catch-ups pending as its concurrent-catch-up
/// limit, or, with no limit (0), any catch-up pending. A pod below its limit
/// completes catch-ups as fast as they arrive, so its interval says nothing
/// about its drain rate and leaves the prior in place. A saturated
/// interval's rate becomes the pod's average; later samples join it with
/// weight 0.3; a stalled pod is floored at 1 per second; and a counter that
/// went backwards (a restarted process under the same pod UID) restarts the
/// history.
#[test]
fn catchup_rate_is_learned_only_from_intervals_saturated_at_both_ends() {
    // (pending, completed) at each poll, POLL apart; the pod's
    // concurrent-catch-up limit; and λ̂ after the last poll.
    type Polls = &'static [(u64, u64)];
    let cases: [(&str, Polls, u32, f64); 11] = [
        // Limit 0 (unlimited): any pending catch-up saturates.
        (
            "unlimited, idle at the start",
            &[(0, 0), (50, 100)],
            0,
            CATCHUP_RATE_PRIOR,
        ),
        (
            "unlimited, idle at the end",
            &[(50, 0), (0, 100)],
            0,
            CATCHUP_RATE_PRIOR,
        ),
        ("unlimited, busy at both ends", &[(50, 0), (50, 30)], 0, 3.0),
        (
            "unlimited, a second busy interval joins with weight 0.3",
            &[(50, 0), (50, 30), (50, 160)],
            0,
            0.3 * 13.0 + 0.7 * 3.0,
        ),
        (
            "unlimited, a stalled pod",
            &[(50, 0), (50, 0)],
            0,
            CATCHUP_RATE_FLOOR,
        ),
        (
            "unlimited, a counter that went backwards",
            &[(50, 100), (50, 5)],
            0,
            CATCHUP_RATE_PRIOR,
        ),
        // A limit of 8: saturated means at least 8 pending.
        (
            "limit 8, exactly at it at both ends",
            &[(8, 0), (8, 30)],
            8,
            3.0,
        ),
        (
            "limit 8, above it at both ends",
            &[(20, 0), (9, 40)],
            8,
            4.0,
        ),
        (
            "limit 8, saturated at the start only",
            &[(8, 0), (7, 30)],
            8,
            CATCHUP_RATE_PRIOR,
        ),
        (
            "limit 8, saturated at the end only",
            &[(7, 0), (8, 30)],
            8,
            CATCHUP_RATE_PRIOR,
        ),
        (
            "limit 8, busy but below it at both ends",
            &[(7, 0), (7, 30)],
            8,
            CATCHUP_RATE_PRIOR,
        ),
    ];
    for (case, polls, limit, expected) in cases {
        let t0 = Instant::now();
        let mut engine = Engine::new(config(), t0);
        let mut last = None;
        for (i, &(pending, completed)) in polls.iter().enumerate() {
            let s = snap(|s| {
                s.catchups_pending = pending;
                s.catchups_completed_total = completed;
                s.max_concurrent_catchups = limit;
            });
            last = Some(engine.decide(&observe(t0 + POLL * i as u32, 1, vec![reporting("p", s)])));
        }
        let rate = last.expect("at least one poll").catchup_rate;
        assert!(
            (rate - expected).abs() < 1e-9,
            "{case}: λ̂ must be {expected}, got {rate}"
        );
    }
}

// ── Occupancy term ─────────────────────────────────────────────────────────

/// The occupancy term adds replicas before a replica fills: 450 admitted and
/// 30 queued against a 100-subscription ceiling at 80 % headroom need
/// ceil(480 ÷ 80) = 6 replicas. With every ceiling unlimited (`0`) the term
/// is off and the same load moves nothing.
#[test]
fn occupancy_term_scales_up_before_a_replica_fills_and_is_off_without_a_ceiling() {
    let pods = |ceiling: u64| -> Vec<PodObservation> {
        (0..3)
            .map(|i| {
                reporting(
                    &format!("p{i}"),
                    snap(|s| {
                        s.subscriptions_admitted = 150;
                        s.admission_waiters = if i == 0 { 30 } else { 0 };
                        s.max_subscriptions = ceiling;
                    }),
                )
            })
            .collect()
    };
    let t0 = Instant::now();

    let d = Engine::new(config(), t0).decide(&observe(t0, 3, pods(100)));
    assert_eq!(d.terms.occupancy, Some(6));
    assert_eq!(
        d.action,
        Action::ScaleUp {
            to: 6,
            reason: UpReason::Occupancy
        }
    );

    let d = Engine::new(config(), t0).decide(&observe(t0, 3, pods(0)));
    assert_eq!(
        d.terms.occupancy, None,
        "an unlimited ceiling turns the term off"
    );
    assert!(
        !matches!(d.action, Action::ScaleUp { .. }),
        "with the term off the load moves nothing, got {:?}",
        d.action
    );
}

// ── Scale-down timing ──────────────────────────────────────────────────────

/// The baseline every hold below is measured against: an idle fleet of three
/// that has asked for one replica for a whole window loses one pod.
#[test]
fn a_settled_idle_fleet_past_its_window_scales_down_by_one() {
    let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
    let d = engine.decide(&observe(t, 3, idle_fleet(3)));
    assert_eq!(d.desired, 1);
    assert_eq!(
        outcome(&d),
        (scale_down(2), None),
        "one pod, not straight to 1"
    );
}

/// No scale-down until every desired count over a full window is below the
/// current count. One poll at `t0 + 300 s` asked for more, so the window
/// that contains it holds `window`, through `t0 + 900 s`; the first window
/// that has left it behind, at `t0 + 910 s`, scales down.
#[test]
fn scale_down_waits_until_desired_has_stayed_below_the_count_for_a_whole_window() {
    let t0 = Instant::now();
    let mut engine = Engine::new(config(), t0);
    let busy_at = t0 + Duration::from_secs(300);
    let mut t = t0;
    while t <= t0 + Duration::from_secs(910) {
        let mut pods = idle_fleet(3);
        if t == busy_at {
            pods[0].report = PodReport::Snapshot(snap(|s| s.catchups_pending = 300));
        }
        let d = engine.decide(&observe(t, 3, pods));
        let elapsed = t - t0;
        let expected = if t == busy_at {
            (
                Action::ScaleUp {
                    to: 5,
                    reason: UpReason::Catchup,
                },
                None,
            )
        } else if elapsed < WINDOW {
            (Action::Hold, Some(BlockReason::Startup))
        } else if elapsed <= Duration::from_secs(900) {
            (Action::Hold, Some(BlockReason::Window))
        } else {
            (scale_down(2), None)
        };
        assert_eq!(outcome(&d), expected, "at t0 + {elapsed:?}");
        t += POLL;
    }
}

/// At most one pod per interval: after a scale-down the engine is told it
/// was applied, and the next one waits [`INTERVAL`]. A scale-down whose
/// write failed is not recorded, so the next poll retries it.
#[test]
fn scale_down_removes_at_most_one_pod_per_interval() {
    let (mut engine, t) = engine_past_window(config(), 5, &idle_fleet(5));
    let first = engine.decide(&observe(t, 5, idle_fleet(5)));
    assert_eq!(outcome(&first), (scale_down(4), None));

    // Not applied (the PATCH failed): the next poll tries again.
    let retried = engine.decide(&observe(t + POLL, 5, idle_fleet(5)));
    assert_eq!(outcome(&retried), (scale_down(4), None));
    engine.applied(&retried.action, t + POLL);

    let mut at = t + POLL * 2;
    while at < t + POLL + INTERVAL {
        let d = engine.decide(&observe(at, 4, idle_fleet(4)));
        assert_eq!(
            outcome(&d),
            (Action::Hold, Some(BlockReason::Interval)),
            "{:?} after the last removal",
            at - (t + POLL)
        );
        at += POLL;
    }
    let d = engine.decide(&observe(t + POLL + INTERVAL, 4, idle_fleet(4)));
    assert_eq!(outcome(&d), (scale_down(3), None), "one interval later");
}

/// A restarted scaler has forgotten its history. It scales up at once on its
/// first poll, but holds every scale-down `startup` for its first window,
/// whatever the fleet looks like, and scales down only at its end.
#[test]
fn a_fresh_engine_scales_up_at_once_but_never_down_within_its_first_window() {
    let t0 = Instant::now();
    let backlog = vec![
        reporting("p0", snap(|s| s.catchups_pending = 1_000)),
        reporting("p1", idle_snapshot()),
        reporting("p2", idle_snapshot()),
    ];
    let d = Engine::new(config(), t0).decide(&observe(t0, 3, backlog));
    assert!(
        matches!(d.action, Action::ScaleUp { .. }),
        "the first poll scales up, got {:?}",
        d.action
    );

    // `engine_past_window` asserts `startup` on every poll of the first window.
    let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
    assert_eq!(
        outcome(&engine.decide(&observe(t, 3, idle_fleet(3)))),
        (scale_down(2), None),
        "at the window's end"
    );
}

// ── Feasibility gate and deletion cost ─────────────────────────────────────

/// Pods with 10 000, 60 and 10 admitted subscriptions, `pending` catch-ups
/// queued, and none ever completed. A test that queues them on the last poll
/// only keeps λ̂ at the prior, since no interval was saturated at both ends.
fn loaded_fleet(pending: [u64; 3]) -> Vec<PodObservation> {
    [10_000u64, 60, 10]
        .iter()
        .zip(pending)
        .enumerate()
        .map(|(i, (&admitted, pending))| {
            reporting(
                &format!("p{i}"),
                snap(|s| {
                    s.subscriptions_admitted = admitted;
                    s.catchups_pending = pending;
                }),
            )
        })
        .collect()
}

/// The feasibility gate holds a scale-down only while a catch-up queue has
/// built up: while the fleet's pending catch-ups exceed what one replica
/// drains within the target time, `Engine::queue_threshold(λ̂)` = λ̂ × T (70 at
/// the prior and 10 s). It does not size the removed pod's reconnects: with
/// nothing queued, a fleet whose fullest pod holds 10 000 subscriptions
/// still scales down, with or without deletion cost. Above the threshold
/// (here one more than it) it holds `feasibility`; at the threshold and just
/// below, the scale-down goes ahead. The boundary is the fleet sum, not a
/// per-pod figure: the split rows spread the queue over two pods, none of
/// which holds more than the threshold, and the one whose sum is above it
/// still holds.
#[test]
fn feasibility_gate_holds_only_while_the_queue_exceeds_what_one_replica_drains_in_time() {
    let threshold = Engine::new(config(), Instant::now()).queue_threshold(CATCHUP_RATE_PRIOR);
    assert!(
        (threshold - 70.0).abs() < 1e-9,
        "λ̂ × T at the prior and 10 s"
    );
    let at = threshold as u64;
    let half = at / 2;

    for pod_deletion_cost in [false, true] {
        let cfg = EngineConfig {
            pod_deletion_cost,
            ..config()
        };
        for (pending, expected) in [
            ([0, 0, 0], None),
            ([at - 1, 0, 0], None),
            ([at, 0, 0], None),
            ([at + 1, 0, 0], Some(BlockReason::Feasibility)),
            // Split over two pods: 35 + 35 = 70 is at the threshold, and
            // 36 + 35 = 71 is above it with no pod above it.
            ([half, at - half, 0], None),
            ([half + 1, at - half, 0], Some(BlockReason::Feasibility)),
        ] {
            let (mut engine, t) = engine_past_window(cfg.clone(), 3, &loaded_fleet([0; 3]));
            let d = engine.decide(&observe(t, 3, loaded_fleet(pending)));
            assert!(
                (d.catchup_rate - CATCHUP_RATE_PRIOR).abs() < 1e-9,
                "fixture self-check: λ̂ is the prior"
            );
            assert_eq!(
                (
                    matches!(d.action, Action::ScaleDown { to: 2, .. }),
                    d.scale_down_blocked
                ),
                (expected.is_none(), expected),
                "{pending:?} pending (threshold {threshold}), deletion cost {pod_deletion_cost}"
            );
        }
    }
}

/// The feasibility gate uses the λ̂ the poll decided with, floor included,
/// not the prior. p0 holds 20 pending catch-ups and completes none for a
/// whole window, so its learned rate is 0 and λ̂ is floored at 1 per second:
/// the threshold is 1 × 10 s = 10, and 20 pending holds `feasibility`. At
/// the prior (threshold 70) the same queue would scale down. The catch-up
/// term asks for ceil(20 ÷ 10) = 2 replicas, below the 3 running, so the
/// window is covered and the hold is the gate's.
#[test]
fn feasibility_gate_uses_the_learned_catchup_rate() {
    let fleet = loaded_fleet([20, 0, 0]);
    let (mut engine, t) = engine_past_window(config(), 3, &fleet);
    assert!(
        20.0 < engine.queue_threshold(CATCHUP_RATE_PRIOR),
        "fixture self-check: at the prior, 20 pending would not hold"
    );
    let d = engine.decide(&observe(t, 3, fleet));
    assert!(
        (d.catchup_rate - CATCHUP_RATE_FLOOR).abs() < 1e-9,
        "λ̂ is the floor for a stalled pod, got {}",
        d.catchup_rate
    );
    assert_eq!(d.terms.catchup, 2);
    assert_eq!(outcome(&d), (Action::Hold, Some(BlockReason::Feasibility)));
}

/// `Decision.pod_catchup_rates` covers the fleet only — Ready, reporting
/// pods. A pod that stays in the EndpointSlices but turns unready or stops
/// reporting keeps its learned drain rate in the engine's history, yet is
/// exported with no rate, as it is left out of λ̂. Both pods first learn a
/// rate over one saturated interval (p0 3/s, p1 6/s); on the next poll p1
/// leaves the fleet, and only p0's rate is exported, matching λ̂.
#[test]
fn pod_catchup_rates_cover_only_the_fleet() {
    let busy = |completed: u64| {
        snap(|s| {
            s.catchups_pending = 50;
            s.catchups_completed_total = completed;
        })
    };
    /// p1's observation on its last poll, from what it would have reported.
    type Leave = fn(ScalingSnapshot) -> PodObservation;
    let leaves: [(&str, Leave); 3] = [
        ("turns unready, still reporting", |s| {
            pod("p1", false, PodReport::Snapshot(s))
        }),
        ("stops serving the snapshot", |_| {
            pod("p1", true, PodReport::NotServed)
        }),
        ("becomes unreachable", |_| {
            pod("p1", true, PodReport::Unreachable)
        }),
    ];
    for (case, p1_leaves) in leaves {
        let t0 = Instant::now();
        let mut engine = Engine::new(config(), t0);
        engine.decide(&observe(
            t0,
            2,
            vec![reporting("p0", busy(0)), reporting("p1", busy(0))],
        ));
        let d = engine.decide(&observe(
            t0 + POLL,
            2,
            vec![reporting("p0", busy(30)), reporting("p1", busy(60))],
        ));
        assert_eq!(
            d.pod_catchup_rates,
            vec![("pod-p0".to_owned(), 3.0), ("pod-p1".to_owned(), 6.0)],
            "{case}: fixture self-check: both pods learned a rate"
        );

        let d = engine.decide(&observe(
            t0 + POLL * 2,
            2,
            vec![reporting("p0", busy(60)), p1_leaves(busy(120))],
        ));
        let names: Vec<&str> = d
            .pod_catchup_rates
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names, ["pod-p0"], "{case}");
        let (_, p0_rate) = d.pod_catchup_rates[0];
        assert!(
            (p0_rate - 3.0).abs() < 1e-9 && (d.catchup_rate - p0_rate).abs() < 1e-9,
            "{case}: p0's rate is exported and is λ̂, got {p0_rate} and λ̂ {}",
            d.catchup_rate
        );
    }
}

/// With deletion cost on, a scale-down marks every pod with its admitted
/// subscriptions plus pending catch-ups, so the emptiest is marked lowest
/// and removed first.
#[test]
fn deletion_costs_mark_every_pod_and_the_emptiest_lowest() {
    let cfg = EngineConfig {
        pod_deletion_cost: true,
        ..config()
    };
    let (mut engine, t) = engine_past_window(cfg, 3, &loaded_fleet([0; 3]));
    let d = engine.decide(&observe(t, 3, loaded_fleet([0, 0, 2])));
    assert_eq!(
        outcome(&d),
        (
            Action::ScaleDown {
                to: 2,
                deletion_costs: vec![
                    ("pod-p0".to_owned(), 10_000),
                    ("pod-p1".to_owned(), 60),
                    ("pod-p2".to_owned(), 12),
                ],
            },
            None
        )
    );
}

// ── The settled hold ───────────────────────────────────────────────────────

/// Scale-down is held `unsettled` unless the Deployment has exactly its
/// replica count of pods and every server pod in the EndpointSlices is
/// Ready, not terminating and reporting. Each row changes one thing on the
/// final poll of an engine whose settled baseline scales down
/// ([`a_settled_idle_fleet_past_its_window_scales_down_by_one`]).
///
/// The exemption row is the one that does NOT hold: a pod with no address
/// yet (Pending) is in no EndpointSlice, so the fleet has one endpoint fewer
/// than `status.replicas`, and the scale-down goes ahead — the ReplicaSet
/// removes such a pod first, so it ends no stream.
#[test]
fn scale_down_is_held_unless_the_fleet_is_settled() {
    type Edit = fn(&mut Observation);
    let rows: [(&str, Edit, Outcome); 9] = [
        (
            "status.replicas above spec.replicas (a surge pod)",
            |o| o.status_replicas = 4,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "status.replicas below spec.replicas",
            |o| o.status_replicas = 2,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "one terminating endpoint",
            |o| o.terminating = 1,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "one unready, non-terminating endpoint",
            |o| o.pods[2].ready = false,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "one Ready pod answering 404",
            |o| o.pods[2].report = PodReport::NotServed,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "one Ready pod answering 503",
            |o| o.pods[2].report = PodReport::NotRegistered,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "one Ready pod unreachable",
            |o| o.pods[2].report = PodReport::Unreachable,
            (Action::Hold, Some(BlockReason::Unsettled)),
        ),
        (
            "exemption: a Pending pod with no address, status.replicas == replicas",
            |o| {
                o.pods.pop();
            },
            (scale_down(2), None),
        ),
        ("fully settled", |_| {}, (scale_down(2), None)),
    ];
    for (row, edit, expected) in rows {
        let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
        let mut obs = observe(t, 3, idle_fleet(3));
        edit(&mut obs);
        assert_eq!(outcome(&engine.decide(&obs)), expected, "{row}");
    }
}

/// With deletion cost on, a Ready pod that does not report holds the
/// scale-down, so no deletion cost is computed for any pod (the caller
/// writes the costs only as part of a scale-down).
#[test]
fn a_not_reporting_ready_pod_with_deletion_cost_on_holds_and_marks_no_pod() {
    let cfg = EngineConfig {
        pod_deletion_cost: true,
        ..config()
    };
    let (mut engine, t) = engine_past_window(cfg, 3, &idle_fleet(3));
    let mut pods = idle_fleet(3);
    pods[1].report = PodReport::Unreachable;
    let d = engine.decide(&observe(t, 3, pods));
    assert_eq!(outcome(&d), (Action::Hold, Some(BlockReason::Unsettled)));
}

/// A poll held `unsettled` still observed demand, so it keeps the window's
/// run: the next settled poll scales down at once.
#[test]
fn an_unsettled_poll_does_not_restart_the_window() {
    let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
    let mut obs = observe(t, 3, idle_fleet(3));
    obs.terminating = 1;
    assert_eq!(
        outcome(&engine.decide(&obs)),
        (Action::Hold, Some(BlockReason::Unsettled))
    );
    assert_eq!(
        outcome(&engine.decide(&observe(t + POLL, 3, idle_fleet(3)))),
        (scale_down(2), None)
    );
}

// ── Bounds ─────────────────────────────────────────────────────────────────

/// `clamp(max(terms, min), min, max)`: the minimum lifts an idle fleet (and
/// scales up with reason `min`), the maximum caps a backlog, and when the
/// minimum exceeds the maximum the maximum wins.
#[test]
fn desired_is_bounded_by_min_and_max_and_max_wins_over_a_larger_min() {
    let backlog = || vec![reporting("p0", snap(|s| s.catchups_pending = 100_000))];
    let bounded = |min, max| EngineConfig {
        min_replicas: min,
        max_replicas: max,
        ..config()
    };
    let t0 = Instant::now();

    let d = Engine::new(bounded(3, 10), t0).decide(&observe(t0, 1, idle_fleet(1)));
    assert_eq!(d.desired, 3);
    assert_eq!(
        d.action,
        Action::ScaleUp {
            to: 3,
            reason: UpReason::Min
        }
    );

    let d = Engine::new(bounded(2, 5), t0).decide(&observe(t0, 1, backlog()));
    assert_eq!(d.desired, 5, "the maximum caps the catch-up term");

    let d = Engine::new(bounded(6, 4), t0).decide(&observe(t0, 4, idle_fleet(4)));
    assert_eq!(d.desired, 4, "min > max: the maximum wins");
    assert_eq!(outcome(&d), (Action::Hold, None));
}

/// One poll adds at most `max(4, R)` pods, as the HPA's default scale-up
/// policy does.
#[test]
fn scale_up_adds_at_most_max_of_four_and_the_current_count_per_poll() {
    let cfg = EngineConfig {
        max_replicas: 50,
        ..config()
    };
    for (replicas, to) in [(2u32, 6u32), (6, 12), (30, 50)] {
        let t0 = Instant::now();
        let mut pods = idle_fleet(replicas as usize);
        pods[0].report = PodReport::Snapshot(snap(|s| s.catchups_pending = 100_000));
        let d = Engine::new(cfg.clone(), t0).decide(&observe(t0, replicas, pods));
        assert_eq!(
            d.action,
            Action::ScaleUp {
                to,
                reason: UpReason::Catchup
            },
            "from {replicas}"
        );
    }
}

/// Only the fleet — Ready, reporting pods — counts as load. An unready pod's
/// own report of a huge queue and a full ceiling moves no term, and a Ready
/// pod that answered 404 counts as not reporting.
#[test]
fn pods_outside_the_fleet_are_never_counted_as_load() {
    let t0 = Instant::now();
    let pods = vec![
        reporting("p0", snap(|s| s.max_subscriptions = 100)),
        pod(
            "unready",
            false,
            PodReport::Snapshot(snap(|s| {
                s.catchups_pending = 10_000;
                s.subscriptions_admitted = 10_000;
                s.max_subscriptions = 100;
            })),
        ),
        pod("old-release", true, PodReport::NotServed),
    ];
    let d = Engine::new(config(), t0).decide(&observe(t0, 3, pods));
    assert_eq!(
        d.terms,
        Terms {
            cpu: None,
            catchup: 0,
            occupancy: Some(0)
        }
    );
    assert_eq!(
        d.pods,
        PodCounts {
            reporting: 1,
            not_reporting: 1,
            unready: 1,
            terminating: 0
        }
    );
    assert!(
        !matches!(d.action, Action::ScaleUp { .. }),
        "{:?}",
        d.action
    );
}

// ── No data and gaps ───────────────────────────────────────────────────────

/// When no pod reports, nothing is decided: the desired count is the current
/// one and no scale-down is even considered.
#[test]
fn no_decision_is_taken_when_no_pod_reports() {
    let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
    let pods = vec![
        pod("a", true, PodReport::NotServed),
        pod("b", true, PodReport::NotRegistered),
        pod("c", true, PodReport::Unreachable),
    ];
    let d = engine.decide(&observe(t, 3, pods));
    assert_eq!(outcome(&d), (Action::NoData, None));
    assert_eq!(d.desired, 3);
    assert_eq!(d.terms, Terms::default());
}

/// A poll that observed nothing — no pod reported, or the API reads failed
/// (`observation_gap`) — breaks the window's run, so the next scale-down
/// needs a full window of observed polls after it.
#[test]
fn a_poll_that_observed_nothing_restarts_the_window() {
    for gap in ["no pod reported", "the API reads failed"] {
        let (mut engine, t) = engine_past_window(config(), 3, &idle_fleet(3));
        if gap == "no pod reported" {
            let d = engine.decide(&observe(t, 3, vec![pod("a", true, PodReport::Unreachable)]));
            assert_eq!(d.action, Action::NoData);
        } else {
            engine.observation_gap();
        }
        let mut at = t + POLL;
        while at < t + POLL + WINDOW {
            let d = engine.decide(&observe(at, 3, idle_fleet(3)));
            assert_eq!(
                outcome(&d),
                (Action::Hold, Some(BlockReason::Window)),
                "{gap}: {:?} after the gap",
                at - t
            );
            at += POLL;
        }
        assert_eq!(
            outcome(&engine.decide(&observe(at, 3, idle_fleet(3)))),
            (scale_down(2), None),
            "{gap}: a full window of observed polls after the gap"
        );
    }
}

// ── Read budget ────────────────────────────────────────────────────────────

/// `maxReplicas × catch-up read concurrency` above the fleet's budget of 10
/// is reported once, on the first poll that shows it, and never again; at
/// the budget it is not reported.
#[test]
fn the_read_budget_is_reported_once_when_exceeded_and_not_at_the_budget() {
    let fleet = |concurrency: u32| {
        vec![reporting(
            "p0",
            snap(|s| s.catchup_read_concurrency = concurrency),
        )]
    };
    let t0 = Instant::now();

    let mut engine = Engine::new(config(), t0);
    assert_eq!(
        engine
            .decide(&observe(t0, 1, fleet(2)))
            .read_budget_exceeded,
        Some(ReadBudgetExceeded {
            max_replicas: 10,
            read_concurrency: 2
        })
    );
    assert_eq!(
        engine
            .decide(&observe(t0 + POLL, 1, fleet(2)))
            .read_budget_exceeded,
        None,
        "reported once per process"
    );

    let at_budget = EngineConfig {
        max_replicas: 5,
        ..config()
    };
    assert_eq!(
        Engine::new(at_budget, t0)
            .decide(&observe(t0, 1, fleet(2)))
            .read_budget_exceeded,
        None,
        "5 × 2 = 10 is within the budget"
    );
}
