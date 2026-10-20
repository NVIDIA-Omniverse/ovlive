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

//! The decision engine: one poll's observations in, one replica decision out.
//!
//! Pure: no I/O, no clock. The caller passes the time of each observation,
//! so a test drives it through any sequence of polls. Its state is the window
//! of recent decisions, each pod's last counter readings (keyed by pod UID),
//! and when it last removed a pod — all in memory, all forgotten on restart.
//!
//! **The fleet** is the server pods that are Ready, not terminating, and
//! answered `/scaling/v1`. Only the fleet counts as load. A pod that answered
//! 404 (an older release mid-rollout) or 503 (not yet started), or did not
//! answer, is never counted as load and holds scale-down (the fleet is not
//! settled), so no deletion cost is ever written while such a pod is present.
//!
//! **Desired replicas** is `clamp(max(cpu, catchup, occupancy, min), min,
//! max)`, `max` winning if `min > max`:
//!
//! - **CPU** — the HPA's utilization rule over each pod's CPU rate between
//!   polls: `ceil(Σ cpu ÷ (target% × request))`, unchanged within 10 % of the
//!   target. A pod that is not Ready, younger than 30 s, not reporting, or has
//!   no rate yet counts at the target when the fleet would shrink and at zero
//!   when it would grow, so neither a cold start nor a missing reading
//!   triggers a move. Off when no CPU target is configured.
//! - **Catch-up** — enough replicas to drain the pending first catch-ups
//!   within the target time at the measured per-replica drain rate λ̂:
//!   `ceil(Σ pending ÷ (λ̂ × target))`. λ̂ is the mean over the fleet of each
//!   pod's moving average of `Δ completed ÷ Δ t`, sampled only over intervals
//!   in which the pod was saturated at both ends: at least as many catch-ups
//!   pending as its concurrent-catch-up limit (`max_concurrent_catchups`), or,
//!   when that limit is 0 (unlimited), any catch-up pending. A pod that is not
//!   saturated completes catch-ups as fast as they arrive, so its rate is the
//!   arrival rate, not what it can drain; [`CATCHUP_RATE_PRIOR`] until a pod
//!   has a sample. A small queue asks for fewer replicas than the fleet has,
//!   so it never scales up.
//! - **Occupancy** — `ceil(Σ (admitted + admission waiters) ÷ (headroom ×
//!   ceiling))`, so the fleet adds a replica before a replica fills to its
//!   subscription ceiling. Off when every pod reports an unlimited ceiling.
//!
//! **Scale-up** is immediate, at most `max(4, R)` pods per poll. **Scale-down**
//! removes one pod, and only when every condition holds: the scaler has run a
//! full window; every desired count in the window is below `R`; the fleet is
//! settled (the Deployment has exactly its replica count of pods, and every
//! server pod in the Service's EndpointSlices is Ready, not terminating, and
//! reporting); the last removal was at least one interval ago; and no
//! catch-up queue has built up: the fleet's pending catch-ups are at most
//! what one replica drains within the target time, `λ̂ × target` (the
//! feasibility gate, [`Engine::queue_threshold`]).
//!
//! The gate does not size the removed pod's reconnect burst. Every scale-down
//! leaves at least `minReplicas` pods, so the burst of one pod's subscriptions
//! reconnecting to the rest is never larger than the operator already accepts
//! by running at `minReplicas`; spreading those reconnects over time is a
//! matter for how the service ends its streams at shutdown, not for this
//! gate. A gate sized to the
//! victim's subscriptions would instead hold a fleet that grew for a burst far
//! above the size it ran at before. What the gate does refuse is adding a
//! scale-down's reconnects to a queue that is already there. With deletion
//! cost on, the removed pod is the emptiest, marked just before the
//! scale-down, so the fewest clients reconnect; otherwise the ReplicaSet
//! chooses.
//!
//! A pod with no address yet (unscheduled or Pending) is in no EndpointSlice
//! and does not hold scale-down: the ReplicaSet removes such pods before any
//! Ready one, so that scale-down ends no stream, and a fleet scaled past what
//! the cluster can schedule can still shrink.
//!
//! **The window is an unbroken run of observations.** A poll with no
//! reporting pod, or one whose API reads failed ([`Engine::observation_gap`]),
//! clears it, so a scale-down always rests on a full window of polls that
//! each observed demand. A poll held as unsettled still observed demand and
//! keeps the run.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::config::{CpuTarget, ScalerConfig};
use crate::snapshot::ScalingSnapshot;

/// Per-replica catch-up drain rate (catch-ups per second) assumed until a pod
/// has been measured. Inferred from the fleet's catch-up read rate on a
/// three-replica development deployment against a managed store during a join
/// burst; replaced by the measured rate as soon as one exists.
pub const CATCHUP_RATE_PRIOR: f64 = 7.0;

/// Lower bound on λ̂, in catch-ups per second per replica. A pod that completed
/// nothing over a busy interval is stalled, not draining at zero; the bound
/// keeps a stall from turning into an unbounded replica count while the
/// catch-up term still asks for `Σ pending ÷ target` replicas, which the
/// maximum clamps.
pub const CATCHUP_RATE_FLOOR: f64 = 1.0;

/// Weight of the newest sample in each pod's moving average of its drain rate.
pub const CATCHUP_RATE_EWMA_WEIGHT: f64 = 0.3;

/// The CPU term leaves the count unchanged while utilization is within this
/// fraction of the target: the HPA's default tolerance.
pub const CPU_TOLERANCE: f64 = 0.1;

/// A pod younger than this is still warming up: its CPU reading does not count.
pub const WARMUP: Duration = Duration::from_secs(30);

/// Scale-up adds at most `max(MIN_SCALE_UP_STEP, R)` pods per poll: the HPA's
/// default scale-up policy (4 pods or 100 %, whichever is more).
pub const MIN_SCALE_UP_STEP: u32 = 4;

/// The fleet's budget of catch-up atomic snapshot reads running at once
/// against the store. `maxReplicas × LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`
/// above it lets a fully scaled-out fleet block the store for longer per join
/// wave than writers tolerate; the scaler warns once when it sees that.
pub const FLEET_READ_BUDGET: u64 = 10;

/// The engine's settings, taken from [`ScalerConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub cpu: Option<CpuTarget>,
    pub catchup_target: Duration,
    pub occupancy_headroom_percent: u32,
    pub scale_down_window: Duration,
    pub scale_down_interval: Duration,
    pub pod_deletion_cost: bool,
}

impl From<&ScalerConfig> for EngineConfig {
    fn from(c: &ScalerConfig) -> Self {
        Self {
            min_replicas: c.min_replicas,
            max_replicas: c.max_replicas,
            cpu: c.cpu,
            catchup_target: c.catchup_target,
            occupancy_headroom_percent: c.occupancy_headroom_percent,
            scale_down_window: c.scale_down_window,
            scale_down_interval: c.scale_down_interval,
            pod_deletion_cost: c.pod_deletion_cost,
        }
    }
}

/// What one pod's `/scaling/v1` request produced.
#[derive(Debug, Clone, PartialEq)]
pub enum PodReport {
    /// It answered with its snapshot.
    Snapshot(ScalingSnapshot),
    /// 404: it does not serve the snapshot (an older release).
    NotServed,
    /// 503: it serves the path but has not finished starting.
    NotRegistered,
    /// No usable answer: unreachable, timed out, another status, or a body
    /// that did not parse.
    Unreachable,
}

impl PodReport {
    fn snapshot(&self) -> Option<&ScalingSnapshot> {
        match self {
            PodReport::Snapshot(s) => Some(s),
            _ => None,
        }
    }
}

/// One non-terminating server pod at one poll.
#[derive(Debug, Clone, PartialEq)]
pub struct PodObservation {
    pub uid: String,
    pub name: String,
    pub ready: bool,
    pub report: PodReport,
}

/// One poll's input.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub now: Instant,
    /// `spec.replicas` of the Deployment's `/scale`.
    pub replicas: u32,
    /// `status.replicas` of the same `/scale`: the Deployment's pods that are
    /// not terminal or being deleted, Pending and surge pods included. Differs
    /// from `replicas` while the Deployment is adding or removing pods, a
    /// rollout's surge included.
    pub status_replicas: u32,
    /// Every non-terminating server pod. Terminating pods are excluded before
    /// any accounting: they are leaving, and their streams with them.
    pub pods: Vec<PodObservation>,
    /// How many terminating pods were excluded, for the pod-state gauge.
    pub terminating: usize,
}

/// Server pods by state, a partition of the EndpointSlice endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PodCounts {
    /// Ready, not terminating, and answered: the fleet.
    pub reporting: usize,
    /// Ready and not terminating, but no snapshot.
    pub not_reporting: usize,
    /// Not Ready, not terminating.
    pub unready: usize,
    pub terminating: usize,
}

/// Desired replicas by term, before the bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Terms {
    /// `None` when the CPU term is off or no pod has a usable reading.
    pub cpu: Option<u32>,
    pub catchup: u32,
    /// `None` when every pod's subscription ceiling is unlimited.
    pub occupancy: Option<u32>,
}

/// The term that set a scale-up's target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpReason {
    Min,
    Cpu,
    Catchup,
    Occupancy,
}

impl UpReason {
    pub fn label(self) -> &'static str {
        match self {
            UpReason::Min => "min",
            UpReason::Cpu => "cpu",
            UpReason::Catchup => "catchup",
            UpReason::Occupancy => "occupancy",
        }
    }
}

/// Why a scale-down the desired count asked for did not happen this poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    /// The scaler has not yet observed a full window since it started.
    Startup,
    /// A desired count in the window is not below the current count, or the
    /// polls since the last gap in observation do not yet span a window.
    Window,
    /// A server pod is unready, terminating or not reporting, or the
    /// Deployment has more or fewer pods than its replica count.
    Unsettled,
    /// The last removal was less than one interval ago.
    Interval,
    /// Catch-ups are already queued: the fleet's pending catch-ups exceed what
    /// one replica drains within the target time
    /// ([`Engine::queue_threshold`]).
    Feasibility,
}

impl BlockReason {
    pub const ALL: [BlockReason; 5] = [
        BlockReason::Startup,
        BlockReason::Window,
        BlockReason::Unsettled,
        BlockReason::Interval,
        BlockReason::Feasibility,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BlockReason::Startup => "startup",
            BlockReason::Window => "window",
            BlockReason::Unsettled => "unsettled",
            BlockReason::Interval => "interval",
            BlockReason::Feasibility => "feasibility",
        }
    }
}

/// What to do this poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// No pod is in the fleet: nothing was decided, no sample was taken, and
    /// the window's run of observations restarts.
    NoData,
    /// Keep the current count.
    Hold,
    ScaleUp {
        to: u32,
        reason: UpReason,
    },
    /// Remove one pod. With deletion cost on, `deletion_costs` names every
    /// non-terminating pod and its cost, to be written before the scale-down;
    /// otherwise it is empty.
    ScaleDown {
        to: u32,
        deletion_costs: Vec<(String, i32)>,
    },
}

/// `maxReplicas × P_a` over [`FLEET_READ_BUDGET`], reported once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadBudgetExceeded {
    pub max_replicas: u32,
    /// The largest `catchup_read_concurrency` any pod reported.
    pub read_concurrency: u32,
}

/// One poll's decision, with what it was computed from.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub pods: PodCounts,
    pub terms: Terms,
    /// The bounded desired count. Equal to the current count on
    /// [`Action::NoData`].
    pub desired: u32,
    pub action: Action,
    pub scale_down_blocked: Option<BlockReason>,
    /// λ̂, the per-replica drain rate the catch-up term and the feasibility
    /// gate used: the fleet mean, the prior or the floor. On
    /// [`Action::NoData`] the fleet is empty and nothing used it; it is then
    /// the prior.
    pub catchup_rate: f64,
    /// Each fleet pod's own drain-rate average, by pod name, for the pods that
    /// have one (no prior, no floor).
    pub pod_catchup_rates: Vec<(String, f64)>,
    /// `Some` on the first poll that sees the read budget exceeded, and never
    /// again in this process.
    pub read_budget_exceeded: Option<ReadBudgetExceeded>,
}

/// A pod's readings at its previous poll, and the rates derived from them.
#[derive(Debug, Clone)]
struct PodHistory {
    at: Instant,
    cpu_seconds: Option<f64>,
    completed: u64,
    pending: u64,
    /// The pod's concurrent-catch-up limit at this reading; 0 = unlimited.
    catchup_limit: u32,
    /// CPU cores used between the previous two readings.
    cpu_rate: Option<f64>,
    /// Moving average of the drain rate, catch-ups per second.
    catchup_rate: Option<f64>,
}

/// The decision engine. See the module docs.
#[derive(Debug)]
pub struct Engine {
    cfg: EngineConfig,
    started: Instant,
    pods: HashMap<String, PodHistory>,
    /// `(time, desired)` per decided poll since the last gap in observation,
    /// oldest first. Holds the newest sample at or before the window's start
    /// as well, so a full window can be told from a short one; a gap clears
    /// it, so that sample cannot predate the gap.
    window: VecDeque<(Instant, u32)>,
    last_scale_down: Option<Instant>,
    read_budget_warned: bool,
}

impl Engine {
    /// An engine that started at `started`: it will not scale down before
    /// `started + window`.
    pub fn new(cfg: EngineConfig, started: Instant) -> Self {
        Self {
            cfg,
            started,
            pods: HashMap::new(),
            window: VecDeque::new(),
            last_scale_down: None,
            read_budget_warned: false,
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// A poll that observed nothing because its API reads failed. Breaks the
    /// window's run of observations, as a poll with no reporting pod does, so
    /// the next scale-down needs a full window of polls after the gap.
    pub fn observation_gap(&mut self) {
        self.window.clear();
    }

    /// Record an action the caller carried out. Only a completed scale-down
    /// matters: it starts the interval before the next one. An action that
    /// failed is not recorded, so it is retried on a later poll.
    pub fn applied(&mut self, action: &Action, at: Instant) {
        if matches!(action, Action::ScaleDown { .. }) {
            self.last_scale_down = Some(at);
        }
    }

    /// Decide one poll.
    pub fn decide(&mut self, obs: &Observation) -> Decision {
        let pods = count_pods(obs);
        self.update_histories(obs);

        let fleet: Vec<&PodObservation> = obs
            .pods
            .iter()
            .filter(|p| p.ready && p.report.snapshot().is_some())
            .collect();
        let catchup_rate = self.catchup_rate(&fleet);
        let pod_catchup_rates = fleet
            .iter()
            .filter_map(|p| {
                self.pods
                    .get(&p.uid)
                    .and_then(|h| h.catchup_rate)
                    .map(|r| (p.name.clone(), r))
            })
            .collect();
        let read_budget_exceeded = self.check_read_budget(obs);

        if fleet.is_empty() {
            // Nothing observed: the window's run of observations breaks here.
            self.window.clear();
            return Decision {
                pods,
                terms: Terms::default(),
                desired: obs.replicas,
                action: Action::NoData,
                scale_down_blocked: None,
                catchup_rate,
                pod_catchup_rates,
                read_budget_exceeded,
            };
        }

        let snapshots = || fleet.iter().filter_map(|p| p.report.snapshot());
        let pending: u64 = snapshots().map(|s| s.catchups_pending).sum();
        let terms = Terms {
            cpu: self.cpu_term(obs),
            catchup: self.replicas_to_drain(pending, catchup_rate),
            occupancy: self.occupancy_term(snapshots()),
        };
        let (desired, up_reason) = self.bounded(terms);

        self.record_window(obs.now, desired);

        let (action, scale_down_blocked) = if desired > obs.replicas {
            let step = MIN_SCALE_UP_STEP.max(obs.replicas);
            let to = desired.min(obs.replicas.saturating_add(step));
            (
                Action::ScaleUp {
                    to,
                    reason: up_reason,
                },
                None,
            )
        } else if desired < obs.replicas {
            match self.scale_down(obs, pending, catchup_rate) {
                Ok(action) => (action, None),
                Err(reason) => (Action::Hold, Some(reason)),
            }
        } else {
            (Action::Hold, None)
        };

        Decision {
            pods,
            terms,
            desired,
            action,
            scale_down_blocked,
            catchup_rate,
            pod_catchup_rates,
            read_budget_exceeded,
        }
    }

    /// Fold this poll's readings into each pod's history, and forget pods that
    /// left. A counter that went backwards means a new process under the same
    /// UID (a container restart): its history starts again.
    fn update_histories(&mut self, obs: &Observation) {
        let present: std::collections::HashSet<&str> =
            obs.pods.iter().map(|p| p.uid.as_str()).collect();
        self.pods.retain(|uid, _| present.contains(uid.as_str()));

        for pod in &obs.pods {
            let Some(snap) = pod.report.snapshot() else {
                continue;
            };
            let fresh = PodHistory {
                at: obs.now,
                cpu_seconds: snap.cpu_seconds_total,
                completed: snap.catchups_completed_total,
                pending: snap.catchups_pending,
                catchup_limit: snap.max_concurrent_catchups,
                cpu_rate: None,
                catchup_rate: None,
            };
            let next = match self.pods.get(&pod.uid) {
                Some(prev) => advance(prev, fresh),
                None => fresh,
            };
            self.pods.insert(pod.uid.clone(), next);
        }
    }

    /// λ̂: the mean of the fleet's per-pod drain-rate averages, the prior when
    /// none has one, and never below [`CATCHUP_RATE_FLOOR`].
    fn catchup_rate(&self, fleet: &[&PodObservation]) -> f64 {
        let rates: Vec<f64> = fleet
            .iter()
            .filter_map(|p| self.pods.get(&p.uid).and_then(|h| h.catchup_rate))
            .collect();
        let rate = if rates.is_empty() {
            CATCHUP_RATE_PRIOR
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        };
        rate.max(CATCHUP_RATE_FLOOR)
    }

    /// The feasibility gate's threshold: the pending catch-ups one replica
    /// drains within the target time, `λ̂ × target` (70 at the prior and the
    /// default 10 s). Scale-down is held while the fleet's pending catch-ups
    /// exceed it, that is, while a queue has built up.
    ///
    /// An internal choice: a fleet whose whole queue one replica clears in
    /// time has no queue a scale-down's reconnects could pile onto. The size
    /// of those reconnects is not the gate's concern: the operator's
    /// `minReplicas` sets the single-pod-loss reconnect burst they accept, and
    /// spreading a scale-down's reconnects over time belongs to how the
    /// service ends its streams at shutdown.
    pub fn queue_threshold(&self, rate: f64) -> f64 {
        rate * self.cfg.catchup_target.as_secs_f64()
    }

    /// Replicas needed to drain `pending` catch-ups within the target time.
    fn replicas_to_drain(&self, pending: u64, rate: f64) -> u32 {
        if pending == 0 {
            return 0;
        }
        ceil_to_u32(pending as f64 / (rate * self.cfg.catchup_target.as_secs_f64()))
    }

    /// The HPA's CPU utilization rule; see the module docs.
    fn cpu_term(&self, obs: &Observation) -> Option<u32> {
        let target = self.cfg.cpu?.target_cores();
        if target <= 0.0 {
            return None;
        }
        let mut used = 0.0;
        let mut measured = 0u32;
        let mut missing = 0u32;
        for pod in &obs.pods {
            let rate = pod
                .report
                .snapshot()
                .filter(|s| pod.ready && s.uptime_secs >= WARMUP.as_secs())
                .and_then(|_| self.pods.get(&pod.uid))
                .and_then(|h| h.cpu_rate);
            match rate {
                Some(rate) => {
                    used += rate;
                    measured += 1;
                }
                None => missing += 1,
            }
        }
        if measured == 0 {
            return None;
        }
        let current = obs.replicas;
        let ratio = used / (f64::from(measured) * target);
        let within = |r: f64| (1.0 - r).abs() <= CPU_TOLERANCE;
        if missing == 0 {
            return Some(if within(ratio) {
                current
            } else {
                ceil_to_u32(ratio * f64::from(measured))
            });
        }
        // Pods without a reading count at the target when the fleet would
        // shrink and at zero when it would grow, so they damp either move.
        let all = f64::from(measured + missing);
        let filled = if ratio < 1.0 {
            used + f64::from(missing) * target
        } else {
            used
        };
        let new_ratio = filled / (all * target);
        if within(new_ratio) || (ratio < 1.0 && new_ratio > 1.0) || (ratio > 1.0 && new_ratio < 1.0)
        {
            return Some(current);
        }
        let replicas = ceil_to_u32(new_ratio * all);
        if (new_ratio < 1.0 && replicas > current) || (new_ratio > 1.0 && replicas < current) {
            return Some(current);
        }
        Some(replicas)
    }

    /// Replicas that keep the fleet's subscriptions under the headroom share
    /// of the smallest reported finite ceiling. `None` when no pod reports a
    /// finite ceiling.
    fn occupancy_term<'a>(
        &self,
        snapshots: impl Iterator<Item = &'a ScalingSnapshot>,
    ) -> Option<u32> {
        let mut ceiling: Option<u64> = None;
        let mut load = 0u64;
        for s in snapshots {
            load =
                load.saturating_add(s.subscriptions_admitted.saturating_add(s.admission_waiters));
            if s.max_subscriptions > 0 {
                ceiling = Some(ceiling.map_or(s.max_subscriptions, |c| c.min(s.max_subscriptions)));
            }
        }
        let ceiling = ceiling?;
        if load == 0 {
            return Some(0);
        }
        let per_replica = ceiling as f64 * f64::from(self.cfg.occupancy_headroom_percent) / 100.0;
        Some(ceil_to_u32(load as f64 / per_replica))
    }

    /// `clamp(max(terms, min), min, max)`, `max` winning if `min > max`, and
    /// the term that set it.
    fn bounded(&self, terms: Terms) -> (u32, UpReason) {
        let mut raw = self.cfg.min_replicas;
        let mut reason = UpReason::Min;
        for (value, why) in [
            (terms.cpu, UpReason::Cpu),
            (Some(terms.catchup), UpReason::Catchup),
            (terms.occupancy, UpReason::Occupancy),
        ] {
            if let Some(v) = value.filter(|v| *v > raw) {
                raw = v;
                reason = why;
            }
        }
        let desired = raw.max(self.cfg.min_replicas).min(self.cfg.max_replicas);
        (desired, reason)
    }

    /// Add `(now, desired)` and drop samples older than the window, keeping
    /// the newest one at or before its start.
    fn record_window(&mut self, now: Instant, desired: u32) {
        self.window.push_back((now, desired));
        let Some(start) = now.checked_sub(self.cfg.scale_down_window) else {
            return;
        };
        while self
            .window
            .get(1)
            .is_some_and(|&(second, _)| second <= start)
        {
            self.window.pop_front();
        }
    }

    /// The scale-down this poll may make, or why it may not.
    fn scale_down(
        &self,
        obs: &Observation,
        pending: u64,
        rate: f64,
    ) -> Result<Action, BlockReason> {
        let now = obs.now;
        if now.saturating_duration_since(self.started) < self.cfg.scale_down_window {
            return Err(BlockReason::Startup);
        }
        let covered = match (
            self.window.front(),
            now.checked_sub(self.cfg.scale_down_window),
        ) {
            (Some(&(oldest, _)), Some(start)) => oldest <= start,
            // The window reaches back before the clock's origin: the scaler
            // cannot have observed it.
            _ => false,
        };
        let window_max = self
            .window
            .iter()
            .map(|&(_, d)| d)
            .max()
            .unwrap_or(u32::MAX);
        if !covered || window_max >= obs.replicas {
            return Err(BlockReason::Window);
        }
        let pods = count_pods(obs);
        if obs.status_replicas != obs.replicas
            || pods.terminating + pods.unready + pods.not_reporting > 0
        {
            return Err(BlockReason::Unsettled);
        }
        if self
            .last_scale_down
            .is_some_and(|t| now.saturating_duration_since(t) < self.cfg.scale_down_interval)
        {
            return Err(BlockReason::Interval);
        }
        if pending as f64 > self.queue_threshold(rate) {
            return Err(BlockReason::Feasibility);
        }
        let remaining = obs.replicas - 1;
        let deletion_costs = if self.cfg.pod_deletion_cost {
            deletion_costs(obs)
        } else {
            Vec::new()
        };
        Ok(Action::ScaleDown {
            to: remaining,
            deletion_costs,
        })
    }

    /// `true` once `maxReplicas × max(P_a)` exceeds the budget, on the first
    /// such poll only.
    fn check_read_budget(&mut self, obs: &Observation) -> Option<ReadBudgetExceeded> {
        if self.read_budget_warned {
            return None;
        }
        let read_concurrency = obs
            .pods
            .iter()
            .filter_map(|p| p.report.snapshot())
            .map(|s| s.catchup_read_concurrency)
            .max()?;
        if u64::from(self.cfg.max_replicas) * u64::from(read_concurrency) <= FLEET_READ_BUDGET {
            return None;
        }
        self.read_budget_warned = true;
        Some(ReadBudgetExceeded {
            max_replicas: self.cfg.max_replicas,
            read_concurrency,
        })
    }
}

fn count_pods(obs: &Observation) -> PodCounts {
    let mut counts = PodCounts {
        terminating: obs.terminating,
        ..PodCounts::default()
    };
    for pod in &obs.pods {
        match (pod.ready, pod.report.snapshot().is_some()) {
            (false, _) => counts.unready += 1,
            (true, true) => counts.reporting += 1,
            (true, false) => counts.not_reporting += 1,
        }
    }
    counts
}

/// The next history from the previous one and a fresh reading.
fn advance(prev: &PodHistory, mut next: PodHistory) -> PodHistory {
    let elapsed = next.at.saturating_duration_since(prev.at).as_secs_f64();
    if elapsed <= 0.0 {
        return prev.clone();
    }
    let cpu_reset = matches!((prev.cpu_seconds, next.cpu_seconds), (Some(a), Some(b)) if b < a);
    if next.completed < prev.completed || cpu_reset {
        return next;
    }
    next.cpu_rate = match (prev.cpu_seconds, next.cpu_seconds) {
        (Some(a), Some(b)) => Some((b - a) / elapsed),
        _ => None,
    };
    next.catchup_rate = prev.catchup_rate;
    if saturated(prev.pending, prev.catchup_limit) && saturated(next.pending, next.catchup_limit) {
        let sample = (next.completed - prev.completed) as f64 / elapsed;
        next.catchup_rate = Some(match prev.catchup_rate {
            Some(avg) => CATCHUP_RATE_EWMA_WEIGHT * sample + (1.0 - CATCHUP_RATE_EWMA_WEIGHT) * avg,
            None => sample,
        });
    }
    next
}

/// Whether a pod with `pending` catch-ups and a concurrent-catch-up `limit`
/// is saturated: at least as many first catch-ups pending as its
/// concurrent-catch-up limit (`pending ≥ limit`), or, with no limit (0), any
/// catch-up pending. Pending counts catch-ups in their send phase too, past
/// the server's catch-up gate, so this approximates every slot being busy rather than
/// guaranteeing it. Only intervals saturated at both ends measure the pod's
/// drain rate rather than the rate catch-ups arrived at.
fn saturated(pending: u64, limit: u32) -> bool {
    if limit == 0 {
        pending > 0
    } else {
        pending >= u64::from(limit)
    }
}

/// A reporting pod's deletion cost: what removing it would disturb.
fn cost_of(s: &ScalingSnapshot) -> u64 {
    s.subscriptions_admitted.saturating_add(s.catchups_pending)
}

/// Every non-terminating pod's deletion cost: its admitted subscriptions plus
/// pending catch-ups. Only reached for a settled fleet, in which every
/// non-terminating pod reports; the `i32::MAX` for a pod without a snapshot
/// is defensive (it would mark a pod of unknown load to be removed last) and
/// is unreachable under the settled hold.
fn deletion_costs(obs: &Observation) -> Vec<(String, i32)> {
    obs.pods
        .iter()
        .map(|p| {
            let cost = p
                .report
                .snapshot()
                // Defensive: the settled hold leaves no non-reporting pod here.
                .map_or(i32::MAX, |s| i32::try_from(cost_of(s)).unwrap_or(i32::MAX));
            (p.name.clone(), cost)
        })
        .collect()
}

/// `ceil(x)` as a replica count, saturating: a huge or non-finite value
/// becomes `u32::MAX`, which the maximum then clamps.
fn ceil_to_u32(x: f64) -> u32 {
    if x.is_nan() || x <= 0.0 {
        0
    } else if x >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        x.ceil() as u32
    }
}
