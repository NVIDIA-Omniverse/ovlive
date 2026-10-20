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

//! Derived pacing bounds — pure arithmetic over configured cadence knobs.
//!
//! A deliberate *leaf*: this module imports nothing but [`std::time::Duration`]
//! and is depended on by both [`crate::checkpoint`] and
//! [`crate::observability`]. That placement is the point rather than an
//! accident. [`stuck_threshold_secs`] has exactly one consumer in each of those
//! modules — the checkpoint loop's stuck-pending WARN, and the
//! `live_edit_checkpoint_config_stuck_threshold_seconds` gauge — and the two
//! bounds must be the same expression or an operator's alert rule and the
//! service's own verdict disagree. `checkpoint` already depends on
//! `observability::metrics::SharedMetrics`, so hosting the formula in
//! `observability` and importing it from `checkpoint` would work while hosting
//! it in `checkpoint` and importing it from `observability` would invert the
//! layering. Neither direction is a *shared* home, so the expression lives
//! below both, where either may reach it without an edge between them.
//!
//! Ownership of the concept still reads as the checkpoint module's: the
//! function is re-exported as `crate::checkpoint::stuck_threshold_secs`, which
//! is the path callers outside this crate's internals use.

use std::time::Duration;

/// The longest pending age any *configured* checkpoint deferral can explain:
/// `interval + quiesce + tick`, in seconds.
///
/// It is the checkpoint loop's own worst-case promise for a layer that is being
/// deferred legitimately, term by term:
///
/// - `interval` — the cadence upper bound. A pending layer is not even
///   *eligible* on the interval arm until its durable generation is this old,
///   so ages up to `interval` are the design, not a symptom.
/// - `quiesce` — a continuously-written layer fails the quiesce arm every time
///   it is evaluated, so before the interval arm carries it, its age can
///   legitimately drift by up to one quiesce window past the point where a
///   quiet layer would have been written.
/// - `tick` — scheduling slack. The loop only evaluates a layer when its tick
///   comes round, so any deadline can be overshot by one tick.
///
/// Past the sum, no configured deferral explains the age: the layer is stuck.
/// Not `interval` alone — that fires on every healthy interval-arm write,
/// because a cadence-driven write is *by definition* a layer whose age just
/// reached `interval`. Assembly time is not in the sum: it is unbounded in
/// principle and unmeasured here, so folding a guess for it in would make the
/// threshold a fiction; a layer whose assembly genuinely runs longer than a
/// whole quiesce window plus a tick deserves the WARN.
///
/// Two callers, deliberately (b.ijf):
///
/// 1. `PendingAgeAccumulator::publish`, for the stuck-pending WARN.
/// 2. [`crate::observability::metrics::Metrics::register_checkpoint_pacing_gauges`],
///    for the value published as
///    `live_edit_checkpoint_config_stuck_threshold_seconds`. That method
///    *derives* the gauge by calling this function on the same three durations
///    it echoes, rather than accepting the threshold as a parameter — so a
///    caller cannot hand the gauge a bound that differs from the one the WARN
///    applies, even by transposing arguments.
///
/// Evaluating the formula in exactly one place is the point: the bound the
/// service warns on and the bound an operator's alert rule computes cannot
/// drift apart.
///
/// Saturating so an absurd configured duration cannot panic a checkpoint tick;
/// the durations come from `Config::from_env` seconds values, so saturation is
/// unreachable in practice.
pub fn stuck_threshold_secs(interval: Duration, quiesce: Duration, tick: Duration) -> f64 {
    interval
        .saturating_add(quiesce)
        .saturating_add(tick)
        .as_secs_f64()
}
