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

//! The hold verdict, its observation instant and the staleness clock: pure
//! logic, no I/O.

use std::time::Duration;

use tokio::time::Instant;

use super::{STALE_VERDICT_THRESHOLD, UNHOLD_EXIT_JITTER_MAX};
use crate::storage::dr_hold::HoldMarkerFields;

/// What one successful hold read says about the fleet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HoldVerdict {
    /// No hold marker: the fleet may serve.
    Serving,
    /// The hold marker exists: the fleet is held. Each field is the marker's,
    /// `None` when the marker lacks it or, for `detected_at_ms`, when it is not
    /// a whole number of milliseconds. A marker missing fields still holds.
    Held {
        /// The incident an operator Job must name.
        incident: Option<String>,
        /// Why the fleet is held.
        reason: Option<String>,
        /// What the writer of the marker observed.
        evidence: Option<String>,
        /// The store's clock when the hold was written, in milliseconds.
        detected_at_ms: Option<u64>,
    },
}

impl HoldVerdict {
    /// Whether this verdict holds the fleet.
    pub fn is_held(&self) -> bool {
        matches!(self, Self::Held { .. })
    }

    /// The held incident, when the verdict is `Held` and the marker names one.
    pub fn incident(&self) -> Option<&str> {
        match self {
            Self::Held { incident, .. } => incident.as_deref(),
            Self::Serving => None,
        }
    }

    /// The hold's reason, when the verdict is `Held` and the marker names
    /// one.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Held { reason, .. } => reason.as_deref(),
            Self::Serving => None,
        }
    }

    /// The hold's evidence, when the verdict is `Held` and the marker carries
    /// it.
    pub fn evidence(&self) -> Option<&str> {
        match self {
            Self::Held { evidence, .. } => evidence.as_deref(),
            Self::Serving => None,
        }
    }
}

/// The verdict for one successful read of the marker: `Held` exactly when it
/// exists (the hold predicate in this release).
pub fn verdict_from_marker(marker: Option<&HoldMarkerFields>) -> HoldVerdict {
    match marker {
        None => HoldVerdict::Serving,
        Some(fields) => HoldVerdict::Held {
            incident: fields.incident.clone(),
            reason: fields.reason.clone(),
            evidence: fields.evidence.clone(),
            detected_at_ms: fields
                .detected_at_ms
                .as_deref()
                .and_then(|ms| ms.trim().parse::<u64>().ok()),
        },
    }
}

/// A verdict and the instant it was observed: the value the hold poll
/// publishes on its `watch` channel after every successful read, and only
/// then. `at` is therefore the time of the last successful verdict, which is
/// what the staleness clock ([`verdict_age`]) measures from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoldObservation {
    /// The verdict.
    pub verdict: HoldVerdict,
    /// When the read that produced it answered.
    pub at: Instant,
}

impl HoldObservation {
    /// The instant this observation goes stale if no later one replaces it:
    /// `at` plus [`STALE_VERDICT_THRESHOLD`].
    pub fn stale_at(&self) -> Instant {
        self.at + STALE_VERDICT_THRESHOLD
    }

    /// [`verdict_age`] of this observation at `now`.
    pub fn age_at(&self, now: Instant) -> VerdictAge {
        verdict_age(self.at, now)
    }
}

/// How old the last successful verdict is (SDD SR-6.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerdictAge {
    /// Younger than [`STALE_VERDICT_THRESHOLD`].
    Fresh,
    /// At least [`STALE_VERDICT_THRESHOLD`] old, however old: a serving
    /// replica that is not held fails closed on its own, and a held one
    /// reports readiness `NOT_SERVING`. Neither exits on age: both keep
    /// polling until a read succeeds.
    Stale,
}

/// The age class at `now` of a verdict last observed at `last_success`. An
/// observation stamped after `now` counts as fresh.
pub fn verdict_age(last_success: Instant, now: Instant) -> VerdictAge {
    let age = now.saturating_duration_since(last_success);
    if age >= STALE_VERDICT_THRESHOLD {
        VerdictAge::Stale
    } else {
        VerdictAge::Fresh
    }
}

/// The bound on one hold read (SDD SR-6.5 "Poll bounds"): the smaller of the
/// Valkey command timeout and the poll interval, so a read stuck on a dead
/// connection ends before the next poll is due and never holds back the
/// staleness clock. A `command_timeout_ms` of 0 (fred's timeout disabled)
/// leaves the poll interval as the bound.
pub fn poll_call_timeout(command_timeout_ms: u64, poll_interval: Duration) -> Duration {
    if command_timeout_ms == 0 {
        poll_interval
    } else {
        Duration::from_millis(command_timeout_ms).min(poll_interval)
    }
}

/// A random wait drawn uniformly from 0 to [`UNHOLD_EXIT_JITTER_MAX`]
/// inclusive, at millisecond resolution, for a replica about to exit on
/// un-hold or on recovery from a stale verdict (SDD SR-6.4).
pub fn unhold_exit_jitter() -> Duration {
    use rand::Rng;
    let max_ms = u64::try_from(UNHOLD_EXIT_JITTER_MAX.as_millis()).unwrap_or(u64::MAX);
    Duration::from_millis(rand::rng().random_range(0..=max_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A marker as `write_hold_marker` seeds it: every field present.
    fn full_marker() -> HoldMarkerFields {
        HoldMarkerFields {
            incident: Some("inc-1".to_owned()),
            reason: Some("counter lost".to_owned()),
            evidence: Some("counter absent on an established store".to_owned()),
            detected_at_ms: Some("1760000000123".to_owned()),
            detected_by: Some("pod-a".to_owned()),
        }
    }

    // ── The shared constants (SDD SR-6.4, SR-6.5) ─────────────────────────

    /// Tasks 3–6 read these values from here; the spec fixes each one.
    #[test]
    fn hold_constants_are_the_specs_values() {
        assert_eq!(STALE_VERDICT_THRESHOLD, Duration::from_secs(30));
        assert_eq!(UNHOLD_EXIT_JITTER_MAX, Duration::from_secs(5));
    }

    // ── verdict_from_marker ────────────────────────────────────────────────

    #[test]
    fn verdict_from_marker_no_marker_is_serving() {
        let verdict = verdict_from_marker(None);
        assert_eq!(verdict, HoldVerdict::Serving);
        assert!(!verdict.is_held());
        assert_eq!(verdict.incident(), None);
    }

    /// `Held` carries the marker's `incident`, `reason`, `evidence` and
    /// `detected_at_ms` (parsed); `detected_by` stays in the store.
    #[test]
    fn verdict_from_marker_full_marker_is_held_with_its_incident_reason_evidence_and_detection_time(
    ) {
        let verdict = verdict_from_marker(Some(&full_marker()));
        assert_eq!(
            verdict,
            HoldVerdict::Held {
                incident: Some("inc-1".to_owned()),
                reason: Some("counter lost".to_owned()),
                evidence: Some("counter absent on an established store".to_owned()),
                detected_at_ms: Some(1_760_000_000_123),
            }
        );
        assert!(verdict.is_held());
        assert_eq!(verdict.incident(), Some("inc-1"));
        assert_eq!(verdict.reason(), Some("counter lost"));
        assert_eq!(
            verdict.evidence(),
            Some("counter absent on an established store")
        );
    }

    /// `Serving` names no incident, reason or evidence.
    #[test]
    fn serving_verdict_names_no_incident_reason_or_evidence() {
        let verdict = verdict_from_marker(None);
        assert_eq!(verdict.incident(), None);
        assert_eq!(verdict.reason(), None);
        assert_eq!(verdict.evidence(), None);
    }

    /// The predicate is that the marker exists: a marker with no fields at
    /// all still holds the fleet.
    #[test]
    fn verdict_from_marker_marker_without_fields_still_holds() {
        let verdict = verdict_from_marker(Some(&HoldMarkerFields::default()));
        assert_eq!(
            verdict,
            HoldVerdict::Held {
                incident: None,
                reason: None,
                evidence: None,
                detected_at_ms: None,
            }
        );
        assert!(verdict.is_held());
        assert_eq!(verdict.incident(), None);
    }

    /// A `detected_at_ms` that is not a whole number of milliseconds reads as
    /// `None`, and the marker still holds; surrounding whitespace is
    /// tolerated.
    #[test]
    fn verdict_from_marker_unparseable_detection_time_is_none_and_still_holds() {
        for raw in ["", "soon", "-5", "1.5", "1e3", "18446744073709551616"] {
            let marker = HoldMarkerFields {
                detected_at_ms: Some(raw.to_owned()),
                ..full_marker()
            };
            let verdict = verdict_from_marker(Some(&marker));
            assert_eq!(
                verdict,
                HoldVerdict::Held {
                    incident: Some("inc-1".to_owned()),
                    reason: Some("counter lost".to_owned()),
                    evidence: Some("counter absent on an established store".to_owned()),
                    detected_at_ms: None,
                },
                "detected_at_ms = {raw:?}"
            );
        }
        let marker = HoldMarkerFields {
            detected_at_ms: Some(" 42 ".to_owned()),
            ..full_marker()
        };
        assert!(matches!(
            verdict_from_marker(Some(&marker)),
            HoldVerdict::Held {
                detected_at_ms: Some(42),
                ..
            }
        ));
    }

    // ── verdict_age and the observation's clocks ───────────────────────────

    /// Fresh below the stale threshold, stale from it on, however old: the
    /// boundary is inclusive on the older side, and no later class exists.
    #[test]
    fn verdict_age_classifies_each_side_of_the_stale_boundary() {
        let last = Instant::now();
        let ms = Duration::from_millis(1);
        for (age, want) in [
            (Duration::ZERO, VerdictAge::Fresh),
            (STALE_VERDICT_THRESHOLD - ms, VerdictAge::Fresh),
            (STALE_VERDICT_THRESHOLD, VerdictAge::Stale),
            (Duration::from_secs(120), VerdictAge::Stale),
            (Duration::from_secs(3_600), VerdictAge::Stale),
        ] {
            assert_eq!(verdict_age(last, last + age), want, "age {age:?}");
        }
    }

    /// An observation stamped after `now` counts as fresh, never as aged.
    #[test]
    fn verdict_age_observation_after_now_is_fresh() {
        let now = Instant::now();
        assert_eq!(
            verdict_age(now + STALE_VERDICT_THRESHOLD * 4, now),
            VerdictAge::Fresh
        );
    }

    #[test]
    fn hold_observation_deadlines_count_from_its_instant() {
        let observation = HoldObservation {
            verdict: HoldVerdict::Serving,
            at: Instant::now(),
        };
        assert_eq!(
            observation.stale_at(),
            observation.at + STALE_VERDICT_THRESHOLD
        );
        assert_eq!(observation.age_at(observation.at), VerdictAge::Fresh);
        assert_eq!(
            observation.age_at(observation.stale_at()),
            VerdictAge::Stale
        );
    }

    // ── poll_call_timeout (SDD SR-6.5 "Poll bounds") ───────────────────────

    #[test]
    fn poll_call_timeout_is_the_smaller_of_command_timeout_and_interval() {
        let interval = Duration::from_millis(2_000);
        assert_eq!(
            poll_call_timeout(500, interval),
            Duration::from_millis(500),
            "a command timeout below the interval bounds the call"
        );
        assert_eq!(
            poll_call_timeout(5_000, interval),
            interval,
            "an interval below the command timeout bounds the call"
        );
        assert_eq!(poll_call_timeout(2_000, interval), interval, "equal");
    }

    /// A command timeout of 0 is fred's "no timeout": the interval alone bounds
    /// the call, never a zero bound that would fail every read.
    #[test]
    fn poll_call_timeout_zero_command_timeout_leaves_the_interval() {
        let interval = Duration::from_millis(1_500);
        assert_eq!(poll_call_timeout(0, interval), interval);
    }

    // ── unhold_exit_jitter (SDD SR-6.4) ────────────────────────────────────

    /// Every draw is within 0 to the maximum, and the draws vary: a constant,
    /// zero included, would restart the whole fleet at once.
    #[test]
    fn unhold_exit_jitter_stays_within_zero_to_the_maximum_and_varies() {
        let draws: Vec<Duration> = (0..1_000).map(|_| unhold_exit_jitter()).collect();
        for jitter in &draws {
            assert!(
                *jitter <= UNHOLD_EXIT_JITTER_MAX,
                "{jitter:?} exceeds {UNHOLD_EXIT_JITTER_MAX:?}"
            );
        }
        let distinct: std::collections::BTreeSet<Duration> = draws.iter().copied().collect();
        assert!(
            distinct.len() >= 2,
            "1,000 draws over 5,001 milliseconds must not all be equal: {distinct:?}"
        );
        assert!(
            draws.iter().max().is_some_and(|max| *max > Duration::ZERO),
            "the largest of 1,000 draws must be above zero"
        );
    }
}
