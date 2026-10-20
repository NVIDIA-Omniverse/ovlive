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

//! Per-pass timing of the cached structural route's object-scaled passes,
//! for the measurement harnesses: where one structural edit's CPU goes, pass
//! by pass, on the planning, settling and repair sides.
//!
//! Each timed site wraps its pass in [`time`]. In a `test-support` build (and
//! in this crate's unit tests) that adds the pass's elapsed time and one call
//! to a per-thread total for its [`Pass`], which [`take`] returns and clears.
//! In every other build [`time`] is the closure call alone: no clock read, no
//! thread-local, nothing kept.
//!
//! - **Per thread.** A harness that times a route across an `.await` runs it
//!   on a current-thread runtime, so every pass lands on the thread that
//!   reads the totals.
//! - **Passes nest.** Each [`Pass`] names the pass it runs inside, if any, so
//!   a sum over every pass double-counts: a parent's own share is its total
//!   less its nested passes'.
//! - **Both apply modes.** The engine's passes are timed in a commit plan and
//!   in a repair's replay alike; a harness that wants them apart takes the
//!   totals between the two.
//!
//! Pure in the `domain` sense: a monotonic clock read and a thread-local, no
//! I/O and no async.

/// One timed pass. Its discriminant indexes the per-thread totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pass {
    /// `storage::structural_cache::CheckedOut::plan_state`: a copy of the
    /// held entry's state. The cached route plans in place and no longer
    /// makes one; a measurement harness timing that copy still reports it.
    PlanClone,
    /// The cached attempt's re-weigh of a held entry a repair advanced
    /// (`CacheEntry::weights_now`, O(1) from its plan indexes), before its
    /// first charge.
    ChargeReweigh,
    /// The cached attempt's plan, `CheckedOut::plan_in_place`:
    /// `domain::delta::apply_delta_bounded_indexed` on the held entry itself,
    /// under its journal. The engine's passes below run inside it.
    Plan,
    /// The engine's first `(parent_id, name)` probe of a delta, answered by a
    /// scan of every section. Inside [`Pass::Plan`] or [`Pass::RepairReplay`].
    NameScan,
    /// The engine's per-delta `(parent_id, name)` index build, at its second
    /// probe. Inside [`Pass::Plan`] or [`Pass::RepairReplay`].
    NameIndexBuild,
    /// The engine's per-delta adoption index build, at its first birth.
    /// Inside [`Pass::Plan`] or [`Pass::RepairReplay`].
    AdoptionIndexBuild,
    /// The engine's end-of-delta section-tree depth check: on a hooked plan
    /// or replay whose entry passed the full walk (`depth_ok`), the
    /// incremental check over the ids the delta listed
    /// (`domain::plan_index`); otherwise the full post-mutation walk. Inside
    /// [`Pass::Plan`] or [`Pass::RepairReplay`].
    DepthCheck,
    /// The engine's one purge per delta that erases a cascade's field and
    /// time-sample rows: on the compact form, a probe of each purged
    /// section's rows by interned name or a scan of every field row
    /// (`domain::types::PURGE_PROBE_RATIO`), then a scan of every time-sample
    /// row when there are any. Inside [`Pass::Plan`] or
    /// [`Pass::RepairReplay`].
    CascadePurge,
    /// `storage::structural::PreImageDigests::from_plan`: the commit's
    /// pre-image digests over the pre-plan state.
    PreImageDigests,
    /// The untouched external-row counts, when a touched row names an
    /// external hash: on the cached route in O(touched rows), from the plan
    /// indexes' counts through the journal's `PreImageView`; on a plain
    /// pre-plan state, a walk of every field and time-sample row. Inside
    /// [`Pass::PreImageDigests`].
    UntouchedExtCounts,
    /// A committed plan's successor step and normalization: on the cached
    /// route `CheckedOut::advance_committed`, in place over the lists the
    /// plan touched. The copy-based
    /// `storage::structural_cache_repair::successor_entry` — the test-support
    /// shadow check's oracle and the tests' — is timed here too, so a build
    /// with the shadow check on (the test-support default; the measurement
    /// harnesses turn it off) adds the oracle's time.
    SuccessorNormalize,
    /// `CheckedOut`'s settle: the entry's tables compacted and weighed, in
    /// O(1) from its plan indexes, before it goes back.
    SettleWeigh,
    /// `CheckedOut::put_back`'s drop of the predecessor state a given
    /// successor replaces: the copy-based path the tests and oracle take.
    /// The cached route settles a commit in place and drops no state.
    SettleDrop,
    /// The stream repair's replay of one record onto the held entry, its
    /// decode and its charge point ([`Pass::RepairCharge`]) included. The
    /// engine's passes run inside it.
    RepairReplay,
    /// The stream repair's per-record successor step, its children-list
    /// normalization included.
    RepairNormalizeChildren,
    /// The stream repair's per-round `normalize_state`, on an entry holding no
    /// plan indexes only: an indexed entry is normalized per record, over
    /// the lists each record touched.
    RepairNormalizeState,
    /// The stream repair's per-record charge point (RP2), between a record's
    /// decode and its apply: the record's growth estimate
    /// (`storage::structural_cache::estimate_replay_record_growth`) and its
    /// hold in the call's commit charge. Inside [`Pass::RepairReplay`]; a
    /// repair without a commit charge runs none.
    RepairCharge,
}

impl Pass {
    /// Every pass, in discriminant order.
    pub const ALL: [Pass; 17] = [
        Pass::PlanClone,
        Pass::ChargeReweigh,
        Pass::Plan,
        Pass::NameScan,
        Pass::NameIndexBuild,
        Pass::AdoptionIndexBuild,
        Pass::DepthCheck,
        Pass::CascadePurge,
        Pass::PreImageDigests,
        Pass::UntouchedExtCounts,
        Pass::SuccessorNormalize,
        Pass::SettleWeigh,
        Pass::SettleDrop,
        Pass::RepairReplay,
        Pass::RepairNormalizeChildren,
        Pass::RepairNormalizeState,
        Pass::RepairCharge,
    ];

    /// A stable snake-case name, for reports.
    pub fn label(self) -> &'static str {
        match self {
            Pass::PlanClone => "plan_clone",
            Pass::ChargeReweigh => "charge_reweigh",
            Pass::Plan => "plan",
            Pass::NameScan => "name_scan",
            Pass::NameIndexBuild => "name_index_build",
            Pass::AdoptionIndexBuild => "adoption_index_build",
            Pass::DepthCheck => "depth_check",
            Pass::CascadePurge => "cascade_purge",
            Pass::PreImageDigests => "pre_image_digests",
            Pass::UntouchedExtCounts => "untouched_ext_counts",
            Pass::SuccessorNormalize => "successor_normalize",
            Pass::SettleWeigh => "settle_weigh",
            Pass::SettleDrop => "settle_drop",
            Pass::RepairReplay => "repair_replay",
            Pass::RepairNormalizeChildren => "repair_normalize_children",
            Pass::RepairNormalizeState => "repair_normalize_state",
            Pass::RepairCharge => "repair_charge",
        }
    }
}

/// Run `f` as one call of `pass`: in a `test-support` build, its elapsed time
/// is added to this thread's total for `pass` (module docs); otherwise this
/// is `f()` alone.
#[inline(always)]
pub fn time<T>(pass: Pass, f: impl FnOnce() -> T) -> T {
    #[cfg(any(test, feature = "test-support"))]
    {
        let start = std::time::Instant::now();
        let out = f();
        totals::add(pass, start.elapsed());
        out
    }
    #[cfg(not(any(test, feature = "test-support")))]
    {
        let _ = pass;
        f()
    }
}

#[cfg(any(test, feature = "test-support"))]
pub use totals::{reset, take, PassTimings, PassTotal};

/// The per-thread totals behind [`time`]; compiled into `test-support`
/// builds and unit tests only.
#[cfg(any(test, feature = "test-support"))]
mod totals {
    use std::cell::RefCell;
    use std::time::Duration;

    use super::Pass;

    /// One pass's calls and their summed elapsed time on this thread.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct PassTotal {
        pub calls: u64,
        pub elapsed: Duration,
    }

    impl PassTotal {
        const ZERO: Self = Self {
            calls: 0,
            elapsed: Duration::ZERO,
        };
    }

    /// Every pass's total, as [`take`] returned them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PassTimings([PassTotal; Pass::ALL.len()]);

    impl PassTimings {
        /// `pass`'s total.
        pub fn get(&self, pass: Pass) -> PassTotal {
            self.0[pass as usize]
        }

        /// Every pass with its total, in [`Pass::ALL`] order.
        pub fn iter(&self) -> impl Iterator<Item = (Pass, PassTotal)> + '_ {
            Pass::ALL.iter().map(|&pass| (pass, self.get(pass)))
        }
    }

    thread_local! {
        static TOTALS: RefCell<[PassTotal; Pass::ALL.len()]> =
            const { RefCell::new([PassTotal::ZERO; Pass::ALL.len()]) };
    }

    pub(super) fn add(pass: Pass, elapsed: Duration) {
        TOTALS.with(|totals| {
            let total = &mut totals.borrow_mut()[pass as usize];
            total.calls = total.calls.saturating_add(1);
            total.elapsed = total.elapsed.saturating_add(elapsed);
        });
    }

    /// This thread's totals since the last [`take`] or [`reset`], which are
    /// then cleared.
    pub fn take() -> PassTimings {
        PassTimings(TOTALS.with(|totals| std::mem::take(&mut *totals.borrow_mut())))
    }

    /// Clear this thread's totals.
    pub fn reset() {
        let _ = take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_all_is_in_discriminant_order() {
        for (index, pass) in Pass::ALL.iter().enumerate() {
            assert_eq!(*pass as usize, index, "{}", pass.label());
        }
    }

    #[test]
    fn time_returns_the_closures_value_and_counts_one_call_per_run() {
        reset();
        assert_eq!(time(Pass::DepthCheck, || 7), 7);
        assert_eq!(time(Pass::DepthCheck, || 8), 8);
        time(Pass::PlanClone, || ());

        let timings = take();
        assert_eq!(timings.get(Pass::DepthCheck).calls, 2);
        assert_eq!(timings.get(Pass::PlanClone).calls, 1);
        assert_eq!(timings.get(Pass::NameScan), PassTotal::default());
    }

    #[test]
    fn take_clears_the_totals() {
        reset();
        time(Pass::SettleWeigh, || ());
        assert_eq!(take().get(Pass::SettleWeigh).calls, 1);
        assert_eq!(take().get(Pass::SettleWeigh), PassTotal::default());
    }

    #[test]
    fn time_adds_at_least_the_time_the_pass_took() {
        reset();
        let pause = std::time::Duration::from_millis(2);
        time(Pass::Plan, || std::thread::sleep(pause));
        assert!(take().get(Pass::Plan).elapsed >= pause);
    }
}
