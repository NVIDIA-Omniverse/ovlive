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

//! The pod memory plan: every pod-memory budget, derived at startup from the
//! pod memory limit `L` ([`super::pod_memory`]), and the term list that
//! proves they fit in it.
//!
//! # The two sides
//!
//! The pod's memory is split into the **line**, at most 80 % of `L`, and the
//! **margin**, the other 20 %. The line carries every term the service's
//! steady state and load can hold at once ([`PodMemoryPlan::line_terms`]); the
//! margin carries the allocator's slack and the transients the line does not
//! name ([`PodMemoryPlan::margin_terms`]). Each term is computed from the
//! configuration, in bytes, in integer arithmetic.
//!
//! # The derived budgets
//!
//! None has a setting. Each is a fixed share of `L`, declared once as a
//! [`Share`] and rounded down to a whole MiB (the subscription ceiling to a
//! whole subscription) for the value the service uses, chosen so the chart's
//! default 5 GiB limit gives the values the service shipped with:
//!
//! | Budget | Rule | Side | At 5 GiB |
//! |---|---|---|---|
//! | structural cache `C` | `L / 20` | line (its resident budget, `C × (1 + φ)`, and its audit) | 256 MiB |
//! | uploads `B` | `max(⌈floor⌉, 3L / 40)`, the floor `upload_budget_floor_bytes` | line | 384 MiB |
//! | subscriptions `S` | `0.8` per MiB of `L` | line (`S × m`) | 4,096 |
//! | catch-up `T` | the residual, below | line (`T × (1 + P_f)`) | 744 MiB |
//! | retirement `R_ref` | `L / 500` | margin | 10 MiB |
//! | external-value mirror `M_ext` | `max(⌈floor⌉, L / 40)`, the floor one maximum-size value's mirror charge (`checkpoint::ext_mirror::item_charge_kib`), only while the mirror is wired | margin | 128 MiB |
//!
//! `M_ext` is present exactly when checkpointing is on with an adapter
//! ([`Config::checkpoint_adapter_wired`]), and then `main.rs` builds the
//! mirror with it as its byte budget; otherwise the plan has no such term and
//! the mirror is not built. Its floor makes every value under the upload size
//! cap fit the budget at every limit that closes.
//!
//! `T` is `⌊s × (L − floor)⌋` MiB for the largest step `s` of 0.01, at most
//! 1, at which the whole line, charged at exact shares (below), fits within
//! 80 % of `L`: `T × (1 + P_f)` included, and the Valkey connections'
//! buffers taken on a 10-primary cluster (the largest topology figure). The
//! step is computed, not searched: in hundredths it is
//! min(100, ⌊100 × H ÷ ((L − floor) × (1 + P_f))⌋), `H` being the exact room
//! the line's other terms leave within 80 % of `L`.
//!
//! # Exact shares
//!
//! Every term that depends on `L` has two values. Its *built* value is what
//! the service uses and the plan reports (the line and margin terms, their
//! totals, the headroom, the startup line and the gauges): the share rounded
//! down, as above. Its *exact* value is the same rule with each rounding down
//! of a share of `L` removed, and each rounding up of the term replaced by
//! one byte more, so it is never below the built value:
//!
//! | Term | Exact value |
//! |---|---|
//! | established subscriptions, `S × m` | `0.8 × L ÷ 1 MiB × m + 1 B` |
//! | structural cache, resident | `L ÷ 20 × (1 + φ) + 1 B` |
//! | structural cache audit | `2 × min(L ÷ 80, u32::MAX) +` the byte ceiling |
//! | upload budget `B` | `max(⌈floor⌉ MiB, 3L ÷ 40)` |
//! | catch-up `T × (1 + P_f)` | `s × (L − floor) × (1 + P_f)` |
//! | retirement `R_ref` | `L ÷ 500` |
//! | external-value mirror `M_ext` | `max(⌈floor⌉ MiB, L ÷ 40)` |
//! | the allowances | the line `4L ÷ 5`, the margin `L ÷ 5` |
//!
//! Every other term depends on settings and recorded constants only, and its
//! exact value is its built one. Exact values are whole numbers of
//! 1/[`EXACT_DEN`] byte, in `u128`, so no rounding enters them.
//!
//! Whether `L` closes, and `T`'s step, are decided on the exact values. As
//! built ≤ exact ≤ allowance, the built line fits within ⌊80 % of `L`⌋ on
//! every topology and the built margin within the rest. Rounded shares are a
//! sawtooth in `L`: a budget's whole-MiB step can cost more than the 0.8 MiB
//! of room one more MiB of limit adds, so deciding on them would let a limit
//! close while a larger one refuses, and `T` fall as `L` grows. Exact shares
//! are straight lines in `L`, so neither happens (*Adding a budget*).
//!
//! # Settings that stay
//!
//! The connection pools, the stream readers, the catch-up slots, the upload
//! waiters, the upload size cap, the atomic-read byte ceiling and `P_f` stay
//! settings: they are connection, CPU or contract counts, not memory. Their
//! memory is charged here, so raising one lowers `T`, or, past what `L` can
//! carry, refuses startup ([`PodMemoryPlan::derive`]); none can take the pod
//! past `L`.
//!
//! # Refusal
//!
//! The plan closes at `L` when, at exact shares, the margin's named terms fit
//! within 20 % of `L` and the line fits within 80 % with `T` at a step of at
//! least 0.01 and at least 1 MiB. The minimum limit is the smallest whole MiB
//! at which it closes. Closure holds at every limit above the first at which
//! it holds (*Adding a budget*), so a doubling and a bisection over whole MiB
//! find that minimum exactly, and every limit from it up closes.
//!
//! [`PodMemoryPlan::derive`] refuses exactly when `L` is below the minimum,
//! or no limit up to 1 PiB closes, before it evaluates the plan at `L`.
//! Below the minimum, the error names the limit and its source, the minimum,
//! the shortfall, and the terms an operator setting sizes on the side that
//! sets the minimum, the side that does not close one MiB below it: the
//! line's largest four; the margin's (the checkpoint base bootstraps and
//! the mirror's `M_ext`, with the settings that switch them on as well as
//! the ones that size them); or both, when both sides set it. A term of a
//! side that closes there does not lower the minimum, so it is not named.
//! When the margin is named, its remedy puts raising the limit first, then
//! lowering `LIVE_EDIT_MAX_LAYER_SIZE` or, while the base bootstraps are
//! charged, turning off the tail, with its cost; it never offers turning off
//! checkpointing or the adapter, which removes backups, and says so.
//! When no limit fits, it says so and names what to lower instead:
//! `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`, with the largest value that
//! can fit, when `T × (1 + P_f)` outgrows every limit (above 61 at the
//! chart's other settings), else the terms of the side, or sides, that do
//! not close at 1 PiB.
//!
//! At or above the minimum the plan closes, by the exact-share contract. If
//! it ever did not, a term would be breaking that contract: `derive` then
//! returns an error saying so, naming the contract and this file, a defect
//! of the service rather than of its configuration, and the service does
//! not start.
//!
//! # Adding a budget
//!
//! A further pod-memory budget is one more term of the plan and nothing else:
//! its share declared once with [`Share::new`] (numerator, denominator, base
//! `L` or `L − floor`, unit), its built value taken in `budgets_at`, with its
//! fixed minimum where it has one (`B`'s, from the upload size cap; `M_ext`'s,
//! from the same cap); its term, with its built and exact values, on its
//! funding side; and an accessor on [`PodMemoryPlan`] that its consumer
//! reads. No setting, no chart key, no [`Config`](super::Config) field. The
//! line's terms set `T`'s step, so a line term lowers `T`; a margin term only
//! has to fit the 20 %.
//!
//! - **A line term** goes in `line_terms_at`. If its exact value is its
//!   share times a factor (as `S × m` and the cache's `C × (1 + φ)` are),
//!   that product is declared as a share of its own, with [`Share::new`] on
//!   the budget's share's numerator and denominator times the factor's, so
//!   it too is a whole number of 1/[`EXACT_DEN`] byte.
//! - **A margin term that is a share of `L`** is an entry of
//!   [`MARGIN_SHARES`] and nothing else: `margin_terms_at` charges every
//!   entry present for the inputs and no other share of `L`. The entry
//!   carries the term's presence over [`PlanInputs`] (`R_ref` always;
//!   `M_ext` only while the mirror is wired), its fixed minimum in the
//!   share's units (`max(minimum, share)` built and exact alike, so C1 holds
//!   by construction), and the settings the refusal names it with. A margin
//!   term that does not depend on `L` goes in `margin_terms_at` directly, its
//!   exact value its built one.
//!
//! What is automatic: a [`Share`] can be made only through [`Share::new`],
//! whose exact coefficient is an inline constant, so a share whose
//! denominator, in its unit, does not divide [`EXACT_DEN`] does not compile;
//! C3's sum is a compile-time assertion over [`MARGIN_SHARES`], so a margin
//! share that breaks C3 does not compile either. What the author must still
//! do: give a term that rounds up one unit more in its exact value (C1), and
//! keep a line term's shape within C2, which no compile-time check sees; the
//! plan's tests check both term by term.
//!
//! Every term meets the **exact-share contract**:
//!
//! - **C1, dominance.** Its exact value is at least its built value at every
//!   `L`. Deriving both from one [`Share`] gives this for the share itself; a
//!   term that rounds up adds one unit (a byte) to its exact value.
//! - **C2, line shape.** For `L` above the floor, a line term's exact value
//!   per byte of `L − floor` never increases as `L` grows. A non-negative
//!   constant, a non-negative multiple of `L` or of `L − floor`, and a max or
//!   min of such pieces all meet it; a share of `L − x` with `x` above the
//!   floor, or anything that grows faster than its share, does not.
//! - **C3, margin growth.** The margin terms' coefficients on `L` sum to at
//!   most 1/5 (today `R_ref`'s 1/500 and `M_ext`'s 1/40, counted whether or
//!   not `M_ext` is present, which only over-counts). A fixed minimum does not
//!   change a coefficient: where it binds the term is constant, and above it
//!   the term grows at its share's slope.
//!
//! Why it is enough. With `F` the floor, itself a line term, and `e_i` the
//! exact line terms other than the floor and `T`, the room `T` has is
//! `H(L) = 4L/5 − F − Σ e_i(L) = 4(L − F)/5 − [F/5 + Σ e_i(L)]`. By C2 each
//! `e_i(L) ÷ (L − F)`, like `F/5 ÷ (L − F)`, never increases, so
//! `H(L) ÷ (L − F)` never decreases. A step `s` fits iff
//! `s × (1 + P_f) ≤ H(L) ÷ (L − F)`, so the largest step that fits never
//! decreases as `L` grows: `T` = `⌊s × (L − F)⌋` MiB never decreases, and
//! once the smallest step fits with `T` at least 1 MiB, it does at every
//! larger `L`. On the margin, `L/5 −` the margin's terms grows with a slope
//! of at least `1/5 −` the sum of their coefficients, at least 0 by C3, so a
//! margin that fits at `L` fits at every larger `L`. `T`'s own term is
//! charged exactly for the same reason: rounded, its smallest step jumps by
//! `1 + P_f` MiB every 100 MiB of limit, and a limit above the first that
//! closes could still refuse.
//!
//! The guards: at compile time, every share's denominator ([`Share::new`])
//! and C3 over every margin share ([`MARGIN_SHARES`]); the plan's tests,
//! which check C1–C3 term by term at every MiB of the swept configurations;
//! and, at runtime, the defect error (*Refusal*).
//!
//! # The recorded constants
//!
//! Compiled at their records' full precision, in bytes:
//!
//! - the steady-state floor, 155 MiB (dev pods after load; SDD §21.2);
//! - `m`, one established subscription, 49.8 KiB, and `e`, one queued entry
//!   per subscription, 167.1 B
//!   (`docs-internal/validation/stream-mux-hardening-2026-09-30/report.md`);
//! - `X`, one fenced read stage's fan-out window transient, 149,987,424 B,
//!   and the checkpoint reference read, 128.6796875 MiB on short-id reply
//!   bytes, both scaled to real-id bytes by the byte ceiling, and both scaled
//!   up again when the ceiling is raised past its default
//!   (`docs-internal/validation/watch-window-script-b-gvf-20261010/derivation.json`,
//!   `docs-internal/validation/fenced-snapshot-reads-20261002/raw/task10-results/results.json`);
//! - the cold-start ratios `r` and `b`
//!   (`storage::structural_cache_cold_start`), the catch-up slot's 42 bytes per
//!   ceiling byte (`storage::memory_budget`), the allocator's 55 MiB of slack
//!   and the 20 % margin.

use std::sync::OnceLock;

use super::pod_memory::{PodMemoryLimit, PodMemoryLimitSource, ReferenceReason};
use super::Config;
use crate::storage::connection_buffers::{at_rest_bound_bytes, PoolSizes, Topology};

const MIB: u64 = 1024 * 1024;

/// The steady-state floor: what a replica holds after load with no
/// subscription, upload or catch-up in flight, 155 MiB (SDD §21.2).
pub const POD_MEMORY_FLOOR_BYTES: u64 = 155 * MIB;

/// `m`, the memory one established subscription holds, in tenths of a byte:
/// 49.8 KiB = 50,995.2 B (SDD §16).
pub const SUBSCRIPTION_BYTES_DECI: u64 = 509_952;

/// `e`, the memory one queued stream entry holds per subscription, in tenths
/// of a byte: 167.1 B, the allocation slope of 200 stalled subscriptions
/// (SDD §16).
pub const QUEUED_ENTRY_BYTES_DECI: u64 = 1_671;

/// The stream entries one catch-up slot's catch-up queues between its stream
/// registration and its first poll, in hundredths: the design rate of 275
/// commits/s × `t_post`, 0.73 s (SDD §16), each costing `e`.
pub const PRE_POLL_ENTRIES_PER_SLOT_CENTI: u64 = 275 * 73;

/// The payload the catch-ups before their first poll share, 1 MiB (SDD §16).
pub const PRE_POLL_SHARED_PAYLOAD_BYTES: u64 = MIB;

/// Memory per catch-up slot per byte of the atomic-read byte ceiling: a
/// snapshot plus one build (`storage::memory_budget`), 42.
pub const CATCHUP_SLOT_BYTES_PER_CEILING_BYTE: u64 =
    crate::storage::memory_budget::SNAPSHOT_MEMORY_FACTOR
        + crate::storage::memory_budget::BUILD_MEMORY_FACTOR;

/// `X`: one fenced read stage's fan-out window transient, which a flight's
/// grant does not reserve, at the default byte ceiling: 149,987,424 B, the
/// largest the window-script record carries (`x_watch_pool_build`).
pub const WINDOW_TRANSIENT_BYTES: u64 = 149_987_424;

/// The checkpoint fenced read of the reference layer (4× the item ceiling) on
/// short-id reply bytes, as the Task 10 record measured it: 128.6796875 MiB.
pub const REFERENCE_READ_SHORT_ID_BYTES: u64 = 134_930_432;

/// The byte ceiling the Task 10 record measured on, with short ids: 449,938.
pub const RECORD_SHORT_ID_BYTE_CEILING: u64 = 449_938;

/// The byte ceiling `X` and the reference read are recorded at, the default
/// `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES`: 604,619.
pub const RECORD_BYTE_CEILING: u64 = super::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT;

/// The reference read on real-id reply bytes at [`RECORD_BYTE_CEILING`],
/// rounded up: 181,317,211 B.
pub const REFERENCE_READ_BYTES: u64 = (REFERENCE_READ_SHORT_ID_BYTES as u128
    * RECORD_BYTE_CEILING as u128)
    .div_ceil(RECORD_SHORT_ID_BYTE_CEILING as u128) as u64;

/// The unread content one queued upload waiter can hold: one HTTP/2 stream
/// window at hyper's server default, 1 MiB (SDD §5.4).
pub const UPLOAD_WAITER_WINDOW_BYTES: u64 = MIB;

/// The allocator's slack, a margin term: 55 MiB.
pub const ALLOCATOR_SLACK_BYTES: u64 = 55 * MIB;

/// The step of `T`'s share of `L − floor`, in hundredths: 0.01. `T`'s step
/// is a whole number of these, from 0.01 to 1.
pub const CATCHUP_BUDGET_STEP_CENTI: u64 = 1;

/// The subscription ceiling at the reference limit: 4,096, 0.8 per MiB of
/// 5,120 MiB.
pub const SUBSCRIPTION_CEILING_AT_REFERENCE_LIMIT: usize = 4_096;

const _: () = assert!(
    SUBSCRIPTION_CEILING_AT_REFERENCE_LIMIT as u64
        == super::pod_memory::REFERENCE_POD_MEMORY_LIMIT_MIB * 4 / 5
);

/// The settings that size the connection pools' term: every pool whose
/// connections' buffers it charges.
const CONNECTION_POOL_SETTINGS: &str = "LIVE_EDIT_VALKEY_POOL_SIZE, \
    LIVE_EDIT_MAINTENANCE_POOL_SIZE, LIVE_EDIT_SNAPSHOT_POOL_SIZE, \
    LIVE_EDIT_SUBSCRIBER_POOL_SIZE";

/// The largest limit [`PodMemoryPlan::minimum_limit_mib`] searches: 1 PiB.
const SEARCH_CEILING_MIB: u64 = 1 << 30;

/// The plan-wide denominator of the exact values: an exact value is a whole
/// number of 1/`EXACT_DEN` byte (module docs, *Exact shares*). 2^21 × 10^4 =
/// 20,971,520,000: every share, in its unit, divides it, and so does every
/// term's share ([`Share::new`] fails to compile otherwise). At the 1 PiB search ceiling
/// with `P_f` at `u32::MAX` the largest exact value is about 2^117, inside
/// `u128`; `T`'s cost per step is multiplied out checked, so no limit or
/// `P_f` overflows it.
pub(crate) const EXACT_DEN: u128 = (1 << 21) * 10_000;

/// What a [`Share`] is a share of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShareBase {
    /// The limit `L`.
    Limit,
    /// `L − floor`, the limit less the steady-state floor
    /// ([`POD_MEMORY_FLOOR_BYTES`]); 0 at or below the floor.
    AboveFloor,
}

pub(crate) use share::Share;

/// [`Share`], in a module of its own so that its fields are private to it: a
/// share can be made only through [`Share::new`], which works out its exact
/// coefficient in an inline constant, so a share whose denominator does not
/// divide [`EXACT_DEN`] fails the build wherever it is declared or used
/// (declared in a constant, as every share is, `cargo check` reports it too;
/// made inside a function, only the build does), and no share reaches the
/// plan unchecked.
mod share {
    use super::{ShareBase, EXACT_DEN, POD_MEMORY_FLOOR_BYTES};

    /// `n ÷ d`, panicking when it is not a whole number. Called only inside
    /// [`Share::new`]'s inline constant, where that panic is a compile error.
    const fn whole_ratio(n: u128, d: u128) -> u128 {
        assert!(
            n.is_multiple_of(d),
            "a share's denominator, in its unit, must divide EXACT_DEN"
        );
        n / d
    }

    /// A share of the limit, declared once: its built value, what the
    /// service uses, and its exact value, what closure is decided on, both
    /// come from this declaration, so the exact value is never below the
    /// built one (module docs, *Adding a budget*, C1).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Share {
        /// The share's numerator.
        num: u64,
        /// The share's denominator.
        den: u64,
        /// What it is a share of.
        base: ShareBase,
        /// The bytes one unit of the built value stands for: the built value
        /// is a whole number of units, rounded down.
        unit: u64,
        /// The exact value per byte of the base, in units, scaled by
        /// [`EXACT_DEN`]: `num × EXACT_DEN ÷ (den × unit)`, a whole number.
        per_base_byte: u128,
    }

    impl Share {
        /// `NUM / DEN` of `base`, its built value counted in whole units of
        /// `UNIT` bytes. Its exact coefficient is computed in an inline
        /// constant: a `DEN × UNIT` that does not divide `NUM × EXACT_DEN`
        /// is a compile error, never a startup panic.
        pub(crate) const fn new<const NUM: u64, const DEN: u64, const UNIT: u64>(
            base: ShareBase,
        ) -> Self {
            Self {
                num: NUM,
                den: DEN,
                base,
                unit: UNIT,
                per_base_byte: const { whole_ratio(NUM as u128 * EXACT_DEN, DEN as u128 * UNIT as u128) },
            }
        }

        /// The share's numerator.
        pub(crate) const fn num(&self) -> u64 {
            self.num
        }

        /// The share's denominator.
        pub(crate) const fn den(&self) -> u64 {
            self.den
        }

        /// What it is a share of.
        pub(crate) const fn base(&self) -> ShareBase {
            self.base
        }

        /// The bytes one unit of its built value stands for.
        pub(crate) const fn unit(&self) -> u64 {
            self.unit
        }

        /// The bytes the share is taken of at `limit`.
        pub(crate) const fn base_bytes(&self, limit: u64) -> u64 {
            match self.base {
                ShareBase::Limit => limit,
                ShareBase::AboveFloor => limit.saturating_sub(POD_MEMORY_FLOOR_BYTES),
            }
        }

        /// `k` times the share: `T`'s share at a step of `k` hundredths is
        /// [`super::CATCHUP_HUNDREDTH`] times `k`. A whole multiple of a
        /// checked coefficient, so whole too.
        pub(crate) const fn times(self, k: u64) -> Self {
            Self {
                num: self.num * k,
                per_base_byte: self.per_base_byte * k as u128,
                ..self
            }
        }

        /// The exact value per byte of the base, in units, scaled by
        /// [`EXACT_DEN`].
        pub(crate) const fn exact_units_per_base_byte(&self) -> u128 {
            self.per_base_byte
        }

        /// The built value at `limit`: the share in whole units, rounded
        /// down, and at least `minimum_units`.
        pub(crate) fn built_units(&self, limit: u64, minimum_units: u64) -> u64 {
            let units = u128::from(self.base_bytes(limit)) * u128::from(self.num)
                / (u128::from(self.den) * u128::from(self.unit));
            minimum_units.max(u64::try_from(units).unwrap_or(u64::MAX))
        }

        /// The exact value at `limit`, in units scaled by [`EXACT_DEN`]: the
        /// share unrounded, and at least `minimum_units`.
        pub(crate) fn exact_units_scaled(&self, limit: u64, minimum_units: u64) -> u128 {
            (u128::from(self.base_bytes(limit)) * self.per_base_byte)
                .max(u128::from(minimum_units) * EXACT_DEN)
        }
    }
}

/// `S`: 0.8 subscription per MiB of `L`.
pub(crate) const SUBSCRIPTIONS_SHARE: Share = Share::new::<4, 5, MIB>(ShareBase::Limit);

/// `C`: `L / 20`, in MiB.
pub(crate) const STRUCTURAL_CACHE_SHARE: Share = Share::new::<1, 20, MIB>(ShareBase::Limit);

/// `B`: `3L / 40`, in MiB, at least the upload floor.
pub(crate) const UPLOAD_SHARE: Share = Share::new::<3, 40, MIB>(ShareBase::Limit);

/// `R_ref`: `L / 500`, in MiB, a margin term ([`MARGIN_SHARES`]).
pub(crate) const RETIREMENT_SHARE: Share = Share::new::<1, 500, MIB>(ShareBase::Limit);

/// `M_ext`: `L / 40`, in MiB, at least [`ext_mirror_floor_mib`], a margin term
/// present only while the mirror is wired ([`MARGIN_SHARES`]).
pub(crate) const EXT_MIRROR_SHARE: Share = Share::new::<1, 40, MIB>(ShareBase::Limit);

/// One hundredth of `L − floor`, in MiB: `T` at a step of `k` hundredths is
/// this share times `k`.
pub(crate) const CATCHUP_HUNDREDTH: Share = Share::new::<1, 100, MIB>(ShareBase::AboveFloor);

/// The line's allowance: 80 % of `L`, in bytes.
pub(crate) const LINE_ALLOWANCE: Share = Share::new::<4, 5, 1>(ShareBase::Limit);

/// The margin's allowance, exactly: 20 % of `L`, in bytes. (Built, the
/// margin is what the line's ⌊80 %⌋ leaves, never less.)
pub(crate) const MARGIN_ALLOWANCE: Share = Share::new::<1, 5, 1>(ShareBase::Limit);

/// The exact `S × m`, as a share of `L` in bytes: [`SUBSCRIPTIONS_SHARE`]
/// times `m`'s 0.1 B. Exact only; the built term is `S × m` rounded up.
const SUBSCRIPTIONS_TERM_SHARE: Share = Share::new::<
    { SUBSCRIPTIONS_SHARE.num() * SUBSCRIPTION_BYTES_DECI },
    { SUBSCRIPTIONS_SHARE.den() * SUBSCRIPTIONS_SHARE.unit() * 10 },
    1,
>(SUBSCRIPTIONS_SHARE.base());

/// The exact structural cache resident budget, `C × (1 + φ)`, as a share of
/// `L` in bytes: [`STRUCTURAL_CACHE_SHARE`] times `1 + φ`. Exact only.
const STRUCTURAL_CACHE_RESIDENT_SHARE: Share = Share::new::<
    {
        STRUCTURAL_CACHE_SHARE.num()
            * (100 + crate::storage::structural_cache::PLAN_INDEX_BUDGET_CENTI)
    },
    { STRUCTURAL_CACHE_SHARE.den() * 100 },
    1,
>(STRUCTURAL_CACHE_SHARE.base());

/// The exact quarter of `C` the structural cache audit's entry bound takes
/// (`max_entry_bytes_for`), as a share of `L` in bytes. Exact only.
const STRUCTURAL_CACHE_QUARTER_SHARE: Share =
    Share::new::<{ STRUCTURAL_CACHE_SHARE.num() }, { STRUCTURAL_CACHE_SHARE.den() * 4 }, 1>(
        STRUCTURAL_CACHE_SHARE.base(),
    );

/// A margin term that is a share of `L`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MarginShare {
    /// The term's name in the plan.
    pub name: &'static str,
    /// Its share, in whole units when built.
    pub share: Share,
    /// Whether `inputs` charge the term at all. An absent term is not in the
    /// margin, and its budget does not exist.
    pub present: fn(&PlanInputs) -> bool,
    /// Its fixed minimum for `inputs`, in the share's units: its built value
    /// is `max(minimum, ⌊share⌋)` and its exact value `max(minimum, share)`,
    /// so the exact value is never below the built one (C1).
    pub minimum_units: fn(&PlanInputs) -> u64,
    /// The settings the refusal names the term with, comma-separated: those
    /// that switch it on and those that size it. `None` when no setting does.
    pub settings: Option<&'static str>,
}

impl MarginShare {
    /// Its built value at `limit`, in units, or `None` when `inputs` do not
    /// charge it.
    fn built_units(&self, limit: u64, inputs: &PlanInputs) -> Option<u64> {
        (self.present)(inputs).then(|| self.share.built_units(limit, (self.minimum_units)(inputs)))
    }

    /// Its term at `limit`, built and exact, or `None` when `inputs` do not
    /// charge it.
    fn term_at(&self, limit: u64, inputs: &PlanInputs) -> Option<ExactTerm> {
        let units = self.built_units(limit, inputs)?;
        let unit = self.share.unit();
        let bytes = units.saturating_mul(unit);
        let term = match self.settings {
            Some(settings) => PlanTerm::set_by(self.name, bytes, settings),
            None => PlanTerm::fixed(self.name, bytes),
        };
        let exact = self
            .share
            .exact_units_scaled(limit, (self.minimum_units)(inputs))
            * u128::from(unit);
        Some(ExactTerm::new(term, exact))
    }
}

/// `R_ref`'s entry: always charged, no minimum, no setting.
const RETIREMENT_MARGIN: MarginShare = MarginShare {
    name: "retirement allowance",
    share: RETIREMENT_SHARE,
    present: always_present,
    minimum_units: no_minimum,
    settings: None,
};

/// `M_ext`'s entry: charged only while the mirror is wired, at least one
/// maximum-size value's charge.
const EXT_MIRROR_MARGIN: MarginShare = MarginShare {
    name: "backup copies of external values",
    share: EXT_MIRROR_SHARE,
    present: ext_mirror_present,
    minimum_units: ext_mirror_floor_mib,
    settings: Some(EXT_MIRROR_SETTINGS),
};

fn always_present(_: &PlanInputs) -> bool {
    true
}

fn no_minimum(_: &PlanInputs) -> u64 {
    0
}

/// Whether `M_ext` is charged: the mirror is wired ([`PlanInputs::ext_mirror`]).
fn ext_mirror_present(inputs: &PlanInputs) -> bool {
    inputs.ext_mirror
}

/// `M_ext`'s floor, in MiB: one maximum-size value's mirror charge
/// ([`crate::checkpoint::ext_mirror::item_charge_kib`] of
/// `LIVE_EDIT_MAX_LAYER_SIZE`), rounded up to a whole MiB, so every value
/// under the upload size cap fits the mirror's budget: 101 MiB at the default
/// cap. It depends on the cap alone, not on `L`, so closure stays monotone in
/// `L`.
pub(crate) fn ext_mirror_floor_mib(inputs: &PlanInputs) -> u64 {
    crate::checkpoint::ext_mirror::item_charge_kib(inputs.max_layer_size).div_ceil(1024)
}

/// The settings that switch the base bootstraps' margin term on and size it.
const BASE_BOOTSTRAP_SETTINGS: &str = "LIVE_EDIT_CHECKPOINT_TAIL_ENABLED, \
    LIVE_EDIT_CHECKPOINT_ENABLED, LIVE_EDIT_CHECKPOINT_ADAPTER, LIVE_EDIT_MAX_LAYER_SIZE";

/// The settings that switch `M_ext` on and size its floor.
const EXT_MIRROR_SETTINGS: &str =
    "LIVE_EDIT_CHECKPOINT_ENABLED, LIVE_EDIT_CHECKPOINT_ADAPTER, LIVE_EDIT_MAX_LAYER_SIZE";

/// Every margin term that is a share of `L`: `margin_terms_at` charges each
/// of them that is present, and nothing else of `L`, and C3's compile-time
/// sum below covers exactly these, so a margin share cannot be charged
/// without entering it.
pub(crate) const MARGIN_SHARES: [MarginShare; 2] = [RETIREMENT_MARGIN, EXT_MIRROR_MARGIN];

/// C3: the margin's shares' coefficients on `L` sum to at most the margin's
/// 1/5, each counted whether or not it is present for some inputs. And `T`'s
/// step is one hundredth of `L − floor` ([`CATCHUP_HUNDREDTH`]), so its step
/// in hundredths is a count of that share.
const _: () = {
    let mut margin = 0;
    let mut i = 0;
    while i < MARGIN_SHARES.len() {
        let share = MARGIN_SHARES[i].share;
        margin += share.exact_units_per_base_byte() * share.unit() as u128;
        i += 1;
    }
    assert!(
        margin <= MARGIN_ALLOWANCE.exact_units_per_base_byte(),
        "C3: the margin's shares of L sum to more than 1/5"
    );
    assert!(CATCHUP_BUDGET_STEP_CENTI == 1);
};

/// One term of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTerm {
    /// The term in words.
    pub name: &'static str,
    /// Its memory, in bytes.
    pub bytes: u64,
    /// The setting an operator sizes it with, when one does; comma-separated
    /// when several do: the connection pools' settings, and for the
    /// checkpoint margin terms the settings that switch them on as well as
    /// the ones that size them.
    pub setting: Option<&'static str>,
    /// Whether an operator setting sizes it (a count or size setting), so a
    /// refusal can name it; `false` for the recorded constants and for the
    /// budgets derived from `L`.
    pub operator_sized: bool,
}

impl PlanTerm {
    const fn fixed(name: &'static str, bytes: u64) -> Self {
        Self {
            name,
            bytes,
            setting: None,
            operator_sized: false,
        }
    }

    const fn set_by(name: &'static str, bytes: u64, setting: &'static str) -> Self {
        Self {
            name,
            bytes,
            setting: Some(setting),
            operator_sized: true,
        }
    }
}

/// The configuration the plan's terms are computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanInputs {
    /// The connection pools, with "auto" sizes resolved as `connect` does.
    pub pools: PoolSizes,
    /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`.
    pub catchup_slots: u64,
    /// `P_f`, `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`.
    pub fenced_read_concurrency: u64,
    /// `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES`.
    pub byte_ceiling: u64,
    /// `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`.
    pub upload_waiters: u64,
    /// `LIVE_EDIT_MAX_LAYER_SIZE`.
    pub max_layer_size: u64,
    /// Whether the checkpoint base bootstrap can run: checkpointing enabled,
    /// an adapter wired and the tail enabled
    /// ([`Config::checkpoint_base_bootstrap_enabled`]).
    pub base_bootstrap: bool,
    /// Whether the external-value mirror is wired: checkpointing enabled and
    /// an adapter wired ([`Config::checkpoint_adapter_wired`]), with or
    /// without the tail. `M_ext` is charged exactly then.
    pub ext_mirror: bool,
}

impl PlanInputs {
    /// The inputs of `config`.
    pub fn from_config(config: &Config) -> Self {
        Self {
            pools: PoolSizes::from_config(config),
            catchup_slots: config.subscriber_max_concurrent_catchups as u64,
            fenced_read_concurrency: u64::from(config.catchup_fenced_read_concurrency),
            byte_ceiling: config.snapshot_atomic_read_max_bytes,
            upload_waiters: config.upload_admission_max_waiters as u64,
            max_layer_size: u64::try_from(config.max_layer_size).unwrap_or(u64::MAX),
            base_bootstrap: config.checkpoint_base_bootstrap_enabled(),
            ext_mirror: config.checkpoint_adapter_wired(),
        }
    }

    /// The Helm chart's configuration: pools 64 / 64 / 1 / 6 and 256 stream
    /// readers, 8 catch-up slots, `P_f` 1, 64 upload waiters, 100 MiB uploads,
    /// the default byte ceiling, no checkpoint adapter. The inputs of
    /// [`PodMemoryPlan::reference`].
    pub const fn reference() -> Self {
        Self {
            pools: PoolSizes {
                data: 64,
                txn: 64,
                maintenance: 1,
                snapshot: 6,
                readers: 256,
            },
            catchup_slots: 8,
            fenced_read_concurrency: 1,
            byte_ceiling: RECORD_BYTE_CEILING,
            upload_waiters: 64,
            max_layer_size: crate::grpc::upload_admission::MAX_LAYER_SIZE_DEFAULT_BYTES,
            base_bootstrap: false,
            ext_mirror: false,
        }
    }
}

/// The budgets at a limit, built, with the limit and `B`'s floor their
/// exact values are taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Budgets {
    /// `S`.
    max_subscriptions: u64,
    /// `C`, MiB.
    structural_cache_mib: u64,
    /// `B`, MiB.
    upload_mib: u64,
    /// `R_ref`, MiB.
    retirement_mib: u64,
    /// `M_ext`, MiB, `None` when the mirror is not wired.
    ext_mirror_mib: Option<u64>,
    /// The limit they were taken at, bytes: the terms' exact values are
    /// taken from their shares at it.
    limit: u64,
    /// `B`'s fixed minimum, MiB: its built and exact values both take it.
    upload_floor_mib: u64,
}

/// The budgets at a limit of `limit` bytes for `inputs` (module docs, *The
/// derived budgets*), each built from its one [`Share`].
fn budgets_at(limit: u64, inputs: &PlanInputs) -> Budgets {
    let upload_floor_mib =
        crate::grpc::upload_admission::upload_budget_floor_bytes(inputs.max_layer_size)
            .div_ceil(MIB);
    Budgets {
        max_subscriptions: SUBSCRIPTIONS_SHARE.built_units(limit, 0),
        structural_cache_mib: STRUCTURAL_CACHE_SHARE.built_units(limit, 0),
        upload_mib: UPLOAD_SHARE.built_units(limit, upload_floor_mib),
        retirement_mib: RETIREMENT_SHARE.built_units(limit, 0),
        ext_mirror_mib: EXT_MIRROR_MARGIN.built_units(limit, inputs),
        limit,
        upload_floor_mib,
    }
}

/// A term of the plan with its exact value: its built bytes are what the
/// service uses and the plan reports; its exact value, in 1/[`EXACT_DEN`]
/// byte, is what closure and `T`'s step are decided on (module docs, *Exact
/// shares*).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactTerm {
    /// The term, at its built value.
    pub term: PlanTerm,
    /// Its exact value, in 1/[`EXACT_DEN`] byte.
    pub exact: u128,
}

impl ExactTerm {
    /// A term whose exact value is `exact`.
    fn new(term: PlanTerm, exact: u128) -> Self {
        Self { term, exact }
    }

    /// A term that does not depend on `L`: its exact value is its built one.
    fn as_built(term: PlanTerm) -> Self {
        let exact = u128::from(term.bytes) * EXACT_DEN;
        Self { term, exact }
    }
}

fn exact_sum(terms: &[ExactTerm]) -> u128 {
    terms
        .iter()
        .fold(0u128, |total, t| total.saturating_add(t.exact))
}

fn built(terms: Vec<ExactTerm>) -> Vec<PlanTerm> {
    terms.into_iter().map(|t| t.term).collect()
}

/// `value × num ÷ den`, rounded up, in `u128`, saturating at `u64::MAX`.
fn scale_up(value: u64, num: u64, den: u64) -> u64 {
    let scaled = (u128::from(value) * u128::from(num)).div_ceil(u128::from(den));
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

fn sum(terms: &[PlanTerm]) -> u128 {
    terms.iter().map(|t| u128::from(t.bytes)).sum()
}

/// The margin's named terms at `budgets` for `inputs`, built and exact: the
/// fixed ones, then one per entry of [`MARGIN_SHARES`] present for `inputs`,
/// the only shares of `L` the margin charges (C3).
fn margin_terms_at(budgets: &Budgets, inputs: &PlanInputs) -> Vec<ExactTerm> {
    let mut terms = vec![ExactTerm::as_built(PlanTerm::fixed(
        "allocator slack",
        ALLOCATOR_SLACK_BYTES,
    ))];
    if inputs.base_bootstrap {
        terms.push(ExactTerm::as_built(PlanTerm::set_by(
            "checkpoint base bootstraps",
            (2 * crate::checkpoint::BOOTSTRAP_MAX_INFLIGHT as u64)
                .saturating_mul(inputs.max_layer_size),
            BASE_BOOTSTRAP_SETTINGS,
        )));
    }
    terms.extend(
        MARGIN_SHARES
            .iter()
            .filter_map(|margin| margin.term_at(budgets.limit, inputs)),
    );
    terms
}

/// The line's terms other than the catch-up budget's, at `budgets` for
/// `inputs`, with the connection buffers on `topology`: built only.
fn line_terms_without_t(
    budgets: &Budgets,
    inputs: &PlanInputs,
    topology: Topology,
) -> Vec<PlanTerm> {
    built(line_terms_at(budgets, inputs, topology))
}

/// The line's terms other than the catch-up budget's, at `budgets` for
/// `inputs`, with the connection buffers on `topology`, built and exact.
fn line_terms_at(budgets: &Budgets, inputs: &PlanInputs, topology: Topology) -> Vec<ExactTerm> {
    use crate::storage::notifications::{CONSUMER_BACKLOG_BYTES, CONSUMER_CHANNEL_CAPACITY};
    use crate::storage::stream_mux::{BACKLOG_HARD_CAP_FACTOR, XREAD_REPLY_TARGET_BYTES};
    use crate::storage::structural_cache::{
        max_entry_bytes_for, resident_budget_bytes, STRUCTURAL_COLD_START_READ_PERMITS,
    };
    use crate::storage::structural_cache_cold_start::plain_cold_start_hold_allowance_bytes;

    let ceiling = inputs.byte_ceiling;
    // `X` and the reference read are measured reply-byte terms: they grow
    // with a raised ceiling and are not lowered with a lowered one.
    let scaled_ceiling = ceiling.max(RECORD_BYTE_CEILING);
    let cache_bytes = budgets.structural_cache_mib * MIB;
    // The exact values below are their term shares at the limit, in bytes
    // (unit 1), so each is a whole number of 1/EXACT_DEN byte by
    // construction.
    let exact_at = |share: Share| share.exact_units_scaled(budgets.limit, 0);
    let hard_cap_entries = (CONSUMER_CHANNEL_CAPACITY * BACKLOG_HARD_CAP_FACTOR) as u64;
    let readers = inputs.pools.readers as u64;
    let as_built = |terms: Vec<PlanTerm>| terms.into_iter().map(ExactTerm::as_built);
    let mut terms = vec![
        ExactTerm::as_built(PlanTerm::fixed(
            "steady-state floor",
            POD_MEMORY_FLOOR_BYTES,
        )),
        // Built, `S × m` rounded up to a whole byte; exact, `S` unrounded
        // times `m`, one byte more for that rounding up.
        ExactTerm::new(
            PlanTerm::fixed(
                "established subscriptions",
                scale_up(budgets.max_subscriptions, SUBSCRIPTION_BYTES_DECI, 10),
            ),
            exact_at(SUBSCRIPTIONS_TERM_SHARE) + EXACT_DEN,
        ),
    ];
    terms.extend(as_built(vec![
        PlanTerm::set_by(
            "catch-up slots",
            inputs
                .catchup_slots
                .saturating_mul(CATCHUP_SLOT_BYTES_PER_CEILING_BYTE)
                .saturating_mul(ceiling),
            "LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS",
        ),
        PlanTerm::set_by(
            "stream readers",
            readers.saturating_mul(XREAD_REPLY_TARGET_BYTES),
            "LIVE_EDIT_SUBSCRIBER_POOL_SIZE",
        ),
        PlanTerm::fixed(
            "one lagging layer",
            (CONSUMER_BACKLOG_BYTES * BACKLOG_HARD_CAP_FACTOR as u64)
                + scale_up(hard_cap_entries, QUEUED_ENTRY_BYTES_DECI, 10),
        ),
        PlanTerm::fixed(
            "catch-ups before their first poll",
            PRE_POLL_SHARED_PAYLOAD_BYTES.saturating_add(scale_up(
                inputs
                    .catchup_slots
                    .saturating_mul(PRE_POLL_ENTRIES_PER_SLOT_CENTI),
                QUEUED_ENTRY_BYTES_DECI,
                1_000,
            )),
        ),
    ]));
    // Built, `C × (1 + φ)` rounded up to a whole byte; exact, `C` unrounded
    // times `1 + φ`, one byte more for that rounding up.
    terms.push(ExactTerm::new(
        PlanTerm::fixed(
            "structural cache",
            resident_budget_bytes(cache_bytes).unwrap_or(u64::MAX),
        ),
        exact_at(STRUCTURAL_CACHE_RESIDENT_SHARE) + EXACT_DEN,
    ));
    terms.push(ExactTerm::as_built(PlanTerm::fixed(
        "structural cache cold-start reads",
        (STRUCTURAL_COLD_START_READ_PERMITS as u64)
            .saturating_mul(ceiling.saturating_add(plain_cold_start_hold_allowance_bytes(ceiling))),
    )));
    // `max_entry_bytes_for`'s rule, a quarter of `C` capped at `u32::MAX`,
    // on `C` unrounded.
    terms.push(ExactTerm::new(
        PlanTerm::fixed(
            "structural cache audit",
            max_entry_bytes_for(cache_bytes)
                .saturating_mul(2)
                .saturating_add(ceiling),
        ),
        2 * exact_at(STRUCTURAL_CACHE_QUARTER_SHARE).min(u128::from(u32::MAX) * EXACT_DEN)
            + u128::from(ceiling) * EXACT_DEN,
    ));
    terms.extend(as_built(vec![
        PlanTerm::fixed(
            "fan-out window transient",
            scale_up(
                inputs
                    .fenced_read_concurrency
                    .saturating_mul(WINDOW_TRANSIENT_BYTES),
                scaled_ceiling,
                RECORD_BYTE_CEILING,
            ),
        ),
        PlanTerm::fixed(
            "checkpoint reference read",
            scale_up(REFERENCE_READ_BYTES, scaled_ceiling, RECORD_BYTE_CEILING),
        ),
    ]));
    terms.push(ExactTerm::new(
        PlanTerm::fixed("upload budget", budgets.upload_mib.saturating_mul(MIB)),
        UPLOAD_SHARE.exact_units_scaled(budgets.limit, budgets.upload_floor_mib) * u128::from(MIB),
    ));
    terms.extend(as_built(vec![
        PlanTerm::set_by(
            "queued uploads",
            inputs
                .upload_waiters
                .saturating_mul(UPLOAD_WAITER_WINDOW_BYTES),
            "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS",
        ),
        PlanTerm::set_by(
            "one maximum-size upload",
            crate::grpc::upload_admission::parse_charge(
                inputs.max_layer_size,
                0,
                inputs.max_layer_size,
            ),
            "LIVE_EDIT_MAX_LAYER_SIZE",
        ),
        // The connections' buffers: the data and txn pools'
        // (`LIVE_EDIT_VALKEY_POOL_SIZE` sizes both), the maintenance and
        // snapshot pools' and the stream readers' (their replies are the
        // "stream readers" term).
        PlanTerm::set_by(
            "connection pools",
            at_rest_bound_bytes(inputs.pools, topology),
            CONNECTION_POOL_SETTINGS,
        ),
    ]));
    terms
}

/// The catch-up budget's term for `t_mib` and `P_f`: built.
fn catchup_term(t_mib: u64, inputs: &PlanInputs) -> PlanTerm {
    PlanTerm::fixed(
        "catch-up memory budget",
        (t_mib * MIB).saturating_mul(1 + inputs.fenced_read_concurrency),
    )
}

/// `T × (1 + P_f)` at its exact share for one hundredth of `L − floor`, in
/// 1/[`EXACT_DEN`] byte: the cost of each hundredth of `T`'s step. `None`
/// when it does not fit in `u128`, which no room the line leaves can carry.
fn catchup_exact_per_hundredth(limit: u64, inputs: &PlanInputs) -> Option<u128> {
    CATCHUP_HUNDREDTH
        .exact_units_scaled(limit, 0)
        .checked_mul(u128::from(MIB))?
        .checked_mul(u128::from(inputs.fenced_read_concurrency) + 1)
}

/// `T`'s step at `limit`, in hundredths of `L − floor` (each one
/// [`CATCHUP_BUDGET_STEP_CENTI`]): the largest, at most 100, at which
/// `T × (1 + P_f)`, at its exact share, fits `room`, the exact room (in
/// 1/[`EXACT_DEN`] byte) the line's other terms leave within 80 % of `L`.
/// Computed directly, `⌊room ÷ cost per hundredth⌋`; 0 when no step fits,
/// or when `L` is at or below the floor.
fn catchup_step_centi(limit: u64, room: u128, inputs: &PlanInputs) -> u64 {
    match catchup_exact_per_hundredth(limit, inputs) {
        Some(per_hundredth) if per_hundredth > 0 => {
            u64::try_from((room / per_hundredth).min(100)).unwrap_or(100)
        }
        _ => 0,
    }
}

/// The line limit, built: 80 % of `limit`, rounded down to a whole byte.
const fn line_limit(limit: u64) -> u64 {
    ((limit as u128 * 4) / 5) as u64
}

/// Where the catch-up budget's term sits in the line: beside the other
/// fenced-read terms, before this one.
const CATCHUP_TERM_BEFORE: &str = "fan-out window transient";

/// Whether the margin's named terms, at exact shares, fit 20 % of `limit`.
fn margin_fits(limit: u64, margin_terms: &[ExactTerm]) -> bool {
    exact_sum(margin_terms) <= MARGIN_ALLOWANCE.exact_units_scaled(limit, 0)
}

/// `T`'s step, in hundredths, and `T`, in MiB, when the line closes at
/// `limit` with `line_terms` (its terms other than `T`'s, exact): a step of
/// `T` fits the room they leave within 80 % of `limit`, and gives `T` at
/// least 1 MiB. `None` when the line does not close there.
fn line_closure(limit: u64, line_terms: &[ExactTerm], inputs: &PlanInputs) -> Option<(u64, u64)> {
    let room = LINE_ALLOWANCE
        .exact_units_scaled(limit, 0)
        .checked_sub(exact_sum(line_terms))?;
    let step_centi = catchup_step_centi(limit, room, inputs);
    if step_centi == 0 {
        return None;
    }
    let catchup_mib = CATCHUP_HUNDREDTH.times(step_centi).built_units(limit, 0);
    (catchup_mib != 0).then_some((step_centi, catchup_mib))
}

/// Which sides of the plan do not close at `limit` for `inputs`, at exact
/// shares: `(the margin, the line)`. One MiB below the minimum, a side that
/// does not close is one that sets the minimum, as each side alone closes at
/// every limit above the first at which it does (module docs, *Adding a
/// budget*).
fn sides_not_closing(limit: u64, inputs: &PlanInputs) -> (bool, bool) {
    let budgets = budgets_at(limit, inputs);
    let margin = !margin_fits(limit, &margin_terms_at(&budgets, inputs));
    let line_terms = line_terms_at(&budgets, inputs, Topology::cluster(10));
    let line = line_closure(limit, &line_terms, inputs).is_none();
    (margin, line)
}

/// The plan at one limit, when it closes there: built values, decided on
/// exact ones.
#[derive(Debug, Clone)]
struct Evaluation {
    budgets: Budgets,
    catchup_mib: u64,
    step_centi: u64,
    line_terms: Vec<PlanTerm>,
    margin_terms: Vec<PlanTerm>,
}

impl Evaluation {
    /// The plan at `limit` bytes, or `None` when it does not close there: at
    /// exact shares, the margin's named terms exceed 20 % of `limit`, or no
    /// step of `T` fits the line within 80 % of it, or the one that does
    /// gives `T` less than 1 MiB (module docs, *Exact shares*).
    fn new(limit: u64, inputs: &PlanInputs) -> Option<Self> {
        let budgets = budgets_at(limit, inputs);
        let margin_terms = margin_terms_at(&budgets, inputs);
        if !margin_fits(limit, &margin_terms) {
            return None;
        }
        let line_terms = line_terms_at(&budgets, inputs, Topology::cluster(10));
        let (step_centi, catchup_mib) = line_closure(limit, &line_terms, inputs)?;
        let mut line_terms = built(line_terms);
        let at = line_terms
            .iter()
            .position(|t| t.name == CATCHUP_TERM_BEFORE)
            .unwrap_or(line_terms.len());
        line_terms.insert(at, catchup_term(catchup_mib, inputs));
        Some(Self {
            budgets,
            catchup_mib,
            step_centi,
            line_terms,
            margin_terms: built(margin_terms),
        })
    }
}

/// Every term of the plan at `limit` for `inputs`, with its built and exact
/// values: the line's, the connection buffers on a 10-primary cluster and
/// the catch-up budget's term at a step of `step_centi` hundredths, then the
/// margin's. What the exact-share contract is checked on (module docs,
/// *Adding a budget*): C1 on each term, C2 on each line term between two
/// limits ([`line_shape_holds`]), C3 on the margin's. A `T` term whose exact
/// value does not fit in `u128` is given `u128::MAX`.
#[cfg(test)]
pub(crate) fn exact_terms_at(
    limit: u64,
    inputs: &PlanInputs,
    step_centi: u64,
) -> (Vec<ExactTerm>, Vec<ExactTerm>) {
    let budgets = budgets_at(limit, inputs);
    let mut line = line_terms_at(&budgets, inputs, Topology::cluster(10));
    let catchup = ExactTerm::new(
        catchup_term(
            CATCHUP_HUNDREDTH.times(step_centi).built_units(limit, 0),
            inputs,
        ),
        catchup_exact_per_hundredth(limit, inputs)
            .and_then(|per_hundredth| per_hundredth.checked_mul(u128::from(step_centi)))
            .unwrap_or(u128::MAX),
    );
    let at = line
        .iter()
        .position(|t| t.term.name == CATCHUP_TERM_BEFORE)
        .unwrap_or(line.len());
    line.insert(at, catchup);
    (line, margin_terms_at(&budgets, inputs))
}

/// C2's check of one line term between the limits `low` < `high`, given its
/// exact values there (in 1/[`EXACT_DEN`] byte): whether its exact value per
/// byte of `L − floor` does not increase, `exact_high × (low − floor) ≤
/// exact_low × (high − floor)`, in exact integers. Any term can be put
/// through it, a synthetic one included. Vacuously true with `low` at or
/// below the floor, where C2 says nothing.
#[cfg(test)]
pub(crate) fn line_shape_holds(low: u64, exact_low: u128, high: u64, exact_high: u128) -> bool {
    /// `a × b` as (the bits above the low 64, the low 64 bits).
    fn wide(a: u128, b: u64) -> (u128, u64) {
        let b = u128::from(b);
        let lo = (a & u128::from(u64::MAX)) * b;
        let hi = (a >> 64) * b + (lo >> 64);
        (hi, lo as u64)
    }
    let above = |limit: u64| limit.saturating_sub(POD_MEMORY_FLOOR_BYTES);
    wide(exact_high, above(low)) <= wide(exact_low, above(high))
}

/// The smallest limit, in whole MiB, at which the plan closes for `inputs`;
/// `None` when none up to [`SEARCH_CEILING_MIB`] does. Exact: closure holds
/// at every limit above the first at which it holds (module docs, *Adding a
/// budget*), so the bisection's answer is that first limit, and every limit
/// from it up closes.
///
/// Bounded: at most 20 doublings from 1 GiB to the ceiling, then at most 30
/// halvings, each one [`Evaluation::new`], which computes `T`'s step rather
/// than searching it. Every term is computed saturating, checked or in
/// `u128`, so no input overflows.
fn minimum_limit_mib(inputs: &PlanInputs) -> Option<u64> {
    let closes = |mib: u64| Evaluation::new(mib.saturating_mul(MIB), inputs).is_some();
    let mut high = 1_024;
    while !closes(high) {
        if high >= SEARCH_CEILING_MIB {
            return None;
        }
        high *= 2;
    }
    let mut low = 0;
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if closes(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    Some(high)
}

/// `n` with thousands separators.
pub(crate) fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// `bytes` in whole MiB, rounded to nearest, with thousands separators.
fn mib_text(bytes: u64) -> String {
    grouped(bytes.saturating_add(MIB / 2) / MIB)
}

/// The pod memory plan: the limit, the budgets derived from it and the term
/// list they close (module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodMemoryPlan {
    limit: PodMemoryLimit,
    max_subscriptions: u64,
    structural_cache_mib: u64,
    upload_mib: u64,
    retirement_mib: u64,
    ext_mirror_mib: Option<u64>,
    catchup_mib: u64,
    catchup_step_centi: u64,
    line_terms: Vec<PlanTerm>,
    margin_terms: Vec<PlanTerm>,
    minimum_limit_mib: Option<u64>,
}

impl PodMemoryPlan {
    /// The plan for `limit` and `inputs`, or the startup refusal when `limit`
    /// is below the configuration's minimum limit or no limit fits it (module
    /// docs, *Refusal*).
    pub fn derive(limit: PodMemoryLimit, inputs: &PlanInputs) -> Result<Self, String> {
        Self::derive_against(limit, inputs, minimum_limit_mib(inputs))
    }

    /// [`Self::derive`] against `minimum_limit_mib`, the configuration's
    /// minimum limit: the refusal when there is none or `limit` is below it,
    /// before the plan is evaluated; else the plan at `limit`, which closes
    /// there by the exact-share contract, or, if it does not, the defect
    /// error ([`defect`]). `derive` passes the minimum
    /// [`minimum_limit_mib`] finds; a test can pass a false one, as a term
    /// breaking the contract would make it, to reach the defect error.
    fn derive_against(
        limit: PodMemoryLimit,
        inputs: &PlanInputs,
        minimum_limit_mib: Option<u64>,
    ) -> Result<Self, String> {
        let minimum = match minimum_limit_mib {
            Some(minimum) if limit.mib() >= minimum => minimum,
            _ => return Err(refusal(&limit, inputs, minimum_limit_mib)),
        };
        let Some(eval) = Evaluation::new(limit.bytes, inputs) else {
            return Err(defect(&limit, minimum));
        };
        Ok(Self {
            limit,
            max_subscriptions: eval.budgets.max_subscriptions,
            structural_cache_mib: eval.budgets.structural_cache_mib,
            upload_mib: eval.budgets.upload_mib,
            retirement_mib: eval.budgets.retirement_mib,
            ext_mirror_mib: eval.budgets.ext_mirror_mib,
            catchup_mib: eval.catchup_mib,
            catchup_step_centi: eval.step_centi,
            line_terms: eval.line_terms,
            margin_terms: eval.margin_terms,
            minimum_limit_mib,
        })
    }

    /// The plan at the reference limit for the chart's configuration
    /// ([`PlanInputs::reference`]): the plan of a [`Config`] built field by
    /// field rather than by [`Config::from_env`](super::Config::from_env).
    pub fn reference() -> Self {
        static REFERENCE: OnceLock<PodMemoryPlan> = OnceLock::new();
        REFERENCE
            .get_or_init(|| {
                Self::derive(PodMemoryLimit::reference(None), &PlanInputs::reference())
                    .expect("the plan closes at the reference limit for the chart's configuration")
            })
            .clone()
    }

    /// The limit the plan was derived for.
    pub fn limit(&self) -> &PodMemoryLimit {
        &self.limit
    }

    /// `S`: the `SubscribeReadLayer` subscriptions admitted at once.
    pub fn max_subscriptions(&self) -> u64 {
        self.max_subscriptions
    }

    /// `C`, the structural-planning cache's budget, in MiB.
    pub fn structural_cache_budget_mib(&self) -> u64 {
        self.structural_cache_mib
    }

    /// `C` in bytes.
    pub fn structural_cache_budget_bytes(&self) -> u64 {
        self.structural_cache_mib * MIB
    }

    /// The cache's resident budget, `C × (1 + φ)`
    /// (`storage::structural_cache::resident_budget_bytes`): what its weight
    /// gauge is bounded by.
    pub fn structural_cache_resident_budget_bytes(&self) -> u64 {
        crate::storage::structural_cache::resident_budget_bytes(
            self.structural_cache_budget_bytes(),
        )
        .unwrap_or(u64::MAX)
    }

    /// `B`, the upload memory budget, in MiB.
    pub fn upload_budget_mib(&self) -> u64 {
        self.upload_mib
    }

    /// `B` in bytes.
    pub fn upload_budget_bytes(&self) -> u64 {
        self.upload_mib.saturating_mul(MIB)
    }

    /// `T`, the catch-up memory budget, in MiB.
    pub fn catchup_budget_mib(&self) -> u64 {
        self.catchup_mib
    }

    /// `T` in bytes.
    pub fn catchup_budget_bytes(&self) -> u64 {
        self.catchup_mib * MIB
    }

    /// The step `s` `T` was taken at, in hundredths of `L − floor`.
    pub fn catchup_step_centi(&self) -> u64 {
        self.catchup_step_centi
    }

    /// `R_ref`, the retirement memory allowance DeleteLayer, the overwrite
    /// cleanup and the orphan sweep share (`storage::retirement`): `L / 500`,
    /// a margin term.
    pub fn retirement_budget_bytes(&self) -> u64 {
        self.retirement_mib * MIB
    }

    /// `M_ext`, the external-value mirror's byte budget, which the backup
    /// repair's one re-send in flight is charged to as well
    /// (`checkpoint::ext_backup_repair`): `max(L / 40, one
    /// maximum-size value's charge)`, a margin term. `Some` exactly when
    /// checkpointing is on with an adapter ([`PlanInputs::ext_mirror`]), and
    /// `main.rs` builds the mirror exactly then, with this as its budget;
    /// `None` otherwise, and for [`Self::reference`], which has no adapter.
    pub fn ext_mirror_bytes(&self) -> Option<u64> {
        self.ext_mirror_mib.map(|mib| mib.saturating_mul(MIB))
    }

    /// The line's terms, the connection buffers on a 10-primary cluster.
    pub fn line_terms(&self) -> &[PlanTerm] {
        &self.line_terms
    }

    /// The margin's named terms.
    pub fn margin_terms(&self) -> &[PlanTerm] {
        &self.margin_terms
    }

    /// The line's total, in bytes.
    pub fn line_total_bytes(&self) -> u64 {
        u64::try_from(sum(&self.line_terms)).unwrap_or(u64::MAX)
    }

    /// The line's limit, 80 % of `L`, in bytes.
    pub fn line_limit_bytes(&self) -> u64 {
        line_limit(self.limit.bytes)
    }

    /// What the line leaves under its limit on a 10-primary cluster, in
    /// bytes; a smaller cluster or a standalone store leaves a little more.
    pub fn line_headroom_bytes(&self) -> u64 {
        self.line_limit_bytes()
            .saturating_sub(self.line_total_bytes())
    }

    /// The margin's named terms' total, in bytes.
    pub fn margin_named_bytes(&self) -> u64 {
        u64::try_from(sum(&self.margin_terms)).unwrap_or(u64::MAX)
    }

    /// What the margin's named terms leave of its 20 % of `L`, in bytes.
    pub fn margin_left_bytes(&self) -> u64 {
        (self.limit.bytes - self.line_limit_bytes()).saturating_sub(self.margin_named_bytes())
    }

    /// The smallest limit this configuration closes at, in whole MiB, from
    /// which every larger limit closes too; `None` when none up to 1 PiB
    /// does.
    pub fn minimum_limit_mib(&self) -> Option<u64> {
        self.minimum_limit_mib
    }

    /// Log the plan once at startup: INFO, or WARN when the limit is the
    /// reference size on Linux, where a container is expected to have a
    /// readable limit (INFO elsewhere, as for a local run on macOS). The
    /// WARN's remedy follows its reason: set a limit when none is set; make
    /// the cgroup memory files visible when they are not found; fix the file
    /// the reason names when a file the limit is read from cannot be read or
    /// parsed. Every WARN starts with the same text, and every line carries
    /// the same fields.
    pub fn log_startup(&self) {
        let limit = &self.limit;
        let reference = limit.source == PodMemoryLimitSource::Reference;
        macro_rules! plan_line {
            ($level:ident, $message:literal) => {
                tracing::$level!(
                    pod_memory_limit_bytes = limit.bytes,
                    source = limit.source.label(),
                    reason = %limit.reason_text(),
                    upload_budget_bytes = self.upload_budget_bytes(),
                    catchup_budget_bytes = self.catchup_budget_bytes(),
                    catchup_budget_step = self.catchup_step_centi as f64 / 100.0,
                    structural_cache_budget_bytes = self.structural_cache_budget_bytes(),
                    max_subscriptions = self.max_subscriptions,
                    retirement_budget_bytes = self.retirement_budget_bytes(),
                    ext_mirror_budget_bytes = self.ext_mirror_bytes().unwrap_or(0),
                    line_total_bytes = self.line_total_bytes(),
                    line_limit_bytes = self.line_limit_bytes(),
                    line_headroom_bytes = self.line_headroom_bytes(),
                    margin_named_bytes = self.margin_named_bytes(),
                    margin_left_bytes = self.margin_left_bytes(),
                    minimum_limit_mib = self.minimum_limit_mib,
                    $message
                )
            };
        }
        if reference && cfg!(target_os = "linux") {
            match limit.reason {
                Some(ReferenceReason::Unlimited) => plan_line!(
                    warn,
                    "no readable pod memory limit: every memory budget is sized for the 5,120 \
                     MiB reference pod (see reason). No memory limit is set on the container: \
                     set resources.limits.memory so the budgets follow the memory the pod \
                     really has"
                ),
                Some(ReferenceReason::NotFound) => plan_line!(
                    warn,
                    "no readable pod memory limit: every memory budget is sized for the 5,120 \
                     MiB reference pod (see reason). The container cannot see its cgroup memory \
                     files: check that /sys/fs/cgroup is mounted in the container and its memory \
                     controller is enabled, so the limit set on the container can be read"
                ),
                Some(ReferenceReason::Unreadable(_)) => plan_line!(
                    warn,
                    "no readable pod memory limit: every memory budget is sized for the 5,120 \
                     MiB reference pod (see reason). A file the limit is read from could not be \
                     read or parsed: the reason names the file and the error; fix that so the \
                     limit set on the container can be read"
                ),
                // The reference size chosen without a cgroup read
                // (`PodMemorySource::Reference`), which carries no reason.
                None => plan_line!(
                    warn,
                    "no readable pod memory limit: every memory budget is sized for the 5,120 \
                     MiB reference pod (see reason). Set the container's memory limit \
                     (resources.limits.memory) so the budgets follow the memory the pod really \
                     has"
                ),
            }
        } else {
            plan_line!(
                info,
                "pod memory budgets derived from the pod memory limit; the line (80 % of the \
                 limit) and the margin (20 %) close"
            );
        }
    }
}

/// The text naming `limit` and its source.
fn pod_text(limit: &PodMemoryLimit) -> String {
    format!(
        "this pod's memory limit, {} MiB ({})",
        grouped(limit.mib()),
        limit.source.describe()
    )
}

/// `terms`' operator-sized ones, largest first, at most `most` of them, each
/// with its size and settings: the refusal's list.
fn sized_terms_text(terms: Vec<PlanTerm>, most: usize) -> String {
    let mut sized: Vec<PlanTerm> = terms.into_iter().filter(|t| t.operator_sized).collect();
    sized.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    sized
        .iter()
        .take(most)
        .map(|t| match t.setting {
            Some(setting) => format!("{} {} MiB ({setting})", t.name, mib_text(t.bytes)),
            None => format!("{} {} MiB", t.name, mib_text(t.bytes)),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The close of every refusal whose margin sets the minimum, or fits no
/// limit: checkpointing and the adapter switch the margin's terms on, and the
/// message names them, but turning either off removes backups, so it is never
/// a remedy (module docs, *Refusal*).
const KEEP_BACKUPS: &str = "do not turn off LIVE_EDIT_CHECKPOINT_ENABLED or \
                            LIVE_EDIT_CHECKPOINT_ADAPTER for this: that removes backups";

/// The startup refusal for `limit`, below `minimum_mib`, the configuration's
/// minimum limit, or with none (module docs, *Refusal*). It names the terms of
/// the side that sets the minimum, the side that does not close one MiB below
/// it (at 1 PiB when no limit fits): the line's largest four, the margin's,
/// or both. When the margin is named, raising the limit comes first where a
/// limit fits; the margin's remedy is then lowering `LIVE_EDIT_MAX_LAYER_SIZE`
/// or, while the base bootstraps are charged, turning off the tail, with its
/// cost; and the message says not to turn off checkpointing or the adapter.
fn refusal(limit: &PodMemoryLimit, inputs: &PlanInputs, minimum_mib: Option<u64>) -> String {
    let budgets = budgets_at(limit.bytes, inputs);
    let line = sized_terms_text(
        line_terms_without_t(&budgets, inputs, Topology::cluster(10)),
        4,
    );
    let margin = sized_terms_text(built(margin_terms_at(&budgets, inputs)), usize::MAX);
    let binding_mib = minimum_mib.map_or(SEARCH_CEILING_MIB, |minimum| minimum.saturating_sub(1));
    let (margin_binds, line_binds) = sides_not_closing(binding_mib.saturating_mul(MIB), inputs);
    let names_margin = margin_binds && !margin.is_empty();
    let names_line = line_binds || !names_margin;
    let mut named = Vec::with_capacity(2);
    if names_line {
        named.push(format!("the largest fixed terms: {line}"));
    }
    if names_margin {
        named.push(format!(
            "the terms of the 20 % memory margin that settings size: {margin}"
        ));
    }
    let named = named.join("; ");
    let pod = pod_text(limit);
    // The margin's remedy, after raising the limit where one fits: lower the
    // one setting that sizes its terms, or, while the base bootstraps are charged, turn
    // off the tail that switches them on, at the cost it states. Never
    // turning off checkpointing or the adapter, which also switch its terms
    // on: that removes backups, so the message says not to.
    let margin_remedy = if inputs.base_bootstrap {
        "lower LIVE_EDIT_MAX_LAYER_SIZE, or turn off incremental backups \
         (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false), though recovery points then come only from \
         full backup passes, so a newly created or just-restored layer has no backup of its \
         current state until the next pass"
    } else {
        "lower LIVE_EDIT_MAX_LAYER_SIZE"
    };
    // Both sides set the minimum: each must be lowered.
    let lower = if names_line {
        format!("lower the largest fixed terms' settings and, for the margin, {margin_remedy}")
    } else {
        margin_remedy.to_owned()
    };
    match minimum_mib {
        // The line alone sets the minimum: lowering a setting it names lowers
        // the minimum.
        Some(minimum) if !names_margin => format!(
            "{pod}, is below the {} MiB this configuration needs, {} MiB short; {named}; raise \
             resources.limits.memory to at least {} MiB or lower those settings",
            grouped(minimum),
            grouped(minimum.saturating_sub(limit.mib())),
            grouped(minimum),
        ),
        // The margin sets it, alone or with the line: raising the limit comes
        // first, then the margin's remedy above.
        Some(minimum) => format!(
            "{pod}, is below the {} MiB this configuration needs, {} MiB short; {named}; raise \
             resources.limits.memory to at least {} MiB, which costs nothing else; failing that, \
             {lower}; {KEEP_BACKUPS}",
            grouped(minimum),
            grouped(minimum.saturating_sub(limit.mib())),
            grouped(minimum),
        ),
        None => match largest_closing_fenced_read_concurrency(inputs) {
            // The settings alone are not the cause: at P_f = 1 some limit
            // fits, so it is the catch-up budget's bound, T × (1 + P_f), whose
            // smallest step already grows with the limit faster than the line
            // can.
            Some(most) => format!(
                "no pod memory limit can fit this configuration ({pod}): at \
                 LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY={}, the catch-up memory budget's \
                 bound, T x (1 + that setting), grows with the limit faster than the limit \
                 leaves room for it, so a larger limit never helps; lower \
                 LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY to at most {most} (the default is 1)",
                inputs.fenced_read_concurrency,
            ),
            None if !names_margin => format!(
                "no pod memory limit can fit this configuration ({pod}): the memory its \
                 settings fix does not fit any limit up to 1 PiB; {named}; lower those settings"
            ),
            None => format!(
                "no pod memory limit can fit this configuration ({pod}): the memory its \
                 settings fix does not fit any limit up to 1 PiB; {named}; {lower}; \
                 {KEEP_BACKUPS}"
            ),
        },
    }
}

/// The startup error for `limit` at or above `minimum_mib`, the
/// configuration's minimum limit, where the plan still does not close: a
/// term breaks the exact-share contract, which makes closure hold at every
/// limit from the minimum up (module docs, *Refusal*, *Adding a budget*). A
/// defect of the service, not of its configuration; the service fails
/// closed rather than start on a plan that does not close.
fn defect(limit: &PodMemoryLimit, minimum_mib: u64) -> String {
    format!(
        "pod memory plan defect: the plan does not close at {}, at or above the {} MiB \
         minimum it reports for this configuration, so one of its budget terms breaks the \
         exact-share contract (C1 to C3, src/config/memory_plan.rs, module docs, \"Adding a \
         budget\"); this is a defect in the service, not in its configuration: report it",
        pod_text(limit),
        grouped(minimum_mib),
    )
}

/// The largest `P_f` at or below the configured one for which the plan
/// closes at some limit, all other inputs unchanged; `None` when it closes at
/// no limit even at `P_f` = 1. A plan that closes at any limit closes at
/// every larger one, the search ceiling included (module docs, *Adding a
/// budget*: proven, not assumed), so that is the limit each `P_f` is tried
/// at. Bounded: a binary search over `1..=P_f`, at most 64 steps.
fn largest_closing_fenced_read_concurrency(inputs: &PlanInputs) -> Option<u64> {
    let ceiling = SEARCH_CEILING_MIB.saturating_mul(MIB);
    let closes = |p_f: u64| {
        let inputs = PlanInputs {
            fenced_read_concurrency: p_f,
            ..*inputs
        };
        Evaluation::new(ceiling, &inputs).is_some()
    };
    if !closes(1) {
        return None;
    }
    let (mut low, mut high) = (1, inputs.fenced_read_concurrency.max(1));
    if closes(high) {
        return Some(high);
    }
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if closes(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(low)
}

/// The plan's rules, at the plan's own seams: the closure sweep over every
/// whole MiB to 64 GiB with the exact-share contract checked term by term
/// (W4), the configurations whose rounded closure had gaps, the contract's
/// C2 check against a term that breaks it, `T`'s 0.01 step (W5), the
/// refusals and what they name (W6), the defect error, the startup line
/// (W7), and the reference constants a `Config` built field by field
/// carries. The
/// plan against its records is checked in `config`'s tests, beside the
/// records' other pins; the cgroup reader in `pod_memory`'s.
///
/// Build shapes: the line's connection term depends on the `valkey-tls`
/// feature (`connection_buffers`), so a figure the line sets — a minimum the
/// line sets, `T` at a limit — is pinned exactly in a
/// `#[cfg(feature = "valkey-tls")]` test named `…_in_the_shipped_build`,
/// absent from a no-TLS run rather than skipped inside a test that reports
/// passed; the tests that run in both shapes check what holds in either. A
/// minimum the margin sets has no connection term and is pinned in both.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::pod_memory::{PodMemorySource, ReferenceReason};

    /// The limits the closure sweep ends at: 64 GiB.
    const SWEEP_TOP_MIB: u64 = 64 * 1024;

    /// The binary's defaults at one CPU count, with their minimum limits.
    struct BinaryDefaults {
        /// The CPUs the "auto" pool sizes are resolved for.
        cores: u64,
        /// The minimum limit, MiB, without a checkpoint adapter.
        minimum_mib: u64,
        /// The minimum limit, MiB, with a checkpoint adapter.
        minimum_with_adapter_mib: u64,
    }

    /// The CPU counts the binary's "auto" pool sizes are swept at, so the
    /// sweep is the same on every runner (`effective_pool_size` and
    /// `effective_subscriber_pool_size` read the machine's CPUs), each with
    /// its minimum limits, without and with a checkpoint adapter: the figures
    /// SDD §16 and the implementation changelog state. Without an adapter the
    /// line sets the minimum, pinned exactly in the shipped build
    /// (`valkey-tls` on;
    /// [`pod_memory_plan_minimums_for_the_binary_defaults_are_the_tables_in_the_shipped_build`])
    /// and as an upper bound in every shape. With an adapter the
    /// margin sets the minimum (the base bootstraps' 400 MiB and the mirror's
    /// 101 MiB floor), the same 2,809 MiB at 1 to 64 CPUs, the counts swept,
    /// in every shape.
    /// 10 CPUs is the implementation changelog's 10-core host.
    ///
    /// The 2-CPU entry is also a gap row: on budgets rounded down to whole
    /// MiB the plan closed at 2,154 MiB and refused at 2,155, where `T`'s
    /// smallest step, rounded, grew from 19 to 20 MiB; at exact shares its
    /// minimum is 2,158 MiB and every limit from it up closes.
    const BINARY_DEFAULTS: [BinaryDefaults; 8] = [
        BinaryDefaults {
            cores: 1,
            minimum_mib: 2_155,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 2,
            minimum_mib: 2_158,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 4,
            minimum_mib: 2_166,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 8,
            minimum_mib: 2_194,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 10,
            minimum_mib: 2_208,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 16,
            minimum_mib: 2_250,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 32,
            minimum_mib: 2_363,
            minimum_with_adapter_mib: 2_809,
        },
        BinaryDefaults {
            cores: 64,
            minimum_mib: 2_590,
            minimum_with_adapter_mib: 2_809,
        },
    ];

    /// The four store topologies the line must close on.
    const TOPOLOGIES: [(&str, Topology); 4] = [
        ("standalone", Topology::standalone()),
        ("1 primary", Topology::cluster(1)),
        ("3 primaries", Topology::cluster(3)),
        ("10 primaries", Topology::cluster(10)),
    ];

    /// A limit of `mib` MiB, as a cgroup v2 `memory.max` would give it.
    fn limit_of(mib: u64) -> PodMemoryLimit {
        PodMemorySource::Fixed(mib * MIB).resolve()
    }

    fn derive_at(mib: u64, inputs: &PlanInputs) -> Result<PodMemoryPlan, String> {
        PodMemoryPlan::derive(limit_of(mib), inputs)
    }

    /// The binary's own defaults on a machine with `cores` CPUs:
    /// `Config::from_env` with only the two required endpoints set and the
    /// pools its "auto" sizes would resolve to there (`connect`'s rule: the
    /// data and txn pools `max(2 × cores, 8)`, the stream readers
    /// `2 × cores`), set explicitly so the result does not depend on the
    /// machine running the test.
    fn binary_default_inputs(cores: u64) -> PlanInputs {
        let pool = (2 * cores)
            .max(crate::storage::external_values::STORE_VALUES_MAX_INFLIGHT as u64)
            .to_string();
        let readers = (2 * cores).to_string();
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
                    None,
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some(pool.as_str())),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some(readers.as_str())),
            ],
            || {
                let config = Config::from_env_with(PodMemorySource::Reference)
                    .expect("the binary's defaults parse");
                PlanInputs::from_config(&config)
            },
        )
    }

    /// The binary's "auto" resolution is the rule [`binary_default_inputs`]
    /// sets explicitly: on this machine, with the pools unset, `from_env`
    /// derives the inputs the rule gives for its CPU count.
    #[test]
    fn binary_default_inputs_follow_the_auto_pool_rule() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(1);
        let auto = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
                    None,
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", None),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", None),
            ],
            || {
                PlanInputs::from_config(
                    &Config::from_env_with(PodMemorySource::Reference)
                        .expect("the binary's defaults parse"),
                )
            },
        );
        assert_eq!(auto, binary_default_inputs(cores), "{cores} CPUs");
    }

    /// `inputs` with a checkpoint adapter wired and its tail on, so the base
    /// bootstrap's and the mirror's margin terms are charged.
    fn with_adapter(inputs: PlanInputs) -> PlanInputs {
        PlanInputs {
            base_bootstrap: true,
            ext_mirror: true,
            ..inputs
        }
    }

    /// The connection pools' term on each of [`TOPOLOGIES`] for `inputs`,
    /// checked at `budgets` to be the only line term the topology changes:
    /// so the line on another topology is the plan's 10-primary line with
    /// that one term replaced.
    fn connection_terms(
        label: &str,
        budgets: &Budgets,
        inputs: &PlanInputs,
    ) -> [(&'static str, u64); 4] {
        let on_ten = line_terms_without_t(budgets, inputs, Topology::cluster(10));
        TOPOLOGIES.map(|(name, topology)| {
            let pools = at_rest_bound_bytes(inputs.pools, topology);
            let on_this = line_terms_without_t(budgets, inputs, topology);
            assert_eq!(on_this.len(), on_ten.len(), "{label}: {name}'s line");
            for (this, ten) in on_this.iter().zip(&on_ten) {
                let expected = if ten.name == "connection pools" {
                    pools
                } else {
                    ten.bytes
                };
                assert_eq!(
                    (this.name, this.bytes),
                    (ten.name, expected),
                    "{label}: on {name} only the connection pools' term differs from 10 \
                     primaries'"
                );
            }
            (name, pools)
        })
    }

    /// W4: the closure sweep for `inputs`, at every whole MiB from 1 to
    /// [`SWEEP_TOP_MIB`]:
    ///
    /// - the plan closes exactly at the limits from the reported minimum up:
    ///   none below it closes, every one from it up does, and the refusal one
    ///   MiB below it names it;
    /// - `T` never decreases as `L` grows, nor does its step; nor do `S`, `B`,
    ///   `C`, `R_ref` and `M_ext`;
    /// - `M_ext` and its margin term are charged exactly when the mirror is
    ///   wired, at `max(⌊L/40⌋, the floor)` MiB (the floor restated from its
    ///   rule, not read from the code); the term holds the charge the mirror
    ///   takes for one maximum-size value (`item_charge_kib` of the size cap),
    ///   so every value under the cap fits, and is named with the settings
    ///   that switch it on and size its floor;
    /// - at every closing limit the built line, `T × (1 + P_f)` included,
    ///   fits 80 % of the limit on a standalone store and on 1, 3 and 10
    ///   primaries, and the built margin's named terms fit the other 20 %;
    /// - the exact-share contract (module docs, *Adding a budget*), term by
    ///   term through [`exact_terms_at`], `T`'s term at the plan's step: C1,
    ///   every line and margin term's exact value is at least its built one;
    ///   C2, every line term's exact value per byte of `L − floor` does not
    ///   increase from `L` to `L + 1 MiB` ([`line_shape_holds`], `T`'s term at
    ///   `L`'s step at both); C3, the margin's exact terms grow by at most a
    ///   fifth of a MiB per MiB. The built values of the terms
    ///   [`exact_terms_at`] gives are checked to be the plan's own terms.
    ///
    /// Pure arithmetic, three evaluations of the line per MiB: about a second
    /// a configuration in a debug build.
    fn assert_closes_and_grows_from_its_minimum(label: &str, inputs: &PlanInputs) {
        let minimum = minimum_limit_mib(inputs)
            .unwrap_or_else(|| panic!("{label}: the configuration closes at some limit"));
        for mib in 1..minimum {
            assert!(
                Evaluation::new(mib * MIB, inputs).is_none(),
                "{label}: the plan closes at {mib} MiB, below the reported minimum {minimum} MiB"
            );
        }
        let refusal =
            derive_at(minimum - 1, inputs).expect_err("one MiB below the minimum the plan refuses");
        assert!(
            refusal.contains(&format!(
                "raise resources.limits.memory to at least {} MiB",
                grouped(minimum)
            )),
            "{label}: the refusal names the minimum {minimum} MiB: {refusal}"
        );
        assert_eq!(
            derive_at(minimum, inputs)
                .unwrap_or_else(|e| panic!("{label}: the plan closes at its minimum: {e}"))
                .minimum_limit_mib(),
            Some(minimum)
        );

        let pools = connection_terms(label, &budgets_at(minimum * MIB, inputs), inputs);
        let pools_on_ten = at_rest_bound_bytes(inputs.pools, Topology::cluster(10));
        // C3's bound: a fifth of a MiB, in 1/EXACT_DEN byte.
        let margin_growth_bound = u128::from(MIB) * EXACT_DEN / 5;
        // The mirror's charge for one maximum-size value, as the sender
        // charges it: `M_ext` must hold it wherever the plan funds the mirror.
        let mirror_charge =
            crate::checkpoint::ext_mirror::item_charge_kib(inputs.max_layer_size) * 1024;
        let mut previous: Option<Evaluation> = None;
        for mib in minimum..=SWEEP_TOP_MIB {
            let limit = mib * MIB;
            let eval = Evaluation::new(limit, inputs).unwrap_or_else(|| {
                panic!("{label}: the plan closes at {minimum} MiB but not at {mib} MiB")
            });
            let (line, margin) = exact_terms_at(limit, inputs, eval.step_centi);
            assert!(
                line.iter().map(|t| &t.term).eq(&eval.line_terms)
                    && margin.iter().map(|t| &t.term).eq(&eval.margin_terms),
                "{label}: at {mib} MiB the contract's terms are not the plan's"
            );

            let allowance = u128::from(line_limit(limit));
            let on_ten = sum(&eval.line_terms);
            for (topology, pools) in pools {
                let total = on_ten - u128::from(pools_on_ten) + u128::from(pools);
                assert!(
                    total <= allowance,
                    "{label}: at {mib} MiB the line on {topology} is {total} B, over 80 % of the \
                     limit, {allowance} B"
                );
            }
            assert!(
                sum(&eval.margin_terms) <= u128::from(limit) - allowance,
                "{label}: at {mib} MiB the margin's named terms exceed 20 % of the limit"
            );
            let m_ext = inputs
                .ext_mirror
                .then(|| (mib / 40).max(restated_mirror_floor_mib(inputs)) * MIB);
            assert_eq!(
                (
                    eval.budgets.ext_mirror_mib.map(|m| m * MIB),
                    mirror_term(&eval.margin_terms).map(|t| t.bytes)
                ),
                (m_ext, m_ext),
                "{label}: at {mib} MiB M_ext and its margin term are charged exactly with the \
                 mirror, at max(L / 40, the floor)"
            );
            if let Some(term) = mirror_term(&eval.margin_terms) {
                assert_eq!(
                    (term.setting, term.operator_sized),
                    (Some(EXT_MIRROR_SETTINGS), true),
                    "{label}: at {mib} MiB M_ext's term names its settings"
                );
                assert!(
                    term.bytes >= mirror_charge,
                    "{label}: at {mib} MiB M_ext, {} B, does not hold one maximum-size value's \
                     {mirror_charge} B charge",
                    term.bytes
                );
            }

            for t in line.iter().chain(&margin) {
                assert!(
                    t.exact >= u128::from(t.term.bytes) * EXACT_DEN,
                    "{label}: at {mib} MiB the {} term's exact value, {} / EXACT_DEN B, is \
                     below its built {} B (C1)",
                    t.term.name,
                    t.exact,
                    t.term.bytes
                );
            }
            let (next_line, next_margin) = exact_terms_at(limit + MIB, inputs, eval.step_centi);
            for (now, next) in line.iter().zip(&next_line) {
                assert!(
                    line_shape_holds(limit, now.exact, limit + MIB, next.exact),
                    "{label}: the {} term's exact value per byte of L − floor grows from \
                     {mib} to {} MiB (C2): {} then {} / EXACT_DEN B",
                    now.term.name,
                    mib + 1,
                    now.exact,
                    next.exact
                );
            }
            let growth = exact_sum(&next_margin).saturating_sub(exact_sum(&margin));
            assert!(
                growth <= margin_growth_bound,
                "{label}: the margin's exact terms grow by {growth} / EXACT_DEN B from {mib} \
                 to {} MiB, more than a fifth of a MiB (C3)",
                mib + 1
            );

            if let Some(prev) = &previous {
                let (was, now) = (prev.budgets, eval.budgets);
                assert!(
                    now.max_subscriptions >= was.max_subscriptions
                        && now.upload_mib >= was.upload_mib
                        && now.structural_cache_mib >= was.structural_cache_mib
                        && now.retirement_mib >= was.retirement_mib
                        && now.ext_mirror_mib >= was.ext_mirror_mib,
                    "{label}: a budget fell from {} to {mib} MiB: {was:?} then {now:?}",
                    mib - 1
                );
                assert!(
                    eval.catchup_mib >= prev.catchup_mib && eval.step_centi >= prev.step_centi,
                    "{label}: T fell from {} MiB (step {} hundredths) at {} MiB to {} MiB \
                     (step {}) at {mib} MiB",
                    prev.catchup_mib,
                    prev.step_centi,
                    mib - 1,
                    eval.catchup_mib,
                    eval.step_centi
                );
            }
            previous = Some(eval);
        }
    }

    #[test]
    fn pod_memory_plan_closes_and_grows_to_64_gib_for_the_chart_configuration() {
        assert_closes_and_grows_from_its_minimum("chart", &PlanInputs::reference());
    }

    /// The chart's configuration with a checkpoint adapter: its minimum is
    /// 2,809 MiB, above the 2,709 without one because the margin then sets
    /// it, and the W4 sweep holds.
    #[test]
    fn pod_memory_plan_closes_and_grows_to_64_gib_for_the_chart_configuration_with_an_adapter() {
        assert_minimum_and_sweep(
            "chart, adapter on",
            &with_adapter(PlanInputs::reference()),
            2_809,
        );
    }

    /// The binary's defaults at each of [`BINARY_DEFAULTS`]' CPU counts: each
    /// minimum, which the line sets, is at most the table's, and the W4 sweep
    /// holds from it. Eight sweeps, about eight seconds in a debug build. The
    /// table's minimums are pinned exactly in the shipped build by
    /// [`pod_memory_plan_minimums_for_the_binary_defaults_are_the_tables_in_the_shipped_build`].
    #[test]
    fn pod_memory_plan_closes_and_grows_to_64_gib_for_the_binary_defaults() {
        for row in BINARY_DEFAULTS {
            assert_sweep_from_a_minimum_at_most(
                &format!("binary defaults, {} CPUs", row.cores),
                &binary_default_inputs(row.cores),
                row.minimum_mib,
            );
        }
    }

    /// The shipped build's minimums for the binary's defaults at each of
    /// [`BINARY_DEFAULTS`]' CPU counts, without an adapter: the table's,
    /// exactly, each set by the line.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn pod_memory_plan_minimums_for_the_binary_defaults_are_the_tables_in_the_shipped_build() {
        for row in BINARY_DEFAULTS {
            assert_shipped_minimum_set_by_the_line(
                &format!("binary defaults, {} CPUs", row.cores),
                &binary_default_inputs(row.cores),
                row.minimum_mib,
            );
        }
    }

    /// [`pod_memory_plan_closes_and_grows_to_64_gib_for_the_binary_defaults`]
    /// with a checkpoint adapter.
    #[test]
    fn pod_memory_plan_closes_and_grows_to_64_gib_for_the_binary_defaults_with_an_adapter() {
        for row in BINARY_DEFAULTS {
            assert_minimum_and_sweep(
                &format!("binary defaults, {} CPUs, adapter on", row.cores),
                &with_adapter(binary_default_inputs(row.cores)),
                row.minimum_with_adapter_mib,
            );
        }
    }

    /// `inputs`' minimum limit is `minimum` MiB, and the W4 sweep holds for
    /// it: every MiB below the minimum refuses, the refusal one MiB below
    /// names it, and every MiB from it to 64 GiB closes, `T` never
    /// decreasing. A gap row (a configuration whose closure, decided on
    /// budgets rounded down to whole MiB, had gaps: limits that refused above
    /// the first that closed) is checked with it.
    ///
    /// For a configuration whose minimum the margin sets, so `minimum` is
    /// pinned exactly in every build shape: the margin has no connection
    /// term, the one term `valkey-tls` sizes. A fixture self-check holds it
    /// to that: one MiB below the minimum, the margin does not close and the
    /// line does.
    fn assert_minimum_and_sweep(label: &str, inputs: &PlanInputs, minimum: u64) {
        assert_eq!(
            minimum_limit_mib(inputs),
            Some(minimum),
            "{label}: the minimum limit"
        );
        assert_eq!(
            sides_not_closing((minimum - 1) * MIB, inputs),
            (true, false),
            "{label}: fixture self-check: the margin alone sets the minimum, which no build \
             shape changes"
        );
        assert_closes_and_grows_from_its_minimum(label, inputs);
    }

    /// For a configuration whose minimum the line sets, in every build shape:
    /// the build's own minimum is at most `shipped_minimum`, and the W4 sweep
    /// ([`assert_closes_and_grows_from_its_minimum`]) holds from it.
    /// `shipped_minimum` is the shipped build's (the `valkey-tls` feature on,
    /// which sizes the connection pools' buffers); without that feature the
    /// buffers are smaller, so the minimum is at most it. The exact pin is
    /// [`assert_shipped_minimum_set_by_the_line`]'s, in a
    /// `#[cfg(feature = "valkey-tls")]` test.
    fn assert_sweep_from_a_minimum_at_most(label: &str, inputs: &PlanInputs, shipped_minimum: u64) {
        let build_minimum =
            minimum_limit_mib(inputs).unwrap_or_else(|| panic!("{label}: closes at some limit"));
        assert!(
            build_minimum <= shipped_minimum,
            "{label}: the build's minimum, {build_minimum} MiB, is at most the shipped build's \
             {shipped_minimum} MiB"
        );
        assert_closes_and_grows_from_its_minimum(label, inputs);
    }

    /// In the shipped build (`valkey-tls` on), `inputs`' minimum limit is
    /// `minimum` MiB exactly, set by the line: one MiB below it the line does
    /// not close and the margin does (a fixture self-check, as the line's
    /// connection term is why the pin holds in this shape only).
    #[cfg(feature = "valkey-tls")]
    fn assert_shipped_minimum_set_by_the_line(label: &str, inputs: &PlanInputs, minimum: u64) {
        assert_eq!(
            minimum_limit_mib(inputs),
            Some(minimum),
            "{label}: the minimum limit"
        );
        assert_eq!(
            sides_not_closing((minimum - 1) * MIB, inputs),
            (false, true),
            "{label}: fixture self-check: the line alone sets the minimum"
        );
    }

    /// The chart's configuration's minimum limit in the shipped build, MiB:
    /// the line sets it.
    const CHART_MINIMUM_MIB: u64 = 2_709;

    /// The shipped build's minimum for the chart's configuration, which has
    /// no adapter: [`CHART_MINIMUM_MIB`], 2,709 MiB, set by the line.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn pod_memory_plan_minimum_for_the_chart_configuration_is_2709_mib_in_the_shipped_build() {
        assert_shipped_minimum_set_by_the_line(
            "chart",
            &PlanInputs::reference(),
            CHART_MINIMUM_MIB,
        );
    }

    /// Gap rows: the chart's configuration with fewer upload waiters, each of
    /// which left a gap on rounded budgets (with 60, the plan closed at
    /// 2,699 MiB and refused at 2,700, where `C` stepped from 134 to
    /// 135 MiB), as `(waiters, the shipped build's minimum in MiB)`.
    const FEWER_UPLOAD_WAITERS: [(u64, u64); 6] = [
        (60, 2_703),
        (8, 2_623),
        (21, 2_643),
        (31, 2_658),
        (33, 2_662),
        (46, 2_681),
    ];

    /// The chart's configuration with `waiters` upload waiters.
    fn with_upload_waiters(waiters: u64) -> PlanInputs {
        PlanInputs {
            upload_waiters: waiters,
            ..PlanInputs::reference()
        }
    }

    /// [`FEWER_UPLOAD_WAITERS`]' gap rows: each closes exactly from its
    /// minimum, which the line sets, at most the row's (2,703 MiB with 60
    /// waiters, 2,623 with 8, 2,643 with 21, 2,658 with 31, 2,662 with 33,
    /// 2,681 with 46); the row's figure exactly in the shipped build
    /// ([`pod_memory_plan_minimums_for_the_chart_with_fewer_upload_waiters_in_the_shipped_build`]).
    #[test]
    fn pod_memory_plan_for_the_chart_with_fewer_upload_waiters_closes_exactly_from_its_minimum() {
        for (waiters, minimum) in FEWER_UPLOAD_WAITERS {
            assert_sweep_from_a_minimum_at_most(
                &format!("chart, {waiters} upload waiters"),
                &with_upload_waiters(waiters),
                minimum,
            );
        }
    }

    /// The shipped build's minimums for [`FEWER_UPLOAD_WAITERS`]' gap rows:
    /// each row's figure, exactly.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn pod_memory_plan_minimums_for_the_chart_with_fewer_upload_waiters_in_the_shipped_build() {
        for (waiters, minimum) in FEWER_UPLOAD_WAITERS {
            assert_shipped_minimum_set_by_the_line(
                &format!("chart, {waiters} upload waiters"),
                &with_upload_waiters(waiters),
                minimum,
            );
        }
    }

    /// Gap row: the binary's defaults at 2 CPUs with a checkpoint adapter and
    /// `LIVE_EDIT_MAX_LAYER_SIZE` = 141,400,000 B, whose minimum is
    /// 3,684 MiB. The margin sets it: the base bootstraps' 539 MiB and
    /// `M_ext`'s 135 MiB floor, one such value's charge, beside `R_ref`,
    /// which steps up a MiB every 500 MiB of limit, and the allocator slack.
    /// Closure is decided at exact shares, so `R_ref`'s steps leave no gap
    /// above the minimum.
    #[test]
    fn pod_memory_plan_for_2_cpus_with_an_adapter_and_a_raised_size_cap_closes_exactly_from_3684_mib(
    ) {
        assert_minimum_and_sweep(
            "binary defaults, 2 CPUs, adapter on, LIVE_EDIT_MAX_LAYER_SIZE=141400000",
            &PlanInputs {
                max_layer_size: 141_400_000,
                ..with_adapter(binary_default_inputs(2))
            },
            3_684,
        );
    }

    /// Gap rows: the chart's configuration at a high
    /// `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`, where `T`'s smallest step,
    /// rounded, jumped by `1 + P_f` MiB every 100 MiB of limit and left gaps
    /// of up to about 235 MiB, as `(P_f, the shipped build's minimum in MiB)`.
    const HIGH_FENCED_READ_CONCURRENCY: [(u64, u64); 2] = [(20, 10_894), (46, 60_750)];

    /// The chart's configuration at `P_f` = `p_f`.
    fn with_fenced_read_concurrency(p_f: u64) -> PlanInputs {
        PlanInputs {
            fenced_read_concurrency: p_f,
            ..PlanInputs::reference()
        }
    }

    /// [`HIGH_FENCED_READ_CONCURRENCY`]' gap rows: each closes exactly from
    /// its minimum, which the line sets, at most the row's (10,894 MiB at
    /// `P_f` 20, 60,750 at 46); the row's figure exactly in the shipped build
    /// ([`pod_memory_plan_minimums_for_the_chart_at_a_high_fenced_read_concurrency_in_the_shipped_build`]).
    #[test]
    fn pod_memory_plan_for_the_chart_at_a_high_fenced_read_concurrency_closes_exactly_from_its_minimum(
    ) {
        for (p_f, minimum) in HIGH_FENCED_READ_CONCURRENCY {
            assert_sweep_from_a_minimum_at_most(
                &format!("chart, P_f {p_f}"),
                &with_fenced_read_concurrency(p_f),
                minimum,
            );
        }
    }

    /// The shipped build's minimums for [`HIGH_FENCED_READ_CONCURRENCY`]'
    /// gap rows: each row's figure, exactly.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn pod_memory_plan_minimums_for_the_chart_at_a_high_fenced_read_concurrency_in_the_shipped_build(
    ) {
        for (p_f, minimum) in HIGH_FENCED_READ_CONCURRENCY {
            assert_shipped_minimum_set_by_the_line(
                &format!("chart, P_f {p_f}"),
                &with_fenced_read_concurrency(p_f),
                minimum,
            );
        }
    }

    /// The C2 check ([`line_shape_holds`]) tells a term that breaks the line
    /// shape from ones that keep it, at every MiB from 1 to 64 GiB: a share of
    /// `L − 2 GiB` (0 up to 2 GiB, then growing faster than its share of
    /// `L − floor`) is rejected at every step from 2 GiB up and passes below;
    /// a share of `L`, a share of `L − floor` and a constant pass everywhere.
    /// Each synthetic term is a hundredth of its base, in 1/[`EXACT_DEN`]
    /// byte. And its arithmetic is wide: exact values near 2^100 at 1 PiB,
    /// whose products overflow `u128`, are still told apart.
    #[test]
    fn line_shape_check_rejects_a_share_of_l_less_2_gib_and_passes_shares_of_l_and_l_less_floor() {
        let hundredth = |base: u64| u128::from(base) * EXACT_DEN / 100;
        let less = |x: u64| move |limit: u64| hundredth(limit.saturating_sub(x));
        let bad = less(2_048 * MIB);
        let good: [(&str, &dyn Fn(u64) -> u128); 3] = [
            ("a share of L", &|limit| hundredth(limit)),
            ("a share of L − floor", &less(POD_MEMORY_FLOOR_BYTES)),
            ("a constant", &|_| hundredth(MIB)),
        ];
        for mib in 1..SWEEP_TOP_MIB {
            let (low, high) = (mib * MIB, (mib + 1) * MIB);
            assert_eq!(
                line_shape_holds(low, bad(low), high, bad(high)),
                mib < 2_048,
                "a share of L − 2 GiB from {mib} to {} MiB",
                mib + 1
            );
            for (name, term) in good {
                assert!(
                    line_shape_holds(low, term(low), high, term(high)),
                    "{name} from {mib} to {} MiB",
                    mib + 1
                );
            }
        }

        let (low, high) = (SEARCH_CEILING_MIB * MIB, (SEARCH_CEILING_MIB + 1) * MIB);
        let big = 1u128 << 100;
        assert!(
            line_shape_holds(low, big, high, big + 1),
            "a byte more over a MiB more of limit keeps the shape"
        );
        assert!(
            !line_shape_holds(low, big, high, big + (1 << 90)),
            "2^90 / EXACT_DEN B more over a MiB more of limit at 1 PiB breaks it"
        );
    }

    /// The defect error: at or above the minimum the plan closes by the
    /// exact-share contract, so if it ever did not, `derive` returns an
    /// error, not a panic, saying the plan has a defect, naming the contract
    /// and this file, and asking for a report; not a refusal telling the
    /// operator to raise the limit. Reached through `derive_against` with a
    /// minimum, 1,000 MiB, below the chart configuration's real one, at
    /// 2,000 MiB, where its plan does not close: what a term breaking the
    /// contract would make `minimum_limit_mib` report.
    #[test]
    fn derive_at_or_above_a_minimum_where_the_plan_does_not_close_returns_the_defect_error() {
        let inputs = PlanInputs::reference();
        assert!(
            Evaluation::new(2_000 * MIB, &inputs).is_none(),
            "fixture self-check: the plan does not close at 2,000 MiB"
        );
        let err = PodMemoryPlan::derive_against(limit_of(2_000), &inputs, Some(1_000))
            .expect_err("a plan that does not close never starts");
        for part in [
            "pod memory plan defect: the plan does not close at this pod's memory limit, \
             2,000 MiB (cgroup v2), at or above the 1,000 MiB minimum it reports",
            "breaks the exact-share contract (C1 to C3, src/config/memory_plan.rs",
            "this is a defect in the service, not in its configuration: report it",
        ] {
            assert!(err.contains(part), "{part:?} in {err}");
        }
        assert!(
            !err.contains("raise resources.limits.memory"),
            "a defect is not a refusal: {err}"
        );
    }

    /// The checkpoint base bootstrap's margin term is charged exactly when an
    /// adapter can run it: 2 × `BOOTSTRAP_MAX_INFLIGHT` maximum-size layers,
    /// 400 MiB at the default size cap, and nothing without one.
    #[test]
    fn pod_memory_plan_charges_the_base_bootstraps_to_the_margin_only_with_an_adapter() {
        let term = |inputs: &PlanInputs| {
            derive_at(5_120, inputs)
                .expect("closes at 5 GiB")
                .margin_terms()
                .iter()
                .find(|t| t.name == "checkpoint base bootstraps")
                .map(|t| t.bytes)
        };
        assert_eq!(term(&PlanInputs::reference()), None);
        assert_eq!(
            term(&with_adapter(PlanInputs::reference())),
            Some(400 * MIB)
        );
    }

    /// Limits that are not a whole number of MiB, as a decimal-suffix
    /// quantity (`M`, `G`) or a byte count gives: refusal compares the limit
    /// in whole MiB, rounded down, with the minimum, and the plan closes at
    /// every byte from the minimum up, not only at whole MiB. For the chart's
    /// configuration (at the build's own minimum: at most
    /// [`CHART_MINIMUM_MIB`], exactly that in the shipped build, as
    /// [`pod_memory_plan_minimum_for_the_chart_configuration_is_2709_mib_in_the_shipped_build`]
    /// pins): one byte below the minimum refuses,
    /// naming it, and is not the defect error; half a MiB above closes; so
    /// does `3G` (3,000,000,000 B); and at every 4 KiB page from one MiB below
    /// the minimum to one MiB above it, the pages below the minimum refuse and
    /// the rest close, the built line within 80 % of the limit and the margin
    /// within the rest.
    #[test]
    fn pod_memory_plan_at_limits_between_whole_mib_refuses_below_the_minimum_and_closes_from_it() {
        const PAGE: u64 = 4 * 1024;
        let inputs = PlanInputs::reference();
        let minimum = minimum_limit_mib(&inputs).expect("the chart configuration closes");
        assert!(
            minimum <= CHART_MINIMUM_MIB,
            "fixture self-check: the chart's minimum, {minimum} MiB, is at most the shipped \
             build's {CHART_MINIMUM_MIB} MiB"
        );
        let at =
            |bytes: u64| PodMemoryPlan::derive(PodMemorySource::Fixed(bytes).resolve(), &inputs);
        let refuses = |bytes: u64| {
            let err = at(bytes).expect_err("below the minimum the plan refuses");
            assert!(
                err.contains(&format!(
                    "raise resources.limits.memory to at least {} MiB",
                    grouped(minimum)
                )) && !err.contains("pod memory plan defect"),
                "{bytes} B: a refusal naming the minimum, not the defect error: {err}"
            );
        };
        let closes = |bytes: u64| {
            let plan = at(bytes).unwrap_or_else(|e| panic!("{bytes} B: the plan closes: {e}"));
            assert_eq!(plan.limit().bytes, bytes);
            assert_eq!(plan.minimum_limit_mib(), Some(minimum), "{bytes} B");
            assert!(
                plan.line_total_bytes() <= plan.line_limit_bytes()
                    && plan.margin_named_bytes() <= bytes - plan.line_limit_bytes(),
                "{bytes} B: the built line and margin fit"
            );
        };

        refuses(minimum * MIB - 1);
        closes(minimum * MIB + MIB / 2);
        closes(3_000_000_000);
        let pages = ((minimum - 1) * MIB..=(minimum + 1) * MIB).step_by(PAGE as usize);
        assert_eq!(
            pages.clone().count(),
            513,
            "fixture self-check: two MiB of 4 KiB pages, both ends"
        );
        for bytes in pages {
            if bytes < minimum * MIB {
                refuses(bytes);
            } else {
                closes(bytes);
            }
        }
    }

    /// The chart configuration's exact line at 4,096 MiB with `T`'s term at
    /// a step of `step` hundredths, in 1/[`EXACT_DEN`] byte, and `T` there,
    /// MiB.
    fn chart_line_at_4096_mib(step: u64) -> (u128, u64) {
        let inputs = PlanInputs::reference();
        let (line, _) = exact_terms_at(4_096 * MIB, &inputs, step);
        let t = line
            .iter()
            .find(|t| t.term.name == "catch-up memory budget")
            .expect("the line carries T's term")
            .term
            .bytes
            / (1 + inputs.fenced_read_concurrency)
            / MIB;
        (exact_sum(&line), t)
    }

    /// The line's allowance at 4,096 MiB, 80 % of it, in 1/[`EXACT_DEN`]
    /// byte.
    fn line_allowance_at_4096_mib() -> u128 {
        LINE_ALLOWANCE.exact_units_scaled(4_096 * MIB, 0)
    }

    /// W5, in every build shape: at 4,096 MiB the chart configuration's `T`
    /// is at the largest 0.01 step of `L − floor` at which the line fits at
    /// exact shares: the line, `T × (1 + P_f)` included, fits 80 % of the
    /// limit at the plan's step, and not at the next. The step's figures
    /// depend on the build (`valkey-tls` sizes the line's connection term,
    /// so without it `T` may take a larger step) and are pinned in the
    /// shipped build by
    /// [`catchup_budget_at_4096_mib_is_433_mib_at_the_0_11_step_in_the_shipped_build`].
    #[test]
    fn catchup_budget_at_4096_mib_is_at_the_largest_0_01_step_that_fits() {
        let plan = derive_at(4_096, &PlanInputs::reference())
            .expect("the chart configuration closes at 4 GiB");
        let step = plan.catchup_step_centi();
        let allowance = line_allowance_at_4096_mib();

        let (fits, t) = chart_line_at_4096_mib(step);
        assert_eq!(t, plan.catchup_budget_mib(), "T at the plan's step");
        assert!(
            fits <= allowance,
            "the plan's step, {step} hundredths, must fit at exact shares"
        );
        assert!(
            step < 100,
            "fixture self-check: a next step exists at 4 GiB"
        );
        let (over, _) = chart_line_at_4096_mib(step + 1);
        assert!(
            over > allowance,
            "the next step, {} hundredths, must not fit at exact shares, or {step} is not the \
             largest",
            step + 1
        );
    }

    /// W5's figures, in the shipped build (`valkey-tls` on): at 4,096 MiB
    /// the chart configuration's `T` is 433 MiB, at a step of 0.11: there
    /// the exact line, `T × (1 + P_f)` included, leaves 23.4 MiB of 80 % of
    /// the limit; at the next step (0.12, `T` 472 MiB) it is 55.5 MiB over.
    /// The 0.05 rule's largest step there, 0.10, gives 394.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn catchup_budget_at_4096_mib_is_433_mib_at_the_0_11_step_in_the_shipped_build() {
        let plan = derive_at(4_096, &PlanInputs::reference())
            .expect("the chart configuration closes at 4 GiB");
        assert_eq!(
            (plan.catchup_budget_mib(), plan.catchup_step_centi()),
            (433, 11),
            "T at 4,096 MiB"
        );
        let allowance = line_allowance_at_4096_mib();
        // An exact difference, in 1/EXACT_DEN byte, in MiB to 0.1.
        let mib_1dp =
            |exact: u128| (exact as f64 / (EXACT_DEN as f64 * MIB as f64) * 10.0).round() / 10.0;

        let (fits, t) = chart_line_at_4096_mib(11);
        assert_eq!(t, 433, "T at 0.11");
        let room = allowance
            .checked_sub(fits)
            .expect("0.11 fits the line's allowance");
        assert_eq!(mib_1dp(room), 23.4, "the room 0.11 leaves");
        let (over, next_t) = chart_line_at_4096_mib(12);
        assert_eq!(next_t, 472, "T at 0.12");
        let excess = over
            .checked_sub(allowance)
            .expect("0.12 is over the line's allowance");
        assert_eq!(mib_1dp(excess), 55.5, "what 0.12 is over by");
    }

    /// W6: below the minimum the refusal names the limit and its source, the
    /// smallest limit the configuration needs, the shortfall and the largest
    /// terms an operator setting sizes, largest first, with their settings.
    /// At the chart's configuration, with no backup store, the line sets the
    /// minimum, so those are the line's and nothing of the margin is named.
    #[test]
    fn refusal_below_the_minimum_names_the_minimum_the_shortfall_and_the_largest_terms() {
        let inputs = PlanInputs::reference();
        let minimum = minimum_limit_mib(&inputs).expect("the chart configuration closes");
        assert!(
            minimum > 2_048,
            "fixture self-check: 2 GiB is below {minimum}"
        );
        let err = derive_at(2_048, &inputs).expect_err("2 GiB is below the minimum");
        for part in [
            "this pod's memory limit, 2,048 MiB (cgroup v2), is below the".to_owned(),
            format!(
                "{} MiB this configuration needs, {} MiB short",
                grouped(minimum),
                grouped(minimum - 2_048)
            ),
            "the largest fixed terms: one maximum-size upload 409 MiB (LIVE_EDIT_MAX_LAYER_SIZE), \
             stream readers 256 MiB (LIVE_EDIT_SUBSCRIBER_POOL_SIZE), catch-up slots 194 MiB \
             (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS), connection pools "
                .to_owned(),
            // The pools' term names every pool setting it charges.
            "MiB (LIVE_EDIT_VALKEY_POOL_SIZE, LIVE_EDIT_MAINTENANCE_POOL_SIZE, \
             LIVE_EDIT_SNAPSHOT_POOL_SIZE, LIVE_EDIT_SUBSCRIBER_POOL_SIZE); raise"
                .to_owned(),
            format!(
                "raise resources.limits.memory to at least {} MiB",
                grouped(minimum)
            ),
        ] {
            assert!(err.contains(&part), "the refusal carries {part:?}: {err}");
        }
        // Without a backup store the margin names no operator-sized term and
        // closes one MiB below the minimum: only the line is named, with the
        // line's remedy.
        assert!(
            err.ends_with(&format!(
                "raise resources.limits.memory to at least {} MiB or lower those settings",
                grouped(minimum)
            )),
            "the line's remedy: {err}"
        );
        // Nor the margin's remedy: neither its tail clause nor its warning
        // against turning backups off.
        for absent in [
            "20 % memory margin",
            KEEP_BACKUPS,
            "turn off incremental backups",
        ] {
            assert!(!err.contains(absent), "{absent:?} is not named: {err}");
        }
    }

    /// W6, the configuration no limit can fit for its fixed settings rather
    /// than `P_f`: with every catch-up slot the semaphore allows, even 1 PiB
    /// cannot carry their memory at `P_f` = 1. The refusal says no limit
    /// fits, names the catch-up slots as the largest term, tells the operator
    /// to lower those settings, and names neither `P_f` nor a minimum limit.
    #[test]
    fn refusal_when_fixed_settings_fit_no_limit_names_the_largest_terms_and_not_p_f() {
        let inputs = PlanInputs {
            catchup_slots: tokio::sync::Semaphore::MAX_PERMITS as u64,
            ..PlanInputs::reference()
        };
        assert_eq!(minimum_limit_mib(&inputs), None, "no limit fits");
        let err = derive_at(5_120, &inputs).expect_err("no limit fits");
        for part in [
            "no pod memory limit can fit this configuration",
            "the largest fixed terms: catch-up slots ",
            "(LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS)",
            "lower those settings",
        ] {
            assert!(err.contains(part), "{part:?} in {err}");
        }
        for absent in [
            "LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY",
            "at least",
            "this configuration needs",
        ] {
            assert!(!err.contains(absent), "{absent:?} not in {err}");
        }
    }

    /// An operator setting the plan charges, with the term it sizes.
    struct Knob {
        label: &'static str,
        /// The term's name in the plan.
        term: &'static str,
        /// The setting the refusal names beside the term, when one does.
        setting: Option<&'static str>,
        /// The chart's value, in the knob's unit.
        chart: u64,
        set: fn(&mut PlanInputs, u64),
    }

    const KNOBS: [Knob; 4] = [
        Knob {
            label: "catch-up slots",
            term: "catch-up slots",
            setting: Some("LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS"),
            chart: 8,
            set: |inputs, n| inputs.catchup_slots = n,
        },
        Knob {
            label: "data and txn pool size",
            term: "connection pools",
            setting: Some(CONNECTION_POOL_SETTINGS),
            chart: 64,
            set: |inputs, n| {
                inputs.pools.data = n as usize;
                inputs.pools.txn = n as usize;
            },
        },
        Knob {
            label: "upload waiters",
            term: "queued uploads",
            setting: Some("LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS"),
            chart: 64,
            set: |inputs, n| inputs.upload_waiters = n,
        },
        Knob {
            label: "upload size cap, MiB",
            term: "one maximum-size upload",
            setting: Some("LIVE_EDIT_MAX_LAYER_SIZE"),
            chart: 100,
            set: |inputs, n| inputs.max_layer_size = n * MIB,
        },
    ];

    /// The limit, in MiB, a "raise resources.limits.memory to at least N MiB"
    /// refusal names.
    fn named_minimum(refusal: &str) -> u64 {
        let rest = refusal
            .split("raise resources.limits.memory to at least ")
            .nth(1)
            .unwrap_or_else(|| panic!("the refusal names a minimum: {refusal}"));
        rest.split(" MiB")
            .next()
            .expect("a number before MiB")
            .replace(',', "")
            .parse()
            .unwrap_or_else(|e| panic!("the minimum is a number ({e}): {refusal}"))
    }

    /// W6: raising a pool, slot, waiter or upload-size setting is paid from
    /// `T` while the plan closes at the reference limit, so just inside
    /// closure `T` is smaller than at the chart's value; one past it, startup
    /// is refused, naming the term the setting sizes as the largest, and the
    /// smallest limit that configuration needs, which is exactly the first
    /// limit at which it closes.
    #[test]
    fn raising_a_setting_past_closure_refuses_naming_the_minimum_and_just_inside_t_shrinks() {
        let reference_limit = super::super::pod_memory::REFERENCE_POD_MEMORY_LIMIT_MIB;
        for knob in KNOBS {
            let label = knob.label;
            let at = |n: u64| {
                let mut inputs = PlanInputs::reference();
                (knob.set)(&mut inputs, n);
                inputs
            };
            let chart_t = derive_at(reference_limit, &at(knob.chart))
                .expect("the chart's value closes")
                .catchup_budget_mib();
            let mut inside = knob.chart;
            while Evaluation::new(reference_limit * MIB, &at(inside + 1)).is_some() {
                inside += 1;
            }
            assert!(
                inside > knob.chart,
                "{label}: the reference pod has room above the chart's {}",
                knob.chart
            );
            let inside_t = derive_at(reference_limit, &at(inside))
                .expect("just inside closure")
                .catchup_budget_mib();
            assert!(
                inside_t < chart_t,
                "{label}: at {inside} T is {inside_t} MiB, not below the chart's {chart_t} MiB"
            );

            let past = at(inside + 1);
            let err = derive_at(reference_limit, &past).expect_err("one past closure");
            let largest = format!("the largest fixed terms: {} ", knob.term);
            assert!(
                err.contains(&largest),
                "{label}: {} is the largest term named: {err}",
                knob.term
            );
            if let Some(setting) = knob.setting {
                assert!(err.contains(&format!("({setting})")), "{label}: {err}");
            }
            let minimum = named_minimum(&err);
            assert!(minimum > reference_limit, "{label}: {err}");
            assert!(
                Evaluation::new(minimum * MIB, &past).is_some()
                    && Evaluation::new((minimum - 1) * MIB, &past).is_none(),
                "{label}: {minimum} MiB is the first limit at which {} closes",
                inside + 1
            );
        }
    }

    /// W6, the configuration no limit can fit: `T × (1 + P_f)` at its
    /// smallest step grows with the limit faster than the line leaves room
    /// for it once `P_f` passes 61 at the chart's other settings. The refusal
    /// says no limit fits, names the setting and the largest value that does,
    /// and names no minimum limit; 61 itself closes.
    #[test]
    fn fenced_read_concurrency_no_limit_can_fit_names_the_largest_that_fits_and_no_minimum() {
        let at = |p_f: u64| PlanInputs {
            fenced_read_concurrency: p_f,
            ..PlanInputs::reference()
        };
        let fits = minimum_limit_mib(&at(61)).expect("P_f = 61 closes at some limit");
        derive_at(fits, &at(61)).expect("and at its minimum");
        for p_f in [62, 1_000] {
            assert_eq!(minimum_limit_mib(&at(p_f)), None, "P_f = {p_f}");
            let err = derive_at(5_120, &at(p_f)).expect_err("no limit fits");
            for part in [
                "no pod memory limit can fit this configuration".to_owned(),
                format!("LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY={p_f}"),
                "lower LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY to at most 61".to_owned(),
            ] {
                assert!(err.contains(&part), "P_f = {p_f}: {part:?} in {err}");
            }
            for bogus in ["at least", "this configuration needs", "MiB short"] {
                assert!(
                    !err.contains(bogus),
                    "P_f = {p_f}: no limit fits, so no minimum is named ({bogus:?}): {err}"
                );
            }
        }
    }

    /// The constants a `Config` built field by field, rather than derived
    /// from a limit, carries are the reference plan's budgets: so a test
    /// config and a pod with no readable limit agree.
    #[test]
    fn reference_constants_are_the_reference_plans_budgets() {
        let plan = PodMemoryPlan::reference();
        assert_eq!(plan.limit(), &PodMemoryLimit::reference(None));
        assert_eq!(
            super::super::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            plan.catchup_budget_mib()
        );
        assert_eq!(
            super::super::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            plan.structural_cache_budget_mib()
        );
        assert_eq!(
            crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            plan.upload_budget_mib()
        );
        assert_eq!(
            SUBSCRIPTION_CEILING_AT_REFERENCE_LIMIT as u64,
            plan.max_subscriptions(),
            "the validation-records guard reads this constant as the shipped subscription \
             ceiling"
        );
    }

    /// `upload_memory_budget_mib_default`, documented as the plan's `B` rule
    /// at the reference limit, is the plan's `B` there for small, default and
    /// raised size caps (the floor binding past about 188 MiB).
    #[test]
    fn upload_budget_default_for_a_size_cap_is_the_plans_b_at_the_reference_limit() {
        for cap_mib in [10, 100, 188, 200, 1_024] {
            let inputs = PlanInputs {
                max_layer_size: cap_mib * MIB,
                ..PlanInputs::reference()
            };
            let budgets = budgets_at(REFERENCE_LIMIT_BYTES, &inputs);
            assert_eq!(
                crate::grpc::upload_admission::upload_memory_budget_mib_default(cap_mib * MIB),
                budgets.upload_mib,
                "cap {cap_mib} MiB"
            );
        }
    }

    // ── The external-value mirror's term, `M_ext` (b.mhq) ─────────────────

    /// `M_ext`'s margin term, by name.
    const MIRROR_TERM: &str = "backup copies of external values";

    /// `M_ext`'s floor, MiB, restated from its rule rather than read from
    /// the code: one maximum-size value's mirror charge — its size rounded
    /// up to a KiB, plus the one-KiB per-item allowance — rounded up to a
    /// whole MiB.
    fn restated_mirror_floor_mib(inputs: &PlanInputs) -> u64 {
        (inputs.max_layer_size.div_ceil(1024) + 1).div_ceil(1024)
    }

    /// `M_ext`'s term among a plan's margin `terms`, if it charges one.
    fn mirror_term(terms: &[PlanTerm]) -> Option<&PlanTerm> {
        terms.iter().find(|t| t.name == MIRROR_TERM)
    }

    /// T12 (a) and (b): with a backup store, `M_ext` is `max(⌊L/40⌋, the
    /// floor)` MiB at every limit from the configuration's minimum to 64 GiB,
    /// never falling as `L` grows, and it always holds the charge the mirror
    /// takes for one maximum-size value (`item_charge_kib` of the size cap),
    /// so every value under the upload cap fits the budget; its margin term
    /// carries that budget, named with the settings that switch it on and
    /// size its floor. The W4 sweep checks all of that at every MiB
    /// ([`assert_closes_and_grows_from_its_minimum`]) for the chart's
    /// configuration and the binary's defaults with an adapter; here it runs
    /// with the size cap raised to 150 MiB, whose floor (151 MiB) is past the
    /// default cap's, with its minimum, 4,071 MiB, which the margin sets. At
    /// 4, 5 and 8 GiB `M_ext` is 102, 128 and 204 MiB; from the chart's
    /// minimum, 2,809 MiB, to 4,079 MiB it is the 101 MiB floor.
    #[test]
    fn ext_mirror_budget_is_a_fortieth_of_the_limit_floored_at_one_maximum_size_values_charge() {
        let raised_cap = PlanInputs {
            max_layer_size: 150 * MIB,
            ..with_adapter(PlanInputs::reference())
        };
        assert_eq!(restated_mirror_floor_mib(&raised_cap), 151);
        assert_minimum_and_sweep("chart, adapter on, 150 MiB size cap", &raised_cap, 4_071);

        let chart = with_adapter(PlanInputs::reference());
        assert_eq!(
            restated_mirror_floor_mib(&chart),
            101,
            "the floor at the default 100 MiB size cap"
        );
        for (mib, m_ext) in [
            (2_809, 101),
            (4_079, 101),
            (4_080, 102),
            (4_096, 102),
            (5_120, 128),
            (8_192, 204),
        ] {
            assert_eq!(
                derive_at(mib, &chart)
                    .unwrap_or_else(|e| panic!("{mib} MiB closes: {e}"))
                    .ext_mirror_bytes(),
                Some(m_ext * MIB),
                "M_ext at {mib} MiB"
            );
        }
    }

    /// T12 (c), on the plan's inputs: `M_ext` is charged exactly when the
    /// mirror is wired (`PlanInputs::ext_mirror`), whether or not the base
    /// bootstraps are: then `ext_mirror_bytes` is `Some`, the margin carries
    /// its term, and the margin's named bytes are exactly its term above the
    /// same plan without the mirror; otherwise neither. The reference plan,
    /// which has no adapter, funds no mirror.
    #[test]
    fn ext_mirror_term_is_charged_exactly_when_the_mirror_is_wired() {
        for base_bootstrap in [false, true] {
            let without = derive_at(
                5_120,
                &PlanInputs {
                    base_bootstrap,
                    ext_mirror: false,
                    ..PlanInputs::reference()
                },
            )
            .expect("closes at 5 GiB without the mirror");
            let with = derive_at(
                5_120,
                &PlanInputs {
                    base_bootstrap,
                    ext_mirror: true,
                    ..PlanInputs::reference()
                },
            )
            .expect("closes at 5 GiB with the mirror");
            let case = format!("base_bootstrap={base_bootstrap}");
            assert_eq!(without.ext_mirror_bytes(), None, "{case}");
            assert_eq!(mirror_term(without.margin_terms()), None, "{case}");
            assert_eq!(with.ext_mirror_bytes(), Some(128 * MIB), "{case}");
            assert_eq!(
                mirror_term(with.margin_terms()).map(|t| t.bytes),
                Some(128 * MIB),
                "{case}"
            );
            for plan in [&without, &with] {
                assert_eq!(
                    u128::from(plan.margin_named_bytes()),
                    sum(plan.margin_terms()),
                    "{case}: the margin's named bytes are its terms'"
                );
            }
            assert_eq!(
                with.margin_named_bytes() - without.margin_named_bytes(),
                128 * MIB,
                "{case}: the mirror adds exactly its term to the margin"
            );
            assert_eq!(
                with.line_terms(),
                without.line_terms(),
                "{case}: a margin term leaves the line alone"
            );
        }
        assert_eq!(PodMemoryPlan::reference().ext_mirror_bytes(), None);
    }

    /// The text a margin-bound refusal names the margin's operator-sized
    /// terms with, at the default size cap and below 4,080 MiB, where `M_ext`
    /// is its 101 MiB floor.
    const MARGIN_TERMS_AT_THE_DEFAULT_CAP: &str =
        "the terms of the 20 % memory margin that settings size: checkpoint base bootstraps \
         400 MiB (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED, LIVE_EDIT_CHECKPOINT_ENABLED, \
         LIVE_EDIT_CHECKPOINT_ADAPTER, LIVE_EDIT_MAX_LAYER_SIZE), backup copies of external \
         values 101 MiB (LIVE_EDIT_CHECKPOINT_ENABLED, LIVE_EDIT_CHECKPOINT_ADAPTER, \
         LIVE_EDIT_MAX_LAYER_SIZE); ";

    /// The margin's remedy that ends a refusal naming the margin's terms
    /// while the base bootstraps are charged: never turning off checkpointing
    /// or the adapter.
    const MARGIN_REMEDY: &str = "lower LIVE_EDIT_MAX_LAYER_SIZE, or turn off incremental backups \
         (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false), though recovery points then come only from \
         full backup passes, so a newly created or just-restored layer has no backup of its \
         current state until the next pass; do not turn off LIVE_EDIT_CHECKPOINT_ENABLED or \
         LIVE_EDIT_CHECKPOINT_ADAPTER for this: that removes backups";

    /// What follows the limit in a refusal naming the margin's terms, with a
    /// minimum: raising the limit comes first, and the margin's remedy
    /// ([`MARGIN_REMEDY`], after any [`BOTH_SIDES_LOWER`]) follows this.
    const MARGIN_REMEDY_TAIL: &str = ", which costs nothing else; failing that, ";

    /// What a refusal naming both sides asks to lower before the margin's
    /// remedy.
    const BOTH_SIDES_LOWER: &str = "lower the largest fixed terms' settings and, for the margin, ";

    /// The terms of the line a refusal names when the line sets the minimum.
    const LINE_TERM_NAMES: [&str; 6] = [
        "the largest fixed terms",
        "one maximum-size upload",
        "stream readers",
        "catch-up slots",
        "connection pools",
        "queued uploads",
    ];

    /// Decider ruling 1 and T12 (d): with a backup store on at the chart's
    /// configuration the margin sets the minimum, 2,809 MiB: the line
    /// still closes one MiB below it and the margin does not. 2,808 MiB is
    /// refused naming the margin's operator-sized terms — the base
    /// bootstraps and the mirror's `M_ext`, each with the settings that
    /// switch it on and size it — and the margin's remedy, raising the limit
    /// first and never turning off checkpointing or the adapter, and names
    /// no line term, as lowering a line setting
    /// would not lower the minimum. The same limit closes with the backup
    /// store off.
    #[test]
    fn refusal_when_the_margin_sets_the_minimum_names_its_checkpoint_terms_and_no_line_term() {
        let inputs = with_adapter(PlanInputs::reference());
        assert_eq!(minimum_limit_mib(&inputs), Some(2_809));
        assert_eq!(
            sides_not_closing(2_808 * MIB, &inputs),
            (true, false),
            "fixture self-check: one MiB below the minimum only the margin does not close"
        );
        let err = derive_at(2_808, &inputs).expect_err("one MiB below the minimum");
        assert_eq!(
            err,
            format!(
                "this pod's memory limit, 2,808 MiB (cgroup v2), is below the 2,809 MiB this \
                 configuration needs, 1 MiB short; {MARGIN_TERMS_AT_THE_DEFAULT_CAP}raise \
                 resources.limits.memory to at least 2,809 MiB{MARGIN_REMEDY_TAIL}{MARGIN_REMEDY}"
            ),
            "the margin-bound refusal"
        );
        for line_term in LINE_TERM_NAMES {
            assert!(
                !err.contains(line_term),
                "{line_term:?} is not named: {err}"
            );
        }
        derive_at(2_808, &PlanInputs::reference())
            .expect("the same limit closes with the backup store off");
    }

    /// Decider ruling 1: when both sides set the minimum — neither closes
    /// one MiB below it — the refusal names the line's largest terms and
    /// the margin's, and raising the limit first, then lowering both sides'
    /// settings, with the margin's remedy, which keeps backups. The
    /// configuration is the
    /// chart's with a backup store and the byte ceiling raised just far
    /// enough that the line's own minimum reaches the margin's: found by
    /// search ([`both_sides_set_the_minimum`]), as the line's minimum
    /// depends on the build (`valkey-tls` sizes the connection buffers) and
    /// the margin's does not.
    #[test]
    fn refusal_when_both_sides_set_the_minimum_names_the_lines_largest_terms_and_the_margins() {
        let (inputs, minimum) = both_sides_set_the_minimum();
        let err = derive_at(minimum - 1, &inputs).expect_err("one MiB below the minimum");
        let line_at = err
            .find("; the largest fixed terms: one maximum-size upload 409 MiB")
            .unwrap_or_else(|| panic!("the line's largest terms are named: {err}"));
        let margin_at = err
            .find(&format!("; {MARGIN_TERMS_AT_THE_DEFAULT_CAP}"))
            .unwrap_or_else(|| panic!("the margin's terms are named: {err}"));
        assert!(
            line_at < margin_at,
            "the line's terms, then the margin's: {err}"
        );
        assert!(
            err.ends_with(&format!(
                "raise resources.limits.memory to at least {} MiB{MARGIN_REMEDY_TAIL}\
                 {BOTH_SIDES_LOWER}{MARGIN_REMEDY}",
                grouped(minimum)
            )),
            "raising the limit, then both sides' remedy, the margin's keeping backups: {err}"
        );
    }

    /// The inputs of the refusal test above, with their minimum, MiB: the
    /// chart's configuration with a backup store and the byte ceiling raised
    /// just far enough that the line's own minimum reaches the margin's, so
    /// neither side closes one MiB below it.
    fn both_sides_set_the_minimum() -> (PlanInputs, u64) {
        let at_ceiling = |byte_ceiling: u64| PlanInputs {
            byte_ceiling,
            ..PlanInputs::reference()
        };
        let line_minimum = |byte_ceiling: u64| {
            minimum_limit_mib(&at_ceiling(byte_ceiling)).expect("the line closes at some limit")
        };
        let margin_minimum =
            minimum_limit_mib(&with_adapter(PlanInputs::reference())).expect("closes");
        // The smallest ceiling at which the line alone needs the margin's
        // minimum: the line's minimum grows by at most a MiB per byte of
        // ceiling, so it meets the margin's exactly.
        let (mut below, mut at) = (PlanInputs::reference().byte_ceiling, 4 << 20);
        assert!(line_minimum(below) < margin_minimum && line_minimum(at) >= margin_minimum);
        while at - below > 1 {
            let mid = below + (at - below) / 2;
            if line_minimum(mid) < margin_minimum {
                below = mid;
            } else {
                at = mid;
            }
        }
        assert_eq!(
            line_minimum(at),
            margin_minimum,
            "fixture self-check: the line's minimum meets the margin's"
        );
        let inputs = with_adapter(at_ceiling(at));
        let minimum = minimum_limit_mib(&inputs).expect("closes");
        assert_eq!(minimum, margin_minimum);
        assert_eq!(
            sides_not_closing((minimum - 1) * MIB, &inputs),
            (true, true),
            "fixture self-check: one MiB below the minimum neither side closes"
        );
        (inputs, minimum)
    }

    /// Decider ruling 1: with a backup store on, a configuration whose line
    /// needs more than the margin — the chart's with 200 upload waiters —
    /// is refused one MiB below its minimum naming only the line's largest
    /// terms, with the line's remedy: the margin closes there, so its
    /// checkpoint terms do not set the minimum and are not named.
    #[test]
    fn refusal_when_the_line_sets_the_minimum_with_a_backup_store_names_only_the_line() {
        let inputs = PlanInputs {
            upload_waiters: 200,
            ..with_adapter(PlanInputs::reference())
        };
        let minimum = minimum_limit_mib(&inputs).expect("closes");
        assert_eq!(
            sides_not_closing((minimum - 1) * MIB, &inputs),
            (false, true),
            "fixture self-check: one MiB below the minimum only the line does not close"
        );
        let err = derive_at(minimum - 1, &inputs).expect_err("one MiB below the minimum");
        for part in [
            "; the largest fixed terms: one maximum-size upload ",
            ", queued uploads 200 MiB (LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS), ",
        ] {
            assert!(
                err.contains(part),
                "the line's largest terms: {part:?} in {err}"
            );
        }
        assert!(
            err.ends_with(&format!(
                "raise resources.limits.memory to at least {} MiB or lower those settings",
                grouped(minimum)
            )),
            "the line's remedy: {err}"
        );
        // Nor the margin's remedy, though the backup store and its tail are
        // on: neither the tail clause nor the warning against turning
        // backups off.
        for absent in [
            "20 % memory margin",
            MIRROR_TERM,
            "checkpoint base bootstraps",
            KEEP_BACKUPS,
            "turn off incremental backups",
        ] {
            assert!(!err.contains(absent), "{absent:?} is not named: {err}");
        }
    }

    /// The refusal no limit can fit, when neither side closes at 1 PiB: a
    /// backup store with a size cap of 1 PiB. It names the line's largest
    /// terms and the margin's checkpoint terms, then lowering both sides'
    /// settings with the margin's remedy, which keeps backups, and no
    /// minimum.
    #[test]
    fn refusal_when_no_limit_fits_either_side_names_both_sides_and_the_margins_remedy() {
        let inputs = PlanInputs {
            max_layer_size: 1 << 50,
            ..with_adapter(PlanInputs::reference())
        };
        assert_eq!(minimum_limit_mib(&inputs), None, "no limit fits");
        assert_eq!(
            sides_not_closing(SEARCH_CEILING_MIB * MIB, &inputs),
            (true, true),
            "fixture self-check: neither side closes at 1 PiB"
        );
        let err = derive_at(5_120, &inputs).expect_err("no limit fits");
        for part in [
            "no pod memory limit can fit this configuration",
            "; the largest fixed terms: one maximum-size upload ",
            "; the terms of the 20 % memory margin that settings size: checkpoint base \
             bootstraps ",
            ", backup copies of external values ",
        ] {
            assert!(err.contains(part), "{part:?} in {err}");
        }
        assert!(
            err.ends_with(&format!("; {BOTH_SIDES_LOWER}{MARGIN_REMEDY}")),
            "both sides' remedy, the margin's keeping backups: {err}"
        );
        for absent in ["at least", "this configuration needs"] {
            assert!(!err.contains(absent), "{absent:?} not in {err}");
        }
    }

    /// A configuration with a backup store and checkpointing on but the
    /// tail off (`LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false`), at a size cap of
    /// `max_layer_size` bytes: the mirror's `M_ext` is charged, the base
    /// bootstraps are not.
    fn tail_off(max_layer_size: u64) -> PlanInputs {
        PlanInputs {
            max_layer_size,
            base_bootstrap: false,
            ext_mirror: true,
            ..PlanInputs::reference()
        }
    }

    /// The margin's remedy with the tail off: `M_ext` is the margin's only
    /// operator-sized term, so the remedy is lowering
    /// `LIVE_EDIT_MAX_LAYER_SIZE`, which sizes it, then [`KEEP_BACKUPS`],
    /// with no tail clause, the tail being off already. With the tail off
    /// the line needs more for each byte of `LIVE_EDIT_MAX_LAYER_SIZE` than
    /// the margin does (the upload budget's floor and one upload's parse
    /// charge, against `M_ext`'s one value), so the margin is named only
    /// beside the line, where no limit fits: here a 1 PiB size cap. The
    /// fixture self-check pins the first half at the default size cap and
    /// two raised ones: there the line alone sets the minimum.
    #[test]
    fn refusal_naming_the_margin_with_the_tail_off_offers_a_lower_size_cap_and_keeps_backups() {
        for cap_mib in [100, 1_024, 64 * 1_024] {
            let inputs = tail_off(cap_mib * MIB);
            let minimum = minimum_limit_mib(&inputs).expect("closes");
            assert_eq!(
                sides_not_closing((minimum - 1) * MIB, &inputs),
                (false, true),
                "fixture self-check: with the tail off and a {cap_mib} MiB size cap the line \
                 alone sets the minimum"
            );
        }
        let inputs = tail_off(1 << 50);
        assert_eq!(minimum_limit_mib(&inputs), None, "no limit fits");
        assert_eq!(
            sides_not_closing(SEARCH_CEILING_MIB * MIB, &inputs),
            (true, true),
            "fixture self-check: neither side closes at 1 PiB"
        );
        let err = derive_at(5_120, &inputs).expect_err("no limit fits");
        assert!(
            err.contains(
                "; the terms of the 20 % memory margin that settings size: backup copies of \
                 external values "
            ),
            "the margin's one operator-sized term is named: {err}"
        );
        assert!(
            err.ends_with(&format!(
                "; {BOTH_SIDES_LOWER}lower LIVE_EDIT_MAX_LAYER_SIZE; {KEEP_BACKUPS}"
            )),
            "the margin's remedy with the tail off: {err}"
        );
        for absent in [
            "checkpoint base bootstraps",
            "turn off incremental backups",
            "LIVE_EDIT_CHECKPOINT_TAIL_ENABLED",
            "at least",
        ] {
            assert!(!err.contains(absent), "{absent:?} not in {err}");
        }
    }

    /// The one thing a refusal's remedy may offer to turn off, with the
    /// start of its cost.
    const TAIL_OFF_WITH_ITS_COST: &str = "turn off incremental backups \
         (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false), though recovery points then come only from \
         full backup passes";

    /// The remedy part of a refusal: what follows its term lists, from its
    /// first "; raise " or "; lower ", which no term or setting name
    /// contains.
    fn remedy_of(refusal: &str) -> &str {
        let at = ["; raise ", "; lower "]
            .iter()
            .filter_map(|start| refusal.find(start))
            .min()
            .unwrap_or_else(|| panic!("the refusal has a remedy: {refusal}"));
        &refusal[at..]
    }

    /// The blast-radius rule: no startup refusal ever suggests turning off
    /// backups, checkpointing (`LIVE_EDIT_CHECKPOINT_ENABLED`) or the backup
    /// store (`LIVE_EDIT_CHECKPOINT_ADAPTER`). In every refusal case — the
    /// line alone, without and with a backup store; the margin with the tail
    /// on; both sides; and no limit fitting, for the line's settings, for
    /// `P_f` with a backup store, and for both sides with the tail on and
    /// with it off — the remedy ([`remedy_of`]) names neither setting except
    /// in [`KEEP_BACKUPS`]' "do not turn off", and the only thing it offers
    /// to turn off is the tail, with its cost; a refusal naming the margin
    /// ends with [`KEEP_BACKUPS`], and one that does not never carries it.
    /// The term lists name both settings beside the margin's terms they
    /// switch on; this asserts nothing about them.
    #[test]
    fn refusal_remedies_never_suggest_turning_off_checkpointing_or_the_backup_store() {
        let (both, both_minimum) = both_sides_set_the_minimum();
        let line_with_a_store = PlanInputs {
            upload_waiters: 200,
            ..with_adapter(PlanInputs::reference())
        };
        let line_with_a_store_minimum = minimum_limit_mib(&line_with_a_store).expect("closes");
        let cases = [
            ("the line, no backup store", PlanInputs::reference(), 2_048),
            (
                "the line, a backup store",
                line_with_a_store,
                line_with_a_store_minimum - 1,
            ),
            (
                "the margin, the tail on",
                with_adapter(PlanInputs::reference()),
                2_808,
            ),
            ("both sides", both, both_minimum - 1),
            (
                "no limit fits: the line's settings",
                PlanInputs {
                    catchup_slots: tokio::sync::Semaphore::MAX_PERMITS as u64,
                    ..PlanInputs::reference()
                },
                5_120,
            ),
            (
                "no limit fits: P_f, a backup store",
                PlanInputs {
                    fenced_read_concurrency: 62,
                    ..with_adapter(PlanInputs::reference())
                },
                5_120,
            ),
            (
                "no limit fits: both sides, the tail on",
                PlanInputs {
                    max_layer_size: 1 << 50,
                    ..with_adapter(PlanInputs::reference())
                },
                5_120,
            ),
            (
                "no limit fits: both sides, the tail off",
                tail_off(1 << 50),
                5_120,
            ),
        ];
        let mut naming_the_margin = Vec::new();
        for (case, inputs, mib) in cases {
            let err = derive_at(mib, &inputs).expect_err("a refusal");
            let remedy = remedy_of(&err).replace(KEEP_BACKUPS, "");
            for setting in [
                "LIVE_EDIT_CHECKPOINT_ENABLED",
                "LIVE_EDIT_CHECKPOINT_ADAPTER",
            ] {
                assert!(
                    !remedy.contains(setting),
                    "{case}: the remedy names {setting} other than to say not to turn it off: \
                     {err}"
                );
            }
            assert_eq!(
                remedy.matches("turn off").count(),
                remedy.matches(TAIL_OFF_WITH_ITS_COST).count(),
                "{case}: the remedy offers to turn off nothing but the tail, with its cost: {err}"
            );
            if err.contains("20 % memory margin") {
                naming_the_margin.push(case);
                assert!(
                    err.ends_with(&format!("; {KEEP_BACKUPS}")),
                    "{case}: a refusal naming the margin ends saying not to turn backups off: \
                     {err}"
                );
            } else {
                assert!(
                    !err.contains(KEEP_BACKUPS),
                    "{case}: a refusal not naming the margin has no reason to: {err}"
                );
            }
        }
        assert_eq!(
            naming_the_margin,
            [
                "the margin, the tail on",
                "both sides",
                "no limit fits: both sides, the tail on",
                "no limit fits: both sides, the tail off",
            ],
            "fixture self-check: the cases that name the margin"
        );
    }

    const REFERENCE_LIMIT_BYTES: u64 = super::super::pod_memory::REFERENCE_POD_MEMORY_LIMIT_BYTES;

    /// A `MakeWriter` capturing everything written into a shared buffer (the
    /// same small helper `config`'s tests define, per module).
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The one JSON record `plan.log_startup()` writes, under a thread-local
    /// subscriber at INFO.
    fn startup_record(plan: &PodMemoryPlan) -> serde_json::Value {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(buffer.clone()))
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::with_default(subscriber, || plan.log_startup());
        let text = String::from_utf8(buffer.lock().unwrap().clone()).expect("UTF-8 log output");
        let records: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("a JSON log record"))
            .collect();
        assert_eq!(records.len(), 1, "one startup line: {text}");
        records.into_iter().next().expect("one record")
    }

    /// The start every no-readable-limit WARN shares, before its remedy.
    const NO_LIMIT_WARN_PREFIX: &str = "no readable pod memory limit: every memory budget is \
         sized for the 5,120 MiB reference pod (see reason). ";

    /// W7: the startup line is the operator's view of the plan. With no
    /// readable limit it is a WARN on Linux (where a container is expected to
    /// have one), carrying the reference source and why, and its message is
    /// the shared prefix followed by the remedy for that reason and no other
    /// reason's: unlimited, not found, unreadable, and none (the reference
    /// size chosen without a cgroup read). With a limit read it is INFO,
    /// carrying that limit and the budgets derived from it. Either way it
    /// carries every budget, the line, the margin and the minimum; the
    /// external-value mirror's budget is 0 when the plan funds no mirror and
    /// `M_ext` when it does.
    #[test]
    fn log_startup_warns_for_the_reference_size_on_linux_and_informs_for_a_read_limit() {
        let inputs = PlanInputs::reference();
        let read = derive_at(8_192, &inputs).expect("an 8 GiB pod closes");

        let expected_level = if cfg!(target_os = "linux") {
            "WARN"
        } else {
            "INFO"
        };
        let detail = "/sys/fs/cgroup/memory.max: Permission denied (os error 13)";
        let reasons = [
            (
                Some(ReferenceReason::Unlimited),
                "unlimited".to_owned(),
                "No memory limit is set on the container: set resources.limits.memory so the \
                 budgets follow the memory the pod really has",
            ),
            (
                Some(ReferenceReason::NotFound),
                "not_found".to_owned(),
                "The container cannot see its cgroup memory files: check that /sys/fs/cgroup is \
                 mounted in the container and its memory controller is enabled, so the limit \
                 set on the container can be read",
            ),
            (
                Some(ReferenceReason::Unreadable(detail.to_owned())),
                format!("unreadable: {detail}"),
                "A file the limit is read from could not be read or parsed: the reason names the \
                 file and the error; fix that so the limit set on the container can be read",
            ),
            (
                None,
                "none".to_owned(),
                "Set the container's memory limit (resources.limits.memory) so the budgets \
                 follow the memory the pod really has",
            ),
        ];
        for (reason, reason_text, remedy) in &reasons {
            let reference =
                PodMemoryPlan::derive(PodMemoryLimit::reference(reason.clone()), &inputs)
                    .expect("the reference pod closes");
            let warn = startup_record(&reference);
            assert_eq!(warn["level"], expected_level, "{reason:?}: {warn}");
            let fields = &warn["fields"];
            if cfg!(target_os = "linux") {
                let message = fields["message"]
                    .as_str()
                    .unwrap_or_else(|| panic!("{reason:?}: a message: {warn}"));
                let rest = message
                    .strip_prefix(NO_LIMIT_WARN_PREFIX)
                    .unwrap_or_else(|| {
                        panic!("{reason:?}: the WARN starts with the shared prefix: {message}")
                    });
                assert_eq!(rest, *remedy, "{reason:?}: the WARN's remedy");
                for (other, _, other_remedy) in &reasons {
                    assert!(
                        other == reason || !message.contains(other_remedy),
                        "{reason:?}: the WARN carries {other:?}'s remedy: {message}"
                    );
                }
            }
            assert_eq!(fields["source"], "reference", "{reason:?}: {warn}");
            assert_eq!(fields["reason"], reason_text.as_str(), "{reason:?}: {warn}");
            assert_eq!(
                fields["pod_memory_limit_bytes"], REFERENCE_LIMIT_BYTES,
                "{reason:?}: {warn}"
            );
            assert_eq!(fields["max_subscriptions"], 4_096, "{reason:?}: {warn}");
            assert_eq!(
                fields["catchup_budget_bytes"],
                744 * MIB,
                "{reason:?}: {warn}"
            );
        }

        let info = startup_record(&read);
        assert_eq!(info["level"], "INFO", "{info}");
        let fields = &info["fields"];
        assert_eq!(fields["source"], "cgroup_v2", "{info}");
        assert_eq!(fields["reason"], "none", "{info}");
        assert_eq!(fields["pod_memory_limit_bytes"], 8_192 * MIB, "{info}");
        for (field, value) in [
            ("upload_budget_bytes", read.upload_budget_bytes()),
            ("catchup_budget_bytes", read.catchup_budget_bytes()),
            (
                "structural_cache_budget_bytes",
                read.structural_cache_budget_bytes(),
            ),
            ("max_subscriptions", read.max_subscriptions()),
            ("retirement_budget_bytes", read.retirement_budget_bytes()),
            // No backup store: the plan funds no mirror.
            ("ext_mirror_budget_bytes", 0),
            ("line_total_bytes", read.line_total_bytes()),
            ("line_limit_bytes", read.line_limit_bytes()),
            ("line_headroom_bytes", read.line_headroom_bytes()),
            ("margin_named_bytes", read.margin_named_bytes()),
            ("margin_left_bytes", read.margin_left_bytes()),
        ] {
            assert_eq!(fields[field], value, "{field}: {info}");
        }
        assert_eq!(
            fields["minimum_limit_mib"],
            read.minimum_limit_mib().expect("a minimum"),
            "{info}"
        );
        assert_eq!(
            (read.max_subscriptions(), read.catchup_budget_mib()),
            (6_553, 1_607),
            "the 8 GiB pod's own budgets, not the reference pod's"
        );

        // With a backup store the startup line carries the mirror's budget,
        // M_ext.
        let funded = derive_at(8_192, &with_adapter(inputs)).expect("an 8 GiB pod closes");
        assert_eq!(
            startup_record(&funded)["fields"]["ext_mirror_budget_bytes"],
            204 * MIB,
            "the mirror's budget at 8 GiB"
        );
    }
}
