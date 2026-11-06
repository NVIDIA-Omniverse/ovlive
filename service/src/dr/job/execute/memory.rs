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

//! Restore memory (SDD §22.22, *Restore memory*; SR-12): how many layers the
//! restore Job holds in memory at once, as it plans and as it writes.
//!
//! - **The budget.** `R = ⌊k × L ÷ 10⁴⌋` bytes of the Job's pod memory limit
//!   `L`, `k` = [`DR_RESTORE_SHARE_TEN_THOUSANDTHS`] (5,000: half). A code
//!   constant with no setting. It is a term of the Job's own memory line
//!   (`config::memory_plan::DrRestoreLine`), beside the steady-state floor and
//!   the Valkey connections' buffers, and the line must fit 80 % of `L`: the
//!   Job refuses to start (exit 2) below the minimum limit at which it does.
//! - **Admission** ([`RestoreGate`]): at most `LIVE_EDIT_DR_RESTORE_PARALLELISM`
//!   layers at once, each admitted only while its charge fits what the
//!   budget has left, first come first served. A charge larger than the
//!   whole budget is clamped to it, so such a layer runs alone.
//! - **The value removal is admitted once.** Accept-loss's value removal
//!   (`execute::discard`, D5) takes one admission per layer before its index
//!   walk and holds it until its header lands, so the budget's FIFO
//!   admission bounds the removals in flight as it bounds every other
//!   charge. Its charge depends on whether the allocator is purged around
//!   each of its transactions ([`value_removal_admission`]): with a working
//!   purge, [`DR_VALUE_REMOVAL_ADMISSION_BYTES`] (one batch's charge,
//!   [`DR_VALUE_REMOVAL_BATCH_BYTES`], plus one page of rows read at once,
//!   [`DR_VALUE_REMOVAL_PAGE_ROWS`] rows, which is above ⌈1.25 × the worst
//!   resident growth measured with the purge⌉); with none installed, or
//!   once one failed, the un-purged fail-safe,
//!   [`DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES`]. It never waits on the
//!   gate while it holds it: a purge that fails while it holds the purged
//!   figure tops up, forced ([`Admission::top_up`]).
//! - **Serial mode.** When the Job could read no limit (`L` is the reference
//!   size because the cgroup sets none, has no memory files, or one could not
//!   be read), it restores one layer at a time with no byte bound, and the
//!   frame logs one WARN. A `Config` built field by field (the reference
//!   limit with no reason) keeps the byte bound.
//! - **The charges**, estimated before the bytes they cover are resident:
//!   the planner charges `fetched × (1 +` [`DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE`]`) +`
//!   [`DR_PLAN_DECODE_CHARGE_FIXED_BYTES`] before it decodes a lineage it
//!   fetched; the writer charges `fetched +`
//!   [`DR_WRITE_CHARGE_STATE_FACTOR`]` × state +`
//!   [`DR_WRITE_CHARGE_BATCH_BYTES`] `+` [`inline_value_charge`] of the
//!   state's largest inline value, from the footprint the planner recorded,
//!   before it re-fetches. Each is calibrated on a release-build heap
//!   measurement (record `restore-memory-envelope-tv3-20261022`, its
//!   `derivation.json`) and guarded by fixture tests.
//!
//! The Job ticks no metric for its budget: its log is the record.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use crate::config::memory_plan::{DrRestoreLine, DR_RESTORE_SHARE_TEN_THOUSANDTHS};
use crate::config::pod_memory::ReferenceReason;
use crate::config::{Config, PlanInputs, PodMemoryLimitSource};
use crate::dr::job::args::JobMode;
use crate::observability::metrics::SharedMetrics;
use crate::storage::layer::dr_restore::{RESTORE_BATCH_ARG_BYTES, RESTORE_BATCH_COMMANDS};
use crate::storage::memory_budget::{
    BudgetSeries, BudgetWaitOutcome, MemoryBudget, MemoryGrant, MemoryTopUp,
};
use crate::storage::retirement::RETIREMENT_ROW_BYTES;

/// The planner's charge per fetched byte for the state it is about to
/// decode: 13. It charges `fetched × (1 + 13)` before the decode, the
/// fetched bytes plus their decoded state, and
/// [`DR_PLAN_DECODE_CHARGE_FIXED_BYTES`] beside them; a decoded state above
/// `fetched × 13` is logged ([`over_estimate`]), the recalibration trigger.
/// The decode's peak heap reached 10.84 × the fetched bytes on a layer with
/// one children list of 20,000 prims, against 8.65–9.14 × at the record's
/// other shapes [measured: 37.89 MB at 3,494,376 fetched bytes; record
/// `restore-memory-envelope-tv3-20261022`, `raw/e-recal2-job-children20k.jsonl`],
/// so the total is ⌈1.25 × 10.84⌉ = 14 (the record's `derivation.json`
/// `planner_decode_charge`).
pub const DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE: u64 = 13;

/// The planner's fixed charge beside its per-fetched-byte one: 2 MiB, so a
/// lineage is charged `fetched × 14 + 2 MiB`. At a small lineage the decode's
/// resident growth outran the per-byte charge, then `fetched × 12`, by about
/// 1.3 MB, allocator page granularity at MB scale [measured: 4.33 MB resident
/// against a 3.07 MB charge at 255,472 fetched bytes, one run of two; heap
/// 0.72 of the charge; record `restore-memory-envelope-tv3-20261022`,
/// `raw/e-recal-job-batch.jsonl`], which the per-byte factor does not scale
/// to; ⌈1.25 × 1.3 MB⌉, rounded up to whole MiB, is 2 MiB (the record's
/// `derivation.json` `planner_decode_fixed_charge`). Added, not a floor under
/// the product: the charge that was outrun was already above 2 MiB.
pub const DR_PLAN_DECODE_CHARGE_FIXED_BYTES: u64 = 2 * MIB;

/// The writer's charge per byte of decoded state: 3 (PD-7). It charges
/// `fetched + 3 × state + `[`DR_WRITE_CHARGE_BATCH_BYTES`] for an object it
/// restores: the re-fetched bytes; the decoded state, the decode's
/// transient beside it and its flattened copy; and one batch's client
/// memory; plus [`inline_value_charge`] of its largest inline value.
/// Measured with the batch bounded, the writer's peak is its
/// decode's at the record's three layer sizes (27,231, 60,055 and 194,000
/// specs; record `restore-memory-envelope-tv3-20261022`).
pub const DR_WRITE_CHARGE_STATE_FACTOR: u64 = 3;

/// The writer's charge for the one batch of queued commands it holds at a
/// time, whatever the object's size: 40 MiB, ⌈1.25 × (29.11 MB + 4 MiB ×
/// 1.00)⌉ in MiB. 29.11 MB is the heap a full batch of 9,999 commands held
/// above the state (the record's `batch` shape); 4 MiB × 1.00 is the byte
/// limit's term, at the client memory measured per argument byte (record
/// `restore-memory-envelope-tv3-20261022`): the restore batch's two limits,
/// `storage::layer::dr_restore::RESTORE_BATCH_COMMANDS` and
/// `RESTORE_BATCH_ARG_BYTES`, priced. A constant, not a factor of the state:
/// a small layer still queues a whole batch of up to 10,000 commands, each
/// holding about 40 times its share of the state.
pub const DR_WRITE_CHARGE_BATCH_BYTES: u64 = 40 * MIB;

/// The writer's charge per byte of an object's largest inline value, in
/// hundredths: 459, so the term is ⌈4.59 × the value⌉
/// ([`inline_value_charge`]). An inline value over the restore batch's byte
/// limit is queued alone, whole, beside the state that holds it: the
/// writer's peak heap grew 3.667 bytes per byte of one such value, at 16, 64
/// and 96 MiB [measured: 61.8, 246.4 and 369.4 MB; record
/// `restore-memory-envelope-tv3-20261022`, `raw/e-recal2-measure-inline*.jsonl`],
/// which outgrew the 0.8 line of `fetched + 3 × state + 40 MiB` above about
/// 69 MiB [derived] (0.83 at 96 MiB). ⌈1.25 × 366.7⌉ = 459 (the record's
/// `derivation.json` `writer_inline_value_charge`). One upload carries the
/// value (`LIVE_EDIT_MAX_LAYER_SIZE`, 100 MiB by default), so at the default
/// the term is at most about 459 MiB.
pub const DR_WRITE_CHARGE_INLINE_VALUE_CENTI: u64 = 459;

/// A restore batch's client memory per queued command, in whole bytes:
/// 2,912, ⌈2,911.785⌉ [measured: 29.11 MB of heap above the state for the
/// record's `batch` shape's one 9,999-command batch; record
/// `restore-memory-envelope-tv3-20261022`, `derivation.json`
/// `writer_batch_limits`]. The per-command term of the measured value-removal
/// batch that [`DR_VALUE_REMOVAL_BATCH_BYTES`] scales.
pub const DR_RESTORE_BATCH_BYTES_PER_COMMAND: u64 = 2_912;

/// The client memory one accept-loss value-removal batch holds while it is
/// queued: 33,314,304 B (31.8 MiB), `RESTORE_BATCH_COMMANDS ×`
/// [`DR_RESTORE_BATCH_BYTES_PER_COMMAND`] `+ RESTORE_BATCH_ARG_BYTES` at one
/// byte per argument byte, the batch's two limits priced as the writer's
/// batch is (the record's `derivation.json` `value_removal_batch`,
/// `measured_bytes`).
const VALUE_REMOVAL_BATCH_MEASURED_BYTES: u64 =
    RESTORE_BATCH_COMMANDS * DR_RESTORE_BATCH_BYTES_PER_COMMAND + RESTORE_BATCH_ARG_BYTES;

/// The value removal's charge for one batch: 40 MiB, ⌈1.25 ×
/// 33,314,304 B⌉ rounded up to whole MiB, the rule the writer's batch charge
/// ([`DR_WRITE_CHARGE_BATCH_BYTES`]) follows, at the measured batch above.
/// The batch term of the model that prices the value removal's purged
/// admission ([`DR_VALUE_REMOVAL_ADMISSION_BYTES`]).
pub const DR_VALUE_REMOVAL_BATCH_BYTES: u64 =
    (5 * VALUE_REMOVAL_BATCH_MEASURED_BYTES).div_ceil(4 * MIB) * MIB;

/// The value removal's page: 256 rows. Each `ZRANGE` page of a layer's value
/// indices spans this many ranks (the last, lowest page fewer), and the
/// removal reads at most this many value rows at once, each by the readers'
/// rule (`storage::diff_read::read_value_row_external_hash`). Each row read in
/// flight holds at most `storage::retirement::RETIREMENT_ROW_BYTES` (40 KiB:
/// its key, the read's future and its bounded reply [model]), the same read
/// as the retirement walk's, so a page holds at most 10 MiB [model; inside
/// the measured admission: record `restore-memory-envelope-tv3-20261022`,
/// `value_removal_admission`].
pub const DR_VALUE_REMOVAL_PAGE_ROWS: u64 = 256;

/// The value removal's one admission per layer when the allocator is purged
/// around each of its transactions ([`value_removal_admission`]`(true)`):
/// 50 MiB (52,428,800 B), the larger of the batch-and-page model and
/// ⌈1.25 × the worst resident growth measured with the purge⌉ in whole MiB.
/// The model is one batch ([`DR_VALUE_REMOVAL_BATCH_BYTES`]) plus one page of
/// rows read at once ([`DR_VALUE_REMOVAL_PAGE_ROWS`] ×
/// `RETIREMENT_ROW_BYTES`), the price of a batch at both of its limits:
/// 50 MiB. The worst resident growth with the purge was 37,052,416 B
/// (37.1 MB), with 418 B members, the size at which both batch limits bind;
/// 8,200 B members (4,096 B names, the name limit) and 22 B members grew
/// less. ⌈1.25 × 37,052,416 B⌉ is 45 MiB, so the model sets the price
/// [measured to 719,922 rows, with the purge, at 4 and at 16 tokio workers
/// (16 and 64 jemalloc arenas): record `restore-memory-envelope-tv3-20261022`,
/// `value_removal_admission`]. The 8,200 B and 418 B shapes are derived from
/// the limits, not realistic shapes. Taken before the layer's index walk and
/// held until its header lands; a purge that fails while it is held tops it
/// up to the un-purged figure ([`Admission::top_up`]).
pub const DR_VALUE_REMOVAL_ADMISSION_BYTES: u64 =
    DR_VALUE_REMOVAL_BATCH_BYTES + DR_VALUE_REMOVAL_PAGE_ROWS * RETIREMENT_ROW_BYTES;

/// The value removal's one admission per layer when no allocator purge is
/// installed, or once one has failed ([`value_removal_admission`]`(false)`):
/// 158 MiB, ⌈1.25 × 131,944,448 B⌉ in whole MiB. 131,944,448 B is the worst
/// resident growth measured with no purge: 8,200 B members (4,096 B names,
/// the name limit, not a realistic shape) at 719,922 rows, run 1, where the
/// freed pages wait out jemalloc's 10 s decay in every worker arena the task
/// touched [measured: record `restore-memory-envelope-tv3-20261022`,
/// `value_removal_unpurged_admission`]. Not a proven bound: that growth did
/// not plateau, it grows with the runtime's worker count (it was measured
/// at the measuring host's), and the 418 B shape was not measured without a
/// purge. Never charged with a working purge; on linux-gnu the binary
/// installs one, so it is the fail-safe of another target or a binary that
/// forgot the hook, and the accept-loss Job WARNs when it is in force.
pub const DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES: u64 = 158 * MIB;

/// The value removal's one admission per layer: the purged figure,
/// [`DR_VALUE_REMOVAL_ADMISSION_BYTES`], only when `purge_works` (a purge is
/// installed and none has failed); otherwise the un-purged fail-safe,
/// [`DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES`].
pub fn value_removal_admission(purge_works: bool) -> u64 {
    if purge_works {
        DR_VALUE_REMOVAL_ADMISSION_BYTES
    } else {
        DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES
    }
}

const MIB: u64 = 1024 * 1024;

/// The restore budget's series: no metric (Task 6 ruling: the Job's log is
/// the record). Built with no metrics handle, so nothing calls it.
#[derive(Debug)]
pub struct DrRestoreSeries;

/// [`DrRestoreSeries`].
pub static DR_RESTORE_SERIES: DrRestoreSeries = DrRestoreSeries;

impl BudgetSeries for DrRestoreSeries {
    fn add_reserved(&self, _m: &SharedMetrics, _delta: i64) {}

    fn add_overcommit(&self, _m: &SharedMetrics, _delta: i64) {}

    fn record_wait(
        &self,
        _m: &SharedMetrics,
        _waited: std::time::Duration,
        _outcome: BudgetWaitOutcome,
    ) {
    }

    fn record_try_grant(&self, _m: &SharedMetrics) {}
}

/// How the Job bounds the layers it holds at once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreMemory {
    /// At most `parallelism` layers, within `budget` bytes.
    Bytes {
        /// `R`, in bytes.
        budget: u64,
        /// `LIVE_EDIT_DR_RESTORE_PARALLELISM`.
        parallelism: NonZeroUsize,
        /// The Job's memory line it was derived from.
        line: DrRestoreLine,
    },
    /// One layer at a time, no byte bound: the Job could read no limit.
    Serial {
        /// Why no limit was read.
        reason: ReferenceReason,
        /// `LIVE_EDIT_DR_RESTORE_PARALLELISM`, which serial mode does not use.
        parallelism: NonZeroUsize,
    },
}

impl RestoreMemory {
    /// The Job's restore memory from `config`: serial when its limit is the
    /// reference size read for a reason (no limit, no memory files, an
    /// unreadable one); otherwise the budget `R` at the fixed share. `Err`,
    /// the refusal text, naming the Job running in `mode`, when the limit is
    /// below the minimum at which the Job's line closes. Serial mode, which
    /// knows no limit and runs one layer at a time, has no such check.
    pub fn derive(config: &Config, mode: JobMode) -> Result<Self, String> {
        let limit = config.pod_memory.limit();
        let parallelism = config.dr_restore_parallelism;
        if limit.source == PodMemoryLimitSource::Reference {
            if let Some(reason) = &limit.reason {
                return Ok(Self::Serial {
                    reason: reason.clone(),
                    parallelism,
                });
            }
        }
        let line = config.pod_memory.dr_restore_line(
            DR_RESTORE_SHARE_TEN_THOUSANDTHS,
            &PlanInputs::from_config(config),
        );
        if !line.closes {
            return Err(refusal_text(&line, mode));
        }
        Ok(Self::Bytes {
            budget: line.restore_budget_bytes,
            parallelism,
            line,
        })
    }

    /// The most layers at once: 1 in serial mode.
    pub fn parallelism(&self) -> usize {
        match self {
            Self::Bytes { parallelism, .. } => parallelism.get(),
            Self::Serial { .. } => 1,
        }
    }

    /// The configured parallelism, whatever the mode.
    pub fn configured_parallelism(&self) -> usize {
        match self {
            Self::Bytes { parallelism, .. } | Self::Serial { parallelism, .. } => parallelism.get(),
        }
    }

    /// `R` in bytes; `None` in serial mode.
    pub fn budget_bytes(&self) -> Option<u64> {
        match self {
            Self::Bytes { budget, .. } => Some(*budget),
            Self::Serial { .. } => None,
        }
    }

    /// The mode as the start line words it: `up to {n} layers at once within
    /// {budget} MiB` or `one layer at a time`.
    pub fn describe(&self) -> String {
        match self {
            Self::Bytes {
                budget,
                parallelism,
                ..
            } => format!(
                "up to {} layers at once within {} MiB",
                parallelism.get(),
                budget / MIB
            ),
            Self::Serial { .. } => "one layer at a time".to_owned(),
        }
    }

    /// The serial-mode WARN, once per Job (`run_job`): nothing in bytes mode.
    pub fn log_serial_warn(&self, mode: JobMode) {
        if let Self::Serial {
            reason,
            parallelism,
        } = self
        {
            tracing::warn!(
                reason = %reason,
                parallelism = parallelism.get(),
                "the {mode} Job could not read a memory limit ({reason}); it restores one layer \
                 at a time instead of {}, so it never holds two layers' state at once; give the \
                 Job a memory limit (dr.resources) to restore in parallel",
                parallelism.get()
            );
        }
    }
}

/// The exit-2 refusal of a limit below the Job's minimum, for the Job
/// running in `mode`.
fn refusal_text(line: &DrRestoreLine, mode: JobMode) -> String {
    let minimum = line
        .minimum_limit_mib
        .map_or_else(|| "an unbounded".to_owned(), |mib| mib.to_string());
    format!(
        "live-edit dr: the {mode} Job's memory limit of {} MiB is below the {minimum} MiB it \
         needs (the layers in flight, the steady-state floor and the Valkey connection \
         buffers); raise dr.resources",
        line.limit_mib()
    )
}

/// The planner's charge for a lineage of `fetched_bytes`, before its decode:
/// `fetched × 14 + 2 MiB`, saturating.
pub fn plan_decode_charge(fetched_bytes: u64) -> u64 {
    fetched_bytes
        .saturating_mul(1 + DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE)
        .saturating_add(DR_PLAN_DECODE_CHARGE_FIXED_BYTES)
}

/// The writer's charge for an object whose lineage fetched `fetched_bytes`
/// and decoded to `state_bytes`, before its largest inline value's term:
/// `fetched + 3 × state + 40 MiB`. The writer charges this plus
/// [`inline_value_charge`] of the footprint's largest inline value.
pub fn write_charge(fetched_bytes: u64, state_bytes: u64) -> u64 {
    fetched_bytes
        .saturating_add(DR_WRITE_CHARGE_STATE_FACTOR.saturating_mul(state_bytes))
        .saturating_add(DR_WRITE_CHARGE_BATCH_BYTES)
}

/// The writer's term for an object whose largest inline value is
/// `largest_inline_bytes`: ⌈4.59 × the value⌉ bytes
/// ([`DR_WRITE_CHARGE_INLINE_VALUE_CENTI`]), saturating; 0 for none.
pub fn inline_value_charge(largest_inline_bytes: u64) -> u64 {
    let centi = u128::from(largest_inline_bytes) * u128::from(DR_WRITE_CHARGE_INLINE_VALUE_CENTI);
    u64::try_from(centi.div_ceil(100)).unwrap_or(u64::MAX)
}

/// Log the over-estimate WARN when a lineage's decoded state outgrew what the
/// planner's charge allowed for it: `state_bytes` above `fetched_bytes × 13`.
/// The recalibration trigger for [`DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE`].
pub fn over_estimate(object_id: u64, fetched_bytes: u64, state_bytes: u64) {
    let estimate = fetched_bytes.saturating_mul(DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE);
    if state_bytes > estimate {
        let state = mib_tenths(state_bytes);
        let estimate = mib_tenths(estimate);
        tracing::warn!(
            object_id,
            fetched_bytes,
            state_bytes,
            "object {object_id}'s decoded state is {state} MiB, above the {estimate} MiB the \
             restore budget charged before decoding; the budget was over-committed while it \
             decoded"
        );
    }
}

/// `bytes` in MiB with one decimal.
fn mib_tenths(bytes: u64) -> String {
    let tenths = (u128::from(bytes) * 10).div_ceil(u128::from(MIB));
    format!("{}.{}", tenths / 10, tenths % 10)
}

/// One stage's admission gate (module docs): the count bound is the
/// stage's concurrency ([`Self::parallelism`]); the byte bound, in bytes
/// mode, a FIFO [`MemoryBudget`] of `R`. Tracks the most layers admitted at
/// once and the most bytes reserved, for the execution report.
pub struct RestoreGate {
    budget: Option<Arc<MemoryBudget>>,
    parallelism: usize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    peak_reserved: AtomicU64,
    /// Test-only: woken whenever an admission is granted or given back
    /// ([`Self::admissions_changed`]). Compiled out of release builds.
    #[cfg(any(test, feature = "test-support"))]
    changed: tokio::sync::Notify,
}

impl RestoreGate {
    /// The gate for `memory`; `budget_override`, when set, replaces `R` in
    /// bytes mode (a test seam).
    pub fn new(memory: &RestoreMemory, budget_override: Option<u64>) -> Self {
        let budget = memory.budget_bytes().map(|r| {
            let total = budget_override.unwrap_or(r);
            Arc::new(MemoryBudget::with_series(
                total.div_ceil(1024),
                Arc::new(StdMutex::new(None)),
                &DR_RESTORE_SERIES,
            ))
        });
        Self {
            budget,
            parallelism: memory.parallelism(),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            peak_reserved: AtomicU64::new(0),
            #[cfg(any(test, feature = "test-support"))]
            changed: tokio::sync::Notify::new(),
        }
    }

    /// The most layers the stage runs at once.
    pub fn parallelism(&self) -> usize {
        self.parallelism
    }

    /// The byte budget, in bytes; `None` in serial mode.
    pub fn budget_bytes(&self) -> Option<u64> {
        self.budget.as_ref().map(|b| b.total_bytes())
    }

    /// Bytes reserved now.
    pub fn reserved_bytes(&self) -> u64 {
        self.budget.as_ref().map_or(0, |b| b.reserved_bytes())
    }

    /// Bytes held now by forced top-ups ([`Admission::top_up`]), beside
    /// [`Self::reserved_bytes`]; 0 in serial mode.
    pub fn overcommit_bytes(&self) -> u64 {
        self.budget.as_ref().map_or(0, |b| b.overcommit_bytes())
    }

    /// Layers admitted now.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// The most layers admitted at once so far.
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }

    /// The most bytes reserved at once so far.
    pub fn peak_reserved_bytes(&self) -> u64 {
        self.peak_reserved.load(Ordering::SeqCst)
    }

    /// Admit one layer charging `charge` bytes: wait, first come first
    /// served, until its charge (clamped to the whole budget) fits. Serial
    /// mode admits at once (its count bound is the stage's concurrency of
    /// 1). `None` only when the budget refused, which a budget nothing
    /// closes never does.
    pub async fn admit(&self, charge: u64) -> Option<Admission<'_>> {
        let grant = match &self.budget {
            Some(budget) => Some(budget.reserve(charge, None).await.ok()?),
            None => None,
        };
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        self.peak_reserved
            .fetch_max(self.reserved_bytes(), Ordering::SeqCst);
        #[cfg(any(test, feature = "test-support"))]
        self.changed.notify_waiters();
        Some(Admission { gate: self, grant })
    }

    /// Test-only: a future that resolves at the next admission granted
    /// ([`Self::admit`] returning one) or given back (an [`Admission`]
    /// dropped, after its bytes are released), so a test waits on the event
    /// instead of polling the gate. It wakes only futures already waiting
    /// (`Notify::notify_waiters`), so register it before checking the
    /// condition, then await it: `let mut changed =
    /// pin!(gate.admissions_changed()); changed.as_mut().enable();` then
    /// check, then `changed.await`. A forced
    /// top-up ([`Admission::top_up`]) does not wake it. Compiled out of
    /// release builds.
    #[cfg(any(test, feature = "test-support"))]
    pub fn admissions_changed(&self) -> tokio::sync::futures::Notified<'_> {
        self.changed.notified()
    }
}

/// One admitted layer: its memory and its place in the count, given back
/// when dropped, on every exit.
pub struct Admission<'g> {
    gate: &'g RestoreGate,
    grant: Option<MemoryGrant>,
}

impl Admission<'_> {
    /// The bytes this admission holds (0 in serial mode).
    pub fn bytes(&self) -> u64 {
        self.grant.as_ref().map_or(0, MemoryGrant::bytes)
    }

    /// A forced top-up of `bytes` on this admission's budget, for memory
    /// already resident beside it (`MemoryBudget::top_up`): taken at once,
    /// never waiting and never refused, repaid when the returned guard
    /// drops. `None` in serial mode, which has no budget.
    pub fn top_up(&self, bytes: u64) -> Option<MemoryTopUp> {
        self.grant
            .as_ref()
            .map(|grant| grant.top_up_on_budget(bytes))
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.gate.in_flight.fetch_sub(1, Ordering::SeqCst);
        // Test-only: release the bytes now (the field's own drop would do it
        // just after this body, in the same order) so a waiter woken below
        // sees them gone.
        #[cfg(any(test, feature = "test-support"))]
        {
            drop(self.grant.take());
            self.gate.changed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    //! U6's minimum and refusal, U7 (the gate), U9 (the planner's decode
    //! charge against real decodes), the charges, and the charges', the
    //! value-removal batch's and its two admissions' constants against their
    //! committed derivation record. The line's exactness and closure sweeps are
    //! `config::memory_plan`'s; `derive` over a loaded configuration (serial
    //! per reason, bytes for an explicit reference) is
    //! `tests/misc/dr_job_rules.rs`'s.

    use std::pin::pin;

    use ordered_float::OrderedFloat;

    use super::*;
    use crate::config::memory_plan::dr_restore_line_at;
    use crate::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};

    const KIB: u64 = 1024;

    /// At the chart's pools the Job's line closes at its minimum limit and
    /// not one MiB below it. Prints the minimum.
    #[test]
    fn the_restore_line_closes_at_its_minimum_and_not_one_mib_below() {
        let inputs = PlanInputs::reference();
        let k = DR_RESTORE_SHARE_TEN_THOUSANDTHS;
        let minimum = dr_restore_line_at(1 << 32, k, &inputs)
            .minimum_limit_mib
            .expect("a share of 5,000 closes at some limit");
        eprintln!("the restore line's minimum limit at the chart's pools: {minimum} MiB");
        assert!(dr_restore_line_at(minimum * MIB, k, &inputs).closes);
        assert!(!dr_restore_line_at((minimum - 1) * MIB, k, &inputs).closes);
    }

    /// The exit-2 refusal of a limit below the minimum names the limit, the
    /// minimum and the operator's lever, `dr.resources`, and no setting.
    #[test]
    fn the_refusal_names_the_limit_the_minimum_and_dr_resources() {
        let inputs = PlanInputs::reference();
        let k = DR_RESTORE_SHARE_TEN_THOUSANDTHS;
        let minimum = dr_restore_line_at(1 << 32, k, &inputs)
            .minimum_limit_mib
            .expect("a minimum");
        let line = dr_restore_line_at((minimum - 1) * MIB, k, &inputs);
        assert_eq!(
            refusal_text(&line, JobMode::Restore),
            format!(
                "live-edit dr: the restore Job's memory limit of {} MiB is below the {minimum} \
                 MiB it needs (the layers in flight, the steady-state floor and the Valkey \
                 connection buffers); raise dr.resources",
                minimum - 1
            )
        );
    }

    /// The same refusal in accept-loss mode names the accept-loss Job, not
    /// the restore Job: the line an operator reads names the Job they ran.
    #[test]
    fn the_accept_loss_refusal_names_the_accept_loss_jobs_memory_limit() {
        let inputs = PlanInputs::reference();
        let k = DR_RESTORE_SHARE_TEN_THOUSANDTHS;
        let minimum = dr_restore_line_at(1 << 32, k, &inputs)
            .minimum_limit_mib
            .expect("a minimum");
        let line = dr_restore_line_at((minimum - 1) * MIB, k, &inputs);
        assert_eq!(
            refusal_text(&line, JobMode::AcceptLoss),
            format!(
                "live-edit dr: the accept-loss Job's memory limit of {} MiB is below the \
                 {minimum} MiB it needs (the layers in flight, the steady-state floor and the \
                 Valkey connection buffers); raise dr.resources",
                minimum - 1
            )
        );
    }

    /// The planner's charge before a decode is the fetched bytes plus
    /// `DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE` times them, plus the fixed
    /// `DR_PLAN_DECODE_CHARGE_FIXED_BYTES` added beside the product (not a
    /// floor under it), saturating: a lineage of no bytes still pays the
    /// fixed term. Both constants' values are the record's
    /// ([`restore_charges_are_the_recorded_envelope_derivation`]).
    ///
    /// Fails without the fixed term (in-place perturbation of
    /// `plan_decode_charge`): the first two cases are 2 MiB short.
    #[test]
    fn plan_decode_charge_is_fetched_times_one_plus_the_decode_factor_plus_the_fixed_term() {
        assert_eq!(
            plan_decode_charge(1_000),
            1_000 * (1 + DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE)
                + DR_PLAN_DECODE_CHARGE_FIXED_BYTES
        );
        assert_eq!(
            plan_decode_charge(0),
            DR_PLAN_DECODE_CHARGE_FIXED_BYTES,
            "an empty lineage still pays the fixed term"
        );
        assert_eq!(plan_decode_charge(u64::MAX), u64::MAX);
        assert_eq!(plan_decode_charge(u64::MAX / 2), u64::MAX, "saturating");
    }

    /// The writer's term for an object's largest inline value is
    /// `DR_WRITE_CHARGE_INLINE_VALUE_CENTI` hundredths of a byte per value
    /// byte, rounded up, 0 for none, saturating at the largest charge.
    #[test]
    fn inline_value_charge_is_the_centi_factor_of_the_value_rounded_up_saturating() {
        let centi = DR_WRITE_CHARGE_INLINE_VALUE_CENTI;
        assert_eq!(inline_value_charge(0), 0, "no inline value, no term");
        assert_eq!(inline_value_charge(100), centi, "whole hundredths");
        assert_eq!(
            inline_value_charge(1),
            centi.div_ceil(100),
            "a fraction of a byte rounds up"
        );
        assert_eq!(
            inline_value_charge(100 * MIB),
            centi * MIB,
            "a 100 MiB value, one upload's default limit"
        );
        assert_eq!(inline_value_charge(u64::MAX), u64::MAX, "saturating");
    }

    /// The writer's charge is the fetched bytes, `DR_WRITE_CHARGE_STATE_FACTOR`
    /// times the decoded state, and one batch's fixed
    /// `DR_WRITE_CHARGE_BATCH_BYTES`, saturating. A layer of no bytes still
    /// pays the whole batch term, since a small layer still queues a whole
    /// batch's commands.
    ///
    /// Fails without the batch term (in-place perturbation of
    /// `write_charge`): the first two cases are the batch term short.
    #[test]
    fn write_charge_is_fetched_plus_three_states_plus_one_batch_saturating() {
        assert_eq!(
            write_charge(1_000, 5_000),
            1_000 + DR_WRITE_CHARGE_STATE_FACTOR * 5_000 + DR_WRITE_CHARGE_BATCH_BYTES
        );
        assert_eq!(
            write_charge(0, 0),
            DR_WRITE_CHARGE_BATCH_BYTES,
            "an empty layer still pays one batch"
        );
        assert_eq!(write_charge(1, u64::MAX), u64::MAX);
        assert_eq!(write_charge(u64::MAX, 0), u64::MAX);
    }

    /// The recalibration trigger: a decoded state strictly above
    /// `DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE` × the fetched bytes logs one
    /// WARN with both figures in MiB (tenths, rounded up); a state of exactly
    /// that many times logs nothing. At 2 MiB fetched the estimate is a whole
    /// number of MiB, so its tenths digit is 0 and one byte above it reads .1.
    #[test]
    fn over_estimate_warns_once_above_the_decode_factor_times_fetched_and_is_silent_at_it() {
        let capture = super::super::test_harness::LogCapture::default();
        let fetched = 2 * MIB;
        let factor = DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE;
        tracing::subscriber::with_default(
            tracing_subscriber::fmt()
                .json()
                .with_writer(capture.clone())
                .with_max_level(tracing::Level::INFO)
                .finish(),
            || {
                over_estimate(41, fetched, factor * fetched);
                over_estimate(42, fetched, factor * fetched + 1);
            },
        );
        let estimate_mib = 2 * factor;
        assert_eq!(
            capture.messages("WARN"),
            vec![format!(
                "object 42's decoded state is {estimate_mib}.1 MiB, above the {estimate_mib}.0 \
                 MiB the restore budget charged before decoding; the budget was over-committed \
                 while it decoded"
            )]
        );
    }

    /// The committed record at `path` (relative to the crate root), parsed;
    /// `None` in the published source tree, which has no `docs-internal/`
    /// (only there: in this repository a missing record fails).
    fn committed_record(path: &str) -> Option<serde_json::Value> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!("skipped: no docs-internal/ (published source tree), so no {path}");
            return None;
        }
        let raw = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|e| panic!("the committed record {path}: {e}"));
        Some(serde_json::from_str(&raw).expect("the record is valid JSON"))
    }

    /// The value at JSON `pointer` in `record`, as an integer.
    fn recorded_u64(record: &serde_json::Value, pointer: &str) -> u64 {
        record
            .pointer(pointer)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("the record carries {pointer} as a non-negative integer"))
    }

    /// The restore memory envelope's record (plan b.tv3, ruling w6m2), which
    /// derives the restore Job's charges.
    const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
        "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

    /// The value at JSON `pointer` in `record`, as a number.
    fn recorded_f64(record: &serde_json::Value, pointer: &str) -> f64 {
        record
            .pointer(pointer)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| panic!("the record carries {pointer} as a number"))
    }

    /// The restore Job's charge constants are the record's derivation:
    /// the planner's decode factor is the record's value and one less than
    /// its total, ⌈1.25 × the most heap the decode measured per fetched
    /// byte⌉, and its fixed term is ⌈1.25 × the measured excess⌉ in whole
    /// MiB; the writer's batch term is the record's, ⌈1.25 × (the full
    /// batch's measured heap + the byte limit's term)⌉ in whole MiB, that
    /// byte term being the restore batch's byte limit at the measured client
    /// memory per argument byte (1.00 to two places); the writer's inline
    /// value term is ⌈1.25 × the measured heap per value byte⌉ in hundredths;
    /// and the writer's state factor is the record's 3, not scaled. An edit
    /// to a constant or to the record without the other fails here.
    #[test]
    fn restore_charges_are_the_recorded_envelope_derivation() {
        let Some(record) = committed_record(RESTORE_ENVELOPE_DERIVATION_JSON) else {
            return;
        };

        // The planner's decode charge.
        let factor = recorded_u64(&record, "/planner_decode_charge/value");
        let total = recorded_u64(&record, "/planner_decode_charge/total_multiplier");
        let heap_per_fetched_byte = recorded_f64(
            &record,
            "/planner_decode_charge/measured_heap_per_fetched_byte_max",
        );
        assert_eq!(
            DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE, factor,
            "the decode factor is the record's planner_decode_charge.value"
        );
        assert_eq!(total, 1 + factor, "the record's total is 1 + its factor");
        assert_eq!(
            total,
            (1.25 * heap_per_fetched_byte).ceil() as u64,
            "the total is ⌈1.25 × the most heap measured per fetched byte⌉"
        );

        // The planner's fixed term, beside the product.
        let fixed_bytes = recorded_u64(&record, "/planner_decode_fixed_charge/value_bytes");
        let fixed_excess = recorded_u64(
            &record,
            "/planner_decode_fixed_charge/measured_excess_bytes",
        );
        assert_eq!(
            DR_PLAN_DECODE_CHARGE_FIXED_BYTES, fixed_bytes,
            "the planner's fixed term is the record's planner_decode_fixed_charge.value_bytes"
        );
        assert_eq!(
            fixed_bytes,
            recorded_u64(&record, "/planner_decode_fixed_charge/value_mib") * MIB,
            "the record's bytes are its whole MiB"
        );
        assert_eq!(
            fixed_bytes,
            (5 * fixed_excess).div_ceil(4 * MIB) * MIB,
            "the fixed term is ⌈1.25 × the measured excess⌉ in whole MiB"
        );

        // The writer's batch term.
        let batch_bytes = recorded_u64(&record, "/writer_batch_charge/value_bytes");
        let batch_term = recorded_u64(&record, "/writer_batch_charge/batch_term_bytes");
        let byte_limit_term = recorded_u64(&record, "/writer_batch_charge/byte_limit_term_bytes");
        let per_arg_byte = recorded_f64(&record, "/writer_batch_limits/per_arg_byte");
        let recorded_arg_limit =
            recorded_u64(&record, "/writer_batch_limits/restore_batch_arg_bytes");
        assert_eq!(
            DR_WRITE_CHARGE_BATCH_BYTES, batch_bytes,
            "the batch term is the record's writer_batch_charge.value_bytes"
        );
        assert_eq!(
            batch_bytes,
            recorded_u64(&record, "/writer_batch_charge/value_mib") * MIB,
            "the record's bytes are its whole MiB"
        );
        assert_eq!(
            batch_bytes,
            (5 * (batch_term + byte_limit_term)).div_ceil(4 * MIB) * MIB,
            "the batch term is ⌈1.25 × (batch term + byte limit term)⌉ in whole MiB"
        );
        assert_eq!(
            byte_limit_term,
            crate::storage::layer::dr_restore::RESTORE_BATCH_ARG_BYTES,
            "the byte limit's term prices the restore batch's byte limit at 1.00 per byte"
        );
        assert_eq!(
            byte_limit_term,
            (recorded_arg_limit as f64 * (per_arg_byte * 100.0).round() / 100.0) as u64,
            "the byte limit's term is the record's byte limit × its measured client memory per \
             argument byte, to two places"
        );

        // The writer's inline value term.
        let inline_centi = recorded_u64(&record, "/writer_inline_value_charge/value_centi");
        let heap_per_value_byte = recorded_f64(
            &record,
            "/writer_inline_value_charge/measured_heap_per_value_byte",
        );
        assert_eq!(
            DR_WRITE_CHARGE_INLINE_VALUE_CENTI, inline_centi,
            "the inline value term is the record's writer_inline_value_charge.value_centi"
        );
        assert_eq!(
            inline_centi,
            (1.25 * heap_per_value_byte * 100.0).ceil() as u64,
            "the inline value term is ⌈1.25 × the measured heap per value byte × 100⌉"
        );

        // The writer's state factor, kept.
        assert_eq!(
            DR_WRITE_CHARGE_STATE_FACTOR,
            recorded_u64(&record, "/writer_batch_charge/state_factor"),
            "the state factor is the record's writer_batch_charge.state_factor"
        );
    }

    /// The value removal's batch is the record's derivation:
    /// [`DR_RESTORE_BATCH_BYTES_PER_COMMAND`] is the record's
    /// `value_removal_batch.per_command_bytes`, the full batch's measured
    /// client memory per command rounded up to a whole byte; the measured
    /// batch is its `measured_bytes`, the batch's two limits priced (the
    /// record's command limit at that per-command cost and its byte limit at
    /// the measured memory per argument byte, 1.00 to two places); and
    /// [`DR_VALUE_REMOVAL_BATCH_BYTES`] is its `value_bytes`, 1.25 × the
    /// measured batch rounded up to whole MiB, equal to
    /// [`DR_WRITE_CHARGE_BATCH_BYTES`]. An edit to a constant, the batch
    /// limits or the record without the others fails here.
    #[test]
    fn value_removal_batch_is_the_recorded_envelope_derivation() {
        let Some(record) = committed_record(RESTORE_ENVELOPE_DERIVATION_JSON) else {
            return;
        };

        let per_command = recorded_u64(&record, "/value_removal_batch/per_command_bytes");
        let measured_per_command =
            recorded_f64(&record, "/writer_batch_limits/per_command_bytes/at_9999");
        assert_eq!(
            DR_RESTORE_BATCH_BYTES_PER_COMMAND, per_command,
            "the per-command cost is the record's value_removal_batch.per_command_bytes"
        );
        assert_eq!(
            per_command,
            measured_per_command.ceil() as u64,
            "the per-command cost is the full batch's measured cost per command, rounded up"
        );

        let measured = recorded_u64(&record, "/value_removal_batch/measured_bytes");
        let commands = recorded_u64(&record, "/writer_batch_limits/restore_batch_commands");
        let arg_bytes = recorded_u64(&record, "/writer_batch_limits/restore_batch_arg_bytes");
        let per_arg_byte = recorded_f64(&record, "/writer_batch_limits/per_arg_byte");
        assert_eq!(
            VALUE_REMOVAL_BATCH_MEASURED_BYTES, measured,
            "the measured batch is the record's value_removal_batch.measured_bytes"
        );
        assert_eq!(
            measured,
            commands * per_command
                + (arg_bytes as f64 * (per_arg_byte * 100.0).round() / 100.0) as u64,
            "the measured batch is its command limit at the per-command cost plus its byte limit \
             at the measured memory per argument byte"
        );

        let batch = recorded_u64(&record, "/value_removal_batch/value_bytes");
        assert_eq!(
            DR_VALUE_REMOVAL_BATCH_BYTES, batch,
            "the charge is the record's value_removal_batch.value_bytes"
        );
        assert_eq!(
            batch,
            (measured as f64 * 1.25 / MIB as f64).ceil() as u64 * MIB,
            "the charge is 1.25 x the measured batch, rounded up to whole MiB"
        );
        assert_eq!(
            DR_VALUE_REMOVAL_BATCH_BYTES, DR_WRITE_CHARGE_BATCH_BYTES,
            "the value removal's batch is the writer's batch, at the same limits, so the two \
             charges agree"
        );
    }

    /// W-T1, the value removal's two admissions are the record's
    /// derivation (fail-on-old):
    /// - un-purged: `value_removal_admission(false)` is
    ///   [`DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES`], 158 MiB, the
    ///   record's `value_removal_unpurged_admission.value_bytes`, ⌈1.25 ×
    ///   its `measured_bytes`⌉ in whole MiB;
    /// - purged: `value_removal_admission(true)` is
    ///   [`DR_VALUE_REMOVAL_ADMISSION_BYTES`], the record's
    ///   `value_removal_admission.value_bytes`, the larger of its floor (one
    ///   batch, [`DR_VALUE_REMOVAL_BATCH_BYTES`], plus one page of
    ///   [`DR_VALUE_REMOVAL_PAGE_ROWS`] rows at `RETIREMENT_ROW_BYTES`) and
    ///   ⌈1.25 × the worst resident measured with the purge⌉ in whole MiB,
    ///   that worst being the largest of its member sizes' worsts;
    /// - the un-purged charge is at least the purged one.
    ///
    /// An edit to a constant, the selector or the record without the others
    /// fails here. Fails with the selector answering the purged figure
    /// whatever it is asked (in-place perturbation of
    /// [`value_removal_admission`]).
    #[test]
    fn value_removal_admissions_are_the_recorded_envelope_derivation() {
        let Some(record) = committed_record(RESTORE_ENVELOPE_DERIVATION_JSON) else {
            return;
        };

        // The un-purged fail-safe.
        let unpurged = recorded_u64(&record, "/value_removal_unpurged_admission/value_bytes");
        let measured = recorded_u64(&record, "/value_removal_unpurged_admission/measured_bytes");
        assert_eq!(
            value_removal_admission(false),
            DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES,
            "the selector's un-purged figure is the constant"
        );
        assert_eq!(
            DR_VALUE_REMOVAL_UNPURGED_ADMISSION_BYTES, unpurged,
            "the un-purged admission is the record's value_removal_unpurged_admission.value_bytes"
        );
        assert_eq!(unpurged, 158 * MIB, "158 MiB");
        assert_eq!(
            unpurged,
            recorded_u64(&record, "/value_removal_unpurged_admission/value_mib") * MIB,
            "the record's bytes are its whole MiB"
        );
        assert_eq!(
            unpurged,
            (5 * measured).div_ceil(4 * MIB) * MIB,
            "the un-purged admission is ⌈1.25 × the measured resident growth⌉ in whole MiB"
        );

        // The purged admission.
        let purged = recorded_u64(&record, "/value_removal_admission/value_bytes");
        let floor = recorded_u64(&record, "/value_removal_admission/floor_bytes");
        let worst = recorded_u64(&record, "/value_removal_admission/worst_resident_bytes");
        assert_eq!(
            value_removal_admission(true),
            DR_VALUE_REMOVAL_ADMISSION_BYTES,
            "the selector's purged figure is the constant"
        );
        assert_eq!(
            DR_VALUE_REMOVAL_ADMISSION_BYTES, purged,
            "the purged admission is the record's value_removal_admission.value_bytes"
        );
        assert_eq!(
            purged,
            recorded_u64(&record, "/value_removal_admission/value_mib") * MIB,
            "the record's bytes are its whole MiB"
        );
        assert_eq!(
            (
                recorded_u64(&record, "/value_removal_admission/batch_bytes"),
                recorded_u64(&record, "/value_removal_admission/page_rows"),
                recorded_u64(&record, "/value_removal_admission/row_bytes"),
            ),
            (
                DR_VALUE_REMOVAL_BATCH_BYTES,
                DR_VALUE_REMOVAL_PAGE_ROWS,
                RETIREMENT_ROW_BYTES
            ),
            "the floor's terms are the code's batch, page and row"
        );
        assert_eq!(
            floor,
            DR_VALUE_REMOVAL_BATCH_BYTES + DR_VALUE_REMOVAL_PAGE_ROWS * RETIREMENT_ROW_BYTES,
            "the floor is one batch plus one page of rows"
        );
        let shapes = record
            .pointer("/value_removal_admission/shapes")
            .and_then(serde_json::Value::as_object)
            .expect("the record carries the measured shapes");
        assert_eq!(shapes.len(), 3, "three member sizes measured");
        let worst_of_shapes = shapes
            .iter()
            .map(|(shape, block)| {
                block["worst_resident_bytes"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{shape} carries its worst resident bytes"))
            })
            .max()
            .expect("shapes");
        assert_eq!(
            worst, worst_of_shapes,
            "the worst is the largest of the member sizes' worsts"
        );
        assert_eq!(
            purged,
            floor.max((5 * worst).div_ceil(4 * MIB) * MIB),
            "the purged admission is max(floor, ⌈1.25 × the worst⌉ in whole MiB)"
        );

        assert!(
            value_removal_admission(false) >= value_removal_admission(true),
            "the un-purged charge is at least the purged one"
        );
    }

    /// Bytes mode at the reference limit with `parallelism` layers at once.
    fn bytes_mode(parallelism: usize) -> RestoreMemory {
        let line = dr_restore_line_at(
            5_120 * MIB,
            DR_RESTORE_SHARE_TEN_THOUSANDTHS,
            &PlanInputs::reference(),
        );
        RestoreMemory::Bytes {
            budget: line.restore_budget_bytes,
            parallelism: NonZeroUsize::new(parallelism).expect("non-zero"),
            line,
        }
    }

    fn serial_mode(parallelism: usize) -> RestoreMemory {
        RestoreMemory::Serial {
            reason: ReferenceReason::Unlimited,
            parallelism: NonZeroUsize::new(parallelism).expect("non-zero"),
        }
    }

    /// The byte bound is first come, first served: with 6 KiB of a 10 KiB
    /// budget held and a 6 KiB request waiting, a 1 KiB request that would
    /// fit waits behind it; the release serves both, in order.
    #[tokio::test]
    async fn restore_gate_admits_in_arrival_order_within_its_budget() {
        let gate = RestoreGate::new(&bytes_mode(8), Some(10 * KIB));
        assert_eq!(gate.budget_bytes(), Some(10 * KIB));
        let first = gate.admit(6 * KIB).await.expect("admitted at once");
        assert_eq!(first.bytes(), 6 * KIB);

        let mut second = pin!(gate.admit(6 * KIB));
        assert!(
            futures::poll!(second.as_mut()).is_pending(),
            "6 KiB more does not fit"
        );
        let mut third = pin!(gate.admit(KIB));
        assert!(
            futures::poll!(third.as_mut()).is_pending(),
            "1 KiB would fit, but waits behind the earlier request"
        );

        drop(first);
        let second = second.await.expect("served first");
        let third = third.await.expect("served second");
        assert_eq!((second.bytes(), third.bytes()), (6 * KIB, KIB));
        assert_eq!(gate.reserved_bytes(), 7 * KIB);
        assert_eq!(gate.in_flight(), 2);
        assert_eq!(gate.max_in_flight(), 2);
        assert_eq!(gate.peak_reserved_bytes(), 7 * KIB);
    }

    /// A charge above the whole budget is clamped to it: admitted at once
    /// when nothing is held, and then nothing else runs beside it.
    #[tokio::test]
    async fn restore_gate_runs_a_charge_above_the_whole_budget_alone() {
        let gate = RestoreGate::new(&bytes_mode(8), Some(10 * KIB));
        let oversized = gate.admit(50 * KIB).await.expect("admitted at once");
        assert_eq!(oversized.bytes(), 10 * KIB, "clamped to the whole budget");
        let mut small = pin!(gate.admit(KIB));
        assert!(
            futures::poll!(small.as_mut()).is_pending(),
            "nothing runs beside it"
        );
        drop(oversized);
        let small = small.await.expect("admitted once it ends");
        assert_eq!(small.bytes(), KIB);
        assert_eq!(gate.max_in_flight(), 1);
        assert_eq!(gate.in_flight(), 1);
        drop(small);
        assert_eq!(
            gate.in_flight(),
            0,
            "an admission gives its place back when dropped"
        );
        assert_eq!(gate.reserved_bytes(), 0);
    }

    /// The count bound is the stage's concurrency: the configured
    /// parallelism in bytes mode, 1 in serial mode, which has no byte bound
    /// and admits at once.
    #[tokio::test]
    async fn restore_gate_bounds_layers_by_parallelism_and_serial_mode_by_one() {
        let bytes = RestoreGate::new(&bytes_mode(3), None);
        assert_eq!(bytes.parallelism(), 3);
        assert_eq!(
            bytes.budget_bytes(),
            Some(bytes_mode(3).budget_bytes().unwrap())
        );

        let serial = RestoreGate::new(&serial_mode(8), Some(10 * KIB));
        assert_eq!(serial.parallelism(), 1);
        assert_eq!(
            serial.budget_bytes(),
            None,
            "the override applies to bytes mode only"
        );
        let held = serial.admit(u64::MAX).await.expect("no byte bound");
        assert_eq!(held.bytes(), 0);
        let also = serial.admit(u64::MAX).await.expect("still no byte bound");
        assert_eq!(serial.reserved_bytes(), 0);
        assert_eq!(
            serial.max_in_flight(),
            2,
            "the stage's concurrency, not the gate, is 1"
        );
        drop((held, also));

        assert_eq!(serial_mode(8).configured_parallelism(), 8);
        assert_eq!(serial_mode(8).describe(), "one layer at a time");
        assert_eq!(
            bytes_mode(8).describe(),
            "up to 8 layers at once within 2560 MiB"
        );
    }

    // ── U9: the planner's decode charge against real decodes ────────────────

    fn section(id: u64, parent: u64, name: &str, section_type: u8, order: u32) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: name.to_owned(),
            section_type,
            section_order: order,
            first_version: 1,
            last_version: 1,
        }
    }

    fn field(section_id: u64, key: &str, value: FieldValue, order: u64) -> Field {
        Field {
            section_id,
            key_name: key.to_owned(),
            value,
            set_order: order,
            authoring_set_order: order,
            version: 1,
        }
    }

    /// A state builder: sections under the pseudo-root 0, with their
    /// children lists kept.
    struct Builder {
        state: ObjectState,
        next_id: u64,
        next_order: u64,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                state: ObjectState {
                    sequence_num: 1,
                    first_version: 1,
                    source_format: "usda".to_owned(),
                    ..ObjectState::default()
                },
                next_id: 1,
                next_order: 0,
            }
        }

        fn add(&mut self, parent: u64, name: &str, section_type: u8) -> u64 {
            let id = self.next_id;
            self.next_id += 1;
            let siblings = self.state.children.entry(parent).or_default();
            let order = u32::try_from(siblings.len()).expect("fits");
            siblings.push(id);
            self.state
                .sections
                .insert(id, section(id, parent, name, section_type, order));
            id
        }

        fn set(&mut self, section_id: u64, key: &str, value: FieldValue) {
            self.next_order += 1;
            self.state.fields.insert(
                (section_id, key.to_owned()),
                field(section_id, key, value, self.next_order),
            );
        }

        fn sample(&mut self, section_id: u64, time: f64, value: FieldValue) {
            self.state.time_samples.insert(
                (section_id, OrderedFloat(time)),
                TimeSample {
                    section_id,
                    time: OrderedFloat(time),
                    value,
                    version: 1,
                },
            );
        }
    }

    const PRIM: u8 = 6;
    const ATTRIBUTE: u8 = 1;

    fn inline(len: usize) -> FieldValue {
        FieldValue::Inline(vec![0x5A; len])
    }

    /// A reference/xform layer: 400 xform prims under one root, each with
    /// its specifier, type name and a reference, and four xform-op
    /// attributes with a default, a type name and a variability each.
    fn ref_xform() -> ObjectState {
        let mut b = Builder::new();
        let world = b.add(0, "World", PRIM);
        for p in 0..400 {
            let prim = b.add(world, &format!("Xform_{p}"), PRIM);
            b.set(prim, "specifier", inline(1));
            b.set(prim, "typeName", inline(6));
            b.set(prim, "references", inline(64));
            for op in [
                "xformOp:translate",
                "xformOp:orient",
                "xformOp:scale",
                "xformOpOrder",
            ] {
                let attr = b.add(prim, op, ATTRIBUTE);
                b.set(attr, "default", inline(24));
                b.set(attr, "typeName", inline(8));
                b.set(attr, "variability", inline(1));
            }
        }
        b.state
    }

    /// A field-heavy layer: one prim carrying 3,000 small metadata fields.
    fn field_heavy() -> ObjectState {
        let mut b = Builder::new();
        let prim = b.add(0, "Meta", PRIM);
        for i in 0..3_000 {
            b.set(prim, &format!("customData:k{i}"), inline(4));
        }
        b.state
    }

    /// An animated layer: 50 attributes with 200 time samples each.
    fn time_sample_heavy() -> ObjectState {
        let mut b = Builder::new();
        let prim = b.add(0, "Anim", PRIM);
        for a in 0..50 {
            let attr = b.add(prim, &format!("attr{a}"), ATTRIBUTE);
            b.set(attr, "typeName", inline(8));
            for t in 0..200 {
                b.sample(attr, f64::from(t), inline(24));
            }
        }
        b.state
    }

    /// A mesh layer whose large values are external: 500 attributes whose
    /// default is an external value, and 500 time samples naming one.
    fn external_rows() -> ObjectState {
        let mut b = Builder::new();
        let prim = b.add(0, "Mesh", PRIM);
        for a in 0..500u32 {
            let attr = b.add(prim, &format!("points{a}"), ATTRIBUTE);
            let mut hash = [0u8; 32];
            hash[..4].copy_from_slice(&a.to_le_bytes());
            let external = FieldValue::External {
                hash,
                size: 1 << 20,
                ext_value_index: 1,
            };
            b.set(attr, "default", external.clone());
            b.sample(attr, 1.0, external);
        }
        b.state
    }

    /// U9 (PD-7): for each fixture shape, the decoded state's weight is at
    /// most `DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE` times the base it
    /// decodes from, so the planner's `plan_decode_charge` before a decode
    /// covers the decoded state's weight. Prints each measured ratio. The
    /// weigher's estimate, not the release-build heap the constant is
    /// calibrated on ([`restore_charges_are_the_recorded_envelope_derivation`]).
    #[test]
    fn decoded_state_weight_is_within_the_planners_decode_charge() {
        for (name, state) in [
            ("ref-xform", ref_xform()),
            ("field-heavy", field_heavy()),
            ("time-sample-heavy", time_sample_heavy()),
            ("external rows", external_rows()),
        ] {
            let (base, _) = crate::checkpoint::assembler::build_reset_ompf_from_state(&state)
                .expect("the base builds");
            let decoded = crate::checkpoint::apply::decode_checkpoint_ompf(7, &base)
                .expect("the base decodes");
            assert_eq!(
                (
                    decoded.sections.len(),
                    decoded.fields.len(),
                    decoded.time_samples.len()
                ),
                (
                    state.sections.len(),
                    state.fields.len(),
                    state.time_samples.len()
                ),
                "{name}: fixture: the decode holds every row"
            );
            let weight = crate::storage::structural_cache::estimate_state_bytes(&decoded);
            let fetched = u64::try_from(base.len()).expect("fits");
            let ratio = weight as f64 / fetched as f64;
            eprintln!(
                "U9 {name}: base {fetched} B, decoded state {weight} B, ratio {ratio:.2} \
                 (charge allows {DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE})"
            );
            assert!(
                weight <= DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE * fetched,
                "{name}: the decoded state, {weight} B, is above {} × its {fetched} B base \
                 (ratio {ratio:.2}): the planner's decode charge under-charges this shape",
                DR_PLAN_DECODE_CHARGE_PER_FETCHED_BYTE
            );
        }
    }
}
