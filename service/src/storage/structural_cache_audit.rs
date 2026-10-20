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

//! The structural cache's sampled audit: a metric-only correctness check that
//! re-reads an object after a committed structural call and compares the
//! store's state with the cache entry, repaired to the same cut
//! (structural-edits SDD SR-3.6, SR-2.4; the spec SDD's Mechanism lifecycle,
//! "Audit task"; its policy decision 3).
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Wiring
//!
//! `LayerStore` holds one [`StructuralCacheAudit`] per replica, installed by
//! `main.rs` with the configured rate
//! (`LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE`), the above-ceiling switch
//! (`LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING`), the upload gate's audit
//! charges (`grpc::upload_admission::UploadBudgetAuditCharges`) and the
//! service's background-task shutdown token. The trigger is
//! [`StructuralCacheAudit::maybe_spawn`] (through
//! `LayerStore::maybe_spawn_structural_cache_audit`), called by the
//! structural `UpdateLayer` route (`LayerStore::update_layer_once_structural`)
//! after a committed structural call, with the normalized successor entry it
//! is about to put back. A commit whose plan adopted a dangling parent puts
//! no successor back — it discards the entry — and starts no audit.
//!
//! # The audit task
//!
//! - **Started with probability `rate`, one at a time.** The trigger samples
//!   first, then claims the replica's one-audit flag (an `AtomicBool`); only
//!   then does it clone the entry, so a committed call that starts no audit
//!   pays no O(state) clone. The flag is released by RAII when the task ends,
//!   on every exit: completion, cancellation, or a panic unwinding out of it.
//! - **Cancelled by shutdown.** The task races the shutdown token; when it
//!   fires, the audit's future is dropped wherever it is — mid-read, in a
//!   fenced read's `RETRY_AFTER` sleep, mid-repair — ticks nothing, and
//!   releases what it held: a fenced read's lease, permits and memory grant
//!   included.
//! - **No admission permit, no writes.** It holds no per-object permit, no
//!   checkout and no upload admission: it works on its own clone, and holds
//!   its repair on the upload budget through its own charge, which never
//!   waits ("Memory"). It writes nothing to the store;
//!   on `diverged` it invalidates the object in the cache, which advances the
//!   cache-wide epoch so that any put-back by a checkout taken before the drop
//!   is refused (SR-3.4).
//!
//! # One audit
//!
//! 1. **Read** the object with the atomic snapshot read under its own
//!    consumer, `structural_cache_audit`, with `return_dangling`, so the
//!    header's `commit_chain` and the dangling parents come from the read's
//!    cut. Within the shared ceilings that read answers; it takes no
//!    cold-start permit, since at most one audit runs per replica. Above them
//!    the audit reads again only when the above-ceiling switch is on (the
//!    default), with the fenced read ("Above the plain ceilings" below);
//!    otherwise it is `skipped`.
//! 2. **Repair** its clone of the entry to the read's cut `(seq,
//!    commit_chain)` with the stream repair's repair-to-target
//!    ([`repair_to_target`](crate::storage::structural_cache_repair::repair_to_target)): the stream range capped at the cut's `seq`, the
//!    fold checked against the cut's `commit_chain`, under a structural
//!    call's repair bounds, its memory held on the upload budget first
//!    ("Memory"). A cut behind the entry, one more than the repair's budget
//!    (1,024 entries) ahead (`too_far`), one behind a pending entry larger
//!    than the repair's 4 MiB read cap (`too_large`), one the repair cannot
//!    reach, or one whose repair the upload budget does not hold is
//!    `skipped`.
//! 3. **Compare** the repaired, normalized state and dangling parents with
//!    the read's, decoded into the compact form and normalized the way a
//!    cold start builds its entry (a slim read: the plain read sends no
//!    inline value). Equal is `match`. Different is `diverged`: the entry is dropped
//!    (above) and one WARN names the object, the cut and the first
//!    differing key. Each part is scanned in place for its smallest
//!    differing key: beside the two states the compare holds only that key
//!    and the detail — no copy, no map, no sort.
//!
//! An object gone, replaced or without `chs_complete`, and a store error on
//! any read, is `skipped` too: the audit is a sample, not a retry loop.
//!
//! # Above the plain ceilings
//!
//! The cold start loads layers above the plain atomic-read ceilings with
//! b.y3m's fenced read, up to the per-entry weight cap, so the audit checks
//! them the same way: its plain read's `TOO_LARGE` is followed by
//! `read_object_snapshot_fenced` for the `structural_cache_audit` consumer,
//! under the cold start's own hook (`storage::structural_cache_fenced`). It
//! takes the same capacity in the same order — the cache's one-at-a-time and
//! plain cold-start permits, a fenced-class catch-up read permit (`P_f`) and
//! a memory grant inside the catch-up budget `T`, topped up (never waiting) as
//! the read's progress shows it heavier, adjusted after a verified read and
//! dropped once the snapshot is decoded — so its reply is charged
//! like a cold start's. Two differences keep it from costing what it checks:
//!
//! - **It never waits for capacity.** A pair, `P_f` or grant that is not free
//!   at once ends the read, and the audit is `skipped` (`busy`): an audit
//!   never queues ahead of a cold start or a catch-up. It still waits out a
//!   live fence's `RETRY_AFTER`, holding nothing, up to `reader_wait_max(L)`.
//! - **It is paced.** A fenced read pauses the layer's colliding writers for
//!   its hold, so an audit fenced read begins only when no other on the
//!   replica was granted a lease within
//!   [`STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL`]; an audit that would read
//!   fenced sooner is `skipped` (`paced`). The interval runs from the grant —
//!   the start of a pause — so an audit that never got a lease (busy, gone,
//!   an unmarked object, refused before its grant) uses up no interval.
//!
//! Each such audit ticks
//! `live_edit_structural_cache_audit_fenced_total{result=read|paced|busy|failed}`
//! (not an unmarked object, skipped before the fenced read; not an object
//! gone during it, which is `skipped` as within the plain ceilings; not a
//! read ended by shutdown),
//! and every fenced read shows on
//! `live_edit_snapshot_read_*{consumer="structural_cache_audit",path="fenced"}`
//! with its hold — the writers' pause — on
//! `live_edit_snapshot_fence_hold_seconds{consumer="structural_cache_audit"}`.
//!
//! # Metrics (SR-2.4)
//!
//! Each audit that runs to an end ticks
//! `live_edit_structural_cache_audit_total{outcome=match|diverged|skipped}`
//! once, and nothing else of the cache's: not lookups, reloads, replayed
//! entries, repair bytes, divergence or cold-start reads. Its snapshot reads
//! show on `live_edit_snapshot_read_*{consumer="structural_cache_audit"}`, so
//! `consumer="structural_cache"` stays cold starts only. A fenced read shares
//! the cold-start permits with cold starts but is not a cold start. Above the
//! plain ceilings, `live_edit_structural_cache_audit_fenced_total` says
//! whether the audit read fenced ("Above the plain ceilings").
//!
//! # Memory
//!
//! A running audit holds its clone of the entry and the compact state of its
//! read — about twice the entry's weight — and, while it decodes, the raw
//! snapshot reply. A plain read's reply is bounded by the plain byte ceiling;
//! a fenced read's is charged against the catch-up budget `T` through its
//! memory grant until it is decoded. Those two states and the reply are the
//! sampled audit's term in SDD §21.2's line (two entries at the per-entry
//! cap and one plain reply, `config::memory_plan`), outside the cache's
//! budget; at most one audit runs per replica.
//!
//! Its repair to the read's cut is held on the upload budget `B` instead,
//! beside the uploads: once the read has answered, the audit opens one charge
//! (`storage::commit_charge::AuditChargeSource`; on the replica,
//! `grpc::upload_admission::UploadBudgetAuditCharges`), and the repair holds
//! each read, round and record in it before allocating it, priced as a
//! structural call's repair is (`storage::structural_cache_repair`, "The
//! charge points") less the copy's own weight, which the line's term above
//! funds. Its copy holds no plan indexes (`CacheEntry::copy_without_plan`),
//! so each structural record also pays the per-delta name and adoption
//! indexes and the full depth walk; the copy is never re-weighed, and its
//! growth is held as priced record by record.
//!
//! **The copy's bucket bound.** The copy is cloned from the committed
//! successor before its put-back compacts it, so its hash tables keep the
//! successor's tombstones, and the records its repair replays can leave
//! more: their `capacity()` can read fewer buckets than they hold, and a
//! growth priced from it would double from too small a table. The successor
//! weighs each table at its held capacity (its plan indexes' capacity
//! high-water), which the copy, without indexes, does not keep. So the
//! trigger reads those held capacities before it copies
//! (`CacheEntry::held_state_capacities`), and the repair prices each
//! record's tables from a per-table bucket bound started there
//! (`structural_cache::CopyBucketBound`, whose docs carry its raise rule and
//! why it never falls below the tables' true bucket counts). It is a stack
//! value of the repair, created when the repair starts and dropped with it,
//! and raised after each record, never past the bucket count the record's
//! growth was priced to from the bound. A bound above the truth only prices
//! a record's growth from a larger table than it holds — more skips at the
//! margin, never memory.
//!
//! The charge never waits, never
//! takes the upload overflow slot, is refused while an upload is queued or
//! when it would leave less than one upload admission's room, and is capped
//! at one admission per audit
//! (`grpc::upload_admission::SAMPLED_AUDIT_CHARGE_CAP_BYTES`). It is given
//! back after the compare — before a `diverged` invalidates the entry — and
//! on every skip, error, cancellation and panic. The compare itself holds
//! only the smallest differing key it has found and the detail, so a
//! `diverged` adds nothing to the audit's term.
//!
//! **Coverage.** A repair the upload budget does not hold — refused, or over
//! the cap — is `skipped`, as is one past the repair's entry budget
//! (`too_far`) or behind an entry above its 4 MiB read cap (`too_large`)
//! (`live_edit_structural_cache_audit_total{outcome="skipped"}`). Those
//! skips concentrate on busy layers, whose pending records are large or
//! many between a commit and its audit's read, on large layers, whose
//! structural records cost the copy's per-delta indexes and walk, and on a
//! replica whose uploads fill `B`, so the audit's coverage is lowest there.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rand::Rng;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::domain::replay::normalize_state;
use crate::domain::types::{CompactFields, CompactSamples, CompactState, FieldRows, SampleRows};
use crate::error::ServiceError;
use crate::observability::metrics::{
    STRUCTURAL_CACHE_AUDIT_DIVERGED, STRUCTURAL_CACHE_AUDIT_FENCED_BUSY,
    STRUCTURAL_CACHE_AUDIT_FENCED_FAILED, STRUCTURAL_CACHE_AUDIT_FENCED_PACED,
    STRUCTURAL_CACHE_AUDIT_FENCED_READ, STRUCTURAL_CACHE_AUDIT_MATCH,
    STRUCTURAL_CACHE_AUDIT_SKIPPED,
};
use crate::storage::client::ValkeyClient;
use crate::storage::commit_charge::{
    AuditCharge, AuditChargeSource, RefusingAuditCharges, SharedCommitCharge,
};
use crate::storage::diff_read::{
    read_object_snapshot_slim_with_ceilings, AtomicSnapshotOf, SlimSnapshotReadConsumer,
    SnapshotCeilings, SnapshotReadOf,
};
use crate::storage::keys::KeySchema;
use crate::storage::snapshot_fence::read::read_object_snapshot_fenced;
use crate::storage::snapshot_fence::LayerSize;
use crate::storage::structural_cache::{CacheEntry, HeldCapacities, StructuralCache};
use crate::storage::structural_cache_fenced::{
    failed_for_the_call, ColdStartFenceHook, FencedFailure,
};
#[cfg(any(test, feature = "test-support"))]
use crate::storage::structural_cache_repair::TestHook;
use crate::storage::structural_cache_repair::{
    repair_to_target_with, ReloadReason, RepairBounds, RepairHooks, TargetRepair,
};

/// The shortest time between two of a replica's audit fenced reads (module
/// documentation, "Above the plain ceilings"): 10 minutes, the cache's idle
/// TTL. A fenced read pauses the layer's colliding writers for about one
/// polling step (about 1.1 s; the same mechanism as a fenced cold start,
/// measured in the fenced-read Epic's Task 10), so the audit's are paced to one per replica per interval, whatever the audit
/// rate and the edit rate: at the demo's one edit per second, about one
/// paused edit per replica per interval. An audit that would read fenced
/// sooner is `skipped` (`paced`).
pub const STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL: Duration = Duration::from_secs(600);

/// The test-only observer of each audit's end
/// ([`StructuralCacheAudit::with_outcome_observer_for_test`]): the outcome,
/// and the most its charge held on the upload budget at once.
#[cfg(any(test, feature = "test-support"))]
pub type AuditOutcomeObserver = Arc<dyn Fn(&AuditOutcome, u64) + Send + Sync>;

/// The replica's sampled audit of its structural cache (module
/// documentation): the rate, the above-ceiling switch, the one-audit flag,
/// the fenced reads' pacing, where each audit's charge on the upload budget
/// comes from, and the shutdown token audits are cancelled by.
///
/// Cheap to [`Clone`]: clones share the one-audit flag, the charges and the
/// token, so the store and every clone of it audit one at a time.
#[derive(Clone)]
pub struct StructuralCacheAudit {
    /// The probability a committed structural call starts an audit, in
    /// `[0, 1]`.
    rate: f64,
    /// Whether an audit of an object above the shared atomic-read ceilings
    /// reads it with the fenced read (else it is `skipped`).
    above_ceiling: bool,
    /// The shortest time between two audit fenced reads on this replica.
    fenced_min_interval: Duration,
    /// When the last audit fenced read on this replica began; shared by
    /// every clone.
    last_fenced: Arc<Mutex<Option<tokio::time::Instant>>>,
    /// The bounds of the repair to the read's cut.
    repair_bounds: RepairBounds,
    /// Where each audit opens its charge on the upload budget.
    charges: Arc<dyn AuditChargeSource>,
    /// Set while an audit runs; cleared by [`RunningGuard`].
    running: Arc<AtomicBool>,
    /// Cancels a running audit; no audit starts once it has fired.
    shutdown: CancellationToken,
    /// Test seam: run inside the audit task, before its first read.
    #[cfg(any(test, feature = "test-support"))]
    before_read_hook: Option<TestHook>,
    /// Test seam: told each audit's outcome and its charge's peak.
    #[cfg(any(test, feature = "test-support"))]
    outcome_observer: Option<AuditOutcomeObserver>,
}

impl StructuralCacheAudit {
    /// An audit started with probability `rate` after each committed
    /// structural call, reading above the shared ceilings only when
    /// `above_ceiling`, repairing under the production repair bounds with its
    /// repair held on the upload budget through a charge from `charges`
    /// (module documentation, "Memory"), and cancelled when `shutdown` fires.
    /// `main.rs` passes the upload gate's
    /// (`grpc::upload_admission::UploadBudgetAuditCharges`).
    ///
    /// # Panics
    ///
    /// If `rate` is not in `[0, 1]` (NaN included): `Config::from_env`
    /// refuses such a `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE` at startup, and
    /// sampling it would panic later.
    pub fn new(
        rate: f64,
        above_ceiling: bool,
        charges: Arc<dyn AuditChargeSource>,
        shutdown: CancellationToken,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&rate),
            "the structural cache audit rate must be a probability in [0, 1], got {rate}"
        );
        Self {
            rate,
            above_ceiling,
            fenced_min_interval: STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL,
            last_fenced: Arc::new(Mutex::new(None)),
            repair_bounds: RepairBounds::PRODUCTION,
            charges,
            running: Arc::new(AtomicBool::new(false)),
            shutdown,
            #[cfg(any(test, feature = "test-support"))]
            before_read_hook: None,
            #[cfg(any(test, feature = "test-support"))]
            outcome_observer: None,
        }
    }

    /// An audit that never starts (rate 0), with charges that refuse every
    /// growth and a token nothing cancels: what `LayerStore::new` installs
    /// until `main.rs` installs the configured audit with the service's
    /// upload gate and shutdown token. A store built without it, as tests
    /// build one, then reads nothing it was not asked to.
    pub fn disabled() -> Self {
        Self::new(
            0.0,
            false,
            Arc::new(RefusingAuditCharges),
            CancellationToken::new(),
        )
    }

    /// The same audit with the repair to the read's cut under `bounds`.
    /// Production keeps [`RepairBounds::PRODUCTION`]; a test narrows a bound
    /// to reach it without writing a thousand entries.
    pub fn with_repair_bounds(mut self, bounds: RepairBounds) -> Self {
        self.repair_bounds = bounds;
        self
    }

    /// **Test-only.** Call `observer` at the end of each audit that runs to
    /// an end, with its outcome and the most its charge held on the upload
    /// budget at once (`AuditCharge::peak`), after the charge is given back.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_outcome_observer_for_test(mut self, observer: AuditOutcomeObserver) -> Self {
        self.outcome_observer = Some(observer);
        self
    }

    /// The same audit with its fenced reads paced at `interval` instead of
    /// [`STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL`]. Production keeps the
    /// constant; a test shortens or lengthens it to reach both sides.
    pub fn with_fenced_min_interval(mut self, interval: Duration) -> Self {
        self.fenced_min_interval = interval;
        self
    }

    /// **Test-only.** Run `hook` inside each audit task before its first
    /// read, within the part the shutdown token cancels: a test can hold an
    /// audit running there, panic it, or change the store before it reads.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_before_read_hook_for_test(mut self, hook: TestHook) -> Self {
        self.before_read_hook = Some(hook);
        self
    }

    /// The probability a committed structural call starts an audit.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Whether an audit may read above the shared atomic-read ceilings.
    pub fn above_ceiling(&self) -> bool {
        self.above_ceiling
    }

    /// Whether an audit is running on this replica now.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// The trigger, after a committed structural call (SR-3.6): with
    /// probability [`Self::rate`], and only when no audit is running and
    /// shutdown has not begun, clone `entry` — the call's successor entry,
    /// borrowed before it is put back — and spawn one audit of it (module
    /// documentation). Otherwise do nothing, and clone nothing.
    ///
    /// Returns the spawned task's handle, or `None` when no audit started.
    /// The task answers the audit's [`AuditOutcome`], or `None` when the
    /// shutdown token cancelled it. A caller that does not wait for the audit
    /// drops the handle; the task runs on.
    ///
    /// **Memory.** The clone is O(state) and lives until the task ends,
    /// beside the decoded state of the audit's read (module documentation,
    /// "Memory").
    ///
    /// # Panics
    ///
    /// When it starts an audit outside a Tokio runtime, as `tokio::spawn`
    /// does. The one-audit flag is released either way.
    pub fn maybe_spawn(
        &self,
        client: &ValkeyClient,
        keys: &KeySchema,
        cache: &StructuralCache,
        entry: &CacheEntry,
    ) -> Option<JoinHandle<Option<AuditOutcome>>> {
        if self.rate <= 0.0 || self.shutdown.is_cancelled() {
            return None;
        }
        if !rand::rng().random_bool(self.rate) {
            return None;
        }
        // Claimed before the copy, so a panic from here on still releases it.
        // The copy is the cut and the state only, no plan indexes: the audit
        // repairs it unindexed. The capacities the entry weighs its tables at
        // are read first, while it still holds its indexes' capacity
        // high-water: the clone keeps the entry's tombstones but not the
        // high-water, so they start the copy's bucket bound
        // (`structural_cache::CopyBucketBound`).
        let running = RunningGuard::claim(&self.running)?;
        // The entry is the successor `CheckedOut::advance_committed` settled
        // from a plan in place, which builds the indexes of an entry that has
        // none, so it holds them: its held capacities are a bound, where an
        // index-less entry's `capacity()` readings would not be.
        debug_assert!(
            entry.plan.is_some(),
            "the audit's entry holds its plan indexes, so its held capacities bound its tables"
        );
        let start = entry.held_state_capacities();
        let entry = entry.copy_without_plan();
        let client = client.clone();
        let keys = keys.clone();
        let cache = cache.clone();
        let shutdown = self.shutdown.clone();
        let above_ceiling = self.above_ceiling.then(|| FencedPacing {
            min_interval: self.fenced_min_interval,
            last: Arc::clone(&self.last_fenced),
        });
        let repair_bounds = self.repair_bounds;
        let charges = Arc::clone(&self.charges);
        #[cfg(any(test, feature = "test-support"))]
        let before_read_hook = self.before_read_hook.clone();
        #[cfg(any(test, feature = "test-support"))]
        let outcome_observer = self.outcome_observer.clone();
        Some(tokio::spawn(async move {
            // Dropped when the task ends, however it ends.
            let _running = running;
            let audit = async {
                #[cfg(any(test, feature = "test-support"))]
                if let Some(hook) = before_read_hook {
                    hook().await;
                }
                let (outcome, _peak) = audit_entry(
                    &client,
                    &keys,
                    &cache,
                    entry,
                    start,
                    above_ceiling.as_ref(),
                    repair_bounds,
                    &*charges,
                )
                .await;
                #[cfg(any(test, feature = "test-support"))]
                if let Some(observer) = outcome_observer {
                    observer(&outcome, _peak);
                }
                outcome
            };
            tokio::select! {
                biased;
                () = shutdown.cancelled() => None,
                outcome = audit => Some(outcome),
            }
        }))
    }
}

impl fmt::Debug for StructuralCacheAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralCacheAudit")
            .field("rate", &self.rate)
            .field("above_ceiling", &self.above_ceiling)
            .field("fenced_min_interval", &self.fenced_min_interval)
            .field("repair_bounds", &self.repair_bounds)
            .field("running", &self.is_running())
            .finish_non_exhaustive()
    }
}

/// The replica's one-audit flag, held by the running audit and released when
/// it drops: on completion, cancellation, or a panic's unwind.
struct RunningGuard {
    flag: Arc<AtomicBool>,
}

impl RunningGuard {
    /// Claim `flag`, or `None` when an audit already holds it.
    fn claim(flag: &Arc<AtomicBool>) -> Option<Self> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self {
                flag: Arc::clone(flag),
            })
    }
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// What one audit concluded: the `outcome` label of
/// `live_edit_structural_cache_audit_total`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome {
    /// The entry, repaired to the read's cut, equals the store's state there.
    Match,
    /// The repaired entry differs from the store's state at cut `seq`; the
    /// entry was dropped. `first_difference` names the first differing key,
    /// in the order the comparison walks them (the object's own columns,
    /// sections, children lists, fields, time samples, dangling parents).
    Diverged {
        /// The cut compared at.
        seq: u64,
        /// The first differing key, e.g. `section 12`.
        first_difference: String,
    },
    /// The audit could not compare.
    Skipped(AuditSkip),
}

impl AuditOutcome {
    /// The metric label (a
    /// [`crate::observability::metrics::STRUCTURAL_CACHE_AUDIT_OUTCOMES`]
    /// value).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Match => STRUCTURAL_CACHE_AUDIT_MATCH,
            Self::Diverged { .. } => STRUCTURAL_CACHE_AUDIT_DIVERGED,
            Self::Skipped(_) => STRUCTURAL_CACHE_AUDIT_SKIPPED,
        }
    }
}

/// Why an audit could not compare. Every one counts `skipped`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditSkip {
    /// The object is above the shared atomic-read ceilings and the
    /// above-ceiling switch is off.
    AboveCeiling,
    /// Above the shared ceilings, an audit fenced read already ran on this
    /// replica within [`STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL`].
    Paced,
    /// Above the shared ceilings, the cache's cold-start permits, the fenced
    /// read permit or the catch-up memory were not free at once: the audit
    /// never waits for them.
    Busy,
    /// Above the shared ceilings, the fenced read failed: its wait for the
    /// fence passed `reader_wait_max(L)`, or every attempt was broken.
    FencedReadFailed,
    /// The object's header is absent, has no decimal `seq`, or is superseded.
    Gone,
    /// The object's header lacks `chs_complete`, so its dangling children
    /// lists cannot be enumerated.
    NoChsComplete,
    /// The repair could not bring the entry to the read's cut. Carries
    /// whichever [`ReloadReason`] stopped [`repair_to_target`](crate::storage::structural_cache_repair::repair_to_target): `Gap`,
    /// `Identity`, `Regressed`, `Tombstone`, `Parse`, `ReplayErr`,
    /// `DanglingAdopted`, `TooLarge` or `TooFar` (the stream repair module's
    /// checks, with the cut in place of the header). For example,
    /// [`ReloadReason::Regressed`] for a cut behind the entry or a fold that
    /// misses the cut's `commit_chain`, and [`ReloadReason::TooFar`] for a cut
    /// beyond the repair's entry budget.
    Unreachable(ReloadReason),
    /// The repair's charge on the upload budget was refused within the
    /// audit's cap: an upload was queued, or holding it would have left less
    /// than one upload admission's room. The audit never waits for it.
    ChargeRefused,
    /// The repair needed more of the upload budget than one audit may hold
    /// (`AuditCharge::cap`; `grpc::upload_admission::SAMPLED_AUDIT_CHARGE_CAP_BYTES`).
    ChargeOverCap,
    /// A read failed on a store error.
    ReadFailed,
}

/// One audit of `entry`, a clone of a cache entry (module documentation, "One
/// audit"): read, repair under a charge from `charges`, compare; on
/// `diverged` invalidate the object in `cache` and WARN; tick
/// `live_edit_structural_cache_audit_total` once. `start` is the capacities
/// the cloned entry weighed its tables at (`CacheEntry::held_state_capacities`),
/// where the copy's bucket bound starts ("Memory"). Answers the outcome and
/// the most the audit's charge held at once; the charge is given back before
/// the invalidation and the tick.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn audit_entry(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    entry: CacheEntry,
    start: HeldCapacities,
    above_ceiling: Option<&FencedPacing>,
    repair_bounds: RepairBounds,
    charges: &dyn AuditChargeSource,
) -> (AuditOutcome, u64) {
    let object_id = entry.object_id;
    let (outcome, peak) = compare_with_store(
        client,
        keys,
        cache,
        entry,
        start,
        above_ceiling,
        repair_bounds,
        charges,
    )
    .await;
    if let AuditOutcome::Diverged {
        seq,
        first_difference,
    } = &outcome
    {
        // Before the log: the drop is what protects the next plan.
        cache.invalidate(object_id);
        tracing::warn!(
            object_id,
            seq,
            first_difference = %first_difference,
            "structural cache audit: the cached state differs from the store at the same cut; \
             the cache entry was dropped and the object will be read again"
        );
    }
    if let Some(m) = client.metrics_snapshot() {
        m.inc_structural_cache_audit(outcome.label());
    }
    (outcome, peak)
}

/// The store's state at one cut, decoded into the compact form and
/// normalized as a cold start builds its entry.
struct FreshCut {
    seq: u64,
    commit_chain: String,
    dangling_parents: HashSet<u64>,
    state: CompactState,
}

/// Steps 1–3 of one audit, without its effects, and the most the audit's
/// charge held at once.
///
/// The charge is opened once the read has answered, net of the copy's own
/// weight (its stored state weight, which the line's audit term funds), and
/// held only by this function: it is given back when the function returns —
/// after the compare, on every skip and error — or when the shutdown drops
/// the audit's future, or a panic unwinds it, by its `Drop`. The repair's
/// bucket bound on the copy starts at `start` ([`audit_entry`]).
#[allow(clippy::too_many_arguments)]
async fn compare_with_store(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    entry: CacheEntry,
    start: HeldCapacities,
    above_ceiling: Option<&FencedPacing>,
    repair_bounds: RepairBounds,
    charges: &dyn AuditChargeSource,
) -> (AuditOutcome, u64) {
    let fresh = match read_fresh(client, keys, cache, entry.object_id, above_ceiling).await {
        Ok(fresh) => fresh,
        Err(skip) => return (AuditOutcome::Skipped(skip), 0),
    };
    let charge: Box<dyn AuditCharge> = charges.open(entry.state_weight_bytes);
    let mut repaired = entry;
    // A test-support build checks the copy's bucket bound after every record
    // while the cache's shadow check is on; a memory-measuring harness turns
    // it off, since the check clones every table.
    #[cfg(any(test, feature = "test-support"))]
    let hooks = RepairHooks {
        between_reads: None,
        copy_bound_check: cache.shadow_check_for_test(),
    };
    #[cfg(not(any(test, feature = "test-support")))]
    let hooks = RepairHooks::default();
    let repair = repair_to_target_with(
        client,
        keys,
        &mut repaired,
        fresh.seq,
        &fresh.commit_chain,
        repair_bounds,
        Some(&*charge as &dyn SharedCommitCharge),
        Some(start),
        &hooks,
    )
    .await;
    let skip = match repair {
        Ok((TargetRepair::Reached, _)) => None,
        Ok((TargetRepair::Unreachable(reason), _)) => Some(AuditSkip::Unreachable(reason)),
        Ok((TargetRepair::Gone(_), _)) => Some(AuditSkip::Gone),
        Ok((TargetRepair::ChargeRefused { target }, _)) if target > charge.cap() => {
            Some(AuditSkip::ChargeOverCap)
        }
        Ok((TargetRepair::ChargeRefused { .. }, _)) => Some(AuditSkip::ChargeRefused),
        Err(_) => Some(AuditSkip::ReadFailed),
    };
    let outcome = match skip {
        Some(skip) => AuditOutcome::Skipped(skip),
        None => match first_difference(
            &repaired.state,
            &repaired.dangling_parents,
            &fresh.state,
            &fresh.dangling_parents,
        ) {
            None => AuditOutcome::Match,
            Some(first_difference) => AuditOutcome::Diverged {
                seq: fresh.seq,
                first_difference,
            },
        },
    };
    let peak = charge.peak();
    // Given back after the compare, before the caller acts on its outcome.
    drop(charge);
    (outcome, peak)
}

/// The above-ceiling switch's state an audit task carries: the pacing of
/// the replica's audit fenced reads.
pub(crate) struct FencedPacing {
    min_interval: Duration,
    last: Arc<Mutex<Option<tokio::time::Instant>>>,
}

impl FencedPacing {
    /// Whether the replica may begin an audit fenced read now: none was
    /// granted a lease within the interval.
    fn due(&self) -> bool {
        let last = *self.last.lock().unwrap_or_else(PoisonError::into_inner);
        last.is_none_or(|at| {
            tokio::time::Instant::now().saturating_duration_since(at) >= self.min_interval
        })
    }

    /// Record that an audit fenced read was granted a lease at `at`: the
    /// writers' pause the interval paces. An audit that never got a lease
    /// (busy, gone, refused before its grant) paused no one and records
    /// nothing. Only one audit runs per replica, so `due` and `record` never
    /// race with another audit's.
    fn record(&self, at: tokio::time::Instant) {
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(at);
    }
}

/// Step 1: the object's state at one cut, from the plain atomic read, which
/// takes no permit, or — above the shared ceilings, with the above-ceiling
/// switch on — the fenced read (module documentation, "Above the plain
/// ceilings").
async fn read_fresh(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    object_id: u64,
    above_ceiling: Option<&FencedPacing>,
) -> Result<FreshCut, AuditSkip> {
    let oid = object_id.to_string();
    let plain = SnapshotCeilings::of_client(client);
    let snapshot = match read(client, keys, &oid, plain).await? {
        SnapshotReadOf::Snapshot(snapshot) => *snapshot,
        SnapshotReadOf::TooLarge(too_large) => {
            let Some(pacing) = above_ceiling else {
                return Err(AuditSkip::AboveCeiling);
            };
            // An unmarked object cannot be compared at any size: no fenced
            // read, and no pacing slot, for it.
            if too_large.cardinalities.children_index.is_none() {
                return Err(AuditSkip::NoChsComplete);
            }
            let metrics = client.metrics_snapshot();
            let tick = |result: &'static str| {
                if let Some(m) = metrics.as_ref() {
                    m.inc_structural_cache_audit_fenced(result);
                }
            };
            if !pacing.due() {
                tick(STRUCTURAL_CACHE_AUDIT_FENCED_PACED);
                return Err(AuditSkip::Paced);
            }
            let estimate = LayerSize::from_too_large(&too_large, None);
            let mut hook = ColdStartFenceHook::for_audit(client, cache, estimate);
            let read = read_object_snapshot_fenced(
                client,
                keys,
                object_id,
                SlimSnapshotReadConsumer::StructuralCacheAudit,
                estimate,
                &mut hook,
            )
            .await;
            if let Some(granted) = hook.first_grant() {
                pacing.record(granted);
            }
            match read {
                Ok(mut snapshot) => {
                    tick(STRUCTURAL_CACHE_AUDIT_FENCED_READ);
                    // The read's permits and memory, charged in `T` like a
                    // cold start's, go with the snapshot: dropped once it is
                    // decoded, before the repair.
                    let hold = hook.take_hold();
                    drop(hook);
                    if let Some(hold) = hold {
                        drop(hold.read_permits);
                        snapshot.memory = hold.memory;
                    }
                    snapshot
                }
                Err(error) => {
                    let skip = match hook.failure() {
                        Some(FencedFailure::CapacityTimeout) => {
                            tick(STRUCTURAL_CACHE_AUDIT_FENCED_BUSY);
                            AuditSkip::Busy
                        }
                        // The capacity closed at shutdown: not a result of
                        // the audit's, like a cancellation.
                        Some(FencedFailure::Closed) => AuditSkip::ReadFailed,
                        _ if failed_for_the_call(&hook, &error).is_some() => {
                            tick(STRUCTURAL_CACHE_AUDIT_FENCED_FAILED);
                            AuditSkip::FencedReadFailed
                        }
                        // The object went away during the read: `gone`, as
                        // within the plain ceilings, and not a fenced
                        // result.
                        _ if matches!(error, ServiceError::NotFound(_)) => AuditSkip::Gone,
                        _ => {
                            tick(STRUCTURAL_CACHE_AUDIT_FENCED_FAILED);
                            AuditSkip::ReadFailed
                        }
                    };
                    return Err(skip);
                }
            }
        }
    };
    let AtomicSnapshotOf {
        mut state,
        cut,
        structural,
        memory,
        ..
    } = snapshot;
    // Always `Some` for this consumer, which sets `return_dangling`.
    let Some(facts) = structural else {
        return Err(AuditSkip::ReadFailed);
    };
    if facts.is_superseded() {
        return Err(AuditSkip::Gone);
    }
    if !facts.chs_complete_marked() {
        return Err(AuditSkip::NoChsComplete);
    }
    // The read decoded straight into the compact state; the memory a fenced
    // read's grant held for its reply goes before the repair.
    drop(memory);
    // As the cold start seeds an entry's dangling parents (see
    // `storage::structural_cache_cold_start`): the read's element, plus any
    // list the normalization drops.
    let mut dangling_parents: HashSet<u64> = facts.dangling_parents.into_iter().collect();
    dangling_parents.extend(normalize_state(&mut state));
    Ok(FreshCut {
        seq: cut,
        commit_chain: facts.commit_chain,
        dangling_parents,
        state,
    })
}

/// One atomic read of `oid` as the audit's consumer, at `ceilings`. An absent
/// header, or one without a decimal `seq`, is [`AuditSkip::Gone`]; any other
/// error is [`AuditSkip::ReadFailed`] (the read has counted it).
async fn read(
    client: &ValkeyClient,
    keys: &KeySchema,
    oid: &str,
    ceilings: SnapshotCeilings,
) -> Result<SnapshotReadOf<CompactState>, AuditSkip> {
    read_object_snapshot_slim_with_ceilings(
        client,
        keys,
        oid,
        SlimSnapshotReadConsumer::StructuralCacheAudit,
        ceilings,
    )
    .await
    .map_err(|error| match error {
        ServiceError::NotFound(_) => AuditSkip::Gone,
        _ => AuditSkip::ReadFailed,
    })
}

/// The first key at which the repaired entry (`cached`, `cached_dangling`)
/// and the store's state (`fresh`, `fresh_dangling`) differ, or `None` when
/// they are equal. Walks the object's own columns, then sections, children
/// lists, fields and time samples — each part's smallest differing key —
/// then the dangling parents. Value rows are compared by their resolved
/// keys — a field by `(section_id, name)`, names in byte order, a time sample
/// by `(section_id, time bits)` — and their external hashes, never by the
/// interned name ids, which each state numbers on its own.
///
/// Each part is scanned in place for its smallest differing key: beside the
/// two states it holds only that key and the detail — no copy, no map, no
/// sort — so a `diverged` adds nothing to the audit's memory term.
fn first_difference(
    cached: &CompactState,
    cached_dangling: &HashSet<u64>,
    fresh: &CompactState,
    fresh_dangling: &HashSet<u64>,
) -> Option<String> {
    if cached == fresh && cached_dangling == fresh_dangling {
        return None;
    }
    let columns = [
        ("object_id", cached.object_id != fresh.object_id),
        ("sequence_num", cached.sequence_num != fresh.sequence_num),
        ("first_version", cached.first_version != fresh.first_version),
        ("source_format", cached.source_format != fresh.source_format),
        (
            "next_set_order",
            cached.next_set_order != fresh.next_set_order,
        ),
    ];
    if let Some((column, _)) = columns.iter().find(|(_, differs)| *differs) {
        return Some(format!("object column {column}"));
    }
    if let Some(section_id) = min_differing_key(&cached.sections, &fresh.sections) {
        return Some(format!("section {section_id}"));
    }
    if let Some(parent_id) = min_differing_key(&cached.children, &fresh.children) {
        return Some(format!("children list of {parent_id}"));
    }
    if let Some((section_id, name)) = min_differing_field(&cached.fields, &fresh.fields) {
        return Some(format!("field {name:?} of section {section_id}"));
    }
    if let Some((section_id, bits)) =
        min_differing_sample(&cached.time_samples, &fresh.time_samples)
    {
        let time = f64::from_bits(bits);
        return Some(format!("time sample at {time} of section {section_id}"));
    }
    if let Some(parent_id) = cached_dangling.symmetric_difference(fresh_dangling).min() {
        return Some(format!("dangling parent {parent_id}"));
    }
    // Unequal, yet every part walked above is equal: the state has a part
    // this walk does not name.
    Some("object state".to_owned())
}

/// The smallest key whose value differs between `a` and `b`, a key present
/// in only one of them included; `None` when they are equal. Two passes in
/// place — `a`'s keys against `b`, then `b`'s keys missing from `a` — keeping
/// only the running minimum: in hash order the first difference found is
/// random, so neither pass stops early.
fn min_differing_key<K, V>(a: &HashMap<K, V>, b: &HashMap<K, V>) -> Option<K>
where
    K: Ord + Hash + Eq + Copy,
    V: PartialEq,
{
    let differs_or_only_in_a = a
        .iter()
        .filter(|(key, value)| b.get(*key) != Some(*value))
        .map(|(key, _)| *key);
    let only_in_b = b.keys().filter(|key| !a.contains_key(*key)).copied();
    differs_or_only_in_a.chain(only_in_b).min()
}

/// [`min_differing_key`] over two states' field rows, keyed by `(section_id,
/// name)` with names in byte order: a row of `a` that `b` lacks or holds
/// with another external hash, or a row of `b` that `a` lacks. The compact
/// rows' lookups allocate nothing; only the current minimum's name is owned.
fn min_differing_field(a: &CompactFields, b: &CompactFields) -> Option<(u64, String)> {
    let mut least: Option<(u64, String)> = None;
    let mut offer = |section_id: u64, name: &str| {
        let smaller = least.as_ref().is_none_or(|(least_section, least_name)| {
            (section_id, name) < (*least_section, least_name.as_str())
        });
        if smaller {
            least = Some((section_id, name.to_owned()));
        }
    };
    a.for_each(|section_id, name, hash| {
        if !b.contains(section_id, name) || b.ext_hash(section_id, name) != hash.copied() {
            offer(section_id, name);
        }
    });
    b.for_each(|section_id, name, _| {
        if !a.contains(section_id, name) {
            offer(section_id, name);
        }
    });
    least
}

/// [`min_differing_key`] over two states' time-sample rows, keyed by
/// `(section_id, time bits)`, as [`min_differing_field`].
fn min_differing_sample(a: &CompactSamples, b: &CompactSamples) -> Option<(u64, u64)> {
    let mut least: Option<(u64, u64)> = None;
    let mut offer = |key: (u64, u64)| {
        if least.is_none_or(|least| key < least) {
            least = Some(key);
        }
    };
    a.for_each(|section_id, bits, hash| {
        if !b.contains(section_id, bits) || b.ext_hash(section_id, bits) != hash.copied() {
            offer((section_id, bits));
        }
    });
    b.for_each(|section_id, bits, _| {
        if !a.contains(section_id, bits) {
            offer((section_id, bits));
        }
    });
    least
}

/// **Test-support only.** [`first_difference`], for a test that measures
/// what the compare holds.
#[cfg(any(test, feature = "test-support"))]
pub fn first_difference_for_test(
    cached: &CompactState,
    cached_dangling: &HashSet<u64>,
    fresh: &CompactState,
    fresh_dangling: &HashSet<u64>,
) -> Option<String> {
    first_difference(cached, cached_dangling, fresh, fresh_dangling)
}

#[cfg(test)]
mod tests {
    //! `first_difference`'s key naming, walk order and in-place scan for each
    //! part's smallest differing key. The integration tests
    //! (`tests/integration/storage/structural_cache_audit_test.rs`) reach the
    //! section, field, children-list and dangling-parent names through real
    //! out-of-band rewrites; the object columns and time samples no
    //! `rewrite_out_of_band` change reaches are named here. That the compare
    //! holds no copy of either state is measured by the upload memory
    //! envelope (`first_difference_holds_no_copy_of_either_state_fsz`).

    use ordered_float::OrderedFloat;

    use super::*;
    use crate::domain::types::{compact, ObjectState};
    use crate::domain::types::{Field, FieldValue, Section, TimeSample};

    fn section(section_id: u64, name: &str) -> Section {
        Section {
            section_id,
            parent_id: 0,
            name: name.to_owned(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    /// An external value whose hash is `tag` repeated: the part of a value
    /// row the compact form (and so the audit) sees. Two tags are two hashes.
    fn external(tag: u8) -> FieldValue {
        FieldValue::External {
            hash: [tag; 32],
            size: 64,
            ext_value_index: 0,
        }
    }

    fn field(section_id: u64, key_name: &str, tag: u8) -> Field {
        Field {
            section_id,
            key_name: key_name.to_owned(),
            value: external(tag),
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        }
    }

    fn sample(section_id: u64, time: f64, tag: u8) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: external(tag),
            version: 1,
        }
    }

    /// Two sections, a field on each, a time sample on each, a children list.
    /// Every value row is external, so a row whose value changes (to another
    /// tag) changes what the compact form holds.
    fn state() -> ObjectState {
        let mut state = ObjectState {
            object_id: 7,
            sequence_num: 3,
            first_version: 1,
            source_format: "usda".to_owned(),
            next_set_order: 4,
            ..ObjectState::default()
        };
        for s in [section(1, "A"), section(2, "B")] {
            state.sections.insert(s.section_id, s);
        }
        state.children.insert(0, vec![1, 2]);
        for f in [field(1, "x", 1), field(2, "y", 2)] {
            state.fields.insert((f.section_id, f.key_name.clone()), f);
        }
        for t in [sample(1, 0.5, 1), sample(2, 2.5, 2)] {
            state.time_samples.insert((t.section_id, t.time), t);
        }
        state
    }

    fn difference(cached: &ObjectState, fresh: &ObjectState) -> Option<String> {
        let dangling = HashSet::new();
        first_difference(&compact(cached), &dangling, &compact(fresh), &dangling)
    }

    #[test]
    fn first_difference_of_equal_states_is_none() {
        assert_eq!(difference(&state(), &state()), None);
    }

    /// A change to one object column.
    type ColumnChange = fn(&mut ObjectState);

    #[test]
    fn first_difference_names_each_object_column() {
        let changes: [(&str, ColumnChange); 5] = [
            ("object_id", |s| s.object_id += 1),
            ("sequence_num", |s| s.sequence_num += 1),
            ("first_version", |s| s.first_version += 1),
            ("source_format", |s| s.source_format = "usdc".to_owned()),
            ("next_set_order", |s| s.next_set_order += 1),
        ];
        for (column, change) in changes {
            let mut fresh = state();
            change(&mut fresh);
            assert_eq!(
                difference(&state(), &fresh),
                Some(format!("object column {column}")),
                "{column}"
            );
        }
    }

    /// What the compact form cannot see is never a difference: a row's
    /// inline bytes, an external value's size and `ext_value_index`, and
    /// every row's set orders and version. The audit's entry holds none of
    /// them, so a store differing only there matches.
    #[test]
    fn first_difference_ignores_what_the_compact_form_does_not_hold() {
        let mut cached = state();
        cached.fields.insert((1, "i".to_owned()), {
            let mut f = field(1, "i", 0);
            f.value = FieldValue::Inline(vec![1, 2, 3]);
            f
        });
        let mut fresh = cached.clone();
        fresh
            .fields
            .get_mut(&(1, "i".to_owned()))
            .expect("row")
            .value = FieldValue::Inline(vec![9]);
        let x = fresh.fields.get_mut(&(1, "x".to_owned())).expect("row");
        x.set_order = 99;
        x.authoring_set_order = 98;
        x.version = 97;
        x.value = FieldValue::External {
            hash: [1; 32],
            size: 7,
            ext_value_index: 3,
        };
        fresh
            .time_samples
            .get_mut(&(2, OrderedFloat(2.5)))
            .expect("row")
            .version = 42;
        assert_ne!(cached, fresh, "fixture: the full states differ");
        assert_eq!(difference(&cached, &fresh), None);
    }

    /// A time sample differing in its external hash, and one present on one
    /// side only, are named by section and time.
    #[test]
    fn first_difference_names_a_time_sample_by_section_and_time() {
        let mut changed = state();
        changed
            .time_samples
            .insert((2, OrderedFloat(2.5)), sample(2, 2.5, 9));
        let mut added = state();
        added
            .time_samples
            .insert((1, OrderedFloat(4.0)), sample(1, 4.0, 3));

        assert_eq!(
            difference(&state(), &changed),
            Some("time sample at 2.5 of section 2".to_owned())
        );
        assert_eq!(
            difference(&state(), &added),
            Some("time sample at 4 of section 1".to_owned())
        );
    }

    /// The walk is columns, sections, children lists, fields, time samples,
    /// dangling parents, each in key order: with every part differing, each
    /// earlier part is named first, and within a part the smallest key.
    #[test]
    fn first_difference_walks_columns_then_sections_children_fields_samples_dangling() {
        let mut fresh = state();
        fresh.next_set_order += 1;
        fresh.sections.insert(2, section(2, "B2"));
        fresh.sections.insert(1, section(1, "A2"));
        fresh.children.insert(0, vec![2, 1]);
        fresh.fields.insert((1, "x".to_owned()), field(1, "x", 9));
        fresh
            .time_samples
            .insert((1, OrderedFloat(0.5)), sample(1, 0.5, 9));
        let fresh_dangling = HashSet::from([77]);
        let cached = state();
        let none = HashSet::new();
        let mut named = Vec::new();

        loop {
            let Some(name) =
                first_difference(&compact(&cached), &none, &compact(&fresh), &fresh_dangling)
            else {
                break;
            };
            named.push(name.clone());
            // Undo the part just named, so the next call names the next one.
            match name.as_str() {
                "object column next_set_order" => fresh.next_set_order = cached.next_set_order,
                "section 1" => {
                    fresh.sections.insert(1, cached.sections[&1].clone());
                }
                "section 2" => {
                    fresh.sections.insert(2, cached.sections[&2].clone());
                }
                "children list of 0" => fresh.children = cached.children.clone(),
                "field \"x\" of section 1" => fresh.fields = cached.fields.clone(),
                "time sample at 0.5 of section 1" => {
                    fresh.time_samples = cached.time_samples.clone();
                }
                "dangling parent 77" => {
                    assert_eq!(fresh, cached, "the dangling parents are walked last");
                    break;
                }
                other => panic!("unexpected name {other:?} after {named:?}"),
            }
        }

        assert_eq!(
            named,
            [
                "object column next_set_order",
                "section 1",
                "section 2",
                "children list of 0",
                "field \"x\" of section 1",
                "time sample at 0.5 of section 1",
                "dangling parent 77",
            ]
        );
    }

    /// How a test builds one part's differences between the entry (`cached`)
    /// and the store's state (`fresh`): three keys, smallest first — the
    /// first and last each held by one side alone, the middle by both with
    /// different values — and which side holds the first and the last
    /// (`true`: the store's state alone).
    type PartDifferences = fn(&mut ObjectState, &mut ObjectState, [u64; 3], [bool; 2]);

    /// `put` into the store's state when `fresh_only`, else into the entry.
    fn on_one_side(
        fresh_only: bool,
        cached: &mut ObjectState,
        fresh: &mut ObjectState,
        put: impl FnOnce(&mut ObjectState),
    ) {
        put(if fresh_only { fresh } else { cached });
    }

    /// A row with an inline value: the compact form holds no hash for it, so
    /// only its presence tells two states apart.
    fn inline_field(section_id: u64, key_name: &str) -> Field {
        Field {
            value: FieldValue::Inline(vec![1]),
            ..field(section_id, key_name, 0)
        }
    }

    fn inline_sample(section_id: u64, time: f64) -> TimeSample {
        TimeSample {
            value: FieldValue::Inline(vec![1]),
            ..sample(section_id, time, 0)
        }
    }

    /// Within each part the smallest differing key is named, whichever side
    /// holds it alone: for sections, children lists, fields and time samples,
    /// with a key held by one side alone below a key both hold with different
    /// values below a key held by the other side alone, the smallest is
    /// named, with the store's state or the entry holding it. The rows held
    /// by one side alone carry inline values, which the compact form holds no
    /// hash for. Among 1,000 sections differing at once, the smallest is
    /// named. Each part is scanned in hash order, so this fails if a scan
    /// stopped at the first difference it met, skipped the pass over the
    /// store's keys the entry lacks, or read a row the other side lacks as
    /// equal because both have no hash for it.
    #[test]
    fn first_difference_names_the_smallest_differing_key_from_either_side() {
        let sections: PartDifferences =
            |cached, fresh, [low, mid, high], [low_fresh, high_fresh]| {
                on_one_side(low_fresh, cached, fresh, |s| {
                    s.sections.insert(low, section(low, "one side"));
                });
                on_one_side(high_fresh, cached, fresh, |s| {
                    s.sections.insert(high, section(high, "one side"));
                });
                cached.sections.insert(mid, section(mid, "cached"));
                fresh.sections.insert(mid, section(mid, "fresh"));
            };
        let children: PartDifferences =
            |cached, fresh, [low, mid, high], [low_fresh, high_fresh]| {
                on_one_side(low_fresh, cached, fresh, |s| {
                    s.children.insert(low, vec![1]);
                });
                on_one_side(high_fresh, cached, fresh, |s| {
                    s.children.insert(high, vec![1]);
                });
                cached.children.insert(mid, vec![1, 2]);
                fresh.children.insert(mid, vec![2, 1]);
            };
        let fields: PartDifferences = |cached, fresh, [low, mid, high], [low_fresh, high_fresh]| {
            let name = |key: u64| format!("k{key}");
            on_one_side(low_fresh, cached, fresh, |s| {
                s.fields.insert((1, name(low)), inline_field(1, &name(low)));
            });
            on_one_side(high_fresh, cached, fresh, |s| {
                s.fields
                    .insert((1, name(high)), inline_field(1, &name(high)));
            });
            cached
                .fields
                .insert((1, name(mid)), field(1, &name(mid), 8));
            fresh.fields.insert((1, name(mid)), field(1, &name(mid), 9));
        };
        let samples: PartDifferences =
            |cached, fresh, [low, mid, high], [low_fresh, high_fresh]| {
                let time = |key: u64| 10.0 + key as f64;
                on_one_side(low_fresh, cached, fresh, |s| {
                    s.time_samples
                        .insert((1, OrderedFloat(time(low))), inline_sample(1, time(low)));
                });
                on_one_side(high_fresh, cached, fresh, |s| {
                    s.time_samples
                        .insert((1, OrderedFloat(time(high))), inline_sample(1, time(high)));
                });
                cached
                    .time_samples
                    .insert((1, OrderedFloat(time(mid))), sample(1, time(mid), 8));
                fresh
                    .time_samples
                    .insert((1, OrderedFloat(time(mid))), sample(1, time(mid), 9));
            };
        let cases: [(&str, [u64; 3], PartDifferences, &str); 4] = [
            ("sections", [10, 20, 30], sections, "section 10"),
            ("children", [10, 20, 30], children, "children list of 10"),
            ("fields", [1, 2, 3], fields, "field \"k1\" of section 1"),
            (
                "samples",
                [1, 2, 3],
                samples,
                "time sample at 11 of section 1",
            ),
        ];
        for (part, keys, build, named) in cases {
            for (holder, sides) in [
                ("the store's state", [true, false]),
                ("the entry", [false, true]),
            ] {
                let (mut cached, mut fresh) = (state(), state());
                build(&mut cached, &mut fresh, keys, sides);
                assert_eq!(
                    difference(&cached, &fresh),
                    Some(named.to_owned()),
                    "{part}: the smallest key, held by {holder} alone"
                );
            }
        }

        let (mut cached, mut fresh) = (state(), state());
        for id in 1_000..2_000 {
            cached.sections.insert(id, section(id, "cached"));
            fresh.sections.insert(id, section(id, "fresh"));
        }
        assert_eq!(
            difference(&cached, &fresh),
            Some("section 1000".to_owned()),
            "the smallest of 1,000 differing sections"
        );
    }

    /// Field names order by their bytes — "B" (0x42) before "b" (0x62) — and
    /// time samples by their time's bit pattern, as the walk's key order
    /// does: 1.0 before −1.0 (a set sign bit sorts above), and 0.0 before
    /// −0.0, two keys the float order would call equal.
    #[test]
    fn first_difference_orders_field_names_bytewise_and_samples_by_time_bits() {
        let mut fresh = state();
        for name in ["b", "B"] {
            fresh.fields.insert((1, name.to_owned()), field(1, name, 5));
        }
        assert_eq!(
            difference(&state(), &fresh),
            Some("field \"B\" of section 1".to_owned())
        );

        let mut fresh = state();
        for time in [-1.0, 1.0] {
            fresh
                .time_samples
                .insert((1, OrderedFloat(time)), sample(1, time, 5));
        }
        assert_eq!(
            difference(&state(), &fresh),
            Some("time sample at 1 of section 1".to_owned()),
            "1.0's bits sort below -1.0's"
        );

        // 0.0 and −0.0 are one key to the full state's map, so each side
        // holds one of them: the compact form keys them by their bits, two
        // keys, and names 0.0's, the smaller, whichever side holds it.
        for (cached_time, fresh_time) in [(-0.0, 0.0), (0.0, -0.0)] {
            let (mut cached, mut fresh) = (state(), state());
            cached
                .time_samples
                .insert((1, OrderedFloat(cached_time)), sample(1, cached_time, 5));
            fresh
                .time_samples
                .insert((1, OrderedFloat(fresh_time)), sample(1, fresh_time, 5));
            assert_eq!(
                difference(&cached, &fresh),
                Some("time sample at 0 of section 1".to_owned()),
                "the entry at {cached_time:?}, the store at {fresh_time:?}"
            );
        }
    }
}
