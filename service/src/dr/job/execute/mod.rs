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

//! The restore writer (SDD §22.22, *The restore writer*; SR-9.4 to SR-9.7,
//! SR-9.10, SR-12): [`execute_restore`] writes a restore plan. The restore
//! body runs it after the frame opened the write permit; the accept-loss body
//! runs it too, before its discards ([`discard`], SR-10).
//!
//! # Stages, in order
//!
//! 0. **Check** the plan ([`check_executable`]): a defect logs one ERROR and
//!    exits 1 before any write.
//! 1. **Abandoned deletes** ([`session`]), up to
//!    `LIVE_EDIT_DR_RESTORE_PARALLELISM` at once: a session whose deletion
//!    marker survives is not restored; its delete is finished — its objects
//!    registered dead, wiped, unmapped, its path members removed, its backup
//!    deleted, in that order.
//! 2. **Objects** ([`object`]): every restore and flag clear, under the
//!    Job's restore memory ([`memory`]), largest charge first. A restore
//!    re-fetches its planned lineage by its exact handles, decodes it, reads
//!    its stream's high-water, bands and flattens the state, wipes the
//!    object, backfills its tail-carried values to Valkey and the backup
//!    store, and writes its batches, the header last
//!    (`storage::layer::dr_restore`). A flag clear reads the high-water,
//!    removes the retirement carrier, and last stamps the banded `seq` and
//!    clears the flags in one transaction. Kept objects and held objects
//!    write nothing.
//! 3. **Moves** ([`session`]): each layer mapped to a fenced object moves to
//!    its complete successor, after the objects (`storage::dr_write`).
//! 4. **Sessions** ([`session`]), up to `LIVE_EDIT_DR_RESTORE_PARALLELISM` at
//!    once: each restored session's path index (Step A), then, when it needs
//!    one, its record, mappings and layers index (Step B).
//!
//! # Rules everywhere
//!
//! - Every write unit calls `run.permit.begin_unit()` first.
//! - The writer spawns no task: dropping its future (SIGTERM, a lost lease)
//!   stops every write; a decode runs on the blocking pool and writes
//!   nothing, and so does accept-loss's allocator purge
//!   (`storage::allocator_purge`, [`discard`] D5), which a dropped future
//!   leaves to finish, harmless.
//! - **No error path makes a compensating write.** A failure stops the
//!   writer where it is, so it is equivalent to a kill there, and a re-run
//!   converges (below). One exception, in accept-loss's discards: a lost
//!   layer found complete at its header's re-read before its wipe removes
//!   from the dead-object set what that run registered and did not wipe
//!   ([`discard`], D2).
//! - A Valkey write step is retried in place on a transient error,
//!   [`DR_WRITE_STEP_ATTEMPTS`] attempts with full-jitter backoff (100 ms to
//!   1 s); a backup-store call is never retried by the Job (the adapter
//!   retries inside the call). A failure after the retries logs one ERROR
//!   naming the object or session and the step, and exits 1.
//!
//! # A kill converges (SR-9.10)
//!
//! "Complete" is a header that is structurally complete and not retiring;
//! the planner keeps a complete object and restores a partial one, so a
//! re-run plans what is left:
//!
//! - a restore killed before its header still finds its stream, so it bands
//!   at the same floor; its wipe removes what earlier batches wrote; its
//!   backfills are idempotent (`SET NX`, a content-addressed PUT);
//! - a flag clear killed before its last transaction still has its flags
//!   and its stream, so it is planned the same way again;
//! - an abandoned delete registers its objects dead first and carries its
//!   mappings in the plan, so a delete killed after its unmap is still
//!   registered;
//! - a session killed after Step A has a path member without a record,
//!   which the re-run restores with Step B.
//!
//! # Layout
//!
//! - [`object`] — the restore and flag-clear pipelines;
//! - [`session`] — abandoned deletes (and accept-loss's lost sessions, the
//!   same discard), moves and session restores;
//! - [`discard`] — accept-loss's discards, after the writer;
//! - [`memory`] — restore memory: the budget, admission, the charges.

pub mod discard;
pub mod memory;
pub mod object;
pub mod session;

use std::collections::BTreeMap;
use std::future::Future;
use std::time::Duration;

use self::memory::{RestoreMemory, DR_VALUE_REMOVAL_PAGE_ROWS};
use crate::dr::job::plan::decide::check_executable;
use crate::dr::job::plan::model::{ObjectAction, RestorePlan, SessionAction};
use crate::dr::job::{JobExit, JobMode, JobRun};
use crate::error::ServiceError;
use crate::storage::layer::dr_restore::RESTORE_BATCH_COMMANDS;
use crate::storage::layer::{
    full_jitter_backoff, OVERWRITE_WAKE_BACKOFF_CAP, OVERWRITE_WAKE_BACKOFF_FLOOR,
};

/// How many attempts one Valkey write step gets before the Job exits 1: 3
/// (PD-5).
pub const DR_WRITE_STEP_ATTEMPTS: u32 = 3;

// A keyspace walk's step is resent after a transient failure up to the same
// bound as a write step (`storage::scan_guard`): the planner's and the
// recount's keyspace walks (a DR Job's walk resends at the same cursor, as a
// coverage walk does) and the periodic pass's guarded walks through
// `WALK_STEP_ATTEMPTS`; accept-loss's value removal sends each rank page and
// count read through this write step, with one attempt per write-step
// attempt.
const _: () = assert!(
    crate::storage::scan_guard::WALK_STEP_ATTEMPTS == DR_WRITE_STEP_ATTEMPTS,
    "a guarded walk's step attempts must equal the write step's"
);

/// What the writer did, for the run's summary (`super::finish`,
/// `plan::report::log_restore_summary`), and how the planner's lineage stage
/// held memory. Not persisted: the log is the record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionReport {
    /// Each restored or flag-cleared object's write.
    pub objects: BTreeMap<u64, ObjectWrite>,
    /// Each move, `(from, to)`.
    pub remapped: Vec<(u64, u64)>,
    /// Sessions restored with Step B.
    pub sessions_restored: usize,
    /// Sessions restored with Step A alone.
    pub sessions_path_index_only: usize,
    /// The sessions whose abandoned delete was finished.
    pub deletes_completed: Vec<String>,
    /// The tail backfills.
    pub backfill: BackfillReport,
    /// How the objects stage held memory.
    pub memory: MemoryReport,
    /// How the planner's lineage stage held memory, over every planning
    /// pass of the run (the most at once in any pass). Filled by the restore
    /// body from its plan; all zeros when the plan selected no lineage.
    pub planner_memory: MemoryReport,
}

/// One object's write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectWrite {
    /// What was written.
    pub kind: WriteKind,
    /// The band's floor: `max(S, plausible high-water)`.
    pub floor: u64,
    /// The stream's plausible high-water, when one was read.
    pub high_water: Option<u64>,
    /// The band.
    pub band: u64,
    /// The seeded `seq`: `floor + band`.
    pub seeded_seq: u64,
    /// Each transaction, in order; for a restore the header transaction is
    /// last (alone when the object is one transaction).
    pub batches: Vec<BatchRecord>,
}

/// What an object's write was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteKind {
    /// Restored from a backup.
    Restored {
        /// The lineage's head, `S`.
        head_seq: u64,
        /// Its base.
        base_seq: u64,
        /// How many tails it composed.
        tails: usize,
    },
    /// A kept survivor's retirement flags cleared.
    FlagsCleared,
}

/// One transaction the writer sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchRecord {
    /// The commands it queued (`MULTI` and `EXEC` not counted).
    pub commands: u64,
    /// Its client timeout: `txn_timeout(command timeout, commands)`.
    pub timeout: Duration,
}

/// The tail backfills (SR-9.6 step 6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackfillReport {
    /// Tail-carried values stored in Valkey (`SET NX`; already-present ones
    /// included).
    pub valkey_values: u64,
    /// Tail-carried values this run PUT to the backup store.
    pub backup_puts: u64,
}

/// How a stage under the restore memory (the writer's objects stage, or the
/// planner's lineage stage) held memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryReport {
    /// `bytes` or `serial`.
    pub mode: String,
    /// The most layers at once the stage allowed.
    pub parallelism: usize,
    /// The most layers it held at once.
    pub max_in_flight: usize,
    /// The most bytes it reserved at once.
    pub peak_reserved_bytes: u64,
}

impl MemoryReport {
    /// A stage under `memory` that admitted nothing yet.
    pub(crate) fn idle(memory: &RestoreMemory) -> Self {
        Self {
            mode: match memory {
                RestoreMemory::Bytes { .. } => "bytes".to_owned(),
                RestoreMemory::Serial { .. } => "serial".to_owned(),
            },
            parallelism: memory.parallelism(),
            ..Self::default()
        }
    }

    /// Fold in what `gate` held: the larger of each peak.
    pub(crate) fn record(&mut self, gate: &memory::RestoreGate) {
        self.max_in_flight = self.max_in_flight.max(gate.max_in_flight());
        self.peak_reserved_bytes = self.peak_reserved_bytes.max(gate.peak_reserved_bytes());
    }
}

/// Where a test may stop the writer, as a kill would (test-support only).
/// Each object point names the object it stops at (`None`: the first object
/// to reach it); each session point names the session.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KillPoint {
    /// After a restore read its stream's high-water.
    ObjectAfterHighWater(Option<u64>),
    /// After a restore's wipe.
    ObjectAfterWipe(Option<u64>),
    /// After a restore's Valkey backfill.
    ObjectAfterBackfillValkey(Option<u64>),
    /// After a restore's backup-store backfill.
    ObjectAfterBackfillBackup(Option<u64>),
    /// After a restore's row batch `batch` (0-based) landed.
    ObjectAfterBatch {
        /// The object.
        object: Option<u64>,
        /// The batch.
        batch: usize,
    },
    /// After every row batch and before the header transaction.
    ObjectBeforeHeader(Option<u64>),
    /// After a flag clear read its stream's high-water.
    ClearFlagsAfterHighWater(Option<u64>),
    /// After a flag clear removed the retirement carrier.
    ClearFlagsAfterCarrier(Option<u64>),
    /// After an abandoned delete registered its objects dead.
    DeleteAfterRegister(Option<String>),
    /// After an abandoned delete wiped its objects.
    DeleteAfterWipe(Option<String>),
    /// After an abandoned delete removed its mappings.
    DeleteAfterUnmap(Option<String>),
    /// After an abandoned delete removed its path members.
    DeletePathMembersDone(Option<String>),
    /// After a session restore's Step A.
    SessionAfterStepA(Option<String>),
    /// Before a move.
    RemapBefore(Option<u64>),
    /// Accept-loss: after a discard registered its objects dead. `None`
    /// stops at the lost layers' registration, which every run reaches
    /// first (with no lost layer too), so it never reaches a lost session's;
    /// `Some(session)` stops at that lost session's registration only.
    DiscardAfterRegister(Option<String>),
    /// Accept-loss: after a discard wiped its objects (`None` and `Some` as
    /// [`Self::DiscardAfterRegister`]'s).
    DiscardAfterWipe(Option<String>),
    /// Accept-loss: after the rewrite of every over-limit layer's session
    /// backup without that layer, and before the lost layers' unmap, so each
    /// such layer is still mapped while its session backup no longer names
    /// it. Every run reaches it once, after the lost layers' wipe (with no
    /// over-limit layer, or no backup store, too).
    DiscardAfterOverLimitRewrite,
    /// Accept-loss: after a discard removed its mappings (`None` and `Some`
    /// as [`Self::DiscardAfterRegister`]'s).
    DiscardAfterUnmap(Option<String>),
    /// Accept-loss: after a lost session's backup was deleted.
    LostSessionBackupDeleted(Option<String>),
    /// Accept-loss's value removal: after its row batch `batch` (0-based, in
    /// the order the walk filled them) landed. Every removed hash is still
    /// in `obj:ext` there: the `HDEL`s follow the last row batch. Never
    /// reached by a removal whose rows name none of its values.
    ValueRowsAfterBatch {
        /// The object.
        object: Option<u64>,
        /// The row batch.
        batch: usize,
    },
    /// Accept-loss's value removal: after every row batch and every
    /// `obj:ext` batch and before the header transaction, so the last hash
    /// is still in `obj:ext` and a re-run finishes the removal.
    ValueRemovalBeforeHeader(Option<u64>),
    /// Accept-loss: before a session backup's rewrite after the unmaps (not
    /// the over-limit layers' rewrite ahead of them, whose point is
    /// [`Self::DiscardAfterOverLimitRewrite`]).
    SessionRewriteBefore(Option<String>),
    /// The recount (`super::finish`), after its first group of count sets
    /// landed: before the second group's write unit. Never reached by a
    /// recount of at most one group (`RECOUNT_WRITE_CHUNK`, 64, hashes).
    RecountAfterFirstChunk,
    /// After the recount and the summary, immediately before the un-hold
    /// script: the counts are exact and the marker stays.
    BeforeUnhold,
    /// After the un-hold script ran, before its outcome is logged.
    AfterUnhold,
}

/// A test's hook into accept-loss's discards, run once synchronously after
/// D1's registration landed and before D2's first header re-read
/// ([`ExecSeams::after_discard_register`]), with the objects D1 registered
/// (sorted; empty when there were none). A test uses it as a writer that
/// does not honour the hold would act inside the D0 → D2 window: completing
/// a lost layer's header there makes D2 stop with AL9 after D1 registered
/// it.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone)]
pub struct AfterDiscardRegister(pub AfterDiscardRegisterFn);

/// [`AfterDiscardRegister`]'s callback: called with the objects D1
/// registered.
#[cfg(any(test, feature = "test-support"))]
pub type AfterDiscardRegisterFn = std::sync::Arc<dyn Fn(&[u64]) + Send + Sync>;

#[cfg(any(test, feature = "test-support"))]
impl std::fmt::Debug for AfterDiscardRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AfterDiscardRegister(..)")
    }
}

/// A point the writer reaches, checked against the test's kill point.
// The payloads are read only by the test-gated `kill_matches`; a build
// without `test-support` carries the variants but never reads them.
#[cfg_attr(not(any(test, feature = "test-support")), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum At<'a> {
    ObjectAfterHighWater(u64),
    ObjectAfterWipe(u64),
    ObjectAfterBackfillValkey(u64),
    ObjectAfterBackfillBackup(u64),
    ObjectAfterBatch(u64, usize),
    ObjectBeforeHeader(u64),
    ClearFlagsAfterHighWater(u64),
    ClearFlagsAfterCarrier(u64),
    DeleteAfterRegister(&'a str),
    DeleteAfterWipe(&'a str),
    DeleteAfterUnmap(&'a str),
    DeletePathMembersDone(&'a str),
    SessionAfterStepA(&'a str),
    RemapBefore(u64),
    DiscardAfterRegister(Option<&'a str>),
    DiscardAfterWipe(Option<&'a str>),
    DiscardAfterOverLimitRewrite,
    DiscardAfterUnmap(Option<&'a str>),
    LostSessionBackupDeleted(&'a str),
    ValueRowsAfterBatch(u64, usize),
    ValueRemovalBeforeHeader(u64),
    SessionRewriteBefore(&'a str),
    RecountAfterFirstChunk,
    BeforeUnhold,
    AfterUnhold,
}

/// The writer's test seams. [`ExecSeams::production`] in the binary.
#[derive(Clone, Debug)]
pub struct ExecSeams {
    /// The most commands one batch queues: [`RESTORE_BATCH_COMMANDS`] in
    /// production.
    pub batch_commands: u64,
    /// Accept-loss's value removal: the `COUNT` of each index step and the
    /// most rows it reads at once ([`DR_VALUE_REMOVAL_PAGE_ROWS`] in
    /// production; its admission is charged at the production page).
    pub value_removal_page_rows: u64,
    /// Where to stop, as a kill would.
    #[cfg(any(test, feature = "test-support"))]
    pub kill_point: Option<KillPoint>,
    /// The byte budget in place of `R`, in bytes mode, for the objects stage
    /// and, through the restore body, the planner's lineage stage.
    #[cfg(any(test, feature = "test-support"))]
    pub budget_override: Option<u64>,
    /// The lease value the un-hold script is handed instead of the Job's
    /// own (`finish::unhold`), so the script answers `lease_lost`
    /// deterministically.
    #[cfg(any(test, feature = "test-support"))]
    pub unhold_lease_value: Option<String>,
    /// The un-hold's first attempt runs its script and then reports its
    /// reply lost, so the write step retries it (`finish::unhold`): the real
    /// in-doubt path, whose retry answers `already_unheld`.
    #[cfg(any(test, feature = "test-support"))]
    pub unhold_reply_lost: bool,
    /// Accept-loss: run after D1's registration and before D2
    /// ([`AfterDiscardRegister`]).
    #[cfg(any(test, feature = "test-support"))]
    pub after_discard_register: Option<AfterDiscardRegister>,
}

impl Default for ExecSeams {
    fn default() -> Self {
        Self::production()
    }
}

impl ExecSeams {
    /// The production seams: [`RESTORE_BATCH_COMMANDS`],
    /// [`DR_VALUE_REMOVAL_PAGE_ROWS`], no kill point, no budget override.
    pub fn production() -> Self {
        Self {
            batch_commands: RESTORE_BATCH_COMMANDS,
            value_removal_page_rows: DR_VALUE_REMOVAL_PAGE_ROWS,
            #[cfg(any(test, feature = "test-support"))]
            kill_point: None,
            #[cfg(any(test, feature = "test-support"))]
            budget_override: None,
            #[cfg(any(test, feature = "test-support"))]
            unhold_lease_value: None,
            #[cfg(any(test, feature = "test-support"))]
            unhold_reply_lost: false,
            #[cfg(any(test, feature = "test-support"))]
            after_discard_register: None,
        }
    }

    /// Run the test's [`Self::after_discard_register`] hook, if any, with
    /// the objects D1 registered. Never in a release build.
    pub(crate) fn after_discard_register(&self, registered: &[u64]) {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(hook) = &self.after_discard_register {
            (hook.0)(registered);
        }
        let _ = registered;
    }

    /// The lease value the un-hold hands its script: `own`, the Job's, but
    /// for the test seam's ([`Self::unhold_lease_value`]; never in a release
    /// build).
    pub(crate) fn unhold_lease_value<'a>(&'a self, own: &'a str) -> &'a str {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(value) = &self.unhold_lease_value {
            return value;
        }
        own
    }

    /// Whether the un-hold's first attempt reports its reply lost (the test
    /// seam [`Self::unhold_reply_lost`]; never in a release build).
    pub(crate) fn unhold_reply_lost(&self) -> bool {
        #[cfg(any(test, feature = "test-support"))]
        {
            self.unhold_reply_lost
        }
        #[cfg(not(any(test, feature = "test-support")))]
        {
            false
        }
    }

    /// The budget override of the objects stage and the planner's lineage
    /// stage (always `None` in a release build).
    pub(crate) fn budget_override(&self) -> Option<u64> {
        #[cfg(any(test, feature = "test-support"))]
        {
            self.budget_override
        }
        #[cfg(not(any(test, feature = "test-support")))]
        {
            None
        }
    }

    /// Whether the `mode` Job stops at `at`: an INFO line and
    /// `Err(Failed)`, writing nothing more. Never in a release build.
    pub(crate) fn stop_at(&self, mode: JobMode, at: At<'_>) -> Result<(), JobExit> {
        if self.would_stop_at(at) {
            tracing::info!(
                point = ?at,
                "the {mode} Job stopped at its test kill point {at:?}, as a kill there would; \
                 it writes nothing more"
            );
            return Err(JobExit::Failed);
        }
        Ok(())
    }

    /// Whether the test's kill point is `at`. Never in a release build.
    pub(crate) fn would_stop_at(&self, at: At<'_>) -> bool {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(point) = &self.kill_point {
            return kill_matches(point, at);
        }
        let _ = at;
        false
    }
}

#[cfg(any(test, feature = "test-support"))]
fn kill_matches(point: &KillPoint, at: At<'_>) -> bool {
    let object = |filter: &Option<u64>, oid: u64| filter.is_none_or(|f| f == oid);
    let session = |filter: &Option<String>, sid: &str| filter.as_deref().is_none_or(|f| f == sid);
    let discard = |filter: &Option<String>, sid: Option<&str>| match filter {
        None => true,
        Some(f) => sid == Some(f.as_str()),
    };
    match (point, at) {
        (KillPoint::ObjectAfterHighWater(f), At::ObjectAfterHighWater(o))
        | (KillPoint::ObjectAfterWipe(f), At::ObjectAfterWipe(o))
        | (KillPoint::ObjectAfterBackfillValkey(f), At::ObjectAfterBackfillValkey(o))
        | (KillPoint::ObjectAfterBackfillBackup(f), At::ObjectAfterBackfillBackup(o))
        | (KillPoint::ObjectBeforeHeader(f), At::ObjectBeforeHeader(o))
        | (KillPoint::ClearFlagsAfterHighWater(f), At::ClearFlagsAfterHighWater(o))
        | (KillPoint::ClearFlagsAfterCarrier(f), At::ClearFlagsAfterCarrier(o))
        | (KillPoint::RemapBefore(f), At::RemapBefore(o)) => object(f, o),
        (KillPoint::ObjectAfterBatch { object: f, batch }, At::ObjectAfterBatch(o, b)) => {
            object(f, o) && *batch == b
        }
        (KillPoint::DeleteAfterRegister(f), At::DeleteAfterRegister(s))
        | (KillPoint::DeleteAfterWipe(f), At::DeleteAfterWipe(s))
        | (KillPoint::DeleteAfterUnmap(f), At::DeleteAfterUnmap(s))
        | (KillPoint::DeletePathMembersDone(f), At::DeletePathMembersDone(s))
        | (KillPoint::SessionAfterStepA(f), At::SessionAfterStepA(s))
        | (KillPoint::LostSessionBackupDeleted(f), At::LostSessionBackupDeleted(s))
        | (KillPoint::SessionRewriteBefore(f), At::SessionRewriteBefore(s)) => session(f, s),
        (KillPoint::DiscardAfterRegister(f), At::DiscardAfterRegister(s))
        | (KillPoint::DiscardAfterWipe(f), At::DiscardAfterWipe(s))
        | (KillPoint::DiscardAfterUnmap(f), At::DiscardAfterUnmap(s)) => discard(f, s),
        (KillPoint::ValueRowsAfterBatch { object: f, batch }, At::ValueRowsAfterBatch(o, b)) => {
            object(f, o) && *batch == b
        }
        (KillPoint::ValueRemovalBeforeHeader(f), At::ValueRemovalBeforeHeader(o)) => object(f, o),
        (KillPoint::DiscardAfterOverLimitRewrite, At::DiscardAfterOverLimitRewrite)
        | (KillPoint::RecountAfterFirstChunk, At::RecountAfterFirstChunk)
        | (KillPoint::BeforeUnhold, At::BeforeUnhold)
        | (KillPoint::AfterUnhold, At::AfterUnhold) => true,
        _ => false,
    }
}

/// How one attempt of a write step ended, short of success.
#[derive(Debug)]
pub(crate) enum StepError {
    /// Worth another attempt: a transient store error, a lost `WATCH`, a
    /// torn or in-doubt `EXEC`.
    Retry(String),
    /// Not worth another: log it and exit 1.
    Fatal(String),
    /// Exit with this, already logged.
    Exit(JobExit),
}

impl From<ServiceError> for StepError {
    fn from(e: ServiceError) -> Self {
        match e {
            ServiceError::Aborted(_)
            | ServiceError::Unavailable(_)
            | ServiceError::UnavailableWithDetail { .. } => Self::Retry(e.to_string()),
            other => Self::Fatal(other.to_string()),
        }
    }
}

/// Run one write step, retrying it in place ([`DR_WRITE_STEP_ATTEMPTS`]
/// attempts, full-jitter backoff), each attempt behind the write permit.
/// `step` names the step and its subject for the ERROR a failure logs.
pub(crate) async fn write_step<T, F, Fut>(
    run: &JobRun,
    step: &(dyn Fn() -> String + Sync),
    mut attempt: F,
) -> Result<T, JobExit>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, StepError>>,
{
    let mut last = String::new();
    for n in 0..DR_WRITE_STEP_ATTEMPTS {
        if n > 0 {
            tokio::time::sleep(full_jitter_backoff(
                n - 1,
                OVERWRITE_WAKE_BACKOFF_FLOOR,
                OVERWRITE_WAKE_BACKOFF_CAP,
            ))
            .await;
        }
        if run.permit.begin_unit().await.is_err() {
            // The frame logs the revocation and drops this future.
            return Err(JobExit::Failed);
        }
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(StepError::Retry(why)) => last = why,
            Err(StepError::Fatal(why)) => {
                step_failed(run.mode, step, &why, n + 1);
                return Err(JobExit::Failed);
            }
            Err(StepError::Exit(exit)) => return Err(exit),
        }
    }
    step_failed(run.mode, step, &last, DR_WRITE_STEP_ATTEMPTS);
    Err(JobExit::Failed)
}

/// The ERROR of the `mode` Job's write step that failed.
pub(crate) fn step_failed(
    mode: JobMode,
    step: &(dyn Fn() -> String + Sync),
    why: &str,
    attempts: u32,
) {
    let step = step();
    tracing::error!(
        step = %step,
        attempts,
        error = %why,
        "the {mode} Job could not {step} after {attempts} attempts: {why}; it stops writing and \
         exits 1 so Kubernetes retries it, and the service stays held"
    );
}

/// Write `plan` (module docs, *Stages*). `Ok` with what was written; `Err`
/// ends the run, already logged.
pub async fn execute_restore(
    run: &JobRun,
    plan: &RestorePlan,
    seams: &ExecSeams,
) -> Result<ExecutionReport, JobExit> {
    // 0. The consistency check, before any write.
    let job = run.mode;
    let defects = check_executable(plan);
    if let Some(defect) = defects.first() {
        tracing::error!(
            defects = defects.len(),
            defect = %defect,
            "the restore plan is inconsistent: {defect}; a defect in the planner, not the store; \
             the Job wrote nothing and exits 1"
        );
        return Err(JobExit::Failed);
    }
    let memory = RestoreMemory::derive(&run.config, run.mode).map_err(|refusal| {
        tracing::error!(
            refusal = %refusal,
            "the {job} Job could not derive its restore memory, wrote nothing and exits 1"
        );
        JobExit::Failed
    })?;

    let mut counts = (0usize, 0usize, 0usize, 0usize, 0usize);
    for object in &plan.objects {
        match object.action {
            ObjectAction::Restore { .. } => counts.0 += 1,
            ObjectAction::ClearFlags { .. } => counts.1 += 1,
            ObjectAction::Remap { .. } => counts.2 += 1,
            ObjectAction::Keep | ObjectAction::Nothing => {}
        }
    }
    for session in &plan.sessions {
        match session.action {
            SessionAction::Restore { .. } => counts.3 += 1,
            SessionAction::CompleteAbandonedDelete { .. } => counts.4 += 1,
            SessionAction::Nothing => {}
        }
    }
    let (restores, clears, moves, sessions, deletes) = counts;
    let mode = memory.describe();
    tracing::info!(
        restore = restores,
        clear = clears,
        remap = moves,
        sessions,
        deletes,
        mode = %mode,
        "the {job} Job writes the plan: {restores} layers to restore, {clears} flag clears, \
         {moves} moves, {sessions} sessions, {deletes} unfinished deletes; {mode}"
    );

    // 1. Abandoned deletes.
    let mut report = ExecutionReport {
        deletes_completed: session::complete_abandoned_deletes(run, plan, &memory, seams).await?,
        ..ExecutionReport::default()
    };
    // 2. Objects.
    object::write_objects(run, plan, &memory, seams, &mut report).await?;
    // 3. Moves.
    report.remapped = session::remap_layers(run, plan, &memory, seams).await?;
    // 4. Sessions.
    let (restored, path_index_only) = session::restore_sessions(run, plan, &memory, seams).await?;
    report.sessions_restored = restored;
    report.sessions_path_index_only = path_index_only;
    Ok(report)
}

/// What the writer's in-crate tests share: a run over a real store with its
/// write permit open, and a capture of what a test body logs.
#[cfg(test)]
pub(crate) mod test_harness {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::checkpoint::store::CheckpointStore;
    use crate::dr::job::absence::{OwnPuts, StoreClock};
    use crate::dr::job::permit::WritePermit;
    use crate::dr::job::quiet::{QuietPeriod, StoreDeadlines};
    use crate::dr::job::{BackupAccess, JobMode, JobRun};
    use crate::storage::test_support::TestValkey;

    /// A restore run on `tv` reading backups from `store`, its quiet period
    /// and `T_safe` past, and its write permit open, as the frame leaves it
    /// before the body writes. Its allocator purge is a function that
    /// accepts every call and purges nothing (the test binary's allocator
    /// is the system's), so accept-loss's value removal takes its purged
    /// path, the binary's on linux-gnu.
    pub(crate) async fn open_run(tv: &TestValkey, store: Arc<dyn CheckpointStore>) -> JobRun {
        let config = crate::checkpoint::apply::tests::test_config(1_000);
        let now = crate::storage::dr_job::store_time_ms(tv.client(), tv.key_schema())
            .await
            .expect("TIME");
        let permit = WritePermit::new();
        permit.open();
        JobRun {
            mode: JobMode::Restore,
            incident: "20261020t1200-unit".to_owned(),
            quiet: QuietPeriod::from_config(&config),
            not_found_cache: Duration::ZERO,
            config: Arc::new(config),
            valkey: tv.client().clone(),
            keys: tv.key_schema().clone(),
            backup: BackupAccess::Store(store),
            clock: StoreClock::new(
                tv.client().clone(),
                tv.key_schema().clone(),
                StoreDeadlines {
                    detected_at_ms: now - 3_600_000,
                    quiet_ends_at_ms: now - 1,
                    t_safe_ms: now - 1,
                },
                Duration::from_millis(250),
            ),
            permit,
            own_puts: OwnPuts::default(),
            metrics: Arc::new(crate::observability::metrics::metrics_scrape_only(
                "dr-execute-unit",
            )),
            lease_holder: String::new(),
            counters_sidecar_max: std::sync::OnceLock::new(),
            allocator_purge: Some(crate::storage::allocator_purge::AllocatorPurge::new(|| {
                true
            })),
        }
    }

    /// Everything logged at INFO and above while installed, as JSON records.
    /// Installed with the thread-local `set_default`: the test body's awaits
    /// run on the test's own thread (a `#[tokio::test]` body is the runtime's
    /// `block_on` future), and the writer moves no logging work elsewhere.
    #[derive(Clone, Default)]
    pub(crate) struct LogCapture(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
        type Writer = LogCapture;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl LogCapture {
        /// Install the capture on this thread until the guard drops.
        pub(crate) fn install(&self) -> tracing::subscriber::DefaultGuard {
            tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .json()
                    .with_writer(self.clone())
                    .with_max_level(tracing::Level::INFO)
                    .finish(),
            )
        }

        /// The messages logged at `level`, in order.
        pub(crate) fn messages(&self, level: &str) -> Vec<String> {
            let text = String::from_utf8(self.0.lock().unwrap().clone()).expect("UTF-8 log output");
            text.lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a JSON record"))
                .filter(|record| record["level"] == level)
                .filter_map(|record| record["fields"]["message"].as_str().map(str::to_owned))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    //! What the writer's stages need no backup lineage for, against a real
    //! store with the write permit open: a write step's in-place retry
    //! (PD-5), the consistency check before any write, and the identity check
    //! (PD-4) on a plan built by hand, which no well-formed planner run
    //! produces. Still not covered: `read_stream_floor`'s unreadable-stream
    //! arm, which needs a store that fails one command on demand.

    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;

    use super::test_harness::{open_run, LogCapture};
    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::store::{CheckpointMeta, CheckpointStore};
    use crate::domain::types::{ObjectState, Section};
    use crate::dr::job::plan::model::{
        BandInputs, CoveragePlan, LineageFootprint, ObjectPlan, PlanEpoch, PlannedLineage,
        RestoreReason,
    };
    use crate::storage::test_support::TestValkey;

    const STEP: &str = "write the test step";

    fn step() -> String {
        STEP.to_owned()
    }

    /// The ERROR a write step logs after `attempts` attempts ending `why`.
    fn step_error(attempts: u32, why: &str) -> String {
        format!(
            "the restore Job could not {STEP} after {attempts} attempts: {why}; it stops writing \
             and exits 1 so Kubernetes retries it, and the service stays held"
        )
    }

    async fn run_on(tv: &TestValkey) -> JobRun {
        open_run(tv, Arc::new(InMemoryCheckpointStore::new())).await
    }

    /// PD-5: a step that fails transiently once succeeds on its second
    /// attempt, with no ERROR.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_step_retries_a_transient_failure_in_place() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;
        let capture = LogCapture::default();
        let guard = capture.install();
        let attempts = AtomicU32::new(0);

        let result = write_step(&run, &step, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                if n < 2 {
                    Err(StepError::Retry(format!("blip {n}")))
                } else {
                    Ok(n)
                }
            }
        })
        .await;
        drop(guard);

        assert_eq!(result, Ok(2));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(
            capture.messages("ERROR").is_empty(),
            "{:?}",
            capture.messages("ERROR")
        );
    }

    /// PD-5: a step that fails transiently on every attempt stops after
    /// `DR_WRITE_STEP_ATTEMPTS` (3) with one ERROR naming the step, the
    /// attempts and the last failure, and exits 1; a failure not worth
    /// another attempt stops after the first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_step_gives_up_after_three_transient_failures_and_at_once_on_a_fatal_one() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;

        let capture = LogCapture::default();
        let guard = capture.install();
        let attempts = AtomicU32::new(0);
        let result: Result<(), JobExit> = write_step(&run, &step, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
            async move { Err(StepError::Retry(format!("blip {n}"))) }
        })
        .await;
        drop(guard);
        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(attempts.load(Ordering::SeqCst), DR_WRITE_STEP_ATTEMPTS);
        assert_eq!(
            capture.messages("ERROR"),
            vec![step_error(3, "blip 3")],
            "one ERROR, after the third attempt"
        );

        let capture = LogCapture::default();
        let guard = capture.install();
        let attempts = AtomicU32::new(0);
        let result: Result<(), JobExit> = write_step(&run, &step, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err(StepError::Fatal("refused".to_owned())) }
        })
        .await;
        drop(guard);
        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "no second attempt");
        assert_eq!(capture.messages("ERROR"), vec![step_error(1, "refused")]);
    }

    /// Which store errors a step retries: `Aborted` and `Unavailable` (a lost
    /// `WATCH`, a connection error), and nothing else.
    #[test]
    fn a_store_error_is_retried_only_when_transient() {
        for (error, retried) in [
            (ServiceError::Aborted("lost the WATCH".to_owned()), true),
            (
                ServiceError::Unavailable("connection reset".to_owned()),
                true,
            ),
            (ServiceError::Internal("a defect".to_owned()), false),
            (
                ServiceError::FailedPrecondition("a corrupt mapping".to_owned()),
                false,
            ),
        ] {
            let label = format!("{error:?}");
            assert_eq!(
                matches!(StepError::from(error), StepError::Retry(_)),
                retried,
                "{label}"
            );
        }
    }

    const OID: u64 = 7_301;
    const HEAD: u64 = 5;

    /// The fixture layer's state: a root prim with two children.
    fn layer_state() -> ObjectState {
        let mut state = ObjectState {
            object_id: OID,
            source_format: "usda".to_owned(),
            ..ObjectState::default()
        };
        for (id, parent) in [(1, 0), (2, 1), (3, 1)] {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("s{id}"),
                    section_type: 6,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
        }
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2, 3]);
        state
    }

    /// A plan restoring object [`OID`] as layer `l.usda` of session `s` from
    /// its base at [`HEAD`], whose meta names `backup_layer` of `s`; the base
    /// is put to `store`.
    async fn plan_restoring(store: &InMemoryCheckpointStore, backup_layer: &str) -> RestorePlan {
        let (base, _) = crate::checkpoint::assembler::build_reset_ompf_from_state(&layer_state())
            .expect("the base assembles");
        let fetched = base.len() as u64;
        let meta = CheckpointMeta {
            snap_seq: Some(HEAD),
            session_id: "s".to_owned(),
            layer_name: backup_layer.to_owned(),
            created_by: "user".to_owned(),
            server_ts: 1,
            ..CheckpointMeta::default()
        };
        let handle = store
            .put_layer_checkpoint(OID, HEAD, Bytes::from(base), meta.clone())
            .await
            .expect("the base put");
        plan_of(vec![ObjectPlan {
            object_id: OID,
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            action: ObjectAction::Restore {
                reason: RestoreReason::HeaderAbsent,
                lineage: Box::new(PlannedLineage {
                    base: handle,
                    tails: vec![],
                    head_seq: HEAD,
                    meta,
                    tail_carried: Box::new([]),
                    step_downs: vec![],
                    footprint: LineageFootprint {
                        fetched_bytes: fetched,
                        state_bytes: 1,
                        commands: 1,
                        largest_inline_bytes: 0,
                    },
                }),
                band: BandInputs {
                    floor_seq: HEAD,
                    observed_commit_span: 0,
                    written_at_ms: 0,
                    cursor: None,
                },
            },
        }])
    }

    fn plan_of(objects: Vec<ObjectPlan>) -> RestorePlan {
        RestorePlan {
            epoch: PlanEpoch {
                listing_taken_at_ms: Some(0),
                at_or_after_t_safe: true,
                relisted_at_t_safe: false,
            },
            sessions: vec![],
            objects,
            coverage: CoveragePlan::default(),
            held: vec![],
            advisories: vec![],
            listed_max_object_id: None,
            surviving_max_object_id: None,
            sessions_seen: Default::default(),
        }
    }

    /// How many of [`OID`]'s keys a restore of [`layer_state`] writes exist:
    /// its header, its stream, each section and the sections index, each
    /// children list and the children index, and the field, time-sample and
    /// external-value indices. One multi-key `EXISTS`: they share the
    /// object's `{oid}` slot.
    async fn object_keys_present(tv: &TestValkey) -> i64 {
        use fred::interfaces::KeysInterface;
        let keys = tv.key_schema();
        let oid = OID.to_string();
        let mut names = vec![
            keys.obj_header(&oid),
            keys.layer_stream(OID),
            keys.obj_sections_index(&oid),
            keys.obj_children_index(&oid),
            keys.obj_fields_index(&oid),
            keys.obj_time_samples_index(&oid),
            keys.obj_ext_index(&oid),
        ];
        for id in ["0", "1", "2", "3"] {
            names.push(keys.obj_section(&oid, id));
            names.push(keys.obj_children(&oid, id));
        }
        tv.client().pool.exists(names).await.expect("EXISTS")
    }

    /// PD-4: a planned backup whose meta names another layer than the one
    /// the plan maps its object to is never written under the plan's
    /// identity: one ERROR naming both, exit 1, and no key of the object —
    /// the identity check comes before the stream read, the wipe, the
    /// backfill and every batch. The same plan with the backup naming the
    /// mapped layer is written, so the refusal is the identity check's and
    /// not the harness's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn execute_restore_refuses_a_backup_naming_another_layer_and_writes_nothing() {
        let tv = TestValkey::setup().await;
        let store = Arc::new(InMemoryCheckpointStore::new());
        let run = open_run(&tv, Arc::clone(&store) as Arc<dyn CheckpointStore>).await;
        let mismatched = plan_restoring(&store, "other.usda").await;
        let capture = LogCapture::default();
        let guard = capture.install();

        let result = execute_restore(&run, &mismatched, &ExecSeams::production()).await;
        drop(guard);

        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(
            capture.messages("ERROR"),
            vec![format!(
                "the restore Job could not restore layer l.usda of session s (object {OID}): its \
                 backup names layer other.usda of session s; its plan selected generation {HEAD}, \
                 so the Job does not fall back to another backup; it stops writing and exits 1 \
                 so Kubernetes retries it, and the service stays held"
            )]
        );
        assert_eq!(object_keys_present(&tv).await, 0, "nothing written");

        let matching = plan_restoring(&store, "l.usda").await;
        let report = execute_restore(&run, &matching, &ExecSeams::production())
            .await
            .expect("a backup naming the mapped layer is written");
        assert!(report.objects.contains_key(&OID));
        assert!(
            object_keys_present(&tv).await >= 5,
            "fixture: the keys counted are the ones a restore writes (header, sections index, \
             three sections, a children list)"
        );
    }

    /// The consistency check runs first: a plan moving a layer to a
    /// successor it does not plan logs one ERROR naming the defect and exits
    /// 1 before the start line or any write.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn execute_restore_refuses_an_inconsistent_plan_before_any_write() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;
        let plan = plan_of(vec![ObjectPlan {
            object_id: 3,
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            action: ObjectAction::Remap { from: 3, to: 99 },
        }]);
        let capture = LogCapture::default();
        let guard = capture.install();

        let result = execute_restore(&run, &plan, &ExecSeams::production()).await;
        drop(guard);

        assert_eq!(result, Err(JobExit::Failed));
        assert_eq!(
            capture.messages("ERROR"),
            vec![
                "the restore plan is inconsistent: object 3 is moved to object 99, which the plan \
                 does not plan; a defect in the planner, not the store; the Job wrote nothing and \
                 exits 1"
                    .to_owned()
            ]
        );
        assert!(
            capture
                .messages("INFO")
                .iter()
                .all(|m| !m.starts_with("the restore Job writes the plan")),
            "the writer never started: {:?}",
            capture.messages("INFO")
        );
    }
}
