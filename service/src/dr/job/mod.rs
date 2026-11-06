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

//! The restore and accept-loss Jobs: `live-edit dr restore <incident>` and
//! `live-edit dr accept-loss <incident> --confirm discard-lost-data
//! [--confirm discard-unconfirmed-data]` (SDD SR-8.1, SR-9, SR-10). Design
//! and rationale: the Jobs section of SDD §22.
//!
//! # The Job flow (Tasks 6–10)
//!
//! A configuration `main.rs` refuses before the Job connects (the logging
//! and trace export settings, `Config::from_env`, which also refuses a pod
//! memory limit too small for the server's memory plan, and the key schema)
//! exits [`EXIT_USAGE`] in `dr` mode, where the service exits 1. `main.rs`
//! then parses the command line ([`args`]) before connecting: a refused one
//! exits [`EXIT_USAGE`] having touched nothing. So does a Job whose pod
//! memory limit is below the minimum its restore memory needs
//! ([`execute::memory::RestoreMemory::derive`]), or whose OTLP metrics
//! settings are refused. An OTLP exporter that could not be built for a
//! valid endpoint exits 1 instead, since a retry may build it. It then
//! connects Valkey, builds the backup store as the server does, and hands
//! both to [`run_job`] with the Job's body. The frame ([`run`]) logs the
//! restore memory's serial-mode WARN once when the Job could read no memory
//! limit, then runs, in this order:
//!
//! 1. **Precondition** ([`preconditions`]): read the hold marker, writing
//!    nothing. Not held → exit 0 ("nothing to do"); held under another
//!    incident → exit 4 ("held under a different incident <id>"); a marker
//!    without a usable `incident` or `detected_at_ms` → exit 3, never a
//!    guessed value. A restore with the `InMemory` adapter exits 3 here: its
//!    store cannot be read by another process. The lease is never taken on
//!    these paths.
//! 2. **Lease** ([`lease`]): take `le:dr:job:{counter}`, waiting up to one
//!    TTL (30 s) for a busy one ("another DR Job holds the lease" → exit 1).
//!    A background task refreshes it every 10 s for the Job's whole life; a
//!    lost lease revokes the write permit, and the Job exits 1. The lease is
//!    the one key a Job writes before the quiet period ends.
//! 3. **Precondition again**, under the lease: the same exits.
//! 4. **Quiet period** ([`quiet`]): no write until the store's `TIME` reaches
//!    `detected_at_ms` + Q. The body's planning ([`JobBody::plan`]) starts in
//!    the background during the wait; its plan is kept. The restore planner
//!    reads nothing until Q's end ([`plan`]); accept-loss's, with a backup
//!    store, nothing until `T_safe` (PD-3).
//! 5. **Write permit** ([`permit`]): opens on the first reading of the
//!    store's clock at or past Q's end. A lost lease or SIGTERM revokes it
//!    (exit 1), before or after it opened. A revoked permit aborts the work in
//!    flight; the Job releases its lease and exits, and a re-run converges.
//! 6. **Run-wide absence preconditions** ([`absence`]), with a persistent
//!    backup store: the counters sidecar must read present (PD2), and the
//!    list-route canary must pass, each judged at or after `T_safe` before
//!    the Job stays held (exit 3). The `Disabled` and `InMemory` adapters
//!    skip both: the Job branches on the configured adapter, never on errors
//!    or empty answers.
//! 7. **The body** ([`JobBody::execute`]): the restore body ([`restore`])
//!    writes the plan ([`execute`]: the abandoned deletes, the objects, the
//!    moves, the sessions), then finalises the plan's coverage check
//!    ([`plan`]), recounts the large values' references ([`finish`]), logs
//!    the restore summary, and stays held (exit 3) or un-holds ([`finish`]:
//!    the counter floor, then the one script that raises the object-id
//!    counter and deletes the hold marker). The accept-loss body
//!    ([`accept_loss`]) writes the plan the same way, then discards what is
//!    proven lost (`plan::lost`, `execute::discard`): lost layers, lost
//!    sessions and their backups, and the large values in no store, in
//!    place; or, when an item could not be confirmed lost and the run has no
//!    second `--confirm discard-unconfirmed-data`, nothing (exit 3). Then it
//!    finishes as restore does. Every write unit calls the permit first.
//!    Every absence goes through [`absence`].
//!
//! Every exit after step 2 releases the lease. The Job serves its metrics on
//! the metrics port for its whole life, but the chart's ServiceMonitor does
//! not scrape the Job pod: its log is the record.
//!
//! # What "writes nothing" leaves out
//!
//! "Writes nothing" and "wrote nothing" in these docs and in the Job's log
//! lines mean that no Valkey key and no backup-store file or object is
//! written, apart from the lease. One exception is on `LocalDisk`. Building
//! the backup store, as the server does, runs `create_dir_all` on
//! `<root>/.live-edit/{layers,sessions,ext}` before step 1. That happens on
//! the not-held and wrong-incident paths too. It creates empty directories
//! only, never a file, and is a no-op on a root a server already uses. It
//! leaves anything behind only on a root no server has used, such as a
//! mistyped path.
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |---|---|
//! | 0 | done (the hold cleared, or found already cleared), or nothing to do |
//! | 3 | stays held: restore: unrestorable, uncovered, unknown; accept-loss: an item it could not confirm lost, without the second confirm; either: a precondition that cannot pass (an incomplete marker, a restore on the `InMemory` adapter, an absent or unreadable sidecar, a failed canary, a contradiction, a session named by the store or a backup with no listed session backup under an empty session listing), a reference record the recount cannot read, on a cluster a keyspace walk whose `CLUSTER NODES` gives no usable myself line (the planner's and the recount's: no walk is used on a node it cannot show serves its primary), a counter floor above the largest allocatable id, or, at the un-hold, an object-id counter that is not a decimal integer or a lease, counter or marker key of the wrong type |
//! | 4 | held under a different incident, at the start or recorded while the Job ran (its un-hold cleared nothing) |
//! | 1 | failed: lease busy or lost (at the un-hold included), SIGTERM, a store error, a store walk that crossed a change of the cluster's slot map or whose node pin did not reach its primary's slots, or that its completeness check did not show complete, the store having changed while the Job read it (`storage::scan_guard`), a backup-store fault (a session backup's PUT or DELETE among them), a plan the writer cannot write (an inconsistent plan, a planned backup now unreadable), accept-loss's discards inconsistent with its plan, a lost layer's object found complete at its re-read, a write after the recount, an OTLP exporter that could not be built for a valid endpoint. Kubernetes retries it. |
//! | 2 | a refused configuration (a pod memory limit the configuration refuses included) or a refused command line ([`EXIT_USAGE`]: accept-loss without `--confirm discard-lost-data`, or with `--confirm discard-unconfirmed-data` alone, among them), before the Job connects to anything |
//!
//! Any other non-zero code is a crash, a failure like exit 1.
//!
//! # Layout
//!
//! - [`args`] — the command line;
//! - [`preconditions`] — the hold-marker precondition;
//! - [`lease`] — the Job lease, its refresh and its release;
//! - [`quiet`] — Q and `T_safe`;
//! - [`permit`] — the write permit;
//! - [`absence`] — the absence rule's primitives;
//! - [`run`] — the frame, [`JobBody`] and [`run_job`];
//! - [`plan`] — the restore plan and its coverage check;
//! - [`execute`] — the restore writer and the Job's restore memory;
//! - [`finish`] — the recount, the counter floor and the un-hold;
//! - [`restore`] — the restore body;
//! - [`accept_loss`] — the accept-loss body.

pub mod absence;
pub mod accept_loss;
pub mod args;
pub mod execute;
pub mod finish;
pub mod lease;
pub mod permit;
pub mod plan;
pub mod preconditions;
pub mod quiet;
pub mod restore;
pub mod run;

pub use accept_loss::{AcceptLossBody, DiscardObserver};
pub use args::{parse_job_args, ArgsError, JobArgs, JobMode, USAGE};
pub use execute::memory::RestoreMemory;
pub use execute::{execute_restore, ExecSeams, ExecutionReport};
#[cfg(any(test, feature = "test-support"))]
pub use execute::{AfterDiscardRegister, AfterDiscardRegisterFn};
pub use plan::RestorePlanner;
pub use restore::{ExecutionObserver, PlanObserver, RestoreBody};
pub use run::{run_job, BackupAccess, JobBody, JobContext, JobRun, JobTimings, NoBackupStore};

/// Exit code of a refused configuration or command line, before the Job
/// connects to anything: in `dr` mode `main.rs` exits with it on every
/// configuration refusal the same environment would repeat on every run,
/// including a pod memory limit the configuration refuses (the server's
/// memory plan's minimum, or the restore's own).
pub const EXIT_USAGE: i32 = 2;

/// How a Job ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobExit {
    /// The work is done: the hold is cleared (exit 0).
    Done,
    /// The store is not held: nothing to do (exit 0).
    NothingToDo,
    /// The service stays held (exit 3).
    StaysHeld,
    /// Held under a different incident (exit 4).
    DifferentIncident,
    /// Failed: lease busy or lost, SIGTERM, a store error, a store walk that
    /// crossed a change of slot map, whose node pin missed its primary's
    /// slots, or that its completeness check did not show complete (exit 1).
    Failed,
}

impl JobExit {
    /// The process exit code.
    pub fn code(self) -> i32 {
        match self {
            Self::Done | Self::NothingToDo => 0,
            Self::StaysHeld => 3,
            Self::DifferentIncident => 4,
            Self::Failed => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_the_contracts() {
        assert_eq!(JobExit::Done.code(), 0);
        assert_eq!(JobExit::NothingToDo.code(), 0);
        assert_eq!(JobExit::StaysHeld.code(), 3);
        assert_eq!(JobExit::DifferentIncident.code(), 4);
        assert_eq!(JobExit::Failed.code(), 1);
        assert_eq!(EXIT_USAGE, 2);
    }
}
